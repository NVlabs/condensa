# Copyright 2019 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import sys
import numpy as np
import logging
from tqdm import tqdm

import torch.nn.utils
import torch.utils.data as data
from torch.autograd import Variable

import condensa.tensor as T

logger = logging.getLogger(__name__)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

def is_leaf_node(module):
    """
    Checks if given module is a leaf module.

    :param module: PyTorch module
    :type module: `torch.nn.Module`
    :return: Boolean value representing whether module is a leaf.
    :rtype: `bool`
    """
    return list(module.children()) == []

def magnitude_threshold(module, density):
    """
    Computes a magnitude-based threshold for given module.

    :param module: PyTorch module
    :type module: `torch.nn.Module`
    :param density: Desired ratio of nonzeros to total elements
    :type density: `float`
    :return: Magnitude threshold
    :rtype: `float`
    """
    params = torch.nn.utils.parameters_to_vector(module.parameters())
    return T.threshold(params, density)

def empty_stat_fn(model, criterion, dataloader):
    """
    Empty model statistics function: returns loss.

    :param model: PyTorch model
    :type model: `torch.nn.Module`
    :param loss_fn: Loss function
    :param dataloader: Data loader to use
    :return: Tuple of loss, dictionary of statistics
    :rtype: `Tuple(float, dict)`
    """
    return (loss(model, criterion, dataloader), {})

def accuracy(output, target, topk=(1, )):
    """
    Computes the precision@k for the specified values of k

    :param output: Predicted output batch
    :type output: `torch.Tensor`
    :param target: Actual output batch
    :type target: `torch.Tensor`
    :param topk: Top-k value
    :type topk: `Tuple`
    :return: Model accuracy
    :rtype: `float`
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def loss(model, criterion, dataloader):
    """
    Computes loss on given dataset.
  
    :param model: PyTorch model
    :type model: `torch.nn.Module`
    :param loss_fn: Loss function
    :param dataloader: Data loader to use
    :return: Loss
    :rtype: `float`
    """
    losses = AverageMeter()
    model.eval()
    pzero = list(model.parameters())[0]
    if (pzero.dtype != torch.float32 and pzero.dtype != torch.float16):
        raise NotImplementedError('Only FP16 and FP32 weights are supported')
    cast2fp16 = (isinstance(pzero, torch.HalfTensor)
                 or isinstance(pzero, torch.cuda.HalfTensor))
    loss = 0.
    with torch.no_grad():
        for input, target in dataloader:
            if torch.cuda.is_available():
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
            if cast2fp16:
                input = input.half()
            output = model(input)
            loss = criterion(output, target)
            losses.update(to_python_float(loss.data), input.size(0))
    return losses.avg

def cnn_statistics(model, criterion, dataloader):
    """
    Computes accuracy of given CNN model.
  
    :param model: PyTorch model
    :type model: `torch.nn.Module`
    :param criterion: Loss function
    :param dataloader: Data loader to use
    :return: Top-1 and Top-5 accuracies
    :rtype: Tuple(top1, top5)
    """
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    pzero = list(model.parameters())[0]
    if (pzero.dtype != torch.float32 and pzero.dtype != torch.float16):
        raise NotImplementedError('Only FP16 and FP32 weights are supported')
    cast2fp16 = (isinstance(pzero, torch.HalfTensor)
                 or isinstance(pzero, torch.cuda.HalfTensor))
    loss = 0.
    correct = 0.
    with torch.no_grad():
        for input, target in dataloader:
            if torch.cuda.is_available():
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
            if cast2fp16:
                input = input.half()
            output = model(input)
            loss = criterion(output, target)
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(to_python_float(loss.data), input.size(0))
            top1.update(to_python_float(prec1), input.size(0))
            top5.update(to_python_float(prec5), input.size(0))
    return (losses.avg,
            {'top1': top1.avg, 'top5': top5.avg})

def compressed_model_stats(w, wc):
    """
    Retrieve various statistics for compressed model.

    :param w: Original model
    :type w: `torch.nn.Module`
    :param wc: Compressed model
    :type wc: `torch.nn.Module`
    :return: Dictionary of compressed model statistics
    :rtype: `dict`
    """
    stats = dict()
    nparams_w = dict()
    nparams_wc = dict()

    nparams_w['total_nnz'] = torch.nn.utils.parameters_to_vector(
        w.parameters()).view(-1).nonzero().numel()
    nparams_wc['total_nnz'] = torch.nn.utils.parameters_to_vector(
        wc.parameters()).view(-1).nonzero().numel()

    for (name_w, m_w), (name_wc, m_wc) in zip(w.named_modules(),
                                              wc.named_modules()):
        if type(m_w) == torch.nn.Linear or type(m_w) == torch.nn.Conv2d:
            nparams_w[name_w] = torch.nn.utils.parameters_to_vector(
                m_w.parameters()).view(-1).nonzero().numel()
            nparams_wc[name_wc] = torch.nn.utils.parameters_to_vector(
                m_wc.parameters()).view(-1).nonzero().numel()

    stats['num_params'] = nparams_w
    stats['num_params_compressed'] = nparams_wc
    return stats

def pretrain(epochs, model, trainloader, criterion, optimizer):
    """
    No-frills pre-training method.
  
    :param epochs: Number of epochs
    :type epochs: `int`
    :param model: PyTorch model
    :type model: `torch.nn.Module`
    :param trainloader: Training dataloader
    :param criterion: Loss criterion
    :param optimizer: Optimizer to use
    """
    _config = {'epochs': epochs}
    logging.info('[Condensa] PRETRAIN CONFIG [' +
                 ', '.join('{!s}={!r}'.format(k, v)
                           for k, v in _config.items()) + ']')

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()
        model = torch.nn.DataParallel(model)
    mb_iterator = iter(trainloader)
    model.train()
    for j in range(0, epochs):
        if logger.isEnabledFor(logging.INFO):
            pbar = tqdm(total=len(trainloader),
                        ascii=True,
                        desc='Epoch {}'.format(j))
        for input, target in trainloader:
            if torch.cuda.is_available():
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
            input, target = Variable(input), Variable(target)
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if logger.isEnabledFor(logging.INFO):
                pbar.update()
    if logger.isEnabledFor(logging.INFO):
        pbar.close()
    logging.info('')

class EventTimer(object):
    """Simple timer class."""
    def __init__(self):
        """Constructor. Begins timing."""
        self.begin = time.perf_counter()

    def reset(self):
        """Reset timer."""
        self.begin = time.perf_counter()

    @property
    def elapsed_seconds(self):
        """Returns elapsed seconds."""
        return (time.perf_counter() - self.begin)
