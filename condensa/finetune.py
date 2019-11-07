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

import sys
import logging
import numpy as np
from copy import deepcopy

import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import condensa.tensor as T
import condensa.util as util

logger = logging.getLogger(__name__)

class FineTuner(object):
    """Condensa model fine-tuner. Can be used for retraining compressed
       models while keeping all zero-valued parameters clipped to zero."""
    def __init__(self, w, layer_types=None, biases=True):
        self.w = w
        self.layer_types = layer_types
        self.biases = biases
        self._compute_mask_inplace()

    def _compute_mask_inplace(self):
        with torch.no_grad():
            for m in self.w.modules():
                if type(m) in self.layer_types\
                   and not hasattr(m, 'condensa_nocompress'):
                    if hasattr(m, 'weight'):
                        m.mask_w = torch.gt(m.weight.data.abs(), 0.)
                    if self.biases:
                        if hasattr(m, 'bias') and m.bias is not None:
                            m.mask_b = torch.gt(m.bias.data.abs(), 0.)

    def _apply_mask(self):
        with torch.no_grad():
            for m in self.w.modules():
                if hasattr(m, 'mask_w'):
                    T.apply_mask_inplace(m.weight.data, m.mask_w)
                if hasattr(m, 'mask_b'):
                    T.apply_mask_inplace(m.bias.data, m.mask_b)

    def run(self,
            epochs,
            lr,
            lr_end,
            momentum,
            weight_decay,
            criterion,
            trainloader,
            testloader,
            valloader,
            debugging_flags={}):
        """
        Fine-tunes a compressed model. Currently only supports SGD.

        :param epochs: Number of epochs
        :type epochs: `int`
        :param lr: Learning rate
        :type lr: `float`
        :param lr_end: End learning rate
        :type lr_end: `float`
        :param momentum: Momentum
        :type momentum: float
        :param weight_decay: Weight decay
        :type weight_decay: float
        :param criterion: Loss criterion
        :param trainloader: Training dataloader
        :param testloader: Test dataloader
        :param valloader: Validation dataloader
        :param debugging_flags: Debugging flags
        :type debugging_flags: dict
        """
        use_cuda = torch.cuda.is_available()

        validate = (valloader is not None)
        test = (testloader is not None)

        if use_cuda:
            cudnn.benchmark = True
            self.w = self.w.cuda()

        _model_stat_fn = debugging_flags['custom_model_statistics']\
                   if 'custom_model_statistics' in debugging_flags\
                   else util.empty_stat_fn

        if validate:
            val_loss, val_stats = _model_stat_fn(self.w, criterion, valloader)
            logging.info(
                '[Condensa:FineTuner] Original model val_loss: {:.2f}, {}'
                .format(val_loss,
                ', '.join(['{}:{}'.format(k, v) for k,v in val_stats.items()])))
        if test:
            test_loss, test_stats = _model_stat_fn(
                self.w, criterion, testloader)
            logging.info(
                '[Condensa:FineTuner] Original model test_loss: {:.2f}, {} '
                .format(test_loss,
                ', '.join(['{}:{}'.format(k, v) for k,v in test_stats.items()])))

        l_alpha = np.exp((np.log(lr_end) - np.log(lr)) / float(epochs))
        optimizer = torch.optim.SGD(self.w.parameters(),
                                    lr=lr,
                                    momentum=momentum,
                                    weight_decay=weight_decay,
                                    nesterov=False)
        with torch.no_grad():
            best_model = deepcopy(self.w)
        best_loss = sys.float_info.max
        for epoch in range(epochs):
            # Switch to training mode
            self.w.train()
            nbatches = len(trainloader)
            if logger.isEnabledFor(logging.INFO):
                pbar = tqdm(total=nbatches, ascii=True)
            for input, target in trainloader:
                if torch.cuda.is_available():
                    if not input.is_cuda: input = input.cuda()
                    if not target.is_cuda: target = target.cuda()
                output = self.w(input)
                loss = criterion(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Apply mask
                self._apply_mask()
                if logger.isEnabledFor(logging.INFO):
                    pbar.update()
            if logger.isEnabledFor(logging.INFO):
                pbar.close()

            # Switch to eval mode
            self.w.eval()

            if validate:
                val_loss, val_stats = _model_stat_fn(
                    self.w, criterion, valloader)
                logging.info(
                    '[Condensa:FineTuner] Epoch [{}], VAL loss: {:.2f}, {}'
                    .format(epoch, val_loss,
                    ', '.join(['{}:{}'.format(k, v) for k,v in val_stats.items()])))
            if test:
                test_loss, test_stats = _model_stat_fn(
                    self.w, criterion, testloader)
                logging.info(
                    '[Condensa:FineTuner] Epoch [{}], TEST loss: {:.2f}, {}'
                    .format(epoch, test_loss,
                    ', '.join(['{}:{}'.format(k, v) for k,v in test_stats.items()])))

            if validate:
                if val_loss < best_loss:
                    logger.info(
                        '[Condensa:FineTuner] SAVING MODEL based on VAL')
                    best_loss = val_loss
                    best_model = deepcopy(self.w)
            elif test:
                if test_loss < best_loss:
                    logger.info(
                        '[Condensa:FineTuner] SAVING MODEL based on TEST')
                    best_loss = test_loss
                    best_model = deepcopy(self.w)
            else:
                logger.info(
                    '[Condensa:FineTuner] SAVING MODEL based on most recent')
                best_model = deepcopy(self.w)

            lr *= l_alpha
            for g in optimizer.param_groups:
                g['lr'] = lr

        return best_model
