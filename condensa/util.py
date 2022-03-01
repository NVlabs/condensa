# Copyright 2022 NVIDIA Corporation
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

import torch.nn.utils
import torch.utils.data as data
from torch.autograd import Variable

from . import tensor as T

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
