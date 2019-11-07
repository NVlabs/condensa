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

import torch

def l2norm(tensor, dim, keepdim):
    """
    Computes the l2-norm of elements in input tensor.

    :param tensor: PyTorch tensor.
    :type tensor: `torch.nn.Module`
    :param dim: Reduction dimension.
    :type dim: `int`
    :param keepdim: Whether the output has `dim` retained.
    :type keepdim: `bool`
    :return: l2-norm of input tensor.
    """
    return torch.norm(tensor, 2, dim, keepdim)

def max(tensor, dim, keepdim):
    """
    Computes the maximum value of elements in input tensor.

    :param tensor: PyTorch tensor.
    :type tensor: `torch.nn.Module`
    :param dim: Reduction dimension.
    :type dim: `int`
    :param keepdim: Whether the output has `dim` retained.
    :type keepdim: `bool`
    :return: Max of input tensor.
    """
    return torch.max(tensor, dim, keepdim)[0]

def min(tensor, dim, keepdim):
    """
    Computes the minimum value of elements in input tensor.

    :param tensor: PyTorch tensor.
    :type tensor: `torch.nn.Module`
    :param dim: Reduction dimension.
    :type dim: `int`
    :param keepdim: Whether the output has `dim` retained.
    :type keepdim: `bool`
    :return: Min of input tensor.
    """
    return torch.min(tensor, dim, keepdim)[0]

def mean(tensor, dim, keepdim):
    """
    Computes the mean value of elements in input tensor.

    :param tensor: PyTorch tensor.
    :type tensor: `torch.nn.Module`
    :param dim: Reduction dimension.
    :type dim: `int`
    :param keepdim: Whether the output has `dim` retained.
    :type keepdim: `bool`
    :return: Mean value of input tensor.
    """
    return torch.mean(tensor, dim, keepdim)

def sum(tensor, dim, keepdim):
    """
    Computes the sum of elements in input tensor.

    :param tensor: PyTorch tensor.
    :type tensor: `torch.nn.Module`
    :param dim: Reduction dimension.
    :type dim: `int`
    :param keepdim: Whether the output has `dim` retained.
    :type keepdim: `bool`
    :return: Sum of input tensor.
    """
    return torch.sum(tensor, dim, keepdim)
