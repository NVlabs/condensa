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

import numpy as np
import torch

def density(tensor):
    """
    Computes the ratio of nonzeros to total elements in a tensor.

    :param tensor: PyTorch tensor
    :type tensor: `torch.Tensor`
    :return: Ratio of nonzeros to total elements
    :rtype: `float`
    """
    t = tensor.view(-1)
    return float(t.nonzero().numel()) / float(t.numel())

def sparsity(tensor):
    """
    Computes the ratio of zeros to total elements in a tensor.

    :param tensor: PyTorch tensor
    :type tensor: torch.Tensor
    :return: Ratio of zeros to total elements
    :rtype: `float`
    """
    return 1. - density(tensor)

def threshold(tensor, density):
    """
    Computes a magnitude-based threshold for given tensor.

    :param tensor: PyTorch tensor
    :type tensor: `torch.Tensor`
    :param density: Desired ratio of nonzeros to total elements
    :type density: `float`
    :return: Magnitude threshold
    :rtype: `float`
    """
    tf = tensor.abs().view(-1)
    numel = int(density * tf.numel())
    if numel == 0:
        raise RuntimeError('Provided density value causes model to be zero.')

    topk, _ = torch.topk(tf.abs(), numel, sorted=True)
    return topk.data[-1]

def aggregate(tensor, blocksize, criteria):
    """
    Aggregates tensor dimensions according to criteria.

    :param tensor: PyTorch tensor
    :type tensor: `torch.Tensor`
    :param blocksize: Size of blocks to aggregate
    :type blocksize: `Tuple(int)`
    :param criteria: Aggregation criteria
    :type criteria: `condensa.functional`
    :return: Aggregated tensor
    :rtype: `torch.Tensor`
    """
    if tensor.dim() != len(blocksize):
        raise RuntimeError('Tensor and block dimensions do not match')
    ndim = tensor.dim()

    blocksize_flat = np.prod(np.array(blocksize))
    shape = np.array(tensor.shape)
    repeats = (shape / blocksize).astype(int)
    divcheck = (shape % blocksize).astype(int)

    if not np.all(divcheck == 0):
        raise TypeError('Block size must be divisible by tensor size')

    tmpshape = np.column_stack([repeats, blocksize]).ravel()
    order = np.arange(len(tmpshape))
    order = np.concatenate([order[::2], order[1::2]])
    blocks = tensor.abs().reshape(tuple(tmpshape))
    blocks = blocks.permute(tuple(order)).reshape(-1, *blocksize)
    agg = criteria(blocks.reshape(-1, blocksize_flat), dim=1, keepdim=True)
    return agg

def aggregate_neurons(tensor, criteria):
    """
    Aggregates neurons (rows) in given weight matrix.
  
    :param tensor: PyTorch tensor
    :type tensor: `torch.Tensor`
    :param criteria: Aggregation criteria
    :type criteria: `condensa.functional`
    :return: Neuron-aggregated tensor
    :rtype: `torch.Tensor`
    """
    return aggregate(tensor, (1, tensor.shape[1]), criteria)

def aggregate_filters(tensor, criteria):
    """
    Aggregates 3D filters in given weight tensor.
  
    :param tensor: PyTorch tensor
    :type tensor: `torch.Tensor`
    :param criteria: Aggregation criteria
    :type criteria: `condensa.functional`
    :return: Filter-aggregated tensor
    :rtype: `torch.Tensor`
    """
    return aggregate(tensor, (1, *tensor.shape[1:]), criteria)

def simple_mask(tensor, threshold, align=None):
    """
    Computes a simple binary mask for given magnitude threshold.

    :param tensor: PyTorch tensor
    :type tensor: `torch.Tensor`
    :param threshold: magnitude threshold for pruning
    :type threshold: `float`
    :return: Mask
    :rtype: `torch.Tensor`
    """
    assert tensor.dim() == 1
    if align is None:
        return torch.ge(tensor.abs(), threshold)
    else:
        size = tensor.size(0)
        if size < align:
            raise RuntimeError('Tensor too small for given alignment')
        t = tensor.abs()
        nnz = torch.ge(t, threshold).nonzero().size(0)
        nnz = int(nnz / align) * align
        _, indices = torch.topk(t, nnz)
        ones = torch.ones(nnz,
                          dtype=tensor.dtype,
                          layout=tensor.layout,
                          device=tensor.device)
        mask = torch.zeros_like(tensor).scatter_(0, indices, ones)
        return mask

def block_mask(tensor, threshold, blocksize, criteria, align=None):
    """
    Computes an n-D binary mask for given magnitude threshold.

    :param tensor: PyTorch tensor
    :type tensor: `torch.Tensor`
    :param threshold: magnitude threshold for pruning
    :type threshold: `float`
    :param blocksize: desired block size (Tuple)
    :type blocksize: `Tuple`
    :param criteria: aggregation function for thresholding (default: max)
    :type criteria: `condensa.functional`
    :return: Mask
    :rtype: `torch.Tensor`
    """
    # Original implementation at: https://stackoverflow.com/questions/42297115
    # /numpy-split-cube-into-cubes/42298440#42298440
    if tensor.dim() != len(blocksize):
        raise RuntimeError('Tensor and block dimensions do not match')
    ndim = tensor.dim()

    blocksize_flat = np.prod(np.array(blocksize))
    shape = np.array(tensor.shape)
    repeats = (shape / blocksize).astype(int)
    divcheck = (shape % blocksize).astype(int)

    if not np.all(divcheck == 0):
        raise TypeError('Block size must be divisible by tensor size')

    tmpshape = np.column_stack([repeats, blocksize]).ravel()
    order = np.arange(len(tmpshape))
    order = np.concatenate([order[::2], order[1::2]])
    blocks = tensor.abs().reshape(tuple(tmpshape))
    blocks = blocks.permute(tuple(order)).reshape(-1, *blocksize)
    agg = criteria(blocks.reshape(-1, blocksize_flat), dim=1, keepdim=True)

    mask = simple_mask(agg.view(-1), threshold, align)
    mask = mask.view(agg.shape).expand(-1,
                                       blocksize_flat).reshape(blocks.shape)

    N, newshape = mask.shape[0], mask.shape[1:]
    repeats = (shape / newshape).astype(int)
    tmpshape = np.concatenate([repeats, newshape])
    order = np.arange(len(tmpshape)).reshape(2, -1).ravel(order='F')
    return mask.reshape(tuple(tmpshape)).permute(tuple(order)).reshape(
        tuple(shape))

def apply_mask(tensor, mask):
    """
    Computes masked version of tensor.

    :param tensor: PyTorch tensor
    :type tensor: `torch.Tensor`
    :param mask: Binary mask
    :type mask: `torch.Tensor`
    :return: Masked version of `tensor`
    :rtype: `torch.Tensor`
    """
    #assert isinstance(tensor, torch.Tensor)
    return torch.mul(tensor, mask.type(tensor.type()))

def apply_mask_inplace(tensor, mask):
    """
    Applies binary mask in-place.

    :param tensor: PyTorch tensor
    :type tensor: `torch.Tensor`
    :param mask: Binary mask
    :type mask: `torch.Tensor`
    """
    #assert isinstance(tensor, torch.Tensor)
    tensor.mul_(mask.type(tensor.type()))
