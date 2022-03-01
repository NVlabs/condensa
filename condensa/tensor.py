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

import numpy as np
import torch
import torch.nn.functional as F

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

def threshold(tensor, sparsity):
    """
    Computes a magnitude-based threshold for given tensor.

    :param tensor: PyTorch tensor
    :type tensor: `torch.Tensor`
    :param sparsity: Desired ratio of zeros to total elements
    :type sparsity: `float`
    :return: Magnitude threshold
    :rtype: `float`
    """
    density = 1. - sparsity
    numel = int(density * tensor.numel())
    if numel == 0:
        raise RuntimeError('Provided sparsity value causes model to be zero.')

    topk, _ = torch.topk(tensor.abs().view(-1), numel, sorted=True)
    return topk.data[-1]

def aggregate(tensor, blocksize, criteria, padding=False):
    """
    Aggregates tensor dimensions according to criteria.

    :param tensor: PyTorch tensor
    :type tensor: `torch.Tensor`
    :param blocksize: Size of blocks to aggregate
    :type blocksize: `Tuple(int)`
    :param criteria: Aggregation criteria
    :type criteria: `condensa.functional`
    :type padding: Whether to pad with zeros
    :param padding: Boolean
    :return: Aggregated tensor
    :rtype: `torch.Tensor`
    """
    if tensor.dim() != len(blocksize):
        raise RuntimeError('Tensor and block dimensions do not match')
    ndim = tensor.dim()

    shape = np.array(tensor.shape)
    divcheck = (shape % blocksize).astype(int)

    if not np.all(divcheck == 0):
        if padding:
            pad = (((((shape - 1) / blocksize).astype(int) + 1) * blocksize)
                  - shape)
            pad_exp = np.zeros(pad.shape[0] * 2).astype(int)
            pad_exp[1::2] = np.flip(pad, 0)
            # Pad end of each axis with zeros
            tensor = F.pad(input=tensor,
                           pad=tuple(pad_exp),
                           mode='constant',
                           value=0)
            shape = np.array(tensor.shape)
        else:
            raise TypeError('Block size not divisible by tensor size. '
                            'Consider setting padding to True.')

    blocksize_flat = np.prod(np.array(blocksize))
    repeats = (shape / blocksize).astype(int)
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

def simple_mask(tensor, threshold):
    """
    Computes a simple binary mask for given magnitude threshold.

    :param tensor: PyTorch tensor
    :type tensor: `torch.Tensor`
    :param threshold: magnitude threshold for pruning
    :type threshold: `float`
    :return: Mask
    :rtype: `torch.Tensor`
    """
    return torch.ge(tensor.abs(), threshold)

def get_mask(tensor):
    """
    Returns the non-zero mask for tensor.

    :param tensor: PyTorch tensor
    :type tensor: `torch.Tensor`
    :return: Mask
    :rtype: `torch.Tensor`
    """
    return torch.gt(tensor.abs(), 0.)

def simple_aligned_mask(tensor, threshold, align):
    """
    Computes a simple aligned binary mask for given magnitude threshold.

    :param tensor: PyTorch tensor
    :type tensor: `torch.Tensor`
    :param threshold: magnitude threshold for pruning
    :type threshold: `float`
    :param align: alignment
    :type align: `float`
    :return: Mask
    :rtype: `torch.Tensor`
    """
    raise NotImplementedError('simple_aligned_mask')

def block_mask(tensor,
               threshold,
               blocksize,
               criteria,
               padding=False):
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
    :type padding: Whether to pad with zeros
    :param padding: Boolean
    :return: Mask
    :rtype: `torch.Tensor`
    """
    # Original implementation at: https://stackoverflow.com/questions/42297115
    # /numpy-split-cube-into-cubes/42298440#42298440
    if tensor.dim() != len(blocksize):
        raise RuntimeError('Tensor and block dimensions do not match')
    ndim = tensor.dim()

    shape = np.array(tensor.shape)
    divcheck = (shape % blocksize).astype(int)

    pad = None
    if not np.all(divcheck == 0):
        if padding:
            pad = (((((shape - 1) / blocksize).astype(int) + 1) * blocksize)
                  - shape)
            pad_exp = np.zeros(pad.shape[0] * 2).astype(int)
            pad_exp[1::2] = np.flip(pad, 0)
            # Pad end of each axis with zeros
            tensor = F.pad(input=tensor,
                           pad=tuple(pad_exp),
                           mode='constant',
                           value=0)
            shape = np.array(tensor.shape)
        else:
            raise TypeError('Block size not divisible by tensor size. '
                            'Consider setting padding to True.')

    blocksize_flat = np.prod(np.array(blocksize))
    repeats = (shape / blocksize).astype(int)
    tmpshape = np.column_stack([repeats, blocksize]).ravel()
    order = np.arange(len(tmpshape))
    order = np.concatenate([order[::2], order[1::2]])
    blocks = tensor.abs().reshape(tuple(tmpshape))
    blocks = blocks.permute(tuple(order)).reshape(-1, *blocksize)
    agg = criteria(blocks.reshape(-1, blocksize_flat), dim=1, keepdim=True)

    mask = simple_mask(agg.view(-1), threshold)
    mask = mask.view(agg.shape).expand(-1,
                                       blocksize_flat).reshape(blocks.shape)

    N, newshape = mask.shape[0], mask.shape[1:]
    repeats = (shape / newshape).astype(int)
    tmpshape = np.concatenate([repeats, newshape])
    order = np.arange(len(tmpshape)).reshape(2, -1).ravel(order='F')
    mask = mask.reshape(tuple(tmpshape)).permute(tuple(order)).reshape(
                        tuple(shape))
    if pad is not None:
        # Snip padded dimensions back to original size
        viewdim = []
        for i in range(len(pad)):
            col = (-pad[i] if pad[i] != 0 else None)
            viewdim.append(slice(0, col))
        return mask[viewdim]
    return mask

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
