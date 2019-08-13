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
import torch.nn

from condensa import dtypes
from condensa import cfg
import condensa.tensor as T

def __precheck(module):
    if not cfg.__CONDENSA_PI_PRECHECK__: return

    if len(list(module.children())) > 0:
        raise RuntimeError('Only leaf modules may be compressed')
    for name, _ in module.named_parameters():
        if name != 'weight' and name != 'bias':
            raise NotImplementedError(
                'Unknown parameter {} detected'.format(name))

def quantize(module, dtype):
    """
    Quantizes module to given data type (inplace).

    :param module: PyTorch module.
    :type module: `torch.nn.Module`
    :param dtype: Target data type.
    """
    __precheck(module)

    parameters = ['weight']
    #parameters = [name for name, _ in module.named_parameters()]
    if hasattr(module, 'condense'): module.condense |= set(parameters)
    else: module.condense = set(parameters)

    if not cfg.__CONDENSA_RECORD_MODE__:
        if dtype.as_dtype_enum == dtypes.DT_FLOAT16:
            module.half()
        elif dtype.as_dtype_enum == dtypes.DT_FLOAT32:
            module.float()
        else:
            raise TypeError('Unknown data type specified for quantization')

def prune(module, threshold, parameter='weight'):
    """
    Prunes module parameters based on magnitude (inplace).

    :param module: PyTorch module.
    :type module: `torch.nn.Module`
    :param threshold: Magnitude threshold for pruning.
    :type threshold: `float`
    :param parameter: Module parameter to prune (default: 'weight')
    :type parameter: str
    """
    __precheck(module)

    if not hasattr(module, parameter):
        raise ValueError('Could not find parameter \'{}\' in module',
                         parameter)

    if hasattr(module, 'condense'): module.condense.add(parameter)
    else: module.condense = set([parameter])

    if not cfg.__CONDENSA_RECORD_MODE__:
        p = getattr(module, parameter)
        pdata = p.data.view(-1)
        mask = T.simple_mask(pdata, threshold).type(pdata.type())
        T.apply_mask_inplace(pdata, mask)
        p.data = pdata.view_as(p).data
        #if cfg.__CONDENSA_SAVE_MASK__: module.mask = mask.view_as(p).data

def blockprune(module,
               threshold,
               block_size,
               criteria,
               align=None,
               parameter='weight'):
    """
    Prunes blocks of module parameters based on magnitude (inplace).

    :param module: PyTorch module.
    :type module: `torch.nn.Module`
    :param threshold: Magnitude threshold for pruning.
    :type threshold: `float`
    :param block_size: Block size for pruning.
    :type block_size: `Tuple`
    :param criteria: Aggregation function for thresholding.
    :type criteria: `condensa.functional`
    :param align: Alignment of compressed parameters.
    :type align: `int`
    :param parameter: Module parameter to prune (default: 'weight')
    :type parameter: str
    """
    __precheck(module)

    if not hasattr(module, parameter):
        raise ValueError('Could not find parameter \'{}\' in module',
                         parameter)

    p = getattr(module, parameter)
    ndim = p.dim()
    bdim = len(block_size)
    if ndim != bdim:
        raise RuntimeError(
            'Block must have same dimensions as parameter \'{}\''.format(
                parameter))

    if hasattr(module, 'condense'): module.condense.add(parameter)
    else: module.condense = set([parameter])

    if not cfg.__CONDENSA_RECORD_MODE__:
        mask = T.block_mask(p.data, threshold, block_size, criteria, align)
        T.apply_mask_inplace(p.data, mask)
        return mask
    return None

def neuron_prune(module, threshold, criteria, align=None, prune_bias=True):
    """
    Prunes neurons based on magnitude (inplace).

    :param module: PyTorch module.
    :type module: `torch.nn.Module`
    :param threshold: Magnitude threshold for pruning.
    :type threshold: `float`
    :param criteria: Aggregation function for thresholding.
    :type criteria: `condensa.functional`
    :param align: Alignment of compressed parameters.
    :type align: `int`
    :param prune_bias: Whether to prune corresponding biases.
    :type prune_bias: `bool`
    """
    __precheck(module)

    parameter = 'weight'
    if not hasattr(module, parameter):
        raise ValueError('Could not find parameter \'{}\' in module',
                         parameter)

    shape = getattr(module, parameter).data.shape
    if len(shape) != 2:
        raise NotImplementedError(
            'Row pruning currently only supported for 2D parameters')

    if hasattr(module, 'condense'): module.condense.add(parameter)
    else: module.condense = set([parameter])

    if not cfg.__CONDENSA_RECORD_MODE__:
        block_size = (1, shape[1])
        mask = blockprune(module, threshold, block_size, criteria, align,
                          parameter)
        # Prune corresponding bias tensor
        if module.bias is not None and prune_bias is True:
            assert mask.ndimension() == 2
            T.apply_mask_inplace(module.bias.data, mask[:, 0])

def filter_prune(module, threshold, criteria, align=None, prune_bias=True):
    """
    Prunes 3D blocks (filters) of module parameters based on magnitude (inplace).

    :param module: PyTorch module.
    :type module: `torch.nn.Module`
    :param threshold: Magnitude threshold for pruning.
    :type threshold: `float`
    :param criteria: Aggregation function for thresholding.
    :type criteria: `condensa.functional`
    :param align: Alignment of compressed parameters.
    :type align: `int`
    :param prune_bias: Whether to prune corresponding biases.
    :type prune_bias: `bool`
    """
    __precheck(module)

    parameter = 'weight'
    if not hasattr(module, parameter):
        raise ValueError('Could not find parameter \'{}\' in module',
                         parameter)

    p = getattr(module, parameter)
    ndim = p.dim()
    if ndim != 4:
        raise RuntimeError('Filter pruning requires a 4D parameter')

    if hasattr(module, 'condense'): module.condense.add(parameter)
    else: module.condense = set([parameter])

    if not cfg.__CONDENSA_RECORD_MODE__:
        block_size = (1, *p.data.shape[1:])
        mask = T.block_mask(p.data, threshold, block_size, criteria, align)
        T.apply_mask_inplace(p.data, mask)
        # Prune corresponding bias tensor
        if module.bias is not None and prune_bias is True:
            assert mask.ndimension() == 4
            T.apply_mask_inplace(module.bias.data, mask[:, 0, 0, 0])
