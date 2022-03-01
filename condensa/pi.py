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

import torch
import torch.nn

from   condensa import cfg
from   .mask import add_mask_to_module
from   . import tensor as T

def __precheck(module):
    if not cfg.__CONDENSA_PI_PRECHECK__: return

    for name, _ in module.named_parameters():
        if name != 'weight' and name != 'bias':
            raise NotImplementedError(
                'Unknown parameter {} detected'.format(name))

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

    p = getattr(module, parameter)
    pdata = p.data
    mask = T.simple_mask(pdata, threshold).type(pdata.type())
    T.apply_mask_inplace(pdata, mask)
    p.data = pdata.view_as(p).data
    add_mask_to_module(module, parameter, mask.view_as(p).data)

def blockprune(module,
               threshold,
               block_size,
               criteria,
               padding=False,
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
    :param parameter: Module parameter to prune (default: 'weight')
    :type parameter: str
    """
    __precheck(module)

    if not hasattr(module, parameter):
        raise ValueError('Could not find parameter \'{}\' in module',
                         parameter)

    if hasattr(module, 'condense'): module.condense.add(parameter)
    else: module.condense = set([parameter])

    p = getattr(module, parameter)
    t = p.data
    # Try to flatten Conv2d layers if 2d blocks specified
    if isinstance(module, torch.nn.Conv2d) and len(block_size) == 2:
        t = t.reshape(t.shape[0], -1)

    mask = T.block_mask(tensor=t,
                        threshold=threshold,
                        blocksize=block_size,
                        criteria=criteria,
                        padding=padding)

    if isinstance(module, torch.nn.Conv2d) and len(block_size) == 2:
        mask = mask.reshape(p.data.shape)
    T.apply_mask_inplace(p.data, mask)
    add_mask_to_module(module, parameter, mask)
    return mask

def neuron_prune(module,
                 threshold,
                 criteria,
                 prune_bias=True,
                 parameter='weight'):
    """
    Prunes neurons based on magnitude (inplace).

    :param module: PyTorch module.
    :type module: `torch.nn.Module`
    :param threshold: Magnitude threshold for pruning.
    :type threshold: `float`
    :param criteria: Aggregation function for thresholding.
    :type criteria: `condensa.functional`
    :param prune_bias: Whether to prune corresponding biases.
    :type prune_bias: `bool`
    :param parameter: Module parameter to prune (default: 'weight')
    :type parameter: str
    """
    __precheck(module)

    if not hasattr(module, parameter):
        raise ValueError('Could not find parameter \'{}\' in module',
                         parameter)

    shape = getattr(module, parameter).data.shape
    if len(shape) != 2:
        raise NotImplementedError(
            'Row pruning currently only supported for 2D parameters')

    if hasattr(module, 'condense'): module.condense.add(parameter)
    else: module.condense = set([parameter])

    block_size = (1, shape[1])
    mask = blockprune(module, threshold, block_size, criteria, parameter)
    # Prune corresponding bias tensor
    if module.bias is not None and prune_bias is True:
        assert mask.ndimension() == 2
        T.apply_mask_inplace(module.bias.data, mask[:, 0])
    if cfg.__CONDENSA_SAVE_MASKS__:
        raise NotImplementedError('neuron_prune')

def filter_prune(module,
                 threshold,
                 criteria,
                 prune_bias=True,
                 parameter='weight'):
    """
    Prunes 3D blocks (filters) of module parameters based on magnitude (inplace).

    :param module: PyTorch module.
    :type module: `torch.nn.Module`
    :param threshold: Magnitude threshold for pruning.
    :type threshold: `float`
    :param criteria: Aggregation function for thresholding.
    :type criteria: `condensa.functional`
    :param prune_bias: Whether to prune corresponding biases.
    :type prune_bias: `bool`
    :param parameter: Module parameter to prune (default: 'weight')
    :type parameter: str
    """
    __precheck(module)

    if not hasattr(module, parameter):
        raise ValueError('Could not find parameter \'{}\' in module',
                         parameter)

    p = getattr(module, parameter)
    ndim = p.dim()
    if ndim != 4:
        raise RuntimeError('Filter pruning requires a 4D parameter')

    if hasattr(module, 'condense'):
        module.condense.add(parameter)
    else:
        module.condense = set([parameter])

    block_size = (1, *p.data.shape[1:])
    mask = T.block_mask(p.data, threshold, block_size, criteria)
    T.apply_mask_inplace(p.data, mask)
    # Prune corresponding bias tensor
    if module.bias is not None and prune_bias is True:
        assert mask.ndimension() == 4
        T.apply_mask_inplace(module.bias.data, mask[:, 0, 0, 0])
    if cfg.__CONDENSA_SAVE_MASKS__:
        raise NotImplementedError('filter_prune')
