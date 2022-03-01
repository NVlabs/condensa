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

import condensa
import condensa.tensor as T
import condensa.functional as F
from condensa import cfg
from .types import LayerScheme

class LayerPruner(LayerScheme):
    """Prunes individual nonzeros."""
    def __init__(self):
        """
        Creates an instance of `LayerPruner`.
        """
        super().__init__()
    
    def aggregate(self, module, parameter='weight'):
        """
        Aggregate nonzeros.
    
        :param module: PyTorch module.
        :type module: `torch.nn.Module`
        :param parameter: Parameter within module to prune.
        :type parameter: `str`
        """
        if hasattr(module, 'condensa_nocompress'):
            raise RuntimeError(
                'aggregate() called on module with condensa_nocompress set.')
        if not hasattr(module, parameter):
            raise ValueError(
                f'Could not find parameter {parameter} in module')
        w = getattr(module, parameter)
        agg = w.data.reshape(-1)
        return agg

    def pi(self, module, threshold, parameter='weight'):
        """
        Apply pruning to module.
    
        :param module: PyTorch module.
        :type module: `torch.nn.Module`
        :param threshold: Magnitude threshold for pruning.
        :type threshold: `float`
        :param parameter: Parameter within module to prune.
        :type parameter: `str`
        """
        if hasattr(module, 'condensa_nocompress'):
            raise RuntimeError(
                'pi() called on module with condensa_nocompress set.')
        condensa.prune(module,
                       threshold,
                       parameter=parameter)

    def __repr__(self):
        return '<LayerPruner>'