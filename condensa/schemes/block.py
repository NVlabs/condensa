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

class LayerBlockPruner(LayerScheme):
    """Prunes blocks of nonzeros."""
    def __init__(self, block_size, criteria=F.l2norm, padding=True):
        """
        Creates an instance of `LayerBlockPruner`.

        :param block_size: Target block size.
        :type block_size: `Tuple`
        :param criteria: Structure aggregation criteria (default: l2norm).
        :type criteria: `condensa.functional`
        :param padding: Whether to pad with zeros if block shape doesn't divide tensor shape evenly.
        :type padding: `Boolean`
        """
        super().__init__()

        self.block_size = block_size
        self.criteria = criteria
        self.padding = padding
    
    def aggregate(self, module, parameter='weight'):
        """
        Aggregate nonzero blocks.
    
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

        # Try to flatten Conv2d layers if 2d blocks specified
        if (isinstance(module, torch.nn.Conv2d)
            and len(self.block_size) == 2):
            w = w.reshape(w.shape[0], -1)
        
        agg = T.aggregate(w.data, self.block_size, self.criteria, self.padding)
        return agg.view(-1)

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
        condensa.blockprune(module,
                            threshold,
                            block_size=self.block_size,
                            criteria=self.criteria,
                            padding=True,
                            parameter=parameter)

    def __repr__(self):
        return f'<LayerBlockPruner :: block-size:{self.block_size}, '\
               f'criteria:{self.criteria}, padding:{self.padding}>'