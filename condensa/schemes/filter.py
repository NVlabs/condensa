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

class LayerFilterPruner(LayerScheme):
    """Prunes convolutional filters."""
    def __init__(self, criteria=F.l2norm, prune_bias=True):
        """
        Creates an instance of `LayerFilterPruner`.

        :param criteria: Neuron aggregation criteria (default: l2norm).
        :type criteria: `condensa.functional`
        :param prune_bias: Whether to prune corresponding biases (default: True)
        :type prune_bias: `bool`
        """
        super().__init__()

        self.criteria = criteria
        self.prune_bias = prune_bias
    
    def aggregate(self, module, parameter='weight'):
        """
        Aggregates filters.
    
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
        agg = T.aggregate_filters(w.data, self.criteria)
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
        condensa.filter_prune(module,
                              threshold,
                              criteria=self.criteria,
                              prune_bias=self.prune_bias,
                              parameter=parameter)

    def __repr__(self):
        return f'<LayerFilterPruner :: criteria: {self.criteria}, '\
               f'prune_bias: {self.prune_bias}>'
