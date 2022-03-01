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

import inspect
import logging
import torch

import condensa
import condensa.tensor as T
import condensa.functional as F
from condensa import cfg
from .types import NetworkScheme, LayerScheme

logger = logging.getLogger(__name__)

class NetworkPruner(NetworkScheme):
    """Builds a network-wide pruning scheme from layer-wise ones."""
    def __init__(self, sparsity, mapping):
        """
        Creates a `NetworkPruner` instance. The `mapping` specifies which
        scheme to apply to each layer.

        :param sparsity: Target sparsity.
        :type sparsity: `float`
        :param mapping: Layer name -> layer-wise scheme mapping.
        :type mapping: `dict`
        """
        super().__init__()

        self.sparsity = sparsity
        self.mapping = mapping

        for k, v in mapping.items():
            if isinstance(v, list):
                for s in v:
                    if not isinstance(s, LayerScheme):
                        raise TypeError(f'Scheme {s} corresponding to key {k} '
                                        f'must be a subclass of LayerScheme')
                    if not s.fixed_sparsity:
                        raise RuntimeError('Only fixed-sparsity schemes can '
                                           'be stacked together.')
            else:
                if not isinstance(v, LayerScheme):
                    raise TypeError(f'Scheme {v} corresponding to key {k} '
                                    f'must be a subclass of LayerScheme')


    def threshold(self, module):
        """
        Computes magnitude threshold.

        :param module: PyTorch module.
        :type module: `torch.nn.Module`
        """
        vec = []
        for name, m in module.named_modules():
            if hasattr(m, 'condensa_nocompress'):
                continue
            if name in self.mapping:
                # Skip over stacked schemes (fixed sparsity)
                if isinstance(self.mapping[name], list):
                    continue
                if not self.mapping[name].fixed_sparsity:
                    if not hasattr(self.mapping[name], 'aggregate'):
                        raise RuntimeError(f'Aggregation function not found for'
                                           f' scheme {self.mapping[name]}')
                    vec.append(self.mapping[name].aggregate(m))

        return T.threshold(torch.cat(vec), self.sparsity) if vec else 0.

    def pi(self, module):
        """
        Applies compression scheme to module.
    
        :param module: PyTorch module.
        :type module: `torch.nn.Module`
        """
        if hasattr(module, 'module'):
            module = module.module

        if self.sparsity:
            threshold = self.threshold(module)

        for name, m in module.named_modules():
            if hasattr(m, 'condensa_nocompress'):
                continue
            if name in self.mapping:
                if isinstance(self.mapping[name], list):
                    for s in self.mapping[name]:
                        if not hasattr(s, 'pi'):
                            raise RuntimeError(f'Could not find attribute `pi` '
                                               f'for scheme {s}')
                        assert s.fixed_sparsity
                        # No threshold passed here (fixed sparsity assumed)
                        s.pi(m)
                else:
                    if not hasattr(self.mapping[name], 'pi'):
                        raise RuntimeError(f'Could not find attribute `pi` '
                                           f'for scheme {self.mapping[name]}')
                    if self.mapping[name].fixed_sparsity:
                        self.mapping[name].pi(m)
                    else:
                        if self.sparsity is None:
                            raise RuntimeError(f'Global sparsity must be '
                                               f'specified for scheme '
                                               f'{self.mapping[name]}')
                        self.mapping[name].pi(m, threshold)

    def __repr__(self):
        return f'<NetworkPruner :: sparsity: {self.sparsity}, '\
               f'mapping: {self.mapping}>'

class SchemeComposer(NetworkScheme):
    """Composes two or more schemes together."""
    def __init__(self, schemes):
        """
        Creates a `SchemeComposer` instance.

        :param schemes: List of schemes to compose.
        :type schemes: `list`
        """
        super().__init__()

        if not isinstance(schemes, list):
            raise TypeError('Please specify schemes to compose as a list')
        self.schemes = schemes

    def pi(self, module):
        """
        Applies compression scheme to module.
    
        :param module: PyTorch module.
        :type module: `torch.nn.Module`
        """
        for s in self.schemes:
            if not isinstance(s, NetworkScheme):
                raise TypeError('All schemes passed to SchemeComposer must'\
                                'be instances of NetworkScheme.')
            s.pi(module)

    def __repr__(self):
        return f'<SchemeComposer :: schemes: {self.schemes}>'
