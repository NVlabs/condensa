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

import condensa
import condensa.tensor as T
import condensa.functional as F

class Compose(object):
    """Composes two or more schemes together."""
    def __init__(self, schemes):
        """
        Creates a `Compose` instance.

        :param schemes: List of schemes to compose.
        :type schemes: `list`
        """
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
            s.pi(module)

    def delta(self, module):
        """
        Applies de-compression scheme to module.

        :param module: PyTorch module.
        :type module: `torch.nn.Module`
        """
        for s in reversed(self.schemes):
            s.delta(module)

    def __repr__(self):
        return '<Compose: {}>'.format(self.schemes)

class Prune(object):
    """Prunes network to given density."""
    def __init__(self, density):
        """
        Creates a `Prune` instance.

        :param density: Target density.
        :type density: `float`
        """
        self.density = density
        self.layer_types = [torch.nn.Linear, torch.nn.Conv2d]

    def threshold(self, module):
        """
        Computes magnitude threshold.

        :param module: PyTorch module.
        :type module: `torch.nn.Module`
        """
        vec = []
        for m in module.modules():
            if type(m) in self.layer_types and not hasattr(
                    m, 'condensa_nocompress'):
                vec.append(m.weight.data.view(-1))
        return T.threshold(torch.cat(vec), self.density)

    def pi(self, module):
        """
        Applies compression scheme to module.
    
        :param module: PyTorch module.
        :type module: `torch.nn.Module`
        """
        threshold = self.threshold(module)
        for m in module.modules():
            if type(m) in self.layer_types and not hasattr(
                    m, 'condensa_nocompress'):
                condensa.prune(m, threshold)

    def delta(self, module):
        """
        Applies de-compression scheme to module.

        :param module: PyTorch module.
        :type module: `torch.nn.Module`
        """
        pass

    def __repr__(self):
        return '<Prune: density:{}>'.format(self.density)

class Quantize(object):
    """Quantizes network to given data-type."""
    def __init__(self, dtype=condensa.float16):
        """
        Creates `Quantize` class instance.

        :param dtype: Target data type (default: float16).
        """
        self.dtype = dtype
        self.layer_types = [torch.nn.Linear, torch.nn.Conv2d]

    def pi(self, module):
        """
        Applies compression scheme to module.
    
        :param module: PyTorch module.
        :type module: `torch.nn.Module`
        """
        for m in module.modules():
            if type(m) in self.layer_types and not hasattr(
                    m, 'condensa_nocompress'):
                condensa.quantize(m, self.dtype)

    def delta(self, module):
        """
        Applies de-compression scheme to module.

        :param module: PyTorch module.
        :type module: `torch.nn.Module`
        """
        for m in module.modules():
            if type(m) in self.layer_types and not hasattr(
                    m, 'condensa_nocompress'):
                condensa.dequantize(m, condensa.float32)

    def __repr__(self):
        return '<Quantize: dtype:{}>'.format(self.dtype)

class NeuronPrune(object):
    """Prunes neurons from fully-connected layers."""
    def __init__(self, density, align=None, criteria=F.l2norm,
                 prune_bias=True):
        """
        Creates an instance of `NeuronPrune`.

        :param density: Target density.
        :type density: `float`
        :param align: Tensor alignment in compressed model.
        :type align: `int`
        :param criteria: Neuron aggregation criteria (default: l2norm).
        :type criteria: `condensa.functional`
        :param prune_bias: Whether to prune corresponding biases (default: True).
        :type prune_bias: `bool`
        """
        self.density = density
        self.align = align
        self.criteria = criteria
        self.prune_bias = prune_bias

    def threshold(self, module):
        """
        Computes magnitude threshold.

        :param module: PyTorch module.
        :type module: `torch.nn.Module`
        """
        vec = []
        for m in module.modules():
            if isinstance(m, torch.nn.Linear) and not hasattr(
                    m, 'condensa_nocompress'):
                agg = T.aggregate_neurons(m.weight.data, self.criteria)
                vec.append(agg.view(-1))
        return T.threshold(torch.cat(vec), self.density)

    def pi(self, module):
        """
        Applies compression scheme to module.
    
        :param module: PyTorch module.
        :type module: `torch.nn.Module`
        """
        threshold = self.threshold(module)
        for m in module.modules():
            if isinstance(m, torch.nn.Linear) and not hasattr(
                    m, 'condensa_nocompress'):
                condensa.neuron_prune(m,
                                      threshold,
                                      align=self.align,
                                      criteria=self.criteria,
                                      prune_bias=self.prune_bias)

    def delta(self, module):
        """
        Applies de-compression scheme to module.

        :param module: PyTorch module.
        :type module: `torch.nn.Module`
        """
        pass

    def __repr__(self):
        return '<NeuronPrune: density:{}, align:{}, criteria:{}, prune_bias:{}>'.format(
            self.density, self.align, self.criteria, self.prune_bias)

class FilterPrune(object):
    """Prunes filters from convolutional layers."""
    def __init__(self, density, align=None, criteria=F.l2norm,
                 prune_bias=True):
        """
        Creates an instance of `FilterPrune`.

        :param density: Target density.
        :type density: `float`
        :param align: Tensor alignment in compressed model.
        :type align: `int`
        :param criteria: Filter aggregation criteria (default: l2norm).
        :type criteria: `condensa.functional`
        :param prune_bias: Whether to prune corresponding biases (default: True).
        :type prune_bias: `bool`
        """
        self.density = density
        self.align = align
        self.criteria = criteria
        self.prune_bias = prune_bias

    def threshold(self, module):
        """
        Computes magnitude threshold.

        :param module: PyTorch module.
        :type module: `torch.nn.Module`
        """
        vec = []
        for m in module.modules():
            if isinstance(m, torch.nn.Conv2d) and not hasattr(
                    m, 'condensa_nocompress'):
                agg = T.aggregate_filters(m.weight.data, self.criteria)
                vec.append(agg.view(-1))
        return T.threshold(torch.cat(vec), self.density)

    def pi(self, module):
        """
        Applies compression scheme to module.
    
        :param module: PyTorch module.
        :type module: `torch.nn.Module`
        """
        threshold = self.threshold(module)
        for m in module.modules():
            if isinstance(m, torch.nn.Conv2d) and not hasattr(
                    m, 'condensa_nocompress'):
                condensa.filter_prune(m,
                                      threshold,
                                      align=self.align,
                                      criteria=self.criteria,
                                      prune_bias=self.prune_bias)

    def delta(self, module):
        """
        Applies de-compression scheme to module.

        :param module: PyTorch module.
        :type module: `torch.nn.Module`
        """
        pass

    def __repr__(self):
        return '<FilterPrune: density:{}, align:{}, criteria:{}, prune_bias:{}>'.format(
            self.density, self.align, self.criteria, self.prune_bias)

class StructurePrune(object):
    """Combines neuron and filter pruning using a single threshold value."""
    def __init__(self, density, align=None, criteria=F.l2norm,
                 prune_bias=True):
        """
        Creates an instance of `StructurePrune`.

        :param density: Target density.
        :type density: `float`
        :param align: Tensor alignment in compressed model.
        :type align: `int`
        :param criteria: Structure aggregation criteria (default: l2norm).
        :type criteria: `condensa.functional`
        :param prune_bias: Whether to prune corresponding biases (default: True).
        :type prune_bias: `bool`
        """
        self.density = density
        self.align = align
        self.criteria = criteria
        self.prune_bias = prune_bias

    def threshold(self, module):
        """
        Computes magnitude threshold.

        :param module: PyTorch module.
        :type module: `torch.nn.Module`
        """
        vec = []
        for m in module.modules():
            if isinstance(m, torch.nn.Linear) and not hasattr(
                    m, 'condensa_nocompress'):
                agg = T.aggregate_neurons(m.weight.data, self.criteria)
                vec.append(agg.view(-1))
            if isinstance(m, torch.nn.Conv2d) and not hasattr(
                    m, 'condensa_nocompress'):
                agg = T.aggregate_filters(m.weight.data, self.criteria)
                vec.append(agg.view(-1))
        return T.threshold(torch.cat(vec), self.density)

    def pi(self, module):
        """
        Applies compression scheme to module.
    
        :param module: PyTorch module.
        :type module: `torch.nn.Module`
        """
        threshold = self.threshold(module)
        for m in module.modules():
            if isinstance(m, torch.nn.Linear) and not hasattr(
                    m, 'condensa_nocompress'):
                condensa.neuron_prune(m,
                                      threshold,
                                      align=self.align,
                                      criteria=self.criteria,
                                      prune_bias=self.prune_bias)
            if isinstance(m, torch.nn.Conv2d) and not hasattr(
                    m, 'condensa_nocompress'):
                condensa.filter_prune(m,
                                      threshold,
                                      align=self.align,
                                      criteria=self.criteria,
                                      prune_bias=self.prune_bias)

    def delta(self, module):
        """
        Applies de-compression scheme to module.

        :param module: PyTorch module.
        :type module: `torch.nn.Module`
        """
        pass

    def __repr__(self):
        return '<StructurePrune: density:{}, align:{}, criteria:{}, prune_bias:{}>'.format(
            self.density, self.align, self.criteria, self.prune_bias)

class BlockPrune(object):
    """Prunes blocks in Linear layers."""
    def __init__(self, density, block_size, criteria=F.l2norm):
        """
        Creates an instance of `BlockPrune`.

        :param density: Target density.
        :type density: `float`
        :param block_size: Target block size.
        :type block_size: `Tuple`
        :param criteria: Structure aggregation criteria (default: l2norm).
        :type criteria: `condensa.functional`
        """
        self.density = density
        self.block_size = block_size
        self.criteria = criteria
        self.layer_types = [torch.nn.Linear]

    def threshold(self, module):
        """
        Computes magnitude threshold.

        :param module: PyTorch module.
        :type module: `torch.nn.Module`
        """
        vec = []
        for m in module.modules():
            if type(m) in self.layer_types and not hasattr(
                    m, 'condensa_nocompress'):
                agg = T.aggregate(m.weight.data, self.block_size,
                                  self.criteria)
                vec.append(agg.view(-1))
        return T.threshold(torch.cat(vec), self.density)

    def pi(self, module):
        """
        Applies compression scheme to module.
    
        :param module: PyTorch module.
        :type module: `torch.nn.Module`
        """
        threshold = self.threshold(module)
        for m in module.modules():
            if type(m) in self.layer_types and not hasattr(
                    m, 'condensa_nocompress'):
                condensa.blockprune(m,
                                    threshold,
                                    block_size=self.block_size,
                                    criteria=self.criteria)

    def delta(self, module):
        """
        Applies de-compression scheme to module.

        :param module: PyTorch module.
        :type module: `torch.nn.Module`
        """
        pass

    def __repr__(self):
        return '<BlockPrune: density:{}, block_size:{}>'.format(
            self.density, self.block_size)
