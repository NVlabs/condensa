# Copyright 2020 NVIDIA Corporation
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
import condensa.schemes as schemes
import condensa.tensor as T
import condensa.functional as F

def test_prune(device):
    fc = torch.nn.Linear(100, 10, bias=True).to(device)
    scheme = schemes.Prune(0.5)
    threshold = scheme.threshold(fc)
    scheme.pi(fc)

    t = fc.weight.data.abs().view(-1)
    nzs = torch.index_select(t, 0, t.nonzero().view(-1))
    assert (nzs >= threshold).all()

def test_filter_prune(device):
    conv = torch.nn.Conv2d(3,
                           64,
                           kernel_size=11,
                           stride=4,
                           padding=5,
                           bias=True).to(device)

    criteria = F.l2norm
    scheme = schemes.FilterPrune(0.5, criteria=criteria, prune_bias=True)
    threshold = scheme.threshold(conv)
    scheme.pi(conv)

    # Check against threshold
    agg = T.aggregate_filters(conv.weight.data, criteria).view(-1)
    nzs = torch.index_select(agg, 0, agg.nonzero().view(-1))
    assert (nzs >= threshold).all()

    # Check biases: all zero filters must have corresponding zero biases
    zero_indices = (agg == 0).nonzero().view(-1)
    z = torch.index_select(conv.bias.data, 0, zero_indices)
    assert (z == 0.).all()

def test_neuron_prune(device):
    fc = torch.nn.Linear(100, 10, bias=True).to(device)

    criteria = F.l2norm
    scheme = schemes.NeuronPrune(0.5, criteria=criteria, prune_bias=True)
    threshold = scheme.threshold(fc)
    scheme.pi(fc)

    # Check against threshold
    agg = T.aggregate_neurons(fc.weight.data, criteria).view(-1)
    nzs = torch.index_select(agg, 0, agg.nonzero().view(-1))
    assert (nzs >= threshold).all()

    # Check biases: all zero neurons must have corresponding zero biases
    zero_indices = (agg == 0).nonzero().view(-1)
    z = torch.index_select(fc.bias.data, 0, zero_indices)
    assert (z == 0.).all()

def test_block_prune(device, blocksize=(10,10)):
    fc = torch.nn.Linear(100, 100, bias=False).to(device)

    criteria = F.l2norm
    scheme = schemes.BlockPrune(0.5, criteria=criteria, block_size=blocksize)
    threshold = scheme.threshold(fc)
    scheme.pi(fc)

    # Check against threshold
    agg = T.aggregate(fc.weight.data, blocksize, criteria).view(-1)
    nzs = torch.index_select(agg, 0, agg.nonzero().view(-1))
    assert (nzs >= threshold).all()

if __name__ == '__main__':
    test_prune('cpu')
    test_filter_prune('cpu')
    test_neuron_prune('cpu')
    test_block_prune('cpu')

    if torch.cuda.is_available():
        test_prune('cuda:0')
        test_filter_prune('cuda:0')
        test_neuron_prune('cuda:0')
        test_block_prune('cuda:0')
