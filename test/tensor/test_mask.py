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

import condensa.tensor as T


def test_simple_mask(device='cuda'):
    a = torch.randn(20).to(device)
    threshold = T.threshold(a, 0.3)
    mask = T.simple_mask(a, threshold)

    for i in range(len(a)):
        if abs(a[i]) >= threshold: assert mask[i] == 1
        else: assert mask[i] == 0


def test_block_mask(device='cuda'):
    pass


def test_apply_mask(device='cuda'):
    a = torch.randn(20).to(device)
    threshold = T.threshold(a, 0.3)
    mask = T.simple_mask(a, threshold)
    T.apply_mask_inplace(a, mask)

    for i in range(len(a)):
        assert a[i] == 0. or abs(a[i]) >= threshold


def test_get_mask(device='cuda'):
    a = torch.randn(20).to(device)
    threshold = T.threshold(a, 0.3)
    mask = T.simple_mask(a, threshold)
    T.apply_mask_inplace(a, mask)
    assert torch.equal(mask, T.get_mask(a))
    

if __name__ == '__main__':
    test_simple_mask('cpu')
    test_block_mask('cpu')
    test_apply_mask('cpu')
    test_get_mask('cpu')
    if torch.cuda.is_available():
        test_simple_mask('cuda:0')
        test_block_mask('cuda:0')
        test_apply_mask('cuda:0')
        test_get_mask('cuda:0')
