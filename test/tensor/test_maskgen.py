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

import numpy as np
import torch

import condensa.tensor as T

def test_simple_mask():
    a = torch.randn(20).cuda()
    threshold = T.threshold(a, 0.3)
    mask = T.simple_mask(a, threshold)

    for i in range(len(a)):
        if abs(a[i]) >= threshold: assert mask[i] == 1
        else: assert mask[i] == 0

def test_block_mask():
    pass

if __name__ == '__main__':
    test_simple_mask()
    test_block_mask()
