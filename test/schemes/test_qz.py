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
from condensa import schemes

def test_float16(device):
    scheme = schemes.Quantize(condensa.float16)
    fc = torch.nn.Linear(100, 10).float().to(device)

    scheme.pi(fc)
    assert fc.weight.dtype == torch.float16
    scheme.delta(fc)
    assert fc.weight.dtype == torch.float32

if __name__ == '__main__':
    test_float16('cpu')
    if torch.cuda.is_available():
        test_float16('cpu')
