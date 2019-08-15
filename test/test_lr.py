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

import condensa
import condensa.lr as lr

def test_interval_lr():
    schedule = lr.IntervalLR(1., 1e-6, 100)
    assert schedule.learning_rate == 1.
    for i in range(0, 100):
        schedule.step()
    assert np.isclose(schedule.learning_rate, 1e-6)

def test_decayed_lr():
    schedule = lr.DecayedLR(100.0, [10, 20], gamma=0.1)
    for i in range(0, 30):
        schedule.step()
        if i == 10: assert schedule.learning_rate == 10.0
        elif i == 20: assert schedule.learning_rate == 1.0

def test_exp_decayed_lr():
    schedule = lr.ExpDecayedLR(1.0, 0.1)
    for i in range(0, 100):
        schedule.step()
    assert schedule.learning_rate == 1.0 * (0.1**100)

if __name__ == '__main__':
    test_interval_lr()
    test_decayed_lr()
    test_exp_decayed_lr()
