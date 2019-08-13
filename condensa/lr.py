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

class IntervalLR(object):
    """Decays learning rate between two values."""
    def __init__(self, begin, end, n):
        """
        Construct an instance of `IntervalLR`.

        :param begin: Starting learning rate (LR).
        :type begin: `float`
        :param end: Ending LR.
        :type end: `float`
        :param n: Number of iterations.
        :type n: `int`
        """
        self.alpha = np.exp((np.log(end) - np.log(begin)) / n)
        self.lr = begin

    def step(self):
        """Signal end of iteration."""
        self.lr *= self.alpha

    @property
    def learning_rate(self):
        """Returns current learning rate."""
        return self.lr

class DecayedLR(object):
    """Decays learning rate at fixed intervals."""
    def __init__(self, begin, schedule, gamma=0.1):
        """
        Construct an instance of `DecayedLR`.

        :param begin: Starting LR.
        :type begin: `float`
        :param schedule: List of iterations when LR must be adjusted.
        :type schedule: `List/Tuple`
        :param gamma: LR multiplier.
        :type gamma: `float`
        """
        self.gamma = gamma
        self.lr = begin
        self.schedule = schedule
        self.counter = 0

    def step(self):
        """Signal end of iteration."""
        if self.counter in self.schedule:
            self.lr *= self.gamma
        self.counter += 1

    @property
    def learning_rate(self):
        """Returns current learning rate."""
        return self.lr

class ExpDecayedLR(object):
    """Decays learning rate exponentially."""
    def __init__(self, begin, gamma):
        """
        Construct an instance of `ExpDecayedLR`.

        :param begin: Starting LR.
        :type begin: `float`
        :param gamma: LR multiplier.
        :type gamma: `float`
        """
        self.gamma = gamma
        self.lr = begin
        self.counter = 0

    def step(self):
        """Signal end of iteration."""
        self.counter += 1

    @property
    def learning_rate(self):
        """Returns current learning rate."""
        return self.lr * (self.gamma**self.counter)
