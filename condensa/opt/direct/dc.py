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

from copy import deepcopy
import torch

from condensa.util import EventTimer

class DC(object):
    """Condensa direct compression optimizer."""
    def compress(self,
                 w,
                 pi,
                 delta,
                 trainloader,
                 testloader,
                 valloader,
                 criterion):
        """
        Performs model compression using direct optimization.

        :param w: PyTorch model.
        :type w: `torch.nn.Module`
        :param pi: Compression function.
        :param delta: Decompression function.
        :param trainloader: Training dataloader.
        :param testloader: Test dataloader.
        :param valloader: Validation dataloader.
        :param criterion: Loss criterion.
        """
        statistics = dict()
        timer_dc = EventTimer()
        with torch.no_grad():
            compressed = deepcopy(w)
        pi(compressed)
        statistics['total_elapsed'] = timer_dc.elapsed_seconds

        return compressed, statistics
