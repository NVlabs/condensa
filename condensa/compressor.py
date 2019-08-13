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

import torch.nn

class Compressor(object):
    """Condensa model compressor class."""
    def __init__(self,
                 opt,
                 scheme,
                 model,
                 trainloader,
                 testloader,
                 valloader,
                 criterion):
        """
        Creates a `Compressor` instance.

        :param opt: Optimizer.
        :type opt: `condensa.Optimizer`
        :param scheme: Compression scheme (class).
        :param model: PyTorch model.
        :type model: `torch.nn.Module`
        :param trainloader: Training dataloader.
        :param testloader: Test dataloader.
        :param valloader: Validation dataloader.
        :param criterion: Loss criterion.
        """
        assert isinstance(model, torch.nn.Module)

        self.opt = opt
        self.pi = scheme.pi
        self.delta = scheme.delta
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.valloader = valloader
        self.criterion = criterion

        self._statistics = None

    @property
    def statistics(self):
        """
        Retrieves compressed model statistics.

        :return: Model statistics.
        :rtype: `dict`
        """
        return self._statistics

    def run(self):
        """
        Executes model compressor.

        :return: Compressed model.
        :rtype: `torch.nn.Module`
        """
        w, statistics = self.opt.compress(self.model, self.pi, self.delta,
                                          self.trainloader, self.testloader,
                                          self.valloader, self.criterion)
        self._statistics = statistics
        return w
