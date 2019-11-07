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

from collections import defaultdict
import torch

class SGD(object):
    """Custom SGD implementation for L-C optimizer."""
    def __init__(self, w, lr=None, momentum=None, weight_decay=0):
        """
        Creates instance of `SGD`.

        :param w: PyTorch model.
        :type w: torch.nn.Module
        :param lr: Learning rate.
        :type lr: float
        :param momentum: SGD momentum.
        :type momentum: float
        :param weight_decay: Weight decay amount (L2 regularation).
        :type weight_decay: float
        """
        if lr is None or momentum is None:
            raise ValueError('Learning rate and momentum are required')
        if lr < 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if momentum < 0.0:
            raise ValueError('Invalid momentum value: {}'.format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                'Invalid weight decay value: {}'.format(weight_decay))

        try:
            self.w = w.module
        except AttributeError:
            self.w = w

        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.state = defaultdict(dict)

    def zero_grad(self):
        """Zeroes out all gradients."""
        for p in self.w.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def reset_state(self):
        """Resets optimizer state."""
        for p in self.w.parameters():
            if 'velocity' in self.state[p]:
                self.state[p]['velocity'] = torch.zeros_like(p.data)

    def _step(self, p, condense=False, mu=None, p_theta=None, p_lm=None):
        if p.grad is None:
            return
        lr = self.learning_rate
        d_p = p.grad.data
        if self.weight_decay != 0:
            d_p.add_(self.weight_decay, p.data)
        if condense is True:
            assert (mu is not None
                    and p_theta is not None
                    and p_lm is not None)
            d_p.add_(mu * (p.data - p_theta.data) - p_lm.data)
        update = p.data - lr * (d_p)
        if 'velocity' not in self.state[p]:
            velocity = torch.zeros_like(p.data)
        else:
            velocity = self.state[p]['velocity']
        x = self.momentum * velocity + update - p.data
        self.state[p]['velocity'] = x
        p.data = self.momentum * x + update

    def step(self, lr, mu, theta, lm, closure=None):
        """
        Takes one optimizer step.

        :param lr: Current learning rate.
        :type lr: float
        :param mu: L-C mu hyper-parameter value.
        :type mu: float
        :param theta: Compressed model.
        :type theta: torch.nn.Module
        :param lm: Lagrange multiplier.
        :type lm: torch.nn.Module
        :param closure: Loss closure.
        """
        loss = None
        if closure is not None:
            loss = closure()

        self.learning_rate = lr
        for w_m, theta_m, lm_m in zip(self.w.modules(), theta.modules(),
                                      lm.modules()):
            if hasattr(theta_m, 'condense'):
                for pname in theta_m.condense:
                    self._step(getattr(w_m, pname), True, mu,
                               getattr(theta_m, pname), getattr(lm_m, pname))
                params = set([name for name, _ in theta_m.named_parameters()])
                rparams = params - theta_m.condense
                for pname in rparams:
                    self._step(getattr(w_m, pname))
            else:
                for w_p in w_m.parameters(recurse=False):
                    self._step(w_p)
        return loss
