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

import math
from collections import defaultdict

import torch

class Adam(object):
    """Custom Adam implementation for L-C optimizer."""
    def __init__(self,
                 w,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0,
                 amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(
                betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(
                betas[1]))

        try:
            self.w = w.module
        except AttributeError:
            self.w = w

        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad

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
            if 'state' in self.state[p]:
                self.state[p]['step'] = 0
            if 'exp_avg' in self.state[p]:
                self.state[p]['exp_avg'] = torch.zeros_like(p.data)
            if 'exp_avg_sq' in self.state[p]:
                self.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)
            if self.amsgrad and 'max_exp_avg_sq' in self.state[p]:
                self.state[p]['max_exp_avg_sq'] = torch.zeros_like(p.data)

    def _step(self, p, condense=False, mu=None, p_theta=None, p_lm=None):
        if p.grad is None:
            return

        grad = p.grad.data
        if grad.is_sparse:
            raise RuntimeError('Adam does not support sparse gradients.')

        state = self.state[p]
        # State initialization
        if len(state) == 0:
            state['step'] = 0
            # Exponential moving average of gradient values
            state['exp_avg'] = torch.zeros_like(p.data)
            # Exponential moving average of squared gradient values
            state['exp_avg_sq'] = torch.zeros_like(p.data)
            if self.amsgrad:
                # Maintains max of all exp. moving avg. of sq. grad. values
                state['max_exp_avg_sq'] = torch.zeros_like(p.data)

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        if self.amsgrad:
            max_exp_avg_sq = state['max_exp_avg_sq']
        beta1, beta2 = self.betas

        state['step'] += 1

        if self.weight_decay != 0:
            grad.add_(self.weight_decay, p.data)

        if condense is True:
            assert (mu is not None
                    and p_theta is not None
                    and p_lm is not None)
            grad.add_(mu * (p.data - p_theta.data) - p_lm.data)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(1 - beta1, grad)
        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
        if self.amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            # Use the max. for normalizing running avg. of gradient
            denom = max_exp_avg_sq.sqrt().add_(self.eps)
        else:
            denom = exp_avg_sq.sqrt().add_(self.eps)

        bias_correction1 = 1 - beta1**state['step']
        bias_correction2 = 1 - beta2**state['step']
        step_size = self.learning_rate * math.sqrt(
            bias_correction2) / bias_correction1

        p.data.addcdiv_(-step_size, exp_avg, denom)

    def step(self, lr, mu, theta, lm, closure=None):
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
