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

import sys
from copy import deepcopy
import logging
from collections import defaultdict

import numpy as np
import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from condensa.util import EventTimer
from condensa import cfg
from condensa import util
from .sgd import SGD
from condensa.lr import *

logger = logging.getLogger(__name__)

class record_mode(object):
    def __enter__(self):
        cfg.__CONDENSA_RECORD_MODE__ = True

    def __exit__(self, *args):
        cfg.__CONDENSA_RECORD_MODE__ = False
        return False

class LC(object):
    """Condensa L-C compression engine."""
    def __init__(self,
                 steps=30,
                 l_optimizer=None,
                 l_optimizer_params={},
                 lr=None,
                 lr_end=None,
                 lr_decay=None,
                 lr_schedule=None,
                 lr_multiplier=None,
                 mb_iterations_per_l=0,
                 mb_iterations_first_l=0,
                 mu_init=0.,
                 mu_multiplier=1.,
                 mu_cap=10000,
                 distributed=False,
                 debugging_flags={}):
        """
        Constructs an `LC` class instance.

        :param steps: Number of L-C iterations.
        :type steps: float
        :param l_optimizer: L-step optimizer to use.
        :param l_optimizer_params: L-step optimizer hyper-parameters.
        :type l_optimizer_params: dict
        :param lr: Starting learning rate.
        :type lr: float
        :param lr_end: Ending learning rate.
        :type lr_end: float
        :param lr_schedule: Learning rate schedule.
        :type lr_schedule: List
        :param lr_multiplier: Learning rate multiplier.
        :type lr_multiplier: float
        :param mb_iterations_per_l: Number of mini-batch iterations per L-step.
        :type mb_iterations_per_l: int
        :param mb_iterations_first_l: Number of mini-batch iterations for first L-step.
        :type mb_iterations_first_l: int
        :param mu_init: Initial value of `mu`.
        :type mu_init: float
        :param mu_multiplier: Mu multiplier.
        :type mu_multiplier: float
        :param mu_cap: Maximum permitted value for `mu`.
        :type mu_cap: float
        :param distributed: Enable/disable data-parallelism in L-step.
        :type distributed: bool
        :param debugging_flags: Debugging flags
        :type debugging_flags: dict
        """
        self._engine_config = {
            k: v
            for k, v in locals().items() if k != 'self'
        }
        logger.info('[Condensa] LC ENGINE CONFIG [' +
                    ', '.join('{!s}={!r}'.format(k, v)
                              for k, v in self._engine_config.items()) + ']')

        if not 0 <= steps:
            raise ValueError(
                'Invalid steps specified: {}'.format(steps))
        if not isinstance(l_optimizer_params, dict):
            raise TypeError('l_optimizer_params must be a dictionary')
        if not 0. <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if lr_schedule is not None and lr_multiplier is None:
            raise TypeError(
                'Please specify multiplier when using fixed LR schedule')
        if not 0 < mb_iterations_per_l:
            raise ValueError(
                'Invalid mb_iterations_per_l specified: {}'.format(mb_iterations_per_l))
        if not 0 < mb_iterations_first_l:
            raise ValueError(
                'Invalid mb_iterations_first_l specified: {}'.format(mb_iterations_first_l))
        if not isinstance(debugging_flags, dict):
            raise TypeError('debugging_flags must be a dictionary')

        self.use_cuda = torch.cuda.is_available()
        self.steps = steps
        self.l_optimizer = l_optimizer if l_optimizer else SGD
        self.l_optimizer_params = l_optimizer_params
        self.lr = lr
        self.lr_end = lr_end
        self.lr_decay = lr_decay
        self.lr_schedule = lr_schedule
        self.lr_multiplier = lr_multiplier
        self.mb_iterations_per_l = mb_iterations_per_l
        self.mb_iterations_first_l = mb_iterations_first_l
        self.mu_init = mu_init
        self.mu_multiplier = mu_multiplier
        self.mu_cap = mu_cap
        self.distributed = distributed
        self.debugging_flags = debugging_flags

    def condensa_loss(self, mu, w, theta, lm):
        """
        Computes L-C reconstruction loss.
    
        :param mu: LC `mu` hyper-parameter value.
        :type mu: float
        :param w: Input model.
        :type w: torch.nn.Module
        :param theta: Compressed model.
        :type theta: torch.nn.Module
        :param lm: Lagrange multiplier.
        :type lm: torch.nn.Module
        """
        w.eval()
        loss = 0.
        with torch.no_grad():
            for w_m, theta_m, lm_m in zip(w.modules(), theta.modules(),
                                          lm.modules()):
                if hasattr(theta_m, 'condense'):
                    for pname in theta_m.condense:
                        w_p = getattr(w_m, pname).detach()
                        theta_p = getattr(theta_m, pname).data
                        lm_p = getattr(lm_m, pname).data
                        compression_loss = w_p - theta_p
                        qp_term = (compression_loss * compression_loss).sum()
                        al_term = (compression_loss * lm_p).sum()
                        loss += 0.5 * mu * qp_term - al_term
        return loss

    def zero_(self, model):
        """
        Zeroes out model parameters.
    
        :param model: PyTorch model.
        :type model: torch.nn.Module
        """
        with torch.no_grad():
            pflat = torch.nn.utils.parameters_to_vector(
                model.parameters()).fill_(0.)
            torch.nn.utils.vector_to_parameters(pflat, model.parameters())

    def compress(self, w, pi, delta, trainloader, testloader, valloader,
                 loss_fn):
        """
        Main L-C compression method.
    
        :param w: Input model.
        :type w: torch.nn.Module
        :param pi: Compression function.
        :param delta: Decompression function.
        :param trainloader: Training dataloader.
        :param testloader: Test dataloader.
        :param valloader: Validation dataloader.
        :param loss_fn: Loss criterion.
        """

        statistics = {}
        # Save engine configuration
        statistics.update(self._engine_config)

        _print_acc = self.debugging_flags['print_accuracies']\
                     if 'print_accuracies' in self.debugging_flags\
                     else False
        _disable_train_stats = self.debugging_flags['disable_train_stats']\
                     if 'disable_train_stats' in self.debugging_flags\
                     else False
        timer_lc = EventTimer()

        if self.use_cuda: cudnn.benchmark = True
        logger.debug("[Condensa] cuDNN VERSION: {}".format(cudnn.version()))

        validate = (valloader is not None)
        test = (testloader is not None)

        # Copy model to GPU0 memory
        if self.use_cuda: w = w.cuda(0)

        # Mark all compressible modules in w
        with record_mode():
            pi(w)

        with torch.no_grad():
            theta = deepcopy(w)
        self.zero_(theta)

        with torch.no_grad():
            lm = deepcopy(w)
        self.zero_(lm)

        with torch.no_grad():
            best_model = deepcopy(w)

        # Enable data-parallelism in  L step
        if self.use_cuda and self.distributed:
            ngpus = torch.cuda.device_count()
            logger.info('[Condensa] {} GPUs enabled for L-step'.format(ngpus))
            w = torch.nn.DataParallel(w)

        mu = 0.
        learning_rate = self.lr

        optimizer = self.l_optimizer(w,
                                     lr=learning_rate,
                                     **self.l_optimizer_params)
        optimizer.reset_state()

        if _print_acc:
            if not _disable_train_stats:
                w_train_loss, w_train_acc = util.loss_and_accuracy(w, loss_fn, trainloader)
                logger.info('[Condensa] w TRAIN\tloss={:.5f}, acc={:.4f}'.format(w_train_loss, w_train_acc))
            if validate:
                w_val_loss, w_val_acc = util.loss_and_accuracy(w, loss_fn, valloader)
                logger.info('[Condensa] w VAL\tloss={:.5f}, acc={:4f}'.format(w_val_loss, w_val_acc))
            if test:
                w_test_loss, w_test_acc = util.loss_and_accuracy(w, loss_fn, testloader)
                logger.info('[Condensa] w TEST\tloss={:.5f}, acc={:4f}'.format(w_test_loss, w_test_acc))
        else:
            if not _disable_train_stats:
                w_train_loss = util.loss(w, loss_fn, trainloader)
                logger.info('[Condensa] w TRAIN\tloss={:.5f}'.format(w_train_loss))
            if validate:
                w_val_loss = util.loss(w, loss_fn, valloader)
                logger.info('[Condensa] w VAL\tloss={:.5f}'.format(w_val_loss))
            if test:
                w_test_loss = util.loss(w, loss_fn, testloader)
                logger.info('[Condensa] w TEST\tloss={:.5f}'.format(w_test_loss))

        best_loss = sys.float_info.max
        train_losses = []
        if validate: val_losses = []
        if test: test_losses = []
        outer_lr_scheduler = None
        if self.lr_decay is not None:
            outer_lr_scheduler = ExpDecayedLR(self.lr, self.lr_decay)
        elif self.lr_schedule is not None:
            outer_lr_scheduler = DecayedLR(self.lr, self.lr_schedule,
                                           self.lr_multiplier)
        for j in range(0, self.steps):
            n_sgd_iter = (self.mb_iterations_first_l
                          if j == 1 else self.mb_iterations_per_l)

            # Set up outer learning rate
            learning_rate = self.lr
            if outer_lr_scheduler is not None:
                learning_rate = outer_lr_scheduler.learning_rate

            logger.info(
                '[Condensa] LC Iteration {}:\tmu={:.5f}, lr={:.5f}'.format(
                    j, mu, learning_rate))

            inner_lr_scheduler = None
            if self.lr_end is not None:
                inner_lr_scheduler = IntervalLR(learning_rate, self.lr_end,
                                                n_sgd_iter)

            # L step
            # Switch to training mode
            i = 0
            w.train()
            iterator = iter(trainloader)
            if logger.isEnabledFor(logging.INFO) and j>0:
                pbar = tqdm(total=n_sgd_iter, ascii=True)
            while True:
                if j == 0:
                    logger.info('[Condensa] Skipping first L-step')
                    break
                if j == 1 and i >= self.mb_iterations_first_l:
                    break
                if j > 1 and i >= self.mb_iterations_per_l:
                    break

                try:
                    inputs, targets = next(iterator)
                except StopIteration:
                    iterator = iter(trainloader)
                    inputs, targets = next(iterator)

                if self.use_cuda:
                    if not inputs.is_cuda: inputs = inputs.cuda()
                    if not targets.is_cuda:
                        targets = targets.cuda(non_blocking=True)
                outputs = w(inputs)
                loss = loss_fn(outputs, targets)

                optimizer.zero_grad()

                loss.backward()
                optimizer.step(learning_rate, mu, theta, lm)

                if inner_lr_scheduler is not None:
                    inner_lr_scheduler.step()
                    learning_rate = inner_lr_scheduler.learning_rate

                if logger.isEnabledFor(logging.INFO):
                    pbar.update()
                i += 1

            if logger.isEnabledFor(logging.INFO) and j>0:
                pbar.close()
            logger.info('')

            if self.use_cuda: torch.cuda.synchronize()

            w.eval()
            # C step and theta update
            try:
                theta.load_state_dict(w.module.state_dict())
            except AttributeError:
                theta.load_state_dict(w.state_dict())
            if mu > 0:
                try:
                    wmodules = w.module.modules()
                except AttributeError:
                    wmodules = w.modules()
                with record_mode():
                    pi(theta)
                with torch.no_grad():
                    for w_m, theta_m, lm_m in zip(wmodules, theta.modules(),
                                                  lm.modules()):
                        if hasattr(theta_m, 'condense'):
                            for pname in theta_m.condense:
                                getattr(theta_m, pname).data = (
                                    getattr(w_m, pname).detach() -
                                    getattr(lm_m, pname).data / mu)

            pi(theta)

            if _print_acc:
                if not _disable_train_stats:
                    nested_train_loss, nested_train_acc = util.loss_and_accuracy(
                        theta, loss_fn, trainloader)
                    logger.info(
                        '[Condensa] Nested (theta) TRAIN\tloss={:.5f}, acc={:.4f}'.
                        format(nested_train_loss, nested_train_acc))
                if validate:
                    nested_val_loss, nested_val_acc = util.loss_and_accuracy(
                        theta, loss_fn, valloader)
                    logger.info(
                        '[Condensa] Nested (theta) VAL\tloss={:.5f}, acc={:.4f}'.
                        format(nested_val_loss, nested_val_acc))
                if test:
                    nested_test_loss, nested_test_acc = util.loss_and_accuracy(
                        theta, loss_fn, testloader)
                    logger.info(
                        '[Condensa] Nested (theta) TEST\tloss={:.5f}, acc={:.4f}'.
                        format(nested_test_loss, nested_test_acc))
            else:
                if not _disable_train_stats:
                    nested_train_loss = util.loss(theta, loss_fn, trainloader)
                    logger.info(
                        '[Condensa] Nested (theta) TRAIN\tloss={:.5f}'.
                        format(nested_train_loss))
                if validate:
                    nested_val_loss = util.loss(theta, loss_fn, valloader)
                    logger.info(
                        '[Condensa] Nested (theta) VAL\tloss={:.5f}'.
                        format(nested_val_loss))
                if test:
                    nested_test_loss = util.loss(theta, loss_fn, testloader)
                    logger.info(
                        '[Condensa] Nested (theta) TEST\tloss={:.5f}'.
                        format(nested_test_loss))
            if not _disable_train_stats:
                train_losses.append(nested_train_loss)
            if test: test_losses.append(nested_test_loss)
            if validate: val_losses.append(nested_val_loss)

            if validate:
                if nested_val_loss < best_loss:
                    logger.info('[Condensa] SAVING MODEL based on VAL')
                    best_loss = nested_val_loss
                    # Deep-copy required here to preserve dtypes
                    best_model = deepcopy(theta)
            elif test:
                if nested_test_loss < best_loss:
                    logger.info('[Condensa] SAVING MODEL based on TEST')
                    best_loss = nested_test_loss
                    # Deep-copy required here to preserve dtypes
                    best_model = deepcopy(theta)
            else:
                logger.info('[Condensa] SAVING MODEL based on most recent')
                best_model = deepcopy(theta)

            # theta <- delta(theta)
            delta(theta)

            # LM update
            if mu > 0:
                try:
                    wmodules = w.module.modules()
                except AttributeError:
                    wmodules = w.modules()
                for w_m, theta_m, lm_m in zip(wmodules, theta.modules(),
                                              lm.modules()):
                    if hasattr(theta_m, 'condense'):
                        for pname in theta_m.condense:
                            getattr(
                                lm_m,
                                pname).data = (getattr(lm_m, pname).data - mu *
                                               (getattr(w_m, pname).detach() -
                                                getattr(theta_m, pname).data))

            optimizer.reset_state()
            # Update mu
            mu = self._update_mu(mu, self.mu_init, self.mu_multiplier,
                                 self.mu_cap)
            # Update LR schedule
            if outer_lr_scheduler is not None: outer_lr_scheduler.step()

        statistics['elapsed_lc'] = timer_lc.elapsed_seconds
        statistics['train_losses'] = train_losses
        if test: statistics['test_losses'] = test_losses
        if validate: statistics['val_losses'] = val_losses
        return best_model, statistics

    def _update_mu(self, mu, mu_init, mu_multiplier, mu_cap):
        if mu > mu_cap:
            return mu
        if mu != 0:
            return mu * mu_multiplier
        else:
            return mu_init
