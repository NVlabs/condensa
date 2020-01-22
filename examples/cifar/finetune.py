# Copyright 2020 NVIDIA Corporation
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

import os
import sys
import argparse
import logging
import csv

import gzip
import pickle

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.utils
import torchvision.datasets as datasets
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms

import condensa

import util
import models

if __name__ == '__main__':
    model_names = sorted(
        name for name in models.__dict__
        if not name.startswith("__") and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser(description='CIFAR fine-tuning script')
    parser.add_argument('--arch',
                        default='AlexNet',
                        choices=model_names,
                        help='Model architecture: ' + ' | '.join(model_names) +
                        ' (default: alexnet)')
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--model', help='Pretrained model filename')
    parser.add_argument('--epochs',
                        type=int,
                        help='Number of fine-tuning epochs')
    parser.add_argument('--batch_size',
                        type=int,
                        default=128,
                        help='Batch size for training')
    parser.add_argument('--val_batch_size',
                        type=int,
                        default=128,
                        help='Validation batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--lr_end',
                        type=float,
                        default=0.01,
                        help='Ending learning rate')
    parser.add_argument('--momentum',
                        type=float,
                        default=0.9,
                        help='SGD momentum')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0,
                        help='SGD weight decay')
    parser.add_argument('--out',
                        default='finetuned.pth',
                        help='Fine-tuned output model filename')
    parser.add_argument('-v',
                        '--verbose',
                        help='verbose logging output',
                        action='store_true')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(message)s')

    if args.dataset == 'cifar10':
        dataset = datasets.CIFAR10
        num_classes = 10
    elif args.dataset == 'cifar100':
        dataset = datasets.CIFAR100
        num_classes = 100
    else:
        raise RuntimeError('Invalid dataset: must be cifar10 or cifar100')

    # Load model architecture
    if args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](num_classes=num_classes)
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)

    model.load_state_dict(torch.load(args.model))
    # Compute #nonzeros prior to fine-tuning
    nparams_w = torch.nn.utils.parameters_to_vector(
        model.parameters()).view(-1).nonzero().numel()

    # Only fine-tune fully-connected and convolutional layers
    layer_types = [torch.nn.Linear, torch.nn.Conv2d]

    trainloader,valloader = \
        util.cifar_train_val_loader(dataset,
                                    args.batch_size,
                                    args.val_batch_size)
    testloader = util.cifar_test_loader(dataset, args.val_batch_size)
    criterion = torch.nn.CrossEntropyLoss().cuda()
    ft = condensa.FineTuner(model, layer_types)
    w_ft = ft.run(epochs=args.epochs,
                  lr=args.lr,
                  lr_end=args.lr_end,
                  momentum=args.momentum,
                  weight_decay=args.weight_decay,
                  criterion=criterion,
                  trainloader=trainloader,
                  testloader=testloader,
                  valloader=valloader,
                  debugging_flags={'custom_model_statistics':
                                    condensa.util.cnn_statistics})
    nparams_wft = torch.nn.utils.parameters_to_vector(
        w_ft.parameters()).view(-1).nonzero().numel()
    print('#Nonzero parameters: before [{}], after [{}]'.format(
        nparams_w, nparams_wft))

    if args.out is not None:
        torch.save(w_ft.state_dict(), args.out)
        logging.info('[Condensa] Fine-tuned model written to disk')
