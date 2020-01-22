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

import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data

import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

import condensa.data

def cifar_train_val_loader(dataset,
                           train_batch_size,
                           val_batch_size,
                           root='./data',
                           random_seed=42,
                           shuffle=True):
    """
    Splits the CIFAR training set into training and validation
    sets (9:1 split) and returns the corresponding data loaders.
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ])
    trainset = dataset(root=root,
                       train=True,
                       download=True,
                       transform=transform_train)
    valset = dataset(root=root, train=True, download=True, transform=None)
    num_train = len(trainset)
    indices = list(range(num_train))
    split = 5000

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, val_idx = indices[split:], indices[:split]
    trainsampler = SubsetRandomSampler(train_idx)
    valsampler = SubsetRandomSampler(val_idx)

    meanstd = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    trainloader = condensa.data.GPUDataLoader(trainset,
                                              batch_size=train_batch_size,
                                              shuffle=False,
                                              num_workers=8,
                                              sampler=trainsampler,
                                              meanstd=meanstd)
    valloader =   condensa.data.GPUDataLoader(valset,
                                              batch_size=val_batch_size,
                                              shuffle=False,
                                              num_workers=8,
                                              sampler=valsampler,
                                              meanstd=meanstd)

    return (trainloader, valloader)

def cifar_test_loader(dataset, batch_size, root='./data'):
    """
    Construct a CIFAR test dataset loader.
    """
    testset = dataset(root=root, train=False, download=True, transform=None)
    meanstd = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    testloader = condensa.data.GPUDataLoader(testset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=8,
                                             meanstd=meanstd)
    return testloader
