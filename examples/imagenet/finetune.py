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

import os
import warnings
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
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
import torchvision.models as models

import condensa
import condensa.data

if __name__ == '__main__':
    # Suppress EXIF warning messages
    warnings.filterwarnings("ignore",
        "(Possibly )?corrupt EXIF data", UserWarning)
    model_names = sorted(
        name for name in models.__dict__
        if not name.startswith("__") and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser(description='ImageNet fine-tuning script')
    parser.add_argument('data',
                        metavar='DIR',
                        default='/imagenet-py',
                        help='Path to ImageNet dataset')
    parser.add_argument('--arch',
                        default='vgg16_bn',
                        choices=model_names,
                        help='Model architecture: ' + ' | '.join(model_names) +
                        ' (default: alexnet)')
    parser.add_argument('--model', help='Pretrained model filename')
    parser.add_argument('--workers',
                        default=8,
                        type=int,
                        metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--epochs',
                        type=int,
                        help='Number of fine-tuning epochs')
    parser.add_argument('--batch_size',
                        type=int,
                        default=256,
                        help='Batch size for training')
    parser.add_argument('--val_batch_size',
                        type=int,
                        default=256,
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

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    if (args.arch == "inception_v3"):
        raise RuntimeError(
            "Currently, inception_v3 is not supported by this example.")
    else:
        crop_size = 224
        val_size = 256

    trainset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
        ]))
    valset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(val_size),
            transforms.CenterCrop(crop_size),
        ]))

    trainloader = condensa.data.GPUDataLoader(trainset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.workers,
                                              meanstd=((0.485, 0.456, 0.406),
                                                       (0.229, 0.224, 0.225)))
    valloader = condensa.data.GPUDataLoader(valset,
                                            batch_size=args.val_batch_size,
                                            shuffle=False,
                                            num_workers=args.workers,
                                            meanstd=((0.485, 0.456, 0.406),
                                                     (0.229, 0.224, 0.225)))

    # Load pre-trained model
    model = models.__dict__[args.arch](pretrained=False)
    model.load_state_dict(torch.load(args.model))

    # Compute #nonzeros prior to fine-tuning
    nparams_w = torch.nn.utils.parameters_to_vector(
        model.parameters()).view(-1).nonzero().numel()

    # Only fine-tune fully-connected and convolutional layers
    layer_types = [torch.nn.Linear, torch.nn.Conv2d]

    criterion = torch.nn.CrossEntropyLoss().cuda()
    ft = condensa.FineTuner(model, layer_types, distributed=True)
    w_ft = ft.run(epochs=args.epochs,
                  lr=args.lr,
                  lr_end=args.lr_end,
                  momentum=args.momentum,
                  weight_decay=args.weight_decay,
                  criterion=criterion,
                  trainloader=trainloader,
                  testloader=None,
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
