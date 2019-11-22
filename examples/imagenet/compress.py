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
import sys
import argparse
import logging
import csv
import warnings

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
from condensa import schemes

if __name__ == '__main__':
    # Suppress EXIF warning messages
    warnings.filterwarnings("ignore",
        "(Possibly )?corrupt EXIF data", UserWarning)
    model_names = sorted(
        name for name in models.__dict__
        if not name.startswith("__") and callable(models.__dict__[name]))
    valid_schemes = ['PRUNE', 'PQ', 'FILTER']

    parser = argparse.ArgumentParser(
        description='ImageNet LC Compression Script')
    parser.add_argument('data',
                        metavar='DIR',
                        default='/imagenet-py',
                        help='Path to ImageNet dataset')
    parser.add_argument('--arch',
                        default='resnet50',
                        choices=model_names,
                        required=True,
                        help='Model architecture: ' + ' | '.join(model_names) +
                        ' (default: resnet)')
    parser.add_argument('--workers',
                        default=8,
                        type=int,
                        metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--steps',
                        type=int,
                        default=35,
                        help='Number of LC iterations')
    parser.add_argument('--scheme',
                        choices=valid_schemes,
                        required=True,
                        help='Compression scheme')
    parser.add_argument('--density',
                        required=True,
                        type=float,
                        help='Density for pruning')
    parser.add_argument('--l_batch_size',
                        type=int,
                        default=256,
                        help='Batch size for L step')
    parser.add_argument('--val_batch_size',
                        type=int,
                        default=256,
                        help='Validation batch size')
    parser.add_argument('--lr',
                        type=float,
                        default=0.02,
                        help='Initial learning rate')
    parser.add_argument('--lr_end',
                        type=float,
                        default=None,
                        help='Ending learning rate')
    parser.add_argument('--lr_decay',
                        type=float,
                        default=None,
                        help='Learning rate decay')
    parser.add_argument('--lr_schedule',
                        type=int,
                        nargs='+',
                        default=None,
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--lr_multiplier',
                        type=float,
                        default=None,
                        help='Learning rate multiplier')
    parser.add_argument('--momentum',
                        type=float,
                        default=0.9,
                        help='SGD momentum')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0,
                        help='SGD momentum')
    parser.add_argument('--mb_iterations_per_l',
                        type=int,
                        default=2000,
                        help='Minibatch iterations per L step')
    parser.add_argument('--mb_iterations_first_l',
                        type=int,
                        default=10000,
                        help='Minibatch iterations for first L step')
    parser.add_argument('--mu_init',
                        type=float,
                        default=0.001,
                        help='Initial value of mu')
    parser.add_argument('--mu_multiplier', type=float, help='mu multiplier')
    parser.add_argument('--mu_cap', type=float, default=10000, help='mu cap')
    parser.add_argument('--out',
                        default='compressed_model.pth',
                        help='Compressed output model filename')
    parser.add_argument('--csv',
                        default=None,
                        help='compression statistics CSV file')
    parser.add_argument('-v',
                        '--verbose',
                        help='verbose logging output',
                        action='store_true')

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(message)s')

    # Load pre-trained model
    model = models.__dict__[args.arch](pretrained=True)

    if args.scheme == 'PRUNE':
        scheme = schemes.Prune(args.density)
    elif args.scheme == 'PQ':
        scheme = schemes.Compose(
            [schemes.Prune(args.density),
             schemes.Quantize()])
    elif args.scheme == 'FILTER':
        scheme = schemes.FilterPrune(args.density)
    else:
        raise RuntimeError('Unknown scheme: {}'.format(args.scheme))

    print('SCHEME: {}'.format(scheme))

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
                                              batch_size=args.l_batch_size,
                                              shuffle=True,
                                              num_workers=8,
                                              meanstd=((0.485, 0.456, 0.406),
                                                       (0.229, 0.224, 0.225)))
    valloader = condensa.data.GPUDataLoader(valset,
                                            batch_size=args.val_batch_size,
                                            shuffle=False,
                                            num_workers=8,
                                            meanstd=((0.485, 0.456, 0.406),
                                                     (0.229, 0.224, 0.225)))

    # Instantiate LC optimizer
    sgd_params = {'momentum': args.momentum, 'weight_decay': args.weight_decay}
    lc = condensa.opt.LC(steps=args.steps,
                         l_optimizer=condensa.opt.lc.SGD,
                         l_optimizer_params=sgd_params,
                         lr=args.lr,
                         lr_end=args.lr_end,
                         lr_decay=args.lr_decay,
                         lr_schedule=args.lr_schedule,
                         lr_multiplier=args.lr_multiplier,
                         mb_iterations_per_l=args.mb_iterations_per_l,
                         mb_iterations_first_l=args.mb_iterations_first_l,
                         mu_init=args.mu_init,
                         mu_multiplier=args.mu_multiplier,
                         mu_cap=args.mu_cap,
                         distributed=True,
                         debugging_flags={'custom_model_statistics':
                                           condensa.util.cnn_statistics,
                                           'disable_train_stats': True})

    criterion = nn.CrossEntropyLoss().cuda()
    # Compress model using Condensa
    compressor = condensa.Compressor(lc, scheme, model, trainloader, None,
                                     valloader, criterion)
    w = compressor.run()

    if args.out is not None:
        torch.save(w.state_dict(), args.out)
        logging.info('[Condensa] Compressed model written to disk')

    print('\n==== Profiling Results ====')
    for k, v in compressor.statistics.items():
        print('  ' + k + ':', v)
    print('')

    if args.csv is not None:
        with open(args.csv, 'w') as csv_file:
            writer = csv.writer(csv_file)
            for k, v in compressor.statistics.items():
                row = [k]
                if isinstance(v, list): row += [str(x) for x in v]
                else: row.append(str(v))
                writer.writerow(row)
        csv_file.close()
        logging.info('[Condensa] Compression stats written to disk')
