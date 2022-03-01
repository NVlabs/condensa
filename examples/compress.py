# Copyright 2022 NVIDIA Corporation
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

import argparse
import logging
import json
import re

import torch
import torch.nn

import condensa
import condensa.functional as F
import condensa.tensor as T
from   condensa.schemes import (
    LayerBlockPruner,
    LayerPruner,
    NetworkPruner
)


logger = logging.getLogger(__name__)


def nparams(model, modules):
    nparams = 0
    for name, m in model.named_modules():
        if name in modules:
            nparams += m.weight.data.view(-1).nonzero().numel()
    return nparams


def nparams_full(model):
    return torch.nn.utils.parameters_to_vector(
           model.parameters()).view(-1).nonzero().numel()


def main():
    networks = ['bert-base',
                'bert-large',
                'transformer',
                'transformer-xl',
                'resnet50']
    parser = argparse.ArgumentParser()
    parser.add_argument('scheme',
                        type=str,
                        help='Compression specification')
    parser.add_argument('--pretrained',
                        action='store_true',
                        help='Load pretrained weights (default: False)')
    parser.add_argument('--network',
                        choices=networks,
                        type=str.lower,
                        required=True,
                        help='Network')
    parser.add_argument('--out',
                        type=str,
                        default='compressed.pt',
                        help='Compressed model')
    parser.add_argument('--quiet',
                        action='store_true',
                        help='Suppress verbose logging messages')
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.WARNING if args.quiet else logging.INFO)

    print(args)
    try:
        from transformers import BertModel, BertConfig
        from transformers import TransfoXLModel, TransfoXLConfig
        import torchvision.models as models
    except ImportError:
        print('Please install Torchvision and HuggingFace Transformers libraries')

    hfnets = {
        'bert-base': ['bert-base-uncased', BertConfig, BertModel],
        'bert-large': ['bert-large-uncased', BertConfig, BertModel],
        'transformer-xl': ['transfo-xl-wt103', TransfoXLConfig, TransfoXLModel]
    }

    if args.network == 'resnet50':
        model = models.resnet50(pretrained=args.pretrained, progress=True)
    elif args.network == 'transformer':
        logger.info('Loading pretrained Transformer model from Torch Hub')
        torch.hub.list('pytorch/fairseq')
        en2de = torch.hub.load('pytorch/fairseq',
                               'transformer.wmt16.en-de',
                               tokenizer='moses',
                               bpe='subword_nmt')
        # Disable dropout
        en2de.eval()
        model = en2de.models[0]
    elif args.network in ['bert-base', 'bert-large', 'transformer-xl']:
        _name, _config, _model = hfnets[args.network]
        if args.pretrained:
            model = _model.from_pretrained(_name)
        else:
            config = _config.from_pretrained(_name)
            model = _model(config)
    else:
        raise RuntimeError(f'Unknown network: {args.network}')

    logger.info('Finished loading model')
    #print(model)

    modules = []
    if args.network in ['bert-base', 'bert-large']:
        for name, m in model.named_modules():
            if name == 'pooler.dense':
                continue
            if isinstance(m, torch.nn.Linear):
                modules.append(name)
    elif args.network == 'transformer-xl':
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.Linear):
                modules.append(name)
    elif args.network == 'resnet50':
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
                modules.append(name)
    elif args.network == 'transformer':
        for name, m in model.named_modules():
            if name == 'decoder.output_projection':
                continue
            if isinstance(m, torch.nn.Linear):
                modules.append(name)
    else:
        raise RuntimeError(f'Unknown network: {args.network}')

    npo, npo_full = nparams(model, modules), nparams_full(model)

    scheme = condensa.schemes.parse(args.scheme, modules)
    with condensa.save_masks():
        scheme.pi(model)
    logger.info('Done with compression, masks saved')

    npc, npc_full = nparams(model, modules), nparams_full(model)
    cratio = float(npo) / float(npc)
    cratio_full = float(npo_full) / float(npc_full)
    print(f'compression ratio (effective,full): ({cratio},{cratio_full})')
    quit()

    torch.save(model.state_dict(), args.out)
    logger.info('Compressed model saved to disk')


if __name__ == '__main__':
    main()
