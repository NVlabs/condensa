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

import json
import re

import condensa.functional as F

from . import (
    LayerPruner,
    LayerFilterPruner,
    LayerNeuronPruner,
    LayerBlockPruner,
    NetworkPruner
)

def get_layer_scheme(s):
    sname = s['type']
    criteria = getattr(F, s.get('criteria', 'scaled_l2norm'))

    if sname == 'prune':
        scheme = LayerPruner()
    elif sname == 'block':
        scheme = LayerBlockPruner(tuple(s['blocksize']), criteria)
    elif sname == 'filter':
        scheme = LayerFilterPruner(criteria)
    elif sname == 'neuron':
        scheme = LayerNeuronPruner(criteria)
    else:
        raise RuntimeError(f'Unknown scheme: {sname}')
    return scheme
    
def load_json_scheme(json_file):
    with open(json_file, 'r') as f:
        cspec = json.load(f)

    if '*' in cspec.keys():
        cspec['.*'] = cspec.pop('*')

    if '.*' not in cspec.keys():
        raise ValueError('Default or fallback value (*) '
                         'not found in specification')
    
    return cspec
    
def parse(json_file, modules):
    cspec = load_json_scheme(json_file)

    # Read global sparsity value (if any)
    sparsity = None
    if 'sparsity' in cspec.keys():
        sparsity = cspec['sparsity']

    ns = dict()
    for name in modules:
        for rx, schemes in cspec.items():
            if re.match(rx, name):
                for s in schemes:
                    if 'sparsity' in s:
                        raise NotImplementedError('Layer-specific sparsity '
                                                  'values not supported')
                    layerscheme = get_layer_scheme(s)
                    if name not in ns:
                        ns[name] = layerscheme
                    else:
                        if not isinstance(ns[name], list):
                            ns[name] = [ns[name]]
                        ns[name].append(layerscheme)
                break
    return NetworkPruner(sparsity, ns)