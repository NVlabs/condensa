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

from . import cfg

def add_mask_to_module(module, parameter, mask):
    if cfg.__CONDENSA_SAVE_MASKS__:
        maskattr = f'__condensa_mask_{parameter}'
        if hasattr(module, maskattr):
            modmask = getattr(module, maskattr)
            if not isinstance(modmask, list):
                modmask = [modmask]
            modmask.append(mask)
            setattr(module, maskattr, modmask)
        else:
            setattr(module, maskattr, mask)