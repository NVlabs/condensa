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

__CONDENSA_PI_PRECHECK__ = True
__CONDENSA_SAVE_MASKS__  = False

class save_masks(object):
    def __enter__(self):
        global __CONDENSA_SAVE_MASKS__
        __CONDENSA_SAVE_MASKS__ = True
        return True

    def __exit__(self, *args):
        global __CONDENSA_SAVE_MASKS__
        __CONDENSA_SAVE_MASKS__ = False
        return False
