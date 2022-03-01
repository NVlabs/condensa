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

from .types    import NetworkScheme, LayerScheme

from .compose  import NetworkPruner, SchemeComposer

from .prune    import LayerPruner
from .block    import LayerBlockPruner
from .filter   import LayerFilterPruner
from .neuron   import LayerNeuronPruner

from .json     import parse, get_layer_scheme, load_json_scheme