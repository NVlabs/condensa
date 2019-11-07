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

from condensa import dtypes

def dequantize(module, dtype):
    """
    De-quantizes module to given data type (inplace).

    :param module: PyTorch module.
    :type module: `torch.nn.Module`
    :param dtype: Target data type.
    """
    if dtype.as_dtype_enum == dtypes.DT_FLOAT32:
        module.float()
    elif dtype.as_dtype_enum == dtypes.DT_FLOAT64:
        module.double()
    else:
        raise TypeError('Unknown data type specified for de-quantization')
