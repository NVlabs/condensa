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

import numpy as np

from .type_enums import *

class DType(object):
    """Data type for quantization."""
    def __init__(self, dtype):
        self._dtype = dtype

    @property
    def name(self):
        return _DTYPE_TO_STRING[self._dtype]

    @property
    def as_numpy_dtype(self):
        return _TO_NP[self._dtype]

    @property
    def as_dtype_enum(self):
        return self._dtype

    def __int__(self):
        return self._dtype

    def __str__(self):
        return "<dtype: %r>" % self.name

float16 = DType(DT_FLOAT16)
float32 = DType(DT_FLOAT32)
float64 = DType(DT_FLOAT64)
int8 = DType(DT_INT8)
uint8 = DType(DT_UINT8)
int16 = DType(DT_INT16)
uint16 = DType(DT_UINT16)

_DTYPE_TO_STRING = {
    DT_FLOAT16: "float16",
    DT_FLOAT32: "float32",
    DT_FLOAT64: "float64",
    DT_INT8: "int8",
    DT_UINT8: "uint8",
    DT_INT16: "int16",
    DT_UINT16: "uint16"
}

_TO_NP = {
    DT_FLOAT16: np.float16,
    DT_FLOAT32: np.float32,
    DT_FLOAT64: np.float64,
    DT_INT8: np.int8,
    DT_UINT8: np.uint8,
    DT_INT16: np.int16,
    DT_UINT16: np.uint16
}
