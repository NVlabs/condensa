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
from setuptools import setup

cwd = os.path.dirname(os.path.abspath(__file__))
version = '0.5-beta'

def build_deps():
  version_path = os.path.join(cwd, 'condensa', 'version.py')
  with open(version_path, 'w') as f:
    f.write("__version__ = '{}'\n".format(version))

build_deps()

setup(name='condensa',
      version=version,
      description='Condensa Programmable Model Compression Framework',
      url='https://nvlabs.github.io/condensa/',
      author='Saurav Muralidharan',
      author_email='sauravm@nvidia.com',
      license='Apache License 2.0',
      packages=['condensa'],
      zip_safe=False)
