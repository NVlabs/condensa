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
from setuptools import find_packages

cwd = os.path.dirname(os.path.abspath(__file__))
version = '0.5.0-beta'

def build_deps():
  version_path = os.path.join(cwd, 'condensa', 'version.py')
  with open(version_path, 'w') as f:
    f.write("__version__ = '{}'\n".format(version))

build_deps()

with open(os.path.join(cwd, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

install_requires = ['numpy',
                    'torch>=1.0.0',
                    'tqdm']

setup(name='condensa',
      version=version,
      description='Condensa Programmable Model Compression Framework',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/NVLabs/condensa',
      author='Saurav Muralidharan',
      author_email='sauravm@nvidia.com',
      license='Apache License 2.0',
      keywords=['compression', 'quantization', 'pruning'],
      install_requires=install_requires,
      packages=find_packages(),
      classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development :: Libraries',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
      ],
      )
