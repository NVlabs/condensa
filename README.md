# A Programming System for Model Compression

This repository contains **Condensa Lite**, a stripped-down version of Condensa that supports pruning and learning-rate rewinding.

**Status**: Condensa Lite is under active development, and bug reports, pull requests, and other feedback are all highly appreciated. See the contributions section below for more details on how to contribute.

## Prerequisites

Condensa Lite requires:

* A working Linux installation (we use Ubuntu 20.04)
* NVIDIA drivers and CUDA 11+ for GPU support
* Python 3.6 or newer
* PyTorch 1.8 or newer

## Installation

Retrieve the latest source code from the Condensa repository:

```bash
git clone --branch lite https://github.com/NVlabs/condensa.git
```

Navigate to the source code directory and run the following:

```bash
pip install -e .
```

### Test out the Installation

To check the installation, run the unit test suite:

```bash
bash run_all_tests.sh -v
```

## Documentation

Documentation for Condensa Lite is available [here](https://nvlabs.github.io/condensa/lite). Please also check out the original [Condensa paper](https://arxiv.org/abs/1911.02497) for a detailed
description of Condensa's motivation, features, and performance results.

## Contributing

We appreciate all contributions, including bug fixes, new features and documentation, and additional tutorials. You can initiate
contributions via Github pull requests. When making code contributions, please follow the `PEP 8` Python coding standard and provide
unit tests for the new features. Finally, make sure to sign off your commits using the `-s` flag or adding 
`Signed-off-By: Name<Email>` in the commit message.

## Citing Condensa Lite

If you use Condensa Lite for research, please consider citing the following paper:

```
@article{condensa:2020,
    title={A Programmable Approach to Neural Network Compression},
    author={Joseph, Vinu and Gopalakrishnan, Ganesh L
            and Muralidharan, Saurav and Garland, Michael and Garg, Animesh},
    journal={IEEE Micro},
    volume={40},
    number={5},
    pages={17--25},
    year={2020},
    publisher={IEEE}
}
```

## Disclaimer

Condensa Lite is a research prototype and not an official NVIDIA product. Many features are still experimental and yet to be properly documented.
