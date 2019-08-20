# A Programming System for Model Compression

Condensa is a framework for _programmable model compression_ in Python.
It comes with a set of built-in compression operators which may be used to
compose complex compression schemes targeting specific combinations of DNN,
hardware platform, and optimization objective.
Common programming abstractions such as conditionals, iteration, and
recursion are all natively supported.
To recover any accuracy lost during compression, Condensa uses a constrained
optimization formulation of model compression and employs an Augmented Lagrangian-based
algorithm as the optimizer.

**Status**: Condensa is under active development, and bug reports, pull requests, and other feedback are all highly appreciated. See the contributions section below for more details on how to contribute.

## Supported Operators and Schemes

Condensa provides the following set of pre-built compression schemes:

* [Unstructured Pruning](https://nvlabs.github.io/condensa/modules/schemes.html#unstructured-pruning)
* [Filter and Neuron Pruning](https://nvlabs.github.io/condensa/modules/schemes.html#neuron-pruning)
* [Block Pruning](https://nvlabs.github.io/condensa/modules/schemes.html#block-pruning)
* [Quantization](https://nvlabs.github.io/condensa/modules/schemes.html#quantization)
* [Scheme Composition](https://nvlabs.github.io/condensa/modules/schemes.html#composition)

The schemes above are built using one or more [compression operators](https://nvlabs.github.io/condensa/modules/pi.html), which may be combined in various ways to define your own custom schemes.

Please refer to the [documentation](https://nvlabs.github.io/condensa/index.html) for a detailed description of available operators and schemes.

## Prerequisites

Condensa requires:

* A working Linux installation (we use Ubuntu 18.04)
* NVIDIA drivers and CUDA 10+ for GPU support
* Python 3.5 or newer
* PyTorch 1.0 or newer

## Installation

Retrieve the latest source code from the Condensa repository:

```bash
git clone https://github.com/NVlabs/condensa.git
```

Navigate to the source code directory and run the following:

```bash
pip install -r requirements.txt
```

To check the installation, run the unit test suite:

```bash
bash run_all_tests.sh -v
```

## Getting Started

The [MNIST LeNet5 Notebook](https://github.com/NVlabs/condensa/blob/master/notebooks/LeNet5.ipynb) contains a simple step-by-step walkthrough of compressing a pre-trained model using Condensa. Check out the `examples/` folder for additional, more complex examples of using Condensa.

## Documentation

Documentation is available [here](https://nvlabs.github.io/condensa/). We'll soon be releasing a paper with a detailed
description of Condensa's motivation, features, and performance results.

## Contributing

We appreciate all contributions, including bug fixes, new features and documentation, and additional tutorials. You can initiate
contributions via Github pull requests. When making code contributions, please follow the `PEP 8` Python coding standard and provide
unit tests for the new features. Finally, make sure to sign off your commits using the `-s` flag or adding 
`Signed-off-By: Name<Email>` in the commit message.

## Citing Condensa

If you use Condensa for research, please consider citing the following paper:

```
@article{condensa2019,
    title = {A Programming System for Model Compression},
    author = {Joseph, Vinu and Muralidharan, Saurav and Garg, Animesh and Garland, Michael},
    journal = {CoRR},
    volume = {}
    year = {2019},
    url = {}
}
```

## Disclaimer

Condensa is a research prototype and not an official NVIDIA product. Many features are still experimental and yet to be properly documented.
