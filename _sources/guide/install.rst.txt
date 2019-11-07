Installation
============

Prerequisites
-------------

Condensa requires:

* A working Linux installation (we use Ubuntu 18.04)
* NVIDIA drivers and CUDA 10+ for GPU support
* Python 3.5 or newer
* PyTorch 1.0 or newer

Installation with pip
---------------------

The most straightforward way of installing Condensa is via `pip`:

.. code-block:: bash

    pip install condensa

Installation from Source
------------------------

Retrieve the latest source code from the Condensa repository:

.. code-block:: bash

   git clone https://github.com/NVlabs/condensa.git

Navigate to the source code directory and run the following:

.. code-block:: bash

   pip install -e .

To check the installation, run the unit test suite:

.. code-block:: bash

   bash run_all_tests.sh -v
