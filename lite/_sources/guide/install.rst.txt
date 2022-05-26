Installation
============

Prerequisites
-------------

Condensa Lite requires:

* A working Linux installation (we use Ubuntu 20.04)
* NVIDIA drivers and CUDA 11+ for GPU support
* Python 3.6 or newer
* PyTorch 1.8 or newer

Installation from Source
------------------------

Retrieve the latest source code from the Condensa repository:

.. code-block:: bash

   git clone --branch lite https://github.com/NVlabs/condensa.git

Navigate to the source code directory and run the following:

.. code-block:: bash

   pip install -e .

Test out the Installation
-------------------------

To check the installation, run the unit test suite:

.. code-block:: bash

   bash run_all_tests.sh -v
