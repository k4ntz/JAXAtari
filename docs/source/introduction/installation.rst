Installation
============

Install
-------

Create a virtual environment and install JAXAtari:

.. code-block:: bash

   python3 -m venv .venv
   source .venv/bin/activate

   python3 -m pip install -U pip
   pip3 install -e .

.. note::
   This will install JAX without GPU acceleration.

GPU Support (CUDA)
------------------

CUDA users should run the following to add GPU support:

.. code-block:: bash

   pip install -U "jax[cuda12]"

For other accelerator types, please follow the instructions in the
`JAX installation guide <https://docs.jax.dev/en/latest/installation.html>`_.

Sprite Assets
-------------

Next, you need to download the original Atari 2600 sprites. Before downloading,
you will be asked to confirm ownership of the original ROMs:

.. code-block:: bash

   .venv/bin/install_sprites