.. JAXAtari documentation master file

Welcome to JAXAtari's Documentation!
=====================================

**JAXAtari** is a GPU-accelerated, object-centric Atari environment framework built with `JAX <https://github.com/google/jax>`_.  
Inspired by OCAtari, it enables massively parallelized training for reinforcement learning research.

Built and maintained by students from `TU Darmstadt <https://www.ml.informatik.tu-darmstadt.de/>`_.

.. note::
   If you're looking for a quick start, head to the usage section below or browse the API reference.

----

Features
--------

- Object-centric extraction of Atari game states.
- JAX-based vectorized execution with GPU support.
- Compatible API with ALE (Arcade Learning Environment).
- Built-in benchmarking tools.
- Modular wrappers and utilities.

----


Getting Started
---------------


You can install and use JAXAtari as follows:

.. code-block:: bash

   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e .

To run a game manually:

.. code-block:: bash

   python -m jaxatari.games.jax_seaquest

----

.. toctree::
   :maxdepth: 2
   :caption: API
   :hidden:

   api/environment
   api/core
   api/wrappers
   api/rendering
   api/games/index
   
.. toctree::
   :maxdepth: 2
   :caption: Scripts
   :hidden:

   scripts/RAMStateDeltas
   scripts/FrameExtractor
   scripts/spriteEditor

.. toctree::
   :maxdepth: 1
   :caption: Tests & Benchmarks
   :hidden:

   tests/benchmarks


