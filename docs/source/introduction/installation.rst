Installation
============

Requirements
------------

- Python 3.10+
- JAX with GPU support (optional but recommended)

Install from source
-------------------

.. code-block:: bash

   git clone https://github.com/Farama-Foundation/JAXAtari.git
   cd JAXAtari
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e .

Verify installation
-------------------

.. code-block:: python

   import jaxatari
   from jaxatari import JAXtari

   env = JAXtari("pong")
   state = env.get_init_state()
   print("JAXAtari installed successfully!")

Run a game manually
-------------------

.. code-block:: bash

   python -m jaxatari.games.jax_seaquest
