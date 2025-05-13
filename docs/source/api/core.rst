Core
=============

The `core.py` module provides a user-friendly entry point to the JAXAtari environment framework.  
Similar to the interface of OCAtari, it abstracts away low-level configuration details so you can get started quickly with just a few lines of code.

Here’s a minimal example:

.. code-block:: python

    from jaxatari import JAXtari

    env = JAXtari("pong")
    state = env.get_init_state()
    state = env.step_state_only(state, action=0)

.. automodule:: jaxatari.core
   :members:
   :undoc-members:
   :show-inheritance:
