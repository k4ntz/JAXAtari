Core
=============

The `core.py` module provides a user-friendly entry point to the JAXtari environment framework.
Similar to the interface of OCAtari, it abstracts away low-level configuration details so you can get started quickly with just a few lines of code.

Here’s a minimal example:

.. code-block:: python

    import jaxtari

    env = jaxtari.make("pong")
    print(jaxtari.list_available_games())

.. automodule:: jaxtari.core
   :members:
   :show-inheritance:
   :undoc-members:
