Basic Usage
===========

Environment Creation
--------------------

The main entry point is the ``make()`` function:

.. code-block:: python

   import jax
   import jaxatari

   # Create an environment
   env = jaxatari.make("pong")  # or "seaquest", "kangaroo", "freeway", etc.

   # Get available games
   available_games = jaxatari.list_available_games()
   print(f"Available games: {available_games}")

Using Modifications
-------------------

JAXAtari provides pre-implemented game modifications to test agent generalization:

.. code-block:: python

   import jaxatari

   # Pong environment with the lazy_enemy mod
   mod_env = jaxatari.make("pong", mods=["lazy_enemy"])

   # Multiple mods can be applied simultaneously
   mod_env = jaxatari.make("pong", mods=["lazy_enemy", "shift_enemy"])

Custom modifications are well supported via the ``JaxAtariModController``.
Feel free to share them by opening a PR.

Using Wrappers
--------------

JAXAtari provides a comprehensive wrapper system for different observation types:

.. code-block:: python

   import jaxatari
   from jaxatari.wrappers import (
       AtariWrapper,
       ObjectCentricWrapper,
       PixelObsWrapper,
       PixelAndObjectCentricWrapper,
       FlattenObservationWrapper,
       LogWrapper,
   )

   base_env = jaxatari.make("pong")
   atari_env = AtariWrapper(base_env)

   env = ObjectCentricWrapper(atari_env)          # object-centric features
   # OR
   env = PixelObsWrapper(atari_env)               # pixel observations
   # OR
   env = PixelAndObjectCentricWrapper(atari_env)  # both
   # OR
   env = FlattenObservationWrapper(ObjectCentricWrapper(atari_env))  # flattened

   # Add logging wrapper for training
   env = LogWrapper(env)

Vectorized Stepping
-------------------

JAXAtari is designed for massive parallelization via ``jax.vmap`` and ``jax.lax.scan``:

.. code-block:: python

   import jax
   import jaxatari
   from jaxatari.wrappers import AtariWrapper, ObjectCentricWrapper, FlattenObservationWrapper

   base_env = jaxatari.make("pong")
   env = FlattenObservationWrapper(ObjectCentricWrapper(AtariWrapper(base_env)))

   n_envs = 1024
   rng = jax.random.PRNGKey(0)
   reset_keys = jax.random.split(rng, n_envs)

   # Initialize n_envs parallel environments
   init_obs, env_state = jax.vmap(env.reset)(reset_keys)

   # Take one random step in each env
   action = jax.random.randint(rng, (n_envs,), 0, env.action_space().n)
   new_obs, new_env_state, reward, terminated, truncated, info = jax.vmap(env.step)(env_state, action)

   # Take 100 steps with scan
   def step_fn(carry, unused):
       _, env_state = carry
       new_obs, new_env_state, reward, terminated, truncated, info = jax.vmap(env.step)(env_state, action)
       return (new_obs, new_env_state), (reward, terminated, truncated, info)

   carry = (init_obs, env_state)
   _, (rewards, terminations, truncations, infos) = jax.lax.scan(
       step_fn, carry, None, length=100
   )

Manual Play
-----------

To play a game manually with keyboard input, install ``pygame`` and use the provided script:

.. code-block:: bash

   pip install pygame
   python3 scripts/play.py -g Pong