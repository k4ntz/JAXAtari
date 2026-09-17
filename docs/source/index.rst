.. JAXAtari documentation master file

JAXAtari
=====================================

JAXAtari is a GPU-accelerated, object-centric Atari environment framework built on `JAX <https://github.com/google/jax>`_, designed for fast and scalable reinforcement learning research. 
It reimplements classic Atari 2600 games natively in JAX, enabling up to 16,000x faster training speeds through just-in-time (JIT) compilation and massive GPU parallelization,and separates the details of game simulation from agent design.
Users can interact with environments through a flexible wrapper system supporting pixel, object-centric, and combined observations. JAXAtari extends the lineage of OCAtari and HackAtari by providing structured, object-level state representations alongside support for parameterized game modifications to test agent generalization. If you use JAXAtari in your research, we ask that you please cite the paper.

.. code-block:: python

   import jax
   import jaxatari
   from jaxatari.wrappers import AtariWrapper, ObjectCentricWrapper, LogWrapper

   # Create an environment
   env = jaxatari.make("pong")

   # Apply wrappers for object-centric observations and logging
   env = LogWrapper(ObjectCentricWrapper(AtariWrapper(env)))

   # Initialize the environment
   rng = jax.random.PRNGKey(42)
   obs, state = env.reset(rng)

   for _ in range(1000):
      rng, rng_act = jax.random.split(rng)

      # This is where you would insert your policy
      action = jax.random.randint(rng_act, (), 0, env.action_space().n)

      # Step through the environment
      # receiving the next observation, reward, done flag and info
      obs, state, reward, done, info = env.step(state, action)

      # If the episode has ended, reset to start a new one
      if done:
         obs, state = env.reset(rng)


.. toctree::
   :maxdepth: 1
   :caption: Introduction
   :hidden:

   introduction/installation
   introduction/usage
   introduction/environment
   introduction/Environments/index

.. toctree::
   :maxdepth: 2
   :caption: API
   :hidden:

   api/core
   api/wrappers
   api/spaces
   api/rendering
