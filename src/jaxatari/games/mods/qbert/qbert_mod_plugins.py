import jax
import jax.numpy as jnp
from functools import partial
from jaxatari.modification import JaxAtariPostStepModPlugin

class NoRedBallsMod(JaxAtariPostStepModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        return new_state.replace(red_ball_positions=jnp.full((3, 2), -1, dtype=jnp.int32))

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        new_state = self.run(state, state)
        new_obs = self._env._get_observation(new_state)
        return new_obs, new_state

class NoPurpleBallMod(JaxAtariPostStepModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        return new_state.replace(purple_ball_position=jnp.array([-1, -1], dtype=jnp.int32))

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        new_state = self.run(state, state)
        new_obs = self._env._get_observation(new_state)
        return new_obs, new_state

class NoCoilyMod(JaxAtariPostStepModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        return new_state.replace(snake_position=jnp.array([-1, -1], dtype=jnp.int32))

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        new_state = self.run(state, state)
        new_obs = self._env._get_observation(new_state)
        return new_obs, new_state

class NoGreenBallMod(JaxAtariPostStepModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        return new_state.replace(green_ball_position=jnp.array([-1, -1], dtype=jnp.int32))

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        new_state = self.run(state, state)
        new_obs = self._env._get_observation(new_state)
        return new_obs, new_state

class NoSamMod(JaxAtariPostStepModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        return new_state.replace(sam_position=jnp.array([-1, -1], dtype=jnp.int32))

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        new_state = self.run(state, state)
        new_obs = self._env._get_observation(new_state)
        return new_obs, new_state

class NoEnemiesMod(JaxAtariPostStepModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        return new_state.replace(
            red_ball_positions=jnp.full((3, 2), -1, dtype=jnp.int32),
            purple_ball_position=jnp.array([-1, -1], dtype=jnp.int32),
            snake_position=jnp.array([-1, -1], dtype=jnp.int32),
            green_ball_position=jnp.array([-1, -1], dtype=jnp.int32),
            sam_position=jnp.array([-1, -1], dtype=jnp.int32)
        )

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        new_state = self.run(state, state)
        new_obs = self._env._get_observation(new_state)
        return new_obs, new_state
