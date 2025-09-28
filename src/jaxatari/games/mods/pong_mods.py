import functools
from typing import Any, Dict, Tuple, Union


import chex
import jax
import jax.numpy as jnp
from jaxatari.games.jax_pong import PongState

from jaxatari.wrappers import JaxatariWrapper

class LazyEnemyWrapper(JaxatariWrapper):
    """Enemy stops moving after returning a shot."""
    @functools.partial(jax.jit, static_argnums=(0,))
    def make_lazy(self, prev_state: PongState, state: PongState) -> PongState:
        ball_moves_right = state.ball_x > prev_state.ball_x
        new_enemy_y = jnp.where(
            ball_moves_right,
            prev_state.enemy_y, # Keep the previous enemy's y position if the ball is moving right
            state.enemy_y  # Otherwise, use the current enemy's y position
        )
        new_enemy_speed = jnp.where(
            ball_moves_right,
            0,  # Stop the enemy's speed if the ball is moving right
            state.enemy_speed  # Otherwise, keep the current enemy's speed
        )

        new_state = state._replace(
            enemy_y = new_enemy_y,
            enemy_speed = new_enemy_speed,
        )

        return new_state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state: PongState, action: Union[int, float]) -> Tuple[chex.Array, PongState, float, bool, Dict[Any, Any]]:
        new_obs, new_state, reward, done, info = self._env.step(state, action)
        new_state = self.make_lazy(state, new_state)

        return new_obs, new_state, reward, done, info

class RandomizedEnemyWrapper(JaxatariWrapper):
    """Enemy stops moving after returning a shot."""
    @functools.partial(jax.jit, static_argnums=(0,))
    def make_random(self, prev_state: PongState, state: PongState) -> PongState:
        random_cond = state.step_counter % 3 == 0  # make other move every 3 steps
        new_enemy_y = jnp.where(
            random_cond,
            prev_state.enemy_y - (state.enemy_y - prev_state.enemy_y), # Move in the opposite direction than it would normally
            state.enemy_y  # Otherwise, use the current enemy's y position
        )
        new_enemy_speed = jnp.where(
            random_cond,
            -state.enemy_speed,  # Reverse the enemy's speed if the condition is met
            state.enemy_speed  # Otherwise, keep the current enemy's speed
        )

        new_state = state._replace(
            enemy_y = new_enemy_y,
            enemy_speed = new_enemy_speed,
        )

        return new_state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state: PongState, action: Union[int, float]) -> Tuple[chex.Array, PongState, float, bool, Dict[Any, Any]]:
        new_obs, new_state, reward, done, info = self._env.step(state, action)
        new_state = self.make_random(state, new_state)

        return new_obs, new_state, reward, done, info
