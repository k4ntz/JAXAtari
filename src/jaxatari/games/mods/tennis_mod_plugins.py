import functools
from typing import Any, Dict, Tuple, Union


import chex
import jax
import jax.numpy as jnp
from jax import lax
from jaxatari.games.jax_tennis import TennisState

from jaxatari.modification import JaxAtariPostStepModPlugin, JaxAtariInternalModPlugin

class RandomBallSpeedWrapper(JaxAtariPostStepModPlugin):
    """Ball has random speed after every hit."""
    @functools.partial(jax.jit, static_argnums=(0,))
    def make_random(self, prev_state: TennisState, state: TennisState) -> TennisState:
        was_ball_hit = jnp.logical_and(
            prev_state.ball_state.ball_x == prev_state.ball_state.ball_hit_start_x,
            prev_state.ball_state.ball_y == prev_state.ball_state.ball_hit_start_y,
        )

        key, subkey = jax.random.split(state.random_key)
        ball_speed_modifier = jax.random.uniform(subkey, shape=()) + 0.5

        new_move_x = state.ball_state.move_x * ball_speed_modifier
        new_move_y = state.ball_state.move_y * ball_speed_modifier

        new_state = state.replace(
            ball_state = state.ball_state.replace(
                move_x=new_move_x,
                move_y=new_move_y,
            ),
            random_key = key,
        )

        return lax.cond(
            was_ball_hit,
            lambda _: new_state,
            lambda _: state,
            operand=None
        )

    def run(self, prev_state: TennisState, new_state: TennisState) -> TennisState:
        return self.make_random(prev_state, new_state)

class RandomWalkSpeedWrapper(JaxAtariPostStepModPlugin):
    """Player has random walk speed everytime they start moving."""
    @functools.partial(jax.jit, static_argnums=(0,))
    def make_random(self, prev_state: TennisState, state: TennisState) -> TennisState:
        diff_x = state.player_state.player_x - prev_state.player_state.player_x
        diff_y = state.player_state.player_y - prev_state.player_state.player_y

        did_not_walk = jnp.logical_and(diff_x <= 0, diff_y <= 0)

        key, subkey = jax.random.split(state.random_key)
        new_walk_speed = jax.random.uniform(subkey, shape=()) + 0.5

        new_state = state.replace(
            player_state = state.player_state.replace(
                player_walk_speed=new_walk_speed,
            ),
            random_key = key,
        )

        return lax.cond(
            did_not_walk,
            lambda _: new_state,
            lambda _: state,
            operand=None
        )

    def run(self, prev_state: TennisState, new_state: TennisState) -> TennisState:
        return self.make_random(prev_state, new_state)