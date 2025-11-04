from jax import random as jrandom
from jax._src.pjit import JitWrapped
import os
from functools import partial
from typing import NamedTuple, Tuple
import jax.lax
import jax.numpy as jnp
import chex

import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.environment import (
    JaxEnvironment,
    JAXAtariAction as Action,
    EnvObs,
    EnvState,
    EnvInfo,
)
from jaxatari.spaces import Space


class DefenderConstants(NamedTuple):
    MAX_SPEED: int = 12
    ENEMY_STEP_SIZE: int = 2
    WIDTH: int = 160
    HEIGHT: int = 210
    PLAYER_ACCELERATION: chex.Array = jnp.array(
        [6, 3, 1, -1, 1, -1, 0, 0, 1, 0, -1, 0, 1]
    )
    BACKGROUND_COLOR: Tuple[int, int, int] = (144, 72, 17)
    PLAYER_COLOR: Tuple[int, int, int] = (92, 186, 92)
    SCORE_COLOR: Tuple[int, int, int] = (236, 236, 236)
    PLAYER_X: int = 140
    PLAYER_Y: int = 80
    PLAYER_SIZE: Tuple[int, int] = (16, 16)
    WALL_TOP_Y: int = 24
    WALL_TOP_HEIGHT: int = 10
    WALL_BOTTOM_Y: int = 194
    WALL_BOTTOM_HEIGHT: int = 16


# immutable state container
class DefenderState(NamedTuple):
    player_speed: chex.Array
    score: chex.Array
    step_counter: chex.Array
    acceleration_counter: chex.Array
    buffer: chex.Array


class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray


class DefenderObservation(NamedTuple):
    player: EntityPosition
    score: jnp.ndarray


class DefenderInfo(NamedTuple):
    time: jnp.ndarray


class JaxDefender(
    JaxEnvironment[DefenderState, DefenderObservation, DefenderInfo, DefenderConstants]
):

    def __init__(self, consts: DefenderConstants = None):
        consts = consts or DefenderConstants()
        super().__init__(consts)
        self.renderer = DefenderRenderer(self.consts)
        self.action_set = [
            Action.NOOP,
            Action.FIRE,
            Action.UP,
            Action.RIGHT,
            Action.LEFT,
            Action.DOWN,
            Action.UPRIGHT,
            Action.UPLEFT,
            Action.DOWNRIGHT,
            Action.DOWNLEFT,
            Action.UPFIRE,
            Action.RIGHTFIRE,
            Action.LEFTFIRE,
            Action.DOWNFIRE,
            Action.UPRIGHTFIRE,
            Action.UPLEFTFIRE,
            Action.DOWNRIGHTFIRE,
            Action.DOWNLEFTFIRE,
        ]

    def _player_step(self, state: DefenderState, action: chex.Array) -> DefenderState:
        up = action == Action.UP
        down = action == Action.DOWN
        left = action == Action.LEFT
        right = action == Action.RIGHT

        # touches wall here

        player_speed = state.player_speed

        # resume from pong here

        pass

    def reset(self, key=None) -> Tuple[DefenderObservation, DefenderState]:
        pass

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key: jrandom.PRNGKey = None) -> DefenderState:

        pass

    def render(self, state: DefenderState) -> Tuple[DefenderObservation, DefenderState]:
        pass

    def action_space(self) -> spaces.Space:
        pass

    def observation_space(self) -> spaces.Space:
        pass

    def image_space(self) -> Space:
        pass

    def _get_observation(self, state: DefenderState) -> DefenderObservation:
        pass

    def observation_spaces(self) -> spaces.Space:
        pass

    def _get_info(
        self, state: DefenderState, all_rewards: jnp.array = None
    ) -> DefenderInfo:
        pass

    def _get_reward(self, previous_state: DefenderState, state: DefenderState) -> float:
        pass

    def _get_done(self, state: DefenderState) -> bool:
        pass
