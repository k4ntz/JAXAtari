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
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action, EnvObs, EnvState, EnvInfo
from jaxatari.spaces import Space


class DefenderConstants(NamedTuple):
    MAX_SPEED: int = 12
    BALL_SPEED: chex.Array = jnp.array([-1, 1])
    ENEMY_STEP_SIZE: int = 2
    WIDTH: int = 160
    HEIGHT: int = 210
    BASE_BALL_SPEED: int = 1
    BALL_MAX_SPEED: int = 4
    MIN_BALL_SPEED: int = 1
    PLAYER_ACCELERATION: chex.Array = jnp.array([6, 3, 1, -1, 1, -1, 0, 0, 1, 0, -1, 0, 1])
    BALL_START_X: chex.Array = jnp.array(78)
    BALL_START_Y: chex.Array = jnp.array(115)
    BACKGROUND_COLOR: Tuple[int, int, int] = (144, 72, 17)
    PLAYER_COLOR: Tuple[int, int, int] = (92, 186, 92)
    ENEMY_COLOR: Tuple[int, int, int] = (213, 130, 74)
    BALL_COLOR: Tuple[int, int, int] = (236, 236, 236)
    WALL_COLOR: Tuple[int, int, int] = (236, 236, 236)
    SCORE_COLOR: Tuple[int, int, int] = (236, 236, 236)
    PLAYER_X: int = 140
    ENEMY_X: int = 16
    PLAYER_SIZE: Tuple[int, int] = (4, 16)
    BALL_SIZE: Tuple[int, int] = (2, 4)
    ENEMY_SIZE: Tuple[int, int] = (4, 16)
    WALL_TOP_Y: int = 24
    WALL_TOP_HEIGHT: int = 10
    WALL_BOTTOM_Y: int = 194
    WALL_BOTTOM_HEIGHT: int = 16


# immutable state container
class DefenderState(NamedTuple):
    player_y: chex.Array
    player_speed: chex.Array
    ball_x: chex.Array
    ball_y: chex.Array
    enemy_y: chex.Array
    enemy_speed: chex.Array
    ball_vel_x: chex.Array
    ball_vel_y: chex.Array
    player_score: chex.Array
    enemy_score: chex.Array
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
    enemy: EntityPosition
    ball: EntityPosition
    score_player: jnp.ndarray
    score_enemy: jnp.ndarray


class DefenderInfo(NamedTuple):
    time: jnp.ndarray


class JaxDefender(JaxEnvironment[DefenderState, DefenderInfo, DefenderInfo, DefenderConstants]):

    def reset(self, key: jrandom.PRNGKey=None) -> Tuple[EnvObs, EnvState]:
        pass

    def step(self, key: jrandom.PRNGKey=None) -> DefenderState:
        pass

    def render(self, key: jrandom.PRNGKey=None) -> Tuple[EnvObs, EnvState]:
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

    def _get_info(self, state: EnvState, all_rewards: jnp.array = None) -> EnvInfo:
        pass

    def _get_reward(self, previous_state: EnvState, state: EnvState) -> float:
        pass

    def _get_done(self, state: EnvState) -> bool:
        pass