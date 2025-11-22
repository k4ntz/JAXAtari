import os
from enum import IntEnum
from functools import partial
from typing import Callable, NamedTuple, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import jax.test_util
from jax import Array
from numpy import int32

from jaxatari import spaces
from jaxatari.environment import JAXAtariAction as Action
from jaxatari.environment import JaxEnvironment
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils


class Direction(IntEnum):
    RIGHT = 0
    UPRIGHT = 1
    UP = 2
    UPLEFT = 3
    LEFT = 4
    DOWNLEFT = 5
    DOWN = 6
    DOWNRIGHT = 7

    @staticmethod
    def from_flags(up, down, left, right):
        return jnp.select(
            [
                right & ~up & ~down & ~left,
                right & up & ~down & ~left,
                ~right & up & ~down & ~left,
                ~right & up & ~down & left,
                ~right & ~up & ~down & left,
                ~right & ~up & down & left,
                ~right & ~up & down & ~left,
                right & ~up & down & ~left,
            ],
            [
                Direction.RIGHT,
                Direction.UPRIGHT,
                Direction.UP,
                Direction.UPLEFT,
                Direction.LEFT,
                Direction.DOWNLEFT,
                Direction.DOWN,
                Direction.DOWNRIGHT,
            ],
        )

    @staticmethod
    def to_flags(direction):
        up = (
            (direction == Direction.UP)
            | (direction == Direction.UPRIGHT)
            | (direction == Direction.UPLEFT)
        )

        down = (
            (direction == Direction.DOWN)
            | (direction == Direction.DOWNRIGHT)
            | (direction == Direction.DOWNLEFT)
        )

        left = (
            (direction == Direction.LEFT)
            | (direction == Direction.UPLEFT)
            | (direction == Direction.DOWNLEFT)
        )

        right = (
            (direction == Direction.RIGHT)
            | (direction == Direction.UPRIGHT)
            | (direction == Direction.DOWNRIGHT)
        )

        return up, down, left, right
    
    @staticmethod
    def reverse(direction: jnp.ndarray):
        return jnp.array((direction + 4) % 8).astype(jnp.int32)

class YarState(IntEnum):
    STEADY = 0
    MOVING = 1

class Entity(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    w: jnp.ndarray
    h: jnp.ndarray

class DirectionEntity(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    w: jnp.ndarray
    h: jnp.ndarray
    direction: jnp.ndarray


# Yars Revenge, Neutral Zone: https://www.youtube.com/watch?v=5HSjJU562e8


class YarsRevengeConstants(NamedTuple):
    # Constants for game environment
    WIDTH: int = 160
    HEIGHT: int = 210

    INITIAL_LIVES: int = 4

    # Entity sizes (width, height), horizontal orientation
    YAR_SIZE: Tuple[int, int] = (8, 16)  # Facing to right
    QOTILE_SIZE: Tuple[int, int] = (8, 18)
    NEUTRAL_ZONE_SIZE: Tuple[int, int] = (28, HEIGHT)
    SWIRL_SIZE: Tuple[int, int] = (8, 16)
    DESTROYER_SIZE: Tuple[int, int] = (4, 2)
    ENERGY_MISSILE_SIZE: Tuple[int, int] = (1, 2)

    QOTILE_MIN_Y: int = 64
    QOTILE_MAX_Y: int = 146
    # Entity Speeds, X pixel per 1 frame
    QOTILE_SPEED = 0.5
    YAR_SPEED = 2.0
    YAR_DIAGONAL_SPEED = 1.0
    SWIRL_SPEED = 2.0
    DESTROYER_SPEED = 0.125
    ENERGY_MISSILE_SPEED = 4.0

    STEADY_YAR_MOVEMENT_FRAME = (
        4  # Movement animation change interval for Yar (no action)
    )
    MOVING_YAR_MOVEMENT_FRAME = 1  # Movement animation change interval for Yar (moving)

    ENERGY_SHIELD_COLOR: Tuple[int, int, int] = (163, 57, 21)

    # Each energy cell has width = 4, height = 8
    ENERGY_CELL_WIDTH = 4
    ENERGY_CELL_HEIGHT = 8
    # (32, 128) is the end result of energy shield
    # construct a grid of 8 x 16 for energy shields
    INITIAL_ENERGY_SHIELD = jnp.array(
        [
            [
                [0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1],
            ],
            jnp.ones((16, 8)),
        ],
        dtype=jnp.bool,
    )


class YarsRevengeState(NamedTuple):
    step_counter: jnp.ndarray
    level: jnp.ndarray
    score: jnp.ndarray
    lives: jnp.ndarray

    yar: DirectionEntity
    yar_state: jnp.ndarray  # as YarState enum
    qotile: DirectionEntity
    destroyer: Entity

    swirl_exist: jnp.ndarray
    swirl_fired: jnp.ndarray
    swirl: DirectionEntity
    energy_missile_exist: jnp.ndarray
    energy_missile: DirectionEntity

    energy_shield: Entity
    energy_shield_state: jnp.ndarray


class YarsRevengeObservation(NamedTuple):
    yar: DirectionEntity
    qotile: DirectionEntity
    destroyer: Entity
    swirl_exist: jnp.ndarray
    swirl_fired: jnp.ndarray
    swirl: DirectionEntity
    energy_missile_exist: jnp.ndarray
    energy_missile: DirectionEntity
    energy_shield: Entity
    energy_shield_state: jnp.ndarray
    lives: chex.Array


class YarsRevengeInfo(NamedTuple):
    time: jnp.ndarray
    current_level: jnp.ndarray


class JaxYarsRevenge(
    JaxEnvironment[
        YarsRevengeState, YarsRevengeObservation, YarsRevengeInfo, YarsRevengeConstants
    ]
):

    def __init__(
        self,
        consts: Optional[YarsRevengeConstants] = None,
        reward_funcs: Optional[list[Callable]] = None,
    ):
        consts = consts or YarsRevengeConstants()
        super().__init__(consts)
        self.renderer = YarsRevengeRenderer(self.consts)

    def reset(self, key=None):
        state = YarsRevengeState(
            step_counter=jnp.array(0).astype(jnp.int32),
            level=jnp.array(1).astype(jnp.int32),
            score=jnp.array(0).astype(jnp.int32),
            lives=jnp.array(self.consts.INITIAL_LIVES).astype(jnp.int32),
            yar=DirectionEntity(
                x=jnp.array(10).astype(jnp.float32),
                y=jnp.array(105).astype(jnp.float32),
                w=jnp.array(self.consts.YAR_SIZE[0]).astype(jnp.int32),
                h=jnp.array(self.consts.YAR_SIZE[1]).astype(jnp.int32),
                direction=jnp.array(Direction.RIGHT).astype(jnp.int32)
            ),
            yar_state=jnp.array(YarState.STEADY).astype(jnp.int32),
            qotile=DirectionEntity(
                x=jnp.array(152).astype(jnp.float32),
                y=jnp.array(64).astype(jnp.float32),
                w=jnp.array(self.consts.QOTILE_SIZE[0]).astype(jnp.int32),
                h=jnp.array(self.consts.QOTILE_SIZE[1]).astype(jnp.int32),
                direction=jnp.array(Direction.DOWN).astype(jnp.int32)
            ),
            destroyer=Entity(
                x=jnp.array(155).astype(jnp.int32),
                y=jnp.array(100).astype(jnp.int32),
                w=jnp.array(self.consts.DESTROYER_SIZE[0]).astype(jnp.int32),
                h=jnp.array(self.consts.DESTROYER_SIZE[1]).astype(jnp.int32),
            ),
            swirl_exist=jnp.array(0).astype(jnp.bool),
            swirl_fired=jnp.array(0).astype(jnp.bool),
            swirl=DirectionEntity(
                x=jnp.array(0).astype(jnp.int32),
                y=jnp.array(0).astype(jnp.int32),
                w=jnp.array(self.consts.SWIRL_SIZE[0]).astype(jnp.int32),
                h=jnp.array(self.consts.SWIRL_SIZE[1]).astype(jnp.int32),
                direction=jnp.array(Direction.LEFT).astype(jnp.int32),
            ),
            energy_missile_exist=jnp.array(0).astype(jnp.bool),
            energy_missile=DirectionEntity(
                x=jnp.array(0).astype(jnp.int32),
                y=jnp.array(0).astype(jnp.int32),
                w=jnp.array(self.consts.ENERGY_MISSILE_SIZE[0]).astype(jnp.int32),
                h=jnp.array(self.consts.ENERGY_MISSILE_SIZE[1]).astype(jnp.int32),
                direction=jnp.array(Direction.RIGHT).astype(jnp.int32),
            ),
            energy_shield=Entity(
                x=jnp.array(155).astype(jnp.int32),
                y=jnp.array(100).astype(jnp.int32),
                w=jnp.array(self.consts.ENERGY_CELL_WIDTH * self.consts.INITIAL_ENERGY_SHIELD[0].shape[1]).astype(jnp.int32),
                h=jnp.array(self.consts.ENERGY_CELL_WIDTH * self.consts.INITIAL_ENERGY_SHIELD[1].shape[0]).astype(jnp.int32),
            ),
            energy_shield_state=self.consts.INITIAL_ENERGY_SHIELD[0],
        )
        initial_obs = self._get_observation(state)

        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: YarsRevengeState, action: chex.Array):
        # Extract the direction flags from the actions
        up = (
            (action == Action.UP)
            | (action == Action.UPFIRE)
            | (action == Action.UPLEFT)
            | (action == Action.UPLEFTFIRE)
            | (action == Action.UPRIGHT)
            | (action == Action.UPRIGHTFIRE)
        )
        down = (
            (action == Action.DOWN)
            | (action == Action.DOWNFIRE)
            | (action == Action.DOWNLEFT)
            | (action == Action.DOWNLEFTFIRE)
            | (action == Action.DOWNRIGHT)
            | (action == Action.DOWNRIGHTFIRE)
        )
        left = (
            (action == Action.LEFT)
            | (action == Action.LEFTFIRE)
            | (action == Action.UPLEFT)
            | (action == Action.UPLEFTFIRE)
            | (action == Action.DOWNLEFT)
            | (action == Action.DOWNLEFTFIRE)
        )
        right = (
            (action == Action.RIGHT)
            | (action == Action.RIGHTFIRE)
            | (action == Action.UPRIGHT)
            | (action == Action.UPRIGHTFIRE)
            | (action == Action.DOWNRIGHT)
            | (action == Action.DOWNRIGHTFIRE)
        )

        yar_moving = up | down | left | right
        yar_diagonal = (up | down) & (left | right)

        # Calculate the position difference compared to the previous state
        yar_speed = jax.lax.select(
            yar_diagonal, self.consts.YAR_DIAGONAL_SPEED, self.consts.YAR_SPEED
        )
        delta_yar_x = jnp.where(left, -yar_speed, jnp.where(right, yar_speed, 0.0))
        delta_yar_y = jnp.where(up, -yar_speed, jnp.where(down, yar_speed, 0.0))

        # New position calculation with boundary check (y-axis wraps)
        new_yar_x = jnp.clip(state.yar.x + delta_yar_x, 0, self.consts.WIDTH)
        new_yar_y = (state.yar.y + delta_yar_y) % self.consts.HEIGHT

        new_yar_direction = jax.lax.select(
            yar_moving, Direction.from_flags(up, down, left, right), state.yar.direction
        )
        new_yar_state = jax.lax.select(yar_moving, YarState.MOVING, YarState.STEADY)

        # Qotile Movement
        qotile_dy = jnp.where(state.qotile.direction == Direction.UP, 1, -1)
        delta_qotile_y = qotile_dy * self.consts.QOTILE_SPEED
        new_qotile_y = jnp.clip(
            state.qotile.y + delta_qotile_y,
            self.consts.QOTILE_MIN_Y,
            self.consts.QOTILE_MAX_Y,
        )

        qotile_hit_boundary = jnp.logical_or(
            new_qotile_y == self.consts.QOTILE_MIN_Y,
            new_qotile_y == self.consts.QOTILE_MAX_Y,
        )

        new_qotile_direction = jnp.where(qotile_hit_boundary, Direction.reverse(state.qotile.direction), state.qotile.direction)

        # Destroyer Movement
        destroyer_dx = jnp.sign(
            state.yar.x - state.destroyer.x + (self.consts.YAR_SIZE[0] / 2)
        )
        destroyer_dy = jnp.sign(
            state.yar.y - state.destroyer.y + (self.consts.YAR_SIZE[1] / 2)
        )

        new_destroyer_x = state.destroyer.x + self.consts.DESTROYER_SPEED * destroyer_dx
        new_destroyer_y = state.destroyer.y + self.consts.DESTROYER_SPEED * destroyer_dy

        # Energy Missile
        fire = (
            (action == Action.FIRE)
            | (action == Action.RIGHTFIRE)
            | (action == Action.UPRIGHTFIRE)
            | (action == Action.UPFIRE)
            | (action == Action.UPLEFTFIRE)
            | (action == Action.LEFTFIRE)
            | (action == Action.DOWNLEFTFIRE)
            | (action == Action.DOWNFIRE)
            | (action == Action.DOWNRIGHTFIRE)
        )

        energy_missile_hit_boundary = jnp.logical_or(
            jnp.logical_or(
                state.energy_missile.x == 0, state.energy_missile.x == self.consts.WIDTH
            ),
            jnp.logical_or(
                state.energy_missile.y == 0,
                state.energy_missile.y == self.consts.HEIGHT,
            ),
        )

        new_energy_missile_exists = jnp.where(
            energy_missile_hit_boundary,
            jnp.array(0).astype(jnp.bool),
            jnp.where(fire, jnp.array(1).astype(jnp.bool), state.energy_missile_exist),
        )
        energy_missile_exists = state.energy_missile_exist == 1

        em_up, em_down, em_left, em_right = Direction.to_flags(
            state.energy_missile.direction
        )

        new_energy_missile_direction = jnp.where(
            energy_missile_exists, state.energy_missile.direction, state.yar.direction
        )

        energy_missile_dx = jnp.where(
            em_right,
            self.consts.ENERGY_MISSILE_SPEED,
            jnp.where(em_left, -self.consts.ENERGY_MISSILE_SPEED, 0),
        )
        new_energy_missile_x = jnp.clip(
            jnp.where(
                energy_missile_exists,
                state.energy_missile.x + energy_missile_dx,
                state.yar.x + (self.consts.YAR_SIZE[0] / 2),
            ),
            0,
            self.consts.WIDTH,
        )

        energy_missile_dy = jnp.where(
            em_down,
            self.consts.ENERGY_MISSILE_SPEED,
            jnp.where(em_up, -self.consts.ENERGY_MISSILE_SPEED, 0),
        )
        new_energy_missile_y = jnp.clip(
            jnp.where(
                energy_missile_exists,
                state.energy_missile.y + energy_missile_dy,
                state.yar.y + (self.consts.YAR_SIZE[1] / 2),
            ),
            0,
            self.consts.HEIGHT,
        )

        new_state = YarsRevengeState(
            step_counter=state.step_counter + 1,
            level=state.level,
            score=state.score,
            lives=state.lives,
            yar=DirectionEntity(
                x=new_yar_x,
                y=new_yar_y,
                w=state.yar.w,
                h=state.yar.h,
                direction=new_yar_direction
            ),
            yar_state=new_yar_state,
            qotile=DirectionEntity(
                x=state.qotile.x,
                y=new_qotile_y,
                w=state.qotile.w,
                h=state.qotile.h,
                direction=new_qotile_direction
            ),
            destroyer=Entity(
                x=new_destroyer_x,
                y=new_destroyer_y,
                w=state.destroyer.x,
                h=state.destroyer.y,
            ),
            swirl_exist=state.swirl_exist,
            swirl_fired=state.swirl_fired,
            swirl=DirectionEntity(
                x=state.swirl.x,
                y=state.swirl.y,
                w=state.swirl.w,
                h=state.swirl.h,
                direction=state.swirl.direction
            ),
            energy_missile_exist=new_energy_missile_exists,
            energy_missile=DirectionEntity(
                x=new_energy_missile_x,
                y=new_energy_missile_y,
                w=state.energy_missile.w,
                h=state.energy_missile.h,
                direction=new_energy_missile_direction
            ),
            energy_shield=Entity( #TODO: fix the values
                x=jnp.array(155).astype(jnp.int32),
                y=jnp.array(100).astype(jnp.int32),
                w=jnp.array(self.consts.ENERGY_CELL_WIDTH * self.consts.INITIAL_ENERGY_SHIELD[0].shape[1]).astype(jnp.int32),
                h=jnp.array(self.consts.ENERGY_CELL_WIDTH * self.consts.INITIAL_ENERGY_SHIELD[1].shape[0]).astype(jnp.int32),
            ),
            energy_shield_state=state.energy_shield_state,
        )

        observation = self._get_observation(new_state)
        reward = self._get_reward(state, new_state)
        done = self._get_done(new_state)
        info = self._get_info(new_state)
        return observation, new_state, reward, done, info

    def render(self, state):
        return self.renderer.render(state)

    def action_space(self):
        return spaces.Discrete(18)

    def observation_space(self):
        return spaces.Dict(
            {
                "yar": spaces.Dict(
                    {
                        "x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                        "y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                        "w": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                        "h": spaces.Box(
                            low=0, high=210, shape=(), dtype=jnp.int32
                        ),
                    }
                ),
                "qotile": spaces.Dict(
                    {
                        "x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                        "y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                        "w": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                        "h": spaces.Box(
                            low=0, high=210, shape=(), dtype=jnp.int32
                        ),
                    }
                ),
                "destroyer": spaces.Dict(
                    {
                        "x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                        "y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                        "w": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                        "h": spaces.Box(
                            low=0, high=210, shape=(), dtype=jnp.int32
                        ),
                    }
                ),
                "swirl_exists": spaces.Box(low=0, high=1, shape=(), dtype=jnp.bool),
                "swirl_fired": spaces.Box(low=0, high=1, shape=(), dtype=jnp.bool),
                "swirl": spaces.Dict(
                    {
                        "x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                        "y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                        "w": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                        "h": spaces.Box(
                            low=0, high=210, shape=(), dtype=jnp.int32
                        ),
                    }
                ),
                "energy_missile_exists": spaces.Box(
                    low=0, high=1, shape=(), dtype=jnp.bool
                ),
                "energy_missile": spaces.Dict(
                    {
                        "x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                        "y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                        "w": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                        "h": spaces.Box(
                            low=0, high=210, shape=(), dtype=jnp.int32
                        ),
                    }
                ),
                "energy_shield": spaces.Dict(
                    {
                        "x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                        "y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                        "w": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                        "h": spaces.Box(
                            low=0, high=210, shape=(), dtype=jnp.int32
                        ),
                    }
                ),
                "energy_shield_state": spaces.Box(
                    low=0, high=1, shape=(8, 16), dtype=jnp.int32
                ),
                "lives": spaces.Box(low=0, high=4, shape=(), dtype=jnp.int32),
            }
        )

    def image_space(self):
        return spaces.Box(low=0, high=255, shape=(210, 160, 3), dtype=jnp.uint8)

    def _get_observation(self, state: YarsRevengeState) -> YarsRevengeObservation:
        return YarsRevengeObservation(
            yar=state.yar,
            qotile=state.qotile,
            destroyer=state.destroyer,
            swirl_exist=state.swirl_exist,
            swirl_fired=state.swirl_fired,
            swirl=state.swirl,
            energy_missile_exist=state.energy_missile_exist,
            energy_missile=state.energy_missile,
            energy_shield=state.energy_shield,
            energy_shield_state=state.energy_shield_state,
            lives=state.lives,
        )

    def obs_to_flat_array(self, obs):
        return jnp.concatenate(
            [
                obs.yar.x.flatten(),
                obs.yar.y.flatten(),
                obs.yar.w.flatten(),
                obs.yar.h.flatten(),
                obs.yar.direction.flatten(),
                obs.qotile.x.flatten(),
                obs.qotile.y.flatten(),
                obs.qotile.w.flatten(),
                obs.qotile.h.flatten(),
                obs.qotile.direction.flatten(),
                obs.destroyer.x.flatten(),
                obs.destroyer.y.flatten(),
                obs.destroyer.w.flatten(),
                obs.destroyer.h.flatten(),
                obs.swirl_exist.flatten(),
                obs.swirl_fired.flatten(),
                obs.swirl.x.flatten(),
                obs.swirl.y.flatten(),
                obs.swirl.w.flatten(),
                obs.swirl.h.flatten(),
                obs.swirl.direction.flatten(),
                obs.energy_missile_exist.flatten(),
                obs.energy_missile.x.flatten(),
                obs.energy_missile.y.flatten(),
                obs.energy_missile.w.flatten(),
                obs.energy_missile.h.flatten(),
                obs.energy_missile.direction.flatten(),
                obs.energy_shield.x.flatten(),
                obs.energy_shield.y.flatten(),
                obs.energy_shield.w.flatten(),
                obs.energy_shield.h.flatten(),
                obs.energy_shield_state.flatten(),
            ]
        )

    def _get_info(self, state, all_rewards: Optional[Array] = None):
        return YarsRevengeInfo(time=state.step_counter, current_level=state.level)

    def _get_reward(self, previous_state, state):
        return state.score - previous_state.score

    def _get_done(self, state):
        return False


class YarsRevengeRenderer(JAXGameRenderer):
    def __init__(self, consts: Optional[YarsRevengeConstants] = None):
        super().__init__(consts)
        self.consts = consts or YarsRevengeConstants()
        self.config = render_utils.RendererConfig(
            game_dimensions=(210, 160), channels=3
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)

        asset_config = self._get_asset_config()
        sprite_path = (
            f"{os.path.dirname(os.path.abspath(__file__))}/sprites/yarsrevenge"
        )

        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS,
        ) = self.jr.load_and_setup_assets(asset_config, sprite_path)

    def _get_asset_config(self) -> list:
        return [
            {"name": "background", "type": "background", "file": "background.npy"},
            {
                "name": "yar",
                "type": "group",
                "files": [
                    "yar_right_0.npy",
                    "yar_right_1.npy",
                    "yar_upright_0.npy",
                    "yar_upright_1.npy",
                    "yar_up_0.npy",
                    "yar_up_1.npy",
                    "yar_upleft_0.npy",
                    "yar_upleft_1.npy",
                    "yar_left_0.npy",
                    "yar_left_1.npy",
                    "yar_downleft_0.npy",
                    "yar_downleft_1.npy",
                    "yar_down_0.npy",
                    "yar_down_1.npy",
                    "yar_downright_0.npy",
                    "yar_downright_1.npy",
                ],
            },
            {"name": "qotile", "type": "single", "file": "qotile.npy"},
        ]

    def get_animation_idx(
        self,
        step: jnp.ndarray,
        group: jnp.ndarray,
        duration: int,
        group_item_count: int,
    ):
        return group * 2 + (jnp.floor(step / duration).astype(int32) % group_item_count)

    def render(self, state: YarsRevengeState):
        raster = self.jr.create_object_raster(self.BACKGROUND)

        yar_animation_duration = jax.lax.cond(
            state.yar_state == YarState.MOVING,
            lambda: self.consts.MOVING_YAR_MOVEMENT_FRAME,
            lambda: self.consts.STEADY_YAR_MOVEMENT_FRAME,
        )
        yar_sprite_idx = self.get_animation_idx(
            state.step_counter, state.yar.direction, yar_animation_duration, 2
        )
        yar_mask = self.SHAPE_MASKS["yar"][yar_sprite_idx]
        raster = self.jr.render_at(raster, state.yar.x, state.yar.y, yar_mask)

        qotile_mask = self.SHAPE_MASKS["qotile"]
        raster = self.jr.render_at(raster, state.qotile.x, state.qotile.y, qotile_mask)

        destroyer_mask = jnp.ones(
            (self.consts.DESTROYER_SIZE[1], self.consts.DESTROYER_SIZE[0])
        )
        raster = self.jr.render_at(
            raster, state.destroyer.x, state.destroyer.y, destroyer_mask
        )

        energy_missile_mask = jnp.ones(
            (self.consts.ENERGY_MISSILE_SIZE[1], self.consts.ENERGY_MISSILE_SIZE[0])
        )
        raster = jnp.where(
            state.energy_missile_exist,
            self.jr.render_at(
                raster,
                state.energy_missile.x,
                state.energy_missile.y,
                energy_missile_mask,
            ),
            raster,
        )

        return self.jr.render_from_palette(raster, self.PALETTE)
