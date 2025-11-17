import os
from enum import IntEnum
from functools import partial
from typing import Callable, NamedTuple, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
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


class YarState(IntEnum):
    STEADY = 0
    MOVING = 1


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

    # Entity Speeds, 1 pixel per X frame
    QOTILE_SPEED = 0.5
    YAR_SPEED = 2.0
    YAR_DIAGONAL_SPEED = 1.0
    SWIRL_SPEED = 2.0
    DESTROYER_SPEED = 0.125
    ENERGY_MISSILE_SPEED = 2.0

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

    yar_x: jnp.ndarray
    yar_y: jnp.ndarray
    yar_direction: jnp.ndarray  # as Direction enum
    yar_state: jnp.ndarray  # as YarState enum
    qotile_x: jnp.ndarray
    qotile_y: jnp.ndarray
    destroyer_x: jnp.ndarray
    destroyer_y: jnp.ndarray

    swirl_exist: jnp.ndarray
    swirl_fired: jnp.ndarray
    swirl_x: jnp.ndarray
    swirl_y: jnp.ndarray
    energy_missile_exist: jnp.ndarray
    energy_missile_fired: jnp.ndarray
    energy_missile_x: jnp.ndarray
    energy_missile_y: jnp.ndarray

    energy_shield: jnp.ndarray


class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray


class YarsRevengeObservation(NamedTuple):
    yar_pos: EntityPosition
    yar_direction: jnp.ndarray
    qotile_pos: EntityPosition
    destroyer_pos: EntityPosition
    swirl_exist: jnp.ndarray
    swirl_fired: jnp.ndarray
    swirl_pos: EntityPosition
    energy_missile_exist: jnp.ndarray
    energy_missile_fired: jnp.ndarray
    energy_missile_pos: EntityPosition
    energy_shield: jnp.ndarray
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
            yar_x=jnp.array(10).astype(jnp.int32),
            yar_y=jnp.array(100).astype(jnp.int32),
            yar_direction=jnp.array(Direction.RIGHT).astype(jnp.int32),
            yar_state=jnp.array(YarState.STEADY).astype(jnp.int32),
            qotile_x=jnp.array(152).astype(jnp.int32),
            qotile_y=jnp.array(100).astype(jnp.int32),
            destroyer_x=jnp.array(155).astype(jnp.int32),
            destroyer_y=jnp.array(100).astype(jnp.int32),
            swirl_exist=jnp.array(0).astype(jnp.bool),
            swirl_fired=jnp.array(0).astype(jnp.bool),
            swirl_x=jnp.array(0).astype(jnp.int32),
            swirl_y=jnp.array(0).astype(jnp.int32),
            energy_missile_exist=jnp.array(0).astype(jnp.bool),
            energy_missile_fired=jnp.array(0).astype(jnp.bool),
            energy_missile_x=jnp.array(0).astype(jnp.int32),
            energy_missile_y=jnp.array(0).astype(jnp.int32),
            energy_shield=self.consts.INITIAL_ENERGY_SHIELD[0],
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
        new_yar_x = jnp.clip(state.yar_x + delta_yar_x, 0, self.consts.WIDTH)
        new_yar_y = (state.yar_y + delta_yar_y) % self.consts.HEIGHT

        new_yar_direction = jax.lax.select(
            yar_moving, Direction.from_flags(up, down, left, right), state.yar_direction
        )
        new_yar_state = jax.lax.select(yar_moving, YarState.MOVING, YarState.STEADY)

        new_state = YarsRevengeState(
            step_counter=state.step_counter + 1,
            level=state.level,
            score=state.score,
            lives=state.lives,
            yar_x=new_yar_x,
            yar_y=new_yar_y,
            yar_direction=new_yar_direction,
            yar_state=new_yar_state,
            qotile_x=state.qotile_x,
            qotile_y=state.qotile_y,
            destroyer_x=state.destroyer_x,
            destroyer_y=state.destroyer_y,
            swirl_exist=state.swirl_exist,
            swirl_fired=state.swirl_fired,
            swirl_x=state.swirl_x,
            swirl_y=state.swirl_y,
            energy_missile_exist=state.energy_missile_exist,
            energy_missile_fired=state.energy_missile_fired,
            energy_missile_x=state.energy_missile_x,
            energy_missile_y=state.energy_missile_y,
            energy_shield=state.energy_shield,
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
                "yar_pos": spaces.Dict(
                    {
                        "x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                        "y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                        "width": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                        "height": spaces.Box(
                            low=0, high=210, shape=(), dtype=jnp.int32
                        ),
                    }
                ),
                "yar_direction": spaces.Box(low=0, high=7, shape=(), dtype=jnp.int32),
                "qotile_pos": spaces.Dict(
                    {
                        "x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                        "y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                        "width": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                        "height": spaces.Box(
                            low=0, high=210, shape=(), dtype=jnp.int32
                        ),
                    }
                ),
                "destroyer_pos": spaces.Dict(
                    {
                        "x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                        "y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                        "width": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                        "height": spaces.Box(
                            low=0, high=210, shape=(), dtype=jnp.int32
                        ),
                    }
                ),
                "swirl_exists": spaces.Box(low=0, high=1, shape=(), dtype=jnp.bool),
                "swirl_fired": spaces.Box(low=0, high=1, shape=(), dtype=jnp.bool),
                "swirl_pos": spaces.Dict(
                    {
                        "x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                        "y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                        "width": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                        "height": spaces.Box(
                            low=0, high=210, shape=(), dtype=jnp.int32
                        ),
                    }
                ),
                "energy_missile_exists": spaces.Box(
                    low=0, high=1, shape=(), dtype=jnp.bool
                ),
                "energy_missile_fired": spaces.Box(
                    low=0, high=1, shape=(), dtype=jnp.bool
                ),
                "energy_missile_pos": spaces.Dict(
                    {
                        "x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                        "y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                        "width": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                        "height": spaces.Box(
                            low=0, high=210, shape=(), dtype=jnp.int32
                        ),
                    }
                ),
                "energy_shield": spaces.Box(
                    low=0, high=1, shape=(8, 16), dtype=jnp.int32
                ),
                "lives": spaces.Box(low=0, high=4, shape=(), dtype=jnp.int32),
            }
        )

    def image_space(self):
        return spaces.Box(low=0, high=255, shape=(210, 160, 3), dtype=jnp.uint8)

    def _get_observation(self, state: YarsRevengeState) -> YarsRevengeObservation:
        return YarsRevengeObservation(
            yar_pos=EntityPosition(
                x=state.yar_x,
                y=state.yar_y,
                width=jnp.array(self.consts.YAR_SIZE[0]),
                height=jnp.array(self.consts.YAR_SIZE[1]),
            ),
            yar_direction=state.yar_direction,
            qotile_pos=EntityPosition(
                x=state.qotile_x,
                y=state.qotile_y,
                width=jnp.array(self.consts.QOTILE_SIZE[0]),
                height=jnp.array(self.consts.QOTILE_SIZE[1]),
            ),
            destroyer_pos=EntityPosition(
                x=state.destroyer_x,
                y=state.destroyer_y,
                width=jnp.array(self.consts.DESTROYER_SIZE[0]),
                height=jnp.array(self.consts.DESTROYER_SIZE[1]),
            ),
            swirl_exist=state.swirl_exist,
            swirl_fired=state.swirl_fired,
            swirl_pos=EntityPosition(
                x=state.yar_x,
                y=state.yar_y,
                width=jnp.array(self.consts.SWIRL_SIZE[0]),
                height=jnp.array(self.consts.SWIRL_SIZE[1]),
            ),
            energy_missile_exist=state.energy_missile_exist,
            energy_missile_fired=state.energy_missile_fired,
            energy_missile_pos=EntityPosition(
                x=state.yar_x,
                y=state.yar_y,
                width=jnp.array(self.consts.ENERGY_MISSILE_SIZE[0]),
                height=jnp.array(self.consts.ENERGY_MISSILE_SIZE[1]),
            ),
            energy_shield=state.energy_shield,
            lives=state.lives,
        )

    def obs_to_flat_array(self, obs):
        return jnp.concatenate(
            [
                obs.yar_pos.x.flatten(),
                obs.yar_pos.y.flatten(),
                obs.yar_pos.width.flatten(),
                obs.yar_pos.height.flatten(),
                obs.yar_direction.flatten(),
                obs.qotile_pos.x.flatten(),
                obs.qotile_pos.y.flatten(),
                obs.qotile_pos.width.flatten(),
                obs.qotile_pos.height.flatten(),
                obs.destroyer_pos.x.flatten(),
                obs.destroyer_pos.y.flatten(),
                obs.destroyer_pos.width.flatten(),
                obs.destroyer_pos.height.flatten(),
                obs.swirl_exist.flatten(),
                obs.swirl_fired.flatten(),
                obs.swirl_pos.x.flatten(),
                obs.swirl_pos.y.flatten(),
                obs.swirl_pos.width.flatten(),
                obs.swirl_pos.height.flatten(),
                obs.energy_missile_exist.flatten(),
                obs.energy_missile_fired.flatten(),
                obs.energy_missile_pos.x.flatten(),
                obs.energy_missile_pos.y.flatten(),
                obs.energy_missile_pos.width.flatten(),
                obs.energy_missile_pos.height.flatten(),
                obs.energy_shield.flatten(),
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
            state.step_counter, state.yar_direction, yar_animation_duration, 2
        )
        yar_mask = self.SHAPE_MASKS["yar"][yar_sprite_idx]
        raster = self.jr.render_at(raster, state.yar_x, state.yar_y, yar_mask)

        return self.jr.render_from_palette(raster, self.PALETTE)
