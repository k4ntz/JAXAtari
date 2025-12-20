import os
from enum import IntEnum
from functools import partial
from typing import Callable, NamedTuple, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from jax import Array

from jaxatari import spaces
from jaxatari.environment import JAXAtariAction as Action
from jaxatari.environment import JaxEnvironment
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils

type DirectionFlags = Tuple[
    bool | jnp.ndarray,
    bool | jnp.ndarray,
    bool | jnp.ndarray,
    bool | jnp.ndarray,
]


class Direction(IntEnum):
    _CENTER = -1
    RIGHT = 0
    UPRIGHT = 1
    UP = 2
    UPLEFT = 3
    LEFT = 4
    DOWNLEFT = 5
    DOWN = 6
    DOWNRIGHT = 7

    @staticmethod
    @jax.jit
    def from_flags(flags: DirectionFlags):
        up, down, left, right = flags
        return jnp.select(
            [
                ~right & ~up & ~down & ~left,
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
                Direction._CENTER,
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
    @jax.jit
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
    @jax.jit
    def reverse(direction: jnp.ndarray):
        return jnp.where(
            direction == Direction._CENTER,
            jnp.array(Direction._CENTER).astype(jnp.int32),
            jnp.array((direction + 4) % 8).astype(jnp.int32),
        )

    @staticmethod
    @jax.jit
    def to_delta(
        flags: DirectionFlags,
        speed: float | jnp.ndarray,
    ):
        up, down, left, right = flags
        dx = jnp.where(left, -speed, jnp.where(right, speed, 0.0))
        dy = jnp.where(up, -speed, jnp.where(down, speed, 0.0))
        return dx, dy


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


class YarsRevengeConstants(NamedTuple):
    # Constants for game environment
    WIDTH: int = 160
    HEIGHT: int = 210

    INITIAL_LIVES: int = 4
    DEVOUR_THRESHOLD: int = 5

    # Entity sizes (width, height), horizontal orientation
    YAR_SIZE: Tuple[int, int] = (8, 16)  # Facing to right
    QOTILE_SIZE: Tuple[int, int] = (8, 18)
    NEUTRAL_ZONE_SIZE: Tuple[int, int] = (28, HEIGHT)
    SWIRL_SIZE: Tuple[int, int] = (8, 16)
    DESTROYER_SIZE: Tuple[int, int] = (4, 2)
    ENERGY_MISSILE_SIZE: Tuple[int, int] = (1, 2)
    CANNON_SIZE: Tuple[int, int] = (8, 8)

    QOTILE_MIN_Y: int = 60
    QOTILE_MAX_Y: int = 150
    NEUTRAL_ZONE_POSITION: Tuple[int, int] = (50, 0)

    SWIRL_PER_STEP: int = 1000
    SWIRL_FIRE_PER_STEP: int = 250

    # Entity Speeds, X pixel per 1 frame
    QOTILE_SPEED = 0.5
    YAR_SPEED = 2.0
    YAR_DIAGONAL_SPEED = 1.0
    SWIRL_SPEED = 3.0
    DESTROYER_SPEED = 0.125
    ENERGY_MISSILE_SPEED = 4.0
    CANNON_SPEED = 2.0

    SNAKE_FRAME = 4

    STEADY_YAR_MOVEMENT_FRAME = (
        4  # Movement animation change interval for Yar (no action)
    )
    MOVING_YAR_MOVEMENT_FRAME = 1  # Movement animation change interval for Yar (moving)
    CANNON_MOVEMENT_FRAME = 8  # Animation change interval for cannon
    SWIRL_MOVEMENT_FRAME = 2

    ENERGY_SHIELD_COLOR: Tuple[int, int, int] = (163, 57, 21)
    NEUTRAL_ZONE_COLOR: Tuple[int, int, int] = (20, 20, 20)

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
        dtype=jnp.int32,
    )

    MISSILE_HIT_KERNEL = jnp.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=jnp.int32,
    )


class YarsRevengeState(NamedTuple):
    step_counter: jnp.ndarray
    level: jnp.ndarray
    score: jnp.ndarray
    lives: jnp.ndarray
    stage: jnp.ndarray

    yar: DirectionEntity
    yar_state: jnp.ndarray  # as YarState enum
    yar_devour_count: jnp.ndarray
    qotile: DirectionEntity
    destroyer: Entity

    swirl_exist: jnp.ndarray
    swirl: Entity
    swirl_dx: jnp.ndarray
    swirl_dy: jnp.ndarray
    energy_missile_exist: jnp.ndarray
    energy_missile: DirectionEntity
    cannon_exist: jnp.ndarray
    cannon_fired: jnp.ndarray
    cannon: DirectionEntity

    energy_shield: Entity
    energy_shield_state: jnp.ndarray

    neutral_zone: Entity


class YarsRevengeObservation(NamedTuple):
    yar: DirectionEntity
    qotile: DirectionEntity
    destroyer: Entity
    swirl_exist: jnp.ndarray
    swirl: Entity
    swirl_dx: jnp.ndarray
    swirl_dy: jnp.ndarray
    energy_missile_exist: jnp.ndarray
    energy_missile: DirectionEntity
    cannon_exist: jnp.ndarray
    cannon_fired: jnp.ndarray
    cannon: DirectionEntity
    energy_shield: Entity
    energy_shield_state: jnp.ndarray
    lives: jnp.ndarray


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

    @partial(jax.jit, static_argnums=(0,))
    def construct_initial_state(self, stage):
        return YarsRevengeState(
            step_counter=jnp.array(0).astype(jnp.int32),
            level=jnp.array(1).astype(jnp.int32),
            score=jnp.array(0).astype(jnp.int32),
            lives=jnp.array(self.consts.INITIAL_LIVES).astype(jnp.int32),
            stage=jnp.array(stage).astype(jnp.int32),
            yar=DirectionEntity(
                x=jnp.array(10).astype(jnp.float32),
                y=jnp.array(105).astype(jnp.float32),
                w=jnp.array(self.consts.YAR_SIZE[0]).astype(jnp.int32),
                h=jnp.array(self.consts.YAR_SIZE[1]).astype(jnp.int32),
                direction=jnp.array(Direction.RIGHT).astype(jnp.int32),
            ),
            yar_state=jnp.array(YarState.STEADY).astype(jnp.int32),
            yar_devour_count=jnp.array(0).astype(jnp.int32),
            qotile=DirectionEntity(
                x=jnp.array(150).astype(jnp.float32),
                y=jnp.array(self.consts.QOTILE_MIN_Y).astype(jnp.float32),
                w=jnp.array(self.consts.QOTILE_SIZE[0]).astype(jnp.int32),
                h=jnp.array(self.consts.QOTILE_SIZE[1]).astype(jnp.int32),
                direction=jnp.array(Direction.DOWN).astype(jnp.int32),
            ),
            destroyer=Entity(
                x=jnp.array(155).astype(jnp.float32),
                y=jnp.array(100).astype(jnp.float32),
                w=jnp.array(self.consts.DESTROYER_SIZE[0]).astype(jnp.int32),
                h=jnp.array(self.consts.DESTROYER_SIZE[1]).astype(jnp.int32),
            ),
            swirl_exist=jnp.array(0).astype(jnp.int32),
            swirl=Entity(
                x=jnp.array(0).astype(jnp.float32),
                y=jnp.array(0).astype(jnp.float32),
                w=jnp.array(self.consts.SWIRL_SIZE[0]).astype(jnp.int32),
                h=jnp.array(self.consts.SWIRL_SIZE[1]).astype(jnp.int32),
            ),
            swirl_dx=jnp.array(0).astype(jnp.float32),
            swirl_dy=jnp.array(0).astype(jnp.float32),
            energy_missile_exist=jnp.array(0).astype(jnp.int32),
            energy_missile=DirectionEntity(
                x=jnp.array(0).astype(jnp.float32),
                y=jnp.array(0).astype(jnp.float32),
                w=jnp.array(self.consts.ENERGY_MISSILE_SIZE[0]).astype(jnp.int32),
                h=jnp.array(self.consts.ENERGY_MISSILE_SIZE[1]).astype(jnp.int32),
                direction=jnp.array(Direction.RIGHT).astype(jnp.int32),
            ),
            cannon_exist=jnp.array(0).astype(jnp.int32),
            cannon_fired=jnp.array(0).astype(jnp.int32),
            cannon=DirectionEntity(
                x=jnp.array(1).astype(jnp.float32),
                y=jnp.array(0).astype(jnp.float32),
                w=jnp.array(self.consts.CANNON_SIZE[0]).astype(jnp.int32),
                h=jnp.array(self.consts.CANNON_SIZE[1]).astype(jnp.int32),
                direction=jnp.array(Direction.RIGHT).astype(jnp.int32),
            ),
            energy_shield=Entity(
                x=jnp.array(128).astype(jnp.float32),
                y=jnp.array(0).astype(jnp.float32),
                w=jnp.array(
                    self.consts.ENERGY_CELL_WIDTH
                    * self.consts.INITIAL_ENERGY_SHIELD[stage].shape[1]
                ).astype(jnp.int32),
                h=jnp.array(
                    self.consts.ENERGY_CELL_HEIGHT
                    * self.consts.INITIAL_ENERGY_SHIELD[stage].shape[0]
                ).astype(jnp.int32),
            ),
            energy_shield_state=self.consts.INITIAL_ENERGY_SHIELD[stage],
            neutral_zone=Entity(
                x=jnp.array(self.consts.NEUTRAL_ZONE_POSITION[0]).astype(jnp.float32),
                y=jnp.array(self.consts.NEUTRAL_ZONE_POSITION[1]).astype(jnp.float32),
                w=jnp.array(self.consts.NEUTRAL_ZONE_SIZE[0]).astype(jnp.int32),
                h=jnp.array(self.consts.NEUTRAL_ZONE_SIZE[1]).astype(jnp.int32),
            ),
        )

    def reset(self, key=None):
        state = self.construct_initial_state(0)
        initial_obs = self._get_observation(state)

        return initial_obs, state

    # ----------------------------------------------------
    # Helper functions for step processing
    # ----------------------------------------------------

    @staticmethod
    @jax.jit
    def _parse_action_flags(action: chex.Array) -> Tuple[DirectionFlags, jnp.ndarray]:
        """Return up/down/left/right and fire flags from an action."""
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
        return (up, down, left, right), fire

    @staticmethod
    @jax.jit
    def _get_entity_position(
        entity: Entity | DirectionEntity,
        direction: Direction,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Return the absolute (x, y) of an entity given a direction."""
        up, down, left, right = Direction.to_flags(direction)
        w_half = entity.w // 2
        h_half = entity.h // 2

        x_offset = jnp.where(left, 0, jnp.where(right, entity.w, w_half))
        y_offset = jnp.where(up, 0, jnp.where(down, entity.h, h_half))

        return entity.x + x_offset, entity.y + y_offset

    @staticmethod
    @jax.jit
    def _move_entity(
        entity: Entity | DirectionEntity,
        delta_x: jnp.ndarray,
        delta_y: jnp.ndarray,
        min_x: int = 0,
        min_y: int = 0,
        max_x: int = 160,  # default WIDTH
        max_y: int = 210,  # default HEIGHT
        wrap_y: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Move an entity and clip it inside the screen bounds."""
        new_x = jnp.clip(entity.x + delta_x, min_x, max_x - entity.w)
        new_y = entity.y + delta_y
        new_y = jax.lax.cond(
            wrap_y,
            lambda: jnp.mod(new_y, jnp.asarray(max_y - entity.h)),
            lambda: jnp.clip(new_y, min_y, max_y - entity.h),
        )
        return new_x, new_y

    @staticmethod
    @jax.jit
    def _check_entity_boundary(
        entity: Entity | DirectionEntity,
        max_x: int = 160,
        max_y: int = 210,
        min_x: int = 0,
        min_y: int = 0,
    ) -> jnp.ndarray:
        """Return True if the entity touches a screen border."""
        right = entity.x + entity.w >= max_x
        bottom = entity.y + entity.h >= max_y
        left = entity.x <= min_x
        top = entity.y <= min_y
        return jnp.logical_or(jnp.logical_or(right, bottom), jnp.logical_or(left, top))

    @staticmethod
    @jax.jit
    def _check_entity_collusion(
        a: Entity | DirectionEntity, b: Entity | DirectionEntity
    ) -> jnp.ndarray:
        """Return True if two entities overlap."""
        a_up, a_left = JaxYarsRevenge._get_entity_position(a, Direction.UPLEFT)
        a_down, a_right = JaxYarsRevenge._get_entity_position(a, Direction.DOWNRIGHT)
        b_up, b_left = JaxYarsRevenge._get_entity_position(b, Direction.UPLEFT)
        b_down, b_right = JaxYarsRevenge._get_entity_position(b, Direction.DOWNRIGHT)

        horizontal_overlap = jnp.logical_and(a_left < b_right, a_right > b_left)
        vertical_overlap = jnp.logical_and(a_up < b_down, a_down > b_up)
        return horizontal_overlap & vertical_overlap

    @staticmethod
    @jax.jit
    def _check_energy_shield_collusion(
        entity: Entity | DirectionEntity,
        energy_shield: Entity,
        shield_state: jnp.ndarray,
        cell_height: int = 8,
        cell_width: int = 4,
    ) -> jnp.ndarray:
        """Return a boolean mask of cells in the shield that collide with an
        entity.  The returned array has the same shape as *shield_state*.
        """
        rows, cols = shield_state.shape

        e_left, e_up = JaxYarsRevenge._get_entity_position(entity, Direction.UPLEFT)
        e_right, e_down = JaxYarsRevenge._get_entity_position(
            entity, Direction.DOWNRIGHT
        )

        col_x = jnp.arange(cols) * cell_width
        row_y = jnp.arange(rows) * cell_height

        cell_left = energy_shield.x + col_x[None, :]
        cell_top = energy_shield.y + row_y[:, None]
        cell_right = cell_left + cell_width
        cell_bottom = cell_top + cell_height

        horiz_overlap = (e_left < cell_right) & (e_right > cell_left)
        vert_overlap = (e_up < cell_bottom) & (e_down > cell_top)

        collision_mask = horiz_overlap & vert_overlap
        active_hits = collision_mask & shield_state
        return active_hits

    @staticmethod
    @jax.jit
    def _snake_shift(shield: jnp.ndarray) -> jnp.ndarray:
        """Shift the shield in a snake-like pattern."""
        n_rows, n_cols = shield.shape
        r = jnp.arange(n_rows).reshape(-1, 1)
        c = jnp.arange(n_cols).reshape(1, -1)
        idx_normal = r * n_cols + c
        idx_reversed = r * n_cols + (n_cols - 1 - c)
        snake_idx = jnp.where(r % 2 == 0, idx_normal, idx_reversed)

        s_flat_snake = shield.reshape(-1)[snake_idx.ravel()]
        shifted = jnp.roll(s_flat_snake, 1)
        new_snake = shifted.reshape(n_rows, n_cols)
        return jnp.where(r % 2 == 0, new_snake, new_snake[:, ::-1])

    @partial(jax.jit, static_argnums=(0,))
    def _yar_step(self, state: YarsRevengeState, direction_flags: DirectionFlags):
        # Calculate the moving direction
        direction = Direction.from_flags(direction_flags)
        yar_moving = direction != Direction._CENTER
        new_yar_direction = jax.lax.select(yar_moving, direction, state.yar.direction)
        new_yar_state = jax.lax.select(yar_moving, YarState.MOVING, YarState.STEADY)

        yar_diagonal = (direction_flags[0] | direction_flags[1]) & (
            direction_flags[2] | direction_flags[3]
        )

        # Calculate the position difference compared to the previous state
        yar_speed = jax.lax.select(
            yar_diagonal, self.consts.YAR_DIAGONAL_SPEED, self.consts.YAR_SPEED
        )
        delta_yar_x, delta_yar_y = Direction.to_delta(direction_flags, yar_speed)

        # New position calculation with boundary check (y-axis wraps)
        new_yar_x, new_yar_y = self._move_entity(
            state.yar,
            delta_yar_x,
            delta_yar_y,
            wrap_y=True,
        )

        new_yar_entity = state.yar._replace(
            x=new_yar_x, y=new_yar_y, direction=new_yar_direction
        )

        yar_shield_collusion = self._check_energy_shield_collusion(
            new_yar_entity, state.energy_shield, state.energy_shield_state
        )
        yar_hit_shield = jnp.any(yar_shield_collusion)

        # Shift yar to left if it hits shield
        new_yar_x = jnp.where(
            yar_hit_shield,
            new_yar_x - self.consts.ENERGY_CELL_WIDTH,
            new_yar_x,
        )

        new_yar_entity = new_yar_entity._replace(x=new_yar_x)

        yar_qotile_collusion = self._check_entity_collusion(
            new_yar_entity, state.qotile
        )
        new_yar_devour_count = jnp.where(
            yar_hit_shield & yar_moving | yar_qotile_collusion,
            state.yar_devour_count + 1,
            state.yar_devour_count,
        )

        # Devour Check
        def energy_shield_devour():
            row_has_true = jnp.any(yar_shield_collusion, axis=1)
            n_rows = jnp.sum(row_has_true)
            k = n_rows // 2 + 1
            cumulative_row = jnp.cumsum(row_has_true.astype(jnp.int32))
            median_row_idx = jnp.argmax(cumulative_row >= k)
            left_col = jnp.argmax(yar_shield_collusion[median_row_idx])
            return state.energy_shield_state.at[median_row_idx, left_col].set(False)

        devour_reset = new_yar_devour_count == self.consts.DEVOUR_THRESHOLD

        new_energy_shield_state = jax.lax.cond(
            devour_reset,
            energy_shield_devour,
            lambda: state.energy_shield_state,
        )

        new_yar_devour_count = jnp.where(
            devour_reset,
            0,
            new_yar_devour_count,
        )

        yar_neutral = self._check_entity_collusion(state.yar, state.neutral_zone)

        return (
            dict(
                yar=new_yar_entity,
                yar_state=new_yar_state,
                yar_devour_count=new_yar_devour_count,
                energy_shield_state=new_energy_shield_state,
            ),
            yar_neutral,
            devour_reset,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _qotile_step(self, state: YarsRevengeState):
        # Qotile Movement
        qotile_hit_boundary = self._check_entity_boundary(
            state.qotile,
            self.consts.WIDTH,
            max_y=self.consts.QOTILE_MAX_Y,
            min_y=self.consts.QOTILE_MIN_Y,
        )

        new_qotile_direction = jnp.where(
            qotile_hit_boundary,
            Direction.reverse(state.qotile.direction),
            state.qotile.direction,
        )

        _, delta_qotile_y = Direction.to_delta(
            Direction.to_flags(new_qotile_direction), self.consts.QOTILE_SPEED
        )

        _, new_qotile_y = self._move_entity(
            state.qotile,
            _,
            delta_qotile_y,
            min_y=self.consts.QOTILE_MIN_Y,
            max_y=self.consts.QOTILE_MAX_Y,
        )

        # Energy Shield Y calculation
        new_energy_shield_y = (
            self._get_entity_position(state.qotile, Direction._CENTER)[1]
            - state.energy_shield_state.shape[0] * self.consts.ENERGY_CELL_HEIGHT / 2
        )

        return dict(
            qotile=state.qotile._replace(
                y=new_qotile_y, direction=new_qotile_direction
            ),
            energy_shield=state.energy_shield._replace(y=new_energy_shield_y),
        )

    @partial(jax.jit, static_argnums=(0,))
    def _destroyer_step(self, state: YarsRevengeState):
        yar_center_x, yar_center_y = self._get_entity_position(
            state.yar, Direction._CENTER
        )
        dx = jnp.sign(yar_center_x - state.destroyer.x)
        dy = jnp.sign(yar_center_y - state.destroyer.y)

        x = state.destroyer.x + self.consts.DESTROYER_SPEED * dx
        y = state.destroyer.y + self.consts.DESTROYER_SPEED * dy

        return dict(destroyer=state.destroyer._replace(x=x, y=y))

    @partial(jax.jit, static_argnums=(0,))
    def _cannon_step(
        self,
        state: YarsRevengeState,
        fire: jnp.ndarray,
        new_energy_shield: jnp.ndarray,
        devour_reset: jnp.ndarray,
    ):
        cannon_exists = state.cannon_exist == 1
        cannon_fired = state.cannon_fired == 1

        # Cannon Calculations
        cannon_hit_boundary = self._check_entity_boundary(state.cannon)

        cannon_shield = self._check_energy_shield_collusion(
            state.cannon, state.energy_shield, state.energy_shield_state
        )
        cannon_hit_shield = jnp.any(cannon_shield)

        new_energy_shield_state = jnp.where(
            cannon_exists & cannon_shield, False, new_energy_shield
        )

        new_cannon_exists = jnp.logical_or(
            jnp.logical_and(~cannon_exists, devour_reset),
            jnp.logical_and(
                cannon_exists,
                jnp.logical_and(~cannon_hit_boundary, ~cannon_hit_shield),
            ),
        )

        cannon_dx, _ = Direction.to_delta(
            Direction.to_flags(state.cannon.direction),
            self.consts.CANNON_SPEED,
        )

        x = jnp.clip(
            jnp.where(
                cannon_fired,
                state.cannon.x + cannon_dx,
                jnp.array(1).astype(jnp.float32),
            ),
            0,
            self.consts.WIDTH,
        )

        y = jnp.clip(
            jnp.where(
                cannon_fired,
                state.cannon.y,
                self._get_entity_position(state.yar, Direction._CENTER)[1]
                - (state.cannon.h // 2),
            ),
            0,
            self.consts.HEIGHT,
        )

        new_cannon_fired = jnp.logical_or(
            jnp.logical_and(cannon_exists, cannon_fired),
            jnp.logical_and(cannon_exists, fire),
        )

        return (
            dict(
                cannon_exist=new_cannon_exists.astype(jnp.int32),
                cannon_fired=new_cannon_fired.astype(jnp.int32),
                cannon=state.cannon._replace(x=x, y=y),
                energy_shield_state=new_energy_shield_state,
            ),
            cannon_exists,
            cannon_fired,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _energy_missile_step(
        self,
        state: YarsRevengeState,
        fire: jnp.ndarray,
        new_energy_shield: jnp.ndarray,
        yar_neutral: jnp.ndarray,
        cannon_exists: jnp.ndarray,
        cannon_fired: jnp.ndarray,
    ):
        em_exists = state.energy_missile_exist == 1

        # Energy Missile
        energy_missile_hit_boundary = self._check_entity_boundary(state.energy_missile)

        # Energy Missile Collusion
        missile_shield = self._check_energy_shield_collusion(
            state.energy_missile, state.energy_shield, state.energy_shield_state
        )
        missile_hit_shield = jnp.any(missile_shield)

        missile_adj_mask = (
            jax.scipy.signal.convolve(
                missile_shield, self.consts.MISSILE_HIT_KERNEL, mode="same"
            )
            > 0
        )

        new_energy_shield_state = jnp.where(
            em_exists & missile_adj_mask, False, new_energy_shield
        )

        new_em_exists = jnp.logical_or(
            jnp.logical_and(
                jnp.logical_and(
                    ~em_exists,
                    jnp.logical_or(
                        ~cannon_exists, jnp.logical_and(cannon_exists, cannon_fired)
                    ),
                ),
                jnp.logical_and(fire, ~yar_neutral),
            ),
            jnp.logical_and(
                em_exists,
                jnp.logical_and(~energy_missile_hit_boundary, ~missile_hit_shield),
            ),
        )

        new_energy_missile_direction = jnp.where(
            em_exists, state.energy_missile.direction, state.yar.direction
        )

        energy_missile_dx, energy_missile_dy = Direction.to_delta(
            Direction.to_flags(state.energy_missile.direction),
            self.consts.ENERGY_MISSILE_SPEED,
        )

        new_energy_missile_x = jnp.clip(
            jnp.where(
                em_exists,
                state.energy_missile.x + energy_missile_dx,
                self._get_entity_position(state.yar, Direction._CENTER)[0],
            ),
            0,
            self.consts.WIDTH,
        )

        new_energy_missile_y = jnp.clip(
            jnp.where(
                em_exists,
                state.energy_missile.y + energy_missile_dy,
                self._get_entity_position(state.yar, Direction._CENTER)[1],
            ),
            0,
            self.consts.HEIGHT,
        )

        return dict(
            energy_missile_exist=new_em_exists.astype(jnp.int32),
            energy_missile=state.energy_missile._replace(
                x=new_energy_missile_x,
                y=new_energy_missile_y,
                direction=new_energy_missile_direction,
            ),
            energy_shield_state=new_energy_shield_state,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _swirl_step(self, state: YarsRevengeState):
        # Swirl Calculation
        swirl_exists = state.swirl_exist == 1
        swirl_fired = jnp.logical_or(state.swirl_dx != 0, state.swirl_dy != 0)

        should_spawn_swirl = state.step_counter % self.consts.SWIRL_PER_STEP == 0
        swirl_hit_boundary = self._check_entity_boundary(state.swirl)

        new_swirl_exists = jnp.logical_or(
            jnp.logical_and(swirl_exists, ~swirl_hit_boundary),
            jnp.logical_and(
                ~swirl_exists,
                jnp.logical_and(state.step_counter != 0, should_spawn_swirl),
            ),
        )

        should_fire_swirl = state.step_counter % self.consts.SWIRL_FIRE_PER_STEP == 0
        new_swirl_fired = jnp.logical_or(
            jnp.logical_and(swirl_exists, swirl_fired),
            jnp.logical_and(
                swirl_exists, jnp.logical_and(should_fire_swirl, ~swirl_fired)
            ),
        )

        swirl_delta_x = state.yar.x - state.swirl.x
        swirl_delta_y = state.yar.y - state.swirl.y
        swirl_delta = jnp.hypot(swirl_delta_x, swirl_delta_y)

        new_swirl_dx = jnp.where(
            jnp.logical_and(swirl_exists, swirl_fired),
            state.swirl_dx,
            jnp.where(
                new_swirl_fired,
                (self.consts.SWIRL_SPEED * swirl_delta_x / swirl_delta),
                0,
            ),
        )
        new_swirl_dy = jnp.where(
            jnp.logical_and(swirl_exists, swirl_fired),
            state.swirl_dy,
            jnp.where(
                new_swirl_fired,
                (self.consts.SWIRL_SPEED * swirl_delta_y / swirl_delta),
                0,
            ),
        )

        new_swirl_x = jnp.clip(
            jnp.where(
                jnp.logical_and(swirl_exists, swirl_fired),
                state.swirl.x + state.swirl_dx,
                state.qotile.x,
            ),
            0,
            self.consts.WIDTH,
        )

        new_swirl_y = jnp.clip(
            jnp.where(
                jnp.logical_and(swirl_exists, swirl_fired),
                state.swirl.y + state.swirl_dy,
                state.qotile.y,
            ),
            0,
            self.consts.HEIGHT,
        )

        return dict(
            swirl_exist=new_swirl_exists.astype(jnp.int32),
            swirl=state.swirl._replace(x=new_swirl_x, y=new_swirl_y),
            swirl_dx=new_swirl_dx,
            swirl_dy=new_swirl_dy,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _stage_specific_step(
        self, state: YarsRevengeState, energy_shield_state: jnp.ndarray
    ):
        shield_snake_apply = jnp.logical_and(
            state.stage == 1, (state.step_counter % self.consts.SNAKE_FRAME == 0)
        )

        new_energy_shield_state = jnp.where(
            shield_snake_apply,
            self._snake_shift(energy_shield_state),
            energy_shield_state,
        )
        return dict(energy_shield_state=new_energy_shield_state)

    @partial(jax.jit, static_argnums=(0,))
    def _game_ending_calculation(
        self,
        state: YarsRevengeState,
        yar_neutral: jnp.ndarray,
        cannon_exists: jnp.ndarray,
        cannon_fired: jnp.ndarray,
    ):
        yar_destroyer = self._check_entity_collusion(state.yar, state.destroyer)
        yar_destroyer_hits = jnp.logical_and(yar_destroyer, ~yar_neutral)

        qotile_cannon = jnp.logical_and(
            cannon_exists, cannon_fired
        ) & self._check_entity_collusion(state.qotile, state.cannon)
        yar_cannon = jnp.logical_and(
            cannon_exists, cannon_fired
        ) & self._check_entity_collusion(state.yar, state.cannon)

        yar_swirl = self._check_entity_collusion(state.yar, state.swirl)
        yar_swirl_hits = jnp.logical_and(yar_swirl, state.swirl_exist)

        life_lost = yar_destroyer_hits | yar_cannon | yar_swirl_hits
        new_lives = jnp.where(life_lost, state.lives - 1, state.lives)

        game_advance = qotile_cannon

        return dict(lives=new_lives), life_lost, game_advance

    @partial(jax.jit, static_argnums=(0,))
    def _finalize_next_state(
        self,
        new_state: YarsRevengeState,
        life_lost: jnp.ndarray,
        game_advance: jnp.ndarray,
    ):
        branch = jnp.where(
            game_advance,
            0,
            jnp.where(
                life_lost & (new_state.lives == 0),
                1,
                jnp.where(life_lost, 2, 3),
            ),
        )

        return jax.lax.switch(
            branch,
            [
                lambda: (
                    self.construct_initial_state((new_state.stage + 1) % 2)._replace(
                        level=new_state.level + 1
                    )
                ),  # Advance to the next stage
                lambda: self.construct_initial_state(0),  # No lives left, reset
                lambda: self.construct_initial_state(new_state.stage)._replace(
                    level=new_state.level,
                    score=new_state.score,
                    lives=new_state.lives,
                    energy_shield_state=new_state.energy_shield_state,
                ),  # One life lost, continue with the same stage and shield
                lambda: new_state,  # Game continues
            ],
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: YarsRevengeState, action: chex.Array):
        new_state = state

        direction_flags, fire = self._parse_action_flags(action)

        yar_updates, yar_neutral, devour_reset = self._yar_step(state, direction_flags)
        new_state = new_state._replace(**yar_updates)

        qotile_updates = self._qotile_step(state)
        new_state = new_state._replace(**qotile_updates)

        destroyer_updates = self._destroyer_step(state)
        new_state = new_state._replace(**destroyer_updates)

        cannon_updates, cannon_exists, cannon_fired = self._cannon_step(
            state, fire, new_state.energy_shield_state, devour_reset
        )
        new_state = new_state._replace(**cannon_updates)

        em_updates = self._energy_missile_step(
            state,
            fire,
            new_state.energy_shield_state,
            yar_neutral,
            cannon_exists,
            cannon_fired,
        )
        new_state = new_state._replace(**em_updates)

        swirl_updates = self._swirl_step(state)
        new_state = new_state._replace(**swirl_updates)

        stage_specific_updates = self._stage_specific_step(
            state, new_state.energy_shield_state
        )
        new_state = new_state._replace(**stage_specific_updates)

        life_updates, life_lost, game_advance = self._game_ending_calculation(
            state, yar_neutral, cannon_exists, cannon_fired
        )
        new_state = new_state._replace(**life_updates)

        new_state = self._finalize_next_state(new_state, life_lost, game_advance)

        new_state = new_state._replace(step_counter=state.step_counter + 1)

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
                        "x": spaces.Box(
                            low=0.0, high=160.0, shape=(), dtype=jnp.float32
                        ),
                        "y": spaces.Box(
                            low=0.0, high=210.0, shape=(), dtype=jnp.float32
                        ),
                        "w": spaces.Box(low=0, high=160, shape=(), dtype=jnp.float32),
                        "h": spaces.Box(low=0, high=210, shape=(), dtype=jnp.float32),
                        "direction": spaces.Box(
                            low=0, high=7, shape=(), dtype=jnp.float32
                        ),
                    }
                ),
                "qotile": spaces.Dict(
                    {
                        "x": spaces.Box(
                            low=0.0, high=160.0, shape=(), dtype=jnp.float32
                        ),
                        "y": spaces.Box(
                            low=0.0, high=210.0, shape=(), dtype=jnp.float32
                        ),
                        "w": spaces.Box(low=0, high=160, shape=(), dtype=jnp.float32),
                        "h": spaces.Box(low=0, high=210, shape=(), dtype=jnp.float32),
                        "direction": spaces.Box(
                            low=0, high=7, shape=(), dtype=jnp.float32
                        ),
                    }
                ),
                "destroyer": spaces.Dict(
                    {
                        "x": spaces.Box(
                            low=0.0, high=160.0, shape=(), dtype=jnp.float32
                        ),
                        "y": spaces.Box(
                            low=0.0, high=210.0, shape=(), dtype=jnp.float32
                        ),
                        "w": spaces.Box(low=0, high=160, shape=(), dtype=jnp.float32),
                        "h": spaces.Box(low=0, high=210, shape=(), dtype=jnp.float32),
                    }
                ),
                "swirl_exist": spaces.Box(low=0, high=1, shape=(), dtype=jnp.float32),
                "swirl": spaces.Dict(
                    {
                        "x": spaces.Box(
                            low=0.0, high=160.0, shape=(), dtype=jnp.float32
                        ),
                        "y": spaces.Box(
                            low=0.0, high=210.0, shape=(), dtype=jnp.float32
                        ),
                        "w": spaces.Box(low=0, high=160, shape=(), dtype=jnp.float32),
                        "h": spaces.Box(low=0, high=210, shape=(), dtype=jnp.float32),
                    }
                ),
                "swirl_dx": spaces.Box(low=-3.0, high=3.0, shape=(), dtype=jnp.float32),
                "swirl_dy": spaces.Box(low=-3.0, high=3.0, shape=(), dtype=jnp.float32),
                "energy_missile_exist": spaces.Box(
                    low=0, high=1, shape=(), dtype=jnp.float32
                ),
                "energy_missile": spaces.Dict(
                    {
                        "x": spaces.Box(
                            low=0.0, high=160.0, shape=(), dtype=jnp.float32
                        ),
                        "y": spaces.Box(
                            low=0.0, high=210.0, shape=(), dtype=jnp.float32
                        ),
                        "w": spaces.Box(low=0, high=160, shape=(), dtype=jnp.float32),
                        "h": spaces.Box(low=0, high=210, shape=(), dtype=jnp.float32),
                        "direction": spaces.Box(
                            low=0, high=7, shape=(), dtype=jnp.float32
                        ),
                    }
                ),
                "cannon_exist": spaces.Box(low=0, high=1, shape=(), dtype=jnp.float32),
                "cannon_fired": spaces.Box(low=0, high=1, shape=(), dtype=jnp.float32),
                "cannon": spaces.Dict(
                    {
                        "x": spaces.Box(
                            low=0.0, high=160.0, shape=(), dtype=jnp.float32
                        ),
                        "y": spaces.Box(
                            low=0.0, high=210.0, shape=(), dtype=jnp.float32
                        ),
                        "w": spaces.Box(low=0, high=160, shape=(), dtype=jnp.float32),
                        "h": spaces.Box(low=0, high=210, shape=(), dtype=jnp.float32),
                        "direction": spaces.Box(
                            low=0, high=7, shape=(), dtype=jnp.float32
                        ),
                    }
                ),
                "energy_shield": spaces.Dict(
                    {
                        "x": spaces.Box(
                            low=0.0, high=160.0, shape=(), dtype=jnp.float32
                        ),
                        "y": spaces.Box(
                            low=0.0, high=210.0, shape=(), dtype=jnp.float32
                        ),
                        "w": spaces.Box(low=0, high=160, shape=(), dtype=jnp.float32),
                        "h": spaces.Box(low=0, high=210, shape=(), dtype=jnp.float32),
                    }
                ),
                "energy_shield_state": spaces.Box(
                    low=0, high=1, shape=(16, 8), dtype=jnp.float32
                ),
                "lives": spaces.Box(low=0, high=4, shape=(), dtype=jnp.float32),
            }
        )

    def image_space(self):
        return spaces.Box(low=0, high=255, shape=(210, 160, 3), dtype=jnp.uint8)

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: YarsRevengeState) -> YarsRevengeObservation:
        return YarsRevengeObservation(
            yar=state.yar,
            qotile=state.qotile,
            destroyer=state.destroyer,
            swirl_exist=state.swirl_exist,
            swirl=state.swirl,
            swirl_dx=state.swirl_dx,
            swirl_dy=state.swirl_dy,
            energy_missile_exist=state.energy_missile_exist,
            energy_missile=state.energy_missile,
            cannon_exist=state.cannon_exist,
            cannon_fired=state.cannon_fired,
            cannon=state.cannon,
            energy_shield=state.energy_shield,
            energy_shield_state=state.energy_shield_state,
            lives=state.lives,
        )

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs):
        return jnp.concatenate(
            [
                obs.yar.x.flatten().astype(jnp.float32),
                obs.yar.y.flatten().astype(jnp.float32),
                obs.yar.w.flatten().astype(jnp.float32),
                obs.yar.h.flatten().astype(jnp.float32),
                obs.yar.direction.flatten().astype(jnp.float32),
                obs.qotile.x.flatten().astype(jnp.float32),
                obs.qotile.y.flatten().astype(jnp.float32),
                obs.qotile.w.flatten().astype(jnp.float32),
                obs.qotile.h.flatten().astype(jnp.float32),
                obs.qotile.direction.flatten().astype(jnp.float32),
                obs.destroyer.x.flatten().astype(jnp.float32),
                obs.destroyer.y.flatten().astype(jnp.float32),
                obs.destroyer.w.flatten().astype(jnp.float32),
                obs.destroyer.h.flatten().astype(jnp.float32),
                obs.swirl_exist.flatten().astype(jnp.float32),
                obs.swirl.x.flatten().astype(jnp.float32),
                obs.swirl.y.flatten().astype(jnp.float32),
                obs.swirl.w.flatten().astype(jnp.float32),
                obs.swirl.h.flatten().astype(jnp.float32),
                obs.swirl_dx.flatten().astype(jnp.float32),
                obs.swirl_dy.flatten().astype(jnp.float32),
                obs.energy_missile_exist.flatten().astype(jnp.float32),
                obs.energy_missile.x.flatten().astype(jnp.float32),
                obs.energy_missile.y.flatten().astype(jnp.float32),
                obs.energy_missile.w.flatten().astype(jnp.float32),
                obs.energy_missile.h.flatten().astype(jnp.float32),
                obs.energy_missile.direction.flatten().astype(jnp.float32),
                obs.cannon_exist.flatten().astype(jnp.float32),
                obs.cannon_fired.flatten().astype(jnp.float32),
                obs.cannon.x.flatten().astype(jnp.float32),
                obs.cannon.y.flatten().astype(jnp.float32),
                obs.cannon.w.flatten().astype(jnp.float32),
                obs.cannon.h.flatten().astype(jnp.float32),
                obs.cannon.direction.flatten().astype(jnp.float32),
                obs.energy_shield.x.flatten().astype(jnp.float32),
                obs.energy_shield.y.flatten().astype(jnp.float32),
                obs.energy_shield.w.flatten().astype(jnp.float32),
                obs.energy_shield.h.flatten().astype(jnp.float32),
                obs.energy_shield_state.flatten().astype(jnp.float32),
                obs.lives.flatten().astype(jnp.float32),
            ]
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state, all_rewards: Optional[Array] = None):
        return YarsRevengeInfo(time=state.step_counter, current_level=state.level)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state, state):
        return state.score - previous_state.score

    @partial(jax.jit, static_argnums=(0,))
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
            {
                "name": "swirl",
                "type": "group",
                "files": [
                    "swirl_0.npy",
                    "swirl_1.npy",
                    "swirl_2.npy",
                    "swirl_3.npy",
                ],
            },
            {
                "name": "energy_shield",
                "type": "procedural",
                "data": jnp.array(
                    self.consts.ENERGY_SHIELD_COLOR + (255,), dtype=jnp.uint8
                ).reshape(1, 1, 4),
            },
            {
                "name": "neutral_zone",
                "type": "procedural",
                "data": jnp.array(
                    self.consts.NEUTRAL_ZONE_COLOR + (255,), dtype=jnp.uint8
                ).reshape(1, 1, 4),
            },
        ]

    def get_animation_idx(
        self,
        step: jnp.ndarray,
        group: jnp.ndarray | int,
        duration: int,
        group_item_count: int,
    ):
        return group * 2 + (
            jnp.floor(step / duration).astype(jnp.int32) % group_item_count
        )

    def render(self, state: YarsRevengeState):
        raster = self.jr.create_object_raster(self.BACKGROUND)

        neutral_zone_mask = jnp.full(
            (self.consts.NEUTRAL_ZONE_SIZE[1], self.consts.NEUTRAL_ZONE_SIZE[0]),
            self.COLOR_TO_ID[self.consts.NEUTRAL_ZONE_COLOR],
        )
        raster = self.jr.render_at(
            raster, state.neutral_zone.x, state.neutral_zone.y, neutral_zone_mask
        )

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

        raster = self.jr.render_grid_inverse(
            raster=raster,
            grid_state=state.energy_shield_state.astype(jnp.int32),
            grid_origin=(state.energy_shield.x, state.energy_shield.y),
            cell_size=(self.consts.ENERGY_CELL_WIDTH, self.consts.ENERGY_CELL_HEIGHT),
            color_map=jnp.array(
                [
                    self.jr.TRANSPARENT_ID,
                    self.COLOR_TO_ID[self.consts.ENERGY_SHIELD_COLOR],
                ]
            ),
        )

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

        cannon_mask = jnp.where(
            (state.step_counter // self.consts.CANNON_MOVEMENT_FRAME) % 2 == 0,
            jnp.ones((self.consts.CANNON_SIZE[1], self.consts.CANNON_SIZE[0])),
            jnp.concatenate(
                [
                    jnp.ones(
                        (self.consts.CANNON_SIZE[1], self.consts.CANNON_SIZE[0] // 2)
                    ),
                    jnp.full(
                        (self.consts.CANNON_SIZE[1], self.consts.CANNON_SIZE[0] // 2),
                        self.jr.TRANSPARENT_ID,
                    ),
                ],
                axis=1,
            ),
        )

        raster = jnp.where(
            state.cannon_exist,
            self.jr.render_at(raster, state.cannon.x, state.cannon.y, cannon_mask),
            raster,
        )

        qotile_mask = self.SHAPE_MASKS["qotile"]

        swirl_idx = self.get_animation_idx(
            state.step_counter, 0, self.consts.SWIRL_MOVEMENT_FRAME, 4
        )
        swirl_mask = self.SHAPE_MASKS["swirl"][swirl_idx]

        raster = jnp.where(
            state.swirl_exist,
            self.jr.render_at(raster, state.swirl.x, state.swirl.y, swirl_mask),
            self.jr.render_at(raster, state.qotile.x, state.qotile.y, qotile_mask),
        )

        return self.jr.render_from_palette(raster, self.PALETTE)
