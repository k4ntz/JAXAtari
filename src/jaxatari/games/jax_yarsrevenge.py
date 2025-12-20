import os
from enum import IntEnum
from functools import partial
from typing import Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from jaxatari import spaces
from jaxatari.environment import JAXAtariAction as Action
from jaxatari.environment import JaxEnvironment
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils


class Direction(IntEnum):
    """
    Enum that encodes the 8 compass directions plus a special “center”
    (no movement) value. The numeric values are used only for arithmetic
    operations and also storing directions as integer.
    """

    _CENTER = -1  # no direction / stationary
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
    def from_flags(
        flags: Tuple[
            bool | jnp.ndarray,  # up flag
            bool | jnp.ndarray,  # down flag
            bool | jnp.ndarray,  # left flag
            bool | jnp.ndarray,  # right flag
        ],
    ):
        """
        Convert a set of boolean flags (up, down, left, right) into a single `Direction` enum value.
        """
        up, down, left, right = flags
        return jnp.select(
            [
                ~right & ~up & ~down & ~left,  # no flag: centre
                right & ~up & ~down & ~left,  # right only
                right & up & ~down & ~left,  # up-right
                ~right & up & ~down & ~left,  # up
                ~right & up & ~down & left,  # up-left
                ~right & ~up & ~down & left,  # left only
                ~right & ~up & down & left,  # down-left
                ~right & ~up & down & ~left,  # down
                right & ~up & down & ~left,  # down-right
            ],
            [
                Direction._CENTER,  # centre / no movement
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
        """
        Inverse of `from_flags`. Convert a single `Direction` enum into four boolean flags.
        """
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
        """
        Return the opposite direction.  The formula `(d + 4) % 8` works
        because directions are laid out in a circle, adding 4 (half of 8)
        rotates to the opposite side.  The special value `_CENTER` stays
        unchanged.
        """
        return jnp.where(
            direction == Direction._CENTER,
            jnp.array(Direction._CENTER).astype(jnp.int32),
            jnp.array((direction + 4) % 8).astype(jnp.int32),
        )

    @staticmethod
    @jax.jit
    def to_delta(
        flags: Tuple[
            bool | jnp.ndarray,  # up flag
            bool | jnp.ndarray,  # down flag
            bool | jnp.ndarray,  # left flag
            bool | jnp.ndarray,  # right flag
        ],
        speed: float | jnp.ndarray,
    ):
        """
        Convert direction flags into a displacement vector (dx, dy)
        that can be added to an entity's current coordinates.

        * dx  - left/right movement.  `-speed` if left flag is true.
        * dy  - up/down movement.  `-speed` if up flag is true.
        """
        up, down, left, right = flags
        dx = jnp.where(left, -speed, jnp.where(right, speed, 0.0))
        dy = jnp.where(up, -speed, jnp.where(down, speed, 0.0))
        return dx, dy


# Game state enumerations


class YarState(IntEnum):
    """
    Tracks whether the player “Yar” is currently moving or idle.
    """

    STEADY = 0
    MOVING = 1


class Entity(NamedTuple):
    """Simple rectangular entity with only position + size."""

    x: jnp.ndarray  # float32
    y: jnp.ndarray  # float32
    w: jnp.ndarray  # int32
    h: jnp.ndarray  # int32


class DirectionEntity(NamedTuple):
    """
    Same as `Entity` but with an additional direction field.
    All direction values use the `Direction` enum.
    """

    x: jnp.ndarray
    y: jnp.ndarray
    w: jnp.ndarray
    h: jnp.ndarray
    direction: jnp.ndarray


class YarsRevengeConstants(NamedTuple):
    """
    Game-wide constants for Yar's Revenge.
    """

    # Game world size
    WIDTH: int = 160  # horizontal resolution (pixels)
    HEIGHT: int = 210  # vertical resolution (pixels)

    # Player / general gameplay constants
    INITIAL_LIVES: int = 4  # lives at start of game
    DEVOUR_THRESHOLD: int = 5  # number of cells that must be hit to devour

    # Entity sizes (width, height).
    YAR_SIZE: Tuple[int, int] = (8, 16)  # player sprite size
    QOTILE_SIZE: Tuple[int, int] = (8, 18)  # moving enemy
    NEUTRAL_ZONE_SIZE: Tuple[int, int] = (28, HEIGHT)
    SWIRL_SIZE: Tuple[int, int] = (8, 16)
    DESTROYER_SIZE: Tuple[int, int] = (4, 2)
    ENERGY_MISSILE_SIZE: Tuple[int, int] = (1, 2)
    CANNON_SIZE: Tuple[int, int] = (8, 8)

    # Special coordinates
    QOTILE_MIN_Y: int = 60  # lower bound for qotile vertical movement
    QOTILE_MAX_Y: int = 150  # upper bound for qotile vertical movement
    NEUTRAL_ZONE_POSITION: Tuple[int, int] = (50, 0)  # top-left corner of neutral zone

    # Timers / periodic behaviour
    SWIRL_PER_STEP: int = 1000  # steps between swirl spawns
    SWIRL_FIRE_PER_STEP: int = 250  # steps between swarm firing

    # Speeds (pixels per frame)
    QOTILE_SPEED = 0.5  # slow vertical oscillation of qotile
    YAR_SPEED = 2.0  # horizontal move speed
    YAR_DIAGONAL_SPEED = 1.0  # diagonal movement is slower to preserve overall speed
    SWIRL_SPEED = 3.0  # swirl target following speed
    DESTROYER_SPEED = 0.125  # very slow chase of Yar
    ENERGY_MISSILE_SPEED = 4.0  # fast missile from cannon or YAR
    CANNON_SPEED = 2.0  # horizontal cannon movement
    SNAKE_FRAME = 4  # snake shift applied every 4 steps in stage-1

    # Animation frame intervals
    STEADY_YAR_MOVEMENT_FRAME = (
        4  # Movement animation change interval for Yar (no action)
    )
    MOVING_YAR_MOVEMENT_FRAME = 1  # Movement animation change interval for Yar (moving)
    CANNON_MOVEMENT_FRAME = 8  # Animation change interval for cannon
    SWIRL_MOVEMENT_FRAME = 2  # Swirl sprite cycle

    # Colour palette (RGB)
    ENERGY_SHIELD_COLOR: Tuple[int, int, int] = (163, 57, 21)
    NEUTRAL_ZONE_COLOR: Tuple[int, int, int] = (20, 20, 20)

    # Energy shield grid - each cell is `ENERGY_CELL_WIDTH` x
    # `ENERGY_CELL_HEIGHT`.
    ENERGY_CELL_WIDTH = 4
    ENERGY_CELL_HEIGHT = 8

    # 16 rows × 8 columns grid (top-to-bottom, left-to-right)
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

    # Missile hit kernel used for a 3×3 convolution that detects
    # neighbouring hits in a plus shape.
    MISSILE_HIT_KERNEL = jnp.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=jnp.int32,
    )


class YarsRevengeState(NamedTuple):
    """
    Immutable representation of the complete game state.
    """

    # Generic counters / score
    step_counter: jnp.ndarray  # int32  current simulation frame
    score: jnp.ndarray  # int32  accumulated score
    lives: jnp.ndarray  # int32  remaining lives
    stage: jnp.ndarray  # int32  current game stage (0/1)

    # Player and enemy entities
    yar: DirectionEntity  # player
    yar_state: jnp.ndarray  # YarState (int)
    yar_devour_count: jnp.ndarray  # consecutive cells hit by Yar
    qotile: DirectionEntity  # moving enemy
    destroyer: Entity  # slow pursuer

    # Swirl / projectile entities
    swirl_exist: jnp.ndarray  # bool (int32) whether the swirl is on screen
    swirl: Entity  # current position/size of swirl
    swirl_dx: jnp.ndarray  # x velocity (float)
    swirl_dy: jnp.ndarray  # y velocity
    energy_missile_exist: jnp.ndarray  # bool
    energy_missile: DirectionEntity  # missile entity
    cannon_exist: jnp.ndarray  # bool
    cannon_fired: jnp.ndarray  # bool
    cannon: DirectionEntity  # cannon entity

    # Shield + neutral zone
    energy_shield: Entity  # top-left corner of the shield grid
    energy_shield_state: jnp.ndarray  # 16×8 boolean array - live cells
    neutral_zone: Entity  # static area that Yar cannot enter


class YarsRevengeObservation(NamedTuple):
    """
    The part of the state that is returned to the agent.
    """

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
    time: jnp.ndarray  # step counter


class JaxYarsRevenge(
    JaxEnvironment[
        YarsRevengeState,
        YarsRevengeObservation,
        YarsRevengeInfo,
        YarsRevengeConstants,
    ]
):

    def __init__(
        self,
        consts: Optional[YarsRevengeConstants] = None,
        reward_funcs: Optional[list[Callable]] = None,
    ):
        """
        Construct a new environment with a renderer instance for visualisation.
        """
        consts = consts or YarsRevengeConstants()
        super().__init__(consts)
        self.renderer = YarsRevengeRenderer(self.consts)

    @partial(jax.jit, static_argnums=(0,))
    def construct_initial_state(self, stage):
        """
        Build the initial game state for a given *stage* (0 or 1).
        The returned state is an immutable `YarsRevengeState`.
        """
        return YarsRevengeState(
            step_counter=jnp.array(0).astype(jnp.int32),
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
        """
        Return the initial observation and state.
        """
        state = self.construct_initial_state(0)
        initial_obs = self._get_observation(state)

        return initial_obs, state

    @staticmethod
    @jax.jit
    def _parse_action_flags(action: jnp.ndarray) -> Tuple[
        Tuple[
            bool | jnp.ndarray,  # up flag
            bool | jnp.ndarray,  # down flag
            bool | jnp.ndarray,  # left flag
            bool | jnp.ndarray,  # right flag
        ],
        jnp.ndarray,
    ]:
        """
        Convert the discrete action index into four movement flags and a fire flag.
        """
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
        """
        Returns the point (x, y) of the entity with given `direction` parameter.
        """
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
        """
        Move an entity by the given deltas and clip it inside screen bounds.
        `wrap_y` indicates whether vertical movement should wrap around.
        """
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
        """
        Return a boolean mask if *any* side of the entity touches a screen border.
        """
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
        """
        Return True if two axis-aligned bounding boxes overlap.
        It uses the `get_entity_position` helper to compute corners according to
        each entity's direction, which ensures correct collision even when a sprite
        is flipped.
        """
        # Entity A bounds in its own coordinate system
        a_up, a_left = JaxYarsRevenge._get_entity_position(a, Direction.UPLEFT)
        a_down, a_right = JaxYarsRevenge._get_entity_position(a, Direction.DOWNRIGHT)

        # Entity B bounds
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
        """
        Compute a boolean mask the same shape as *shield_state* that indicates
        which shield cells intersect the given entity.

        The algorithm constructs an array of all cell corners, then checks
        for overlap with the entity's bounding box.  The result can be
        used to compute damage or devouring.
        """
        rows, cols = shield_state.shape

        # Entity bounds
        e_left, e_up = JaxYarsRevenge._get_entity_position(entity, Direction.UPLEFT)
        e_right, e_down = JaxYarsRevenge._get_entity_position(
            entity, Direction.DOWNRIGHT
        )

        # Pre-compute cell corner coordinates in a vectorised manner
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
        """
        Shift the shield cells in a snake-like pattern used in stage 1.
        The algorithm creates an index mapping that flips every other row and then rolls by 1.
        It is implemented purely with JAX operations so it can be compiled.
        """
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
    def _yar_step(
        self,
        state: YarsRevengeState,
        direction_flags: Tuple[
            bool | jnp.ndarray,  # up flag
            bool | jnp.ndarray,  # down flag
            bool | jnp.ndarray,  # left flag
            bool | jnp.ndarray,  # right flag
        ],
    ):
        """
        Handle the player movement and all related effects:

            * Update position and speed
            * Detect shield collisions (including devouring)
            * Detect qotile collision
            * Compute life loss at the end of the step

        Returns a tuple that contains:
          - A dictionary with updated fields (`yar`, `yar_state`,
            `yar_devour_count`, `energy_shield_state`)
          - Boolean flag if Yar entered the neutral zone (life lost)
          - Boolean flag indicating whether devouring was reset
        """
        direction = Direction.from_flags(direction_flags)  # current desired direction
        yar_moving = direction != Direction._CENTER
        new_yar_direction = jax.lax.select(yar_moving, direction, state.yar.direction)
        new_yar_state = jax.lax.select(yar_moving, YarState.MOVING, YarState.STEADY)

        # Diagonal speed handling
        yar_diagonal = (direction_flags[0] | direction_flags[1]) & (
            direction_flags[2] | direction_flags[3]
        )
        yar_speed = jax.lax.select(
            yar_diagonal, self.consts.YAR_DIAGONAL_SPEED, self.consts.YAR_SPEED
        )
        delta_yar_x, delta_yar_y = Direction.to_delta(direction_flags, yar_speed)

        # New position - note wrap on y axis for the player
        new_yar_x, new_yar_y = self._move_entity(
            state.yar,
            delta_yar_x,
            delta_yar_y,
            wrap_y=True,
        )

        new_yar_entity = state.yar._replace(
            x=new_yar_x, y=new_yar_y, direction=new_yar_direction
        )

        # Shield collision - shift left if a cell is hit
        yar_shield_collusion = self._check_energy_shield_collusion(
            new_yar_entity, state.energy_shield, state.energy_shield_state
        )
        yar_hit_shield = jnp.any(yar_shield_collusion)

        # Shift Yar left by one cell width if it hits the shield
        new_yar_x = jnp.where(
            yar_hit_shield,
            new_yar_x - self.consts.ENERGY_CELL_WIDTH,
            new_yar_x,
        )
        new_yar_entity = new_yar_entity._replace(x=new_yar_x)

        # Qotile collision detection
        yar_qotile_collusion = self._check_entity_collusion(
            new_yar_entity, state.qotile
        )
        new_yar_devour_count = jnp.where(
            yar_hit_shield & yar_moving | yar_qotile_collusion,
            state.yar_devour_count + 1,
            state.yar_devour_count,
        )

        # Devouring logic to count collusions to remove a cell
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

        # Reset devour counter after a successful devour
        new_yar_devour_count = jnp.where(
            devour_reset,
            0,
            new_yar_devour_count,
        )

        # Neutral zone detection
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
        """
        Move the qotile back and forth between QOTILE_MIN_Y and
        QOTILE_MAX_Y. The sprite reverses direction on each boundary hit.
        Additionally the shield is moved vertically to follow the qotile.
        """
        # Boundary detection for vertical movement
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

        # Apply the vertical movement
        _, new_qotile_y = self._move_entity(
            state.qotile,
            _,
            delta_qotile_y,
            min_y=self.consts.QOTILE_MIN_Y,
            max_y=self.consts.QOTILE_MAX_Y,
        )

        # New shield Y following the qotile’s vertical center
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
        """
        Handles the destroyer logic, which slowly moves towards the player's current position.
        """
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
        """
        Handles the cannon spawn, movement, firing and collision with the energy shield.

        * `fire` indicates that the player wants to shoot.
          The cannon will appear only after a devour reset.
        * `new_energy_shield` is passed so that the shield can be
          updated by an active missile before it hits the cannon again.
        """
        cannon_exists = state.cannon_exist == 1
        cannon_fired = state.cannon_fired == 1
        cannon_hit_boundary = self._check_entity_boundary(state.cannon)

        # Collision with shield, removing the cell from the shield
        cannon_shield = self._check_energy_shield_collusion(
            state.cannon, state.energy_shield, state.energy_shield_state
        )
        cannon_hit_shield = jnp.any(cannon_shield)

        new_energy_shield_state = jnp.where(
            cannon_exists & cannon_shield, False, new_energy_shield
        )

        # Cannon should exist if it was not there before and we just performed a devour reset
        #  or it already exists and has not hit the shield/boundary
        new_cannon_exists = jnp.logical_or(
            jnp.logical_and(~cannon_exists, devour_reset),
            jnp.logical_and(
                cannon_exists,
                jnp.logical_and(~cannon_hit_boundary, ~cannon_hit_shield),
            ),
        )

        # Compute horizontal velocity (constant speed)
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
        """
        The energy missile is a fast projectile that can be fired from
        either the player or the cannon.  It interacts with the shield
        (via convolution) and can destroy the shield cells.

        `yar_neutral` indicates if the player has entered the neutral zone;
        if so, a newly spawned missile will be prevented.
        """
        em_exists = state.energy_missile_exist == 1
        energy_missile_hit_boundary = self._check_entity_boundary(state.energy_missile)

        # Collision with shield
        missile_shield = self._check_energy_shield_collusion(
            state.energy_missile, state.energy_shield, state.energy_shield_state
        )
        missile_hit_shield = jnp.any(missile_shield)

        # A 3x3 convolution is used to detect neighbouring cells that
        # should be destroyed when a missile passes over them.
        missile_adj_mask = (
            jax.scipy.signal.convolve(
                missile_shield, self.consts.MISSILE_HIT_KERNEL, mode="same"
            )
            > 0
        )

        new_energy_shield_state = jnp.where(
            em_exists & missile_adj_mask, False, new_energy_shield
        )

        # Missile existence logic, a missile appears if the player fires
        # and has not entered neutral zone; it also continues until it hits
        # shield/boundary.
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

        # Direction of the missile, if it already exists keep its
        # original direction, else inherit from the player’s current direction.
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
        """
        The swirl entity is a homing projectile that follows the player.
        It spawns every `SWIRL_PER_STEP` frames and starts moving after
        `SWIRL_FIRE_PER_STEP`.  The velocity components are normalised so
        that the speed equals `SWIRL_SPEED`.
        """
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

        # Velocity components, point from swirl to player
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

        # New position, where the swirl starts at the qotile
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
        """
        Stage-specific logic, only stage 1 performs the snake shift of the
        shield at every `SNAKE_FRAME` interval.
        """
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
        """
        Evaluate the end-of-step conditions:

            * Whether a life is lost
            * Whether the stage should advance (qotile hits cannon)
            * Update lives accordingly

        Returns an updated `lives` field, a bool for life loss and a
        bool that indicates whether we should advance to the next stage.
        """
        # Collision of Yar with Destroyer
        yar_destroyer = self._check_entity_collusion(state.yar, state.destroyer)
        yar_destroyer_hits = jnp.logical_and(yar_destroyer, ~yar_neutral)

        # Collision of Qotile with cannon (stage progression)
        qotile_cannon = jnp.logical_and(
            cannon_exists, cannon_fired
        ) & self._check_entity_collusion(state.qotile, state.cannon)
        yar_cannon = jnp.logical_and(
            cannon_exists, cannon_fired
        ) & self._check_entity_collusion(state.yar, state.cannon)

        # Collision of Yar with swirl
        yar_swirl = self._check_entity_collusion(state.yar, state.swirl)
        yar_swirl_hits = jnp.logical_and(yar_swirl, state.swirl_exist)

        life_lost = yar_destroyer_hits | yar_cannon | yar_swirl_hits
        new_lives = jnp.where(life_lost, state.lives - 1, state.lives)

        # Stage advancement occurs only when the qotile hits the cannon
        game_advance = qotile_cannon

        return dict(lives=new_lives), life_lost, game_advance

    @partial(jax.jit, static_argnums=(0,))
    def _finalize_next_state(
        self,
        new_state: YarsRevengeState,
        life_lost: jnp.ndarray,
        game_advance: jnp.ndarray,
    ):
        """
        Branch the next state based on the outcome:

            * Advance stage if `game_advance` (new stage)
            * Reset when lives are exhausted
            * Continue same stage with shield reset after a life loss
            * Otherwise keep running

        The new state is built by re-using `construct_initial_state`
        and replacing only the fields that must change.
        """
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
                    self.construct_initial_state((new_state.stage + 1) % 2)
                ),  # Advance to the next stage
                lambda: self.construct_initial_state(0),  # No lives left, reset
                lambda: self.construct_initial_state(new_state.stage)._replace(
                    score=new_state.score,
                    lives=new_state.lives,
                    energy_shield_state=new_state.energy_shield_state,
                ),  # One life lost, continue with the same stage and shield
                lambda: new_state,  # Game continues
            ],
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: YarsRevengeState, action: jnp.ndarray):
        """
        Full environment step, it calls all sub-steps in the correct order,
        updates the immutable state and returns the observation, reward and
        termination flag.  The returned observation is a `YarsRevengeObservation`
        instance that contains everything the agent needs to decide on the
        next action.
        """
        new_state = state

        # Parse input action into flags
        direction_flags, fire = self._parse_action_flags(action)

        # Player
        yar_updates, yar_neutral, devour_reset = self._yar_step(state, direction_flags)
        new_state = new_state._replace(**yar_updates)

        # Qotile
        qotile_updates = self._qotile_step(state)
        new_state = new_state._replace(**qotile_updates)

        # Destroyer
        destroyer_updates = self._destroyer_step(state)
        new_state = new_state._replace(**destroyer_updates)

        # Cannon
        cannon_updates, cannon_exists, cannon_fired = self._cannon_step(
            state, fire, new_state.energy_shield_state, devour_reset
        )
        new_state = new_state._replace(**cannon_updates)

        # Energy missile
        em_updates = self._energy_missile_step(
            state,
            fire,
            new_state.energy_shield_state,
            yar_neutral,
            cannon_exists,
            cannon_fired,
        )
        new_state = new_state._replace(**em_updates)

        # Swirl
        swirl_updates = self._swirl_step(state)
        new_state = new_state._replace(**swirl_updates)

        # Stage specific
        stage_specific_updates = self._stage_specific_step(
            state, new_state.energy_shield_state
        )
        new_state = new_state._replace(**stage_specific_updates)

        # Game ending (lives and advancement)
        life_updates, life_lost, game_advance = self._game_ending_calculation(
            state, yar_neutral, cannon_exists, cannon_fired
        )
        new_state = new_state._replace(**life_updates)

        # Finalise next state
        new_state = self._finalize_next_state(new_state, life_lost, game_advance)

        # Increment step counter
        new_state = new_state._replace(step_counter=state.step_counter + 1)

        # Observation / reward / done flag
        observation = self._get_observation(new_state)
        reward = self._get_reward(state, new_state)
        done = self._get_done(new_state)
        info = self._get_info(new_state)
        return observation, new_state, reward, done, info

    def render(self, state):
        """Return a uint8 image array with shape (210, 160, 3)."""
        return self.renderer.render(state)

    def action_space(self):
        """Return the available number of actions."""
        return spaces.Discrete(18)

    def observation_space(self):
        """
        Return a `spaces.Dict` that matches `YarsRevengeObservation`.
        """
        return spaces.Dict(
            {
                "yar": spaces.Dict(
                    {
                        "x": spaces.Box(
                            low=0.0, high=self.consts.WIDTH, shape=(), dtype=jnp.float32
                        ),
                        "y": spaces.Box(
                            low=0.0,
                            high=self.consts.HEIGHT,
                            shape=(),
                            dtype=jnp.float32,
                        ),
                        "w": spaces.Box(
                            low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.float32
                        ),
                        "h": spaces.Box(
                            low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.float32
                        ),
                        "direction": spaces.Box(
                            low=0, high=7, shape=(), dtype=jnp.float32
                        ),
                    }
                ),
                "qotile": spaces.Dict(
                    {
                        "x": spaces.Box(
                            low=0.0, high=self.consts.WIDTH, shape=(), dtype=jnp.float32
                        ),
                        "y": spaces.Box(
                            low=0.0,
                            high=self.consts.HEIGHT,
                            shape=(),
                            dtype=jnp.float32,
                        ),
                        "w": spaces.Box(
                            low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.float32
                        ),
                        "h": spaces.Box(
                            low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.float32
                        ),
                        "direction": spaces.Box(
                            low=0, high=7, shape=(), dtype=jnp.float32
                        ),
                    }
                ),
                "destroyer": spaces.Dict(
                    {
                        "x": spaces.Box(
                            low=0.0, high=self.consts.WIDTH, shape=(), dtype=jnp.float32
                        ),
                        "y": spaces.Box(
                            low=0.0,
                            high=self.consts.HEIGHT,
                            shape=(),
                            dtype=jnp.float32,
                        ),
                        "w": spaces.Box(
                            low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.float32
                        ),
                        "h": spaces.Box(
                            low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.float32
                        ),
                    }
                ),
                "swirl_exist": spaces.Box(low=0, high=1, shape=(), dtype=jnp.float32),
                "swirl": spaces.Dict(
                    {
                        "x": spaces.Box(
                            low=0.0, high=self.consts.WIDTH, shape=(), dtype=jnp.float32
                        ),
                        "y": spaces.Box(
                            low=0.0,
                            high=self.consts.HEIGHT,
                            shape=(),
                            dtype=jnp.float32,
                        ),
                        "w": spaces.Box(
                            low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.float32
                        ),
                        "h": spaces.Box(
                            low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.float32
                        ),
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
                            low=0.0, high=self.consts.WIDTH, shape=(), dtype=jnp.float32
                        ),
                        "y": spaces.Box(
                            low=0.0,
                            high=self.consts.HEIGHT,
                            shape=(),
                            dtype=jnp.float32,
                        ),
                        "w": spaces.Box(
                            low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.float32
                        ),
                        "h": spaces.Box(
                            low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.float32
                        ),
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
                            low=0.0, high=self.consts.WIDTH, shape=(), dtype=jnp.float32
                        ),
                        "y": spaces.Box(
                            low=0.0,
                            high=self.consts.HEIGHT,
                            shape=(),
                            dtype=jnp.float32,
                        ),
                        "w": spaces.Box(
                            low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.float32
                        ),
                        "h": spaces.Box(
                            low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.float32
                        ),
                        "direction": spaces.Box(
                            low=0, high=7, shape=(), dtype=jnp.float32
                        ),
                    }
                ),
                "energy_shield": spaces.Dict(
                    {
                        "x": spaces.Box(
                            low=0.0, high=self.consts.WIDTH, shape=(), dtype=jnp.float32
                        ),
                        "y": spaces.Box(
                            low=0.0,
                            high=self.consts.HEIGHT,
                            shape=(),
                            dtype=jnp.float32,
                        ),
                        "w": spaces.Box(
                            low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.float32
                        ),
                        "h": spaces.Box(
                            low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.float32
                        ),
                    }
                ),
                "energy_shield_state": spaces.Box(
                    low=0, high=1, shape=(16, 8), dtype=jnp.float32
                ),
                "lives": spaces.Box(low=0, high=4, shape=(), dtype=jnp.float32),
            }
        )

    def image_space(self):
        """Returns the pixel representation."""
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.consts.HEIGHT, self.consts.WIDTH, 3),
            dtype=jnp.uint8,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: YarsRevengeState) -> YarsRevengeObservation:
        """Return the observation structure for the agent."""
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
        """
        Convert the observation into a 1-D float array.
        """
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
        """Return auxiliary information about the current state."""
        return YarsRevengeInfo(time=state.step_counter)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state, state):
        """Reward is simply the score delta - never negative in this game."""
        return state.score - previous_state.score

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state):
        """The original code never terminates - always False."""
        return False


class YarsRevengeRenderer(JAXGameRenderer):
    """
    Handles the conversion from a `YarsRevengeState` to an RGB image.
    The renderer uses pre-loaded sprite masks and a palette.  It is
    intentionally kept separate from the environment logic so that
    visualisation can be reused in other contexts.
    """

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

        # Load all sprites
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS,
        ) = self.jr.load_and_setup_assets(asset_config, sprite_path)

    def _get_asset_config(self) -> list:
        """Return the asset description for the renderer."""
        return [
            {"name": "background", "type": "background", "file": "background.npy"},
            {
                "name": "yar",
                "type": "group",
                "files": [
                    # 16 sprites - two per direction (0,1)
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
            # Shield is generated procedurally
            {
                "name": "energy_shield",
                "type": "procedural",
                "data": jnp.array(
                    self.consts.ENERGY_SHIELD_COLOR + (255,), dtype=jnp.uint8
                ).reshape(1, 1, 4),
            },
            # Neutral zone sprite
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
        """
        Compute the index of a sprite animation frame.

        The formula `group * 2 + floor(step / duration) % group_item_count`
        yields an even-odd pair for each direction - e.g. direction 0 has
        frames 0 and 1, direction 1 has frames 2 and 3, etc.
        """
        return group * 2 + (
            jnp.floor(step / duration).astype(jnp.int32) % group_item_count
        )

    def render(self, state: YarsRevengeState):
        """
        Render the complete frame by compositing all game elements onto a background raster.
        """
        # Start with an empty background
        raster = self.jr.create_object_raster(self.BACKGROUND)

        # Neutral zone overlay
        neutral_zone_mask = jnp.full(
            (self.consts.NEUTRAL_ZONE_SIZE[1], self.consts.NEUTRAL_ZONE_SIZE[0]),
            self.COLOR_TO_ID[self.consts.NEUTRAL_ZONE_COLOR],
        )
        raster = self.jr.render_at(
            raster, state.neutral_zone.x, state.neutral_zone.y, neutral_zone_mask
        )

        # Yar sprite, choose animation frame based on movement speed to render
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

        # Energy shield
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

        # Destroyer sprite
        destroyer_mask = jnp.ones(
            (self.consts.DESTROYER_SIZE[1], self.consts.DESTROYER_SIZE[0])
        )
        raster = self.jr.render_at(
            raster, state.destroyer.x, state.destroyer.y, destroyer_mask
        )

        # Energy missile, only when it exists
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

        # Cannon alternates between full and half sprite every frame
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

        # Qotile and swirl, one sprite each (swirl has its own animation)
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

        # Convert the raster from indices to RGB values
        return self.jr.render_from_palette(raster, self.PALETTE)
