import os
from functools import partial
from typing import Any, Dict, NamedTuple, Optional, Tuple
import jax
import jax.lax
import jax.numpy as jnp
import chex

import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action


# --- Level Configuration ---
class RoadSectionConfig(NamedTuple):
    scroll_start: int
    scroll_end: int
    road_width: int
    road_top: int
    road_height: int
    road_pattern_style: int = 0


class LevelConfig(NamedTuple):
    level_number: int
    scroll_distance_to_complete: int
    road_sections: Tuple[RoadSectionConfig, ...]
    spawn_seeds: bool
    spawn_trucks: bool
    spawn_ravines: bool = False
    decorations: Tuple[Tuple[int, int, int, int], ...] = ()
    seed_spawn_config: Optional[Tuple[int, int]] = None
    truck_spawn_config: Optional[Tuple[int, int]] = None
    ravine_spawn_config: Optional[Tuple[int, int]] = None
    spawn_landmines: bool = False
    landmine_spawn_config: Optional[Tuple[int, int]] = None
    future_entity_types: Dict[str, Any] = {}


# --- Constants ---
class RoadRunnerConstants(NamedTuple):
    WIDTH: int = 160
    HEIGHT: int = 210
    PLAYER_MOVE_SPEED: int = 3
    PLAYER_ANIMATION_SPEED: int = 2
    # If the players x coordinate would be below this value after applying movement, we move everything one to the right to simulate movement.
    X_SCROLL_THRESHOLD: int = 50
    ENEMY_MOVE_SPEED: int = 2
    ENEMY_REACTION_DELAY: int = 6
    PLAYER_START_X: int = 70
    PLAYER_START_Y: int = 120
    ENEMY_X: int = 140
    ENEMY_Y: int = 120
    PLAYER_SIZE: Tuple[int, int] = (8, 32)
    ENEMY_SIZE: Tuple[int, int] = (4, 4)
    SEED_SIZE: Tuple[int, int] = (5, 5)
    PLAYER_PICKUP_OFFSET: int = PLAYER_SIZE[1] * 3 // 4  # Bottom 25% of player height
    WALL_TOP_Y: int = 24
    WALL_TOP_HEIGHT: int = 10
    WALL_BOTTOM_Y: int = 194
    WALL_BOTTOM_HEIGHT: int = 16
    ROAD_HEIGHT: int = 90
    ROAD_TOP_Y: int = 110
    ROAD_DASH_LENGTH: int = 5
    ROAD_GAP_HEIGHT: int = 17
    ROAD_PATTERN_WIDTH: int = ROAD_DASH_LENGTH * 4
    SPAWN_Y_RANDOM_OFFSET_MIN: int = -20
    SPAWN_Y_RANDOM_OFFSET_MAX: int = 20
    BACKGROUND_COLOR: Tuple[int, int, int] = (236, 168, 128)
    PLAYER_COLOR: Tuple[int, int, int] = (92, 186, 92)
    ENEMY_COLOR: Tuple[int, int, int] = (213, 130, 74)
    WALL_COLOR: Tuple[int, int, int] = (236, 236, 236)
    SEED_SPAWN_MIN_INTERVAL: int = 5
    SEED_SPAWN_MAX_INTERVAL: int = 20
    MAX_STREAK: int = 10
    SEED_BASE_VALUE: int = 100
    TRUCK_SIZE: Tuple[int, int] = (15, 15)
    TRUCK_COLLISION_OFFSET: int = TRUCK_SIZE[1] // 2  # Bottom half of truck height
    TRUCK_COLOR: Tuple[int, int, int] = (255, 0, 0)
    TRUCK_SPEED: int = 3
    TRUCK_SPAWN_MIN_INTERVAL: int = 30
    TRUCK_SPAWN_MAX_INTERVAL: int = 80
    LEVEL_TRANSITION_DURATION: int = 30
    LEVEL_COMPLETE_SCROLL_DISTANCE: int = 1500
    STARTING_LIVES: int = 3
    JUMP_TIME_DURATION: int = 20  # Jump duration in steps (~0.33 seconds at 60 FPS)
    SIDE_MARGIN: int = 8
    RAVINE_SIZE: Tuple[int, int] = (13, 32)
    RAVINE_SPAWN_MIN_INTERVAL: int = 30
    RAVINE_SPAWN_MAX_INTERVAL: int = 60
    LANDMINE_SIZE: Tuple[int, int] = (4, 4)
    LANDMINE_SPAWN_MIN_INTERVAL: int = 120
    LANDMINE_SPAWN_MAX_INTERVAL: int = 240
    DEATH_ANIMATION_DURATION: int = 60  # 1 second at 60 FPS
    # Enemy speed variation - speeds as offsets from PLAYER_MOVE_SPEED
    ENEMY_SLOW_SPEED_OFFSET: int = -1        # Speed = PLAYER_MOVE_SPEED - 1 = 2
    ENEMY_FAST_SPEED_OFFSET: int = 1         # Speed = PLAYER_MOVE_SPEED + 1 = 4
    ENEMY_SAME_SPEED_OFFSET: int = 0         # Speed = PLAYER_MOVE_SPEED = 3
    # Enemy speed cycle durations (in scroll distance units)
    ENEMY_SLOW_DURATION: int = 60            # Default slow phase duration
    ENEMY_FAST_DURATION: int = 60            # ~1 second at 60 FPS  
    ENEMY_SAME_DURATION: int = 300           # ~5 seconds at 60 FPS
    # Enemy approach slowdown/reversal multiplier (when player moves right)
    # Positive values slow down (0.5 = half speed), negative values reverse direction (-0.5 = move away at half speed)
    ENEMY_APPROACH_SLOWDOWN: float = -0.5     # Moves backwards at half speed when player approaches
    # --- Decoration Type Constants ---
    DECO_CACTUS = 0
    DECO_SIGN_THIS_WAY = 1
    DECO_SIGN_BIRD_SEED = 2
    DECO_SIGN_CARS_AHEAD = 3
    DECO_SIGN_EXIT = 4
    levels: Tuple[LevelConfig, ...] = ()


_BASE_CONSTS = RoadRunnerConstants()
_DEFAULT_ROAD_HEIGHT = _BASE_CONSTS.ROAD_HEIGHT


def _centered_top(height: int) -> int:
    return max((_DEFAULT_ROAD_HEIGHT - height) // 2, 0)


RoadRunner_Level_1 = LevelConfig(
    level_number=1,
    scroll_distance_to_complete=_BASE_CONSTS.LEVEL_COMPLETE_SCROLL_DISTANCE,
    road_sections=(
        RoadSectionConfig(
            scroll_start=0,
            scroll_end=_BASE_CONSTS.LEVEL_COMPLETE_SCROLL_DISTANCE,
            road_width=_BASE_CONSTS.WIDTH - 2 * _BASE_CONSTS.SIDE_MARGIN,
            road_top=0,
            road_height=_DEFAULT_ROAD_HEIGHT,
            road_pattern_style=0,
        ),
    ),
    spawn_seeds=True,
    spawn_trucks=True,
    seed_spawn_config=(
        _BASE_CONSTS.SEED_SPAWN_MIN_INTERVAL,
        _BASE_CONSTS.SEED_SPAWN_MAX_INTERVAL,
    ),
    truck_spawn_config=(
        _BASE_CONSTS.TRUCK_SPAWN_MIN_INTERVAL,
        _BASE_CONSTS.TRUCK_SPAWN_MAX_INTERVAL,
    ),
    decorations=(
        # --- INTRO (0-6s) ---
        (50, 60, 1, _BASE_CONSTS.DECO_SIGN_THIS_WAY),
        (0, 45, 2, _BASE_CONSTS.DECO_CACTUS),
        (30, 55, 1, _BASE_CONSTS.DECO_CACTUS),
        (180, 70, 1, _BASE_CONSTS.DECO_SIGN_BIRD_SEED),
        (350, 60, 1, _BASE_CONSTS.DECO_SIGN_CARS_AHEAD),

        # --- THE DESERT RUN (13 Cacti) ---
        (420, 45, 3, _BASE_CONSTS.DECO_CACTUS),
        (500, 55, 2, _BASE_CONSTS.DECO_CACTUS),
        (580, 45, 3, _BASE_CONSTS.DECO_CACTUS),
        (660, 55, 2, _BASE_CONSTS.DECO_CACTUS),
        (740, 45, 3, _BASE_CONSTS.DECO_CACTUS),
        (820, 55, 2, _BASE_CONSTS.DECO_CACTUS),
        (900, 45, 3, _BASE_CONSTS.DECO_CACTUS),
        (980, 55, 2, _BASE_CONSTS.DECO_CACTUS),
        (1060, 45, 3, _BASE_CONSTS.DECO_CACTUS),
        (1140, 55, 2, _BASE_CONSTS.DECO_CACTUS),
        (1220, 45, 3, _BASE_CONSTS.DECO_CACTUS),
        (1300, 55, 2, _BASE_CONSTS.DECO_CACTUS),
        (1380, 45, 3, _BASE_CONSTS.DECO_CACTUS),

        # --- OUTRO ---
        (1800, 60, 1, _BASE_CONSTS.DECO_SIGN_EXIT),
    ),
)

RoadRunner_Level_2 = LevelConfig(
    level_number=2,
    scroll_distance_to_complete=_BASE_CONSTS.LEVEL_COMPLETE_SCROLL_DISTANCE,
    road_sections=(
        RoadSectionConfig(
            scroll_start=0,
            scroll_end=300,
            road_width=_BASE_CONSTS.WIDTH - 2 * _BASE_CONSTS.SIDE_MARGIN,
            road_top=_centered_top(32),
            road_height=32,
        ),
        RoadSectionConfig(
            scroll_start=300,
            scroll_end=700,
            road_width=_BASE_CONSTS.WIDTH - 2 * _BASE_CONSTS.SIDE_MARGIN,
            road_top=_centered_top(32),
            road_height=32,
        ),
        RoadSectionConfig(
            scroll_start=700,
            scroll_end=_BASE_CONSTS.LEVEL_COMPLETE_SCROLL_DISTANCE,
            road_width=_BASE_CONSTS.WIDTH - 2 * _BASE_CONSTS.SIDE_MARGIN,
            road_top=_centered_top(32),
            road_height=32,
        ),
    ),
    spawn_seeds=False,
    spawn_trucks=False,
    spawn_ravines=True,
    truck_spawn_config=(
        _BASE_CONSTS.TRUCK_SPAWN_MIN_INTERVAL,
        _BASE_CONSTS.TRUCK_SPAWN_MAX_INTERVAL,
    ),
    spawn_landmines=True,
    landmine_spawn_config=(
        _BASE_CONSTS.LANDMINE_SPAWN_MIN_INTERVAL,
        _BASE_CONSTS.LANDMINE_SPAWN_MAX_INTERVAL,
    ),
)

DEFAULT_LEVELS: Tuple[LevelConfig, ...] = (
    RoadRunner_Level_1,
    RoadRunner_Level_2,
)


# --- Helper Functions ---
def _build_road_section_arrays(
    levels: Tuple[LevelConfig, ...], consts: RoadRunnerConstants
) -> Tuple[int, jnp.ndarray, jnp.ndarray]:
    """
    Build road section data arrays from level configs.
    Returns (max_road_sections, road_section_data, road_section_counts).
    """
    if not levels:
        return 0, jnp.array([], dtype=jnp.int32).reshape(0, 0, 6), jnp.array([], dtype=jnp.int32)

    max_road_sections = max(len(cfg.road_sections) for cfg in levels)
    road_sections_data = []
    road_section_counts = []
    default_section = [
        0,
        consts.LEVEL_COMPLETE_SCROLL_DISTANCE,
        consts.WIDTH,
        0,
        0,
        consts.ROAD_HEIGHT,
    ]
    for cfg in levels:
        rows = [
            [
                section.scroll_start,
                section.scroll_end,
                section.road_width,
                section.road_pattern_style,
                section.road_top,
                section.road_height,
            ]
            for section in cfg.road_sections
        ]
        if not rows:
            rows = [default_section[:]]
        road_section_counts.append(len(rows))
        while len(rows) < max_road_sections:
            rows.append(rows[-1][:])
        road_sections_data.append(rows)

    return (
        max_road_sections,
        jnp.array(road_sections_data, dtype=jnp.int32),
        jnp.array(road_section_counts, dtype=jnp.int32),
    )


def _build_spawn_interval_array(
    levels: Tuple[LevelConfig, ...],
    config_attr: str,
    default_min: int,
    default_max: int,
) -> jnp.ndarray:
    """
    Build spawn interval array from level configs.

    Args:
        levels: Tuple of level configurations
        config_attr: Name of the config attribute (e.g., 'seed_spawn_config')
        default_min: Default minimum interval
        default_max: Default maximum interval

    Returns:
        Array of shape (num_levels, 2) with [min, max] intervals per level
    """
    if not levels:
        return jnp.array([], dtype=jnp.int32).reshape(0, 2)

    return jnp.array(
        [
            [
                (getattr(cfg, config_attr) or (default_min, default_max))[0],
                (getattr(cfg, config_attr) or (default_min, default_max))[1],
            ]
            for cfg in levels
        ],
        dtype=jnp.int32,
    )


def _build_spawn_enabled_array(
    levels: Tuple[LevelConfig, ...],
    spawn_attr: str,
) -> jnp.ndarray:
    """
    Build spawn enabled boolean array from level configs.

    Args:
        levels: Tuple of level configurations
        spawn_attr: Name of the spawn attribute (e.g., 'spawn_seeds')

    Returns:
        Boolean array indicating if spawning is enabled per level
    """
    if not levels:
        return jnp.array([], dtype=jnp.bool_)

    return jnp.array(
        [getattr(cfg, spawn_attr) for cfg in levels], dtype=jnp.bool_
    )


def _check_aabb_collision(
    x1: chex.Array, y1: chex.Array, w1: int, h1: int,
    x2: chex.Array, y2: chex.Array, w2: int, h2: int,
) -> chex.Array:
    """
    Check if two axis-aligned bounding boxes overlap.

    Args:
        x1, y1: Top-left corner of first box
        w1, h1: Width and height of first box
        x2, y2: Top-left corner of second box
        w2, h2: Width and height of second box

    Returns:
        Boolean indicating if boxes overlap
    """
    overlap_x = (x1 < x2 + w2) & (x1 + w1 > x2)
    overlap_y = (y1 < y2 + h2) & (y1 + h1 > y2)
    return overlap_x & overlap_y


def _update_orientation(
    vel_x: chex.Array,
    current_looks_right: chex.Array,
) -> chex.Array:
    """
    Update orientation based on horizontal velocity.

    Args:
        vel_x: Horizontal velocity
        current_looks_right: Current orientation

    Returns:
        Updated orientation (True = right, False = left)
    """
    return jax.lax.cond(
        vel_x > 0,
        lambda: True,
        lambda: jax.lax.cond(
            vel_x < 0, lambda: False, lambda: current_looks_right
        ),
    )


def _get_road_section_for_scroll(
    state: "RoadRunnerState",
    level_count: int,
    max_road_sections: int,
    road_section_data: jnp.ndarray,
    road_section_counts: jnp.ndarray,
    consts: RoadRunnerConstants,
) -> RoadSectionConfig:
    """
    Get the current road section based on scroll position.

    Args:
        state: Current game state
        level_count: Number of levels
        max_road_sections: Maximum number of road sections across all levels
        road_section_data: Array of road section data per level
        road_section_counts: Array of section counts per level
        consts: Game constants

    Returns:
        The current RoadSectionConfig
    """
    if level_count == 0 or max_road_sections == 0:
        return RoadSectionConfig(
            0,
            consts.LEVEL_COMPLETE_SCROLL_DISTANCE,
            consts.WIDTH,
            0,
            consts.ROAD_HEIGHT,
            0,
        )

    max_index = level_count - 1
    level_idx = jnp.clip(state.current_level, 0, max_index).astype(jnp.int32)
    sections = road_section_data[level_idx]
    section_count = road_section_counts[level_idx]
    indices = jnp.arange(max_road_sections, dtype=jnp.int32)
    valid_section = indices < section_count
    counter = state.scrolling_step_counter
    in_section = (counter >= sections[:, 0]) & (counter < sections[:, 1])
    active_sections = jnp.where(valid_section, in_section, False)
    no_match_value = jnp.array(max_road_sections, dtype=jnp.int32)
    candidate_indices = jnp.where(active_sections, indices, no_match_value)
    match_idx = jnp.min(candidate_indices)
    fallback_idx = jnp.maximum(section_count - 1, 0)
    section_idx = jnp.where(match_idx < max_road_sections, match_idx, fallback_idx)
    selected = sections[section_idx]
    return RoadSectionConfig(
        selected[0],
        selected[1],
        selected[2],
        selected[4],
        selected[5],
        selected[3],
    )


# --- State and Observation ---
class RoadRunnerState(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    player_x_history: chex.Array
    player_y_history: chex.Array
    enemy_x: chex.Array
    enemy_y: chex.Array
    step_counter: chex.Array
    player_is_moving: chex.Array
    player_looks_right: chex.Array
    enemy_is_moving: chex.Array
    enemy_looks_right: chex.Array
    score: chex.Array
    is_scrolling: chex.Array
    scrolling_step_counter: chex.Array
    is_round_over: chex.Array
    seeds: chex.Array # 2D array of shape (4, 3)
    next_seed_spawn_scroll_step: chex.Array # Scrolling step counter value at which to spawn next seed
    rng: chex.Array # PRNG state
    seed_pickup_streak: chex.Array
    last_picked_up_seed_id: chex.Array
    next_seed_id: chex.Array
    truck_x: chex.Array
    truck_y: chex.Array
    next_truck_spawn_step: chex.Array
    current_level: chex.Array
    level_transition_timer: chex.Array
    is_in_transition: chex.Array
    lives: chex.Array
    jump_timer: chex.Array  # Countdown timer for jump (0 when not jumping)
    is_jumping: chex.Array  # Boolean flag indicating if player is currently jumping
    ravines: chex.Array 
    next_ravine_spawn_scroll_step: chex.Array
    landmine_x: chex.Array
    landmine_y: chex.Array
    next_landmine_spawn_step: chex.Array
    death_timer: chex.Array
    instant_death: chex.Array # Boolean, if true, skip death animation/delay
    enemy_speed_phase_start: chex.Array  # Scroll step when current speed phase cycle began

class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray


class RoadRunnerObservation(NamedTuple):
    player: EntityPosition
    enemy: EntityPosition
    score: jnp.ndarray
    ravine: EntityPosition


# --- Main Environment Class ---
class JaxRoadRunner(
    JaxEnvironment[RoadRunnerState, RoadRunnerObservation, None, RoadRunnerConstants]
):
    def __init__(self, consts: RoadRunnerConstants = None):
        if consts is None:
            consts = RoadRunnerConstants(levels=DEFAULT_LEVELS)
        elif len(consts.levels) == 0:
            consts = consts._replace(levels=DEFAULT_LEVELS)
        super().__init__(consts)
        self.renderer = RoadRunnerRenderer(self.consts)
        self.action_set = [
            Action.NOOP,
            Action.FIRE,
            Action.UP,
            Action.DOWN,
            Action.LEFT,
            Action.RIGHT,
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
        self.obs_size = 2 * 4  # Simplified

        # Pre-calculate normalized velocities
        sqrt2_inv = 1 / jnp.sqrt(2)
        self._velocities = (
            jnp.array(
                [
                    [0, 0],  # NOOP
                    [0, 0],  # FIRE (jump handled separately)
                    [0, -1],  # UP
                    [0, 1],  # DOWN
                    [-1, 0],  # LEFT
                    [1, 0],  # RIGHT
                    [sqrt2_inv, -sqrt2_inv],  # UPRIGHT
                    [-sqrt2_inv, -sqrt2_inv],  # UPLEFT
                    [sqrt2_inv, sqrt2_inv],  # DOWNRIGHT
                    [-sqrt2_inv, sqrt2_inv],  # DOWNLEFT
                    [0, -1],  # UPFIRE (jump + up)
                    [1, 0],  # RIGHTFIRE (jump + right)
                    [-1, 0],  # LEFTFIRE (jump + left)
                    [0, 1],  # DOWNFIRE (jump + down)
                    [sqrt2_inv, -sqrt2_inv],  # UPRIGHTFIRE (jump + upright)
                    [-sqrt2_inv, -sqrt2_inv],  # UPLEFTFIRE (jump + upleft)
                    [sqrt2_inv, sqrt2_inv],  # DOWNRIGHTFIRE (jump + downright)
                    [-sqrt2_inv, sqrt2_inv],  # DOWNLEFTFIRE (jump + downleft)
                ]
            )
            * self.consts.PLAYER_MOVE_SPEED
        )
        self._level_count = len(self.consts.levels)
        levels = self.consts.levels

        # Build spawn enabled arrays
        self._level_spawn_seeds = _build_spawn_enabled_array(levels, 'spawn_seeds')
        self._level_spawn_trucks = _build_spawn_enabled_array(levels, 'spawn_trucks')
        self._level_spawn_ravines = _build_spawn_enabled_array(levels, 'spawn_ravines')
        self._level_spawn_landmines = _build_spawn_enabled_array(levels, 'spawn_landmines')

        # Build spawn interval arrays
        self._seed_spawn_intervals = _build_spawn_interval_array(
            levels, 'seed_spawn_config',
            self.consts.SEED_SPAWN_MIN_INTERVAL, self.consts.SEED_SPAWN_MAX_INTERVAL
        )
        self._truck_spawn_intervals = _build_spawn_interval_array(
            levels, 'truck_spawn_config',
            self.consts.TRUCK_SPAWN_MIN_INTERVAL, self.consts.TRUCK_SPAWN_MAX_INTERVAL
        )
        self._ravine_spawn_intervals = _build_spawn_interval_array(
            levels, 'ravine_spawn_config',
            self.consts.RAVINE_SPAWN_MIN_INTERVAL, self.consts.RAVINE_SPAWN_MAX_INTERVAL
        )
        self._landmine_spawn_intervals = _build_spawn_interval_array(
            levels, 'landmine_spawn_config',
            self.consts.LANDMINE_SPAWN_MIN_INTERVAL, self.consts.LANDMINE_SPAWN_MAX_INTERVAL
        )

        # Build road section arrays
        (
            self._max_road_sections,
            self._road_section_data,
            self._road_section_counts,
        ) = _build_road_section_arrays(levels, self.consts)

    def _handle_input(self, action: chex.Array) -> tuple[chex.Array, chex.Array, chex.Array]:
        """Handles user input to determine player velocity and jump action."""
        # Map action to the corresponding index in the action_set
        action_idx = jnp.argmax(jnp.array(self.action_set) == action)
        vel = self._velocities[action_idx]
        # Check if action involves FIRE (jump): FIRE (1) or any *FIRE action (10-17)
        is_fire_action = (action == Action.FIRE) | ((action >= Action.UPFIRE) & (action <= Action.DOWNLEFTFIRE))
        return vel[0], vel[1], is_fire_action

    def _check_player_bounds(
        self, state: RoadRunnerState, x_pos: chex.Array, y_pos: chex.Array
    ) -> tuple[chex.Array, chex.Array]:
        road_top, road_bottom, _ = self._get_road_bounds(state)
        min_y = road_top - (self.consts.PLAYER_SIZE[1] - 5)
        max_y = road_bottom - self.consts.PLAYER_SIZE[1]
        checked_y = jnp.clip(y_pos, min_y, max_y)
        checked_x = jnp.clip(
            x_pos,
            self.consts.SIDE_MARGIN,
            self.consts.WIDTH - self.consts.PLAYER_SIZE[0] - self.consts.SIDE_MARGIN,
        )
        return (checked_x, checked_y)

    def _check_enemy_bounds(
        self, state: RoadRunnerState, x_pos: chex.Array, y_pos: chex.Array
    ) -> tuple[chex.Array, chex.Array]:
        road_top, road_bottom, _ = self._get_road_bounds(state)
        min_y = road_top - (self.consts.PLAYER_SIZE[1] // 3)
        max_y = road_bottom - self.consts.PLAYER_SIZE[1]
        checked_y = jnp.clip(y_pos, min_y, max_y)

        # Only clip x on the left side
        # TODO Generalize this so we don't need to duplicate the bounds checking
        checked_x = jnp.maximum(x_pos, self.consts.SIDE_MARGIN)

        return (checked_x, checked_y)

    def _handle_scrolling(self, state: RoadRunnerState, x_pos: chex.Array):
        return jax.lax.cond(
            state.is_scrolling,
            lambda: x_pos + self.consts.PLAYER_MOVE_SPEED,
            lambda: x_pos,
        )

    def _player_step(
        self, state: RoadRunnerState, action: chex.Array
    ) -> RoadRunnerState:

        # --- Update Player Position ---
        input_vel_x, input_vel_y, is_fire_action = self._handle_input(action)

        # Handle jump logic (simple boolean state - no position checking)
        # If FIRE is pressed and not already jumping, start jump
        # Otherwise, count down the jump timer
        can_start_jump = (state.jump_timer == 0) & jnp.logical_not(state.is_round_over)
        should_start_jump = is_fire_action & can_start_jump

        new_jump_timer = jax.lax.cond(
            should_start_jump,
            lambda: jnp.array(self.consts.JUMP_TIME_DURATION, dtype=jnp.int32),
            lambda: jnp.maximum(state.jump_timer - 1, 0),
        )

        # Determine if currently jumping
        is_jumping = new_jump_timer > 0

        # If round is over, player is forced to move right.
        vel_x = jax.lax.cond(
            state.is_round_over,
            lambda: jnp.array(self.consts.PLAYER_MOVE_SPEED, dtype=jnp.float32),
            lambda: input_vel_x,
        )
        vel_y = jax.lax.cond(state.is_round_over, lambda: 0.0, lambda: input_vel_y)

        # Determine if scrolling should happen based on the potential next position.
        tentative_player_x = state.player_x + vel_x
        is_scrolling = tentative_player_x < self.consts.X_SCROLL_THRESHOLD

        # When scrolling, the player's horizontal velocity should counteract the scroll.
        # We use the original vel_x for non-scrolling movement.
        final_vel_x = jax.lax.cond(
            is_scrolling,
            lambda: -float(self.consts.PLAYER_MOVE_SPEED),
            lambda: vel_x,
        )

        player_x = state.player_x + final_vel_x
        player_y = state.player_y + vel_y

        player_x, player_y = self._check_player_bounds(state, player_x, player_y)

        is_moving = (vel_x != 0) | (vel_y != 0)

        # Update player orientation based on horizontal movement
        player_looks_right = _update_orientation(vel_x, state.player_looks_right)

        # Update the state with the scrolling flag for other parts of the game (e.g., rendering).
        state = state._replace(is_scrolling=is_scrolling)
        state = jax.lax.cond(
            state.is_scrolling,
            lambda: state._replace(scrolling_step_counter=state.scrolling_step_counter + 1),
            lambda: state,
        )

        # Apply the scroll offset to the player's final position.
        player_x = self._handle_scrolling(state, player_x)

        # Update player position history for enemy AI
        new_x_history = jnp.roll(state.player_x_history, shift=1)
        new_x_history = new_x_history.at[0].set(state.player_x)
        new_y_history = jnp.roll(state.player_y_history, shift=1)
        new_y_history = new_y_history.at[0].set(state.player_y)

        return state._replace(
            player_x=player_x.astype(jnp.int32),
            player_y=player_y.astype(jnp.int32),
            player_is_moving=is_moving,
            player_looks_right=player_looks_right,
            player_x_history=new_x_history,
            player_y_history=new_y_history,
            jump_timer=new_jump_timer,
            is_jumping=is_jumping,
        )

    def _enemy_step(self, state: RoadRunnerState) -> RoadRunnerState:
        def game_over_logic(st: RoadRunnerState) -> RoadRunnerState:
            new_enemy_x = st.enemy_x + self.consts.PLAYER_MOVE_SPEED
            new_enemy_x, new_enemy_y = self._check_enemy_bounds(st, new_enemy_x, st.enemy_y)
            return st._replace(
                enemy_x=new_enemy_x,
                enemy_y=new_enemy_y,
                enemy_is_moving=True,
                enemy_looks_right=True,
            )

        def normal_logic(st: RoadRunnerState) -> RoadRunnerState:
            # Calculate current speed phase based on scroll distance
            total_cycle = (self.consts.ENEMY_SLOW_DURATION + 
                           self.consts.ENEMY_FAST_DURATION + 
                           self.consts.ENEMY_SAME_DURATION)
            
            cycle_progress = (st.scrolling_step_counter - st.enemy_speed_phase_start) % total_cycle
            
            # Determine current phase speed offset
            in_slow = cycle_progress < self.consts.ENEMY_SLOW_DURATION
            in_fast = (cycle_progress >= self.consts.ENEMY_SLOW_DURATION) & \
                      (cycle_progress < self.consts.ENEMY_SLOW_DURATION + self.consts.ENEMY_FAST_DURATION)
            
            speed_offset = jnp.where(
                in_slow,
                self.consts.ENEMY_SLOW_SPEED_OFFSET,
                jnp.where(
                    in_fast,
                    self.consts.ENEMY_FAST_SPEED_OFFSET,
                    self.consts.ENEMY_SAME_SPEED_OFFSET
                )
            )
            
            base_speed = self.consts.PLAYER_MOVE_SPEED + speed_offset
            
            # Check if player is moving right (approaching enemy)
            # Player velocity is the difference between current position and previous position
            player_vel_x = st.player_x - st.player_x_history[0]
            is_approaching = player_vel_x > 0
            
            slowdown = jnp.where(
                is_approaching,
                self.consts.ENEMY_APPROACH_SLOWDOWN,
                1.0
            )
            
            # Calculate final speed (absolute value for clipping bounds)
            # Negative slowdown will reverse direction via the multiplier on delta
            final_speed = (base_speed * jnp.abs(slowdown)).astype(jnp.int32)
            final_speed = jnp.maximum(final_speed, 1)

            # Get the distance to the player, with a configurable frame delay.
            delayed_player_x = st.player_x_history[
                self.consts.ENEMY_REACTION_DELAY - 1
            ]
            delayed_player_y = st.player_y_history[
                self.consts.ENEMY_REACTION_DELAY - 1
            ]
            delta_x = delayed_player_x - st.enemy_x
            delta_y = delayed_player_y - st.enemy_y
            
            # Apply direction modifier (negative slowdown reverses direction)
            direction_modifier = jnp.where(slowdown < 0, -1.0, 1.0)
            modified_delta_x = delta_x * direction_modifier
            modified_delta_y = delta_y * direction_modifier

            # Determine enemy movement and orientation
            enemy_is_moving = (delta_x != 0) | (delta_y != 0)
            enemy_looks_right = _update_orientation(modified_delta_x, st.enemy_looks_right)

            # Update enemy position, clipping movement to final_speed to prevent jittering
            new_enemy_x = st.enemy_x + jnp.clip(
                modified_delta_x, -final_speed, final_speed
            )
            new_enemy_y = st.enemy_y + jnp.clip(
                modified_delta_y, -final_speed, final_speed
            )

            new_enemy_x = self._handle_scrolling(st, new_enemy_x)

            new_enemy_x, new_enemy_y = self._check_enemy_bounds(st, new_enemy_x, new_enemy_y)
            return st._replace(
                enemy_x=new_enemy_x.astype(jnp.int32),
                enemy_y=new_enemy_y.astype(jnp.int32),
                enemy_is_moving=enemy_is_moving,
                enemy_looks_right=enemy_looks_right,
            )

        return jax.lax.cond(state.is_round_over, game_over_logic, normal_logic, state)

    def _check_game_over(self, state: RoadRunnerState) -> RoadRunnerState:
        # Check if the enemy and the player overlap
        collision = _check_aabb_collision(
            state.player_x, state.player_y,
            self.consts.PLAYER_SIZE[0], self.consts.PLAYER_SIZE[1],
            state.enemy_x, state.enemy_y,
            self.consts.ENEMY_SIZE[0], self.consts.ENEMY_SIZE[1],
        )

        return jax.lax.cond(
            collision,
            lambda st: st._replace(
                is_round_over=True,
                player_x=(st.enemy_x + self.consts.ENEMY_SIZE[0] + 2).astype(jnp.int32),
                player_y=st.enemy_y.astype(jnp.int32),
            ),
            lambda st: st,
            state,
        )


    def update_streak(self, state: RoadRunnerState, seed_idx: int, max_streak: int) -> RoadRunnerState:
        def restart_streak(state:RoadRunnerState) -> RoadRunnerState:
            return state._replace(seed_pickup_streak=1)

        def advance_streak(state:RoadRunnerState, max_streak: int) -> RoadRunnerState:
            new_streak = jax.lax.min(state.seed_pickup_streak+1, max_streak)
            return state._replace(seed_pickup_streak=new_streak)

        last_picked_up_seed_id = state.last_picked_up_seed_id
        state = state._replace(last_picked_up_seed_id=state.seeds[seed_idx, 2])
        return jax.lax.cond(
            state.seeds[seed_idx, 2] == last_picked_up_seed_id+1,
            lambda: advance_streak(state, max_streak),
            lambda: restart_streak(state),
        )

    def _seed_picked_up(self, state: RoadRunnerState, seed_idx: int) -> RoadRunnerState:
        state = self.update_streak(state, seed_idx, self.consts.MAX_STREAK)

        # Set seed to inactive (-1, -1)
        updated_seeds = state.seeds.at[seed_idx].set(
            jnp.array([-1, -1, -1], dtype=jnp.int32)
        )
        # Increment score by 100 Placeholder value
        new_score = state.score + self.consts.SEED_BASE_VALUE * state.seed_pickup_streak
        return state._replace(
            seeds=updated_seeds,
            score=new_score.astype(jnp.int32),
        )

    def _check_seed_collisions(self, state: RoadRunnerState) -> RoadRunnerState:
        """
        Check for collisions between player and all active seeds.
        Uses AABB (Axis-Aligned Bounding Box) collision detection.
        """
        # Player pickup area starts at PLAYER_PICKUP_OFFSET from top
        player_pickup_y = state.player_y + self.consts.PLAYER_PICKUP_OFFSET
        pickup_height = self.consts.PLAYER_SIZE[1] - self.consts.PLAYER_PICKUP_OFFSET

        def check_and_pickup_seed(i: int, st: RoadRunnerState) -> RoadRunnerState:
            """Check collision for seed at index i and pick it up if colliding."""
            seed_x = st.seeds[i, 0]
            is_active = seed_x >= 0
            return jax.lax.cond(
                is_active,
                lambda s: _check_collision_and_pickup(s, i),
                lambda s: s,
                st,
            )

        def _check_collision_and_pickup(st: RoadRunnerState, seed_idx: int) -> RoadRunnerState:
            """Check collision for active seed and pick it up if colliding."""
            seed_x = st.seeds[seed_idx, 0]
            seed_y = st.seeds[seed_idx, 1]

            collision = _check_aabb_collision(
                state.player_x, player_pickup_y,
                self.consts.PLAYER_SIZE[0], pickup_height,
                seed_x, seed_y,
                self.consts.SEED_SIZE[0], self.consts.SEED_SIZE[1],
            )

            return jax.lax.cond(
                collision,
                lambda s: self._seed_picked_up(s, seed_idx),
                lambda s: s,
                st,
            )

        # Check all seeds using fori_loop
        return jax.lax.fori_loop(
            0,
            state.seeds.shape[0],
            check_and_pickup_seed,
            state,
        )

    def _update_and_spawn_seeds(self, state: RoadRunnerState) -> RoadRunnerState:
        """
        Update seed positions (apply scrolling, despawn off-screen) and spawn new seeds.
        Combined function for efficiency - seeds update and spawn logic together.
        """
        consts = self.consts
        level_idx = self._get_level_index(state)
        road_top, road_bottom, _ = self._get_road_bounds(state)
        road_top = road_top.astype(jnp.int32)
        road_bottom = road_bottom.astype(jnp.int32)
        if self._level_count > 0:
            spawn_seeds_enabled = self._level_spawn_seeds[level_idx]
            seed_spawn_bounds = self._seed_spawn_intervals[level_idx]
        else:
            spawn_seeds_enabled = jnp.array(True, dtype=jnp.bool_)
            seed_spawn_bounds = jnp.array(
                [consts.SEED_SPAWN_MIN_INTERVAL, consts.SEED_SPAWN_MAX_INTERVAL],
                dtype=jnp.int32,
            )
        # Update seed positions: apply scrolling and despawn off-screen seeds
        seed_x = state.seeds[:, 0]
        # Move active seeds when scrolling, then despawn off-screen ones
        scroll_offset = jnp.where(state.is_scrolling, consts.PLAYER_MOVE_SPEED, 0)
        updated_x = jnp.where(seed_x >= 0, seed_x + scroll_offset, seed_x)
        # Mark seeds as inactive if they moved off-screen (x >= WIDTH)
        seed_active = (updated_x >= 0) & (updated_x < consts.WIDTH)
        updated_x = jnp.where(seed_active, updated_x, -1)
        updated_seeds = (
            state.seeds.at[:, 0]
            .set(updated_x)
            .at[:, 1]
            .set(jnp.where(seed_active, state.seeds[:, 1], -1))
        )

        # Prepare for spawning: split RNG and check conditions
        rng_spawn_y, rng_interval, rng_after = jax.random.split(state.rng, 3)
        available_slots = updated_x == -1
        should_spawn = (
            state.is_scrolling
            & (state.scrolling_step_counter >= state.next_seed_spawn_scroll_step)
            & jnp.any(available_slots)
            & spawn_seeds_enabled
        )

        def _spawn(st: RoadRunnerState) -> RoadRunnerState:
            slot_idx = jnp.argmax(available_slots)
            # Generate random Y position within road bounds
            spawn_min = road_top
            spawn_max = road_bottom - consts.SEED_SIZE[1]
            seed_y = jax.random.randint(
                rng_spawn_y,
                (),
                spawn_min,
                jnp.maximum(spawn_min + 1, spawn_max + 1),
                dtype=jnp.int32,
            )
            # Spawn at x=0, update next spawn step
            next_spawn_step = state.scrolling_step_counter + jax.random.randint(
                rng_interval,
                (),
                seed_spawn_bounds[0],
                seed_spawn_bounds[1] + 1,
                dtype=jnp.int32,
            )
            # Get the seeds id, then increment the next id in the state
            seed_id = st.next_seed_id
            st = st._replace(next_seed_id=seed_id+1)

            return st._replace(
                seeds=updated_seeds.at[slot_idx].set(
                    jnp.array([0, seed_y, seed_id], dtype=jnp.int32)
                ),
                next_seed_spawn_scroll_step=next_spawn_step,
                rng=rng_after,
            )

        return jax.lax.cond(
            should_spawn,
            _spawn,
            lambda st: st._replace(seeds=updated_seeds, rng=rng_after),
            state,
        )

    def _update_and_spawn_truck(self, state: RoadRunnerState) -> RoadRunnerState:
        """
        Update truck position (move right at TRUCK_SPEED + scroll offset) and spawn new truck.
        Trucks spawn regardless of scrolling state, using step_counter.
        Trucks are affected by road scrolling - they move with the scroll offset.
        """
        consts = self.consts
        level_idx = self._get_level_index(state)
        road_top, road_bottom, _ = self._get_road_bounds(state)
        road_top = road_top.astype(jnp.int32)
        road_bottom = road_bottom.astype(jnp.int32)
        if self._level_count > 0:
            spawn_trucks_enabled = self._level_spawn_trucks[level_idx]
            truck_spawn_bounds = self._truck_spawn_intervals[level_idx]
        else:
            spawn_trucks_enabled = jnp.array(True, dtype=jnp.bool_)
            truck_spawn_bounds = jnp.array(
                [consts.TRUCK_SPAWN_MIN_INTERVAL, consts.TRUCK_SPAWN_MAX_INTERVAL],
                dtype=jnp.int32,
            )

        # Update truck position: move active truck right, apply scrolling offset
        # Move by TRUCK_SPEED, plus scroll offset when scrolling is active
        scroll_offset = jnp.where(state.is_scrolling, consts.PLAYER_MOVE_SPEED, 0)
        updated_truck_x = jnp.where(
            state.truck_x >= 0,
            state.truck_x + consts.TRUCK_SPEED + scroll_offset,
            state.truck_x
        )
        # Despawn if off-screen
        truck_active = (updated_truck_x >= 0) & (updated_truck_x < consts.WIDTH)
        updated_truck_x = jnp.where(truck_active, updated_truck_x, -1)
        updated_truck_y = jnp.where(truck_active, state.truck_y, -1)

        # Prepare for spawning: split RNG and check conditions
        rng_spawn_y, rng_interval, rng_after = jax.random.split(state.rng, 3)
        should_spawn = (
            (updated_truck_x < 0)  # No truck currently active
            & (state.step_counter >= state.next_truck_spawn_step)
            & spawn_trucks_enabled
        )

        def _spawn(st: RoadRunnerState) -> RoadRunnerState:
            # Generate random Y position within road bounds
            spawn_min = road_top
            spawn_max = road_bottom - consts.TRUCK_SIZE[1]
            truck_y = jax.random.randint(
                rng_spawn_y,
                (),
                spawn_min,
                jnp.maximum(spawn_min + 1, spawn_max + 1),
                dtype=jnp.int32,
            )
            # Spawn at x=0, update next spawn step
            next_spawn_step = st.step_counter + jax.random.randint(
                rng_interval,
                (),
                truck_spawn_bounds[0],
                truck_spawn_bounds[1] + 1,
                dtype=jnp.int32,
            )

            return st._replace(
                truck_x=jnp.array(0, dtype=jnp.int32),
                truck_y=truck_y,
                next_truck_spawn_step=next_spawn_step,
                rng=rng_after,
            )

        return jax.lax.cond(
            should_spawn,
            _spawn,
            lambda st: st._replace(
                truck_x=updated_truck_x,
                truck_y=updated_truck_y,
                rng=rng_after
            ),
            state,
        )

    def _check_truck_collisions(self, state: RoadRunnerState) -> RoadRunnerState:
        """
        Check for collisions between truck and player/enemy.
        Uses AABB (Axis-Aligned Bounding Box) collision detection.
        """
        # Early return if truck is inactive
        truck_active = state.truck_x >= 0
        return jax.lax.cond(
            truck_active,
            lambda st: self._check_truck_collisions_active(st),
            lambda st: st,
            state,
        )

    def _check_truck_collisions_active(self, state: RoadRunnerState) -> RoadRunnerState:
        """Check collisions when truck is active."""
        # Truck collision area (lower half, using TRUCK_COLLISION_OFFSET)
        truck_collision_y = state.truck_y + self.consts.TRUCK_COLLISION_OFFSET
        truck_collision_height = self.consts.TRUCK_SIZE[1] - self.consts.TRUCK_COLLISION_OFFSET

        # Player pickup area (lower portion)
        player_pickup_y = state.player_y + self.consts.PLAYER_PICKUP_OFFSET
        pickup_height = self.consts.PLAYER_SIZE[1] - self.consts.PLAYER_PICKUP_OFFSET

        # Check player-truck collision
        player_collision = _check_aabb_collision(
            state.player_x, player_pickup_y,
            self.consts.PLAYER_SIZE[0], pickup_height,
            state.truck_x, truck_collision_y,
            self.consts.TRUCK_SIZE[0], truck_collision_height,
        )

        # Check enemy-truck collision
        enemy_collision = _check_aabb_collision(
            state.enemy_x, state.enemy_y,
            self.consts.ENEMY_SIZE[0], self.consts.ENEMY_SIZE[1],
            state.truck_x, truck_collision_y,
            self.consts.TRUCK_SIZE[0], truck_collision_height,
        )

        # Handle player collision (triggers round reset)
        def handle_player_collision(st: RoadRunnerState) -> RoadRunnerState:
            return st._replace(
                is_round_over=True,
                player_x=(st.truck_x + self.consts.TRUCK_SIZE[0] + 2).astype(jnp.int32),
                player_y=st.player_y,
            )

        state_after_player = jax.lax.cond(
            player_collision,
            handle_player_collision,
            lambda st: st,
            state,
        )

        # Handle enemy collision (print debug message)
        def handle_enemy_collision(st: RoadRunnerState) -> RoadRunnerState:
            jax.debug.print("Enemy hit by truck!")
            return st

        state_after_enemy = jax.lax.cond(
            enemy_collision,
            handle_enemy_collision,
            lambda st: st,
            state_after_player,
        )

        return state_after_enemy
    
    def _update_and_spawn_ravines(self, state: RoadRunnerState) -> RoadRunnerState:
        """
        Update ravine positions (move left with scroll speed) and spawn new ravines.
        Ravines are fixed to the road, so they move exactly with the scroll speed.
        """
        consts = self.consts
        level_idx = self._get_level_index(state)
        if self._level_count > 0:
            spawn_ravines_enabled = self._level_spawn_ravines[level_idx]
            ravine_spawn_bounds = self._ravine_spawn_intervals[level_idx]
        else:
            spawn_ravines_enabled = jnp.array(False, dtype=jnp.bool_)
            ravine_spawn_bounds = jnp.array(
                [consts.RAVINE_SPAWN_MIN_INTERVAL, consts.RAVINE_SPAWN_MAX_INTERVAL],
                dtype=jnp.int32,
            )

        # Get current road bounds to check if height matches ravine height
        road_top, road_bottom, road_height = self._get_road_bounds(state)
        
        # Only spawn if road height is compatible (== 32)
        height_compatible = road_height == 32
        
        should_spawn_active = spawn_ravines_enabled & height_compatible
        
        # Update ravine positions: move active ravines.
        # Ravines move ONLY when scrolling happens.
        ravine_x = state.ravines[:, 0]
        
        scroll_offset = jnp.where(state.is_scrolling, consts.PLAYER_MOVE_SPEED, 0)
        
        # Update positions
        ravine_x = jnp.where(
            state.ravines[:, 0] >= 0,
            state.ravines[:, 0] + scroll_offset,
            state.ravines[:, 0]
        )
        
        # Despawn if off-screen (>= WIDTH)
        ravine_active = (ravine_x >= 0) & (ravine_x < consts.WIDTH)
        updated_x = jnp.where(ravine_active, ravine_x, -1)
        updated_y = jnp.where(ravine_active, state.ravines[:, 1], -1)
        
        updated_ravines = jnp.stack([updated_x, updated_y], axis=-1)
        
        # Spawning Logic
        rng_spawn, rng_interval, rng_after = jax.random.split(state.rng, 3)
        available_slots = updated_x == -1
        
        should_spawn = (
            state.is_scrolling
            & (state.scrolling_step_counter >= state.next_ravine_spawn_scroll_step)
            & jnp.any(available_slots)
            & should_spawn_active
        )
        
        def _spawn_ravine(st: RoadRunnerState) -> RoadRunnerState:
            slot_idx = jnp.argmax(available_slots)
            
            spawn_y = road_top
            
            next_spawn_step = state.scrolling_step_counter + jax.random.randint(
                rng_interval,
                (),
                ravine_spawn_bounds[0],
                ravine_spawn_bounds[1] + 1,
                dtype=jnp.int32,
            )
            
            new_ravine = jnp.array([0, spawn_y], dtype=jnp.int32)
            
            return st._replace(
                ravines=updated_ravines.at[slot_idx].set(new_ravine),
                next_ravine_spawn_scroll_step=next_spawn_step,
                rng=rng_after,
            )

        return jax.lax.cond(
            should_spawn,
            _spawn_ravine,
            lambda st: st._replace(ravines=updated_ravines, rng=rng_after),
            state,
        )

    def _check_ravine_collisions(self, state: RoadRunnerState) -> RoadRunnerState:
        """
        Check collision with ravines.
        If player overlaps with ravine AND is NOT jumping, they fall (die).
        """
        # Player feet area (bottom 4 pixels)
        player_feet_y = state.player_y + self.consts.PLAYER_SIZE[1] - 4
        feet_height = 4

        def check_ravine(i, st):
            r_x = st.ravines[i, 0]
            r_y = st.ravines[i, 1]
            active = r_x >= 0

            overlap = _check_aabb_collision(
                state.player_x, player_feet_y,
                self.consts.PLAYER_SIZE[0], feet_height,
                r_x, r_y,
                self.consts.RAVINE_SIZE[0], self.consts.RAVINE_SIZE[1],
            )

            collision = active & overlap & jnp.logical_not(st.is_jumping)

            return jax.lax.cond(
                collision,
                lambda s: s._replace(instant_death=True),
                lambda s: s,
                st
            )

        return jax.lax.fori_loop(0, 3, check_ravine, state)

    def _update_and_spawn_landmines(self, state: RoadRunnerState) -> RoadRunnerState:
        """
        Update landmine positions (move with scroll) and spawn new landmines.
        Only one landmine active at a time.
        """
        consts = self.consts
        level_idx = self._get_level_index(state)
        road_top, road_bottom, _ = self._get_road_bounds(state)
        road_top = road_top.astype(jnp.int32)
        road_bottom = road_bottom.astype(jnp.int32)
        
        if self._level_count > 0:
            spawn_landmines_enabled = self._level_spawn_landmines[level_idx]
            landmine_spawn_bounds = self._landmine_spawn_intervals[level_idx]
        else:
            spawn_landmines_enabled = jnp.array(False, dtype=jnp.bool_)
            landmine_spawn_bounds = jnp.array(
                [consts.LANDMINE_SPAWN_MIN_INTERVAL, consts.LANDMINE_SPAWN_MAX_INTERVAL],
                dtype=jnp.int32,
            )

        # Update landmine position
        # Move by scroll offset when scrolling is active
        scroll_offset = jnp.where(state.is_scrolling, consts.PLAYER_MOVE_SPEED, 0)
        
        updated_landmine_x = jnp.where(
            state.landmine_x >= 0,
            state.landmine_x + scroll_offset,
            state.landmine_x
        )
        
        # Despawn if off-screen
        landmine_active = (updated_landmine_x >= 0) & (updated_landmine_x < consts.WIDTH)
        updated_landmine_x = jnp.where(landmine_active, updated_landmine_x, -1)
        updated_landmine_y = jnp.where(landmine_active, state.landmine_y, -1)

        # Prepare for spawning
        rng_spawn_y, rng_interval, rng_after = jax.random.split(state.rng, 3)
        
        should_spawn = (
            (updated_landmine_x < 0)  # No landmine currently active
            & (state.step_counter >= state.next_landmine_spawn_step)
            & spawn_landmines_enabled
            & (state.death_timer == 0) # Don't spawn if dying
            & jnp.logical_not(state.is_in_transition)
        )

        def _spawn(st: RoadRunnerState) -> RoadRunnerState:
            # Generate random Y position within road bounds
            spawn_min = road_top
            spawn_max = road_bottom - consts.LANDMINE_SIZE[1]
            landmine_y = jax.random.randint(
                rng_spawn_y,
                (),
                spawn_min,
                jnp.maximum(spawn_min + 1, spawn_max + 1),
                dtype=jnp.int32,
            )
            
            landmine_x = jnp.array(0, dtype=jnp.int32)
            
            next_spawn_step = st.step_counter + jax.random.randint(
                rng_interval,
                (),
                landmine_spawn_bounds[0],
                landmine_spawn_bounds[1] + 1,
                dtype=jnp.int32,
            )

            return st._replace(
                landmine_x=landmine_x,
                landmine_y=landmine_y,
                next_landmine_spawn_step=next_spawn_step,
                rng=rng_after,
            )

        return jax.lax.cond(
            should_spawn,
            _spawn,
            lambda st: st._replace(
                landmine_x=updated_landmine_x,
                landmine_y=updated_landmine_y,
                rng=rng_after
            ),
            state,
        )

    def _check_landmine_collisions(self, state: RoadRunnerState) -> RoadRunnerState:
        """
        Check collision between player and landmine.
        """
        active = state.landmine_x >= 0

        # Player pickup area (lower portion)
        player_pickup_y = state.player_y + self.consts.PLAYER_PICKUP_OFFSET
        pickup_height = self.consts.PLAYER_SIZE[1] - self.consts.PLAYER_PICKUP_OFFSET

        overlap = _check_aabb_collision(
            state.player_x, player_pickup_y,
            self.consts.PLAYER_SIZE[0], pickup_height,
            state.landmine_x, state.landmine_y,
            self.consts.LANDMINE_SIZE[0], self.consts.LANDMINE_SIZE[1],
        )
        collision = active & overlap & (state.death_timer == 0)

        def handle_collision(st: RoadRunnerState) -> RoadRunnerState:
            # Trigger death animation
            # Set timer. The step function handles the rest (freeze and then reset).
            return st._replace(
                death_timer=jnp.array(self.consts.DEATH_ANIMATION_DURATION, dtype=jnp.int32),
                landmine_x=jnp.array(-1, dtype=jnp.int32) # Remove the mine
            )
            
        return jax.lax.cond(
            collision,
            handle_collision,
            lambda s: s,
            state
        )

    def reset(self, key=None) -> Tuple[RoadRunnerObservation, RoadRunnerState]:
        # Initialize RNG key
        if key is None:
            key = jax.random.PRNGKey(42)

        state = RoadRunnerState(
            player_x=jnp.array(self.consts.PLAYER_START_X, dtype=jnp.int32),
            player_y=jnp.array(self.consts.PLAYER_START_Y, dtype=jnp.int32),
            player_x_history=jnp.array(
                [self.consts.PLAYER_START_X] * self.consts.ENEMY_REACTION_DELAY,
                dtype=jnp.int32,
            ),
            player_y_history=jnp.array(
                [self.consts.PLAYER_START_Y] * self.consts.ENEMY_REACTION_DELAY,
                dtype=jnp.int32,
            ),
            enemy_x=jnp.array(self.consts.ENEMY_X, dtype=jnp.int32),
            enemy_y=jnp.array(self.consts.ENEMY_Y, dtype=jnp.int32),
            step_counter=jnp.array(0, dtype=jnp.int32),
            player_is_moving=jnp.array(False, dtype=jnp.bool_),
            player_looks_right=jnp.array(False, dtype=jnp.bool_),
            enemy_is_moving=jnp.array(False, dtype=jnp.bool_),
            enemy_looks_right=jnp.array(False, dtype=jnp.bool_),
            score=jnp.array(0, dtype=jnp.int32),
            is_scrolling=jnp.array(False, dtype=jnp.bool_),
            scrolling_step_counter=jnp.array(0, dtype=jnp.int32),
            is_round_over=jnp.array(False, dtype=jnp.bool_),
            seeds=jnp.full((4, 3), -1, dtype=jnp.int32),  # Initialize all seeds as inactive (-1, -1)
            next_seed_spawn_scroll_step=jnp.array(0, dtype=jnp.int32),
            rng=key,
            seed_pickup_streak=jnp.array(0, dtype=jnp.int32),
            next_seed_id=jnp.array(0, dtype=jnp.int32),
            last_picked_up_seed_id=jnp.array(0, dtype=jnp.int32),
            truck_x=jnp.array(-1, dtype=jnp.int32),
            truck_y=jnp.array(-1, dtype=jnp.int32),
            next_truck_spawn_step=jnp.array(0, dtype=jnp.int32),
            current_level=jnp.array(0, dtype=jnp.int32),
            level_transition_timer=jnp.array(0, dtype=jnp.int32),
            is_in_transition=jnp.array(False, dtype=jnp.bool_),
            lives=jnp.array(self.consts.STARTING_LIVES, dtype=jnp.int32),
            jump_timer=jnp.array(0, dtype=jnp.int32),
            is_jumping=jnp.array(False, dtype=jnp.bool_),
            ravines=jnp.full((3, 2), -1, dtype=jnp.int32),
            next_ravine_spawn_scroll_step=jnp.array(0, dtype=jnp.int32),
            instant_death=jnp.array(False, dtype=jnp.bool_),
            landmine_x=jnp.array(-1, dtype=jnp.int32),
            landmine_y=jnp.array(-1, dtype=jnp.int32),
            next_landmine_spawn_step=jnp.array(0, dtype=jnp.int32),
            death_timer=jnp.array(0, dtype=jnp.int32),
            enemy_speed_phase_start=jnp.array(0, dtype=jnp.int32),
        )
        state = self._initialize_spawn_timers(state, jnp.array(0, dtype=jnp.int32))
        initial_obs = self._get_observation(state)
        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, state: RoadRunnerState, action: chex.Array
    ) -> Tuple[RoadRunnerObservation, RoadRunnerState, float, bool, None]:
        state = self._handle_level_transition(state)
        operand = (state, action)

        def _transition_branch(data):
            st, _ = data
            st = st._replace(step_counter=st.step_counter + 1)
            obs = self._get_observation(st)
            return obs, st, 0.0, False, None

        def _gameplay_branch(data):
            st, act = data
            st = self._player_step(st, act)
            st = self._enemy_step(st)
            st = self._check_game_over(st)
            st = self._update_and_spawn_seeds(st)
            st = self._check_seed_collisions(st)
            st = self._update_and_spawn_truck(st)
            st = self._check_truck_collisions(st)
            st = self._update_and_spawn_ravines(st)
            st = self._check_ravine_collisions(st)
            st = self._update_and_spawn_landmines(st)
            st = self._check_landmine_collisions(st)
            st = self._check_level_completion(st)

            player_at_end = st.player_x >= self.consts.WIDTH - self.consts.PLAYER_SIZE[0]
            # Check if we should reset immediately (instant_death) OR if standard round end condition met (player reached end)
            should_reset = st.instant_death | (st.is_round_over & player_at_end)

            st = jax.lax.cond(
                should_reset, self._handle_round_end, lambda inner: inner, st
            )

            st = st._replace(step_counter=st.step_counter + 1)
            obs = self._get_observation(st)
            return obs, st, 0.0, False, None

        def _death_timer_branch(data):
             st, _ = data
             # Decrement death timer
             st = st._replace(death_timer=jnp.maximum(st.death_timer - 1, 0))

             # If timer finished, trigger life loss
             should_reset = st.death_timer == 0
             st = jax.lax.cond(
                 should_reset,
                 lambda s: s._replace(instant_death=True),
                 lambda s: s,
                 st
             )

             st = jax.lax.cond(
                 st.instant_death, self._handle_round_end, lambda inner: inner, st
             )

             obs = self._get_observation(st)
             return obs, st, 0.0, False, None

        # Check for death timer
        is_dying = state.death_timer > 0

        def _gameplay_or_death_branch(op):
            s, a = op
            return jax.lax.cond(
                s.death_timer > 0,
                _death_timer_branch,
                _gameplay_branch,
                (s, a)
            )

        observation, state, reward, done, info = jax.lax.cond(
            state.is_in_transition, _transition_branch, _gameplay_or_death_branch, operand
        )

        return observation, state, reward, done, info

    def _game_over_reset(self, state: RoadRunnerState) -> RoadRunnerState:
        """Game Over: Restart from beginning."""
        rng, new_key = jax.random.split(state.rng)
        _, new_state = self.reset(new_key)
        return new_state

    def _next_life_reset(self, state: RoadRunnerState) -> RoadRunnerState:
        """Lost a life: Partial reset."""
        reset_state = state._replace(
            player_x=jnp.array(self.consts.PLAYER_START_X, dtype=jnp.int32),
            player_y=jnp.array(self.consts.PLAYER_START_Y, dtype=jnp.int32),
            player_x_history=jnp.array(
                [self.consts.PLAYER_START_X] * self.consts.ENEMY_REACTION_DELAY,
                dtype=jnp.int32,
            ),
            player_y_history=jnp.array(
                [self.consts.PLAYER_START_Y] * self.consts.ENEMY_REACTION_DELAY,
                dtype=jnp.int32,
            ),
            enemy_x=jnp.array(self.consts.ENEMY_X, dtype=jnp.int32),
            enemy_y=jnp.array(self.consts.ENEMY_Y, dtype=jnp.int32),
            is_round_over=jnp.array(False, dtype=jnp.bool_),
            seeds=jnp.full((4, 3), -1, dtype=jnp.int32),
            scrolling_step_counter=jnp.array(0, dtype=jnp.int32),
            seed_pickup_streak=jnp.array(0, dtype=jnp.int32),
            last_picked_up_seed_id=jnp.array(0, dtype=jnp.int32),
            next_seed_id=jnp.array(0, dtype=jnp.int32),
            truck_x=jnp.array(-1, dtype=jnp.int32),
            truck_y=jnp.array(-1, dtype=jnp.int32),
            next_seed_spawn_scroll_step=jnp.array(0, dtype=jnp.int32),
            next_truck_spawn_step=jnp.array(0, dtype=jnp.int32),
            lives=state.lives - 1,
            jump_timer=jnp.array(0, dtype=jnp.int32),
            is_jumping=jnp.array(False, dtype=jnp.bool_),
            instant_death=jnp.array(False, dtype=jnp.bool_),
            ravines=jnp.full((3, 2), -1, dtype=jnp.int32),
            landmine_x=jnp.array(-1, dtype=jnp.int32),
            landmine_y=jnp.array(-1, dtype=jnp.int32),
            next_landmine_spawn_step=jnp.array(0, dtype=jnp.int32),
            death_timer=jnp.array(0, dtype=jnp.int32),
            enemy_speed_phase_start=jnp.array(0, dtype=jnp.int32),
        )
        level_idx = self._get_level_index(reset_state)
        return self._initialize_spawn_timers(reset_state, level_idx)

    def _handle_round_end(self, state: RoadRunnerState) -> RoadRunnerState:
        """Handle end of round - either next life or game over."""
        return jax.lax.cond(
            state.lives > 1,
            self._next_life_reset,
            self._game_over_reset,
            state
        )

    def _check_level_completion(self, state: RoadRunnerState) -> RoadRunnerState:
        target_distance = self.consts.LEVEL_COMPLETE_SCROLL_DISTANCE
        level_complete = state.scrolling_step_counter >= target_distance
        max_level_index = max(self._level_count - 1, 0)
        has_next_level = state.current_level < max_level_index
        ready_for_transition = (
            level_complete & has_next_level & jnp.logical_not(state.is_in_transition)
        )

        def _start_transition(st: RoadRunnerState) -> RoadRunnerState:
            jax.debug.print(
                "Level {lvl} reached scroll {scroll} → preparing transition",
                lvl=st.current_level + 1,
                scroll=st.scrolling_step_counter,
            )
            return st._replace(
                is_in_transition=jnp.array(True, dtype=jnp.bool_),
                level_transition_timer=jnp.array(
                    self.consts.LEVEL_TRANSITION_DURATION, dtype=jnp.int32
                ),
                current_level=jnp.array(
                    jnp.minimum(st.current_level + 1, max_level_index), dtype=jnp.int32
                ),
                scrolling_step_counter=jnp.array(0, dtype=jnp.int32),
            )

        return jax.lax.cond(ready_for_transition, _start_transition, lambda st: st, state)

    def _handle_level_transition(self, state: RoadRunnerState) -> RoadRunnerState:
        def _no_transition(st: RoadRunnerState) -> RoadRunnerState:
            return st

        def _process_transition(st: RoadRunnerState) -> RoadRunnerState:
            new_timer = jnp.maximum(st.level_transition_timer - 1, 0)
            st = st._replace(level_transition_timer=new_timer)
            transition_complete = new_timer == 0

            def _complete(s: RoadRunnerState) -> RoadRunnerState:
                reset_state = self._reset_level_entities(s)
                level_idx = self._get_level_index(reset_state)
                reset_state = self._initialize_spawn_timers(reset_state, level_idx)
                return reset_state._replace(
                    is_in_transition=jnp.array(False, dtype=jnp.bool_),
                    level_transition_timer=jnp.array(0, dtype=jnp.int32),
                )

            return jax.lax.cond(transition_complete, _complete, lambda s: s, st)

        return jax.lax.cond(
            state.is_in_transition, _process_transition, _no_transition, state
        )

    def _reset_level_entities(self, state: RoadRunnerState) -> RoadRunnerState:
        history_x = jnp.full(
            (self.consts.ENEMY_REACTION_DELAY,),
            self.consts.PLAYER_START_X,
            dtype=jnp.int32,
        )
        history_y = jnp.full(
            (self.consts.ENEMY_REACTION_DELAY,),
            self.consts.PLAYER_START_Y,
            dtype=jnp.int32,
        )
        cleared_seeds = jnp.full_like(state.seeds, -1)
        return state._replace(
            player_x=jnp.array(self.consts.PLAYER_START_X, dtype=jnp.int32),
            player_y=jnp.array(self.consts.PLAYER_START_Y, dtype=jnp.int32),
            player_x_history=history_x,
            player_y_history=history_y,
            player_is_moving=jnp.array(False, dtype=jnp.bool_),
            player_looks_right=jnp.array(False, dtype=jnp.bool_),
            enemy_x=jnp.array(self.consts.ENEMY_X, dtype=jnp.int32),
            enemy_y=jnp.array(self.consts.ENEMY_Y, dtype=jnp.int32),
            enemy_is_moving=jnp.array(False, dtype=jnp.bool_),
            enemy_looks_right=jnp.array(False, dtype=jnp.bool_),
            seeds=cleared_seeds,
            seed_pickup_streak=jnp.array(0, dtype=jnp.int32),
            last_picked_up_seed_id=jnp.array(0, dtype=jnp.int32),
            next_seed_id=jnp.array(0, dtype=jnp.int32),
            truck_x=jnp.array(-1, dtype=jnp.int32),
            truck_y=jnp.array(-1, dtype=jnp.int32),
            is_round_over=jnp.array(False, dtype=jnp.bool_),
            is_scrolling=jnp.array(False, dtype=jnp.bool_),
            jump_timer=jnp.array(0, dtype=jnp.int32),
            is_jumping=jnp.array(False, dtype=jnp.bool_),
            ravines=jnp.full((3, 2), -1, dtype=jnp.int32),
            next_ravine_spawn_scroll_step=jnp.array(0, dtype=jnp.int32),
            instant_death=jnp.array(False, dtype=jnp.bool_),
            enemy_speed_phase_start=jnp.array(0, dtype=jnp.int32),
        )

    def _get_level_index(self, state: RoadRunnerState) -> jnp.ndarray:
        if self._level_count == 0:
            return jnp.array(0, dtype=jnp.int32)
        max_index = self._level_count - 1
        return jnp.clip(state.current_level, 0, max_index).astype(jnp.int32)

    def _get_current_level_config(self, state: RoadRunnerState) -> LevelConfig:
        if not self.consts.levels:
            return RoadRunner_Level_1
        max_index = len(self.consts.levels) - 1
        level_idx = jnp.clip(state.current_level, 0, max_index).astype(jnp.int32)
        branches = tuple(lambda cfg=cfg: cfg for cfg in self.consts.levels)
        return jax.lax.switch(level_idx, branches)

    def _initialize_spawn_timers(
        self, state: RoadRunnerState, level_idx: jnp.ndarray
    ) -> RoadRunnerState:
        if self._level_count > 0:
            seed_bounds = self._seed_spawn_intervals[level_idx]
            truck_bounds = self._truck_spawn_intervals[level_idx]
            ravine_bounds = self._ravine_spawn_intervals[level_idx]
            landmine_bounds = self._landmine_spawn_intervals[level_idx]
        else:
            seed_bounds = jnp.array(
                [self.consts.SEED_SPAWN_MIN_INTERVAL, self.consts.SEED_SPAWN_MAX_INTERVAL],
                dtype=jnp.int32,
            )
            truck_bounds = jnp.array(
                [self.consts.TRUCK_SPAWN_MIN_INTERVAL, self.consts.TRUCK_SPAWN_MAX_INTERVAL],
                dtype=jnp.int32,
            )
            ravine_bounds = jnp.array(
                [
                    self.consts.RAVINE_SPAWN_MIN_INTERVAL,
                    self.consts.RAVINE_SPAWN_MAX_INTERVAL,
                ],
                dtype=jnp.int32,
            )

        rng, seed_key = jax.random.split(state.rng)
        next_seed_spawn_scroll_step = state.scrolling_step_counter + jax.random.randint(
            seed_key,
            (),
            seed_bounds[0],
            seed_bounds[1] + 1,
            dtype=jnp.int32,
        )
        rng, truck_key = jax.random.split(rng)
        next_truck_spawn_step = state.step_counter + jax.random.randint(
            truck_key,
            (),
            truck_bounds[0],
            truck_bounds[1] + 1,
            dtype=jnp.int32,
        )
        rng, ravine_key = jax.random.split(rng)
        next_ravine_spawn_scroll_step = state.scrolling_step_counter + jax.random.randint(
            ravine_key,
            (),
            ravine_bounds[0],
            ravine_bounds[1] + 1,
            dtype=jnp.int32,
        )
        rng, landmine_key = jax.random.split(rng)
        next_landmine_spawn_step = state.step_counter + jax.random.randint(
            landmine_key,
            (),
            landmine_bounds[0],
            landmine_bounds[1] + 1,
            dtype=jnp.int32,
        )
        return state._replace(
            rng=rng,
            next_seed_spawn_scroll_step=next_seed_spawn_scroll_step,
            next_truck_spawn_step=next_truck_spawn_step,
            next_ravine_spawn_scroll_step=next_ravine_spawn_scroll_step,
            next_landmine_spawn_step=next_landmine_spawn_step,
        )

    def _get_current_road_section(self, state: RoadRunnerState) -> RoadSectionConfig:
        return _get_road_section_for_scroll(
            state,
            self._level_count,
            self._max_road_sections,
            self._road_section_data,
            self._road_section_counts,
            self.consts,
        )

    def _get_road_bounds(self, state: RoadRunnerState) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        section = self._get_current_road_section(state)
        road_height = jnp.clip(section.road_height, 1, self.consts.ROAD_HEIGHT)
        max_top = self.consts.ROAD_HEIGHT - road_height
        section_top = jnp.clip(section.road_top, 0, max_top)
        road_top = self.consts.ROAD_TOP_Y + section_top
        road_bottom = road_top + road_height
        return road_top, road_bottom, road_height

    def render(self, state: RoadRunnerState) -> jnp.ndarray:
        return self.renderer.render(state)

    def _get_observation(self, state: RoadRunnerState) -> RoadRunnerObservation:
        player = EntityPosition(
            x=state.player_x,
            y=state.player_y,
            width=jnp.array(self.consts.PLAYER_SIZE[0]),
            height=jnp.array(self.consts.PLAYER_SIZE[1]),
        )
        enemy = EntityPosition(
            x=state.enemy_x,
            y=state.enemy_y,
            width=jnp.array(self.consts.ENEMY_SIZE[0]),
            height=jnp.array(self.consts.ENEMY_SIZE[1]),
        )
        
        # Valid ravines have x >= 0 (strictly active in our logic, although we set to -1 when inactive)
        active_ravines_mask = state.ravines[:, 0] >= 0

        # Let's find the ravine with smallest x >= 0
        ravine_x = state.ravines[:, 0]
        ravine_y = state.ravines[:, 1]
        
        # Mask out inactive ones with a large value
        masked_x = jnp.where(active_ravines_mask, ravine_x, self.consts.WIDTH * 2)
        idx = jnp.argmin(masked_x)
        
        nearest_ravine_x = ravine_x[idx]
        nearest_ravine_y = ravine_y[idx]
        
        is_active = active_ravines_mask[idx]
        
        ravine_obs = EntityPosition(
            x=jnp.where(is_active, nearest_ravine_x, jnp.array(0, dtype=jnp.int32)),
            y=jnp.where(is_active, nearest_ravine_y, jnp.array(0, dtype=jnp.int32)),
            width=jnp.where(is_active, jnp.array(self.consts.RAVINE_SIZE[0]), jnp.array(0)),
            height=jnp.where(is_active, jnp.array(self.consts.RAVINE_SIZE[1]), jnp.array(0)),
        )

        return RoadRunnerObservation(
             player=player, 
             enemy=enemy, 
             score=state.score, 
             ravine=ravine_obs
        )

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))

    def observation_space(self) -> spaces.Dict:
        # Simplified observation space
        return spaces.Dict(
            {
                "player": spaces.Dict(
                    {
                        "x": spaces.Box(
                            low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32
                        ),
                        "y": spaces.Box(
                            low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.int32
                        ),
                        "width": spaces.Box(
                            low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32
                        ),
                        "height": spaces.Box(
                            low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.int32
                        ),
                    }
                ),
                "enemy": spaces.Dict(
                    {
                        "x": spaces.Box(
                            low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32
                        ),
                        "y": spaces.Box(
                            low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.int32
                        ),
                        "width": spaces.Box(
                            low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32
                        ),
                        "height": spaces.Box(
                            low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.int32
                        ),
                    }
                ),
                "score": spaces.Box(
                    low=0, high=jnp.iinfo(jnp.int32).max, shape=(), dtype=jnp.int32
                ),
                "ravine": spaces.Dict(
                    {
                        "x": spaces.Box(
                            low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32
                        ),
                        "y": spaces.Box(
                            low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.int32
                        ),
                        "width": spaces.Box(
                            low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32
                        ),
                        "height": spaces.Box(
                            low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.int32
                        ),
                    }
                ),
            }
        )

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.consts.HEIGHT, self.consts.WIDTH, 3),
            dtype=jnp.uint8,
        )


# --- Renderer Class (Simplified) ---
class RoadRunnerRenderer(JAXGameRenderer):
    def __init__(self, consts: RoadRunnerConstants = None):
        super().__init__()
        self.consts = consts or RoadRunnerConstants()
        self.deco_id_to_sprite = {
            0: "cactus",
            1: "sign_this_way",
            2: "sign_birdseed",
            3: "sign_cars_ahead",
            4: "sign_exit"
        }

        self.config = render_utils.RendererConfig(
            game_dimensions=(self.consts.HEIGHT, self.consts.WIDTH),
            channels=3,
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)

        wall_sprite_top = self._create_wall_sprite(self.consts.WALL_TOP_HEIGHT)
        wall_sprite_bottom = self._create_wall_sprite(self.consts.WALL_BOTTOM_HEIGHT)
        road_sprite = self._create_road_sprite()
        life_sprite = self._create_life_sprite()
        asset_config = self._get_asset_config(
            road_sprite, wall_sprite_bottom, life_sprite
        )
        sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/roadrunner"

        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS,
        ) = self.jr.load_and_setup_assets(asset_config, sprite_path)
        self._level_count = len(self.consts.levels)
        (
            self._max_road_sections,
            self._road_section_data,
            self._road_section_counts,
        ) = _build_road_section_arrays(self.consts.levels, self.consts)

    def _create_road_sprite(self) -> jnp.ndarray:
        ROAD_HEIGHT = self.consts.ROAD_HEIGHT
        WIDTH = self.consts.WIDTH
        DASH_LENGTH = self.consts.ROAD_DASH_LENGTH
        GAP_HEIGHT = self.consts.ROAD_GAP_HEIGHT
        PATTERN_WIDTH = self.consts.ROAD_PATTERN_WIDTH

        # Create a wider road for scrolling
        SCROLL_WIDTH = WIDTH + PATTERN_WIDTH

        road_color_rgba = jnp.array([0, 0, 0, 255], dtype=jnp.uint8)
        marking_color_rgba = jnp.array([255, 255, 255, 255], dtype=jnp.uint8)

        # Create a coordinate grid for the wider sprite
        y, x = jnp.indices((ROAD_HEIGHT, SCROLL_WIDTH))

        # Define the pattern using modular arithmetic
        is_marking_col = (x % PATTERN_WIDTH) >= (3 * DASH_LENGTH)
        is_marking_row = (y % (GAP_HEIGHT + 1)) == GAP_HEIGHT
        is_not_last_row = y < (ROAD_HEIGHT - 1)
        is_marking = is_marking_col & is_marking_row & is_not_last_row

        # Use jnp.where to create the sprite from the pattern
        road_sprite = jnp.where(
            is_marking[:, :, jnp.newaxis],
            marking_color_rgba,
            road_color_rgba,
        )

        return road_sprite

    def _create_wall_sprite(self, height: int) -> jnp.ndarray:
        wall_color_rgba = (*self.consts.WALL_COLOR, 255)
        wall_shape = (height, self.consts.WIDTH, 4)
        return jnp.tile(
            jnp.array(wall_color_rgba, dtype=jnp.uint8), (*wall_shape[:2], 1)
        )

    def _create_seed_sprite(self) -> jnp.ndarray:
        seed_color_rgba = (0, 0, 255, 255)
        seed_shape = (self.consts.SEED_SIZE[0], self.consts.SEED_SIZE[1], 4)
        return jnp.tile(
            jnp.array(seed_color_rgba, dtype=jnp.uint8), (*seed_shape[:2], 1)
        )

    def _create_life_sprite(self) -> jnp.ndarray:
        # Green square for lives
        life_color_rgba = (*self.consts.PLAYER_COLOR, 255)
        life_shape = (6, 6, 4) # 6x6 square
        return jnp.tile(
            jnp.array(life_color_rgba, dtype=jnp.uint8), (*life_shape[:2], 1)
        )

    def _get_render_section(self, state: RoadRunnerState) -> RoadSectionConfig:
        return _get_road_section_for_scroll(
            state,
            self._level_count,
            self._max_road_sections,
            self._road_section_data,
            self._road_section_counts,
            self.consts,
        )

    def _get_asset_config(
        self,
        road_sprite: jnp.ndarray,
        wall_sprite_bottom: jnp.ndarray,
        life_sprite: jnp.ndarray,
    ) -> list:
        asset_config = [
            {"name": "background", "type": "background", "file": "background.npy"},
            {"name": "player", "type": "single", "file": "roadrunner_stand.npy"},
            {"name": "player_run1", "type": "single", "file": "roadrunner_run1.npy"},
            {"name": "player_run2", "type": "single", "file": "roadrunner_run2.npy"},
            {"name": "player_jump", "type": "single", "file": "roadrunner_jump.npy"},
            {"name": "enemy", "type": "single", "file": "enemy_stand.npy"},
            {"name": "enemy_run1", "type": "single", "file": "enemy_run1.npy"},
            {"name": "enemy_run2", "type": "single", "file": "enemy_run2.npy"},
            {"name": "road", "type": "procedural", "data": road_sprite},
            {"name": "wall_bottom", "type": "procedural", "data": wall_sprite_bottom},
            {"name": "score_digits", "type": "digits", "pattern": "score_{}.npy"},
            {"name": "score_blank", "type": "single", "file": "score_10.npy"},
            {"name": "seed", "type": "single", "file": "birdseed.npy"},
            {"name": "truck", "type": "single", "file": "truck.npy"},
            {"name": "life", "type": "single", "file": "lives.npy"},
            {"name": "ravine", "type": "single", "file": "ravine.npy"},
            {"name": "landmine", "type": "single", "file": "landmine.npy"},
            {"name": "player_burnt", "type": "single", "file": "roadrunner_burnt.npy"},
            {"name": "end_of_level_1", "type": "single", "file": "end_of_level_1.npy"},
            {"name": "cactus", "type": "single", "file": "cactus.npy"},
            {"name": "tumbleweed", "type": "single", "file": "tumbleweed.npy"},
            {"name": "sign_this_way", "type": "single", "file": "sign_this_way.npy"},
            {"name": "sign_birdseed", "type": "single", "file": "sign_birdseed.npy"},
            {"name": "sign_cars_ahead", "type": "single", "file": "sign_cars_ahead.npy"},
            {"name": "sign_exit", "type": "single", "file": "sign_exit.npy"},
        ]

        return asset_config

    def _render_score(self, canvas: jnp.ndarray, score: jnp.ndarray) -> jnp.ndarray:
        MIN_DIGITS = 2
        MAX_DIGITS = 6

        safe_score = jnp.where(score == 0, 1, score)
        score_digit_count = (jnp.floor(jnp.log10(safe_score)) + 1).astype(int)
        visible_count = jnp.clip(score_digit_count, MIN_DIGITS, MAX_DIGITS)

        raw_digits = self.jr.int_to_digits(score, max_digits=MAX_DIGITS)

        digit_indices = jnp.arange(MAX_DIGITS)
        cutoff_index = MAX_DIGITS - visible_count
        is_visible_mask = digit_indices >= cutoff_index

        EMPTY_INDEX = 10
        score_digits = jnp.where(is_visible_mask, raw_digits, EMPTY_INDEX)

        original_masks = self.SHAPE_MASKS["score_digits"]
        blank_sprite = self.SHAPE_MASKS["score_blank"]
        score_digit_masks = jnp.concatenate([original_masks, blank_sprite[None, ...]], axis=0)

        score_x = (self.consts.WIDTH // 2 - (MAX_DIGITS * 6) // 2)
        score_y = 4

        canvas = self.jr.render_label_selective(
            canvas,
            score_x,
            score_y,
            score_digits,
            score_digit_masks,
            0,
            MAX_DIGITS,
            spacing=8, # offset, so width of digit + space
            max_digits_to_render=MAX_DIGITS,
        )
        return canvas

    def _render_lives(self, canvas: jnp.ndarray, lives: jnp.ndarray) -> jnp.ndarray:
        num_squares = jnp.maximum(lives - 1, 0)

        start_y = 24
        square_size = 6
        spacing = 2

        start_x = (self.consts.WIDTH // 3) + 2

        def render_square(i, r):
            x = start_x + i * (square_size + spacing)
            return self.jr.render_at(r, x, start_y, self.SHAPE_MASKS["life"])

        return jax.lax.fori_loop(0, num_squares, render_square, canvas)

    def _render_seeds(self, canvas: jnp.ndarray, seeds: jnp.ndarray) -> jnp.ndarray:
        # Only render active seeds (x >= 0)
        def render_seed(i, c):
            seed_x = seeds[i, 0]
            seed_y = seeds[i, 1]
            # Only render if seed is active (x >= 0)
            return jax.lax.cond(
                seed_x >= 0,
                lambda can: self.jr.render_at(
                    can, seed_x, seed_y, self.SHAPE_MASKS["seed"]
                ),
                lambda can: can,
                c,
            )

        # Use fori_loop to render all seeds
        return jax.lax.fori_loop(0, seeds.shape[0], render_seed, canvas)

    def _render_truck(self, canvas: jnp.ndarray, truck_x: chex.Array, truck_y: chex.Array) -> jnp.ndarray:
        # Only render if truck is active (x >= 0)
        return jax.lax.cond(
            truck_x >= 0,
            lambda can: self.jr.render_at(can, truck_x, truck_y, self.SHAPE_MASKS["truck"]),
            lambda can: can,
            canvas,
        )

    def _render_decorations(self, canvas: jnp.ndarray, state: RoadRunnerState) -> jnp.ndarray:
        # Calculate scroll position
        get_scroll_x = lambda slowdown: state.scrolling_step_counter * self.consts.PLAYER_MOVE_SPEED / (2 * slowdown)
        # Iterate over all levels defined in constants
        for i, level_cfg in enumerate(self.consts.levels):
            # Check if this is the active level
            # We use a relaxed check: (state.current_level == i)
            # Since we are inside a Python loop unrolling, we need to defer the check to JAX execution time.
            is_active_level = (state.current_level == i)

            # Iterate over all decorations in this level's config
            for deco in level_cfg.decorations:
                d_x, d_y, d_slowdown, d_type = deco

                # Get the sprite mask name for this type
                sprite_name = self.deco_id_to_sprite[d_type]
                sprite = self.SHAPE_MASKS[sprite_name]

                # Calculate screen position
                screen_x = get_scroll_x(d_slowdown) - d_x

                # Check visibility
                is_visible = (screen_x > 0) & (screen_x < self.consts.WIDTH - 16)

                # Combined condition: Level is active AND decoration is visible
                should_render = is_active_level & is_visible

                # Render
                canvas = jax.lax.cond(
                    should_render,
                    lambda c: self.jr.render_at(c, screen_x, d_y, sprite),
                    lambda c: c,
                    canvas
                )

        return canvas
    
    def _render_ravines(self, canvas: jnp.ndarray, ravines: jnp.ndarray) -> jnp.ndarray:
        # Only render active ravines (x >= 0)
        def render_ravine(i, c):
            r_x = ravines[i, 0]
            r_y = ravines[i, 1]
            # Only render if active
            return jax.lax.cond(
                r_x >= 0,
                lambda can: self.jr.render_at(
                    can, r_x, r_y, self.SHAPE_MASKS["ravine"]
                ),
                lambda can: can,
                c,
            )
        
        return jax.lax.fori_loop(0, ravines.shape[0], render_ravine, canvas)

    def _render_landmine(self, canvas: jnp.ndarray, landmine_x: chex.Array, landmine_y: chex.Array) -> jnp.ndarray:
        # Only render if active
        return jax.lax.cond(
            landmine_x >= 0,
            lambda can: self.jr.render_at(
                can, landmine_x, landmine_y, self.SHAPE_MASKS["landmine"]
            ),
            lambda can: can,
            canvas,
        )

    def _get_animated_sprite(
        self,
        is_moving: chex.Array,
        looks_right: chex.Array,
        step_counter: chex.Array,
        animation_speed: int,
        stand_sprite: jnp.ndarray,
        run1_sprite: jnp.ndarray,
        run2_sprite: jnp.ndarray,
    ) -> jnp.ndarray:
        sprites = (stand_sprite, run1_sprite, run2_sprite)
        run_frame_idx = (step_counter // animation_speed % 2) + 1
        sprite_idx = jax.lax.cond(
            is_moving,
            lambda: run_frame_idx,
            lambda: 0,
        )
        mask = jax.lax.switch(
            sprite_idx,
            [
                lambda: sprites[0],
                lambda: sprites[1],
                lambda: sprites[2],
            ],
        )
        return jax.lax.cond(
            looks_right,
            lambda: jnp.fliplr(mask),
            lambda: mask,
        )

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: RoadRunnerState) -> jnp.ndarray:
        canvas = self.jr.create_object_raster(self.BACKGROUND)

        # --- Animate Road ---
        PATTERN_WIDTH = self.consts.ROAD_PATTERN_WIDTH

        # Calculate the horizontal offset for scrolling
        offset = PATTERN_WIDTH - (
            (state.scrolling_step_counter * self.consts.PLAYER_MOVE_SPEED)
            % PATTERN_WIDTH
        )

        # Slice the wide road mask to get the current frame's view
        road_mask = jax.lax.dynamic_slice(
            self.SHAPE_MASKS["road"],
            (0, offset),
            (self.consts.ROAD_HEIGHT, self.consts.WIDTH),
        )
        section = self._get_render_section(state)
        desired_width = jnp.clip(section.road_width, 1, self.consts.WIDTH)
        left_padding = jnp.clip(
            (self.consts.WIDTH - desired_width) // 2, 0, self.consts.WIDTH
        )
        right_boundary = jnp.clip(left_padding + desired_width, 0, self.consts.WIDTH)
        x_coords = jnp.arange(self.consts.WIDTH)
        column_mask = (x_coords >= left_padding) & (x_coords < right_boundary)
        mask_height, mask_width = road_mask.shape
        column_mask = column_mask[jnp.newaxis, :]
        column_mask = jnp.broadcast_to(column_mask, (mask_height, mask_width))
        background_value = jnp.array(
            self.COLOR_TO_ID.get(
                self.consts.BACKGROUND_COLOR, 0
            ),
            dtype=road_mask.dtype,
        )
        road_mask = jnp.where(column_mask, road_mask, background_value)

        desired_height = jnp.clip(section.road_height, 1, self.consts.ROAD_HEIGHT)
        top_within_sprite = jnp.clip(
            section.road_top, 0, self.consts.ROAD_HEIGHT - desired_height
        )
        row_indices = jnp.arange(self.consts.ROAD_HEIGHT)
        row_mask = (row_indices >= top_within_sprite) & (
            row_indices < top_within_sprite + desired_height
        )
        row_mask = row_mask[:, jnp.newaxis]
        row_mask = jnp.broadcast_to(row_mask, (mask_height, mask_width))
        road_mask = jnp.where(row_mask, road_mask, background_value)

        # Render the sliced road portion
        canvas = self.jr.render_at(canvas, 0, self.consts.ROAD_TOP_Y, road_mask)

        # Render Ravines
        canvas = self._render_ravines(canvas, state.ravines)

        # Render Seeds
        canvas = self._render_seeds(canvas, state.seeds)
        
        # Render Landmine
        canvas = self._render_landmine(canvas, state.landmine_x, state.landmine_y)

        # Render score
        canvas = self._render_score(canvas, state.score)

        # Render Lives
        canvas = self._render_lives(canvas, state.lives)

        canvas = self._render_decorations(canvas, state)

        # Render Player
        def _render_burnt_player(c):
             return self.jr.render_at(c, state.player_x, state.player_y, self.SHAPE_MASKS["player_burnt"])

        def _render_normal_player(c):
            player_mask = self._get_animated_sprite(
                state.player_is_moving,
                state.player_looks_right,
                state.step_counter,
                self.consts.PLAYER_ANIMATION_SPEED,
                self.SHAPE_MASKS["player"],
                self.SHAPE_MASKS["player_run1"],
                self.SHAPE_MASKS["player_run2"],
            )
            return self.jr.render_at(c, state.player_x, state.player_y, player_mask)

        def _render_jumping_player(c):
            jump_mask = self.SHAPE_MASKS["player_jump"]
            # Flip jump sprite if player looks right
            jump_mask = jax.lax.cond(
                state.player_looks_right,
                lambda: jnp.fliplr(jump_mask),
                lambda: jump_mask,
            )
            return self.jr.render_at(c, state.player_x, state.player_y, jump_mask)

        # Switch between burnt, jumping, Normal
        # Priority: Burnt (Death) > Jump > Normal
        
        def _render_alive_player(c):
            return jax.lax.cond(
                state.is_jumping,
                _render_jumping_player,
                _render_normal_player,
                c,
            )

        canvas = jax.lax.cond(
            state.death_timer > 0,
            _render_burnt_player,
            _render_alive_player,
            canvas,
        )

        # Render Enemy
        def _render_enemy(c):
            enemy_mask = self._get_animated_sprite(
                state.enemy_is_moving,
                state.enemy_looks_right,
                state.step_counter,
                self.consts.PLAYER_ANIMATION_SPEED,  # Assuming same speed for now
                self.SHAPE_MASKS["enemy"],
                self.SHAPE_MASKS["enemy_run1"],
                self.SHAPE_MASKS["enemy_run2"],
            )
            return self.jr.render_at(c, state.enemy_x, state.enemy_y, enemy_mask)

        # Render the enemy only if it is on screen
        # else return the canvas unchanged
        canvas = jax.lax.cond(
            state.enemy_x < self.consts.WIDTH,
            _render_enemy,
            lambda c: c,
            canvas,
        )

        def _render_transition(_):
            t_canvas = self.jr.create_object_raster(self.BACKGROUND)

            t_canvas = self.jr.render_at(
                t_canvas,
                0,
                0,
                self.SHAPE_MASKS["end_of_level_1"]
            )

            return self.jr.render_from_palette(t_canvas, self.PALETTE)

        # Render Truck
        canvas = self._render_truck(canvas, state.truck_x, state.truck_y)

        final_frame = self.jr.render_from_palette(canvas, self.PALETTE)

        # --- Mask Side Margins ---
        # Force pixels in the side margins to be black
        margin = self.consts.SIDE_MARGIN
        width = self.consts.WIDTH

        # Create a mask for valid gameplay area (True for valid, False for margin)
        col_indices = jnp.arange(width)
        valid_cols = (col_indices >= margin) & (col_indices < (width - margin))
        # Broadcast to full image shape (H, W, 3)
        margin_mask = jnp.broadcast_to(valid_cols[None, :, None], final_frame.shape)

        # Use black for the margins
        final_frame = jnp.where(margin_mask, final_frame, jnp.zeros_like(final_frame))

        return jax.lax.cond(
            state.is_in_transition,
            _render_transition,
            lambda _: final_frame,
            operand=None,
        )
