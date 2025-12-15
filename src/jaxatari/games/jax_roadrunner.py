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
    seed_spawn_config: Optional[Tuple[int, int]] = None
    truck_spawn_config: Optional[Tuple[int, int]] = None
    future_entity_types: Dict[str, Any] = {}


# --- Constants ---
class RoadRunnerConstants(NamedTuple):
    WIDTH: int = 160
    HEIGHT: int = 210
    PLAYER_MOVE_SPEED: int = 4
    PLAYER_ANIMATION_SPEED: int = 2
    # If the players x coordinate would be below this value after applying movement, we move everything one to the right to simulate movement.
    X_SCROLL_THRESHOLD: int = 50
    ENEMY_MOVE_SPEED: int = 3
    ENEMY_REACTION_DELAY: int = 6
    PLAYER_START_X: int = 140
    PLAYER_START_Y: int = 96
    ENEMY_X: int = 16
    ENEMY_Y: int = 96
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
    TRUCK_SPEED: int = 5
    TRUCK_SPAWN_MIN_INTERVAL: int = 30
    TRUCK_SPAWN_MAX_INTERVAL: int = 80
    LEVEL_TRANSITION_DURATION: int = 30
    LEVEL_COMPLETE_SCROLL_DISTANCE: int = 100
    JUMP_TIME_DURATION: int = 20  # Jump duration in steps (~0.33 seconds at 60 FPS)
    SIDE_MARGIN: int = 8
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
)

RoadRunner_Level_2 = LevelConfig(
    level_number=2,
    scroll_distance_to_complete=_BASE_CONSTS.LEVEL_COMPLETE_SCROLL_DISTANCE,
    road_sections=(
        RoadSectionConfig(
            scroll_start=0,
            scroll_end=300,
            road_width=_BASE_CONSTS.WIDTH - 2 * _BASE_CONSTS.SIDE_MARGIN,
            road_top=_centered_top(70),
            road_height=70,
        ),
        RoadSectionConfig(
            scroll_start=300,
            scroll_end=700,
            road_width=_BASE_CONSTS.WIDTH - 2 * _BASE_CONSTS.SIDE_MARGIN,
            road_top=_centered_top(50),
            road_height=50,
        ),
        RoadSectionConfig(
            scroll_start=700,
            scroll_end=_BASE_CONSTS.LEVEL_COMPLETE_SCROLL_DISTANCE,
            road_width=_BASE_CONSTS.WIDTH - 2 * _BASE_CONSTS.SIDE_MARGIN,
            road_top=_centered_top(60),
            road_height=60,
        ),
    ),
    spawn_seeds=False,
    spawn_trucks=True,
    truck_spawn_config=(
        _BASE_CONSTS.TRUCK_SPAWN_MIN_INTERVAL,
        _BASE_CONSTS.TRUCK_SPAWN_MAX_INTERVAL,
    ),
)

DEFAULT_LEVELS: Tuple[LevelConfig, ...] = (
    RoadRunner_Level_1,
    RoadRunner_Level_2,
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
    jump_timer: chex.Array  # Countdown timer for jump (0 when not jumping)
    is_jumping: chex.Array  # Boolean flag indicating if player is currently jumping

class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray


class RoadRunnerObservation(NamedTuple):
    player: EntityPosition
    enemy: EntityPosition
    score: jnp.ndarray


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
        if self._level_count > 0:
            self._level_spawn_seeds = jnp.array(
                [cfg.spawn_seeds for cfg in self.consts.levels], dtype=jnp.bool_
            )
            self._seed_spawn_intervals = jnp.array(
                [
                    [
                        (cfg.seed_spawn_config or (
                            self.consts.SEED_SPAWN_MIN_INTERVAL,
                            self.consts.SEED_SPAWN_MAX_INTERVAL,
                        ))[0],
                        (cfg.seed_spawn_config or (
                            self.consts.SEED_SPAWN_MIN_INTERVAL,
                            self.consts.SEED_SPAWN_MAX_INTERVAL,
                        ))[1],
                    ]
                    for cfg in self.consts.levels
                ],
                dtype=jnp.int32,
            )
            self._level_spawn_trucks = jnp.array(
                [cfg.spawn_trucks for cfg in self.consts.levels], dtype=jnp.bool_
            )
            self._truck_spawn_intervals = jnp.array(
                [
                    [
                        (cfg.truck_spawn_config or (
                            self.consts.TRUCK_SPAWN_MIN_INTERVAL,
                            self.consts.TRUCK_SPAWN_MAX_INTERVAL,
                        ))[0],
                        (cfg.truck_spawn_config or (
                            self.consts.TRUCK_SPAWN_MIN_INTERVAL,
                            self.consts.TRUCK_SPAWN_MAX_INTERVAL,
                        ))[1],
                    ]
                    for cfg in self.consts.levels
                ],
                dtype=jnp.int32,
            )
            self._max_road_sections = max(
                len(cfg.road_sections) for cfg in self.consts.levels
            )
            road_sections_data = []
            road_section_counts = []
            default_section = [
                0,
                self.consts.LEVEL_COMPLETE_SCROLL_DISTANCE,
                self.consts.WIDTH,
                0,
                0,
                self.consts.ROAD_HEIGHT,
            ]
            for cfg in self.consts.levels:
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
                while len(rows) < self._max_road_sections:
                    rows.append(rows[-1][:])
                road_sections_data.append(rows)
            self._road_section_data = jnp.array(road_sections_data, dtype=jnp.int32)
            self._road_section_counts = jnp.array(
                road_section_counts, dtype=jnp.int32
            )
        else:
            self._level_spawn_seeds = jnp.array([], dtype=jnp.bool_)
            self._seed_spawn_intervals = jnp.array([], dtype=jnp.int32).reshape(0, 2)
            self._level_spawn_trucks = jnp.array([], dtype=jnp.bool_)
            self._truck_spawn_intervals = jnp.array([], dtype=jnp.int32).reshape(0, 2)
            self._max_road_sections = 0
            self._road_section_data = jnp.array([], dtype=jnp.int32).reshape(0, 0, 6)
            self._road_section_counts = jnp.array([], dtype=jnp.int32)

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
        min_y = road_top - (self.consts.PLAYER_SIZE[1] // 3)
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
        player_looks_right = jax.lax.cond(
            vel_x > 0,
            lambda: True,
            lambda: jax.lax.cond(
                vel_x < 0, lambda: False, lambda: state.player_looks_right
            ),
        )

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
            # Get the distance to the player, with a configurable frame delay.
            delayed_player_x = st.player_x_history[
                self.consts.ENEMY_REACTION_DELAY - 1
            ]
            delayed_player_y = st.player_y_history[
                self.consts.ENEMY_REACTION_DELAY - 1
            ]
            delta_x = delayed_player_x - st.enemy_x
            delta_y = delayed_player_y - st.enemy_y

            # Determine enemy movement and orientation
            enemy_is_moving = (delta_x != 0) | (delta_y != 0)
            enemy_looks_right = jax.lax.cond(
                delta_x > 0,
                lambda: True,
                lambda: jax.lax.cond(
                    delta_x < 0, lambda: False, lambda: st.enemy_looks_right
                ),
            )

            # Update enemy position, clipping movement to ENEMY_MOVE_SPEED to prevent jittering
            new_enemy_x = st.enemy_x + jnp.clip(
                delta_x, -self.consts.ENEMY_MOVE_SPEED, self.consts.ENEMY_MOVE_SPEED
            )
            new_enemy_y = st.enemy_y + jnp.clip(
                delta_y, -self.consts.ENEMY_MOVE_SPEED, self.consts.ENEMY_MOVE_SPEED
            )

            new_enemy_x = self._handle_scrolling(st, new_enemy_x)

            new_enemy_x, new_enemy_y = self._check_enemy_bounds(st, new_enemy_x, new_enemy_y)
            return st._replace(
                enemy_x=new_enemy_x,
                enemy_y=new_enemy_y,
                enemy_is_moving=enemy_is_moving,
                enemy_looks_right=enemy_looks_right,
            )

        return jax.lax.cond(state.is_round_over, game_over_logic, normal_logic, state)

    def _check_game_over(self, state: RoadRunnerState) -> RoadRunnerState:
        # Here we check if the enemy and the player overlap
        player_x2 = state.player_x + self.consts.PLAYER_SIZE[0]
        player_y2 = state.player_y + self.consts.PLAYER_SIZE[1]
        enemy_x2 = state.enemy_x + self.consts.ENEMY_SIZE[0]
        enemy_y2 = state.enemy_y + self.consts.ENEMY_SIZE[1]

        # Check for overlap on both axes
        overlap_x = (state.player_x < enemy_x2) & (player_x2 > state.enemy_x)
        overlap_y = (state.player_y < enemy_y2) & (player_y2 > state.enemy_y)

        # Collision happens if there is overlap on both axes
        collision = overlap_x & overlap_y

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
        # Calculate player pickup area bounding box
        player_left_x = state.player_x
        player_right_x = state.player_x + self.consts.PLAYER_SIZE[0]
        # Pickup area starts at PLAYER_PICKUP_OFFSET from top of player
        player_pickup_y = state.player_y + self.consts.PLAYER_PICKUP_OFFSET
        player_bottom_y = state.player_y + self.consts.PLAYER_SIZE[1]

        def check_and_pickup_seed(i: int, st: RoadRunnerState) -> RoadRunnerState:
            """Check collision for seed at index i and pick it up if colliding."""
            seed_x = st.seeds[i, 0]
            seed_y = st.seeds[i, 1]

            # Early return if seed is inactive
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

            # Calculate seed bounding box
            seed_left_x = seed_x
            seed_right_x = seed_x + self.consts.SEED_SIZE[0]
            seed_top_y = seed_y
            seed_bottom_y = seed_y + self.consts.SEED_SIZE[1]

            # Check for overlap on both axes (AABB collision)
            overlap_x = (player_left_x < seed_right_x) & (player_right_x > seed_left_x)
            overlap_y = (player_pickup_y < seed_bottom_y) & (player_bottom_y > seed_top_y)
            collision = overlap_x & overlap_y

            # Pick up seed if collision detected
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
        # Calculate truck collision area (only lower half, using TRUCK_COLLISION_OFFSET)
        truck_left_x = state.truck_x
        truck_right_x = state.truck_x + self.consts.TRUCK_SIZE[0]
        # Collision area starts at TRUCK_COLLISION_OFFSET from top of truck (lower half)
        truck_collision_y = state.truck_y + self.consts.TRUCK_COLLISION_OFFSET
        truck_bottom_y = state.truck_y + self.consts.TRUCK_SIZE[1]

        # Calculate player pickup area bounding box (same as seeds)
        player_left_x = state.player_x
        player_right_x = state.player_x + self.consts.PLAYER_SIZE[0]
        # Pickup area starts at PLAYER_PICKUP_OFFSET from top of player
        player_pickup_y = state.player_y + self.consts.PLAYER_PICKUP_OFFSET
        player_bottom_y = state.player_y + self.consts.PLAYER_SIZE[1]

        # Calculate enemy bounding box
        enemy_left_x = state.enemy_x
        enemy_right_x = state.enemy_x + self.consts.ENEMY_SIZE[0]
        enemy_top_y = state.enemy_y
        enemy_bottom_y = state.enemy_y + self.consts.ENEMY_SIZE[1]

        # Check player-truck collision (player uses pickup area, truck uses collision area)
        player_overlap_x = (player_left_x < truck_right_x) & (player_right_x > truck_left_x)
        player_overlap_y = (player_pickup_y < truck_bottom_y) & (player_bottom_y > truck_collision_y)
        player_collision = player_overlap_x & player_overlap_y

        # Check enemy-truck collision (truck uses collision area)
        enemy_overlap_x = (enemy_left_x < truck_right_x) & (enemy_right_x > truck_left_x)
        enemy_overlap_y = (enemy_top_y < truck_bottom_y) & (enemy_bottom_y > truck_collision_y)
        enemy_collision = enemy_overlap_x & enemy_overlap_y

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
            jump_timer=jnp.array(0, dtype=jnp.int32),
            is_jumping=jnp.array(False, dtype=jnp.bool_),
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
            st = self._check_level_completion(st)

            def reset_round(inner_state: RoadRunnerState) -> RoadRunnerState:
                reset_state = inner_state._replace(
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
                    jump_timer=jnp.array(0, dtype=jnp.int32),
                    is_jumping=jnp.array(False, dtype=jnp.bool_),
                )
                level_idx = self._get_level_index(reset_state)
                return self._initialize_spawn_timers(reset_state, level_idx)

            player_at_end = st.player_x >= self.consts.WIDTH - self.consts.PLAYER_SIZE[0]
            st = jax.lax.cond(
                st.is_round_over & player_at_end, reset_round, lambda inner: inner, st
            )

            st = st._replace(step_counter=st.step_counter + 1)
            obs = self._get_observation(st)
            return obs, st, 0.0, False, None

        observation, state, reward, done, info = jax.lax.cond(
            state.is_in_transition, _transition_branch, _gameplay_branch, operand
        )

        return observation, state, reward, done, info

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
        else:
            seed_bounds = jnp.array(
                [self.consts.SEED_SPAWN_MIN_INTERVAL, self.consts.SEED_SPAWN_MAX_INTERVAL],
                dtype=jnp.int32,
            )
            truck_bounds = jnp.array(
                [self.consts.TRUCK_SPAWN_MIN_INTERVAL, self.consts.TRUCK_SPAWN_MAX_INTERVAL],
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
        return state._replace(
            rng=rng,
            next_seed_spawn_scroll_step=next_seed_spawn_scroll_step,
            next_truck_spawn_step=next_truck_spawn_step,
        )

    def _get_current_road_section(self, state: RoadRunnerState) -> RoadSectionConfig:
        if self._level_count == 0 or self._max_road_sections == 0:
            return RoadSectionConfig(
                0,
                self.consts.LEVEL_COMPLETE_SCROLL_DISTANCE,
                self.consts.WIDTH,
                0,
                self.consts.ROAD_HEIGHT,
                0,
            )
        level_idx = self._get_level_index(state)
        sections = self._road_section_data[level_idx]
        section_count = self._road_section_counts[level_idx]
        indices = jnp.arange(self._max_road_sections, dtype=jnp.int32)
        valid_section = indices < section_count
        counter = state.scrolling_step_counter
        in_section = (counter >= sections[:, 0]) & (counter < sections[:, 1])
        active_sections = jnp.where(valid_section, in_section, False)
        no_match_value = jnp.array(self._max_road_sections, dtype=jnp.int32)
        candidate_indices = jnp.where(
            active_sections, indices, no_match_value
        )
        match_idx = jnp.min(candidate_indices)
        fallback_idx = jnp.maximum(section_count - 1, 0)
        section_idx = jnp.where(match_idx < self._max_road_sections, match_idx, fallback_idx)
        selected = sections[section_idx]
        return RoadSectionConfig(
            selected[0],
            selected[1],
            selected[2],
            selected[4],
            selected[5],
            selected[3],
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
        return RoadRunnerObservation(player=player, enemy=enemy, score=state.score)

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
        self.config = render_utils.RendererConfig(
            game_dimensions=(self.consts.HEIGHT, self.consts.WIDTH),
            channels=3,
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)

        wall_sprite_top = self._create_wall_sprite(self.consts.WALL_TOP_HEIGHT)
        wall_sprite_bottom = self._create_wall_sprite(self.consts.WALL_BOTTOM_HEIGHT)
        road_sprite = self._create_road_sprite()
        truck_sprite = self._create_truck_sprite()
        asset_config = self._get_asset_config(
            road_sprite, wall_sprite_bottom, truck_sprite
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
        if self._level_count > 0:
            self._max_road_sections = max(
                len(cfg.road_sections) for cfg in self.consts.levels
            )
            road_sections_data = []
            road_section_counts = []
            default_section = [
                0,
                self.consts.LEVEL_COMPLETE_SCROLL_DISTANCE,
                self.consts.WIDTH,
                0,
                0,
                self.consts.ROAD_HEIGHT,
            ]
            for cfg in self.consts.levels:
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
                while len(rows) < self._max_road_sections:
                    rows.append(rows[-1][:])
                road_sections_data.append(rows)
            self._road_section_data = jnp.array(road_sections_data, dtype=jnp.int32)
            self._road_section_counts = jnp.array(
                road_section_counts, dtype=jnp.int32
            )
        else:
            self._max_road_sections = 0
            self._road_section_data = jnp.array([], dtype=jnp.int32).reshape(0, 0, 6)
            self._road_section_counts = jnp.array([], dtype=jnp.int32)

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

    def _create_truck_sprite(self) -> jnp.ndarray:
        truck_color_rgba = (*self.consts.TRUCK_COLOR, 255)
        truck_shape = (self.consts.TRUCK_SIZE[0], self.consts.TRUCK_SIZE[1], 4)
        return jnp.tile(
            jnp.array(truck_color_rgba, dtype=jnp.uint8), (*truck_shape[:2], 1)
        )

    def _get_render_section(self, state: RoadRunnerState) -> RoadSectionConfig:
        if self._level_count == 0 or self._max_road_sections == 0:
            return RoadSectionConfig(
                0,
                self.consts.LEVEL_COMPLETE_SCROLL_DISTANCE,
                self.consts.WIDTH,
                0,
                self.consts.ROAD_HEIGHT,
                0,
            )
        max_index = self._level_count - 1
        level_idx = jnp.clip(state.current_level, 0, max_index).astype(jnp.int32)
        sections = self._road_section_data[level_idx]
        section_count = self._road_section_counts[level_idx]
        indices = jnp.arange(self._max_road_sections, dtype=jnp.int32)
        valid_section = indices < section_count
        counter = state.scrolling_step_counter
        in_section = (counter >= sections[:, 0]) & (counter < sections[:, 1])
        active_sections = jnp.where(valid_section, in_section, False)
        no_match_value = jnp.array(self._max_road_sections, dtype=jnp.int32)
        candidate_indices = jnp.where(active_sections, indices, no_match_value)
        match_idx = jnp.min(candidate_indices)
        fallback_idx = jnp.maximum(section_count - 1, 0)
        section_idx = jnp.where(
            match_idx < self._max_road_sections, match_idx, fallback_idx
        )
        selected = sections[section_idx]
        return RoadSectionConfig(
            selected[0],
            selected[1],
            selected[2],
            selected[4],
            selected[5],
            selected[3],
        )

    def _get_asset_config(
        self,
        road_sprite: jnp.ndarray,
        wall_sprite_bottom: jnp.ndarray,
        truck_sprite: jnp.ndarray,
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
            {"name": "seed", "type": "procedural", "data": self._create_seed_sprite()},
            {"name": "truck", "type": "procedural", "data": truck_sprite},
        ]

        return asset_config

    def _render_score(self, canvas: jnp.ndarray, score: jnp.ndarray) -> jnp.ndarray:
        score_digits = self.jr.int_to_digits(score, max_digits=6)
        score_digit_masks = self.SHAPE_MASKS["score_digits"]

        # Position the score at the top center
        score_x = (
            self.consts.WIDTH // 2 - (score_digits.shape[0] * 6) // 2
        )  # Assuming digit width of 6
        score_y = 4 # 2 for the black border, 2 for spacing

        canvas = self.jr.render_label_selective(
            canvas,
            score_x,
            score_y,
            score_digits,
            score_digit_masks,
            0,
            score_digits.shape[0],
            spacing=8, # offset, so width of digit + space
            max_digits_to_render=6,
        )
        return canvas

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

        # Render score
        canvas = self._render_score(canvas, state.score)

        # Render Player
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

        canvas = jax.lax.cond(
            state.is_jumping,
            _render_jumping_player,
            _render_normal_player,
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

        # Render Seeds
        canvas = self._render_seeds(canvas, state.seeds)

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

        transition_frame = jnp.zeros_like(final_frame)
        return jax.lax.cond(
            state.is_in_transition,
            lambda _: transition_frame,
            lambda _: final_frame,
            operand=None,
        )
