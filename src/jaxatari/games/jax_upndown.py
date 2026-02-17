import os
from functools import partial
from typing import NamedTuple, Tuple

import jax
import jax.lax
import jax.numpy as jnp
import chex

import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action

class UpNDownConstants(NamedTuple):
    FRAME_SKIP: int = 4
    DIFFICULTIES: chex.Array = jnp.array([0, 1, 2, 3, 4, 5])
    MAX_SPEED: int = 7
    INITIAL_LIVES: int = 5
    JUMP_ARC_HEIGHT: float = 22.0
    RESPAWN_DELAY_FRAMES: int = 60
    RESPAWN_Y: int = 0
    RESPAWN_X: int = 30
    ALL_FLAGS_BONUS: int = 1000
    # Enemy spawning and movement
    MAX_ENEMY_CARS: int = 8
    ENEMY_SPAWN_INTERVAL_BASE: int = 30  # Base spawn interval
    ENEMY_SPAWN_INTERVAL_MAX: int = 60  # Max spawn interval when many enemies exist
    ENEMY_MIN_VISIBLE_COUNT: int = 2  # Minimum enemies to keep on screen
    ENEMY_VISIBLE_DISTANCE: int = 120  # Distance within which enemies are considered "visible"
    ENEMY_DESPAWN_DISTANCE: int = 250
    ENEMY_SPEED_MIN: int = 3
    ENEMY_SPEED_MAX: int = 5
    ENEMY_DIRECTION_SWITCH_PROB: float = 0.0001
    ENEMY_SPAWN_OFFSET_MIN: float = 70.0  # Closer spawn distance
    ENEMY_SPAWN_OFFSET_MAX: float = 130.0  # Max spawn offset
    ENEMY_MIN_SPAWN_GAP: float = 25.0  # Reduced gap between spawns
    ENEMY_MAX_AGE: int = 1900
    INITIAL_ENEMY_COUNT: int = 4
    INITIAL_ENEMY_BASE_OFFSET: float = 35.0  # Closer initial enemies
    INITIAL_ENEMY_GAP: float = 25.0  # Tighter initial spacing
    ENEMY_TYPE_CAMERO: int = 0
    ENEMY_TYPE_FLAG_CARRIER: int = 1
    ENEMY_TYPE_PICKUP: int = 2
    ENEMY_TYPE_TRUCK: int = 3
    JUMP_FRAMES: int = 28
    POST_JUMP_DELAY: int = 10
    LANDING_TOLERANCE: int = 20  # Pixels tolerance for landing on a road (increased by 5 for wider landing zone)
    LATE_JUMP_COLLISION_FRAMES: int = 2
    LANDING_COLLISION_DISTANCE: float = 12.0  # Larger collision distance when landing (increased for easier enemy kills)
    GROUND_COLLISION_DISTANCE: float = 3.0  # Tight collision distance for ground collisions
    LATE_JUMP_ENEMY_SCORE: int = 400
    STEEP_ROAD_SPEED_REDUCTION_INTERVAL: int = 8  # Frames between each speed reduction on steep roads
    PASSIVE_SCORE_INTERVAL: int = 60  # Steps between passive score awards
    PASSIVE_SCORE_AMOUNT: int = 10  # Points awarded for passive scoring
    COLLISION_THRESHOLD: float = 5.0  # Distance threshold for flag/collectible collision
    ACCELERATION_INTERVAL: int = 6  # Frames between speed changes when holding up/down
    EXTRA_LIFE_THRESHOLD: int = 10000  # Score threshold for extra life
    TRACK_LENGTH: int = 1036
    FIRST_TRACK_CORNERS_X: chex.Array = jnp.array([30, 75, 128, 75, 21, 75, 131, 111, 150, 95, 150, 115, 150, 108, 150, 115, 115, 75, 18, 38, 67, 38, 38, 20, 64, 30])
    TRACK_CORNERS_Y: chex.Array = jnp.array([0, -40, -98, -155, -203, -268, -327, -347, -382, -467, -525, -565, -597, -625, -670, -705, -738, -788, -838, -862, -898, -925, -950, -972, -1000, -1035])
    SECOND_TRACK_CORNERS_X: chex.Array = jnp.array([115, 75, 20, 75, 133, 75, 22, 37, 63, 27, 66, 30, 63, 24, 60, 38, 38, 75, 131, 111, 150, 118, 118, 98, 150, 115])
    PLAYER_SIZE: Tuple[int, int] = (4, 16)
    INITIAL_ROAD_POS_Y: int = 25
    # Flag constants - 8 flags with different colors matching the top row
    NUM_FLAGS: int = 8
    # Flag colors as RGBA values (matching the top row from left to right)
    FLAG_COLORS: chex.Array = jnp.array([
        [184, 50, 50, 255],    # Red
        [181, 83, 40, 255],    # Orange
        [162, 98, 33, 255],    # Dark orange
        [134, 134, 29, 255],   # Yellow/olive
        [200, 72, 72, 255],    # Pink (original)
        [168, 48, 143, 255],   # Magenta
        [125, 48, 173, 255],   # Purple
        [78, 50, 181, 255],    # Blue
    ])
    # Top display positions for each flag (x coordinates where blackout squares appear)
    FLAG_TOP_X_POSITIONS: chex.Array = jnp.array([13, 30, 47, 64, 82, 98, 118, 134])
    FLAG_TOP_Y: int = 20
    FLAG_BLACKOUT_SIZE: Tuple[int, int] = (14, 14)  # Size of blackout square
    FLAG_COLLECTION_SCORE: int = 75  # Points awarded for collecting a flag
    # Life display constants - positions of life cars at the bottom
    LIFE_BOTTOM_X_POSITIONS: chex.Array = jnp.array([13, 18, 25, 33, 33])  # X positions for 5 life cars
    LIFE_BOTTOM_Y: int = 195
    # Collectible constants - unified dynamic spawning
    MAX_COLLECTIBLES: int = 1  # Maximum collectibles that can exist at once (pool of mixed types)
    COLLECTIBLE_SPAWN_INTERVAL: int = 200  # Steps between spawn attempts
    COLLECTIBLE_DESPAWN_DISTANCE: int = 500  # Distance beyond which collectibles despawn
    # Collectible types (indices for type field)
    COLLECTIBLE_TYPE_CHERRY: int = 0
    COLLECTIBLE_TYPE_BALLOON: int = 1
    COLLECTIBLE_TYPE_LOLLYPOP: int = 2
    COLLECTIBLE_TYPE_ICE_CREAM: int = 3
    # Collectible type spawn probabilities (cumulative thresholds for random sampling)
    COLLECTIBLE_SPAWN_PROBABILITIES: chex.Array = jnp.array([35, 65, 90, 100], dtype=jnp.int32)  # Cherry: 35%, Balloon: 30%, Lollypop: 25%, IceCream: 10%
    # Collectible type scores
    COLLECTIBLE_SCORES: chex.Array = jnp.array([50, 65, 70, 75], dtype=jnp.int32)  # [cherry, balloon, lollypop, ice_cream]
    # Shared collectible colors
    COLLECTIBLE_COLORS: chex.Array = FLAG_COLORS



# immutable state container
class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray

class Car(NamedTuple):
    position: EntityPosition
    speed: chex.Array
    type: chex.Array
    current_road: chex.Array
    road_index_A: chex.Array
    road_index_B: chex.Array
    direction_x: chex.Array

class Flag(NamedTuple):
    """Represents a collectible flag on the road."""
    y: chex.Array  # Y position in world coordinates (like player_car.position.y)
    road: chex.Array  # Which road the flag is on (0 or 1)
    road_segment: chex.Array  # Which road segment index the flag is on
    color_idx: chex.Array  # Index into FLAG_COLORS array
    collected: chex.Array  # Whether this flag has been collected

class Collectible(NamedTuple):
    """Represents a dynamically spawning collectible item on the road.
    
    Can be any type: cherry (0), balloon (1), lollypop (2), or ice cream (3).
    The type determines the sprite and point value.
    """
    y: chex.Array  # Y position in world coordinates
    x: chex.Array  # X position on the road
    road: chex.Array  # Which road the collectible is on (0 or 1)
    color_idx: chex.Array  # Index into COLLECTIBLE_COLORS array
    type_id: chex.Array  # Type of collectible (0=cherry, 1=balloon, 2=lollypop, 3=ice_cream)
    active: chex.Array  # Whether this collectible slot is active (spawned)


class EnemyCars(NamedTuple):
    """Pool of enemy cars that share the same road-following logic as the player."""
    position: EntityPosition  # vectorized position fields, size MAX_ENEMY_CARS
    speed: chex.Array  # signed speed per car
    type: chex.Array  # type id per car
    current_road: chex.Array
    road_index_A: chex.Array
    road_index_B: chex.Array
    direction_x: chex.Array
    active: chex.Array
    age: chex.Array

class UpNDownState(NamedTuple):
    score: chex.Array
    difficulty: chex.Array
    jump_cooldown: chex.Array
    post_jump_cooldown: chex.Array
    is_jumping: chex.Array
    is_on_road: chex.Array
    player_car: Car
    lives: chex.Array
    is_dead: chex.Array
    respawn_timer: chex.Array
    step_counter: chex.Array
    rng_key: chex.PRNGKey
    round_started: chex.Array
    movement_steps: chex.Array
    steep_road_timer: chex.Array  # Timer for steep road speed reduction
    jump_slope: chex.Array  # X movement per Y step, locked at jump start (float)
    # Flag state - tracks all 8 flags
    flags: Flag  # Contains arrays of size NUM_FLAGS for each field
    flags_collected_mask: chex.Array  # Boolean mask of which flag colors have been collected (size NUM_FLAGS)
    # Collectible state - dynamic spawning (mixed types: cherry, balloon, lollypop, ice cream)
    collectibles: Collectible  # Contains arrays of size MAX_COLLECTIBLES for each field
    collectible_spawn_timer: chex.Array  # Counter for collectible spawn timing
    # Enemy cars - dynamic spawning and movement
    enemy_cars: EnemyCars
    enemy_spawn_timer: chex.Array
    # Death/respawn state - player is dead and waiting for input to respawn
    awaiting_respawn: chex.Array  # True when player died and is waiting for input
    # Round start state - everything frozen and hidden until player presses input
    awaiting_round_start: chex.Array  # True at game start and after respawn until input received
    # Input debounce - requires button release before next input triggers round start
    input_released: chex.Array  # True when player has released all buttons since last state change
    jump_key_released: chex.Array  # True if jump button was NOT pressed in previous step
    last_extra_life_score: chex.Array  # Score at which last extra life was awarded
    jump_total_duration: chex.Array  # Total duration of the current/last jump for rendering arc



class UpNDownObservation(NamedTuple):
    player_car: Car
    enemy_cars: EnemyCars
    flags: Flag
    collectibles: Collectible
    flags_collected_mask: chex.Array  # Shape (NUM_FLAGS,) - int32 (0 or 1)
    player_score: chex.Array
    lives: chex.Array
    is_jumping: chex.Array
    jump_cooldown: chex.Array
    is_on_steep_road: chex.Array
    round_started: chex.Array


class UpNDownInfo(NamedTuple):
    """Additional info for debugging and analysis."""
    step_counter: jnp.ndarray  # Total steps taken
    difficulty: jnp.ndarray  # Current difficulty level
    movement_steps: jnp.ndarray  # Steps since round started
    jump_slope: jnp.ndarray  # Current jump trajectory slope
    player_road_segment: jnp.ndarray  # Current road segment index
class JaxUpNDown(JaxEnvironment[UpNDownState, UpNDownObservation, UpNDownInfo, UpNDownConstants]):
    def __init__(self, consts: UpNDownConstants = None, reward_funcs: list[callable]=None):
        consts = consts or UpNDownConstants()
        super().__init__(consts)
        self.renderer = UpNDownRenderer(self.consts)
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.action_set = [
            Action.NOOP,
            Action.FIRE,
            Action.UPFIRE,
            Action.UP,
            Action.DOWN,
            Action.DOWNFIRE,
        ]
        # Calculate obs_size based on observation structure:
        # Player car: 10 values (x, y, w, h, speed, type, road, road_index_A, road_index_B, direction_x)
        # Enemy cars: MAX_ENEMY_CARS * 12 = 8 * 12 = 96 (x, y, w, h, speed, type, road, road_index_A, road_index_B, direction_x, active, age)
        # Flags: NUM_FLAGS * 5 = 8 * 5 = 40 (y, road, segment, color, collected per flag)
        # Collectibles: MAX_COLLECTIBLES * 6 = 1 * 6 = 6 (y, x, road, color_idx, type, active per collectible)
        # Flags collected mask: NUM_FLAGS = 8
        # Score, lives, is_jumping, jump_cooldown, is_on_steep_road, round_started: 6
        # Total: 10 + 96 + 40 + 6 + 8 + 6 = 166
        self.obs_size = (
            10 +  # player car
            self.consts.MAX_ENEMY_CARS * 12 +  # enemy cars (all fields)
            self.consts.NUM_FLAGS * 5 +  # flags
            self.consts.MAX_COLLECTIBLES * 6 +  # collectibles (all fields)
            self.consts.NUM_FLAGS +  # flags_collected_mask
            6  # score, lives, is_jumping, jump_cooldown, is_on_steep_road, round_started
        )
        # Speed dividers for movement timing (indexed by speed level)
        self._speed_dividers = jnp.array([0, 1, 2, 4, 8, 16, 16, 16, 16])

    @partial(jax.jit, static_argnums=(0,))
    def _compute_movement_timing(self, speed: chex.Array, step_counter: chex.Array) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        """Calculate movement timing parameters based on speed.
        
        Returns:
            Tuple of (move_y, move_x, step_size, speed_sign)
        """
        abs_speed = jnp.abs(speed)
        speed_index = jnp.minimum(abs_speed, jnp.int32(self._speed_dividers.shape[0] - 1))
        speed_divider = self._speed_dividers[speed_index]
        effective_divider = jnp.maximum(1, speed_divider)
        period = jnp.maximum(1, 16 // effective_divider)
        half_period = jnp.maximum(1, period // 2)
        speed_sign = jnp.sign(speed).astype(jnp.float32)
        
        move_y = jnp.logical_and((step_counter % period) == (half_period % period), speed != 0)
        move_x = jnp.logical_and((step_counter % period) == 0, speed != 0)
        step_size = jnp.where(speed_index >= 6, 1.5 + (speed_index - 6) * 0.2, 1.0)
        
        return move_y, move_x, step_size, speed_sign

    @partial(jax.jit, static_argnums=(0,))
    def _get_slope_and_intercept_from_indices(self, current_road: chex.Array, road_index_A: chex.Array, road_index_B: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Calculate slope and intercept for the current road segment."""
        road_index = jnp.where(current_road == 0, road_index_A, road_index_B)
        x1 = jnp.where(current_road == 0, 
                       self.consts.FIRST_TRACK_CORNERS_X[road_index], 
                       self.consts.SECOND_TRACK_CORNERS_X[road_index])
        x2 = jnp.where(current_road == 0, 
                       self.consts.FIRST_TRACK_CORNERS_X[road_index + 1], 
                       self.consts.SECOND_TRACK_CORNERS_X[road_index + 1])
        y1 = self.consts.TRACK_CORNERS_Y[road_index]
        y2 = self.consts.TRACK_CORNERS_Y[road_index + 1]
        
        dx = x2 - x1
        dy = y2 - y1
        slope = jnp.where(dx != 0, dy / dx, 300.0)
        b = y1 - slope * x1
        return slope, b

    @partial(jax.jit, static_argnums=(0,))
    def _is_on_line_for_position(self, position: EntityPosition, slope: chex.Array, b: chex.Array, player_speed: chex.Array, turn: chex.Array) -> chex.Array:
        x_step = abs(jnp.subtract(position.y, slope * (position.x) + b))
        y_step = abs(jnp.subtract(position.y - player_speed, slope * position.x + b))
        prefer_y = jnp.less_equal(y_step, x_step)
        return jnp.logical_or(
            jnp.logical_and(turn == 1, prefer_y),
            jnp.logical_and(turn == 2, jnp.logical_not(prefer_y)),
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_x_on_road(self, y: chex.Array, road_segment: chex.Array, track_corners_x: chex.Array) -> chex.Array:
        """Calculate the X position on a road given a Y coordinate and road segment."""
        y1 = self.consts.TRACK_CORNERS_Y[road_segment]
        y2 = self.consts.TRACK_CORNERS_Y[road_segment + 1]
        x1 = track_corners_x[road_segment]
        x2 = track_corners_x[road_segment + 1]
        
        # Linear interpolation: x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        t = jnp.where(y2 != y1, (y - y1) / (y2 - y1), 0.0)
        return x1 + t * (x2 - x1)

    @partial(jax.jit, static_argnums=(0,))
    def _get_x_for_road_index(self, y: chex.Array, road_segment: chex.Array, road_index: chex.Array) -> chex.Array:
        """Get X position on road A (index 0) or road B (index 1) for given Y and segment."""
        track_corners = jnp.where(
            road_index == 0,
            self.consts.FIRST_TRACK_CORNERS_X[road_segment],
            self.consts.SECOND_TRACK_CORNERS_X[road_segment],
        )
        track_corners_next = jnp.where(
            road_index == 0,
            self.consts.FIRST_TRACK_CORNERS_X[road_segment + 1],
            self.consts.SECOND_TRACK_CORNERS_X[road_segment + 1],
        )
        y1 = self.consts.TRACK_CORNERS_Y[road_segment]
        y2 = self.consts.TRACK_CORNERS_Y[road_segment + 1]
        t = jnp.where(y2 != y1, (y - y1) / (y2 - y1), 0.0)
        return track_corners + t * (track_corners_next - track_corners)

    @partial(jax.jit, static_argnums=(0,))
    def _get_road_segment(self, y: chex.Array) -> chex.Array:
        """Return the road segment index for a given y position."""
        segments = jnp.sum(self.consts.TRACK_CORNERS_Y > y, dtype=jnp.int32)
        max_idx = jnp.int32(len(self.consts.TRACK_CORNERS_Y) - 1)
        return jnp.clip(segments - 1, 0, max_idx)

    @partial(jax.jit, static_argnums=(0,))
    def _compute_direction_x(self, current_road: chex.Array, road_index_A: chex.Array, road_index_B: chex.Array) -> chex.Array:
        """Calculate the X direction for movement on the current road segment.
        
        Returns:
            Direction as int32: -1 for left, 1 for right (defaults to -1 for vertical segments)
        """
        # Select the road index based on which road we're on
        road_index = jnp.where(current_road == 0, road_index_A, road_index_B)
        # Select corners for the current road
        x_curr = jnp.where(current_road == 0, 
                           self.consts.FIRST_TRACK_CORNERS_X[road_index], 
                           self.consts.SECOND_TRACK_CORNERS_X[road_index])
        x_next = jnp.where(current_road == 0, 
                           self.consts.FIRST_TRACK_CORNERS_X[road_index + 1], 
                           self.consts.SECOND_TRACK_CORNERS_X[road_index + 1])
        direction_raw = x_next - x_curr
        return jnp.where(direction_raw == 0, -1, jnp.sign(direction_raw)).astype(jnp.int32)

    @partial(jax.jit, static_argnums=(0,))
    def _move_on_road(
        self,
        position: EntityPosition,
        slope: chex.Array,
        b: chex.Array,
        speed_sign: chex.Array,
        step_size: chex.Array,
        car_direction_x: chex.Array,
        move_y: chex.Array,
        move_x: chex.Array,
    ) -> Tuple[chex.Array, chex.Array]:
        """Move a car on the road based on timing and geometry.
        
        Returns:
            Tuple of (new_x, new_y) positions
        """
        new_y = jnp.where(
            jnp.logical_and(move_y, self._is_on_line_for_position(position, slope, b, speed_sign, 1)),
            position.y + speed_sign * -step_size,
            position.y,
        )
        
        new_x = jnp.where(
            jnp.logical_and(move_x, self._is_on_line_for_position(position, slope, b, speed_sign, 2)),
            position.x + speed_sign * car_direction_x * step_size,
            position.x,
        )
        
        return new_x, new_y

    @partial(jax.jit, static_argnums=(0,))
    def _is_steep_road_segment(self, current_road: chex.Array, road_index_A: chex.Array, road_index_B: chex.Array) -> chex.Array:
        """Check if the current road segment is steep (no X direction change).
        
        A steep segment is one where the X coordinates of consecutive corners are the same,
        meaning the road goes straight up/down with no horizontal movement.
        
        Returns True if the segment is steep (requires jump to pass when going up).
        """
        # Get the X difference for the current road segment
        road_index = jnp.where(current_road == 0, road_index_A, road_index_B)
        x_curr = jnp.where(current_road == 0, 
                           self.consts.FIRST_TRACK_CORNERS_X[road_index], 
                           self.consts.SECOND_TRACK_CORNERS_X[road_index])
        x_next = jnp.where(current_road == 0, 
                           self.consts.FIRST_TRACK_CORNERS_X[road_index + 1], 
                           self.consts.SECOND_TRACK_CORNERS_X[road_index + 1])
        x_diff = jnp.abs(x_next - x_curr)
        # A segment is steep if there's no X change (or very small change)
        return x_diff < 1.0

    @partial(jax.jit, static_argnums=(0,))
    def _get_steep_segment_progress(self, position_y: chex.Array, current_road: chex.Array, 
                                     road_index_A: chex.Array, road_index_B: chex.Array) -> chex.Array:
        """Calculate progress (0.0 to 1.0) through the current steep road segment.
        
        0.0 = at the bottom (start) of the steep segment
        1.0 = at the top (end) of the steep segment
        
        Progress is measured in the direction of forward travel (upward = positive Y direction in game space,
        but Y decreases as we go forward on the track).
        """
        road_index = jnp.where(current_road == 0, road_index_A, road_index_B)
        # Y coordinates of segment boundaries
        y_start = self.consts.TRACK_CORNERS_Y[road_index]      # Start of segment (lower Y = further ahead)
        y_end = self.consts.TRACK_CORNERS_Y[road_index + 1]    # End of segment (higher Y in absolute terms)
        
        # Calculate progress: how far through the segment are we?
        # Since Y decreases as we go forward, we need to invert
        segment_length = jnp.abs(y_end - y_start)
        # Distance from segment start (in forward direction)
        distance_from_start = jnp.abs(position_y - y_start)
        
        progress = jnp.where(segment_length > 0.001, distance_from_start / segment_length, 0.0)
        return jnp.clip(progress, 0.0, 1.0)

    @partial(jax.jit, static_argnums=(0,))
    def _check_landing_position(
        self,
        road_index_A: chex.Array,
        road_index_B: chex.Array,
        new_position_x: chex.Array,
        new_position_y: chex.Array,
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        """Check if a position is valid for landing (on or between roads).
        
        Returns:
            Tuple of (landing_in_water, between_roads, road_A_x, road_B_x)
        """
        # Calculate X position on road A at the given Y
        y_ratio_A = (new_position_y - self.consts.TRACK_CORNERS_Y[road_index_A]) / (
            self.consts.TRACK_CORNERS_Y[road_index_A + 1] - self.consts.TRACK_CORNERS_Y[road_index_A]
        )
        road_A_x = y_ratio_A * (
            self.consts.FIRST_TRACK_CORNERS_X[road_index_A + 1] - self.consts.FIRST_TRACK_CORNERS_X[road_index_A]
        ) + self.consts.FIRST_TRACK_CORNERS_X[road_index_A]
        
        # Calculate X position on road B at the given Y
        y_ratio_B = (new_position_y - self.consts.TRACK_CORNERS_Y[road_index_B]) / (
            self.consts.TRACK_CORNERS_Y[road_index_B + 1] - self.consts.TRACK_CORNERS_Y[road_index_B]
        )
        road_B_x = y_ratio_B * (
            self.consts.SECOND_TRACK_CORNERS_X[road_index_B + 1] - self.consts.SECOND_TRACK_CORNERS_X[road_index_B]
        ) + self.consts.SECOND_TRACK_CORNERS_X[road_index_B]
        
        distance_to_road_A = jnp.abs(new_position_x - road_A_x)
        distance_to_road_B = jnp.abs(new_position_x - road_B_x)
        landing_in_water = jnp.logical_and(
            distance_to_road_A > self.consts.LANDING_TOLERANCE,
            distance_to_road_B > self.consts.LANDING_TOLERANCE,
        )
        between_roads = jnp.logical_and(
            new_position_x > jnp.minimum(road_A_x, road_B_x),
            new_position_x < jnp.maximum(road_A_x, road_B_x),
        )
        return landing_in_water, between_roads, road_A_x, road_B_x

    @partial(jax.jit, static_argnums=(0,))
    def _advance_player_car(
        self,
        position_x: chex.Array,
        position_y: chex.Array,
        road_index_A: chex.Array,
        road_index_B: chex.Array,
        current_road: chex.Array,
        speed: chex.Array,
        is_jumping: chex.Array,
        step_counter: chex.Array,
        width: chex.Array,
        height: chex.Array,
        car_type: chex.Array,
        is_landing: chex.Array,
        stored_jump_slope: chex.Array,
        jump_progress: chex.Array,
    ) -> Car:
        """
        Advance the player car position.
        
        Jump logic:
        - Car jumps in the direction of the road it's on at current speed
        - While jumping, car moves freely (not constrained to road)
        - On landing: check if car is on/near a road or between roads
        - If between roads: snap to nearest road
        - If too far from both roads (outside the road area): crash (water)
        """
        # Calculate movement timing using helper
        move_y, move_x, step_size, speed_sign = self._compute_movement_timing(speed, step_counter)

        # Get slope and intercept for current road
        slope, b = self._get_slope_and_intercept_from_indices(current_road, road_index_A, road_index_B)

        # Determine X direction based on current road segment (for normal movement)
        car_direction_x = self._compute_direction_x(current_road, road_index_A, road_index_B)

        position = EntityPosition(x=position_x, y=position_y, width=width, height=height)

        # === CALCULATE ROAD-BASED MOVEMENT (used when not jumping) ===
        road_x, road_y = self._move_on_road(
            position, slope, b, speed_sign, step_size, car_direction_x, move_y, move_x
        )

        # === JUMP PHYSICS NORMALIZATION ===
        # Normalize jump velocity so total speed (Euclidean) matches 'step_size'
        # Without this, diagonal jumps cover more distance per frame than straight road movement
        # stored_jump_slope is dX/dY
        # Scaling factor = 1 / sqrt(1 + slope^2)
        jump_speed_scaling = 1.0 / jnp.sqrt(1.0 + stored_jump_slope**2)
        jump_step_size = step_size * jump_speed_scaling

        # === Y MOVEMENT ===
        # When jumping: move freely in Y direction but with normalized speed
        # When on road: use road-based movement result
        # Note: We must apply step_y on move_y ticks to keep sync with engine heartbeat
        jump_y = jnp.where(move_y, position_y + speed_sign * -jump_step_size, position_y)
        new_player_y = jnp.where(is_jumping, jump_y, road_y)

        # === X MOVEMENT ===
        # When jumping: use stored_jump_slope (locked at jump start) - moves X proportionally to Y
        # Use jump_step_size to maintain correct trajectory and speed
        # X step = slope * Y step magnitude = slope * jump_step_size
        raw_jump_x = jnp.where(move_x, position_x - speed_sign * stored_jump_slope * jump_step_size, position_x)
        
        # === AIR STEERING / MAGNETISM ===
        # Gradually steer towards the nearest road while in the air to prevent "teleporting" on landing
        segment_curr = self._get_road_segment(new_player_y)
        road_A_x_curr = self._get_x_on_road(new_player_y, segment_curr, self.consts.FIRST_TRACK_CORNERS_X)
        road_B_x_curr = self._get_x_on_road(new_player_y, segment_curr, self.consts.SECOND_TRACK_CORNERS_X)
        
        dist_A = jnp.abs(raw_jump_x - road_A_x_curr)
        dist_B = jnp.abs(raw_jump_x - road_B_x_curr)
        
        # Find closest road center
        target_road_x = jnp.where(dist_A < dist_B, road_A_x_curr, road_B_x_curr)
        dist_to_target = target_road_x - raw_jump_x
        
        # Only nudge in the last 25% of the jump (progress > 0.75)
        # when reasonably close to a road (within 2x tolerance)
        # and only when player is between the two roads
        
        is_late_jump = jump_progress > 0.75
        is_reasonably_close = jnp.abs(dist_to_target) < (self.consts.LANDING_TOLERANCE * 2.0)
        
        # Check if player is between the two roads
        min_road_x_curr = jnp.minimum(road_A_x_curr, road_B_x_curr)
        max_road_x_curr = jnp.maximum(road_A_x_curr, road_B_x_curr)
        is_between_roads = jnp.logical_and(raw_jump_x > min_road_x_curr, raw_jump_x < max_road_x_curr)
        
        should_magnet = jnp.logical_and(is_late_jump, jnp.logical_and(is_reasonably_close, is_between_roads))
        
        # Nudge factor: reduced to 2% steering strength (very subtle)
        nudge_amount = dist_to_target * 0.08
        
        jump_x = raw_jump_x + jnp.where(should_magnet, nudge_amount, 0.0)
        
        new_player_x = jnp.where(is_jumping, jump_x, road_x)

        # === LANDING LOGIC ===
        # Get the current road segment based on new Y position
        segment = self._get_road_segment(new_player_y)
        
        # Calculate X positions of both roads at the new Y position
        road_A_x = self._get_x_on_road(new_player_y, segment, self.consts.FIRST_TRACK_CORNERS_X)
        road_B_x = self._get_x_on_road(new_player_y, segment, self.consts.SECOND_TRACK_CORNERS_X)
        
        # Calculate distances to each road
        dist_to_road_A = jnp.abs(new_player_x - road_A_x)
        dist_to_road_B = jnp.abs(new_player_x - road_B_x)
        
        # Check if player is close enough to either road (within tolerance)
        on_road_A = dist_to_road_A <= self.consts.LANDING_TOLERANCE
        on_road_B = dist_to_road_B <= self.consts.LANDING_TOLERANCE
        on_any_road = jnp.logical_or(on_road_A, on_road_B)
        
        # Check if player is between the two roads
        min_road_x = jnp.minimum(road_A_x, road_B_x)
        max_road_x = jnp.maximum(road_A_x, road_B_x)
        between_roads = jnp.logical_and(new_player_x > min_road_x, new_player_x < max_road_x)
        
        # Determine which road is closer
        closer_to_A = dist_to_road_A < dist_to_road_B
        nearest_road_x = jnp.where(closer_to_A, road_A_x, road_B_x)
        nearest_road_id = jnp.where(closer_to_A, jnp.int32(0), jnp.int32(1))
        
        # === LANDING OUTCOMES ===
        # Valid landing: on a road OR between roads (will snap to nearest)
        valid_landing = jnp.logical_or(on_any_road, between_roads)
        
        # Bridge crossing physics: if speed is high, we can "skip" small water gaps (land on nearest road)
        # In original game, bridges allow crossing without jumping if you have speed
        can_bridge_gap = jnp.abs(speed) >= 5
        
        # If landing and between roads but not directly on a road, snap to nearest road
        should_snap = jnp.logical_and(is_landing, jnp.logical_and(between_roads, jnp.logical_not(on_any_road)))
        # Also snap if we are "in water" but have speed to bridge the gap
        should_snap_bridge = jnp.logical_and(is_landing, jnp.logical_and(can_bridge_gap, jnp.logical_not(valid_landing)))
        
        final_player_x = jnp.where(jnp.logical_or(should_snap, should_snap_bridge), nearest_road_x, new_player_x)
        
        # Water landing (crash): Only if NOT on road AND NOT between roads (i.e., landed completely outside)
        # User clarification: "crashing should only be possible if you dont land in betweeen or on the roads"
        
        # Safe if: ON ROAD or BETWEEN ROADS
        is_safe_landing = jnp.logical_or(on_any_road, between_roads)
        
        landing_in_water = jnp.logical_and(
            is_landing, 
            jnp.logical_not(is_safe_landing)
        )
        
        # Snap logic: 
        # If landing BETWEEN roads but not ON a road -> snap to nearest (safe!)
        # (Outside landings are now crashes, so no need to snap them)
        should_snap = jnp.logical_and(is_landing, jnp.logical_and(between_roads, jnp.logical_not(on_any_road)))
        
        # Also snap if bridging (fast jump across water gap)
        should_snap_bridge = jnp.logical_and(is_landing, jnp.logical_and(between_roads, can_bridge_gap))
        
        final_player_x = jnp.where(
            jnp.logical_or(should_snap, should_snap_bridge), 
            nearest_road_x, 
            new_player_x
        )
        
        # === UPDATE ROAD STATE ===
        # Determine which road to assign on landing (priority: road A > road B > nearest)
        landed_road = jnp.where(on_road_A, jnp.int32(0), jnp.where(on_road_B, jnp.int32(1), nearest_road_id))
        
        # Update current_road using nested jnp.where for vectorized execution
        # Priority: water crash > landing > jumping (frozen) > recover from water > normal
        normal_road = jnp.where(current_road == 2, nearest_road_id, current_road)
        jumping_road = jnp.where(is_jumping, current_road, normal_road)
        landing_road = jnp.where(is_landing, landed_road, jumping_road)
        updated_current_road = jnp.where(landing_in_water, jnp.int32(2), landing_road)
        
        # Update road indices to match current segment when not jumping
        not_jumping_on_road_A = jnp.logical_and(jnp.logical_not(is_jumping), updated_current_road == 0)
        not_jumping_on_road_B = jnp.logical_and(jnp.logical_not(is_jumping), updated_current_road == 1)
        next_road_index_A = jnp.where(not_jumping_on_road_A, segment, road_index_A)
        next_road_index_B = jnp.where(not_jumping_on_road_B, segment, road_index_B)

        # Wrap Y position for looping track
        wrapped_y = -((new_player_y * -1) % self.consts.TRACK_LENGTH)

        return Car(
            position=EntityPosition(
                x=final_player_x,
                y=wrapped_y,
                width=width,
                height=height,
            ),
            speed=speed,
            direction_x=car_direction_x,
            current_road=updated_current_road,
            road_index_A=next_road_index_A,
            road_index_B=next_road_index_B,
            type=car_type,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _advance_car_core(
        self,
        position_x: chex.Array,
        position_y: chex.Array,
        road_index_A: chex.Array,
        road_index_B: chex.Array,
        current_road: chex.Array,
        speed: chex.Array,
        step_counter: chex.Array,
        width: chex.Array,
        height: chex.Array,
        car_type: chex.Array,
    ) -> Car:
        """Simplified car advancement for enemy cars (no jumping/landing logic)."""
        # Calculate movement timing using helper
        move_y, move_x, step_size, speed_sign = self._compute_movement_timing(speed, step_counter)
        slope, b = self._get_slope_and_intercept_from_indices(current_road, road_index_A, road_index_B)
        car_direction_x = self._compute_direction_x(current_road, road_index_A, road_index_B)
        
        position = EntityPosition(x=position_x, y=position_y, width=width, height=height)
        
        # Use shared movement helper
        new_x, new_y = self._move_on_road(
            position, slope, b, speed_sign, step_size, car_direction_x, move_y, move_x
        )

        wrapped_y = -((new_y * -1) % self.consts.TRACK_LENGTH)
        
        # Update road segment indices based on new position
        segment_from_y = self._get_road_segment(new_y)
        
        # Update road indices to track the current segment (use jnp.where for branchless execution)
        next_road_index_A = jnp.where(current_road == 0, segment_from_y, road_index_A)
        next_road_index_B = jnp.where(current_road == 1, segment_from_y, road_index_B)

        return Car(
            position=EntityPosition(
                x=new_x,
                y=wrapped_y,
                width=width,
                height=height,
            ),
            speed=speed,
            direction_x=car_direction_x,
            current_road=current_road,
            road_index_A=next_road_index_A,
            road_index_B=next_road_index_B,
            type=car_type,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _flag_step(self, state: UpNDownState, new_player_y: chex.Array, player_x: chex.Array, current_road: chex.Array) -> Tuple[Flag, chex.Array, chex.Array]:
        """Update flag collection state and score (vectorized)."""
        # Calculate flag X positions on both roads
        # _get_x_on_road supports array inputs via advanced indexing
        x_road_0 = self._get_x_on_road(state.flags.y, state.flags.road_segment, self.consts.FIRST_TRACK_CORNERS_X)
        x_road_1 = self._get_x_on_road(state.flags.y, state.flags.road_segment, self.consts.SECOND_TRACK_CORNERS_X)
        
        flag_x = jnp.where(state.flags.road == 0, x_road_0, x_road_1)
        
        # Vectorized distance check
        y_dist = jnp.abs(new_player_y - state.flags.y)
        x_dist = jnp.abs(player_x - flag_x)
        same_road = (current_road == state.flags.road)
        
        new_collections = jnp.logical_and(
            jnp.logical_and(y_dist < self.consts.COLLISION_THRESHOLD, x_dist < self.consts.COLLISION_THRESHOLD),
            jnp.logical_and(same_road, ~state.flags.collected)
        )
        
        # Update flags collected state
        new_flags_collected = jnp.logical_or(state.flags.collected, new_collections)
        new_flags_collected_mask = jnp.logical_or(state.flags_collected_mask, new_collections)
        
        # Update score based on collected flags
        flag_score = jnp.sum(new_collections.astype(jnp.int32) * self.consts.FLAG_COLLECTION_SCORE)
        
        new_flags = Flag(
            y=state.flags.y,
            road=state.flags.road,
            road_segment=state.flags.road_segment,
            color_idx=state.flags.color_idx,
            collected=new_flags_collected,
        )
        
        return new_flags, flag_score, new_flags_collected_mask

    @partial(jax.jit, static_argnums=(0,))
    def _collectible_step(self, state: UpNDownState, new_player_y: chex.Array, player_x: chex.Array, current_road: chex.Array, rng_key: chex.PRNGKey) -> Tuple[Collectible, chex.Array, chex.Array, chex.PRNGKey]:
        """Update collectible spawning, despawning, and collection (unified for all types).

        Handles mixed-type collectibles (cherry, balloon, lollypop, ice cream) in a single pool.
        Type is randomized on spawn with probabilities defined in COLLECTIBLE_SPAWN_PROBABILITIES.
        
        Args:
            state: Current game state
            new_player_y: Updated player Y position after movement
            player_x: Current player X position
            current_road: Current road player is on
            rng_key: PRNG key to drive spawn randomness
            
        Returns:
            Tuple of (updated_collectibles, score_delta, new_spawn_timer, new_rng_key)
        """
        rng_key, key1, key2, key3, key4 = jax.random.split(rng_key, 5)

        # Collectible spawning logic - decrement timer and spawn when ready (use jnp.where for branchless)
        new_collectible_timer = jnp.where(
            state.collectible_spawn_timer <= 0,
            self.consts.COLLECTIBLE_SPAWN_INTERVAL,
            state.collectible_spawn_timer - 1,
        )
        
        # Attempt to spawn when timer hits 0
        should_spawn = state.collectible_spawn_timer <= 0
        
        inactive_mask = ~state.collectibles.active
        first_inactive = jnp.argmax(inactive_mask.astype(jnp.int32))
        has_inactive_slot = jnp.any(inactive_mask)
        spawn_idx = jnp.where(has_inactive_slot, first_inactive, jnp.array(0, dtype=jnp.int32))
        
        y_spawn = jax.random.uniform(key1, minval=-900.0, maxval=-100.0)
        road_spawn = jnp.array(jax.random.randint(key2, shape=(), minval=0, maxval=2), dtype=jnp.int32)
        color_spawn = jnp.array(jax.random.randint(key3, shape=(), minval=0, maxval=len(self.consts.COLLECTIBLE_COLORS)), dtype=jnp.int32)
        
        # Randomly select collectible type using cumulative probability thresholds
        # COLLECTIBLE_SPAWN_PROBABILITIES contains cumulative values: [35, 65, 90, 100]
        # Cherry: [0-35), Balloon: [35-65), Lollypop: [65-90), IceCream: [90-100]
        rand_type = jax.random.uniform(key4, minval=0.0, maxval=100.0)
        
        # Use searchsorted for efficient threshold lookup
        type_id_spawn = jnp.searchsorted(self.consts.COLLECTIBLE_SPAWN_PROBABILITIES, rand_type, side='right')
        type_id_spawn = jnp.clip(type_id_spawn, 0, 3).astype(jnp.int32)
        
        # Calculate X position on road (use jnp.where for branchless)
        segment_spawn = self._get_road_segment(y_spawn)
        x_spawn = jnp.where(
            road_spawn == 0,
            self._get_x_on_road(y_spawn, segment_spawn, self.consts.FIRST_TRACK_CORNERS_X),
            self._get_x_on_road(y_spawn, segment_spawn, self.consts.SECOND_TRACK_CORNERS_X),
        )
        
        # Create mask for which collectibles to update
        update_mask = (jnp.arange(self.consts.MAX_COLLECTIBLES) == spawn_idx) & should_spawn & has_inactive_slot
        
        # Update collectibles with proper masking - spawn new items
        spawned_y = jnp.where(update_mask, y_spawn, state.collectibles.y)
        spawned_x = jnp.where(update_mask, x_spawn, state.collectibles.x)
        spawned_road = jnp.where(update_mask, road_spawn, state.collectibles.road)
        spawned_color_idx = jnp.where(update_mask, color_spawn, state.collectibles.color_idx)
        spawned_type_id = jnp.where(update_mask, type_id_spawn, state.collectibles.type_id)
        spawned_active = jnp.where(update_mask, True, state.collectibles.active)
        
        # Despawn logic - remove collectibles too far from player
        def check_despawn(idx):
            c_y = spawned_y[idx]
            c_active = spawned_active[idx]
            distance = jnp.abs(new_player_y - c_y)
            too_far = distance > self.consts.COLLECTIBLE_DESPAWN_DISTANCE
            should_despawn = jnp.logical_and(c_active, too_far)
            return should_despawn
        
        despawn_mask = jax.vmap(check_despawn)(jnp.arange(self.consts.MAX_COLLECTIBLES))
        active_after_despawn = jnp.logical_and(spawned_active, ~despawn_mask)
        
        # Collision detection
        def check_collision(idx):
            c_y = spawned_y[idx]
            c_x = spawned_x[idx]
            c_road = spawned_road[idx]
            c_active = spawned_active[idx]
            
            y_distance = jnp.abs(new_player_y - c_y)
            x_distance = jnp.abs(player_x - c_x)
            same_road = (current_road == c_road)
            
            collision = jnp.logical_and(
                jnp.logical_and(y_distance < self.consts.COLLISION_THRESHOLD, x_distance < self.consts.COLLISION_THRESHOLD),
                jnp.logical_and(same_road, c_active)
            )
            return collision
        
        collections = jax.vmap(check_collision)(jnp.arange(self.consts.MAX_COLLECTIBLES))
        
        # Deactivate collected items
        final_active = jnp.logical_and(active_after_despawn, ~collections)
        
        # Update score - vectorized lookup without vmap overhead
        scores = self.consts.COLLECTIBLE_SCORES[spawned_type_id]
        score_delta = jnp.sum(jnp.where(collections, scores, 0))
        
        # Create final collectibles state
        updated_collectibles = Collectible(
            y=spawned_y,
            x=spawned_x,
            road=spawned_road,
            color_idx=spawned_color_idx,
            type_id=spawned_type_id,
            active=final_active,
        )
        
        return updated_collectibles, score_delta, new_collectible_timer, rng_key

    @partial(jax.jit, static_argnums=(0,))
    def _death_step(self, state: UpNDownState) -> UpNDownState:
        """Handle player death - this is now only used for water crashes during landing.
        
        When the player dies:
        - Lives are decremented
        - is_dead is set to True
        - awaiting_respawn is set to True
        - Player car is moved off-screen (despawned)
        - Game waits for player input before respawning
        """
        # Skip if already awaiting respawn
        already_awaiting = state.awaiting_respawn
        
        # Player on water road (index 2 assumed water) and not already dead
        died = jnp.logical_and(
            jnp.logical_and(
                state.player_car.current_road == 2,
                ~state.is_dead,
            ),
            ~already_awaiting,
        )

        # Use jnp.where for branchless execution
        lives = jnp.where(died, state.lives - 1, state.lives)
        is_dead = jnp.logical_or(state.is_dead, died)
        awaiting_respawn = jnp.logical_or(state.awaiting_respawn, died)
        
        # Stop player movement but keep position (renderer will hide player when awaiting_respawn)
        player_car = state.player_car._replace(
            speed=jnp.where(died, 0, state.player_car.speed),
        )

        return state._replace(
            lives=lives,
            is_dead=is_dead,
            awaiting_respawn=awaiting_respawn,
            player_car=player_car,
        )
    

    @partial(jax.jit, static_argnums=(0,))
    def _player_step(self, state: UpNDownState, action: chex.Array) -> UpNDownState:
        up = jnp.logical_or(action == Action.UP, action == Action.UPFIRE)
        down = jnp.logical_or(action == Action.DOWN, action == Action.DOWNFIRE)
        jump_pressed = jnp.logical_or(action == Action.FIRE, jnp.logical_or(action == Action.UPFIRE, action == Action.DOWNFIRE))
        
        # Check if on a steep road section FIRST (before applying speed changes)
        is_on_steep_road = self._is_steep_road_segment(
            state.player_car.current_road,
            state.player_car.road_index_A,
            state.player_car.road_index_B,
        )
        
        # Calculate progress through steep segment (0.0 = bottom, 1.0 = top)
        steep_progress = self._get_steep_segment_progress(
            state.player_car.position.y,
            state.player_car.current_road,
            state.player_car.road_index_A,
            state.player_car.road_index_B,
        )
        
        # Determine if player is on steep road going up (not jumping)
        on_steep_not_jumping = jnp.logical_and(is_on_steep_road, jnp.logical_not(state.is_jumping))
        
        # Start with current speed
        player_speed = state.player_car.speed
        
        # === FRICTION & MOMENTUM LOGIC ===
        is_accelerating = up
        is_braking = down
        
        # No friction - speed stays constant when no input
        # Speed changes gradually (periodically, not every frame)
        should_change_speed = (state.step_counter % self.consts.ACCELERATION_INTERVAL) == 0
        
        # === ACCELERATION (UP) ===
        # On steep road: UP action has NO effect (can't accelerate while on steep section)
        can_accelerate = jnp.logical_not(on_steep_not_jumping)
        
        player_speed = jnp.where(
            jnp.logical_and(
                jnp.logical_and(should_change_speed, is_accelerating), 
                jnp.logical_and(player_speed < self.consts.MAX_SPEED, can_accelerate)
            ),
            player_speed + 1,
            player_speed,
        )
        
        # === BRAKING (DOWN) ===
        # DOWN action always works (can brake/reverse)
        player_speed = jnp.where(
            jnp.logical_and(
                jnp.logical_and(should_change_speed, is_braking),
                player_speed > -self.consts.MAX_SPEED
            ),
            player_speed - 1,
            player_speed,
        )
        
        # === STEEP ROAD SPEED REDUCTION & SLIDE BACK ===
        # Only apply when on steep road, not jumping, and trying to go up (positive speed)
        on_steep_going_up = jnp.logical_and(on_steep_not_jumping, player_speed > 0)
        
        # Update steep road timer - increment when on steep road going up
        steep_road_timer = jnp.where(
            on_steep_going_up,
            state.steep_road_timer + 1,
            jnp.array(0, dtype=jnp.int32),
        )
        
        # Check if player has reached halfway point (50% progress through segment)
        past_halfway = steep_progress >= 0.5
        
        # Check if player has enough momentum to climb steep road
        MIN_CLIMB_SPEED = 5
        has_momentum = player_speed >= MIN_CLIMB_SPEED
        
        # Two behaviors based on progress:
        # 1. Before halfway: gradually reduce speed using timer
        # 2. At/past halfway: immediately slide back UNLESS we have enough momentum
        
        # Before halfway: reduce speed periodically using timer
        should_reduce_speed = jnp.logical_and(
            on_steep_going_up,
            jnp.logical_and(
                jnp.logical_not(past_halfway),
                steep_road_timer >= self.consts.STEEP_ROAD_SPEED_REDUCTION_INTERVAL
            )
        )
        player_speed = jnp.where(
            should_reduce_speed,
            jnp.maximum(player_speed - 1, jnp.int32(0)),  # Reduce but not below 0 yet
            player_speed,
        )
        # Reset timer after speed reduction
        steep_road_timer = jnp.where(
            should_reduce_speed,
            jnp.array(0, dtype=jnp.int32),
            steep_road_timer,
        )
        
        # At/past halfway: force speed to -2 (slide back down) IF momentum is lost
        should_slide_back = jnp.logical_and(
            on_steep_going_up, 
            jnp.logical_and(past_halfway, jnp.logical_not(has_momentum))
        )
        player_speed = jnp.where(
            should_slide_back,
            jnp.int32(-3),
            player_speed,
        )

        # === JUMP LOGIC ===
        can_start_jump = jnp.logical_and(
            state.jump_cooldown == 0, 
            jnp.logical_and(state.post_jump_cooldown == 0, state.jump_key_released)
        )
        is_jumping = jnp.logical_or(
            jnp.logical_and(state.is_jumping, state.jump_cooldown > 0),
            jnp.logical_and(state.is_on_road, jnp.logical_and(player_speed >= 0, jnp.logical_and(can_start_jump, jump_pressed))),
        )
        
        # Detect when a new jump is starting (was not jumping, now is jumping)
        starting_jump = jnp.logical_and(is_jumping, jnp.logical_not(state.is_jumping))
        
        # Calculate jump slope at jump start (X change per Y step)
        # Uses the road segment slope to follow the road trajectory
        # Use jnp.where for branchless execution
        road_index = jnp.where(
            state.player_car.current_road == 0,
            state.player_car.road_index_A,
            state.player_car.road_index_B,
        )
        
        # Get corner coordinates for the current segment
        # Segment goes from corner[road_index] to corner[road_index+1]
        # Use jnp.where for branchless execution
        start_x = jnp.where(
            state.player_car.current_road == 0,
            self.consts.FIRST_TRACK_CORNERS_X[road_index],
            self.consts.SECOND_TRACK_CORNERS_X[road_index],
        )
        end_x = jnp.where(
            state.player_car.current_road == 0,
            self.consts.FIRST_TRACK_CORNERS_X[road_index + 1],
            self.consts.SECOND_TRACK_CORNERS_X[road_index + 1],
        )
        start_y = self.consts.TRACK_CORNERS_Y[road_index]

        end_y = jnp.where(
            jnp.equal(self.consts.FIRST_TRACK_CORNERS_X[road_index + 1], self.consts.FIRST_TRACK_CORNERS_X[road_index + 2]),
            self.consts.TRACK_CORNERS_Y[road_index + 2],
            self.consts.TRACK_CORNERS_Y[road_index + 1],
        )
        
        # Calculate slope: how much X changes per unit Y change
        delta_x = end_x - start_x
        delta_y = end_y - start_y
        # Avoid division by zero for horizontal segments (use jnp.where)
        new_jump_slope = jnp.where(
            jnp.abs(delta_y) > 0.001,
            jnp.float32(delta_x) / jnp.float32(delta_y),
            jnp.float32(0.0),
        )
        
        # Lock slope at jump start, keep previous slope during jump (use jnp.where)
        jump_slope = jnp.where(starting_jump, new_jump_slope, state.jump_slope)

        # Calculate dynamic jump duration based on speed
        # Faster speed = shorter jump duration (covering gap faster)
        # Increased base duration for more "air time" as requested
        # Formula: 48 - 2 * abs(speed) -> Speed 8 = 32 frames (was 24 before)
        current_jump_duration = 48 - 2 * jnp.abs(player_speed)
        jump_duration = jnp.where(starting_jump, current_jump_duration.astype(jnp.int32), state.jump_total_duration)

        # Use jnp.where for branchless execution of jump_cooldown
        jump_cooldown = jnp.where(
            state.jump_cooldown > 0,
            state.jump_cooldown - 1,
            jnp.where(is_jumping, jump_duration, 0),
        )

        # Use jnp.where for branchless execution of post_jump_cooldown
        is_landing_now = jnp.logical_and(state.jump_cooldown == 1, jump_cooldown == 0)
        post_jump_cooldown = jnp.where(
            is_landing_now,
            self.consts.POST_JUMP_DELAY,
            jnp.where(state.post_jump_cooldown > 0, state.post_jump_cooldown - 1, state.post_jump_cooldown),
        )
        is_on_road = ~is_jumping
        is_landing = is_landing_now

        # Calculate jump progress for magnetism
        # Progress = (Total - Remaining) / Total
        # Use jnp.maximum(..., 1.0) to avoid division by zero
        safe_total_duration = jnp.maximum(state.jump_total_duration, 1.0)
        jump_progress = (safe_total_duration - jump_cooldown.astype(jnp.float32)) / safe_total_duration
        jump_progress = jnp.clip(jump_progress, 0.0, 1.0)

        updated_player_car = self._advance_player_car(
            position_x=state.player_car.position.x,
            position_y=state.player_car.position.y,
            road_index_A=state.player_car.road_index_A,
            road_index_B=state.player_car.road_index_B,
            current_road=state.player_car.current_road,
            speed=player_speed,
            is_jumping=is_jumping,
            step_counter=state.step_counter,
            width=state.player_car.position.width,
            height=state.player_car.position.height,
            car_type=state.player_car.type,
            is_landing=is_landing,
            stored_jump_slope=jump_slope,
            jump_progress=jump_progress,
        )

        # Check if a speed-changing action (UP or DOWN) was taken
        speed_action_taken = jnp.logical_or(up, down)
        # Round starts only after a speed-changing action
        round_started_now = jnp.logical_or(state.round_started, speed_action_taken)
        
        # Track jump key release for preventing held-key jumps
        next_jump_key_released = jnp.logical_not(jump_pressed)

        next_state = state._replace(
            jump_cooldown=jump_cooldown,
            post_jump_cooldown=post_jump_cooldown,
            is_jumping=is_jumping,
            is_on_road=is_on_road,
            player_car=updated_player_car,
            step_counter=state.step_counter + 1,
            round_started=round_started_now,
            movement_steps=jnp.where(round_started_now, state.movement_steps + 1, state.movement_steps),
            steep_road_timer=steep_road_timer,
            jump_slope=jump_slope,
            jump_key_released=next_jump_key_released,
            jump_total_duration=jump_duration,
        )

        water_crash = jnp.logical_and(is_landing, updated_player_car.current_road == 2)

        # On water crash, trigger death state instead of immediate respawn
        def trigger_death(s):
            # Stop player but keep position (renderer will hide player when awaiting_respawn)
            dead_car = s.player_car._replace(
                speed=jnp.array(0, dtype=jnp.int32),
            )
            return s._replace(
                lives=s.lives - 1,
                is_dead=jnp.array(True),
                awaiting_respawn=jnp.array(True),
                player_car=dead_car,
            )

        return jax.lax.cond(
            water_crash,
            lambda _: trigger_death(next_state),
            lambda _: next_state,
            operand=None,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _flag_step_main(self, state: UpNDownState) -> UpNDownState:
        """Update flag collection state and score."""
        new_player_y = state.player_car.position.y
        player_x = state.player_car.position.x
        current_road = state.player_car.current_road
        
        new_flags, flag_score, new_flags_collected_mask = self._flag_step(
            state, new_player_y, player_x, current_road
        )
        
        return state._replace(
            score=state.score + flag_score,
            flags=new_flags,
            flags_collected_mask=new_flags_collected_mask,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _level_progression_step(self, state: UpNDownState) -> UpNDownState:
        """Handle level completion: award bonus and reset flags."""
        all_flags_collected = jnp.all(state.flags_collected_mask)
        
        bonus = jnp.where(all_flags_collected, self.consts.ALL_FLAGS_BONUS, 0)
        
        # Reset flags if all collected
        new_collected = jnp.where(all_flags_collected, jnp.zeros_like(state.flags.collected), state.flags.collected)
        new_mask = jnp.where(all_flags_collected, jnp.zeros_like(state.flags_collected_mask), state.flags_collected_mask)
        
        updated_flags = state.flags._replace(collected=new_collected)
        
        return state._replace(
            score=state.score + bonus,
            flags=updated_flags,
            flags_collected_mask=new_mask
        )

    @partial(jax.jit, static_argnums=(0,))
    def _extra_life_step(self, state: UpNDownState) -> UpNDownState:
        """Award extra life every 10000 points."""
        next_milestone = state.last_extra_life_score + self.consts.EXTRA_LIFE_THRESHOLD
        should_award = state.score >= next_milestone
        
        new_lives = jnp.where(should_award, state.lives + 1, state.lives)
        new_last_score = jnp.where(should_award, next_milestone, state.last_extra_life_score)
        
        return state._replace(lives=new_lives, last_extra_life_score=new_last_score)

    @partial(jax.jit, static_argnums=(0,))
    def _collectible_step_main(self, state: UpNDownState) -> UpNDownState:
        """Update collectible spawning, despawning, and collection."""
        new_player_y = state.player_car.position.y
        player_x = state.player_car.position.x
        current_road = state.player_car.current_road
        
        updated_collectibles, collectible_score, new_collectible_timer, rng_key = self._collectible_step(
            state, new_player_y, player_x, current_road, state.rng_key
        )
        
        return state._replace(
            score=state.score + collectible_score,
            collectibles=updated_collectibles,
            collectible_spawn_timer=new_collectible_timer,
            rng_key=rng_key,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _initialize_collectibles(self) -> Collectible:
        """Return a cleared collectible pool."""
        return Collectible(
            y=jnp.zeros(self.consts.MAX_COLLECTIBLES),
            x=jnp.zeros(self.consts.MAX_COLLECTIBLES),
            road=jnp.zeros(self.consts.MAX_COLLECTIBLES, dtype=jnp.int32),
            color_idx=jnp.zeros(self.consts.MAX_COLLECTIBLES, dtype=jnp.int32),
            type_id=jnp.zeros(self.consts.MAX_COLLECTIBLES, dtype=jnp.int32),
            active=jnp.zeros(self.consts.MAX_COLLECTIBLES, dtype=jnp.bool_),
        )

    @partial(jax.jit, static_argnums=(0,))
    def _initialize_enemies(self, key: chex.Array, player_start_y: chex.Array) -> EnemyCars:
        """Seed the initial set of visible enemies around the player."""
        key_init, key_type, key_road, key_speed, key_sign = jax.random.split(key, 5)

        offsets = self.consts.INITIAL_ENEMY_BASE_OFFSET + self.consts.INITIAL_ENEMY_GAP * jnp.arange(self.consts.INITIAL_ENEMY_COUNT)
        spawn_signs = jax.random.choice(key_sign, jnp.array([-1.0, 1.0]), shape=(self.consts.INITIAL_ENEMY_COUNT,))
        raw_spawn_y = player_start_y + spawn_signs * offsets
        init_y = -(((raw_spawn_y) * -1) % self.consts.TRACK_LENGTH)
        init_road = jax.random.randint(key_road, shape=(self.consts.INITIAL_ENEMY_COUNT,), minval=0, maxval=2)

        init_segments = jax.vmap(self._get_road_segment)(init_y)

        init_x = jax.vmap(lambda y, seg, road: jax.lax.cond(
            road == 0,
            lambda _: self._get_x_on_road(y, seg, self.consts.FIRST_TRACK_CORNERS_X),
            lambda _: self._get_x_on_road(y, seg, self.consts.SECOND_TRACK_CORNERS_X),
            operand=None,
        ))(init_y, init_segments, init_road)

        init_type = jax.random.randint(key_type, shape=(self.consts.INITIAL_ENEMY_COUNT,), minval=0, maxval=4)
        init_speed_mag = jax.random.randint(key_speed, shape=(self.consts.INITIAL_ENEMY_COUNT,), minval=self.consts.ENEMY_SPEED_MIN, maxval=self.consts.ENEMY_SPEED_MAX + 1)
        init_speed_sign = jax.random.choice(key_init, jnp.array([-1, 1]), shape=(self.consts.INITIAL_ENEMY_COUNT,))
        init_speed = init_speed_mag * init_speed_sign

        def init_direction(seg, road):
            raw = jax.lax.cond(
                road == 0,
                lambda _: self.consts.FIRST_TRACK_CORNERS_X[seg+1] - self.consts.FIRST_TRACK_CORNERS_X[seg],
                lambda _: self.consts.SECOND_TRACK_CORNERS_X[seg+1] - self.consts.SECOND_TRACK_CORNERS_X[seg],
                operand=None,
            )
            return jax.lax.cond(raw > 0, lambda _: 1, lambda _: -1, operand=None)

        init_dir = jax.vmap(init_direction)(init_segments, init_road)

        pad = self.consts.MAX_ENEMY_CARS - self.consts.INITIAL_ENEMY_COUNT

        return EnemyCars(
            position=EntityPosition(
                x=jnp.concatenate([init_x, jnp.zeros(pad, dtype=jnp.float32)]),
                y=jnp.concatenate([init_y, jnp.zeros(pad, dtype=jnp.float32)]),
                width=jnp.full((self.consts.MAX_ENEMY_CARS,), self.consts.PLAYER_SIZE[0]),
                height=jnp.full((self.consts.MAX_ENEMY_CARS,), self.consts.PLAYER_SIZE[1]),
            ),
            speed=jnp.concatenate([init_speed, jnp.zeros(pad, dtype=jnp.int32)]),
            type=jnp.concatenate([init_type, jnp.zeros(pad, dtype=jnp.int32)]),
            current_road=jnp.concatenate([init_road, jnp.zeros(pad, dtype=jnp.int32)]),
            road_index_A=jnp.concatenate([init_segments, jnp.zeros(pad, dtype=jnp.int32)]),
            road_index_B=jnp.concatenate([init_segments, jnp.zeros(pad, dtype=jnp.int32)]),
            direction_x=jnp.concatenate([init_dir, jnp.zeros(pad, dtype=jnp.int32)]),
            active=jnp.concatenate([jnp.ones(self.consts.INITIAL_ENEMY_COUNT, dtype=jnp.bool_), jnp.zeros(pad, dtype=jnp.bool_)]),
            age=jnp.concatenate([jnp.zeros(self.consts.INITIAL_ENEMY_COUNT, dtype=jnp.int32), jnp.zeros(pad, dtype=jnp.int32)]),
        )

    @partial(jax.jit, static_argnums=(0,))
    def _enemy_step_main(self, state: UpNDownState) -> UpNDownState:
        """Spawn and move enemy cars with adaptive spawning for consistent enemy presence."""
        # Split RNG keys - use more splits to ensure better randomization
        rng_key, key_spawn_offset, key_spawn_side, key_spawn_speed, key_spawn_direction, key_spawn_type, key_spawn_sign, key_flip_root, key_extra = jax.random.split(state.rng_key, 9)
        
        # Further split key_spawn_type to get more entropy for type selection
        key_spawn_type = jax.random.fold_in(key_spawn_type, state.step_counter)

        active_mask = state.enemy_cars.active
        active_count = jnp.sum(active_mask.astype(jnp.int32))
        can_spawn = active_count < self.consts.MAX_ENEMY_CARS

        # Calculate how many enemies are "visible" (within visible distance of player)
        player_y = state.player_car.position.y
        enemy_distances = jnp.abs(state.enemy_cars.position.y - player_y)
        wrapped_distances = jnp.minimum(enemy_distances, self.consts.TRACK_LENGTH - enemy_distances)
        visible_mask = jnp.logical_and(active_mask, wrapped_distances < self.consts.ENEMY_VISIBLE_DISTANCE)
        visible_count = jnp.sum(visible_mask.astype(jnp.int32))

        # Adaptive spawn interval: spawn faster when fewer visible enemies
        # If below minimum, spawn immediately (interval = 0)
        # Otherwise scale between BASE and MAX based on visible count
        needs_urgent_spawn = visible_count < self.consts.ENEMY_MIN_VISIBLE_COUNT
        spawn_interval = jnp.where(
            needs_urgent_spawn,
            jnp.int32(0),  # Spawn immediately when too few visible
            jnp.int32(self.consts.ENEMY_SPAWN_INTERVAL_BASE + 
                     (visible_count * (self.consts.ENEMY_SPAWN_INTERVAL_MAX - self.consts.ENEMY_SPAWN_INTERVAL_BASE)) // 
                     self.consts.MAX_ENEMY_CARS)
        )

        # Spawn when timer expires OR when we urgently need more enemies
        timer_expired = state.enemy_spawn_timer <= 0
        should_spawn = jnp.logical_and(
            jnp.logical_or(timer_expired, needs_urgent_spawn),
            can_spawn
        )
        
        # Reset timer with adaptive interval
        spawn_timer = jnp.where(
            should_spawn,
            spawn_interval,
            jnp.maximum(state.enemy_spawn_timer - 1, 0),
        )

        inactive_mask = jnp.logical_not(active_mask)
        first_inactive = jnp.argmax(inactive_mask.astype(jnp.int32))
        has_inactive = jnp.any(inactive_mask)
        spawn_idx = jnp.where(has_inactive, first_inactive, jnp.array(0, dtype=jnp.int32))
        spawn_mask = (jnp.arange(self.consts.MAX_ENEMY_CARS) == spawn_idx) & should_spawn & has_inactive

        # Spawn closer when urgent (fewer visible enemies), farther when plenty exist
        base_offset = jnp.where(
            needs_urgent_spawn,
            self.consts.ENEMY_SPAWN_OFFSET_MIN,  # Spawn closer when needed
            self.consts.ENEMY_SPAWN_OFFSET_MIN + visible_count * 10.0  # Farther when plenty exist
        )
        spawn_offset = base_offset + jax.random.uniform(key_spawn_offset, minval=0.0, maxval=30.0)
        
        spawn_side = jax.random.choice(key_spawn_side, jnp.array([-1.0, 1.0]))
        raw_spawn_y = state.player_car.position.y + spawn_side * spawn_offset
        spawn_y = -(((raw_spawn_y) * -1) % self.consts.TRACK_LENGTH)
        spawn_road = jax.random.randint(key_spawn_direction, shape=(), minval=0, maxval=2)

        segment_spawn = self._get_road_segment(spawn_y)
        spawn_x = jnp.where(
            spawn_road == 0,
            self._get_x_on_road(spawn_y, segment_spawn, self.consts.FIRST_TRACK_CORNERS_X),
            self._get_x_on_road(spawn_y, segment_spawn, self.consts.SECOND_TRACK_CORNERS_X),
        )

        spawn_speed_mag = jax.random.randint(key_spawn_speed, shape=(), minval=self.consts.ENEMY_SPEED_MIN, maxval=self.consts.ENEMY_SPEED_MAX + 1)
        spawn_speed_sign = jax.random.choice(key_spawn_sign, jnp.array([-1, 1]))
        spawn_speed = spawn_speed_mag * spawn_speed_sign
        spawn_type = jax.random.randint(key_spawn_type, shape=(), minval=0, maxval=4)

        direction_raw = jnp.where(
            spawn_road == 0,
            self.consts.FIRST_TRACK_CORNERS_X[segment_spawn+1] - self.consts.FIRST_TRACK_CORNERS_X[segment_spawn],
            self.consts.SECOND_TRACK_CORNERS_X[segment_spawn+1] - self.consts.SECOND_TRACK_CORNERS_X[segment_spawn],
        )
        spawn_direction_x = jnp.where(direction_raw > 0, 1, -1)

        enemy_position_x = jnp.where(spawn_mask, spawn_x, state.enemy_cars.position.x)
        enemy_position_y = jnp.where(spawn_mask, spawn_y, state.enemy_cars.position.y)
        enemy_width = state.enemy_cars.position.width
        enemy_height = state.enemy_cars.position.height
        enemy_speed = jnp.where(spawn_mask, spawn_speed, state.enemy_cars.speed)
        enemy_type = jnp.where(spawn_mask, spawn_type, state.enemy_cars.type)
        enemy_current_road = jnp.where(spawn_mask, spawn_road, state.enemy_cars.current_road)
        enemy_road_index_A = jnp.where(spawn_mask, segment_spawn, state.enemy_cars.road_index_A)
        enemy_road_index_B = jnp.where(spawn_mask, segment_spawn, state.enemy_cars.road_index_B)
        enemy_direction_x = jnp.where(spawn_mask, spawn_direction_x, state.enemy_cars.direction_x)
        enemy_active = jnp.where(spawn_mask, True, state.enemy_cars.active)
        enemy_age = jnp.where(spawn_mask, jnp.zeros_like(state.enemy_cars.age), state.enemy_cars.age)

        flip_keys = jax.random.split(key_flip_root, self.consts.MAX_ENEMY_CARS)
        flip_mask = jax.vmap(lambda k: jax.random.uniform(k) < self.consts.ENEMY_DIRECTION_SWITCH_PROB)(flip_keys)
        enemy_speed = jnp.where(jnp.logical_and(enemy_active, flip_mask), -enemy_speed, enemy_speed)

        move_fn = lambda px, py, ra, rb, cr, sp, tp: self._advance_car_core(
            position_x=px,
            position_y=py,
            road_index_A=ra,
            road_index_B=rb,
            current_road=cr,
            speed=sp,
            step_counter=state.step_counter,
            width=self.consts.PLAYER_SIZE[0],
            height=self.consts.PLAYER_SIZE[1],
            car_type=tp,
        )

        advanced_cars = jax.vmap(move_fn)(
            enemy_position_x,
            enemy_position_y,
            enemy_road_index_A,
            enemy_road_index_B,
            enemy_current_road,
            enemy_speed,
            enemy_type,
        )

        moved_position_x = jnp.where(enemy_active, advanced_cars.position.x, enemy_position_x)
        moved_position_y = jnp.where(enemy_active, advanced_cars.position.y, enemy_position_y)
        moved_road_index_A = jnp.where(enemy_active, advanced_cars.road_index_A, enemy_road_index_A)
        moved_road_index_B = jnp.where(enemy_active, advanced_cars.road_index_B, enemy_road_index_B)
        moved_current_road = jnp.where(enemy_active, advanced_cars.current_road, enemy_current_road)
        moved_direction_x = jnp.where(enemy_active, advanced_cars.direction_x, enemy_direction_x)

        enemy_age = jnp.where(enemy_active, enemy_age + 1, enemy_age)

        delta_y = moved_position_y - state.player_car.position.y
        wrapped_dist = jnp.minimum(jnp.abs(delta_y), self.consts.TRACK_LENGTH - jnp.abs(delta_y))
        far_mask = wrapped_dist > self.consts.ENEMY_DESPAWN_DISTANCE
        age_mask = enemy_age > self.consts.ENEMY_MAX_AGE
        despawn_mask = jnp.logical_and(enemy_active, jnp.logical_or(far_mask, age_mask))
        final_active = jnp.logical_and(enemy_active, jnp.logical_not(despawn_mask))
        enemy_age = jnp.where(despawn_mask, jnp.zeros_like(enemy_age), enemy_age)

        next_enemy_cars = EnemyCars(
            position=EntityPosition(
                x=moved_position_x,
                y=moved_position_y,
                width=enemy_width,
                height=enemy_height,
            ),
            speed=enemy_speed,
            type=enemy_type,
            current_road=moved_current_road,
            road_index_A=moved_road_index_A,
            road_index_B=moved_road_index_B,
            direction_x=moved_direction_x,
            active=final_active,
            age=enemy_age,
        )

        return state._replace(
            enemy_cars=next_enemy_cars,
            enemy_spawn_timer=spawn_timer,
            rng_key=rng_key,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _respawn_after_collision(self, state: UpNDownState, new_lives: chex.Array) -> UpNDownState:
        """Respawn the player on a random road while preserving score and flags."""
        rng_key, road_key, enemy_key = jax.random.split(state.rng_key, 3)

        player_start_y = jnp.array(0.0)
        start_segment = jnp.array(0, dtype=jnp.int32)
        respawn_road = jax.random.randint(road_key, shape=(), minval=0, maxval=2)

        start_x = jax.lax.cond(
            respawn_road == 0,
            lambda _: self._get_x_on_road(player_start_y, start_segment, self.consts.FIRST_TRACK_CORNERS_X),
            lambda _: self._get_x_on_road(player_start_y, start_segment, self.consts.SECOND_TRACK_CORNERS_X),
            operand=None,
        )

        enemy_cars = self._initialize_enemies(enemy_key, player_start_y)
        collectibles = self._initialize_collectibles()

        player_car = Car(
            position=EntityPosition(
                x=jnp.asarray(start_x, dtype=jnp.float32),
                y=jnp.asarray(player_start_y, dtype=jnp.float32),
                width=self.consts.PLAYER_SIZE[0],
                height=self.consts.PLAYER_SIZE[1],
            ),
            speed=jnp.array(0, dtype=jnp.int32),
            direction_x=jnp.array(0, dtype=jnp.int32),
            current_road=respawn_road,
            road_index_A=start_segment,
            road_index_B=start_segment,
            type=jnp.array(0, dtype=jnp.int32),
        )

        return UpNDownState(
            score=state.score,
            lives=new_lives,
            is_dead=jnp.array(False),
            respawn_timer=jnp.array(0, dtype=jnp.int32),
            difficulty=state.difficulty,
            jump_cooldown=jnp.array(0, dtype=jnp.int32),
            post_jump_cooldown=jnp.array(0, dtype=jnp.int32),
            is_jumping=jnp.array(False),
            is_on_road=jnp.array(True),
            player_car=player_car,
            step_counter=state.step_counter,
            round_started=jnp.array(False),
            movement_steps=jnp.array(0),
            steep_road_timer=jnp.array(0, dtype=jnp.int32),
            jump_slope=jnp.array(0.0, dtype=jnp.float32),
            flags=state.flags,
            flags_collected_mask=state.flags_collected_mask,
            collectibles=collectibles,
            collectible_spawn_timer=jnp.array(0, dtype=jnp.int32),
            enemy_cars=enemy_cars,
            enemy_spawn_timer=jnp.array(self.consts.ENEMY_SPAWN_INTERVAL_BASE, dtype=jnp.int32),
            awaiting_respawn=jnp.array(False),
            awaiting_round_start=jnp.array(True),  # Wait for input to start round after respawn
            input_released=jnp.array(False),  # Require button release before round can start
            jump_key_released=jnp.array(True),
            last_extra_life_score=state.last_extra_life_score,
            jump_total_duration=jnp.array(self.consts.JUMP_FRAMES, dtype=jnp.int32),
            rng_key=rng_key,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _enemy_collision_step_main(self, state: UpNDownState) -> UpNDownState:
        """Handle collisions between the player and enemy cars.

        - While airborne, collisions are ignored except for the final jump frames,
          where hitting an enemy despawns it and awards a bonus.
        - On ground collisions, the player loses a life and the stage soft-resets
          without clearing score or collected flags.
        - Landing collisions use a larger distance and are road-independent (for crossings).
        """
        player_x = state.player_car.position.x
        player_y = state.player_car.position.y

        dx = jnp.abs(state.enemy_cars.position.x - player_x)
        dy = jnp.abs(state.enemy_cars.position.y - player_y)
        wrapped_dy = jnp.minimum(dy, self.consts.TRACK_LENGTH - dy)

        # For ground collision: only trigger when enemy position is within tight distance
        overlap_x_ground = dx <= self.consts.GROUND_COLLISION_DISTANCE
        overlap_y_ground = wrapped_dy <= self.consts.GROUND_COLLISION_DISTANCE
        # For late jump collision: use larger overlap based on car dimensions plus extra tolerance
        # "slightly more forgiving"
        jump_tolerance = 4.0
        overlap_x_jump = dx <= (state.player_car.position.width + state.enemy_cars.position.width) / 2.0 + jump_tolerance
        overlap_y_jump = wrapped_dy <= (state.player_car.position.height + state.enemy_cars.position.height) / 2.0 + jump_tolerance
        same_road = state.enemy_cars.current_road == state.player_car.current_road

        # Ground collision mask uses tight 3-pixel distance and same road
        ground_collision_mask = jnp.logical_and(state.enemy_cars.active, jnp.logical_and(same_road, jnp.logical_and(overlap_x_ground, overlap_y_ground)))
        # Jump collision mask is road-independent - can destroy enemies on either road when jumping
        jump_collision_mask = jnp.logical_and(state.enemy_cars.active, jnp.logical_and(overlap_x_jump, overlap_y_jump))
        collision_mask = jump_collision_mask  # For late jump scoring
        
        any_jump_collision = jnp.any(jump_collision_mask)
        any_ground_collision = jnp.any(ground_collision_mask)

        # Check if player is in post-landing invincibility phase
        is_invincible = state.post_jump_cooldown > 0
        
        late_jump_window = jnp.logical_and(state.is_jumping, state.jump_cooldown <= self.consts.LATE_JUMP_COLLISION_FRAMES)
        late_jump_collision = jnp.logical_and(any_jump_collision, late_jump_window)
        # Ground collision only applies when not jumping AND not in post-landing invincibility
        grounded_collision = jnp.logical_and(
            any_ground_collision,
            jnp.logical_and(jnp.logical_not(state.is_jumping), jnp.logical_not(is_invincible))
        )

        def handle_late_jump():
            hits = collision_mask.astype(jnp.int32)
            bonus = jnp.sum(hits) * self.consts.LATE_JUMP_ENEMY_SCORE
            new_enemy_active = jnp.logical_and(state.enemy_cars.active, jnp.logical_not(collision_mask))
            new_enemy_age = jnp.where(collision_mask, jnp.zeros_like(state.enemy_cars.age), state.enemy_cars.age)
            new_enemy_cars = EnemyCars(
                position=state.enemy_cars.position,
                speed=state.enemy_cars.speed,
                type=state.enemy_cars.type,
                current_road=state.enemy_cars.current_road,
                road_index_A=state.enemy_cars.road_index_A,
                road_index_B=state.enemy_cars.road_index_B,
                direction_x=state.enemy_cars.direction_x,
                active=new_enemy_active,
                age=new_enemy_age,
            )

            return state._replace(score=state.score + bonus, enemy_cars=new_enemy_cars)

        def handle_ground_collision():
            # Trigger death state - stop player but keep position (renderer hides player when awaiting_respawn)
            dead_car = state.player_car._replace(
                speed=jnp.array(0, dtype=jnp.int32),
            )
            return state._replace(
                lives=state.lives - 1,
                is_dead=jnp.array(True),
                awaiting_respawn=jnp.array(True),
                player_car=dead_car,
            )

        # Ground collision causes death (landing is now protected by invincibility)
        any_fatal_collision = grounded_collision

        return jax.lax.cond(
            late_jump_collision,
            lambda _: handle_late_jump(),
            lambda _: jax.lax.cond(
                any_fatal_collision,
                lambda _: handle_ground_collision(),
                lambda _: state,
                operand=None,
            ),
            operand=None,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _passive_score_step_main(self, state: UpNDownState) -> UpNDownState:
        """Award passive score at regular intervals after the player has started moving."""
        should_award = jnp.logical_and(
            state.round_started,
            state.movement_steps % self.consts.PASSIVE_SCORE_INTERVAL == 0,
        )
        bonus = jnp.where(should_award, jnp.int32(self.consts.PASSIVE_SCORE_AMOUNT), jnp.int32(0))

        return state._replace(score=state.score + bonus)
    
    @partial(jax.jit, static_argnums=(0,))
    def _reset_jit(self, key: chex.PRNGKey) -> Tuple[UpNDownObservation, UpNDownState]:
        rng_key, flag_key, enemy_key = jax.random.split(key, 3)

        # Evenly spread flags along the track with small jitter
        base_y = jnp.linspace(-900.0, -100.0, self.consts.NUM_FLAGS)
        jitter = jax.random.uniform(flag_key, shape=(self.consts.NUM_FLAGS,), minval=-40.0, maxval=40.0)
        flag_y_offsets = base_y + jitter

        # Alternate roads 0/1 for variety
        flag_roads = jnp.arange(self.consts.NUM_FLAGS) % 2

        # Calculate which road segment each flag is on based on Y position
        flag_segments = jax.vmap(self._get_road_segment)(flag_y_offsets)

        # Each flag color index corresponds to its position (0-7)
        flag_color_indices = jnp.arange(self.consts.NUM_FLAGS)

        flags = Flag(
            y=flag_y_offsets,
            road=flag_roads,
            road_segment=flag_segments,
            color_idx=flag_color_indices,
            collected=jnp.zeros(self.consts.NUM_FLAGS, dtype=jnp.bool_),
        )

        # Initialize collectibles as all inactive (will spawn dynamically with mixed types)
        collectibles = self._initialize_collectibles()

        # Seed initial visible enemies spaced around the player
        player_start_y = jnp.array(0.0)
        enemy_cars = self._initialize_enemies(enemy_key, player_start_y)

        state = UpNDownState(
            score=0,
            lives=jnp.array(self.consts.INITIAL_LIVES, dtype=jnp.int32),
            is_dead=jnp.array(False),
            respawn_timer=jnp.array(0, dtype=jnp.int32),
            difficulty=self.consts.DIFFICULTIES[0],
            jump_cooldown=0,
            post_jump_cooldown=0,
            is_jumping=False,
            is_on_road=True,
            player_car=Car(
                position=EntityPosition(
                    x=jnp.asarray(30, dtype=jnp.float32),
                    y=jnp.asarray(0, dtype=jnp.float32),
                    width=self.consts.PLAYER_SIZE[0],
                    height=self.consts.PLAYER_SIZE[1],
                ),
                speed=0,
                direction_x=0,
                current_road=0,
                road_index_A=0,
                road_index_B=0,
                type=0,
            ),
            step_counter=jnp.array(0),
            rng_key=rng_key,
            round_started=jnp.array(False),
            movement_steps=jnp.array(0),
            steep_road_timer=jnp.array(0, dtype=jnp.int32),
            jump_slope=jnp.array(0.0, dtype=jnp.float32),
            flags=flags,
            flags_collected_mask=jnp.zeros(self.consts.NUM_FLAGS, dtype=jnp.bool_),
            collectibles=collectibles,
            collectible_spawn_timer=jnp.array(0, dtype=jnp.int32),
            enemy_cars=enemy_cars,
            enemy_spawn_timer=jnp.array(self.consts.ENEMY_SPAWN_INTERVAL_BASE, dtype=jnp.int32),
            awaiting_respawn=jnp.array(False),
            awaiting_round_start=jnp.array(True),  # Start frozen until first input
            input_released=jnp.array(True),  # Can start immediately at game start
            jump_key_released=jnp.array(True),
            last_extra_life_score=jnp.array(0, dtype=jnp.int32),
            jump_total_duration=jnp.array(self.consts.JUMP_FRAMES, dtype=jnp.int32),
        )
        initial_obs = self._get_observation(state)
        return initial_obs, state
    
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key=None) -> Tuple[UpNDownObservation, UpNDownState]:
        if key is None:
            key = jax.random.PRNGKey(42)
        return self._reset_jit(key)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: UpNDownState, action: chex.Array) -> Tuple[UpNDownObservation, UpNDownState, float, bool, UpNDownInfo]:
        previous_state = state
        
        any_action = action != Action.NOOP
        
        # Track input release - set to True when no button is pressed
        input_released = jnp.where(any_action, state.input_released, jnp.array(True))
        state = state._replace(input_released=input_released)
        
        # Check if we're awaiting respawn - if so, check for input to trigger respawn
        should_respawn = jnp.logical_and(state.awaiting_respawn, any_action)
        
        # Respawn if player pressed any key while awaiting
        state = jax.lax.cond(
            should_respawn,
            lambda s: self._respawn_after_collision(s, s.lives),  # lives already decremented
            lambda s: s,
            state,
        )
        
        # Check if we're awaiting round start - if so, check for input to start round
        # Only start if input was released since respawn (prevents holding button through)
        should_start_round = jnp.logical_and(
            jnp.logical_and(state.awaiting_round_start, any_action),
            state.input_released  # Must have released button first
        )
        state = jax.lax.cond(
            should_start_round,
            lambda s: s._replace(awaiting_round_start=jnp.array(False)),
            lambda s: s,
            state,
        )
        
        # Skip all game logic if awaiting respawn OR awaiting round start
        is_frozen = jnp.logical_or(state.awaiting_respawn, state.awaiting_round_start)
        
        def run_game_logic(s):
            s = self._player_step(s, action)
            s = self._death_step(s)
            s = self._passive_score_step_main(s)
            s = self._flag_step_main(s)
            s = self._level_progression_step(s)
            s = self._extra_life_step(s)
            s = self._collectible_step_main(s)
            s = self._enemy_step_main(s)
            s = self._enemy_collision_step_main(s)
            return s
        
        def freeze_game(s):
            # Only increment step counter while frozen, everything else paused
            return s._replace(step_counter=s.step_counter + 1)
        
        # Run game logic only if not frozen
        state = jax.lax.cond(
            is_frozen,
            freeze_game,
            run_game_logic,
            state,
        )

        done = self._get_done(state)
        env_reward = self._get_reward(previous_state, state)
        info = self._get_info(state)
        observation = self._get_observation(state)

        return observation, state, env_reward, done, info


    def render(self, state: UpNDownState) -> jnp.ndarray:
        frame = self.renderer.render(state)
        return jnp.asarray(frame, dtype=jnp.uint8)

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: UpNDownState) -> UpNDownObservation:
        """Build complete observation for RL agents.
        
        Reuses existing game classes directly. Extra fields are filtered during flatten.
        """
        # Check if on steep road
        is_on_steep_road = self._is_steep_road_segment(
            state.player_car.current_road,
            state.player_car.road_index_A,
            state.player_car.road_index_B,
        )
        
        return UpNDownObservation(
            player_car=state.player_car,
            enemy_cars=state.enemy_cars,
            flags=state.flags,
            collectibles=state.collectibles,
            flags_collected_mask=state.flags_collected_mask.astype(jnp.int32),
            player_score=jnp.int32(state.score),
            lives=jnp.int32(state.lives),
            is_jumping=jnp.int32(state.is_jumping),
            jump_cooldown=jnp.int32(state.jump_cooldown),
            is_on_steep_road=jnp.int32(is_on_steep_road),
            round_started=jnp.int32(state.round_started),
        )

    @partial(jax.jit, static_argnums=(0,))
    def flatten_car(self, car: Car) -> jnp.ndarray:
        """Flatten a Car to a 1D array."""
        return jnp.concatenate([
            jnp.array([car.position.x], dtype=jnp.int32),
            jnp.array([car.position.y], dtype=jnp.int32),
            jnp.array([car.position.width], dtype=jnp.int32),
            jnp.array([car.position.height], dtype=jnp.int32),
            jnp.array([car.speed], dtype=jnp.int32),
            jnp.array([car.type], dtype=jnp.int32),
            jnp.array([car.current_road], dtype=jnp.int32),
            jnp.array([car.road_index_A], dtype=jnp.int32),
            jnp.array([car.road_index_B], dtype=jnp.int32),
            jnp.array([car.direction_x], dtype=jnp.int32),
        ])

    @partial(jax.jit, static_argnums=(0,))
    def flatten_enemy_cars(self, enemy_cars: EnemyCars) -> jnp.ndarray:
        """Flatten EnemyCars to a 1D array (all fields)."""
        return jnp.concatenate([
            enemy_cars.position.x.astype(jnp.int32),
            enemy_cars.position.y.astype(jnp.int32),
            enemy_cars.position.width.astype(jnp.int32),
            enemy_cars.position.height.astype(jnp.int32),
            enemy_cars.speed.astype(jnp.int32),
            enemy_cars.type.astype(jnp.int32),
            enemy_cars.current_road.astype(jnp.int32),
            enemy_cars.road_index_A.astype(jnp.int32),
            enemy_cars.road_index_B.astype(jnp.int32),
            enemy_cars.direction_x.astype(jnp.int32),
            enemy_cars.active.astype(jnp.int32),
            enemy_cars.age.astype(jnp.int32),
        ])

    @partial(jax.jit, static_argnums=(0,))
    def flatten_flags(self, flags: Flag) -> jnp.ndarray:
        """Flatten Flag to a 1D array."""
        return jnp.concatenate([
            flags.y.astype(jnp.int32),
            flags.road.astype(jnp.int32),
            flags.road_segment.astype(jnp.int32),
            flags.color_idx.astype(jnp.int32),
            flags.collected.astype(jnp.int32),
        ])

    @partial(jax.jit, static_argnums=(0,))
    def flatten_collectibles(self, collectibles: Collectible) -> jnp.ndarray:
        """Flatten Collectible to a 1D array (all fields)."""
        return jnp.concatenate([
            collectibles.y.astype(jnp.int32),
            collectibles.x.astype(jnp.int32),
            collectibles.road.astype(jnp.int32),
            collectibles.color_idx.astype(jnp.int32),
            collectibles.type_id.astype(jnp.int32),
            collectibles.active.astype(jnp.int32),
        ])

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: UpNDownObservation) -> jnp.ndarray:
        """Flatten the complete observation to a 1D array for RL.
        
        Order:
        - Player car: 10 values (x, y, w, h, speed, type, road, road_index_A, road_index_B, direction_x)
        - Enemy cars: MAX_ENEMY_CARS * 12 values (x, y, w, h, speed, type, road, road_index_A, road_index_B, direction_x, active, age)
        - Flags: NUM_FLAGS * 5 values (y, road, segment, color, collected per flag)
        - Collectibles: MAX_COLLECTIBLES * 6 values (y, x, road, color_idx, type, active per collectible)
        - Flags collected mask: NUM_FLAGS values
        - Score, lives, is_jumping, jump_cooldown, is_on_steep_road, round_started: 6 values
        """
        return jnp.concatenate([
            self.flatten_car(obs.player_car),
            self.flatten_enemy_cars(obs.enemy_cars),
            self.flatten_flags(obs.flags),
            self.flatten_collectibles(obs.collectibles),
            obs.flags_collected_mask.flatten().astype(jnp.int32),
            jnp.array([obs.player_score], dtype=jnp.int32),
            jnp.array([obs.lives], dtype=jnp.int32),
            jnp.array([obs.is_jumping], dtype=jnp.int32),
            jnp.array([obs.jump_cooldown], dtype=jnp.int32),
            jnp.array([obs.is_on_steep_road], dtype=jnp.int32),
            jnp.array([obs.round_started], dtype=jnp.int32),
        ])

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(6)

    def observation_space(self) -> spaces.Dict:
        """Returns the observation space for Up N Down.
        
        The observation reuses existing game classes:
        - player_car: Car with position (x, y, w, h), speed, type, current_road, direction_x
        - enemy_cars: EnemyCars with positions, speeds, types, roads, active flags
        - flags: Flag with y, road, road_segment, color_idx, collected
        - collectibles: Collectible with y, x, road, type_id, active
        - flags_collected_mask: boolean array of shape (NUM_FLAGS,)
        - player_score: int (0-999999)
        - lives: int (0-5)
        - is_jumping: int (0 or 1)
        - jump_cooldown: int (0-28)
        - is_on_steep_road: int (0 or 1)
        - round_started: int (0 or 1)
        """
        return spaces.Dict({
            "player_car": spaces.Dict({
                "position": spaces.Dict({
                    "x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                    "y": spaces.Box(low=-2000, high=0, shape=(), dtype=jnp.int32),
                    "width": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                    "height": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                }),
                "speed": spaces.Box(low=-6, high=6, shape=(), dtype=jnp.int32),
                "type": spaces.Box(low=0, high=3, shape=(), dtype=jnp.int32),
                "current_road": spaces.Box(low=0, high=2, shape=(), dtype=jnp.int32),
                "road_index_A": spaces.Box(low=0, high=30, shape=(), dtype=jnp.int32),
                "road_index_B": spaces.Box(low=0, high=30, shape=(), dtype=jnp.int32),
                "direction_x": spaces.Box(low=-1, high=1, shape=(), dtype=jnp.int32),
            }),
            "enemy_cars": spaces.Dict({
                "position": spaces.Dict({
                    "x": spaces.Box(low=0, high=160, shape=(self.consts.MAX_ENEMY_CARS,), dtype=jnp.int32),
                    "y": spaces.Box(low=-2000, high=0, shape=(self.consts.MAX_ENEMY_CARS,), dtype=jnp.int32),
                    "width": spaces.Box(low=0, high=160, shape=(self.consts.MAX_ENEMY_CARS,), dtype=jnp.int32),
                    "height": spaces.Box(low=0, high=210, shape=(self.consts.MAX_ENEMY_CARS,), dtype=jnp.int32),
                }),
                "speed": spaces.Box(low=-6, high=6, shape=(self.consts.MAX_ENEMY_CARS,), dtype=jnp.int32),
                "type": spaces.Box(low=0, high=3, shape=(self.consts.MAX_ENEMY_CARS,), dtype=jnp.int32),
                "current_road": spaces.Box(low=0, high=2, shape=(self.consts.MAX_ENEMY_CARS,), dtype=jnp.int32),
                "road_index_A": spaces.Box(low=0, high=30, shape=(self.consts.MAX_ENEMY_CARS,), dtype=jnp.int32),
                "road_index_B": spaces.Box(low=0, high=30, shape=(self.consts.MAX_ENEMY_CARS,), dtype=jnp.int32),
                "direction_x": spaces.Box(low=-1, high=1, shape=(self.consts.MAX_ENEMY_CARS,), dtype=jnp.int32),
                "active": spaces.Box(low=0, high=1, shape=(self.consts.MAX_ENEMY_CARS,), dtype=jnp.int32),
                "age": spaces.Box(low=0, high=10000, shape=(self.consts.MAX_ENEMY_CARS,), dtype=jnp.int32),
            }),
            "flags": spaces.Dict({
                "y": spaces.Box(low=-2000, high=0, shape=(self.consts.NUM_FLAGS,), dtype=jnp.int32),
                "road": spaces.Box(low=0, high=1, shape=(self.consts.NUM_FLAGS,), dtype=jnp.int32),
                "road_segment": spaces.Box(low=0, high=30, shape=(self.consts.NUM_FLAGS,), dtype=jnp.int32),
                "color_idx": spaces.Box(low=0, high=7, shape=(self.consts.NUM_FLAGS,), dtype=jnp.int32),
                "collected": spaces.Box(low=0, high=1, shape=(self.consts.NUM_FLAGS,), dtype=jnp.int32),
            }),
            "collectibles": spaces.Dict({
                "y": spaces.Box(low=-2000, high=0, shape=(self.consts.MAX_COLLECTIBLES,), dtype=jnp.int32),
                "x": spaces.Box(low=0, high=160, shape=(self.consts.MAX_COLLECTIBLES,), dtype=jnp.int32),
                "road": spaces.Box(low=0, high=1, shape=(self.consts.MAX_COLLECTIBLES,), dtype=jnp.int32),
                "color_idx": spaces.Box(low=0, high=7, shape=(self.consts.MAX_COLLECTIBLES,), dtype=jnp.int32),
                "type_id": spaces.Box(low=0, high=3, shape=(self.consts.MAX_COLLECTIBLES,), dtype=jnp.int32),
                "active": spaces.Box(low=0, high=1, shape=(self.consts.MAX_COLLECTIBLES,), dtype=jnp.int32),
            }),
            "flags_collected_mask": spaces.Box(low=0, high=1, shape=(self.consts.NUM_FLAGS,), dtype=jnp.int32),
            "player_score": spaces.Box(low=0, high=999999, shape=(), dtype=jnp.int32),
            "lives": spaces.Box(low=0, high=5, shape=(), dtype=jnp.int32),
            "is_jumping": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
            "jump_cooldown": spaces.Box(low=0, high=28, shape=(), dtype=jnp.int32),
            "is_on_steep_road": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
            "round_started": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
        })

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(210, 160, 3),
            dtype=jnp.uint8
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: UpNDownState) -> UpNDownInfo:
        """Build info dict with additional debugging/analysis data."""
        # Get current road segment for player
        road_index = jnp.where(
            state.player_car.current_road == 0,
            state.player_car.road_index_A,
            state.player_car.road_index_B,
        )
        
        return UpNDownInfo(
            step_counter=jnp.int32(state.step_counter),
            difficulty=jnp.int32(state.difficulty),
            movement_steps=jnp.int32(state.movement_steps),
            jump_slope=jnp.float32(state.jump_slope),
            player_road_segment=jnp.int32(road_index),
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: UpNDownState, state: UpNDownState):
        base_delta = jnp.asarray(state.score - previous_state.score, dtype=jnp.float32)
        if self.reward_funcs:
            extras = jnp.sum(jnp.array([fn(previous_state, state) for fn in self.reward_funcs], dtype=jnp.float32))
            return base_delta + extras
        return base_delta

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: UpNDownState) -> bool:
        return state.lives <= 0

class UpNDownRenderer(JAXGameRenderer):
    def __init__(self, consts: UpNDownConstants = None):
        super().__init__()
        self.consts = consts or UpNDownConstants()
        self.config = render_utils.RendererConfig(
            game_dimensions=(210, 160),
            channels=3,
            #downscale=(84, 84)
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)

        background = self._createBackgroundSprite(self.config.game_dimensions)
        top_block = self._createBackgroundSprite((25, self.config.game_dimensions[1]))
        bottom_block = self._createBackgroundSprite((16, self.config.game_dimensions[1]))
        temp_pointer = self._createBackgroundSprite((1, 1))
        blackout_square = self._createBackgroundSprite(self.consts.FLAG_BLACKOUT_SIZE)
        
        # Build asset config locally (matches other games' pattern)
        asset_config, road_files = self._get_asset_config(background, top_block, bottom_block, temp_pointer, blackout_square)
        sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/up_n_down/"

        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(asset_config, sprite_path)
        self.road_sizes, self.complete_road_size = self._get_road_sprite_sizes(road_files)
        self.view_height = self.config.game_dimensions[0]
        # Precompute offsets so repeated road tiles can wrap seamlessly without gaps.
        road_cycle = max(1, self.complete_road_size)
        repeats = max(1, int(-(-self.view_height // road_cycle)) + 2)  # Ceiling division trick
        self._road_tile_offsets = jnp.arange(-repeats, repeats + 1, dtype=jnp.int32) * jnp.int32(self.complete_road_size)
        self._num_road_tiles = int(self._road_tile_offsets.shape[0])

        self.enemy_sprite_names = {
            self.consts.ENEMY_TYPE_CAMERO: "camero_left",
            self.consts.ENEMY_TYPE_FLAG_CARRIER: "flag_carrier_left",
            self.consts.ENEMY_TYPE_PICKUP: "pick_up_truck_left",
            self.consts.ENEMY_TYPE_TRUCK: "truck_left",
        }

        # Pre-pad enemy masks to a common shape so switch/array indexing works under jit
        # Only use left sprites - right sprites are created by flipping horizontally
        enemy_left_raw = [
            self.SHAPE_MASKS["camero_left"],
            self.SHAPE_MASKS["flag_carrier_left"],
            self.SHAPE_MASKS["pick_up_truck_left"],
            self.SHAPE_MASKS["truck_left"],
        ]
        max_h = max([m.shape[0] for m in enemy_left_raw])
        max_w = max([m.shape[1] for m in enemy_left_raw])

        def _pad_mask(mask):
            pad_h = max_h - mask.shape[0]
            pad_w = max_w - mask.shape[1]
            return jnp.pad(mask, ((0, pad_h), (0, pad_w)), constant_values=self.jr.TRANSPARENT_ID)

        self.enemy_left_masks = jnp.stack([_pad_mask(m) for m in enemy_left_raw], axis=0)
        # Create right-facing masks by horizontally flipping the left masks
        self.enemy_right_masks = jnp.flip(self.enemy_left_masks, axis=2)
        
        # Precompute flag mask data for recoloring without special-casing pink
        self.flag_base_mask = self.SHAPE_MASKS["pink_flag"]
        self.flag_solid_mask = self.flag_base_mask != self.jr.TRANSPARENT_ID
        self.flag_palette_ids = self._compute_flag_palette_ids()
        
        # Precompute collectible mask data for recoloring (unified for all types)
        # Reuse the same palette IDs since all collectibles use FLAG_COLORS
        self.collectible_palette_ids = self.flag_palette_ids
        
        self.cherry_base_mask = self.SHAPE_MASKS["cherry"]
        self.cherry_solid_mask = self.cherry_base_mask != self.jr.TRANSPARENT_ID
        
        self.balloon_base_mask = self.SHAPE_MASKS["balloon"]
        self.balloon_solid_mask = self.balloon_base_mask != self.jr.TRANSPARENT_ID
        
        self.lollypop_base_mask = self.SHAPE_MASKS["lollypop"]
        self.lollypop_solid_mask = self.lollypop_base_mask != self.jr.TRANSPARENT_ID
        
        self.ice_cream_base_mask = self.SHAPE_MASKS["ice_cream"]
        self.ice_cream_solid_mask = self.ice_cream_base_mask != self.jr.TRANSPARENT_ID

        # Score rendering helpers
        self.score_digit_masks = self.SHAPE_MASKS["score_digits"]
        self.score_max_digits = 6
        self.score_digit_spacing = int(self.score_digit_masks.shape[2]) + 1
        self.score_render_y = 6
        self.score_center_x = self.config.game_dimensions[1] // 2 - self.config.game_dimensions[1] // 4

    def _createBackgroundSprite(self, dimensions: Tuple[int, int]) -> jnp.ndarray:
        """Creates a procedural background sprite for the game."""
        height, width = dimensions
        color = (0, 0, 0, 255)  # RGBA for wall color
        shape = (height, width, 4)  # Height, Width, RGBA channels
        sprite = jnp.tile(jnp.array(color, dtype=jnp.uint8), (*shape[:2], 1))
        return sprite
    
    def _get_road_sprite_sizes(self, road_files: list[str]) -> list:
        """Returns the sizes of the road sprites limited to the configured files."""
        road_dir = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/up_n_down/roads"
        sizes = []
        for file in road_files:
            sprite_name = os.path.basename(file)
            sprite = jnp.load(f"{road_dir}/{sprite_name}")
            sizes.append(sprite.shape[0])
        complete_size = int(sum(sizes))
        return sizes, complete_size

    def _get_asset_config(self, backgroundSprite: jnp.ndarray, topBlockSprite: jnp.ndarray, bottomBlockSprite: jnp.ndarray, tempPointer: jnp.ndarray, blackoutSquare: jnp.ndarray) -> tuple[list, list[str]]:
        """Return asset manifest and ordered road files (renderer-local like other games)."""
        road_dir = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/up_n_down/roads"
        road_files = sorted(
            file for file in os.listdir(road_dir)
            if file.endswith(".npy")
        )
        roads = [f"roads/{file}" for file in road_files]
        return [
            {'name': 'background', 'type': 'background', 'data': backgroundSprite},
            {'name': 'road', 'type': 'group', 'files': roads},
            {'name': 'player', 'type': 'single', 'file': 'player_car.npy'},
            {'name': 'camero_left', 'type': 'single', 'file': 'enemy_cars/camero_left.npy'},
            {'name': 'flag_carrier_left', 'type': 'single', 'file': 'enemy_cars/flag_carrier_left.npy'},
            {'name': 'pick_up_truck_left', 'type': 'single', 'file': 'enemy_cars/pick_up_truck_left.npy'},
            {'name': 'truck_left', 'type': 'single', 'file': 'enemy_cars/truck_left.npy'},
            {'name': 'wall_top', 'type': 'procedural', 'data': topBlockSprite},
            {'name': 'wall_bottom', 'type': 'procedural', 'data': bottomBlockSprite},
            {'name': 'all_flags_top', 'type': 'single', 'file': 'all_flags_top.npy'},
            {'name': 'all_lives_bottom', 'type': 'single', 'file': 'all_lives_bottom.npy'},
            {'name': 'score_digits', 'type': 'digits', 'pattern': 'score/score_{}.npy'},
            {'name': 'pink_flag', 'type': 'single', 'file': 'pink_flag.npy'},
            {'name': 'flag_pole', 'type': 'single', 'file': 'flag_pole.npy'},
            {'name': 'cherry', 'type': 'single', 'file': 'cherry.npy'},
            {'name': 'balloon', 'type': 'single', 'file': 'balloon.npy'},
            {'name': 'lollypop', 'type': 'single', 'file': 'lollypop.npy'},
            {'name': 'ice_cream', 'type': 'single', 'file': 'ice_cream_cone.npy'},
            {'name': 'tempPointer', 'type': 'procedural', 'data': tempPointer},
            {'name': 'blackout_square', 'type': 'procedural', 'data': blackoutSquare},
        ], roads
    
    def _get_x_on_road(self, y: chex.Array, road_segment: chex.Array, track_corners_x: chex.Array) -> chex.Array:
        """Calculate the X position on a road given a Y coordinate and road segment."""
        y1 = self.consts.TRACK_CORNERS_Y[road_segment]
        y2 = self.consts.TRACK_CORNERS_Y[road_segment + 1]
        x1 = track_corners_x[road_segment]
        x2 = track_corners_x[road_segment + 1]
        t = jnp.where(y2 != y1, (y - y1) / (y2 - y1), 0.0)
        return x1 + t * (x2 - x1)
    
    def _find_palette_id(self, rgba: jnp.ndarray) -> int:
        """Return palette index for an RGBA color, falling back to first entry if missing."""
        color_rgb = rgba[:3]
        palette_rgb = self.PALETTE[:, :3]
        matches = jnp.all(palette_rgb == color_rgb, axis=1)
        found = jnp.argmax(matches)
        # If no match, fallback to 0 (background) to avoid crashes
        return int(found)
    
    def _compute_flag_palette_ids(self) -> jnp.ndarray:
        """Precompute palette indices for each flag color without special-casing pink."""
        return jnp.array([self._find_palette_id(color) for color in self.consts.FLAG_COLORS], dtype=jnp.int32)

    @partial(jax.jit, static_argnums=(0,))
    def _jump_arc_offset(self, jump_cooldown: chex.Array, total_duration: chex.Array) -> chex.Array:
        """Return a simple parabolic jump height based on remaining jump frames."""
        total = total_duration.astype(jnp.float32)
        remaining = jnp.array(jump_cooldown, dtype=jnp.float32)
        progress = jnp.clip((total - remaining) / jnp.maximum(total, 1.0), 0.0, 1.0)
        centered = (progress - 0.5) * 2.0
        return self.consts.JUMP_ARC_HEIGHT * (1.0 - centered * centered)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        raster = self.jr.create_object_raster(self.BACKGROUND)
        road_diff = (-state.player_car.position.y) % self.complete_road_size

        # Vectorized road rendering: compute all Y offsets, stamp via vmap, fold overlays.
        road_masks = self.SHAPE_MASKS["road"]  # shape: (N, H, W)
        num_segments = road_masks.shape[0]

        sizes = jnp.array(self.road_sizes, dtype=jnp.int32)
        # Offsets: [0, cumsum(sizes[1:])]
        offsets = jnp.concatenate([
            jnp.array([0], dtype=jnp.int32),
            jnp.cumsum(sizes[1:], axis=0)
        ], axis=0)

        base_y = jnp.asarray(self.consts.INITIAL_ROAD_POS_Y, dtype=jnp.int32)
        y_positions = base_y + (road_diff.astype(jnp.int32)) - offsets

        tile_offsets = self._road_tile_offsets
        tile_count = self._num_road_tiles
        tiled_y = (y_positions[None, :] + tile_offsets[:, None]).reshape(-1)
        tiled_masks = jnp.tile(road_masks, (tile_count, 1, 1))
        tiled_sizes = jnp.tile(sizes, tile_count)

        visible = jnp.logical_and(
            tiled_y < self.view_height,
            (tiled_y + tiled_sizes) > 0
        )

        empty_raster = jnp.full_like(self.BACKGROUND, self.jr.TRANSPARENT_ID)

        def stamp(y, mask, is_visible):
            return jax.lax.cond(
                is_visible,
                lambda _: self.jr.render_at_clipped(empty_raster, 10, y, mask),
                lambda _: empty_raster,
                operand=None,
            )

        overlays = jax.vmap(stamp)(tiled_y, tiled_masks, visible)

        total_segments = tile_count * num_segments

        def combine(i, acc):
            over = overlays[i]
            return jnp.where(over != self.jr.TRANSPARENT_ID, over, acc)

        raster = jax.lax.fori_loop(0, total_segments, combine, raster)

        def select_enemy_mask(enemy_type: chex.Array, going_left: chex.Array):
            """Select enemy mask: left masks are base, right masks are horizontally flipped."""
            left_mask = self.enemy_left_masks[enemy_type]
            right_mask = self.enemy_right_masks[enemy_type]
            return jnp.where(going_left, left_mask, right_mask)

        # Pre-cast enemy properties to optimal types for rendering BEFORE the scan loop
        enemy_active_arr = state.enemy_cars.active
        enemy_x_arr = state.enemy_cars.position.x.astype(jnp.int32)
        enemy_y_arr = state.enemy_cars.position.y
        enemy_type_arr = state.enemy_cars.type
        enemy_direction_x_arr = state.enemy_cars.direction_x

        def render_enemy(carry, enemy_idx):
            raster = carry
            enemy_active = enemy_active_arr[enemy_idx]
            enemy_x = enemy_x_arr[enemy_idx]
            enemy_y = enemy_y_arr[enemy_idx]
            enemy_type = enemy_type_arr[enemy_idx]
            direction_x = enemy_direction_x_arr[enemy_idx]
            screen_y = 105 + (enemy_y - state.player_car.position.y)
            # Hide enemies when awaiting round start or awaiting respawn
            should_hide = jnp.logical_or(state.awaiting_round_start, state.awaiting_respawn)
            is_visible = jnp.logical_and(
                jnp.logical_and(enemy_active, jnp.logical_and(screen_y > 25, screen_y < 195)),
                ~should_hide
            )
            enemy_mask = select_enemy_mask(enemy_type, direction_x < 0)

            raster = jax.lax.cond(
                is_visible,
                lambda r: self.jr.render_at(r, enemy_x, screen_y.astype(jnp.int32), enemy_mask),
                lambda r: r,
                operand=raster,
            )
            return raster, None

        raster_enemies, _ = jax.lax.scan(render_enemy, raster, jnp.arange(self.consts.MAX_ENEMY_CARS))

        jump_offset = jax.lax.cond(
            state.is_jumping,
            lambda _: self._jump_arc_offset(state.jump_cooldown, state.jump_total_duration),
            lambda _: jnp.array(0.0, dtype=jnp.float32),
            operand=None,
        )

        player_screen_y = jnp.int32(105 - jump_offset)
        player_mask = self.SHAPE_MASKS["player"]
        # Skip rendering player when awaiting respawn OR awaiting round start
        should_hide_player = jnp.logical_or(state.awaiting_respawn, state.awaiting_round_start)
        raster_player = jax.lax.cond(
            should_hide_player,
            lambda _: raster_enemies,  # Don't render player
            lambda _: self.jr.render_at_clipped(raster_enemies, state.player_car.position.x, player_screen_y, player_mask),
            operand=None,
        )

        wall_top_mask = self.SHAPE_MASKS["wall_top"]
        raster_wall_top = self.jr.render_at(raster_player, 0, 0, wall_top_mask)

        wall_bottom_mask = self.SHAPE_MASKS["wall_bottom"]
        raster_wall_bottom = self.jr.render_at(raster_wall_top, 0, 210 - wall_bottom_mask.shape[0], wall_bottom_mask)

        all_flags_top_mask = self.SHAPE_MASKS["all_flags_top"]
        raster_flags_top = self.jr.render_at(raster_wall_bottom, 10, 20, all_flags_top_mask)

        # Render score centered at the top using dedicated score digit sprites
        score_digits = self.jr.int_to_digits(state.score, max_digits=self.score_max_digits)
        non_zero_mask = score_digits != 0
        has_non_zero = jnp.any(non_zero_mask)
        first_non_zero = jnp.argmax(non_zero_mask)
        start_index = jax.lax.select(has_non_zero, first_non_zero, self.score_max_digits - 1)
        num_to_render = jax.lax.select(has_non_zero, self.score_max_digits - start_index, 1)

        total_width = num_to_render * self.score_digit_spacing
        score_x = self.score_center_x - (total_width // 2)

        raster_score = self.jr.render_label_selective(
            raster_flags_top,
            jnp.int32(score_x),
            self.score_render_y,
            score_digits,
            self.score_digit_masks,
            start_index,
            num_to_render,
            spacing=self.score_digit_spacing,
            max_digits_to_render=self.score_max_digits,
        )

        # Render flags on the road
        flag_pole_mask = self.SHAPE_MASKS["flag_pole"]
        
        def render_flag(carry, flag_idx):
            raster = carry
            flag_y = state.flags.y[flag_idx]
            flag_road = state.flags.road[flag_idx]
            flag_segment = state.flags.road_segment[flag_idx]
            flag_collected = state.flags.collected[flag_idx]
            flag_color_idx = state.flags.color_idx[flag_idx]
            
            flag_x = jax.lax.cond(
                flag_road == 0,
                lambda _: self._get_x_on_road(flag_y, flag_segment, self.consts.FIRST_TRACK_CORNERS_X),
                lambda _: self._get_x_on_road(flag_y, flag_segment, self.consts.SECOND_TRACK_CORNERS_X),
                operand=None,
            )
            screen_y = 105 + (flag_y - state.player_car.position.y)
            # Hide flags when awaiting round start or awaiting respawn
            should_hide = jnp.logical_or(state.awaiting_round_start, state.awaiting_respawn)
            is_visible = jnp.logical_and(
                jnp.logical_and(screen_y > 25, screen_y < 195),
                jnp.logical_and(~flag_collected, ~should_hide)
            )
            color_id = self.flag_palette_ids[flag_color_idx]
            colored_flag_mask = jnp.where(
                self.flag_solid_mask,
                color_id,
                self.flag_base_mask,
            )
            raster = jax.lax.cond(
                is_visible,
                lambda r: self.jr.render_at(
                    self.jr.render_at(r, flag_x.astype(jnp.int32), screen_y.astype(jnp.int32), colored_flag_mask),
                    (flag_x + 5).astype(jnp.int32), screen_y.astype(jnp.int32), flag_pole_mask
                ),
                lambda r: r,
                operand=raster,
            )
            return raster, None
        
        raster_flags, _ = jax.lax.scan(render_flag, raster_score, jnp.arange(self.consts.NUM_FLAGS))
        
        blackout_mask = self.SHAPE_MASKS["blackout_square"]
        
        def render_blackout(carry, flag_idx):
            raster = carry
            flag_collected = state.flags_collected_mask[flag_idx]
            blackout_x = self.consts.FLAG_TOP_X_POSITIONS[flag_idx]
            blackout_y = self.consts.FLAG_TOP_Y
            raster = jax.lax.cond(
                flag_collected,
                lambda r: self.jr.render_at(r, blackout_x, blackout_y, blackout_mask),
                lambda r: r,
                operand=raster,
            )
            return raster, None
        
        raster_blackout, _ = jax.lax.scan(render_blackout, raster_flags, jnp.arange(self.consts.NUM_FLAGS))

        def render_collectible(carry, collectible_idx):
            raster = carry
            collectible_y = state.collectibles.y[collectible_idx]
            collectible_x = state.collectibles.x[collectible_idx]
            collectible_active = state.collectibles.active[collectible_idx]
            collectible_color_idx = state.collectibles.color_idx[collectible_idx]
            collectible_type_id = state.collectibles.type_id[collectible_idx]
            screen_y = 105 + (collectible_y - state.player_car.position.y)
            # Hide collectibles when awaiting round start or awaiting respawn
            should_hide = jnp.logical_or(state.awaiting_round_start, state.awaiting_respawn)
            is_visible = jnp.logical_and(
                jnp.logical_and(screen_y > 25, screen_y < 195),
                jnp.logical_and(collectible_active, ~should_hide)
            )

            def get_sprite_and_mask(type_id):
                # Use switch for O(1) lookup instead of nested conditionals
                def get_cherry(_):
                    return (self.cherry_base_mask, self.cherry_solid_mask, self.collectible_palette_ids)
                def get_balloon(_):
                    return (self.balloon_base_mask, self.balloon_solid_mask, self.collectible_palette_ids)
                def get_lollypop(_):
                    return (self.lollypop_base_mask, self.lollypop_solid_mask, self.collectible_palette_ids)
                def get_ice_cream(_):
                    return (self.ice_cream_base_mask, self.ice_cream_solid_mask, self.collectible_palette_ids)
                
                return jax.lax.switch(
                    type_id,
                    [get_cherry, get_balloon, get_lollypop, get_ice_cream],
                    None,
                )

            base_mask, solid_mask, palette_ids = get_sprite_and_mask(collectible_type_id)
            color_id = palette_ids[collectible_color_idx]
            colored_mask = jnp.where(
                (base_mask != self.jr.TRANSPARENT_ID) & (base_mask != 0),
                color_id,

                base_mask,
            )
            raster = jax.lax.cond(
                is_visible,
                lambda r: self.jr.render_at(r, collectible_x.astype(jnp.int32), screen_y.astype(jnp.int32), colored_mask),
                lambda r: r,
                operand=raster,
            )
            return raster, None

        raster_collectibles, _ = jax.lax.scan(render_collectible, raster_blackout, jnp.arange(self.consts.MAX_COLLECTIBLES))

        all_lives_bottom_mask = self.SHAPE_MASKS["all_lives_bottom"]
        raster_lives = self.jr.render_at(raster_collectibles, 10, 195, all_lives_bottom_mask)

        # Black out lost lives (similar to flag blackout)
        blackout_mask = self.SHAPE_MASKS["blackout_square"]
        lives_lost = self.consts.INITIAL_LIVES - state.lives
        
        def render_life_blackout(carry, life_idx):
            raster = carry
            # Black out this life if it has been lost (life_idx < lives_lost)
            should_blackout = life_idx < lives_lost
            blackout_x = self.consts.LIFE_BOTTOM_X_POSITIONS[life_idx]
            blackout_y = self.consts.LIFE_BOTTOM_Y
            raster = jax.lax.cond(
                should_blackout,
                lambda r: self.jr.render_at(r, blackout_x, blackout_y, blackout_mask),
                lambda r: r,
                operand=raster,
            )
            return raster, None
        
        raster_lives_blackout, _ = jax.lax.scan(render_life_blackout, raster_lives, jnp.arange(self.consts.INITIAL_LIVES))

        wall_bottom_mask = self.SHAPE_MASKS["tempPointer"]
        raster_pointer = self.jr.render_at(raster_lives_blackout, 140, 25, wall_bottom_mask)

        return self.jr.render_from_palette(raster_pointer, self.PALETTE)