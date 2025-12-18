from jax._src.pjit import JitWrapped
import os
import math
from functools import partial
from typing import NamedTuple, Tuple
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
    ACTION_REPEAT_PROBS: float = 0.25
    MAX_SPEED: int = 5
    # Enemy spawning and movement
    MAX_ENEMY_CARS: int = 6
    ENEMY_SPAWN_INTERVAL: int = 80
    ENEMY_DESPAWN_DISTANCE: int = 300
    ENEMY_SPEED_MIN: int = 2
    ENEMY_SPEED_MAX: int = 5
    ENEMY_DIRECTION_SWITCH_PROB: float = 0.005
    ENEMY_OFFSCREEN_SPAWN_OFFSET: float = 100.0
    ENEMY_MIN_SPAWN_GAP: float = 40.0
    ENEMY_MAX_AGE: int = 900
    INITIAL_ENEMY_COUNT: int = 3
    INITIAL_ENEMY_BASE_OFFSET: float = 40.0
    INITIAL_ENEMY_GAP: float = 50.0
    ENEMY_TYPE_CAMERO: int = 0
    ENEMY_TYPE_FLAG_CARRIER: int = 1
    ENEMY_TYPE_PICKUP: int = 2
    ENEMY_TYPE_TRUCK: int = 3
    JUMP_FRAMES: int = 10
    LANDING_ZONE: int = 15
    FIRST_ROAD_LENGTH: int = 4
    SECOND_ROAD_LENGTH: int = 4
    FIRST_TRACK_CORNERS_X: chex.Array = jnp.array([30, 75, 128, 75, 21, 75, 131, 111, 150, 95, 150, 115, 150, 108, 150, 115, 115, 75, 18, 38, 67, 38, 38, 20, 64, 30]) 
    TRACK_CORNERS_Y: chex.Array = jnp.array([0, -40, -98, -155, -203, -268, -327, -347, -382, -467, -525, -565, -597, -625, -670, -705, -738, -788, -838, -862, -898, -925, -950, -972, -1000, -1036])
    SECOND_TRACK_CORNERS_X: chex.Array = jnp.array([115, 75, 20, 75, 133, 75, 22, 37, 63, 27, 66, 30, 63, 24, 60, 38, 38, 75, 131, 111, 150, 118, 118, 98, 150, 115]) 
    PLAYER_SIZE: Tuple[int, int] = (4, 16)
    INITIAL_ROAD_POS_Y: int = 25
    # Flag constants - 8 flags with different colors matching the top row
    NUM_FLAGS: int = 8
    FLAG_SIZE: Tuple[int, int] = (11, 6)  # height, width of the flag sprite
    FLAG_POLE_SIZE: Tuple[int, int] = (7, 2)  # height, width of the pole sprite
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
    FLAG_TOP_X_POSITIONS: chex.Array = jnp.array([13, 30, 47, 64, 82, 98, 118, 132])
    FLAG_TOP_Y: int = 20
    FLAG_BLACKOUT_SIZE: Tuple[int, int] = (14, 14)  # Size of blackout square
    FLAG_COLLECTION_SCORE: int = 75  # Points awarded for collecting a flag
    PICKUP_SCORE: int = 100  # Points awarded for jumping on a pickup truck
    FLAG_CARRIER_SCORE: int = 125  # Points awarded for jumping on a flag carrier
    CAMARO_SCORE: int = 150  # Points awarded for jumping on a camaro
    TRUCK_SCORE: int = 175  # Points awarded for jumping on a truck
    # Collectible constants - unified dynamic spawning
    MAX_COLLECTIBLES: int = 2  # Maximum collectibles that can exist at once (pool of mixed types)
    COLLECTIBLE_SIZE: Tuple[int, int] = (8, 8)  # height, width of collectible sprite
    COLLECTIBLE_SPAWN_INTERVAL: int = 200  # Steps between spawn attempts
    COLLECTIBLE_DESPAWN_DISTANCE: int = 500  # Distance beyond which collectibles despawn
    # Collectible types (indices for type field)
    COLLECTIBLE_TYPE_CHERRY: int = 0
    COLLECTIBLE_TYPE_BALLOON: int = 1
    COLLECTIBLE_TYPE_LOLLYPOP: int = 2
    COLLECTIBLE_TYPE_ICE_CREAM: int = 3
    # Collectible type spawn probabilities (must sum to 100)
    COLLECTIBLE_SPAWN_PROBABILITIES: chex.Array = jnp.array([40, 20, 20, 20], dtype=jnp.int32)  # Cherry: 40%, Balloon: 20%, Lollypop: 20%, IceCream: 20%
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
    is_jumping: chex.Array
    is_on_road: chex.Array
    player_car: Car
    step_counter: chex.Array
    round_started: chex.Array
    movement_steps: chex.Array
    # Flag state - tracks all 8 flags
    flags: Flag  # Contains arrays of size NUM_FLAGS for each field
    flags_collected_mask: chex.Array  # Boolean mask of which flag colors have been collected (size NUM_FLAGS)
    # Collectible state - dynamic spawning (mixed types: cherry, balloon, lollypop, ice cream)
    collectibles: Collectible  # Contains arrays of size MAX_COLLECTIBLES for each field
    collectible_spawn_timer: chex.Array  # Counter for collectible spawn timing
    # Enemy cars - dynamic spawning and movement
    enemy_cars: EnemyCars
    enemy_spawn_timer: chex.Array




class UpNDownObservation(NamedTuple):
    player: EntityPosition

class UpNDownInfo(NamedTuple):
    time: jnp.ndarray


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
        self.obs_size = 3*4+1+1

    @partial(jax.jit, static_argnums=(0,))
    def _get_slope_and_intercept_from_indices(self, current_road: chex.Array, road_index_A: chex.Array, road_index_B: chex.Array) -> Tuple[chex.Array, chex.Array]:
        trackx, tracky, road_index = jax.lax.cond(
            current_road == 0,
            lambda _: (self.consts.FIRST_TRACK_CORNERS_X, self.consts.TRACK_CORNERS_Y, road_index_A),
            lambda _: (self.consts.SECOND_TRACK_CORNERS_X, self.consts.TRACK_CORNERS_Y, road_index_B),
            operand=None,
        )
        slope = jax.lax.cond(
            trackx[road_index+1] - trackx[road_index] != 0,
            lambda _: (tracky[road_index+1] - tracky[road_index]) / (trackx[road_index+1] - trackx[road_index]),
            lambda _: 300.0,
            operand=None,
        )
        b = tracky[road_index] - slope * trackx[road_index]
        return slope, b

    @partial(jax.jit, static_argnums=(0,))
    def _getSlopeAndB(self, state: UpNDownState) -> chex.Array:
        return self._get_slope_and_intercept_from_indices(
            state.player_car.current_road,
            state.player_car.road_index_A,
            state.player_car.road_index_B,
        )

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
        t = jax.lax.cond(
            y2 != y1,
            lambda _: (y - y1) / (y2 - y1),
            lambda _: 0.0,
            operand=None,
        )
        return x1 + t * (x2 - x1)
    
    @partial(jax.jit, static_argnums=(0,))
    def _isOnLine(self, state: UpNDownState,  player_speed: chex.Array, turn: chex.Array) -> chex.Array:
        slope, b = self._getSlopeAndB(state)
        return self._is_on_line_for_position(state.player_car.position, slope, b, player_speed, turn)
    
    @partial(jax.jit, static_argnums=(0,))
    def _landing_in_water(self, state: UpNDownState, new_position_x: chex.Array, new_position_y: chex.Array) -> chex.Array:
        road_A_x = ((new_position_y - self.consts.TRACK_CORNERS_Y[state.player_car.road_index_A]) / (self.consts.TRACK_CORNERS_Y[state.player_car.road_index_A+1] - self.consts.TRACK_CORNERS_Y[state.player_car.road_index_A])) * (self.consts.FIRST_TRACK_CORNERS_X[state.player_car.road_index_A+1] - self.consts.FIRST_TRACK_CORNERS_X[state.player_car.road_index_A]) + self.consts.FIRST_TRACK_CORNERS_X[state.player_car.road_index_A]
        road_B_x = ((new_position_y - self.consts.TRACK_CORNERS_Y[state.player_car.road_index_B]) / (self.consts.TRACK_CORNERS_Y[state.player_car.road_index_B+1] - self.consts.TRACK_CORNERS_Y[state.player_car.road_index_B])) * (self.consts.SECOND_TRACK_CORNERS_X[state.player_car.road_index_B+1] - self.consts.SECOND_TRACK_CORNERS_X[state.player_car.road_index_B]) + self.consts.SECOND_TRACK_CORNERS_X[state.player_car.road_index_B]
        distance_to_road_A = jnp.abs(new_position_x - road_A_x)
        distance_to_road_B = jnp.abs(new_position_x - road_B_x)
        landing_in_Water = jnp.logical_and(distance_to_road_A > self.consts.LANDING_ZONE, distance_to_road_B > self.consts.LANDING_ZONE)
        between_roads = jnp.logical_and(new_position_x > jnp.minimum(road_A_x, road_B_x), new_position_x < jnp.maximum(road_A_x, road_B_x))
        return landing_in_Water, between_roads, road_A_x, road_B_x

    @partial(jax.jit, static_argnums=(0,))
    def _landing_in_water_for_indices(self, road_index_A: chex.Array, road_index_B: chex.Array, new_position_x: chex.Array, new_position_y: chex.Array) -> chex.Array:
        road_A_x = ((new_position_y - self.consts.TRACK_CORNERS_Y[road_index_A]) / (self.consts.TRACK_CORNERS_Y[road_index_A+1] - self.consts.TRACK_CORNERS_Y[road_index_A])) * (self.consts.FIRST_TRACK_CORNERS_X[road_index_A+1] - self.consts.FIRST_TRACK_CORNERS_X[road_index_A]) + self.consts.FIRST_TRACK_CORNERS_X[road_index_A]
        road_B_x = ((new_position_y - self.consts.TRACK_CORNERS_Y[road_index_B]) / (self.consts.TRACK_CORNERS_Y[road_index_B+1] - self.consts.TRACK_CORNERS_Y[road_index_B])) * (self.consts.SECOND_TRACK_CORNERS_X[road_index_B+1] - self.consts.SECOND_TRACK_CORNERS_X[road_index_B]) + self.consts.SECOND_TRACK_CORNERS_X[road_index_B]
        distance_to_road_A = jnp.abs(new_position_x - road_A_x)
        distance_to_road_B = jnp.abs(new_position_x - road_B_x)
        landing_in_water = jnp.logical_and(distance_to_road_A > self.consts.LANDING_ZONE, distance_to_road_B > self.consts.LANDING_ZONE)
        between_roads = jnp.logical_and(new_position_x > jnp.minimum(road_A_x, road_B_x), new_position_x < jnp.maximum(road_A_x, road_B_x))
        return landing_in_water, between_roads, road_A_x, road_B_x

    @partial(jax.jit, static_argnums=(0,))
    def _advance_car_core(
        self,
        position_x: chex.Array,
        position_y: chex.Array,
        road_index_A: chex.Array,
        road_index_B: chex.Array,
        current_road: chex.Array,
        speed: chex.Array,
        is_jumping: chex.Array,
        is_on_road: chex.Array,
        step_counter: chex.Array,
        width: chex.Array,
        height: chex.Array,
        car_type: chex.Array,
        landing_check: chex.Array,
    ) -> Car:
        dividers = jnp.array([0, 1, 2, 4, 8, 16])
        abs_speed = jnp.abs(speed)
        speed_divider = dividers[abs_speed]
        effective_divider = jnp.maximum(1, speed_divider)
        period = jnp.maximum(1, 16 // effective_divider)
        half_period = jnp.maximum(1, period // 2)
        speed_sign = jax.lax.cond(speed != 0, lambda _: jax.lax.abs(speed) / speed, lambda _: jnp.array(0.0), operand=None)

        slope, b = self._get_slope_and_intercept_from_indices(current_road, road_index_A, road_index_B)

        direction_raw = jax.lax.cond(
            current_road == 0,
            lambda _: self.consts.FIRST_TRACK_CORNERS_X[road_index_A+1] - self.consts.FIRST_TRACK_CORNERS_X[road_index_A],
            lambda _: self.consts.SECOND_TRACK_CORNERS_X[road_index_B+1] - self.consts.SECOND_TRACK_CORNERS_X[road_index_B],
            operand=None,
        )
        car_direction_x = jax.lax.cond(direction_raw > 0, lambda _: 1, lambda _: -1, operand=None)

        move_y = jnp.logical_and((step_counter % period) == (half_period % period), speed != 0)
        move_x = jnp.logical_and((step_counter % period) == 0, speed != 0)

        position = EntityPosition(x=position_x, y=position_y, width=width, height=height)

        new_player_y = jax.lax.cond(
            move_y,
            lambda _: jax.lax.cond(
                is_jumping,
                lambda _: position_y + speed_sign * -1,
                lambda _: jax.lax.cond(
                    self._is_on_line_for_position(position, slope, b, speed_sign, 1),
                    lambda _: position_y + speed_sign * -1,
                    lambda _: jnp.array(position_y, float),
                    operand=None,
                ),
                operand=None,
            ),
            lambda _: jnp.array(position_y, float),
            operand=None,
        )

        new_player_x = jax.lax.cond(
            move_x,
            lambda _: jax.lax.cond(
                is_jumping,
                lambda _: position_x + speed_sign * car_direction_x,
                lambda _: jax.lax.cond(
                    self._is_on_line_for_position(position, slope, b, speed_sign, 2),
                    lambda _: position_x + speed_sign * car_direction_x,
                    lambda _: jnp.array(position_x, float),
                    operand=None,
                ),
                operand=None,
            ),
            lambda _: jnp.array(position_x, float),
            operand=None,
        )

        landing_in_water, between_roads, road_A_x, road_B_x = self._landing_in_water_for_indices(road_index_A, road_index_B, new_player_x, new_player_y)
        landing_in_water = jnp.logical_and(landing_check, landing_in_water)

        updated_current_road = jax.lax.cond(
            landing_in_water,
            lambda _: 2,
            lambda _: jax.lax.cond(
                is_on_road,
                lambda _: current_road,
                lambda _: jax.lax.cond(
                    jnp.abs(new_player_x - road_A_x) < jnp.abs(new_player_x - road_B_x),
                    lambda _: 0,
                    lambda _: 1,
                    operand=None,
                ),
                operand=None,
            ),
            operand=None,
        )

        next_road_index_A = jax.lax.cond(
            updated_current_road == 2,
            lambda _: road_index_A,
            lambda _: jax.lax.cond(
                self.consts.TRACK_CORNERS_Y[road_index_A] < new_player_y,
                lambda _: road_index_A - 1,
                lambda _: jax.lax.cond(
                    len(self.consts.TRACK_CORNERS_Y) == road_index_A + 1,
                    lambda _: jax.lax.cond(
                        self.consts.TRACK_CORNERS_Y[0] > new_player_y,
                        lambda _: 0,
                        lambda _: road_index_A,
                        operand=None,
                    ),
                    lambda _: jax.lax.cond(
                        self.consts.TRACK_CORNERS_Y[road_index_A+1] > new_player_y,
                        lambda _: road_index_A + 1,
                        lambda _: road_index_A,
                        operand=None,
                    ),
                    operand=None,
                ),
                operand=None,
            ),
            operand=None,
        )

        next_road_index_B = jax.lax.cond(
            updated_current_road == 2,
            lambda _: road_index_B,
            lambda _: jax.lax.cond(
                self.consts.TRACK_CORNERS_Y[road_index_B] < new_player_y,
                lambda _: road_index_B - 1,
                lambda _: jax.lax.cond(
                    len(self.consts.TRACK_CORNERS_Y) == road_index_B + 1,
                    lambda _: jax.lax.cond(
                        self.consts.TRACK_CORNERS_Y[0] > new_player_y,
                        lambda _: 0,
                        lambda _: road_index_B,
                        operand=None,
                    ),
                    lambda _: jax.lax.cond(
                        self.consts.TRACK_CORNERS_Y[road_index_B+1] > new_player_y,
                        lambda _: road_index_B + 1,
                        lambda _: road_index_B,
                        operand=None,
                    ),
                    operand=None,
                ),
                operand=None,
            ),
            operand=None,
        )

        wrapped_y = -((new_player_y * -1) % 1036)

        return Car(
            position=EntityPosition(
                x=new_player_x,
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

    def _flag_step(self, state: UpNDownState, new_player_y: chex.Array, player_x: chex.Array, current_road: chex.Array) -> Tuple[Flag, chex.Array, chex.Array]:
        """Update flag collection state and score.
        
        Args:
            state: Current game state
            new_player_y: Updated player Y position after movement
            player_x: Current player X position
            current_road: Current road player is on
            
        Returns:
            Tuple of (updated_flags, score_delta, flags_collected_mask)
        """
        # Check collision for each flag
        def check_flag_collision(flag_idx):
            flag_y = state.flags.y[flag_idx]
            flag_road = state.flags.road[flag_idx]
            flag_collected = state.flags.collected[flag_idx]
            
            # Calculate flag X position on its road
            flag_segment = state.flags.road_segment[flag_idx]
            flag_x = jax.lax.cond(
                flag_road == 0,
                lambda _: self._get_x_on_road(flag_y, flag_segment, self.consts.FIRST_TRACK_CORNERS_X),
                lambda _: self._get_x_on_road(flag_y, flag_segment, self.consts.SECOND_TRACK_CORNERS_X),
                operand=None,
            )
            
            # Check if player is close enough to collect the flag
            y_distance = jnp.abs(new_player_y - flag_y)
            x_distance = jnp.abs(player_x - flag_x)
            same_road = jnp.logical_or(
                jnp.logical_and(current_road == 0, flag_road == 0),
                jnp.logical_and(current_road == 1, flag_road == 1),
            )

            collision = jnp.logical_and(
                jnp.logical_and(y_distance < 5, x_distance < 5),   #change the distance threshold if needed
                jnp.logical_and(same_road, ~flag_collected)
            )
            return collision
        
        new_collections = jax.vmap(check_flag_collision)(jnp.arange(self.consts.NUM_FLAGS))
        
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
    
    def _collectible_step(self, state: UpNDownState, new_player_y: chex.Array, player_x: chex.Array, current_road: chex.Array) -> Tuple[Collectible, chex.Array, chex.Array]:
        """Update collectible spawning, despawning, and collection (unified for all types).
        
        Handles mixed-type collectibles (cherry, balloon, lollypop, ice cream) in a single pool.
        Type is randomized on spawn with probabilities defined in COLLECTIBLE_SPAWN_PROBABILITIES.
        
        Args:
            state: Current game state
            new_player_y: Updated player Y position after movement
            player_x: Current player X position
            current_road: Current road player is on
            
        Returns:
            Tuple of (updated_collectibles, score_delta, new_spawn_timer)
        """
        # Collectible spawning logic - decrement timer and spawn when ready
        new_collectible_timer = jax.lax.cond(
            state.collectible_spawn_timer <= 0,
            lambda _: self.consts.COLLECTIBLE_SPAWN_INTERVAL,
            lambda _: state.collectible_spawn_timer - 1,
            operand=None,
        )
        
        # Attempt to spawn when timer hits 0
        should_spawn = state.collectible_spawn_timer <= 0
        
        # Find first inactive collectible slot
        def find_inactive_idx(collectibles_in):
            inactive_mask = ~collectibles_in.active
            first_inactive = jnp.argmax(inactive_mask.astype(jnp.int32))
            has_inactive = jnp.any(inactive_mask)
            return jax.lax.cond(
                has_inactive,
                lambda _: first_inactive,
                lambda _: jnp.array(0, dtype=jnp.int32),
                operand=None,
            ), has_inactive
        
        spawn_idx, has_inactive_slot = find_inactive_idx(state.collectibles)
        
        # Generate random spawn position using fold_in for deterministic randomness
        base_key = jax.random.PRNGKey(0)
        key_for_spawn = jax.random.fold_in(base_key, state.step_counter)
        key1, key2, key3, key4, key5 = jax.random.split(key_for_spawn, 5)
        y_spawn = jax.random.uniform(key1, minval=-900.0, maxval=-100.0)
        road_spawn = jnp.array(jax.random.randint(key2, shape=(), minval=0, maxval=2), dtype=jnp.int32)
        color_spawn = jnp.array(jax.random.randint(key3, shape=(), minval=0, maxval=len(self.consts.COLLECTIBLE_COLORS)), dtype=jnp.int32)
        
        # Randomly select collectible type based on spawn probabilities
        # Convert probabilities (%) to cumulative distribution for sampling
        rand_type = jax.random.uniform(key4, minval=0.0, maxval=100.0)
        
        # Use cumulative probabilities: cherry [0-40], balloon [40-60], lollypop [60-80], ice_cream [80-100]
        def select_type(rand_val):
            # Returns 0=cherry, 1=balloon, 2=lollypop, 3=ice_cream
            type_id = jnp.where(
                rand_val < self.consts.COLLECTIBLE_SPAWN_PROBABILITIES[0],
                jnp.int32(self.consts.COLLECTIBLE_TYPE_CHERRY),
                jnp.where(
                    rand_val < self.consts.COLLECTIBLE_SPAWN_PROBABILITIES[1],
                    jnp.int32(self.consts.COLLECTIBLE_TYPE_BALLOON),
                    jnp.where(
                        rand_val < self.consts.COLLECTIBLE_SPAWN_PROBABILITIES[2],
                        jnp.int32(self.consts.COLLECTIBLE_TYPE_LOLLYPOP),
                        jnp.int32(self.consts.COLLECTIBLE_TYPE_ICE_CREAM)
                    )
                )
            )
            return type_id
        
        type_id_spawn = select_type(rand_type)
        
        # Calculate X position on road
        def get_road_segment(y):
            segments = jnp.sum(self.consts.TRACK_CORNERS_Y > y)
            return jnp.clip(segments - 1, 0, len(self.consts.TRACK_CORNERS_Y) - 2)
        
        segment_spawn = get_road_segment(y_spawn)
        x_spawn = jax.lax.cond(
            road_spawn == 0,
            lambda _: self._get_x_on_road(y_spawn, segment_spawn, self.consts.FIRST_TRACK_CORNERS_X),
            lambda _: self._get_x_on_road(y_spawn, segment_spawn, self.consts.SECOND_TRACK_CORNERS_X),
            operand=None,
        )
        
        # Create mask for which collectibles to update
        update_mask = (jnp.arange(self.consts.MAX_COLLECTIBLES) == spawn_idx) & should_spawn & has_inactive_slot
        
        # Update collectibles with proper masking
        updated_collectibles = Collectible(
            y=jnp.where(update_mask, y_spawn, state.collectibles.y),
            x=jnp.where(update_mask, x_spawn, state.collectibles.x),
            road=jnp.where(update_mask, road_spawn, state.collectibles.road),
            color_idx=jnp.where(update_mask, color_spawn, state.collectibles.color_idx),
            type_id=jnp.where(update_mask, type_id_spawn, state.collectibles.type_id),
            active=jnp.where(update_mask, True, state.collectibles.active),
        )
        
        # Despawn logic - remove collectibles too far from player
        def check_despawn(idx):
            c_y = updated_collectibles.y[idx]
            c_active = updated_collectibles.active[idx]
            distance = jnp.abs(new_player_y - c_y)
            too_far = distance > self.consts.COLLECTIBLE_DESPAWN_DISTANCE
            should_despawn = jnp.logical_and(c_active, too_far)
            return should_despawn
        
        despawn_mask = jax.vmap(check_despawn)(jnp.arange(self.consts.MAX_COLLECTIBLES))
        new_active = jnp.logical_and(updated_collectibles.active, ~despawn_mask)
        
        # Collision detection
        def check_collision(idx):
            c_y = updated_collectibles.y[idx]
            c_x = updated_collectibles.x[idx]
            c_road = updated_collectibles.road[idx]
            c_active = updated_collectibles.active[idx]
            
            y_distance = jnp.abs(new_player_y - c_y)
            x_distance = jnp.abs(player_x - c_x)
            same_road = jnp.logical_or(
                jnp.logical_and(current_road == 0, c_road == 0),
                jnp.logical_and(current_road == 1, c_road == 1),
            )
            
            collision = jnp.logical_and(
                jnp.logical_and(y_distance < 5, x_distance < 5),
                jnp.logical_and(same_road, c_active)
            )
            return collision
        
        collections = jax.vmap(check_collision)(jnp.arange(self.consts.MAX_COLLECTIBLES))
        
        # Deactivate collected items
        new_active = jnp.logical_and(new_active, ~collections)
        
        # Update score - use type_id to look up score value
        def get_collection_score(idx):
            is_collected = collections[idx]
            type_id = updated_collectibles.type_id[idx]
            # Look up score based on type_id using array indexing
            score = self.consts.COLLECTIBLE_SCORES[type_id]
            return jnp.where(is_collected, score, 0)
        
        score_array = jax.vmap(get_collection_score)(jnp.arange(self.consts.MAX_COLLECTIBLES))
        score_delta = jnp.sum(score_array)
        
        updated_collectibles = Collectible(
            y=updated_collectibles.y,
            x=updated_collectibles.x,
            road=updated_collectibles.road,
            color_idx=updated_collectibles.color_idx,
            type_id=updated_collectibles.type_id,
            active=new_active,
        )
        
        return updated_collectibles, score_delta, new_collectible_timer

    def _player_step(self, state: UpNDownState, action: chex.Array) -> UpNDownState:
        up = jnp.logical_or(action == Action.UP, action == Action.UPFIRE)
        down = jnp.logical_or(action == Action.DOWN, action == Action.DOWNFIRE)
        jump = jnp.logical_or(action == Action.FIRE, jnp.logical_or(action == Action.UPFIRE, action == Action.DOWNFIRE))
        player_speed = state.player_car.speed

        player_speed = jax.lax.cond(
            jnp.logical_and(state.player_car.speed < self.consts.MAX_SPEED, up),
            lambda s: s + 1,
            lambda s: s,
            operand=player_speed,
        )

        player_speed = jax.lax.cond(
            jnp.logical_and(state.player_car.speed > -self.consts.MAX_SPEED, down),
            lambda s: s - 1,
            lambda s: s,
            operand=player_speed,
        )

        is_jumping = jnp.logical_or(jnp.logical_and(state.is_jumping, state.jump_cooldown > 0), jnp.logical_and(state.is_on_road, jnp.logical_and(player_speed >= 0, jnp.logical_and(state.jump_cooldown == 0, jump))))
        jump_cooldown = jax.lax.cond(
            state.jump_cooldown > 0,
            lambda s: s - 1,
            lambda s: jax.lax.cond(is_jumping,
                               lambda _: self.consts.JUMP_FRAMES,
                               lambda _: 0, 
                               operand=None),
            operand=state.jump_cooldown,
        )
        is_on_road = ~is_jumping
        is_landing = jnp.logical_and(state.jump_cooldown == 1, jump_cooldown == 0)

        updated_player_car = self._advance_car_core(
            position_x=state.player_car.position.x,
            position_y=state.player_car.position.y,
            road_index_A=state.player_car.road_index_A,
            road_index_B=state.player_car.road_index_B,
            current_road=state.player_car.current_road,
            speed=player_speed,
            is_jumping=is_jumping,
            is_on_road=is_on_road,
            step_counter=state.step_counter,
            width=state.player_car.position.width,
            height=state.player_car.position.height,
            car_type=state.player_car.type,
            landing_check=is_landing,
        )

        return UpNDownState(
            score=state.score,
            difficulty=state.difficulty,
            jump_cooldown=jump_cooldown,
            is_jumping=is_jumping,
            is_on_road=is_on_road,
            player_car=updated_player_car,
            step_counter=state.step_counter + 1,
            round_started=jnp.logical_or(state.round_started, player_speed != 0),
            movement_steps=jax.lax.cond(
                jnp.logical_or(state.round_started, player_speed != 0),
                lambda s: state.movement_steps + 1,
                lambda s: state.movement_steps,
                operand=None,
            ),
            flags=state.flags,
            flags_collected_mask=state.flags_collected_mask,
            collectibles=state.collectibles,
            collectible_spawn_timer=state.collectible_spawn_timer,
            enemy_cars=state.enemy_cars,
            enemy_spawn_timer=state.enemy_spawn_timer,
        )

    def _flag_step_main(self, state: UpNDownState) -> UpNDownState:
        """Update flag collection state and score."""
        new_player_y = state.player_car.position.y
        player_x = state.player_car.position.x
        current_road = state.player_car.current_road
        
        new_flags, flag_score, new_flags_collected_mask = self._flag_step(
            state, new_player_y, player_x, current_road
        )
        
        return UpNDownState(
            score=state.score + flag_score,
            difficulty=state.difficulty,
            jump_cooldown=state.jump_cooldown,
            is_jumping=state.is_jumping,
            is_on_road=state.is_on_road,
            player_car=state.player_car,
            step_counter=state.step_counter,
            round_started=state.round_started,
            movement_steps=state.movement_steps,
            flags=new_flags,
            flags_collected_mask=new_flags_collected_mask,
            collectibles=state.collectibles,
            collectible_spawn_timer=state.collectible_spawn_timer,
            enemy_cars=state.enemy_cars,
            enemy_spawn_timer=state.enemy_spawn_timer,
        )
    
    def _collectible_step_main(self, state: UpNDownState) -> UpNDownState:
        """Update collectible spawning, despawning, and collection."""
        new_player_y = state.player_car.position.y
        player_x = state.player_car.position.x
        current_road = state.player_car.current_road
        
        updated_collectibles, collectible_score, new_collectible_timer = self._collectible_step(
            state, new_player_y, player_x, current_road
        )
        
        return UpNDownState(
            score=state.score + collectible_score,
            difficulty=state.difficulty,
            jump_cooldown=state.jump_cooldown,
            is_jumping=state.is_jumping,
            is_on_road=state.is_on_road,
            player_car=state.player_car,
            step_counter=state.step_counter,
            round_started=state.round_started,
            movement_steps=state.movement_steps,
            flags=state.flags,
            flags_collected_mask=state.flags_collected_mask,
            collectibles=updated_collectibles,
            collectible_spawn_timer=new_collectible_timer,
            enemy_cars=state.enemy_cars,
            enemy_spawn_timer=state.enemy_spawn_timer,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _enemy_step_main(self, state: UpNDownState) -> UpNDownState:
        """Spawn and move enemy cars that share the player's road logic."""
        base_key = jax.random.PRNGKey(2025)
        step_key = jax.random.fold_in(base_key, state.step_counter)
        key_spawn_offset, key_spawn_side, key_spawn_speed, key_spawn_direction, key_spawn_type, key_spawn_sign, key_flip_root = jax.random.split(step_key, 7)

        active_mask = state.enemy_cars.active
        active_count = jnp.sum(active_mask.astype(jnp.int32))
        can_spawn = active_count < self.consts.MAX_ENEMY_CARS

        spawn_timer = jax.lax.cond(
            jnp.logical_and(state.enemy_spawn_timer <= 0, can_spawn),
            lambda _: self.consts.ENEMY_SPAWN_INTERVAL,
            lambda _: state.enemy_spawn_timer - 1,
            operand=None,
        )
        should_spawn = jnp.logical_and(state.enemy_spawn_timer <= 0, can_spawn)

        inactive_mask = jnp.logical_not(active_mask)
        first_inactive = jnp.argmax(inactive_mask.astype(jnp.int32))
        has_inactive = jnp.any(inactive_mask)
        spawn_idx = jax.lax.cond(has_inactive, lambda _: first_inactive, lambda _: jnp.array(0, dtype=jnp.int32), operand=None)
        spawn_mask = (jnp.arange(self.consts.MAX_ENEMY_CARS) == spawn_idx) & should_spawn & has_inactive

        spawn_offset = self.consts.ENEMY_OFFSCREEN_SPAWN_OFFSET + active_count * self.consts.ENEMY_MIN_SPAWN_GAP + jax.random.uniform(key_spawn_offset, minval=0.0, maxval=40.0)
        spawn_side = jax.random.choice(key_spawn_side, jnp.array([-1.0, 1.0]))
        raw_spawn_y = state.player_car.position.y + spawn_side * spawn_offset
        spawn_y = -(((raw_spawn_y) * -1) % 1036)
        spawn_road = jax.random.randint(key_spawn_direction, shape=(), minval=0, maxval=2)

        def get_road_segment(y):
            segments = jnp.sum(self.consts.TRACK_CORNERS_Y > y)
            return jnp.clip(segments - 1, 0, len(self.consts.TRACK_CORNERS_Y) - 2)

        segment_spawn = get_road_segment(spawn_y)
        spawn_x = jax.lax.cond(
            spawn_road == 0,
            lambda _: self._get_x_on_road(spawn_y, segment_spawn, self.consts.FIRST_TRACK_CORNERS_X),
            lambda _: self._get_x_on_road(spawn_y, segment_spawn, self.consts.SECOND_TRACK_CORNERS_X),
            operand=None,
        )

        spawn_speed_mag = jax.random.randint(key_spawn_speed, shape=(), minval=self.consts.ENEMY_SPEED_MIN, maxval=self.consts.ENEMY_SPEED_MAX + 1)
        spawn_speed_sign = jax.random.choice(key_spawn_sign, jnp.array([-1, 1]))
        spawn_speed = spawn_speed_mag * spawn_speed_sign
        spawn_type = jax.random.randint(key_spawn_type, shape=(), minval=0, maxval=4)

        direction_raw = jax.lax.cond(
            spawn_road == 0,
            lambda _: self.consts.FIRST_TRACK_CORNERS_X[segment_spawn+1] - self.consts.FIRST_TRACK_CORNERS_X[segment_spawn],
            lambda _: self.consts.SECOND_TRACK_CORNERS_X[segment_spawn+1] - self.consts.SECOND_TRACK_CORNERS_X[segment_spawn],
            operand=None,
        )
        spawn_direction_x = jax.lax.cond(direction_raw > 0, lambda _: 1, lambda _: -1, operand=None)

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
            is_jumping=False,
            is_on_road=True,
            step_counter=state.step_counter,
            width=self.consts.PLAYER_SIZE[0],
            height=self.consts.PLAYER_SIZE[1],
            car_type=tp,
            landing_check=False,
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
        wrapped_dist = jnp.minimum(jnp.abs(delta_y), 1036 - jnp.abs(delta_y))
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

        return UpNDownState(
            score=state.score,
            difficulty=state.difficulty,
            jump_cooldown=state.jump_cooldown,
            is_jumping=state.is_jumping,
            is_on_road=state.is_on_road,
            player_car=state.player_car,
            step_counter=state.step_counter,
            round_started=state.round_started,
            movement_steps=state.movement_steps,
            flags=state.flags,
            flags_collected_mask=state.flags_collected_mask,
            collectibles=state.collectibles,
            collectible_spawn_timer=state.collectible_spawn_timer,
            enemy_cars=next_enemy_cars,
            enemy_spawn_timer=spawn_timer,
        )

    def _passive_score_step_main(self, state: UpNDownState) -> UpNDownState:
        """Award passive score every 60 steps after the player has started moving."""
        bonus = jax.lax.cond(
            jnp.logical_and(state.round_started, state.movement_steps % 60 == 0),
            lambda _: jnp.int32(10),
            lambda _: jnp.int32(0),
            operand=None,
        )

        return UpNDownState(
            score=state.score + bonus,
            difficulty=state.difficulty,
            jump_cooldown=state.jump_cooldown,
            is_jumping=state.is_jumping,
            is_on_road=state.is_on_road,
            player_car=state.player_car,
            step_counter=state.step_counter,
            round_started=state.round_started,
            movement_steps=state.movement_steps,
            flags=state.flags,
            flags_collected_mask=state.flags_collected_mask,
            collectibles=state.collectibles,
            collectible_spawn_timer=state.collectible_spawn_timer,
            enemy_cars=state.enemy_cars,
            enemy_spawn_timer=state.enemy_spawn_timer,
        )


    def reset(self, key=None) -> Tuple[UpNDownObservation, UpNDownState]:
        # Initialize flags at random positions along the track
        # Use key for randomness if provided, otherwise use default positions
        if key is None:
            key = jax.random.PRNGKey(42)
        
        # Evenly spread flags along the track with small jitter
        key, subkey = jax.random.split(key)
        base_y = jnp.linspace(-900.0, -100.0, self.consts.NUM_FLAGS)
        jitter = jax.random.uniform(subkey, shape=(self.consts.NUM_FLAGS,), minval=-40.0, maxval=40.0)
        flag_y_offsets = base_y + jitter

        # Alternate roads 0/1 for variety
        flag_roads = jnp.arange(self.consts.NUM_FLAGS) % 2
        
        # Calculate which road segment each flag is on based on Y position
        def get_road_segment(y):
            # Find the segment where TRACK_CORNERS_Y[i] > y >= TRACK_CORNERS_Y[i+1]
            segments = jnp.sum(self.consts.TRACK_CORNERS_Y > y)
            return jnp.clip(segments - 1, 0, len(self.consts.TRACK_CORNERS_Y) - 2)
        
        flag_segments = jax.vmap(get_road_segment)(flag_y_offsets)
        
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
        collectibles = Collectible(
            y=jnp.zeros(self.consts.MAX_COLLECTIBLES),
            x=jnp.zeros(self.consts.MAX_COLLECTIBLES),
            road=jnp.zeros(self.consts.MAX_COLLECTIBLES, dtype=jnp.int32),
            color_idx=jnp.zeros(self.consts.MAX_COLLECTIBLES, dtype=jnp.int32),
            type_id=jnp.zeros(self.consts.MAX_COLLECTIBLES, dtype=jnp.int32),
            active=jnp.zeros(self.consts.MAX_COLLECTIBLES, dtype=jnp.bool_),
        )

        def get_road_segment(y):
            segments = jnp.sum(self.consts.TRACK_CORNERS_Y > y)
            return jnp.clip(segments - 1, 0, len(self.consts.TRACK_CORNERS_Y) - 2)

        # Seed initial visible enemies spaced around the player
        key_init, key_type, key_road, key_speed, key_sign = jax.random.split(key, 5)
        player_start_y = 0.0
        offsets = self.consts.INITIAL_ENEMY_BASE_OFFSET + self.consts.INITIAL_ENEMY_GAP * jnp.arange(self.consts.INITIAL_ENEMY_COUNT)
        spawn_signs = jax.random.choice(key_sign, jnp.array([-1.0, 1.0]), shape=(self.consts.INITIAL_ENEMY_COUNT,))
        raw_spawn_y = player_start_y + spawn_signs * offsets
        init_y = -(((raw_spawn_y) * -1) % 1036)
        init_road = jax.random.randint(key_road, shape=(self.consts.INITIAL_ENEMY_COUNT,), minval=0, maxval=2)
        init_segments = jax.vmap(get_road_segment)(init_y)
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
        enemy_cars = EnemyCars(
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
        
        state = UpNDownState(
            score=0,
            difficulty=self.consts.DIFFICULTIES[0],
            jump_cooldown=0,
            is_jumping=False,
            is_on_road=True,
            player_car=Car(
                position=EntityPosition(
                    x=30,
                    y= 0,
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
            round_started=jnp.array(False),
            movement_steps=jnp.array(0),
            flags=flags,
            flags_collected_mask=jnp.zeros(self.consts.NUM_FLAGS, dtype=jnp.bool_),
            collectibles=collectibles,
            collectible_spawn_timer=jnp.array(0, dtype=jnp.int32),
            enemy_cars=enemy_cars,
            enemy_spawn_timer=jnp.array(self.consts.ENEMY_SPAWN_INTERVAL, dtype=jnp.int32),
        )
        initial_obs = self._get_observation(state)
        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: UpNDownState, action: chex.Array) -> Tuple[UpNDownObservation, UpNDownState, float, bool, UpNDownInfo]:
        previous_state = state
        state = self._player_step(state, action)
        state = self._passive_score_step_main(state)
        state = self._flag_step_main(state)
        state = self._collectible_step_main(state)
        state = self._enemy_step_main(state)

        done = self._get_done(state)
        env_reward = self._get_reward(previous_state, state)
        info = self._get_info(state)
        observation = self._get_observation(state)

        return observation, state, env_reward, done, info


    def render(self, state: UpNDownState) -> jnp.ndarray:
        return self.renderer.render(state)

    def _get_observation(self, state: UpNDownState):
        player = EntityPosition(
            x=jnp.array(state.player_car.position.x),
            y=jnp.array(state.player_car.position.y),
            width=jnp.array(self.consts.PLAYER_SIZE[0]),
            height=jnp.array(self.consts.PLAYER_SIZE[1]),
        )
        return UpNDownObservation(
            player=player,
        )

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: UpNDownObservation) -> jnp.ndarray:
           return jnp.concatenate([
               obs.player.x.flatten(),
               obs.player.y.flatten(),
               obs.player.height.flatten(),
               obs.player.width.flatten(),
            ]
           )

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(6)

    def observation_space(self) -> spaces:
        return spaces.Dict({
            "player": spaces.Dict({
                "x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
            }),
        })

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(210, 160, 3),
            dtype=jnp.uint8
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: UpNDownState, ) -> UpNDownInfo:
        return UpNDownInfo(time=1)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: UpNDownState, state: UpNDownState):
        return state.score

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: UpNDownState) -> bool:
        return jnp.logical_not(True)

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
        
        # 2. Update asset config to include both walls
        asset_config, road_files = self._get_asset_config(background, top_block, bottom_block, temp_pointer, blackout_square)
        sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/up_n_down/"

        # 3. Make a single call to the setup function
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
        repeats = max(1, math.ceil(self.view_height / road_cycle) + 2)
        self._road_tile_offsets = jnp.arange(-repeats, repeats + 1, dtype=jnp.int32) * jnp.int32(self.complete_road_size)
        self._num_road_tiles = int(self._road_tile_offsets.shape[0])

        self.enemy_sprite_names = {
            self.consts.ENEMY_TYPE_CAMERO: ("camero_left", "camero_right"),
            self.consts.ENEMY_TYPE_FLAG_CARRIER: ("flag_carrier_left", "flag_carrier_right"),
            self.consts.ENEMY_TYPE_PICKUP: ("pick_up_truck_left", "pick_up_truck_right"),
            self.consts.ENEMY_TYPE_TRUCK: ("truck_left", "truck_right"),
        }

        # Pre-pad enemy masks to a common shape so switch/array indexing works under jit
        enemy_left_raw = [
            self.SHAPE_MASKS["camero_left"],
            self.SHAPE_MASKS["flag_carrier_left"],
            self.SHAPE_MASKS["pick_up_truck_left"],
            self.SHAPE_MASKS["truck_left"],
        ]
        enemy_right_raw = [
            self.SHAPE_MASKS["camero_right"],
            self.SHAPE_MASKS["flag_carrier_right"],
            self.SHAPE_MASKS["pick_up_truck_right"],
            self.SHAPE_MASKS["truck_right"],
        ]
        max_h = max([m.shape[0] for m in enemy_left_raw + enemy_right_raw])
        max_w = max([m.shape[1] for m in enemy_left_raw + enemy_right_raw])

        def _pad_mask(mask):
            pad_h = max_h - mask.shape[0]
            pad_w = max_w - mask.shape[1]
            return jnp.pad(mask, ((0, pad_h), (0, pad_w)), constant_values=self.jr.TRANSPARENT_ID)

        self.enemy_left_masks = jnp.stack([_pad_mask(m) for m in enemy_left_raw], axis=0)
        self.enemy_right_masks = jnp.stack([_pad_mask(m) for m in enemy_right_raw], axis=0)
        
        # Precompute flag mask data for recoloring without special-casing pink
        self.flag_base_mask = self.SHAPE_MASKS["pink_flag"]
        self.flag_solid_mask = self.flag_base_mask != self.jr.TRANSPARENT_ID
        self.flag_palette_ids = self._compute_flag_palette_ids()
        
        # Precompute collectible mask data for recoloring (unified for all types: cherry, balloon, lollypop, ice cream)
        self.cherry_base_mask = self.SHAPE_MASKS["cherry"]
        self.cherry_solid_mask = self.cherry_base_mask != self.jr.TRANSPARENT_ID
        self.cherry_palette_ids = self._compute_flag_palette_ids()
        
        self.balloon_base_mask = self.SHAPE_MASKS["balloon"]
        self.balloon_solid_mask = self.balloon_base_mask != self.jr.TRANSPARENT_ID
        self.balloon_palette_ids = self._compute_flag_palette_ids()
        
        self.lollypop_base_mask = self.SHAPE_MASKS["lollypop"]
        self.lollypop_solid_mask = self.lollypop_base_mask != self.jr.TRANSPARENT_ID
        self.lollypop_palette_ids = self._compute_flag_palette_ids()
        
        self.ice_cream_base_mask = self.SHAPE_MASKS["ice_cream"]
        self.ice_cream_solid_mask = self.ice_cream_base_mask != self.jr.TRANSPARENT_ID
        self.ice_cream_palette_ids = self._compute_flag_palette_ids()

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
        jax.debug.print("Complete road size: {}", complete_size)
        return sizes, complete_size
    
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

    def _get_asset_config(self, backgroundSprite: jnp.ndarray, topBlockSprite: jnp.ndarray, bottomBlockSprite: jnp.ndarray, tempPointer: jnp.ndarray, blackoutSquare: jnp.ndarray) -> tuple[list, list[str]]:
        """Returns the asset manifest and ordered road files."""
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
            {'name': 'camero_right', 'type': 'single', 'file': 'enemy_cars/camero_right.npy'},
            {'name': 'flag_carrier_left', 'type': 'single', 'file': 'enemy_cars/flag_carrier_left.npy'},
            {'name': 'flag_carrier_right', 'type': 'single', 'file': 'enemy_cars/flag_carrier_right.npy'},
            {'name': 'pick_up_truck_left', 'type': 'single', 'file': 'enemy_cars/pick_up_truck_left.npy'},
            {'name': 'pick_up_truck_right', 'type': 'single', 'file': 'enemy_cars/pick_up_truck_right.npy'},
            {'name': 'truck_left', 'type': 'single', 'file': 'enemy_cars/truck_left.npy'},
            {'name': 'truck_right', 'type': 'single', 'file': 'enemy_cars/truck_right.npy'},
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
            left_mask = self.enemy_left_masks[enemy_type]
            right_mask = self.enemy_right_masks[enemy_type]
            return jax.lax.cond(going_left, lambda _: left_mask, lambda _: right_mask, operand=None)

        def render_enemy(carry, enemy_idx):
            raster = carry
            enemy_active = state.enemy_cars.active[enemy_idx]
            enemy_x = state.enemy_cars.position.x[enemy_idx]
            enemy_y = state.enemy_cars.position.y[enemy_idx]
            enemy_type = state.enemy_cars.type[enemy_idx]
            direction_x = state.enemy_cars.direction_x[enemy_idx]
            screen_y = 105 + (enemy_y - state.player_car.position.y)
            is_visible = jnp.logical_and(enemy_active, jnp.logical_and(screen_y > 25, screen_y < 195))
            enemy_mask = select_enemy_mask(enemy_type, direction_x < 0)

            raster = jax.lax.cond(
                is_visible,
                lambda r: self.jr.render_at(r, enemy_x.astype(jnp.int32), screen_y.astype(jnp.int32), enemy_mask),
                lambda r: r,
                operand=raster,
            )
            return raster, None

        raster, _ = jax.lax.scan(render_enemy, raster, jnp.arange(self.consts.MAX_ENEMY_CARS))

        player_mask = self.SHAPE_MASKS["player"]
        raster = self.jr.render_at(raster, state.player_car.position.x, 105, player_mask)

        wall_top_mask = self.SHAPE_MASKS["wall_top"]
        raster = self.jr.render_at(raster, 0, 0, wall_top_mask)

        wall_bottom_mask = self.SHAPE_MASKS["wall_bottom"]
        raster = self.jr.render_at(raster, 0, 210 - wall_bottom_mask.shape[0], wall_bottom_mask)

        all_flags_top_mask = self.SHAPE_MASKS["all_flags_top"]
        raster = self.jr.render_at(raster, 10, 20, all_flags_top_mask)

        # Render score centered at the top using dedicated score digit sprites
        score_digits = self.jr.int_to_digits(state.score, max_digits=self.score_max_digits)
        non_zero_mask = score_digits != 0
        has_non_zero = jnp.any(non_zero_mask)
        first_non_zero = jnp.argmax(non_zero_mask)
        start_index = jax.lax.select(has_non_zero, first_non_zero, self.score_max_digits - 1)
        num_to_render = jax.lax.select(has_non_zero, self.score_max_digits - start_index, 1)

        total_width = num_to_render * self.score_digit_spacing
        score_x = self.score_center_x - (total_width // 2)

        raster = self.jr.render_label_selective(
            raster,
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
            
            # Calculate flag X position on its road
            flag_x = jax.lax.cond(
                flag_road == 0,
                lambda _: self._get_flag_x_on_road(flag_y, flag_segment, self.consts.FIRST_TRACK_CORNERS_X),
                lambda _: self._get_flag_x_on_road(flag_y, flag_segment, self.consts.SECOND_TRACK_CORNERS_X),
                operand=None,
            )
            
            # Calculate screen Y position relative to player
            # The player is always rendered at Y=105, so flags scroll based on player position
            screen_y = 105 + (flag_y - state.player_car.position.y)
            
            # Check if flag is visible on screen and not collected
            is_visible = jnp.logical_and(
                jnp.logical_and(screen_y > 25, screen_y < 195),
                ~flag_collected
            )
            
            # Colorize the base flag mask
            color_id = self.flag_palette_ids[flag_color_idx]
            colored_flag_mask = jnp.where(
                self.flag_solid_mask,
                color_id,
                self.flag_base_mask,
            )
            
            # Render flag if visible
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
        
        raster, _ = jax.lax.scan(render_flag, raster, jnp.arange(self.consts.NUM_FLAGS))
        
        # Black out collected flags at the top
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
        
        raster, _ = jax.lax.scan(render_blackout, raster, jnp.arange(self.consts.NUM_FLAGS))

        # Render collectibles (unified for all types: cherry, balloon, lollypop, ice cream)
        def render_collectible(carry, collectible_idx):
            raster = carry
            collectible_y = state.collectibles.y[collectible_idx]
            collectible_x = state.collectibles.x[collectible_idx]
            collectible_active = state.collectibles.active[collectible_idx]
            collectible_color_idx = state.collectibles.color_idx[collectible_idx]
            collectible_type_id = state.collectibles.type_id[collectible_idx]
            
            # Calculate screen Y position relative to player
            screen_y = 105 + (collectible_y - state.player_car.position.y)
            
            # Check if collectible is visible on screen and active
            is_visible = jnp.logical_and(
                jnp.logical_and(screen_y > 25, screen_y < 195),
                collectible_active
            )
            
            # Select sprite based on type_id
            # type_id: 0=cherry, 1=balloon, 2=lollypop, 3=ice_cream
            def get_sprite_and_mask(type_id):
                cherry_result = (self.cherry_base_mask, self.cherry_solid_mask, self.cherry_palette_ids)
                balloon_result = (self.balloon_base_mask, self.balloon_solid_mask, self.balloon_palette_ids)
                lollypop_result = (self.lollypop_base_mask, self.lollypop_solid_mask, self.lollypop_palette_ids)
                ice_cream_result = (self.ice_cream_base_mask, self.ice_cream_solid_mask, self.ice_cream_palette_ids)
                
                # Use conditional branching to select sprite
                result = jax.lax.cond(
                    type_id == self.consts.COLLECTIBLE_TYPE_CHERRY,
                    lambda _: cherry_result,
                    lambda _: jax.lax.cond(
                        type_id == self.consts.COLLECTIBLE_TYPE_BALLOON,
                        lambda _: balloon_result,
                        lambda _: jax.lax.cond(
                            type_id == self.consts.COLLECTIBLE_TYPE_LOLLYPOP,
                            lambda _: lollypop_result,
                            lambda _: ice_cream_result,
                            operand=None,
                        ),
                        operand=None,
                    ),
                    operand=None,
                )
                return result
            
            base_mask, solid_mask, palette_ids = get_sprite_and_mask(collectible_type_id)
            
            # Only colorize inner pixels, keep black edges (palette ID 0 is black)
            color_id = palette_ids[collectible_color_idx]
            colored_mask = jnp.where(
                (base_mask != self.jr.TRANSPARENT_ID) & (base_mask != 0),
                color_id,
                base_mask,
            )
            
            # Render collectible if visible
            raster = jax.lax.cond(
                is_visible,
                lambda r: self.jr.render_at(r, collectible_x.astype(jnp.int32), screen_y.astype(jnp.int32), colored_mask),
                lambda r: r,
                operand=raster,
            )
            return raster, None
        
        raster, _ = jax.lax.scan(render_collectible, raster, jnp.arange(self.consts.MAX_COLLECTIBLES))

        all_lives_bottom_mask = self.SHAPE_MASKS["all_lives_bottom"]
        raster = self.jr.render_at(raster, 10, 195, all_lives_bottom_mask)

        wall_bottom_mask = self.SHAPE_MASKS["tempPointer"]
        raster = self.jr.render_at(raster, 140, 25, wall_bottom_mask)

        return self.jr.render_from_palette(raster, self.PALETTE)
    
    def _get_flag_x_on_road(self, y: chex.Array, road_segment: chex.Array, track_corners_x: chex.Array) -> chex.Array:
        """Calculate the X position on a road given a Y coordinate and road segment."""
        y1 = self.consts.TRACK_CORNERS_Y[road_segment]
        y2 = self.consts.TRACK_CORNERS_Y[road_segment + 1]
        x1 = track_corners_x[road_segment]
        x2 = track_corners_x[road_segment + 1]
        
        # Linear interpolation: x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        t = jax.lax.cond(
            y2 != y1,
            lambda _: (y - y1) / (y2 - y1),
            lambda _: 0.0,
            operand=None,
        )
        return x1 + t * (x2 - x1)