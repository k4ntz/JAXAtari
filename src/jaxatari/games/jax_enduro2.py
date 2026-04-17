import chex
from functools import partial
import jax
from jax import numpy as jnp, lax
import jax.random as jrandom
from flax import struct
import os
import numpy as np

from typing import Tuple, Optional, List, Dict, Any

# jaxatari
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.modification import AutoDerivedConstants
import jaxatari.spaces as spaces

@struct.dataclass
class Enduro2GameState:
    player_x: jnp.ndarray  # Float position
    player_y: jnp.ndarray
    step_count: jnp.ndarray
    day_count: jnp.ndarray
    level: jnp.ndarray
    level_passed: jnp.ndarray
    game_over: jnp.ndarray
    player_speed: jnp.ndarray
    distance: jnp.ndarray
    cars_to_pass: jnp.ndarray
    track_top_x: jnp.ndarray  # Float position
    track_top_x_curve_offset: jnp.ndarray
    collision_mode: jnp.ndarray  # bool
    collision_steps: jnp.ndarray  # int
    mountain_left_x: jnp.ndarray
    mountain_right_x: jnp.ndarray
    # opponents
    adjusted_opponent_index: jnp.ndarray # jnp.int32
    adjusted_opponent_lane: jnp.ndarray  # jnp.int32
    visible_opponent_positions: jnp.ndarray # shape (7, 3) [x, y, color_idx]
    opponent_index: jnp.ndarray
    weather_index: jnp.ndarray # jnp.int32

@struct.dataclass
class Enduro2Observation:
    # Minimal observation for now
    pass

@struct.dataclass
class Enduro2Info:
    pass

def _get_weather_color_codes() -> jnp.ndarray:
    """Returns the RGB color codes for each weather and each sprite scraped from the game."""
    return jnp.array([
        # sky,          gras,       mountains,      horizon 1,      horizon 2,  horizon 3 (highest)

        # day
        [[24, 26, 167], [0, 68, 0], [134, 134, 29], [24, 26, 167], [24, 26, 167], [24, 26, 167], ],  # day 1
        [[45, 50, 184], [0, 68, 0], [136, 146, 62], [45, 50, 184], [45, 50, 184], [45, 50, 184]],  # day 2
        [[45, 50, 184], [0, 68, 0], [192, 192, 192], [45, 50, 184], [45, 50, 184], [45, 50, 184]],  # day white mountain
        [[45, 50, 184], [236, 236, 236], [214, 214, 214], [45, 50, 184], [45, 50, 184], [45, 50, 184]],  # snow

        # Sunsets
        [[24, 26, 167], [20, 60, 0], [0, 68, 0], [24, 26, 167], [24, 26, 167], [24, 26, 167]],  # 1
        [[24, 26, 167], [20, 60, 0], [0, 68, 0], [104, 25, 154], [51, 26, 163], [24, 26, 167]],  # 2
        [[51, 26, 163], [20, 60, 0], [0, 68, 0], [151, 25, 122], [104, 25, 154], [51, 26, 163]],  # 3
        [[51, 26, 163], [20, 60, 0], [0, 68, 0], [167, 26, 26], [151, 25, 122], [104, 25, 154]],  # 4
        [[104, 25, 154], [48, 56, 0], [0, 0, 0], [163, 57, 21], [167, 26, 26], [151, 25, 122]],  # 5
        [[151, 25, 122], [48, 56, 0], [0, 0, 0], [181, 83, 40], [163, 57, 21], [167, 26, 26]],  # 6
        [[167, 26, 26], [48, 56, 0], [0, 0, 0], [162, 98, 33], [181, 83, 40], [163, 57, 21]],  # 7
        [[163, 57, 21], [48, 56, 0], [0, 0, 0], [134, 134, 29], [162, 98, 33], [181, 83, 40]],  # 8

        # night
        [[74, 74, 74], [0, 0, 0], [142, 142, 142], [74, 74, 74], [74, 74, 74], [74, 74, 74]],  # night 1
        [[74, 74, 74], [0, 0, 0], [142, 142, 142], [74, 74, 74], [74, 74, 74], [74, 74, 74]],  # fog night
        [[74, 74, 74], [0, 0, 0], [142, 142, 142], [74, 74, 74], [74, 74, 74], [74, 74, 74]],  # night 2

        # dawn
        [[111, 111, 111], [0, 0, 0], [181, 83, 40], [111, 111, 111], [111, 111, 111], [111, 111, 111]],  # dawn

    ], dtype=jnp.int32)

@partial(jax.jit, static_argnums=(0, 1, 2, 3))
def precompute_all_track_curves(max_offset: int, track_height: int, track_width: int, left: bool) -> jnp.ndarray:
    """
    Precomputes all possible track curves using integer offsets.
    """
    offset_range = jnp.arange(-max_offset, max_offset + 1)

    i = jnp.arange(track_height)
    perspective_offsets = jnp.where(i < 2, 0, (i - 1) // 2)
    depth_ratio = (track_height - i) / track_height
    curved_depth_ratio = jnp.power(depth_ratio, 3.0)

    track_spaces = jnp.where(i < 2, -1, jnp.minimum(i - 2, track_width)).astype(jnp.int32)
    base_left_xs = -perspective_offsets

    def compute_single_curve(offset):
        curve_shifts = jnp.floor(offset * curved_depth_ratio).astype(jnp.int32)
        final_left_xs = base_left_xs + curve_shifts
        final_left_xs = final_left_xs.at[-1].set(final_left_xs[-2])

        final_right_xs = jnp.where(
            track_spaces == -1,
            final_left_xs,
            final_left_xs + track_spaces + 1
        )
        final_right_xs = final_right_xs.at[-1].set(final_right_xs[-2])
        
        return jnp.where(left, final_left_xs.astype(jnp.int32), final_right_xs.astype(jnp.int32))

    return jax.vmap(compute_single_curve)(offset_range)

@partial(jax.jit, static_argnums=(0, 1, 2, 3, 4, 5))
def _compute_static_track(track_seed: int, max_track_length: float, min_track_section_length: float, max_track_section_length: float, straight_km_start: float, max_segments_buffer: int) -> jnp.ndarray:
    key = jax.random.PRNGKey(track_seed)
    max_segments = int(max_track_length) + max_segments_buffer
    key, subkey = jax.random.split(key)
    directions = jax.random.choice(subkey, jnp.array([-1, 0, 1]), shape=(max_segments,), replace=True)
    key, subkey = jax.random.split(key)
    segment_lengths = jax.random.uniform(subkey, shape=(max_segments,), minval=min_track_section_length, maxval=max_track_section_length)
    track_starts = jnp.cumsum(jnp.concatenate([jnp.array([straight_km_start]), segment_lengths[:-1]]))
    first_segment = jnp.array([[0.0, 0.0]])
    rest_segments = jnp.stack([directions, track_starts], axis=1)
    return jnp.concatenate([first_segment, rest_segments], axis=0)

@partial(jax.jit, static_argnums=(0, 1, 2, 3, 4))
def _compute_base_opponents(opponent_spawn_seed: int, opponent_density: float, length_of_opponent_array: int, opponent_delay_slots: int, player_color_idx: int) -> jnp.ndarray:
    key = jax.random.PRNGKey(opponent_spawn_seed)
    key, key_colors = jax.random.split(key)
    num_occupied = int(round(opponent_density * length_of_opponent_array))
    key, key_positions = jax.random.split(key)
    all_indices = jnp.arange(length_of_opponent_array)
    shuffled_indices = jax.random.permutation(key_positions, all_indices)
    occupied_positions = shuffled_indices[:num_occupied]
    occupancy_mask = jnp.zeros(length_of_opponent_array, dtype=jnp.bool_)
    occupancy_mask = occupancy_mask.at[occupied_positions].set(True)
    key, key_lanes = jax.random.split(key)
    lane_choices = jax.random.randint(key_lanes, (length_of_opponent_array,), 0, 3, dtype=jnp.int8)

    def generate_opponent_color_index(color_key):
        idx = jax.random.randint(color_key, (), 0, 13)
        return jnp.where(idx >= player_color_idx, idx + 1, idx).astype(jnp.int32)

    color_keys = jax.random.split(key_colors, length_of_opponent_array)
    colors = jax.vmap(generate_opponent_color_index)(color_keys)

    def process_slot(carry, inputs):
        key_step, last_two_lanes, non_gap_count = carry
        is_occupied, candidate_lane, color = inputs
        key_step, key_fix = jax.random.split(key_step)
        has_valid_triple = ((non_gap_count >= 2) & (last_two_lanes[0] != last_two_lanes[1]) & (candidate_lane != last_two_lanes[0]) & (candidate_lane != last_two_lanes[1]))
        fixed_lane = jax.random.choice(key_fix, jnp.array([last_two_lanes[0], last_two_lanes[1]]))
        final_lane = jax.lax.select(has_valid_triple & is_occupied, fixed_lane, candidate_lane)
        final_val = jax.lax.select(is_occupied, final_lane, jnp.array(-1, dtype=jnp.int8))
        new_non_gap = jax.lax.select(is_occupied, non_gap_count + 1, 0)
        new_last_two = jax.lax.select(is_occupied, jnp.array([last_two_lanes[1], final_lane]), last_two_lanes)
        new_color = jax.lax.select(is_occupied, color, -1)
        return (key_step, new_last_two, new_non_gap), jnp.array([final_val, new_color])

    init_carry = (key, jnp.array([-1, -1], dtype=jnp.int8), 0)
    _, final_opponents = jax.lax.scan(process_slot, init_carry, (occupancy_mask, lane_choices, colors))
    final_opponents = jnp.transpose(final_opponents)
    delay_block = jnp.full((2, opponent_delay_slots), -1, dtype=jnp.int32)
    return jnp.concatenate([final_opponents, delay_block], axis=1)

class Enduro2Constants(AutoDerivedConstants):
    """Game configuration parameters for Enduro2"""
    screen_width: int = struct.field(pytree_node=False, default=160)
    screen_height: int = struct.field(pytree_node=False, default=210)
    
    player_x_start: float = struct.field(pytree_node=False, default=76.0)
    player_y_start: float = struct.field(pytree_node=False, default=144.0)
    
    horizontal_movement_slope: float = struct.field(pytree_node=False, default=0.015)
    horizontal_movement_offset: float = struct.field(pytree_node=False, default=0.2)
    max_speed: float = struct.field(pytree_node=False, default=120.0)
    min_speed: float = struct.field(pytree_node=False, default=6.0)
    initial_speed: float = struct.field(pytree_node=False, default=120.0)
    frame_rate: float = struct.field(pytree_node=False, default=60.0)
    centripedal_shift_ratio: float = struct.field(pytree_node=False, default=256.0)

    # Acceleration and Braking
    acceleration_slow_down_threshold: float = struct.field(pytree_node=False, default=45.0)
    render_full_road: bool = struct.field(pytree_node=False, default=False)
    
    window_offset_left: int = struct.field(pytree_node=False, default=8)
    window_offset_bottom: int = struct.field(pytree_node=False, default=55)
    sky_height: int = struct.field(pytree_node=False, default=50)

    # Track
    track_width: int = struct.field(pytree_node=False, default=98)
    track_seed: int = struct.field(pytree_node=False, default=42)
    max_track_length: float = struct.field(pytree_node=False, default=9999.9)
    straight_km_start: float = struct.field(pytree_node=False, default=5.0)
    min_track_section_length: float = struct.field(pytree_node=False, default=1.0)
    max_track_section_length: float = struct.field(pytree_node=False, default=15.0)
    track_max_top_x_offset: float = struct.field(pytree_node=False, default=50.0)
    initial_track_top_x_curve_offset: float = struct.field(pytree_node=False, default=0.0)
    curve_rate: float = struct.field(pytree_node=False, default=0.05)
    side_collision_speed_drop: float = struct.field(pytree_node=False, default=0.75)
    collision_push_back: float = struct.field(pytree_node=False, default=0.5)
    collision_duration: int = struct.field(pytree_node=False, default=14)

    # Track Colors and Animation
    track_colors: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array([
        [74, 74, 74],  # top
        [111, 111, 111],  # moving top
        [170, 170, 170],  # moving bottom
        [192, 192, 192],  # bottom - rest
    ], dtype=jnp.int32))
    track_top_min_length: int = struct.field(pytree_node=False, default=33)
    track_moving_top_length: int = struct.field(pytree_node=False, default=13)
    track_moving_bottom_length: int = struct.field(pytree_node=False, default=18)
    track_move_range: int = struct.field(pytree_node=False, default=12)
    track_moving_bottom_spawn_step: int = struct.field(pytree_node=False, default=6)
    track_color_move_speed_per_speed: float = struct.field(pytree_node=False, default=0.05)
    track_speed_animation_factor: float = struct.field(pytree_node=False, default=0.085)

    # Opponents
    opponent_speed: int = struct.field(pytree_node=False, default=24)
    opponent_relative_speed_factor: float = struct.field(pytree_node=False, default=2.5)
    opponent_spawn_seed: int = struct.field(pytree_node=False, default=42)
    length_of_opponent_array: int = struct.field(pytree_node=False, default=5000)
    opponent_density: float = struct.field(pytree_node=False, default=0.2)
    opponent_delay_slots: int = struct.field(pytree_node=False, default=10)
    car_zero_y_pixel_range: int = struct.field(pytree_node=False, default=20)
    lane_ratios: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array([0.25, 0.5, 0.75], dtype=jnp.float32))
    PLAYER_COLOR_INDEX: int = struct.field(pytree_node=False, default=15)
    opponent_animation_steps: int = struct.field(pytree_node=False, default=8)
    car_widths: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array([16, 12, 8, 6, 4, 4, 2], dtype=jnp.int32))
    car_heights: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array([11, 8, 6, 4, 3, 2, 1], dtype=jnp.int32))
    car_width_0: int = struct.field(pytree_node=False, default=16)
    car_height_0: int = struct.field(pytree_node=False, default=11)

    # Mountains
    mountain_left_x_pos: float = struct.field(pytree_node=False, default=40.0)
    mountain_right_x_pos: float = struct.field(pytree_node=False, default=108.0)
    mountain_pixel_movement_per_frame_per_speed_unit: float = struct.field(pytree_node=False, default=0.01)

    # === Weather ===
    day_weather_index: int = struct.field(pytree_node=False, default=1)
    snow_weather_index: int = struct.field(pytree_node=False, default=3)
    sunset_weather_index: int = struct.field(pytree_node=False, default=8)
    night_weather_index: int = struct.field(pytree_node=False, default=12)
    fog_weather_index: int = struct.field(pytree_node=False, default=13)
    dawn_weather_index: int = struct.field(pytree_node=False, default=15)
    steering_snow_factor: float = struct.field(pytree_node=False, default=2.0)
    weather_with_night_car_sprite: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array([12, 13, 14], dtype=jnp.int32))
    fog_height: int = struct.field(pytree_node=False, default=80)
    weather_cycle_distance: float = struct.field(pytree_node=False, default=0.0)

    # Dynamic derived constants for weather
    weather_starts_s: Optional[jnp.ndarray] = struct.field(pytree_node=False, default=None)
    day_cycle_time: Optional[jnp.ndarray] = struct.field(pytree_node=False, default=None)
    weather_color_codes: jnp.ndarray = struct.field(pytree_node=False, default_factory=_get_weather_color_codes)

    # UI Positions
    info_box_x_pos: int = struct.field(pytree_node=False, default=48)
    info_box_y_pos: int = struct.field(pytree_node=False, default=161)
    distance_odometer_start_x: int = struct.field(pytree_node=False, default=65)
    score_start_x: int = struct.field(pytree_node=False, default=81)
    level_x: int = struct.field(pytree_node=False, default=57)
    score_start_y: Optional[int] = struct.field(pytree_node=False, default=None)
    distance_odometer_start_y: Optional[int] = struct.field(pytree_node=False, default=None)
    level_y: Optional[int] = struct.field(pytree_node=False, default=None)

    # Asset config for simplified renderer
    ASSET_CONFIG: tuple = struct.field(pytree_node=False, default_factory=lambda: (
        {'name': 'background', 'type': 'background', 'file': 'backgrounds/background.npy'},
        {'name': 'background_overlay', 'type': 'single', 'file': 'backgrounds/background_overlay.npy'},
        {'name': 'mountain_left', 'type': 'single', 'file': 'misc/mountain_left.npy'},
        {'name': 'mountain_right', 'type': 'single', 'file': 'misc/mountain_right.npy'},
        {'name': 'score_box', 'type': 'single', 'file': 'backgrounds/score_box.npy'},
        {'name': 'track_colors', 'type': 'procedural', 'data': jnp.array([[list(c) + [255] for c in [
            [74, 74, 74], [111, 111, 111], [170, 170, 170], [192, 192, 192]
        ]]], dtype=jnp.uint8)},
        {'name': 'digits_black', 'type': 'digits', 'pattern': 'digits/{}_black.npy'},
        {'name': 'black_digit_array', 'type': 'single', 'file': 'digits/black_digit_array.npy'},
        {'name': 'brown_digit_array', 'type': 'single', 'file': 'digits/brown_digit_array.npy'},
    ))

    def compute_derived(self) -> Dict[str, Any]:
        game_window_height = self.screen_height - self.window_offset_bottom
        km_per_second_per_speed_unit = 0.035 / self.min_speed
        km_per_speed_unit_per_frame = km_per_second_per_speed_unit / self.frame_rate
        
        track_height = game_window_height - self.sky_height - 1
        curve_offset_base = int(self.track_max_top_x_offset)
        precomputed_left_curves = precompute_all_track_curves(curve_offset_base, track_height, self.track_width, left=True)
        precomputed_right_curves = precompute_all_track_curves(curve_offset_base, track_height, self.track_width, left=False)
        whole_track = _compute_static_track(self.track_seed, self.max_track_length, self.min_track_section_length, self.max_track_section_length, self.straight_km_start, 100)
        
        opponent_slot_ys = jnp.array([
            game_window_height - self.car_zero_y_pixel_range,
            game_window_height - self.car_zero_y_pixel_range - 20,
            game_window_height - self.car_zero_y_pixel_range - 20 - 20,
            game_window_height - self.car_zero_y_pixel_range - 20 - 20 - 10,
            game_window_height - self.car_zero_y_pixel_range - 20 - 20 - 10 - 10,
            game_window_height - self.car_zero_y_pixel_range - 20 - 20 - 10 - 10 - 6,
            game_window_height - self.car_zero_y_pixel_range - 20 - 20 - 10 - 10 - 6 - 5,
        ], dtype=jnp.int32)

        base_opponents = _compute_base_opponents(
            self.opponent_spawn_seed, self.opponent_density,
            self.length_of_opponent_array, self.opponent_delay_slots, self.PLAYER_COLOR_INDEX
        )

        weather_starts_s = jnp.array([
            34,  # day 1
            34 + 34,  # day 2 (lighter)
            34 + 34 + 34,  # day 3 (white mountains)
            34 + 34 + 34 + 69,  # snow (steering is more difficult)
            34 + 34 + 34 + 69 + 8 * 1,  # Sunset 1
            34 + 34 + 34 + 69 + 8 * 2,  # Sunset 2
            34 + 34 + 34 + 69 + 8 * 3,  # Sunset 3
            34 + 34 + 34 + 69 + 8 * 4,  # Sunset 4
            34 + 34 + 34 + 69 + 8 * 5,  # Sunset 5
            34 + 34 + 34 + 69 + 8 * 6,  # Sunset 6
            34 + 34 + 34 + 69 + 8 * 7,  # Sunset 7
            34 + 34 + 34 + 69 + 8 * 8,  # Sunset 8
            34 + 34 + 34 + 69 + 8 * 8 + 69,  # night 1
            34 + 34 + 34 + 69 + 8 * 8 + 69 + 69,  # fog night
            34 + 34 + 34 + 69 + 8 * 8 + 69 + 69 + 34,  # night 2
            34 + 34 + 34 + 69 + 8 * 8 + 69 + 69 + 34 + 34,  # dawn
        ], dtype=jnp.int32)
        day_cycle_time = weather_starts_s[15]

        return {
            'game_window_height': game_window_height,
            'km_per_speed_unit_per_frame': km_per_speed_unit_per_frame,
            'distance_odometer_start_y': game_window_height + 9,
            'score_start_y': game_window_height + 25,
            'level_y': game_window_height + 25,
            'track_height': track_height,
            'curve_offset_base': curve_offset_base,
            'precomputed_left_curves': precomputed_left_curves,
            'precomputed_right_curves': precomputed_right_curves,
            'whole_track': whole_track,
            'opponent_slot_ys': opponent_slot_ys,
            'base_opponents': base_opponents,
            'weather_starts_s': weather_starts_s,
            'day_cycle_time': day_cycle_time,
        }
    
    game_window_height: Optional[int] = struct.field(pytree_node=False, default=None)
    km_per_speed_unit_per_frame: Optional[float] = struct.field(pytree_node=False, default=None)
    track_height: Optional[int] = struct.field(pytree_node=False, default=None)
    curve_offset_base: Optional[int] = struct.field(pytree_node=False, default=None)
    precomputed_left_curves: Optional[chex.Array] = struct.field(pytree_node=False, default=None)
    precomputed_right_curves: Optional[chex.Array] = struct.field(pytree_node=False, default=None)
    whole_track: Optional[chex.Array] = struct.field(pytree_node=False, default=None)
    opponent_slot_ys: Optional[jnp.ndarray] = struct.field(pytree_node=False, default=None)
    base_opponents: Optional[chex.Array] = struct.field(pytree_node=False, default=None)



class Enduro2Renderer(JAXGameRenderer):
    """
    Simplified renderer for Enduro2
    """
    def __init__(self, consts: Enduro2Constants = None, config: render_utils.RendererConfig = None):
        self.consts = consts or Enduro2Constants()
        super().__init__(self.consts)
        
        if config is None:
            self.config = render_utils.RendererConfig(
                game_dimensions=(self.consts.screen_height, self.consts.screen_width),
                channels=3,
                downscale=None
            )
        else:
            self.config = config
            
        self.jr = render_utils.JaxRenderingUtils(self.config)
        
        # Asset base path
        self._sprite_path = os.path.join(render_utils.get_base_sprite_dir(), "enduro")

        # Load player car manually because it's 4D (animated) and the standard
        # loadFrame only supports 3D sprites.
        player_car_path = os.path.join(self._sprite_path, 'cars/car_0.npy')
        player_car_night_path = os.path.join(self._sprite_path, 'cars/car_0_night.npy')
        
        player_car_data = jnp.load(player_car_path)
        asset_config = list(self.consts.ASSET_CONFIG)
        asset_config.append({'name': 'player_car', 'type': 'procedural', 'data': player_car_data})

        flags_path = os.path.join(self._sprite_path, 'misc/flags.npy')
        if os.path.exists(flags_path):
            flags_data = jnp.load(flags_path)
            asset_config.append({'name': 'flags', 'type': 'procedural', 'data': flags_data})

        if os.path.exists(player_car_night_path):
            player_car_night_data = jnp.load(player_car_night_path)
            asset_config.append({'name': 'player_car_night', 'type': 'procedural', 'data': player_car_night_data})

        # Add car colors to asset config to ensure they are in the palette
        car_rgbs_to_add = [
            (136, 146, 62), (72, 160, 72), (104, 72, 198), (66, 136, 176), 
            (66, 114, 194), (198, 108, 58), (162, 162, 42), (66, 158, 130), 
            (162, 134, 56), (110, 156, 66), (184, 70, 162), (66, 72, 200), (200, 72, 72)
        ]
        color_dummy = jnp.array([[list(rgb) + [255] for rgb in car_rgbs_to_add]], dtype=jnp.uint8)
        asset_config.append({'name': 'car_colors', 'type': 'procedural', 'data': color_dummy})

        # Add all 16 weather colors to palette so they exist in COLOR_TO_ID
        weather_colors = self.consts.weather_color_codes
        num_weathers = weather_colors.shape[0]
        for weather_idx in range(num_weathers):
            for color_idx in range(6): # sky, grass, mountain, horizon1, 2, 3
                color = np.array(list(weather_colors[weather_idx, color_idx]) + [255], dtype=np.uint8)
                asset_config.append({'name': f'weather_color_{weather_idx}_{color_idx}', 'type': 'procedural', 'data': color.reshape(1, 1, 4)})

        # Load car sprites manually
        for i in range(7):
            # For the largest opponent slot (0), we use car_1.npy to avoid using the player car sprite.
            actual_idx = i if i > 0 else 1
            car_path = os.path.join(self._sprite_path, f'cars/car_{actual_idx}.npy')
            if os.path.exists(car_path):
                car_data = jnp.load(car_path)
                asset_config.append({'name': f'car_{i}', 'type': 'procedural', 'data': car_data})
                
            car_night_path = os.path.join(self._sprite_path, f'cars/car_{actual_idx}_night.npy')
            if os.path.exists(car_night_path):
                car_night_data = jnp.load(car_night_path)
                asset_config.append({'name': f'car_night_{i}', 'type': 'procedural', 'data': car_night_data})
        
        # Load assets
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS,
        ) = self.jr.load_and_setup_assets(asset_config, self._sprite_path)
        
        # Store WEATHER Color IDs [weather_idx, color_idx (sky, grass, mountain, h1, h2, h3)]
        weather_ids = []
        for w_idx in range(num_weathers):
            w_colors = []
            for c_idx in range(6):
                rgb = tuple(weather_colors[w_idx, c_idx].tolist())
                w_colors.append(self.COLOR_TO_ID.get(rgb, 0))
            weather_ids.append(w_colors)
        self.WEATHER_COLOR_IDS = jnp.array(weather_ids, dtype=jnp.uint8)

        # Store Track Color IDs
        track_rgbs = [tuple(c.tolist()) for c in self.consts.track_colors]
        self.TRACK_COLOR_IDS = jnp.array([self.COLOR_TO_ID.get(rgb, 0) for rgb in track_rgbs], dtype=jnp.uint8)

        # Store Odometer Sheet ID Masks
        self.black_digit_sheet_mask = self.SHAPE_MASKS['black_digit_array']
        self.brown_digit_sheet_mask = self.SHAPE_MASKS['brown_digit_array']

        # Store Car Color IDs for recoloring
        car_rgbs = [
            (136, 146, 62), (72, 160, 72), (104, 72, 198), (66, 136, 176), 
            (66, 114, 194), (198, 108, 58), (162, 162, 42), (66, 158, 130), 
            (162, 134, 56), (110, 156, 66), (184, 70, 162), (66, 72, 200), (200, 72, 72)
        ]
        self.CAR_COLOR_IDS = jnp.array([self.COLOR_TO_ID.get(rgb, 0) for rgb in car_rgbs], dtype=jnp.uint8)
        self.black_id = self.COLOR_TO_ID.get((0, 0, 0), 0)
        self.white_id = self.COLOR_TO_ID.get((255, 255, 255), 0)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: Enduro2GameState) -> jnp.ndarray:
        # Start with the static background (mostly black)
        raster = self.BACKGROUND
        
        xx, yy = self.jr._xx, self.jr._yy
        is_fog = state.weather_index == self.consts.fog_weather_index

        # Get current weather colors [sky, grass, mountain, h1, h2, h3]
        current_weather_colors = self.WEATHER_COLOR_IDS[state.weather_index]

        # 1. Draw Sky (top part of game window)
        sky_color = current_weather_colors[0]
        sky_mask = (xx >= self.consts.window_offset_left) & (yy < self.consts.sky_height - 6)
        raster = jnp.where(sky_mask, sky_color, raster)

        # 1.1 Draw Horizons (h3, h2, h1 from top to bottom)
        h3_mask = (xx >= self.consts.window_offset_left) & (yy >= self.consts.sky_height - 6) & (yy < self.consts.sky_height - 4)
        raster = jnp.where(h3_mask, current_weather_colors[5], raster)

        h2_mask = (xx >= self.consts.window_offset_left) & (yy >= self.consts.sky_height - 4) & (yy < self.consts.sky_height - 2)
        raster = jnp.where(h2_mask, current_weather_colors[4], raster)

        h1_mask = (xx >= self.consts.window_offset_left) & (yy >= self.consts.sky_height - 2) & (yy < self.consts.sky_height)
        raster = jnp.where(h1_mask, current_weather_colors[3], raster)

        
        # 2. Draw Mountains
        raster = self._render_mountains(raster, state)

        # 3. Draw Grass (bottom part of game window)
        grass_color = jnp.where(is_fog, current_weather_colors[0], current_weather_colors[1])
        grass_mask = (xx >= self.consts.window_offset_left) & (yy >= self.consts.sky_height) & (yy < self.consts.game_window_height)
        raster = jnp.where(grass_mask, grass_color, raster)
        
        # 4. Draw curved track with perspective
        raster = self._render_track(raster, state, xx, yy)
        
        # 5. Render the lower background overlays
        raster = self.jr.render_at(raster, 0, self.consts.game_window_height - 1, self.SHAPE_MASKS['background_overlay'])
        raster = self.jr.render_at(raster, self.consts.info_box_x_pos, self.consts.info_box_y_pos, self.SHAPE_MASKS['score_box'])
        
        # 6. Render UI
        raster = self._render_distance_odometer(raster, state)
        raster = self._render_cars_to_pass(raster, state)
        raster = self._render_day_counter(raster, state)

        # 7. Render Opponents
        raster = self._render_opponent_cars(raster, state)

        # 8. Render player car
        is_night = jnp.isin(state.weather_index, self.consts.weather_with_night_car_sprite)

        # Calculate animation period based on player speed
        animation_period = self.consts.opponent_animation_steps - (state.player_speed - self.consts.opponent_speed) / (
                self.consts.max_speed - self.consts.opponent_speed) * (self.consts.opponent_animation_steps - 1)
        animation_period = jnp.maximum(1.0, animation_period)
        
        # Faster vibration during collision
        animation_period = jnp.where(state.collision_mode, 2.0, animation_period)

        animation_step = jnp.floor(state.step_count / animation_period)
        frame_index = (animation_step % 2).astype(jnp.int32)

        def get_player_mask():
            mask_day = self.SHAPE_MASKS['player_car']
            mask_night = self.SHAPE_MASKS.get('player_car_night', mask_day)
            
            def get_day_frame():
                if mask_day.ndim == 3:
                    return mask_day[jnp.minimum(frame_index, mask_day.shape[0] - 1)]
                return mask_day
                
            def get_night_frame():
                if mask_night.ndim == 3:
                    return mask_night[jnp.minimum(frame_index, mask_night.shape[0] - 1)]
                return mask_night

            return jax.lax.cond(is_night, get_night_frame, get_day_frame)

        player_mask = get_player_mask()
        raster = self.jr.render_at(raster, state.player_x.astype(jnp.int32), state.player_y.astype(jnp.int32), player_mask)
        # Convert ID raster to RGB
        return self.jr.render_from_palette(raster, self.PALETTE)

    @partial(jax.jit, static_argnums=(0,))
    def _render_opponent_cars(self, raster: jnp.ndarray, state: Enduro2GameState) -> jnp.ndarray:
        """
        Renders visible opponent cars.
        """
        # Calculate animation period based on player speed
        animation_period = self.consts.opponent_animation_steps - (state.player_speed - self.consts.opponent_speed) / (
                self.consts.max_speed - self.consts.opponent_speed) * (self.consts.opponent_animation_steps - 1)
        animation_period = jnp.maximum(1.0, animation_period)
        animation_step = jnp.floor(state.step_count / animation_period)
        frame_index = (animation_step % 2).astype(jnp.int32)
        
        is_fog = state.weather_index == self.consts.fog_weather_index
        is_night = jnp.isin(state.weather_index, self.consts.weather_with_night_car_sprite)

        def render_slot(r, i):
            pos = state.visible_opponent_positions[i]
            x, y, color_idx = pos[0], pos[1], pos[2].astype(jnp.int32)
            
            # -1 indicates no car in this slot
            exists = x != -1
            # In fog, hide if above fog_height
            visible = jnp.where(is_fog, y >= self.consts.fog_height, True)
            
            def do_render(r_inner):
                mask_day = self.SHAPE_MASKS[f'car_{i}']
                mask_night = self.SHAPE_MASKS[f'car_night_{i}']
                
                # Make sure both have the same number of dimensions before jax.lax.cond
                if mask_day.ndim == 3 and mask_night.ndim == 2:
                    mask_night = jnp.expand_dims(mask_night, axis=0)
                    mask_night = jnp.repeat(mask_night, mask_day.shape[0], axis=0)

                mask = jax.lax.cond(is_night, lambda: mask_night, lambda: mask_day)
                
                if mask.ndim == 3:
                    # Use animated frame if available
                    num_frames = mask.shape[0]
                    mask = mask[jnp.minimum(frame_index, num_frames - 1)]
                
                # Dynamic recoloring
                car_color_id = self.CAR_COLOR_IDS[color_idx % len(self.CAR_COLOR_IDS)]
                # Replace everything that is NOT transparent, NOT black, and NOT white
                is_body = (mask != self.jr.TRANSPARENT_ID) & (mask != self.black_id) & (mask != self.white_id)
                
                # If it's night, we don't recolor the body because it's only lights. We only draw the original mask.
                mask = jnp.where(~is_night & is_body, car_color_id, mask)
                
                return self.jr.render_at_clipped(r_inner, x.astype(jnp.int32), y.astype(jnp.int32), mask)

            return jax.lax.cond(exists & visible, do_render, lambda r_in: r_in, r)

        # Render from far (6) to near (0) so near cars are on top
        for i in range(6, -1, -1):
            raster = render_slot(raster, i)
            
        return raster


    @partial(jax.jit, static_argnums=(0,))
    def _render_mountains(self, raster: jnp.ndarray, state: Enduro2GameState) -> jnp.ndarray:
        """
        Renders mountains using dynamic scrolling and wrapping.
        """
        is_fog = state.weather_index == self.consts.fog_weather_index
        def skip_render(r):
            return r

        def do_render_mountains(r):
            mountain_left_sprite = self.SHAPE_MASKS['mountain_left']
            mountain_right_sprite = self.SHAPE_MASKS['mountain_right']

            # Dynamic recoloring for snow and night
            current_weather_colors = self.WEATHER_COLOR_IDS[state.weather_index]
            mountain_color_id = current_weather_colors[2]
            
            def recolor_mountain(mask):
                is_body = (mask != self.jr.TRANSPARENT_ID)
                return jnp.where(is_body, mountain_color_id, mask)

            mountain_left_sprite = recolor_mountain(mountain_left_sprite)
            mountain_right_sprite = recolor_mountain(mountain_right_sprite)

            # Positions
            x_left = state.mountain_left_x.astype(jnp.int32)
            x_right = state.mountain_right_x.astype(jnp.int32)
            
            y_left = self.consts.sky_height - mountain_left_sprite.shape[0]
            y_right = self.consts.sky_height - mountain_right_sprite.shape[0]

            # Visible interval
            lo = self.consts.window_offset_left
            hi = self.consts.screen_width
            period = hi - lo + 1

            # 1) Base draw
            r = self.jr.render_at_clipped(r, x_left, y_left, mountain_left_sprite)
            r = self.jr.render_at_clipped(r, x_right, y_right, mountain_right_sprite)

            # 2) Wrap-around
            overflow_left = (x_left + mountain_left_sprite.shape[1]) > (hi + 1)
            overflow_right = (x_right + mountain_right_sprite.shape[1]) > (hi + 1)

            r = jax.lax.cond(
                overflow_left,
                lambda r_inner: self.jr.render_at_clipped(r_inner, x_left - period, y_left, mountain_left_sprite),
                lambda r_inner: r_inner,
                r
            )
            r = jax.lax.cond(
                overflow_right,
                lambda r_inner: self.jr.render_at_clipped(r_inner, x_right - period, y_right, mountain_right_sprite),
                lambda r_inner: r_inner,
                r
            )
            return r

        return jax.lax.cond(is_fog, skip_render, do_render_mountains, raster)

    @partial(jax.jit, static_argnums=(0,))
    def _render_track(self, raster: jnp.ndarray, state: Enduro2GameState, xx: jnp.ndarray, yy: jnp.ndarray) -> jnp.ndarray:
        """
        Renders the track. If render_full_road is True, it renders the full road area.
        Otherwise, it renders only the side boundaries of the road.
        """
        left_xs, right_xs = self._generate_viewable_track_lookup(state.track_top_x, state.track_top_x_curve_offset)
        
        # We need to map yy to track rows (0 to track_height-1)
        # track starts at sky_height and ends at game_window_height - 1
        track_row = (yy - self.consts.sky_height).astype(jnp.int32)
        
        # Boundary check for track_row
        is_track_row = (yy >= self.consts.sky_height) & (yy < self.consts.game_window_height - 1)
        
        # In fog, hide track above fog_height
        is_fog = state.weather_index == self.consts.fog_weather_index
        is_visible = jnp.where(is_fog, yy >= self.consts.fog_height, True)

        # Get boundaries for each pixel's row
        l_x = left_xs[jnp.clip(track_row, 0, self.consts.track_height - 1)]
        r_x = right_xs[jnp.clip(track_row, 0, self.consts.track_height - 1)]
        
        # Determine the track mask based on the render_full_road flag
        track_mask = jax.lax.cond(
            self.consts.render_full_road,
            lambda: (xx >= l_x) & (xx <= r_x),
            lambda: (xx == l_x) | (xx == r_x)
        )
        
        track_mask = is_track_row & track_mask & is_visible

        # Calculate animation step
        effective_speed = (self.consts.min_speed +
                           (state.player_speed - self.consts.min_speed) * self.consts.track_speed_animation_factor)
        animation_step = jnp.floor(
            effective_speed * state.step_count * self.consts.track_color_move_speed_per_speed
        ) % self.consts.track_move_range
        animation_step = animation_step.astype(jnp.int32)

        # Calculate color indices for each row
        top_region_end = self.consts.track_top_min_length + animation_step
        moving_top_end = top_region_end + self.consts.track_moving_top_length
        spawn_moving_bottom = animation_step >= self.consts.track_moving_bottom_spawn_step
        moving_bottom_end = jnp.where(
            spawn_moving_bottom,
            moving_top_end + self.consts.track_moving_bottom_length,
            moving_top_end
        )

        color_indices = jnp.where(
            track_row < top_region_end,
            0,
            jnp.where(
                track_row < moving_top_end,
                1,
                jnp.where(
                    (track_row < moving_bottom_end) & spawn_moving_bottom,
                    2,
                    3
                )
            )
        ).astype(jnp.int32)

        track_pixel_color_ids = self.TRACK_COLOR_IDS[color_indices]
        
        raster = jnp.where(track_mask, track_pixel_color_ids, raster)
        return raster

    @partial(jax.jit, static_argnums=(0,))
    def _generate_viewable_track_lookup(
            self,
            top_x: jnp.ndarray,
            top_x_curve_offset: jnp.float32
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Fast track generation using precomputed curve lookup.
        """
        offset_int = jnp.clip(
            jnp.floor(top_x_curve_offset).astype(jnp.int32),
            -self.consts.curve_offset_base,
            self.consts.curve_offset_base
        )

        curve_index = offset_int + self.consts.curve_offset_base

        base_left_curve = self.consts.precomputed_left_curves[curve_index]
        base_right_curve = self.consts.precomputed_right_curves[curve_index]

        final_left_xs = base_left_curve + top_x.astype(jnp.int32)
        final_right_xs = base_right_curve + top_x.astype(jnp.int32)

        return final_left_xs, final_right_xs

    @partial(jax.jit, static_argnums=(0,))
    def _render_distance_odometer(self, raster: jnp.ndarray, state: Enduro2GameState) -> jnp.ndarray:
        """
        Renders the rolling odometer using dynamic slices from ID mask sheets.
        """
        # Get digit dimensions from the pre-loaded digit masks
        digit_mask_sample = self.SHAPE_MASKS['digits_black'][0]
        window_height = digit_mask_sample.shape[0] + 2
        digit_width = digit_mask_sample.shape[1]

        # Get the ID mask sheets
        digit_sheet_black = self.black_digit_sheet_mask
        digit_sheet_brown = self.brown_digit_sheet_mask

        # determine the base position in the sprite that represents the lowest y for the window
        base_y = digit_sheet_brown.shape[0] - window_height + 1

        # Calculate how many 0.0125 increments have passed for decimal animation
        increments_passed = jnp.floor(state.distance / 0.0125).astype(jnp.int32)
        y_offset = increments_passed % 80

        # Extract actual digit values from distance
        distance_int = jnp.floor(state.distance).astype(jnp.int32)
        decimal_digit = (state.distance * 10) % 1
        thousands_digit = (distance_int // 1000) % 10
        hundreds_digit = (distance_int // 100) % 10
        tens_digit = (distance_int // 10) % 10
        ones_digit = distance_int % 10

        # synchronize the movement of the digits
        decimal_y = base_y - y_offset
        ones_y = base_y - ones_digit * (window_height - 1) - jnp.clip(window_height - decimal_y - 2, 0)
        tens_y = base_y - tens_digit * (window_height - 1) - jnp.clip(window_height - ones_y - 2, 0)
        hundreds_y = base_y - hundreds_digit * (window_height - 1) - jnp.clip(window_height - tens_y - 2, 0)
        thousands_y = base_y - thousands_digit * (window_height - 1) - jnp.clip(window_height - hundreds_y - 2, 0)

        # Reset to base_y when we complete a full cycle
        decimal_y = jnp.where(decimal_digit < 0.001, base_y, decimal_y)
        ones_y = jnp.where(ones_y < 0.001, base_y, ones_y)
        tens_y = jnp.where(tens_y < 0.001, base_y, tens_y)
        hundreds_y = jnp.where(hundreds_y < 0.001, base_y, hundreds_y)
        thousands_y = jnp.where(thousands_y < 0.001, base_y, thousands_y)

        # Extract decimal digit window (ID MASK)
        digit_window = jax.lax.dynamic_slice(
            digit_sheet_brown,
            (decimal_y, 0),  # start indices (y, x)
            (window_height, digit_width)
        )
        
        ones_window = jax.lax.dynamic_slice(
            digit_sheet_black, (ones_y, 0), (window_height, digit_width)
        )
        tens_window = jax.lax.dynamic_slice(
            digit_sheet_black, (tens_y, 0), (window_height, digit_width)
        )
        hundreds_window = jax.lax.dynamic_slice(
            digit_sheet_black, (hundreds_y, 0), (window_height, digit_width)
        )
        thousands_window = jax.lax.dynamic_slice(
            digit_sheet_black, (thousands_y, 0), (window_height, digit_width)
        )

        # === Render all number ID masks ===
        render_y = self.consts.distance_odometer_start_y
        spacing = digit_width + 2
        decimal_x = self.consts.distance_odometer_start_x + 4 * spacing
        raster = self.jr.render_at(raster, decimal_x, render_y, digit_window)
        ones_x = self.consts.distance_odometer_start_x + 3 * spacing
        raster = self.jr.render_at(raster, ones_x, render_y, ones_window)
        tens_x = self.consts.distance_odometer_start_x + 2 * spacing
        raster = self.jr.render_at(raster, tens_x, render_y, tens_window)
        hundreds_x = self.consts.distance_odometer_start_x + 1 * spacing
        raster = self.jr.render_at(raster, hundreds_x, render_y, hundreds_window)
        thousands_x = self.consts.distance_odometer_start_x
        raster = self.jr.render_at(raster, thousands_x, render_y, thousands_window)

        return raster

    @partial(jax.jit, static_argnums=(0,))
    def _render_cars_to_pass(self, raster: jnp.ndarray, state: Enduro2GameState) -> jnp.ndarray:
        """
        Renders the "Cars to pass" digits or green flags if level passed.
        """
        def render_flags(r):
            # Animate the flags: switch frame every 8 steps
            animation_step = (state.step_count // 8) % 2
            flag_mask = self.SHAPE_MASKS['flags'][animation_step.astype(jnp.int32)]
            # Draw flags with black background
            return self.jr.render_at(r, self.consts.score_start_x - 9, self.consts.score_start_y - 1, flag_mask)
            
        def render_digits(r):
            digit_sprites = self.SHAPE_MASKS['digits_black']
            
            hundreds = (state.cars_to_pass // 100) % 10
            tens = (state.cars_to_pass // 10) % 10
            ones = state.cars_to_pass % 10
            
            spacing = digit_sprites.shape[2] + 2
            
            # Only show hundreds if > 0
            hundreds_mask = digit_sprites[hundreds]
            r = jax.lax.cond(
                state.cars_to_pass >= 100,
                lambda r_in: self.jr.render_at(r_in, self.consts.score_start_x, self.consts.score_start_y, hundreds_mask),
                lambda r_in: r_in,
                r
            )
            
            # Only show tens if >= 10
            tens_mask = digit_sprites[tens]
            r = jax.lax.cond(
                state.cars_to_pass >= 10,
                lambda r_in: self.jr.render_at(r_in, self.consts.score_start_x + spacing, self.consts.score_start_y, tens_mask),
                lambda r_in: r_in,
                r
            )
            
            ones_mask = digit_sprites[ones]
            r = self.jr.render_at(r, self.consts.score_start_x + 2 * spacing, self.consts.score_start_y, ones_mask)
            
            return r

        return jax.lax.cond(
            state.level_passed > 0,
            render_flags,
            render_digits,
            raster
        )

    @partial(jax.jit, static_argnums=(0,))
    def _render_day_counter(self, raster: jnp.ndarray, state: Enduro2GameState) -> jnp.ndarray:
        """
        Renders the current day (level) digit.
        """
        # In Enduro, the day counter is the level number (starting from 1)
        current_day = state.level
        
        # Clip current_day to 0-9 for safety, though level should be 1-5
        safe_day = jnp.clip(current_day, 0, 9).astype(jnp.int32)
        
        digit_sprites = self.SHAPE_MASKS['digits_black']
        day_digit_mask = digit_sprites[safe_day]
        
        # Render the day digit at the designated position
        raster = self.jr.render_at(raster, self.consts.level_x, self.consts.level_y, day_digit_mask)
        
        return raster


class JaxEnduro2(JaxEnvironment[Enduro2GameState, Enduro2Observation, Enduro2Info, Enduro2Constants]):
    ACTION_SET: jnp.ndarray = jnp.array(
        [
            Action.NOOP,
            Action.FIRE,
            Action.RIGHT,
            Action.LEFT,
            Action.DOWN,
            Action.DOWNRIGHT,
            Action.DOWNLEFT,
            Action.RIGHTFIRE,
            Action.LEFTFIRE,
        ],
        dtype=jnp.int32,
    )

    def __init__(self, consts: Enduro2Constants = None):
        super().__init__(consts or Enduro2Constants())
        self.renderer = Enduro2Renderer(consts=self.consts)

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTION_SET))

    def observation_space(self) -> spaces.Space:
        return spaces.Dict({"image": self.image_space()})

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.consts.screen_height, self.consts.screen_width, 3),
            dtype=jnp.uint8
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jrandom.PRNGKey = None) -> Tuple[Enduro2Observation, Enduro2GameState]:
        # Generate initial visible opponent positions
        opponent_index = jnp.array(0.0, dtype=jnp.float32)
        
        # We need a dummy track to calculate initial positions
        # In reset, the track is straight (offset 0)
        track_height = self.consts.track_height
        curve_offset_base = self.consts.curve_offset_base
        base_left_curve = self.consts.precomputed_left_curves[curve_offset_base]
        base_right_curve = self.consts.precomputed_right_curves[curve_offset_base]
        track_top_x = 84.0
        visible_track_left = base_left_curve + track_top_x
        visible_track_right = base_right_curve + track_top_x

        visible_opponent_positions = self._get_visible_opponent_positions(
            opponent_index,
            self.consts.base_opponents,
            jnp.array(-1, dtype=jnp.int32),
            jnp.array(-1, dtype=jnp.int32),
            visible_track_left,
            visible_track_right
        )

        state = Enduro2GameState(
            player_x=jnp.array(self.consts.player_x_start, dtype=jnp.float32),
            player_y=jnp.array(self.consts.player_y_start, dtype=jnp.float32),
            step_count=jnp.array(0, dtype=jnp.int32),
            day_count=jnp.array(0, dtype=jnp.int32),
            level=jnp.array(1, dtype=jnp.int32),
            level_passed=jnp.array(0, dtype=jnp.int32),
            game_over=jnp.array(False, dtype=jnp.bool_),
            player_speed=jnp.array(self.consts.initial_speed, dtype=jnp.float32),
            distance=jnp.array(0.0, dtype=jnp.float32),
            cars_to_pass=jnp.array(200, dtype=jnp.int32),
            track_top_x=jnp.array(track_top_x, dtype=jnp.float32),
            track_top_x_curve_offset=jnp.array(self.consts.initial_track_top_x_curve_offset, dtype=jnp.float32),
            collision_mode=jnp.array(False, dtype=jnp.bool_),
            collision_steps=jnp.array(0, dtype=jnp.int32),
            mountain_left_x=jnp.array(self.consts.mountain_left_x_pos, dtype=jnp.float32),
            mountain_right_x=jnp.array(self.consts.mountain_right_x_pos, dtype=jnp.float32),
            adjusted_opponent_index=jnp.array(-1, dtype=jnp.int32),
            adjusted_opponent_lane=jnp.array(-1, dtype=jnp.int32),
            visible_opponent_positions=visible_opponent_positions,
            opponent_index=opponent_index,
            weather_index=jnp.array(0, dtype=jnp.int32)
        )
        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def _get_visible_opponent_positions(self, opponent_index: jnp.ndarray,
                                        base_opponents: jnp.ndarray,
                                        adjusted_opponent_index: jnp.ndarray,
                                        adjusted_opponent_lane: jnp.ndarray,
                                        visible_track_left: jnp.ndarray,
                                        visible_track_right: jnp.ndarray) -> jnp.ndarray:
        """
        Calculate the x,y positions and colors of all 7 opponent slots.
        Adapted from jax_enduro.py
        """
        base_y_positions = self.consts.opponent_slot_ys
        slot_heights = jnp.diff(base_y_positions, prepend=self.consts.game_window_height)
        slot_heights = jnp.abs(slot_heights)

        index_integer = jnp.floor(opponent_index).astype(jnp.int32)
        index_decimal = opponent_index - index_integer

        opponent_array_size = base_opponents.shape[1]
        slot_indices = (index_integer + jnp.arange(7)) % opponent_array_size
        
        current_slots = base_opponents[0][slot_indices]
        current_colors = base_opponents[1][slot_indices]

        is_adjusted = slot_indices == adjusted_opponent_index
        current_slots = jnp.where(is_adjusted & (adjusted_opponent_lane != -1), adjusted_opponent_lane, current_slots)

        y_offsets = jnp.floor(index_decimal * slot_heights).astype(jnp.int32)
        y_positions = base_y_positions + y_offsets

        track_row_indices = y_positions - self.consts.sky_height
        track_row_indices = jnp.clip(track_row_indices, 0, self.consts.track_height - 1)

        left_boundaries = visible_track_left[track_row_indices]
        right_boundaries = visible_track_right[track_row_indices]
        track_widths = right_boundaries - left_boundaries

        car_widths = self.consts.car_widths

        def calculate_x_for_lane(slot_idx, lane_code, left_bound, track_width):
            valid_lane = jnp.clip(lane_code, 0, 2)
            ratio = self.consts.lane_ratios[valid_lane]
            center_x = left_bound + (track_width * ratio)
            car_width = car_widths[slot_idx]
            leftmost_x = center_x - (car_width // 2)
            return jnp.where(lane_code == -1, -1.0, leftmost_x)

        local_slot_indices = jnp.arange(7)
        x_positions = jax.vmap(calculate_x_for_lane)(
            local_slot_indices, current_slots, left_boundaries, track_widths
        ).astype(jnp.int32)

        no_opponent_mask = x_positions == -1
        y_positions_final = jnp.where(no_opponent_mask, -1, y_positions)
        result = jnp.stack([x_positions, y_positions_final, current_colors], axis=1)
        return result

    @partial(jax.jit, static_argnums=(0,))
    def _adjust_opponent_positions_when_overtaking(self, state: Enduro2GameState, new_opponent_index: jnp.ndarray, 
                                                  visible_track_left: jnp.ndarray, visible_track_right: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        new_integer_index = jnp.floor(new_opponent_index).astype(jnp.int32)
        previous_integer_index = jnp.floor(state.opponent_index).astype(jnp.int32)
        slots_retreated = jnp.maximum(0, previous_integer_index - new_integer_index)

        def adjust_opponents():
            player_row = (state.player_y - self.consts.sky_height).astype(jnp.int32)
            left_boundary = visible_track_left[player_row]
            right_boundary = visible_track_right[player_row]
            track_width = right_boundary - left_boundary

            player_left_ratio = (state.player_x - left_boundary) / track_width
            player_right_ratio = (state.player_x + self.consts.car_width_0 - left_boundary) / track_width

            player_in_lane_0 = (player_left_ratio < 1 / 3) & (player_right_ratio > 0)
            player_in_lane_1 = (player_left_ratio < 2 / 3) & (player_right_ratio > 1 / 3)
            player_in_lane_2 = (player_left_ratio < 1.0) & (player_right_ratio > 2 / 3)

            opponent_array_size = self.consts.base_opponents.shape[1]
            slot_0_index = new_integer_index % opponent_array_size
            
            base_lane = self.consts.base_opponents[0, slot_0_index]
            slot_0_lane = jnp.where((slot_0_index == state.adjusted_opponent_index) & (state.adjusted_opponent_lane != -1),
                                    state.adjusted_opponent_lane, base_lane)

            would_collide = ((slot_0_lane == 0) & player_in_lane_0) | \
                            ((slot_0_lane == 1) & player_in_lane_1) | \
                            ((slot_0_lane == 2) & player_in_lane_2)

            safe_lane = jnp.where(~player_in_lane_0, 0, jnp.where(~player_in_lane_2, 2, 1))

            needs_adjustment = would_collide & (slot_0_lane != -1)
            new_lane = jnp.where(needs_adjustment, safe_lane, slot_0_lane)
            
            return jnp.where(needs_adjustment, slot_0_index, state.adjusted_opponent_index), \
                   jnp.where(needs_adjustment, new_lane, state.adjusted_opponent_lane)

        return lax.cond(
            slots_retreated > 0,
            adjust_opponents,
            lambda: (state.adjusted_opponent_index, state.adjusted_opponent_lane)
        )

    @partial(jax.jit, static_argnums=(0,))
    def _check_car_opponent_collision_optimized(
            self,
            player_car_x: jnp.int32,
            player_car_y: jnp.int32,
            visible_opponents: jnp.ndarray
    ) -> jnp.ndarray:
        car_0_x = visible_opponents[0, 0]
        car_0_y = visible_opponents[0, 1]
        car_1_x = visible_opponents[1, 0]
        car_1_y = visible_opponents[1, 1]

        car_0_exists = car_0_x != -1
        car_1_exists = car_1_x != -1

        player_car_width = 16
        car_0_width = 16
        car_1_width = 12 # from car_widths[1]
        car_height = 8

        def check_car_0():
            player_left = player_car_x
            player_right = player_car_x + player_car_width
            car_0_left = car_0_x
            car_0_right = car_0_x + car_0_width
            return (player_left < car_0_right) & (car_0_left < player_right)

        def check_car_1():
            x_distance = jnp.abs(player_car_x - car_1_x)
            y_distance = jnp.abs(player_car_y - car_1_y)
            too_far = (x_distance >= car_1_width) | (y_distance >= car_height)

            def detailed_check():
                player_left = player_car_x
                player_right = player_car_x + player_car_width
                car_1_left = car_1_x
                car_1_right = car_1_x + car_1_width
                x_overlap = (player_left < car_1_right) & (car_1_left < player_right)

                player_top = player_car_y
                player_bottom = player_car_y + car_height
                car_1_top = car_1_y
                car_1_bottom = car_1_y + car_height
                y_overlap = (player_top < car_1_bottom) & (car_1_top < player_bottom)

                basic_collision = x_overlap & y_overlap
                car_1_y_int = car_1_y.astype(jnp.int32)
                is_special_y = (car_1_y_int == 133) | (car_1_y_int == 134)
                overlap_left = jnp.maximum(player_left, car_1_left)
                overlap_right = jnp.minimum(player_right, car_1_right)
                overlap_pixels = jnp.maximum(0, overlap_right - overlap_left)
                exception = is_special_y & (overlap_pixels <= 2)
                return basic_collision & ~exception

            return jnp.where(too_far, False, detailed_check())

        collision_0 = jnp.where(car_0_exists, check_car_0(), False)
        collision_1 = jnp.where(car_1_exists, check_car_1(), False)
        return collision_0 | collision_1

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: Enduro2GameState, action: int) -> Tuple[Enduro2Observation, Enduro2GameState, float, bool, Enduro2Info]:
        # Translate compact agent action index to ALE console action
        atari_action = jnp.take(self.ACTION_SET, action.astype(jnp.int32))

        # 1. Handle Speed (Acceleration and Braking)
        is_fire = (atari_action == Action.FIRE) | (atari_action == Action.LEFTFIRE) | (atari_action == Action.RIGHTFIRE)
        is_down = (atari_action == Action.DOWN) | (atari_action == Action.DOWNLEFT) | (atari_action == Action.DOWNRIGHT)
        
        # Acceleration: if speed <= 45: +2, else +1
        accel_amount = jnp.where(state.player_speed <= self.consts.acceleration_slow_down_threshold, 2.0, 1.0)
        
        # Speed delta: -1 every step if DOWN, +accel every 8 steps if FIRE, otherwise 0
        speed_delta = jnp.where(
            is_down,
            -1.0,
            jnp.where(
                is_fire & (state.step_count % 8 == 0),
                accel_amount,
                0.0
            )
        )
        
        new_speed = jnp.clip(state.player_speed + speed_delta, self.consts.min_speed, self.consts.max_speed)

        # 2. Handle Steering and Drift
        is_left = (atari_action == Action.LEFT) | (atari_action == Action.LEFTFIRE) | (atari_action == Action.DOWNLEFT)
        is_right = (atari_action == Action.RIGHT) | (atari_action == Action.RIGHTFIRE) | (atari_action == Action.DOWNRIGHT)

        # Steering delta (speed-dependent)
        steering_speed = self.consts.horizontal_movement_slope * new_speed + self.consts.horizontal_movement_offset
        
        # Add snow steering effect (slower steering)
        steering_speed = steering_speed / jnp.where(
            state.weather_index == self.consts.snow_weather_index,
            self.consts.steering_snow_factor,
            1.0
        )
        
        steering_delta = jnp.where(is_left, -steering_speed, jnp.where(is_right, steering_speed, 0.0))

        # Centripetal drift (pushes car away from curve proportional to visual curvature and speed)
        # Curve ratio is from -1.0 to 1.0 based on current visual track curve
        curve_ratio = state.track_top_x_curve_offset / self.consts.track_max_top_x_offset
        drift_delta = -curve_ratio * (new_speed / self.consts.centripedal_shift_ratio)

        # 3. Handle Collision and Horizontal Movement
        offset_int = jnp.clip(
            jnp.floor(state.track_top_x_curve_offset).astype(jnp.int32),
            -self.consts.curve_offset_base,
            self.consts.curve_offset_base
        )
        curve_index = offset_int + self.consts.curve_offset_base
        base_left_curve = self.consts.precomputed_left_curves[curve_index]
        base_right_curve = self.consts.precomputed_right_curves[curve_index]
        
        visible_track_left = base_left_curve + state.track_top_x
        visible_track_right = base_right_curve + state.track_top_x
        
        player_row = (state.player_y - self.consts.sky_height).astype(jnp.int32)
        l_x = visible_track_left[player_row]
        r_x = visible_track_right[player_row]

        # ====== OPPONENT MOVEMENT AND OVERTAKING ======
        base_progression_rate = self.consts.opponent_relative_speed_factor / self.consts.frame_rate
        relative_speed = (new_speed - self.consts.opponent_speed) / self.consts.opponent_speed * base_progression_rate
        new_opponent_index = state.opponent_index + relative_speed

        new_adjusted_idx, new_adjusted_lane = self._adjust_opponent_positions_when_overtaking(state, new_opponent_index, visible_track_left, visible_track_right)

        new_visible_opponent_positions = self._get_visible_opponent_positions(
            new_opponent_index,
            self.consts.base_opponents,
            new_adjusted_idx,
            new_adjusted_lane,
            visible_track_left,
            visible_track_right
        )

        # Overtaking detection
        old_window_start = jnp.floor(state.opponent_index).astype(jnp.int32)
        new_window_start = jnp.floor(new_opponent_index).astype(jnp.int32)
        window_moved = new_window_start - old_window_start

        cars_overtaken_change = jnp.where(
            (window_moved > 0) & (state.visible_opponent_positions[0, 0] > -1),
            1, 0
        )
        cars_overtaken_change -= jnp.where(
            (window_moved < 0) & (new_visible_opponent_positions[0, 0] > -1),
            1, 0
        )
        new_cars_to_pass = jnp.maximum(0, state.cars_to_pass - cars_overtaken_change)

        # ====== COLLISIONS ======
        def collision_update(_):
            new_collision_steps = state.collision_steps - 1
            new_collision_mode = new_collision_steps > 0
            # Ignore acceleration/braking during collision push-back
            coll_speed = state.player_speed
            push_direction = jnp.where(state.player_x < self.consts.player_x_start, 1.0, -1.0)
            coll_player_x = state.player_x + push_direction * self.consts.collision_push_back
            # During collision, move back to base y
            coll_player_y = self.consts.player_y_start
            return coll_player_x, coll_player_y, coll_speed, new_collision_mode, new_collision_steps

        def normal_update(_):
            # Apply horizontal movement
            normal_player_x = jnp.clip(state.player_x + steering_delta + drift_delta, 0.0, self.consts.screen_width - 16.0)
            
            # Update player_y based on speed: move up to 10 pixels forward as speed increases
            normal_player_y = self.consts.player_y_start - jnp.floor(new_speed / (self.consts.max_speed / 10.0))

            # Side Collision
            is_side_colliding = (normal_player_x <= l_x) | (normal_player_x + 16.0 >= r_x)
            
            # Opponent Collision
            is_opponent_colliding = self._check_car_opponent_collision_optimized(
                normal_player_x.astype(jnp.int32),
                normal_player_y.astype(jnp.int32),
                new_visible_opponent_positions
            )
            
            is_colliding = is_side_colliding | is_opponent_colliding
            
            return jax.lax.cond(
                is_colliding,
                lambda: (state.player_x, state.player_y, jnp.where(is_opponent_colliding, jnp.minimum(new_speed * self.consts.side_collision_speed_drop, 6.0), new_speed * self.consts.side_collision_speed_drop), True, jnp.array(self.consts.collision_duration, dtype=jnp.int32)),
                lambda: (normal_player_x, normal_player_y, new_speed, False, jnp.array(0, dtype=jnp.int32))
            )

        new_player_x, new_player_y, new_speed, new_collision_mode, new_collision_steps = jax.lax.cond(
            state.collision_mode,
            collision_update,
            normal_update,
            None
        )

        # 4. Handle Track Curvature (Top X and Offset)
        # Track top_x moves opposite to player movement on X axis
        new_track_top_x = 84.0 + self.consts.player_x_start - new_player_x

        # Get curvature from whole_track for track bending
        track_starts = self.consts.whole_track[:, 1]
        directions = self.consts.whole_track[:, 0]
        segment_index = jnp.searchsorted(track_starts, state.distance, side='right') - 1
        curvature = directions[segment_index]

        # ====== MOUNTAINS ======
        horizon_shift = new_track_top_x - state.track_top_x
        mountain_movement = -curvature * self.consts.mountain_pixel_movement_per_frame_per_speed_unit * new_speed + horizon_shift
        
        # Wrapping logic
        lo = self.consts.window_offset_left
        hi = self.consts.screen_width
        period = hi - lo + 1
        
        new_mountain_left_x = lo + jnp.mod(state.mountain_left_x + mountain_movement - lo, period)
        new_mountain_right_x = lo + jnp.mod(state.mountain_right_x + mountain_movement - lo, period)

        # Target offset based on curvature
        target_offset = curvature * self.consts.track_max_top_x_offset
        current_offset = target_offset - state.track_top_x_curve_offset
        
        # Speed-adjusted curve rate
        speed_multiplier = 1 + (new_speed - self.consts.min_speed) / (self.consts.max_speed - self.consts.min_speed) * 2.0
        adjusted_curve_rate = self.consts.curve_rate * speed_multiplier
        
        offset_change = jnp.clip(current_offset, -adjusted_curve_rate, adjusted_curve_rate)
        new_top_x_curve_offset = state.track_top_x_curve_offset + offset_change

        # 4. Update distance
        distance_delta = new_speed * self.consts.km_per_speed_unit_per_frame
        new_distance = state.distance + distance_delta

        # 5. Update Weather and Day Cycle
        new_step_count = state.step_count + 1

        # Calculate time-based weather and day
        cycled_time = (new_step_count / self.consts.frame_rate) % self.consts.day_cycle_time
        time_weather_index = jnp.searchsorted(self.consts.weather_starts_s, cycled_time, side='right')
        time_day_count = jnp.floor(new_step_count / self.consts.frame_rate / self.consts.day_cycle_time).astype(jnp.int32)

        # Calculate distance-based weather and day
        # In distance mode, weather cycles every 16km, changing index every 1km
        dist_weather_index = (new_distance // 1.0).astype(jnp.int32) % 16
        dist_day_count = (new_distance // 16.0).astype(jnp.int32)

        # Use distance-based if configured
        use_dist = self.consts.weather_cycle_distance > 0.0

        new_weather_index = jax.lax.select(use_dist, dist_weather_index, time_weather_index)
        new_day_count = jax.lax.select(use_dist, dist_day_count, time_day_count)

        # Day transition and level logic        # Check if the player passed the level (cars_to_pass reaches 0)
        new_level_passed = jnp.logical_or(
            state.level_passed,
            new_cars_to_pass <= 0
        ).astype(jnp.int32)

        def reset_day():
            # If day advances and level is passed, advance to next level.
            # If day advances and level is not passed, game over!
            is_game_over = jnp.logical_not(new_level_passed).astype(jnp.bool_)
            level = jnp.where(is_game_over, state.level, state.level + 1)
            # When advancing level, reset cars to pass. For simplicity here, just using a fixed 200/300 etc if needed
            # Or just keep it 200 for now. Original enduro increases it by 100 per level up to max.
            cars_increase_per_level = 100
            cars_to_pass = jnp.where(is_game_over, new_cars_to_pass, 200 + cars_increase_per_level * (level - 1))
            return level, jnp.array(0, dtype=jnp.int32), is_game_over, cars_to_pass

        def do_nothing():
            return state.level, new_level_passed, state.game_over, new_cars_to_pass

        new_level, final_level_passed, new_game_over, final_cars_to_pass = lax.cond(
            new_day_count > state.day_count,
            reset_day,
            do_nothing,
        )

        new_state = state.replace(
            player_x=new_player_x,
            player_y=new_player_y,
            player_speed=new_speed,
            distance=new_distance,
            step_count=new_step_count,
            day_count=new_day_count,
            level=new_level,
            level_passed=final_level_passed,
            game_over=new_game_over,
            weather_index=new_weather_index,
            track_top_x=new_track_top_x,
            track_top_x_curve_offset=new_top_x_curve_offset,
            collision_mode=new_collision_mode,
            collision_steps=new_collision_steps,
            mountain_left_x=new_mountain_left_x,
            mountain_right_x=new_mountain_right_x,
            opponent_index=new_opponent_index,
            visible_opponent_positions=new_visible_opponent_positions,
            adjusted_opponent_index=new_adjusted_idx,
            adjusted_opponent_lane=new_adjusted_lane,
            cars_to_pass=final_cars_to_pass
        )

        obs = self._get_observation(new_state)
        reward = self._get_reward(state, new_state)
        done = self._get_done(new_state)
        info = self._get_info(new_state)

        return obs, new_state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, state: Enduro2GameState, new_state: Enduro2GameState) -> float:
        cars_overtaken = (state.cars_to_pass - new_state.cars_to_pass).astype(jnp.float32)
        distance_reward = new_state.distance - state.distance - self.consts.km_per_speed_unit_per_frame
        return cars_overtaken + distance_reward

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: Enduro2GameState) -> bool:
        return state.game_over

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: Enduro2GameState) -> Enduro2Info:
        return Enduro2Info()

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: Enduro2GameState) -> jnp.ndarray:        return self.renderer.render(state)

    def _get_observation(self, state: Enduro2GameState) -> Enduro2Observation:
        return Enduro2Observation()
