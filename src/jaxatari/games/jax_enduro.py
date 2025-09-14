import chex
from functools import partial
import jax
from jax import numpy as jnp, lax
import jax.random as jrandom
import numpy as np
import os
from pathlib import Path
from typing import Tuple, NamedTuple, Any

# jaxatari
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as aj
import jaxatari.spaces as spaces

TODOS = """
Config:

Opponents:

Steering:
- Drift speed based?

Rendering:
- wheel animation speed

Track:
- Better curve
- lost x pixel of right track
- bumpers
- curve move speed based on speed

Game Logic:

Observation:
- improve observation function 
- reduce observation when fog


"""


class EnduroConstants(NamedTuple):
    """Game configuration parameters"""
    # Game runs at 60 frames per second. This is used to approximate the configs with values from play-testing the game.
    # Only change this variable if you are sure the original Enduro implementation ran at a lower rate!
    frame_rate: int = 60

    # ====================
    # === Window Sizes ===
    # ====================
    screen_width: int = 160
    screen_height: int = 210

    # Enduro has a game window that is smaller
    window_offset_left: int = 8
    window_offset_bottom: int = 55
    game_window_height: int = screen_height - window_offset_bottom
    game_window_width: int = screen_width - window_offset_left
    game_screen_middle: int = game_window_width // 2

    # the track is in the game window below the sky
    sky_height = 50

    # ============
    # === Cars ===
    # ============
    # car sizes from close to far
    car_width_0: int = 16
    car_height_0: int = 11

    # for all different car sizes the widths and heights
    car_widths = jnp.array([16, 12, 8, 6, 4, 4, 2], dtype=jnp.int32)
    car_heights = jnp.array([11, 8, 6, 4, 3, 2, 1], dtype=jnp.int32)

    # player car start position
    player_x_start: float = game_screen_middle
    player_y_start: float = game_window_height - car_height_0 - 1

    # =============
    # === Track ===
    # =============
    track_width: int = 97
    track_height: int = game_window_height - sky_height - 2
    max_track_length: float = 9999.9  # in km
    track_seed: int = 42
    min_track_section_length = 1.0  # how long a curve or straight passage is at least
    max_track_section_length = 15.0
    track_x_start: int = player_x_start
    track_max_curvature_width: int = 17
    # How many pixels the top-x of the track moves in a curve into the curve direction for the full curve
    track_max_top_x_offset: float = 50.0
    # how fast the track curve starts to build in the game when going from a straight track into a curve
    curve_rate: float = 0.25

    # track colors
    track_colors = jnp.array([
        [74, 74, 74],  # top
        [111, 111, 111],  # moving top - movement range: track_move_range
        [170, 170, 170],  # moving bottom - spawns after track_moving_bottom_spawn_step
        [192, 192, 192],  # bottom - rest
    ], dtype=jnp.int32)
    track_top_min_length: int = 33  # or 32
    track_moving_top_length: int = 13
    track_moving_bottom_length: int = 18
    track_move_range: int = 12
    track_moving_bottom_spawn_step: int = 6
    track_color_move_speed_per_speed: float = 0.05
    track_speed_animation_factor: float = 0.085  # determines how fast the animation speed increases

    # === Track collision ===
    track_collision_kickback_pixels: float = 3.0
    track_collision_speed_reduction_per_speed_unit: float = 0.25  # from RAM extraction

    # ======================
    # === Speed controls ===
    # ======================
    min_speed: int = 6  # from RAM state 22
    max_speed: int = 120  # from RAM state 22
    # measured by starting the original game and letting the car progress with min speed for 5 km --> 2:23 min
    # 1/ 143 seconds / 5 km =~ 0.035
    km_per_second_per_speed_unit: float = 0.035 / min_speed
    km_per_speed_unit_per_frame: float = km_per_second_per_speed_unit / frame_rate

    # The acceleration per second (as frame rate)
    acceleration_per_frame: float = 10.5 / frame_rate
    slower_acceleration_per_frame: float = 3.75 / frame_rate
    # at which speed the slower_acceleration is applied
    acceleration_slow_down_threshold: float = 46.0

    breaking_per_second: float = 30.0  # controls how fast the car break
    breaking_per_frame: float = breaking_per_second / frame_rate

    # ================
    # === Steering ===
    # ================
    # how many pixels the car can move from one edge of the track to the other one
    steering_range_in_pixels: int = 28
    # How much the car moves per steering input (absolute units)
    steering_sensitivity: float = steering_range_in_pixels / 3.0 / frame_rate

    # with increasing speed the car moves faster on the x-axis.
    # When moving faster than sensitivity_change_speed the sensitivity rate becomes lower
    # sensitivity(speed) = steering_range_in_pixels / (base_sensitivity + sensitivity_per_speed * speed) / frame_rate
    slow_base_sensitivity: float = 8.0
    fast_base_sensitivity: float = 4.86
    slow_steering_sensitivity_per_speed_unit: float = -0.15  # speed <= 32
    fast_steering_sensitivity_per_speed_unit: float = -0.056  # speed > 32
    sensitivity_change_speed: int = 32
    minimum_steering_sensitivity: float = 1.0  # from play-testing
    steering_snow_factor: float = 2.0  # during snow the steering becomes much worse

    drift_per_second_pixels: float = 2.5  # controls how much the car drifts in a curve
    drift_per_frame: float = drift_per_second_pixels / frame_rate

    # ===============
    # === Weather ===
    # ===============
    # Start times in seconds for each phase. Written in a way to allow easy replacements.
    weather_starts_s: jnp.ndarray = jnp.array([
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
    # weather_starts_s: jnp.ndarray = jnp.arange(0, 32, 2, dtype=jnp.int32)  # for debugging
    # special events in the weather:
    snow_weather_index: int = 3  # which part of the weather array is snow (reduced steering)
    night_fog_index: int = 13  # which part of the weather array has the reduced visibility (fog)
    weather_with_night_car_sprite = jnp.array([12, 13, 14], dtype=jnp.int32)  # renders only the back lights
    day_cycle_time: int = weather_starts_s[15]

    # The rgb color codes for each weather and each sprite scraped from the game
    weather_color_codes: jnp.ndarray = jnp.array([
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

    # =================
    # === Opponents ===
    # =================
    opponent_speed: int = 24  # measured from RAM state
    # a factor of 1 translates into overtake time of 1 second when speed is twice as high as the opponent's
    opponent_relative_speed_factor: float = 2.5

    opponent_spawn_seed: int = 42

    length_of_opponent_array = 5000
    opponent_density = 0.3
    opponent_delay_slots = 10

    # How many opponents to overtake to progress into the next level
    cars_to_pass_per_level: int = 200
    cars_increase_per_level: int = 100
    max_increase_level: int = 5

    # defines how many y pixels the car size will have size 0
    car_zero_y_pixel_range = 20

    # slots where the equivalent car size is rendered.
    # It is written this way to easily see how many pixels each slot has and replacements are easier
    opponent_slot_ys = jnp.array([
        game_window_height - car_zero_y_pixel_range,
        game_window_height - car_zero_y_pixel_range - 20,
        game_window_height - car_zero_y_pixel_range - 20 - 20,
        game_window_height - car_zero_y_pixel_range - 20 - 20 - 10,
        game_window_height - car_zero_y_pixel_range - 20 - 20 - 10 - 10,
        game_window_height - car_zero_y_pixel_range - 20 - 20 - 10 - 10 - 6,
        game_window_height - car_zero_y_pixel_range - 20 - 20 - 10 - 10 - 6 - 5,
    ], dtype=jnp.int32)

    # Opponent lane position
    # The ratio of where in the track the opponents are rendered. From left, middle to right
    lane_ratios = jnp.array([0.25, 0.5, 0.75], dtype=jnp.float32)

    # ===========================
    # === Opponents Collision ===
    # ===========================
    car_crash_cooldown_seconds: float = 3.0
    car_crash_cooldown_frames: int = jnp.array(car_crash_cooldown_seconds * frame_rate)
    crash_kickback_speed_per_frame: float = track_width / car_crash_cooldown_seconds / frame_rate / 3

    # =================
    # === Cosmetics ===
    # =================
    logo_x_position: int = 20
    logo_y_position: int = 196

    info_box_x_pos: int = 48
    info_box_y_pos: int = 161

    distance_odometer_start_x: int = 65
    distance_odometer_start_y: int = game_window_height + 9

    score_start_x: int = 81
    score_start_y: int = game_window_height + 25

    level_x: int = 57
    level_y: int = score_start_y

    mountain_left_x_pos: float = 40.0
    mountain_right_x_pos: float = 120.0
    mountain_pixel_movement_per_frame_per_speed_unit: float = 0.01

    # how many steps per animation
    opponent_animation_steps: int = 8


class EnduroGameState(NamedTuple):
    """Represents the current state of the game"""

    step_count: jnp.int32  # incremented every step
    day_count: jnp.int32  # incremented every day-night cycle, starts by 0

    # visible (mirror in Observation)
    player_y_abs_position: chex.Array
    player_x_abs_position: chex.Array
    cars_overtaken: chex.Array
    cars_to_overtake: chex.Array  # goal for current level
    distance: chex.Array
    level: chex.Array
    level_passed: chex.Array

    # opponents
    opponent_pos_and_color: chex.Array  # shape (N, 2) where [:, 0] is x, [:, 1] is y
    visible_opponent_positions: chex.Array
    opponent_index: chex.Array
    opponent_window: chex.Array
    is_collision: chex.Array

    # visible but implicit
    weather_index: chex.Array
    mountain_left_x: chex.Array
    mountain_right_x: chex.Array
    cooldown_drift_direction: chex.Array

    # track
    track_top_x: chex.Array
    track_top_x_curve_offset: chex.Array  # The amount that the top_x moves further into the curve direction
    visible_track_left: chex.Array  # shape: (track_height,), dtype=int32 the absolute x position of the left track
    visible_track_right: chex.Array  # shape: (track_height,), dtype=int32 the absolute x position of the right track
    visible_track_spaces: chex.Array  # shape: (track_height,), dtype=int32, the spaces between the left and right track

    # invisible
    whole_track: chex.Array  # shape (N, 2), where track[i] = [direction, start_km]
    player_speed: chex.Array
    cooldown: chex.Array  # cooldown after collision with another car
    game_over: chex.Array  # game over if you fail to pass enough cars before the day ends
    time_remaining: chex.Array
    total_cars_overtaken: chex.Array
    total_time_elapsed: chex.Array
    total_frames_elapsed: chex.Array


class VehicleSpec:
    """
    Holds all static specifications for a vehicle type (e.g., the player's car).
    This includes all possible sprites for perspective scaling and their masks.
    This data is created once at the start and does not change.
    """

    def __init__(self, sprite_path_car):
        """
        Args:
            sprite_path_car: Path to the .npy file for the largest sprite,
                                  which may contain multiple animation frames.
        """
        # load full path
        module_dir = os.path.dirname(os.path.abspath(__file__))
        sprite_path = Path(sprite_path_car)
        if not sprite_path.is_absolute():
            sprite_path = module_dir / sprite_path

        # Load the sprite with the absolute path
        largest_sprite_data = np.load(str(sprite_path))

        # --- Create the Union Collision Mask ---

        # 1. Initialize an empty boolean mask with the correct dimensions (H, W).
        height, width, _ = largest_sprite_data.shape[1:]
        union_mask = np.zeros((height, width), dtype=bool)

        # 2. Iterate through each animation frame using a standard Python loop.
        #    This is fine because this is a one-time setup operation, not part of the JIT path.
        for frame in largest_sprite_data:
            # Determine which pixels are solid in the current frame (white is transparent).
            is_solid_in_frame = np.sum(frame[:, :, :3], axis=-1) < (255 * 3)

            # Use a logical OR to add the solid pixels from this frame to our union_mask.
            # If a pixel is solid in *any* frame, it will become True in the union_mask.
            union_mask = np.logical_or(union_mask, is_solid_in_frame)

        # 3. The `union_mask` is now our final, most reliable collision mask.
        #    Convert it to a JAX array.
        self.collision_mask = jnp.array(union_mask)

        # --- Pre-calculate Collision Coordinates ---
        # Pre-calculate the relative coordinates from this robust union mask.
        solid_y, solid_x = jnp.where(
            self.collision_mask,
            size=self.collision_mask.size,  # Guarantee static shape
            fill_value=-1  # Fill with an invalid value
        )
        self.collision_mask_relative_xs = solid_x
        self.collision_mask_relative_ys = solid_y
        self.num_solid_pixels = jnp.sum(self.collision_mask).astype(jnp.int32)
        self.height = height
        self.width = width


class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray


class EnduroObservation(NamedTuple):
    # cars
    car: EntityPosition  # player car position
    visible_opponents: chex.Array

    # score box
    cars_to_overtake: jnp.ndarray  # goal for current level
    distance: jnp.ndarray
    level: jnp.ndarray

    # track
    track_top_x: chex.Array
    track_left_xs: chex.Array
    track_right_xs: chex.Array
    curvature: jnp.ndarray  # one of -1, 0, or 1


class EnduroInfo(NamedTuple):
    distance: jnp.ndarray
    level: jnp.ndarray


StepResult = Tuple[EnduroObservation, EnduroGameState, jnp.ndarray, bool, EnduroInfo]


# https://www.free80sarcade.com/atari2600_Enduro.php
class JaxEnduro(JaxEnvironment[EnduroGameState, EnduroObservation, EnduroInfo, EnduroConstants]):
    def __init__(self):
        super().__init__()
        self.frame_stack_size = 4
        self.config = EnduroConstants()
        self.state = self.reset()
        self.car_0_spec = VehicleSpec("sprites/enduro/cars/car_0.npy")
        self.car_1_spec = VehicleSpec("sprites/enduro/cars/car_1.npy")

        self.action_set = [
            Action.NOOP,
            Action.FIRE,  # gas
            Action.LEFT,  # steer left
            Action.RIGHT,  # steer right
            Action.DOWN,  # brake
            Action.LEFTFIRE,  # steer left + gas
            Action.RIGHTFIRE,  # steer right + gas
            Action.DOWNFIRE  # brake + gas (will just override to gas)
        ]

        self.renderer = EnduroRenderer()

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(210, 160, 3),
            dtype=jnp.uint8
        )

    def observation_space(self) -> spaces.Dict:
        return spaces.Dict({
            "car": spaces.Dict({
                "x": spaces.Box(low=0, high=self.config.screen_width, shape=(1, 1), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=self.config.screen_height, shape=(1, 1), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=self.config.screen_width, shape=(1, 1), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=self.config.screen_height, shape=(1, 1), dtype=jnp.int32),
            }),
            "visible_opponents": spaces.Box(
                low=0,
                high=1,
                shape=(self.config.length_of_opponent_array,),
                dtype=jnp.int32
            ),
            "cars_to_overtake": spaces.Box(low=0, high=1000, shape=(1,), dtype=jnp.int32),
            "distance": spaces.Box(low=0.0, high=self.config.max_track_length, shape=(1,), dtype=jnp.float32),
            "level": spaces.Box(low=1, high=10, shape=(1,), dtype=jnp.int32),
            "track_top_x": spaces.Box(low=0, high=self.config.screen_width, shape=(1,), dtype=jnp.int32),
            "track_left_xs": spaces.Box(
                low=0,
                high=self.config.screen_width,
                shape=(self.config.track_height,),
                dtype=jnp.int32
            ),
            "track_right_xs": spaces.Box(
                low=0,
                high=self.config.screen_width,
                shape=(self.config.track_height,),
                dtype=jnp.int32
            ),
            "curvature": spaces.Box(low=-1, high=1, shape=(1,), dtype=jnp.int32),
        })

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: EnduroObservation) -> jnp.ndarray:
        return jnp.concatenate([
            # cars
            obs.car.x.flatten(),
            obs.car.y.flatten(),
            obs.car.height.flatten(),
            obs.car.width.flatten(),

            obs.visible_opponents.flatten(),

            # score box
            obs.cars_to_overtake.flatten(),
            obs.distance.flatten(),
            obs.level.flatten(),

            # track
            obs.track_top_x.flatten(),
            obs.track_left_xs.flatten(),
            obs.track_right_xs.flatten(),
            obs.curvature.flatten(),
        ]
        )

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: EnduroGameState) -> jnp.ndarray:
        return self.renderer.render(state)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jrandom.PRNGKey = None) -> Tuple[EnduroObservation, EnduroGameState]:
        whole_track = self._build_whole_track(seed=self.config.track_seed)
        # use same position as the player
        top_x = jnp.round(self.config.player_x_start).astype(jnp.int32)
        left_xs, right_xs = self._generate_viewable_track(top_x, 0.0)

        # opponents
        opponent_spawns = self._generate_opponent_spawns(
            seed=self.config.opponent_spawn_seed,
            length_of_opponent_array=self.config.length_of_opponent_array,
            opponent_density=self.config.opponent_density,
            opponent_delay_slots=self.config.opponent_delay_slots
        )
        visible_opponent_positions = self._get_visible_opponent_positions(jnp.array(0.0), opponent_spawns, left_xs,
                                                                          right_xs)

        state = EnduroGameState(
            # visible
            step_count=jnp.array(0),
            day_count=jnp.array(0),
            player_x_abs_position=jnp.array(self.config.player_x_start),
            player_y_abs_position=jnp.array(self.config.player_y_start),
            cars_to_overtake=jnp.array(self.config.cars_to_pass_per_level),
            cars_overtaken=jnp.array(0),
            distance=jnp.array(0.0, dtype=jnp.float32),
            level=jnp.array(1),
            level_passed=jnp.array(False),

            # opponents
            opponent_pos_and_color=opponent_spawns,
            visible_opponent_positions=visible_opponent_positions,
            opponent_index=jnp.array(0.0),
            opponent_window=jnp.array(0.0),
            is_collision=jnp.bool_(False),

            # visible but implicit
            weather_index=jnp.array(0),
            mountain_left_x=jnp.array(self.config.mountain_left_x_pos),
            mountain_right_x=jnp.array(self.config.mountain_right_x_pos),
            cooldown_drift_direction=jnp.array(0),

            # track
            track_top_x=jnp.array(0),
            track_top_x_curve_offset=jnp.array(0),
            visible_track_left=left_xs,
            visible_track_right=right_xs,
            visible_track_spaces=self._generate_track_spaces(),
            whole_track=whole_track,

            player_speed=jnp.array(0.0, dtype=jnp.float32),
            cooldown=jnp.array(0.0, dtype=jnp.float32),
            game_over=jnp.array(False),
            time_remaining=jnp.array(self.config.day_cycle_time),
            total_cars_overtaken=jnp.array(0),
            total_time_elapsed=jnp.array(0.0, dtype=jnp.float32),
            total_frames_elapsed=jnp.array(0),
        )

        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: EnduroGameState, action: int) -> StepResult:
        """
        Performs a single frame update of the Enduro environment.

        Applies the action to update the player's position and increments the
        total frame counter. Computes the new observation, reward, and done state.

        Args:
            state (EnduroGameState): The current game state.
            action (int): The discrete action to apply (e.g., LEFT, RIGHT, etc.).

        Returns:
            Tuple[EnduroObservation, EnduroGameState, float, bool, EnduroInfo]:
                - The new observation after the action.
                - The updated game state.
                - The reward for this step.
                - A boolean indicating if the episode has ended.
                - Additional info such as level and distance.
        """
        # ====== COOLDOWN MANAGEMENT ======
        # Always decrement cooldown first
        new_cooldown = jnp.maximum(0, state.cooldown - 1)
        is_cooldown_active = state.cooldown > 0

        # ===== Track position =====
        directions = state.whole_track[:, 0]
        track_starts = state.whole_track[:, 1]
        segment_index = jnp.searchsorted(track_starts, state.distance, side='right') - 1
        curvature = directions[segment_index]

        # ===== Weather position =====
        # determine the position in the weather array
        cycled_time = (state.step_count / self.config.frame_rate) % self.config.day_cycle_time
        new_weather_index = jnp.searchsorted(
            self.config.weather_starts_s,
            cycled_time,
            side='right')

        def regular_handling() -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            # ====== GAS ======
            is_gas = jnp.isin(action, jnp.array([
                Action.FIRE,
                Action.LEFTFIRE,
                Action.RIGHTFIRE,
                Action.DOWNFIRE  # explicitly included, as it implies FIRE
            ]))

            # ====== BRAKE (only when DOWN) ======
            is_brake = (action == Action.DOWN)

            # Final speed delta
            speed_delta = jnp.where(
                is_gas,
                # accelerate according to the current speed
                jnp.where(
                    state.player_speed < self.config.acceleration_slow_down_threshold,
                    self.config.acceleration_per_frame,
                    self.config.slower_acceleration_per_frame
                )
                ,
                jnp.where(is_brake, -self.config.breaking_per_frame, 0.0)
            )

            speed = jnp.clip(state.player_speed + speed_delta, self.config.min_speed, self.config.max_speed)

            # ====== STEERING ======
            # Determine if action is a left-turn
            is_left = jnp.logical_or(action == Action.LEFT, action == Action.LEFTFIRE)
            # Determine if action is a right-turn
            is_right = jnp.logical_or(action == Action.RIGHT, action == Action.RIGHTFIRE)

            # calculate the time it should take to steer from left to right (seconds) based on the current speed
            time_from_left_to_right = jnp.clip(
                jnp.where(
                    speed > self.config.sensitivity_change_speed,
                    self.config.slow_base_sensitivity + self.config.slow_steering_sensitivity_per_speed_unit * speed,
                    self.config.fast_base_sensitivity + self.config.fast_steering_sensitivity_per_speed_unit * speed,
                ),
                self.config.minimum_steering_sensitivity  # never let the sensitivity go below a threshold
            )
            # add the snow effects when applicable
            time_from_left_to_right = time_from_left_to_right * jnp.where(
                new_weather_index == self.config.snow_weather_index,
                self.config.steering_snow_factor,
                1)

            # calculate the final steering sensitivity
            current_steering_sensitivity = (self.config.steering_range_in_pixels /
                                            time_from_left_to_right / self.config.frame_rate)
            # calculate the steering delta based on sensitivity and player input
            steering_delta = jnp.where(is_left, -1 * current_steering_sensitivity,
                                       jnp.where(is_right, current_steering_sensitivity, 0.0))

            # ====== DRIFT ======
            drift_delta = -curvature * self.config.drift_per_frame  # drift opposes curve

            # Combine steering and drift
            total_delta_x = steering_delta + drift_delta
            x_abs = state.player_x_abs_position + total_delta_x

            # ====== Car y-Position ======
            # move one pixel forward for every 5th speed increase
            y_abs = jnp.subtract(self.config.player_y_start, jnp.floor_divide(speed, self.config.max_speed / 10))

            return speed, x_abs, y_abs

        def cooldown_handling() -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            new_x = state.player_x_abs_position + state.cooldown_drift_direction * self.config.crash_kickback_speed_per_frame
            return (
                jnp.array(self.config.min_speed, dtype=jnp.float32),
                jnp.array(new_x, dtype=jnp.float32),
                jnp.array(self.config.player_y_start, dtype=jnp.float32)
            )

        new_speed, new_x_abs, new_y_abs = lax.cond(
            is_cooldown_active,
            lambda _: cooldown_handling(),
            lambda _: regular_handling(),
            None
        )

        # ====== TRACK ======
        # 1. Draw the top of the track based on the car position.
        #    The track moves in the opposite position of the car.
        new_track_top_x = (self.config.track_x_start + self.config.player_x_start - new_x_abs).astype(jnp.int32)

        # make sure the curve is not turning further than the bottom of the track
        # min_offset_allowed = jnp.min(state.visible_track_left) - new_track_top_x
        # max_offset_allowed = jnp.max(state.visible_track_right) - new_track_top_x

        # 2. Define the target offset based on the curvature.
        #    This is the value we want to eventually reach when the curve is fully curved.
        target_offset = curvature * self.config.track_max_top_x_offset  # e.g., -1 * 50 = -50, or 0 * 50 = 0

        # 3. Calculate the difference (the "offset") between where we are and where we want to be in terms of curvature
        current_offset = target_offset - state.track_top_x_curve_offset

        # 4. Limit the change per step. The change cannot be faster than curve_rate.
        #    jnp.clip is perfect here. This lets the offset move towards the target without overshooting.
        offset_change = jnp.clip(current_offset, -self.config.curve_rate, self.config.curve_rate)

        # 5. Apply the calculated change to the current offset.
        new_top_x_curve_offset = state.track_top_x_curve_offset + offset_change

        # 6. Generate the new track with the top_x of the track and its offset
        new_left_xs, new_right_xs = self._generate_viewable_track(new_track_top_x, new_top_x_curve_offset)

        # ====== TRACK COLLISION ======
        # 1. Check whether the player car collided with the track
        collision_side = self._check_car_track_collision(
            car_x_abs=new_x_abs.astype(jnp.int32),
            car_y_abs=new_y_abs.astype(jnp.int32),
            left_track_xs=new_left_xs,
            right_track_xs=new_right_xs
        )
        collided_track = (collision_side != 0)

        # 2. Calculate the speed with collision penalty.
        new_speed = jnp.where(
            collided_track,
            # If collided, reduce speed.
            new_speed - self.config.track_collision_speed_reduction_per_speed_unit * new_speed,
            new_speed  # If not, keep the new speed.
        )
        # Ensure speed does not drop below the minimum value
        new_speed = jnp.maximum(1.0, new_speed)  # Use maximum() to enforce a floor.

        # 3. Kickback
        # The kickback direction is simply the inverse of `collision_side`.
        track_kickback_direction = -collision_side
        # add a special treatment for cooldown where kickback is minimal
        kickback_pixels = jnp.where(is_cooldown_active, 1, self.config.track_collision_kickback_pixels)

        new_x_abs = jnp.where(
            collided_track,
            # Apply the kickback based on the actual collision side.
            new_x_abs + (kickback_pixels * track_kickback_direction),
            new_x_abs  # If not collided, do nothing.
        )

        # 4. Handle cooldowns
        new_cooldown_drift_direction = jnp.where(
            collided_track,
            # Change the cooldown drift direction if the car crashes into the track while in cooldown
            state.cooldown_drift_direction * -1,
            state.cooldown_drift_direction
        )

        # ====== Opponents ======
        # This should be calibrated so that at opponent_speed, we move at "normal" rate
        base_progression_rate = self.config.opponent_relative_speed_factor / self.config.frame_rate
        # Relative speed: how much faster/slower we are compared to opponents
        relative_speed = (new_speed - self.config.opponent_speed) / self.config.opponent_speed * base_progression_rate
        # calculate new the index where we are at the opponent array
        new_opponent_index = state.opponent_index + relative_speed

        # calculate the absolute positions of all opponents
        new_visible_opponent_positions = self._get_visible_opponent_positions(
            new_opponent_index,
            state.opponent_pos_and_color,
            new_left_xs, new_right_xs)

        # adjust the opponents lane if necessary
        adjusted_opponents_pos = self._adjust_opponent_positions_when_overtaking(state, new_opponent_index)
        state = state._replace(opponent_pos_and_color=adjusted_opponents_pos)

        # calculate which opponents are in the 7 visible opponent slots (for debugging only)
        new_opponent_window = state.opponent_pos_and_color[0][
            (jnp.floor(state.opponent_index).astype(jnp.int32)
             + jnp.arange(7)) % state.opponent_pos_and_color[0].shape[0]
            ]

        # ====== Overtaking ======
        # Simple overtaking logic
        old_window_start = jnp.floor(state.opponent_index).astype(jnp.int32)
        new_window_start = jnp.floor(new_opponent_index).astype(jnp.int32)
        window_moved = new_window_start - old_window_start

        cars_overtaken_change = 0
        # If we moved forward, check if we overtook a car (old slot 0 had a car)
        cars_overtaken_change += jnp.where(
            (window_moved > 0) & (state.visible_opponent_positions[0] > -1),
            1, 0
        )
        # If we moved backward, check if a car overtook us (new slot 0 has a car)
        cars_overtaken_change -= jnp.where(
            (window_moved < 0) & (new_visible_opponent_positions[0] > -1),
            1, 0
        )
        # don't allow negative numbers here
        new_cars_overtaken = jnp.clip(state.cars_overtaken + cars_overtaken_change, 0)[0]
        new_cars_to_overtake = self.config.cars_to_pass_per_level + self.config.cars_increase_per_level * (
                    state.level - 1)

        # ===== Opponent Collision =====
        collided_car = self._check_car_opponent_collision(
            new_x_abs.astype(jnp.int32),
            new_y_abs.astype(jnp.int32),
            new_visible_opponent_positions)

        # apply a cooldown if there was a collision
        new_cooldown = jnp.where(
            collided_car,
            self.config.car_crash_cooldown_frames,  # Set cooldown if collision
            new_cooldown  # Keep decremented cooldown if no collision from the beginning of the function
        )

        # Determine which direction the car drifts when there is a collision with a car
        new_cooldown_drift_direction = jnp.where(
            collided_car,
            self._determine_kickback_direction(state),
            new_cooldown_drift_direction
        )

        # ====== DISTANCE ======
        # New distance
        distance_delta = new_speed.astype(jnp.float32) * jnp.float32(self.config.km_per_speed_unit_per_frame)
        new_distance = state.distance + distance_delta

        # ====== MOUNTAINS ======
        # mountains move opposing to the curve and the move faster with higher speed
        mountain_movement = -curvature * self.config.mountain_pixel_movement_per_frame_per_speed_unit * new_speed

        # make sure the mountain x is always within the game screen
        new_mountain_left_x = self.config.window_offset_left + jnp.mod(
            state.mountain_left_x + mountain_movement - self.config.window_offset_left,
            self.config.screen_width - self.config.window_offset_left + 1,
        )
        new_mountain_right_x = self.config.window_offset_left + jnp.mod(
            state.mountain_right_x + mountain_movement - self.config.window_offset_left,
            self.config.screen_width - self.config.window_offset_left + 1,
        )

        # ====== Game Over ======
        # Compute game over condition based on elapsed frames
        # day_length_frames = self.config.day_night_cycle_seconds * self.config.frame_rate
        # game_over = state.total_frames_elapsed >= self.config.day_length_frames

        # Check whether the current level is passed.
        # Once a level is passed it does not matter whether opponents will overtake the player again.
        new_level_passed = jnp.logical_or(
            state.level_passed,
            new_cars_overtaken >=
            self.config.cars_to_pass_per_level + self.config.cars_increase_per_level * (state.level - 1)
        )

        # ===== New Day handling =====
        def reset_day():
            # do not allow level to go beyond the max level
            level = jnp.clip(state.level + 1, 1, self.config.max_increase_level)
            # cars_overtaken, level increase, level passed, game_over
            # if a new day starts and the level is not passed it is game over
            return 0, level, False, jnp.logical_not(new_level_passed)

        def do_nothing():
            # cars_overtaken, level increase, level passed, game_over
            return new_cars_overtaken, state.level, new_level_passed, state.game_over

        # Calculate current and previous day numbers
        new_day_count = jnp.floor(state.step_count / self.config.frame_rate / self.config.day_cycle_time).astype(
            jnp.int32)

        new_cars_overtaken, new_level, new_level_passed, new_game_over = lax.cond(
            new_day_count > state.day_count,  # New day started
            lambda: reset_day(),
            lambda: do_nothing(),
        )

        # Build new state with updated positions
        new_state: EnduroGameState = state._replace(
            step_count=state.step_count + 1,
            day_count=new_day_count,

            player_x_abs_position=new_x_abs,
            player_y_abs_position=new_y_abs,
            total_time_elapsed=state.step_count / self.config.frame_rate,
            total_frames_elapsed=state.total_frames_elapsed + 1,
            distance=new_distance,
            player_speed=new_speed,
            level=new_level,
            level_passed=new_level_passed,

            opponent_index=new_opponent_index,
            visible_opponent_positions=new_visible_opponent_positions,
            opponent_window=new_opponent_window,
            cars_overtaken=new_cars_overtaken,
            cars_to_overtake=new_cars_to_overtake,
            is_collision=collided_car,

            weather_index=new_weather_index,
            mountain_left_x=new_mountain_left_x,
            mountain_right_x=new_mountain_right_x,
            cooldown_drift_direction=new_cooldown_drift_direction,

            track_top_x=new_track_top_x,
            track_top_x_curve_offset=new_top_x_curve_offset,
            visible_track_left=new_left_xs.astype(jnp.int32),
            visible_track_right=new_right_xs.astype(jnp.int32),
            cooldown=new_cooldown,

            game_over=new_game_over,
        )

        # Return updated observation and state
        obs = self._get_observation(new_state)
        reward = self._get_reward(state, new_state)
        done = self._get_done(new_state)
        info = self._get_info(new_state)

        return obs, new_state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: EnduroGameState):
        track = state.whole_track
        directions = track[:, 0]
        track_starts = track[:, 1]

        segment_index = jnp.searchsorted(track_starts, state.distance, side='right') - 1
        curvature = directions[segment_index]

        # take the position of the car and the size of the config
        car = EntityPosition(
            x=state.player_x_abs_position,
            y=state.player_y_abs_position,
            width=jnp.array(self.config.car_width_0),
            height=jnp.array(self.config.car_height_0),
        )

        return EnduroObservation(
            car=car,
            visible_opponents=state.visible_opponent_positions,

            # score box
            cars_to_overtake=state.cars_to_overtake,
            distance=state.distance,
            level=state.level,

            # track
            track_top_x=state.track_top_x,
            track_left_xs=state.visible_track_left,
            track_right_xs=state.visible_track_right,
            curvature=curvature
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: EnduroGameState) -> EnduroInfo:
        return EnduroInfo(distance=state.distance, level=state.level)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: EnduroGameState, state: EnduroGameState) -> jnp.ndarray:
        return (state.total_cars_overtaken - previous_state.total_cars_overtaken) \
            + (state.distance - previous_state.distance) \
            - (state.total_time_elapsed - previous_state.total_time_elapsed)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: EnduroGameState) -> jnp.array(bool):
        return state.game_over

    @partial(jax.jit, static_argnums=(0,))
    def _generate_viewable_track(
            self,
            top_x: jnp.int32,
            top_x_curve_offset: jnp.float32  # Your signed offset, e.g., -50 to +50
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Generates the viewable track by applying a progressive horizontal shift.

        Args:
            top_x: The base position of the track at the horizon.
            top_x_curve_offset: The maximum shift at the horizon, indicating curve direction and magnitude.

        Returns:
            Two arrays that represent the x coordinates of the left and right track
        """

        # --- Step 1: Calculate the base straight track with perspective ---
        i = jnp.arange(self.config.track_height)  # Row index array: 0, 1, 2...
        perspective_offsets = jnp.where(i < 2, 0, (i - 1) // 2)
        straight_left_xs = top_x - perspective_offsets

        # --- Step 2: Calculate the progressive curve shift for EACH row ---

        # Create a normalized "depth" factor that goes from 1.0 (at the horizon) to 0.0 (at the bottom).
        # We use (track_height - i) to invert the range.
        depth_ratio = (self.config.track_height - i) / self.config.track_height

        # Apply a curve to this ratio (e.g., squaring it) to make the turn feel more natural
        # and less like a linear ramp. This makes the shift stronger at the top.
        curved_depth_ratio = jnp.power(depth_ratio, 3.0)  # You can tune the exponent (1.5, 2.0, etc.)

        # Calculate the horizontal shift for each row by scaling the max offset by the depth ratio.
        # The result is an array where the shift is `top_x_curve_offset` at the top row
        # and smoothly falls to 0 at the bottom row.
        curve_shifts = jnp.floor(top_x_curve_offset * curved_depth_ratio)

        # --- Step 3: Combine perspective and curve shifts ---
        # We simply ADD the curve shift to the base track coordinates. No more `jnp.where`!
        # Remember to SUBTRACT to match the visual direction from your last fix.
        final_left_xs = straight_left_xs + curve_shifts

        # We add one more pixel to the left track because the right track starts a pixel lower.
        # final_left_xs = jnp.concatenate([final_left_xs, final_left_xs[-1:]], axis=0)

        # --- Step 4: Generate the right track based on the final left track ---
        final_right_xs = self._generate_other_track_side_coords(final_left_xs)

        return final_left_xs.astype(jnp.int32), final_right_xs.astype(jnp.int32)

    @partial(jax.jit, static_argnums=(0,))
    def _generate_other_track_side_coords(self, left_xs: jnp.ndarray) -> jnp.ndarray:
        """
        Returns (x_coords, y_coords) for the right boundary of the track.
        Skips rows where space == -1 by collapsing to left_x.
        """

        track_spaces = self._generate_track_spaces()
        x = jnp.where(spaces == -1, left_xs, left_xs + track_spaces + 1)  # +1 to include gap
        return x  # use same Y as left

    @partial(jax.jit, static_argnums=(0,))
    def _generate_track_spaces(self) -> jnp.ndarray:
        """
        Generates a JAX array of shape (track_height,) containing the visual width of the track.
        First two rows are -1 (right boundary not drawn),
        then the width increases by 1 per row until capped at self.game_config.track_width (97).
        """
        max_width = self.config.track_width

        def body_fn(i, widths):
            """
            Computes the width for row i:
            - Rows 0 and 1 are set to -1 (skip rendering).
            - From row 2 onward, width increases by 1 each row.
            - Width is capped at max_width.
            """
            width = lax.select(i < 2, -1, jnp.minimum(i - 2, max_width))
            return widths.at[i].set(width)

        track_spaces = jnp.zeros(self.config.track_height, dtype=jnp.int32)
        track_spaces = lax.fori_loop(0, 103, body_fn, track_spaces)

        return track_spaces

    @partial(jax.jit, static_argnums=(0, 2, 3, 4))
    def _generate_opponent_spawns(
            self,
            seed: int,
            length_of_opponent_array: int,
            opponent_density: float,
            opponent_delay_slots: int
    ) -> jnp.ndarray:
        """
        Generate a precomputed spawn sequence with an *exact* occupancy equal to
        round(opponent_density * number_of_enemies) while forbidding any contiguous
        triple of non-gaps that covers all three lanes {0,1,2} in any order.

        Args:
            seed: Random seed for deterministic generation
            length_of_opponent_array: Total length of the main opponent processing array (including empty slots)
            opponent_density: Fraction (0.0-1.0) of slots that will contain actual opponents (lane 0,1,2 vs -1)
            opponent_delay_slots: Number of guaranteed empty slots (-1) added at the beginning of the final array

        Returns:
            2D array of shape (2, total_length) where:
            - Row 0: lane positions (-1 = empty, 0 = left, 1 = middle, 2 = right)
            - Row 1: colors (0 = no opponent, RGB packed as single int for opponents)

        Encoding:
            -1 = empty slot (gap), 0 = left, 1 = middle, 2 = right
        """
        key = jax.random.PRNGKey(seed)
        key, key_colors = jax.random.split(key)  # Split key for color generation

        # Calculate exact number of occupied slots
        num_occupied = int(round(opponent_density * length_of_opponent_array))

        # Generate random positions for occupied slots
        key, key_positions = jax.random.split(key)
        all_indices = jnp.arange(length_of_opponent_array)
        shuffled_indices = jax.random.permutation(key_positions, all_indices)
        occupied_positions = shuffled_indices[:num_occupied]

        # Create occupancy mask
        occupancy_mask = jnp.zeros(length_of_opponent_array, dtype=jnp.bool_)
        occupancy_mask = occupancy_mask.at[occupied_positions].set(True)

        # Generate lane assignments for occupied slots
        key, key_lanes = jax.random.split(key)
        lane_choices = jax.random.randint(key_lanes, (length_of_opponent_array,), 0, 3, dtype=jnp.int8)

        # Generate colors for all positions (we'll mask out non-opponents later)
        def generate_non_red_color(color_key):
            """Generate vibrant colors avoiding pure red"""
            keys = jax.random.split(color_key, 3)

            r = jax.random.randint(keys[0], (), 50, 256)
            g = jax.random.randint(keys[1], (), 0, 256)
            b = jax.random.randint(keys[2], (), 0, 256)

            # Ensure at least G or B is >128 to avoid pure red
            max_gb = jnp.maximum(g, b)
            adjustment_needed = jnp.where(max_gb < 128, 128 - max_gb, 0)

            g = jnp.where((g >= b) & (max_gb < 128), g + adjustment_needed, g)
            b = jnp.where((b > g) & (max_gb < 128), b + adjustment_needed, b)

            # Pack RGB into single integer: (R << 16) | (G << 8) | B
            color = jnp.clip(r, 0, 255) * 65536 + jnp.clip(g, 0, 255) * 256 + jnp.clip(b, 0, 255)
            return color.astype(jnp.int32)

        # Generate colors for each position
        color_keys = jax.random.split(key_colors, length_of_opponent_array)
        colors = jax.vmap(generate_non_red_color)(color_keys)

        def process_slot(carry, inputs):
            """Process each slot, enforcing the no-triple-lane constraint"""
            key_step, last_two_lanes, non_gap_count = carry
            is_occupied, candidate_lane, color = inputs

            key_step, key_fix = jax.random.split(key_step)

            # Check if placing candidate would create a {0,1,2} triple
            has_valid_triple = (
                    (non_gap_count >= 2) &
                    (last_two_lanes[0] != last_two_lanes[1]) &
                    (candidate_lane != last_two_lanes[0]) &
                    (candidate_lane != last_two_lanes[1])
            )

            # If violation, randomly pick one of the last two lanes
            fix_choice = jax.random.randint(key_fix, (), 0, 2)
            fixed_lane = jnp.where(fix_choice == 0, last_two_lanes[0], last_two_lanes[1])
            final_lane = jnp.where(has_valid_triple, fixed_lane, candidate_lane)

            # Output: -1 for gap, final_lane for occupied
            output_lane = jnp.where(is_occupied, final_lane, jnp.int8(-1))
            output_color = jnp.where(is_occupied, color, jnp.int32(0))

            # Update carry for next iteration
            new_non_gap_count = jnp.where(is_occupied, non_gap_count + 1, jnp.int32(0))
            new_last_two = jnp.where(
                is_occupied,
                jnp.array([last_two_lanes[1], final_lane], dtype=jnp.int8),
                jnp.array([-1, -1], dtype=jnp.int8)
            )

            new_carry = (key_step, new_last_two, new_non_gap_count)
            return new_carry, (output_lane, output_color)

        # Initial state: no lane history, zero non-gap count
        initial_carry = (
            key,
            jnp.array([-1, -1], dtype=jnp.int8),
            jnp.int32(0)
        )

        # Process all slots
        inputs = (occupancy_mask, lane_choices, colors)
        _, (lane_sequence, color_sequence) = jax.lax.scan(process_slot, initial_carry, inputs)

        # Add delay slots at the beginning
        delay_lanes = jnp.full((opponent_delay_slots,), -1, dtype=jnp.int8)
        delay_colors = jnp.full((opponent_delay_slots,), 0, dtype=jnp.int32)

        final_lanes = jnp.concatenate([delay_lanes, lane_sequence])
        final_colors = jnp.concatenate([delay_colors, color_sequence])

        # Stack into 2D array: [lanes, colors]
        result = jnp.stack([final_lanes.astype(jnp.int32), final_colors], axis=0)

        return result

    @partial(jax.jit, static_argnums=(0,))
    def _get_visible_opponent_positions(self, opponent_index: jnp.ndarray,
                                        opponent_pos_and_color: jnp.ndarray,
                                        visible_track_left: jnp.ndarray,
                                        visible_track_right: jnp.ndarray) -> jnp.ndarray:
        """
        Calculate the x,y positions and colors of all 7 opponent slots based on the current opponent_index.

        The opponent_index acts as a sliding window into the opponent spawn array.
        The integer part determines which 7 consecutive slots to use.
        The decimal part determines vertical positioning within each slot's pixel range.

        Args:
            opponent_index: Current position in opponent array (float with decimal movement)
            opponent_pos_and_color: Array of opponent spawn data with position [0] and color [1] of the car.
            visible_track_left: Left track boundary x-coordinates for each track row
            visible_track_right: Right track boundary x-coordinates for each track row

        Returns:
            jnp.ndarray of shape (7, 3) where each row is [x_position, y_position, color]
            For empty slots: x_position = -1, y_position = slot_y, color = 0
        """
        # Base y positions for each slot (top of each slot)
        base_y_positions = self.config.opponent_slot_ys

        # Calculate slot heights from consecutive y positions
        slot_heights = jnp.diff(base_y_positions, prepend=self.config.game_window_height)
        slot_heights = jnp.abs(slot_heights)  # Ensure positive values

        # Extract integer and decimal parts of opponent_index
        index_integer = jnp.floor(opponent_index).astype(jnp.int32)
        index_decimal = opponent_index - index_integer

        # Get the 7 consecutive opponent slots starting from index_integer
        # Use modulo to handle array bounds safely
        opponent_array_size = opponent_pos_and_color[0].shape[0]
        slot_indices = (index_integer + jnp.arange(7)) % opponent_array_size
        current_slots = opponent_pos_and_color[0][slot_indices]  # Shape: (7,) with lane values
        current_colors = opponent_pos_and_color[1][slot_indices]  # Shape: (7,) with color values

        # Calculate y-positions based on decimal part of opponent_index
        # The decimal part determines how far through each slot the opponents have moved
        y_offsets = jnp.floor(index_decimal * slot_heights).astype(jnp.int32)
        y_positions = base_y_positions + y_offsets

        # Convert y positions to track row indices (relative to sky height)
        track_row_indices = y_positions - self.config.sky_height
        track_row_indices = jnp.clip(track_row_indices, 0, self.config.track_height - 1)

        # Get track boundaries for each opponent slot's y position
        left_boundaries = visible_track_left[track_row_indices]
        right_boundaries = visible_track_right[track_row_indices]
        track_widths = right_boundaries - left_boundaries

        # Calculate x-positions based on lane assignments using track-relative positions
        # Get car widths for each opponent slot (0=closest/largest, 6=farthest/smallest)
        car_widths = self.config.car_widths  # Shape: (7,)

        def calculate_x_for_lane(slot_idx, lane_code, left_bound, track_width):
            # For empty slots (-1), return -1 as "not visible" marker
            valid_lane = jnp.clip(lane_code, 0, 2)  # Clamp to valid range
            ratio = self.config.lane_ratios[valid_lane]

            # Calculate center position of the car
            center_x = left_bound + (track_width * ratio)

            # Adjust to leftmost pixel by subtracting half the car width
            car_width = car_widths[slot_idx]
            leftmost_x = center_x - (car_width // 2)

            return jnp.where(lane_code == -1, -1.0, leftmost_x)

        # Vectorized calculation for all opponent slots
        slot_indices = jnp.arange(7)
        x_positions = jax.vmap(calculate_x_for_lane)(
            slot_indices, current_slots, left_boundaries, track_widths
        ).astype(jnp.int32)

        # Combine into final result: [x, y, color] for each slot
        result = jnp.stack([x_positions, y_positions, current_colors], axis=1)

        return result

    @partial(jax.jit, static_argnums=(0,))
    def _adjust_opponent_positions_when_overtaking(self, state, new_opponent_index: jnp.ndarray):
        """
        Prevents unavoidable collisions when opponents overtake the player by automatically moving
        opponents to safe lanes. When opponents are faster than the player, new cars spawn in slot 0
        (closest position) as the player moves backwards through the opponent array. Without this
        adjustment, opponents could spawn directly on the player's position.

        Args:
            state: Current game state containing player position, opponent data, and track boundaries.
            new_opponent_index: The opponent array index after this step, used to detect if new
                opponents are becoming visible.

        Returns:
            Modified opponent_pos_and_color array with collision-avoiding lane changes, or the
            original array if no adjustments were needed.
        """
        # Check if opponents are overtaking (moving backwards through array)
        new_integer_index = jnp.floor(new_opponent_index).astype(jnp.int32)
        previous_integer_index = jnp.floor(state.opponent_index).astype(jnp.int32)
        slots_retreated = jnp.maximum(0, previous_integer_index - new_integer_index)

        def adjust_opponents():
            # Calculate which lane(s) the player occupies
            # Get track boundaries at player's y position
            # track_row_index = jnp.clip(state.player_y_abs_position - self.config.sky_height,
            #                            0, self.config.track_height - 1).astype(jnp.int32)  # Convert to int!
            left_boundary = state.visible_track_left[self.config.player_y_start]
            right_boundary = state.visible_track_right[self.config.player_y_start]
            track_width = right_boundary - left_boundary

            # Calculate player car's position relative to track (left and right edges)
            player_left_ratio = (state.player_x_abs_position - left_boundary) / track_width
            player_right_ratio = (state.player_x_abs_position + self.config.car_width_0 - left_boundary) / track_width

            # Check which lanes the player car overlaps (lanes are roughly at 0-1/3, 1/3-2/3, 2/3-1)
            player_in_lane_0 = (player_left_ratio < 1 / 3) & (player_right_ratio > 0)
            player_in_lane_1 = (player_left_ratio < 2 / 3) & (player_right_ratio > 1 / 3)
            player_in_lane_2 = (player_left_ratio < 1.0) & (player_right_ratio > 2 / 3)

            # Adjust opponent in slot 0 if it would collide
            opponent_array_size = state.opponent_pos_and_color.shape[1]
            slot_0_index = new_integer_index % opponent_array_size
            slot_0_lane = state.opponent_pos_and_color[0, slot_0_index]

            # Check for collision
            would_collide = ((slot_0_lane == 0) & player_in_lane_0) | \
                            ((slot_0_lane == 1) & player_in_lane_1) | \
                            ((slot_0_lane == 2) & player_in_lane_2)

            # Find a safe lane (prefer one the player doesn't occupy)
            safe_lane = jnp.where(~player_in_lane_0, 0,
                                  jnp.where(~player_in_lane_2, 2, 1))

            # Update lane if collision and opponent exists
            new_lane = jnp.where(would_collide & (slot_0_lane != -1), safe_lane, slot_0_lane)

            # Update the opponent array
            modified_positions = state.opponent_pos_and_color[0].at[slot_0_index].set(new_lane)
            return jnp.array([modified_positions, state.opponent_pos_and_color[1]])

        # Only adjust when opponents are actually overtaking
        return jnp.where(slots_retreated > 0,
                         adjust_opponents(),
                         state.opponent_pos_and_color)

    @partial(jax.jit, static_argnums=(0,))
    def _build_whole_track(self, seed: int) -> jnp.ndarray:
        """
        Generate a precomputed Enduro track up to (and beyond) `self.config.max_track_length`.

        The track begins with a fixed 100 meters (0.1 km) of straight driving, followed by
        randomly generated segments.

        Each track segment is defined by:
            - direction: -1 (left), 0 (straight), or 1 (right)
            - start_distance: cumulative distance at which the segment begins [in km]

        To avoid JAX tracing issues (e.g., with boolean indexing), we generate slightly more
        segments than strictly necessary and do not mask or slice the output dynamically.

        Returns:
            track: jnp.ndarray of shape (N, 2), where each row is [direction, start_distance]
        """
        key = jax.random.PRNGKey(seed)

        # Add buffer so we never run short
        max_segments = int(self.config.max_track_length) + 100

        key, subkey = jax.random.split(key)
        directions = jax.random.choice(subkey, jnp.array([-1, 0, 1]), shape=(max_segments,), replace=True)

        key, subkey = jax.random.split(key)
        segment_lengths = jax.random.uniform(subkey,
                                             shape=(max_segments,),
                                             minval=self.config.min_track_section_length,
                                             maxval=self.config.max_track_section_length)

        track_starts = jnp.cumsum(jnp.concatenate([jnp.array([0.1]), segment_lengths[:-1]]))

        # Combine fixed start + rest (no masking)
        first_segment = jnp.array([[0.0, 0.0]])  # straight start
        rest_segments = jnp.stack([directions, track_starts], axis=1)

        track = jnp.concatenate([first_segment, rest_segments], axis=0)

        return track

    @partial(jax.jit, static_argnums=(0,))
    def _check_car_track_collision(
            self,
            car_x_abs: jnp.int32,
            car_y_abs: jnp.int32,
            left_track_xs: jnp.ndarray,
            right_track_xs: jnp.ndarray
    ) -> jnp.bool_:
        """
        Checks for pixel-perfect collision between the player car and the track boundaries.

        Args:
            car_x_abs: The absolute x-coordinate of the car sprite's top-left corner.
            car_y_abs: The absolute y-coordinate of the car sprite's top-left corner.
            left_track_xs: Array of x-coordinates for the left track boundary.
            right_track_xs: Array of x-coordinates for the right track boundary.

        Returns:
            An integer: 0 for no collision, -1 for a left-side collision,
            1 for a right-side collision.
        """
        spec = self.car_0_spec  # Get the pre-calculated car spec

        # --- Calculate the absolute screen coordinates of all solid car pixels ---
        absolute_car_xs = car_x_abs + spec.collision_mask_relative_xs
        absolute_car_ys = car_y_abs + spec.collision_mask_relative_ys

        def check_one_pixel(collision_side_so_far, i):
            """
            This is the core logic that lax.scan will run for each pixel.
            It checks the i-th solid pixel of the car for a collision.
            Args:
                collision_side_so_far: is an integer (0, 1, or -1) which indicates the side of a collision
                i: The index of the pixel
            """
            # A check to ensure we only process valid pixels, not the padded values.
            is_valid_pixel = i < spec.num_solid_pixels

            # Get the absolute coordinates of this specific car pixel
            pixel_x = absolute_car_xs[i]
            pixel_y = absolute_car_ys[i]

            # Convert the pixel's absolute Y position to a track row index because the track is saved as an x value.
            # This is how we look up the track boundaries for that specific row.
            track_row_index = pixel_y - self.config.sky_height

            # --- Perform collision check for this single pixel ---

            # 1. Is the pixel vertically within the drawable track area?
            is_y_on_track = (track_row_index >= 0) & (track_row_index < self.config.track_height)

            # 2. To prevent out-of-bounds indexing, we clamp the index.
            #    The result is only used if is_y_on_track is True anyway.
            clamped_row = jnp.clip(track_row_index, 0, self.config.track_height - 1)
            left_boundary_at_row = left_track_xs[clamped_row]
            right_boundary_at_row = right_track_xs[clamped_row]

            # 3. Is the pixel's x-coordinate outside the track boundaries for its row?
            x_collides_left = (pixel_x <= left_boundary_at_row)
            x_collides_right = (pixel_x >= right_boundary_at_row)

            # 4. Determine the collision side for *this pixel*
            # If it collides left, side is -1. If right, side is 1, otherwise it is 0.
            this_pixel_side = jnp.where(x_collides_left, -1, 0) + jnp.where(x_collides_right, 1, 0)
            # A valid collision only happens if it's on the track vertically.
            this_pixel_side = jnp.where(is_valid_pixel & is_y_on_track, this_pixel_side, 0)

            # 5. Update the overall collision side.
            # We only update if we haven't found a collision yet. This gives priority
            # to the first pixel that collides.
            return jnp.where(collision_side_so_far != 0, collision_side_so_far, this_pixel_side), None

        # Use lax.scan to iterate over every potential pixel in our padded array.
        # The initial value for `collision_so_far` is False.
        final_collision_side, _ = jax.lax.scan(
            check_one_pixel,
            0,  # Initial carry (0 equals no collision, else it shows the kickback direction)
            jnp.arange(spec.collision_mask.size)  # Iterate from 0 to max possible pixels
        )

        return final_collision_side

    @partial(jax.jit, static_argnums=(0,))
    def _check_car_opponent_collision(
            self,
            player_car_x: jnp.int32,
            player_car_y: jnp.int32,
            visible_opponents: jnp.ndarray
    ) -> jnp.bool_:
        """
        Checks for pixel-perfect collision between the player car and the opponent cars.

        Args:
            player_car_x: The absolute x-coordinate of the car sprite's top-left corner.
            player_car_y: The absolute y-coordinate of the car sprite's top-left corner.
            visible_opponents:
                jnp.ndarray of shape (7, 3) where each row is [x_position, y_position, color] of an opponent.
                For empty slots: x_position = -1, y_position = slot_y, color = 0

        Returns:
            An integer: 0 for no collision, 1 for collision
        """
        player_spec = self.car_0_spec
        spec_0 = self.car_0_spec
        spec_1 = self.car_1_spec

        # --- Calculate the absolute screen coordinates of all solid player car pixels ---
        absolute_player_xs = player_car_x + player_spec.collision_mask_relative_xs
        absolute_player_ys = player_car_y + player_spec.collision_mask_relative_ys

        def _check_one_opponent(opponent_x, opponent_y, opponent_spec):
            """
            Helper function to check for collision between the player and a single opponent.
            """
            # --- Calculate the absolute screen coordinates of all solid opponent pixels ---
            absolute_opponent_xs = opponent_x + opponent_spec.collision_mask_relative_xs
            absolute_opponent_ys = opponent_y + opponent_spec.collision_mask_relative_ys

            # --- Create masks to only consider valid, non-padded pixels ---
            player_valid_mask = jnp.arange(player_spec.collision_mask.size) < player_spec.num_solid_pixels
            opponent_valid_mask = jnp.arange(opponent_spec.collision_mask.size) < opponent_spec.num_solid_pixels

            # --- Efficiently check for any overlapping pixels using broadcasting ---
            # Reshape arrays to (num_pixels, 1) and (1, num_pixels) to compare all pairs.
            x_matches = absolute_player_xs[:, None] == absolute_opponent_xs[None, :]
            y_matches = absolute_player_ys[:, None] == absolute_opponent_ys[None, :]

            # A collision occurs if both X and Y coordinates match for any pair of pixels.
            pixel_collisions = x_matches & y_matches

            # Create a combined validity mask for the comparison matrix.
            valid_comparison_mask = player_valid_mask[:, None] & opponent_valid_mask[None, :]

            # A true collision only happens if the colliding pixels are not padding.
            any_collision = jnp.any(pixel_collisions & valid_comparison_mask)

            return any_collision

        # --- Check against Opponent 0 (car_0 type) ---
        car_0_x, car_0_y = visible_opponents[0, 0], visible_opponents[0, 1]
        car_0_valid = car_0_x != -1
        # Note: The collision check will run regardless, but the result is only used if the car is valid.
        collision_with_car_0 = _check_one_opponent(car_0_x, car_0_y, spec_0)
        # The final collision result for this car is True only if it's valid AND a collision occurred.
        final_collision_0 = car_0_valid & collision_with_car_0

        # --- Check against Opponent 1 (car_1 type) ---
        car_1_x, car_1_y = visible_opponents[1, 0], visible_opponents[1, 1]
        car_1_valid = car_1_x != -1
        collision_with_car_1 = _check_one_opponent(car_1_x, car_1_y, spec_1)
        final_collision_1 = car_1_valid & collision_with_car_1

        # --- Final result is true if there is a collision with either car ---
        return jnp.where(final_collision_0 | final_collision_1, 1, 0)

    def _determine_kickback_direction(self, state: EnduroGameState):
        """
        Determine the initial kickback direction of a car collision.
        If the car is more left of the track it is to the right and vice versa.
        Args:
            state: the enduro GameState

        Returns:
            a direction either left (-1) or right (1)
        """
        # calculate based on the track position of the player
        left_boundary = state.visible_track_left[self.config.player_y_start]
        right_boundary = state.visible_track_right[self.config.player_y_start]
        track_width = right_boundary - left_boundary

        # Calculate player car's position relative to track (left and right edges)
        player_left_ratio = (state.player_x_abs_position - left_boundary) / track_width
        player_right_ratio = (state.player_x_abs_position + self.config.car_width_0 - left_boundary) / track_width

        # Calculate center position of player car
        player_center_ratio = (player_left_ratio + player_right_ratio) / 2.0

        return lax.cond(
            player_center_ratio < 0.5,  # Car is on left half of track
            lambda: 1,  # Kick to the right
            lambda: -1  # Kick to the left
        )


# --------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------


class EnduroRenderer(JAXGameRenderer):
    """
    Renders the jax_enduro game
    """

    def __init__(self):
        super().__init__()
        self.config = EnduroConstants()

        # sprite sizes to easily and dynamically adjust renderings
        self.background_sizes: dict[str, tuple[int, int]] = {}
        self.sprites = self._load_sprites()

        self.track_height = self.background_sizes['background_gras.npy'][0]

        # render all background that do not change only once
        self.static_background = self._render_static_background()

    def _load_sprites(self):
        module_dir = os.path.dirname(os.path.abspath(__file__))

        sprites: dict[str, Any] = {}

        # required sprite folders
        folders = ['backgrounds', 'cars', 'digits', 'misc']

        # using a folder structure is much easier than keeping track of all required files one by one.
        # the game will only load sprites that are in the folders above
        for folder in folders:
            folder_path = os.path.join(module_dir, f"sprites/enduro/{folder}/")
            # --- load sprites ---
            for filename in os.listdir(folder_path):
                # load npy files
                if filename.endswith(".npy"):
                    full_path = os.path.join(folder_path, filename)
                    frame = aj.load_frame_with_animation(full_path, transpose=False)
                    # save with the full extension, so remember to also load them with .npy
                    sprites[filename] = frame.astype(jnp.uint8)

                    # Store size info for backgrounds
                    if folder == 'backgrounds':
                        height = frame.shape[1]  # (N, H, W, C)
                        width = frame.shape[2]
                        self.background_sizes[filename] = (height, width)

        return sprites

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: EnduroGameState) -> jnp.ndarray:
        """Render the game state to a raster image."""
        raster = self.static_background.copy()

        raster = self._render_weather(raster, state)

        # render the player car
        raster = self._render_player_car(raster, state)

        # render the opponents
        raster = self._render_opponent_cars(raster, state)

        # render the lower background again to make opponents below the screen disappear
        raster = self._render_lower_background(raster)

        # render the track
        raster = self._render_track_from_state(raster, state)

        # render the distance odometer, level score and cars to overtake
        raster = self._render_distance_odometer(raster, state)
        raster = self._render_cars_to_overtake_score(raster, state)  # must be rendered before level, due to background
        raster = self._render_level_score(raster, state)

        # render the mountains
        raster = self._render_mountains(raster, state)

        # render the fog as the last thing!
        raster = self._render_fog(raster, state)

        return raster

    @partial(jax.jit, static_argnums=(0,))
    def _render_player_car(self, raster: jnp.ndarray, state: EnduroGameState):
        """
        Renders the player car. The animation speed depends on the player speed
        Args:
            raster: the raster to draw in
            state: the enduro Game State

        Returns: the final raster with the rendered track
        """
        # speed 24 → period 2.0 (like opponents), speed 120 → period 1.0
        animation_period = 2.0 - (state.player_speed - self.config.opponent_speed) / (
                self.config.max_speed - self.config.opponent_speed)

        # Calculate animation step (slower at low speeds, faster at high speeds)
        animation_step = jnp.floor(state.step_count / animation_period)

        # Alternate between frame 0 and 1 based on animation step
        frame_index = (animation_step % 2).astype(jnp.int32)

        # render player car position. The player's car is always in size 0 (largest)
        player_car = aj.get_sprite_frame(self.sprites['car_0.npy'], frame_index)

        # get the car position in an absolute coordinate
        raster = aj.render_at(raster, state.player_x_abs_position, state.player_y_abs_position, player_car)

        return raster

    @partial(jax.jit, static_argnums=(0,))
    def _render_track_from_state(self, raster: jnp.ndarray, state: EnduroGameState):
        """
        Renders the track pixels from the Enduro Game State with animated colors.
        """
        # get a less steep slope for the acceleration of the animation
        effective_speed = (self.config.min_speed +
                           (state.player_speed - self.config.min_speed) * self.config.track_speed_animation_factor)
        # Calculate animation step
        animation_step = jnp.floor(
            effective_speed * state.step_count * self.config.track_color_move_speed_per_speed
        ) % self.config.track_move_range
        animation_step = animation_step.astype(jnp.int32)

        # build the y array
        sky_height = self.background_sizes['background_sky.npy'][0]
        y = jnp.add(jnp.arange(self.track_height), sky_height)

        # Concatenate both sides and create a grid of x & y coordinates
        x_coords = jnp.concatenate([state.visible_track_left, state.visible_track_right])
        y_coords = jnp.concatenate([y, y])

        # Create track sprite with animated colors
        track_sprite = self._draw_animated_track_sprite(x_coords, y_coords, animation_step)
        # Render to raster
        raster = aj.render_at(raster, 0, 0, track_sprite)

        return raster

    @partial(jax.jit, static_argnums=(0,))
    def _draw_animated_track_sprite(self, x_coords: jnp.ndarray, y_coords: jnp.ndarray,
                                    animation_step: jnp.int32) -> jnp.ndarray:
        """
        Creates a sprite for the track with animated colors based on animation step.
        """
        # Create full-screen-sized sprite
        sprite = jnp.zeros((self.config.screen_height, self.config.screen_width, 4), dtype=jnp.uint8)

        # Calculate color regions based on animation step
        sky_height = self.background_sizes['background_sky.npy'][0]

        def get_track_color(track_row_index: jnp.int32) -> jnp.ndarray:
            """Determine color for a given track row based on animation step."""

            # Calculate boundaries (these shift with animation_step)
            top_region_end = self.config.track_top_min_length + animation_step
            moving_top_end = top_region_end + self.config.track_moving_top_length

            # Check if we should spawn moving bottom (step >= 6)
            spawn_moving_bottom = animation_step >= self.config.track_moving_bottom_spawn_step
            moving_bottom_end = jnp.where(
                spawn_moving_bottom,
                moving_top_end + self.config.track_moving_bottom_length,
                moving_top_end
            )

            # Determine color based on position
            color = jnp.where(
                track_row_index < top_region_end,
                self.config.track_colors[0],  # top color
                jnp.where(
                    track_row_index < moving_top_end,
                    self.config.track_colors[1],  # moving top color
                    jnp.where(
                        (track_row_index < moving_bottom_end) & spawn_moving_bottom,
                        self.config.track_colors[2],  # moving bottom color
                        self.config.track_colors[3]  # bottom/rest color
                    )
                )
            )

            return color

        def draw_pixel(i, s):
            x = x_coords[i]
            y = y_coords[i]
            track_row_index = y - sky_height
            color = get_track_color(track_row_index)
            # Add alpha channel (assuming track should be opaque)
            color_with_alpha = jnp.append(color, 255)
            return s.at[y, x].set(color_with_alpha)

        sprite = jax.lax.fori_loop(0, x_coords.shape[0], draw_pixel, sprite)
        return sprite

    @partial(jax.jit, static_argnums=(0,))
    def _render_track_from_state2(self, raster: jnp.ndarray, state: EnduroGameState):
        """
        Renders the track pixels from the Enduro Game State.
        Args:
            raster: the raster to draw in
            state: the enduro Game State

        Returns: the final raster with the rendered track

        """
        track_pixel = aj.get_sprite_frame(self.sprites['track_boundary.npy'], 0)

        track_color = track_pixel[0, 0]  # (4,)

        # build the y array
        sky_height = self.background_sizes['background_sky.npy'][0]
        y = jnp.add(
            jnp.arange(self.track_height),
            sky_height
        )

        # Concatenate both sides and create a grid of x & y coordinates
        x_coords = jnp.concatenate([state.visible_track_left, state.visible_track_right])
        y_coords = jnp.concatenate([y, y])

        # Create track sprite
        track_sprite = self._draw_track_sprite(x_coords, y_coords, track_color)
        # Render to raster
        raster = aj.render_at(raster, 0, 0, track_sprite)

        return raster

    @partial(jax.jit, static_argnums=(0,))
    def _draw_track_sprite2(self, x_coords: jnp.ndarray, y_coords: jnp.ndarray, color: jnp.ndarray) -> jnp.ndarray:
        """
        Creates a sprite for the track that covers the whole screen, which makes it a little easier to draw,
        because you can use absolute x,y positions for drawing the pixels.
        """
        # Create full-screen-sized sprite
        sprite = jnp.zeros((self.config.screen_width, self.config.screen_height, 4), dtype=jnp.uint8)

        def draw_pixel(i, s):
            x = x_coords[i]
            y = y_coords[i]
            return s.at[y, x].set(color)

        sprite = jax.lax.fori_loop(0, x_coords.shape[0], draw_pixel, sprite)
        return sprite

    @partial(jax.jit, static_argnums=(0,))
    def _render_opponent_cars(self, raster: jnp.ndarray, state: EnduroGameState) -> jnp.ndarray:
        """
        Renders all opponent cars into the game with custom colors
        Args:
            raster: the raster for rendering
            state: the game state

        Returns: the new raster with colored opponents
        """
        # adjust the animation speed for opponents
        animation_step = jnp.floor(state.step_count / self.config.opponent_animation_steps)

        # Alternate between frame 0 and 1 based on step count
        frame_index = (animation_step % 2).astype(jnp.int32)

        is_day = ~jnp.isin(state.weather_index, self.config.weather_with_night_car_sprite)

        # Load all car sprites as separate variables (they have different shapes) depending on the current weather
        car_0 = lax.cond(
            is_day,
            lambda: aj.get_sprite_frame(self.sprites['car_0.npy'], frame_index),
            lambda: aj.get_sprite_frame(self.sprites['car_0_night.npy'], 0)
        )
        car_1 = lax.cond(
            is_day,
            lambda: aj.get_sprite_frame(self.sprites['car_1.npy'], frame_index),
            lambda: aj.get_sprite_frame(self.sprites['car_1_night.npy'], 0)
        )
        car_2 = lax.cond(
            is_day,
            lambda: aj.get_sprite_frame(self.sprites['car_2.npy'], frame_index),
            lambda: aj.get_sprite_frame(self.sprites['car_2_night.npy'], 0)
        )
        car_3 = lax.cond(
            is_day,
            lambda: aj.get_sprite_frame(self.sprites['car_3.npy'], frame_index),
            lambda: aj.get_sprite_frame(self.sprites['car_3_night.npy'], 0)
        )
        car_4 = lax.cond(
            is_day,
            lambda: aj.get_sprite_frame(self.sprites['car_4.npy'], frame_index),
            lambda: aj.get_sprite_frame(self.sprites['car_4_night.npy'], 0)
        )
        car_5 = lax.cond(
            is_day,
            lambda: aj.get_sprite_frame(self.sprites['car_5.npy'], frame_index),
            lambda: aj.get_sprite_frame(self.sprites['car_5_night.npy'], 0)
        )
        car_6 = lax.cond(
            is_day,
            lambda: aj.get_sprite_frame(self.sprites['car_6.npy'], frame_index),
            lambda: aj.get_sprite_frame(self.sprites['car_6_night.npy'], 0)
        )

        # Get opponent positions from precomputed state
        opponent_positions = state.visible_opponent_positions

        # Extract positions and colors
        xs = opponent_positions[:, 0]  # x positions
        ys = opponent_positions[:, 1]  # y positions
        colors = opponent_positions[:, 2]  # packed RGB colors

        def apply_color_to_sprite(sprite, packed_color):
            """Apply custom color to sprite, replacing non-black pixels"""
            # Unpack RGB from integer
            r = (packed_color >> 16) & 0xFF
            g = (packed_color >> 8) & 0xFF
            b = packed_color & 0xFF
            return aj.change_sprite_color(sprite, jnp.stack([r, g, b], dtype=jnp.int32))

        # Apply colors to each sprite when it is day. At night just keep the sprite (rear lights)
        colored_car_0 = lax.cond(is_day, lambda: apply_color_to_sprite(car_0, colors[0]), lambda: car_0)
        colored_car_1 = lax.cond(is_day, lambda: apply_color_to_sprite(car_1, colors[1]), lambda: car_1)
        colored_car_2 = lax.cond(is_day, lambda: apply_color_to_sprite(car_2, colors[2]), lambda: car_2)
        colored_car_3 = lax.cond(is_day, lambda: apply_color_to_sprite(car_3, colors[3]), lambda: car_3)
        colored_car_4 = lax.cond(is_day, lambda: apply_color_to_sprite(car_4, colors[4]), lambda: car_4)
        colored_car_5 = lax.cond(is_day, lambda: apply_color_to_sprite(car_5, colors[5]), lambda: car_5)
        colored_car_6 = lax.cond(is_day, lambda: apply_color_to_sprite(car_6, colors[6]), lambda: car_6)

        # Render each car individually with explicit conditionals
        # Car 0 (closest)
        raster = jnp.where((xs[0] != -1), aj.render_at(raster, xs[0], ys[0], colored_car_0), raster)
        # Car 1
        raster = jnp.where((xs[1] != -1), aj.render_at(raster, xs[1], ys[1], colored_car_1), raster)
        # Car 2
        raster = jnp.where((xs[2] != -1), aj.render_at(raster, xs[2], ys[2], colored_car_2), raster)
        # Car 3
        raster = jnp.where((xs[3] != -1), aj.render_at(raster, xs[3], ys[3], colored_car_3), raster)
        # Car 4
        raster = jnp.where((xs[4] != -1), aj.render_at(raster, xs[4], ys[4], colored_car_4), raster)
        # Car 5
        raster = jnp.where((xs[5] != -1), aj.render_at(raster, xs[5], ys[5], colored_car_5), raster)
        # Car 6 (farthest)
        raster = jnp.where((xs[6] != -1), aj.render_at(raster, xs[6], ys[6], colored_car_6), raster)

        return raster

    @partial(jax.jit, static_argnums=(0,))
    def _render_distance_odometer(self, raster: jnp.ndarray, state: EnduroGameState) -> jnp.ndarray:
        """
        Renders the odometer that shows the distance that we've covered.
        first 4 digits are in black font and the decimal digit is in brown font.
        The digits roll down like in a car odometer.
        Args:
            raster: the raster for rendering
            state: the game state

        Returns: the new raster with odometer
        """
        # Get digit dimensions from a sample sprite
        digit_0_black = aj.get_sprite_frame(self.sprites['0_black.npy'], 0)
        window_height = digit_0_black.shape[0] + 2
        digit_width = digit_0_black.shape[1]

        # get the black and brown digit sprites that have all digits
        digit_sprite_black = aj.get_sprite_frame(self.sprites['black_digit_array.npy'], 0)
        digit_sprite_brown = aj.get_sprite_frame(self.sprites['brown_digit_array.npy'], 0)

        # determine the base position in the sprite that represents the lowest y for the window
        base_y = digit_sprite_brown.shape[0] - window_height + 1

        # Calculate how many 0.0125 increments have passed for decimal animation
        increments_passed = jnp.floor(state.distance / 0.0125)
        increments_passed = increments_passed.astype(jnp.int32)

        # Adjust y by 1 for each increment, using modulo to reset when decimal digit rolls over
        # Each full decimal digit (0-9) represents 80 increments (1.0 / 0.0125 = 80)
        y_offset = increments_passed % 80

        # Extract actual digit values from distance
        distance_int = jnp.floor(state.distance).astype(jnp.int32)
        decimal_digit = (state.distance * 10) % 1
        thousands_digit = (distance_int // 1000) % 10
        hundreds_digit = (distance_int // 100) % 10
        tens_digit = (distance_int // 10) % 10
        ones_digit = distance_int % 10

        # synchronize the movement of the digits when they spin from 9 to 0
        decimal_y = base_y - y_offset
        ones_y = base_y - ones_digit * (window_height - 1) - jnp.clip(window_height - decimal_y - 2, 0)
        tens_y = base_y - tens_digit * (window_height - 1) - jnp.clip(window_height - ones_y - 2, 0)
        hundreds_y = base_y - hundreds_digit * (window_height - 1) - jnp.clip(window_height - tens_y - 2, 0)
        thousands_y = base_y - thousands_digit * (window_height - 1) - jnp.clip(window_height - hundreds_y - 2, 0)

        # Reset to base_y when we complete a full cycle (decimal part hits 0 again)
        decimal_y = jnp.where(decimal_digit < 0.001, base_y, decimal_y)
        ones_y = jnp.where(ones_y < 0.001, base_y, ones_y)
        tens_y = jnp.where(tens_y < 0.001, base_y, tens_y)
        hundreds_y = jnp.where(hundreds_y < 0.001, base_y, hundreds_y)
        thousands_y = jnp.where(thousands_y < 0.001, base_y, thousands_y)

        # Extract decimal digit window
        digit_window = jax.lax.dynamic_slice(
            digit_sprite_brown,
            (decimal_y, 0, 0),  # start indices (y position in sprite, x=0, channel=0)
            (window_height, digit_width, digit_sprite_brown.shape[2])
        )

        # Add rolling animation offsets when appropriate
        # Ones digit - moves when decimal digit is rolling
        ones_window = jax.lax.dynamic_slice(
            digit_sprite_black,
            (ones_y, 0, 0),
            (window_height, digit_width, digit_sprite_black.shape[2])
        )

        # Tens digit - moves when ones digit is rolling
        tens_window = jax.lax.dynamic_slice(
            digit_sprite_black,
            (tens_y, 0, 0),
            (window_height, digit_width, digit_sprite_black.shape[2])
        )

        # Hundreds digit - moves when hundreds digit is rolling
        hundreds_window = jax.lax.dynamic_slice(
            digit_sprite_black,
            (hundreds_y, 0, 0),
            (window_height, digit_width, digit_sprite_black.shape[2])
        )

        # Thousands digit - moves when hundreds digit is rolling
        thousands_window = jax.lax.dynamic_slice(
            digit_sprite_black,
            (thousands_y, 0, 0),
            (window_height, digit_width, digit_sprite_black.shape[2])
        )

        # === Render all numbers ===
        render_y = self.config.distance_odometer_start_y

        # Render the decimal digit window at the specified position
        decimal_x = self.config.distance_odometer_start_x + 4 * (digit_width + 2)  # +2 for the spaces between digits
        raster = aj.render_at(raster, decimal_x, render_y, digit_window)

        # Render the ones digit window at the specified position
        ones_x = self.config.distance_odometer_start_x + 3 * (digit_width + 2)
        raster = aj.render_at(raster, ones_x, render_y, ones_window)

        # Render the tens digit window at the specified position
        tens_x = self.config.distance_odometer_start_x + 2 * (digit_width + 2)
        raster = aj.render_at(raster, tens_x, render_y, tens_window)

        # Render the hundreds digit window at the specified position
        hundreds_x = self.config.distance_odometer_start_x + (digit_width + 2)
        raster = aj.render_at(raster, hundreds_x, render_y, hundreds_window)

        # Render the thousands digit window at the specified position
        thousands_x = self.config.distance_odometer_start_x
        raster = aj.render_at(raster, thousands_x, render_y, thousands_window)

        return raster

    @partial(jax.jit, static_argnums=(0,))
    def _render_level_score(self, raster: jnp.ndarray, state: EnduroGameState) -> jnp.ndarray:
        """
        Renders the current level digit.
        Args:
            raster: the raster for rendering
            state: the game state

        Returns: the new raster with the level score

        """
        # Get digit dimensions from a sample sprite
        current_level = state.level

        # create an array with all sprites to access them based on the current level
        digit_sprites = jnp.stack([
            aj.get_sprite_frame(self.sprites['0_black.npy'], 0),
            aj.get_sprite_frame(self.sprites['1_black.npy'], 0),
            aj.get_sprite_frame(self.sprites['2_black.npy'], 0),
            aj.get_sprite_frame(self.sprites['3_black.npy'], 0),
            aj.get_sprite_frame(self.sprites['4_black.npy'], 0),
            aj.get_sprite_frame(self.sprites['5_black.npy'], 0),
        ])

        raster = aj.render_at(raster, self.config.level_x, self.config.level_y, digit_sprites[current_level])
        return raster

    @partial(jax.jit, static_argnums=(0,))
    def _render_cars_to_overtake_score(self, raster: jnp.ndarray, state: EnduroGameState) -> jnp.ndarray:
        """
        Renders the score that shows how many cars still have to be overtaken.
        Renders flags instead if the level goal has been reached.

        Args:
            raster: the raster for rendering
            state: the game state

        Returns: the new raster with the score
        """

        def render_level_passed(flag_raster) -> jnp.ndarray:
            # change the flag animation every second
            frame_index = (state.step_count // 60) % 2
            flag_sprite = aj.get_sprite_frame(self.sprites['flags.npy'], frame_index)

            # render the flags at the hundreds position - the car symbol
            x_pos = self.config.score_start_x - 9
            flag_raster = aj.render_at(flag_raster, x_pos, self.config.score_start_y - 1, flag_sprite)

            # render the background color of the level score differently (in green)
            background_sprite = aj.get_sprite_frame(self.sprites['green_level_background.npy'], 0)
            flag_raster = aj.render_at(flag_raster, self.config.level_x - 1, self.config.level_y - 1, background_sprite)

            return flag_raster

        def render_digits(digit_raster) -> jnp.ndarray:
            # Get digit dimensions from a sample sprite
            cars_to_overtake = state.cars_to_overtake - state.cars_overtaken
            # create an array with all sprites to access them based on the current score digit
            digit_sprites = jnp.stack([
                aj.get_sprite_frame(self.sprites['0_black.npy'], 0),
                aj.get_sprite_frame(self.sprites['1_black.npy'], 0),
                aj.get_sprite_frame(self.sprites['2_black.npy'], 0),
                aj.get_sprite_frame(self.sprites['3_black.npy'], 0),
                aj.get_sprite_frame(self.sprites['4_black.npy'], 0),
                aj.get_sprite_frame(self.sprites['5_black.npy'], 0),
                aj.get_sprite_frame(self.sprites['6_black.npy'], 0),
                aj.get_sprite_frame(self.sprites['7_black.npy'], 0),
                aj.get_sprite_frame(self.sprites['8_black.npy'], 0),
                aj.get_sprite_frame(self.sprites['9_black.npy'], 0),
            ])

            digit_width = digit_sprites[0].shape[1]

            ones_digit = cars_to_overtake % 10
            tens_digit = (cars_to_overtake // 10) % 10
            hundreds_digit = (cars_to_overtake // 100) % 10

            # load the sprite depending on the digit
            ones_sprite = digit_sprites[ones_digit]
            tens_sprite = digit_sprites[tens_digit]
            hundreds_sprite = digit_sprites[hundreds_digit]

            # Render the ones digit window at the specified position
            ones_x = self.config.score_start_x + 2 * (digit_width + 2)
            digit_raster = aj.render_at(digit_raster, ones_x, self.config.score_start_y, ones_sprite)

            # Only render tens digit if number >= 10
            tens_x = self.config.score_start_x + (digit_width + 2)
            digit_raster = jnp.where(
                cars_to_overtake >= 10,
                aj.render_at(digit_raster, tens_x, self.config.score_start_y, tens_sprite),
                digit_raster
            )

            # Only render hundreds digit if number >= 100
            hundreds_x = self.config.score_start_x
            digit_raster = jnp.where(
                cars_to_overtake >= 100,
                aj.render_at(digit_raster, hundreds_x, self.config.score_start_y, hundreds_sprite),
                digit_raster
            )

            return digit_raster

        # check whether to render the digits or the flags
        raster = lax.cond(
            state.level_passed,
            lambda x: render_level_passed(x),
            lambda x: render_digits(x),
            raster
        )

        return raster

    @partial(jax.jit, static_argnums=(0,))
    def _render_mountains(self, raster: jnp.ndarray, state: EnduroGameState) -> jnp.ndarray:
        """
        Renders mountains. If a mountain sprite would extend beyond the right edge (hi),
        draw an additional wrapped copy at x - period so it appears on the left.
        Returns: the raster for the rendering
        """
        # load mountains
        mountain_left_sprite = aj.get_sprite_frame(self.sprites['mountain_left.npy'], 0)
        mountain_right_sprite = aj.get_sprite_frame(self.sprites['mountain_right.npy'], 0)

        # color them according to the weather
        weather_index = self.config.weather_color_codes[state.weather_index]
        mountain_left_sprite = aj.change_sprite_color(mountain_left_sprite, weather_index[2])
        mountain_right_sprite = aj.change_sprite_color(mountain_right_sprite, weather_index[2])

        # Geometry / sizes
        sky_height = self.background_sizes['background_sky.npy'][0]
        mountain_left_height = self.sprites['mountain_left.npy'].shape[1]
        mountain_right_height = self.sprites['mountain_right.npy'].shape[1]
        mountain_left_width = self.sprites['mountain_left.npy'].shape[2]
        mountain_right_width = self.sprites['mountain_right.npy'].shape[2]

        # Visible interval and period (inclusive interval -> +1)
        lo = self.config.window_offset_left
        hi = self.config.screen_width
        period = (hi - lo + 1)

        # Positions of the mountain edges
        x_left_mountain = state.mountain_left_x
        x_right_mountain = state.mountain_right_x
        y_left_mountain = sky_height - mountain_left_height
        y_right_mountain = sky_height - mountain_right_height

        # 1) Base draw at current positions
        raster = aj.render_at(raster, x_left_mountain, y_left_mountain, mountain_left_sprite)
        raster = aj.render_at(raster, x_right_mountain, y_right_mountain, mountain_right_sprite)

        # 2) If the sprite overflows the right edge (x + width > hi + 1),
        #    draw a wrapped copy at x - period (which lands on the left side).
        overflow_left = (x_left_mountain + mountain_left_width) > (hi + 1)
        overflow_right = (x_right_mountain + mountain_right_width) > (hi + 1)

        raster = lax.cond(
            overflow_left,
            lambda _: aj.render_at(raster, x_left_mountain - period, y_left_mountain, mountain_left_sprite),
            lambda _: raster,
            operand=None,
        )
        raster = lax.cond(
            overflow_right,
            lambda _: aj.render_at(raster, x_right_mountain - period, y_right_mountain, mountain_right_sprite),
            lambda _: raster,
            operand=None,
        )

        # 3) Mask out all pixels at or to the left of window_offset_left
        # Create a mask for valid x coordinates
        x_coords = jnp.arange(raster.shape[1])
        valid_x_mask = x_coords > self.config.window_offset_left

        # Expand mask to match raster dimensions
        valid_x_mask = jnp.expand_dims(valid_x_mask, axis=(0, 2))  # (1, width, 1)
        valid_x_mask = jnp.broadcast_to(valid_x_mask, raster.shape)  # (width, height, channels)

        # Apply mask - set invalid pixels to black (0)
        raster = jnp.where(valid_x_mask, raster, 0)

        return raster

    @partial(jax.jit, static_argnums=(0,))
    def _render_weather(self, raster: jnp.ndarray, state: EnduroGameState) -> jnp.ndarray:
        """
        Renders the skybox and the track background in the color of the current weather.
        Args:
            raster: the current raster
            state: the current game state

        Returns:
            a raster with the weather effects.
        """
        # load the rgb codes for the current weather
        weather_colors = self.config.weather_color_codes[state.weather_index]

        # sky background
        sky = aj.get_sprite_frame(self.sprites['background_sky.npy'], 0)
        colored_sky = aj.change_sprite_color(sky, weather_colors[0])
        raster = aj.render_at(raster, self.config.window_offset_left, 0, colored_sky)

        # green background
        gras = aj.get_sprite_frame(self.sprites['background_gras.npy'], 0)
        colored_gras = aj.change_sprite_color(gras, weather_colors[1])
        raster = aj.render_at(raster, self.config.window_offset_left, sky.shape[0], colored_gras)

        # render the horizon stripes
        stripe_1 = aj.get_sprite_frame(self.sprites['background_horizon.npy'], 0)
        colored_stripe_1 = aj.change_sprite_color(stripe_1, weather_colors[3])
        raster = aj.render_at(raster, self.config.window_offset_left, sky.shape[0] - 2, colored_stripe_1)

        stripe_2 = aj.get_sprite_frame(self.sprites['background_horizon.npy'], 0)
        colored_stripe_2 = aj.change_sprite_color(stripe_2, weather_colors[4])
        raster = aj.render_at(raster, self.config.window_offset_left, sky.shape[0] - 4, colored_stripe_2)

        stripe_3 = aj.get_sprite_frame(self.sprites['background_horizon.npy'], 0)
        colored_stripe_3 = aj.change_sprite_color(stripe_3, weather_colors[5])
        raster = aj.render_at(raster, self.config.window_offset_left, sky.shape[0] - 6, colored_stripe_3)

        return raster

    @partial(jax.jit, static_argnums=(0,))
    def _render_lower_background(self, raster: jnp.ndarray) -> jnp.ndarray:
        """
        Renders the background and score box under the player screen to avoid that opponent cars are visible there.
        Args:
            raster: the current raster
            state: the current game state

        Returns:
            a raster with the background overlay
        """

        # black background
        background_overlay = aj.get_sprite_frame(self.sprites['background_overlay.npy'], 0)
        raster = aj.render_at(raster, 0, self.config.game_window_height - 1, background_overlay)

        # score box
        score_box = aj.get_sprite_frame(self.sprites['score_box.npy'], 0)
        raster = aj.render_at(raster, self.config.info_box_x_pos, self.config.info_box_y_pos, score_box)

        return raster

    def _render_static_background(self) -> jnp.ndarray:
        """
        The background only needs to be rendered or loaded once. For all future frames the array can just be copied,
        which saves some performance. So, only load static sprites here that do not change throughout the game.
        """
        raster = jnp.zeros((self.config.screen_height, self.config.screen_width, 3), dtype=jnp.uint8)

        # black background
        background = aj.get_sprite_frame(self.sprites['background.npy'], 0)
        raster = aj.render_at(raster, 0, 0, background)

        # activision logo
        logo = aj.get_sprite_frame(self.sprites['activision_logo.npy'], 0)
        raster = aj.render_at(raster, self.config.logo_x_position, self.config.logo_y_position, logo)

        return raster

    @partial(jax.jit, static_argnums=(0,))
    def _render_fog(self, raster: jnp.ndarray, state: EnduroGameState) -> jnp.ndarray:
        """
        Renders the fog if applicable. Needs to be done as the last thing, so we don't draw anything on top.
        Args:
            raster: the raster
            state: the Enduro Game State

        Returns:
            a raster with the rendered fog
        """
        # fog bar if the weather is fog
        fog = aj.get_sprite_frame(self.sprites['fog_box.npy'], 0)
        raster = jnp.where(
            state.weather_index == self.config.night_fog_index,
            aj.render_at(raster, self.config.window_offset_left, 0, fog),
            raster
        )
        return raster


"""
ACTIVISION (R)

RULES AND REGULATIONS

ENDURO TM



ACTIVISION NATIONAL ENDURO TM RULES AND REGULATIONS

Strap on your goggles. Sink into your seat. And leave all your
fears in the pit. You're about to enter the race of your life.
You'll be required to pass lots of cars each day. Through sun
and snow and fog and ice, sunrise to sunset - as fast as you
can. Welcome to the National Enduro!

ENDURO TM BASICS

1.   Hook up your video game system. Follow manufacturer's
     instructions.

2.   With power OFF, plug in game cartridge.

3.   Turn the power ON. If no picture appears, check
     connection of your game system to your TV; then repeat
     steps 1-3.

4.   Plug in the LEFT Joystick Controller (right Controller is
     not used).

5.   The difficulty switch and game select switch are not
     used.

6.   To start, press game reset switch.

7.   The Joystick Controller is held with the red button in
     the upper left position. Push the Joystick right or left
     top move your car right or left. The red button is your
     accelerator. The longer you keep the button depressed,
     the faster your car will go, until it reaches top speed.
     To coast at a constant speed, press the red button until
     the desired speed is reached. When you release the
     button, this speed will be maintained. To slow down,
     release the red button and apply the brakes by pulling
     the Joystick back.

8.   Passing cars. The number of cars you must pass is posted
     at the beginning of each day in the lower right corner of
     your instrument panel (200 on the first day, 300 on
     subsequent days). Each time you pass a car, this meter
     counts off by one. When you pass the required number of
     cars, green flags appear. But keep going. All additional
     kilometres are added to your total. You'll move on to the
     next day when the present day ends. If you don't pass the
     required number of cars by daybreak, the game ends.

SPECIAL FEATURES OF ENDURO TM

Time of day. From dawn till the black of night, you'll be on
the road. Pay attention to the lighting and scenery. It
represents the time of day, letting you know how much time is
remaining. And, use caution at night. You can only see the
tail lights of the other cars.

Weather conditions keep changing, so brace yourself. Can you
hang in through ice and fog? A white, icy road means your car
will be less responsive to your steering. A thick, fog-
shrouded screen gives you less time to react, since it will
take you longer to see the cars up ahead.

Days and kilometres. A realistic odometer registers the
kilometres you've covered. Beneath the odometer is the day
indicator, which keeps track of the number of days you've been
on the Enduro circuit. When the race is over, the kilometre
reading on the odometer and the day on the indicator represent
your racing results or score.

Increasing difficulty. The race gets tougher with each new
day. The other cars travel faster and spread out across the
road more and more, making it harder to pass them.

GETTING THE FEEL OF ENDURO RACING

In preparing for a race, every pro driver checks out the
course. Be sure to do the same thing. Get to know the timing
of the weather and lighting conditions. Learn how your car
responds to your touch.

Slow down on the ice and keep your eyes on the patterns of the
cars in the distance. Drive defensively, since the other cars
will not get out of your way. The fog will really test your
reflexes. You'll need to slow down and develop a rapid
steering response to make up for the limited visibility.

JOIN THE ACTIVISION (R) "ROADBUSTERS" 

Do you have the drive, the stamina, the grit to endure this
race for 5 days or more? If so, an on-screen racing trophy
will pop up before your very eyes. Now you can join the
"Roadbusters" and really start breaking records. Send a photo
of the TV screen showing your winning race results, along with
your name and address, to your nearest Activision distributor
(a complete list enclosed). We'll send you the official high
performance emblem.

HOW TO BECOME A "ROADBUSTER"
Tips from Larry Miller, designer of Enduro TM

Larry Miller is a powerhouse game designer with a PhD in
physics. When he isn't designing games, he may be sailing,
skiing or playing the piano. His most recent hit was Spider
Fighter TM.

"The best way to outlast other drivers is to pace yourself.
You won't survive long if you stay at maximum speed because
you'll keep hitting the other cars. Go only as fast as it
takes to pass the required number of cars each day.

"If you can choose between steering into the side of the road
or hitting another car, always steer into the roadside. It's
just a minor setback, and you won't lose as much time.

"Also, it's always better to go around diagonally paired cars
than to squeeze between them. But, if you must squeeze between
them, keep your speed just above theirs and be careful!

"Here's another tip; If you approach a group of cars that are
really blocking the road - slow down. Let them disappear back
into the distance ahead of you. Then, accelerate. When you
meet up with these cars again, they will have probably changed
their positions.

"I hope you enjoy the National Enduro as much as I enjoyed
designing it. Drop me a card from your next pit stop - I'd
love to hear from you. And please, remember to fasten your
seatbelts."

[Photo - Larry Miller beside a 1934 Invicta, one of only five
remaining in the world (courtesy of Paradise Motorcars,
Sacramento, California, USA).]

ACTIVISION (R) VIDEO GAME CARTRIDGE
LIMITED ONE YEAR WARRANTY

Activision, Inc. warrants to the original consumer purchaser
of this Activision video game cartridge that, if the cartridge
is discovered to be defective in materials or workmanship
within one (1) year from the date of purchase, Activision will
either repair or replace, at its option, such cartridge free
of charge, upon receipt of the cartridge, postage prepaid,
with proof of date of purchase, at its distribution center.

This warranty is limited to the electronic circuitry and
mechanical parts originally provided by Activision and is not
applicable to normal wear and tear. This warranty shall not be
applicable and shall be void if the defect in the cartridge
has arisen through abuse, unreasonable use, mistreatment or
neglect. Except as specified in this warranty, Activision
gives no express or implied guarantees, undertakings,
conditions or warranties and makes no representations
concerning the cartridges. In no event will Activision be
responsible under this warranty for any special, incidental,
or consequential damage incurred by any consumer producer.

This warranty and the statements contained herein do not
affect any statutory rights of the consumer against the
manufacturer or supplier of the cartridge.

NOTE: For service in your area, please see the distributor
list.

YOUR BEST GAME SCORES
"""
