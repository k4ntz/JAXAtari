"""
Dear Quentin, Raban, Jannis, Paul, Sebastian,

I implemented the Enduro game.
https://www.free80sarcade.com/atari2600_Enduro.php

I recommend to add a function like change_sprite_color to aj or to allow get_sprite_frame to take an custom rgb array.

"""

from functools import partial
import chex
import jax
import jax.numpy as jnp
import jax.random as jrandom
from dataclasses import dataclass
from typing import Tuple, NamedTuple
import numpy as np

# from jax import Array
# from jax import debug
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
import jax.lax as lax

"""
Observations from playing the game

Driving:
- Car speed is maintained, the gas doe not need to be pressed like a pedal
    - once accelerated the car can never stop again, it always maintains a minimum speed
    - slowest is 0,1 km per second
    - fastest is 0,5 km per second
    - It takes 5 second to get to full speed
    - The tire animation speed increases wit increasing speed
- With increasing speed the car moves forward, 1 pixel at a time
    - 5 presses of the trigger equals one pixel
    - opponent speed equals 10 presses of the action button
    - animation becomes faster when speed is higher
- Breaking reduces your speed by ~3
- Drift is about 2-3 pixels per second
- Hitting the side of the road reduces you speed a bit

opponent Cars:
- Enemies do not change lane position
    - they do have the same speed (around 10) that does not change
    - they spawn in a way that is non-blocking
    - if you slow down they overtake you again
    - they do not crash into you when overtaking, they spawn at the other side to avoid collision
- Cars overtaking you makes the counter go backwards
- Hitting an opponent reduces speed to 6 (not zero) and creates a cooldown
    - during cooldown there is no steering and accelerating
    - 
- No red cars

Environment:
- Weather cycles:
    - Day
    - Fog 0:52
    - Evening: 1:25
    - Night: 2:00
    - Fog 2:35
    - Dawn 3:08
    - Over 3:42
- Curves are 1-15 km long
Track:
- steering causes the track to move in the opposite direction, the car moves in the steering direction
- every 400 m the Track has a "bumper"


"""

class EnduroConstants(NamedTuple):
    FRAME_RATE: int = 60

@dataclass
class GameConfig:
    """Game configuration parameters"""
    # Game runs at 60 frames per second
    frame_rate: int = 60
    # Game logic steps every Nth frame (standard is 4)
    frame_skip: int = 4

    screen_width: int = 160
    screen_height: int = 210

    # Enduro has a window that is smaller
    window_offset_left: int = 8
    window_offset_bottom: int = 55
    game_window_width: int = screen_width - window_offset_left
    game_window_height: int = screen_height - window_offset_bottom
    game_screen_middle: int = (screen_width + window_offset_left) // 2

    # the track is in the game window below the sky
    sky_height = 50

    # car sizes from close to far
    car_width_0: int = 16
    car_height_0: int = 11

    # for all different car sizes the widths and heights
    car_widths = jnp.array([16, 12, 8, 6, 4, 4, 2], dtype=jnp.int32)
    car_heights = jnp.array([11, 8, 6, 4, 3, 2, 1], dtype=jnp.int32)

    score_box_y: int = 10
    score_box_height: int = 27

    # player car position
    player_x_start: float = (screen_width + window_offset_left) / 2
    player_y_start: float = game_window_height - car_height_0 - 1

    # === Track ===
    track_width: int = 97
    track_height: int = game_window_height - sky_height - 2
    max_track_length: float = 9999.9  # in km
    track_seed: int = 42
    track_x_start: int = player_x_start
    track_max_curvature_width: int = 17
    track_max_top_x_offset: float = 50.0
    curve_rate: float = 1.0 / frame_skip  # how fast the track curve turns in the game

    cars_to_pass_per_level: int = 200
    cars_increase_per_level: int = 100
    max_increase_level: int = 5

    # === Speed controls ===
    min_speed: int = 6  # from RAM state 22
    max_speed: int = 120  # from RAM state 22
    km_per_speed_unit_per_second: int = 0.0028  # from playtesting
    km_per_speed_unit_per_frame: int = km_per_speed_unit_per_second / frame_rate

    # from measuring the RAM states the car accelerates with this function, where t = number of seconds:
    # f(t) = 10.5t where f <= 46
    # f(t) = 3.75t where f > 46
    acceleration_per_second: float = 10.5
    acceleration_per_frame: float = acceleration_per_second / frame_rate
    acceleration_slow_down_factor: float = 0.5
    acceleration_slow_down_threshold: float = 46.0

    breaking_per_second: float = 30.0  # controls how fast the car break
    breaking_per_frame: float = breaking_per_second / frame_rate

    # === Steering ===
    steering_range_in_pixels: int = 28
    # How much the car moves per steering input (absolute units)
    steering_sensitivity: float = steering_range_in_pixels / (3.0 * frame_rate)  # ~3 seconds from edge to edge

    # drift_per_second_relative: float = 0.2
    # drift_per_frame: float = drift_per_second_relative / frame_rate
    drift_per_second_pixels: float = 2.5  # controls how much the car drifts in a curve
    drift_per_frame: float = drift_per_second_pixels / frame_rate

    # === Track collision ===
    track_collision_kickback_pixels: float = 3.0
    track_collision_speed_reduction: float = 15.0  # from RAM extraction (15)

    # === Weather ===
    night_fog_index: int = 12
    # Start times in seconds for each phase. Written in a way to allow easy replacements.
    weather_starts_s: jnp.ndarray = jnp.array([
        0,  # day 1
        34,  # day 2 (lighter)
        34 + 34,  # day 3 (white mountains)
        34 + 34 + 69,  # fog day
        34 + 34 + 69 + 8 * 1,  # Sunset 1
        34 + 34 + 69 + 8 * 2,  # Sunset 2
        34 + 34 + 69 + 8 * 3,  # Sunset 3
        34 + 34 + 69 + 8 * 4,  # Sunset 4
        34 + 34 + 69 + 8 * 5,  # Sunset 5
        34 + 34 + 69 + 8 * 6,  # Sunset 6
        34 + 34 + 69 + 8 * 7,  # Sunset 7
        34 + 34 + 69 + 8 * 8,  # Sunset 8
        34 + 34 + 69 + 8 * 8 + 69,  # night
        34 + 34 + 69 + 8 * 8 + 69 + 69,  # fog night
        34 + 34 + 69 + 8 * 8 + 69 + 69 + 34,  # night 2
        34 + 34 + 69 + 8 * 8 + 69 + 69 + 34 + 34,  # dawn
    ], dtype=jnp.int32)
    day_night_cycle_seconds: int = weather_starts_s[15]

    # The rgb color codes for each weather and each sprite scraped from the game
    weather_color_codes: jnp.ndarray = jnp.array([
        # sky,          gras,       mountains,      horizon 1,      horizon 2,  horizon 3 (highest)

        # day
        [[24, 26, 167], [0, 68, 0], [134, 134, 29], [24, 26, 167], [24, 26, 167], [24, 26, 167], ],  # day 1
        [[45, 50, 184], [0, 68, 0], [136, 146, 62], [45, 50, 184], [45, 50, 184], [45, 50, 184]],  # day 2
        [[45, 50, 184], [0, 68, 0], [192, 192, 192], [45, 50, 184], [45, 50, 184], [45, 50, 184]],  # day white mountain
        [[45, 50, 184], [236, 236, 236], [214, 214, 214], [45, 50, 184], [45, 50, 184], [45, 50, 184]],  # fog day

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
        [[74, 74, 74], [0, 0, 0], [142, 142, 142], [74, 74, 74], [74, 74, 74], [74, 74, 74]],
        [[74, 74, 74], [0, 0, 0], [142, 142, 142], [74, 74, 74], [74, 74, 74], [74, 74, 74]],
        [[74, 74, 74], [0, 0, 0], [142, 142, 142], [74, 74, 74], [74, 74, 74], [74, 74, 74]],

        # dawn
        [[111, 111, 111], [0, 0, 0], [181, 83, 40], [111, 111, 111], [111, 111, 111], [111, 111, 111]],

    ], dtype=jnp.int32)

    # === Opponents ===
    opponent_speed: int = 24  # measured from RAM state
    # a factor of 1 translates into overtake time of 1 second when speed is twice as high as the opponent's
    opponent_relative_speed_factor: float = 1.0

    number_of_opponents = 5000
    opponent_density = 0.2
    opponent_delay_slots = 10

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

    # === Other ===

    car_crash_cooldown_seconds: float = 3.0
    car_crash_cooldown_frames: int = jnp.array(car_crash_cooldown_seconds * frame_rate)
    crash_kickback_speed_per_frame: float = track_width / car_crash_cooldown_seconds / frame_rate / 3

    # === Cosmetics ===
    logo_x_position: int = 20
    logo_y_position: int = 196

    info_box_x_pos: int = 48
    info_box_y_pos: int = 161

    distance_odometer_start_x: int = 65
    distance_odometer_start_y: int = game_window_height + 9

    score_start_x: int = 80
    score_start_y: int = game_window_height + 25

    level_x: int = 57
    level_y: int = score_start_y

    mountain_left_x_pos: float = 40.0
    mountain_right_x_pos: float = 120.0
    mountain_pixel_movement_per_frame_per_speed_unit: float = 0.01

    # how many steps per animation
    opponent_animation_steps: int = 2

    day_length_frames = day_night_cycle_seconds * frame_rate


class EnduroGameState(NamedTuple):
    """Represents the current state of the game"""

    step_count: jnp.int32  # incremented every step (so every n-th frame depending on frame skip)

    # visible (mirror in Observation)
    player_y_abs_position: chex.Array
    player_x_abs_position: chex.Array
    cars_overtaken: chex.Array
    cars_to_overtake: chex.Array  # goal for current level
    distance: chex.Array
    level: chex.Array

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
    cooldown: chex.Array  # cooldown after collision
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

    def __init__(self, sprite_path_car="sprites/enduro/cars/car_0.npy"):
        """
        Args:
            sprite_path_car: Path to the .npy file for the largest sprite,
                                  which may contain multiple animation frames.
        """

        # We only need the collision data from the largest sprite (car_0)
        largest_sprite_data = np.load(sprite_path_car)  # 'car_0.npy'

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


class OpponentCar(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    scale_level: jnp.ndarray


class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray


class EnduroObservation(NamedTuple):
    car: EntityPosition  # player car position
    cars_to_overtake: jnp.ndarray  # goal for current level
    distance: jnp.ndarray
    level: jnp.ndarray
    visible_opponents: chex.Array
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
        self.config = GameConfig()
        self.state = self.reset()
        self.car_0_spec = VehicleSpec()
        self.car_1_spec = VehicleSpec("sprites/enduro/cars/car_1.npy")

        self.action_set = [
            Action.NOOP,
            Action.FIRE,  # gas
            Action.LEFT,  # steer left
            Action.RIGHT,  # steer right
            Action.DOWN,  # brake
            Action.LEFTFIRE,  # steer left + gas
            Action.RIGHTFIRE,  # steer right + gas
            Action.DOWNFIRE  # brake + gas (not common but valid)
        ]

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jrandom.PRNGKey = None) -> Tuple[EnduroObservation, EnduroGameState]:
        whole_track = self.build_whole_track(seed=self.config.track_seed)
        # use same position as the player
        top_x = jnp.round(self.config.player_x_start).astype(jnp.int32)
        left_xs, right_xs = self.generate_viewable_track(top_x, 0.0)

        # opponents
        opponent_spawns = self.generate_opponent_spawns(
            seed=self.config.track_seed,
            number_of_opponents=self.config.number_of_opponents,
            opponent_density=self.config.opponent_density,
            opponent_delay_slots=self.config.opponent_delay_slots
        )
        visible_opponent_positions = self.get_visible_opponent_positions(jnp.array(0.0), opponent_spawns, left_xs,
                                                                         right_xs)

        state = EnduroGameState(
            # visible
            step_count=jnp.array(0),
            player_x_abs_position=jnp.array(self.config.player_x_start),
            player_y_abs_position=jnp.array(self.config.player_y_start),
            cars_to_overtake=jnp.array(self.config.cars_to_pass_per_level),
            cars_overtaken=jnp.array(0),
            distance=jnp.array(0.0, dtype=jnp.float32),
            level=jnp.array(1),

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
            time_remaining=jnp.array(self.config.day_night_cycle_seconds),
            total_cars_overtaken=jnp.array(0),
            total_time_elapsed=jnp.array(0.0, dtype=jnp.float32),
            total_frames_elapsed=jnp.array(0),
        )

        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: EnduroGameState, action: int) -> StepResult:
        """
        Performs a frame-skipped step in the Enduro environment.

        Executes the environment step logic multiple times (as configured by
        `frame_skip`) to simulate time progression, returning only the final
        observation and total accumulated reward. Skipping stops early if the
        game ends.

        Args:
            state (EnduroGameState): The current game state.
            action (int): The discrete action to apply repeatedly.

        Returns:
            Tuple[EnduroObservation, EnduroGameState, float, bool, EnduroInfo]:
                - The final observation after frame skipping.
                - The updated game state.
                - The total accumulated reward over skipped frames.
                - A boolean indicating if the episode has ended.
                - Additional info such as level and distance.
        """

        def skip_step(i, carry):
            """
            Applies a single frame update during frame skipping.
            Carries forward the observation, state, and accumulated reward,
            and stops early if the game ends.
            """
            obs_i, state_i, reward_acc, done_flag = carry
            obs_next, state_next, reward_i, done_i, _ = self._step_single(state_i, action)
            reward_acc += reward_i
            return obs_next, state_next, reward_acc, jnp.logical_or(done_flag, done_i)

        initial_obs, initial_state, initial_reward, initial_done, _ = self._step_single(state, action)

        obs, final_state, total_reward, done = jax.lax.fori_loop(
            1,  # start at 1 because we already did the first frame
            self.config.frame_skip,
            skip_step,
            (initial_obs, initial_state, initial_reward, initial_done)
        )
        # add one logical step
        final_state: EnduroGameState = final_state._replace(step_count=state.step_count + 1)

        info = self._get_info(final_state)
        return obs, final_state, total_reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _step_single(self, state: EnduroGameState, action: int) -> StepResult:
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
        new_weather_index = jnp.searchsorted(
            self.config.weather_starts_s,
            state.step_count / self.config.frame_rate,
            side='right') - 1

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
                    self.config.acceleration_per_frame * self.config.acceleration_slow_down_factor
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

            steering_delta = jnp.where(is_left, -self.config.steering_sensitivity,
                                       jnp.where(is_right, self.config.steering_sensitivity, 0.0))

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
        #    This is the value we want to eventually reach.
        target_offset = curvature * self.config.track_max_top_x_offset  # e.g., -1 * 50 = -50, or 0 * 50 = 0

        # 3. Calculate the difference (the "error") between where we are and where we want to be.
        offset_error = target_offset - state.track_top_x_curve_offset

        # 4. Limit the change per step. The change cannot be faster than curve_rate.
        #    jnp.clip is perfect here. This moves us towards the target without overshooting.
        offset_change = jnp.clip(offset_error, -self.config.curve_rate, self.config.curve_rate)

        # 5. Apply the calculated change to the current offset.
        new_top_x_curve_offset = state.track_top_x_curve_offset + offset_change

        # 6. Generate the new track with the top_x of the track and its offset
        new_left_xs, new_right_xs = self.generate_viewable_track(new_track_top_x, new_top_x_curve_offset)

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
            new_speed - self.config.track_collision_speed_reduction,  # If collided, reduce speed.
            new_speed  # If not, keep the new speed.
        )
        # Ensure speed does not drop below the minimum value
        new_speed = jnp.maximum(1.0, new_speed)  # Use maximum() to enforce a floor.

        # 3. Kickback
        # The kickback direction is simply the inverse of `collision_side`.
        track_kickback_direction = -collision_side

        new_x_abs = jnp.where(
            collided_track,
            # Apply the kickback based on the actual collision side.
            new_x_abs + (self.config.track_collision_kickback_pixels * track_kickback_direction),
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
        new_visible_opponent_positions = self.get_visible_opponent_positions(
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

        # ===== Level =====
        # a level is a full day cycle
        new_level = jnp.where(
            state.level * self.config.day_night_cycle_seconds * self.config.frame_rate > state.step_count,
            state.level + 1,
            state.level
        )

        level_goal_reached = jnp.where(
            new_cars_overtaken >= self.config.cars_to_pass_per_level + self.config.cars_increase_per_level * state.level,
            1,
            0
        )

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

        # ====== OTHER ======
        # Compute game over condition based on elapsed frames
        # day_length_frames = self.config.day_night_cycle_seconds * self.config.frame_rate
        # game_over = state.total_frames_elapsed >= self.config.day_length_frames

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

        # Build new state with updated positions
        new_state: EnduroGameState = state._replace(
            player_x_abs_position=new_x_abs,
            player_y_abs_position=new_y_abs,
            total_time_elapsed=state.step_count / self.config.frame_rate,
            total_frames_elapsed=state.total_frames_elapsed + 1,
            distance=new_distance,
            player_speed=new_speed,
            level=new_level,

            opponent_index=new_opponent_index,
            visible_opponent_positions=new_visible_opponent_positions,
            opponent_window=new_opponent_window,
            cars_overtaken=new_cars_overtaken,
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
            cars_to_overtake=state.cars_to_overtake,
            distance=state.distance,
            level=state.level,
            visible_opponents=state.visible_opponent_positions,
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
    def get_action_space(self):
        return jnp.array(self.action_set)

    @partial(jax.jit, static_argnums=(0,))
    def generate_viewable_track(
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

        # --- Step 4: Generate the right track based on the final left track ---
        final_right_xs = self._generate_other_track_side_coords(final_left_xs)

        return final_left_xs.astype(jnp.int32), final_right_xs.astype(jnp.int32)

    @partial(jax.jit, static_argnums=(0,))
    def _generate_other_track_side_coords(self, left_xs: jnp.ndarray) -> jnp.ndarray:
        """
        Returns (x_coords, y_coords) for the right boundary of the track.
        Skips rows where space == -1 by collapsing to left_x.
        """

        spaces = self._generate_track_spaces()
        x = jnp.where(spaces == -1, left_xs, left_xs + spaces + 1)  # +1 to include gap
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

        spaces = jnp.zeros(self.config.track_height, dtype=jnp.int32)
        spaces = lax.fori_loop(0, 103, body_fn, spaces)

        return spaces

    @partial(jax.jit, static_argnums=(0, 2, 3, 4))
    def generate_opponent_spawns(
            self,
            seed: int,
            number_of_opponents: int,
            opponent_density: float,
            opponent_delay_slots: int
    ) -> jnp.ndarray:
        """
        Generate a precomputed spawn sequence with an *exact* occupancy equal to
        round(opponent_density * number_of_enemies) while forbidding any contiguous
        triple of non-gaps that covers all three lanes {0,1,2} in any order.

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
        num_occupied = int(round(opponent_density * number_of_opponents))

        # Generate random positions for occupied slots
        key, key_positions = jax.random.split(key)
        all_indices = jnp.arange(number_of_opponents)
        shuffled_indices = jax.random.permutation(key_positions, all_indices)
        occupied_positions = shuffled_indices[:num_occupied]

        # Create occupancy mask
        occupancy_mask = jnp.zeros(number_of_opponents, dtype=jnp.bool_)
        occupancy_mask = occupancy_mask.at[occupied_positions].set(True)

        # Generate lane assignments for occupied slots
        key, key_lanes = jax.random.split(key)
        lane_choices = jax.random.randint(key_lanes, (number_of_opponents,), 0, 3, dtype=jnp.int8)

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
        color_keys = jax.random.split(key_colors, number_of_opponents)
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
    def get_visible_opponent_positions(self, opponent_index: jnp.ndarray,
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
        # Lane ratios: left = 0.25, middle = 0.5, right = 0.75 of track width
        lane_ratios = jnp.array([0.25, 0.5, 0.75], dtype=jnp.float32)

        # Get car widths for each opponent slot (0=closest/largest, 6=farthest/smallest)
        car_widths = self.config.car_widths  # Shape: (7,)

        def calculate_x_for_lane(slot_idx, lane_code, left_bound, track_width):
            # For empty slots (-1), return -1 as "not visible" marker
            valid_lane = jnp.clip(lane_code, 0, 2)  # Clamp to valid range
            ratio = lane_ratios[valid_lane]

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
    def build_whole_track(self, seed: int) -> jnp.ndarray:
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
        segment_lengths = jax.random.uniform(subkey, shape=(max_segments,), minval=1.0, maxval=15.0)

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

if __name__ == '__main__':
    vs = VehicleSpec()
    print(vs.collision_mask)
    print(vs.collision_mask_relative_ys)
    print(vs.collision_mask_relative_xs)
    print(vs.num_solid_pixels)

    # # Mapping string inputs to action enum
    # INPUT_TO_ACTION = {
    #     "": Action.NOOP,
    #     "noop": Action.NOOP,
    #     "fire": Action.FIRE,
    #     "left": Action.LEFT,
    #     "right": Action.RIGHT,
    #     "down": Action.DOWN,
    #     "leftfire": Action.LEFTFIRE,
    #     "rightfire": Action.RIGHTFIRE,
    #     "downfire": Action.DOWNFIRE,
    # }
    #
    # game = JaxEnduro()
    # game_obs, game_state = game.reset()
    #
    # step_counter = 0
    # is_game_over = False
    #
    # DIRECTION_LABELS = {
    #     -1: "Left",
    #     0: "Straight",
    #     1: "Right"
    # }
    #
    # print("\n🕹️ Welcome to JAX Enduro (Manual Test Loop)")
    # print("Valid inputs:", list(INPUT_TO_ACTION.keys()))
    # print("Type `exit` to quit\n")
    #
    # # --- Game Loop ---
    # while not is_game_over:
    #     action_str = input("Enter action: ").strip().lower()
    #     if action_str == "exit":
    #         break
    #
    #     if action_str not in INPUT_TO_ACTION:
    #         print("Invalid action. Try again.")
    #         continue
    #
    #     user_action = INPUT_TO_ACTION[action_str]
    #
    #     # Step the environment
    #     game_obs, game_state, game_reward, is_game_over, game_info = game.step(game_state, user_action)
    #
    #     # Print step info
    #     print(f"\n🎬 Step {step_counter}")
    #     track_direction_starts_at = game_state.whole_track[:, 1]
    #     track_segment_index = int(jnp.searchsorted(track_direction_starts_at, game_state.distance, side='right') - 1)
    #     track_direction = game_state.whole_track[track_segment_index, 0]
    #     print(f"Track direction: {DIRECTION_LABELS.get(int(track_direction))}")
    #     print(f"Action: {action_str.upper()} ({user_action})")
    #     print(f"Car X Position (abs): {float(game_state.player_x_abs_position):.3f}")
    #     print(f"Cars to Overtake: {int(game_state.cars_to_overtake)}")
    #     print(f"Distance: {float(game_state.distance):.6f}")
    #     print(f"Speed: {float(game_state.player_speed)}")
    #     print(f"Level: {int(game_state.level)}")
    #     print(f"Reward: {float(game_reward)}")
    #     print(f"Game Over: {bool(is_game_over)}")
    #     print(f"Total Frames Elapsed: {int(game_state.total_frames_elapsed)}\n")
    #
    #     step_counter += 1
    #
    # print("🏁 Game Ended.")

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
