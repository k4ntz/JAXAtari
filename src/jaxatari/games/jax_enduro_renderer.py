import os
from typing import Any

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
import pygame

from jaxatari.renderers import JAXGameRenderer
import jaxatari.rendering.jax_rendering_utils as aj

from jax_enduro import GameConfig, JaxEnduro, EnduroGameState

from jaxatari.environment import JAXAtariAction as Action


class EnduroRenderer(JAXGameRenderer):
    """
    Renders the jax_enduro game
    """

    def __init__(self):
        super().__init__()
        self.config = GameConfig()

        # sprite sizes to easily and dynamically  adjust renderings
        self.background_sizes: dict[str, tuple[int, int]] = {}
        self.sprites = self._load_sprites()

        self.track_height = self.background_sizes['background_gras.npy'][1]

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
                    frame = aj.load_frame_with_animation(full_path)
                    # save with the full extension, so remember to also load them with .npy
                    sprites[filename] = frame.astype(jnp.uint8)

                    # Store size info for backgrounds
                    if folder == 'backgrounds':
                        width = frame.shape[1]  # (N, W, H, C)
                        height = frame.shape[2]
                        self.background_sizes[filename] = (width, height)

        return sprites

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: EnduroGameState):
        """Render the game state to a raster image."""
        raster = self.static_background.copy()

        raster = self._render_weather(raster, state)

        # render the player car
        raster = self._render_player_car(raster, state)

        # render the opponents
        raster = self._render_opponent_cars(raster, state)

        # render the lower background again to make opponents below the screen disappear
        raster = self._render_lower_background(raster, state)

        # render the track
        raster = self._render_track_from_state(raster, state)

        # render the distance odometer, level score and cars to overtake
        raster = self._render_distance_odometer(raster, state)
        raster = self._render_level_score(raster, state)
        raster = self._render_cars_to_overtake_score(raster, state)

        # render the mountains
        raster = self._render_mountains(raster, state)

        # render the fog as the last thing!
        raster = self._render_fog(raster, state)

        return raster

    @partial(jax.jit, static_argnums=(0,))
    def _render_player_car(self, raster, state: EnduroGameState):
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
    def _render_track_from_state(self, raster, state: EnduroGameState):
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
        sky_height = self.background_sizes['background_sky.npy'][1]
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
    def _draw_track_sprite(self, x_coords: jnp.ndarray, y_coords: jnp.ndarray, color: jnp.ndarray) -> jnp.ndarray:
        """
        Creates a sprite for the track that covers the whole screen, which makes it a little easier to draw,
        because you can use absolute x,y positions for drawing the pixels.
        """
        # Create full-screen-sized sprite
        sprite = jnp.zeros((self.config.screen_width, self.config.screen_height, 4), dtype=jnp.uint8)

        def draw_pixel(i, s):
            x = x_coords[i]
            y = y_coords[i]
            return s.at[x, y].set(color)

        sprite = jax.lax.fori_loop(0, x_coords.shape[0], draw_pixel, sprite)
        return sprite

    @partial(jax.jit, static_argnums=(0,))
    def _render_opponent_cars(self, raster, state: EnduroGameState) -> jnp.ndarray:
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

        # Load all car sprites as separate variables (they have different shapes)
        car_0 = aj.get_sprite_frame(self.sprites['car_0.npy'], frame_index)
        car_1 = aj.get_sprite_frame(self.sprites['car_1.npy'], frame_index)
        car_2 = aj.get_sprite_frame(self.sprites['car_2.npy'], frame_index)
        car_3 = aj.get_sprite_frame(self.sprites['car_3.npy'], frame_index)
        car_4 = aj.get_sprite_frame(self.sprites['car_4.npy'], frame_index)
        car_5 = aj.get_sprite_frame(self.sprites['car_5.npy'], frame_index)
        car_6 = aj.get_sprite_frame(self.sprites['car_6.npy'], frame_index)

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

        # Apply colors to each sprite
        colored_car_0 = apply_color_to_sprite(car_0, colors[0])
        colored_car_1 = apply_color_to_sprite(car_1, colors[1])
        colored_car_2 = apply_color_to_sprite(car_2, colors[2])
        colored_car_3 = apply_color_to_sprite(car_3, colors[3])
        colored_car_4 = apply_color_to_sprite(car_4, colors[4])
        colored_car_5 = apply_color_to_sprite(car_5, colors[5])
        colored_car_6 = apply_color_to_sprite(car_6, colors[6])

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
    def _render_distance_odometer(self, raster, state: EnduroGameState) -> jnp.ndarray:
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
        window_height = digit_0_black.shape[1] + 2
        digit_width = digit_0_black.shape[0]

        # get the black and brown digit sprites that have all digits
        digit_sprite_black = aj.get_sprite_frame(self.sprites['black_digit_array.npy'], 0)
        digit_sprite_brown = aj.get_sprite_frame(self.sprites['brown_digit_array.npy'], 0)

        # determine the base position in the sprite that represents the lowest y for the window
        base_y = digit_sprite_brown.shape[1] - window_height + 1

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
            (0, decimal_y, 0),  # start indices (y position in sprite, x=0, channel=0)
            (digit_width, window_height, digit_sprite_brown.shape[2])
        )

        # Add rolling animation offsets when appropriate
        # Ones digit - moves when decimal digit is rolling
        ones_window = jax.lax.dynamic_slice(
            digit_sprite_black,
            (0, ones_y, 0),
            (digit_width, window_height, digit_sprite_black.shape[2])
        )

        # Tens digit - moves when ones digit is rolling
        tens_window = jax.lax.dynamic_slice(
            digit_sprite_black,
            (0, tens_y, 0),
            (digit_width, window_height, digit_sprite_black.shape[2])
        )

        # Hundreds digit - moves when hundreds digit is rolling
        hundreds_window = jax.lax.dynamic_slice(
            digit_sprite_black,
            (0, hundreds_y, 0),
            (digit_width, window_height, digit_sprite_black.shape[2])
        )

        # Thousands digit - moves when hundreds digit is rolling
        thousands_window = jax.lax.dynamic_slice(
            digit_sprite_black,
            (0, thousands_y, 0),
            (digit_width, window_height, digit_sprite_black.shape[2])
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
    def _render_level_score(self, raster, state: EnduroGameState) -> jnp.ndarray:
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
    def _render_cars_to_overtake_score(self, raster, state: EnduroGameState) -> jnp.ndarray:
        """
        Renders the score that shows how many cars still have to be overtaken.

        Args:
            raster: the raster for rendering
            state: the game state

        Returns: the new raster with the score
        """
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

        digit_width = digit_sprites[0].shape[0]

        ones_digit = cars_to_overtake % 10
        tens_digit = (cars_to_overtake // 10) % 10
        hundreds_digit = (cars_to_overtake // 100) % 10

        # load the sprite depending on the digit
        ones_sprite = digit_sprites[ones_digit]
        tens_sprite = digit_sprites[tens_digit]
        hundreds_sprite = digit_sprites[hundreds_digit]

        # Render the ones digit window at the specified position
        ones_x = self.config.score_start_x + 2 * (digit_width + 2)
        raster = aj.render_at(raster, ones_x, self.config.score_start_y, ones_sprite)

        # Render the tens digit window at the specified position
        tens_x = self.config.score_start_x + (digit_width + 2)
        raster = aj.render_at(raster, tens_x, self.config.score_start_y, tens_sprite)

        # Render the hundreds digit window at the specified position
        hundreds_x = self.config.score_start_x
        raster = aj.render_at(raster, hundreds_x, self.config.score_start_y, hundreds_sprite)

        return raster

    def _render_mountains(self, raster, state: EnduroGameState) -> jnp.ndarray:
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
        sky_height = self.background_sizes['background_sky.npy'][1]
        mountain_left_height = self.sprites['mountain_left.npy'].shape[2]
        mountain_right_height = self.sprites['mountain_right.npy'].shape[2]
        mountain_left_width = self.sprites['mountain_left.npy'].shape[1]
        mountain_right_width = self.sprites['mountain_right.npy'].shape[1]

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
        x_coords = jnp.arange(raster.shape[0])
        valid_x_mask = x_coords > self.config.window_offset_left

        # Expand mask to match raster dimensions
        valid_x_mask = jnp.expand_dims(valid_x_mask, axis=(1, 2))  # (width, 1, 1)
        valid_x_mask = jnp.broadcast_to(valid_x_mask, raster.shape)  # (width, height, channels)

        # Apply mask - set invalid pixels to black (0)
        raster = jnp.where(valid_x_mask, raster, 0)

        return raster

    def _render_weather(self, raster, state: EnduroGameState) -> jnp.ndarray:
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
        raster = aj.render_at(raster, self.config.window_offset_left, sky.shape[1], colored_gras)

        # render the horizon stripes
        stripe_1 = aj.get_sprite_frame(self.sprites['background_horizon.npy'], 0)
        colored_stripe_1 = aj.change_sprite_color(stripe_1, weather_colors[3])
        raster = aj.render_at(raster, self.config.window_offset_left, sky.shape[1] - 2, colored_stripe_1)

        stripe_2 = aj.get_sprite_frame(self.sprites['background_horizon.npy'], 0)
        colored_stripe_2 = aj.change_sprite_color(stripe_2, weather_colors[4])
        raster = aj.render_at(raster, self.config.window_offset_left, sky.shape[1] - 4, colored_stripe_2)

        stripe_3 = aj.get_sprite_frame(self.sprites['background_horizon.npy'], 0)
        colored_stripe_3 = aj.change_sprite_color(stripe_3, weather_colors[5])
        raster = aj.render_at(raster, self.config.window_offset_left, sky.shape[1] - 6, colored_stripe_3)

        return raster

    @partial(jax.jit, static_argnums=(0,))
    def _render_lower_background(self, raster, state: EnduroGameState) -> jnp.ndarray:
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
        raster = jnp.zeros((self.config.screen_width, self.config.screen_height, 3), dtype=jnp.uint8)

        # black background
        background = aj.get_sprite_frame(self.sprites['background.npy'], 0)
        raster = aj.render_at(raster, 0, 0, background)

        # activision logo
        logo = aj.get_sprite_frame(self.sprites['activision_logo.npy'], 0)
        raster = aj.render_at(raster, self.config.logo_x_position, self.config.logo_y_position, logo)

        return raster

    def _render_fog(self, raster, state: EnduroGameState) -> jnp.ndarray:
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


DIRECTION_LABELS = {
    -1: "Left",
    0: "Straight",
    1: "Right"
}


def render_debug_overlay(screen, state: EnduroGameState, font, game_config):
    """Render debug information as pygame text overlay"""
    track_direction_starts_at = state.whole_track[:, 1]
    track_segment_index = int(jnp.searchsorted(track_direction_starts_at, state.distance, side='right') - 1)
    track_direction = state.whole_track[track_segment_index, 0]
    debug_info = [
        f"Speed: {float(state.player_speed):.2f}",  # Convert JAX arrays to Python floats
        f"Player X (abs): {state.player_x_abs_position}",
        f"Player Y (abs): {state.player_y_abs_position}",
        # f"Distance: {state.distance}",
        f"Level: {state.level}",
        f"Time: {state.total_time_elapsed}",
        f"Left Mountain x: {state.mountain_left_x}",
        f"Opponent Index: {state.opponent_index}",
        # f"Opponent window: {state.opponent_window}",
        # f"Opponents: {state.visible_opponent_positions}",
        # f"Cars overtaken: {state.cars_overtaken}",
        f"Opponent Collision: {state.is_collision}",
        # f"Cooldown Drift direction: {state.cooldown_drift_direction}"
        f"Weather: {state.weather_index}",
        # f"Track direction: {DIRECTION_LABELS.get(int(track_direction))} ({track_direction})",
        # f"Track top X: {state.track_top_x}",
        # f"Top X Offset: {state.track_top_x_curve_offset}",
    ]

    # Semi-transparent background for better readability
    overlay = pygame.Surface((250, len(debug_info) * 25 + 20))  # Made slightly larger
    overlay.set_alpha(180)
    overlay.fill((0, 0, 0))
    screen.blit(overlay, (10, 10))

    # Render each debug line
    for i, text in enumerate(debug_info):
        text_surface = font.render(text, True, (255, 255, 255))
        screen.blit(text_surface, (15, 15 + i * 25))


def play_enduro(debug_mode=True):
    """
    Plays the game with a renderer

    Args:
        debug_mode: If True, shows debug overlay. Set to False for production/optimized runs.
    """
    pygame.init()
    # Initialize game and renderer
    game = JaxEnduro()
    renderer = EnduroRenderer()
    scaling = 4

    screen = pygame.display.set_mode((160 * scaling, 210 * scaling))
    pygame.display.set_caption("Enduro" + (" - DEBUG MODE" if debug_mode else ""))

    font = pygame.font.Font(None, 20)  # You can adjust size as needed
    small_font = pygame.font.Font(None, 16)

    # Always JIT compile the core game functions
    # This ensures JIT compatibility is tested even during debugging
    step_fn = jax.jit(game.step)
    render_fn = jax.jit(renderer.render)
    reset_fn = jax.jit(game.reset)

    init_obs, state = reset_fn()

    # Setup game loop
    clock = pygame.time.Clock()
    running = True
    done = False

    print(f"Starting game in {'DEBUG' if debug_mode else 'PRODUCTION'} mode")
    print("Core game functions are JIT compiled for performance and compatibility testing")
    if debug_mode:
        print("Press 'D' to toggle debug overlay")

    show_debug = debug_mode  # Can be toggled during gameplay

    while running and not done:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_d and debug_mode:
                    show_debug = not show_debug
                    print(f"Debug overlay: {'ON' if show_debug else 'OFF'}")

        # Handle input
        keys = pygame.key.get_pressed()
        # allow arrows and wsad
        if (keys[pygame.K_a] or keys[pygame.K_LEFT]) and keys[pygame.K_SPACE]:
            action = Action.LEFTFIRE
        elif (keys[pygame.K_d] or keys[pygame.K_RIGHT]) and keys[pygame.K_SPACE]:
            action = Action.RIGHTFIRE
        elif (keys[pygame.K_s] or keys[pygame.K_DOWN]) and keys[pygame.K_SPACE]:
            action = Action.DOWNFIRE
        elif keys[pygame.K_SPACE]:
            action = Action.FIRE
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
            action = Action.LEFT
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            action = Action.RIGHT
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            action = Action.DOWN
        else:
            action = Action.NOOP

        # Update game state
        obs, state, reward, done, info = step_fn(state, action)

        # Render game frame
        frame = render_fn(state)
        aj.update_pygame(screen, frame, scaling, 160, 210)

        # Add debug overlay if enabled
        if debug_mode and show_debug:
            render_debug_overlay(screen, state, font, renderer.config)

            # Add controls help in corner
            help_text = small_font.render("Press 'D' to toggle debug", True, (200, 200, 200))
            screen.blit(help_text, (screen.get_width() - 180, screen.get_height() - 20))

        pygame.display.flip()

        # Cap at 60 FPS (or 30 for debug mode to make it easier to read)
        clock.tick(30 if debug_mode else 60)

    # If game over, wait before closing
    if done:
        pygame.time.wait(2000)


if __name__ == '__main__':
    # For debugging and development
    play_enduro(debug_mode=True)
