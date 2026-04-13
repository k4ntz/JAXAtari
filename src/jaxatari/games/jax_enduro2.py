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
    player_x: jnp.ndarray
    player_y: jnp.ndarray
    step_count: jnp.ndarray
    player_speed: jnp.ndarray
    distance: jnp.ndarray
    cars_to_pass: jnp.ndarray

@struct.dataclass
class Enduro2Observation:
    # Minimal observation for now
    pass

@struct.dataclass
class Enduro2Info:
    pass

class Enduro2Constants(AutoDerivedConstants):
    """Game configuration parameters for Enduro2"""
    screen_width: int = struct.field(pytree_node=False, default=160)
    screen_height: int = struct.field(pytree_node=False, default=210)
    
    player_x_start: float = struct.field(pytree_node=False, default=72.0)
    player_y_start: float = struct.field(pytree_node=False, default=144.0)
    
    steering_speed: float = struct.field(pytree_node=False, default=2.0)
    max_speed: float = struct.field(pytree_node=False, default=120.0)
    min_speed: float = struct.field(pytree_node=False, default=6.0)
    frame_rate: float = struct.field(pytree_node=False, default=60.0)
    
    window_offset_left: int = struct.field(pytree_node=False, default=8)
    window_offset_bottom: int = struct.field(pytree_node=False, default=55)
    sky_height: int = struct.field(pytree_node=False, default=50)

    # UI Positions
    info_box_x_pos: int = struct.field(pytree_node=False, default=48)
    info_box_y_pos: int = struct.field(pytree_node=False, default=161)
    distance_odometer_start_x: int = struct.field(pytree_node=False, default=65)
    score_start_x: int = struct.field(pytree_node=False, default=81)
    score_start_y: Optional[int] = struct.field(pytree_node=False, default=None)
    distance_odometer_start_y: Optional[int] = struct.field(pytree_node=False, default=None)

    # Asset config for simplified renderer
    ASSET_CONFIG: tuple = struct.field(pytree_node=False, default_factory=lambda: (
        {'name': 'background', 'type': 'background', 'file': 'backgrounds/background.npy'},
        {'name': 'background_overlay', 'type': 'single', 'file': 'backgrounds/background_overlay.npy'},
        {'name': 'score_box', 'type': 'single', 'file': 'backgrounds/score_box.npy'},
        {'name': 'sky_color', 'type': 'procedural', 'data': jnp.array([[[45, 50, 184, 255]]], dtype=jnp.uint8)},
        {'name': 'grass_color', 'type': 'procedural', 'data': jnp.array([[[0, 68, 0, 255]]], dtype=jnp.uint8)},
        {'name': 'track_color', 'type': 'procedural', 'data': jnp.array([[[111, 111, 111, 255]]], dtype=jnp.uint8)},
        {'name': 'digits_black', 'type': 'digits', 'pattern': 'digits/{}_black.npy'},
        {'name': 'black_digit_array', 'type': 'single', 'file': 'digits/black_digit_array.npy'},
        {'name': 'brown_digit_array', 'type': 'single', 'file': 'digits/brown_digit_array.npy'},
    ))

    def compute_derived(self) -> Dict[str, Any]:
        game_window_height = self.screen_height - self.window_offset_bottom
        km_per_second_per_speed_unit = 0.035 / self.min_speed
        km_per_speed_unit_per_frame = km_per_second_per_speed_unit / self.frame_rate
        
        return {
            'game_window_height': game_window_height,
            'km_per_speed_unit_per_frame': km_per_speed_unit_per_frame,
            'distance_odometer_start_y': game_window_height + 9,
            'score_start_y': game_window_height + 25,
        }
    
    game_window_height: Optional[int] = struct.field(pytree_node=False, default=None)
    km_per_speed_unit_per_frame: Optional[float] = struct.field(pytree_node=False, default=None)

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
        if not os.path.exists(player_car_path):
            pass
        player_car_data = jnp.load(player_car_path)
        
        asset_config = list(self.consts.ASSET_CONFIG)
        asset_config.append({'name': 'player_car', 'type': 'procedural', 'data': player_car_data})
        
        # Load assets
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS,
        ) = self.jr.load_and_setup_assets(asset_config, self._sprite_path)
        
        # Store color IDs
        self.sky_id = self.COLOR_TO_ID.get((45, 50, 184), 0)
        self.grass_id = self.COLOR_TO_ID.get((0, 68, 0), 0)
        self.track_id = self.COLOR_TO_ID.get((111, 111, 111), 0)

        # Store Odometer Sheet ID Masks
        self.black_digit_sheet_mask = self.SHAPE_MASKS['black_digit_array']
        self.brown_digit_sheet_mask = self.SHAPE_MASKS['brown_digit_array']

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: Enduro2GameState) -> jnp.ndarray:
        # Start with the static background (mostly black)
        raster = self.BACKGROUND
        
        xx, yy = self.jr._xx, self.jr._yy
        
        # 1. Draw Sky (top part of game window)
        sky_mask = (xx >= self.consts.window_offset_left) & (yy < self.consts.sky_height)
        raster = jnp.where(sky_mask, self.sky_id, raster)
        
        # 2. Draw Grass (bottom part of game window)
        grass_mask = (xx >= self.consts.window_offset_left) & (yy >= self.consts.sky_height) & (yy < self.consts.game_window_height)
        raster = jnp.where(grass_mask, self.grass_id, raster)
        
        # 3. Draw straight track with perspective
        track_center = self.consts.window_offset_left + (self.consts.screen_width - self.consts.window_offset_left) // 2
        y_rel = yy - self.consts.sky_height
        track_height = self.consts.game_window_height - self.consts.sky_height
        # Scale track width from narrow at horizon to wide at bottom
        track_half_width = 8 + (45 * y_rel / jnp.maximum(1, track_height))
        
        track_mask = (yy >= self.consts.sky_height) & (yy < self.consts.game_window_height) & (jnp.abs(xx - track_center) <= track_half_width)
        raster = jnp.where(track_mask, self.track_id, raster)
        
        # 4. Render the lower background overlays
        raster = self.jr.render_at(raster, 0, self.consts.game_window_height - 1, self.SHAPE_MASKS['background_overlay'])
        raster = self.jr.render_at(raster, self.consts.info_box_x_pos, self.consts.info_box_y_pos, self.SHAPE_MASKS['score_box'])
        
        # 5. Render UI
        raster = self._render_distance_odometer(raster, state)
        raster = self._render_cars_to_pass(raster, state)

        # 6. Render player car
        player_mask = self.SHAPE_MASKS['player_car']
        if player_mask.ndim == 3:
             player_mask = player_mask[0]
             
        raster = self.jr.render_at(raster, state.player_x, state.player_y, player_mask)
        
        # Convert ID raster to RGB
        return self.jr.render_from_palette(raster, self.PALETTE)

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
        Renders the "Cars to pass" digits.
        """
        digit_sprites = self.SHAPE_MASKS['digits_black']
        
        hundreds = (state.cars_to_pass // 100) % 10
        tens = (state.cars_to_pass // 10) % 10
        ones = state.cars_to_pass % 10
        
        spacing = digit_sprites.shape[2] + 2
        
        # Only show hundreds if > 0
        hundreds_mask = digit_sprites[hundreds]
        raster = jax.lax.cond(
            state.cars_to_pass >= 100,
            lambda r: self.jr.render_at(r, self.consts.score_start_x, self.consts.score_start_y, hundreds_mask),
            lambda r: r,
            raster
        )
        
        # Only show tens if >= 10
        tens_mask = digit_sprites[tens]
        raster = jax.lax.cond(
            state.cars_to_pass >= 10,
            lambda r: self.jr.render_at(r, self.consts.score_start_x + spacing, self.consts.score_start_y, tens_mask),
            lambda r: r,
            raster
        )
        
        ones_mask = digit_sprites[ones]
        raster = self.jr.render_at(raster, self.consts.score_start_x + 2 * spacing, self.consts.score_start_y, ones_mask)
        
        return raster


class JaxEnduro2(JaxEnvironment[Enduro2GameState, Enduro2Observation, Enduro2Info, Enduro2Constants]):
    ACTION_SET: jnp.ndarray = jnp.array(
        [
            Action.NOOP,
            Action.FIRE,
            Action.RIGHT,
            Action.LEFT,
            Action.DOWN,
        ],
        dtype=jnp.int32,
    )

    def __init__(self, consts: Enduro2Constants = None):
        super().__init__(consts or Enduro2Constants())
        self.renderer = Enduro2Renderer(consts=self.consts)

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTION_SET))

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.consts.screen_height, self.consts.screen_width, 3),
            dtype=jnp.uint8
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jrandom.PRNGKey = None) -> Tuple[Enduro2Observation, Enduro2GameState]:
        state = Enduro2GameState(
            player_x=jnp.array(self.consts.player_x_start, dtype=jnp.float32),
            player_y=jnp.array(self.consts.player_y_start, dtype=jnp.float32),
            step_count=jnp.array(0, dtype=jnp.int32),
            player_speed=jnp.array(self.consts.min_speed, dtype=jnp.float32),
            distance=jnp.array(0.0, dtype=jnp.float32),
            cars_to_pass=jnp.array(200, dtype=jnp.int32)
        )
        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: Enduro2GameState, action: int) -> Tuple[Enduro2Observation, Enduro2GameState, float, bool, Enduro2Info]:
        # Translate compact agent action index to ALE console action
        atari_action = jnp.take(self.ACTION_SET, action.astype(jnp.int32))

        # Simple steering
        is_left = (atari_action == Action.LEFT) | (atari_action == Action.LEFTFIRE)
        is_right = (atari_action == Action.RIGHT) | (atari_action == Action.RIGHTFIRE)

        dx = jnp.where(is_left, -self.consts.steering_speed, jnp.where(is_right, self.consts.steering_speed, 0.0))

        # Basic movement logic
        new_player_x = jnp.clip(state.player_x + dx, 0, self.consts.screen_width - 16)

        # Update distance
        distance_delta = state.player_speed * self.consts.km_per_speed_unit_per_frame
        new_distance = state.distance + distance_delta

        new_state = state.replace(
            player_x=new_player_x,
            distance=new_distance,
            step_count=state.step_count + 1
        )

        obs = self._get_observation(new_state)
        reward = self._get_reward(state, new_state)
        done = self._get_done(new_state)
        info = self._get_info(new_state)

        return obs, new_state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, state: Enduro2GameState, new_state: Enduro2GameState) -> float:
        return 0.0

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: Enduro2GameState) -> bool:
        return False

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: Enduro2GameState) -> Enduro2Info:
        return Enduro2Info()

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: Enduro2GameState) -> jnp.ndarray:        return self.renderer.render(state)

    def _get_observation(self, state: Enduro2GameState) -> Enduro2Observation:
        return Enduro2Observation()
