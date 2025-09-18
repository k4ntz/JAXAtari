"""
JAX Slot Machine - A classic slot machine implementation for JAXAtari.

This module implements a traditional three-reel slot machine game using JAX for GPU acceleration
and JIT compilation. The game features authentic slot machine mechanics with spinning reels,
various symbols, and a payout system.

Author: Ashish Bhandari, ashish.bhandari@stud.tu-darmstadt.de, https://github.com/zatakashish

License: TU Darmstadt, All rights reserved.

========================================================================================================================
                                                GAMEPLAY MECHANICS
========================================================================================================================

There are currently two tabs one at the top of the third reel and one at the bottom.

- The upper right tab shows credit
- The lower right tab shows wager

How to play ?
- SPACE to turn on the reels
- UP to increase wager
- DOWN to decrease wager

Reward System: (More information under https://www.atarimania.com/game-atari-2600-vcs-slot-machine_8212.html)

The payline (from left to right) pays as follows:

- Cactus on reel 1 only -> 2x wager
- Cactus on reels 1 & 2 -> 5x wager
- Table, Table, Bar     -> 10x wager
- Table, Table, Table   -> 10x wager
- TV, TV, Bar           -> 14x wager
- TV, TV, TV            -> 14x wager
- Bell, Bell, Bar       -> 18x wager
- Bell, Bell, Bell      -> 18x wager
- Bar, Bar, Bar         -> 100x wager
- Car, Car, Car         -> 200x wager

========================================================================================================================
"""

import os
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Any, Optional, NamedTuple, Tuple

import chex
import jax
import jax.lax
import jax.numpy as jnp
import jax.random

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as aj
import jaxatari.spaces as spaces


@dataclass(frozen=True)
class SlotMachineConfig:
    """

    Configuration class for Slot Machine game parameters.
    This class holds all the tweakable parameters for the slot machine.

    """
    # Screen dimensions
    screen_width: int = 160
    screen_height: int = 210
    scaling_factor: int = 3

    # Reel layout
    num_reels: int = 3
    reel_width: int = 40
    reel_height: int = 120
    reel_spacing: int = 10
    symbols_per_reel: int = 3
    total_symbols_per_reel: int = 20

    # Symbol configuration
    num_symbol_types: int = 6
    symbol_height: int = 28
    symbol_width: int = 28

    # Game start finances
    starting_credits: int = 25
    bet_amount: int = 1
    min_wager: int = 1
    max_wager: int = 5
    #max_wager: int = 500 # For debug only. TODO Remove later

    # Reel timing
    min_spin_duration: int = 60
    max_spin_duration: int = 120
    reel_stop_delay: int = 30

    # Symbol probability weights (higher = more common)
    # Symbols mapping ("Cactus", "Table", "Bar", "TV", "Bell", "Car")
    symbol_weights: jnp.ndarray = field(
        default_factory=lambda: jnp.array([2, 3, 2, 3, 2, 1], dtype=jnp.float32)
    )

    # Reel positions - UI layout coordinates
    reel_start_x: int = 11
    reel_start_y: int = 50


class SlotMachineState(NamedTuple):
    """
    Complete immutable game state for Slot Machine.
    """

    # Core game progression state
    credits: chex.Array
    total_winnings: chex.Array
    spins_played: chex.Array

    # Betting system
    current_wager: chex.Array

    # Reel mechanics
    reel_positions: chex.Array
    reel_spinning: chex.Array
    spin_timers: chex.Array
    reel_speeds: chex.Array

    # Input handling with a button debouncing feature so that we ignore repeated presses
    spin_button_prev: chex.Array
    up_button_prev: chex.Array
    down_button_prev: chex.Array
    spin_cooldown: chex.Array

    # Visual effects
    win_flash_timer: chex.Array
    last_payout: chex.Array
    last_reward: chex.Array

    # RNG key that keeps the spins honest
    rng: chex.Array


class SlotMachineObservation(NamedTuple):
    """Observation returned to the agent each step."""

    credits: jnp.ndarray          # Current credits
    current_wager: jnp.ndarray    # Current bet amount
    reel_symbols: jnp.ndarray     # Visible symbols (shape: num_reels × symbols_per_reel)
    is_spinning: jnp.ndarray      # Is the machine currently spinning?
    last_payout: jnp.ndarray      # Last win amount
    last_reward: jnp.ndarray      # Last reward


class SlotMachineInfo(NamedTuple):
    """
    Information about the game state.
    """

    # Saves total winnings
    total_winnings: jnp.ndarray

    # Total number of spins altogether, not currently needed. Introduced for debug and win statistics.
    spins_played: jnp.ndarray


class SlotMachineConstants(NamedTuple):
    """
    Game constants and symbol definitions.
    """

    symbol_names: tuple = ("Cactus", "Table", "Bar", "TV", "Bell", "Car")


class SlotMachineRenderer(JAXGameRenderer):
    """

    This class loads authentic slot machine symbols from .npy sprite files.
    All sprite files are present in src/jaxatari/games/sprites/slotmachine/.

    Sprite Files (All files are 40x40 RGBA numpy arrays)
    - Cactus.npy
    - Table.npy
    - Bar.npy
    - TV.npy
    - Bell.npy
    - Car.npy

    """

    def __init__(self, config: SlotMachineConfig = None):

        super().__init__()
        self.config = config or SlotMachineConfig()
        self.sprites = self._load_sprites()

    def _load_sprites(self) -> Dict[str, Any]:
        """
        Load all necessary sprites and rescale them by 70%. 70% is just a design choice so it looks good.

        """
        sprites = {}

        # Load sprite files from the sprites directory
        sprite_dir = "src/jaxatari/games/sprites/slotmachine"

        # Scale factor for all sprites (70%)
        scale_factor = 0.7

        # Symbol names mapping
        symbol_names = ["Cactus", "Table", "Bar", "TV", "Bell", "Car"]

        for i, symbol_name in enumerate(symbol_names):
            npy_file = f"{sprite_dir}/{symbol_name}.npy"

            import numpy as np

            sprite_data = np.load(npy_file)
            original_sprite = jnp.array(sprite_data, dtype=jnp.uint8)

            rescaled_sprite = self._rescale_sprite(original_sprite, scale_factor)

            sprites[f'symbol_{i}'] = rescaled_sprite
            sprites[symbol_name] = rescaled_sprite

        sprites['background'] = self._create_background()
        sprites['reel_frame'] = self._create_reel_frame()
        #sprites['digit_sprites'] = self._load_digit_sprites()

        return sprites

    def _rescale_sprite(self, sprite: jnp.ndarray, scale_factor: float) -> jnp.ndarray:

        """Resize ``sprite`` with nearest-neighbour sampling. Used 40x40 array at first, but then I had to do this to
        downsize the sprites, I know this is overengineering but  that's what I learnt  in computer vision :D """

        original_height, original_width = sprite.shape[:2]
        new_height = int(original_height * scale_factor)
        new_width = int(original_width * scale_factor)

        y_coords = jnp.linspace(0, original_height - 1, new_height).astype(jnp.int32)
        x_coords = jnp.linspace(0, original_width - 1, new_width).astype(jnp.int32)

        y_grid, x_grid = jnp.meshgrid(y_coords, x_coords, indexing='ij')

        rescaled_sprite = sprite[y_grid, x_grid]

        return rescaled_sprite

    def _create_background(self) -> jnp.ndarray:

        """
         Create classic atari greenish tint with blue frame.

         """

        h, w = self.config.screen_height, self.config.screen_width
        bg = jnp.zeros((h, w, 4), dtype=jnp.uint8)

        # Fill entire background with greenish tint
        bg = bg.at[:, :, :3].set(jnp.array([140, 208, 140], dtype=jnp.uint8))
        bg = bg.at[:, :, 3].set(jnp.uint8(255))

        # Add blue border padding with 15 pixel in upper and lower corners, and 3 pixel left and right corner.
        # Since our game is in Portrait mode, this is exactly the opposite in real game.
        border_color = jnp.array([70, 82, 184], dtype=jnp.uint8)

        bg = bg.at[:15, :, :3].set(border_color)
        bg = bg.at[-15:, :, :3].set(border_color)
        bg = bg.at[:, :3, :3].set(border_color)
        bg = bg.at[:, -3:, :3].set(border_color)

        return bg

    def _create_reel_frame(self) -> jnp.ndarray:
        # Exact reel size
        h, w = self.config.reel_height, self.config.reel_width
        frame = jnp.zeros((h, w, 4), dtype=jnp.uint8)

        # Plain filled interior
        light_blue = jnp.array([70, 82, 184], dtype=jnp.uint8)  # RGB
        alpha = jnp.uint8(255)  # opaque; lower for translucency if desired

        frame = frame.at[..., :3].set(light_blue)
        frame = frame.at[..., 3].set(alpha)
        return frame

    def _get_symbol_sprite(self, symbol_type: chex.Array) -> jnp.ndarray:
        """Fetch the sprite that corresponds to ``symbol_type``."""
        symbol_names = ["Cactus", "Table", "Bar", "TV", "Bell", "Car"]

        symbol_sprites = jnp.stack([
            self.sprites[name] for name in symbol_names
        ])

        return symbol_sprites[symbol_type]

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: SlotMachineState) -> chex.Array:
        """
        Render the complete game state to a pixel array.

        Rendering steps:
        1. Copy in the background.
        2. Draw each reel frame and its three symbols.
        3. Dim symbols on reels that are still spinning.
        4. Overlay the HUD counters (credits and wager panels).
        5. Flash the whole scene if we are celebrating a win.

        """
        cfg = self.config

        # Start with classic attari green background
        raster = self.sprites['background'][..., :3]

        # Render each reel
        for reel_idx in range(cfg.num_reels):
            reel_x = cfg.reel_start_x + reel_idx * (cfg.reel_width + cfg.reel_spacing)
            reel_y = cfg.reel_start_y

            # Render reel frame first
            frame_sprite = self.sprites['reel_frame']
            raster = aj.render_at(raster, reel_x, reel_y, frame_sprite)

            # Get reel state for this specific reel
            reel_pos = state.reel_positions[reel_idx]
            is_spinning = state.reel_spinning[reel_idx]

            # Render the 3 visible symbols in this reel
            for symbol_slot in range(cfg.symbols_per_reel):
                symbol_slot_y = reel_y + symbol_slot * 40

                # Calculate which symbol to show at this position
                # This math wraps around the 20-symbol cycle as in the original game
                symbol_index = (reel_pos + symbol_slot) % cfg.total_symbols_per_reel
                symbol_type = symbol_index % cfg.num_symbol_types

                # Get the sprite for this symbol type
                symbol_sprite = self._get_symbol_sprite(symbol_type)

                # Apply blur effect if spinning (makes it look like the reel is in motion)
                # This simple darkening effect si what gives it a spinning illusion
                sprite_to_render = jax.lax.cond(
                    is_spinning,
                    lambda s: (s.at[..., :3].multiply(0.7)).astype(s.dtype),
                    lambda s: s,
                    symbol_sprite
                )

                # Center the sprite within the 40x40 slot and position in the middle as rescaled to 0,7
                centered_x = reel_x + 6
                centered_y = symbol_slot_y + 6

                raster = aj.render_at(raster, centered_x, centered_y, sprite_to_render)

        # Render UI elements
        raster = self._render_credits_display(raster, state.credits, state.current_wager, state)

        # Apply win flash effect if we hit a win
        raster = jax.lax.cond(
            state.win_flash_timer > 0,
            lambda r: self._render_win_flash(r, state.win_flash_timer),
            lambda r: r,
            raster
        )

        return raster

    def _render_credits_display(self, raster: jnp.ndarray, credits: chex.Array, wager: chex.Array, state: SlotMachineState) -> jnp.ndarray:
        """Draw the credits and wager box."""
        raster = self._render_text_labels(raster, credits, wager)

        return raster

    def _render_text_labels(self, raster: jnp.ndarray, credits: chex.Array, wager: chex.Array) -> jnp.ndarray:
        """Helper function to draw the HUD plates and insert the credit and wager.
        """
        cfg = self.config

        # Calculate third reel position
        third_reel_x = cfg.reel_start_x + 2 * (cfg.reel_width + cfg.reel_spacing)
        third_reel_y = cfg.reel_start_y

        # Colors
        box_color = jnp.array([70, 82, 184], dtype=jnp.uint8)
        number_color = jnp.array([194, 67, 115], dtype=jnp.uint8)

        # UI for credits
        credits_y = third_reel_y - 22
        raster = self._draw_colored_box(raster, third_reel_x - 3, credits_y - 2, 47, 16, box_color)
        credits_digits = aj.int_to_digits(credits, max_digits=4)
        raster = self._render_colored_number(raster, credits_digits, third_reel_x + 2, credits_y + 2, number_color)

        # UI for wager
        wager_y = third_reel_y + cfg.reel_height + 10
        raster = self._draw_colored_box(raster, third_reel_x + 7, wager_y - 4, 27, 16, box_color)
        wager_digits = aj.int_to_digits(wager, max_digits=2)
        raster = self._render_colored_number(raster, wager_digits, third_reel_x + 12, wager_y, number_color)

        return raster

    def _render_colored_number(self, raster: jnp.ndarray, digits: jnp.ndarray, x: int, y: int, color: jnp.ndarray) -> jnp.ndarray:
        """
        Render a number using colored digit sprites.
        """

        spacing = 10

        for i in range(digits.shape[0]):
            digit_idx = digits[i]
            digit_x = x + i * spacing

            raster = jax.lax.cond(
                digit_idx >= 0,
                lambda r: self._render_colored_digit(r, digit_idx, digit_x, y, color),
                lambda r: r,
                raster
            )

        return raster

    def _render_colored_digit(self, raster: jnp.ndarray, digit: int, x: int, y: int, color: jnp.ndarray) -> jnp.ndarray:

        # # Pre-define all digit patterns as a JAX array

        digit_patterns = jnp.array([
            # 0
            [[0,1,1,1,1,0], [1,1,0,0,1,1], [1,1,0,0,1,1], [1,1,0,0,1,1], [1,1,0,0,1,1], [1,1,0,0,1,1], [1,1,0,0,1,1], [0,1,1,1,1,0]],
            # 1
            [[0,0,1,1,0,0], [0,1,1,1,0,0], [1,1,1,1,0,0], [0,0,1,1,0,0], [0,0,1,1,0,0], [0,0,1,1,0,0], [0,0,1,1,0,0], [1,1,1,1,1,1]],
            # 2
            [[1,1,1,1,1,0], [1,1,0,0,1,1], [0,0,0,0,1,1], [0,0,1,1,1,0], [0,1,1,0,0,0], [1,1,0,0,0,0], [1,1,0,0,0,0], [1,1,1,1,1,1]],
            # 3
            [[1,1,1,1,1,0], [0,0,0,0,1,1], [0,0,0,0,1,1], [0,1,1,1,1,0], [0,0,0,0,1,1], [0,0,0,0,1,1], [0,0,0,0,1,1], [1,1,1,1,1,0]],
            # 4
            [[1,1,0,0,1,1], [1,1,0,0,1,1], [1,1,0,0,1,1], [1,1,1,1,1,1], [0,0,0,0,1,1], [0,0,0,0,1,1], [0,0,0,0,1,1], [0,0,0,0,1,1]],
            # 5
            [[1,1,1,1,1,1], [1,1,0,0,0,0], [1,1,0,0,0,0], [1,1,1,1,1,0], [0,0,0,0,1,1], [0,0,0,0,1,1], [0,0,0,0,1,1], [1,1,1,1,1,0]],
            # 6
            [[0,1,1,1,1,0], [1,1,0,0,0,0], [1,1,0,0,0,0], [1,1,1,1,1,0], [1,1,0,0,1,1], [1,1,0,0,1,1], [1,1,0,0,1,1], [0,1,1,1,1,0]],
            # 7
            [[1,1,1,1,1,1], [0,0,0,0,1,1], [0,0,0,1,1,0], [0,0,1,1,0,0], [0,1,1,0,0,0], [0,1,1,0,0,0], [0,1,1,0,0,0], [0,1,1,0,0,0]],
            # 8
            [[0,1,1,1,1,0], [1,1,0,0,1,1], [1,1,0,0,1,1], [0,1,1,1,1,0], [1,1,0,0,1,1], [1,1,0,0,1,1], [1,1,0,0,1,1], [0,1,1,1,1,0]],
            # 9
            [[0,1,1,1,1,0], [1,1,0,0,1,1], [1,1,0,0,1,1], [1,1,0,0,1,1], [0,1,1,1,1,1], [0,0,0,0,1,1], [0,0,0,0,1,1], [0,1,1,1,1,0]],
        ], dtype=jnp.int32)

        # Get the pattern for a certain digit
        pattern = digit_patterns[digit]

        # Create coordinate grids
        rows, cols = jnp.ogrid[0:8, 0:6]
        pixel_y = y + rows
        pixel_x = x + cols

        # Create masks for bounds checking and pixel values
        in_bounds_y = (pixel_y >= 0) & (pixel_y < raster.shape[0])
        in_bounds_x = (pixel_x >= 0) & (pixel_x < raster.shape[1])
        in_bounds = in_bounds_y & in_bounds_x
        should_draw = (pattern > 0) & in_bounds

        # Create indices for valid pixels
        valid_indices = jnp.where(should_draw, 1, 0)

        def update_pixel(i, raster_state):
            row_i = i // 6
            col_i = i % 6
            py = y + row_i
            px = x + col_i

            # Check if this pixel should be drawn
            should_update = (pattern[row_i, col_i] > 0) & (py >= 0) & (py < raster.shape[0]) & (px >= 0) & (px < raster.shape[1])

            return jax.lax.cond(
                should_update,
                lambda r: r.at[py, px, :].set(color),
                lambda r: r,
                raster_state
            )

        # Apply updates using JAX scan
        return jax.lax.fori_loop(0, 8 * 6, update_pixel, raster)

    def _draw_colored_box(self, raster: jnp.ndarray, x: int, y: int, width: int, height: int, color: jnp.ndarray) -> jnp.ndarray:
        """Fill a rectangle and clamp edges so we do not paint off-screen."""
        x1 = max(0, min(x, raster.shape[1]))
        y1 = max(0, min(y, raster.shape[0]))
        x2 = max(0, min(x + width, raster.shape[1]))
        y2 = max(0, min(y + height, raster.shape[0]))

        raster = raster.at[y1:y2, x1:x2, :].set(color)

        return raster

    def _render_win_flash(self, raster: jnp.ndarray, flash_timer: chex.Array) -> jnp.ndarray:
        """Apply a brief brightness pulse after a win."""
        flash_intensity = jnp.sin(flash_timer * 0.3) * 0.2 + 0.8

        raster_float = raster.astype(jnp.float32)
        flashed_raster = raster_float * flash_intensity

        return jnp.clip(flashed_raster, 0, 255).astype(jnp.uint8)


class JaxSlotMachine(JaxEnvironment[SlotMachineState, SlotMachineObservation, SlotMachineInfo, SlotMachineConstants]):
    """
    JAX-accelerated implementation of a classic slot machine game.

    This is the main game class that ties everything together. It implements
    the JAXAtari environment interface so it can be used with RL frameworks,
    analysis tools, and the standard JAXAtari ecosystem.

    """

    def __init__(
            self,
            config: SlotMachineConfig = None,
            reward_funcs: list[callable] = None,
    ):
        """Instantiate the environment and its renderer."""
        self.config = config or SlotMachineConfig()
        consts = SlotMachineConstants()
        super().__init__(consts)

        self.renderer = SlotMachineRenderer(self.config)

        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs

        # Define available actions
        self.action_set = [
            Action.NOOP,
            Action.FIRE,
            Action.UP,
            Action.DOWN,
        ]

    def reset(
            self, key: jax.random.PRNGKey = None
    ) -> Tuple[SlotMachineObservation, SlotMachineState]:
        """
        Reset the environment to initial state for a new game.

        Creates a fresh game state with starting credits and random reel positions.
        This is called at the beginning of each episode for RL training,
        or when the player runs out of credits.
        """
        cfg = self.config

        # Generate a truly random key if none provided.
        # I would be interested to know if in the future, after millions of epoch, the RL Agent finds a pattern to when
        # the game is run. Please keep me posted.
        if key is None:
            import time
            key = jax.random.PRNGKey(int(time.time() * 1000000) % (2**31))

        # Initialize reels to random positions to prevent every fresh game from starting with the same symbols showing
        key, *reel_keys = jax.random.split(key, cfg.num_reels + 1)
        initial_positions = jnp.array([
            jax.random.randint(reel_key, (), 0, cfg.total_symbols_per_reel)
            for reel_key in reel_keys
        ])

        # Create completely fresh initial state
        initial_state = SlotMachineState(
            # Economic state at start
            credits=jnp.array(cfg.starting_credits, dtype=jnp.int32), # Start with 25
            total_winnings=jnp.array(0, dtype=jnp.int32),
            spins_played=jnp.array(0, dtype=jnp.int32),

            # Betting state
            current_wager=jnp.array(cfg.min_wager, dtype=jnp.int32),   # Start with 1

            # Reel state - random positions, everything stopped
            reel_positions=initial_positions,                          # Random start positions
            reel_spinning=jnp.zeros(cfg.num_reels, dtype=jnp.bool_),  # All reels stopped
            spin_timers=jnp.zeros(cfg.num_reels, dtype=jnp.int32),    # No active timers
            reel_speeds=jnp.ones(cfg.num_reels, dtype=jnp.int32),     # Default speed

            # Input state - no buttons pressed, no cooldowns
            spin_button_prev=jnp.array(False, dtype=jnp.bool_),       # FIRE not pressed
            up_button_prev=jnp.array(False, dtype=jnp.bool_),         # UP not pressed
            down_button_prev=jnp.array(False, dtype=jnp.bool_),       # DOWN not pressed
            spin_cooldown=jnp.array(0, dtype=jnp.int32),              # No cooldown

            # Visual effects state - clean slate
            win_flash_timer=jnp.array(0, dtype=jnp.int32),            # No win flash
            last_payout=jnp.array(0, dtype=jnp.int32),                # No recent payout
            last_reward=jnp.array(0.0, dtype=jnp.float32),            # No recent reward

            # Randomness state
            rng=key,
        )

        obs = self._get_observation(initial_state)
        return obs, initial_state

    def step(
            self, state: SlotMachineState, action: int
    ) -> Tuple[SlotMachineObservation, SlotMachineState, float, bool, SlotMachineInfo]:
        """
        Execute one step of the slot machine game. This is the heart of the game logic. Called every frame to process
        player input and update the game state.

        Processing Pipeline:
        1. Process player input (button presses)
        2. Handle wager adjustments
        3. Start new spins if requested
        4. Update reel animations
        5. Check for wins when reels stop
        6. Update timers and effects
        7. Calculate rewards and check game over

        """
        cfg = self.config

        # Update RNG key EVERY step, not just during spins
        step_key, new_rng = jax.random.split(state.rng)

        # Process player input
        fire_pressed = (action == Action.FIRE)
        up_pressed = (action == Action.UP)
        down_pressed = (action == Action.DOWN)

        # Detect "just pressed" events to prevent button mashing
        fire_just_pressed = fire_pressed & (~state.spin_button_prev)
        up_just_pressed = up_pressed & (~state.up_button_prev)
        down_just_pressed = down_pressed & (~state.down_button_prev)

        # Determine if player can spin
        # Need: no cooldown, button just pressed, enough credits, reels not spinning
        can_spin = (state.spin_cooldown == 0) & fire_just_pressed & (state.credits >= state.current_wager)
        can_spin = can_spin & (~jnp.any(state.reel_spinning))

        # Handle wager changes (only when machine is idle)
        can_change_wager = ~jnp.any(state.reel_spinning) & (state.spin_cooldown == 0)

        # Wager adjustment logic
        new_wager = jax.lax.cond(
            can_change_wager & up_just_pressed,
            lambda w: jnp.minimum(w + 1, cfg.max_wager),
            lambda w: jax.lax.cond(
                can_change_wager & down_just_pressed,
                lambda w: jnp.maximum(w - 1, cfg.min_wager),
                lambda w: w,
                w
            ),
            state.current_wager
        )

        # Update state with input tracking, wager changes and most importantly new RNG
        new_state = state._replace(
            spin_button_prev=fire_pressed,
            up_button_prev=up_pressed,
            down_button_prev=down_pressed,
            current_wager=new_wager,
            rng=new_rng
        )

        # Start spin if conditions are met
        new_state = jax.lax.cond(
            can_spin,
            lambda s: self._start_spin(s, step_key),
            lambda s: s,
            new_state
        )

        # Update reel animations and physics
        new_state = self._update_reels(new_state)

        # Check for wins when all reels have stopped
        new_state = self._check_for_wins(new_state)

        # Update various timers and cooldowns
        new_state = self._update_timers(new_state)

        # Calculate reward for this step
        reward = jnp.where(
            (new_state.last_payout > 0) & (state.last_payout == 0),
            new_state.last_payout.astype(jnp.float32),
            0.0
        )

        # Store reward in state for display purposes. Introduced during debug to check if the reward provided is correct.
        new_state = new_state._replace(last_reward=reward)

        # Game over condition - can't afford next minimum wager AND reels are not spinning
        done = (new_state.credits < cfg.min_wager) & (~jnp.any(new_state.reel_spinning))

        obs = self._get_observation(new_state)
        info = SlotMachineInfo(
            total_winnings=new_state.total_winnings,
            spins_played=new_state.spins_played
        )

        return obs, new_state, reward, done, info

    def _start_spin(self, state: SlotMachineState, key: jax.random.PRNGKey) -> SlotMachineState:
        """
        Start spinning the reels using provided RNG key. This function sets up a new spin: deducts credits,
        generates random outcomes, and starts the reel animations.
        """
        cfg = self.config

        # Deduct the wager immediately
        new_credits = state.credits - state.current_wager
        new_spins = state.spins_played + 1

        # Use provided key instead of state.rng to prevents reusing the same key pattern
        key, *reel_keys = jax.random.split(key, cfg.num_reels + 1)

        # Generate random spin durations with sequential stopping. Each reel stops progressively later for a
        # dramatic effect
        spin_durations = jnp.array([
            jax.random.randint(
                reel_key, (),
                cfg.min_spin_duration + i * cfg.reel_stop_delay,
                cfg.max_spin_duration + i * cfg.reel_stop_delay
            )
            for i, reel_key in enumerate(reel_keys)
        ])

        # Generate random final positions using probability weights
        final_positions = jnp.array([
            jax.random.choice(
                reel_key,
                jnp.arange(cfg.total_symbols_per_reel),
                p=self._get_symbol_probabilities()
            )
            for reel_key in reel_keys
        ])

        return state._replace(
            # Economic state
            credits=new_credits,                                     # Wager deducted
            spins_played=new_spins,                                  # Increment spin counter

            # Reel state - everything starts spinning
            reel_spinning=jnp.ones(cfg.num_reels, dtype=jnp.bool_), # All reels spinning
            spin_timers=spin_durations,                              # Countdown timers
            reel_positions=final_positions,                          # Final outcomes (hidden)

            # Control state
            spin_cooldown=jnp.array(10, dtype=jnp.int32),           # Brief cooldown

            # Reset displays to build suspense
            last_payout=jnp.array(0, dtype=jnp.int32),              # Clear old payout
            last_reward=jnp.array(0.0, dtype=jnp.float32),          # Clear old reward
        )

    def _get_symbol_probabilities(self) -> jnp.ndarray:
        """
        Get normalized probabilities for symbol selection. Creates a probability distribution that makes rare symbols
        actually rare. This is what gives the slot machine its house edge and authentic feel.

        Probability Math:
        - Each symbol type appears multiple times in the 20-position reel
        - Symbol weights determine relative frequency
        - For example TV is 3x (weight 3)  more likely than car (weight 1)
        - Creates realistic slot machine odds

        """
        cfg = self.config

        # Create probability array for all 20 positions on the reel
        probs = jnp.zeros(cfg.total_symbols_per_reel)

        # Assign weights to each symbol type
        for symbol_type in range(cfg.num_symbol_types):
            # Each symbol type appears at regular intervals on the reel
            # e.g., Cactus at positions 0, 6, 12, 18 for 6 symbol types
            symbol_positions = jnp.arange(symbol_type, cfg.total_symbols_per_reel, cfg.num_symbol_types)
            probs = probs.at[symbol_positions].set(cfg.symbol_weights[symbol_type])

        # Normalize to create valid probability distribution
        return probs / jnp.sum(probs)

    def _update_reels(self, state: SlotMachineState) -> SlotMachineState:
        """
        Handles the visual animation of spinning reels and the timing of when
        they stop.

        Animation System:
        - Timers count down each frame
        - When timer hits 0, reel stops spinning
        - Visual position updates create spinning effect
        - Final position is revealed when reel stops
        """
        cfg = self.config

        # Decrement spin timers (countdown to reel stop)
        new_timers = jnp.maximum(state.spin_timers - 1, 0)

        # Stop reels when their timer reaches zero
        new_spinning = state.reel_spinning & (new_timers > 0)

        # Animate spinning reels visual effect. This is what creates the illusion of spinning
        spin_speed = 2
        animated_positions = jnp.where(
            state.reel_spinning,
            (state.reel_positions + spin_speed) % cfg.total_symbols_per_reel,
            state.reel_positions
        )

        return state._replace(
            spin_timers=new_timers,
            reel_spinning=new_spinning,
            reel_positions=animated_positions,
        )

    def _check_for_wins(self, state: SlotMachineState) -> SlotMachineState:
        """
        Check for winning combinations after all reels have stopped.

        The payline (from left to right) pays as follows:

        - Cactus on reel 1 only -> 2x wager
        - Cactus on reels 1 & 2 -> 5x wager
        - Table, Table, Bar     -> 10x wager
        - Table, Table, Table   -> 10x wager
        - TV, TV, Bar           -> 14x wager
        - TV, TV, TV            -> 14x wager
        - Bell, Bell, Bar       -> 18x wager
        - Bell, Bell, Bell      -> 18x wager
        - Bar, Bar, Bar         -> 100x wager
        - Car, Car, Car         -> 200x wager

        """
        cfg = self.config

        # Only check for wins when all reels have stopped, and we haven't already processed a win
        # And we have actually played at least one spin to prevent initial state reward
        all_stopped = ~jnp.any(state.reel_spinning)
        has_spun = state.spins_played > 0
        should_check_win = all_stopped & (state.last_payout == 0) & has_spun

        def _process_win(s: SlotMachineState) -> SlotMachineState:
            """
            Helper function to process win calculation with reward system.
            """

            # Get the center symbol from each reel (the payline)
            center_symbols = jnp.array([
                (s.reel_positions[i] + 1) % cfg.total_symbols_per_reel % cfg.num_symbol_types
                for i in range(cfg.num_reels)
            ])

            reel0, reel1, reel2 = center_symbols

            payout = jnp.select([
                jnp.all(center_symbols == 5),
                jnp.all(center_symbols == 2),
                jnp.all(center_symbols == 4),
                (reel0 == 4) & (reel1 == 4) & (reel2 == 2),
                jnp.all(center_symbols == 3),
                (reel0 == 3) & (reel1 == 3) & (reel2 == 2),
                jnp.all(center_symbols == 1),
                (reel0 == 1) & (reel1 == 1) & (reel2 == 2),
                (reel0 == 0) & (reel1 == 0) & (reel2 != 0),
                (reel0 == 0) & (reel1 != 0) & (reel2 != 0),

            ], [
                200 * s.current_wager,  # Three Cars - Ultimate Jackpot
                100 * s.current_wager,  # Three Bars - Special Jackpot
                18 * s.current_wager,   # Three Bells
                18 * s.current_wager,   # Two Bells + One Bar
                14 * s.current_wager,   # Three TVs
                14 * s.current_wager,   # Two TVs + One Bar
                10 * s.current_wager,   # Three Tables
                10 * s.current_wager,   # Two Tables + One Bar
                5 * s.current_wager,    # Cactus on reels 1 & 2
                2 * s.current_wager,    # One Cactus on first reel only
            ],
            default=0)  # No payout for other combinations

            """
            Reward System: (More info under https://www.atarimania.com/game-atari-2600-vcs-slot-machine_8212.html)
            """

            # Update game state with winnings
            new_credits = s.credits + payout
            new_winnings = s.total_winnings + payout
            new_flash_timer = jnp.where(payout > 0, 60, 0)  # 1 second flash at 60fps

            return s._replace(
                credits=new_credits,
                total_winnings=new_winnings,
                last_payout=payout,
                win_flash_timer=new_flash_timer,
            )

        # Process wins when appropriate
        return jax.lax.cond(
            should_check_win,
            _process_win,
            lambda s: s,
            state
        )

    def _update_timers(self, state: SlotMachineState) -> SlotMachineState:
        """
        Update various timers and cooldowns. Manages all the timing-based effects in the game.

        Timer Types:
        - Spin cooldown: Prevents accidental double-spins
        - Win flash: Controls celebration effect duration

        All timers count down to 0 and stop there (no negative values).

        """
        new_cooldown = jnp.maximum(state.spin_cooldown - 1, 0)
        new_flash_timer = jnp.maximum(state.win_flash_timer - 1, 0)

        return state._replace(
            spin_cooldown=new_cooldown,
            win_flash_timer=new_flash_timer,
        )

    def _get_observation(self, state: SlotMachineState) -> SlotMachineObservation:
        """
        Extract observation from game state.
        """

        cfg = self.config

        # Get currently visible symbols for each reel
        reel_symbols = jnp.zeros((cfg.num_reels, cfg.symbols_per_reel), dtype=jnp.int32)

        # Build the symbol grid that matches what's visually displayed
        for reel_idx in range(cfg.num_reels):
            for symbol_slot in range(cfg.symbols_per_reel):
                # Calculate which symbol appears at this visual position
                symbol_index = (state.reel_positions[reel_idx] + symbol_slot) % cfg.total_symbols_per_reel
                symbol_type = symbol_index % cfg.num_symbol_types
                reel_symbols = reel_symbols.at[reel_idx, symbol_slot].set(symbol_type)

        return SlotMachineObservation(
            credits=state.credits,
            current_wager=state.current_wager,
            reel_symbols=reel_symbols,
            is_spinning=jnp.array(jnp.any(state.reel_spinning), dtype=jnp.int32),
            last_payout=state.last_payout,
            last_reward=state.last_reward,
        )

    def render(self, state: SlotMachineState) -> chex.Array:
        """ Render the current game state. Simple wrapper around the renderer. Initial idea was to keep this separate
        to allow for easy swapping of rendering backends.
        """
        return self.renderer.render(state)

    def action_space(self) -> spaces.Discrete:
        """
        Get the action space for this environment.

        Action Mapping:
        0 = NOOP (wait/observe)
        1 = FIRE (spin reels)
        2 = UP (increase wager)
        3 = DOWN (decrease wager)
        """
        return spaces.Discrete(len(self.action_set))

    def observation_space(self) -> spaces.Space:
        """
        Get the observation space for this environment. Defines the structure and bounds of observations.

        Space Design:
        - Credits: 0-9999

        - Wager: min_wager to max_wager for configurable betting range. (0-5)

        - Reel symbols: 0 to num_symbol_types-1.  Symbol type indices currently
         ( 0 => "Cactus", 1 => "Table", 2 => "Bar", 3 => "TV", 4 => "Bell", 5 => "Car")

        - Spinning: boolean (machine status)

        - Payouts/rewards (Integer to keep track of payouts. Important for debugging.)
        """
        cfg = self.config
        return spaces.Dict({
            'credits': spaces.Box(low=0, high=9999, shape=(), dtype=jnp.int32),
            'current_wager': spaces.Box(low=cfg.min_wager, high=cfg.max_wager, shape=(), dtype=jnp.int32),
            'reel_symbols': spaces.Box(
                low=0,
                high=cfg.num_symbol_types - 1,
                shape=(cfg.num_reels, cfg.symbols_per_reel),
                dtype=jnp.int32
            ),
            'is_spinning': spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
            'last_payout': spaces.Box(low=0, high=1000, shape=(), dtype=jnp.int32),
            'last_reward': spaces.Box(low=0.0, high=1000.0, shape=(), dtype=jnp.float32),
        })

    def image_space(self) -> spaces.Space:
        """Image space describing rendered RGB frames."""
        cfg = self.config
        return spaces.Box(
            low=0,
            high=255,
            shape=(cfg.screen_height, cfg.screen_width, 3),
            dtype=jnp.uint8,
        )

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: SlotMachineObservation) -> jnp.ndarray:
        """Flatten the structured observation into a 1-D array."""
        components = [
            jnp.atleast_1d(obs.credits).astype(jnp.float32),
            jnp.atleast_1d(obs.current_wager).astype(jnp.float32),
            obs.reel_symbols.astype(jnp.float32).ravel(),
            jnp.atleast_1d(obs.is_spinning).astype(jnp.float32),
            jnp.atleast_1d(obs.last_payout).astype(jnp.float32),
            jnp.atleast_1d(obs.last_reward).astype(jnp.float32),
        ]
        return jnp.concatenate(components, axis=0)

    def _get_info(self, state: SlotMachineState) -> SlotMachineInfo:

        return SlotMachineInfo(
            total_winnings=state.total_winnings,
            spins_played=state.spins_played,
        )

    def _get_reward(self, previous_state: SlotMachineState, state: SlotMachineState) -> float:
        """Return the associated reward. """
        return float(jnp.asarray(state.last_reward))

    def _get_done(self, state: SlotMachineState) -> bool:
        """Check if the player can no longer place the minimum wager, or reached the max credits"""
        cfg = self.config
        credits = int(jnp.asarray(state.credits))
        spinning = bool(jnp.asarray(jnp.any(state.reel_spinning)))
        max_credits_reached = credits >= 999
        return ((credits < cfg.min_wager) and (not spinning)) or max_credits_reached


def main():
    """
    Simple test function to run the slot machine game with proper randomness.

    This is a basic demo. For actual gameplay, use:
    python scripts/play.py --game slotmachine
    """
    import pygame
    import time
    import numpy as np

    pygame.init()

    config = SlotMachineConfig()
    game = JaxSlotMachine(config)
    symbol_names = SlotMachineConstants().symbol_names

    jitted_step = jax.jit(game.step)
    jitted_reset = jax.jit(game.reset)

    random_seed = int(time.time() * 1000000) % (2**31)
    key = jax.random.PRNGKey(random_seed)
    obs, state = jitted_reset(key)

    screen_size = (
        config.screen_width * config.scaling_factor,
        config.screen_height * config.scaling_factor
    )
    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption("JAX Slot Machine")
    clock = pygame.time.Clock()

    print(" JAX SLOT MACHINE")
    print(f"Random seed: {random_seed}")
    print("Controls: SPACE=Spin, UP/DOWN=Wager, ESC=Quit")
    print("For full gameplay, use: python scripts/play.py --game slotmachine")

    # Game loop
    running = True
    action = Action.NOOP

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    action = Action.FIRE
                elif event.key == pygame.K_UP:
                    action = Action.UP
                elif event.key == pygame.K_DOWN:
                    action = Action.DOWN
                else:
                    action = Action.NOOP
            elif event.type == pygame.KEYUP:
                action = Action.NOOP

        obs, state, reward, done, info = jitted_step(state, action)

        reward_value = float(np.array(reward))

        if reward_value > 0:
            center_symbols = []
            center_positions = []
            reel_positions = np.array(state.reel_positions)
            for reel_idx in range(config.num_reels):
                pos = int(reel_positions[reel_idx])
                center_positions.append(pos)
                symbol_index = (pos + 1) % config.total_symbols_per_reel
                symbol_type = symbol_index % config.num_symbol_types
                center_symbols.append(symbol_names[symbol_type])


            # Debug statement to see if the reward mechanism is working. The debug works only when ran  directly
            # via python and not play.py


            # print(
            #     "[DEBUG] Reward: {:.0f} | Center symbols: {} | Reel positions: {}".format(
            #         reward_value,
            #         ", ".join(center_symbols),
            #         ", ".join(str(p) for p in center_positions),
            #     )
            # )

        if done:
            print("You're out of credits. Thanks for playing!")
            running = False

        if not running:
            break

        try:
            frame = game.render(state)
            if frame.shape[-1] == 3:  # RGB format
                frame_np = jnp.array(frame, dtype=jnp.uint8)
                scaled_frame = jnp.repeat(
                    jnp.repeat(frame_np, config.scaling_factor, axis=0),
                    config.scaling_factor, axis=1
                )
                surf = pygame.surfarray.make_surface(scaled_frame.swapaxes(0, 1))
                screen.blit(surf, (0, 0))
        except Exception as e:
            print(f"Rendering error: {e}")

        # Update display
        pygame.display.flip()
        clock.tick(60)

        action = Action.NOOP

    pygame.quit()
    print("Game ended. Use scripts/play.py for full gameplay!")


if __name__ == "__main__":
    main()
