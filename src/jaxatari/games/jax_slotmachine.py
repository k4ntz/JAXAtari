"""
JAX Slot Machine - A two-player competitive slot machine implementation for JAXAtari.

This module implements a traditional three-reel slot machine Atari 2600 game using JAX for GPU acceleration
and JIT compilation. The game features authentic slot machine mechanics with spinning reels,
various symbols, and a payout system. Two players compete against each other until one of them goes broke.

Author: Ashish Bhandari, ashish.bhandari@stud.tu-darmstadt.de, https://github.com/zatakashish

License: TU Darmstadt, All rights reserved.

========================================================================================================================
                                                GAMEPLAY MECHANICS
========================================================================================================================

Two players compete with separate credit pools and wagers displayed on opposite sides of the screen until one of them
runs out of credit.

- Player 1: Credits and wager displayed on the right side (above and below reel 3)
The player 1 is the main human player.

- Player 2: Credits and wager displayed on the left side (above and below reel 1)
The player 2 is the Computer or the second human player.

How to play ? (Run command: python scripts/play.py --game slotmachine)
- SPACE to spin the reels (both players participate)
- Player 1: ARROWDOWN / ARROWUP to decrease/increase wager
- Player 2: ARROWRIGHT /ARROWLEFT to decrease/increase wager

Game addons:

- UPLEFT arrow for jackpot mode. Can be enabled/disabled before the game is started.
 (HINT: UPLEFT Arrow may result in wager being increased, please check wager before hitting spin. Currently not so many
 options within ALE's action space )

Game End Conditions:
- Either player reaches 0 credits (other player wins)
- Either player reaches more than 999 credits (that player wins)

Reward System: (More information under https://www.atarimania.com/game-atari-2600-vcs-slot-machine_8212.html)

The payline (from left to right) pays as follows for both players in normal mode (payout):

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

The payline (from left to right) pays as follows for both players jackpot mode:

- Any three reels       -> 20x wager
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
from jaxatari.rendering import jax_rendering_utils_legacy as aj
import jaxatari.spaces as spaces


SLOT_MACHINE_SYMBOL_NAMES: Tuple[str, ...] = (
    "Cactus",
    "Table",
    "Bar",
    "TV",
    "Bell",
    "Car",
    "Empty",
)


def _get_default_asset_config() -> tuple:
    """
    Returns the default declarative asset manifest for Slot Machine.
    Kept immutable (tuple of dicts) to fit NamedTuple defaults.
    """
    assets = []
    for idx, symbol_name in enumerate(SLOT_MACHINE_SYMBOL_NAMES):
        assets.append(
            {
                "name": f"symbol_{idx}",
                "type": "single",
                "file": f"{symbol_name}.npy",
            }
        )
    return tuple(assets)


@dataclass(frozen=True)
class SlotMachineConstants:
    """
    Unified constants for Slot Machine game parameters and assets.
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
    
    # Reel timing
    min_spin_duration: int = 60
    max_spin_duration: int = 120
    reel_stop_delay: int = 30
    
    reel_layouts: jnp.ndarray = field(
        default_factory=lambda: jnp.array([
            [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1],
            [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1],
            [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1],
        ], dtype=jnp.int32)
    )
    
    reel_layouts_jackpot: jnp.ndarray = field(
        default_factory=lambda: jnp.array([
            [0, 1, 2, 3, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            [0, 1, 2, 3, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            [0, 1, 2, 3, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
        ], dtype=jnp.int32)
    )
    
    # Reel positions - UI layout coordinates
    reel_start_x: int = 11
    reel_start_y: int = 50
    
    # Asset config baked into constants
    ASSET_CONFIG: tuple = field(default_factory=lambda: tuple([
        {
            'name': 'symbols',
            'type': 'group',
            'files': [
                # Explicit paths relative to sprite_dir -> sprites/slotmachine/
                'Cactus.npy',
                'Table.npy',
                'Bar.npy',
                'TV.npy',
                'Bell.npy',
                'Car.npy',
                'Empty.npy',
            ]
        },
        # Background and reel frame are added procedurally in the renderer init
    ]))
    sprite_scale_factor: float = 0.7


class SlotMachineState(NamedTuple):
    """
    Complete immutable game state for Slot Machine.
    """

    # Core game progression state - Player 1
    player1_credits: chex.Array
    player1_total_winnings: chex.Array
    player1_spins_played: chex.Array
    player1_wager: chex.Array

    # Core game progression state - Player 2
    player2_credits: chex.Array
    player2_total_winnings: chex.Array
    player2_spins_played: chex.Array
    player2_wager: chex.Array

    # Reel mechanics
    reel_positions: chex.Array
    reel_spinning: chex.Array
    spin_timers: chex.Array
    reel_speeds: chex.Array
    reel_layouts: chex.Array

    # Input handling with a button debouncing feature so that we ignore repeated presses
    spin_button_prev: chex.Array
    up_button_prev: chex.Array
    down_button_prev: chex.Array
    left_button_prev: chex.Array
    right_button_prev: chex.Array
    jackpot_button_prev: chex.Array  # For jackpot mode toggle
    spin_cooldown: chex.Array

    # Jackpot mode
    jackpot_mode: chex.Array
    jackpot_message_timer: chex.Array

    # Visual effects
    win_flash_timer: chex.Array
    last_payout_p1: chex.Array
    last_payout_p2: chex.Array
    last_reward_p1: chex.Array
    last_reward_p2: chex.Array

    # Game end state
    game_over: chex.Array
    winner: chex.Array  # 0 = no winner, 1 = player1, 2 = player2
    win_display_timer: chex.Array

    # RNG key that keeps the spins honest
    rng: chex.Array


class SlotMachineObservation(NamedTuple):
    """Observation returned to the agent each step."""

    player1_credits: jnp.ndarray          # Player 1 current credits
    player1_wager: jnp.ndarray           # Player 1 current bet amount
    player2_credits: jnp.ndarray          # Player 2 current credits
    player2_wager: jnp.ndarray           # Player 2 current bet amount
    reel_symbols: jnp.ndarray     # Visible symbols (shape: num_reels × symbols_per_reel)
    is_spinning: jnp.ndarray      # Is the machine currently spinning?
    last_payout_p1: jnp.ndarray      # Last win amount for player 1
    last_payout_p2: jnp.ndarray      # Last win amount for player 2
    last_reward_p1: jnp.ndarray      # Last reward for player 1
    last_reward_p2: jnp.ndarray      # Last reward for player 2
    jackpot_mode: jnp.ndarray     # Is jackpot mode enabled?
    game_over: jnp.ndarray        # Is the game over?
    winner: jnp.ndarray           # Who won (0=none, 1=player1, 2=player2)


class SlotMachineInfo(NamedTuple):
    """
    Information about the game state.
    """

    # Saves total winnings for both players
    player1_total_winnings: jnp.ndarray
    player2_total_winnings: jnp.ndarray

    # Total number of spins altogether, not currently needed. Introduced for debug and win statistics.
    player1_spins_played: jnp.ndarray
    player2_spins_played: jnp.ndarray


class SlotMachineConstantsRuntime(NamedTuple):
    """
    Symbol names and any runtime constants that are convenience only.
    """
    symbol_names: tuple = SLOT_MACHINE_SYMBOL_NAMES


from jaxatari.rendering import jax_rendering_utils as render_utils


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

    def __init__(self, consts: SlotMachineConstants = None):
    
        super().__init__()
        self.consts = consts or SlotMachineConstants()
        self.runtime = SlotMachineConstantsRuntime()
        self.sprite_dir = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/slotmachine"
        # Configure new render utils
        self.ru_config = render_utils.RendererConfig(
            game_dimensions=(self.consts.screen_height, self.consts.screen_width),
            channels=3,
        )
        self.jr = render_utils.JaxRenderingUtils(self.ru_config)
        # Build asset config from constants and inject pre-scaled symbol data (slightly hacky to keep compatibility with modification pipeline)
        final_asset_config = [dict(item) for item in self.consts.ASSET_CONFIG]
        for item in final_asset_config:
            if item.get('name') == 'symbols':
                item.pop('files', None)
                item['data'] = self._load_scaled_symbols_rgba(self.consts.sprite_scale_factor)
        final_asset_config.append({'name': 'background', 'type': 'background', 'data': self._create_background_rgba()})
        final_asset_config.append({'name': 'reel_frame', 'type': 'procedural', 'data': self._create_reel_frame_rgba()})
        # Load assets
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(final_asset_config, self.sprite_dir)

    def _load_scaled_symbols_rgba(self, scale_factor: float) -> Any:
        """Load symbol RGBA sprites from disk and downscale them by scale_factor, keep dtype uint8."""
        import numpy as np
        import jax
        from jax import image as jimage
        scaled_list = []
        for name in SLOT_MACHINE_SYMBOL_NAMES:
            npy_path = os.path.join(self.sprite_dir, f"{name}.npy")
            rgba_np = np.load(npy_path)
            sprite = jnp.array(rgba_np, dtype=jnp.uint8)
            h, w = sprite.shape[:2]
            new_h = jnp.maximum(1, jnp.round(h * scale_factor)).astype(jnp.int32)
            new_w = jnp.maximum(1, jnp.round(w * scale_factor)).astype(jnp.int32)
            # jax.image.resize expects shape as (H, W, C)
            resized = jax.image.resize(sprite, (new_h, new_w, sprite.shape[2]), method='nearest').astype(jnp.uint8)
            scaled_list.append(resized)
        # Return as a stack for efficiency; loader accepts list or stack
        return jnp.stack(scaled_list)

    def _create_background_rgba(self) -> jnp.ndarray:

        """
         Create classic atari greenish tint with blue frame.

         """

        h, w = self.consts.screen_height, self.consts.screen_width
        bg = jnp.zeros((h, w, 4), dtype=jnp.uint8)

        # Fill entire background with greenish tint
        green_color = jnp.array([140, 208, 140], dtype=jnp.uint8)
        alpha_value = jnp.uint8(255)
        border_color = jnp.array([70, 82, 184], dtype=jnp.uint8)

        bg = bg.at[:, :, :3].set(green_color)
        bg = bg.at[:, :, 3].set(alpha_value)

        # Add borders
        bg = bg.at[:15, :, :3].set(border_color)
        bg = bg.at[-15:, :, :3].set(border_color)
        bg = bg.at[:, :3, :3].set(border_color)
        bg = bg.at[:, -3:, :3].set(border_color)

        return bg

    def _create_reel_frame_rgba(self) -> jnp.ndarray:
        """Create reel frame """
        h, w = self.consts.reel_height, self.consts.reel_width
        frame = jnp.zeros((h, w, 4), dtype=jnp.uint8)

        light_blue = jnp.array([70, 82, 184], dtype=jnp.uint8)
        alpha = jnp.uint8(255)

        frame = frame.at[..., :3].set(light_blue)
        frame = frame.at[..., 3].set(alpha)
        return frame

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
        cfg = self.consts
    
        # Start with palette-id raster background
        raster = self.jr.create_object_raster(self.BACKGROUND)
    
        # Render each reel
        for reel_idx in range(cfg.num_reels):
            reel_x = cfg.reel_start_x + reel_idx * (cfg.reel_width + cfg.reel_spacing)
            reel_y = cfg.reel_start_y
    
            # Render reel frame first
            frame_mask = self.SHAPE_MASKS['reel_frame']
            raster = self.jr.render_at(raster, reel_x, reel_y, frame_mask, flip_offset=self.FLIP_OFFSETS['reel_frame'])
    
            # Get reel state for this specific reel
            reel_pos = state.reel_positions[reel_idx]
    
            # Render the 3 visible symbols in this reel
            for symbol_slot in range(cfg.symbols_per_reel):
                symbol_slot_y = reel_y + symbol_slot * 40
    
                # Calculate which symbol to show at this position
                symbol_index = (reel_pos + symbol_slot) % cfg.total_symbols_per_reel
                symbol_type = state.reel_layouts[reel_idx, symbol_index]
    
                # Select mask from grouped symbols
                symbol_mask_stack = self.SHAPE_MASKS['symbols']
                symbol_mask = symbol_mask_stack[symbol_type]
    
                # Center the 28x28 (padded) sprite within the 40x40 slot
                centered_x = reel_x + 6
                centered_y = symbol_slot_y + 6
    
                raster = self.jr.render_at(raster, centered_x, centered_y, symbol_mask, flip_offset=self.FLIP_OFFSETS['symbols'])
    
        # Render UI elements (boxes and numbers) directly onto the raster
        raster = self._render_credits_display_ids(raster, state)
    
        # No per-pixel flash multiplier here; could recolor via palette if needed
    
        # Convert palette-id raster to RGB
        return self.jr.render_from_palette(raster, self.PALETTE)

    def _render_winner_message(self, raster: jnp.ndarray, winner: chex.Array) -> jnp.ndarray:
        """Render winner message in the center of the screen."""
        cfg = self.consts
        
        # Message position (center of screen)
        msg_x = cfg.screen_width // 2 - 50  # Increased space for longer messages
        msg_y = cfg.screen_height // 2 + 1

        # Background box for message
        box_color = jnp.array([70, 82, 184], dtype=jnp.uint8)
        text_color = jnp.array([255, 255, 255], dtype=jnp.uint8)

        # Draw background box (made wider for "GAME OVER")
        raster = self._draw_colored_box(raster, msg_x - 15, msg_y - 5, 130, 20, box_color)

        # Handle different winner cases
        def render_player_wins(r, player_num):
            r = self._render_simple_text(r, "PLAYER ", msg_x, msg_y, text_color)
            r = self._render_colored_digit(r, player_num, msg_x + 42, msg_y, text_color)
            r = self._render_simple_text(r, " WINS", msg_x + 52, msg_y, text_color)
            return r
        
        def render_game_over(r):
            r = self._render_simple_text(r, "GAME OVER", msg_x + 35, msg_y, text_color)
            return r

        # Conditional rendering based on winner value
        raster = jax.lax.cond(
            winner == 1,
            lambda r: render_player_wins(r, 1),
            lambda r: jax.lax.cond(
                winner == 2,
                lambda r: render_player_wins(r, 2),
                lambda r: render_game_over(r),  # For winner == 0 or any other case
                r
            ),
            raster
        )

        return raster

    def _render_simple_text(self, raster: jnp.ndarray, text: str, x: int, y: int, color: jnp.ndarray) -> jnp.ndarray:
        """Render simple text using a basic bitmap font."""
        # Simple character patterns for basic text rendering
        char_patterns = {
            'P': [[1,1,1,1], [1,0,0,1], [1,1,1,1], [1,0,0,0], [1,0,0,0], [1,0,0,0]],
            'L': [[1,0,0,0], [1,0,0,0], [1,0,0,0], [1,0,0,0], [1,0,0,0], [1,1,1,1]],
            'A': [[0,1,1,0], [1,0,0,1], [1,0,0,1], [1,1,1,1], [1,0,0,1], [1,0,0,1]],
            'Y': [[1,0,0,1], [1,0,0,1], [0,1,1,0], [0,1,1,0], [0,1,1,0], [0,1,1,0]],
            'E': [[1,1,1,1], [1,0,0,0], [1,1,1,0], [1,0,0,0], [1,0,0,0], [1,1,1,1]],
            'R': [[1,1,1,0], [1,0,0,1], [1,1,1,0], [1,1,0,0], [1,0,1,0], [1,0,0,1]],
            'W': [[1,0,0,1], [1,0,0,1], [1,0,0,1], [1,0,1,1], [1,1,0,1], [1,0,0,1]],
            'I': [[1,1,1], [0,1,0], [0,1,0], [0,1,0], [0,1,0], [1,1,1]],
            'N': [[1,0,0,1], [1,1,0,1], [1,0,1,1], [1,0,0,1], [1,0,0,1], [1,0,0,1]],
            'S': [[0,1,1,1], [1,0,0,0], [0,1,1,0], [0,0,0,1], [0,0,0,1], [1,1,1,0]],
            'O': [[0,1,1,0], [1,0,0,1], [1,0,0,1], [1,0,0,1], [1,0,0,1], [0,1,1,0]],
            'T': [[1,1,1,1,1], [0,0,1,0,0], [0,0,1,0,0], [0,0,1,0,0], [0,0,1,0,0], [0,0,1,0,0]],
            'C': [[0,1,1,1], [1,0,0,0], [1,0,0,0], [1,0,0,0], [1,0,0,0], [0,1,1,1]],
            'K': [[1,0,0,1], [1,0,1,0], [1,1,0,0], [1,1,0,0], [1,0,1,0], [1,0,0,1]],
            'J': [[0,0,0,1], [0,0,0,1], [0,0,0,1], [0,0,0,1], [1,0,0,1], [0,1,1,0]],
            'M': [[1,0,0,0,1], [1,1,0,1,1], [1,0,1,0,1], [1,0,0,0,1], [1,0,0,0,1], [1,0,0,0,1]],
            'D': [[1,1,1,0], [1,0,0,1], [1,0,0,1], [1,0,0,1], [1,0,0,1], [1,1,1,0]],
            'F': [[1,1,1,1], [1,0,0,0], [1,1,1,0], [1,0,0,0], [1,0,0,0], [1,0,0,0]],
            'G': [[0,1,1,1], [1,0,0,0], [1,0,1,1], [1,0,0,1], [1,0,0,1], [0,1,1,0]],
            'V': [[1,0,0,0,1], [1,0,0,0,1], [1,0,0,0,1], [0,1,0,1,0], [0,1,0,1,0], [0,0,1,0,0]],
            ' ': [[0,0], [0,0], [0,0], [0,0], [0,0], [0,0]]  # Space character
        }
        
        current_x = x
        for char in text.upper():
            if char in char_patterns:
                pattern = char_patterns[char]
                char_width = len(pattern[0])
                char_height = len(pattern)
                
                # Render this character
                for row in range(char_height):
                    for col in range(char_width):
                        if pattern[row][col] == 1:
                            pixel_x = current_x + col
                            pixel_y = y + row
                            if 0 <= pixel_x < raster.shape[1] and 0 <= pixel_y < raster.shape[0]:
                                raster = raster.at[pixel_y, pixel_x, :].set(color)
                
                current_x += char_width + 1  # Add 1 pixel spacing between characters
        
        return raster

    def _render_credits_display_ids(self, raster: jnp.ndarray, state: SlotMachineState) -> jnp.ndarray:
        """Draw the credits and wager boxes for both players using ID raster operations."""
        return self._render_text_labels_ids(raster, state)

    def _render_text_labels_ids(self, raster: jnp.ndarray, state: SlotMachineState) -> jnp.ndarray:
        """Helper function to draw the HUD plates and insert credits and wagers for both players.
        """
        cfg = self.consts

        # Calculate reel positions
        first_reel_x = cfg.reel_start_x
        third_reel_x = cfg.reel_start_x + 2 * (cfg.reel_width + cfg.reel_spacing)
        reel_y = cfg.reel_start_y

        # Colors
        # Pick palette IDs closest to the intended colors
        box_rgb = (70, 82, 184)
        number_rgb = (194, 67, 115)
        box_id = self.COLOR_TO_ID.get(box_rgb, 0)
        num_id = self.COLOR_TO_ID.get(number_rgb, 0)

        # Player 1 (right side)
        # UI for player 1 credits
        credits_y = reel_y - 22
        raster = self._draw_colored_box_ids(raster, third_reel_x - 3, credits_y - 2, 47, 16, box_id)
        credits_digits = self.jr.int_to_digits(state.player1_credits, max_digits=4)
        raster = self._render_digits_ids(raster, credits_digits, third_reel_x + 2, credits_y + 2, num_id)

        # UI for player 1 wager
        wager_y = reel_y + cfg.reel_height + 10
        raster = self._draw_colored_box_ids(raster, third_reel_x + 7, wager_y - 4, 27, 16, box_id)
        wager_digits = self.jr.int_to_digits(state.player1_wager, max_digits=2)
        raster = self._render_digits_ids(raster, wager_digits, third_reel_x + 12, wager_y, num_id)

        # Player 2 (left side)
        # UI for player 2 credits (above first reel)
        raster = self._draw_colored_box_ids(raster, first_reel_x - 3, credits_y - 2, 47, 16, box_id)
        credits_digits_p2 = self.jr.int_to_digits(state.player2_credits, max_digits=4)
        raster = self._render_digits_ids(raster, credits_digits_p2, first_reel_x + 2, credits_y + 2, num_id)

        # UI for player 2 wager (below first reel)
        raster = self._draw_colored_box_ids(raster, first_reel_x + 7, wager_y - 4, 27, 16, box_id)
        wager_digits_p2 = self.jr.int_to_digits(state.player2_wager, max_digits=2)
        raster = self._render_digits_ids(raster, wager_digits_p2, first_reel_x + 12, wager_y, num_id)

        # Render jackpot mode message if timer is active
        raster = jax.lax.cond(
            state.jackpot_message_timer > 0,
            lambda r: self._render_jackpot_message_ids(r, state.jackpot_mode, box_id),
            lambda r: r,
            raster
        )

        # Check if game is over, reels have stopped, and display winner message
        raster = jax.lax.cond(
            state.game_over & (state.win_display_timer > 0) & (~jnp.any(state.reel_spinning)),
            lambda r: self._render_winner_message_ids(r, state.winner, box_id),
            lambda r: r,
            raster
        )

        return raster

    def _render_jackpot_message_ids(self, raster: jnp.ndarray, jackpot_mode: chex.Array, box_id: int) -> jnp.ndarray:
        """Render jackpot mode toggle message using ID raster ops."""
        cfg = self.consts
        msg_x = cfg.screen_width // 2 - 59
        msg_y = cfg.screen_height // 2 + 1
        raster = self._draw_colored_box_ids(raster, msg_x - 10, msg_y - 5, 140, 20, box_id)
        white_id = self.COLOR_TO_ID.get((255, 255, 255), 0)
        def render_on(r):
            return self._render_simple_text_ids(r, "JACKPOT MODE ON", msg_x, msg_y, white_id)
        def render_off(r):
            return self._render_simple_text_ids(r, "JACKPOT MODE OFF", msg_x, msg_y, white_id)
        return jax.lax.cond(jackpot_mode, render_on, render_off, raster)

    def _render_winner_message_ids(self, raster: jnp.ndarray, winner: chex.Array, box_id: int) -> jnp.ndarray:
        """Render winner message in the center of the screen using ID raster ops."""
        cfg = self.consts
        msg_x = cfg.screen_width // 2 - 50
        msg_y = cfg.screen_height // 2 + 1
        raster = self._draw_colored_box_ids(raster, msg_x - 15, msg_y - 5, 130, 20, box_id)
        white_id = self.COLOR_TO_ID.get((255, 255, 255), 0)
        def render_player(r, num):
            r = self._render_simple_text_ids(r, "PLAYER ", msg_x, msg_y, white_id)
            r = self._render_simple_text_ids(r, "1" if num == 1 else "2", msg_x + 42, msg_y, white_id)
            r = self._render_simple_text_ids(r, " WINS", msg_x + 52, msg_y, white_id)
            return r
        def render_over(r):
            return self._render_simple_text_ids(r, "GAME OVER", msg_x + 35, msg_y, white_id)
        return jax.lax.cond(
            winner == 1,
            lambda r: render_player(r, 1),
            lambda r: jax.lax.cond(winner == 2, lambda r2: render_player(r2, 2), render_over, r),
            raster
        )

    def _render_jackpot_message(self, raster: jnp.ndarray, jackpot_mode: chex.Array) -> jnp.ndarray:
        """Render jackpot mode toggle message."""
        cfg = self.consts
        
        # Message position (center of screen)
        msg_x = cfg.screen_width // 2 - 59
        msg_y = cfg.screen_height // 2 + 1
        
        # Background box for message
        box_color = jnp.array([70, 82, 184], dtype=jnp.uint8)
        text_color = jnp.array([255, 255, 255], dtype=jnp.uint8)
        
        # Draw background box
        raster = self._draw_colored_box(raster, msg_x - 10, msg_y - 5, 140, 20, box_color)
        
        # Render message based on jackpot mode state
        def render_jackpot_on(r):
            return self._render_simple_text(r, "JACKPOT MODE ON", msg_x, msg_y, text_color)
            
        def render_jackpot_off(r):
            return self._render_simple_text(r, "JACKPOT MODE OFF", msg_x, msg_y, text_color)
        
        raster = jax.lax.cond(
            jackpot_mode,
            render_jackpot_on,
            render_jackpot_off,
            raster
        )
        
        return raster

    def _render_digits_ids(self, raster: jnp.ndarray, digits: jnp.ndarray, x: int, y: int, color_id: int) -> jnp.ndarray:
        spacing = 10
    
        def body(i, r_in):
            digit_idx = digits[i]
            return jax.lax.cond(
                digit_idx >= 0,
                lambda r: self._render_digit_bitmap_id(r, digit_idx, x + i * spacing, y, color_id),
                lambda r: r,
                r_in
            )
        return jax.lax.fori_loop(0, digits.shape[0], body, raster)

    def _render_digit_bitmap_id(self, raster: jnp.ndarray, digit: int, x: int, y: int, color_id: int) -> jnp.ndarray:

        # Pre-define digit patterns
        digit_patterns = jnp.array([
            [[0, 1, 1, 1, 1, 0], [1, 1, 0, 0, 1, 1], [1, 1, 0, 0, 1, 1], [1, 1, 0, 0, 1, 1], [1, 1, 0, 0, 1, 1],
             [1, 1, 0, 0, 1, 1], [1, 1, 0, 0, 1, 1], [0, 1, 1, 1, 1, 0]],
            [[0, 0, 1, 1, 0, 0], [0, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 0, 0],
             [0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]],
            [[1, 1, 1, 1, 1, 0], [1, 1, 0, 0, 1, 1], [0, 0, 0, 0, 1, 1], [0, 1, 1, 1, 1, 0], [1, 1, 0, 0, 0, 0],
             [1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]],
            [[1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 1, 1], [0, 1, 1, 1, 1, 0], [0, 0, 0, 0, 1, 1],
             [0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 1, 1], [1, 1, 1, 1, 1, 0]],
            [[1, 1, 0, 0, 1, 1], [1, 1, 0, 0, 1, 1], [1, 1, 0, 0, 1, 1], [1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1],
             [0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 1, 1]],
            [[1, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 1, 1],
             [0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 1, 1], [1, 1, 1, 1, 1, 0]],
            [[0, 1, 1, 1, 1, 0], [1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0], [1, 1, 0, 0, 1, 1],
             [1, 1, 0, 0, 1, 1], [1, 1, 0, 0, 1, 1], [0, 1, 1, 1, 1, 0]],
            [[1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1], [0, 0, 0, 1, 1, 0], [0, 0, 1, 1, 0, 0], [0, 1, 1, 0, 0, 0],
             [0, 1, 1, 0, 0, 0], [0, 1, 1, 0, 0, 0], [0, 1, 1, 0, 0, 0]],
            [[0, 1, 1, 1, 1, 0], [1, 1, 0, 0, 1, 1], [1, 1, 0, 0, 1, 1], [0, 1, 1, 1, 1, 0], [1, 1, 0, 0, 1, 1],
             [1, 1, 0, 0, 1, 1], [1, 1, 0, 0, 1, 1], [0, 1, 1, 1, 1, 0]],
            [[0, 1, 1, 1, 1, 0], [1, 1, 0, 0, 1, 1], [1, 1, 0, 0, 1, 1], [1, 1, 0, 0, 1, 1], [0, 1, 1, 1, 1, 1],
             [0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 1, 1], [0, 1, 1, 1, 1, 0]],
        ], dtype=jnp.int32)

        pattern = digit_patterns[digit]
        digit_h, digit_w = pattern.shape

        # Bounds checking
        valid_x = (x >= 0) & (x + digit_w <= raster.shape[1])
        valid_y = (y >= 0) & (y + digit_h <= raster.shape[0])
        valid_render = valid_x & valid_y

        def render_digit_ids():
            # Build a sprite ID mask: pattern>0 -> color_id, else TRANSPARENT
            sprite_mask = jnp.where(pattern > 0,
                                    jnp.asarray(color_id, dtype=jnp.uint8),
                                    jnp.asarray(self.jr.TRANSPARENT_ID, dtype=jnp.uint8))
            # Stamp via render_utils (handles dynamic positions)
            return self.jr.render_at(raster, x, y, sprite_mask)
    
        return jax.lax.cond(valid_render, render_digit_ids, lambda: raster)

    def _draw_colored_box_ids(self, raster: jnp.ndarray, x: int, y: int, width: int, height: int, color_id: int) -> jnp.ndarray:
        """Fill a rectangle using scaled boolean masks (compatible with downscaling)."""
        # Scale geometric parameters according to renderer config
        scaled_x = jnp.round(x * self.ru_config.width_scaling).astype(jnp.int32)
        scaled_y = jnp.round(y * self.ru_config.height_scaling).astype(jnp.int32)
        scaled_w = jnp.maximum(1, jnp.round(width * self.ru_config.width_scaling)).astype(jnp.int32)
        scaled_h = jnp.maximum(1, jnp.round(height * self.ru_config.height_scaling)).astype(jnp.int32)
        # Build mask over cached grids
        xx, yy = self.jr._xx, self.jr._yy
        mask = (xx >= scaled_x) & (xx < scaled_x + scaled_w) & (yy >= scaled_y) & (yy < scaled_y + scaled_h)
        return jnp.where(mask, jnp.asarray(color_id, raster.dtype), raster)

    def _render_win_flash(self, raster: jnp.ndarray, flash_timer: chex.Array) -> jnp.ndarray:
        """Apply a brief brightness pulse after a win."""
        flash_intensity = jnp.sin(flash_timer * 0.3) * 0.2 + 0.8

        raster_float = raster.astype(jnp.float32)
        flashed_raster = raster_float * flash_intensity

        return jnp.clip(flashed_raster, 0, 255).astype(jnp.uint8)
    
    def _render_simple_text_ids(self, raster: jnp.ndarray, text: str, x: int, y: int, color_id: int) -> jnp.ndarray:
        """Render simple text using a basic bitmap font to the ID raster."""
        char_patterns = {
            'P': jnp.array([[1,1,1,1],[1,0,0,1],[1,1,1,1],[1,0,0,0],[1,0,0,0],[1,0,0,0]], dtype=jnp.int32),
            'L': jnp.array([[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,1,1,1]], dtype=jnp.int32),
            'A': jnp.array([[0,1,1,0],[1,0,0,1],[1,0,0,1],[1,1,1,1],[1,0,0,1],[1,0,0,1]], dtype=jnp.int32),
            'Y': jnp.array([[1,0,0,1],[1,0,0,1],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0]], dtype=jnp.int32),
            'E': jnp.array([[1,1,1,1],[1,0,0,0],[1,1,1,0],[1,0,0,0],[1,0,0,0],[1,1,1,1]], dtype=jnp.int32),
            'R': jnp.array([[1,1,1,0],[1,0,0,1],[1,1,1,0],[1,1,0,0],[1,0,1,0],[1,0,0,1]], dtype=jnp.int32),
            'W': jnp.array([[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,1,1],[1,1,0,1],[1,0,0,1]], dtype=jnp.int32),
            'I': jnp.array([[1,1,1],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[1,1,1]], dtype=jnp.int32),
            'N': jnp.array([[1,0,0,1],[1,1,0,1],[1,0,1,1],[1,0,0,1],[1,0,0,1],[1,0,0,1]], dtype=jnp.int32),
            'S': jnp.array([[0,1,1,1],[1,0,0,0],[0,1,1,0],[0,0,0,1],[0,0,0,1],[1,1,1,0]], dtype=jnp.int32),
            'O': jnp.array([[0,1,1,0],[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,0,1],[0,1,1,0]], dtype=jnp.int32),
            'T': jnp.array([[1,1,1,1,1],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0]], dtype=jnp.int32),
            'C': jnp.array([[0,1,1,1],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[0,1,1,1]], dtype=jnp.int32),
            'K': jnp.array([[1,0,0,1],[1,0,1,0],[1,1,0,0],[1,1,0,0],[1,0,1,0],[1,0,0,1]], dtype=jnp.int32),
            'J': jnp.array([[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1],[1,0,0,1],[0,1,1,0]], dtype=jnp.int32),
            'M': jnp.array([[1,0,0,0,1],[1,1,0,1,1],[1,0,1,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1]], dtype=jnp.int32),
            'D': jnp.array([[1,1,1,0],[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,1,1,0]], dtype=jnp.int32),
            'F': jnp.array([[1,1,1,1],[1,0,0,0],[1,1,1,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]], dtype=jnp.int32),
            'G': jnp.array([[0,1,1,1],[1,0,0,0],[1,0,1,1],[1,0,0,1],[1,0,0,1],[0,1,1,0]], dtype=jnp.int32),
            'V': jnp.array([[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[0,1,0,1,0],[0,1,0,1,0],[0,0,1,0,0]], dtype=jnp.int32),
            ' ': jnp.array([[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]], dtype=jnp.int32),
        }
        cur_x = x
        for ch in text.upper():
            pattern = char_patterns.get(ch)
            if pattern is None:
                cur_x += 2
                continue
            sprite_mask = jnp.where(
                pattern > 0,
                jnp.asarray(color_id, dtype=jnp.uint8),
                jnp.asarray(self.jr.TRANSPARENT_ID, dtype=jnp.uint8)
            )
            raster = self.jr.render_at(raster, cur_x, y, sprite_mask)
            cur_x += pattern.shape[1] + 1
        return raster


class JaxSlotMachine(JaxEnvironment[SlotMachineState, SlotMachineObservation, SlotMachineInfo, SlotMachineConstants]):
    """
    JAX-accelerated implementation of a classic slot machine game.

    This is the main game class that ties everything together. It implements
    the JAXAtari environment interface so it can be used with RL frameworks,
    analysis tools, and the standard JAXAtari ecosystem.

    """
    # Minimal ALE action set for Slot Machine (from scripts/action_space_helper.py)
    ACTION_SET: jnp.ndarray = jnp.array(
        [
            Action.NOOP,
            Action.FIRE,
            Action.UP,    # Player 2 increase wager
            Action.DOWN,  # Player 2 decrease wager
            Action.LEFT,  # Player 1 decrease wager
            Action.RIGHT, # Player 1 increase wager
            Action.UPLEFT, # Toggle jackpot mode
        ],
        dtype=jnp.int32,
    )

    def __init__(
            self,
            consts: SlotMachineConstants = None,
    ):
        """Instantiate the environment and its renderer."""
        consts = consts or SlotMachineConstants()
        super().__init__(consts)
    
        self.renderer = SlotMachineRenderer(consts)

    def reset(self, key: jax.random.PRNGKey = None) -> Tuple[SlotMachineObservation, SlotMachineState]:
        """
        Reset the environment to initial state for a new game.

        Creates a fresh game state with starting credits and random reel positions.
        This is called at the beginning of each episode for RL training,
        or when the player runs out of credits.

        """

        cfg = self.consts

        if key is None:
            import time
            key = jax.random.PRNGKey(int(time.time() * 1000000) % (2 ** 31))

        key, layout_key, pos_key = jax.random.split(key, 3)

        # Use JAX's optimized shuffle instead of manual permutation
        def shuffle_reel(reel_key, base_layout):
            return jax.random.permutation(reel_key, base_layout)

        layout_keys = jax.random.split(layout_key, cfg.num_reels)
        base_layouts = cfg.reel_layouts

        # Vectorized shuffle for all reels
        initial_layouts = jax.vmap(shuffle_reel)(layout_keys, base_layouts)

        # Random initial positions
        pos_keys = jax.random.split(pos_key, cfg.num_reels)
        initial_positions = jax.vmap(
            lambda k: jax.random.randint(k, (), 0, cfg.total_symbols_per_reel)
        )(pos_keys)

        initial_state = SlotMachineState(
            player1_credits=jnp.array(cfg.starting_credits, dtype=jnp.int32),
            player1_total_winnings=jnp.array(0, dtype=jnp.int32),
            player1_spins_played=jnp.array(0, dtype=jnp.int32),
            player1_wager=jnp.array(cfg.min_wager, dtype=jnp.int32),
            player2_credits=jnp.array(cfg.starting_credits, dtype=jnp.int32),
            player2_total_winnings=jnp.array(0, dtype=jnp.int32),
            player2_spins_played=jnp.array(0, dtype=jnp.int32),
            player2_wager=jnp.array(cfg.min_wager, dtype=jnp.int32),
            reel_positions=initial_positions,
            reel_spinning=jnp.zeros(cfg.num_reels, dtype=jnp.bool_),
            spin_timers=jnp.zeros(cfg.num_reels, dtype=jnp.int32),
            reel_speeds=jnp.ones(cfg.num_reels, dtype=jnp.int32),
            reel_layouts=initial_layouts,
            spin_button_prev=jnp.array(False, dtype=jnp.bool_),
            up_button_prev=jnp.array(False, dtype=jnp.bool_),
            down_button_prev=jnp.array(False, dtype=jnp.bool_),
            left_button_prev=jnp.array(False, dtype=jnp.bool_),
            right_button_prev=jnp.array(False, dtype=jnp.bool_),
            jackpot_button_prev=jnp.array(False, dtype=jnp.bool_),
            spin_cooldown=jnp.array(0, dtype=jnp.int32),
            jackpot_mode=jnp.array(False, dtype=jnp.bool_),
            jackpot_message_timer=jnp.array(0, dtype=jnp.int32),
            win_flash_timer=jnp.array(0, dtype=jnp.int32),
            last_payout_p1=jnp.array(0, dtype=jnp.int32),
            last_payout_p2=jnp.array(0, dtype=jnp.int32),
            last_reward_p1=jnp.array(0.0, dtype=jnp.float32),
            last_reward_p2=jnp.array(0.0, dtype=jnp.float32),
            game_over=jnp.array(False, dtype=jnp.bool_),
            winner=jnp.array(0, dtype=jnp.int32),
            win_display_timer=jnp.array(0, dtype=jnp.int32),
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
        1. Process player input (button presses for both players)
        2. Handle wager adjustments for both players
        3. Start new spins if requested (shared reels)
        4. Update reel animations
        5. Check for wins when reels stop (both players can win)
        6. Update timers and effects
        7. Calculate rewards and check game over

        """
        cfg = self.consts
        previous_state = state

        # Check if game is already over and display timer has expired
        game_completely_over = state.game_over & (state.win_display_timer <= 0)
        
        def early_return():
            obs = self._get_observation(state)
            info = self._get_info(state)
            return obs, state, 0.0, True, info
            
        def continue_game():
            # Translate compact agent action index to ALE console action
            atari_action = jnp.take(self.ACTION_SET, action.astype(jnp.int32))
            
            # Update RNG key EVERY step
            step_key, new_rng = jax.random.split(state.rng)

            # Process player input
            fire_pressed = (atari_action == Action.FIRE)
            up_pressed = (atari_action == Action.UP)        # Player 2 increase wager
            down_pressed = (atari_action == Action.DOWN)    # Player 2 decrease wager
            left_pressed = (atari_action == Action.LEFT)    # Player 1 decrease wager
            right_pressed = (atari_action == Action.RIGHT)  # Player 1 increase wager
            j_pressed = (atari_action == Action.UPLEFT)     # Jackpot mode toggle

            # Detect "just pressed" events to prevent button mashing
            fire_just_pressed = fire_pressed & (~state.spin_button_prev)
            up_just_pressed = up_pressed & (~state.up_button_prev)
            down_just_pressed = down_pressed & (~state.down_button_prev)
            left_just_pressed = left_pressed & (~state.left_button_prev)
            right_just_pressed = right_pressed & (~state.right_button_prev)
            j_just_pressed = j_pressed & (~state.jackpot_button_prev)

            # Handle jackpot mode toggle (only when no spins have been played)
            spins_played = state.player1_spins_played + state.player2_spins_played
            can_toggle_jackpot = (spins_played == 0) & (~jnp.any(state.reel_spinning)) & j_just_pressed
            
            new_jackpot_mode = jax.lax.cond(
                can_toggle_jackpot,
                lambda jm: ~jm,  # Toggle jackpot mode
                lambda jm: jm,   # Keep current state
                state.jackpot_mode
            )
            
            # Set message timer when toggling jackpot mode
            new_jackpot_message_timer = jax.lax.cond(
                can_toggle_jackpot,
                lambda _: jnp.array(120, dtype=jnp.int32),  # Show message for 2 seconds at 60fps
                lambda _: state.jackpot_message_timer,
                None
            )

            # Determine if players can spin (both players need enough credits)
            can_spin = (state.spin_cooldown == 0) & fire_just_pressed
            can_spin = can_spin & (state.player1_credits >= state.player1_wager)
            can_spin = can_spin & (state.player2_credits >= state.player2_wager)
            can_spin = can_spin & (~jnp.any(state.reel_spinning))
            can_spin = can_spin & (~state.game_over)

            # Handle wager changes (only when machine is idle)
            can_change_wager = ~jnp.any(state.reel_spinning) & (state.spin_cooldown == 0) & (~state.game_over)

            # Player 1 wager adjustment logic
            new_p1_wager = jax.lax.cond(
                can_change_wager & up_just_pressed,
                lambda w: jnp.minimum(w + 1, cfg.max_wager),
                lambda w: jax.lax.cond(
                    can_change_wager & down_just_pressed,
                    lambda w: jnp.maximum(w - 1, cfg.min_wager),
                    lambda w: w,
                    w
                ),
                state.player1_wager
            )

            # Player 2 wager adjustment logic
            new_p2_wager = jax.lax.cond(
                can_change_wager & right_just_pressed,
                lambda w: jnp.minimum(w + 1, cfg.max_wager),
                lambda w: jax.lax.cond(
                    can_change_wager & left_just_pressed,
                    lambda w: jnp.maximum(w - 1, cfg.min_wager),
                    lambda w: w,
                    w
                ),
                state.player2_wager
            )

            # Update state with input tracking, wager changes and new RNG
            new_state = state._replace(
                spin_button_prev=fire_pressed,
                up_button_prev=up_pressed,
                down_button_prev=down_pressed,
                left_button_prev=left_pressed,
                right_button_prev=right_pressed,
                jackpot_button_prev=j_pressed,
                player1_wager=new_p1_wager,
                player2_wager=new_p2_wager,
                jackpot_mode=new_jackpot_mode,
                jackpot_message_timer=new_jackpot_message_timer,
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

            # Check game over conditions and set winner
            new_state = self._check_game_over(new_state)

            # Calculate combined reward for this step
            reward = jnp.where(
                (new_state.last_payout_p1 > 0) & (state.last_payout_p1 == 0),
                new_state.last_payout_p1.astype(jnp.float32),
                0.0
            ) + jnp.where(
                (new_state.last_payout_p2 > 0) & (state.last_payout_p2 == 0),
                new_state.last_payout_p2.astype(jnp.float32),
                0.0
            )

            # Update reward in state for display purposes
            new_state = new_state._replace(
                last_reward_p1=jnp.where(
                    (new_state.last_payout_p1 > 0) & (state.last_payout_p1 == 0),
                    new_state.last_payout_p1.astype(jnp.float32),
                    0.0
                ),
                last_reward_p2=jnp.where(
                    (new_state.last_payout_p2 > 0) & (state.last_payout_p2 == 0),
                    new_state.last_payout_p2.astype(jnp.float32),
                    0.0
                )
            )

            # Game over condition
            done = self._get_done(new_state)

            obs = self._get_observation(new_state)
            info = self._get_info(new_state)

            return obs, new_state, reward, done, info
        
        # Use jax.lax.cond for the conditional logic
        return jax.lax.cond(
            game_completely_over,
            lambda _: early_return(),
            lambda _: continue_game(),
            None
        )



    def _start_spin(self, state: SlotMachineState, key: jax.random.PRNGKey) -> SlotMachineState:
        """
        Start spinning the reels using provided RNG key. This function sets up a new spin: deducts credits,
        generates random outcomes, and starts the reel animations for both players.
        """
        cfg = self.consts

        # Deduct wagers from both players
        new_p1_credits = state.player1_credits - state.player1_wager
        new_p2_credits = state.player2_credits - state.player2_wager
        
        new_p1_spins = state.player1_spins_played + 1
        new_p2_spins = state.player2_spins_played + 1

        key, *reel_keys = jax.random.split(key, cfg.num_reels + 1)

        # Instead of full permutation, just shuffle positions
        def generate_reel_data(reel_key, base_layout):
            duration_key, pos_key = jax.random.split(reel_key)

            # Simple shuffle is more efficient than full permutation
            layout = jax.random.permutation(duration_key, base_layout)

            # Generate other parameters
            duration = jax.random.randint(
                duration_key, (),
                cfg.min_spin_duration,
                cfg.max_spin_duration
            )
            position = jax.random.randint(pos_key, (), 0, cfg.total_symbols_per_reel)

            return layout, duration, position

        # Vectorize across all reels
        reel_keys_array = jnp.array(reel_keys)
        
        # Use jackpot layouts if jackpot mode is enabled, otherwise use normal layouts
        base_layouts = jax.lax.cond(
            state.jackpot_mode,
            lambda: cfg.reel_layouts_jackpot,
            lambda: cfg.reel_layouts
        )

        layouts, durations, positions = jax.vmap(generate_reel_data)(reel_keys_array, base_layouts)

        # Add staggered delays for realistic reel stopping
        stagger_delays = jnp.arange(cfg.num_reels) * cfg.reel_stop_delay
        final_durations = durations + stagger_delays

        return state._replace(
            player1_credits=new_p1_credits,
            player2_credits=new_p2_credits,
            player1_spins_played=new_p1_spins,
            player2_spins_played=new_p2_spins,
            reel_spinning=jnp.ones(cfg.num_reels, dtype=jnp.bool_),
            spin_timers=final_durations,
            reel_positions=positions,
            reel_layouts=layouts,
            spin_cooldown=jnp.array(10, dtype=jnp.int32),
            last_payout_p1=jnp.array(0, dtype=jnp.int32),
            last_payout_p2=jnp.array(0, dtype=jnp.int32),
            last_reward_p1=jnp.array(0.0, dtype=jnp.float32),
            last_reward_p2=jnp.array(0.0, dtype=jnp.float32),
        )

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
        cfg = self.consts

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
        Check for winning combinations after all reels have stopped for both players.

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
        cfg = self.consts

        all_stopped = ~jnp.any(state.reel_spinning)
        has_spun = (state.player1_spins_played > 0) | (state.player2_spins_played > 0)
        should_check_win = all_stopped & (state.last_payout_p1 == 0) & (state.last_payout_p2 == 0) & has_spun

        def process_win_vectorized(s: SlotMachineState) -> SlotMachineState:
            """
            Helper function to process win calculation with reward system for both players.
            """

            center_indices = (s.reel_positions + 1) % cfg.total_symbols_per_reel
            center_symbols = s.reel_layouts[jnp.arange(cfg.num_reels), center_indices]

            s0, s1, s2 = center_symbols[0], center_symbols[1], center_symbols[2]

            # Calculate winnings for payout mode
            def payout_mode_winnings():
                #  Win condition evaluation for normal mode
                win_conditions = jnp.array([
                    jnp.all(center_symbols == 5),  # Three Cars - 200x
                    jnp.all(center_symbols == 2),  # Three Bars - 100x
                    jnp.all(center_symbols == 4),  # Three Bells - 18x
                    (s0 == 4) & (s1 == 4) & (s2 == 2),  # Two Bells + Bar - 18x
                    jnp.all(center_symbols == 3),  # Three TVs - 14x
                    (s0 == 3) & (s1 == 3) & (s2 == 2),  # Two TVs + Bar - 14x
                    jnp.all(center_symbols == 1),  # Three Tables - 10x
                    (s0 == 1) & (s1 == 1) & (s2 == 2),  # Two Tables + Bar - 10x
                    (s0 == 0) & (s1 == 0),  # Cactus on reels 1 & 2 - 5x
                    (s0 == 0) & (s1 != 0) & (s2 != 0),  # Cactus on reel 1 only - 2x
                ])

                multipliers = jnp.array([200, 100, 18, 18, 14, 14, 10, 10, 5, 2])
                
                # Find first winning condition (highest priority)
                win_indices = jnp.where(win_conditions, jnp.arange(len(win_conditions)), len(win_conditions))
                first_win = jnp.min(win_indices)
                has_win = first_win < len(win_conditions)
                
                return has_win, jax.lax.cond(has_win, lambda: multipliers[first_win], lambda: 0)

            # Calculate winnings for payout mode
            def jackpot_mode_winnings():
                # Jackpot mode payout rules:
                # - Three cars (all matching) -> 200x 
                # - Three bars (all matching) -> 100x
                # - Any three symbols on center line (not empty) -> 20x
                
                # Check if all center symbols are non-empty
                all_non_empty = jnp.all(center_symbols != 6)
                
                # Check for specific high-value combinations
                three_cars = jnp.all(center_symbols == 5)
                three_bars = jnp.all(center_symbols == 2)
                
                # Determine multiplier with priority order
                def determine_multiplier():
                    return jax.lax.cond(
                        three_cars,
                        lambda: 200,  # Three cars gets highest priority
                        lambda: jax.lax.cond(
                            three_bars,
                            lambda: 100,  # Three bars gets second priority
                            lambda: 20    # Any other three non-empty symbols get 20x
                        )
                    )
                
                # Calculate final multiplier
                multiplier = jax.lax.cond(
                    all_non_empty,
                    determine_multiplier,
                    lambda: 0  # If any symbol is empty, no payout
                )
                
                return all_non_empty, multiplier

            # Use appropriate payout calculation based on jackpot mode
            has_win, multiplier = jax.lax.cond(
                s.jackpot_mode,
                jackpot_mode_winnings,
                payout_mode_winnings
            )

            # Calculate payouts for both players
            payout_p1 = jax.lax.cond(
                has_win,
                lambda: multiplier * s.player1_wager,
                lambda: 0
            )
            
            payout_p2 = jax.lax.cond(
                has_win,
                lambda: multiplier * s.player2_wager,
                lambda: 0
            )

            # Determine flash timer - flash if either player wins
            flash_timer = jnp.where((payout_p1 > 0) | (payout_p2 > 0), 60, 0)

            return s._replace(
                player1_credits=s.player1_credits + payout_p1,
                player2_credits=s.player2_credits + payout_p2,
                player1_total_winnings=s.player1_total_winnings + payout_p1,
                player2_total_winnings=s.player2_total_winnings + payout_p2,
                last_payout_p1=payout_p1,
                last_payout_p2=payout_p2,
                win_flash_timer=flash_timer,
            )

        return jax.lax.cond(should_check_win, process_win_vectorized, lambda s: s, state)

    def _update_timers(self, state: SlotMachineState) -> SlotMachineState:
        """
        Update various timers and cooldowns. Manages all the timing-based effects in the game.

        Timer Types:
        - Spin cooldown: Prevents accidental double-spins
        - Win flash: Controls celebration effect duration
        - Win display: Controls how long winner message is shown
        - Jackpot message: Controls how long jackpot mode message is shown

        All timers count down to 0 and stop there (no negative values).

        """
        new_cooldown = jnp.maximum(state.spin_cooldown - 1, 0)
        new_flash_timer = jnp.maximum(state.win_flash_timer - 1, 0)
        new_win_display_timer = jnp.maximum(state.win_display_timer - 1, 0)
        new_jackpot_message_timer = jnp.maximum(state.jackpot_message_timer - 1, 0)

        return state._replace(
            spin_cooldown=new_cooldown,
            win_flash_timer=new_flash_timer,
            win_display_timer=new_win_display_timer,
            jackpot_message_timer=new_jackpot_message_timer,
        )

    def _check_game_over(self, state: SlotMachineState) -> SlotMachineState:
        """
        Check game over conditions and determine winner.
        Game ends when:
        1. Either player reaches 0 credits AND cannot afford their current wager
        2. Either player reaches more than 999 credits
        
        Important: Game should only end AFTER reels stop and final payouts are processed
        """
        cfg = self.consts

        # Check if game should end - but only when reels are not spinning
        # This ensures final spins complete and payouts are processed
        reels_stopped = ~jnp.any(state.reel_spinning)
        
        # Check bankruptcy conditions - player must not afford their next wager
        p1_broke = (state.player1_credits < 1) & reels_stopped
        p2_broke = (state.player2_credits < 1) & reels_stopped
        
        # Check win conditions - over 999 credits
        p1_rich = (state.player1_credits > 999) & reels_stopped
        p2_rich = (state.player2_credits > 999) & reels_stopped

        should_end = p1_broke | p2_broke | p1_rich | p2_rich
        
        # Determine winner
        # Priority: Rich > not broke, but if both broke = game over (no winner)
        winner = jax.lax.cond(
            should_end,
            lambda: jax.lax.cond(
                p1_rich,
                lambda: 1,  # Player 1 wins (got rich)
                lambda: jax.lax.cond(
                    p2_rich,
                    lambda: 2,  # Player 2 wins (got rich)
                    lambda: jax.lax.cond(
                        p1_broke & p2_broke,
                        lambda: 0,  # Both players broke - game over, no winner
                        lambda: jax.lax.cond(
                            p1_broke,
                            lambda: 2,  # Player 2 wins (Player 1 broke)
                            lambda: jax.lax.cond(
                                p2_broke,
                                lambda: 1,  # Player 1 wins (Player 2 broke)
                                lambda: 0   # No winner yet
                            )
                        )
                    )
                )
            ),
            lambda: 0  # Game continues
        )
        
        # Set display timer when game just ended
        display_timer = jax.lax.cond(
            should_end & (~state.game_over),
            lambda: 60,  # Show winner for 1 second (60 frames)
            lambda: state.win_display_timer
        )
        
        return state._replace(
            game_over=should_end,
            winner=winner,
            win_display_timer=display_timer
        )

    def _get_observation(self, state: SlotMachineState) -> SlotMachineObservation:
        """
        Extract observation from game state.
        """

        cfg = self.consts

        # Get currently visible symbols for each reel from static layout
        reel_indices = jnp.arange(cfg.num_reels)[:, None]
        slot_indices = jnp.arange(cfg.symbols_per_reel)[None, :]

        # Compute all symbol indices at once
        symbol_indices = (state.reel_positions[:, None] + slot_indices) % cfg.total_symbols_per_reel
        reel_symbols = state.reel_layouts[reel_indices, symbol_indices]

        return SlotMachineObservation(
            player1_credits=state.player1_credits,
            player1_wager=state.player1_wager,
            player2_credits=state.player2_credits,
            player2_wager=state.player2_wager,
            reel_symbols=reel_symbols,
            is_spinning=jnp.array(jnp.any(state.reel_spinning), dtype=jnp.int32),
            last_payout_p1=state.last_payout_p1,
            last_payout_p2=state.last_payout_p2,
            last_reward_p1=state.last_reward_p1,
            last_reward_p2=state.last_reward_p2,
            jackpot_mode=state.jackpot_mode,
            game_over=state.game_over,
            winner=state.winner,
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
        2 = UP (Player 1 increase wager)
        3 = DOWN (Player 1 decrease wager)
        4 = LEFT (Player 2 decrease wager)
        5 = RIGHT (Player 2 increase wager)
        6 = UPLEFT (Toggle jackpot mode)
        """
        return spaces.Discrete(len(self.ACTION_SET))

    def observation_space(self) -> spaces.Space:
        """
        Get the observation space for this environment. Defines the structure and bounds of observations.

        Space Design:
        - Player 1 & 2 Credits: 0-9999
        - Player 1 & 2 Wager: min_wager to max_wager for configurable betting range. (0-5)
        - Reel symbols: 0 to num_symbol_types-1.  Symbol type indices currently
         ( 0 => "Cactus", 1 => "Table", 2 => "Bar", 3 => "TV", 4 => "Bell", 5 => "Car")
        - Spinning: boolean (machine status)
        - Payouts/rewards (Integer to keep track of payouts for both players)
        - Jackpot mode: boolean (is jackpot mode enabled)
        - Game over and winner status
        """
        cfg = self.consts

        return spaces.Dict({
            'player1_credits': spaces.Box(low=0, high=9999, shape=(), dtype=jnp.float32),
            'player1_wager': spaces.Box(low=cfg.min_wager, high=cfg.max_wager, shape=(), dtype=jnp.float32),
            'player2_credits': spaces.Box(low=0, high=9999, shape=(), dtype=jnp.float32),
            'player2_wager': spaces.Box(low=cfg.min_wager, high=cfg.max_wager, shape=(), dtype=jnp.float32),
            'reel_symbols': spaces.Box(
                low=0, high=cfg.num_symbol_types - 1,
                shape=(cfg.num_reels, cfg.symbols_per_reel),
                dtype=jnp.float32
            ),
            'is_spinning': spaces.Box(low=0, high=1, shape=(), dtype=jnp.float32),
            'last_payout_p1': spaces.Box(low=0, high=999, shape=(), dtype=jnp.float32),
            'last_payout_p2': spaces.Box(low=0, high=999, shape=(), dtype=jnp.float32),
            'last_reward_p1': spaces.Box(low=0.0, high=999.0, shape=(), dtype=jnp.float32),
            'last_reward_p2': spaces.Box(low=0.0, high=999.0, shape=(), dtype=jnp.float32),
            'jackpot_mode': spaces.Box(low=0, high=1, shape=(), dtype=jnp.float32),
            'game_over': spaces.Box(low=0, high=1, shape=(), dtype=jnp.float32),
            'winner': spaces.Box(low=0, high=2, shape=(), dtype=jnp.float32),
        })

    def image_space(self) -> spaces.Space:
        """Image space describing rendered RGB frames."""
        cfg = self.consts
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
            jnp.atleast_1d(obs.player1_credits).astype(jnp.float32),
            jnp.atleast_1d(obs.player1_wager).astype(jnp.float32),
            jnp.atleast_1d(obs.player2_credits).astype(jnp.float32),
            jnp.atleast_1d(obs.player2_wager).astype(jnp.float32),
            obs.reel_symbols.astype(jnp.float32).ravel(),
            jnp.atleast_1d(obs.is_spinning).astype(jnp.float32),
            jnp.atleast_1d(obs.last_payout_p1).astype(jnp.float32),
            jnp.atleast_1d(obs.last_payout_p2).astype(jnp.float32),
            jnp.atleast_1d(obs.last_reward_p1).astype(jnp.float32),
            jnp.atleast_1d(obs.last_reward_p2).astype(jnp.float32),
            jnp.atleast_1d(obs.jackpot_mode).astype(jnp.float32),
            jnp.atleast_1d(obs.game_over).astype(jnp.float32),
            jnp.atleast_1d(obs.winner).astype(jnp.float32),
        ]
        return jnp.concatenate(components, axis=0)

    def _get_info(
        self,
        state: SlotMachineState,
    ) -> SlotMachineInfo:

        return SlotMachineInfo(
            player1_total_winnings=state.player1_total_winnings,
            player2_total_winnings=state.player2_total_winnings,
            player1_spins_played=state.player1_spins_played,
            player2_spins_played=state.player2_spins_played,
        )

    def _get_reward(
        self,
        previous_state: SlotMachineState,
        state: SlotMachineState,
    ) -> jnp.ndarray:
        """Return the associated reward (combined for both players)."""
        reward_p1 = jnp.asarray(state.last_reward_p1, dtype=jnp.float32)
        reward_p2 = jnp.asarray(state.last_reward_p2, dtype=jnp.float32)
        return reward_p1 + reward_p2

    def _get_done(self, state: SlotMachineState) -> jnp.bool_:
        """Check if the game is over and display timer has expired"""
        return state.game_over & (state.win_display_timer <= 0)