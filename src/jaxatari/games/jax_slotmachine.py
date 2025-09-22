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

Game End Conditions:
- Either player reaches 0 credits (other player wins)
- Either player reaches more than 999 credits (that player wins)

Reward System: (More information under https://www.atarimania.com/game-atari-2600-vcs-slot-machine_8212.html)

The payline (from left to right) pays as follows for both players:

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

    # Tuned reel strips after ten thousand simulations to keep the bankroll drifting downward on the long run.
    reel_layouts: jnp.ndarray = field(
        default_factory=lambda: jnp.array([
            [0, 1, 3, 4, 0, 2, 3, 4, 1, 3, 4, 2, 3, 4, 1, 3, 5, 1, 3, 4],
            [0, 2, 3, 1, 5, 0, 2, 3, 1, 5, 0, 2, 3, 5, 5, 0, 2, 3, 4, 5],
            [0, 0, 3, 4, 5, 0, 0, 1, 4, 5, 0, 0, 1, 4, 5, 0, 2, 3, 4, 5],
        ], dtype=jnp.int32)
    )

    # Reel positions - UI layout coordinates
    reel_start_x: int = 11
    reel_start_y: int = 50


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
    spin_cooldown: chex.Array

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

    # Keeps track of rewards for use by external api
    all_rewards: Optional[jnp.ndarray] = None


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

    def _create_reel_frame(self) -> jnp.ndarray:
        """Create reel frame """
        h, w = self.config.reel_height, self.config.reel_width
        frame = jnp.zeros((h, w, 4), dtype=jnp.uint8)

        light_blue = jnp.array([70, 82, 184], dtype=jnp.uint8)
        alpha = jnp.uint8(255)

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
                symbol_type = state.reel_layouts[reel_idx, symbol_index]

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
        raster = self._render_credits_display(raster, state)

        # Apply win flash effect if we hit a win
        raster = jax.lax.cond(
            state.win_flash_timer > 0,
            lambda r: self._render_win_flash(r, state.win_flash_timer),
            lambda r: r,
            raster
        )

        return raster

    def _render_winner_message(self, raster: jnp.ndarray, winner: chex.Array) -> jnp.ndarray:
        """Render winner message in the center of the screen."""
        cfg = self.config
        
        # Message position (center of screen)
        msg_x = cfg.screen_width // 2 - 50  # Increased space for longer messages
        msg_y = cfg.screen_height // 2 - 10

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
            r = self._render_simple_text(r, "GAME OVER", msg_x + 10, msg_y, text_color)
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

    def _render_credits_display(self, raster: jnp.ndarray, state: SlotMachineState) -> jnp.ndarray:
        """Draw the credits and wager boxes for both players."""
        raster = self._render_text_labels(raster, state)
        return raster

    def _render_text_labels(self, raster: jnp.ndarray, state: SlotMachineState) -> jnp.ndarray:
        """Helper function to draw the HUD plates and insert credits and wagers for both players.
        """
        cfg = self.config

        # Calculate reel positions
        first_reel_x = cfg.reel_start_x
        third_reel_x = cfg.reel_start_x + 2 * (cfg.reel_width + cfg.reel_spacing)
        reel_y = cfg.reel_start_y

        # Colors
        box_color = jnp.array([70, 82, 184], dtype=jnp.uint8)
        number_color = jnp.array([194, 67, 115], dtype=jnp.uint8)

        # Player 1 (right side)
        # UI for player 1 credits
        credits_y = reel_y - 22
        raster = self._draw_colored_box(raster, third_reel_x - 3, credits_y - 2, 47, 16, box_color)
        credits_digits = aj.int_to_digits(state.player1_credits, max_digits=4)
        raster = self._render_colored_number(raster, credits_digits, third_reel_x + 2, credits_y + 2, number_color)

        # UI for player 1 wager
        wager_y = reel_y + cfg.reel_height + 10
        raster = self._draw_colored_box(raster, third_reel_x + 7, wager_y - 4, 27, 16, box_color)
        wager_digits = aj.int_to_digits(state.player1_wager, max_digits=2)
        raster = self._render_colored_number(raster, wager_digits, third_reel_x + 12, wager_y, number_color)

        # Player 2 (left side)
        # UI for player 2 credits (above first reel)
        raster = self._draw_colored_box(raster, first_reel_x - 3, credits_y - 2, 47, 16, box_color)
        credits_digits_p2 = aj.int_to_digits(state.player2_credits, max_digits=4)
        raster = self._render_colored_number(raster, credits_digits_p2, first_reel_x + 2, credits_y + 2, number_color)

        # UI for player 2 wager (below first reel)
        raster = self._draw_colored_box(raster, first_reel_x + 7, wager_y - 4, 27, 16, box_color)
        wager_digits_p2 = aj.int_to_digits(state.player2_wager, max_digits=2)
        raster = self._render_colored_number(raster, wager_digits_p2, first_reel_x + 12, wager_y, number_color)

        # Check if game is over, reels have stopped, and display winner message
        raster = jax.lax.cond(
            state.game_over & (state.win_display_timer > 0) & (~jnp.any(state.reel_spinning)),
            lambda r: self._render_winner_message(r, state.winner),
            lambda r: r,
            raster
        )

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

        def render_digit():
            # Get the region where the digits should be rendered
            target_region = raster[y:y+digit_h, x:x+digit_w, :]
            
            # Oonly render visible pixels
            digit_mask = pattern > 0
            
            # Only update pixels where the pattern is > 0, leave others unchanged (transparent)
            new_region = jnp.where(
                digit_mask[..., None],
                color[None, None, :],
                target_region
            )
            
            # Update only the digit region
            return raster.at[y:y+digit_h, x:x+digit_w, :].set(new_region)

        return jax.lax.cond(valid_render, render_digit, lambda: raster)

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
            Action.UP,    # Player 2 increase wager
            Action.DOWN,  # Player 2 decrease wager
            Action.LEFT,  # Player 1 decrease wager
            Action.RIGHT, # Player 1 increase wager
        ]

    def reset(self, key: jax.random.PRNGKey = None) -> Tuple[SlotMachineObservation, SlotMachineState]:
        """
        Reset the environment to initial state for a new game.

        Creates a fresh game state with starting credits and random reel positions.
        This is called at the beginning of each episode for RL training,
        or when the player runs out of credits.

        """

        cfg = self.config

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
            spin_cooldown=jnp.array(0, dtype=jnp.int32),
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
        cfg = self.config
        previous_state = state

        # Check if game is already over and display timer has expired
        game_completely_over = state.game_over & (state.win_display_timer <= 0)
        
        def early_return():
            obs = self._get_observation(state)
            all_rewards = self._get_all_reward(previous_state, state)
            info = self._get_info(state, all_rewards)
            return obs, state, 0.0, True, info
            
        def continue_game():
            # Update RNG key EVERY step
            step_key, new_rng = jax.random.split(state.rng)

            # Process player input
            fire_pressed = (action == Action.FIRE)
            up_pressed = (action == Action.UP)        # Player 2 increase wager
            down_pressed = (action == Action.DOWN)    # Player 2 decrease wager
            left_pressed = (action == Action.LEFT)    # Player 1 decrease wager
            right_pressed = (action == Action.RIGHT)  # Player 1 increase wager

            # Detect "just pressed" events to prevent button mashing
            fire_just_pressed = fire_pressed & (~state.spin_button_prev)
            up_just_pressed = up_pressed & (~state.up_button_prev)
            down_just_pressed = down_pressed & (~state.down_button_prev)
            left_just_pressed = left_pressed & (~state.left_button_prev)
            right_just_pressed = right_pressed & (~state.right_button_prev)

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
                player1_wager=new_p1_wager,
                player2_wager=new_p2_wager,
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
            all_rewards = self._get_all_reward(previous_state, new_state)
            info = self._get_info(new_state, all_rewards)

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
        cfg = self.config

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
        base_layouts = cfg.reel_layouts

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
        cfg = self.config

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

            #  Win condition evaluation
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

            # Calculate payouts for both players
            payout_p1 = jax.lax.cond(
                has_win,
                lambda: multipliers[first_win] * s.player1_wager,
                lambda: 0
            )
            
            payout_p2 = jax.lax.cond(
                has_win,
                lambda: multipliers[first_win] * s.player2_wager,
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

        All timers count down to 0 and stop there (no negative values).

        """
        new_cooldown = jnp.maximum(state.spin_cooldown - 1, 0)
        new_flash_timer = jnp.maximum(state.win_flash_timer - 1, 0)
        new_win_display_timer = jnp.maximum(state.win_display_timer - 1, 0)

        return state._replace(
            spin_cooldown=new_cooldown,
            win_flash_timer=new_flash_timer,
            win_display_timer=new_win_display_timer,
        )

    def _check_game_over(self, state: SlotMachineState) -> SlotMachineState:
        """
        Check game over conditions and determine winner.
        Game ends when:
        1. Either player reaches 0 credits AND cannot afford their current wager
        2. Either player reaches more than 999 credits
        
        Important: Game should only end AFTER reels stop and final payouts are processed
        """
        cfg = self.config

        # Check if game should end - but only when reels are not spinning
        # This ensures final spins complete and payouts are processed
        reels_stopped = ~jnp.any(state.reel_spinning)
        
        # Check bankruptcy conditions - player must not afford their next wager
        p1_broke = (state.player1_credits < state.player1_wager) & reels_stopped
        p2_broke = (state.player2_credits < state.player2_wager) & reels_stopped
        
        # Check win conditions - over 999 credits
        p1_rich = (state.player1_credits > 999) & reels_stopped
        p2_rich = (state.player2_credits > 999) & reels_stopped

        should_end = p1_broke | p2_broke | p1_rich | p2_rich
        
        # Determine winner
        # Priority: Rich > not broke
        winner = jax.lax.cond(
            should_end,
            lambda: jax.lax.cond(
                p1_rich,
                lambda: 1,  # Player 1 wins (got rich)
                lambda: jax.lax.cond(
                    p2_rich,
                    lambda: 2,  # Player 2 wins (got rich)
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

        cfg = self.config

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
        """
        return spaces.Discrete(len(self.action_set))

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
        - Game over and winner status
        """
        cfg = self.config

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
            'game_over': spaces.Box(low=0, high=1, shape=(), dtype=jnp.float32),
            'winner': spaces.Box(low=0, high=2, shape=(), dtype=jnp.float32),
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
            jnp.atleast_1d(obs.game_over).astype(jnp.float32),
            jnp.atleast_1d(obs.winner).astype(jnp.float32),
        ]
        return jnp.concatenate(components, axis=0)

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(
        self,
        previous_state: SlotMachineState,
        state: SlotMachineState,
    ) -> jnp.ndarray:
        if self.reward_funcs is None:
            return jnp.zeros(1, dtype=jnp.float32)
        rewards = jnp.array(
            [reward_func(previous_state, state) for reward_func in self.reward_funcs],
            dtype=jnp.float32,
        )
        return rewards

    def _get_info(
        self,
        state: SlotMachineState,
        all_rewards: Optional[jnp.ndarray] = None,
    ) -> SlotMachineInfo:

        return SlotMachineInfo(
            player1_total_winnings=state.player1_total_winnings,
            player2_total_winnings=state.player2_total_winnings,
            player1_spins_played=state.player1_spins_played,
            player2_spins_played=state.player2_spins_played,
            all_rewards=all_rewards,
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

    print(" JAX SLOT MACHINE - TWO PLAYER")
    print(f"Random seed: {random_seed}")
    print("Controls:")
    print("  SPACE = Spin (both players)")
    print("  Player 2: LEFT/RIGHT = Decrease/Increase wager")
    print("  Player 1: DOWN/UP = Decrease/Increase wager")
    print("  ESC = Quit")
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
                    action = Action.UP      # Player 1 increase wager
                elif event.key == pygame.K_DOWN:
                    action = Action.DOWN    # Player 1 decrease wager
                elif event.key == pygame.K_LEFT:
                    action = Action.LEFT    # Player 2 increase wager (W key)
                elif event.key == pygame.K_RIGHT:
                    action = Action.RIGHT   # Player 2 decrease wager (S key)
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


        if done:
            # Check who won
            winner = int(np.array(state.winner))
            if winner == 1:
                print("Player 1 wins! Thanks for playing!")
            elif winner == 2:
                print("Player 2 wins! Thanks for playing!")
            else:
                print("Game over! Thanks for playing!")
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
    print("Game ended. Use 'python scripts/play.py --game slotmachine' for full gameplay !")


if __name__ == "__main__":
    main()
