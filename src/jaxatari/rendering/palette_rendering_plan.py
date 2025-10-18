from pathlib import Path, PureWindowsPath
import os
import numpy as np
import jax.numpy as jnp
import jax
from functools import partial
from jax import lax, ShapeDtypeStruct
from typing import Dict, List, NamedTuple, Tuple
from jax.experimental import pallas as pl

TRANSPARENT_ID = 255

'''
General structure for this renderer:
1. Sprite loaders (mixture of palette and planned rendering), preprocesses sprite to have masks, sprite info table, background, etc. and scales them!
2. all render functions from planned rendering, as they should be identical -> maybe include pre-calculations into render_at..
3. execute plan function that takes the plan and the assets and builds the mask raster (correctly scaled) -> pallas kernel?
'''

class RendererConfig(NamedTuple):
    """Configuration for the rendering pipeline."""
    # TODO: uses HWC since everything does right now, but might be counterintuitive during usage
    # Target dimensions
    game_dimensions: Tuple[int, int] = (210, 160)  # (height, width) this is normally constant except for some games (sir lancelot for example)
    channels: int = 3  # 1 for grayscale, 3 for RGB
    downscale: Tuple[int, int] = None  # (height, width) to downscale to, or None for no downscaling

    @property
    def width_scaling(self) -> float:
        return self.downscale[1] / self.game_dimensions[1] if self.downscale else 1.0

    @property
    def height_scaling(self) -> float:
        return self.downscale[0] / self.game_dimensions[0] if self.downscale else 1.0


'''
Commands:
    0	screen_x	The final X coordinate on the screen.
    1	screen_y	The final Y coordinate on the screen.
    2	atlas_u	The top-left X coordinate of the sprite in the Texture Atlas.
    3	atlas_v	The top-left Y coordinate of the sprite in the Texture Atlas.
    4	width	The width of the sprite.
    5	height	The height of the sprite.
    6	flip_h	A flag (0 or 1) for horizontal flipping.
    7	flip_v	A flag (0 or 1) for vertical flipping.
    8	depth_z	The draw order. Higher numbers are drawn on top.
'''
class RenderPlan(NamedTuple):
    # A (N, 9) array where N is the max number of sprites on screen.
    # Each row is a single draw command.
    commands: jnp.ndarray
    # A single integer tracking how many commands have been added.
    command_count: jnp.ndarray
    # Small metadata table for sprites: (num_sprites, 4) -> (atlas_u, atlas_v, w, h)
    sprite_info_table: jnp.ndarray


class AgnosticPath(Path):
    """A class that can handle input with Windows (\\) and/or posix (/) separators for paths"""

    def __new__(cls, *args, **kwargs):
        win_path = PureWindowsPath(*args)
        parts = win_path.parts
        if os.name != "nt" and len(parts) > 0:
            if len(parts[0]) == 2 and parts[0][1] == ":":
                parts = parts[1:]
            if parts and not parts[0] in ("/", "\\"):
                parts = ("/",) + parts
        return super().__new__(cls, *parts, **kwargs)


def _create_id_mask(
    sprite_data: jnp.ndarray, color_to_id: Dict[Tuple[int, int, int], int]
) -> jnp.ndarray:
    """Converts a single RGBA sprite into a 2D palette ID mask."""
    height, width, _ = sprite_data.shape
    id_mask = np.zeros((height, width), dtype=np.uint8)
    pixels_np = np.array(sprite_data)
    for r in range(height):
        for c in range(width):
            pixel = pixels_np[r, c]
            if pixel[3] > 0:  # Check alpha channel for transparency
                # Round to handle potential floating point artifacts from downscaling
                rgb = (
                    int(np.round(pixel[0])),
                    int(np.round(pixel[1])),
                    int(np.round(pixel[2])),
                )
                if rgb in color_to_id:
                    id_mask[r, c] = color_to_id[rgb]
    return jnp.asarray(id_mask)


class JaxRenderingUtils:
    def __init__(self, config: RendererConfig):
        self.config = config

    # ============= Non-jitted setup functions =============
    def loadFrame(self, fileName, transpose=False):
        """Loads a frame from .npy, ensuring output is RGBA (Height, Width, 4).

        Args:
            fileName: Path to the .npy file.
            transpose: If True, assumes source is (W, H, C) and transposes to (H, W, C).

        Returns:
            JAX array of shape (Height, Width, 4).
        """
        frame = jnp.load(fileName)
        if frame.ndim != 3 or frame.shape[2] != 4:
            raise ValueError(
                f"Invalid frame format in {fileName}. Source .npy must be loadable with 3 dims and 4 channels (RGBA)."
            )

        if transpose:
            # Source assumed W, H, C -> transpose to H, W, C
            frame = jnp.transpose(frame, (1, 0, 2))
        # Return the full RGBA frame. Do NOT slice with [:,:,:3]
        return frame

    def pad_to_match(self, sprites: List[jnp.ndarray]) -> Tuple[List[jnp.ndarray], List[jnp.ndarray]]:
        """
        Pads HWC sprites to a uniform dimension, aligning content to the top-left.
        The returned sprites are padded to the max dimensions, and the flip_offsets are the amount of padding to the left and top.
        This makes sure that the padding required by jax is not leading to incorrect flipping.

        Args:
            sprites: A list of JAX arrays (H, W, C).

        Returns:
            A tuple containing:
            - padded_sprites: A list of JAX arrays padded to max dimensions.
            - flip_offsets: A list of [dx, dy] arrays for correct flipping.
        """
        if not sprites:
            return [], []

        # For HWC sprites, shape[0] is height, shape[1] is width
        max_height = max(s.shape[0] for s in sprites)
        max_width = max(s.shape[1] for s in sprites)

        padded_sprites = []
        max_padding_x = 0
        max_padding_y = 0

        for sprite in sprites:
            pad_h = max_height - sprite.shape[0] # pad height (bottom)
            pad_w = max_width - sprite.shape[1]  # pad width (right)

            # Pad spec for HWC: ((pad_H_top, pad_H_bottom), (pad_W_left, pad_W_right), ...)
            pad_spec = ((0, pad_h), (0, pad_w), (0, 0))
            padded_sprite = jnp.pad(sprite, pad_spec, mode="constant", constant_values=0)

            max_padding_y = max(max_padding_y, pad_h)
            max_padding_x = max(max_padding_x, pad_w)

            padded_sprites.append(padded_sprite)

        flip_offsets = [jnp.array([max_padding_x, max_padding_y]) for _ in sprites]

        return padded_sprites, flip_offsets

    def load_and_pad_digits(self, path_pattern, num_chars=10):
        """Loads digit sprites, pads them to the max dimensions, assuming (H, W, C) format.

        Args:
            path_pattern: String pattern for digit filenames (e.g., "./digits/{}.npy").
            num_chars: Number of digits to load (e.g., 10 for 0-9).

        Returns:
            JAX array of shape (num_chars, max_Height, max_Width, 4).
        """
        digits = []
        max_height, max_width = 0, 0

        # Load digits assuming loadFrame returns (H, W, C)
        for i in range(num_chars):
            digit = self.loadFrame(path_pattern.format(i), transpose=False) # Ensure HWC
            max_height = max(max_height, digit.shape[0]) # Axis 0 is Height
            max_width = max(max_width, digit.shape[1])   # Axis 1 is Width
            digits.append(digit)

        # Pad digits to max dimensions (H, W)
        padded_digits = []
        for digit in digits:
            pad_h = max_height - digit.shape[0]
            pad_w = max_width - digit.shape[1]
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left

            # Padding order for HWC: ((pad_H_before, after), (pad_W_before, after), ...)
            padded_digit = jnp.pad(
                digit,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            padded_digits.append(padded_digit)

        return jnp.array(padded_digits)


    def setup_rendering_assets(
            self, loaded_sprites: Dict[str, jnp.ndarray]
        ) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray], jnp.ndarray, Dict[Tuple[int, int, int], int]]:
            """
            Analyzes all loaded RGBA sprites to generate a unified mask table (atlas),
            a metadata dictionary for sprite lookup, a color palette, and a pre-rendered background.

            This function performs the following steps:
            1.  Optionally downscales all sprites for performance or stylistic reasons.
            2.  Discovers all unique colors across all sprites to build a unified palette.
            3.  Converts the background sprite into a pre-rendered grid of palette IDs.
            4.  Converts all other sprites (single and batched) into palette ID masks.
            5.  Stacks all generated masks into a single, efficient vertical 'mask_table'.
            6.  Creates a 'sprite_info' dictionary that maps sprite names to their rendering
                metadata (location and size in the mask_table).
            """
            if "background" not in loaded_sprites:
                raise ValueError("A 'background' sprite must be provided in loaded_sprites.")

            ## 1. Downscale Sprites (Optional)
            # Downscaling is performed first to ensure colors blend correctly before palette extraction.
            sprites_to_use = {}
            if self.config.downscale:
                for name, sprite_data in loaded_sprites.items():
                    is_batched = sprite_data.ndim == 4
                    h_idx, w_idx = (1, 2) if is_batched else (0, 1)
                    new_h = int(sprite_data.shape[h_idx] * self.config.height_scaling)
                    new_w = int(sprite_data.shape[w_idx] * self.config.width_scaling)
                    
                    output_shape = (sprite_data.shape[0], new_h, new_w, 4) if is_batched else (new_h, new_w, 4)
                    
                    sprites_to_use[name] = jax.image.resize(
                        sprite_data, output_shape, method='linear'
                    )
            else:
                sprites_to_use = loaded_sprites

            ## 2. Generate Universal Color Palette
            color_to_id = {}
            palette_list = []
            next_id = 0

            for sprite_data in sprites_to_use.values():
                sprites_to_process = sprite_data if sprite_data.ndim == 4 else jnp.expand_dims(sprite_data, axis=0)
                for sprite in sprites_to_process:
                    pixels = np.array(sprite.reshape(-1, 4))
                    for r, g, b, a in pixels:
                        if a > 0:  # Only consider non-transparent pixels
                            rgb = (int(np.round(r)), int(np.round(g)), int(np.round(b)))
                            if rgb not in color_to_id:
                                color_to_id[rgb] = next_id
                                palette_list.append(rgb)
                                next_id += 1
            
            if self.config.channels == 1:
                # Handle grayscale conversion
                gray_palette = [int(0.299 * r + 0.587 * g + 0.114 * b) for r, g, b in palette_list]
                PALETTE = jnp.array(gray_palette, dtype=jnp.uint8).reshape(-1, 1)
            else:
                PALETTE = jnp.array(palette_list, dtype=jnp.uint8)

            ## 3. Pre-render Background
            bg_sprite = sprites_to_use["background"]
            PRE_RENDERED_BG = _create_id_mask(bg_sprite, color_to_id)

            ## 4. Generate Mask Table and Sprite Info
            all_masks = []
            sprite_info = {}
            v_cursor = 0

            # Sort items for deterministic order, which is good practice
            sorted_sprite_items = sorted(sprites_to_use.items(), key=lambda x: x[0])

            for name, sprite_data in sorted_sprite_items:
                if name == "background":
                    continue

                is_batched = sprite_data.ndim == 4
                sprites_to_process = sprite_data if is_batched else jnp.expand_dims(sprite_data, axis=0)

                info_list_for_sprite = []
                for sprite_frame in sprites_to_process:
                    id_mask = _create_id_mask(sprite_frame, color_to_id)
                    h, w = id_mask.shape
                    
                    # Store metadata: [start_row_in_atlas, width, height]
                    # This is everything needed to slice the mask from the table.
                    info_list_for_sprite.append([v_cursor, w, h])
                    
                    all_masks.append(id_mask)
                    v_cursor += h
                
                # Store info as a single array for one sprite or an array of arrays for a batch
                if is_batched:
                    sprite_info[name] = jnp.array(info_list_for_sprite, dtype=jnp.int32)
                else:
                    sprite_info[name] = jnp.array(info_list_for_sprite[0], dtype=jnp.int32)

            ## 5. Assemble the Final Mask Table (Atlas)
            if all_masks:
                max_width = max(mask.shape[1] for mask in all_masks) if all_masks else 0
                padded_masks = [
                    jnp.pad(m, ((0, 0), (0, max_width - m.shape[1])), mode='constant')
                    for m in all_masks
                ]
                mask_table = jnp.concatenate(padded_masks, axis=0)
            else:
                # Handle case with no sprites other than the background
                mask_table = jnp.array([[]], dtype=jnp.uint8)

            return PALETTE, mask_table, sprite_info, PRE_RENDERED_BG, color_to_id



    # ============= Jitted planning functions =============
    @partial(jax.jit, static_argnames=['max_sprites'])
    def create_initial_frame(self, max_sprites=32, sprite_info_table: jnp.ndarray | None = None):
        """Creates an initial, empty RenderPlan."""
        if sprite_info_table is None:
            sprite_info_table = jnp.zeros((0, 9), dtype=jnp.int32)
        return RenderPlan(
            commands=jnp.zeros((max_sprites, 9), dtype=jnp.int32), # TODO: do we need max sprites? Try out when using pallas
            command_count=jnp.array(0, dtype=jnp.int32),
            sprite_info_table=sprite_info_table,
        )


    @jax.jit
    def render_at(self, plan: RenderPlan, x, y, sprite_id, depth_z=1,
                    flip_h=False, flip_v=False):
        """Adds a single draw command to the RenderPlan."""
        # Get the index for the new command.
        command_index = plan.command_count

        # Look up the sprite's static info (atlas coords, size) using its ID.
        sprite_info = plan.sprite_info_table[sprite_id]
        atlas_u, atlas_v, width, height = sprite_info

        # Create the new command row.
        new_command = jnp.array([
            x, y,
            atlas_u, atlas_v,
            width, height,
            flip_h,
            flip_v,
            depth_z
        ])

        # Add the command to the commands array.
        updated_commands = plan.commands.at[command_index].set(new_command)

        # Return a new plan with the updated commands and incremented count.
        return plan._replace(
            commands=updated_commands,
            command_count=plan.command_count + 1
        )


    @jax.jit
    def render_label(self, raster, x, y, text_digits, char_sprites, spacing=15):
        """Renders a sequence of digits horizontally starting at (x, y)."""
        sprites = char_sprites[text_digits]
        def render_char(i, current_raster):
            char_x = x + i * spacing
            # Use a (0,0) pivot to maintain top-left rendering for each character
            return self.render_at(current_raster, char_x, y, sprites[i], flip_offset=jnp.array([0.0, 0.0]))

        raster = jax.lax.fori_loop(0, sprites.shape[0], render_char, raster)
        return raster


    @partial(jax.jit, static_argnames=["max_digits_to_render", "spacing"])
    def render_label_selective(self, plan: RenderPlan, x, y,
                        all_digits,
                        num_to_render,  # This can now be an array
                        char_sprite_ids,
                        start_index,
                        max_digits_to_render=2,
                        spacing=16,
                        depth_z=3):
        """Adds draw commands for a sequence of digits, supporting batching."""
        def plan_char(i, current_plan):
            command_index = current_plan.command_count
            digit_index_in_array = start_index + i

            # --- SIMPLIFIED AND CORRECTED LOGIC ---
            # This single line works for both batched (2D) and single (1D) `all_digits` arrays.
            # For a 2D array, it selects the column `digit_index_in_array` for all rows.
            # For a 1D array, it just selects the element at `digit_index_in_array`.
            digit_value = all_digits[..., digit_index_in_array]

            # The rest of the function remains the same, but now it receives correctly shaped data.
            should_draw_mask = (i < num_to_render)
            sprite_id_to_render = char_sprite_ids[digit_value]
            sprite_info = plan.sprite_info_table[sprite_id_to_render]

            render_x = x + i * spacing
            new_command = jnp.stack([
                render_x, y,
                sprite_info[..., 0], sprite_info[..., 1],
                sprite_info[..., 2], sprite_info[..., 3],
                jnp.zeros_like(command_index),
                jnp.zeros_like(command_index),
                jnp.full_like(command_index, depth_z)
            ], axis=-1)

            original_command = current_plan.commands[command_index]

            # Promote the mask for broadcasting if we are in a batched context
            where_mask = should_draw_mask
            if where_mask.ndim < new_command.ndim:
                where_mask = should_draw_mask[..., None]

            chosen_command = jnp.where(where_mask, new_command, original_command)
            updated_commands = current_plan.commands.at[command_index].set(chosen_command)
            updated_count = current_plan.command_count + should_draw_mask.astype(jnp.int32)

            return current_plan._replace(
                commands=updated_commands,
                command_count=updated_count
            )

        return jax.lax.fori_loop(0, max_digits_to_render, plan_char, plan)

    @partial(jax.jit, static_argnames=["max_digits"])
    def int_to_digits(self, n, max_digits=8):
        """
        Convert a non-negative integer or a batch of integers to a fixed-length
        JAX array of digits. Handles both scalar and batched inputs.
        """
        # This logic works whether 'n' is a scalar or a batched array.
        n = jnp.maximum(n, 0)
        max_val = 10**max_digits - 1
        n = jnp.minimum(n, max_val)

        def scan_body(carry, _):
            digit = carry % 10
            next_carry = carry // 10
            return next_carry, digit

        # lax.scan on a batched `n` produces a shape of (length, batch_size).
        # On a scalar `n`, it produces a shape of (length,).
        _, digits_reversed = lax.scan(scan_body, n, None, length=max_digits)

        # Flip to get digits in the correct order (most significant first).
        digits = jnp.flip(digits_reversed, axis=0)

        # Transpose the result.
        # If the input was a batch, this converts (length, batch) -> (batch, length).
        # If the input was a scalar, this is a no-op on the 1D array.
        return digits.T

    '''
    TODO: make these work with a planned renderer, right now they try to render into a raster
    @jax.jit
    def render_indicator(self, raster, x, y, value, sprite, spacing=15):
        """Renders 'value' copies of 'sprite' horizontally starting at (x, y)."""
        def render_single_indicator(i, current_raster):
            indicator_x = x + i * spacing
            # Use a (0,0) pivot for top-left rendering
            return self.render_at(current_raster, indicator_x, y, sprite, flip_offset=jnp.array([0.0, 0.0]))

        return jax.lax.fori_loop(0, value, render_single_indicator, raster)


    @partial(jax.jit, static_argnames=["width", "height"])
    def render_bar(self, raster, x, y, value, max_value, width, height, color, default_color):
        """Renders a horizontal progress bar at (x, y) with specified geometry."""
        color = jnp.asarray(color, dtype=jnp.uint8)
        default_color = jnp.asarray(default_color, dtype=jnp.uint8)

        fill_width = jnp.clip(jnp.nan_to_num((value / max_value) * width), 0, width).astype(jnp.int32)
        # Use 'xy' indexing for an (H, W) grid
        bar_xx, _ = jnp.meshgrid(jnp.arange(width), jnp.arange(height), indexing='xy')
        fill_mask = (bar_xx < fill_width)[..., None]

        bar_content = jnp.where(
            fill_mask,
            color,
            default_color
        )

        # Render the generated bar using a (0,0) pivot for top-left behavior
        raster = self.render_at(raster, x, y, bar_content, flip_offset=jnp.array([0.0, 0.0]))

        return raster
    '''

    @partial(jax.jit, static_argnames=["screen_height", "screen_width", "max_sprite_w", "max_sprite_h"])
    def execute_plan_vectorized(self, plan: RenderPlan, background: jnp.ndarray, atlas: jnp.ndarray):