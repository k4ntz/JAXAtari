from pathlib import Path, PureWindowsPath
import os
import numpy as np
import jax.numpy as jnp
import jax
from functools import partial
from jax import lax, ShapeDtypeStruct
from typing import Dict, List, NamedTuple, Tuple
import jax.experimental.pallas as pallas

BORDER = False

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


@partial(jax.jit, static_argnames=['max_sprites'])
def create_initial_frame(max_sprites=32, sprite_info_table: jnp.ndarray | None = None):
    """Creates an initial frame in HWC format (Height, Width, Channels).
    Arguments are still in the x,y order since this is the coordinate order the environments use internally.

    Args:
        width: Width of the frame (default: 160)
        height: Height of the frame (default: 210)
        channels: Number of color channels (default: 3 for RGB)

    Returns:
        JAX array of shape (height, width, channels) filled with zeros.
    """
    if sprite_info_table is None:
        sprite_info_table = jnp.zeros((0, 4), dtype=jnp.int32)
    return RenderPlan(
        commands=jnp.zeros((max_sprites, 9), dtype=jnp.int32),
        command_count=jnp.array(0, dtype=jnp.int32),
        sprite_info_table=sprite_info_table,
    )


def add_border(frame):
    if frame.shape[:2] == (210, 160):
        return frame  # No border for background
    h, w, c = frame.shape
    flat_frame = frame.reshape(h*w, 4)  # Ensure the last dimension is RGBA
    border_color = jnp.array([255, 255, 255, 50], dtype=jnp.uint8) # set alpha to 50 transparency
    # Top and bottom borders
    frame = frame.at[0, :, :].set(border_color)
    frame = frame.at[-1, :, :].set(border_color)

    # Left and right borders
    frame = frame.at[:, 0, :].set(border_color)
    frame = frame.at[:, -1, :].set(border_color)
    return frame


def loadFrame(fileName, transpose=False):
    """Loads a frame from .npy, ensuring output is (Height, Width, Channels).

    Args:
        fileName: Path to the .npy file.
        transpose: If True, assumes source is (W, H, C) and transposes
                   to (H, W, C). If False (default), assumes source is already (H, W, C).

    Returns:
        JAX array of shape (Height, Width, 4).
    """
    frame_np = np.load(fileName)
    if frame_np.ndim != 3 or frame_np.shape[2] != 4:
         raise ValueError(
            f"Invalid frame format in {fileName}. Source .npy must be loadable with 3 dims and 4 channels."
        )

    if transpose:
        # Source assumed W, H, C -> transpose to H, W, C
        frame_np = np.transpose(frame_np, (1, 0, 2))
    return jnp.asarray(frame_np)


def load_and_pad_digits(path_pattern, num_chars=10):
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
        digit = loadFrame(path_pattern.format(i), transpose=False) # Ensure HWC
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


@jax.jit
def get_sprite_frame(frames, frame_idx, loop=True):
    """
    TODO: testing
    for now, just return the ID we get in + the frame_idx
    """
    return frames + frame_idx


@jax.jit
def render_at(plan: RenderPlan, x, y, sprite_id, depth_z=1,
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


MAX_LABEL_WIDTH = 100
MAX_LABEL_HEIGHT = 20


@jax.jit
def render_label(raster, x, y, text_digits, char_sprites, spacing=15):
    """Renders a sequence of digits horizontally starting at (x, y)."""
    sprites = char_sprites[text_digits]
    def render_char(i, current_raster):
        char_x = x + i * spacing
        # Use a (0,0) pivot to maintain top-left rendering for each character
        return render_at(current_raster, char_x, y, sprites[i], flip_offset=jnp.array([0.0, 0.0]))

    raster = jax.lax.fori_loop(0, sprites.shape[0], render_char, raster)
    return raster


@partial(jax.jit, static_argnames=["max_digits_to_render", "spacing"])
def render_label_selective(plan: RenderPlan, x, y,
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


@jax.jit
def render_indicator(raster, x, y, value, sprite, spacing=15):
    """Renders 'value' copies of 'sprite' horizontally starting at (x, y)."""
    def render_single_indicator(i, current_raster):
        indicator_x = x + i * spacing
        # Use a (0,0) pivot for top-left rendering
        return render_at(current_raster, indicator_x, y, sprite, flip_offset=jnp.array([0.0, 0.0]))

    return jax.lax.fori_loop(0, value, render_single_indicator, raster)


@partial(jax.jit, static_argnames=["width", "height"])
def render_bar(raster, x, y, value, max_value, width, height, color, default_color):
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
    raster = render_at(raster, x, y, bar_content, flip_offset=jnp.array([0.0, 0.0]))

    return raster


def _find_content_bbox_np(sprite_frame: np.ndarray) -> tuple[int, int, int, int]:
    """Finds the bounding box of non-transparent content in an HWC NumPy array."""
    alpha_channel = np.asarray(sprite_frame[:, :, 3])
    if np.all(alpha_channel == 0):
        return 0, 0, 0, 0
    # For HWC, where returns (rows, cols) which are (y, x)
    rows, cols = np.where(alpha_channel > 0)
    min_x, max_x = np.min(cols), np.max(cols)
    min_y, max_y = np.min(rows), np.max(rows)
    return int(min_x), int(min_y), int(max_x), int(max_y)

def pad_to_match(sprites: List[jnp.ndarray]) -> Tuple[List[jnp.ndarray], List[jnp.ndarray]]:
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

        if BORDER:
            padded_sprite = add_border(padded_sprite)

        padded_sprites.append(padded_sprite)

    flip_offsets = [jnp.array([max_padding_x, max_padding_y]) for _ in sprites]

    return padded_sprites, flip_offsets


@partial(jax.jit, static_argnames=["max_digits"])
def int_to_digits(n, max_digits=8):
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


@partial(jax.jit, static_argnames=["screen_height", "screen_width", "max_sprite_w", "max_sprite_h"])
def execute_plan_vectorized(plan: RenderPlan, background: jnp.ndarray, atlas: jnp.ndarray, 
                           screen_height: int, screen_width: int, 
                           max_sprite_w: int, max_sprite_h: int):
    """
    Executes a render plan using a fully static, JIT-compatible scatter
    operation with a segment_max z-buffer. This is the definitive version.
    """
    # 1. Generate pixel data for ALL potential sprite locations (same as before)
    all_commands = plan.commands
    command_indices = jnp.arange(all_commands.shape[0])
    max_lx, max_ly = jnp.meshgrid(jnp.arange(max_sprite_w), jnp.arange(max_sprite_h))

    def get_sprite_pixels(command, index):
        is_active = index < plan.command_count
        sx, sy, u, v, w, h, fh, fv, z = command

        # Create a mask for the actual size of the current sprite
        valid_pixel_mask = (max_lx < w) & (max_ly < h)

        # --- KEY CHANGE: Use `where` instead of boolean indexing ---
        # Calculate atlas coordinates for the full grid, but use (0,0) for invalid pixels.
        # These "garbage" coordinates are fine because we'll mask them out later.
        atlas_x_base = jnp.where(fh == 1, u + (w - 1 - max_lx), u + max_lx)
        atlas_y_base = jnp.where(fv == 1, v + (h - 1 - max_ly), v + max_ly)
        
        atlas_x = jnp.where(valid_pixel_mask, atlas_x_base, 0)
        atlas_y = jnp.where(valid_pixel_mask, atlas_y_base, 0)
        
        # Gather pixels for the entire max_size grid
        pixels_rgba = atlas[atlas_y, atlas_x]
        
        # Calculate screen coordinates for the full grid
        screen_x = sx + max_lx
        screen_y = sy + max_ly
        
        # The final mask includes activity, valid pixel area, and transparency
        final_mask = is_active & valid_pixel_mask & (pixels_rgba[..., 3] > 0)
        
        # All returned arrays now have a fixed size (max_sprite_w * max_sprite_h)
        return (
            screen_x.flatten(),
            screen_y.flatten(),
            pixels_rgba[..., :3].reshape(-1, 3), # Flattened RGB
            jnp.full(max_sprite_w * max_sprite_h, z).flatten(), # Flattened Z
            final_mask.flatten() # Flattened final validity mask
        )

    all_x, all_y, all_rgb, all_z, final_mask = jax.vmap(get_sprite_pixels)(all_commands, command_indices)
    
    # 2. Flatten all data into 1D arrays. NO boolean indexing is used.
    all_x = all_x.flatten()
    all_y = all_y.flatten()
    all_rgb = all_rgb.reshape(-1, 3)
    all_z = all_z.flatten()
    final_mask = final_mask.flatten()

    # 3. Perform the parallel Z-buffer using segment_max
    # Set the z-depth of invalid pixels to -1 so they can't win
    effective_z = jnp.where(final_mask, all_z, -1.0)
    
    # Create a unique ID for each screen pixel location
    linear_indices = all_y * screen_width + all_x
    
    # We need the INDEX of the max Z value for each segment. JAX doesn't have
    # segment_argmax, so we use this trick: we combine the z-value and the
    # pixel's own index into a single payload. The max of this payload will be
    # from the pixel with the highest Z.
    pixel_indices = jnp.arange(all_x.shape[0])
    payload = (effective_z.astype(jnp.float32) * all_x.shape[0] + pixel_indices).astype(jnp.int32)
    
    # Find the winning payload for each screen pixel location
    # `num_segments` is the total number of pixels on the screen.
    winning_payloads = jax.ops.segment_max(payload, linear_indices, num_segments=screen_height * screen_width)
    
    # Extract the original index of the winning pixels from the payload
    winning_pixel_indices = winning_payloads % all_x.shape[0]

    # A segment is active if its max z-depth was greater than -1
    active_segment_mask = (winning_payloads >= 0).reshape(screen_height, screen_width)
    
    # 4. Gather the final colors and scatter them onto the canvas
    # The winning indices point to the correct colors in our original flat list
    winning_colors = all_rgb[winning_pixel_indices].reshape(screen_height, screen_width, 3)
    
    canvas = background[..., :3]
    final_canvas = jnp.where(
        active_segment_mask[..., None], # Broadcast mask over RGB channels
        winning_colors,
        canvas
    )
    
    return final_canvas.astype(jnp.uint8)