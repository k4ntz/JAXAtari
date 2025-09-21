from pathlib import Path, PureWindowsPath
import os
import numpy as np
import jax.numpy as jnp
import jax
from functools import partial
from jax import lax
from typing import List, Tuple

BORDER = False

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


@partial(jax.jit, static_argnames=["width", "height", "channels"])
def create_initial_frame(width=160, height=210, channels=3):
    """Creates an initial frame in HWC format (Height, Width, Channels).
    Arguments are still in the x,y order since this is the coordinate order the environments use internally.
    
    Args:
        width: Width of the frame (default: 160)
        height: Height of the frame (default: 210) 
        channels: Number of color channels (default: 3 for RGB)
        
    Returns:
        JAX array of shape (height, width, channels) filled with zeros.
    """
    # uses HWC same as ALE
    return jnp.zeros((height, width, channels), dtype=jnp.uint8)


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
    frame = jnp.load(fileName)
    if frame.ndim != 3 or frame.shape[2] != 4:
         raise ValueError(
            f"Invalid frame format in {fileName}. Source .npy must be loadable with 3 dims and 4 channels."
        )

    if transpose:
        # Source assumed W, H, C -> transpose to H, W, C
        frame = jnp.transpose(frame, (1, 0, 2))
    return frame


@partial(jax.jit, static_argnames=["path_pattern", "num_chars"])
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
    """Extracts a single sprite frame from an animation sequence.

    Args:
        frames: JAX array of shape (NumFrames, Height, Width, Channels).
        frame_idx: Index of the frame to retrieve.
        loop: If True, frame_idx wraps around using modulo.

    Returns:
        JAX array of shape (Height, Width, Channels) for the selected frame.
    """
    num_frames = frames.shape[0]
    frame_idx_looped = jnp.mod(frame_idx, num_frames)
    frame_idx_converted = lax.cond(loop, lambda: frame_idx_looped, lambda: frame_idx)
    valid_frame = (frame_idx_converted >= 0) & (frame_idx_converted < num_frames)

    # Get dimensions from input array shape (N, H, W, C)
    frame_height = frames.shape[1]   # Axis 1 is Height
    frame_width = frames.shape[2]    # Axis 2 is Width
    frame_channels = frames.shape[3] # Axis 3 is Channels
    blank_frame = jnp.zeros((frame_height, frame_width, frame_channels), dtype=frames.dtype)

    return lax.cond(
        valid_frame,
        lambda: frames[frame_idx_converted],
        lambda: blank_frame,
    )

@jax.jit
def render_at(raster, x, y, sprite_frame, 
              flip_horizontal=False, 
              flip_vertical=False, 
              flip_offset: jnp.ndarray = jnp.array([0, 0])):
    """
    Renders a sprite, when using padded sprites can correct displacing flipping logic using the flip offsets.

    Args:
        raster: JAX array (H, W, C) for the target image.
        x: World x-coordinate for the top-left of the sprite's content.
        y: World y-coordinate for the top-left of the sprite's content.
        sprite_frame: JAX array (H, W, 4) with sprite data.
        flip_horizontal: Boolean flag to flip the sprite horizontally.
        flip_vertical: Boolean flag to flip the sprite vertically.
        flip_offset: A [dx, dy] array (width, height padding) for flip correction.
    """
    # --- Setup ---
    x, y = jnp.asarray(x, dtype=jnp.int32), jnp.asarray(y, dtype=jnp.int32)
    raster_height, raster_width, _ = raster.shape
    sprite_height, sprite_width, _ = sprite_frame.shape

    # --- Position Calculation with Flip Correction ---
    top_left_x, top_left_y = x, y

    # If flipping horizontally, shift the drawing position left by the offset
    # to compensate for the padding that is now on the left side.
    top_left_x = jax.lax.cond(
        flip_horizontal,
        lambda: (x - flip_offset[0]).astype(jnp.int32),
        lambda: x.astype(jnp.int32)
    )
    # Apply the same logic for vertical flipping.
    top_left_y = jax.lax.cond(
        flip_vertical,
        lambda: (y - flip_offset[1]).astype(jnp.int32),
        lambda: y.astype(jnp.int32)
    )

    # --- Sprite Flipping ---
    sprite = jax.lax.cond(flip_horizontal, lambda s: jnp.flip(s, axis=1), lambda s: s, sprite_frame) # Axis 1 is Width
    sprite = jax.lax.cond(flip_vertical,   lambda s: jnp.flip(s, axis=0), lambda s: s, sprite)          # Axis 0 is Height

    # --- Blending Logic ---
    # Use 'xy' indexing for HWC to get grids of shape (H, W)
    raster_xx, raster_yy = jnp.meshgrid(jnp.arange(raster_width), jnp.arange(raster_height), indexing='xy')
    
    sprite_coord_x = raster_xx - top_left_x
    sprite_coord_y = raster_yy - top_left_y
    sprite_bounds_mask = (sprite_coord_x >= 0) & (sprite_coord_x < sprite_width) & \
                         (sprite_coord_y >= 0) & (sprite_coord_y < sprite_height)
    
    pad_width_spec = ((1, 1), (1, 1), (0, 0)) # Pad H and W axes
    sprite_padded = jnp.pad(sprite, pad_width_spec, mode='constant', constant_values=0)
    
    sprite_coord_x_padded = sprite_coord_x + 1
    sprite_coord_y_padded = sprite_coord_y + 1
    
    # Indexing for HWC is [y, x]
    gathered_sprite_rgba = sprite_padded[sprite_coord_y_padded, sprite_coord_x_padded]
    
    gathered_sprite_rgb = gathered_sprite_rgba[..., :3].astype(jnp.float32)
    gathered_sprite_alpha = (gathered_sprite_rgba[..., 3:].astype(jnp.float32) / 255.0)
    current_raster_rgb = raster[..., :raster.shape[2]].astype(jnp.float32)
    blended_rgb = gathered_sprite_rgb * gathered_sprite_alpha + current_raster_rgb * (1.0 - gathered_sprite_alpha)
    final_mask_broadcasted = sprite_bounds_mask[..., None]
    new_raster_float = jnp.where(final_mask_broadcasted, blended_rgb, current_raster_rgb)
    
    return new_raster_float.astype(raster.dtype)

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


@jax.jit
def render_label_selective(raster, x, y,
                           all_digits,
                           char_sprites,
                           start_index,
                           num_to_render,
                           spacing=15):
    """Renders a specified number of digits from a digit array at (x, y)."""
    def render_char(i, current_raster):
        digit_index_in_array = start_index + i
        digit_value = all_digits[digit_index_in_array]
        sprite_to_render = char_sprites[digit_value]
        render_x = x + i * spacing
        # Use a (0,0) pivot for top-left rendering
        return render_at(current_raster, render_x, y, sprite_to_render, flip_offset=jnp.array([0.0, 0.0]))

    raster = jax.lax.fori_loop(0, num_to_render, render_char, raster)
    return raster


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
    """Convert a non-negative integer to a fixed-length JAX array of digits (most significant first).

    Args:
        n: The integer to convert.
        max_digits: The fixed number of digits in the output array.

    Returns:
        A 1D JAX array of length `max_digits`.
    """
    # Ensure n is non-negative
    n = jnp.maximum(n, 0)
    # Clip n to the maximum value representable by max_digits
    max_val = 10**max_digits - 1
    n = jnp.minimum(n, max_val)

    # Use lax.scan to extract digits efficiently
    def scan_body(carry, _):
        current_n = carry
        digit = current_n % 10
        next_n = current_n // 10
        # Return next carry and the extracted digit
        return next_n, digit

    # Initial carry is the number itself
    # Scan over a dummy array of the correct length
    _, digits_reversed = lax.scan(scan_body, n, None, length=max_digits)

    # Digits are generated least significant first, flip them
    return jnp.flip(digits_reversed)
