from pathlib import Path, PureWindowsPath
import os
import numpy as np
import jax.numpy as jnp
import jax
from functools import partial
from jax import ShapeDtypeStruct, lax
from typing import List, Tuple
import jax.experimental.pallas as pallas

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


def loadFrame(fileName, transpose=False):
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


class RenderingManager:
    def __init__(self, mode: str = 'performance'):
        self.mode = mode
        if self.mode == 'performance':
            self.target_shape = (84, 84, 1)  # H, W, C for the final raster
            self.scale_y = 84 / 210
            self.scale_x = 84 / 160
            self.is_grayscale = True
        else:  # 'fidelity' mode
            self.target_shape = (210, 160, 3) # Final raster is RGB
            self.scale_y = 1.0
            self.scale_x = 1.0
            self.is_grayscale = False

    def _preprocess_sprite(self, sprite):
        """
        Helper to SCALE a sprite or batch of sprites to the correct dimensions
        for the current mode, preserving RGBA data.
        """
        is_batched = sprite.ndim == 4

        # Get original sprite dimensions
        if is_batched:
            # Assumes all sprites in a batch have the same H/W dimensions,
            # which is ensured by the padding functions.
            _, h, w, _ = sprite.shape
        else:
            h, w, _ = sprite.shape

        # Calculate SCALED target dimensions
        target_h = jnp.round(h * self.scale_y).astype(jnp.int32)
        target_w = jnp.round(w * self.scale_x).astype(jnp.int32)

        # Ensure dimensions are at least 1 pixel after scaling
        target_h = jnp.maximum(target_h, 1)
        target_w = jnp.maximum(target_w, 1)

        # The resize operation will be on RGBA data
        resize_shape_rgba = (target_h, target_w, 4)

        sprite_float = sprite.astype(jnp.float32)

        if is_batched:
            # Since the target shape is now computed from the input shape,
            # it remains consistent for all items in a padded batch.
            vmap_resize = jax.vmap(lambda s: jax.image.resize(s, resize_shape_rgba, 'bilinear'))
            resized_sprite_float = vmap_resize(sprite_float)
        else:
            resized_sprite_float = jax.image.resize(sprite_float, resize_shape_rgba, 'bilinear')

        # Clip and convert back to uint8 (this part was already correct)
        return jnp.round(jnp.clip(resized_sprite_float, 0, 255)).astype(jnp.uint8)


    @partial(jax.jit, static_argnames=("self",))
    def create_initial_frame(self):
        return jnp.zeros(self.target_shape, dtype=jnp.uint8)

    MAX_LABEL_WIDTH = 100
    MAX_LABEL_HEIGHT = 20


    @partial(jax.jit, static_argnums=(0,))
    def render_at(self, raster, x, y, sprite_frame,
                flip_horizontal=False,
                flip_vertical=False,
                flip_offset: jnp.ndarray = jnp.array([0, 0])):
        """
        Renders an RGBA sprite onto the raster using masking (ignores pixels with alpha=0).
        """
        # 1. Scale coordinates and handle flipping (same as before)
        scaled_y = jnp.round(y * self.scale_y).astype(jnp.int32)
        scaled_x = jnp.round(x * self.scale_x).astype(jnp.int32)

        processed_sprite = sprite_frame
        processed_sprite = lax.cond(
            flip_horizontal, lambda s: jnp.flip(s, axis=1), lambda s: s, processed_sprite
        )
        processed_sprite = lax.cond(
            flip_vertical, lambda s: jnp.flip(s, axis=0), lambda s: s, processed_sprite
        )

        sprite_h, sprite_w, _ = processed_sprite.shape

        # 2. Create a boolean mask from the sprite's alpha channel
        # The new dimension allows it to work with both single-channel and multi-channel rasters.
        opaque_mask = processed_sprite[..., 3:] > 128  # Shape becomes (H, W, 1)

        # 3. Prepare the sprite's color data, matching the raster's format (this is fine in a jitted context since is_grayscale is static)
        if self.is_grayscale:
            # Average the RGB channels for grayscale output
            sprite_color = jnp.mean(processed_sprite[..., :3], axis=-1, keepdims=True)
        else:
            # Use the RGB channels directly
            sprite_color = processed_sprite[..., :3]

        # 4. Get the corresponding slice from the background raster
        start_indices = (scaled_y, scaled_x, 0)
        background_slice = lax.dynamic_slice(raster, start_indices,
                                            (sprite_h, sprite_w, raster.shape[-1]))

        # 5. Use the mask to conditionally select pixels
        # Where the mask is True, use the sprite's color; otherwise, keep the background.
        output_slice = jnp.where(
            opaque_mask,                         # The condition
            sprite_color.astype(raster.dtype),   # Value if True (from sprite)
            background_slice                     # Value if False (from background)
        )

        # 6. Update the raster with the result
        return lax.dynamic_update_slice(
            raster,
            output_slice,
            start_indices
        )



    @partial(jax.jit, static_argnums=(0,))
    def draw_rect_fill(self, raster, x, y, width, height, color):
        """Draws a filled rectangle using a JIT-compatible masking approach."""
        # 1. Scale geometry
        x_start = jnp.round(x * self.scale_x).astype(jnp.int32)
        y_start = jnp.round(y * self.scale_y).astype(jnp.int32)
        x_end = jnp.round((x + width) * self.scale_x).astype(jnp.int32)
        y_end = jnp.round((y + height) * self.scale_y).astype(jnp.int32)

        # 2. Process color for the current mode
        color_arr = jnp.asarray(color, dtype=jnp.uint8)
        if self.is_grayscale:
            # Correctly shape the color for broadcasting: (1,) for a single channel
            processed_color = jnp.mean(color_arr).astype(jnp.uint8).reshape(1)
        else:
            processed_color = color_arr

        # 3. Create coordinate grids and mask for the rectangle area
        H, W, _ = self.target_shape
        yy, xx = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing='ij')
        mask = (xx >= x_start) & (xx < x_end) & (yy >= y_start) & (yy < y_end)
        
        # 4. Use jnp.where to apply the color, expanding mask to match channel dimension
        return jnp.where(mask[..., None], processed_color, raster)

    # The other rendering methods (render_label, render_indicator, etc.) do not need
    # changes as they correctly delegate the core drawing logic to render_at.
    @jax.jit
    def render_label(self, raster, x, y, text_digits, char_sprites, spacing=15):
        """Renders a sequence of digits horizontally starting at (x, y)."""
        sprites = char_sprites[text_digits]
        def render_char(i, current_raster):
            char_x = x + i * spacing
            return self.render_at(current_raster, char_x, y, sprites[i], flip_offset=jnp.array([0.0, 0.0]))

        raster = jax.lax.fori_loop(0, sprites.shape[0], render_char, raster)
        return raster
    
    @partial(jax.jit, static_argnums=(0,))
    def render_label_selective(self, raster, x, y,
                               all_digits,
                               char_sprites,
                               start_index,
                               num_to_render,
                               spacing=15):
        """Renders a label, automatically scaling position and spacing based on mode."""

        def render_char(i, current_raster):
            digit_index_in_array = start_index + i
            digit_value = all_digits[digit_index_in_array]
            sprite_to_render = char_sprites[digit_value]
            render_x = x + i * spacing
            return self.render_at(current_raster, render_x, y, sprite_to_render)

        return lax.fori_loop(0, num_to_render, render_char, raster)


    @jax.jit
    def render_indicator(self, raster, x, y, value, sprite, spacing=15):
        """Renders 'value' copies of 'sprite' horizontally starting at (x, y)."""
        def render_single_indicator(i, current_raster):
            indicator_x = x + i * spacing
            return self.render_at(current_raster, indicator_x, y, sprite, flip_offset=jnp.array([0.0, 0.0]))

        return jax.lax.fori_loop(0, value, render_single_indicator, raster)


    @partial(jax.jit, static_argnames=["width", "height"])
    def render_bar(self, raster, x, y, value, max_value, width, height, color, default_color):
        """Renders a horizontal progress bar at (x, y) with specified geometry."""
        color = jnp.asarray(color, dtype=jnp.uint8)
        default_color = jnp.asarray(default_color, dtype=jnp.uint8)
        
        fill_width = jnp.clip(jnp.nan_to_num((value / max_value) * width), 0, width).astype(jnp.int32)
        bar_xx, _ = jnp.meshgrid(jnp.arange(width), jnp.arange(height), indexing='xy')
        fill_mask = (bar_xx < fill_width)[..., None]
        
        bar_content = jnp.where(
            fill_mask,
            color,
            default_color
        )
        raster = self.render_at(raster, x, y, bar_content, flip_offset=jnp.array([0.0, 0.0]))

        return raster


def add_border(self, frame):
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
