from pathlib import Path, PureWindowsPath
import os
import numpy as np
import jax.numpy as jnp
import jax
from functools import partial
import pygame
from jax import lax


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


def loadFrame(fileName, transpose=True):
    """Loads a frame from .npy, ensuring output is (Width, Height, Channels).

    Args:
        fileName: Path to the .npy file.
        transpose: If True (default), assumes source is (H, W, C) and transposes
                   to (W, H, C). If False, assumes source is already (W, H, C).

    Returns:
        JAX array of shape (Width, Height, 4).
    """
    frame = jnp.load(fileName)
    if frame.ndim != 3 or frame.shape[2] != 4:
         raise ValueError(
            f"Invalid frame format in {fileName}. Source .npy must be loadable with 3 dims and 4 channels."
        )

    if transpose:
        # Source assumed H, W, C -> transpose to W, H, C
        return jnp.transpose(frame, (1, 0, 2))
    else:
         # Source assumed W, H, C
        return frame


@partial(jax.jit, static_argnames=["path_pattern", "num_chars"])
def load_and_pad_digits(path_pattern, num_chars=10):
    """Loads digit sprites, pads them to the max dimensions, assuming (W, H, C) format.

    Args:
        path_pattern: String pattern for digit filenames (e.g., "./digits/{}.npy").
        num_chars: Number of digits to load (e.g., 10 for 0-9).

    Returns:
        JAX array of shape (num_chars, max_Width, max_Height, 4).
    """
    digits = []
    max_width, max_height = 0, 0

    # Load digits assuming loadFrame returns (W, H, C)
    for i in range(num_chars):
        # Load with transpose=True (default) assuming source is H, W, C
        digit = loadFrame(path_pattern.format(i))
        max_width = max(max_width, digit.shape[0])   # Axis 0 is Width
        max_height = max(max_height, digit.shape[1]) # Axis 1 is Height
        digits.append(digit)

    # Pad digits to max dimensions (W, H)
    padded_digits = []
    for digit in digits:
        pad_w = max_width - digit.shape[0]  # Pad width (axis 0)
        pad_h = max_height - digit.shape[1] # Pad height (axis 1)
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        # Padding order: ((pad_axis0_before, after), (pad_axis1_before, after), ...)
        padded_digit = jnp.pad(
            digit,
            ((pad_left, pad_right), (pad_top, pad_bottom), (0, 0)), # Pad Width (axis 0), then Height (axis 1)
            mode="constant",
            constant_values=0,
        )
        padded_digits.append(padded_digit)

    return jnp.array(padded_digits)


@jax.jit
def get_sprite_frame(frames, frame_idx, loop=True):
    """Extracts a single sprite frame from an animation sequence.

    Args:
        frames: JAX array of shape (NumFrames, Width, Height, Channels).
        frame_idx: Index of the frame to retrieve.
        loop: If True, frame_idx wraps around using modulo.

    Returns:
        JAX array of shape (Width, Height, Channels) for the selected frame,
        or a blank frame if index is invalid and loop is False.
    """
    num_frames = frames.shape[0]

    frame_idx_looped = jnp.mod(frame_idx, num_frames)
    frame_idx_converted = jax.lax.cond(
        loop, lambda _: frame_idx_looped, lambda _: frame_idx, operand=None
    )

    valid_frame = jnp.logical_and(
        frame_idx_converted >= 0, frame_idx_converted < num_frames
    )

    # Get dimensions from input array shape (N, W, H, C)
    frame_width = frames.shape[1]    # Axis 1 is Width
    frame_height = frames.shape[2]   # Axis 2 is Height
    frame_channels = frames.shape[3] # Axis 3 is Channels
    blank_frame = jnp.zeros(
        (frame_width, frame_height, frame_channels), dtype=frames.dtype # W, H, C
    )

    return jax.lax.cond(
        valid_frame,
        lambda _: frames[frame_idx_converted], # Shape (W, H, C)
        lambda _: blank_frame,                 # Shape (W, H, C)
        operand=None,
    )


@jax.jit
def render_at(raster, x, y, sprite_frame, flip_horizontal=False, flip_vertical=False):
    """Renders a sprite onto a raster at position (x, y) top-left, with clipping and optional flipping.

    Args:
        raster: JAX array of shape (Width, Height, 3/4) for the target image.
        x: Integer x coordinate (left edge, horizontal) for sprite placement.
        y: Integer y coordinate (top edge, vertical) for sprite placement.
        sprite_frame: JAX array of shape (Width, Height, 4) containing RGB + alpha.
        flip_horizontal: Boolean flag to flip the sprite horizontally (left-right).
        flip_vertical: Boolean flag to flip the sprite vertically (top-bottom).

    Returns:
        A new raster JAX array (Width, Height, 3/4) with the sprite rendered.
    """
    # --- Input Validation and Setup ---
    x, y = jnp.asarray(x, dtype=jnp.int32), jnp.asarray(y, dtype=jnp.int32)
    # Arrays are (Width, Height, Channels)
    sprite_frame = jnp.asarray(sprite_frame) # Assume concrete shape (W, H, 4)
    raster = jnp.asarray(raster)             # Assume shape (W, H, 3 or 4)
    raster_width, raster_height, raster_channels = raster.shape
    sprite_width, sprite_height, _ = sprite_frame.shape # Need concrete shape here

    # --- Sprite Flipping ---
    sprite = sprite_frame
    # Flip horizontal means flipping along the Width axis (Axis 0)
    sprite = jax.lax.cond(flip_horizontal, lambda s: jnp.flip(s, axis=0), lambda s: s, sprite)
    # Flip vertical means flipping along the Height axis (Axis 1)
    sprite = jax.lax.cond(flip_vertical,   lambda s: jnp.flip(s, axis=1), lambda s: s, sprite)

    # --- Coordinate Calculation & Masking ---
    # Create coordinate grids for the *entire* raster (W, H)
    # 'ij' indexing: xx varies along axis 0 (Width), yy varies along axis 1 (Height)
    raster_xx, raster_yy = jnp.meshgrid(
        jnp.arange(raster_width),   # X coords (0..W-1)
        jnp.arange(raster_height),  # Y coords (0..H-1)
        indexing='ij'
    ) # raster_xx shape (W, H), raster_yy shape (W, H)

    # Calculate corresponding coordinates relative to the sprite's origin (top-left)
    sprite_coord_x = raster_xx - x # X position on sprite for each raster pixel
    sprite_coord_y = raster_yy - y # Y position on sprite for each raster pixel

    # Create mask: identifies raster pixels (at raster_xx, raster_yy) that
    # correspond to valid coordinates *within* the sprite's bounds (0..W-1, 0..H-1)
    sprite_bounds_mask = (sprite_coord_x >= 0) & (sprite_coord_x < sprite_width) & \
                         (sprite_coord_y >= 0) & (sprite_coord_y < sprite_height)
    # sprite_bounds_mask has shape (W, H)

    # --- Safe Gathering using Padding ---
    # Pad the sprite (W, H, C) to handle potential out-of-bounds access.
    pad_width_spec = ((1, 1), (1, 1), (0, 0)) # Pad W (axis 0), then H (axis 1)
    sprite_padded = jnp.pad(sprite, pad_width_spec, mode='constant', constant_values=0)
    # sprite_padded has shape (W+2, H+2, 4)

    # Adjust coordinates to index into the *padded* sprite
    sprite_coord_x_padded = sprite_coord_x + 1 # X index for padded sprite
    sprite_coord_y_padded = sprite_coord_y + 1 # Y index for padded sprite

    # Gather RGBA values from the padded sprite using calculated indices.
    # Indexing for (W, H, C) array is [x_index, y_index]
    gathered_sprite_rgba = sprite_padded[sprite_coord_x_padded, sprite_coord_y_padded]
    # gathered_sprite_rgba has shape (W, H, 4)

    # --- Blending Calculation (for all raster pixels) ---
    gathered_sprite_rgb = gathered_sprite_rgba[..., :3].astype(jnp.float32)
    gathered_sprite_alpha = (gathered_sprite_rgba[..., 3:].astype(jnp.float32) / 255.0) # Shape (W, H, 1)

    # Get current raster RGB (shape W, H, C)
    current_raster_rgb = raster[..., :raster_channels].astype(jnp.float32)

    # Perform alpha blending calculation everywhere
    blended_rgb = gathered_sprite_rgb * gathered_sprite_alpha + \
                  current_raster_rgb * (1.0 - gathered_sprite_alpha)
    # blended_rgb has shape (W, H, C)

    # --- Apply Mask with jnp.where ---
    final_mask_broadcasted = sprite_bounds_mask[..., None] # Shape (W, H, 1)

    # Where the mask is True (pixel corresponds to valid sprite area), select blended_rgb.
    # Where the mask is False (pixel outside sprite area), select the original current_raster_rgb.
    new_raster_float = jnp.where(
        final_mask_broadcasted, # Condition (W, H, 1)
        blended_rgb,            # Value if True (W, H, C)
        current_raster_rgb      # Value if False (W, H, C)
    )

    # Cast final result back to original raster dtype
    new_raster = new_raster_float.astype(raster.dtype) # Shape (W, H, C)

    return new_raster


def update_pygame(pygame_screen, raster, SCALING_FACTOR=3, WIDTH=400, HEIGHT=300):
    """Updates the Pygame display with the rendered raster.

    Args:
        pygame_screen: The Pygame screen surface.
        raster: JAX array of shape (Width, Height, 3/4) containing the image data.
        SCALING_FACTOR: Factor to scale the raster for display.
        WIDTH: Expected width of the input raster (used for scaling calculation).
        HEIGHT: Expected height of the input raster (used for scaling calculation).
    """
    pygame_screen.fill((0, 0, 0))

    # Convert JAX array (W, H, C) to NumPy (W, H, C)
    raster_np = np.array(raster)
    raster_np = raster_np.astype(np.uint8)

    # Pygame surface needs (W, H). make_surface expects (W, H, C) correctly.
    frame_surface = pygame.surfarray.make_surface(raster_np)

    # Pygame scale expects target (width, height)
    target_width_px = int(WIDTH * SCALING_FACTOR)
    target_height_px = int(HEIGHT * SCALING_FACTOR)
    # Optional: Adjust scaling if raster size differs from constants
    if raster_np.shape[0] != WIDTH or raster_np.shape[1] != HEIGHT:
        target_width_px = int(raster_np.shape[0] * SCALING_FACTOR)
        target_height_px = int(raster_np.shape[1] * SCALING_FACTOR)


    frame_surface_scaled = pygame.transform.scale(
        frame_surface, (target_width_px, target_height_px)
    )

    pygame_screen.blit(frame_surface_scaled, (0, 0))
    pygame.display.flip()


MAX_LABEL_WIDTH = 100
MAX_LABEL_HEIGHT = 20


@jax.jit
def render_label(raster, x, y, text_digits, char_sprites, spacing=15):
    """Renders a sequence of digits horizontally starting at (x, y).

    Args:
        raster: Target raster (W, H, C).
        x: Left x coordinate for the start of the text.
        y: Top y coordinate for the text.
        text_digits: 1D JAX array of integer digits to render.
        char_sprites: JAX array of sprites (NumChars, W, H, C).
        spacing: Horizontal spacing between character origins.

    Returns:
        Updated raster.
    """
    # Assumes char_sprites is (NumChars, W, H, C)
    sprites = char_sprites[text_digits] # Fetches sprites -> shape (NumDigits, W, H, C)

    def render_char(i, current_raster):
        char_x = x + i * spacing # Calculate x position for the i-th character
        # Render the i-th sprite at (char_x, y)
        return render_at(current_raster, char_x, y, sprites[i])

    raster = jax.lax.fori_loop(0, sprites.shape[0], render_char, raster)
    return raster


@jax.jit
def render_label_selective(raster, x, y,
                           all_digits,    # JAX array (e.g., length 2 or more)
                           char_sprites,  # (10, W, H, C)
                           start_index,   # Concrete integer (0 or 1 usually)
                           num_to_render, # Concrete integer (1 or 2 usually)
                           spacing=15):
    """Renders a specified number of digits from a digit array at (x, y).

    Args:
        raster: Target raster (W, H, C).
        x: Left x coordinate for the *first rendered digit*.
        y: Top y coordinate.
        all_digits: JAX array containing all potential digits.
        char_sprites: JAX array of sprite frames for each digit (0-9).
        start_index: The index within `all_digits` to start rendering from.
        num_to_render: How many digits to render sequentially from `start_index`.
        spacing: Horizontal space between digits.

    Returns:
        Updated raster.
    """
    def render_char(i, current_raster):
        # i is the loop index (0 up to num_to_render-1)
        digit_index_in_array = start_index + i
        digit_value = all_digits[digit_index_in_array]
        sprite_to_render = char_sprites[digit_value] # Gets (W, H, C) sprite
        render_x = x + i * spacing # Calculate x position based on loop index
        return render_at(current_raster, render_x, y, sprite_to_render)

    raster = jax.lax.fori_loop(0, num_to_render, render_char, raster)
    return raster


@jax.jit
def render_indicator(raster, x, y, value, sprite, spacing=15):
    """Renders 'value' copies of 'sprite' horizontally starting at (x, y).

    Args:
        raster: Target raster (W, H, C).
        x: Left x coordinate for the first indicator.
        y: Top y coordinate for the indicators.
        value: Number of times to render the sprite.
        sprite: The sprite to render (W, H, C).
        spacing: Horizontal spacing between sprite origins.

    Returns:
        Updated raster.
    """
    # Assumes sprite is (W, H, C)
    def render_single_indicator(i, current_raster):
        indicator_x = x + i * spacing # Calculate x for this instance
        return render_at(current_raster, indicator_x, y, sprite)

    return jax.lax.fori_loop(0, value, render_single_indicator, raster)


@partial(jax.jit, static_argnames=["width", "height"])
def render_bar(raster, x, y, value, max_value, width, height, color, default_color):
    """Renders a horizontal progress bar at (x, y) with specified geometry.

    Args:
        raster: Target raster (W, H, C).
        x: Left x coordinate of the bar.
        y: Top y coordinate of the bar.
        value: Current value of the bar.
        max_value: Maximum value for the bar.
        width: Geometric width of the bar in pixels.
        height: Geometric height of the bar in pixels.
        color: RGBA tuple/list/array for the filled portion.
        default_color: RGBA tuple/list/array for the unfilled portion.

    Returns:
        Updated raster.
    """
    color = jnp.asarray(color, dtype=jnp.uint8) # Use uint8 for direct use
    default_color = jnp.asarray(default_color, dtype=jnp.uint8)
    if color.shape[0] != 4 or default_color.shape[0] != 4:
        raise ValueError("Color and default_color must be RGBA")

    # Create the bar shape directly as (Width, Height, 4)
    bar_shape = (width, height, 4)

    # Compute the filled portion width (along axis 0)
    fill_width = jnp.clip(jnp.nan_to_num((value / max_value) * width), 0, width).astype(jnp.int32)

    # Create coordinate grids for the bar itself (W, H)
    bar_xx, bar_yy = jnp.meshgrid(jnp.arange(width), jnp.arange(height), indexing='ij')

    # Create a mask for the filled portion
    fill_mask = (bar_xx < fill_width)[..., None] # Shape (W, H, 1)

    # Use jnp.where to create the bar content (W, H, 4) directly as uint8
    bar_content = jnp.where(
        fill_mask,      # Condition
        color,          # Value if True (broadcasts to (W, H, 4))
        default_color   # Value if False (broadcasts)
    )

    # Render the generated bar (W, H, 4) onto the raster at (x, y)
    raster = render_at(raster, x, y, bar_content)

    return raster


@jax.jit
def pad_to_match(sprites):
    """Pads a list of sprites to the maximum dimensions found in the list.

    Args:
        sprites: A list of JAX arrays, each assumed shape (W, H, C).

    Returns:
        A list of JAX arrays, all padded to (maxW, maxH, C).
    """
    max_width = 0
    max_height = 0
    for sprite in sprites:
        max_width = max(max_width, sprite.shape[0])  # Axis 0 is Width
        max_height = max(max_height, sprite.shape[1]) # Axis 1 is Height

    padded_sprites = []
    for sprite in sprites:
        pad_w = max_width - sprite.shape[0]
        pad_h = max_height - sprite.shape[1]
        # Padding spec: ((pad_axis0_before, after), (pad_axis1_before, after), ...)
        pad_spec = ((0, pad_w), (0, pad_h), (0, 0)) # Pad Width (axis 0), then Height (axis 1)
        padded_sprite = jnp.pad(
            sprite,
            pad_spec,
            mode="constant",
            constant_values=0,
        )
        padded_sprites.append(padded_sprite)

    return padded_sprites

@jax.jit
def pad_to_match_top(sprites):
    max_height = max(sprite.shape[0] for sprite in sprites)
    max_width = max(sprite.shape[1] for sprite in sprites)

    def pad_sprite(sprite):
        pad_height = max_height - sprite.shape[0]
        pad_width = max_width - sprite.shape[1]
        return jnp.pad(
            sprite,
            ((pad_height, 0), (pad_width, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    return [pad_sprite(sprite) for sprite in sprites]


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


# debug code
if __name__ == "__main__":

    # Load frames assuming loadFrame default transpose=True gives (W, H, C)
    sub1 = loadFrame("./sprites/seaquest/player_sub/1.npy")
    sub2 = loadFrame("./sprites/seaquest/player_sub/2.npy")
    sub3 = loadFrame("./sprites/seaquest/player_sub/3.npy")
    # pad_to_match expects list of (W, H, C), returns list of (maxW, maxH, C)
    sub_sprite_list = pad_to_match([sub1, sub2, sub3])
    # Ensure stacking results in (N, W, H, C)
    sub_frames_repeated = [
        jnp.repeat(sub_sprite_list[0][None], 4, axis=0),
        jnp.repeat(sub_sprite_list[1][None], 4, axis=0),
        jnp.repeat(sub_sprite_list[2][None], 4, axis=0),
    ]
    SPRITE_PL_SUB = jnp.concatenate(sub_frames_repeated, axis=0) # Shape (N, W, H, C)

    shark1 = loadFrame("./sprites/seaquest/shark/1.npy")
    shark2 = loadFrame("./sprites/seaquest/shark/2.npy")
    shark_sprite_list = pad_to_match([shark1, shark2])
    shark_frames_repeated = [
         jnp.repeat(shark_sprite_list[0][None], 16, axis=0),
         jnp.repeat(shark_sprite_list[1][None], 8, axis=0),
    ]
    SPRITE_SHARK = jnp.concatenate(shark_frames_repeated, axis=0) # Shape (N, W, H, C)


    # load_and_pad_digits returns (NumDigits, W, H, C)
    digits_array = load_and_pad_digits("./sprites/seaquest/digits/{}.npy")

    pygame.init()

    SCALING_FACTOR = 3
    # These constants define the raster size (W, H)
    WIDTH = 400
    HEIGHT = 300

    # Pygame screen size uses (width, height) tuple
    screen = pygame.display.set_mode((int(WIDTH * SCALING_FACTOR), int(HEIGHT * SCALING_FACTOR)))
    clock = pygame.time.Clock()
    running = True
    frame_idx = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        # Create a fresh black raster each frame (W, H, C)
        raster = jnp.zeros((WIDTH, HEIGHT, 3), dtype=jnp.uint8)

        # Get sprite frames - expected shape (W, H, C)
        sub_frame = get_sprite_frame(SPRITE_PL_SUB, frame_idx, loop=True)
        shark_frame = get_sprite_frame(SPRITE_SHARK, frame_idx, loop=True)

        # Render using (x, y) coordinates
        raster = render_at(raster, x=300, y=140, sprite_frame=sub_frame, flip_horizontal=True)
        raster = render_at(raster, x=100, y=100, sprite_frame=shark_frame)

        # Render labels/indicators using (x, y)
        digits = int_to_digits(114514)
        raster = render_label(raster, x=10, y=10, text_digits=digits, char_sprites=digits_array)
        raster = render_indicator(raster, x=10, y=30, value=5, sprite=sub_frame)

        # Render bar using (x, y) and geometric width/height
        raster = render_bar(
            raster, x=10, y=280, value=5, max_value=10, width=100, height=10,
            color=(255, 0, 0, 255), default_color=(0, 0, 255, 255)
        )

        # Update display - expects raster as (W, H, C)
        update_pygame(screen, raster, SCALING_FACTOR, WIDTH, HEIGHT)
        frame_idx += 1
        clock.tick(60)
    pygame.quit()
