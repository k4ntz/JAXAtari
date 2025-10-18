from pathlib import Path, PureWindowsPath
import os
import numpy as np
import jax.numpy as jnp
import jax
from functools import partial
from jax import lax, ShapeDtypeStruct
from typing import List, NamedTuple, Tuple
import jax.experimental.pallas as pallas

BORDER = False

class RendererConfig(NamedTuple):
    """Configuration for the rendering pipeline with downscaling support."""
    # Source dimensions
    source_dimensions: Tuple[int, int] = (210, 160)  # (height, width)
    channels: int = 3  # 1 for grayscale, 3 for RGB
    # Target dimensions for downscaling
    target_dimensions: Tuple[int, int] = None  # (height, width) to downscale to, or None for no downscaling

    @property
    def width_scaling(self) -> float:
        return self.target_dimensions[1] / self.source_dimensions[1] if self.target_dimensions else 1.0

    @property
    def height_scaling(self) -> float:
        return self.target_dimensions[0] / self.source_dimensions[0] if self.target_dimensions else 1.0

    @property
    def output_shape(self) -> Tuple[int, int, int]:
        if self.target_dimensions:
            return (self.target_dimensions[0], self.target_dimensions[1], self.channels)
        else:
            return (self.source_dimensions[0], self.source_dimensions[1], self.channels)

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


def preprocess_assets_for_downscaling(background: jnp.ndarray, atlas: jnp.ndarray, config: RendererConfig):
    """
    Preprocesses background and atlas for downscaling.
    This should be called once during setup, not during rendering.
    """
    if not config.target_dimensions:
        return background, atlas
    
    # Downscale background
    if background.ndim == 3:
        target_bg_shape = (config.target_dimensions[0], config.target_dimensions[1], background.shape[2])
        background_scaled = jax.image.resize(background, target_bg_shape, method='linear')
    else:
        background_scaled = background
    
    # Downscale atlas
    if atlas.ndim == 3:
        # Calculate scaling factors for atlas
        atlas_h, atlas_w = atlas.shape[:2]
        new_atlas_h = int(atlas_h * config.height_scaling)
        new_atlas_w = int(atlas_w * config.width_scaling)
        target_atlas_shape = (new_atlas_h, new_atlas_w, atlas.shape[2])
        atlas_scaled = jax.image.resize(atlas, target_atlas_shape, method='linear')
    else:
        atlas_scaled = atlas
    
    return background_scaled, atlas_scaled


@partial(jax.jit, static_argnames=["screen_height", "screen_width", "max_sprite_w", "max_sprite_h", "config"])
def execute_plan_vectorized_downscaled(plan: RenderPlan, background: jnp.ndarray, atlas: jnp.ndarray, 
                                      screen_height: int, screen_width: int, 
                                      max_sprite_w: int, max_sprite_h: int,
                                      config: RendererConfig):
    """
    Executes a render plan with downscaling support using a fully static, JIT-compatible scatter
    operation with a segment_max z-buffer.
    """
    # 1. Generate pixel data for ALL potential sprite locations
    all_commands = plan.commands
    command_indices = jnp.arange(all_commands.shape[0])
    max_lx, max_ly = jnp.meshgrid(jnp.arange(max_sprite_w), jnp.arange(max_sprite_h))

    def get_sprite_pixels(command, index):
        is_active = index < plan.command_count
        sx, sy, u, v, w, h, fh, fv, z = command

        # Create a mask for the actual size of the current sprite
        valid_pixel_mask = (max_lx < w) & (max_ly < h)

        # Calculate atlas coordinates for the full grid
        atlas_x_base = jnp.where(fh == 1, u + (w - 1 - max_lx), u + max_lx)
        atlas_y_base = jnp.where(fv == 1, v + (h - 1 - max_ly), v + max_ly)
        
        atlas_x = jnp.where(valid_pixel_mask, atlas_x_base, 0)
        atlas_y = jnp.where(valid_pixel_mask, atlas_y_base, 0)
        
        # Gather pixels for the entire max_size grid
        pixels_rgba = atlas[atlas_y, atlas_x]
        
        # *** KEY CHANGE: Scale screen coordinates ***
        scaled_sx = jnp.round(sx * config.width_scaling).astype(jnp.int32)
        scaled_sy = jnp.round(sy * config.height_scaling).astype(jnp.int32)
        
        # Calculate screen coordinates for the full grid (now scaled)
        screen_x = scaled_sx + max_lx
        screen_y = scaled_sy + max_ly
        
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
    
    # 2. Flatten all data into 1D arrays
    all_x = all_x.flatten()
    all_y = all_y.flatten()
    all_rgb = all_rgb.reshape(-1, 3)
    all_z = all_z.flatten()
    final_mask = final_mask.flatten()

    # 3. Perform the parallel Z-buffer using segment_max
    effective_z = jnp.where(final_mask, all_z, -1.0)
    
    # *** KEY CHANGE: Use scaled screen dimensions ***
    scaled_screen_height, scaled_screen_width = config.output_shape[:2]
    linear_indices = all_y * scaled_screen_width + all_x
    
    pixel_indices = jnp.arange(all_x.shape[0])
    payload = (effective_z.astype(jnp.float32) * all_x.shape[0] + pixel_indices).astype(jnp.int32)
    
    # Find the winning payload for each screen pixel location
    winning_payloads = jax.ops.segment_max(payload, linear_indices, num_segments=scaled_screen_height * scaled_screen_width)
    
    # Extract the original index of the winning pixels from the payload
    winning_pixel_indices = winning_payloads % all_x.shape[0]

    # A segment is active if its max z-depth was greater than -1
    active_segment_mask = (winning_payloads >= 0).reshape(scaled_screen_height, scaled_screen_width)
    
    # 4. Gather the final colors and scatter them onto the canvas
    winning_colors = all_rgb[winning_pixel_indices].reshape(scaled_screen_height, scaled_screen_width, 3)
    
    canvas = background[..., :3]
    final_canvas = jnp.where(
        active_segment_mask[..., None], # Broadcast mask over RGB channels
        winning_colors,
        canvas
    )
    
    return final_canvas.astype(jnp.uint8)


# Convenience function for 84x84 downscaling
def execute_plan_84x84(plan: RenderPlan, background: jnp.ndarray, atlas: jnp.ndarray, 
                       screen_height: int, screen_width: int, 
                       max_sprite_w: int, max_sprite_h: int):
    """
    Convenience function for 84x84 downscaling (common Atari RL resolution).
    """
    config = RendererConfig(
        source_dimensions=(210, 160),
        target_dimensions=(84, 84),
        channels=3
    )
    
    # Preprocess assets for downscaling
    background_scaled, atlas_scaled = preprocess_assets_for_downscaling(background, atlas, config)
    
    return execute_plan_vectorized_downscaled(
        plan, background_scaled, atlas_scaled, 
        screen_height, screen_width, max_sprite_w, max_sprite_h, config
    )


# Convenience function for grayscale 84x84 downscaling
def execute_plan_84x84_grayscale(plan: RenderPlan, background: jnp.ndarray, atlas: jnp.ndarray, 
                                 screen_height: int, screen_width: int, 
                                 max_sprite_w: int, max_sprite_h: int):
    """
    Convenience function for grayscale 84x84 downscaling.
    """
    config = RendererConfig(
        source_dimensions=(210, 160),
        target_dimensions=(84, 84),
        channels=1
    )
    
    # Preprocess assets for downscaling
    background_scaled, atlas_scaled = preprocess_assets_for_downscaling(background, atlas, config)
    
    # Convert to grayscale
    if background_scaled.shape[2] == 3:
        background_gray = jnp.mean(background_scaled, axis=2, keepdims=True)
    else:
        background_gray = background_scaled
    
    if atlas_scaled.shape[2] == 3:
        atlas_gray = jnp.mean(atlas_scaled, axis=2, keepdims=True)
    else:
        atlas_gray = atlas_scaled
    
    return execute_plan_vectorized_downscaled(
        plan, background_gray, atlas_gray, 
        screen_height, screen_width, max_sprite_w, max_sprite_h, config
    )


# Additional functions needed for compatibility
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


@partial(jax.jit, static_argnames=["max_digits"])
def int_to_digits(n, max_digits=8):
    """Convert a non-negative integer to a fixed-length JAX array of digits."""
    n = jnp.maximum(n, 0)
    max_val = 10**max_digits - 1
    n = jnp.minimum(n, max_val)

    def scan_body(carry, _):
        digit = carry % 10
        next_carry = carry // 10
        return next_carry, digit

    _, digits_reversed = lax.scan(scan_body, n, None, length=max_digits)
    digits = jnp.flip(digits_reversed, axis=0)
    return digits.T


# All other functions remain the same...
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
    """Loads a frame from .npy, ensuring output is (Height, Width, Channels)."""
    frame_np = np.load(fileName)
    if frame_np.ndim != 3 or frame_np.shape[2] != 4:
         raise ValueError(
            f"Invalid frame format in {fileName}. Source .npy must be loadable with 3 dims and 4 channels."
        )

    if transpose:
        frame_np = np.transpose(frame_np, (1, 0, 2))
    return jnp.asarray(frame_np)


@jax.jit
def render_at(plan: RenderPlan, x, y, sprite_id, depth_z=1,
                   flip_h=False, flip_v=False):
    """Adds a single draw command to the RenderPlan."""
    command_index = plan.command_count
    sprite_info = plan.sprite_info_table[sprite_id]
    atlas_u, atlas_v, width, height = sprite_info

    new_command = jnp.array([
        x, y,
        atlas_u, atlas_v,
        width, height,
        flip_h,
        flip_v,
        depth_z
    ])

    updated_commands = plan.commands.at[command_index].set(new_command)
    return plan._replace(
        commands=updated_commands,
        command_count=plan.command_count + 1
    )


@partial(jax.jit, static_argnames=["max_digits"])
def int_to_digits(n, max_digits=8):
    """Convert a non-negative integer to a fixed-length JAX array of digits."""
    n = jnp.maximum(n, 0)
    max_val = 10**max_digits - 1
    n = jnp.minimum(n, max_val)

    def scan_body(carry, _):
        digit = carry % 10
        next_carry = carry // 10
        return next_carry, digit

    _, digits_reversed = lax.scan(scan_body, n, None, length=max_digits)
    digits = jnp.flip(digits_reversed, axis=0)
    return digits.T
