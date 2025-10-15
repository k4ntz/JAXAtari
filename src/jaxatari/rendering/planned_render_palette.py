from pathlib import Path, PureWindowsPath
import os
import numpy as np
import jax.numpy as jnp
import jax
from functools import partial
from jax import lax, ShapeDtypeStruct
from typing import List, NamedTuple, Tuple

# -> MODIFIED: Command structure is now for the palette renderer.
'''
Commands:
    0   screen_x        The final X coordinate on the screen.
    1   screen_y        The final Y coordinate on the screen.
    2   shape_mask_id   The index of the boolean shape mask in the SHAPE_MASKS array.
    3   color_id        The index of the color in the PALETTE array.
    4   depth_z         The draw order. Higher numbers are drawn on top.
'''

class RenderPlan(NamedTuple):
    # -> MODIFIED: A (N, 5) array for the simpler palette commands.
    commands: jnp.ndarray
    command_count: jnp.ndarray
    # -> NOTE: sprite_info_table is no longer used by the palette renderer.
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
    """Creates an initial, empty RenderPlan."""
    # The sprite_info_table is not strictly needed for the palette renderer,
    # but we keep the argument for API consistency for now.
    if sprite_info_table is None:
        sprite_info_table = jnp.zeros((0, 4), dtype=jnp.int32)
    return RenderPlan(
        # -> MODIFIED: Command buffer now has 5 columns.
        commands=jnp.zeros((max_sprites, 5), dtype=jnp.int32),
        command_count=jnp.array(0, dtype=jnp.int32),
        sprite_info_table=sprite_info_table,
    )


def loadFrame(fileName, transpose=False):
    """Loads a frame from .npy, ensuring output is RGBA (Height, Width, 4)."""
    frame_np = np.load(fileName)
    if frame_np.ndim != 3 or frame_np.shape[2] != 4:
         raise ValueError(
            f"Invalid frame format in {fileName}. Source .npy must be loadable with 3 dims and 4 channels."
        )
    if transpose:
        frame_np = np.transpose(frame_np, (1, 0, 2))
    return jnp.asarray(frame_np)

# -> MODIFIED: Removed the @jax.jit decorator. JIT-compiled functions cannot perform file I/O.
# This function is part of the one-time setup, so there is no performance loss.
def load_and_pad_digits(path_pattern, num_chars=10):
    """Loads digit sprites, pads them to the max dimensions, assuming (H, W, C) format."""
    digits = []
    max_height, max_width = 0, 0
    for i in range(num_chars):
        digit = loadFrame(path_pattern.format(i), transpose=False)
        max_height = max(max_height, digit.shape[0])
        max_width = max(max_width, digit.shape[1])
        digits.append(digit)

    padded_digits = []
    for digit in digits:
        pad_h = max_height - digit.shape[0]
        pad_w = max_width - digit.shape[1]
        pad_top, pad_bottom = pad_h // 2, pad_h - (pad_h // 2)
        pad_left, pad_right = pad_w // 2, pad_w - (pad_w // 2)
        padded_digit = jnp.pad(
            digit,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        padded_digits.append(padded_digit)
    return jnp.array(padded_digits)


@jax.jit
def render_at(plan: RenderPlan, x: int, y: int, shape_mask_id: int, color_id: int, depth_z: int = 1):
    """ -> MODIFIED: Adds a palette-based draw command to the RenderPlan."""
    command_index = plan.command_count
    new_command = jnp.array([x, y, shape_mask_id, color_id, depth_z])
    updated_commands = plan.commands.at[command_index].set(new_command)
    return plan._replace(
        commands=updated_commands,
        command_count=plan.command_count + 1
    )


@partial(jax.jit, static_argnames=["max_digits_to_render", "spacing"])
def render_label_selective(plan: RenderPlan, x: int, y: int,
                           all_digits,
                           num_to_render,
                           char_shape_ids, # -> MODIFIED: Now takes shape IDs
                           color_id: int,  # -> MODIFIED: Now takes a single color ID
                           start_index: int,
                           max_digits_to_render=2,
                           spacing=8,
                           depth_z=3):
    """ -> MODIFIED: Adds draw commands for a sequence of digits for the palette renderer."""
    def plan_char(i, current_plan):
        command_index = current_plan.command_count
        digit_index_in_array = start_index + i
        digit_value = all_digits[..., digit_index_in_array]

        should_draw_mask = (i < num_to_render)
        shape_id_to_render = char_shape_ids[digit_value]

        render_x = x + i * spacing
        new_command = jnp.stack([
            render_x, y,
            shape_id_to_render,
            jnp.full_like(command_index, color_id),
            jnp.full_like(command_index, depth_z)
        ], axis=-1)

        original_command = current_plan.commands[command_index]
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
def int_to_digits(n, max_digits=8):
    """Convert a non-negative integer to a fixed-length array of digits."""
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

@partial(jax.jit, static_argnames=["screen_height", "screen_width", "max_pixels_per_sprite"])
def execute_plan_palette(plan: RenderPlan, background_raster: jnp.ndarray,
                         all_shape_coords_lx: jnp.ndarray,
                         all_shape_coords_ly: jnp.ndarray,
                         screen_height: int, screen_width: int, max_pixels_per_sprite: int):
    """
    Renders a single environment's object raster from a plan.
    This kernel is designed to be externally vectorized via jax.vmap.
    """
    # The internal logic of this function is IDENTICAL to the _render_single_env_palette
    # function from the previous step. No changes are needed inside this function.
    def get_sprite_pixels(command, index):
        is_active = index < plan.command_count
        sx, sy, shape_id, color_id, z = command
        lx = all_shape_coords_lx[shape_id]
        ly = all_shape_coords_ly[shape_id]
        valid_pixel_mask = (lx != -1)
        screen_x, screen_y = sx + lx, sy + ly
        pixel_color_ids = jnp.full_like(lx, color_id)
        pixel_depths = jnp.full_like(lx, z)
        final_mask = is_active & valid_pixel_mask
        return screen_x, screen_y, pixel_color_ids, pixel_depths, final_mask

    all_x, all_y, all_ids, all_z, final_mask = jax.vmap(get_sprite_pixels)(
        plan.commands, jnp.arange(plan.commands.shape[0])
    )

    all_x, all_y, all_ids, all_z, final_mask = (
        all_x.flatten(), all_y.flatten(), all_ids.flatten(), all_z.flatten(), final_mask.flatten()
    )

    effective_z = jnp.where(final_mask, all_z, -1)
    all_x_clipped = jnp.clip(all_x, 0, screen_width - 1)
    all_y_clipped = jnp.clip(all_y, 0, screen_height - 1)
    linear_indices = all_y_clipped * screen_width + all_x_clipped

    pixel_indices = jnp.arange(all_x.shape[0])
    payload = (effective_z * all_x.shape[0] + pixel_indices).astype(jnp.int32)
    winning_payloads = jax.ops.segment_max(payload, linear_indices, num_segments=screen_height * screen_width)
    winning_pixel_indices = winning_payloads % all_x.shape[0]

    active_segment_mask = (winning_payloads >= 0)
    winning_ids = all_ids[winning_pixel_indices]

    final_raster = jnp.where(
        active_segment_mask.reshape(screen_height, screen_width),
        winning_ids.reshape(screen_height, screen_width),
        background_raster
    )
    return final_raster
