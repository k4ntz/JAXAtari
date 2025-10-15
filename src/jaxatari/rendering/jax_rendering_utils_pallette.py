import jax.numpy as jnp
import jax
from functools import partial
import numpy as np
from typing import Dict, Any, List, Tuple, NamedTuple

# A special ID to represent transparency. Must be an ID not used by any color.
# Using a high value like 255 is a safe choice if you have < 255 colors.
TRANSPARENT_ID = 255 

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


    def _create_id_mask(self, sprite_data, color_to_id: Dict) -> np.ndarray:
        """Converts a single 3D RGBA sprite into a 2D integer ID mask."""
        sprite_data_np = np.array(sprite_data)

        h, w, _ = sprite_data_np.shape
        id_mask = np.full((h, w), TRANSPARENT_ID, dtype=np.uint8)
        
        for r in range(h):
            for c in range(w):
                pixel = sprite_data_np[r, c]
                if pixel[3] > 0:
                    # Round to handle interpolation artifacts from downscaling
                    rgb = (int(np.round(pixel[0])), int(np.round(pixel[1])), int(np.round(pixel[2])))
                    if rgb in color_to_id:
                        id_mask[r, c] = color_to_id[rgb]
        return id_mask

    def _create_id_masks_from_batch(self, sprite_batch_data, color_to_id: Dict) -> List[np.ndarray]:
        """Converts a 4D RGBA sprite batch into a list of 2D integer ID masks."""
        id_masks_list = []
        sprite_batch_np = np.array(sprite_batch_data)
        
        for single_sprite_np in sprite_batch_np:
            id_masks_list.append(self._create_id_mask(single_sprite_np, color_to_id))
        return id_masks_list
    

    def _pad_and_offset_masks(self, id_masks: List[np.ndarray]) -> Tuple[List[np.ndarray], jnp.ndarray]:
        """
        Pads a list of 2D ID masks to the same dimensions and calculates the flip_offset.
        This logic is adapted from your original `pad_to_match` function.
        """
        if not id_masks:
            return [], jnp.array([0, 0])

        # Find the max dimensions within this list of masks
        max_height = max(m.shape[0] for m in id_masks)
        max_width = max(m.shape[1] for m in id_masks)

        padded_masks = []
        for mask in id_masks:
            h, w = mask.shape
            pad_h = max_height - h
            pad_w = max_width - w

            # Pad on the right and bottom with the transparent ID
            padded_mask = np.pad(
                mask,
                ((0, pad_h), (0, pad_w)),
                mode="constant",
                constant_values=TRANSPARENT_ID
            )
            padded_masks.append(padded_mask)

        # The flip_offset is the maximum padding added, which corrects for the largest
        # possible displacement when a smaller sprite in the set is flipped.
        # Note: In this implementation, the offset is constant for the whole animation set.
        max_pad_h = max_height - min(m.shape[0] for m in id_masks)
        max_pad_w = max_width - min(m.shape[1] for m in id_masks)
        
        # flip_offset[0] for width (dx), flip_offset[1] for height (dy)
        flip_offset = jnp.array([max_pad_w, max_pad_h], dtype=jnp.int32)

        return padded_masks, flip_offset
    

    def setup_rendering_assets(
        self, loaded_sprites: Dict[str, jnp.ndarray]
    ) -> Tuple[jnp.ndarray, Dict[str, Any], jnp.ndarray, Dict[Tuple[int, int, int], int]]:
        """
        Analyzes RGBA sprites to generate assets for a palette renderer,
        including a pre-rendered background and optional downscaling.
        """
        if "background" not in loaded_sprites:
            raise ValueError("A 'background' sprite must be provided in loaded_sprites.")

        # 1. Optionally downscale all RGBA sprites FIRST (before palette extraction)
        # This allows colors to blend smoothly during downscaling
        if self.config.downscale:
            downscaled_sprites = {}
            for name, sprite_data in loaded_sprites.items():
                if sprite_data.ndim == 3:
                    # Single sprite: (H, W, 4)
                    new_h = int(sprite_data.shape[0] * self.config.height_scaling)
                    new_w = int(sprite_data.shape[1] * self.config.width_scaling)
                    downscaled_sprites[name] = jax.image.resize(
                        sprite_data,
                        (new_h, new_w, 4),
                        method='linear'  # Smooth interpolation for color blending
                    )
                elif sprite_data.ndim == 4:
                    # Batched sprites: (N, H, W, 4)
                    new_h = int(sprite_data.shape[1] * self.config.height_scaling)
                    new_w = int(sprite_data.shape[2] * self.config.width_scaling)
                    downscaled_sprites[name] = jax.image.resize(
                        sprite_data,
                        (sprite_data.shape[0], new_h, new_w, 4),
                        method='linear'  # Smooth interpolation for color blending
                    )
            # Use downscaled sprites for the rest of the pipeline
            sprites_to_use = downscaled_sprites
        else:
            sprites_to_use = loaded_sprites

        # 2. Discover unique colors from ALL (potentially downscaled) sprites to build the palette
        color_to_id = {}
        palette_list = []
        next_id = 0

        for sprite_data in sprites_to_use.values():
            sprites_to_process = sprite_data if sprite_data.ndim >= 4 else jnp.expand_dims(sprite_data, axis=0)
            for sprite in sprites_to_process:
                pixels = np.array(sprite.reshape(-1, 4))
                for r, g, b, a in pixels:
                    if a > 0:
                        # Round to nearest integer to handle interpolation artifacts
                        rgb = (int(np.round(r)), int(np.round(g)), int(np.round(b)))
                        if rgb not in color_to_id:
                            color_to_id[rgb] = next_id
                            palette_list.append(rgb)
                            next_id += 1

        if self.config.channels == 1:
            # For grayscale, create a palette mapping ID to intensity (0-255)
            # We'll use a simple grayscale conversion formula for color mapping.
            gray_palette = []
            for r, g, b in palette_list:
                intensity = 0.299 * r + 0.587 * g + 0.114 * b
                gray_palette.append(int(intensity))
            PALETTE = jnp.array(gray_palette, dtype=jnp.uint8).reshape(-1, 1)
        else:
            PALETTE = jnp.array(palette_list, dtype=jnp.uint8)

        # 3. Create the pre-rendered background raster from (potentially downscaled) background
        bg_sprite = sprites_to_use["background"]
        height, width, _ = bg_sprite.shape
        pre_rendered_bg_hr = np.zeros((height, width), dtype=np.uint8)
        bg_pixels_np = np.array(bg_sprite)
        for r in range(height):
            for c in range(width):
                pixel = bg_pixels_np[r, c]
                if pixel[3] > 0:
                    rgb = (int(np.round(pixel[0])), int(np.round(pixel[1])), int(np.round(pixel[2])))
                    pre_rendered_bg_hr[r, c] = color_to_id[rgb]
        
        PRE_RENDERED_BG = jnp.asarray(pre_rendered_bg_hr)

        # 4. Create ID masks and Flip Offsets from (potentially downscaled) sprites
        SHAPE_MASKS = {}
        FLIP_OFFSETS = {} # NEW: Create the separate dictionary

        for name, sprite_data in sprites_to_use.items():
            if name == "background":
                continue

            is_batched = sprite_data.ndim == 4

            if is_batched:
                # For batched sprites, we need to pad them and calculate the offset
                id_masks_list = self._create_id_masks_from_batch(sprite_data, color_to_id)
                
                # Use a new helper to pad the masks and get the offset
                padded_masks, flip_offset = self._pad_and_offset_masks(id_masks_list)

                SHAPE_MASKS[name] = jnp.stack(padded_masks)
                FLIP_OFFSETS[name] = flip_offset # NEW: Store offset in its own dict
            else:
                # For single sprites, just create the ID mask
                SHAPE_MASKS[name] = self._create_id_mask(sprite_data, color_to_id)

        return PALETTE, SHAPE_MASKS, PRE_RENDERED_BG, color_to_id, FLIP_OFFSETS
    

    # ============= Jitted planning and execution functions =============
    @partial(jax.jit, static_argnames=['self'])
    def create_object_raster(self, pre_rendered_bg: jnp.ndarray) -> jnp.ndarray:
        """Creates the initial 2D object raster from the pre-rendered background."""
        return pre_rendered_bg
    

    @partial(jax.jit, static_argnums=(0,))
    def render_at(self, object_raster: jnp.ndarray, x: int, y: int, sprite_mask: jnp.ndarray, flip_horizontal: bool = False, flip_vertical: bool = False, flip_offset: jnp.ndarray = jnp.array([0, 0])) -> jnp.ndarray:
        """
        Stamps an object's ID onto a raster using an efficient local slice update.
        
        Args:
            flip_offset: [width_pad, height_pad] - padding added to sprites for consistent dimensions
        """

        # --- 1. Flip the ID Mask ---
        flipped_mask = jax.lax.cond(
            flip_horizontal,
            lambda m: jnp.flip(m, axis=1),
            lambda m: m,
            sprite_mask
        )
        flipped_mask = jax.lax.cond(
            flip_vertical,
            lambda m: jnp.flip(m, axis=0),
            lambda m: m,
            flipped_mask
        )

        # --- 2. Correct the Drawing Position ---
        # If flipping, shift the position to compensate for padding.
        # flip_offset = [width_pad, height_pad] for HWC sprites
        corrected_x = jax.lax.cond(
            flip_horizontal,
            lambda val: val - flip_offset[0], # offset[0] is width padding - ADD to shift right
            lambda val: val,
            x
        )
        corrected_y = jax.lax.cond(
            flip_vertical,
            lambda val: val - flip_offset[1], # offset[1] is height padding - ADD to shift down
            lambda val: val,
            y
        )
        
        # --- 3. Scale and Stamp ---
        # The rest of the function proceeds as before, using the
        # flipped_mask and the corrected_x/y coordinates.
        scaled_x = jnp.round(corrected_x * self.config.width_scaling).astype(jnp.int32)
        scaled_y = jnp.round(corrected_y * self.config.height_scaling).astype(jnp.int32)
        
        target_slice = jax.lax.dynamic_slice(object_raster, (scaled_y, scaled_x), flipped_mask.shape)
        updated_slice = jnp.where(flipped_mask != TRANSPARENT_ID, flipped_mask, target_slice)
        return jax.lax.dynamic_update_slice(object_raster, updated_slice, (scaled_y, scaled_x))
    

    @partial(jax.jit, static_argnames=['self', 'width', 'height'])
    def draw_filled_rect(self, object_raster: jnp.ndarray, x: int, y: int, 
                         width: int, height: int, object_id: int) -> jnp.ndarray:
        """
        Draws a filled rectangle of a single object ID directly onto the raster.
        Enforces static width and height for efficient execution.
        """
        # Scale geometry from source to target resolution
        scaled_x = jnp.round(x * self.config.width_scaling).astype(jnp.int32)
        scaled_y = jnp.round(y * self.config.height_scaling).astype(jnp.int32)
        scaled_w = jnp.round(width * self.config.width_scaling).astype(jnp.int32)
        scaled_h = jnp.round(height * self.config.height_scaling).astype(jnp.int32)
        
        # Create the content for the rectangle (a small 2D array)
        rect_content = jnp.full((scaled_h, scaled_w), object_id, dtype=object_raster.dtype)

        # Update the raster with the new rectangle
        return jax.lax.dynamic_update_slice(object_raster, rect_content, (scaled_y, scaled_x))
    
    @partial(jax.jit, static_argnames=['self', 'max_width', 'max_height'])
    def draw_filled_rect_variable(self, object_raster: jnp.ndarray, x: int, y: int,
                                  width: int, height: int, object_id: int,
                                  max_width: int, max_height: int) -> jnp.ndarray:
        """
        Draws a filled rectangle with runtime width/height by writing a fixed-size block
        (max_width x max_height) and masking within it. Shapes remain static for JIT.
        """
        # Scale geometry
        scaled_x = jnp.round(x * self.config.width_scaling).astype(jnp.int32)
        scaled_y = jnp.round(y * self.config.height_scaling).astype(jnp.int32)
        scaled_w = jnp.round(width * self.config.width_scaling).astype(jnp.int32)
        scaled_h = jnp.round(height * self.config.height_scaling).astype(jnp.int32)

        # Compute scaled maximums as Python ints so dynamic_slice sizes are static
        scaled_max_w = int(round(max_width * self.config.width_scaling))
        scaled_max_h = int(round(max_height * self.config.height_scaling))

        # Fixed-size target slice
        target_slice = jax.lax.dynamic_slice(
            object_raster, (scaled_y, scaled_x), (scaled_max_h, scaled_max_w)
        )

        # Build mask for active area within fixed-size block
        xx, yy = jnp.meshgrid(jnp.arange(scaled_max_w), jnp.arange(scaled_max_h))
        inside = (xx < scaled_w) & (yy < scaled_h)

        # Compose content with constant shape
        content = jnp.where(inside, jnp.asarray(object_id, object_raster.dtype), target_slice)

        return jax.lax.dynamic_update_slice(object_raster, content, (scaled_y, scaled_x))

    # ============= Static mask builders for padded geometry (CPU-time helpers) =============
    def build_platform_mask_padded(self, max_width: int, height: int, actual_width: int, color_id: int) -> jnp.ndarray:
        """
        Build a platform ID mask of shape (height, max_width), filled with TRANSPARENT_ID,
        with the first actual_width columns set to color_id.
        """
        max_width = int(max_width)
        height = int(height)
        actual_width = int(max(0, min(actual_width, max_width)))
        mask_np = np.full((height, max_width), TRANSPARENT_ID, dtype=np.uint8)
        if actual_width > 0:
            mask_np[:, :actual_width] = np.uint8(color_id)
        return jnp.asarray(mask_np)

    def build_ladder_mask_padded(self, max_width: int, max_height: int, ladder_w: int, ladder_h: int,
                                 rung_h: int, space_h: int, color_id: int) -> jnp.ndarray:
        """
        Build a ladder ID mask of shape (max_height, max_width) with rails and rungs,
        padded with TRANSPARENT_ID. Ladder occupies width=ladder_w and height=ladder_h
        anchored at top-left.
        """
        max_width = int(max_width)
        max_height = int(max_height)
        ladder_w = int(max(0, min(ladder_w, max_width)))
        ladder_h = int(max(0, min(ladder_h, max_height)))
        rung_h = int(max(1, rung_h))
        space_h = int(max(0, space_h))
        color_u8 = np.uint8(color_id)

        mask_np = np.full((max_height, max_width), TRANSPARENT_ID, dtype=np.uint8)
        if ladder_w <= 0 or ladder_h <= 0:
            return jnp.asarray(mask_np)

        # Rails at x=0 and x=ladder_w-1
        mask_np[:ladder_h, 0] = color_u8
        if ladder_w > 1:
            mask_np[:ladder_h, ladder_w - 1] = color_u8

        inner_w = max(0, ladder_w - 2)
        if inner_w > 0 and rung_h > 0:
            step = rung_h + space_h
            y = 0
            while y < ladder_h:
                h_here = min(rung_h, ladder_h - y)
                if h_here > 0:
                    mask_np[y:y + h_here, 1:1 + inner_w] = color_u8
                y += step

        return jnp.asarray(mask_np)

    # ============= Various rendering functions =============

    @partial(jax.jit, static_argnames=['self', 'spacing', 'max_digits'])
    def render_label(self, object_raster: jnp.ndarray, x: int, y: int, digits: jnp.ndarray, digit_masks: jnp.ndarray,
                    spacing: int, max_digits: int = 2) -> jnp.ndarray:
        """Stamps a sequence of digits onto the object raster."""
        def render_char(i, current_raster):
            digit_value = digits[i]
            char_mask = digit_masks[digit_value]
            char_x = x + i * spacing
            return self.render_at(current_raster, char_x, y, char_mask)

        return jax.lax.fori_loop(0, max_digits, render_char, object_raster)
    
    @partial(jax.jit, static_argnames=['self', 'spacing', 'max_digits_to_render'])
    def render_label_selective(self, object_raster: jnp.ndarray, x: int, y: int,
                               all_digits: jnp.ndarray,
                               digit_id_masks: jnp.ndarray, # Changed from digit_masks
                               start_index: int,
                               num_to_render: int,
                               spacing: int = 16,
                               max_digits_to_render: int = 2) -> jnp.ndarray:
        """
        Renders a specified number of digits using pre-baked Object ID masks.
        """
        def render_char(i, current_raster):
            should_draw = (i < num_to_render)

            def true_fn(raster_in):
                digit_index_in_array = start_index + i
                digit_value = all_digits[digit_index_in_array]
                
                # Select the correct INTEGER ID mask for the digit
                char_id_mask = digit_id_masks[digit_value]
                
                render_x = x + i * spacing
                # Call the new render_at, which accepts the integer ID mask
                return self.render_at(raster_in, render_x, y, char_id_mask)

            def false_fn(raster_in):
                return raster_in

            return jax.lax.cond(should_draw, true_fn, false_fn, current_raster)

        return jax.lax.fori_loop(0, max_digits_to_render, render_char, object_raster)
    


    @partial(jax.jit, static_argnames=['self', 'spacing', 'max_value'])
    def render_indicator(self, object_raster: jnp.ndarray, x: int, y: int,
                         value: int,
                         shape_mask: jnp.ndarray,
                         spacing: int = 15,
                         max_value: int = 5) -> jnp.ndarray:
        """
        Renders 'value' copies of a sprite using lax.cond for efficiency.
        """
        def render_single_indicator(i, current_raster):
            should_draw = (i < value)

            def true_fn(raster_in):
                indicator_x = x + i * spacing
                return self.render_at(raster_in, indicator_x, y, shape_mask)

            def false_fn(raster_in):
                return raster_in

            return jax.lax.cond(should_draw, true_fn, false_fn, current_raster)

        return jax.lax.fori_loop(0, max_value, render_single_indicator, object_raster)


    @partial(jax.jit, static_argnames=['self'])
    def render_bar(self, object_raster: jnp.ndarray, x: int, y: int,
                   value: float, max_value: float,
                   width: int, height: int,
                   color_id: int, default_color_id: int) -> jnp.ndarray:
        """
        Renders a horizontal progress bar by directly modifying the object raster.
        """
        # --- Scale all geometric parameters based on the renderer config ---
        scaled_x = jnp.round(x * self.config.width_scaling).astype(jnp.int32)
        scaled_y = jnp.round(y * self.config.height_scaling).astype(jnp.int32)
        scaled_width = jnp.round(width * self.config.width_scaling).astype(jnp.int32)
        scaled_height = jnp.round(height * self.config.height_scaling).astype(jnp.int32)

        # --- Create the bar content as a small 2D array of Object IDs ---
        fill_width = jnp.clip(jnp.nan_to_num((value / max_value) * scaled_width), 0, scaled_width).astype(jnp.int32)
        
        # Create a grid of x-coordinates for the small bar sprite
        bar_xx, _ = jnp.meshgrid(jnp.arange(scaled_width), jnp.arange(scaled_height))
        
        # Use the grid to create the content of the bar
        bar_content = jnp.where(
            bar_xx < fill_width,
            color_id,
            default_color_id
        ).astype(object_raster.dtype)

        # --- Place the generated bar content directly onto the main raster ---
        return jax.lax.dynamic_update_slice(
            object_raster,
            bar_content,
            (scaled_y, scaled_x)
        )
    

    # ========= Final rendering step: palette lookup ===========
    @partial(jax.jit, static_argnames=['self'])
    def render_from_palette(self, object_raster: jnp.ndarray, palette: jnp.ndarray) -> jnp.ndarray:
        """Generates the final image using a palette lookup."""
        final_image = palette[object_raster]

        if self.config.channels == 1 and final_image.ndim == 2:
            final_image = final_image[..., None] # Ensure channel dim exists for grayscale
        
        return final_image


    # --- Utility function: integer to digit array conversion ---
    @partial(jax.jit, static_argnames=["max_digits", "self"])
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
        _, digits_reversed = jax.lax.scan(scan_body, n, None, length=max_digits)

        # Flip to get digits in the correct order (most significant first).
        digits = jnp.flip(digits_reversed, axis=0)

        # Transpose the result.
        # If the input was a batch, this converts (length, batch) -> (batch, length).
        # If the input was a scalar, this is a no-op on the 1D array.
        return digits.T