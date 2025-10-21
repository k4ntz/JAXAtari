import os
import jax.numpy as jnp
import jax
from functools import partial
import numpy as np
from typing import Dict, Any, List, Tuple, NamedTuple
from jax.scipy.ndimage import map_coordinates

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
    def __init__(self, config: RendererConfig, transparent_id: int = 255):
        self.config = config
        # A special ID to represent transparency. Must be an ID not used by any color (No Atari game should have more than 255 colors in the palette)
        # there should never be a need to change this!
        self.TRANSPARENT_ID = transparent_id

        # Precompute full-raster coordinate grids for mask-based drawing.
        # The target raster size is determined by the renderer config.
        if self.config.downscale:
            target_h, target_w = self.config.downscale[0], self.config.downscale[1]
        else:
            target_h, target_w = self.config.game_dimensions[0], self.config.game_dimensions[1]

        # Cached coordinate grids (H, W)
        self._xx, self._yy = jnp.meshgrid(jnp.arange(target_w), jnp.arange(target_h), indexing='xy')

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
        id_mask = np.full((h, w), self.TRANSPARENT_ID, dtype=np.uint8)
        
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
                constant_values=self.TRANSPARENT_ID
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

    def pad_to_match(self, sprites: List[jnp.ndarray]) -> Tuple[List[jnp.ndarray], List[jnp.ndarray]]:
        """
        Pads RGBA sprites by anchoring them to the top-left, and calculates the
        correct PER-SPRITE flip offset for each.
        """
        if not sprites:
            return [], []

        max_height = max(sprite.shape[0] for sprite in sprites)
        max_width = max(sprite.shape[1] for sprite in sprites)

        padded_sprites = []
        flip_offsets = []
        
        for sprite in sprites:
            h, w, c = sprite.shape
            pad_h = max_height - h
            pad_w = max_width - w

            # Use the correct top-left anchor padding
            padded_sprite = jnp.pad(
                sprite,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode="constant",
                constant_values=0
            )
            padded_sprites.append(padded_sprite)
            
            # --- THE FINAL, CRITICAL FIX ---
            # The flip_offset for EACH sprite is its OWN padding amount,
            # not the maximum padding of the group.
            flip_offset = jnp.array([pad_w, pad_h], dtype=jnp.int32)
            flip_offsets.append(flip_offset)

        return padded_sprites, flip_offsets
    
    # ---------------------------------------------------------------------------- #
    #                          Internal Processing Methods                         #
    # ---------------------------------------------------------------------------- #
    
    def _create_palette(self, assets: List[jnp.ndarray]) -> Tuple[jnp.ndarray, Dict[Tuple, int]]:
        """Generates a color palette and mapping from all provided RGBA assets."""
        color_to_id = {}
        palette_list = []
        next_id = 0

        for asset_data in assets:
            # Handle both single (3D) and batched (4D) assets
            sprites_to_process = asset_data if asset_data.ndim == 4 else jnp.expand_dims(asset_data, 0)
            for sprite in sprites_to_process:
                pixels = np.array(sprite.reshape(-1, 4))
                for r, g, b, a in pixels:
                    if a > 128:
                        rgb = (int(r), int(g), int(b))
                        if rgb not in color_to_id:
                            color_to_id[rgb] = next_id
                            palette_list.append(rgb)
                            next_id += 1
        
        # Create the final JAX array for the palette
        if self.config.channels == 1:
            gray_palette = [int(0.299*r + 0.587*g + 0.114*b) for r, g, b in palette_list]
            PALETTE = jnp.array(gray_palette, dtype=jnp.uint8).reshape(-1, 1)
        else:
            PALETTE = jnp.array(palette_list, dtype=jnp.uint8)
            
        return PALETTE, color_to_id

    def _create_id_mask(self, rgba_sprite: jnp.ndarray, color_to_id: Dict) -> jnp.ndarray:
        """Converts a single RGBA sprite to a palette-ID mask."""
        h, w, _ = rgba_sprite.shape
        id_mask = np.full((h, w), self.TRANSPARENT_ID, dtype=np.uint8)

        # Use numpy for faster iteration
        sprite_np = np.array(rgba_sprite)
        for r_idx in range(h):
            for c_idx in range(w):
                r, g, b, a = sprite_np[r_idx, c_idx]
                if a > 128:
                    id_mask[r_idx, c_idx] = color_to_id[(int(r), int(g), int(b))]
        return jnp.asarray(id_mask)

    def _create_shape_masks(self, sprites_dict: Dict, color_to_id: Dict) -> Dict:
        """Converts a dictionary of RGBA sprites/stacks to ID masks."""
        shape_masks = {}
        for name, data in sprites_dict.items():
            if data.ndim == 4: # Batched sprites
                masks = [self._create_id_mask(s, color_to_id) for s in data]
                shape_masks[name] = jnp.stack(masks)
            else: # Single sprite
                shape_masks[name] = self._create_id_mask(data, color_to_id)
            
            # Scale sprite masks proportionally if downscaling is configured
            if self.config.downscale:
                if data.ndim == 4: # Batched sprites
                    scaled_masks = []
                    for mask in shape_masks[name]:
                        # Calculate new dimensions based on scaling factors
                        original_h, original_w = mask.shape
                        scaled_h = int(original_h * self.config.height_scaling)
                        scaled_w = int(original_w * self.config.width_scaling)
                        
                        scaled_mask = jax.image.resize(
                            mask, 
                            (scaled_h, scaled_w), 
                            method='nearest'
                        )
                        scaled_masks.append(scaled_mask)
                    shape_masks[name] = jnp.stack(scaled_masks)
                else: # Single sprite
                    # Calculate new dimensions based on scaling factors
                    original_h, original_w = shape_masks[name].shape
                    scaled_h = int(original_h * self.config.height_scaling)
                    scaled_w = int(original_w * self.config.width_scaling)
                    
                    scaled_mask = jax.image.resize(
                        shape_masks[name], 
                        (scaled_h, scaled_w), 
                        method='nearest'
                    )
                    shape_masks[name] = scaled_mask
        
        return shape_masks

    def _create_background_raster(self, bg_rgba: jnp.ndarray, color_to_id: Dict) -> jnp.ndarray:
        # First create the ID mask from the original background
        id_mask = self._create_id_mask(bg_rgba, color_to_id)
        
        # Scale to target dimensions if downscaling is configured
        if self.config.downscale:
            target_h, target_w = self.config.downscale[0], self.config.downscale[1]
            # Use JAX's resize function to scale the background
            id_mask_scaled = jax.image.resize(
                id_mask, 
                (target_h, target_w), 
                method='nearest'
            )
            return id_mask_scaled
        
        return id_mask


    def load_and_setup_assets(self, asset_config: list, base_path: str):
        """
        Loads, processes, and prepares all game assets from a configuration list.

        This function handles:
        - Loading individual .npy files.
        - Padding and stacking sprite groups.
        - Calculating and storing flip offsets correctly and internally.
        - Generating the palette, shape masks, and background raster.

        Args:
            base_path: The directory path where sprite .npy files are located.
            asset_config: A list of dictionaries defining the assets to load.
                          Each dict should have 'name', 'type', and path info.

        Returns:
            A tuple: (PALETTE, SHAPE_MASKS, BACKGROUND, COLOR_TO_ID, FLIP_OFFSETS)
        """
        raw_sprites_dict = {}
        FLIP_OFFSETS = {}
        background_rgba = None

        # 1. Load all assets from the configuration manifest
        for asset in asset_config:
            name, asset_type = asset.get('name'), asset.get('type')

            if asset_type == 'background':
                # Support either file-backed or procedural backgrounds
                if 'file' in asset:
                    path = os.path.join(base_path, asset['file'])
                    background_rgba = self.loadFrame(path)
                elif 'data' in asset:
                    background_rgba = asset['data']
                else:
                    raise ValueError("Background asset must include 'file' or 'data'.")
                continue

            if asset_type == 'single':
                if 'file' in asset:
                    path = os.path.join(base_path, asset['file'])
                    raw_sprites_dict[name] = self.loadFrame(path)
                elif 'data' in asset:
                    raw_sprites_dict[name] = asset['data']
                else:
                    raise ValueError(f"Single asset '{name}' must include 'file' or 'data'.")
                FLIP_OFFSETS[name] = jnp.array([0, 0])

            elif asset_type == 'group':
                if 'files' in asset:
                    paths = [os.path.join(base_path, f) for f in asset['files']]
                    sprites_to_pad = [self.loadFrame(p) for p in paths]
                elif 'data' in asset:
                    # Allow passing a list/stack of RGBA sprites directly
                    data = asset['data']
                    sprites_to_pad = list(data) if isinstance(data, (list, tuple)) else [s for s in data]
                else:
                    raise ValueError(f"Group asset '{name}' must include 'files' or 'data'.")

                padded_sprites, flip_offset = self.pad_to_match(sprites_to_pad)
                raw_sprites_dict[name] = jnp.stack(padded_sprites)
                # Use per-sprite offsets; keep API-compatible scalar by picking first if uniform
                FLIP_OFFSETS[name] = flip_offset[0]

            elif asset_type == 'digits':
                if 'pattern' in asset:
                    pattern = os.path.join(base_path, asset['pattern'])
                    raw_sprites_dict[name] = self.load_and_pad_digits(pattern)
                elif 'data' in asset:
                    # Accept preloaded/padded digit stacks directly
                    raw_sprites_dict[name] = asset['data']
                else:
                    raise ValueError(f"Digits asset '{name}' must include 'pattern' or 'data'.")
                FLIP_OFFSETS[name] = jnp.array([0, 0])

            elif asset_type == 'procedural':
                # For procedural assets, the data is already provided.
                raw_sprites_dict[name] = asset['data']
                FLIP_OFFSETS[name] = jnp.array([0, 0])
                
        if background_rgba is None:
            raise ValueError("Asset config must include an asset of type 'background'")

        # 2. Build palette from all loaded sprites and the background
        all_assets = list(raw_sprites_dict.values()) + [background_rgba]
        PALETTE, COLOR_TO_ID = self._create_palette(all_assets)

        # 3. Create palette-indexed shape masks for all sprites
        SHAPE_MASKS = self._create_shape_masks(raw_sprites_dict, COLOR_TO_ID)

        # 4. Create the final background raster
        BACKGROUND = self._create_background_raster(background_rgba, COLOR_TO_ID)
        
        return PALETTE, SHAPE_MASKS, BACKGROUND, COLOR_TO_ID, FLIP_OFFSETS

    
    

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
        corrected_x = jax.lax.select(
            flip_horizontal,
            x - flip_offset[0], # offset[0] is width padding - ADD to shift right
            x,
        )
        corrected_y = jax.lax.select(
            flip_vertical,
            y - flip_offset[1], # offset[1] is height padding - ADD to shift down
            y,
        )
        
        # --- 3. Scale and Stamp ---
        # The rest of the function proceeds as before, using the
        # flipped_mask and the corrected_x/y coordinates.
        scaled_x = jnp.round(corrected_x * self.config.width_scaling).astype(jnp.int32)
        scaled_y = jnp.round(corrected_y * self.config.height_scaling).astype(jnp.int32)
        
        target_slice = jax.lax.dynamic_slice(object_raster, (scaled_y, scaled_x), flipped_mask.shape)
        updated_slice = jnp.where(flipped_mask != self.TRANSPARENT_ID, flipped_mask, target_slice).astype(object_raster.dtype)
    
        return jax.lax.dynamic_update_slice(object_raster, updated_slice, (scaled_y, scaled_x))

    @partial(jax.jit, static_argnums=(0,))
    def render_at_clipped(self, object_raster: jnp.ndarray, x: int, y: int, sprite_mask: jnp.ndarray, flip_horizontal: bool = False, flip_vertical: bool = False, flip_offset: jnp.ndarray = jnp.array([0, 0])) -> jnp.ndarray:
        """
        Modified version fo render_at that can clip sprites at the edges of the raster.
        Specifically use this if your sprites clip out of the screen like for example the sharks in seaquest, bullets that can go off-screen, etc.
        """
        # --- 1. & 2. Flipping and Position Correction (Unchanged) ---
        flipped_mask = jax.lax.cond(
            flip_horizontal, lambda m: jnp.flip(m, axis=1), lambda m: m, sprite_mask
        )
        flipped_mask = jax.lax.cond(
            flip_vertical, lambda m: jnp.flip(m, axis=0), lambda m: m, flipped_mask
        )
        corrected_x = jax.lax.select(flip_horizontal, x - flip_offset[0], x)
        corrected_y = jax.lax.select(flip_vertical, y - flip_offset[1], y)

        # --- 3. Scale Coordinates (Unchanged) ---
        scaled_x = jnp.round(corrected_x * self.config.width_scaling).astype(jnp.int32)
        scaled_y = jnp.round(corrected_y * self.config.height_scaling).astype(jnp.int32)
        
        # --- 4. NEW: Create Sprite Coordinate Maps for the Full Raster ---
        # For each pixel on the main raster (self._xx, self._yy), calculate its
        # corresponding coordinate on the sprite mask.
        sprite_coords_y = self._yy - scaled_y
        sprite_coords_x = self._xx - scaled_x
        
        # --- 5. NEW: Sample the Sprite Mask using Inverse Mapping ---
        # map_coordinates will look up the values from flipped_mask at each of the
        # coordinates we just calculated.
        # - order=0: Use nearest-neighbor lookup (no interpolation).
        # - cval=TRANSPARENT_ID: Any coordinate that falls outside the bounds of
        #   the sprite mask (i.e., is off-screen) gets the transparent value.
        sampled_sprite_ids = map_coordinates(
            flipped_mask,
            [sprite_coords_y, sprite_coords_x],
            order=0,
            cval=self.TRANSPARENT_ID
        ).astype(object_raster.dtype)

        # --- 6. NEW: Merge the Sampled Sprite onto the Raster ---
        # Where the sampled map is not transparent, use its value. Otherwise,
        # keep the original raster's value. This works for both on- and off-screen
        # parts of the sprite in a single, vectorized operation.
        return jnp.where(
            sampled_sprite_ids != self.TRANSPARENT_ID,
            sampled_sprite_ids,
            object_raster
        )

    @partial(jax.jit, static_argnums=(0,))
    def render_batch(
        self, 
        object_raster: jnp.ndarray, 
        xs: jnp.ndarray, 
        ys: jnp.ndarray, 
        sprite_masks: jnp.ndarray, 
        flip_horizontals: jnp.ndarray, 
        flip_verticals: jnp.ndarray, 
        flip_offsets: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Stamps a BATCH of object IDs onto a raster efficiently.
        
        Args:
            object_raster: The initial (H, W) background canvas.
            xs: (N,) array of x positions.
            ys: (N,) array of y positions.
            sprite_masks: (N, sprite_h, sprite_w) array of all sprite masks.
            ... other args are also batched (N,) ...
        """
        
        def _render_single_layer_to_transparent(base_canvas_shape, sprite_masks, idx, x, y, flip_horizontal, flip_vertical, flip_offset):
            """
            Renders a single sprite onto a new, full-size transparent canvas.
            This function is designed to be vmapped.
            """
            # 1. Start with a blank canvas
            # (Assuming TRANSPARENT_ID is a known constant, e.g., 0)
            # sprite_mask = sprite_masks[idx]
            branches = [lambda: arr for arr in sprite_masks]
            sprite_mask = jax.lax.switch(
                idx,
                branches
            )
            transparent_canvas = jnp.full(base_canvas_shape, self.TRANSPARENT_ID, dtype=jnp.int32)
            
            # 2. Perform all the flip and position correction logic from render_at
            flipped_mask = jax.lax.cond(
                flip_horizontal, lambda m: jnp.flip(m, axis=1), lambda m: m, sprite_mask
            )
            flipped_mask = jax.lax.cond(
                flip_vertical, lambda m: jnp.flip(m, axis=0), lambda m: m, flipped_mask
            ).astype(jnp.int32)
            corrected_x = jax.lax.cond(
                flip_horizontal, lambda val: val - flip_offset[0], lambda val: val, x
            )
            corrected_y = jax.lax.cond(
                flip_vertical, lambda val: val - flip_offset[1], lambda val: val, y
            )
            
            # 3. Scale and Stamp
            scaled_x = jnp.round(corrected_x * self.config.width_scaling).astype(jnp.int32)
            scaled_y = jnp.round(corrected_y * self.config.height_scaling).astype(jnp.int32)

            # 4. Stamp the mask onto the TRANSPARENT canvas
            # No need for jnp.where or dynamic_slice, just a direct update.
            return jax.lax.dynamic_update_slice(transparent_canvas, flipped_mask, (scaled_y, scaled_x))
        
        # === STEP 1: PARALLEL STAMPING ===
        # Vectorize the single-layer render function across all N sprites.
        # Result is a 3D stack of (N, H, W)
        idxs = jnp.arange(len(sprite_masks))
        all_layers = jax.vmap(
            _render_single_layer_to_transparent,
            in_axes=(None, None, 0, 0, 0, 0, 0, 0)
        )(
            object_raster.shape,
            sprite_masks,
            idxs,
            xs,
            ys,
            flip_horizontals,
            flip_verticals,
            flip_offsets
        )
        
        # Composite one layer onto the background
        def _composite_step(background_canvas, foreground_layer):
            """
            Blends a single foreground layer onto the background.
            
            background_canvas: The canvas so far (from the previous scan step).
            foreground_layer: The new layer to draw on top.
            """
            # Use jnp.where to overlay the new layer.
            # Where the foreground is NOT transparent, use its pixels.
            # Otherwise, keep the background pixels.
            updated_canvas = jnp.where(
                foreground_layer != self.TRANSPARENT_ID, 
                foreground_layer, 
                background_canvas
            )
            
            # The new canvas becomes the "carry" for the next step.
            # The second element (None) is for the "ys" (outputs) of the scan, which we don't need.
            return updated_canvas, None 
            
        # === STEP 2: SEQUENTIAL COMPOSITING ===
        # Use jax.lax.scan to "fold" the layers together.
        # This runs the _composite_step function N times.
        #
        # scan args: (function, initial_carry, inputs_to_iterate_over)
        #
        # final_canvas is the last "carry" value.
        final_canvas, _ = jax.lax.scan(
            _composite_step,
            object_raster.astype(jnp.int32),  # Start with the initial background
            all_layers      # Iterate over the (N, H, W) stack
        )
        
        return final_canvas
    

    # ============= Various sequential rendering functions =============
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
        Renders a horizontal progress bar onto the object raster without creating
        any dynamic-shaped intermediate arrays. This is JIT-friendly because it
        computes boolean masks over the full raster and conditionally writes
        colors based on those masks.
        """
        # --- Scale geometric parameters ---
        scaled_x = jnp.round(x * self.config.width_scaling).astype(jnp.int32)
        scaled_y = jnp.round(y * self.config.height_scaling).astype(jnp.int32)
        scaled_width = jnp.maximum(1, jnp.round(width * self.config.width_scaling)).astype(jnp.int32)
        scaled_height = jnp.maximum(1, jnp.round(height * self.config.height_scaling)).astype(jnp.int32)

        # Compute fill width (clamped) in scaled coordinates
        ratio = jnp.nan_to_num(value / jnp.maximum(1.0, max_value))
        ratio = jnp.clip(ratio, 0.0, 1.0)
        fill_width = jnp.round(ratio * scaled_width).astype(jnp.int32)

        # Full-raster coordinate grids (cached at init)
        xx, yy = self._xx, self._yy

        # Background bar area mask
        bg_mask = (xx >= scaled_x) & (xx < scaled_x + scaled_width) & \
                  (yy >= scaled_y) & (yy < scaled_y + scaled_height)

        # Filled portion mask (subset of background)
        fill_mask = (xx >= scaled_x) & (xx < scaled_x + fill_width) & \
                    (yy >= scaled_y) & (yy < scaled_y + scaled_height)

        # First, paint the background area with default_color_id
        raster_after_bg = jnp.where(bg_mask, jnp.asarray(default_color_id, object_raster.dtype), object_raster)

        # Then, paint the filled portion with color_id
        final_raster = jnp.where(fill_mask, jnp.asarray(color_id, object_raster.dtype), raster_after_bg)

        return final_raster



    # ============= Rendering functions for very specific use cases =============

    @partial(jax.jit, static_argnums=(0,))
    def render_grid_inverse(
        self,
        raster: jnp.ndarray,
        grid_state: jnp.ndarray,
        grid_origin: Tuple[int, int],
        cell_size: Tuple[int, int],
        color_map: jnp.ndarray,
        cell_padding: Tuple[int, int] = (0, 0)
    ) -> jnp.ndarray:
        """
        Renders a uniform grid of objects using the inverse mapping technique.

        This is a highly efficient, parallel method for drawing grids where each
        cell's state is stored in a 2D array.

        This function is useful for:
        - Uniform Grid Structure: All objects (or "cells") are the same size and are arranged in a perfect grid of rows and columns.

        - Axis-Aligned: The grid is not rotated or skewed.

        - State Mapped to Grid: The visibility or type of each object can be stored in a 2D array that directly maps to the grid layout.

        - Simple Color Logic: The color for any cell can be easily looked up, typically based on its row, column, or state.

        Examples:
        - Breakout blocks
        - Pacman pellets

        Args:
            raster: The current (H, W) raster of palette IDs to draw on.
            grid_state: A 2D JAX array representing the state of the grid (e.g., 1 for active, 0 for inactive).
            grid_origin: A tuple (start_x, start_y) of the grid's top-left corner in native game coordinates.
            cell_size: A tuple (cell_width, cell_height) of a single cell in native game coordinates.
            color_map: A 1D JAX array that maps object IDs to palette color IDs.
                            The index of this array corresponds to the object ID found in
                            `grid_state`, and the value at that index is the resulting
                            palette color ID. Conventionally, an object ID of 0 represents
                            an empty cell (i.e. background).
            cell_padding: A tuple (padding_width, padding_height) of the padding around each cell in native game coordinates.
        Returns:
            The updated raster with the grid drawn.
        """
        xx, yy = self._xx, self._yy

        # --- 1. Scale all geometric parameters ---
        start_x = grid_origin[0] * self.config.width_scaling
        start_y = grid_origin[1] * self.config.height_scaling
        cell_w = cell_size[0] * self.config.width_scaling
        cell_h = cell_size[1] * self.config.height_scaling
        pad_w = cell_padding[0] * self.config.width_scaling
        pad_h = cell_padding[1] * self.config.height_scaling

        # --- 2. Update Inverse Mapping Calculation ---
        # The total space occupied by one cell plus its padding
        step_w = cell_w + pad_w
        step_h = cell_h + pad_h
        
        # Calculate which cell index each pixel belongs to
        col_idx = jnp.floor((xx - start_x) / step_w).astype(jnp.int32)
        row_idx = jnp.floor((yy - start_y) / step_h).astype(jnp.int32)

        # --- 3. Update Masks to check if a pixel is INSIDE a cell (not in the padding) ---
        # Find the top-left corner of the cell each pixel belongs to
        cell_origin_x = start_x + col_idx * step_w
        cell_origin_y = start_y + row_idx * step_h
        in_cell_mask = (xx >= cell_origin_x) & (xx < cell_origin_x + cell_w) & \
                    (yy >= cell_origin_y) & (yy < cell_origin_y + cell_h)

        num_rows, num_cols = grid_state.shape
        grid_mask = (row_idx >= 0) & (row_idx < num_rows) & \
                    (col_idx >= 0) & (col_idx < num_cols)

        safe_row = jnp.clip(row_idx, 0, num_rows - 1)
        safe_col = jnp.clip(col_idx, 0, num_cols - 1)
        
        # --- UPGRADE ---
        # Look up the OBJECT ID for each pixel from the grid state.
        object_id_map = grid_state[safe_row, safe_col]
        
        # Use the object ID to look up the final COLOR ID from the new map.
        color_id_map = color_map[object_id_map]
        
        # The mask now checks for any non-empty cell (object ID > 0).
        final_mask = grid_mask & in_cell_mask & (object_id_map > 0)

        return jnp.where(final_mask, color_id_map, raster)

    @partial(jax.jit, static_argnums=(0,))
    def draw_rects(
        self,
        raster: jnp.ndarray,
        positions: jnp.ndarray,
        sizes: jnp.ndarray,
        color_id: int,
    ) -> jnp.ndarray:
        """
        Draws multiple filled rectangles onto the raster using a vectorized approach.

        Args:
            raster: The 2D raster array to draw on.
            positions: An array of (x, y) coordinates for the top-left corner of each rectangle.
            sizes: An array of (width, height) for each rectangle.
            color_id: The palette ID to use for filling the rectangles.

        Returns:
            The modified raster with the rectangles drawn.
        """
        # Use cached coordinate grid matching the raster's dimensions
        xx, yy = self._xx, self._yy

        # Scale geometry from game coordinates to raster coordinates
        width_scale = self.config.width_scaling
        height_scale = self.config.height_scaling
        
        pos_scaled = jnp.stack([
            jnp.round(positions[:, 0] * width_scale),
            jnp.round(positions[:, 1] * height_scale),
        ], axis=1).astype(jnp.int32)
        
        size_scaled = jnp.stack([
            jnp.maximum(1, jnp.round(sizes[:, 0] * width_scale)),
            jnp.maximum(1, jnp.round(sizes[:, 1] * height_scale)),
        ], axis=1).astype(jnp.int32)

        def _create_single_mask(pos, size):
            should_draw = pos[0] != -1 # Common convention to hide objects
            x_start, y_start = pos[0], pos[1]
            width, height = size[0], size[1]
            
            mask = (xx >= x_start) & (xx < x_start + width) & \
                   (yy >= y_start) & (yy < y_start + height)
                   
            return jax.lax.select(should_draw, mask, jnp.zeros_like(mask))

        # Vectorize the mask creation over all rectangles
        all_masks = jax.vmap(_create_single_mask)(pos_scaled, size_scaled)
        
        # Combine all individual masks into one
        combined_mask = jnp.logical_or.reduce(all_masks, axis=0)
        
        return jnp.where(combined_mask, jnp.asarray(color_id, raster.dtype), raster)


    @partial(jax.jit, static_argnums=(0, 4, 5, 6))
    def draw_ladders(
        self,
        raster: jnp.ndarray,
        positions: jnp.ndarray,
        sizes: jnp.ndarray,
        rung_height: int,
        space_height: int,
        color_id: int,
    ) -> jnp.ndarray:
        """
        Draws multiple ladders (rectangles with a repeating rung pattern). Examples include the ladders in the kangaroo game.

        Args:
            raster: The 2D raster array to draw on.
            positions: An array of (x, y) coordinates for the top-left of each ladder.
            sizes: An array of (width, height) for each ladder.
            rung_height: The height of each ladder rung in game coordinates.
            space_height: The height of the space between rungs in game coordinates.
            color_id: The palette ID to use for the rungs.

        Returns:
            The modified raster with the ladders drawn.
        """
        xx, yy = self._xx, self._yy
        
        # Scale geometry
        height_scale = self.config.height_scaling
        width_scale = self.config.width_scaling
        pos_scaled = jnp.round(positions * jnp.array([width_scale, height_scale])).astype(jnp.int32)
        size_scaled = jnp.round(sizes * jnp.array([width_scale, height_scale])).astype(jnp.int32)
        rung_scaled = int(max(1, round(rung_height * height_scale)))
        space_scaled = int(max(1, round(space_height * height_scale)))
        
        def _create_single_ladder_mask(pos, size):
            should_draw = pos[0] != -1
            x_start, y_start = pos[0], pos[1]
            width, height = size[0], size[1]
            
            area_mask = (xx >= x_start) & (xx < x_start + width) & \
                        (yy >= y_start) & (yy < y_start + height)
            
            relative_y = yy - y_start
            pattern_height = rung_scaled + space_scaled
            pattern_mask = (relative_y % pattern_height) < rung_scaled
            
            final_mask = area_mask & pattern_mask
            return jax.lax.select(should_draw, final_mask, jnp.zeros_like(final_mask))

        all_masks = jax.vmap(_create_single_ladder_mask)(pos_scaled, size_scaled)
        combined_mask = jnp.logical_or.reduce(all_masks, axis=0)
        
        return jnp.where(combined_mask, jnp.asarray(color_id, raster.dtype), raster)
    

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