import os
from functools import partial
from typing import Tuple, NamedTuple
import chex
import jax
from jax import tree_util
import jax.numpy as jnp
import jax.random as jrandom

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
# Import the new rendering utils
import jaxatari.rendering.jax_rendering_utils as render_utils
# We need the legacy loader *only* for its .npy loading functions
import jaxatari.rendering.jax_rendering_utils_legacy as jr_legacy


def _create_static_procedural_sprites() -> dict:
    """Creates procedural sprites that don't depend on dynamic values."""
    # Black bar for left gutter (procedural)
    # Using constants: SCREEN_HEIGHT=210, PLAYFIELD_LEFT=8
    black_bar = jnp.zeros((210, 8, 4), dtype=jnp.uint8)
    black_bar = black_bar.at[:, :, 3].set(255)
    return {
        'black_bar': black_bar,
    }

def _get_default_asset_config() -> tuple:
    """
    Returns the default declarative asset manifest for Frostbite.
    Most assets are loaded from files and processed dynamically, so this returns
    only the procedural assets that can be statically defined.
    Kept immutable (tuple of dicts) to fit NamedTuple defaults.
    """
    static_procedural = _create_static_procedural_sprites()
    return (
        {'name': 'black_bar', 'type': 'procedural', 'data': static_procedural['black_bar']},
    )


# ==========================================================================================
# CONSTANTS
# ==========================================================================================

class FrostbiteConstants(NamedTuple):
    """Complete constants"""
    
    # Screen dimensions
    SCREEN_WIDTH: int = 160
    SCREEN_HEIGHT: int = 210
    XMIN: int = 0
    XMAX: int = 160

    # Playfield boundaries (screen is 0..160; playfield is 8..160)
    PLAYFIELD_LEFT: int = 8
    PLAYFIELD_RIGHT: int = 160
    PLAYFIELD_WIDTH: int = 152  # PLAYFIELD_RIGHT - PLAYFIELD_LEFT
    
    # Bailey Y position boundaries
    YMIN_BAILEY: int = 27 + 28 + 1   # Shore Y position
    YMAX_BAILEY: int = 112 + 28  # Arctic Sea Y position
    
    # Ice row Y positions (for rendering the ice blocks)
    ICE_ROW_Y: tuple = (98, 122, 147, 172)
    
    # Bailey X boundaries
    SHORE_X_MIN: int = 8    # Allow Bailey to go all the way to the left edge for safe zone
    SHORE_X_MAX: int = 150  # XMAX - 10
    ICE_X_MIN: int = 8     # On ice blocks (XMIN + 8)
    ICE_X_MAX: int = 150    # XMAX - 10
    
    # Initial Values
    INIT_BAILEY_HORIZ_POS: int = 64
    INIT_POLAR_GRIZZLY_HORIZ_POS: int = 140
    INIT_IGLOO_STATUS: int = 0
    
    # Debug/Testing
    START_LEVEL: int = 1
    
    # Bailey jump offset tables (modified for symmetric jumping)
    BAILEY_JUMP_OFFSETS: tuple = (
        6, 5, 5, 5, 4, 3, 2, 1, 0, 0, 0, 0, -1, -2, -3,
        0,
        2, 2, 1, 0, 0, 0, 0, 0, -1, -2, -3, -4, -5, -6, -9,
        0
    )
    
    
    # Speed reference values
    BAILEY_WALK_SPEED_FRAC: int = 4
    
    # Colors
    COLOR_ICE_WHITE: int = 0x0E
    COLOR_ICE_BLUE: int = 0x98
    
    # Igloo constants
    IGLOO_X: int = 154  # X position of igloo (far right side of screen)
    IGLOO_Y: int = 44   # Y position at top of Bailey's head when on shore
    
    # Game Constants
    MAX_IGLOO_INDEX: int = 15  # Complete igloo has 16 blocks (0-15)
    MAX_EATEN_FISH: int = 12  # Max fish that can be eaten per level
    MAX_RESERVED_LIVES: int = 9  # Maximum reserve lives
    
    # Status Masks
    OBSTACLE_DIR_MASK: int = 0x80
    ICE_BLOCK_DIR_MASK: int = 0x40
    OBSTACLE_TYPE_MASK: int = 0x03
    DEMO_MODE: int = 0x80
    
    # Level Status Flags
    BAILEY_SINKING: int = 0x80
    LEVEL_COMPLETE: int = 0x40
    SWAP_PLAYERS: int = 0x20
    INCREMENT_LEVEL: int = 0x08
    
    # Special Levels
    MAGIC_FISH_LEVEL: int = 20
    POLAR_GRIZZLY_LEVEL: int = 3
    
    # Frame delays (adjusted for 60fps)
    # Phase 1: 4 seconds for igloo blocks = 240 frames
    # Phase 2: Temperature countdown after blocks

    # Collision helpers
    BAILEY_BOUNDING_WIDTH: int = 8
    ICE_WIDE_LEFT_MARGIN: int = -2
    ICE_WIDE_RIGHT_MARGIN: int = -4
    ICE_NARROW_LEFT_MARGIN: int = -2
    ICE_NARROW_RIGHT_MARGIN: int = 0
    ICE_MIN_OVERLAP: int = 1
    ICE_WRAP_OFFSETS: tuple = (-152, 0, 152)  # Use playfield width for wrapping
    ICE_UNUSED_POS: int = -512
    ICE_NARROW_SPACING: int = 16
    ICE_WIDE_SPACING: int = 32
    ICE_BREATH_MIN_LEVEL: int = 5
    ICE_BREATH_BLEND_STEPS: tuple = (
        0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1
    )
    REMOVED_FISH_MASKS: tuple = (3, 5, 6, 3, 3, 1, 5, 4, 6, 6, 4)
    
    # Temperature and lives
    INIT_TEMPERATURE: int = 0x45  # Initial temperature 45°
    INIT_LIVES: int = 3  # Starting reserve lives
    
    # Obstacle constants
    ID_SNOW_GOOSE: int = 0
    ID_FISH: int = 1  # Fish obstacle type
    ID_KING_CRAB: int = 2  # King crab obstacle type
    ID_KILLER_CLAM: int = 3  # Killer clam obstacle type
    
    # Obstacle Y positions (between ice rows)
    OBSTACLE_Y: tuple = (82, 107, 132, 157)
    
    # Spawn and despawn boundaries
    OBSTACLE_SPAWN_LEFT: int = -40  # Spawn outside left edge moving right
    OBSTACLE_SPAWN_RIGHT: int = 168 # Spawn outside right edge moving left
    OBSTACLE_DESPAWN_X: int = 40    # Despawn when 40 pixels off either side
    
    # Stutter movement logic for Level 5+
    OBSTACLE_STUTTER_LEVEL: int = 5
    OBSTACLE_STUTTER_MASK: int = 0x40  # Pause when (frame_count & 0x40) != 0

    # Floating obstacle animation
    FLOATING_OBSTACLE_OFFSETS: tuple = (4, 3, 2, 1, 0, 0, 0, 1, 2, 3, 4, 3, 2, 1, 0, 0)
    FLOATING_OBSTACLE_MAX_OFFSET: int = 4
    FLOATING_OBSTACLE_PHASE_MASK: int = 0x1F  # 5-bit sawtooth mask after shifting frame count
    FLOATING_OBSTACLE_PHASE_SHIFT: int = 3    # Divide frame count by 8 (>> 3)
    OBSTACLE_ANIMATION_MASK_DEFAULT: int = 0x20  # Slower flip for geese/fish/crab (adjusted for halved step rate)
    OBSTACLE_ANIMATION_MASK_CLAM: int = 0x40     # Clams animate even slower (double the default period)

    # Polar grizzly animation map - defines which sprite frame (0 or 1) to use for each animation step
    POLAR_GRIZZLY_ANIM_MAP: tuple = (0, 0, 1, 1, 0, 0, 1, 1)
    
    # Frame delays for level progression
    INIT_DELAY_ACTION_VALUE: int = 240  # 4 seconds for block removal only  
    INCREMENT_SCORE_FRAME_DELAY: int = 15  # 240/16 = 15 frames per block
    TEMP_DECREMENT_DELAY: int = 2  # 2 frames per temperature degree

    # Sprite duplication modes
    SPRITE_SINGLE: int = 0b000
    SPRITE_DOUBLE: int = 0b001
    SPRITE_DOUBLE_SPACED: int = 0b010
    SPRITE_TRIPLE: int = 0b011
    SPRITE_DOUBLE_WIDE: int = 0b100
    SPRITE_SIZE_2X: int = 0b101     # not used for geese here
    SPRITE_TRIPLE_SPACED: int = 0b110
    SPRITE_SIZE_4X: int = 0b111       # not used for geese here

    # Sprite spacing distances in pixels
    SPACING_NARROW: int = 16
    SPACING_MEDIUM: int = 32
    SPACING_WIDE: int = 32
    # Asset config baked into constants (immutable default) for asset overrides
    ASSET_CONFIG: tuple = _get_default_asset_config()


# ==========================================================================================
# GAME STATE
# ==========================================================================================


def _compute_row_segments(
    consts: FrostbiteConstants,
    block_positions: jnp.ndarray,
    block_count: jnp.ndarray,
    fine_index: jnp.ndarray,
    breathing_active: jnp.ndarray,
    row_min_x: jnp.ndarray,   # kept for signature compatibility but no longer used
):
    """
    Return per-segment geometry (positions, widths, mask) for an ice row.
    Breathing is computed relative to each base wide block's own position
    (no dependence on a row anchor), and every returned x is canonicalized
    into [-width, 160) to avoid wrap pops.
    """
    sentinel = jnp.int32(consts.ICE_UNUSED_POS)

    block_positions = block_positions.astype(jnp.int32)
    block_count = block_count.astype(jnp.int32)
    fine_index = (fine_index.astype(jnp.int32)) & 0x0F
    breathing_active = jnp.asarray(breathing_active, dtype=jnp.bool_)

    idx6 = jnp.arange(6, dtype=jnp.int32)
    default_active = (idx6 < block_count).astype(jnp.bool_)
    narrow_w = jnp.int32(12)
    wide_w   = jnp.int32(24)

    default_w = jnp.where(block_count == 6, narrow_w, wide_w)
    default_pos = jnp.where(default_active, block_positions, sentinel)
    default_widths = jnp.where(default_active, default_w, jnp.int32(0))

    # table 0..15 -> 0..7..0  (your ICE_BREATH_BLEND_STEPS)
    blend_table = jnp.array(consts.ICE_BREATH_BLEND_STEPS, dtype=jnp.int32)
    step = blend_table[fine_index].astype(jnp.float32) / jnp.float32(7.0)

    # ----- helpers -----
    def canonicalize(x, width):
        """Keep left edge in [L - width, R) where L=8, R=160, W=152."""
        L = consts.PLAYFIELD_LEFT
        R = consts.PLAYFIELD_RIGHT
        W = consts.PLAYFIELD_WIDTH
        x = jnp.where(x >= R, x - W, x)
        x = jnp.where(x < L - width, x + W, x)
        return x

    base = block_positions[:3]                      # 3 wide blocks
    base_count = jnp.minimum(block_count, 3)
    base_active = (jnp.arange(3, dtype=jnp.int32) < base_count)

    # Fix: Derive right block from left to ensure perfect alignment
    # Pair center for a wide block is at base+12; each narrow is 12 wide => half-width = 6
    center = base + jnp.int32(12)

    # How "hard" the breathing tries to push (pixels). Increased to make seam visible
    amp = jnp.int32(4)

    # Smooth integer offset 0..amp based on the 0..1 step
    offset = jnp.rint(step * amp.astype(jnp.float32)).astype(jnp.int32)

    # Split around center; overlap by 4px when closed to ensure connection
    left = canonicalize(center - jnp.int32(12) - offset, narrow_w)
    right = canonicalize(center - jnp.int32(4) + offset, narrow_w)
    childL, childR = left, right

    # Interleave L/R into 6 entries; widths are 12 for both children
    positions_split = jnp.stack((childL, childR), axis=1).reshape(-1)
    mask_split = jnp.stack((base_active, base_active), axis=1).reshape(-1)
    widths_split = jnp.where(mask_split, narrow_w, jnp.int32(0))

    # No breathing: return blocks as-is (could be 6 narrow or 3 wide)
    def no_breathing():
        # Use the default positions/widths which handles both 6 narrow and 3 wide cases
        return default_pos, default_widths, default_active

    return jax.lax.cond(
        breathing_active,
        lambda: (positions_split, widths_split, mask_split),
        no_breathing
    )
class FrostbiteState(NamedTuple):
    """Game state including Bailey and ice blocks"""
    
    # Frame counter
    frame_count: chex.Array
    
    # Bailey position and movement
    bailey_x: chex.Array
    bailey_y: chex.Array
    bailey_jumping_idx: chex.Array      # 0=not jumping, 1-30 for jump position
    bailey_direction: chex.Array        # 0=right, 1=left (REFLECT state)
    bailey_animation_idx: chex.Array    # 0-1 for walking, 2 for jumping, 3-4 for death
    bailey_alive: chex.Array            # 1=alive, 0=dead (fell in water)
    bailey_death_frame: chex.Array      # Death animation frame counter (0-60)
    bailey_frozen: chex.Array           # 1=died from freezing, 0=normal/water death
    bailey_visible: chex.Array          # 1=visible, 0=invisible (entered igloo)
    bailey_sinking: chex.Array          # 1=sinking through ice, 0=normal
    bailey_landing_status: chex.Array   # %10000000 when landing from jump (for sinking check)
    
    # Speed system (fractional movement)
    bailey_speed_whole: chex.Array      # Walking whole pixels per frame
    bailey_speed_frac: chex.Array       # Walking fractional part (0-15)
    bailey_walk_frac_accumulator: chex.Array  # Walk-mode fractional accumulator
    bailey_jump_speed_whole: chex.Array # Jump whole pixels per frame
    bailey_jump_speed_frac: chex.Array  # Jump fractional part (0-15)
    bailey_frac_accumulator: chex.Array # Jump-mode fractional accumulator
    
    # Input tracking for button press detection
    last_action: chex.Array             # Previous frame's action for press detection
    
    # Ice blocks (4 rows)
    ice_x: chex.Array                   # Shape (4,) - anchor positions (legacy)
    ice_block_positions: chex.Array     # Shape (4, 6) - per-block left edges
    ice_block_counts: chex.Array        # Shape (4,) - active blocks per row (3 or 6)
    ice_directions: chex.Array          # Shape (4,) - 0=right, 1=left
    ice_colors: chex.Array              # Shape (4,) - white/blue
    ice_patterns: chex.Array            # Shape (4,) - pattern types
    ice_speed_whole: chex.Array         # Ice block speed (whole part)
    ice_speed_frac: chex.Array          # Ice block speed (fractional)
    ice_frac_accumulators: chex.Array   # Shape (4,) - fractional accumulators
    ice_dx_last_frame: chex.Array       # Shape (4,) - pixels moved during last frame
    ice_fine_motion_index: chex.Array   # 0-15 for ice wobble phase
    
    # Collision
    bailey_ice_collision_idx: chex.Array # -1=not on ice, 0-3=on that row
    
    # Ice block reset delay
    completed_ice_blocks_delay: chex.Array  # 16 frame countdown when all blocks are blue
    
    # Level (affects speed)
    level: chex.Array
    
    # Score (BCD format)
    score: chex.Array  # 3 bytes: [ten-thousands/thousands, hundreds/tens, ones]
    
    # Igloo building
    building_igloo_idx: chex.Array  # -1 = no blocks, 0-14 = building, 15 = complete
    igloo_entry_status: chex.Array  # 0 = not entering, 0x80 = entering igloo
    
    # Temperature and lives
    temperature: chex.Array  # Current temperature in BCD (0x45 = 45°)
    remaining_lives: chex.Array  # Remaining reserve lives
    
    # Level completion handling
    frame_delay: chex.Array  # Delay counter for level complete animations
    current_level_status: chex.Array  # Full status flags (SINKING|COMPLETE|SWAP|FREEZING|INCREMENT)
    
    # Obstacles (4 max)
    obstacle_x: chex.Array  # Shape (4,) - X positions
    obstacle_y: chex.Array  # Shape (4,) - Y positions
    obstacle_types: chex.Array  # Shape (4,) - Type IDs (0=snow geese, etc.)
    obstacle_directions: chex.Array  # Shape (4,) - 0=right, 1=left
    obstacle_active: chex.Array  # Shape (4,) - 1=active, 0=inactive
    obstacle_speed_whole: chex.Array  # Obstacle speed (whole part)
    obstacle_speed_frac: chex.Array  # Obstacle speed (fractional part)
    obstacle_frac_accumulators: chex.Array  # Shape (4,) - fractional accumulators
    obstacle_duplication_mode: chex.Array  # Shape (4,) - Sprite duplication mode (0..7)
    obstacle_animation_idx: chex.Array # Animation index is now per-obstacle
    obstacle_float_offsets: chex.Array  # Vertical bob offsets for aquatic enemies
    obstacle_dx_last_frame: chex.Array  # Shape (4,) - dx actually applied this frame (for collision)
    
    # State fields to support goose logic
    obstacle_pattern_index: chex.Array      # Determines goose flock size/type
    obstacle_max_copies: chex.Array         # Maximum number of geese in a flock
    obstacle_attributes: chex.Array  # Shape (4,) - Full attribute byte with type and direction
    
    # Obstacle collision tracking
    bailey_obstacle_collision_idx: chex.Array  # -1=no collision, 0-3=collided with that obstacle
    obstacle_collision_index: chex.Array  # For ice direction changes
    
    # Missing state variables
    number_of_fish_eaten: chex.Array  # Track fish eaten (max 12 per level)
    fish_alive_mask: chex.Array  # Shape (4,) bitmask for fish copies (bits 0..2)
    polar_grizzly_x: chex.Array  # Separate position for polar bear
    polar_grizzly_active: chex.Array  # 1=active (level 3+), 0=inactive
    polar_grizzly_direction: chex.Array  # 0=right, 1=left
    polar_grizzly_animation_idx: chex.Array  # Animation frame (countdown 7->0)
    polar_grizzly_frac_accumulator: chex.Array  # Fractional movement accumulator
    bailey_grizzly_collision_value: chex.Array  # Bailey-Grizzly collision flag
    bailey_grizzly_collision_timer: chex.Array  # How many frames bear has been dragging Bailey
    action_button_debounce: chex.Array  # Button press tracking
    game_selection: chex.Array  # Game mode selection
    select_debounce: chex.Array  # Select button debounce
    demo_mode: chex.Array  # 1=demo mode, 0=playing
    igloo_status: chex.Array  # Full igloo status flags
    reserve_lives: chex.Array  # Max 9 reserve lives
    
    # JAX
    rng_key: chex.PRNGKey


# ==========================================================================================
# OBSERVATION AND INFO
# ==========================================================================================

class FrostbiteObservation(NamedTuple):
    """Simple observation"""
    bailey_x: jnp.ndarray
    bailey_y: jnp.ndarray
    bailey_animation: jnp.ndarray
    bailey_jumping: jnp.ndarray
    frame_count: jnp.ndarray


class FrostbiteInfo(NamedTuple):
    """Simple info"""
    level: jnp.ndarray


# ==========================================================================================
# MAIN GAME CLASS
# ==========================================================================================

class JaxFrostbite(JaxEnvironment[FrostbiteState, FrostbiteObservation, FrostbiteInfo, FrostbiteConstants]):
    """Bailey-only Frostbite implementation"""
    
    def __init__(self, consts: FrostbiteConstants = None):
        if consts is None:
            consts = FrostbiteConstants()
        super().__init__(consts)

        # how many simulation ticks to run per external step
        # Default to 2 (equivalent to engine_fps=30)
        self._substeps = 2

        self.renderer = FrostbiteRenderer(self.consts)
        
        # Action set
        self.action_set = [
            Action.NOOP,
            Action.UP,
            Action.RIGHT,
            Action.DOWN,
            Action.LEFT,
            Action.UPRIGHT,
            Action.DOWNRIGHT,
            Action.DOWNLEFT,
            Action.UPLEFT,
            Action.FIRE,
        ]
    
    def _get_point_value_for_level(self, level: jnp.ndarray):
        """Get point value for level × 10 in BCD format"""
        # Level 1 = 10 points (0x0010 BCD)
        # Level 9 = 90 points (0x0090 BCD)
        # Level 9+ = 90 points (stays at 0x0090 BCD)

        points = level * 10
        # Cap at 90 for level 9+
        points = jnp.minimum(points, 90)

        # Convert decimal to BCD format
        hundreds = points // 100
        tens = (points % 100) // 10
        ones = points % 10

        # Pack into 16-bit BCD (we'll use lower 2 bytes)
        return (hundreds << 8) | (tens << 4) | ones
    
    def _get_obstacle_pattern_mask(self, level: jnp.ndarray):
        """Get pattern mask to control obstacle density based on level.

        Args:
            level: Current game level

        Returns:
            Pattern mask value that affects obstacle spawn patterns
        """
        # This affects obstacle density and spawn patterns
        mask_table = jnp.array([6, 4, 0, 0, 0, 0, 1, 3, 7], dtype=jnp.int32)
        
        # Clamp level index to table size (9+)
        level_idx = jnp.minimum(level - 1, 8)  # 0-based index, capped at 8
        
        # Simply index into the table
        return mask_table[level_idx]
    
    def _calculate_speeds_for_level(self, level: jnp.ndarray):
        """Calculate per-entity speeds with a 7-level sawtooth ramp."""
        level_idx = jnp.maximum(level - 1, 0)

        # Base speed
        # 7-level sawtooth: every time the raw speed reaches 15 + 7 * k we
        # reduce it by 3 to keep the pacing in sync.
        S_raw = level_idx + 4
        excess = S_raw - 15
        reductions = jnp.where(
            excess >= 0,
            jnp.floor_divide(excess, 7) + 1,
            jnp.zeros_like(excess),
        )
        S = jnp.maximum(S_raw - (reductions * 3), 0)

        # Ice speed: (S >> 1) in fractional units
        ice_speed_raw = S >> 1
        ice_speed_whole = ice_speed_raw // 16
        ice_speed_frac = ice_speed_raw % 16

        # Obstacle speed: (S - 1) in fractional units
        obstacle_speed_raw = jnp.maximum(S - 1, 0)
        obstacle_speed_whole = obstacle_speed_raw // 16
        obstacle_speed_frac = obstacle_speed_raw % 16

        # Bailey jump speed: S in fractional units
        bailey_jump_speed_raw = S
        bailey_jump_speed_whole = bailey_jump_speed_raw // 16
        bailey_jump_speed_frac = bailey_jump_speed_raw % 16
        
        # Bailey walk speed: (4/16 px per frame). Fractional math handles the sub-pixel steps.
        bailey_walk_speed_whole = 0
        bailey_walk_speed_frac = self.consts.BAILEY_WALK_SPEED_FRAC
        
        return {
            'ice_speed_whole': ice_speed_whole,
            'ice_speed_frac': ice_speed_frac,
            'obstacle_speed_whole': obstacle_speed_whole,
            'obstacle_speed_frac': obstacle_speed_frac,
            'bailey_jump_speed_whole': bailey_jump_speed_whole,
            'bailey_jump_speed_frac': bailey_jump_speed_frac,
            'bailey_walk_speed_whole': bailey_walk_speed_whole,
            'bailey_walk_speed_frac': bailey_walk_speed_frac
        }

    def _determine_ice_layout(self, level: jnp.ndarray, fine_motion_index: jnp.ndarray):
        """Return (use_narrow, block_count, block_width, spacing) for the current layout."""
        is_odd_level = (level % 2) == 1
        base_use_narrow = jnp.logical_not(is_odd_level)

        is_level_5_plus = level >= self.consts.ICE_BREATH_MIN_LEVEL
        is_breathing_level = jnp.logical_and(is_odd_level, is_level_5_plus)
        use_narrow = jnp.where(is_breathing_level, False, base_use_narrow)
        block_count = jnp.where(use_narrow, jnp.int32(6), jnp.int32(3))
        block_width = jnp.where(use_narrow, jnp.int32(12), jnp.int32(24))
        spacing = jnp.where(
            use_narrow,
            jnp.int32(self.consts.ICE_NARROW_SPACING),
            jnp.int32(self.consts.ICE_WIDE_SPACING)
        )
        return use_narrow, block_count, block_width, spacing

    def _init_block_positions(self, ice_x: jnp.ndarray, level: jnp.ndarray, fine_motion_index: jnp.ndarray):
        """Compute per-block positions for all rows and return (positions, counts)."""
        _, block_count, _, spacing = self._determine_ice_layout(level, fine_motion_index)
        block_indices = jnp.arange(6, dtype=jnp.int32)
        base_positions = ice_x[:, None] + block_indices * spacing
        sentinel = jnp.int32(self.consts.ICE_UNUSED_POS)
        mask = block_indices < block_count
        positions = jnp.where(mask[None, :], base_positions, sentinel)
        counts = jnp.full((ice_x.shape[0],), block_count, dtype=jnp.int32)
        return positions, counts

    def _get_row_segments(self, state: FrostbiteState, row_idx: int):
        block_positions = state.ice_block_positions[row_idx]
        block_count = state.ice_block_counts[row_idx]
        breathing_active = (
            (state.level >= self.consts.ICE_BREATH_MIN_LEVEL) &
            ((state.level & 1) == 1) &
            (block_count <= 3)
        )
        return _compute_row_segments(
            self.consts,
            block_positions,
            block_count,
            state.ice_fine_motion_index,
            breathing_active,
            state.ice_x[row_idx]
        )
    
    def reset(self, key: jax.random.PRNGKey = None) -> Tuple[FrostbiteObservation, FrostbiteState]:
        """Initialize game state"""
        
        if key is None:
            key = jrandom.PRNGKey(0)
        
        # Start at configured level (for testing)
        level = jnp.array(self.consts.START_LEVEL, dtype=jnp.int32)
        
        # Calculate level-dependent speeds
        speeds = self._calculate_speeds_for_level(level)
        
        ice_x = jnp.array([68, -12, 68, -12], dtype=jnp.int32)
        ice_directions = jnp.array([1, 0, 1, 0], dtype=jnp.int32)
        ice_colors = jnp.array([self.consts.COLOR_ICE_WHITE] * 4, dtype=jnp.int32)
        ice_patterns = jnp.array([3, 3, 3, 3], dtype=jnp.int32)
        
        # Ice fine motion phase: odd levels start at 0, even levels start at 8
        ice_fine_motion_index = jnp.where(level % 2 == 1, 0, 8)

        ice_block_positions, ice_block_counts = self._init_block_positions(ice_x, level, ice_fine_motion_index)

        # Create a minimal state just for spawning obstacles
        initial_state_for_spawn = FrostbiteState(
            # Frame counter
            frame_count=jnp.array(0, dtype=jnp.int32),
            # Bailey fields (dummy values for spawn)
            bailey_x=jnp.array(self.consts.INIT_BAILEY_HORIZ_POS, dtype=jnp.int32),
            bailey_y=jnp.array(self.consts.YMIN_BAILEY, dtype=jnp.int32),
            bailey_jumping_idx=jnp.array(0, dtype=jnp.int32),
            bailey_direction=jnp.array(0, dtype=jnp.int32),
            bailey_animation_idx=jnp.array(0, dtype=jnp.int32),
            bailey_alive=jnp.array(1, dtype=jnp.int32),
            bailey_death_frame=jnp.array(0, dtype=jnp.int32),
            bailey_frozen=jnp.array(0, dtype=jnp.int32),
            bailey_visible=jnp.array(1, dtype=jnp.int32),
            bailey_sinking=jnp.array(0, dtype=jnp.int32),  # NEW
            bailey_landing_status=jnp.array(0, dtype=jnp.int32),  # NEW
            bailey_speed_whole=speeds['bailey_walk_speed_whole'],
            bailey_speed_frac=speeds['bailey_walk_speed_frac'],
            bailey_walk_frac_accumulator=jnp.array(0, dtype=jnp.int32),
            bailey_jump_speed_whole=speeds['bailey_jump_speed_whole'],
            bailey_jump_speed_frac=speeds['bailey_jump_speed_frac'],
            bailey_frac_accumulator=jnp.array(0, dtype=jnp.int32),
            last_action=jnp.array(0, dtype=jnp.int32),
            # Ice fields (dummy values for spawn)
            ice_x=ice_x,
            ice_block_positions=ice_block_positions,
            ice_block_counts=ice_block_counts,
            ice_directions=ice_directions,
            ice_colors=ice_colors,
            ice_patterns=ice_patterns,
            ice_speed_whole=speeds['ice_speed_whole'],
            ice_speed_frac=speeds['ice_speed_frac'],
            ice_frac_accumulators=jnp.zeros(4, dtype=jnp.int32),
            ice_dx_last_frame=jnp.zeros(4, dtype=jnp.int32),
            ice_fine_motion_index=ice_fine_motion_index,
            bailey_ice_collision_idx=jnp.array(-1, dtype=jnp.int32),
            completed_ice_blocks_delay=jnp.array(0, dtype=jnp.int32),
            # Level and score
            level=level,
            score=jnp.zeros(3, dtype=jnp.int32),
            building_igloo_idx=jnp.array(-1, dtype=jnp.int32),
            igloo_entry_status=jnp.array(0, dtype=jnp.int32),
            temperature=jnp.array(self.consts.INIT_TEMPERATURE, dtype=jnp.int32),
            remaining_lives=jnp.array(self.consts.INIT_LIVES, dtype=jnp.int32),
            frame_delay=jnp.array(0, dtype=jnp.int32),
            current_level_status=jnp.array(0, dtype=jnp.int32),
            # Obstacle fields (these will be set by spawn)
            obstacle_x=jnp.zeros(4, dtype=jnp.int32),
            obstacle_y=jnp.array(self.consts.OBSTACLE_Y, dtype=jnp.int32),
            obstacle_types=jnp.full(4, self.consts.ID_SNOW_GOOSE, dtype=jnp.int32),
            obstacle_directions=jnp.zeros(4, dtype=jnp.int32),
            obstacle_active=jnp.zeros(4, dtype=jnp.int32),
            obstacle_speed_whole=speeds['obstacle_speed_whole'],
            obstacle_speed_frac=speeds['obstacle_speed_frac'],
            obstacle_frac_accumulators=jnp.zeros(4, dtype=jnp.int32),
            obstacle_duplication_mode=jnp.zeros(4, dtype=jnp.int32),
            obstacle_animation_idx=jnp.zeros(4, dtype=jnp.int32),
            obstacle_float_offsets=jnp.zeros(4, dtype=jnp.int32),
            obstacle_dx_last_frame=jnp.zeros(4, dtype=jnp.int32),
            obstacle_pattern_index=jnp.zeros(4, dtype=jnp.int32),
            obstacle_max_copies=jnp.zeros(4, dtype=jnp.int32),
            obstacle_attributes=jnp.zeros(4, dtype=jnp.int32),  # NEW
            bailey_obstacle_collision_idx=jnp.array(-1, dtype=jnp.int32),
            obstacle_collision_index=jnp.array(-1, dtype=jnp.int32),
            # State variables
            number_of_fish_eaten=jnp.array(0, dtype=jnp.int32),
            fish_alive_mask=jnp.zeros(4, dtype=jnp.int32),
            polar_grizzly_x=jnp.array(self.consts.INIT_POLAR_GRIZZLY_HORIZ_POS, dtype=jnp.int32),
            polar_grizzly_active=jnp.array(0, dtype=jnp.int32),
            polar_grizzly_direction=jnp.array(1, dtype=jnp.int32),
            polar_grizzly_animation_idx=jnp.array(7, dtype=jnp.int32),  # Start countdown at 7
            polar_grizzly_frac_accumulator=jnp.array(0, dtype=jnp.int32),
            bailey_grizzly_collision_value=jnp.array(0, dtype=jnp.int32),
            bailey_grizzly_collision_timer=jnp.array(0, dtype=jnp.int32),
            action_button_debounce=jnp.array(0, dtype=jnp.int32),
            game_selection=jnp.array(0, dtype=jnp.int32),
            select_debounce=jnp.array(0, dtype=jnp.int32),
            demo_mode=jnp.array(0, dtype=jnp.int32),  # Included but not used
            igloo_status=jnp.array(self.consts.INIT_IGLOO_STATUS, dtype=jnp.int32),
            reserve_lives=jnp.array(self.consts.INIT_LIVES, dtype=jnp.int32),
            rng_key=key
        )
        # Spawn all 4 obstacles at the start
        # _spawn_obstacles_vec updates rng_key internally
        state_after_spawn = self._spawn_obstacles_vec(initial_state_for_spawn, jnp.ones((4,), dtype=jnp.bool_))

        obstacle_x = state_after_spawn.obstacle_x
        obstacle_directions = state_after_spawn.obstacle_directions
        obstacle_pattern_index = state_after_spawn.obstacle_pattern_index
        obstacle_max_copies = state_after_spawn.obstacle_max_copies
        obstacle_types = state_after_spawn.obstacle_types  # Use the spawned types!
        fish_alive_mask = state_after_spawn.fish_alive_mask  # Get fish masks from spawn
        key = state_after_spawn.rng_key
        
        obstacle_y = jnp.array(self.consts.OBSTACLE_Y, dtype=jnp.int32)
        
        state = FrostbiteState(
            frame_count=jnp.array(0, dtype=jnp.int32),
            bailey_x=jnp.array(self.consts.INIT_BAILEY_HORIZ_POS, dtype=jnp.int32),
            bailey_y=jnp.array(self.consts.YMIN_BAILEY, dtype=jnp.int32),
            bailey_jumping_idx=jnp.array(0, dtype=jnp.int32),
            bailey_direction=jnp.array(0, dtype=jnp.int32),
            bailey_animation_idx=jnp.array(0, dtype=jnp.int32),
            bailey_alive=jnp.array(1, dtype=jnp.int32),
            bailey_death_frame=jnp.array(0, dtype=jnp.int32),
            bailey_frozen=jnp.array(0, dtype=jnp.int32),
            bailey_visible=jnp.array(1, dtype=jnp.int32),
            bailey_sinking=jnp.array(0, dtype=jnp.int32),  # NEW
            bailey_landing_status=jnp.array(0, dtype=jnp.int32),  # NEW
            bailey_speed_whole=speeds['bailey_walk_speed_whole'],
            bailey_speed_frac=speeds['bailey_walk_speed_frac'],
            bailey_walk_frac_accumulator=jnp.array(0, dtype=jnp.int32),
            bailey_jump_speed_whole=speeds['bailey_jump_speed_whole'],
            bailey_jump_speed_frac=speeds['bailey_jump_speed_frac'],
            bailey_frac_accumulator=jnp.array(0, dtype=jnp.int32),
            last_action=jnp.array(0, dtype=jnp.int32),
            ice_x=ice_x,
            ice_block_positions=ice_block_positions,
            ice_block_counts=ice_block_counts,
            ice_directions=ice_directions,
            ice_colors=ice_colors,
            ice_patterns=ice_patterns,
            ice_speed_whole=speeds['ice_speed_whole'],
            ice_speed_frac=speeds['ice_speed_frac'],
            ice_frac_accumulators=jnp.zeros(4, dtype=jnp.int32),
            ice_dx_last_frame=jnp.zeros(4, dtype=jnp.int32),
            ice_fine_motion_index=ice_fine_motion_index,
            bailey_ice_collision_idx=jnp.array(-1, dtype=jnp.int32),
            completed_ice_blocks_delay=jnp.array(0, dtype=jnp.int32),
            level=level,
            score=jnp.zeros(3, dtype=jnp.int32),
            building_igloo_idx=jnp.array(-1, dtype=jnp.int32),
            igloo_entry_status=jnp.array(0, dtype=jnp.int32),
            temperature=jnp.array(self.consts.INIT_TEMPERATURE, dtype=jnp.int32),
            remaining_lives=jnp.array(self.consts.INIT_LIVES, dtype=jnp.int32),
            frame_delay=jnp.array(0, dtype=jnp.int32),
            current_level_status=jnp.array(0, dtype=jnp.int32),
            obstacle_x=obstacle_x,
            obstacle_y=obstacle_y,
            obstacle_types=obstacle_types,
            obstacle_directions=obstacle_directions,
            obstacle_active=jnp.ones(4, dtype=jnp.int32), # Start active
            obstacle_speed_whole=speeds['obstacle_speed_whole'],
            obstacle_speed_frac=speeds['obstacle_speed_frac'],
            obstacle_frac_accumulators=jnp.zeros(4, dtype=jnp.int32),
            obstacle_duplication_mode=jnp.zeros(4, dtype=jnp.int32), # Will be set in update
            obstacle_animation_idx=jnp.zeros(4, dtype=jnp.int32),
            obstacle_float_offsets=jnp.zeros(4, dtype=jnp.int32),
            obstacle_dx_last_frame=jnp.zeros(4, dtype=jnp.int32),
            obstacle_pattern_index=obstacle_pattern_index,
            obstacle_max_copies=obstacle_max_copies,
            obstacle_attributes=jnp.zeros(4, dtype=jnp.int32),  # NEW
            bailey_obstacle_collision_idx=jnp.array(-1, dtype=jnp.int32),
            obstacle_collision_index=jnp.array(-1, dtype=jnp.int32),
            # State variables
            number_of_fish_eaten=jnp.array(0, dtype=jnp.int32),
            fish_alive_mask=fish_alive_mask,
            polar_grizzly_x=jnp.array(self.consts.INIT_POLAR_GRIZZLY_HORIZ_POS, dtype=jnp.int32),
            polar_grizzly_active=jnp.where(
                (level - 1) >= self.consts.POLAR_GRIZZLY_LEVEL,
                1,
                0
            ),  # Active at level 4 in actual gameplay
            polar_grizzly_direction=jnp.array(1, dtype=jnp.int32),
            polar_grizzly_animation_idx=jnp.array(7, dtype=jnp.int32),  # Start countdown at 7
            polar_grizzly_frac_accumulator=jnp.array(0, dtype=jnp.int32),
            bailey_grizzly_collision_value=jnp.array(0, dtype=jnp.int32),
            bailey_grizzly_collision_timer=jnp.array(0, dtype=jnp.int32),
            action_button_debounce=jnp.array(0, dtype=jnp.int32),
            game_selection=jnp.array(0, dtype=jnp.int32),
            select_debounce=jnp.array(0, dtype=jnp.int32),
            demo_mode=jnp.array(0, dtype=jnp.int32),  # Included but not used
            igloo_status=jnp.array(self.consts.INIT_IGLOO_STATUS, dtype=jnp.int32),
            reserve_lives=jnp.array(self.consts.INIT_LIVES, dtype=jnp.int32),
            rng_key=key
        )
        
        obs = self._get_observation(state)
        return obs, state
    
    def _get_observation(self, state: FrostbiteState) -> FrostbiteObservation:
        """Convert state to observation"""
        return FrostbiteObservation(
            bailey_x=state.bailey_x,
            bailey_y=state.bailey_y,
            bailey_animation=state.bailey_animation_idx,
            bailey_jumping=jnp.int32(state.bailey_jumping_idx > 0),
            frame_count=state.frame_count
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def _decode_sprite_duplication(self, code: jnp.ndarray):
        """Decode sprite duplication mode to get number of copies and spacing.

        Args:
            code: Duplication mode code (0-7)

        Returns:
            Tuple of (number_of_copies, spacing_in_pixels)
        """
        close = self.consts.SPACING_NARROW
        med   = self.consts.SPACING_MEDIUM
        wide  = self.consts.SPACING_WIDE

        # Defaults
        copies = jnp.array(1, dtype=jnp.int32)
        spacing = jnp.array(0, dtype=jnp.int32)

        # Map codes used for geese
        copies = jnp.where(code == self.consts.SPRITE_SINGLE,           1, copies)
        spacing = jnp.where(code == self.consts.SPRITE_SINGLE,          0, spacing)

        copies = jnp.where(code == self.consts.SPRITE_DOUBLE,         2, copies)
        spacing = jnp.where(code == self.consts.SPRITE_DOUBLE,        close, spacing)

        copies = jnp.where(code == self.consts.SPRITE_DOUBLE_SPACED,     2, copies)
        spacing = jnp.where(code == self.consts.SPRITE_DOUBLE_SPACED,    med, spacing)

        copies = jnp.where(code == self.consts.SPRITE_DOUBLE_WIDE,    2, copies)
        spacing = jnp.where(code == self.consts.SPRITE_DOUBLE_WIDE,   wide, spacing)

        copies = jnp.where(code == self.consts.SPRITE_TRIPLE,       3, copies)
        spacing = jnp.where(code == self.consts.SPRITE_TRIPLE,      close, spacing)

        copies = jnp.where(code == self.consts.SPRITE_TRIPLE_SPACED,   3, copies)
        spacing = jnp.where(code == self.consts.SPRITE_TRIPLE_SPACED,  med, spacing)

        return copies, spacing
    
    # Helper function to spawn a single obstacle
    def _spawn_obstacle(self, state: FrostbiteState, obstacle_idx: int) -> FrostbiteState:
        """Spawn a new obstacle with type based on level."""
        # --- pick 0..7 ---
        rand_key, next_key = jrandom.split(state.rng_key)
        rand_val = jrandom.randint(rand_key, (), 0, 256, dtype=jnp.int32)

        # 0-based currentLevel (on-screen level - 1)
        level_idx = jnp.maximum(state.level - 1, 0)

        # Clamp random 0..7 to ≤ min(currentLevel, 7)
        level_cap = jnp.minimum(level_idx, 7)
        raw_type = rand_val & 7
        raw_type = jnp.minimum(raw_type, level_cap)

        # If 12 fish eaten in this level, bump any fish roll to crab (increment temp)
        fish_exhausted = state.number_of_fish_eaten >= self.consts.MAX_EATEN_FISH
        would_be_fish = ((raw_type & self.consts.OBSTACLE_TYPE_MASK) == self.consts.ID_FISH)
        raw_type = jnp.where(fish_exhausted & would_be_fish, raw_type + 1, raw_type)

        # Final obstacle type = low 2 bits
        new_type = raw_type & self.consts.OBSTACLE_TYPE_MASK

        # Direction: XOR previous direction with random bit 7
        previous_attr = state.obstacle_attributes[obstacle_idx]
        preserved_bits = previous_attr & (self.consts.OBSTACLE_DIR_MASK | self.consts.ICE_BLOCK_DIR_MASK)
        tmp_attr = preserved_bits | new_type
        direction_flip = rand_val & self.consts.OBSTACLE_DIR_MASK
        new_attr = tmp_attr ^ direction_flip
        new_dir = (new_attr >> 7) & 1

        # Spawn position depends on travel direction
        new_x = jnp.where(new_dir == 0, self.consts.OBSTACLE_SPAWN_LEFT, self.consts.OBSTACLE_SPAWN_RIGHT)

        # Special case: if level == POLAR_GRIZZLY_LEVEL (3), use index 0
        # Otherwise: use (level & 3) as index
        pattern_table_idx = jnp.where(
            level_idx == self.consts.POLAR_GRIZZLY_LEVEL,
            0,  # Special case for polar bear level
            level_idx & 3  # Normal case: level mod 4
        )
        
        pattern_indexes = jnp.array([1, 5, 5, 7], dtype=jnp.int32)
        new_pattern_index = pattern_indexes[pattern_table_idx]
        
        # Pattern index determines number of copies for ALL enemy types
        # Pattern 1: 1 copy (single enemy)
        # Pattern 5: 2 copies (two enemies with medium spacing)
        # Pattern 7: 3 copies (three enemies with close spacing)
        max_copies = jnp.where(
            new_pattern_index == 1,
            1,
            jnp.where(
                new_pattern_index == 5,
                2,
                3  # pattern 7
            )
        )

        # Bitmask of alive fish copies: 1, 3, or 7 for 1/2/3 fish.
        initial_fish_mask = jnp.where(new_type == self.consts.ID_FISH,
                                      (jnp.int32(1) << max_copies) - 1,
                                      jnp.int32(0))

        state = state._replace(
            obstacle_x=state.obstacle_x.at[obstacle_idx].set(new_x),
            obstacle_types=state.obstacle_types.at[obstacle_idx].set(new_type),
            obstacle_directions=state.obstacle_directions.at[obstacle_idx].set(new_dir),
            obstacle_active=state.obstacle_active.at[obstacle_idx].set(1),
            obstacle_pattern_index=state.obstacle_pattern_index.at[obstacle_idx].set(new_pattern_index),
            obstacle_max_copies=state.obstacle_max_copies.at[obstacle_idx].set(max_copies),
            obstacle_attributes=state.obstacle_attributes.at[obstacle_idx].set(new_attr),
            obstacle_frac_accumulators=state.obstacle_frac_accumulators.at[obstacle_idx].set(0),
            obstacle_dx_last_frame=state.obstacle_dx_last_frame.at[obstacle_idx].set(0),
            fish_alive_mask=state.fish_alive_mask.at[obstacle_idx].set(initial_fish_mask),
            rng_key=next_key
        )
        return state

    @partial(jax.jit, static_argnums=(0,))
    def _spawn_obstacles_vec(self, state: FrostbiteState, spawn_mask: jnp.ndarray) -> FrostbiteState:
        """Vectorized obstacle spawn for 4 lanes.

        spawn_mask: shape (4,), bool/int (1=respawn this index, 0=leave as-is)
        Keeps RNG semantics correct and matches _spawn_obstacle behavior closely.
        """
        n = 4
        spawn_mask = spawn_mask.astype(jnp.bool_)

        # One split; use 'key' to draw an (n,) vector and carry 'new_key' forward.
        key, new_key = jrandom.split(state.rng_key)
        rand_vals = jrandom.randint(key, (n,), 0, 256, dtype=jnp.int32)

        level_idx = jnp.maximum(state.level - 1, 0)        # scalar
        level_cap = jnp.minimum(level_idx, 7)              # scalar

        raw_type = jnp.minimum(rand_vals & 7, level_cap)   # (n,)
        fish_exhausted = state.number_of_fish_eaten >= self.consts.MAX_EATEN_FISH
        raw_type = jnp.where(
            fish_exhausted & (((raw_type & self.consts.OBSTACLE_TYPE_MASK) == self.consts.ID_FISH)),
            raw_type + 1,
            raw_type
        )
        new_type = raw_type & self.consts.OBSTACLE_TYPE_MASK  # (n,)

        preserved_bits = state.obstacle_attributes & (self.consts.OBSTACLE_DIR_MASK | self.consts.ICE_BLOCK_DIR_MASK)
        new_attr = (preserved_bits | new_type) ^ (rand_vals & self.consts.OBSTACLE_DIR_MASK)
        new_dir = (new_attr >> 7) & 1
        new_x  = jnp.where(new_dir == 0, self.consts.OBSTACLE_SPAWN_LEFT, self.consts.OBSTACLE_SPAWN_RIGHT)

        # Pattern index selection (scalar -> broadcast), same logic as _spawn_obstacle
        pattern_table_idx = jnp.where(
            level_idx == self.consts.POLAR_GRIZZLY_LEVEL,
            0,
            level_idx & 3
        )
        pattern_indexes = jnp.array([1, 5, 5, 7], dtype=jnp.int32)
        pat = pattern_indexes[pattern_table_idx]  # scalar

        # max copies derived from pattern
        max_copies_scalar = jnp.where(pat == 1, 1, jnp.where(pat == 5, 2, 3))
        max_copies = jnp.full((n,), max_copies_scalar, dtype=jnp.int32)

        # Fish alive bitmask per obstacle
        initial_fish_mask = jnp.where(
            new_type == self.consts.ID_FISH,
            (jnp.int32(1) << max_copies) - 1,   # 1, 3, or 7
            jnp.int32(0)
        )

        # Apply per-lane updates only where spawn_mask is True
        def sel(old, new): return jnp.where(spawn_mask, new, old)

        state = state._replace(
            obstacle_x=sel(state.obstacle_x, new_x),
            obstacle_types=sel(state.obstacle_types, new_type),
            obstacle_directions=sel(state.obstacle_directions, new_dir),
            obstacle_active=sel(state.obstacle_active, jnp.ones((n,), dtype=jnp.int32)),
            obstacle_pattern_index=sel(state.obstacle_pattern_index, jnp.full((n,), pat, dtype=jnp.int32)),
            obstacle_max_copies=sel(state.obstacle_max_copies, max_copies),
            obstacle_attributes=sel(state.obstacle_attributes, new_attr),
            obstacle_frac_accumulators=sel(state.obstacle_frac_accumulators, jnp.zeros((n,), dtype=jnp.int32)),
            obstacle_dx_last_frame=sel(state.obstacle_dx_last_frame, jnp.zeros((n,), dtype=jnp.int32)),
            fish_alive_mask=sel(state.fish_alive_mask, initial_fish_mask),
            rng_key=new_key
        )
        return state

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: FrostbiteState, state: FrostbiteState) -> chex.Array:
        """Calculate reward based on score difference."""
        # Convert BCD scores to decimal values
        prev_score_val = self._bcd_to_decimal(previous_state.score)
        curr_score_val = self._bcd_to_decimal(state.score)
        return (curr_score_val - prev_score_val).astype(jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: FrostbiteState) -> chex.Array:
        """Check if game is over."""
        frozen_death_complete = (state.temperature == 0) & (state.bailey_death_frame >= 120)
        final_death_complete = (state.bailey_death_frame >= 120) & (state.remaining_lives == 0)
        return frozen_death_complete | final_death_complete

    def _bcd_to_decimal(self, bcd_score: jnp.ndarray) -> jnp.ndarray:
        """Convert 3-byte BCD score to decimal value."""
        # Each byte contains two BCD digits - convert in single expression
        # This is more JIT-friendly than sequential operations
        return (((bcd_score[0] >> 4) & 0xF) * 100000 +
                (bcd_score[0] & 0xF) * 10000 +
                ((bcd_score[1] >> 4) & 0xF) * 1000 +
                (bcd_score[1] & 0xF) * 100 +
                ((bcd_score[2] >> 4) & 0xF) * 10 +
                (bcd_score[2] & 0xF)).astype(jnp.int32)

    def _get_info(self, state: FrostbiteState) -> FrostbiteInfo:
        """Get game info"""
        return FrostbiteInfo(level=state.level)

    @partial(jax.jit, static_argnums=(0,))
    def _step_once(self, state: FrostbiteState, action: int) -> tuple:
        """Execute one game step, updating all game entities and state.

        This is the main game loop that processes player input, updates all
        game objects, handles collisions, and manages game progression.

        Args:
            state: Current game state
            action: Player action (movement/jump)

        Returns:
            Tuple of (observation, new_state, reward, done, info)
        """
        # Save original state for reward computation
        prev_state = state

        # Increment frame counter for timing-based events
        state = state._replace(frame_count=state.frame_count + 1)

        # Ice wobble animation for high odd-numbered levels
        # Creates a visual "breathing" effect on ice blocks
        should_drift = (
            (state.level >= self.consts.ICE_BREATH_MIN_LEVEL) &  # Level 7+
            (state.level % 2 == 1) &  # Odd levels only
            ((state.frame_count & 7) == 0)  # Update every 8 frames
        )
        new_fine_motion_index = jnp.where(
            should_drift,
            (state.ice_fine_motion_index - 1) % 16,  # Cycle through 16 animation frames
            state.ice_fine_motion_index
        )
        state = state._replace(ice_fine_motion_index=new_fine_motion_index)

        # Core game update sequence
        state = self._process_level_complete(state)  # Handle level completion animations
        state = self._update_ice_blocks(state, action)  # Move ice and handle player landing

        # Update all obstacles (fish, crabs, clams, birds)
        state = self._update_obstacles(state)

        state = self._check_obstacle_collisions(state)  # Check Bailey-obstacle collisions
        state = self._update_bailey(state, action)  # Update player position and state
        state = self._update_polar_grizzly(state)  # Update bear behavior

        def decrement_bcd(temp):
            """Decrement temperature in BCD (Binary Coded Decimal) format.

            BCD stores each decimal digit in 4 bits, so 45° is stored as 0x45.
            This function properly handles the decimal carry when decrementing.
            """
            ones = temp & 0x0F  # Extract ones digit
            tens = (temp >> 4) & 0x0F  # Extract tens digit

            # Handle BCD borrow: if ones is 0, borrow from tens and set ones to 9
            new_ones = jax.lax.cond(ones == 0, lambda: 9, lambda: ones - 1)
            new_tens = jax.lax.cond(ones == 0, lambda: jnp.maximum(0, tens - 1), lambda: tens)

            # Combine digits back to BCD
            result = (new_tens << 4) | new_ones

            # Return 0 if already at 0
            return jax.lax.cond((tens == 0) & (ones == 0), lambda: 0, lambda: result)

        # Temperature decreases every ~2.17 seconds (130 frames at 60 FPS)
        should_decrease_temp = (state.frame_count % 130) == 0

        # Only decrease temperature during active gameplay
        is_playing = (state.bailey_death_frame == 0)  # Not in death animation
        is_entering_igloo = (state.igloo_entry_status == 0x80)  # Not entering igloo
        level_complete = (state.current_level_status == self.consts.LEVEL_COMPLETE)  # Not completing level

        new_temperature = jax.lax.cond(
            should_decrease_temp & is_playing & ~is_entering_igloo & ~level_complete,
            lambda t: decrement_bcd(t), lambda t: t, state.temperature
        )
        state = state._replace(temperature=new_temperature)

        # Check for collisions and special conditions
        state = self._check_collisions(state)  # Handle Bailey-ice collisions
        state = self._check_igloo_entry(state, action)  # Check if Bailey enters igloo

        # Calculate reward and done using dedicated methods
        reward = self._get_reward(prev_state, state)
        done = self._get_done(state)

        obs = self._get_observation(state)
        info = self._get_info(state)

        return obs, state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: FrostbiteState, action: int):
        """Framesafe wrapper: run the 60 Hz simulation multiple times per engine tick."""
        def body(carry, _):
            st, reward_sum = carry
            obs, st, r, done, info = self._step_once(st, action)
            # We keep simulating for determinism even if done flips; callers can stop next frame.
            return (st, reward_sum + r), (obs, done, info)

        (state_out, reward_total), (obs_seq, done_seq, info_seq) = jax.lax.scan(
            body, (state, jnp.array(0.0, dtype=jnp.float32)), None, length=self._substeps
        )

        # take the last substep's values
        obs_last  = tree_util.tree_map(lambda x: x[-1], obs_seq)
        info_last = tree_util.tree_map(lambda x: x[-1], info_seq)
        done_last = done_seq[-1]

        return obs_last, state_out, reward_total, done_last, info_last

    def _process_level_complete(self, state: FrostbiteState):
        """Handle level completion sequence with igloo deconstruction.

        This method manages the end-of-level bonus sequence where:
        1. Igloo blocks are removed one by one, awarding points
        2. Temperature is decremented to zero, awarding more points
        3. Level is advanced and game state is reset for next level

        Args:
            state: Current game state

        Returns:
            Updated game state with level completion processing
        """

        # Check if we're in level completion mode
        is_level_complete = (state.current_level_status & self.consts.LEVEL_COMPLETE) != 0

        # Countdown timer for animation delays
        new_delay = jnp.maximum(0, state.frame_delay - 1)

        # Calculate how many frames have elapsed since level completion started
        elapsed_frames = self.consts.INIT_DELAY_ACTION_VALUE - state.frame_delay

        # Phase 1: Remove igloo blocks one by one
        # Each block is removed every INCREMENT_SCORE_FRAME_DELAY frames
        should_remove_block = (
            is_level_complete &  # Level is complete
            (state.building_igloo_idx >= 0) &  # Still have blocks to remove
            (elapsed_frames > 0) &  # Timer has started
            ((elapsed_frames % self.consts.INCREMENT_SCORE_FRAME_DELAY) == 0)  # On the right frame
        )

        # Decrement igloo block count when removing
        new_building_idx = jnp.where(
            should_remove_block, state.building_igloo_idx - 1, state.building_igloo_idx
        )

        # Award points for each removed block (level × 10 in BCD)
        point_value = self._get_point_value_for_level(state.level)
        block_points = jnp.where(should_remove_block, point_value, 0)
        new_score = self._add_bcd_score(state.score, block_points)
        new_remaining_lives = self._check_extra_life(state.score, new_score, state.remaining_lives)

        # When all blocks are removed, start temperature countdown phase
        blocks_just_finished = is_level_complete & (state.building_igloo_idx == 0) & (new_building_idx < 0)
        new_delay = jnp.where(blocks_just_finished, 90, new_delay)  # Reset timer for temp phase

        # Phase 2: Decrement temperature to zero
        in_temp_phase = is_level_complete & (new_building_idx < 0) & (state.temperature > 0)
        temp_elapsed = 90 - new_delay  # Frames elapsed in temperature phase

        # Decrement temperature every TEMP_DECREMENT_DELAY frames
        should_decrement_temp = (
            in_temp_phase & (new_delay < 90) & (temp_elapsed > 0) &
            ((temp_elapsed % self.consts.TEMP_DECREMENT_DELAY) == 0)
        )

        def decrement_temp_bcd(temp):
            """Decrement temperature by 1 degree in BCD format."""
            ones = temp & 0x0F
            tens = (temp >> 4) & 0x0F
            # Handle BCD borrow when ones digit is 0
            new_ones = jnp.where(ones > 0, ones - 1, 9)
            new_tens = jnp.where(ones == 0, jnp.maximum(0, tens - 1), tens)
            result = (new_tens << 4) | new_ones
            return jnp.where((tens == 0) & (ones == 0), 0, result)

        # Update temperature during countdown
        new_temperature = jnp.where(
            should_decrement_temp, decrement_temp_bcd(state.temperature), state.temperature
        )

        # Award points for each temperature degree (same as block points)
        temp_points = jnp.where(should_decrement_temp, point_value, 0)
        old_score_for_temp = new_score  # Use the already-updated score from block removal
        new_score = self._add_bcd_score(new_score, temp_points)
        new_remaining_lives = self._check_extra_life(old_score_for_temp, new_score, new_remaining_lives)
        
        # Phase 3: Reset game state for next level when temperature reaches zero
        level_reset_complete = (
            is_level_complete & (new_building_idx < 0) &  # All blocks removed
            (state.temperature != 0) & (new_temperature == 0)  # Temperature just hit zero
        )

        # Reset Bailey to starting position
        new_bailey_x = jnp.where(level_reset_complete, 64, state.bailey_x)
        new_bailey_y = jnp.where(level_reset_complete, self.consts.YMIN_BAILEY, state.bailey_y)
        new_bailey_visible = jnp.where(level_reset_complete, 1, state.bailey_visible)

        # Advance to next level
        new_level = jnp.where(level_reset_complete, state.level + 1, state.level)
        new_level_status = jnp.where(level_reset_complete, 0, state.current_level_status)

        # Reset all ice blocks to white (uncollected state)
        new_ice_colors = jnp.where(
            level_reset_complete, jnp.full(4, self.consts.COLOR_ICE_WHITE), state.ice_colors
        )

        # Reset ice floe positions to starting configuration
        # Rows 1 & 3 start on right moving left, rows 2 & 4 start on left moving right
        new_ice_x = jnp.where(
            level_reset_complete, jnp.array([68, -12, 68, -12], dtype=jnp.int32), state.ice_x
        )
        new_ice_directions = jnp.where(
            level_reset_complete, jnp.array([1, 0, 1, 0], dtype=jnp.int32), state.ice_directions
        )

        # Initialize ice block positions for the new level
        # Odd levels have different patterns than even levels
        reset_block_positions, reset_block_counts = self._init_block_positions(
            jnp.array([68, -12, 68, -12], dtype=jnp.int32),
            new_level,
            jnp.where(new_level % 2 == 1, 0, 8)  # Fine motion index for odd/even levels
        )
        new_ice_block_positions = jnp.where(
            level_reset_complete,
            reset_block_positions,
            state.ice_block_positions
        )
        new_ice_block_counts = jnp.where(
            level_reset_complete,
            reset_block_counts,
            state.ice_block_counts
        )

        # Clear Bailey's collision and movement states
        new_bailey_jumping_idx = jnp.where(level_reset_complete, 0, state.bailey_jumping_idx)
        new_bailey_ice_collision_idx = jnp.where(level_reset_complete, -1, state.bailey_ice_collision_idx)
        new_bailey_obstacle_collision_idx = jnp.where(level_reset_complete, -1, state.bailey_obstacle_collision_idx)

        # Clear all active obstacles for fresh start
        new_obstacle_active = jnp.where(
            level_reset_complete, jnp.zeros(4, dtype=jnp.int32), state.obstacle_active
        )

        # Calculate movement speeds for the new level
        current_speeds = self._calculate_speeds_for_level(state.level)
        new_level_speeds = self._calculate_speeds_for_level(new_level)
        
        # Always use speeds that match the current level, update immediately when level changes
        new_ice_speed_whole = jnp.where(level_reset_complete, new_level_speeds['ice_speed_whole'], current_speeds['ice_speed_whole'])
        new_ice_speed_frac = jnp.where(level_reset_complete, new_level_speeds['ice_speed_frac'], current_speeds['ice_speed_frac'])
        new_obstacle_speed_whole = jnp.where(level_reset_complete, new_level_speeds['obstacle_speed_whole'], current_speeds['obstacle_speed_whole'])
        new_obstacle_speed_frac = jnp.where(level_reset_complete, new_level_speeds['obstacle_speed_frac'], current_speeds['obstacle_speed_frac'])
        new_bailey_speed_whole = jnp.where(level_reset_complete, new_level_speeds['bailey_walk_speed_whole'], current_speeds['bailey_walk_speed_whole'])
        new_bailey_speed_frac = jnp.where(level_reset_complete, new_level_speeds['bailey_walk_speed_frac'], current_speeds['bailey_walk_speed_frac'])
        new_bailey_jump_speed_whole = jnp.where(level_reset_complete, new_level_speeds['bailey_jump_speed_whole'], current_speeds['bailey_jump_speed_whole'])
        new_bailey_jump_speed_frac = jnp.where(level_reset_complete, new_level_speeds['bailey_jump_speed_frac'], current_speeds['bailey_jump_speed_frac'])

        reset_walk_acc = jnp.where(level_reset_complete, jnp.array(0, dtype=jnp.int32), state.bailey_walk_frac_accumulator)
        reset_jump_acc = jnp.where(level_reset_complete, jnp.array(0, dtype=jnp.int32), state.bailey_frac_accumulator)
        reset_ice_frac = jnp.where(
            level_reset_complete,
            jnp.zeros(4, dtype=jnp.int32),
            state.ice_frac_accumulators
        )
        reset_ice_dx = jnp.where(
            level_reset_complete,
            jnp.zeros(4, dtype=jnp.int32),
            state.ice_dx_last_frame
        )
        
        # Reset ice fine motion based on new level parity (odd=0, even=8)
        new_ice_fine_motion_index = jnp.where(
            level_reset_complete,
            jnp.where(new_level % 2 == 1, 0, 8),
            state.ice_fine_motion_index
        )

        # Reset fish counter and fish_alive_mask on level completion
        new_number_of_fish_eaten = jnp.where(level_reset_complete, jnp.int32(0), state.number_of_fish_eaten)
        new_fish_alive_mask = jnp.where(level_reset_complete, jnp.zeros(4, dtype=jnp.int32), state.fish_alive_mask)

        next_state = state._replace(
            frame_delay=new_delay,
            building_igloo_idx=new_building_idx,
            score=new_score,
            remaining_lives=new_remaining_lives,
            bailey_x=new_bailey_x,
            bailey_y=new_bailey_y,
            bailey_visible=new_bailey_visible,
            bailey_jumping_idx=new_bailey_jumping_idx,
            bailey_ice_collision_idx=new_bailey_ice_collision_idx,
            bailey_obstacle_collision_idx=new_bailey_obstacle_collision_idx,
            bailey_speed_whole=new_bailey_speed_whole,
            bailey_speed_frac=new_bailey_speed_frac,
            bailey_walk_frac_accumulator=reset_walk_acc,
            bailey_jump_speed_whole=new_bailey_jump_speed_whole,
            bailey_jump_speed_frac=new_bailey_jump_speed_frac,
            bailey_frac_accumulator=reset_jump_acc,
            level=new_level,
            temperature=jnp.where(level_reset_complete, self.consts.INIT_TEMPERATURE, new_temperature),
            current_level_status=new_level_status,
            ice_x=new_ice_x,
            ice_block_positions=new_ice_block_positions,
            ice_block_counts=new_ice_block_counts,
            ice_directions=new_ice_directions,
            ice_colors=new_ice_colors,
            ice_speed_whole=new_ice_speed_whole,
            ice_speed_frac=new_ice_speed_frac,
            ice_frac_accumulators=reset_ice_frac,
            ice_dx_last_frame=reset_ice_dx,
            ice_fine_motion_index=new_ice_fine_motion_index,
            obstacle_active=new_obstacle_active,
            obstacle_speed_whole=new_obstacle_speed_whole,
            obstacle_speed_frac=new_obstacle_speed_frac,
            number_of_fish_eaten=new_number_of_fish_eaten,
            fish_alive_mask=new_fish_alive_mask
        )

        # Spawn 4 new obstacles when level resets
        def _spawn_all(s):
            return self._spawn_obstacles_vec(s, jnp.ones((4,), dtype=jnp.bool_))

        # On the exact frame we finish the level reset, spawn 4 new obstacles
        next_state = jax.lax.cond(
            level_reset_complete,
            _spawn_all,
            lambda s: s,
            next_state
        )

        return next_state
    
    def _update_bailey(self, state: FrostbiteState, action: int):
        """Update Bailey's position, jumping state, and animation.

        This method handles all Bailey movement including:
        - Horizontal movement (walking/jumping)
        - Vertical movement (jumping between ice rows)
        - Automatic movement (igloo entry, ice drift)
        - Movement restrictions (obstacles, bear chase)
        - Animation frame selection

        Args:
            state: Current game state
            action: Player input action

        Returns:
            Updated state with new Bailey position and status
        """

        # Determine if Bailey can be controlled
        if_alive = (state.bailey_alive == 1) & (state.bailey_death_frame == 0) & (state.bailey_visible == 1)
        is_jumping_now = state.bailey_jumping_idx > 0

        # CRITICAL: Bear chase disables ALL player controls
        being_chased_by_bear = state.bailey_grizzly_collision_value != 0

        # Controls enabled unless blocked by obstacle (except mid-jump) or being chased
        controls_enabled = ((state.bailey_obstacle_collision_idx < 0) | is_jumping_now) & ~being_chased_by_bear
        is_entering_igloo = state.igloo_entry_status == 0x80

        # Parse directional inputs from action
        moving_left = if_alive & controls_enabled & ~is_entering_igloo & ((action == Action.LEFT) | (action == Action.UPLEFT) | (action == Action.DOWNLEFT))
        moving_right = if_alive & controls_enabled & ~is_entering_igloo & ((action == Action.RIGHT) | (action == Action.UPRIGHT) | (action == Action.DOWNRIGHT))
        moving_up = if_alive & controls_enabled & ((action == Action.UP) | (action == Action.UPLEFT) | (action == Action.UPRIGHT))
        moving_down = if_alive & controls_enabled & ((action == Action.DOWN) | (action == Action.DOWNLEFT) | (action == Action.DOWNRIGHT))

        # Calculate horizontal movement speed
        # Walking uses fixed 4/16 pixel per frame rate
        # Jumping uses level-scaled speed from speed tables
        if_jumping = is_jumping_now
        attempting_move = moving_left | moving_right

        walk_active = attempting_move & ~if_jumping
        jump_active = attempting_move & if_jumping

        # Fractional movement accumulator for smooth sub-pixel motion
        # Accumulates fractional parts until they make a whole pixel
        walk_frac_add = jnp.where(walk_active, state.bailey_speed_frac, jnp.int32(0))
        walk_frac_sum = state.bailey_walk_frac_accumulator + walk_frac_add
        walk_pixels = state.bailey_speed_whole + (walk_frac_sum // 16)  # Convert to pixels
        walk_pixels = jnp.where(walk_active, walk_pixels, jnp.int32(0))
        new_walk_acc = jnp.where(walk_active, walk_frac_sum % 16, state.bailey_walk_frac_accumulator)  # Keep remainder

        # Jump movement uses separate accumulator and speed values
        jump_frac_add = jnp.where(jump_active, state.bailey_jump_speed_frac, jnp.int32(0))
        jump_frac_sum = state.bailey_frac_accumulator + jump_frac_add
        jump_pixels = state.bailey_jump_speed_whole + (jump_frac_sum // 16)
        jump_pixels = jnp.where(jump_active, jump_pixels, jnp.int32(0))
        new_jump_acc = jnp.where(jump_active, jump_frac_sum % 16, state.bailey_frac_accumulator)

        # Select appropriate speed based on movement state
        actual_speed = jnp.where(if_jumping, jump_pixels, walk_pixels)

        # Set horizontal boundaries based on Bailey's vertical position
        on_shore = state.bailey_y == self.consts.YMIN_BAILEY
        x_min = jnp.where(on_shore, self.consts.SHORE_X_MIN, self.consts.ICE_X_MIN)
        x_max = self.consts.SHORE_X_MAX

        # Automatic movement toward igloo during entry sequence
        target_igloo_x = 123  # Fixed igloo X position
        auto_dx = jnp.where(
            is_entering_igloo,
            jnp.sign(target_igloo_x - state.bailey_x) * jnp.minimum(2, jnp.abs(target_igloo_x - state.bailey_x)),
            0
        )

        # Calculate base horizontal movement delta
        dx = jnp.where(
            is_entering_igloo, auto_dx,  # Override with auto-movement when entering igloo
            jnp.where(moving_left, -actual_speed, jnp.where(moving_right, actual_speed, 0))
        )

        # Check if Bailey is blocked by an obstacle
        trapped = state.bailey_obstacle_collision_idx >= 0

        # Add ice drift when Bailey is on an ice floe
        row_idx = state.bailey_ice_collision_idx
        safe_row_idx = jnp.clip(row_idx, 0, 3)  # Prevent array bounds issues
        row_dx = state.ice_dx_last_frame[safe_row_idx]  # Get ice movement from last frame
        ice_dx = jnp.where((~trapped) & (row_idx >= 0), row_dx, jnp.int32(0))

        # Bear chase overrides all movement except the bear's push
        being_chased_by_bear = state.bailey_grizzly_collision_value != 0
        total_dx = jnp.where(being_chased_by_bear, 0, dx + ice_dx)  # Zero out movement when chased

        # Boundary enforcement
        # Normal gameplay respects shore boundaries (x_min)
        # But bear can push Bailey all the way to SHORE_X_MIN for death animation
        effective_x_min = jnp.where(being_chased_by_bear, self.consts.SHORE_X_MIN, x_min)

        # Apply movement with boundary clamping
        new_x = jnp.clip(state.bailey_x + total_dx, effective_x_min, x_max)
        
        # Update Bailey's facing direction
        new_direction = jnp.where(
            is_entering_igloo, 0,  # Face right when entering igloo
            jnp.where(moving_left, 1, jnp.where(moving_right, 0, state.bailey_direction))
        )

        # Jumping system timing and state checks
        can_update_jump = (state.frame_count % 4) == 0  # Jump updates every 4 frames
        can_start_jump = state.bailey_jumping_idx == 0  # Not already jumping

        # Special case: automatic jump when entering igloo
        at_igloo_x = jnp.abs(state.bailey_x - 123) <= 1
        should_jump_for_igloo = is_entering_igloo & at_igloo_x

        # Edge detection for jump inputs (prevent holding)
        was_not_pressing_up = (state.last_action != Action.UP) & (state.last_action != Action.UPLEFT) & (state.last_action != Action.UPRIGHT)
        was_not_pressing_down = (state.last_action != Action.DOWN) & (state.last_action != Action.DOWNLEFT) & (state.last_action != Action.DOWNRIGHT)

        # Determine if Bailey can initiate a jump
        # Up jump: allowed from ice (not shore) or for igloo entry
        # Bear chase disables all jumping
        can_jump_up = (((moving_up & was_not_pressing_up) | (should_jump_for_igloo & (state.bailey_y > 6))) &
                       can_start_jump & ((state.bailey_y > self.consts.YMIN_BAILEY) | should_jump_for_igloo)) & ~being_chased_by_bear
        can_jump_down = (moving_down & was_not_pressing_down) & can_start_jump & \
                        (state.bailey_y < self.consts.YMAX_BAILEY) & ~is_entering_igloo & ~being_chased_by_bear

        # Set jump index: 31 for up jump, 15 for down jump
        # These indices map to the jump offset tables
        new_jump_idx = jnp.where(can_jump_up, 31, jnp.where(can_jump_down, 15, state.bailey_jumping_idx))

        # NOTE: Bear collision is NOT cleared when starting a jump
        # It's only cleared when the jump COMPLETES (handled in ice collision check)

        # Continue ongoing jump animation
        new_jump_idx, new_y = jax.lax.cond(
            can_update_jump & (new_jump_idx > 0) & ~is_entering_igloo,
            lambda: self._continue_jump(state.bailey_y, new_jump_idx),
            lambda: (new_jump_idx, state.bailey_y)
        )

        # Special vertical movement for igloo entry
        should_move_up_for_igloo = is_entering_igloo & at_igloo_x & (state.bailey_y > 6)
        new_y = jnp.where(should_move_up_for_igloo, jnp.maximum(0, state.bailey_y - 2), new_y)
        new_jump_idx = jnp.where(is_entering_igloo, 0, new_jump_idx)  # Cancel jump during igloo entry

        # Determine animation frame based on movement state
        jump_val = new_jump_idx & 0x0F  # Lower 4 bits indicate jump progress
        is_jumping_sprite = (jump_val >= 7) & ~is_entering_igloo  # Show jump sprite in mid-air

        # Walking animation cycles between two frames
        # Bear chase overrides walking animation (controlled by bear logic)
        is_moving = (moving_left | moving_right | is_entering_igloo) & ~being_chased_by_bear
        walk_frame = jnp.where(is_moving, (state.frame_count >> 3) & 1, 0)  # Toggle every 8 frames

        # Select final animation frame
        new_animation = jnp.where(
            being_chased_by_bear,
            state.bailey_animation_idx,  # Preserve animation from bear update
            jnp.where(is_jumping_sprite, 2, walk_frame)  # Jump sprite (2) or walk frame (0/1)
        )
        
        return state._replace(
            bailey_x=new_x,
            bailey_y=new_y,
            bailey_jumping_idx=new_jump_idx,
            bailey_direction=new_direction,
            bailey_animation_idx=new_animation,
            bailey_walk_frac_accumulator=new_walk_acc,
            bailey_frac_accumulator=new_jump_acc,
            last_action=action
        )
    
    def _continue_jump(self, current_y: jnp.ndarray, jump_idx: jnp.ndarray):
        """Continue Bailey's jump animation by one frame.

        Uses a lookup table of Y offsets to create smooth jump arcs.
        Handles special cases for landing on shore vs ice rows.

        Args:
            current_y: Bailey's current Y position
            jump_idx: Current jump animation index (0-31)

        Returns:
            Tuple of (new_jump_index, new_y_position)
        """

        # Load jump offset table - contains Y deltas for each frame of jump
        offset_table = jnp.array(self.consts.BAILEY_JUMP_OFFSETS)

        # Advance jump animation by one frame
        raw_next_idx = jump_idx - 1
        table_idx = jnp.clip(raw_next_idx, 0, 31)  # Ensure valid table index
        offset = offset_table[table_idx]  # Get Y offset for this frame
        new_y = current_y + offset

        # Handle jump completion
        # Index 16 is the transition point between down and up jumps
        next_idx = jnp.where(raw_next_idx == 16, 0, raw_next_idx)
        next_idx = jnp.maximum(0, next_idx)

        # Special case: Add extra drop when landing on first ice row
        # This ensures Bailey lands properly on the ice platform
        first_row_target = self.consts.YMIN_BAILEY + 26  # First ice row Y position
        apply_first_drop = (
            (jump_idx <= 16) &  # Downward jump
            (next_idx == 0) &    # Jump completing
            (offset > 0) &       # Moving downward
            (new_y <= first_row_target)  # Near first row
        )
        new_y = jnp.where(apply_first_drop, new_y + 3, new_y)  # Add 3 pixels to reach ice

        # Shore landing: Snap Bailey to exact shore Y position
        shore_clamp = self.consts.YMIN_BAILEY
        is_upward_phase = jump_idx > 16  # Jumping up (indices 17-31)

        # Different tolerance for up vs down jumps
        shore_snap = (
            (next_idx == 0) & (  # Jump completing
                (new_y <= shore_clamp + 1) |  # Down jump: 1 pixel tolerance
                (is_upward_phase & (new_y <= shore_clamp + 3))  # Up jump: 3 pixel tolerance
            )
        )
        new_y = jnp.where(shore_snap, shore_clamp, new_y)

        return next_idx, new_y

    def _update_ice_blocks(self, state: FrostbiteState, action: int):
        """Update ice block positions, handle reversals, and manage block movement.

        This method handles:
        - Ice floe movement and wrapping
        - Fire button reversals (costs igloo blocks)
        - Ice block positions for collision detection

        Args:
            state: Current game state
            action: Player input action

        Returns:
            Updated state with new ice positions
        """
        
        # Fire button reversal mechanic
        # Pressing FIRE reverses ice direction at the cost of an igloo block
        is_pressing_fire = action == Action.FIRE
        was_not_pressing_fire = state.last_action != Action.FIRE
        fire_pressed = is_pressing_fire & was_not_pressing_fire  # Edge detection

        # Requirements for ice reversal:
        # 1. Bailey must be standing on ice (not jumping, not on shore)
        # 2. Fire button must be newly pressed (not held)
        # 3. Bailey must be alive and visible
        # 4. Must have collected at least 1 igloo block
        on_ice = state.bailey_ice_collision_idx >= 0
        is_jumping = state.bailey_jumping_idx > 0
        has_blocks = state.building_igloo_idx >= 0  # At least 1 block collected

        can_reverse = (
            fire_pressed &
            on_ice &
            ~is_jumping &
            has_blocks &  # Must have blocks to spend
            (state.bailey_alive == 1) &
            (state.bailey_visible == 1)
        )

        # Identify which ice row Bailey is standing on
        ice_row = state.bailey_ice_collision_idx

        # Vectorized ice reversal - replace the 4-iteration loop
        row_mask = (jnp.arange(4, dtype=jnp.int32) == ice_row)
        should_reverse_vec = can_reverse & row_mask
        new_ice_directions = jnp.where(
            should_reverse_vec,
            1 - state.ice_directions,
            state.ice_directions
        )

        # Block cost mechanic:
        # - Normal gameplay: Costs 1 igloo block to reverse
        # - Complete igloo (15 blocks): Reversals are FREE
        igloo_complete = state.building_igloo_idx >= self.consts.MAX_IGLOO_INDEX
        should_penalize = can_reverse & ~igloo_complete  # Only charge if igloo incomplete

        new_building_idx = jnp.where(
            should_penalize,
            state.building_igloo_idx - 1,  # Deduct one block
            state.building_igloo_idx
        )
        
        # Check if ice layout needs updating (for breathing animation)
        _, target_block_count, _, _ = self._determine_ice_layout(state.level, state.ice_fine_motion_index)
        base_positions, base_counts = self._init_block_positions(state.ice_x, state.level, state.ice_fine_motion_index)
        # Better: check if any row needs layout update (they are uniform, so this is clearer)
        layout_changed = jnp.any(state.ice_block_counts != target_block_count)

        # Update block configuration if layout changed
        block_positions = jnp.where(layout_changed, base_positions, state.ice_block_positions)
        block_counts = jnp.where(layout_changed, base_counts, state.ice_block_counts)

        # Ice movement calculation using fractional accumulator system
        # This provides smooth sub-pixel movement across different speeds
        should_move = jnp.bitwise_and(state.frame_count, 1) == 0  # Move every other frame

        # Whole pixel component of speed
        speed_whole = jnp.where(
            should_move,
            state.ice_speed_whole.astype(jnp.int32),
            jnp.zeros_like(state.ice_speed_whole, dtype=jnp.int32)
        )

        # Fractional component (in 1/16 pixel units)
        frac_increment = jnp.where(
            should_move,
            state.ice_speed_frac,
            jnp.zeros_like(state.ice_speed_frac)
        )

        # Accumulate fractional movement and convert to pixels
        frac_sum = state.ice_frac_accumulators + frac_increment
        extra_pixels = frac_sum // 16  # Convert 16ths to whole pixels
        new_frac = jnp.where(should_move, frac_sum % 16, state.ice_frac_accumulators)

        # Calculate total movement for this frame
        move_pixels = speed_whole + extra_pixels
        direction = jnp.where(new_ice_directions == 0, 1, -1)  # 0=right(+1), 1=left(-1)
        move_dx = direction * move_pixels
        move_dx = jnp.where(block_counts > 0, move_dx, jnp.int32(0))  # Only move active rows

        # Determine block spacing based on count (6 narrow, 3 wide)
        is_narrow = block_counts == 6
        spacing = jnp.where(
            is_narrow,
            jnp.int32(self.consts.ICE_NARROW_SPACING),
            jnp.int32(self.consts.ICE_WIDE_SPACING)
        )
        block_width = jnp.where(is_narrow, jnp.int32(12), jnp.int32(24))

        # Create mask for active blocks in each row
        block_indices = jnp.arange(6, dtype=jnp.int32)
        active_mask = block_indices[None, :] < block_counts[:, None]

        # Apply movement to all active blocks
        dx_matrix = move_dx[:, None] * active_mask.astype(jnp.int32)
        updated_positions = block_positions + dx_matrix

        # Playfield wrapping logic for ice blocks
        # Blocks wrap around at playfield boundaries (x=8 and x=160)
        L = jnp.int32(self.consts.PLAYFIELD_LEFT)
        R = jnp.int32(self.consts.PLAYFIELD_RIGHT)
        W = jnp.int32(self.consts.PLAYFIELD_WIDTH)

        # Check which blocks need to wrap
        # Right-moving blocks wrap when X >= right boundary
        # Left-moving blocks wrap when X < left boundary - block_width
        wrap_right = ((new_ice_directions == 0)[:, None] & active_mask & (updated_positions >= R))
        wrap_left = ((new_ice_directions == 1)[:, None] & active_mask & (updated_positions < L - block_width[:, None]))

        # Apply wrapping by adjusting X position by playfield width
        updated_positions = jnp.where(wrap_right, updated_positions - W, updated_positions)
        updated_positions = jnp.where(wrap_left, updated_positions + W, updated_positions)

        # Sentinel value marks inactive/unused block positions
        sentinel = jnp.int32(self.consts.ICE_UNUSED_POS)

        # Sort blocks by X position to maintain consistent left-to-right order
        # This is important for collision detection and rendering
        large_val = jnp.int32(1 << 15)  # Large value for sorting inactive blocks to end
        sort_keys = jnp.where(active_mask, updated_positions, large_val)
        sort_indices = jnp.argsort(sort_keys, axis=1)

        # Reorder positions and masks based on sort
        sorted_positions = jnp.take_along_axis(updated_positions, sort_indices, axis=1)
        sorted_mask = jnp.take_along_axis(active_mask, sort_indices, axis=1)

        # Apply sentinel value to inactive positions
        updated_positions = jnp.where(sorted_mask, sorted_positions, sentinel)

        # NEW: advance a persistent per-row anchor by the same dx as the row moved
        # This keeps the anchor stable and prevents teleporting during breathing animations
        anchor = state.ice_x + move_dx

        # Wrap the anchor into the same coordinate system as the blocks (using playfield)
        anchor = jnp.where(
            new_ice_directions == 0,  # moving right
            jnp.where(anchor >= R, anchor - W, anchor),
            jnp.where(anchor < L - block_width, anchor + W, anchor),  # moving left
        )

        new_ice_x = anchor

        # Return updated state with new ice positions and directions
        new_block_counts = block_counts
        return state._replace(
            ice_x=new_ice_x,  # Leftmost position of each row
            ice_block_positions=updated_positions,  # All block positions
            ice_block_counts=new_block_counts,  # Blocks per row (3 or 6)
            ice_directions=new_ice_directions,  # Movement direction per row
            ice_frac_accumulators=new_frac,  # Fractional movement remainder
            ice_dx_last_frame=move_dx,  # Movement from this frame (for Bailey drift)
            building_igloo_idx=new_building_idx  # Updated after reversal cost
        )
    
    def _update_obstacles(self, state: FrostbiteState):
        """Update all obstacles (birds, fish, crabs, clams) with movement and spawning.

        This method handles:
        - Movement calculation with fractional accumulation
        - Level 5+ stuttering behavior (crabs/clams alternate freezing)
        - Despawn/respawn when off-screen
        - Animation frame cycling
        - Sprite duplication patterns for gradual reveal

        Args:
            state: Current game state

        Returns:
            Updated state with new obstacle positions and properties
        """

        # Phase 1: Calculate base movement for all active obstacles
        # Obstacles move every other frame for smoother motion
        should_move = jnp.bitwise_and(state.frame_count, 1) == 0
        move_mask = jnp.where(should_move, 1, 0).astype(jnp.int32)

        # Fractional movement accumulator (same system as ice blocks)
        frac_increment = state.obstacle_speed_frac * move_mask
        frac_acc = state.obstacle_frac_accumulators + frac_increment
        pixels = (state.obstacle_speed_whole.astype(jnp.int32) * move_mask) + (frac_acc // 16)
        new_frac = jnp.where(should_move, frac_acc % 16, state.obstacle_frac_accumulators)

        # Apply direction to movement (0=right, 1=left)
        dx = jnp.where(state.obstacle_directions == 0, pixels, -pixels)

        # Phase 2: Level 5+ stuttering movement pattern
        # Creates alternating movement for crabs and clams
        # - Killer Clams (ID=3): Freeze when frame bit 0x40 is 0
        # - King Crabs (ID=2): Freeze when frame bit 0x40 is 1
        # - Snow Geese (ID=0) and Fish (ID=1): Always move
        frame_bit = (state.frame_count & self.consts.OBSTACLE_STUTTER_MASK) != 0

        # Vectorized stutter gating
        is_clam = state.obstacle_types == self.consts.ID_KILLER_CLAM
        is_crab = state.obstacle_types == self.consts.ID_KING_CRAB
        should_freeze = (state.level >= self.consts.OBSTACLE_STUTTER_LEVEL) & (
            (is_clam & ~frame_bit) | (is_crab & frame_bit)
        )
        dx = jnp.where(should_freeze, 0, dx)

        # Only move active obstacles
        dx = jnp.where(state.obstacle_active == 1, dx, 0)
        
        # Phase 3: Apply movement and update positions
        new_x = state.obstacle_x + dx
        new_frac = jnp.where(state.obstacle_active == 1, new_frac, state.obstacle_frac_accumulators)

        # Phase 4: Check for despawn and trigger respawn
        # Obstacles despawn when they move fully off-screen plus margin
        despawn_margin = self.consts.OBSTACLE_DESPAWN_X
        off_left = new_x < -despawn_margin
        off_right = new_x > self.consts.SCREEN_WIDTH + despawn_margin
        should_respawn = (off_left | off_right) & (state.obstacle_active == 1)

        # Update state with new positions before respawning
        state_after_move = state._replace(
            obstacle_x=new_x,
            obstacle_frac_accumulators=new_frac,
            obstacle_dx_last_frame=dx,  # Store for Bailey collision push
        )

        # Respawn obstacles that went off-screen
        state = self._spawn_obstacles_vec(state_after_move, should_respawn)

        # Phase 5: Update animation frames and sprite duplication patterns

        # Animation frame cycling
        # Different obstacle types have different animation speeds
        frame_count = state.frame_count.astype(jnp.int32)
        obstacle_types = state.obstacle_types.astype(jnp.int32)

        # Clams animate slower than other obstacles
        default_mask = jnp.int32(self.consts.OBSTACLE_ANIMATION_MASK_DEFAULT)
        clam_mask = jnp.int32(self.consts.OBSTACLE_ANIMATION_MASK_CLAM)
        anim_mask = jnp.where(obstacle_types == self.consts.ID_KILLER_CLAM, clam_mask, default_mask)

        # Toggle animation frame based on frame count
        animation_bit = jnp.bitwise_and(frame_count, anim_mask)
        new_anim_idx = jnp.where(animation_bit == 0, jnp.int32(1), jnp.int32(0))

        # Floating animation for fish (vertical bobbing)
        # Uses a sine-like wave pattern for smooth motion
        phase = jnp.bitwise_and(
            frame_count >> self.consts.FLOATING_OBSTACLE_PHASE_SHIFT,
            self.consts.FLOATING_OBSTACLE_PHASE_MASK
        )
        # Mirror the phase to create a smooth up-down motion
        mirrored_phase = jnp.where(
            phase < len(self.consts.FLOATING_OBSTACLE_OFFSETS),
            phase,
            jnp.bitwise_xor(phase, self.consts.FLOATING_OBSTACLE_PHASE_MASK)
        )
        float_table = jnp.array(self.consts.FLOATING_OBSTACLE_OFFSETS, dtype=jnp.int32)
        float_offset_value = float_table[jnp.clip(mirrored_phase, 0, float_table.shape[0] - 1)]

        # Only fish float (ID >= 1)
        float_offsets = jnp.where(
            obstacle_types >= self.consts.ID_FISH,
            jnp.full_like(obstacle_types, float_offset_value),
            jnp.zeros_like(obstacle_types)
        )

        # Apply to active obstacles only
        new_anim_idx = jnp.where(state.obstacle_active == 1, new_anim_idx, jnp.zeros_like(new_anim_idx))
        float_offsets = jnp.where(state.obstacle_active == 1, float_offsets, jnp.zeros_like(float_offsets))

        # Sprite duplication pattern calculation
        # Creates gradual reveal effect as obstacles enter from screen edge
        # Birds appear as 1, then 2, then 3 copies as they move onto screen

        # Position threshold for pattern calculation
        XMAX_MINUS_32 = 128  # Screen edge minus margin

        # Calculate masking index based on X position
        # Far off-screen obstacles use full mask (index 8)
        # As they approach, mask reduces to reveal more copies
        masking_idx = jnp.where(
            state.obstacle_x < XMAX_MINUS_32,
            8,  # Full mask when far off-screen
            jnp.minimum((state.obstacle_x - XMAX_MINUS_32) // 16, 8)  # Gradual reveal
        )

        # Masking values control how many bits of pattern index are used
        # Lower values = fewer copies, higher values = more copies
        masking_values = jnp.array([6, 4, 0, 0, 0, 0, 1, 3, 7], dtype=jnp.int32)
        masking_idx = jnp.clip(masking_idx, 0, 8)
        mask_value = masking_values[masking_idx]

        # Apply mask to pattern index to determine duplication mode
        table_idx = state.obstacle_pattern_index & mask_value

        # Lookup table for sprite duplication modes
        # Maps pattern index to number of copies and spacing
        duplication_mode_table = jnp.array([
            self.consts.SPRITE_SINGLE,           # 0: Single sprite
            self.consts.SPRITE_SINGLE,           # 1: Single sprite
            self.consts.SPRITE_SINGLE,           # 2: Single sprite
            self.consts.SPRITE_DOUBLE,           # 3: Two close sprites
            self.consts.SPRITE_SINGLE,           # 4: Single sprite
            self.consts.SPRITE_DOUBLE_SPACED,    # 5: Two spaced sprites
            self.consts.SPRITE_DOUBLE,           # 6: Two close sprites
            self.consts.SPRITE_TRIPLE            # 7: Three sprites
        ], dtype=jnp.int32)

        # Look up duplication mode for each obstacle
        table_idx = jnp.clip(table_idx, 0, 7)
        new_duplication_mode = duplication_mode_table[table_idx]

        # Inactive obstacles always show as single sprite
        new_duplication_mode = jnp.where(
            state.obstacle_active == 1,
            new_duplication_mode,
            self.consts.SPRITE_SINGLE
        )
        
        return state._replace(
            obstacle_duplication_mode=new_duplication_mode,
            obstacle_animation_idx=new_anim_idx,
            obstacle_float_offsets=float_offsets,
        )
    
    def _check_collisions(self, state: FrostbiteState):
        """Check Bailey's collision with ice blocks and water.

        This method handles:
        - Ice block collision detection with wrapping
        - Water collision (falling between ice blocks)
        - Death conditions (drowning, freezing)
        - Bear collision clearing after jump
        - Death animation and life management

        Args:
            state: Current game state

        Returns:
            Updated state with collision results
        """

        # Only check collisions when Bailey is visible
        if_checking_collisions = state.bailey_visible == 1

        # Bear collision escape mechanism
        # Jumping is the ONLY way Bailey can escape from the bear
        was_jumping_before = (state.bailey_landing_status & 0x40) != 0
        is_jumping_now = state.bailey_jumping_idx > 0
        just_completed_jump = was_jumping_before & ~is_jumping_now

        # Clear bear collision when jump completes
        new_bailey_grizzly_collision_value = jnp.where(
            just_completed_jump,
            0,  # Escape successful!
            state.bailey_grizzly_collision_value
        )

        # Determine which ice row Bailey might be standing on
        # Y positions: shore at 56, then ice rows at 84, 109, 134, 159
        bailey_ice_y = jnp.array([84, 109, 134, 159])
        y_distances = jnp.abs(state.bailey_y - bailey_ice_y)
        bailey_row = jnp.where(jnp.min(y_distances) < 3, jnp.argmin(y_distances), -1)

        def check_collision_for_row(row_idx):
            """Check if Bailey is standing on any ice block in the specified row."""
            # Get ice block positions for this row (including breathing animation)
            segment_positions, segment_widths, segment_mask = self._get_row_segments(state, row_idx)
            W = self.consts.PLAYFIELD_WIDTH
            wrap_offsets = jnp.array((-W, 0, W), dtype=jnp.int32)

            # Bailey's horizontal bounds
            bailey_left = state.bailey_x
            bailey_right = state.bailey_x + self.consts.BAILEY_BOUNDING_WIDTH

            # Check all blocks in this row using vmap
            def check_single_block(block_idx):
                """Check collision with a single ice block segment."""
                is_active = segment_mask[block_idx]
                width = segment_widths[block_idx]
                base_left = segment_positions[block_idx]

                # Different margins for narrow (6-block) vs wide (3-block) ice
                is_narrow = width == 12
                left_margin = jnp.where(
                    is_narrow,
                    jnp.int32(self.consts.ICE_NARROW_LEFT_MARGIN),
                    jnp.int32(self.consts.ICE_WIDE_LEFT_MARGIN)
                )
                right_margin = jnp.where(
                    is_narrow,
                    jnp.int32(self.consts.ICE_NARROW_RIGHT_MARGIN),
                    jnp.int32(self.consts.ICE_WIDE_RIGHT_MARGIN)
                )

                # Calculate block collision bounds with margins
                block_left = base_left - left_margin
                block_right = base_left + width + right_margin

                # Check all 3 wrap positions for this block using vmap
                def check_single_wrap(offset):
                    left = block_left + offset
                    right = block_right + offset
                    L = self.consts.PLAYFIELD_LEFT
                    R = self.consts.PLAYFIELD_RIGHT
                    on_screen = (right > L) & (left < R)
                    overlap = jnp.minimum(bailey_right, right) - jnp.maximum(bailey_left, left)
                    contact = on_screen & (overlap >= self.consts.ICE_MIN_OVERLAP)
                    return contact

                check_all_wraps = jax.vmap(check_single_wrap)
                wrap_hits = check_all_wraps(wrap_offsets)
                block_hit = jnp.any(wrap_hits)

                # Only return true if block is active
                return is_active & block_hit

            # Apply check to all blocks in parallel
            block_indices = jnp.arange(segment_positions.shape[0], dtype=jnp.int32)
            check_all_blocks = jax.vmap(check_single_block)
            block_hits = check_all_blocks(block_indices)
            return jnp.any(block_hits)

        # Execute collision check for Bailey's current row
        on_ice = jax.lax.cond(
            bailey_row >= 0, lambda: check_collision_for_row(bailey_row), lambda: jnp.array(False)
        )

        # Update collision index (-1 means not on ice)
        new_collision_idx = jnp.where(bailey_row >= 0, jnp.where(on_ice, bailey_row, -1), -1)

        # Death condition checks
        on_shore = state.bailey_y == self.consts.YMIN_BAILEY
        is_entering_igloo = state.igloo_entry_status == 0x80
        level_complete = state.current_level_status == self.consts.LEVEL_COMPLETE

        # Water collision: Bailey falls in water if not on ice/shore/jumping
        in_water = if_checking_collisions & ~on_shore & ~is_jumping_now & ~on_ice & ~is_entering_igloo & ~level_complete

        # Freezing: Bailey freezes when temperature reaches 0
        frozen = if_checking_collisions & (state.temperature == 0) & ~level_complete

        # Update Bailey's alive status
        new_alive = jnp.where(in_water | frozen, 0, state.bailey_alive)
        new_frozen = jnp.where((state.bailey_alive == 1) & (new_alive == 0) & frozen, 1, state.bailey_frozen)

        # Bailey stays at current Y
        new_bailey_y_from_sinking = state.bailey_y

        # Death animation management
        just_died = (state.bailey_alive == 1) & (new_alive == 0)
        animation_in_progress = (state.bailey_death_frame > 0) & (state.bailey_death_frame < 120)

        # Start or continue death animation
        new_death_frame = jnp.where(
            just_died, 1,  # Start animation
            jnp.where(animation_in_progress, state.bailey_death_frame + 1, state.bailey_death_frame)
        )

        # Life management after death animation completes
        death_animation_complete = state.bailey_death_frame >= 120
        new_remaining_lives = jnp.where(
            death_animation_complete & (state.bailey_death_frame == 120),
            jnp.maximum(0, state.remaining_lives - 1),
            state.remaining_lives
        )

        # Check for game over
        final_death = death_animation_complete & (state.remaining_lives == 0)
        # Determine if Bailey should respawn (has lives remaining)
        should_respawn = death_animation_complete & ~final_death

        # Initialize state variables for updates
        new_bailey_sinking = state.bailey_sinking
        new_level_status = state.current_level_status

        # Respawn Bailey at shore if lives remain
        new_bailey_y = jnp.where(should_respawn, self.consts.YMIN_BAILEY, new_bailey_y_from_sinking)
        new_bailey_alive = jnp.where(should_respawn, 1, new_alive)
        final_death_frame = jnp.where(should_respawn, 0, new_death_frame)
        new_frozen = jnp.where(should_respawn, 0, new_frozen)
        new_bailey_sinking = jnp.where(should_respawn, 0, new_bailey_sinking)

        # Clear status flags on respawn
        new_level_status = jnp.where(
            should_respawn,
            state.current_level_status & ~self.consts.BAILEY_SINKING,
            new_level_status
        )

        # Reset game state when Bailey respawns
        # Ice blocks return to starting positions
        initial_ice_x = jnp.array([68, -12, 68, -12], dtype=jnp.int32)  # Row 1&3 right, 2&4 left
        initial_ice_directions = jnp.array([1, 0, 1, 0], dtype=jnp.int32)  # Alternating directions

        # Fine motion index determines initial ice layout (odd/even levels)
        initial_fine_motion = jnp.where(state.level % 2 == 1, 0, 8)
        initial_block_positions, initial_block_counts = self._init_block_positions(
            initial_ice_x,
            state.level,
            initial_fine_motion
        )

        # Reset ice positions and directions
        new_ice_x = jnp.where(should_respawn, initial_ice_x, state.ice_x)
        new_ice_directions = jnp.where(should_respawn, initial_ice_directions, state.ice_directions)
        new_ice_block_positions = jnp.where(
            should_respawn,
            initial_block_positions,
            state.ice_block_positions
        )
        new_ice_block_counts = jnp.where(
            should_respawn,
            initial_block_counts,
            state.ice_block_counts
        )
        new_ice_fine_motion = jnp.where(should_respawn, initial_fine_motion, state.ice_fine_motion_index)

        # Reset igloo entry and temperature
        new_igloo_entry_status = jnp.where(should_respawn | (new_alive == 0), 0, state.igloo_entry_status)
        new_temperature = jnp.where(should_respawn, self.consts.INIT_TEMPERATURE, state.temperature)

        # Clear all obstacles when Bailey respawns
        new_obstacle_x = jnp.where(
            should_respawn,
            jnp.array([-20, 160, 160, -20], dtype=jnp.int32),  # Off-screen positions
            state.obstacle_x
        )
        new_obstacle_active = jnp.where(
            should_respawn,
            jnp.zeros(4, dtype=jnp.int32),  # All inactive
            state.obstacle_active
        )

        # Reset bear to initial position
        new_polar_grizzly_x = jnp.where(
            should_respawn,
            self.consts.INIT_POLAR_GRIZZLY_HORIZ_POS,
            state.polar_grizzly_x
        )
        new_polar_grizzly_frac = jnp.where(
            should_respawn,
            0,
            state.polar_grizzly_frac_accumulator
        )
        # Clear bear collision state on respawn
        new_bailey_grizzly_collision_timer = jnp.where(
            should_respawn,
            0,
            state.bailey_grizzly_collision_timer
        )
        new_bailey_grizzly_collision_value = jnp.where(
            should_respawn,
            0,
            new_bailey_grizzly_collision_value  # May have been cleared by jump completion
        )

        # Track Bailey's landing status for ice collection
        landed_on_ice = just_completed_jump & on_ice
        new_landing_status = jnp.where(
            is_jumping_now, 0x40,  # Currently jumping
            jnp.where(landed_on_ice, 0x80, 0)  # Just landed on ice
        )

        # Bailey can walk on both white and blue ice (no sinking)
        new_level_status = state.current_level_status
        new_bailey_sinking = state.bailey_sinking

        # Ice collection mechanic
        # When Bailey lands on white ice, it turns blue and awards points
        # Once igloo is complete (15 blocks), ice no longer changes color
        igloo_complete = state.building_igloo_idx >= self.consts.MAX_IGLOO_INDEX
        new_ice_colors = state.ice_colors

        # Check each ice row for color change (vectorized)
        # row_changed: True if the matched row was white -> blue
        target_row = jnp.equal(new_collision_idx, jnp.arange(4, dtype=jnp.int32))
        can_blue   = (state.ice_colors == self.consts.COLOR_ICE_WHITE)
        row_changed_mask = (landed_on_ice & ~igloo_complete) & target_row & can_blue
        row_changed = jnp.any(row_changed_mask)

        new_ice_colors = jnp.where(
            row_changed_mask,
            jnp.full_like(state.ice_colors, self.consts.COLOR_ICE_BLUE),
            state.ice_colors
        )

        # Award points and igloo blocks for collecting ice
        new_score = state.score  # No need for .copy() in JAX
        new_building_idx = state.building_igloo_idx

        # Points based on level (level × 10 in BCD)
        point_value = self._get_point_value_for_level(state.level)
        points = jnp.where(row_changed, point_value, 0)
        new_score = self._add_bcd_score(new_score, points)
        
        # Check for extra life
        new_remaining_lives = self._check_extra_life(state.score, new_score, new_remaining_lives)
        
        new_building_idx = jnp.where(
            row_changed & (state.building_igloo_idx < self.consts.MAX_IGLOO_INDEX),
            jnp.minimum(state.building_igloo_idx + 1, self.consts.MAX_IGLOO_INDEX),
            state.building_igloo_idx
        )
        
        all_blue = jnp.all(new_ice_colors == self.consts.COLOR_ICE_BLUE)
        new_delay = jnp.where(
            all_blue & (state.completed_ice_blocks_delay == 0), 32,
            jnp.maximum(0, state.completed_ice_blocks_delay - 1)
        )
        
        should_reset = all_blue & (new_delay == 0) & (state.completed_ice_blocks_delay == 1)
        new_ice_colors = jnp.where(should_reset, jnp.full(4, self.consts.COLOR_ICE_WHITE), new_ice_colors)

        # Reset ice colors to white when Bailey respawns after death
        new_ice_colors = jnp.where(
            should_respawn,
            jnp.full(4, self.consts.COLOR_ICE_WHITE, dtype=jnp.int32),
            new_ice_colors
        )

        # Only update bailey_x if respawning
        # DO NOT overwrite bailey_x otherwise - bear push has already happened!
        
        updated_state = state._replace(
            bailey_x=jnp.where(should_respawn, 64, state.bailey_x),
            bailey_y=new_bailey_y,
            bailey_ice_collision_idx=new_collision_idx,
            bailey_alive=new_bailey_alive,
            bailey_frozen=new_frozen,
            bailey_landing_status=new_landing_status,
            bailey_sinking=new_bailey_sinking,
            bailey_walk_frac_accumulator=jnp.where(should_respawn, jnp.int32(0), state.bailey_walk_frac_accumulator),
            bailey_frac_accumulator=jnp.where(should_respawn, jnp.int32(0), state.bailey_frac_accumulator),
            current_level_status=new_level_status,
            ice_x=new_ice_x,
            ice_block_positions=new_ice_block_positions,
            ice_block_counts=new_ice_block_counts,
            ice_directions=new_ice_directions,
            ice_fine_motion_index=new_ice_fine_motion,
            ice_colors=new_ice_colors,
            ice_frac_accumulators=jnp.where(
                should_respawn,
                jnp.zeros(4, dtype=jnp.int32),
                state.ice_frac_accumulators
            ),
            ice_dx_last_frame=jnp.where(
                should_respawn,
                jnp.zeros(4, dtype=jnp.int32),
                state.ice_dx_last_frame
            ),
            completed_ice_blocks_delay=new_delay,
            bailey_death_frame=final_death_frame,
            score=new_score,
            building_igloo_idx=new_building_idx,
            remaining_lives=new_remaining_lives,
            temperature=new_temperature,
            igloo_entry_status=new_igloo_entry_status,
            obstacle_x=new_obstacle_x,
            obstacle_active=new_obstacle_active,
            polar_grizzly_x=new_polar_grizzly_x,
            polar_grizzly_frac_accumulator=new_polar_grizzly_frac,
            bailey_grizzly_collision_timer=new_bailey_grizzly_collision_timer,
            bailey_grizzly_collision_value=new_bailey_grizzly_collision_value
        )

        def _spawn_all(s):
            return self._spawn_obstacles_vec(s, jnp.ones((4,), dtype=jnp.bool_))

        updated_state = jax.lax.cond(
            should_respawn,
            _spawn_all,
            lambda s: s,
            updated_state
        )

        return updated_state
    
    def _add_bcd_score(self, score, points):
        """Add points to BCD score (Binary Coded Decimal).

        The original Atari used BCD format where each nibble (4 bits) represents
        a decimal digit 0-9. This method converts BCD points to decimal, adds them
        to the score, then stores the result back in our simplified BCD format.

        Args:
            score: 3-element array [ten-thousands/thousands, hundreds/tens, ones]
            points: Points to add in BCD format (e.g., 0x10 = 10 points, 0x90 = 90 points)

        Returns:
            Updated score array in the same format
        """
        # Convert the 3-element score array to a single decimal value
        # score[0] holds digits 5-4 (ten-thousands and thousands)
        # score[1] holds digits 3-2 (hundreds and tens)
        # score[2] holds digits 1-0 (ones)
        current = self._bcd_to_decimal(score)

        # Convert BCD points to decimal (handles up to 3 digits)
        # For values like 0x0100 (100 in BCD), extract all digits
        hundreds = (points >> 8) & 0x0F
        tens = (points >> 4) & 0x0F
        ones = points & 0x0F
        decimal_points = hundreds * 100 + tens * 10 + ones
        new_total = current + decimal_points

        # Convert back to BCD format (each byte stores 2 BCD digits)
        d5 = (new_total // 100000) % 10
        d4 = (new_total // 10000) % 10
        d3 = (new_total // 1000) % 10
        d2 = (new_total // 100) % 10
        d1 = (new_total // 10) % 10
        d0 = new_total % 10

        new_score = score.at[0].set((d5 << 4) | d4)
        new_score = new_score.at[1].set((d3 << 4) | d2)
        new_score = new_score.at[2].set((d1 << 4) | d0)

        return new_score
    
    def _check_extra_life(self, old_score, new_score, lives):
        """Check if score crossed a 5000-point boundary and award extra life.

        In Frostbite, players earn an extra life every 5000 points. This method
        checks if the score increase crossed one or more 5000-point thresholds
        and awards lives accordingly. Maximum of 9 reserve lives is enforced.

        Args:
            old_score: Previous score array [ten-thousands/thousands, hundreds/tens, ones]
            new_score: Updated score array after points were added
            lives: Current number of reserve lives

        Returns:
            Updated number of lives, capped at 9 (hardcoded limit)
        """
        # Convert BCD scores to decimal for threshold calculation
        old_total = self._bcd_to_decimal(old_score)
        new_total = self._bcd_to_decimal(new_score)

        # Calculate how many 5000-point thresholds were crossed
        # Integer division gives us the number of extra lives earned so far
        old_lives_earned = old_total // 5000
        new_lives_earned = new_total // 5000

        # Award extra lives for all thresholds crossed, but enforce the 9-life maximum
        # Note: MAX_RESERVED_LIVES constant exists but isn't used - limit is hardcoded
        lives_to_add = new_lives_earned - old_lives_earned
        new_lives = jnp.where(lives_to_add > 0, jnp.minimum(lives + lives_to_add, 9), lives)

        return new_lives
    
    def _add_bcd_score_decimal(self, score, decimal_points):
        """Add decimal points directly to BCD score.

        Unlike _add_bcd_score which expects BCD-formatted points, this method
        accepts points already converted to decimal. Used for temperature bonus
        scoring where the temperature value is already in a decimal format.

        Args:
            score: 3-element array [ten-thousands/thousands, hundreds/tens, ones]
            decimal_points: Points to add as a decimal integer (e.g., 45 for 45 points)

        Returns:
            Updated score array maintaining the same 3-element format
        """
        # Convert current score from 3-element array to decimal
        current = self._bcd_to_decimal(score)

        # Add the decimal points directly (no BCD conversion needed)
        new_total = current + decimal_points

        # Convert back to BCD format (each byte stores 2 BCD digits)
        d5 = (new_total // 100000) % 10
        d4 = (new_total // 10000) % 10
        d3 = (new_total // 1000) % 10
        d2 = (new_total // 100) % 10
        d1 = (new_total // 10) % 10
        d0 = new_total % 10

        new_score = score.at[0].set((d5 << 4) | d4)
        new_score = new_score.at[1].set((d3 << 4) | d2)
        new_score = new_score.at[2].set((d1 << 4) | d0)

        return new_score
    
    @partial(jax.jit, static_argnums=(0,))
    def _check_obstacle_collisions(self, state: FrostbiteState):
        """Check Bailey collision with obstacles and handle fish eating.

        This method performs collision detection between Bailey and all active
        obstacles (fish, crabs, clams, geese). It handles both harmful collisions
        (which kill Bailey) and beneficial ones (eating fish for points).

        The collision system accounts for sprite duplication patterns where a
        single obstacle can appear as multiple copies (e.g., a flock of geese).
        """
        if_checking = (state.bailey_alive == 1) & (state.bailey_visible == 1)
        bailey_w, bailey_h = 8, 18
        obs_w, obs_h = 8, 8
        is_jumping = state.bailey_jumping_idx > 0

        # Decode duplication for all lanes via table lookup (avoids per-lane calls)
        close = self.consts.SPACING_NARROW
        med   = self.consts.SPACING_MEDIUM
        wide  = self.consts.SPACING_WIDE
        #               0   1    2    3    4    5    6    7      (dup codes)
        copies_tbl  = jnp.array([1,  2,   2,   3,   2,   1,   3,   1], dtype=jnp.int32)
        spacing_tbl = jnp.array([0, close, med, close, wide, 0,  med,  0], dtype=jnp.int32)

        codes   = state.obstacle_duplication_mode
        copies  = copies_tbl[jnp.clip(codes, 0, 7)]
        spacing = spacing_tbl[jnp.clip(codes, 0, 7)]

        x0 = state.obstacle_x
        x1 = x0 + spacing
        x2 = x0 + 2 * spacing

        # X overlaps for the three potential copies
        xl = state.bailey_x
        def x_overlap(a_left, a_w, b_left, b_w):
            return (a_left < b_left + b_w) & (a_left + a_w > b_left)

        x_ov0 = x_overlap(xl, bailey_w, x0, obs_w)
        x_ov1 = x_overlap(xl, bailey_w, x1, obs_w)
        x_ov2 = x_overlap(xl, bailey_w, x2, obs_w)

        base_y = state.obstacle_y
        float_y = base_y + jnp.where(state.obstacle_types >= self.consts.ID_FISH, state.obstacle_float_offsets, 0)
        y_overlap = (state.bailey_y < float_y + obs_h) & (state.bailey_y + bailey_h > float_y)

        is_active = state.obstacle_active == 1
        gate = if_checking & is_active & (~is_jumping)

        is_fish = (state.obstacle_types == self.consts.ID_FISH)
        alive_mask = state.fish_alive_mask

        def alive_bit(mask, k): return ((mask >> jnp.int32(k)) & 1) == 1

        c0 = jnp.where(is_fish, x_ov0 & alive_bit(alive_mask, 0), x_ov0)
        c1 = jnp.where(is_fish, (copies >= 2) & x_ov1 & alive_bit(alive_mask, 1), (copies >= 2) & x_ov1)
        c2 = jnp.where(is_fish, (copies >= 3) & x_ov2 & alive_bit(alive_mask, 2), (copies >= 3) & x_ov2)

        any_x = c0 | c1 | c2
        collided_lane = gate & any_x & y_overlap

        # Select first collided index (0..3) if any
        lane_indices = jnp.arange(4, dtype=jnp.int32)
        masked = jnp.where(collided_lane, lane_indices, jnp.int32(999))
        has_hit = jnp.any(collided_lane)
        hit_idx = jnp.where(has_hit, jnp.argmin(masked), jnp.int32(-1))

        # Which copy (for fish) on the winning lane
        which_copy_per_lane = jnp.where(c0, 0, jnp.where(c1, 1, 2)).astype(jnp.int32)
        hit_copy = jnp.where(has_hit, which_copy_per_lane[jnp.clip(hit_idx, 0, 3)], jnp.int32(-1))

        # Apply effects
        hit_type = jnp.where(has_hit, state.obstacle_types[jnp.clip(hit_idx, 0, 3)], jnp.int32(-1))
        hit_is_fish = has_hit & (hit_type == self.consts.ID_FISH)
        can_award = hit_is_fish & (state.number_of_fish_eaten < self.consts.MAX_EATEN_FISH)

        new_score = jax.lax.cond(
            can_award,
            lambda s: self._add_bcd_score(s, jnp.int32(0x0200)),
            lambda s: s,
            state.score
        )
        new_remaining_lives = jax.lax.cond(
            can_award,
            lambda _: self._check_extra_life(state.score, new_score, state.remaining_lives),
            lambda _: state.remaining_lives,
            operand=None
        )
        new_eaten = jnp.where(can_award, state.number_of_fish_eaten + 1, state.number_of_fish_eaten)

        # Clear only that fish copy bit
        def clear_bit(mask, k):
            return mask & ~(jnp.left_shift(jnp.int32(1), jnp.clip(k, 0, 2)))

        idx = jnp.clip(hit_idx, 0, 3)
        new_mask_at_idx = jax.lax.cond(
            hit_is_fish,
            lambda _: clear_bit(state.fish_alive_mask[idx], hit_copy),
            lambda _: state.fish_alive_mask[idx],
            operand=None
        )
        new_fish_masks = state.fish_alive_mask.at[idx].set(
            jnp.where(hit_is_fish, new_mask_at_idx, state.fish_alive_mask[idx])
        )

        # If the school is now empty: force immediate despawn of that lane
        forced_despawn_x = jnp.int32(self.consts.SCREEN_WIDTH + self.consts.OBSTACLE_DESPAWN_X + 1)
        empty_school = hit_is_fish & (new_mask_at_idx == 0)
        new_obstacle_x = state.obstacle_x.at[idx].set(
            jnp.where(empty_school, forced_despawn_x, state.obstacle_x[idx])
        )

        # Push Bailey once for harmful obstacles
        push_amount = jnp.where(
            has_hit & ~hit_is_fish,
            state.obstacle_dx_last_frame[idx],
            jnp.int32(0)
        )
        being_chased = state.bailey_grizzly_collision_value != 0
        min_x = jnp.where(being_chased, 0, self.consts.ICE_X_MIN)
        new_bailey_x = jnp.clip(state.bailey_x + push_amount, min_x, self.consts.ICE_X_MAX)

        # Collision flag: clear if fish, else set to lane index; -1 if none
        new_collision_flag = jnp.where(
            hit_is_fish, jnp.int32(-1),
            jnp.where(has_hit, idx, jnp.int32(-1))
        )

        return state._replace(
            bailey_x=new_bailey_x,
            bailey_obstacle_collision_idx=new_collision_flag,
            obstacle_collision_index=jnp.where(has_hit & ~hit_is_fish, idx, state.obstacle_collision_index),
            number_of_fish_eaten=new_eaten,
            fish_alive_mask=new_fish_masks,
            obstacle_x=new_obstacle_x,
            score=new_score,
            remaining_lives=new_remaining_lives
        )
    
    def _update_polar_grizzly(self, state: FrostbiteState):
        """Update Polar Grizzly (bear) movement and collision.

        The polar bear appears starting at level 3 and patrols the shore area.
        When it catches Bailey, it drags him to the left shore boundary (x=8),
        resulting in death. The bear moves at a constant speed and reverses
        direction when reaching the screen edges.

        Collision mechanics:
        - Bear can only catch Bailey when on shore (not jumping)
        - Once caught, Bailey is dragged left at bear's speed
        - Death occurs when Bailey is dragged to the shore boundary (x <= 8)
        """
        level_idx = jnp.maximum(state.level - 1, 0)
        is_active = level_idx >= self.consts.POLAR_GRIZZLY_LEVEL
        new_active = jnp.where(is_active, 1, 0)

        # Bear-Bailey collision detection
        bailey_on_shore = state.bailey_y == self.consts.YMIN_BAILEY
        bear_width = jnp.int32(20)  # Bear sprite is wider than most
        bailey_width = jnp.int32(16)  # Bailey's collision width

        # Calculate bounding boxes
        bear_left = state.polar_grizzly_x
        bear_right = state.polar_grizzly_x + bear_width
        bailey_left = state.bailey_x
        bailey_right = state.bailey_x + bailey_width

        # Check for overlap using AABB collision
        overlap = (bear_right > bailey_left) & (bear_left < bailey_right)

        # Bear can only catch Bailey when:
        # - Bear is active (level 3+)
        # - Bailey is on shore (not on ice)
        # - Sprites are overlapping
        # - Bailey is alive
        # - Bailey is not jumping (can jump over bear)
        has_collision = (
            is_active &
            bailey_on_shore &
            overlap &
            (state.bailey_alive == 1) &
            (state.bailey_jumping_idx == 0)
        )

        # Track if Bailey is being dragged by the bear
        was_already_caught = state.bailey_grizzly_collision_value != 0
        # Set collision flag when bear catches Bailey
        new_collision_value = jnp.where(
            is_active & (was_already_caught | has_collision),
            jnp.int32(0x80),  # Collision flag value
            jnp.int32(0)
        )

        # Check if Bailey is currently being chased/dragged
        bailey_being_chased = new_collision_value != 0

        # Define bear movement boundaries
        # Safe zone: Bear can't go closer to left edge
        # This creates a safe area where Bailey can hide
        left_bound = jnp.int32(self.consts.XMIN + 26)  # Safe zone on left
        right_bound = jnp.int32(self.consts.SCREEN_WIDTH - 20)  # Bear sprite width

        # Calculate bear movement using fractional accumulator system
        # This provides smooth sub-pixel movement
        base_speed_frac = jnp.where(is_active, state.obstacle_speed_frac, jnp.int32(0))
        polar_frac_sum = state.polar_grizzly_frac_accumulator + base_speed_frac
        extra_pixel = polar_frac_sum // 16  # Whole pixel when accumulator overflows
        new_polar_frac = jnp.where(is_active, polar_frac_sum % 16, state.polar_grizzly_frac_accumulator)
        base_speed_whole = state.obstacle_speed_whole.astype(jnp.int32)
        polar_pixels = jnp.where(
            is_active,
            base_speed_whole + extra_pixel.astype(jnp.int32),
            jnp.int32(0)
        )

        # Animation timing
        frame_mod4 = state.frame_count & 3  # For chase movement (every 4th frame)
        bailey_chase_animation = (state.frame_count >> 1) & 3  # Bailey's dragged animation

        def chase_branch(_):
            """Handle bear dragging Bailey off-screen to his doom."""
            # Bear moves left slowly (3 out of 4 frames) while dragging Bailey
            bear_moves = (frame_mod4 != 0) & (state.polar_grizzly_x > left_bound)
            chase_bear_x = jnp.where(bear_moves, state.polar_grizzly_x - 1, state.polar_grizzly_x)

            # Bailey is dragged left at same speed
            chase_bailey_x = state.bailey_x - 1

            # Bailey dies when dragged to the shore minimum boundary (x <= SHORE_X_MIN)
            bailey_should_die = (chase_bailey_x <= self.consts.SHORE_X_MIN) & bailey_on_shore
            new_bailey_alive = jnp.where(bailey_should_die, 0, state.bailey_alive)
            new_death_frame = jnp.where(bailey_should_die & (state.bailey_alive == 1), 1, state.bailey_death_frame)

            # Update chase timer for animation
            chase_timer = state.bailey_grizzly_collision_timer + 1
            chase_anim_idx = ((chase_timer >> 3) & 7).astype(jnp.int32)  # 8-frame animation cycle, slower (every 8 frames)

            return state._replace(
                polar_grizzly_active=new_active,
                polar_grizzly_x=chase_bear_x,
                polar_grizzly_direction=jnp.int32(1),
                polar_grizzly_animation_idx=chase_anim_idx,
                polar_grizzly_frac_accumulator=new_polar_frac,
                bailey_grizzly_collision_value=new_collision_value,
                bailey_grizzly_collision_timer=chase_timer,
                bailey_x=chase_bailey_x,
                bailey_direction=jnp.int32(1),
                bailey_animation_idx=bailey_chase_animation.astype(jnp.int32),
                bailey_alive=new_bailey_alive,
                bailey_death_frame=new_death_frame
            )

        def patrol_branch(_):
            """Handle bear's normal patrol and pursuit behavior.

            The bear has two AI modes:
            1. Pursuit mode: When Bailey is on shore, actively chase him
            2. Patrol mode: When Bailey escapes to ice, patrol back and forth

            The bear bounces off screen edges and maintains the safe zone boundary.
            """
            desired_dir = state.polar_grizzly_direction

            # AI Pursuit Mode: When Bailey is on shore, intelligently chase toward him
            # Turn left if Bailey is to the left
            desired_dir = jnp.where(
                bailey_on_shore & is_active & (state.bailey_x < state.polar_grizzly_x),
                jnp.int32(1),  # Direction 1 = moving left
                desired_dir
            )
            # Turn right if Bailey is to the right
            desired_dir = jnp.where(
                bailey_on_shore & is_active & (state.bailey_x > state.polar_grizzly_x),
                jnp.int32(0),  # Direction 0 = moving right
                desired_dir
            )

            # AI Patrol Mode: When Bailey escapes to ice, patrol shore area
            # Bounce off left boundary (respecting safe zone)
            desired_dir = jnp.where(
                (~bailey_on_shore) & is_active & (state.polar_grizzly_direction == 1) & (state.polar_grizzly_x <= left_bound),
                jnp.int32(0),  # Turn around to right
                desired_dir
            )
            # Bounce off right boundary
            desired_dir = jnp.where(
                (~bailey_on_shore) & is_active & (state.polar_grizzly_direction == 0) & (state.polar_grizzly_x >= right_bound),
                jnp.int32(1),  # Turn around to left
                desired_dir
            )

            # Calculate movement delta based on direction
            raw_delta = jnp.where(
                desired_dir == 1,
                -polar_pixels,  # Moving left (negative delta)
                polar_pixels    # Moving right (positive delta)
            )

            # Ensure bear stays within boundaries (safe zone on left, screen edge on right)
            max_left = state.polar_grizzly_x - left_bound   # Maximum leftward movement
            max_right = right_bound - state.polar_grizzly_x  # Maximum rightward movement
            delta = jnp.clip(raw_delta, -max_left, max_right)  # Clamp to valid range
            next_x = state.polar_grizzly_x + delta

            # Animation system: cycles through 8 frames (7 down to 0)
            # Slow down animation by only updating every 2 frames
            step_nonzero = delta != 0  # Only animate when actually moving
            should_animate_frame = (state.frame_count & 1) == 0  # Update every 2nd frame
            anim_gate = step_nonzero & should_animate_frame
            current_anim = state.polar_grizzly_animation_idx
            # Decrement animation counter, wrap from 0 to 7
            decremented_anim = jnp.where(current_anim > 0, current_anim - 1, jnp.int32(7))
            new_anim = jnp.where(anim_gate, decremented_anim, current_anim)

            # Update direction only when bear is active
            new_direction = jnp.where(is_active, desired_dir, state.polar_grizzly_direction)

            return state._replace(
                polar_grizzly_active=new_active,
                polar_grizzly_x=next_x,
                polar_grizzly_direction=new_direction,
                polar_grizzly_animation_idx=new_anim,
                polar_grizzly_frac_accumulator=new_polar_frac,
                bailey_grizzly_collision_value=new_collision_value,
                bailey_grizzly_collision_timer=jnp.int32(0)  # Reset timer when not chasing
            )

        # Branch between chase mode (dragging Bailey) and patrol mode (normal movement)
        return jax.lax.cond(
            bailey_being_chased,  # Condition: is bear currently dragging Bailey?
            chase_branch,         # True: execute chase/drag behavior
            patrol_branch,        # False: execute normal patrol behavior
            operand=None
        )
    
    def _check_igloo_entry(self, state: FrostbiteState, action: int):
        """Check if Bailey can enter the completed igloo to end the level.

        The igloo serves as the level exit. Once all 15 ice blocks have been
        collected to build it, Bailey can enter by pressing UP while standing
        at the door. This triggers the level completion sequence.

        Entry conditions:
        - Igloo must be complete (15 blocks collected)
        - Bailey must be on shore (not on ice)
        - Bailey must be near the door position
        - Player must press UP
        - Bailey must be alive and not jumping

        Args:
            state: Current game state
            action: Player's input action

        Returns:
            Updated state with igloo entry status and level completion flags
        """
        # Check current entry status
        already_entering = state.igloo_entry_status == 0x80  # Entry animation in progress
        igloo_complete = state.building_igloo_idx >= self.consts.MAX_IGLOO_INDEX  # All 15 blocks collected

        # Position checks
        at_shore = jnp.abs(state.bailey_y - self.consts.YMIN_BAILEY) <= 2  # Within 2 pixels of shore Y
        not_jumping = state.bailey_jumping_idx == 0
        not_entering = state.igloo_entry_status == 0

        # Door collision box (igloo door is at specific X position)
        door_x = 122
        bailey_right_edge = state.bailey_x + 16
        near_door = (bailey_right_edge >= door_x) & (state.bailey_x <= door_x + 8)

        # Check if player is pressing UP (or diagonal up)
        pressing_up = (action == Action.UP) | (action == Action.UPLEFT) | (action == Action.UPRIGHT)

        # Determine if Bailey can start entering igloo
        can_enter = igloo_complete & at_shore & near_door & pressing_up & (state.bailey_alive == 1) & not_jumping & not_entering

        # Entry animation completes when Bailey reaches top of screen (y <= 6)
        completed_entry = already_entering & (state.bailey_y <= 6)

        # Continue entry if started but not completed
        should_be_entering = can_enter | (already_entering & ~completed_entry)

        # Update entry status: 0x80 = entering, 0 = not entering/completed
        new_entry_status = jnp.where(completed_entry, 0, jnp.where(should_be_entering, 0x80, state.igloo_entry_status))

        # Bailey becomes invisible after entering igloo
        new_visible = jnp.where(completed_entry, 0, state.bailey_visible)

        # Trigger level completion sequence when entry animation finishes
        new_level_status = jnp.where(completed_entry, self.consts.LEVEL_COMPLETE, state.current_level_status)
        new_frame_delay = jnp.where(completed_entry, self.consts.INIT_DELAY_ACTION_VALUE, state.frame_delay)

        return state._replace(
            bailey_visible=new_visible,
            igloo_entry_status=new_entry_status,
            current_level_status=new_level_status,
            frame_delay=new_frame_delay
        )
    
    def render(self, state: FrostbiteState) -> jnp.ndarray:
        """Render the game"""
        return self.renderer.render(state)
    
    def action_space(self):
        """Return the action space"""
        return spaces.Discrete(18)
    
    def observation_space(self) -> spaces.Dict:
        """Return the observation space as Dict matching FrostbiteObservation fields"""
        return spaces.Dict({
            "bailey_x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
            "bailey_y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
            "bailey_animation": spaces.Box(low=0, high=10, shape=(), dtype=jnp.int32),
            "bailey_jumping": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
            "frame_count": spaces.Box(low=0, high=2**31-1, shape=(), dtype=jnp.int32),
        })

    def obs_to_flat_array(self, obs: FrostbiteObservation) -> jnp.ndarray:
        """Convert observation to flat array"""
        return jnp.concatenate([
            obs.bailey_x.flatten(),
            obs.bailey_y.flatten(),
            obs.bailey_animation.flatten(),
            obs.bailey_jumping.flatten(),
            obs.frame_count.flatten(),
        ])

    def image_space(self) -> spaces.Box:
        """Returns the image space for Frostbite.
        The image is a RGB image with shape (210, 160, 3).
        """
        return spaces.Box(
            low=0,
            high=255,
            shape=(210, 160, 3),
            dtype=jnp.uint8
        )


# ==========================================================================================
# RENDERER
# ==========================================================================================

class FrostbiteRenderer(JAXGameRenderer):
    """Renderer for the Frostbite game with sprite-based graphics.

    This renderer handles all visual aspects of the game including:
    - Bailey's multiple animation states (walking, jumping, death)
    - Ice blocks with white/blue coloring for temperature states
    - Obstacles (fish, crabs, clams, geese) with sprite duplication
    - Polar bear animations (appears at level 3+)
    - Igloo construction progress
    - HUD elements (score, temperature, lives)
    - Day/night cycle (switches every 4 levels)

    The renderer uses pre-loaded numpy arrays (.npy files) for sprites
    and composites them onto a 160x210 pixel display using JAX operations.
    """

    def __init__(self, consts: FrostbiteConstants = None):
        """Initialize the renderer with game constants and load all sprites.

        Args:
            consts: Game constants defining screen dimensions and positions
        """
        super().__init__()
        self.consts = consts or FrostbiteConstants()
        
        # 1. Configure the new renderer
        self.config = render_utils.RendererConfig(
            game_dimensions=(self.consts.SCREEN_HEIGHT, self.consts.SCREEN_WIDTH),
            channels=3,
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)
        self.sprite_path = os.path.join(os.path.dirname(__file__), "sprites", "frostbite")
        # 2. Call the asset preparation helper
        self._load_and_prepare_assets()
    
    def _load_frame_legacy(self, file_name):
        """Helper to load a single .npy file as an RGBA array."""
        # We use a temporary instance of the *new* utils class,
        # which has the same .loadFrame utility method.
        return render_utils.JaxRenderingUtils.loadFrame(
            self, os.path.join(self.sprite_path, file_name)
        )
    
    def _load_and_prepare_assets(self):
        """
        Loads all base sprites and pre-generates all required variations
        (tints, flips, masks) as RGBA arrays.
        """
        
        # --- Load Base Sprites ---
        bg_day = self._load_frame_legacy("background_day.npy")
        bg_night = self._load_frame_legacy("background_night.npy")
        bailey_0 = self._load_frame_legacy("bailey_walking_00.npy")
        bailey_1 = jnp.flip(self._load_frame_legacy("bailey_walking_01.npy"), axis=1)
        bailey_2 = jnp.flip(self._load_frame_legacy("bailey_jumping_00.npy"), axis=1)
        bailey_3 = self._load_frame_legacy("bailey_death_00.npy")
        bailey_4 = self._load_frame_legacy("bailey_death_01.npy")
        
        ice_wide_white = self._load_frame_legacy("ice_float_wide.npy")
        ice_narrow_white = self._load_frame_legacy("ice_float_narrow.npy")
        geese_0 = self._load_frame_legacy("snow_geese_00.npy")
        geese_1 = self._load_frame_legacy("snow_geese_01.npy")
        fish_0 = self._load_frame_legacy("fish_00.npy")
        fish_1 = jnp.flip(self._load_frame_legacy("fish_01.npy"), axis=1)
        crab_0 = self._load_frame_legacy("king_crab_00.npy")
        crab_1 = self._load_frame_legacy("king_crab_01.npy")
        clam_0 = jnp.flip(self._load_frame_legacy("clam_00.npy"), axis=1)
        clam_1 = jnp.flip(self._load_frame_legacy("clam_01.npy"), axis=1)
        bear_0 = self._load_frame_legacy("bear_00.npy")
        bear_1 = self._load_frame_legacy("bear_01.npy")
        igloo_block = self._load_frame_legacy("igloo_block_00.npy")
        igloo_door = self._load_frame_legacy("igloo_door.npy")
        degree_symbol = self._load_frame_legacy("degree_symbol.npy")
        
        # Use legacy digit loader
        digit_path_pattern = os.path.join(self.sprite_path, "digit_{}.npy")
        digits_array = jr_legacy.load_and_pad_digits(digit_path_pattern, num_chars=10)
        digits_list = [digits_array[i] for i in range(10)]
        
        # --- Pre-generate Variations ---
        
        # Bailey (Frozen)
        bailey_frames = [bailey_0, bailey_1, bailey_2, bailey_3, bailey_4]
        bailey_frozen_frames = [self._apply_frozen_tint(f) for f in bailey_frames]
        
        # Ice (Blue)
        ice_wide_blue = self._apply_ice_color(ice_wide_white, is_blue=True)
        ice_narrow_blue = self._apply_ice_color(ice_narrow_white, is_blue=True)
        
        # Bear (Lightened for Night)
        bear_0_light = self._lighten_bear(bear_0)
        bear_1_light = self._lighten_bear(bear_1)
        
        # Obstacles (Floating)
        # Max float offset is 4. We need 5 variations (0, 1, 2, 3, 4)
        float_offsets = jnp.arange(5)
        
        def create_float_variations(sprite):
            return jnp.stack([self._apply_float_mask(sprite, offset) for offset in float_offsets])
        
        fish_0_floats = create_float_variations(fish_0)     # (5, H, W, 4)
        fish_1_floats = create_float_variations(fish_1)     # (5, H, W, 4)
        crab_0_floats = create_float_variations(crab_0)     # (5, H, W, 4)
        crab_1_floats = create_float_variations(crab_1)     # (5, H, W, 4)
        clam_0_floats = create_float_variations(clam_0)     # (5, H, W, 4)
        clam_1_floats = create_float_variations(clam_1)     # (5, H, W, 4)
        
        # 3. Start from asset config in constants and extend with dynamically loaded assets
        final_asset_config = list(self.consts.ASSET_CONFIG)
        
        # Get black_bar from constants (already in ASSET_CONFIG)
        black_bar = None
        for asset in final_asset_config:
            if asset['name'] == 'black_bar':
                black_bar = asset['data']
                break
        
        # Add dynamically loaded and processed assets
        final_asset_config.extend([
            {'name': 'background_day', 'type': 'background', 'data': bg_day},
            {'name': 'background_night', 'type': 'procedural', 'data': bg_night},
            
            # Bailey (5 frames)
            {'name': 'bailey', 'type': 'group', 'data': bailey_frames},
            {'name': 'bailey_frozen', 'type': 'group', 'data': bailey_frozen_frames},
            
            # Ice (4 variations: wide/narrow, white/blue)
            {'name': 'ice', 'type': 'group', 'data': [ice_wide_white, ice_wide_blue, ice_narrow_white, ice_narrow_blue]},
            
            # Obstacles (pre-floated)
            {'name': 'geese', 'type': 'group', 'data': [geese_0, geese_1]}, # 2 frames
            {'name': 'fish', 'type': 'group', 'data': jnp.concatenate([fish_0_floats, fish_1_floats], axis=0)}, # 10 frames
            {'name': 'crab', 'type': 'group', 'data': jnp.concatenate([crab_0_floats, crab_1_floats], axis=0)}, # 10 frames
            {'name': 'clam', 'type': 'group', 'data': jnp.concatenate([clam_0_floats, clam_1_floats], axis=0)}, # 10 frames
            
            # Bear (4 variations: 2 anim frames, day/night)
            {'name': 'bear', 'type': 'group', 'data': [bear_0, bear_1]},
            {'name': 'bear_light', 'type': 'group', 'data': [bear_0_light, bear_1_light]},
            
            # UI (1 igloo block, 1 door, 1 degree, 10 digits)
            {'name': 'igloo', 'type': 'group', 'data': [igloo_block, igloo_door]},
            {'name': 'degree', 'type': 'procedural', 'data': degree_symbol},
            {'name': 'digits', 'type': 'group', 'data': digits_list},
        ])
        
        asset_config = final_asset_config
        
        # 4. Load all assets and build palette/masks
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(asset_config, self.sprite_path)
        
        # 5. Store helper dimensions
        self.DIGIT_MASKS = self.SHAPE_MASKS['digits']
        self.BAILEY_MASKS = self.SHAPE_MASKS['bailey']
        self.BAILEY_FROZEN_MASKS = self.SHAPE_MASKS['bailey_frozen']
        self.ICE_MASKS = self.SHAPE_MASKS['ice']
        self.GEESE_MASKS = self.SHAPE_MASKS['geese']
        self.FISH_MASKS = self.SHAPE_MASKS['fish']
        self.CRAB_MASKS = self.SHAPE_MASKS['crab']
        self.CLAM_MASKS = self.SHAPE_MASKS['clam']
        self.BEAR_MASKS = self.SHAPE_MASKS['bear']
        self.BEAR_LIGHT_MASKS = self.SHAPE_MASKS['bear_light']
        self.IGLOO_BLOCK_MASK = self.SHAPE_MASKS['igloo'][0]
        self.IGLOO_DOOR_MASK = self.SHAPE_MASKS['igloo'][1]
        self.DEGREE_MASK = self.SHAPE_MASKS['degree']
        self.BLACK_BAR_MASK = self.SHAPE_MASKS['black_bar']
        
        # Convert ICE_ROW_Y tuple to JAX array for dynamic indexing
        self.ICE_ROW_Y_ARRAY = jnp.array(self.consts.ICE_ROW_Y, dtype=jnp.int32)
    
    @staticmethod
    def _decode_sprite_duplication(consts, code: jnp.ndarray):
        """Decode sprite duplication mode to get number of copies and spacing.
        (Static copy of game logic needed for rendering)
        """
        close = consts.SPACING_NARROW
        med   = consts.SPACING_MEDIUM
        wide  = consts.SPACING_WIDE
        copies = jnp.array(1, dtype=jnp.int32)
        spacing = jnp.array(0, dtype=jnp.int32)
        copies = jnp.where(code == consts.SPRITE_SINGLE, 1, copies)
        spacing = jnp.where(code == consts.SPRITE_SINGLE, 0, spacing)
        copies = jnp.where(code == consts.SPRITE_DOUBLE, 2, copies)
        spacing = jnp.where(code == consts.SPRITE_DOUBLE, close, spacing)
        copies = jnp.where(code == consts.SPRITE_DOUBLE_SPACED, 2, copies)
        spacing = jnp.where(code == consts.SPRITE_DOUBLE_SPACED, med, spacing)
        copies = jnp.where(code == consts.SPRITE_DOUBLE_WIDE, 2, copies)
        spacing = jnp.where(code == consts.SPRITE_DOUBLE_WIDE, wide, spacing)
        copies = jnp.where(code == consts.SPRITE_TRIPLE, 3, copies)
        spacing = jnp.where(code == consts.SPRITE_TRIPLE, close, spacing)
        copies = jnp.where(code == consts.SPRITE_TRIPLE_SPACED, 3, copies)
        spacing = jnp.where(code == consts.SPRITE_TRIPLE_SPACED, med, spacing)
        return copies, spacing
    
    # --- Tinting helpers (used only in __init__) ---
    
    @staticmethod
    def _apply_ice_color(block_sprite, is_blue):
        """Apply color tinting to ice block sprites."""
        if not is_blue:
            return block_sprite
        
        # Apply blue tint
        return jnp.where(
            block_sprite[..., 3:4] > 0,  # Only tint non-transparent pixels
            jnp.concatenate([
                (block_sprite[..., 0:1] * 0.3).astype(block_sprite.dtype),  # Reduce red channel
                (block_sprite[..., 1:2] * 0.5).astype(block_sprite.dtype),  # Reduce green channel
                (block_sprite[..., 2:3] * 0.9).astype(block_sprite.dtype),  # Keep most blue
                block_sprite[..., 3:4]  # Preserve alpha channel
            ], axis=-1), block_sprite).astype(block_sprite.dtype)
    
    @staticmethod
    def _apply_frozen_tint(frame_bailey):
        """Apply blue tint when Bailey is frozen."""
        return jnp.where(
            frame_bailey[..., 3:4] > 0,  # Only tint non-transparent pixels
            jnp.concatenate([
                jnp.ones_like(frame_bailey[..., 0:1]) * 100,  # Reduce red
                jnp.ones_like(frame_bailey[..., 1:2]) * 160,  # Slightly reduce green
                jnp.ones_like(frame_bailey[..., 2:3]) * 220,  # Keep blue high
                frame_bailey[..., 3:4]  # Preserve alpha
            ], axis=-1), frame_bailey).astype(frame_bailey.dtype)

    @staticmethod
    def _lighten_bear(sprite):
        """Make bear lighter during night cycles for better visibility."""
        return jnp.where(
            sprite[..., 3:4] > 0,  # Where alpha > 0
            jnp.concatenate([
                jnp.minimum(sprite[..., 0:1] + 40, 255).astype(sprite.dtype),  # R
                jnp.minimum(sprite[..., 1:2] + 40, 255).astype(sprite.dtype),  # G
                jnp.minimum(sprite[..., 2:3] + 40, 255).astype(sprite.dtype),  # B
                sprite[..., 3:4]  # Keep alpha unchanged
            ], axis=-1),
            sprite
        )

    @staticmethod
    def _apply_float_mask(s, float_offset_int):
        """Mask sprite to create partial submersion effect."""
        rows = jnp.arange(s.shape[0], dtype=jnp.int32).reshape((-1, 1, 1))
        height = jnp.int32(s.shape[0])
        cutoff = jnp.maximum(height - float_offset_int, 0)  # Hide bottom pixels
        mask = rows < cutoff  # Only show pixels above water line
        return jnp.where(mask, s, jnp.zeros_like(s))

    # --- JIT-compiled Render Helpers ---
    @partial(jax.jit, static_argnums=(0,))
    def _render_with_wrap(self, raster, x, y, sprite_mask):
        """Render a sprite with horizontal wrapping at playfield boundaries."""
        W = self.consts.PLAYFIELD_WIDTH
        
        # Draw main sprite (clipped to screen edges)
        raster = self.jr.render_at_clipped(raster, x, y, sprite_mask)
        
        # Draw wrap copies
        raster = self.jr.render_at_clipped(raster, x - W, y, sprite_mask)
        raster = self.jr.render_at_clipped(raster, x + W, y, sprite_mask)
        
        return raster

    @partial(jax.jit, static_argnums=(0,))
    def _bcd_to_decimal(self, bcd_score: jnp.ndarray) -> jnp.ndarray:
        """Convert 3-byte BCD score to decimal value."""
        return (((bcd_score[0] >> 4) & 0xF) * 100000 +
                (bcd_score[0] & 0xF) * 10000 +
                ((bcd_score[1] >> 4) & 0xF) * 1000 +
                (bcd_score[1] & 0xF) * 100 +
                ((bcd_score[2] >> 4) & 0xF) * 10 +
                (bcd_score[2] & 0xF)).astype(jnp.int32)
    
    @partial(jax.jit, static_argnums=(0,))
    def _render_hud(self, raster, state):
        """Render HUD elements at top of screen."""
        digits_masks = self.DIGIT_MASKS
        score_y, lives_y, temp_y = 10, 22, 22
        should_flash = state.temperature < 0x10
        is_visible = ~should_flash | ((state.frame_count % 90) < 45)
        # Render lives
        lives_x = 59
        lives_clamped = jnp.clip(state.remaining_lives, 0, 9)
        lives_sprite = digits_masks[lives_clamped]
        raster = jax.lax.cond(
            is_visible, lambda r: self.jr.render_at(r, lives_x, lives_y, lives_sprite), lambda r: r, raster
        )
        
        # Render temperature
        temp_x_base = lives_x - 16 - 6
        temp_tens, temp_ones = (state.temperature >> 4) & 0x0F, state.temperature & 0x0F
        temp_x = jnp.where(temp_tens > 0, temp_x_base - 16, temp_x_base - 8)
        x_offset = temp_x
        tens_sprite = digits_masks[temp_tens]
        raster = jax.lax.cond(
            (temp_tens > 0) & is_visible, lambda r, x: self.jr.render_at(r, x, temp_y, tens_sprite),
            lambda r, x: r, raster, x_offset
        )
        x_offset += jnp.where(temp_tens > 0, 8, 0)
        ones_sprite = digits_masks[temp_ones]
        raster = jax.lax.cond(
            is_visible, lambda r: self.jr.render_at(r, x_offset, temp_y, ones_sprite), lambda r: r, raster
        )
        degree_sprite = self.DEGREE_MASK
        raster = jax.lax.cond(
            is_visible, lambda r: self.jr.render_at(r, temp_x_base, temp_y, degree_sprite), lambda r: r, raster
        )
        
        # Render score
        total_score = self._bcd_to_decimal(state.score)
        num_digits = jnp.where(
            total_score == 0,
            1,
            jnp.floor(jnp.log10(jnp.maximum(1, total_score) + 0.5)).astype(jnp.int32) + 1
        )
        score_x_start = lives_x - ((num_digits - 1) * 4)
        score_digits = self.jr.int_to_digits(total_score, max_digits=6)
        start_index = 6 - num_digits
        
        raster = self.jr.render_label_selective(
            raster, score_x_start, score_y,
            score_digits, digits_masks,
            start_index, num_digits,
            spacing=8,
            max_digits_to_render=6
        )
        
        return raster
    
    @partial(jax.jit, static_argnums=(0,))
    def _render_igloo_blocks(self, raster, state):
        """Render igloo blocks in the specific construction pattern."""
        block_mask = self.IGLOO_BLOCK_MASK
        door_mask = self.IGLOO_DOOR_MASK
        base_x, base_y = self.consts.IGLOO_X, self.consts.IGLOO_Y
        igloo_offset = -43
        
        # Precompute all positions
        x_pos = jnp.array([
            base_x + igloo_offset + 0,  # 0
            base_x + igloo_offset + 8,  # 1
            base_x + igloo_offset + 16, # 2
            base_x + igloo_offset + 24, # 3
            base_x + igloo_offset + 24, # 4
            base_x + igloo_offset + 16, # 5
            base_x + igloo_offset + 8,  # 6
            base_x + igloo_offset + 0,  # 7
            base_x + igloo_offset + 0,  # 8
            base_x + igloo_offset + 8,  # 9
            base_x + igloo_offset + 16, # 10
            base_x + igloo_offset + 24, # 11
            base_x + igloo_offset + 4,  # 12
            base_x + igloo_offset + 16, # 13
            base_x + igloo_offset + 8,  # 14
            122                       # 15 (door)
        ], dtype=jnp.int32)
        y_pos = jnp.array([
            base_y + 7, base_y + 7, base_y + 7, base_y + 7, # 0-3
            base_y + 3, base_y + 3, base_y + 3, base_y + 3, # 4-7
            base_y - 1, base_y - 1, base_y - 1, base_y - 1, # 8-11
            base_y - 5, base_y - 5, # 12-13
            base_y - 9, # 14
            base_y + 3  # 15 (door)
        ], dtype=jnp.int32)
        
        def render_block(i, r):
            should_render = i <= state.building_igloo_idx
            x, y = x_pos[i], y_pos[i]
            
            def draw_fn(r_in):
                # Render block 15 (door)
                r_out = jax.lax.cond(
                    i == 15,
                    lambda r: self.jr.render_at(r, x, y, door_mask),
                    lambda r: r,
                    r_in)
                
                # Render blocks 0-14
                r_out = jax.lax.cond(
                    i < 15,
                    lambda r: self.jr.render_at(r, x, y, block_mask),
                    lambda r: r,
                    r_out)
                
                # Render extra sprites for wider blocks
                r_out = jax.lax.cond(
                    (i >= 12) & (i <= 13), # Blocks 12, 13
                    lambda r: self.jr.render_at(r, x + 4, y, block_mask),
                    lambda r: r,
                    r_out)
                r_out = jax.lax.cond(
                    i == 14, # Block 14
                    lambda r: self.jr.render_at(r, x + 8, y, block_mask),
                    lambda r: r,
                    r_out)
                
                return r_out
            return jax.lax.cond(should_render, draw_fn, lambda r: r, r)
        raster = jax.lax.fori_loop(0, 16, render_block, raster)
        return raster
    
    @partial(jax.jit, static_argnums=(0,))
    def _render_obstacles(self, raster, state):
        """Render all active obstacles with sprite duplication patterns."""
        
        def render_lane(i, r):
            if_render = state.obstacle_active[i] == 1
            
            def draw_lane(r_in):
                anim_idx = state.obstacle_animation_idx[i]
                direction = state.obstacle_directions[i]
                x_pos = state.obstacle_x[i]
                y_pos = state.obstacle_y[i]
                float_offset_int = jnp.clip(state.obstacle_float_offsets[i].astype(jnp.int32), 0, 4)
                obstacle_type = state.obstacle_types[i]
                alive_mask_i = state.fish_alive_mask[i]
                
                copies, spacing = self._decode_sprite_duplication(self.consts, state.obstacle_duplication_mode[i])
                
                # Helper function to render a single obstacle type
                def render_obstacle_type(base_sprite, is_aquatic_flag):
                    # Adjust Y position for floating
                    y_render = jnp.where(is_aquatic_flag, y_pos + float_offset_int, y_pos)
                    # Flip sprite based on direction
                    sprite = jax.lax.cond(
                        direction == 1, lambda s: jnp.flip(s, axis=1), lambda s: s, base_sprite
                    )
                    
                    # Render copies
                    def render_copy(k, r_copy):
                        xp = x_pos + k * spacing
                        copy_alive = jnp.where(
                            obstacle_type == self.consts.ID_FISH,
                            ((alive_mask_i >> jnp.int32(k)) & 1) == 1,
                            True
                        )
                        should_show = (k < copies) & copy_alive
                        
                        return jax.lax.cond(
                            should_show,
                            lambda r: self.jr.render_at_clipped(r, xp, y_render, sprite),
                            lambda r: r,
                            r_copy
                        )
                    
                    return jax.lax.fori_loop(0, 3, render_copy, r_in)
                
                # Select the correct sprite animation stack
                # flat_idx = anim_idx * 5 + float_offset_int
                fish_mask = self.FISH_MASKS[anim_idx * 5 + float_offset_int]
                crab_mask = self.CRAB_MASKS[anim_idx * 5 + float_offset_int]
                clam_mask = self.CLAM_MASKS[anim_idx * 5 + float_offset_int]
                
                # Render each obstacle type (each branch returns the raster, so shapes match)
                def render_geese(r):
                    sprite = self.GEESE_MASKS[anim_idx]
                    return render_obstacle_type(sprite, False)
                
                def render_fish(r):
                    return render_obstacle_type(fish_mask, True)
                
                def render_crab(r):
                    return render_obstacle_type(crab_mask, True)
                
                def render_clam(r):
                    return render_obstacle_type(clam_mask, True)
                
                # Select and render the correct obstacle type
                r_out = jax.lax.cond(
                    obstacle_type == self.consts.ID_SNOW_GOOSE,
                    render_geese,
                    lambda r: jax.lax.cond(
                        obstacle_type == self.consts.ID_FISH,
                        render_fish,
                        lambda r: jax.lax.cond(
                            obstacle_type == self.consts.ID_KING_CRAB,
                            render_crab,
                            render_clam,
                            r
                        ),
                        r
                    ),
                    r_in
                )
                return r_out
            return jax.lax.cond(if_render, draw_lane, lambda r: r, r)
        raster = jax.lax.fori_loop(0, 4, render_lane, raster)
        return raster
    
    @partial(jax.jit, static_argnums=(0,))
    def _render_black_bar(self, raster):
        """Renders the black gutter bar on the left to hide obstacle spawns."""
        return self.jr.render_at(raster, 0, 0, self.BLACK_BAR_MASK)
    
    @partial(jax.jit, static_argnums=(0,))
    def _render_polar_grizzly(self, raster, state):
        """Render the polar grizzly (bear) when active."""
        should_render = state.polar_grizzly_active == 1
        def draw_bear(r):
            is_night = ((state.level - 1) // 4) % 2 == 1
            
            bear_stack = jax.lax.select(is_night, self.BEAR_LIGHT_MASKS, self.BEAR_MASKS)
            
            anim_idx = jnp.clip(state.polar_grizzly_animation_idx, 0, 7)
            animation_frame_idx = jnp.array(self.consts.POLAR_GRIZZLY_ANIM_MAP)[anim_idx]
            bear_sprite = bear_stack[animation_frame_idx]
            bear_y_offset = jnp.where(animation_frame_idx == 1, 1, 0)
            bear_y = self.consts.YMIN_BAILEY + bear_y_offset
            
            bear_sprite_flipped = jax.lax.cond(
                state.polar_grizzly_direction == 0,
                lambda s: jnp.flip(s, axis=1),
                lambda s: s,
                bear_sprite
            )
            
            return self.jr.render_at(r, state.polar_grizzly_x, bear_y, bear_sprite_flipped)
        return jax.lax.cond(should_render, draw_bear, lambda r: r, raster)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: FrostbiteState) -> jnp.ndarray:
        """Render the complete game frame from the current state."""
        
        # 1. Render background (day/night)
        raster = self.jr.create_object_raster(self.BACKGROUND)
        is_night = ((state.level - 1) // 4) % 2 == 1
        raster = jax.lax.cond(
            is_night,
            lambda r: self.jr.render_at(r, 0, 0, self.SHAPE_MASKS['background_night']),
            lambda r: r,
            raster
        )
        # 2. Render ice blocks
        ice_masks = self.ICE_MASKS # (4, H, W) [wide_w, wide_b, narrow_w, narrow_b]
        def render_row(row_idx, raster_in):
            y_pos = self.ICE_ROW_Y_ARRAY[row_idx]
            is_blue = state.ice_colors[row_idx] == self.consts.COLOR_ICE_BLUE
            block_positions = state.ice_block_positions[row_idx]
            block_count = state.ice_block_counts[row_idx]
            is_narrow = (block_count == 6)
            
            breathing_active = (
                (state.level >= self.consts.ICE_BREATH_MIN_LEVEL) &
                ((state.level & 1) == 1) &
                ~is_narrow
            )
            segment_positions, segment_widths, segment_mask = _compute_row_segments(
                self.consts,
                block_positions,
                block_count,
                state.ice_fine_motion_index,
                breathing_active,
                state.ice_x[row_idx]
            )
            
            # Select base masks: 0=wide_w, 1=wide_b
            base_idx = jnp.int32(is_blue) 
            # Select narrow masks: 2=narrow_w, 3=narrow_b
            narrow_idx_offset = jnp.int32(is_narrow) * 2 
            
            def render_one_segment(i, raster_seg):
                active = segment_mask[i]
                width = segment_widths[i] # 12 or 24
                
                # Must use lax.select
                sprite_mask = jax.lax.select(width == 12, 
                                             ice_masks[2 + base_idx], # narrow_mask
                                             ice_masks[0 + base_idx]) # wide_mask
                
                return jax.lax.cond(
                    active,
                    lambda r: self._render_with_wrap(r, segment_positions[i], y_pos, sprite_mask),
                    lambda r: r,
                    raster_seg
                )
            final_raster = jax.lax.fori_loop(0, segment_positions.shape[0], render_one_segment, raster_in)
            return final_raster
        raster = jax.lax.fori_loop(0, 4, render_row, raster)
        # 3. Render game entities in proper z-order
        raster = self._render_hud(raster, state)
        raster = self._render_igloo_blocks(raster, state)
        raster = self._render_obstacles(raster, state)
        raster = self._render_polar_grizzly(raster, state)
        # 4. Render Bailey
        is_dead = (state.bailey_alive == 0) | (state.bailey_death_frame > 0)
        is_visible = state.bailey_visible == 1
        
        # Select correct animation frame
        animation_idx = jnp.where(
            is_dead, 3 + ((state.bailey_death_frame // 30) % 2), state.bailey_animation_idx
        )
        
        bailey_stack = jax.lax.select(
            state.bailey_frozen == 1,
            self.BAILEY_FROZEN_MASKS,
            self.BAILEY_MASKS
        )
        
        # Get the final mask
        frame_bailey = bailey_stack[animation_idx] # Dynamic slice
        
        # Flip sprite horizontally when facing left
        flip = (state.bailey_direction == 1) & (animation_idx < 3)
        frame_bailey_flipped = jax.lax.cond(
            flip,
            lambda f: jnp.flip(f, axis=1), lambda f: f, frame_bailey
        )
        
        # Adjust Y for walking frame
        y_offset = jnp.where(animation_idx == 1, -1, 0)
        adjusted_y = state.bailey_y + y_offset
        # Render if visible
        raster = jax.lax.cond(
            is_visible,
            lambda r: self.jr.render_at(r, state.bailey_x, adjusted_y, frame_bailey_flipped),
            lambda r: r, 
            raster
        )
        
        # 5. Render the black bar over the gutter to hide spawns
        raster = self._render_black_bar(raster)
        
        # 6. Final palette lookup
        return self.jr.render_from_palette(raster, self.PALETTE)