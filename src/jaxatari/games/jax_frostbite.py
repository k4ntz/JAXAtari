import os
from functools import partial
from typing import Tuple, NamedTuple, Optional
import chex
import jax
from jax import tree_util
import jax.numpy as jnp
import jax.random as jrandom
from flax import struct

from jaxatari.environment import JaxEnvironment, ObjectObservation, JAXAtariAction as Action
import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
# Import the new rendering utils
import jaxatari.rendering.jax_rendering_utils as render_utils


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

class FrostbiteConstants(struct.PyTreeNode):
    """Complete constants"""
    
    # Screen dimensions
    SCREEN_WIDTH: int = struct.field(pytree_node=False, default=160)
    SCREEN_HEIGHT: int = struct.field(pytree_node=False, default=210)
    XMIN: int = struct.field(pytree_node=False, default=0)
    XMAX: int = struct.field(pytree_node=False, default=160)

    # Playfield boundaries (screen is 0..160; playfield is 8..160)
    PLAYFIELD_LEFT: int = struct.field(pytree_node=False, default=8)
    PLAYFIELD_RIGHT: int = struct.field(pytree_node=False, default=160)
    PLAYFIELD_WIDTH: int = struct.field(pytree_node=False, default=152)  # PLAYFIELD_RIGHT - PLAYFIELD_LEFT
    
    # Bailey Y position boundaries
    YMIN_BAILEY: int = struct.field(pytree_node=False, default_factory=lambda: 27 + 28 + 1)   # Shore Y position
    YMAX_BAILEY: int = struct.field(pytree_node=False, default_factory=lambda: 112 + 28)  # Arctic Sea Y position
    
    # Ice row Y positions (for rendering the ice blocks)
    ICE_ROW_Y: tuple = struct.field(pytree_node=False, default_factory=lambda: (98, 122, 147, 172))
    
    # Bailey X boundaries
    SHORE_X_MIN: int = struct.field(pytree_node=False, default=8)    # Allow Bailey to go all the way to the left edge for safe zone
    SHORE_X_MAX: int = struct.field(pytree_node=False, default=150)  # XMAX - 10
    ICE_X_MIN: int = struct.field(pytree_node=False, default=8)     # On ice blocks (XMIN + 8)
    ICE_X_MAX: int = struct.field(pytree_node=False, default=150)    # XMAX - 10
    
    # Initial Values
    INIT_BAILEY_HORIZ_POS: int = struct.field(pytree_node=False, default=64)
    INIT_POLAR_GRIZZLY_HORIZ_POS: int = struct.field(pytree_node=False, default=140)
    INIT_IGLOO_STATUS: int = struct.field(pytree_node=False, default=0)
    
    # Debug/Testing
    START_LEVEL: int = struct.field(pytree_node=False, default=1)
    
    # Bailey jump offset tables (Y deltas per ALE frame; consumed 2 entries per step for 60 Hz parity)
    BAILEY_JUMP_OFFSETS: tuple = struct.field(pytree_node=False, default_factory=lambda: (
        6, 5, 5, 5, 4, 3, 2, 1, 0, 0, 0, 0, -1, -2, -3,
        0,
        2, 2, 1, 0, 0, 0, 0, 0, -1, -2, -3, -4, -5, -6, -9,
        0
    ))


    # Speed reference values
    BAILEY_WALK_SPEED_FRAC: int = struct.field(pytree_node=False, default=8)
    
    # Colors
    COLOR_ICE_WHITE: int = struct.field(pytree_node=False, default=0x0E)
    COLOR_ICE_BLUE: int = struct.field(pytree_node=False, default=0x98)

    # RGB Overrides for mods (if set, overrides the actual rendered color of the ice blocks)
    RGB_ICE_WHITE: Optional[Tuple[int, int, int]] = struct.field(pytree_node=False, default=None)
    RGB_ICE_BLUE: Optional[Tuple[int, int, int]] = struct.field(pytree_node=False, default=None)

    # Igloo constants    IGLOO_X: int = struct.field(pytree_node=False, default=154)  # X position of igloo (far right side of screen)
    IGLOO_X: int = struct.field(pytree_node=False, default=154)
    IGLOO_Y: int = struct.field(pytree_node=False, default=44)   # Y position at top of Bailey's head when on shore
    
    # Game Constants
    MAX_IGLOO_INDEX: int = struct.field(pytree_node=False, default=15)  # Complete igloo has 16 blocks (0-15)
    MAX_EATEN_FISH: int = struct.field(pytree_node=False, default=12)  # Max fish that can be eaten per level
    MAX_RESERVED_LIVES: int = struct.field(pytree_node=False, default=9)  # Maximum reserve lives
    
    # Status Masks
    OBSTACLE_DIR_MASK: int = struct.field(pytree_node=False, default=0x80)
    ICE_BLOCK_DIR_MASK: int = struct.field(pytree_node=False, default=0x40)
    OBSTACLE_TYPE_MASK: int = struct.field(pytree_node=False, default=0x03)
    DEMO_MODE: int = struct.field(pytree_node=False, default=0x80)
    
    # Level Status Flags
    BAILEY_SINKING: int = struct.field(pytree_node=False, default=0x80)
    LEVEL_COMPLETE: int = struct.field(pytree_node=False, default=0x40)
    SWAP_PLAYERS: int = struct.field(pytree_node=False, default=0x20)
    INCREMENT_LEVEL: int = struct.field(pytree_node=False, default=0x08)
    
    # Special Levels
    MAGIC_FISH_LEVEL: int = struct.field(pytree_node=False, default=20)
    POLAR_GRIZZLY_LEVEL: int = struct.field(pytree_node=False, default=3)
    
    # Frame delays (adjusted for 60fps)
    # Phase 1: 4 seconds for igloo blocks = 240 frames
    # Phase 2: Temperature countdown after blocks

    # Collision helpers
    BAILEY_BOUNDING_WIDTH: int = struct.field(pytree_node=False, default=8)
    ICE_WIDE_LEFT_MARGIN: int = struct.field(pytree_node=False, default=-2)
    ICE_WIDE_RIGHT_MARGIN: int = struct.field(pytree_node=False, default=-4)
    ICE_NARROW_LEFT_MARGIN: int = struct.field(pytree_node=False, default=-2)
    ICE_NARROW_RIGHT_MARGIN: int = struct.field(pytree_node=False, default=0)
    ICE_MIN_OVERLAP: int = struct.field(pytree_node=False, default=1)
    ICE_WRAP_OFFSETS: tuple = struct.field(pytree_node=False, default_factory=lambda: (-152, 0, 152))  # Use playfield width for wrapping
    ICE_UNUSED_POS: int = struct.field(pytree_node=False, default=-512)
    ICE_NARROW_SPACING: int = struct.field(pytree_node=False, default=16)
    ICE_WIDE_SPACING: int = struct.field(pytree_node=False, default=32)
    ICE_BREATH_MIN_LEVEL: int = struct.field(pytree_node=False, default=5)
    ICE_BREATH_BLEND_STEPS: tuple = struct.field(pytree_node=False, default_factory=lambda: (
        0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1
    ))
    REMOVED_FISH_MASKS: tuple = struct.field(pytree_node=False, default_factory=lambda: (3, 5, 6, 3, 3, 1, 5, 4, 6, 6, 4))
    
    # Temperature and lives
    INIT_TEMPERATURE: int = struct.field(pytree_node=False, default=0x45)  # Initial temperature 45°
    INIT_LIVES: int = struct.field(pytree_node=False, default=3)  # Starting reserve lives
    
    # Obstacle constants
    ID_SNOW_GOOSE: int = struct.field(pytree_node=False, default=0)
    ID_FISH: int = struct.field(pytree_node=False, default=1)  # Fish obstacle type
    ID_KING_CRAB: int = struct.field(pytree_node=False, default=2)  # King crab obstacle type
    ID_KILLER_CLAM: int = struct.field(pytree_node=False, default=3)  # Killer clam obstacle type
    
    # Obstacle Y positions (between ice rows)
    OBSTACLE_Y: tuple = struct.field(pytree_node=False, default_factory=lambda: (82, 107, 132, 157))
    
    # Spawn and despawn boundaries
    OBSTACLE_SPAWN_LEFT: int = struct.field(pytree_node=False, default=-40)  # Spawn outside left edge moving right
    OBSTACLE_SPAWN_RIGHT: int = struct.field(pytree_node=False, default=168) # Spawn outside right edge moving left
    OBSTACLE_DESPAWN_X: int = struct.field(pytree_node=False, default=40)    # Despawn when 40 pixels off either side
    
    # Stutter movement logic for Level 5+
    OBSTACLE_STUTTER_LEVEL: int = struct.field(pytree_node=False, default=5)
    OBSTACLE_STUTTER_MASK: int = struct.field(pytree_node=False, default=0x20)  # Pause when (frame_count & 0x20) != 0

    # Floating obstacle animation
    FLOATING_OBSTACLE_OFFSETS: tuple = struct.field(pytree_node=False, default_factory=lambda: (4, 3, 2, 1, 0, 0, 0, 1, 2, 3, 4, 3, 2, 1, 0, 0))
    FLOATING_OBSTACLE_MAX_OFFSET: int = struct.field(pytree_node=False, default=4)
    FLOATING_OBSTACLE_PHASE_MASK: int = struct.field(pytree_node=False, default=0x1F)  # 5-bit sawtooth mask after shifting frame count
    FLOATING_OBSTACLE_PHASE_SHIFT: int = struct.field(pytree_node=False, default=2)    # Divide frame count by 4 (>> 2)
    OBSTACLE_ANIMATION_MASK_DEFAULT: int = struct.field(pytree_node=False, default=0x10)  # Flip every 16 frames
    OBSTACLE_ANIMATION_MASK_CLAM: int = struct.field(pytree_node=False, default=0x20)     # Clams animate slower (every 32 frames)

    # Polar grizzly animation map - defines which sprite frame (0 or 1) to use for each animation step
    POLAR_GRIZZLY_ANIM_MAP: tuple = struct.field(pytree_node=False, default_factory=lambda: (0, 0, 1, 1, 0, 0, 1, 1))
    
    # Frame delays for level progression
    INIT_DELAY_ACTION_VALUE: int = struct.field(pytree_node=False, default=120)  # 2 seconds for block removal
    INCREMENT_SCORE_FRAME_DELAY: int = struct.field(pytree_node=False, default=8)   # 120/16 ≈ 8 frames per block
    TEMP_DECREMENT_DELAY: int = struct.field(pytree_node=False, default=1)  # 1 frame per temperature degree

    # Sprite duplication modes
    SPRITE_SINGLE: int = struct.field(pytree_node=False, default=0b000)
    SPRITE_DOUBLE: int = struct.field(pytree_node=False, default=0b001)
    SPRITE_DOUBLE_SPACED: int = struct.field(pytree_node=False, default=0b010)
    SPRITE_TRIPLE: int = struct.field(pytree_node=False, default=0b011)
    SPRITE_DOUBLE_WIDE: int = struct.field(pytree_node=False, default=0b100)
    SPRITE_SIZE_2X: int = struct.field(pytree_node=False, default=0b101)     # not used for geese here
    SPRITE_TRIPLE_SPACED: int = struct.field(pytree_node=False, default=0b110)
    SPRITE_SIZE_4X: int = struct.field(pytree_node=False, default=0b111)       # not used for geese here

    # Sprite spacing distances in pixels
    SPACING_NARROW: int = struct.field(pytree_node=False, default=16)
    SPACING_MEDIUM: int = struct.field(pytree_node=False, default=32)
    SPACING_WIDE: int = struct.field(pytree_node=False, default=32)
    # Asset config baked into constants (immutable default) for asset overrides
    ASSET_CONFIG: tuple = struct.field(pytree_node=False, default_factory=_get_default_asset_config)


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


@struct.dataclass
class FrostbiteState:
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

    @property
    def lives(self):
        """Alias used by wrappers (e.g., episodic_life) to track life loss."""
        return self.remaining_lives
    
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

    # Precomputed render/collision segments for ice rows (shape: 4x6)
    ice_segments_x: chex.Array
    ice_segments_w: chex.Array
    
    # JAX
    rng_key: chex.PRNGKey


# ==========================================================================================
# OBSERVATION AND INFO
# ==========================================================================================
@struct.dataclass
class FrostbiteObservation:
    bailey: ObjectObservation
    obstacles: ObjectObservation
    bear: ObjectObservation
    ice_grid: jnp.ndarray
    igloo_progress: jnp.ndarray
    temperature: jnp.ndarray
    score: jnp.ndarray
    lives: jnp.ndarray
    level: jnp.ndarray


@struct.dataclass
class FrostbiteInfo:
    """Simple info"""
    level: jnp.ndarray


# ==========================================================================================
# MAIN GAME CLASS
# ==========================================================================================

class JaxFrostbite(JaxEnvironment[FrostbiteState, FrostbiteObservation, FrostbiteInfo, FrostbiteConstants]):
    """Bailey-only Frostbite implementation"""
    
    # Minimal ALE action set for Frostbite
    ACTION_SET: jnp.ndarray = jnp.array(
        [
            Action.NOOP,
            Action.FIRE,
            Action.UP,
            Action.RIGHT,
            Action.LEFT,
            Action.DOWN,
            Action.UPRIGHT,
            Action.UPLEFT,
            Action.DOWNRIGHT,
            Action.DOWNLEFT,
            Action.UPFIRE,
            Action.RIGHTFIRE,
            Action.LEFTFIRE,
            Action.DOWNFIRE,
            Action.UPRIGHTFIRE,
            Action.UPLEFTFIRE,
            Action.DOWNRIGHTFIRE,
            Action.DOWNLEFTFIRE,
        ],
        dtype=jnp.int32,
    )
    
    def __init__(self, consts: FrostbiteConstants = None):
        if consts is None:
            consts = FrostbiteConstants()
        super().__init__(consts)

        self.renderer = FrostbiteRenderer(self.consts)
    
    def _get_point_value_for_level(self, level: jnp.ndarray):
        """Get point value for level × 10 in BCD format"""
        points = level * 10
        points = jnp.minimum(points, 90)
        hundreds = points // 100
        tens = (points % 100) // 10
        ones = points % 10
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

        # Ice speed: doubled for single-substep 60 Hz parity
        ice_speed_raw = S
        ice_speed_whole = ice_speed_raw // 16
        ice_speed_frac = ice_speed_raw % 16

        # Obstacle speed: doubled for single-substep 60 Hz parity
        obstacle_speed_raw = jnp.maximum(2 * (S - 1), 0)
        obstacle_speed_whole = obstacle_speed_raw // 16
        obstacle_speed_frac = obstacle_speed_raw % 16

        # Bailey jump speed: doubled for single-substep 60 Hz parity
        bailey_jump_speed_raw = 2 * S
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
        segment_positions = state.ice_segments_x[row_idx]
        segment_widths = state.ice_segments_w[row_idx]
        segment_mask = segment_widths > 0
        return segment_positions, segment_widths, segment_mask

    @partial(jax.jit, static_argnums=(0,))
    def _compute_ice_segments(self, state: FrostbiteState) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Precompute per-row ice segment positions and widths for render/collision."""
        def row_segments(row_idx):
            block_positions = state.ice_block_positions[row_idx]
            block_count = state.ice_block_counts[row_idx]
            breathing_active = (
                (state.level >= self.consts.ICE_BREATH_MIN_LEVEL) &
                ((state.level & 1) == 1) &
                (block_count <= 3)
            )
            pos, widths, _ = _compute_row_segments(
                self.consts,
                block_positions,
                block_count,
                state.ice_fine_motion_index,
                breathing_active,
                state.ice_x[row_idx],
            )
            return pos, widths

        return jax.vmap(row_segments)(jnp.arange(4, dtype=jnp.int32))

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
            ice_segments_x=jnp.full((4, 6), self.consts.ICE_UNUSED_POS, dtype=jnp.int32),
            ice_segments_w=jnp.zeros((4, 6), dtype=jnp.int32),
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
            ice_segments_x=jnp.full((4, 6), self.consts.ICE_UNUSED_POS, dtype=jnp.int32),
            ice_segments_w=jnp.zeros((4, 6), dtype=jnp.int32),
            rng_key=key
        )
        seg_x, seg_w = self._compute_ice_segments(state)
        state = state.replace(ice_segments_x=seg_x, ice_segments_w=seg_w)
        
        obs = self._get_observation(state)
        return obs, state
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: FrostbiteState) -> FrostbiteObservation:
        # --- Bailey ---
        # State: 0=Walking, 1=Jumping
        bailey_status = (state.bailey_jumping_idx > 0).astype(jnp.int32)
        # Orientation: 0(Right)->90, 1(Left)->270
        bailey_ori = jnp.where(state.bailey_direction == 0, 90.0, 270.0)
        
        bailey = ObjectObservation.create(
            x=jnp.clip(state.bailey_x, 0, self.consts.SCREEN_WIDTH),
            y=jnp.clip(state.bailey_y, 0, self.consts.SCREEN_HEIGHT),
            width=jnp.array(self.consts.BAILEY_BOUNDING_WIDTH, dtype=jnp.int32),
            height=jnp.array(18, dtype=jnp.int32), # Approx height
            orientation=bailey_ori.astype(jnp.float32),
            state=bailey_status,
            active=state.bailey_visible.astype(jnp.int32)
        )

        # --- Obstacles ---
        # Vectorize over 4 rows without vmap
        x_base = state.obstacle_x[:, None] # (4, 1)
        y = state.obstacle_y[:, None] # (4, 1)
        active = state.obstacle_active[:, None] # (4, 1)
        dir_ = state.obstacle_directions[:, None] # (4, 1)
        mode = state.obstacle_duplication_mode # (4,)
        
        # _decode_sprite_duplication handles vectors automatically via jnp.where
        copies, spacing = self._decode_sprite_duplication(mode)
        copies = copies[:, None] # (4, 1)
        spacing = spacing[:, None] # (4, 1)
        
        slots = jnp.arange(3)[None, :] # (1, 3)
        xs = x_base + slots * spacing # (4, 3)
        valid = (slots < copies) & (active == 1) # (4, 3)
        
        # Filter out off-screen copies (simple bounding box check)
        on_screen = (xs > -8) & (xs < self.consts.SCREEN_WIDTH)
        
        final_active = valid & on_screen
        final_ori = jnp.where(dir_ == 0, 90.0, 270.0) # (4, 1)
        
        ys = jnp.broadcast_to(y, (4, 3))
        oris = jnp.broadcast_to(final_ori, (4, 3))
        actives = final_active.astype(jnp.int32)
        
        # Flatten (4, 3) -> (12,)
        obstacles = ObjectObservation.create(
            x=jnp.clip(xs.flatten().astype(jnp.int32), 0, self.consts.SCREEN_WIDTH),
            y=jnp.clip(ys.flatten().astype(jnp.int32), 0, self.consts.SCREEN_HEIGHT),
            width=jnp.full((12,), 8, dtype=jnp.int32),
            height=jnp.full((12,), 8, dtype=jnp.int32),
            orientation=oris.flatten().astype(jnp.float32),
            active=actives.flatten().astype(jnp.int32)
        )

        # --- Polar Bear ---
        bear_ori = jnp.where(state.polar_grizzly_direction == 0, 90.0, 270.0)
        bear = ObjectObservation.create(
            x=jnp.clip(state.polar_grizzly_x, 0, self.consts.SCREEN_WIDTH),
            y=jnp.clip(jnp.array(self.consts.YMIN_BAILEY, dtype=jnp.int32), 0, self.consts.SCREEN_HEIGHT),
            width=jnp.array(20, dtype=jnp.int32),
            height=jnp.array(14, dtype=jnp.int32),
            orientation=bear_ori.astype(jnp.float32),
            active=state.polar_grizzly_active.astype(jnp.int32)
        )

        # --- Ice Grid (Procedural Generation) ---
        # Generate a grid representation of where valid ice exists
        # We sample the "active block" logic at regular intervals
        grid_width = 16 # Discretize screen width into 16 chunks
        sample_xs = jnp.linspace(self.consts.PLAYFIELD_LEFT, self.consts.PLAYFIELD_RIGHT, grid_width).astype(jnp.int32)
        
        pos = state.ice_segments_x # (4, 6)
        widths = state.ice_segments_w # (4, 6)
        mask = widths > 0 # (4, 6)
        
        px = sample_xs.reshape(1, 1, 16)
        seg_x = pos.reshape(4, 6, 1)
        seg_w = widths.reshape(4, 6, 1)
        active = mask.reshape(4, 6, 1)
        
        hits = active & (px >= seg_x) & (px < seg_x + seg_w) # Shape: (4, 6, 16)
        ice_grid = jnp.any(hits, axis=1).astype(jnp.int32) # Shape: (4, 16)

        score_val = self._bcd_to_decimal(state.score)
        temp_val = self._bcd_to_decimal(jnp.array([0, 0, state.temperature], dtype=jnp.int32))

        return FrostbiteObservation(
            bailey=bailey,
            obstacles=obstacles,
            bear=bear,
            ice_grid=ice_grid,
            igloo_progress=state.building_igloo_idx + 1, # -1..15 -> 0..16
            temperature=temp_val.astype(jnp.int32),
            score=score_val.astype(jnp.int32),
            lives=state.remaining_lives.astype(jnp.int32),
            level=state.level.astype(jnp.int32)
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

        state = state.replace(
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

        def do_spawn(_):
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

            return state.replace(
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

        return jax.lax.cond(
            jnp.any(spawn_mask),
            do_spawn,
            lambda _: state,
            None
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: FrostbiteState, state: FrostbiteState) -> chex.Array:
        """Calculate reward based on score difference."""
        prev_score_val = self._bcd_to_decimal(previous_state.score)
        curr_score_val = self._bcd_to_decimal(state.score)
        return (curr_score_val - prev_score_val).astype(jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: FrostbiteState) -> chex.Array:
        """Check if game is over."""
        frozen_death_complete = (state.temperature == 0) & (state.bailey_death_frame >= 60)
        final_death_complete = (state.bailey_death_frame >= 60) & (state.remaining_lives == 0)
        return frozen_death_complete | final_death_complete

    def _bcd_to_decimal(self, bcd_score: jnp.ndarray) -> jnp.ndarray:
        """Convert 3-byte BCD score to decimal value."""
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
        state = state.replace(frame_count=state.frame_count + 1)

        # Ice wobble animation for high odd-numbered levels
        # Creates a visual "breathing" effect on ice blocks
        should_drift = (
            (state.level >= self.consts.ICE_BREATH_MIN_LEVEL) &  # Level 7+
            (state.level % 2 == 1) &  # Odd levels only
            ((state.frame_count & 3) == 0)  # Update every 4 frames
        )
        new_fine_motion_index = jnp.where(
            should_drift,
            (state.ice_fine_motion_index - 1) % 16,  # Cycle through 16 animation frames
            state.ice_fine_motion_index
        )
        state = state.replace(ice_fine_motion_index=new_fine_motion_index)

        # Core game update sequence
        state = self._process_level_complete(state)  # Handle level completion animations
        state = self._update_ice_blocks(state, action)  # Move ice and handle player landing

        # Update all obstacles (fish, crabs, clams, birds)
        state = self._update_obstacles(state)

        state = self._check_obstacle_collisions(state)  # Check Bailey-obstacle collisions
        state = self._update_bailey(state, action)  # Update player position and state
        state = self._update_polar_grizzly(state)  # Update bear behavior

        def decrement_bcd(temp):
            ones = temp & 0x0F
            tens = (temp >> 4) & 0x0F
            new_ones = jax.lax.cond(ones == 0, lambda: 9, lambda: ones - 1)
            new_tens = jax.lax.cond(ones == 0, lambda: jnp.maximum(0, tens - 1), lambda: tens)
            result = (new_tens << 4) | new_ones
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
        state = state.replace(temperature=new_temperature)

        # Precompute ice segments for rendering and collision
        seg_x, seg_w = self._compute_ice_segments(state)
        state = state.replace(ice_segments_x=seg_x, ice_segments_w=seg_w)

        # Check for collisions and special conditions
        state = self._check_collisions(state)  # Handle Bailey-ice collisions
        state = self._check_igloo_entry(state, action)  # Check if Bailey enters igloo

        # Calculate reward and done using dedicated methods
        reward = self._get_reward(prev_state, state)
        done = self._get_done(state)

        return state, reward, done

    @partial(jax.jit, static_argnums=(0,))
    def _check_jump_intent(self, state: FrostbiteState, action: jnp.ndarray, moving_up: jnp.ndarray, moving_down: jnp.ndarray):
        """Check if the user intends to jump. Continuous jumping is allowed."""
        return moving_up, moving_down

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: FrostbiteState, action: int):
        atari_action = jnp.take(self.ACTION_SET, jnp.asarray(action, dtype=jnp.int32))

        state_out, reward, done = self._step_once(state, atari_action)

        obs_last = self._get_observation(state_out)
        info_last = self._get_info(state_out)

        return obs_last, state_out, reward, done, info_last

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
        new_delay = jnp.where(blocks_just_finished, 45, new_delay)  # Reset timer for temp phase

        # Phase 2: Decrement temperature to zero
        in_temp_phase = is_level_complete & (new_building_idx < 0) & (state.temperature > 0)
        temp_elapsed = 45 - new_delay  # Frames elapsed in temperature phase

        # Decrement temperature every TEMP_DECREMENT_DELAY frames
        should_decrement_temp = (
            in_temp_phase & (new_delay < 90) & (temp_elapsed > 0) &
            ((temp_elapsed % self.consts.TEMP_DECREMENT_DELAY) == 0)
        )

        # Update temperature during countdown
        def decrement_temp_bcd(temp):
            ones = temp & 0x0F
            tens = (temp >> 4) & 0x0F
            new_ones = jnp.where(ones > 0, ones - 1, 9)
            new_tens = jnp.where(ones == 0, jnp.maximum(0, tens - 1), tens)
            result = (new_tens << 4) | new_ones
            return jnp.where((tens == 0) & (ones == 0), 0, result)
        new_temperature = jnp.where(should_decrement_temp, decrement_temp_bcd(state.temperature), state.temperature)

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

        # Reset bear to initial position on level completion
        new_polar_grizzly_x = jnp.where(
            level_reset_complete,
            self.consts.INIT_POLAR_GRIZZLY_HORIZ_POS,
            state.polar_grizzly_x
        )
        new_polar_grizzly_frac = jnp.where(
            level_reset_complete,
            0,
            state.polar_grizzly_frac_accumulator
        )
        new_polar_grizzly_direction = jnp.where(
            level_reset_complete,
            1,
            state.polar_grizzly_direction
        )
        new_polar_grizzly_animation_idx = jnp.where(
            level_reset_complete,
            7,
            state.polar_grizzly_animation_idx
        )
        new_bailey_grizzly_collision_timer = jnp.where(
            level_reset_complete,
            0,
            state.bailey_grizzly_collision_timer
        )
        new_bailey_grizzly_collision_value = jnp.where(
            level_reset_complete,
            0,
            state.bailey_grizzly_collision_value
        )

        next_state = state.replace(
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
            fish_alive_mask=new_fish_alive_mask,
            polar_grizzly_x=new_polar_grizzly_x,
            polar_grizzly_frac_accumulator=new_polar_grizzly_frac,
            polar_grizzly_direction=new_polar_grizzly_direction,
            polar_grizzly_animation_idx=new_polar_grizzly_animation_idx,
            bailey_grizzly_collision_timer=new_bailey_grizzly_collision_timer,
            bailey_grizzly_collision_value=new_bailey_grizzly_collision_value
        )
        seg_x, seg_w = self._compute_ice_segments(next_state)
        next_state = next_state.replace(ice_segments_x=seg_x, ice_segments_w=seg_w)

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

        # Determine jump intent using the refactored method
        intent_jump_up, intent_jump_down = self._check_jump_intent(state, action, moving_up, moving_down)

        # Determine if Bailey can initiate a jump
        # Up jump: allowed from ice (not shore) or for igloo entry
        # Bear chase disables all jumping
        can_jump_up = ((intent_jump_up | (should_jump_for_igloo & (state.bailey_y > 6))) &
                       can_start_jump & ((state.bailey_y > self.consts.YMIN_BAILEY) | should_jump_for_igloo)) & ~being_chased_by_bear
        can_jump_down = intent_jump_down & can_start_jump & \
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
        walk_frame = jnp.where(is_moving, (state.frame_count >> 2) & 1, 0)  # Toggle every 4 frames

        # Select final animation frame
        new_animation = jnp.where(
            being_chased_by_bear,
            state.bailey_animation_idx,  # Preserve animation from bear update
            jnp.where(is_jumping_sprite, 2, walk_frame)  # Jump sprite (2) or walk frame (0/1)
        )
        
        return state.replace(
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

        # Load jump offset table - contains Y deltas per ALE frame.
        # Advance by 2 entries per step (2 ALE frames per JAX step for 60 Hz parity).
        offset_table = jnp.array(self.consts.BAILEY_JUMP_OFFSETS)

        idx_a_raw = jump_idx - 1
        idx_b_raw = jump_idx - 2
        offset_a = offset_table[jnp.clip(idx_a_raw, 0, 31)]
        # Only add second entry if it's a valid (non-negative) index
        offset_b = jnp.where(idx_b_raw >= 0, offset_table[jnp.clip(idx_b_raw, 0, 31)], 0)
        offset = offset_a + offset_b
        new_y = current_y + offset

        # Handle jump completion: index 16 is the transition / termination point
        raw_next_idx = jump_idx - 2
        next_idx = jnp.where((jump_idx - 1 == 16) | (jump_idx - 2 == 16), 0, raw_next_idx)
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
        should_move = jnp.bitwise_and(state.frame_count, 1) == 0  # Move every other frame
        speed_whole = jnp.where(
            should_move,
            state.ice_speed_whole.astype(jnp.int32),
            jnp.zeros_like(state.ice_speed_whole, dtype=jnp.int32)
        )
        frac_increment = jnp.where(
            should_move,
            state.ice_speed_frac,
            jnp.zeros_like(state.ice_speed_frac)
        )
        frac_sum = state.ice_frac_accumulators + frac_increment
        extra_pixels = frac_sum // 16
        new_frac = jnp.where(should_move, frac_sum % 16, state.ice_frac_accumulators)
        move_pixels = speed_whole + extra_pixels
        direction = jnp.where(new_ice_directions == 0, 1, -1)
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
        next_state = state.replace(
            ice_x=new_ice_x,  # Leftmost position of each row
            ice_block_positions=updated_positions,  # All block positions
            ice_block_counts=new_block_counts,  # Blocks per row (3 or 6)
            ice_directions=new_ice_directions,  # Movement direction per row
            ice_frac_accumulators=new_frac,  # Fractional movement remainder
            ice_dx_last_frame=move_dx,  # Movement from this frame (for Bailey drift)
            building_igloo_idx=new_building_idx  # Updated after reversal cost
        )
        return next_state
    
    def _update_obstacles(self, state: FrostbiteState):
        """Update all obstacles (birds, fish, crabs, clams) with movement and spawning.

        This method handles:
        - Movement calculation with fractional accumulation
        - Level 5+ stuttering behavior (crabs/clams alternate freezing)
        - Despawn/respawn when off-screen
        - Animation frame cycling
        - Sprite duplication patterns

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
        state_after_move = state.replace(
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
        # Duplication mode is stable for the full screen traversal; derived
        # directly from the pattern index assigned at spawn time.
        table_idx = state.obstacle_pattern_index & 7

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
        
        return state.replace(
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
        bailey_row = jnp.where(jnp.min(y_distances) < 8, jnp.argmin(y_distances), -1)

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
        animation_in_progress = (state.bailey_death_frame > 0) & (state.bailey_death_frame < 60)

        # Start or continue death animation
        new_death_frame = jnp.where(
            just_died, 1,  # Start animation
            jnp.where(animation_in_progress, state.bailey_death_frame + 1, state.bailey_death_frame)
        )

        # Life management after death animation completes
        death_animation_complete = state.bailey_death_frame >= 60
        new_remaining_lives = jnp.where(
            death_animation_complete & (state.bailey_death_frame == 60),
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
        new_polar_grizzly_direction = jnp.where(
            should_respawn,
            1,
            state.polar_grizzly_direction
        )
        new_polar_grizzly_animation_idx = jnp.where(
            should_respawn,
            7,
            state.polar_grizzly_animation_idx
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
            all_blue & (state.completed_ice_blocks_delay == 0), 16,
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
        
        updated_state = state.replace(
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
            polar_grizzly_direction=new_polar_grizzly_direction,
            polar_grizzly_animation_idx=new_polar_grizzly_animation_idx,
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
        """Add points to BCD score (Binary Coded Decimal)."""
        current = self._bcd_to_decimal(score)
        hundreds = (points >> 8) & 0x0F
        tens = (points >> 4) & 0x0F
        ones = points & 0x0F
        decimal_points = hundreds * 100 + tens * 10 + ones
        new_total = current + decimal_points
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
            old_score: Previous integer score
            new_score: Updated integer score after points were added
            lives: Current number of reserve lives

        Returns:
            Updated number of lives, capped at 9 (hardcoded limit)
        """
        # Calculate how many 5000-point thresholds were crossed
        # Integer division gives us the number of extra lives earned so far
        old_total = self._bcd_to_decimal(old_score)
        new_total = self._bcd_to_decimal(new_score)
        old_lives_earned = old_total // 5000
        new_lives_earned = new_total // 5000

        # Award extra lives for all thresholds crossed, but enforce the 9-life maximum
        # Note: MAX_RESERVED_LIVES constant exists but isn't used - limit is hardcoded
        lives_to_add = new_lives_earned - old_lives_earned
        new_lives = jnp.where(lives_to_add > 0, jnp.minimum(lives + lives_to_add, 9), lives)

        return new_lives
    
    def _add_bcd_score_decimal(self, score, decimal_points):
        current = self._bcd_to_decimal(score)
        new_total = current + decimal_points
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

        return state.replace(
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
        bailey_chase_animation = state.frame_count & 3  # Bailey's dragged animation

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
            chase_anim_idx = ((chase_timer >> 2) & 7).astype(jnp.int32)  # 8-frame animation cycle (every 4 frames)

            return state.replace(
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

            return state.replace(
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

        return state.replace(
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
        return spaces.Discrete(len(self.ACTION_SET))
    
    def observation_space(self) -> spaces.Dict:
        return spaces.Dict({
            "bailey": spaces.get_object_space(n=None, screen_size=(self.consts.SCREEN_HEIGHT, self.consts.SCREEN_WIDTH)),
            # Obstacles: Max 4 rows * 3 copies = 12 potential objects
            "obstacles": spaces.get_object_space(n=12, screen_size=(self.consts.SCREEN_HEIGHT, self.consts.SCREEN_WIDTH)),
            "bear": spaces.get_object_space(n=None, screen_size=(self.consts.SCREEN_HEIGHT, self.consts.SCREEN_WIDTH)),
            # Ice Grid: 4 rows, discretized horizontally into ~10-12 pixel chunks (width 152 / 12 ~= 12 blocks)
            "ice_grid": spaces.Box(low=0, high=1, shape=(4, 16), dtype=jnp.int32),
            "igloo_progress": spaces.Box(low=0, high=16, shape=(), dtype=jnp.int32),
            "temperature": spaces.Box(low=0, high=99, shape=(), dtype=jnp.int32),
            "score": spaces.Box(low=0, high=999999, shape=(), dtype=jnp.int32),
            "lives": spaces.Box(low=0, high=9, shape=(), dtype=jnp.int32),
            "level": spaces.Box(low=1, high=99, shape=(), dtype=jnp.int32),
        })

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

    def __init__(self, consts: FrostbiteConstants = None, config: render_utils.RendererConfig = None):
        """Initialize the renderer with game constants and load all sprites.

        Args:
            consts: Game constants defining screen dimensions and positions
        """
        self.consts = consts or FrostbiteConstants()
        super().__init__(self.consts)
        
        # Use injected config if provided, else default
        if config is None:
            self.config = render_utils.RendererConfig(
                game_dimensions=(self.consts.SCREEN_HEIGHT, self.consts.SCREEN_WIDTH),
                channels=3,
                downscale=None
            )
        else:
            self.config = config
        self.jr = render_utils.JaxRenderingUtils(self.config)
        self.sprite_path = os.path.join(render_utils.get_base_sprite_dir(), "frostbite")
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
        
        digit_path_pattern = os.path.join(self.sprite_path, "digit_{}.npy")
        digits_array = self.jr.load_and_pad_digits(digit_path_pattern, num_chars=10)
        digits_list = [digits_array[i] for i in range(10)]
        
        # --- Pre-generate Variations ---
        
        # Bailey (Frozen)
        bailey_frames = [bailey_0, bailey_1, bailey_2, bailey_3, bailey_4]
        bailey_frozen_frames = [self._apply_frozen_tint(f) for f in bailey_frames]
        
        # Ice (Blue)
        ice_wide_blue = self._apply_ice_color(ice_wide_white, is_blue=True)
        ice_narrow_blue = self._apply_ice_color(ice_narrow_white, is_blue=True)

        # Apply custom RGB colors if set by mods
        if self.consts.RGB_ICE_WHITE is not None:
            r, g, b = self.consts.RGB_ICE_WHITE
            ice_wide_white = jnp.where(
                ice_wide_white[..., 3:4] > 0,
                jnp.concatenate([
                    jnp.full_like(ice_wide_white[..., 0:1], r),
                    jnp.full_like(ice_wide_white[..., 1:2], g),
                    jnp.full_like(ice_wide_white[..., 2:3], b),
                    ice_wide_white[..., 3:4]
                ], axis=-1),
                ice_wide_white
            ).astype(ice_wide_white.dtype)
            ice_narrow_white = jnp.where(
                ice_narrow_white[..., 3:4] > 0,
                jnp.concatenate([
                    jnp.full_like(ice_narrow_white[..., 0:1], r),
                    jnp.full_like(ice_narrow_white[..., 1:2], g),
                    jnp.full_like(ice_narrow_white[..., 2:3], b),
                    ice_narrow_white[..., 3:4]
                ], axis=-1),
                ice_narrow_white
            ).astype(ice_narrow_white.dtype)

        if self.consts.RGB_ICE_BLUE is not None:
            r, g, b = self.consts.RGB_ICE_BLUE
            ice_wide_blue = jnp.where(
                ice_wide_blue[..., 3:4] > 0,
                jnp.concatenate([
                    jnp.full_like(ice_wide_blue[..., 0:1], r),
                    jnp.full_like(ice_wide_blue[..., 1:2], g),
                    jnp.full_like(ice_wide_blue[..., 2:3], b),
                    ice_wide_blue[..., 3:4]
                ], axis=-1),
                ice_wide_blue
            ).astype(ice_wide_blue.dtype)
            ice_narrow_blue = jnp.where(
                ice_narrow_blue[..., 3:4] > 0,
                jnp.concatenate([
                    jnp.full_like(ice_narrow_blue[..., 0:1], r),
                    jnp.full_like(ice_narrow_blue[..., 1:2], g),
                    jnp.full_like(ice_narrow_blue[..., 2:3], b),
                    ice_narrow_blue[..., 3:4]
                ], axis=-1),
                ice_narrow_blue
            ).astype(ice_narrow_blue.dtype)
        
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

        obstacle_target_h = max(
            self.GEESE_MASKS.shape[1],
            self.FISH_MASKS.shape[1],
            self.CRAB_MASKS.shape[1],
            self.CLAM_MASKS.shape[1],
        )
        obstacle_target_w = max(
            self.GEESE_MASKS.shape[2],
            self.FISH_MASKS.shape[2],
            self.CRAB_MASKS.shape[2],
            self.CLAM_MASKS.shape[2],
        )
        self.GEESE_MASKS = self._pad_mask_stack_to_shape(
            self.GEESE_MASKS, obstacle_target_h, obstacle_target_w, self.jr.TRANSPARENT_ID
        )
        # Pad Geese to 10 frames to match fish/crab/clam stacks for vectorized selection
        self.GEESE_MASKS = jnp.pad(
            self.GEESE_MASKS,
            ((0, 8), (0, 0), (0, 0)),
            mode="constant",
            constant_values=self.jr.TRANSPARENT_ID,
        )
        self.FISH_MASKS = self._pad_mask_stack_to_shape(
            self.FISH_MASKS, obstacle_target_h, obstacle_target_w, self.jr.TRANSPARENT_ID
        )
        self.CRAB_MASKS = self._pad_mask_stack_to_shape(
            self.CRAB_MASKS, obstacle_target_h, obstacle_target_w, self.jr.TRANSPARENT_ID
        )
        self.CLAM_MASKS = self._pad_mask_stack_to_shape(
            self.CLAM_MASKS, obstacle_target_h, obstacle_target_w, self.jr.TRANSPARENT_ID
        )

        # Pad ice masks to uniform shape for batched segment rendering
        ice_target_h = jnp.max(jnp.array([m.shape[0] for m in self.ICE_MASKS]))
        ice_target_w = jnp.max(jnp.array([m.shape[1] for m in self.ICE_MASKS]))
        self.ICE_MASKS = self._pad_mask_stack_to_shape(
            self.ICE_MASKS, ice_target_h, ice_target_w, self.jr.TRANSPARENT_ID
        )

        self.BEAR_MASKS = self.SHAPE_MASKS['bear']
        self.BEAR_LIGHT_MASKS = self.SHAPE_MASKS['bear_light']
        self.IGLOO_BLOCK_MASK = self.SHAPE_MASKS['igloo'][0]
        self.IGLOO_DOOR_MASK = self.SHAPE_MASKS['igloo'][1]
        self.DEGREE_MASK = self.SHAPE_MASKS['degree']
        # Pad degree mask to digit shape for vectorized HUD rendering
        self.DEGREE_MASK = self._pad_mask_stack_to_shape(
            self.DEGREE_MASK[jnp.newaxis],
            self.DIGIT_MASKS.shape[1],
            self.DIGIT_MASKS.shape[2],
            self.jr.TRANSPARENT_ID
        )[0]
        self.BLACK_BAR_MASK = self.SHAPE_MASKS['black_bar']

        # Pre-calculate a single stack of all obstacle masks for vectorized rendering
        # Geese: 10 (padded). Fish: 10. Crab: 10. Clam: 10. Total 40.
        # Geese: [0..9]  (frames 0-1 real, 2-9 transparent padding)
        # Fish:  [10..19] (anim*5 + float_offset)
        # Crab:  [20..29]
        # Clam:  [30..39]
        self.ALL_OBSTACLE_MASKS = jnp.concatenate([
            self.GEESE_MASKS,
            self.FISH_MASKS,
            self.CRAB_MASKS,
            self.CLAM_MASKS,
        ], axis=0)
        
        # Convert ICE_ROW_Y tuple to JAX array for dynamic indexing
        self.ICE_ROW_Y_ARRAY = jnp.array(self.consts.ICE_ROW_Y, dtype=jnp.int32)

        # Pre-tile one 152px-wide palette-ID strip per ice variant so the render
        # loop stamps one strip per row (3 dynamic_update_slice calls) instead of
        # one sprite per segment (up to 18 calls).  Built once at init, zero cost
        # at render time.
        _STRIP_W = self.consts.PLAYFIELD_WIDTH   # 152
        _TRANSP  = self.jr.TRANSPARENT_ID

        def _build_strip(block_mask, n_blocks, spacing):
            H, BW = block_mask.shape
            strip = jnp.full((H, _STRIP_W), _TRANSP, dtype=block_mask.dtype)
            for i in range(n_blocks):
                off = i * spacing
                if off >= _STRIP_W:
                    break
                actual_end = min(off + BW, _STRIP_W)
                piece = block_mask[:, :actual_end - off]
                strip = strip.at[:, off:actual_end].set(
                    jnp.where(piece != _TRANSP, piece, strip[:, off:actual_end])
                )
            return strip

        _ws = self.consts.ICE_WIDE_SPACING    # 32
        _ns = self.consts.ICE_NARROW_SPACING  # 16
        self.ICE_STRIP_WIDE_WHITE   = _build_strip(self.ICE_MASKS[0], 3, _ws)
        self.ICE_STRIP_WIDE_BLUE    = _build_strip(self.ICE_MASKS[1], 3, _ws)
        self.ICE_STRIP_NARROW_WHITE = _build_strip(self.ICE_MASKS[2], 6, _ns)
        self.ICE_STRIP_NARROW_BLUE  = _build_strip(self.ICE_MASKS[3], 6, _ns)

        self.ICE_STRIPS = jnp.stack([
            self.ICE_STRIP_WIDE_WHITE,    # 0
            self.ICE_STRIP_WIDE_BLUE,     # 1
            self.ICE_STRIP_NARROW_WHITE,  # 2
            self.ICE_STRIP_NARROW_BLUE,   # 3
        ])  # (4, H, STRIP_W)

        # Pre-build 17 igloo canvas layers (indices 0..16, where index = building_igloo_idx+1).
        # Each canvas is a 24×32 palette-ID sprite covering the full igloo bounding box.
        # At render time we index once and stamp with a single render_at call instead of
        # running fori_loop(0,16) with multiple lax.cond calls per iteration.
        import numpy as _np

        _CH, _CW = 24, 32
        _cx0, _cy0 = 111, 35   # canvas origin in raster coords

        _bm = _np.array(self.IGLOO_BLOCK_MASK)   # (8,8) palette IDs
        _dm = _np.array(self.IGLOO_DOOR_MASK)    # (8,8) palette IDs
        _TRANSP_NP = int(self.jr.TRANSPARENT_ID)

        # Canvas-relative (x, y) for each block 0..15
        _BLK_CX = _np.array([0, 8, 16, 24, 24, 16, 8, 0, 0, 8, 16, 24, 4, 16, 8, 11], dtype=_np.int32)
        _BLK_CY = _np.array([16,16, 16,16, 12, 12,12,12, 8, 8,  8,  8, 4,  4, 0, 12], dtype=_np.int32)

        def _stamp_np(canvas, bx, by, mask):
            ey = min(by + mask.shape[0], _CH)
            ex = min(bx + mask.shape[1], _CW)
            if ey <= by or ex <= bx:
                return canvas
            src = mask[:ey - by, :ex - bx]
            dst = canvas[by:ey, bx:ex]
            canvas[by:ey, bx:ex] = _np.where(src != _TRANSP_NP, src, dst)
            return canvas

        _igloo_layers = []
        for _n in range(17):   # 0 = no blocks, n = first n blocks (0..n-1) visible
            _canvas = _np.full((_CH, _CW), _TRANSP_NP, dtype=_bm.dtype)
            for _b in range(min(_n, 16)):
                if _b == 15:
                    _canvas = _stamp_np(_canvas, int(_BLK_CX[15]), int(_BLK_CY[15]), _dm)
                else:
                    _canvas = _stamp_np(_canvas, int(_BLK_CX[_b]), int(_BLK_CY[_b]), _bm)
                    if _b in (12, 13):
                        _canvas = _stamp_np(_canvas, int(_BLK_CX[_b]) + 4, int(_BLK_CY[_b]), _bm)
                    if _b == 14:
                        _canvas = _stamp_np(_canvas, int(_BLK_CX[_b]) + 8, int(_BLK_CY[_b]), _bm)
            _igloo_layers.append(_canvas)

        self.IGLOO_LAYERS = jnp.stack([jnp.array(c) for c in _igloo_layers])  # (17, 24, 32)
        self._IGLOO_CX0 = _cx0
        self._IGLOO_CY0 = _cy0

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

    @staticmethod
    def _pad_mask_stack_to_shape(mask_stack, target_h, target_w, transparent_id):
        """Pad an (N,H,W) mask stack to a uniform (N,target_h,target_w)."""
        return jnp.pad(
            mask_stack,
            (
                (0, 0),
                (0, target_h - mask_stack.shape[1]),
                (0, target_w - mask_stack.shape[2]),
            ),
            mode="constant",
            constant_values=transparent_id,
        )

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
        """Render HUD elements at top of screen using vectorized batch rendering."""
        digits_masks = self.DIGIT_MASKS
        score_y, lives_y, temp_y = 10, 22, 22
        should_flash = state.temperature < 0x10
        is_visible = ~should_flash | ((state.frame_count % 45) < 22)
        
        # 1. Lives
        lives_clamped = jnp.clip(state.remaining_lives, 0, 9)
        lives_mask = digits_masks[lives_clamped]
        lives_x = 59
        
        # 2. Temperature
        temp_tens, temp_ones = (state.temperature >> 4) & 0x0F, state.temperature & 0x0F
        temp_x_base = lives_x - 16 - 6
        temp_x = jnp.where(temp_tens > 0, temp_x_base - 16, temp_x_base - 8)
        
        # 3. Score
        total_score = self._bcd_to_decimal(state.score)
        num_digits = jnp.where(
            total_score == 0,
            1,
            jnp.floor(jnp.log10(jnp.maximum(1, total_score) + 0.5)).astype(jnp.int32) + 1
        )
        score_x_start = lives_x - ((num_digits - 1) * 4)
        score_digits = self.jr.int_to_digits(total_score, max_digits=6)
        start_index = 6 - num_digits
        
        # Collect everything into a batch
        # 0: Lives, 1: Temp Tens, 2: Temp Ones, 3: Degree, 4-9: Score Digits
        hud_masks = jnp.zeros((10, *digits_masks.shape[1:]), dtype=digits_masks.dtype)
        hud_x = jnp.zeros(10, dtype=jnp.int32)
        hud_y = jnp.zeros(10, dtype=jnp.int32)
        hud_active = jnp.zeros(10, dtype=jnp.bool_)
        
        # Lives
        hud_masks = hud_masks.at[0].set(lives_mask)
        hud_x = hud_x.at[0].set(lives_x)
        hud_y = hud_y.at[0].set(lives_y)
        hud_active = hud_active.at[0].set(is_visible)
        
        # Temp Tens
        hud_masks = hud_masks.at[1].set(digits_masks[temp_tens])
        hud_x = hud_x.at[1].set(temp_x)
        hud_y = hud_y.at[1].set(temp_y)
        hud_active = hud_active.at[1].set(is_visible & (temp_tens > 0))
        
        # Temp Ones
        hud_masks = hud_masks.at[2].set(digits_masks[temp_ones])
        hud_x = hud_x.at[2].set(jnp.where(temp_tens > 0, temp_x + 8, temp_x))
        hud_y = hud_y.at[2].set(temp_y)
        hud_active = hud_active.at[2].set(is_visible)
        
        # Degree
        hud_masks = hud_masks.at[3].set(self.DEGREE_MASK)
        hud_x = hud_x.at[3].set(temp_x_base)
        hud_y = hud_y.at[3].set(temp_y)
        hud_active = hud_active.at[3].set(is_visible)
        
        # Score Digits
        score_idx = jnp.arange(6)
        score_mask_idx = jnp.take(score_digits, start_index + score_idx)
        score_masks = digits_masks[score_mask_idx]
        score_x = score_x_start + score_idx * 8
        
        hud_masks = hud_masks.at[4:].set(score_masks)
        hud_x = hud_x.at[4:].set(score_x)
        hud_y = hud_y.at[4:].set(score_y)
        hud_active = hud_active.at[4:].set(score_idx < num_digits)
        
        # Final filtering
        hud_masks = jnp.where(hud_active[:, jnp.newaxis, jnp.newaxis], hud_masks, self.jr.TRANSPARENT_ID)
        
        return self.jr.render_at_batch(raster, hud_x, hud_y, hud_masks)

    
    @partial(jax.jit, static_argnums=(0,))
    def _render_igloo_blocks(self, raster, state):
        """Render igloo blocks using a pre-built canvas layer (one render_at call)."""
        canvas_idx = jnp.clip(state.building_igloo_idx + 1, 0, 16)
        canvas = self.IGLOO_LAYERS[canvas_idx]
        return self.jr.render_at(raster, self._IGLOO_CX0, self._IGLOO_CY0, canvas)
    
    @partial(jax.jit, static_argnums=(0,))
    def _render_obstacles(self, raster, state):
        """Render all active obstacle copies using sequential render_at calls."""
        copies_per_lane, spacing_per_lane = self._decode_sprite_duplication(
            self.consts, state.obstacle_duplication_mode
        )
        sprite_w = self.ALL_OBSTACLE_MASKS.shape[2]

        for lane in range(4):
            is_active = state.obstacle_active[lane] == 1
            ob_type   = state.obstacle_types[lane]
            anim      = state.obstacle_animation_idx[lane]
            float_off = jnp.clip(state.obstacle_float_offsets[lane].astype(jnp.int32), 0, 4)
            direction = state.obstacle_directions[lane]
            base_x    = state.obstacle_x[lane]
            base_y    = state.obstacle_y[lane]
            copies    = copies_per_lane[lane]
            spacing   = spacing_per_lane[lane]

            is_aquatic = ob_type != self.consts.ID_SNOW_GOOSE
            y_render   = jnp.where(is_aquatic, base_y + float_off, base_y)

            selector = jnp.where(ob_type == self.consts.ID_SNOW_GOOSE, 0,
                       jnp.where(ob_type == self.consts.ID_FISH, 1,
                       jnp.where(ob_type == self.consts.ID_KING_CRAB, 2, 3)))
            mask_idx = jnp.where(selector == 0, anim,
                       jnp.where(selector == 1, 10 + anim * 5 + float_off,
                       jnp.where(selector == 2, 20 + anim * 5 + float_off,
                       30 + anim * 5 + float_off)))

            sprite = self.ALL_OBSTACLE_MASKS[mask_idx]
            sprite = jnp.where(direction == 1, jnp.flip(sprite, axis=1), sprite)

            for copy_idx in range(3):
                x = base_x + copy_idx * spacing
                fish_alive  = ((state.fish_alive_mask[lane] >> jnp.int32(copy_idx)) & 1) == 1
                copy_active = is_active & (jnp.int32(copy_idx) < copies)
                copy_active = jnp.where(ob_type == self.consts.ID_FISH,
                                        copy_active & fish_alive, copy_active)
                on_screen   = (x >= 0) & (x + sprite_w - 1 < self.consts.SCREEN_WIDTH)

                raster = jax.lax.cond(
                    copy_active & on_screen,
                    lambda r, _x=x, _y=y_render, _s=sprite: self.jr.render_at(r, _x, _y, _s),
                    lambda r: r,
                    raster,
                )

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
    def _render_ice_vectorized(self, raster, state):
        """Render all ice segments using vectorized render_at_batch."""
        # 4 rows, 6 segments per row = 24 segments
        # Each can have 3 wrap copies = 72 total entries
        row_indices = jnp.arange(4)
        seg_indices = jnp.arange(6)
        R, S = jnp.meshgrid(row_indices, seg_indices, indexing='ij')
        R = R.flatten() # (24,)
        S = S.flatten() # (24,)

        y_pos = self.ICE_ROW_Y_ARRAY[R]
        is_blue = state.ice_colors[R] == self.consts.COLOR_ICE_BLUE

        seg_x = state.ice_segments_x[R, S]
        seg_w = state.ice_segments_w[R, S]
        active = seg_w > 0

        # Mask selection
        is_narrow = seg_w == 12
        base_idx = jnp.int32(is_blue)
        mask_idx = jnp.where(is_narrow, 2 + base_idx, base_idx)
        masks = self.ICE_MASKS[mask_idx] # (24, H, W)

        # Handle wrap copies
        W = self.consts.PLAYFIELD_WIDTH
        xs = jnp.stack([seg_x, seg_x - W, seg_x + W], axis=1).flatten()
        ys = jnp.stack([y_pos, y_pos, y_pos], axis=1).flatten()
        final_masks = jnp.repeat(masks, 3, axis=0)

        wrap_active = jnp.repeat(active, 3, axis=0)

        xs = jnp.where(wrap_active, xs, -100)
        ys = jnp.where(wrap_active, ys, -100)

        return self.jr.render_at_batch(raster, xs, ys, final_masks)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: FrostbiteState) -> jnp.ndarray:
        """Render the complete game frame from the current state."""

        # 1. Create raster and render background (day/night)
        raster = self.jr.create_object_raster(self.BACKGROUND).astype(self.PALETTE.dtype)
        is_night = ((state.level - 1) // 4) % 2 == 1
        raster = jax.lax.cond(
            is_night,
            lambda r: self.jr.render_at(r, 0, 0, self.SHAPE_MASKS['background_night']),
            lambda r: r,
            raster
        )

        # 2. Render ice blocks (vectorized)
        raster = self._render_ice_vectorized(raster, state)

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
            is_dead, 3 + ((state.bailey_death_frame // 15) % 2), state.bailey_animation_idx
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