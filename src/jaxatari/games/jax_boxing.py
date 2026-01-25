"""
Boxing - JAXAtari Implementation (Phase 1: Minimum Viable Game)

A GPU-accelerated, JAX-based implementation of the Atari 2600 Boxing game.
Phase 1 implements basic environment setup, input handling, and a movable player dot.

Technical Specification Reference: reference_material/TECHNICAL_SPECIFICATION.md
"""

import os
from functools import partial
from typing import NamedTuple, Tuple

import chex
import jax
import jax.numpy as jnp

import jaxatari.spaces as spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils


# =============================================================================
# Asset Config (declarative sprite manifest)
# =============================================================================

def _get_default_asset_config() -> tuple:
    """
    Returns the default declarative asset manifest for Boxing.
    Kept immutable (tuple of dicts) to fit NamedTuple defaults.
    """
    return (
        {'name': 'background', 'type': 'background', 'file': 'background.npy'},
        # Idle sprites
        {'name': 'white_idle', 'type': 'single', 'file': 'white_idle.npy'},
        {'name': 'black_idle', 'type': 'single', 'file': 'black_idle.npy'},
        # White boxer punch animation (left and right direction)
        {'name': 'white_punch_left_0', 'type': 'single', 'file': 'white_boxing_animation_left/0.npy'},
        {'name': 'white_punch_left_1', 'type': 'single', 'file': 'white_boxing_animation_left/1.npy'},
        {'name': 'white_punch_left_2', 'type': 'single', 'file': 'white_boxing_animation_left/2.npy'},
        {'name': 'white_punch_left_3', 'type': 'single', 'file': 'white_boxing_animation_left/3.npy'},
        {'name': 'white_punch_right_0', 'type': 'single', 'file': 'white_boxing_animation_right/0.npy'},
        {'name': 'white_punch_right_1', 'type': 'single', 'file': 'white_boxing_animation_right/1.npy'},
        {'name': 'white_punch_right_2', 'type': 'single', 'file': 'white_boxing_animation_right/2.npy'},
        {'name': 'white_punch_right_3', 'type': 'single', 'file': 'white_boxing_animation_right/3.npy'},
        # Black boxer punch animation (left and right direction)
        {'name': 'black_punch_left_0', 'type': 'single', 'file': 'black_boxing_animation_left/0.npy'},
        {'name': 'black_punch_left_1', 'type': 'single', 'file': 'black_boxing_animation_left/1.npy'},
        {'name': 'black_punch_left_2', 'type': 'single', 'file': 'black_boxing_animation_left/2.npy'},
        {'name': 'black_punch_left_3', 'type': 'single', 'file': 'black_boxing_animation_left/3.npy'},
        {'name': 'black_punch_right_0', 'type': 'single', 'file': 'black_boxing_animation_right/0.npy'},
        {'name': 'black_punch_right_1', 'type': 'single', 'file': 'black_boxing_animation_right/1.npy'},
        {'name': 'black_punch_right_2', 'type': 'single', 'file': 'black_boxing_animation_right/2.npy'},
        {'name': 'black_punch_right_3', 'type': 'single', 'file': 'black_boxing_animation_right/3.npy'},
        # Digit sprites for HUD
        {'name': 'digits_white', 'type': 'digits', 'pattern': 'digits_white/{}.npy'},
        {'name': 'digits_black', 'type': 'digits', 'pattern': 'digits_black/{}.npy'},
        {'name': 'digits_time', 'type': 'digits', 'pattern': 'digits_time/{}.npy'},
    )


# =============================================================================
# Constants (immutable game parameters from Technical Specification)
# =============================================================================

class BoxingConstants(NamedTuple):
    """
    Immutable game constants derived from the Boxing Technical Specification.
    
    Boundaries:
        XMIN_BOXER (30) to XMAX_BOXER (109) for horizontal movement
        YMIN (3) to YMAX (87) for vertical movement
    """
    # Screen dimensions (standard Atari)
    WIDTH: int = 160
    HEIGHT: int = 210
    
    # Boxer boundaries - based on actual ring in background sprite
    # Ring inner area: X from ~32 to ~127, Y from ~34 to ~178
    # Boxer sprite: 14 wide, 47 tall
    XMIN_BOXER: int = 32   # Left edge of playable ring
    XMAX_BOXER: int = 113  # 127 - 14 = 113 so right edge of sprite at 127
    YMIN: int = 34         # Top edge of playable ring
    YMAX: int = 131        # 178 - 47 = 131 so bottom edge of sprite at 178
    
    # Boxer dimensions from spec (sprite is 47x14, but game uses 48 for collision)
    H_BOXER: int = 48  # Boxer height (3 sections × 16 pixels)
    W_BOXER: int = 14  # Sprite width
    SPRITE_HEIGHT: int = 47  # Actual sprite height
    
    # Movement speed (fixed for Phase 1)
    MOVE_SPEED: int = 1
    
    # Initial positions - centered in ring, boxers facing each other
    LEFT_BOXER_START_X: int = 40   # Left side of ring
    RIGHT_BOXER_START_X: int = 105 # Right side of ring
    BOXER_START_Y: int = 82        # Centered vertically: (34 + 131) / 2 ≈ 82
    
    # Timer settings (for future phases)
    CLOCK_MINUTES_START: int = 2
    CLOCK_SECONDS_START: int = 0
    FRAMES_PER_SECOND: int = 60  # NTSC
    
    # Scoring
    MAX_SCORE: int = 100  # KO score
    
    # Hit detection thresholds from spec
    HIT_DISTANCE_HORIZONTAL: int = 29  # (8 * 3) + 5 pixels
    HIT_DISTANCE_VERTICAL: int = 48    # H_BOXER
    STUN_DURATION: int = 15            # Frames boxer is stunned after hit
    
    # Punch animation settings (matching original assembly)
    # Animation values range 0-72, increment by 8 for extension, -2 for retraction
    PUNCH_EXTEND_RATE: int = 8     # +8 per frame while button held
    PUNCH_RETRACT_RATE: int = 2    # -2 per frame when button released
    MAX_PUNCH_EXTENSION_FAR: int = 72    # 9*8: full extension when far apart
    MAX_PUNCH_EXTENSION_MED: int = 56    # 7*8: medium extension
    MAX_PUNCH_EXTENSION_SHORT: int = 40  # 5*8: short extension when close
    # Sprite frame = animation_value / 8, indexes into offset table
    # Frames 0-5: idle/returning, 6-7: punch stage 2, 8-9: full extension
    
    # Colors from boxing.asm (NTSC palette approximations)
    BACKGROUND_COLOR: Tuple[int, int, int] = (0, 100, 0)  # Dark green ring
    LEFT_BOXER_COLOR: Tuple[int, int, int] = (236, 236, 236)  # White/light gray
    RIGHT_BOXER_COLOR: Tuple[int, int, int] = (0, 0, 0)  # Black
    RING_COLOR: Tuple[int, int, int] = (200, 72, 72)  # Red ring posts
    
    # HUD positions (approximate based on original game)
    SCORE_Y: int = 5
    LEFT_SCORE_X: int = 20
    RIGHT_SCORE_X: int = 130
    TIMER_X: int = 70
    DIGIT_SPACING: int = 8  # Space between digits
    
    # Asset config (immutable default for asset overrides)
    ASSET_CONFIG: tuple = _get_default_asset_config()


# =============================================================================
# State (mutable game state - spec-compliant structure)
# =============================================================================

class BoxingState(NamedTuple):
    """
    Complete game state for Boxing, structured per Technical Specification.
    
    Phase 1 uses only player position fields; others are placeholders for future phases.
    """
    # Left boxer (player 1) position
    left_boxer_x: chex.Array
    left_boxer_y: chex.Array
    
    # Right boxer (player 2 / CPU) position - placeholder for future phases
    right_boxer_x: chex.Array
    right_boxer_y: chex.Array
    
    # Scores (BCD 0-99, or 100 for KO)
    left_boxer_score: chex.Array
    right_boxer_score: chex.Array
    
    # Timer (BCD format)
    clock_minutes: chex.Array
    clock_seconds: chex.Array
    frame_count: chex.Array  # Frames within current second
    
    # Combat state - placeholder for future phases
    hit_boxer_stun_timer: chex.Array
    hit_boxer_index: chex.Array  # 0 = left, 1 = right
    
    # Animation state - placeholder for future phases
    boxer_animation_values: chex.Array  # 8-element array
    
    # Punch state (animation values 0-72, not frame indices)
    extended_arm_maximum: chex.Array  # 2-element array: current max extension per boxer
    left_boxer_punch_active: chex.Array  # 1 if punching, 0 if not
    right_boxer_punch_active: chex.Array  # 1 if punching, 0 if not
    left_boxer_animation_value: chex.Array  # Current animation value (0-72)
    right_boxer_animation_value: chex.Array  # Current animation value (0-72)
    left_boxer_punch_landed: chex.Array  # 1 if punch already scored this extension
    right_boxer_punch_landed: chex.Array  # 1 if punch already scored this extension
    left_boxer_last_arm: chex.Array  # 0 = left arm, 1 = right arm (for alternating)
    right_boxer_last_arm: chex.Array  # 0 = left arm, 1 = right arm
    
    # CPU AI state
    cpu_target_x: chex.Array  # Target X position CPU is tracking
    cpu_target_y: chex.Array  # Target Y position CPU is tracking
    cpu_horiz_offset: chex.Array  # Random horizontal offset (0-31)
    cpu_vert_offset: chex.Array  # Random vertical offset (0-63)
    cpu_dancing_value: chex.Array  # Timer controlling CPU "dancing" behavior
    
    # Game flow
    game_state: chex.Array  # 0 = active, 0xFF = game over
    step_counter: chex.Array
    
    # PRNG key for randomness
    key: chex.PRNGKey


# =============================================================================
# Observation (what the agent sees - spec-compliant)
# =============================================================================

class EntityPosition(NamedTuple):
    """Position and dimensions of a game entity."""
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray


class BoxingObservation(NamedTuple):
    """
    Observable game state for Boxing.
    
    Includes both boxer positions and game state information
    for object-centric reinforcement learning.
    """
    left_boxer: EntityPosition
    right_boxer: EntityPosition
    score_left: jnp.ndarray
    score_right: jnp.ndarray
    clock_minutes: jnp.ndarray
    clock_seconds: jnp.ndarray


# =============================================================================
# Info (auxiliary information)
# =============================================================================

class BoxingInfo(NamedTuple):
    """Auxiliary info returned with each step."""
    time: jnp.ndarray  # Total frames elapsed
    clock_minutes: jnp.ndarray
    clock_seconds: jnp.ndarray


# =============================================================================
# Main Environment Class
# =============================================================================

class JaxBoxing(JaxEnvironment[BoxingState, BoxingObservation, BoxingInfo, BoxingConstants]):
    """
    JAX-based Boxing environment.
    
    Phase 1 MVP: Single movable dot representing the player.
    Responds to directional input and respects boundary constraints.
    """
    
    def __init__(self, consts: BoxingConstants = None):
        consts = consts or BoxingConstants()
        super().__init__(consts)
        self.renderer = BoxingRenderer(self.consts)
        
        # Full action set for Boxing (all directions + punch combinations)
        self.action_set = [
            Action.NOOP,
            Action.FIRE,          # Punch
            Action.UP,
            Action.RIGHT,
            Action.LEFT,
            Action.DOWN,
            Action.UPRIGHT,
            Action.UPLEFT,
            Action.DOWNRIGHT,
            Action.DOWNLEFT,
            Action.UPFIRE,        # Move + punch combinations
            Action.RIGHTFIRE,
            Action.LEFTFIRE,
            Action.DOWNFIRE,
            Action.UPRIGHTFIRE,
            Action.UPLEFTFIRE,
            Action.DOWNRIGHTFIRE,
            Action.DOWNLEFTFIRE,
        ]
    
    def reset(self, key: chex.PRNGKey = jax.random.PRNGKey(42)) -> Tuple[BoxingObservation, BoxingState]:
        """Initialize game state per Technical Specification."""
        state_key, _step_key = jax.random.split(key)
        
        state = BoxingState(
            # Left boxer starts at left side, centered vertically
            left_boxer_x=jnp.array(self.consts.LEFT_BOXER_START_X, dtype=jnp.int32),
            left_boxer_y=jnp.array(self.consts.BOXER_START_Y, dtype=jnp.int32),
            
            # Right boxer starts at right side (placeholder for future)
            right_boxer_x=jnp.array(self.consts.RIGHT_BOXER_START_X, dtype=jnp.int32),
            right_boxer_y=jnp.array(self.consts.BOXER_START_Y, dtype=jnp.int32),
            
            # Scores start at 0
            left_boxer_score=jnp.array(0, dtype=jnp.int32),
            right_boxer_score=jnp.array(0, dtype=jnp.int32),
            
            # Timer starts at 2:00
            clock_minutes=jnp.array(self.consts.CLOCK_MINUTES_START, dtype=jnp.int32),
            clock_seconds=jnp.array(self.consts.CLOCK_SECONDS_START, dtype=jnp.int32),
            frame_count=jnp.array(0, dtype=jnp.int32),
            
            # Combat state (inactive)
            hit_boxer_stun_timer=jnp.array(0, dtype=jnp.int32),
            hit_boxer_index=jnp.array(0, dtype=jnp.int32),
            
            # Animation state (idle)
            boxer_animation_values=jnp.zeros(8, dtype=jnp.int32),
            
            # Punch state (animation values 0-72)
            extended_arm_maximum=jnp.zeros(2, dtype=jnp.int32),
            left_boxer_punch_active=jnp.array(0, dtype=jnp.int32),
            right_boxer_punch_active=jnp.array(0, dtype=jnp.int32),
            left_boxer_animation_value=jnp.array(0, dtype=jnp.int32),  # 0-72 range
            right_boxer_animation_value=jnp.array(0, dtype=jnp.int32),
            left_boxer_punch_landed=jnp.array(0, dtype=jnp.int32),  # Debounce: 1 if scored
            right_boxer_punch_landed=jnp.array(0, dtype=jnp.int32),
            left_boxer_last_arm=jnp.array(0, dtype=jnp.int32),  # Alternating arms
            right_boxer_last_arm=jnp.array(0, dtype=jnp.int32),
            
            # CPU AI state
            cpu_target_x=jnp.array(self.consts.LEFT_BOXER_START_X, dtype=jnp.int32),
            cpu_target_y=jnp.array(self.consts.BOXER_START_Y, dtype=jnp.int32),
            cpu_horiz_offset=jnp.array(0, dtype=jnp.int32),
            cpu_vert_offset=jnp.array(0, dtype=jnp.int32),
            cpu_dancing_value=jnp.array(0, dtype=jnp.int32),
            
            # Game active
            game_state=jnp.array(0, dtype=jnp.int32),
            step_counter=jnp.array(0, dtype=jnp.int32),
            
            key=state_key,
        )
        
        initial_obs = self._get_observation(state)
        return initial_obs, state
    
    def _player_step(self, state: BoxingState, action: chex.Array) -> BoxingState:
        """
        Handle player movement based on joystick input.
        
        Phase 1: Simple directional movement with boundary clamping.
        Movement is blocked if stunned.
        """
        # Check if player is stunned (hit_boxer_index == 0 means left boxer was hit)
        is_stunned = jnp.logical_and(
            state.hit_boxer_stun_timer > 0,
            state.hit_boxer_index == 0
        )
        
        speed = self.consts.MOVE_SPEED
        
        # Decode directional input from action
        up = jnp.isin(action, jnp.array([
            Action.UP, Action.UPRIGHT, Action.UPLEFT,
            Action.UPFIRE, Action.UPRIGHTFIRE, Action.UPLEFTFIRE
        ]))
        down = jnp.isin(action, jnp.array([
            Action.DOWN, Action.DOWNRIGHT, Action.DOWNLEFT,
            Action.DOWNFIRE, Action.DOWNRIGHTFIRE, Action.DOWNLEFTFIRE
        ]))
        left = jnp.isin(action, jnp.array([
            Action.LEFT, Action.UPLEFT, Action.DOWNLEFT,
            Action.LEFTFIRE, Action.UPLEFTFIRE, Action.DOWNLEFTFIRE
        ]))
        right = jnp.isin(action, jnp.array([
            Action.RIGHT, Action.UPRIGHT, Action.DOWNRIGHT,
            Action.RIGHTFIRE, Action.UPRIGHTFIRE, Action.DOWNRIGHTFIRE
        ]))
        
        # Calculate movement deltas
        dx = jnp.where(right, speed, jnp.where(left, -speed, 0))
        dy = jnp.where(down, speed, jnp.where(up, -speed, 0))
        
        # Apply movement with boundary clamping (blocked if stunned)
        new_x = jnp.where(
            is_stunned,
            state.left_boxer_x,
            jnp.clip(
                state.left_boxer_x + dx,
                self.consts.XMIN_BOXER,
                self.consts.XMAX_BOXER
            )
        )
        new_y = jnp.where(
            is_stunned,
            state.left_boxer_y,
            jnp.clip(
                state.left_boxer_y + dy,
                self.consts.YMIN,
                self.consts.YMAX
            )
        )
        
        return state._replace(
            left_boxer_x=new_x.astype(jnp.int32),
            left_boxer_y=new_y.astype(jnp.int32),
        )
    
    def _calculate_max_extension(self, horiz_dist: chex.Array, vert_dist: chex.Array) -> chex.Array:
        """
        Calculate maximum punch extension value based on distance between boxers.
        
        Per boxing.asm lines 703-729:
        - Far apart (horiz > 26): full extension (72 = 9*8)
        - Medium distance (horiz > 18): check vertical
        - Close (horiz <= 18): short extension (40 = 5*8)
        
        Returns maximum animation value (40, 56, or 72).
        """
        # Far apart: full extension
        far_apart = horiz_dist > 26  # (8 * 3) + 2
        
        # Medium distance  
        medium_dist = horiz_dist > 18  # (8 * 2) + 2
        
        # Vertical distance thresholds
        vert_close = vert_dist < 7   # H_KERNEL_SECTION / 2 - 1
        vert_medium = vert_dist < 28  # (H_KERNEL_SECTION * 2) - 4
        vert_far = vert_dist >= 47   # (H_KERNEL_SECTION * 3) - 1
        
        # Per assembly logic:
        # If horiz > 26: max = 72 (full)
        # Else if horiz > 18:
        #   Start with 56
        #   If vert < 7: stay at 56 (will become 40 after adjustment? Actually no increment)
        #   If vert < 28: increment to 72
        #   If vert < 47: stay at 56
        #   Else: increment to 72
        # Else (close):
        #   Start with 40
        #   If vert < 7: stay at 40
        #   If vert < 28: increment to 56
        #   Else: stay at 40
        
        max_extension = jnp.where(
            far_apart,
            self.consts.MAX_PUNCH_EXTENSION_FAR,  # 72
            jnp.where(
                medium_dist,
                # Medium horizontal distance
                jnp.where(
                    vert_close,
                    self.consts.MAX_PUNCH_EXTENSION_SHORT,  # 40 - vert < 7
                    jnp.where(
                        vert_medium,
                        self.consts.MAX_PUNCH_EXTENSION_FAR,  # 72 - vert < 28, increment
                        jnp.where(
                            vert_far,
                            self.consts.MAX_PUNCH_EXTENSION_FAR,  # 72 - vert >= 47
                            self.consts.MAX_PUNCH_EXTENSION_MED  # 56 - middle range
                        )
                    )
                ),
                # Close horizontal distance
                jnp.where(
                    vert_close,
                    self.consts.MAX_PUNCH_EXTENSION_SHORT,  # 40 - vert < 7
                    jnp.where(
                        vert_medium,
                        self.consts.MAX_PUNCH_EXTENSION_MED,  # 56 - vert < 28
                        self.consts.MAX_PUNCH_EXTENSION_SHORT  # 40 - far vertically
                    )
                )
            )
        ).astype(jnp.int32)
        
        return max_extension

    def _punch_step(self, state: BoxingState, action: chex.Array) -> BoxingState:
        """
        Handle punch action with button-driven animation per original assembly.
        
        Per boxing.asm:
        - Animation value ranges 0-72
        - While FIRE held: value += 8 (fast extension)
        - When FIRE released: value -= 2 (slow retraction)
        - Value capped at extendedArmMaximum (which adjusts toward maximumPunchExtension)
        - Hit detection triggers once when first reaching max extension
        """
        # Check if FIRE is pressed (any action containing FIRE)
        fire_pressed = jnp.isin(action, jnp.array([
            Action.FIRE,
            Action.UPFIRE, Action.DOWNFIRE, Action.LEFTFIRE, Action.RIGHTFIRE,
            Action.UPLEFTFIRE, Action.UPRIGHTFIRE, Action.DOWNLEFTFIRE, Action.DOWNRIGHTFIRE
        ]))
        
        # Calculate maximum punch extension based on distance to opponent
        horiz_dist = jnp.abs(state.left_boxer_x - state.right_boxer_x)
        vert_dist = jnp.abs(state.left_boxer_y - state.right_boxer_y)
        max_extension = self._calculate_max_extension(horiz_dist, vert_dist)
        
        # Get current extended arm maximum (per-boxer cap that adjusts over time)
        # For simplicity, we'll use the calculated max_extension directly
        # (original uses gradual adjustment, but this captures the key behavior)
        current_max = max_extension
        
        # Calculate new animation value based on button state
        current_anim = state.left_boxer_animation_value
        
        new_anim = jnp.where(
            fire_pressed,
            # Button held: extend (+8 per frame, capped at max)
            jnp.minimum(current_anim + self.consts.PUNCH_EXTEND_RATE, current_max),
            # Button released: retract (-2 per frame, min 0)
            jnp.maximum(current_anim - self.consts.PUNCH_RETRACT_RATE, 0)
        ).astype(jnp.int32)
        
        # Punch is active when animation value > 0
        punch_active = jnp.where(new_anim > 0, 1, 0).astype(jnp.int32)
        
        # Detect when punch FIRST reaches max extension (for hit detection trigger)
        # This is when: current < max AND new >= max (transition to max)
        just_reached_max = jnp.logical_and(
            current_anim < current_max,
            new_anim >= current_max
        )
        
        # Reset punch_landed when animation goes to 0 (allows new punch cycle)
        animation_reset = new_anim == 0
        new_punch_landed = jnp.where(
            animation_reset,
            0,  # Reset debounce when fully retracted
            state.left_boxer_punch_landed
        ).astype(jnp.int32)
        
        # Toggle arm when starting a new punch (transition from 0 to > 0)
        starting_punch = jnp.logical_and(current_anim == 0, new_anim > 0)
        new_last_arm = jnp.where(
            starting_punch,
            1 - state.left_boxer_last_arm,  # Toggle: 0 -> 1 or 1 -> 0
            state.left_boxer_last_arm
        ).astype(jnp.int32)
        
        # Store whether we just reached max (for hit detection in next step)
        # We'll use extended_arm_maximum[0] to track this
        new_extended_arm_max = state.extended_arm_maximum.at[0].set(
            jnp.where(just_reached_max, 1, 0).astype(jnp.int32)
        )
        
        return state._replace(
            left_boxer_punch_active=punch_active,
            left_boxer_animation_value=new_anim,
            left_boxer_punch_landed=new_punch_landed,
            left_boxer_last_arm=new_last_arm,
            extended_arm_maximum=new_extended_arm_max,
        )
    
    def _hit_detection_step(self, state: BoxingState) -> BoxingState:
        """
        Check if player's punch hits the opponent.
        
        Per boxing.asm (CheckToScoreBoxerForPunch):
        - Hit only triggers ONCE when punch FIRST reaches max extension
        - Horizontal distance <= (8*3)+5 = 29 pixels
        - Vertical distance < H_BOXER (48 pixels)
        - Fine vertical alignment: (verticalDistance - 11) < 18
        - Opponent must not already be stunned
        - Punch must not have already landed this cycle (debounce)
        
        Key: Hit detection uses the just_reached_max flag from _punch_step.
        """
        # Calculate distances
        horiz_dist = jnp.abs(state.left_boxer_x - state.right_boxer_x)
        vert_dist = jnp.abs(state.left_boxer_y - state.right_boxer_y)
        
        # Check horizontal range: must be within (8*3)+5 = 29 pixels
        in_horiz_range = horiz_dist <= self.consts.HIT_DISTANCE_HORIZONTAL
        
        # Check vertical range: must be less than H_BOXER (48 pixels)
        in_vert_range = vert_dist < self.consts.HIT_DISTANCE_VERTICAL
        
        # Fine vertical alignment check per spec: (verticalDistance - 11) < 18
        vert_offset = vert_dist - 11
        fine_vert_aligned = vert_offset < 18
        
        # Hit only triggers on the frame when punch FIRST reaches max extension
        # extended_arm_maximum[0] is set to 1 by _punch_step when this happens
        just_reached_max = state.extended_arm_maximum[0] == 1
        
        punch_active = state.left_boxer_punch_active > 0
        
        # Check debounce - only score if this punch hasn't already landed
        punch_not_landed_yet = state.left_boxer_punch_landed == 0
        
        # Per original assembly: hit requires:
        # 1. Punch active and just reached max extension
        # 2. Horizontal distance <= 29 (in punching range)
        # 3. Vertical distance < 48 (H_BOXER)
        # 4. Fine vertical alignment: (vert_dist - 11) < 18 (head level)
        # 5. Haven't already scored on this punch
        hit_landed = jnp.logical_and(
            jnp.logical_and(punch_active, just_reached_max),
            jnp.logical_and(
                jnp.logical_and(in_horiz_range, in_vert_range),
                jnp.logical_and(fine_vert_aligned, punch_not_landed_yet)
            )
        )
        
        # Only register hit if opponent is not already stunned
        opponent_not_stunned = jnp.logical_or(
            state.hit_boxer_stun_timer == 0,
            state.hit_boxer_index != 1  # 1 = right boxer
        )
        valid_hit = jnp.logical_and(hit_landed, opponent_not_stunned)
        
        # Set debounce flag if hit landed (prevents multiple hits per punch)
        new_punch_landed = jnp.where(
            valid_hit,
            1,  # Mark as landed
            state.left_boxer_punch_landed
        ).astype(jnp.int32)
        
        # Increment score if hit (1 point per hit)
        new_score = jnp.where(
            valid_hit,
            state.left_boxer_score + 1,
            state.left_boxer_score
        ).astype(jnp.int32)
        
        # Set stun timer for opponent (right boxer = index 1)
        new_stun_timer = jnp.where(
            valid_hit,
            self.consts.STUN_DURATION,
            state.hit_boxer_stun_timer
        ).astype(jnp.int32)
        
        new_hit_index = jnp.where(
            valid_hit,
            1,  # Right boxer got hit
            state.hit_boxer_index
        ).astype(jnp.int32)
        
        # Set dancing value when player scores (affects CPU behavior)
        new_dancing = jnp.where(
            valid_hit,
            57,  # Per spec: cpuBoxerDancingValue = 57
            state.cpu_dancing_value
        ).astype(jnp.int32)
        
        return state._replace(
            left_boxer_score=new_score,
            hit_boxer_stun_timer=new_stun_timer,
            hit_boxer_index=new_hit_index,
            left_boxer_punch_landed=new_punch_landed,
            cpu_dancing_value=new_dancing,
        )
    
    def _timer_step(self, state: BoxingState) -> BoxingState:
        """
        Decrement the game clock.
        
        Clock counts down from 2:00 to 0:00 at 60 frames per second.
        Game ends when timer reaches 0:00.
        """
        # Increment frame count
        new_frame_count = state.frame_count + 1
        
        # Check if a second has passed (60 frames = 1 second for NTSC)
        second_passed = new_frame_count >= self.consts.FRAMES_PER_SECOND
        
        # Reset frame count if second passed
        new_frame_count = jnp.where(second_passed, 0, new_frame_count)
        
        # Decrement seconds if a second passed
        new_seconds = jnp.where(
            second_passed,
            state.clock_seconds - 1,
            state.clock_seconds
        )
        
        # Handle seconds underflow (59 -> 0 -> wrap to 59, decrement minute)
        seconds_underflow = jnp.logical_and(second_passed, state.clock_seconds == 0)
        new_seconds = jnp.where(seconds_underflow, 59, new_seconds)
        
        # Decrement minutes on seconds underflow
        new_minutes = jnp.where(
            seconds_underflow,
            state.clock_minutes - 1,
            state.clock_minutes
        )
        
        # Check for timer expired (would go below 0:00)
        timer_expired = jnp.logical_and(
            seconds_underflow,
            state.clock_minutes == 0
        )
        
        # Set game over if timer expired
        new_game_state = jnp.where(
            timer_expired,
            0xFF,  # Game over
            state.game_state
        ).astype(jnp.int32)
        
        # Clamp values to valid ranges
        new_minutes = jnp.maximum(new_minutes, 0).astype(jnp.int32)
        new_seconds = jnp.clip(new_seconds, 0, 59).astype(jnp.int32)
        
        return state._replace(
            frame_count=new_frame_count.astype(jnp.int32),
            clock_seconds=new_seconds,
            clock_minutes=new_minutes,
            game_state=new_game_state,
        )
    
    def _collision_step(self, state: BoxingState, prev_left_x: chex.Array, prev_left_y: chex.Array,
                        prev_right_x: chex.Array, prev_right_y: chex.Array) -> BoxingState:
        """
        Check for boxer-boxer collision and revert positions if overlapping.
        
        Per spec: Boxers cannot overlap. If collision detected, revert to previous position.
        Collision box: W_BOXER (14) wide, H_BOXER (48) tall
        """
        # Calculate current distances
        horiz_dist = jnp.abs(state.left_boxer_x - state.right_boxer_x)
        vert_dist = jnp.abs(state.left_boxer_y - state.right_boxer_y)
        
        # Check if boxers are overlapping
        horiz_overlap = horiz_dist < self.consts.W_BOXER
        vert_overlap = vert_dist < self.consts.H_BOXER
        collision = jnp.logical_and(horiz_overlap, vert_overlap)
        
        # Revert both boxers to previous positions on collision
        new_left_x = jnp.where(collision, prev_left_x, state.left_boxer_x).astype(jnp.int32)
        new_left_y = jnp.where(collision, prev_left_y, state.left_boxer_y).astype(jnp.int32)
        new_right_x = jnp.where(collision, prev_right_x, state.right_boxer_x).astype(jnp.int32)
        new_right_y = jnp.where(collision, prev_right_y, state.right_boxer_y).astype(jnp.int32)
        
        return state._replace(
            left_boxer_x=new_left_x,
            left_boxer_y=new_left_y,
            right_boxer_x=new_right_x,
            right_boxer_y=new_right_y,
        )
    
    def _cpu_movement_step(self, state: BoxingState) -> BoxingState:
        """
        CPU AI movement logic based on Technical Specification.
        
        The CPU tracks the player with some randomized offset and moves toward them.
        Has "dancing" behavior after scoring or being hit.
        """
        # Check if CPU is stunned (hit_boxer_index == 1 means right/CPU boxer was hit)
        is_stunned = jnp.logical_and(
            state.hit_boxer_stun_timer > 0,
            state.hit_boxer_index == 1
        )
        
        # Split key for random decisions
        key, subkey1, subkey2, subkey3 = jax.random.split(state.key, 4)
        
        # Periodically update target position (every ~8 frames based on random)
        random_val = jax.random.randint(subkey1, (), 0, 256)
        update_target = (random_val & 0x07) == 0  # ~1/8 chance per frame
        
        # Generate new random offsets
        new_horiz_offset = jax.random.randint(subkey2, (), 0, 32)  # 0-31
        new_vert_offset = jax.random.randint(subkey3, (), 0, 64)   # 0-63
        
        # Update target to track player position
        cpu_target_x = jnp.where(
            update_target,
            state.left_boxer_x,
            state.cpu_target_x
        ).astype(jnp.int32)
        cpu_target_y = jnp.where(
            update_target,
            state.left_boxer_y,
            state.cpu_target_y
        ).astype(jnp.int32)
        cpu_horiz_offset = jnp.where(
            update_target,
            new_horiz_offset,
            state.cpu_horiz_offset
        ).astype(jnp.int32)
        cpu_vert_offset = jnp.where(
            update_target,
            new_vert_offset,
            state.cpu_vert_offset
        ).astype(jnp.int32)
        
        # Calculate target position with offset
        # CPU tries to stay at a fighting distance from player
        target_x = cpu_target_x + 20 + (cpu_horiz_offset - 16)  # Offset from player
        target_y = cpu_target_y + (cpu_vert_offset - 32)
        
        # Clamp target to ring boundaries
        target_x = jnp.clip(target_x, self.consts.XMIN_BOXER, self.consts.XMAX_BOXER)
        target_y = jnp.clip(target_y, self.consts.YMIN, self.consts.YMAX)
        
        # Determine movement direction
        move_right = target_x > state.right_boxer_x
        move_left = target_x < state.right_boxer_x
        move_down = target_y > state.right_boxer_y
        move_up = target_y < state.right_boxer_y
        
        # "Dancing" behavior - reverse horizontal movement when dancing and not hit
        dancing = state.cpu_dancing_value >= 16
        cpu_not_hit = state.hit_boxer_index != 1
        reverse_horiz = jnp.logical_and(dancing, cpu_not_hit)
        
        # Apply reversal
        move_right_final = jnp.where(reverse_horiz, move_left, move_right)
        move_left_final = jnp.where(reverse_horiz, move_right, move_left)
        
        # Calculate deltas
        dx = jnp.where(move_right_final, 1, jnp.where(move_left_final, -1, 0))
        dy = jnp.where(move_down, 1, jnp.where(move_up, -1, 0))
        
        # Apply movement (blocked if stunned)
        new_x = jnp.where(
            is_stunned,
            state.right_boxer_x,
            jnp.clip(
                state.right_boxer_x + dx,
                self.consts.XMIN_BOXER,
                self.consts.XMAX_BOXER
            )
        ).astype(jnp.int32)
        new_y = jnp.where(
            is_stunned,
            state.right_boxer_y,
            jnp.clip(
                state.right_boxer_y + dy,
                self.consts.YMIN,
                self.consts.YMAX
            )
        ).astype(jnp.int32)
        
        # Decrement dancing value
        new_dancing = jnp.maximum(state.cpu_dancing_value - 1, 0).astype(jnp.int32)
        
        return state._replace(
            right_boxer_x=new_x,
            right_boxer_y=new_y,
            cpu_target_x=cpu_target_x,
            cpu_target_y=cpu_target_y,
            cpu_horiz_offset=cpu_horiz_offset,
            cpu_vert_offset=cpu_vert_offset,
            cpu_dancing_value=new_dancing,
            key=key,
        )
    
    def _cpu_punch_step(self, state: BoxingState) -> BoxingState:
        """
        CPU punch decision and animation logic (combined for CPU).
        
        CPU decides when to START a punch based on range and randomness.
        Once started, CPU holds fire until reaching max extension, then releases.
        Uses same animation value system as player (+8/-2).
        """
        # Calculate distances for punch decision
        horiz_dist = jnp.abs(state.left_boxer_x - state.right_boxer_x)
        vert_dist = jnp.abs(state.left_boxer_y - state.right_boxer_y)
        
        # Get current animation value and max extension
        current_anim = state.right_boxer_animation_value
        max_extension = self._calculate_max_extension(horiz_dist, vert_dist)
        
        # Check if in punching range
        in_horiz_range = horiz_dist <= self.consts.HIT_DISTANCE_HORIZONTAL + 10
        in_vert_range = vert_dist < self.consts.H_BOXER
        in_range = jnp.logical_and(in_horiz_range, in_vert_range)
        
        # Random punch decision (only matters when NOT already punching)
        key, subkey = jax.random.split(state.key)
        random_val = jax.random.randint(subkey, (), 0, 256)
        
        # More aggressive when losing, less when winning
        score_diff = state.right_boxer_score - state.left_boxer_score
        aggressiveness = jnp.where(score_diff >= 0, 40, 20)
        
        # Random threshold check for STARTING a punch
        should_start_punch = random_val < aggressiveness
        
        # Don't start a punch while dancing (unless CPU was hit)
        dancing = state.cpu_dancing_value > 0
        cpu_was_hit = state.hit_boxer_index == 1
        can_punch_dancing = jnp.logical_or(~dancing, cpu_was_hit)
        
        # Determine if CPU should "hold fire":
        # 1. If already punching (anim > 0) and hasn't reached max yet, keep holding
        # 2. If not punching, start if in range and random says so
        already_punching = current_anim > 0
        reached_max = current_anim >= max_extension
        
        cpu_fire_held = jnp.where(
            already_punching,
            # Already punching: keep holding until we reach max
            ~reached_max,
            # Not punching: decide whether to start
            jnp.logical_and(
                jnp.logical_and(in_range, should_start_punch),
                can_punch_dancing
            )
        )
        
        # Calculate new animation value based on CPU decision
        new_anim = jnp.where(
            cpu_fire_held,
            # "Fire held": extend (+8 per frame, capped at max)
            jnp.minimum(current_anim + self.consts.PUNCH_EXTEND_RATE, max_extension),
            # "Fire released": retract (-2 per frame, min 0)
            jnp.maximum(current_anim - self.consts.PUNCH_RETRACT_RATE, 0)
        ).astype(jnp.int32)
        
        # Punch is active when animation value > 0
        punch_active = jnp.where(new_anim > 0, 1, 0).astype(jnp.int32)
        
        # Detect when punch FIRST reaches max extension
        just_reached_max = jnp.logical_and(
            current_anim < max_extension,
            new_anim >= max_extension
        )
        
        # Reset punch_landed when animation goes to 0
        animation_reset = new_anim == 0
        new_punch_landed = jnp.where(
            animation_reset,
            0,
            state.right_boxer_punch_landed
        ).astype(jnp.int32)
        
        # Toggle arm when starting a new punch
        starting_punch = jnp.logical_and(current_anim == 0, new_anim > 0)
        new_last_arm = jnp.where(
            starting_punch,
            1 - state.right_boxer_last_arm,
            state.right_boxer_last_arm
        ).astype(jnp.int32)
        
        # Store whether we just reached max (for hit detection)
        new_extended_arm_max = state.extended_arm_maximum.at[1].set(
            jnp.where(just_reached_max, 1, 0).astype(jnp.int32)
        )
        
        return state._replace(
            right_boxer_punch_active=punch_active,
            right_boxer_animation_value=new_anim,
            right_boxer_punch_landed=new_punch_landed,
            right_boxer_last_arm=new_last_arm,
            extended_arm_maximum=new_extended_arm_max,
            key=key,
        )
    
    def _cpu_hit_detection_step(self, state: BoxingState) -> BoxingState:
        """
        Check if CPU's punch hits the player.
        
        Per boxing.asm (CheckToScoreBoxerForPunch):
        - Hit only triggers ONCE when punch FIRST reaches max extension
        - Horizontal distance <= (8*3)+5 = 29 pixels
        - Vertical distance < H_BOXER (48 pixels)
        - Fine vertical alignment: (verticalDistance - 11) < 18
        - Opponent must not already be stunned
        """
        # Calculate distances
        horiz_dist = jnp.abs(state.left_boxer_x - state.right_boxer_x)
        vert_dist = jnp.abs(state.left_boxer_y - state.right_boxer_y)
        
        # Check horizontal range
        in_horiz_range = horiz_dist <= self.consts.HIT_DISTANCE_HORIZONTAL
        
        # Check vertical range
        in_vert_range = vert_dist < self.consts.HIT_DISTANCE_VERTICAL
        
        # Fine vertical alignment check per spec
        vert_offset = vert_dist - 11
        fine_vert_aligned = vert_offset < 18
        
        # Hit only triggers on the frame when punch FIRST reaches max extension
        just_reached_max = state.extended_arm_maximum[1] == 1
        
        punch_active = state.right_boxer_punch_active > 0
        punch_not_landed_yet = state.right_boxer_punch_landed == 0
        
        # Per original assembly: hit requires:
        # 1. Punch active and just reached max extension
        # 2. Horizontal distance <= 29 (in punching range)
        # 3. Vertical distance < 48 (H_BOXER)
        # 4. Fine vertical alignment: (vert_dist - 11) < 18 (head level)
        # 5. Haven't already scored on this punch
        hit_landed = jnp.logical_and(
            jnp.logical_and(punch_active, just_reached_max),
            jnp.logical_and(
                jnp.logical_and(in_horiz_range, in_vert_range),
                jnp.logical_and(fine_vert_aligned, punch_not_landed_yet)
            )
        )
        
        # Only register hit if player is not already stunned
        player_not_stunned = jnp.logical_or(
            state.hit_boxer_stun_timer == 0,
            state.hit_boxer_index != 0  # 0 = left boxer
        )
        valid_hit = jnp.logical_and(hit_landed, player_not_stunned)
        
        # Set debounce flag
        new_punch_landed = jnp.where(
            valid_hit,
            1,
            state.right_boxer_punch_landed
        ).astype(jnp.int32)
        
        # Increment CPU score
        new_score = jnp.where(
            valid_hit,
            state.right_boxer_score + 1,
            state.right_boxer_score
        ).astype(jnp.int32)
        
        # Set stun timer for player (left boxer = index 0)
        new_stun_timer = jnp.where(
            valid_hit,
            self.consts.STUN_DURATION,
            state.hit_boxer_stun_timer
        ).astype(jnp.int32)
        
        new_hit_index = jnp.where(
            valid_hit,
            0,  # Left boxer got hit
            state.hit_boxer_index
        ).astype(jnp.int32)
        
        # Set dancing value when CPU scores
        new_dancing = jnp.where(
            valid_hit,
            57,  # Per spec: cpuBoxerDancingValue = 57
            state.cpu_dancing_value
        ).astype(jnp.int32)
        
        return state._replace(
            right_boxer_score=new_score,
            hit_boxer_stun_timer=new_stun_timer,
            hit_boxer_index=new_hit_index,
            right_boxer_punch_landed=new_punch_landed,
            cpu_dancing_value=new_dancing,
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: BoxingState, action: chex.Array) -> Tuple[BoxingObservation, BoxingState, float, bool, BoxingInfo]:
        """Execute one game step."""
        # Split PRNG key
        new_state_key, step_key = jax.random.split(state.key)
        previous_state = state
        
        # Store previous positions for collision detection
        prev_left_x = state.left_boxer_x
        prev_left_y = state.left_boxer_y
        prev_right_x = state.right_boxer_x
        prev_right_y = state.right_boxer_y
        
        # Update key for this step
        state = state._replace(key=step_key)
        
        # Process player movement
        state = self._player_step(state, action)
        
        # Process player punch
        state = self._punch_step(state, action)
        
        # Process CPU movement
        state = self._cpu_movement_step(state)
        
        # Check for boxer-boxer collision (revert positions if overlapping)
        state = self._collision_step(state, prev_left_x, prev_left_y, prev_right_x, prev_right_y)
        
        # Process CPU punch decision and animation
        state = self._cpu_punch_step(state)
        
        # Check for player hits on CPU
        state = self._hit_detection_step(state)
        
        # Check for CPU hits on player
        state = self._cpu_hit_detection_step(state)
        
        # Decrement stun timer
        new_stun = jnp.maximum(state.hit_boxer_stun_timer - 1, 0).astype(jnp.int32)
        state = state._replace(hit_boxer_stun_timer=new_stun)
        
        # Update game timer
        state = self._timer_step(state)
        
        # Increment step counter
        state = state._replace(
            step_counter=state.step_counter + 1,
            key=new_state_key,
        )
        
        # Get outputs
        done = self._get_done(state)
        reward = self._get_reward(previous_state, state)
        info = self._get_info(state)
        observation = self._get_observation(state)
        
        return observation, state, reward, done, info
    
    def render(self, state: BoxingState) -> jnp.ndarray:
        """Render the current game state to an image."""
        return self.renderer.render(state)
    
    def _get_observation(self, state: BoxingState) -> BoxingObservation:
        """Extract observable state."""
        left_boxer = EntityPosition(
            x=state.left_boxer_x,
            y=state.left_boxer_y,
            width=jnp.array(self.consts.W_BOXER),
            height=jnp.array(self.consts.SPRITE_HEIGHT),
        )
        
        right_boxer = EntityPosition(
            x=state.right_boxer_x,
            y=state.right_boxer_y,
            width=jnp.array(self.consts.W_BOXER),
            height=jnp.array(self.consts.SPRITE_HEIGHT),
        )
        
        return BoxingObservation(
            left_boxer=left_boxer,
            right_boxer=right_boxer,
            score_left=state.left_boxer_score,
            score_right=state.right_boxer_score,
            clock_minutes=state.clock_minutes,
            clock_seconds=state.clock_seconds,
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: BoxingObservation) -> jnp.ndarray:
        """Flatten observation for neural network input."""
        return jnp.concatenate([
            obs.left_boxer.x.flatten(),
            obs.left_boxer.y.flatten(),
            obs.left_boxer.width.flatten(),
            obs.left_boxer.height.flatten(),
            obs.right_boxer.x.flatten(),
            obs.right_boxer.y.flatten(),
            obs.right_boxer.width.flatten(),
            obs.right_boxer.height.flatten(),
            obs.score_left.flatten(),
            obs.score_right.flatten(),
            obs.clock_minutes.flatten(),
            obs.clock_seconds.flatten(),
        ])
    
    def action_space(self) -> spaces.Discrete:
        """Return the action space (18 actions for Boxing)."""
        return spaces.Discrete(18)
    
    def observation_space(self) -> spaces.Dict:
        """Return the observation space structure."""
        return spaces.Dict({
            "left_boxer": spaces.Dict({
                "x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
            }),
            "right_boxer": spaces.Dict({
                "x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
            }),
            "score_left": spaces.Box(low=0, high=100, shape=(), dtype=jnp.int32),
            "score_right": spaces.Box(low=0, high=100, shape=(), dtype=jnp.int32),
            "clock_minutes": spaces.Box(low=0, high=2, shape=(), dtype=jnp.int32),
            "clock_seconds": spaces.Box(low=0, high=59, shape=(), dtype=jnp.int32),
        })
    
    def image_space(self) -> spaces.Box:
        """Return the image observation space."""
        return spaces.Box(
            low=0,
            high=255,
            shape=(210, 160, 3),
            dtype=jnp.uint8
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: BoxingState) -> BoxingInfo:
        """Get auxiliary info."""
        return BoxingInfo(
            time=state.step_counter,
            clock_minutes=state.clock_minutes,
            clock_seconds=state.clock_seconds,
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: BoxingState, state: BoxingState) -> float:
        """
        Calculate reward based on score difference.
        
        Positive reward for landing punches, negative for getting hit.
        """
        prev_diff = previous_state.left_boxer_score - previous_state.right_boxer_score
        curr_diff = state.left_boxer_score - state.right_boxer_score
        return (curr_diff - prev_diff).astype(jnp.float32)
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: BoxingState) -> bool:
        """
        Check if game is over.
        
        Game ends when:
        - Either boxer reaches 100 points (KO)
        - Timer reaches 0:00
        - game_state set to 0xFF
        """
        ko_left = jnp.greater_equal(state.left_boxer_score, self.consts.MAX_SCORE)
        ko_right = jnp.greater_equal(state.right_boxer_score, self.consts.MAX_SCORE)
        game_over_flag = jnp.equal(state.game_state, 0xFF)
        timer_expired = jnp.logical_and(
            state.clock_minutes == 0,
            state.clock_seconds == 0
        )
        return jnp.logical_or(
            jnp.logical_or(ko_left, ko_right),
            jnp.logical_or(game_over_flag, timer_expired)
        )


# =============================================================================
# Renderer
# =============================================================================

class BoxingRenderer(JAXGameRenderer):
    """
    Renderer for Boxing game using proper sprite assets.
    
    Uses extracted sprites from the original Atari game including:
    - Background with boxing ring
    - White and black boxer idle sprites
    - Punch animation frames (4 frames per direction per boxer)
    - Digit sprites for score and timer display
    """
    
    def __init__(self, consts: BoxingConstants = None):
        super().__init__(consts)
        self.consts = consts or BoxingConstants()
        self.config = render_utils.RendererConfig(
            game_dimensions=(210, 160),
            channels=3,
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)
        
        # Load sprites from asset config
        final_asset_config = list(self.consts.ASSET_CONFIG)
        
        sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/boxing"
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(final_asset_config, sprite_path)
        
        # Pre-build arrays of punch animation masks for efficient indexed lookup
        # White boxer punch animations (left direction = punching toward left side of screen)
        self.white_punch_left_masks = [
            self.SHAPE_MASKS["white_punch_left_0"],
            self.SHAPE_MASKS["white_punch_left_1"],
            self.SHAPE_MASKS["white_punch_left_2"],
            self.SHAPE_MASKS["white_punch_left_3"],
        ]
        self.white_punch_right_masks = [
            self.SHAPE_MASKS["white_punch_right_0"],
            self.SHAPE_MASKS["white_punch_right_1"],
            self.SHAPE_MASKS["white_punch_right_2"],
            self.SHAPE_MASKS["white_punch_right_3"],
        ]
        # Black boxer punch animations
        self.black_punch_left_masks = [
            self.SHAPE_MASKS["black_punch_left_0"],
            self.SHAPE_MASKS["black_punch_left_1"],
            self.SHAPE_MASKS["black_punch_left_2"],
            self.SHAPE_MASKS["black_punch_left_3"],
        ]
        self.black_punch_right_masks = [
            self.SHAPE_MASKS["black_punch_right_0"],
            self.SHAPE_MASKS["black_punch_right_1"],
            self.SHAPE_MASKS["black_punch_right_2"],
            self.SHAPE_MASKS["black_punch_right_3"],
        ]

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: BoxingState) -> jnp.ndarray:
        """Render the game state to a 210x160x3 RGB image."""
        # Start with background
        raster = self.jr.create_object_raster(self.BACKGROUND)
        
        # Determine facing direction based on relative positions
        # White boxer (left) faces right toward opponent by default
        # When white is to the right of black, white faces left
        white_faces_left = state.left_boxer_x > state.right_boxer_x
        
        # --- Render left boxer (white) ---
        is_punching_left = state.left_boxer_punch_active > 0
        # Convert animation value (0-72) to sprite frame (0-3)
        # animation_value / 8 gives index into sprite table
        punch_frame_left = jnp.clip(state.left_boxer_animation_value // 8, 0, 3).astype(jnp.int32)
        
        # Use last_arm to determine which arm is punching (alternates between punches)
        # 0 = left arm (uses "left" sprites), 1 = right arm (uses "right" sprites)
        white_use_left_arm = state.left_boxer_last_arm == 0
        
        # Select appropriate sprite based on punch state and arm (alternating)
        # Note: We use jax.lax.switch for frame selection within direction
        def render_white_idle(raster):
            return self.jr.render_at(
                raster, state.left_boxer_x, state.left_boxer_y,
                self.SHAPE_MASKS["white_idle"]
            )
        
        def render_white_punch_left(raster):
            # Select frame based on punch_frame_left
            raster = jax.lax.switch(
                punch_frame_left,
                [
                    lambda r: self.jr.render_at(r, state.left_boxer_x, state.left_boxer_y, self.white_punch_left_masks[0]),
                    lambda r: self.jr.render_at(r, state.left_boxer_x, state.left_boxer_y, self.white_punch_left_masks[1]),
                    lambda r: self.jr.render_at(r, state.left_boxer_x, state.left_boxer_y, self.white_punch_left_masks[2]),
                    lambda r: self.jr.render_at(r, state.left_boxer_x, state.left_boxer_y, self.white_punch_left_masks[3]),
                ],
                raster
            )
            return raster
        
        def render_white_punch_right(raster):
            raster = jax.lax.switch(
                punch_frame_left,
                [
                    lambda r: self.jr.render_at(r, state.left_boxer_x, state.left_boxer_y, self.white_punch_right_masks[0]),
                    lambda r: self.jr.render_at(r, state.left_boxer_x, state.left_boxer_y, self.white_punch_right_masks[1]),
                    lambda r: self.jr.render_at(r, state.left_boxer_x, state.left_boxer_y, self.white_punch_right_masks[2]),
                    lambda r: self.jr.render_at(r, state.left_boxer_x, state.left_boxer_y, self.white_punch_right_masks[3]),
                ],
                raster
            )
            return raster
        
        # Render white boxer - use alternating arms based on last_arm
        raster = jax.lax.cond(
            is_punching_left,
            lambda r: jax.lax.cond(
                white_use_left_arm,
                render_white_punch_left,
                render_white_punch_right,
                r
            ),
            render_white_idle,
            raster
        )
        
        # --- Render right boxer (black) ---
        # Black faces left toward white by default
        black_faces_right = state.right_boxer_x < state.left_boxer_x
        is_punching_right = state.right_boxer_punch_active > 0
        # Convert animation value (0-72) to sprite frame (0-3)
        punch_frame_right = jnp.clip(state.right_boxer_animation_value // 8, 0, 3).astype(jnp.int32)
        
        # Use last_arm to determine which arm is punching (alternates between punches)
        black_use_left_arm = state.right_boxer_last_arm == 0
        
        # Black boxer punch sprites extend to the LEFT (arm goes toward player)
        # We need to offset x position to keep body in place
        # Offsets: frame 0=0, frame 1=8, frame 2=17, frame 3=8
        punch_x_offsets = jnp.array([0, 8, 17, 8])
        black_punch_x_offset = punch_x_offsets[punch_frame_right]
        black_punch_x = state.right_boxer_x - black_punch_x_offset
        
        def render_black_idle(raster):
            return self.jr.render_at(
                raster, state.right_boxer_x, state.right_boxer_y,
                self.SHAPE_MASKS["black_idle"]
            )
        
        def render_black_punch_left(raster):
            raster = jax.lax.switch(
                punch_frame_right,
                [
                    lambda r: self.jr.render_at(r, black_punch_x, state.right_boxer_y, self.black_punch_left_masks[0]),
                    lambda r: self.jr.render_at(r, black_punch_x, state.right_boxer_y, self.black_punch_left_masks[1]),
                    lambda r: self.jr.render_at(r, black_punch_x, state.right_boxer_y, self.black_punch_left_masks[2]),
                    lambda r: self.jr.render_at(r, black_punch_x, state.right_boxer_y, self.black_punch_left_masks[3]),
                ],
                raster
            )
            return raster
        
        def render_black_punch_right(raster):
            raster = jax.lax.switch(
                punch_frame_right,
                [
                    lambda r: self.jr.render_at(r, black_punch_x, state.right_boxer_y, self.black_punch_right_masks[0]),
                    lambda r: self.jr.render_at(r, black_punch_x, state.right_boxer_y, self.black_punch_right_masks[1]),
                    lambda r: self.jr.render_at(r, black_punch_x, state.right_boxer_y, self.black_punch_right_masks[2]),
                    lambda r: self.jr.render_at(r, black_punch_x, state.right_boxer_y, self.black_punch_right_masks[3]),
                ],
                raster
            )
            return raster
        
        # Render black boxer - use alternating arms based on last_arm
        raster = jax.lax.cond(
            is_punching_right,
            lambda r: jax.lax.cond(
                black_use_left_arm,
                render_black_punch_left,
                render_black_punch_right,
                r
            ),
            render_black_idle,
            raster
        )
        
        # --- Render HUD (scores and timer) using digit sprites ---
        
        # Left boxer score (white digits) - top left
        white_digit_masks = self.SHAPE_MASKS["digits_white"]
        left_score_digits = self.jr.int_to_digits(
            jnp.clip(state.left_boxer_score, 0, 99), max_digits=2
        )
        raster = self.jr.render_label(
            raster, self.consts.LEFT_SCORE_X, self.consts.SCORE_Y,
            left_score_digits, white_digit_masks, spacing=self.consts.DIGIT_SPACING
        )
        
        # Right boxer score (black digits) - top right
        black_digit_masks = self.SHAPE_MASKS["digits_black"]
        right_score_digits = self.jr.int_to_digits(
            jnp.clip(state.right_boxer_score, 0, 99), max_digits=2
        )
        raster = self.jr.render_label(
            raster, self.consts.RIGHT_SCORE_X, self.consts.SCORE_Y,
            right_score_digits, black_digit_masks, spacing=self.consts.DIGIT_SPACING
        )
        
        # Timer (time digits) - center top
        time_digit_masks = self.SHAPE_MASKS["digits_time"]
        
        # Format timer as M:SS (minutes + 2-digit seconds)
        # First render minutes (single digit)
        minutes_digit = self.jr.int_to_digits(
            jnp.clip(state.clock_minutes, 0, 9), max_digits=1
        )
        raster = self.jr.render_label(
            raster, self.consts.TIMER_X, self.consts.SCORE_Y,
            minutes_digit, time_digit_masks, spacing=0
        )
        
        # Note: Colon would need a separate sprite - for now we skip it
        # and just render seconds after a gap
        seconds_digits = self.jr.int_to_digits(
            jnp.clip(state.clock_seconds, 0, 59), max_digits=2
        )
        raster = self.jr.render_label(
            raster, self.consts.TIMER_X + 12, self.consts.SCORE_Y,
            seconds_digits, time_digit_masks, spacing=self.consts.DIGIT_SPACING
        )
        
        return self.jr.render_from_palette(raster, self.PALETTE)
