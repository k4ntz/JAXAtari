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
    
    # Boxer boundaries from spec
    XMIN_BOXER: int = 30
    XMAX_BOXER: int = 109
    YMIN: int = 3
    YMAX: int = 87
    
    # Boxer dimensions from spec
    H_BOXER: int = 48  # Boxer height (3 sections × 16 pixels)
    
    # Movement speed (fixed for Phase 1)
    MOVE_SPEED: int = 1
    
    # Initial positions from spec
    LEFT_BOXER_START_X: int = 30
    RIGHT_BOXER_START_X: int = 109
    BOXER_START_Y: int = 45  # Centered vertically in play area
    
    # Timer settings (for future phases)
    CLOCK_MINUTES_START: int = 2
    CLOCK_SECONDS_START: int = 0
    FRAMES_PER_SECOND: int = 60  # NTSC
    
    # Scoring
    MAX_SCORE: int = 100  # KO score
    
    # Colors from boxing.asm (NTSC palette approximations)
    # COLOR_LEFT_BOXER = BLACK + 12 = light gray/white
    # COLOR_RIGHT_BOXER = BLACK = black  
    # COLOR_BACKGROUND = DK_GREEN + 6 = green ring
    # COLOR_BOXING_RING = LT_RED + 8 = red ring posts
    BACKGROUND_COLOR: Tuple[int, int, int] = (0, 100, 0)  # Dark green ring
    LEFT_BOXER_COLOR: Tuple[int, int, int] = (236, 236, 236)  # White/light gray
    RIGHT_BOXER_COLOR: Tuple[int, int, int] = (0, 0, 0)  # Black
    RING_COLOR: Tuple[int, int, int] = (200, 72, 72)  # Red ring posts
    
    # Player dot size for Phase 1 MVP (temporary, will be replaced with sprites)
    PLAYER_DOT_SIZE: int = 4


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
    
    # Punch state - placeholder for future phases
    extended_arm_maximum: chex.Array  # 2-element array (one per boxer)
    
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
            
            # Punch state (not extended)
            extended_arm_maximum=jnp.zeros(2, dtype=jnp.int32),
            
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
        Movement is blocked if stunned (future phases).
        """
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
        
        # Apply movement with boundary clamping
        new_x = jnp.clip(
            state.left_boxer_x + dx,
            self.consts.XMIN_BOXER,
            self.consts.XMAX_BOXER
        )
        new_y = jnp.clip(
            state.left_boxer_y + dy,
            self.consts.YMIN,
            self.consts.YMAX
        )
        
        return state._replace(
            left_boxer_x=new_x.astype(jnp.int32),
            left_boxer_y=new_y.astype(jnp.int32),
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: BoxingState, action: chex.Array) -> Tuple[BoxingObservation, BoxingState, float, bool, BoxingInfo]:
        """Execute one game step."""
        # Split PRNG key
        new_state_key, step_key = jax.random.split(state.key)
        previous_state = state
        
        # Update key for this step
        state = state._replace(key=step_key)
        
        # Process player movement
        state = self._player_step(state, action)
        
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
            width=jnp.array(self.consts.PLAYER_DOT_SIZE),
            height=jnp.array(self.consts.PLAYER_DOT_SIZE),
        )
        
        right_boxer = EntityPosition(
            x=state.right_boxer_x,
            y=state.right_boxer_y,
            width=jnp.array(self.consts.PLAYER_DOT_SIZE),
            height=jnp.array(self.consts.PLAYER_DOT_SIZE),
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
        - Timer reaches 0:00 (not implemented in Phase 1)
        """
        ko_left = jnp.greater_equal(state.left_boxer_score, self.consts.MAX_SCORE)
        ko_right = jnp.greater_equal(state.right_boxer_score, self.consts.MAX_SCORE)
        game_over_flag = jnp.equal(state.game_state, 0xFF)
        return jnp.logical_or(jnp.logical_or(ko_left, ko_right), game_over_flag)


# =============================================================================
# Renderer
# =============================================================================

class BoxingRenderer(JAXGameRenderer):
    """
    Renderer for Boxing game.
    
    Phase 1: Renders a simple green background with player dot.
    Future phases will add sprites, ring, and score display.
    """
    
    def __init__(self, consts: BoxingConstants = None):
        super().__init__(consts)
        self.consts = consts or BoxingConstants()
        self.config = render_utils.RendererConfig(
            game_dimensions=(210, 160),
            channels=3,
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)
        
        # Create simple background (solid green for ring)
        bg_color = jnp.array(self.consts.BACKGROUND_COLOR, dtype=jnp.uint8)
        self.background = jnp.tile(
            bg_color.reshape(1, 1, 3),
            (self.consts.HEIGHT, self.consts.WIDTH, 1)
        )
        
        # Create player dot sprite (white square)
        dot_size = self.consts.PLAYER_DOT_SIZE
        dot_color = jnp.array(self.consts.LEFT_BOXER_COLOR, dtype=jnp.uint8)
        self.player_dot = jnp.tile(
            dot_color.reshape(1, 1, 3),
            (dot_size, dot_size, 1)
        )
        
        # Create opponent dot sprite (black square) - for future use
        opponent_color = jnp.array(self.consts.RIGHT_BOXER_COLOR, dtype=jnp.uint8)
        self.opponent_dot = jnp.tile(
            opponent_color.reshape(1, 1, 3),
            (dot_size, dot_size, 1)
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: BoxingState) -> jnp.ndarray:
        """Render the game state to a 210x160x3 RGB image."""
        # Start with background
        image = self.background.copy()
        
        # Draw player dot (left boxer)
        dot_size = self.consts.PLAYER_DOT_SIZE
        x = state.left_boxer_x
        y = state.left_boxer_y
        
        # Clamp coordinates to valid range
        x_start = jnp.clip(x, 0, self.consts.WIDTH - dot_size)
        y_start = jnp.clip(y, 0, self.consts.HEIGHT - dot_size)
        
        # Use dynamic_update_slice to place the dot
        image = jax.lax.dynamic_update_slice(
            image,
            self.player_dot,
            (y_start.astype(jnp.int32), x_start.astype(jnp.int32), 0)
        )
        
        # Draw opponent dot (right boxer) - visible but stationary for now
        x2 = state.right_boxer_x
        y2 = state.right_boxer_y
        x2_start = jnp.clip(x2, 0, self.consts.WIDTH - dot_size)
        y2_start = jnp.clip(y2, 0, self.consts.HEIGHT - dot_size)
        
        image = jax.lax.dynamic_update_slice(
            image,
            self.opponent_dot,
            (y2_start.astype(jnp.int32), x2_start.astype(jnp.int32), 0)
        )
        
        return image.astype(jnp.uint8)
