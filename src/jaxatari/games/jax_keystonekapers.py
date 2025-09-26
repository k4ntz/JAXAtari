"""
JAX-based implementation of KeystoneKapers Atari game.

KeystoneKapers is a multi-floor department store chase game where Officer Kelly pursues a thief.
The player must navigate through 3 floors + roof using elevators and escalators while avoiding
obstacles and collecting items, all within a time limit.

Implementation notes:
- Uses JAX for GPU acceleration and functional programming
- Immutable state management with NamedTuples
- Vectorized collision detection and obstacle spawning
- Level-based difficulty scaling for speed and spawn rates
- Configurable game parameters for easy tuning

Structure:
- Constants: Game parameters and configuration
- State: Complete game state including player, thief, obstacles, timer
- Observation: Filtered state for agent consumption
- Info: Additional metadata for debugging/analysis
- Game Logic: Pure functional game mechanics with JAX optimizations
- Renderer: Sprite-based rendering compatible with JAX/GPU
"""

import os
from functools import partial
from typing import NamedTuple, Tuple, Dict, Any
import jax
import jax.lax
import jax.numpy as jnp
import jax.random as jrandom
import chex

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import JAXGameRenderer
import jaxatari.rendering.jax_rendering_utils as jr
import jaxatari.spaces as spaces


class KeystoneKapersConstants(NamedTuple):
    """Game constants and configuration parameters."""

    # Screen dimensions - Match original Atari Keystone Kapers (250x160)
    TOTAL_SCREEN_WIDTH: int = 160   # Original Atari width
    TOTAL_SCREEN_HEIGHT: int = 250  # Original Atari height (much taller)

    # Game area dimensions (includes minimap within the area)
    GAME_AREA_WIDTH: int = 152      # Slightly smaller for left border
    GAME_AREA_HEIGHT: int = 177     # Reduced to end right after minimap area #@adham: revert to 175 if issues arise

    # Border offsets to match original layout
    GAME_AREA_OFFSET_X: int = 8     # Left border (keep same)
    GAME_AREA_OFFSET_Y: int = 29    # Top border (keep same) #@adham: revert to 30 if issues arise
    # Bottom border will be: 250 - 30 - 175 = 45 pixels (bigger bottom border)

    # Legacy constants for compatibility
    SCREEN_WIDTH: int = 152         # Match game area
    SCREEN_HEIGHT: int = 177        # Match game area #@adham: revert to 175 if issues arise

    # Building structure - 7 sections of horizontal scrolling
    BUILDING_SECTIONS: int = 7
    SECTION_WIDTH: int = 152  # Match game area width to prevent disappearing
    TOTAL_BUILDING_WIDTH: int = 7 * 152  # 1064 pixels total width

    # Floor positions (Y coordinates) - shifted up further to eliminate blue gap above minimap
    FLOOR_1_Y: int = 132  # Ground floor (was 140, shifted up by 5 more to close gap)
    FLOOR_2_Y: int = 100   # Middle floor (was 100, shifted up by 5)
    FLOOR_3_Y: int = 68   # Top floor (was 60, shifted up by 5)
    ROOF_Y: int = 36      # Roof
    FLOOR_HEIGHT: int = 26 #@adham: revert to 20 if issues arise

    # Minimap area configuration (at bottom of game area)
    MINIMAP_HEIGHT: int = 35  # Grey area height (increased from original value 20)
    MINIMAP_COLOR: tuple = (151, 151, 151)  # #979797 in RGB

    # Actual minimap display area (within grey area)
    MINIMAP_DISPLAY_WIDTH: int = 100  # Actual minimap width
    MINIMAP_DISPLAY_HEIGHT: int = 16   # Actual minimap height (increased for better layer visibility)
    MINIMAP_DISPLAY_OFFSET_X: int = 26  # Center within grey area
    MINIMAP_DISPLAY_OFFSET_Y: int = 2   # Vertical offset within grey area (adjusted)

    # Minimap floor layer colors
    MINIMAP_TAN: tuple = (190, 156, 72)    # Tan #be9c48
    MINIMAP_YELLOW: tuple = (207, 175, 92)  # Yellow #cfaf5c
    MINIMAP_DARK_TAN: tuple = (171, 135, 50)  # Dark tan #ab8732
    MINIMAP_GREEN: tuple = (50, 152, 82)   # Green #329852

    # Escalator positions and configuration
    ESCALATOR_1_OFFSET: int = 40   # Left escalator in each section
    ESCALATOR_2_OFFSET: int = 120  # Right escalator in each section
    ESCALATOR_WIDTH: int = 16
    ESCALATOR_HEIGHT: int = 25     # Diagonal length
    ESCALATOR_COLOR: Tuple[int, int, int] = (52, 0, 128)  # Purple #340080
    ESCALATOR_STEP_COLOR: Tuple[int, int, int] = (255, 255, 255)  # White steps
    ESCALATOR_ANIMATION_SPEED: int = 32  # Frames per step animation cycle

    # Three specific escalators in the middle of the last section (absolute positions)
    # ESCALATOR_FLOOR1_X: int = 988   # Floor 1: middle of last section (6*152 + 76 = 988)
    # ESCALATOR_FLOOR2_X: int = 988   # Floor 2: middle of last section
    # ESCALATOR_FLOOR3_X: int = 988   # Floor 3: middle of last section

    # Three specific escalators in the middle of first and last sections (absolute positions)
    ESCALATOR_FLOOR1_X: int = 76    # Floor 1: middle of first section (0 + 76 = 76)
    ESCALATOR_FLOOR2_X: int = 968   # Floor 2: middle of last section (6*152 + 76 = 988)
    ESCALATOR_FLOOR3_X: int = 76    # Floor 3: middle of first section (0 + 76 = 76)

    # Elevator configuration (positioned in middle of entire building)
    ELEVATOR_BUILDING_X: int = 520  # Middle of 1064px building (3.5 * 152) - adjusted for new section width
    ELEVATOR_WIDTH: int = 16
    ELEVATOR_MOVE_TIME: int = 120   # 2 seconds to move between floors
    ELEVATOR_DOOR_TIME: int = 60    # 1 second for doors to open/close
    ELEVATOR_WAIT_TIME: int = 180   # 3 seconds doors stay open

    # Player configuration
    PLAYER_WIDTH: int = 8
    PLAYER_HEIGHT: int = 16
    PLAYER_CROUCH_HEIGHT: int = 8  # Half height when crouching
    PLAYER_SPEED: int = 2
    JUMP_HEIGHT: int = 12
    JUMP_DURATION: int = 20
    JUMP_GRAVITY: float = 0.8

    # Camera/Viewport system for scrolling
    # Camera system - section-based (not smooth following)
    # Camera jumps to show the section the player is currently in

    # Thief configuration
    THIEF_WIDTH: int = 8
    THIEF_HEIGHT: int = 16
    THIEF_BASE_SPEED: float = 0.75  # Slower than player (player speed is 2.0)
    THIEF_SPEED_SCALE: float = 0.08  # 8% increase per level
    THIEF_MAX_SPEED: float = 1.8  # Slightly less than player's max speed

    # Obstacle configurations
    CART_WIDTH: int = 12
    CART_HEIGHT: int = 8
    CART_BASE_SPEED: float = 1.5
    CART_SPEED_SCALE: float = 0.08
    CART_MIN_SPAWN_INTERVAL: float = 1.5  # seconds
    CART_MAX_SPAWN_INTERVAL: float = 3.0

    BALL_WIDTH: int = 6
    BALL_HEIGHT: int = 6
    BALL_BASE_SPEED: float = 2.5  # Slower horizontal speed for more travel time
    BALL_BOUNCE_HEIGHT: int = 3  # Lower bounce height for longer horizontal travel
    BALL_MIN_SPAWN_INTERVAL: float = 0.3  # Earlier spawning - reduced from 0.5
    BALL_MAX_SPAWN_INTERVAL: float = 1.2  # Earlier spawning - reduced from 2.0
    BALL_GRAVITY: float = 0.6  # Lower gravity for slower falling
    BALL_TIME_PENALTY: int = 10  # Time penalty in seconds when hit by ball
    BALL_FREEZE_TIME: int = 24  # Freeze time in frames (~0.4s at 60fps)

    PLANE_WIDTH: int = 16
    PLANE_HEIGHT: int = 8
    PLANE_BASE_SPEED: float = 3.0
    PLANE_SPEED_SCALE: float = 0.12
    PLANE_MIN_SPAWN_INTERVAL: float = 3.0
    PLANE_MAX_SPAWN_INTERVAL: float = 6.0

    # Stationary obstacle configurations
    OBSTACLE_WIDTH: int = 8
    OBSTACLE_HEIGHT: int = 8
    OBSTACLE_TIME_PENALTY: int = 600  # 10 seconds at 60fps
    MAX_STATIONARY_OBSTACLES: int = 4

    # Shopping cart configurations
    SHOPPING_CART_WIDTH: int = 12
    SHOPPING_CART_HEIGHT: int = 8
    SHOPPING_CART_BASE_SPEED: float = 1.8
    SHOPPING_CART_SPEED_SCALE: float = 0.08
    SHOPPING_CART_MIN_SPAWN_INTERVAL: float = 2.0
    SHOPPING_CART_MAX_SPAWN_INTERVAL: float = 4.0
    SHOPPING_CART_TIME_PENALTY: int = 600  # 10 seconds at 60fps
    MAX_SHOPPING_CARTS: int = 3

    # Obstacle types
    OBSTACLE_TYPE_STATIONARY: int = 0
    OBSTACLE_TYPE_SHOPPING_CART: int = 1

    # Collectible configurations
    ITEM_WIDTH: int = 8
    ITEM_HEIGHT: int = 8

    # Timing and scoring
    BASE_TIMER: int = 3600  # 60 seconds at 60fps
    TIMER_REDUCTION_PER_LEVEL: int = 300  # 5 seconds
    COLLISION_PENALTY: int = 300  # 5 seconds
    CATCH_THIEF_POINTS: int = 0  # Base points, actual calculation is time left * 100
    TIME_BONUS_MULTIPLIER: int = 100  # Updated to match spec (time left * 100)
    ITEM_POINTS: int = 50  # Updated to match spec (50 points per collectible)
    JUMP_POINTS: int = 0  # No points for jumping according to spec
    EXTRA_LIFE_THRESHOLD: int = 10000  # Extra life every 10,000 points

    # Difficulty scaling
    OBSTACLE_SPAWN_SCALE: float = 0.12
    MAX_OBSTACLES: int = 8
    MAX_ITEMS: int = 4

    # Item types
    ITEM_TYPE_MONEYBAG: int = 0
    ITEM_TYPE_SUITCASE: int = 1

    # Colors (RGB)
    BACKGROUND_COLOR: Tuple[int, int, int] = (50, 152, 82)  # Green
    PLAYER_COLOR: Tuple[int, int, int] = (0, 255, 0)      # Green
    THIEF_COLOR: Tuple[int, int, int] = (255, 255, 0)     # Yellow
    FLOOR_TAN_COLOR: Tuple[int, int, int] = (190, 156, 72)  # Tan (#be9c48)
    FLOOR_YELLOW_COLOR: Tuple[int, int, int] = (207, 175, 92)  # Yellow (#cfaf5c)
    ESCALATOR_COLOR: Tuple[int, int, int] = (128, 128, 128)  # Gray
    ELEVATOR_COLOR: Tuple[int, int, int] = (64, 64, 64)   # Dark gray
    CART_COLOR: Tuple[int, int, int] = (255, 0, 0)        # Red
    BALL_COLOR: Tuple[int, int, int] = (255, 255, 255)    # White
    PLANE_COLOR: Tuple[int, int, int] = (255, 165, 0)     # Orange
    ITEM_COLOR: Tuple[int, int, int] = (255, 0, 255)      # Magenta


class PlayerState(NamedTuple):
    """Player state including position, movement, and actions."""
    x: chex.Array           # Absolute X position in building (0 to TOTAL_BUILDING_WIDTH)
    y: chex.Array           # Y position on screen
    floor: chex.Array       # Current floor (0=ground, 1=middle, 2=top, 3=roof)
    vel_x: chex.Array
    vel_y: chex.Array
    is_jumping: chex.Array
    is_crouching: chex.Array
    jump_timer: chex.Array
    jump_start_y: chex.Array  # Y position when jump started
    freeze_timer: chex.Array  # Freeze timer after ball collision
    # Escalator state
    on_escalator: chex.Array    # Which escalator (0=none, 1=floor1, 2=floor2, 3=floor3)
    escalator_progress: chex.Array  # Progress along escalator (0.0 to 1.0)
    # Elevator state
    is_on_elevator: chex.Array
    in_elevator: chex.Array   # Actually inside elevator car


class ThiefState(NamedTuple):
    """Thief state and AI behavior."""
    x: chex.Array
    y: chex.Array
    floor: chex.Array
    speed: chex.Array
    direction: chex.Array  # 1 for right, -1 for left
    escaped: chex.Array


class ObstacleState(NamedTuple):
    """State for multiple obstacles of each type."""
    # Shopping carts
    cart_x: chex.Array          # Shape: (MAX_OBSTACLES,)
    cart_y: chex.Array
    cart_active: chex.Array
    cart_speed: chex.Array
    cart_spawn_timer: chex.Array

    # Bouncing balls
    ball_x: chex.Array
    ball_y: chex.Array
    ball_active: chex.Array
    ball_vel_x: chex.Array
    ball_vel_y: chex.Array
    ball_spawn_timer: chex.Array
    ball_floor: chex.Array      # Which floor (0-2 for floors 1-3)
    ball_is_bouncing: chex.Array # True when ball is touching floor (for sprite selection)

    # Toy planes
    plane_x: chex.Array
    plane_y: chex.Array
    plane_active: chex.Array
    plane_speed: chex.Array
    plane_spawn_timer: chex.Array


class ElevatorState(NamedTuple):
    """Elevator state machine."""
    floor: chex.Array       # Current floor (0=ground, 1=middle, 2=top)
    state: chex.Array       # 0=IdleOpen, 1=Closing, 2=Moving, 3=Opening
    timer: chex.Array       # Timer for current state
    target_floor: chex.Array # Target floor when moving
    has_player: chex.Array  # Whether player is inside elevator


class GameState(NamedTuple):
    """Complete game state."""
    player: PlayerState
    thief: ThiefState
    obstacles: ObstacleState
    elevator: ElevatorState

    # Camera system for scrolling
    camera_x: chex.Array    # Camera X position in building coordinates

    # Game state
    score: chex.Array
    lives: chex.Array
    level: chex.Array
    timer: chex.Array
    step_counter: chex.Array

    # Escalator animation frame
    escalator_frame: chex.Array  # Animation frame for moving steps

    # Items and collectibles
    item_x: chex.Array          # Shape: (MAX_ITEMS,)
    item_y: chex.Array
    item_active: chex.Array
    item_type: chex.Array

    # New obstacles
    stationary_obstacle_x: chex.Array      # Shape: (MAX_STATIONARY_OBSTACLES,)
    stationary_obstacle_y: chex.Array
    stationary_obstacle_active: chex.Array
    stationary_obstacle_floor: chex.Array

    # Shopping carts
    shopping_cart_x: chex.Array            # Shape: (MAX_SHOPPING_CARTS,)
    shopping_cart_y: chex.Array
    shopping_cart_active: chex.Array
    shopping_cart_floor: chex.Array
    shopping_cart_direction: chex.Array    # 1 for right, -1 for left
    shopping_cart_spawn_timer: chex.Array

    # Game flags
    game_over: chex.Array
    thief_caught: chex.Array
    level_complete: chex.Array


class KeystoneKapersObservation(NamedTuple):
    """Observation space for the agent."""
    player_x: chex.Array
    player_y: chex.Array
    player_floor: chex.Array
    thief_x: chex.Array
    thief_y: chex.Array
    thief_floor: chex.Array

    # Obstacles (vectorized)
    obstacle_positions: chex.Array  # Shape: (MAX_OBSTACLES*3, 3) for [x, y, type]

    # Items
    item_positions: chex.Array      # Shape: (MAX_ITEMS, 3) for [x, y, type]

    # Game state
    score: chex.Array
    timer: chex.Array
    level: chex.Array
    lives: chex.Array

    # Elevator state
    elevator_position: chex.Array
    elevator_is_open: chex.Array


class KeystoneKapersInfo(NamedTuple):
    """Additional game information."""
    step_count: chex.Array
    thief_caught: chex.Array
    thief_escaped: chex.Array
    items_collected: chex.Array
    obstacles_hit: chex.Array
    all_rewards: chex.Array


class JaxKeystoneKapers(JaxEnvironment[GameState, KeystoneKapersObservation, KeystoneKapersInfo, KeystoneKapersConstants]):
    """JAX-based KeystoneKapers environment implementation."""

    def __init__(self, consts: KeystoneKapersConstants = None, frameskip: int = 1, reward_funcs: list[callable] = None):
        super().__init__(consts or KeystoneKapersConstants())
        self.frameskip = frameskip
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs

        # Action set for KeystoneKapers
        self.action_set = [
            Action.NOOP,
            Action.UP,
            Action.DOWN,
            Action.LEFT,
            Action.RIGHT,
            Action.FIRE,      # Jump
            Action.UPLEFT,
            Action.UPRIGHT,
            Action.DOWNLEFT,
            Action.DOWNRIGHT,
            Action.LEFTFIRE,  # Jump left
            Action.RIGHTFIRE, # Jump right
        ]

        self.obs_size = 8 + self.consts.MAX_OBSTACLES * 3 * 3 + self.consts.MAX_ITEMS * 3 + 6
        self.renderer = KeystoneKapersRenderer(self.consts)

    @partial(jax.jit, static_argnums=(0,))
    def _floor_y_position(self, floor: chex.Array) -> chex.Array:
        """Get Y position for a given floor number."""
        return jax.lax.select(
            floor == 0, self.consts.FLOOR_1_Y + 2,  # Ground floor, lowered by 2 pixels
            jax.lax.select(
                floor == 1, self.consts.FLOOR_2_Y + 2,  # Middle floor, lowered by 2 pixels
                jax.lax.select(
                    floor == 2, self.consts.FLOOR_3_Y + 2,  # Top floor, lowered by 2 pixels
                    self.consts.ROOF_Y + 2  # Roof, lowered by 2 pixels
                )
            )
        )

    @partial(jax.jit, static_argnums=(0,))
    def _is_on_escalator(self, x: chex.Array, floor: chex.Array) -> chex.Array:
        """Check if position is on an escalator."""
        # Convert absolute building position to section-relative position
        section_x = x % self.consts.SECTION_WIDTH

        on_esc1 = jnp.logical_and(
            section_x >= self.consts.ESCALATOR_1_OFFSET,
            section_x <= self.consts.ESCALATOR_1_OFFSET + self.consts.ESCALATOR_WIDTH
        )
        on_esc2 = jnp.logical_and(
            section_x >= self.consts.ESCALATOR_2_OFFSET,
            section_x <= self.consts.ESCALATOR_2_OFFSET + self.consts.ESCALATOR_WIDTH
        )
        # Escalators don't connect to roof
        has_escalator = floor < 3
        return jnp.logical_and(jnp.logical_or(on_esc1, on_esc2), has_escalator)

    @partial(jax.jit, static_argnums=(0,))
    def _is_on_elevator(self, x: chex.Array, floor: chex.Array, elevator: ElevatorState) -> chex.Array:
        """Check if position is on elevator at current floor."""
        on_elevator_x = jnp.logical_and(
            x >= self.consts.ELEVATOR_BUILDING_X,
            x <= self.consts.ELEVATOR_BUILDING_X + self.consts.ELEVATOR_WIDTH
        )
        elevator_at_floor = elevator.floor == floor
        elevator_doors_open = elevator.state == 0  # IdleOpen state
        return jnp.logical_and(
            jnp.logical_and(on_elevator_x, elevator_at_floor),
            elevator_doors_open
        )

    @partial(jax.jit, static_argnums=(0,))
    def _entities_collide(self, x1: chex.Array, y1: chex.Array, w1: int, h1: int,
                         x2: chex.Array, y2: chex.Array, w2: int, h2: int) -> chex.Array:
        """Check collision between two rectangular entities."""
        return jnp.logical_and(
            jnp.logical_and(x1 < x2 + w2, x1 + w1 > x2),
            jnp.logical_and(y1 < y2 + h2, y1 + h1 > y2)
        )

    @partial(jax.jit, static_argnums=(0,))
    def _update_player(self, state: GameState, action: chex.Array) -> PlayerState:
        """Update player state based on action."""
        player = state.player

        # If player is frozen from ball collision, only decrease freeze timer
        is_frozen = player.freeze_timer > 0
        new_freeze_timer = jnp.maximum(0, player.freeze_timer - 1)

        # Handle both scalar actions (from JAX environment) and array actions
        # Convert scalar action to movement flags
        action_int = jnp.asarray(action, dtype=jnp.int32)

        # Extract action components - handle both scalar and array inputs
        # Disable all actions if player is frozen
        move_left = jnp.logical_and(
            jnp.logical_not(is_frozen),
            jnp.logical_or(
                jnp.logical_or(action_int == Action.LEFT, action_int == Action.UPLEFT),
                jnp.logical_or(action_int == Action.DOWNLEFT, action_int == Action.LEFTFIRE)
            )
        )
        move_right = jnp.logical_and(
            jnp.logical_not(is_frozen),
            jnp.logical_or(
                jnp.logical_or(action_int == Action.RIGHT, action_int == Action.UPRIGHT),
                jnp.logical_or(action_int == Action.DOWNRIGHT, action_int == Action.RIGHTFIRE)
            )
        )
        move_up = jnp.logical_and(
            jnp.logical_not(is_frozen),
            jnp.logical_or(
                jnp.logical_or(action_int == Action.UP, action_int == Action.UPLEFT),
                action_int == Action.UPRIGHT
            )
        )
        move_down = jnp.logical_and(
            jnp.logical_not(is_frozen),
            jnp.logical_or(
                jnp.logical_or(action_int == Action.DOWN, action_int == Action.DOWNLEFT),
                action_int == Action.DOWNRIGHT
            )
        )
        jump = jnp.logical_and(
            jnp.logical_not(is_frozen),
            jnp.logical_or(
                jnp.logical_or(action_int == Action.FIRE, action_int == Action.LEFTFIRE),
                action_int == Action.RIGHTFIRE
            )
        )
        crouch = move_down  # Crouch is triggered by down input when not moving vertically

        # Horizontal movement with building traversal
        vel_x = jnp.where(
            move_left, -self.consts.PLAYER_SPEED,
            jnp.where(move_right, self.consts.PLAYER_SPEED, 0)
        )

        # Update X position - simple boundary checking
        new_x = player.x + vel_x
        new_x = jnp.clip(new_x, 0, self.consts.TOTAL_BUILDING_WIDTH - self.consts.PLAYER_WIDTH).astype(jnp.int32)

        # Jumping mechanics with gravity
        start_jump = jnp.logical_and(jump, jnp.logical_not(player.is_jumping))
        continue_jump = jnp.logical_and(player.is_jumping, player.jump_timer > 0)

        # Jump physics - parabolic arc (keep internal calculations in float, but cast final Y to int32)
        jump_progress = (self.consts.JUMP_DURATION - player.jump_timer) / self.consts.JUMP_DURATION
        jump_height = self.consts.JUMP_HEIGHT * jnp.sin(jnp.pi * jump_progress)

        new_jump_timer = jnp.where(
            start_jump, self.consts.JUMP_DURATION,
            jnp.where(continue_jump, player.jump_timer - 1, 0)
        )
        new_is_jumping = new_jump_timer > 0

        new_jump_start_y = jnp.where(
            start_jump, player.y,
            player.jump_start_y
        )

        # Y position with jumping (cast to int32 to maintain dtype consistency)
        floor_y = self._floor_y_position(player.floor)
        jump_y = jnp.where(
            new_is_jumping,
            (new_jump_start_y - jump_height).astype(jnp.int32),
            floor_y
        ).astype(jnp.int32)

        # Crouching (only when not jumping and not in elevator)
        can_crouch = jnp.logical_and(
            jnp.logical_not(new_is_jumping),
            jnp.logical_not(player.in_elevator)
        )
        new_is_crouching = jnp.logical_and(crouch, can_crouch)

        # Elevator interaction
        on_elevator = self._is_on_elevator(new_x, player.floor, state.elevator)

        # Enter elevator only if player presses UP and doors are open
        wants_elevator = move_up  # Only UP to enter
        elevator_available = jnp.logical_and(state.elevator.state == 0, on_elevator)  # IdleOpen and on platform
        enter_elevator = jnp.logical_and(
            jnp.logical_and(wants_elevator, elevator_available),
            jnp.logical_and(jnp.logical_not(player.in_elevator), jnp.logical_not(new_is_jumping))
        )
        new_in_elevator = jnp.logical_or(player.in_elevator, enter_elevator)

        # Exit elevator if doors are open and player presses DOWN
        exit_elevator = jnp.logical_and(
            jnp.logical_and(player.in_elevator, state.elevator.state == 0),  # IdleOpen
            move_down
        )
        new_in_elevator = jnp.logical_and(new_in_elevator, jnp.logical_not(exit_elevator))

        # If in elevator, follow elevator position
        elevator_x = self.consts.ELEVATOR_BUILDING_X + self.consts.ELEVATOR_WIDTH // 2

        # Escalator system - three specific escalators
        current_floor = player.floor

        # Floor 1 escalator: leftmost, going up diagonally left (use absolute building coordinates)
        on_escalator_1 = jnp.logical_and(
            jnp.logical_and(current_floor == 0, new_x >= self.consts.ESCALATOR_FLOOR1_X - 5),
            new_x <= self.consts.ESCALATOR_FLOOR1_X + self.consts.ESCALATOR_WIDTH + 5
        )

        # Floor 2 escalator: rightmost, going up diagonally right (use absolute building coordinates)
        on_escalator_2 = jnp.logical_and(
            jnp.logical_and(current_floor == 1, new_x >= self.consts.ESCALATOR_FLOOR2_X - 5),
            new_x <= self.consts.ESCALATOR_FLOOR2_X + self.consts.ESCALATOR_WIDTH + 5
        )

        # Floor 3 escalator: leftmost, going up diagonally left (use absolute building coordinates)
        on_escalator_3 = jnp.logical_and(
            jnp.logical_and(current_floor == 2, new_x >= self.consts.ESCALATOR_FLOOR3_X - 5),
            new_x <= self.consts.ESCALATOR_FLOOR3_X + self.consts.ESCALATOR_WIDTH + 5
        )

        # Determine if player is starting escalator ride
        starting_escalator = jnp.logical_and(
            player.on_escalator == 0,  # Not already on escalator
            jnp.logical_or(jnp.logical_or(on_escalator_1, on_escalator_2), on_escalator_3)
        )

        # Determine escalator number (1, 2, 3, or 0 for none)
        new_escalator_num = jnp.where(
            on_escalator_1, 1,
            jnp.where(on_escalator_2, 2, jnp.where(on_escalator_3, 3, 0))
        )

        # Update escalator state
        new_on_escalator = jnp.where(
            starting_escalator, new_escalator_num,
            jnp.where(player.escalator_progress >= 1.0, 0, player.on_escalator)  # Reset when complete
        )

        # Escalator progress and movement speed
        escalator_speed = 0.05  # Speed of escalator movement
        new_escalator_progress = jnp.where(
            new_on_escalator > 0,
            jnp.minimum(player.escalator_progress + escalator_speed, 1.0),
            0.0
        )

        # When on escalator, override player movement
        escalator_active = new_on_escalator > 0

        # Calculate escalator diagonal movement (horizontal and vertical)
        escalator_start_x = jnp.where(
            new_on_escalator == 1, self.consts.ESCALATOR_FLOOR1_X,
            jnp.where(new_on_escalator == 2, self.consts.ESCALATOR_FLOOR2_X,
                     jnp.where(new_on_escalator == 3, self.consts.ESCALATOR_FLOOR3_X, new_x))
        )

        # Move horizontally along escalator (different directions for different escalators)
        # Escalators 1 & 3 (floors 1 & 3): right-to-left movement (negative)
        # Escalator 2 (floor 2): left-to-right movement (positive)
        escalator_x_direction = jnp.where(
            jnp.logical_or(new_on_escalator == 1, new_on_escalator == 3), -1.0,  # Right-to-left
            1.0  # Left-to-right for escalator 2
        )
        escalator_x_offset = new_escalator_progress * (self.consts.ESCALATOR_WIDTH) * escalator_x_direction
        escalator_x = escalator_start_x + escalator_x_offset

        # Final X position: elevator overrides escalator, escalator overrides normal movement
        final_x = jnp.where(
            new_in_elevator, elevator_x,
            jnp.where(escalator_active, escalator_x, new_x)
        ).astype(jnp.int32)

        # Calculate escalator movement
        escalator_start_floor = jnp.where(
            new_on_escalator == 1, 0,  # Floor 1 escalator starts at ground
            jnp.where(new_on_escalator == 2, 1, 2)  # Floor 2->3, Floor 3->roof
        )
        escalator_end_floor = escalator_start_floor + 1

        # Interpolate position during escalator ride
        escalator_floor = escalator_start_floor + new_escalator_progress

        # Override movement when on escalator
        escalator_up = jnp.logical_and(escalator_active, new_escalator_progress >= 1.0)
        escalator_down = False  # These escalators only go up

        # Elevator floor changes (when elevator moves)
        elevator_floor_change = jnp.logical_and(
            new_in_elevator,
            state.elevator.floor != player.floor
        )

        # Update floor
        new_floor = jax.lax.select(
            elevator_floor_change, state.elevator.floor,
            jax.lax.select(
                escalator_up, current_floor + 1,
                jax.lax.select(escalator_down, current_floor - 1, current_floor)
            )
        )

        # Update Y position based on final floor and escalator movement
        escalator_start_y = jnp.where(
            new_on_escalator == 1, self.consts.FLOOR_1_Y,
            jnp.where(new_on_escalator == 2, self.consts.FLOOR_2_Y,
                     jnp.where(new_on_escalator == 3, self.consts.FLOOR_3_Y, 0))
        )
        escalator_end_y = jnp.where(
            new_on_escalator == 1, self.consts.FLOOR_2_Y,
            jnp.where(new_on_escalator == 2, self.consts.FLOOR_3_Y,
                     jnp.where(new_on_escalator == 3, self.consts.ROOF_Y, 0))
        )

        # Interpolate Y position along escalator diagonal
        escalator_y = escalator_start_y + (escalator_end_y - escalator_start_y) * new_escalator_progress

        final_y = jnp.where(
            new_is_jumping, jump_y,
            jnp.where(escalator_active, escalator_y, self._floor_y_position(new_floor))
        ).astype(jnp.int32)

        return PlayerState(
            x=final_x,
            y=final_y,
            floor=new_floor,
            vel_x=vel_x,
            vel_y=0,
            is_jumping=new_is_jumping,
            is_crouching=new_is_crouching,
            jump_timer=new_jump_timer,
            jump_start_y=new_jump_start_y,
            freeze_timer=new_freeze_timer,
            on_escalator=new_on_escalator,
            escalator_progress=new_escalator_progress,
            is_on_elevator=state.player.is_on_elevator,
            in_elevator=new_in_elevator
        )
        jump_timer = jax.lax.select(
            start_jump, self.consts.JUMP_DURATION,
            jax.lax.select(
                player.is_jumping, jnp.maximum(player.jump_timer - 1, 0),
                0
            )
        )
        is_jumping = jump_timer > 0

        # Vertical movement (escalator/elevator)
        current_floor = player.floor

        # Check escalator usage (DISABLED - escalators not implemented)
        on_escalator = False
        can_use_escalator = False

        escalator_up = False
        escalator_down = False

        # Check elevator usage
        on_elevator = self._is_on_elevator(
            new_x, current_floor, state.elevator.position, state.elevator.is_open
        )
        can_use_elevator = jnp.logical_and(on_elevator, jnp.logical_not(is_jumping))

        elevator_up = jnp.logical_and(
            jnp.logical_and(can_use_elevator, move_up),
            current_floor < 3  # Can go to roof via elevator
        )
        elevator_down = jnp.logical_and(
            jnp.logical_and(can_use_elevator, move_down),
            current_floor > 0
        )

        # Update floor
        new_floor = jax.lax.select(
            jnp.logical_or(escalator_up, elevator_up), current_floor + 1,
            jax.lax.select(
                jnp.logical_or(escalator_down, elevator_down), current_floor - 1,
                current_floor
            )
        )

        # Update Y position based on floor
        new_y = self._floor_y_position(new_floor)

        return PlayerState(
            x=new_x.astype(jnp.int32),
            y=new_y.astype(jnp.int32),
            floor=new_floor,
            vel_x=vel_x,
            vel_y=0,  # Simplified for now
            is_jumping=is_jumping,
            is_crouching=state.player.is_crouching,
            jump_timer=jump_timer,
            jump_start_y=state.player.jump_start_y,
            freeze_timer=new_freeze_timer,
            on_escalator=jnp.array(0),
            escalator_progress=jnp.array(0.0),
            is_on_elevator=on_elevator,
            in_elevator=state.player.in_elevator
        )

    @partial(jax.jit, static_argnums=(0,))
    def _update_thief(self, state: GameState) -> ThiefState:
        """Update thief state with AI behavior for building traversal."""
        thief = state.thief

        # Level-scaled speed (use integer speed to ensure movement)
        # Convert to integer pixels per frame for consistent movement
        level_speed_float = self.consts.THIEF_BASE_SPEED * (
            1.0 + self.consts.THIEF_SPEED_SCALE * state.level
        )
        level_speed_float = jnp.minimum(level_speed_float, self.consts.THIEF_MAX_SPEED)

        # Use a counter to accumulate fractional movement
        # For every frame, add the float speed to an accumulator
        # When accumulator reaches 1.0 or more, move 1 pixel
        # Store fractional part for next frame
        pixel_move = jnp.floor(level_speed_float).astype(jnp.int32)

        # Always move at least 1 pixel every few frames
        # For very slow speeds below 1, move 1 pixel every N frames
        # where N is inverse of speed
        move_this_frame = ((state.step_counter % jnp.maximum(1, jnp.floor(1.0/level_speed_float))) == 0)
        final_move = jnp.maximum(pixel_move, jnp.where(move_this_frame, 1, 0))

        # For debugging
        level_speed = level_speed_float

        # Thief AI: Real game behavior - moves horizontally, teleports up at building edges
        # Pattern: right->up->left->up->repeat (no escalator usage)

        # Move horizontally in current direction - use integer movement
        new_x = thief.x + final_move * thief.direction

        # Check if reached building edges
        hit_left_edge = new_x <= 0
        hit_right_edge = new_x >= self.consts.TOTAL_BUILDING_WIDTH - self.consts.THIEF_WIDTH
        hit_edge = jnp.logical_or(hit_left_edge, hit_right_edge)

        # Check escape condition BEFORE updating floor (if already on roof and hit edge)
        escaped = jnp.logical_and(thief.floor >= 3, hit_edge)

        # When hitting edge: teleport up one floor and reverse direction (unless escaping)
        new_floor = jnp.where(
            jnp.logical_and(hit_edge, jnp.logical_not(escaped)),
            jnp.minimum(thief.floor + 1, 3),  # Go up one floor, max is roof (floor 3)
            thief.floor
        )

        # Reverse direction when hitting edge (unless escaping)
        new_direction = jnp.where(
            jnp.logical_and(hit_edge, jnp.logical_not(escaped)),
            -thief.direction,  # Reverse direction
            thief.direction
        )

        # Clamp position within building boundaries
        new_x = jnp.clip(new_x, 0, self.consts.TOTAL_BUILDING_WIDTH - self.consts.THIEF_WIDTH).astype(jnp.int32)

        # Update Y position based on floor
        new_y = self._floor_y_position(new_floor).astype(jnp.int32)

        return ThiefState(
            x=new_x,
            y=new_y,
            floor=new_floor,
            speed=level_speed_float,  # Store the float speed for next frame calculation
            direction=new_direction,
            escaped=escaped
        )

    @partial(jax.jit, static_argnums=(0,))
    def _update_elevator(self, state: GameState, action: chex.Array) -> ElevatorState:
        """Update autonomous elevator state machine that cycles between floors automatically."""
        elevator = state.elevator
        player = state.player

        # Elevator state machine:
        # 0 = IdleOpen (doors open, waiting)
        # 1 = Closing (doors closing)
        # 2 = Moving (between floors)
        # 3 = Opening (doors opening at destination)

        # Autonomous elevator logic - move one floor at a time in sequence 0->1->2->1->0
        # Use step counter to determine which phase of the cycle we're in

        at_target = elevator.floor == elevator.target_floor
        doors_open = elevator.state == 0  # IdleOpen
        ready_to_move = jnp.logical_and(at_target, doors_open)

        # Simple cycle using step counter to avoid direction confusion
        # Each complete cycle takes about 20 seconds (4 moves * 5 seconds each)
        cycle_step = (state.step_counter // 300) % 4  # 5 second phases

        # Define the sequence: step 0->1, step 1->2, step 2->1, step 3->0
        target_sequence = jnp.array([1, 2, 1, 0])
        time_based_target = target_sequence[cycle_step]

        # Only change target when ready to move and target would be adjacent
        next_target = jnp.where(
            ready_to_move,
            jnp.where(
                jnp.abs(time_based_target - elevator.floor) == 1,  # Adjacent floor
                time_based_target,
                jnp.where(
                    time_based_target > elevator.floor, elevator.floor + 1,  # Move up one
                    elevator.floor - 1  # Move down one
                )
            ),
            elevator.target_floor  # Keep current target if not ready
        )

        auto_target_floor = next_target

        # Always want to move (autonomous operation)
        elevator_wants_move = True

        # Determine target floor
        target_floor = auto_target_floor

        # State transitions with automatic timing
        new_timer = elevator.timer + 1

        # IdleOpen -> Closing (after waiting time)
        close_doors = jnp.logical_and(
            elevator.state == 0,
            elevator.timer >= self.consts.ELEVATOR_DOOR_TIME
        )

        # Closing -> Moving (after door close time)
        start_moving = jnp.logical_and(
            elevator.state == 1,
            elevator.timer >= self.consts.ELEVATOR_DOOR_TIME
        )

        # Moving -> Opening (after move time)
        arrive_at_floor = jnp.logical_and(
            elevator.state == 2,
            elevator.timer >= self.consts.ELEVATOR_MOVE_TIME
        )

        # Opening -> IdleOpen (after door open time)
        finish_opening = jnp.logical_and(
            elevator.state == 3,
            elevator.timer >= self.consts.ELEVATOR_DOOR_TIME
        )

        # Update state - autonomous operation, no idle timeout
        new_state = jax.lax.select(
            close_doors, 1,  # Start closing
            jax.lax.select(
                start_moving, 2,  # Start moving
                jax.lax.select(
                    arrive_at_floor, 3,  # Start opening
                    jax.lax.select(
                        finish_opening, 0,  # Back to idle open
                        elevator.state  # No change
                    )
                )
            )
        )

        # Reset timer on state changes
        new_timer = jax.lax.select(
            new_state != elevator.state, 0, new_timer
        )

        # Update floor when moving completes
        new_floor = jax.lax.select(
            arrive_at_floor, elevator.target_floor, elevator.floor
        )

        # Update target floor when starting to move
        new_target_floor = jax.lax.select(
            close_doors, target_floor, elevator.target_floor
        )

        # Update player presence
        new_has_player = player.in_elevator

        return ElevatorState(
            floor=new_floor,
            state=new_state,
            timer=new_timer,
            target_floor=new_target_floor,
            has_player=new_has_player
        )

    @partial(jax.jit, static_argnums=(0,))
    def _spawn_obstacle(self, key: chex.PRNGKey, state: GameState, obstacle_type: int) -> ObstacleState:
        """Spawn a new obstacle of given type."""
        obstacles = state.obstacles

        # Level-based spawn rate scaling
        spawn_rate_scale = 1.0 + self.consts.OBSTACLE_SPAWN_SCALE * state.level

        def spawn_cart():
            # Find first inactive slot for shopping cart
            inactive_mask = jnp.logical_not(obstacles.cart_active)
            spawn_idx = jnp.argmax(inactive_mask)
            should_spawn = inactive_mask[spawn_idx]

            # Random spawn parameters
            spawn_x = jrandom.uniform(key, (), minval=0, maxval=self.consts.SCREEN_WIDTH - self.consts.CART_WIDTH).astype(jnp.int32)
            spawn_floor = jrandom.randint(key, (), 0, 3)  # Floors 1-3
            spawn_y = self._floor_y_position(spawn_floor).astype(jnp.int32)
            spawn_speed = self.consts.CART_BASE_SPEED * (1.0 + self.consts.CART_SPEED_SCALE * state.level)

            new_cart_x = obstacles.cart_x.at[spawn_idx].set(
                jax.lax.select(should_spawn, spawn_x, obstacles.cart_x[spawn_idx])
            )
            new_cart_y = obstacles.cart_y.at[spawn_idx].set(
                jax.lax.select(should_spawn, spawn_y, obstacles.cart_y[spawn_idx])
            )
            new_cart_active = obstacles.cart_active.at[spawn_idx].set(
                jax.lax.select(should_spawn, True, obstacles.cart_active[spawn_idx])
            )
            new_cart_speed = obstacles.cart_speed.at[spawn_idx].set(
                jax.lax.select(should_spawn, spawn_speed, obstacles.cart_speed[spawn_idx])
            )

            # Return complete ObstacleState with all fields to match spawn_ball return type
            return ObstacleState(
                cart_x=new_cart_x,
                cart_y=new_cart_y,
                cart_active=new_cart_active,
                cart_speed=new_cart_speed,
                cart_spawn_timer=obstacles.cart_spawn_timer,
                ball_x=obstacles.ball_x,
                ball_y=obstacles.ball_y,
                ball_active=obstacles.ball_active,
                ball_vel_x=obstacles.ball_vel_x,
                ball_vel_y=obstacles.ball_vel_y,
                ball_spawn_timer=obstacles.ball_spawn_timer,
                ball_floor=obstacles.ball_floor,
                ball_is_bouncing=obstacles.ball_is_bouncing,
                plane_x=obstacles.plane_x,
                plane_y=obstacles.plane_y,
                plane_active=obstacles.plane_active,
                plane_speed=obstacles.plane_speed,
                plane_spawn_timer=obstacles.plane_spawn_timer
            )

        def spawn_ball():
            # Find first inactive slot for bouncing ball
            inactive_mask = jnp.logical_not(obstacles.ball_active)
            spawn_idx = jnp.argmax(inactive_mask)
            can_spawn_slot = inactive_mask[spawn_idx]

            # Ball spawns only on the floor the player is currently on
            player_y = state.player.y

            # Determine which floor the player is on based on Y position
            # Floor 1 (ground): around FLOOR_1_Y
            # Floor 3 (top): around FLOOR_3_Y (skip floor 2 as specified)
            on_floor_1 = jnp.abs(player_y - self.consts.FLOOR_1_Y) <= 20  # Within 20 pixels
            on_floor_3 = jnp.abs(player_y - self.consts.FLOOR_3_Y) <= 20  # Within 20 pixels

            # Default to floor 1 if player is not clearly on floor 3
            spawn_floor = jnp.where(on_floor_3, 2, 0)  # Floor 3 index = 2, Floor 1 index = 0

            floor_y = jnp.where(
                spawn_floor == 0,
                self.consts.FLOOR_1_Y + 2,  # Ground floor (with +2 offset like player)
                self.consts.FLOOR_3_Y + 2   # Top floor (with +2 offset like player)
            )

            # Get current player section to spawn ball in current screen area
            current_section = state.player.x // self.consts.SCREEN_WIDTH

            # Calculate elevator section (middle section of 7 total sections)
            elevator_section = self.consts.ELEVATOR_BUILDING_X // self.consts.SCREEN_WIDTH

            # Exception sections where balls should NOT spawn:
            # - First section (0) and last section (6)
            # - Elevator section (around section 3)
            total_sections = 7  # Based on TOTAL_BUILDING_WIDTH / SCREEN_WIDTH
            is_first_section = current_section == 0
            is_last_section = current_section == (total_sections - 1)
            is_elevator_section = current_section == elevator_section

            # Don't spawn if in any exception section
            in_exception_section = jnp.logical_or(
                jnp.logical_or(is_first_section, is_last_section),
                is_elevator_section
            )

            # Only spawn if we have a slot AND not in exception section
            should_spawn = jnp.logical_and(can_spawn_slot, jnp.logical_not(in_exception_section))
            section_start_x = current_section * self.consts.SCREEN_WIDTH

            # Spawn from left side only (as balls should come from the direction of running)
            spawn_x = section_start_x + 10  # Always start from left side of current section

            # Always move right (direction of running)
            spawn_vel_x = jnp.array(int(self.consts.BALL_BASE_SPEED), dtype=jnp.int32)

            spawn_vel_y = jnp.array(0, dtype=jnp.int32)  # Start on floor with no vertical velocity

            new_ball_x = obstacles.ball_x.at[spawn_idx].set(
                jax.lax.select(should_spawn, spawn_x, obstacles.ball_x[spawn_idx])
            )
            new_ball_y = obstacles.ball_y.at[spawn_idx].set(
                jax.lax.select(should_spawn, floor_y.astype(jnp.int32), obstacles.ball_y[spawn_idx])
            )
            new_ball_active = obstacles.ball_active.at[spawn_idx].set(
                jax.lax.select(should_spawn, True, obstacles.ball_active[spawn_idx])
            )
            new_ball_vel_x = obstacles.ball_vel_x.at[spawn_idx].set(
                jax.lax.select(should_spawn, spawn_vel_x, obstacles.ball_vel_x[spawn_idx])
            )
            new_ball_vel_y = obstacles.ball_vel_y.at[spawn_idx].set(
                jax.lax.select(should_spawn, spawn_vel_y, obstacles.ball_vel_y[spawn_idx])
            )
            new_ball_floor = obstacles.ball_floor.at[spawn_idx].set(
                jax.lax.select(should_spawn, spawn_floor, obstacles.ball_floor[spawn_idx])
            )

            # Return complete ObstacleState with all fields to match spawn_cart return type
            return ObstacleState(
                cart_x=obstacles.cart_x,
                cart_y=obstacles.cart_y,
                cart_active=obstacles.cart_active,
                cart_speed=obstacles.cart_speed,
                cart_spawn_timer=obstacles.cart_spawn_timer,
                ball_x=new_ball_x,
                ball_y=new_ball_y,
                ball_active=new_ball_active,
                ball_vel_x=new_ball_vel_x,
                ball_vel_y=new_ball_vel_y,
                ball_spawn_timer=obstacles.ball_spawn_timer,
                ball_floor=new_ball_floor,
                ball_is_bouncing=obstacles.ball_is_bouncing,
                plane_x=obstacles.plane_x,
                plane_y=obstacles.plane_y,
                plane_active=obstacles.plane_active,
                plane_speed=obstacles.plane_speed,
                plane_spawn_timer=obstacles.plane_spawn_timer
            )

        # Use jax.lax.cond to handle obstacle type branching
        return jax.lax.cond(
            obstacle_type == 0,
            spawn_cart,
            spawn_ball
        )

    @partial(jax.jit, static_argnums=(0,))
    def _update_obstacles(self, state: GameState, key: chex.PRNGKey) -> ObstacleState:
        """Update all obstacles including movement and spawning."""
        obstacles = state.obstacles

        # Update shopping carts
        new_cart_x = (obstacles.cart_x + obstacles.cart_speed * obstacles.cart_active).astype(jnp.int32)
        cart_out_of_bounds = jnp.logical_or(
            new_cart_x < -self.consts.CART_WIDTH,
            new_cart_x > self.consts.SCREEN_WIDTH
        )
        new_cart_active = jnp.logical_and(obstacles.cart_active, jnp.logical_not(cart_out_of_bounds))

        # Update bouncing balls with proper physics
        new_ball_x = (obstacles.ball_x + obstacles.ball_vel_x * obstacles.ball_active).astype(jnp.int32)
        new_ball_y = (obstacles.ball_y + obstacles.ball_vel_y * obstacles.ball_active).astype(jnp.int32)

        # Enhanced ball bouncing logic with floor-specific collision detection
        # Get floor Y positions for each ball based on their assigned floor (consistent with spawn)
        ball_floor_y = jnp.where(
            obstacles.ball_floor == 0,
            self.consts.FLOOR_1_Y + 2,  # Ground floor surface (with +2 offset like player)
            jnp.where(
                obstacles.ball_floor == 1,
                self.consts.FLOOR_2_Y + 2,  # Middle floor (with +2 offset like player)
                self.consts.FLOOR_3_Y + 2   # Top floor (with +2 offset like player)
            )
        )

        # Ball hits floor when it reaches or goes below the floor level
        ball_hit_floor = new_ball_y >= ball_floor_y

        # Apply gravity and bouncing physics with proper parabolic motion
        # First apply gravity to all balls with proper fractional handling
        # Scale gravity by 10 to handle decimals, then divide by 10
        scaled_gravity = jnp.int32(self.consts.BALL_GRAVITY * 10)
        gravity_applied_vel_y = obstacles.ball_vel_y + (scaled_gravity // 10)

        # Add fractional part every 10 frames for smoother motion
        fractional_part = scaled_gravity % 10
        add_fractional = (obstacles.ball_y % 10) < fractional_part  # Use position as pseudo-random for fractional gravity
        gravity_applied_vel_y = jnp.where(add_fractional, gravity_applied_vel_y + 1, gravity_applied_vel_y)

        # Then check for bouncing when hitting floor
        new_ball_vel_y = jnp.where(
            jnp.logical_and(ball_hit_floor, obstacles.ball_vel_y >= 0),  # Only bounce when falling down and hitting floor
            jnp.array(-self.consts.BALL_BOUNCE_HEIGHT, dtype=jnp.int32),  # Bounce upward
            gravity_applied_vel_y  # Apply gravity
        )

        # Clamp ball Y position to floor level when hitting floor (don't let it go through)
        new_ball_y = jnp.where(ball_hit_floor, ball_floor_y.astype(jnp.int32), new_ball_y)

        # Update bouncing state (true when ball is touching the floor)
        new_ball_is_bouncing = ball_hit_floor

        # Ball goes out of bounds horizontally (strict section bounds)
        # Get current player section for section-based ball management
        current_section = state.player.x // self.consts.SCREEN_WIDTH
        section_start_x = current_section * self.consts.SCREEN_WIDTH
        section_end_x = section_start_x + self.consts.SCREEN_WIDTH

        # Ball disappears when it completely leaves the current section
        ball_out_of_bounds = jnp.logical_or(
            new_ball_x + self.consts.BALL_WIDTH < section_start_x,  # Completely left of section
            new_ball_x > section_end_x                              # Completely right of section
        )
        new_ball_active = jnp.logical_and(obstacles.ball_active, jnp.logical_not(ball_out_of_bounds))

        # Update toy planes
        new_plane_x = (obstacles.plane_x + obstacles.plane_speed * obstacles.plane_active).astype(jnp.int32)
        plane_out_of_bounds = jnp.logical_or(
            new_plane_x < -self.consts.PLANE_WIDTH,
            new_plane_x > self.consts.SCREEN_WIDTH
        )
        new_plane_active = jnp.logical_and(obstacles.plane_active, jnp.logical_not(plane_out_of_bounds))

        # Update spawn timers
        new_cart_spawn_timer = jnp.maximum(obstacles.cart_spawn_timer - 1, 0)
        new_ball_spawn_timer = jnp.maximum(obstacles.ball_spawn_timer - 1, 0)
        new_plane_spawn_timer = jnp.maximum(obstacles.plane_spawn_timer - 1, 0)

        # Spawn new obstacles when timers reach zero
        obstacles_with_updates = ObstacleState(
            cart_x=new_cart_x,
            cart_y=obstacles.cart_y,
            cart_active=new_cart_active,
            cart_speed=obstacles.cart_speed,
            cart_spawn_timer=new_cart_spawn_timer,

            ball_x=new_ball_x,
            ball_y=new_ball_y,
            ball_active=new_ball_active,
            ball_vel_x=obstacles.ball_vel_x,
            ball_vel_y=new_ball_vel_y,
            ball_spawn_timer=new_ball_spawn_timer,
            ball_floor=obstacles.ball_floor,
            ball_is_bouncing=new_ball_is_bouncing,

            plane_x=new_plane_x,
            plane_y=obstacles.plane_y,
            plane_active=new_plane_active,
            plane_speed=obstacles.plane_speed,
            plane_spawn_timer=new_plane_spawn_timer
        )

        # Spawn new balls when timer reaches zero (with proper spacing)
        ball_spawn_key = jrandom.split(key)[0]
        should_spawn_ball = new_ball_spawn_timer == 0

        # Only spawn if there aren't any active balls in current section
        current_section = state.player.x // self.consts.SCREEN_WIDTH
        section_start_x = current_section * self.consts.SCREEN_WIDTH
        section_end_x = section_start_x + self.consts.SCREEN_WIDTH

        # Count active balls in current section only (strict boundaries)
        balls_in_current_section = jnp.sum(
            jnp.logical_and(
                obstacles_with_updates.ball_active,
                jnp.logical_and(
                    obstacles_with_updates.ball_x >= section_start_x,
                    obstacles_with_updates.ball_x <= section_end_x
                )
            )
        )

        # Only spawn if there are no balls in current section
        can_spawn_ball = balls_in_current_section == 0
        should_spawn_ball_final = jnp.logical_and(should_spawn_ball, can_spawn_ball)

        obstacles_after_ball_spawn = jax.lax.cond(
            should_spawn_ball_final,
            lambda: self._spawn_obstacle(ball_spawn_key, state._replace(obstacles=obstacles_with_updates), 1),
            lambda: obstacles_with_updates
        )

        # Reset ball spawn timer if we spawned a ball, otherwise use longer interval
        spawn_interval_frames = jnp.where(
            should_spawn_ball_final,
            # If we spawned, use normal interval
            jrandom.uniform(
                ball_spawn_key, (),
                minval=self.consts.BALL_MIN_SPAWN_INTERVAL * 60,  # Convert seconds to frames
                maxval=self.consts.BALL_MAX_SPAWN_INTERVAL * 60
            ).astype(jnp.int32),
            # If we didn't spawn due to too many balls, wait longer before trying again
            jrandom.uniform(
                ball_spawn_key, (),
                minval=5.0 * 60,  # Wait at least 5 seconds
                maxval=8.0 * 60   # Up to 8 seconds
            ).astype(jnp.int32)
        )

        final_ball_spawn_timer = jax.lax.select(
            should_spawn_ball,  # Reset timer if original condition was met
            spawn_interval_frames,
            obstacles_after_ball_spawn.ball_spawn_timer
        )

        return obstacles_after_ball_spawn._replace(ball_spawn_timer=final_ball_spawn_timer)

    @partial(jax.jit, static_argnums=(0,))
    def _spawn_items(self, key: chex.PRNGKey) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        """Spawn collectible items (moneybags/suitcases) in valid sections on each floor."""
        # Valid sections for item spawning (exclude first, middle/elevator, and last sections)
        # Sections: 0, 1, 2, 3(elevator), 4, 5, 6
        # Valid sections: 1, 2, 4, 5
        valid_sections = jnp.array([1, 2, 4, 5])
        floors = jnp.array([0, 1, 2])  # Floor indices

        # Split key for different random choices
        key, subkey = jrandom.split(key)

        # For each floor, spawn 1-2 items
        items_per_floor = 1  # Keep it simple: 1 item per floor on valid sections
        total_items = len(floors) * items_per_floor

        # Initialize arrays
        item_x = jnp.zeros(self.consts.MAX_ITEMS, dtype=jnp.int32)
        item_y = jnp.zeros(self.consts.MAX_ITEMS, dtype=jnp.int32)
        item_active = jnp.zeros(self.consts.MAX_ITEMS, dtype=bool)
        item_type = jnp.zeros(self.consts.MAX_ITEMS, dtype=jnp.int32)

        # Generate items for each floor
        for floor_idx in range(3):  # 3 floors
            if floor_idx < self.consts.MAX_ITEMS:
                # Choose random valid section for this floor
                section_key, key = jrandom.split(key)
                section_idx = jrandom.choice(section_key, valid_sections)

                # Calculate item position (center of section)
                item_x_pos = section_idx * self.consts.SECTION_WIDTH + self.consts.SECTION_WIDTH // 2
                item_y_pos = self._floor_y_position(jnp.array(floor_idx))

                # Choose random item type
                type_key, key = jrandom.split(key)
                item_type_val = jrandom.choice(type_key, jnp.array([self.consts.ITEM_TYPE_MONEYBAG, self.consts.ITEM_TYPE_SUITCASE]))

                # Update arrays
                item_x = item_x.at[floor_idx].set(item_x_pos)
                item_y = item_y.at[floor_idx].set(item_y_pos)
                item_active = item_active.at[floor_idx].set(True)
                item_type = item_type.at[floor_idx].set(item_type_val)

        return item_x, item_y, item_active, item_type

    @partial(jax.jit, static_argnums=(0,))
    def _spawn_stationary_obstacles(self, state: GameState, key: chex.PRNGKey) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        """Spawn stationary obstacles only after thief caught once, on floors 2-3 and roof."""
        # Only spawn if thief has been caught at least once
        should_spawn = state.thief_caught

        # Valid sections for obstacle spawning (exclude first=0, middle/elevator=3, and last=6 sections)
        # Valid sections: 1, 2, 4, 5
        valid_sections = jnp.array([1, 2, 4, 5])
        # Valid floors: 1 (middle floor), 2 (top floor) - these are floors 2-3 in user terms
        valid_floors = jnp.array([1, 2])  # Skip floor 0 (ground floor)

        # Initialize arrays
        obstacle_x = jnp.zeros(self.consts.MAX_STATIONARY_OBSTACLES, dtype=jnp.int32)
        obstacle_y = jnp.zeros(self.consts.MAX_STATIONARY_OBSTACLES, dtype=jnp.int32)
        obstacle_active = jnp.zeros(self.consts.MAX_STATIONARY_OBSTACLES, dtype=bool)
        obstacle_floor = jnp.zeros(self.consts.MAX_STATIONARY_OBSTACLES, dtype=jnp.int32)

        # Generate obstacles on each valid floor
        def spawn_obstacles():
            obs_x, obs_y, obs_active, obs_floor = obstacle_x, obstacle_y, obstacle_active, obstacle_floor

            # For each valid floor, try to spawn one obstacle
            for i, floor_idx in enumerate(valid_floors):
                if i < self.consts.MAX_STATIONARY_OBSTACLES:
                    # Choose random section for this floor
                    floor_key = jrandom.fold_in(key, i)
                    section_idx = jrandom.choice(floor_key, valid_sections)

                    # Calculate obstacle position (center of section)
                    obstacle_x_pos = section_idx * self.consts.SECTION_WIDTH + self.consts.SECTION_WIDTH // 2
                    obstacle_y_pos = self._floor_y_position(jnp.array(floor_idx)) + 4  # Match item positioning

                    # Update arrays
                    obs_x = obs_x.at[i].set(obstacle_x_pos)
                    obs_y = obs_y.at[i].set(obstacle_y_pos)
                    obs_active = obs_active.at[i].set(True)
                    obs_floor = obs_floor.at[i].set(floor_idx)

            return obs_x, obs_y, obs_active, obs_floor

        # Use JAX conditional to spawn or not based on thief_caught
        return jax.lax.cond(
            should_spawn,
            lambda: spawn_obstacles(),
            lambda: (obstacle_x, obstacle_y, obstacle_active, obstacle_floor)
        )

    @partial(jax.jit, static_argnums=(0,))
    def _update_shopping_carts(self, state: GameState, key: chex.PRNGKey) -> GameState:
        """Update shopping cart positions and spawn new ones."""
        # Update spawn timer
        new_spawn_timer = jnp.maximum(0, state.shopping_cart_spawn_timer - 1)

        # Check if we should spawn a new cart
        # Shopping carts only appear after the thief has been caught at least twice (level >= 2)
        level_requirement_met = state.level >= 0

        # Check section restrictions (similar to ball logic)
        current_section = state.player.x // self.consts.SCREEN_WIDTH
        elevator_section = self.consts.ELEVATOR_BUILDING_X // self.consts.SCREEN_WIDTH
        total_sections = 7  # Based on TOTAL_BUILDING_WIDTH / SCREEN_WIDTH

        # Exception sections where shopping carts should NOT spawn
        is_first_section = current_section == 0
        is_last_section = current_section == (total_sections - 1)
        is_elevator_section = current_section == elevator_section

        in_exception_section = jnp.logical_or(
            jnp.logical_or(is_first_section, is_last_section),
            is_elevator_section
        )

        should_spawn = jnp.logical_and(
            jnp.logical_and(
                jnp.logical_and(
                    new_spawn_timer == 0,
                    jnp.sum(state.shopping_cart_active) < self.consts.MAX_SHOPPING_CARTS
                ),
                level_requirement_met
            ),
            jnp.logical_not(in_exception_section)
        )

        # Update existing cart positions
        new_x = (state.shopping_cart_x + state.shopping_cart_direction * self.consts.SHOPPING_CART_BASE_SPEED).astype(jnp.int32)

        # Section-based bounds (similar to ball logic)
        # Get current player section for section-based cart management
        current_section = state.player.x // self.consts.SCREEN_WIDTH
        section_start_x = current_section * self.consts.SCREEN_WIDTH
        section_end_x = section_start_x + self.consts.SCREEN_WIDTH

        # Shopping cart disappears when it completely leaves the current section
        cart_out_of_bounds = jnp.logical_or(
            new_x + self.consts.SHOPPING_CART_WIDTH < section_start_x,  # Completely left of section
            new_x > section_end_x                                      # Completely right of section
        )
        new_active = jnp.logical_and(state.shopping_cart_active, jnp.logical_not(cart_out_of_bounds))

        # Spawn new cart if needed
        def spawn_new_cart():
            # Choose random floor (0, 1, 2 for floors 1, 2, 3)
            spawn_key = jrandom.fold_in(key, state.step_counter)
            floor_idx = jrandom.choice(spawn_key, jnp.array([0, 1, 2]))

            # Determine direction and start position based on floor
            direction = jnp.where(
                floor_idx == 1,  # Floor 2 (middle) goes right to left
                -1,
                1  # Floors 1 and 3 go left to right
            )

            # Start position: spawn within current section bounds (similar to ball logic)
            # Get current player section for section-based spawning
            current_section = state.player.x // self.consts.SCREEN_WIDTH
            section_start_x = current_section * self.consts.SCREEN_WIDTH
            section_end_x = section_start_x + self.consts.SCREEN_WIDTH

            start_x = jnp.where(
                direction == 1,
                section_start_x,  # Start at left edge of current section
                section_end_x - self.consts.SHOPPING_CART_WIDTH  # Start at right edge of current section
            ).astype(jnp.int32)

            # Y position for the floor
            cart_y = (self._floor_y_position(floor_idx) + 6).astype(jnp.int32)  # Match ball positioning offset

            # Find first inactive cart slot
            first_inactive = jnp.argmax(jnp.logical_not(new_active))

            # Update cart arrays
            updated_x = new_x.at[first_inactive].set(start_x)
            updated_active = new_active.at[first_inactive].set(True)
            updated_floor = state.shopping_cart_floor.at[first_inactive].set(floor_idx)
            updated_direction = state.shopping_cart_direction.at[first_inactive].set(direction)
            updated_y = state.shopping_cart_y.at[first_inactive].set(cart_y)

            # Reset spawn timer
            spawn_interval = jrandom.uniform(
                spawn_key, (),
                minval=self.consts.SHOPPING_CART_MIN_SPAWN_INTERVAL * 60,
                maxval=self.consts.SHOPPING_CART_MAX_SPAWN_INTERVAL * 60
            ).astype(jnp.int32)

            return state._replace(
                shopping_cart_x=updated_x,
                shopping_cart_y=updated_y,
                shopping_cart_active=updated_active,
                shopping_cart_floor=updated_floor,
                shopping_cart_direction=updated_direction,
                shopping_cart_spawn_timer=spawn_interval
            )

        def no_spawn():
            return state._replace(
                shopping_cart_x=new_x,
                shopping_cart_active=new_active,
                shopping_cart_spawn_timer=new_spawn_timer
            )

        return jax.lax.cond(should_spawn, spawn_new_cart, no_spawn)

    @partial(jax.jit, static_argnums=(0,))
    def _check_collisions(self, state: GameState) -> Tuple[bool, bool, bool, int, chex.Array]:
        """Check all collision types with proper hitbox handling for jump/crouch."""
        player = state.player

        # Player hitbox depends on crouching state
        player_height = jax.lax.select(
            player.is_crouching,
            self.consts.PLAYER_CROUCH_HEIGHT,
            self.consts.PLAYER_HEIGHT
        )

        # Convert player bottom position to top position for collision detection
        # player.y represents the bottom of the player, we need top for collision detection
        player_top_y = player.y - player_height

        # Convert thief bottom position to top position for collision detection
        # state.thief.y also represents the bottom of the thief, we need top for collision detection
        thief_top_y = state.thief.y - self.consts.THIEF_HEIGHT

        # Player-thief collision
        thief_collision = self._entities_collide(
            player.x, player_top_y, self.consts.PLAYER_WIDTH, player_height,
            state.thief.x, thief_top_y, self.consts.THIEF_WIDTH, self.consts.THIEF_HEIGHT
        )
        thief_caught = jnp.logical_and(thief_collision, player.floor == state.thief.floor)

        # Player-obstacle collisions with jump/crouch mechanics
        # Check cart collisions (ground level, can crouch under)
        cart_collisions = jax.vmap(
            lambda i: jnp.logical_and(
                state.obstacles.cart_active[i],
                self._entities_collide(
                    player.x, player_top_y, self.consts.PLAYER_WIDTH, player_height,
                    state.obstacles.cart_x[i], state.obstacles.cart_y[i],
                    self.consts.CART_WIDTH, self.consts.CART_HEIGHT
                )
            )
        )(jnp.arange(self.consts.MAX_OBSTACLES))

        # Check ball collisions (can jump over)
        # Note: both player.y and ball_y represent bottom positions, so we need to adjust for collision detection
        ball_collisions = jax.vmap(
            lambda i: jnp.logical_and(
                state.obstacles.ball_active[i],
                self._entities_collide(
                    player.x, player_top_y, self.consts.PLAYER_WIDTH, player_height,
                    state.obstacles.ball_x[i], state.obstacles.ball_y[i] - self.consts.BALL_HEIGHT,
                    self.consts.BALL_WIDTH, self.consts.BALL_HEIGHT
                )
            )
        )(jnp.arange(self.consts.MAX_OBSTACLES))

        # Check plane collisions (fly overhead, can crouch under)
        plane_collisions = jax.vmap(
            lambda i: jnp.logical_and(
                state.obstacles.plane_active[i],
                self._entities_collide(
                    player.x, player_top_y, self.consts.PLAYER_WIDTH, player_height,
                    state.obstacles.plane_x[i], state.obstacles.plane_y[i],
                    self.consts.PLANE_WIDTH, self.consts.PLANE_HEIGHT
                )
            )
        )(jnp.arange(self.consts.MAX_OBSTACLES))

        # Separate ball collisions from other obstacles (balls have special collision effects)
        # Balls can be avoided by jumping over them, but still hit the player's full height
        ball_hit = jnp.logical_and(jnp.logical_not(player.is_jumping), jnp.any(ball_collisions))

        # Other obstacle collisions (carts and planes)
        other_obstacle_collisions = jnp.logical_or(jnp.any(cart_collisions), jnp.any(plane_collisions))

        # Check stationary obstacle collisions
        stationary_obstacle_collisions = jax.vmap(
            lambda i: jnp.logical_and(
                state.stationary_obstacle_active[i],
                self._entities_collide(
                    player.x, player_top_y, self.consts.PLAYER_WIDTH, player_height,
                    state.stationary_obstacle_x[i], state.stationary_obstacle_y[i] - self.consts.OBSTACLE_HEIGHT,
                    self.consts.OBSTACLE_WIDTH, self.consts.OBSTACLE_HEIGHT
                )
            )
        )(jnp.arange(self.consts.MAX_STATIONARY_OBSTACLES))

        # Check shopping cart collisions
        shopping_cart_collisions = jax.vmap(
            lambda i: jnp.logical_and(
                state.shopping_cart_active[i],
                self._entities_collide(
                    player.x, player_top_y, self.consts.PLAYER_WIDTH, player_height,
                    state.shopping_cart_x[i], state.shopping_cart_y[i] - self.consts.SHOPPING_CART_HEIGHT,
                    self.consts.SHOPPING_CART_WIDTH, self.consts.SHOPPING_CART_HEIGHT
                )
            )
        )(jnp.arange(self.consts.MAX_SHOPPING_CARTS))

        # Combine all obstacle collisions
        new_obstacle_collisions = jnp.logical_or(
            jnp.any(stationary_obstacle_collisions),
            jnp.any(shopping_cart_collisions)
        )
        all_obstacle_collisions = jnp.logical_or(other_obstacle_collisions, new_obstacle_collisions)

        # Only count obstacle hits if player is not jumping (jumping avoids most obstacles)
        # Exception: planes can still hit if player jumps too high
        obstacle_hit = jnp.logical_and(jnp.logical_not(player.is_jumping), all_obstacle_collisions)

        # Item collection
        item_collisions = jax.vmap(
            lambda i: jnp.logical_and(
                state.item_active[i],
                self._entities_collide(
                    player.x, player_top_y, self.consts.PLAYER_WIDTH, player_height,
                    state.item_x[i], state.item_y[i] - self.consts.ITEM_HEIGHT,  # Convert item bottom to top position
                    self.consts.ITEM_WIDTH, self.consts.ITEM_HEIGHT
                )
            )
        )(jnp.arange(self.consts.MAX_ITEMS))
        items_collected = jnp.sum(item_collisions)

        return obstacle_hit, ball_hit, thief_caught, items_collected, ball_collisions, item_collisions, stationary_obstacle_collisions, shopping_cart_collisions

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey = None) -> Tuple[KeystoneKapersObservation, GameState]:
        """Reset the game to initial state with building traversal setup."""
        if key is None:
            key = jrandom.PRNGKey(0)

        # Initialize player at rightmost section of ground floor (as per original game)
        # player_start_x = 76  # Center of first section (0 + 152/2 = 76)
        player_start_x = self.consts.TOTAL_BUILDING_WIDTH - 78  # Near right edge
        player = PlayerState(
            x=jnp.array(player_start_x),
            y=jnp.array(self.consts.FLOOR_1_Y + 2),  # Lowered by 2 pixels
            floor=jnp.array(0),
            vel_x=jnp.array(0),
            vel_y=jnp.array(0),
            is_jumping=jnp.array(False),
            is_crouching=jnp.array(False),
            jump_timer=jnp.array(0),
            jump_start_y=jnp.array(self.consts.FLOOR_1_Y + 2),  # Lowered by 2 pixels
            freeze_timer=jnp.array(0),  # Added freeze timer for ball collisions
            on_escalator=jnp.array(0),  # 0 = not on escalator
            escalator_progress=jnp.array(0.0),
            is_on_elevator=jnp.array(False),
            in_elevator=jnp.array(False)
        )

        # Initialize thief at elevator position on floor 1 (middle floor), moving right initially
        thief_start_x = self.consts.ELEVATOR_BUILDING_X + self.consts.ELEVATOR_WIDTH // 2
        thief = ThiefState(
            x=jnp.array(thief_start_x),
            y=jnp.array(self.consts.FLOOR_2_Y + 2),  # Lowered by 2 pixels
            floor=jnp.array(1),  # Floor 1 = middle floor in this numbering system
            speed=jnp.array(self.consts.THIEF_BASE_SPEED),
            direction=jnp.array(1),  # Moving right initially
            escaped=jnp.array(False)
        )

        # Initialize empty obstacles
        obstacles = ObstacleState(
            cart_x=jnp.zeros(self.consts.MAX_OBSTACLES, dtype=jnp.int32),
            cart_y=jnp.zeros(self.consts.MAX_OBSTACLES, dtype=jnp.int32),
            cart_active=jnp.zeros(self.consts.MAX_OBSTACLES, dtype=bool),
            cart_speed=jnp.zeros(self.consts.MAX_OBSTACLES),  # Keep speed as float
            cart_spawn_timer=jnp.array(60),  # 1 second

            ball_x=jnp.zeros(self.consts.MAX_OBSTACLES, dtype=jnp.int32),
            ball_y=jnp.zeros(self.consts.MAX_OBSTACLES, dtype=jnp.int32),
            ball_active=jnp.zeros(self.consts.MAX_OBSTACLES, dtype=bool),
            ball_vel_x=jnp.zeros(self.consts.MAX_OBSTACLES, dtype=jnp.int32),
            ball_vel_y=jnp.zeros(self.consts.MAX_OBSTACLES, dtype=jnp.int32),
            ball_spawn_timer=jnp.array(45),  # 0.75 seconds - reduced from 120 for earlier spawning
            ball_floor=jnp.zeros(self.consts.MAX_OBSTACLES, dtype=jnp.int32),  # Floor assignment (0-2)
            ball_is_bouncing=jnp.zeros(self.consts.MAX_OBSTACLES, dtype=bool),  # Bouncing state

            plane_x=jnp.zeros(self.consts.MAX_OBSTACLES, dtype=jnp.int32),
            plane_y=jnp.zeros(self.consts.MAX_OBSTACLES, dtype=jnp.int32),
            plane_active=jnp.zeros(self.consts.MAX_OBSTACLES, dtype=bool),
            plane_speed=jnp.zeros(self.consts.MAX_OBSTACLES),  # Keep speed as float
            plane_spawn_timer=jnp.array(180)  # 3 seconds
        )

        # Initialize elevator with new state machine
        elevator = ElevatorState(
            floor=jnp.array(0),         # Start at ground floor
            state=jnp.array(0),         # IdleOpen
            timer=jnp.array(0),
            target_floor=jnp.array(1),  # Start by going up to floor 1
            has_player=jnp.array(False)
        )

        # Initialize camera to show player's current section
        player_section = player_start_x // self.consts.SECTION_WIDTH
        camera_x = jnp.array(player_section * self.consts.SECTION_WIDTH, dtype=jnp.int32)

        # Spawn items
        key, item_key = jrandom.split(key)
        item_x, item_y, item_active, item_type = self._spawn_items(item_key)

        # Initialize game state
        state = GameState(
            player=player,
            thief=thief,
            obstacles=obstacles,
            elevator=elevator,

            # Camera system for scrolling
            camera_x=jnp.array(camera_x),

            score=jnp.array(0),
            lives=jnp.array(3),
            level=jnp.array(1),
            timer=jnp.array(self.consts.BASE_TIMER),
            step_counter=jnp.array(0),
            escalator_frame=jnp.array(0),  # Animation frame for escalator steps

            item_x=item_x,
            item_y=item_y,
            item_active=item_active,
            item_type=item_type,

            # Initialize new obstacles
            stationary_obstacle_x=jnp.zeros(self.consts.MAX_STATIONARY_OBSTACLES, dtype=jnp.int32),
            stationary_obstacle_y=jnp.zeros(self.consts.MAX_STATIONARY_OBSTACLES, dtype=jnp.int32),
            stationary_obstacle_active=jnp.zeros(self.consts.MAX_STATIONARY_OBSTACLES, dtype=bool),
            stationary_obstacle_floor=jnp.zeros(self.consts.MAX_STATIONARY_OBSTACLES, dtype=jnp.int32),

            # Shopping carts
            shopping_cart_x=jnp.zeros(self.consts.MAX_SHOPPING_CARTS, dtype=jnp.int32),
            shopping_cart_y=jnp.zeros(self.consts.MAX_SHOPPING_CARTS, dtype=jnp.int32),
            shopping_cart_active=jnp.zeros(self.consts.MAX_SHOPPING_CARTS, dtype=bool),
            shopping_cart_floor=jnp.zeros(self.consts.MAX_SHOPPING_CARTS, dtype=jnp.int32),
            shopping_cart_direction=jnp.ones(self.consts.MAX_SHOPPING_CARTS, dtype=jnp.int32),  # Start moving right
            shopping_cart_spawn_timer=jnp.array(120),  # 2 seconds

            game_over=jnp.array(False),
            thief_caught=jnp.array(False),
            level_complete=jnp.array(False)
        )

        observation = self._get_observation(state)
        return observation, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: GameState, action: chex.Array) -> Tuple[KeystoneKapersObservation, GameState, float, bool, KeystoneKapersInfo]:
        """Step the environment forward by one frame with camera updates."""
        # Generate random key for this step
        step_key = jrandom.PRNGKey(state.step_counter)

        # Update all game components
        new_player = self._update_player(state, action)
        new_thief = self._update_thief(state)
        new_elevator = self._update_elevator(state, action)
        new_obstacles = self._update_obstacles(state, step_key)

        # Update shopping carts
        updated_state = self._update_shopping_carts(state._replace(
            player=new_player,
            thief=new_thief,
            obstacles=new_obstacles,
            elevator=new_elevator
        ), step_key)

        # True section-based camera system
        # Camera always shows the section that the player is currently in
        # Section boundaries are exact: 0-159 = section 0, 160-319 = section 1, etc.

        # Calculate which section the player is currently in
        player_section = new_player.x // self.consts.SECTION_WIDTH

        # Camera always shows the player's current section (no thresholds, no delays)
        new_camera_x = jnp.array(player_section * self.consts.SECTION_WIDTH, dtype=jnp.int32)

        # Clamp camera to valid building bounds
        max_camera_x = self.consts.TOTAL_BUILDING_WIDTH - self.consts.SECTION_WIDTH
        new_camera_x = jnp.clip(new_camera_x, 0, max_camera_x).astype(jnp.int32)

        # Check collisions
        obstacle_hit, ball_hit, thief_caught, items_collected, ball_collisions, item_collisions, stationary_obstacle_collisions, shopping_cart_collisions = self._check_collisions(updated_state)

        # Handle ball collision effects
        ball_timer_penalty = jax.lax.select(ball_hit, self.consts.BALL_TIME_PENALTY * 60, 0)  # Convert seconds to frames
        new_freeze_timer = jax.lax.select(ball_hit, self.consts.BALL_FREEZE_TIME, jnp.maximum(0, new_player.freeze_timer - 1))

        # Deactivate balls that collided with the player (only if not jumping to match collision logic)
        collision_mask = jnp.logical_and(ball_collisions, jnp.logical_not(new_player.is_jumping))
        new_ball_active = jnp.logical_and(new_obstacles.ball_active, jnp.logical_not(collision_mask))

        # Update obstacles with deactivated balls
        new_obstacles = new_obstacles._replace(ball_active=new_ball_active)

        # Handle obstacle collisions - calculate penalties and deactivate obstacles
        stationary_obstacle_hit = jnp.any(stationary_obstacle_collisions)
        shopping_cart_hit = jnp.any(shopping_cart_collisions)

        # Calculate obstacle timer penalties (10 seconds each)
        obstacle_timer_penalty = jax.lax.select(
            jnp.logical_or(stationary_obstacle_hit, shopping_cart_hit),
            self.consts.OBSTACLE_TIME_PENALTY,  # 10 seconds in frames
            0
        )

        # Deactivate obstacles that were hit
        new_stationary_obstacle_active = jnp.logical_and(
            updated_state.stationary_obstacle_active,
            jnp.logical_not(stationary_obstacle_collisions)
        )
        new_shopping_cart_active = jnp.logical_and(
            updated_state.shopping_cart_active,
            jnp.logical_not(shopping_cart_collisions)
        )

        # Update player with freeze timer
        player_with_freeze = new_player._replace(freeze_timer=new_freeze_timer)

        # Update timer with penalties (ball penalty + obstacle penalty + original collision penalty)
        timer_penalty = (jax.lax.select(obstacle_hit, self.consts.COLLISION_PENALTY, 0) +
                        ball_timer_penalty + obstacle_timer_penalty)
        new_timer = jnp.maximum(state.timer - 1 - timer_penalty, 0)

        # Update score
        # Calculate time bonus: remaining time * 100 when thief is caught
        time_bonus = jax.lax.select(
            thief_caught,
            (new_timer // 60) * self.consts.TIME_BONUS_MULTIPLIER,  # Convert frames to seconds, then multiply by 100
            0
        )
        # Item points: 50 points per collected item
        item_points = items_collected * self.consts.ITEM_POINTS
        # No jump points according to the specification
        jump_points = 0

        score_increase = time_bonus + item_points + jump_points
        new_score = state.score + score_increase

        # Check if player earned extra life (every 10,000 points)
        # Calculate how many thresholds were crossed with this score increase
        old_thresholds_crossed = state.score // self.consts.EXTRA_LIFE_THRESHOLD
        new_thresholds_crossed = new_score // self.consts.EXTRA_LIFE_THRESHOLD
        extra_lives_earned = jnp.maximum(0, new_thresholds_crossed - old_thresholds_crossed)

        # Award extra lives
        lives_with_bonus = jnp.minimum(3, state.lives + extra_lives_earned)

        # Check game end conditions
        time_up = new_timer <= 0
        thief_escaped = new_thief.escaped
        level_complete = thief_caught

        # Update lives - subtract life if lost, but also include any earned bonus lives
        life_lost = jnp.logical_or(time_up, thief_escaped)
        lives_after_penalty = jax.lax.select(life_lost, state.lives - 1, state.lives)

        # Final lives value considers both bonuses and penalties
        new_lives = jax.lax.select(life_lost,
                                  jnp.minimum(9, lives_after_penalty + extra_lives_earned),
                                  lives_with_bonus)

        # Level progression when thief is caught
        new_level = jax.lax.select(thief_caught, state.level + 1, state.level)

        # Spawn stationary obstacles when thief is caught (updates state for future levels)
        def spawn_stationary_obstacles():
            obstacle_key = jrandom.fold_in(step_key, state.step_counter + 1)
            obs_x, obs_y, obs_active, obs_floor = self._spawn_stationary_obstacles(
                updated_state._replace(thief_caught=True), obstacle_key
            )
            return updated_state._replace(
                stationary_obstacle_x=obs_x,
                stationary_obstacle_y=obs_y,
                stationary_obstacle_active=obs_active,
                stationary_obstacle_floor=obs_floor
            )

        def no_spawn_obstacles():
            return updated_state

        # Update state with stationary obstacles if thief is caught
        state_with_obstacles = jax.lax.cond(thief_caught, spawn_stationary_obstacles, no_spawn_obstacles)
        updated_state = state_with_obstacles

        # Reset level when ANY of these conditions occur:
        # 1. Thief is caught (advance to next level)
        # 2. Life is lost (time up or thief escaped) - restart current level
        should_reset_level = jnp.logical_or(thief_caught, life_lost)

        # When thief is caught, reset thief to starting position and reset timer
        thief_reset_x = self.consts.ELEVATOR_BUILDING_X + self.consts.ELEVATOR_WIDTH // 2
        thief_reset_y = self.consts.FLOOR_2_Y + 2  # Lowered by 2 pixels
        thief_reset_floor = 1  # Middle floor

        # Reset thief state when level resets (either caught or life lost)
        final_thief = ThiefState(
            x=jax.lax.select(should_reset_level, jnp.array(thief_reset_x), new_thief.x),
            y=jax.lax.select(should_reset_level, jnp.array(thief_reset_y), new_thief.y),
            floor=jax.lax.select(should_reset_level, jnp.array(thief_reset_floor), new_thief.floor),
            speed=jax.lax.select(should_reset_level,
                                jnp.array(self.consts.THIEF_BASE_SPEED * (1.0 + new_level * self.consts.THIEF_SPEED_SCALE)),
                                new_thief.speed),
            direction=jax.lax.select(should_reset_level, jnp.array(1), new_thief.direction),
            escaped=jax.lax.select(should_reset_level, jnp.array(False), new_thief.escaped)
        )

        # Reset timer when level resets
        final_timer = jax.lax.select(should_reset_level, self.consts.BASE_TIMER, new_timer)

        # Reset player position when level resets
        player_reset_x = self.consts.TOTAL_BUILDING_WIDTH - 78  # Near right edge
        final_player = PlayerState(
            x=jax.lax.select(should_reset_level, jnp.array(player_reset_x), player_with_freeze.x),
            y=jax.lax.select(should_reset_level, jnp.array(self.consts.FLOOR_1_Y + 2), player_with_freeze.y),  # Lowered by 2 pixels
            floor=jax.lax.select(should_reset_level, jnp.array(0), player_with_freeze.floor),
            vel_x=jax.lax.select(should_reset_level, jnp.array(0), player_with_freeze.vel_x),
            vel_y=jax.lax.select(should_reset_level, jnp.array(0), player_with_freeze.vel_y),
            is_jumping=jax.lax.select(should_reset_level, jnp.array(False), player_with_freeze.is_jumping),
            is_crouching=jax.lax.select(should_reset_level, jnp.array(False), player_with_freeze.is_crouching),
            jump_timer=jax.lax.select(should_reset_level, jnp.array(0), player_with_freeze.jump_timer),
            jump_start_y=jax.lax.select(should_reset_level, jnp.array(self.consts.FLOOR_1_Y + 2), player_with_freeze.jump_start_y),  # Lowered by 2 pixels
            freeze_timer=jax.lax.select(should_reset_level, jnp.array(0), player_with_freeze.freeze_timer),
            on_escalator=jax.lax.select(should_reset_level, jnp.array(0), player_with_freeze.on_escalator),
            escalator_progress=jax.lax.select(should_reset_level, jnp.array(0.0), player_with_freeze.escalator_progress),
            is_on_elevator=jax.lax.select(should_reset_level, jnp.array(False), player_with_freeze.is_on_elevator),
            in_elevator=jax.lax.select(should_reset_level, jnp.array(False), player_with_freeze.in_elevator)
        )

        # Game over condition - only end game when lives are depleted, not when level is complete
        # In Keystone Kapers, catching the thief should advance to next level, not end the game
        game_over = new_lives <= 0  # Only end game when no lives left

        # Deactivate collected items
        new_item_active = jnp.logical_and(state.item_active, jnp.logical_not(item_collisions))

        # Create new state with camera update
        new_state = GameState(
            player=final_player,
            thief=final_thief,
            obstacles=new_obstacles,
            elevator=new_elevator,

            # Update camera position
            camera_x=new_camera_x,

            score=new_score,
            lives=new_lives,
            level=new_level,  # Level increases when thief is caught
            timer=final_timer,
            step_counter=state.step_counter + 1,

            item_x=state.item_x,
            item_y=state.item_y,
            item_active=new_item_active,
            item_type=state.item_type,

            # Update new obstacles with deactivated states
            stationary_obstacle_x=updated_state.stationary_obstacle_x,
            stationary_obstacle_y=updated_state.stationary_obstacle_y,
            stationary_obstacle_active=new_stationary_obstacle_active,
            stationary_obstacle_floor=updated_state.stationary_obstacle_floor,

            # Shopping carts
            shopping_cart_x=updated_state.shopping_cart_x,
            shopping_cart_y=updated_state.shopping_cart_y,
            shopping_cart_active=new_shopping_cart_active,
            shopping_cart_floor=updated_state.shopping_cart_floor,
            shopping_cart_direction=updated_state.shopping_cart_direction,
            shopping_cart_spawn_timer=updated_state.shopping_cart_spawn_timer,

            # Update escalator animation frame
            escalator_frame=(state.escalator_frame + 1) % self.consts.ESCALATOR_ANIMATION_SPEED,

            game_over=game_over,
            thief_caught=thief_caught,
            level_complete=level_complete
        )

        # Calculate reward and info
        reward = self._get_reward(state, new_state)
        all_rewards = self._get_all_rewards(state, new_state)
        info = self._get_info(new_state, all_rewards)
        observation = self._get_observation(new_state)
        done = self._get_done(new_state)

        return observation, new_state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: GameState) -> KeystoneKapersObservation:
        """Extract observation from game state."""
        # Flatten obstacle positions
        obstacle_positions = jnp.concatenate([
            jnp.stack([state.obstacles.cart_x, state.obstacles.cart_y,
                      jnp.zeros_like(state.obstacles.cart_x)], axis=1),  # Type 0 for carts
            jnp.stack([state.obstacles.ball_x, state.obstacles.ball_y,
                      jnp.ones_like(state.obstacles.ball_x)], axis=1),   # Type 1 for balls
            jnp.stack([state.obstacles.plane_x, state.obstacles.plane_y,
                      2 * jnp.ones_like(state.obstacles.plane_x)], axis=1)  # Type 2 for planes
        ], axis=0)

        # Flatten item positions
        item_positions = jnp.stack([
            state.item_x, state.item_y, state.item_type
        ], axis=1)

        return KeystoneKapersObservation(
            player_x=state.player.x,
            player_y=state.player.y,
            player_floor=state.player.floor,
            thief_x=state.thief.x,
            thief_y=state.thief.y,
            thief_floor=state.thief.floor,
            obstacle_positions=obstacle_positions,
            item_positions=item_positions,
            score=state.score,
            timer=state.timer,
            level=state.level,
            lives=state.lives,
            elevator_position=state.elevator.floor,
            elevator_is_open=(state.elevator.state == 0)  # IdleOpen state
        )

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: KeystoneKapersObservation) -> chex.Array:
        """Convert observation to flat array for neural networks."""
        return jnp.concatenate([
            obs.player_x.flatten(),
            obs.player_y.flatten(),
            obs.player_floor.flatten(),
            obs.thief_x.flatten(),
            obs.thief_y.flatten(),
            obs.thief_floor.flatten(),
            obs.obstacle_positions.flatten(),
            obs.item_positions.flatten(),
            obs.score.flatten(),
            obs.timer.flatten(),
            obs.level.flatten(),
            obs.lives.flatten(),
            obs.elevator_position.flatten(),
            obs.elevator_is_open.flatten()
        ])

    def action_space(self) -> spaces.Discrete:
        """Return the action space."""
        return spaces.Discrete(len(self.action_set))

    def observation_space(self) -> spaces.Dict:
        """Return the observation space."""
        return spaces.Dict({
            "player_x": spaces.Box(low=0, high=self.consts.TOTAL_BUILDING_WIDTH, shape=(), dtype=jnp.int32),
            "player_y": spaces.Box(low=0, high=self.consts.SCREEN_HEIGHT, shape=(), dtype=jnp.int32),
            "player_floor": spaces.Box(low=0, high=3, shape=(), dtype=jnp.int32),
            "thief_x": spaces.Box(low=0, high=self.consts.TOTAL_BUILDING_WIDTH, shape=(), dtype=jnp.int32),
            "thief_y": spaces.Box(low=0, high=self.consts.SCREEN_HEIGHT, shape=(), dtype=jnp.int32),
            "thief_floor": spaces.Box(low=0, high=3, shape=(), dtype=jnp.int32),
            "obstacle_positions": spaces.Box(
                low=-1, high=max(self.consts.TOTAL_BUILDING_WIDTH, self.consts.SCREEN_HEIGHT),
                shape=(self.consts.MAX_OBSTACLES * 3, 3), dtype=jnp.int32
            ),
            "item_positions": spaces.Box(
                low=-1, high=max(self.consts.TOTAL_BUILDING_WIDTH, self.consts.SCREEN_HEIGHT),
                shape=(self.consts.MAX_ITEMS, 3), dtype=jnp.int32
            ),
            "score": spaces.Box(low=0, high=999999, shape=(), dtype=jnp.int32),
            "timer": spaces.Box(low=0, high=self.consts.BASE_TIMER, shape=(), dtype=jnp.int32),
            "level": spaces.Box(low=1, high=99, shape=(), dtype=jnp.int32),
            "lives": spaces.Box(low=0, high=9, shape=(), dtype=jnp.int32),
            "elevator_position": spaces.Box(low=0, high=2, shape=(), dtype=jnp.int32),
            "elevator_is_open": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32)
        })

    def image_space(self) -> spaces.Box:
        """Return the image space."""
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.consts.TOTAL_SCREEN_HEIGHT, self.consts.TOTAL_SCREEN_WIDTH, 3),
            dtype=jnp.uint8
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: GameState, all_rewards: chex.Array = None) -> KeystoneKapersInfo:
        """Extract additional information from game state."""
        return KeystoneKapersInfo(
            step_count=state.step_counter,
            thief_caught=state.thief_caught,
            thief_escaped=state.thief.escaped,
            items_collected=jnp.array(0),  # TODO: Track this
            obstacles_hit=jnp.array(0),    # TODO: Track this
            all_rewards=all_rewards if all_rewards is not None else jnp.zeros(1)
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: GameState, state: GameState) -> float:
        """Calculate reward for this step."""
        # Score-based reward
        score_reward = state.score - previous_state.score

        # Time penalty (small negative reward for each step)
        time_penalty = -0.1

        # Large negative penalty for losing (time up or thief escaped)
        game_lost = jnp.logical_or(
            jnp.logical_and(state.timer <= 0, previous_state.timer > 0),
            jnp.logical_and(state.thief.escaped, jnp.logical_not(previous_state.thief.escaped))
        )
        loss_penalty = jax.lax.select(game_lost, -1000.0, 0.0)

        # Large positive reward for catching thief
        thief_caught_reward = jax.lax.select(
            jnp.logical_and(state.thief_caught, jnp.logical_not(previous_state.thief_caught)),
            1000.0,
            0.0
        )

        return score_reward + time_penalty + loss_penalty + thief_caught_reward

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_rewards(self, previous_state: GameState, state: GameState) -> chex.Array:
        """Get all reward components if custom reward functions are defined."""
        if self.reward_funcs is None:
            return jnp.array([self._get_reward(previous_state, state)])

        rewards = jnp.array([
            reward_func(previous_state, state) for reward_func in self.reward_funcs
        ])
        return rewards

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: GameState) -> bool:
        """Check if the episode is finished."""
        # Only end when all lives are depleted
        # Don't end when thief is caught or escapes - that should just reset the level
        return state.game_over  # game_over is True only when lives <= 0

    def render(self, state: GameState) -> jnp.ndarray:
        """Render the current game state."""
        return self.renderer.render(state)


class KeystoneKapersRenderer(JAXGameRenderer):
    """Renderer for KeystoneKapers using simple colored rectangles."""

    def __init__(self, consts: KeystoneKapersConstants = None):
        super().__init__()
        self.consts = consts or KeystoneKapersConstants()
        self.sprites = self._create_simple_sprites()

    def _create_simple_sprites(self) -> Dict[str, Any]:
        """Create simple colored rectangle sprites and load actual sprites."""
        sprites = {}

        # Load the sky and buildings sprite
        try:
            sky_sprite_path = os.path.join(os.path.dirname(__file__), 'sprites', 'keystonekapers', 'sky_and_buildings.npy')
            sky_sprite_rgba = jr.loadFrame(sky_sprite_path)
            # Convert RGBA to RGB by taking first 3 channels
            sprites['sky'] = sky_sprite_rgba[:, :, :3].astype(jnp.uint8)
        except:
            # Fallback to simple blue rectangle if sprite loading fails
            sprites['sky'] = jnp.ones((40, 152, 3), dtype=jnp.uint8) * jnp.array([135, 206, 235], dtype=jnp.uint8)  # Sky blue

        # Load the kop life sprite
        try:
            kop_life_sprite_path = os.path.join(os.path.dirname(__file__), 'sprites', 'keystonekapers', 'kop_life.npy')
            kop_life_sprite_rgba = jr.loadFrame(kop_life_sprite_path)

            # Handle RGBA properly - use alpha channel for transparency
            rgb_data = kop_life_sprite_rgba[:, :, :3]
            alpha_data = kop_life_sprite_rgba[:, :, 3:4]

            # Convert to RGB, handling transparency by using alpha blending with white background
            # Where alpha is 0 (transparent), use white; where alpha is 255, use the RGB color
            alpha_normalized = alpha_data.astype(jnp.float32) / 255.0
            white_background = jnp.ones_like(rgb_data) * 255

            sprites['kop_life'] = (rgb_data.astype(jnp.float32) * alpha_normalized +
                                 white_background * (1 - alpha_normalized)).astype(jnp.uint8)
        except Exception as e:
            # Fallback to simple green rectangle if sprite loading fails
            sprites['kop_life'] = jnp.ones((8, 10, 3), dtype=jnp.uint8) * jnp.array([0, 255, 0], dtype=jnp.uint8)  # Green

        # Load black digit sprites for countdown timer (all digits 0-9)
        black_digits = {}

        for digit in range(10):  # Load all digits 0-9
            try:
                digit_sprite_path = os.path.join(os.path.dirname(__file__), 'sprites', 'keystonekapers', f'black_{digit}.npy')
                digit_sprite_rgba = jr.loadFrame(digit_sprite_path)

                # Handle RGBA properly with alpha blending
                rgb_data = digit_sprite_rgba[:, :, :3]
                alpha_data = digit_sprite_rgba[:, :, 3:4]
                alpha_normalized = alpha_data.astype(jnp.float32) / 255.0
                white_background = jnp.ones_like(rgb_data) * 255

                black_digits[digit] = (rgb_data.astype(jnp.float32) * alpha_normalized +
                                     white_background * (1 - alpha_normalized)).astype(jnp.uint8)
            except:
                # Fallback to simple black rectangle for missing digits
                black_digits[digit] = jnp.ones((8, 6, 3), dtype=jnp.uint8) * jnp.array([0, 0, 0], dtype=jnp.uint8)

        sprites['black_digits'] = black_digits

        # Load white digit sprites for score display (all digits 0-9)
        # Use exactly the same approach as the black digits
        white_digits = {}

        for digit in range(10):  # Load all digits 0-9
            try:
                digit_sprite_path = os.path.join(os.path.dirname(__file__), 'sprites', 'keystonekapers', f'white_{digit}.npy')
                digit_sprite_rgba = jr.loadFrame(digit_sprite_path)

                # Handle RGBA exactly the same way as black digits
                rgb_data = digit_sprite_rgba[:, :, :3]
                alpha_data = digit_sprite_rgba[:, :, 3:4]
                alpha_normalized = alpha_data.astype(jnp.float32) / 255.0
                white_background = jnp.ones_like(rgb_data) * 255

                white_digits[digit] = (rgb_data.astype(jnp.float32) * alpha_normalized +
                                     white_background * (1 - alpha_normalized)).astype(jnp.uint8)
                print(f"Loaded white digit {digit}: {white_digits[digit].shape}")
            except Exception as e:
                print(f"Failed to load white digit {digit}: {e}")
                # Fallback to simple white rectangle for missing digits
                white_digits[digit] = jnp.ones((8, 6, 3), dtype=jnp.uint8) * jnp.array([255, 255, 255], dtype=jnp.uint8)

        sprites['white_digits'] = white_digits

        # Create an invisible placeholder sprite for leading zeros
        # Same size as other digits but matching background color #578bc9
        invisible_sprite = jnp.ones((8, 6, 3), dtype=jnp.uint8) * jnp.array([87, 139, 201], dtype=jnp.uint8)  # #578bc9 background color
        sprites['invisible_digit'] = invisible_sprite

        # Load the Activision logo sprite
        try:
            activision_logo_path = os.path.join(os.path.dirname(__file__), 'sprites', 'keystonekapers', 'activision logo.npy')
            activision_logo_rgba = jr.loadFrame(activision_logo_path)

            # Handle RGBA properly with alpha blending
            rgb_data = activision_logo_rgba[:, :, :3]
            alpha_data = activision_logo_rgba[:, :, 3:4]
            alpha_normalized = alpha_data.astype(jnp.float32) / 255.0
            white_background = jnp.ones_like(rgb_data) * 255

            sprites['activision_logo'] = (rgb_data.astype(jnp.float32) * alpha_normalized +
                                        white_background * (1 - alpha_normalized)).astype(jnp.uint8)
        except:
            # Fallback to simple colored rectangle if sprite loading fails
            sprites['activision_logo'] = jnp.ones((10, 40, 3), dtype=jnp.uint8) * jnp.array([255, 255, 255], dtype=jnp.uint8)  # White

        # Load escalator sprites (4 animation frames for both directions)
        # First, collect all sprites from all sets to find global max dimensions
        all_sprite_sets = []
        global_max_height = 0
        global_max_width = 0

        sprite_prefixes = [
            'escalator_right_facing',
            'escalator_left_facing_floor_1',
            'escalator_left_facing_floor_3'
        ]

        # Load all sprites and find global maximum dimensions
        for sprite_prefix in sprite_prefixes:
            sprite_set = []
            for frame in range(1, 5):  # Load frames 1, 2, 3, 4
                try:
                    escalator_sprite_path = os.path.join(os.path.dirname(__file__), 'sprites', 'keystonekapers', f'{sprite_prefix}_{frame}.npy')
                    escalator_sprite_rgba = jr.loadFrame(escalator_sprite_path)

                    # Extract RGB channels and use alpha channel to create transparent pixels
                    # When alpha is low (transparent), set to white (will be treated as transparent)
                    rgb_data = escalator_sprite_rgba[:, :, :3]
                    alpha_data = escalator_sprite_rgba[:, :, 3:4]
                    alpha_normalized = alpha_data.astype(jnp.float32) / 255.0
                    white_background = jnp.ones_like(rgb_data) * 255

                    # For pixels with low alpha (below 0.5), use white (will be treated as transparent)
                    # For pixels with high alpha, use the original RGB color
                    escalator_frame_sprite = jnp.where(
                        alpha_normalized > 0.5,
                        rgb_data,
                        white_background
                    ).astype(jnp.uint8)
                    sprite_set.append(escalator_frame_sprite)
                    global_max_height = max(global_max_height, escalator_frame_sprite.shape[0])
                    global_max_width = max(global_max_width, escalator_frame_sprite.shape[1])

                    # Debug: Log dimensions of each sprite
                    print(f"Loaded sprite from {escalator_sprite_path}: {escalator_frame_sprite.shape}")
                except Exception as e:
                    # Fallback to simple purple rectangle for missing sprites
                    fallback_sprite = jnp.ones((25, 16, 3), dtype=jnp.uint8) * jnp.array(self.consts.ESCALATOR_COLOR, dtype=jnp.uint8)
                    sprite_set.append(fallback_sprite)
                    global_max_height = max(global_max_height, 25)
                    global_max_width = max(global_max_width, 16)

            all_sprite_sets.append(sprite_set)

        # Now pad all sprites to the global maximum dimensions
        def pad_sprite_set_to_global_size(sprite_set):
            padded_set = []
            for sprite in sprite_set:
                height_diff = global_max_height - sprite.shape[0]
                width_diff = global_max_width - sprite.shape[1]

                # Pad with white (255, 255, 255) - will be treated as transparent by draw_sprite_with_transparency
                padded_sprite = jnp.pad(
                    sprite,
                    ((0, height_diff), (0, width_diff), (0, 0)),
                    mode='constant',
                    constant_values=255
                )
                padded_set.append(padded_sprite)
            return padded_set

        # Pad all sprite sets to global dimensions
        escalator_right_sprites = pad_sprite_set_to_global_size(all_sprite_sets[0])
        escalator_left_floor_1_sprites = pad_sprite_set_to_global_size(all_sprite_sets[1])
        escalator_left_floor_3_sprites = pad_sprite_set_to_global_size(all_sprite_sets[2])

        # Stack escalator sprites for JAX-compatible indexing
        sprites['escalator_right_frames'] = jnp.stack(escalator_right_sprites, axis=0)
        sprites['escalator_left_floor_1_frames'] = jnp.stack(escalator_left_floor_1_sprites, axis=0)
        sprites['escalator_left_floor_3_frames'] = jnp.stack(escalator_left_floor_3_sprites, axis=0)

        # Create simple colored rectangles for each entity
        sprites['player'] = jnp.ones((self.consts.PLAYER_HEIGHT, self.consts.PLAYER_WIDTH, 3), dtype=jnp.uint8) * jnp.array(self.consts.PLAYER_COLOR, dtype=jnp.uint8)
        sprites['thief'] = jnp.ones((self.consts.THIEF_HEIGHT, self.consts.THIEF_WIDTH, 3), dtype=jnp.uint8) * jnp.array(self.consts.THIEF_COLOR, dtype=jnp.uint8)
        sprites['cart'] = jnp.ones((self.consts.CART_HEIGHT, self.consts.CART_WIDTH, 3), dtype=jnp.uint8) * jnp.array(self.consts.CART_COLOR, dtype=jnp.uint8)
        sprites['ball'] = jnp.ones((self.consts.BALL_HEIGHT, self.consts.BALL_WIDTH, 3), dtype=jnp.uint8) * jnp.array(self.consts.BALL_COLOR, dtype=jnp.uint8)
        sprites['plane'] = jnp.ones((self.consts.PLANE_HEIGHT, self.consts.PLANE_WIDTH, 3), dtype=jnp.uint8) * jnp.array(self.consts.PLANE_COLOR, dtype=jnp.uint8)

        # Load collectible item sprites (moneybag and suitcase)
        # Both sprites will be padded/cropped to ITEM_HEIGHT x ITEM_WIDTH for consistency
        target_height = self.consts.ITEM_HEIGHT
        target_width = self.consts.ITEM_WIDTH

        # Load moneybag sprite
        moneybag_sprite_path = os.path.join(os.path.dirname(__file__), 'sprites', 'keystonekapers', 'money_bag.npy')
        try:
            moneybag_sprite_rgba = jr.loadFrame(moneybag_sprite_path)

            # Handle RGBA properly with alpha blending
            rgb_data = moneybag_sprite_rgba[:, :, :3]
            alpha_data = moneybag_sprite_rgba[:, :, 3:4]
            alpha_normalized = alpha_data.astype(jnp.float32) / 255.0
            white_background = jnp.ones_like(rgb_data) * 255

            moneybag_raw = (rgb_data.astype(jnp.float32) * alpha_normalized +
                          white_background * (1 - alpha_normalized)).astype(jnp.uint8)

            # Use the sprite as-is without forced resizing to avoid shape conflicts
            # The sprite dimensions are reasonable (11x7), let's keep them
            sprites['moneybag'] = moneybag_raw
        except Exception as e:
            print(f"Failed to load moneybag sprite: {e}")
            # Fallback to yellow rectangle if sprite loading fails
            sprites['moneybag'] = jnp.ones((target_height, target_width, 3), dtype=jnp.uint8) * jnp.array([255, 255, 0], dtype=jnp.uint8)

        # Load suitcase sprite
        try:
            suitcase_sprite_path = os.path.join(os.path.dirname(__file__), 'sprites', 'keystonekapers', 'suitcase.npy')
            suitcase_sprite_rgba = jr.loadFrame(suitcase_sprite_path)

            # Handle RGBA properly with alpha blending
            rgb_data = suitcase_sprite_rgba[:, :, :3]
            alpha_data = suitcase_sprite_rgba[:, :, 3:4]
            alpha_normalized = alpha_data.astype(jnp.float32) / 255.0
            white_background = jnp.ones_like(rgb_data) * 255

            suitcase_raw = (rgb_data.astype(jnp.float32) * alpha_normalized +
                          white_background * (1 - alpha_normalized)).astype(jnp.uint8)

            # Use the sprite as-is without forced resizing
            sprites['suitcase'] = suitcase_raw
        except Exception as e:
            print(f"Failed to load suitcase sprite: {e}")
            # Fallback to blue rectangle if sprite loading fails
            sprites['suitcase'] = jnp.ones((target_height, target_width, 3), dtype=jnp.uint8) * jnp.array([0, 100, 255], dtype=jnp.uint8)

        # Load obstacle sprites
        obstacle_target_height = self.consts.OBSTACLE_HEIGHT
        obstacle_target_width = self.consts.OBSTACLE_WIDTH

        # Load stationary obstacle sprite
        try:
            obstacle_sprite_path = os.path.join(os.path.dirname(__file__), 'sprites', 'keystonekapers', 'obstacle.npy')
            obstacle_sprite_rgba = jr.loadFrame(obstacle_sprite_path)

            # Handle RGBA properly with alpha blending
            rgb_data = obstacle_sprite_rgba[:, :, :3]
            alpha_data = obstacle_sprite_rgba[:, :, 3:4]
            alpha_normalized = alpha_data.astype(jnp.float32) / 255.0
            white_background = jnp.ones_like(rgb_data) * 255

            obstacle_raw = (rgb_data.astype(jnp.float32) * alpha_normalized +
                          white_background * (1 - alpha_normalized)).astype(jnp.uint8)

            # Use the sprite as-is without forced resizing
            sprites['obstacle'] = obstacle_raw
        except Exception as e:
            print(f"Failed to load obstacle sprite: {e}")
            # Fallback to red rectangle if sprite loading fails
            sprites['obstacle'] = jnp.ones((obstacle_target_height, obstacle_target_width, 3), dtype=jnp.uint8) * jnp.array([255, 0, 0], dtype=jnp.uint8)

        # Load shopping cart sprite
        cart_target_height = self.consts.SHOPPING_CART_HEIGHT
        cart_target_width = self.consts.SHOPPING_CART_WIDTH

        try:
            shopping_cart_sprite_path = os.path.join(os.path.dirname(__file__), 'sprites', 'keystonekapers', 'shoppingkart.npy')
            shopping_cart_sprite_rgba = jr.loadFrame(shopping_cart_sprite_path)

            # Handle RGBA properly with alpha blending
            rgb_data = shopping_cart_sprite_rgba[:, :, :3]
            alpha_data = shopping_cart_sprite_rgba[:, :, 3:4]
            alpha_normalized = alpha_data.astype(jnp.float32) / 255.0
            white_background = jnp.ones_like(rgb_data) * 255

            shopping_cart_raw = (rgb_data.astype(jnp.float32) * alpha_normalized +
                               white_background * (1 - alpha_normalized)).astype(jnp.uint8)

            # Use the sprite as-is without forced resizing
            sprites['shopping_cart'] = shopping_cart_raw
        except Exception as e:
            print(f"Failed to load shopping cart sprite: {e}")
            # Fallback to orange rectangle if sprite loading fails
            sprites['shopping_cart'] = jnp.ones((cart_target_height, cart_target_width, 3), dtype=jnp.uint8) * jnp.array([255, 165, 0], dtype=jnp.uint8)

        # Load Kop sprites (standing and running animations)
        kop_sprites = {}

        # Load standing sprite and convert RGBA to RGB
        try:
            kop_standing_path = os.path.join(os.path.dirname(__file__), 'sprites', 'keystonekapers', 'kop_standing.npy')
            kop_standing_rgba = jr.loadFrame(kop_standing_path)

            # Handle RGBA properly with alpha blending
            rgb_data = kop_standing_rgba[:, :, :3]
            alpha_data = kop_standing_rgba[:, :, 3:4]
            alpha_normalized = alpha_data.astype(jnp.float32) / 255.0
            white_background = jnp.ones_like(rgb_data) * 255

            kop_sprites['standing'] = (rgb_data.astype(jnp.float32) * alpha_normalized +
                                     white_background * (1 - alpha_normalized)).astype(jnp.uint8)
            print(f"Loaded Kop standing sprite: {kop_sprites['standing'].shape}")
        except Exception as e:
            print(f"Failed to load Kop standing sprite: {e}")
            # Fallback to simple blue rectangle if sprite loading fails
            kop_sprites['standing'] = jnp.ones((23, 11, 3), dtype=jnp.uint8) * jnp.array([0, 0, 255], dtype=jnp.uint8)

        # Load running left sprites and convert RGBA to RGB
        kop_running_left_sprites = []
        for frame in range(1, 5):
            try:
                kop_running_left_path = os.path.join(os.path.dirname(__file__), 'sprites', 'keystonekapers', f'kop_facing_left_run_frame_{frame}.npy')
                sprite_rgba = jr.loadFrame(kop_running_left_path)

                # Handle RGBA properly with alpha blending
                rgb_data = sprite_rgba[:, :, :3]
                alpha_data = sprite_rgba[:, :, 3:4]
                alpha_normalized = alpha_data.astype(jnp.float32) / 255.0
                white_background = jnp.ones_like(rgb_data) * 255

                sprite_rgb = (rgb_data.astype(jnp.float32) * alpha_normalized +
                            white_background * (1 - alpha_normalized)).astype(jnp.uint8)
                kop_running_left_sprites.append(sprite_rgb)
                print(f"Loaded Kop running left sprite frame {frame}: {sprite_rgb.shape}")
            except Exception as e:
                print(f"Failed to load Kop running left sprite frame {frame}: {e}")
                # Fallback to simple blue rectangle
                fallback_sprite = jnp.ones((20, 8, 3), dtype=jnp.uint8) * jnp.array([0, 0, 255], dtype=jnp.uint8)
                kop_running_left_sprites.append(fallback_sprite)

        # Determine the maximum height and width among ALL Kop sprites (standing + running)
        all_kop_sprites = kop_running_left_sprites + [kop_sprites['standing']]
        global_max_height = max(sprite.shape[0] for sprite in all_kop_sprites)
        global_max_width = max(sprite.shape[1] for sprite in all_kop_sprites)

        print(f"Global Kop sprite dimensions: {global_max_height}x{global_max_width}")

        # Pad standing sprite to global dimensions
        standing_height_diff = global_max_height - kop_sprites['standing'].shape[0]
        standing_width_diff = global_max_width - kop_sprites['standing'].shape[1]
        kop_sprites['standing'] = jnp.pad(
            kop_sprites['standing'],
            ((0, standing_height_diff), (0, standing_width_diff), (0, 0)),
            mode='constant',
            constant_values=255  # Use white padding to match background
        )

        # Pad all running left sprites to the same global dimensions
        kop_running_left_sprites = [
            jnp.pad(
                sprite,
                ((0, global_max_height - sprite.shape[0]), (0, global_max_width - sprite.shape[1]), (0, 0)),
                mode='constant',
                constant_values=255  # Use white padding to match background
            )
            for sprite in kop_running_left_sprites
        ]

        # Mirror running left sprites to create running right sprites (integer-based logic)
        kop_running_right_sprites = [sprite[:, ::-1, :] for sprite in kop_running_left_sprites]
        print("Mirrored Kop running left sprites to create running right sprites using integer-based logic.")

        # Store running animations in the dictionary
        kop_sprites['running_left'] = jnp.stack(kop_running_left_sprites, axis=0)
        kop_sprites['running_right'] = jnp.stack(kop_running_right_sprites, axis=0)

        # Add Kop sprites to the main sprite dictionary
        sprites['kop'] = kop_sprites

        # Load Thief running sprites (right-facing frames, then mirror for left)
        thief_sprites = {}

        # Load running right sprites and convert RGBA to RGB
        thief_running_right_sprites = []
        for frame in range(1, 5):
            try:
                thief_running_right_path = os.path.join(os.path.dirname(__file__), 'sprites', 'keystonekapers', f'thief_run_right_{frame}.npy')
                sprite_rgba = jr.loadFrame(thief_running_right_path)

                # Handle RGBA properly with alpha blending
                rgb_data = sprite_rgba[:, :, :3]
                alpha_data = sprite_rgba[:, :, 3:4]
                alpha_normalized = alpha_data.astype(jnp.float32) / 255.0
                white_background = jnp.ones_like(rgb_data) * 255

                sprite_rgb = (rgb_data.astype(jnp.float32) * alpha_normalized +
                            white_background * (1 - alpha_normalized)).astype(jnp.uint8)
                thief_running_right_sprites.append(sprite_rgb)
                print(f"Loaded Thief running right sprite frame {frame}: {sprite_rgb.shape}")
            except Exception as e:
                print(f"Failed to load Thief running right sprite frame {frame}: {e}")
                # Fallback to simple red rectangle
                thief_running_right_sprites.append(jnp.ones((20, 8, 3), dtype=jnp.uint8) * jnp.array([255, 0, 0], dtype=jnp.uint8))

        # Determine the maximum height and width among the thief running sprites
        thief_max_height = max(sprite.shape[0] for sprite in thief_running_right_sprites)
        thief_max_width = max(sprite.shape[1] for sprite in thief_running_right_sprites)

        print(f"Thief sprite dimensions: {thief_max_height}x{thief_max_width}")

        # Pad all running right sprites to the same dimensions
        thief_running_right_sprites = [
            jnp.pad(
                sprite,
                ((0, thief_max_height - sprite.shape[0]), (0, thief_max_width - sprite.shape[1]), (0, 0)),
                mode='constant',
                constant_values=255  # Use white padding to match background
            )
            for sprite in thief_running_right_sprites
        ]

        # Mirror running right sprites to create running left sprites (integer-based logic)
        thief_running_left_sprites = [sprite[:, ::-1, :] for sprite in thief_running_right_sprites]
        print("Mirrored Thief running right sprites to create running left sprites using integer-based logic.")

        # Store running animations in the dictionary
        thief_sprites['running_left'] = jnp.stack(thief_running_left_sprites, axis=0)
        thief_sprites['running_right'] = jnp.stack(thief_running_right_sprites, axis=0)

        # Add Thief sprites to the main sprite dictionary
        sprites['thief'] = thief_sprites

        # Load bouncing ball sprites
        # Load ball sprites first to find common dimensions
        ball_sprites = {}
        ball_sprites_raw = {}

        try:
            # Load ball_air.npy sprite
            ball_air_path = os.path.join(os.path.dirname(__file__), 'sprites', 'keystonekapers', 'ball_air.npy')
            ball_sprites_raw['ball_air'] = jr.loadFrame(ball_air_path)
        except Exception as e:
            print(f"Failed to load ball_air.npy: {e}")
            # Fallback to simple white circle
            ball_sprites_raw['ball_air'] = jnp.ones((6, 6, 4), dtype=jnp.uint8) * jnp.array([255, 255, 255, 255], dtype=jnp.uint8)

        try:
            # Load ball_bounce.npy sprite
            ball_bounce_path = os.path.join(os.path.dirname(__file__), 'sprites', 'keystonekapers', 'ball_bounce.npy')
            ball_sprites_raw['ball_bounce'] = jr.loadFrame(ball_bounce_path)
        except Exception as e:
            print(f"Failed to load ball_bounce.npy: {e}")
            # Fallback to simple white circle
            ball_sprites_raw['ball_bounce'] = jnp.ones((6, 6, 4), dtype=jnp.uint8) * jnp.array([255, 255, 255, 255], dtype=jnp.uint8)

        # Find the maximum dimensions for both ball sprites
        ball_max_height = max(ball_sprites_raw['ball_air'].shape[0], ball_sprites_raw['ball_bounce'].shape[0])
        ball_max_width = max(ball_sprites_raw['ball_air'].shape[1], ball_sprites_raw['ball_bounce'].shape[1])

        # Process both ball sprites to same dimensions
        for sprite_name, sprite_data in ball_sprites_raw.items():
            # Pad sprite to maximum dimensions with white pixels
            height, width = sprite_data.shape[:2]
            pad_height = (ball_max_height - height) // 2
            pad_width = (ball_max_width - width) // 2

            # Pad with white pixels (255, 255, 255, 255 for RGBA or 255, 255, 255 for RGB)
            if sprite_data.shape[2] == 4:  # RGBA
                white_pixel = jnp.array([255, 255, 255, 255], dtype=jnp.uint8)
                padded_sprite = jnp.pad(sprite_data,
                                      ((pad_height, ball_max_height - height - pad_height),
                                       (pad_width, ball_max_width - width - pad_width),
                                       (0, 0)),
                                      mode='constant', constant_values=255)

                # Handle alpha blending
                alpha_channel = padded_sprite[:, :, 3:4] / 255.0
                rgb_channels = padded_sprite[:, :, :3]
                white_bg = jnp.ones_like(rgb_channels) * 255
                blended_sprite = (rgb_channels * alpha_channel + white_bg * (1 - alpha_channel)).astype(jnp.uint8)
                sprites[sprite_name] = blended_sprite
            else:  # RGB
                padded_sprite = jnp.pad(sprite_data,
                                      ((pad_height, ball_max_height - height - pad_height),
                                       (pad_width, ball_max_width - width - pad_width),
                                       (0, 0)),
                                      mode='constant', constant_values=255)
                sprites[sprite_name] = padded_sprite

        return sprites

    @partial(jax.jit, static_argnums=(0,))
    def _get_escalator_sprite_frame(self, state: GameState) -> chex.Array:
        """Get the current escalator sprite frame (0-3) based on animation counter."""
        # Use escalator_frame (which increments every step) to determine sprite frame
        # Animation cycles every 2 frames (slower animation)
        return (state.escalator_frame // 8) % 4

    @partial(jax.jit, static_argnums=(0,))
    def _get_escalator_sprite_for_floor(self, floor_num: int, sprite_frame: chex.Array) -> chex.Array:
        """Get the appropriate escalator sprite based on floor number."""
        # Floor 1 (index 0): left-facing floor 1 sprites
        # Floor 2 (index 1): right-facing sprites
        # Floor 3 (index 2): left-facing floor 3 sprites
        return jnp.where(
            floor_num == 0, self.sprites['escalator_left_floor_1_frames'][sprite_frame],
            jnp.where(
                floor_num == 1, self.sprites['escalator_right_frames'][sprite_frame],
                self.sprites['escalator_left_floor_3_frames'][sprite_frame]
            )
        )

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: GameState) -> jnp.ndarray:
        """Render the game state with Atari-style black borders."""
        # Create full screen with black borders
        frame = jnp.zeros(
            (self.consts.TOTAL_SCREEN_HEIGHT, self.consts.TOTAL_SCREEN_WIDTH, 3),
            dtype=jnp.uint8
        )

        # Create game area background
        game_area = jnp.ones(
            (self.consts.GAME_AREA_HEIGHT, self.consts.GAME_AREA_WIDTH, 3),
            dtype=jnp.uint8
        ) * jnp.array(self.consts.BACKGROUND_COLOR, dtype=jnp.uint8)

        # Helper function to convert building coordinates to screen coordinates
        def building_to_screen_x(building_x):
            """Convert building coordinate to screen coordinate using camera."""
            return building_x - state.camera_x

        # Simple helper to draw a filled rectangle (fixed size for JAX compatibility)
        def draw_rectangle_simple(game_area, x, y, width, height, color):
            """Draw a rectangle with static dimensions in game area."""
            # Ensure parameters are integers and arrays
            x = jnp.clip(jnp.asarray(x, dtype=jnp.int32), 0, self.consts.GAME_AREA_WIDTH - 1)
            y = jnp.clip(jnp.asarray(y, dtype=jnp.int32), 0, self.consts.GAME_AREA_HEIGHT - 1)
            width = jnp.clip(jnp.asarray(width, dtype=jnp.int32), 1, self.consts.GAME_AREA_WIDTH - x)
            height = jnp.clip(jnp.asarray(height, dtype=jnp.int32), 1, self.consts.GAME_AREA_HEIGHT - y)

            # Create indices for the rectangle area
            y_indices = jnp.arange(self.consts.GAME_AREA_HEIGHT)[:, None]
            x_indices = jnp.arange(self.consts.GAME_AREA_WIDTH)[None, :]

            # Create mask for the rectangle
            mask = ((y_indices >= y) & (y_indices < y + height) &
                   (x_indices >= x) & (x_indices < x + width))

            # Apply color where mask is True
            return jnp.where(mask[:, :, None], color, game_area)

        # Helper function to draw a sprite at a specific position
        def draw_sprite(game_area, sprite, x, y):
            """Draw a sprite at the given position in the game area."""
            sprite_height, sprite_width = sprite.shape[:2]

            # Ensure coordinates are within bounds and convert to integers
            x = jnp.clip(x, 0, self.consts.GAME_AREA_WIDTH - sprite_width).astype(jnp.int32)
            y = jnp.clip(y, 0, self.consts.GAME_AREA_HEIGHT - sprite_height).astype(jnp.int32)

            # Calculate the actual placement region
            x_start = x
            x_end = x + sprite_width
            y_start = y
            y_end = y + sprite_height

            # Use dynamic_slice to get the game area region where sprite should be placed
            game_region = jax.lax.dynamic_slice(
                game_area,
                (y_start, x_start, 0),
                (sprite_height, sprite_width, game_area.shape[2])
            )

            # Pad or crop sprite to match the region size if needed
            sprite_to_place = sprite

            # Create the updated game area by using dynamic_update_slice
            return jax.lax.dynamic_update_slice(
                game_area,
                sprite_to_place,
                (y_start, x_start, 0)
            )

        # Helper function to draw a sprite with transparency (white pixels = transparent)
        def draw_sprite_with_transparency(game_area, sprite, x, y, transparent_color=jnp.array([255, 255, 255], dtype=jnp.uint8)):
            """Draw a sprite at the given position treating specified color as transparent."""
            sprite_height, sprite_width = sprite.shape[:2]

            # Ensure coordinates are within bounds and convert to integers
            x = jnp.clip(x, 0, self.consts.GAME_AREA_WIDTH - sprite_width).astype(jnp.int32)
            y = jnp.clip(y, 0, self.consts.GAME_AREA_HEIGHT - sprite_height).astype(jnp.int32)

            # Calculate the actual placement region
            x_start = x
            x_end = x + sprite_width
            y_start = y
            y_end = y + sprite_height

            # Get the game area region where sprite should be placed
            game_region = jax.lax.dynamic_slice(
                game_area,
                (y_start, x_start, 0),
                (sprite_height, sprite_width, game_area.shape[2])
            )

            # Create transparency mask - pixels that are NOT the transparent color
            is_transparent = jnp.all(sprite == transparent_color, axis=-1, keepdims=True)
            is_opaque = jnp.logical_not(is_transparent)

            # Blend sprite with existing game region using transparency mask
            blended_region = jnp.where(is_opaque, sprite, game_region)

            # Update the game area with the blended region
            return jax.lax.dynamic_update_slice(
                game_area,
                blended_region,
                (y_start, x_start, 0)
            )

        # Draw sky and buildings sprite above the roof level (covering all green background)
        sky_sprite = self.sprites['sky']
        sky_height, sky_width = sky_sprite.shape[:2]

        # Position sky to cover from top of game area down to roof floor
        roof_floor_top = self.consts.ROOF_Y + self.consts.FLOOR_HEIGHT
        sky_bottom = 0  # Start from very top of game area
        sky_visible_height = roof_floor_top - sky_bottom

        if sky_visible_height > 0:
            # Tile the sky sprite horizontally to cover the game area width
            tiles_needed = (self.consts.GAME_AREA_WIDTH + sky_width - 1) // sky_width

            # Create horizontal tiling of sky sprite
            tiled_sky = jnp.tile(sky_sprite, (1, tiles_needed, 1))
            # Crop to exact game area width
            tiled_sky = tiled_sky[:, :self.consts.GAME_AREA_WIDTH, :]

            # If sky sprite is smaller than needed height, tile it vertically too
            if sky_height < sky_visible_height:
                vertical_tiles_needed = (sky_visible_height + sky_height - 1) // sky_height
                tiled_sky = jnp.tile(tiled_sky, (vertical_tiles_needed, 1, 1))

            # Take only the portion that fits in the visible area
            sky_to_draw = tiled_sky[:sky_visible_height, :, :]

            # Create mask for sky area (from top down to roof floor)
            y_indices = jnp.arange(self.consts.GAME_AREA_HEIGHT)[:, None]
            x_indices = jnp.arange(self.consts.GAME_AREA_WIDTH)[None, :]
            sky_mask = ((y_indices >= sky_bottom) & (y_indices < roof_floor_top) &
                       (x_indices >= 0) & (x_indices < self.consts.GAME_AREA_WIDTH))

            # Apply sky sprite to game area
            game_area = jnp.where(
                sky_mask[:, :, None],
                sky_to_draw[y_indices - sky_bottom, x_indices, :],
                game_area
            )

        # Draw floors with tan and yellow layers (like minimap)
        floor_tan_color = jnp.array(self.consts.FLOOR_TAN_COLOR, dtype=jnp.uint8)
        floor_yellow_color = jnp.array(self.consts.FLOOR_YELLOW_COLOR, dtype=jnp.uint8)
        floor_thickness = 6
        layer_thickness = 3

        # Floor 1 - yellow bottom, tan top
        game_area = draw_rectangle_simple(
            game_area, 0, self.consts.FLOOR_1_Y + self.consts.FLOOR_HEIGHT - floor_thickness,
            self.consts.GAME_AREA_WIDTH, layer_thickness, floor_yellow_color
        )
        game_area = draw_rectangle_simple(
            game_area, 0, self.consts.FLOOR_1_Y + self.consts.FLOOR_HEIGHT - layer_thickness,
            self.consts.GAME_AREA_WIDTH, layer_thickness, floor_tan_color
        )

        # Floor 2 - yellow bottom, tan top
        game_area = draw_rectangle_simple(
            game_area, 0, self.consts.FLOOR_2_Y + self.consts.FLOOR_HEIGHT - floor_thickness,
            self.consts.GAME_AREA_WIDTH, layer_thickness, floor_yellow_color
        )
        game_area = draw_rectangle_simple(
            game_area, 0, self.consts.FLOOR_2_Y + self.consts.FLOOR_HEIGHT - layer_thickness,
            self.consts.GAME_AREA_WIDTH, layer_thickness, floor_tan_color
        )

        # Floor 3 - yellow bottom, tan top
        game_area = draw_rectangle_simple(
            game_area, 0, self.consts.FLOOR_3_Y + self.consts.FLOOR_HEIGHT - floor_thickness,
            self.consts.GAME_AREA_WIDTH, layer_thickness, floor_yellow_color
        )
        game_area = draw_rectangle_simple(
            game_area, 0, self.consts.FLOOR_3_Y + self.consts.FLOOR_HEIGHT - layer_thickness,
            self.consts.GAME_AREA_WIDTH, layer_thickness, floor_tan_color
        )

        # Roof (4th level) - yellow bottom, tan top
        game_area = draw_rectangle_simple(
            game_area, 0, self.consts.ROOF_Y + self.consts.FLOOR_HEIGHT - floor_thickness,
            self.consts.GAME_AREA_WIDTH, layer_thickness, floor_yellow_color
        )
        game_area = draw_rectangle_simple(
            game_area, 0, self.consts.ROOF_Y + self.consts.FLOOR_HEIGHT - layer_thickness,
            self.consts.GAME_AREA_WIDTH, layer_thickness, floor_tan_color
        )

        # Draw escalators using animated sprites for all floors
        sprite_frame = self._get_escalator_sprite_frame(state)

        # Draw escalator 1 (Floor 1 - left-facing)
        escalator_1_building_x = self.consts.ESCALATOR_FLOOR1_X
        escalator_1_screen_x = building_to_screen_x(escalator_1_building_x)
        escalator_1_visible = (escalator_1_screen_x >= -self.consts.ESCALATOR_WIDTH) & (escalator_1_screen_x < self.consts.GAME_AREA_WIDTH)

        current_escalator_1_sprite = self._get_escalator_sprite_for_floor(0, sprite_frame)
        escalator_1_sprite_height = current_escalator_1_sprite.shape[0]
        escalator_1_sprite_y = self.consts.FLOOR_1_Y - (escalator_1_sprite_height - self.consts.FLOOR_HEIGHT) - 0

        game_area = jnp.where(
            escalator_1_visible,
            draw_sprite_with_transparency(game_area, current_escalator_1_sprite, 0, escalator_1_sprite_y),
            game_area
        )

        # Draw escalator 2 (Floor 2 - right-facing)
        escalator_2_building_x = self.consts.ESCALATOR_FLOOR2_X
        escalator_2_screen_x = building_to_screen_x(escalator_2_building_x)
        escalator_2_visible = (escalator_2_screen_x >= -self.consts.ESCALATOR_WIDTH) & (escalator_2_screen_x < self.consts.GAME_AREA_WIDTH)

        current_escalator_2_sprite = self._get_escalator_sprite_for_floor(1, sprite_frame)
        escalator_2_sprite_height = current_escalator_2_sprite.shape[0]
        escalator_2_sprite_y = self.consts.FLOOR_2_Y - (escalator_2_sprite_height - self.consts.FLOOR_HEIGHT) - 0

        game_area = jnp.where(
            escalator_2_visible,
            draw_sprite_with_transparency(game_area, current_escalator_2_sprite, 0, escalator_2_sprite_y),
            game_area
        )

        # Draw escalator 3 (Floor 3 - left-facing)
        escalator_3_building_x = self.consts.ESCALATOR_FLOOR3_X
        escalator_3_screen_x = building_to_screen_x(escalator_3_building_x)
        escalator_3_visible = (escalator_3_screen_x >= -self.consts.ESCALATOR_WIDTH) & (escalator_3_screen_x < self.consts.GAME_AREA_WIDTH)

        current_escalator_3_sprite = self._get_escalator_sprite_for_floor(2, sprite_frame)
        escalator_3_sprite_height = current_escalator_3_sprite.shape[0]
        escalator_3_sprite_y = self.consts.FLOOR_3_Y - (escalator_3_sprite_height - self.consts.FLOOR_HEIGHT) - 0

        game_area = jnp.where(
            escalator_3_visible,
            draw_sprite_with_transparency(game_area, current_escalator_3_sprite, 0, escalator_3_sprite_y),
            game_area
        )

        # Draw elevator shaft and doors on all floors
        elevator_building_x = self.consts.ELEVATOR_BUILDING_X
        elevator_screen_x = building_to_screen_x(elevator_building_x)
        elevator_visible = (elevator_screen_x >= -self.consts.ELEVATOR_WIDTH) & (elevator_screen_x < self.consts.GAME_AREA_WIDTH)

        # Elevator shaft background (dark blue/purple) on all floors
        shaft_color = jnp.array([64, 64, 128], dtype=jnp.uint8)  # Dark blue shaft

        # Draw shaft on floor 1 (ground)
        game_area = jnp.where(
            elevator_visible,
            draw_rectangle_simple(game_area, elevator_screen_x, self.consts.FLOOR_1_Y,
                                self.consts.ELEVATOR_WIDTH, self.consts.FLOOR_HEIGHT, shaft_color),
            game_area
        )

        # Draw shaft on floor 2 (middle)
        game_area = jnp.where(
            elevator_visible,
            draw_rectangle_simple(game_area, elevator_screen_x, self.consts.FLOOR_2_Y,
                                self.consts.ELEVATOR_WIDTH, self.consts.FLOOR_HEIGHT, shaft_color),
            game_area
        )

        # Draw shaft on floor 3 (top)
        game_area = jnp.where(
            elevator_visible,
            draw_rectangle_simple(game_area, elevator_screen_x, self.consts.FLOOR_3_Y,
                                self.consts.ELEVATOR_WIDTH, self.consts.FLOOR_HEIGHT, shaft_color),
            game_area
        )

        # Draw elevator car only at its current floor
        elevator_car_color = jnp.array([96, 96, 196], dtype=jnp.uint8)  # Lighter blue for car
        elevator_floor_y = jnp.where(
            state.elevator.floor == 0, self.consts.FLOOR_1_Y,
            jnp.where(
                state.elevator.floor == 1, self.consts.FLOOR_2_Y,
                self.consts.FLOOR_3_Y  # floor == 2 (top floor)
            )
        )

        # Only draw elevator car when doors are not fully open
        show_car = state.elevator.state != 0  # Not IdleOpen
        game_area = jnp.where(
            jnp.logical_and(elevator_visible, show_car),
            draw_rectangle_simple(game_area, elevator_screen_x, elevator_floor_y,
                                self.consts.ELEVATOR_WIDTH, self.consts.FLOOR_HEIGHT, elevator_car_color),
            game_area
        )

        # Old player rendering (now replaced by Kop sprite above)
        # player_screen_x = building_to_screen_x(state.player.x)
        # player_visible = (player_screen_x >= 0) & (player_screen_x <= self.consts.GAME_AREA_WIDTH - self.consts.PLAYER_WIDTH)
        # player_color = jnp.array([0, 255, 0], dtype=jnp.uint8)  # Green
        # game_area = jnp.where(
        #     player_visible,
        #     draw_rectangle_simple(game_area, player_screen_x, state.player.y,
        #                         self.consts.PLAYER_WIDTH, self.consts.PLAYER_HEIGHT, player_color),
        #     game_area
        # )

        # Draw thief with animated sprites and transparency (only if not escaped)
        thief_screen_x = building_to_screen_x(state.thief.x)
        thief_visible = (thief_screen_x >= 0) & (thief_screen_x <= self.consts.GAME_AREA_WIDTH - self.consts.THIEF_WIDTH) & jnp.logical_not(state.thief.escaped)

        # Only render thief if visible and not escaped
        def render_thief():
            # Thief is always moving (running) when not escaped
            # Use thief direction: 1 for right, -1 for left
            thief_is_facing_left = state.thief.direction < 0

            # Select sprite based on direction (thief is always running)
            thief_sprite = jnp.where(
                thief_is_facing_left,
                self.sprites['thief']['running_left'][(state.step_counter // 4) % 4],
                self.sprites['thief']['running_right'][(state.step_counter // 4) % 4]
            )

            # Position sprite so its bottom aligns with the floor
            thief_sprite_height = thief_sprite.shape[0]
            thief_screen_y = state.thief.y - thief_sprite_height + self.consts.THIEF_HEIGHT + 4  # Adjusted for floor height increase

            return draw_sprite_with_transparency(game_area, thief_sprite, thief_screen_x, thief_screen_y)

        # Render thief only if visible and not escaped
        game_area = jnp.where(
            thief_visible,
            render_thief(),
            game_area
        )

        # Draw the Kop sprite (representing the player character)
        # Determine if player is moving and select appropriate sprite
        is_moving = jnp.abs(state.player.vel_x) > 0.1
        is_facing_left = state.player.vel_x < 0

        # Select sprite based on movement
        kop_sprite = jnp.where(
            is_moving,
            # Running animation - cycle through 4 frames based on step counter
            jnp.where(
                is_facing_left,
                self.sprites['kop']['running_left'][(state.step_counter // 4) % 4],
                self.sprites['kop']['running_right'][(state.step_counter // 4) % 4]
            ),
            # Standing sprite
            self.sprites['kop']['standing']
        )

        # Calculate player's position on the screen
        player_screen_x = building_to_screen_x(state.player.x)

        # Position sprite so its bottom aligns with the floor
        kop_sprite_height = kop_sprite.shape[0]
        player_screen_y = state.player.y - kop_sprite_height + self.consts.PLAYER_HEIGHT + 4  # Adjusted for floor height increase

        # Draw the Kop sprite at the player's position with transparency (white pixels = transparent)
        game_area = draw_sprite_with_transparency(game_area, kop_sprite, player_screen_x, player_screen_y)

        # Draw bouncing balls
        ball_air_sprite = self.sprites['ball_air']
        ball_bounce_sprite = self.sprites['ball_bounce']

        def draw_ball(i):
            """Draw a single ball if it's active and visible on screen."""
            ball_active = state.obstacles.ball_active[i]
            ball_x = state.obstacles.ball_x[i]
            ball_y = state.obstacles.ball_y[i]
            ball_is_bouncing = state.obstacles.ball_is_bouncing[i]

            # Convert ball position to screen coordinates
            ball_screen_x = building_to_screen_x(ball_x)

            # Check if ball is visible on current screen
            ball_visible = jnp.logical_and(
                ball_active,
                jnp.logical_and(
                    ball_screen_x >= -self.consts.BALL_WIDTH,
                    ball_screen_x <= self.consts.GAME_AREA_WIDTH
                )
            )

            # Select appropriate sprite based on bouncing state
            current_ball_sprite = jnp.where(ball_is_bouncing, ball_bounce_sprite, ball_air_sprite)

            # Position ball sprite correctly - ball_y is the physics bottom position of the ball
            ball_sprite_height = current_ball_sprite.shape[0]
            # SINGLE POINT FOR BALL VISUAL ADJUSTMENT: Change +7 to move ball up/down on screen
            ball_screen_y = ball_y - ball_sprite_height + 22

            # Draw ball with transparency
            return jnp.where(
                ball_visible,
                draw_sprite_with_transparency(game_area, current_ball_sprite, ball_screen_x, ball_screen_y),
                game_area
            )

        # Draw all active balls
        for i in range(self.consts.MAX_OBSTACLES):
            game_area = draw_ball(i)

        # Draw collectible items (moneybags and suitcases)
        def draw_item(i):
            """Draw a single item with sprite and transparency."""
            item_active = state.item_active[i]
            item_x = state.item_x[i]
            item_y = state.item_y[i]
            item_type = state.item_type[i]

            # Convert building coordinates to screen coordinates
            item_screen_x = building_to_screen_x(item_x)

            # Check if item is visible on current screen
            item_visible = jnp.logical_and(
                item_active,
                jnp.logical_and(
                    item_screen_x >= -self.consts.ITEM_WIDTH,
                    item_screen_x <= self.consts.GAME_AREA_WIDTH
                )
            )

            # Choose sprite based on item type - use conditional logic instead of jnp.where
            # to avoid broadcasting issues with different sprite dimensions
            def draw_moneybag():
                current_sprite = self.sprites['moneybag']
                sprite_height = current_sprite.shape[0]
                # Use same positioning logic as player: object_y - sprite_height + object_height + 4
                screen_y = item_y - sprite_height + self.consts.ITEM_HEIGHT + 10
                return jnp.where(
                    item_visible,
                    draw_sprite_with_transparency(game_area, current_sprite, item_screen_x, screen_y),
                    game_area
                )

            def draw_suitcase():
                current_sprite = self.sprites['suitcase']
                sprite_height = current_sprite.shape[0]
                # Use same positioning logic as player: object_y - sprite_height + object_height + 4
                screen_y = item_y - sprite_height + self.consts.ITEM_HEIGHT + 10
                return jnp.where(
                    item_visible,
                    draw_sprite_with_transparency(game_area, current_sprite, item_screen_x, screen_y),
                    game_area
                )

            # Use lax.cond instead of jnp.where to avoid broadcasting issues
            return jax.lax.cond(
                item_type == self.consts.ITEM_TYPE_MONEYBAG,
                lambda _: draw_moneybag(),
                lambda _: draw_suitcase(),
                None
            )

        # Draw all active items
        for i in range(self.consts.MAX_ITEMS):
            game_area = draw_item(i)

        # Draw stationary obstacles
        def draw_stationary_obstacle(i):
            """Draw a single stationary obstacle."""
            obstacle_active = state.stationary_obstacle_active[i]
            obstacle_x = state.stationary_obstacle_x[i]
            obstacle_y = state.stationary_obstacle_y[i]

            # Convert world coordinates to screen coordinates
            obstacle_screen_x = building_to_screen_x(obstacle_x)
            obstacle_visible = ((obstacle_screen_x >= -self.consts.OBSTACLE_WIDTH) &
                              (obstacle_screen_x <= self.consts.GAME_AREA_WIDTH))

            def draw_obstacle_sprite():
                current_sprite = self.sprites['obstacle']
                sprite_height = current_sprite.shape[0]
                # Position sprite so its bottom aligns with the obstacle position
                screen_y = obstacle_y - sprite_height + self.consts.OBSTACLE_HEIGHT + 4
                return draw_sprite_with_transparency(game_area, current_sprite, obstacle_screen_x, screen_y)

            return jnp.where(
                jnp.logical_and(obstacle_active, obstacle_visible),
                draw_obstacle_sprite(),
                game_area
            )

        # Draw shopping carts
        def draw_shopping_cart(i):
            """Draw a single shopping cart."""
            cart_active = state.shopping_cart_active[i]
            cart_x = state.shopping_cart_x[i]
            cart_y = state.shopping_cart_y[i]

            # Convert world coordinates to screen coordinates
            cart_screen_x = building_to_screen_x(cart_x)
            cart_visible = ((cart_screen_x >= -self.consts.SHOPPING_CART_WIDTH) &
                          (cart_screen_x <= self.consts.GAME_AREA_WIDTH))

            def draw_cart_sprite():
                current_sprite = self.sprites['shopping_cart']
                sprite_height = current_sprite.shape[0]
                # Position sprite so its bottom aligns with the cart position
                screen_y = cart_y - sprite_height + self.consts.SHOPPING_CART_HEIGHT + 6
                return draw_sprite_with_transparency(game_area, current_sprite, cart_screen_x, screen_y)

            return jnp.where(
                jnp.logical_and(cart_active, cart_visible),
                draw_cart_sprite(),
                game_area
            )

        # Draw all active obstacles
        for i in range(self.consts.MAX_STATIONARY_OBSTACLES):
            game_area = draw_stationary_obstacle(i)

        for i in range(self.consts.MAX_SHOPPING_CARTS):
            game_area = draw_shopping_cart(i)

        # Draw simple UI elements on game area
        ui_color = jnp.array([255, 255, 255], dtype=jnp.uint8)  # White UI

        # Lives indicator (top left) - 3 kop life sprites next to each other
        kop_life_sprite = self.sprites['kop_life']
        life_sprite_width = kop_life_sprite.shape[1]

        # Draw 3 life sprites horizontally on the left side
        life_start_x = 10  # Start from the left edge
        life_y = 15  # Top of the screen

        # Draw each life sprite using JAX where for conditional drawing
        for i in range(3):
            life_x = life_start_x + (i * (life_sprite_width + 2))  # 2 pixel spacing
            # Use JAX where instead of Python if
            should_draw_life = state.lives > i
            game_area = jnp.where(
                should_draw_life,
                draw_sprite(game_area, kop_life_sprite, life_x, life_y),
                game_area
            )

        # Position timer to the right of lives - move this calculation earlier
        timer_start_x = life_start_x + (3 * (life_sprite_width + 2)) + 10
        timer_y = life_y

        # Score display above the timer - white digits showing current score
        white_digits = self.sprites['white_digits']
        invisible_digit = self.sprites['invisible_digit']

        # Use the actual score from state
        score_value = jnp.minimum(999999, state.score)  # Cap at 999,999 for display

        # Extract all 6 digits for scores up to 999,999
        score_hundred_thousands = (score_value // 100000) % 10
        score_ten_thousands = (score_value // 10000) % 10
        score_thousands = (score_value // 1000) % 10
        score_hundreds = (score_value // 100) % 10
        score_tens = (score_value // 10) % 10
        score_units = score_value % 10

        # Position score directly above the timer
        score_start_x = timer_start_x - 28  # Shifted further to the left
        score_y = timer_y - 10  # Position above timer

        # Helper function to draw score digit or invisible placeholder
        def draw_score_digit_or_invisible(game_area, digit_value, x, y, should_show_digit):
            # Use JAX select to choose appropriate sprite for each digit 0-9
            result_game_area = game_area

            # If we should show the digit, draw the actual digit sprite
            # If not, draw the invisible sprite
            for digit in range(10):
                digit_matches = jnp.logical_and(digit_value == digit, should_show_digit)
                digit_sprite = white_digits[digit]
                result_game_area = jnp.where(
                    digit_matches,
                    draw_sprite(result_game_area, digit_sprite, x, y),
                    result_game_area
                )

            # Draw invisible sprite when we shouldn't show the digit
            result_game_area = jnp.where(
                jnp.logical_not(should_show_digit),
                draw_sprite(result_game_area, invisible_digit, x, y),
                result_game_area
            )

            return result_game_area

        # Determine which digits should be visible (suppress leading zeros)
        # A digit is visible if it's non-zero OR any digit to its left is non-zero OR it's the units digit
        has_hundred_thousands = score_hundred_thousands > 0
        has_ten_thousands = jnp.logical_or(has_hundred_thousands, score_ten_thousands > 0)
        has_thousands = jnp.logical_or(has_ten_thousands, score_thousands > 0)
        has_hundreds = jnp.logical_or(has_thousands, score_hundreds > 0)
        has_tens = jnp.logical_or(has_hundreds, score_tens > 0)
        # Units digit is always shown

        # Draw all 6 digits
        digit_width = 6  # Approximate digit width
        digit_spacing = 1  # Space between digits

        game_area = draw_score_digit_or_invisible(game_area, score_hundred_thousands,
                                                score_start_x, score_y, has_hundred_thousands)
        game_area = draw_score_digit_or_invisible(game_area, score_ten_thousands,
                                                score_start_x + (digit_width + digit_spacing), score_y, has_ten_thousands)
        game_area = draw_score_digit_or_invisible(game_area, score_thousands,
                                                score_start_x + 2 * (digit_width + digit_spacing), score_y, has_thousands)
        game_area = draw_score_digit_or_invisible(game_area, score_hundreds,
                                                score_start_x + 3 * (digit_width + digit_spacing), score_y, has_hundreds)
        game_area = draw_score_digit_or_invisible(game_area, score_tens,
                                                score_start_x + 4 * (digit_width + digit_spacing), score_y, has_tens)
        # Units digit is always visible
        game_area = draw_score_digit_or_invisible(game_area, score_units,
                                                score_start_x + 5 * (digit_width + digit_spacing), score_y, True)

        # Countdown timer (to the right of lives) - black digits counting down from 50 to 0
        black_digits = self.sprites['black_digits']

        # Calculate timer value - show countdown from 50 to 0 regardless of actual timer value
        timer_seconds = jnp.maximum(0, state.timer // 60)  # Convert frames to seconds

        # Map the timer to a 50-second countdown (assuming timer starts at 60 seconds)
        # If timer is 60 seconds, show 50; if timer is 10 seconds, show 0
        max_display_time = 50
        timer_range = 60  # Assuming game timer starts at 60 seconds

        # Calculate display timer: starts at 50, ends at 0
        current_timer = jnp.maximum(0, jnp.minimum(max_display_time,
                                                  (timer_seconds * max_display_time) // timer_range))

        # Convert timer to tens and units digits
        tens_digit = current_timer // 10
        units_digit = current_timer % 10

        # Position timer to the right of lives
        timer_start_x = life_start_x + (3 * (life_sprite_width + 2)) + 10  # After lives + spacing
        timer_y = life_y

        # Helper function to draw digit sprite (all digits 0-9 available)
        def draw_digit_sprite(game_area, digit_value, x, y):
            # Use JAX select to choose appropriate sprite for each digit 0-9
            result_game_area = game_area

            # Check each digit 0-9 and draw if it matches
            for digit in range(10):
                digit_matches = digit_value == digit
                digit_sprite = black_digits[digit]
                result_game_area = jnp.where(
                    digit_matches,
                    draw_sprite(result_game_area, digit_sprite, x, y),
                    result_game_area
                )

            return result_game_area

        # Draw tens digit
        game_area = draw_digit_sprite(game_area, tens_digit, timer_start_x, timer_y)

        # Draw units digit
        digit_width = 6  # Approximate digit width
        game_area = draw_digit_sprite(game_area, units_digit, timer_start_x + digit_width + 1, timer_y)

        # Timer indicator (top right) - removed white bar, can add actual timer display later if needed

        # Camera position indicator (above minimap)
        # camera_indicator_x = (state.camera_x / self.consts.TOTAL_BUILDING_WIDTH * self.consts.GAME_AREA_WIDTH)
        # game_area = draw_rectangle_simple(game_area, camera_indicator_x, self.consts.FLOOR_1_Y + self.consts.FLOOR_HEIGHT + 5, 20, 5,
        #                                 jnp.array([255, 255, 0], dtype=jnp.uint8))  # Yellow camera indicator

        # Add minimap area right after floor 1 (eliminate blue gap)
        minimap_y_start = self.consts.FLOOR_1_Y + self.consts.FLOOR_HEIGHT  # Position right after floor 1
        minimap_color = jnp.array(self.consts.MINIMAP_COLOR, dtype=jnp.uint8)

        # Draw grey minimap background directly in game area (full width)
        game_area = draw_rectangle_simple(game_area, 0, minimap_y_start,
                                        self.consts.GAME_AREA_WIDTH, self.consts.MINIMAP_HEIGHT,
                                        minimap_color)

        # MINIMAP IMPLEMENTATION - Compact overview of entire building
        # Draw colored minimap area within grey background
        minimap_display_x = self.consts.MINIMAP_DISPLAY_OFFSET_X
        minimap_display_y = minimap_y_start + self.consts.MINIMAP_DISPLAY_OFFSET_Y
        minimap_width = self.consts.MINIMAP_DISPLAY_WIDTH
        minimap_height = self.consts.MINIMAP_DISPLAY_HEIGHT

        # Draw detailed floor layers
        floors_count = 4  # Ground, Floor 2, Floor 3, Roof
        floor_stripe_height = minimap_height // floors_count  # Should be 4 pixels per floor with 16px total

        # Floor layer colors in corrected order: tan, yellow, dark_tan, green
        floor_colors = [
            jnp.array(self.consts.MINIMAP_GREEN, dtype=jnp.uint8),
            jnp.array(self.consts.MINIMAP_DARK_TAN, dtype=jnp.uint8),
            jnp.array(self.consts.MINIMAP_YELLOW, dtype=jnp.uint8),
            jnp.array(self.consts.MINIMAP_TAN, dtype=jnp.uint8)
        ]

        # Roof colors (only tan and yellow)
        roof_colors = [
            jnp.array(self.consts.MINIMAP_TAN, dtype=jnp.uint8),
            jnp.array(self.consts.MINIMAP_YELLOW, dtype=jnp.uint8)
        ]

        # Draw floors in reverse order to match visual layout (roof at top, ground at bottom)
        # Visual layout: floor_idx 0=roof(top), 1=top_floor, 2=middle_floor, 3=ground(bottom)
        for visual_floor_idx in range(floors_count):
            floor_y = minimap_display_y + visual_floor_idx * floor_stripe_height

            if visual_floor_idx == 0:  # Roof (top of minimap) - only 2 layers (tan, yellow)
                layer_height = max(1, floor_stripe_height // 4)  # Same layer height as regular floors
                # Draw layers shifted down by 2 pixels to eliminate gap
                for layer_idx in range(2):
                    layer_y = floor_y + layer_idx * layer_height + 2  # Shift down by 2 pixels
                    game_area = draw_rectangle_simple(game_area, minimap_display_x, layer_y,
                                                    minimap_width, layer_height,
                                                    roof_colors[layer_idx])
            else:  # Regular floors (1,2,3) - ALL get 4 layers each (tan, yellow, dark_tan, green)
                layer_height = max(1, floor_stripe_height // 4)
                # Draw layers from top to bottom: tan, yellow, dark_tan, green
                for layer_idx in range(4):
                    layer_y = floor_y + layer_idx * layer_height
                    game_area = draw_rectangle_simple(game_area, minimap_display_x, layer_y,
                                                    minimap_width, layer_height,
                                                    floor_colors[layer_idx])

        def world_to_minimap_x(world_x):
            """Convert world X coordinate to minimap X coordinate"""
            return jnp.clip(
                jnp.floor(world_x * minimap_width / self.consts.TOTAL_BUILDING_WIDTH).astype(jnp.int32) + minimap_display_x,
                minimap_display_x, minimap_display_x + minimap_width - 1
            )

        def floor_to_minimap_y(floor_index):
            """Convert floor index to minimap Y coordinate (roof=0, ground=3)"""
            return minimap_display_y + (3 - floor_index) * floor_stripe_height + floor_stripe_height // 2

        # Draw escalators (diagonal staircase sprites in black)
        escalator_color = jnp.array([0, 0, 0], dtype=jnp.uint8)  # Black

        # Floor 1: escalator on leftmost side
        floor1_y = floor_to_minimap_y(0)
        escalator_x_left = minimap_display_x + 2  # Leftmost position
        # Draw diagonal line pattern (going up-right)
        for i in range(3):
            game_area = draw_rectangle_simple(game_area, escalator_x_left + i, floor1_y - 1 + i, 1, 1, escalator_color)

        # Floor 2: escalator on rightmost side (flipped vertically)
        floor2_y = floor_to_minimap_y(1)
        escalator_x_right = minimap_display_x + minimap_width - 5  # Rightmost position
        # Draw diagonal line pattern (going down-right, flipped vertically)
        for i in range(3):
            game_area = draw_rectangle_simple(game_area, escalator_x_right + i, floor2_y + 1 - i, 1, 1, escalator_color)

        # Floor 3: escalator on leftmost side
        floor3_y = floor_to_minimap_y(2)
        escalator_x_left = minimap_display_x + 2  # Leftmost position
        # Draw diagonal line pattern (going up-right)
        for i in range(3):
            game_area = draw_rectangle_simple(game_area, escalator_x_left + i, floor3_y - 1 + i, 1, 1, escalator_color)

        # Draw elevator (black vertical line at center, moving between floors)
        elevator_world_x = self.consts.ELEVATOR_BUILDING_X + self.consts.ELEVATOR_WIDTH // 2
        elevator_minimap_x = world_to_minimap_x(elevator_world_x)

        # elevator_minimap_x = minimap_width // 2  # Center of minimap
        elevator_floor_y = floor_to_minimap_y(state.elevator.floor)
        elevator_color = jnp.array([0, 0, 0], dtype=jnp.uint8)  # Black
        # Vertical line for elevator
        game_area = draw_rectangle_simple(game_area, elevator_minimap_x, elevator_floor_y - 2, 2, 4, elevator_color)

        # Draw cop (black marker)
        cop_minimap_x = world_to_minimap_x(state.player.x)
        cop_floor_y = floor_to_minimap_y(state.player.floor)
        cop_color = jnp.array([0, 0, 0], dtype=jnp.uint8)  # Black
        game_area = draw_rectangle_simple(game_area, cop_minimap_x, cop_floor_y - 1, 2, 2, cop_color)

        # Draw robber (white marker)
        robber_minimap_x = world_to_minimap_x(state.thief.x)
        robber_floor_y = floor_to_minimap_y(state.thief.floor)
        robber_color = jnp.array([255, 255, 255], dtype=jnp.uint8)  # White
        game_area = draw_rectangle_simple(game_area, robber_minimap_x, robber_floor_y - 1, 2, 2, robber_color)

        # Place game area into the bordered frame at offset position
        frame = frame.at[
            self.consts.GAME_AREA_OFFSET_Y:self.consts.GAME_AREA_OFFSET_Y + self.consts.GAME_AREA_HEIGHT,
            self.consts.GAME_AREA_OFFSET_X:self.consts.GAME_AREA_OFFSET_X + self.consts.GAME_AREA_WIDTH
        ].set(game_area)

        # Add Activision logo to the bottom black border area (left side)
        activision_logo = self.sprites['activision_logo']
        logo_height, logo_width = activision_logo.shape[:2]

        # Position logo on the left side of the bottom black border
        bottom_border_start_y = self.consts.GAME_AREA_OFFSET_Y + self.consts.GAME_AREA_HEIGHT
        bottom_border_height = self.consts.TOTAL_SCREEN_HEIGHT - bottom_border_start_y

        # Position logo on the left side with some margin, near the top of bottom border
        logo_x = 20  # Small margin from left edge
        logo_y = bottom_border_start_y + 5  # Small margin from top of bottom border

        # Apply logo to frame
        frame = frame.at[
            logo_y:logo_y + logo_height,
            logo_x:logo_x + logo_width
        ].set(activision_logo)

        return frame
