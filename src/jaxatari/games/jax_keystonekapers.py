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
    GAME_AREA_HEIGHT: int = 175     # Reduced to end right after minimap area

    # Border offsets to match original layout
    GAME_AREA_OFFSET_X: int = 8     # Left border (keep same)
    GAME_AREA_OFFSET_Y: int = 30    # Top border (keep same)
    # Bottom border will be: 250 - 30 - 175 = 45 pixels (bigger bottom border)

    # Legacy constants for compatibility
    SCREEN_WIDTH: int = 152         # Match game area
    SCREEN_HEIGHT: int = 175        # Match game area

    # Building structure - 7 sections of horizontal scrolling
    BUILDING_SECTIONS: int = 7
    SECTION_WIDTH: int = 160  # Each section is one screen width
    TOTAL_BUILDING_WIDTH: int = 7 * 160  # 1120 pixels total width

    # Floor positions (Y coordinates) - shifted up further to eliminate blue gap above minimap
    FLOOR_1_Y: int = 135  # Ground floor (was 140, shifted up by 5 more to close gap)
    FLOOR_2_Y: int = 105   # Middle floor (was 100, shifted up by 5)
    FLOOR_3_Y: int = 70   # Top floor (was 60, shifted up by 5)
    ROOF_Y: int = 40      # Roof (was 19, shifted up by 4)
    FLOOR_HEIGHT: int = 20

    # Minimap area configuration (at bottom of game area)
    MINIMAP_HEIGHT: int = 20
    MINIMAP_COLOR: tuple = (151, 151, 151)  # #979797 in RGB

    # Escalator positions (relative to each section, 2 per section)
    ESCALATOR_1_OFFSET: int = 40   # Left escalator in each section
    ESCALATOR_2_OFFSET: int = 120  # Right escalator in each section
    ESCALATOR_WIDTH: int = 16

    # Elevator configuration (positioned in middle of entire building)
    ELEVATOR_BUILDING_X: int = 560  # Middle of 1120px building (3.5 * 160)
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
    THIEF_BASE_SPEED: float = 1.0
    THIEF_SPEED_SCALE: float = 0.10
    THIEF_MAX_SPEED: float = 3.0

    # Obstacle configurations
    CART_WIDTH: int = 12
    CART_HEIGHT: int = 8
    CART_BASE_SPEED: float = 1.5
    CART_SPEED_SCALE: float = 0.08
    CART_MIN_SPAWN_INTERVAL: float = 1.5  # seconds
    CART_MAX_SPAWN_INTERVAL: float = 3.0

    BALL_WIDTH: int = 6
    BALL_HEIGHT: int = 6
    BALL_BASE_SPEED: float = 2.0
    BALL_BOUNCE_HEIGHT: int = 20
    BALL_MIN_SPAWN_INTERVAL: float = 2.0
    BALL_MAX_SPAWN_INTERVAL: float = 4.0

    PLANE_WIDTH: int = 16
    PLANE_HEIGHT: int = 8
    PLANE_BASE_SPEED: float = 3.0
    PLANE_SPEED_SCALE: float = 0.12
    PLANE_MIN_SPAWN_INTERVAL: float = 3.0
    PLANE_MAX_SPAWN_INTERVAL: float = 6.0

    # Collectible configurations
    ITEM_WIDTH: int = 8
    ITEM_HEIGHT: int = 8

    # Timing and scoring
    BASE_TIMER: int = 3600  # 60 seconds at 60fps
    TIMER_REDUCTION_PER_LEVEL: int = 300  # 5 seconds
    COLLISION_PENALTY: int = 300  # 5 seconds
    CATCH_THIEF_POINTS: int = 3000
    TIME_BONUS_MULTIPLIER: int = 50
    ITEM_POINTS: int = 100
    JUMP_POINTS: int = 50
    EXTRA_LIFE_THRESHOLD: int = 10000

    # Difficulty scaling
    OBSTACLE_SPAWN_SCALE: float = 0.12
    MAX_OBSTACLES: int = 8
    MAX_ITEMS: int = 4

    # Colors (RGB)
    BACKGROUND_COLOR: Tuple[int, int, int] = (0, 0, 128)  # Dark blue
    PLAYER_COLOR: Tuple[int, int, int] = (0, 255, 0)      # Green
    THIEF_COLOR: Tuple[int, int, int] = (255, 255, 0)     # Yellow
    FLOOR_COLOR: Tuple[int, int, int] = (139, 69, 19)     # Brown
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
    is_on_elevator: chex.Array
    is_on_escalator: chex.Array
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

    # Items and collectibles
    item_x: chex.Array          # Shape: (MAX_ITEMS,)
    item_y: chex.Array
    item_active: chex.Array
    item_type: chex.Array

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
            floor == 0, self.consts.FLOOR_1_Y,
            jax.lax.select(
                floor == 1, self.consts.FLOOR_2_Y,
                jax.lax.select(
                    floor == 2, self.consts.FLOOR_3_Y,
                    self.consts.ROOF_Y
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

        # Handle both scalar actions (from JAX environment) and array actions
        # Convert scalar action to movement flags
        action_int = jnp.asarray(action, dtype=jnp.int32)

        # Extract action components - handle both scalar and array inputs
        move_left = jnp.logical_or(
            jnp.logical_or(action_int == Action.LEFT, action_int == Action.UPLEFT),
            jnp.logical_or(action_int == Action.DOWNLEFT, action_int == Action.LEFTFIRE)
        )
        move_right = jnp.logical_or(
            jnp.logical_or(action_int == Action.RIGHT, action_int == Action.UPRIGHT),
            jnp.logical_or(action_int == Action.DOWNRIGHT, action_int == Action.RIGHTFIRE)
        )
        move_up = jnp.logical_or(
            jnp.logical_or(action_int == Action.UP, action_int == Action.UPLEFT),
            action_int == Action.UPRIGHT
        )
        move_down = jnp.logical_or(
            jnp.logical_or(action_int == Action.DOWN, action_int == Action.DOWNLEFT),
            action_int == Action.DOWNRIGHT
        )
        jump = jnp.logical_or(
            jnp.logical_or(action_int == Action.FIRE, action_int == Action.LEFTFIRE),
            action_int == Action.RIGHTFIRE
        )
        crouch = move_down  # Crouch is triggered by down input when not moving vertically

        # Horizontal movement with building traversal
        vel_x = jnp.where(
            move_left, -self.consts.PLAYER_SPEED,
            jnp.where(move_right, self.consts.PLAYER_SPEED, 0)
        )

        # Update X position with wrapping at building edges
        new_x = player.x + vel_x
        new_x = jnp.clip(new_x, 0, self.consts.TOTAL_BUILDING_WIDTH - self.consts.PLAYER_WIDTH)

        # Jumping mechanics with gravity
        start_jump = jnp.logical_and(jump, jnp.logical_not(player.is_jumping))
        continue_jump = jnp.logical_and(player.is_jumping, player.jump_timer > 0)

        # Jump physics - parabolic arc
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

        # Y position with jumping
        floor_y = self._floor_y_position(player.floor)
        jump_y = jnp.where(
            new_is_jumping,
            new_jump_start_y - jump_height,
            floor_y
        )

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
        final_x = jnp.where(new_in_elevator, elevator_x, new_x)

        # Escalator movement (only when not jumping and not in elevator)
        current_floor = player.floor
        on_escalator = self._is_on_escalator(final_x, current_floor)
        can_use_escalator = jnp.logical_and(
            jnp.logical_and(on_escalator, jnp.logical_not(new_is_jumping)),
            jnp.logical_not(new_in_elevator)
        )

        escalator_up = jnp.logical_and(
            jnp.logical_and(can_use_escalator, move_up),
            current_floor < 2  # Can't go above floor 3 via escalator
        )
        escalator_down = jnp.logical_and(
            jnp.logical_and(can_use_escalator, move_down),
            current_floor > 0  # Can't go below ground floor
        )

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

        # Update Y position based on final floor
        final_y = jnp.where(
            new_is_jumping, jump_y,
            self._floor_y_position(new_floor)
        )

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
            is_on_elevator=on_elevator,
            is_on_escalator=on_escalator,
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

        # Check escalator usage
        on_escalator = self._is_on_escalator(new_x, current_floor)
        can_use_escalator = jnp.logical_and(on_escalator, jnp.logical_not(is_jumping))

        escalator_up = jnp.logical_and(
            jnp.logical_and(can_use_escalator, move_up),
            current_floor < 2  # Can't go above floor 2 via escalator
        )
        escalator_down = jnp.logical_and(
            jnp.logical_and(can_use_escalator, move_down),
            current_floor > 0  # Can't go below floor 0
        )

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
            x=new_x,
            y=new_y,
            floor=new_floor,
            vel_x=vel_x,
            vel_y=0,  # Simplified for now
            is_jumping=is_jumping,
            jump_timer=jump_timer,
            is_on_elevator=on_elevator,
            is_on_escalator=on_escalator
        )

    @partial(jax.jit, static_argnums=(0,))
    @partial(jax.jit, static_argnums=(0,))
    def _update_thief(self, state: GameState) -> ThiefState:
        """Update thief state with AI behavior for building traversal."""
        thief = state.thief

        # Level-scaled speed
        level_speed = self.consts.THIEF_BASE_SPEED * (
            1.0 + self.consts.THIEF_SPEED_SCALE * state.level
        )
        level_speed = jnp.minimum(level_speed, self.consts.THIEF_MAX_SPEED)

        # Thief AI: Move horizontally across building, then up via escalator
        # Pattern: left->right->left->escalator up->repeat

        # Move horizontally
        new_x = thief.x + level_speed * thief.direction

        # Check for wrapping at building edges
        hit_left_edge = new_x <= 0
        hit_right_edge = new_x >= self.consts.TOTAL_BUILDING_WIDTH - self.consts.THIEF_WIDTH

        # Reverse direction at edges
        new_direction = jax.lax.select(
            jnp.logical_or(hit_left_edge, hit_right_edge),
            -thief.direction,
            thief.direction
        )

        # Clamp position within building
        new_x = jnp.clip(new_x, 0, self.consts.TOTAL_BUILDING_WIDTH - self.consts.THIEF_WIDTH)

        # Check for escalator usage (randomly use escalators when encountered)
        # Only use escalators when moving in certain direction
        section_x = new_x % self.consts.SECTION_WIDTH
        on_escalator_1 = jnp.logical_and(
            section_x >= self.consts.ESCALATOR_1_OFFSET,
            section_x <= self.consts.ESCALATOR_1_OFFSET + self.consts.ESCALATOR_WIDTH
        )
        on_escalator_2 = jnp.logical_and(
            section_x >= self.consts.ESCALATOR_2_OFFSET,
            section_x <= self.consts.ESCALATOR_2_OFFSET + self.consts.ESCALATOR_WIDTH
        )
        on_escalator = jnp.logical_or(on_escalator_1, on_escalator_2)

        # Use escalator with some probability when moving right and can go up
        should_use_escalator = jnp.logical_and(
            jnp.logical_and(on_escalator, new_direction > 0),
            jnp.logical_and(thief.floor < 2, state.step_counter % 120 == 0)  # Every 2 seconds
        )

        # Check escape condition (reached roof)
        escaped = thief.floor >= 3

        # Try to reach roof via escalator when on top floor (but only at building edges)
        at_building_edge = jnp.logical_or(
            new_x <= self.consts.THIEF_WIDTH,  # Left edge
            new_x >= self.consts.TOTAL_BUILDING_WIDTH - 2 * self.consts.THIEF_WIDTH  # Right edge
        )
        try_escape = jnp.logical_and(
            jnp.logical_and(thief.floor == 2, at_building_edge),
            jnp.logical_and(on_escalator, state.step_counter % 240 == 0)  # Only every 4 seconds
        )

        # Update floor if using escalator
        new_floor = jax.lax.select(
            try_escape, 3,  # Go to roof (escape)
            jax.lax.select(should_use_escalator, thief.floor + 1, thief.floor)
        )

        # Update Y position based on floor
        new_y = self._floor_y_position(new_floor)

        return ThiefState(
            x=new_x,
            y=new_y,
            floor=new_floor,
            speed=level_speed,
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

        # Find first inactive slot for obstacle type
        if obstacle_type == 0:  # Shopping cart
            inactive_mask = jnp.logical_not(obstacles.cart_active)
            spawn_idx = jnp.argmax(inactive_mask)
            should_spawn = inactive_mask[spawn_idx]

            # Random spawn parameters
            spawn_x = jrandom.uniform(key, (), minval=0, maxval=self.consts.SCREEN_WIDTH - self.consts.CART_WIDTH)
            spawn_floor = jrandom.randint(key, (), 0, 3)  # Floors 1-3
            spawn_y = self._floor_y_position(spawn_floor)
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

            return obstacles._replace(
                cart_x=new_cart_x,
                cart_y=new_cart_y,
                cart_active=new_cart_active,
                cart_speed=new_cart_speed
            )

        # Similar logic for balls and planes...
        return obstacles

    @partial(jax.jit, static_argnums=(0,))
    def _update_obstacles(self, state: GameState, key: chex.PRNGKey) -> ObstacleState:
        """Update all obstacles including movement and spawning."""
        obstacles = state.obstacles

        # Update shopping carts
        new_cart_x = obstacles.cart_x + obstacles.cart_speed * obstacles.cart_active
        cart_out_of_bounds = jnp.logical_or(
            new_cart_x < -self.consts.CART_WIDTH,
            new_cart_x > self.consts.SCREEN_WIDTH
        )
        new_cart_active = jnp.logical_and(obstacles.cart_active, jnp.logical_not(cart_out_of_bounds))

        # Update bouncing balls
        new_ball_x = obstacles.ball_x + obstacles.ball_vel_x * obstacles.ball_active
        new_ball_y = obstacles.ball_y + obstacles.ball_vel_y * obstacles.ball_active

        # Ball bouncing logic
        ball_hit_floor = new_ball_y >= self._floor_y_position(0) + self.consts.FLOOR_HEIGHT
        new_ball_vel_y = jnp.where(
            ball_hit_floor,
            -jnp.abs(obstacles.ball_vel_y),
            obstacles.ball_vel_y + 0.5  # Gravity
        )

        ball_out_of_bounds = jnp.logical_or(
            new_ball_x < -self.consts.BALL_WIDTH,
            new_ball_x > self.consts.SCREEN_WIDTH
        )
        new_ball_active = jnp.logical_and(obstacles.ball_active, jnp.logical_not(ball_out_of_bounds))

        # Update toy planes
        new_plane_x = obstacles.plane_x + obstacles.plane_speed * obstacles.plane_active
        plane_out_of_bounds = jnp.logical_or(
            new_plane_x < -self.consts.PLANE_WIDTH,
            new_plane_x > self.consts.SCREEN_WIDTH
        )
        new_plane_active = jnp.logical_and(obstacles.plane_active, jnp.logical_not(plane_out_of_bounds))

        # Update spawn timers
        new_cart_spawn_timer = jnp.maximum(obstacles.cart_spawn_timer - 1, 0)
        new_ball_spawn_timer = jnp.maximum(obstacles.ball_spawn_timer - 1, 0)
        new_plane_spawn_timer = jnp.maximum(obstacles.plane_spawn_timer - 1, 0)

        return ObstacleState(
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

            plane_x=new_plane_x,
            plane_y=obstacles.plane_y,
            plane_active=new_plane_active,
            plane_speed=obstacles.plane_speed,
            plane_spawn_timer=new_plane_spawn_timer
        )

    @partial(jax.jit, static_argnums=(0,))
    def _check_collisions(self, state: GameState) -> Tuple[bool, bool, int]:
        """Check all collision types with proper hitbox handling for jump/crouch."""
        player = state.player

        # Player hitbox depends on crouching state
        player_height = jax.lax.select(
            player.is_crouching,
            self.consts.PLAYER_CROUCH_HEIGHT,
            self.consts.PLAYER_HEIGHT
        )

        # Player-thief collision
        thief_collision = self._entities_collide(
            player.x, player.y, self.consts.PLAYER_WIDTH, player_height,
            state.thief.x, state.thief.y, self.consts.THIEF_WIDTH, self.consts.THIEF_HEIGHT
        )
        thief_caught = jnp.logical_and(thief_collision, player.floor == state.thief.floor)

        # Player-obstacle collisions with jump/crouch mechanics
        # Check cart collisions (ground level, can crouch under)
        cart_collisions = jax.vmap(
            lambda i: jnp.logical_and(
                state.obstacles.cart_active[i],
                self._entities_collide(
                    player.x, player.y, self.consts.PLAYER_WIDTH, player_height,
                    state.obstacles.cart_x[i], state.obstacles.cart_y[i],
                    self.consts.CART_WIDTH, self.consts.CART_HEIGHT
                )
            )
        )(jnp.arange(self.consts.MAX_OBSTACLES))

        # Check ball collisions (can jump over)
        ball_collisions = jax.vmap(
            lambda i: jnp.logical_and(
                state.obstacles.ball_active[i],
                self._entities_collide(
                    player.x, player.y, self.consts.PLAYER_WIDTH, player_height,
                    state.obstacles.ball_x[i], state.obstacles.ball_y[i],
                    self.consts.BALL_WIDTH, self.consts.BALL_HEIGHT
                )
            )
        )(jnp.arange(self.consts.MAX_OBSTACLES))

        # Check plane collisions (fly overhead, can crouch under)
        plane_collisions = jax.vmap(
            lambda i: jnp.logical_and(
                state.obstacles.plane_active[i],
                self._entities_collide(
                    player.x, player.y, self.consts.PLAYER_WIDTH, player_height,
                    state.obstacles.plane_x[i], state.obstacles.plane_y[i],
                    self.consts.PLANE_WIDTH, self.consts.PLANE_HEIGHT
                )
            )
        )(jnp.arange(self.consts.MAX_OBSTACLES))

        # Combine all obstacle collisions
        all_obstacle_collisions = jnp.logical_or(
            jnp.any(cart_collisions),
            jnp.logical_or(jnp.any(ball_collisions), jnp.any(plane_collisions))
        )

        # Only count obstacle hits if player is not jumping (jumping avoids most obstacles)
        # Exception: planes can still hit if player jumps too high
        obstacle_hit = jnp.logical_and(jnp.logical_not(player.is_jumping), all_obstacle_collisions)

        # Item collection
        item_collisions = jax.vmap(
            lambda i: jnp.logical_and(
                state.item_active[i],
                self._entities_collide(
                    player.x, player.y, self.consts.PLAYER_WIDTH, self.consts.PLAYER_HEIGHT,
                    state.item_x[i], state.item_y[i],
                    self.consts.ITEM_WIDTH, self.consts.ITEM_HEIGHT
                )
            )
        )(jnp.arange(self.consts.MAX_ITEMS))
        items_collected = jnp.sum(item_collisions)

        return obstacle_hit, thief_caught, items_collected

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey = None) -> Tuple[KeystoneKapersObservation, GameState]:
        """Reset the game to initial state with building traversal setup."""
        if key is None:
            key = jrandom.PRNGKey(0)

        # Initialize player at rightmost section of ground floor (as per original game)
        player_start_x = self.consts.TOTAL_BUILDING_WIDTH - 100  # Near right edge
        player = PlayerState(
            x=jnp.array(player_start_x),
            y=jnp.array(self.consts.FLOOR_1_Y),
            floor=jnp.array(0),
            vel_x=jnp.array(0),
            vel_y=jnp.array(0),
            is_jumping=jnp.array(False),
            is_crouching=jnp.array(False),
            jump_timer=jnp.array(0),
            jump_start_y=jnp.array(self.consts.FLOOR_1_Y),
            is_on_elevator=jnp.array(False),
            is_on_escalator=jnp.array(False),
            in_elevator=jnp.array(False)
        )

        # Initialize thief at leftmost section of top floor
        thief_start_x = 50  # Near left edge of building
        thief = ThiefState(
            x=jnp.array(thief_start_x),
            y=jnp.array(self.consts.FLOOR_3_Y),
            floor=jnp.array(2),
            speed=jnp.array(self.consts.THIEF_BASE_SPEED),
            direction=jnp.array(1),  # Moving right initially
            escaped=jnp.array(False)
        )

        # Initialize empty obstacles
        obstacles = ObstacleState(
            cart_x=jnp.zeros(self.consts.MAX_OBSTACLES),
            cart_y=jnp.zeros(self.consts.MAX_OBSTACLES),
            cart_active=jnp.zeros(self.consts.MAX_OBSTACLES, dtype=bool),
            cart_speed=jnp.zeros(self.consts.MAX_OBSTACLES),
            cart_spawn_timer=jnp.array(60),  # 1 second

            ball_x=jnp.zeros(self.consts.MAX_OBSTACLES),
            ball_y=jnp.zeros(self.consts.MAX_OBSTACLES),
            ball_active=jnp.zeros(self.consts.MAX_OBSTACLES, dtype=bool),
            ball_vel_x=jnp.zeros(self.consts.MAX_OBSTACLES),
            ball_vel_y=jnp.zeros(self.consts.MAX_OBSTACLES),
            ball_spawn_timer=jnp.array(120),  # 2 seconds

            plane_x=jnp.zeros(self.consts.MAX_OBSTACLES),
            plane_y=jnp.zeros(self.consts.MAX_OBSTACLES),
            plane_active=jnp.zeros(self.consts.MAX_OBSTACLES, dtype=bool),
            plane_speed=jnp.zeros(self.consts.MAX_OBSTACLES),
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
        camera_x = player_section * self.consts.SECTION_WIDTH

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

            item_x=jnp.zeros(self.consts.MAX_ITEMS),
            item_y=jnp.zeros(self.consts.MAX_ITEMS),
            item_active=jnp.zeros(self.consts.MAX_ITEMS, dtype=bool),
            item_type=jnp.zeros(self.consts.MAX_ITEMS),

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

        # True section-based camera system
        # Camera always shows the section that the player is currently in
        # Section boundaries are exact: 0-159 = section 0, 160-319 = section 1, etc.

        # Calculate which section the player is currently in
        player_section = new_player.x // self.consts.SECTION_WIDTH

        # Camera always shows the player's current section (no thresholds, no delays)
        new_camera_x = player_section * self.consts.SECTION_WIDTH

        # Clamp camera to valid building bounds
        max_camera_x = self.consts.TOTAL_BUILDING_WIDTH - self.consts.SECTION_WIDTH
        new_camera_x = jnp.clip(new_camera_x, 0, max_camera_x)

        # Check collisions
        obstacle_hit, thief_caught, items_collected = self._check_collisions(state._replace(
            player=new_player,
            thief=new_thief,
            obstacles=new_obstacles
        ))

        # Update timer and game state
        timer_penalty = jax.lax.select(obstacle_hit, self.consts.COLLISION_PENALTY, 0)
        new_timer = jnp.maximum(state.timer - 1 - timer_penalty, 0)

        # Update score
        thief_points = jax.lax.select(thief_caught, self.consts.CATCH_THIEF_POINTS, 0)
        time_bonus = jax.lax.select(
            thief_caught,
            new_timer * self.consts.TIME_BONUS_MULTIPLIER // 60,  # Convert to time bonus
            0
        )
        item_points = items_collected * self.consts.ITEM_POINTS
        jump_points = jax.lax.select(
            jnp.logical_and(new_player.is_jumping, jnp.logical_not(state.player.is_jumping)),
            self.consts.JUMP_POINTS,
            0
        )

        score_increase = thief_points + time_bonus + item_points + jump_points
        new_score = state.score + score_increase

        # Check game end conditions
        time_up = new_timer <= 0
        thief_escaped = new_thief.escaped
        level_complete = thief_caught

        # Update lives
        life_lost = jnp.logical_or(time_up, thief_escaped)
        new_lives = jax.lax.select(life_lost, state.lives - 1, state.lives)

        # Game over condition
        game_over = jnp.logical_or(new_lives <= 0, level_complete)

        # Deactivate collected items
        new_item_active = state.item_active  # TODO: Implement item collection logic

        # Create new state with camera update
        new_state = GameState(
            player=new_player,
            thief=new_thief,
            obstacles=new_obstacles,
            elevator=new_elevator,

            # Update camera position
            camera_x=new_camera_x,

            score=new_score,
            lives=new_lives,
            level=state.level,  # Level progression happens in reset for next level
            timer=new_timer,
            step_counter=state.step_counter + 1,

            item_x=state.item_x,
            item_y=state.item_y,
            item_active=new_item_active,
            item_type=state.item_type,

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
            obs.elevator_is_open.astype(jnp.float32).flatten()
        ])

    def action_space(self) -> spaces.Discrete:
        """Return the action space."""
        return spaces.Discrete(len(self.action_set))

    def observation_space(self) -> spaces.Dict:
        """Return the observation space."""
        return spaces.Dict({
            "player_x": spaces.Box(low=0, high=self.consts.SCREEN_WIDTH, shape=(), dtype=jnp.int32),
            "player_y": spaces.Box(low=0, high=self.consts.SCREEN_HEIGHT, shape=(), dtype=jnp.int32),
            "player_floor": spaces.Box(low=0, high=3, shape=(), dtype=jnp.int32),
            "thief_x": spaces.Box(low=0, high=self.consts.SCREEN_WIDTH, shape=(), dtype=jnp.int32),
            "thief_y": spaces.Box(low=0, high=self.consts.SCREEN_HEIGHT, shape=(), dtype=jnp.int32),
            "thief_floor": spaces.Box(low=0, high=3, shape=(), dtype=jnp.int32),
            "obstacle_positions": spaces.Box(
                low=-1, high=max(self.consts.SCREEN_WIDTH, self.consts.SCREEN_HEIGHT),
                shape=(self.consts.MAX_OBSTACLES * 3, 3), dtype=jnp.int32
            ),
            "item_positions": spaces.Box(
                low=-1, high=max(self.consts.SCREEN_WIDTH, self.consts.SCREEN_HEIGHT),
                shape=(self.consts.MAX_ITEMS, 3), dtype=jnp.int32
            ),
            "score": spaces.Box(low=0, high=999999, shape=(), dtype=jnp.int32),
            "timer": spaces.Box(low=0, high=self.consts.BASE_TIMER, shape=(), dtype=jnp.int32),
            "level": spaces.Box(low=1, high=99, shape=(), dtype=jnp.int32),
            "lives": spaces.Box(low=0, high=9, shape=(), dtype=jnp.int32),
            "elevator_position": spaces.Box(low=0, high=2, shape=(), dtype=jnp.int32),
            "elevator_is_open": spaces.Box(low=0, high=1, shape=(), dtype=jnp.bool_)
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
        return jnp.logical_or(
            state.game_over,
            jnp.logical_or(state.timer <= 0, state.thief.escaped)
        )

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
        """Create simple colored rectangle sprites."""
        sprites = {}

        # Create simple colored rectangles for each entity
        sprites['player'] = jnp.ones((self.consts.PLAYER_HEIGHT, self.consts.PLAYER_WIDTH, 3), dtype=jnp.uint8) * jnp.array(self.consts.PLAYER_COLOR, dtype=jnp.uint8)
        sprites['thief'] = jnp.ones((self.consts.THIEF_HEIGHT, self.consts.THIEF_WIDTH, 3), dtype=jnp.uint8) * jnp.array(self.consts.THIEF_COLOR, dtype=jnp.uint8)
        sprites['cart'] = jnp.ones((self.consts.CART_HEIGHT, self.consts.CART_WIDTH, 3), dtype=jnp.uint8) * jnp.array(self.consts.CART_COLOR, dtype=jnp.uint8)
        sprites['ball'] = jnp.ones((self.consts.BALL_HEIGHT, self.consts.BALL_WIDTH, 3), dtype=jnp.uint8) * jnp.array(self.consts.BALL_COLOR, dtype=jnp.uint8)
        sprites['plane'] = jnp.ones((self.consts.PLANE_HEIGHT, self.consts.PLANE_WIDTH, 3), dtype=jnp.uint8) * jnp.array(self.consts.PLANE_COLOR, dtype=jnp.uint8)
        sprites['item'] = jnp.ones((self.consts.ITEM_HEIGHT, self.consts.ITEM_WIDTH, 3), dtype=jnp.uint8) * jnp.array(self.consts.ITEM_COLOR, dtype=jnp.uint8)

        return sprites

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

        # Draw floors (simplified - just draw visible game area width)
        floor_color = jnp.array(self.consts.FLOOR_COLOR, dtype=jnp.uint8)
        floor_thickness = 4

        # Floor 1
        game_area = draw_rectangle_simple(
            game_area, 0, self.consts.FLOOR_1_Y + self.consts.FLOOR_HEIGHT - floor_thickness,
            self.consts.GAME_AREA_WIDTH, floor_thickness, floor_color
        )

        # Floor 2
        game_area = draw_rectangle_simple(
            game_area, 0, self.consts.FLOOR_2_Y + self.consts.FLOOR_HEIGHT - floor_thickness,
            self.consts.GAME_AREA_WIDTH, floor_thickness, floor_color
        )

        # Floor 3
        game_area = draw_rectangle_simple(
            game_area, 0, self.consts.FLOOR_3_Y + self.consts.FLOOR_HEIGHT - floor_thickness,
            self.consts.GAME_AREA_WIDTH, floor_thickness, floor_color
        )

        # Roof (4th level)
        game_area = draw_rectangle_simple(
            game_area, 0, self.consts.ROOF_Y + self.consts.FLOOR_HEIGHT - floor_thickness,
            self.consts.GAME_AREA_WIDTH, floor_thickness, floor_color
        )

        # Draw escalators (only those visible on screen)
        escalator_color = jnp.array(self.consts.ESCALATOR_COLOR, dtype=jnp.uint8)

        # Check if escalators are visible and draw them
        escalator_1_screen_x = building_to_screen_x(self.consts.ESCALATOR_1_OFFSET)
        escalator_1_visible = (escalator_1_screen_x >= -self.consts.ESCALATOR_WIDTH) & (escalator_1_screen_x < self.consts.GAME_AREA_WIDTH)

        # Draw escalator 1 if visible
        game_area = jnp.where(
            escalator_1_visible,
            draw_rectangle_simple(game_area, escalator_1_screen_x, self.consts.FLOOR_2_Y,
                                self.consts.ESCALATOR_WIDTH, self.consts.FLOOR_HEIGHT, escalator_color),
            game_area
        )
        game_area = jnp.where(
            escalator_1_visible,
            draw_rectangle_simple(game_area, escalator_1_screen_x, self.consts.FLOOR_3_Y,
                                self.consts.ESCALATOR_WIDTH, self.consts.FLOOR_HEIGHT, escalator_color),
            game_area
        )

        escalator_2_screen_x = building_to_screen_x(self.consts.ESCALATOR_2_OFFSET)
        escalator_2_visible = (escalator_2_screen_x >= -self.consts.ESCALATOR_WIDTH) & (escalator_2_screen_x < self.consts.GAME_AREA_WIDTH)

        # Draw escalator 2 if visible
        game_area = jnp.where(
            escalator_2_visible,
            draw_rectangle_simple(game_area, escalator_2_screen_x, self.consts.FLOOR_2_Y,
                                self.consts.ESCALATOR_WIDTH, self.consts.FLOOR_HEIGHT, escalator_color),
            game_area
        )
        game_area = jnp.where(
            escalator_2_visible,
            draw_rectangle_simple(game_area, escalator_2_screen_x, self.consts.FLOOR_3_Y,
                                self.consts.ESCALATOR_WIDTH, self.consts.FLOOR_HEIGHT, escalator_color),
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

        # Draw player with camera adjustment
        player_screen_x = building_to_screen_x(state.player.x)
        player_visible = (player_screen_x >= -self.consts.PLAYER_WIDTH) & (player_screen_x < self.consts.GAME_AREA_WIDTH)

        player_color = jnp.array([0, 255, 0], dtype=jnp.uint8)  # Green player
        game_area = jnp.where(
            player_visible,
            draw_rectangle_simple(game_area, player_screen_x, state.player.y,
                                self.consts.PLAYER_WIDTH, self.consts.PLAYER_HEIGHT, player_color),
            game_area
        )

        # Draw thief with camera adjustment (only if not escaped)
        thief_screen_x = building_to_screen_x(state.thief.x)
        thief_visible = (thief_screen_x >= -self.consts.THIEF_WIDTH) & (thief_screen_x < self.consts.GAME_AREA_WIDTH) & jnp.logical_not(state.thief.escaped)

        thief_color = jnp.array([255, 0, 0], dtype=jnp.uint8)  # Red thief
        game_area = jnp.where(
            thief_visible,
            draw_rectangle_simple(game_area, thief_screen_x, state.thief.y,
                                self.consts.THIEF_WIDTH, self.consts.THIEF_HEIGHT, thief_color),
            game_area
        )

        # Draw simple UI elements on game area
        ui_color = jnp.array([255, 255, 255], dtype=jnp.uint8)  # White UI

        # Score indicator (top left)
        game_area = draw_rectangle_simple(game_area, 10, 10, 30, 8, ui_color)

        # Lives indicator (top center)
        game_area = draw_rectangle_simple(game_area, self.consts.GAME_AREA_WIDTH // 2 - 15, 10, 30, 8, ui_color)

        # Timer indicator (top right)
        game_area = draw_rectangle_simple(game_area, self.consts.GAME_AREA_WIDTH - 40, 10, 30, 8, ui_color)

        # Camera position indicator (above minimap)
        camera_indicator_x = (state.camera_x / self.consts.TOTAL_BUILDING_WIDTH * self.consts.GAME_AREA_WIDTH)
        game_area = draw_rectangle_simple(game_area, camera_indicator_x, self.consts.FLOOR_1_Y + self.consts.FLOOR_HEIGHT + 5, 20, 5,
                                        jnp.array([255, 255, 0], dtype=jnp.uint8))  # Yellow camera indicator

        # Add minimap area right after floor 1 (eliminate blue gap)
        minimap_y_start = self.consts.FLOOR_1_Y + self.consts.FLOOR_HEIGHT  # Position right after floor 1
        minimap_color = jnp.array(self.consts.MINIMAP_COLOR, dtype=jnp.uint8)

        # Draw minimap background directly in game area
        game_area = draw_rectangle_simple(game_area, 0, minimap_y_start,
                                        self.consts.GAME_AREA_WIDTH, self.consts.MINIMAP_HEIGHT,
                                        minimap_color)

        # Place game area into the bordered frame at offset position
        frame = frame.at[
            self.consts.GAME_AREA_OFFSET_Y:self.consts.GAME_AREA_OFFSET_Y + self.consts.GAME_AREA_HEIGHT,
            self.consts.GAME_AREA_OFFSET_X:self.consts.GAME_AREA_OFFSET_X + self.consts.GAME_AREA_WIDTH
        ].set(game_area)

        return frame
