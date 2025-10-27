import pygame
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit
from typing import Tuple, NamedTuple
import chex
from flax import struct
import sys
from functools import partial
from jax import lax
from jaxatari.environment import JaxEnvironment
from jaxatari.renderers import JAXGameRenderer
import jaxatari.spaces as spaces


class BeamRiderConstants(NamedTuple):
    """Container for all game constants"""

    # Screen dimensions
    SCREEN_WIDTH = 160
    SCREEN_HEIGHT = 210

    # Entity dimensions
    SHIP_WIDTH = 16
    SHIP_HEIGHT = 8
    PROJECTILE_WIDTH = 2
    PROJECTILE_HEIGHT = 4
    ENEMY_WIDTH = 8
    ENEMY_HEIGHT = 8

    # Entity limits
    MAX_PROJECTILES = 8
    MAX_ENEMIES = 16

    # Beam system
    NUM_BEAMS = 5
    BEAM_WIDTH = 4

    # Colors (RGB values for rendering)
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)
    GREEN = (0, 255, 0)
    YELLOW = (255, 255, 0)

    # Game mechanics
    PROJECTILE_SPEED = 4.0
    ENEMY_SPEED = 1.0
    ENEMY_SPAWN_INTERVAL = 60  # frames between enemy spawns
    INITIAL_LIVES = 3
    INITIAL_LEVEL = 1
    POINTS_PER_ENEMY = 10

    # Ship positioning
    SHIP_BOTTOM_OFFSET = 20
    INITIAL_BEAM = 2  # Center beam

    # Torpedo system
    TORPEDOES_PER_SECTOR = 3
    TORPEDO_SPEED = 2.0  # Faster than regular projectiles
    TORPEDO_WIDTH = 3
    TORPEDO_HEIGHT = 6

    # Sector progression
    ENEMIES_PER_SECTOR = 15
    BASE_ENEMY_SPAWN_INTERVAL = 90  # Start slower (was 60)
    MIN_ENEMY_SPAWN_INTERVAL = 12  # End faster (was 20)
    MAX_ENEMY_SPEED = 2.5  # Maximum enemy speed at sector 99

    # Enemy spawn position
    ENEMY_SPAWN_Y = 10
    # Enemy types
    ENEMY_TYPE_WHITE_SAUCER = 0
    ENEMY_TYPE_BROWN_DEBRIS = 1
    ENEMY_TYPE_YELLOW_CHIRPER = 2
    ENEMY_TYPE_GREEN_BLOCKER = 3
    ENEMY_TYPE_GREEN_BOUNCE = 4
    ENEMY_TYPE_BLUE_CHARGER = 5
    ENEMY_TYPE_ORANGE_TRACKER = 6
    ENEMY_TYPE_SENTINEL_SHIP = 7
    ENEMY_TYPE_YELLOW_REJUVENATOR = 8
    ENEMY_TYPE_REJUVENATOR_DEBRIS = 9
    # White saucer behavior constants
    WHITE_SAUCER_SHOOT_CHANCE = 0.2  # 20% of white saucers can shoot
    WHITE_SAUCER_JUMP_CHANCE = 0.15  # 15% chance for beam jumping
    WHITE_SAUCER_REVERSE_CHANCE = 0.1  # 10% chance for reverse movement
    WHITE_SAUCER_FIRING_INTERVAL = 90  # Frames between shots
    WHITE_SAUCER_PROJECTILE_SPEED = 2.5  # Speed of white saucer projectiles
    WHITE_SAUCER_JUMP_INTERVAL = 120  # Frames between beam jumps
    WHITE_SAUCER_REVERSE_SPEED = -1.5  # Reverse movement speed (going back up)
    # White saucer movement patterns
    WHITE_SAUCER_STRAIGHT_DOWN = 0
    WHITE_SAUCER_BEAM_JUMP = 1
    WHITE_SAUCER_REVERSE_UP = 2
    WHITE_SAUCER_SHOOTING = 4
    # White saucer reverse pattern constants
    WHITE_SAUCER_REVERSE_TRIGGER_Y = 150  # Y position where reverse pattern triggers
    WHITE_SAUCER_REVERSE_SPEED_FAST = -4.0
    UPPER_THIRD_Y = 70  # Y = 70, upper third boundary
    # White saucer beam change and retreat constants
    WHITE_SAUCER_BEAM_CHANGE_CHANCE = 0.4  # 40% chance to change beam before retreating
    WHITE_SAUCER_RETREAT_BEAM_CHANGE_TIME = 30  # Frames to move to new beam before retreating
    WHITE_SAUCER_RETREAT_AFTER_SHOT = 1  # State flag for retreating after shooting
    WHITE_SAUCER_RETREAT_SPEED = -3.0  # Faster retreat speed after shooting
    # White saucer ramming pattern
    WHITE_SAUCER_RAMMING = 6  # New movement pattern for ramming
    WHITE_SAUCER_RAMMING_CHANCE = 0.15  # 15% chance for ramming pattern
    WHITE_SAUCER_RAMMING_SPEED = 3.5  # Fast speed for ramming movement
    WHITE_SAUCER_RAMMING_MIN_SECTOR = 10  # Start ramming from sector 10
    WHITE_SAUCER_RAMMING_INCREASED_CHANCE_SECTOR = 20  # Increased chance from sector 20
    WHITE_SAUCER_RAMMING_HIGH_SECTOR_CHANCE = 0.25  # 25% chance in high sectors
    # Horizon patrol system
    HORIZON_LINE_Y = 25  # Y position of the horizon line where saucers patrol
    WHITE_SAUCER_HORIZON_PATROL = 5  # New movement pattern for horizon patrol

    # Horizon patrol behavior
    HORIZON_PATROL_SPEED = 0.8  # Speed while patrolling horizon
    HORIZON_PATROL_PAUSE_TIME = 45  # Frames to pause after reaching new lane
    HORIZON_JUMP_MIN_LANES = 1  # Minimum lanes to jump
    HORIZON_JUMP_MAX_LANES = 3  # Maximum lanes to jump
    HORIZON_DIRECTION_CHANGE_CHANCE = 0.3  # 30% chance to change direction when pause ends
    HORIZON_PATROL_TIME = 180  # Frames to patrol before diving (3 seconds)
    HORIZON_DIVE_CHANCE = 0.3  # 30% chance to dive when patrol timer expires

    # White saucer smooth movement
    WHITE_SAUCER_HORIZONTAL_SPEED = 1.5  # Pixels per frame horizontal movement
    WHITE_SAUCER_BEAM_SNAP_DISTANCE = 3

    # Sentinel ship specific constants
    SENTINEL_SHIP_SPEED = 0.3  # Moderate base speed, will scale
    SENTINEL_SHIP_POINTS = 200  # High points when destroyed with torpedo
    SENTINEL_SHIP_COLOR = (192, 192, 192)  # Silver/grey color RGB
    SENTINEL_SHIP_SPAWN_SECTOR = 1  # Starts appearing from sector 1
    SENTINEL_SHIP_SPAWN_CHANCE = 0.05  # 5% chance to spawn sentinel ship (rare)
    SENTINEL_SHIP_WIDTH = 12  # Larger than regular enemies
    SENTINEL_SHIP_HEIGHT = 10  # Larger than regular enemies
    SENTINEL_SHIP_FIRING_INTERVAL = 120  # Frames between shots (2 seconds at 60fps)
    SENTINEL_SHIP_PROJECTILE_SPEED = 3.0  # Speed of sentinel projectiles
    SENTINEL_SHIP_HEALTH = 1  # Takes 1 torpedo hit to destroy

    # Orange tracker specific constants
    ORANGE_TRACKER_SPEED = 0.9  # Slower base tracking speed
    ORANGE_TRACKER_POINTS = 50  # Points when destroyed with torpedo
    ORANGE_TRACKER_COLOR = (255, 165, 0)  # Orange color RGB
    ORANGE_TRACKER_SPAWN_SECTOR = 12  # Starts appearing from sector 12
    ORANGE_TRACKER_SPAWN_CHANCE = 0.08  # 8% chance to spawn orange tracker
    ORANGE_TRACKER_CHANGE_DIRECTION_INTERVAL = 90  # Frames between direction changes

    # Tracker course change limits based on sector
    ORANGE_TRACKER_BASE_COURSE_CHANGES = 1  # Base number of course changes allowed
    ORANGE_TRACKER_COURSE_CHANGE_INCREASE_SECTOR = 5  # Every X sectors, add 1 more course change

    # Blue charger specific constants
    BLUE_CHARGER_SPEED = 1.1  # Slower base speed
    BLUE_CHARGER_POINTS = 30  # Points when destroyed
    BLUE_CHARGER_COLOR = (0, 0, 255)  # Blue color RGB
    BLUE_CHARGER_SPAWN_SECTOR = 10  # Starts appearing from sector 10
    BLUE_CHARGER_SPAWN_CHANCE = 0.1  # 10% chance to spawn blue charger
    BLUE_CHARGER_LINGER_TIME = 180  # Frames to stay at bottom (3 seconds at 60fps)
    BLUE_CHARGER_DEFLECT_SPEED = -2.0  # Speed when deflected upward by laser

    # Brown debris specific constants
    BROWN_DEBRIS_SPEED = 1.0  # Slower base speed
    BROWN_DEBRIS_POINTS = 25  # Bonus points when destroyed with torpedo
    BROWN_DEBRIS_COLOR = (139, 69, 19)  # Brown color RGB

    # Spawn probabilities (add to existing constants)
    BROWN_DEBRIS_SPAWN_SECTOR = 2  # Starts appearing from sector 2
    BROWN_DEBRIS_SPAWN_CHANCE = 0.15  # 15% chance to spawn brown debris

    # Yellow chirper specific constants
    YELLOW_CHIRPER_SPEED = 0.7  # Slower horizontal movement speed
    YELLOW_CHIRPER_POINTS = 50  # Bonus points for shooting them
    YELLOW_CHIRPER_COLOR = (255, 255, 0)  # Yellow color RGB
    YELLOW_CHIRPER_SPAWN_Y_MIN = 50  # Minimum Y position for horizontal flight
    YELLOW_CHIRPER_SPAWN_Y_MAX = 150  # Maximum Y position for horizontal flight

    YELLOW_CHIRPER_SPAWN_SECTOR = 4  # Starts appearing from sector 4
    YELLOW_CHIRPER_SPAWN_CHANCE = 0.1  # 10% chance to spawn yellow chirper

    # Green blocker specific constants
    GREEN_BLOCKER_SPEED = 0.8  # Much slower ramming speed
    GREEN_BLOCKER_POINTS = 75  # High points when destroyed
    GREEN_BLOCKER_COLOR = (0, 255, 0)  # Green color RGB
    GREEN_BLOCKER_SPAWN_Y_MIN = 30  # Spawn higher up for targeting
    GREEN_BLOCKER_SPAWN_Y_MAX = 80  # Range for side spawning
    GREEN_BLOCKER_LOCK_DISTANCE = 100  # Distance at which they lock onto player beam
    GREEN_BLOCKER_SPAWN_SECTOR = 6  # Starts appearing from sector 6
    GREEN_BLOCKER_SPAWN_CHANCE = 0.12  # 12% chance to spawn green blocker
    GREEN_BLOCKER_SENTINEL_SPAWN_CHANCE = 0.3  # 30% chance when sentinel is active in sectors 1-5


    # Green bounce craft specific constants - UPDATED speeds
    GREEN_BOUNCE_SPEED = 1.5  # Slower bouncing speed
    GREEN_BOUNCE_POINTS = 100  # Very high points when destroyed with torpedo
    GREEN_BOUNCE_COLOR = (0, 200, 0)  # Slightly different green than blockers
    GREEN_BOUNCE_SPAWN_SECTOR = 7  # Starts appearing from sector 7
    GREEN_BOUNCE_SPAWN_CHANCE = 0.08  # 8% chance to spawn green bounce craft

    # Yellow rejuvenator specific constants
    YELLOW_REJUVENATOR_SPEED = 0.5  # Slow float speed
    YELLOW_REJUVENATOR_POINTS = 0  # No points for shooting (discourage shooting)
    YELLOW_REJUVENATOR_LIFE_BONUS = 1  # Adds 1 life when collected
    YELLOW_REJUVENATOR_COLOR = (255, 255, 100)  # Bright yellow color RGB
    YELLOW_REJUVENATOR_SPAWN_SECTOR = 1  # Starts appearing from sector 1
    YELLOW_REJUVENATOR_SPAWN_CHANCE = 0.04  # 4% chance to spawn
    YELLOW_REJUVENATOR_OSCILLATION_AMPLITUDE = 15  # Horizontal oscillation range
    YELLOW_REJUVENATOR_OSCILLATION_FREQUENCY = 0.06  # Oscillation frequency

    # Rejuvenator debris constants (when shot)
    REJUVENATOR_DEBRIS_SPEED = 1.5  # Fast moving debris
    REJUVENATOR_DEBRIS_COLOR = (255, 0, 0)  # Red explosive debris
    REJUVENATOR_DEBRIS_COUNT = 1  # Number of debris pieces created

    # HUD margins
    TOP_MARGIN = int(210 * 0.12)

    @classmethod
    def get_beam_positions(cls) -> jnp.ndarray:
        """Calculate 5 beam positions to match classic BeamRider layout with bounds checking"""
        center_x = cls.SCREEN_WIDTH // 2
        beam_spacing = 24

        positions = jnp.array([
            center_x - 2 * beam_spacing,
            center_x - beam_spacing,
            center_x,
            center_x + beam_spacing,
            center_x + 2 * beam_spacing
        ], dtype=jnp.float32)

        # Ensure beams stay within reasonable screen bounds
        min_x = 16  # Minimum distance from left edge
        max_x = cls.SCREEN_WIDTH - 16  # Maximum distance from right edge

        positions = jnp.clip(positions, min_x, max_x)

        return positions
@struct.dataclass
class Ship:
    # Represents the player-controlled ship: position, beam lane, and active status.
    x: float
    y: float
    beam_position: int  # Index of the current beam (0–4)
    target_beam: int  # Target beam the ship is moving toward
    active: bool = True  # Whether the ship is currently active (alive)
    last_action: int = 0  # Add this to track previous input
    movement_cooldown: int = 0  # Add this to prevent rapid direction changes


@struct.dataclass
class Projectile:
    # Represents a player-fired projectile (laser or torpedo), with position, speed, and type.
    x: float  # Horizontal position of the projectile (in pixels)
    y: float  # Vertical position of the projectile (in pixels)
    active: bool  # Whether the projectile is currently in play
    speed: float  # Vertical movement speed (positive = upward)
    projectile_type: int  # 0 = laser, 1 = torpedo


@struct.dataclass
class Enemy:
    """Enemy ship state"""
    x: float
    y: float
    beam_position: int  # Which beam the enemy is on (0-4) OR direction for special enemies
    active: bool
    speed: float = BeamRiderConstants.ENEMY_SPEED
    enemy_type: int = 0  # Different enemy types for variety
    # Additional fields for behavior
    direction_x: float = 1.0  # Horizontal direction for bounce craft
    direction_y: float = 1.0  # Vertical direction for bounce craft
    bounce_count: int = 0  # Number of bounces remaining for bounce craft
    linger_timer: int = 0  # Timer for blue chargers lingering at bottom
    tracker_timer: int = 0  # Timer for orange tracker direction changes


@struct.dataclass
class BeamRiderState:
    """Complete game state"""
    #Game entities
    ship: Ship
    projectiles: chex.Array
    enemies: chex.Array

    # Game state
    score: int
    lives: int
    level: int
    game_over: bool
    # Random state
    rng_key: chex.PRNGKey
    # Timing and spawning
    frame_count: int
    enemy_spawn_timer: int

    # Torpedo system
    torpedoes_remaining: int
    torpedo_projectiles: chex.Array
    current_sector: int
    enemies_killed_this_sector: int

    # Sentinel ship projectiles
    sentinel_projectiles: chex.Array
    # Track sentinel status for current sector
    sentinel_spawned_this_sector: bool
    enemy_spawn_interval: int = BeamRiderConstants.ENEMY_SPAWN_INTERVAL


class BeamRiderObservation(NamedTuple):
    """BeamRider observation structure"""
    ship_x: jnp.ndarray
    ship_y: jnp.ndarray
    ship_beam: jnp.ndarray
    projectiles: jnp.ndarray  # All projectile data [x, y, active, speed]
    torpedo_projectiles: jnp.ndarray  # All torpedo data [x, y, active, speed]
    enemies: jnp.ndarray  # All enemy data (18 columns as in state)
    score: jnp.ndarray
    lives: jnp.ndarray
    current_sector: jnp.ndarray
    torpedoes_remaining: jnp.ndarray


class BeamRiderInfo(NamedTuple):
    """BeamRider info structure"""
    frame_count: jnp.ndarray
    enemies_killed_this_sector: jnp.ndarray
    enemy_spawn_timer: jnp.ndarray
    sentinel_spawned_this_sector: jnp.ndarray
    all_rewards: chex.Array


class BeamRiderEnv(JaxEnvironment[BeamRiderState, BeamRiderObservation, BeamRiderInfo, BeamRiderConstants]):
    """BeamRider environment following JAXAtari structure"""

    def __init__(self, consts: BeamRiderConstants = None, reward_funcs: list[callable] = None):
        consts = consts or BeamRiderConstants()
        consts = BeamRiderConstants()
        super().__init__(consts)
        self.constants = BeamRiderConstants()
        self.screen_width = self.constants.SCREEN_WIDTH
        self.screen_height = self.constants.SCREEN_HEIGHT
        self.action_space_size = 18
        self.beam_positions = self.constants.get_beam_positions()
        self.renderer = BeamRiderRenderer()
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs

    def action_space(self) -> spaces.Discrete:
        """Returns the action space for BeamRider"""
        return spaces.Discrete(self.action_space_size)

    def observation_space(self) -> spaces.Dict:
        """Returns the observation space for BeamRider - FIXED: Correct bounds and dtypes"""
        return spaces.Dict({
            "ship_x": spaces.Box(low=0, high=self.constants.SCREEN_WIDTH, shape=(), dtype=jnp.float32),
            "ship_y": spaces.Box(low=0, high=self.constants.SCREEN_HEIGHT, shape=(), dtype=jnp.float32),
            "ship_beam": spaces.Box(low=0, high=self.constants.NUM_BEAMS - 1, shape=(), dtype=jnp.int8),

            # FIXED: Projectiles have 5 columns [x, y, active, speed, beam_idx]
            # Speed can be negative (-4.0 for player projectiles) to positive (~6.25 for fast enemies)
            "projectiles": spaces.Box(
                low=jnp.array([-50, -50, 0, -10.0, 0]),  # [x, y, active, speed, beam_idx]
                high=jnp.array([self.constants.SCREEN_WIDTH + 50, self.constants.SCREEN_HEIGHT + 50, 1, 10.0,
                                self.constants.NUM_BEAMS - 1]),
                shape=(self.constants.MAX_PROJECTILES, 5),
                dtype=jnp.float32
            ),

            # FIXED: Same structure as projectiles
            "torpedo_projectiles": spaces.Box(
                low=jnp.array([-50, -50, 0, -10.0, 0]),  # [x, y, active, speed, beam_idx]
                high=jnp.array([self.constants.SCREEN_WIDTH + 50, self.constants.SCREEN_HEIGHT + 50, 1, 10.0,
                                self.constants.NUM_BEAMS - 1]),
                shape=(self.constants.MAX_PROJECTILES, 5),
                dtype=jnp.float32
            ),

            # Enemies bounds are mostly correct but need slight adjustment
            "enemies": spaces.Box(low=-100, high=max(self.constants.SCREEN_WIDTH, self.constants.SCREEN_HEIGHT) + 100,
                                  shape=(self.constants.MAX_ENEMIES, 17), dtype=jnp.float32),
            "score": spaces.Box(low=0, high=999999, shape=(), dtype=jnp.float32),
            "lives": spaces.Box(low=0, high=10, shape=(), dtype=jnp.int8),
            "current_sector": spaces.Box(low=1, high=99, shape=(), dtype=jnp.int8),
            "torpedoes_remaining": spaces.Box(low=0, high=self.constants.TORPEDOES_PER_SECTOR, shape=(),
                                              dtype=jnp.int8),
        })
    def image_space(self) -> spaces.Box:
        """Returns the image space for BeamRider"""
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.constants.SCREEN_HEIGHT, self.constants.SCREEN_WIDTH, 3),
            dtype=jnp.uint8
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(self, previous_state: BeamRiderState, state: BeamRiderState):
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array(
            [reward_func(previous_state, state) for reward_func in self.reward_funcs]
        )
        return rewards

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: BeamRiderState) -> BeamRiderObservation:
        """Convert state to observation"""
        return BeamRiderObservation(
            ship_x=jnp.array(state.ship.x, dtype=jnp.float32),
            ship_y=jnp.array(state.ship.y, dtype=jnp.float32),
            ship_beam=jnp.array(state.ship.beam_position, dtype=jnp.int32),
            projectiles=state.projectiles.astype(jnp.float32),
            torpedo_projectiles=state.torpedo_projectiles.astype(jnp.float32),
            enemies=state.enemies.astype(jnp.float32),
            score=jnp.array(state.score, dtype=jnp.int32),
            lives=jnp.array(state.lives, dtype=jnp.int32),
            current_sector=jnp.array(state.current_sector, dtype=jnp.int32),
            torpedoes_remaining=jnp.array(state.torpedoes_remaining, dtype=jnp.int32),
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: BeamRiderState, all_rewards: chex.Array = None) -> BeamRiderInfo:
        """Extract info from state"""
        return BeamRiderInfo(
            frame_count=jnp.array(state.frame_count, dtype=jnp.int32),
            enemies_killed_this_sector=jnp.array(state.enemies_killed_this_sector, dtype=jnp.int32),
            enemy_spawn_timer=jnp.array(state.enemy_spawn_timer, dtype=jnp.int32),
            sentinel_spawned_this_sector=jnp.array(state.sentinel_spawned_this_sector, dtype=jnp.bool_),
            all_rewards=all_rewards
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: BeamRiderState, state: BeamRiderState) -> jnp.ndarray:
        """Calculate reward from state difference"""
        score_diff = state.score - previous_state.score
        return jnp.array(score_diff, dtype=jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: BeamRiderState) -> jnp.ndarray:
        """Determine if episode is done"""
        return jnp.array(state.game_over, dtype=jnp.bool_)

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: BeamRiderObservation) -> jnp.ndarray:
        """Convert observation to flat array with consistent float32 dtype."""
        flat_components = [
            obs.ship_x.flatten().astype(jnp.float32),
            obs.ship_y.flatten().astype(jnp.float32),
            obs.ship_beam.flatten().astype(jnp.float32),
            obs.projectiles.flatten().astype(jnp.float32),
            obs.torpedo_projectiles.flatten().astype(jnp.float32),
            obs.enemies.flatten().astype(jnp.float32),
            obs.score.flatten().astype(jnp.float32),
            obs.lives.flatten().astype(jnp.float32),
            obs.current_sector.flatten().astype(jnp.float32),
            obs.torpedoes_remaining.flatten().astype(jnp.float32),
        ]
        return jnp.concatenate(flat_components).astype(jnp.float32)

    def render(self, state: BeamRiderState) -> jnp.ndarray:
        """Render the current game state"""
        return self.renderer.render(state)

    def reset(self, rng_key: chex.PRNGKey = None) -> Tuple[BeamRiderObservation, BeamRiderState]:
        """Reset the game to initial state"""
        # Initialize ship at bottom center beam
        initial_beam = self.constants.INITIAL_BEAM
        ship = Ship(
            x=self.beam_positions[initial_beam] - self.constants.SHIP_WIDTH // 2,
            y=self.constants.SCREEN_HEIGHT - self.constants.SHIP_BOTTOM_OFFSET,
            beam_position=initial_beam,
            target_beam=initial_beam,  # Start with target same as current
            active=True
        )

        # Initialize empty projectiles arrays (4 columns each)
        projectiles = jnp.zeros((self.constants.MAX_PROJECTILES, 5), dtype=jnp.float32)  # x, y, active, speed
        torpedo_projectiles = jnp.zeros((self.constants.MAX_PROJECTILES, 5), dtype=jnp.float32)  # x, y, active, speed
        sentinel_projectiles = jnp.zeros((self.constants.MAX_PROJECTILES, 5), dtype=jnp.float32)  # x, y, active, speed

        # Initialize empty enemies array
        enemies = jnp.zeros((self.constants.MAX_ENEMIES, 17), dtype=jnp.float32)

        state = BeamRiderState(
            ship=ship,
            projectiles=projectiles,
            enemies=enemies,
            torpedo_projectiles=torpedo_projectiles,
            sentinel_projectiles=sentinel_projectiles,
            score=0,
            lives=self.constants.INITIAL_LIVES,
            level=1,
            game_over=False,
            frame_count=0,
            enemy_spawn_timer=0,
            current_sector=1,
            enemies_killed_this_sector=0,
            torpedoes_remaining=self.constants.TORPEDOES_PER_SECTOR,
            sentinel_spawned_this_sector=False,
            enemy_spawn_interval=self.constants.BASE_ENEMY_SPAWN_INTERVAL,
            rng_key=rng_key
        )

        obs = self._get_observation(state)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: BeamRiderState, action: int) -> Tuple[
        BeamRiderObservation, BeamRiderState, jnp.ndarray, jnp.ndarray, BeamRiderInfo]:
        """Execute one game step - JIT-compiled implementation"""
        # Store previous state for reward calculation
        previous_state = state

        # Process player input and update ship
        state = self._update_ship(state, action)

        # Handle projectile firing
        state = self._handle_firing(state, action)

        # Update projectiles
        state = self._update_projectiles(state)

        # Spawn enemies
        state = self._spawn_enemies(state)

        # Update enemies
        state = self._update_enemies(state)
        # Handle white saucer shooting
        state = self._handle_white_saucer_shooting(state)

        # Update sentinel ship projectiles
        state = self._update_sentinel_projectiles(state)

        state = self._check_rejuvenator_interactions(state)
        state = self._check_debris_collision(state)

        # Check collisions
        state = self._check_collisions(state)

        # Check sector progression
        state = self._check_sector_progression(state)

        # Check game over conditions
        state = self._check_game_over(state)

        # Update frame count only once at the end
        state = state.replace(frame_count=state.frame_count + 1)

        # Create return values
        obs = self._get_observation(state)
        reward = self._get_reward(previous_state, state)
        all_rewards = self._get_all_reward(previous_state, state)
        done = self._get_done(state)
        info = self._get_info(state, all_rewards)

        return obs, state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _update_ship(self, state: BeamRiderState, action: int) -> BeamRiderState:
        """Update ship position with smooth beam-to-beam movement"""
        ship = state.ship
        current_beam = ship.beam_position
        target_beam = ship.target_beam

        # Calculate current target X position
        current_target_x = self.beam_positions[target_beam] - self.constants.SHIP_WIDTH // 2
        current_x = ship.x

        # Determine which direction we're currently moving (if any)
        currently_moving_left = current_x > current_target_x + 1.0
        currently_moving_right = current_x < current_target_x - 1.0
        at_target = ~currently_moving_left & ~currently_moving_right

        # Handle input
        wants_to_move_left = (action == 4)
        wants_to_move_right = (action == 3)

        # Only allow target change if:
        # 1. We're at the target position, OR
        # 2. We want to move in the SAME direction we're already moving, OR
        # 3. We want to REVERSE direction (this allows immediate response to opposite input)
        can_change_target = (
                at_target |
                (wants_to_move_left & currently_moving_left) |
                (wants_to_move_right & currently_moving_right) |
                (wants_to_move_left & currently_moving_right) |  # Reversal
                (wants_to_move_right & currently_moving_left)  # Reversal
        )

        # Calculate potential new target
        desired_target = jnp.where(
            wants_to_move_left,
            target_beam - 1,
            jnp.where(
                wants_to_move_right,
                target_beam + 1,
                target_beam
            )
        )

        # Clamp to valid beam range
        desired_target = jnp.clip(desired_target, 0, self.constants.NUM_BEAMS - 1)

        # Only update target if we can
        new_target_beam = jnp.where(
            can_change_target,
            desired_target,
            target_beam  # Keep current target if we can't change
        )

        # Calculate movement toward target
        target_x = self.beam_positions[new_target_beam] - self.constants.SHIP_WIDTH // 2
        x_diff = target_x - current_x
        movement_needed = jnp.abs(x_diff) > 1.0

        # Smooth movement
        movement_speed = 4.5
        new_x = jnp.where(
            movement_needed,
            current_x + jnp.sign(x_diff) * jnp.minimum(movement_speed, jnp.abs(x_diff)),
            target_x  # Snap to exact position when close
        )

        # Update beam position when ship reaches a beam center
        # Find closest beam to current position
        ship_center_x = new_x + self.constants.SHIP_WIDTH // 2
        beam_distances = jnp.abs(self.beam_positions - ship_center_x)
        closest_beam = jnp.argmin(beam_distances)
        at_beam_center = beam_distances[closest_beam] < 2.0  # Within 2 pixels of beam center

        new_beam_position = jnp.where(
            at_beam_center,
            closest_beam,
            current_beam  # Keep current beam if between beams
        )

        return state.replace(ship=ship.replace(
            x=new_x,
            beam_position=new_beam_position,
            target_beam=new_target_beam
        ))
    @partial(jax.jit, static_argnums=(0,))
    def _select_white_saucer_movement_pattern(self, rng_key: chex.PRNGKey) -> int:
        """Select movement pattern for a new white saucer - UPDATED: includes horizon patrol"""
        # Generate random value for pattern selection
        pattern_rand = random.uniform(rng_key, (), minval=0.0, maxval=1.0)

        # Determine movement pattern based on probabilities
        pattern = jnp.where(
            pattern_rand < 0.4,  # 40% start with horizon patrol
            self.constants.WHITE_SAUCER_HORIZON_PATROL,
            jnp.where(
                pattern_rand < 0.55,  # 15% reverse pattern
                self.constants.WHITE_SAUCER_REVERSE_UP,
                jnp.where(
                    pattern_rand < 0.7,  # 15% beam jump
                    self.constants.WHITE_SAUCER_BEAM_JUMP,
                    jnp.where(
                        pattern_rand < 0.85,  # 15% shooting
                        self.constants.WHITE_SAUCER_SHOOTING,
                        self.constants.WHITE_SAUCER_STRAIGHT_DOWN  # 15% straight down
                    )
                )
            )
        )

        return pattern

    @partial(jax.jit, static_argnums=(0,))
    def _handle_white_saucer_shooting(self, state: BeamRiderState) -> BeamRiderState:
        """Handle white saucer projectile firing with random shooting while moving down"""
        enemies = state.enemies

        # Find white saucers that could potentially shoot
        white_saucer_mask = enemies[:, 5] == self.constants.ENEMY_TYPE_WHITE_SAUCER
        active_mask = enemies[:, 3] == 1

        # Prevent shooting while still at or near horizon line
        away_from_horizon_mask = enemies[:, 1] > (self.constants.HORIZON_LINE_Y + 15)

        # Check if saucer is moving downward (not retreating)
        not_retreating = enemies[:, 13] == 0  # Not in any retreat state
        moving_downward = enemies[:, 4] > 0  # Positive speed means moving down

        # Exclude horizon patrol saucers from random shooting
        not_horizon_patrol = enemies[:, 14] != self.constants.WHITE_SAUCER_HORIZON_PATROL

        # Any white saucer can potentially shoot if it's active, away from horizon, and moving down
        can_potentially_shoot_random = (
                white_saucer_mask &
                active_mask &
                away_from_horizon_mask &
                not_retreating &
                moving_downward &
                not_horizon_patrol
        )

        # Generate random values for shooting decision
        rng_key, shoot_rng = random.split(state.rng_key)
        shoot_random_values = random.uniform(
            shoot_rng,
            shape=(self.constants.MAX_ENEMIES,),
            minval=0.0,
            maxval=1.0
        )

        # Random shooting chance per frame
        RANDOM_SHOOT_CHANCE = 0.004  # ~0.8% chance per frame

        # Check if firing timer has cooled down
        firing_timer_ready = enemies[:, 15] <= 0

        # Determine which saucers will shoot randomly this frame
        will_shoot_random = shoot_random_values < RANDOM_SHOOT_CHANCE
        can_shoot_random = can_potentially_shoot_random & firing_timer_ready & will_shoot_random

        # === ORIGINAL SHOOTING PATTERN LOGIC ===
        shooting_pattern_mask = enemies[:, 14] == self.constants.WHITE_SAUCER_SHOOTING
        ready_to_fire_pattern = enemies[:, 15] == 0

        can_shoot_pattern = (
                white_saucer_mask &
                active_mask &
                shooting_pattern_mask &
                ready_to_fire_pattern &
                not_retreating &
                away_from_horizon_mask
        )

        # === SECOND SHOT LOGIC ===
        in_retreat_state_mask = enemies[:, 13] == self.constants.WHITE_SAUCER_RETREAT_AFTER_SHOT
        beam_change_timer = enemies[:, 16].astype(int)
        finished_beam_change_mask = beam_change_timer == 1

        can_shoot_second = (
                white_saucer_mask &
                active_mask &
                shooting_pattern_mask &
                in_retreat_state_mask &
                finished_beam_change_mask &
                away_from_horizon_mask
        )

        # Combine all shooting conditions
        can_shoot = can_shoot_random | can_shoot_pattern | can_shoot_second

        # Find any white saucer that can shoot
        any_can_shoot = jnp.any(can_shoot)

        # Find first shooting white saucer
        shooter_idx = jnp.argmax(can_shoot)

        # Get shooter position
        shooter_x = enemies[shooter_idx, 0]
        shooter_y = enemies[shooter_idx, 1]
        shooter_beam = enemies[shooter_idx, 2].astype(int)

        # Create projectile
        projectile_x = shooter_x + self.constants.ENEMY_WIDTH // 2 - self.constants.PROJECTILE_WIDTH // 2
        projectile_y = shooter_y + self.constants.ENEMY_HEIGHT

        new_projectile = jnp.array([
            projectile_x,
            projectile_y,
            1,  # active
            self.constants.WHITE_SAUCER_PROJECTILE_SPEED,
            shooter_beam
        ])

        # Find first inactive slot in sentinel projectiles array
        sentinel_projectiles = state.sentinel_projectiles
        inactive_mask = sentinel_projectiles[:, 2] == 0
        first_inactive = jnp.argmax(inactive_mask)
        can_spawn_projectile = inactive_mask[first_inactive]

        # Create and add projectile if conditions are met
        should_fire = any_can_shoot & can_spawn_projectile

        # Update sentinel projectiles array
        sentinel_projectiles = jnp.where(
            should_fire,
            sentinel_projectiles.at[first_inactive].set(new_projectile),
            sentinel_projectiles
        )

        # === RETREAT LOGIC FOR ALL SHOTS ===
        # Check what type of shot this was
        is_random_shot = can_shoot_random[shooter_idx] & should_fire
        is_pattern_first_shot = can_shoot_pattern[shooter_idx] & should_fire

        # For random shots, immediately set to retreat (no beam change)
        enemies = jnp.where(
            is_random_shot,
            enemies.at[shooter_idx, 13].set(self.constants.WHITE_SAUCER_RETREAT_AFTER_SHOT),
            enemies
        )

        # Immediately switch to REVERSE_UP for random shots
        enemies = jnp.where(
            is_random_shot,
            enemies.at[shooter_idx, 14].set(self.constants.WHITE_SAUCER_REVERSE_UP),
            enemies
        )

        # For pattern shots, handle beam change logic
        retreat_rng = random.fold_in(shoot_rng, state.frame_count + 5000)
        should_change_beam_rng = random.uniform(retreat_rng, (), minval=0.0, maxval=1.0, dtype=jnp.float32)
        should_change_beam = should_change_beam_rng < self.constants.WHITE_SAUCER_BEAM_CHANGE_CHANCE

        beam_rng = random.fold_in(retreat_rng, 1)
        random_beam = random.randint(beam_rng, (), 0, self.constants.NUM_BEAMS)
        new_retreat_beam = jnp.where(
            random_beam == shooter_beam,
            (shooter_beam + 1) % self.constants.NUM_BEAMS,
            random_beam
        )

        # Set retreat state for pattern shots
        enemies = jnp.where(
            is_pattern_first_shot,
            enemies.at[shooter_idx, 13].set(self.constants.WHITE_SAUCER_RETREAT_AFTER_SHOT),
            enemies
        )

        # Handle beam change for pattern shots
        enemies = jnp.where(
            is_pattern_first_shot & should_change_beam,
            enemies.at[shooter_idx, 10].set(new_retreat_beam),
            enemies
        )

        enemies = jnp.where(
            is_pattern_first_shot & should_change_beam,
            enemies.at[shooter_idx, 16].set(self.constants.WHITE_SAUCER_RETREAT_BEAM_CHANGE_TIME),
            enemies
        )

        enemies = jnp.where(
            is_pattern_first_shot & ~should_change_beam,
            enemies.at[shooter_idx, 14].set(self.constants.WHITE_SAUCER_REVERSE_UP),
            enemies
        )

        # === RESET FIRING TIMER ===
        cooldown_time = jnp.where(
            is_random_shot,
            45,  # Shorter cooldown for random shots
            self.constants.WHITE_SAUCER_FIRING_INTERVAL
        )

        enemies = jnp.where(
            should_fire,
            enemies.at[shooter_idx, 15].set(cooldown_time),
            enemies
        )

        # Decrement firing timers ONLY for non-horizon-patrol saucers
        # (Horizon patrol uses timer for different purpose)
        should_decrement_timer = (
                white_saucer_mask &
                active_mask &
                (enemies[:, 14] != self.constants.WHITE_SAUCER_HORIZON_PATROL) &
                (enemies[:, 15] > 0)
        )

        enemies = enemies.at[:, 15].set(
            jnp.where(
                should_decrement_timer,
                enemies[:, 15] - 1,
                enemies[:, 15]
            )
        )

        return state.replace(
            enemies=enemies,
            sentinel_projectiles=sentinel_projectiles,
            rng_key=shoot_rng
        )
    @partial(jax.jit, static_argnums=(0,))
    def _update_white_saucer_movement(self, state: BeamRiderState) -> BeamRiderState:
        """
        Enhanced white saucer movement patterns with horizon patrol system and player beam check for kamikaze.

        This method handles complex movement for white saucer enemies, implementing multiple behavioral patterns
        including horizon patrol, beam jumping, shooting, ramming, and intelligent retreat/kamikaze logic.

        Key Features:
        - 5 distinct movement patterns with unique behaviors
        - Dynamic retreat system that triggers kamikaze mode when player is on same beam
        - Horizon patrol system with randomized diving patterns
        - Beam curve adherence for realistic movement along the game's perspective
        - Timer-based state management for complex behaviors

        Movement Patterns:
        - STRAIGHT_DOWN (0): Basic downward movement
        - BEAM_JUMP (1): Horizontal movement between beams with targeting
        - REVERSE_UP (2): Upward retreat movement
        - HORIZON_PATROL (3): Patrol behavior at horizon line
        - SHOOTING (4): Stationary shooting with beam repositioning
        - RAMMING (5): High-speed kamikaze attack

        Args:
            state: Current game state containing enemy positions, timers, and game properties

        Returns:
            Updated BeamRiderState with modified enemy positions and properties
        """
        enemies = state.enemies

        # === ENEMY IDENTIFICATION AND MASKING ===
        # Identify active white saucers for processing
        white_saucer_mask = enemies[:, 5] == self.constants.ENEMY_TYPE_WHITE_SAUCER
        active_mask = enemies[:, 3] == 1
        white_saucer_active = white_saucer_mask & active_mask

        # === POSITION AND PROPERTY EXTRACTION ===
        # Extract current state from enemy array
        current_x = enemies[:, 0]  # X position
        current_y = enemies[:, 1]  # Y position
        current_beam = enemies[:, 2].astype(int)  # Current beam (0-15)
        current_speed = enemies[:, 4]  # Movement speed
        movement_pattern = enemies[:, 14].astype(int)  # Current movement pattern
        firing_timer = enemies[:, 15].astype(int)  # Multi-purpose timer
        jump_timer = enemies[:, 16].astype(int)  # Beam jump/pause timer
        target_beam = enemies[:, 10].astype(int)  # Target beam for movement
        direction_x = enemies[:, 6]  # Horizontal movement direction (-1 or 1)

        # === DYNAMIC BOUNDARY CALCULATION ===
        # Calculate dotted-line boundaries at current Y position for each saucer
        # This ensures saucers stay within the playable beam area
        left_dotted_x = self._beam_curve_x(current_y, 0, self.constants.ENEMY_WIDTH)
        right_dotted_x = self._beam_curve_x(current_y, self.constants.NUM_BEAMS - 1, self.constants.ENEMY_WIDTH)
        dotted_min_x = jnp.minimum(left_dotted_x, right_dotted_x)
        dotted_max_x = jnp.maximum(left_dotted_x, right_dotted_x)

        # === TIMER UPDATES ===
        # Decrement active timers (clamped to 0)
        new_firing_timer = jnp.maximum(0, firing_timer - 1)
        new_jump_timer = jnp.maximum(0, jump_timer - 1)

        # === RETREAT AND KAMIKAZE LOGIC ===
        # Core decision system for when saucers should retreat vs. kamikaze
        reached_reverse_point = current_y >= self.constants.WHITE_SAUCER_REVERSE_TRIGGER_Y
        retreat_flag = enemies[:, 13].astype(int)

        # Critical kamikaze decision: if player is on same beam when reaching reverse point, continue attacking
        player_beam = state.ship.beam_position
        on_same_beam_as_player = current_beam == player_beam

        # Kamikaze mode: saucer continues downward at high speed when player is on same beam
        should_kamikaze = white_saucer_active & reached_reverse_point & on_same_beam_as_player & (retreat_flag == 0)

        # Normal retreat: only reverse if NOT on same beam as player
        should_start_retreat = white_saucer_active & reached_reverse_point & ~on_same_beam_as_player & (
                    retreat_flag == 0)

        # SHOOTING SAUCERS: Also reverse immediately when switched to REVERSE_UP pattern
        switched_to_reverse = white_saucer_active & (movement_pattern == self.constants.WHITE_SAUCER_REVERSE_UP) & (
                retreat_flag == self.constants.WHITE_SAUCER_RETREAT_AFTER_SHOT
        )

        # Determine movement direction (up vs down)
        should_be_moving_up = white_saucer_active & (
                (retreat_flag == self.constants.WHITE_SAUCER_RETREAT_AFTER_SHOT) |
                (should_start_retreat) |
                switched_to_reverse
        ) & ~should_kamikaze  # Don't move up if kamikazing

        # === RETREAT FLAG MANAGEMENT ===
        # Update retreat status with kamikaze flag handling
        start_retreat_now = should_start_retreat
        new_retreat_flag = jnp.where(start_retreat_now, self.constants.WHITE_SAUCER_RETREAT_AFTER_SHOT,
                                     enemies[:, 13].astype(int))

        # Clear retreat flag when reaching top of screen
        clear_retreat = should_be_moving_up & (current_y <= self.constants.HORIZON_LINE_Y)
        final_retreat_flag = jnp.where(clear_retreat, 0, new_retreat_flag)

        # Mark kamikazing saucers with special flag (99) for tracking
        KAMIKAZE_FLAG = 99
        final_retreat_flag = jnp.where(should_kamikaze, KAMIKAZE_FLAG, final_retreat_flag)

        # === MOVEMENT PATTERN 1: BEAM_JUMP WITH HORIZONTAL MOVEMENT ===
        """
        Beam Jump Pattern: Saucers move horizontally between beams in upper third of screen.
        - Generate random target beams when timer expires
        - Move horizontally toward target beam at increased speed
        - Snap to beam when close enough
        - Supports both normal descent and kamikaze mode
        """
        jump_mask = white_saucer_active & (movement_pattern == self.constants.WHITE_SAUCER_BEAM_JUMP)
        jump_current_target_beam = target_beam.astype(int)

        # Target selection logic: only in upper third and when timer expires
        in_upper_third = current_y <= self.constants.UPPER_THIRD_Y
        need_new_target = jump_mask & (new_jump_timer == 0) & in_upper_third & ~should_be_moving_up

        # Generate new random target beams using frame-based RNG
        jump_indices = jnp.arange(self.constants.MAX_ENEMIES)
        jump_rng_keys = jax.vmap(lambda i: random.fold_in(state.rng_key, state.frame_count + i + 4000))(jump_indices)
        new_random_beams = jax.vmap(lambda key: random.randint(key, (), 0, self.constants.NUM_BEAMS))(jump_rng_keys)

        # Update target beam and reset timer
        jump_new_target_beam = jnp.where(need_new_target, new_random_beams, jump_current_target_beam)
        jump_new_jump_timer = jnp.where(need_new_target, self.constants.WHITE_SAUCER_JUMP_INTERVAL, new_jump_timer)

        # Horizontal movement calculation
        actively_jumping = jump_mask & (jump_current_target_beam != current_beam) & ~should_be_moving_up
        jump_target_x = self._beam_curve_x(current_y, jump_new_target_beam, self.constants.ENEMY_WIDTH)
        jump_x_diff = jump_target_x - current_x
        close_enough = jnp.abs(jump_x_diff) <= self.constants.WHITE_SAUCER_BEAM_SNAP_DISTANCE

        # Fast horizontal movement when actively jumping
        jump_horizontal_movement = jnp.where(
            actively_jumping & ~close_enough,
            jnp.sign(jump_x_diff) * self.constants.WHITE_SAUCER_HORIZONTAL_SPEED * 2.0,
            0.0
        )

        # Position updates with boundary clamping
        jump_new_x = jnp.where(
            actively_jumping,
            jnp.clip(current_x + jump_horizontal_movement, dotted_min_x, dotted_max_x),
            current_x
        )

        # Vertical movement with kamikaze check
        is_kamikazing = final_retreat_flag == KAMIKAZE_FLAG
        jump_new_y = jnp.where(
            is_kamikazing,
            current_y + self.constants.WHITE_SAUCER_RAMMING_SPEED,  # High-speed kamikaze descent
            jnp.where(
                should_be_moving_up,
                current_y + self.constants.WHITE_SAUCER_REVERSE_SPEED_FAST,  # Fast retreat
                current_y + current_speed  # Normal descent
            )
        )

        # Beam position snapping when close to target
        jump_new_beam = jnp.where(
            jump_mask & close_enough,
            jump_new_target_beam,
            current_beam
        )

        # Speed updates based on current mode
        jump_new_speed = jnp.where(
            is_kamikazing,
            self.constants.WHITE_SAUCER_RAMMING_SPEED,
            jnp.where(
                should_be_moving_up,
                self.constants.WHITE_SAUCER_REVERSE_SPEED_FAST,
                current_speed
            )
        )

        # === MOVEMENT PATTERN 4: SHOOTING WITH BEAM REPOSITIONING ===
        """
        Shooting Pattern: Saucers can fire projectiles and reposition between beams during retreat.
        - Handles beam changes during retreat phase
        - Manages pre-shooting positioning near horizon
        - Supports transition to REVERSE_UP pattern after shooting
        """
        shooting_mask = white_saucer_active & (movement_pattern == self.constants.WHITE_SAUCER_SHOOTING)
        retreat_state = enemies[:, 13].astype(int)
        is_retreating = retreat_state == self.constants.WHITE_SAUCER_RETREAT_AFTER_SHOT
        beam_change_timer = jump_timer
        has_beam_change_timer = beam_change_timer > 0

        # Beam repositioning during retreat
        changing_beam_before_retreat = shooting_mask & is_retreating & has_beam_change_timer
        retreat_target_x = self._beam_curve_x(current_y, target_beam, self.constants.ENEMY_WIDTH)
        retreat_x_diff = retreat_target_x - current_x
        retreat_close_to_target = jnp.abs(retreat_x_diff) <= 3

        # Horizontal movement toward retreat target beam
        retreat_horizontal_movement = jnp.where(
            changing_beam_before_retreat & ~retreat_close_to_target,
            jnp.sign(retreat_x_diff) * self.constants.WHITE_SAUCER_HORIZONTAL_SPEED,
            0.0
        )

        # Timer management for beam changes
        new_beam_change_timer = jnp.where(
            changing_beam_before_retreat,
            jnp.maximum(0, beam_change_timer - 1),
            beam_change_timer
        )

        # Detection of beam change completion
        beam_change_complete = shooting_mask & is_retreating & (beam_change_timer == 1)

        # Pre-shooting positioning logic
        normal_shooting = shooting_mask & ~is_retreating
        stored_dive_depth = enemies[:, 7]
        shooting_near_horizon = current_y <= (self.constants.HORIZON_LINE_Y + stored_dive_depth)
        should_move_down_before_shooting = normal_shooting & shooting_near_horizon

        # Position updates for shooting pattern
        shooting_new_x = jnp.where(
            changing_beam_before_retreat,
            current_x + retreat_horizontal_movement,
            current_x
        )
        shooting_new_x = jnp.clip(shooting_new_x, dotted_min_x, dotted_max_x)

        # Vertical movement with multiple speed modes
        shooting_new_y = jnp.where(
            is_kamikazing,
            current_y + self.constants.WHITE_SAUCER_RAMMING_SPEED,
            jnp.where(
                should_be_moving_up,
                current_y + self.constants.WHITE_SAUCER_REVERSE_SPEED_FAST,
                jnp.where(
                    should_move_down_before_shooting,
                    current_y + 2.0,  # Faster descent for positioning
                    current_y + current_speed
                )
            )
        )

        # Beam updates when reaching target position
        shooting_new_beam = jnp.where(
            changing_beam_before_retreat & retreat_close_to_target,
            target_beam,
            current_beam
        )

        # Speed management for shooting pattern
        shooting_new_speed = jnp.where(
            is_kamikazing,
            self.constants.WHITE_SAUCER_RAMMING_SPEED,
            jnp.where(
                should_be_moving_up,
                self.constants.WHITE_SAUCER_REVERSE_SPEED_FAST,
                current_speed
            )
        )

        # === MOVEMENT PATTERN 3: HORIZON PATROL ===
        """
        Horizon Patrol Pattern: Sophisticated AI behavior at the horizon line.
        - Moves to horizon line and patrols horizontally
        - Randomly selects diving patterns based on sector and probability
        - Includes beam alignment logic and direction changes
        - Manages patrol timing and pause intervals
        """
        horizon_patrol_mask = white_saucer_active & (movement_pattern == self.constants.WHITE_SAUCER_HORIZON_PATROL)
        at_horizon = jnp.abs(current_y - self.constants.HORIZON_LINE_Y) <= 5

        # Initial movement to horizon
        moving_to_horizon = horizon_patrol_mask & ~at_horizon & (current_y < self.constants.HORIZON_LINE_Y)
        horizon_new_y_moving = current_y + 2.0

        # Patrol state management
        patrolling_horizon = horizon_patrol_mask & at_horizon
        is_paused = new_jump_timer > 0
        patrol_time = firing_timer
        patrol_time_expired = patrol_time >= self.constants.HORIZON_PATROL_TIME

        # Beam alignment system
        beam_distances = jnp.abs(current_x[..., None] - self.beam_positions[None, :])
        nearest_beam_distance = jnp.min(beam_distances, axis=1)
        nearest_beam_idx = jnp.argmin(beam_distances, axis=1)

        BEAM_ALIGNMENT_THRESHOLD = 4.0
        aligned_with_beam = nearest_beam_distance <= BEAM_ALIGNMENT_THRESHOLD

        # RNG setup for patrol decisions
        patrol_indices = jnp.arange(self.constants.MAX_ENEMIES)
        patrol_rng_keys = jax.vmap(lambda i: random.fold_in(state.rng_key, state.frame_count + i + 1000))(
            patrol_indices)

        # Diving decision logic
        dive_rng_keys = jax.vmap(lambda i: random.fold_in(state.rng_key, state.frame_count + i + 2000))(patrol_indices)
        should_dive_rng = jax.vmap(lambda key: random.uniform(key, (), minval=0.0, maxval=1.0, dtype=jnp.float32))(
            dive_rng_keys)
        should_dive = should_dive_rng < self.constants.HORIZON_DIVE_CHANCE

        # Extended patrol time forces diving
        extended_patrol_time = patrol_time >= (self.constants.HORIZON_PATROL_TIME * 2)

        # Diving trigger conditions
        start_diving = patrolling_horizon & (
                (patrol_time_expired & should_dive & aligned_with_beam) |
                extended_patrol_time
        )

        # Continue patrol condition
        continue_patrolling = patrolling_horizon & patrol_time_expired & (
                ~should_dive | ~aligned_with_beam) & ~extended_patrol_time

        # === DIVING PATTERN SELECTION ===
        """
        Sector-based probability system for selecting diving patterns:
        - Higher sectors increase ramming probability
        - Weighted random selection between ramming, shooting, beam jumping, and straight down
        """
        sector = state.current_sector
        ram_p = jnp.where(
            sector >= self.constants.WHITE_SAUCER_RAMMING_MIN_SECTOR,
            jnp.where(sector >= self.constants.WHITE_SAUCER_RAMMING_INCREASED_CHANCE_SECTOR,
                      self.constants.WHITE_SAUCER_RAMMING_HIGH_SECTOR_CHANCE,
                      self.constants.WHITE_SAUCER_RAMMING_CHANCE),
            0.0
        )

        # Weighted pattern selection
        pat_rng_keys = jax.vmap(lambda i: random.fold_in(state.rng_key, state.frame_count + i + 3001))(patrol_indices)
        u_pat = jax.vmap(lambda key: random.uniform(key, (), minval=0.0, maxval=1.0, dtype=jnp.float32))(pat_rng_keys)

        # Probability thresholds
        t0 = ram_p
        t1 = t0 + self.constants.WHITE_SAUCER_SHOOT_CHANCE
        t2 = t1 + self.constants.WHITE_SAUCER_JUMP_CHANCE

        # Pattern selection based on probability ranges
        selected_dive_pattern = jnp.where(
            u_pat < t0, self.constants.WHITE_SAUCER_RAMMING,
            jnp.where(
                u_pat < t1, self.constants.WHITE_SAUCER_SHOOTING,
                jnp.where(u_pat < t2, self.constants.WHITE_SAUCER_BEAM_JUMP,
                          self.constants.WHITE_SAUCER_STRAIGHT_DOWN)
            )
        )

        # Movement pattern updates
        new_movement_pattern = jnp.where(
            start_diving,
            selected_dive_pattern,
            jnp.where(
                beam_change_complete,
                self.constants.WHITE_SAUCER_REVERSE_UP,
                movement_pattern
            )
        )

        # === HORIZONTAL PATROL MOVEMENT ===
        """
        Complex horizontal movement system with lane jumping and direction changes:
        - Manages multi-lane jumps with random distances
        - Handles direction changes and boundary conditions
        - Includes pause intervals and beam alignment
        """
        diving = horizon_patrol_mask & ~should_be_moving_up & (current_y > self.constants.HORIZON_LINE_Y) & (
                    current_y < self.constants.WHITE_SAUCER_REVERSE_TRIGGER_Y)
        doing_horizontal_patrol = patrolling_horizon & ~start_diving
        pause_ending = doing_horizontal_patrol & (jump_timer == 1) & (new_jump_timer == 0)

        # Lane jump distance calculation
        lane_jump_rng = jax.vmap(lambda key: random.randint(key, (),
                                                            minval=self.constants.HORIZON_JUMP_MIN_LANES,
                                                            maxval=self.constants.HORIZON_JUMP_MAX_LANES + 1))(
            patrol_rng_keys)

        # Direction change logic
        direction_rng = jax.vmap(lambda key: random.uniform(key, (), minval=0.0, maxval=1.0, dtype=jnp.float32))(
            patrol_rng_keys)
        should_change_direction = direction_rng < self.constants.HORIZON_DIRECTION_CHANGE_CHANCE

        # Boundary condition handling
        current_direction = direction_x
        at_leftmost_beam = current_beam == 0
        at_rightmost_beam = current_beam == (self.constants.NUM_BEAMS - 1)

        # Force direction change at boundaries
        forced_direction = jnp.where(
            at_leftmost_beam, 1.0,
            jnp.where(at_rightmost_beam, -1.0, current_direction)
        )

        # New direction calculation
        new_direction = jnp.where(
            at_leftmost_beam | at_rightmost_beam,
            forced_direction,
            jnp.where(should_change_direction, -current_direction, current_direction)
        )

        # Target beam calculation with lane jumping
        direction_for_jump = jnp.where(pause_ending, new_direction, direction_x)
        new_target_beam_calc = current_beam + (direction_for_jump * lane_jump_rng).astype(int)
        new_target_beam_clamped = jnp.clip(new_target_beam_calc, 0, self.constants.NUM_BEAMS - 1)

        # Beam alignment handling
        needs_beam_alignment = doing_horizontal_patrol & ~aligned_with_beam & patrol_time_expired

        horizon_new_target_beam = jnp.where(
            needs_beam_alignment,
            nearest_beam_idx,
            jnp.where(pause_ending, new_target_beam_clamped, target_beam)
        )

        horizon_new_direction = jnp.where(pause_ending, new_direction, direction_x)

        # Horizontal movement calculation
        target_x_pos = self._beam_curve_x(current_y, horizon_new_target_beam, self.constants.ENEMY_WIDTH)
        x_diff = target_x_pos - current_x
        close_to_target = jnp.abs(x_diff) <= 3

        should_move_horizontally = doing_horizontal_patrol & ~is_paused & ~close_to_target
        horizontal_movement = jnp.where(
            should_move_horizontally,
            jnp.sign(x_diff) * self.constants.HORIZON_PATROL_SPEED,
            0.0
        )

        # Pause timer management
        reached_target = doing_horizontal_patrol & close_to_target & (new_jump_timer == 0)
        horizon_new_pause_timer = jnp.where(
            reached_target,
            self.constants.HORIZON_PATROL_PAUSE_TIME,
            new_jump_timer
        )

        # Position updates for horizon patrol
        horizon_new_beam = jnp.where(
            start_diving,
            nearest_beam_idx,
            jnp.where(
                doing_horizontal_patrol & close_to_target,
                horizon_new_target_beam,
                current_beam
            )
        )

        horizon_new_x = jnp.where(
            moving_to_horizon,
            current_x,
            jnp.where(
                doing_horizontal_patrol & close_to_target,
                target_x_pos,
                jnp.clip(current_x + horizontal_movement, dotted_min_x, dotted_max_x)
            )
        )

        # Vertical position updates with kamikaze support
        horizon_new_y = jnp.where(
            is_kamikazing,
            current_y + self.constants.WHITE_SAUCER_RAMMING_SPEED,
            jnp.where(
                moving_to_horizon,
                horizon_new_y_moving,
                jnp.where(
                    diving | start_diving, current_y + 2.0,
                    jnp.where(
                        should_be_moving_up, current_y + self.constants.WHITE_SAUCER_REVERSE_SPEED_FAST,
                        self.constants.HORIZON_LINE_Y
                    )
                )
            )
        )

        # Patrol timer updates
        horizon_new_patrol_timer = jnp.where(
            doing_horizontal_patrol & ~start_diving,
            patrol_time + 1,
            jnp.where(
                start_diving,
                0,
                jnp.where(continue_patrolling, patrol_time + 1, patrol_time)
            )
        )

        horizon_new_speed = self.constants.HORIZON_PATROL_SPEED

        # === MOVEMENT PATTERN 0: STRAIGHT_DOWN ===
        """
        Straight Down Pattern: Simple vertical descent with speed variations.
        - Basic downward movement with kamikaze support
        - Maintains current beam position
        - Speed varies based on retreat/kamikaze state
        """
        straight_mask = white_saucer_active & (movement_pattern == self.constants.WHITE_SAUCER_STRAIGHT_DOWN)
        straight_new_x = current_x

        straight_new_y = jnp.where(
            is_kamikazing,
            current_y + self.constants.WHITE_SAUCER_RAMMING_SPEED,
            jnp.where(
                should_be_moving_up,
                current_y + self.constants.WHITE_SAUCER_REVERSE_SPEED_FAST,
                current_y + current_speed
            )
        )
        straight_new_beam = current_beam
        straight_new_speed = jnp.where(
            is_kamikazing,
            self.constants.WHITE_SAUCER_RAMMING_SPEED,
            jnp.where(
                should_be_moving_up,
                self.constants.WHITE_SAUCER_REVERSE_SPEED_FAST,
                current_speed
            )
        )
        straight_new_target_beam = target_beam

        # === MOVEMENT PATTERN 2: REVERSE_UP ===
        """
        Reverse Up Pattern: Upward retreat movement.
        - Used during retreat phases
        - Fast upward movement with kamikaze override
        - Maintains current beam and target
        """
        reverse_mask = white_saucer_active & (movement_pattern == self.constants.WHITE_SAUCER_REVERSE_UP)
        reverse_new_speed = jnp.where(
            is_kamikazing,
            self.constants.WHITE_SAUCER_RAMMING_SPEED,
            jnp.where(
                should_be_moving_up,
                self.constants.WHITE_SAUCER_REVERSE_SPEED_FAST,
                current_speed
            )
        )
        reverse_new_x = current_x
        reverse_new_y = jnp.where(
            is_kamikazing,
            current_y + self.constants.WHITE_SAUCER_RAMMING_SPEED,
            jnp.where(
                should_be_moving_up,
                current_y + self.constants.WHITE_SAUCER_REVERSE_SPEED_FAST,
                current_y + current_speed
            )
        )
        reverse_new_beam = current_beam
        reverse_new_target_beam = target_beam

        # === MOVEMENT PATTERN 5: RAMMING ===
        """
        Ramming Pattern: High-speed kamikaze attack.
        - Maximum speed downward movement
        - No horizontal deviation
        - Used for aggressive attacks in higher sectors
        """
        ramming_mask = white_saucer_active & (movement_pattern == self.constants.WHITE_SAUCER_RAMMING)
        ramming_new_speed = self.constants.WHITE_SAUCER_RAMMING_SPEED
        ramming_new_x = current_x
        ramming_new_y = current_y + ramming_new_speed
        ramming_new_beam = current_beam
        ramming_new_target_beam = target_beam
        ramming_new_direction_x = direction_x

        # === APPLY MOVEMENT PATTERNS ===
        """
        Unified position and property updates using pattern-based selection.
        Each movement pattern contributes its calculated values, with the final
        state determined by the current movement pattern.
        """
        new_x = jnp.where(horizon_patrol_mask, horizon_new_x,
                          jnp.where(ramming_mask, ramming_new_x,
                                    jnp.where(straight_mask, straight_new_x,
                                              jnp.where(jump_mask, jump_new_x,
                                                        jnp.where(reverse_mask, reverse_new_x,
                                                                  jnp.where(shooting_mask, shooting_new_x,
                                                                            current_x))))))

        new_y = jnp.where(horizon_patrol_mask, horizon_new_y,
                          jnp.where(ramming_mask, ramming_new_y,
                                    jnp.where(straight_mask, straight_new_y,
                                              jnp.where(jump_mask, jump_new_y,
                                                        jnp.where(reverse_mask, reverse_new_y,
                                                                  jnp.where(shooting_mask, shooting_new_y,
                                                                            current_y))))))

        new_beam = jnp.where(horizon_patrol_mask, horizon_new_beam,
                             jnp.where(ramming_mask, ramming_new_beam,
                                       jnp.where(straight_mask, straight_new_beam,
                                                 jnp.where(jump_mask, jump_new_beam,
                                                           jnp.where(reverse_mask, reverse_new_beam,
                                                                     jnp.where(shooting_mask, shooting_new_beam,
                                                                               current_beam))))))

        new_speed = jnp.where(horizon_patrol_mask, horizon_new_speed,
                              jnp.where(ramming_mask, ramming_new_speed,
                                        jnp.where(straight_mask, straight_new_speed,
                                                  jnp.where(jump_mask, jump_new_speed,
                                                            jnp.where(reverse_mask, reverse_new_speed,
                                                                      jnp.where(shooting_mask, shooting_new_speed,
                                                                                current_speed))))))

        new_target_beam = jnp.where(horizon_patrol_mask, horizon_new_target_beam,
                                    jnp.where(ramming_mask, ramming_new_target_beam,
                                              jnp.where(straight_mask, straight_new_target_beam,
                                                        jnp.where(jump_mask, jump_new_target_beam,
                                                                  jnp.where(reverse_mask, reverse_new_target_beam,
                                                                            target_beam)))))

        new_direction_x = jnp.where(horizon_patrol_mask, horizon_new_direction,
                                    jnp.where(ramming_mask, ramming_new_direction_x,
                                              jnp.where(straight_mask, enemies[:, 6],
                                                        jnp.where(jump_mask, enemies[:, 6],
                                                                  jnp.where(reverse_mask, enemies[:, 6],
                                                                            jnp.where(shooting_mask, enemies[:, 6],
                                                                                      enemies[:, 6]))))))

        new_pause_timer = jnp.where(horizon_patrol_mask, horizon_new_pause_timer,
                                    jnp.where(straight_mask, new_jump_timer,
                                              jnp.where(jump_mask, jump_new_jump_timer,
                                                        jnp.where(reverse_mask, new_jump_timer,
                                                                  jnp.where(shooting_mask, new_beam_change_timer,
                                                                            new_jump_timer)))))

        updated_firing_timer = jnp.where(horizon_patrol_mask, horizon_new_patrol_timer, new_firing_timer)

        # === BEAM CURVE ADHERENCE ===
        """
        Critical positioning system that ensures saucers follow the beam curves.
        Only applies to vertically-moving saucers; excludes beam-jumping saucers
        that are actively moving horizontally to maintain realistic movement.
        """
        vertical_phase = white_saucer_active & (
                (straight_mask | reverse_mask | ramming_mask | shooting_mask) |
                (jump_mask & ~actively_jumping)  # Apply curve only when NOT actively jumping
        )

        beam_for_curve = new_beam
        curved_ws_x = self._beam_curve_x(new_y, beam_for_curve, self.constants.ENEMY_WIDTH)
        curved_ws_x = jnp.clip(curved_ws_x, dotted_min_x, dotted_max_x)

        # Apply beam curve positioning to appropriate saucers
        new_x = jnp.where(vertical_phase, curved_ws_x, new_x)

        # === FINAL STATE MANAGEMENT ===
        """
        Final retreat flag updates and enemy deactivation logic.
        Handles the transition between retreat states and manages enemy cleanup.
        """
        will_be_off_top_next = (
                                           current_y + self.constants.WHITE_SAUCER_REVERSE_SPEED_FAST) <= -self.constants.ENEMY_HEIGHT
        start_universal_retreat = white_saucer_active & should_start_retreat

        retreat_flag_next = jnp.where(
            start_universal_retreat, self.constants.WHITE_SAUCER_RETREAT_AFTER_SHOT,
            jnp.where(will_be_off_top_next, 0,
                      jnp.where(should_kamikaze, KAMIKAZE_FLAG, final_retreat_flag))
        )

        is_horizon_patrol = movement_pattern == self.constants.WHITE_SAUCER_HORIZON_PATROL

        # Enemy deactivation logic with kamikaze support
        new_active = white_saucer_active & (
                (is_horizon_patrol & (new_y >= self.constants.HORIZON_LINE_Y)) |
                (~is_horizon_patrol & (new_y > self.constants.HORIZON_LINE_Y)) |
                (is_kamikazing & (new_y < self.constants.SCREEN_HEIGHT))  # Keep kamikaze active until bottom
        )

        # === ENEMY ARRAY UPDATES ===
        """
        Apply all calculated updates to the enemy array.
        Updates positions, speeds, timers, and state flags for all white saucers.
        """
        enemies = enemies.at[:, 0].set(jnp.where(white_saucer_active, new_x, enemies[:, 0]))
        enemies = enemies.at[:, 1].set(jnp.where(white_saucer_active, new_y, enemies[:, 1]))
        enemies = enemies.at[:, 2].set(jnp.where(white_saucer_active, new_beam, enemies[:, 2]))
        enemies = enemies.at[:, 3].set(jnp.where(white_saucer_active, new_active.astype(jnp.float32), enemies[:, 3]))
        enemies = enemies.at[:, 4].set(jnp.where(white_saucer_active, new_speed, enemies[:, 4]))
        enemies = enemies.at[:, 6].set(jnp.where(white_saucer_active, new_direction_x, enemies[:, 6]))
        enemies = enemies.at[:, 10].set(jnp.where(white_saucer_active, new_target_beam, enemies[:, 10]))
        enemies = enemies.at[:, 13].set(jnp.where(white_saucer_active, retreat_flag_next, enemies[:, 13]))
        enemies = enemies.at[:, 14].set(jnp.where(white_saucer_active, new_movement_pattern, enemies[:, 14]))
        enemies = enemies.at[:, 15].set(jnp.where(white_saucer_active, updated_firing_timer, enemies[:, 15]))
        enemies = enemies.at[:, 16].set(jnp.where(white_saucer_active, new_pause_timer, enemies[:, 16]))

        return state.replace(enemies=enemies)
    @partial(jax.jit, static_argnums=(0,))
    def _handle_firing(self, state: BeamRiderState, action: int) -> BeamRiderState:
        """Handle both laser and torpedo firing"""

        # Torpedo firing - T key maps to action 2 (UP)
        should_fire_torpedo = (action == 2)
        state = self._fire_torpedo(state, should_fire_torpedo)

        # Laser firing - SPACE key maps to action 1 (FIRE)
        should_fire_laser = (action == 1)
        state = self._fire_laser(state, should_fire_laser)

        return state

    @partial(jax.jit, static_argnums=(0,))
    def _fire_laser(self, state: BeamRiderState, should_fire: bool) -> BeamRiderState:
        """Fire regular laser projectile"""
        projectiles = state.projectiles
        any_active = jnp.any(projectiles[:, 2] == 1)
        can_fire = ~any_active & should_fire  # only fire if none are active

        # New projectile to be fired
        new_projectile = jnp.array([
            state.ship.x + self.constants.SHIP_WIDTH // 2,  # x
            state.ship.y,  # y
            1,  # active
            -self.constants.PROJECTILE_SPEED,
            state.ship.beam_position
        ])

        # Find first available slot
        active_mask = projectiles[:, 2] == 0
        first_inactive = jnp.argmax(active_mask)

        # Conditionally insert new projectile
        projectiles = jnp.where(
            can_fire,
            projectiles.at[first_inactive].set(new_projectile),
            projectiles
        )

        return state.replace(projectiles=projectiles)

    @partial(jax.jit, static_argnums=(0,))
    def _fire_torpedo(self, state: BeamRiderState, should_fire: bool) -> BeamRiderState:
        """Fire torpedo projectile (if any remaining)"""
        torpedo_projectiles = state.torpedo_projectiles

        # Check if any torpedo slot is available
        any_torpedo_active = jnp.any(torpedo_projectiles[:, 2] == 1)
        has_torpedoes = state.torpedoes_remaining > 0

        # Allow firing if no torpedoes are active and we have torpedoes remaining
        can_fire = ~any_torpedo_active & should_fire & has_torpedoes

        # Find first available slot
        active_mask = torpedo_projectiles[:, 2] == 0  # inactive torpedoes
        first_inactive = jnp.argmax(active_mask)

        # Define the new torpedo
        new_torpedo = jnp.array([
            state.ship.x + self.constants.SHIP_WIDTH // 2,  # Center of ship
            state.ship.y,  # Launch from ship's current y
            1,  # Active
            -self.constants.TORPEDO_SPEED,  # Upward speed
            state.ship.beam_position
        ])

        # Insert new torpedo into first inactive slot, if allowed
        torpedo_projectiles = jnp.where(
            can_fire,
            torpedo_projectiles.at[first_inactive].set(new_torpedo),
            torpedo_projectiles
        )

        # Decrease torpedo count only if a torpedo was fired
        torpedoes_remaining = jnp.where(
            can_fire,
            state.torpedoes_remaining - 1,
            state.torpedoes_remaining
        )

        # Return updated game state
        return state.replace(
            torpedo_projectiles=torpedo_projectiles,
            torpedoes_remaining=torpedoes_remaining
        )

    @partial(jax.jit, static_argnums=(0,))
    def _update_projectiles(self, state: BeamRiderState) -> BeamRiderState:
        """Update all projectiles"""

        # Define update function for single projectile
        def update_single_projectile(x, y, active, speed, beam_idx):
            new_y = y + speed
            new_x = self._beam_curve_x(new_y, beam_idx, self.constants.PROJECTILE_WIDTH)
            new_active = active & (new_y > self.constants.TOP_MARGIN) & (new_y < self.constants.SCREEN_HEIGHT)
            return new_x, new_y, new_active

        # Update regular projectiles
        vmapped_update_laser = jax.vmap(update_single_projectile)
        new_laser_x, new_laser_y, new_laser_active = vmapped_update_laser(
            state.projectiles[:, 0],
            state.projectiles[:, 1],
            state.projectiles[:, 2] == 1,
            state.projectiles[:, 3],
            state.projectiles[:, 4].astype(int)
        )

        projectiles = state.projectiles.at[:, 0].set(new_laser_x)
        projectiles = projectiles.at[:, 1].set(new_laser_y)
        projectiles = projectiles.at[:, 2].set(new_laser_active.astype(jnp.float32))

        # Update torpedo projectiles with different width
        def update_torpedo(x, y, active, speed, beam_idx):
            new_y = y + speed
            new_x = self._beam_curve_x(new_y, beam_idx, self.constants.TORPEDO_WIDTH)
            new_active = active & (new_y > 0) & (new_y < self.constants.SCREEN_HEIGHT)
            return new_x, new_y, new_active

        vmapped_update_torpedo = jax.vmap(update_torpedo)
        new_torp_x, new_torp_y, new_torp_active = vmapped_update_torpedo(
            state.torpedo_projectiles[:, 0],
            state.torpedo_projectiles[:, 1],
            state.torpedo_projectiles[:, 2] == 1,
            state.torpedo_projectiles[:, 3],
            state.torpedo_projectiles[:, 4].astype(int)
        )

        torpedo_projectiles = state.torpedo_projectiles.at[:, 0].set(new_torp_x)
        torpedo_projectiles = torpedo_projectiles.at[:, 1].set(new_torp_y)
        torpedo_projectiles = torpedo_projectiles.at[:, 2].set(new_torp_active.astype(jnp.float32))

        return state.replace(projectiles=projectiles, torpedo_projectiles=torpedo_projectiles)

    @partial(jax.jit, static_argnums=(0,))
    def _update_sentinel_projectiles(self, state: BeamRiderState) -> BeamRiderState:
        """Update sentinel ship projectiles - UPDATED: follow beam curves"""
        sentinel_projectiles = state.sentinel_projectiles

        # Move sentinel projectiles downward
        new_y = sentinel_projectiles[:, 1] + sentinel_projectiles[:, 3]  # y + speed
        beam_indices = sentinel_projectiles[:, 4].astype(int)  # Get beam indices

        # Calculate curved X positions using beam curve with projectile width
        new_x = self._beam_curve_x(new_y, beam_indices, self.constants.PROJECTILE_WIDTH)

        # Deactivate projectiles that go off screen
        active = (
                (sentinel_projectiles[:, 2] == 1) &
                (new_y > 0) &
                (new_y < self.constants.SCREEN_HEIGHT)
        )

        sentinel_projectiles = sentinel_projectiles.at[:, 0].set(new_x)  # Update x position
        sentinel_projectiles = sentinel_projectiles.at[:, 1].set(new_y)  # Update y position
        sentinel_projectiles = sentinel_projectiles.at[:, 2].set(active.astype(jnp.float32))

        return state.replace(sentinel_projectiles=sentinel_projectiles)

    @partial(jax.jit, static_argnums=(0,))
    def _spawn_enemies(self, state: BeamRiderState) -> BeamRiderState:
        """Spawn new enemies with dynamic speed scaling based on sector"""

        # Check if white saucers are complete for this sector
        white_saucers_complete = state.enemies_killed_this_sector >= self.constants.ENEMIES_PER_SECTOR

        # Count currently active white saucers
        active_white_saucers = jnp.sum(
            (state.enemies[:, 3] == 1) &
            (state.enemies[:, 5] == self.constants.ENEMY_TYPE_WHITE_SAUCER)
        )

        # Check if we can spawn more white saucers (max 3 at once)
        can_spawn_white_saucer = active_white_saucers < 3

        # Check if a sentinel ship is currently active
        sentinel_active = jnp.any(
            (state.enemies[:, 3] == 1) & (state.enemies[:, 5] == self.constants.ENEMY_TYPE_SENTINEL_SHIP)
        )
        active_green_blockers = jnp.sum(
            (state.enemies[:, 3] == 1) &
            (state.enemies[:, 5] == self.constants.ENEMY_TYPE_GREEN_BLOCKER)
        )

        state = state.replace(enemy_spawn_timer=state.enemy_spawn_timer + 1)

        # Determine spawning conditions
        normal_enemy_spawn_allowed = ~white_saucers_complete
        blocker_spawn_allowed = white_saucers_complete & sentinel_active

        should_spawn_normal = (state.enemy_spawn_timer >= state.enemy_spawn_interval) & normal_enemy_spawn_allowed

        # More balanced spawn rate for green blockers
        early_sector = state.current_sector <= 5

        # Significantly longer spawn intervals and stricter limits for green blockers
        blocker_spawn_interval = jnp.where(
            early_sector,
            # Early sectors: Much slower spawning
            jnp.maximum(180, state.enemy_spawn_interval * 2),
            # Later sectors: Still conservative
            jnp.maximum(50, state.enemy_spawn_interval + 30)
        )
        max_green_blockers = jnp.where(
            early_sector,
            2,  # Max 2 green blockers in early sectors (sectors 1-5)
            4  # Max 4 green blockers in later sectors
        )
        can_spawn_green_blocker = active_green_blockers < max_green_blockers

        should_spawn_blocker = (
                (state.enemy_spawn_timer >= blocker_spawn_interval) &
                blocker_spawn_allowed &
                can_spawn_green_blocker  # Check blocker limit
        )
        should_spawn = should_spawn_normal | should_spawn_blocker

        # Generate random values
        rng_key, subkey1 = random.split(state.rng_key)
        rng_key, subkey2 = random.split(rng_key)
        rng_key, subkey3 = random.split(rng_key)

        # Determine enemy type using normal enemy selection (includes white saucers at normal rates)
        enemy_type = jnp.where(
            should_spawn_blocker,
            self.constants.ENEMY_TYPE_GREEN_BLOCKER,
            self._select_enemy_type_excluding_blockers_early_sectors(state.current_sector, subkey1)
        )

        # Check if selected enemy is white saucer when we can't spawn them
        # If so, skip this spawn frame entirely (maintaining normal spawn rates for other enemies)
        selected_white_saucer_when_cant = (
                                                      enemy_type == self.constants.ENEMY_TYPE_WHITE_SAUCER) & ~can_spawn_white_saucer

        # Can actually spawn if: we should spawn AND we didn't select a white saucer when we can't spawn one
        can_actually_spawn = should_spawn & ~selected_white_saucer_when_cant

        # Reset spawn timer when spawn attempt was made (even if we skipped due to white saucer limit)
        # Reset spawn timer based on what type of spawn was attempted
        reset_for_blocker = should_spawn_blocker
        reset_for_normal = should_spawn_normal

        new_spawn_timer = jnp.where(
            reset_for_blocker | reset_for_normal,
            0,
            state.enemy_spawn_timer
        )
        state = state.replace(enemy_spawn_timer=new_spawn_timer)

        # Find inactive enemy slot
        enemies = state.enemies
        active_mask = enemies[:, 3] == 0

        # YELLOW CHIRPERS: Special side spawning with random Y position
        is_yellow_chirper = enemy_type == self.constants.ENEMY_TYPE_YELLOW_CHIRPER

        # Choose spawn side randomly (0 = left, 1 = right)
        rng_key, spawn_side_key = random.split(rng_key)
        spawn_from_right = random.randint(spawn_side_key, (), 0, 2)  # 0 or 1

        # Choose random Y position within chirper range
        rng_key, spawn_y_key = random.split(rng_key)
        chirper_spawn_y = random.uniform(
            spawn_y_key, (),
            minval=self.constants.YELLOW_CHIRPER_SPAWN_Y_MIN,
            maxval=self.constants.YELLOW_CHIRPER_SPAWN_Y_MAX,
            dtype=jnp.float32
        )

        # Yellow chirper spawn positions (from sides, off-screen)
        chirper_spawn_x = jnp.where(
            spawn_from_right,
            self.constants.SCREEN_WIDTH + self.constants.ENEMY_WIDTH,  # Spawn from right side (off-screen)
            -self.constants.ENEMY_WIDTH  # Spawn from left side (off-screen)
        )

        # GREEN BLOCKERS: Target fixed X coordinate (where player was when blocker spawned)
        is_green_blocker = enemy_type == self.constants.ENEMY_TYPE_GREEN_BLOCKER

        # Get player beam position (simpler since ship can only be on beam centers)
        player_beam_x = self.beam_positions[state.ship.beam_position]
        player_beam_index = state.ship.beam_position  # NEW: Store the beam index too

        # Choose spawn side randomly for blockers (0 = left, 1 = right)
        rng_key, blocker_side_key = random.split(rng_key)
        blocker_spawn_from_right = random.randint(blocker_side_key, (), 0, 2)  # 0 or 1

        # Green blocker spawn positions (from sides)
        blocker_spawn_x = jnp.where(
            blocker_spawn_from_right,
            self.constants.SCREEN_WIDTH + self.constants.ENEMY_WIDTH,  # Spawn from right side (off-screen)
            -self.constants.ENEMY_WIDTH  # Spawn from left side (off-screen)
        )

        # Green blockers spawn at upper third of screen (higher up)
        blocker_spawn_y = self.constants.SCREEN_HEIGHT // 3

        # GREEN BOUNCE CRAFT: NEW SPAWN LOGIC
        is_green_bounce = enemy_type == self.constants.ENEMY_TYPE_GREEN_BOUNCE

        # Choose spawn side randomly (0 = left, 1 = right)
        rng_key, bounce_side_key = random.split(rng_key)
        bounce_spawn_from_right = random.randint(bounce_side_key, (), 0, 2)  # 0 or 1

        # Choose random Y position in upper area of screen
        rng_key, bounce_y_key = random.split(rng_key)
        bounce_spawn_y = random.uniform(
            bounce_y_key, (),
            minval=self.constants.SCREEN_HEIGHT * 0.15,  # Upper area
            maxval=self.constants.SCREEN_HEIGHT * 0.25,  # Keep them high
            dtype=jnp.float32
        )

        # Green bounce craft spawn positions (from sides, off-screen)
        bounce_spawn_x = jnp.where(
            bounce_spawn_from_right,
            self.constants.SCREEN_WIDTH + self.constants.ENEMY_WIDTH,  # Spawn off right edge
            -self.constants.ENEMY_WIDTH  # Spawn off left edge
        )

        # Determine initial target beam (first or last depending on spawn side)
        initial_bounce_target_beam = jnp.where(
            bounce_spawn_from_right,
            self.constants.NUM_BEAMS - 1,  # Start from rightmost beam if spawning from right
            0  # Start from leftmost beam if spawning from left
        )

        # Set movement direction (0=left-to-right, 1=right-to-left)
        bounce_movement_direction = bounce_spawn_from_right.astype(jnp.float32)

        # ORANGE TRACKERS: Spawn from sides, move horizontally first
        is_orange_tracker = enemy_type == self.constants.ENEMY_TYPE_ORANGE_TRACKER

        # Choose spawn side randomly (0 = left, 1 = right)
        rng_key, tracker_side_key = random.split(rng_key)
        tracker_spawn_from_right = random.randint(tracker_side_key, (), 0, 2)  # 0 or 1

        # Orange tracker spawn positions (from sides, off-screen)
        tracker_spawn_x = jnp.where(
            tracker_spawn_from_right,
            self.constants.SCREEN_WIDTH + self.constants.ENEMY_WIDTH,  # Spawn from right side (off-screen)
            -self.constants.ENEMY_WIDTH  # Spawn from left side (off-screen)
        )

        # Spawn at a reasonable height (upper third of screen)
        rng_key, tracker_y_key = random.split(rng_key)
        tracker_spawn_y = random.uniform(
            tracker_y_key, (),
            minval=self.constants.SCREEN_HEIGHT * 0.2,  # Upper area
            maxval=self.constants.SCREEN_HEIGHT * 0.35,  # But not too high
            dtype=jnp.float32
        )

        # YELLOW REJUVENATORS: Spawn from top on random beam
        is_yellow_rejuvenator = enemy_type == self.constants.ENEMY_TYPE_YELLOW_REJUVENATOR

        # Choose random beam for rejuvenator spawning
        rng_key, rejuv_beam_key = random.split(rng_key)
        rejuv_spawn_beam = random.randint(rejuv_beam_key, (), 0, self.constants.NUM_BEAMS)
        rejuv_spawn_x = self.beam_positions[rejuv_spawn_beam]
        rejuv_spawn_y = self.constants.HORIZON_LINE_Y

        # Regular enemy spawn (from top, random beam)
        regular_spawn_beam = random.randint(subkey3, (), 0, self.constants.NUM_BEAMS)
        regular_spawn_x = self.beam_positions[regular_spawn_beam]
        regular_spawn_y = self.constants.HORIZON_LINE_Y

        # Choose final spawn position based on enemy type
        spawn_x = jnp.where(
            is_yellow_chirper,
            chirper_spawn_x,  # Chirpers spawn from sides
            jnp.where(
                is_green_blocker,
                blocker_spawn_x,  # Blockers spawn from sides
                jnp.where(
                    is_green_bounce,
                    bounce_spawn_x,  # Bounce craft spawn from sides
                    jnp.where(
                        is_orange_tracker,
                        tracker_spawn_x,  # Orange trackers spawn from sides
                        jnp.where(
                            is_yellow_rejuvenator,
                            rejuv_spawn_x,  # Rejuvenators spawn from top
                            regular_spawn_x  # Regular enemies spawn from top
                        )
                    )
                )
            )
        )

        spawn_y = jnp.where(
            is_yellow_chirper,
            chirper_spawn_y,  # Chirpers use random Y in their range
            jnp.where(
                is_green_blocker,
                blocker_spawn_y,  # Blockers spawn high
                jnp.where(
                    is_green_bounce,
                    bounce_spawn_y,  # Bounce craft spawn in upper area
                    jnp.where(
                        is_orange_tracker,
                        tracker_spawn_y,  # Orange trackers use their spawn Y
                        jnp.where(
                            is_yellow_rejuvenator,
                            rejuv_spawn_y,  # Rejuvenators spawn from top
                            regular_spawn_y  # Regular enemies spawn from top
                        )
                    )
                )
            )
        )

        # Calculate enemy speed with sector scaling
        base_speed = jnp.where(
            enemy_type == self.constants.ENEMY_TYPE_WHITE_SAUCER,
            self.constants.ENEMY_SPEED,
            jnp.where(
                enemy_type == self.constants.ENEMY_TYPE_BROWN_DEBRIS,
                self.constants.BROWN_DEBRIS_SPEED,
                jnp.where(
                    enemy_type == self.constants.ENEMY_TYPE_YELLOW_CHIRPER,
                    self.constants.YELLOW_CHIRPER_SPEED,
                    jnp.where(
                        enemy_type == self.constants.ENEMY_TYPE_GREEN_BLOCKER,
                        self.constants.GREEN_BLOCKER_SPEED,
                        jnp.where(
                            enemy_type == self.constants.ENEMY_TYPE_GREEN_BOUNCE,
                            self.constants.GREEN_BOUNCE_SPEED,
                            jnp.where(
                                enemy_type == self.constants.ENEMY_TYPE_BLUE_CHARGER,
                                self.constants.BLUE_CHARGER_SPEED,
                                jnp.where(
                                    enemy_type == self.constants.ENEMY_TYPE_ORANGE_TRACKER,
                                    self.constants.ORANGE_TRACKER_SPEED,
                                    jnp.where(
                                        enemy_type == self.constants.ENEMY_TYPE_YELLOW_REJUVENATOR,
                                        self.constants.YELLOW_REJUVENATOR_SPEED,
                                        self.constants.ENEMY_SPEED  # Default
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )

        # Smooth speed scaling across 99 sectors
        # Linear scaling factor from 1.0 (sector 1) to 2.5 (sector 99)
        speed_scale_factor = 1.0 + ((state.current_sector - 1) / 98.0) * 1.5  # Goes from 1.0 to 2.5
        final_enemy_speed = base_speed * speed_scale_factor

        direction_y = 1.0  # All enemies move down by default

        # Calculate final spawn beam position for tracking
        final_spawn_beam = jnp.where(
            is_yellow_chirper,
            0,  # Side spawners that don't use beam positions
            jnp.where(
                is_green_blocker,
                state.ship.beam_position,  # Store player's beam index for blockers
                jnp.where(
                    is_green_bounce,
                    initial_bounce_target_beam,  # Bounce craft uses target beam
                    jnp.where(
                        is_orange_tracker,
                        0,  # Orange trackers don't track beam initially (use state machine)
                        regular_spawn_beam  # Use calculated spawn beam
                    )
                )
            )
        )

        # Enemy health (sentinel ships have 1 health, others have 1)
        enemy_health = jnp.where(
            enemy_type == self.constants.ENEMY_TYPE_SENTINEL_SHIP,
            self.constants.SENTINEL_SHIP_HEALTH,
            1  # All other enemies
        )

        # Calculate course changes for orange trackers
        base_changes = self.constants.ORANGE_TRACKER_BASE_COURSE_CHANGES
        bonus_changes = (state.current_sector - 1) // self.constants.ORANGE_TRACKER_COURSE_CHANGE_INCREASE_SECTOR
        course_changes_remaining = jnp.where(
            enemy_type == self.constants.ENEMY_TYPE_ORANGE_TRACKER,
            base_changes + bonus_changes,
            jnp.where(
                is_green_bounce,
                bounce_movement_direction,  # Store movement direction for bounce craft
                0  # Not applicable for other enemies
            )
        )

        # White Saucer movement patterns
        is_white_saucer = enemy_type == self.constants.ENEMY_TYPE_WHITE_SAUCER

        # Select movement pattern for white saucers
        movement_pattern = jnp.where(
            is_white_saucer,
            self._select_white_saucer_movement_pattern(subkey2),
            jnp.where(
                is_green_bounce,
                0,  # Start bounce craft in state 0 (moving to beam)
                0  # Default pattern for non-white saucers
            )
        )

        # White Saucer Ramming: Override pattern for higher sectors
        is_ramming_available = state.current_sector >= self.constants.WHITE_SAUCER_RAMMING_MIN_SECTOR
        ramming_chance = jnp.where(
            state.current_sector >= self.constants.WHITE_SAUCER_RAMMING_INCREASED_CHANCE_SECTOR,
            self.constants.WHITE_SAUCER_RAMMING_HIGH_SECTOR_CHANCE,
            self.constants.WHITE_SAUCER_RAMMING_CHANCE
        )

        # Generate additional random value for ramming check
        rng_key, ramming_key = random.split(rng_key)
        ramming_rand = random.uniform(ramming_key, (), minval=0.0, maxval=1.0)
        should_ram = is_white_saucer & is_ramming_available & (ramming_rand < ramming_chance)

        # Override movement pattern if ramming
        movement_pattern = jnp.where(
            should_ram,
            self.constants.WHITE_SAUCER_RAMMING,
            movement_pattern
        )

        # Set initial firing timer for shooting white saucers
        can_shoot = movement_pattern == self.constants.WHITE_SAUCER_SHOOTING
        initial_firing_timer = jnp.where(
            is_white_saucer & can_shoot,
            self.constants.WHITE_SAUCER_FIRING_INTERVAL,
            0
        )

        # Generate random dive depth for shooting white saucers (15-40 pixels below horizon)
        rng_key, dive_depth_key = random.split(rng_key)
        shooting_dive_depth = random.uniform(
            dive_depth_key, (),
            minval=40.0,
            maxval=80.0,
            dtype=jnp.float32
        )

        # Store dive depth in direction_y field for shooting white saucers (they don't use direction_y otherwise)
        initial_direction_y = jnp.where(
            is_white_saucer & can_shoot,
            shooting_dive_depth,  # Store random dive depth for shooting saucers
            direction_y  # Keep default 1.0 for others
        )

        # Set initial jump timer for jumping white saucers
        can_jump = movement_pattern == self.constants.WHITE_SAUCER_BEAM_JUMP
        initial_jump_timer = jnp.where(
            is_white_saucer & can_jump,
            self.constants.WHITE_SAUCER_JUMP_INTERVAL,
            0
        )

        # Add Horizon Patrol Initialization
        # Set initial direction and target for horizon patrol saucers
        is_horizon_patrol = movement_pattern == self.constants.WHITE_SAUCER_HORIZON_PATROL

        # Generate random direction for horizon patrol (-1 or 1)
        rng_key, patrol_dir_key = random.split(rng_key)
        random_direction = jnp.where(
            random.uniform(patrol_dir_key, (), minval=0.0, maxval=1.0, dtype=jnp.float32) < 0.5, -1.0, 1.0
        )

        # Generate random target beam offset (1-3 lanes away)
        rng_key, patrol_target_key = random.split(rng_key)
        target_offset = random.randint(patrol_target_key, (), minval=1, maxval=4) * random_direction

        # Calculate initial target beam for horizon patrol
        horizon_target_beam = jnp.clip(
            regular_spawn_beam + target_offset.astype(int),
            0, self.constants.NUM_BEAMS - 1
        )

        # Modify existing direction_x assignment to include horizon patrol and orange trackers
        direction_x = jnp.where(
            is_yellow_chirper,
            jnp.where(spawn_from_right, -1.0, 1.0),  # Chirpers move toward center
            jnp.where(
                is_green_blocker,
                jnp.where(blocker_spawn_from_right, -1.0, 1.0),  # Blockers move toward player
                jnp.where(
                    is_green_bounce,
                    1.0,  # Bounce craft will use state machine for direction
                    jnp.where(
                        is_orange_tracker,
                        jnp.where(tracker_spawn_from_right, -1.0, 1.0),  # Move toward center
                        jnp.where(
                            is_white_saucer & is_horizon_patrol,
                            random_direction,  # Horizon patrol gets random direction
                            1.0  # Default right movement for others
                        )
                    )
                )
            )
        )

        target_x = jnp.where(
            is_green_blocker,
            player_beam_x,  # Blocker targets player's current position
            jnp.where(
                is_orange_tracker,
                -1,  # No initial target beam - will be set when switching to vertical
                jnp.where(
                    is_green_bounce,
                    bounce_spawn_y,  # Store spawn height for bounce craft
                    jnp.where(
                        is_white_saucer & is_horizon_patrol,
                        horizon_target_beam,  # Store target beam in target_x field for horizon patrol
                        jnp.where(
                            is_white_saucer & can_jump,
                            regular_spawn_beam,  # Initialize beam jumpers with their spawn beam
                            0.0  # Default for others
                        )
                    )
                )
            )
        )

        # For beam jumping white saucers, set an initial target beam different from spawn beam
        rng_key, jump_target_key = random.split(rng_key)
        jump_target_offset = random.randint(jump_target_key, (), minval=1, maxval=self.constants.NUM_BEAMS)
        initial_jump_target = (regular_spawn_beam + jump_target_offset) % self.constants.NUM_BEAMS

        # Update target_x for beam jumping saucers to have a different initial target
        target_x = jnp.where(
            is_white_saucer & can_jump,
            initial_jump_target,  # Set to a different beam than spawn beam
            target_x
        )

        # Use column 12 to indicate orange tracker hasn't started vertical movement yet (0 = horizontal, 1 = vertical)
        tracker_movement_phase = jnp.where(
            is_orange_tracker,
            0,  # Start in horizontal movement phase
            0  # Default for others
        )

        # Create new enemy array
        new_enemy = jnp.zeros(17, dtype=jnp.float32)  # 17 columns for all enemy data
        new_enemy = new_enemy.at[0].set(spawn_x)  # x
        new_enemy = new_enemy.at[1].set(spawn_y)  # y
        new_enemy = new_enemy.at[2].set(final_spawn_beam)  # beam_position (or target beam for trackers/bounce)
        new_enemy = new_enemy.at[3].set(1)  # active
        new_enemy = new_enemy.at[4].set(final_enemy_speed)  # speed
        new_enemy = new_enemy.at[5].set(enemy_type)  # type
        new_enemy = new_enemy.at[6].set(direction_x)  # direction_x
        new_enemy = new_enemy.at[7].set(initial_direction_y)  # direction_y (or dive depth for shooting white saucers)
        new_enemy = new_enemy.at[8].set(0)  # bounce_count/lock status (0 for most enemies)
        new_enemy = new_enemy.at[9].set(0)  # linger_timer
        new_enemy = new_enemy.at[10].set(target_x)  # target_x (or spawn height for bounce craft)
        new_enemy = new_enemy.at[11].set(enemy_health)  # health
        new_enemy = new_enemy.at[12].set(
            tracker_movement_phase)  # firing_timer/debris lifetime/movement phase for trackers
        new_enemy = new_enemy.at[13].set(course_changes_remaining)  # course changes (or movement direction for bounce)
        new_enemy = new_enemy.at[14].set(movement_pattern)  # movement_pattern (or state machine for bounce craft)
        new_enemy = new_enemy.at[15].set(initial_firing_timer)  # white_saucer_firing_timer
        new_enemy = new_enemy.at[16].set(initial_jump_timer)  # jump_timer

        # Find first inactive enemy and place new enemy only if can_actually_spawn is True
        first_inactive = jnp.argmax(active_mask)
        can_spawn_enemy = active_mask[first_inactive] & can_actually_spawn

        enemies = jnp.where(
            can_spawn_enemy,
            enemies.at[first_inactive].set(new_enemy),
            enemies
        )

        return state.replace(enemies=enemies, rng_key=rng_key)

    @partial(jax.jit, static_argnums=(0,))
    def _handle_white_saucer_limit(self, enemy_type: int, sector: int, can_spawn_white_saucer: bool,
                                   rng_key: chex.PRNGKey) -> tuple:
        """Handle white saucer limit - either select alternative or prevent spawning"""

        # If enemy type is white saucer but we can't spawn more
        white_saucer_limit_reached = (enemy_type == self.constants.ENEMY_TYPE_WHITE_SAUCER) & ~can_spawn_white_saucer

        # Check what other enemy types are available in this sector
        has_alternative_enemies = self._has_available_non_white_enemies(sector)

        # If white saucer limit reached and we have alternatives, select alternative
        # If white saucer limit reached and no alternatives, don't spawn anything
        new_enemy_type = jnp.where(
            white_saucer_limit_reached & has_alternative_enemies,
            self._select_non_white_saucer_enemy_type(sector, rng_key),
            enemy_type  # Keep original type if no limit issue
        )

        # Can only spawn if:
        # 1. Original enemy type is fine (not white saucer or white saucer limit not reached), or
        # 2. White saucer limit reached but we have alternatives
        can_spawn = ~white_saucer_limit_reached | (white_saucer_limit_reached & has_alternative_enemies)

        return new_enemy_type, can_spawn

    @partial(jax.jit, static_argnums=(0,))
    def _has_available_non_white_enemies(self, sector: int) -> bool:
        """Check if any non-white-saucer enemies are available in current sector"""

        brown_debris_available = sector >= self.constants.BROWN_DEBRIS_SPAWN_SECTOR  # Sector 2
        yellow_chirper_available = sector >= self.constants.YELLOW_CHIRPER_SPAWN_SECTOR  # Sector 4
        green_bounce_available = sector >= self.constants.GREEN_BOUNCE_SPAWN_SECTOR  # Sector 7
        blue_charger_available = sector >= self.constants.BLUE_CHARGER_SPAWN_SECTOR  # Sector 10
        orange_tracker_available = sector >= self.constants.ORANGE_TRACKER_SPAWN_SECTOR  # Sector 12
        yellow_rejuvenator_available = sector >= self.constants.YELLOW_REJUVENATOR_SPAWN_SECTOR  # Sector 5
        green_blocker_available = sector > 5  # Sector 6+

        # Return True if any other enemy type is available
        return (brown_debris_available | yellow_chirper_available | green_bounce_available |
                blue_charger_available | orange_tracker_available | yellow_rejuvenator_available |
                green_blocker_available)
    # Select non-white saucer enemy types when white saucer limit is reached
    @partial(jax.jit, static_argnums=(0,))
    @partial(jax.jit, static_argnums=(0,))
    def _select_non_white_saucer_enemy_type(self, sector: int, rng_key: chex.PRNGKey) -> int:
        """Select enemy type excluding white saucers (when white saucer limit is reached)

        This method should only be called when _has_available_non_white_enemies returns True.
        It will select from available enemy types based on sector rules.
        """

        # Generate random value for enemy type selection
        rand_val = random.uniform(rng_key, (), minval=0.0, maxval=1.0, dtype=jnp.float32)

        # Check availability based on sector
        brown_debris_available = sector >= self.constants.BROWN_DEBRIS_SPAWN_SECTOR  # Sector 2
        yellow_chirper_available = sector >= self.constants.YELLOW_CHIRPER_SPAWN_SECTOR  # Sector 4
        green_bounce_available = sector >= self.constants.GREEN_BOUNCE_SPAWN_SECTOR  # Sector 7
        blue_charger_available = sector >= self.constants.BLUE_CHARGER_SPAWN_SECTOR  # Sector 10
        orange_tracker_available = sector >= self.constants.ORANGE_TRACKER_SPAWN_SECTOR  # Sector 12
        yellow_rejuvenator_available = sector >= self.constants.YELLOW_REJUVENATOR_SPAWN_SECTOR  # Sector 5
        green_blocker_available = sector > 5  # Sector 6+

        # Calculate spawn probabilities (excluding white saucers)
        brown_debris_chance = jnp.where(brown_debris_available, self.constants.BROWN_DEBRIS_SPAWN_CHANCE, 0.0)
        yellow_chirper_chance = jnp.where(yellow_chirper_available, self.constants.YELLOW_CHIRPER_SPAWN_CHANCE, 0.0)
        green_blocker_chance = jnp.where(green_blocker_available, self.constants.GREEN_BLOCKER_SPAWN_CHANCE, 0.0)
        green_bounce_chance = jnp.where(green_bounce_available, self.constants.GREEN_BOUNCE_SPAWN_CHANCE, 0.0)
        blue_charger_chance = jnp.where(blue_charger_available, self.constants.BLUE_CHARGER_SPAWN_CHANCE, 0.0)
        orange_tracker_chance = jnp.where(orange_tracker_available, self.constants.ORANGE_TRACKER_SPAWN_CHANCE, 0.0)
        yellow_rejuvenator_chance = jnp.where(yellow_rejuvenator_available,
                                              self.constants.YELLOW_REJUVENATOR_SPAWN_CHANCE, 0.0)

        # Calculate total probability
        total_chance = (brown_debris_chance + yellow_chirper_chance + green_blocker_chance +
                        green_bounce_chance + blue_charger_chance + orange_tracker_chance +
                        yellow_rejuvenator_chance)

        # Normalize probabilities (total_chance should be > 0 since this method is only called
        # when _has_available_non_white_enemies returns True, but we'll be safe)
        norm_factor = jnp.maximum(total_chance, 0.001)  # Small value to prevent division by zero

        brown_debris_norm = brown_debris_chance / norm_factor
        yellow_chirper_norm = yellow_chirper_chance / norm_factor
        green_blocker_norm = green_blocker_chance / norm_factor
        green_bounce_norm = green_bounce_chance / norm_factor
        blue_charger_norm = blue_charger_chance / norm_factor
        orange_tracker_norm = orange_tracker_chance / norm_factor
        yellow_rejuvenator_norm = yellow_rejuvenator_chance / norm_factor

        # Calculate cumulative probabilities
        yellow_rejuvenator_threshold = yellow_rejuvenator_norm
        orange_tracker_threshold = yellow_rejuvenator_threshold + orange_tracker_norm
        blue_charger_threshold = orange_tracker_threshold + blue_charger_norm
        bounce_threshold = blue_charger_threshold + green_bounce_norm
        blocker_threshold = bounce_threshold + green_blocker_norm
        chirper_threshold = blocker_threshold + yellow_chirper_norm
        debris_threshold = chirper_threshold + brown_debris_norm

        # Select enemy type using thresholds (no white saucers)
        enemy_type = jnp.where(
            rand_val < yellow_rejuvenator_threshold,
            self.constants.ENEMY_TYPE_YELLOW_REJUVENATOR,
            jnp.where(
                rand_val < orange_tracker_threshold,
                self.constants.ENEMY_TYPE_ORANGE_TRACKER,
                jnp.where(
                    rand_val < blue_charger_threshold,
                    self.constants.ENEMY_TYPE_BLUE_CHARGER,
                    jnp.where(
                        rand_val < bounce_threshold,
                        self.constants.ENEMY_TYPE_GREEN_BOUNCE,
                        jnp.where(
                            rand_val < blocker_threshold,
                            self.constants.ENEMY_TYPE_GREEN_BLOCKER,
                            jnp.where(
                                rand_val < chirper_threshold,
                                self.constants.ENEMY_TYPE_YELLOW_CHIRPER,
                                self.constants.ENEMY_TYPE_BROWN_DEBRIS  # Default fallback
                            )
                        )
                    )
                )
            )
        )

        return enemy_type

    @partial(jax.jit, static_argnums=(0,))
    def _calculate_enemy_speed(self, base_speed: float, current_sector: int) -> float:
        """Calculate enemy speed based on sector with smooth scaling"""
        # Scale from base_speed to MAX_ENEMY_SPEED over 99 sectors
        progress_ratio = jnp.minimum((current_sector - 1) / 98.0, 1.0)  # Cap at 1.0

        # Use square root scaling for gradual increase that accelerates later
        speed_multiplier = 1.0 + (jnp.sqrt(progress_ratio) * 2.0)  # 1.0x to 3.0x multiplier

        final_speed = jnp.minimum(
            base_speed * speed_multiplier,
            self.constants.MAX_ENEMY_SPEED
        )

        return final_speed

    @partial(jax.jit, static_argnums=(0,))
    def _select_enemy_type_excluding_blockers_early_sectors(self, sector: int, rng_key: chex.PRNGKey) -> int:
        """Select enemy type - FIXED: Full spawn chance for yellow rejuvenators"""

        # Generate random value for enemy type selection
        rand_val = random.uniform(rng_key, (), minval=0.0, maxval=1.0, dtype=jnp.float32)

        # Check availability based on sector
        brown_debris_available = sector >= self.constants.BROWN_DEBRIS_SPAWN_SECTOR
        yellow_chirper_available = sector >= self.constants.YELLOW_CHIRPER_SPAWN_SECTOR
        green_bounce_available = sector >= self.constants.GREEN_BOUNCE_SPAWN_SECTOR
        blue_charger_available = sector >= self.constants.BLUE_CHARGER_SPAWN_SECTOR
        orange_tracker_available = sector >= self.constants.ORANGE_TRACKER_SPAWN_SECTOR
        yellow_rejuvenator_available = sector >= self.constants.YELLOW_REJUVENATOR_SPAWN_SECTOR
        green_blocker_available = sector > 5

        # Calculate spawn probabilities: Full spawn chance for yellow rejuvenators
        brown_debris_chance = jnp.where(brown_debris_available, self.constants.BROWN_DEBRIS_SPAWN_CHANCE * 0.5, 0.0)
        yellow_chirper_chance = jnp.where(yellow_chirper_available, self.constants.YELLOW_CHIRPER_SPAWN_CHANCE * 0.5,
                                          0.0)
        green_blocker_chance = jnp.where(green_blocker_available, self.constants.GREEN_BLOCKER_SPAWN_CHANCE * 0.5, 0.0)
        green_bounce_chance = jnp.where(green_bounce_available, self.constants.GREEN_BOUNCE_SPAWN_CHANCE * 0.5, 0.0)
        blue_charger_chance = jnp.where(blue_charger_available, self.constants.BLUE_CHARGER_SPAWN_CHANCE * 0.5, 0.0)
        orange_tracker_chance = jnp.where(orange_tracker_available, self.constants.ORANGE_TRACKER_SPAWN_CHANCE * 0.5,0.0)

        # Remove the * 0.5 multiplier for yellow rejuvenators
        yellow_rejuvenator_chance = jnp.where(yellow_rejuvenator_available,
                                              self.constants.YELLOW_REJUVENATOR_SPAWN_CHANCE,  # Full spawn chance
                                              0.0)

        # Leave more probability for white saucers
        total_special_chance = (brown_debris_chance + yellow_chirper_chance + green_blocker_chance +
                                green_bounce_chance + blue_charger_chance + orange_tracker_chance +
                                yellow_rejuvenator_chance)

        # Calculate cumulative thresholds - Put yellow rejuvenators first for higher priority
        yellow_rejuvenator_threshold = yellow_rejuvenator_chance
        orange_tracker_threshold = yellow_rejuvenator_threshold + orange_tracker_chance
        blue_charger_threshold = orange_tracker_threshold + blue_charger_chance
        green_bounce_threshold = blue_charger_threshold + green_bounce_chance
        green_blocker_threshold = green_bounce_threshold + green_blocker_chance
        yellow_chirper_threshold = green_blocker_threshold + yellow_chirper_chance
        brown_debris_threshold = yellow_chirper_threshold + brown_debris_chance
        # Remaining probability goes to white saucers

        # Select enemy type using thresholds
        enemy_type = jnp.where(
            rand_val < yellow_rejuvenator_threshold,
            self.constants.ENEMY_TYPE_YELLOW_REJUVENATOR,
            jnp.where(
                rand_val < orange_tracker_threshold,
                self.constants.ENEMY_TYPE_ORANGE_TRACKER,
                jnp.where(
                    rand_val < blue_charger_threshold,
                    self.constants.ENEMY_TYPE_BLUE_CHARGER,
                    jnp.where(
                        rand_val < green_bounce_threshold,
                        self.constants.ENEMY_TYPE_GREEN_BOUNCE,
                        jnp.where(
                            rand_val < green_blocker_threshold,
                            self.constants.ENEMY_TYPE_GREEN_BLOCKER,
                            jnp.where(
                                rand_val < yellow_chirper_threshold,
                                self.constants.ENEMY_TYPE_YELLOW_CHIRPER,
                                jnp.where(
                                    rand_val < brown_debris_threshold,
                                    self.constants.ENEMY_TYPE_BROWN_DEBRIS,
                                    self.constants.ENEMY_TYPE_WHITE_SAUCER  # Default fallback
                                )
                            )
                        )
                    )
                )
            )
        )

        return enemy_type

    @partial(jax.jit, static_argnums=(0,))
    def _select_enemy_type(self, sector: int, rng_key: chex.PRNGKey) -> int:
        """Select enemy type based on current sector"""

        # Generate random value for enemy type selection
        rand_val = random.uniform(rng_key, (), minval=0.0, maxval=1.0, dtype=jnp.float32)

        # Check availability based on sector
        brown_debris_available = sector >= self.constants.BROWN_DEBRIS_SPAWN_SECTOR
        yellow_chirper_available = sector >= self.constants.YELLOW_CHIRPER_SPAWN_SECTOR
        green_blocker_available = sector >= self.constants.GREEN_BLOCKER_SPAWN_SECTOR
        green_bounce_available = sector >= self.constants.GREEN_BOUNCE_SPAWN_SECTOR
        blue_charger_available = sector >= self.constants.BLUE_CHARGER_SPAWN_SECTOR
        orange_tracker_available = sector >= self.constants.ORANGE_TRACKER_SPAWN_SECTOR

        # Calculate spawn probabilities
        brown_debris_chance = jnp.where(brown_debris_available, self.constants.BROWN_DEBRIS_SPAWN_CHANCE, 0.0)
        yellow_chirper_chance = jnp.where(yellow_chirper_available, self.constants.YELLOW_CHIRPER_SPAWN_CHANCE, 0.0)
        green_blocker_chance = jnp.where(green_blocker_available, self.constants.GREEN_BLOCKER_SPAWN_CHANCE, 0.0)
        green_bounce_chance = jnp.where(green_bounce_available, self.constants.GREEN_BOUNCE_SPAWN_CHANCE, 0.0)
        blue_charger_chance = jnp.where(blue_charger_available, self.constants.BLUE_CHARGER_SPAWN_CHANCE, 0.0)
        orange_tracker_chance = jnp.where(orange_tracker_available, self.constants.ORANGE_TRACKER_SPAWN_CHANCE, 0.0)

        # Calculate cumulative probabilities
        orange_tracker_threshold = orange_tracker_chance
        blue_charger_threshold = orange_tracker_threshold + blue_charger_chance
        bounce_threshold = blue_charger_threshold + green_bounce_chance
        blocker_threshold = bounce_threshold + green_blocker_chance
        chirper_threshold = blocker_threshold + yellow_chirper_chance
        debris_threshold = chirper_threshold + brown_debris_chance

        # Select enemy type using thresholds
        enemy_type = jnp.where(
            rand_val < orange_tracker_threshold,
            self.constants.ENEMY_TYPE_ORANGE_TRACKER,
            jnp.where(
                rand_val < blue_charger_threshold,
                self.constants.ENEMY_TYPE_BLUE_CHARGER,
                jnp.where(
                    rand_val < bounce_threshold,
                    self.constants.ENEMY_TYPE_GREEN_BOUNCE,
                    jnp.where(
                        rand_val < blocker_threshold,
                        self.constants.ENEMY_TYPE_GREEN_BLOCKER,
                        jnp.where(
                            rand_val < chirper_threshold,
                            self.constants.ENEMY_TYPE_YELLOW_CHIRPER,
                            jnp.where(
                                rand_val < debris_threshold,
                                self.constants.ENEMY_TYPE_BROWN_DEBRIS,
                                self.constants.ENEMY_TYPE_WHITE_SAUCER  # Default
                            )
                        )
                    )
                )
            )
        )

        return enemy_type

    @partial(jax.jit, static_argnums=(0,))
    def _beam_curve_x(self, y: chex.Array, beam_idx: chex.Array, entity_width: int = None) -> chex.Array:
        """Beam Center x at vertical position y (matches renderer's perspective)."""
        width = self.constants.SCREEN_WIDTH
        height = self.constants.SCREEN_HEIGHT
        center_x = width / 2.0

        # same margins as renderer
        top_margin = int(height * 0.12)
        bottom_margin = int(height * 0.14)
        y0 = height - bottom_margin
        y1 = -height * 0.7
        t_top = jnp.clip((top_margin - y0) / (y1 - y0), 0.0, 1.0)

        t = jnp.clip((y - y0) / (y1 - y0), 0.0, t_top)
        x0 = self.beam_positions[beam_idx]
        x1 = center_x + (x0 - center_x) * 0.05
        x = x0 + (x1 - x0) * t
        return x

    @partial(jax.jit, static_argnums=(0,))
    def _update_enemies(self, state: BeamRiderState) -> BeamRiderState:
        """Update enemy positions"""

        # Handle white saucer movement patterns
        state = self._update_white_saucer_movement(state)
        enemies = state.enemies  # Get updated enemies array after white saucer movement

        # Extract enemy data for use throughout the method
        active_mask = enemies[:, 3] == 1
        current_x = enemies[:, 0]
        current_y = enemies[:, 1]
        current_speed = enemies[:, 4]
        enemy_types = enemies[:, 5]  # Get enemy types
        direction_x = enemies[:, 6]
        direction_y = enemies[:, 7]
        linger_timer = enemies[:, 9].astype(int)

        # Check for white saucer activity (they're handled separately)
        white_saucer_active = active_mask & (enemy_types == self.constants.ENEMY_TYPE_WHITE_SAUCER)

        # Remove WHITE_SAUCER from regular_enemy_mask since they're handled separately now
        regular_enemy_mask = ((enemy_types == self.constants.ENEMY_TYPE_BROWN_DEBRIS) |
                              (
                                          enemy_types == self.constants.ENEMY_TYPE_YELLOW_REJUVENATOR))  # Added yellow rejuvenators
        regular_new_y = enemies[:, 1] + enemies[:, 4]  # y + speed

        # Yellow chirpers move horizontally
        chirper_mask = enemy_types == self.constants.ENEMY_TYPE_YELLOW_CHIRPER
        chirper_new_x = enemies[:, 0] + enemies[:, 4]  # x + speed (horizontal movement)

        # Green blockers: complex targeting behavior: prevent jittering
        blocker_mask = enemy_types == self.constants.ENEMY_TYPE_GREEN_BLOCKER

        # Get blocker current positions and targets
        blocker_x = enemies[:, 0]
        blocker_y = enemies[:, 1]
        blocker_target_x = enemies[:, 10]  # Fixed target_x stored when spawned
        blocker_direction_x = enemies[:, 6]  # movement direction
        blocker_beam_idx = enemies[:, 2].astype(int)  # Beam they should follow when moving down

        # Use bounce_count field to track if blocker has "locked" onto vertical movement
        blocker_locked_vertical = enemies[:, 8] == 1  # 1 = locked in vertical mode, 0 = still moving horizontally

        # Calculate movement toward fixed target X coordinate
        distance_to_target = jnp.abs(blocker_x - blocker_target_x)
        reached_target = distance_to_target < (self.constants.GREEN_BLOCKER_SPEED * 2)  # Close enough threshold

        # Once reached target, lock into vertical movement mode (prevent switching back)
        new_blocker_locked = jnp.where(
            blocker_mask & (reached_target | blocker_locked_vertical),
            1,  # Lock into vertical mode
            0  # Stay in horizontal mode
        )

        # Phase 1: Horizontal movement (only when not locked in vertical mode)
        should_move_horizontally = blocker_mask & ~blocker_locked_vertical & ~reached_target

        blocker_new_x_horizontal = jnp.where(
            should_move_horizontally,
            blocker_x + (blocker_direction_x * self.constants.GREEN_BLOCKER_SPEED),
            blocker_x  # No horizontal movement
        )

        # Phase 2: Vertical movement with beam curve (when locked in vertical mode)
        should_move_vertically = blocker_mask & (blocker_locked_vertical | reached_target)

        blocker_new_y_vertical = jnp.where(
            should_move_vertically,
            blocker_y + self.constants.GREEN_BLOCKER_SPEED,
            blocker_y  # No vertical movement yet
        )

        # Calculate beam curve position for vertical movement
        blocker_new_x_curved = self._beam_curve_x(blocker_new_y_vertical, blocker_beam_idx, self.constants.ENEMY_WIDTH)

        # Final positions: use curve when moving vertically, horizontal when moving horizontally
        blocker_new_x = jnp.where(
            should_move_vertically,
            blocker_new_x_curved,  # Use beam curve when moving down
            blocker_new_x_horizontal  # Use horizontal movement when approaching target
        )

        blocker_new_y = blocker_new_y_vertical  # Always update Y (will be unchanged if not moving vertically)

        # Green Bounce Craft: movement pattern
        bounce_mask = enemy_types == self.constants.ENEMY_TYPE_GREEN_BOUNCE
        bounce_x = enemies[:, 0]
        bounce_y = enemies[:, 1]

        # Use column 2 to track current target beam index
        current_target_beam = enemies[:, 2].astype(int)

        # Use column 14 to track movement state (0=moving to beam, 1=checking, 2=descending, 3=ascending)
        bounce_state = enemies[:, 14].astype(int)

        # Use column 10 to store initial spawn height
        spawn_height = enemies[:, 10]

        # Use column 13 to track if moving left-to-right (0) or right-to-left (1)
        movement_direction = enemies[:, 13].astype(int)

        # Constants for movement
        BEAM_THRESHOLD = 8.0
        HORIZONTAL_SPEED = 2.0
        VERTICAL_SPEED = 1.5
        CHECK_DEPTH = self.constants.WHITE_SAUCER_REVERSE_TRIGGER_Y

        # Get target beam position
        target_beam_x = self.beam_positions[jnp.clip(current_target_beam, 0, self.constants.NUM_BEAMS - 1)]

        # Check if we're at the target beam
        at_target_beam = jnp.abs(bounce_x - target_beam_x) < BEAM_THRESHOLD

        # Check if player is on current beam
        player_beam = state.ship.beam_position
        player_on_current_beam = (current_target_beam == player_beam)

        # State 0: Moving horizontally to target beam
        moving_to_beam = bounce_state == 0
        new_x_moving = bounce_x + jnp.sign(target_beam_x - bounce_x) * HORIZONTAL_SPEED

        # Transition from state 0 to state 1 when reaching beam
        new_state_from_0 = jnp.where(at_target_beam, 1, 0)

        # State 1: At beam, checking (move down to check depth)
        checking = bounce_state == 1
        new_y_checking = bounce_y + VERTICAL_SPEED
        at_check_depth = bounce_y >= CHECK_DEPTH

        # Decide next state from checking
        new_state_from_1 = jnp.where(
            at_check_depth,
            jnp.where(player_on_current_beam, 2, 3),  # 2=descend if player on beam, 3=ascend if not
            1  # Keep checking if not at depth yet
        )

        # State 2: Descending (player was on beam)
        descending = bounce_state == 2
        new_y_descending = bounce_y + VERTICAL_SPEED

        # Always use beam curve for descending bounce crafts
        descending_curved_x = self._beam_curve_x(new_y_descending, current_target_beam, self.constants.ENEMY_WIDTH)

        # Deactivate if reached bottom
        reached_bottom = new_y_descending >= self.constants.SCREEN_HEIGHT

        # State 3: Ascending back to spawn height
        ascending = bounce_state == 3
        new_y_ascending = bounce_y - VERTICAL_SPEED
        at_spawn_height = bounce_y <= spawn_height

        # Always use beam curve for ascending bounce crafts too
        ascending_curved_x = self._beam_curve_x(new_y_ascending, current_target_beam, self.constants.ENEMY_WIDTH)

        # Determine next beam
        next_beam = jnp.where(
            movement_direction == 0,  # Moving left to right
            jnp.where(
                current_target_beam >= self.constants.NUM_BEAMS - 1,
                -1,  # Signal to deactivate (reached right edge)
                current_target_beam + 1
            ),
            jnp.where(  # Moving right to left
                current_target_beam <= 0,
                -1,  # Signal to deactivate (reached left edge)
                current_target_beam - 1
            )
        )

        # Transition from state 3 when back at spawn height
        new_state_from_3 = jnp.where(
            at_spawn_height,
            jnp.where(next_beam >= 0, 0, -1),  # Back to moving state or deactivate
            3  # Keep ascending
        )

        # Update target beam when transitioning to next
        new_target_beam = jnp.where(
            (bounce_state == 3) & at_spawn_height & (next_beam >= 0),
            next_beam,
            current_target_beam
        )

        # Calculate final positions based on state
        final_bounce_x = jnp.where(
            moving_to_beam,
            new_x_moving,  # Horizontal movement to beam
            jnp.where(
                checking,
                self._beam_curve_x(new_y_checking, current_target_beam, self.constants.ENEMY_WIDTH),
                # Use curve while checking
                jnp.where(
                    descending,
                    descending_curved_x,  # Use pre-calculated curved position
                    jnp.where(
                        ascending,
                        ascending_curved_x,  # Use pre-calculated curved position
                        bounce_x  # Default
                    )
                )
            )
        )

        final_bounce_y = jnp.where(
            moving_to_beam,
            bounce_y,  # Stay at same Y while moving horizontally
            jnp.where(
                checking,
                new_y_checking,  # Move down while checking
                jnp.where(
                    descending,
                    new_y_descending,  # Move down while descending
                    jnp.where(
                        ascending,
                        new_y_ascending,  # Move up while ascending
                        bounce_y
                    )
                )
            )
        )

        # Update state machine
        new_bounce_state = jnp.where(
            moving_to_beam,
            new_state_from_0,
            jnp.where(
                checking,
                new_state_from_1,
                jnp.where(
                    descending,
                    2,  # Stay in descending state
                    jnp.where(
                        ascending,
                        new_state_from_3,
                        bounce_state
                    )
                )
            )
        )

        # Deactivate conditions for bounce craft
        bounce_should_deactivate = bounce_mask & (
                reached_bottom |  # Hit bottom while descending
                (new_bounce_state == -1) |  # Reached edge of screen
                ((bounce_state == 3) & at_spawn_height & (next_beam < 0))  # No more beams
        )

        bounce_craft_active = (enemies[:, 3] == 1) & ~bounce_should_deactivate

        # Orange Trackers: horizontal movement then vertical tracking
        # Orange Trackers: horizontal movement across screen, then vertical tracking
        tracker_mask = enemy_types == self.constants.ENEMY_TYPE_ORANGE_TRACKER

        # Get tracker data
        tracker_x = enemies[:, 0]
        tracker_y = enemies[:, 1]
        tracker_movement_phase = enemies[:, 12].astype(int)  # 0 = horizontal, 1 = vertical
        tracker_direction_x = enemies[:, 6]  # Horizontal movement direction (-1 or 1)
        tracker_course_changes_remaining = enemies[:, 13].astype(int)  # Course changes left
        tracker_current_beam = enemies[:, 10].astype(int)  # Current beam being tracked (only valid in vertical phase)

        # Get current player beam position
        current_player_beam = state.ship.beam_position
        current_player_beam_x = self.beam_positions[current_player_beam]

        # Phase 1: Horizontal movement across screen
        in_horizontal_phase = tracker_mask & (tracker_movement_phase == 0)

        # Check if tracker is currently over the player's beam (within threshold)
        tracker_over_player_beam = jnp.abs(tracker_x - current_player_beam_x) < 8.0

        # Switch to vertical phase when passing over player's beam
        should_switch_to_vertical = in_horizontal_phase & tracker_over_player_beam
        new_movement_phase = jnp.where(
            should_switch_to_vertical,
            1,  # Switch to vertical tracking phase
            tracker_movement_phase
        )

        # Horizontal movement (Phase 0) - just move straight across
        tracker_new_x_horizontal = jnp.where(
            in_horizontal_phase,
            tracker_x + (tracker_direction_x * self.constants.ORANGE_TRACKER_SPEED * 2.0),  # Faster horizontal movement
            tracker_x
        )

        # Phase 2: Vertical movement with tracking
        in_vertical_phase = tracker_mask & (new_movement_phase == 1)

        # When switching to vertical, lock onto current player beam
        new_tracker_current_beam = jnp.where(
            should_switch_to_vertical,
            current_player_beam,  # Lock onto player's beam when starting vertical
            jnp.where(
                in_vertical_phase,
                tracker_current_beam,  # Keep tracking during vertical phase
                -1  # Invalid beam during horizontal phase
            )
        )

        # Check if player changed beams and tracker can still change course
        player_changed_beam = (new_tracker_current_beam != current_player_beam) & (new_tracker_current_beam >= 0)
        can_change_course = tracker_course_changes_remaining > 0
        should_change_course = in_vertical_phase & player_changed_beam & can_change_course

        # Update tracked beam when changing course
        new_target_beam_tracker = jnp.where(
            should_change_course,
            current_player_beam,  # Follow player to new beam
            new_tracker_current_beam  # Keep current tracked beam
        )

        # Decrease course changes remaining when used
        new_course_changes_remaining = jnp.where(
            should_change_course,
            tracker_course_changes_remaining - 1,
            tracker_course_changes_remaining
        )

        # Vertical phase movement (with horizontal adjustment to track beam)
        target_beam_x_for_vertical = self.beam_positions[
            jnp.clip(new_target_beam_tracker, 0, self.constants.NUM_BEAMS - 1)]

        # Calculate movement for vertical phase
        distance_to_beam = jnp.abs(tracker_x - target_beam_x_for_vertical)
        at_target_beam = distance_to_beam < 3.0

        # Horizontal adjustment while moving down (only in vertical phase)
        horizontal_adjustment = jnp.where(
            in_vertical_phase & ~at_target_beam,
            jnp.sign(target_beam_x_for_vertical - tracker_x) * jnp.minimum(
                self.constants.ORANGE_TRACKER_SPEED * 1.2,  # Horizontal speed while tracking
                distance_to_beam * 0.3
            ),
            0.0
        )

        tracker_new_x_vertical = jnp.where(
            in_vertical_phase,
            tracker_x + horizontal_adjustment,
            tracker_x
        )

        # Final X position
        tracker_new_x = jnp.where(
            in_horizontal_phase,
            tracker_new_x_horizontal,  # Just move straight across
            jnp.where(
                in_vertical_phase,
                tracker_new_x_vertical,  # Track player beam while descending
                tracker_x
            )
        )

        # Y movement (only in vertical phase)
        tracker_new_y = jnp.where(
            in_vertical_phase,
            tracker_y + self.constants.ORANGE_TRACKER_SPEED,  # Move down at normal speed
            tracker_y  # Stay at same Y during horizontal phase
        )

        # Check if tracker has reached bottom (they should disappear here)
        tracker_at_bottom = tracker_new_y >= self.constants.SCREEN_HEIGHT

        # Deactivation conditions - include bottom and off-screen
        tracker_off_screen = (tracker_new_x < -self.constants.ENEMY_WIDTH) | (
                    tracker_new_x > self.constants.SCREEN_WIDTH)
        tracker_active = (enemies[:, 3] == 1) & ~tracker_at_bottom & ~tracker_off_screen

        # Blue Chargers:
        charger_mask = enemy_types == self.constants.ENEMY_TYPE_BLUE_CHARGER
        charger_linger_timer = enemies[:, 9].astype(int)  # linger_timer column

        # Define bottom position where chargers should stop
        bottom_position = self.constants.SCREEN_HEIGHT - self.constants.ENEMY_HEIGHT - 10

        # Check if charger has reached or passed bottom position
        charger_reached_bottom = enemies[:, 1] >= bottom_position

        # If speed is positive, move down. If speed is negative (deflected), move up.
        charger_new_y = jnp.where(
            charger_mask & charger_reached_bottom,
            bottom_position,  # Stay exactly at bottom position when reached
            enemies[:, 1] + enemies[:, 4]  # Normal movement: current Y + speed (can be + or -)
        )

        # Sentinel Ship: horizontal cruise using direction
        sentinel_mask = enemy_types == self.constants.ENEMY_TYPE_SENTINEL_SHIP
        sentinel_direction_x = enemies[:, 6]  # Get direction from direction_x field
        sentinel_new_x = enemies[:, 0] + (sentinel_direction_x * enemies[:, 4])  # Move using direction * speed
        sentinel_new_y = enemies[:, 1]  # Stay at same Y level

        # =================================================================
        # Rejuvenator Debris: Move straight down
        # =================================================================
        debris_mask = active_mask & (enemy_types == self.constants.ENEMY_TYPE_REJUVENATOR_DEBRIS)

        # Debris moves straight down at constant speed (follows beam curve)
        debris_new_x = current_x  # Will be updated by beam curve below
        debris_new_y = current_y + current_speed  # Move down at debris speed

        # Debris active state: only deactivate when reaching bottom of screen
        debris_active = (enemies[:, 3] == 1) & (debris_new_y < self.constants.SCREEN_HEIGHT)

        # =================================================================
        # Linger Timer Updates
        # =================================================================

        # Update linger timer - handles both blue chargers and debris lifetime
        new_linger_timer = jnp.where(
            charger_mask & charger_reached_bottom & (charger_linger_timer == 0),
            self.constants.BLUE_CHARGER_LINGER_TIME,  # Start lingering when first reaching bottom
            jnp.where(
                charger_mask & charger_reached_bottom & (charger_linger_timer > 0),
                charger_linger_timer - 1,  # Count down while at bottom
                linger_timer  # Keep current value for others (debris keeps 0)
            )
        )

        # =================================================================
        # Combine all movement patterns
        # =================================================================

        # Update X positions based on enemy type (WHITE SAUCERS EXCLUDED - handled separately)
        new_x = jnp.where(
            chirper_mask,
            chirper_new_x,  # Chirpers move horizontally
            jnp.where(
                blocker_mask,
                blocker_new_x,  # Blockers use fixed X-coordinate targeting
                jnp.where(
                    bounce_mask,
                    final_bounce_x,  # Bounce craft - NEW movement pattern
                    jnp.where(
                        charger_mask,
                        enemies[:, 0],  # Blue chargers don't change X position
                        jnp.where(
                            tracker_mask,
                            tracker_new_x,  # Orange trackers use beam following
                            jnp.where(
                                sentinel_mask,
                                sentinel_new_x,  # Sentinels move horizontally
                                jnp.where(
                                    debris_mask,
                                    debris_new_x,  # Debris moves in explosion pattern
                                    enemies[:, 0]  # Default: no X change (includes white saucers and regular enemies)
                                )
                            )
                        )
                    )
                )
            )
        )

        # Update Y positions based on enemy type
        new_y = jnp.where(
            regular_enemy_mask & ~charger_mask & ~tracker_mask,  # Regular enemies (brown debris + yellow rejuvenators)
            regular_new_y,  # Regular enemies move down
            jnp.where(
                blocker_mask,
                blocker_new_y,  # Blockers use fixed X-coordinate targeting Y movement
                jnp.where(
                    bounce_mask,
                    final_bounce_y,
                    jnp.where(
                        charger_mask,
                        charger_new_y,
                        jnp.where(
                            tracker_mask,
                            tracker_new_y,  # Orange trackers use beam following Y movement
                            jnp.where(
                                sentinel_mask,
                                sentinel_new_y,  # Sentinels stay at same Y
                                jnp.where(
                                    debris_mask,
                                    debris_new_y,  # Debris moves in explosion pattern
                                    enemies[:, 1]  # Default: no Y change (includes white saucers AND chirpers)
                                )
                            )
                        )
                    )
                )
            )
        )

        # Make vertical movers follow dotted beams (perspective curve)
        beam_idx_now = enemies[:, 2].astype(int)

        # FIXED: Only apply beam curve to trackers when they're in vertical phase
        tracker_in_vertical_phase = tracker_mask & (new_movement_phase == 1)

        vertical_mask = (
                                (regular_enemy_mask | charger_mask | debris_mask)  # Removed tracker_mask from here
                                & ~chirper_mask & ~bounce_mask & ~sentinel_mask & ~blocker_mask
                        ) | tracker_in_vertical_phase  # Add only vertical-phase trackers

        curved_x = self._beam_curve_x(new_y, beam_idx_now, self.constants.ENEMY_WIDTH)
        new_x = jnp.where(vertical_mask, curved_x, new_x)
        # =================================================================
        # Active State Calculations
        # =================================================================

        # Regular enemies: deactivate when they go below screen (brown debris + yellow rejuvenators)
        regular_active = (enemies[:, 3] == 1) & (regular_new_y < self.constants.SCREEN_HEIGHT)

        # Orange trackers: deactivate when they reach bottom of screen or go off sides
        tracker_active = (enemies[:, 3] == 1) & ~tracker_at_bottom & ~tracker_off_screen

        # Chirpers: deactivate when they go off either side
        chirper_active = (enemies[:, 3] == 1) & (chirper_new_x > -self.constants.ENEMY_WIDTH) & (
                chirper_new_x < self.constants.SCREEN_WIDTH + self.constants.ENEMY_WIDTH)

        # Green blockers: deactivate when they go off any edge or reach bottom
        blocker_active = (enemies[:, 3] == 1) & \
                         (blocker_new_x > -self.constants.ENEMY_WIDTH) & \
                         (blocker_new_x < self.constants.SCREEN_WIDTH + self.constants.ENEMY_WIDTH) & \
                         (blocker_new_y > -self.constants.ENEMY_HEIGHT) & \
                         (blocker_new_y < self.constants.SCREEN_HEIGHT)

        # Bounce craft stay active unless deactivation conditions met
        bounce_active = bounce_craft_active

        # Stay active until they reach bottom and linger timer expires, or until they go off top when deflected
        charger_off_top = charger_new_y < -self.constants.ENEMY_HEIGHT  # Deflected chargers going off top
        charger_active = (enemies[:, 3] == 1) & ~charger_off_top & (
                (~charger_reached_bottom) |  # Still moving down, stay active
                (charger_reached_bottom & (new_linger_timer > 0))  # At bottom but timer not expired
        )

        # Sentinel ships: deactivate when they go completely off screen (either side)
        sentinel_off_screen = (
                (sentinel_new_x > (
                        self.constants.SCREEN_WIDTH + self.constants.SENTINEL_SHIP_WIDTH)) |  # Off right side
                (sentinel_new_x < (-self.constants.SENTINEL_SHIP_WIDTH))  # Off left side
        )
        sentinel_active = (enemies[:, 3] == 1) & ~sentinel_off_screen

        # Combine active states based on enemy type
        active = jnp.where(
            white_saucer_active,
            white_saucer_active,  # White saucers handled by their own logic
            jnp.where(
                regular_enemy_mask & ~charger_mask & ~tracker_mask,
                regular_active,  # Regular enemies: deactivate when below screen
                jnp.where(
                    chirper_mask,
                    chirper_active,  # Chirpers: deactivate when off either side
                    jnp.where(
                        blocker_mask,
                        blocker_active,  # Blockers: deactivate when below screen
                        jnp.where(
                            bounce_mask,
                            bounce_active,  # Bounce craft: deactivate based on state machine
                            jnp.where(
                                charger_mask,
                                charger_active,  # Blue chargers: deactivate when below screen
                                jnp.where(
                                    tracker_mask,
                                    tracker_active,  # Orange trackers: deactivate when at bottom
                                    jnp.where(
                                        sentinel_mask,
                                        sentinel_active,  # Sentinels: handled separately
                                        jnp.where(
                                            debris_mask,
                                            debris_active,  # Debris active until lifetime expires
                                            enemies[:, 3]  # Default: keep current active state
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )

        # Update enemy array
        enemies = enemies.at[:, 0].set(new_x)  # Update x positions
        enemies = enemies.at[:, 1].set(new_y)  # Update y positions
        enemies = enemies.at[:, 2].set(  # Update target beam for trackers and bounce craft
            jnp.where(
                tracker_mask,
                new_target_beam_tracker,
                jnp.where(
                    bounce_mask,
                    new_target_beam,
                    enemies[:, 2]
                )
            )
        )
        enemies = enemies.at[:, 3].set(active.astype(jnp.float32))  # Update active states

        # Update blocker lock status (stored in column 8)
        enemies = enemies.at[:, 8].set(
            jnp.where(
                blocker_mask,
                new_blocker_locked,  # Update lock status for green blockers
                enemies[:, 8]  # Keep existing values for others
            )
        )

        enemies = enemies.at[:, 9].set(new_linger_timer)  # Update linger timer

        # Update target X for trackers (stored in column 10) - NOW STORING BEAM INDEX
        enemies = enemies.at[:, 10].set(
            jnp.where(
                tracker_mask,
                new_target_beam_tracker,  # Store target beam index for trackers
                enemies[:, 10]  # Keep existing values for others (including bounce craft spawn height)
            )
        )

        # Update movement phase for trackers (column 12)
        enemies = enemies.at[:, 12].set(
            jnp.where(
                tracker_mask,
                new_movement_phase,  # Update movement phase for trackers
                enemies[:, 12]
            )
        )

        # Update course changes remaining for trackers (column 13)
        enemies = enemies.at[:, 13].set(
            jnp.where(
                tracker_mask,
                new_course_changes_remaining,  # Update course changes for trackers
                enemies[:, 13]  # Keep existing values for others (including bounce craft movement direction)
            )
        )

        # Update bounce craft state machine (column 14)
        enemies = enemies.at[:, 14].set(
            jnp.where(
                bounce_mask,
                new_bounce_state,
                enemies[:, 14]
            )
        )

        return state.replace(enemies=enemies)

    @partial(jax.jit, static_argnums=(0,))
    def _check_collisions(self, state: BeamRiderState) -> BeamRiderState:
        """Check for collisions - OPTIMIZED with vmap for parallel collision detection"""
        projectiles = state.projectiles
        torpedo_projectiles = state.torpedo_projectiles
        sentinel_projectiles = state.sentinel_projectiles
        enemies = state.enemies
        score = state.score

        # Enemy properties
        enemy_active = enemies[:, 3] == 1
        enemy_types = enemies[:, 5]
        enemy_x = enemies[:, 0]
        enemy_y = enemies[:, 1]

        # Enemy dimensions
        enemy_widths = jnp.where(
            enemy_types == self.constants.ENEMY_TYPE_SENTINEL_SHIP,
            self.constants.SENTINEL_SHIP_WIDTH,
            self.constants.ENEMY_WIDTH
        )
        enemy_heights = jnp.where(
            enemy_types == self.constants.ENEMY_TYPE_SENTINEL_SHIP,
            self.constants.SENTINEL_SHIP_HEIGHT,
            self.constants.ENEMY_HEIGHT
        )

        # Protection masks
        at_horizon = jnp.abs(enemy_y - self.constants.HORIZON_LINE_Y) <= 15
        white_saucer_protected = (enemy_types == self.constants.ENEMY_TYPE_WHITE_SAUCER) & at_horizon

        enemy_vulnerable_to_lasers = (
                ((enemy_types == self.constants.ENEMY_TYPE_WHITE_SAUCER) & ~white_saucer_protected) |
                (enemy_types == self.constants.ENEMY_TYPE_YELLOW_CHIRPER) |
                (enemy_types == self.constants.ENEMY_TYPE_BLUE_CHARGER)
        )

        enemy_blocks_lasers = (
                (enemy_types == self.constants.ENEMY_TYPE_BROWN_DEBRIS) |
                (enemy_types == self.constants.ENEMY_TYPE_GREEN_BLOCKER) |
                (enemy_types == self.constants.ENEMY_TYPE_SENTINEL_SHIP) |
                (enemy_types == self.constants.ENEMY_TYPE_ORANGE_TRACKER) |
                (enemy_types == self.constants.ENEMY_TYPE_GREEN_BOUNCE) |
                white_saucer_protected
        )

        enemy_interacts_with_lasers = enemy_vulnerable_to_lasers | enemy_blocks_lasers

        # VMAP: Define single collision check function
        def check_single_collision(proj_x, proj_y, proj_w, proj_h,
                                   enemy_x, enemy_y, enemy_w, enemy_h):
            return (
                    (proj_x < enemy_x + enemy_w) &
                    (proj_x + proj_w > enemy_x) &
                    (proj_y < enemy_y + enemy_h) &
                    (proj_y + proj_h > enemy_y)
            )

        # VMAP: Create collision checking functions for all pairs
        check_proj_vs_all_enemies = jax.vmap(
            check_single_collision,
            in_axes=(None, None, None, None, 0, 0, 0, 0)
        )

        check_all_laser_collisions = jax.vmap(
            check_proj_vs_all_enemies,
            in_axes=(0, 0, None, None, None, None, None, None)
        )

        # Check laser collisions
        proj_active = projectiles[:, 2] == 1
        laser_collision_matrix = check_all_laser_collisions(
            projectiles[:, 0], projectiles[:, 1],
            self.constants.PROJECTILE_WIDTH, self.constants.PROJECTILE_HEIGHT,
            enemy_x, enemy_y, enemy_widths, enemy_heights
        )

        # Apply masks
        laser_collision_matrix = (
                laser_collision_matrix &
                proj_active[:, None] &
                enemy_active[None, :] &
                enemy_interacts_with_lasers[None, :]
        )

        laser_proj_hits = jnp.any(laser_collision_matrix, axis=1)
        laser_damage_collisions = laser_collision_matrix & enemy_vulnerable_to_lasers[None, :]
        laser_enemy_hits = jnp.any(laser_damage_collisions, axis=0)

        # Blue charger deflection
        charger_hits = laser_enemy_hits & (enemy_types == self.constants.ENEMY_TYPE_BLUE_CHARGER)
        enemies = enemies.at[:, 4].set(
            jnp.where(charger_hits, self.constants.BLUE_CHARGER_DEFLECT_SPEED, enemies[:, 4])
        )
        laser_enemy_hits = laser_enemy_hits & ~charger_hits

        # VMAP: Torpedo collisions
        check_all_torpedo_collisions = jax.vmap(
            check_proj_vs_all_enemies,
            in_axes=(0, 0, None, None, None, None, None, None)
        )

        torpedo_active = torpedo_projectiles[:, 2] == 1
        torpedo_collision_matrix = check_all_torpedo_collisions(
            torpedo_projectiles[:, 0], torpedo_projectiles[:, 1],
            self.constants.TORPEDO_WIDTH, self.constants.TORPEDO_HEIGHT,
            enemy_x, enemy_y, enemy_widths, enemy_heights
        )

        torpedo_collision_matrix = (
                torpedo_collision_matrix &
                torpedo_active[:, None] &
                enemy_active[None, :] &
                ~white_saucer_protected[None, :]
        )

        torpedo_proj_hits = jnp.any(torpedo_collision_matrix, axis=1)
        torpedo_enemy_hits = jnp.any(torpedo_collision_matrix, axis=0)

        # Sentinel health handling
        sentinel_hits = torpedo_enemy_hits & (enemy_types == self.constants.ENEMY_TYPE_SENTINEL_SHIP)
        enemies = enemies.at[:, 11].set(
            jnp.where(sentinel_hits, jnp.maximum(0, enemies[:, 11] - 1), enemies[:, 11])
        )
        sentinel_destroyed = sentinel_hits & (enemies[:, 11] == 0)
        torpedo_enemy_hits = jnp.where(
            enemy_types == self.constants.ENEMY_TYPE_SENTINEL_SHIP,
            sentinel_destroyed,
            torpedo_enemy_hits
        )

        # VMAP: Score calculation
        def calculate_enemy_score(hit, enemy_type, is_torpedo):
            laser_points = jnp.select(
                [
                    enemy_type == self.constants.ENEMY_TYPE_WHITE_SAUCER,
                    enemy_type == self.constants.ENEMY_TYPE_YELLOW_CHIRPER,
                ],
                [self.constants.POINTS_PER_ENEMY, self.constants.YELLOW_CHIRPER_POINTS],
                default=0
            )

            torpedo_points = jnp.select(
                [
                    enemy_type == self.constants.ENEMY_TYPE_WHITE_SAUCER,
                    enemy_type == self.constants.ENEMY_TYPE_BROWN_DEBRIS,
                    enemy_type == self.constants.ENEMY_TYPE_YELLOW_CHIRPER,
                    enemy_type == self.constants.ENEMY_TYPE_GREEN_BLOCKER,
                    enemy_type == self.constants.ENEMY_TYPE_GREEN_BOUNCE,
                    enemy_type == self.constants.ENEMY_TYPE_BLUE_CHARGER,
                    enemy_type == self.constants.ENEMY_TYPE_ORANGE_TRACKER,
                    enemy_type == self.constants.ENEMY_TYPE_SENTINEL_SHIP,
                ],
                [
                    self.constants.POINTS_PER_ENEMY * 2,
                    self.constants.BROWN_DEBRIS_POINTS,
                    self.constants.YELLOW_CHIRPER_POINTS,
                    self.constants.GREEN_BLOCKER_POINTS,
                    self.constants.GREEN_BOUNCE_POINTS,
                    self.constants.BLUE_CHARGER_POINTS,
                    self.constants.ORANGE_TRACKER_POINTS,
                    self.constants.SENTINEL_SHIP_POINTS,
                ],
                default=0
            )

            return jnp.where(hit, jnp.where(is_torpedo, torpedo_points, laser_points), 0)

        vmapped_score = jax.vmap(calculate_enemy_score, in_axes=(0, 0, None))

        laser_scores = vmapped_score(laser_enemy_hits, enemy_types, False)
        torpedo_scores = vmapped_score(torpedo_enemy_hits, enemy_types, True)

        total_score = jnp.sum(laser_scores) + jnp.sum(torpedo_scores)
        sentinel_bonus = jnp.sum(sentinel_hits) * (state.lives * 100)
        score += total_score + sentinel_bonus

        # VMAP: Ship collision checks
        ship_x, ship_y = state.ship.x, state.ship.y

        def check_ship_enemy_collision(enemy_x, enemy_y, enemy_w, enemy_h, active, can_collide):
            return (
                    (ship_x < enemy_x + enemy_w) &
                    (ship_x + self.constants.SHIP_WIDTH > enemy_x) &
                    (ship_y < enemy_y + enemy_h) &
                    (ship_y + self.constants.SHIP_HEIGHT > enemy_y) &
                    active & can_collide
            )

        can_collide_with_ship = (
                (enemy_types != self.constants.ENEMY_TYPE_YELLOW_CHIRPER) &
                (enemy_types != self.constants.ENEMY_TYPE_SENTINEL_SHIP)
        )

        vmapped_ship_collision = jax.vmap(check_ship_enemy_collision)
        ship_collisions = vmapped_ship_collision(
            enemy_x, enemy_y, enemy_widths, enemy_heights,
            enemy_active, can_collide_with_ship
        )

        # Sentinel projectile collisions --> COULD BE REMOVED
        def check_sentinel_proj_ship(proj_x, proj_y, active):
            return (
                    (ship_x < proj_x + self.constants.PROJECTILE_WIDTH) &
                    (ship_x + self.constants.SHIP_WIDTH > proj_x) &
                    (ship_y < proj_y + self.constants.PROJECTILE_HEIGHT) &
                    (ship_y + self.constants.SHIP_HEIGHT > proj_y) &
                    active
            )


        vmapped_sentinel_check = jax.vmap(check_sentinel_proj_ship)
        sentinel_proj_ship_collisions = vmapped_sentinel_check(
            sentinel_projectiles[:, 0],
            sentinel_projectiles[:, 1],
            sentinel_projectiles[:, 2] == 1
        )

        any_ship_collision = jnp.any(ship_collisions) | jnp.any(sentinel_proj_ship_collisions)

        # Update game state
        total_enemy_hits = laser_enemy_hits | torpedo_enemy_hits
        white_saucer_hits = total_enemy_hits & (enemy_types == self.constants.ENEMY_TYPE_WHITE_SAUCER)

        lives = jnp.where(any_ship_collision, state.lives - 1, state.lives)
        center_beam = self.constants.INITIAL_BEAM

        ship = state.ship.replace(
            x=jnp.where(any_ship_collision,
                        self.beam_positions[center_beam] - self.constants.SHIP_WIDTH // 2,
                        state.ship.x),
            beam_position=jnp.where(any_ship_collision, center_beam, state.ship.beam_position),
            target_beam=jnp.where(any_ship_collision, center_beam, state.ship.beam_position)
        )

        # Update arrays
        projectiles = projectiles.at[:, 2].set(projectiles[:, 2] * (~laser_proj_hits))
        torpedo_projectiles = torpedo_projectiles.at[:, 2].set(torpedo_projectiles[:, 2] * (~torpedo_proj_hits))
        sentinel_projectiles = sentinel_projectiles.at[:, 2].set(
            sentinel_projectiles[:, 2] * (~sentinel_proj_ship_collisions)
        )
        enemies = enemies.at[:, 3].set(enemies[:, 3] * (~total_enemy_hits) * (~ship_collisions))

        return state.replace(
            projectiles=projectiles,
            torpedo_projectiles=torpedo_projectiles,
            sentinel_projectiles=sentinel_projectiles,
            enemies=enemies,
            score=score,
            ship=ship,
            lives=lives,
            enemies_killed_this_sector=state.enemies_killed_this_sector + jnp.sum(white_saucer_hits)
        )

    @partial(jax.jit, static_argnums=(0,))
    def _check_rejuvenator_interactions(self, state: BeamRiderState) -> BeamRiderState:
        """Handle rejuvenator interactions - OPTIMIZED with vectorized collision detection"""
        enemies = state.enemies
        ship = state.ship
        projectiles = state.projectiles
        torpedo_projectiles = state.torpedo_projectiles

        # Find active rejuvenators
        rejuvenator_mask = (enemies[:, 3] == 1) & (enemies[:, 5] == self.constants.ENEMY_TYPE_YELLOW_REJUVENATOR)
        rejuvenator_x = enemies[:, 0]
        rejuvenator_y = enemies[:, 1]

        # Check collection (landing on deck)
        collection_mask = (
                rejuvenator_mask &
                (rejuvenator_y >= ship.y - self.constants.ENEMY_HEIGHT) &
                (rejuvenator_x + self.constants.ENEMY_WIDTH > ship.x) &
                (rejuvenator_x < ship.x + self.constants.SHIP_WIDTH)
        )

        collected_count = jnp.sum(collection_mask)
        new_lives = state.lives + (collected_count * self.constants.YELLOW_REJUVENATOR_LIFE_BONUS)
        enemies = enemies.at[:, 3].set(jnp.where(collection_mask, 0, enemies[:, 3]))

        # OPTIMIZED: Vectorized projectile collision detection
        projectile_active = projectiles[:, 2] == 1

        # Create collision matrix for regular projectiles
        proj_x_expanded = projectiles[:, 0:1]
        proj_y_expanded = projectiles[:, 1:2]
        rejuv_x_expanded = rejuvenator_x[None, :]
        rejuv_y_expanded = rejuvenator_y[None, :]

        projectile_hits_matrix = (
                projectile_active[:, None] &
                rejuvenator_mask[None, :] &
                (proj_x_expanded + self.constants.PROJECTILE_WIDTH > rejuv_x_expanded) &
                (proj_x_expanded < rejuv_x_expanded + self.constants.ENEMY_WIDTH) &
                (proj_y_expanded + self.constants.PROJECTILE_HEIGHT > rejuv_y_expanded) &
                (proj_y_expanded < rejuv_y_expanded + self.constants.ENEMY_HEIGHT)
        )

        projectile_hit_any = jnp.any(projectile_hits_matrix, axis=1)
        rejuvenator_hit_by_projectile = jnp.any(projectile_hits_matrix, axis=0)

        # Update projectiles
        projectiles = projectiles.at[:, 2].set(
            jnp.where(projectile_hit_any, 0, projectiles[:, 2])
        )

        # OPTIMIZED: Vectorized torpedo collision detection
        torpedo_active = torpedo_projectiles[:, 2] == 1

        torp_x_expanded = torpedo_projectiles[:, 0:1]
        torp_y_expanded = torpedo_projectiles[:, 1:2]

        torpedo_hits_matrix = (
                torpedo_active[:, None] &
                rejuvenator_mask[None, :] &
                (torp_x_expanded + self.constants.TORPEDO_WIDTH > rejuv_x_expanded) &
                (torp_x_expanded < rejuv_x_expanded + self.constants.ENEMY_WIDTH) &
                (torp_y_expanded + self.constants.TORPEDO_HEIGHT > rejuv_y_expanded) &
                (torp_y_expanded < rejuv_y_expanded + self.constants.ENEMY_HEIGHT)
        )

        torpedo_hit_any = jnp.any(torpedo_hits_matrix, axis=1)
        rejuvenator_hit_by_torpedo = jnp.any(torpedo_hits_matrix, axis=0)

        torpedo_projectiles = torpedo_projectiles.at[:, 2].set(
            jnp.where(torpedo_hit_any, 0, torpedo_projectiles[:, 2])
        )

        # Combine hits and deactivate
        rejuvenator_hit_mask = rejuvenator_hit_by_projectile | rejuvenator_hit_by_torpedo
        enemies = enemies.at[:, 3].set(jnp.where(rejuvenator_hit_mask, 0, enemies[:, 3]))

        # Update state
        state = state.replace(
            enemies=enemies,
            lives=new_lives,
            projectiles=projectiles,
            torpedo_projectiles=torpedo_projectiles
        )

        # Spawn debris for hit rejuvenators
        state = self._spawn_rejuvenator_debris(state, rejuvenator_hit_mask, enemies)

        return state

    @partial(jax.jit, static_argnums=(0,))
    def _check_debris_collision(self, state: BeamRiderState) -> BeamRiderState:
        """Check for deadly collision with rejuvenator debris"""
        enemies = state.enemies
        ship = state.ship

        # Find active debris
        debris_mask = (enemies[:, 3] == 1) & (enemies[:, 5] == self.constants.ENEMY_TYPE_REJUVENATOR_DEBRIS)

        debris_x = enemies[:, 0]
        debris_y = enemies[:, 1]

        # Check collision with ship
        debris_collision = (
                debris_mask &
                (debris_x + self.constants.ENEMY_WIDTH > ship.x) &
                (debris_x < ship.x + self.constants.SHIP_WIDTH) &
                (debris_y + self.constants.ENEMY_HEIGHT > ship.y) &
                (debris_y < ship.y + self.constants.SHIP_HEIGHT)
        )

        # If any debris hits ship, lose a life
        hit_by_debris = jnp.any(debris_collision)
        new_lives = jnp.where(hit_by_debris, state.lives - 1, state.lives)

        # Deactivate debris that hit the ship
        enemies = enemies.at[:, 3].set(
            jnp.where(debris_collision, 0, enemies[:, 3])
        )

        return state.replace(enemies=enemies, lives=new_lives)

    @partial(jax.jit, static_argnums=(0,))
    def _spawn_rejuvenator_debris(self, state: BeamRiderState, rejuvenator_hit_mask: chex.Array,
                                  enemies: chex.Array) -> BeamRiderState:
        """Spawn explosive debris when rejuvenator is shot"""

        def spawn_debris_for_rejuvenator(i, state_enemies):
            state_inner, enemies_inner = state_enemies

            # Check if this rejuvenator was hit
            rejuvenator_hit = rejuvenator_hit_mask[i]

            # Get rejuvenator position and beam
            rejuv_x = enemies[i, 0]
            rejuv_y = enemies[i, 1]
            rejuv_beam = enemies[i, 2].astype(int)  # Get the beam the rejuvenator was on

            def spawn_single_debris(debris_idx, enemies_state):
                # Find first available slot
                first_inactive = jnp.argmax(enemies_state[:, 3] == 0)
                can_spawn = (enemies_state[first_inactive, 3] == 0) & rejuvenator_hit & (debris_idx < 1)

                # Create debris enemy - moves straight down, no horizontal component
                new_debris = jnp.array([
                    rejuv_x,  # x - start at rejuvenator position
                    rejuv_y,  # y - start at rejuvenator position
                    rejuv_beam,  # beam_position - inherit rejuvenator's beam (IMPORTANT: this enables curve following)
                    1,  # active
                    self.constants.REJUVENATOR_DEBRIS_SPEED,  # speed
                    self.constants.ENEMY_TYPE_REJUVENATOR_DEBRIS,  # type
                    0.0,  # direction_x - no horizontal movement needed (curve will handle this)
                    1.0,  # direction_y - move down
                    0,  # bounce_count
                    0,  # linger_timer
                    0,  # target_x
                    1,  # health
                    0,  # firing_timer
                    0,  # maneuver_timer
                    0,  # movement_pattern
                    0,  # white_saucer_firing_timer
                    0,  # jump_timer
                ])

                enemies_state = jnp.where(
                    can_spawn,
                    enemies_state.at[first_inactive].set(new_debris),
                    enemies_state
                )

                return enemies_state

            # Spawn single debris piece
            enemies_inner = jax.lax.fori_loop(0, 1, spawn_single_debris, enemies_inner)

            return (state_inner, enemies_inner)

        # Apply debris spawning for all enemies
        state, enemies = jax.lax.fori_loop(0, self.constants.MAX_ENEMIES, spawn_debris_for_rejuvenator,
                                           (state, enemies))

        return state.replace(enemies=enemies)
    @partial(jax.jit, static_argnums=(0,))
    def _check_sector_progression(self, state: BeamRiderState) -> BeamRiderState:
        """Check if sector is complete and advance to next sector - Updated with smooth 99-sector scaling"""

        # Check if we've killed enough WHITE SAUCERS
        white_saucers_complete = state.enemies_killed_this_sector >= self.constants.ENEMIES_PER_SECTOR

        # Check sentinel status BEFORE spawning
        sentinel_active_before = jnp.any(
            (state.enemies[:, 3] == 1) & (state.enemies[:, 5] == self.constants.ENEMY_TYPE_SENTINEL_SHIP)
        )

        # Spawn sentinel if: white saucers done AND sentinel not spawned yet AND sector requires sentinel
        should_spawn_sentinel = (
                white_saucers_complete &
                ~state.sentinel_spawned_this_sector &  # Haven't spawned one yet
                (state.current_sector >= self.constants.SENTINEL_SHIP_SPAWN_SECTOR)
        )

        # Spawn sentinel if needed (and despawn any lingering white saucers)
        state = jax.lax.cond(
            should_spawn_sentinel,
            lambda s: self._spawn_sentinel(
                self._despawn_white_saucers(s)
            ).replace(sentinel_spawned_this_sector=True),
            lambda s: s,
            state
        )

        # NOW check sentinel status AFTER potential spawning
        sentinel_active_after = jnp.any(
            (state.enemies[:, 3] == 1) & (state.enemies[:, 5] == self.constants.ENEMY_TYPE_SENTINEL_SHIP)
        )

        # Sector is complete when:
        # 1. White saucers are done AND
        # 2. Either (no sentinel needed) OR (sentinel spawned and now gone)
        sentinel_requirement_met = jnp.where(
            state.current_sector >= self.constants.SENTINEL_SHIP_SPAWN_SECTOR,
            state.sentinel_spawned_this_sector & ~sentinel_active_after,  # Sentinel was spawned and is now gone
            True  # No sentinel required for early sectors
        )

        # IMPORTANT: If we just spawned a sentinel, the sector should NOT be complete yet
        # The sentinel needs to be destroyed first
        just_spawned_sentinel = should_spawn_sentinel  # We just spawned one this frame

        # Override sector completion if we just spawned a sentinel
        sector_truly_complete = white_saucers_complete & sentinel_requirement_met & ~just_spawned_sentinel

        # Calculate new values when sector is truly complete
        new_sector = jnp.where(sector_truly_complete, state.current_sector + 1, state.current_sector)
        new_level = new_sector
        new_enemies_killed = jnp.where(sector_truly_complete, 0, state.enemies_killed_this_sector)
        new_torpedoes = jnp.where(sector_truly_complete, self.constants.TORPEDOES_PER_SECTOR, state.torpedoes_remaining)

        # Reset sentinel flag when advancing to new sector
        new_sentinel_spawned = jnp.where(sector_truly_complete, False, state.sentinel_spawned_this_sector)

        # Reset ship position to center beam when sector completes
        center_beam = self.constants.INITIAL_BEAM
        new_ship_x = jnp.where(
            sector_truly_complete,
            self.beam_positions[center_beam] - self.constants.SHIP_WIDTH // 2,
            state.ship.x
        )

        new_ship_beam = jnp.where(
            sector_truly_complete,
            center_beam,
            state.ship.beam_position
        )

        # Clear all projectiles when sector completes
        cleared_projectiles = jnp.where(
            sector_truly_complete,
            jnp.zeros_like(state.projectiles),
            state.projectiles
        )
        cleared_torpedo_projectiles = jnp.where(
            sector_truly_complete,
            jnp.zeros_like(state.torpedo_projectiles),
            state.torpedo_projectiles
        )
        cleared_sentinel_projectiles = jnp.where(
            sector_truly_complete,
            jnp.zeros_like(state.sentinel_projectiles),
            state.sentinel_projectiles
        )

        # Clear all enemies when sector completes
        cleared_enemies = jnp.where(
            sector_truly_complete,
            jnp.zeros_like(state.enemies),
            state.enemies
        )

        # Reset spawn timer when sector completes
        new_spawn_timer = jnp.where(sector_truly_complete, 0, state.enemy_spawn_timer)

        # UPDATED: Smooth difficulty scaling across 99 sectors
        # Calculate spawn interval scaling (90 frames to 12 frames over 99 sectors)
        # Uses exponential decay for smoother progression
        progress_ratio = jnp.minimum((new_sector - 1) / 98.0, 1.0)  # 0.0 to 1.0 over sectors 1-99, capped at 1.0

        # Exponential spawn rate increase (starts slow, accelerates later)
        spawn_decay_factor = jnp.exp(-3.0 * progress_ratio)  # Exponential curve
        calculated_spawn_interval = (
                self.constants.MIN_ENEMY_SPAWN_INTERVAL +
                (
                        self.constants.BASE_ENEMY_SPAWN_INTERVAL - self.constants.MIN_ENEMY_SPAWN_INTERVAL) * spawn_decay_factor
        )
        calculated_spawn_interval = jnp.round(calculated_spawn_interval).astype(jnp.int32)

        # Apply new spawn interval only when sector completes
        spawn_interval = jnp.where(
            sector_truly_complete,
            calculated_spawn_interval,
            state.enemy_spawn_interval
        )

        # Create updated ship struct
        ship = state.ship.replace(
            x=new_ship_x,
            beam_position=new_ship_beam,
            target_beam=new_ship_beam  # Add this line
        )

        return state.replace(
            ship=ship,
            projectiles=cleared_projectiles,
            torpedo_projectiles=cleared_torpedo_projectiles,
            sentinel_projectiles=cleared_sentinel_projectiles,
            sentinel_spawned_this_sector=new_sentinel_spawned,
            enemies=cleared_enemies,
            current_sector=new_sector,
            level=new_level,
            enemies_killed_this_sector=new_enemies_killed,
            torpedoes_remaining=new_torpedoes,
            enemy_spawn_timer=new_spawn_timer,
            enemy_spawn_interval=spawn_interval
        )

    @partial(jax.jit, static_argnums=(0,))
    def _despawn_white_saucers(self, state: BeamRiderState) -> BeamRiderState:
        """Immediately deactivate all remaining white saucers (e.g., when the sentinel appears)."""
        enemies = state.enemies
        white_mask = enemies[:, 5] == self.constants.ENEMY_TYPE_WHITE_SAUCER
        # Set 'active' (col 3) to 0 for all white saucers
        enemies = enemies.at[:, 3].set(jnp.where(white_mask, 0, enemies[:, 3]))
        return state.replace(enemies=enemies)


    @partial(jax.jit, static_argnums=(0,))
    def _check_game_over(self, state: BeamRiderState) -> BeamRiderState:
        """Check if game is over - Updated to include sector 99 limit"""
        # Original game over condition: no lives left
        lives_game_over = state.lives <= 0

        # New game over condition: reached sector 99 limit
        sector_limit_reached = state.current_sector > 99

        # Game is over if either condition is met
        game_over = lives_game_over | sector_limit_reached

        return state.replace(game_over=game_over)

    @partial(jax.jit, static_argnums=(0,))
    def _spawn_sentinel(self, state: BeamRiderState) -> BeamRiderState:
        """Spawn the sector sentinel ship - UPDATED: Random spawn side"""
        enemies = state.enemies

        # Find first inactive enemy slot
        active_mask = enemies[:, 3] == 0
        first_inactive = jnp.argmax(active_mask)
        can_spawn = active_mask[first_inactive]

        # Generate random spawn side
        rng_key, spawn_side_key = random.split(state.rng_key)
        spawn_from_left = random.randint(spawn_side_key, (), 0, 2) == 0  # 50% chance for left

        # Set spawn position and direction based on side
        sentinel_spawn_x = jnp.where(
            spawn_from_left,
            -self.constants.SENTINEL_SHIP_WIDTH,  # Start off-screen left
            self.constants.SCREEN_WIDTH  # Start off-screen right
        )

        sentinel_direction_x = jnp.where(
            spawn_from_left,
            1.0,  # Move right when spawning from left
            -1.0  # Move left when spawning from right
        )

        sentinel_spawn_y = self.constants.TOP_MARGIN + 10

        new_sentinel = jnp.array([
            sentinel_spawn_x,  # 0: x
            sentinel_spawn_y,  # 1: y
            0,  # 2: beam_position
            1,  # 3: active
            self.constants.SENTINEL_SHIP_SPEED,  # 4: speed
            self.constants.ENEMY_TYPE_SENTINEL_SHIP,  # 5: type
            sentinel_direction_x,  # 6: direction_x (random direction)
            0.0,  # 7: direction_y
            0,  # 8: bounce_count
            0,  # 9: linger_timer
            0,  # 10: target_x
            1,  # 11: health
            self.constants.SENTINEL_SHIP_FIRING_INTERVAL,  # 12: firing_timer
            0,  # 13: maneuver_timer (UNUSED - no more maneuvers)
            0,  # 14: movement_pattern
            0,  # 15: white_saucer_firing_timer
            0,  # 16: jump_timer
        ])

        enemies = jnp.where(
            can_spawn,
            enemies.at[first_inactive].set(new_sentinel),
            enemies
        )

        return state.replace(enemies=enemies, rng_key=rng_key)

class BeamRiderRenderer(JAXGameRenderer):
    """Unified renderer for BeamRider game with both JAX rendering and Pygame display"""

    def __init__(self, scale=3, enable_pygame=False):
        super().__init__()
        self.constants = BeamRiderConstants()
        self.screen_width = self.constants.SCREEN_WIDTH
        self.screen_height = self.constants.SCREEN_HEIGHT
        self.beam_positions = self.constants.get_beam_positions()

        # White saucer sprite
        self.white_saucer_sprite = jnp.array([
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0],
            [0, 1, 0, 1, 1, 0, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0],
        ], dtype=jnp.uint8)
        # Brown debris sprite
        self.brown_debris_sprite = jnp.array([
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 1, 1, 0, 1, 1, 0, 0],
            [1, 0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0, 1],
            [1, 0, 1, 1, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 0, 0],
        ], dtype=jnp.uint8)
        # Green blocker sprite
        self.green_blocker_sprite = jnp.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=jnp.uint8)
        # Green bounce craft sprite
        self.green_bounce_sprite = jnp.array([
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 0, 1, 1, 1, 1, 1, 0, 1],
            [1, 0, 1, 0, 0, 0, 1, 0, 1],
            [1, 0, 1, 1, 1, 1, 1, 0, 1],
            [0, 1, 0, 1, 1, 1, 0, 1, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
        ], dtype=jnp.uint8)
        self.blue_charger_sprite = jnp.array([
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 0, 0, 0, 1, 1, 0],
            [1, 1, 0, 0, 0, 0, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 1, 1, 1, 1, 0, 1],
            [1, 0, 0, 1, 1, 1, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1],
        ], dtype=jnp.uint8)
        self.orange_tracker_sprite = jnp.array([
            [0, 1, 0, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 0, 0, 0, 0, 0, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
        ], dtype=jnp.uint8)
        self.yellow_rejuv = jnp.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=jnp.uint8)
        self.debris_sprite = jnp.array([
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 1, 1, 0, 1, 1, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 1],
        ])

        # Yellow chirper sprite - closed mouth frame
        self.yellow_chirper_closed = jnp.array([
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0],
        ], dtype=jnp.uint8)

        # Yellow chirper sprite - open mouth frame
        self.yellow_chirper_open = jnp.array([
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0],
        ], dtype=jnp.uint8)

        # JAX rendering components
        self.ship_sprite_surface = self._create_ship_surface()
        self.small_ship_surface = self._create_small_ship_surface()

        # JIT-compile the render function
        self.render = jit(self._render_impl)

        # Pygame components (optional)
        self.enable_pygame = enable_pygame
        if enable_pygame:
            pygame.init()
            self.scale = scale
            self.pygame_screen_width = self.screen_width * scale
            self.pygame_screen_height = self.screen_height * scale
            self.pygame_screen = pygame.display.set_mode((self.pygame_screen_width, self.pygame_screen_height))
            pygame.display.set_caption("BeamRider - JAX Implementation")
            self.clock = pygame.time.Clock()
            self.env = BeamRiderEnv()

    @partial(jax.jit, static_argnums=(0,))
    def _get_chirper_animation_frame(self, frame_count: int) -> int:
        """Get current animation frame for yellow chirper (0=closed, 1=open)"""
        # Change animation every 20 frames (about 3 times per second at 60fps)
        animation_speed = 20
        return (frame_count // animation_speed) % 2

    @partial(jax.jit, static_argnums=(0,))
    def _draw_animated_chirper_sprite(self, screen: chex.Array, x: int, y: int,
                                      scale: float, color: chex.Array, frame_count: int) -> chex.Array:
        """Draw animated yellow chirper sprite with opening/closing mouth"""

        # Get current animation frame
        anim_frame = self._get_chirper_animation_frame(frame_count)

        # Select sprite based on animation frame
        sprite = jnp.where(
            anim_frame == 0,
            self.yellow_chirper_closed,
            self.yellow_chirper_open
        )

        sprite_height, sprite_width = sprite.shape

        # Calculate scaled dimensions
        scaled_width = jnp.maximum(1, (sprite_width * scale).astype(jnp.int32))
        scaled_height = jnp.maximum(1, (sprite_height * scale).astype(jnp.int32))

        # For very small scales, just draw a dot
        is_dot = scale < 0.25

        def draw_dot():
            y_indices = jnp.arange(self.constants.SCREEN_HEIGHT)
            x_indices = jnp.arange(self.constants.SCREEN_WIDTH)
            y_grid, x_grid = jnp.meshgrid(y_indices, x_indices, indexing='ij')

            dot_mask = (x_grid == x) & (y_grid == y) & (x >= 0) & (x < self.constants.SCREEN_WIDTH) & (y >= 0) & (
                        y < self.constants.SCREEN_HEIGHT)

            return jnp.where(
                dot_mask[..., None],
                color,
                screen
            ).astype(jnp.uint8)

        def draw_scaled_sprite():
            y_indices = jnp.arange(self.constants.SCREEN_HEIGHT)
            x_indices = jnp.arange(self.constants.SCREEN_WIDTH)
            y_grid, x_grid = jnp.meshgrid(y_indices, x_indices, indexing='ij')

            # Center the scaled sprite
            start_x = x - scaled_width // 2
            start_y = y - scaled_height // 2

            # Create mask for scaled sprite area
            sprite_mask = (
                    (x_grid >= start_x) &
                    (x_grid < start_x + scaled_width) &
                    (y_grid >= start_y) &
                    (y_grid < start_y + scaled_height) &
                    (start_x >= 0) & (start_x + scaled_width <= self.constants.SCREEN_WIDTH) &
                    (start_y >= 0) & (start_y + scaled_height <= self.constants.SCREEN_HEIGHT)
            )

            # Map screen coordinates to original sprite coordinates
            sprite_x_coords = ((x_grid - start_x) * sprite_width / scaled_width).astype(jnp.int32)
            sprite_y_coords = ((y_grid - start_y) * sprite_height / scaled_height).astype(jnp.int32)

            # Clamp coordinates to sprite bounds
            sprite_x_coords = jnp.clip(sprite_x_coords, 0, sprite_width - 1)
            sprite_y_coords = jnp.clip(sprite_y_coords, 0, sprite_height - 1)

            # Get sprite pixel values
            sprite_values = sprite[sprite_y_coords, sprite_x_coords]

            # Apply color where sprite has value 1 and mask is True
            draw_mask = sprite_mask & (sprite_values == 1)

            return jnp.where(
                draw_mask[..., None],
                color,
                screen
            ).astype(jnp.uint8)

        # Choose between dot or scaled sprite based on scale
        return jax.lax.cond(is_dot, draw_dot, draw_scaled_sprite)


    def _create_ship_surface(self):
        # Create the main ship sprite surface using a pixel array and color map.

        # Sprite design using pixel values:
        # 0 = transparent, 1 = yellow, 2 = purple
        ship_sprite = np.array([
            [0, 0, 0, 2, 2, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        # Map from pixel value to RGBA color
        colors = {
            0: (0, 0, 0, 0),  # transparent
            1: (255, 255, 0, 255),  # yellow
            2: (160, 32, 240, 255),  # purple
        }

        h, w = ship_sprite.shape

        # Create a Pygame surface with alpha channel
        surface = pygame.Surface((w, h), pygame.SRCALPHA)

        # Paint each pixel based on the sprite array
        for y in range(h):
            for x in range(w):
                surface.set_at((x, y), colors[ship_sprite[y, x]])

        # Scale the sprite up for visibility (6x enlargement)
        return pygame.transform.scale(surface, (w * 6, h * 6))

    def _create_small_ship_surface(self):
        """Creates a small version of the ship sprite for UI (lives display)"""
        small_sprite = pygame.transform.scale(self.ship_sprite_surface, (16, 10))
        return small_sprite

    @partial(jax.jit, static_argnums=(0,))
    def _render_impl(self, state: BeamRiderState) -> chex.Array:
        """Render the current game state to a screen buffer"""
        # Create screen buffer (RGB)
        screen = jnp.zeros((self.constants.SCREEN_HEIGHT, self.constants.SCREEN_WIDTH, 3), dtype=jnp.uint8)

        # Render 3D dotted tunnel grid
        screen = self._draw_3d_grid(screen, state.frame_count)

        # Render projectiles (lasers)
        screen = self._draw_projectiles(screen, state.projectiles)

        screen = self._draw_ship(screen, state.ship)

        # Render torpedo projectiles
        screen = self._draw_torpedo_projectiles(screen, state.torpedo_projectiles)

        # Render sentinel projectiles
        screen = self._draw_white_saucer_projectiles(screen, state.sentinel_projectiles)

        # Render enemies
        screen = self._draw_enemies(screen, state.enemies, state)

        # Render UI (score, lives, torpedoes, sector progress)
        screen = self._draw_ui(screen, state)

        return screen

    @partial(jax.jit, static_argnums=(0,))
    def _draw_3d_grid(self, screen: chex.Array, frame_count: int) -> chex.Array:
        """Optimized 3D grid drawing"""

        height = self.constants.SCREEN_HEIGHT
        width = self.constants.SCREEN_WIDTH
        line_color = jnp.array([0, 180, 200], dtype=jnp.uint8)  # Brighter cornflower blue

        top_margin = int(height * 0.12)
        bottom_margin = int(height * 0.14)
        grid_height = height - top_margin - bottom_margin

        # === Horizontal Lines - Batch computation ===
        num_hlines = 7
        phase = (frame_count * 0.006) % 1.0

        # Calculate all line positions at once
        line_indices = jnp.arange(num_hlines)
        t_values = (phase + line_indices / num_hlines) % 1.0
        y_positions = jnp.round((t_values ** 3.0) * grid_height).astype(int) + top_margin
        y_positions = jnp.clip(y_positions, 0, height - 1)

        # Draw all horizontal lines using vectorized operations
        def draw_hline(i, scr):
            y = y_positions[i]
            valid = (y >= 0) & (y < height)
            return jax.lax.cond(
                valid,
                lambda s: s.at[y, :].set(line_color),
                lambda s: s,
                scr
            )

        screen = jax.lax.fori_loop(0, num_hlines, draw_hline, screen)

        # === Vertical Dotted Lines - More dots, brighter ===
        beam_positions = self.beam_positions
        center_x = width / 2
        y0 = height - bottom_margin
        y1 = -height * 0.7

        # Increased dot count for better visibility
        num_dots_per_beam = 12  # Increased from 12 for denser dots
        t_top = jnp.clip((top_margin - y0) / (y1 - y0), 0.0, 1.0)

        def draw_beam(beam_idx, scr):
            x0 = beam_positions[beam_idx]
            x1 = center_x + (x0 - center_x) * 0.05

            # Calculate dot positions
            dot_ts = jnp.linspace(0, t_top, num_dots_per_beam)

            def draw_dot(dot_idx, scr_inner):
                t = dot_ts[dot_idx]
                y = jnp.round(y0 + (y1 - y0) * t).astype(int)
                x = jnp.round(x0 + (x1 - x0) * t).astype(int)

                valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)

                return jax.lax.cond(
                    valid,
                    lambda s: s.at[y, x].set(line_color),
                    lambda s: s,
                    scr_inner
                )

            return jax.lax.fori_loop(0, num_dots_per_beam, draw_dot, scr)

        screen = jax.lax.fori_loop(0, self.constants.NUM_BEAMS, draw_beam, screen)

        # === Edge Dots (4 on each side in upper half)
        spacing = self.beam_positions[1] - self.beam_positions[0]
        edge_offset = 1.5 * spacing  # was 1.0 * spacing; push a bit more outside
        left_x0 = self.beam_positions[0] - edge_offset
        right_x0 = self.beam_positions[-1] + edge_offset

        left_x1 = center_x + (left_x0 - center_x) * 0.05
        right_x1 = center_x + (right_x0 - center_x) * 0.05

        upper_half_start = top_margin + 25  # Start below top margin
        upper_half_end = height // 2 + 10  # End around middle
        edge_dot_positions = jnp.linspace(upper_half_start, upper_half_end, 4).astype(int)

        def draw_edge_dots(i, scr):
            y = edge_dot_positions[i]
            # compute t along the same perspective line used for the beam dots
            t = jnp.clip((y - y0) / (y1 - y0), 0.0, 1.0)
            lx = jnp.clip(jnp.round(left_x0 + (left_x1 - left_x0) * t).astype(int), 0, width - 1)
            rx = jnp.clip(jnp.round(right_x0 + (right_x1 - right_x0) * t).astype(int), 0, width - 1)
            scr = scr.at[y, lx].set(line_color)
            scr = scr.at[y, rx].set(line_color)
            return scr

        screen = jax.lax.fori_loop(0, 4, draw_edge_dots, screen)

        return screen



    @partial(jax.jit, static_argnums=(0,))
    def _draw_sentinel_sprite(self, screen: chex.Array, x: int, y: int, scale: float) -> chex.Array:
        """Draw sentinel ship sprite with multiple colors - simplified for scale=1.0"""

        # Define the 16x7 sentinel sprite
        sentinel_sprite = jnp.array([
            [0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1],
            [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0],
            [2, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 2],
        ], dtype=jnp.uint8)

        # Define colors
        main_red = jnp.array([255, 69, 0], dtype=jnp.uint8)
        orange_highlight = jnp.array([255, 165, 0], dtype=jnp.uint8)
        dark_red = jnp.array([160, 32, 240], dtype=jnp.uint8)

        sprite_h, sprite_w = sentinel_sprite.shape
        start_x = (x - sprite_w // 2).astype(int)
        start_y = (y - sprite_h // 2).astype(int)

        # Create coordinate grids (simplified since no scaling)
        y_indices = jnp.arange(self.constants.SCREEN_HEIGHT)
        x_indices = jnp.arange(self.constants.SCREEN_WIDTH)
        y_grid, x_grid = jnp.meshgrid(y_indices, x_indices, indexing='ij')

        # JAX-compatible bounds checking
        in_bounds = (
                (start_x >= 0) & (start_x + sprite_w <= self.constants.SCREEN_WIDTH) &
                (start_y >= 0) & (start_y + sprite_h <= self.constants.SCREEN_HEIGHT)
        )

        # Create sprite area mask
        sprite_area_mask = (
                (x_grid >= start_x) &
                (x_grid < start_x + sprite_w) &
                (y_grid >= start_y) &
                (y_grid < start_y + sprite_h) &
                in_bounds
        )

        # Map screen coordinates to sprite coordinates (no scaling)
        sprite_x_coords = (x_grid - start_x).astype(int)
        sprite_y_coords = (y_grid - start_y).astype(int)

        # Clamp to sprite bounds
        sprite_x_coords = jnp.clip(sprite_x_coords, 0, sprite_w - 1)
        sprite_y_coords = jnp.clip(sprite_y_coords, 0, sprite_h - 1)

        # Get sprite pixel values
        sprite_values = sentinel_sprite[sprite_y_coords, sprite_x_coords]

        # Create color masks and apply colors
        main_hull_mask = sprite_area_mask & (sprite_values == 1)
        highlight_mask = sprite_area_mask & (sprite_values == 2)
        detail_mask = sprite_area_mask & (sprite_values == 3)

        screen = jnp.where(main_hull_mask[..., None], main_red, screen)
        screen = jnp.where(highlight_mask[..., None], orange_highlight, screen)
        screen = jnp.where(detail_mask[..., None], dark_red, screen)

        return screen.astype(jnp.uint8)

    @partial(jax.jit, static_argnums=(0,))
    def _draw_torpedo_projectiles(self, screen: chex.Array, torpedo_projectiles: chex.Array) -> chex.Array:
        """Draw all active torpedo projectiles"""

        H, W = self.constants.SCREEN_HEIGHT, self.constants.SCREEN_WIDTH
        y_idx = jnp.arange(H)
        x_idx = jnp.arange(W)
        y_grid, x_grid = jnp.meshgrid(y_idx, x_idx, indexing='ij')

        t_w = self.constants.TORPEDO_WIDTH
        t_h = self.constants.TORPEDO_HEIGHT
        torpedo_color = jnp.array(self.constants.WHITE, dtype=jnp.uint8)

        # Returns an (H, W) bool mask for a single projectile
        def single_torpedo_mask(proj):
            x = proj[0].astype(int)
            y = proj[1].astype(int)
            active = (proj[2] == 1)

            # Match original semantics: only draw if top-left is within the screen
            valid = active & (x >= 0) & (x < W) & (y >= 0) & (y < H)

            rect = (
                    (x_grid >= x) & (x_grid < x + t_w) &
                    (y_grid >= y) & (y_grid < y + t_h)
            )
            return rect & valid

        # (N, H, W) -> (H, W) by logical OR across torpedoes
        masks = jax.vmap(single_torpedo_mask)(torpedo_projectiles)
        any_mask = jnp.any(masks, axis=0)

        # Single write to the screen buffer
        screen = jnp.where(any_mask[..., None], torpedo_color, screen).astype(jnp.uint8)
        return screen
    @partial(jax.jit, static_argnums=(0,))
    def _draw_white_saucer_projectiles(self, screen: chex.Array, sentinel_projectiles: chex.Array) -> chex.Array:
        """Draw all active sentinel projectiles"""
        H, W = self.constants.SCREEN_HEIGHT, self.constants.SCREEN_WIDTH
        y_idx = jnp.arange(H)
        x_idx = jnp.arange(W)
        y_grid, x_grid = jnp.meshgrid(y_idx, x_idx, indexing='ij')

        p_w = self.constants.PROJECTILE_WIDTH
        p_h = self.constants.PROJECTILE_HEIGHT
        projectile_color = jnp.array(self.constants.RED, dtype=jnp.uint8)

        # Per-projectile (H, W) mask
        def single_mask(proj):
            x0 = proj[0].astype(jnp.int32)
            y0 = proj[1].astype(jnp.int32)
            active = (proj[2] == 1)

            valid = active & (x0 >= 0) & (x0 < W) & (y0 >= 0) & (y0 < H)
            rect = (
                    (x_grid >= x0) & (x_grid < x0 + p_w) &
                    (y_grid >= y0) & (y_grid < y0 + p_h)
            )
            return rect & valid

        masks = jax.vmap(single_mask)(sentinel_projectiles)  # (N, H, W)
        any_mask = jnp.any(masks, axis=0)  # (H, W)

        return jnp.where(any_mask[..., None], projectile_color, screen).astype(jnp.uint8)

    @partial(jax.jit, static_argnums=(0,))
    def _draw_ship(self, screen: chex.Array, ship: Ship) -> chex.Array:
        """Draw the player ship with the actual sprite design"""
        x, y = ship.x.astype(int), ship.y.astype(int)

        # Colors
        yellow = jnp.array([255, 255, 0], dtype=jnp.uint8)
        purple = jnp.array([160, 32, 240], dtype=jnp.uint8)

        # Create coordinate grids
        y_indices = jnp.arange(self.constants.SCREEN_HEIGHT)
        x_indices = jnp.arange(self.constants.SCREEN_WIDTH)
        y_grid, x_grid = jnp.meshgrid(y_indices, x_indices, indexing='ij')

        # Scale factor
        scale = 2

        # Define ship shape regions (scaled)
        # Purple tip (top rows)
        purple_mask = (
            # Row 0: Purple tip center
                ((y_grid >= y) & (y_grid < y + scale) &
                 (x_grid >= x + 3 * scale) & (x_grid < x + 5 * scale)) |
                # Additional purple pixels can be added here
                False  # Placeholder for OR operations
        )

        # Yellow body
        yellow_mask = (
            # Row 1: Upper body
                ((y_grid >= y + scale) & (y_grid < y + 2 * scale) &
                 (x_grid >= x + 2 * scale) & (x_grid < x + 6 * scale)) |
                # Row 2: Middle body
                ((y_grid >= y + 2 * scale) & (y_grid < y + 3 * scale) &
                 (x_grid >= x + scale) & (x_grid < x + 7 * scale)) |
                # Row 3: Full width
                ((y_grid >= y + 3 * scale) & (y_grid < y + 4 * scale) &
                 (x_grid >= x) & (x_grid < x + 8 * scale)) |
                # Row 4: Lower body with gap
                ((y_grid >= y + 4 * scale) & (y_grid < y + 5 * scale) &
                 ((x_grid >= x) & (x_grid < x + 3 * scale) |
                  (x_grid >= x + 5 * scale) & (x_grid < x + 8 * scale))) |
                # Row 5: Bottom
                ((y_grid >= y + 5 * scale) & (y_grid < y + 6 * scale) &
                 ((x_grid >= x) & (x_grid < x + 2 * scale) |
                  (x_grid >= x + 6 * scale) & (x_grid < x + 8 * scale)))
        )

        # Apply colors where masks are True
        screen = jnp.where(
            purple_mask[..., None],
            purple,
            screen
        )

        screen = jnp.where(
            yellow_mask[..., None],
            yellow,
            screen
        )

        return screen

    @partial(jax.jit, static_argnums=(0,))
    def _draw_projectiles(self, screen: chex.Array, projectiles: chex.Array) -> chex.Array:
        """Draw projectiles - OPTIMIZED with vmap"""

        y_indices = jnp.arange(self.constants.SCREEN_HEIGHT)
        x_indices = jnp.arange(self.constants.SCREEN_WIDTH)
        y_grid, x_grid = jnp.meshgrid(y_indices, x_indices, indexing='ij')

        def create_projectile_mask(x, y, active):
            return (
                    (x_grid >= x) &
                    (x_grid < x + self.constants.PROJECTILE_WIDTH) &
                    (y_grid >= y) &
                    (y_grid < y + self.constants.PROJECTILE_HEIGHT) &
                    (x >= 0) & (x < self.constants.SCREEN_WIDTH) &
                    (y >= 0) & (y < self.constants.SCREEN_HEIGHT) &
                    active
            )

        # Vectorize over all projectiles
        vmapped_mask = jax.vmap(create_projectile_mask)
        all_masks = vmapped_mask(
            projectiles[:, 0].astype(int),
            projectiles[:, 1].astype(int),
            projectiles[:, 2] == 1
        )

        # Combine masks
        combined_mask = jnp.any(all_masks, axis=0)

        # Apply color
        projectile_color = jnp.array(self.constants.YELLOW, dtype=jnp.uint8)
        return jnp.where(combined_mask[..., None], projectile_color, screen).astype(jnp.uint8)

    @partial(jax.jit, static_argnums=(0,))
    def _get_enemy_scale(self, enemy_y: float) -> float:
        """Calculate enemy scale based on Y position with grid-based boundaries"""
        # Calculate grid-based scaling positions
        height = self.constants.SCREEN_HEIGHT
        top_margin = int(height * 0.12)
        bottom_margin = int(height * 0.14)
        grid_height = height - top_margin - bottom_margin

        # Calculate row positions (7 horizontal lines total)
        num_grid_rows = 7
        row_spacing = grid_height / (num_grid_rows + 1)

        # Define scaling boundaries based on grid rows
        horizon_y = top_margin  # Top of screen
        third_row_y = top_margin + (3 * row_spacing)  # 3rd row from top
        stop_scaling_y = height - bottom_margin - (3 * row_spacing)  # 3rd row from bottom

        # Scale parameters
        min_scale = 0.1  # Tiny dots at horizon
        mid_scale = 0.5  # Half size at 3rd row
        max_scale = 1.0  # Full size at 3rd last row

        # Three-phase scaling
        scale_factor = jnp.where(
            enemy_y <= horizon_y,
            min_scale,  # Stay tiny at horizon
            jnp.where(
                enemy_y <= third_row_y,
                # Phase 1: Scale from 0.1 to 0.5 (horizon to 3rd row)
                min_scale + (mid_scale - min_scale) * jnp.clip((enemy_y - horizon_y) / (third_row_y - horizon_y), 0.0,
                                                               1.0),
                jnp.where(
                    enemy_y <= stop_scaling_y,
                    # Phase 2: Scale from 0.5 to 1.5 (3rd row to 3rd last row)
                    mid_scale + (max_scale - mid_scale) * jnp.clip(
                        (enemy_y - third_row_y) / (stop_scaling_y - third_row_y), 0.0, 1.0),
                    # Phase 3: Stop growing (stay at 1.5)
                    max_scale
                )
            )
        )

        return scale_factor

    @partial(jax.jit, static_argnums=(0,))
    def _draw_scaled_enemy_sprite(self, screen: chex.Array, x: int, y: int, sprite: chex.Array,
                                  scale: float, color: chex.Array) -> chex.Array:
        """Simplified sprite rendering using full screen masking"""
        sprite_height, sprite_width = sprite.shape
        scaled_width = jnp.maximum(1, (sprite_width * scale).astype(jnp.int32))
        scaled_height = jnp.maximum(1, (sprite_height * scale).astype(jnp.int32))

        is_dot = scale < 0.3

        # For very small sprites, just draw a dot
        def draw_dot():
            valid = (x >= 0) & (x < self.constants.SCREEN_WIDTH) & \
                    (y >= 0) & (y < self.constants.SCREEN_HEIGHT)
            return jax.lax.cond(
                valid,
                lambda s: s.at[y, x].set(color),
                lambda s: s,
                screen
            )

        def draw_scaled_sprite():
            # Create coordinate grids for entire screen (fixed size)
            y_indices = jnp.arange(self.constants.SCREEN_HEIGHT)
            x_indices = jnp.arange(self.constants.SCREEN_WIDTH)
            y_grid, x_grid = jnp.meshgrid(y_indices, x_indices, indexing='ij')

            # Calculate sprite boundaries
            start_x = x - scaled_width // 2
            start_y = y - scaled_height // 2

            # Create mask for sprite area
            sprite_mask = (
                    (x_grid >= start_x) &
                    (x_grid < start_x + scaled_width) &
                    (y_grid >= start_y) &
                    (y_grid < start_y + scaled_height) &
                    (start_x >= 0) & (start_x + scaled_width <= self.constants.SCREEN_WIDTH) &
                    (start_y >= 0) & (start_y + scaled_height <= self.constants.SCREEN_HEIGHT)
            )

            # Map screen coordinates to sprite coordinates
            sprite_x_coords = ((x_grid - start_x) * sprite_width / scaled_width).astype(jnp.int32)
            sprite_y_coords = ((y_grid - start_y) * sprite_height / scaled_height).astype(jnp.int32)

            # Clip to sprite bounds
            sprite_x_coords = jnp.clip(sprite_x_coords, 0, sprite_width - 1)
            sprite_y_coords = jnp.clip(sprite_y_coords, 0, sprite_height - 1)

            # Get sprite pixel values
            sprite_values = sprite[sprite_y_coords, sprite_x_coords]

            # Apply color where sprite has value 1 and mask is True
            draw_mask = sprite_mask & (sprite_values == 1)

            return jnp.where(
                draw_mask[..., None],
                color,
                screen
            ).astype(jnp.uint8)

        return jax.lax.cond(is_dot, draw_dot, draw_scaled_sprite)
    @partial(jax.jit, static_argnums=(0,))
    def _draw_enemies(self, screen: chex.Array, enemies: chex.Array, state: BeamRiderState) -> chex.Array:
        """Optimized enemy drawing - removes the nested loop bottleneck"""

        # Pre-calculate coordinate grids once
        y_indices = jnp.arange(self.constants.SCREEN_HEIGHT)
        x_indices = jnp.arange(self.constants.SCREEN_WIDTH)
        y_grid, x_grid = jnp.meshgrid(y_indices, x_indices, indexing='ij')

        # Vectorized calculation of enemy properties
        def calculate_enemy_properties(enemy):
            x, y = enemy[0], enemy[1]
            active = enemy[3] == 1
            enemy_type = enemy[5].astype(int)

            # Check enemy type categories
            is_sentinel = enemy_type == self.constants.ENEMY_TYPE_SENTINEL_SHIP
            is_side_spawner = (
                    (enemy_type == self.constants.ENEMY_TYPE_YELLOW_CHIRPER) |
                    (enemy_type == self.constants.ENEMY_TYPE_GREEN_BLOCKER) |
                    (enemy_type == self.constants.ENEMY_TYPE_GREEN_BOUNCE) |
                    (enemy_type == self.constants.ENEMY_TYPE_ORANGE_TRACKER)
            )
            no_scaling = is_sentinel | is_side_spawner

            # Get base dimensions
            base_width = jnp.where(
                is_sentinel,
                self.constants.SENTINEL_SHIP_WIDTH,
                self.constants.ENEMY_WIDTH
            )
            base_height = jnp.where(
                is_sentinel,
                self.constants.SENTINEL_SHIP_HEIGHT,
                self.constants.ENEMY_HEIGHT
            )

            # Calculate scale
            scale_factor = jnp.where(no_scaling, jnp.array(1.0), self._get_enemy_scale(y))

            # Calculate scaled dimensions
            scaled_width = jnp.where(
                no_scaling,
                base_width.astype(int),
                jnp.maximum(1, (base_width * scale_factor).astype(int))
            )
            scaled_height = jnp.where(
                no_scaling,
                base_height.astype(int),
                jnp.maximum(1, (base_height * scale_factor).astype(int))
            )

            # Calculate centering offsets
            x_offset = ((base_width - scaled_width) / 2).astype(int)
            y_offset = ((base_height - scaled_height) / 2).astype(int)

            draw_x = (x + x_offset).astype(int)
            draw_y = (y + y_offset).astype(int)

            # Check visibility
            partially_visible = (
                    (draw_x < self.constants.SCREEN_WIDTH) &
                    (draw_x + scaled_width > 0) &
                    (draw_y < self.constants.SCREEN_HEIGHT) &
                    (draw_y + scaled_height > 0)
            )

            # Determine rendering type
            is_dot = (~no_scaling) & (scale_factor < 0.25)

            # Determine which sprite to use
            use_sprite = active & partially_visible & (
                    ((enemy_type == self.constants.ENEMY_TYPE_WHITE_SAUCER) & (scale_factor >= 0.4)) |
                    ((enemy_type == self.constants.ENEMY_TYPE_BROWN_DEBRIS) & (scale_factor >= 0.4)) |
                    (enemy_type == self.constants.ENEMY_TYPE_GREEN_BLOCKER) |
                    (enemy_type == self.constants.ENEMY_TYPE_GREEN_BOUNCE) |
                    (enemy_type == self.constants.ENEMY_TYPE_BLUE_CHARGER) |
                    (enemy_type == self.constants.ENEMY_TYPE_ORANGE_TRACKER) |
                    (enemy_type == self.constants.ENEMY_TYPE_YELLOW_REJUVENATOR) |
                    (enemy_type == self.constants.ENEMY_TYPE_SENTINEL_SHIP) |
                    (enemy_type == self.constants.ENEMY_TYPE_REJUVENATOR_DEBRIS) |
                    (enemy_type == self.constants.ENEMY_TYPE_YELLOW_CHIRPER)
            )

            return {
                'x': x.astype(int),
                'y': y.astype(int),
                'draw_x': draw_x,
                'draw_y': draw_y,
                'active': active,
                'enemy_type': enemy_type,
                'scale_factor': scale_factor,
                'scaled_width': scaled_width,
                'scaled_height': scaled_height,
                'partially_visible': partially_visible,
                'is_dot': is_dot,
                'use_sprite': use_sprite
            }

        # Use vmap to calculate all enemy properties at once
        enemy_props = jax.vmap(calculate_enemy_properties)(enemies)

        # Process sprites using a loop
        def draw_single_enemy(i, screen):
            props = jax.tree.map(lambda x: x[i], enemy_props)

            # WHITE_SAUCER - use optimized sprite drawing
            screen = jax.lax.cond(
                props['use_sprite'] & (props['enemy_type'] == self.constants.ENEMY_TYPE_WHITE_SAUCER),
                lambda s: self._draw_scaled_enemy_sprite(
                    s, props['x'], props['y'],
                    self.white_saucer_sprite,
                    props['scale_factor'],
                    jnp.array(self.constants.WHITE, dtype=jnp.uint8)
                ),
                lambda s: s,
                screen
            )

            # BROWN_DEBRIS
            screen = jax.lax.cond(
                props['use_sprite'] & (props['enemy_type'] == self.constants.ENEMY_TYPE_BROWN_DEBRIS),
                lambda s: self._draw_scaled_enemy_sprite(
                    s, props['x'], props['y'],
                    self.brown_debris_sprite,
                    props['scale_factor'],
                    jnp.array(self.constants.BROWN_DEBRIS_COLOR, dtype=jnp.uint8)
                ),
                lambda s: s,
                screen
            )

            # GREEN_BLOCKER
            screen = jax.lax.cond(
                props['use_sprite'] & (props['enemy_type'] == self.constants.ENEMY_TYPE_GREEN_BLOCKER),
                lambda s: self._draw_scaled_enemy_sprite(
                    s, props['x'], props['y'],
                    self.green_blocker_sprite,
                    props['scale_factor'],
                    jnp.array(self.constants.GREEN_BLOCKER_COLOR, dtype=jnp.uint8)
                ),
                lambda s: s,
                screen
            )

            # GREEN_BOUNCE
            screen = jax.lax.cond(
                props['use_sprite'] & (props['enemy_type'] == self.constants.ENEMY_TYPE_GREEN_BOUNCE),
                lambda s: self._draw_scaled_enemy_sprite(
                    s, props['x'], props['y'],
                    self.green_bounce_sprite,
                    props['scale_factor'],
                    jnp.array(self.constants.GREEN_BOUNCE_COLOR, dtype=jnp.uint8)
                ),
                lambda s: s,
                screen
            )

            # BLUE_CHARGER
            screen = jax.lax.cond(
                props['use_sprite'] & (props['enemy_type'] == self.constants.ENEMY_TYPE_BLUE_CHARGER),
                lambda s: self._draw_scaled_enemy_sprite(
                    s, props['x'], props['y'],
                    self.blue_charger_sprite,
                    props['scale_factor'],
                    jnp.array(self.constants.BLUE_CHARGER_COLOR, dtype=jnp.uint8)
                ),
                lambda s: s,
                screen
            )

            # ORANGE_TRACKER
            screen = jax.lax.cond(
                props['use_sprite'] & (props['enemy_type'] == self.constants.ENEMY_TYPE_ORANGE_TRACKER),
                lambda s: self._draw_scaled_enemy_sprite(
                    s, props['x'], props['y'],
                    self.orange_tracker_sprite,
                    props['scale_factor'],
                    jnp.array(self.constants.ORANGE_TRACKER_COLOR, dtype=jnp.uint8)
                ),
                lambda s: s,
                screen
            )

            # YELLOW_REJUVENATOR
            screen = jax.lax.cond(
                props['use_sprite'] & (props['enemy_type'] == self.constants.ENEMY_TYPE_YELLOW_REJUVENATOR),
                lambda s: self._draw_scaled_enemy_sprite(
                    s, props['x'], props['y'],
                    self.yellow_rejuv,
                    props['scale_factor'],
                    jnp.array(self.constants.YELLOW_REJUVENATOR_COLOR, dtype=jnp.uint8)
                ),
                lambda s: s,
                screen
            )

            # SENTINEL_SHIP
            screen = jax.lax.cond(
                props['use_sprite'] & (props['enemy_type'] == self.constants.ENEMY_TYPE_SENTINEL_SHIP),
                lambda s: self._draw_sentinel_sprite(s, props['x'], props['y'], props['scale_factor']),
                lambda s: s,
                screen
            )

            # REJUVENATOR_DEBRIS
            screen = jax.lax.cond(
                props['use_sprite'] & (props['enemy_type'] == self.constants.ENEMY_TYPE_REJUVENATOR_DEBRIS),
                lambda s: self._draw_scaled_enemy_sprite(
                    s, props['x'], props['y'],
                    self.debris_sprite,
                    props['scale_factor'],
                    jnp.array(self.constants.REJUVENATOR_DEBRIS_COLOR, dtype=jnp.uint8)
                ),
                lambda s: s,
                screen
            )

            # YELLOW_CHIRPER
            screen = jax.lax.cond(
                props['use_sprite'] & (props['enemy_type'] == self.constants.ENEMY_TYPE_YELLOW_CHIRPER),
                lambda s: self._draw_animated_chirper_sprite(
                    s, props['x'], props['y'],
                    props['scale_factor'],
                    jnp.array(self.constants.YELLOW_CHIRPER_COLOR, dtype=jnp.uint8),
                    state.frame_count
                ),
                lambda s: s,
                screen
            )

            # Draw non-sprite enemies (rectangles or dots)
            dot_mask = (
                    props['is_dot'] &
                    props['active'] &
                    props['partially_visible'] &
                    ~props['use_sprite'] &
                    (x_grid == jnp.clip(props['draw_x'] + props['scaled_width'] // 2, 0,
                                        self.constants.SCREEN_WIDTH - 1)) &
                    (y_grid == jnp.clip(props['draw_y'] + props['scaled_height'] // 2, 0,
                                        self.constants.SCREEN_HEIGHT - 1))
            )

            rect_mask = (
                    ~props['is_dot'] &
                    ~props['use_sprite'] &
                    props['active'] &
                    props['partially_visible'] &
                    (x_grid >= props['draw_x']) &
                    (x_grid < props['draw_x'] + props['scaled_width']) &
                    (y_grid >= props['draw_y']) &
                    (y_grid < props['draw_y'] + props['scaled_height']) &
                    (x_grid >= 0) & (x_grid < self.constants.SCREEN_WIDTH) &
                    (y_grid >= 0) & (y_grid < self.constants.SCREEN_HEIGHT)
            )

            enemy_mask = dot_mask | rect_mask

            # Get color for non-sprite enemies
            enemy_color = jnp.select(
                [
                    props['enemy_type'] == self.constants.ENEMY_TYPE_WHITE_SAUCER,
                    props['enemy_type'] == self.constants.ENEMY_TYPE_BROWN_DEBRIS,
                    props['enemy_type'] == self.constants.ENEMY_TYPE_YELLOW_CHIRPER,
                    props['enemy_type'] == self.constants.ENEMY_TYPE_BLUE_CHARGER,
                    props['enemy_type'] == self.constants.ENEMY_TYPE_GREEN_BOUNCE,
                    props['enemy_type'] == self.constants.ENEMY_TYPE_ORANGE_TRACKER,
                    props['enemy_type'] == self.constants.ENEMY_TYPE_GREEN_BLOCKER,
                    props['enemy_type'] == self.constants.ENEMY_TYPE_SENTINEL_SHIP,
                    props['enemy_type'] == self.constants.ENEMY_TYPE_YELLOW_REJUVENATOR,
                    props['enemy_type'] == self.constants.ENEMY_TYPE_REJUVENATOR_DEBRIS,
                ],
                [
                    jnp.array(self.constants.WHITE, dtype=jnp.uint8),
                    jnp.array(self.constants.BROWN_DEBRIS_COLOR, dtype=jnp.uint8),
                    jnp.array(self.constants.YELLOW_CHIRPER_COLOR, dtype=jnp.uint8),
                    jnp.array(self.constants.BLUE_CHARGER_COLOR, dtype=jnp.uint8),
                    jnp.array(self.constants.GREEN_BOUNCE_COLOR, dtype=jnp.uint8),
                    jnp.array(self.constants.ORANGE_TRACKER_COLOR, dtype=jnp.uint8),
                    jnp.array(self.constants.GREEN_BLOCKER_COLOR, dtype=jnp.uint8),
                    jnp.array(self.constants.SENTINEL_SHIP_COLOR, dtype=jnp.uint8),
                    jnp.array(self.constants.YELLOW_REJUVENATOR_COLOR, dtype=jnp.uint8),
                    jnp.array(self.constants.REJUVENATOR_DEBRIS_COLOR, dtype=jnp.uint8),
                ],
                default=jnp.array(self.constants.WHITE, dtype=jnp.uint8)
            )

            # Apply color for non-sprite enemies
            screen = jnp.where(enemy_mask[..., None], enemy_color, screen)

            return screen.astype(jnp.uint8)

        # Process all enemies with the loop
        screen = jax.lax.fori_loop(0, self.constants.MAX_ENEMIES, draw_single_enemy, screen)

        return screen

    @partial(jax.jit, static_argnums=(0,))
    def _draw_ui(self, screen: chex.Array, state) -> chex.Array:
        """
        HUD with per-component scales and margins:
          - enemies-left (middle-left, green)
          - SCORE / SECTOR (centered)
          - torpedo boxes (top-right)
          - lives as yellow rejuvenator sprites (bottom-left)
        """

        # =========================
        # ===== CONFIG / SIZES ====
        # =========================
        SCALE_SCORE = 1
        SCALE_ENEMIES = 2
        SCALE_TORPS = 1
        SCALE_LIVES = 1

        spacing = 1

        # Enemies (middle-left) - GREEN COLOR, MIDDLE POSITION
        MARGIN_ENEMIES_X = 10
        MARGIN_ENEMIES_Y = 10  # Middle-left position

        # Score & Sector (center top)
        MARGIN_SCORE_Y = 4
        LINE_GAP_SCORE = 2
        # Torpedoes (top-right)
        MARGIN_TORPS_X = 0
        MARGIN_TORPS_Y = 8
        # Lives (bottom-left)
        MARGIN_LIVES_X = 5
        MARGIN_LIVES_Y = 1

        H = self.constants.SCREEN_HEIGHT
        W = self.constants.SCREEN_WIDTH

        # Colors - GREEN for enemies left
        GREEN = (0, 255, 0)  # GREEN for enemies counter
        GOLD = (255, 220, 100)
        PURP = (160, 32, 240)
        YELL_RGB = (255, 255, 0)

        # =========================
        # ===== BITMAP FONTS  =====
        # =========================
        DIGITS = jnp.stack([
            jnp.array([[1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=jnp.uint8),  # 0
            jnp.array([[0, 1, 0], [1, 1, 0], [0, 1, 0], [0, 1, 0], [1, 1, 1]], dtype=jnp.uint8),  # 1
            jnp.array([[1, 1, 1], [0, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1]], dtype=jnp.uint8),  # 2
            jnp.array([[1, 1, 1], [0, 0, 1], [0, 1, 1], [0, 0, 1], [1, 1, 1]], dtype=jnp.uint8),  # 3
            jnp.array([[1, 0, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1]], dtype=jnp.uint8),  # 4
            jnp.array([[1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 1], [1, 1, 1]], dtype=jnp.uint8),  # 5
            jnp.array([[1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=jnp.uint8),  # 6
            jnp.array([[1, 1, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=jnp.uint8),  # 7
            jnp.array([[1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=jnp.uint8),  # 8
            jnp.array([[1, 1, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1]], dtype=jnp.uint8),  # 9
        ], axis=0)

        FONT = {
            'S': jnp.array([[1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 1], [1, 1, 1]], dtype=jnp.uint8),
            'C': jnp.array([[1, 1, 1], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=jnp.uint8),
            'O': jnp.array([[1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=jnp.uint8),
            'R': jnp.array([[1, 1, 0], [1, 0, 1], [1, 1, 0], [1, 0, 1], [1, 0, 1]], dtype=jnp.uint8),
            'E': jnp.array([[1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 0, 0], [1, 1, 1]], dtype=jnp.uint8),
            'T': jnp.array([[1, 1, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=jnp.uint8),
        }

        # =========================
        # ======= HELPERS =========
        # =========================
        def draw_rect(scr, x0, y0, w, h, color_rgb):
            x0 = jnp.clip(x0, 0, W - 1)
            y0 = jnp.clip(y0, 0, H - 1)
            x1 = jnp.clip(x0 + w, 0, W)
            y1 = jnp.clip(y0 + h, 0, H)
            ys = jnp.arange(H)[:, None]
            xs = jnp.arange(W)[None, :]
            mask = (ys >= y0) & (ys < y1) & (xs >= x0) & (xs < x1)
            color = jnp.array(color_rgb, dtype=jnp.uint8)
            return jnp.where(mask[..., None], color, scr)

        def draw_bitmap(scr, x, y, bitmap, color_rgb, scale):
            bmp = jnp.kron(bitmap, jnp.ones((scale, scale), dtype=jnp.uint8))
            h, w = bmp.shape
            x = jnp.clip(x, 0, W - w)
            y = jnp.clip(y, 0, H - h)
            pad = ((0, H - h), (0, W - w))
            bmp_padded = jnp.pad(bmp, pad)
            bmp_shifted = jnp.roll(jnp.roll(bmp_padded, y, axis=0), x, axis=1).astype(bool)
            color = jnp.array(color_rgb, dtype=jnp.uint8)
            return jnp.where(bmp_shifted[..., None], color, scr)

        def draw_digit(scr, x, y, d, color_rgb, scale):
            bmp = DIGITS[jnp.clip(d, 0, 9)]
            return draw_bitmap(scr, x, y, bmp, color_rgb, scale)

        def draw_number(scr, x, y, value, width, color_rgb, scale, spacing_):
            value = jnp.maximum(jnp.asarray(value, jnp.int32), 0)

            def body(i, carry):
                n, out = carry
                d = n % 10
                out = out.at[width - 1 - i].set(d)
                n = n // 10
                return (n, out)

            out_init = jnp.zeros((width,), dtype=jnp.int32)
            _, digits = lax.fori_loop(0, width, body, (value, out_init))

            def draw_i(i, scr_):
                return draw_digit(scr_, x + i * (3 * scale + spacing_), y, digits[i], color_rgb, scale)

            return lax.fori_loop(0, width, lambda i, scr_: draw_i(i, scr_), scr)

        def draw_label(scr, x, y, text, color_rgb, scale, spacing_):
            cur_x = x
            for ch in text:
                if ch == ' ':
                    cur_x += (3 * scale + spacing_)
                else:
                    scr = draw_bitmap(scr, cur_x, y, FONT[ch], color_rgb, scale)
                    cur_x += (3 * scale + spacing_)
            return scr

        # YELLOW REJUVENATOR SPRITE for lives
        def draw_rejuvenator_sprite(scr, x, y, scale):
            return draw_bitmap(scr, x, y, self.yellow_rejuv, YELL_RGB, scale)

        # =========================
        # ======== CONTENT ========
        # =========================

        # Enemies left (middle-left, GREEN color)
        ENEMIES_PER_SECTOR = getattr(self.constants, "ENEMIES_PER_SECTOR", 15)
        enemies_left = jnp.maximum(0, ENEMIES_PER_SECTOR - jnp.asarray(state.enemies_killed_this_sector, jnp.int32))
        screen = draw_number(
            screen, MARGIN_ENEMIES_X, MARGIN_ENEMIES_Y,
            enemies_left, width=2, color_rgb=GREEN, scale=SCALE_ENEMIES, spacing_=spacing
        )

        # SCORE (centered)
        label_score = "SCORE "
        char_px = 3 * SCALE_SCORE + spacing
        score_label_px = len(label_score) * char_px
        score_digits = 6
        score_total_px = score_label_px + score_digits * char_px
        score_x = (W - score_total_px) // 2
        score_y = MARGIN_SCORE_Y

        screen = draw_label(screen, score_x, score_y, label_score, GOLD, scale=SCALE_SCORE, spacing_=spacing)
        screen = draw_number(screen, score_x + score_label_px, score_y, jnp.asarray(state.score, jnp.int32),
                             width=6, color_rgb=GOLD, scale=SCALE_SCORE, spacing_=spacing)

        # SECTOR (centered below SCORE)
        label_sector = "SECTOR "
        sector_label_px = len(label_sector) * char_px
        sector_digits = 2
        sector_total_px = sector_label_px + sector_digits * char_px
        sector_x = (W - sector_total_px) // 2
        sector_y = score_y + (5 * SCALE_SCORE) + LINE_GAP_SCORE

        screen = draw_label(screen, sector_x, sector_y, label_sector, GOLD, scale=SCALE_SCORE, spacing_=spacing)
        screen = draw_number(screen, sector_x + sector_label_px, sector_y, jnp.asarray(state.level, jnp.int32),
                             width=2, color_rgb=GOLD, scale=SCALE_SCORE, spacing_=spacing)

        # Torpedoes (top-right)
        torps = jnp.asarray(state.torpedoes_remaining, jnp.int32)
        TORP_MAX = 8
        TORP_SIZE = 5 * SCALE_TORPS
        TORP_STEP = int(7 * SCALE_TORPS)

        def torp_body(i, scr_):
            draw_it = i < jnp.clip(torps, 0, TORP_MAX)
            x = W - MARGIN_TORPS_X - (i + 1) * TORP_STEP
            y = MARGIN_TORPS_Y
            scr2 = draw_rect(scr_, x, y, TORP_SIZE, TORP_SIZE, PURP)
            return jnp.where(draw_it, scr2, scr_)

        screen = lax.fori_loop(0, TORP_MAX, torp_body, screen)

        # Lives (bottom-left) – YELLOW REJUVENATOR SPRITES
        lives = jnp.asarray(state.lives, jnp.int32)
        LIVES_MAX = 6
        REJUV_W = 9 * SCALE_LIVES  # Width of rejuvenator sprite
        REJUV_H = 7 * SCALE_LIVES  # Height of rejuvenator sprite
        ICON_GAP = 2 * SCALE_LIVES  # Gap between sprites
        BASE_X = MARGIN_LIVES_X
        BASE_Y = H - REJUV_H - MARGIN_LIVES_Y

        def lives_body(i, scr_):
            draw_it = i < jnp.clip(lives, 0, LIVES_MAX)
            x = BASE_X + i * (REJUV_W + ICON_GAP)
            y = BASE_Y
            scr2 = draw_rejuvenator_sprite(scr_, x, y, SCALE_LIVES)
            return jnp.where(draw_it, scr2, scr_)

        screen = lax.fori_loop(0, LIVES_MAX, lives_body, screen)

        return screen
    # PYGAME DISPLAY METHODS (moved from BeamRiderPygameRenderer)
    # ============================================================================

    def run_game(self):
        #Main game loop with torpedo support - requires pygame to be enabled
        if not self.enable_pygame:
            raise RuntimeError("pygame must be enabled to run the game. Initialize with enable_pygame=True")

        key = random.PRNGKey(42)
        obs, state = self.env.reset(key)

        running = True
        paused = False

        while running and not state.game_over:
            # Handle quit & pause events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        paused = not paused

            if not paused:
                # Poll real-time key states for smoother controls
                keys = pygame.key.get_pressed()

                # Determine action based on key combinations
                action = 0  # default no-op

                # TORPEDO ACTIONS (actions 6, 7, 8) - CHECK FIRST!
                if keys[pygame.K_UP]:  # T for torpedo only
                    action = 2
                # LASER ACTIONS (actions 3, 4, 5)
                elif keys[pygame.K_SPACE]:
                    action = 1  # fire laser only
                # MOVEMENT ACTIONS (actions 1, 2)
                elif keys[pygame.K_LEFT]:
                    action = 4  # left
                elif keys[pygame.K_RIGHT]:
                    action = 3  # right

                # Step and render
                prev_state = state  # Store previous state
                obs, state, reward, done, info = self.env.step(state, action)

                # Check for sector completion
                self._show_sector_complete(state)
                screen_buffer = self.render(state)
                self._draw_screen(screen_buffer, state)
                self._draw_ui(screen_buffer,state)

                pygame.display.flip()
                self.clock.tick(60)
            else:
                self._draw_pause_overlay()
                pygame.display.flip()
                self.clock.tick(15)

        # Game over screen
        if state.game_over:
            self._show_game_over(state)

        pygame.quit()
        sys.exit()

    def _show_sector_complete(self, state):
        #Show sector completion message (called when sector advances)
        if hasattr(self, '_last_sector') and state.current_sector > self._last_sector:
            # Sector just advanced - show visual feedback

            # Create semi-transparent overlay
            overlay = pygame.Surface((self.pygame_screen_width, self.pygame_screen_height))
            overlay.set_alpha(180)
            overlay.fill((0, 0, 50))  # Dark blue overlay
            self.pygame_screen.blit(overlay, (0, 0))

            # Sector completion message
            sector_complete_text = self.font.render("SECTOR COMPLETE!", True, (0, 255, 0))
            next_sector_text = self.font.render(f"Advancing to Sector {state.current_sector}", True, (255, 255, 255))
            torpedoes_text = self.font.render("Torpedoes Refilled!", True, (255, 255, 0))

            # Center the messages
            sector_rect = sector_complete_text.get_rect(
                center=(self.pygame_screen_width // 2, self.pygame_screen_height // 2 - 40))
            next_rect = next_sector_text.get_rect(
                center=(self.pygame_screen_width // 2, self.pygame_screen_height // 2))
            torpedo_rect = torpedoes_text.get_rect(
                center=(self.pygame_screen_width // 2, self.pygame_screen_height // 2 + 40))

            # Draw the messages
            self.pygame_screen.blit(sector_complete_text, sector_rect)
            self.pygame_screen.blit(next_sector_text, next_rect)
            self.pygame_screen.blit(torpedoes_text, torpedo_rect)

            # Show for 2 seconds
            pygame.display.flip()
            pygame.time.wait(2000)

        # Update tracked sector
        self._last_sector = state.current_sector

    def _draw_pause_overlay(self):
        pause_text = self.render("PAUSED", True, (255, 220, 100))
        rect = pause_text.get_rect(center=(self.pygame_screen_width // 2, self.pygame_screen_height // 2))
        self.pygame_screen.blit(pause_text, rect)

    def _draw_screen(self, screen_buffer, state):
       #Draws the game screen buffer and overlays the ship sprite
        screen_np = np.array(screen_buffer)
        scaled_screen = np.repeat(np.repeat(screen_np, self.scale, axis=0), self.scale, axis=1)

        surf = pygame.surfarray.make_surface(scaled_screen.swapaxes(0, 1))
        self.pygame_screen.blit(surf, (0, 0))

        # === OVERLAY THE SHIP SPRITE ===
        ship_x = int(state.ship.x) * self.scale
        ship_y = int(state.ship.y) * self.scale
        self.pygame_screen.blit(self.ship_sprite_surface, (ship_x, ship_y))

    def _draw_ui_overlay(self, state):
       #Draw centered Score and Level UI - UPDATED: shows sentinel ship info
        # Enemies left number (top-left)
        enemies_left = 15 - state.enemies_killed_this_sector
        enemies_text = self.font.render(str(enemies_left), True, (255, 0, 0))  # red number

        # Small offset from top-left corner
        self.pygame_screen.blit(enemies_text, (10, 10))

        score_text = self.font.render(f"SCORE {state.score:06}", True, (255, 220, 100))
        level_text = self.font.render(f"SECTOR {state.level:02}", True, (255, 220, 100))

        score_rect = score_text.get_rect(center=(self.pygame_screen_width // 2, 20))
        level_rect = level_text.get_rect(center=(self.pygame_screen_width // 2, 42))

        self.pygame_screen.blit(score_text, score_rect)
        self.pygame_screen.blit(level_text, level_rect)

        # Draw purple torpedo cubes (top-right corner)
        cube_size = 16
        spacing = 6
        torpedoes = state.torpedoes_remaining

        for i in range(torpedoes):
            x = self.pygame_screen_width - (cube_size + spacing) * (i + 1) - 10  # from right edge
            y = 10  # a bit down from top
            pygame.draw.rect(self.pygame_screen, (160, 32, 240), (x, y, cube_size, cube_size))

        # === DRAW LIVES INDICATORS ===
        for i in range(state.lives):
            x = 30 + i * 36  # spacing between icons
            y = self.pygame_screen_height - 20  # near bottom
            scaled_ship = pygame.transform.scale(self.small_ship_surface,
                                                 (int(self.small_ship_surface.get_width() * 1.5),
                                                  int(self.small_ship_surface.get_height() * 1.5)))
            self.pygame_screen.blit(scaled_ship, (x, y))

    def _show_game_over(self, state):
        # Show Game Over screen
        overlay = pygame.Surface((self.pygame_screen_width, self.pygame_screen_height))
        overlay.set_alpha(128)
        overlay.fill((0, 0, 0))
        self.pygame_screen.blit(overlay, (0, 0))

        game_over_text = self.font.render("GAME OVER", True, (255, 0, 0))
        final_score_text = self.font.render(f"Final Score: {state.score}", True, (255, 255, 255))
        sector_text = self.font.render(f"Reached Sector: {state.current_sector}", True, (255, 255, 255))

        # Center the text
        game_over_rect = game_over_text.get_rect(
            center=(self.pygame_screen_width // 2, self.pygame_screen_height // 2 - 40))
        score_rect = final_score_text.get_rect(center=(self.pygame_screen_width // 2, self.pygame_screen_height // 2))
        sector_rect = sector_text.get_rect(center=(self.pygame_screen_width // 2, self.pygame_screen_height // 2 + 30))

        self.pygame_screen.blit(game_over_text, game_over_rect)
        self.pygame_screen.blit(final_score_text, score_rect)
        self.pygame_screen.blit(sector_text, sector_rect)



if __name__ == "__main__":
    game = BeamRiderRenderer(enable_pygame=True)
    game.run_game()