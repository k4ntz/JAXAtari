from functools import partial
import pygame
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit
from typing import Tuple, Dict, Any, NamedTuple
import chex
from flax import struct
import sys

from jaxatari import spaces
from jaxatari.environment import JaxEnvironment
from jaxatari.renderers import JAXGameRenderer


class BeamRiderConstants(NamedTuple):
    """Container for all game constants - MERGED from both versions"""

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
    BASE_ENEMY_SPAWN_INTERVAL = 90  # Start slower
    MIN_ENEMY_SPAWN_INTERVAL = 12  # End faster
    MAX_ENEMY_SPEED = 2.5  # Maximum enemy speed at sector 99

    # Enemy spawn position
    ENEMY_SPAWN_Y = 10

    # Enemy types - COMPLETE SET from old version
    ENEMY_TYPE_WHITE_SAUCER = 0
    ENEMY_TYPE_BROWN_DEBRIS = 1
    ENEMY_TYPE_YELLOW_CHIRPER = 2
    ENEMY_TYPE_GREEN_BLOCKER = 3
    ENEMY_TYPE_GREEN_BOUNCE = 4  # Green bounce craft
    ENEMY_TYPE_BLUE_CHARGER = 5  # Blue charger
    ENEMY_TYPE_ORANGE_TRACKER = 6  # Orange tracker
    ENEMY_TYPE_SENTINEL_SHIP = 7  # Sentinel ship
    ENEMY_TYPE_YELLOW_REJUVENATOR = 8  # Yellow rejuvenator
    ENEMY_TYPE_REJUVENATOR_DEBRIS = 9  # Explosive debris from shot rejuvenators

    # White saucer behavior constants - FROM CURRENT VERSION
    WHITE_SAUCER_SHOOT_CHANCE = 0.2  # 20% of white saucers can shoot
    WHITE_SAUCER_JUMP_CHANCE = 0.15  # 15% chance for beam jumping
    WHITE_SAUCER_REVERSE_CHANCE = 0.1  # 10% chance for reverse movement
    WHITE_SAUCER_ZIGZAG_CHANCE = 0.1  # 10% chance for zigzag movement
    WHITE_SAUCER_FIRING_INTERVAL = 90  # Frames between shots
    WHITE_SAUCER_PROJECTILE_SPEED = 2.5  # Speed of white saucer projectiles
    WHITE_SAUCER_JUMP_INTERVAL = 30  # Frames between beam jumps
    WHITE_SAUCER_REVERSE_SPEED = -1.5  # Reverse movement speed (going back up)
    WHITE_SAUCER_ZIGZAG_AMPLITUDE = 15  # Horizontal movement range for zigzag
    WHITE_SAUCER_ZIGZAG_FREQUENCY = 0.1  # Zigzag frequency

    # White saucer movement patterns
    WHITE_SAUCER_STRAIGHT_DOWN = 0
    WHITE_SAUCER_BEAM_JUMP = 1
    WHITE_SAUCER_REVERSE_UP = 2
    WHITE_SAUCER_ZIGZAG = 3
    WHITE_SAUCER_SHOOTING = 4
    # White saucer reverse pattern constants
    WHITE_SAUCER_REVERSE_TRIGGER_Y = 150  # Y position where reverse pattern triggers
    WHITE_SAUCER_REVERSE_SPEED_FAST = -4.0
    UPPER_THIRD_Y = 70  # Y = 70, upper third boundary

    # White saucer smooth movement
    WHITE_SAUCER_HORIZONTAL_SPEED = 1.5  # Pixels per frame horizontal movement
    WHITE_SAUCER_BEAM_SNAP_DISTANCE = 3  # Distance to snap to target beam

    # Sentinel ship specific constants - FROM OLD VERSION
    SENTINEL_SHIP_SPEED = 0.1  # Moderate base speed, will scale
    SENTINEL_SHIP_POINTS = 200  # High points when destroyed with torpedo
    SENTINEL_SHIP_COLOR = (192, 192, 192)  # Silver/grey color RGB
    SENTINEL_SHIP_SPAWN_SECTOR = 1  # Starts appearing from sector 1
    SENTINEL_SHIP_SPAWN_CHANCE = 0.05  # 5% chance to spawn sentinel ship (rare)
    SENTINEL_SHIP_WIDTH = 12  # Larger than regular enemies
    SENTINEL_SHIP_HEIGHT = 10  # Larger than regular enemies
    SENTINEL_SHIP_FIRING_INTERVAL = 120  # Frames between shots (2 seconds at 60fps)
    SENTINEL_SHIP_PROJECTILE_SPEED = 3.0  # Speed of sentinel projectiles
    SENTINEL_SHIP_HEALTH = 1  # Takes 1 torpedo hit to destroy

    # Orange tracker specific constants - FROM OLD VERSION
    ORANGE_TRACKER_SPEED = 0.9  # Slower base tracking speed
    ORANGE_TRACKER_POINTS = 50  # Points when destroyed with torpedo
    ORANGE_TRACKER_COLOR = (255, 165, 0)  # Orange color RGB
    ORANGE_TRACKER_SPAWN_SECTOR = 12  # Starts appearing from sector 12
    ORANGE_TRACKER_SPAWN_CHANCE = 0.08  # 8% chance to spawn orange tracker
    ORANGE_TRACKER_CHANGE_DIRECTION_INTERVAL = 90  # Frames between direction changes

    # Tracker course change limits based on sector
    ORANGE_TRACKER_BASE_COURSE_CHANGES = 1  # Base number of course changes allowed
    ORANGE_TRACKER_COURSE_CHANGE_INCREASE_SECTOR = 5  # Every X sectors, add 1 more course change

    # Blue charger specific constants - FROM OLD VERSION
    BLUE_CHARGER_SPEED = 1.1  # Slower base speed
    BLUE_CHARGER_POINTS = 30  # Points when destroyed
    BLUE_CHARGER_COLOR = (0, 0, 255)  # Blue color RGB
    BLUE_CHARGER_SPAWN_SECTOR = 1  # Starts appearing from sector 10
    BLUE_CHARGER_SPAWN_CHANCE = 0.1  # 10% chance to spawn blue charger
    BLUE_CHARGER_LINGER_TIME = 180  # Frames to stay at bottom (3 seconds at 60fps)
    BLUE_CHARGER_DEFLECT_SPEED = -2.0  # Speed when deflected upward by laser

    # Brown debris specific constants - FROM OLD VERSION
    BROWN_DEBRIS_SPEED = 1.0  # Slower base speed
    BROWN_DEBRIS_POINTS = 25  # Bonus points when destroyed with torpedo
    BROWN_DEBRIS_COLOR = (139, 69, 19)  # Brown color RGB

    # Spawn probabilities
    BROWN_DEBRIS_SPAWN_SECTOR = 2  # Starts appearing from sector 2
    BROWN_DEBRIS_SPAWN_CHANCE = 0.15  # 15% chance to spawn brown debris

    # Yellow chirper specific constants - FROM OLD VERSION
    YELLOW_CHIRPER_SPEED = 0.7  # Slower horizontal movement speed
    YELLOW_CHIRPER_POINTS = 50  # Bonus points for shooting them
    YELLOW_CHIRPER_COLOR = (255, 255, 0)  # Yellow color RGB
    YELLOW_CHIRPER_SPAWN_Y_MIN = 50  # Minimum Y position for horizontal flight
    YELLOW_CHIRPER_SPAWN_Y_MAX = 150  # Maximum Y position for horizontal flight

    YELLOW_CHIRPER_SPAWN_SECTOR = 4  # Starts appearing from sector 4
    YELLOW_CHIRPER_SPAWN_CHANCE = 0.1  # 10% chance to spawn yellow chirper

    # Green blocker specific constants - FROM OLD VERSION
    GREEN_BLOCKER_SPEED = 0.15  # Much slower ramming speed
    GREEN_BLOCKER_POINTS = 75  # High points when destroyed
    GREEN_BLOCKER_COLOR = (0, 255, 0)  # Green color RGB
    GREEN_BLOCKER_SPAWN_Y_MIN = 30  # Spawn higher up for targeting
    GREEN_BLOCKER_SPAWN_Y_MAX = 80  # Range for side spawning
    GREEN_BLOCKER_LOCK_DISTANCE = 100  # Distance at which they lock onto player beam
    GREEN_BLOCKER_SPAWN_SECTOR = 6  # Starts appearing from sector 6
    GREEN_BLOCKER_SPAWN_CHANCE = 0.12  # 12% chance to spawn green blocker
    GREEN_BLOCKER_SENTINEL_SPAWN_CHANCE = 0.3  # 30% chance when sentinel is active in sectors 1-5

    # Green bounce craft specific constants - FROM OLD VERSION
    GREEN_BOUNCE_SPEED = 1.5  # Slower bouncing speed
    GREEN_BOUNCE_POINTS = 100  # Very high points when destroyed with torpedo
    GREEN_BOUNCE_COLOR = (0, 200, 0)  # Slightly different green than blockers
    GREEN_BOUNCE_SPAWN_SECTOR = 8  # Starts appearing from sector 8
    GREEN_BOUNCE_SPAWN_CHANCE = 0.08  # 8% chance to spawn green bounce craft
    GREEN_BOUNCE_MAX_BOUNCES = 6  # Maximum number of bounces before disappearing

    # Yellow rejuvenator specific constants - FROM OLD VERSION
    YELLOW_REJUVENATOR_SPEED = 0.5  # Slow float speed
    YELLOW_REJUVENATOR_POINTS = 0  # No points for shooting (discourage shooting)
    YELLOW_REJUVENATOR_LIFE_BONUS = 1  # Adds 1 life when collected
    YELLOW_REJUVENATOR_COLOR = (255, 255, 100)  # Bright yellow color RGB
    YELLOW_REJUVENATOR_SPAWN_SECTOR = 5  # Starts appearing from sector 5
    YELLOW_REJUVENATOR_SPAWN_CHANCE = 0.04  # 4% chance to spawn (rare)
    YELLOW_REJUVENATOR_OSCILLATION_AMPLITUDE = 15  # Horizontal oscillation range
    YELLOW_REJUVENATOR_OSCILLATION_FREQUENCY = 0.06  # Oscillation frequency

    # Rejuvenator debris constants (when shot) - FROM OLD VERSION
    REJUVENATOR_DEBRIS_SPEED = 1.5  # Fast moving debris
    REJUVENATOR_DEBRIS_COLOR = (255, 0, 0)  # Red explosive debris
    REJUVENATOR_DEBRIS_COUNT = 4  # Number of debris pieces created
    REJUVENATOR_DEBRIS_SPREAD = 30  # Spread angle for debris
    REJUVENATOR_DEBRIS_LIFETIME = 180  # Frames before debris disappears

    # HUD margins
    TOP_MARGIN = int(210 * 0.12)

    @classmethod
    def get_beam_positions(cls) -> jnp.ndarray:
        """Calculate 5 beam positions evenly spaced across the screen width - FIXED"""
        # Simple, direct calculation for 5 evenly spaced beams
        # Beam 0: leftmost, Beam 2: center, Beam 4: rightmost

        # Leave some margin from screen edges
        margin = 20
        usable_width = cls.SCREEN_WIDTH - (2 * margin)

        # Create 5 evenly spaced positions
        beam_spacing = usable_width / (cls.NUM_BEAMS - 1)
        positions = jnp.array([
            margin + i * beam_spacing for i in range(cls.NUM_BEAMS)
        ])

        return positions

@struct.dataclass
class Ship:
    # Represents the player-controlled ship: position, beam lane, and active status.
    x: float
    y: float
    beam_position: int  # Index of the current beam (0â€"4)
    active: bool = True  # Whether the ship is currently active (alive)


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
    # NEW: Sentinel ship specific fields
    health: int = 1  # Health for sentinel ships (default 1 for other enemies)
    firing_timer: int = 0  # Timer for sentinel ship firing
    maneuver_timer: int = 0  # Timer for evasive maneuvers


# FROM CURRENT VERSION - JAXAtari interface structures
class BeamRiderObservation(NamedTuple):
    ship_beam: jnp.ndarray
    ship_x: jnp.ndarray
    ship_y: jnp.ndarray
    ship_active: jnp.ndarray
    enemies: jnp.ndarray
    projectiles: jnp.ndarray
    torpedo_projectiles: jnp.ndarray
    sentinel_projectiles: jnp.ndarray
    score: jnp.ndarray
    lives: jnp.ndarray
    level: jnp.ndarray
    current_sector: jnp.ndarray
    torpedoes_remaining: jnp.ndarray
    frame_count: jnp.ndarray


class BeamRiderInfo(NamedTuple):
    enemies_killed_this_sector: jnp.ndarray
    enemy_spawn_timer: jnp.ndarray
    enemy_spawn_interval: jnp.ndarray
    sentinel_spawned_this_sector: jnp.ndarray
    game_over: jnp.ndarray


@struct.dataclass
class BeamRiderState:
    """Complete game state"""
    # Game entities
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
    # Fields WITH defaults (must come last)
    enemy_spawn_interval: int = BeamRiderConstants.ENEMY_SPAWN_INTERVAL


class BeamRiderEnv(JaxEnvironment[BeamRiderState, BeamRiderObservation, BeamRiderInfo, BeamRiderConstants]):
    """BeamRider environment following JAXAtari structure"""

    def __init__(self):
        self.constants = BeamRiderConstants()
        self.screen_width = self.constants.SCREEN_WIDTH
        self.screen_height = self.constants.SCREEN_HEIGHT
        self.action_space_size = 18  # Updated to use full JAXAtari action space
        self.beam_positions = self.constants.get_beam_positions()

        # Initialize renderer
        self.renderer = BeamRiderRenderer()

    def reset(self, rng_key: chex.PRNGKey) -> Tuple[BeamRiderObservation, BeamRiderState]:
        """Reset the game to initial state - FROM CURRENT VERSION"""
        # Initialize ship at bottom center beam
        initial_beam = self.constants.INITIAL_BEAM
        ship = Ship(
            x=self.beam_positions[initial_beam] - self.constants.SHIP_WIDTH // 2,
            y=self.constants.SCREEN_HEIGHT - self.constants.SHIP_BOTTOM_OFFSET,
            beam_position=initial_beam,
            active=True
        )

        # Initialize empty projectiles arrays (4 columns each)
        projectiles = jnp.zeros((self.constants.MAX_PROJECTILES, 4))  # x, y, active, speed
        torpedo_projectiles = jnp.zeros((self.constants.MAX_PROJECTILES, 4))  # x, y, active, speed
        sentinel_projectiles = jnp.zeros((self.constants.MAX_PROJECTILES, 4))  # x, y, active, speed

        # Initialize empty enemies array - 18 columns for white saucer enhancements
        enemies = jnp.zeros((self.constants.MAX_ENEMIES, 18))
        # x, y, beam_position, active, speed, type, direction_x, direction_y,
        # bounce_count, linger_timer, target_x, health, firing_timer, maneuver_timer,
        # movement_pattern, white_saucer_firing_timer, jump_timer, zigzag_offset

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

        # Return observation and state
        obs = self._get_observation(state)
        return obs, state

    def action_space(self) -> spaces.Discrete:
        """Returns the action space for BeamRider - FROM CURRENT VERSION"""
        return spaces.Discrete(18)  # All standard Atari actions available

    def observation_space(self) -> spaces.Box:
        """Returns the observation space for BeamRider - FROM CURRENT VERSION"""
        return spaces.Box(
            low=jnp.full((100,), -1e6, dtype=jnp.float32),
            high=jnp.full((100,), 1e6, dtype=jnp.float32),
            dtype=jnp.float32
        )

    def image_space(self) -> spaces.Box:
        """Returns the image space for BeamRider rendering - FROM CURRENT VERSION"""
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.constants.SCREEN_HEIGHT, self.constants.SCREEN_WIDTH, 3),
            dtype=jnp.uint8
        )

    def render(self, state: BeamRiderState) -> jnp.ndarray:
        """Render the current game state - delegates to renderer"""
        return self.renderer.render(state)

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: BeamRiderState) -> BeamRiderObservation:
        """Extract observation from game state - FROM CURRENT VERSION"""
        return BeamRiderObservation(
            ship_beam=jnp.array(state.ship.beam_position, dtype=jnp.float32),
            ship_x=jnp.array(state.ship.x, dtype=jnp.float32),
            ship_y=jnp.array(state.ship.y, dtype=jnp.float32),
            ship_active=jnp.array(state.ship.active, dtype=jnp.float32),
            enemies=state.enemies,
            projectiles=state.projectiles,
            torpedo_projectiles=state.torpedo_projectiles,
            sentinel_projectiles=state.sentinel_projectiles,
            score=jnp.array(state.score, dtype=jnp.float32),
            lives=jnp.array(state.lives, dtype=jnp.float32),
            level=jnp.array(state.level, dtype=jnp.float32),
            current_sector=jnp.array(state.current_sector, dtype=jnp.float32),
            torpedoes_remaining=jnp.array(state.torpedoes_remaining, dtype=jnp.float32),
            frame_count=jnp.array(state.frame_count, dtype=jnp.float32)
        )

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: BeamRiderObservation) -> jnp.ndarray:
        """Convert observation to flat array for ML models - FROM CURRENT VERSION"""
        flat_components = [
            obs.ship_beam.flatten(),
            obs.ship_x.flatten(),
            obs.ship_y.flatten(),
            obs.ship_active.flatten(),
            obs.enemies.flatten(),
            obs.projectiles.flatten(),
            obs.torpedo_projectiles.flatten(),
            obs.sentinel_projectiles.flatten(),
            obs.score.flatten(),
            obs.lives.flatten(),
            obs.level.flatten(),
            obs.current_sector.flatten(),
            obs.torpedoes_remaining.flatten(),
            obs.frame_count.flatten()
        ]

        flat_obs = jnp.concatenate(flat_components)

        # Pad or truncate to fixed size (100 elements)
        target_size = 100
        if len(flat_obs) < target_size:
            flat_obs = jnp.pad(flat_obs, (0, target_size - len(flat_obs)))
        else:
            flat_obs = flat_obs[:target_size]

        return flat_obs

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: BeamRiderState) -> BeamRiderInfo:
        """Extract additional info from game state - FROM CURRENT VERSION"""
        return BeamRiderInfo(
            enemies_killed_this_sector=jnp.array(state.enemies_killed_this_sector),
            enemy_spawn_timer=jnp.array(state.enemy_spawn_timer),
            enemy_spawn_interval=jnp.array(state.enemy_spawn_interval),
            sentinel_spawned_this_sector=jnp.array(state.sentinel_spawned_this_sector),
            game_over=jnp.array(state.game_over)
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: BeamRiderState, state: BeamRiderState) -> float:
        """Calculate reward based on state changes - FROM CURRENT VERSION"""
        # Primary reward: score increase
        score_reward = state.score - previous_state.score

        # Penalty for losing a life
        life_penalty = (previous_state.lives - state.lives) * 100

        # Small bonus for staying alive
        survival_reward = jnp.where(state.ship.active, 1.0, 0.0)

        # Bonus for advancing sectors
        sector_bonus = (state.current_sector - previous_state.current_sector) * 500

        # Small penalty for using torpedoes (encourage strategic use)
        torpedo_penalty = (previous_state.torpedoes_remaining - state.torpedoes_remaining) * 5

        total_reward = score_reward + sector_bonus + survival_reward - life_penalty - torpedo_penalty

        return total_reward

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: BeamRiderState) -> bool:
        """Determine if the game is over - FROM CURRENT VERSION"""
        return state.game_over

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: BeamRiderState, action: int) -> Tuple[
        BeamRiderObservation, BeamRiderState, float, bool, BeamRiderInfo]:
        """Execute one game step - FROM CURRENT VERSION"""
        previous_state = state
        new_state = self._step_impl(state, action)

        obs = self._get_observation(new_state)
        reward = self._get_reward(previous_state, new_state)
        done = self._get_done(new_state)
        info = self._get_info(new_state)

        return obs, new_state, reward, done, info

    def _step_impl(self, state: BeamRiderState, action: int) -> BeamRiderState:
        """Execute one game step - MIXED: Current version structure with old version logic"""
        # Process player input and update ship - FROM OLD VERSION (smooth movement)
        state = self._update_ship(state, action)

        # Handle projectile firing - FROM OLD VERSION
        state = self._handle_firing(state, action)

        # Update projectiles - FROM OLD VERSION
        state = self._update_projectiles(state)

        # Spawn enemies - FROM OLD VERSION (complex spawning logic)
        state = self._spawn_enemies(state)

        # Update enemies - FROM OLD VERSION (except white saucers from current)
        state = self._update_enemies(state)

        # Handle white saucer shooting - FROM CURRENT VERSION
        state = self._handle_white_saucer_shooting(state)

        # Update sentinel ship projectiles - FROM OLD VERSION
        state = self._update_sentinel_projectiles(state)

        # Check collisions - FROM OLD VERSION
        state = self._check_collisions(state)

        # Check sector progression - FROM OLD VERSION
        state = self._check_sector_progression(state)

        # Check game over conditions - FROM OLD VERSION
        state = self._check_game_over(state)

        # Update frame count only once at the end
        state = state.replace(frame_count=state.frame_count + 1)

        return state

    def _update_ship(self, state: BeamRiderState, action: int) -> BeamRiderState:
        """Update ship position using discrete beam movement"""
        ship = state.ship
        current_beam = ship.beam_position

        # Discrete beam movement:
        # - Move left if action == 1 and not at leftmost beam
        # - Move right if action == 2 and not at rightmost beam
        # - Stay in place otherwise
        new_beam_position = jnp.where(
            (action == 4) & (current_beam > 0),  # Move left
            current_beam - 1,
            jnp.where(
                (action == 3) & (current_beam < self.constants.NUM_BEAMS - 1),  # Move right
                current_beam + 1,
                current_beam  # No movement or invalid action
            )
        )

        # Set ship x position to exactly match the beam center
        new_x = self.beam_positions[new_beam_position] - self.constants.SHIP_WIDTH // 2

        return state.replace(ship=ship.replace(x=new_x, beam_position=new_beam_position))
    def _select_white_saucer_movement_pattern(self, rng_key: chex.PRNGKey) -> int:
        """Select movement pattern for a new white saucer - FROM CURRENT VERSION"""
        # Generate random value for pattern selection
        pattern_rand = random.uniform(rng_key, (), minval=0.0, maxval=1.0)

        # Determine movement pattern based on probabilities
        pattern = jnp.where(
            pattern_rand < self.constants.WHITE_SAUCER_REVERSE_CHANCE,
            self.constants.WHITE_SAUCER_REVERSE_UP,
            jnp.where(
                pattern_rand < (self.constants.WHITE_SAUCER_REVERSE_CHANCE +
                                self.constants.WHITE_SAUCER_ZIGZAG_CHANCE),
                self.constants.WHITE_SAUCER_ZIGZAG,
                jnp.where(
                    pattern_rand < (self.constants.WHITE_SAUCER_REVERSE_CHANCE +
                                    self.constants.WHITE_SAUCER_ZIGZAG_CHANCE +
                                    self.constants.WHITE_SAUCER_JUMP_CHANCE),
                    self.constants.WHITE_SAUCER_BEAM_JUMP,
                    jnp.where(
                        pattern_rand < (self.constants.WHITE_SAUCER_REVERSE_CHANCE +
                                        self.constants.WHITE_SAUCER_ZIGZAG_CHANCE +
                                        self.constants.WHITE_SAUCER_JUMP_CHANCE +
                                        self.constants.WHITE_SAUCER_SHOOT_CHANCE),
                        self.constants.WHITE_SAUCER_SHOOTING,
                        self.constants.WHITE_SAUCER_STRAIGHT_DOWN
                    )
                )
            )
        )

        return pattern

    def _handle_white_saucer_shooting(self, state: BeamRiderState) -> BeamRiderState:
        """Handle white saucer projectile firing - FROM CURRENT VERSION"""
        enemies = state.enemies

        # Find shooting white saucers that are ready to fire
        white_saucer_mask = enemies[:, 5] == self.constants.ENEMY_TYPE_WHITE_SAUCER
        active_mask = enemies[:, 3] == 1
        shooting_pattern_mask = enemies[:, 14] == self.constants.WHITE_SAUCER_SHOOTING
        ready_to_fire_mask = enemies[:, 15] == 0  # white_saucer_firing_timer is 0

        can_shoot = white_saucer_mask & active_mask & shooting_pattern_mask & ready_to_fire_mask

        # Find any white saucer that can shoot
        any_can_shoot = jnp.any(can_shoot)

        # Find first shooting white saucer
        shooter_idx = jnp.argmax(can_shoot)

        # Get shooter position (only valid if any_can_shoot is True)
        shooter_x = enemies[shooter_idx, 0]
        shooter_y = enemies[shooter_idx, 1]

        # Create projectile at saucer position, moving downward
        projectile_x = shooter_x + self.constants.ENEMY_WIDTH // 2 - self.constants.PROJECTILE_WIDTH // 2
        projectile_y = shooter_y + self.constants.ENEMY_HEIGHT

        new_projectile = jnp.array([
            projectile_x,
            projectile_y,
            1,  # active
            self.constants.WHITE_SAUCER_PROJECTILE_SPEED  # speed (positive = downward)
        ])

        # Find first inactive slot in sentinel projectiles array (reuse for white saucer projectiles)
        sentinel_projectiles = state.sentinel_projectiles
        inactive_mask = sentinel_projectiles[:, 2] == 0
        first_inactive = jnp.argmax(inactive_mask)
        can_spawn_projectile = inactive_mask[first_inactive]

        # Create and add projectile if conditions are met
        should_fire = any_can_shoot & can_spawn_projectile

        # Update sentinel projectiles array (using it for white saucer projectiles too)
        sentinel_projectiles = jnp.where(
            should_fire,
            sentinel_projectiles.at[first_inactive].set(new_projectile),
            sentinel_projectiles
        )

        # Reset firing timer for the shooter
        enemies = jnp.where(
            should_fire,
            enemies.at[shooter_idx, 15].set(self.constants.WHITE_SAUCER_FIRING_INTERVAL),
            enemies
        )

        return state.replace(
            enemies=enemies,
            sentinel_projectiles=sentinel_projectiles
        )

    def _update_white_saucer_movement(self, state: BeamRiderState) -> BeamRiderState:
        """Enhanced white saucer movement patterns - FROM CURRENT VERSION"""
        enemies = state.enemies

        # Get white saucer mask
        white_saucer_mask = enemies[:, 5] == self.constants.ENEMY_TYPE_WHITE_SAUCER
        active_mask = enemies[:, 3] == 1
        white_saucer_active = white_saucer_mask & active_mask

        # Get current positions and properties
        current_x = enemies[:, 0]
        current_y = enemies[:, 1]
        current_beam = enemies[:, 2].astype(int)
        current_speed = enemies[:, 4]
        movement_pattern = enemies[:, 14].astype(int)
        firing_timer = enemies[:, 15].astype(int)
        jump_timer = enemies[:, 16].astype(int)
        zigzag_offset = enemies[:, 17]
        target_beam = enemies[:, 10].astype(int)  # Using target_x field for target beam

        # Update timers
        new_firing_timer = jnp.maximum(0, firing_timer - 1)
        new_jump_timer = jnp.maximum(0, jump_timer - 1)

        # UNIVERSAL REVERSE CONDITION - ALL WHITE SAUCERS REVERSE WHEN THEY GET TOO LOW
        reached_reverse_point = current_y >= self.constants.WHITE_SAUCER_REVERSE_TRIGGER_Y
        should_be_moving_up = white_saucer_active & reached_reverse_point

        # PATTERN 0: STRAIGHT_DOWN
        straight_mask = white_saucer_active & (movement_pattern == self.constants.WHITE_SAUCER_STRAIGHT_DOWN)
        straight_new_x = current_x
        # Apply reverse logic: move up if past trigger, down if not
        straight_new_y = jnp.where(
            should_be_moving_up,
            current_y + self.constants.WHITE_SAUCER_REVERSE_SPEED_FAST,  # Move up fast
            current_y + current_speed  # Normal downward movement
        )
        straight_new_beam = current_beam
        straight_new_speed = jnp.where(
            should_be_moving_up,
            self.constants.WHITE_SAUCER_REVERSE_SPEED_FAST,
            current_speed
        )
        straight_new_target_beam = target_beam

        # PATTERN 1: SMOOTH_BEAM_JUMP
        jump_mask = white_saucer_active & (movement_pattern == self.constants.WHITE_SAUCER_BEAM_JUMP)

        # Check if in upper third of screen for new target selection (only when moving down)
        in_upper_third = current_y <= self.constants.UPPER_THIRD_Y
        can_select_new_target = jump_mask & (new_jump_timer == 0) & in_upper_third & ~should_be_moving_up

        # Generate new target beam only if conditions are met
        jump_indices = jnp.arange(self.constants.MAX_ENEMIES)
        jump_rng_keys = jax.vmap(lambda i: random.fold_in(state.rng_key, state.frame_count + i))(jump_indices)
        new_random_beams = jax.vmap(lambda key: random.randint(key, (), 0, self.constants.NUM_BEAMS))(jump_rng_keys)

        # Update target beam: set new target if can select, otherwise keep current target
        jump_new_target_beam = jnp.where(
            can_select_new_target,
            new_random_beams,
            target_beam
        )

        # Reset timer only when new target is selected
        new_jump_timer = jnp.where(
            can_select_new_target,
            self.constants.WHITE_SAUCER_JUMP_INTERVAL,
            new_jump_timer
        )

        # Calculate target position
        target_x = self.beam_positions[jump_new_target_beam] - self.constants.ENEMY_WIDTH // 2

        # Smooth horizontal movement toward target (ONLY when moving down)
        x_diff = target_x - current_x
        movement_needed = jnp.abs(x_diff) > self.constants.WHITE_SAUCER_BEAM_SNAP_DISTANCE

        # Calculate horizontal movement direction and speed (ONLY when moving down)
        horizontal_direction = jnp.sign(x_diff)
        horizontal_movement = jnp.where(
            movement_needed & ~should_be_moving_up,  # NO horizontal movement when reversing
            horizontal_direction * self.constants.WHITE_SAUCER_HORIZONTAL_SPEED,
            0.0
        )

        # Update x position - STOP all horizontal movement when reversing
        jump_new_x = jnp.where(
            jump_mask,
            jnp.where(
                should_be_moving_up,
                current_x,  # NO horizontal movement when moving up - stay on current x
                jnp.where(
                    movement_needed,
                    current_x + horizontal_movement,
                    target_x  # Snap to target if close enough (only when moving down)
                )
            ),
            current_x
        )

        # Update current beam - STOP all beam changes when reversing
        close_to_target = jnp.abs(jump_new_x - target_x) <= self.constants.WHITE_SAUCER_BEAM_SNAP_DISTANCE
        jump_new_beam = jnp.where(
            jump_mask & close_to_target & ~should_be_moving_up,  # NO beam changes when reversing
            jump_new_target_beam,
            current_beam  # Keep current beam when moving up
        )

        # Vertical movement: reverse if past trigger, normal movement if not
        jump_new_y = jnp.where(
            should_be_moving_up,
            current_y + self.constants.WHITE_SAUCER_REVERSE_SPEED_FAST,  # Move up fast
            current_y + current_speed  # Normal downward movement
        )
        jump_new_speed = jnp.where(
            should_be_moving_up,
            self.constants.WHITE_SAUCER_REVERSE_SPEED_FAST,
            current_speed
        )

        # PATTERN 2: REVERSE_UP (this pattern already reverses, but now all do)
        reverse_mask = white_saucer_active & (movement_pattern == self.constants.WHITE_SAUCER_REVERSE_UP)
        reverse_new_speed = jnp.where(
            should_be_moving_up,
            self.constants.WHITE_SAUCER_REVERSE_SPEED_FAST,
            current_speed
        )
        reverse_new_x = current_x
        reverse_new_y = jnp.where(
            should_be_moving_up,
            current_y + self.constants.WHITE_SAUCER_REVERSE_SPEED_FAST,
            current_y + current_speed
        )
        reverse_new_beam = current_beam
        reverse_new_target_beam = target_beam

        # PATTERN 3: ZIGZAG
        zigzag_mask = white_saucer_active & (movement_pattern == self.constants.WHITE_SAUCER_ZIGZAG)
        new_zigzag_offset = jnp.where(
            zigzag_mask & ~should_be_moving_up,  # Only zigzag when moving down
            zigzag_offset + self.constants.WHITE_SAUCER_ZIGZAG_FREQUENCY,
            zigzag_offset
        )
        beam_center_x = self.beam_positions[current_beam]
        zigzag_delta = jnp.where(
            should_be_moving_up,
            0.0,  # NO zigzag when moving up - completely straight
            jnp.sin(new_zigzag_offset) * self.constants.WHITE_SAUCER_ZIGZAG_AMPLITUDE
        )
        zigzag_new_x = jnp.where(
            should_be_moving_up,
            current_x,  # KEEP current x position when moving up - no horizontal movement
            jnp.clip(
                beam_center_x + zigzag_delta - self.constants.ENEMY_WIDTH // 2,
                0, self.constants.SCREEN_WIDTH - self.constants.ENEMY_WIDTH
            )
        )
        zigzag_new_y = jnp.where(
            should_be_moving_up,
            current_y + self.constants.WHITE_SAUCER_REVERSE_SPEED_FAST,
            current_y + current_speed
        )
        zigzag_new_beam = current_beam  # NEVER change beam when moving up
        zigzag_new_speed = jnp.where(
            should_be_moving_up,
            self.constants.WHITE_SAUCER_REVERSE_SPEED_FAST,
            current_speed
        )
        zigzag_new_target_beam = target_beam

        # PATTERN 4: SHOOTING
        shooting_mask = white_saucer_active & (movement_pattern == self.constants.WHITE_SAUCER_SHOOTING)
        shooting_new_x = current_x
        shooting_new_y = jnp.where(
            should_be_moving_up,
            current_y + self.constants.WHITE_SAUCER_REVERSE_SPEED_FAST,
            current_y + current_speed
        )
        shooting_new_beam = current_beam
        shooting_new_speed = jnp.where(
            should_be_moving_up,
            self.constants.WHITE_SAUCER_REVERSE_SPEED_FAST,
            current_speed
        )
        shooting_new_target_beam = target_beam

        # Apply movement patterns
        new_x = jnp.where(straight_mask, straight_new_x,
                          jnp.where(jump_mask, jump_new_x,
                                    jnp.where(reverse_mask, reverse_new_x,
                                              jnp.where(zigzag_mask, zigzag_new_x,
                                                        jnp.where(shooting_mask, shooting_new_x, current_x)))))

        new_y = jnp.where(straight_mask, straight_new_y,
                          jnp.where(jump_mask, jump_new_y,
                                    jnp.where(reverse_mask, reverse_new_y,
                                              jnp.where(zigzag_mask, zigzag_new_y,
                                                        jnp.where(shooting_mask, shooting_new_y, current_y)))))

        new_beam = jnp.where(straight_mask, straight_new_beam,
                             jnp.where(jump_mask, jump_new_beam,
                                       jnp.where(reverse_mask, reverse_new_beam,
                                                 jnp.where(zigzag_mask, zigzag_new_beam,
                                                           jnp.where(shooting_mask, shooting_new_beam, current_beam)))))

        new_speed = jnp.where(straight_mask, straight_new_speed,
                              jnp.where(jump_mask, jump_new_speed,
                                        jnp.where(reverse_mask, reverse_new_speed,
                                                  jnp.where(zigzag_mask, zigzag_new_speed,
                                                            jnp.where(shooting_mask, shooting_new_speed,
                                                                      current_speed)))))

        new_target_beam = jnp.where(straight_mask, straight_new_target_beam,
                                    jnp.where(jump_mask, jump_new_target_beam,
                                              jnp.where(reverse_mask, reverse_new_target_beam,
                                                        jnp.where(zigzag_mask, zigzag_new_target_beam,
                                                                  jnp.where(shooting_mask, shooting_new_target_beam,
                                                                            target_beam)))))

        # DEACTIVATE WHITE SAUCERS THAT GO TOO HIGH (off the top of screen)
        new_active = white_saucer_active & (new_y > -self.constants.ENEMY_HEIGHT)

        # Update enemy array with new positions and timers
        enemies = enemies.at[:, 0].set(jnp.where(white_saucer_active, new_x, enemies[:, 0]))  # x position
        enemies = enemies.at[:, 1].set(jnp.where(white_saucer_active, new_y, enemies[:, 1]))  # y position
        enemies = enemies.at[:, 2].set(jnp.where(white_saucer_active, new_beam, enemies[:, 2]))  # current beam
        enemies = enemies.at[:, 3].set(
            jnp.where(white_saucer_active, new_active.astype(jnp.float32), enemies[:, 3]))  # active
        enemies = enemies.at[:, 4].set(jnp.where(white_saucer_active, new_speed, enemies[:, 4]))  # speed
        enemies = enemies.at[:, 10].set(jnp.where(white_saucer_active, new_target_beam, enemies[:, 10]))  # target beam
        enemies = enemies.at[:, 15].set(
            jnp.where(white_saucer_active, new_firing_timer, enemies[:, 15]))  # firing timer
        enemies = enemies.at[:, 16].set(jnp.where(white_saucer_active, new_jump_timer, enemies[:, 16]))  # jump timer
        enemies = enemies.at[:, 17].set(
            jnp.where(white_saucer_active, new_zigzag_offset, enemies[:, 17]))  # zigzag offset

        return state.replace(enemies=enemies)

    def _handle_firing(self, state: BeamRiderState, action: int) -> BeamRiderState:
        """Handle both laser and torpedo firing - MIXED: current action mapping with old logic"""

        # TORPEDO FIRING - Check for torpedo-specific actions first
        # Using available actions that we can map to torpedo firing
        # Based on JAXAtari action constants, we'll use:
        # UPFIRE (10), RIGHTFIRE (11), LEFTFIRE (12) for torpedo + movement
        should_fire_torpedo = (action == 10) | (action == 11) | (action == 12)  # UPFIRE, RIGHTFIRE, LEFTFIRE

        # LASER FIRING - Standard firing actions
        should_fire_laser = (action == 1) | (action == 13) | (action == 14) | (action == 15) | (action == 16) | (
                action == 17)  # FIRE and other fire combos

        # Handle torpedo firing first (higher priority)
        state = jax.lax.cond(
            should_fire_torpedo,
            lambda s: self._fire_torpedo(s, True),
            lambda s: s,
            state
        )

        # Handle laser firing if no torpedo fired
        state = jax.lax.cond(
            should_fire_laser & ~should_fire_torpedo,  # Only fire laser if not firing torpedo
            lambda s: self._fire_laser(s, True),
            lambda s: s,
            state
        )

        return state

    def _fire_laser(self, state: BeamRiderState, should_fire: bool) -> BeamRiderState:
        """Fire regular laser projectile - FROM OLD VERSION"""
        projectiles = state.projectiles
        any_active = jnp.any(projectiles[:, 2] == 1)
        can_fire = ~any_active & should_fire  # only fire if none are active

        # New projectile to be fired
        new_projectile = jnp.array([
            state.ship.x + self.constants.SHIP_WIDTH // 2,  # x
            state.ship.y,  # y
            1,  # active
            -self.constants.PROJECTILE_SPEED  # speed
        ])

        # Find first available (inactive) slot
        active_mask = projectiles[:, 2] == 0
        first_inactive = jnp.argmax(active_mask)

        # Conditionally insert new projectile
        projectiles = jnp.where(
            can_fire,
            projectiles.at[first_inactive].set(new_projectile),
            projectiles
        )

        return state.replace(projectiles=projectiles)

    def _fire_torpedo(self, state: BeamRiderState, should_fire: bool) -> BeamRiderState:
        """Fire torpedo projectile (if any remaining) - FROM OLD VERSION"""
        torpedo_projectiles = state.torpedo_projectiles

        # Check if ANY torpedo slot is available
        any_torpedo_active = jnp.any(torpedo_projectiles[:, 2] == 1)
        has_torpedoes = state.torpedoes_remaining > 0

        # Allow firing if no torpedoes are active AND we have torpedoes remaining
        can_fire = ~any_torpedo_active & should_fire & has_torpedoes

        # Find first available (inactive) slot
        active_mask = torpedo_projectiles[:, 2] == 0  # inactive torpedoes
        first_inactive = jnp.argmax(active_mask)

        # Define the new torpedo
        new_torpedo = jnp.array([
            state.ship.x + self.constants.SHIP_WIDTH // 2,  # Center of ship
            state.ship.y,  # Launch from ship's current y
            1,  # Active
            -self.constants.TORPEDO_SPEED  # Upward speed
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

    def _update_projectiles(self, state: BeamRiderState) -> BeamRiderState:
        """Update all projectiles (lasers and torpedoes) - FROM OLD VERSION"""
        projectiles = state.projectiles
        new_y = projectiles[:, 1] + projectiles[:, 3]  # y + speed

        # Deactivate projectiles that go off screen
        active = (
                (projectiles[:, 2] == 1) &
                (new_y > self.constants.TOP_MARGIN) &
                (new_y < self.constants.SCREEN_HEIGHT)
        )

        # Apply updated positions and active status
        projectiles = projectiles.at[:, 1].set(new_y)
        projectiles = projectiles.at[:, 2].set(active.astype(jnp.float32))

        # Update torpedo projectiles
        torpedo_projectiles = state.torpedo_projectiles
        torpedo_new_y = torpedo_projectiles[:, 1] + torpedo_projectiles[:, 3]  # y + speed
        torpedo_active = (torpedo_projectiles[:, 2] == 1) & (torpedo_new_y > 0) & (
                torpedo_new_y < self.constants.SCREEN_HEIGHT)
        torpedo_projectiles = torpedo_projectiles.at[:, 1].set(torpedo_new_y)
        torpedo_projectiles = torpedo_projectiles.at[:, 2].set(torpedo_active.astype(jnp.float32))

        return state.replace(
            projectiles=projectiles,
            torpedo_projectiles=torpedo_projectiles
        )

    def _update_sentinel_projectiles(self, state: BeamRiderState) -> BeamRiderState:
        """Update sentinel ship projectiles - FROM OLD VERSION"""
        sentinel_projectiles = state.sentinel_projectiles

        # Move sentinel projectiles downward
        new_y = sentinel_projectiles[:, 1] + sentinel_projectiles[:, 3]  # y + speed

        # Deactivate projectiles that go off screen
        active = (
                (sentinel_projectiles[:, 2] == 1) &
                (new_y > 0) &
                (new_y < self.constants.SCREEN_HEIGHT)
        )

        sentinel_projectiles = sentinel_projectiles.at[:, 1].set(new_y)
        sentinel_projectiles = sentinel_projectiles.at[:, 2].set(active.astype(jnp.float32))

        return state.replace(sentinel_projectiles=sentinel_projectiles)

    def _spawn_enemies(self, state: BeamRiderState) -> BeamRiderState:
        """Spawn new enemies with dynamic speed scaling - FROM OLD VERSION with yellow rejuvenators"""

        # Check if white saucers are complete for this sector
        white_saucers_complete = state.enemies_killed_this_sector >= self.constants.ENEMIES_PER_SECTOR

        # Check if a sentinel ship is currently active
        sentinel_active = jnp.any(
            (state.enemies[:, 3] == 1) & (state.enemies[:, 5] == self.constants.ENEMY_TYPE_SENTINEL_SHIP)
        )

        state = state.replace(enemy_spawn_timer=state.enemy_spawn_timer + 1)

        # Determine spawning conditions
        normal_enemy_spawn_allowed = ~white_saucers_complete
        blocker_spawn_allowed = white_saucers_complete & sentinel_active

        should_spawn_normal = (state.enemy_spawn_timer >= state.enemy_spawn_interval) & normal_enemy_spawn_allowed

        # FIXED: More balanced spawn rate for green blockers
        early_sector = state.current_sector <= 5

        # More conservative blocker spawn timing - fast enough to challenge sentinel but not spam
        blocker_spawn_interval = jnp.where(
            early_sector,
            jnp.maximum(45, state.enemy_spawn_interval // 2),  # 45-60 frames (0.75-1.0 seconds)
            state.enemy_spawn_interval  # Normal rate for sectors 6+ (60 frames)
        )

        should_spawn_blocker = (state.enemy_spawn_timer >= blocker_spawn_interval) & blocker_spawn_allowed

        should_spawn = should_spawn_normal | should_spawn_blocker

        # Reset spawn timer when spawning occurs
        new_spawn_timer = jnp.where(should_spawn, 0, state.enemy_spawn_timer)
        state = state.replace(enemy_spawn_timer=new_spawn_timer)

        # Find inactive enemy slot
        enemies = state.enemies
        active_mask = enemies[:, 3] == 0

        # Generate random values
        rng_key, subkey1 = random.split(state.rng_key)
        rng_key, subkey2 = random.split(rng_key)
        rng_key, subkey3 = random.split(rng_key)

        # Determine enemy type
        enemy_type = jnp.where(
            should_spawn_blocker,
            self.constants.ENEMY_TYPE_GREEN_BLOCKER,
            self._select_enemy_type_excluding_blockers_early_sectors(state.current_sector, subkey1)
        )

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

        # Calculate player ship center X coordinate (exact position)
        player_ship_center_x = state.ship.x + self.constants.SHIP_WIDTH // 2

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

        # YELLOW REJUVENATORS: Spawn from top on random beam
        is_yellow_rejuvenator = enemy_type == self.constants.ENEMY_TYPE_YELLOW_REJUVENATOR

        # Choose random beam for rejuvenator spawning
        rng_key, rejuv_beam_key = random.split(rng_key)
        rejuv_spawn_beam = random.randint(rejuv_beam_key, (), 0, self.constants.NUM_BEAMS)
        rejuv_spawn_x = self.beam_positions[rejuv_spawn_beam] - self.constants.ENEMY_WIDTH // 2
        rejuv_spawn_y = self.constants.ENEMY_SPAWN_Y

        # Regular enemy spawn (from top, random beam)
        regular_spawn_beam = random.randint(subkey3, (), 0, self.constants.NUM_BEAMS)
        regular_spawn_x = self.beam_positions[regular_spawn_beam] - self.constants.ENEMY_WIDTH // 2
        regular_spawn_y = self.constants.ENEMY_SPAWN_Y

        # Choose final spawn position based on enemy type
        spawn_x = jnp.where(
            is_yellow_chirper,
            chirper_spawn_x,  # Chirpers spawn from sides
            jnp.where(
                is_green_blocker,
                blocker_spawn_x,
                jnp.where(
                    is_yellow_rejuvenator,
                    rejuv_spawn_x,  # Rejuvenators spawn from top
                    regular_spawn_x  # Regular enemies spawn from top
                )
            )
        )

        spawn_y = jnp.where(
            is_yellow_chirper,
            chirper_spawn_y,  # Chirpers use random Y in their range
            jnp.where(
                is_green_blocker,
                blocker_spawn_y,
                jnp.where(
                    is_yellow_rejuvenator,
                    rejuv_spawn_y,  # Rejuvenators spawn from top
                    regular_spawn_y  # Regular enemies spawn from top
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

        # UPDATED: Smooth speed scaling across 99 sectors
        # Linear scaling factor from 1.0 (sector 1) to 2.5 (sector 99)
        speed_scale_factor = 1.0 + ((state.current_sector - 1) / 98.0) * 1.5  # Goes from 1.0 to 2.5
        final_enemy_speed = base_speed * speed_scale_factor

        # Set direction for horizontal moving enemies
        direction_x = jnp.where(
            is_yellow_chirper,
            jnp.where(spawn_from_right, -1.0, 1.0),  # Chirpers move toward center
            jnp.where(
                is_green_blocker,
                jnp.where(blocker_spawn_from_right, -1.0, 1.0),  # Blockers move toward player
                1.0  # Default right movement for bounce craft
            )
        )

        direction_y = 1.0  # All enemies move down by default

        # Set target X for blockers (where player was when spawned)
        target_x = jnp.where(
            is_green_blocker,
            player_ship_center_x,  # Blocker targets player's current position
            jnp.where(
                enemy_type == self.constants.ENEMY_TYPE_ORANGE_TRACKER,
                self.beam_positions[state.ship.beam_position],  # Tracker targets current player beam
                0.0  # Default for others
            )
        )

        # Calculate final spawn beam position for tracking
        final_spawn_beam = jnp.where(
            is_yellow_chirper | is_green_blocker,
            0,  # Side spawners don't use beam positions
            jnp.where(
                enemy_type == self.constants.ENEMY_TYPE_ORANGE_TRACKER,
                state.ship.beam_position,  # Track player's current beam
                regular_spawn_beam  # Use calculated spawn beam
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
            0  # Not applicable for other enemies
        )

        # WHITE SAUCER MOVEMENT PATTERNS - FROM CURRENT VERSION
        is_white_saucer = enemy_type == self.constants.ENEMY_TYPE_WHITE_SAUCER

        # Select movement pattern for white saucers
        movement_pattern = jnp.where(
            is_white_saucer,
            self._select_white_saucer_movement_pattern(subkey2),
            0  # Default pattern for non-white saucers
        )

        # Set initial firing timer for shooting white saucers
        can_shoot = movement_pattern == self.constants.WHITE_SAUCER_SHOOTING
        initial_firing_timer = jnp.where(
            is_white_saucer & can_shoot,
            self.constants.WHITE_SAUCER_FIRING_INTERVAL,
            0
        )

        # Set initial jump timer for jumping white saucers
        can_jump = movement_pattern == self.constants.WHITE_SAUCER_BEAM_JUMP
        initial_jump_timer = jnp.where(
            is_white_saucer & can_jump,
            self.constants.WHITE_SAUCER_JUMP_INTERVAL,
            0
        )

        # Create new enemy array
        new_enemy = jnp.zeros(18)  # 18 columns for all enemy data
        new_enemy = new_enemy.at[0].set(spawn_x)  # x
        new_enemy = new_enemy.at[1].set(spawn_y)  # y
        new_enemy = new_enemy.at[2].set(final_spawn_beam)  # beam_position (or target beam for trackers)
        new_enemy = new_enemy.at[3].set(1)  # active
        new_enemy = new_enemy.at[4].set(final_enemy_speed)  # speed
        new_enemy = new_enemy.at[5].set(enemy_type)  # type
        new_enemy = new_enemy.at[6].set(direction_x)  # direction_x
        new_enemy = new_enemy.at[7].set(direction_y)  # direction_y
        new_enemy = new_enemy.at[8].set(0)  # bounce_count
        new_enemy = new_enemy.at[9].set(0)  # linger_timer
        new_enemy = new_enemy.at[10].set(target_x)  # target_x
        new_enemy = new_enemy.at[11].set(enemy_health)  # health
        new_enemy = new_enemy.at[12].set(0)  # firing_timer
        new_enemy = new_enemy.at[13].set(course_changes_remaining)  # maneuver_timer (course changes for trackers)
        new_enemy = new_enemy.at[14].set(movement_pattern)  # movement_pattern
        new_enemy = new_enemy.at[15].set(initial_firing_timer)  # white_saucer_firing_timer
        new_enemy = new_enemy.at[16].set(initial_jump_timer)  # jump_timer
        new_enemy = new_enemy.at[17].set(0.0)  # zigzag_offset (also used for rejuvenator oscillation)

        # Find first inactive enemy and place new enemy
        first_inactive = jnp.argmax(active_mask)
        can_spawn_enemy = active_mask[first_inactive] & should_spawn

        enemies = jnp.where(
            can_spawn_enemy,
            enemies.at[first_inactive].set(new_enemy),
            enemies
        )

        return state.replace(enemies=enemies, rng_key=rng_key)

    def _select_enemy_type_excluding_blockers_early_sectors(self, sector: int, rng_key: chex.PRNGKey) -> int:
        """Select enemy type - includes yellow rejuvenators - FROM OLD VERSION"""

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

        # Calculate spawn probabilities
        brown_debris_chance = jnp.where(brown_debris_available, self.constants.BROWN_DEBRIS_SPAWN_CHANCE, 0.0)
        yellow_chirper_chance = jnp.where(yellow_chirper_available, self.constants.YELLOW_CHIRPER_SPAWN_CHANCE, 0.0)
        green_blocker_chance = jnp.where(green_blocker_available, self.constants.GREEN_BLOCKER_SPAWN_CHANCE, 0.0)
        green_bounce_chance = jnp.where(green_bounce_available, self.constants.GREEN_BOUNCE_SPAWN_CHANCE, 0.0)
        blue_charger_chance = jnp.where(blue_charger_available, self.constants.BLUE_CHARGER_SPAWN_CHANCE, 0.0)
        orange_tracker_chance = jnp.where(orange_tracker_available, self.constants.ORANGE_TRACKER_SPAWN_CHANCE, 0.0)
        yellow_rejuvenator_chance = jnp.where(yellow_rejuvenator_available,
                                              self.constants.YELLOW_REJUVENATOR_SPAWN_CHANCE, 0.0)

        # Calculate cumulative probabilities
        yellow_rejuvenator_threshold = yellow_rejuvenator_chance
        orange_tracker_threshold = yellow_rejuvenator_threshold + orange_tracker_chance
        blue_charger_threshold = orange_tracker_threshold + blue_charger_chance
        bounce_threshold = blue_charger_threshold + green_bounce_chance
        blocker_threshold = bounce_threshold + green_blocker_chance
        chirper_threshold = blocker_threshold + yellow_chirper_chance
        debris_threshold = chirper_threshold + brown_debris_chance

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
                                    self.constants.ENEMY_TYPE_WHITE_SAUCER
                                )
                            )
                        )
                    )
                )
            )
        )

        return enemy_type

    def _update_enemies(self, state: BeamRiderState) -> BeamRiderState:
        """Update enemy positions - FROM OLD VERSION with current white saucer movement"""

        # FIRST: Handle white saucer movement patterns - FROM CURRENT VERSION
        state = self._update_white_saucer_movement(state)
        enemies = state.enemies  # Get updated enemies array after white saucer movement

        # Handle different movement patterns based on enemy type - FROM OLD VERSION
        enemy_types = enemies[:, 5]  # Get enemy types

        # IMPORTANT: Remove WHITE_SAUCER from regular_enemy_mask since they're handled separately now
        regular_enemy_mask = (enemy_types == self.constants.ENEMY_TYPE_BROWN_DEBRIS)  # REMOVED WHITE_SAUCER
        regular_new_y = enemies[:, 1] + enemies[:, 4]  # y + speed

        # Yellow chirpers move horizontally - FROM OLD VERSION
        chirper_mask = enemy_types == self.constants.ENEMY_TYPE_YELLOW_CHIRPER
        chirper_new_x = enemies[:, 0] + enemies[:, 4]  # x + speed (horizontal movement)

        # Green blockers: complex targeting behavior - FROM OLD VERSION
        blocker_mask = enemy_types == self.constants.ENEMY_TYPE_GREEN_BLOCKER

        # Get blocker current positions and targets
        blocker_x = enemies[:, 0]
        blocker_y = enemies[:, 1]
        blocker_target_x = enemies[:, 10]  # Fixed target_x stored when spawned
        blocker_direction_x = enemies[:, 6]  # movement direction

        # Calculate movement toward fixed target X coordinate
        distance_to_target = jnp.abs(blocker_x - blocker_target_x)
        reached_target = distance_to_target < (self.constants.GREEN_BLOCKER_SPEED * 2)  # Close enough threshold

        # Move horizontally toward fixed target X coordinate
        blocker_new_x = jnp.where(
            blocker_mask & ~reached_target,
            blocker_x + (blocker_direction_x * self.constants.GREEN_BLOCKER_SPEED),
            blocker_x  # Stop moving when reached target
        )

        # Once reached target X coordinate, move down slowly
        blocker_new_y = jnp.where(
            blocker_mask & reached_target,
            blocker_y + (self.constants.GREEN_BLOCKER_SPEED * 0.5),  # Slower downward movement
            blocker_y  # Don't move down until reached target
        )

        # ORANGE TRACKERS: beam following with limited course changes - FROM OLD VERSION
        tracker_mask = enemy_types == self.constants.ENEMY_TYPE_ORANGE_TRACKER

        # Get tracker data
        tracker_x = enemies[:, 0]
        tracker_y = enemies[:, 1]
        tracker_current_target_beam = enemies[:, 2].astype(int)  # Current target beam
        tracker_target_x = enemies[:, 10]  # Target X coordinate for current beam
        tracker_course_changes_remaining = enemies[:, 13].astype(int)  # Course changes left

        # Get current player beam position
        current_player_beam = state.ship.beam_position

        # Check if player changed beams and tracker can still change course
        player_changed_beam = current_player_beam != tracker_current_target_beam
        can_change_course = tracker_course_changes_remaining > 0
        should_change_course = tracker_mask & player_changed_beam & can_change_course

        # Update target beam and target X when changing course
        new_target_beam = jnp.where(
            should_change_course,
            current_player_beam,  # Follow player to new beam
            tracker_current_target_beam  # Keep current target
        )

        new_target_x = jnp.where(
            should_change_course,
            self.beam_positions[current_player_beam],  # New beam X position
            tracker_target_x  # Keep current target X
        )

        # Decrease course changes remaining when used
        new_course_changes_remaining = jnp.where(
            should_change_course,
            tracker_course_changes_remaining - 1,
            tracker_course_changes_remaining
        )

        # Calculate movement toward target beam
        distance_to_target_x = jnp.abs(tracker_x - new_target_x)
        reached_target_beam = distance_to_target_x < (self.constants.ORANGE_TRACKER_SPEED * 2)

        # Horizontal movement toward target beam
        horizontal_direction = jnp.sign(new_target_x - tracker_x)
        tracker_new_x = jnp.where(
            tracker_mask & ~reached_target_beam,
            tracker_x + (horizontal_direction * self.constants.ORANGE_TRACKER_SPEED),
            tracker_x  # Stop horizontal movement when aligned
        )

        # Vertical movement (always moving down, but faster when aligned)
        vertical_speed = jnp.where(
            tracker_mask & reached_target_beam,
            self.constants.ORANGE_TRACKER_SPEED * 1.5,  # Faster when aligned with beam
            self.constants.ORANGE_TRACKER_SPEED * 0.5  # Slower when moving to beam
        )

        tracker_new_y = jnp.where(
            tracker_mask,
            tracker_y + vertical_speed,
            tracker_y
        )

        # Check if tracker has reached bottom
        tracker_at_bottom = tracker_new_y >= (self.constants.SCREEN_HEIGHT - self.constants.ENEMY_HEIGHT)

        # Don't move if at bottom
        tracker_new_x = jnp.where(tracker_mask & tracker_at_bottom, tracker_x, tracker_new_x)
        tracker_new_y = jnp.where(tracker_mask & tracker_at_bottom, tracker_y, tracker_new_y)

        # BLUE CHARGERS: WORKING VERSION - SIMPLE, DIRECT LOGIC - FROM OLD VERSION
        charger_mask = enemy_types == self.constants.ENEMY_TYPE_BLUE_CHARGER
        charger_linger_timer = enemies[:, 9].astype(int)  # linger_timer column

        # Define bottom position where chargers should stop
        bottom_position = self.constants.SCREEN_HEIGHT - self.constants.ENEMY_HEIGHT - 10

        # Check if charger has reached or passed bottom position
        charger_reached_bottom = enemies[:, 1] >= bottom_position

        # WORKING VERSION: Simple Y movement - just apply the speed directly
        # If speed is positive, move down. If speed is negative (deflected), move up.
        charger_new_y = jnp.where(
            charger_mask & charger_reached_bottom,
            bottom_position,  # Stay exactly at bottom position when reached
            enemies[:, 1] + enemies[:, 4]  # Normal movement: current Y + speed (can be + or -)
        )

        # WORKING VERSION: Simple linger timer logic (only affects normal movement)
        new_linger_timer = jnp.where(
            charger_mask & charger_reached_bottom & (charger_linger_timer == 0),
            self.constants.BLUE_CHARGER_LINGER_TIME,  # Start lingering when first reaching bottom
            jnp.where(
                charger_mask & charger_reached_bottom & (charger_linger_timer > 0),
                charger_linger_timer - 1,  # Count down while at bottom
                charger_linger_timer  # Keep current value for others
            )
        )

        # GREEN BOUNCE CRAFT: bouncing behavior - FROM OLD VERSION
        bounce_mask = enemy_types == self.constants.ENEMY_TYPE_GREEN_BOUNCE

        # Get current bounce directions
        bounce_dir_x = enemies[:, 6]  # direction_x
        bounce_dir_y = enemies[:, 7]  # direction_y

        # Calculate new positions based on current direction and speed
        bounce_new_x = enemies[:, 0] + (bounce_dir_x * enemies[:, 4])
        bounce_new_y = enemies[:, 1] + (bounce_dir_y * enemies[:, 4])

        # Check for bouncing off screen edges
        hit_left_edge = bounce_new_x <= 0
        hit_right_edge = bounce_new_x >= (self.constants.SCREEN_WIDTH - self.constants.ENEMY_WIDTH)
        hit_top_edge = bounce_new_y <= self.constants.TOP_MARGIN
        hit_bottom_edge = bounce_new_y >= (self.constants.SCREEN_HEIGHT - self.constants.ENEMY_HEIGHT)

        # Check if any bounce occurred
        any_bounce = hit_left_edge | hit_right_edge | hit_top_edge | hit_bottom_edge

        # Decrement bounce count when bouncing occurs
        current_bounce_count = enemies[:, 8]  # bounce_count column
        new_bounce_count = jnp.where(
            bounce_mask & any_bounce,
            jnp.maximum(0, current_bounce_count - 1),  # Decrement, but don't go below 0
            current_bounce_count
        )

        # Reverse directions when hitting edges (only if still have bounces left)
        can_bounce = new_bounce_count > 0

        new_bounce_dir_x = jnp.where(
            bounce_mask & (hit_left_edge | hit_right_edge) & can_bounce,
            -bounce_dir_x,
            bounce_dir_x
        )

        new_bounce_dir_y = jnp.where(
            bounce_mask & (hit_top_edge | hit_bottom_edge) & can_bounce,
            -bounce_dir_y,
            bounce_dir_y
        )

        # Clamp bounce positions to screen bounds
        bounce_clamped_x = jnp.clip(bounce_new_x, 0, self.constants.SCREEN_WIDTH - self.constants.ENEMY_WIDTH)
        bounce_clamped_y = jnp.clip(bounce_new_y, self.constants.TOP_MARGIN,
                                    self.constants.SCREEN_HEIGHT - self.constants.ENEMY_HEIGHT)

        # SENTINEL SHIP: simple horizontal cruise across top - FROM OLD VERSION
        sentinel_mask = enemy_types == self.constants.ENEMY_TYPE_SENTINEL_SHIP
        sentinel_new_x = enemies[:, 0] + enemies[:, 4]  # Move horizontally at constant speed
        sentinel_new_y = enemies[:, 1]  # Stay at same Y level

        # YELLOW REJUVENATORS: oscillating float movement - FROM OLD VERSION
        rejuvenator_mask = enemy_types == self.constants.ENEMY_TYPE_YELLOW_REJUVENATOR

        # Update oscillation offset
        rejuv_zigzag_offset = enemies[:, 17]
        new_rejuv_zigzag_offset = jnp.where(
            rejuvenator_mask,
            rejuv_zigzag_offset + self.constants.YELLOW_REJUVENATOR_OSCILLATION_FREQUENCY,
            rejuv_zigzag_offset
        )

        # Calculate horizontal oscillation
        rejuv_beam_center_x = self.beam_positions[enemies[:, 2].astype(int)]
        rejuv_oscillation = jnp.sin(new_rejuv_zigzag_offset) * self.constants.YELLOW_REJUVENATOR_OSCILLATION_AMPLITUDE

        rejuvenator_new_x = jnp.where(
            rejuvenator_mask,
            jnp.clip(
                rejuv_beam_center_x + rejuv_oscillation - self.constants.ENEMY_WIDTH // 2,
                0, self.constants.SCREEN_WIDTH - self.constants.ENEMY_WIDTH
            ),
            enemies[:, 0]
        )

        rejuvenator_new_y = jnp.where(
            rejuvenator_mask,
            enemies[:, 1] + enemies[:, 4],  # Move down slowly
            enemies[:, 1]
        )

        # REJUVENATOR DEBRIS: explosive debris movement - FROM OLD VERSION
        debris_mask = enemy_types == self.constants.ENEMY_TYPE_REJUVENATOR_DEBRIS

        # Move in direction with speed
        debris_new_x = jnp.where(
            debris_mask,
            enemies[:, 0] + (enemies[:, 6] * enemies[:, 4]),  # direction_x * speed
            enemies[:, 0]
        )

        debris_new_y = jnp.where(
            debris_mask,
            enemies[:, 1] + (enemies[:, 7] * enemies[:, 4]),  # direction_y * speed
            enemies[:, 1]
        )

        # Decrease lifetime for debris
        debris_lifetime = enemies[:, 9].astype(int)  # Using linger_timer as lifetime
        new_debris_lifetime = jnp.where(
            debris_mask,
            jnp.maximum(0, debris_lifetime - 1),
            debris_lifetime
        )

        # Update X positions based on enemy type (WHITE SAUCERS EXCLUDED - handled separately)
        new_x = jnp.where(
            chirper_mask,
            chirper_new_x,  # Chirpers move horizontally
            jnp.where(
                blocker_mask,
                blocker_new_x,  # Blockers use fixed X-coordinate targeting
                jnp.where(
                    bounce_mask,
                    bounce_clamped_x,  # Bounce craft bounce around
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
                                    rejuvenator_mask,
                                    rejuvenator_new_x,  # Rejuvenators oscillate
                                    jnp.where(
                                        debris_mask,
                                        debris_new_x,  # Debris moves in direction
                                        enemies[:, 0]  # Default: no X change (includes white saucers)
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )

        # Update Y positions based on enemy type (WHITE SAUCERS EXCLUDED - handled separately)
        new_y = jnp.where(
            regular_enemy_mask & ~charger_mask & ~tracker_mask,  # Regular enemies (only brown debris now)
            regular_new_y,  # Regular enemies move down
            jnp.where(
                blocker_mask,
                blocker_new_y,  # Blockers use fixed X-coordinate targeting Y movement
                jnp.where(
                    bounce_mask,
                    bounce_clamped_y,  # Bounce craft bounce around
                    jnp.where(
                        charger_mask,
                        charger_new_y,  # Blue chargers use WORKING simple logic
                        jnp.where(
                            tracker_mask,
                            tracker_new_y,  # Orange trackers use beam following Y movement
                            jnp.where(
                                sentinel_mask,
                                sentinel_new_y,  # Sentinels stay at same Y
                                jnp.where(
                                    rejuvenator_mask,
                                    rejuvenator_new_y,  # Rejuvenators float down
                                    jnp.where(
                                        debris_mask,
                                        debris_new_y,  # Debris moves in direction
                                        enemies[:, 1]  # Default: no Y change (includes white saucers AND chirpers)
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )

        # Deactivate enemies that go off screen - FROM OLD VERSION
        # Regular enemies: deactivate when they go below screen (only brown debris now)
        regular_active = (enemies[:, 3] == 1) & (regular_new_y < self.constants.SCREEN_HEIGHT)

        # Orange trackers: deactivate when they reach bottom of screen
        tracker_active = (enemies[:, 3] == 1) & ~tracker_at_bottom

        # Chirpers: deactivate when they go off either side
        chirper_active = (enemies[:, 3] == 1) & (chirper_new_x > -self.constants.ENEMY_WIDTH) & (
                chirper_new_x < self.constants.SCREEN_WIDTH + self.constants.ENEMY_WIDTH)

        # Green blockers: deactivate when they go off any edge OR reach bottom
        blocker_active = (enemies[:, 3] == 1) & \
                         (blocker_new_x > -self.constants.ENEMY_WIDTH) & \
                         (blocker_new_x < self.constants.SCREEN_WIDTH + self.constants.ENEMY_WIDTH) & \
                         (blocker_new_y > -self.constants.ENEMY_HEIGHT) & \
                         (blocker_new_y < self.constants.SCREEN_HEIGHT)

        # Bounce craft: stay active as long as they have bounces remaining
        bounce_active = (enemies[:, 3] == 1) & (new_bounce_count > 0)

        # WORKING VERSION: Blue chargers - simple active logic
        # Stay active until they reach bottom AND linger timer expires, OR until they go off top when deflected
        charger_off_top = charger_new_y < -self.constants.ENEMY_HEIGHT  # Deflected chargers going off top
        charger_active = (enemies[:, 3] == 1) & ~charger_off_top & (
                (~charger_reached_bottom) |  # Still moving down, stay active
                (charger_reached_bottom & (new_linger_timer > 0))  # At bottom but timer not expired
        )

        # Sentinel ships: deactivate when they go completely off screen
        sentinel_off_screen = sentinel_new_x > (self.constants.SCREEN_WIDTH + self.constants.SENTINEL_SHIP_WIDTH)
        sentinel_active = (enemies[:, 3] == 1) & ~sentinel_off_screen

        # Rejuvenators: deactivate when they go off bottom
        rejuvenator_active = (enemies[:, 3] == 1) & (rejuvenator_new_y < self.constants.SCREEN_HEIGHT)

        # Debris: deactivate when lifetime expires or goes off screen
        debris_active = (enemies[:, 3] == 1) & (new_debris_lifetime > 0) & \
                        (debris_new_x > -self.constants.ENEMY_WIDTH) & \
                        (debris_new_x < self.constants.SCREEN_WIDTH + self.constants.ENEMY_WIDTH) & \
                        (debris_new_y > -self.constants.ENEMY_HEIGHT) & \
                        (debris_new_y < self.constants.SCREEN_HEIGHT + self.constants.ENEMY_HEIGHT)

        # Combine active states based on enemy type (WHITE SAUCERS EXCLUDED - handled separately)
        active = jnp.where(
            regular_enemy_mask & ~charger_mask & ~tracker_mask,
            regular_active,
            jnp.where(
                chirper_mask,
                chirper_active,
                jnp.where(
                    blocker_mask,
                    blocker_active,  # Use blocker-specific active logic
                    jnp.where(
                        bounce_mask,
                        bounce_active,
                        jnp.where(
                            charger_mask,
                            charger_active,  # WORKING VERSION: simple charger active logic
                            jnp.where(
                                tracker_mask,
                                tracker_active,  # Use tracker-specific active logic
                                jnp.where(
                                    sentinel_mask,
                                    sentinel_active,
                                    jnp.where(
                                        rejuvenator_mask,
                                        rejuvenator_active,
                                        jnp.where(
                                            debris_mask,
                                            debris_active,
                                            enemies[:, 3]  # Default: keep current active state (includes white saucers)
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
        enemies = enemies.at[:, 2].set(  # Update target beam for trackers
            jnp.where(tracker_mask, new_target_beam, enemies[:, 2])
        )
        enemies = enemies.at[:, 3].set(active.astype(jnp.float32))  # Update active states

        # Update direction arrays - prioritize bounce craft, keep blocker directions
        enemies = enemies.at[:, 6].set(
            jnp.where(
                bounce_mask,
                new_bounce_dir_x,
                enemies[:, 6]  # Keep existing direction_x for blockers and others
            )
        )
        enemies = enemies.at[:, 7].set(
            jnp.where(
                bounce_mask,
                new_bounce_dir_y,
                enemies[:, 7]  # Keep existing direction_y for others
            )
        )

        enemies = enemies.at[:, 8].set(new_bounce_count)  # Update bounce count
        enemies = enemies.at[:, 9].set(
            jnp.where(
                charger_mask,
                new_linger_timer,  # Update linger timer for chargers
                jnp.where(
                    debris_mask,
                    new_debris_lifetime,  # Update lifetime for debris
                    enemies[:, 9]  # Keep existing values for others
                )
            )
        )

        # Update target X for trackers and blockers
        enemies = enemies.at[:, 10].set(
            jnp.where(
                tracker_mask,
                new_target_x,  # Update target X for trackers
                enemies[:, 10]  # Keep target_x for blockers, existing values for others
            )
        )

        # Update course changes remaining for trackers
        enemies = enemies.at[:, 13].set(
            jnp.where(
                tracker_mask,
                new_course_changes_remaining,  # Update course changes for trackers
                enemies[:, 13]  # Keep existing values for others
            )
        )

        # Update zigzag offset for rejuvenators
        enemies = enemies.at[:, 17].set(
            jnp.where(
                rejuvenator_mask,
                new_rejuv_zigzag_offset,  # Update oscillation for rejuvenators
                enemies[:, 17]  # Keep existing values for others
            )
        )

        return state.replace(enemies=enemies)

    def _check_collisions(self, state: BeamRiderState) -> BeamRiderState:
        """Check for collisions between projectiles and enemies - FROM OLD VERSION"""
        projectiles = state.projectiles
        torpedo_projectiles = state.torpedo_projectiles
        sentinel_projectiles = state.sentinel_projectiles
        enemies = state.enemies
        score = state.score

        # Vectorized collision detection for LASER projectiles vs enemies
        proj_active = projectiles[:, 2] == 1
        enemy_active = enemies[:, 3] == 1

        # Enemies that take damage from lasers (can be destroyed)
        enemy_vulnerable_to_lasers = (
                (enemies[:, 5] == self.constants.ENEMY_TYPE_WHITE_SAUCER) |
                (enemies[:, 5] == self.constants.ENEMY_TYPE_YELLOW_CHIRPER) |
                (enemies[:, 5] == self.constants.ENEMY_TYPE_BLUE_CHARGER)
        )

        # Enemies that BLOCK lasers (destroy the laser but don't take damage)
        enemy_blocks_lasers = (
                (enemies[:, 5] == self.constants.ENEMY_TYPE_BROWN_DEBRIS) |
                (enemies[:, 5] == self.constants.ENEMY_TYPE_GREEN_BLOCKER) |
                (enemies[:, 5] == self.constants.ENEMY_TYPE_SENTINEL_SHIP)
        )

        # Enemies that lasers can interact with (either damage or block)
        enemy_interacts_with_lasers = enemy_vulnerable_to_lasers | enemy_blocks_lasers

        # Broadcast projectile and enemy positions for vectorized collision check
        proj_x = projectiles[:, 0:1]
        proj_y = projectiles[:, 1:2]
        enemy_x = enemies[:, 0:1].T
        enemy_y = enemies[:, 1:2].T

        # Get enemy dimensions (sentinel ships are larger)
        enemy_width = jnp.where(
            enemies[:, 5] == self.constants.ENEMY_TYPE_SENTINEL_SHIP,
            self.constants.SENTINEL_SHIP_WIDTH,
            self.constants.ENEMY_WIDTH
        )
        enemy_height = jnp.where(
            enemies[:, 5] == self.constants.ENEMY_TYPE_SENTINEL_SHIP,
            self.constants.SENTINEL_SHIP_HEIGHT,
            self.constants.ENEMY_HEIGHT
        )

        # Vectorized bounding box collision check for lasers (ANY interaction)
        laser_collisions = (
                (proj_x < enemy_x + enemy_width[None, :]) &
                (proj_x + self.constants.PROJECTILE_WIDTH > enemy_x) &
                (proj_y < enemy_y + enemy_height[None, :]) &
                (proj_y + self.constants.PROJECTILE_HEIGHT > enemy_y) &
                proj_active[:, None] &
                enemy_active[None, :] &
                enemy_interacts_with_lasers[None, :]
        )

        # Lasers get destroyed when hitting ANY enemy they interact with
        laser_proj_hits = jnp.any(laser_collisions, axis=1)

        # Only vulnerable enemies take damage from lasers
        laser_damage_collisions = laser_collisions & enemy_vulnerable_to_lasers[None, :]
        laser_enemy_hits = jnp.any(laser_damage_collisions, axis=0)

        # WORKING VERSION: Simple blue charger deflection - ONLY set speed
        charger_laser_hits = laser_enemy_hits & (enemies[:, 5] == self.constants.ENEMY_TYPE_BLUE_CHARGER)
        enemies = enemies.at[:, 4].set(
            jnp.where(
                charger_laser_hits,
                self.constants.BLUE_CHARGER_DEFLECT_SPEED,  # -2.0
                enemies[:, 4]
            )
        )

        # Don't deactivate blue chargers when hit by lasers (they just get deflected)
        laser_enemy_hits = laser_enemy_hits & (enemies[:, 5] != self.constants.ENEMY_TYPE_BLUE_CHARGER)

        # Handle rejuvenator collision with ship (life bonus)
        ship_x, ship_y = state.ship.x, state.ship.y
        rejuvenator_ship_collisions = (
                (ship_x < enemies[:, 0] + self.constants.ENEMY_WIDTH) &
                (ship_x + self.constants.SHIP_WIDTH > enemies[:, 0]) &
                (ship_y < enemies[:, 1] + self.constants.ENEMY_HEIGHT) &
                (ship_y + self.constants.SHIP_HEIGHT > enemies[:, 1]) &
                enemy_active &
                (enemies[:, 5] == self.constants.ENEMY_TYPE_YELLOW_REJUVENATOR)
        )

        rejuvenator_collected = jnp.any(rejuvenator_ship_collisions)
        # Add life bonus when rejuvenator is collected
        lives = jnp.where(rejuvenator_collected, state.lives + self.constants.YELLOW_REJUVENATOR_LIFE_BONUS,
                          state.lives)

        # Deactivate collected rejuvenators
        enemies = enemies.at[:, 3].set(enemies[:, 3] * (~rejuvenator_ship_collisions))

        # Handle rejuvenator being shot (spawn debris)
        rejuvenator_shot = laser_enemy_hits & (enemies[:, 5] == self.constants.ENEMY_TYPE_YELLOW_REJUVENATOR)
        any_rejuvenator_shot = jnp.any(rejuvenator_shot)

        # Spawn debris when rejuvenator is shot
        state = jax.lax.cond(
            any_rejuvenator_shot,
            lambda s: self._spawn_rejuvenator_debris(s, rejuvenator_shot, enemies),
            lambda s: s,
            state.replace(lives=lives)  # Update lives first
        )

        # Update enemies array in state
        enemies = state.enemies

        # Vectorized collision detection for TORPEDO projectiles vs enemies
        torpedo_active = torpedo_projectiles[:, 2] == 1
        torpedo_x = torpedo_projectiles[:, 0:1]
        torpedo_y = torpedo_projectiles[:, 1:2]

        # Torpedoes can hit all enemy types
        torpedo_collisions = (
                (torpedo_x < enemy_x + enemy_width[None, :]) &
                (torpedo_x + self.constants.TORPEDO_WIDTH > enemy_x) &
                (torpedo_y < enemy_y + enemy_height[None, :]) &
                (torpedo_y + self.constants.TORPEDO_HEIGHT > enemy_y) &
                torpedo_active[:, None] &
                enemy_active[None, :]
        )

        # Find collisions for torpedo projectiles
        torpedo_proj_hits = jnp.any(torpedo_collisions, axis=1)
        torpedo_enemy_hits = jnp.any(torpedo_collisions, axis=0)

        # Handle sentinel ship health reduction
        sentinel_torpedo_hits = torpedo_enemy_hits & (enemies[:, 5] == self.constants.ENEMY_TYPE_SENTINEL_SHIP)

        # Reduce sentinel health when hit by torpedo
        enemies = enemies.at[:, 11].set(  # health column
            jnp.where(
                sentinel_torpedo_hits,
                jnp.maximum(0, enemies[:, 11] - 1),  # Reduce health by 1, minimum 0
                enemies[:, 11]
            )
        )

        # Only destroy sentinels when health reaches 0
        sentinel_destroyed = sentinel_torpedo_hits & (enemies[:, 11] <= 1)  # Will be 0 after reduction

        # Update torpedo hits to only include destroyed sentinels
        torpedo_enemy_hits = jnp.where(
            enemies[:, 5] == self.constants.ENEMY_TYPE_SENTINEL_SHIP,
            sentinel_destroyed,  # Only destroy if health will reach 0
            torpedo_enemy_hits  # Normal destruction for other enemies
        )

        # Combine enemy hits from laser and torpedo
        total_enemy_hits = laser_enemy_hits | torpedo_enemy_hits

        # Count only WHITE SAUCER kills for sector progression
        white_saucer_hits = total_enemy_hits & (enemies[:, 5] == self.constants.ENEMY_TYPE_WHITE_SAUCER)
        enemies_killed_this_frame = jnp.sum(white_saucer_hits)

        # Calculate score with different point values - FROM OLD VERSION
        laser_score = (
                jnp.sum(laser_enemy_hits & (enemies[:,
                                            5] == self.constants.ENEMY_TYPE_WHITE_SAUCER)) * self.constants.POINTS_PER_ENEMY +
                jnp.sum(laser_enemy_hits & (enemies[:,
                                            5] == self.constants.ENEMY_TYPE_YELLOW_CHIRPER)) * self.constants.YELLOW_CHIRPER_POINTS
        )

        torpedo_score = (
                jnp.sum(torpedo_enemy_hits & (enemies[:,
                                              5] == self.constants.ENEMY_TYPE_WHITE_SAUCER)) * self.constants.POINTS_PER_ENEMY * 2 +
                jnp.sum(torpedo_enemy_hits & (enemies[:,
                                              5] == self.constants.ENEMY_TYPE_BROWN_DEBRIS)) * self.constants.BROWN_DEBRIS_POINTS +
                jnp.sum(torpedo_enemy_hits & (enemies[:,
                                              5] == self.constants.ENEMY_TYPE_YELLOW_CHIRPER)) * self.constants.YELLOW_CHIRPER_POINTS +
                jnp.sum(torpedo_enemy_hits & (enemies[:,
                                              5] == self.constants.ENEMY_TYPE_GREEN_BLOCKER)) * self.constants.GREEN_BLOCKER_POINTS +
                jnp.sum(torpedo_enemy_hits & (enemies[:,
                                              5] == self.constants.ENEMY_TYPE_GREEN_BOUNCE)) * self.constants.GREEN_BOUNCE_POINTS +
                jnp.sum(torpedo_enemy_hits & (enemies[:,
                                              5] == self.constants.ENEMY_TYPE_BLUE_CHARGER)) * self.constants.BLUE_CHARGER_POINTS +
                jnp.sum(torpedo_enemy_hits & (enemies[:,
                                              5] == self.constants.ENEMY_TYPE_ORANGE_TRACKER)) * self.constants.ORANGE_TRACKER_POINTS +
                jnp.sum(torpedo_enemy_hits & (enemies[:,
                                              5] == self.constants.ENEMY_TYPE_SENTINEL_SHIP)) * self.constants.SENTINEL_SHIP_POINTS
        )

        score += laser_score + torpedo_score

        # Check sentinel projectile vs player collisions
        sentinel_proj_active = sentinel_projectiles[:, 2] == 1

        sentinel_proj_ship_collisions = (
                (ship_x < sentinel_projectiles[:, 0] + self.constants.PROJECTILE_WIDTH) &
                (ship_x + self.constants.SHIP_WIDTH > sentinel_projectiles[:, 0]) &
                (ship_y < sentinel_projectiles[:, 1] + self.constants.PROJECTILE_HEIGHT) &
                (ship_y + self.constants.SHIP_HEIGHT > sentinel_projectiles[:, 1]) &
                sentinel_proj_active
        )

        sentinel_hit_ship = jnp.any(sentinel_proj_ship_collisions)

        # Deactivate sentinel projectiles that hit ship
        sentinel_projectiles = sentinel_projectiles.at[:, 2].set(
            sentinel_projectiles[:, 2] * (~sentinel_proj_ship_collisions)
        )

        # Check regular enemy-ship collisions (exclude chirpers, sentinels, and rejuvenators)
        can_collide_with_ship = (
                (enemies[:, 5] != self.constants.ENEMY_TYPE_YELLOW_CHIRPER) &
                (enemies[:, 5] != self.constants.ENEMY_TYPE_SENTINEL_SHIP) &
                (enemies[:, 5] != self.constants.ENEMY_TYPE_YELLOW_REJUVENATOR)  # Rejuvenators give life, don't hurt
        )

        ship_collisions = (
                (ship_x < enemies[:, 0] + enemy_width) &
                (ship_x + self.constants.SHIP_WIDTH > enemies[:, 0]) &
                (ship_y < enemies[:, 1] + enemy_height) &
                (ship_y + self.constants.SHIP_HEIGHT > enemies[:, 1]) &
                enemy_active &
                can_collide_with_ship
        )

        regular_ship_collision = jnp.any(ship_collisions)

        # Combine all ship collisions
        any_ship_collision = regular_ship_collision | sentinel_hit_ship

        # Handle ship collision
        lives = jnp.where(any_ship_collision, state.lives - 1, lives)  # Use updated lives

        # Reset ship position on collision
        center_beam = self.constants.INITIAL_BEAM
        new_ship_x = jnp.where(
            any_ship_collision,
            self.beam_positions[center_beam] - self.constants.SHIP_WIDTH // 2,
            state.ship.x
        )
        new_ship_beam = jnp.where(
            any_ship_collision,
            center_beam,
            state.ship.beam_position
        )

        ship = state.ship.replace(x=new_ship_x, beam_position=new_ship_beam)

        # Update projectile and enemy states
        projectiles = projectiles.at[:, 2].set(
            projectiles[:, 2] * (~laser_proj_hits))  # Lasers destroyed by ANY collision
        torpedo_projectiles = torpedo_projectiles.at[:, 2].set(torpedo_projectiles[:, 2] * (~torpedo_proj_hits))
        enemies = enemies.at[:, 3].set(enemies[:, 3] * (~total_enemy_hits))  # Only enemies that should be destroyed
        enemies = enemies.at[:, 3].set(enemies[:, 3] * (~ship_collisions))  # Deactivate enemies that hit ship

        return state.replace(
            projectiles=projectiles,
            torpedo_projectiles=torpedo_projectiles,
            sentinel_projectiles=sentinel_projectiles,
            enemies=enemies,
            score=score,
            ship=ship,
            lives=lives,
            enemies_killed_this_sector=state.enemies_killed_this_sector + enemies_killed_this_frame
        )

    def _spawn_rejuvenator_debris(self, state: BeamRiderState, rejuvenator_hit_mask: chex.Array,
                                  enemies: chex.Array) -> BeamRiderState:
        """Spawn explosive debris when rejuvenator is shot - FROM OLD VERSION"""

        # Find rejuvenators that were hit
        def spawn_debris_for_rejuvenator(i, state_enemies):
            state_inner, enemies_inner = state_enemies

            # Check if this rejuvenator was hit
            rejuvenator_hit = rejuvenator_hit_mask[i]

            # Get rejuvenator position
            rejuv_x = enemies[i, 0]
            rejuv_y = enemies[i, 1]

            # Spawn 4 debris pieces in different directions
            directions = jnp.array([
                [-1.0, -0.5],  # Up-left
                [1.0, -0.5],  # Up-right
                [-0.5, 1.0],  # Down-left
                [0.5, 1.0]  # Down-right
            ])

            def spawn_single_debris(debris_idx, enemies_state):
                # Find first available slot
                first_inactive = jnp.argmax(enemies_state[:, 3] == 0)
                can_spawn = (enemies_state[first_inactive, 3] == 0) & rejuvenator_hit & (debris_idx < 4)

                # Create debris enemy
                direction_x = directions[debris_idx, 0]
                direction_y = directions[debris_idx, 1]

                # Add some randomness to debris position
                debris_x = rejuv_x + (debris_idx - 2) * 4  # Spread debris out
                debris_y = rejuv_y

                new_debris = jnp.array([
                    debris_x,  # x
                    debris_y,  # y
                    0,  # beam_position (not used)
                    1,  # active
                    self.constants.REJUVENATOR_DEBRIS_SPEED,  # speed
                    self.constants.ENEMY_TYPE_REJUVENATOR_DEBRIS,  # type
                    direction_x,  # direction_x
                    direction_y,  # direction_y
                    0,  # bounce_count
                    self.constants.REJUVENATOR_DEBRIS_LIFETIME,  # linger_timer (used as lifetime)
                    0,  # target_x
                    1,  # health
                    0,  # firing_timer
                    0,  # maneuver_timer
                    0,  # movement_pattern
                    0,  # white_saucer_firing_timer
                    0,  # jump_timer
                    0.0,  # zigzag_offset
                ])

                enemies_state = jnp.where(
                    can_spawn,
                    enemies_state.at[first_inactive].set(new_debris),
                    enemies_state
                )

                return enemies_state

            # Spawn all 4 debris pieces
            enemies_inner = jax.lax.fori_loop(0, 4, spawn_single_debris, enemies_inner)

            return (state_inner, enemies_inner)

        # Apply debris spawning for all enemies
        state, enemies = jax.lax.fori_loop(0, self.constants.MAX_ENEMIES, spawn_debris_for_rejuvenator,
                                           (state, enemies))

        return state.replace(enemies=enemies)

    def _check_sector_progression(self, state: BeamRiderState) -> BeamRiderState:
        """Check if sector is complete and advance to next sector - FROM OLD VERSION"""

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

        # Spawn sentinel if needed
        state = jax.lax.cond(
            should_spawn_sentinel,
            lambda s: self._spawn_sentinel(s).replace(sentinel_spawned_this_sector=True),  # Mark as spawned
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
            beam_position=new_ship_beam
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

    def _check_game_over(self, state: BeamRiderState) -> BeamRiderState:
        """Check if game is over - FROM OLD VERSION"""
        # Original game over condition: no lives left
        lives_game_over = state.lives <= 0

        # New game over condition: reached sector 99 limit
        sector_limit_reached = state.current_sector > 99

        # Game is over if either condition is met
        game_over = lives_game_over | sector_limit_reached

        return state.replace(game_over=game_over)

    def _spawn_sentinel(self, state: BeamRiderState) -> BeamRiderState:
        """Spawn the sector sentinel ship - FROM OLD VERSION"""
        enemies = state.enemies

        # Find first inactive enemy slot
        active_mask = enemies[:, 3] == 0
        first_inactive = jnp.argmax(active_mask)
        can_spawn = active_mask[first_inactive]

        # Sentinel always spawns from LEFT side and moves RIGHT
        sentinel_spawn_x = -self.constants.SENTINEL_SHIP_WIDTH  # Start off-screen left
        sentinel_spawn_y = self.constants.TOP_MARGIN + 10
        sentinel_direction_x = 1.0  # Always move right

        new_sentinel = jnp.array([
            sentinel_spawn_x,  # 0: x
            sentinel_spawn_y,  # 1: y
            0,  # 2: beam_position
            1,  # 3: active
            self.constants.SENTINEL_SHIP_SPEED,  # 4: speed
            self.constants.ENEMY_TYPE_SENTINEL_SHIP,  # 5: type
            sentinel_direction_x,  # 6: direction_x (always 1.0 = right)
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
            0.0,  # 17: zigzag_offset
        ])

        enemies = jnp.where(
            can_spawn,
            enemies.at[first_inactive].set(new_sentinel),
            enemies
        )

        return state.replace(enemies=enemies)


class BeamRiderRenderer(JAXGameRenderer):
    """Unified renderer for BeamRider game - FROM OLD VERSION"""

    def __init__(self, scale=3, enable_pygame=False):
        super().__init__()
        self.constants = BeamRiderConstants()
        self.screen_width = self.constants.SCREEN_WIDTH
        self.screen_height = self.constants.SCREEN_HEIGHT
        self.beam_positions = self.constants.get_beam_positions()

        # SET enable_pygame FIRST before calling other methods
        self.enable_pygame = enable_pygame

        self.white_saucer_sprite = jnp.array([
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 0, 1, 0, 1],
            [1, 1, 1, 1, 1]
        ], dtype=jnp.uint8)

        # JAX rendering components
        self.ship_sprite_surface = self._create_ship_surface()
        self.small_ship_surface = self._create_small_ship_surface()

        # JIT-compile the render function
        self.render = jit(self._render_impl)

        # Pygame components (optional)
        if enable_pygame:
            pygame.init()
            self.scale = scale
            self.pygame_screen_width = self.screen_width * scale
            self.pygame_screen_height = self.screen_height * scale
            self.pygame_screen = pygame.display.set_mode((self.pygame_screen_width, self.pygame_screen_height))
            pygame.display.set_caption("BeamRider - JAX Implementation")
            self.clock = pygame.time.Clock()
            import os
            font_path = os.path.join(os.path.dirname(__file__), "../../../assets/PressStart2P.ttf")
            try:
                self.font = pygame.font.Font(font_path, 16)
            except:
                self.font = pygame.font.Font(None, 24)  # Fallback
    def _create_ship_surface(self):
        """Create the main ship sprite surface using a pixel array and color map."""

        if not self.enable_pygame:
            return None

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
        if not self.enable_pygame or self.ship_sprite_surface is None:
            return None
        small_sprite = pygame.transform.scale(self.ship_sprite_surface, (16, 10))
        return small_sprite

    def _render_impl(self, state: BeamRiderState) -> chex.Array:
        """Render the current game state to a screen buffer - FROM OLD VERSION"""
        # Create screen buffer (RGB)
        screen = jnp.zeros((self.constants.SCREEN_HEIGHT, self.constants.SCREEN_WIDTH, 3), dtype=jnp.uint8)

        # Render 3D dotted tunnel grid
        screen = self._draw_3d_grid(screen, state.frame_count)

        # ADD THIS LINE - Render the ship
        screen = self._draw_ship(screen, state.ship)

        # Render projectiles (lasers)
        screen = self._draw_projectiles(screen, state.projectiles)

        # Render torpedo projectiles
        screen = self._draw_torpedo_projectiles(screen, state.torpedo_projectiles)

        # Render sentinel projectiles
        screen = self._draw_sentinel_projectiles(screen, state.sentinel_projectiles)

        # Render enemies
        screen = self._draw_enemies(screen, state.enemies)

        # Render UI (score, lives, torpedoes, sector progress)
        screen = self._draw_ui(screen, state)

        return screen

    def _draw_ship(self, screen: chex.Array, ship: Ship) -> chex.Array:
        """Draw the player ship with proper sprite - FIXED to match old version"""
        x, y = ship.x.astype(int), ship.y.astype(int)

        # Ship sprite design (same as old version)
        # 0 = transparent, 1 = yellow, 2 = purple
        ship_sprite = jnp.array([
            [0, 0, 0, 2, 2, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=jnp.uint8)

        # Color mapping
        colors = jnp.array([
            [0, 0, 0],  # 0 = transparent (black background)
            [255, 255, 0],  # 1 = yellow
            [160, 32, 240],  # 2 = purple
        ], dtype=jnp.uint8)

        sprite_h, sprite_w = ship_sprite.shape

        # Create coordinate grids
        y_indices = jnp.arange(self.constants.SCREEN_HEIGHT)
        x_indices = jnp.arange(self.constants.SCREEN_WIDTH)
        y_grid, x_grid = jnp.meshgrid(y_indices, x_indices, indexing='ij')

        # Scale up the sprite (6x like old version)
        scale = 6

        def draw_sprite_pixel(i, scr):
            sprite_y = i // sprite_w
            sprite_x = i % sprite_w
            pixel_value = ship_sprite[sprite_y, sprite_x]

            # Skip transparent pixels
            def draw_pixel(scr):  # ADD scr parameter here
                pixel_color = colors[pixel_value]

                # Scale up the pixel (6x6 block)
                start_y = y + sprite_y * scale
                end_y = start_y + scale
                start_x = x + sprite_x * scale
                end_x = start_x + scale

                # Create mask for this scaled pixel block
                pixel_mask = (
                        (y_grid >= start_y) & (y_grid < end_y) &
                        (x_grid >= start_x) & (x_grid < end_x) &
                        (start_y >= 0) & (end_y <= self.constants.SCREEN_HEIGHT) &
                        (start_x >= 0) & (end_x <= self.constants.SCREEN_WIDTH)
                )

                return jnp.where(pixel_mask[..., None], pixel_color, scr)

            return jax.lax.cond(
                pixel_value > 0,  # Only draw non-transparent pixels
                draw_pixel,
                lambda scr: scr,  # ADD scr parameter here too
                scr
            )
        # Draw all sprite pixels
        screen = jax.lax.fori_loop(0, sprite_h * sprite_w, draw_sprite_pixel, screen)
        return screen

    @partial(jax.jit, static_argnums=(0,))
    def _draw_3d_grid(self, screen: chex.Array, frame_count: int) -> chex.Array:
        """Draw 3D grid - FIXED to show exactly 5 beams"""
        height = self.constants.SCREEN_HEIGHT
        width = self.constants.SCREEN_WIDTH
        line_color = jnp.array([64, 64, 255], dtype=jnp.uint8)

        # Create coordinate grids
        y_indices = jnp.arange(height)
        x_indices = jnp.arange(width)
        y_grid, x_grid = jnp.meshgrid(y_indices, x_indices, indexing="ij")

        # === HORIZONTAL LINES (animated) ===
        top_margin = int(height * 0.12)
        bottom_margin = int(height * 0.14)
        grid_height = height - top_margin - bottom_margin

        num_hlines = 7
        phase = (frame_count * 0.003) % 1.0

        def draw_hline(i, scr):
            t = (phase + i / num_hlines) % 1.0
            y = jnp.round((t ** 3.0) * grid_height).astype(int) + top_margin
            y = jnp.clip(y, 0, height - 1)
            mask = y_grid == y
            return jnp.where(mask[..., None], line_color, scr)

        screen = jax.lax.fori_loop(0, num_hlines, draw_hline, screen)

        # === VERTICAL BEAM LINES (exactly 5 beams) ===
        def draw_vbeam(i, scr):
            beam_x = self.beam_positions[i].astype(int)  # FIXED: Use .astype(int) instead of int()
            # Draw vertical line at each beam position
            beam_mask = (x_grid == beam_x) & (y_grid >= top_margin) & (y_grid <= height - bottom_margin)
            return jnp.where(beam_mask[..., None], line_color, scr)

        # Draw exactly NUM_BEAMS (5) vertical lines
        screen = jax.lax.fori_loop(0, self.constants.NUM_BEAMS, draw_vbeam, screen)

        return screen
    def _draw_torpedo_projectiles(self, screen: chex.Array, torpedo_projectiles: chex.Array) -> chex.Array:
        """Draw all active torpedo projectiles - FROM OLD VERSION"""

        # Vectorized drawing function
        def draw_single_torpedo(i, screen):
            x, y = torpedo_projectiles[i, 0].astype(int), torpedo_projectiles[i, 1].astype(int)
            active = torpedo_projectiles[i, 2] == 1

            # Create coordinate grids
            y_indices = jnp.arange(self.constants.SCREEN_HEIGHT)
            x_indices = jnp.arange(self.constants.SCREEN_WIDTH)
            y_grid, x_grid = jnp.meshgrid(y_indices, x_indices, indexing='ij')

            # Create mask for torpedo pixels (slightly larger than regular projectiles)
            torpedo_mask = (
                    (x_grid >= x) &
                    (x_grid < x + self.constants.TORPEDO_WIDTH) &
                    (y_grid >= y) &
                    (y_grid < y + self.constants.TORPEDO_HEIGHT) &
                    active &
                    (x >= 0) & (x < self.constants.SCREEN_WIDTH) &
                    (y >= 0) & (y < self.constants.SCREEN_HEIGHT)
            )

            # Apply torpedo color where mask is True (WHITE for torpedoes vs YELLOW for lasers)
            torpedo_color = jnp.array(self.constants.WHITE, dtype=jnp.uint8)
            screen = jnp.where(
                torpedo_mask[..., None],  # Add dimension for RGB
                torpedo_color,
                screen
            ).astype(jnp.uint8)

            return screen

        # Apply to all torpedo projectiles
        screen = jax.lax.fori_loop(0, self.constants.MAX_PROJECTILES, draw_single_torpedo, screen)
        return screen

    def _draw_sentinel_projectiles(self, screen: chex.Array, sentinel_projectiles: chex.Array) -> chex.Array:
        """Draw all active sentinel projectiles - FROM OLD VERSION"""

        # Vectorized drawing function
        def draw_single_sentinel_projectile(i, screen):
            x, y = sentinel_projectiles[i, 0].astype(int), sentinel_projectiles[i, 1].astype(int)
            active = sentinel_projectiles[i, 2] == 1

            # Create coordinate grids
            y_indices = jnp.arange(self.constants.SCREEN_HEIGHT)
            x_indices = jnp.arange(self.constants.SCREEN_WIDTH)
            y_grid, x_grid = jnp.meshgrid(y_indices, x_indices, indexing='ij')

            # Create mask for sentinel projectile pixels
            projectile_mask = (
                    (x_grid >= x) &
                    (x_grid < x + self.constants.PROJECTILE_WIDTH) &
                    (y_grid >= y) &
                    (y_grid < y + self.constants.PROJECTILE_HEIGHT) &
                    active &
                    (x >= 0) & (x < self.constants.SCREEN_WIDTH) &
                    (y >= 0) & (y < self.constants.SCREEN_HEIGHT)
            )

            # Apply sentinel projectile color (RED to distinguish from player projectiles)
            projectile_color = jnp.array(self.constants.RED, dtype=jnp.uint8)
            screen = jnp.where(
                projectile_mask[..., None],  # Add dimension for RGB
                projectile_color,
                screen
            ).astype(jnp.uint8)

            return screen

        # Apply to all sentinel projectiles
        screen = jax.lax.fori_loop(0, self.constants.MAX_PROJECTILES, draw_single_sentinel_projectile, screen)
        return screen

    def _draw_projectiles(self, screen: chex.Array, projectiles: chex.Array) -> chex.Array:
        """Draw all active projectiles - FROM OLD VERSION"""

        # Vectorized drawing function
        def draw_single_projectile(i, screen):
            x, y = projectiles[i, 0].astype(int), projectiles[i, 1].astype(int)
            active = projectiles[i, 2] == 1

            # Create coordinate grids
            y_indices = jnp.arange(self.constants.SCREEN_HEIGHT)
            x_indices = jnp.arange(self.constants.SCREEN_WIDTH)
            y_grid, x_grid = jnp.meshgrid(y_indices, x_indices, indexing='ij')

            # Create mask for projectile pixels
            projectile_mask = (
                    (x_grid >= x) &
                    (x_grid < x + self.constants.PROJECTILE_WIDTH) &
                    (y_grid >= y) &
                    (y_grid < y + self.constants.PROJECTILE_HEIGHT) &
                    active &
                    (x >= 0) & (x < self.constants.SCREEN_WIDTH) &
                    (y >= 0) & (y < self.constants.SCREEN_HEIGHT)
            )

            # Apply projectile color where mask is True
            projectile_color = jnp.array(self.constants.YELLOW, dtype=jnp.uint8)
            screen = jnp.where(
                projectile_mask[..., None],  # Add dimension for RGB
                projectile_color,
                screen
            ).astype(jnp.uint8)

            return screen

        # Apply to all projectiles
        screen = jax.lax.fori_loop(0, self.constants.MAX_PROJECTILES, draw_single_projectile, screen)
        return screen

    def _draw_enemies(self, screen: chex.Array, enemies: chex.Array) -> chex.Array:
        """Draw all active enemies - FROM OLD VERSION with complete enemy type support"""

        # Vectorized drawing function
        def draw_single_enemy(i, screen):
            x, y = enemies[i, 0].astype(int), enemies[i, 1].astype(int)
            active = enemies[i, 3] == 1
            enemy_type = enemies[i, 5].astype(int)

            # Create coordinate grids
            y_indices = jnp.arange(self.constants.SCREEN_HEIGHT)
            x_indices = jnp.arange(self.constants.SCREEN_WIDTH)
            y_grid, x_grid = jnp.meshgrid(y_indices, x_indices, indexing='ij')

            # Get enemy dimensions (sentinel ships are larger)
            enemy_width = jnp.where(
                enemy_type == self.constants.ENEMY_TYPE_SENTINEL_SHIP,
                self.constants.SENTINEL_SHIP_WIDTH,
                self.constants.ENEMY_WIDTH
            )
            enemy_height = jnp.where(
                enemy_type == self.constants.ENEMY_TYPE_SENTINEL_SHIP,
                self.constants.SENTINEL_SHIP_HEIGHT,
                self.constants.ENEMY_HEIGHT
            )

            # Create mask for enemy pixels
            enemy_mask = (
                    (x_grid >= x) &
                    (x_grid < x + enemy_width) &
                    (y_grid >= y) &
                    (y_grid < y + enemy_height) &
                    active &
                    (x >= 0) & (x < self.constants.SCREEN_WIDTH) &
                    (y >= 0) & (y < self.constants.SCREEN_HEIGHT)
            )

            # Select enemy color based on type - COMPLETE SET FROM OLD VERSION
            enemy_color = jnp.where(
                enemy_type == self.constants.ENEMY_TYPE_BROWN_DEBRIS,
                jnp.array(self.constants.BROWN_DEBRIS_COLOR, dtype=jnp.uint8),
                jnp.where(
                    enemy_type == self.constants.ENEMY_TYPE_YELLOW_CHIRPER,
                    jnp.array(self.constants.YELLOW_CHIRPER_COLOR, dtype=jnp.uint8),
                    jnp.where(
                        enemy_type == self.constants.ENEMY_TYPE_GREEN_BLOCKER,
                        jnp.array(self.constants.GREEN_BLOCKER_COLOR, dtype=jnp.uint8),
                        jnp.where(
                            enemy_type == self.constants.ENEMY_TYPE_GREEN_BOUNCE,
                            jnp.array(self.constants.GREEN_BOUNCE_COLOR, dtype=jnp.uint8),
                            jnp.where(
                                enemy_type == self.constants.ENEMY_TYPE_BLUE_CHARGER,
                                jnp.array(self.constants.BLUE_CHARGER_COLOR, dtype=jnp.uint8),
                                jnp.where(
                                    enemy_type == self.constants.ENEMY_TYPE_ORANGE_TRACKER,
                                    jnp.array(self.constants.ORANGE_TRACKER_COLOR, dtype=jnp.uint8),
                                    jnp.where(
                                        enemy_type == self.constants.ENEMY_TYPE_SENTINEL_SHIP,
                                        jnp.array(self.constants.RED, dtype=jnp.uint8),
                                        jnp.where(
                                            enemy_type == self.constants.ENEMY_TYPE_YELLOW_REJUVENATOR,
                                            jnp.array(self.constants.YELLOW_REJUVENATOR_COLOR, dtype=jnp.uint8),
                                            jnp.where(
                                                enemy_type == self.constants.ENEMY_TYPE_REJUVENATOR_DEBRIS,
                                                jnp.array(self.constants.REJUVENATOR_DEBRIS_COLOR, dtype=jnp.uint8),
                                                jnp.array(self.constants.WHITE, dtype=jnp.uint8)  # Default white saucer
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )

            # Apply enemy color where mask is True
            screen = jnp.where(
                enemy_mask[..., None],  # Add dimension for RGB
                enemy_color,
                screen
            ).astype(jnp.uint8)

            return screen

        # Apply to all enemies
        screen = jax.lax.fori_loop(0, self.constants.MAX_ENEMIES, draw_single_enemy, screen)
        return screen

    def _draw_ui(self, screen: chex.Array, state: BeamRiderState) -> chex.Array:
        """Draw UI elements - FIXED to show lives and torpedoes like old version"""

        # === DRAW TORPEDO INDICATORS (top-right) ===
        cube_size = 5  # Small cubes for torpedoes
        torpedo_color = jnp.array([160, 32, 240], dtype=jnp.uint8)  # Purple

        def draw_torpedo_cube(i, scr):
            # Position from right edge
            cube_x = self.constants.SCREEN_WIDTH - (cube_size + 2) * (i + 1) - 5
            cube_y = 5

            # Create coordinate grids
            y_indices = jnp.arange(self.constants.SCREEN_HEIGHT)
            x_indices = jnp.arange(self.constants.SCREEN_WIDTH)
            y_grid, x_grid = jnp.meshgrid(y_indices, x_indices, indexing='ij')

            # Create mask for torpedo cube
            cube_mask = (
                    (x_grid >= cube_x) & (x_grid < cube_x + cube_size) &
                    (y_grid >= cube_y) & (y_grid < cube_y + cube_size) &
                    (i < state.torpedoes_remaining)  # Only draw if torpedo available
            )

            return jnp.where(cube_mask[..., None], torpedo_color, scr)

        # Draw torpedo cubes
        screen = jax.lax.fori_loop(0, self.constants.TORPEDOES_PER_SECTOR, draw_torpedo_cube, screen)

        # === DRAW LIVES INDICATORS (bottom) ===
        # Small ship sprites for lives
        mini_ship = jnp.array([
            [0, 2, 2, 0],
            [1, 1, 1, 1],
            [1, 0, 0, 1],
        ], dtype=jnp.uint8)

        life_colors = jnp.array([
            [0, 0, 0],  # 0 = transparent
            [255, 255, 0],  # 1 = yellow
            [160, 32, 240],  # 2 = purple
        ], dtype=jnp.uint8)

        def draw_life_indicator(life_i, scr):
            # Position near bottom left
            life_x = 10 + life_i * 20
            life_y = self.constants.SCREEN_HEIGHT - 15

            mini_h, mini_w = mini_ship.shape

            def draw_life_pixel(pixel_i, scr_inner):
                pixel_y = pixel_i // mini_w
                pixel_x = pixel_i % mini_w
                pixel_value = mini_ship[pixel_y, pixel_x]

                def draw_life_pixel_inner(scr_inner):  # ADD scr_inner parameter here
                    pixel_color = life_colors[pixel_value]

                    # Create coordinate grids for this pixel
                    y_indices = jnp.arange(self.constants.SCREEN_HEIGHT)
                    x_indices = jnp.arange(self.constants.SCREEN_WIDTH)
                    y_grid, x_grid = jnp.meshgrid(y_indices, x_indices, indexing='ij')

                    # Scale up 2x
                    start_y = life_y + pixel_y * 2
                    end_y = start_y + 2
                    start_x = life_x + pixel_x * 2
                    end_x = start_x + 2

                    pixel_mask = (
                            (y_grid >= start_y) & (y_grid < end_y) &
                            (x_grid >= start_x) & (x_grid < end_x) &
                            (start_y >= 0) & (end_y <= self.constants.SCREEN_HEIGHT) &
                            (start_x >= 0) & (end_x <= self.constants.SCREEN_WIDTH)
                    )

                    return jnp.where(pixel_mask[..., None], pixel_color, scr_inner)

                return jax.lax.cond(
                    (pixel_value > 0) & (life_i < state.lives),  # Only draw if life exists
                    draw_life_pixel_inner,
                    lambda scr_inner: scr_inner,  # ADD scr_inner parameter here too
                    scr_inner
                )
            return jax.lax.fori_loop(0, mini_h * mini_w, draw_life_pixel, scr)

        # Draw life indicators
        screen = jax.lax.fori_loop(0, self.constants.INITIAL_LIVES, draw_life_indicator, screen)

        return screen

    # ============================================================================
    # PYGAME DISPLAY METHODS (optional for enhanced gameplay)
    # ============================================================================

    def run_game(self):
        """Main game loop with torpedo support - requires pygame to be enabled"""
        if not self.enable_pygame:
            raise RuntimeError("pygame must be enabled to run the game. Initialize with enable_pygame=True")

        env = BeamRiderEnv()
        key = random.PRNGKey(42)
        obs, state = env.reset(key)

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

                # TORPEDO ACTIONS - Map to JAXAtari action constants
                if keys[pygame.K_t] and keys[pygame.K_UP]:
                    action = 10  # UPFIRE - torpedo + up
                elif keys[pygame.K_t] and keys[pygame.K_LEFT]:
                    action = 12  # LEFTFIRE - torpedo + left
                elif keys[pygame.K_t] and keys[pygame.K_RIGHT]:
                    action = 11  # RIGHTFIRE - torpedo + right
                elif keys[pygame.K_t]:
                    action = 10  # UPFIRE - torpedo only (default up)
                # LASER ACTIONS
                elif keys[pygame.K_SPACE]:
                    action = 1  # FIRE - fire laser only
                # MOVEMENT ACTIONS
                elif keys[pygame.K_LEFT]:
                    action = 4  # LEFT
                elif keys[pygame.K_RIGHT]:
                    action = 3  # RIGHT

                # Step and render
                obs, state, reward, done, info = env.step(state, action)

                # Check for sector completion
                self._show_sector_complete(state)
                screen_buffer = self.render(state)
                self._draw_screen(screen_buffer, state)
                self._draw_ui_overlay(state)

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
        """Show sector completion message (called when sector advances)"""
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
        if not self.enable_pygame:
            return
        pause_text = self.font.render("PAUSED", True, (255, 220, 100))
        rect = pause_text.get_rect(center=(self.pygame_screen_width // 2, self.pygame_screen_height // 2))
        self.pygame_screen.blit(pause_text, rect)

    def _draw_screen(self, screen_buffer, state):
        """Draws the game screen buffer and overlays the ship sprite"""
        if not self.enable_pygame:
            return

        screen_np = np.array(screen_buffer)
        scaled_screen = np.repeat(np.repeat(screen_np, self.scale, axis=0), self.scale, axis=1)

        surf = pygame.surfarray.make_surface(scaled_screen.swapaxes(0, 1))
        self.pygame_screen.blit(surf, (0, 0))

        # === OVERLAY THE SHIP SPRITE ===
        if self.ship_sprite_surface is not None:
            ship_x = int(state.ship.x) * self.scale
            ship_y = int(state.ship.y) * self.scale
            self.pygame_screen.blit(self.ship_sprite_surface, (ship_x, ship_y))

    def _draw_ui_overlay(self, state):
        """Draw centered Score and Level UI"""
        if not self.enable_pygame:
            return

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
        if self.small_ship_surface is not None:
            for i in range(state.lives):
                x = 30 + i * 36  # spacing between icons
                y = self.pygame_screen_height - 20  # near bottom
                scaled_ship = pygame.transform.scale(self.small_ship_surface,
                                                     (int(self.small_ship_surface.get_width() * 1.5),
                                                      int(self.small_ship_surface.get_height() * 1.5)))
                self.pygame_screen.blit(scaled_ship, (x, y))

    def _show_game_over(self, state):
        """Show Game Over screen"""
        if not self.enable_pygame:
            return

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
    # For standalone pygame gameplay, create environment and run with pygame
    import pygame
    import sys

    # Create the game environment
    env = BeamRiderEnv()

    # Initialize pygame
    pygame.init()
    scale = 3
    screen_width = env.constants.SCREEN_WIDTH * scale
    screen_height = env.constants.SCREEN_HEIGHT * scale
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("BeamRider - JAX Implementation")
    clock = pygame.time.Clock()

    # Load font (you may need to adjust the path)
    try:
        font = pygame.font.Font("PressStart2P.ttf", 16)
    except:
        font = pygame.font.Font(None, 24)  # Fallback to default font

    # Initialize game state
    key = random.PRNGKey(42)
    obs, state = env.reset(key)

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

            # TORPEDO ACTIONS - Map to JAXAtari action constants
            if keys[pygame.K_t] and keys[pygame.K_UP]:
                action = 10  # UPFIRE - torpedo + up
            elif keys[pygame.K_t] and keys[pygame.K_LEFT]:
                action = 12  # LEFTFIRE - torpedo + left
            elif keys[pygame.K_t] and keys[pygame.K_RIGHT]:
                action = 11  # RIGHTFIRE - torpedo + right
            elif keys[pygame.K_t]:
                action = 10  # UPFIRE - torpedo only (default up)
            # LASER ACTIONS
            elif keys[pygame.K_SPACE]:
                action = 1  # FIRE - fire laser only
            # MOVEMENT ACTIONS
            elif keys[pygame.K_LEFT]:
                action = 4  # LEFT
            elif keys[pygame.K_RIGHT]:
                action = 3  # RIGHT

            # Step the game
            obs, state, reward, done, info = env.step(state, action)

            # Render the game
            screen_buffer = env.render(state)

            # Convert JAX array to numpy and scale up
            screen_np = np.array(screen_buffer)
            scaled_screen = np.repeat(np.repeat(screen_np, scale, axis=0), scale, axis=1)

            # Create pygame surface and blit to screen
            surf = pygame.surfarray.make_surface(scaled_screen.swapaxes(0, 1))
            screen.blit(surf, (0, 0))

            # Draw simple UI overlay
            score_text = font.render(f"SCORE {state.score:06}", True, (255, 255, 255))
            level_text = font.render(f"SECTOR {state.level:02}", True, (255, 255, 255))
            lives_text = font.render(f"LIVES {state.lives}", True, (255, 255, 255))
            torpedoes_text = font.render(f"TORPEDOES {state.torpedoes_remaining}", True, (255, 255, 255))

            screen.blit(score_text, (10, 10))
            screen.blit(level_text, (10, 35))
            screen.blit(lives_text, (10, 60))
            screen.blit(torpedoes_text, (10, 85))

            pygame.display.flip()
            clock.tick(60)
        else:
            # Pause overlay
            pause_text = font.render("PAUSED", True, (255, 255, 0))
            pause_rect = pause_text.get_rect(center=(screen_width // 2, screen_height // 2))
            screen.blit(pause_text, pause_rect)
            pygame.display.flip()
            clock.tick(15)

    # Game over screen
    if state.game_over:
        overlay = pygame.Surface((screen_width, screen_height))
        overlay.set_alpha(128)
        overlay.fill((0, 0, 0))
        screen.blit(overlay, (0, 0))

        game_over_text = font.render("GAME OVER", True, (255, 0, 0))
        final_score_text = font.render(f"Final Score: {state.score}", True, (255, 255, 255))
        sector_text = font.render(f"Reached Sector: {state.current_sector}", True, (255, 255, 255))

        # Center the text
        game_over_rect = game_over_text.get_rect(center=(screen_width // 2, screen_height // 2 - 40))
        score_rect = final_score_text.get_rect(center=(screen_width // 2, screen_height // 2))
        sector_rect = sector_text.get_rect(center=(screen_width // 2, screen_height // 2 + 30))

        screen.blit(game_over_text, game_over_rect)
        screen.blit(final_score_text, score_rect)
        screen.blit(sector_text, sector_rect)

        pygame.display.flip()
        pygame.time.wait(3000)  # Show for 3 seconds

    pygame.quit()
    sys.exit()