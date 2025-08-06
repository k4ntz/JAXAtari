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

from jaxatari.environment import JaxEnvironment
from jaxatari.renderers import JAXGameRenderer

"""TODOS:
- Adjust the spawn rate of green blockers in sectors 1 - 5 (currently the sentinel moves faster through the screen than they can spawn) -> somewhat done
- Change enemy speeds
- Change enemy movements
- Add torpedoes(got lost for some reason) (and the handling for the torpedoes against enemies)
- Torpedo and laser should not travel after collision with enemy
- Only allow one laser per shot and not mutliple back to back
- Documentation
- Should be playable through script
- White saucers movement needs to be update --> No teleporation and zigzag movement needs to be broader

Nice to have:
- Enemies get smaller/bigger according to the 3d rendering"""


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
    ENEMY_TYPE_GREEN_BOUNCE = 4  # Green bounce craft
    ENEMY_TYPE_BLUE_CHARGER = 5  # Blue charger
    ENEMY_TYPE_ORANGE_TRACKER = 6  # Orange tracker
    ENEMY_TYPE_SENTINEL_SHIP = 7  # NEW: Sentinel ship

    # White saucer behavior constants
    WHITE_SAUCER_SHOOT_CHANCE = 0.2  # 20% of white saucers can shoot
    WHITE_SAUCER_JUMP_CHANCE = 0.15  # 15% chance for beam jumping
    WHITE_SAUCER_REVERSE_CHANCE = 0.1  # 10% chance for reverse movement
    WHITE_SAUCER_ZIGZAG_CHANCE = 0.1  # 10% chance for zigzag movement
    WHITE_SAUCER_FIRING_INTERVAL = 90  # Frames between shots
    WHITE_SAUCER_PROJECTILE_SPEED = 2.5  # Speed of white saucer projectiles
    WHITE_SAUCER_JUMP_INTERVAL = 120  # Frames between beam jumps
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

    # Sentinel ship specific constants - UPDATED speeds
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

    # Orange tracker specific constants - UPDATED speeds
    ORANGE_TRACKER_SPEED = 0.9  # Slower base tracking speed
    ORANGE_TRACKER_POINTS = 50  # Points when destroyed with torpedo
    ORANGE_TRACKER_COLOR = (255, 165, 0)  # Orange color RGB
    ORANGE_TRACKER_SPAWN_SECTOR = 12  # Starts appearing from sector 12
    ORANGE_TRACKER_SPAWN_CHANCE = 0.08  # 8% chance to spawn orange tracker
    ORANGE_TRACKER_CHANGE_DIRECTION_INTERVAL = 90  # Frames between direction changes

    # Tracker course change limits based on sector
    ORANGE_TRACKER_BASE_COURSE_CHANGES = 1  # Base number of course changes allowed
    ORANGE_TRACKER_COURSE_CHANGE_INCREASE_SECTOR = 5  # Every X sectors, add 1 more course change

    # Blue charger specific constants - UPDATED speeds
    BLUE_CHARGER_SPEED = 1.1  # Slower base speed
    BLUE_CHARGER_POINTS = 30  # Points when destroyed
    BLUE_CHARGER_COLOR = (0, 0, 255)  # Blue color RGB
    BLUE_CHARGER_SPAWN_SECTOR = 10  # Starts appearing from sector 10
    BLUE_CHARGER_SPAWN_CHANCE = 0.1  # 10% chance to spawn blue charger
    BLUE_CHARGER_LINGER_TIME = 180  # Frames to stay at bottom (3 seconds at 60fps)
    BLUE_CHARGER_DEFLECT_SPEED = -2.0  # Speed when deflected upward by laser

    # Brown debris specific constants - UPDATED speeds
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
    GREEN_BLOCKER_SPEED = 0.15  # Much slower ramming speed
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
    GREEN_BOUNCE_SPAWN_SECTOR = 8  # Starts appearing from sector 8
    GREEN_BOUNCE_SPAWN_CHANCE = 0.08  # 8% chance to spawn green bounce craft
    GREEN_BOUNCE_MAX_BOUNCES = 6  # Maximum number of bounces before disappearing
    # Enemy types (add this new type after ENEMY_TYPE_SENTINEL_SHIP = 7)
    ENEMY_TYPE_YELLOW_REJUVENATOR = 8  # NEW: Yellow rejuvenator
    ENEMY_TYPE_REJUVENATOR_DEBRIS = 9  # NEW: Explosive debris from shot rejuvenators

    # Yellow rejuvenator specific constants
    YELLOW_REJUVENATOR_SPEED = 0.5  # Slow float speed
    YELLOW_REJUVENATOR_POINTS = 0  # No points for shooting (discourage shooting)
    YELLOW_REJUVENATOR_LIFE_BONUS = 1  # Adds 1 life when collected
    YELLOW_REJUVENATOR_COLOR = (255, 255, 100)  # Bright yellow color RGB
    YELLOW_REJUVENATOR_SPAWN_SECTOR = 5  # Starts appearing from sector 5
    YELLOW_REJUVENATOR_SPAWN_CHANCE = 0.04  # 4% chance to spawn (rare)
    YELLOW_REJUVENATOR_OSCILLATION_AMPLITUDE = 15  # Horizontal oscillation range
    YELLOW_REJUVENATOR_OSCILLATION_FREQUENCY = 0.06  # Oscillation frequency

    # Rejuvenator debris constants (when shot)
    REJUVENATOR_DEBRIS_SPEED = 1.5  # Fast moving debris
    REJUVENATOR_DEBRIS_COLOR = (255, 0, 0)  # Red explosive debris
    REJUVENATOR_DEBRIS_COUNT = 4  # Number of debris pieces created
    REJUVENATOR_DEBRIS_SPREAD = 30  # Spread angle for debris
    REJUVENATOR_DEBRIS_LIFETIME = 180  # Frames before debris disappears

    # HUD margins
    TOP_MARGIN = int(210 * 0.12)

    @classmethod
    def get_beam_positions(cls) -> jnp.ndarray:
        """Calculate 5 beam positions evenly spaced across the screen width"""
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
    beam_position: int  # Index of the current beam (0–4)
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

        # Initialize renderer (this was missing)
        self.renderer = BeamRiderRenderer()

    def reset(self, rng_key: chex.PRNGKey) -> Tuple[BeamRiderObservation, BeamRiderState]:
        """Reset the game to initial state"""
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

    def action_space(self):
        """Returns the action space for BeamRider"""
        from jaxatari.spaces import Discrete
        return Discrete(18)  # All standard Atari actions available

    def observation_space(self):
        """Returns the observation space for BeamRider"""
        from jaxatari.spaces import Box
        # Observation includes: ship position, beam, enemies, projectiles, score, lives, etc.
        return Box(
            low=jnp.full((100,), -1e6, dtype=jnp.float32),
            high=jnp.full((100,), 1e6, dtype=jnp.float32),
            dtype=jnp.float32
        )

    def image_space(self):
        """Returns the image space for BeamRider rendering"""
        from jaxatari.spaces import Box
        return Box(
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
        """Extract observation from game state"""
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
        """Convert observation to flat array for ML models"""
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
        """Extract additional info from game state"""
        return BeamRiderInfo(
            enemies_killed_this_sector=jnp.array(state.enemies_killed_this_sector),
            enemy_spawn_timer=jnp.array(state.enemy_spawn_timer),
            enemy_spawn_interval=jnp.array(state.enemy_spawn_interval),
            sentinel_spawned_this_sector=jnp.array(state.sentinel_spawned_this_sector),
            game_over=jnp.array(state.game_over)
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: BeamRiderState, state: BeamRiderState) -> float:
        """Calculate reward based on state changes"""
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
        """Determine if the game is over"""
        return state.game_over

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: BeamRiderState, action: int) -> Tuple[BeamRiderObservation, BeamRiderState, float, bool, BeamRiderInfo]:
        """Execute one game step"""
        previous_state = state
        new_state = self._step_impl(state, action)

        obs = self._get_observation(new_state)
        reward = self._get_reward(previous_state, new_state)
        done = self._get_done(new_state)
        info = self._get_info(new_state)

        return obs, new_state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _step_impl(self, state: BeamRiderState, action: int) -> BeamRiderState:
        """Execute one game step - JIT-compiled implementation"""
        # Process player input and update ship
        state = self._update_ship(state, action)

        # Handle projectile firing (UPDATED to include torpedoes)
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

        # Check collisions
        state = self._check_collisions(state)

        # Check sector progression
        state = self._check_sector_progression(state)

        # Check game over conditions
        state = self._check_game_over(state)

        # Update frame count only once at the end
        state = state.replace(frame_count=state.frame_count + 1)

        return state

    @partial(jax.jit, static_argnums=(0,))
    def _update_ship(self, state: BeamRiderState, action: int) -> BeamRiderState:
        """Update ship position using discrete beam movement with cooldown"""
        ship = state.ship
        current_beam = ship.beam_position

        # Add movement cooldown to prevent rapid beam jumping
        movement_cooldown = 8  # Frames to wait between movements (adjust if needed)
        frames_since_last_move = state.frame_count % movement_cooldown
        can_move_this_frame = frames_since_last_move == 0

        # Use standard JAXAtari action constants: LEFT=4, RIGHT=3
        should_move_left = (action == 4) & (current_beam > 0) & can_move_this_frame
        should_move_right = (action == 3) & (current_beam < self.constants.NUM_BEAMS - 1) & can_move_this_frame

        # Discrete beam movement with cooldown
        new_beam_position = jnp.where(
            should_move_left,  # Move left
            current_beam - 1,
            jnp.where(
                should_move_right,  # Move right
                current_beam + 1,
                current_beam  # No movement
            )
        )

        # Set ship x position to exactly match the beam center
        new_x = self.beam_positions[new_beam_position] - self.constants.SHIP_WIDTH // 2

        return state.replace(ship=ship.replace(x=new_x, beam_position=new_beam_position))

    @partial(jax.jit, static_argnums=(0,))
    def _select_white_saucer_movement_pattern(self, rng_key: chex.PRNGKey) -> int:
        """Select movement pattern for a new white saucer"""
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

    @partial(jax.jit, static_argnums=(0,))
    def _handle_white_saucer_shooting(self, state: BeamRiderState) -> BeamRiderState:
        """Handle white saucer projectile firing"""
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

    @partial(jax.jit, static_argnums=(0,))
    def _update_white_saucer_movement(self, state: BeamRiderState) -> BeamRiderState:
        """Enhanced white saucer movement patterns"""
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

        # Update timers
        new_firing_timer = jnp.maximum(0, firing_timer - 1)
        new_jump_timer = jnp.maximum(0, jump_timer - 1)

        # PATTERN 2: REVERSE_UP (move down first, then rapidly back up)
        reverse_mask = white_saucer_active & (movement_pattern == self.constants.WHITE_SAUCER_REVERSE_UP)

        # IMPORTANT: Only trigger reverse when saucer has moved down significantly
        # Check if reverse saucers have reached the trigger point (150 pixels down from top)
        reached_reverse_point = current_y >= self.constants.WHITE_SAUCER_REVERSE_TRIGGER_Y

        # Debug: Let's make sure the logic is clear
        # Before trigger: use current_speed (should be positive)
        # After trigger: use fast negative speed
        reverse_new_speed = jnp.where(
            reverse_mask & reached_reverse_point,
            self.constants.WHITE_SAUCER_REVERSE_SPEED_FAST,  # -4.0 (fast upward)
            current_speed  # Keep current speed (should be positive when spawned)
        )

        reverse_new_x = current_x
        reverse_new_y = current_y + reverse_new_speed  # Apply the speed
        reverse_new_beam = current_beam

        # [Keep all other pattern logic the same...]

        # PATTERN 0: STRAIGHT_DOWN
        straight_mask = white_saucer_active & (movement_pattern == self.constants.WHITE_SAUCER_STRAIGHT_DOWN)
        straight_new_x = current_x
        straight_new_y = current_y + current_speed
        straight_new_beam = current_beam
        straight_new_speed = current_speed

        # PATTERN 1: BEAM_JUMP
        jump_mask = white_saucer_active & (movement_pattern == self.constants.WHITE_SAUCER_BEAM_JUMP)
        should_jump = jump_mask & (new_jump_timer == 0)

        jump_indices = jnp.arange(self.constants.MAX_ENEMIES)
        jump_rng_keys = jax.vmap(lambda i: random.fold_in(state.rng_key, state.frame_count + i))(jump_indices)
        new_random_beams = jax.vmap(lambda key: random.randint(key, (), 0, self.constants.NUM_BEAMS))(jump_rng_keys)

        jump_new_beam = jnp.where(should_jump, new_random_beams, current_beam)
        jump_new_x = jnp.where(
            should_jump,
            self.beam_positions[jump_new_beam] - self.constants.ENEMY_WIDTH // 2,
            current_x
        )
        new_jump_timer = jnp.where(should_jump, self.constants.WHITE_SAUCER_JUMP_INTERVAL, new_jump_timer)
        jump_new_y = current_y + current_speed
        jump_new_speed = current_speed

        # PATTERN 3: ZIGZAG
        zigzag_mask = white_saucer_active & (movement_pattern == self.constants.WHITE_SAUCER_ZIGZAG)
        new_zigzag_offset = jnp.where(
            zigzag_mask,
            zigzag_offset + self.constants.WHITE_SAUCER_ZIGZAG_FREQUENCY,
            zigzag_offset
        )
        beam_center_x = self.beam_positions[current_beam]
        zigzag_delta = jnp.sin(new_zigzag_offset) * self.constants.WHITE_SAUCER_ZIGZAG_AMPLITUDE
        zigzag_new_x = jnp.clip(
            beam_center_x + zigzag_delta - self.constants.ENEMY_WIDTH // 2,
            0, self.constants.SCREEN_WIDTH - self.constants.ENEMY_WIDTH
        )
        zigzag_new_y = current_y + current_speed
        zigzag_new_beam = current_beam
        zigzag_new_speed = current_speed

        # PATTERN 4: SHOOTING
        shooting_mask = white_saucer_active & (movement_pattern == self.constants.WHITE_SAUCER_SHOOTING)
        shooting_new_x = current_x
        shooting_new_y = current_y + current_speed
        shooting_new_beam = current_beam
        shooting_new_speed = current_speed

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
                                        jnp.where(reverse_mask, reverse_new_speed,  # This is KEY - speed changes here
                                                  jnp.where(zigzag_mask, zigzag_new_speed,
                                                            jnp.where(shooting_mask, shooting_new_speed,
                                                                      current_speed)))))

        # Update all white saucer properties
        enemies = enemies.at[:, 0].set(jnp.where(white_saucer_active, new_x, enemies[:, 0]))
        enemies = enemies.at[:, 1].set(jnp.where(white_saucer_active, new_y, enemies[:, 1]))
        enemies = enemies.at[:, 2].set(jnp.where(white_saucer_active, new_beam, enemies[:, 2]))
        enemies = enemies.at[:, 4].set(
            jnp.where(white_saucer_active, new_speed, enemies[:, 4]))  # CRITICAL: Update speed
        enemies = enemies.at[:, 15].set(jnp.where(white_saucer_active, new_firing_timer, enemies[:, 15]))
        enemies = enemies.at[:, 16].set(jnp.where(white_saucer_active, new_jump_timer, enemies[:, 16]))
        enemies = enemies.at[:, 17].set(jnp.where(white_saucer_active, new_zigzag_offset, enemies[:, 17]))

        # Handle off-screen deactivation
        standard_off_screen = new_y > self.constants.SCREEN_HEIGHT
        reverse_off_screen = reverse_mask & (new_y < -self.constants.ENEMY_HEIGHT)
        white_saucer_off_screen = standard_off_screen | reverse_off_screen

        current_active = enemies[:, 3] == 1
        new_active_bool = current_active & jnp.where(white_saucer_active, ~white_saucer_off_screen, True)
        enemies = enemies.at[:, 3].set(new_active_bool.astype(jnp.float32))

        return state.replace(enemies=enemies)

    @partial(jax.jit, static_argnums=(0,))
    def _handle_firing(self, state: BeamRiderState, action: int) -> BeamRiderState:
        """Handle both laser and torpedo firing with proper action mapping"""

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

    @partial(jax.jit, static_argnums=(0,))
    def _fire_laser(self, state: BeamRiderState, should_fire: bool) -> BeamRiderState:
        """Fire regular laser projectile"""
        projectiles = state.projectiles

        # Find first inactive projectile slot
        inactive_mask = projectiles[:, 2] == 0
        first_inactive = jnp.argmax(inactive_mask)
        can_fire = inactive_mask[first_inactive] & should_fire

        # Create new projectile at ship position
        ship_center_x = state.ship.x + self.constants.SHIP_WIDTH // 2
        new_projectile = jnp.array([
            ship_center_x - self.constants.PROJECTILE_WIDTH // 2,  # x
            state.ship.y,  # y
            1.0,  # active
            -self.constants.PROJECTILE_SPEED  # speed (negative = upward)
        ])

        # Update projectiles array
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

        # Find ALL inactive torpedo slots
        inactive_mask = torpedo_projectiles[:, 2] == 0

        # Check if ANY slot is available
        has_torpedo_slot = jnp.any(inactive_mask)
        has_torpedoes = state.torpedoes_remaining > 0

        # Can fire if we have a slot AND torpedoes remaining
        can_fire = has_torpedo_slot & has_torpedoes & should_fire

        # Find the first actual inactive slot (only if we can fire)
        first_inactive = jnp.argmax(inactive_mask)

        # Create new torpedo at ship position
        ship_center_x = state.ship.x + self.constants.SHIP_WIDTH // 2
        new_torpedo = jnp.array([
            ship_center_x - self.constants.TORPEDO_WIDTH // 2,  # x
            state.ship.y,  # y
            1.0,  # active
            -self.constants.TORPEDO_SPEED  # speed (negative = upward)
        ])

        # Update torpedo projectiles array - only update if we can actually fire
        torpedo_projectiles = jnp.where(
            can_fire,
            torpedo_projectiles.at[first_inactive].set(new_torpedo),
            torpedo_projectiles
        )

        # Decrease torpedo count
        torpedoes_remaining = jnp.where(
            can_fire,
            state.torpedoes_remaining - 1,
            state.torpedoes_remaining
        )

        return state.replace(
            torpedo_projectiles=torpedo_projectiles,
            torpedoes_remaining=torpedoes_remaining
        )

    @partial(jax.jit, static_argnums=(0,))
    def _update_projectiles(self, state: BeamRiderState) -> BeamRiderState:
        """Update all projectiles (lasers and torpedoes)"""
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
        torpedo_active = (torpedo_projectiles[:, 2] == 1) & (torpedo_new_y >= 0)  # Keep active while y >= 0
        torpedo_projectiles = torpedo_projectiles.at[:, 1].set(torpedo_new_y)
        torpedo_projectiles = torpedo_projectiles.at[:, 2].set(torpedo_active.astype(jnp.float32))

        return state.replace(
            projectiles=projectiles,
            torpedo_projectiles=torpedo_projectiles
        )

    @partial(jax.jit, static_argnums=(0,))
    def _update_sentinel_projectiles(self, state: BeamRiderState) -> BeamRiderState:
        """Update sentinel ship projectiles"""
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

    @partial(jax.jit, static_argnums=(0,))
    def _spawn_enemies(self, state: BeamRiderState) -> BeamRiderState:
        """Spawn new enemies with proper sector-based enemy types and discrete beam positioning"""

        # Check if white saucers are complete for this sector
        white_saucers_complete = state.enemies_killed_this_sector >= self.constants.ENEMIES_PER_SECTOR

        # Check if it's time to spawn an enemy
        should_spawn = (state.enemy_spawn_timer <= 0) & (~white_saucers_complete)

        enemies = state.enemies

        # Find first inactive enemy slot
        active_enemies = enemies[:, 3] == 1
        inactive_mask = ~active_enemies
        first_inactive = jnp.where(inactive_mask, jnp.arange(self.constants.MAX_ENEMIES),
                                   self.constants.MAX_ENEMIES).min()

        # Check if we have space
        can_spawn = first_inactive < self.constants.MAX_ENEMIES

        # Only spawn if conditions are met
        do_spawn = should_spawn & can_spawn

        # Generate random keys
        rng_key, subkey1, subkey2, subkey3, subkey4 = random.split(state.rng_key, 5)

        # PROPER SECTOR-BASED ENEMY TYPE SELECTION
        enemy_type = self._select_enemy_type_by_sector(state.current_sector, subkey1)

        # All enemies spawn on discrete beams only
        spawn_beam = random.randint(subkey3, (), 0, self.constants.NUM_BEAMS)
        spawn_x = self.beam_positions[spawn_beam] - self.constants.ENEMY_WIDTH // 2
        spawn_y = self.constants.ENEMY_SPAWN_Y

        # Calculate enemy speed based on sector (proper scaling)
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
                        self.constants.ENEMY_SPEED  # Default
                    )
                )
            )
        )

        # Smooth speed scaling across sectors (1.0x to 2.5x multiplier)
        speed_scale_factor = 1.0 + ((state.current_sector - 1) / 98.0) * 1.5
        enemy_speed = base_speed * speed_scale_factor

        # Movement patterns for white saucers
        movement_pattern = jnp.where(
            enemy_type == self.constants.ENEMY_TYPE_WHITE_SAUCER,
            self._select_white_saucer_movement_pattern(subkey2),
            0  # Default pattern for non-white saucers
        )

        # Create new enemy array entry
        new_enemy = jnp.array([
            spawn_x,  # 0: x
            spawn_y,  # 1: y
            spawn_beam,  # 2: beam_position
            1.0,  # 3: active
            enemy_speed,  # 4: speed
            enemy_type,  # 5: type
            1.0,  # 6: direction_x (default right)
            1.0,  # 7: direction_y (default down)
            0,  # 8: bounce_count
            0,  # 9: linger_timer
            spawn_x,  # 10: target_x (for blockers, set to spawn position initially)
            1,  # 11: health
            0,  # 12: firing_timer
            0,  # 13: maneuver_timer
            movement_pattern,  # 14: movement_pattern
            0,  # 15: white_saucer_firing_timer
            0,  # 16: jump_timer
            0.0,  # 17: zigzag_offset
        ])

        # Update enemies array only if spawning
        enemies = jnp.where(
            do_spawn,
            enemies.at[first_inactive].set(new_enemy),
            enemies
        )

        # Update spawn timer: reset if spawning, otherwise decrement
        new_spawn_timer = jnp.where(
            do_spawn,
            state.enemy_spawn_interval,
            state.enemy_spawn_timer - 1
        )

        return state.replace(
            enemies=enemies,
            enemy_spawn_timer=new_spawn_timer,
            rng_key=rng_key
        )

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
    def _select_enemy_type_by_sector(self, sector: int, rng_key: chex.PRNGKey) -> int:
        """Select enemy type based on current sector with proper spawn rules"""

        # Generate random value for enemy type selection
        rand_val = random.uniform(rng_key, (), minval=0.0, maxval=1.0, dtype=jnp.float32)

        # Check availability based on sector
        brown_debris_available = sector >= self.constants.BROWN_DEBRIS_SPAWN_SECTOR  # Sector 2
        yellow_chirper_available = sector >= self.constants.YELLOW_CHIRPER_SPAWN_SECTOR  # Sector 4
        green_blocker_available = sector >= self.constants.GREEN_BLOCKER_SPAWN_SECTOR  # Sector 6

        # Calculate spawn probabilities ONLY for available enemy types
        brown_debris_chance = jnp.where(brown_debris_available, self.constants.BROWN_DEBRIS_SPAWN_CHANCE, 0.0)
        yellow_chirper_chance = jnp.where(yellow_chirper_available, self.constants.YELLOW_CHIRPER_SPAWN_CHANCE, 0.0)
        green_blocker_chance = jnp.where(green_blocker_available, self.constants.GREEN_BLOCKER_SPAWN_CHANCE, 0.0)

        # Calculate cumulative thresholds
        green_blocker_threshold = green_blocker_chance
        yellow_chirper_threshold = green_blocker_threshold + yellow_chirper_chance
        brown_debris_threshold = yellow_chirper_threshold + brown_debris_chance
        # Remaining probability goes to white saucers

        # Select enemy type using thresholds
        enemy_type = jnp.where(
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

        return enemy_type

    @partial(jax.jit, static_argnums=(0,))
    def _select_enemy_type_excluding_blockers_early_sectors(self, sector: int, rng_key: chex.PRNGKey) -> int:
        """Select enemy type - includes yellow rejuvenators"""

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

    @partial(jax.jit, static_argnums=(0,))
    def _select_enemy_type(self, sector: int, rng_key: chex.PRNGKey) -> int:
        """Select enemy type based on current sector - UPDATED: includes sentinel ship"""

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
    def _update_enemies(self, state: BeamRiderState) -> BeamRiderState:
        """Update enemy positions using discrete beam movement for ALL enemy types"""
        enemies = state.enemies

        # Only update active enemies
        active_mask = enemies[:, 3] == 1

        # Get enemy data
        current_x = enemies[:, 0]
        current_y = enemies[:, 1]
        current_beam = enemies[:, 2].astype(int)
        current_speed = enemies[:, 4]
        enemy_types = enemies[:, 5].astype(int)
        direction_x = enemies[:, 6]
        direction_y = enemies[:, 7]
        bounce_count = enemies[:, 8].astype(int)
        linger_timer = enemies[:, 9].astype(int)
        target_x = enemies[:, 10]
        health = enemies[:, 11].astype(int)
        firing_timer = enemies[:, 12].astype(int)
        maneuver_timer = enemies[:, 13].astype(int)
        movement_pattern = enemies[:, 14].astype(int)
        white_saucer_firing_timer = enemies[:, 15].astype(int)
        jump_timer = enemies[:, 16].astype(int)
        zigzag_offset = enemies[:, 17]

        # Update timers
        new_jump_timer = jnp.maximum(0, jump_timer - 1)
        new_linger_timer = jnp.maximum(0, linger_timer - 1)
        new_firing_timer = jnp.maximum(0, firing_timer - 1)
        new_maneuver_timer = jnp.maximum(0, maneuver_timer - 1)
        new_white_saucer_firing_timer = jnp.maximum(0, white_saucer_firing_timer - 1)

        # =================================================================
        # WHITE SAUCER MOVEMENT PATTERNS (Discrete Beam-Based)
        # =================================================================
        white_saucer_active = active_mask & (enemy_types == self.constants.ENEMY_TYPE_WHITE_SAUCER)

        # PATTERN 0: STRAIGHT DOWN (stay on same beam)
        straight_mask = white_saucer_active & (movement_pattern == self.constants.WHITE_SAUCER_STRAIGHT_DOWN)
        straight_new_x = self.beam_positions[current_beam] - self.constants.ENEMY_WIDTH // 2
        straight_new_y = current_y + current_speed
        straight_new_beam = current_beam

        # PATTERN 1: BEAM_JUMP (discrete beam hopping)
        jump_mask = white_saucer_active & (movement_pattern == self.constants.WHITE_SAUCER_BEAM_JUMP)
        should_jump = jump_mask & (new_jump_timer == 0)

        # Generate new random beam for jumping
        jump_indices = jnp.arange(self.constants.MAX_ENEMIES)
        jump_rng_keys = jax.vmap(lambda i: random.fold_in(state.rng_key, state.frame_count + i))(jump_indices)
        new_random_beams = jax.vmap(lambda key: random.randint(key, (), 0, self.constants.NUM_BEAMS))(jump_rng_keys)

        jump_new_beam = jnp.where(should_jump, new_random_beams, current_beam)
        jump_new_x = self.beam_positions[jump_new_beam] - self.constants.ENEMY_WIDTH // 2
        jump_new_y = current_y + current_speed
        new_jump_timer = jnp.where(should_jump, self.constants.WHITE_SAUCER_JUMP_INTERVAL, new_jump_timer)

        # PATTERN 2: REVERSE_UP (move up on same beam)
        reverse_mask = white_saucer_active & (movement_pattern == self.constants.WHITE_SAUCER_REVERSE_UP)
        reverse_new_x = self.beam_positions[current_beam] - self.constants.ENEMY_WIDTH // 2
        reverse_new_y = current_y + self.constants.WHITE_SAUCER_REVERSE_SPEED  # negative speed
        reverse_new_beam = current_beam

        # PATTERN 3: ZIGZAG (oscillate between adjacent beams discretely)
        zigzag_mask = white_saucer_active & (movement_pattern == self.constants.WHITE_SAUCER_ZIGZAG)
        new_zigzag_offset = jnp.where(
            zigzag_mask,
            zigzag_offset + self.constants.WHITE_SAUCER_ZIGZAG_FREQUENCY,
            zigzag_offset
        )

        # Zigzag between current beam and adjacent beams (discrete)
        zigzag_beam_offset = jnp.round(jnp.sin(new_zigzag_offset) * 1.0).astype(int)  # -1, 0, or 1
        zigzag_new_beam = jnp.clip(current_beam + zigzag_beam_offset, 0, self.constants.NUM_BEAMS - 1)
        zigzag_new_x = self.beam_positions[zigzag_new_beam] - self.constants.ENEMY_WIDTH // 2
        zigzag_new_y = current_y + current_speed

        # PATTERN 4: SHOOTING (stay on same beam, move straight down)
        shooting_mask = white_saucer_active & (movement_pattern == self.constants.WHITE_SAUCER_SHOOTING)
        shooting_new_x = self.beam_positions[current_beam] - self.constants.ENEMY_WIDTH // 2
        shooting_new_y = current_y + current_speed
        shooting_new_beam = current_beam

        # Apply white saucer movement patterns
        ws_new_x = jnp.where(straight_mask, straight_new_x,
                             jnp.where(jump_mask, jump_new_x,
                                       jnp.where(reverse_mask, reverse_new_x,
                                                 jnp.where(zigzag_mask, zigzag_new_x,
                                                           jnp.where(shooting_mask, shooting_new_x, current_x)))))

        ws_new_y = jnp.where(straight_mask, straight_new_y,
                             jnp.where(jump_mask, jump_new_y,
                                       jnp.where(reverse_mask, reverse_new_y,
                                                 jnp.where(zigzag_mask, zigzag_new_y,
                                                           jnp.where(shooting_mask, shooting_new_y, current_y)))))

        ws_new_beam = jnp.where(straight_mask, straight_new_beam,
                                jnp.where(jump_mask, jump_new_beam,
                                          jnp.where(reverse_mask, reverse_new_beam,
                                                    jnp.where(zigzag_mask, zigzag_new_beam,
                                                              jnp.where(shooting_mask, shooting_new_beam,
                                                                        current_beam)))))

        # =================================================================
        # BROWN DEBRIS: Move straight down on assigned beam
        # =================================================================
        brown_debris_mask = active_mask & (enemy_types == self.constants.ENEMY_TYPE_BROWN_DEBRIS)
        brown_new_x = self.beam_positions[current_beam] - self.constants.ENEMY_WIDTH // 2
        brown_new_y = current_y + current_speed
        brown_new_beam = current_beam

        # =================================================================
        # YELLOW CHIRPER: Move horizontally between beams (discrete)
        # =================================================================
        chirper_mask = active_mask & (enemy_types == self.constants.ENEMY_TYPE_YELLOW_CHIRPER)

        # Move to adjacent beam based on direction
        chirper_beam_delta = jnp.where(direction_x > 0, 1, -1)  # Move right or left beam
        chirper_new_beam = jnp.clip(current_beam + chirper_beam_delta, 0, self.constants.NUM_BEAMS - 1)

        # If reached edge beams, reverse direction
        reached_left_edge = (current_beam == 0) & (direction_x < 0)
        reached_right_edge = (current_beam == self.constants.NUM_BEAMS - 1) & (direction_x > 0)
        should_reverse = reached_left_edge | reached_right_edge

        new_direction_x = jnp.where(
            chirper_mask & should_reverse,
            -direction_x,  # Reverse direction
            direction_x  # Keep same direction
        )

        chirper_new_x = self.beam_positions[chirper_new_beam] - self.constants.ENEMY_WIDTH // 2
        chirper_new_y = current_y + current_speed * 0.3  # Slower vertical movement

        # =================================================================
        # GREEN BLOCKER: Move toward player's beam (discrete)
        # =================================================================
        blocker_mask = active_mask & (enemy_types == self.constants.ENEMY_TYPE_GREEN_BLOCKER)

        # Get player's current beam
        player_beam = state.ship.beam_position

        # Calculate direction to player's beam
        beam_diff = player_beam - current_beam
        blocker_beam_direction = jnp.where(
            beam_diff > 0, 1,  # Move right toward player
            jnp.where(beam_diff < 0, -1, 0)  # Move left toward player, or stay if same beam
        )

        blocker_new_beam = jnp.clip(
            current_beam + blocker_beam_direction,
            0, self.constants.NUM_BEAMS - 1
        )

        blocker_new_x = self.beam_positions[blocker_new_beam] - self.constants.ENEMY_WIDTH // 2
        blocker_new_y = current_y + current_speed

        # =================================================================
        # BLUE CHARGER: Move down on beam, linger at bottom
        # =================================================================
        blue_charger_mask = active_mask & (enemy_types == self.constants.ENEMY_TYPE_BLUE_CHARGER)

        # Normal downward movement
        charger_new_x = self.beam_positions[current_beam] - self.constants.ENEMY_WIDTH // 2
        charger_new_y = current_y + current_speed
        charger_new_beam = current_beam

        # If reached bottom and has linger time, stay there
        at_bottom = current_y >= (self.constants.SCREEN_HEIGHT - self.constants.SHIP_BOTTOM_OFFSET - 30)
        should_linger = blue_charger_mask & at_bottom & (new_linger_timer > 0)

        charger_new_y = jnp.where(should_linger, current_y, charger_new_y)  # Don't move if lingering

        # =================================================================
        # ORANGE TRACKER: Track player beam with limited course changes
        # =================================================================
        tracker_mask = active_mask & (enemy_types == self.constants.ENEMY_TYPE_ORANGE_TRACKER)

        # Move toward player's beam (similar to blocker but with limits)
        tracker_beam_diff = player_beam - current_beam
        tracker_beam_direction = jnp.where(
            tracker_beam_diff > 0, 1,
            jnp.where(tracker_beam_diff < 0, -1, 0)
        )

        tracker_new_beam = jnp.clip(
            current_beam + tracker_beam_direction,
            0, self.constants.NUM_BEAMS - 1
        )

        tracker_new_x = self.beam_positions[tracker_new_beam] - self.constants.ENEMY_WIDTH // 2
        tracker_new_y = current_y + current_speed

        # =================================================================
        # SENTINEL SHIP: Move horizontally across screen and disappear
        # =================================================================
        sentinel_mask = active_mask & (enemy_types == self.constants.ENEMY_TYPE_SENTINEL_SHIP)

        # Move continuously across beams until off-screen
        sentinel_beam_delta = jnp.where(direction_x > 0, 1, -1)
        sentinel_new_beam = current_beam + sentinel_beam_delta

        # Don't clip to beam boundaries - let them go off-screen
        # But set x position based on beam if still on valid beams
        on_valid_beam = (sentinel_new_beam >= 0) & (sentinel_new_beam < self.constants.NUM_BEAMS)

        sentinel_new_x = jnp.where(
            on_valid_beam,
            self.beam_positions[sentinel_new_beam] - self.constants.SENTINEL_SHIP_WIDTH // 2,  # Use beam position
            current_x + (direction_x * self.constants.SENTINEL_SHIP_SPEED * 10)  # Continue moving off-screen
        )

        sentinel_new_y = current_y + current_speed

        # Keep same direction - no reversing
        new_sentinel_direction_x = direction_x

        # =================================================================
        # DEFAULT: All other enemies move straight down on their beam
        # =================================================================
        default_mask = active_mask & ~(
                white_saucer_active | brown_debris_mask | chirper_mask | blocker_mask | blue_charger_mask | tracker_mask | sentinel_mask)
        default_new_x = self.beam_positions[current_beam] - self.constants.ENEMY_WIDTH // 2
        default_new_y = current_y + current_speed
        default_new_beam = current_beam

        # =================================================================
        # COMBINE ALL MOVEMENT PATTERNS
        # =================================================================
        new_x = jnp.where(white_saucer_active, ws_new_x,
                          jnp.where(brown_debris_mask, brown_new_x,
                                    jnp.where(chirper_mask, chirper_new_x,
                                              jnp.where(blocker_mask, blocker_new_x,
                                                        jnp.where(blue_charger_mask, charger_new_x,
                                                                  jnp.where(tracker_mask, tracker_new_x,
                                                                            jnp.where(sentinel_mask, sentinel_new_x,
                                                                                      jnp.where(default_mask,
                                                                                                default_new_x,
                                                                                                current_x))))))))

        new_y = jnp.where(white_saucer_active, ws_new_y,
                          jnp.where(brown_debris_mask, brown_new_y,
                                    jnp.where(chirper_mask, chirper_new_y,
                                              jnp.where(blocker_mask, blocker_new_y,
                                                        jnp.where(blue_charger_mask, charger_new_y,
                                                                  jnp.where(tracker_mask, tracker_new_y,
                                                                            jnp.where(sentinel_mask, sentinel_new_y,
                                                                                      jnp.where(default_mask,
                                                                                                default_new_y,
                                                                                                current_y))))))))

        new_beam = jnp.where(white_saucer_active, ws_new_beam,
                             jnp.where(brown_debris_mask, brown_new_beam,
                                       jnp.where(chirper_mask, chirper_new_beam,
                                                 jnp.where(blocker_mask, blocker_new_beam,
                                                           jnp.where(blue_charger_mask, charger_new_beam,
                                                                     jnp.where(tracker_mask, tracker_new_beam,
                                                                               jnp.where(sentinel_mask,
                                                                                         sentinel_new_beam,
                                                                                         jnp.where(default_mask,
                                                                                                   default_new_beam,
                                                                                                   current_beam))))))))

        # Update direction_x for enemies that reverse
        final_direction_x = jnp.where(chirper_mask, new_direction_x,
                                      jnp.where(sentinel_mask, new_sentinel_direction_x, direction_x))

        # =================================================================
        # DEACTIVATE OFF-SCREEN ENEMIES
        # =================================================================
        new_active = active_mask & (new_y < self.constants.SCREEN_HEIGHT) & (new_y > -self.constants.ENEMY_HEIGHT)

        # For reverse-moving white saucers, also deactivate if they go too high
        new_active = jnp.where(
            reverse_mask & (new_y < 0),
            False,
            new_active
        )

        # For sentinel ships, deactivate when they go off-screen horizontally
        sentinel_off_screen = sentinel_mask & (
                (new_x < -self.constants.SENTINEL_SHIP_WIDTH) |  # Gone off left side
                (new_x > self.constants.SCREEN_WIDTH)  # Gone off right side
        )
        new_active = jnp.where(sentinel_off_screen, False, new_active)

        # =================================================================
        # UPDATE ENEMY ARRAY
        # =================================================================
        enemies = enemies.at[:, 0].set(new_x)  # x
        enemies = enemies.at[:, 1].set(new_y)  # y
        enemies = enemies.at[:, 2].set(new_beam)  # beam_position
        enemies = enemies.at[:, 3].set(new_active.astype(jnp.float32))  # active
        enemies = enemies.at[:, 6].set(final_direction_x)  # direction_x
        enemies = enemies.at[:, 9].set(new_linger_timer)  # linger_timer
        enemies = enemies.at[:, 12].set(new_firing_timer)  # firing_timer
        enemies = enemies.at[:, 13].set(new_maneuver_timer)  # maneuver_timer
        enemies = enemies.at[:, 15].set(new_white_saucer_firing_timer)  # white_saucer_firing_timer
        enemies = enemies.at[:, 16].set(new_jump_timer)  # jump_timer
        enemies = enemies.at[:, 17].set(new_zigzag_offset)  # zigzag_offset

        return state.replace(enemies=enemies)

    @partial(jax.jit, static_argnums=(0,))
    def _check_collisions(self, state: BeamRiderState) -> BeamRiderState:
        """Check for collisions between projectiles and enemies"""
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

        # Calculate score with different point values
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
        ship_x, ship_y = state.ship.x, state.ship.y
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

        # Check regular enemy-ship collisions (exclude chirpers and sentinels)
        can_collide_with_ship = (
                (enemies[:, 5] != self.constants.ENEMY_TYPE_YELLOW_CHIRPER) &
                (enemies[:, 5] != self.constants.ENEMY_TYPE_SENTINEL_SHIP)  # Sentinels don't collide directly
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
        lives = jnp.where(any_ship_collision, state.lives - 1, state.lives)

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

    @partial(jax.jit, static_argnums=(0,))
    def _spawn_rejuvenator_debris(self, state: BeamRiderState, rejuvenator_hit_mask: chex.Array,
                                  enemies: chex.Array) -> BeamRiderState:
        """Spawn explosive debris when rejuvenator is shot"""

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
        """Spawn the sector sentinel ship - FIXED: Always spawns from left side"""
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
    """Unified renderer for BeamRider game with both JAX rendering and Pygame display"""

    def __init__(self):
        super().__init__()
        self.constants = BeamRiderConstants()
        self.screen_width = self.constants.SCREEN_WIDTH
        self.screen_height = self.constants.SCREEN_HEIGHT
        self.beam_positions = self.constants.get_beam_positions()

        # JIT-compile the render function
        self.render = jit(self._render_impl)

    """def __init__(self, scale=3, enable_pygame=False):
        super().__init__()
        self.constants = BeamRiderConstants()
        self.screen_width = self.constants.SCREEN_WIDTH
        self.screen_height = self.constants.SCREEN_HEIGHT
        self.beam_positions = self.constants.get_beam_positions()

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
        self.enable_pygame = enable_pygame
        if enable_pygame:
            pygame.init()
            self.scale = scale
            self.pygame_screen_width = self.screen_width * scale
            self.pygame_screen_height = self.screen_height * scale
            self.pygame_screen = pygame.display.set_mode((self.pygame_screen_width, self.pygame_screen_height))
            pygame.display.set_caption("BeamRider - JAX Implementation")
            self.clock = pygame.time.Clock()
            import os  # at the top of the file if not already there
            font_path = os.path.join(os.path.dirname(__file__), "../../../assets/PressStart2P.ttf")
            self.font = pygame.font.Font(font_path, 16)
            self.env = BeamRiderEnv()"""

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
        """Render the current game state to a screen buffer - JIT compiled implementation"""
        # Create screen buffer (RGB)
        screen = jnp.zeros((self.constants.SCREEN_HEIGHT, self.constants.SCREEN_WIDTH, 3), dtype=jnp.uint8)

        # Render 3D dotted tunnel grid
        screen = self._draw_3d_grid(screen, state.frame_count)

        # Render ship
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

    def _draw_3d_grid(self, screen: chex.Array, frame_count: int) -> chex.Array:
        """Draw 3D grid with 7 animated horizontal lines and 5 vertical gameplay beams"""

        height = self.constants.SCREEN_HEIGHT
        width = self.constants.SCREEN_WIDTH
        line_color = jnp.array([64, 64, 255], dtype=jnp.uint8)  # Blueish grid color

        # === Margins ===
        top_margin = int(height * 0.12)  # Reserved space for HUD
        bottom_margin = int(height * 0.14)  # Reserved space below ship
        grid_height = height - top_margin - bottom_margin

        # Generate mesh grid for pixel coordinates
        y_indices = jnp.arange(height)
        x_indices = jnp.arange(width)
        y_grid, x_grid = jnp.meshgrid(y_indices, x_indices, indexing="ij")

        # === Horizontal Lines ===
        num_hlines = 7  # Number of animated lines
        spacing = grid_height // (num_hlines + 1)
        phase = (frame_count * 0.003) % 1.0  # Smooth looping animation phase

        def draw_hline(i, scr):
            # Animate line position using easing (t^3 curve)
            t = (phase + i / num_hlines) % 1.0
            y = jnp.round((t ** 3.0) * grid_height).astype(int) + top_margin
            y = jnp.clip(y, 0, height - 1)
            mask = y_grid == y
            return jnp.where(mask[..., None], line_color, scr)

        # Draw each horizontal line
        screen = jax.lax.fori_loop(0, num_hlines, draw_hline, screen)

        # === Vertical Beam Lines ===
        # Draw 5 vertical beams that match our gameplay beam positions
        y0 = height - bottom_margin  # Line starts here (bottom)
        y1 = top_margin  # Line ends here (top)

        def draw_vbeam(i, scr):
            # Get the actual gameplay beam position
            beam_x = self.beam_positions[i]

            # Draw vertical line from bottom to top
            beam_mask = jnp.abs(x_grid - beam_x) < 1  # 1-pixel wide line
            vertical_mask = (y_grid >= y1) & (y_grid <= y0)
            line_mask = beam_mask & vertical_mask

            return jnp.where(line_mask[..., None], line_color, scr)

        # Draw each vertical beam
        screen = jax.lax.fori_loop(0, self.constants.NUM_BEAMS, draw_vbeam, screen)

        return screen

    def _draw_ship(self, screen: chex.Array, ship: Ship) -> chex.Array:
        """Draw the player ship"""
        x, y = ship.x.astype(int), ship.y.astype(int)

        # Create coordinate grids
        y_indices = jnp.arange(self.constants.SCREEN_HEIGHT)
        x_indices = jnp.arange(self.constants.SCREEN_WIDTH)
        y_grid, x_grid = jnp.meshgrid(y_indices, x_indices, indexing='ij')

        # Create mask for ship pixels
        ship_mask = (
                (x_grid >= x) &
                (x_grid < x + self.constants.SHIP_WIDTH) &
                (y_grid >= y) &
                (y_grid < y + self.constants.SHIP_HEIGHT)
        )

        # Apply ship color where mask is True
        ship_color = jnp.array(self.constants.BLUE, dtype=jnp.uint8)
        screen = jnp.where(
            ship_mask[..., None],  # Add dimension for RGB
            ship_color,
            screen
        ).astype(jnp.uint8)

        return screen

    def _draw_torpedo_projectiles(self, screen: chex.Array, torpedo_projectiles: chex.Array) -> chex.Array:
        """Draw all active torpedo projectiles - vectorized for JIT"""

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
        """Draw all active sentinel projectiles - vectorized for JIT"""

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
        """Draw all active projectiles"""

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
        """Draw all active enemies"""

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

            # Select enemy color based on type
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
                                        jnp.array(self.constants.WHITE, dtype=jnp.uint8)  # Default white saucer
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
        """Draw UI elements - placeholder for now"""
        return screen

    # ============================================================================
    # PYGAME DISPLAY METHODS (moved from BeamRiderPygameRenderer)
    # ============================================================================

    """def run_game(self):
        Main game loop with torpedo support - requires pygame to be enabled
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
                prev_state = state  # Store previous state
                obs, state, reward, done, info = self.env.step(state, action)

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
       Show sector completion message (called when sector advances)
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
        pause_text = self.font.render("PAUSED", True, (255, 220, 100))
        rect = pause_text.get_rect(center=(self.pygame_screen_width // 2, self.pygame_screen_height // 2))
        self.pygame_screen.blit(pause_text, rect)

    def _draw_screen(self, screen_buffer, state):
        Draws the game screen buffer and overlays the ship sprite
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
        Show Game Over screen
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
        self.pygame_screen.blit(sector_text, sector_rect)"""


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