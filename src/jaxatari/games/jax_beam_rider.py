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
- Add torpedo shooting -> done
- Add the rest of the enemies -> Orange Trackers and Sentinel Ship left
- Merge the renderers(try this)
----If the steps above are finished, ask for feedback----
- Add more movement Types for white enemies
- Difficulty Scaling
- Change enemy speeds
- Adjust points according to enemies -> done
- Documentation
- Should be playable through script
- Yellow Rejuvinators(?)
- Maybe some environment changes(depending on feedback)

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
    INITIAL_LIVES = 100
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
    BASE_ENEMY_SPAWN_INTERVAL = 60
    MIN_ENEMY_SPAWN_INTERVAL = 20  # Fastest spawn rate

    # Enemy spawn position
    ENEMY_SPAWN_Y = 10

    # Enemy types
    ENEMY_TYPE_WHITE_SAUCER = 0
    ENEMY_TYPE_BROWN_DEBRIS = 1
    ENEMY_TYPE_YELLOW_CHIRPER = 2
    ENEMY_TYPE_GREEN_BLOCKER = 3
    ENEMY_TYPE_GREEN_BOUNCE = 4  # NEW: Green bounce craft
    ENEMY_TYPE_BLUE_CHARGER = 5  # NEW: Blue charger
    ENEMY_TYPE_ORANGE_TRACKER = 6  # NEW: Orange tracker

    # Orange tracker specific constants
    ORANGE_TRACKER_SPEED = 1.2  # Tracking movement speed
    ORANGE_TRACKER_POINTS = 50  # Points when destroyed with torpedo
    ORANGE_TRACKER_COLOR = (255, 165, 0)  # Orange color RGB
    ORANGE_TRACKER_SPAWN_SECTOR = 1  # Starts appearing from sector 12
    ORANGE_TRACKER_SPAWN_CHANCE = 0.08  # 8% chance to spawn orange tracker
    ORANGE_TRACKER_CHANGE_DIRECTION_INTERVAL = 90  # Frames between direction changes
    # Blue charger specific constants
    BLUE_CHARGER_SPEED = 1.5  # Speed moving down beam
    BLUE_CHARGER_POINTS = 30  # Points when destroyed
    BLUE_CHARGER_COLOR = (0, 0, 255)  # Blue color RGB
    BLUE_CHARGER_SPAWN_SECTOR = 10  # Starts appearing from sector 10
    BLUE_CHARGER_SPAWN_CHANCE = 0.1  # 10% chance to spawn blue charger
    BLUE_CHARGER_LINGER_TIME = 180  # Frames to stay at bottom (3 seconds at 60fps)
    BLUE_CHARGER_DEFLECT_SPEED = -2.0  # Speed when deflected upward by laser
    # Brown debris specific constants
    BROWN_DEBRIS_SPEED = 1.5  # Slightly faster than regular enemies
    BROWN_DEBRIS_POINTS = 25  # Bonus points when destroyed with torpedo
    BROWN_DEBRIS_COLOR = (139, 69, 19)  # Brown color RGB

    # Spawn probabilities (add to existing constants)
    BROWN_DEBRIS_SPAWN_SECTOR = 2  # Starts appearing from sector 2
    BROWN_DEBRIS_SPAWN_CHANCE = 0.15  # 15% chance to spawn brown debris

    # Yellow chirper specific constants
    YELLOW_CHIRPER_SPEED = 1.0  # Horizontal movement speed
    YELLOW_CHIRPER_POINTS = 50  # Bonus points for shooting them
    YELLOW_CHIRPER_COLOR = (255, 255, 0)  # Yellow color RGB
    YELLOW_CHIRPER_SPAWN_Y_MIN = 50  # Minimum Y position for horizontal flight
    YELLOW_CHIRPER_SPAWN_Y_MAX = 150  # Maximum Y position for horizontal flight

    YELLOW_CHIRPER_SPAWN_SECTOR = 4  # Starts appearing from sector 4
    YELLOW_CHIRPER_SPAWN_CHANCE = 0.1  # 10% chance to spawn yellow chirper

    # Green blocker specific constants
    GREEN_BLOCKER_SPEED = 0.2  # Fast ramming speed
    GREEN_BLOCKER_POINTS = 75  # High points when destroyed
    GREEN_BLOCKER_COLOR = (0, 255, 0)  # Green color RGB
    GREEN_BLOCKER_SPAWN_Y_MIN = 30  # Spawn higher up for targeting
    GREEN_BLOCKER_SPAWN_Y_MAX = 80  # Range for side spawning
    GREEN_BLOCKER_LOCK_DISTANCE = 100  # Distance at which they lock onto player beam
    GREEN_BLOCKER_SPAWN_SECTOR = 6  # Starts appearing from sector 6
    GREEN_BLOCKER_SPAWN_CHANCE = 0.12  # 12% chance to spawn green blocker

    # Green bounce craft specific constants
    GREEN_BOUNCE_SPEED = 2.0  # Fast bouncing speed
    GREEN_BOUNCE_POINTS = 100  # Very high points when destroyed with torpedo
    GREEN_BOUNCE_COLOR = (0, 200, 0)  # Slightly different green than blockers
    GREEN_BOUNCE_SPAWN_SECTOR = 8  # Starts appearing from sector 8
    GREEN_BOUNCE_SPAWN_CHANCE = 0.08  # 8% chance to spawn green bounce craft
    GREEN_BOUNCE_MAX_BOUNCES = 6  # Maximum number of bounces before disappearing
    # HUD margins
    TOP_MARGIN = int(210 * 0.12)

    @classmethod
    def get_beam_positions(cls) -> jnp.ndarray:
        """Calculate beam positions based on screen width"""
        return jnp.array([
            cls.SCREEN_WIDTH // 8,  # Beam 0 (leftmost)
            cls.SCREEN_WIDTH // 4,  # Beam 1
            cls.SCREEN_WIDTH // 2,  # Beam 2 (center)
            3 * cls.SCREEN_WIDTH // 4,  # Beam 3
            7 * cls.SCREEN_WIDTH // 8  # Beam 4 (rightmost)
        ])


@struct.dataclass
class Ship:
    """Player ship state"""
    x: float
    y: float
    beam_position: int  # Which beam the ship is on (0-4) - FIXED: Added missing field
    active: bool = True


@struct.dataclass
class Projectile:
    """Player projectile state"""
    x: float
    y: float
    active: bool
    speed: float
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
    tracker_timer: int = 0  # NEW: Timer for orange tracker direction changes

@struct.dataclass
class BeamRiderState:
    """Complete game state"""
    # Game entities (no defaults)
    ship: Ship
    projectiles: chex.Array
    enemies: chex.Array

    # Game state (no defaults)
    score: int
    lives: int
    level: int
    game_over: bool
    # Random state
    rng_key: chex.PRNGKey
    # Timing and spawning
    frame_count: int
    enemy_spawn_timer: int

    # Torpedo system (no defaults)
    torpedoes_remaining: int
    torpedo_projectiles: chex.Array
    current_sector: int
    enemies_killed_this_sector: int

    # Fields WITH defaults (must come last)
    enemy_spawn_interval: int = BeamRiderConstants.ENEMY_SPAWN_INTERVAL


class BeamRiderEnv(JaxEnvironment[BeamRiderState, jnp.ndarray, dict, BeamRiderConstants]):
    """BeamRider environment following JAXAtari structure"""

    def __init__(self):
        self.constants = BeamRiderConstants()
        self.screen_width = self.constants.SCREEN_WIDTH
        self.screen_height = self.constants.SCREEN_HEIGHT
        self.action_space_size = 9  # Updated from 6 to 9 for torpedo actions
        self.beam_positions = self.constants.get_beam_positions()

        # JIT-compile the step function for performance
        self.step = jit(self._step_impl)

    def reset(self, rng_key: chex.PRNGKey) -> BeamRiderState:
        """Reset the game to initial state"""
        # Initialize ship at bottom center beam
        initial_beam = self.constants.INITIAL_BEAM
        ship = Ship(
            x=self.beam_positions[initial_beam] - self.constants.SHIP_WIDTH // 2,  # FIXED: Position correctly
            y=self.constants.SCREEN_HEIGHT - self.constants.SHIP_BOTTOM_OFFSET,
            beam_position=initial_beam,  # FIXED: Added missing beam_position
            active=True
        )

        # Initialize empty projectiles arrays (4 columns each)
        projectiles = jnp.zeros((self.constants.MAX_PROJECTILES, 4))  # x, y, active, speed
        torpedo_projectiles = jnp.zeros((self.constants.MAX_PROJECTILES, 4))  # x, y, active, speed

        # Initialize empty enemies array - UPDATED: now 11 columns for orange trackers
        enemies = jnp.zeros((self.constants.MAX_ENEMIES,
                             11))  # x, y, beam_position, active, speed, type, direction_x, direction_y, bounce_count, linger_timer, tracker_timer
        return BeamRiderState(
            ship=ship,
            projectiles=projectiles,
            enemies=enemies,
            torpedo_projectiles=torpedo_projectiles,
            score=0,
            lives=self.constants.INITIAL_LIVES,
            level=1,  # Start at sector 1 (kept for backward compatibility)
            game_over=False,
            frame_count=0,
            enemy_spawn_timer=0,
            current_sector=1,
            enemies_killed_this_sector=0,
            torpedoes_remaining=self.constants.TORPEDOES_PER_SECTOR,
            enemy_spawn_interval=self.constants.BASE_ENEMY_SPAWN_INTERVAL,
            rng_key=rng_key
        )

    def _step_impl(self, state: BeamRiderState, action: int) -> BeamRiderState:
        """Execute one game step - JIT-compiled implementation"""
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

        # Check collisions
        state = self._check_collisions(state)

        # Check sector progression
        state = self._check_sector_progression(state)

        # Check game over conditions
        state = self._check_game_over(state)

        # FIXED: Update frame count only once at the end
        state = state.replace(frame_count=state.frame_count + 1)

        return state

    def _update_ship(self, state: BeamRiderState, action: int) -> BeamRiderState:
        """Update ship position smoothly using left/right actions"""
        ship = state.ship
        speed = 1.5  # adjust this for faster/slower ship

        new_x = jnp.where(
            action == 1,  # left
            jnp.maximum(0, ship.x - speed),
            jnp.where(
                action == 2,  # right
                jnp.minimum(self.constants.SCREEN_WIDTH - self.constants.SHIP_WIDTH, ship.x + speed),
                ship.x  # no movement
            )
        )

        return state.replace(ship=ship.replace(x=new_x))

    def _handle_firing(self, state: BeamRiderState, action: int) -> BeamRiderState:
        """Handle both laser and torpedo firing"""

        # Laser firing (actions 3, 4, 5)
        should_fire_laser = jnp.isin(action, jnp.array([3, 4, 5]))
        state = self._fire_laser(state, should_fire_laser)

        # Torpedo firing (actions 6, 7, 8)
        should_fire_torpedo = jnp.isin(action, jnp.array([6, 7, 8]))
        state = self._fire_torpedo(state, should_fire_torpedo)

        return state

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
        """Fire torpedo projectile (if any remaining) - FIXED VERSION"""
        torpedo_projectiles = state.torpedo_projectiles

        # FIXED: Check if ANY torpedo slot is available (like laser logic)
        any_torpedo_active = jnp.any(torpedo_projectiles[:, 2] == 1)
        has_torpedoes = state.torpedoes_remaining > 0

        # FIXED: Allow firing if no torpedoes are active AND we have torpedoes remaining
        can_fire = ~any_torpedo_active & should_fire & has_torpedoes

        # Find first available (inactive) slot
        active_mask = torpedo_projectiles[:, 2] == 0  # inactive torpedoes
        first_inactive = jnp.argmax(active_mask)

        new_torpedo = jnp.array([
            state.ship.x + self.constants.SHIP_WIDTH // 2,  # x
            state.ship.y,  # y
            1,  # active
            -self.constants.TORPEDO_SPEED  # speed (faster than laser)
        ])

        torpedo_projectiles = jnp.where(
            can_fire,
            torpedo_projectiles.at[first_inactive].set(new_torpedo),
            torpedo_projectiles
        )

        # Decrease torpedo count when fired
        torpedoes_remaining = jnp.where(
            can_fire,
            state.torpedoes_remaining - 1,
            state.torpedoes_remaining
        )

        return state.replace(
            torpedo_projectiles=torpedo_projectiles,
            torpedoes_remaining=torpedoes_remaining
        )

    def _update_projectiles(self, state: BeamRiderState) -> BeamRiderState:
        """Update all projectiles (lasers and torpedoes)"""
        # Update regular projectiles
        projectiles = state.projectiles
        new_y = projectiles[:, 1] + projectiles[:, 3]  # y + speed

        # Deactivate projectiles that go off screen
        active = (
                (projectiles[:, 2] == 1) &
                (new_y > self.constants.TOP_MARGIN) &
                (new_y < self.constants.SCREEN_HEIGHT)
        )

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

    def _spawn_enemies(self, state: BeamRiderState) -> BeamRiderState:
        """Spawn new enemies on random beams - Updated with green bounce craft"""
        state = state.replace(enemy_spawn_timer=state.enemy_spawn_timer + 1)

        # Check if it's time to spawn an enemy
        should_spawn = state.enemy_spawn_timer >= state.enemy_spawn_interval

        # Reset spawn timer when spawning occurs
        new_spawn_timer = jnp.where(should_spawn, 0, state.enemy_spawn_timer)
        state = state.replace(enemy_spawn_timer=new_spawn_timer)

        # Find inactive enemy slot
        enemies = state.enemies
        active_mask = enemies[:, 3] == 0  # active column (now at index 3)


        # Generate spawn position based on enemy type
        rng_key, subkey1 = random.split(state.rng_key)
        rng_key, subkey2 = random.split(rng_key)
        rng_key, subkey3 = random.split(rng_key)
        rng_key, subkey4 = random.split(rng_key)
        rng_key, subkey5 = random.split(rng_key)
        rng_key, subkey6 = random.split(rng_key)

        tracker_dir_x = random.choice(subkey6, jnp.array([-1.0, 1.0]))
        tracker_dir_y = 1.0  # Always start moving down

        tracker_spawn_beam = random.randint(subkey5, (), 0, self.constants.NUM_BEAMS)
        tracker_spawn_x = self.beam_positions[tracker_spawn_beam] - self.constants.ENEMY_WIDTH // 2
        tracker_spawn_y = self.constants.ENEMY_SPAWN_Y
        # Determine enemy type first
        enemy_type = self._select_enemy_type(state.current_sector, subkey1)

        # Set spawn position based on enemy type
        # Regular enemies (white saucer, brown debris) spawn at top on random beam
        spawn_beam = random.randint(subkey2, (), 0, self.constants.NUM_BEAMS)
        regular_spawn_x = self.beam_positions[spawn_beam] - self.constants.ENEMY_WIDTH // 2
        regular_spawn_y = self.constants.ENEMY_SPAWN_Y

        # Yellow chirper spawns at random Y position on left or right side
        chirper_spawn_y = random.uniform(subkey3, (),
                                         minval=self.constants.YELLOW_CHIRPER_SPAWN_Y_MIN,
                                         maxval=self.constants.YELLOW_CHIRPER_SPAWN_Y_MAX,
                                         dtype=jnp.float32)
        # Randomly choose left or right side entrance
        chirper_direction = random.randint(subkey3, (), 0, 2)  # 0 = left to right, 1 = right to left
        chirper_spawn_x = jnp.where(
            chirper_direction == 0,
            -self.constants.ENEMY_WIDTH,  # Start from left side
            self.constants.SCREEN_WIDTH  # Start from right side
        )

        # Green blocker spawns from sides at random Y, will target player beam
        blocker_spawn_y = random.uniform(subkey4, (),
                                         minval=self.constants.GREEN_BLOCKER_SPAWN_Y_MIN,
                                         maxval=self.constants.GREEN_BLOCKER_SPAWN_Y_MAX,
                                         dtype=jnp.float32)
        blocker_direction = random.randint(subkey4, (), 0, 2)  # 0 = left to right, 1 = right to left
        blocker_spawn_x = jnp.where(
            blocker_direction == 0,
            -self.constants.ENEMY_WIDTH,  # Start from left side
            self.constants.SCREEN_WIDTH  # Start from right side
        )

        # Green bounce craft spawns at random position on screen edge
        bounce_spawn_side = random.randint(subkey5, (), 0, 4)  # 0=top, 1=right, 2=bottom, 3=left
        bounce_spawn_x = jnp.where(
            bounce_spawn_side == 0,  # top
            random.uniform(subkey5, (),
                           minval=0,
                           maxval=self.constants.SCREEN_WIDTH - self.constants.ENEMY_WIDTH,
                           dtype=jnp.float32),
            jnp.where(
                bounce_spawn_side == 1,  # right
                self.constants.SCREEN_WIDTH,
                jnp.where(
                    bounce_spawn_side == 2,  # bottom
                    random.uniform(subkey5, (),
                                   minval=0,
                                   maxval=self.constants.SCREEN_WIDTH - self.constants.ENEMY_WIDTH,
                                   dtype=jnp.float32),
                    -self.constants.ENEMY_WIDTH  # left
                )
            )
        )
        bounce_spawn_y = jnp.where(
            bounce_spawn_side == 0,  # top
            self.constants.TOP_MARGIN,
            jnp.where(
                bounce_spawn_side == 2,  # bottom
                self.constants.SCREEN_HEIGHT - self.constants.ENEMY_HEIGHT,
                random.uniform(subkey5, (),
                               minval=self.constants.TOP_MARGIN,
                               maxval=self.constants.SCREEN_HEIGHT - self.constants.ENEMY_HEIGHT,
                               dtype=jnp.float32)  # left/right sides
            )
        )

        # Set initial bounce directions
        bounce_dir_x = random.choice(subkey5, jnp.array([-1.0, 1.0]))
        bounce_dir_y = random.choice(subkey5, jnp.array([-1.0, 1.0]))
        # Blue charger spawns at top of random beam
        charger_spawn_beam = random.randint(subkey5, (), 0, self.constants.NUM_BEAMS)
        charger_spawn_x = self.beam_positions[charger_spawn_beam] - self.constants.ENEMY_WIDTH // 2
        charger_spawn_y = self.constants.ENEMY_SPAWN_Y

        spawn_x = jnp.where(
            enemy_type == self.constants.ENEMY_TYPE_YELLOW_CHIRPER,
            chirper_spawn_x,
            jnp.where(
                enemy_type == self.constants.ENEMY_TYPE_GREEN_BLOCKER,
                blocker_spawn_x,
                jnp.where(
                    enemy_type == self.constants.ENEMY_TYPE_GREEN_BOUNCE,
                    bounce_spawn_x,
                    jnp.where(
                        enemy_type == self.constants.ENEMY_TYPE_BLUE_CHARGER,
                        charger_spawn_x,
                        jnp.where(
                            enemy_type == self.constants.ENEMY_TYPE_ORANGE_TRACKER,
                            tracker_spawn_x,
                            regular_spawn_x
                        )
                    )
                )
            )
        )

        spawn_y = jnp.where(
            enemy_type == self.constants.ENEMY_TYPE_YELLOW_CHIRPER,
            chirper_spawn_y,
            jnp.where(
                enemy_type == self.constants.ENEMY_TYPE_GREEN_BLOCKER,
                blocker_spawn_y,
                jnp.where(
                    enemy_type == self.constants.ENEMY_TYPE_GREEN_BOUNCE,
                    bounce_spawn_y,
                    jnp.where(
                        enemy_type == self.constants.ENEMY_TYPE_BLUE_CHARGER,
                        charger_spawn_y,
                        jnp.where(
                            enemy_type == self.constants.ENEMY_TYPE_ORANGE_TRACKER,
                            tracker_spawn_y,
                            regular_spawn_y
                        )
                    )
                )
            )
        )

        # Set speed and direction based on enemy type
        regular_speed = jnp.where(
            enemy_type == self.constants.ENEMY_TYPE_BROWN_DEBRIS,
            self.constants.BROWN_DEBRIS_SPEED,
            self.constants.ENEMY_SPEED
        )

        # For chirper: positive speed = left to right, negative = right to left
        chirper_speed = jnp.where(
            chirper_direction == 0,
            self.constants.YELLOW_CHIRPER_SPEED,  # Left to right
            -self.constants.YELLOW_CHIRPER_SPEED  # Right to left
        )

        # For blocker: start with horizontal movement, will change to diagonal when locking
        blocker_speed = jnp.where(
            blocker_direction == 0,
            self.constants.GREEN_BLOCKER_SPEED,  # Left to right initially
            -self.constants.GREEN_BLOCKER_SPEED  # Right to left initially
        )

        # Update speed selection
        enemy_speed = jnp.where(
            enemy_type == self.constants.ENEMY_TYPE_BROWN_DEBRIS,
            self.constants.BROWN_DEBRIS_SPEED,
            jnp.where(
                enemy_type == self.constants.ENEMY_TYPE_YELLOW_CHIRPER,
                chirper_speed,
                jnp.where(
                    enemy_type == self.constants.ENEMY_TYPE_GREEN_BLOCKER,
                    blocker_speed,
                    jnp.where(
                        enemy_type == self.constants.ENEMY_TYPE_GREEN_BOUNCE,
                        self.constants.GREEN_BOUNCE_SPEED,
                        jnp.where(
                            enemy_type == self.constants.ENEMY_TYPE_BLUE_CHARGER,
                            self.constants.BLUE_CHARGER_SPEED,
                            jnp.where(
                                enemy_type == self.constants.ENEMY_TYPE_ORANGE_TRACKER,
                                self.constants.ORANGE_TRACKER_SPEED,
                                self.constants.ENEMY_SPEED
                            )
                        )
                    )
                )
            )
        )

        # Create new enemy data - UPDATED: now includes direction_x and direction_y for bounce craft
        # For blockers: beam_position stores direction (0=L->R, 1=R->L), speed stores velocity
        # For chirpers: beam_position stores direction, speed stores horizontal velocity
        # For bounce craft: beam_position unused, direction_x/y store bounce directions
        # For regular: beam_position stores actual beam, speed stores downward velocity
        # Update beam position selection
        enemy_beam_or_direction = jnp.where(
            enemy_type == self.constants.ENEMY_TYPE_YELLOW_CHIRPER,
            chirper_direction,
            jnp.where(
                enemy_type == self.constants.ENEMY_TYPE_GREEN_BLOCKER,
                blocker_direction,
                jnp.where(
                    enemy_type == self.constants.ENEMY_TYPE_GREEN_BOUNCE,
                    -1,  # Not used for bounce craft
                    jnp.where(
                        enemy_type == self.constants.ENEMY_TYPE_BLUE_CHARGER,
                        charger_spawn_beam,  # Blue chargers use actual beam
                        jnp.where(
                            enemy_type == self.constants.ENEMY_TYPE_ORANGE_TRACKER,
                            tracker_spawn_beam,  # Orange trackers start on a beam
                            spawn_beam
                        )
                    )
                )
            )
        )

        # Set initial bounce count for bounce craft
        initial_bounce_count = jnp.where(
            enemy_type == self.constants.ENEMY_TYPE_GREEN_BOUNCE,
            self.constants.GREEN_BOUNCE_MAX_BOUNCES,
            0  # Other enemy types don't use bounce count
        )

        new_enemy = jnp.array([
            spawn_x,  # x
            spawn_y,  # y
            enemy_beam_or_direction,  # beam_position (or direction for special enemies)
            1,  # active
            enemy_speed,  # speed (varies by type and direction)
            enemy_type,  # type
            jnp.where(enemy_type == self.constants.ENEMY_TYPE_ORANGE_TRACKER, tracker_dir_x, bounce_dir_x),
            # direction_x
            jnp.where(enemy_type == self.constants.ENEMY_TYPE_ORANGE_TRACKER, tracker_dir_y, bounce_dir_y),
            # direction_y
            initial_bounce_count,  # bounce_count (only used by bounce craft)
            0,  # linger_timer (only used by blue chargers)
            jnp.where(enemy_type == self.constants.ENEMY_TYPE_ORANGE_TRACKER,
                      self.constants.ORANGE_TRACKER_CHANGE_DIRECTION_INTERVAL, 0)  # tracker_timer
        ])

        # Find first available slot
        first_inactive = jnp.argmax(active_mask)
        can_spawn = active_mask[first_inactive] & should_spawn

        # Update enemies array conditionally
        enemies = jnp.where(
            can_spawn,
            enemies.at[first_inactive].set(new_enemy),
            enemies
        )

        return state.replace(enemies=enemies, rng_key=rng_key)

    def _select_enemy_type(self, sector: int, rng_key: chex.PRNGKey) -> int:
        """Select enemy type based on current sector using JAX conditionals - UPDATED: includes green bounce craft"""

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

    def _update_enemies(self, state: BeamRiderState) -> BeamRiderState:
        """Update enemy positions - Updated to handle green bounce craft bouncing behavior"""
        enemies = state.enemies

        # Handle different movement patterns based on enemy type
        enemy_types = enemies[:, 5]  # Get enemy types

        # Regular enemies (white saucer, brown debris) move vertically down
        # Regular enemies (white saucer, brown debris, blue chargers) move vertically down
        regular_enemy_mask = (enemy_types == self.constants.ENEMY_TYPE_WHITE_SAUCER) | (
                enemy_types == self.constants.ENEMY_TYPE_BROWN_DEBRIS) | (
                                     enemy_types == self.constants.ENEMY_TYPE_BLUE_CHARGER)
        regular_new_y = enemies[:, 1] + enemies[:, 4]  # y + speed

        # Yellow chirpers move horizontally
        chirper_mask = enemy_types == self.constants.ENEMY_TYPE_YELLOW_CHIRPER
        chirper_new_x = enemies[:, 0] + enemies[:, 4]  # x + speed (horizontal movement)

        # Green blockers: complex targeting behavior (same as before)
        blocker_mask = enemy_types == self.constants.ENEMY_TYPE_GREEN_BLOCKER

        # Get player ship position for targeting - FIXED: Use beam position from ship
        player_x = state.ship.x + self.constants.SHIP_WIDTH // 2
        # Blue chargers: move down beam, then linger at bottom
        # Blue chargers: move down beam, then linger at bottom
        charger_mask = enemy_types == self.constants.ENEMY_TYPE_BLUE_CHARGER
        charger_linger_timer = enemies[:, 9].astype(int)  # linger_timer column

        # Define bottom position where chargers should stop
        bottom_position = self.constants.SCREEN_HEIGHT - self.constants.ENEMY_HEIGHT - 10

        # Check if charger has reached or passed bottom position
        charger_reached_bottom = enemies[:, 1] >= bottom_position

        # Calculate new Y position: move down until bottom, then stay at bottom
        charger_new_y = jnp.where(
            charger_mask & charger_reached_bottom,
            bottom_position,  # Stay exactly at bottom position
            enemies[:, 1] + enemies[:, 4]  # Continue moving down
        )

        # Update linger timer logic:
        # - If just reached bottom (timer is 0 and at bottom), start timer
        # - If already at bottom with active timer, decrement it
        # - Otherwise, keep current timer
        new_linger_timer = jnp.where(
            charger_mask & charger_reached_bottom & (charger_linger_timer == 0),
            self.constants.BLUE_CHARGER_LINGER_TIME,  # Start lingering
            jnp.where(
                charger_mask & charger_reached_bottom & (charger_linger_timer > 0),
                charger_linger_timer - 1,  # Count down
                charger_linger_timer  # Keep current value
            )
        )

        # Move down until reaching bottom, then stay at bottom while timer > 0
        charger_new_y = jnp.where(
            charger_mask & charger_reached_bottom,
            jnp.minimum(enemies[:, 1], bottom_position),  # Clamp at bottom
            enemies[:, 1] + enemies[:, 4]  # Continue moving down
        )
        # Calculate blocker behavior
        blocker_x = enemies[:, 0]
        blocker_y = enemies[:, 1]
        blocker_direction = enemies[:, 2]  # 0 = L->R, 1 = R->L

        # Orange trackers: simple movement toward player
        tracker_mask = enemy_types == self.constants.ENEMY_TYPE_ORANGE_TRACKER
        tracker_timer = enemies[:, 10].astype(int)  # tracker_timer column

        # Get player position for tracking
        player_x = state.ship.x + self.constants.SHIP_WIDTH // 2
        player_y = state.ship.y

        # Calculate simple movement toward player
        dx_to_player = player_x - enemies[:, 0]
        dy_to_player = player_y - enemies[:, 1]

        # Check if tracker has reached bottom
        tracker_at_bottom = enemies[:, 1] >= (self.constants.SCREEN_HEIGHT - self.constants.ENEMY_HEIGHT)

        # Simple movement: move a step toward player (but stop if at bottom)
        move_x = jnp.where(
            tracker_mask & ~tracker_at_bottom,  # Only move if not at bottom
            jnp.sign(dx_to_player) * enemies[:, 4] * 0.7,  # Move toward player (slower)
            0.0
        )

        move_y = jnp.where(
            tracker_mask & ~tracker_at_bottom,  # Only move if not at bottom
            jnp.sign(dy_to_player) * enemies[:, 4] * 0.7,  # Move toward player (slower)
            0.0
        )

        # Calculate new tracker position
        tracker_new_x = enemies[:, 0] + move_x
        tracker_new_y = enemies[:, 1] + move_y

        # Decrement timer
        new_tracker_timer = jnp.where(
            tracker_mask,
            jnp.maximum(0, tracker_timer - 1),
            tracker_timer
        )

        # Simple direction tracking for array updates
        new_tracker_dir_x = jnp.where(tracker_mask, jnp.sign(dx_to_player), enemies[:, 6])
        new_tracker_dir_y = jnp.where(tracker_mask, jnp.sign(dy_to_player), enemies[:, 7])


        # Check if blocker should lock onto player beam (when close enough horizontally)
        distance_to_player = jnp.abs(blocker_x - player_x)
        should_lock = distance_to_player < self.constants.GREEN_BLOCKER_LOCK_DISTANCE

        # Calculate target position (player ship center)
        target_x = player_x

        # Calculate movement direction towards target
        dx = target_x - blocker_x
        dy = state.ship.y - blocker_y  # Move towards player ship Y

        # Normalize movement vector for consistent speed
        distance = jnp.sqrt(dx * dx + dy * dy)
        # Avoid division by zero
        safe_distance = jnp.maximum(distance, 1.0)

        # Calculate velocity components for diagonal movement
        velocity_x = (dx / safe_distance) * self.constants.GREEN_BLOCKER_SPEED
        velocity_y = (dy / safe_distance) * self.constants.GREEN_BLOCKER_SPEED

        # When locked: move diagonally towards player, when not locked: move horizontally
        blocker_new_x = jnp.where(
            should_lock & blocker_mask,
            blocker_x + velocity_x,  # Diagonal movement when locked
            blocker_x + enemies[:, 4]  # Horizontal movement when not locked
        )

        blocker_new_y = jnp.where(
            should_lock & blocker_mask,
            blocker_y + velocity_y,  # Diagonal movement when locked
            blocker_y  # Stay at same Y when moving horizontally
        )

        # GREEN BOUNCE CRAFT: bouncing behavior
        bounce_mask = enemy_types == self.constants.ENEMY_TYPE_GREEN_BOUNCE

        # Get current bounce directions
        bounce_dir_x = enemies[:, 6]  # direction_x
        bounce_dir_y = enemies[:, 7]  # direction_y

        # Calculate new positions based on current direction and speed
        bounce_new_x = enemies[:, 0] + (bounce_dir_x * enemies[:, 4])
        bounce_new_y = enemies[:, 1] + (bounce_dir_y * enemies[:, 4])

        # Check for bouncing off screen edges
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

        # Update positions based on enemy type
        new_x = jnp.where(
            chirper_mask,
            chirper_new_x,  # Chirpers move horizontally
            jnp.where(
                blocker_mask,
                blocker_new_x,  # Blockers use complex targeting
                jnp.where(
                    bounce_mask,
                    bounce_clamped_x,  # Bounce craft bounce around
                    jnp.where(
                        charger_mask,
                        enemies[:, 0],  # Blue chargers don't change X
                        jnp.where(
                            tracker_mask,
                            tracker_new_x,  # Orange trackers follow player
                            enemies[:, 0]  # Regular enemies don't change X
                        )
                    )
                )
            )
        )

        new_y = jnp.where(
            regular_enemy_mask & ~charger_mask & ~tracker_mask,  # Regular enemies but not chargers or trackers
            regular_new_y,  # Regular enemies move down
            jnp.where(
                blocker_mask,
                blocker_new_y,  # Blockers use targeting Y
                jnp.where(
                    bounce_mask,
                    bounce_clamped_y,  # Bounce craft bounce around
                    jnp.where(
                        charger_mask,
                        charger_new_y,  # Blue chargers use special logic
                        jnp.where(
                            tracker_mask,
                            jnp.where(
                                tracker_new_y >= self.constants.SCREEN_HEIGHT,  # If tracker reached bottom
                                enemies[:, 1],  # Don't update position
                                tracker_new_y  # Otherwise update position
                            ),
                            enemies[:, 1]  # Chirpers don't change Y
                        )
                    )
                )
            )
        )
        # Blue chargers: stay active until linger timer expires (but only if they've reached bottom)
        charger_active = (enemies[:, 3] == 1) & (
                ~charger_reached_bottom |  # Still moving down, stay active
                (charger_reached_bottom & (new_linger_timer > 0))  # At bottom but timer not expired
        )
        # Deactivate enemies that go off screen
        # Regular enemies: deactivate when they go below screen
        regular_active = (enemies[:, 3] == 1) & (regular_new_y < self.constants.SCREEN_HEIGHT)
        # Orange trackers: deactivate when they reach bottom of screen
        tracker_active = (enemies[:, 3] == 1) & ~tracker_at_bottom
        chirper_active = (enemies[:, 3] == 1) & (chirper_new_x > -self.constants.ENEMY_WIDTH) & (
                chirper_new_x < self.constants.SCREEN_WIDTH + self.constants.ENEMY_WIDTH)

        # Blockers: deactivate when they go off any edge OR reach bottom
        blocker_active = (enemies[:, 3] == 1) & \
                         (blocker_new_x > -self.constants.ENEMY_WIDTH) & \
                         (blocker_new_x < self.constants.SCREEN_WIDTH + self.constants.ENEMY_WIDTH) & \
                         (blocker_new_y > -self.constants.ENEMY_HEIGHT) & \
                         (blocker_new_y < self.constants.SCREEN_HEIGHT)  # Remove the + ENEMY_HEIGHT buffer

        # Bounce craft: stay active as long as they're bouncing (they never leave screen)
        # Bounce craft: stay active as long as they have bounces remaining
        bounce_active = (enemies[:, 3] == 1) & (new_bounce_count > 0)  # Deactivate when out of bounces
        # Combine active states based on enemy type
        active = jnp.where(
            regular_enemy_mask & ~charger_mask & ~tracker_mask,
            regular_active,
            jnp.where(
                chirper_mask,
                chirper_active,
                jnp.where(
                    blocker_mask,
                    blocker_active,
                    jnp.where(
                        bounce_mask,
                        bounce_active,
                        jnp.where(
                            charger_mask,
                            charger_active,
                            jnp.where(
                                tracker_mask,
                                tracker_active,
                                enemies[:, 3] == 1  # Default: stay active
                            )
                        )
                    )
                )
            )
        )

        # Update enemy array
        # Update enemy array
        enemies = enemies.at[:, 0].set(new_x)  # Update x positions
        enemies = enemies.at[:, 1].set(new_y)  # Update y positions
        enemies = enemies.at[:, 3].set(active.astype(jnp.float32))  # Update active states
        # Update direction arrays for both bounce craft AND trackers
        enemies = enemies.at[:, 6].set(
            jnp.where(bounce_mask, new_bounce_dir_x, new_tracker_dir_x))  # Update direction X
        enemies = enemies.at[:, 7].set(
            jnp.where(bounce_mask, new_bounce_dir_y, new_tracker_dir_y))  # Update direction Y
        enemies = enemies.at[:, 8].set(new_bounce_count)  # Update bounce count
        enemies = enemies.at[:, 9].set(new_linger_timer)  # Update linger timer (MAKE SURE THIS IS HERE)
        # Add tracker updates to enemy array updates
        enemies = enemies.at[:, 10].set(new_tracker_timer)  # Update tracker timer

        return state.replace(enemies=enemies)

    def _check_collisions(self, state: BeamRiderState) -> BeamRiderState:
        """Check for collisions between projectiles and enemies - Updated for green bounce craft"""
        projectiles = state.projectiles
        torpedo_projectiles = state.torpedo_projectiles
        enemies = state.enemies
        score = state.score

        # Vectorized collision detection for LASER projectiles vs enemies
        proj_active = projectiles[:, 2] == 1  # active projectiles
        enemy_active = enemies[:, 3] == 1  # active enemies

        # Brown debris, green blockers, and green bounce craft are immune to lasers
        # Only chirpers and white saucers are vulnerable
        # Brown debris, green blockers, and green bounce craft are immune to lasers
        # White saucers, chirpers, and blue chargers are vulnerable (chargers get deflected)
        enemy_vulnerable_to_lasers = (enemies[:, 5] == self.constants.ENEMY_TYPE_WHITE_SAUCER) | (
                enemies[:, 5] == self.constants.ENEMY_TYPE_YELLOW_CHIRPER) | (
                                             enemies[:, 5] == self.constants.ENEMY_TYPE_BLUE_CHARGER)

        # Broadcast projectile and enemy positions for vectorized collision check
        proj_x = projectiles[:, 0:1]  # shape (MAX_PROJECTILES, 1)
        proj_y = projectiles[:, 1:2]  # shape (MAX_PROJECTILES, 1)
        enemy_x = enemies[:, 0:1].T  # shape (1, MAX_ENEMIES)
        enemy_y = enemies[:, 1:2].T  # shape (1, MAX_ENEMIES)

        # Vectorized bounding box collision check for lasers
        laser_collisions = (
                (proj_x < enemy_x + self.constants.ENEMY_WIDTH) &
                (proj_x + self.constants.PROJECTILE_WIDTH > enemy_x) &
                (proj_y < enemy_y + self.constants.ENEMY_HEIGHT) &
                (proj_y + self.constants.PROJECTILE_HEIGHT > enemy_y) &
                proj_active[:, None] &  # broadcast projectile active state
                enemy_active[None, :] &  # broadcast enemy active state
                enemy_vulnerable_to_lasers[None, :]  # immunity check
        )
        # Find collisions for laser projectiles
        laser_proj_hits = jnp.any(laser_collisions, axis=1)
        laser_enemy_hits = jnp.any(laser_collisions, axis=0)

        # Special handling for blue charger deflection by lasers
        charger_laser_hits = laser_enemy_hits & (enemies[:, 5] == self.constants.ENEMY_TYPE_BLUE_CHARGER)

        # Deflect blue chargers instead of destroying them
        enemies = enemies.at[:, 4].set(  # Update speed (negative = upward)
            jnp.where(
                charger_laser_hits,
                self.constants.BLUE_CHARGER_DEFLECT_SPEED,  # Deflect upward
                enemies[:, 4]  # Keep original speed
            )
        )

        # Don't deactivate blue chargers hit by lasers (they get deflected instead)
        laser_enemy_hits = laser_enemy_hits & (enemies[:, 5] != self.constants.ENEMY_TYPE_BLUE_CHARGER)

        # Vectorized collision detection for TORPEDO projectiles vs enemies
        torpedo_active = torpedo_projectiles[:, 2] == 1  # active torpedoes
        torpedo_x = torpedo_projectiles[:, 0:1]  # shape (MAX_PROJECTILES, 1)
        torpedo_y = torpedo_projectiles[:, 1:2]  # shape (MAX_PROJECTILES, 1)

        # Vectorized bounding box collision check for torpedoes (can hit all enemy types)
        torpedo_collisions = (
                (torpedo_x < enemy_x + self.constants.ENEMY_WIDTH) &
                (torpedo_x + self.constants.TORPEDO_WIDTH > enemy_x) &
                (torpedo_y < enemy_y + self.constants.ENEMY_HEIGHT) &
                (torpedo_y + self.constants.TORPEDO_HEIGHT > enemy_y) &
                torpedo_active[:, None] &  # broadcast torpedo active state
                enemy_active[None, :]  # broadcast enemy active state
        )

        # Find collisions for torpedo projectiles
        torpedo_proj_hits = jnp.any(torpedo_collisions, axis=1)
        torpedo_enemy_hits = jnp.any(torpedo_collisions, axis=0)

        # Combine enemy hits from both laser and torpedo
        total_enemy_hits = laser_enemy_hits | torpedo_enemy_hits

        # Count only WHITE SAUCER kills for sector progression (brown debris and chirpers don't count)
        white_saucer_hits = total_enemy_hits & (enemies[:, 5] == self.constants.ENEMY_TYPE_WHITE_SAUCER)
        enemies_killed_this_frame = jnp.sum(white_saucer_hits)

        # Update projectile and enemy states
        projectiles = projectiles.at[:, 2].set(projectiles[:, 2] * (~laser_proj_hits))
        torpedo_projectiles = torpedo_projectiles.at[:, 2].set(torpedo_projectiles[:, 2] * (~torpedo_proj_hits))
        enemies = enemies.at[:, 3].set(enemies[:, 3] * (~total_enemy_hits))

        # Calculate score with different point values for different enemy types
        # Laser hits
        laser_white_saucer_hits = laser_enemy_hits & (enemies[:, 5] == self.constants.ENEMY_TYPE_WHITE_SAUCER)
        laser_chirper_hits = laser_enemy_hits & (enemies[:, 5] == self.constants.ENEMY_TYPE_YELLOW_CHIRPER)

        laser_score = (jnp.sum(laser_white_saucer_hits) * self.constants.POINTS_PER_ENEMY +
                       jnp.sum(laser_chirper_hits) * self.constants.YELLOW_CHIRPER_POINTS)

        # Torpedo hits
        torpedo_white_saucer_hits = torpedo_enemy_hits & (enemies[:, 5] == self.constants.ENEMY_TYPE_WHITE_SAUCER)
        torpedo_brown_debris_hits = torpedo_enemy_hits & (enemies[:, 5] == self.constants.ENEMY_TYPE_BROWN_DEBRIS)
        torpedo_chirper_hits = torpedo_enemy_hits & (enemies[:, 5] == self.constants.ENEMY_TYPE_YELLOW_CHIRPER)
        torpedo_blocker_hits = torpedo_enemy_hits & (enemies[:, 5] == self.constants.ENEMY_TYPE_GREEN_BLOCKER)
        torpedo_bounce_hits = torpedo_enemy_hits & (enemies[:, 5] == self.constants.ENEMY_TYPE_GREEN_BOUNCE)
        torpedo_blue_charger_hits = torpedo_enemy_hits & (enemies[:, 5] == self.constants.ENEMY_TYPE_BLUE_CHARGER)
        torpedo_orange_tracker_hits = torpedo_enemy_hits & (enemies[:, 5] == self.constants.ENEMY_TYPE_ORANGE_TRACKER)

        torpedo_score = (jnp.sum(torpedo_white_saucer_hits) * self.constants.POINTS_PER_ENEMY * 2 +
                         jnp.sum(torpedo_brown_debris_hits) * self.constants.BROWN_DEBRIS_POINTS +
                         jnp.sum(torpedo_chirper_hits) * self.constants.YELLOW_CHIRPER_POINTS +
                         jnp.sum(torpedo_blocker_hits) * self.constants.GREEN_BLOCKER_POINTS +
                         jnp.sum(torpedo_bounce_hits) * self.constants.GREEN_BOUNCE_POINTS +
                         jnp.sum(torpedo_blue_charger_hits) * self.constants.BLUE_CHARGER_POINTS +
                         jnp.sum(torpedo_orange_tracker_hits) * self.constants.ORANGE_TRACKER_POINTS)

        score += laser_score + torpedo_score

        # Update enemy kill count for sector progression
        enemies_killed_this_sector = state.enemies_killed_this_sector + enemies_killed_this_frame

        # Check enemy-ship collisions (vectorized) - YELLOW CHIRPERS CANNOT COLLIDE WITH SHIP
        ship_x, ship_y = state.ship.x, state.ship.y

        # Only non-chirper enemies can collide with ship (including green blockers and bounce craft)
        can_collide_with_ship = (enemies[:, 5] != self.constants.ENEMY_TYPE_YELLOW_CHIRPER)

        ship_collisions = (
                (ship_x < enemies[:, 0] + self.constants.ENEMY_WIDTH) &
                (ship_x + self.constants.SHIP_WIDTH > enemies[:, 0]) &
                (ship_y < enemies[:, 1] + self.constants.ENEMY_HEIGHT) &
                (ship_y + self.constants.SHIP_HEIGHT > enemies[:, 1]) &
                enemy_active &  # Use the original enemy_active, before projectile collisions
                can_collide_with_ship  # Chirpers cannot collide
        )

        ship_collision = jnp.any(ship_collisions)

        # Deactivate enemies that hit the ship
        enemies = enemies.at[:, 3].set(enemies[:, 3] * (~ship_collisions))

        # Handle ship collision - use conditional logic for struct updates
        lives = jnp.where(ship_collision, state.lives - 1, state.lives)

        # Update ship position conditionally using scalar values
        center_beam = self.constants.INITIAL_BEAM
        new_ship_x = jnp.where(
            ship_collision,
            self.beam_positions[center_beam] - self.constants.SHIP_WIDTH // 2,
            state.ship.x
        )
        new_ship_beam = jnp.where(
            ship_collision,
            center_beam,
            state.ship.beam_position
        )

        ship = state.ship.replace(x=new_ship_x, beam_position=new_ship_beam)

        return state.replace(
            projectiles=projectiles,
            torpedo_projectiles=torpedo_projectiles,
            enemies=enemies,
            score=score,
            ship=ship,
            lives=lives,
            enemies_killed_this_sector=enemies_killed_this_sector
        )

    def _check_sector_progression(self, state: BeamRiderState) -> BeamRiderState:
        """Check if sector is complete and advance to next sector"""

        # Check if we've killed enough enemies to complete the sector
        sector_complete = state.enemies_killed_this_sector >= self.constants.ENEMIES_PER_SECTOR

        # Calculate new values when sector is complete
        new_sector = jnp.where(sector_complete, state.current_sector + 1, state.current_sector)
        new_level = new_sector  # Keep level synced with sector for backward compatibility
        new_enemies_killed = jnp.where(sector_complete, 0, state.enemies_killed_this_sector)
        new_torpedoes = jnp.where(sector_complete, self.constants.TORPEDOES_PER_SECTOR, state.torpedoes_remaining)

        # Reset ship position to center beam when sector completes
        center_beam = self.constants.INITIAL_BEAM
        new_ship_x = jnp.where(
            sector_complete,
            self.beam_positions[center_beam] - self.constants.SHIP_WIDTH // 2,
            state.ship.x
        )
        new_ship_beam = jnp.where(
            sector_complete,
            center_beam,
            state.ship.beam_position
        )

        # Clear all projectiles when sector completes
        cleared_projectiles = jnp.where(
            sector_complete,
            jnp.zeros_like(state.projectiles),
            state.projectiles
        )
        cleared_torpedo_projectiles = jnp.where(
            sector_complete,
            jnp.zeros_like(state.torpedo_projectiles),
            state.torpedo_projectiles
        )

        # Clear all enemies when sector completes
        cleared_enemies = jnp.where(
            sector_complete,
            jnp.zeros_like(state.enemies),
            state.enemies
        )

        # Reset spawn timer when sector completes
        new_spawn_timer = jnp.where(sector_complete, 0, state.enemy_spawn_timer)

        # Increase difficulty: spawn enemies faster in higher sectors
        # Spawn rate increases every 2 sectors, but never goes below minimum
        difficulty_factor = jnp.maximum(1, (new_sector - 1) // 2)
        new_spawn_interval = jnp.maximum(
            self.constants.MIN_ENEMY_SPAWN_INTERVAL,
            self.constants.BASE_ENEMY_SPAWN_INTERVAL - (difficulty_factor * 5)
        )

        # Only update spawn interval when sector changes
        spawn_interval = jnp.where(sector_complete, new_spawn_interval, state.enemy_spawn_interval)

        # Create updated ship struct
        ship = state.ship.replace(
            x=new_ship_x,
            beam_position=new_ship_beam
        )

        return state.replace(
            ship=ship,
            projectiles=cleared_projectiles,
            torpedo_projectiles=cleared_torpedo_projectiles,
            enemies=cleared_enemies,
            current_sector=new_sector,
            level=new_level,
            enemies_killed_this_sector=new_enemies_killed,
            torpedoes_remaining=new_torpedoes,
            enemy_spawn_timer=new_spawn_timer,
            enemy_spawn_interval=spawn_interval
        )

    def _check_game_over(self, state: BeamRiderState) -> BeamRiderState:
        """Check if game is over"""
        game_over = state.lives <= 0
        return state.replace(game_over=game_over)


class BeamRiderRenderer(JAXGameRenderer):
    """Renderer for BeamRider game"""

    def __init__(self):
        self.constants = BeamRiderConstants()
        self.screen_width = self.constants.SCREEN_WIDTH
        self.screen_height = self.constants.SCREEN_HEIGHT
        self.beam_positions = self.constants.get_beam_positions()
        self.ship_sprite_surface = self._create_ship_surface()
        self.small_ship_surface = self._create_small_ship_surface()

        # JIT-compile the render function
        self.render = jit(self._render_impl)

    def _create_ship_surface(self):
        # Pixel values: 0=transparent, 1=yellow, 2=purple
        ship_sprite = np.array([
            [0, 0, 0, 2, 2, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])
        colors = {
            0: (0, 0, 0, 0),  # transparent
            1: (255, 255, 0, 255),  # yellow
            2: (160, 32, 240, 255),  # purple
        }
        h, w = ship_sprite.shape
        surface = pygame.Surface((w, h), pygame.SRCALPHA)
        for y in range(h):
            for x in range(w):
                surface.set_at((x, y), colors[ship_sprite[y, x]])
        return pygame.transform.scale(surface, (w * 6, h * 6))

    def _create_small_ship_surface(self):
        """Creates a small version of the ship sprite for UI (lives display)"""
        small_sprite = pygame.transform.scale(self.ship_sprite_surface, (16, 10))  # adjust size as needed
        return small_sprite

    def _render_impl(self, state: BeamRiderState) -> chex.Array:
        """Render the current game state to a screen buffer - JIT-compiled"""
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

        # Render enemies
        screen = self._draw_enemies(screen, state.enemies)

        # Render UI (score, lives, torpedoes, sector progress)
        screen = self._draw_ui(screen, state)

        return screen

    def _draw_3d_grid(self, screen: chex.Array, frame_count: int) -> chex.Array:
        """Draw 3D grid with 7 animated horizontal lines and 9 vertical beam positions, skipping 2 & 8"""

        height = self.constants.SCREEN_HEIGHT
        width = self.constants.SCREEN_WIDTH
        line_color = jnp.array([64, 64, 255], dtype=jnp.uint8)

        # === Margins for HUD (top) and player (bottom) ===
        top_margin = int(height * 0.12)
        bottom_margin = int(height * 0.14)
        grid_height = height - top_margin - bottom_margin

        y_indices = jnp.arange(height)
        x_indices = jnp.arange(width)
        y_grid, x_grid = jnp.meshgrid(y_indices, x_indices, indexing="ij")

        # === Horizontal lines ===
        num_hlines = 7
        speed = 1  # pixels per frame
        spacing = grid_height // (num_hlines + 1)
        phase = (frame_count * 0.003) % 1.0  # Controls global animation phase

        def draw_hline(i, scr):
            t = (phase + i / num_hlines) % 1.0
            y = jnp.round((t ** 3.0) * grid_height).astype(int) + top_margin
            y = jnp.clip(y, 0, height - 1)
            mask = y_grid == y
            return jnp.where(mask[..., None], line_color, scr)

        screen = jax.lax.fori_loop(0, num_hlines, draw_hline, screen)

        # === Vertical lines (9 positions, skip 1 and 7) ===
        total_beams = 9
        rel_positions = jnp.linspace(-1.0, 1.0, total_beams)  # full spread
        draw_indices = jnp.array([0, 2, 3, 4, 5, 6, 8])  # skip index 1 and 7 (2nd and 8th from left)

        center_x = width / 2
        bottom_spread = width * 1.6
        y0 = height - bottom_margin
        y1 = -height * 0.7  # vanishing point above screen

        def draw_vline(i, scr):
            idx = draw_indices[i]
            rel = rel_positions[idx]
            x0 = center_x + rel * (bottom_spread / 2.0)
            x1 = center_x

            # Compute upper limit in t where y reaches top_margin
            t_top = (top_margin - y0) / (y1 - y0)
            t_top = jnp.clip(t_top, 0.0, 1.0)  # prevent overflow

            num_steps = 200
            dot_spacing = 25

            def body_fn(j, scr_inner):
                t = j / (num_steps - 1)
                t_clipped = t * t_top  # scale to [0, t_top]

                y = y0 + (y1 - y0) * t_clipped
                x = x0 + (x1 - x0) * t_clipped

                xi = jnp.clip(jnp.round(x).astype(int), 0, width - 1)
                yi = jnp.clip(jnp.round(y).astype(int), 0, height - 1)

                return jax.lax.cond(
                    j % dot_spacing == 0,
                    lambda s: s.at[yi, xi].set(line_color),
                    lambda s: s,
                    scr_inner
                )

            return jax.lax.fori_loop(0, num_steps, body_fn, scr)

        screen = jax.lax.fori_loop(0, draw_indices.shape[0], draw_vline, screen)

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

    def _draw_projectiles(self, screen: chex.Array, projectiles: chex.Array) -> chex.Array:
        """Draw all active projectiles - vectorized for JIT"""

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
        """Draw all active enemies - vectorized for JIT with enemy type support including green bounce craft"""

        # Vectorized drawing function
        def draw_single_enemy(i, screen):
            x, y = enemies[i, 0].astype(int), enemies[i, 1].astype(int)
            active = enemies[i, 3] == 1
            enemy_type = enemies[i, 5].astype(int)

            # Create coordinate grids
            y_indices = jnp.arange(self.constants.SCREEN_HEIGHT)
            x_indices = jnp.arange(self.constants.SCREEN_WIDTH)
            y_grid, x_grid = jnp.meshgrid(y_indices, x_indices, indexing='ij')

            # Create mask for enemy pixels
            enemy_mask = (
                    (x_grid >= x) &
                    (x_grid < x + self.constants.ENEMY_WIDTH) &
                    (y_grid >= y) &
                    (y_grid < y + self.constants.ENEMY_HEIGHT) &
                    active &
                    (x >= 0) & (x < self.constants.SCREEN_WIDTH) &
                    (y >= 0) & (y < self.constants.SCREEN_HEIGHT)
            )

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
                                    jnp.array(self.constants.WHITE, dtype=jnp.uint8)  # Default white saucer color
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


class BeamRiderPygameRenderer:
    """Pygame-based visualization for BeamRider"""

    def __init__(self, scale=3):
        pygame.init()
        self.scale = scale
        self.screen_width = BeamRiderConstants.SCREEN_WIDTH * scale
        self.screen_height = BeamRiderConstants.SCREEN_HEIGHT * scale

        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("BeamRider - JAX Implementation")

        self.clock = pygame.time.Clock()
        # FIXED: Use default pygame font instead of missing font file
        self.font = pygame.font.Font(None, 16)

        # Create BeamRider components
        self.env = BeamRiderEnv()
        self.renderer = BeamRiderRenderer()

    def _show_sector_complete(self, state):
        """Show sector completion message (called when sector advances)"""
        if hasattr(self, '_last_sector') and state.current_sector > self._last_sector:
            # Sector just advanced - show visual feedback

            # Create semi-transparent overlay
            overlay = pygame.Surface((self.screen_width, self.screen_height))
            overlay.set_alpha(180)
            overlay.fill((0, 0, 50))  # Dark blue overlay
            self.screen.blit(overlay, (0, 0))

            # Sector completion message
            sector_complete_text = self.font.render("SECTOR COMPLETE!", True, (0, 255, 0))
            next_sector_text = self.font.render(f"Advancing to Sector {state.current_sector}", True, (255, 255, 255))
            torpedoes_text = self.font.render("Torpedoes Refilled!", True, (255, 255, 0))

            # Center the messages
            sector_rect = sector_complete_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2 - 40))
            next_rect = next_sector_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
            torpedo_rect = torpedoes_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2 + 40))

            # Draw the messages
            self.screen.blit(sector_complete_text, sector_rect)
            self.screen.blit(next_sector_text, next_rect)
            self.screen.blit(torpedoes_text, torpedo_rect)

            # Show for 2 seconds
            pygame.display.flip()
            pygame.time.wait(2000)

        # Update tracked sector
        self._last_sector = state.current_sector

    def run_game(self):
        """Main game loop with torpedo support"""
        key = random.PRNGKey(42)
        state = self.env.reset(key)

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
                if keys[pygame.K_t]:  # T for torpedo only
                    action = 6
                elif keys[pygame.K_q]:  # Q for left + torpedo
                    action = 7
                elif keys[pygame.K_e]:  # E for right + torpedo
                    action = 8
                # LASER ACTIONS (actions 3, 4, 5)
                elif keys[pygame.K_LEFT] and keys[pygame.K_SPACE]:
                    action = 4  # left + fire laser
                elif keys[pygame.K_RIGHT] and keys[pygame.K_SPACE]:
                    action = 5  # right + fire laser
                elif keys[pygame.K_SPACE]:
                    action = 3  # fire laser only
                # MOVEMENT ACTIONS (actions 1, 2)
                elif keys[pygame.K_LEFT]:
                    action = 1  # left
                elif keys[pygame.K_RIGHT]:
                    action = 2  # right

                # Step and render
                prev_state = state  # Store previous state
                state = self.env.step(state, action)

                # Check for sector completion
                self._show_sector_complete(state)
                screen_buffer = self.renderer.render(state)
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

    def _draw_pause_overlay(self):
        pause_text = self.font.render("PAUSED", True, (255, 220, 100))
        rect = pause_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
        self.screen.blit(pause_text, rect)

    def _get_action_from_key(self, key):
        """Convert keyboard inputs to actions"""
        if key == pygame.K_LEFT:
            return 1  # left
        elif key == pygame.K_RIGHT:
            return 2  # right
        elif key == pygame.K_SPACE:
            return 3  # fire laser
        elif key == pygame.K_a:  # A for left+laser
            return 4
        elif key == pygame.K_d:  # D for right+laser
            return 5
        elif key == pygame.K_t:  # T for torpedo
            return 6  # fire torpedo
        elif key == pygame.K_q:  # Q for left+torpedo
            return 7
        elif key == pygame.K_e:  # E for right+torpedo
            return 8
        return 0  # no-op

    def _draw_screen(self, screen_buffer, state):
        """Draws the game screen buffer and overlays the ship sprite"""
        screen_np = np.array(screen_buffer)
        scaled_screen = np.repeat(np.repeat(screen_np, self.scale, axis=0), self.scale, axis=1)

        surf = pygame.surfarray.make_surface(scaled_screen.swapaxes(0, 1))
        self.screen.blit(surf, (0, 0))

        # === OVERLAY THE SHIP SPRITE ===
        ship_x = int(state.ship.x) * self.scale
        ship_y = int(state.ship.y) * self.scale
        self.screen.blit(self.renderer.ship_sprite_surface, (ship_x, ship_y))

    def _draw_ui_overlay(self, state):
        """Draw centered Score and Level UI like Atari - UPDATED: shows green bounce craft info"""
        score_text = self.font.render(f"SCORE {state.score:06}", True, (255, 220, 100))  # padded 6-digit score
        level_text = self.font.render(f"SECTOR {state.level:02}", True, (255, 220, 100))  # padded 2-digit sector

        score_rect = score_text.get_rect(center=(self.screen_width // 2, 20))
        level_rect = level_text.get_rect(center=(self.screen_width // 2, 42))

        self.screen.blit(score_text, score_rect)
        self.screen.blit(level_text, level_rect)

        # Torpedoes remaining
        torpedoes_text = self.font.render(f"Torpedoes: {state.torpedoes_remaining}", True, (255, 255, 0))
        self.screen.blit(torpedoes_text, (10, 90))

        # Enemy progress in current sector
        enemies_remaining = 15 - state.enemies_killed_this_sector
        progress_text = self.font.render(f"Enemies Left: {enemies_remaining}", True, (255, 100, 100))
        self.screen.blit(progress_text, (10, 170))

        # NEW: Show when green bounce craft become available
        if state.current_sector >= BeamRiderConstants.GREEN_BOUNCE_SPAWN_SECTOR:
            bounce_text = self.font.render("Green Bounce Craft Active!", True, (0, 255, 0))
            self.screen.blit(bounce_text, (10, 190))
        elif state.current_sector == BeamRiderConstants.GREEN_BOUNCE_SPAWN_SECTOR - 1:
            next_text = self.font.render("Green Bounce Craft in Next Sector!", True, (255, 165, 0))
            self.screen.blit(next_text, (10, 190))

        # === DRAW LIVES INDICATORS ===
        for i in range(state.lives):
            x = 30 + i * 36  # spacing between icons
            y = self.screen_height - 20  # near bottom
            scaled_ship = pygame.transform.scale(self.renderer.small_ship_surface,
                                                 (int(self.renderer.small_ship_surface.get_width() * 1.5),
                                                  int(self.renderer.small_ship_surface.get_height() * 1.5)))
            self.screen.blit(scaled_ship, (x, y))

    def _show_game_over(self, state):
        """Show Game Over screen"""
        overlay = pygame.Surface((self.screen_width, self.screen_height))
        overlay.set_alpha(128)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))

        game_over_text = self.font.render("GAME OVER", True, (255, 0, 0))
        final_score_text = self.font.render(f"Final Score: {state.score}", True, (255, 255, 255))
        sector_text = self.font.render(f"Reached Sector: {state.current_sector}", True, (255, 255, 255))

        # Center the text
        game_over_rect = game_over_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2 - 40))
        score_rect = final_score_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
        sector_rect = sector_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2 + 30))

        self.screen.blit(game_over_text, game_over_rect)
        self.screen.blit(final_score_text, score_rect)
        self.screen.blit(sector_text, sector_rect)

        pygame.display.flip()

        # Wait for input
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                    waiting = False


if __name__ == "__main__":
    game = BeamRiderPygameRenderer()
    game.run_game()