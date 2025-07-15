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
    BASE_ENEMY_SPAWN_INTERVAL = 60
    MIN_ENEMY_SPAWN_INTERVAL = 20  # Fastest spawn rate

    # Enemy spawn position
    ENEMY_SPAWN_Y = 10

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
    beam_position: int  # Which beam the ship is on (0-4)
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
    beam_position: int  # Which beam the enemy is on (0-4)
    active: bool
    speed: float = BeamRiderConstants.ENEMY_SPEED
    enemy_type: int = 0  # Different enemy types for variety


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
    rng_key: chex.PRNGKey
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
            x=self.beam_positions[initial_beam] - self.constants.SHIP_WIDTH // 2,
            y=self.constants.SCREEN_HEIGHT - self.constants.SHIP_BOTTOM_OFFSET,
            beam_position=initial_beam,
            active=True
        )

        # Initialize empty projectiles arrays (4 columns each)
        projectiles = jnp.zeros((self.constants.MAX_PROJECTILES, 4))  # x, y, active, speed
        torpedo_projectiles = jnp.zeros((self.constants.MAX_PROJECTILES, 4))  # x, y, active, speed

        # Initialize empty enemies array
        enemies = jnp.zeros((self.constants.MAX_ENEMIES, 6))  # x, y, beam_position, active, speed, type

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
        # Update frame count
        state = state.replace(frame_count=state.frame_count + 1)

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

        # Check sector progression (NEW)
        state = self._check_sector_progression(state)

        # Check game over conditions
        state = self._check_game_over(state)

        return state


    def _update_ship(self, state: BeamRiderState, action: int) -> BeamRiderState:
        """Update ship position based on action - moves between beams"""
        ship = state.ship

        # Handle beam movement using JAX conditionals
        new_beam_position = jnp.where(
            jnp.isin(action, jnp.array([1, 4, 7])),  # left, left+laser, left+torpedo
            jnp.maximum(0, ship.beam_position - 1),
            jnp.where(
                jnp.isin(action, jnp.array([2, 5, 8])),  # right, right+laser, right+torpedo
                jnp.minimum(self.constants.NUM_BEAMS - 1, ship.beam_position + 1),
                ship.beam_position  # no movement
            )
        )

        # Update ship position to match beam
        new_x = self.beam_positions[new_beam_position] - self.constants.SHIP_WIDTH // 2

        ship = ship.replace(
            x=new_x,
            beam_position=new_beam_position
        )

        return state.replace(ship=ship)

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
        active_mask = projectiles[:, 2] == 0  # inactive projectiles

        first_inactive = jnp.argmax(active_mask)
        can_fire = active_mask[first_inactive] & should_fire

        new_projectile = jnp.array([
            state.ship.x + self.constants.SHIP_WIDTH // 2,  # x
            state.ship.y,  # y
            1,  # active
            -self.constants.PROJECTILE_SPEED  # speed (negative = upward)
        ])

        projectiles = jnp.where(
            can_fire,
            projectiles.at[first_inactive].set(new_projectile),
            projectiles
        )

        return state.replace(projectiles=projectiles)

    def _fire_torpedo(self, state: BeamRiderState, should_fire: bool) -> BeamRiderState:
        """Fire torpedo projectile (if any remaining)"""
        torpedo_projectiles = state.torpedo_projectiles
        active_mask = torpedo_projectiles[:, 2] == 0  # inactive torpedoes

        first_inactive = jnp.argmax(active_mask)
        has_torpedoes = state.torpedoes_remaining > 0
        can_fire = active_mask[first_inactive] & should_fire & has_torpedoes

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
        active = (projectiles[:, 2] == 1) & (new_y > 0) & (new_y < self.constants.SCREEN_HEIGHT)
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
        """Spawn new enemies on random beams"""
        state = state.replace(enemy_spawn_timer=state.enemy_spawn_timer + 1)

        # Check if it's time to spawn an enemy
        should_spawn = state.enemy_spawn_timer >= state.enemy_spawn_interval

        # Reset spawn timer when spawning occurs
        new_spawn_timer = jnp.where(should_spawn, 0, state.enemy_spawn_timer)
        state = state.replace(enemy_spawn_timer=new_spawn_timer)

        # Find inactive enemy slot
        enemies = state.enemies
        active_mask = enemies[:, 3] == 0  # active column (now at index 3)

        # Generate new enemy on random beam
        rng_key, subkey = random.split(state.rng_key)
        spawn_beam = random.randint(subkey, (), 0, self.constants.NUM_BEAMS)
        spawn_x = self.beam_positions[spawn_beam] - self.constants.ENEMY_WIDTH // 2

        # Create new enemy data
        new_enemy = jnp.array([
            spawn_x,  # x
            self.constants.ENEMY_SPAWN_Y,  # y (spawn near top)
            spawn_beam,  # beam_position
            1,  # active
            self.constants.ENEMY_SPEED,  # speed
            0  # type
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

    def _update_enemies(self, state: BeamRiderState) -> BeamRiderState:
        """Update enemy positions - they move down their assigned beam"""
        enemies = state.enemies

        # Move enemies down their beam
        new_y = enemies[:, 1] + enemies[:, 4]  # y + speed (speed now at index 4)

        # Deactivate enemies that go off screen
        active = (enemies[:, 3] == 1) & (new_y < self.constants.SCREEN_HEIGHT)  # active now at index 3

        enemies = enemies.at[:, 1].set(new_y)
        enemies = enemies.at[:, 3].set(active.astype(jnp.float32))

        return state.replace(enemies=enemies)

    def _check_collisions(self, state: BeamRiderState) -> BeamRiderState:
        """Check for collisions between projectiles and enemies - JAX-compatible"""
        projectiles = state.projectiles
        torpedo_projectiles = state.torpedo_projectiles
        enemies = state.enemies
        score = state.score

        # Vectorized collision detection for LASER projectiles vs enemies
        proj_active = projectiles[:, 2] == 1  # active projectiles
        enemy_active = enemies[:, 3] == 1  # active enemies

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
                enemy_active[None, :]  # broadcast enemy active state
        )

        # Find collisions for laser projectiles
        laser_proj_hits = jnp.any(laser_collisions, axis=1)
        laser_enemy_hits = jnp.any(laser_collisions, axis=0)

        # Vectorized collision detection for TORPEDO projectiles vs enemies
        torpedo_active = torpedo_projectiles[:, 2] == 1  # active torpedoes
        torpedo_x = torpedo_projectiles[:, 0:1]  # shape (MAX_PROJECTILES, 1)
        torpedo_y = torpedo_projectiles[:, 1:2]  # shape (MAX_PROJECTILES, 1)

        # Vectorized bounding box collision check for torpedoes
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

        # Count total enemies killed this frame for sector progression
        enemies_killed_this_frame = jnp.sum(total_enemy_hits)

        # Update projectile and enemy states
        projectiles = projectiles.at[:, 2].set(projectiles[:, 2] * (~laser_proj_hits))
        torpedo_projectiles = torpedo_projectiles.at[:, 2].set(torpedo_projectiles[:, 2] * (~torpedo_proj_hits))
        enemies = enemies.at[:, 3].set(enemies[:, 3] * (~total_enemy_hits))

        # Update score (torpedoes give double points)
        laser_score = jnp.sum(laser_enemy_hits) * self.constants.POINTS_PER_ENEMY
        torpedo_score = jnp.sum(torpedo_enemy_hits) * self.constants.POINTS_PER_ENEMY * 2  # Double points
        score += laser_score + torpedo_score

        # Update enemy kill count for sector progression
        enemies_killed_this_sector = state.enemies_killed_this_sector + enemies_killed_this_frame

        # Check enemy-ship collisions (vectorized) - use original enemy_active
        ship_x, ship_y = state.ship.x, state.ship.y
        ship_collisions = (
                (ship_x < enemies[:, 0] + self.constants.ENEMY_WIDTH) &
                (ship_x + self.constants.SHIP_WIDTH > enemies[:, 0]) &
                (ship_y < enemies[:, 1] + self.constants.ENEMY_HEIGHT) &
                (ship_y + self.constants.SHIP_HEIGHT > enemies[:, 1]) &
                enemy_active  # Use the original enemy_active, before projectile collisions
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

        # Create updated ship struct
        ship = state.ship.replace(
            x=new_ship_x,
            beam_position=new_ship_beam
        )

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

        # Increase difficulty: spawn enemies faster in higher sectors
        # Spawn rate increases every 2 sectors, but never goes below minimum
        difficulty_factor = jnp.maximum(1, (new_sector - 1) // 2)
        new_spawn_interval = jnp.maximum(
            self.constants.MIN_ENEMY_SPAWN_INTERVAL,
            self.constants.BASE_ENEMY_SPAWN_INTERVAL - (difficulty_factor * 5)
        )

        # Only update spawn interval when sector changes
        spawn_interval = jnp.where(sector_complete, new_spawn_interval, state.enemy_spawn_interval)

        return state.replace(
            current_sector=new_sector,
            level=new_level,
            enemies_killed_this_sector=new_enemies_killed,
            torpedoes_remaining=new_torpedoes,
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

        # JIT-compile the render function
        self.render = jit(self._render_impl)

    def _render_impl(self, state: BeamRiderState) -> chex.Array:
        """Render the current game state to a screen buffer - JIT-compiled"""
        # Create screen buffer (RGB)
        screen = jnp.zeros((self.constants.SCREEN_HEIGHT, self.constants.SCREEN_WIDTH, 3), dtype=jnp.uint8)

        # Render the 5 beams first
        screen = self._draw_beams(screen)

        # Render ship
        screen = jnp.where(
            state.ship.active,
            self._draw_ship(screen, state.ship),
            screen
        )

        # Render projectiles (lasers)
        screen = self._draw_projectiles(screen, state.projectiles)

        # Render torpedo projectiles
        screen = self._draw_torpedo_projectiles(screen, state.torpedo_projectiles)

        # Render enemies
        screen = self._draw_enemies(screen, state.enemies)

        # Render UI (score, lives, torpedoes, sector progress)
        screen = self._draw_ui(screen, state)

        return screen

    def _draw_beams(self, screen: chex.Array) -> chex.Array:
        """Draw the 5 vertical beams"""

        # Use JAX-compatible loop instead of Python for loop
        def draw_single_beam(i, screen):
            beam_x = self.beam_positions[i].astype(int)

            # Create a mask for the beam pixels
            y_indices = jnp.arange(self.constants.SCREEN_HEIGHT)
            x_indices = jnp.arange(self.constants.SCREEN_WIDTH)

            # Create coordinate grids
            y_grid, x_grid = jnp.meshgrid(y_indices, x_indices, indexing='ij')

            # Create mask for beam pixels (2 pixels wide)
            beam_mask = (x_grid >= beam_x) & (x_grid < beam_x + 2)

            # Apply beam color where mask is True - ensure uint8 type
            beam_color = jnp.array([64, 64, 64], dtype=jnp.uint8)  # Dark gray
            screen = jnp.where(
                beam_mask[..., None],  # Add dimension for RGB
                beam_color,
                screen
            ).astype(jnp.uint8)  # Ensure output type matches input

            return screen

        # Draw all beams using JAX loop
        screen = jax.lax.fori_loop(0, self.constants.NUM_BEAMS, draw_single_beam, screen)
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
        """Draw all active enemies - vectorized for JIT"""

        # Vectorized drawing function
        def draw_single_enemy(i, screen):
            x, y = enemies[i, 0].astype(int), enemies[i, 1].astype(int)
            active = enemies[i, 3] == 1

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

            # Apply enemy color where mask is True
            enemy_color = jnp.array(self.constants.RED, dtype=jnp.uint8)
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
        """Draw UI elements (score, lives, torpedoes, sector progress, etc.)"""

        # Create coordinate grids for UI elements
        y_indices = jnp.arange(self.constants.SCREEN_HEIGHT)
        x_indices = jnp.arange(self.constants.SCREEN_WIDTH)
        y_grid, x_grid = jnp.meshgrid(y_indices, x_indices, indexing='ij')

        # Score indicator (top line)
        score_bars = jnp.minimum(state.score // 100, self.constants.SCREEN_WIDTH // 4)
        score_mask = (y_grid < 2) & (x_grid < score_bars)
        screen = jnp.where(
            score_mask[..., None],
            jnp.array(self.constants.GREEN, dtype=jnp.uint8),
            screen
        )

        # Lives indicator (second line)
        lives_bars = state.lives * 10
        lives_mask = (y_grid >= 2) & (y_grid < 4) & (x_grid < lives_bars)
        screen = jnp.where(
            lives_mask[..., None],
            jnp.array(self.constants.WHITE, dtype=jnp.uint8),
            screen
        )

        # Torpedo indicator (third line)
        torpedo_bars = state.torpedoes_remaining * 15
        torpedo_mask = (y_grid >= 4) & (y_grid < 6) & (x_grid < torpedo_bars)
        screen = jnp.where(
            torpedo_mask[..., None],
            jnp.array(self.constants.YELLOW, dtype=jnp.uint8),
            screen
        )

        # Enemy progress indicator (fourth line) - shows enemies remaining in sector
        enemies_remaining = self.constants.ENEMIES_PER_SECTOR - state.enemies_killed_this_sector
        progress_bars = enemies_remaining * 8  # Scale factor for visibility
        progress_mask = (y_grid >= 6) & (y_grid < 8) & (x_grid < progress_bars)
        screen = jnp.where(
            progress_mask[..., None],
            jnp.array(self.constants.RED, dtype=jnp.uint8),  # Red for remaining enemies
            screen
        )

        return screen

class BeamRiderPygameRenderer:
    """Pygame-basierte Visualisierung für BeamRider"""

    def __init__(self, scale=3):
        pygame.init()
        self.scale = scale
        self.screen_width = BeamRiderConstants.SCREEN_WIDTH * scale
        self.screen_height = BeamRiderConstants.SCREEN_HEIGHT * scale

        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("BeamRider - JAX Implementation")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)

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
        """Main game loop"""
        key = random.PRNGKey(42)
        state = self.env.reset(key)

        running = True
        action = 0

        while running and not state.game_over:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    action = self._get_action_from_key(event.key)

            # Update game state
            prev_state = state  # Store previous state
            state = self.env.step(state, action)

            # Check for sector completion
            self._show_sector_complete(state)

            # Render game
            screen_buffer = self.renderer.render(state)
            self._draw_screen(screen_buffer)

            # Draw UI
            self._draw_ui_overlay(state)

            pygame.display.flip()
            self.clock.tick(60)  # 60 FPS

            # Reset action to no-op after processing
            action = 0

        # Game over screen
        if state.game_over:
            self._show_game_over(state)

        pygame.quit()
        sys.exit()

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

    def _draw_screen(self, screen_buffer):
        """Zeichnet den Spielbildschirm"""
        # Konvertiere JAX Array zu NumPy
        screen_np = np.array(screen_buffer)

        # Skaliere das Bild
        scaled_screen = np.repeat(np.repeat(screen_np, self.scale, axis=0), self.scale, axis=1)

        # Erstelle pygame Surface
        surf = pygame.surfarray.make_surface(scaled_screen.swapaxes(0, 1))
        self.screen.blit(surf, (0, 0))

    def _draw_ui_overlay(self, state):
        """Draw UI overlay with text"""
        # Score
        score_text = self.font.render(f"Score: {state.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

        # Lives
        lives_text = self.font.render(f"Lives: {state.lives}", True, (255, 255, 255))
        self.screen.blit(lives_text, (10, 50))

        # Torpedoes remaining
        torpedoes_text = self.font.render(f"Torpedoes: {state.torpedoes_remaining}", True, (255, 255, 0))
        self.screen.blit(torpedoes_text, (10, 90))

        # Sector information
        sector_text = self.font.render(f"Sector: {state.current_sector}", True, (255, 255, 255))
        self.screen.blit(sector_text, (10, 130))

        # Enemy progress in current sector
        enemies_remaining = 15 - state.enemies_killed_this_sector
        progress_text = self.font.render(f"Enemies Left: {enemies_remaining}", True, (255, 100, 100))
        self.screen.blit(progress_text, (10, 170))

        # Controls
        controls_text = [
            "Controls:",
            "← → : Move",
            "Space: Fire Laser",
            "T: Fire Torpedo",
            "A/D: Move+Laser",
            "Q/E: Move+Torpedo"
        ]

        for i, text in enumerate(controls_text):
            rendered = self.font.render(text, True, (200, 200, 200))
            self.screen.blit(rendered, (self.screen_width - 200, 10 + i * 25))

    def _show_game_over(self, state):
        """Zeigt Game Over Bildschirm"""
        overlay = pygame.Surface((self.screen_width, self.screen_height))
        overlay.set_alpha(128)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))

        game_over_text = self.font.render("GAME OVER", True, (255, 0, 0))
        final_score_text = self.font.render(f"Final Score: {state.score}", True, (255, 255, 255))

        # Zentriere den Text
        game_over_rect = game_over_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2 - 30))
        score_rect = final_score_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2 + 30))

        self.screen.blit(game_over_text, game_over_rect)
        self.screen.blit(final_score_text, score_rect)

        pygame.display.flip()

        # Warte auf Eingabe
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                    waiting = False


if __name__ == "__main__":
    game = BeamRiderPygameRenderer()
    game.run_game()