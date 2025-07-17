import pygame
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit
from typing import Tuple, Dict, Any
import chex
from flax import struct
import sys

from jaxatari.environment import JaxEnvironment


class GameConstants:
    """Container for all game constants"""

    # Screen dimensions
    SCREEN_WIDTH = 240
    SCREEN_HEIGHT = 160

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

    # Enemy spawn position
    ENEMY_SPAWN_Y = 10

    # HUD margins
    TOP_MARGIN = int(210* 0.12)


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
    #beam_position: int  # Which beam the ship is on (0-4)
    active: bool = True


@struct.dataclass
class Projectile:
    """Player projectile state"""
    x: float
    y: float
    active: bool
    speed: float = GameConstants.PROJECTILE_SPEED


@struct.dataclass
class Enemy:
    """Enemy ship state"""
    x: float
    y: float
    beam_position: int  # Which beam the enemy is on (0-4)
    active: bool
    speed: float = GameConstants.ENEMY_SPEED
    enemy_type: int = 0  # Different enemy types for variety


@struct.dataclass
class BeamRiderState:
    """Complete game state"""
    # Game entities
    ship: Ship
    projectiles: chex.Array  # Array of Projectile structs
    enemies: chex.Array  # Array of Enemy structs

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
    enemy_spawn_interval: int = GameConstants.ENEMY_SPAWN_INTERVAL


class BeamRiderEnv:
    """BeamRider environment following JAXAtari structure"""

    def __init__(self):
        self.constants = GameConstants()
        self.screen_width = self.constants.SCREEN_WIDTH
        self.screen_height = self.constants.SCREEN_HEIGHT
        self.action_space_size = 6  # 0: no-op, 1: left, 2: right, 3: fire, 4: left+fire, 5: right+fire
        self.beam_positions = self.constants.get_beam_positions()

        # JIT-compile the step function for performance
        self.step = jit(self._step_impl)

    def reset(self, rng_key: chex.PRNGKey) -> BeamRiderState:
        """Reset the game to initial state"""
        # Initialize ship at bottom center beam
        initial_beam = self.constants.INITIAL_BEAM
        ship = Ship(
        x=self.constants.SCREEN_WIDTH // 2 - self.constants.SHIP_WIDTH // 2,
        y=self.constants.SCREEN_HEIGHT - self.constants.SHIP_BOTTOM_OFFSET,
        active=True
    )

        # Initialize empty projectiles array
        projectiles = jnp.zeros((self.constants.MAX_PROJECTILES, 4))  # x, y, active, speed

        # Initialize empty enemies array - now with beam_position
        enemies = jnp.zeros((self.constants.MAX_ENEMIES, 6))  # x, y, beam_position, active, speed, type

        return BeamRiderState(
            ship=ship,
            projectiles=projectiles,
            enemies=enemies,
            score=0,
            lives=self.constants.INITIAL_LIVES,
            level=self.constants.INITIAL_LEVEL,
            game_over=False,
            frame_count=0,
            enemy_spawn_timer=0,
            enemy_spawn_interval=self.constants.ENEMY_SPAWN_INTERVAL,
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

        # Check game over conditions
        state = self._check_game_over(state)

        state = state.replace(
            frame_count=state.frame_count + 1,
        )

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
        """Allow firing only when no other projectile is active"""

        # Check if fire key is pressed
        should_fire = jnp.isin(action, jnp.array([3, 4, 5]))  # fire, left+fire, right+fire

        # Check if all projectiles are inactive (3rd value == 0)
        projectiles = state.projectiles
        any_active = jnp.any(projectiles[:, 2] == 1)
        can_fire = ~any_active & should_fire  # only fire if none are active

        # New projectile to be fired
        new_projectile = jnp.array([
            state.ship.x + self.constants.SHIP_WIDTH // 2,  # x
            state.ship.y,                                   # y
            1,                                              # active
            -self.constants.PROJECTILE_SPEED               # speed
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



    def _update_projectiles(self, state: BeamRiderState) -> BeamRiderState:
        """Update all projectiles"""
        projectiles = state.projectiles

        # Update y position for active projectiles
        new_y = projectiles[:, 1] + projectiles[:, 3]  # y + speed

        # Deactivate projectiles that go off screen
        active = (
            (projectiles[:, 2] == 1) &
            (new_y > self.constants.TOP_MARGIN) &
            (new_y < self.constants.SCREEN_HEIGHT)
        )

        projectiles = projectiles.at[:, 1].set(new_y)
        projectiles = projectiles.at[:, 2].set(active.astype(jnp.float32))

        return state.replace(projectiles=projectiles)

    def _spawn_enemies(self, state: BeamRiderState) -> BeamRiderState:
        """Spawn new enemies on random beams"""
        state = state.replace(enemy_spawn_timer=state.enemy_spawn_timer + 1)

        # Check if it's time to spawn an enemy - JAX-kompatible Version
        should_spawn = state.enemy_spawn_timer >= state.enemy_spawn_interval

        # Reset spawn timer wenn spawning erfolgt
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
        enemies = state.enemies
        score = state.score

        # Vectorized collision detection for projectiles vs enemies
        proj_active = projectiles[:, 2] == 1  # active projectiles
        enemy_active = enemies[:, 3] == 1  # active enemies

        # Broadcast projectile and enemy positions for vectorized collision check
        proj_x = projectiles[:, 0:1]  # shape (MAX_PROJECTILES, 1)
        proj_y = projectiles[:, 1:2]  # shape (MAX_PROJECTILES, 1)
        enemy_x = enemies[:, 0:1].T  # shape (1, MAX_ENEMIES)
        enemy_y = enemies[:, 1:2].T  # shape (1, MAX_ENEMIES)

        # Vectorized bounding box collision check
        collisions = (
                (proj_x < enemy_x + self.constants.ENEMY_WIDTH) &
                (proj_x + self.constants.PROJECTILE_WIDTH > enemy_x) &
                (proj_y < enemy_y + self.constants.ENEMY_HEIGHT) &
                (proj_y + self.constants.PROJECTILE_HEIGHT > enemy_y) &
                proj_active[:, None] &  # broadcast projectile active state
                enemy_active[None, :]  # broadcast enemy active state
        )

        # Find first collision for each projectile
        proj_hits = jnp.any(collisions, axis=1)
        enemy_hits = jnp.any(collisions, axis=0)

        # Update projectile and enemy states
        projectiles = projectiles.at[:, 2].set(projectiles[:, 2] * (~proj_hits))
        enemies = enemies.at[:, 3].set(enemies[:, 3] * (~enemy_hits))

        # Update score
        score += jnp.sum(enemy_hits) * self.constants.POINTS_PER_ENEMY

        # Check enemy-ship collisions (vectorized)
        ship_x, ship_y = state.ship.x, state.ship.y
        ship_collisions = (
                (ship_x < enemies[:, 0] + self.constants.ENEMY_WIDTH) &
                (ship_x + self.constants.SHIP_WIDTH > enemies[:, 0]) &
                (ship_y < enemies[:, 1] + self.constants.ENEMY_HEIGHT) &
                (ship_y + self.constants.SHIP_HEIGHT > enemies[:, 1]) &
                enemy_active
        )

        ship_collision = jnp.any(ship_collisions)

        # Handle ship collision - use conditional logic for struct updates
        lives = jnp.where(ship_collision, state.lives - 1, state.lives)

        # Update ship position conditionally using scalar values
        center_beam = self.constants.INITIAL_BEAM
        new_ship_x = jnp.where(
            ship_collision,
            self.beam_positions[center_beam] - self.constants.SHIP_WIDTH // 2,
            state.ship.x
        )
        """ new_ship_beam = jnp.where(
            ship_collision,
            center_beam,
            state.ship.beam_position
        ) """

        ship = state.ship.replace(x=new_ship_x)


        return state.replace(
            projectiles=projectiles,
            enemies=enemies,
            score=score,
            ship=ship,
            lives=lives
        )

    def _check_game_over(self, state: BeamRiderState) -> BeamRiderState:
        """Check if game is over"""
        game_over = state.lives <= 0
        return state.replace(game_over=game_over)


class BeamRiderRenderer:
    """Renderer for BeamRider game"""

    def __init__(self):
        self.constants = GameConstants()
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
            0: (0, 0, 0, 0),           # transparent
            1: (255, 255, 0, 255),     # yellow
            2: (160, 32, 240, 255),    # purple
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


        """ # Render ship
        screen = jnp.where(
            state.ship.active,
            self._draw_ship(screen, state.ship),
            screen
        ) """

        # Render projectiles
        screen = self._draw_projectiles(screen, state.projectiles)

        # Render enemies
        screen = self._draw_enemies(screen, state.enemies)

        # Render UI (score, lives)
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
        return screen


class BeamRiderPygameRenderer:
    """Pygame-basierte Visualisierung für BeamRider"""

    def __init__(self, scale=3):
        pygame.init()
        self.scale = scale
        self.screen_width = GameConstants.SCREEN_WIDTH * scale
        self.screen_height = GameConstants.SCREEN_HEIGHT * scale

        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("BeamRider - JAX Implementation")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font("PressStart2P.ttf", 16)

        # Create BeamRider components
        self.env = BeamRiderEnv()
        self.renderer = BeamRiderRenderer()

    def run_game(self):
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
                if keys[pygame.K_LEFT] and keys[pygame.K_SPACE]:
                    action = 4  # left + fire
                elif keys[pygame.K_RIGHT] and keys[pygame.K_SPACE]:
                    action = 5  # right + fire
                elif keys[pygame.K_LEFT]:
                    action = 1  # left
                elif keys[pygame.K_RIGHT]:
                    action = 2  # right
                elif keys[pygame.K_SPACE]:
                    action = 3  # fire
                else:
                    action = 0  # no-op

                # Step and render
                state = self.env.step(state, action)
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
        """Konvertiert Tastatureingaben zu Aktionen"""
        if key == pygame.K_LEFT:
            return 1  # left
        elif key == pygame.K_RIGHT:
            return 2  # right
        elif key == pygame.K_SPACE:
            return 3  # fire
        elif key == pygame.K_a:  # A + Space für left+fire
            return 4
        elif key == pygame.K_d:  # D + Space für right+fire
            return 5
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
        """Draw centered Score and Level UI like Atari"""
        score_text = self.font.render(f"SCORE {state.score:06}", True, (255, 220, 100))  # padded 6-digit score
        level_text = self.font.render(f"SECTOR {state.level:02}", True, (255, 220, 100))  # padded 2-digit sector

        score_rect = score_text.get_rect(center=(self.screen_width // 2, 20))
        level_rect = level_text.get_rect(center=(self.screen_width // 2, 42))

        self.screen.blit(score_text, score_rect)
        self.screen.blit(level_text, level_rect)

        # === DRAW LIVES INDICATORS ===
        for i in range(state.lives):
            x = 30 + i * 36  # spacing between icons
            y = self.screen_height - 20  # near bottom
            scaled_ship = pygame.transform.scale(self.renderer.small_ship_surface, (int(self.renderer.small_ship_surface.get_width() * 1.5), int(self.renderer.small_ship_surface.get_height() * 1.5)))
            self.screen.blit(scaled_ship, (x, y))
    

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