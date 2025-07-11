from functools import partial
import pygame
import chex
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Tuple, NamedTuple
import random
import os
from sys import maxsize
import numpy as np
import math

from jaxatari.environment import JaxEnvironment
from jaxatari.renderers import AtraJaxisRenderer
from jaxatari.rendering import atraJaxis as aj

# Action constants for BattleZone
NOOP = 0
FIRE = 1
UP = 2
RIGHT = 3
LEFT = 4
DOWN = 5
UPRIGHT = 6
UPLEFT = 7
DOWNRIGHT = 8
DOWNLEFT = 9
UPFIRE = 10
RIGHTFIRE = 11
LEFTFIRE = 12
DOWNFIRE = 13
UPRIGHTFIRE = 14
UPLEFTFIRE = 15
DOWNRIGHTFIRE = 16
DOWNLEFTFIRE = 17

# Game constants
WIDTH = 160
HEIGHT = 210
MAX_BULLETS = 16       # Total bullets in the pool
PLAYER_BULLET_LIMIT = 8  # Reserve first 8 for the player
MAX_OBSTACLES = 16


# Tank movement constants
TANK_SPEED = 2.0
TANK_TURN_SPEED = 0.1
BULLET_SPEED = 4.0
BULLET_LIFETIME = 120  # frames

# 3D rendering constants
HORIZON_Y = 105  # Middle of screen
GROUND_COLOR = (0, 100, 0)  # Dark green
SKY_COLOR = (0, 0, 0)  # Black
WIREFRAME_COLOR = (0, 255, 0)  # Green
BULLET_COLOR = (255, 255, 255)  # White

# World boundaries
WORLD_SIZE = 2000
BOUNDARY_MIN = -WORLD_SIZE // 2
BOUNDARY_MAX = WORLD_SIZE // 2

class Tank(NamedTuple):
    x: chex.Array
    y: chex.Array
    angle: chex.Array  # Facing direction
    alive: chex.Array

class Bullet(NamedTuple):
    x: chex.Array
    y: chex.Array
    z: chex.Array  # Add this line
    vel_x: chex.Array
    vel_y: chex.Array
    active: chex.Array
    lifetime: chex.Array
    owner: chex.Array

class Obstacle(NamedTuple):
    x: chex.Array
    y: chex.Array
    obstacle_type: chex.Array  # 0: cube, 1: pyramid 

class BattleZoneState(NamedTuple):
    player_tank: Tank
    bullets: Bullet
    obstacles: Obstacle
    step_counter: chex.Array

class BattleZoneObservation(NamedTuple):
    player_tank: Tank
    bullets: Bullet
    obstacles: Obstacle

class BattleZoneInfo(NamedTuple):
    time: jnp.ndarray
    all_rewards: chex.Array

def get_human_action() -> chex.Array:
    """Get human input for BattleZone controls (arrow keys only for movements and space for shooting)."""
    keys = pygame.key.get_pressed()
    # Movement + Fire combinations
    if keys[pygame.K_UP] and keys[pygame.K_SPACE]:
        return jnp.array(UPFIRE)
    elif keys[pygame.K_DOWN] and keys[pygame.K_SPACE]:
        return jnp.array(DOWNFIRE)
    elif keys[pygame.K_LEFT] and keys[pygame.K_SPACE]:
        return jnp.array(LEFTFIRE)
    elif keys[pygame.K_RIGHT] and keys[pygame.K_SPACE]:
        return jnp.array(RIGHTFIRE)
    # Movement only
    elif keys[pygame.K_UP]:
        return jnp.array(UP)
    elif keys[pygame.K_DOWN]:
        return jnp.array(DOWN)
    elif keys[pygame.K_LEFT]:
        return jnp.array(LEFT)
    elif keys[pygame.K_RIGHT]:
        return jnp.array(RIGHT)
    # Fire only
    elif keys[pygame.K_SPACE]:
        return jnp.array(FIRE)
    else:
        return jnp.array(NOOP)

@jax.jit
def update_tank_position(tank: Tank, action: chex.Array) -> Tank:
    """Update tank position and angle based on arrow key action."""
    move_speed = TANK_SPEED
    
    # Start with current position
    new_x = tank.x
    new_y = tank.y
    angle = tank.angle
    
    # Handle all movement actions (including fire combinations)
    # LEFT movements
    left_actions = jnp.array([LEFT, LEFTFIRE])
    is_left = jnp.any(action == left_actions)
    new_x = jnp.where(is_left, tank.x - move_speed, new_x)
    angle = jnp.where(is_left, jnp.array(math.pi), angle)
    
    # RIGHT movements  
    right_actions = jnp.array([RIGHT, RIGHTFIRE])
    is_right = jnp.any(action == right_actions)
    new_x = jnp.where(is_right, tank.x + move_speed, new_x)
    angle = jnp.where(is_right, jnp.array(0.0), angle)
    
    # UP movements
    up_actions = jnp.array([UP, UPFIRE])
    is_up = jnp.any(action == up_actions)
    new_y = jnp.where(is_up, tank.y - move_speed, new_y)
    angle = jnp.where(is_up, jnp.array(-math.pi/2), angle)
    
    # DOWN movements
    down_actions = jnp.array([DOWN, DOWNFIRE])
    is_down = jnp.any(action == down_actions)
    new_y = jnp.where(is_down, tank.y + move_speed, new_y)
    angle = jnp.where(is_down, jnp.array(math.pi/2), angle)

    # Boundary checking
    new_x = jnp.clip(new_x, BOUNDARY_MIN, BOUNDARY_MAX)
    new_y = jnp.clip(new_y, BOUNDARY_MIN, BOUNDARY_MAX)

    return Tank(
        x=new_x,
        y=new_y,
        angle=angle,
        alive=tank.alive
    )

@jax.jit
def should_fire(action: chex.Array) -> chex.Array:
    """Check if the action includes firing."""
    fire_actions = jnp.array([FIRE, UPFIRE, RIGHTFIRE, LEFTFIRE, DOWNFIRE,
                             UPRIGHTFIRE, UPLEFTFIRE, DOWNRIGHTFIRE, DOWNLEFTFIRE])
    return jnp.any(action == fire_actions)

@jax.jit
def create_bullet(tank: Tank, owner: chex.Array) -> Bullet:
    """Create a new bullet slightly in front of tank's barrel."""
    angle = tank.angle
    offset = 10.0  # Offset in front of tank

    # Use consistent angle direction - no flip needed here
    vel_x = jnp.cos(angle) * BULLET_SPEED
    vel_y = jnp.sin(angle) * BULLET_SPEED
    spawn_x = tank.x + jnp.cos(angle) * offset
    spawn_y = tank.y + jnp.sin(angle) * offset

    return Bullet(
        x=spawn_x,
        y=spawn_y,
        z=jnp.array(10.0),  # Add this line - bullets fly at height 10
        vel_x=vel_x,
        vel_y=vel_y,
        active=jnp.array(1, dtype=jnp.int32),
        lifetime=jnp.array(BULLET_LIFETIME),
        owner=owner
    )

@jax.jit
def update_bullets(bullets: Bullet) -> Bullet:
    """Update all bullet positions and lifetimes."""
    new_x = bullets.x + bullets.vel_x
    new_y = bullets.y + bullets.vel_y
    new_lifetime = bullets.lifetime - 1
    
    # Deactivate bullets that go out of bounds or expire
    out_of_bounds = jnp.logical_or(
        jnp.logical_or(new_x < BOUNDARY_MIN, new_x > BOUNDARY_MAX),
        jnp.logical_or(new_y < BOUNDARY_MIN, new_y > BOUNDARY_MAX)
    )
    
    expired = new_lifetime <= 0
    
    active = jnp.logical_and(
        bullets.active,
        jnp.logical_and(jnp.logical_not(out_of_bounds), jnp.logical_not(expired))
    ).astype(jnp.int32)
    
    return Bullet(
        x=new_x,
        y=new_y,
        z=bullets.z,  # Add this line - maintain Z height
        vel_x=bullets.vel_x,
        vel_y=bullets.vel_y,
        active=active,
        lifetime=new_lifetime,
        owner=bullets.owner
    )

class JaxBattleZone(JaxEnvironment[BattleZoneState, BattleZoneObservation, BattleZoneInfo]):
    def __init__(self, reward_funcs: list[callable] = None):
        super().__init__()
        self.frame_stack_size = 4
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.action_set = list(range(18))  # All 18 BattleZone actions
        self.obs_size = 50

    def reset(self, key=None) -> Tuple[BattleZoneObservation, BattleZoneState]:
        """Reset the game to initial state."""
        # Initialize player tank at center
        player_tank = Tank(
            x=jnp.array(0.0),
            y=jnp.array(0.0),
            angle=jnp.array(0.0),
            alive=jnp.array(1, dtype=jnp.int32)
        )
        
        bullets = Bullet(
            x=jnp.zeros(MAX_BULLETS),
            y=jnp.zeros(MAX_BULLETS),
            z=jnp.zeros(MAX_BULLETS),  # Add this line
            vel_x=jnp.zeros(MAX_BULLETS),
            vel_y=jnp.zeros(MAX_BULLETS),
            active=jnp.zeros(MAX_BULLETS, dtype=jnp.int32),
            lifetime=jnp.zeros(MAX_BULLETS),
            owner=jnp.zeros(MAX_BULLETS)
        )
        
        # Initialize obstacles (cubes and pyramids scattered around)
        obstacle_positions_x = jnp.array([100.0, -150.0, 250.0, -250.0, 350.0, -350.0, 450.0, -450.0,
                                         150.0, -100.0, 300.0, -300.0, 400.0, -400.0, 500.0, -500.0])
        obstacle_positions_y = jnp.array([150.0, 200.0, -200.0, 250.0, -300.0, 300.0, -400.0, 400.0,
                                         -150.0, -200.0, 200.0, -250.0, 300.0, -300.0, 400.0, -400.0])
        obstacle_types = jnp.array([0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0])  # Mix of cubes and pyramids
        
        obstacles = Obstacle(
            x=obstacle_positions_x,
            y=obstacle_positions_y,
            obstacle_type=obstacle_types
        )
        
        state = BattleZoneState(
            player_tank=player_tank,
            bullets=bullets,
            obstacles=obstacles,
            step_counter=jnp.array(0)
        )
        
        observation = self._get_observation(state)
        return observation, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: BattleZoneState, action: chex.Array) -> Tuple[BattleZoneObservation, BattleZoneState, float, bool, BattleZoneInfo]:
        """Perform a step in the BattleZone environment. 
        This updates the game state based on the action taken.
        It is the main update LOOP for the game.
        """
        # Update player tank
        new_player_tank = update_tank_position(state.player_tank, action)
        
        # Handle player firing
        fire_bullet = should_fire(action)

        # Reserve only the first PLAYER_BULLET_LIMIT slots for the player
        inactive_player_slots = jnp.logical_not(state.bullets.active[:PLAYER_BULLET_LIMIT])
        inactive_bullet_idx = jnp.argmax(PLAYER_BULLET_LIMIT) #(inactive_player_slots)
        can_fire = inactive_player_slots[inactive_bullet_idx]

        
        # Create new bullet if firing and slot available
        new_bullet = create_bullet(new_player_tank, jnp.array(0))
        
        # Update bullets array with new player bullet
        updated_bullets = jax.lax.cond(
            jnp.logical_and(fire_bullet, can_fire),
            lambda b: Bullet(
                x=b.x.at[inactive_bullet_idx].set(new_bullet.x),
                y=b.y.at[inactive_bullet_idx].set(new_bullet.y),
                z=b.z.at[inactive_bullet_idx].set(new_bullet.z), 
                vel_x=b.vel_x.at[inactive_bullet_idx].set(new_bullet.vel_x),
                vel_y=b.vel_y.at[inactive_bullet_idx].set(new_bullet.vel_y),
                active=b.active.at[inactive_bullet_idx].set(1),
                lifetime=b.lifetime.at[inactive_bullet_idx].set(BULLET_LIFETIME),
                owner=b.owner.at[inactive_bullet_idx].set(0)
            ),
            lambda b: b,
            operand=state.bullets
        )
        
        # Update all bullet positions
        updated_bullets = update_bullets(updated_bullets)
        
        new_state = BattleZoneState(
            player_tank=new_player_tank,
            bullets=updated_bullets,
            obstacles=state.obstacles,  # Obstacles don't change
            step_counter=state.step_counter + 1
        )
        
        observation = self._get_observation(new_state)
        reward = self._get_env_reward(state, new_state)
        done = self._get_done(new_state)
        all_rewards = self._get_all_reward(state, new_state)
        info = self._get_info(new_state, all_rewards)

        return observation, new_state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: BattleZoneState) -> BattleZoneObservation:
        return BattleZoneObservation(
            player_tank=state.player_tank,
            bullets=state.bullets,
            obstacles=state.obstacles
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_env_reward(self, previous_state: BattleZoneState, state: BattleZoneState) -> float:
        return 0.0  # No reward system for now

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(self, previous_state: BattleZoneState, state: BattleZoneState):
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array([reward_func(previous_state, state) for reward_func in self.reward_funcs])
        return rewards

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: BattleZoneState) -> bool:
        return False  # Game never ends

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: BattleZoneState, all_rewards: chex.Array) -> BattleZoneInfo:
        return BattleZoneInfo(time=state.step_counter, all_rewards=all_rewards)

class BattleZoneRenderer:
    """3D wireframe renderer for BattleZone in the style of the original Atari 2600 game."""
    
    def __init__(self):
        self.view_distance = 400.0  # How far we can see
        self.fov = 60.0  # Field of view in degrees

        # Load player tank sprite
        try:
            path = "/Users/stevenmkhitarian/Documents/KI_Praktikum/BattleZone/src/jaxatari/games/sprites/battlezone/player_tank2.npy"
            # path = "/Users/stevenmkhitarian/Documents/KI_Praktikum/BattleZone/src/jaxatari/games/sprites/battlezone/player_tank_squeezed.npy"
            self.player_tank_sprite = np.load(path)
            self.sprite_loaded = True
        except FileNotFoundError:
            self.sprite_loaded = False
        except Exception as e:
            self.sprite_loaded = False

    def draw_player_bullet(self, screen, state: BattleZoneState):
        """Draw the player's bullet as a moving line toward the horizon."""
        bullets = state.bullets
        player_bullets = jnp.logical_and(bullets.active, bullets.owner == 0)
        if not jnp.any(player_bullets):
            return

        # Take the first active player bullet
        bullet_idx = jnp.argmax(player_bullets)

        bullet_x = bullets.x[bullet_idx]
        bullet_y = bullets.y[bullet_idx]

        # # Transform bullet position to screen
        # screen_x, screen_y, distance, visible = self.world_to_screen_3d(
        #     bullet_x, bullet_y,
        #     state.player_tank.x,
        #     state.player_tank.y,
        #     state.player_tank.angle
        # )

        # if visible and 0 <= screen_x < WIDTH and 0 <= screen_y < HEIGHT:
        #     # Draw the bullet as a short thick vertical line
        #     pygame.draw.line(
        #         screen,
        #         BULLET_COLOR,
        #         (screen_x, screen_y - 3),
        #         (screen_x, screen_y + 3),
        #         3
        #     )


    def numpy_to_pygame_surface(self, array):
        """Convert numpy array to pygame surface."""
        if array.dtype == np.uint8:
            sprite_array = array
        elif array.dtype == np.float32 or array.dtype == np.float64:
            sprite_array = (array * 255).astype(np.uint8)
        else:
            sprite_array = array.astype(np.uint8)

        if len(sprite_array.shape) == 2:
            height, width = sprite_array.shape
            rgb_array = np.stack([sprite_array] * 3, axis=-1)
            surface = pygame.Surface((width, height))
            pygame.surfarray.blit_array(surface, rgb_array.swapaxes(0, 1))

        elif len(sprite_array.shape) == 3:
            height, width, channels = sprite_array.shape
            if channels == 1:
                rgb_array = np.repeat(sprite_array, 3, axis=-1)
                surface = pygame.Surface((width, height))
                pygame.surfarray.blit_array(surface, rgb_array.swapaxes(0, 1))
            elif channels == 3:
                rgb_array = sprite_array
                surface = pygame.Surface((width, height))
                pygame.surfarray.blit_array(surface, rgb_array.swapaxes(0, 1))
            elif channels == 4:
                # RGBA support via frombuffer
                rgba_array = sprite_array
                surface = pygame.image.frombuffer(rgba_array.tobytes(), (width, height), "RGBA").convert_alpha()
            else:
                raise ValueError(f"Unsupported number of channels: {channels}")
        else:
            raise ValueError(f"Unsupported array shape: {sprite_array.shape}")
        
        return surface

    def world_to_screen_3d(self, world_x, world_y, player_x, player_y, player_angle):
        """Convert world coordinates to 3D screen coordinates relative to player."""
        # Translate to player-relative coordinates
        rel_x = world_x - player_x
        rel_y = world_y - player_y
        
        # Rotate by player angle to get view-relative coordinates
        cos_a = jnp.cos(player_angle)
        sin_a = jnp.sin(player_angle)
        
        # Correct transformation: forward is positive Y in view space
        # When player faces right (angle=0), forward should be +Y direction
        view_x = rel_x * cos_a + rel_y * sin_a   # Right/left relative to player
        view_y = -rel_x * sin_a + rel_y * cos_a  # Forward/back relative to player
                
        # Perspective projection
        if view_y > 1.0:  # Object is in front of player
            # Standard perspective projection
            screen_x = int(WIDTH // 2 + (view_x / view_y) * 100)
            # Objects closer to horizon (far away) have smaller y, objects close have larger y
            screen_y = int(HORIZON_Y + (10 / view_y) * 20)  # Simplified object positioning
            
            distance = view_y
            return screen_x, screen_y, distance, True
        else:
            return 0, 0, 0, False  # Behind player or too close

    def draw_wireframe_cube(self, screen, x, y, distance, color):
        """Draw a 3D wireframe cube."""
        if distance > self.view_distance:
            return
        scale = max(1, int(15 / max(distance / 50, 1)))
        # Perspective offset
        dz = scale // 2
        # 8 corners of the cube
        pts = [
            (x - scale, y - scale),           # 0: front-top-left
            (x + scale, y - scale),           # 1: front-top-right
            (x + scale, y + scale),           # 2: front-bottom-right
            (x - scale, y + scale),           # 3: front-bottom-left
            (x - scale + dz, y - scale - dz), # 4: back-top-left
            (x + scale + dz, y - scale - dz), # 5: back-top-right
            (x + scale + dz, y + scale - dz), # 6: back-bottom-right
            (x - scale + dz, y + scale - dz), # 7: back-bottom-left
        ]
        edges = [
            (0,1),(1,2),(2,3),(3,0), # front face
            (4,5),(5,6),(6,7),(7,4), # back face
            (0,4),(1,5),(2,6),(3,7)  # connections
        ]
        try:
            for e in edges:
                pygame.draw.line(screen, color, pts[e[0]], pts[e[1]], 1)
        except:
            pass

    def draw_wireframe_pyramid(self, screen, x, y, distance, color):
        """Draw a 3D wireframe pyramid."""
        if distance > self.view_distance:
            return
        scale = max(1, int(15 / max(distance / 50, 1)))
        # Base corners
        base = [
            (x - scale, y + scale),
            (x + scale, y + scale),
            (x + scale, y - scale),
            (x - scale, y - scale)
        ]
        # Apex
        apex = (x, y - scale*2)
        try:
            # Base square
            pygame.draw.line(screen, color, base[0], base[1], 1)
            pygame.draw.line(screen, color, base[1], base[2], 1)
            pygame.draw.line(screen, color, base[2], base[3], 1)
            pygame.draw.line(screen, color, base[3], base[0], 1)
            # Sides
            for b in base:
                pygame.draw.line(screen, color, b, apex, 1)
        except:
            pass

    def draw_player_tank(self, screen):
        """Draw player tank using sprite or wireframe fallback."""
        base_x = WIDTH // 2
        # Place the tank just above the bottom edge, not below the ground
        # The tank should be visually on the ground, not at HEIGHT-10
        # Place it a few pixels above the bottom, but below the horizon
        base_y = HEIGHT - 28  # This value works well for a 20px tall sprite

        if self.sprite_loaded:
            sprite_surface = self.numpy_to_pygame_surface(self.player_tank_sprite)
            sprite_width, sprite_height = sprite_surface.get_size()
            scale_factor = 0.1  # Adjust this to make sprite bigger/smaller
            scaled_width = int(sprite_width * scale_factor)
            scaled_height = int(sprite_height * scale_factor)
            sprite_surface = pygame.transform.scale(sprite_surface, (scaled_width, scaled_height))
            sprite_rect = sprite_surface.get_rect()
            # Align the bottom center of the sprite with the ground
            sprite_rect.midbottom = (base_x, HEIGHT - 2)
            screen.blit(sprite_surface, sprite_rect)
        else:
            # Fallback to wireframe rendering
            pygame.draw.ellipse(screen, (0, 255, 0), (base_x - 30, base_y - 10, 60, 20), 1)
            pygame.draw.rect(screen, (0, 255, 0), (base_x - 3, base_y - 30, 6, 20), 1)
            pygame.draw.rect(screen, (0, 255, 0), (base_x - 40, base_y - 10, 10, 20), 1)
            pygame.draw.rect(screen, (0, 255, 0), (base_x + 30, base_y - 10, 10, 20), 1)

    def draw_radar(self, screen, state: BattleZoneState):
        """Draw the BattleZone radar as a small circle with a sweeping white line and enemy dots."""
        # Radar should cover ~20% of the screen width, and not steal space from the main game area
        radar_radius = int(WIDTH * 0.12)  # ~20% of width as diameter
        radar_center_x = WIDTH - radar_radius - 8  # Right margin
        radar_center_y = radar_radius + 8          # Top margin

        # Draw black bar background (just behind radar)
        bar_width = radar_radius * 2 + 8
        bar_height = radar_radius * 2 + 8
        pygame.draw.rect(
            screen, (0, 0, 0),
            (radar_center_x - radar_radius - 4, radar_center_y - radar_radius - 4, bar_width, bar_height)
        )

        # Draw radar circle
        pygame.draw.circle(screen, (0, 255, 0), (radar_center_x, radar_center_y), radar_radius, 1)

        # Draw crosshairs
        pygame.draw.line(
            screen, (0, 80, 0),
            (radar_center_x - radar_radius, radar_center_y),
            (radar_center_x + radar_radius, radar_center_y), 1
        )
        pygame.draw.line(
            screen, (0, 80, 0),
            (radar_center_x, radar_center_y - radar_radius),
            (radar_center_x, radar_center_y + radar_radius), 1
        )

        # Player is always at center
        pygame.draw.circle(screen, (0, 255, 0), (radar_center_x, radar_center_y), 2)

        # Radar sweep: white line circulating clockwise
        sweep_speed = 0.025  # radians per frame
        angle = float(state.step_counter) * sweep_speed % (2 * math.pi)
        sweep_length = radar_radius - 2
        sweep_x = int(radar_center_x + sweep_length * math.cos(angle - math.pi/2))
        sweep_y = int(radar_center_y + sweep_length * math.sin(angle - math.pi/2))
        pygame.draw.line(screen, (255, 255, 255), (radar_center_x, radar_center_y), (sweep_x, sweep_y), 2)

        # Radar scale: fit WORLD_SIZE into radar
        scale = (radar_radius - 4) / (WORLD_SIZE / 2)

        # Draw obstacles (enemies) as dots
        player_x = state.player_tank.x
        player_y = state.player_tank.y
        for ox, oy in zip(state.obstacles.x, state.obstacles.y):
            dx = ox - player_x
            dy = oy - player_y
            rx = int(radar_center_x + dx * scale)
            ry = int(radar_center_y + dy * scale)
            # Only draw if inside radar circle
            if (rx - radar_center_x) ** 2 + (ry - radar_center_y) ** 2 <= (radar_radius - 3) ** 2:
                pygame.draw.circle(screen, (255, 0, 0), (rx, ry), 2)

        # Optionally: draw player bullets as white dots
        bullets = state.bullets
        for i in range(len(bullets.x)):
            if bullets.active[i] and bullets.owner[i] == 0:
                dx = bullets.x[i] - player_x
                dy = bullets.y[i] - player_y
                rx = int(radar_center_x + dx * scale)
                ry = int(radar_center_y + dy * scale)
                if (rx - radar_center_x) ** 2 + (ry - radar_center_y) ** 2 <= (radar_radius - 3) ** 2:
                    pygame.draw.circle(screen, (255, 255, 255), (rx, ry), 1)

    def render(self, state: BattleZoneState, screen):
        """Render the 3D wireframe view with dynamic ground and moving sky/mountains."""

        player = state.player_tank
        # --- Dynamic sky with moving mountains ---
        sky_height = HORIZON_Y
        sky_bands = 24
        # Blue gradient for sky
        for y in range(sky_height):
            t = y / max(sky_height - 1, 1)
            r = int(60 * (1 - t) + 10 * t)
            g = int(120 * (1 - t) + 40 * t)
            b = int(200 * (1 - t) + 120 * t)
            pygame.draw.line(screen, (r, g, b), (0, y), (WIDTH, y))

        # --- Moving mountains (parallax effect) ---
        mountain_layers = [
            # (color, amplitude, freq, y_offset, speed_factor)
            ((110, 110, 110), 18, 0.025, 30, 0.25),  # farthest, lightest, slowest
            ((80, 80, 80), 28, 0.035, 18, 0.5),      # middle
            ((50, 50, 50), 38, 0.045, 0, 1.0),       # closest, darkest, fastest
        ]
        for color, amp, freq, y_off, speed in mountain_layers:
            points = []
            phase = (player.x * speed * 0.03) % (2 * math.pi)
            for x in range(0, WIDTH + 1, 2):
                y = int(
                    sky_height
                    - (math.sin(freq * (x + phase * 120)) * amp + y_off)
                )
                points.append((x, y))
            points.append((WIDTH, sky_height))
            points.append((0, sky_height))
            pygame.draw.polygon(screen, color, points)

        # --- Restore previous ground logic (dynamic bands and perspective lines) ---
        ground_bands = 16
        ground_colors = [
            (60, 120, 60), (70, 130, 60), (80, 140, 60), (90, 150, 60),
            (100, 160, 60), (110, 170, 60), (120, 180, 60), (130, 190, 60),
            (140, 200, 60), (150, 210, 60), (160, 220, 60), (170, 230, 60),
            (180, 240, 60), (170, 230, 60), (160, 220, 60), (150, 210, 60)
        ]
        ground_offset = int(player.y * 0.15) % ground_bands
        band_height = (HEIGHT - HORIZON_Y) // ground_bands
        for i in range(ground_bands):
            color = ground_colors[(i + ground_offset) % ground_bands]
            y1 = HORIZON_Y + i * band_height
            y2 = HORIZON_Y + (i + 1) * band_height if i < ground_bands - 1 else HEIGHT
            pygame.draw.rect(screen, color, (0, y1, WIDTH, y2 - y1))

        # --- Draw perspective ground lines (simulate 3D effect) ---
        num_lines = 14
        for i in range(1, num_lines):
            t = i / num_lines
            y = int(HORIZON_Y + (HEIGHT - HORIZON_Y) * t * t)
            # Use a thin, light gray line instead of thick black
            # pygame.draw.line(screen, (100, 100, 100), (0, y), (WIDTH, y), 1)

        # --- Draw horizon line ---
        pygame.draw.line(screen, WIREFRAME_COLOR, (0, HORIZON_Y), (WIDTH, HORIZON_Y), 1)

        # --- Draw game objects ---
        # Draw obstacles (cubes and pyramids)
        obstacles = state.obstacles
        for i in range(len(obstacles.x)):
            obstacle_x = obstacles.x[i]
            obstacle_y = obstacles.y[i]
            obstacle_type = obstacles.obstacle_type[i]
            
            # Transform obstacle position to screen coordinates
            screen_x, screen_y, distance, visible = self.world_to_screen_3d(
                obstacle_x, obstacle_y,
                state.player_tank.x,
                state.player_tank.y,
                state.player_tank.angle
            )
            
            if visible and 0 <= screen_x < WIDTH and 0 <= screen_y < HEIGHT:
                if obstacle_type == 0:  # Cube
                    self.draw_wireframe_cube(screen, screen_x, screen_y, distance, WIREFRAME_COLOR)
                else:  # Pyramid
                    self.draw_wireframe_pyramid(screen, screen_x, screen_y, distance, WIREFRAME_COLOR)

        # Draw bullets
        self.draw_player_bullet(screen, state)

        # Draw player tank (this was missing!)
        self.draw_player_tank(screen)

        # --- Draw radar ---
        self.draw_radar(screen, state)



if __name__ == "__main__":
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH * 3, HEIGHT * 3))
    pygame.display.set_caption("BattleZone - Simplified")
    clock = pygame.time.Clock()
    
    # Initialize game
    game = JaxBattleZone()
    renderer = BattleZoneRenderer()
    
    # Get jitted functions
    jitted_step = jax.jit(game.step)
    jitted_reset = jax.jit(game.reset)
    
    obs, curr_state = jitted_reset()
    
    # Game loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, curr_state = jitted_reset()
        
        action = get_human_action()
        obs, curr_state, reward, done, info = jitted_step(curr_state, action)
        
        # Create a surface for the game area
        game_surface = pygame.Surface((WIDTH, HEIGHT))
        renderer.render(curr_state, game_surface)
        
        # Scale up the game surface
        scaled_surface = pygame.transform.scale(game_surface, (WIDTH * 3, HEIGHT * 3))
        screen.blit(scaled_surface, (0, 0))
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()