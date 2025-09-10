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
import math
import numpy as np

from jaxatari.environment import JaxEnvironment
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as aj
import jaxatari.spaces as spaces

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
ENEMY_BULLET_LIMIT = 8   # Reserve last 8 for enemies
MAX_OBSTACLES = 16

# Enemy AI constants
ENEMY_DETECTION_RANGE = 500.0
ENEMY_MOVE_SPEED = 0.15
ENEMY_TURN_SPEED = 0.008
ENEMY_FIRE_COOLDOWN = 180  # frames between shots (3 seconds at 60fps)
ENEMY_FIRE_RANGE = 400.0
ENEMY_OPTIMAL_DISTANCE = 120.0  # Preferred engagement distance
ENEMY_MIN_DISTANCE = 60.0  # Too close - retreat
ENEMY_MAX_DISTANCE = 200.0  # Too far - advance

# Enemy AI states
ENEMY_STATE_PATROL = 0
ENEMY_STATE_HUNT = 1
ENEMY_STATE_ENGAGE = 2
ENEMY_STATE_RETREAT = 3


# Tank movement constants
TANK_SPEED = 0.2  # Reduced from 0.5
TANK_TURN_SPEED = 0.008  # Reduced from 0.02
BULLET_SPEED = 1.0 # 4.0
BULLET_LIFETIME = 120  # frames

# Enemy spawning constants
ENEMY_SPAWN_DISTANCE_MIN = 150.0  # Minimum spawn distance from player
ENEMY_SPAWN_DISTANCE_MAX = 300.0  # Maximum spawn distance from player
MAX_ACTIVE_ENEMIES = 4  # Maximum enemies active at once
ENEMY_SPAWN_COOLDOWN = 300  # frames between spawns (5 seconds at 60fps)
ENEMY_SPAWN_WAIT = 120  # frames enemies wait after spawn before aiming/shooting

# =============================================================
# Atari BattleZone-inspired enemy AI tuning constants (new)
# Enemy AI adapted to Atari 2600 Battlezone rules — see Atari manual & gameplay.
# Tunable parameters for designers.
# =============================================================
DEBUG_ENEMY_AI = False  # Set True to enable on-screen and console debug for enemy AI
DEBUG_ENEMY_SPAWN = False  # Set True to enable spawn-time debug prints
DEBUG_ENEMY_STEERING = False  # Set True to enable steering debug prints
DEBUG_ENEMY_FIRING = False  # Set True to enable firing debug prints

# Enemy types
ENEMY_TYPE_TANK = 0
ENEMY_TYPE_SUPERTANK = 1
ENEMY_TYPE_FIGHTER = 2
ENEMY_TYPE_SAUCER = 3

# Speed factors relative to player speed (player uses TANK_SPEED constant)
SLOW_TANK_SPEED_FACTOR = 0.85
SUPERTANK_SPEED_FACTOR = 1.25
FIGHTER_SPEED_FACTOR = 1.6
SAUCER_SPEED_FACTOR = 0.6

# Turn rates (degrees per second) - converted to per-frame using 60 fps assumption
TANK_MAX_TURN_DEG = 120.0
SUPERTANK_MAX_TURN_DEG = 200.0
FIGHTER_MAX_TURN_DEG = 380.0

# Firing / engagement parameters
FIRING_ANGLE_THRESHOLD_DEG = 6.0  # degrees tolerance to consider 'on target'
FIRING_RANGE = 400.0  # max distance to consider firing
ENEMY_NO_FIRE_AFTER_SPAWN_SEC = 2.0  # seconds to wait before firing after spawn
FPS = 60.0
ENEMY_NO_FIRE_AFTER_SPAWN = int(ENEMY_NO_FIRE_AFTER_SPAWN_SEC * FPS)

# Fire cooldowns in seconds -> frames
ENEMY_FIRE_COOLDOWN_TANK_SEC = 1.2
ENEMY_FIRE_COOLDOWN_SUPERTANK_SEC = 0.8
ENEMY_FIRE_COOLDOWN_FIGHTER_SEC = 1.0
ENEMY_FIRE_COOLDOWN_TANK = int(ENEMY_FIRE_COOLDOWN_TANK_SEC * FPS)
ENEMY_FIRE_COOLDOWN_SUPERTANK = int(ENEMY_FIRE_COOLDOWN_SUPERTANK_SEC * FPS)
ENEMY_FIRE_COOLDOWN_FIGHTER = int(ENEMY_FIRE_COOLDOWN_FIGHTER_SEC * FPS)

# Fighter zigzag parameters
def _deg_per_sec_to_rad_per_frame(deg_per_sec: float) -> float:
    """Convert degrees/sec to radians/frame assuming FPS frames per second."""
    return (deg_per_sec * (math.pi / 180.0)) / FPS

# Per-type speed multipliers (relative to ENEMY_MOVE_SPEED)
ENEMY_SPEED_MULTIPLIERS = jnp.array([
    SLOW_TANK_SPEED_FACTOR,
    SUPERTANK_SPEED_FACTOR,
    FIGHTER_SPEED_FACTOR,
    SAUCER_SPEED_FACTOR,
], dtype=jnp.float32)

# Per-type firing parameters (based on ENEMY_FIRING spec)
ENEMY_FIRING_ANGLE_DEG = jnp.array([3.0, 3.0, 3.0, 3.0], dtype=jnp.float32)
ENEMY_FIRING_ANGLE_THRESH_RAD = (ENEMY_FIRING_ANGLE_DEG * (math.pi / 180.0)) / 1.0
ENEMY_FIRING_RANGE = jnp.array([30.0, 30.0, 15.0, 0.0], dtype=jnp.float32)
ENEMY_NO_FIRE_AFTER_SPAWN_FRAMES = jnp.array([int(2.0 * FPS), int(2.0 * FPS), int(0.5 * FPS), int(2.0 * FPS)], dtype=jnp.int32)
FIGHTER_POINT_BLANK_DIST = 5.0
FIGHTER_VEER_ANGLE_RAD = math.radians(90.0)

# Lateral angle to apply during HUNT state so enemies move left/right rather than directly at player
# Values per subtype (TANK, SUPERTANK, FIGHTER, SAUCER)
ENEMY_HUNT_LATERAL_ANGLE_DEG = jnp.array([35.0, 20.0, 0.0, 0.0], dtype=jnp.float32)
ENEMY_HUNT_LATERAL_ANGLE_RAD = ENEMY_HUNT_LATERAL_ANGLE_DEG * (math.pi / 180.0)

# Fighter zigzag parameters

ENEMY_TURN_RATES = jnp.array([
    _deg_per_sec_to_rad_per_frame(TANK_MAX_TURN_DEG),
    _deg_per_sec_to_rad_per_frame(SUPERTANK_MAX_TURN_DEG),
    _deg_per_sec_to_rad_per_frame(FIGHTER_MAX_TURN_DEG),
    _deg_per_sec_to_rad_per_frame(60.0),
], dtype=jnp.float32)

ENEMY_FIRE_COOLDOWNS = jnp.array([
    ENEMY_FIRE_COOLDOWN_TANK,
    ENEMY_FIRE_COOLDOWN_SUPERTANK,
    ENEMY_FIRE_COOLDOWN_FIGHTER,
    0,
], dtype=jnp.int32)

# Per-type can_fire flags (1 = can fire, 0 = cannot)
ENEMY_CAN_FIRE = jnp.array([1, 1, 1, 0], dtype=jnp.int32)


# World boundaries
WORLD_SIZE = 2000
BOUNDARY_MIN = -WORLD_SIZE // 2
BOUNDARY_MAX = WORLD_SIZE // 2

# Map-based spawn radii (computed from boundaries)
MAP_RADIUS = float(BOUNDARY_MAX - BOUNDARY_MIN) / 2.0
SPAWN_NEAR_RADIUS = MAP_RADIUS * 0.375
SPAWN_FAR_RADIUS = MAP_RADIUS * 0.75

# Projectile tuning (enemy)
ENEMY_PROJECTILE_SPEED = BULLET_SPEED * 0.9
ENEMY_PROJECTILE_LIFETIME = BULLET_LIFETIME

# Ensure existing uses of ENEMY_MOVE_SPEED are preserved for legacy behavior but will be adapted
# ...existing code continues...

# 3D rendering constants
HORIZON_Y = 105  # Middle of screen
GROUND_COLOR = (0, 100, 0)  # Dark green
SKY_COLOR = (0, 0, 0)  # Black
WIREFRAME_COLOR = (0, 255, 0)  # Green
BULLET_COLOR = (255, 255, 255)  # White

# Per-enemy-subtype colors
TANK_COLOR = (0, 200, 0)        # standard tank - greenish
SUPERTANK_COLOR = (255, 60, 60)  # supertank - reddish
FIGHTER_COLOR = (200, 200, 0)    # fighter - yellowish
SAUCER_COLOR = (100, 200, 255)   # saucer - cyan
HUD_ACCENT_COLOR = (47, 151, 119)  # #2f9777

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
    obstacle_type: chex.Array  # 0: enemy tank, 1: other obstacles
    enemy_subtype: chex.Array  # Enemy subtype (ENEMY_TYPE_*) when obstacle_type==0, else -1
    angle: chex.Array  # Add tank facing direction
    alive: chex.Array  # Add alive status for enemy tanks
    fire_cooldown: chex.Array  # Cooldown timer for enemy firing
    ai_state: chex.Array  # AI state for enemy behavior
    target_angle: chex.Array  # Target angle for patrol/movement
    state_timer: chex.Array  # Timer for current state

class BattleZoneState(NamedTuple):
    player_tank: Tank
    bullets: Bullet
    obstacles: Obstacle
    step_counter: chex.Array
    spawn_timer: chex.Array  # Timer for enemy spawning
    prev_player_x: chex.Array  # Previous player x (for motion-based rendering)
    prev_player_y: chex.Array  # Previous player y (for motion-based rendering)
    player_score: chex.Array   # HUD: player's score
    player_lives: chex.Array   # HUD: player's remaining lives

class BattleZoneObservation(NamedTuple):
    player_tank: Tank
    bullets: Bullet
    obstacles: Obstacle

class BattleZoneInfo(NamedTuple):
    time: jnp.ndarray
    all_rewards: chex.Array
    player_shot: chex.Array  # New flag: 1 if player was shot this step, 0 otherwise

def get_human_action() -> chex.Array:
    """Get human input for BattleZone controls (arrow keys for movements and space for shooting)."""
    keys = pygame.key.get_pressed()
    
    # Diagonal Movement + Fire combinations
    if keys[pygame.K_UP] and keys[pygame.K_RIGHT] and keys[pygame.K_SPACE]:
        return jnp.array(UPRIGHTFIRE)
    elif keys[pygame.K_UP] and keys[pygame.K_LEFT] and keys[pygame.K_SPACE]:
        return jnp.array(UPLEFTFIRE)
    elif keys[pygame.K_DOWN] and keys[pygame.K_RIGHT] and keys[pygame.K_SPACE]:
        return jnp.array(DOWNRIGHTFIRE)
    elif keys[pygame.K_DOWN] and keys[pygame.K_LEFT] and keys[pygame.K_SPACE]:
        return jnp.array(DOWNLEFTFIRE)
    # Single Direction + Fire combinations
    elif keys[pygame.K_UP] and keys[pygame.K_SPACE]:
        return jnp.array(UPFIRE)
    elif keys[pygame.K_DOWN] and keys[pygame.K_SPACE]:
        return jnp.array(DOWNFIRE)
    elif keys[pygame.K_LEFT] and keys[pygame.K_SPACE]:
        return jnp.array(LEFTFIRE)
    elif keys[pygame.K_RIGHT] and keys[pygame.K_SPACE]:
        return jnp.array(RIGHTFIRE)
    # Diagonal Movement only
    elif keys[pygame.K_UP] and keys[pygame.K_RIGHT]:
        return jnp.array(UPRIGHT)
    elif keys[pygame.K_UP] and keys[pygame.K_LEFT]:
        return jnp.array(UPLEFT)
    elif keys[pygame.K_DOWN] and keys[pygame.K_RIGHT]:
        return jnp.array(DOWNRIGHT)
    elif keys[pygame.K_DOWN] and keys[pygame.K_LEFT]:
        return jnp.array(DOWNLEFT)
    # Single Direction Movement only
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
    """
    Update tank position and angle based on action with realistic tank movement.
    Tank physics:
    - LEFT/RIGHT: Turn the tank (rotate in place)
    - UP: Move forward in the direction the tank is facing
    - DOWN: Move backward (reverse)
    - Diagonals: Combine turning and movement
    
    Coordinate system:
    - angle=0: facing right (+X direction)
    - angle=π/2: facing down (+Y direction) 
    - angle=π: facing left (-X direction)
    - angle=-π/2: facing up (-Y direction)
    """
    move_speed = TANK_SPEED
    turn_speed = TANK_TURN_SPEED
    
    # Start with current position and angle
    new_x = tank.x
    new_y = tank.y
    angle = tank.angle
    
    # Handle pure turning (LEFT/RIGHT) - tank rotates in place
    left_actions = jnp.array([LEFT, LEFTFIRE])
    is_left = jnp.any(action == left_actions)
    angle = jnp.where(is_left, tank.angle - turn_speed, angle)
    
    right_actions = jnp.array([RIGHT, RIGHTFIRE])
    is_right = jnp.any(action == right_actions)
    angle = jnp.where(is_right, tank.angle + turn_speed, angle)
    
    # Handle forward movement (UP) - move in direction tank is facing
    up_actions = jnp.array([UP, UPFIRE])
    is_up = jnp.any(action == up_actions)
    new_x = jnp.where(is_up, tank.x + jnp.cos(angle) * move_speed, new_x)
    new_y = jnp.where(is_up, tank.y + jnp.sin(angle) * move_speed, new_y)
    
    # Handle backward movement (DOWN) - move opposite to tank facing direction
    down_actions = jnp.array([DOWN, DOWNFIRE])
    is_down = jnp.any(action == down_actions)
    new_x = jnp.where(is_down, tank.x - jnp.cos(angle) * move_speed, new_x)
    new_y = jnp.where(is_down, tank.y - jnp.sin(angle) * move_speed, new_y)

    # Handle diagonal movements - combinations of turning and moving
    # UPRIGHT: Turn right while moving forward
    upright_actions = jnp.array([UPRIGHT, UPRIGHTFIRE])
    is_upright = jnp.any(action == upright_actions)
    # Update angle first, then move in new direction
    new_angle_upright = tank.angle + turn_speed
    angle = jnp.where(is_upright, new_angle_upright, angle)
    new_x = jnp.where(is_upright, tank.x + jnp.cos(new_angle_upright) * move_speed, new_x)
    new_y = jnp.where(is_upright, tank.y + jnp.sin(new_angle_upright) * move_speed, new_y)
    
    # UPLEFT: Turn left while moving forward
    upleft_actions = jnp.array([UPLEFT, UPLEFTFIRE])
    is_upleft = jnp.any(action == upleft_actions)
    new_angle_upleft = tank.angle - turn_speed
    angle = jnp.where(is_upleft, new_angle_upleft, angle)
    new_x = jnp.where(is_upleft, tank.x + jnp.cos(new_angle_upleft) * move_speed, new_x)
    new_y = jnp.where(is_upleft, tank.y + jnp.sin(new_angle_upleft) * move_speed, new_y)
    
    # DOWNRIGHT: Turn right while moving backward
    downright_actions = jnp.array([DOWNRIGHT, DOWNRIGHTFIRE])
    is_downright = jnp.any(action == downright_actions)
    new_angle_downright = tank.angle + turn_speed
    angle = jnp.where(is_downright, new_angle_downright, angle)
    new_x = jnp.where(is_downright, tank.x - jnp.cos(new_angle_downright) * move_speed, new_x)
    new_y = jnp.where(is_downright, tank.y - jnp.sin(new_angle_downright) * move_speed, new_y)
    
    # DOWNLEFT: Turn left while moving backward
    downleft_actions = jnp.array([DOWNLEFT, DOWNLEFTFIRE])
    is_downleft = jnp.any(action == downleft_actions)
    new_angle_downleft = tank.angle - turn_speed
    angle = jnp.where(is_downleft, new_angle_downleft, angle)
    new_x = jnp.where(is_downleft, tank.x - jnp.cos(new_angle_downleft) * move_speed, new_x)
    new_y = jnp.where(is_downleft, tank.y - jnp.sin(new_angle_downleft) * move_speed, new_y)

    # Normalize angle to [-π, π] range to prevent angle drift
    angle = jnp.arctan2(jnp.sin(angle), jnp.cos(angle))

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
def check_bullet_obstacle_collisions(bullets: Bullet, obstacles: Obstacle) -> Tuple[Bullet, Obstacle, chex.Array]:
    """Check for collisions between bullets and obstacles, removing both on hit.
    Returns updated bullets, updated obstacles, and score_delta (1000 per obstacle killed by a player bullet).
    """
    
    # Define collision radius
    collision_radius = 15.0
    
    # Create meshgrid for bullet-obstacle pairs
    bullet_x = bullets.x[:, None]  # Shape: (num_bullets, 1)
    bullet_y = bullets.y[:, None]  # Shape: (num_bullets, 1)
    bullet_active = bullets.active[:, None]  # Shape: (num_bullets, 1)
    
    obstacle_x = obstacles.x[None, :]  # Shape: (1, num_obstacles)
    obstacle_y = obstacles.y[None, :]  # Shape: (1, num_obstacles)
    obstacle_alive = obstacles.alive[None, :]  # Add alive check
    
    # Calculate distances between all bullet-obstacle pairs
    dx = bullet_x - obstacle_x  # Shape: (num_bullets, num_obstacles)
    dy = bullet_y - obstacle_y  # Shape: (num_bullets, num_obstacles)
    distances = jnp.sqrt(dx * dx + dy * dy)
    
    # Check collisions (only for active bullets and alive obstacles)
    collisions = jnp.logical_and(
        jnp.logical_and(bullet_active, obstacle_alive),  # Only active bullets and alive obstacles
        distances < collision_radius
    )
    
    # Mark bullets for removal (any bullet that collides with any obstacle)
    bullets_to_remove = jnp.any(collisions, axis=1)  # Shape: (num_bullets,)
    
    # Mark obstacles for removal (any obstacle that collides with any bullet)
    obstacles_to_remove = jnp.any(collisions, axis=0)  # Shape: (num_obstacles,)
    
    # Compute score delta: count obstacles killed by player bullets (owner == 0) and multiply by 1000
    # collisions_by_player: True where a player bullet hits an obstacle
    player_bullet_mask = (bullets.owner[:, None] == 0)
    collisions_by_player = jnp.logical_and(collisions, player_bullet_mask)
    obstacles_killed_by_player = jnp.any(collisions_by_player, axis=0)  # per-obstacle
    score_delta = jnp.sum(obstacles_killed_by_player).astype(jnp.int32) * jnp.array(1000, dtype=jnp.int32)
    
    # Update bullets - set collided bullets to inactive
    new_bullet_active = jnp.where(
        bullets_to_remove,
        jnp.zeros_like(bullets.active),
        bullets.active
    )
    
    # Update obstacles - set collided obstacles to not alive
    new_obstacle_alive = jnp.where(obstacles_to_remove, 0, obstacles.alive)
    
    # Create updated structures
    updated_bullets = Bullet(
        x=bullets.x,
        y=bullets.y,
        z=bullets.z,
        vel_x=bullets.vel_x,
        vel_y=bullets.vel_y,
        active=new_bullet_active,
        lifetime=bullets.lifetime,
        owner=bullets.owner
    )
    
    updated_obstacles = Obstacle(
        x=obstacles.x,
        y=obstacles.y,
        obstacle_type=obstacles.obstacle_type,
    enemy_subtype=getattr(obstacles, 'enemy_subtype', jnp.full_like(obstacles.x, -1)),
        angle=obstacles.angle,
        alive=new_obstacle_alive,
        fire_cooldown=obstacles.fire_cooldown,
        ai_state=obstacles.ai_state,
        target_angle=obstacles.target_angle,
        state_timer=obstacles.state_timer
    )
    
    return updated_bullets, updated_obstacles, score_delta
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
    offset = 0.1  # Offset in front of tank

    # Use consistent angle direction - no flip needed here
    vel_x = jnp.cos(angle) * BULLET_SPEED
    vel_y = jnp.sin(angle) * BULLET_SPEED
    spawn_x = tank.x + jnp.cos(angle) * offset
    spawn_y = tank.y + jnp.sin(angle) * offset

    return Bullet(
        x=spawn_x,
        y=spawn_y,
        z=jnp.array(3.0),  # Add this line - bullets fly at height 10
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

@jax.jit
def check_player_hit(player_tank: Tank, bullets: Bullet) -> Tank:
    """Check if player is hit by enemy bullets."""
    
    # Check collision with enemy bullets
    enemy_bullets = jnp.logical_and(bullets.active, bullets.owner > 0)
    
    # Calculate distances to player
    dx = bullets.x - player_tank.x
    dy = bullets.y - player_tank.y
    distances = jnp.sqrt(dx * dx + dy * dy)
    
    # Check for hits (collision radius)
    collision_radius = 10.0
    hits = jnp.logical_and(enemy_bullets, distances < collision_radius)
    
    # If any hit, player dies
    player_hit = jnp.any(hits)
    new_alive = jnp.where(player_hit, 0, player_tank.alive)
    
    return Tank(
        x=player_tank.x,
        y=player_tank.y,
        angle=player_tank.angle,
        alive=new_alive
    )

@jax.jit
def update_enemy_tanks(obstacles: Obstacle, player_tank: Tank, bullets: Bullet, step_counter: chex.Array) -> Tuple[Obstacle, Bullet]:
    """JAX-friendly enemy AI that more closely matches Atari BattleZone behaviour.

    Key behaviours restored:
    - Instant facing toward player when in detection range.
    - Simple distance-band movement (advance/retreat/hold).
    - Fire only while in engagement band, roughly frontal, within range, cooldown ready,
      and not during the spawn waiting timer.
    - Deterministic per-enemy bullet slot assignment (JAX fori_loop).
    """

    def update_single_enemy(i):
        enemy_x = obstacles.x[i]
        enemy_y = obstacles.y[i]
        enemy_angle = obstacles.angle[i]
        enemy_alive = obstacles.alive[i]
        enemy_cooldown = obstacles.fire_cooldown[i]
        enemy_state = obstacles.ai_state[i]
        enemy_target_angle = obstacles.target_angle[i]
        enemy_state_timer = obstacles.state_timer[i]
        is_enemy_tank = obstacles.obstacle_type[i] == 0

        should_process = jnp.logical_and(enemy_alive == 1, is_enemy_tank)

        # Vector to player and normalized direction
        dx = player_tank.x - enemy_x
        dy = player_tank.y - enemy_y
        distance_to_player = jnp.sqrt(dx * dx + dy * dy) + 1e-8
        dir_x = dx / distance_to_player
        dir_y = dy / distance_to_player
        angle_to_player = jnp.arctan2(dy, dx)

        # Timers and defaults
        new_state_timer = jnp.maximum(0, enemy_state_timer - 1)
        waiting = new_state_timer > 0

        # Default outputs equal inputs
        new_x = enemy_x
        new_y = enemy_y
        new_angle = enemy_angle
        new_state = enemy_state
        new_target_angle = enemy_target_angle
        new_cooldown = jnp.maximum(0, enemy_cooldown - 1)
        should_fire = jnp.array(False)

        # If player is within detection range, allow AI to respond (steering/firing)
        in_detection = jnp.logical_and(should_process, distance_to_player < ENEMY_DETECTION_RANGE)

        # Movement bands (advance if too far, retreat if too close)
        too_close = distance_to_player < ENEMY_MIN_DISTANCE
        too_far = distance_to_player > ENEMY_MAX_DISTANCE
        in_engage_band = jnp.logical_and(distance_to_player <= ENEMY_MAX_DISTANCE, distance_to_player >= ENEMY_MIN_DISTANCE)

        move_forward = jnp.logical_and(in_detection, jnp.logical_or(too_far, jnp.logical_and(in_engage_band, distance_to_player > ENEMY_OPTIMAL_DISTANCE)))
        move_backward = jnp.logical_and(in_detection, too_close)

        # Determine subtype early so we can treat tank-like enemies differently
        subtype = obstacles.enemy_subtype[i]
        is_tank_subtype = jnp.logical_or(subtype == ENEMY_TYPE_TANK, subtype == ENEMY_TYPE_SUPERTANK)

        # For non-tank obstacles, use simple direct band movement; tank-like enemies use steering below
        move_forward_non_tank = jnp.logical_and(move_forward, jnp.logical_not(is_tank_subtype))
        move_backward_non_tank = jnp.logical_and(move_backward, jnp.logical_not(is_tank_subtype))

        new_x = jnp.where(move_forward_non_tank, enemy_x + dir_x * ENEMY_MOVE_SPEED, new_x)
        new_y = jnp.where(move_forward_non_tank, enemy_y + dir_y * ENEMY_MOVE_SPEED, new_y)
        new_x = jnp.where(move_backward_non_tank, enemy_x - dir_x * ENEMY_MOVE_SPEED, new_x)
        new_y = jnp.where(move_backward_non_tank, enemy_y - dir_y * ENEMY_MOVE_SPEED, new_y)

        # --- Atari-style steering for TANK and SUPERTANK ---
        # integer index for arrays
        subtype_idx = subtype.astype(jnp.int32)

        # Select per-type turn rate and speed multiplier
        turn_rate = jnp.where(subtype == ENEMY_TYPE_SUPERTANK,
                              ENEMY_TURN_RATES[ENEMY_TYPE_SUPERTANK],
                              ENEMY_TURN_RATES[ENEMY_TYPE_TANK])

        speed_multiplier = jnp.where(subtype == ENEMY_TYPE_SUPERTANK,
                                     ENEMY_SPEED_MULTIPLIERS[ENEMY_TYPE_SUPERTANK],
                                     ENEMY_SPEED_MULTIPLIERS[ENEMY_TYPE_TANK])

        # Desired heading towards player
        desired_heading = angle_to_player

        # Lateral hunt offset applied per-subtype while in HUNT state (blue tanks only)
        is_hunt = enemy_state == ENEMY_STATE_HUNT
        is_blue_tank = subtype == ENEMY_TYPE_TANK
        lateral_rad = ENEMY_HUNT_LATERAL_ANGLE_RAD[subtype_idx]
        sign = jnp.where(jnp.sin(step_counter * 0.13 + i * 0.41) > 0, 1.0, -1.0)
        lateral_offset = lateral_rad * sign
        desired_heading = jnp.where(jnp.logical_and(is_hunt, is_blue_tank), desired_heading + lateral_offset, desired_heading)

        # Steering: limited turn toward desired heading
        raw_delta = desired_heading - new_angle
        delta_angle = jnp.arctan2(jnp.sin(raw_delta), jnp.cos(raw_delta))
        max_turn = turn_rate
        turn_amount = jnp.clip(delta_angle, -max_turn, max_turn)
        apply_steer = jnp.logical_and(should_process, in_detection)
        applied_turn = jnp.where(jnp.logical_and(apply_steer, is_tank_subtype), turn_amount, jnp.array(0.0))

        steered_angle = new_angle + applied_turn
        steered_speed = ENEMY_MOVE_SPEED * speed_multiplier

        # Update heading for tank-like enemies when steering applies
        new_angle = jnp.where(jnp.logical_and(apply_steer, is_tank_subtype), steered_angle, new_angle)

        # Tanks move forward along heading but use a periodic move/stop phase (strafe then hold to fire)
        base_move_speed = ENEMY_MOVE_SPEED * speed_multiplier
        move_angle = jnp.where(jnp.logical_and(apply_steer, is_tank_subtype), steered_angle, new_angle)
        tank_move_cond = jnp.logical_and(should_process, is_tank_subtype)

        # Periodic phase controls whether the tank is in move/strafe mode or holding to fire
        phase = jnp.sin(step_counter * 0.08 + i * 0.3)
        move_phase = phase > 0.0
        move_multiplier = jnp.where(move_phase, 1.0, 0.0)

        tank_next_x = new_x + jnp.cos(move_angle) * base_move_speed * move_multiplier
        tank_next_y = new_y + jnp.sin(move_angle) * base_move_speed * move_multiplier
        new_x = jnp.where(tank_move_cond, tank_next_x, new_x)
        new_y = jnp.where(tank_move_cond, tank_next_y, new_y)

        if DEBUG_ENEMY_STEERING:
            jax.debug.print('[STEER] idx={} type={} heading={:.2f} desired={:.2f} delta={:.2f} applied={:.2f}',
                            i, subtype, new_angle, desired_heading, delta_angle, applied_turn)

        # Firing conditions
        angle_diff = angle_to_player - new_angle
        angle_diff = jnp.arctan2(jnp.sin(angle_diff), jnp.cos(angle_diff))
        is_frontal = jnp.abs(angle_diff) < 0.35

        time_since_spawn = ENEMY_SPAWN_WAIT - enemy_state_timer
        allowed_by_spawn_time = time_since_spawn > ENEMY_NO_FIRE_AFTER_SPAWN_FRAMES[subtype_idx]

        ang_diff = jnp.arctan2(jnp.sin(new_angle - angle_to_player), jnp.cos(new_angle - angle_to_player))
        abs_ang_diff = jnp.abs(ang_diff)
        angle_ok = abs_ang_diff <= ENEMY_FIRING_ANGLE_THRESH_RAD[subtype_idx]
        range_ok = distance_to_player <= ENEMY_FIRING_RANGE[subtype_idx]

        cooldown_ok = enemy_cooldown <= 0
        type_can_fire = ENEMY_CAN_FIRE[subtype_idx] == 1
        base_can_fire = jnp.logical_and(jnp.logical_and(type_can_fire, cooldown_ok), allowed_by_spawn_time)
        firing_condition = jnp.logical_and(jnp.logical_and(base_can_fire, angle_ok), range_ok)

        # Fighter special-case veer & immediate fire
        is_fighter = subtype == ENEMY_TYPE_FIGHTER
        fighter_point_blank = jnp.logical_and(is_fighter, distance_to_player < FIGHTER_POINT_BLANK_DIST)
        sin_val = jnp.sin(step_counter * 0.123 + i * 0.71)
        veer_sign = jnp.where(sin_val >= 0.0, 1.0, -1.0)
        veer_angle = veer_sign * FIGHTER_VEER_ANGLE_RAD
        fighter_fire_now = jnp.logical_and(fighter_point_blank, base_can_fire)
        new_angle = jnp.where(fighter_fire_now, new_angle + veer_angle, new_angle)

        # Tanks are more likely to fire while in the hold phase (not moving)
        in_hold_phase = jnp.logical_and(tank_move_cond, jnp.logical_not(move_phase))
        hold_fire_boost = jnp.where(in_hold_phase, 1, 0)

        should_fire = jnp.logical_or(jnp.logical_and(firing_condition, jnp.logical_not(waiting)), fighter_fire_now)
        should_fire = jnp.logical_or(should_fire, jnp.logical_and(in_hold_phase, base_can_fire))

        new_state = jnp.where(in_engage_band, ENEMY_STATE_ENGAGE, new_state)
        new_cooldown = jnp.where(should_fire, ENEMY_FIRE_COOLDOWNS[subtype_idx], new_cooldown)

        if DEBUG_ENEMY_FIRING:
            jax.debug.print('[FIRE] idx={} type={} cooldown={} dist={:.2f} ang_diff={:.2f} allowed_by_spawn={} fired={}',
                            i, subtype_idx, enemy_cooldown, distance_to_player, abs_ang_diff, allowed_by_spawn_time, should_fire)

        # Bullet spawn and aim
        bullet_offset = 15.0
        spawn_bx = new_x + jnp.cos(new_angle) * bullet_offset
        spawn_by = new_y + jnp.sin(new_angle) * bullet_offset
        aim_dx = player_tank.x - new_x
        aim_dy = player_tank.y - new_y
        aim_dist = jnp.sqrt(aim_dx * aim_dx + aim_dy * aim_dy) + 1e-6
        aim_vx = (aim_dx / aim_dist) * BULLET_SPEED
        aim_vy = (aim_dy / aim_dist) * BULLET_SPEED

        # Clamp position and normalize angle
        new_x = jnp.clip(new_x, BOUNDARY_MIN, BOUNDARY_MAX)
        new_y = jnp.clip(new_y, BOUNDARY_MIN, BOUNDARY_MAX)
        new_angle = jnp.arctan2(jnp.sin(new_angle), jnp.cos(new_angle))

        return (new_x, new_y, new_angle, enemy_alive, new_cooldown, new_state,
                new_target_angle, new_state_timer, should_fire, spawn_bx, spawn_by, aim_vx, aim_vy)

    obstacles_count = obstacles.x.shape[0]
    idxs = jnp.arange(obstacles_count)
    results = jax.vmap(update_single_enemy)(idxs)

    (new_x, new_y, new_angle, new_alive, new_cooldown, new_state,
     new_target_angle, new_state_timer, fire_flags, bullet_xs, bullet_ys, bullet_vel_xs, bullet_vel_ys) = results

    updated_obstacles = Obstacle(
        x=new_x,
        y=new_y,
    obstacle_type=obstacles.obstacle_type,
    enemy_subtype=getattr(obstacles, 'enemy_subtype', jnp.full_like(obstacles.x, -1)),
        angle=new_angle,
        alive=new_alive,
        fire_cooldown=new_cooldown,
        ai_state=new_state,
        target_angle=new_target_angle,
        state_timer=new_state_timer
    )

    # Enemy bullets: deterministic per-enemy slots using fori_loop to keep JAX-friendly
    updated_bullets = bullets
    enemy_bullet_start = PLAYER_BULLET_LIMIT
    num_enemies = obstacles_count

    def add_enemy_bullets_loop(bullets_state):
        def body(i, b):
            slot = enemy_bullet_start + (i % ENEMY_BULLET_LIMIT)
            cond = fire_flags[i]
            def set_b(bb):
                return Bullet(
                    x=bb.x.at[slot].set(bullet_xs[i]),
                    y=bb.y.at[slot].set(bullet_ys[i]),
                    z=bb.z.at[slot].set(3.0),
                    vel_x=bb.vel_x.at[slot].set(bullet_vel_xs[i]),
                    vel_y=bb.vel_y.at[slot].set(bullet_vel_ys[i]),
                    active=bb.active.at[slot].set(1),
                    lifetime=bb.lifetime.at[slot].set(BULLET_LIFETIME),
                    owner=bb.owner.at[slot].set(i + 1)
                )
            return jax.lax.cond(cond, set_b, lambda bb: bb, b)

        return jax.lax.fori_loop(0, num_enemies, body, bullets_state)

    updated_bullets = add_enemy_bullets_loop(updated_bullets)

    return updated_obstacles, updated_bullets

@jax.jit
def spawn_new_enemy(obstacles: Obstacle, player_tank: Tank, step_counter: chex.Array) -> Obstacle:
    """Spawn a new enemy according to type-specific Atari BattleZone rules.

    Rules implemented:
    - TANK/SUPERTANK: spawn at random angle around player; choose NEAR or FAR radius 50/50
    - FIGHTER: spawn directly in front of the player at a small offset
    - SAUCER: spawn at a random angle and radius anywhere in the map; may spawn while hostiles are active

    Only one hostile (tank/supertank/fighter) will be active at once. If a hostile is active,
    this function will only allow saucer spawns.
    """

    # Count total active entities
    active_enemies = jnp.sum(obstacles.alive)

    # Determine whether any hostile (non-saucer) enemy is active
    hostile_mask = jnp.logical_and(obstacles.obstacle_type == 0,
                                   jnp.logical_and(obstacles.alive == 1, obstacles.enemy_subtype != ENEMY_TYPE_SAUCER))
    hostile_active = jnp.any(hostile_mask)

    # Only spawn if we have room and there is a dead slot
    should_spawn = active_enemies < MAX_ACTIVE_ENEMIES
    dead_enemy_idx = jnp.argmax(jnp.logical_not(obstacles.alive))
    has_dead_slot = jnp.logical_not(obstacles.alive[dead_enemy_idx])
    can_spawn = jnp.logical_and(should_spawn, has_dead_slot)

    # Deterministic pseudo-random scalars based on step_counter and slot index
    r1 = jnp.abs(jnp.sin(step_counter * 0.093 + dead_enemy_idx * 0.37))
    r2 = jnp.abs(jnp.cos(step_counter * 0.127 + dead_enemy_idx * 0.19))

    # Choose subtype: if a hostile is already active, only allow SAUCER; otherwise pick from hostile types
    def choose_hostile():
        c = jnp.floor(r1 * 3.0).astype(jnp.int32)
        return jnp.where(c == 0, ENEMY_TYPE_TANK, jnp.where(c == 1, ENEMY_TYPE_SUPERTANK, ENEMY_TYPE_FIGHTER))

    chosen_subtype = jax.lax.cond(hostile_active, lambda: ENEMY_TYPE_SAUCER, choose_hostile)

    # Compute spawn angle and distance per-subtype
    angle_noise = (r1 * 2.0 * math.pi + r2 * 1.3) % (2.0 * math.pi)
    near_choice = (jnp.floor(r2 * 2.0) % 2) == 0

    # defaults
    spawn_angle = angle_noise
    spawn_distance = ENEMY_SPAWN_DISTANCE_MAX

    # TANK / SUPERTANK: random angle, choose NEAR or FAR
    is_tank_like = jnp.logical_or(chosen_subtype == ENEMY_TYPE_TANK, chosen_subtype == ENEMY_TYPE_SUPERTANK)
    spawn_angle = jnp.where(is_tank_like, angle_noise, spawn_angle)
    spawn_distance = jnp.where(is_tank_like, jnp.where(near_choice, SPAWN_NEAR_RADIUS, SPAWN_FAR_RADIUS), spawn_distance)

    # FIGHTER: spawn directly in front of player's facing direction at small offset
    fighter_offset = SPAWN_NEAR_RADIUS * 0.5
    spawn_angle = jnp.where(chosen_subtype == ENEMY_TYPE_FIGHTER, player_tank.angle, spawn_angle)
    spawn_distance = jnp.where(chosen_subtype == ENEMY_TYPE_FIGHTER, fighter_offset, spawn_distance)

    # SAUCER: random radius anywhere in map
    saucer_radius = jnp.abs(jnp.sin(step_counter * 0.19 + dead_enemy_idx * 0.11)) * MAP_RADIUS
    spawn_angle = jnp.where(chosen_subtype == ENEMY_TYPE_SAUCER, angle_noise, spawn_angle)
    spawn_distance = jnp.where(chosen_subtype == ENEMY_TYPE_SAUCER, saucer_radius, spawn_distance)

    # Convert polar to Cartesian
    # Ensure spawn distance is within configured min/max so enemies are visible on the ground
    spawn_distance = jnp.clip(spawn_distance, ENEMY_SPAWN_DISTANCE_MIN, ENEMY_SPAWN_DISTANCE_MAX)

    spawn_x = player_tank.x + jnp.cos(spawn_angle) * spawn_distance
    spawn_y = player_tank.y + jnp.sin(spawn_angle) * spawn_distance

    # Clamp spawn position
    spawn_x = jnp.clip(spawn_x, BOUNDARY_MIN, BOUNDARY_MAX)
    spawn_y = jnp.clip(spawn_y, BOUNDARY_MIN, BOUNDARY_MAX)

    initial_angle = jnp.arctan2(player_tank.y - spawn_y, player_tank.x - spawn_x)

    # Prepare updated arrays, set subtype for the new slot
    new_x = jnp.where(can_spawn, obstacles.x.at[dead_enemy_idx].set(spawn_x), obstacles.x)
    new_y = jnp.where(can_spawn, obstacles.y.at[dead_enemy_idx].set(spawn_y), obstacles.y)
    new_angle = jnp.where(can_spawn, obstacles.angle.at[dead_enemy_idx].set(initial_angle), obstacles.angle)
    new_alive = jnp.where(can_spawn, obstacles.alive.at[dead_enemy_idx].set(1), obstacles.alive)
    new_fire_cooldown = jnp.where(can_spawn, obstacles.fire_cooldown.at[dead_enemy_idx].set(ENEMY_SPAWN_WAIT), obstacles.fire_cooldown)
    new_ai_state = jnp.where(can_spawn, obstacles.ai_state.at[dead_enemy_idx].set(ENEMY_STATE_HUNT), obstacles.ai_state)
    new_target_angle = jnp.where(can_spawn, obstacles.target_angle.at[dead_enemy_idx].set(initial_angle), obstacles.target_angle)
    new_state_timer = jnp.where(can_spawn, obstacles.state_timer.at[dead_enemy_idx].set(ENEMY_SPAWN_WAIT), obstacles.state_timer)
    new_obstacle_type = jnp.where(can_spawn, obstacles.obstacle_type.at[dead_enemy_idx].set(0), obstacles.obstacle_type)
    new_enemy_subtype = jnp.where(can_spawn, obstacles.enemy_subtype.at[dead_enemy_idx].set(chosen_subtype), obstacles.enemy_subtype)

    # Debug printing
    if DEBUG_ENEMY_SPAWN:
        jax.debug.print('[SPAWN] idx={} type={} angle={:.2f} dist={:.1f}', dead_enemy_idx, chosen_subtype, spawn_angle, spawn_distance)

    # TODO: add overlap/obstacle checks to avoid spawning inside other objects

    return Obstacle(
        x=new_x,
        y=new_y,
        obstacle_type=new_obstacle_type,
        enemy_subtype=new_enemy_subtype,
        angle=new_angle,
        alive=new_alive,
        fire_cooldown=new_fire_cooldown,
        ai_state=new_ai_state,
        target_angle=new_target_angle,
        state_timer=new_state_timer
    )

class JaxBattleZone(JaxEnvironment[BattleZoneState, BattleZoneObservation, chex.Array, BattleZoneInfo]):
    def __init__(self, reward_funcs: list[callable] = None):
        super().__init__()
        self.frame_stack_size = 4
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.action_set = list(range(18))  # All 18 BattleZone actions
        self.obs_size = 50
        # Non-jitted renderer for raster output (used by image_space/render)
        try:
            self.renderer = BattleZoneRenderer()
        except Exception:
            self.renderer = None

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
            z=jnp.zeros(MAX_BULLETS),
            vel_x=jnp.zeros(MAX_BULLETS),
            vel_y=jnp.zeros(MAX_BULLETS),
            active=jnp.zeros(MAX_BULLETS, dtype=jnp.int32),
            lifetime=jnp.zeros(MAX_BULLETS),
            owner=jnp.zeros(MAX_BULLETS)
        )
        
        # Start with fewer enemies, closer to player like original game
        enemy_positions_x = jnp.array([200.0, -200.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        enemy_positions_y = jnp.array([0.0, 0.0, 200.0, -200.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        enemy_types = jnp.zeros(16)  # All enemy tanks (type 0)
        enemy_subtypes = jnp.array([0, 0, 0, 0] + [-1] * 12)
        # Initial angles for enemy tanks
        enemy_angles = jnp.array([jnp.pi, 0.0, -jnp.pi/2, jnp.pi/2, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # Only first 4 enemies alive initially
        enemy_alive = jnp.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)
        enemy_fire_cooldown = jnp.zeros(16)
        enemy_ai_state = jnp.zeros(16, dtype=jnp.int32)
        enemy_target_angle = enemy_angles.copy()
        enemy_state_timer = jnp.zeros(16)
        
        obstacles = Obstacle(
            x=enemy_positions_x,
            y=enemy_positions_y,
            obstacle_type=enemy_types,
            enemy_subtype=enemy_subtypes,
            angle=enemy_angles,
            alive=enemy_alive,
            fire_cooldown=enemy_fire_cooldown,
            ai_state=enemy_ai_state,
            target_angle=enemy_target_angle,
            state_timer=enemy_state_timer
        )
        
        state = BattleZoneState(
            player_tank=player_tank,
            bullets=bullets,
            obstacles=obstacles,
            step_counter=jnp.array(0),
            spawn_timer=jnp.array(ENEMY_SPAWN_COOLDOWN),
            prev_player_x=player_tank.x,
            prev_player_y=player_tank.y,
            player_score=jnp.array(0),   # initialize score
            player_lives=jnp.array(5)    # initialize lives (5 as requested)
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
        inactive_player_slots = jnp.logical_not(state.bullets.active[:PLAYER_BULLET_LIMIT])
        inactive_bullet_idx = jnp.argmax(inactive_player_slots)
        can_fire = inactive_player_slots[inactive_bullet_idx]
        
        new_bullet = create_bullet(new_player_tank, jnp.array(0))
        
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
        
        # Update enemy tanks AI
        updated_obstacles, updated_bullets = update_enemy_tanks(
            state.obstacles, new_player_tank, updated_bullets, state.step_counter
        )
        
        # Handle enemy spawning
        new_spawn_timer = state.spawn_timer - 1
        should_try_spawn = new_spawn_timer <= 0
        
        updated_obstacles = jax.lax.cond(
            should_try_spawn,
            lambda obs: spawn_new_enemy(obs, new_player_tank, state.step_counter),
            lambda obs: obs,
            updated_obstacles
        )
        
        new_spawn_timer = jnp.where(should_try_spawn, ENEMY_SPAWN_COOLDOWN, new_spawn_timer)
        
        # Check for bullet-obstacle collisions (now returns score delta when player killed obstacles)
        updated_bullets, updated_obstacles, score_delta = check_bullet_obstacle_collisions(updated_bullets, updated_obstacles)
        
        # Check if player is hit by enemy bullets
        new_player_tank = check_player_hit(new_player_tank, updated_bullets)

        # Detect whether the player was shot this step (alive -> dead)
        player_was_shot = jnp.logical_and(state.player_tank.alive == 1, new_player_tank.alive == 0)

        # After computing new_player_tank and collisions, we already created a tentative new_state.
        # If the player was shot this frame, decrement lives. If lives remain, restart game (reset)
        lives_after = state.player_lives - 1

        # Build a reset-state (same layout as reset) but preserve score and set player_lives = lives_after
        def build_reset_state():
            # player tank reset
            player_tank_rst = Tank(x=jnp.array(0.0), y=jnp.array(0.0), angle=jnp.array(0.0), alive=jnp.array(1, dtype=jnp.int32))
            # bullets cleared
            bullets_rst = Bullet(
                x=jnp.zeros(MAX_BULLETS),
                y=jnp.zeros(MAX_BULLETS),
                z=jnp.zeros(MAX_BULLETS),
                vel_x=jnp.zeros(MAX_BULLETS),
                vel_y=jnp.zeros(MAX_BULLETS),
                active=jnp.zeros(MAX_BULLETS, dtype=jnp.int32),
                lifetime=jnp.zeros(MAX_BULLETS),
                owner=jnp.zeros(MAX_BULLETS)
            )
            # initial enemy positions/angles (same as reset)
            enemy_positions_x = jnp.array([200.0, -200.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            enemy_positions_y = jnp.array([0.0, 0.0, 200.0, -200.0, 0.0, 0.0, 0.0, 0.0,
                                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            enemy_types = jnp.zeros(16)
            enemy_angles = jnp.array([jnp.pi, 0.0, -jnp.pi/2, jnp.pi/2] + [0.0]*12)
            enemy_alive = jnp.array([1,1,1,1] + [0]*12, dtype=jnp.int32)
            enemy_fire_cooldown = jnp.zeros(16)
            enemy_ai_state = jnp.zeros(16, dtype=jnp.int32)
            enemy_target_angle = enemy_angles.copy()
            enemy_state_timer = jnp.zeros(16)

            obstacles_rst = Obstacle(
                x=enemy_positions_x,
                y=enemy_positions_y,
                obstacle_type=enemy_types,
                enemy_subtype=jnp.array([0,0,0,0] + [-1]*12),
                angle=enemy_angles,
                alive=enemy_alive,
                fire_cooldown=enemy_fire_cooldown,
                ai_state=enemy_ai_state,
                target_angle=enemy_target_angle,
                state_timer=enemy_state_timer
            )

            return BattleZoneState(
                player_tank=player_tank_rst,
                bullets=bullets_rst,
                obstacles=obstacles_rst,
                step_counter=jnp.array(0),
                spawn_timer=jnp.array(ENEMY_SPAWN_COOLDOWN),
                prev_player_x=player_tank_rst.x,
                prev_player_y=player_tank_rst.y,
                player_score=state.player_score,      # preserve score
                player_lives=lives_after
            )

        # If player was shot and still has lives left -> restart. If no lives left, keep dead state (game over).
        def keep_or_gameover():
            # if lives_after > 0 then return reset state else return the normal new_state with decremented lives (game over)
            def return_reset():
                return build_reset_state()
            def return_gameover():
                # keep new_state but with updated lives (0) so _get_done() will see player dead
                return BattleZoneState(
                    player_tank=new_player_tank,
                    bullets=updated_bullets,
                    obstacles=updated_obstacles,
                    step_counter=state.step_counter + 1,
                    spawn_timer=new_spawn_timer,
                    prev_player_x=state.player_tank.x,
                    prev_player_y=state.player_tank.y,
                    player_score=state.player_score + score_delta,
                    player_lives=lives_after
                )
            return jax.lax.cond(lives_after > 0, return_reset, return_gameover)

        final_state = jax.lax.cond(player_was_shot, lambda: keep_or_gameover(), lambda: BattleZoneState(
            player_tank=new_player_tank,
            bullets=updated_bullets,
            obstacles=updated_obstacles,
            step_counter=state.step_counter + 1,
            spawn_timer=new_spawn_timer,
            prev_player_x=state.player_tank.x,
            prev_player_y=state.player_tank.y,
            player_score=state.player_score + score_delta,
            player_lives=state.player_lives
        ))

        observation = self._get_observation(final_state)
        reward = self._get_env_reward(state, final_state)
        done = self._get_done(final_state)
        all_rewards = self._get_all_reward(state, final_state)
        info = self._get_info(final_state, all_rewards, player_was_shot)

        return observation, final_state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: BattleZoneState) -> BattleZoneObservation:
        return BattleZoneObservation(
            player_tank=state.player_tank,
            bullets=state.bullets,
            obstacles=state.obstacles
        )

    @property
    def action_space(self) -> spaces.Discrete:
        """Discrete action space as an attribute (gym-style).

        Returning a property ensures wrappers that expect `env.action_space`
        (attribute access) get a Space instance rather than a bound method.
        """
        # action_set is defined elsewhere on this class; keep the same size
        return spaces.Discrete(len(self.action_set))

    @property
    def observation_space(self) -> spaces.Dict:
        """Observation space as an attribute (gym-style) using numpy dtypes.

        Tests and wrappers expect numpy dtypes (np.float32 / np.int32).
        """
        # Constants referenced below exist earlier in the file (MAX_BULLETS, MAX_OBSTACLES, etc.)
        MAX_BULLETS = getattr(self, 'MAX_BULLETS', 32)
        MAX_OBSTACLES = getattr(self, 'MAX_OBSTACLES', 32)
        BOUNDARY_MIN = getattr(self, 'WORLD_MIN', -1000.0)
        BOUNDARY_MAX = getattr(self, 'WORLD_MAX', 1000.0)
        BULLET_LIFETIME = getattr(self, 'BULLET_LIFETIME', 120)
        ENEMY_FIRE_COOLDOWN = getattr(self, 'ENEMY_FIRE_COOLDOWN', 60)
        ENEMY_SPAWN_COOLDOWN = getattr(self, 'ENEMY_SPAWN_COOLDOWN', 180)

        player_space = spaces.Dict({
            "x": spaces.Box(low=BOUNDARY_MIN, high=BOUNDARY_MAX, shape=(), dtype=np.float32),
            "y": spaces.Box(low=BOUNDARY_MIN, high=BOUNDARY_MAX, shape=(), dtype=np.float32),
            "angle": spaces.Box(low=-math.pi, high=math.pi, shape=(), dtype=np.float32),
            "alive": spaces.Box(low=0, high=1, shape=(), dtype=np.int32),
        })

        bullets_space = spaces.Dict({
            "x": spaces.Box(low=BOUNDARY_MIN, high=BOUNDARY_MAX, shape=(MAX_BULLETS,), dtype=np.float32),
            "y": spaces.Box(low=BOUNDARY_MIN, high=BOUNDARY_MAX, shape=(MAX_BULLETS,), dtype=np.float32),
            "z": spaces.Box(low=0.0, high=1000.0, shape=(MAX_BULLETS,), dtype=np.float32),
            "vel_x": spaces.Box(low=-1000.0, high=1000.0, shape=(MAX_BULLETS,), dtype=np.float32),
            "vel_y": spaces.Box(low=-1000.0, high=1000.0, shape=(MAX_BULLETS,), dtype=np.float32),
            "active": spaces.Box(low=0, high=1, shape=(MAX_BULLETS,), dtype=np.int32),
            "lifetime": spaces.Box(low=0, high=BULLET_LIFETIME, shape=(MAX_BULLETS,), dtype=np.int32),
            "owner": spaces.Box(low=0, high=MAX_BULLETS, shape=(MAX_BULLETS,), dtype=np.int32),
        })

        obstacles_space = spaces.Dict({
            "x": spaces.Box(low=BOUNDARY_MIN, high=BOUNDARY_MAX, shape=(MAX_OBSTACLES,), dtype=np.float32),
            "y": spaces.Box(low=BOUNDARY_MIN, high=BOUNDARY_MAX, shape=(MAX_OBSTACLES,), dtype=np.float32),
            "obstacle_type": spaces.Box(low=0, high=10, shape=(MAX_OBSTACLES,), dtype=np.int32),
            "enemy_subtype": spaces.Box(low=-1, high=10, shape=(MAX_OBSTACLES,), dtype=np.int32),
            "angle": spaces.Box(low=-math.pi, high=math.pi, shape=(MAX_OBSTACLES,), dtype=np.float32),
            "alive": spaces.Box(low=0, high=1, shape=(MAX_OBSTACLES,), dtype=np.int32),
            "fire_cooldown": spaces.Box(low=0, high=ENEMY_FIRE_COOLDOWN * 4, shape=(MAX_OBSTACLES,), dtype=np.int32),
            "ai_state": spaces.Box(low=0, high=10, shape=(MAX_OBSTACLES,), dtype=np.int32),
            "target_angle": spaces.Box(low=-math.pi, high=math.pi, shape=(MAX_OBSTACLES,), dtype=np.float32),
            "state_timer": spaces.Box(low=0, high=ENEMY_SPAWN_COOLDOWN * 4, shape=(MAX_OBSTACLES,), dtype=np.int32),
        })

        return spaces.Dict({
            "player_tank": player_space,
            "bullets": bullets_space,
            "obstacles": obstacles_space,
            "step_counter": spaces.Box(low=0, high=np.iinfo(np.int32).max, shape=(), dtype=np.int32),
            "spawn_timer": spaces.Box(low=0, high=ENEMY_SPAWN_COOLDOWN * 4, shape=(), dtype=np.int32),
            "player_score": spaces.Box(low=0, high=np.iinfo(np.int32).max, shape=(), dtype=np.int32),
            "player_lives": spaces.Box(low=0, high=99, shape=(), dtype=np.int32),
        })

    def obs_to_flat_array(self, obs) -> np.ndarray:
        """Convert observation to a flat numpy float32 array.

        Tests/wrappers expect a numpy ndarray (not a jax array) from this method.
        """
        # Player
        pt = obs.player_tank
        player_flat = np.array([float(pt.x), float(pt.y), float(pt.angle), int(pt.alive)], dtype=np.float32).reshape(-1)

        # Bullets
        b = obs.bullets
        # Convert each field to numpy and concatenate
        bullets_flat = np.concatenate([
            np.asarray(b.x, dtype=np.float32).reshape(-1),
            np.asarray(b.y, dtype=np.float32).reshape(-1),
            np.asarray(b.z, dtype=np.float32).reshape(-1),
            np.asarray(b.vel_x, dtype=np.float32).reshape(-1),
            np.asarray(b.vel_y, dtype=np.float32).reshape(-1),
            np.asarray(b.active, dtype=np.float32).reshape(-1),
            np.asarray(b.lifetime, dtype=np.float32).reshape(-1),
            np.asarray(b.owner, dtype=np.float32).reshape(-1),
        ])

        # Obstacles
        o = obs.obstacles
        obstacles_flat = np.concatenate([
            np.asarray(o.x, dtype=np.float32).reshape(-1),
            np.asarray(o.y, dtype=np.float32).reshape(-1),
            np.asarray(o.obstacle_type, dtype=np.float32).reshape(-1),
            np.asarray(getattr(o, 'enemy_subtype', np.full_like(o.x, -1)), dtype=np.float32).reshape(-1),
            np.asarray(o.angle, dtype=np.float32).reshape(-1),
            np.asarray(o.alive, dtype=np.float32).reshape(-1),
            np.asarray(o.fire_cooldown, dtype=np.float32).reshape(-1),
            np.asarray(o.ai_state, dtype=np.float32).reshape(-1),
            np.asarray(o.target_angle, dtype=np.float32).reshape(-1),
            np.asarray(o.state_timer, dtype=np.float32).reshape(-1),
        ])

        return np.concatenate([player_flat, bullets_flat, obstacles_flat]).astype(np.float32)

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
        # Game ends if player dies
        return state.player_tank.alive == 0

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: BattleZoneState, all_rewards: chex.Array, player_shot: chex.Array) -> BattleZoneInfo:
        return BattleZoneInfo(time=state.step_counter, all_rewards=all_rewards, player_shot=player_shot)

class BattleZoneRenderer:
    """3D wireframe renderer for BattleZone in the style of the original Atari 2600 game."""

    def __init__(self):
        self.view_distance = 400.0  # How far we can see
        self.fov = 60.0  # Field of view in degrees

        # HUD bar height used for top radar container and bottom HUD
        self.hud_bar_height = 28

        # HUD font (requires pygame to be initialized before renderer creation)
        try:
            pygame.font.init()
            self.hud_font = pygame.font.SysFont("monospace", 14, bold=True)
        except Exception:
            self.hud_font = None

        # Load player tank sprite
        try:
            # Try both possible locations for the sprite
            sprite_paths = [
                os.path.join(os.path.dirname(__file__), "../sprites/battlezone/player_tank2.npy"),
                os.path.join(os.path.dirname(__file__), "sprites/battlezone/player_tank2.npy"),
            ]
            for path in sprite_paths:
                if os.path.exists(path):
                    self.player_tank_sprite = np.load(path)
                    self.sprite_loaded = True
                    break
            else:
                self.sprite_loaded = False
        except Exception:
            self.sprite_loaded = False

        # Load life icon sprite (scaled down in HUD). Try a couple of likely paths.
        try:
            life_paths = [
                os.path.join(os.path.dirname(__file__), "../sprites/battlezone/tank_life.npy"),
                os.path.join(os.path.dirname(__file__), "sprites/battlezone/tank_life.npy"),
            ]
            self.life_sprite = None
            self.life_sprite_loaded = False
            for lp in life_paths:
                if os.path.exists(lp):
                    try:
                        self.life_sprite = np.load(lp)
                        self.life_sprite_loaded = True
                        break
                    except Exception:
                        continue
        except Exception:
            self.life_sprite = None
            self.life_sprite_loaded = False

    def draw_player_bullet(self, screen, state: BattleZoneState):
        """Draw all bullets (player and enemy) as moving lines."""
        bullets = state.bullets
        
        # Draw all active bullets
        for i in range(MAX_BULLETS):
            if bullets.active[i]:
                bullet_x = bullets.x[i]
                bullet_y = bullets.y[i]
                owner = bullets.owner[i]

                # Transform bullet position to screen
                screen_x, screen_y, distance, visible = self.world_to_screen_3d(
                    bullet_x, bullet_y,
                    state.player_tank.x,
                    state.player_tank.y,
                    state.player_tank.angle
                )

                if visible and 0 <= screen_x < WIDTH and 0 <= screen_y < HEIGHT:
                    # Different colors for player vs enemy bullets
                    color = BULLET_COLOR if owner == 0 else (255, 100, 100)  # Red for enemy bullets
                    pygame.draw.line(
                        screen,
                        color,
                        (screen_x, screen_y - 3),
                        (screen_x, screen_y + 3),
                        3
                    )


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
        """Convert world coordinates to 3D screen coordinates relative to player.
        
        Coordinate system:
        - World: X increases right, Y increases down
        - Player angle: 0=right, π/2=down, π=left, -π/2=up
        - View space: X is left(-)/right(+) relative to player, Y is forward(+)/back(-) relative to player
        - Screen: X increases right, Y increases down, with horizon at HORIZON_Y
        """
        # Translate to player-relative coordinates
        rel_x = world_x - player_x
        rel_y = world_y - player_y  
        
        # Rotate by player angle to get view-relative coordinates
        # We need to rotate the world coordinates to align with player's facing direction
        cos_a = jnp.cos(player_angle)
        sin_a = jnp.sin(player_angle)
        
        # CORRECTED transformation matrix:
        # When player faces right (angle=0): forward should be +X direction in world
        # When player faces down (angle=π/2): forward should be +Y direction in world
        # The transformation should map world directions to view directions correctly
        
        # For a proper view transformation:
        # - view_x (left/right): perpendicular to player's facing direction
        # - view_y (forward/back): aligned with player's facing direction
        view_x = -rel_x * sin_a + rel_y * cos_a   # Right/left relative to player (perpendicular to facing)
        view_y = rel_x * cos_a + rel_y * sin_a    # Forward/back relative to player (parallel to facing)
                
        # Perspective projection
        if view_y > 1.0:  # Object is in front of player (positive forward distance)
            # Standard perspective projection with proper FOV scaling
            fov_scale = 80.0  # Field of view scaling factor
            screen_x = int(WIDTH // 2 + (view_x / view_y) * fov_scale)
            
            # For proper 3D perspective:
            # - Objects farther away should appear higher on screen (closer to horizon)
            # - Objects closer should appear lower on screen (away from horizon)
            # - The perspective scale should make closer objects larger
            perspective_scale = 100.0  # Controls how much perspective affects Y position
            screen_y = int(HORIZON_Y + (perspective_scale / view_y))
            
            distance = view_y
            return screen_x, screen_y, distance, True
        else:
            return 0, 0, 0, False  # Behind player or too close

    def draw_enemy_tank_frontal(self, screen, x, y, scale, color):
        """Draw frontal view of enemy tank matching BattleZone style."""
        try:
            # Tank body - more angular and authentic BattleZone style
            body_width = int(scale * 1.0)
            body_height = int(scale * 0.7)
            
            # Main body - trapezoidal shape
            body_points = [
                (x - body_width//2, y + body_height//2),      # bottom-left
                (x + body_width//2, y + body_height//2),      # bottom-right
                (x + body_width//3, y - body_height//2),      # top-right
                (x - body_width//3, y - body_height//2),      # top-left
            ]
            pygame.draw.polygon(screen, color, body_points, 1)
            
            # Turret - small rectangular turret
            turret_width = scale // 3
            turret_height = scale // 3
            pygame.draw.rect(screen, color,
                           (x - turret_width//2, y - turret_height//2, turret_width, turret_height), 1)
            
            # Cannon - thin line extending forward
            cannon_length = scale // 2
            pygame.draw.line(screen, color, (x, y), (x, y + cannon_length), 1)
            
            # Tank treads/tracks - vertical lines on sides
            track_height = body_height
            for i in range(3):
                track_x_left = x - body_width//2 - 2
                track_x_right = x + body_width//2 + 2
                track_y = y - track_height//2 + i * (track_height//3)
                pygame.draw.line(screen, color, (track_x_left, track_y), (track_x_left, track_y + 3), 1)
                pygame.draw.line(screen, color, (track_x_right, track_y), (track_x_right, track_y + 3), 1)
            
        except:
            pass

    def draw_enemy_tank_profile_left(self, screen, x, y, scale, color):
        """Draw left profile of enemy tank matching BattleZone style."""
        try:
            # Tank body - elongated hexagonal shape
            body_width = int(scale * 1.8)
            body_height = int(scale * 0.6)
            
            # Main body points for side view
            body_points = [
                (x - body_width//2, y + body_height//3),      # rear-bottom
                (x - body_width//3, y + body_height//2),      # rear-bottom-slope
                (x + body_width//3, y + body_height//2),      # front-bottom-slope
                (x + body_width//2, y + body_height//3),      # front-bottom
                (x + body_width//2 - 2, y - body_height//2),  # front-top
                (x - body_width//2 + 2, y - body_height//2),  # rear-top
            ]
            pygame.draw.polygon(screen, color, body_points, 1)
            
            # Turret - offset rectangular turret
            turret_width = scale // 2
            turret_height = scale // 4
            turret_x = x - scale // 6
            pygame.draw.rect(screen, color,
                           (turret_x - turret_width//2, y - turret_height//2, turret_width, turret_height), 1)
            
            # Cannon pointing left
            cannon_length = int(scale * 1.2)
            pygame.draw.line(screen, color, (turret_x, y), (turret_x - cannon_length, y), 2)
            
            # Track wheels/details
            wheel_y = y + body_height//2 + 2
            for i in range(0, body_width - 4, 6):
                wheel_x = x - body_width//2 + 2 + i
                pygame.draw.circle(screen, color, (wheel_x, wheel_y), 2, 1)
            
            # Track line
            pygame.draw.line(screen, color, 
                           (x - body_width//2, wheel_y), 
                           (x + body_width//2, wheel_y), 1)
            
        except:
            pass

    def draw_enemy_tank_profile_right(self, screen, x, y, scale, color):
        """Draw right profile of enemy tank matching BattleZone style."""
        try:
            # Tank body - elongated hexagonal shape (mirrored)
            body_width = int(scale * 1.8)
            body_height = int(scale * 0.6)
            
            # Main body points for side view
            body_points = [
                (x + body_width//2, y + body_height//3),      # front-bottom
                (x + body_width//3, y + body_height//2),      # front-bottom-slope
                (x - body_width//3, y + body_height//2),      # rear-bottom-slope
                (x - body_width//2, y + body_height//3),      # rear-bottom
                (x - body_width//2 + 2, y - body_height//2),  # rear-top
                (x + body_width//2 - 2, y - body_height//2),  # front-top
            ]
            pygame.draw.polygon(screen, color, body_points, 1)
            
            # Turret - offset rectangular turret
            turret_width = scale // 2
            turret_height = scale // 4
            turret_x = x + scale // 6
            pygame.draw.rect(screen, color,
                           (turret_x - turret_width//2, y - turret_height//2, turret_width, turret_height), 1)
            
            # Cannon pointing right
            cannon_length = int(scale * 1.2)
            pygame.draw.line(screen, color, (turret_x, y), (turret_x + cannon_length, y), 2)
            
            # Track wheels/details
            wheel_y = y + body_height//2 + 2
            for i in range(0, body_width - 4, 6):
                wheel_x = x - body_width//2 + 2 + i
                pygame.draw.circle(screen, color, (wheel_x, wheel_y), 2, 1)
            
            # Track line
            pygame.draw.line(screen, color, 
                           (x - body_width//2, wheel_y), 
                           (x + body_width//2, wheel_y), 1)
            
        except:
            pass

    def draw_enemy_tank(self, screen, x, y, distance, color, tank_angle, player_angle):
        """Draw enemy tank with directional appearance based on relative orientation."""
        if distance > self.view_distance:
            return
            
        # Scale based on distance for perspective
        scale = max(4, int(20 / max(distance / 50, 1)))
        
        # Calculate relative angle between tank and player
        # Determine which view to show based on tank's orientation relative to player's view
        relative_angle = tank_angle - player_angle
        
        # Normalize angle to [-π, π]
        relative_angle = math.atan2(math.sin(relative_angle), math.cos(relative_angle))
        
        # Determine tank appearance based on relative angle
        if abs(relative_angle) < math.pi/4 or abs(relative_angle) > 3*math.pi/4:
            # Tank is facing toward or away from player (frontal view)
            self.draw_enemy_tank_frontal(screen, x, y, scale, color)
        elif relative_angle > 0:
            # Tank is facing to the left relative to player view
            self.draw_enemy_tank_profile_left(screen, x, y, scale, color)
        else:
            # Tank is facing to the right relative to player view
            self.draw_enemy_tank_profile_right(screen, x, y, scale, color)

    def draw_player_tank(self, screen):
        """Draw player tank using sprite or wireframe fallback."""
        base_x = WIDTH // 2
        # Place the tank directly above the bottom HUD bar so it is visible over the scene
        base_y = HEIGHT - self.hud_bar_height - 10  # slight vertical offset above the bar

        if self.sprite_loaded:
            sprite_surface = self.numpy_to_pygame_surface(self.player_tank_sprite)
            sprite_width, sprite_height = sprite_surface.get_size()
            scale_factor = 0.1  # Adjust this to make sprite bigger/smaller
            scaled_width = int(sprite_width * scale_factor)
            scaled_height = int(sprite_height * scale_factor)
            sprite_surface = pygame.transform.scale(sprite_surface, (scaled_width, scaled_height))
            sprite_rect = sprite_surface.get_rect()
            # Align the bottom center of the sprite directly above the bottom HUD bar
            sprite_rect.midbottom = (base_x, HEIGHT - self.hud_bar_height - 1)
            screen.blit(sprite_surface, sprite_rect)
        else:
            # Fallback to wireframe rendering
            pygame.draw.ellipse(screen, WIREFRAME_COLOR, (base_x - 30, base_y - 10, 60, 20), 1)
            pygame.draw.rect(screen, WIREFRAME_COLOR, (base_x - 3, base_y - 30, 6, 20), 1)
            pygame.draw.rect(screen, WIREFRAME_COLOR, (base_x - 40, base_y - 10, 10, 20), 1)
            pygame.draw.rect(screen, WIREFRAME_COLOR, (base_x + 30, base_y - 10, 10, 20), 1)

    def draw_radar(self, screen, state: BattleZoneState):
        """Draw the BattleZone radar matching the classic appearance with proper rotation."""
        radar_radius = int(WIDTH * 0.12)
        # Top black bar to contain radar (full width)
        top_bar_height = radar_radius * 2 + 12
        pygame.draw.rect(screen, (0, 0, 0), (0, 0, WIDTH, top_bar_height))

        # Center the radar horizontally in that top bar
        radar_center_x = WIDTH // 2
        radar_center_y = top_bar_height // 2

        # Draw radar circle at centered location
        pygame.draw.circle(screen, (0, 255, 0), (radar_center_x, radar_center_y), radar_radius, 1)

        # Radar sweep (keep same visual behaviour but centered)
        sweep_speed = 0.025
        angle = float(state.step_counter) * sweep_speed % (2 * math.pi)
        sweep_length = radar_radius - 2
        sweep_x = int(radar_center_x + sweep_length * math.cos(angle - math.pi/2))
        sweep_y = int(radar_center_y + sweep_length * math.sin(angle - math.pi/2))
        pygame.draw.line(screen, (255, 255, 255), (radar_center_x, radar_center_y), (sweep_x, sweep_y), 2)

        # Add iconic short diagonal ticks at top-left and top-right of the radar (as in original)
        try:
            # angles slightly left/right of the top (-pi/2)
            left_ang = -math.pi/2 - 0.6
            right_ang = -math.pi/2 + 0.6
            # start just inside the circle edge and draw inward toward center
            edge_r = radar_radius - 2
            # original outward length was roughly 12; use 60% of that (~7 pixels)
            tick_len = max(4, int(12 * 0.6))

            lx_start = int(radar_center_x + edge_r * math.cos(left_ang))
            ly_start = int(radar_center_y + edge_r * math.sin(left_ang))
            lx_end = int(radar_center_x + (edge_r - tick_len) * math.cos(left_ang))
            ly_end = int(radar_center_y + (edge_r - tick_len) * math.sin(left_ang))

            rx_start = int(radar_center_x + edge_r * math.cos(right_ang))
            ry_start = int(radar_center_y + edge_r * math.sin(right_ang))
            rx_end = int(radar_center_x + (edge_r - tick_len) * math.cos(right_ang))
            ry_end = int(radar_center_y + (edge_r - tick_len) * math.sin(right_ang))

            pygame.draw.line(screen, WIREFRAME_COLOR, (lx_start, ly_start), (lx_end, ly_end), 2)
            pygame.draw.line(screen, WIREFRAME_COLOR, (rx_start, ry_start), (rx_end, ry_end), 2)
        except Exception:
            pass

        # Radar scale as before
        scale = (radar_radius - 4) / (WORLD_SIZE / 2)

        # Convert player values
        player_x = float(state.player_tank.x)
        player_y = float(state.player_tank.y)
        player_angle = float(state.player_tank.angle)
        
        # Draw obstacles/bullets relative positions using centered radar origin
        for i, (ox, oy, alive) in enumerate(zip(state.obstacles.x, state.obstacles.y, state.obstacles.alive)):
            if alive:
                ox_f = float(ox); oy_f = float(oy)
                # Choose color by enemy subtype when available
                try:
                    subtype_val = int(state.obstacles.enemy_subtype[i])
                except Exception:
                    subtype_val = -1
                if subtype_val == ENEMY_TYPE_SUPERTANK:
                    enemy_color = SUPERTANK_COLOR
                elif subtype_val == ENEMY_TYPE_TANK:
                    enemy_color = TANK_COLOR
                elif subtype_val == ENEMY_TYPE_FIGHTER:
                    enemy_color = FIGHTER_COLOR
                elif subtype_val == ENEMY_TYPE_SAUCER:
                    enemy_color = SAUCER_COLOR
                else:
                    enemy_color = (255, 0, 0)
                rel_x = ox_f - player_x
                rel_y = oy_f - player_y
                cos_a = math.cos(player_angle); sin_a = math.sin(player_angle)
                view_x = -rel_x * sin_a + rel_y * cos_a
                view_y = rel_x * cos_a + rel_y * sin_a
                radar_dx = view_x
                radar_dy = -view_y
                rx = int(radar_center_x + radar_dx * scale)
                ry = int(radar_center_y + radar_dy * scale)
                if (rx - radar_center_x) ** 2 + (ry - radar_center_y) ** 2 <= (radar_radius - 3) ** 2:
                    pygame.draw.circle(screen, enemy_color, (rx, ry), 3)
                    enemy_angle = float(state.obstacles.angle[i])
                    rel_angle = enemy_angle - player_angle
                    indicator_length = 4
                    end_x = rx + int(indicator_length * math.cos(rel_angle))
                    end_y = ry + int(indicator_length * math.sin(rel_angle))
                    pygame.draw.line(screen, enemy_color, (rx, ry), (end_x, end_y), 1)

        bullets = state.bullets
        for i in range(len(bullets.x)):
            if bullets.active[i]:
                bx = float(bullets.x[i]); by = float(bullets.y[i])
                rel_x = bx - player_x; rel_y = by - player_y
                cos_a = math.cos(player_angle); sin_a = math.sin(player_angle)
                view_x = -rel_x * sin_a + rel_y * cos_a
                view_y = rel_x * cos_a + rel_y * sin_a
                radar_dx = view_x; radar_dy = -view_y
                rx = int(radar_center_x + radar_dx * scale)
                ry = int(radar_center_y + radar_dy * scale)
                if (rx - radar_center_x) ** 2 + (ry - radar_center_y) ** 2 <= (radar_radius - 3) ** 2:
                    color = (255, 255, 255) if bullets.owner[i] == 0 else (255, 100, 100)
                    pygame.draw.circle(screen, color, (rx, ry), 1)

        # Bottom black bar for HUD (score & lives) - use shared HUD bar height
        bottom_bar_height = self.hud_bar_height
        pygame.draw.rect(screen, (0, 0, 0), (0, HEIGHT - bottom_bar_height, WIDTH, bottom_bar_height))

        # Render score (centered) and lives (icons left of center)
        try:
            score_val = int(float(state.player_score))
        except Exception:
            score_val = 0
        try:
            lives_val = int(float(state.player_lives))
        except Exception:
            lives_val = 0

        # Draw score centered (accent color)
        if self.hud_font is not None:
            score_surf = self.hud_font.render(f"{score_val:03d}", True, HUD_ACCENT_COLOR)
            score_rect = score_surf.get_rect(center=(WIDTH // 2, HEIGHT - bottom_bar_height // 2 - 6))
            screen.blit(score_surf, score_rect)

        # Draw life icons centered below the score (max 5)
        max_lives_to_show = 5
        lives_to_draw = min(max_lives_to_show, max(0, lives_val))
        life_spacing = 16
        # Compute start x so icons are centered horizontally
        total_width = (lives_to_draw - 1) * life_spacing + 10 if lives_to_draw > 0 else 0
        life_start_x = (WIDTH // 2) - (total_width // 2)
        # Position icons below the score slightly
        life_y = HEIGHT - bottom_bar_height // 2 + 6

        # If life sprite loaded, scale it down to the size of the previous rectangles and tint
        if getattr(self, 'life_sprite_loaded', False) and self.life_sprite is not None:
            try:
                life_surf_orig = self.numpy_to_pygame_surface(self.life_sprite)
                # Target size similar to previous rectangles (10x12)
                target_w, target_h = (12, 12)
                life_surf = pygame.transform.scale(life_surf_orig, (target_w, target_h)).convert_alpha()

                # Tint life surf to HUD_ACCENT_COLOR
                tint = pygame.Surface((target_w, target_h), pygame.SRCALPHA)
                tint.fill(HUD_ACCENT_COLOR + (0,))
                life_surf.blit(tint, (0, 0), special_flags=pygame.BLEND_RGB_MULT)

                for i in range(lives_to_draw):
                    lx = int(life_start_x + i * life_spacing)
                    ly = int(life_y - target_h//2)
                    screen.blit(life_surf, (lx, ly))
            except Exception:
                # Fallback to rectangles on any error
                for i in range(lives_to_draw):
                    lx = int(life_start_x + i * life_spacing)
                    ly = int(life_y - 6)
                    pygame.draw.rect(screen, HUD_ACCENT_COLOR, (lx, ly, 10, 12), 0)
        else:
            for i in range(lives_to_draw):
                lx = int(life_start_x + i * life_spacing)
                ly = int(life_y - 6)
                pygame.draw.rect(screen, HUD_ACCENT_COLOR, (lx, ly, 10, 12), 0)

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

        # --- Moving mountains (parallax effect) - slower movement ---
        mountain_layers = [
            # (color, amplitude, freq, y_offset, speed_factor)
            ((110, 110, 110), 18, 0.025, 30, 0.1),   # Reduced speed from 0.25
            ((80, 80, 80), 28, 0.035, 18, 0.2),      # Reduced speed from 0.5
            ((50, 50, 50), 38, 0.045, 0, 0.4),       # Reduced speed from 1.0
        ]
        for color, amp, freq, y_off, speed in mountain_layers:
            points = []
            # Slower phase calculation to match reduced movement speed
            phase = (player.x * speed * 0.01 + player.angle * 8.0) % (2 * math.pi)
            for x in range(0, WIDTH + 1, 2):
                y = int(
                    sky_height
                    - (math.sin(freq * (x + phase * 120)) * amp + y_off)
                )
                points.append((x, y))
            points.append((WIDTH, sky_height))
            points.append((0, sky_height))
            pygame.draw.polygon(screen, color, points)

        # --- Ground rendering with restored movement sensitivity for forward/back ---
        ground_bands = 16
        ground_colors = [
            (60, 120, 60), (70, 130, 60), (80, 140, 60), (90, 150, 60),
            (100, 160, 60), (110, 170, 60), (120, 180, 60), (130, 190, 60),
            (140, 200, 60), (150, 210, 60), (160, 220, 60), (170, 230, 60),
            (180, 240, 60), (170, 230, 60), (160, 220, 60), (150, 210, 60)
        ]
        # Compute player's forward/backward movement since last frame and move ground opposite
        try:
            player_x = float(player.x)
            player_y = float(player.y)
            prev_x = float(state.prev_player_x)
            prev_y = float(state.prev_player_y)
            player_angle = float(player.angle)
            # forward movement scalar (positive = moved forward in facing direction)
            forward_delta = (math.cos(player_angle) * (player_x - prev_x)
                             + math.sin(player_angle) * (player_y - prev_y))
        except Exception:
            forward_delta = 0.0

        # Use forward_delta to shift ground in opposite direction to create motion illusion.
        # Scale chosen experimentally to produce visible but not excessive parallax.
        ground_offset = int((player.y * 0.15 - forward_delta * 20.0 + player_angle * 2.0)) % ground_bands
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

        # --- Draw player crosshair (vertical bar) ---
        # Centered horizontally in the viewport, a few pixels above the horizon
        try:
            center_x = WIDTH // 2
            # place the bar a few pixels higher above the horizon
            cross_bottom = HORIZON_Y - 6
            # Reduce thickness to half of previous default (previously 3 -> now floor(3/2)=1)
            original_thickness = 3
            cross_thickness = max(1, original_thickness // 2)
            # height is twice the width (thickness), ensures proportional shape
            cross_height = cross_thickness * 4
            cross_top = cross_bottom - cross_height

            # Default fill color: black; if any visible enemy is horizontally aligned, fill white
            fill_color = (0, 0, 0)
            outline_color = WIREFRAME_COLOR
            tolerance_pixels = 2  # how close to center counts as 'aligned'

            # Check obstacles for horizontal alignment with center of screen
            obs = state.obstacles
            for i in range(len(obs.x)):
                try:
                    alive_val = int(obs.alive[i])
                except Exception:
                    try:
                        alive_val = int(float(obs.alive[i]))
                    except Exception:
                        alive_val = 0
                if alive_val == 0:
                    continue

                sx, sy, dist, vis = self.world_to_screen_3d(
                    obs.x[i], obs.y[i],
                    state.player_tank.x, state.player_tank.y, state.player_tank.angle
                )

                # Convert visibility to bool safely
                try:
                    vis_bool = bool(vis)
                except Exception:
                    try:
                        vis_bool = bool(int(vis))
                    except Exception:
                        vis_bool = False

                if not vis_bool:
                    continue

                try:
                    sx_int = int(sx)
                except Exception:
                    try:
                        sx_int = int(float(sx))
                    except Exception:
                        continue

                if abs(sx_int - center_x) <= tolerance_pixels:
                    fill_color = (255, 255, 255)
                    outline_color = (255, 255, 255)
                    break

            # Draw the vertical bar with outline so it's visible on dark backgrounds
            outer_rect = (center_x - (cross_thickness // 2) - 1, cross_top - 1, cross_thickness + 2, cross_height + 2)
            inner_rect = (center_x - cross_thickness // 2, cross_top, cross_thickness, cross_height)
            pygame.draw.rect(screen, outline_color, outer_rect, 1)
            pygame.draw.rect(screen, fill_color, inner_rect)
        except Exception:
            # Fail silently to avoid breaking rendering
            pass

        # --- Draw game objects ---
        # Draw obstacles (now enemy tanks and other objects)
        obstacles = state.obstacles
        for i in range(len(obstacles.x)):
            obstacle_x = obstacles.x[i]
            obstacle_y = obstacles.y[i]
            obstacle_type = obstacles.obstacle_type[i]
            obstacle_angle = obstacles.angle[i]
            obstacle_alive = obstacles.alive[i]
            
            # Only draw if alive
            if not obstacle_alive:
                continue
            
            # Transform obstacle position to screen coordinates
            screen_x, screen_y, distance, visible = self.world_to_screen_3d(
                obstacle_x, obstacle_y,
                state.player_tank.x,
                state.player_tank.y,
                state.player_tank.angle
            )
            
            if visible and 0 <= screen_x < WIDTH and 0 <= screen_y < HEIGHT:
                if obstacle_type == 0:  # Enemy tank
                    # Choose color by enemy subtype when available
                    try:
                        subtype_val = int(state.obstacles.enemy_subtype[i])
                    except Exception:
                        subtype_val = -1
                    if subtype_val == ENEMY_TYPE_SUPERTANK:
                        enemy_color = SUPERTANK_COLOR
                    elif subtype_val == ENEMY_TYPE_TANK:
                        enemy_color = TANK_COLOR
                    elif subtype_val == ENEMY_TYPE_FIGHTER:
                        enemy_color = FIGHTER_COLOR
                    elif subtype_val == ENEMY_TYPE_SAUCER:
                        enemy_color = SAUCER_COLOR
                    else:
                        enemy_color = WIREFRAME_COLOR

                    self.draw_enemy_tank(screen, screen_x, screen_y, distance, enemy_color,
                                       obstacle_angle, state.player_tank.angle)
                elif obstacle_type == 1:  # Cube obstacle
                    self.draw_wireframe_cube(screen, screen_x, screen_y, distance, WIREFRAME_COLOR)
                else:  # Pyramid obstacle
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
        
        # Print to console when the player gets shot by an enemy bullet.
        try:
            if int(info.player_shot) == 1:
                print("Player got shot!")
        except Exception:
            # If conversion fails (shouldn't under normal execution), ignore.
            pass

        # Create a surface for the game area
        game_surface = pygame.Surface((WIDTH, HEIGHT))
        renderer.render(curr_state, game_surface)
        
        # Scale up the game surface
        scaled_surface = pygame.transform.scale(game_surface, (WIDTH * 3, HEIGHT * 3))
        screen.blit(scaled_surface, (0, 0))
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()