from functools import partial
import chex
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import lax
from dataclasses import dataclass
from typing import Tuple, NamedTuple
import random
import os
from sys import maxsize
import math
import numpy as np

from jaxatari.environment import JaxEnvironment, EnvState
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as aj
from jaxatari.spaces import Space
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
ENEMY_MOVE_SPEED = 0.35
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
ENEMY_SPAWN_DISTANCE_MIN = 80.0  # Minimum spawn distance from player (closer)
ENEMY_SPAWN_DISTANCE_MAX = 200.0  # Maximum spawn distance from player (closer)
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

# Integer limits (avoid calling np.iinfo on traced values)
INT32_MAX = 2**31 - 1

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

# Per-enemy-subtype colors (mapped to requested appearance)
# Tank: blue turret (standard enemy, slow)
TANK_COLOR = (20, 110, 220)      # blue
# Supertank: yellow turret (faster and more aggressive)
SUPERTANK_COLOR = (220, 200, 30)  # yellow
# Fighter / Missile: red (zig-zagging aerial enemy)
FIGHTER_COLOR = (220, 40, 40)     # red
# Flying Saucer: white (UFO shape). Note: saucer will be hidden from radar below.
SAUCER_COLOR = (240, 240, 240)    # white
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
    step_counter: chex.Array
    spawn_timer: chex.Array
    player_score: chex.Array
    player_lives: chex.Array

class BattleZoneInfo(NamedTuple):
    time: jnp.ndarray
    all_rewards: chex.Array
    player_shot: chex.Array  # New flag: 1 if player was shot this step, 0 otherwise

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
    
    # Compute score delta: account for per-subtype point values when player bullets kill obstacles
    # collisions_by_player: True where a player bullet hits an obstacle
    player_bullet_mask = (bullets.owner[:, None] == 0)
    collisions_by_player = jnp.logical_and(collisions, player_bullet_mask)
    obstacles_killed_by_player = jnp.any(collisions_by_player, axis=0)  # per-obstacle

    # Points per subtype: TANK=1000, SUPERTANK=3000, FIGHTER=2000, SAUCER=5000
    points_map = jnp.array([1000, 3000, 2000, 5000], dtype=jnp.int32)
    enemy_sub = getattr(obstacles, 'enemy_subtype', jnp.full_like(obstacles.x, -1))
    # Clip negative/invalid subtype to 0 but mask with killed flag so invalids contribute 0
    subtype_idx = jnp.clip(enemy_sub, 0, points_map.shape[0]-1).astype(jnp.int32)
    # Multiply per-obstacle killed mask by corresponding points and sum
    points_per_obstacle = points_map[subtype_idx] * obstacles_killed_by_player.astype(jnp.int32)
    score_delta = jnp.sum(points_per_obstacle).astype(jnp.int32)
    
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
    new_lifetime = jnp.maximum(new_lifetime, 0.0)
    
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
        # Extract per-enemy values
        enemy_x = obstacles.x[i]
        enemy_y = obstacles.y[i]
        enemy_angle = obstacles.angle[i]
        enemy_alive = obstacles.alive[i]
        enemy_cooldown = obstacles.fire_cooldown[i]
        enemy_state = obstacles.ai_state[i]
        enemy_target_angle = obstacles.target_angle[i]
        enemy_state_timer = obstacles.state_timer[i]
        is_enemy_tank = obstacles.obstacle_type[i] == 0

        # Only process alive enemies that are obstacle_type==0 (enemies)
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

        # Default outputs mirror inputs
        new_x = enemy_x
        new_y = enemy_y
        new_angle = enemy_angle
        new_state = enemy_state
        new_target_angle = enemy_target_angle
        new_cooldown = jnp.maximum(0, enemy_cooldown - 1)
        should_fire = jnp.array(False)

        # Detection and movement bands
        in_detection = jnp.logical_and(should_process, distance_to_player < ENEMY_DETECTION_RANGE)
        too_close = distance_to_player < ENEMY_MIN_DISTANCE
        too_far = distance_to_player > ENEMY_MAX_DISTANCE
        in_engage_band = jnp.logical_and(distance_to_player <= ENEMY_MAX_DISTANCE, distance_to_player >= ENEMY_MIN_DISTANCE)

        move_forward = jnp.logical_and(in_detection, jnp.logical_or(too_far, jnp.logical_and(in_engage_band, distance_to_player > ENEMY_OPTIMAL_DISTANCE)))
        move_backward = jnp.logical_and(in_detection, too_close)

        # Subtype handling
        subtype = obstacles.enemy_subtype[i]
        is_tank_subtype = jnp.logical_or(subtype == ENEMY_TYPE_TANK, subtype == ENEMY_TYPE_SUPERTANK)

        # Non-tank types move directly along the player vector
        move_forward_non_tank = jnp.logical_and(move_forward, jnp.logical_not(is_tank_subtype))
        move_backward_non_tank = jnp.logical_and(move_backward, jnp.logical_not(is_tank_subtype))
        new_x = jnp.where(move_forward_non_tank, enemy_x + dir_x * ENEMY_MOVE_SPEED, new_x)
        new_y = jnp.where(move_forward_non_tank, enemy_y + dir_y * ENEMY_MOVE_SPEED, new_y)
        new_x = jnp.where(move_backward_non_tank, enemy_x - dir_x * ENEMY_MOVE_SPEED, new_x)
        new_y = jnp.where(move_backward_non_tank, enemy_y - dir_y * ENEMY_MOVE_SPEED, new_y)

        # Fighter behaviour: zigzag while advancing and a stronger veer when close
        is_fighter = subtype == ENEMY_TYPE_FIGHTER
        perp_x = -dir_y
        perp_y = dir_x
        zig_freq = 0.35
        zig_phase = i * 0.71
        zig = jnp.sin(step_counter * zig_freq + zig_phase)
        fighter_base_speed = ENEMY_MOVE_SPEED * ENEMY_SPEED_MULTIPLIERS[ENEMY_TYPE_FIGHTER]
        zig_amp = fighter_base_speed * 1.4
        lateral_dx = perp_x * zig * zig_amp
        lateral_dy = perp_y * zig * zig_amp

        # Fighters should relentlessly advance toward the player until point-blank
        # (don't use the generic move_forward/optimal-distance gating used by tanks)
        fighter_advancing = jnp.logical_and(is_fighter, distance_to_player > FIGHTER_POINT_BLANK_DIST)
        fighter_next_x = enemy_x + dir_x * fighter_base_speed + lateral_dx
        fighter_next_y = enemy_y + dir_y * fighter_base_speed + lateral_dy
        new_x = jnp.where(fighter_advancing, fighter_next_x, new_x)
        new_y = jnp.where(fighter_advancing, fighter_next_y, new_y)

        fighter_close_radius = ENEMY_FIRING_RANGE[subtype.astype(jnp.int32)] * 1.2
        fighter_pre_veer = jnp.logical_and(is_fighter, distance_to_player <= fighter_close_radius)
        veer_boost = 1.8
        fighter_veer_x = enemy_x + dir_x * fighter_base_speed + perp_x * (zig * zig_amp * veer_boost)
        fighter_veer_y = enemy_y + dir_y * fighter_base_speed + perp_y * (zig * zig_amp * veer_boost)
        new_x = jnp.where(fighter_pre_veer, fighter_veer_x, new_x)
        new_y = jnp.where(fighter_pre_veer, fighter_veer_y, new_y)

        # Saucer behaviour: independent drifting, no pursuit
        is_saucer = subtype == ENEMY_TYPE_SAUCER
        saucer_speed = ENEMY_MOVE_SPEED * ENEMY_SPEED_MULTIPLIERS[ENEMY_TYPE_SAUCER]
        saucer_dx = jnp.cos(step_counter * 0.12 + i * 0.33) * saucer_speed
        saucer_dy = jnp.sin(step_counter * 0.15 + i * 0.29) * saucer_speed
        saucer_next_x = enemy_x + saucer_dx
        saucer_next_y = enemy_y + saucer_dy
        new_x = jnp.where(jnp.logical_and(is_saucer, should_process), saucer_next_x, new_x)
        new_y = jnp.where(jnp.logical_and(is_saucer, should_process), saucer_next_y, new_y)

        # --- Tank-like steering and movement ---
        subtype_idx = subtype.astype(jnp.int32)
        turn_rate = jnp.where(subtype == ENEMY_TYPE_SUPERTANK,
                              ENEMY_TURN_RATES[ENEMY_TYPE_SUPERTANK],
                              ENEMY_TURN_RATES[ENEMY_TYPE_TANK])

        speed_multiplier = jnp.where(subtype == ENEMY_TYPE_SUPERTANK,
                                     ENEMY_SPEED_MULTIPLIERS[ENEMY_TYPE_SUPERTANK],
                                     ENEMY_SPEED_MULTIPLIERS[ENEMY_TYPE_TANK])

        desired_heading = angle_to_player
        is_hunt = enemy_state == ENEMY_STATE_HUNT
        is_blue_tank = subtype == ENEMY_TYPE_TANK
        lateral_rad = ENEMY_HUNT_LATERAL_ANGLE_RAD[subtype_idx]
        sign = jnp.where(jnp.sin(step_counter * 0.13 + i * 0.41) > 0, 1.0, -1.0)
        lateral_offset = lateral_rad * sign
        desired_heading = jnp.where(jnp.logical_and(is_hunt, is_blue_tank), desired_heading + lateral_offset, desired_heading)

        raw_delta = desired_heading - new_angle
        delta_angle = jnp.arctan2(jnp.sin(raw_delta), jnp.cos(raw_delta))
        max_turn = turn_rate
        turn_amount = jnp.clip(delta_angle, -max_turn, max_turn)
        apply_steer = jnp.logical_and(should_process, in_detection)
        applied_turn = jnp.where(jnp.logical_and(apply_steer, is_tank_subtype), turn_amount, jnp.array(0.0))

        steered_angle = new_angle + applied_turn
        base_move_speed = ENEMY_MOVE_SPEED * speed_multiplier
        new_angle = jnp.where(jnp.logical_and(apply_steer, is_tank_subtype), steered_angle, new_angle)

        move_angle = jnp.where(jnp.logical_and(apply_steer, is_tank_subtype), steered_angle, new_angle)
        tank_move_cond = jnp.logical_and(should_process, is_tank_subtype)

        phase = jnp.sin(step_counter * 0.08 + i * 0.3)
        move_phase = phase > 0.0
        move_multiplier = jnp.where(move_phase, 1.0, 0.0)

        stop_advance_threshold = ENEMY_FIRING_RANGE[subtype_idx]
        close_enough_to_stop = distance_to_player <= stop_advance_threshold
        effective_move_multiplier = jnp.where(close_enough_to_stop, 0.0, move_multiplier)

        tank_next_x = new_x + jnp.cos(move_angle) * base_move_speed * effective_move_multiplier
        tank_next_y = new_y + jnp.sin(move_angle) * base_move_speed * effective_move_multiplier

        agg_radius = ENEMY_FIRING_RANGE[subtype_idx] * 1.5 + 1e-6
        aggression = jnp.clip((agg_radius - distance_to_player) / agg_radius, 0.0, 1.0)

        lateral_base = 0.45
        lateral_strength = base_move_speed * (lateral_base + aggression * 1.2)
        lateral_dir = jnp.sign(player_tank.x - enemy_x)
        lateral_correction = lateral_strength * lateral_dir

        hold_scale = 0.25
        applied_lateral = jnp.where(effective_move_multiplier > 0,
                                    lateral_correction * (1.0 + aggression),
                                    lateral_correction * (hold_scale + aggression * 0.5))

        tank_next_x = tank_next_x + applied_lateral
        new_x = jnp.where(tank_move_cond, tank_next_x, new_x)
        new_y = jnp.where(tank_move_cond, tank_next_y, new_y)

        # Firing logic
        angle_diff = angle_to_player - new_angle
        angle_diff = jnp.arctan2(jnp.sin(angle_diff), jnp.cos(angle_diff))

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

        # Fighter point-blank veer/fire
        fighter_point_blank = jnp.logical_and(is_fighter, distance_to_player < FIGHTER_POINT_BLANK_DIST)
        sin_val = jnp.sin(step_counter * 0.123 + i * 0.71)
        veer_sign = jnp.where(sin_val >= 0.0, 1.0, -1.0)
        veer_angle = veer_sign * FIGHTER_VEER_ANGLE_RAD
        fighter_fire_now = jnp.logical_and(fighter_point_blank, base_can_fire)
        new_angle = jnp.where(fighter_fire_now, new_angle + veer_angle, new_angle)

        # If a fighter fires at point-blank, it instantly disappears (suicide attack).
        # Mark the enemy slot as dead so it is removed next frame.
        new_alive = jnp.where(fighter_fire_now, jnp.array(0, dtype=enemy_alive.dtype), enemy_alive)

        in_hold_phase = jnp.logical_and(tank_move_cond, jnp.logical_not(move_phase))
        should_fire = jnp.logical_or(jnp.logical_and(firing_condition, jnp.logical_not(waiting)), fighter_fire_now)
        should_fire = jnp.logical_or(should_fire, jnp.logical_and(in_hold_phase, base_can_fire))

        new_state = jnp.where(in_engage_band, ENEMY_STATE_ENGAGE, new_state)
        new_cooldown = jnp.where(should_fire, ENEMY_FIRE_COOLDOWNS[subtype_idx], new_cooldown)

        # Bullet spawn and aim
        bullet_offset = 15.0
        spawn_bx = new_x + jnp.cos(new_angle) * bullet_offset
        spawn_by = new_y + jnp.sin(new_angle) * bullet_offset
        # Guarantee fatal fighter shot: if fighter fired point-blank, place bullet at player's position
        spawn_bx = jnp.where(fighter_fire_now, player_tank.x, spawn_bx)
        spawn_by = jnp.where(fighter_fire_now, player_tank.y, spawn_by)
        aim_dx = player_tank.x - new_x
        aim_dy = player_tank.y - new_y
        aim_dist = jnp.sqrt(aim_dx * aim_dx + aim_dy * aim_dy) + 1e-6
        aim_vx = (aim_dx / aim_dist) * BULLET_SPEED
        aim_vy = (aim_dy / aim_dist) * BULLET_SPEED

        # Clamp position and normalize angle
        new_x = jnp.clip(new_x, BOUNDARY_MIN, BOUNDARY_MAX)
        new_y = jnp.clip(new_y, BOUNDARY_MIN, BOUNDARY_MAX)
        new_angle = jnp.arctan2(jnp.sin(new_angle), jnp.cos(new_angle))

        return (new_x, new_y, new_angle, new_alive, new_cooldown, new_state,
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
        # Weighted choice: Tank (45%), Supertank (45%), Fighter (10%)
        v = r1
        return jnp.where(v < 0.45, ENEMY_TYPE_TANK,
                         jnp.where(v < 0.90, ENEMY_TYPE_SUPERTANK, ENEMY_TYPE_FIGHTER))

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
    def image_space(self) -> Space:
        """Returns the image space for BattleZone (RGB 210x160)."""
        return spaces.Box(
            low=0,
            high=255,
            shape=(210, 160, 3),
            dtype=jnp.uint8,
        )

    def reset(self, key: jrandom.PRNGKey = None) -> Tuple[BattleZoneObservation, BattleZoneState]:
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
        enemy_positions_x = jnp.array([
            120.0, -120.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ])
        enemy_positions_y = jnp.array([
            0.0, 0.0, 120.0, -120.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ])
        enemy_types = jnp.zeros(16)  # All enemy obstacles default to type 0 slot (enemy when alive)
        # Start with a single primary hostile (tank) alive; other slots are empty (-1 subtype)
        enemy_subtypes = jnp.array([0] + [-1] * 15)
        # Initial angles for enemy tanks (primary at slot 0)
        enemy_angles = jnp.array([jnp.pi] + [0.0] * 15)
        # Only first primary enemy alive initially (sequential hostiles); saucers may spawn later
        enemy_alive = jnp.array([1] + [0] * 15, dtype=jnp.int32)
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
    def step(self, state: BattleZoneState, action) -> Tuple[BattleZoneObservation, BattleZoneState, float, bool, BattleZoneInfo]:
        """Perform a step in the BattleZone environment. 
        This updates the game state based on the action taken.
        It is the main update LOOP for the game.
        """
                # --- Normalize external action from play.py ---
        # Mappe den externen Action-Index robust auf die 18 gültigen BattleZone-Aktionen.
        # Alles außerhalb 0–17 wird auf NOOP gesetzt.
        action = jnp.asarray(action, dtype=jnp.int32)

        # Aktions-Lookup Tabelle (Index = Action-ID aus play.py)
        action_table = jnp.array([
            NOOP,
            FIRE,
            UP,
            RIGHT,
            LEFT,
            DOWN,
            UPRIGHT,
            UPLEFT,
            DOWNRIGHT,
            DOWNLEFT,
            UPFIRE,
            RIGHTFIRE,
            LEFTFIRE,
            DOWNFIRE,
            UPRIGHTFIRE,
            UPLEFTFIRE,
            DOWNRIGHTFIRE,
            DOWNLEFTFIRE,
        ], dtype=jnp.int32)

        # Index clampen, damit out-of-range Werte nicht crashen
        idx = jnp.clip(action, 0, action_table.shape[0] - 1)
        norm_action = action_table[idx]

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
            enemy_positions_x = jnp.array([120.0, -120.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            enemy_positions_y = jnp.array([0.0, 0.0, 120.0, -120.0, 0.0, 0.0, 0.0, 0.0,
                                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            enemy_types = jnp.zeros(16)
            # Primary reset: single primary hostile alive; others empty
            enemy_angles = jnp.array([jnp.pi] + [0.0] * 15)
            enemy_alive = jnp.array([1] + [0] * 15, dtype=jnp.int32)
            enemy_fire_cooldown = jnp.zeros(16)
            enemy_ai_state = jnp.zeros(16, dtype=jnp.int32)
            enemy_target_angle = enemy_angles.copy()
            enemy_state_timer = jnp.zeros(16)

            obstacles_rst = Obstacle(
                x=enemy_positions_x,
                y=enemy_positions_y,
                obstacle_type=enemy_types,
                # Single primary hostile at slot 0, other slots empty (-1)
                enemy_subtype=jnp.array([0] + [-1] * 15),
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
        info = self._get_info_full(final_state, all_rewards, player_was_shot)

        return observation, final_state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: BattleZoneState) -> BattleZoneObservation:
        return BattleZoneObservation(
            player_tank=state.player_tank,
            bullets=state.bullets,
            obstacles=state.obstacles,
            step_counter=state.step_counter,
            spawn_timer=state.spawn_timer,
            player_score=state.player_score,
            player_lives=state.player_lives,
        )

    def action_space(self) -> Space:
        """Discrete action space (callable) to match JaxEnvironment API and tests."""
        return spaces.Discrete(len(self.action_set))

    def observation_space(self) -> Space:
        # Use module-level constants directly to avoid mismatches
        player_space = spaces.Dict({
            "x": spaces.Box(low=BOUNDARY_MIN, high=BOUNDARY_MAX, shape=(), dtype=np.float32),
            "y": spaces.Box(low=BOUNDARY_MIN, high=BOUNDARY_MAX, shape=(), dtype=np.float32),
            "angle": spaces.Box(low=-math.pi, high=math.pi, shape=(), dtype=np.float32),
            "alive": spaces.Box(low=0, high=1, shape=(), dtype=np.float32),
        })

        bullets_space = spaces.Dict({
            "x": spaces.Box(low=BOUNDARY_MIN, high=BOUNDARY_MAX, shape=(MAX_BULLETS,), dtype=np.float32),
            "y": spaces.Box(low=BOUNDARY_MIN, high=BOUNDARY_MAX, shape=(MAX_BULLETS,), dtype=np.float32),
            "z": spaces.Box(low=0.0, high=1000.0, shape=(MAX_BULLETS,), dtype=np.float32),
            "vel_x": spaces.Box(low=-1000.0, high=1000.0, shape=(MAX_BULLETS,), dtype=np.float32),
            "vel_y": spaces.Box(low=-1000.0, high=1000.0, shape=(MAX_BULLETS,), dtype=np.float32),
            "active": spaces.Box(low=0, high=1, shape=(MAX_BULLETS,), dtype=np.float32),
            # keep >=0 since we’ll clamp lifetime to 0 below
            "lifetime": spaces.Box(low=0, high=BULLET_LIFETIME, shape=(MAX_BULLETS,), dtype=np.float32),
            "owner": spaces.Box(low=0, high=MAX_BULLETS, shape=(MAX_BULLETS,), dtype=np.float32),
        })

        obstacles_space = spaces.Dict({
            "x": spaces.Box(low=BOUNDARY_MIN, high=BOUNDARY_MAX, shape=(MAX_OBSTACLES,), dtype=np.float32),
            "y": spaces.Box(low=BOUNDARY_MIN, high=BOUNDARY_MAX, shape=(MAX_OBSTACLES,), dtype=np.float32),
            "obstacle_type": spaces.Box(low=0, high=10, shape=(MAX_OBSTACLES,), dtype=np.float32),
            "enemy_subtype": spaces.Box(low=-1, high=10, shape=(MAX_OBSTACLES,), dtype=np.float32),
            "angle": spaces.Box(low=-math.pi, high=math.pi, shape=(MAX_OBSTACLES,), dtype=np.float32),
            "alive": spaces.Box(low=0, high=1, shape=(MAX_OBSTACLES,), dtype=np.float32),
            "fire_cooldown": spaces.Box(low=0, high=ENEMY_FIRE_COOLDOWN * 4, shape=(MAX_OBSTACLES,), dtype=np.float32),
            "ai_state": spaces.Box(low=0, high=10, shape=(MAX_OBSTACLES,), dtype=np.float32),
            "target_angle": spaces.Box(low=-math.pi, high=math.pi, shape=(MAX_OBSTACLES,), dtype=np.float32),
            "state_timer": spaces.Box(low=0, high=ENEMY_SPAWN_COOLDOWN * 4, shape=(MAX_OBSTACLES,), dtype=np.float32),
        })

        return spaces.Dict({
            "player_tank": player_space,
            "bullets": bullets_space,
            "obstacles": obstacles_space,
            "step_counter": spaces.Box(low=0, high=INT32_MAX, shape=(), dtype=np.float32),
            "spawn_timer": spaces.Box(low=0, high=ENEMY_SPAWN_COOLDOWN * 4, shape=(), dtype=np.float32),
            "player_score": spaces.Box(low=0, high=INT32_MAX, shape=(), dtype=np.float32),
            "player_lives": spaces.Box(low=0, high=99, shape=(), dtype=np.float32),
        })

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs) -> jnp.ndarray:
        """Convert a single-frame observation (pytree of jax arrays) to a flat jax array.

        This implementation is JAX-friendly (no Python float/int conversions) so it can be
        jitted and vmapped by wrappers.
        """
        # Player
        pt = obs.player_tank
        player_flat = jnp.stack([pt.x.astype(jnp.float32), pt.y.astype(jnp.float32), pt.angle.astype(jnp.float32), pt.alive.astype(jnp.float32)])

        # Bullets
        b = obs.bullets
        bullets_flat = jnp.concatenate([
            jnp.ravel(b.x.astype(jnp.float32)),
            jnp.ravel(b.y.astype(jnp.float32)),
            jnp.ravel(b.z.astype(jnp.float32)),
            jnp.ravel(b.vel_x.astype(jnp.float32)),
            jnp.ravel(b.vel_y.astype(jnp.float32)),
            jnp.ravel(b.active.astype(jnp.float32)),
            jnp.ravel(b.lifetime.astype(jnp.float32)),
            jnp.ravel(b.owner.astype(jnp.float32)),
        ])

        # Obstacles
        o = obs.obstacles
        enemy_sub = getattr(o, 'enemy_subtype', None)
        if enemy_sub is None:
            enemy_sub = jnp.full_like(o.x, -1)

        obstacles_flat = jnp.concatenate([
            jnp.ravel(o.x.astype(jnp.float32)),
            jnp.ravel(o.y.astype(jnp.float32)),
            jnp.ravel(o.obstacle_type.astype(jnp.float32)),
            jnp.ravel(enemy_sub.astype(jnp.float32)),
            jnp.ravel(o.angle.astype(jnp.float32)),
            jnp.ravel(o.alive.astype(jnp.float32)),
            jnp.ravel(o.fire_cooldown.astype(jnp.float32)),
            jnp.ravel(o.ai_state.astype(jnp.float32)),
            jnp.ravel(o.target_angle.astype(jnp.float32)),
            jnp.ravel(o.state_timer.astype(jnp.float32)),
        ])
        scalars = jnp.array([
            obs.step_counter, obs.spawn_timer, obs.player_score, obs.player_lives
        ], dtype=jnp.float32)

        return jnp.concatenate([player_flat, bullets_flat, obstacles_flat, scalars]).astype(jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: EnvState) -> Tuple[jnp.ndarray]:
        """Render the state via the JAX renderer (uint8 HxWx3). Falls back to zeros."""
        img_shape = (HEIGHT, WIDTH, 3)
        def _fallback(_: EnvState):
            return jnp.zeros(img_shape, dtype=jnp.uint8)
        def _draw(st: EnvState):
            return self.renderer.render(st)
        has_renderer = jnp.array(self.renderer is not None)
        return jax.lax.cond(has_renderer, _draw, _fallback, operand=state)

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
    def _get_info(self, state: BattleZoneState) -> BattleZoneInfo:
        """Single-argument form required by JaxEnvironment: return defaults."""
        return BattleZoneInfo(time=state.step_counter, all_rewards=jnp.zeros(1), player_shot=jnp.array(0, dtype=jnp.int32))

    @partial(jax.jit, static_argnums=(0,))
    def _get_info_full(self, state: BattleZoneState, all_rewards: chex.Array = None, player_shot: chex.Array = None) -> BattleZoneInfo:
        """Extended internal helper used by this module when extra args are available."""
        if all_rewards is None:
            all_rewards = jnp.zeros(1)
        if player_shot is None:
            player_shot = jnp.array(0, dtype=jnp.int32)
        return BattleZoneInfo(time=state.step_counter, all_rewards=all_rewards, player_shot=player_shot)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: BattleZoneState, state: BattleZoneState) -> float:
        """
        Compatibility wrapper for the abstract environment's `_get_reward` API.
        Delegates to the existing `_get_env_reward` implementation to preserve
        the locally-named method used in this module.
        """
        return self._get_env_reward(previous_state, state)

# Annahme: folgende Konstanten/Farben sind bereits im Modul definiert:
# WIDTH, HEIGHT, HORIZON_Y, WORLD_SIZE
# WIREFRAME_COLOR, HUD_ACCENT_COLOR, BULLET_COLOR
# SUPERTANK_COLOR, TANK_COLOR, FIGHTER_COLOR, SAUCER_COLOR
# ENEMY_TYPE_SUPERTANK, ENEMY_TYPE_TANK, ENEMY_TYPE_FIGHTER, ENEMY_TYPE_SAUCER
# MAX_BULLETS
# und ein BattleZoneState mit Feldern: player_tank (x,y,angle), prev_player_x/y,
# obstacles (x,y,angle,alive,obstacle_type,enemy_subtype), bullets (x,y,active,owner),
# player_score, player_lives, step_counter
# Außerdem: eine Basisklasse JAXGameRenderer ist optional; falls nicht vorhanden, entferne die Klammer.


class BattleZoneRenderer(JAXGameRenderer):
    """
    JAX-only renderer that mirrors the Pygame version's look-and-feel:
    - identical world_to_screen_3d projection math
    - dynamic sky + mountains, horizon, ground bands
    - enemy drawing (frontal/profile), saucer, fighter
    - bullets as short lines (player white, enemy reddish)
    - centered radar in a top bar with sweep + ticks; saucer hidden on radar
    - bottom HUD bar with centered score placeholder and life icons as rectangles
    - player tank wireframe at bottom center (sprite-less fallback)
    Notes:
      * No pygame, no fonts, no image loading. Fully jit-friendly.
    """

    def __init__(self):
        self.view_distance = 400.0
        self.fov = 60.0
        self.hud_bar_height = 28
        
        # Sprite laden (Masken- oder RGBA-Variante wird unterstützt)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        sprite_dir = os.path.join(base_dir, "sprites", "battlezone")
        life_sprite_np = np.load(os.path.join(sprite_dir, "tank_life.npy"))
        # Downscale overly large life sprite once (outside JIT) to avoid huge constant folding
        if life_sprite_np.ndim >= 2:
            sh, sw = life_sprite_np.shape[:2]
            max_h, max_w = 6, 9  # HUD icon target size
            sy = max(1, int(math.ceil(sh / max_h)))
            sx = max(1, int(math.ceil(sw / max_w)))
            life_sprite_np = life_sprite_np[::sy, ::sx, ...] if life_sprite_np.ndim == 3 else life_sprite_np[::sy, ::sx]
        self.life_sprite = jnp.asarray(life_sprite_np)

        # Für Layout/Spacing im HUD nützlich:
        if self.life_sprite.ndim == 2:
            self.life_h, self.life_w = self.life_sprite.shape
        elif self.life_sprite.ndim == 3:
            self.life_h, self.life_w = self.life_sprite.shape[:2]
        else:
            raise ValueError("tank_life.npy hat ein unerwartetes Format.")
        # --- Player tank sprite (player_tank2.npy) ---
        player_sprite_np = np.load(os.path.join(sprite_dir, "player_tank2.npy"))
        # Optional downscale to keep small footprint in HUD area
        if player_sprite_np.ndim >= 2:
            psh, psw = player_sprite_np.shape[:2]
            pmax_h, pmax_w = 120, 192  # target maximum for bottom-center tank icon
            psy = max(1, int(math.ceil(psh / pmax_h)))
            psx = max(1, int(math.ceil(psw / pmax_w)))
            player_sprite_np = player_sprite_np[::psy, ::psx, ...] if player_sprite_np.ndim == 3 else player_sprite_np[::psy, ::psx]
        self.player_sprite = jnp.asarray(player_sprite_np)
        if self.player_sprite.ndim == 2:
            self.player_h, self.player_w = self.player_sprite.shape
        elif self.player_sprite.ndim == 3:
            self.player_h, self.player_w = self.player_sprite.shape[:2]
        else:
            raise ValueError("player_tank2.npy hat ein unerwartetes Format.")

    
    def _blit_sprite(self, im, sprite, x, y, tint_rgb=None):
            """
            JIT-freundliches Blitting:
            - dynamische Indizes via lax.dynamic_slice
            - feste Slice-Größen via Padding
            - dtype-Konflikte gelöst (uint8 <-> float32)
            - NEU: Wenn tint_rgb gesetzt ist, wird unabhängig vom Spriteformat getintet.
                   Für RGBA nutzen wir den Alphakanal als Maske (>0). Für RGB/Masken
                   verwenden wir eine Luminanz-/Schwellenmaske (>0).
            """
            H, W, _ = im.shape
    
            # Sprite-Formate erkennen
            if sprite.ndim == 2:          # Maske
                sh, sw = sprite.shape
                is_mask = True
                is_rgba = False
                C = 1
            elif sprite.ndim == 3 and sprite.shape[-1] in (3, 4):
                sh, sw, C = sprite.shape
                is_mask = False
                is_rgba = (C == 4)
            else:
                raise ValueError("Sprite muss HxW (Maske) oder HxWx3/4 (RGB/RGBA) sein.")
    
            sh_c = int(sh)
            sw_c = int(sw)
    
            # Padding vorbereiten
            pad_y = sh_c
            pad_x = sw_c
            im_padded = jnp.pad(im, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode="constant")
    
            # Startkoordinaten (dynamisch)
            y_start = jnp.int32(y + pad_y)
            x_start = jnp.int32(x + pad_x)
    
            # Ziel-Ausschnitt aus dem Bild (immer konstante Größe sh_c x sw_c)
            dst = lax.dynamic_slice(im_padded, (y_start, x_start, 0), (sh_c, sw_c, 3))
    
            # Wenn Tint angefordert ist -> immer TINT-MODUS, egal ob Masken- oder RGBA/RGB-Sprite
            if tint_rgb is not None:
                if is_mask:
                    spr_slice = lax.dynamic_slice(sprite, (0, 0), (sh_c, sw_c))
                    mask = spr_slice > 0
                else:
                    spr_slice = lax.dynamic_slice(sprite, (0, 0, 0), (sh_c, sw_c, C))
                    if is_rgba:
                        alpha = spr_slice[..., 3:4].astype(jnp.float32) / 255.0
                        # Alpha>0 als Maske
                        mask = (alpha > 0.0)[..., 0]
                    else:
                        # RGB -> Luminanz als Maske (>0)
                        rgb = spr_slice[..., :3].astype(jnp.float32)
                        lum = (0.299*rgb[...,0] + 0.587*rgb[...,1] + 0.114*rgb[...,2])
                        mask = lum > 0.0
    
                color = jnp.asarray(tint_rgb, dtype=im.dtype)
                color_patch = jnp.broadcast_to(color, (sh_c, sw_c, 3))
                out = jnp.where(mask[..., None], color_patch, dst).astype(im.dtype)
            else:
                # Standardpfad: Sprite direkt blenden (RGB/RGBA) oder Maske einfärben (HUD-Farbe)
                if is_mask:
                    spr_slice = lax.dynamic_slice(sprite, (0, 0), (sh_c, sw_c))
                    mask = spr_slice > 0
                    color = jnp.asarray(HUD_ACCENT_COLOR, dtype=im.dtype)
                    color_patch = jnp.broadcast_to(color, (sh_c, sw_c, 3))
                    out = jnp.where(mask[..., None], color_patch, dst).astype(im.dtype)
                else:
                    spr_slice = lax.dynamic_slice(sprite, (0, 0, 0), (sh_c, sw_c, C))
                    if is_rgba:
                        rgb = spr_slice[..., :3].astype(jnp.float32)
                        alpha = spr_slice[..., 3:4].astype(jnp.float32) / 255.0
                    else:
                        rgb = spr_slice.astype(jnp.float32)
                        alpha = jnp.ones((sh_c, sw_c, 1), dtype=jnp.float32)
    
                    dst_f = dst.astype(jnp.float32)
                    out_f = rgb * alpha + dst_f * (1.0 - alpha)
    
                    # zurückcasten auf Bild-dtype
                    if im.dtype == jnp.uint8:
                        out = jnp.clip(out_f, 0.0, 255.0).astype(jnp.uint8)
                    else:
                        out = out_f.astype(im.dtype)
    
            # zurückschreiben ins gepaddete Bild
            im_padded = lax.dynamic_update_slice(im_padded, out, (y_start, x_start, 0))
    
            # wieder zurückcroppen
            im_final = lax.dynamic_slice(im_padded, (pad_y, pad_x, 0), (H, W, 3))
            return im_final
        
        # --- tiny HUD font (3x5) for score ---
    _DIGITS_3x5 = jnp.array([
        # 0..9, each 5 rows à 3 cols (1=on)
        # 0
        [[1,1,1],
         [1,0,1],
         [1,0,1],
         [1,0,1],
         [1,1,1]],
        # 1
        [[0,1,0],
         [1,1,0],
         [0,1,0],
         [0,1,0],
         [1,1,1]],
        # 2
        [[1,1,1],
         [0,0,1],
         [1,1,1],
         [1,0,0],
         [1,1,1]],
        # 3
        [[1,1,1],
         [0,0,1],
         [0,1,1],
         [0,0,1],
         [1,1,1]],
        # 4
        [[1,0,1],
         [1,0,1],
         [1,1,1],
         [0,0,1],
         [0,0,1]],
        # 5
        [[1,1,1],
         [1,0,0],
         [1,1,1],
         [0,0,1],
         [1,1,1]],
        # 6
        [[1,1,1],
         [1,0,0],
         [1,1,1],
         [1,0,1],
         [1,1,1]],
        # 7
        [[1,1,1],
         [0,0,1],
         [0,1,0],
         [0,1,0],
         [0,1,0]],
        # 8
        [[1,1,1],
         [1,0,1],
         [1,1,1],
         [1,0,1],
         [1,1,1]],
        # 9
        [[1,1,1],
         [1,0,1],
         [1,1,1],
         [0,0,1],
         [1,1,1]],
    ], dtype=jnp.uint8)

    def _draw_digit(self, img, x, y, d, color, scale=2):
        pat = self._DIGITS_3x5[d]
        h, w = pat.shape
        im = img
        def row_body(r, imc1):
            def col_body(c, imc2):
                on = pat[r, c] == 1
                def draw(_):
                    return self._fill_rect(imc2, x + c*scale, y + r*scale, scale, scale, color)
                return lax.cond(on, draw, lambda _: imc2, operand=None)
            return lax.fori_loop(0, w, col_body, imc1)
        return lax.fori_loop(0, h, row_body, im)

    def _draw_score_hud(self, img, score_val):
        # --- vorbereiten ---
        s = jnp.asarray(score_val, jnp.int32)
        s = jnp.maximum(s, 0)

        # bis zu 9 Stellen unterstützen (0..999,999,999); bei Bedarf MAX_DIGITS erhöhen
        POW10 = jnp.asarray([1,10,100,1000,10000,100000,1000000,10000000,100000000], jnp.int32)
        MAX_DIGITS = POW10.shape[0]

        # Ziffern vektoriell berechnen
        divs = s // POW10                     # int32[9]
        digits_right = divs % 10              # ones,tens,hundreds,...
        digits_leftpad = digits_right[::-1]   # linksbündig mit Nullen aufgefüllt

        # Anzahl tatsächlicher Stellen (0 -> 1 Stelle)
        n_raw = jnp.sum(divs > 0)             # wie floor(log10(s))+1, aber ohne float
        n = jnp.maximum(n_raw, 1)

        # --- Layout ---
        scale   = 2
        glyph_w = 3 * scale
        gap     = scale
        total_w = n * glyph_w + (n - 1) * gap

        base_x = (WIDTH // 2) - (total_w // 2)      # mittig-rechts
        base_y = HEIGHT - self.hud_bar_height + 4        # über den Leben
        col    = jnp.asarray(HUD_ACCENT_COLOR, jnp.uint8)

        # --- Zeichnen: genau n Ziffern (links -> rechts) ---
        start = MAX_DIGITS - n

        im = img
        def body(i, imc):
            dx = base_x + i * (glyph_w + gap)
            d  = digits_leftpad[start + i].astype(jnp.int32)
            return self._draw_digit(imc, dx, base_y, d, col, scale=scale)
        im = lax.fori_loop(0, n, body, im)

        return im

    # ---------------- low-level primitives (pure JAX) ----------------

    @staticmethod
    def _put_pixel(img, x, y, color):
        h, w = img.shape[0], img.shape[1]
        x = jnp.asarray(x, jnp.int32)
        y = jnp.asarray(y, jnp.int32)
        color = jnp.asarray(color, jnp.uint8)
        def body(_):
            im = img
            im = im.at[y, x, 0].set(color[0])
            im = im.at[y, x, 1].set(color[1])
            im = im.at[y, x, 2].set(color[2])
            return im
        inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        return lax.cond(inside, body, lambda _: img, operand=None)

    @staticmethod
    def _draw_line(img, x0, y0, x1, y1, color, samples=256):
        # Parametric line sampling (jit-friendly; Bresenham avoids floats but needs while loops)
        t = jnp.linspace(0.0, 1.0, samples)
        xs = jnp.round(x0 + (x1 - x0) * t).astype(jnp.int32)
        ys = jnp.round(y0 + (y1 - y0) * t).astype(jnp.int32)
        color = jnp.asarray(color, jnp.uint8)
        im = img
        im = im.at[ys.clip(0, im.shape[0]-1), xs.clip(0, im.shape[1]-1), 0].set(color[0])
        im = im.at[ys.clip(0, im.shape[0]-1), xs.clip(0, im.shape[1]-1), 1].set(color[1])
        im = im.at[ys.clip(0, im.shape[0]-1), xs.clip(0, im.shape[1]-1), 2].set(color[2])
        return im

    
    @staticmethod
    def _fill_rect(img, x, y, w_, h_, color):
        # Tracer-safe: fill rectangle using boolean masking over the full (static) image.
        color = jnp.asarray(color, jnp.uint8)
        H = img.shape[0]
        W = img.shape[1]
        x0 = jnp.clip(jnp.int32(x), 0, W)
        y0 = jnp.clip(jnp.int32(y), 0, H)
        x1 = jnp.clip(jnp.int32(x + w_), 0, W)
        y1 = jnp.clip(jnp.int32(y + h_), 0, H)

        ys = jnp.arange(H)[:, None]     # static extent
        xs = jnp.arange(W)[None, :]     # static extent
        mask = (ys >= y0) & (ys < y1) & (xs >= x0) & (xs < x1)
        # Broadcast color to image shape
        col = jnp.broadcast_to(jnp.reshape(color, (1,1,3)), img.shape)
        mask3 = jnp.stack([mask, mask, mask], axis=-1)
        return jnp.where(mask3, col, img)

    @staticmethod
    def _draw_circle_outline(img, cx, cy, r, color, samples=360):
        ang = jnp.linspace(0, 2*jnp.pi, samples, endpoint=False)
        xs = jnp.round(cx + r * jnp.cos(ang)).astype(jnp.int32)
        ys = jnp.round(cy + r * jnp.sin(ang)).astype(jnp.int32)
        color = jnp.asarray(color, jnp.uint8)
        im = img
        im = im.at[ys.clip(0, im.shape[0]-1), xs.clip(0, im.shape[1]-1), 0].set(color[0])
        im = im.at[ys.clip(0, im.shape[0]-1), xs.clip(0, im.shape[1]-1), 1].set(color[1])
        im = im.at[ys.clip(0, im.shape[0]-1), xs.clip(0, im.shape[1]-1), 2].set(color[2])
        return im

    # ---------------- math: projection identical to pygame version ----------------

    @staticmethod
    def _world_to_screen_3d(world_x, world_y, player_x, player_y, player_angle):
        rel_x = world_x - player_x
        rel_y = world_y - player_y
        cos_a = jnp.cos(player_angle)
        sin_a = jnp.sin(player_angle)
        # view_x left/right, view_y forward
        view_x = -rel_x * sin_a + rel_y * cos_a
        view_y =  rel_x * cos_a + rel_y * sin_a

        # in front?
        in_front = view_y > 1.0

        fov_scale = 80.0
        perspective_scale = 100.0
        screen_x = (WIDTH // 2) + (view_x / view_y) * fov_scale
        screen_y = HORIZON_Y + (perspective_scale / view_y)

        sx = jnp.where(in_front, jnp.int32(jnp.round(screen_x)), jnp.int32(0))
        sy = jnp.where(in_front, jnp.int32(jnp.round(screen_y)), jnp.int32(0))
        dist = jnp.where(in_front, view_y, 0.0)
        return sx, sy, dist, in_front

    # ---------------- high-level elements ----------------

    def _draw_sky(self, img, px, pang):
        sky_height = HORIZON_Y
        # blue-ish gradient like pygame line fill
        yy = jnp.arange(sky_height)
        t = yy / jnp.maximum(sky_height - 1, 1)
        r = (60 * (1 - t) + 10 * t).astype(jnp.uint8)
        g = (120 * (1 - t) + 40 * t).astype(jnp.uint8)
        b = (200 * (1 - t) + 120 * t).astype(jnp.uint8)
        im = img
        im = im.at[yy, :, 0].set(r[:, None])
        im = im.at[yy, :, 1].set(g[:, None])
        im = im.at[yy, :, 2].set(b[:, None])
        # mountains (3 layers)
        layers = (
            ((110,110,110), 18.0, 0.025, 30.0, 0.1),
            ((80,80,80),   28.0, 0.035, 18.0, 0.2),
            ((50,50,50),   38.0, 0.045,  0.0, 0.4),
        )
        # We need player pose for phase; pass via closure at render call (set attributes temporarily)
        px = px
        pang = pang
        im2 = im
        for color, amp, freq, y_off, speed in layers:
            # phase uses player.x and angle
            phase = (px * speed * 0.01 + pang * 8.0)
            xs = jnp.arange(0, WIDTH+1, 2)
            ycurve = jnp.int32(jnp.round(
                sky_height - (jnp.sin(freq * (xs + phase * 120.0)) * amp + y_off)
            ))
            # polygon fill: draw vertical lines from curve to horizon
            def draw_col(i, imc):
                x = xs[i]
                y = jnp.clip(ycurve[i], 0, sky_height-1)
                return self._fill_rect(imc, x, y, 2, sky_height - y, color)
            im2 = lax.fori_loop(0, xs.shape[0], draw_col, im2)
        return im2

    def _draw_ground(self, img, state):
        ground_bands = 16
        colors = jnp.array([
            [60,120,60],[70,130,60],[80,140,60],[90,150,60],
            [100,160,60],[110,170,60],[120,180,60],[130,190,60],
            [140,200,60],[150,210,60],[160,220,60],[170,230,60],
            [180,240,60],[170,230,60],[160,220,60],[150,210,60]
        ], dtype=jnp.uint8)

        # forward delta since last frame
        px = state.player_tank.x; py = state.player_tank.y; pang = state.player_tank.angle
        prev_x = state.prev_player_x; prev_y = state.prev_player_y
        forward_delta = (jnp.cos(pang)*(px-prev_x) + jnp.sin(pang)*(py-prev_y))
        ground_offset = jnp.int32(jnp.mod((py*0.15 - forward_delta*20.0 + pang*2.0), ground_bands))
        band_h = (HEIGHT - HORIZON_Y) // ground_bands
        im = img
        for i in range(ground_bands):
            color = colors[(i + ground_offset) % ground_bands]
            y1 = HORIZON_Y + i * band_h
            y2 = HORIZON_Y + (i + 1) * band_h if i < ground_bands - 1 else HEIGHT
            im = self._fill_rect(im, 0, y1, WIDTH, y2 - y1, color)
        return im

    def _draw_crosshair(self, img, state, wire):
        center_x = WIDTH // 2
        cross_bottom = HORIZON_Y - 6
        cross_thickness = max(1, 3 // 2)
        cross_height = cross_thickness * 4
        cross_top = cross_bottom - cross_height

        # determine if any alive/visible obstacle is horizontally aligned
        px = state.player_tank.x; py = state.player_tank.y; pang = state.player_tank.angle
        ox = state.obstacles.x; oy = state.obstacles.y; alive = state.obstacles.alive.astype(jnp.bool_)

        def check(i, acc):
            cond = acc
            vis = alive[i]
            def when_vis(_):
                sx, sy, dist, vis2 = self._world_to_screen_3d(ox[i], oy[i], px, py, pang)
                aligned = (jnp.abs(sx - center_x) <= 2) & vis2
                return cond | aligned
            return lax.cond(vis, when_vis, lambda _: cond, operand=None)

        any_aligned = lax.fori_loop(0, ox.shape[0], check, jnp.bool_(False))
        fill = jnp.where(any_aligned, jnp.array([255,255,255], jnp.uint8), jnp.array([0,0,0], jnp.uint8))
        outline = jnp.where(any_aligned, jnp.array([255,255,255], jnp.uint8), jnp.asarray(wire, jnp.uint8))

        img = self._draw_line(img, center_x - (cross_thickness//2) - 1, cross_top - 1,
                                   center_x + (cross_thickness//2) + 1, cross_top - 1, outline, samples=16)
        img = self._draw_line(img, center_x - (cross_thickness//2) - 1, cross_bottom + 1,
                                   center_x + (cross_thickness//2) + 1, cross_bottom + 1, outline, samples=16)
        # vertical fill
        img = self._fill_rect(img, center_x - cross_thickness//2, cross_top, cross_thickness, cross_height, fill)
        return img

    def _draw_enemy_tank_frontal(self, img, x, y, scale, color):
        # Trapezoid body
        bw = jnp.int32(scale * 1.0)
        bh = jnp.int32(scale * 0.7)
        pts = jnp.array([
            [x - bw//2, y + bh//2],
            [x + bw//2, y + bh//2],
            [x + bw//3, y - bh//2],
            [x - bw//3, y - bh//2],
            [x - bw//2, y + bh//2]  # close
        ], dtype=jnp.int32)
        im = img
        for i in range(4):
            im = self._draw_line(im, pts[i,0], pts[i,1], pts[i+1,0], pts[i+1,1], color, samples=64)
        # turret
        tw = jnp.int32(scale//3); th = jnp.int32(scale//3)
        im = self._fill_rect(im, x - tw//2, y - th//2, tw, th, color)
        # cannon
        im = self._draw_line(im, x, y, x, y + jnp.int32(scale//2), color, samples=32)
        # tracks (vertical dashes)
        thh = bh
        for i in range(3):
            txl = x - bw//2 - 2; txr = x + bw//2 + 2
            ty = y - thh//2 + i * (thh//3)
            im = self._draw_line(im, txl, ty, txl, ty + 3, color, samples=4)
            im = self._draw_line(im, txr, ty, txr, ty + 3, color, samples=4)
        return im

    
    def _draw_enemy_tank_profile(self, img, x, y, scale, color, left=True):
        # Eliminate Python branching on tracer `left` by computing both paths and selecting via lax.cond.
        bw = jnp.int32(scale * 1.8)
        bh = jnp.int32(scale * 0.6)

        def draw_left(_):
            pts = jnp.array([
                [x - bw//2, y + bh//3],
                [x - bw//3, y + bh//2],
                [x + bw//3, y + bh//2],
                [x + bw//2, y + bh//3],
                [x + bw//2 - 2, y - bh//2],
                [x - bw//2 + 2, y - bh//2],
                [x - bw//2, y + bh//3]
            ], dtype=jnp.int32)
            turret_x = x - scale//6
            cannon_end_x = turret_x - jnp.int32(scale*1.2)
            im = img
            for i in range(pts.shape[0]-1):
                im = self._draw_line(im, pts[i,0], pts[i,1], pts[i+1,0], pts[i+1,1], color, samples=64)
            tw = jnp.int32(scale//2); th = jnp.int32(scale//4)
            im = self._fill_rect(im, turret_x - tw//2, y - th//2, tw, th, color)
            im = self._draw_line(im, turret_x, y, cannon_end_x, y, color, samples=32)
            wheel_y = y + bh//2 + 2
            im = self._draw_line(im, x - bw//2, wheel_y, x + bw//2, wheel_y, color, samples=64)
            # wheel dots
            def wheel_body(j, imc):
                cx = x - bw//2 + 2 + j*6
                return self._put_pixel(imc, cx, wheel_y, color)
            num = jnp.maximum(0, (bw-4)//6)
            im = lax.fori_loop(0, num, wheel_body, im)
            return im

        def draw_right(_):
            pts = jnp.array([
                [x + bw//2, y + bh//3],
                [x + bw//3, y + bh//2],
                [x - bw//3, y + bh//2],
                [x - bw//2, y + bh//3],
                [x - bw//2 + 2, y - bh//2],
                [x + bw//2 - 2, y - bh//2],
                [x + bw//2, y + bh//3]
            ], dtype=jnp.int32)
            turret_x = x + scale//6
            cannon_end_x = turret_x + jnp.int32(scale*1.2)
            im = img
            for i in range(pts.shape[0]-1):
                im = self._draw_line(im, pts[i,0], pts[i,1], pts[i+1,0], pts[i+1,1], color, samples=64)
            tw = jnp.int32(scale//2); th = jnp.int32(scale//4)
            im = self._fill_rect(im, turret_x - tw//2, y - th//2, tw, th, color)
            im = self._draw_line(im, turret_x, y, cannon_end_x, y, color, samples=32)
            wheel_y = y + bh//2 + 2
            im = self._draw_line(im, x - bw//2, wheel_y, x + bw//2, wheel_y, color, samples=64)
            # wheel dots
            def wheel_body(j, imc):
                cx = x - bw//2 + 2 + j*6
                return self._put_pixel(imc, cx, wheel_y, color)
            num = jnp.maximum(0, (bw-4)//6)
            im = lax.fori_loop(0, num, wheel_body, im)
            return im

        # Ensure `left` is boolean tensor
        left_bool = jnp.asarray(left, dtype=jnp.bool_)
        return lax.cond(left_bool, draw_left, draw_right, operand=None)


    def _draw_saucer(self, img, x, y, distance, color):
        size = jnp.maximum(6, jnp.int32(18 / jnp.maximum(distance / 50.0, 1.0)))
        # Ellipse body (width=2*size, height≈size/1.5), dome (width=size, height≈size/2)
        H = img.shape[0]; W = img.shape[1]
        Y = jnp.arange(H)[:, None]; X = jnp.arange(W)[None, :]

        # Body ellipse: ((X-x)^2)/(a^2) + ((Y-(y - h/3))^2)/(b^2) <= 1
        a = size.astype(jnp.float32); b = (size/1.5).astype(jnp.float32)
        xc = x.astype(jnp.float32); yc = y.astype(jnp.float32) - (b/2)
        body_mask = (( (X - xc)**2 )/(a*a) + ( (Y - yc)**2 )/(b*b)) <= 1.0

        # Dome ellipse on top: ((X-x)^2)/(ad^2) + ((Y-(y - b))^2)/(bd^2) <= 1
        ad = (size/2).astype(jnp.float32); bd = (size/2).astype(jnp.float32)
        ycd = y.astype(jnp.float32) - bd
        dome_mask = (((X - xc)**2)/(ad*ad) + ((Y - ycd)**2)/(bd*bd)) <= 1.0

        mask = body_mask | dome_mask

        col = jnp.asarray(color, jnp.uint8)
        col_img = jnp.broadcast_to(col, img.shape)
        mask3 = jnp.stack([mask, mask, mask], axis=-1)
        im = jnp.where(mask3, col_img, img)

        # glow dots (3 pixels) near bottom of body
        for dx in (-size//3, 0, size//3):
            im = self._put_pixel(im, x + dx, y + size//6, color)
        return im

    def _draw_fighter(self, img, x, y, distance, color):
        size = jnp.maximum(8, jnp.int32(20 / jnp.maximum(distance / 60.0, 1.0)))
        body_w = jnp.int32(size*1.0); body_h = jnp.int32(size*0.7)
        im = self._fill_rect(img, x - body_w//2, y - body_h//2, body_w, body_h, color)
        # outline (thin)
        im = self._draw_line(im, x - body_w//2, y - body_h//2, x + body_w//2, y - body_h//2, (0,0,0), samples=64)
        im = self._draw_line(im, x - body_w//2, y + body_h//2, x + body_w//2, y + body_h//2, (0,0,0), samples=64)
        im = self._draw_line(im, x - body_w//2, y - body_h//2, x - body_w//2, y + body_h//2, (0,0,0), samples=48)
        im = self._draw_line(im, x + body_w//2, y - body_h//2, x + body_w//2, y + body_h//2, (0,0,0), samples=48)
        # wings + stripes
        wing_w = jnp.int32(size*0.9); wing_h = jnp.int32(size*0.5)
        left_x = x - body_w//2 - wing_w; right_x = x + body_w//2; wing_y = y - wing_h//2
        im = self._fill_rect(im, left_x, wing_y, wing_w, wing_h, color)
        im = self._fill_rect(im, right_x, wing_y, wing_w, wing_h, color)
        im = self._draw_line(im, left_x, wing_y, left_x+wing_w, wing_y, (0,0,0), samples=48)
        im = self._draw_line(im, right_x, wing_y, right_x+wing_w, wing_y, (0,0,0), samples=48)
        for i in range(2):
            sy = wing_y + jnp.int32((i+1)*wing_h/3)
            im = self._draw_line(im, left_x+2, sy, left_x+wing_w-2, sy, (0,0,0), samples=44)
            im = self._draw_line(im, right_x+2, sy, right_x+wing_w-2, sy, (0,0,0), samples=44)
        # nose block
        nose_w = jnp.int32(body_w*0.4); nose_h = jnp.int32(body_h*0.5)
        im = self._fill_rect(im, x + body_w//2 - 2, y - nose_h//2, nose_w, nose_h, (0,0,0))
        return im

    
    
    def _draw_enemy_tank(self, img, x, y, distance, color, tank_angle, player_angle):
        # Use lax.cond instead of Python if to remain jit-safe.
        def do_nothing(_):
            return img
        def do_draw(_):
            scale = jnp.maximum(4, jnp.int32(20 / jnp.maximum(distance / 50.0, 1.0)))
            rel = tank_angle - player_angle
            rel = jnp.arctan2(jnp.sin(rel), jnp.cos(rel))
            frontal = (jnp.abs(rel) < jnp.pi/4) | (jnp.abs(rel) > 3*jnp.pi/4)
            return lax.cond(
                frontal,
                lambda __: self._draw_enemy_tank_frontal(img, x, y, scale, color),
                lambda __: self._draw_enemy_tank_profile(img, x, y, scale, color, left=(rel>0)),
                operand=None
            )
        return lax.cond(distance > self.view_distance, do_nothing, do_draw, operand=None)


    
    def _draw_wire_primitives(self, img, state, wire):
        px = state.player_tank.x; py = state.player_tank.y; pang = state.player_tank.angle
        obs = state.obstacles
        im = img

        def draw_enemy(im_in, i):
            # Body draws conditionally on visibility/inside
            sx, sy, dist, vis = self._world_to_screen_3d(obs.x[i], obs.y[i], px, py, pang)
            inside = (sx >= 0) & (sx < WIDTH) & (sy >= 0) & (sy < HEIGHT)
            cond_visible = vis & inside

            def when_visible(_):
                # choose subtype color
                subtype = jnp.int32(obs.enemy_subtype[i]) if hasattr(obs, "enemy_subtype") else jnp.int32(-1)

                def pick_color():
                    # default wireframe
                    col = jnp.array(WIREFRAME_COLOR, jnp.uint8)
                    col = jnp.where(subtype == ENEMY_TYPE_SUPERTANK, jnp.array(SUPERTANK_COLOR, jnp.uint8), col)
                    col = jnp.where(subtype == ENEMY_TYPE_TANK,      jnp.array(TANK_COLOR, jnp.uint8),      col)
                    col = jnp.where(subtype == ENEMY_TYPE_FIGHTER,   jnp.array(FIGHTER_COLOR, jnp.uint8),   col)
                    col = jnp.where(subtype == ENEMY_TYPE_SAUCER,    jnp.array(SAUCER_COLOR, jnp.uint8),    col)
                    return col

                color = pick_color()

                def draw_saucer(_):
                    return self._draw_saucer(im_in, sx, sy, dist, SAUCER_COLOR)

                def draw_fighter(_):
                    return self._draw_fighter(im_in, sx, sy, dist, FIGHTER_COLOR)

                def draw_tank(col):
                    return self._draw_enemy_tank(im_in, sx, sy, dist, col, obs.angle[i], pang)

                # route by subtype
                im_out = lax.cond(subtype == ENEMY_TYPE_SAUCER,
                                  draw_saucer,
                                  lambda _: im_in, operand=None)
                im_out = lax.cond(subtype == ENEMY_TYPE_FIGHTER,
                                  draw_fighter,
                                  lambda _: im_out, operand=None)
                # For tanks / default: draw tank with selected color
                is_tank_like = (subtype == ENEMY_TYPE_TANK) | (subtype == ENEMY_TYPE_SUPERTANK) | ((subtype != ENEMY_TYPE_FIGHTER) & (subtype != ENEMY_TYPE_SAUCER))
                im_out = lax.cond(is_tank_like,
                                  lambda _: draw_tank(color),
                                  lambda _: im_out, operand=None)
                return im_out

            return lax.cond(cond_visible, when_visible, lambda _: im_in, operand=None)

        # Loop over obstacles with fori_loop to keep jit-friendly semantics
        def obs_body(i, im_carry):
            alive_i = (obs.alive[i] == 1) if obs.alive.dtype != jnp.bool_ else obs.alive[i]
            return lax.cond(alive_i, lambda _: draw_enemy(im_carry, i), lambda _: im_carry, operand=None)

        im = lax.fori_loop(0, obs.x.shape[0], obs_body, im)

        # bullets
        b = state.bullets
        def bullets_body(i, im_carry):
            active_i = b.active[i]
            def when_active(_):
                sx, sy, dist, vis = self._world_to_screen_3d(b.x[i], b.y[i], px, py, pang)
                inside = (sx >= 0) & (sx < WIDTH) & (sy >= 0) & (sy < HEIGHT)
                col = jnp.where(b.owner[i] == 0,
                                jnp.array(BULLET_COLOR, jnp.uint8),
                                jnp.array([255,100,100], jnp.uint8))
                return lax.cond(vis & inside,
                                lambda _: self._draw_line(im_carry, sx, sy-3, sx, sy+3, col, samples=8),
                                lambda _: im_carry, operand=None)
            return lax.cond(active_i, when_active, lambda _: im_carry, operand=None)

        im = lax.fori_loop(0, b.x.shape[0], bullets_body, im)

        # player tank sprite (bottom center)
        base_x = WIDTH // 2
        base_y = HEIGHT - self.hud_bar_height - 10
        # top-left so that sprite is centered around (base_x, base_y)
        px0 = jnp.int32(base_x - (self.player_w // 2))
        py0 = jnp.int32(base_y - (self.player_h // 2))
        # If sprite is a mask: tint with WIREFRAME_COLOR; if RGB/RGBA: draw as-is
        im = self._blit_sprite(im, self.player_sprite, px0, py0, tint_rgb=WIREFRAME_COLOR if self.player_sprite.ndim == 2 else None)
        return im


    
    def _draw_radar(self, img, state):
        radar_radius = int(WIDTH * 0.12)
        top_bar_h = radar_radius * 2 + 12
        im = self._fill_rect(img, 0, 0, WIDTH, top_bar_h, (0,0,0))
        cx = WIDTH // 2; cy = top_bar_h // 2
        # circle
        im = self._draw_circle_outline(im, cx, cy, radar_radius, (0,255,0), samples=180)
        # sweep
        sweep_speed = 0.025
        angle = (state.step_counter.astype(jnp.float32) * sweep_speed) % (2*jnp.pi)
        sx = jnp.int32(jnp.round(cx + (radar_radius-2)*jnp.cos(angle - jnp.pi/2)))
        sy = jnp.int32(jnp.round(cy + (radar_radius-2)*jnp.sin(angle - jnp.pi/2)))
        im = self._draw_line(im, cx, cy, sx, sy, (255,255,255), samples=radar_radius)
        # ticks
        left_ang = -jnp.pi/2 - 0.6; right_ang = -jnp.pi/2 + 0.6
        edge_r = radar_radius - 2; tick_len = jnp.maximum(4, jnp.int32(12*0.6))
        def tick(a, imc):
            x0 = jnp.int32(jnp.round(cx + edge_r*jnp.cos(a)))
            y0 = jnp.int32(jnp.round(cy + edge_r*jnp.sin(a)))
            x1 = jnp.int32(jnp.round(cx + (edge_r - tick_len)*jnp.cos(a)))
            y1 = jnp.int32(jnp.round(cy + (edge_r - tick_len)*jnp.sin(a)))
            return self._draw_line(imc, x0, y0, x1, y1, WIREFRAME_COLOR, samples=16)
        im = tick(left_ang, im); im = tick(right_ang, im)

        # scale
        scale = (radar_radius - 4) / (WORLD_SIZE / 2.0)
        px = state.player_tank.x; py = state.player_tank.y; pang = state.player_tank.angle
        cos_a = jnp.cos(pang); sin_a = jnp.sin(pang)

        # enemies (loop with fori_loop, no Python bool)
        o = state.obstacles
        def enemies_body(i, im_carry):
            alive_i = o.alive[i]
            def when_alive(_):
                subtype = jnp.int32(o.enemy_subtype[i]) if hasattr(o, "enemy_subtype") else jnp.int32(-1)
                enemy_color = jnp.array([255,0,0], jnp.uint8)
                enemy_color = jnp.where(subtype == ENEMY_TYPE_SUPERTANK, jnp.array(SUPERTANK_COLOR, jnp.uint8), enemy_color)
                enemy_color = jnp.where(subtype == ENEMY_TYPE_TANK,      jnp.array(TANK_COLOR, jnp.uint8),      enemy_color)
                enemy_color = jnp.where(subtype == ENEMY_TYPE_FIGHTER,   jnp.array(FIGHTER_COLOR, jnp.uint8),   enemy_color)
                # hide saucer
                def draw_enemy(_):
                    rel_x = (o.x[i] - px).astype(jnp.float32); rel_y = (o.y[i] - py).astype(jnp.float32)
                    view_x = -rel_x * sin_a + rel_y * cos_a
                    view_y =  rel_x * cos_a + rel_y * sin_a
                    rx = jnp.int32(jnp.round(cx + view_x * scale)); ry = jnp.int32(jnp.round(cy - view_y * scale))
                    inside = ((rx-cx)**2 + (ry-cy)**2) <= (radar_radius-3)**2
                    def when_inside(_):
                        im2 = self._fill_rect(im_carry, rx-1, ry-1, 3, 3, enemy_color)
                        enemy_angle = o.angle[i].astype(jnp.float32)
                        rel = enemy_angle - pang.astype(jnp.float32)
                        end_x = rx + jnp.int32(jnp.round(4 * jnp.cos(rel)))
                        end_y = ry + jnp.int32(jnp.round(4 * jnp.sin(rel)))
                        im2 = self._draw_line(im2, rx, ry, end_x, end_y, enemy_color, samples=6)
                        return im2
                    return lax.cond(inside, when_inside, lambda _: im_carry, operand=None)
                return lax.cond(subtype == ENEMY_TYPE_SAUCER, lambda _: im_carry, draw_enemy, operand=None)
            return lax.cond(alive_i.astype(jnp.bool_), when_alive, lambda _: im_carry, operand=None)

        im = lax.fori_loop(0, o.x.shape[0], enemies_body, im)

        # bullets
        b = state.bullets
        def bullets_body(i, im_carry):
            active_i = b.active[i]
            def when_active(_):
                rel_x = (b.x[i] - px).astype(jnp.float32); rel_y = (b.y[i] - py).astype(jnp.float32)
                view_x = -rel_x * sin_a + rel_y * cos_a
                view_y =  rel_x * cos_a + rel_y * sin_a
                rx = jnp.int32(jnp.round(cx + view_x * scale)); ry = jnp.int32(jnp.round(cy - view_y * scale))
                inside = ((rx-cx)**2 + (ry-cy)**2) <= (radar_radius-3)**2
                col = jnp.where(b.owner[i] == 0,
                                jnp.array(BULLET_COLOR, jnp.uint8),
                                jnp.array([255,100,100], jnp.uint8))
                def when_inside(_):
                    return self._put_pixel(im_carry, rx, ry, col)
                return lax.cond(inside, when_inside, lambda _: im_carry, operand=None)
            return lax.cond(active_i.astype(jnp.bool_), when_active, lambda _: im_carry, operand=None)

        im = lax.fori_loop(0, b.x.shape[0], bullets_body, im)

        # bottom HUD bar
        im = self._fill_rect(im, 0, HEIGHT - self.hud_bar_height, WIDTH, self.hud_bar_height, (0,0,0))

        # draw lives as rectangles
        try:
            lives_val = jnp.int32(state.player_lives)
        except Exception:
            lives_val = jnp.int32(0)
        lives_to_draw = jnp.clip(state.player_lives, 0, 9)  # z.B. max. 9 anzeigen
        # Layout: mittig unten in der HUD-Bar
        life_spacing = self.life_w + 6  # 6px Abstand zwischen Sprites
        total_w = lives_to_draw * life_spacing - jnp.where(lives_to_draw > 0, 6, 0)
        life_start_x = (WIDTH // 2) - (total_w // 2)
        # leicht über der Mitte der HUD-Bar platzieren
        # Score sitzt bei y = HEIGHT - self.hud_bar_height + 4, Höhe der Ziffern = 10 (scale=2)
        # Abstand 2px darunter für die Leben (Sprite-Top)
        score_top = HEIGHT - self.hud_bar_height + 4
        score_h   = 10
        life_gap  = 2
        life_y    = score_top + score_h + life_gap
        
        def lives_body(i, im_carry):
            lx = jnp.int32(life_start_x + i * life_spacing)
            ly = jnp.int32(life_y)
            return self._blit_sprite(im_carry, self.life_sprite, lx, ly, tint_rgb=HUD_ACCENT_COLOR)

        im = lax.fori_loop(0, lives_to_draw, lives_body, im)

        # --- SCORE über den Leben, mittig rechts, wie Screenshot ---
        im = self._draw_score_hud(im, state.player_score)

        return im


    # ---------------- main render ----------------

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state) -> jnp.ndarray:
        # create base image
        img = jnp.zeros((HEIGHT, WIDTH, 3), dtype=jnp.uint8)

        # colors
        wire = jnp.asarray(WIREFRAME_COLOR, jnp.uint8)
        hud  = jnp.asarray(HUD_ACCENT_COLOR, jnp.uint8)

        # store pose for sky parallax
        # sky + ground + horizon
        img = self._draw_sky(img, state.player_tank.x, state.player_tank.angle)
        img = self._draw_ground(img, state)
        img = self._draw_line(img, 0, HORIZON_Y, WIDTH-1, HORIZON_Y, hud, samples=WIDTH)

        # enemies + bullets + player
        img = self._draw_wire_primitives(img, state, wire)

        # crosshair
        img = self._draw_crosshair(img, state, wire)

        # radar + bottom HUD
        img = self._draw_radar(img, state)

        return img
