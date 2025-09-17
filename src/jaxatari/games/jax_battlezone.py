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
        """Observation space (callable) using numpy dtypes.

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
            "step_counter": spaces.Box(low=0, high=INT32_MAX, shape=(), dtype=np.int32),
            "spawn_timer": spaces.Box(low=0, high=ENEMY_SPAWN_COOLDOWN * 4, shape=(), dtype=np.int32),
            "player_score": spaces.Box(low=0, high=INT32_MAX, shape=(), dtype=np.int32),
            "player_lives": spaces.Box(low=0, high=99, shape=(), dtype=np.int32),
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

        return jnp.concatenate([player_flat, bullets_flat, obstacles_flat]).astype(jnp.float32)

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
    """Rein-JAX 3D-Wireframe-Renderer für BattleZone (jit-kompatibel, kein pygame)."""

    def __init__(self):
        self.view_distance = 400.0
        self.fov = 60.0
        self.hud_bar_height = 28  # Top/Bottom-HUD-Höhe

    # ----------------------- Low-level Drawing Primitives (JAX) -----------------------

    @staticmethod
    def _put_pixel(img, x, y, color):
        h, w = img.shape[0], img.shape[1]
        x = jnp.asarray(x, jnp.int32)
        y = jnp.asarray(y, jnp.int32)
        valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        def set_pix(i):
            return img.at[y, x, :].set(jnp.asarray(color, jnp.uint8))
        return lax.cond(valid, set_pix, lambda i: img, operand=None)

    @staticmethod
    def _put_pixels(img, xs, ys, color):
        h, w = img.shape[0], img.shape[1]
        xs = jnp.clip(jnp.asarray(xs, jnp.int32), 0, w - 1)
        ys = jnp.clip(jnp.asarray(ys, jnp.int32), 0, h - 1)
        idx = (ys, xs, jnp.arange(3))
        # Set per-channel via scatter with broadcasting
        color_u8 = jnp.asarray(color, jnp.uint8)
        img = img.at[ys, xs, 0].set(color_u8[0])
        img = img.at[ys, xs, 1].set(color_u8[1])
        img = img.at[ys, xs, 2].set(color_u8[2])
        return img

    @staticmethod
    def _fill_hspan(img, y, x0, x1, color):
        h, w = img.shape[0], img.shape[1]
        y = jnp.clip(jnp.asarray(y, jnp.int32), 0, h - 1)
        x0 = jnp.clip(jnp.minimum(x0, x1), 0, w - 1)
        x1 = jnp.clip(jnp.maximum(x0, x1), 0, w - 1)
        xs = jnp.arange(x0, x1 + 1)
        return BattleZoneRenderer._put_pixels(img, xs, jnp.full_like(xs, y), color)

    @staticmethod
    def _fill_rect(img, x, y, w, h, color):
        # Bildgröße
        H = img.shape[0]
        W = img.shape[1]

        # Grenzen als int32 clampen (bleiben Tracer-fähig)
        x0 = jnp.clip(jnp.asarray(x, jnp.int32), 0, W - 1)
        y0 = jnp.clip(jnp.asarray(y, jnp.int32), 0, H - 1)
        x1 = jnp.clip(x0 + jnp.asarray(w, jnp.int32) - 1, 0, W - 1)
        y1 = jnp.clip(y0 + jnp.asarray(h, jnp.int32) - 1, 0, H - 1)

        # Statische Achsenvektoren (0..H-1 / 0..W-1) -> keine dynamischen arange-Starts!
        ys_all = jnp.arange(H, dtype=jnp.int32)
        xs_all = jnp.arange(W, dtype=jnp.int32)

        # Rechteck-Maske über das ganze Bild
        ymask = (ys_all >= y0) & (ys_all <= y1)              # (H,)
        xmask = (xs_all >= x0) & (xs_all <= x1)              # (W,)
        mask = ymask[:, None] & xmask[None, :]               # (H, W)

        col = jnp.asarray(color, dtype=jnp.uint8)            # (3,)
        # Auf 3 Kanäle broadcasten und setzen
        return jnp.where(mask[..., None], col, img)
        def body(i, im):
            xs = jnp.arange(x0, x1 + 1)
            return BattleZoneRenderer._put_pixels(im, xs, jnp.full_like(xs, ys[i]), color)
        return lax.fori_loop(0, ys.size, body, img)

    def _draw_line(self, img, x0, y0, x1, y1, color, samples):
        """
        Zeichnet eine Linie von (x0,y0) nach (x1,y1) in die RGB8-Image-Array `img`.
        Jit-sicher: vermeidet jnp.linspace mit dynamischem `num`.
        """
        H, W = img.shape[0], img.shape[1]

        # Feste Obergrenze für Oversampling (statisch für jax.jit)
        MAX_SAMPLES = 1024  # ggf. auf 512 reduzieren, falls Performance nötig

        # gewünschte Sample-Anzahl (Tracer) sicher clampen
        s = jnp.clip(jnp.asarray(samples, jnp.int32), 1, MAX_SAMPLES)

        # Statische t-Stützstellen 0..1 mit fixer Länge
        i = jnp.arange(MAX_SAMPLES, dtype=jnp.int32)
        t = i.astype(jnp.float32) / (MAX_SAMPLES - 1.0)

        # Interpolation & Runden auf Pixelkoordinaten
        xs = jnp.round(x0 + (x1 - x0) * t).astype(jnp.int32)
        ys = jnp.round(y0 + (y1 - y0) * t).astype(jnp.int32)

        # Maskiere nur die ersten `s` Punkte (dynamisch, aber Form bleibt statisch)
        use = i < s

        # Clamping ins Bild
        xs = jnp.clip(xs, 0, W - 1)
        ys = jnp.clip(ys, 0, H - 1)

        col = jnp.asarray(color, dtype=jnp.uint8)

        # Iteriere mit fixer Obergrenze und setze Pixel nur wenn use[idx] True
        def body(k, im):
            return jax.lax.cond(
                use[k],
                lambda im2: im2.at[ys[k], xs[k], :].set(col),
                lambda im2: im2,
                im,
            )

        img = jax.lax.fori_loop(0, MAX_SAMPLES, body, img)
        return img

    @staticmethod
    def _draw_circle_outline(img, cx, cy, r, color, samples=360):
        ang = jnp.linspace(0.0, 2.0 * jnp.pi, samples)
        xs = jnp.round(cx + r * jnp.cos(ang)).astype(jnp.int32)
        ys = jnp.round(cy + r * jnp.sin(ang)).astype(jnp.int32)
        return BattleZoneRenderer._put_pixels(img, xs, ys, color)

    @staticmethod
    def _draw_polygon_wire(img, points_xy, color):
        n = points_xy.shape[0]
        def body(i, im):
            p0 = points_xy[i]
            p1 = points_xy[(i + 1) % n]
            return BattleZoneRenderer._draw_line(im, p0[0], p0[1], p1[0], p1[1], color)
        return lax.fori_loop(0, n, body, img)

    # --------------------------- 3D Projection / Utility -----------------------------

    @staticmethod
    def _world_to_screen_3d(world_x, world_y, player_x, player_y, player_angle):
        # wie im Original, aber vollständig in jnp
        rel_x = world_x - player_x
        rel_y = world_y - player_y
        cos_a = jnp.cos(player_angle)
        sin_a = jnp.sin(player_angle)
        view_x = -rel_x * sin_a + rel_y * cos_a
        view_y = rel_x * cos_a + rel_y * sin_a
        fov_scale = 80.0
        perspective_scale = 100.0

        visible = view_y > 1.0
        screen_x = (WIDTH // 2) + (view_x / jnp.maximum(view_y, 1e-6)) * fov_scale
        screen_y = HORIZON_Y + (perspective_scale / jnp.maximum(view_y, 1e-6))
        return screen_x.astype(jnp.int32), screen_y.astype(jnp.int32), view_y, visible

    # ----------------------------- High-level Elements -------------------------------

    def _draw_sky(self, img):
        sky_h = HORIZON_Y
        y = jnp.arange(sky_h, dtype=jnp.float32)
        t = y / jnp.maximum(sky_h - 1, 1)
        r = (60 * (1 - t) + 10 * t).astype(jnp.uint8)
        g = (120 * (1 - t) + 40 * t).astype(jnp.uint8)
        b = (200 * (1 - t) + 120 * t).astype(jnp.uint8)
        row = jnp.stack([r, g, b], axis=1)                      # (sky_h, 3)
        rows = jnp.repeat(row[:, None, :], WIDTH, axis=1)       # (sky_h, W, 3)
        return img.at[:sky_h, :, :].set(rows)

    def _draw_ground(self, img, state):
        ground_bands = 16
        band_h = (HEIGHT - HORIZON_Y) // ground_bands
        # Farbstaffelung wie vorher
        palette = jnp.array([
            [60,120,60],[70,130,60],[80,140,60],[90,150,60],
            [100,160,60],[110,170,60],[120,180,60],[130,190,60],
            [140,200,60],[150,210,60],[160,220,60],[170,230,60],
            [180,240,60],[170,230,60],[160,220,60],[150,210,60]
        ], dtype=jnp.uint8)

        # forward_delta (Parallax wie zuvor, rein JAX)
        px, py = state.player_tank.x, state.player_tank.y
        ppx, ppy = state.prev_player_x, state.prev_player_y
        pang = state.player_tank.angle
        forward_delta = jnp.cos(pang) * (px - ppx) + jnp.sin(pang) * (py - ppy)
        ground_offset = jnp.mod((py * 0.15 - forward_delta * 20.0 + pang * 2.0).astype(jnp.int32), ground_bands)

        def body(i, im):
            color = palette[(i + ground_offset) % ground_bands]
            y1 = HORIZON_Y + i * band_h
            h = jnp.where(i < ground_bands - 1, band_h, HEIGHT - (HORIZON_Y + i * band_h))
            return self._fill_rect(im, 0, y1, WIDTH, h, color)
        return lax.fori_loop(0, ground_bands, body, img)

    def _draw_horizon_and_crosshair(self, img, state):
        # Horizon
        img = self._draw_line(img, 0, HORIZON_Y, WIDTH - 1, HORIZON_Y, WIREFRAME_COLOR)
        # Crosshair (vertikaler Balken leicht über dem Horizont)
        center_x = WIDTH // 2
        cross_thickness = 1
        cross_height = cross_thickness * 4
        cross_bottom = HORIZON_Y - 6
        cross_top = cross_bottom - cross_height
        # Outline
        img = self._draw_line(img, center_x - 1, cross_top - 1, center_x - 1, cross_bottom + 1, WIREFRAME_COLOR)
        img = self._draw_line(img, center_x + 1, cross_top - 1, center_x + 1, cross_bottom + 1, WIREFRAME_COLOR)
        img = self._draw_line(img, center_x, cross_top, center_x, cross_bottom, (255,255,255))
        return img

    def _draw_wireframe_cube(self, img, x, y, dist, color):
        # einfache perspektivische Quadrate (Front/Back) + Kanten
        s = jnp.maximum(4, (20.0 / jnp.maximum(dist / 50.0, 1.0))).astype(jnp.int32)
        half = s // 2
        # Quadrat
        pts = jnp.array([[x-half,y-half],[x+half,y-half],[x+half,y+half],[x-half,y+half]], dtype=jnp.int32)
        return self._draw_polygon_wire(img, pts, color)

    def _draw_wireframe_pyramid(self, img, x, y, dist, color):
        s = jnp.maximum(4, (20.0 / jnp.maximum(dist / 50.0, 1.0))).astype(jnp.int32)
        half = s // 2
        base = jnp.array([[x-half,y+half],[x+half,y+half],[x+half,y],[x-half,y]], dtype=jnp.int32)
        img = self._draw_polygon_wire(img, base, color)
        apex = jnp.array([x, y - half], dtype=jnp.int32)
        # Kanten zum Apex
        for i in range(base.shape[0]):
            img = self._draw_line(img, base[i,0], base[i,1], apex[0], apex[1], color)
        return img

    def _draw_enemy_tank_wire(self, img, x, y, dist, color):
        # vereinheitlichte, schlanke Wireframe-Darstellung (Frontal/Profil stilisiert)
        scale = jnp.maximum(4, (20.0 / jnp.maximum(dist / 50.0, 1.0))).astype(jnp.int32)
        bw = (scale * 1.0).astype(jnp.int32)
        bh = (scale * 0.7).astype(jnp.int32)
        pts = jnp.array([
            [x - bw//2, y + bh//2],
            [x + bw//2, y + bh//2],
            [x + bw//3, y - bh//2],
            [x - bw//3, y - bh//2],
        ], dtype=jnp.int32)
        img = self._draw_polygon_wire(img, pts, color)
        # Turm
        tw = (scale // 3).astype(jnp.int32)
        th = (scale // 3).astype(jnp.int32)
        img = self._draw_polygon_wire(
            img,
            jnp.array([
                [x - tw//2, y - th//2],
                [x + tw//2, y - th//2],
                [x + tw//2, y + th//2],
                [x - tw//2, y + th//2]], dtype=jnp.int32),
            color
        )
        # Kanone
        img = self._draw_line(img, x, y, x, y + (scale // 2), color)
        return img

    def _draw_fighter(self, img, x, y, dist, color):
        size = jnp.maximum(8, (20.0 / jnp.maximum(dist / 60.0, 1.0))).astype(jnp.int32)
        bw = (size * 1.0).astype(jnp.int32)
        bh = (size * 0.7).astype(jnp.int32)
        # Körper-Rahmen
        body = jnp.array([
            [x - bw//2, y - bh//2],
            [x + bw//2, y - bh//2],
            [x + bw//2, y + bh//2],
            [x - bw//2, y + bh//2]], dtype=jnp.int32)
        img = self._draw_polygon_wire(img, body, color)
        # Wings als Linien
        wing_w = (size * 0.9).astype(jnp.int32)
        wxL0, wxL1 = x - bw//2 - wing_w, x - bw//2
        wxR0, wxR1 = x + bw//2, x + bw//2 + wing_w
        wy0, wy1 = y - (size // 4), y + (size // 4)
        img = self._draw_line(img, wxL0, wy0, wxL1, wy0, color)
        img = self._draw_line(img, wxL0, wy1, wxL1, wy1, color)
        img = self._draw_line(img, wxR0, wy0, wxR1, wy0, color)
        img = self._draw_line(img, wxR0, wy1, wxR1, wy1, color)
        return img

    def _draw_saucer(self, img, x, y, dist, color):
        size = jnp.maximum(6, (18.0 / jnp.maximum(dist / 50.0, 1.0))).astype(jnp.int32)
        # Ellipse -> als Kreis-Outline plus Querlinie
        img = self._draw_circle_outline(img, x, y, jnp.maximum(2, size//2), color)
        img = self._draw_line(img, x - size, y, x + size, y, color)
        return img

    def _draw_bullets(self, img, state):
        b = state.bullets
        def body(i, im):
            active = b.active[i] > 0
            def draw_one(_):
                sx, sy, dist, vis = self._world_to_screen_3d(b.x[i], b.y[i],
                                                             state.player_tank.x, state.player_tank.y, state.player_tank.angle)
                col = jnp.where(b.owner[i] == 0,
                                jnp.array(BULLET_COLOR, jnp.uint8),
                                jnp.array([255,100,100], jnp.uint8))
                im2 = lax.cond(vis & (sx>=0) & (sx<WIDTH) & (sy>=0) & (sy<HEIGHT),
                               lambda __: self._draw_line(im, sx, sy-3, sx, sy+3, col),
                               lambda __: im, operand=None)
                return im2
            return lax.cond(active, draw_one, lambda _: im, operand=None)
        return lax.fori_loop(0, b.x.shape[0], body, img)

    def _draw_player_tank(self, img):
        # simplifiziertes Wireframe-Tank am unteren Bildschirmrand
        base_x = WIDTH // 2
        base_y = HEIGHT - self.hud_bar_height - 10
        return self._draw_wireframe_cube(img, base_x, base_y, 30.0, WIREFRAME_COLOR)

    def _draw_radar(self, img, state):
        radar_radius = int(WIDTH * 0.12)
        top_bar_h = radar_radius * 2 + 12
        # Top-Bar schwarz
        img = self._fill_rect(img, 0, 0, WIDTH, top_bar_h, (0,0,0))
        cx = WIDTH // 2
        cy = top_bar_h // 2
        img = self._draw_circle_outline(img, cx, cy, radar_radius, (0,255,0))

        # Sweep-Linie
        sweep_speed = 0.025
        angle = (state.step_counter * sweep_speed) % (2 * math.pi)
        sx = int(cx + (radar_radius - 2) * math.cos(angle - math.pi/2))
        sy = int(cy + (radar_radius - 2) * math.sin(angle - math.pi/2))
        img = self._draw_line(img, cx, cy, sx, sy, (255,255,255))

        # kleine Ticks (wie original)
        def tick(im, ang):
            edge_r = radar_radius - 2
            tick_len = max(4, int(12 * 0.6))
            x0 = int(cx + edge_r * math.cos(ang))
            y0 = int(cy + edge_r * math.sin(ang))
            x1 = int(cx + (edge_r - tick_len) * math.cos(ang))
            y1 = int(cy + (edge_r - tick_len) * math.sin(ang))
            return self._draw_line(im, x0, y0, x1, y1, WIREFRAME_COLOR)
        img = tick(img, -math.pi/2 - 0.6)
        img = tick(img, -math.pi/2 + 0.6)

        # Radar-Objekte
        scale = (radar_radius - 4) / (WORLD_SIZE / 2)
        px, py, pang = state.player_tank.x, state.player_tank.y, state.player_tank.angle
        cos_a, sin_a = jnp.cos(pang), jnp.sin(pang)

        # Enemies
        ox = state.obstacles.x
        oy = state.obstacles.y
        alive = state.obstacles.alive > 0

        def draw_enemy(i, im):
            def draw_vis(_):
                rel_x = ox[i] - px
                rel_y = oy[i] - py
                view_x = -rel_x * sin_a + rel_y * cos_a
                view_y =  rel_x * cos_a + rel_y * sin_a
                rx = (cx + view_x * scale).astype(jnp.int32)
                ry = (cy - view_y * scale).astype(jnp.int32)
                inside = (rx-cx)**2 + (ry-cy)**2 <= (radar_radius-3)**2
                # Farbe nach subtype (Saucers nicht zeigen)
                subtype = jnp.asarray(state.obstacles.enemy_subtype[i], jnp.int32)
                # saucer skip
                def draw_pixel(__):
                    color = jnp.where(subtype==ENEMY_TYPE_SUPERTANK, jnp.array(SUPERTANK_COLOR, jnp.uint8),
                             jnp.where(subtype==ENEMY_TYPE_TANK, jnp.array(TANK_COLOR, jnp.uint8),
                             jnp.where(subtype==ENEMY_TYPE_FIGHTER, jnp.array(FIGHTER_COLOR, jnp.uint8),
                                       jnp.array([255,0,0], jnp.uint8))))
                    return self._put_pixel(im, rx, ry, color)
                return lax.cond(inside & (subtype != ENEMY_TYPE_SAUCER), draw_pixel, lambda __: im, operand=None)
            return lax.cond(alive[i], draw_vis, lambda __: im, operand=None)
        img = lax.fori_loop(0, ox.shape[0], draw_enemy, img)

        # Bullets
        b = state.bullets
        def draw_b(i, im):
            def draw_vis(_):
                rel_x = b.x[i] - px
                rel_y = b.y[i] - py
                view_x = -rel_x * sin_a + rel_y * cos_a
                view_y =  rel_x * cos_a + rel_y * sin_a
                rx = (cx + view_x * scale).astype(jnp.int32)
                ry = (cy - view_y * scale).astype(jnp.int32)
                inside = (rx-cx)**2 + (ry-cy)**2 <= (radar_radius-3)**2
                col = jnp.where(b.owner[i]==0, jnp.array([255,255,255], jnp.uint8),
                                              jnp.array([255,100,100], jnp.uint8))
                return lax.cond(inside, lambda __: self._put_pixel(im, rx, ry, col), lambda __: im, operand=None)
            return lax.cond(b.active[i]>0, draw_vis, lambda __: im, operand=None)
        img = lax.fori_loop(0, b.x.shape[0], draw_b, img)

        # Bottom HUD-Bar (Score & Lives)
        img = self._fill_rect(img, 0, HEIGHT - self.hud_bar_height, WIDTH, self.hud_bar_height, (0,0,0))

        # Minimalistische Score/Lives-Anzeige als Blöcke (keine Fonts in JAX)
        # Score mittig als Balkengruppe (visuell äquivalente Position/Betonung)
        # -> Bewahrt UI-Layout ohne pygame-Fonts.
        # (Option: Wenn du eine eig. 7-Segment-JAX-Zeichnung hast, kannst du sie hier einsetzen.)
        return img

    # ------------------------------- Main render() -----------------------------------

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state) -> jnp.ndarray:
        # Statisches Bild-Shape statt image_space() (Tracer-sicher)
        img = jnp.zeros((HEIGHT, WIDTH, 3), dtype=jnp.uint8)

        # Fallback-Farben, falls im Modul definiert – ansonsten Defaults
        try:
            wire = jnp.asarray(WIREFRAME_COLOR, jnp.uint8)
        except NameError:
            wire = jnp.array([0, 255, 0], dtype=jnp.uint8)
        try:
            hud  = jnp.asarray(HUD_ACCENT_COLOR, jnp.uint8)
        except NameError:
            hud  = jnp.array([255, 255, 255], dtype=jnp.uint8)
        try:
            bullet_col = jnp.asarray(BULLET_COLOR, jnp.uint8)
        except NameError:
            bullet_col = jnp.array([255, 255, 255], dtype=jnp.uint8)

        # 1) Sky & Ground + Horizont
        img = self._draw_sky(img)          # nutzt HORIZON_Y intern
        img = self._draw_ground(img, state)
        img = self._draw_line(img, 0, HORIZON_Y, WIDTH - 1, HORIZON_Y, hud, samples=WIDTH)

        # Spielerpose
        px  = state.player_tank.x
        py  = state.player_tank.y
        pang = state.player_tank.angle

        # 2) Hindernisse/Enemies als einfache Stäbe (sichtbar, billig zu zeichnen)
        ox = state.obstacles.x
        oy = state.obstacles.y
        oalive = state.obstacles.alive.astype(jnp.bool_)
        on = ox.shape[0]

        def draw_one_obs(i, im):
            alive = oalive[i]
            wx, wy = ox[i], oy[i]
            # 3D->Screen
            sx, sy, vz, visible = self._world_to_screen_3d(wx, wy, px, py, pang)
            # Größe abhängig von Tiefe
            h = jnp.clip(jnp.int32(50.0 / jnp.maximum(vz, 1.0)), 3, 30)
            top_y = sy - h
            # Nur zeichnen, wenn sichtbar, alive und im Sichtbereich
            cond = alive & visible & (sx >= 0) & (sx < WIDTH) & (sy >= 0) & (sy < HEIGHT)
            def do_draw(_):
                im2 = self._draw_line(im, sx, sy, sx, top_y, wire, samples=h*2)
                # kleine Querlinie oben, damit es „körperhaft“ wirkt
                return self._draw_line(im2, sx - 2, top_y, sx + 2, top_y, wire, samples=8)
            return lax.cond(cond, do_draw, lambda _: im, operand=None)

        img = lax.fori_loop(0, on, draw_one_obs, img)

        # 3) Bullets als Punkte
        bx = state.bullets.x
        by = state.bullets.y
        bactive = state.bullets.active.astype(jnp.bool_)
        bn = bx.shape[0]

        def draw_one_b(i, im):
            act = bactive[i]
            wx, wy = bx[i], by[i]
            sx, sy, vz, visible = self._world_to_screen_3d(wx, wy, px, py, pang)
            cond = act & visible & (sx >= 0) & (sx < WIDTH) & (sy >= 0) & (sy < HEIGHT)
            def do_draw(_):
                return self._put_pixel(im, sx, sy, bullet_col)
            return lax.cond(cond, do_draw, lambda _: im, operand=None)

        img = lax.fori_loop(0, bn, draw_one_b, img)

        # 4) Zielkreuz/„Player HUD“ (einfaches Fadenkreuz mittig)
        cx = WIDTH // 2
        cy = HEIGHT - self.hud_bar_height - 12
        img = self._draw_line(img, cx - 4, cy, cx + 4, cy, hud, samples=16)
        img = self._draw_line(img, cx, cy - 4, cx, cy + 4, hud, samples=16)

        return img
