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
import gym
from gym import spaces

from jaxatari.environment import JaxEnvironment
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as aj

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
    obstacle_type: chex.Array  # 0: enemy tank, 1: other obstacles
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
    """Update enemy tank AI with Atari-like behaviour: instantly face player and approach along the direct vector."""
    
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
        
        should_process = jnp.logical_and(enemy_alive, is_enemy_tank)

        # Vector to player and distance
        dx = player_tank.x - enemy_x
        dy = player_tank.y - enemy_y
        distance_to_player = jnp.sqrt(dx * dx + dy * dy) + 1e-8
        dir_x = dx / distance_to_player
        dir_y = dy / distance_to_player
        angle_to_player = jnp.arctan2(dy, dx)

        # Defaults
        new_x = enemy_x
        new_y = enemy_y
        new_angle = enemy_angle
        new_state = enemy_state
        new_target_angle = enemy_target_angle
        new_state_timer = jnp.maximum(0, enemy_state_timer - 1)
        should_fire = False

        # Spawn/waiting patrol (do not aggressively approach while waiting)
        waiting = new_state_timer > 0
        patrol_condition = jnp.logical_and(should_process, waiting)
        patrol_turn_speed = ENEMY_TURN_SPEED * 0.4
        angle_diff_to_target = new_target_angle - enemy_angle
        angle_diff_to_target = jnp.arctan2(jnp.sin(angle_diff_to_target), jnp.cos(angle_diff_to_target))
        new_angle = jnp.where(jnp.logical_and(patrol_condition, jnp.abs(angle_diff_to_target) > 0.1),
                              enemy_angle + jnp.sign(angle_diff_to_target) * patrol_turn_speed,
                              new_angle)
        new_x = jnp.where(patrol_condition, enemy_x + jnp.cos(new_angle) * ENEMY_MOVE_SPEED * 0.5, new_x)
        new_y = jnp.where(patrol_condition, enemy_y + jnp.sin(new_angle) * ENEMY_MOVE_SPEED * 0.5, new_y)
        new_state = jnp.where(patrol_condition, ENEMY_STATE_PATROL, new_state)

        # HUNT: instantly face the player and move directly toward them (classic 2600 behaviour)
        hunt_condition = jnp.logical_and(should_process, distance_to_player < ENEMY_DETECTION_RANGE)
        # Instant facing
        new_angle = jnp.where(hunt_condition, angle_to_player, new_angle)
        # Move along direct vector so they reliably approach and get larger on screen
        move_allowed = jnp.where(distance_to_player > ENEMY_MIN_DISTANCE, 1.0, 0.0)
        new_x = jnp.where(hunt_condition, enemy_x + dir_x * ENEMY_MOVE_SPEED * move_allowed, new_x)
        new_y = jnp.where(hunt_condition, enemy_y + dir_y * ENEMY_MOVE_SPEED * move_allowed, new_y)
        new_state = jnp.where(hunt_condition, ENEMY_STATE_HUNT, new_state)

        # ENGAGE: if within engagement band, stop big advances and allow small adjustments; since we face instantly,
        # alignment is immediate and firing will be more responsive (like the original)
        engage_condition = jnp.logical_and(should_process,
                                           jnp.logical_and(distance_to_player <= ENEMY_MAX_DISTANCE,
                                                           distance_to_player >= ENEMY_MIN_DISTANCE))
        # Small corrections (soft)
        too_close = distance_to_player < ENEMY_OPTIMAL_DISTANCE
        too_far = distance_to_player > ENEMY_OPTIMAL_DISTANCE
        new_x = jnp.where(jnp.logical_and(engage_condition, too_close), new_x - dir_x * ENEMY_MOVE_SPEED * 0.12, new_x)
        new_y = jnp.where(jnp.logical_and(engage_condition, too_close), new_y - dir_y * ENEMY_MOVE_SPEED * 0.12, new_y)
        new_x = jnp.where(jnp.logical_and(engage_condition, too_far), new_x + dir_x * ENEMY_MOVE_SPEED * 0.18, new_x)
        new_y = jnp.where(jnp.logical_and(engage_condition, too_far), new_y + dir_y * ENEMY_MOVE_SPEED * 0.18, new_y)

        # With instant facing, frontal alignment is true (or nearly true); use a small angular tolerance in case
        angle_diff = angle_to_player - new_angle
        angle_diff = jnp.arctan2(jnp.sin(angle_diff), jnp.cos(angle_diff))
        is_frontal = jnp.abs(angle_diff) < 0.35

        # Fire when frontal, in range and cooldown ready; respect spawn waiting via cooldown/state_timer
        can_fire = enemy_cooldown <= 0
        should_fire = jnp.logical_and(jnp.logical_and(engage_condition, is_frontal),
                                      jnp.logical_and(can_fire, distance_to_player < ENEMY_FIRE_RANGE))
        new_state = jnp.where(engage_condition, ENEMY_STATE_ENGAGE, new_state)
        new_cooldown = jnp.where(should_fire, ENEMY_FIRE_COOLDOWN, jnp.maximum(0, enemy_cooldown - 1))

        # Bullet spawn/aim toward player's current position
        bullet_offset = 15.0
        spawn_bx = new_x + jnp.cos(new_angle) * bullet_offset
        spawn_by = new_y + jnp.sin(new_angle) * bullet_offset
        aim_dx = player_tank.x - new_x
        aim_dy = player_tank.y - new_y
        aim_dist = jnp.sqrt(aim_dx * aim_dx + aim_dy * aim_dy) + 1e-6
        aim_vx = (aim_dx / aim_dist) * BULLET_SPEED
        aim_vy = (aim_dy / aim_dist) * BULLET_SPEED

        # Finalize
        new_x = jnp.clip(new_x, BOUNDARY_MIN, BOUNDARY_MAX)
        new_y = jnp.clip(new_y, BOUNDARY_MIN, BOUNDARY_MAX)
        new_angle = jnp.arctan2(jnp.sin(new_angle), jnp.cos(new_angle))

        return (new_x, new_y, new_angle, enemy_alive, new_cooldown, new_state,
                new_target_angle, new_state_timer, should_fire, spawn_bx, spawn_by, aim_vx, aim_vy)

    results = jax.vmap(update_single_enemy)(jnp.arange(len(obstacles.x)))
    (new_x, new_y, new_angle, new_alive, new_cooldown, new_state,
     new_target_angle, new_state_timer, fire_flags, bullet_xs, bullet_ys, bullet_vel_xs, bullet_vel_ys) = results

    updated_obstacles = Obstacle(
        x=new_x,
        y=new_y,
        obstacle_type=obstacles.obstacle_type,
        angle=new_angle,
        alive=new_alive,
        fire_cooldown=new_cooldown,
        ai_state=new_state,
        target_angle=new_target_angle,
        state_timer=new_state_timer
    )
    
    # Handle enemy bullets using JAX-compatible operations
    updated_bullets = bullets
    enemy_bullet_start = PLAYER_BULLET_LIMIT
    
    # Use a more JAX-friendly approach for bullet creation
    def add_enemy_bullets(bullets_state):
        # Find available slots for enemy bullets
        enemy_slots = jnp.arange(enemy_bullet_start, MAX_BULLETS)
        available_slots = jnp.logical_not(bullets_state.active[enemy_bullet_start:MAX_BULLETS])
        
        # For each enemy that wants to fire, try to add a bullet
        for i in range(len(fire_flags)):
            # Find first available slot for this enemy
            slot_mask = jnp.logical_and(fire_flags[i], available_slots)
            
            # If any slot is available, use the first one
            slot_available = jnp.any(slot_mask)
            first_slot_idx = jnp.argmax(slot_mask)
            actual_slot = enemy_bullet_start + first_slot_idx
            
            # Update the bullets array conditionally
            bullets_state = jax.lax.cond(
                slot_available,
                lambda b: Bullet(
                    x=b.x.at[actual_slot].set(bullet_xs[i]),
                    y=b.y.at[actual_slot].set(bullet_ys[i]),
                    z=b.z.at[actual_slot].set(3.0),
                    vel_x=b.vel_x.at[actual_slot].set(bullet_vel_xs[i]),
                    vel_y=b.vel_y.at[actual_slot].set(bullet_vel_ys[i]),
                    active=b.active.at[actual_slot].set(1),
                    lifetime=b.lifetime.at[actual_slot].set(BULLET_LIFETIME),
                    owner=b.owner.at[actual_slot].set(i + 1)
                ),
                lambda b: b,
                bullets_state
            )
            
            # Mark this slot as used
            available_slots = available_slots.at[first_slot_idx].set(
                jnp.logical_and(available_slots[first_slot_idx], jnp.logical_not(slot_available))
            )
        
        return bullets_state
    
    updated_bullets = add_enemy_bullets(updated_bullets)
    
    return updated_obstacles, updated_bullets

@jax.jit
def spawn_new_enemy(obstacles: Obstacle, player_tank: Tank, step_counter: chex.Array) -> Obstacle:
	"""Spawn a new enemy in front of the player (on the horizon) so it approaches the player."""
	
	# Count active enemies
	active_enemies = jnp.sum(obstacles.alive)
	
	# Only spawn if we have room for more enemies
	should_spawn = active_enemies < MAX_ACTIVE_ENEMIES
	
	# Find first dead enemy slot for respawning
	dead_enemy_idx = jnp.argmax(jnp.logical_not(obstacles.alive))
	has_dead_slot = jnp.logical_not(obstacles.alive[dead_enemy_idx])
	
	can_spawn = jnp.logical_and(should_spawn, has_dead_slot)

	# Spawn in front of player (horizon): use player's facing direction plus small deterministic spread
	# deterministic offset based on step_counter so spawns are varied but repeatable
	spread = (jnp.sin(step_counter * 0.13) * 0.9)  # radians offset in [-0.9,0.9]
	spawn_angle = player_tank.angle + spread

	# spawn at the maximum spawn distance in front of the player (the horizon area)
	spawn_distance = ENEMY_SPAWN_DISTANCE_MAX

	spawn_x = player_tank.x + jnp.cos(spawn_angle) * spawn_distance
	spawn_y = player_tank.y + jnp.sin(spawn_angle) * spawn_distance

	# Clamp spawn position to world bounds
	spawn_x = jnp.clip(spawn_x, BOUNDARY_MIN, BOUNDARY_MAX)
	spawn_y = jnp.clip(spawn_y, BOUNDARY_MIN, BOUNDARY_MAX)

	# Initial enemy angle: roughly face towards the player so they move inward
	initial_angle = jnp.arctan2(player_tank.y - spawn_y, player_tank.x - spawn_x)

	# Write into the dead slot and set to HUNT so it will approach the player; give short wait before shooting
	new_x = jnp.where(can_spawn, obstacles.x.at[dead_enemy_idx].set(spawn_x), obstacles.x)
	new_y = jnp.where(can_spawn, obstacles.y.at[dead_enemy_idx].set(spawn_y), obstacles.y)
	new_angle = jnp.where(can_spawn, obstacles.angle.at[dead_enemy_idx].set(initial_angle), obstacles.angle)
	new_alive = jnp.where(can_spawn, obstacles.alive.at[dead_enemy_idx].set(1), obstacles.alive)
	new_fire_cooldown = jnp.where(can_spawn, obstacles.fire_cooldown.at[dead_enemy_idx].set(ENEMY_SPAWN_WAIT), obstacles.fire_cooldown)
	# Immediately set AI to HUNT so it will approach the player (but it will wait to shoot until state_timer expires)
	new_ai_state = jnp.where(can_spawn, obstacles.ai_state.at[dead_enemy_idx].set(ENEMY_STATE_HUNT), obstacles.ai_state)
	new_target_angle = jnp.where(can_spawn, obstacles.target_angle.at[dead_enemy_idx].set(initial_angle), obstacles.target_angle)
	new_state_timer = jnp.where(can_spawn, obstacles.state_timer.at[dead_enemy_idx].set(ENEMY_SPAWN_WAIT), obstacles.state_timer)

	return Obstacle(
		x=new_x,
		y=new_y,
		obstacle_type=obstacles.obstacle_type,
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
        # compute flattened observation size so obs_to_flat_array is consistent with the representation used by wrappers/tests
        # layout: player (x,y,angle,alive) = 4
        # bullets: MAX_BULLETS * (x,y,active,owner) = MAX_BULLETS*4
        # obstacles: MAX_OBSTACLES * (x,y,angle,type,alive) = MAX_OBSTACLES*5
        self.obs_size = 4 + MAX_BULLETS * 4 + MAX_OBSTACLES * 5

        # Create renderer instance for env-level rendering (pygame Surface drawing does not require display mode)
        try:
            self.renderer = BattleZoneRenderer()
        except Exception:
            self.renderer = None

        # Gym-style spaces used by wrappers/tests
        # Observation is the flattened vector defined by obs_to_flat_array
        self.observation_space = spaces.Box(
            low=-np.finfo(np.float32).max,
            high=np.finfo(np.float32).max,
            shape=(self.obs_size,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(self.action_set))

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
            player_lives=jnp.array(3)    # initialize lives (3 as example)
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
                    enemy_color = (255, 0, 0)
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

        # Draw score centered
        if self.hud_font is not None:
            score_surf = self.hud_font.render(f"{score_val:03d}", True, WIREFRAME_COLOR)
            score_rect = score_surf.get_rect(center=(WIDTH // 2, HEIGHT - bottom_bar_height // 2))
            screen.blit(score_surf, score_rect)

        # Draw simple life icons (small green rectangles) left of the score
        life_start_x = WIDTH // 2 - 60
        life_y = HEIGHT - bottom_bar_height // 2
        life_spacing = 14
        for i in range(max(0, lives_val)):
            lx = int(life_start_x + i * life_spacing)
            ly = int(life_y - 6)
            pygame.draw.rect(screen, WIREFRAME_COLOR, (lx, ly, 10, 12), 1)  # outline tank icon

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
                    self.draw_enemy_tank(screen, screen_x, screen_y, distance, WIREFRAME_COLOR, 
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