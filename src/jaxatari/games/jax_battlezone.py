
# jax_battlezone.py  — pygame-frei & jax.jit-kompatibel (patched to avoid tracer hashing in indexing)
from functools import partial
import chex
import jax
import jax.numpy as jnp
import jax.random as jrandom
from typing import Tuple, NamedTuple
import math
import numpy as np

from jaxatari.environment import JaxEnvironment, EnvState
from jaxatari.spaces import Space
import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer

# --- Actions (unverändert) ---
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

# --- Spielkonstanten (unverändert) ---
WIDTH = 160
HEIGHT = 210
MAX_BULLETS = 16
PLAYER_BULLET_LIMIT = 8
ENEMY_BULLET_LIMIT = 8
MAX_OBSTACLES = 16

# Enemy AI constants (wie gehabt)
ENEMY_DETECTION_RANGE = 500.0
ENEMY_MOVE_SPEED = 0.35
ENEMY_TURN_SPEED = 0.008
ENEMY_FIRE_COOLDOWN = 180
ENEMY_FIRE_RANGE = 400.0
ENEMY_OPTIMAL_DISTANCE = 120.0
ENEMY_MIN_DISTANCE = 60.0
ENEMY_MAX_DISTANCE = 200.0

ENEMY_STATE_PATROL = 0
ENEMY_STATE_HUNT = 1
ENEMY_STATE_ENGAGE = 2
ENEMY_STATE_RETREAT = 3

# Tank movement
TANK_SPEED = 0.2
TANK_TURN_SPEED = 0.008
BULLET_SPEED = 1.0
BULLET_LIFETIME = 120

# Spawning
ENEMY_SPAWN_DISTANCE_MIN = 80.0
ENEMY_SPAWN_DISTANCE_MAX = 200.0
MAX_ACTIVE_ENEMIES = 4
ENEMY_SPAWN_COOLDOWN = 300
ENEMY_SPAWN_WAIT = 120

# Types
ENEMY_TYPE_TANK = 0
ENEMY_TYPE_SUPERTANK = 1
ENEMY_TYPE_FIGHTER = 2
ENEMY_TYPE_SAUCER = 3

SLOW_TANK_SPEED_FACTOR = 0.85
SUPERTANK_SPEED_FACTOR = 1.25
FIGHTER_SPEED_FACTOR = 1.6
SAUCER_SPEED_FACTOR = 0.6

FPS = 60.0

def _deg_per_sec_to_rad_per_frame(deg_per_sec: float) -> float:
    return (deg_per_sec * (math.pi / 180.0)) / FPS

ENEMY_SPEED_MULTIPLIERS = jnp.array([
    SLOW_TANK_SPEED_FACTOR, SUPERTANK_SPEED_FACTOR, FIGHTER_SPEED_FACTOR, SAUCER_SPEED_FACTOR
], dtype=jnp.float32)

ENEMY_FIRING_ANGLE_DEG = jnp.array([3.0, 3.0, 3.0, 3.0], dtype=jnp.float32)
ENEMY_FIRING_ANGLE_THRESH_RAD = (ENEMY_FIRING_ANGLE_DEG * (math.pi / 180.0))
ENEMY_FIRING_RANGE = jnp.array([30.0, 30.0, 15.0, 0.0], dtype=jnp.float32)
ENEMY_NO_FIRE_AFTER_SPAWN_FRAMES = jnp.array([int(2.0 * FPS), int(2.0 * FPS), int(0.5 * FPS), int(2.0 * FPS)], dtype=jnp.int32)

ENEMY_TURN_RATES = jnp.array([
    _deg_per_sec_to_rad_per_frame(120.0),
    _deg_per_sec_to_rad_per_frame(200.0),
    _deg_per_sec_to_rad_per_frame(380.0),
    _deg_per_sec_to_rad_per_frame(60.0),
], dtype=jnp.float32)

ENEMY_FIRE_COOLDOWNS = jnp.array([int(1.2*FPS), int(0.8*FPS), int(1.0*FPS), 0], dtype=jnp.int32)
ENEMY_CAN_FIRE = jnp.array([1, 1, 1, 0], dtype=jnp.int32)

# Welt
WORLD_SIZE = 2000
BOUNDARY_MIN = -WORLD_SIZE // 2
BOUNDARY_MAX = WORLD_SIZE // 2

MAP_RADIUS = float(BOUNDARY_MAX - BOUNDARY_MIN) / 2.0
SPAWN_NEAR_RADIUS = MAP_RADIUS * 0.375
SPAWN_FAR_RADIUS = MAP_RADIUS * 0.75

ENEMY_PROJECTILE_SPEED = BULLET_SPEED * 0.9
ENEMY_PROJECTILE_LIFETIME = BULLET_LIFETIME

# Rendering-Farben (UI-Stil beibehalten)
HORIZON_Y = 105
GROUND_COLOR = jnp.array([0, 100, 0], dtype=jnp.uint8)
SKY_COLOR = jnp.array([0, 0, 0], dtype=jnp.uint8)
WIREFRAME_COLOR = jnp.array([0, 255, 0], dtype=jnp.uint8)
BULLET_COLOR = jnp.array([255, 255, 255], dtype=jnp.uint8)

TANK_COLOR = jnp.array([20, 110, 220], dtype=jnp.uint8)
SUPERTANK_COLOR = jnp.array([220, 200, 30], dtype=jnp.uint8)
FIGHTER_COLOR = jnp.array([220, 40, 40], dtype=jnp.uint8)
SAUCER_COLOR = jnp.array([240, 240, 240], dtype=jnp.uint8)
HUD_ACCENT_COLOR = jnp.array([47, 151, 119], dtype=jnp.uint8)

# --- Datenstrukturen ---
class Tank(NamedTuple):
    x: chex.Array
    y: chex.Array
    angle: chex.Array
    alive: chex.Array

class Bullet(NamedTuple):
    x: chex.Array
    y: chex.Array
    z: chex.Array
    vel_x: chex.Array
    vel_y: chex.Array
    active: chex.Array
    lifetime: chex.Array
    owner: chex.Array  # 0: Spieler, >0: Gegnerindex

class Obstacle(NamedTuple):
    x: chex.Array
    y: chex.Array
    obstacle_type: chex.Array
    enemy_subtype: chex.Array
    angle: chex.Array
    alive: chex.Array
    fire_cooldown: chex.Array
    ai_state: chex.Array
    target_angle: chex.Array
    state_timer: chex.Array

class BattleZoneState(NamedTuple):
    player_tank: Tank
    bullets: Bullet
    obstacles: Obstacle
    step_counter: chex.Array
    spawn_timer: chex.Array
    prev_player_x: chex.Array
    prev_player_y: chex.Array
    player_score: chex.Array
    player_lives: chex.Array

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
    player_shot: chex.Array

# --- Bewegungen & Logik (JAX-vektorisiert, unveränderte Mechanik) ---
@jax.jit
def update_tank_position(tank: Tank, action: chex.Array) -> Tank:
    move_speed = TANK_SPEED
    turn_speed = TANK_TURN_SPEED

    new_x = tank.x
    new_y = tank.y
    angle = tank.angle

    is_left = jnp.any(action == jnp.array([LEFT, LEFTFIRE]))
    angle = jnp.where(is_left, angle - turn_speed, angle)

    is_right = jnp.any(action == jnp.array([RIGHT, RIGHTFIRE]))
    angle = jnp.where(is_right, angle + turn_speed, angle)

    is_up = jnp.any(action == jnp.array([UP, UPFIRE]))
    new_x = jnp.where(is_up, new_x + jnp.cos(angle) * move_speed, new_x)
    new_y = jnp.where(is_up, new_y + jnp.sin(angle) * move_speed, new_y)

    is_down = jnp.any(action == jnp.array([DOWN, DOWNFIRE]))
    new_x = jnp.where(is_down, new_x - jnp.cos(angle) * move_speed, new_x)
    new_y = jnp.where(is_down, new_y - jnp.sin(angle) * move_speed, new_y)

    # Diagonalen
    upr = jnp.any(action == jnp.array([UPRIGHT, UPRIGHTFIRE]))
    ang_upr = angle + turn_speed
    angle = jnp.where(upr, ang_upr, angle)
    new_x = jnp.where(upr, new_x + jnp.cos(ang_upr) * move_speed, new_x)
    new_y = jnp.where(upr, new_y + jnp.sin(ang_upr) * move_speed, new_y)

    upl = jnp.any(action == jnp.array([UPLEFT, UPLEFTFIRE]))
    ang_upl = angle - turn_speed
    angle = jnp.where(upl, ang_upl, angle)
    new_x = jnp.where(upl, new_x + jnp.cos(ang_upl) * move_speed, new_x)
    new_y = jnp.where(upl, new_y + jnp.sin(ang_upl) * move_speed, new_y)

    dnr = jnp.any(action == jnp.array([DOWNRIGHT, DOWNRIGHTFIRE]))
    ang_dnr = angle + turn_speed
    angle = jnp.where(dnr, ang_dnr, angle)
    new_x = jnp.where(dnr, new_x - jnp.cos(ang_dnr) * move_speed, new_x)
    new_y = jnp.where(dnr, new_y - jnp.sin(ang_dnr) * move_speed, new_y)

    dnl = jnp.any(action == jnp.array([DOWNLEFT, DOWNLEFTFIRE]))
    ang_dnl = angle - turn_speed
    angle = jnp.where(dnl, ang_dnl, angle)
    new_x = jnp.where(dnl, new_x - jnp.cos(ang_dnl) * move_speed, new_x)
    new_y = jnp.where(dnl, new_y - jnp.sin(ang_dnl) * move_speed, new_y)

    angle = jnp.arctan2(jnp.sin(angle), jnp.cos(angle))
    new_x = jnp.clip(new_x, BOUNDARY_MIN, BOUNDARY_MAX)
    new_y = jnp.clip(new_y, BOUNDARY_MIN, BOUNDARY_MAX)
    return Tank(x=new_x, y=new_y, angle=angle, alive=tank.alive)

@jax.jit
def should_fire(action: chex.Array) -> chex.Array:
    fire_actions = jnp.array([FIRE, UPFIRE, RIGHTFIRE, LEFTFIRE, DOWNFIRE,
                              UPRIGHTFIRE, UPLEFTFIRE, DOWNRIGHTFIRE, DOWNLEFTFIRE])
    return jnp.any(action == fire_actions)

@jax.jit
def create_bullet(tank: Tank, owner: chex.Array) -> Bullet:
    angle = tank.angle
    offset = 0.1
    vel_x = jnp.cos(angle) * BULLET_SPEED
    vel_y = jnp.sin(angle) * BULLET_SPEED
    spawn_x = tank.x + jnp.cos(angle) * offset
    spawn_y = tank.y + jnp.sin(angle) * offset
    return Bullet(
        x=spawn_x, y=spawn_y, z=jnp.array(3.0),
        vel_x=vel_x, vel_y=vel_y,
        active=jnp.array(1, dtype=jnp.int32),
        lifetime=jnp.array(BULLET_LIFETIME),
        owner=owner
    )

@jax.jit
def update_bullets(bullets: Bullet) -> Bullet:
    new_x = bullets.x + bullets.vel_x
    new_y = bullets.y + bullets.vel_y
    new_lifetime = bullets.lifetime - 1
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
        x=new_x, y=new_y, z=bullets.z,
        vel_x=bullets.vel_x, vel_y=bullets.vel_y,
        active=active, lifetime=new_lifetime, owner=bullets.owner
    )

@jax.jit
def check_bullet_obstacle_collisions(bullets: Bullet, obstacles: Obstacle) -> Tuple[Bullet, Obstacle, chex.Array]:
    collision_radius = 15.0
    bx = bullets.x[:, None]
    by = bullets.y[:, None]
    bactive = bullets.active[:, None]
    ox = obstacles.x[None, :]
    oy = obstacles.y[None, :]
    oalive = obstacles.alive[None, :]
    dx = bx - ox
    dy = by - oy
    distances = jnp.sqrt(dx * dx + dy * dy)
    collisions = jnp.logical_and(jnp.logical_and(bactive, oalive), distances < collision_radius)
    bullets_to_remove = jnp.any(collisions, axis=1)
    obstacles_to_remove = jnp.any(collisions, axis=0)

    player_mask = (bullets.owner[:, None] == 0)
    collisions_by_player = jnp.logical_and(collisions, player_mask)
    killed = jnp.any(collisions_by_player, axis=0)

    points_map = jnp.array([1000, 3000, 2000, 5000], dtype=jnp.int32)
    enemy_sub = getattr(obstacles, 'enemy_subtype', jnp.full_like(obstacles.x, -1))
    subtype_idx = jnp.clip(enemy_sub, 0, points_map.shape[0]-1).astype(jnp.int32)
    points_per_obstacle = points_map[subtype_idx] * killed.astype(jnp.int32)
    score_delta = jnp.sum(points_per_obstacle).astype(jnp.int32)

    new_b_active = jnp.where(bullets_to_remove, jnp.zeros_like(bullets.active), bullets.active)
    new_o_alive = jnp.where(obstacles_to_remove, 0, obstacles.alive)

    updated_bullets = Bullet(
        x=bullets.x, y=bullets.y, z=bullets.z,
        vel_x=bullets.vel_x, vel_y=bullets.vel_y,
        active=new_b_active, lifetime=bullets.lifetime, owner=bullets.owner
    )
    updated_obstacles = Obstacle(
        x=obstacles.x, y=obstacles.y, obstacle_type=obstacles.obstacle_type,
        enemy_subtype=getattr(obstacles, 'enemy_subtype', jnp.full_like(obstacles.x, -1)),
        angle=obstacles.angle, alive=new_o_alive, fire_cooldown=obstacles.fire_cooldown,
        ai_state=obstacles.ai_state, target_angle=obstacles.target_angle, state_timer=obstacles.state_timer
    )
    return updated_bullets, updated_obstacles, score_delta

@jax.jit
def check_player_hit(player_tank: Tank, bullets: Bullet) -> Tank:
    enemy_bullets = jnp.logical_and(bullets.active, bullets.owner > 0)
    dx = bullets.x - player_tank.x
    dy = bullets.y - player_tank.y
    distances = jnp.sqrt(dx * dx + dy * dy)
    hits = jnp.logical_and(enemy_bullets, distances < 10.0)
    player_hit = jnp.any(hits)
    new_alive = jnp.where(player_hit, 0, player_tank.alive)
    return Tank(x=player_tank.x, y=player_tank.y, angle=player_tank.angle, alive=new_alive)

# --- Gegner-Update & Spawns (JAX) ---
@jax.jit
def update_enemy_tanks(obstacles: Obstacle, player_tank: Tank, bullets: Bullet, step_counter: chex.Array) -> Tuple[Obstacle, Bullet]:
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

        dx = player_tank.x - enemy_x
        dy = player_tank.y - enemy_y
        distance = jnp.sqrt(dx * dx + dy * dy) + 1e-8
        dir_x = dx / distance
        dir_y = dy / distance
        angle_to_player = jnp.arctan2(dy, dx)

        new_state_timer = jnp.maximum(0, enemy_state_timer - 1)
        waiting = new_state_timer > 0

        new_x = enemy_x
        new_y = enemy_y
        new_angle = enemy_angle
        new_state = enemy_state
        new_target_angle = enemy_target_angle
        new_cooldown = jnp.maximum(0, enemy_cooldown - 1)
        fire_flag = jnp.array(False)

        too_close = distance < ENEMY_MIN_DISTANCE
        too_far = distance > ENEMY_MAX_DISTANCE
        in_engage_band = jnp.logical_and(distance <= ENEMY_MAX_DISTANCE, distance >= ENEMY_MIN_DISTANCE)

        move_forward = jnp.logical_and(should_process, jnp.logical_or(too_far, jnp.logical_and(in_engage_band, distance > ENEMY_OPTIMAL_DISTANCE)))
        move_backward = jnp.logical_and(should_process, too_close)

        subtype = obstacles.enemy_subtype[i]
        is_tank_subtype = jnp.logical_or(subtype == ENEMY_TYPE_TANK, subtype == ENEMY_TYPE_SUPERTANK)

        # Non-tank movement (vereinfachte Direktbewegung)
        new_x = jnp.where(jnp.logical_and(move_forward, jnp.logical_not(is_tank_subtype)), enemy_x + dir_x * ENEMY_MOVE_SPEED, new_x)
        new_y = jnp.where(jnp.logical_and(move_forward, jnp.logical_not(is_tank_subtype)), enemy_y + dir_y * ENEMY_MOVE_SPEED, new_y)
        new_x = jnp.where(jnp.logical_and(move_backward, jnp.logical_not(is_tank_subtype)), enemy_x - dir_x * ENEMY_MOVE_SPEED, new_x)
        new_y = jnp.where(jnp.logical_and(move_backward, jnp.logical_not(is_tank_subtype)), enemy_y - dir_y * ENEMY_MOVE_SPEED, new_y)

        # Tank-artiges Steuern
        turn_rate = jnp.where(subtype == ENEMY_TYPE_SUPERTANK, ENEMY_TURN_RATES[ENEMY_TYPE_SUPERTANK], ENEMY_TURN_RATES[ENEMY_TYPE_TANK])
        speed_multiplier = jnp.where(subtype == ENEMY_TYPE_SUPERTANK, ENEMY_SPEED_MULTIPLIERS[ENEMY_TYPE_SUPERTANK], ENEMY_SPEED_MULTIPLIERS[ENEMY_TYPE_TANK])

        desired_heading = angle_to_player
        raw_delta = desired_heading - new_angle
        delta_angle = jnp.arctan2(jnp.sin(raw_delta), jnp.cos(raw_delta))
        turn_amount = jnp.clip(delta_angle, -turn_rate, turn_rate)
        apply_steer = jnp.logical_and(should_process, is_tank_subtype)
        steered_angle = new_angle + jnp.where(apply_steer, turn_amount, 0.0)
        new_angle = jnp.where(apply_steer, steered_angle, new_angle)

        base_move_speed = ENEMY_MOVE_SPEED * speed_multiplier
        move_angle = jnp.where(apply_steer, steered_angle, new_angle)
        phase = jnp.sin(step_counter * 0.08 + i * 0.3)
        move_phase = phase > 0.0
        effective_mult = jnp.where(jnp.logical_and(in_engage_band, distance <= ENEMY_FIRING_RANGE[jnp.clip(subtype,0,3)]), 0.0, jnp.where(move_phase, 1.0, 0.0))
        tank_next_x = new_x + jnp.cos(move_angle) * base_move_speed * effective_mult
        tank_next_y = new_y + jnp.sin(move_angle) * base_move_speed * effective_mult
        new_x = jnp.where(apply_steer, tank_next_x, new_x)
        new_y = jnp.where(apply_steer, tank_next_y, new_y)

        # Feuerlogik
        subtype_idx = jnp.clip(subtype, 0, 3).astype(jnp.int32)
        time_since_spawn = ENEMY_SPAWN_WAIT - new_state_timer
        allowed_by_spawn = time_since_spawn > ENEMY_NO_FIRE_AFTER_SPAWN_FRAMES[subtype_idx]
        ang_diff = jnp.abs(jnp.arctan2(jnp.sin(new_angle - angle_to_player), jnp.cos(new_angle - angle_to_player)))
        angle_ok = ang_diff <= ENEMY_FIRING_ANGLE_THRESH_RAD[subtype_idx]
        range_ok = distance <= ENEMY_FIRING_RANGE[subtype_idx]
        cooldown_ok = new_cooldown <= 0
        type_can_fire = ENEMY_CAN_FIRE[subtype_idx] == 1
        fire_flag = jnp.logical_and.reduce((type_can_fire, cooldown_ok, allowed_by_spawn, angle_ok, range_ok))

        new_state = jnp.where(in_engage_band, ENEMY_STATE_ENGAGE, new_state)
        new_cooldown = jnp.where(fire_flag, ENEMY_FIRE_COOLDOWNS[subtype_idx], new_cooldown)

        # Bullet: Zielrichtung Spieler
        bullet_offset = 15.0
        spawn_bx = new_x + jnp.cos(new_angle) * bullet_offset
        spawn_by = new_y + jnp.sin(new_angle) * bullet_offset
        aim_dx = player_tank.x - new_x
        aim_dy = player_tank.y - new_y
        aim_dist = jnp.sqrt(aim_dx * aim_dx + aim_dy * aim_dy) + 1e-6
        aim_vx = (aim_dx / aim_dist) * BULLET_SPEED
        aim_vy = (aim_dy / aim_dist) * BULLET_SPEED

        # begrenzen & normalisieren
        new_x = jnp.clip(new_x, BOUNDARY_MIN, BOUNDARY_MAX)
        new_y = jnp.clip(new_y, BOUNDARY_MIN, BOUNDARY_MAX)
        new_angle = jnp.arctan2(jnp.sin(new_angle), jnp.cos(new_angle))

        return (new_x, new_y, new_angle, enemy_alive, new_cooldown, new_state,
                new_target_angle, new_state_timer, fire_flag, spawn_bx, spawn_by, aim_vx, aim_vy)

    n = obstacles.x.shape[0]
    idxs = jnp.arange(n)
    results = jax.vmap(update_single_enemy)(idxs)

    (new_x, new_y, new_angle, new_alive, new_cooldown, new_state,
     new_target_angle, new_state_timer, fire_flags, bx, by, bvx, bvy) = results

    updated_obstacles = Obstacle(
        x=new_x, y=new_y, obstacle_type=obstacles.obstacle_type,
        enemy_subtype=getattr(obstacles, 'enemy_subtype', jnp.full_like(obstacles.x, -1)),
        angle=new_angle, alive=new_alive, fire_cooldown=new_cooldown,
        ai_state=new_state, target_angle=new_target_angle, state_timer=new_state_timer
    )

    updated_bullets = bullets
    enemy_bullet_start = PLAYER_BULLET_LIMIT

    def add_enemy_bullets_loop(bullets_state):
        def body(i, b):
            slot = enemy_bullet_start + (i % ENEMY_BULLET_LIMIT)
            cond = fire_flags[i]
            def set_b(bb):
                return Bullet(
                    x=bb.x.at[slot].set(bx[i]),
                    y=bb.y.at[slot].set(by[i]),
                    z=bb.z.at[slot].set(3.0),
                    vel_x=bb.vel_x.at[slot].set(bvx[i]),
                    vel_y=bb.vel_y.at[slot].set(bvy[i]),
                    active=bb.active.at[slot].set(1),
                    lifetime=bb.lifetime.at[slot].set(BULLET_LIFETIME),
                    owner=bb.owner.at[slot].set(i + 1)
                )
            return jax.lax.cond(cond, set_b, lambda bb: bb, b)
        return jax.lax.fori_loop(0, n, body, bullets_state)

    updated_bullets = add_enemy_bullets_loop(updated_bullets)
    return updated_obstacles, updated_bullets

@jax.jit
def spawn_new_enemy(obstacles: Obstacle, player_tank: Tank, step_counter: chex.Array) -> Obstacle:
    active_enemies = jnp.sum(obstacles.alive)
    hostile_mask = jnp.logical_and(obstacles.obstacle_type == 0,
                                   jnp.logical_and(obstacles.alive == 1, obstacles.enemy_subtype != ENEMY_TYPE_SAUCER))
    hostile_active = jnp.any(hostile_mask)

    should_spawn = active_enemies < MAX_ACTIVE_ENEMIES
    dead_enemy_idx = jnp.argmax(jnp.logical_not(obstacles.alive))
    has_dead_slot = jnp.logical_not(obstacles.alive[dead_enemy_idx])
    can_spawn = jnp.logical_and(should_spawn, has_dead_slot)

    r1 = jnp.abs(jnp.sin(step_counter * 0.093 + dead_enemy_idx * 0.37))
    r2 = jnp.abs(jnp.cos(step_counter * 0.127 + dead_enemy_idx * 0.19))

    def choose_hostile():
        v = r1
        return jnp.where(v < 0.45, ENEMY_TYPE_TANK,
                         jnp.where(v < 0.90, ENEMY_TYPE_SUPERTANK, ENEMY_TYPE_FIGHTER))
    chosen_subtype = jax.lax.cond(hostile_active, lambda: ENEMY_TYPE_SAUCER, choose_hostile)

    angle_noise = (r1 * 2.0 * math.pi + r2 * 1.3) % (2.0 * math.pi)
    near_choice = (jnp.floor(r2 * 2.0) % 2) == 0

    spawn_angle = angle_noise
    spawn_distance = ENEMY_SPAWN_DISTANCE_MAX

    is_tank_like = jnp.logical_or(chosen_subtype == ENEMY_TYPE_TANK, chosen_subtype == ENEMY_TYPE_SUPERTANK)
    spawn_angle = jnp.where(is_tank_like, angle_noise, spawn_angle)
    spawn_distance = jnp.where(is_tank_like, jnp.where(near_choice, SPAWN_NEAR_RADIUS, SPAWN_FAR_RADIUS), spawn_distance)

    fighter_offset = SPAWN_NEAR_RADIUS * 0.5
    spawn_angle = jnp.where(chosen_subtype == ENEMY_TYPE_FIGHTER, player_tank.angle, spawn_angle)
    spawn_distance = jnp.where(chosen_subtype == ENEMY_TYPE_FIGHTER, fighter_offset, spawn_distance)

    saucer_radius = jnp.abs(jnp.sin(step_counter * 0.19 + dead_enemy_idx * 0.11)) * MAP_RADIUS
    spawn_angle = jnp.where(chosen_subtype == ENEMY_TYPE_SAUCER, angle_noise, spawn_angle)
    spawn_distance = jnp.where(chosen_subtype == ENEMY_TYPE_SAUCER, saucer_radius, spawn_distance)

    spawn_distance = jnp.clip(spawn_distance, ENEMY_SPAWN_DISTANCE_MIN, ENEMY_SPAWN_DISTANCE_MAX)
    spawn_x = player_tank.x + jnp.cos(spawn_angle) * spawn_distance
    spawn_y = player_tank.y + jnp.sin(spawn_angle) * spawn_distance

    spawn_x = jnp.clip(spawn_x, BOUNDARY_MIN, BOUNDARY_MAX)
    spawn_y = jnp.clip(spawn_y, BOUNDARY_MIN, BOUNDARY_MAX)

    initial_angle = jnp.arctan2(player_tank.y - spawn_y, player_tank.x - spawn_x)

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

    return Obstacle(
        x=new_x, y=new_y, obstacle_type=new_obstacle_type, enemy_subtype=new_enemy_subtype,
        angle=new_angle, alive=new_alive, fire_cooldown=new_fire_cooldown,
        ai_state=new_ai_state, target_angle=new_target_angle, state_timer=new_state_timer
    )

class JaxBattleZone(JaxEnvironment[BattleZoneState, BattleZoneObservation, chex.Array, BattleZoneInfo]):
    def __init__(self, reward_funcs: list[callable] = None):
        super().__init__()
        self.frame_stack_size = 4
        self.reward_funcs = tuple(reward_funcs) if reward_funcs is not None else None
        self.action_set = list(range(18))
        self.obs_size = 50
        # KEIN pygame: Dummy-Renderer liefert nur das Signal "es gibt einen Renderer"
        self.renderer = BattleZoneRenderer(JAXGameRenderer)

    def image_space(self) -> Space:
        return spaces.Box(low=0, high=255, shape=(210, 160, 3), dtype=jnp.uint8)

    def reset(self, key: jrandom.PRNGKey = None) -> Tuple[BattleZoneObservation, BattleZoneState]:
        player_tank = Tank(x=jnp.array(0.0), y=jnp.array(0.0), angle=jnp.array(0.0), alive=jnp.array(1, dtype=jnp.int32))
        bullets = Bullet(
            x=jnp.zeros(MAX_BULLETS), y=jnp.zeros(MAX_BULLETS), z=jnp.zeros(MAX_BULLETS),
            vel_x=jnp.zeros(MAX_BULLETS), vel_y=jnp.zeros(MAX_BULLETS),
            active=jnp.zeros(MAX_BULLETS, dtype=jnp.int32),
            lifetime=jnp.zeros(MAX_BULLETS), owner=jnp.zeros(MAX_BULLETS)
        )
        enemy_positions_x = jnp.array([120.0, -120.0, 0.0, -60.0, 0.0, 0.0, 0.0, 0.0,
                                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        enemy_positions_y = jnp.array([0.0, 0.0, 120.0, -120.0, 0.0, 0.0, 0.0, 0.0,
                                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        enemy_types = jnp.zeros(16)
        enemy_subtypes = jnp.array([0] + [-1] * 15)
        enemy_angles = jnp.array([jnp.pi] + [0.0] * 15)
        enemy_alive = jnp.array([1] + [0] * 15, dtype=jnp.int32)
        enemy_fire_cooldown = jnp.zeros(16)
        enemy_ai_state = jnp.zeros(16, dtype=jnp.int32)
        enemy_target_angle = enemy_angles
        enemy_state_timer = jnp.zeros(16)

        obstacles = Obstacle(
            x=enemy_positions_x, y=enemy_positions_y, obstacle_type=enemy_types,
            enemy_subtype=enemy_subtypes, angle=enemy_angles, alive=enemy_alive,
            fire_cooldown=enemy_fire_cooldown, ai_state=enemy_ai_state,
            target_angle=enemy_target_angle, state_timer=enemy_state_timer
        )
        state = BattleZoneState(
            player_tank=player_tank, bullets=bullets, obstacles=obstacles,
            step_counter=jnp.array(0), spawn_timer=jnp.array(ENEMY_SPAWN_COOLDOWN),
            prev_player_x=player_tank.x, prev_player_y=player_tank.y,
            player_score=jnp.array(0), player_lives=jnp.array(5)
        )
        observation = self._get_observation(state)
        return observation, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: BattleZoneState, action) -> Tuple[BattleZoneObservation, BattleZoneState, float, bool, BattleZoneInfo]:
        new_player_tank = update_tank_position(state.player_tank, action)

        fire_b = should_fire(action)
        inactive_player_slots = jnp.logical_not(state.bullets.active[:PLAYER_BULLET_LIMIT])
        inactive_bullet_idx = jnp.argmax(inactive_player_slots)
        can_fire = inactive_player_slots[inactive_bullet_idx]
        new_bullet = create_bullet(new_player_tank, jnp.array(0))

        updated_bullets = jax.lax.cond(
            jnp.logical_and(fire_b, can_fire),
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
        updated_bullets = update_bullets(updated_bullets)

        updated_obstacles, updated_bullets = update_enemy_tanks(state.obstacles, new_player_tank, updated_bullets, state.step_counter)

        new_spawn_timer = state.spawn_timer - 1
        should_try_spawn = new_spawn_timer <= 0
        updated_obstacles = jax.lax.cond(
            should_try_spawn,
            lambda obs: spawn_new_enemy(obs, new_player_tank, state.step_counter),
            lambda obs: obs,
            updated_obstacles
        )
        new_spawn_timer = jnp.where(should_try_spawn, ENEMY_SPAWN_COOLDOWN, new_spawn_timer)

        updated_bullets, updated_obstacles, score_delta = check_bullet_obstacle_collisions(updated_bullets, updated_obstacles)
        new_player_tank2 = check_player_hit(new_player_tank, updated_bullets)

        player_was_shot = jnp.logical_and(state.player_tank.alive == 1, new_player_tank2.alive == 0)
        lives_after = state.player_lives - 1

        def build_reset_state():
            player_tank_rst = Tank(x=jnp.array(0.0), y=jnp.array(0.0), angle=jnp.array(0.0), alive=jnp.array(1, dtype=jnp.int32))
            bullets_rst = Bullet(
                x=jnp.zeros(MAX_BULLETS), y=jnp.zeros(MAX_BULLETS), z=jnp.zeros(MAX_BULLETS),
                vel_x=jnp.zeros(MAX_BULLETS), vel_y=jnp.zeros(MAX_BULLETS),
                active=jnp.zeros(MAX_BULLETS, dtype=jnp.int32),
                lifetime=jnp.zeros(MAX_BULLETS), owner=jnp.zeros(MAX_BULLETS)
            )
            enemy_positions_x = jnp.array([120.0, -120.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            enemy_positions_y = jnp.array([0.0, 0.0, 120.0, -120.0, 0.0, 0.0, 0.0, 0.0,
                                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            enemy_types = jnp.zeros(16)
            enemy_angles = jnp.array([jnp.pi] + [0.0] * 15)
            enemy_alive = jnp.array([1] + [0] * 15, dtype=jnp.int32)
            enemy_fire_cooldown = jnp.zeros(16)
            enemy_ai_state = jnp.zeros(16, dtype=jnp.int32)
            enemy_target_angle = enemy_angles
            enemy_state_timer = jnp.zeros(16)
            obstacles_rst = Obstacle(
                x=enemy_positions_x, y=enemy_positions_y, obstacle_type=enemy_types,
                enemy_subtype=jnp.array([0] + [-1] * 15), angle=enemy_angles, alive=enemy_alive,
                fire_cooldown=enemy_fire_cooldown, ai_state=enemy_ai_state,
                target_angle=enemy_target_angle, state_timer=enemy_state_timer
            )
            return BattleZoneState(
                player_tank=player_tank_rst, bullets=bullets_rst, obstacles=obstacles_rst,
                step_counter=jnp.array(0), spawn_timer=jnp.array(ENEMY_SPAWN_COOLDOWN),
                prev_player_x=player_tank_rst.x, prev_player_y=player_tank_rst.y,
                player_score=state.player_score, player_lives=lives_after
            )

        def keep_or_gameover():
            def return_reset():
                return build_reset_state()
            def return_gameover():
                return BattleZoneState(
                    player_tank=new_player_tank2, bullets=updated_bullets, obstacles=updated_obstacles,
                    step_counter=state.step_counter + 1, spawn_timer=new_spawn_timer,
                    prev_player_x=state.player_tank.x, prev_player_y=state.player_tank.y,
                    player_score=state.player_score + score_delta, player_lives=lives_after
                )
            return jax.lax.cond(lives_after > 0, return_reset, return_gameover)

        final_state = jax.lax.cond(
            player_was_shot,
            lambda: keep_or_gameover(),
            lambda: BattleZoneState(
                player_tank=new_player_tank2, bullets=updated_bullets, obstacles=updated_obstacles,
                step_counter=state.step_counter + 1, spawn_timer=new_spawn_timer,
                prev_player_x=state.player_tank.x, prev_player_y=state.player_tank.y,
                player_score=state.player_score + score_delta, player_lives=state.player_lives
            )
        )

        observation = self._get_observation(final_state)
        reward = self._get_env_reward(state, final_state)
        done = self._get_done(final_state)
        all_rewards = self._get_all_reward(state, final_state)
        info = self._get_info_full(final_state, all_rewards, player_was_shot)
        return observation, final_state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: BattleZoneState) -> BattleZoneObservation:
        return BattleZoneObservation(
            player_tank=state.player_tank, bullets=state.bullets, obstacles=state.obstacles,
            step_counter=state.step_counter, spawn_timer=state.spawn_timer,
            player_score=state.player_score, player_lives=state.player_lives,
        )

    def action_space(self) -> Space:
        return spaces.Discrete(len(self.action_set))

    def observation_space(self) -> Space:
        MAX_BULLETS_L = getattr(self, 'MAX_BULLETS', 32)
        MAX_OBSTACLES_L = getattr(self, 'MAX_OBSTACLES', 32)
        return spaces.Dict({
            "player": spaces.Dict({
                "x": spaces.Box(low=BOUNDARY_MIN, high=BOUNDARY_MAX, shape=(), dtype=np.float32),
                "y": spaces.Box(low=BOUNDARY_MIN, high=BOUNDARY_MAX, shape=(), dtype=np.float32),
                "angle": spaces.Box(low=-math.pi, high=math.pi, shape=(), dtype=np.float32),
                "alive": spaces.Box(low=0, high=1, shape=(), dtype=np.int32),
            }),
            "bullets": spaces.Dict({
                "x": spaces.Box(low=BOUNDARY_MIN, high=BOUNDARY_MAX, shape=(MAX_BULLETS_L,), dtype=np.float32),
                "y": spaces.Box(low=BOUNDARY_MIN, high=BOUNDARY_MAX, shape=(MAX_BULLETS_L,), dtype=np.float32),
                "z": spaces.Box(low=0.0, high=1000.0, shape=(MAX_BULLETS_L,), dtype=np.float32),
                "vel_x": spaces.Box(low=-1000.0, high=1000.0, shape=(MAX_BULLETS_L,), dtype=np.float32),
                "vel_y": spaces.Box(low=-1000.0, high=1000.0, shape=(MAX_BULLETS_L,), dtype=np.float32),
                "active": spaces.Box(low=0, high=1, shape=(MAX_BULLETS_L,), dtype=np.int32),
                "lifetime": spaces.Box(low=0, high=BULLET_LIFETIME, shape=(MAX_BULLETS_L,), dtype=np.int32),
                "owner": spaces.Box(low=0, high=MAX_OBSTACLES_L, shape=(MAX_BULLETS_L,), dtype=np.int32),
            })
        })

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: EnvState) -> jnp.ndarray:
        # Delegate to JIT-capable renderer
        return self.renderer.render(state)

    @partial(jax.jit, static_argnums=(0,))
    def _get_env_reward(self, previous_state: BattleZoneState, state: BattleZoneState) -> float:
        return 0.0

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(self, previous_state: BattleZoneState, state: BattleZoneState):
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array([reward_func(previous_state, state) for reward_func in self.reward_funcs])
        return rewards

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: BattleZoneState) -> bool:
        return state.player_tank.alive == 0

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: BattleZoneState) -> BattleZoneInfo:
        return BattleZoneInfo(time=state.step_counter, all_rewards=jnp.zeros(1), player_shot=jnp.array(0, dtype=jnp.int32))

    @partial(jax.jit, static_argnums=(0,))
    def _get_info_full(self, state: BattleZoneState, all_rewards: chex.Array = None, player_shot: chex.Array = None) -> BattleZoneInfo:
        if all_rewards is None:
            all_rewards = jnp.zeros(1)
        if player_shot is None:
            player_shot = jnp.array(0, dtype=jnp.int32)
        return BattleZoneInfo(time=state.step_counter, all_rewards=all_rewards, player_shot=player_shot)

class BattleZoneRenderer(JAXGameRenderer):

    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def _paint_mask(img: jnp.ndarray, mask: chex.Array, color: chex.Array) -> jnp.ndarray:
        # Ensure mask has shape (H,W,1)
        mask3 = mask[..., None] if mask.ndim == 2 else mask
        color3 = jnp.asarray(color).reshape((1,1,-1))
        return jnp.where(mask3, color3, img)

    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def _points_mask(xs: chex.Array, ys: chex.Array, valid: chex.Array) -> chex.Array:
        """Erzeuge einen (H,W)-Bool-Mask für Punkte xs/ys ohne .at[...] indexing.
        Nutzt reine Broadcast-Logik und vermeidet Hashing von Tracern.
        """
        xs = jnp.where(valid, xs, -1)
        ys = jnp.where(valid, ys, -1)
        # Raster
        grid_y = jnp.arange(HEIGHT)[:, None]
        grid_x = jnp.arange(WIDTH)[None, :]
        # Vergleiche gegen alle Punkte (broadcast)
        # shape: (H, W, N) -> any über N
        eq_y = (grid_y[..., None] == ys[None, None, :])
        eq_x = (grid_x[..., None] == xs[None, None, :])
        mask = jnp.any(jnp.logical_and(eq_y, eq_x), axis=-1)
        return mask

    # ----------------- JIT-Renderer -----------------
    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: EnvState) -> jnp.ndarray:
        """Reines JAX-Rendering: Himmel/Boden, Gegner & Kugeln als Punkte, Radar/HUD."""
        h, w = HEIGHT, WIDTH
        img = jnp.zeros((h, w, 3), dtype=jnp.uint8)

        # Himmel & Boden
        yy = jnp.arange(h)
        sky_mask = yy[:, None] < HORIZON_Y
        img = self._paint_mask(img, sky_mask, SKY_COLOR)
        img = self._paint_mask(img, jnp.logical_not(sky_mask), GROUND_COLOR)

        # einfache Orthoprojektion Welt->Bild (nur Demo, Logik bleibt unberührt)
        def world_to_screen(wx, wy, px, py):
            # relative Koords
            rx = wx - px
            ry = wy - py
            # simple Projektion
            sx = jnp.clip(jnp.floor((rx / 8.0) + (w/2)).astype(jnp.int32), 0, w-1)
            sy = jnp.clip(jnp.floor(HORIZON_Y + (ry / 16.0)).astype(jnp.int32), 0, h-1)
            dist = jnp.sqrt(rx*rx + ry*ry) + 1e-6
            return sx, sy, dist

        # Gegnerpunkte
        sx, sy, _ = world_to_screen(state.obstacles.x, state.obstacles.y, state.player_tank.x, state.player_tank.y)
        alive = (state.obstacles.alive == 1)
        enemy_mask = self._points_mask(sx, sy, alive)
        img = self._paint_mask(img, enemy_mask, WIREFRAME_COLOR)

        # Kugeln
        bx, by, _ = world_to_screen(state.bullets.x, state.bullets.y, state.player_tank.x, state.player_tank.y)
        bvalid = (state.bullets.active == 1)
        bullet_mask = self._points_mask(bx, by, bvalid)
        img = self._paint_mask(img, bullet_mask, BULLET_COLOR)

        # HUD/Radar (linke obere Box)
        radar_w, radar_h = 48, 24
        img = img.at[0:radar_h, 0:radar_w, :].set(SKY_COLOR)
        # Rahmen
        img = img.at[0, 0:radar_w, :].set(HUD_ACCENT_COLOR)
        img = img.at[radar_h-1, 0:radar_w, :].set(HUD_ACCENT_COLOR)
        img = img.at[0:radar_h, 0, :].set(HUD_ACCENT_COLOR)
        img = img.at[0:radar_h, radar_w-1, :].set(HUD_ACCENT_COLOR)
        # Punkte im Radar (runterskaliert) – ebenfalls über Masken
        rx = jnp.clip(jnp.floor((state.obstacles.x - state.player_tank.x)/40.0 + radar_w/2).astype(jnp.int32), 1, radar_w-2)
        ry = jnp.clip(jnp.floor((state.obstacles.y - state.player_tank.y)/40.0 + radar_h/2).astype(jnp.int32), 1, radar_h-2)
        rvalid = (state.obstacles.alive == 1)
        # Radarbereich -> globale Koords
        radar_mask = self._points_mask(rx, ry, rvalid)
        # in Radarfenster verschieben
        # Baue leeres Maskenbild und kopiere den Radarbereich hinein
        full_mask = jnp.zeros((HEIGHT, WIDTH), dtype=bool)
        # Wir erzeugen eine Maske gleicher Größe, die nur im Radarbereich die Punkte trägt
        mask_radar_area = jnp.zeros((HEIGHT, WIDTH), dtype=bool)
        mask_radar_area = mask_radar_area.at[0:radar_h, 0:radar_w].set(radar_mask[0:radar_h, 0:radar_w])
        img = self._paint_mask(img, mask_radar_area, WIREFRAME_COLOR)

        return img

    @partial(jax.jit, static_argnums=(0,))
    def _get_env_reward(self, previous_state, state) -> float:
        return 0.0

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(self, previous_state, state):
        return jnp.zeros(1)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state) -> bool:
        return state.player_tank.alive == 0

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state) -> BattleZoneInfo:
        return BattleZoneInfo(time=state.step_counter, all_rewards=jnp.zeros(1), player_shot=jnp.array(0, dtype=jnp.int32))

    @partial(jax.jit, static_argnums=(0,))
    def _get_info_full(self, state, all_rewards: chex.Array = None, player_shot: chex.Array = None) -> BattleZoneInfo:
        if all_rewards is None:
            all_rewards = jnp.zeros(1)
        if player_shot is None:
            player_shot = jnp.array(0, dtype=jnp.int32)
        return BattleZoneInfo(time=state.step_counter, all_rewards=all_rewards, player_shot=player_shot)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state, state) -> float:
        return 0.0
