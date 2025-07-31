import os
import jax
import chex
import jax.numpy as jnp
import pygame
from functools import partial
import numpy as np
from jaxatari.core import JaxEnvironment
import jax.random as jrandom
from typing import Tuple, NamedTuple
import math
from typing import NamedTuple, Tuple, Dict, Any, Optional

"""
    Group member of the Gravitar: Xusong Yin, Elizaveta Kuznetsova, Li Dai
"""

# ========== Constants ==========
MAX_BULLETS = 64
MAX_ENEMIES = 16
# 18 discrete spaceship action constants
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

# HUD settings
HUD_HEIGHT = 30
MAX_LIVES = 6
HUD_PADDING = 5
HUD_SHIP_WIDTH = 10
HUD_SHIP_HEIGHT = 12
HUD_SHIP_SPACING = 12

# Pygame window dimensions
WINDOW_WIDTH = 160 * 3
WINDOW_HEIGHT = 210 * 3


# ========== Bullet State ==========
# 定义子弹的状态
class Bullets(NamedTuple):
    x: jnp.ndarray  # shape(MAX_BULLETS, )
    y: jnp.ndarray
    vx: jnp.ndarray
    vy: jnp.ndarray
    alive: jnp.ndarray  # bool array


# ========== Enemies States ==========
# 初始化Enemy
class Enemies(NamedTuple):
    x: jnp.ndarray  # shape (MAX_ENEMIES,)
    y: jnp.ndarray
    w: jnp.ndarray
    h: jnp.ndarray
    vx: jnp.ndarray


# ========== Ship State ==========
class ShipState(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    vx: jnp.ndarray
    vy: jnp.ndarray
    angle: jnp.ndarray


# ========== Env State ==========
class EnvState(NamedTuple):
    mode: int
    state: ShipState
    bullets: Bullets
    cooldown: jnp.ndarray
    enemies: Enemies
    enemy_bullets: Bullets
    fire_cooldown: jnp.ndarray
    key: jnp.ndarray
    key_alt: jnp.ndarray
    score: jnp.ndarray
    done: bool
    lives: jnp.ndarray


# ========== Init Function ==========
# 初始化空子弹
def create_empty_bullets_fixed(size: int) -> Bullets:
    return Bullets(
        x=jnp.zeros((size,), dtype=jnp.float32),
        y=jnp.zeros((size,), dtype=jnp.float32),
        vx=jnp.zeros((size,), dtype=jnp.float32),
        vy=jnp.zeros((size,), dtype=jnp.float32),
        alive=jnp.zeros((size,), dtype=bool)
    )


def create_empty_bullets_64():
    return create_empty_bullets_fixed(64)


def create_empty_bullets_16():
    return create_empty_bullets_fixed(16)


@jax.jit
def create_empty_enemies():
    return Enemies(
        x=jnp.zeros((MAX_ENEMIES,), dtype=jnp.float32),
        y=jnp.zeros((MAX_ENEMIES,), dtype=jnp.float32),
        w=jnp.zeros((MAX_ENEMIES,), dtype=jnp.float32),
        h=jnp.zeros((MAX_ENEMIES,), dtype=jnp.float32),
        vx=jnp.zeros((MAX_ENEMIES,), dtype=jnp.float32)
    )


@jax.jit
def create_env_state(rng: jnp.ndarray) -> EnvState:
    return EnvState(
        mode=1,
        state=ShipState(
            x=jnp.array(100.0),
            y=jnp.array(100.0),
            vx=jnp.array(0.0),
            vy=jnp.array(0.0),
            angle=jnp.array(0.0),
        ),
        bullets=create_empty_bullets_64(),
        cooldown=jnp.array(0, dtype=jnp.int32),
        enemies=create_empty_enemies(),
        enemy_bullets=create_empty_bullets_16(),
        fire_cooldown=jnp.zeros((MAX_ENEMIES,), dtype=jnp.int32),
        key=rng,
        key_alt=rng,
        score=jnp.array(0.0),
        done=False,
        lives=jnp.array(6, dtype=jnp.int32)
    )


# ========== Update Bullets ==========
@jax.jit
def update_bullets(bullets: Bullets) -> Bullets:
    new_x = bullets.x + bullets.vx
    new_y = bullets.y + bullets.vy

    valid_x = (new_x >= 0) & (new_x <= WINDOW_WIDTH)
    valid_y = (new_y >= HUD_HEIGHT) & (new_y <= WINDOW_HEIGHT)
    valid = valid_x & valid_y & bullets.alive

    return Bullets(
        x=new_x,
        y=new_y,
        vx=bullets.vx,
        vy=bullets.vy,
        alive=valid
    )


# ========== Merge Bullets ==========
@jax.jit
def merge_bullets(b1: Bullets, b2: Bullets, max_len: int = 16) -> Bullets:
    x = jnp.concatenate([b1.x, b2.x], axis=0)
    y = jnp.concatenate([b1.y, b2.y], axis=0)
    vx = jnp.concatenate([b1.vx, b2.vx], axis=0)
    vy = jnp.concatenate([b1.vy, b2.vy], axis=0)
    alive = jnp.concatenate([b1.alive, b2.alive], axis=0)

    # 排序：优先把 alive == True 的排前面
    sort_key = alive.astype(jnp.int32)
    indices = jnp.argsort(-sort_key)  # True 在前
    x = x[indices][:max_len]
    y = y[indices][:max_len]
    vx = vx[indices][:max_len]
    vy = vy[indices][:max_len]
    alive = alive[indices][:max_len]

    return Bullets(x=x, y=y, vx=vx, vy=vy, alive=alive)


# ========== Fire Bullet ==========
# 发射子弹
@jax.jit
def fire_bullet(bullets: Bullets, ship_x, ship_y, ship_angle, bullet_speed):
    def add_bullet(_):
        idx = jnp.argmax(bullets.alive == False)
        new_vx = jnp.cos(ship_angle) * bullet_speed
        new_vy = jnp.sin(ship_angle) * bullet_speed

        return Bullets(
            x=bullets.x.at[idx].set(ship_x),
            y=bullets.y.at[idx].set(ship_y),
            vx=bullets.vx.at[idx].set(new_vx),
            vy=bullets.vy.at[idx].set(new_vy),
            alive=bullets.alive.at[idx].set(True)
        )

    def skip_bullet(_):
        return bullets

    can_fire = jnp.any(bullets.alive == False)
    return jax.lax.cond(can_fire, add_bullet, skip_bullet, operand=None)


# ========== Ship Collision Utilities ==========
# 飞船碰撞情况
@jax.jit
def check_ship_crash(state: ShipState, enemies: Enemies, hitbox_size: float) -> bool:
    sx1 = state.x - hitbox_size
    sx2 = state.x + hitbox_size
    sy1 = state.y - hitbox_size
    sy2 = state.y + hitbox_size

    ex1 = enemies.x - enemies.w / 2
    ex2 = enemies.x + enemies.w / 2
    ey1 = enemies.y - enemies.h / 2
    ey2 = enemies.y + enemies.h / 2

    overlap_x = (sx1 <= ex2) & (sx2 >= ex1)
    overlap_y = (sy1 <= ey2) & (sy2 >= ey1)

    return jnp.any(overlap_x & overlap_y)


@jax.jit
def check_ship_hit(state: ShipState, bullets: Bullets, hitbox_size: float) -> bool:
    sx1 = state.x - hitbox_size
    sx2 = state.x + hitbox_size
    sy1 = state.y - hitbox_size
    sy2 = state.y + hitbox_size

    within_x = (bullets.x >= sx1) & (bullets.x <= sx2)
    within_y = (bullets.y >= sy1) & (bullets.y <= sy2)

    return jnp.any(within_x & within_y & bullets.alive)


@jax.jit
def check_enemy_hit(bullets: Bullets, enemies: Enemies) -> Tuple[Bullets, Enemies]:
    padding = 2.0
    ex1 = enemies.x - enemies.w / 2 - padding
    ex2 = enemies.x + enemies.w / 2 + padding
    ey1 = enemies.y - enemies.h / 2 - padding
    ey2 = enemies.y + enemies.h / 2 + padding

    bx = bullets.x[:, None]
    by = bullets.y[:, None]

    cond_x = (bx >= ex1) & (bx <= ex2)
    cond_y = (by >= ey1) & (by <= ey2)
    hit_matrix = cond_x & cond_y & bullets.alive[:, None]

    bullet_hit = jnp.any(hit_matrix, axis=1)
    enemy_hit = jnp.any(hit_matrix, axis=0)

    new_bullets = Bullets(
        x=bullets.x,
        y=bullets.y,
        vx=bullets.vx,
        vy=bullets.vy,
        alive=bullets.alive & (~bullet_hit)
    )

    new_enemies = Enemies(
        x=enemies.x,
        y=enemies.y,
        w=jnp.where(enemy_hit, 0.0, enemies.w),
        h=jnp.where(enemy_hit, 0.0, enemies.h),
        vx=enemies.vx
    )
    print("enemy_hit:", enemy_hit)
    print("enemies.w before:", enemies.w)
    print("enemies.w after:", new_enemies.w)
    return new_bullets, new_enemies


# ========== Ship Step ==========
# 飞船移动不用dict
@jax.jit
def ship_step(state: ShipState,
              action: int,
              window_size: tuple[int, int],
              hud_height: int) -> ShipState:
    rotation_speed = 0.2
    thrust_power = 0.4
    gravity = 0.08
    bounce_damping = 0.2  # Damping factor for bounce velocity

    rotate_right_actions = jnp.array([3, 6, 8, 11, 14, 16])
    rotate_left_actions = jnp.array([4, 7, 9, 12, 15, 17])
    thrust_actions = jnp.array([2, 6, 7, 10, 14, 15])
    down_thrust_actions = jnp.array([5, 8, 9, 13, 16, 17])

    right = jnp.isin(action, rotate_right_actions)
    left = jnp.isin(action, rotate_left_actions)
    thrust = jnp.isin(action, thrust_actions)
    down_thrust = jnp.isin(action, down_thrust_actions)

    angle = jnp.where(right, state.angle + rotation_speed, state.angle)
    angle = jnp.where(left, angle - rotation_speed, angle)

    vx = jnp.where(thrust, state.vx + jnp.cos(angle) * thrust_power, state.vx)
    vy = jnp.where(thrust, state.vy + jnp.sin(angle) * thrust_power, state.vy)
    vx = jnp.where(down_thrust, vx - jnp.cos(angle) * thrust_power, vx)
    vy = jnp.where(down_thrust, vy - jnp.sin(angle) * thrust_power, vy)

    vy += gravity

    next_x_unclipped = state.x + vx
    next_y_unclipped = state.y + vy
    ship_half_size = 5
    window_width, window_height = window_size

    # --- Boundary collision and bounce logic ---
    # Horizontal bounce
    hit_left = next_x_unclipped < ship_half_size
    hit_right = next_x_unclipped > window_width - ship_half_size

    # Store old vx and vy for bounce calculation
    old_vx = vx
    old_vy = vy

    # Apply bounce to velocities
    vx = jnp.where(hit_left | hit_right, -old_vx * bounce_damping, old_vx)

    # Update position based on potentially bounced velocity, then clip
    x = state.x + vx  # Use the (potentially bounced) new vx
    x = jnp.clip(x, ship_half_size, window_width - ship_half_size)  # Clip to ensure it's precisely at the edge

    # Vertical bounce
    hit_top = next_y_unclipped < hud_height + ship_half_size
    hit_bottom = next_y_unclipped > window_height - ship_half_size

    vy = jnp.where(hit_top | hit_bottom, -old_vy * bounce_damping, old_vy)

    # Update position based on potentially bounced velocity, then clip
    y = state.y + vy  # Use the (potentially bounced) new vy
    y = jnp.clip(y, hud_height + ship_half_size, window_height - ship_half_size)

    return ShipState(x=x, y=y, vx=vx, vy=vy, angle=angle)


# ========== Enemy Step ==========
# Enemy的移动
@jax.jit
def enemy_step(enemies: Enemies, window_width: int) -> Enemies:
    x = enemies.x + enemies.vx
    left_hit = x <= 0
    right_hit = (x + enemies.w) >= window_width
    hit_edge = left_hit | right_hit
    vx = jnp.where(hit_edge, -enemies.vx, enemies.vx)
    return Enemies(x=x, y=enemies.y, w=enemies.w, h=enemies.h, vx=vx)


# ========== Enemy Fire ==========
@jax.jit
def enemy_fire(enemies: Enemies,
               ship_x: float,
               ship_y: float,
               enemy_bullet_speed: float,
               fire_cooldown: jnp.ndarray,  # shape should match len(enemies.x)
               fire_interval: int,
               key: jax.random.PRNGKey
               ) -> tuple[Bullets, jnp.ndarray, jax.random.PRNGKey]:
    ex_center = enemies.x + enemies.w / 2
    ey_center = enemies.y + enemies.h / 2

    dx = ship_x - ex_center  # shape=(N,)
    dy = ship_y - ey_center  # shape=(N,)
    dist = jnp.sqrt(dx ** 2 + dy ** 2)
    dist = jnp.where(dist < 1e-3, 1.0, dist)  # 避免除以过小的数

    vx = dx / dist * enemy_bullet_speed
    vy = dy / dist * enemy_bullet_speed

    # 判断敌人是否还活着（例如用宽度是否为 0）
    alive_mask = enemies.w > 0.0  # 或 enemies.h > 0.0
    should_fire = (fire_cooldown == 0) & alive_mask

    fire_cooldown = fire_cooldown - 1
    fire_cooldown = jnp.where(fire_cooldown < 0, 0, fire_cooldown)
    fire_cooldown = jnp.where(should_fire, fire_interval, fire_cooldown)

    x_out = jnp.where(should_fire, ex_center, -1.0)
    y_out = jnp.where(should_fire, ey_center, -1.0)
    vx_out = jnp.where(should_fire, vx, 0.0)
    vy_out = jnp.where(should_fire, vy, 0.0)

    bullets_out = Bullets(
        x=x_out,
        y=y_out,
        vx=vx_out,
        vy=vy_out,
        alive=should_fire
    )

    return bullets_out, fire_cooldown, key


# ========== Collision Detection ==========
@jax.jit
def check_collision(bullets: Bullets, enemies: Enemies):
    def bullet_hits_enemy(i, carry):  # carry是累计结果，shape是(MAX_BULLETS,)的bool数组
        x = bullets.x[i]  # x,y 当前子弹坐标
        y = bullets.y[i]
        alive = bullets.alive[i]  # 子弹是否还存活

        def check_each_enemy(j, hit):
            within_x = (x > enemies.x[j]) & (x < enemies.x[j] + enemies.w[j])  # 矩形包围盒，Enemy的包围盒：(x,x+w), (y, y+h)
            within_y = (y > enemies.y[j]) & (y < enemies.y[j] + enemies.h[j])
            return hit | (within_x & within_y)

        hit_any = jax.lax.fori_loop(0, MAX_ENEMIES, check_each_enemy, False)
        return carry.at[i].set(hit_any & alive)

    hits = jnp.zeros((MAX_BULLETS,), dtype=bool)
    hits = jax.lax.fori_loop(0, MAX_BULLETS, bullet_hits_enemy, hits)
    return hits


# ========== Step Core ==========
@jax.jit
def step_core(env_state: EnvState, action: int):
    def step_map(env_state):
        obs, new_state, reward, done, info, reset, level = step_core_map(
            env_state.state,
            action,
            (WINDOW_WIDTH, WINDOW_HEIGHT),
            HUD_HEIGHT
        )
        info = {
            "crash": jnp.array(False),
            "hit_by_bullet": jnp.array(False)
        }
        return (
            obs,
            env_state._replace(
                state=new_state,
                score=env_state.score + reward,  # 以后进入一个星球加分
                done=done,
                enemy_bullets=create_empty_bullets_16(),
            ),
            reward,
            done,
            info,
            reset,
            level
        )

    def step_level(env_state):
        obs, new_state, bullets, cooldown, enemies, enemy_bullets, fire_cooldown, key, reward, score_delta, done, reset, level = step_core_level(
            env_state.state,
            action,
            env_state.bullets,
            env_state.cooldown,
            env_state.enemies,
            env_state.enemy_bullets,
            env_state.fire_cooldown,
            env_state.key,
            (WINDOW_WIDTH, WINDOW_HEIGHT),
            HUD_HEIGHT,
            cooldown_max=5,
            bullet_speed=5.0,
            enemy_bullet_speed=5.0,
            fire_interval=60
        )
        return (
            obs,
            env_state._replace(
                state=new_state,
                bullets=bullets,
                cooldown=cooldown,
                enemies=enemies,
                enemy_bullets=enemy_bullets,
                fire_cooldown=fire_cooldown,
                key=key,
                score=env_state.score + score_delta,
                done=done
            ),
            reward,
            done,
            {
                "crash": jnp.array(False),
                "hit_by_bullet": jnp.array(False)
            },
            reset,  # reset
            level  # level
        )

    return jax.lax.cond(
        env_state.mode == 0,
        lambda _: step_map(env_state),
        lambda _: step_level(env_state),
        operand=None
    )


# ========== Step Core Map ==========
@jax.jit
def step_core_map(state: ShipState,
                  action: int,
                  window_size: Tuple[int, int],
                  hud_height: int
                  ) -> Tuple[jnp.ndarray, ShipState, float, bool, dict, bool, int]:
    new_state = ship_step(state, action, window_size, hud_height)

    obs = jnp.array([
        new_state.x,
        new_state.y,
        new_state.vx,
        new_state.vy,
        new_state.angle
    ])

    reward = 0.0
    done = False
    info = {}

    # 星球数据（不可用 Python 的 self.planets，需转成静态数组）
    planet_x = jnp.array([60.0, 120.0, 200.0])
    planet_y = jnp.array([120.0, 200.0, 80.0])
    planet_r = jnp.array([15.0, 15.0, 15.0])
    level_ids = jnp.array([0, 1, 2])

    dx = planet_x - new_state.x
    dy = planet_y - new_state.y
    dists = jnp.sqrt(dx ** 2 + dy ** 2)
    within_planet = dists < planet_r

    reset = jnp.any(within_planet)
    level_idx = jnp.argmax(within_planet)
    level = jnp.where(reset, level_ids[level_idx], -1)

    return obs, new_state, reward, done, info, reset, level


# ========== Step Core Level Skeleton ==========
@jax.jit
def step_core_level(state: ShipState,
                    action: int,
                    bullets: Bullets,
                    cooldown: int,
                    enemies: Enemies,
                    enemy_bullets: Bullets,
                    fire_cooldown: jnp.ndarray,
                    key: jax.random.PRNGKey,
                    window_size: Tuple[int, int],
                    hud_height: int,
                    cooldown_max: int,
                    bullet_speed: float,
                    enemy_bullet_speed: float,
                    fire_interval: int) -> Tuple[
    jnp.ndarray, ShipState, Bullets, int, Enemies, Bullets, jnp.ndarray, jax.random.PRNGKey, float, float, bool, bool, int]:
    # 飞船移动
    state = ship_step(state, action, window_size, hud_height)

    # 处理射击行为
    fire_actions = jnp.array([1, 10, 11, 12, 13, 14, 15, 16, 17])
    is_firing = jnp.isin(action, fire_actions)
    bullets = jax.lax.cond(
        is_firing & (cooldown == 0),
        lambda _: fire_bullet(bullets, state.x, state.y, state.angle, bullet_speed),
        lambda _: bullets,
        operand=None
    )
    bullets = update_bullets(bullets)
    cooldown = jnp.where(is_firing & (cooldown == 0), cooldown_max, jnp.maximum(cooldown - 1, 0))

    # 敌人移动
    enemies = enemy_step(enemies, window_width=window_size[0])

    new_enemy_bullets, fire_cooldown, key = enemy_fire(
        enemies, state.x, state.y, enemy_bullet_speed, fire_cooldown, fire_interval, key
    )

    # ========== 子弹合并 + 更新（静态长度控制） ==========

    def pad_bullets(b: Bullets, target_len=16):
        def pad_array(arr, pad_val=0.0):
            cur_len = arr.shape[0]
            pad_len = max(target_len - cur_len, 0)
            return jnp.pad(arr[:target_len], (0, pad_len), constant_values=pad_val)

        return Bullets(
            x=pad_array(b.x),
            y=pad_array(b.y),
            vx=pad_array(b.vx),
            vy=pad_array(b.vy),
            alive=pad_array(b.alive, pad_val=False)
        )

    def truncate_bullets(b: Bullets, max_len=16):
        return Bullets(
            x=b.x[:max_len],
            y=b.y[:max_len],
            vx=b.vx[:max_len],
            vy=b.vy[:max_len],
            alive=b.alive[:max_len]
        )

    new_enemy_bullets = pad_bullets(new_enemy_bullets, target_len=16)
    enemy_bullets = pad_bullets(enemy_bullets, target_len=16)
    enemy_bullets = merge_bullets(enemy_bullets, new_enemy_bullets)
    enemy_bullets = update_bullets(enemy_bullets)
    enemy_bullets = truncate_bullets(enemy_bullets, max_len=16)
    # 子弹与敌人碰撞检测
    # 保存前一帧敌人宽度（用来判断是否新死亡）
    w_prev = enemies.w

    # 命中判定
    bullets, enemies = check_enemy_hit(bullets, enemies)

    # 新增死亡敌人（从 w > 0 到 w == 0）
    new_killed = (w_prev > 0.0) & (enemies.w == 0.0)
    num_new_killed = jnp.sum(new_killed)
    score_delta = num_new_killed.astype(jnp.float32) * 10.0
    # 敌人或子弹命中飞船
    crashed = check_ship_crash(state, enemies, hitbox_size=10.0)
    hit_by_bullet = check_ship_hit(state, enemy_bullets, hitbox_size=10.0)
    dead = crashed | hit_by_bullet

    reward = jnp.where(dead, -10.0, -1.0)

    all_dead = jnp.all(enemies.w == 0.0)
    done = dead

    # 地图跳转（不是死亡，是胜利）
    reset = all_dead & (~dead)  # 胜利进入地图模式（不能同时死亡和胜利）
    level = jnp.where(reset, -1, -1)  # 回地图时设为 -1

    obs = jnp.array([state.x, state.y, state.vx, state.vy, state.angle])
    print("enemy_bullets.alive:", enemy_bullets.alive)
    return obs, state, bullets, cooldown, enemies, enemy_bullets, fire_cooldown, key, reward, score_delta, done, reset, level


def get_action_from_key():
    keys = pygame.key.get_pressed()

    thrust = keys[pygame.K_UP]
    rotate_left = keys[pygame.K_LEFT]
    rotate_right = keys[pygame.K_RIGHT]
    fire = keys[pygame.K_SPACE]
    down = keys[pygame.K_DOWN]

    # Map action combinations    映射动作组合
    if fire:
        return 1
    elif thrust:
        return 2
    elif rotate_right:
        return 3  # RIGHT
    elif rotate_left:
        return 4  # LEFT
    elif down:
        return 5  # DOWN
    elif thrust and rotate_right:
        return 6  # UPRIGHT
    elif thrust and rotate_left:
        return 7
    elif down and rotate_right:
        return 8  # DOWNRIGHT
    elif down and rotate_left:
        return 9  # DOWNLEFT
    elif thrust and fire:
        return 10  # UPFIRE
    elif rotate_right and fire:
        return 11  # RIGHTFIRE
    elif rotate_left and fire:
        return 12  # LEFTFIRE
    elif down and fire:
        return 13  # DOWNFIRE
    elif thrust and rotate_right and fire:
        return 14  # UPRIGHTFIRE
    elif thrust and rotate_left and fire:
        return 15  # UPLEFTFIRE
    elif down and rotate_right and fire:
        return 16  # DOWNRIGHTFIRE
    elif down and rotate_left and fire:
        return 17  # DOWNLEFTFIRE

    # default return to action 0     默认返回action 0
    return 0


class JaxGravitar(JaxEnvironment):
    def __init__(self):
        super().__init__()
        self.obs_shape = (5,)  # [x, y, vx, vy, angle]
        self.num_actions = 18

        # Initialize font library     初始化字体库
        pygame.font.init()
        # Initialize score and health    初始化分数和血条
        self.score = 0
        self.lives = 6
        # Initialize the map    初始化地图
        self.mode = "map"
        self.planets = [
            {"x": 60, "y": 120, "r": 15, "level_id": 0, "color": (255, 100, 100)},
            {"x": 120, "y": 200, "r": 15, "level_id": 1, "color": (100, 255, 100)},
            {"x": 200, "y": 80, "r": 15, "level_id": 2, "color": (100, 100, 255)}
        ]
        self.current_level = None
        self.game_over = False
        self.done = False
        self.key = jrandom.PRNGKey(0)
        # List of bullets     子弹列表
        self.bullets = Bullets(  # Store all active bullets: x, y, vx, vy        存放所有存活子弹，x, y, vx, vy,
            x=jnp.zeros((0,)),
            y=jnp.zeros((0,)),
            vx=jnp.zeros((0,)),
            vy=jnp.zeros((0,)),
            alive=jnp.zeros((0,), dtype=bool)
        )
        self.bullets_speed = 5.0

        # Enemy bullet list    enemy子弹列表
        self.enemy_bullets = Bullets(
            x=jnp.zeros((0,)),
            y=jnp.zeros((0,)),
            vx=jnp.zeros((0,)),
            vy=jnp.zeros((0,)),
            alive=jnp.zeros((0,), dtype=bool)
        )
        self.enemy_fire_cooldown = 0
        self.enemy_fire_interval = 60
        self.enemy_bullet_speed = 5.0

        # Shooting cooldown    射击cd
        self.cooldown = 0
        self.cooldown_max = 5

        # Set window size     设置窗口大小
        self.screen_size = (WINDOW_WIDTH, WINDOW_HEIGHT)
        pygame.init()
        self.screen = pygame.display.set_mode(self.screen_size)
        pygame.display.set_caption("Gravitar")

    def draw_hud(self):
        # Background    背景
        pygame.draw.rect(self.screen, (0, 0, 0), (0, 0, WINDOW_WIDTH, HUD_HEIGHT))

        # Score   分数
        font = pygame.font.SysFont("Arial", 14, bold=True)
        score_text = font.render(str(self.score), True, (255, 128, 255))
        text_rect = score_text.get_rect(center=(WINDOW_WIDTH // 2, HUD_PADDING + 6))
        self.screen.blit(score_text, text_rect)

        # Health bar   血条
        total_width = self.lives * HUD_SHIP_SPACING
        start_x = (WINDOW_WIDTH - total_width) // 2
        for i in range(self.lives):
            x_offset = start_x + i * HUD_SHIP_SPACING
            y_offset = HUD_PADDING + 16
            pygame.draw.rect(self.screen, (135, 206, 250), (x_offset, y_offset, HUD_SHIP_WIDTH, HUD_SHIP_HEIGHT))

    def get_action_space(self) -> Tuple[int]:
        """
        Returns the action space of the environment.
        Returns: The action space of the environment as a tuple.
        """
        # raise NotImplementedError("Abstract method")
        return (18,)  # Supports actions numbered 0 to 17    支持编号0到17的动作

    def get_observation_space(self) -> Tuple[int]:
        """
        Returns the observation space of the environment.
        Returns: The observation space of the environment as a tuple.
        """
        # raise NotImplementedError("Abstract method")
        return (5,)

    def render(self, state: ShipState) -> Tuple[jnp.ndarray]:
        """
        Renders the environment state to a single image.
        Args:
            state: The environment state.

        Returns: A single image of the environment state.

        """
        # raise NotImplementedError("Abstract method")

        self.screen.fill((0, 0, 0))
        if self.mode == "map":
            # Draw planets    画星球
            for planet in self.planets:
                color = planet.get("color", (0, 255, 255))
                pygame.draw.circle(self.screen, color, (planet["x"], planet["y"]), planet["r"])
        # Render triangular spaceship      渲染三角形飞船
        # Spaceship center position    飞船中心位置
        cx, cy = float(state.x), float(state.y)
        angle = float(state.angle)
        size = 10  # Size of the spaceship tip    飞船箭头大小
        # Three triangle vertices, angle indicates direction      三角形三个顶点位置，angle指向朝向
        tip = (cx + math.cos(angle) * size, cy + math.sin(angle) * size)
        left = (cx + math.cos(angle + 2.5) * size, cy + math.sin(angle + 2.5) * size)
        right = (cx + math.cos(angle - 2.5) * size, cy + math.sin(angle - 2.5) * size)
        # Clear background   清空背景
        self.draw_hud()
        # Draw spaceship  画飞船
        pygame.draw.polygon(self.screen, (0, 255, 0), [tip, left, right])

        # Draw enemy   画enemy
        if self.mode == "level":
            for i in range(len(self.enemies.x)):
                if self.enemies.w[i] == 0:
                    continue
                ex = self.enemies.x[i]
                ey = self.enemies.y[i]
                ew = self.enemies.w[i]
                eh = self.enemies.h[i]
                pygame.draw.rect(self.screen, (255, 0, 0), (int(ex), int(ey), int(ew), int(eh)))

        # Score  得分
        font = pygame.font.SysFont(None, 24)
        score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        # Render bullets   渲染子弹
        for i in range(len(self.bullets.x)):
            if not self.bullets.alive[i]:
                continue
            bx = self.bullets.x[i]
            by = self.bullets.y[i]
            if bx >= 0 and by >= 0:  # 排除 dummy bullets
                pygame.draw.circle(self.screen, (255, 255, 0), (int(bx), int(by)), 2)
        # Render enemy bullets    渲染enemy子弹

        for i in range(len(self.enemy_bullets.x)):
            if not self.enemy_bullets.alive[i]:
                continue
            bx = self.enemy_bullets.x[i]
            by = self.enemy_bullets.y[i]
            if bx >= 0 and by >= 0:
                pygame.draw.circle(self.screen, (255, 100, 0), (int(bx), int(by)), 3)
        # gameover screen
        if hasattr(self, "done") and self.done:
            font = pygame.font.SysFont(None, 72)
            text = font.render("game over", True, (255, 0, 0))
            text_rect = text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
            self.screen.blit(text, text_rect)

        pygame.display.flip()

        dummy_image = jnp.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3))
        return (dummy_image,)

    def reset_map(self, key: jnp.ndarray) -> Tuple[jnp.ndarray, EnvState]:
        ship_state = ShipState(
            x=jnp.array(WINDOW_WIDTH / 2, dtype=jnp.float32),
            y=jnp.array((WINDOW_HEIGHT + HUD_HEIGHT) / 2, dtype=jnp.float32),
            vx=jnp.array(0.0, dtype=jnp.float32),
            vy=jnp.array(0.0, dtype=jnp.float32),
            angle=jnp.array(-jnp.pi / 2, dtype=jnp.float32)
        )

        env_state = EnvState(
            mode=0,
            state=ship_state,
            bullets=create_empty_bullets_64(),
            cooldown=jnp.array(0, dtype=jnp.int32),
            enemies=create_empty_enemies(),
            enemy_bullets=create_empty_bullets_16(),
            fire_cooldown=jnp.zeros((MAX_ENEMIES,), dtype=jnp.int32),
            key=key,
            key_alt=key,
            score=jnp.array(0.0, dtype=jnp.float32),
            done=False,
            lives=jnp.array(6, dtype=jnp.int32)
        )

        obs = jnp.array([
            ship_state.x,
            ship_state.y,
            ship_state.vx,
            ship_state.vy,
            ship_state.angle
        ])

        return obs, env_state

    def reset(self, key: jnp.ndarray = jax.random.PRNGKey(0)) -> Tuple[jnp.ndarray, EnvState]:
        obs, env_state = self.reset_map(key)
        return obs, env_state

    def reset_level(self, key: jnp.ndarray, level_id: int) -> Tuple[jnp.ndarray, EnvState]:
        ship_state = ShipState(
            x=jnp.array(WINDOW_WIDTH / 2, dtype=jnp.float32),
            y=jnp.array((WINDOW_HEIGHT + HUD_HEIGHT) / 2, dtype=jnp.float32),
            vx=jnp.array(0.0, dtype=jnp.float32),
            vy=jnp.array(0.0, dtype=jnp.float32),
            angle=jnp.array(-jnp.pi / 2, dtype=jnp.float32)
        )

        # 根据 level_id 配置参数
        if level_id == 0:
            num_enemies = 3
            enemy_speed = 1.0
            spawn_x_range = (60, WINDOW_WIDTH - 80)
            spawn_y_range = (60, WINDOW_HEIGHT - 100)
        elif level_id == 1:
            num_enemies = 5
            enemy_speed = 1.5
            spawn_x_range = (100, WINDOW_WIDTH - 100)
            spawn_y_range = (100, WINDOW_HEIGHT - 120)
        else:
            num_enemies = 7
            enemy_speed = 2.0
            spawn_x_range = (120, WINDOW_WIDTH - 120)
            spawn_y_range = (120, WINDOW_HEIGHT - 140)

        enemy_w, enemy_h = 14.0, 14.0
        keys = jax.random.split(key, num_enemies * 3)

        x = jnp.array([
            jax.random.randint(keys[i * 3 + 0], (), spawn_x_range[0], spawn_x_range[1])
            for i in range(num_enemies)
        ], dtype=jnp.float32)

        y = jnp.array([
            jax.random.randint(keys[i * 3 + 1], (), spawn_y_range[0], spawn_y_range[1])
            for i in range(num_enemies)
        ], dtype=jnp.float32)

        vx = jnp.array([
            jnp.where(
                jax.random.bernoulli(keys[i * 3 + 2]),
                enemy_speed,
                -enemy_speed
            )
            for i in range(num_enemies)
        ], dtype=jnp.float32)

        def pad(arr, fill_val=0.0):
            return jnp.pad(arr, (0, MAX_ENEMIES - arr.shape[0]), constant_values=fill_val)

        enemies = Enemies(
            x=pad(x),
            y=pad(y),
            w=pad(jnp.full((num_enemies,), enemy_w, dtype=jnp.float32)),
            h=pad(jnp.full((num_enemies,), enemy_h, dtype=jnp.float32)),
            vx=pad(vx)
        )

        fire_cooldown = jnp.full((MAX_ENEMIES,), 999, dtype=jnp.int32).at[:num_enemies].set(0)

        env_state = EnvState(
            mode=1,
            state=ship_state,
            bullets=create_empty_bullets_64(),
            cooldown=jnp.array(0, dtype=jnp.int32),  # 飞船的cd
            enemies=enemies,
            enemy_bullets=create_empty_bullets_16(),
            fire_cooldown=fire_cooldown,
            key=key,
            key_alt=key,
            score=jnp.array(0.0, dtype=jnp.float32),
            done=False,
            lives=jnp.array(6, dtype=jnp.int32)
        )

        obs = jnp.array([
            ship_state.x,
            ship_state.y,
            ship_state.vx,
            ship_state.vy,
            ship_state.angle
        ])

        return obs, env_state

    def step(self, env_state: EnvState, action: int):
        obs, new_env_state, reward, done, info, reset, level = step_core(env_state, action)
        return obs, new_env_state, reward, done, info, reset, level


if __name__ == "__main__":
    env = JaxGravitar()
    # Initialize random seed    初始化随机种子
    key = jax.random.PRNGKey(0)
    # Get initial observation and state            得到初始观测和状态obs, state
    obs, env_state = env.reset(key)
    # use the loopto play the game     循环
    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()

        if env_state.done:
            env.lives -= 1
            if env.lives <= 0:
                env.done = True
                env.render(env_state.state)
                pygame.time.wait(100)
                continue
            else:
                key, subkey = jrandom.split(key)
                obs, env_state = env.reset_level(subkey, env.current_level)
                # 更新可视化用的敌人和子弹状态
                env.enemies = env_state.enemies
                env.bullets = env_state.bullets
                continue

        env.render(env_state.state)
        action = get_action_from_key()

        # Use jittable step
        obs, env_state, reward, done, info, reset, level = env.step(env_state, action)

        reset_flag = bool(reset)
        level_id = int(level)

        if reset_flag and level_id >= 0:
            print(f"进入星球 {level_id}，切换到 level 模式")
            env.mode = "level"
            env.current_level = level_id

            key, subkey = jrandom.split(key)
            obs, env_state = env.reset_level(subkey, level_id)

            # 同步子弹和敌人信息用于渲染
            env.enemies = env_state.enemies
            env.bullets = env_state.bullets
            continue

        # 如果敌人全部被击败，回到地图
        if reset_flag and level_id == -1:
            print("敌人全部消灭，返回地图选择界面")
            env.mode = "map"

            key, subkey = jrandom.split(key)
            obs, env_state = env.reset_map(subkey)
            obs = jnp.array([
                env_state.state.x,
                env_state.state.y,
                env_state.state.vx,
                env_state.state.vy,
                env_state.state.angle
            ])

            # 同步渲染信息
            env.bullets = env_state.bullets
            env.enemy_bullets = env_state.enemy_bullets
            env.enemies = env_state.enemies
            continue

        if env.mode == "level":
            env.enemies = env_state.enemies
            env.bullets = env_state.bullets
            env.enemy_bullets = env_state.enemy_bullets
            env.score = env_state.score
        clock.tick(30)
    
    
    
def _load_sprites(self) -> dict[str, Any]:
    """Loads all necessary sprites from .npy files."""
sprites: Dict[str, Any] = {}       
       
       
import jaxatari.rendering.atraJaxis as aj
from jaxatari.renderers import AtraJaxisRenderer
        
class GravitarRenderer(AtraJaxisRenderer):
    # Type hint for sprites dictionary
    sprites: Dict[str, Any]       
    
    def __init__(self):
        """
        Initializes the renderer by loading sprites, including level backgrounds.

        Args:
            sprite_path: Path to the directory containing sprite .npy files.
        """
        self.sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/gravitar"
        self.sprites = self._load_sprites()
        # Store background sprites directly for use in render function
        
        """
        ///////////////////////////////////////
        self.background_0 = self.sprites.get('background_0')
        self.background_1 = self.sprites.get('background_1')
        self.background_2 = self.sprites.get('background_2')
        """

    def _load_sprites(self) -> dict[str, Any]:
        """Loads all necessary sprites from .npy files."""
        sprites: Dict[str, Any] = {}
        
        # Helper function to load a single sprite frame
        def _load_sprite_frame(name: str) -> Optional[chex.Array]:
            path = os.path.join(self.sprite_path, f'{name}.npy')
            frame = aj.loadFrame(path)
            if isinstance(frame, jnp.ndarray) and frame.ndim >= 2:
                return frame.astype(jnp.uint8)
            
        
        # --- Load Sprites ---
        # Backgrounds + Dynamic elements + UI elements
        # background_0 is black background
        sprite_names = [
            'background_0', 'background_1', 
            'purple_planet', 'green_planet', 'grey_planet', 'brown_planet',
            'blue_planet', 'turquoise_planet', 'pink_planet', 
            'spacecraft', 'enemy', 'blue_dot',
        ]
        for name in sprite_names:
            loaded_sprite = _load_sprite_frame(name)
            if loaded_sprite is not None:
                 sprites[name] = loaded_sprite    
            
        
        # pad the kangaroo and monkey sprites since they have to be used interchangeably (and jax enforces same sizes)
        planet_sprites = aj.pad_to_match([sprites['purple_planet'], 
                                          sprites['green_planet'], 
                                          sprites['grey_planet'], 
                                          sprites['brown_planet'], 
                                          sprites['blue_planet'], 
                                          sprites['turquoise_planet'], 
                                          sprites['pink_planet']])



        sprites['purple_planet'] = planet_sprites[0]
        sprites['green_planet'] = planet_sprites[1]
        sprites['grey_planet'] = planet_sprites[2]
        sprites['brown_planet'] = planet_sprites[3]
        sprites['blue_planet'] = planet_sprites[4]
        sprites['turquoise_planet'] = planet_sprites[5]
        sprites['pink_planet'] = planet_sprites[6]


         # expand all sprites similar to the Pong/Seaquest loading
        for key in sprites.keys():
            if isinstance(sprites[key], (list, tuple)):
                sprites[key] = [jnp.expand_dims(sprite, axis=0) for sprite in sprites[key]]
            else:
                sprites[key] = jnp.expand_dims(sprites[key], axis=0)

        return sprites
            