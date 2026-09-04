import os
import time
from functools import partial
from typing import Tuple

import chex
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np

from jaxatari.environment import JAXAtariAction as Action, JaxEnvironment, ObjectObservation
import jaxatari.rendering.jax_rendering_utils as render_utils
from jaxatari.renderers import JAXGameRenderer
import jaxatari.spaces as spaces


# quick type helpers to stop jax from complaining about type mismatches in loops
def to_bool(x):
    return jnp.asarray(x, dtype=jnp.bool_)


def to_int(x):
    return jnp.asarray(x, dtype=jnp.int32)


# basic 2D bounding box collision test
def check_aabb_overlap(box1_x, box1_y, box1_w, box1_h, box2_x, box2_y, box2_w, box2_h):
    horizontal_overlap = (box1_x < box2_x + box2_w) & (box1_x + box1_w > box2_x)
    vertical_overlap = (box1_y < box2_y + box2_h) & (box1_y + box1_h > box2_y)
    return horizontal_overlap & vertical_overlap


def build_default_asset_manifest() -> tuple:
    return (
        {'name': 'background', 'type': 'background', 'file': 'background.npy'},
        {'name': 'player', 'type': 'single', 'file': 'player.npy'},
        {'name': 'gripper', 'type': 'single', 'file': 'gripper.npy'},
        {'name': 'knife_thrower', 'type': 'single', 'file': 'knife_thrower.npy'},
        {'name': 'tomtom', 'type': 'single', 'file': 'tomtom.npy'},
        {'name': 'dragon', 'type': 'single', 'file': 'dragon.npy'},
        {'name': 'vase', 'type': 'single', 'file': 'vase.npy'},
        {'name': 'snake', 'type': 'single', 'file': 'snake.npy'},
        {'name': 'ball', 'type': 'single', 'file': 'ball.npy'},
        {'name': 'boss', 'type': 'single', 'file': 'boss.npy'},
    )


# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
# all the magic numbers, hitboxes, and custom mod flags go here
class KungFuMasterConstants(struct.PyTreeNode):
    # World geometry
    SCREEN_WIDTH: int = struct.field(pytree_node=False, default=160)
    SCREEN_HEIGHT: int = struct.field(pytree_node=False, default=210)
    FLOOR_Y: int = struct.field(pytree_node=False, default=160)
    CEILING_Y: int = struct.field(pytree_node=False, default=48)

    # Player physics & limits
    PLAYER_WIDTH: int = struct.field(pytree_node=False, default=16)
    PLAYER_HEIGHT: int = struct.field(pytree_node=False, default=24)
    PLAYER_SPEED: int = struct.field(pytree_node=False, default=2)
    JUMP_VEL: int = struct.field(pytree_node=False, default=-8)
    GRAVITY: int = struct.field(pytree_node=False, default=1)
    MAX_ENERGY: int = struct.field(pytree_node=False, default=100)
    MAX_LIVES: int = struct.field(pytree_node=False, default=3)
    FLOOR_TIMER: int = struct.field(pytree_node=False, default=2000)
    PLAYER_SPAWN_X: int = struct.field(pytree_node=False, default=8)
    SHAKE_NEEDED: int = struct.field(pytree_node=False, default=6)
    CROUCH_H_OFFSET: int = struct.field(pytree_node=False, default=12)

    # Capacity limits
    MAX_ENEMIES: int = struct.field(pytree_node=False, default=8)
    MAX_PROJ: int = struct.field(pytree_node=False, default=8)

    # Player attack hitboxes
    PUNCH_X_OFF: int = struct.field(pytree_node=False, default=8)
    PUNCH_W: int = struct.field(pytree_node=False, default=20)
    PUNCH_Y_OFF: int = struct.field(pytree_node=False, default=2)
    PUNCH_H: int = struct.field(pytree_node=False, default=14)

    KICK_X_OFF: int = struct.field(pytree_node=False, default=10)
    KICK_W: int = struct.field(pytree_node=False, default=26)
    KICK_Y_OFF: int = struct.field(pytree_node=False, default=10)
    KICK_H: int = struct.field(pytree_node=False, default=10)

    CROUCH_KICK_X_OFF: int = struct.field(pytree_node=False, default=10)
    CROUCH_KICK_W: int = struct.field(pytree_node=False, default=26)
    CROUCH_KICK_Y_OFF: int = struct.field(pytree_node=False, default=16)
    CROUCH_KICK_H: int = struct.field(pytree_node=False, default=8)

    JUMP_PUNCH_X_OFF: int = struct.field(pytree_node=False, default=8)
    JUMP_PUNCH_W: int = struct.field(pytree_node=False, default=20)
    JUMP_PUNCH_Y_OFF: int = struct.field(pytree_node=False, default=0)
    JUMP_PUNCH_H: int = struct.field(pytree_node=False, default=18)

    JUMP_KICK_X_OFF: int = struct.field(pytree_node=False, default=10)
    JUMP_KICK_W: int = struct.field(pytree_node=False, default=28)
    JUMP_KICK_Y_OFF: int = struct.field(pytree_node=False, default=8)
    JUMP_KICK_H: int = struct.field(pytree_node=False, default=12)

    # Enemy dimension constants
    ENEMY_WIDTH: int = struct.field(pytree_node=False, default=14)
    ENEMY_HEIGHT: int = struct.field(pytree_node=False, default=22)
    SNAKE_HEIGHT: int = struct.field(pytree_node=False, default=10)

    # Rewards & score thresholds
    SC_KICK: int = struct.field(pytree_node=False, default=100)
    SC_PUNCH: int = struct.field(pytree_node=False, default=200)
    SC_JUMP_KICK: int = struct.field(pytree_node=False, default=200)
    SC_KNIFE_K: int = struct.field(pytree_node=False, default=500)
    SC_KNIFE_P: int = struct.field(pytree_node=False, default=800)
    SC_DRAGON: int = struct.field(pytree_node=False, default=2000)
    SC_BALL: int = struct.field(pytree_node=False, default=1000)
    SC_VASE_K: int = struct.field(pytree_node=False, default=200)
    SC_VASE_P: int = struct.field(pytree_node=False, default=100)
    EXTRA_LIFE_PTS: int = struct.field(pytree_node=False, default=40_000)

    # Combo settings
    COMBO_MAX: int = struct.field(pytree_node=False, default=4)
    COMBO_RESET_FRAMES: int = struct.field(pytree_node=False, default=90)

    NUM_FLOORS: int = struct.field(pytree_node=False, default=8)

    # Attack rates and durations
    KNIFE_THROW_CD: int = struct.field(pytree_node=False, default=40)
    DRAGON_FIRE_CD: int = struct.field(pytree_node=False, default=55)
    BALL_SPLIT_CD: int = struct.field(pytree_node=False, default=80)
    BOSS_ATTACK_CD: int = struct.field(pytree_node=False, default=45)
    BOSS_STICK_RANGE: int = struct.field(pytree_node=False, default=24)

    SPD_KNIFE_PROJ: int = struct.field(pytree_node=False, default=4)
    SPD_FIRE_PROJ: int = struct.field(pytree_node=False, default=3)
    SPD_SHRAP_PROJ: int = struct.field(pytree_node=False, default=3)
    BOOMERANG_SPEED: int = struct.field(pytree_node=False, default=3)

    # Damage settings
    DMG_GRAB: int = struct.field(pytree_node=False, default=1)
    DMG_KNIFE: int = struct.field(pytree_node=False, default=5)
    DMG_FIRE: int = struct.field(pytree_node=False, default=8)
    DMG_DRAGON: int = struct.field(pytree_node=False, default=6)
    DMG_SNAKE: int = struct.field(pytree_node=False, default=4)
    DMG_BALL: int = struct.field(pytree_node=False, default=4)
    DMG_BOSS: int = struct.field(pytree_node=False, default=6)
    DMG_SHRAP: int = struct.field(pytree_node=False, default=3)
    GRAB_DRAIN_CD: int = struct.field(pytree_node=False, default=4)

    TOMTOM_JUMP_VEL: int = struct.field(pytree_node=False, default=-6)
    TOMTOM_JUMP_CD: int = struct.field(pytree_node=False, default=60)
    DRAGON_RETREAT_DUR: int = struct.field(pytree_node=False, default=20)

    MAX_STEPS: int = struct.field(pytree_node=False, default=27_000)
    DIFFICULTY: int = struct.field(pytree_node=False, default=1)

    # Custom game modifiers requested in the proposals
    MOD_NO_KNIVES: bool = struct.field(pytree_node=False, default=False)
    MOD_DOUBLE_SPEED: bool = struct.field(pytree_node=False, default=False)
    MOD_INFINITE_ENERGY: bool = struct.field(pytree_node=False, default=False)
    MOD_ONE_HIT_BOSS: bool = struct.field(pytree_node=False, default=False)
    MOD_NO_GRABS: bool = struct.field(pytree_node=False, default=False)
    MOD_REVERSED_FLOORS: bool = struct.field(pytree_node=False, default=False)
    MOD_ALL_KNIFE_FLOOR: bool = struct.field(pytree_node=False, default=False)
    MOD_BOSS_RUSH: bool = struct.field(pytree_node=False, default=False)
    MOD_MIRROR_PLAYER: bool = struct.field(pytree_node=False, default=False)

    FLOOR_ENEMY_COUNT: tuple = struct.field(
        pytree_node=False, default=(6, 8, 10, 10, 12, 12, 14, 16)
    )
    FLOOR_BOSS_HP: tuple = struct.field(
        pytree_node=False, default=(8, 10, 12, 12, 14, 14, 16, 20)
    )
    SPAWN_INTERVAL: tuple = struct.field(
        pytree_node=False, default=(35, 28, 22, 20, 18, 16, 14, 12)
    )
    SCORE_BOSS: tuple = struct.field(
        pytree_node=False, default=(0, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000)
    )

    ASSET_CONFIG: tuple = struct.field(pytree_node=False, default_factory=build_default_asset_manifest)


# Enemy entity identifiers
ENEMY_NONE = 0
ENEMY_GRIPPER = 1
ENEMY_KNIFE = 2
ENEMY_TOMTOM = 3
ENEMY_DRAGON = 4
ENEMY_VASE = 5
ENEMY_SNAKE = 6
ENEMY_BALL = 7
ENEMY_BOSS = 9

# Enemy behavioral states
STATE_WALK = 0
STATE_JUMP_UP = 1
STATE_JUMP_DOWN = 2
STATE_RETREAT = 3

# Projectile types
PROJ_NONE = 0
PROJ_KNIFE_HIGH = 1
PROJ_KNIFE_LOW = 2
PROJ_FIRE = 3
PROJ_SHRAPNEL = 4
PROJ_BOOMERANG = 5


# ==========================================
# STATE & OBSERVATION DEFINITIONS
# ==========================================
# keeping all state flat in a PyTreeNode for jax compilation
class KungFuMasterState(struct.PyTreeNode):
    player_x: chex.Array
    player_y: chex.Array
    player_vel_y: chex.Array
    player_dir: chex.Array
    player_energy: chex.Array
    is_crouching: chex.Array
    is_jumping: chex.Array
    is_grabbed: chex.Array
    grab_timer: chex.Array
    shake_last: chex.Array
    shake_count: chex.Array

    combo: chex.Array
    combo_timer: chex.Array

    en_type: chex.Array
    en_x: chex.Array
    en_y: chex.Array
    en_vel_y: chex.Array
    en_dir: chex.Array
    en_active: chex.Array
    en_hp: chex.Array
    en_cd: chex.Array
    en_state: chex.Array
    en_timer: chex.Array

    pr_type: chex.Array
    pr_x: chex.Array
    pr_y: chex.Array
    pr_vx: chex.Array
    pr_vy: chex.Array
    pr_active: chex.Array

    floor: chex.Array
    loop: chex.Array
    floor_done: chex.Array
    floor_timer: chex.Array
    enemies_left: chex.Array
    spawn_timer: chex.Array
    boss_spawned: chex.Array
    transition_timer: chex.Array

    score: chex.Array
    lives: chex.Array
    extra_life_thr: chex.Array
    step_count: chex.Array
    key: chex.PRNGKey


class KungFuMasterObservation(struct.PyTreeNode):
    player: ObjectObservation
    enemies: ObjectObservation
    projectiles: ObjectObservation
    score: jnp.ndarray
    lives: jnp.ndarray
    floor: jnp.ndarray


class KungFuMasterInfo(struct.PyTreeNode):
    score: chex.Array
    floor: chex.Array
    loop: chex.Array


# ==========================================
# MAIN ENVIRONMENT LOGIC
# ==========================================
class JaxKungFuMaster(
    JaxEnvironment[KungFuMasterState, KungFuMasterObservation, KungFuMasterInfo, KungFuMasterConstants]
):
    # ALE action mapping
    ACTION_SET: jnp.ndarray = jnp.array(
        [
            Action.NOOP, Action.FIRE, Action.UP, Action.RIGHT, Action.LEFT, Action.DOWN,
            Action.UPRIGHT, Action.UPLEFT, Action.DOWNRIGHT, Action.DOWNLEFT,
            Action.UPFIRE, Action.RIGHTFIRE, Action.LEFTFIRE, Action.DOWNFIRE,
        ],
        dtype=jnp.int32,
    )

    def __init__(self, consts: KungFuMasterConstants = None):
        consts = consts or KungFuMasterConstants()
        super().__init__(consts)
        self.renderer = KungFuMasterRenderer(self.consts)

    # --- Difficulty scaling helpers ---
    def _get_speed_multiplier(self) -> float:
        scale_map = (0.75, 1.0, 1.5)
        base = scale_map[self.consts.DIFFICULTY]
        return base * 2.0 if self.consts.MOD_DOUBLE_SPEED else base

    def _calculate_enemy_quota(self, current_floor, current_loop):
        base_counts = jnp.array(self.consts.FLOOR_ENEMY_COUNT, dtype=jnp.int32)
        clamped_idx = jnp.clip(current_floor - 1, 0, self.consts.NUM_FLOORS - 1)
        base_total = base_counts[clamped_idx] + current_loop * to_int(2)
        diff_scalar = (0.75, 1.0, 1.25)[self.consts.DIFFICULTY]
        scaled = jnp.round(base_total.astype(jnp.float32) * diff_scalar).astype(jnp.int32)
        return jnp.clip(scaled, 4, 30)

    def _get_initial_floor_time(self):
        time_scalar = (1.25, 1.0, 0.8)[self.consts.DIFFICULTY]
        return jnp.round(jnp.float32(self.consts.FLOOR_TIMER) * time_scalar).astype(jnp.int32)

    def _calculate_boss_health(self, current_floor, current_loop):
        hp_table = jnp.array(self.consts.FLOOR_BOSS_HP, dtype=jnp.int32)
        clamped_idx = jnp.clip(current_floor - 1, 0, self.consts.NUM_FLOORS - 1)
        base_hp = hp_table[clamped_idx] + current_loop * to_int(2)
        if self.consts.MOD_ONE_HIT_BOSS or self.consts.MOD_BOSS_RUSH:
            return to_int(1)
        return base_hp

    def _get_stage_direction(self, current_floor):
        base_dir = jnp.where(current_floor % 2 == 1, to_int(1), to_int(-1))
        return -base_dir if self.consts.MOD_REVERSED_FLOORS else base_dir

    def _decode_action(self, action):
        # convert raw int actions to directional boolean flags
        is_right = (
            (action == Action.RIGHT) | (action == Action.UPRIGHT) |
            (action == Action.RIGHTFIRE) | (action == Action.DOWNRIGHT)
        )
        is_left = (
            (action == Action.LEFT) | (action == Action.UPLEFT) |
            (action == Action.LEFTFIRE) | (action == Action.DOWNLEFT)
        )
        horizontal_move = jnp.where(is_right, to_int(1), jnp.where(is_left, to_int(-1), to_int(0)))

        is_jumping = to_bool(
            (action == Action.UP) | (action == Action.UPRIGHT) |
            (action == Action.UPLEFT) | (action == Action.UPFIRE)
        )
        is_punching = to_bool((action == Action.FIRE) | (action == Action.UPFIRE))
        is_kicking = to_bool(
            (action == Action.RIGHTFIRE) | (action == Action.LEFTFIRE) |
            (action == Action.DOWNFIRE) | (action == Action.DOWNRIGHT) |
            (action == Action.DOWNLEFT)
        )
        is_crouching = to_bool(
            (action == Action.DOWN) | (action == Action.DOWNFIRE) |
            (action == Action.DOWNRIGHT) | (action == Action.DOWNLEFT)
        )

        if self.consts.MOD_MIRROR_PLAYER:
            horizontal_move = -horizontal_move

        return horizontal_move, is_jumping, is_punching, is_kicking, is_crouching

    # --- Player Physics & Input ---
    def _update_player_position(self, state, action):
        cfg = self.consts
        horiz_dir, jump_cmd, punch_cmd, kick_cmd, crouch_cmd = self._decode_action(action)

        ground_y = cfg.FLOOR_Y - cfg.PLAYER_HEIGHT
        is_grounded = state.player_y >= to_int(ground_y)

        # block movement if grabbed by a gripper
        move_step = horiz_dir * to_int(cfg.PLAYER_SPEED) * to_int(~state.is_grabbed)
        next_x = jnp.clip(state.player_x + move_step, 0, cfg.SCREEN_WIDTH - cfg.PLAYER_WIDTH)
        next_dir = jnp.where(horiz_dir != 0, horiz_dir, state.player_dir)

        # gravity and jump logic
        jump_allowed = jump_cmd & is_grounded & ~state.is_grabbed
        updated_vel_y = jnp.where(jump_allowed, to_int(cfg.JUMP_VEL), state.player_vel_y)
        updated_vel_y = jnp.where(~is_grounded, updated_vel_y + to_int(cfg.GRAVITY), updated_vel_y)
        next_y = state.player_y + updated_vel_y

        hit_floor = next_y >= to_int(ground_y)
        next_y = jnp.where(hit_floor, to_int(ground_y), next_y)
        updated_vel_y = jnp.where(hit_floor, to_int(0), updated_vel_y)

        hit_ceiling = next_y < to_int(cfg.CEILING_Y)
        next_y = jnp.where(hit_ceiling, to_int(cfg.CEILING_Y), next_y)
        updated_vel_y = jnp.where(hit_ceiling, to_int(0), updated_vel_y)

        in_air = to_bool(~(next_y >= to_int(ground_y)))
        is_ducking = to_bool(crouch_cmd & ~in_air)

        # shake mechanic to break free from grippers
        user_moved = horiz_dir != 0
        switched_dir = to_bool(horiz_dir != state.shake_last)
        shake_acc = jnp.where(user_moved & switched_dir, state.shake_count + to_int(1), state.shake_count)
        last_dir = jnp.where(user_moved, horiz_dir, state.shake_last)

        broke_free = to_bool(shake_acc >= to_int(cfg.SHAKE_NEEDED)) & state.is_grabbed
        currently_grabbed = to_bool(state.is_grabbed & ~broke_free)
        shake_acc = jnp.where(broke_free, to_int(0), shake_acc)

        return state.replace(
            player_x=next_x,
            player_y=next_y,
            player_vel_y=updated_vel_y,
            player_dir=next_dir,
            is_jumping=in_air,
            is_crouching=is_ducking,
            is_grabbed=currently_grabbed,
            shake_last=last_dir,
            shake_count=shake_acc,
        )

    # --- Enemy AI & Movement ---
    def _update_all_enemies(self, state):
        cfg = self.consts
        speed_scale = self._get_speed_multiplier()

        def step_enemy(type_code, pos_x, pos_y, vel_y, facing, active_flag, health, cooldown, fsm_state, timer):
            player_x, player_y = state.player_x, state.player_y

            base_spd = jnp.where(
                type_code == ENEMY_TOMTOM, to_int(2),
                jnp.where(
                    type_code == ENEMY_DRAGON, to_int(1),
                    jnp.where(
                        type_code == ENEMY_SNAKE, to_int(1),
                        jnp.where(
                            type_code == ENEMY_BALL, to_int(2),
                            jnp.where(type_code == ENEMY_BOSS, to_int(1), to_int(1))
                        )
                    )
                )
            )
            step_dist = jnp.round(base_spd.astype(jnp.float32) * speed_scale).astype(jnp.int32)

            chase_dir = jnp.sign(player_x - pos_x).astype(jnp.int32)
            chase_dir = jnp.where(chase_dir == 0, facing, chase_dir)

            # define stop conditions for ranged attackers
            stop_knife = (type_code == ENEMY_KNIFE) & (jnp.abs(player_x - pos_x) < 56)
            stop_dragon = (type_code == ENEMY_DRAGON) & (fsm_state == to_int(STATE_WALK)) & (jnp.abs(player_x - pos_x) < 72)
            stop_gripper = to_bool(cfg.MOD_NO_GRABS) & (type_code == ENEMY_GRIPPER) & (jnp.abs(player_x - pos_x) < 40)
            is_retreating = fsm_state == to_int(STATE_RETREAT)
            hold_position = stop_knife | stop_dragon | stop_gripper | is_retreating

            # tomtom jumping logic
            tomtom_in_range = (type_code == ENEMY_TOMTOM) & (jnp.abs(player_x - pos_x) < 48) & (cooldown <= 0)
            tomtom_grounded = pos_y >= to_int(cfg.FLOOR_Y - cfg.PLAYER_HEIGHT - 2)
            should_jump = tomtom_in_range & tomtom_grounded & (fsm_state == to_int(STATE_WALK))

            state_tomtom = jnp.where(should_jump, to_int(STATE_JUMP_UP), fsm_state)
            vy_tomtom = jnp.where(should_jump & (type_code == ENEMY_TOMTOM), to_int(cfg.TOMTOM_JUMP_VEL), vel_y)
            cd_tomtom = jnp.where(should_jump, to_int(cfg.TOMTOM_JUMP_CD), cooldown - to_int(1))

            in_midair = (type_code == ENEMY_TOMTOM) & (fsm_state != to_int(STATE_WALK))
            updated_vy = jnp.where(in_midair, vy_tomtom + to_int(1), vy_tomtom)
            updated_y_tt = jnp.where(active_flag & in_midair, pos_y + updated_vy, pos_y)
            floor_landed = updated_y_tt >= to_int(cfg.FLOOR_Y - cfg.PLAYER_HEIGHT)

            updated_y_tt = jnp.where(floor_landed, to_int(cfg.FLOOR_Y - cfg.PLAYER_HEIGHT), updated_y_tt)
            updated_vy = jnp.where(floor_landed & (type_code == ENEMY_TOMTOM), to_int(0), updated_vy)
            state_tomtom = jnp.where(floor_landed & (type_code == ENEMY_TOMTOM), to_int(STATE_WALK), state_tomtom)

            # dragon retreat logic when hit
            dragon_timer = jnp.where((type_code == ENEMY_DRAGON) & is_retreating, timer - to_int(1), timer)
            end_retreat = (type_code == ENEMY_DRAGON) & is_retreating & (dragon_timer <= 0)
            state_dragon = jnp.where(end_retreat, to_int(STATE_WALK), state_tomtom)

            final_state = jnp.where(
                type_code == ENEMY_TOMTOM, state_tomtom,
                jnp.where(type_code == ENEMY_DRAGON, state_dragon, fsm_state)
            )
            final_timer = jnp.where(type_code == ENEMY_DRAGON, dragon_timer, timer)
            final_cd = jnp.where(type_code == ENEMY_TOMTOM, cd_tomtom, cooldown)

            travel_dir = jnp.where(
                type_code == ENEMY_VASE, to_int(0),
                jnp.where(
                    type_code == ENEMY_SNAKE, facing,
                    jnp.where(is_retreating, -chase_dir, jnp.where(hold_position, to_int(0), chase_dir))
                )
            )

            vase_drop_vel = jnp.where(type_code == ENEMY_VASE, to_int(2), to_int(0))
            next_x = jnp.where(active_flag & ~in_midair, pos_x + travel_dir * step_dist, pos_x)
            next_y = jnp.where(active_flag, jnp.where(in_midair, updated_y_tt, pos_y + vase_drop_vel), pos_y)
            next_dir = jnp.where(
                active_flag & ~hold_position & (type_code != ENEMY_VASE) & (type_code != ENEMY_SNAKE),
                chase_dir, facing
            )

            out_of_bounds = (next_x < -32) | (next_x > cfg.SCREEN_WIDTH + 32)
            next_active = to_bool(active_flag & ~(active_flag & out_of_bounds))

            return next_x, next_y, updated_vy, next_dir, next_active, final_state, final_timer, final_cd

        # vmap over all enemies at once
        res_x, res_y, res_vy, res_dir, res_active, res_state, res_timer, res_cd = jax.vmap(step_enemy)(
            state.en_type, state.en_x, state.en_y, state.en_vel_y, state.en_dir,
            state.en_active, state.en_hp, state.en_cd, state.en_state, state.en_timer,
        )

        return state.replace(
            en_x=res_x,
            en_y=res_y,
            en_vel_y=res_vy,
            en_dir=res_dir,
            en_active=res_active,
            en_state=res_state,
            en_timer=res_timer,
            en_cd=res_cd,
        )

    # --- Projectile & Spawning Logic ---
    def _spawn_projectile(self, state, ptype, px, py, vx, vy):
        first_free_idx = jnp.argmin(state.pr_active)
        slot_available = ~state.pr_active[first_free_idx]
        return state.replace(
            pr_type=jnp.where(slot_available, state.pr_type.at[first_free_idx].set(ptype), state.pr_type),
            pr_x=jnp.where(slot_available, state.pr_x.at[first_free_idx].set(px), state.pr_x),
            pr_y=jnp.where(slot_available, state.pr_y.at[first_free_idx].set(py), state.pr_y),
            pr_vx=jnp.where(slot_available, state.pr_vx.at[first_free_idx].set(vx), state.pr_vx),
            pr_vy=jnp.where(slot_available, state.pr_vy.at[first_free_idx].set(vy), state.pr_vy),
            pr_active=jnp.where(slot_available, state.pr_active.at[first_free_idx].set(to_bool(True)), state.pr_active),
        )

    def _update_enemy_attacks(self, state):
        cfg = self.consts
        next_key, *worker_keys = jax.random.split(state.key, cfg.MAX_ENEMIES + 1)
        subkeys_tensor = jnp.stack(worker_keys)

        def generate_attacks(type_id, ex, ey, facing, is_active, cd_val, countdown, rng):
            # Knife thrower logic
            can_throw = is_active & (type_id == ENEMY_KNIFE) & (cd_val <= 0) & to_bool(not cfg.MOD_NO_KNIVES)
            throw_high = to_bool(jax.random.uniform(rng) > 0.5)
            knife_proj_type = jnp.where(throw_high, to_int(PROJ_KNIFE_HIGH), to_int(PROJ_KNIFE_LOW))
            knife_launch_y = jnp.where(throw_high, ey + to_int(4), ey + to_int(cfg.PLAYER_HEIGHT - 8))
            knife_vx = jnp.where(can_throw, facing * to_int(cfg.SPD_KNIFE_PROJ), to_int(0))
            knife_py = jnp.where(can_throw, knife_launch_y, to_int(-999))
            knife_px = jnp.where(can_throw, ex, to_int(-999))
            knife_tag = jnp.where(can_throw, knife_proj_type, to_int(PROJ_NONE))

            # Dragon fire logic
            can_breathe = is_active & (type_id == ENEMY_DRAGON) & (cd_val <= 0)
            fire_px = jnp.where(can_breathe, ex, to_int(-999))
            fire_py = jnp.where(can_breathe, to_int(cfg.FLOOR_Y - 6), to_int(-999))
            fire_vx = jnp.where(can_breathe, facing * to_int(cfg.SPD_FIRE_PROJ), to_int(0))
            fire_tag = jnp.where(can_breathe, to_int(PROJ_FIRE), to_int(PROJ_NONE))

            # Exploding ball logic
            ball_detonate = is_active & (type_id == ENEMY_BALL) & (countdown <= 0)
            shrap_vx = jnp.where(ball_detonate, facing * to_int(cfg.SPD_SHRAP_PROJ), to_int(0))
            shrap_px = jnp.where(ball_detonate, ex, to_int(-999))
            shrap_tag = jnp.where(ball_detonate, to_int(PROJ_SHRAPNEL), to_int(PROJ_NONE))

            # Floor 2 Boss (Boomerang thrower)
            can_throw_boomerang = is_active & (type_id == ENEMY_BOSS) & (state.floor == 2) & (cd_val <= 0)
            boom_px = jnp.where(can_throw_boomerang, ex, to_int(-999))
            boom_py = jnp.where(can_throw_boomerang, ey + to_int(6), to_int(-999))
            boom_vx = jnp.where(can_throw_boomerang, facing * to_int(cfg.BOOMERANG_SPEED), to_int(0))
            boom_tag = jnp.where(can_throw_boomerang, to_int(PROJ_BOOMERANG), to_int(PROJ_NONE))

            next_cd = jnp.where(
                can_throw, to_int(cfg.KNIFE_THROW_CD),
                jnp.where(
                    can_breathe, to_int(cfg.DRAGON_FIRE_CD),
                    jnp.where(can_throw_boomerang, to_int(cfg.BOSS_ATTACK_CD), cd_val - to_int(1))
                )
            )
            next_timer = jnp.where(ball_detonate, to_int(cfg.BALL_SPLIT_CD), countdown - to_int(1))
            next_active = to_bool(is_active & ~ball_detonate)

            spawn_candidates = jnp.stack([
                jnp.array([knife_px, knife_py, knife_vx, to_int(0), knife_tag]),
                jnp.array([fire_px, fire_py, fire_vx, to_int(0), fire_tag]),
                jnp.array([boom_px, boom_py, boom_vx, to_int(0), boom_tag]),
                jnp.array([shrap_px, ey, shrap_vx, to_int(-3), shrap_tag]),
                jnp.array([shrap_px, ey, shrap_vx, to_int(0), shrap_tag]),
                jnp.array([shrap_px, ey, shrap_vx, to_int(3), shrap_tag]),
            ])

            return next_cd, next_timer, next_active, spawn_candidates

        new_cds, new_timers, new_actives, raw_candidates = jax.vmap(generate_attacks)(
            state.en_type, state.en_x, state.en_y, state.en_dir,
            state.en_active, state.en_cd, state.en_timer, subkeys_tensor,
        )

        flattened_candidates = raw_candidates.reshape(-1, 5)

        # fold over candidates to add them to the projectile pool safely
        def allocate_slot(carry, candidate):
            tags, xs, ys, vxs, vys, active_flags = carry
            px, py, vx, vy, ptype = candidate[0], candidate[1], candidate[2], candidate[3], candidate[4]
            is_valid = ptype != to_int(PROJ_NONE)
            slot_idx = jnp.argmin(active_flags)
            is_open = ~active_flags[slot_idx]
            should_place = is_valid & is_open

            tags = jnp.where(should_place, tags.at[slot_idx].set(ptype), tags)
            xs = jnp.where(should_place, xs.at[slot_idx].set(px), xs)
            ys = jnp.where(should_place, ys.at[slot_idx].set(py), ys)
            vxs = jnp.where(should_place, vxs.at[slot_idx].set(vx), vxs)
            vys = jnp.where(should_place, vys.at[slot_idx].set(vy), vys)
            active_flags = jnp.where(should_place, active_flags.at[slot_idx].set(to_bool(True)), active_flags)
            return (tags, xs, ys, vxs, vys, active_flags), None

        (out_type, out_x, out_y, out_vx, out_vy, out_act), _ = jax.lax.scan(
            allocate_slot,
            (state.pr_type, state.pr_x, state.pr_y, state.pr_vx, state.pr_vy, state.pr_active),
            flattened_candidates,
        )

        return state.replace(
            en_cd=new_cds,
            en_timer=new_timers,
            en_active=new_actives,
            pr_type=out_type,
            pr_x=out_x,
            pr_y=out_y,
            pr_vx=out_vx,
            pr_vy=out_vy,
            pr_active=out_act,
            key=next_key,
        )

    def _update_projectiles(self, state):
        next_x = state.pr_x + state.pr_vx
        next_y = state.pr_y + state.pr_vy
        next_vy = state.pr_vy + jnp.where(state.pr_type == PROJ_SHRAPNEL, to_int(1), to_int(0))

        out_of_bounds = (
            (next_x < -8) | (next_x > self.consts.SCREEN_WIDTH + 8) | (next_y > self.consts.SCREEN_HEIGHT)
        )
        return state.replace(
            pr_x=next_x,
            pr_y=next_y,
            pr_vy=next_vy,
            pr_active=to_bool(state.pr_active & ~out_of_bounds),
        )

    def _update_vase_transformations(self, state):
        cfg = self.consts
        # turn dropped vases into snakes when they hit the floor
        is_vase = state.en_active & (state.en_type == ENEMY_VASE)
        reached_floor = state.en_y >= to_int(cfg.FLOOR_Y - cfg.PLAYER_HEIGHT)
        transform_mask = is_vase & reached_floor
        transformed_types = jnp.where(transform_mask, to_int(ENEMY_SNAKE), state.en_type)
        return state.replace(en_type=transformed_types)

    def _select_enemy_archetype(self, current_floor, rand_val):
        cfg = self.consts
        # probability mapping for different enemy spawns on each floor
        archetype = jnp.where(
            current_floor <= 1,
            jnp.where(rand_val < 0.55, to_int(ENEMY_GRIPPER), to_int(ENEMY_KNIFE)),
            jnp.where(
                current_floor == 2,
                jnp.where(
                    rand_val < 0.40, to_int(ENEMY_GRIPPER),
                    jnp.where(rand_val < 0.75, to_int(ENEMY_KNIFE), to_int(ENEMY_DRAGON))
                ),
                jnp.where(
                    current_floor <= 4,
                    jnp.where(
                        rand_val < 0.35, to_int(ENEMY_GRIPPER),
                        jnp.where(
                            rand_val < 0.60, to_int(ENEMY_KNIFE),
                            jnp.where(rand_val < 0.80, to_int(ENEMY_TOMTOM), to_int(ENEMY_DRAGON))
                        )
                    ),
                    jnp.where(
                        rand_val < 0.28, to_int(ENEMY_GRIPPER),
                        jnp.where(
                            rand_val < 0.48, to_int(ENEMY_KNIFE),
                            jnp.where(
                                rand_val < 0.63, to_int(ENEMY_TOMTOM),
                                jnp.where(
                                    rand_val < 0.76, to_int(ENEMY_DRAGON),
                                    jnp.where(rand_val < 0.88, to_int(ENEMY_VASE), to_int(ENEMY_BALL))
                                )
                            )
                        )
                    )
                )
            )
        )
        if cfg.MOD_ALL_KNIFE_FLOOR:
            return to_int(ENEMY_KNIFE)
        return archetype

    def _spawn_single_enemy(self, state):
        cfg = self.consts
        next_key, sample_key, pos_key = jax.random.split(state.key, 3)
        rand_scalar = jax.random.uniform(sample_key)
        enemy_type = self._select_enemy_archetype(state.floor, rand_scalar)
        flow_dir = self._get_stage_direction(state.floor)

        spawn_x = jnp.where(flow_dir > 0, to_int(cfg.SCREEN_WIDTH + 4), to_int(-cfg.PLAYER_WIDTH - 4))
        spawn_y = to_int(cfg.FLOOR_Y - cfg.PLAYER_HEIGHT)

        vase_x_pos = to_int(20) + jax.random.randint(pos_key, (), 0, 120).astype(jnp.int32)
        spawn_x = jnp.where(enemy_type == ENEMY_VASE, vase_x_pos, spawn_x)
        spawn_y = jnp.where(enemy_type == ENEMY_VASE, to_int(cfg.CEILING_Y), spawn_y)

        slot_idx = jnp.argmin(state.en_active)
        slot_empty = ~state.en_active[slot_idx]
        allow_spawn = slot_empty & (state.enemies_left > 0) & ~state.boss_spawned

        initial_hp = jnp.where(
            enemy_type == ENEMY_KNIFE, to_int(2),
            jnp.where(enemy_type == ENEMY_DRAGON, to_int(3), to_int(1))
        )
        initial_cd = jnp.where(
            enemy_type == ENEMY_KNIFE, to_int(cfg.KNIFE_THROW_CD // 2),
            jnp.where(enemy_type == ENEMY_DRAGON, to_int(cfg.DRAGON_FIRE_CD // 2), to_int(0))
        )
        initial_timer = jnp.where(enemy_type == ENEMY_BALL, to_int(cfg.BALL_SPLIT_CD), to_int(0))

        return state.replace(
            en_type=jnp.where(allow_spawn, state.en_type.at[slot_idx].set(enemy_type), state.en_type),
            en_x=jnp.where(allow_spawn, state.en_x.at[slot_idx].set(spawn_x), state.en_x),
            en_y=jnp.where(allow_spawn, state.en_y.at[slot_idx].set(spawn_y), state.en_y),
            en_vel_y=jnp.where(allow_spawn, state.en_vel_y.at[slot_idx].set(to_int(0)), state.en_vel_y),
            en_dir=jnp.where(allow_spawn, state.en_dir.at[slot_idx].set(-flow_dir), state.en_dir),
            en_active=jnp.where(allow_spawn, state.en_active.at[slot_idx].set(to_bool(True)), state.en_active),
            en_hp=jnp.where(allow_spawn, state.en_hp.at[slot_idx].set(initial_hp), state.en_hp),
            en_cd=jnp.where(allow_spawn, state.en_cd.at[slot_idx].set(initial_cd), state.en_cd),
            en_state=jnp.where(allow_spawn, state.en_state.at[slot_idx].set(to_int(STATE_WALK)), state.en_state),
            en_timer=jnp.where(allow_spawn, state.en_timer.at[slot_idx].set(initial_timer), state.en_timer),
            enemies_left=jnp.where(allow_spawn, state.enemies_left - to_int(1), state.enemies_left),
            key=next_key,
        )

    def _tick_spawner(self, state):
        cfg = self.consts
        intervals = jnp.array(cfg.SPAWN_INTERVAL, dtype=jnp.int32)
        remaining_ticks = state.spawn_timer - to_int(1)
        trigger_spawn = (remaining_ticks <= 0) & ~state.boss_spawned
        reset_duration = intervals[jnp.clip(state.floor - 1, 0, cfg.NUM_FLOORS - 1)]
        state = state.replace(spawn_timer=jnp.where(trigger_spawn, reset_duration, remaining_ticks))
        return jax.lax.cond(trigger_spawn, lambda s: self._spawn_single_enemy(s), lambda s: s, state)

    def _check_and_spawn_boss(self, state):
        cfg = self.consts
        field_cleared = to_bool(~jnp.any(state.en_active)) & (state.enemies_left <= 0)
        trigger_boss = field_cleared & ~state.boss_spawned

        boss_x_pos = jnp.where(
            self._get_stage_direction(state.floor) > 0,
            to_int(cfg.SCREEN_WIDTH - cfg.PLAYER_WIDTH - 16),
            to_int(16),
        )
        boss_hp_val = self._calculate_boss_health(state.floor, state.loop)

        return state.replace(
            en_type=jnp.where(trigger_boss, state.en_type.at[0].set(to_int(ENEMY_BOSS)), state.en_type),
            en_x=jnp.where(trigger_boss, state.en_x.at[0].set(boss_x_pos), state.en_x),
            en_y=jnp.where(trigger_boss, state.en_y.at[0].set(to_int(cfg.FLOOR_Y - cfg.PLAYER_HEIGHT)), state.en_y),
            en_vel_y=jnp.where(trigger_boss, state.en_vel_y.at[0].set(to_int(0)), state.en_vel_y),
            en_dir=jnp.where(trigger_boss, state.en_dir.at[0].set(to_int(-1)), state.en_dir),
            en_active=jnp.where(trigger_boss, state.en_active.at[0].set(to_bool(True)), state.en_active),
            en_hp=jnp.where(trigger_boss, state.en_hp.at[0].set(boss_hp_val), state.en_hp),
            en_state=jnp.where(trigger_boss, state.en_state.at[0].set(to_int(STATE_WALK)), state.en_state),
            boss_spawned=to_bool(state.boss_spawned | trigger_boss),
        )

    # --- Combat & Collisions ---
    def _resolve_attack_bounds(self, px, py, facing, punching, kicking, crouching, jumping):
        cfg = self.consts
        # figure out which hitbox is active based on the player's pose
        punch_std_x, punch_std_y = px + facing * to_int(cfg.PUNCH_X_OFF), py + to_int(cfg.PUNCH_Y_OFF)
        kick_std_x, kick_std_y = px + facing * to_int(cfg.KICK_X_OFF), py + to_int(cfg.KICK_Y_OFF)
        kick_cr_x, kick_cr_y = px + facing * to_int(cfg.CROUCH_KICK_X_OFF), py + to_int(cfg.CROUCH_KICK_Y_OFF)
        punch_jmp_x, punch_jmp_y = px + facing * to_int(cfg.JUMP_PUNCH_X_OFF), py + to_int(cfg.JUMP_PUNCH_Y_OFF)
        kick_jmp_x, kick_jmp_y = px + facing * to_int(cfg.JUMP_KICK_X_OFF), py + to_int(cfg.JUMP_KICK_Y_OFF)

        crouch_kick = kicking & crouching & ~jumping
        standing_kick = kicking & ~crouching & ~jumping
        jump_punch = punching & jumping
        jump_kick = kicking & jumping
        standing_punch = punching & ~jumping

        box_x = jnp.where(
            standing_punch, punch_std_x,
            jnp.where(
                standing_kick, kick_std_x,
                jnp.where(
                    crouch_kick, kick_cr_x,
                    jnp.where(jump_punch, punch_jmp_x, jnp.where(jump_kick, kick_jmp_x, to_int(0)))
                )
            )
        )
        box_y = jnp.where(
            standing_punch, punch_std_y,
            jnp.where(
                standing_kick, kick_std_y,
                jnp.where(
                    crouch_kick, kick_cr_y,
                    jnp.where(jump_punch, punch_jmp_y, jnp.where(jump_kick, kick_jmp_y, to_int(0)))
                )
            )
        )
        box_w = jnp.where(
            standing_punch, to_int(cfg.PUNCH_W),
            jnp.where(
                standing_kick, to_int(cfg.KICK_W),
                jnp.where(
                    crouch_kick, to_int(cfg.CROUCH_KICK_W),
                    jnp.where(jump_punch, to_int(cfg.JUMP_PUNCH_W), jnp.where(jump_kick, to_int(cfg.JUMP_KICK_W), to_int(0)))
                )
            )
        )
        box_h = jnp.where(
            standing_punch, to_int(cfg.PUNCH_H),
            jnp.where(
                standing_kick, to_int(cfg.KICK_H),
                jnp.where(
                    crouch_kick, to_int(cfg.CROUCH_KICK_H),
                    jnp.where(jump_punch, to_int(cfg.JUMP_PUNCH_H), jnp.where(jump_kick, to_int(cfg.JUMP_KICK_H), to_int(0)))
                )
            )
        )
        return box_x, box_y, box_w, box_h

    def _process_player_combat(self, state, action):
        cfg = self.consts
        _, is_airborne, is_punch, is_kick, is_duck = self._decode_action(action)
        is_striking = to_bool(is_punch | is_kick)

        atk_x, atk_y, atk_w, atk_h = self._resolve_attack_bounds(
            state.player_x, state.player_y, state.player_dir, is_punch, is_kick, is_duck, is_airborne
        )

        def eval_target_hit(enemy_type, enemy_x, enemy_y, is_active, health, fsm_state, timer):
            enemy_width = to_int(cfg.ENEMY_WIDTH)
            enemy_height = jnp.where(enemy_type == ENEMY_SNAKE, to_int(cfg.SNAKE_HEIGHT), to_int(cfg.ENEMY_HEIGHT))
            aligned_y = jnp.where(
                enemy_type == ENEMY_SNAKE,
                to_int(cfg.FLOOR_Y - cfg.SNAKE_HEIGHT),
                enemy_y + to_int((cfg.PLAYER_HEIGHT - cfg.ENEMY_HEIGHT) // 2),
            )
            collided = is_striking & is_active & check_aabb_overlap(
                atk_x, atk_y, atk_w, atk_h, enemy_x, aligned_y, enemy_width, enemy_height
            )
            updated_hp = health - jnp.where(collided, to_int(1), to_int(0))
            is_defeated = to_bool(collided & (updated_hp <= 0))

            # calculate score rewards
            base_points = jnp.where(
                is_punch & is_airborne, to_int(cfg.SC_JUMP_KICK),
                jnp.where(is_kick, to_int(cfg.SC_KICK), to_int(cfg.SC_PUNCH))
            )
            base_points = jnp.where(
                enemy_type == ENEMY_KNIFE,
                jnp.where(is_kick, to_int(cfg.SC_KNIFE_K), to_int(cfg.SC_KNIFE_P)),
                base_points,
            )
            base_points = jnp.where(enemy_type == ENEMY_DRAGON, to_int(cfg.SC_DRAGON), base_points)
            base_points = jnp.where(enemy_type == ENEMY_BALL, to_int(cfg.SC_BALL), base_points)
            base_points = jnp.where(
                enemy_type == ENEMY_VASE,
                jnp.where(is_kick, to_int(cfg.SC_VASE_K), to_int(cfg.SC_VASE_P)),
                base_points,
            )

            boss_payout_table = jnp.array(cfg.SCORE_BOSS, dtype=jnp.int32)
            boss_payout = boss_payout_table[jnp.clip(state.floor, 1, cfg.NUM_FLOORS)]
            awarded_pts = jnp.where(enemy_type == ENEMY_BOSS, boss_payout, base_points)
            awarded_pts = jnp.where(is_defeated, awarded_pts * jnp.clip(state.combo, 1, cfg.COMBO_MAX), to_int(0))

            dragon_damaged = collided & ~is_defeated & (enemy_type == ENEMY_DRAGON)
            updated_fsm = jnp.where(dragon_damaged, to_int(STATE_RETREAT), fsm_state)
            updated_timer = jnp.where(dragon_damaged, to_int(cfg.DRAGON_RETREAT_DUR), timer)

            return collided, is_defeated, updated_hp, awarded_pts, updated_fsm, updated_timer

        hit_flags, kill_flags, post_hp, points_per_enemy, next_fsm, next_timer = jax.vmap(eval_target_hit)(
            state.en_type, state.en_x, state.en_y, state.en_active, state.en_hp, state.en_state, state.en_timer
        )

        turn_reward = jnp.sum(points_per_enemy)
        boss_slain = jnp.any(kill_flags & (state.en_type == ENEMY_BOSS))
        total_kills = jnp.sum(kill_flags).astype(jnp.int32)

        has_kills = total_kills > 0
        new_combo_val = jnp.where(has_kills, jnp.clip(state.combo + total_kills, 0, cfg.COMBO_MAX), state.combo)
        new_combo_duration = jnp.where(has_kills, to_int(cfg.COMBO_RESET_FRAMES), state.combo_timer)

        return state.replace(
            en_active=to_bool(state.en_active & ~kill_flags),
            en_hp=jnp.where(hit_flags, post_hp, state.en_hp),
            en_state=jnp.where(hit_flags & ~kill_flags, next_fsm, state.en_state),
            en_timer=jnp.where(hit_flags & ~kill_flags, next_timer, state.en_timer),
            floor_done=to_bool(state.floor_done | boss_slain),
            combo=new_combo_val,
            combo_timer=new_combo_duration,
        ), turn_reward

    def _resolve_enemy_collisions(self, state):
        cfg = self.consts
        player_x, player_y = state.player_x, state.player_y
        is_ducking = state.is_crouching

        effective_y = jnp.where(is_ducking, player_y + to_int(cfg.CROUCH_H_OFFSET), player_y)
        effective_height = jnp.where(is_ducking, to_int(cfg.PLAYER_HEIGHT - cfg.CROUCH_H_OFFSET), to_int(cfg.PLAYER_HEIGHT))

        def check_enemy_contact(is_active, enemy_type, ex, ey):
            extra_reach = jnp.where((enemy_type == ENEMY_BOSS) & (state.floor == 1), to_int(cfg.BOSS_STICK_RANGE), to_int(0))
            effective_enemy_w = to_int(cfg.ENEMY_WIDTH) + extra_reach
            adjusted_ex = jnp.where(state.en_dir[0] < 0, ex - extra_reach, ex)

            overlapping = is_active & check_aabb_overlap(
                player_x, effective_y, to_int(cfg.PLAYER_WIDTH), effective_height,
                adjusted_ex, ey, effective_enemy_w, to_int(cfg.ENEMY_HEIGHT)
            )
            grab_applied = overlapping & (enemy_type == ENEMY_GRIPPER) & to_bool(not cfg.MOD_NO_GRABS)
            drain_dmg = jnp.where(grab_applied & (state.grab_timer <= 0), to_int(cfg.DMG_GRAB), to_int(0))

            boss_dmg = jnp.where(state.floor >= 3, to_int(cfg.DMG_BOSS + 2), to_int(cfg.DMG_BOSS))
            direct_hit_dmg = (
                jnp.where(overlapping & (enemy_type == ENEMY_BOSS), boss_dmg, to_int(0)) +
                jnp.where(overlapping & (enemy_type == ENEMY_DRAGON), to_int(cfg.DMG_DRAGON), to_int(0)) +
                jnp.where(overlapping & (enemy_type == ENEMY_SNAKE), to_int(cfg.DMG_SNAKE), to_int(0)) +
                jnp.where(overlapping & (enemy_type == ENEMY_BALL), to_int(cfg.DMG_BALL), to_int(0))
            )
            return grab_applied, drain_dmg + direct_hit_dmg

        grabbed_flags, damage_amounts = jax.vmap(check_enemy_contact)(
            state.en_active, state.en_type, state.en_x, state.en_y
        )

        total_contact_damage = jnp.sum(damage_amounts)
        is_any_grabbed = jnp.any(grabbed_flags)
        grabbed_state = to_bool(state.is_grabbed | is_any_grabbed)

        updated_grab_cd = jnp.where(
            grabbed_state,
            jnp.where(state.grab_timer <= 0, to_int(cfg.GRAB_DRAIN_CD), state.grab_timer - to_int(1)),
            to_int(0),
        )

        gripper_still_touching = jnp.any(
            state.en_active & (state.en_type == ENEMY_GRIPPER) &
            (jnp.abs(state.en_x - player_x) < to_int(cfg.PLAYER_WIDTH)) &
            (jnp.abs(state.en_y - player_y) < to_int(cfg.PLAYER_HEIGHT))
        )
        grabbed_state = to_bool(grabbed_state & gripper_still_touching)

        return state.replace(is_grabbed=grabbed_state, grab_timer=updated_grab_cd), total_contact_damage

    def _resolve_projectile_collisions(self, state):
        cfg = self.consts
        player_x, player_y = state.player_x, state.player_y
        is_ducking, is_airborne = state.is_crouching, state.is_jumping

        effective_y = jnp.where(is_ducking, player_y + to_int(cfg.CROUCH_H_OFFSET), player_y)
        effective_height = jnp.where(is_ducking, to_int(cfg.PLAYER_HEIGHT - cfg.CROUCH_H_OFFSET), to_int(cfg.PLAYER_HEIGHT))

        def evaluate_projectile(is_active, proj_type, px, py):
            overlapping = is_active & check_aabb_overlap(
                px, py, to_int(6), to_int(4),
                player_x, effective_y, to_int(cfg.PLAYER_WIDTH), effective_height
            )
            # handle crouching/jumping evasion mechanics
            evaded = jnp.where(
                proj_type == PROJ_KNIFE_HIGH, is_ducking,
                jnp.where(
                    proj_type == PROJ_KNIFE_LOW, is_airborne,
                    jnp.where(
                        proj_type == PROJ_FIRE, is_airborne,
                        jnp.where(
                            proj_type == PROJ_BOOMERANG, is_ducking,
                            jnp.where(proj_type == PROJ_SHRAPNEL, to_bool(is_ducking | is_airborne), to_bool(False))
                        )
                    )
                )
            )
            has_impacted = overlapping & ~evaded
            inflicted_dmg = jnp.where(
                has_impacted,
                jnp.where(
                    proj_type == PROJ_FIRE, to_int(cfg.DMG_FIRE),
                    jnp.where(
                        proj_type == PROJ_SHRAPNEL, to_int(cfg.DMG_SHRAP),
                        jnp.where(proj_type == PROJ_BOOMERANG, to_int(cfg.DMG_KNIFE + 2), to_int(cfg.DMG_KNIFE))
                    )
                ),
                to_int(0)
            )
            return has_impacted, inflicted_dmg

        impact_flags, projectile_damage = jax.vmap(evaluate_projectile)(
            state.pr_active, state.pr_type, state.pr_x, state.pr_y
        )

        total_damage = jnp.sum(projectile_damage)
        return state.replace(pr_active=to_bool(state.pr_active & ~impact_flags)), total_damage

    # --- Floor Progression & Restarts ---
    def _reset_entity_pools(self):
        cfg = self.consts
        zero_enemies = jnp.zeros(cfg.MAX_ENEMIES, dtype=jnp.int32)
        false_enemies = jnp.zeros(cfg.MAX_ENEMIES, dtype=jnp.bool_)
        zero_proj = jnp.zeros(cfg.MAX_PROJ, dtype=jnp.int32)
        false_proj = jnp.zeros(cfg.MAX_PROJ, dtype=jnp.bool_)
        return dict(
            en_type=zero_enemies, en_x=zero_enemies, en_y=zero_enemies,
            en_vel_y=zero_enemies, en_dir=zero_enemies, en_active=false_enemies,
            en_hp=zero_enemies, en_cd=zero_enemies, en_state=zero_enemies, en_timer=zero_enemies,
            pr_type=zero_proj, pr_x=zero_proj, pr_y=zero_proj,
            pr_vx=zero_proj, pr_vy=zero_proj, pr_active=false_proj,
        )

    def _progress_to_next_floor(self, state):
        cfg = self.consts
        time_bonus = state.floor_timer
        target_floor = state.floor + to_int(1)
        cycle_loop = to_bool(target_floor > to_int(cfg.NUM_FLOORS))
        updated_floor = jnp.where(cycle_loop, to_int(1), target_floor)
        updated_loop = state.loop + jnp.where(cycle_loop, to_int(1), to_int(0))

        interval_table = jnp.array(cfg.SPAWN_INTERVAL, dtype=jnp.int32)
        fresh_interval = interval_table[jnp.clip(updated_floor - 1, 0, cfg.NUM_FLOORS - 1)]

        starting_x = jnp.where(
            self._get_stage_direction(updated_floor) > 0,
            to_int(cfg.PLAYER_SPAWN_X),
            to_int(cfg.SCREEN_WIDTH - cfg.PLAYER_WIDTH - cfg.PLAYER_SPAWN_X),
        )

        return state.replace(
            floor=updated_floor,
            loop=updated_loop,
            floor_done=to_bool(False),
            floor_timer=self._get_initial_floor_time(),
            enemies_left=self._calculate_enemy_quota(updated_floor, updated_loop),
            spawn_timer=fresh_interval,
            boss_spawned=to_bool(False),
            transition_timer=to_int(0),
            score=state.score + time_bonus,
            player_x=starting_x,
            player_y=to_int(cfg.FLOOR_Y - cfg.PLAYER_HEIGHT),
            player_vel_y=to_int(0),
            player_energy=to_int(cfg.MAX_ENERGY),
            is_grabbed=to_bool(False),
            is_jumping=to_bool(False),
            is_crouching=to_bool(False),
            grab_timer=to_int(0),
            shake_count=to_int(0),
            combo=to_int(0),
            combo_timer=to_int(0),
            **self._reset_entity_pools(),
        )

    def _handle_player_defeat(self, state):
        cfg = self.consts
        interval_table = jnp.array(cfg.SPAWN_INTERVAL, dtype=jnp.int32)
        fresh_interval = interval_table[jnp.clip(state.floor - 1, 0, cfg.NUM_FLOORS - 1)]

        starting_x = jnp.where(
            self._get_stage_direction(state.floor) > 0,
            to_int(cfg.PLAYER_SPAWN_X),
            to_int(cfg.SCREEN_WIDTH - cfg.PLAYER_WIDTH - cfg.PLAYER_SPAWN_X),
        )

        return state.replace(
            lives=state.lives - to_int(1),
            player_energy=to_int(cfg.MAX_ENERGY),
            floor_timer=self._get_initial_floor_time(),
            enemies_left=self._calculate_enemy_quota(state.floor, state.loop),
            spawn_timer=fresh_interval,
            boss_spawned=to_bool(False),
            transition_timer=to_int(0),
            player_x=starting_x,
            player_y=to_int(cfg.FLOOR_Y - cfg.PLAYER_HEIGHT),
            player_vel_y=to_int(0),
            is_grabbed=to_bool(False),
            is_jumping=to_bool(False),
            is_crouching=to_bool(False),
            grab_timer=to_int(0),
            shake_count=to_int(0),
            combo=to_int(0),
            combo_timer=to_int(0),
            **self._reset_entity_pools(),
        )

    # --- JAX API (Reset / Step / Obs) ---
    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state):
        cfg = self.consts
        player_obs = ObjectObservation.create(
            x=state.player_x,
            y=state.player_y,
            width=jnp.array(cfg.PLAYER_WIDTH),
            height=jnp.array(cfg.PLAYER_HEIGHT),
            active=jnp.array(1, dtype=jnp.int32),
            state=jnp.where(state.is_grabbed, 2, jnp.where(state.is_crouching, 1, 0)).astype(jnp.int32),
            orientation=jnp.where(state.player_dir > 0, 90.0, 270.0).astype(jnp.float32),
        )
        enemy_widths = jnp.full((cfg.MAX_ENEMIES,), cfg.ENEMY_WIDTH, dtype=jnp.int32)
        enemy_heights = jnp.full((cfg.MAX_ENEMIES,), cfg.ENEMY_HEIGHT, dtype=jnp.int32)

        enemies_obs = ObjectObservation.create(
            x=state.en_x,
            y=state.en_y,
            width=enemy_widths,
            height=enemy_heights,
            active=state.en_active.astype(jnp.int32),
            visual_id=state.en_type.astype(jnp.int32),
            state=state.en_state.astype(jnp.int32),
            orientation=jnp.where(state.en_dir > 0, 90.0, 270.0).astype(jnp.float32),
        )
        projectiles_obs = ObjectObservation.create(
            x=state.pr_x,
            y=state.pr_y,
            width=jnp.full((cfg.MAX_PROJ,), 6, dtype=jnp.int32),
            height=jnp.full((cfg.MAX_PROJ,), 4, dtype=jnp.int32),
            active=state.pr_active.astype(jnp.int32),
            visual_id=state.pr_type.astype(jnp.int32),
        )
        return KungFuMasterObservation(
            player=player_obs,
            enemies=enemies_obs,
            projectiles=projectiles_obs,
            score=state.score,
            lives=state.lives,
            floor=state.floor,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state, all_rewards=None):
        return KungFuMasterInfo(score=state.score, floor=state.floor, loop=state.loop)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, prev_state, state):
        return (state.score - prev_state.score).astype(jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state):
        return jnp.logical_or(state.lives <= 0, state.step_count >= self.consts.MAX_STEPS)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key=jax.random.PRNGKey(42)):
        cfg = self.consts
        next_key, init_key = jax.random.split(key)
        zero_enemies = jnp.zeros(cfg.MAX_ENEMIES, dtype=jnp.int32)
        false_enemies = jnp.zeros(cfg.MAX_ENEMIES, dtype=jnp.bool_)
        zero_proj = jnp.zeros(cfg.MAX_PROJ, dtype=jnp.int32)
        false_proj = jnp.zeros(cfg.MAX_PROJ, dtype=jnp.bool_)

        initial_state = KungFuMasterState(
            player_x=to_int(cfg.PLAYER_SPAWN_X),
            player_y=to_int(cfg.FLOOR_Y - cfg.PLAYER_HEIGHT),
            player_vel_y=to_int(0),
            player_dir=to_int(1),
            player_energy=to_int(cfg.MAX_ENERGY),
            is_crouching=to_bool(False),
            is_jumping=to_bool(False),
            is_grabbed=to_bool(False),
            grab_timer=to_int(0),
            shake_last=to_int(0),
            shake_count=to_int(0),
            combo=to_int(0),
            combo_timer=to_int(0),
            en_type=zero_enemies,
            en_x=zero_enemies,
            en_y=zero_enemies,
            en_vel_y=zero_enemies,
            en_dir=zero_enemies,
            en_active=false_enemies,
            en_hp=zero_enemies,
            en_cd=zero_enemies,
            en_state=zero_enemies,
            en_timer=zero_enemies,
            pr_type=zero_proj,
            pr_x=zero_proj,
            pr_y=zero_proj,
            pr_vx=zero_proj,
            pr_vy=zero_proj,
            pr_active=false_proj,
            floor=to_int(1),
            loop=to_int(0),
            floor_done=to_bool(False),
            floor_timer=self._get_initial_floor_time(),
            enemies_left=self._calculate_enemy_quota(to_int(1), to_int(0)),
            spawn_timer=jnp.array(cfg.SPAWN_INTERVAL, dtype=jnp.int32)[0],
            boss_spawned=to_bool(False),
            transition_timer=to_int(0),
            score=to_int(0),
            lives=to_int(cfg.MAX_LIVES),
            extra_life_thr=to_int(cfg.EXTRA_LIFE_PTS),
            step_count=to_int(0),
            key=init_key,
        )
        return self._get_observation(initial_state), initial_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        cfg = self.consts
        native_action = jnp.take(self.ACTION_SET, action.astype(jnp.int32))
        previous_state = state

        # step everything forward
        state = self._update_player_position(state, native_action)
        state = self._update_all_enemies(state)
        state = self._update_vase_transformations(state)
        state = self._update_enemy_attacks(state)
        state = self._update_projectiles(state)
        state = self._tick_spawner(state)

        cleared_quota = jnp.where(cfg.MOD_BOSS_RUSH, to_int(0), state.enemies_left)
        state = state.replace(enemies_left=cleared_quota)
        state = self._check_and_spawn_boss(state)

        # resolve all hitboxes
        state, attack_reward = self._process_player_combat(state, native_action)
        state, melee_dmg = self._resolve_enemy_collisions(state)
        state, ranged_dmg = self._resolve_projectile_collisions(state)

        applied_damage = to_int(0) if cfg.MOD_INFINITE_ENERGY else melee_dmg + ranged_dmg
        remaining_health = jnp.clip(state.player_energy - applied_damage, 0, cfg.MAX_ENERGY)
        state = state.replace(player_energy=remaining_health)

        updated_combo_timer = jnp.where(state.combo_timer > 0, state.combo_timer - to_int(1), to_int(0))
        retained_combo = jnp.where(updated_combo_timer <= 0, to_int(0), state.combo)
        state = state.replace(combo=retained_combo, combo_timer=updated_combo_timer)

        state = state.replace(floor_timer=state.floor_timer - to_int(1))

        player_died = to_bool((remaining_health <= 0) | (state.floor_timer <= 0))
        state = jax.lax.cond(player_died, lambda s: self._handle_player_defeat(s), lambda s: s, state)

        state = state.replace(
            transition_timer=jnp.where(
                state.floor_done & (state.transition_timer == 0),
                to_int(60),
                state.transition_timer,
            )
        )
        state = state.replace(
            transition_timer=jnp.where(
                state.transition_timer > 0,
                state.transition_timer - to_int(1),
                state.transition_timer,
            )
        )

        floor_complete = state.floor_done & (state.transition_timer == 0)
        state = jax.lax.cond(floor_complete, lambda s: self._progress_to_next_floor(s), lambda s: s, state)

        awarded_extra_life = to_bool(state.score + attack_reward >= state.extra_life_thr)
        state = state.replace(
            lives=state.lives + jnp.where(awarded_extra_life, to_int(1), to_int(0)),
            extra_life_thr=jnp.where(
                awarded_extra_life,
                state.extra_life_thr + to_int(cfg.EXTRA_LIFE_PTS),
                state.extra_life_thr,
            ),
        )

        state = state.replace(score=state.score + attack_reward, step_count=state.step_count + to_int(1))

        _, fresh_seed = jax.random.split(state.key)
        state = state.replace(key=fresh_seed)

        terminal = self._get_done(state)
        step_reward = self._get_reward(previous_state, state)
        info_dict = self._get_info(state)
        observation = self._get_observation(state)

        return observation, state, step_reward, terminal, info_dict

    def render(self, state):
        return self.renderer.render(state)

    def action_space(self):
        return spaces.Discrete(len(self.ACTION_SET))

    def observation_space(self):
        cfg = self.consts
        screen_bounds = (cfg.SCREEN_HEIGHT, cfg.SCREEN_WIDTH)
        return spaces.Dict({
            "player": spaces.get_object_space(n=None, screen_size=screen_bounds),
            "enemies": spaces.get_object_space(n=cfg.MAX_ENEMIES, screen_size=screen_bounds),
            "projectiles": spaces.get_object_space(n=cfg.MAX_PROJ, screen_size=screen_bounds),
            "score": spaces.Box(low=0, high=999_999, shape=(), dtype=jnp.int32),
            "lives": spaces.Box(low=0, high=9, shape=(), dtype=jnp.int32),
            "floor": spaces.Box(low=1, high=cfg.NUM_FLOORS, shape=(), dtype=jnp.int32),
        })

    def image_space(self):
        return spaces.Box(low=0, high=255, shape=(210, 160, 3), dtype=jnp.uint8)


# ==========================================
# RENDERER
# ==========================================

# 5x7 digit bitmaps for drawing authentic scores
DIGIT_BITMAPS = jnp.array([
    # 0
    [[1,1,1,1,1],
     [1,0,0,0,1],
     [1,0,0,0,1],
     [1,0,0,0,1],
     [1,0,0,0,1],
     [1,0,0,0,1],
     [1,1,1,1,1]],
    # 1
    [[0,0,1,0,0],
     [0,1,1,0,0],
     [0,0,1,0,0],
     [0,0,1,0,0],
     [0,0,1,0,0],
     [0,0,1,0,0],
     [0,1,1,1,0]],
    # 2
    [[1,1,1,1,1],
     [0,0,0,0,1],
     [0,0,0,0,1],
     [1,1,1,1,1],
     [1,0,0,0,0],
     [1,0,0,0,0],
     [1,1,1,1,1]],
    # 3
    [[1,1,1,1,1],
     [0,0,0,0,1],
     [0,0,0,0,1],
     [1,1,1,1,1],
     [0,0,0,0,1],
     [0,0,0,0,1],
     [1,1,1,1,1]],
    # 4
    [[1,0,0,0,1],
     [1,0,0,0,1],
     [1,0,0,0,1],
     [1,1,1,1,1],
     [0,0,0,0,1],
     [0,0,0,0,1],
     [0,0,0,0,1]],
    # 5
    [[1,1,1,1,1],
     [1,0,0,0,0],
     [1,0,0,0,0],
     [1,1,1,1,1],
     [0,0,0,0,1],
     [0,0,0,0,1],
     [1,1,1,1,1]],
    # 6
    [[1,1,1,1,1],
     [1,0,0,0,0],
     [1,0,0,0,0],
     [1,1,1,1,1],
     [1,0,0,0,1],
     [1,0,0,0,1],
     [1,1,1,1,1]],
    # 7
    [[1,1,1,1,1],
     [0,0,0,0,1],
     [0,0,0,0,1],
     [0,0,0,1,0],
     [0,0,1,0,0],
     [0,1,0,0,0],
     [0,1,0,0,0]],
    # 8
    [[1,1,1,1,1],
     [1,0,0,0,1],
     [1,0,0,0,1],
     [1,1,1,1,1],
     [1,0,0,0,1],
     [1,0,0,0,1],
     [1,1,1,1,1]],
    # 9
    [[1,1,1,1,1],
     [1,0,0,0,1],
     [1,0,0,0,1],
     [1,1,1,1,1],
     [0,0,0,0,1],
     [0,0,0,0,1],
     [1,1,1,1,1]],
], dtype=jnp.bool_)


class KungFuMasterRenderer(JAXGameRenderer):
    def __init__(self, consts=None, config=None):
        self.consts = consts or KungFuMasterConstants()
        super().__init__(self.consts)
        self.config = config or render_utils.RendererConfig(
            game_dimensions=(210, 160), channels=3, downscale=None
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)

        # load pre-extracted sprites from local numpy arrays
        asset_dir = "src/jaxatari/assets/kungfumaster"
        try:
            self.spr_bg = jnp.array(np.load(os.path.join(asset_dir, "background.npy")), dtype=jnp.uint8)
            self.spr_player_stand = jnp.array(np.load(os.path.join(asset_dir, "player_stand.npy")), dtype=jnp.uint8)
            self.spr_player_walk = jnp.array(np.load(os.path.join(asset_dir, "player_walk.npy")), dtype=jnp.uint8)
            self.spr_player_punch = jnp.array(np.load(os.path.join(asset_dir, "player_punch.npy")), dtype=jnp.uint8)
            self.spr_player_kick = jnp.array(np.load(os.path.join(asset_dir, "player_kick.npy")), dtype=jnp.uint8)

            self.spr_gripper = jnp.array(np.load(os.path.join(asset_dir, "gripper.npy")), dtype=jnp.uint8)
            self.spr_knife = jnp.array(np.load(os.path.join(asset_dir, "knife_thrower.npy")), dtype=jnp.uint8)
            self.spr_tomtom = jnp.array(np.load(os.path.join(asset_dir, "tomtom.npy")), dtype=jnp.uint8)
            self.spr_dragon = jnp.array(np.load(os.path.join(asset_dir, "dragon.npy")), dtype=jnp.uint8)
            self.spr_boss = jnp.array(np.load(os.path.join(asset_dir, "boss.npy")), dtype=jnp.uint8)
            
            # Load the timer HUD digits
            self.spr_hud_digits = jnp.array([np.load(os.path.join(asset_dir, f"hud_digit_{i}.npy")) for i in range(10)], dtype=jnp.uint8)
            
            self.has_assets = True
        except Exception:
            self.has_assets = False

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        cfg = self.consts

        # custom fast rectangle fill for jax
        def fill_rectangle(surface, rect_x, rect_y, rect_w, rect_h, rgb_color):
            clamped_x = jnp.clip(rect_x, 0, cfg.SCREEN_WIDTH - 1).astype(jnp.int32)
            clamped_y = jnp.clip(rect_y, 0, cfg.SCREEN_HEIGHT - 1).astype(jnp.int32)
            grid_x = jnp.arange(cfg.SCREEN_WIDTH, dtype=jnp.int32)
            grid_y = jnp.arange(cfg.SCREEN_HEIGHT, dtype=jnp.int32)

            col_mask = (grid_x >= clamped_x) & (grid_x < clamped_x + rect_w)
            row_mask = (grid_y >= clamped_y) & (grid_y < clamped_y + rect_h)
            fill_mask = row_mask[:, None] & col_mask[None, :]
            return jnp.where(fill_mask[..., None], rgb_color, surface)

        # custom alpha blitting using dynamic slices
        def blit_patch(canvas, sprite, x, y, flip=False):
            sprite_to_draw = jnp.where(flip, jnp.fliplr(sprite), sprite)
            h, w, _ = sprite.shape
            x_c = jnp.clip(x, 0, cfg.SCREEN_WIDTH - w).astype(jnp.int32)
            y_c = jnp.clip(y, 0, cfg.SCREEN_HEIGHT - h).astype(jnp.int32)

            alpha = jnp.any(sprite_to_draw > 0, axis=-1, keepdims=True)
            patch = jax.lax.dynamic_slice(canvas, (y_c, x_c, 0), (h, w, 3))
            blended = jnp.where(alpha, sprite_to_draw, patch)
            return jax.lax.dynamic_update_slice(canvas, blended, (y_c, x_c, 0))

        # 1. Base Wall & Backdrop
        fallback_bg = jnp.full((cfg.SCREEN_HEIGHT, cfg.SCREEN_WIDTH, 3), 40, dtype=jnp.uint8)
        fallback_bg = fallback_bg.at[cfg.CEILING_Y:cfg.FLOOR_Y, :, :].set(jnp.array([28, 48, 172], dtype=jnp.uint8))
        framebuffer = jnp.where(self.has_assets, self.spr_bg, fallback_bg)

        # EXACT HUD GRAY: Sampled safely from the middle of the HUD
        hud_bg_color = jnp.where(self.has_assets, self.spr_bg[20, 80], jnp.array([92, 92, 92], dtype=jnp.uint8))

        # Fix 3: Wall pattern detail
        brown_bar = jnp.array([139, 69, 19], dtype=jnp.uint8)
        red_dash = jnp.array([210, 30, 30], dtype=jnp.uint8)
        yellow_dash = jnp.array([220, 200, 40], dtype=jnp.uint8)

        def draw_striped_pillar(p_idx, canvas):
            px = 12 + p_idx * 32
            start_y = cfg.CEILING_Y + 7
            c = fill_rectangle(canvas, px, start_y, 4, cfg.FLOOR_Y - start_y, brown_bar)

            def draw_dashes(step_idx, c_inner):
                base_y = start_y + step_idx * 8
                c_inner = fill_rectangle(c_inner, px, base_y, 3, 2, red_dash)
                c_inner = fill_rectangle(c_inner, px, base_y + 4, 3, 2, yellow_dash)
                return c_inner

            num_steps = (cfg.FLOOR_Y - start_y) // 8
            return jax.lax.fori_loop(0, num_steps, draw_dashes, c)

        framebuffer = jax.lax.fori_loop(0, 5, draw_striped_pillar, framebuffer)

        # Fix 4: Floor border
        floor_color = jnp.array([120, 68, 28], dtype=jnp.uint8)
        red_floor_border = jnp.array([210, 30, 30], dtype=jnp.uint8)
        dark_red_floor_border = jnp.array([110, 15, 15], dtype=jnp.uint8)

        framebuffer = fill_rectangle(framebuffer, 0, cfg.FLOOR_Y - 2, cfg.SCREEN_WIDTH, 2, red_floor_border)
        framebuffer = fill_rectangle(framebuffer, 0, cfg.FLOOR_Y, cfg.SCREEN_WIDTH, 14, floor_color)
        framebuffer = fill_rectangle(framebuffer, 0, cfg.FLOOR_Y + 14, cfg.SCREEN_WIDTH, 2, dark_red_floor_border)

        # Fix 5: Bottom ornamental symbols
        ornament_color = jnp.array([200, 160, 40], dtype=jnp.uint8)
        def draw_bracket(b_idx, canvas):
            bx = 16 + b_idx * 32
            by = 180
            c = fill_rectangle(canvas, bx + 2, by, 3, 12, ornament_color)
            c = fill_rectangle(c, bx, by, 8, 2, ornament_color)
            c = fill_rectangle(c, bx, by + 10, 8, 2, ornament_color)
            return c

        framebuffer = jax.lax.fori_loop(0, 5, draw_bracket, framebuffer)

        # HUD CLEARING
        # Clear baked timer
        framebuffer = fill_rectangle(framebuffer, 28, 8, 52, 12, hud_bg_color)
        # Clear baked green score
        framebuffer = fill_rectangle(framebuffer, 45, 20, 50, 12, hud_bg_color)
        # Clear baked '3' lives text
        framebuffer = fill_rectangle(framebuffer, 85, 28, 25, 14, hud_bg_color)
        # Clear baked energy bars
        framebuffer = fill_rectangle(framebuffer, 48, 32, 85, 20, hud_bg_color)
        # Clear the old tiny baked-in blue life squares above PLAYER
        framebuffer = fill_rectangle(framebuffer, 24, 24, 30, 8, hud_bg_color)

        # HUD DRAWING
        
        # Timer
        def draw_timer_digit(idx, buffer):
            divisor = jnp.array([1000, 100, 10, 1], dtype=jnp.int32)[idx]
            d_val = (jnp.clip(state.floor_timer, 0, 9999) // divisor) % 10
            d_spr = self.spr_hud_digits[d_val]
            return blit_patch(buffer, d_spr, 34 + idx * 8, 8)
        framebuffer = jax.lax.fori_loop(0, 4, draw_timer_digit, framebuffer)

        # Score
        digit_white = jnp.array([255, 255, 255], dtype=jnp.uint8)
        def draw_bitmap_score_digit(d_idx, canvas):
            divisor = jnp.array([100000, 10000, 1000, 100, 10, 1], dtype=jnp.int32)[d_idx]
            d_val = (jnp.clip(state.score, 0, 999999) // divisor) % 10
            glyph = DIGIT_BITMAPS[d_val]
            gx = 70 + d_idx * 7
            gy = 8

            grid_x = jnp.arange(cfg.SCREEN_WIDTH, dtype=jnp.int32)
            grid_y = jnp.arange(cfg.SCREEN_HEIGHT, dtype=jnp.int32)
            within_x = (grid_x >= gx) & (grid_x < gx + 5)
            within_y = (grid_y >= gy) & (grid_y < gy + 7)

            col_offset = jnp.clip(grid_x - gx, 0, 4)
            row_offset = jnp.clip(grid_y - gy, 0, 6)
            pixel_active = glyph[row_offset[:, None], col_offset[None, :]]
            pixel_mask = within_y[:, None] & within_x[None, :] & pixel_active
            return jnp.where(pixel_mask[..., None], digit_white, canvas)

        framebuffer = jax.lax.fori_loop(0, 6, draw_bitmap_score_digit, framebuffer)

        # Life squares
        life_blue = jnp.array([80, 120, 220], dtype=jnp.uint8)
        dark_border = jnp.array([20, 30, 60], dtype=jnp.uint8)
        def draw_life_square(l_idx, canvas):
            lx = 70 + l_idx * 10 
            ly = 20
            is_visible = l_idx < state.lives
            c = fill_rectangle(canvas, lx, ly, 8, 8, dark_border)
            c = fill_rectangle(c, lx + 1, ly + 1, 6, 6, life_blue)
            return jnp.where(is_visible, c, canvas)

        framebuffer = jax.lax.fori_loop(0, 3, draw_life_square, framebuffer)

        # Energy bars
        border_white = jnp.array([255, 255, 255], dtype=jnp.uint8)
        bar_dark_red = jnp.array([120, 20, 20], dtype=jnp.uint8)
        player_yellow = jnp.array([223, 183, 85], dtype=jnp.uint8)
        enemy_red = jnp.array([180, 50, 50], dtype=jnp.uint8)

        # PLAYER Bar
        framebuffer = fill_rectangle(framebuffer, 49, 33, 82, 10, border_white)
        framebuffer = fill_rectangle(framebuffer, 50, 34, 80, 8, bar_dark_red)
        player_bar_w = jnp.clip(
            jnp.round(state.player_energy.astype(jnp.float32) / 100.0 * 80).astype(jnp.int32), 0, 80
        )
        framebuffer = fill_rectangle(framebuffer, 50, 34, player_bar_w, 8, player_yellow)

        # ENEMY Bar
        framebuffer = fill_rectangle(framebuffer, 49, 43, 82, 10, border_white)
        framebuffer = fill_rectangle(framebuffer, 50, 44, 80, 8, bar_dark_red)
        boss_max_hp = jnp.array(cfg.FLOOR_BOSS_HP, dtype=jnp.int32)[jnp.clip(state.floor - 1, 0, cfg.NUM_FLOORS - 1)]
        boss_curr_hp = jnp.where(state.boss_spawned, state.en_hp[0], 0)
        enemy_bar_w = jnp.clip(
            jnp.round(boss_curr_hp.astype(jnp.float32) / boss_max_hp * 80).astype(jnp.int32), 0, 80
        )
        framebuffer = fill_rectangle(framebuffer, 50, 44, enemy_bar_w, 8, enemy_red)

        # ENTITY DRAWING
        
        # Player Animation & Sprites
        walk_frame = (state.step_count // 6) % 2 == 0
        base_spr = jnp.where(walk_frame, self.spr_player_walk, self.spr_player_stand)
        framebuffer = blit_patch(framebuffer, base_spr, state.player_x, state.player_y, flip=(state.player_dir < 0))

        # Enemies
        def draw_single_enemy(idx, buffer):
            pos_x, pos_y = state.en_x[idx], state.en_y[idx]
            tag, is_active = state.en_type[idx], state.en_active[idx]
            facing_left = state.en_dir[idx] < 0

            rendered = jnp.where(
                tag == 1, blit_patch(buffer, self.spr_gripper, pos_x, pos_y, flip=facing_left),
                jnp.where(
                    tag == 2, blit_patch(buffer, self.spr_knife, pos_x, pos_y, flip=facing_left),
                    jnp.where(
                        tag == 3, blit_patch(buffer, self.spr_tomtom, pos_x, pos_y, flip=facing_left),
                        jnp.where(
                            tag == 4, blit_patch(buffer, self.spr_dragon, pos_x, pos_y, flip=facing_left),
                            jnp.where(
                                tag == 9, blit_patch(buffer, self.spr_boss, pos_x, pos_y, flip=facing_left),
                                buffer
                            )
                        )
                    )
                )
            )
            return jnp.where(is_active, rendered, buffer)

        framebuffer = jax.lax.fori_loop(0, cfg.MAX_ENEMIES, draw_single_enemy, framebuffer)

        # Projectiles
        proj_rgb = jnp.array([255, 255, 255], dtype=jnp.uint8)
        def draw_single_proj(idx, buffer):
            rendered = fill_rectangle(buffer, state.pr_x[idx], state.pr_y[idx], 6, 4, proj_rgb)
            return jnp.where(state.pr_active[idx], rendered, buffer)

        framebuffer = jax.lax.fori_loop(0, cfg.MAX_PROJ, draw_single_proj, framebuffer)

        return framebuffer


if __name__ == "__main__":
    print("Running KungFu Master smoke test...")

    env = JaxKungFuMaster()
    master_key = jax.random.PRNGKey(0)

    obs, game_state = env.reset(master_key)
    print(f"Reset OK -> Floor: {int(game_state.floor)}, Lives: {int(game_state.lives)}, Energy: {int(game_state.player_energy)}")

    start_time = time.time()
    obs, game_state, r, is_done, info = env.step(game_state, jnp.int32(3))
    obs, game_state, r, is_done, info = env.step(game_state, jnp.int32(11))
    print(f"JIT Compilation completed in: {time.time() - start_time:.2f}s")

    # quick random policy benchmark
    TOTAL_STEPS = 5000
    current_state = game_state
    accumulated_reward = 0.0
    bench_start = time.time()

    for step_idx in range(TOTAL_STEPS):
        action_sample = jax.random.randint(jax.random.PRNGKey(step_idx), (), 0, 14)
        obs, current_state, r, is_done, _ = env.step(current_state, action_sample)
        accumulated_reward += float(r)
        if is_done:
            obs, current_state = env.reset(jax.random.PRNGKey(step_idx))

    duration = time.time() - bench_start
    print(f"Executed {TOTAL_STEPS} steps in {duration:.2f}s ({TOTAL_STEPS / duration:,.0f} SPS)")
    print(f"Total reward with uniform random policy: {accumulated_reward:.0f}")

    screen_dump = env.render(game_state)
    print(f"Render output verified. Shape: {screen_dump.shape}")

    # make sure vmap doesn't crash
    batch_keys = jax.random.split(master_key, 16)
    obs_batch, state_batch = jax.vmap(env.reset)(batch_keys)
    obs_batch, state_batch, r_batch, d_batch, _ = jax.vmap(env.step)(
        state_batch, jnp.zeros(16, dtype=jnp.int32)
    )
    print(f"Batch vmap check OK -> Batch rewards shape: {r_batch.shape}")

    # test modifiers
    modifier_configs = [
        ("no_knives", {"MOD_NO_KNIVES": True}),
        ("double_speed", {"MOD_DOUBLE_SPEED": True}),
        ("infinite_energy", {"MOD_INFINITE_ENERGY": True}),
        ("one_hit_boss", {"MOD_ONE_HIT_BOSS": True}),
        ("no_grabs", {"MOD_NO_GRABS": True}),
        ("all_knife_floor", {"MOD_ALL_KNIFE_FLOOR": True}),
        ("boss_rush", {"MOD_BOSS_RUSH": True}),
        ("mirror_player", {"MOD_MIRROR_PLAYER": True}),
    ]

    for label, options in modifier_configs:
        variant_env = JaxKungFuMaster(KungFuMasterConstants(**options))
        _, temp_state = variant_env.reset(master_key)
        variant_env.step(temp_state, jnp.int32(3))
        print(f"  Configuration '{label}': OK")

    print("All tests completed successfully.")