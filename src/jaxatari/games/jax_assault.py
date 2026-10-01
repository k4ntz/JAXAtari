import os
from functools import partial
from typing import Tuple
import numpy as np
import jax
import jax.lax
import jax.numpy as jnp
import chex
from flax import struct

import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action, ObjectObservation
from jaxatari.modification import AutoDerivedConstants

WIDTH = 160
HEIGHT = 210

SPEED = 2  # player horizontal speed (px/frame)
# ALE enemies/mothership step ±2 only on alternate frames (~1 px/frame average).
ENEMY_SPEED = 2
ENEMY_MOVE_PERIOD = 2
PLAYER_PROJECTILE_SPEED = 4  # ALE moves 8px every 2 frames
ENEMY_PROJECTILE_SPEED = 4
MOTHERSHIP_Y = 18
PLAYER_Y = 178
PLAYER_X_MIN = 13
PLAYER_X_MAX = 129  # ALE travel limits; 13px left / ~23px right of the 8px sprite
# Vertical bolt: ALE spawns at y=175 and vanishes near y=52 (top of play band).
PLAYER_PROJECTILE_SPAWN_Y = PLAYER_Y - 3  # 175
PLAYER_PROJECTILE_MIN_Y = 52
MAX_HEAT = 9  # ALE heat bar grows 8,12,...,44 (= 8 + 4*heat) then overheats
COOLDOWN_STEPS = 32  # ALE cools by one segment every 32 frames
HEAT_GAIN = 1  # one bar segment per shot
HEAT_BAR_SEGMENT = 4
HEAT_BAR_BASE = 8
MAX_LIVES = 4
LIVES_Y = 193
LIFE_ONE_X = 15
LIFE_OFFSET = 16
SCORE_X = 56
SCORE_Y = 2
SCORE_SPACING = 8
SCORE_MAX_DIGITS = 6
HEAT_BAR_X = 96
HEAT_BAR_Y = 192
HEAT_BAR_WIDTH = HEAT_BAR_BASE + HEAT_BAR_SEGMENT * MAX_HEAT  # 44
HEAT_BAR_HEIGHT = 8
# Early-game ALE fires on a fixed 128-frame cadence (not a high Bernoulli rate).
ENEMY_FIRE_INTERVAL = 128
FIRE_MAX_PROB = 0.02  # retained for stage scaling hooks; gated by ENEMY_FIRE_INTERVAL

# ALE early-game: three enemy slots locked to these Y lanes (no lane hopping).
ENEMY_Y_POSITIONS = (53, 78, 103)
# ALE left/right bounce for crab left-edge (~11 .. ~139).
ENEMY_X_MIN = 11
ENEMY_X_MAX = 139
# ALE mid-screen "sudden change of direction" ~every 64–128 frames.
ENEMY_REVERSE_PERIOD = 128

PLAYER_SIZE = (8, 8)
ENEMY_SIZE = (16, 8)
# Kept for step_counter wrap only; lane descent removed (not ALE early-game).
Y_STEP_DELAY = 70
# ALE drops/spawns a crab about every 16 frames early on.
ENEMY_SPAWN_DELAY = 16
MOTHERSHIP_SIZE = (32, 16)
# ALE awards 21 points for every alien kill (display: 21, 42, 63, ...).
SCORE_PER_KILL = 21
# Manual: bonus cannon every 10000 points, max 4 cannons.
BONUS_LIFE_SCORE = 10000
MAX_BONUS_LIVES_CAP = 4  # max lives/cannons including starting lives


def _get_default_asset_config() -> tuple:
    """Returns the default asset configuration for Assault."""
    return (
        {'name': 'background', 'type': 'background', 'file': 'background.npy'},
        {'name': 'player', 'type': 'single', 'file': 'player.npy'},
        {'name': 'mothership', 'type': 'group', 'files': ['mothership_0.npy', 'mothership_1.npy']},
        {'name': 'enemy', 'type': 'group', 'files': ['enemy_0.npy', 'enemy_1.npy']},
        {'name': 'enemy_tiny', 'type': 'single', 'file': 'enemy_tiny.npy'},
        {'name': 'player_projectile', 'type': 'single', 'file': 'player_projectile.npy'},
        {'name': 'enemy_projectile', 'type': 'single', 'file': 'enemy_projectile.npy'},
        {'name': 'proj_wide', 'type': 'single', 'file': 'proj_wide.npy'},
        {'name': 'proj_sphere', 'type': 'single', 'file': 'proj_sphere.npy'},
        {'name': 'proj_lateral', 'type': 'single', 'file': 'proj_lateral.npy'},
        {'name': 'life', 'type': 'single', 'file': 'life.npy'},
        {'name': 'digits', 'type': 'digits', 'pattern': 'number_{}.npy'},
    )


class AssaultConstants(AutoDerivedConstants):
    WIDTH: int = struct.field(pytree_node=False, default=WIDTH)
    HEIGHT: int = struct.field(pytree_node=False, default=HEIGHT)
    SPEED: int = struct.field(pytree_node=False, default=SPEED)
    ENEMY_SPEED: int = struct.field(pytree_node=False, default=ENEMY_SPEED)
    ENEMY_MOVE_PERIOD: int = struct.field(pytree_node=False, default=ENEMY_MOVE_PERIOD)
    PLAYER_PROJECTILE_SPEED: int = struct.field(pytree_node=False, default=PLAYER_PROJECTILE_SPEED)
    ENEMY_PROJECTILE_SPEED: int = struct.field(pytree_node=False, default=ENEMY_PROJECTILE_SPEED)
    MOTHERSHIP_Y: int = struct.field(pytree_node=False, default=MOTHERSHIP_Y)
    PLAYER_Y: int = struct.field(pytree_node=False, default=PLAYER_Y)
    PLAYER_X_MIN: int = struct.field(pytree_node=False, default=PLAYER_X_MIN)
    PLAYER_X_MAX: int = struct.field(pytree_node=False, default=PLAYER_X_MAX)
    PLAYER_PROJECTILE_SPAWN_Y: int = struct.field(pytree_node=False, default=PLAYER_PROJECTILE_SPAWN_Y)
    PLAYER_PROJECTILE_MIN_Y: int = struct.field(pytree_node=False, default=PLAYER_PROJECTILE_MIN_Y)
    MAX_HEAT: int = struct.field(pytree_node=False, default=MAX_HEAT)
    COOLDOWN_STEPS: int = struct.field(pytree_node=False, default=COOLDOWN_STEPS)
    HEAT_GAIN: int = struct.field(pytree_node=False, default=HEAT_GAIN)
    HEAT_BAR_SEGMENT: int = struct.field(pytree_node=False, default=HEAT_BAR_SEGMENT)
    HEAT_BAR_BASE: int = struct.field(pytree_node=False, default=HEAT_BAR_BASE)
    MAX_LIVES: int = struct.field(pytree_node=False, default=MAX_LIVES)
    LIVES_Y: int = struct.field(pytree_node=False, default=LIVES_Y)
    LIFE_ONE_X: int = struct.field(pytree_node=False, default=LIFE_ONE_X)
    LIFE_OFFSET: int = struct.field(pytree_node=False, default=LIFE_OFFSET)
    SCORE_X: int = struct.field(pytree_node=False, default=SCORE_X)
    SCORE_Y: int = struct.field(pytree_node=False, default=SCORE_Y)
    SCORE_SPACING: int = struct.field(pytree_node=False, default=SCORE_SPACING)
    SCORE_MAX_DIGITS: int = struct.field(pytree_node=False, default=SCORE_MAX_DIGITS)
    HEAT_BAR_X: int = struct.field(pytree_node=False, default=HEAT_BAR_X)
    HEAT_BAR_Y: int = struct.field(pytree_node=False, default=HEAT_BAR_Y)
    HEAT_BAR_WIDTH: int = struct.field(pytree_node=False, default=HEAT_BAR_WIDTH)
    HEAT_BAR_HEIGHT: int = struct.field(pytree_node=False, default=HEAT_BAR_HEIGHT)
    ENEMY_FIRE_INTERVAL: int = struct.field(pytree_node=False, default=ENEMY_FIRE_INTERVAL)
    FIRE_MAX_PROB: float = struct.field(pytree_node=False, default=FIRE_MAX_PROB)
    Y_STEP_DELAY: int = struct.field(pytree_node=False, default=Y_STEP_DELAY)
    ENEMY_SPAWN_DELAY: int = struct.field(pytree_node=False, default=ENEMY_SPAWN_DELAY)
    ENEMY_Y_POSITIONS: Tuple[int, int, int] = struct.field(pytree_node=False, default=ENEMY_Y_POSITIONS)
    ENEMY_X_MIN: int = struct.field(pytree_node=False, default=ENEMY_X_MIN)
    ENEMY_X_MAX: int = struct.field(pytree_node=False, default=ENEMY_X_MAX)
    ENEMY_REVERSE_PERIOD: int = struct.field(pytree_node=False, default=ENEMY_REVERSE_PERIOD)
    PLAYER_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default=PLAYER_SIZE)
    ENEMY_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default=ENEMY_SIZE)
    MOTHERSHIP_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default=MOTHERSHIP_SIZE)
    SCORE_PER_KILL: int = struct.field(pytree_node=False, default=SCORE_PER_KILL)
    BONUS_LIFE_SCORE: int = struct.field(pytree_node=False, default=BONUS_LIFE_SCORE)
    MAX_BONUS_LIVES_CAP: int = struct.field(pytree_node=False, default=MAX_BONUS_LIVES_CAP)
    ASSET_CONFIG: tuple = struct.field(pytree_node=False, default_factory=_get_default_asset_config)


@struct.dataclass
class AssaultState:
    player_x: chex.Array
    player_speed: chex.Array
    enemy_projectile_x: chex.Array
    enemy_projectile_y: chex.Array
    enemy_projectile_dir: chex.Array
    mothership_x: chex.Array
    mothership_dir: chex.Array
    enemy_1_x: chex.Array
    enemy_1_y: chex.Array
    enemy_1_dir: chex.Array
    enemy_1_split: chex.Array
    enemy_2_x: chex.Array
    enemy_2_y: chex.Array
    enemy_2_dir: chex.Array
    enemy_2_split: chex.Array
    enemy_3_x: chex.Array
    enemy_3_y: chex.Array
    enemy_3_dir: chex.Array
    enemy_3_split: chex.Array
    enemy_4_x: chex.Array
    enemy_4_y: chex.Array
    enemy_4_dir: chex.Array
    enemy_5_x: chex.Array
    enemy_5_y: chex.Array
    enemy_5_dir: chex.Array
    enemy_6_x: chex.Array
    enemy_6_y: chex.Array
    enemy_6_dir: chex.Array
    player_projectile_x: chex.Array
    player_projectile_y: chex.Array
    player_projectile_dir: chex.Array
    score: chex.Array
    player_lives: chex.Array
    heat: chex.Array
    stage: chex.Array
    buffer: chex.Array
    occupied_y: chex.Array
    step_counter: chex.Array
    enemies_killed: chex.Array
    current_stage: chex.Array
    enemies_spawned_this_stage: chex.Array
    enemies_invisible: chex.Array
    cooldown: chex.Array
    fired: chex.Array
    enemy_fire_cooldown: chex.Array
    key: chex.PRNGKey


@struct.dataclass
class AssaultObservation:
    player: ObjectObservation
    mothership: ObjectObservation
    enemy_1: ObjectObservation
    enemy_2: ObjectObservation
    enemy_3: ObjectObservation
    enemy_4: ObjectObservation
    enemy_5: ObjectObservation
    enemy_6: ObjectObservation
    enemy_projectile: ObjectObservation
    player_projectile: ObjectObservation
    lives: jnp.ndarray
    score: jnp.ndarray


@struct.dataclass
class AssaultInfo:
    time: jnp.ndarray


class JaxAssault(JaxEnvironment[AssaultState, AssaultObservation, AssaultInfo, AssaultConstants]):
    # Minimal ALE action set for Assault (same order/meanings as ALE/Assault-v5).
    # Vertical shot is UP (not FIRE). FIRE alone is a no-op in the ROM.
    # LEFTFIRE/RIGHTFIRE shoot sideways without moving.
    ACTION_SET: jnp.ndarray = jnp.array(
        [
            Action.NOOP,
            Action.FIRE,
            Action.UP,
            Action.RIGHT,
            Action.LEFT,
            Action.RIGHTFIRE,
            Action.LEFTFIRE,
        ],
        dtype=jnp.int32,
    )

    def __init__(self, consts: AssaultConstants = None):
        consts = consts or AssaultConstants()
        super().__init__(consts)
        self.renderer = AssaultRenderer(self.consts)

    def reset(self, key: chex.PRNGKey = jax.random.PRNGKey(42)) -> Tuple[AssaultObservation, AssaultState]:
        state_key, _step_key = jax.random.split(key)
        state = AssaultState(
            player_x=jnp.array(80, dtype=jnp.int32),
            player_speed=jnp.array(0, dtype=jnp.int32),
            enemy_projectile_x=jnp.array(0, dtype=jnp.int32),
            enemy_projectile_y=jnp.array(-1, dtype=jnp.int32),
            enemy_projectile_dir=jnp.array(0, dtype=jnp.int32),
            mothership_x=jnp.array(64, dtype=jnp.int32),
            mothership_dir=jnp.array(1, dtype=jnp.int32),
            enemy_1_x=jnp.array(-1, dtype=jnp.int32),
            enemy_1_y=jnp.array(HEIGHT + 1, dtype=jnp.int32),
            enemy_1_dir=jnp.array(1, dtype=jnp.int32),
            enemy_1_split=jnp.array(0, dtype=jnp.int32),
            enemy_2_x=jnp.array(-1, dtype=jnp.int32),
            enemy_2_y=jnp.array(HEIGHT + 1, dtype=jnp.int32),
            enemy_2_dir=jnp.array(1, dtype=jnp.int32),
            enemy_2_split=jnp.array(0, dtype=jnp.int32),
            enemy_3_x=jnp.array(-1, dtype=jnp.int32),
            enemy_3_y=jnp.array(HEIGHT + 1, dtype=jnp.int32),
            enemy_3_dir=jnp.array(1, dtype=jnp.int32),
            enemy_3_split=jnp.array(0, dtype=jnp.int32),
            enemy_4_x=jnp.array(-1, dtype=jnp.int32),
            enemy_4_y=jnp.array(HEIGHT + 1, dtype=jnp.int32),
            enemy_4_dir=jnp.array(1, dtype=jnp.int32),
            enemy_5_x=jnp.array(-1, dtype=jnp.int32),
            enemy_5_y=jnp.array(HEIGHT + 1, dtype=jnp.int32),
            enemy_5_dir=jnp.array(1, dtype=jnp.int32),
            enemy_6_x=jnp.array(-1, dtype=jnp.int32),
            enemy_6_y=jnp.array(HEIGHT + 1, dtype=jnp.int32),
            enemy_6_dir=jnp.array(1, dtype=jnp.int32),
            player_projectile_x=jnp.array(-1, dtype=jnp.int32),
            player_projectile_y=jnp.array(-1, dtype=jnp.int32),
            player_projectile_dir=jnp.array(0, dtype=jnp.int32),
            score=jnp.array(0, dtype=jnp.int32),
            player_lives=jnp.array(MAX_LIVES, dtype=jnp.int32),
            heat=jnp.array(0, dtype=jnp.int32),
            stage=jnp.array(1, dtype=jnp.int32),
            buffer=jnp.array(0, dtype=jnp.int32),
            occupied_y=jnp.array([0, 0, 0], dtype=jnp.int32),
            step_counter=jnp.array(0, dtype=jnp.int32),
            enemies_killed=jnp.array(0, dtype=jnp.int32),
            current_stage=jnp.array(0, dtype=jnp.int32),
            enemies_spawned_this_stage=jnp.array(0, dtype=jnp.int32),
            enemies_invisible=jnp.array(0, dtype=jnp.int32),
            cooldown=jnp.array(0, dtype=jnp.int32),
            fired=jnp.array(0, dtype=jnp.int32),
            enemy_fire_cooldown=jnp.array(0, dtype=jnp.int32),
            key=state_key,
        )
        initial_obs = self._get_observation(state)
        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: AssaultState, action: chex.Array) -> Tuple[AssaultObservation, AssaultState, float, bool, AssaultInfo]:
        # Translate compact action to ALE action
        atari_action = jnp.take(self.ACTION_SET, action.astype(jnp.int32))
        
        previous_state = state
        
        # Update game state
        state = self._player_step(state, atari_action)
        state = self._player_projectile_step(state, atari_action)
        state = self._enemy_projectile_step(state)
        state = self._enemy_step(state)
        state = self._mothership_step(state)
        state = self._cooldown_step(state)
        state = self._handle_collisions(state)
        
        # Advance RNG key
        _, next_rng = jax.random.split(state.key)
        state = state.replace(key=next_rng)
        
        done = self._get_done(state)
        reward = self._get_reward(previous_state, state)
        obs = self._get_observation(state)
        info = self._get_info(state)
        
        return obs, state, reward, done, info

    def _player_step(self, state: AssaultState, action: chex.Array) -> AssaultState:
        # ALE: LEFT/RIGHT move; LEFTFIRE/RIGHTFIRE only shoot (no translation).
        move_left = action == Action.LEFT
        move_right = action == Action.RIGHT
        speed = jnp.where(move_left, -SPEED, jnp.where(move_right, SPEED, 0))
        new_x = jnp.clip(state.player_x + speed, PLAYER_X_MIN, PLAYER_X_MAX)
        return state.replace(player_x=new_x, player_speed=speed)

    def _player_projectile_step(self, state: AssaultState, action: chex.Array) -> AssaultState:
        # ALE: UP fires vertically; FIRE alone does nothing; *FIRE are lateral shots.
        fire_up_action = action == Action.UP
        fire_left_action = action == Action.LEFTFIRE
        fire_right_action = action == Action.RIGHTFIRE
        fire_action = fire_up_action + fire_left_action * 2 + fire_right_action * 3
        can_fire = state.player_projectile_y < 0
        fire_action = fire_action * can_fire
        spawn_proj = jnp.logical_and(fire_action > 0, can_fire)
        
        spawn_x = jnp.array([state.player_projectile_x, state.player_x, state.player_x + 12, state.player_x - 4])
        spawn_y = jnp.array([state.player_projectile_y, PLAYER_PROJECTILE_SPAWN_Y, PLAYER_Y, PLAYER_Y])
        new_fired = jnp.where(spawn_proj, jnp.array(1), state.fired)
        new_proj_x = spawn_x[fire_action]
        new_proj_y = spawn_y[fire_action]
        new_proj_dir = jnp.where(spawn_proj, fire_action, state.player_projectile_dir)
        
        moving_y = jnp.logical_and(
            jnp.logical_and(new_proj_y >= PLAYER_PROJECTILE_MIN_Y, new_proj_dir == 1),
            jnp.logical_not(spawn_proj),
        )
        moving_x = jnp.logical_and.reduce(
            jnp.array(
                [
                    new_proj_x >= 0,
                    new_proj_x <= WIDTH,
                    new_proj_dir > 1,
                    jnp.logical_not(spawn_proj),
                ]
            )
        )
        moved_proj_y = jnp.where(moving_y, new_proj_y - PLAYER_PROJECTILE_SPEED, new_proj_y)
        x_dir = jnp.where(new_proj_dir == 2, -1, 1)
        moved_proj_x = jnp.where(moving_x, new_proj_x + x_dir * PLAYER_PROJECTILE_SPEED, new_proj_x)

        # Vertical shots despawn at the top play band; lateral shots at the screen edges.
        vertical_oob = jnp.logical_and(new_proj_dir == 1, moved_proj_y < PLAYER_PROJECTILE_MIN_Y)
        lateral_oob = jnp.logical_and(
            new_proj_dir > 1,
            jnp.logical_or(moved_proj_x < 0, moved_proj_x > WIDTH),
        )
        out_of_bounds = jnp.logical_or(vertical_oob, lateral_oob)
        final_proj_y = jnp.where(out_of_bounds, jnp.array(-1), moved_proj_y)
        final_proj_x = jnp.where(out_of_bounds, jnp.array(-1), moved_proj_x)
        final_proj_dir = jnp.where(out_of_bounds, jnp.array(0), new_proj_dir)
        
        return state.replace(
            player_projectile_x=final_proj_x,
            player_projectile_y=final_proj_y,
            player_projectile_dir=final_proj_dir,
            fired=new_fired
        )

    def _cooldown_step(self, state: AssaultState) -> AssaultState:
        """Heat: +HEAT_GAIN per shot; -1 every COOLDOWN_STEPS while fully idle.

        Cool timer freezes while a bolt is in flight so rapid fire can stack heat
        the way ALE's bar does (one segment per shot, ~32f to cool one segment).
        """
        proj_active = state.player_projectile_y >= 0
        # Freeze countdown during flight; only tick while the gun is clear.
        new_cooldown = jnp.where(
            proj_active,
            state.cooldown,
            jnp.where(state.cooldown > 0, state.cooldown - 1, state.cooldown),
        )
        tick = jnp.logical_and(jnp.logical_not(proj_active), new_cooldown == 0)

        heat_from_shot = jnp.where(state.fired, state.heat + HEAT_GAIN, state.heat)
        new_fired = jnp.where(state.fired, jnp.array(0), state.fired)

        should_cool = jnp.logical_and(tick, jnp.logical_not(state.fired))
        heat_after_cool = jnp.where(should_cool, jnp.maximum(heat_from_shot - 1, 0), heat_from_shot)

        overheat = heat_after_cool > MAX_HEAT
        new_heat = jnp.where(overheat, jnp.array(0), heat_after_cool)
        new_lives = jnp.where(overheat, jnp.maximum(state.player_lives - 1, 0), state.player_lives)
        new_cooldown = jnp.where(
            state.fired,
            jnp.array(COOLDOWN_STEPS, dtype=jnp.int32),
            jnp.where(tick, jnp.array(COOLDOWN_STEPS, dtype=jnp.int32), new_cooldown),
        )
        return state.replace(heat=new_heat, cooldown=new_cooldown, player_lives=new_lives, fired=new_fired)

    def _enemy_projectile_step(self, state: AssaultState) -> AssaultState:
        can_fire = jnp.logical_and(state.enemy_projectile_y < 0, state.enemy_fire_cooldown <= 0)
        
        chosen_enemy_x = jnp.array(0)
        chosen_enemy_y = jnp.array(-1)
        
        # Find enemy with largest y
        for enemy_x, enemy_y in [(state.enemy_1_x, state.enemy_1_y), (state.enemy_2_x, state.enemy_2_y),
                                  (state.enemy_3_x, state.enemy_3_y), (state.enemy_4_x, state.enemy_4_y),
                                  (state.enemy_5_x, state.enemy_5_y), (state.enemy_6_x, state.enemy_6_y)]:
            is_active = enemy_y < HEIGHT
            is_better = jnp.logical_and(is_active, enemy_y > chosen_enemy_y)
            chosen_enemy_x = jnp.where(is_better, enemy_x, chosen_enemy_x)
            chosen_enemy_y = jnp.where(is_better, enemy_y, chosen_enemy_y)
        
        has_active_enemy = chosen_enemy_y >= 0
        # ALE early-game cadence is a fixed interval; later stages fire proportionally sooner.
        stage_scale = 1.0 + 0.25 * state.current_stage.astype(jnp.float32)
        fire_interval = jnp.maximum(
            1, jnp.round(ENEMY_FIRE_INTERVAL / stage_scale).astype(jnp.int32)
        )
        effective_spawn = jnp.logical_and(can_fire, has_active_enemy)
        
        new_proj_x = jnp.where(effective_spawn, chosen_enemy_x + ENEMY_SIZE[0] // 2, state.enemy_projectile_x)
        new_proj_y = jnp.where(effective_spawn, chosen_enemy_y + ENEMY_SIZE[1], state.enemy_projectile_y)
        new_proj_dir = jnp.where(effective_spawn, 1, state.enemy_projectile_dir)
        new_fire_cd = jnp.where(
            effective_spawn,
            fire_interval,
            jnp.maximum(state.enemy_fire_cooldown - 1, 0),
        )
        
        moving = new_proj_y >= 0
        is_special_stage = jnp.equal(jnp.mod(state.current_stage + 1, 4), 0)
        near_player_level = jnp.logical_and(new_proj_y >= PLAYER_Y - 5, new_proj_y <= PLAYER_Y + 5)
        should_track = jnp.logical_and(is_special_stage, near_player_level)
        
        player_direction = jnp.sign(state.player_x - new_proj_x)
        moved_proj_y = jnp.where(jnp.logical_and(moving, should_track), PLAYER_Y, 
                                 new_proj_y + new_proj_dir * ENEMY_PROJECTILE_SPEED)
        moved_proj_x = jnp.where(jnp.logical_and(should_track, moving),
                                 new_proj_x + player_direction * (ENEMY_PROJECTILE_SPEED / 2), new_proj_x)
        
        final_proj_y = jnp.where(moved_proj_y > HEIGHT, -1, moved_proj_y)
        final_proj_x = jnp.where(moved_proj_y > HEIGHT, -1, moved_proj_x)
        final_proj_dir = jnp.where(moved_proj_y > HEIGHT, 0, new_proj_dir)
        
        player_hit = self._check_collision(final_proj_x, final_proj_y, state.player_x, PLAYER_Y, PLAYER_SIZE[0], PLAYER_SIZE[1])
        
        final_proj_y = jnp.where(player_hit, -1, final_proj_y)
        final_proj_x = jnp.where(player_hit, -1, final_proj_x)
        final_proj_dir = jnp.where(player_hit, 0, final_proj_dir)
        new_lives = jnp.where(player_hit, state.player_lives - 1, state.player_lives)
        
        return state.replace(
            enemy_projectile_x=final_proj_x,
            enemy_projectile_y=final_proj_y,
            enemy_projectile_dir=final_proj_dir,
            player_lives=new_lives,
            enemy_fire_cooldown=new_fire_cd,
        )

    def _enemy_step(self, state: AssaultState) -> AssaultState:
        """Early-game ALE: fixed lanes, survivors shuffle down, new crabs only enter at top."""
        allow_lane_tick = jnp.equal(jnp.mod(state.step_counter, ENEMY_SPAWN_DELAY), 0)
        allow_x_move = jnp.equal(jnp.mod(state.step_counter, ENEMY_MOVE_PERIOD), 0)
        can_spawn_more = jnp.less(state.enemies_spawned_this_stage, 10)
        inactive_y = jnp.array(HEIGHT + 1, dtype=jnp.int32)
        top_y = jnp.array(ENEMY_Y_POSITIONS[0], dtype=jnp.int32)
        mid_y = jnp.array(ENEMY_Y_POSITIONS[1], dtype=jnp.int32)
        bot_y = jnp.array(ENEMY_Y_POSITIONS[2], dtype=jnp.int32)

        def move_enemy_x(x, y, dir, phase, linked_enemy_x):
            is_active = jnp.less(y, HEIGHT)
            at_left = jnp.less_equal(x, ENEMY_X_MIN)
            at_right = jnp.greater_equal(x, ENEMY_X_MAX)
            new_dir = jnp.where(at_left, 1, jnp.where(at_right, -1, dir))

            reverse_tick = jnp.equal(
                jnp.mod(state.step_counter + phase, ENEMY_REVERSE_PERIOD), 0
            )
            can_mid_reverse = jnp.logical_not(jnp.logical_or(at_left, at_right))
            new_dir = jnp.where(
                jnp.logical_and(reverse_tick, can_mid_reverse), -new_dir, new_dir
            )

            collision = jnp.logical_not(
                jnp.logical_or(
                    x > linked_enemy_x + ENEMY_SIZE[0] // 2,
                    x < linked_enemy_x - ENEMY_SIZE[0] // 2,
                )
            )
            new_dir = jnp.where(collision, -new_dir, new_dir)

            step = jnp.where(
                jnp.logical_and(is_active, allow_x_move), new_dir * ENEMY_SPEED, 0
            )
            new_x = jnp.where(
                is_active,
                jnp.clip(x + step, ENEMY_X_MIN, ENEMY_X_MAX),
                x,
            )
            new_dir = jnp.where(is_active, new_dir, dir)
            return new_x, new_dir

        # enemy_1=top(53), enemy_2=mid(78), enemy_3=bot(103); 4/5/6 are split pairs.
        e1_x, e1_dir = move_enemy_x(
            state.enemy_1_x, state.enemy_1_y, state.enemy_1_dir, 0,
            jnp.where(state.enemy_4_y < HEIGHT, state.enemy_4_x, WIDTH + 1),
        )
        e2_x, e2_dir = move_enemy_x(
            state.enemy_2_x, state.enemy_2_y, state.enemy_2_dir, 43,
            jnp.where(state.enemy_5_y < HEIGHT, state.enemy_5_x, WIDTH + 1),
        )
        e3_x, e3_dir = move_enemy_x(
            state.enemy_3_x, state.enemy_3_y, state.enemy_3_dir, 86,
            jnp.where(state.enemy_6_y < HEIGHT, state.enemy_6_x, WIDTH + 1),
        )
        e4_x, e4_dir = move_enemy_x(
            state.enemy_4_x, state.enemy_4_y, state.enemy_4_dir, 21,
            jnp.where(state.enemy_1_y < HEIGHT, state.enemy_1_x, WIDTH + 1),
        )
        e5_x, e5_dir = move_enemy_x(
            state.enemy_5_x, state.enemy_5_y, state.enemy_5_dir, 64,
            jnp.where(state.enemy_2_y < HEIGHT, state.enemy_2_x, WIDTH + 1),
        )
        e6_x, e6_dir = move_enemy_x(
            state.enemy_6_x, state.enemy_6_y, state.enemy_6_dir, 107,
            jnp.where(state.enemy_3_y < HEIGHT, state.enemy_3_x, WIDTH + 1),
        )

        e1_y, e1_split = state.enemy_1_y, state.enemy_1_split
        e2_y, e2_split = state.enemy_2_y, state.enemy_2_split
        e3_y, e3_split = state.enemy_3_y, state.enemy_3_split
        e4_y, e5_y, e6_y = state.enemy_4_y, state.enemy_5_y, state.enemy_6_y

        top_active = jnp.less(e1_y, HEIGHT)
        mid_active = jnp.less(e2_y, HEIGHT)
        bot_active = jnp.less(e3_y, HEIGHT)

        # ALE priority: refill top from mothership first; only then shuffle down.
        do_spawn_top = jnp.logical_and.reduce(
            jnp.array(
                [
                    allow_lane_tick,
                    jnp.logical_not(top_active),
                    jnp.logical_not(jnp.less(e4_y, HEIGHT)),
                    can_spawn_more,
                ]
            )
        )
        do_mid_to_bot = jnp.logical_and.reduce(
            jnp.array(
                [
                    allow_lane_tick,
                    jnp.logical_not(do_spawn_top),
                    jnp.logical_not(bot_active),
                    mid_active,
                ]
            )
        )
        do_top_to_mid = jnp.logical_and.reduce(
            jnp.array(
                [
                    allow_lane_tick,
                    jnp.logical_not(do_spawn_top),
                    jnp.logical_not(do_mid_to_bot),
                    jnp.logical_not(mid_active),
                    top_active,
                ]
            )
        )

        # mid → bottom (survivor keeps x/dir; split pair follows).
        e3_x = jnp.where(do_mid_to_bot, e2_x, e3_x)
        e3_y = jnp.where(do_mid_to_bot, bot_y, e3_y)
        e3_dir = jnp.where(do_mid_to_bot, e2_dir, e3_dir)
        e3_split = jnp.where(do_mid_to_bot, e2_split, e3_split)
        e6_x = jnp.where(do_mid_to_bot, e5_x, e6_x)
        e6_y = jnp.where(
            do_mid_to_bot,
            jnp.where(jnp.less(e5_y, HEIGHT), bot_y, inactive_y),
            e6_y,
        )
        e6_dir = jnp.where(do_mid_to_bot, e5_dir, e6_dir)
        e2_x = jnp.where(do_mid_to_bot, jnp.array(-1, dtype=jnp.int32), e2_x)
        e2_y = jnp.where(do_mid_to_bot, inactive_y, e2_y)
        e2_dir = jnp.where(do_mid_to_bot, jnp.array(1, dtype=jnp.int32), e2_dir)
        e2_split = jnp.where(do_mid_to_bot, jnp.array(0, dtype=jnp.int32), e2_split)
        e5_x = jnp.where(do_mid_to_bot, jnp.array(-1, dtype=jnp.int32), e5_x)
        e5_y = jnp.where(do_mid_to_bot, inactive_y, e5_y)
        e5_dir = jnp.where(do_mid_to_bot, jnp.array(1, dtype=jnp.int32), e5_dir)

        # top → middle
        e2_x = jnp.where(do_top_to_mid, e1_x, e2_x)
        e2_y = jnp.where(do_top_to_mid, mid_y, e2_y)
        e2_dir = jnp.where(do_top_to_mid, e1_dir, e2_dir)
        e2_split = jnp.where(do_top_to_mid, e1_split, e2_split)
        e5_x = jnp.where(do_top_to_mid, e4_x, e5_x)
        e5_y = jnp.where(
            do_top_to_mid,
            jnp.where(jnp.less(e4_y, HEIGHT), mid_y, inactive_y),
            e5_y,
        )
        e5_dir = jnp.where(do_top_to_mid, e4_dir, e5_dir)
        e1_x = jnp.where(do_top_to_mid, jnp.array(-1, dtype=jnp.int32), e1_x)
        e1_y = jnp.where(do_top_to_mid, inactive_y, e1_y)
        e1_dir = jnp.where(do_top_to_mid, jnp.array(1, dtype=jnp.int32), e1_dir)
        e1_split = jnp.where(do_top_to_mid, jnp.array(0, dtype=jnp.int32), e1_split)
        e4_x = jnp.where(do_top_to_mid, jnp.array(-1, dtype=jnp.int32), e4_x)
        e4_y = jnp.where(do_top_to_mid, inactive_y, e4_y)
        e4_dir = jnp.where(do_top_to_mid, jnp.array(1, dtype=jnp.int32), e4_dir)

        # mothership drop only into the top lane
        e1_x = jnp.where(do_spawn_top, state.mothership_x, e1_x)
        e1_y = jnp.where(do_spawn_top, top_y, e1_y)
        e1_dir = jnp.where(do_spawn_top, state.mothership_dir, e1_dir)
        e1_split = jnp.where(do_spawn_top, jnp.array(0, dtype=jnp.int32), e1_split)

        new_enemies_spawned_this_stage = state.enemies_spawned_this_stage + do_spawn_top.astype(
            jnp.int32
        )

        # Keep split bodies on their parent's lane when not shuffling.
        e4_y = jnp.where(
            jnp.logical_and(jnp.less(e4_y, HEIGHT), jnp.less(e1_y, HEIGHT)), e1_y, e4_y
        )
        e5_y = jnp.where(
            jnp.logical_and(jnp.less(e5_y, HEIGHT), jnp.less(e2_y, HEIGHT)), e2_y, e5_y
        )
        e6_y = jnp.where(
            jnp.logical_and(jnp.less(e6_y, HEIGHT), jnp.less(e3_y, HEIGHT)), e3_y, e6_y
        )

        occupied_y = jnp.array(
            [
                (jnp.less(e1_y, HEIGHT).astype(jnp.int32) + jnp.less(e4_y, HEIGHT).astype(jnp.int32)),
                (jnp.less(e2_y, HEIGHT).astype(jnp.int32) + jnp.less(e5_y, HEIGHT).astype(jnp.int32)),
                (jnp.less(e3_y, HEIGHT).astype(jnp.int32) + jnp.less(e6_y, HEIGHT).astype(jnp.int32)),
            ],
            dtype=jnp.int32,
        )

        new_step_counter = jnp.mod(state.step_counter + 1, Y_STEP_DELAY * 100000)

        return state.replace(
            enemy_1_x=e1_x,
            enemy_1_y=e1_y,
            enemy_1_dir=e1_dir,
            enemy_1_split=e1_split,
            enemy_2_x=e2_x,
            enemy_2_y=e2_y,
            enemy_2_dir=e2_dir,
            enemy_2_split=e2_split,
            enemy_3_x=e3_x,
            enemy_3_y=e3_y,
            enemy_3_dir=e3_dir,
            enemy_3_split=e3_split,
            enemy_4_x=e4_x,
            enemy_4_y=e4_y,
            enemy_4_dir=e4_dir,
            enemy_5_x=e5_x,
            enemy_5_y=e5_y,
            enemy_5_dir=e5_dir,
            enemy_6_x=e6_x,
            enemy_6_y=e6_y,
            enemy_6_dir=e6_dir,
            occupied_y=occupied_y,
            enemies_spawned_this_stage=new_enemies_spawned_this_stage,
            step_counter=new_step_counter,
        )

    def _mothership_step(self, state: AssaultState) -> AssaultState:
        at_left = jnp.greater_equal(0, state.mothership_x)
        at_right = jnp.greater_equal(state.mothership_x, 160 - MOTHERSHIP_SIZE[0])
        new_dir = jnp.where(at_left, 1, jnp.where(at_right, -1, state.mothership_dir))
        allow_x_move = jnp.equal(jnp.mod(state.step_counter, ENEMY_MOVE_PERIOD), 0)
        step = jnp.where(allow_x_move, new_dir * ENEMY_SPEED, 0)
        new_x = jnp.clip(state.mothership_x + step, 0, 160 - MOTHERSHIP_SIZE[0])
        return state.replace(mothership_x=new_x, mothership_dir=new_dir)

    def _check_collision(self, px, py, ex, ey, ew, eh):
        return jnp.logical_and(
            jnp.logical_and(px >= ex, px < ex + ew),
            jnp.logical_and(py >= ey, py < ey + eh)
        )

    def _handle_collisions(self, state: AssaultState) -> AssaultState:
        occupied_y = state.occupied_y
        
        player_proj_active = jnp.greater_equal(state.player_projectile_y, 0)
        enemy_proj_active = jnp.greater_equal(state.enemy_projectile_y, 0)
        enemy_proj_lateral = jnp.equal(state.enemy_projectile_y, PLAYER_Y)
        
        enemy_proj_horizontal_dir = jnp.sign(state.player_x - state.enemy_projectile_x)
        player_proj_horizontal_dir = jnp.sign(state.player_projectile_dir)
        
        enemy_proj_prev_x = state.enemy_projectile_x - enemy_proj_horizontal_dir * (ENEMY_PROJECTILE_SPEED / 2)
        player_proj_prev_x = state.player_projectile_x - player_proj_horizontal_dir * PLAYER_PROJECTILE_SPEED
        
        prev_distance = player_proj_prev_x - enemy_proj_prev_x
        current_distance = state.player_projectile_x - state.enemy_projectile_x
        prev_sign = jnp.sign(prev_distance)
        current_sign = jnp.sign(current_distance)
        have_intersected = jnp.not_equal(prev_sign, current_sign)
        
        projectiles_intersecting = jnp.logical_and.reduce(jnp.array([
            player_proj_active, enemy_proj_active, enemy_proj_lateral, have_intersected
        ]))
        
        new_player_proj_x = jnp.where(projectiles_intersecting, -1, state.player_projectile_x)
        new_player_proj_y = jnp.where(projectiles_intersecting, -1, state.player_projectile_y)
        new_player_proj_dir = jnp.where(projectiles_intersecting, 0, state.player_projectile_dir)
        
        new_enemy_proj_x = jnp.where(projectiles_intersecting, -1, state.enemy_projectile_x)
        new_enemy_proj_y = jnp.where(projectiles_intersecting, -1, state.enemy_projectile_y)
        new_enemy_proj_dir = jnp.where(projectiles_intersecting, 0, state.enemy_projectile_dir)
        
        def split_condition(stage):
            return stage >= 4
        
        def kill_enemy(arr):
            ex, ey, ew, eh, proj_x, proj_y, occupied_y_val, linked_y = arr
            hit = self._check_collision(proj_x, proj_y, ex, ey, ew, eh)
            matches = jnp.array(ENEMY_Y_POSITIONS) == ey
            has_match = jnp.any(matches)
            idx = jnp.argmax(matches)
            
            new_occupied_y_val = jax.lax.cond(
                jnp.logical_and.reduce(jnp.array([hit, has_match, linked_y > HEIGHT])),
                lambda _: occupied_y_val.at[idx].set(0),
                lambda _: occupied_y_val,
                operand=None
            )
            
            new_ex = jnp.where(hit, -1, ex)
            new_ey = jnp.where(hit, HEIGHT + 1, ey)
            return new_ex, new_ey, hit, new_occupied_y_val
        
        def split_enemy(arr):
            ex, ey, ew, eh, proj_x, proj_y, occupied_y_val, _ = arr
            hit = self._check_collision(proj_x, proj_y, ex, ey, ew, eh)
            new_ex = jnp.where(hit, ex - ENEMY_SIZE[0], ex)
            return new_ex, ey, hit, occupied_y_val
        
        def spawn_enemy(arr):
            ex, ey = arr[:2]
            new_ex = jnp.where(ex + 3 >= WIDTH, WIDTH, ex + ENEMY_SIZE[0])
            new_ey = ey
            matches = jnp.array(ENEMY_Y_POSITIONS) == ey
            idx = jnp.argmax(matches)
            new_occupied_y_val = occupied_y.at[idx].set(occupied_y[idx] + 1)
            return new_ex, new_ey, False, new_occupied_y_val
        
        splitting_enemies = split_condition(state.current_stage)
        
        arg_1 = [state.enemy_1_x, state.enemy_1_y, ENEMY_SIZE[0], ENEMY_SIZE[1],
                 new_player_proj_x, new_player_proj_y, occupied_y, state.enemy_4_y]
        e1_x, e1_y, hit1, occupied_y = jax.lax.cond(
            jnp.logical_and(splitting_enemies, jnp.logical_not(state.enemy_1_split)),
            split_enemy, kill_enemy, operand=arg_1
        )
        e1_split = jnp.where(jnp.logical_and(hit1, e1_y < HEIGHT + 1), 1, state.enemy_1_split)
        
        arg_2 = [state.enemy_2_x, state.enemy_2_y, ENEMY_SIZE[0], ENEMY_SIZE[1],
                 new_player_proj_x, new_player_proj_y, occupied_y, state.enemy_5_y]
        e2_x, e2_y, hit2, occupied_y = jax.lax.cond(
            jnp.logical_and(splitting_enemies, jnp.logical_not(state.enemy_2_split)),
            split_enemy, kill_enemy, operand=arg_2
        )
        e2_split = jnp.where(jnp.logical_and(hit2, e2_y < HEIGHT + 1), 1, state.enemy_2_split)
        
        arg_3 = [state.enemy_3_x, state.enemy_3_y, ENEMY_SIZE[0], ENEMY_SIZE[1],
                 new_player_proj_x, new_player_proj_y, occupied_y, state.enemy_6_y]
        e3_x, e3_y, hit3, occupied_y = jax.lax.cond(
            jnp.logical_and(splitting_enemies, jnp.logical_not(state.enemy_3_split)),
            split_enemy, kill_enemy, operand=arg_3
        )
        e3_split = jnp.where(jnp.logical_and(hit3, e3_y < HEIGHT + 1), 1, state.enemy_3_split)
        was_split = jnp.logical_or.reduce(jnp.array([e1_split, e2_split, e3_split]))
        
        xy4 = jnp.array([state.enemy_4_x, state.enemy_4_y])
        spawn4 = jnp.array([e1_x, e1_y])
        arr4 = jnp.where(jnp.logical_and(splitting_enemies, jnp.logical_and(hit1, was_split)), spawn4, xy4)
        arg_4 = [arr4[0], arr4[1], ENEMY_SIZE[0], ENEMY_SIZE[1],
                 new_player_proj_x, new_player_proj_y, occupied_y, e1_y]
        e4_x, e4_y, hit4, occupied_y = jax.lax.cond(
            jnp.logical_and(hit1, was_split), spawn_enemy, kill_enemy, operand=arg_4
        )
        
        xy5 = jnp.array([state.enemy_5_x, state.enemy_5_y])
        spawn5 = jnp.array([e2_x, e2_y])
        arr5 = jnp.where(jnp.logical_and(splitting_enemies, jnp.logical_and(hit2, was_split)), spawn5, xy5)
        arg_5 = [arr5[0], arr5[1], ENEMY_SIZE[0], ENEMY_SIZE[1],
                 new_player_proj_x, new_player_proj_y, occupied_y, e2_y]
        e5_x, e5_y, hit5, occupied_y = jax.lax.cond(
            jnp.logical_and(hit2, was_split), spawn_enemy, kill_enemy, operand=arg_5
        )
        
        xy6 = jnp.array([state.enemy_6_x, state.enemy_6_y])
        spawn6 = jnp.array([e3_x, e3_y])
        arr6 = jnp.where(jnp.logical_and(splitting_enemies, jnp.logical_and(hit3, was_split)), spawn6, xy6)
        arg_6 = [arr6[0], arr6[1], ENEMY_SIZE[0], ENEMY_SIZE[1],
                 new_player_proj_x, new_player_proj_y, occupied_y, e3_y]
        e6_x, e6_y, hit6, occupied_y = jax.lax.cond(
            jnp.logical_and(hit3, was_split), spawn_enemy, kill_enemy, operand=arg_6
        )
        
        any_hit = hit1 | hit2 | hit3 | hit4 | hit5 | hit6
        new_player_proj_x = jnp.where(any_hit, -1, new_player_proj_x)
        new_player_proj_y = jnp.where(any_hit, -1, new_player_proj_y)
        new_player_proj_dir = jnp.where(any_hit, 0, new_player_proj_dir)
        
        score_incr = (
            hit1.astype(jnp.int32) + hit2.astype(jnp.int32) + hit3.astype(jnp.int32) +
            hit4.astype(jnp.int32) + hit5.astype(jnp.int32) + hit6.astype(jnp.int32)
        ) * SCORE_PER_KILL
        
        enemy_1_killed = jnp.logical_and(hit1, jnp.logical_not(state.enemy_1_split))
        enemy_2_killed = jnp.logical_and(hit2, jnp.logical_not(state.enemy_2_split))
        enemy_3_killed = jnp.logical_and(hit3, jnp.logical_not(state.enemy_3_split))
        
        kills_incr = (enemy_1_killed.astype(jnp.int32) + enemy_2_killed.astype(jnp.int32) +
                      enemy_3_killed.astype(jnp.int32))
        
        new_score = state.score + score_incr
        # Bonus cannon every BONUS_LIFE_SCORE points (manual), capped at MAX_BONUS_LIVES_CAP.
        crossed_bonus = jnp.logical_and(
            jnp.floor_divide(state.score, BONUS_LIFE_SCORE)
            < jnp.floor_divide(new_score, BONUS_LIFE_SCORE),
            score_incr > 0,
        )
        new_lives = jnp.where(
            crossed_bonus,
            jnp.minimum(state.player_lives + 1, MAX_BONUS_LIVES_CAP),
            state.player_lives,
        )
        new_enemies_killed = state.enemies_killed + kills_incr
        all_rows_empty = jnp.array_equal(occupied_y, jnp.array([0, 0, 0]))
        
        stage_complete = jnp.logical_and.reduce(jnp.array([
            jnp.greater(new_enemies_killed, 0),
            jnp.equal(jnp.mod(new_enemies_killed, 10), 0),
            all_rows_empty
        ]))
        new_enemies_killed = jnp.where(stage_complete, 0, new_enemies_killed)
        new_enemies_spawned_this_stage = jnp.where(stage_complete, 0, state.enemies_spawned_this_stage)
        new_current_stage = jnp.where(stage_complete, state.current_stage + 1, state.current_stage)
        
        _, invis_key = jax.random.split(state.key)
        invis_action = jax.lax.cond(
            jnp.logical_and(state.current_stage >= 8, state.current_stage <= 11),
            lambda _: jax.random.uniform(invis_key, shape=()) < 0.01,
            lambda _: jnp.array(False),
            operand=None
        )
        enemies_invisible = jnp.where(invis_action, jnp.logical_not(state.enemies_invisible), state.enemies_invisible)
        enemies_invisible = jnp.where(stage_complete, jnp.array(0), enemies_invisible)
        
        return state.replace(
            player_projectile_x=new_player_proj_x,
            player_projectile_y=new_player_proj_y,
            player_projectile_dir=new_player_proj_dir,
            enemy_projectile_x=jnp.int32(new_enemy_proj_x),
            enemy_projectile_y=jnp.int32(new_enemy_proj_y),
            enemy_projectile_dir=new_enemy_proj_dir,
            enemy_1_x=e1_x, enemy_1_y=e1_y,
            enemy_1_split=jnp.int32(jnp.logical_or(state.enemy_1_split, e1_split)),
            enemy_1_dir=jnp.where(e1_split, -1, state.enemy_1_dir),
            enemy_2_x=e2_x, enemy_2_y=e2_y,
            enemy_2_split=jnp.int32(jnp.logical_or(state.enemy_2_split, e2_split)),
            enemy_2_dir=jnp.where(e2_split, -1, state.enemy_2_dir),
            enemy_3_x=e3_x, enemy_3_y=e3_y,
            enemy_3_split=jnp.int32(jnp.logical_or(state.enemy_3_split, e3_split)),
            enemy_3_dir=jnp.where(e3_split, -1, state.enemy_3_dir),
            enemy_4_x=e4_x, enemy_4_y=e4_y,
            enemy_4_dir=jnp.where(e1_split, 1, state.enemy_4_dir),
            enemy_5_x=e5_x, enemy_5_y=e5_y,
            enemy_5_dir=jnp.where(e2_split, 1, state.enemy_5_dir),
            enemy_6_x=e6_x, enemy_6_y=e6_y,
            enemy_6_dir=jnp.where(e3_split, 1, state.enemy_6_dir),
            score=new_score,
            player_lives=new_lives,
            enemies_killed=new_enemies_killed,
            current_stage=new_current_stage,
            occupied_y=occupied_y,
            enemies_spawned_this_stage=new_enemies_spawned_this_stage,
            enemies_invisible=enemies_invisible,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: AssaultState) -> AssaultObservation:
        def enemy_entity(x, y):
            return ObjectObservation.create(
                x=x, y=y,
                width=jnp.array(ENEMY_SIZE[0]),
                height=jnp.array(ENEMY_SIZE[1]),
            )
        
        player = ObjectObservation.create(
            x=state.player_x, y=jnp.array(PLAYER_Y),
            width=jnp.array(PLAYER_SIZE[0]),
            height=jnp.array(PLAYER_SIZE[1]),
        )
        mothership = ObjectObservation.create(
            x=state.mothership_x, y=jnp.array(MOTHERSHIP_Y),
            width=jnp.array(MOTHERSHIP_SIZE[0]),
            height=jnp.array(MOTHERSHIP_SIZE[1]),
        )
        enemy_projectile = ObjectObservation.create(
            x=state.enemy_projectile_x, y=state.enemy_projectile_y,
            width=jnp.array(2), height=jnp.array(4),
        )
        player_projectile = ObjectObservation.create(
            x=state.player_projectile_x, y=state.player_projectile_y,
            width=jnp.array(2), height=jnp.array(4),
        )
        
        return AssaultObservation(
            player=player,
            mothership=mothership,
            enemy_1=enemy_entity(state.enemy_1_x, state.enemy_1_y),
            enemy_2=enemy_entity(state.enemy_2_x, state.enemy_2_y),
            enemy_3=enemy_entity(state.enemy_3_x, state.enemy_3_y),
            enemy_4=enemy_entity(state.enemy_4_x, state.enemy_4_y),
            enemy_5=enemy_entity(state.enemy_5_x, state.enemy_5_y),
            enemy_6=enemy_entity(state.enemy_6_x, state.enemy_6_y),
            enemy_projectile=enemy_projectile,
            player_projectile=player_projectile,
            lives=state.player_lives,
            score=state.score,
        )

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTION_SET))

    def observation_space(self) -> spaces.Dict:
        object_space = spaces.get_object_space(
            n=None, screen_size=(self.consts.HEIGHT + 1, self.consts.WIDTH), xy_low=-1
        )
        return spaces.Dict({
            "player": object_space,
            "mothership": object_space,
            "enemy_1": object_space,
            "enemy_2": object_space,
            "enemy_3": object_space,
            "enemy_4": object_space,
            "enemy_5": object_space,
            "enemy_6": object_space,
            "enemy_projectile": object_space,
            "player_projectile": object_space,
            "lives": spaces.Box(low=0, high=MAX_LIVES, shape=(), dtype=jnp.int32),
            "score": spaces.Box(low=0, high=jnp.iinfo(jnp.int32).max, shape=(), dtype=jnp.int32),
        })

    def image_space(self) -> spaces.Box:
        return spaces.Box(low=0, high=255, shape=(210, 160, 3), dtype=jnp.uint8)

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: AssaultState) -> AssaultInfo:
        return AssaultInfo(time=state.step_counter)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: AssaultState, state: AssaultState):
        # Match ALE: reward is the score delta (typically +21 per kill).
        return (state.score - previous_state.score).astype(jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: AssaultState) -> bool:
        return jnp.less_equal(state.player_lives, 0)

    def render(self, state: AssaultState) -> jnp.ndarray:
        return self.renderer.render(state)


class AssaultRenderer(JAXGameRenderer):
    def __init__(self, consts: AssaultConstants = None, config: render_utils.RendererConfig = None):
        self.consts = consts or AssaultConstants()
        super().__init__(self.consts)
        
        if config is None:
            self.config = render_utils.RendererConfig(
                game_dimensions=(210, 160),
                channels=3,
                downscale=None
            )
        else:
            self.config = config
        
        self.jr = render_utils.JaxRenderingUtils(self.config)
        
        asset_config = list(self.consts.ASSET_CONFIG)
        
        # Create horizontal projectile sprite (height=1, width=5; ALE lateral bolt ~2–3px)
        player_projectile_sideways = jnp.full((1, 5, 4), 0, dtype=jnp.uint8)
        player_projectile_sideways = player_projectile_sideways.at[:, :, :3].set(236)  # RGB
        player_projectile_sideways = player_projectile_sideways.at[:, :, 3].set(255)  # Alpha
        
        asset_config.append({
            'name': 'player_projectile_sideways',
            'type': 'procedural',
            'data': player_projectile_sideways
        })
        
        local_sprite_path = os.path.join(os.path.dirname(__file__), "sprites", "assault")
        
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(asset_config, local_sprite_path)

        # Stage recoloring must not mutate shared palette entries (mothership reuses
        # some enemy colors). Bake per-stage enemy masks with private color IDs.
        self._init_enemy_stage_masks()

    def _stage_recolor_rgb(self, rgb: Tuple[int, int, int], stage: int) -> Tuple[int, int, int]:
        r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
        if stage == 0:
            return (r, g, b)
        if stage == 1:
            return (g, r, b)
        if stage == 2:
            return (b, g, r)
        return (g, b, r)

    def _remap_mask_colors(self, mask: jnp.ndarray, id_map: dict) -> jnp.ndarray:
        mask_np = np.asarray(mask)
        out = mask_np.copy()
        for src, dst in id_map.items():
            out[mask_np == src] = dst
        return jnp.asarray(out, dtype=mask.dtype)

    def _init_enemy_stage_masks(self) -> None:
        """Build per-stage enemy masks so recoloring does not affect shared sprites."""
        transparent = int(self.jr.TRANSPARENT_ID)
        base_enemy = np.asarray(self.SHAPE_MASKS["enemy"][0])
        base_tiny = np.asarray(self.SHAPE_MASKS["enemy_tiny"])
        base_ids = sorted(
            {int(x) for x in np.unique(np.concatenate([base_enemy.ravel(), base_tiny.ravel()]))}
            - {transparent, 0}
        )

        enemy_masks = []
        tiny_masks = []
        palette = self.PALETTE
        for stage in range(4):
            id_map = {}
            for color_id in base_ids:
                rgb = tuple(int(x) for x in np.asarray(palette[color_id]))
                new_rgb = self._stage_recolor_rgb(rgb, stage)
                if stage == 0 or new_rgb == rgb:
                    id_map[color_id] = color_id
                else:
                    palette, new_id = self.jr.add_palette_color(palette, new_rgb)
                    id_map[color_id] = int(new_id)
            enemy_masks.append(self._remap_mask_colors(base_enemy, id_map))
            tiny_masks.append(self._remap_mask_colors(base_tiny, id_map))

        self.PALETTE = palette
        self.ENEMY_MASKS_BY_STAGE = jnp.stack(enemy_masks)
        self.ENEMY_TINY_MASKS_BY_STAGE = jnp.stack(tiny_masks)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        raster = self.jr.create_object_raster(self.BACKGROUND)
        
        mothership_mask = self.SHAPE_MASKS["mothership"][0]
        raster = self.jr.render_at(raster, state.mothership_x, MOTHERSHIP_Y, mothership_mask)
        
        player_mask = self.SHAPE_MASKS["player"]
        raster = self.jr.render_at(raster, state.player_x, PLAYER_Y, player_mask)

        stage_mod = jnp.mod(state.current_stage, 4).astype(jnp.int32)
        enemy_mask = self.ENEMY_MASKS_BY_STAGE[stage_mod]
        enemy_tiny_mask = self.ENEMY_TINY_MASKS_BY_STAGE[stage_mod]
        
        def render_split_enemy(xy):
            x, y, raster_val = xy
            should_render = jnp.logical_and(y < HEIGHT + 1, jnp.logical_not(state.enemies_invisible))
            return jax.lax.cond(should_render, lambda _: self.jr.render_at(raster_val, x, y, enemy_tiny_mask),
                                lambda _: raster_val, operand=None)
        
        def render_enemy(xy):
            x, y, raster_val = xy
            should_render = jnp.logical_and(y < HEIGHT + 1, jnp.logical_not(state.enemies_invisible))
            return jax.lax.cond(should_render, lambda _: self.jr.render_at(raster_val, x, y, enemy_mask),
                                lambda _: raster_val, operand=None)
        
        def render_tiny_enemy(xy):
            x, y, raster_val = xy
            should_render = jnp.logical_and(y < HEIGHT + 1, jnp.logical_not(state.enemies_invisible))
            return jax.lax.cond(should_render, lambda _: self.jr.render_at(raster_val, x, y, enemy_tiny_mask),
                                lambda _: raster_val, operand=None)
        
        raster = jax.lax.cond(state.enemy_1_split == 1, render_split_enemy, render_enemy,
                              [state.enemy_1_x, state.enemy_1_y, raster])
        raster = jax.lax.cond(state.enemy_2_split == 1, render_split_enemy, render_enemy,
                              [state.enemy_2_x, state.enemy_2_y, raster])
        raster = jax.lax.cond(state.enemy_3_split == 1, render_split_enemy, render_enemy,
                              [state.enemy_3_x, state.enemy_3_y, raster])
        
        raster = render_tiny_enemy([state.enemy_4_x, state.enemy_4_y, raster])
        raster = render_tiny_enemy([state.enemy_5_x, state.enemy_5_y, raster])
        raster = render_tiny_enemy([state.enemy_6_x, state.enemy_6_y, raster])
        
        def render_player_proj(_):
            frame_proj = self.SHAPE_MASKS["player_projectile"]
            return self.jr.render_at(raster, state.player_projectile_x, state.player_projectile_y, frame_proj)
        
        def render_player_proj_sideways_fn(_):
            frame_proj = self.SHAPE_MASKS["player_projectile_sideways"]
            return self.jr.render_at(raster, state.player_projectile_x, PLAYER_Y + 2, frame_proj)
        
        def skip_player_proj(_):
            return raster
        
        raster = jax.lax.cond(
            jnp.logical_and(jnp.greater_equal(state.player_projectile_y, 0),
                            jnp.not_equal(state.player_projectile_y, PLAYER_Y)),
            render_player_proj, skip_player_proj, operand=None
        )
        raster = jax.lax.cond(
            jnp.equal(state.player_projectile_y, PLAYER_Y),
            render_player_proj_sideways_fn, lambda _: raster, operand=None
        )
        
        def render_enemy_proj(_):
            is_stage_4 = jnp.equal(jnp.mod(state.current_stage + 1, 4), 0)
            is_stage_3 = jnp.equal(jnp.mod(state.current_stage + 2, 4), 0)
            is_lateral = jnp.equal(state.enemy_projectile_y, PLAYER_Y)
            
            def stage3_proj(_):
                return self.jr.render_at(raster, state.enemy_projectile_x, state.enemy_projectile_y,
                                         self.SHAPE_MASKS["proj_wide"])
            
            def other_stages(_):
                def stage4_proj(_):
                    def lateral_proj(_):
                        return self.jr.render_at(raster, state.enemy_projectile_x, state.enemy_projectile_y,
                                                 self.SHAPE_MASKS["proj_lateral"])
                    
                    def sphere_proj(_):
                        return self.jr.render_at(raster, state.enemy_projectile_x, state.enemy_projectile_y,
                                                 self.SHAPE_MASKS["proj_sphere"])
                    
                    return jax.lax.cond(is_lateral, lateral_proj, sphere_proj, operand=None)
                
                def standard_proj(_):
                    return self.jr.render_at(raster, state.enemy_projectile_x, state.enemy_projectile_y,
                                             self.SHAPE_MASKS["enemy_projectile"])
                
                return jax.lax.cond(is_stage_4, stage4_proj, standard_proj, operand=None)
            
            return jax.lax.cond(is_stage_3, stage3_proj, other_stages, operand=None)
        
        def skip_enemy_proj(_):
            return raster
        
        raster = jax.lax.cond(
            jnp.greater_equal(state.enemy_projectile_y, 0),
            render_enemy_proj, skip_enemy_proj, operand=None
        )
        
        score_digits = self.jr.int_to_digits(state.score, max_digits=SCORE_MAX_DIGITS)
        raster = self.jr.render_label_selective(
            raster,
            SCORE_X,
            SCORE_Y,
            score_digits,
            self.SHAPE_MASKS["digits"],
            0,
            SCORE_MAX_DIGITS,
            spacing=SCORE_SPACING,
            max_digits_to_render=SCORE_MAX_DIGITS,
        )

        def lives_fn(i, raster_val):
            return self.jr.render_at(raster_val, LIFE_ONE_X + i * LIFE_OFFSET, LIVES_Y, self.SHAPE_MASKS["life"])
        
        raster = jax.lax.fori_loop(0, state.player_lives, lives_fn, raster)
        
        def heat_bar_fn(heat, raster_val):
            # ALE: idle bar is 8px; each heat level adds 4px (8,12,...,44).
            green_rgb = (72, 160, 72)
            black_rgb = (0, 0, 0)
            green_id = self.COLOR_TO_ID.get(green_rgb, 9)
            black_id = self.COLOR_TO_ID.get(black_rgb, 0)
            fill = HEAT_BAR_BASE + HEAT_BAR_SEGMENT * heat
            return self.jr.render_bar(
                raster_val,
                HEAT_BAR_X,
                HEAT_BAR_Y,
                fill,
                HEAT_BAR_WIDTH,
                HEAT_BAR_WIDTH,
                HEAT_BAR_HEIGHT,
                green_id,
                black_id,
            )
        
        raster = heat_bar_fn(state.heat, raster)

        return self.jr.render_from_palette(raster, self.PALETTE)
