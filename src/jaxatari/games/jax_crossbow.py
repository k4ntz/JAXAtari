from jax._src.pjit import JitWrapped
import os
from functools import partial
from typing import NamedTuple, Tuple
import jax
import jax.lax
import jax.numpy as jnp
import chex

import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action

def _get_default_asset_config() -> tuple:
    return (
        {'name': 'background', 'type': 'background', 'file': 'background.npy'},
        {
            'name': 'backgrounds', 'type': 'group',
            'files': [
                'backgrounds/start_screen.npy',
                'backgrounds/get_ready.npy',
                'backgrounds/desert_map.npy',
                'backgrounds/cave_map.npy',
                'backgrounds/forest_map.npy',
                'backgrounds/map_4.npy',
                'backgrounds/demon_map.npy',
                'backgrounds/map_6.npy'
            ]
        },
        {
            'name': 'friend', 'type': 'group',
            'files': [
                'friend/friend_walking_1.npy',
                'friend/friend_walking_2.npy',
                'friend/friend_walking_3.npy'
            ]
        },
        {'name': 'scorpion', 'type': 'single', 'file': 'scorpion.npy'},
        {'name': 'ant', 'type': 'single', 'file': 'ant.npy'},
        {'name': 'vulture', 'type': 'single', 'file': 'vulture.npy'},
        {'name': 'spawn', 'type': 'single', 'file': 'spawn.npy'},
        {'name': 'cursor', 'type': 'single', 'file': 'cursor.npy'},
        {'name': 'enemy', 'type': 'single', 'file': 'enemy.npy'},
        {'name': 'shot', 'type': 'single', 'file': 'shot.npy'},
        {'name': 'digits', 'type': 'digits', 'pattern': 'digits/digit_{}.npy'},
    )

class GamePhase:
    START_SCREEN = 0
    GET_READY = 1
    DESERT_MAP = 2
    CAVE_MAP = 3
    FOREST_MAP = 4
    MAP_4 = 5
    DEMON_MAP = 6
    MAP_6 = 7

class EnemyType:
    GENERIC = 0
    SCORPION = 1
    ANT = 2
    VULTURE = 3
    SPAWN = 4

class CrossbowConstants(NamedTuple):
    WIDTH: int = 160
    HEIGHT: int = 210

    # Gameplay
    CURSOR_SPEED: int = 2
    CURSOR_SIZE: Tuple[int, int] = (4, 4)
    FRIEND_SPEED: float = 0.5
    ENEMY_SPEED: float = 0.5
    MAX_ENEMIES: int = 10
    MAX_LIVES: int = 3
    DYING_DURATION: int = 45

    # Timing
    GET_READY_DURATION: int = 180
    FADE_OUT_DURATION: int = 45
    FADE_IN_DURATION: int = 30

    MAX_SCATTER_PIXELS: int = 100

    # Dimensions
    FRIEND_SIZE: Tuple[int, int] = (8, 27)
    ENEMY_SIZE: Tuple[int, int] = (8, 8)
    SHOT_SIZE: Tuple[int, int] = (6, 6)
    PLAY_AREA_HEIGHT: int = 180

    GROUND_Y_MIN: int = 130
    GROUND_Y_MAX: int = 165

    # Demon Map Spawn Center
    DEMON_MAP_CENTER_X: int = 80
    DEMON_MAP_CENTER_Y: int = 90

    ASSET_CONFIG: tuple = _get_default_asset_config()

def _get_initial_scatter_state(consts: CrossbowConstants):
    num_pixels = consts.MAX_SCATTER_PIXELS
    rel_x = jnp.tile(jnp.arange(0, 8), 13)[:num_pixels]
    rel_y = jnp.repeat(jnp.arange(0, 27, 2), 8)[:num_pixels]
    colors = jnp.concatenate([jnp.full(32, 7), jnp.full(32, 5), jnp.full(36, 6)])[:num_pixels]
    center_x = 4.0; center_y = 13.5
    base_dx = (rel_x - center_x) * 0.3
    base_dy = (rel_y - center_y) * 0.3
    return rel_x, rel_y, base_dx, base_dy, colors

class CrossbowState(NamedTuple):
    cursor_x: chex.Array
    cursor_y: chex.Array
    is_firing: chex.Array
    friend_x: chex.Array
    friend_y: chex.Array
    friend_active: chex.Array
    dying_timer: chex.Array

    get_ready_timer: chex.Array
    fade_in_timer: chex.Array
    selected_target_map: chex.Array

    scatter_px_x: chex.Array
    scatter_px_y: chex.Array
    scatter_px_dx: chex.Array
    scatter_px_dy: chex.Array
    scatter_px_color_idx: chex.Array
    scatter_px_active: chex.Array

    enemies_x: chex.Array
    enemies_y: chex.Array
    enemies_active: chex.Array
    enemies_type: chex.Array

    enemies_dx: chex.Array
    enemies_dy: chex.Array
    enemies_floor_y: chex.Array

    game_phase: chex.Array
    score: chex.Array
    lives: chex.Array
    step_counter: chex.Array
    key: chex.PRNGKey


class CrossbowObservation(NamedTuple):
    cursor_x: jnp.ndarray
    cursor_y: jnp.ndarray
    friend_x: jnp.ndarray
    game_phase: jnp.ndarray
    lives: jnp.ndarray
    score: jnp.ndarray

class CrossbowInfo(NamedTuple):
    time: jnp.ndarray

class JaxCrossbow(JaxEnvironment[CrossbowState, CrossbowObservation, CrossbowInfo, CrossbowConstants]):
    def __init__(self, consts: CrossbowConstants = None):
        consts = consts or CrossbowConstants()
        super().__init__(consts)
        self.renderer = CrossbowRenderer(self.consts)
        self.action_set = Action.get_all_values()

    # ---DEATH LOGIC ---
    def _init_scatter_pixels(self, state: CrossbowState, rng_key: chex.PRNGKey) -> CrossbowState:
        rel_x, rel_y, base_dx, base_dy, colors = _get_initial_scatter_state(self.consts)
        num_pixels = self.consts.MAX_SCATTER_PIXELS
        key_dx, key_dy = jax.random.split(rng_key)
        rand_dx = jax.random.uniform(key_dx, (num_pixels,), minval=-1.5, maxval=1.5)
        rand_dy = jax.random.uniform(key_dy, (num_pixels,), minval=-3.0, maxval=-0.5)
        final_dx = base_dx + rand_dx
        final_dy = base_dy + rand_dy
        return state._replace(
            scatter_px_x=(state.friend_x + rel_x).astype(jnp.float32),
            scatter_px_y=(state.friend_y + rel_y).astype(jnp.float32),
            scatter_px_dx=final_dx,
            scatter_px_dy=final_dy,
            scatter_px_color_idx=colors.astype(jnp.uint8),
            scatter_px_active=jnp.ones(num_pixels, dtype=bool)
        )

    def _update_scatter_pixels(self, state: CrossbowState) -> CrossbowState:
        GRAVITY = 0.15
        new_x = state.scatter_px_x + state.scatter_px_dx
        new_y = state.scatter_px_y + state.scatter_px_dy
        new_dy = state.scatter_px_dy + GRAVITY
        is_active = jnp.logical_and(state.scatter_px_active, jnp.logical_and(jnp.logical_and(new_x >= -5, new_x < self.consts.WIDTH + 5), jnp.logical_and(new_y >= -5, new_y < self.consts.HEIGHT + 5)))
        return state._replace(scatter_px_x=new_x, scatter_px_y=new_y, scatter_px_dy=new_dy, scatter_px_active=is_active)

    def _handle_common_death_logic(self, state: CrossbowState, any_friend_hit: chex.Array, scatter_key: chex.PRNGKey) -> Tuple[CrossbowState, bool]:

        is_dying = state.dying_timer > 0
        state = jax.lax.cond(any_friend_hit, lambda s: self._init_scatter_pixels(s, scatter_key), lambda s: s, state)
        state = jax.lax.cond(is_dying, lambda s: self._update_scatter_pixels(s), lambda s: s, state)
        new_timer = jnp.where(any_friend_hit, self.consts.DYING_DURATION, jnp.maximum(0, state.dying_timer - 1))

        timer_finished = jnp.logical_and(state.dying_timer > 0, new_timer == 0)
        enemies_active_next = jnp.where(timer_finished, jnp.zeros_like(state.enemies_active), state.enemies_active)
        friend_x_next = jnp.where(timer_finished, 20, state.friend_x).astype(jnp.int32)
        scatter_active_next = jnp.where(timer_finished, jnp.zeros_like(state.scatter_px_active), state.scatter_px_active)
        new_lives = jnp.maximum(0, state.lives - jnp.where(any_friend_hit, 1, 0))
        is_game_over = jnp.logical_and(any_friend_hit, state.lives == 0)

        new_state = state._replace(
            lives=new_lives.astype(jnp.int32),
            dying_timer=new_timer.astype(jnp.int32),
            friend_x=friend_x_next,
            enemies_active=enemies_active_next,
            scatter_px_active=scatter_active_next
        )
        return new_state, is_game_over

    def _cursor_step(self, state: CrossbowState, action: chex.Array) -> CrossbowState:
        is_up = jnp.isin(action, jnp.array([Action.UP, Action.UPRIGHT, Action.UPLEFT, Action.UPFIRE, Action.UPRIGHTFIRE, Action.UPLEFTFIRE]))
        is_down = jnp.isin(action, jnp.array([Action.DOWN, Action.DOWNRIGHT, Action.DOWNLEFT, Action.DOWNFIRE, Action.DOWNRIGHTFIRE, Action.DOWNLEFTFIRE]))
        is_left = jnp.isin(action, jnp.array([Action.LEFT, Action.UPLEFT, Action.DOWNLEFT, Action.LEFTFIRE, Action.UPLEFTFIRE, Action.DOWNLEFTFIRE]))
        is_right = jnp.isin(action, jnp.array([Action.RIGHT, Action.UPRIGHT, Action.DOWNRIGHT, Action.RIGHTFIRE, Action.UPRIGHTFIRE, Action.DOWNLEFTFIRE]))

        dx = jnp.where(is_left, -self.consts.CURSOR_SPEED, 0)
        dx = jnp.where(is_right, self.consts.CURSOR_SPEED, dx)
        dy = jnp.where(is_up, -self.consts.CURSOR_SPEED, 0)
        dy = jnp.where(is_down, self.consts.CURSOR_SPEED, dy)

        new_x = jnp.clip(state.cursor_x + dx, 13, self.consts.WIDTH - 5 - self.consts.CURSOR_SIZE[0])
        new_y = jnp.clip(state.cursor_y + dy, 18, self.consts.PLAY_AREA_HEIGHT - self.consts.CURSOR_SIZE[1])

        is_fire = jnp.isin(action, jnp.array([
            Action.FIRE, Action.UPFIRE, Action.RIGHTFIRE, Action.LEFTFIRE,
            Action.DOWNFIRE, Action.UPRIGHTFIRE, Action.UPLEFTFIRE,
            Action.DOWNRIGHTFIRE, Action.DOWNLEFTFIRE
        ]))
        return state._replace(cursor_x=new_x.astype(jnp.int32), cursor_y=new_y.astype(jnp.int32), is_firing=is_fire)

    def _friend_step(self, state: CrossbowState) -> CrossbowState:
        is_dying = state.dying_timer > 0
        should_move = state.step_counter % 8 == 0
        new_x = state.friend_x + jnp.where(should_move & state.friend_active, 1, 0)
        reached_goal = new_x > self.consts.WIDTH
        final_x = jnp.where(is_dying, state.friend_x, jnp.where(reached_goal, 0, new_x))
        return state._replace(friend_x=final_x.astype(jnp.int32))

    def _demon_map_logic(self, state: CrossbowState, action: chex.Array) -> Tuple[CrossbowState, bool]:
        rng, spawn_key, physics_key, scatter_key, floor_key = jax.random.split(state.key, 5)
        is_dying = state.dying_timer > 0
        LEFT_WALL, RIGHT_WALL = 13, self.consts.WIDTH - 15
        WALK_SPEED, CENTER_X = 0.8, self.consts.WIDTH // 2

        # --- COMBAT LOGIC ---
        HIT_TOLERANCE = 8
        cx, cy = state.cursor_x, state.cursor_y
        ex, ey = state.enemies_x, state.enemies_y
        hit_x = jnp.logical_and(cx < ex + self.consts.ENEMY_SIZE[0] + HIT_TOLERANCE, cx + self.consts.CURSOR_SIZE[0] > ex - HIT_TOLERANCE)
        hit_y = jnp.logical_and(cy < ey + self.consts.ENEMY_SIZE[1] + HIT_TOLERANCE, cy + self.consts.CURSOR_SIZE[1] > ey - HIT_TOLERANCE)
        is_hit = jnp.logical_and(hit_x, hit_y)
        valid_kill = jnp.logical_and(state.is_firing, jnp.logical_and(state.enemies_active, is_hit))
        surviving_enemies = jnp.logical_and(state.enemies_active, jnp.logical_not(valid_kill))

        # --- MOVEMENT ---
        is_walking = (state.enemies_dy == 0.0)
        current_dy = jnp.where(is_walking, 0.0, state.enemies_dy + 0.05)
        new_y = state.enemies_y + current_dy

        # Land on individualized floor positions
        on_ground = new_y >= state.enemies_floor_y
        final_y = jnp.where(on_ground, state.enemies_floor_y, new_y)
        final_dy = jnp.where(on_ground, 0.0, current_dy)

        hit_wall = jnp.logical_or(state.enemies_x <= LEFT_WALL, state.enemies_x >= RIGHT_WALL)
        just_landed = jnp.logical_and(jnp.logical_not(is_walking), jnp.logical_or(on_ground, hit_wall))
        landed_dx = jnp.where(state.enemies_x < CENTER_X, WALK_SPEED, -WALK_SPEED)
        final_dx = jnp.where(just_landed, landed_dx, state.enemies_dx)
        new_x = state.enemies_x + final_dx

        vanished = jnp.logical_and(is_walking, jnp.logical_or(
            jnp.logical_and(final_dx > 0, new_x >= RIGHT_WALL),
            jnp.logical_and(final_dx < 0, new_x <= LEFT_WALL)
        ))

        # --- SPAWN LOGIC ---
        spawn_chance = jax.random.uniform(spawn_key, shape=(self.consts.MAX_ENEMIES,)) < 0.03
        should_spawn = jnp.logical_and(jnp.logical_not(surviving_enemies), spawn_chance)

        # Mark Landing Area: randomize floor_y between 130 and 165
        new_floor_y = jax.random.uniform(floor_key, (self.consts.MAX_ENEMIES,), minval=130.0, maxval=165.0)

        spawn_x = jnp.full((self.consts.MAX_ENEMIES,), float(self.consts.DEMON_MAP_CENTER_X))
        spawn_y = jnp.full((self.consts.MAX_ENEMIES,), float(self.consts.DEMON_MAP_CENTER_Y))
        rand_vx = jax.random.uniform(physics_key, (self.consts.MAX_ENEMIES,), minval=-2.5, maxval=2.5)
        rand_vy = jax.random.uniform(physics_key, (self.consts.MAX_ENEMIES,), minval=-2.0, maxval=-0.5)

        enemies_active_next = jnp.logical_or(surviving_enemies, should_spawn)
        final_active = jnp.logical_and(enemies_active_next, jnp.logical_not(vanished))
        final_active = jnp.logical_and(final_active, final_y < self.consts.PLAY_AREA_HEIGHT)

        final_x_state = jnp.where(should_spawn, spawn_x, new_x)
        final_y_state = jnp.where(should_spawn, spawn_y, final_y)
        final_dx_state = jnp.where(should_spawn, rand_vx, final_dx)
        final_dy_state = jnp.where(should_spawn, rand_vy, final_dy)
        final_floor_y_state = jnp.where(should_spawn, new_floor_y, state.enemies_floor_y)

        # --- COLLISION ---
        fx, fy = state.friend_x, state.friend_y
        danger_x = jnp.logical_and(final_x_state < fx + self.consts.FRIEND_SIZE[0], final_x_state + self.consts.ENEMY_SIZE[0] > fx)
        danger_y = jnp.logical_and(final_y_state < fy + self.consts.FRIEND_SIZE[1], final_y_state + self.consts.ENEMY_SIZE[1] > fy)
        friend_hit = jnp.any(jnp.logical_and(jnp.logical_and(danger_x, danger_y), surviving_enemies))
        any_friend_hit = jnp.logical_and(jnp.logical_and(friend_hit, state.friend_active), jnp.logical_not(is_dying))

        intermediate_state = state._replace(
            score=(state.score + jnp.sum(valid_kill) * 3).astype(jnp.int32),
            enemies_active=final_active,
            enemies_x=final_x_state.astype(jnp.float32),
            enemies_y=final_y_state.astype(jnp.float32),
            enemies_dx=final_dx_state,
            enemies_dy=final_dy_state,
            enemies_floor_y=final_floor_y_state,
            enemies_type=jnp.where(should_spawn, EnemyType.SPAWN, state.enemies_type).astype(jnp.int32),
            key=rng
        )
        return self._handle_common_death_logic(intermediate_state, any_friend_hit, scatter_key)

    def _desert_map_logic(self, state: CrossbowState, action: chex.Array) -> Tuple[CrossbowState, bool]:
        rng, spawn_key, type_key, scatter_key = jax.random.split(state.key, 4)
        HIT_TOLERANCE = 8
        cx, cy = state.cursor_x, state.cursor_y
        ex, ey = state.enemies_x, state.enemies_y
        hit_x = jnp.logical_and(cx < ex + self.consts.ENEMY_SIZE[0] + HIT_TOLERANCE, cx + self.consts.CURSOR_SIZE[0] > ex - HIT_TOLERANCE)
        hit_y = jnp.logical_and(cy < ey + self.consts.ENEMY_SIZE[1] + HIT_TOLERANCE, cy + self.consts.CURSOR_SIZE[1] > ey - HIT_TOLERANCE)
        is_hit = jnp.logical_and(hit_x, hit_y)
        valid_kill = jnp.logical_and(state.is_firing, jnp.logical_and(state.enemies_active, is_hit))
        surviving_enemies = jnp.logical_and(state.enemies_active, jnp.logical_not(valid_kill))

        move_freq = jnp.select([state.enemies_type == EnemyType.ANT, state.enemies_type == EnemyType.VULTURE], [2, 3], default=4)
        should_move = (state.step_counter % move_freq) == 0
        new_x = state.enemies_x + jnp.where(should_move, -1.0, 0.0)
        new_y = state.enemies_y + jnp.where(jnp.logical_and(state.enemies_type == EnemyType.VULTURE, should_move), 1.0, 0.0)

        fx, fy = state.friend_x, state.friend_y
        danger_x = jnp.logical_and(new_x < fx + self.consts.FRIEND_SIZE[0], new_x + self.consts.ENEMY_SIZE[0] > fx)
        danger_y = jnp.logical_and(new_y < fy + self.consts.FRIEND_SIZE[1] + 15, new_y + self.consts.ENEMY_SIZE[1] > fy - 5)
        friend_hit = jnp.any(jnp.logical_and(jnp.logical_and(danger_x, danger_y), surviving_enemies))
        any_friend_hit = jnp.logical_and(friend_hit, state.dying_timer == 0)

        spawn_chance = jax.random.uniform(spawn_key, shape=(self.consts.MAX_ENEMIES,)) < 0.03
        should_spawn = jnp.logical_and(jnp.logical_not(surviving_enemies), spawn_chance)
        new_types = jax.random.randint(type_key, (self.consts.MAX_ENEMIES,), 1, 4)
        spawn_x = jax.random.randint(spawn_key, (self.consts.MAX_ENEMIES,), 140, 160).astype(jnp.float32)
        spawn_ground_y = jax.random.randint(spawn_key, (self.consts.MAX_ENEMIES,), self.consts.GROUND_Y_MIN, self.consts.GROUND_Y_MAX).astype(jnp.float32)
        spawn_air_y = jax.random.randint(spawn_key, (self.consts.MAX_ENEMIES,), 20, 80).astype(jnp.float32)
        spawn_y = jnp.where(new_types == EnemyType.VULTURE, spawn_air_y, spawn_ground_y)

        final_x = jnp.where(should_spawn, spawn_x, new_x)
        final_y = jnp.where(should_spawn, spawn_y, new_y)
        final_active = jnp.logical_and(jnp.logical_or(surviving_enemies, should_spawn), final_x > 0)

        intermediate_state = state._replace(
            score=(state.score + jnp.sum(valid_kill) * 2).astype(jnp.int32),
            enemies_active=final_active, enemies_x=final_x, enemies_y=final_y,
            enemies_type=jnp.where(should_spawn, new_types, state.enemies_type).astype(jnp.int32),
            enemies_dx=jnp.zeros_like(state.enemies_dx), enemies_dy=jnp.zeros_like(state.enemies_dy),
            enemies_floor_y=jnp.zeros_like(state.enemies_floor_y),
            key=rng
        )
        return self._handle_common_death_logic(intermediate_state, any_friend_hit, scatter_key)

    def _generic_map_logic(self, state: CrossbowState, action: chex.Array) -> Tuple[CrossbowState, bool]:
        rng, spawn_key, scatter_key = jax.random.split(state.key, 3)
        HIT_TOLERANCE = 8
        cx, cy = state.cursor_x, state.cursor_y
        ex, ey = state.enemies_x, state.enemies_y
        hit_x = jnp.logical_and(cx < ex + self.consts.ENEMY_SIZE[0] + HIT_TOLERANCE, cx + self.consts.CURSOR_SIZE[0] > ex - HIT_TOLERANCE)
        hit_y = jnp.logical_and(cy < ey + self.consts.ENEMY_SIZE[1] + HIT_TOLERANCE, cy + self.consts.CURSOR_SIZE[1] > ey - HIT_TOLERANCE)
        is_hit = jnp.logical_and(hit_x, hit_y)
        valid_kill = jnp.logical_and(state.is_firing, jnp.logical_and(state.enemies_active, is_hit))
        surviving_enemies = jnp.logical_and(state.enemies_active, jnp.logical_not(valid_kill))

        new_y = state.enemies_y + jnp.where(state.step_counter % 3 == 0, 1.0, 0.0)
        fx, fy = state.friend_x, state.friend_y
        danger_x = jnp.logical_and(state.enemies_x < fx + self.consts.FRIEND_SIZE[0], state.enemies_x + self.consts.ENEMY_SIZE[0] > fx)
        danger_y = jnp.logical_and(new_y < fy + self.consts.FRIEND_SIZE[1], new_y + self.consts.ENEMY_SIZE[1] > fy)
        friend_hit = jnp.any(jnp.logical_and(jnp.logical_and(danger_x, danger_y), surviving_enemies))
        any_friend_hit = jnp.logical_and(friend_hit, state.dying_timer == 0)

        spawn_chance = jax.random.uniform(spawn_key, shape=(self.consts.MAX_ENEMIES,)) < 0.05
        should_spawn = jnp.logical_and(jnp.logical_not(surviving_enemies), spawn_chance)
        spawn_x = jax.random.randint(spawn_key, (self.consts.MAX_ENEMIES,), 20, self.consts.WIDTH - 20).astype(jnp.float32)
        spawn_y = jax.random.randint(spawn_key, (self.consts.MAX_ENEMIES,), 20, 50).astype(jnp.float32)

        final_x = jnp.where(should_spawn, spawn_x, state.enemies_x)
        final_y = jnp.where(should_spawn, spawn_y, new_y)
        final_active = jnp.logical_and(jnp.logical_or(surviving_enemies, should_spawn), final_y < self.consts.PLAY_AREA_HEIGHT)

        intermediate_state = state._replace(
            score=(state.score + jnp.sum(valid_kill)).astype(jnp.int32),
            enemies_active=final_active, enemies_x=final_x, enemies_y=final_y,
            enemies_type=jnp.where(should_spawn, EnemyType.GENERIC, state.enemies_type).astype(jnp.int32),
            enemies_dx=jnp.zeros_like(state.enemies_dx), enemies_dy=jnp.zeros_like(state.enemies_dy),
            enemies_floor_y=jnp.zeros_like(state.enemies_floor_y), # Carry over
            key=rng
        )
        return self._handle_common_death_logic(intermediate_state, any_friend_hit, scatter_key)

    def _update_game_phase(self, state: CrossbowState, action: chex.Array) -> CrossbowState:
        on_start = state.game_phase == GamePhase.START_SCREEN
        on_ready = state.game_phase == GamePhase.GET_READY
        cx, cy = state.cursor_x, state.cursor_y

        sel = [
            jnp.logical_and(jnp.logical_and(cx >= 47, cx <= 64), jnp.logical_and(cy >= 27, cy <= 51)),
            jnp.logical_and(jnp.logical_and(cx >= 113, cx <= 128), jnp.logical_and(cy >= 38, cy <= 52)),
            jnp.logical_and(jnp.logical_and(cx >= 117, cx <= 132), jnp.logical_and(cy >= 101, cy <= 118)),
            jnp.logical_and(jnp.logical_and(cx >= 53, cx <= 68), jnp.logical_and(cy >= 96, cy <= 118)),
            jnp.logical_and(jnp.logical_and(cx >= 97, cx <= 112), jnp.logical_and(cy >= 130, cy <= 151)),
            jnp.logical_and(jnp.logical_and(cx >= 33, cx <= 48), jnp.logical_and(cy >= 125, cy <= 151)),
        ]

        trigger = jnp.logical_and(on_start, state.is_firing)
        target = jnp.select(sel, [GamePhase.DESERT_MAP, GamePhase.CAVE_MAP, GamePhase.FOREST_MAP, GamePhase.MAP_4, GamePhase.DEMON_MAP, GamePhase.MAP_6], default=GamePhase.START_SCREEN)
        ready_done = jnp.logical_and(on_ready, state.get_ready_timer == 0)

        return state._replace(
            game_phase=jnp.where(jnp.logical_and(trigger, target != GamePhase.START_SCREEN), GamePhase.GET_READY, jnp.where(ready_done, state.selected_target_map, state.game_phase)).astype(jnp.int32),
            selected_target_map=jnp.where(trigger, target, state.selected_target_map).astype(jnp.int32),
            get_ready_timer=jnp.where(trigger, self.consts.GET_READY_DURATION, jnp.where(on_ready, jnp.maximum(0, state.get_ready_timer - 1), 0)).astype(jnp.int32),
            fade_in_timer=jnp.where(ready_done, self.consts.FADE_IN_DURATION, jnp.maximum(0, state.fade_in_timer - 1)).astype(jnp.int32)
        )

    def reset(self, key: chex.PRNGKey = jax.random.PRNGKey(42)) -> Tuple[CrossbowObservation, CrossbowState]:
        state_key, _ = jax.random.split(key)
        state = CrossbowState(
            cursor_x=jnp.array(self.consts.WIDTH // 2, dtype=jnp.int32), cursor_y=jnp.array(self.consts.HEIGHT // 2, dtype=jnp.int32),
            is_firing=jnp.array(False), friend_x=jnp.array(20, dtype=jnp.int32), friend_y=jnp.array(128, dtype=jnp.int32),
            friend_active=jnp.array(True), dying_timer=jnp.array(0, dtype=jnp.int32), get_ready_timer=jnp.array(0, dtype=jnp.int32),
            fade_in_timer=jnp.array(0, dtype=jnp.int32), selected_target_map=jnp.array(0, dtype=jnp.int32),
            scatter_px_x=jnp.zeros(100, dtype=jnp.float32), scatter_px_y=jnp.zeros(100, dtype=jnp.float32),
            scatter_px_dx=jnp.zeros(100, dtype=jnp.float32), scatter_px_dy=jnp.zeros(100, dtype=jnp.float32),
            scatter_px_color_idx=jnp.zeros(100, dtype=jnp.uint8), scatter_px_active=jnp.zeros(100, dtype=bool),
            enemies_x=jnp.zeros(self.consts.MAX_ENEMIES, dtype=jnp.float32), enemies_y=jnp.zeros(self.consts.MAX_ENEMIES, dtype=jnp.float32),
            enemies_dx=jnp.zeros(self.consts.MAX_ENEMIES, dtype=jnp.float32), enemies_dy=jnp.zeros(self.consts.MAX_ENEMIES, dtype=jnp.float32),
            enemies_floor_y=jnp.zeros(self.consts.MAX_ENEMIES, dtype=jnp.float32),
            enemies_active=jnp.zeros(self.consts.MAX_ENEMIES, dtype=bool), enemies_type=jnp.zeros(self.consts.MAX_ENEMIES, dtype=jnp.int32),
            game_phase=jnp.array(GamePhase.START_SCREEN, dtype=jnp.int32), score=jnp.array(0, dtype=jnp.int32),
            lives=jnp.array(self.consts.MAX_LIVES, dtype=jnp.int32), step_counter=jnp.array(0, dtype=jnp.int32), key=state_key
        )
        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: CrossbowState, action: chex.Array):
        prev_score = state.score
        new_key, step_key = jax.random.split(state.key)
        state = self._cursor_step(state._replace(key=step_key), action)
        state = self._update_game_phase(state, action)

        is_gameplay = state.game_phase >= GamePhase.DESERT_MAP
        state = jax.lax.cond(jnp.logical_and(is_gameplay, state.friend_active), lambda s: self._friend_step(s), lambda s: s, state)

        def _combat_router(s):
            return jax.lax.cond(s.game_phase == GamePhase.DESERT_MAP, lambda _s: self._desert_map_logic(_s, action),
                                lambda _s: jax.lax.cond(_s.game_phase == GamePhase.DEMON_MAP, lambda __s: self._demon_map_logic(__s, action),
                                                        lambda __s: self._generic_map_logic(__s, action), _s), s)

        state, game_over = jax.lax.cond(jnp.logical_and(is_gameplay, state.friend_active), _combat_router, lambda s: (s, False), state)
        state = state._replace(step_counter=state.step_counter + 1, key=new_key)
        return self._get_observation(state), state, (state.score - prev_score).astype(float), jnp.logical_or(game_over, state.step_counter > 4000), self._get_info(state)

    def _get_observation(self, state): return CrossbowObservation(state.cursor_x, state.cursor_y, state.friend_x, state.game_phase, state.lives, state.score)
    def _get_info(self, state): return CrossbowInfo(time=state.step_counter)
    def obs_to_flat_array(self, obs): return jnp.array([0])
    def action_space(self): return spaces.Discrete(18)
    def observation_space(self): return spaces.Dict({})
    def image_space(self): return spaces.Box(0, 255, (210, 160, 3), jnp.uint8)
    def render(self, state: CrossbowState) -> jnp.ndarray: return self.renderer.render(state)


class CrossbowRenderer(JAXGameRenderer):
    def __init__(self, consts: CrossbowConstants = None):
        super().__init__(consts)
        self.consts = consts or CrossbowConstants()
        self.jr = render_utils.JaxRenderingUtils(render_utils.RendererConfig(game_dimensions=(210, 160), channels=3))
        base_dir = os.path.dirname(os.path.abspath(__file__))
        (self.PALETTE, self.SHAPE_MASKS, self.BACKGROUND, _, _) = self.jr.load_and_setup_assets(self.consts.ASSET_CONFIG, os.path.join(base_dir, "sprites", "crossbow"))
        self.pixel_masks = {c: jnp.array([[c]], dtype=jnp.uint8) for c in range(1, 9)}

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: CrossbowState):
        raster = self.jr.render_at(self.BACKGROUND, -16, 0, self.SHAPE_MASKS["backgrounds"][state.game_phase])
        is_gameplay, is_dying = state.game_phase >= GamePhase.DESERT_MAP, state.dying_timer > 0

        # Friend
        f_mask = self.SHAPE_MASKS["friend"][(state.step_counter // 8) % len(self.SHAPE_MASKS["friend"])]
        raster = jax.lax.cond(jnp.logical_and(state.friend_active, jnp.logical_and(is_gameplay, jnp.logical_not(is_dying))), lambda r: self.jr.render_at(r, state.friend_x, state.friend_y, f_mask), lambda r: r, raster)

        # Scatter
        def _draw_px(i, r):
            pixel_mask = jax.lax.switch(state.scatter_px_color_idx[i] - 1, [lambda: self.pixel_masks[c] for c in range(1, 9)])
            return jax.lax.cond(state.scatter_px_active[i], lambda _r: self.jr.render_at(_r, state.scatter_px_x[i].astype(jnp.int32), state.scatter_px_y[i].astype(jnp.int32), pixel_mask), lambda _r: _r, r)
        raster = jax.lax.cond(is_dying, lambda r: jax.lax.fori_loop(0, 100, _draw_px, r), lambda r: r, raster)

        # Enemies
        masks = [self.SHAPE_MASKS["enemy"], self.SHAPE_MASKS["scorpion"], self.SHAPE_MASKS["ant"], self.SHAPE_MASKS["vulture"], self.SHAPE_MASKS["spawn"]]
        def _draw_e(i, r):
            m = jax.lax.switch(state.enemies_type[i], [lambda: masks[0], lambda: masks[1], lambda: masks[2], lambda: masks[3], lambda: masks[4]])
            return jax.lax.cond(jnp.logical_and(state.enemies_active[i], is_gameplay), lambda _r: self.jr.render_at(_r, state.enemies_x[i].astype(jnp.int32), state.enemies_y[i].astype(jnp.int32), m), lambda _r: _r, r)
        raster = jax.lax.fori_loop(0, self.consts.MAX_ENEMIES, _draw_e, raster)

        raster = self.jr.render_at(raster, state.cursor_x, state.cursor_y, self.SHAPE_MASKS["cursor"])
        raster = jax.lax.cond(state.is_firing, lambda r: self.jr.render_at(r, state.cursor_x, state.cursor_y, self.SHAPE_MASKS["shot"]), lambda r: r, raster)

        score_digits = self.jr.int_to_digits(state.score, max_digits=6)
        num_digits = jnp.select([state.score < 10, state.score < 100, state.score < 1000, state.score < 10000, state.score < 100000], [1, 2, 3, 4, 5], 6)
        raster = self.jr.render_label_selective(raster, 98 - 8 * (num_digits - 1), 186, score_digits, self.SHAPE_MASKS["digits"], 6 - num_digits, num_digits, spacing=8, max_digits_to_render=6)

        img = self.jr.render_from_palette(raster, self.PALETTE)
        fade = jnp.select([jnp.logical_and(state.game_phase == GamePhase.GET_READY, state.get_ready_timer < 45), state.fade_in_timer > 0],
                          [state.get_ready_timer / 45, (30 - state.fade_in_timer) / 30], 1.0)
        return (img * fade).astype(jnp.uint8)