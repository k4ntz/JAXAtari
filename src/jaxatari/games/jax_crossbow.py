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
                'backgrounds/drawbridge_map.npy',
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
        {
            'name': 'snake', 'type': 'group',
            'files': ['snake_1.npy', 'snake_2.npy']
        },
        {'name': 'archer', 'type': 'single', 'file': 'archer.npy'},
        {'name': 'arrow', 'type': 'single', 'file': 'arrow.npy'},
        {'name': 'cursor', 'type': 'single', 'file': 'cursor.npy'},
        {'name': 'enemy', 'type': 'single', 'file': 'enemy.npy'},
        {'name': 'shot', 'type': 'single', 'file': 'shot.npy'},
        {'name': 'digits', 'type': 'digits', 'pattern': 'digits/digit_{}.npy'},
    )

def _get_initial_scatter_state(consts):
    num_pixels = consts.MAX_SCATTER_PIXELS
    rel_x = jnp.tile(jnp.arange(0, 8), 13)[:num_pixels]
    rel_y = jnp.repeat(jnp.arange(0, 27, 2), 8)[:num_pixels]
    colors = jnp.concatenate([jnp.full(32, 7), jnp.full(32, 5), jnp.full(36, 6)])[:num_pixels]
    center_x = 4.0; center_y = 13.5
    base_dx = (rel_x - center_x) * 0.3
    base_dy = (rel_y - center_y) * 0.3
    return rel_x, rel_y, base_dx, base_dy, colors

class GamePhase:
    START_SCREEN = 0
    GET_READY = 1
    DESERT_MAP = 2
    CAVE_MAP = 3
    FOREST_MAP = 4
    MAP_4 = 5
    DRAWBRIDGE_MAP = 6
    MAP_6 = 7

class EnemyType:
    GENERIC = 0
    SCORPION = 1
    ANT = 2
    VULTURE = 3
    SNAKE = 4
    ARCHER = 5
    ARROW = 6

class CrossbowConstants(NamedTuple):
    WIDTH: int = 160
    HEIGHT: int = 210
    CURSOR_SPEED: int = 2
    CURSOR_SIZE: Tuple[int, int] = (4, 4)
    FRIEND_SPEED: float = 0.5
    ENEMY_SPEED: float = 0.5
    MAX_ENEMIES: int = 10
    MAX_LIVES: int = 3
    DYING_DURATION: int = 45
    GET_READY_DURATION: int = 180
    FADE_OUT_DURATION: int = 45
    FADE_IN_DURATION: int = 30
    MAX_SCATTER_PIXELS: int = 100
    FRIEND_SIZE: Tuple[int, int] = (8, 27)
    ENEMY_SIZE: Tuple[int, int] = (8, 8)
    SHOT_SIZE: Tuple[int, int] = (6, 6)
    PLAY_AREA_HEIGHT: int = 180
    GROUND_Y_MIN: int = 130
    GROUND_Y_MAX: int = 165
    ASSET_CONFIG: tuple = _get_default_asset_config()

    ROPE_1_POS: Tuple[int, int] = (110, 60)
    ROPE_2_POS: Tuple[int, int] = (110, 100)

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
    enemies_timer: chex.Array
    game_phase: chex.Array
    score: chex.Array
    lives: chex.Array
    step_counter: chex.Array
    key: chex.PRNGKey
    rope_1_broken: chex.Array
    rope_2_broken: chex.Array

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
        is_active = jnp.logical_and(state.scatter_px_active,
                                    jnp.logical_and(jnp.logical_and(new_x >= -5, new_x < self.consts.WIDTH + 5),
                                                    jnp.logical_and(new_y >= -5, new_y < self.consts.HEIGHT + 5)))
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
            lives=new_lives.astype(jnp.int32), dying_timer=new_timer.astype(jnp.int32),
            friend_x=friend_x_next, enemies_active=enemies_active_next, scatter_px_active=scatter_active_next
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
        is_fire = jnp.isin(action, jnp.array([Action.FIRE, Action.UPFIRE, Action.RIGHTFIRE, Action.LEFTFIRE, Action.DOWNFIRE, Action.UPRIGHTFIRE, Action.UPLEFTFIRE, Action.DOWNRIGHTFIRE, Action.DOWNLEFTFIRE]))
        return state._replace(cursor_x=new_x.astype(jnp.int32), cursor_y=new_y.astype(jnp.int32), is_firing=is_fire)

    def _friend_step(self, state: CrossbowState) -> CrossbowState:
        is_dying = state.dying_timer > 0
        should_move = state.step_counter % 8 == 0
        new_x = state.friend_x + jnp.where(should_move & state.friend_active, 1, 0)
        reached_goal = new_x > self.consts.WIDTH
        final_x = jnp.where(is_dying, state.friend_x, jnp.where(reached_goal, 0, new_x))
        return state._replace(friend_x=final_x.astype(jnp.int32))

    def _drawbridge_map_logic(self, state: CrossbowState, action: chex.Array) -> Tuple[CrossbowState, bool]:
        rng, key_spawn_general, key_type_general, key_x_archer, key_y_archer, key_y_vulture, key_scatter = jax.random.split(state.key, 7)
        is_dying = state.dying_timer > 0
        HIT_TOLERANCE = 8
        ARCHER_LIFESPAN = 120
        ARCHER_SHOOT_TIME = 60
        ARROW_SPEED = 3.0
        MAX_ARCHERS = 4
        ARCHER_MIN_SEP = 25.0

        cx, cy = state.cursor_x, state.cursor_y
        rope_1_hit = jnp.logical_and(state.is_firing, jnp.logical_and(jnp.abs(cx - self.consts.ROPE_1_POS[0]) < 15, jnp.abs(cy - self.consts.ROPE_1_POS[1]) < 15))
        rope_2_hit = jnp.logical_and(state.is_firing, jnp.logical_and(jnp.abs(cx - self.consts.ROPE_2_POS[0]) < 15, jnp.abs(cy - self.consts.ROPE_2_POS[1]) < 15))
        new_rope_1 = jnp.logical_or(state.rope_1_broken, rope_1_hit)
        new_rope_2 = jnp.logical_or(state.rope_2_broken, rope_2_hit)
        bridge_open = jnp.logical_and(new_rope_1, new_rope_2)
        finished_map = jnp.logical_and(bridge_open, state.friend_x >= self.consts.WIDTH - 10)
        MOAT_EDGE_X = 40
        friend_x_constrained = jnp.where(jnp.logical_not(bridge_open), jnp.minimum(state.friend_x, MOAT_EDGE_X), state.friend_x)

        ex, ey = state.enemies_x, state.enemies_y
        hit_x = jnp.logical_and(cx < ex + self.consts.ENEMY_SIZE[0] + HIT_TOLERANCE, cx + self.consts.CURSOR_SIZE[0] > ex - HIT_TOLERANCE)
        hit_y = jnp.logical_and(cy < ey + self.consts.ENEMY_SIZE[1] + HIT_TOLERANCE, cy + self.consts.CURSOR_SIZE[1] > ey - HIT_TOLERANCE)
        is_hit = jnp.logical_and(hit_x, hit_y)
        valid_kill = jnp.logical_and(state.is_firing, jnp.logical_and(state.enemies_active, is_hit))
        point_values = jnp.array([0, 0, 0, 200, 0, 100, 50])
        reward = jnp.sum(jnp.where(valid_kill, point_values[state.enemies_type], 0))
        surviving_enemies = jnp.logical_and(state.enemies_active, jnp.logical_not(valid_kill))

        is_archer = state.enemies_type == EnemyType.ARCHER
        new_timer = jnp.where(jnp.logical_and(surviving_enemies, is_archer), state.enemies_timer - 1, state.enemies_timer)
        archer_expired = jnp.logical_and(is_archer, new_timer <= 0)
        enemies_active_after_expiry = jnp.logical_and(surviving_enemies, jnp.logical_not(archer_expired))

        archers_shooting = jnp.logical_and(enemies_active_after_expiry, jnp.logical_and(is_archer, new_timer == ARCHER_SHOOT_TIME))
        shooter_idx = jnp.argmax(archers_shooting)
        can_shoot = jnp.any(archers_shooting)
        shooter_x, shooter_y = state.enemies_x[shooter_idx], state.enemies_y[shooter_idx]

        available_slots = jnp.logical_not(enemies_active_after_expiry)
        arrow_slot_idx = jnp.argmax(available_slots)
        has_slot = jnp.any(available_slots)
        spawn_arrow_now = jnp.logical_and(can_shoot, has_slot)

        target_center_x = friend_x_constrained + 4.0
        target_center_y = state.friend_y + 13.0
        shooter_center_x = shooter_x + 4.0
        shooter_center_y = shooter_y + 4.0

        delta_x = target_center_x - shooter_center_x
        delta_y = target_center_y - shooter_center_y
        dist = jnp.sqrt(delta_x**2 + delta_y**2) + 1e-5

        arrow_dx = (delta_x / dist) * ARROW_SPEED
        arrow_dy = (delta_y / dist) * ARROW_SPEED

        enemies_active_w_arrow = jnp.where(spawn_arrow_now, enemies_active_after_expiry.at[arrow_slot_idx].set(True), enemies_active_after_expiry)
        enemies_type_w_arrow = jnp.where(spawn_arrow_now, state.enemies_type.at[arrow_slot_idx].set(EnemyType.ARROW), state.enemies_type)
        enemies_x_w_arrow = jnp.where(spawn_arrow_now, state.enemies_x.at[arrow_slot_idx].set(shooter_x), state.enemies_x)
        enemies_y_w_arrow = jnp.where(spawn_arrow_now, state.enemies_y.at[arrow_slot_idx].set(shooter_y), state.enemies_y)

        current_dx_state = state.enemies_dx
        current_dy_state = state.enemies_dy
        enemies_dx_w_arrow = jnp.where(spawn_arrow_now, current_dx_state.at[arrow_slot_idx].set(arrow_dx), current_dx_state)
        enemies_dy_w_arrow = jnp.where(spawn_arrow_now, current_dy_state.at[arrow_slot_idx].set(arrow_dy), current_dy_state)

        spawn_roll = jax.random.uniform(key_spawn_general, shape=(self.consts.MAX_ENEMIES,))
        raw_should_spawn = jnp.logical_and(jnp.logical_not(enemies_active_w_arrow), spawn_roll < 0.012)

        type_roll = jax.random.uniform(key_type_general, (self.consts.MAX_ENEMIES,))
        spawn_type_candidate = jnp.where(type_roll < 0.6, EnemyType.VULTURE, EnemyType.ARCHER)
        spawn_x_archer = jax.random.uniform(key_x_archer, (self.consts.MAX_ENEMIES,), minval=135.0, maxval=150.0)
        spawn_y_archer = jax.random.uniform(key_y_archer, (self.consts.MAX_ENEMIES,), minval=50.0, maxval=140.0)
        spawn_x_vulture = 155.0
        spawn_y_vulture = jax.random.uniform(key_y_vulture, (self.consts.MAX_ENEMIES,), minval=20.0, maxval=80.0)

        active_archers_mask = jnp.logical_and(enemies_active_w_arrow, enemies_type_w_arrow == EnemyType.ARCHER)
        num_active_archers = jnp.sum(active_archers_mask)
        limit_reached = num_active_archers >= MAX_ARCHERS

        dist_to_others = jnp.abs(enemies_y_w_arrow - spawn_y_archer)
        conflict_mask = jnp.logical_and(active_archers_mask, dist_to_others < ARCHER_MIN_SEP)
        has_overlap = jnp.any(conflict_mask)

        is_archer_attempt = spawn_type_candidate == EnemyType.ARCHER
        spawn_allowed = jnp.logical_or(jnp.logical_not(is_archer_attempt),
                                       jnp.logical_and(jnp.logical_not(limit_reached), jnp.logical_not(has_overlap)))
        should_spawn_final = jnp.logical_and(raw_should_spawn, spawn_allowed)

        final_active = jnp.logical_or(enemies_active_w_arrow, should_spawn_final)
        final_type = jnp.where(should_spawn_final, spawn_type_candidate, enemies_type_w_arrow)

        spawn_x_final = jnp.where(spawn_type_candidate == EnemyType.ARCHER, spawn_x_archer, spawn_x_vulture)
        spawn_y_final = jnp.where(spawn_type_candidate == EnemyType.ARCHER, spawn_y_archer, spawn_y_vulture)

        current_x = jnp.where(should_spawn_final, spawn_x_final, enemies_x_w_arrow)
        current_y = jnp.where(should_spawn_final, spawn_y_final, enemies_y_w_arrow)

        spawn_timer = jnp.where(spawn_type_candidate == EnemyType.ARCHER, ARCHER_LIFESPAN, 0)
        final_timer = jnp.where(should_spawn_final, spawn_timer, new_timer)

        final_dx_state = jnp.where(should_spawn_final, 0.0, enemies_dx_w_arrow)
        final_dy_state = jnp.where(should_spawn_final, 0.0, enemies_dy_w_arrow)

        VULTURE_SPEED = -1.0
        target_y = state.friend_y
        dy_homing = jnp.sign(target_y - current_y) * 0.6
        vulture_wobble = 0.6 * jnp.sin(state.step_counter * 0.15)
        vulture_dy = dy_homing + vulture_wobble
        vulture_dx = VULTURE_SPEED

        step_dx = jnp.select([final_type == EnemyType.ARROW, final_type == EnemyType.VULTURE], [final_dx_state, vulture_dx], default=0.0)
        step_dy = jnp.select([final_type == EnemyType.ARROW, final_type == EnemyType.VULTURE], [final_dy_state, vulture_dy], default=0.0)

        final_x_moved = current_x + step_dx
        final_y_moved = current_y + step_dy

        is_on_screen = jnp.logical_and(final_x_moved > -20, final_x_moved < self.consts.WIDTH + 10)
        is_on_screen_y = jnp.logical_and(final_y_moved > -20, final_y_moved < self.consts.HEIGHT + 20)
        final_active_culled = jnp.logical_and(final_active, jnp.logical_and(is_on_screen, is_on_screen_y))

        fx, fy = friend_x_constrained, state.friend_y
        danger_x = jnp.logical_and(final_x_moved < fx + self.consts.FRIEND_SIZE[0], final_x_moved + self.consts.ENEMY_SIZE[0] > fx)
        danger_y = jnp.logical_and(final_y_moved < fy + self.consts.FRIEND_SIZE[1], final_y_moved + self.consts.ENEMY_SIZE[1] > fy)
        friend_hit = jnp.any(jnp.logical_and(jnp.logical_and(danger_x, danger_y), final_active_culled))
        any_friend_hit = jnp.logical_and(jnp.logical_and(friend_hit, state.friend_active), jnp.logical_not(is_dying))

        intermediate_state = state._replace(
            score=(state.score + reward).astype(jnp.int32),
            enemies_active=final_active_culled,
            enemies_x=final_x_moved.astype(jnp.float32),
            enemies_y=final_y_moved.astype(jnp.float32),
            enemies_dx=final_dx_state.astype(jnp.float32),
            enemies_dy=final_dy_state.astype(jnp.float32),
            enemies_type=final_type.astype(jnp.int32),
            enemies_timer=final_timer.astype(jnp.int32),
            key=rng,
            rope_1_broken=new_rope_1,
            rope_2_broken=new_rope_2,
            friend_x=friend_x_constrained.astype(jnp.int32),
            game_phase=jnp.where(finished_map, GamePhase.MAP_6, state.game_phase)
        )
        return self._handle_common_death_logic(intermediate_state, any_friend_hit, key_scatter)

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

        point_values = jnp.array([0, 50, 50, 100, 200, 100, 50])
        reward = jnp.sum(jnp.where(valid_kill, point_values[state.enemies_type], 0))

        move_freq = jnp.select([state.enemies_type == EnemyType.ANT, state.enemies_type == EnemyType.SNAKE], [2, 2], default=3)
        should_move = (state.step_counter % move_freq) == 0

        is_vulture = state.enemies_type == EnemyType.VULTURE
        move_x = jnp.where(should_move, -1.0, 0.0)
        move_y = jnp.where(jnp.logical_and(state.enemies_type == EnemyType.VULTURE, should_move), 1.0, 0.0)

        new_x = state.enemies_x + move_x
        new_y = state.enemies_y + move_y

        spawn_chance = jax.random.uniform(spawn_key, shape=(self.consts.MAX_ENEMIES,)) < 0.03
        should_spawn = jnp.logical_and(jnp.logical_not(surviving_enemies), spawn_chance)
        new_types = jax.random.randint(type_key, (self.consts.MAX_ENEMIES,), 1, 5)

        spawn_x = jax.random.randint(spawn_key, (self.consts.MAX_ENEMIES,), 140, 160).astype(jnp.float32)
        spawn_ground_y = jax.random.randint(spawn_key, (self.consts.MAX_ENEMIES,), self.consts.GROUND_Y_MIN, self.consts.GROUND_Y_MAX).astype(jnp.float32)
        spawn_air_y = jax.random.randint(spawn_key, (self.consts.MAX_ENEMIES,), 20, 80).astype(jnp.float32)
        spawn_y = jnp.where(new_types == EnemyType.VULTURE, spawn_air_y, spawn_ground_y)

        final_x = jnp.where(should_spawn, spawn_x, new_x)
        final_y = jnp.where(should_spawn, spawn_y, new_y)
        final_active = jnp.logical_and(jnp.logical_or(surviving_enemies, should_spawn), final_x > -10)

        fx, fy = state.friend_x, state.friend_y
        danger_x = jnp.logical_and(final_x < fx + self.consts.FRIEND_SIZE[0], final_x + self.consts.ENEMY_SIZE[0] > fx)
        danger_y = jnp.logical_and(final_y < fy + self.consts.FRIEND_SIZE[1] + 10, final_y + self.consts.ENEMY_SIZE[1] > fy - 5)
        friend_hit = jnp.any(jnp.logical_and(jnp.logical_and(danger_x, danger_y), final_active))
        any_friend_hit = jnp.logical_and(friend_hit, state.dying_timer == 0)

        intermediate_state = state._replace(
            score=(state.score + reward).astype(jnp.int32),
            enemies_active=final_active,
            enemies_x=final_x.astype(jnp.float32),
            enemies_y=final_y.astype(jnp.float32),
            enemies_type=jnp.where(should_spawn, new_types, state.enemies_type).astype(jnp.int32),
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
        final_x, final_y = jnp.where(should_spawn, spawn_x, state.enemies_x), jnp.where(should_spawn, spawn_y, new_y)
        final_active = jnp.logical_and(jnp.logical_or(surviving_enemies, should_spawn), final_y < self.consts.PLAY_AREA_HEIGHT)
        intermediate_state = state._replace(
            score=(state.score + jnp.sum(valid_kill)).astype(jnp.int32), enemies_active=final_active,
            enemies_x=final_x, enemies_y=final_y, enemies_type=jnp.where(should_spawn, EnemyType.GENERIC, state.enemies_type).astype(jnp.int32), key=rng
        )
        return self._handle_common_death_logic(intermediate_state, any_friend_hit, scatter_key)

    def _update_game_phase(self, state: CrossbowState, action: chex.Array) -> CrossbowState:
        on_start, on_ready = state.game_phase == GamePhase.START_SCREEN, state.game_phase == GamePhase.GET_READY
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
        target = jnp.select(sel, [GamePhase.DESERT_MAP, GamePhase.CAVE_MAP, GamePhase.FOREST_MAP, GamePhase.MAP_4, GamePhase.DRAWBRIDGE_MAP, GamePhase.MAP_6], default=GamePhase.START_SCREEN)
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
            enemies_timer=jnp.zeros(self.consts.MAX_ENEMIES, dtype=jnp.int32),
            enemies_active=jnp.zeros(self.consts.MAX_ENEMIES, dtype=bool), enemies_type=jnp.zeros(self.consts.MAX_ENEMIES, dtype=jnp.int32),
            game_phase=jnp.array(GamePhase.START_SCREEN, dtype=jnp.int32), score=jnp.array(0, dtype=jnp.int32),
            lives=jnp.array(self.consts.MAX_LIVES, dtype=jnp.int32), step_counter=jnp.array(0, dtype=jnp.int32), key=state_key,
            rope_1_broken=jnp.array(False), rope_2_broken=jnp.array(False)
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
                                lambda _s: jax.lax.cond(_s.game_phase == GamePhase.DRAWBRIDGE_MAP, lambda __s: self._drawbridge_map_logic(__s, action),
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
        (self.PALETTE, self.SHAPE_MASKS, self.BACKGROUND, _, _) = self.jr.load_and_setup_assets(
            self.consts.ASSET_CONFIG, os.path.join(base_dir, "sprites", "crossbow")
        )
        self.pixel_masks = {c: jnp.array([[c]], dtype=jnp.uint8) for c in range(1, 9)}

        target_h, target_w = 16, 16
        def pad_to_target(mask):
            h, w = mask.shape
            return jnp.pad(mask, ((0, target_h - h), (0, target_w - w)), mode='constant', constant_values=255)

        self.harmonized_enemy_masks = [
            pad_to_target(self.SHAPE_MASKS["enemy"]),    # 0
            pad_to_target(self.SHAPE_MASKS["scorpion"]), # 1
            pad_to_target(self.SHAPE_MASKS["ant"]),      # 2
            pad_to_target(self.SHAPE_MASKS["vulture"]),  # 3
            pad_to_target(self.SHAPE_MASKS["archer"]),   # 5
            pad_to_target(self.SHAPE_MASKS["arrow"]),    # 6
        ]

        self.snake_anim_masks = jnp.stack([
            pad_to_target(self.SHAPE_MASKS["snake"][0]),
            pad_to_target(self.SHAPE_MASKS["snake"][1])
        ])

        ink_color = jnp.min(self.SHAPE_MASKS["arrow"])
        base_right = jnp.full((16, 16), 255, dtype=jnp.uint8)
        base_right = base_right.at[8, 6:11].set(ink_color)
        base_right = base_right.at[8, 11].set(ink_color)
        base_right = base_right.at[7, 10].set(ink_color)
        base_right = base_right.at[9, 10].set(ink_color)
        base_dr = jnp.full((16, 16), 255, dtype=jnp.uint8)
        diag_idx = jnp.arange(6, 11)
        base_dr = base_dr.at[diag_idx, diag_idx].set(ink_color)
        base_dr = base_dr.at[11, 11].set(ink_color)
        base_dr = base_dr.at[10, 11].set(ink_color)
        base_dr = base_dr.at[11, 10].set(ink_color)
        arrow_right = base_right; arrow_down = jnp.rot90(base_right, k=3); arrow_left = jnp.rot90(base_right, k=2); arrow_up = jnp.rot90(base_right, k=1)
        arrow_dr = base_dr; arrow_dl = jnp.rot90(base_dr, k=3); arrow_ul = jnp.rot90(base_dr, k=2); arrow_ur = jnp.rot90(base_dr, k=1)
        self.arrow_sprites = jnp.stack([arrow_right, arrow_dr, arrow_down, arrow_dl, arrow_left, arrow_ul, arrow_up, arrow_ur])

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: CrossbowState):
        raster = self.jr.render_at(self.BACKGROUND, -16, 0, self.SHAPE_MASKS["backgrounds"][state.game_phase])
        is_gameplay, is_dying = state.game_phase >= GamePhase.DESERT_MAP, state.dying_timer > 0

        # Friend
        f_mask = self.SHAPE_MASKS["friend"][(state.step_counter // 8) % len(self.SHAPE_MASKS["friend"])]
        raster = jax.lax.cond(jnp.logical_and(state.friend_active, jnp.logical_and(is_gameplay, jnp.logical_not(is_dying))),
                              lambda r: self.jr.render_at(r, state.friend_x, state.friend_y, f_mask), lambda r: r, raster)

        # Ropes
        def _draw_ropes(r):
            rope_sprite = jnp.full((20, 4), 6, dtype=jnp.uint8)
            r = jax.lax.cond(jnp.logical_not(state.rope_1_broken),
                             lambda _r: self.jr.render_at(_r, self.consts.ROPE_1_POS[0], self.consts.ROPE_1_POS[1], rope_sprite),
                             lambda _r: _r, r)
            r = jax.lax.cond(jnp.logical_not(state.rope_2_broken),
                             lambda _r: self.jr.render_at(_r, self.consts.ROPE_2_POS[0], self.consts.ROPE_2_POS[1], rope_sprite),
                             lambda _r: _r, r)
            return r
        raster = jax.lax.cond(state.game_phase == GamePhase.DRAWBRIDGE_MAP, _draw_ropes, lambda r: r, raster)

        # Scatter
        def _draw_px(i, r):
            pixel_mask = jax.lax.switch(state.scatter_px_color_idx[i] - 1, [lambda: self.pixel_masks[c] for c in range(1, 9)])
            return jax.lax.cond(state.scatter_px_active[i], lambda _r: self.jr.render_at(_r, state.scatter_px_x[i].astype(jnp.int32), state.scatter_px_y[i].astype(jnp.int32), pixel_mask), lambda _r: _r, r)
        raster = jax.lax.cond(is_dying, lambda r: jax.lax.fori_loop(0, 100, _draw_px, r), lambda r: r, raster)

        # Enemies
        def _draw_e(i, r):
            is_arrow = state.enemies_type[i] == 6

            # --- Arrow Logic ---
            dx, dy = state.enemies_dx[i], state.enemies_dy[i]
            theta = jnp.arctan2(dy, dx)
            deg = jnp.degrees(theta)
            deg = jnp.where(deg < 0, deg + 360, deg)
            is_right = jnp.logical_or(deg >= 347.5, deg < 12.5)
            is_down = jnp.logical_and(deg >= 77.5, deg < 102.5)
            is_left = jnp.logical_and(deg >= 167.5, deg < 192.5)
            is_up = jnp.logical_and(deg >= 257.5, deg < 282.5)
            quad_idx = (deg // 90).astype(jnp.int32)
            diag_sprite_idx = 1 + (quad_idx * 2)
            final_arrow_idx = jnp.select([is_right, is_down, is_left, is_up], [0, 2, 4, 6], default=diag_sprite_idx)

            # --- Snake Logic ---
            snake_frame_idx = (state.step_counter // 8) % 2

            # Pick the mask
            m = jax.lax.cond(
                is_arrow,
                lambda: self.arrow_sprites[final_arrow_idx],
                lambda: jax.lax.switch(state.enemies_type[i], [
                    lambda: self.harmonized_enemy_masks[0], # 0: Generic
                    lambda: self.harmonized_enemy_masks[1], # 1: Scorpion
                    lambda: self.harmonized_enemy_masks[2], # 2: Ant
                    lambda: self.harmonized_enemy_masks[3], # 3: Vulture
                    lambda: self.snake_anim_masks[snake_frame_idx], # 4: SNAKE (Animated)
                    lambda: self.harmonized_enemy_masks[4], # 5: Archer (Shifted index in list due to snake handling)
                    lambda: self.harmonized_enemy_masks[5], # 6: Arrow (Fallback)
                ])
            )

            return jax.lax.cond(jnp.logical_and(state.enemies_active[i], is_gameplay),
                                lambda _r: self.jr.render_at(_r, state.enemies_x[i].astype(jnp.int32), state.enemies_y[i].astype(jnp.int32), m),
                                lambda _r: _r, r)

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