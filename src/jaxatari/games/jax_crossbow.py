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
                'backgrounds/map_1.npy',
                'backgrounds/map_2.npy',
                'backgrounds/map_3.npy',
                'backgrounds/map_4.npy',
                'backgrounds/map_5.npy',
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
        {'name': 'cursor', 'type': 'single', 'file': 'cursor.npy'},
        {'name': 'enemy', 'type': 'single', 'file': 'enemy.npy'},
        {'name': 'shot', 'type': 'single', 'file': 'shot.npy'},
        {'name': 'digits', 'type': 'digits', 'pattern': 'digits/digit_{}.npy'},
    )

class GamePhase:
    START_SCREEN = 0
    GET_READY = 1
    MAP_1 = 2
    MAP_2 = 3
    MAP_3 = 4
    MAP_4 = 5
    MAP_5 = 6
    MAP_6 = 7

class EnemyType:
    GENERIC = 0
    SCORPION = 1
    ANT = 2
    VULTURE = 3

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

    def _handle_common_death_logic(self, state: CrossbowState, any_friend_hit: chex.Array, rng_key: chex.PRNGKey) -> Tuple[CrossbowState, bool]:

        is_dying = state.dying_timer > 0

        # Init Scatter on new hit
        state = jax.lax.cond(any_friend_hit, lambda s: self._init_scatter_pixels(s, rng_key), lambda s: s, state)

        # Update Scatter Physics if Dying
        state = jax.lax.cond(is_dying, lambda s: self._update_scatter_pixels(s), lambda s: s, state)

        # Update Timer
        new_timer = jnp.where(any_friend_hit, self.consts.DYING_DURATION, jnp.maximum(0, state.dying_timer - 1))

        # Handle Respawn / End of Death
        timer_finished = jnp.logical_and(state.dying_timer > 0, new_timer == 0)

        # Reset friend to 0 only when timer finishes
        friend_x_next = jnp.where(timer_finished, 0, state.friend_x).astype(jnp.int32)

        # Clear scatter pixels when finished
        scatter_active_next = jnp.where(timer_finished, jnp.zeros_like(state.scatter_px_active), state.scatter_px_active)

        new_lives = jnp.maximum(0, state.lives - jnp.where(any_friend_hit, 1, 0))
        is_game_over = jnp.logical_and(any_friend_hit, state.lives == 0)

        new_state = state._replace(
            lives=new_lives.astype(jnp.int32),
            dying_timer=new_timer.astype(jnp.int32),
            friend_x=friend_x_next,
            scatter_px_active=scatter_active_next
        )
        return new_state, is_game_over

    def _cursor_step(self, state: CrossbowState, action: chex.Array) -> CrossbowState:
        is_up = jnp.isin(action, jnp.array([Action.UP, Action.UPRIGHT, Action.UPLEFT, Action.UPFIRE, Action.UPRIGHTFIRE, Action.UPLEFTFIRE]))
        is_down = jnp.isin(action, jnp.array([Action.DOWN, Action.DOWNRIGHT, Action.DOWNLEFT, Action.DOWNFIRE, Action.DOWNRIGHTFIRE, Action.DOWNLEFTFIRE]))
        is_left = jnp.isin(action, jnp.array([Action.LEFT, Action.UPLEFT, Action.DOWNLEFT, Action.LEFTFIRE, Action.UPLEFTFIRE, Action.DOWNLEFTFIRE]))
        is_right = jnp.isin(action, jnp.array([Action.RIGHT, Action.UPRIGHT, Action.DOWNRIGHT, Action.RIGHTFIRE, Action.UPRIGHTFIRE, Action.DOWNRIGHTFIRE]))

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

    def _desert_map_logic(self, state: CrossbowState, action: chex.Array) -> Tuple[CrossbowState, bool]:
        rng, spawn_key, type_key, scatter_key = jax.random.split(state.key, 4)
        is_dying = state.dying_timer > 0

        HIT_TOLERANCE = 8
        cx, cy = state.cursor_x, state.cursor_y
        ex, ey = state.enemies_x, state.enemies_y

        hit_x = jnp.logical_and(cx < ex + self.consts.ENEMY_SIZE[0] + HIT_TOLERANCE, cx + self.consts.CURSOR_SIZE[0] > ex - HIT_TOLERANCE)
        hit_y = jnp.logical_and(cy < ey + self.consts.ENEMY_SIZE[1] + HIT_TOLERANCE, cy + self.consts.CURSOR_SIZE[1] > ey - HIT_TOLERANCE)
        is_hit = jnp.logical_and(hit_x, hit_y)

        valid_kill = jnp.logical_and(state.is_firing, jnp.logical_and(state.enemies_active, is_hit))
        surviving_enemies = jnp.logical_and(state.enemies_active, jnp.logical_not(valid_kill))

        is_ant = state.enemies_type == EnemyType.ANT
        is_vulture = state.enemies_type == EnemyType.VULTURE

        move_freq = jnp.select([is_ant, is_vulture], [2, 3], default=4)
        should_move = (state.step_counter % move_freq) == 0

        new_x = state.enemies_x + jnp.where(should_move, -1, 0)
        new_y = state.enemies_y + jnp.where(jnp.logical_and(is_vulture, should_move), 1, 0)

        final_x = jnp.where(surviving_enemies, new_x, state.enemies_x)
        final_y = jnp.where(surviving_enemies, new_y, state.enemies_y)

        DEPTH_TOLERANCE = 15
        fx, fy = state.friend_x, state.friend_y
        danger_x = jnp.logical_and(
            final_x < fx + self.consts.FRIEND_SIZE[0],
            final_x + self.consts.ENEMY_SIZE[0] > fx
        )
        danger_y = jnp.logical_and(
            final_y < fy + self.consts.FRIEND_SIZE[1] + DEPTH_TOLERANCE,
            final_y + self.consts.ENEMY_SIZE[1] > fy - 5
        )
        friend_hit = jnp.logical_and(jnp.logical_and(danger_x, danger_y), jnp.logical_and(surviving_enemies, state.friend_active))
        any_friend_hit = jnp.logical_and(jnp.any(friend_hit), jnp.logical_not(is_dying))

        spawn_chance = jax.random.uniform(spawn_key, shape=(self.consts.MAX_ENEMIES,)) < 0.03
        should_spawn = jnp.logical_and(jnp.logical_not(surviving_enemies), spawn_chance)
        new_types = jax.random.randint(type_key, (self.consts.MAX_ENEMIES,), 1, 4)

        spawn_x = jax.random.randint(spawn_key, (self.consts.MAX_ENEMIES,), 140, 160)
        spawn_ground_y = jax.random.randint(spawn_key, (self.consts.MAX_ENEMIES,), self.consts.GROUND_Y_MIN, self.consts.GROUND_Y_MAX)
        spawn_air_y = jax.random.randint(spawn_key, (self.consts.MAX_ENEMIES,), 20, 80)
        spawn_y = jnp.where(new_types == EnemyType.VULTURE, spawn_air_y, spawn_ground_y)

        enemies_active_next = jnp.logical_or(surviving_enemies, should_spawn)
        final_x = jnp.where(should_spawn, spawn_x, final_x)
        final_y = jnp.where(should_spawn, spawn_y, final_y)
        final_types = jnp.where(should_spawn, new_types, state.enemies_type)
        final_active = jnp.logical_and(enemies_active_next, final_x > 0)
        final_active = jnp.logical_and(final_active, final_y < self.consts.PLAY_AREA_HEIGHT)

        new_score = state.score + jnp.sum(valid_kill) * 2

        intermediate_state = state._replace(
            score=new_score.astype(jnp.int32),
            enemies_active=final_active,
            enemies_x=final_x.astype(jnp.int32),
            enemies_y=final_y.astype(jnp.int32),
            enemies_type=final_types.astype(jnp.int32),
            key=rng
        )

        return self._handle_common_death_logic(intermediate_state, any_friend_hit, scatter_key)


    def _generic_map_logic(self, state: CrossbowState, action: chex.Array) -> Tuple[CrossbowState, bool]:
        rng, spawn_key, scatter_key = jax.random.split(state.key, 3)
        is_dying = state.dying_timer > 0

        HIT_TOLERANCE = 8
        cx, cy = state.cursor_x, state.cursor_y
        ex, ey = state.enemies_x, state.enemies_y
        hit_x = jnp.logical_and(cx < ex + self.consts.ENEMY_SIZE[0] + HIT_TOLERANCE, cx + self.consts.CURSOR_SIZE[0] > ex - HIT_TOLERANCE)
        hit_y = jnp.logical_and(cy < ey + self.consts.ENEMY_SIZE[1] + HIT_TOLERANCE, cy + self.consts.CURSOR_SIZE[1] > ey - HIT_TOLERANCE)
        is_hit = jnp.logical_and(hit_x, hit_y)
        valid_kill = jnp.logical_and(state.is_firing, jnp.logical_and(state.enemies_active, is_hit))
        surviving_enemies = jnp.logical_and(state.enemies_active, jnp.logical_not(valid_kill))

        should_move_enemy = state.step_counter % 3 == 0
        new_enemy_y = state.enemies_y + jnp.where(should_move_enemy, 1, 0)
        fx, fy = state.friend_x, state.friend_y
        danger_x = jnp.logical_and(ex < fx + self.consts.FRIEND_SIZE[0], ex + self.consts.ENEMY_SIZE[0] > fx)
        danger_y = jnp.logical_and(new_enemy_y < fy + self.consts.FRIEND_SIZE[1], new_enemy_y + self.consts.ENEMY_SIZE[1] > fy)
        friend_hit = jnp.logical_and(jnp.logical_and(danger_x, danger_y), jnp.logical_and(surviving_enemies, state.friend_active))
        any_friend_hit = jnp.logical_and(jnp.any(friend_hit), jnp.logical_not(is_dying))

        spawn_chance = jax.random.uniform(spawn_key, shape=(self.consts.MAX_ENEMIES,)) < 0.05
        should_spawn = jnp.logical_and(jnp.logical_not(surviving_enemies), spawn_chance)
        spawn_x = jax.random.randint(spawn_key, (self.consts.MAX_ENEMIES,), 20, self.consts.WIDTH - 20)
        spawn_y = jax.random.randint(spawn_key, (self.consts.MAX_ENEMIES,), 20, 50)
        enemies_active_next = jnp.logical_or(surviving_enemies, should_spawn)
        final_x = jnp.where(should_spawn, spawn_x, state.enemies_x)
        final_y = jnp.where(should_spawn, spawn_y, new_enemy_y)
        final_types = jnp.where(should_spawn, EnemyType.GENERIC, state.enemies_type)
        final_active = jnp.logical_and(enemies_active_next, final_y < self.consts.PLAY_AREA_HEIGHT)

        score_gain = jnp.sum(valid_kill) * 1
        new_score = state.score + score_gain

        intermediate_state = state._replace(
            score=new_score.astype(jnp.int32),
            enemies_active=final_active,
            enemies_x=final_x.astype(jnp.int32),
            enemies_y=final_y.astype(jnp.int32),
            enemies_type=final_types.astype(jnp.int32),
            is_firing=state.is_firing,
            key=rng
        )

        return self._handle_common_death_logic(intermediate_state, any_friend_hit, scatter_key)

    def _update_game_phase(self, state: CrossbowState, action: chex.Array) -> CrossbowState:
        on_start_screen = state.game_phase == GamePhase.START_SCREEN
        on_get_ready = state.game_phase == GamePhase.GET_READY

        map1_icon_range_x = jnp.logical_and(state.cursor_x >= 47, state.cursor_x <= 64)
        map1_icon_range_y = jnp.logical_and(state.cursor_y >= 27, state.cursor_y <= 51)
        select_map_1 = jnp.logical_and(map1_icon_range_x, map1_icon_range_y)

        map2_icon_range_x = jnp.logical_and(state.cursor_x >= 113, state.cursor_x <= 128)
        map2_icon_range_y = jnp.logical_and(state.cursor_y >= 38, state.cursor_y <= 52)
        select_map_2 = jnp.logical_and(map2_icon_range_x, map2_icon_range_y)

        map3_icon_range_x = jnp.logical_and(state.cursor_x >= 117, state.cursor_x <= 132)
        map3_icon_range_y = jnp.logical_and(state.cursor_y >=  101, state.cursor_y <= 118)
        select_map_3 = jnp.logical_and(map3_icon_range_x, map3_icon_range_y)

        map4_icon_range_x = jnp.logical_and(state.cursor_x >= 53, state.cursor_x <= 68)
        map4_icon_range_y = jnp.logical_and(state.cursor_y >= 96, state.cursor_y <= 118)
        select_map_4 = jnp.logical_and(map4_icon_range_x, map4_icon_range_y)

        map5_icon_range_x = jnp.logical_and(state.cursor_x >= 97, state.cursor_x <= 112)
        map5_icon_range_y = jnp.logical_and(state.cursor_y >= 130, state.cursor_y <= 151)
        select_map_5 = jnp.logical_and(map5_icon_range_x, map5_icon_range_y)

        map6_icon_range_x = jnp.logical_and(state.cursor_x >= 33, state.cursor_x <= 48)
        map6_icon_range_y = jnp.logical_and(state.cursor_y >= 125, state.cursor_y <= 151)
        select_map_6 = jnp.logical_and(map6_icon_range_x, map6_icon_range_y)

        input_trigger = jnp.logical_and(on_start_screen, state.is_firing)
        target_map = jnp.select(
            [select_map_1, select_map_2, select_map_3, select_map_4, select_map_5, select_map_6],
            [GamePhase.MAP_1, GamePhase.MAP_2, GamePhase.MAP_3, GamePhase.MAP_4, GamePhase.MAP_5, GamePhase.MAP_6],
            default=GamePhase.START_SCREEN
        )

        get_ready_done = jnp.logical_and(on_get_ready, state.get_ready_timer == 0)

        next_phase = jnp.where(jnp.logical_and(input_trigger, state.selected_target_map != GamePhase.START_SCREEN), GamePhase.GET_READY, jnp.where(get_ready_done, state.selected_target_map, state.game_phase))
        next_target_map = jnp.where(input_trigger, target_map, state.selected_target_map)
        next_get_ready_timer = jnp.where(input_trigger, self.consts.GET_READY_DURATION, jnp.where(on_get_ready, jnp.maximum(0, state.get_ready_timer - 1), 0))
        next_fade_in_timer = jnp.where(get_ready_done, self.consts.FADE_IN_DURATION, jnp.maximum(0, state.fade_in_timer - 1))

        return state._replace(
            game_phase=next_phase.astype(jnp.int32),
            selected_target_map=next_target_map.astype(jnp.int32),
            get_ready_timer=next_get_ready_timer.astype(jnp.int32),
            fade_in_timer=next_fade_in_timer.astype(jnp.int32)
        )

    def reset(self, key: chex.PRNGKey = jax.random.PRNGKey(42)) -> Tuple[CrossbowObservation, CrossbowState]:
        state_key, _ = jax.random.split(key)
        num_pixels = self.consts.MAX_SCATTER_PIXELS
        state = CrossbowState(
            cursor_x=jnp.array(self.consts.WIDTH // 2, dtype=jnp.int32),
            cursor_y=jnp.array(self.consts.HEIGHT // 2, dtype=jnp.int32),
            is_firing=jnp.array(False),
            friend_x=jnp.array(0, dtype=jnp.int32),
            friend_y=jnp.array(128, dtype=jnp.int32),
            friend_active=jnp.array(True),
            dying_timer=jnp.array(0, dtype=jnp.int32),

            get_ready_timer=jnp.array(0, dtype=jnp.int32),
            fade_in_timer=jnp.array(0, dtype=jnp.int32),
            selected_target_map=jnp.array(0, dtype=jnp.int32),

            scatter_px_x=jnp.zeros(num_pixels, dtype=jnp.float32),
            scatter_px_y=jnp.zeros(num_pixels, dtype=jnp.float32),
            scatter_px_dx=jnp.zeros(num_pixels, dtype=jnp.float32),
            scatter_px_dy=jnp.zeros(num_pixels, dtype=jnp.float32),
            scatter_px_color_idx=jnp.zeros(num_pixels, dtype=jnp.uint8),
            scatter_px_active=jnp.zeros(num_pixels, dtype=bool),

            enemies_x=jnp.zeros(self.consts.MAX_ENEMIES, dtype=jnp.int32),
            enemies_y=jnp.zeros(self.consts.MAX_ENEMIES, dtype=jnp.int32),
            enemies_active=jnp.zeros(self.consts.MAX_ENEMIES, dtype=bool),
            enemies_type=jnp.zeros(self.consts.MAX_ENEMIES, dtype=jnp.int32),
            game_phase=jnp.array(GamePhase.START_SCREEN, dtype=jnp.int32),
            score=jnp.array(0, dtype=jnp.int32),
            lives=jnp.array(self.consts.MAX_LIVES, dtype=jnp.int32),
            step_counter=jnp.array(0, dtype=jnp.int32),
            key=state_key
        )
        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: CrossbowState, action: chex.Array):
        prev_score = state.score
        new_key, step_key = jax.random.split(state.key)
        state = state._replace(key=step_key)

        state = self._cursor_step(state, action)
        state = self._update_game_phase(state, action)

        is_gameplay = state.game_phase >= GamePhase.MAP_1

        state = jax.lax.cond(jnp.logical_and(is_gameplay, state.friend_active), lambda s: self._friend_step(s), lambda s: s, state)

        def _combat_router(s):
            return jax.lax.cond(
                s.game_phase == GamePhase.MAP_1,
                lambda _s: self._desert_map_logic(_s, action),
                lambda _s: self._generic_map_logic(_s, action),
                s
            )

        state, game_over = jax.lax.cond(jnp.logical_and(is_gameplay, state.friend_active), _combat_router, lambda s: (s, False), state)
        state = state._replace(step_counter=state.step_counter + 1, key=new_key)
        reward = (state.score - prev_score).astype(float)
        done = jnp.logical_or(game_over, state.step_counter > 4000)

        return self._get_observation(state), state, reward, done, self._get_info(state)

    def _get_observation(self, state):
        return CrossbowObservation(state.cursor_x, state.cursor_y, state.friend_x, state.game_phase, state.lives, state.score)
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
        self.config = render_utils.RendererConfig(game_dimensions=(210, 160), channels=3)
        self.jr = render_utils.JaxRenderingUtils(self.config)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        sprite_path = os.path.join(base_dir, "sprites", "crossbow")

        (self.PALETTE, self.SHAPE_MASKS, self.BACKGROUND, self.COLOR_TO_ID, self.FLIP_OFFSETS) = self.jr.load_and_setup_assets(self.consts.ASSET_CONFIG, sprite_path)
        self.pixel_masks = {c: jnp.array([[c]], dtype=jnp.uint8) for c in range(1, 9)}

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: CrossbowState):
        raster = self.jr.create_object_raster(self.SHAPE_MASKS["backgrounds"][state.game_phase])
        is_gameplay = state.game_phase >= GamePhase.MAP_1
        is_dying = state.dying_timer > 0

        # Friend
        friend_walk_mask = self.SHAPE_MASKS["friend"][(state.step_counter // 8) % len(self.SHAPE_MASKS["friend"])]
        raster = jax.lax.cond(jnp.logical_and(state.friend_active, jnp.logical_and(is_gameplay, jnp.logical_not(is_dying))), lambda r: self.jr.render_at(r, state.friend_x, state.friend_y, friend_walk_mask), lambda r: r, raster)

        # Scatter Pixels
        def _draw_scatter_pixel(i, r):
            px_x = state.scatter_px_x[i].astype(jnp.int32); px_y = state.scatter_px_y[i].astype(jnp.int32); color_idx = state.scatter_px_color_idx[i]
            pixel_mask = jax.lax.switch(color_idx - 1, [lambda: self.pixel_masks[c] for c in range(1, 9)])
            return jax.lax.cond(state.scatter_px_active[i], lambda _r: self.jr.render_at(_r, px_x, px_y, pixel_mask), lambda _r: _r, r)
        raster = jax.lax.cond(is_dying, lambda r: jax.lax.fori_loop(0, self.consts.MAX_SCATTER_PIXELS, _draw_scatter_pixel, r), lambda r: r, raster)

        # Enemies
        mask_generic = self.SHAPE_MASKS["enemy"]; mask_scorpion = self.SHAPE_MASKS["scorpion"]; mask_ant = self.SHAPE_MASKS["ant"]; mask_vulture = self.SHAPE_MASKS["vulture"]
        def _draw_enemy(i, r):
            selected_mask = jax.lax.switch(state.enemies_type[i], [lambda: mask_generic, lambda: mask_scorpion, lambda: mask_ant, lambda: mask_vulture])
            return jax.lax.cond(jnp.logical_and(state.enemies_active[i], is_gameplay), lambda _r: self.jr.render_at(_r, state.enemies_x[i], state.enemies_y[i], selected_mask), lambda _r: _r, r)
        raster = jax.lax.fori_loop(0, self.consts.MAX_ENEMIES, _draw_enemy, raster)

        raster = self.jr.render_at(raster, state.cursor_x, state.cursor_y, self.SHAPE_MASKS["cursor"])
        raster = jax.lax.cond(state.is_firing, lambda r: self.jr.render_at(r, state.cursor_x, state.cursor_y, self.SHAPE_MASKS["shot"]), lambda r: r, raster)

        digit_masks = self.SHAPE_MASKS["digits"]
        def _get_number_of_digits(val): return jax.lax.cond(val < 10, lambda: 1, lambda: jax.lax.cond(val < 100, lambda: 2, lambda: jax.lax.cond(val < 1000, lambda: 3, lambda: jax.lax.cond(val < 10000, lambda: 4, lambda: jax.lax.cond(val < 100000, lambda: 5, lambda: 6)))))
        player_score_digits = self.jr.int_to_digits(state.score, max_digits=6)
        num_score_digits = _get_number_of_digits(state.score)
        raster = self.jr.render_label_selective(raster, 98 - 8 * (num_score_digits - 1), 186, player_score_digits, digit_masks, 6 - num_score_digits, num_score_digits, spacing=8, max_digits_to_render=6)

        img = self.jr.render_from_palette(raster, self.PALETTE)

        is_fading_out = jnp.logical_and(state.game_phase == GamePhase.GET_READY, state.get_ready_timer < self.consts.FADE_OUT_DURATION)
        is_fading_in = state.fade_in_timer > 0

        def _apply_fade(image):
            factor = jax.lax.cond(
                is_fading_out,
                lambda: state.get_ready_timer / self.consts.FADE_OUT_DURATION,
                lambda: (self.consts.FADE_IN_DURATION - state.fade_in_timer) / self.consts.FADE_IN_DURATION
            )
            return (image * factor).astype(jnp.uint8)

        img = jax.lax.cond(jnp.logical_or(is_fading_out, is_fading_in), _apply_fade, lambda x: x, img)
        return img