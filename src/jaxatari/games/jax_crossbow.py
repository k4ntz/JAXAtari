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


# --- ASSETS ---
def _get_default_asset_config() -> tuple:
    return (
        {'name': 'background', 'type': 'background', 'file': 'background.npy'},
        {'name': 'backgrounds', 'type': 'group',
         'files': [
             'backgrounds/start_screen.npy',
             'backgrounds/get_ready.npy',
             'backgrounds/desert_map.npy',
             'backgrounds/cave_map.npy',
             'backgrounds/forest_map.npy',
             'backgrounds/map_4.npy',
             'backgrounds/castle_hall_map.npy',
             'backgrounds/map_6.npy'
         ]},
        {'name': 'friend', 'type': 'group',
         'files': ['friend/friend_walking_1.npy',
                   'friend/friend_walking_2.npy',
                   'friend/friend_walking_3.npy']},
        {'name': 'scorpion', 'type': 'single', 'file': 'scorpion.npy'},
        {'name': 'ant', 'type': 'single', 'file': 'ant.npy'},
        {'name': 'vulture', 'type': 'single', 'file': 'vulture.npy'},
        {'name': 'spawn', 'type': 'single', 'file': 'spawn.npy'},
        {'name': 'lava_rock', 'type': 'single', 'file': 'lava_rock.npy'},
        {'name': 'falling_rock', 'type': 'single', 'file': 'falling_rock.npy'},
        {'name': 'resting_rock', 'type': 'single', 'file': 'spawn.npy'},
        {'name': 'monkey', 'type': 'single', 'file': 'monkey.npy'},
        {'name': 'coconut', 'type': 'single', 'file': 'coconut.npy'},
        {'name': 'voracious_plant', 'type': 'single', 'file': 'voracious_plant.npy'},
        {'name': 'snake', 'type': 'group', 'files': ['snake_1.npy', 'snake_2.npy']},
        {
            'name': 'bat', 'type': 'group',
            'files': [
                'enemies/cavern/bat/bat_spawning_1.npy',
                'enemies/cavern/bat/bat_spawning_2.npy',
                'enemies/cavern/bat/bat_spawning_3.npy',
                'enemies/cavern/bat/bat_flying_1.npy',
                'enemies/cavern/bat/bat_flying_2.npy'
            ]
        },
        {
            'name': 'stalactite', 'type': 'group',
            'files': [
                'enemies/cavern/stalactite/stalactite_falling.npy',
                'enemies/cavern/stalactite/stalactite_hanging.npy'
            ]
        },
        {'name': 'castle_arrow', 'type': 'single', 'file': 'enemies/castle_hall/arrow.npy'},
        {'name': 'archer', 'type': 'single', 'file': 'archer.npy'},
        {'name': 'arrow', 'type': 'single', 'file': 'arrow.npy'},
        {'name': 'cursor', 'type': 'single', 'file': 'cursor.npy'},
        {'name': 'enemy', 'type': 'single', 'file': 'enemy.npy'},
        {'name': 'shot', 'type': 'single', 'file': 'shot.npy'},
        {'name': 'digits', 'type': 'digits', 'pattern': 'digits/digit_{}.npy'},
    )


# --- SCATTER PIXELS ---
def _get_initial_scatter_state(consts: 'CrossbowConstants'):
    num_pixels = consts.MAX_SCATTER_PIXELS
    rel_x = jnp.tile(jnp.arange(0, 8), 13)[:num_pixels]
    rel_y = jnp.repeat(jnp.arange(0, 27, 2), 8)[:num_pixels]
    colors = jnp.concatenate([jnp.full(32, 7), jnp.full(32, 5), jnp.full(36, 6)])[:num_pixels]
    center_x, center_y = 4.0, 13.5
    base_dx = (rel_x - center_x) * 0.3
    base_dy = (rel_y - center_y) * 0.3
    return rel_x, rel_y, base_dx, base_dy, colors


# --- GAME PHASES ---
class GamePhase:
    START_SCREEN = 0
    GET_READY = 1
    DESERT_MAP = 2
    CAVE_MAP = 3
    VOLCANO_MAP = 4
    JUNGLE_MAP = 5
    DRAWBRIDGE = 6
    CASTLE_HALL = 7


# --- ENEMIES ---
class EnemyType:
    GENERIC = 0
    SCORPION = 1
    ANT = 2
    VULTURE = 3
    SPAWN = 4

    # Volcano
    BURNING_LAVA = 5
    FALLING_ROCK = 6
    RESTING_ROCK = 7

    # Jungle
    MONKEY = 8
    COCONUT = 9
    VORACIOUS_PLANT = 10

    # Original extra enemies
    SNAKE = 11
    ARCHER = 12
    ARROW = 13

    # Cavern
    BAT = 14
    STALACTITE_FALLING = 15
    STALACTITE_HANGING = 16

    # Castle Hall
    CASTLE_ARROW = 17
    CASTLE_FALLING_LAVA = 18


# --- CONSTANTS ---
class CrossbowConstants(NamedTuple):
    WIDTH: int = 160
    HEIGHT: int = 210

    CURSOR_SPEED: int = 2
    CURSOR_SIZE: Tuple[int, int] = (4, 4)
    FRIEND_SPEED: float = 0.5
    ENEMY_SPEED: float = 0.5
    MAX_ENEMIES: int = 6
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

    BAT_DIMENSIONS: Tuple[int, int] = (8, 9)
    STALACTITE_DIMENSIONS: Tuple[int, int] = (8, 16)

    DEMON_MAP_CENTER_X: int = 80
    DEMON_MAP_CENTER_Y: int = 90

    ASSET_CONFIG: tuple = _get_default_asset_config()

    ROPE_1_POS: Tuple[int, int] = (110, 60)
    ROPE_2_POS: Tuple[int, int] = (110, 100)


# --- STATE ---
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
    enemies_age: chex.Array
    game_phase: chex.Array
    score: chex.Array
    lives: chex.Array
    step_counter: chex.Array
    key: chex.PRNGKey
    rope_1_broken: chex.Array
    rope_2_broken: chex.Array


# --- OBSERVATION & INFO ---
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

    @property
    def num_actions(self) -> int:
        return 18

    @property
    def max_episode_steps(self) -> int:
        return 4000

    def get_legal_actions(self, state: CrossbowState) -> chex.Array:
        return jnp.ones((self.num_actions,), dtype=bool)

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

        is_bat = state.enemies_type == EnemyType.BAT
        is_bat_attacking = jnp.logical_and(is_bat, jnp.logical_or(state.enemies_dx != 0, state.enemies_dy != 0))
        fx, fy = state.friend_x, state.friend_y
        bat_on_x = jnp.logical_and(
            state.enemies_x < fx + self.consts.FRIEND_SIZE[0],
            state.enemies_x + self.consts.BAT_DIMENSIONS[0] > fx
        )
        bat_on_y = jnp.logical_and(
            state.enemies_y < fy + 10,
            state.enemies_y + self.consts.BAT_DIMENSIONS[1] > fy - 5
        )
        bat_on_friend = jnp.any(jnp.logical_and(
            jnp.logical_and(bat_on_x, bat_on_y),
            jnp.logical_and(is_bat_attacking, state.enemies_active)
        ))

        should_move = jnp.logical_and(state.step_counter % 8 == 0, jnp.logical_not(bat_on_friend))
        new_x = state.friend_x + jnp.where(should_move & state.friend_active, 1, 0)
        reached_goal = new_x > self.consts.WIDTH
        final_x = jnp.where(is_dying, state.friend_x, jnp.where(reached_goal, 0, new_x))

        return state._replace(
            friend_x=final_x.astype(jnp.int32),
            friend_y=jnp.where(state.game_phase == GamePhase.CAVE_MAP, 100, 128).astype(jnp.int32)
        )

    def _get_enemy_shape(self, enemies_type: chex.Array) -> Tuple[chex.Array, chex.Array]:
        is_bat = enemies_type == EnemyType.BAT
        is_stallactite = jnp.logical_or(
            enemies_type == EnemyType.STALACTITE_HANGING,
            enemies_type == EnemyType.STALACTITE_FALLING
        )

        width = jnp.select(
            [is_bat, is_stallactite],
            [self.consts.BAT_DIMENSIONS[0], self.consts.STALACTITE_DIMENSIONS[0]],
            default=self.consts.ENEMY_SIZE[0]
        ).astype(jnp.int32)

        height = jnp.select(
            [is_bat, is_stallactite],
            [self.consts.BAT_DIMENSIONS[1], self.consts.STALACTITE_DIMENSIONS[1]],
            default=self.consts.ENEMY_SIZE[1]
        ).astype(jnp.int32)

        return width, height

    def _drawbridge_map_logic(self, state: CrossbowState) -> Tuple[CrossbowState, bool]:
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
            game_phase=jnp.where(finished_map, GamePhase.CASTLE_HALL, state.game_phase)
        )
        return self._handle_common_death_logic(intermediate_state, any_friend_hit, key_scatter)

    def _castle_hall_map_logic(self, state: CrossbowState) -> Tuple[CrossbowState, bool]:
        rng, spawn_key, type_key, scatter_key, lava_key = jax.random.split(state.key, 5)
        is_dying = state.dying_timer > 0
        HIT_TOLERANCE = 8
        LAVA_SPEED = 1.5
        ARROW_FALL_SPEED = 1.5
        MAX_CONCURRENT_ENEMIES = 2

        cx, cy = state.cursor_x, state.cursor_y
        ex, ey = state.enemies_x, state.enemies_y

        hit_x = jnp.logical_and(cx < ex + self.consts.ENEMY_SIZE[0] + HIT_TOLERANCE,
                                cx + self.consts.CURSOR_SIZE[0] > ex - HIT_TOLERANCE)
        hit_y = jnp.logical_and(cy < ey + self.consts.ENEMY_SIZE[1] + HIT_TOLERANCE,
                                cy + self.consts.CURSOR_SIZE[1] > ey - HIT_TOLERANCE)
        is_hit = jnp.logical_and(hit_x, hit_y)
        valid_kill = jnp.logical_and(state.is_firing, jnp.logical_and(state.enemies_active, is_hit))
        surviving_enemies = jnp.logical_and(state.enemies_active, jnp.logical_not(valid_kill))

        is_castle_arrow = state.enemies_type == EnemyType.CASTLE_ARROW
        is_castle_lava = state.enemies_type == EnemyType.CASTLE_FALLING_LAVA
        reward = jnp.sum(jnp.where(valid_kill, 100, 0))

        arrow_dy = jnp.where(is_castle_arrow, ARROW_FALL_SPEED, 0.0)

        target_x = state.friend_x + 4.0
        target_y = state.friend_y + 13.0
        lava_center_x = state.enemies_x + 4.0
        lava_center_y = state.enemies_y + 4.0
        delta_x = target_x - lava_center_x
        delta_y = target_y - lava_center_y
        dist = jnp.sqrt(delta_x**2 + delta_y**2) + 1e-5
        lava_dx = jnp.where(is_castle_lava, (delta_x / dist) * LAVA_SPEED, 0.0)
        lava_dy = jnp.where(is_castle_lava, (delta_y / dist) * LAVA_SPEED, 0.0)

        new_x = state.enemies_x + lava_dx
        new_y = state.enemies_y + arrow_dy + lava_dy

        current_enemy_count = jnp.sum(surviving_enemies)
        spawn_chance = jax.random.uniform(spawn_key, shape=(self.consts.MAX_ENEMIES,)) < 0.03

        spawn_count = jnp.cumsum(spawn_chance & ~surviving_enemies)
        allowed_spawns = spawn_count <= (MAX_CONCURRENT_ENEMIES - current_enemy_count)
        should_spawn = jnp.logical_and(jnp.logical_not(surviving_enemies), jnp.logical_and(spawn_chance, allowed_spawns))

        type_roll = jax.random.uniform(type_key, (self.consts.MAX_ENEMIES,))
        spawn_type = jnp.where(type_roll < 0.5, EnemyType.CASTLE_ARROW, EnemyType.CASTLE_FALLING_LAVA)

        spawn_x_arrow = jax.random.randint(spawn_key, (self.consts.MAX_ENEMIES,), 10, 60).astype(jnp.float32)
        spawn_x_lava = jax.random.randint(lava_key, (self.consts.MAX_ENEMIES,), 100, 150).astype(jnp.float32)
        spawn_y = jax.random.randint(spawn_key, (self.consts.MAX_ENEMIES,), 18, 30).astype(jnp.float32)

        spawn_x = jnp.where(spawn_type == EnemyType.CASTLE_ARROW, spawn_x_arrow, spawn_x_lava)

        final_x = jnp.where(should_spawn, spawn_x, new_x)
        final_y = jnp.where(should_spawn, spawn_y, new_y)
        final_type = jnp.where(should_spawn, spawn_type, state.enemies_type)

        is_on_screen = jnp.logical_and(final_x > -20, final_x < self.consts.WIDTH + 20)
        is_on_screen_y = jnp.logical_and(final_y > -20, final_y < self.consts.HEIGHT + 20)
        final_active = jnp.logical_and(
            jnp.logical_or(surviving_enemies, should_spawn),
            jnp.logical_and(is_on_screen, is_on_screen_y)
        )

        fx, fy = state.friend_x, state.friend_y
        danger_x = jnp.logical_and(final_x < fx + self.consts.FRIEND_SIZE[0],
                                   final_x + self.consts.ENEMY_SIZE[0] > fx)
        danger_y = jnp.logical_and(final_y < fy + self.consts.FRIEND_SIZE[1],
                                   final_y + self.consts.ENEMY_SIZE[1] > fy)
        friend_hit = jnp.any(jnp.logical_and(jnp.logical_and(danger_x, danger_y), final_active))
        any_friend_hit = jnp.logical_and(jnp.logical_and(friend_hit, state.friend_active), jnp.logical_not(is_dying))

        intermediate_state = state._replace(
            score=(state.score + reward).astype(jnp.int32),
            enemies_active=final_active,
            enemies_x=final_x.astype(jnp.float32),
            enemies_y=final_y.astype(jnp.float32),
            enemies_type=final_type.astype(jnp.int32),
            key=rng
        )
        return self._handle_common_death_logic(intermediate_state, any_friend_hit, scatter_key)

    def _desert_map_logic(self, state: CrossbowState) -> Tuple[CrossbowState, bool]:
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
        enemy_types = jnp.array([EnemyType.SCORPION, EnemyType.ANT, EnemyType.VULTURE, EnemyType.SNAKE])
        type_indices = jax.random.randint(type_key, (self.consts.MAX_ENEMIES,), 0, 4)
        new_types = enemy_types[type_indices]

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

    def _cavern_map_logic(self, state: CrossbowState) -> Tuple[CrossbowState, bool]:
        rng, spawn_key, type_key, scatter_key, attack_key = jax.random.split(state.key, 5)
        spawn_chance_key, bat_x_key, stal_x_key, air_y_key, ceiling_y_key = jax.random.split(spawn_key, 5)
        is_dying = state.dying_timer > 0

        MAX_BATS = 1
        MAX_STALACTITES = 2
        MAX_ENEMIES = MAX_BATS + MAX_STALACTITES
        BAT_ATTACK_PROB = 0.02
        BAT_ATTACK_SPEED = 1.5
        BAT_KILLING_FRAMES = 45

        HIT_TOLERANCE = 8
        cx, cy = state.cursor_x, state.cursor_y
        ex, ey = state.enemies_x, state.enemies_y
        ew, eh = self._get_enemy_shape(state.enemies_type)

        hit_x = jnp.logical_and(cx < ex + ew + HIT_TOLERANCE, cx + self.consts.CURSOR_SIZE[0] > ex - HIT_TOLERANCE)
        hit_y = jnp.logical_and(cy < ey + eh + HIT_TOLERANCE, cy + self.consts.CURSOR_SIZE[1] > ey - HIT_TOLERANCE)
        is_hit = jnp.logical_and(hit_x, hit_y)

        valid_kill = jnp.logical_and(state.is_firing, jnp.logical_and(state.enemies_active, is_hit))
        surviving_enemies = jnp.logical_and(state.enemies_active, jnp.logical_not(valid_kill))

        is_bat = state.enemies_type == EnemyType.BAT
        is_falling_stal = state.enemies_type == EnemyType.STALACTITE_FALLING

        is_bat_spawning = jnp.logical_and(is_bat, state.enemies_age < 48)
        is_bat_flying = jnp.logical_and(is_bat, state.enemies_age >= 48)
        is_bat_attacking = jnp.logical_and(is_bat, jnp.logical_or(state.enemies_dx != 0, state.enemies_dy != 0))
        is_bat_just_flying = jnp.logical_and(is_bat_flying, jnp.logical_not(is_bat_attacking))

        attack_probs = jax.random.uniform(attack_key, shape=(self.consts.MAX_ENEMIES,))
        start_attacking = jnp.logical_and(
            jnp.logical_and(is_bat_just_flying, surviving_enemies),
            attack_probs < BAT_ATTACK_PROB
        )

        target_x = state.friend_x + 4.0
        target_y = state.friend_y
        bat_center_x = state.enemies_x + 4.0
        bat_center_y = state.enemies_y + 4.0
        delta_x = target_x - bat_center_x
        delta_y = target_y - bat_center_y
        dist = jnp.sqrt(delta_x**2 + delta_y**2) + 1e-5
        attack_dx = (delta_x / dist) * BAT_ATTACK_SPEED
        attack_dy = (delta_y / dist) * BAT_ATTACK_SPEED

        new_dx = jnp.where(start_attacking, attack_dx, state.enemies_dx)
        new_dy = jnp.where(start_attacking, attack_dy, state.enemies_dy)

        fx, fy = state.friend_x, state.friend_y
        bat_on_x = jnp.logical_and(ex < fx + self.consts.FRIEND_SIZE[0], ex + self.consts.BAT_DIMENSIONS[0] > fx)
        bat_on_y = jnp.logical_and(ey < fy + 10, ey + self.consts.BAT_DIMENSIONS[1] > fy - 5)
        bat_on_friend = jnp.logical_and(
            jnp.logical_and(bat_on_x, bat_on_y),
            jnp.logical_and(is_bat_attacking, surviving_enemies)
        )

        bat_spawn_move = jnp.logical_and(is_bat_spawning, state.step_counter % 4 == 0)
        bat_fly_move = jnp.logical_and(is_bat_just_flying, state.step_counter % 2 == 0)
        bat_should_fly = jnp.logical_or(bat_spawn_move, bat_fly_move)
        bat_attack_move = jnp.logical_and(is_bat_attacking, jnp.logical_not(bat_on_friend))

        dx = jnp.where(bat_should_fly, 1.0, 0.0)
        dx = jnp.where(bat_attack_move, new_dx, dx)
        dy = jnp.where(is_falling_stal, 1.0, 0.0)
        dy = jnp.where(bat_attack_move, new_dy, dy)

        new_x = state.enemies_x + dx
        new_y = state.enemies_y + dy
        final_x = jnp.where(surviving_enemies, new_x, state.enemies_x)
        final_y = jnp.where(surviving_enemies, new_y, state.enemies_y)

        new_timer = jnp.where(bat_on_friend, state.enemies_timer + 1, state.enemies_timer)
        new_timer = jnp.where(jnp.logical_not(surviving_enemies), 0, new_timer)

        bat_kills_friend = jnp.any(jnp.logical_and(bat_on_friend, new_timer >= BAT_KILLING_FRAMES))

        DEPTH_TOLERANCE = 10
        danger_x = jnp.logical_and(final_x < fx + self.consts.FRIEND_SIZE[0], final_x + ew > fx)
        danger_y = jnp.logical_and(final_y < fy + self.consts.FRIEND_SIZE[1] + DEPTH_TOLERANCE, final_y + eh > fy - 5)
        friend_hit = jnp.logical_and(jnp.logical_and(danger_x, danger_y), jnp.logical_and(surviving_enemies, state.friend_active))

        stallactice_hit = jnp.logical_and(friend_hit, jnp.logical_not(is_bat))
        any_friend_hit = jnp.logical_and(
            jnp.logical_or(jnp.any(stallactice_hit), bat_kills_friend),
            jnp.logical_not(is_dying)
        )

        spawn_chance = jax.random.uniform(spawn_chance_key, shape=(self.consts.MAX_ENEMIES,)) < 0.10
        should_spawn = jnp.logical_and(jnp.logical_not(surviving_enemies), spawn_chance)

        spawn_is_bat = jax.random.randint(type_key, (self.consts.MAX_ENEMIES,), 0, 2) == 0
        new_types = jnp.where(spawn_is_bat, EnemyType.BAT, EnemyType.STALACTITE_FALLING).astype(jnp.int32)

        active_bats_mask = jnp.logical_and(surviving_enemies, state.enemies_type == EnemyType.BAT)
        active_stal_mask = jnp.logical_and(surviving_enemies, state.enemies_type == EnemyType.STALACTITE_FALLING)
        num_active_bats = jnp.sum(active_bats_mask)
        num_active_stal = jnp.sum(active_stal_mask)

        bat_limit_reached = num_active_bats >= MAX_BATS
        stal_limit_reached = num_active_stal >= MAX_STALACTITES
        spawn_allowed = jnp.where(spawn_is_bat, jnp.logical_not(bat_limit_reached), jnp.logical_not(stal_limit_reached))
        should_spawn = jnp.logical_and(should_spawn, spawn_allowed)

        spawn_bat_x = jax.random.randint(bat_x_key, (self.consts.MAX_ENEMIES,), 10, 30)
        spawn_stal_x = jax.random.randint(stal_x_key, (self.consts.MAX_ENEMIES,), 30, 150)
        spawn_x = jnp.where(spawn_is_bat, spawn_bat_x, spawn_stal_x)
        spawn_air_y = jax.random.randint(air_y_key, (self.consts.MAX_ENEMIES,), 50, 90)
        spawn_ceiling_y = jax.random.randint(ceiling_y_key, (self.consts.MAX_ENEMIES,), 18, 40)
        spawn_y = jnp.where(new_types == EnemyType.BAT, spawn_air_y, spawn_ceiling_y)

        enemies_active_next = jnp.logical_or(surviving_enemies, should_spawn)
        final_x = jnp.where(should_spawn, spawn_x, final_x)
        final_y = jnp.where(should_spawn, spawn_y, final_y)
        final_types = jnp.where(should_spawn, new_types, state.enemies_type)

        final_dx = jnp.where(should_spawn, 0.0, new_dx)
        final_dy = jnp.where(should_spawn, 0.0, new_dy)
        final_timer = jnp.where(should_spawn, 0, new_timer)

        bat_exited = jnp.logical_and(
            final_types == EnemyType.BAT,
            final_x > self.consts.WIDTH + self.consts.BAT_DIMENSIONS[0]
        )
        stal_exited = jnp.logical_and(
            final_types == EnemyType.STALACTITE_FALLING,
            final_y >= self.consts.PLAY_AREA_HEIGHT
        )
        enemy_exited = jnp.logical_or(bat_exited, stal_exited)
        final_active = jnp.logical_and(enemies_active_next, jnp.logical_not(enemy_exited))

        point_values = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 50, 50])
        reward = jnp.sum(jnp.where(valid_kill, point_values[state.enemies_type], 0))
        new_score = state.score + reward
        age_next = jnp.where(final_active, state.enemies_age + 1, 0)
        age_next = jnp.where(should_spawn, 0, age_next)

        intermediate_state = state._replace(
            score=new_score.astype(jnp.int32),
            enemies_active=final_active,
            enemies_x=final_x.astype(jnp.float32),
            enemies_y=final_y.astype(jnp.float32),
            enemies_dx=final_dx.astype(jnp.float32),
            enemies_dy=final_dy.astype(jnp.float32),
            enemies_timer=final_timer.astype(jnp.int32),
            enemies_type=final_types.astype(jnp.int32),
            enemies_age=age_next.astype(jnp.int32),
            key=rng
        )

        return self._handle_common_death_logic(intermediate_state, any_friend_hit, scatter_key)

    def _volcano_map_logic(self, state: CrossbowState) -> Tuple[CrossbowState, bool]:
        rng, spawn_key, physics_key, scatter_key, floor_key = jax.random.split(state.key, 5)
        MAX_CONCURRENT_ENEMIES: int = 1

        # --- Hit Detection ---
        HIT_TOLERANCE = 8
        cx, cy = state.cursor_x, state.cursor_y
        ex, ey = state.enemies_x, state.enemies_y
        hit_x = jnp.logical_and(cx < ex + self.consts.ENEMY_SIZE[0] + HIT_TOLERANCE,
                                cx + self.consts.CURSOR_SIZE[0] > ex - HIT_TOLERANCE)
        hit_y = jnp.logical_and(cy < ey + self.consts.ENEMY_SIZE[1] + HIT_TOLERANCE,
                                cy + self.consts.CURSOR_SIZE[1] > ey - HIT_TOLERANCE)
        is_hit = jnp.logical_and(hit_x, hit_y)
        valid_kill = jnp.logical_and(state.is_firing, jnp.logical_and(state.enemies_active, is_hit))
        surviving_enemies = jnp.logical_and(state.enemies_active, jnp.logical_not(valid_kill))


        # --- Movement ---
        new_dx = state.enemies_dx
        new_dy = state.enemies_dy

        # Falling rocks
        gravity = 0.15
        is_falling_rock = state.enemies_type == EnemyType.FALLING_ROCK
        new_dy = jnp.where(is_falling_rock, state.enemies_dy + gravity, state.enemies_dy)
        new_y = state.enemies_y + new_dy


        # Lava moves horizontally
        is_lava = state.enemies_type == EnemyType.BURNING_LAVA
        lava_dx = jnp.where(state.enemies_x < 20, 0.5, jnp.where(state.enemies_x > self.consts.WIDTH-20, -0.5, state.enemies_dx))
        new_x = state.enemies_x + jnp.where(is_lava, lava_dx, state.enemies_dx)

        # Resting rocks stay static unless hit
        new_x = jnp.where(state.enemies_type == EnemyType.RESTING_ROCK, state.enemies_x, new_x)
        new_y = jnp.where(state.enemies_type == EnemyType.RESTING_ROCK, state.enemies_y, new_y)

        # --- Spawn Logic ---
        current_enemy_count = jnp.sum(surviving_enemies)
        spawn_chance = jax.random.uniform(spawn_key, (self.consts.MAX_ENEMIES,)) < 0.02
        spawn_count = jnp.cumsum(spawn_chance & ~surviving_enemies)
        allowed_spawns = spawn_count <= (MAX_CONCURRENT_ENEMIES - current_enemy_count)
        should_spawn = jnp.logical_and(jnp.logical_not(surviving_enemies), jnp.logical_and(spawn_chance, allowed_spawns))

        spawn_type = jax.random.randint(physics_key, (self.consts.MAX_ENEMIES,), 5, 8)  # Lava/Rock types
        spawn_x = jax.random.randint(physics_key, (self.consts.MAX_ENEMIES,), 20, self.consts.WIDTH-20).astype(jnp.float32)
        is_ground_enemy = jnp.logical_or(spawn_type == EnemyType.BURNING_LAVA, spawn_type == EnemyType.RESTING_ROCK)
        spawn_y = jnp.where(is_ground_enemy, float(self.consts.GROUND_Y_MIN), 0.0)

        final_x = jnp.where(should_spawn, spawn_x, new_x)
        final_y = jnp.where(should_spawn, spawn_y, new_y)
        final_active = jnp.logical_or(surviving_enemies, should_spawn)

        # --- Collision with friend ---
        fx, fy = state.friend_x, state.friend_y
        danger_x = jnp.logical_and(final_x < fx + self.consts.FRIEND_SIZE[0],
                                   final_x + self.consts.ENEMY_SIZE[0] > fx)
        danger_y = jnp.logical_and(final_y < fy + self.consts.FRIEND_SIZE[1],
                                   final_y + self.consts.ENEMY_SIZE[1] > fy)
        any_friend_hit = jnp.any(jnp.logical_and(danger_x, danger_y))

        intermediate_state = state._replace(
            score=(state.score + jnp.sum(valid_kill) * 2).astype(jnp.int32),enemies_active=final_active,enemies_x=final_x,enemies_y=final_y,enemies_dx=new_dx,enemies_dy=new_dy,enemies_type=jnp.where(should_spawn, spawn_type, state.enemies_type),key=rng
        )

        return self._handle_common_death_logic(intermediate_state, any_friend_hit, scatter_key)


    def _jungle_map_logic(self, state: CrossbowState) -> Tuple[CrossbowState, bool]:
        rng, spawn_key, physics_key, scatter_key = jax.random.split(state.key, 4)
        HIT_TOLERANCE = 8
        MAX_CONCURRENT_ENEMIES = 5
        cx, cy = state.cursor_x, state.cursor_y
        ex, ey = state.enemies_x, state.enemies_y
        hit_x = jnp.logical_and(cx < ex + self.consts.ENEMY_SIZE[0] + HIT_TOLERANCE,
                                cx + self.consts.CURSOR_SIZE[0] > ex - HIT_TOLERANCE)

        hit_y = jnp.logical_and(cy < ey + self.consts.ENEMY_SIZE[1] + HIT_TOLERANCE,
                                cy + self.consts.CURSOR_SIZE[1] > ey - HIT_TOLERANCE)

        is_hit = jnp.logical_and(hit_x, hit_y)
        valid_kill = jnp.logical_and(state.is_firing, jnp.logical_and(state.enemies_active, is_hit))
        surviving_enemies = jnp.logical_and(state.enemies_active, jnp.logical_not(valid_kill))

        # --- Movement ---
        new_dx = state.enemies_dx
        new_dy = state.enemies_dy

        # Coconuts fall
        is_coconut = state.enemies_type == EnemyType.COCONUT
        gravity = 0.15
        new_dy = jnp.where(is_coconut, state.enemies_dy + gravity, state.enemies_dy)
        new_y = state.enemies_y + new_dy

        # Monkeys move horizontally
        is_monkey = state.enemies_type == EnemyType.MONKEY
        monkey_dx = jnp.where(state.enemies_x < 20, 0.5, jnp.where(state.enemies_x > self.consts.WIDTH-20, -0.5, state.enemies_dx))
        new_x = state.enemies_x + jnp.where(is_monkey, monkey_dx, state.enemies_dx)

        # Plants static
        new_x = jnp.where(state.enemies_type == EnemyType.VORACIOUS_PLANT, state.enemies_x, new_x)
        new_y = jnp.where(state.enemies_type == EnemyType.VORACIOUS_PLANT, state.enemies_y, new_y)

        current_enemy_count = jnp.sum(surviving_enemies)
        spawn_chance = jax.random.uniform(spawn_key, (self.consts.MAX_ENEMIES,)) < 0.03
        spawn_count = jnp.cumsum(spawn_chance & ~surviving_enemies)
        allowed_spawns = spawn_count <= (MAX_CONCURRENT_ENEMIES - current_enemy_count)
        should_spawn = jnp.logical_and(jnp.logical_not(surviving_enemies), jnp.logical_and(spawn_chance, allowed_spawns))

        spawn_type = jax.random.randint(physics_key, (self.consts.MAX_ENEMIES,), 8, 11)  # Monkey/Coconut/Plant
        spawn_x = jax.random.randint(physics_key, (self.consts.MAX_ENEMIES,), 20, self.consts.WIDTH-20).astype(jnp.float32)
        spawn_y = jnp.select(
            [spawn_type == EnemyType.MONKEY, spawn_type == EnemyType.VORACIOUS_PLANT],
            [50.0, float(self.consts.GROUND_Y_MIN)],
            default=0.0
        )

        final_x = jnp.where(should_spawn, spawn_x, new_x)
        final_y = jnp.where(should_spawn, spawn_y, new_y)
        final_active = jnp.logical_or(surviving_enemies, should_spawn)

        # --- Collision with friend ---
        fx, fy = state.friend_x, state.friend_y
        danger_x = jnp.logical_and(final_x < fx + self.consts.FRIEND_SIZE[0],
                                   final_x + self.consts.ENEMY_SIZE[0] > fx)

        danger_y = jnp.logical_and(final_y < fy + self.consts.FRIEND_SIZE[1],
                                   final_y + self.consts.ENEMY_SIZE[1] > fy)

        any_friend_hit = jnp.any(jnp.logical_and(danger_x, danger_y))

        intermediate_state = state._replace(
            score=(state.score + jnp.sum(valid_kill) * 2).astype(jnp.int32), enemies_active=final_active, enemies_x=final_x,
            enemies_y=final_y,  enemies_dx=new_dx, enemies_dy=new_dy,  enemies_type=jnp.where(should_spawn, spawn_type, state.enemies_type), key=rng

        )

        return self._handle_common_death_logic(intermediate_state, any_friend_hit, scatter_key)



    def _generic_map_logic(self, state: CrossbowState) -> Tuple[CrossbowState, bool]:
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

        target = jnp.select(sel, [GamePhase.DESERT_MAP, GamePhase.CAVE_MAP, GamePhase.VOLCANO_MAP, GamePhase.JUNGLE_MAP, GamePhase.DRAWBRIDGE, GamePhase.CASTLE_HALL], default=GamePhase.START_SCREEN)

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
            enemies_age=jnp.zeros(self.consts.MAX_ENEMIES, dtype=jnp.int32),
            game_phase=jnp.array(GamePhase.START_SCREEN, dtype=jnp.int32), score=jnp.array(0, dtype=jnp.int32),
            lives=jnp.array(self.consts.MAX_LIVES, dtype=jnp.int32), step_counter=jnp.array(0, dtype=jnp.int32), key=state_key,
            rope_1_broken=jnp.array(False), rope_2_broken=jnp.array(False)
        )
        return self.get_obs(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: CrossbowState, action: chex.Array):
        prev_score = state.score
        new_key, step_key = jax.random.split(state.key)
        state = self._cursor_step(state._replace(key=step_key), action)
        state = self._update_game_phase(state, action)
        is_gameplay = state.game_phase >= GamePhase.DESERT_MAP
        state = jax.lax.cond(jnp.logical_and(is_gameplay, state.friend_active), lambda s: self._friend_step(s), lambda s: s, state)
        def _combat_router(s):
            return jax.lax.switch(
                s.game_phase - GamePhase.DESERT_MAP,
                [
                    lambda _s: self._desert_map_logic(_s),      # DESERT_MAP
                    lambda _s: self._cavern_map_logic(_s),      # CAVE_MAP
                    lambda _s: self._jungle_map_logic(_s),      # game_phase 4 - forest_map.npy (jungle)
                    lambda _s: self._volcano_map_logic(_s),     # game_phase 5 - map_4.npy (volcano)
                    lambda _s: self._castle_hall_map_logic(_s),  # DRAWBRIDGE (6)
                    lambda _s: self._drawbridge_map_logic(_s), # CASTLE_HALL (7) - final demon map
                ],
                s
            )


        state, game_over = jax.lax.cond(jnp.logical_and(is_gameplay, state.friend_active), _combat_router, lambda s: (s, False), state)
        state = state._replace(step_counter=state.step_counter + 1, key=new_key)
        return self.get_obs(state), state, (state.score - prev_score).astype(float), jnp.logical_or(game_over, state.step_counter > 4000), self.get_info(state)

    def get_obs(self, state): return CrossbowObservation(state.cursor_x, state.cursor_y, state.friend_x, state.game_phase, state.lives, state.score)
    def get_info(self, state): return CrossbowInfo(time=state.step_counter)

    def obs_to_flat_array(self, obs: CrossbowObservation) -> jnp.ndarray:
        return jnp.stack([
            obs.cursor_x,
            obs.cursor_y,
            obs.friend_x,
            obs.game_phase,
            obs.lives,
            obs.score
        ], axis=-1).astype(jnp.int32)

    def action_space(self): return spaces.Discrete(18)
    def observation_space(self):
        return spaces.Dict({
            "cursor_x": spaces.Box(0, self.consts.WIDTH, (), jnp.int32),
            "cursor_y": spaces.Box(0, self.consts.HEIGHT, (), jnp.int32),
            "friend_x": spaces.Box(0, self.consts.WIDTH, (), jnp.int32),
            "game_phase": spaces.Discrete(8),
            "lives": spaces.Box(0, self.consts.MAX_LIVES, (), jnp.int32),
            "score": spaces.Box(0, 9999999, (), jnp.int32),
        })
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

        target_h, target_w = 24, 16
        def pad_to_target(mask):
            h, w = mask.shape
            pad_h = max(0, target_h - h)
            pad_w = max(0, target_w - w)
            return jnp.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=255)[:target_h, :target_w]

        self.harmonized_enemy_masks = [
            pad_to_target(self.SHAPE_MASKS["enemy"]),           # 0: GENERIC
            pad_to_target(self.SHAPE_MASKS["scorpion"]),        # 1: SCORPION
            pad_to_target(self.SHAPE_MASKS["ant"]),             # 2: ANT
            pad_to_target(self.SHAPE_MASKS["vulture"]),         # 3: VULTURE
            pad_to_target(self.SHAPE_MASKS["spawn"]),           # 4: SPAWN
            pad_to_target(self.SHAPE_MASKS["lava_rock"]),       # 5: BURNING_LAVA
            pad_to_target(self.SHAPE_MASKS["falling_rock"]),    # 6: FALLING_ROCK
            pad_to_target(self.SHAPE_MASKS["resting_rock"]),    # 7: RESTING_ROCK
            pad_to_target(self.SHAPE_MASKS["monkey"]),          # 8: MONKEY
            pad_to_target(self.SHAPE_MASKS["coconut"]),         # 9: COCONUT
            pad_to_target(self.SHAPE_MASKS["voracious_plant"]), # 10: VORACIOUS_PLANT
            pad_to_target(self.SHAPE_MASKS["archer"]),          # 11: ARCHER
            pad_to_target(self.SHAPE_MASKS["arrow"]),           # 12: ARROW (fallback)
            pad_to_target(self.SHAPE_MASKS["castle_arrow"]),    # 13: CASTLE_ARROW
            pad_to_target(self.SHAPE_MASKS["lava_rock"]),       # 14: CASTLE_FALLING_LAVA
        ]

        self.snake_anim_masks = jnp.stack([
            pad_to_target(self.SHAPE_MASKS["snake"][0]),
            pad_to_target(self.SHAPE_MASKS["snake"][1])
        ])

        self.bat_anim_masks = jnp.stack([pad_to_target(self.SHAPE_MASKS["bat"][j]) for j in range(5)])
        self.stalactite_masks = jnp.stack([pad_to_target(self.SHAPE_MASKS["stalactite"][j]) for j in range(2)])

        ink_color = jnp.min(self.SHAPE_MASKS["arrow"])
        base_right = jnp.full((target_h, target_w), 255, dtype=jnp.uint8)
        base_right = base_right.at[8, 6:11].set(ink_color)
        base_right = base_right.at[8, 11].set(ink_color)
        base_right = base_right.at[7, 10].set(ink_color)
        base_right = base_right.at[9, 10].set(ink_color)
        base_dr = jnp.full((target_h, target_w), 255, dtype=jnp.uint8)
        diag_idx = jnp.arange(6, 11)
        base_dr = base_dr.at[diag_idx, diag_idx].set(ink_color)
        base_dr = base_dr.at[11, 11].set(ink_color)
        base_dr = base_dr.at[10, 11].set(ink_color)
        base_dr = base_dr.at[11, 10].set(ink_color)
        arrow_right = base_right; arrow_down = jnp.rot90(base_right, k=3); arrow_left = jnp.rot90(base_right, k=2); arrow_up = jnp.rot90(base_right, k=1)
        arrow_dr = base_dr; arrow_dl = jnp.rot90(base_dr, k=3); arrow_ul = jnp.rot90(base_dr, k=2); arrow_ur = jnp.rot90(base_dr, k=1)
        def pad_arrow(arr):
            h, w = arr.shape
            pad_h = max(0, target_h - h)
            pad_w = max(0, target_w - w)
            return jnp.pad(arr, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=255)[:target_h, :target_w]
        self.arrow_sprites = jnp.stack([pad_arrow(a) for a in [arrow_right, arrow_dr, arrow_down, arrow_dl, arrow_left, arrow_ul, arrow_up, arrow_ur]])

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
        raster = jax.lax.cond(state.game_phase == GamePhase.CASTLE_HALL, _draw_ropes, lambda r: r, raster)



        # Scatter
        def _draw_px(i, r):
            pixel_mask = jax.lax.switch(state.scatter_px_color_idx[i] - 1, [lambda: self.pixel_masks[c] for c in range(1, 9)])
            return jax.lax.cond(state.scatter_px_active[i], lambda _r: self.jr.render_at(_r, state.scatter_px_x[i].astype(jnp.int32), state.scatter_px_y[i].astype(jnp.int32), pixel_mask), lambda _r: _r, r)
        raster = jax.lax.cond(is_dying, lambda r: jax.lax.fori_loop(0, 100, _draw_px, r), lambda r: r, raster)

        # Enemies

        masks = [self.SHAPE_MASKS["enemy"], self.SHAPE_MASKS["scorpion"], self.SHAPE_MASKS["ant"], self.SHAPE_MASKS["vulture"], self.SHAPE_MASKS["spawn"], self.SHAPE_MASKS["lava_rock"],self.SHAPE_MASKS["falling_rock"],
                  self.SHAPE_MASKS["resting_rock"],self.SHAPE_MASKS["monkey"],self.SHAPE_MASKS["coconut"],
                  self.SHAPE_MASKS["voracious_plant"] 
        ]

        

        def _draw_e(i, r):
            enemy_type = state.enemies_type[i]
            age = state.enemies_age[i]
            ex, ey = state.enemies_x[i].astype(jnp.int32), state.enemies_y[i].astype(jnp.int32)
            is_active = jnp.logical_and(state.enemies_active[i], is_gameplay)
            
            is_arrow = enemy_type == EnemyType.ARROW

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

            snake_frame_idx = (state.step_counter // 8) % 2
            bat_idx = jnp.where(age < 48, jnp.minimum(age // 16, 2), 3 + ((age - 48) // 8) % 2)
            stal_idx = jnp.where(enemy_type == EnemyType.STALACTITE_HANGING, 1, 0)

            snake_mask = jax.lax.switch(snake_frame_idx, [lambda j=j: self.snake_anim_masks[j] for j in range(2)])
            bat_mask = jax.lax.switch(bat_idx, [lambda j=j: self.bat_anim_masks[j] for j in range(5)])
            stal_mask = jax.lax.switch(stal_idx, [lambda j=j: self.stalactite_masks[j] for j in range(2)])
            arrow_mask = jax.lax.switch(final_arrow_idx, [lambda j=j: self.arrow_sprites[j] for j in range(8)])

            # Pick the mask
            m = jax.lax.switch(enemy_type, [
                lambda: self.harmonized_enemy_masks[0],  # 0: GENERIC
                lambda: self.harmonized_enemy_masks[1],  # 1: SCORPION
                lambda: self.harmonized_enemy_masks[2],  # 2: ANT
                lambda: self.harmonized_enemy_masks[3],  # 3: VULTURE
                lambda: self.harmonized_enemy_masks[4],  # 4: SPAWN
                lambda: self.harmonized_enemy_masks[5],  # 5: BURNING_LAVA
                lambda: self.harmonized_enemy_masks[6],  # 6: FALLING_ROCK
                lambda: self.harmonized_enemy_masks[7],  # 7: RESTING_ROCK
                lambda: self.harmonized_enemy_masks[8],  # 8: MONKEY
                lambda: self.harmonized_enemy_masks[9],  # 9: COCONUT
                lambda: self.harmonized_enemy_masks[10], # 10: VORACIOUS_PLANT
                lambda: snake_mask,                      # 11: SNAKE (animated)
                lambda: self.harmonized_enemy_masks[11], # 12: ARCHER
                lambda: arrow_mask,                      # 13: ARROW
                lambda: bat_mask,                        # 14: BAT
                lambda: stal_mask,                       # 15: STALACTITE_FALLING
                lambda: stal_mask,                       # 16: STALACTITE_HANGING
                lambda: self.harmonized_enemy_masks[13], # 17: CASTLE_ARROW
                lambda: self.harmonized_enemy_masks[14], # 18: CASTLE_FALLING_LAVA
            ])

            return jax.lax.cond(is_active,
                                lambda _r: self.jr.render_at(_r, ex, ey, m),
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