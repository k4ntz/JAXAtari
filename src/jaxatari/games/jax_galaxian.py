import os
from typing import NamedTuple, Tuple
import jax.numpy as jnp
import chex
import pygame
from functools import partial
from jax import lax
import jax.lax

from gymnax.environments import spaces

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import AtraJaxisRenderer
from jaxatari.rendering import atraJaxis as aj


"""
README README README README README README README 
Aaron Reinhardt
Aaron Weis
Leon Denis Kristof

Starting our game with direct call provides more features than the play.py call.
Game over, death and score display do not work in play.py call.

Float casting is a temporary solution and will be fixed.

direct call: python -m jaxatari.games.jax_galaxian
play.py call: python scripts/play.py --game src/jaxatari/games/jax_galaxian.py --record my_record_file.npz
"""



# -------- Game constants --------
SHOOTING_COOLDOWN = 0 #TODO set back to 80 before merge
ENEMY_MOVE_SPEED = 0.5
BULLET_MOVE_SPEED = 5
GRID_ROWS = 6
GRID_COLS = 5 #TODO needs to be 7, but >5 is clamping right
NATIVE_GAME_WIDTH = 160
NATIVE_GAME_HEIGHT = 210
PYGAME_SCALE_FACTOR = 3
PYGAME_WINDOW_WIDTH = NATIVE_GAME_WIDTH * PYGAME_SCALE_FACTOR
PYGAME_WINDOW_HEIGHT = NATIVE_GAME_HEIGHT * PYGAME_SCALE_FACTOR
ENEMY_SPACING_X = 20
ENEMY_SPACING_Y = 11
ENEMY_GRID_Y = 80
START_X = NATIVE_GAME_WIDTH // 4
START_Y = NATIVE_GAME_HEIGHT
ENEMY_ATTACK_SPEED = 2
ENEMY_ATTACK_TURN_TIME = 30
ENEMY_ATTACK_BULLET_SPEED = 5
ENEMY_ATTACK_BULLET_DELAY = 75
ENEMY_ATTACK_MAX_BULLETS = 2
LIVES = 3
PLAYER_DEATH_DELAY = 50
ENEMY_LEFT_BOUND = 17
ENEMY_RIGHT_BOUND = NATIVE_GAME_WIDTH - 25
DIVE_KILL_Y = 175
DIVE_SPEED = 0.5
MAX_DIVERS = 5
BASE_DIVE_PROBABILITY = 5

class GalaxianState(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    player_shooting_cooldown: chex.Array
    bullet_x: chex.Array
    bullet_y: chex.Array
    enemy_grid_x: chex.Array
    enemy_grid_y: chex.Array
    enemy_grid_alive: chex.Array        # 0: dead, 1: alive, 2: attacking
    enemy_grid_direction: chex.Array
    enemy_attack_states: chex.Array      # 0: noop, 1: attack, 2: respawn
    enemy_attack_pos: chex.Array
    enemy_attack_x: chex.Array
    enemy_attack_y: chex.Array
    enemy_attack_direction: chex.Array
    enemy_attack_target_x: chex.Array# -1: left, 1: right
    enemy_attack_target_y: chex.Array
    enemy_attack_turning: chex.Array    # -1: turning left, 1: turning right, 0: no turning
    enemy_attack_turn_timer: chex.Array
    enemy_attack_respawn_timer: chex.Array
    enemy_attack_bullet_x: chex.Array
    enemy_attack_bullet_y: chex.Array
    enemy_attack_bullet_timer: chex.Array
    lives: chex.Array
    player_alive: chex.Array
    player_respawn_timer: chex.Array
    score: chex.Array
    step_counter: chex.Array
    dive_probability: chex.Array
    enemy_bullet_max_cooldown: chex.Array


@jax.jit
def update_player_position(state: GalaxianState, action) -> GalaxianState:

    press_right = jnp.any(
        jnp.array([action == Action.RIGHT, action == Action.RIGHTFIRE])
    )

    press_left = jnp.any(
        jnp.array([action == Action.LEFT, action == Action.LEFTFIRE])
    )

    # rohe neue X-Position
    new_x = state.player_x + (press_right * 5) - (press_left * 5)

    # clamp zwischen 0 und (SCREEN_WIDTH − PLAYER_WIDTH)
    new_x = jnp.clip(new_x, 17, NATIVE_GAME_WIDTH - 25)
    return state._replace(player_x=new_x)

@jax.jit
def update_enemy_positions(state: GalaxianState) -> GalaxianState:

    new_x = state.enemy_grid_x + ENEMY_MOVE_SPEED * state.enemy_grid_direction

    # clamp auf [LEFT, RIGHT]
    new_x = jnp.clip(new_x, ENEMY_LEFT_BOUND, ENEMY_RIGHT_BOUND)

    # Rand-Bounce wie gehabt
    hit_left = jnp.any(new_x <= ENEMY_LEFT_BOUND)
    hit_right = jnp.any(new_x >= ENEMY_RIGHT_BOUND)
    new_dir = jnp.where(hit_left, 1,
                        jnp.where(hit_right, -1, state.enemy_grid_direction))

    return state._replace(
        enemy_grid_x=new_x,
        enemy_grid_direction=new_dir
    )



@jax.jit
def update_enemy_attack(state: GalaxianState) -> GalaxianState:
    # random ob gedived wird
    # falls ja, gehe erst zu initialise_new_dive und im anschluss zu continue_active_dives(state)
    # falls nein, gehe direkt zu continue_active_dives(state)
    @jax.jit
    def test_for_new_dive(state):
        key = jax.random.PRNGKey(state.step_counter)
        do_new_dive = jax.random.uniform(key, shape=(), minval=0, maxval=100) < state.dive_probability / 10
        return jax.lax.cond(do_new_dive, lambda state: initialise_new_dive(state), lambda state: continue_active_dives(state), state)

    @jax.jit
    def initialise_new_dive(state):
        key = jax.random.PRNGKey(state.step_counter + 101)  # currently deterministic
        key_choice, key_dir_if_zero = jax.random.split(key)

        # finde freien slot im diver array und nimm einfach den ersten (es muss einen geben, weil wir vorher geprüft haben, dass noch Platz ist)
        available_slots_indices = jnp.where(jnp.atleast_1d(state.enemy_attack_states == 0), size=MAX_DIVERS, fill_value=-1)[0]
        diver_idx = available_slots_indices[0]


        # finde zufälligen Angreifer
        alive_grid_mask = (state.enemy_grid_alive == 1)
        num_alive_in_grid = jnp.sum(alive_grid_mask)

        # hole deren Positionen im Array
        grid_rows_indices, grid_cols_indices = jnp.where(alive_grid_mask, size=GRID_ROWS * GRID_COLS, fill_value=-1)

        # Filter fill values
        valid_rows = jnp.where(grid_rows_indices != -1, grid_rows_indices, 0)
        valid_cols = jnp.where(grid_rows_indices != -1, grid_cols_indices, 0)

        # random selection
        random_choice_idx = jax.random.randint(key_choice, shape=(), minval=0, maxval=num_alive_in_grid)
        chosen_enemy_row = valid_rows[random_choice_idx]
        chosen_enemy_col = valid_cols[random_choice_idx]

        # e.g. enemy_attack_states=[1. 0. 0. 1. 1.] -> enemy_attack_states=[1. 1. 0. 1. 1.]
        new_attack_states = state.enemy_attack_states.at[diver_idx].set(1)  # 1: actively attacking

        # alte Position für respawn speichern
        new_attack_pos = state.enemy_attack_pos.at[diver_idx].set(
            jnp.array([chosen_enemy_row, chosen_enemy_col], dtype=jnp.int32)
        )
        new_grid_alive = state.enemy_grid_alive.at[chosen_enemy_row, chosen_enemy_col].set(2)

        # init diver
        start_dive_x = state.enemy_grid_x[chosen_enemy_row, chosen_enemy_col]
        start_dive_y = state.enemy_grid_y[chosen_enemy_row, chosen_enemy_col]
        new_attack_x = state.enemy_attack_x.at[diver_idx].set(start_dive_x)
        new_attack_y = state.enemy_attack_y.at[diver_idx].set(start_dive_y)

        # timer
        new_attack_turning = state.enemy_attack_turning.at[diver_idx].set(0)
        new_attack_turn_timer = state.enemy_attack_turn_timer.at[diver_idx].set(ENEMY_ATTACK_TURN_TIME)
        new_attack_respawn_timer = state.enemy_attack_respawn_timer.at[diver_idx].set(0)

        # Reset bullet
        new_attack_bullet_x = state.enemy_attack_bullet_x.at[diver_idx].set(-1.0)
        new_attack_bullet_y = state.enemy_attack_bullet_y.at[diver_idx].set(-1.0)
        new_attack_bullet_timer = state.enemy_attack_bullet_timer.at[diver_idx].set(ENEMY_ATTACK_BULLET_DELAY)

        return state._replace(
            enemy_attack_states=new_attack_states,
            enemy_attack_pos=new_attack_pos,
            enemy_attack_x=new_attack_x,
            enemy_attack_y=new_attack_y,
            enemy_grid_alive=new_grid_alive,
            enemy_attack_turning=new_attack_turning,
            enemy_attack_turn_timer=new_attack_turn_timer,
            enemy_attack_respawn_timer=new_attack_respawn_timer,
            enemy_attack_bullet_x=new_attack_bullet_x,
            enemy_attack_bullet_y=new_attack_bullet_y,
            enemy_attack_bullet_timer=new_attack_bullet_timer,
        )

    @jax.jit
    def continue_active_dives(state: GalaxianState) -> GalaxianState:
        # e.g. is_attacking: [ True  False  True  False  True] (secound killed by player)
        is_attacking = (state.enemy_attack_states == 1)
        target_x = state.player_x
        target_y = state.player_y

        curr_x = state.enemy_attack_x
        curr_y = state.enemy_attack_y

        dx = target_x - curr_x
        dy = target_y - curr_y
        norm = jnp.sqrt(dx ** 2 + dy ** 2) + 1e-6  #TODO sinnvoller dive

        step_x = curr_x + (dx / norm) * DIVE_SPEED
        step_y = curr_y + (dy / norm) * DIVE_SPEED

        new_attack_x = jnp.where(is_attacking, step_x, curr_x)
        new_attack_y = jnp.where(is_attacking, step_y, curr_y)

        return state._replace(
            enemy_attack_x=new_attack_x,
            enemy_attack_y=new_attack_y
        )

    def respawn_finished_dives(state: GalaxianState) -> GalaxianState:
        return 420;


    #new_state = respawn_finished_dives(state)
    free_slots = jnp.sum(state.enemy_attack_states == 0)
    return jax.lax.cond(
        free_slots > 0,
        lambda state: test_for_new_dive(state),
        lambda state: continue_active_dives(state),
        state
    )


@jax.jit
def update_enemy_bullets(state: 'GalaxianState') -> 'GalaxianState':
    bullet_x = state.enemy_attack_bullet_x
    bullet_y = state.enemy_attack_bullet_y
    bullet_timers = state.enemy_attack_bullet_timer

    # continue active bullets
    is_active_bullet_mask = bullet_y >= 0.0
    moved_bullet_y = jnp.where(is_active_bullet_mask, bullet_y + ENEMY_ATTACK_BULLET_SPEED, bullet_y)
    moved_bullet_x = bullet_x # X position usually doesn't change

    # remove bullets at the bottom
    off_screen_mask = moved_bullet_y > NATIVE_GAME_HEIGHT
    current_bullet_x_after_move = jnp.where(off_screen_mask, -1.0, moved_bullet_x)
    current_bullet_y_after_move = jnp.where(off_screen_mask, -1.0, moved_bullet_y)

    # lower timers
    is_diver_attacking_mask = (state.enemy_attack_states == 1)
    decremented_bullet_timers = jnp.where(is_diver_attacking_mask, bullet_timers - 1.0, bullet_timers)
    decremented_bullet_timers = jnp.maximum(decremented_bullet_timers, 0.0)


    # always shoots when ready
    ready_to_shoot_timer_wise = jnp.logical_and(is_diver_attacking_mask, decremented_bullet_timers <= 0.0)
    slot_is_free_mask = current_bullet_y_after_move < 0.0
    can_spawn_new_bullet_mask = jnp.logical_and(ready_to_shoot_timer_wise, slot_is_free_mask)

    final_bullet_x = jnp.where(can_spawn_new_bullet_mask, state.enemy_attack_x, current_bullet_x_after_move)
    final_bullet_y = jnp.where(can_spawn_new_bullet_mask, state.enemy_attack_y, current_bullet_y_after_move)

    timer_dtype = state.enemy_attack_bullet_timer.dtype
    final_bullet_timers = jnp.where(
        can_spawn_new_bullet_mask,
        jnp.array(state.enemy_bullet_max_cooldown, dtype=timer_dtype),
        decremented_bullet_timers
    )


    return state._replace(
        enemy_attack_bullet_x=final_bullet_x.astype(state.enemy_attack_bullet_x.dtype),
        enemy_attack_bullet_y=final_bullet_y.astype(state.enemy_attack_bullet_y.dtype),
        enemy_attack_bullet_timer=final_bullet_timers.astype(timer_dtype)
    )


def get_action_from_keyboard():
    keys = pygame.key.get_pressed()
    left = keys[pygame.K_a] or keys[pygame.K_LEFT]
    right = keys[pygame.K_d] or keys[pygame.K_RIGHT]
    shooting = keys[pygame.K_SPACE]

    left_only = left and not right
    right_only = right and not left

    if shooting:
        if left_only:
            return Action.LEFTFIRE
        elif right_only:
            return Action.RIGHTFIRE
        else:
            return Action.FIRE
    else:
        if left_only:
            return Action.LEFT
        elif right_only:
            return Action.RIGHT
        else:
            return Action.NOOP



@jax.jit
def handle_shooting(state: GalaxianState, action) -> GalaxianState:
    def shoot(_):
        return state._replace(
            bullet_x=jnp.array(state.player_x, dtype=state.bullet_x.dtype),
            bullet_y=jnp.array(state.player_y, dtype=state.bullet_y.dtype),
            player_shooting_cooldown=jnp.array(SHOOTING_COOLDOWN)
        )

    def decrease_cooldown(_):
        return state._replace(
            player_shooting_cooldown=state.player_shooting_cooldown - 1
        )

    def idle(_):
        return state


    shooting = jnp.any(
        jnp.array(
            [
                action == Action.FIRE,
                action == Action.RIGHTFIRE,
                action == Action.LEFTFIRE,
            ]
        )
    )

    can_shoot = jnp.logical_and(shooting, state.player_shooting_cooldown == 0)
    on_cooldown = jnp.logical_not(can_shoot) & (state.player_shooting_cooldown > 0)

    return lax.cond(
        can_shoot,
        shoot,
        lambda _: lax.cond(on_cooldown, decrease_cooldown, idle, operand=None),
        operand=None
    )


@jax.jit
def update_bullets(state: GalaxianState) -> GalaxianState:
    new_bullets_y = jnp.array(state.bullet_y - BULLET_MOVE_SPEED)
    return state._replace(bullet_y=new_bullets_y)


@jax.jit
def remove_bullets(state: GalaxianState) -> GalaxianState:
    # Wenn die Spieler-Kugel den Bildschirm (y<0) verlässt, reset Kugel + Cooldown
    def bullet_reset(state: GalaxianState) -> GalaxianState:
        return state._replace(
            bullet_x                = jnp.array(-1.0, dtype=state.bullet_x.dtype),
            bullet_y                = jnp.array(-1.0, dtype=state.bullet_y.dtype),
        )
    def keep(state: GalaxianState) -> GalaxianState:
        return state

    return lax.cond(state.bullet_y < 0,
                    bullet_reset,
                    keep,
                    state)


@jax.jit
def bullet_collision(state: GalaxianState) -> GalaxianState:
    x_diff = jnp.abs(state.bullet_x - state.enemy_grid_x)
    y_diff = jnp.abs(state.bullet_y - state.enemy_grid_y)

    mask = (x_diff <= 10) & (y_diff <= 10) & (state.enemy_grid_alive == 1)
    hit  = jnp.any(mask)

    def process_hit(s_and_idx):
        s, (rows, cols) = s_and_idx
        # setze alle getroffenen auf dead
        new_alive = state.enemy_grid_alive.at[rows, cols].set(0)
        return state._replace(
            enemy_grid_alive         = new_alive,
            bullet_x                 = jnp.array(-1.0, dtype=s.bullet_x.dtype),
            bullet_y                 = jnp.array(-1.0, dtype=s.bullet_y.dtype),
            score                    = s.score + 30,
        )

    def no_hit(state_and_idx):
        state, _ = state_and_idx
        return state

    max_size = state.enemy_grid_alive.size
    collision_indices = jnp.where(mask, size=max_size, fill_value=-1)
    hit_rows = collision_indices[0]
    hit_cols = collision_indices[1]

    return lax.cond(hit,
                    process_hit,
                    no_hit,
                    (state, (hit_rows, hit_cols)))

@jax.jit
def bullet_collision_attack(state: GalaxianState) -> GalaxianState:
    # Abstände
    x_diff = jnp.abs(state.bullet_x - state.enemy_attack_x)
    y_diff = jnp.abs(state.bullet_y - state.enemy_attack_y)
    # Kollisionsmaske: innerhalb 10px und aktuell angreifend (state 1)
    mask = (x_diff <= 10) & (y_diff <= 10) & (state.enemy_attack_states == 1)
    hit = jnp.any(mask)

    def process_hit(state: GalaxianState) -> GalaxianState:
        hit_indices = jnp.where(mask, size=MAX_DIVERS, fill_value=-1)[0]
        hit_idx = hit_indices[0]
        pos = state.enemy_attack_pos[hit_idx]
        new_grid = state.enemy_grid_alive.at[tuple(pos)].set(0)

        new_attack_states = state.enemy_attack_states.at[hit_idx].set(0)

        return state._replace(
            enemy_grid_alive=new_grid,
            bullet_x=jnp.array(-1.0, dtype=state.bullet_x.dtype),
            bullet_y=jnp.array(-1.0, dtype=state.bullet_y.dtype),
            enemy_attack_states=new_attack_states,
            score=state.score + jnp.array(50, dtype=state.score.dtype)
        )

    def no_hit(state: GalaxianState) -> GalaxianState:
        return state

    return lax.cond(hit, process_hit, no_hit, state)

@jax.jit
def check_player_death_by_enemy(state: GalaxianState) -> GalaxianState:
    x_diff = jnp.abs(state.player_x - state.enemy_attack_x)
    y_diff = jnp.abs(state.player_y - state.enemy_attack_y)
    is_active = (state.enemy_attack_states == 1)
    grid_alive = state.enemy_grid_alive[tuple(state.enemy_attack_pos.T)] != 0

    collision = (x_diff <= 10) & (y_diff <= 10) & is_active & grid_alive
    hit = jnp.any(collision)

    def process_hit(current_state):
        hit_indices = jnp.where(collision, size=MAX_DIVERS, fill_value=-1)[0]
        hit_idx = hit_indices[0]
        pos = current_state.enemy_attack_pos[hit_idx]
        new_lives = current_state.lives - 1
        new_enemy_grid_alive = current_state.enemy_grid_alive.at[tuple(pos)].set(0)
        new_attack_states = current_state.enemy_attack_states.at[hit_idx].set(0)
        return current_state._replace(
            lives=new_lives,
            enemy_grid_alive=new_enemy_grid_alive,
            enemy_attack_states=new_attack_states
        )

    return lax.cond(hit, process_hit, lambda s: s, state)

@jax.jit
def check_player_death_by_bullet(state: GalaxianState) -> GalaxianState:
    x_diff = jnp.abs(state.player_x - state.enemy_attack_bullet_x)
    y_diff = jnp.abs(state.player_y - state.enemy_attack_bullet_y)

    collision_mask = (x_diff <= 10) & (y_diff <= 10) & (state.enemy_attack_bullet_y >= 0)
    hit = jnp.any(collision_mask)

    def process_hit(current_state):
        # reset bullets
        hit_indices = jnp.where(collision_mask, size=MAX_DIVERS * 5, fill_value=-1)[0]  # size müsste man irgendwann vllt anpassen
        new_bullet_x = current_state.enemy_attack_bullet_x.at[hit_indices].set(-1.0)
        new_bullet_y = current_state.enemy_attack_bullet_y.at[hit_indices].set(-1.0)

        # subtract 1 life
        new_lives = current_state.lives - jnp.sum(collision_mask)

        return current_state._replace(
            lives=new_lives,
            enemy_attack_bullet_x=new_bullet_x,
            enemy_attack_bullet_y=new_bullet_y
        )

    return lax.cond(hit, process_hit, lambda s: s, state)

@jax.jit
def enter_new_wave(state: GalaxianState) -> GalaxianState:
    new_grid = jnp.ones(state.enemy_grid_alive.shape)
    new_prop = jnp.array(state.dive_probability * 1.1, dtype=state.dive_probability.dtype)
    new_attack_bullet_cd = jnp.array(state.enemy_bullet_max_cooldown * 0.9, dtype=state.enemy_bullet_max_cooldown.dtype)
    return state._replace(enemy_grid_alive=new_grid,
                          dive_probability=new_prop)


class GalaxianObservation(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    bullet_x: chex.Array
    bullet_y: chex.Array
    enemy_grid_x: chex.Array
    enemy_grid_y: chex.Array
    enemy_grid_alive: chex.Array  # 0: dead, 1: alive, 2: attacking
    enemy_attack_pos: chex.Array
    enemy_attack_x: chex.Array
    enemy_attack_y: chex.Array

class GalaxianInfo(NamedTuple):
    time: jnp.ndarray
    all_rewards: chex.Array

class JaxGalaxian(JaxEnvironment[GalaxianState, GalaxianObservation, GalaxianInfo]):
    def __init__(self, frameskip: int = 0, reward_funcs: list[callable]=None):
        super().__init__()
        self.frameskip = frameskip + 1  #den stuff kp hab copy paste aus pong
        self.frame_stack_size = 4
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.action_set = {
            Action.NOOP,
            Action.FIRE,
            Action.RIGHT,
            Action.LEFT,
            Action.RIGHTFIRE,
            Action.LEFTFIRE
        }
        self.obs_size = 3*2+1
        #Größe der Beobachtungsdaten
        #3*4 = 3 entitys mit je 2 werten: enemy, player, bullet -> x,y
        # +1 ein extra wert für den score
        # sind erstmal nur temporary values

    def reset(self, key = None) -> Tuple[GalaxianObservation, GalaxianState]:
        grid_rows = GRID_ROWS
        grid_cols = GRID_COLS
        enemy_spacing_x = ENEMY_SPACING_X
        start_x = START_X

        x_positions = jnp.arange(grid_cols) * enemy_spacing_x + start_x  # arange schreibt so 0 1 2 3....
        enemy_grid = jnp.tile(x_positions, (grid_rows, 1))  # kopiert die zeile untereinander

        row_indices = jnp.arange(grid_rows).reshape(-1, 1)  # erzeugt 1. enemy jeder zeile
        enemy_y_rows = ENEMY_GRID_Y - row_indices * ENEMY_SPACING_Y  # jeweils y pos
        enemy_grid_y = jnp.broadcast_to(enemy_y_rows, (
            grid_rows, grid_cols))  # kopiert die werte für 1. enemy auf die rechts in der zeile

        enemy_alive = jnp.ones((grid_rows, grid_cols))  # alles auf 1



        state = GalaxianState(player_x=jnp.array(NATIVE_GAME_WIDTH / 2.0),
                              player_y=jnp.array(NATIVE_GAME_HEIGHT - 40.0),
                              player_shooting_cooldown=jnp.array(0),
                              bullet_x=jnp.array(-1.0,dtype=jnp.float32),
                              bullet_y=jnp.array(-1.0,dtype=jnp.float32),
                              enemy_grid_x=enemy_grid.astype(jnp.float32),
                              enemy_grid_y=enemy_grid_y.astype(jnp.float32),
                              enemy_grid_alive=enemy_alive,
                              enemy_grid_direction=jnp.array(1),
                              enemy_attack_states=jnp.zeros(MAX_DIVERS),
                              enemy_attack_pos=jnp.full((MAX_DIVERS, 2), -1, dtype=jnp.int32),
                              enemy_attack_direction=jnp.zeros(MAX_DIVERS),
                              enemy_attack_turning=jnp.zeros(MAX_DIVERS),
                              enemy_attack_turn_timer=jnp.zeros(MAX_DIVERS),
                              enemy_attack_x=jnp.zeros(MAX_DIVERS),
                              enemy_attack_y=jnp.zeros(MAX_DIVERS),
                              enemy_attack_respawn_timer=jnp.zeros(MAX_DIVERS),
                              enemy_attack_bullet_x=jnp.zeros(MAX_DIVERS),
                              enemy_attack_bullet_y=jnp.zeros(MAX_DIVERS),
                              enemy_attack_bullet_timer=jnp.zeros(MAX_DIVERS),
                              lives=jnp.array(3),
                              player_alive=jnp.array(True),
                              player_respawn_timer=jnp.array(PLAYER_DEATH_DELAY),
                              score=jnp.array(0, dtype=jnp.int32),
                              enemy_attack_target_x=jnp.zeros(MAX_DIVERS),
                              enemy_attack_target_y=jnp.zeros(MAX_DIVERS),
                              step_counter=jnp.array(0),
                              dive_probability=jnp.array(BASE_DIVE_PROBABILITY),
                              enemy_bullet_max_cooldown=ENEMY_ATTACK_BULLET_DELAY
                             )

        initial_obs = self._get_observation(state)
        return initial_obs, state

        #alles aus galaxianState was man aus Spielersicht wahrnimmt
    def _get_observation(self, state: GalaxianState) -> GalaxianObservation:
        return GalaxianObservation(
            player_x=state.player_x,
            player_y=state.player_y,
            bullet_x=state.bullet_x,
            bullet_y=state.bullet_y,
            enemy_grid_x=state.enemy_grid_x,
            enemy_grid_y=state.enemy_grid_y,
            enemy_grid_alive=state.enemy_grid_alive,
            enemy_attack_pos=state.enemy_attack_pos,
            enemy_attack_x=state.enemy_attack_x,
            enemy_attack_y=state.enemy_attack_y,
        )

    @partial(jax.jit, static_argnums=(0,))
    def get_action_space(self):
        return jnp.array([Action.NOOP, Action.LEFT, Action.RIGHT, Action.FIRE, Action.LEFTFIRE, Action.RIGHTFIRE])


    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))

    def observation_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=None,
            dtype=jnp.uint8,
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(
            self, state: GalaxianState, action: chex.Array
    ) -> Tuple[GalaxianObservation, GalaxianState, float, bool, GalaxianInfo]:
        #TODO: refactor like the other games
        # create new state instead of replacing old state step by step
        new_state = update_player_position(state, action)
        new_state = handle_shooting(new_state, action)
        new_state = update_enemy_positions(new_state)
        new_state = update_bullets(new_state)
        new_state = remove_bullets(new_state)
        new_state = bullet_collision(new_state)
        new_state = bullet_collision_attack(new_state)
        new_state = update_enemy_attack(new_state)
        new_state = update_enemy_bullets(new_state)
        new_state = check_player_death_by_enemy(new_state)
        new_state = check_player_death_by_bullet(new_state)
        new_state = new_state._replace(step_counter=new_state.step_counter + 1)

        new_state = jax.lax.cond(jnp.logical_not(jnp.any(state.enemy_grid_alive == 1)), lambda new_state: enter_new_wave(new_state), lambda s: s, new_state)

        done = self._get_done(new_state)
        env_reward = self._get_env_reward(state, new_state)
        all_rewards = self._get_all_reward(state, new_state)
        info = self._get_info(new_state, all_rewards)

        observation = self._get_observation(new_state)

        return observation, new_state, env_reward, done, info


    @partial(jax.jit, static_argnums=(0,))
    def _get_env_reward(self, previous_state: GalaxianState, state: GalaxianState):
        return state.score - previous_state.score


    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(self, previous_state: GalaxianState, state: GalaxianState) -> chex.Array:
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array(
            [reward_func(previous_state, state) for reward_func in self.reward_funcs]
        )
        return rewards

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: GalaxianState) -> bool:
        # Game Over, wenn alle Leben weg sind
        no_lives = state.lives <= 0
        # You Win, wenn **keine** lebenden Enemies (==1) mehr da sind
        no_enemies = jnp.all(state.enemy_grid_alive == 0)
        return jnp.logical_or(no_lives, no_enemies)

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: GalaxianState, all_rewards: chex.Array) -> GalaxianInfo:
        return GalaxianInfo(time=state.step_counter, all_rewards=all_rewards)


def load_sprites():
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

    # load sprites
    bg = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/galaxian/background.npy"),transpose=True)
    player = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/galaxian/player.npy"),transpose=True)
    bullet = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/galaxian/bullet.npy"),transpose=True)
    enemy_gray = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/galaxian/gray_enemy_1.npy"),transpose=True)
    life = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/galaxian/life.npy"),transpose=True)
    enemy_bullet = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/galaxian/enemy_bullet.npy"),transpose=True)
    enemy_red = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/galaxian/red_orange_enemy_1.npy"),transpose=True)
    enemy_blue = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/galaxian/purple_blue_enemy_1.npy"),transpose=True)
    enemy_white = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/galaxian/white_enemy_1.npy"),transpose=True)
    SPRITE_BG = jnp.expand_dims(bg, axis= 0)
    SPRITE_PLAYER = jnp.expand_dims(player, axis = 0)
    SPRITE_BULLET = jnp.expand_dims(bullet, axis = 0)
    SPRITE_ENEMY_GRAY = jnp.expand_dims(enemy_gray, axis=0)
    SPRITE_ENEMY_RED = jnp.expand_dims(enemy_red, axis=0)
    SPRITE_ENEMY_BLUE = jnp.expand_dims(enemy_blue, axis=0)
    SPRITE_ENEMY_WHITE = jnp.expand_dims(enemy_white, axis=0)
    SPRITE_LIFE = jnp.expand_dims(life, axis=0)
    SPRITE_ENEMY_BULLET = jnp.expand_dims(enemy_bullet, axis=0)
    return(
        SPRITE_BG,
        SPRITE_PLAYER,
        SPRITE_BULLET,
        SPRITE_ENEMY_GRAY,
        SPRITE_ENEMY_RED,
        SPRITE_ENEMY_BLUE,
        SPRITE_ENEMY_WHITE,
        SPRITE_LIFE,
        SPRITE_ENEMY_BULLET
    )

class GalaxianRenderer(AtraJaxisRenderer):
    def __init__(self):
        (
            self.SPRITE_BG,
            self.SPRITE_PLAYER,
            self.SPRITE_BULLET,
            self.SPRITE_ENEMY_GRAY,
            self.SPRITE_ENEMY_RED,
            self.SPRITE_ENEMY_BLUE,
            self.SPRITE_ENEMY_WHITE,
            self.SPRITE_LIFE,
            self.SPRITE_ENEMY_BULLET,
        ) = load_sprites()

        # Sprite-Dimensionen für Life-Icons
        life_frame = jnp.squeeze(self.SPRITE_LIFE, axis=0)  # (h, w, 4)
        self.life_h, self.life_w, _ = life_frame.shape
        self.life_spacing = 5

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: GalaxianState):
        # Hintergrund
        raster = jnp.zeros((NATIVE_GAME_WIDTH, NATIVE_GAME_HEIGHT, 3), dtype=jnp.uint8)
        bg_frame = aj.get_sprite_frame(self.SPRITE_BG, 0)
        raster = aj.render_at(raster, 0, 0, bg_frame)

        # Spieler
        player_frame = aj.get_sprite_frame(self.SPRITE_PLAYER, 0)
        px = jnp.round(state.player_x).astype(jnp.int32)
        py = jnp.round(state.player_y).astype(jnp.int32)
        raster = aj.render_at(raster, px, py, player_frame)

        # Spieler-Kugel
        def draw_bullet(r):
            b = aj.get_sprite_frame(self.SPRITE_BULLET, 0)
            bx = jnp.round(state.bullet_x).astype(jnp.int32)
            by = jnp.round(state.bullet_y).astype(jnp.int32)
            return aj.render_at(r, bx, by, b)
        raster = lax.cond(state.bullet_y > 0, draw_bullet, lambda r: r, raster)

        enemy_bullet_sprite = aj.get_sprite_frame(self.SPRITE_ENEMY_BULLET, 0)

        def _draw_single_enemy_bullet(i, r_acc):
            return lax.cond(
                state.enemy_attack_bullet_y[i] >= 0,  # Active bullet
                lambda r: aj.render_at(r,
                                       jnp.round(state.enemy_attack_bullet_x[i]).astype(jnp.int32),
                                       jnp.round(state.enemy_attack_bullet_y[i]).astype(jnp.int32),
                                       enemy_bullet_sprite),
                lambda r: r,
                r_acc
            )

        raster = lax.fori_loop(0, MAX_DIVERS, _draw_single_enemy_bullet, raster)


        def draw_attackers(r):
            e = aj.get_sprite_frame(self.SPRITE_ENEMY_GRAY, 0)

            def draw_single_attacker(r, i):
                cond = state.enemy_attack_states[i] == 1

                def true_fn(r):
                    ex = jnp.round(state.enemy_attack_x[i]).astype(jnp.int32)
                    ey = jnp.round(state.enemy_attack_y[i]).astype(jnp.int32)
                    return aj.render_at(r, ex, ey, e)

                def false_fn(r):
                    return r

                return lax.cond(cond, true_fn, false_fn, r)

            for i in range(MAX_DIVERS):
                r = draw_single_attacker(r, i)

            return r
        raster = lax.cond(jnp.any(state.enemy_attack_states != 0), draw_attackers, lambda r: r, raster)


       # Feindgitter
        def row_body(i, r_acc):
            def col_body(j, r_inner):
                conditions = [
                    i == 5,
                    i == 4,
                    i == 3
                ]
                choices = [
                    aj.get_sprite_frame(self.SPRITE_ENEMY_WHITE, 0),
                    aj.get_sprite_frame(self.SPRITE_ENEMY_RED, 0),
                    aj.get_sprite_frame(self.SPRITE_ENEMY_GRAY, 0) #TODO make purple after sprite is fixed
                ]
                default_choice = aj.get_sprite_frame(self.SPRITE_ENEMY_GRAY, 0)
                enemy_sprite = jnp.select(conditions, choices, default_choice)
                cond = state.enemy_grid_alive[i, j] == 1
                def draw(r0):
                    x = jnp.round(state.enemy_grid_x[i, j]).astype(jnp.int32)
                    y = jnp.round(state.enemy_grid_y[i, j]).astype(jnp.int32)
                    return aj.render_at(r0, x, y, enemy_sprite)
                return lax.cond(cond, draw, lambda r0: r0, r_inner)
            return lax.fori_loop(0, GRID_COLS, col_body, r_acc)
        raster = lax.fori_loop(0, GRID_ROWS, row_body, raster)

        # Lebens-Icons unten rechts
        life_sprite = aj.get_sprite_frame(self.SPRITE_LIFE, 0)
        def life_loop_body(i, r_acc):
            # nur zeichnen, wenn Leben vorhanden
            def draw(r0):
                x0 = jnp.int32(
                    NATIVE_GAME_WIDTH - (i + 1) * (self.life_w + self.life_spacing)
                )
                y0 = jnp.int32(
                    NATIVE_GAME_HEIGHT - self.life_h - self.life_spacing
                )
                return aj.render_at(r0, x0, y0, life_sprite)
            return lax.cond(i < state.lives, draw, lambda r0: r0, r_acc)
        raster = lax.fori_loop(0, LIVES, life_loop_body, raster)

        return raster

# run with: python -m jaxatari.games.jax_galaxian
# run with: python scripts/play.py --game src/jaxatari/games/jax_galaxian.py --record my_record_file.npz
if __name__ == "__main__":
    pygame.init()
    font = pygame.font.Font(None, 24)

    game = JaxGalaxian(frameskip=1)
    jitted_step = jax.jit(game.step)
    jitted_reset = jax.jit(game.reset)
    initial_observation, state  = jitted_reset()

    screen = pygame.display.set_mode((PYGAME_WINDOW_WIDTH, PYGAME_WINDOW_HEIGHT))
    pygame.display.set_caption("Galaxian")
    clock = pygame.time.Clock()

    renderer = GalaxianRenderer()

    initial_observation, state = jax.jit(game.reset)()

    jitted_step = game.step

    running = True
    while running:
        action_int = Action.NOOP
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.fill((0, 0, 0))
        pygame.display.flip()
        action = get_action_from_keyboard()
        observation, state, reward, done, info = jitted_step(state, action)

        render_output = renderer.render(state)
        aj.update_pygame(screen, render_output, PYGAME_SCALE_FACTOR, NATIVE_GAME_WIDTH,
                         NATIVE_GAME_HEIGHT)
        score_surf = font.render(f"Score: {int(state.score)}", True, (255, 255, 255))
        screen.blit(score_surf, (10, 10))

        font = pygame.font.Font(None, 24)

        # Score
        score_surf = font.render(f"Score: {int(state.score)}", True, (255, 255, 255))
        screen.blit(score_surf, (10, 10))

        pygame.display.flip()
        if done:
            # Win vs. Game Over
            if int(state.lives) > 0:
                text = "You Win!"
            else:
                text = "Game Over!"
            msg = font.render(text, True, (255, 255, 255))
            x = (PYGAME_WINDOW_WIDTH - msg.get_width()) // 2
            y = (PYGAME_WINDOW_HEIGHT - msg.get_height()) // 2
            screen.blit(msg, (x, y))
            pygame.display.flip()
            pygame.time.wait(2000)
            break
        clock.tick(30)

    pygame.quit()




