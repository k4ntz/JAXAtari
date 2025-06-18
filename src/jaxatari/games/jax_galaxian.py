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
from jaxatari.rendering.atraJaxis import render_at, get_sprite_frame

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
ENEMY_ATTACK_TURN_TIME = 32
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
BASE_DIVE_PROBABILITY = 30 #TODO back to 5 before merge
MAX_SHOTS_PER_VOLLEY = 4
VOLLEY_SHOT_DELAY = 10

PLAYER_BULLET_Y_OFFSET = 3
PLAYER_BULLET_X_OFFSET = 3

ATTACK_MOVE_PATTERN = jnp.array([
    [1,1],[0,1],[1,1],[1,1],[0,1],[1,1],[1,1],[0,1],[1,1],[1,1],[1,1],[0,1],[1,1],[1,1],[0,1]
])

ATTACK_TURN_PATTERN = jnp.array([
    [-1,1],[0,1],[-1,1],[0,1],[0,1],[-1,1],[0,1],[0,1],[-1,1],[0,1],[0,1],[0,1],[-1,1],[0,1],[0,1],[0,1],
    [0,1],[0,1],[0,1],[1,1],[0,1],[0,1],[0,1],[1,1],[0,1],[0,1],[1,1],[0,1],[0,1],[1,1],[0,1],[0,1],[1,1]
])

ATTACK_PAUSE_PATTERN = jnp.array([
    1,0,1,0,1,1,1,0
])

class GalaxianState(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    player_shooting_cooldown: chex.Array
    bullet_x: chex.Array
    bullet_y: chex.Array
    enemy_grid_x: chex.Array
    enemy_grid_y: chex.Array
    enemy_grid_alive: chex.Array # 0: dead, 1: alive, 2: attacking
    enemy_death_frame: chex.Array  # 0 = not-dying, 1–5 = which death sprite to show
    enemy_grid_direction: chex.Array
    enemy_attack_states: chex.Array      # 0: noop, 1: attack, 2: respawn
    enemy_attack_pos: chex.Array
    enemy_attack_x: chex.Array
    enemy_attack_y: chex.Array
    enemy_attack_direction: chex.Array  # -1: left, 1: right
    enemy_attack_target_x: chex.Array
    enemy_attack_target_y: chex.Array
    enemy_attack_turning: chex.Array    # -1: turning left, 1: turning right, 0: no turning
    enemy_attack_turn_step: chex.Array
    enemy_attack_move_step: chex.Array
    enemy_attack_respawn_timer: chex.Array
    enemy_attack_bullet_x: chex.Array
    enemy_attack_bullet_y: chex.Array
    enemy_attack_bullet_timer: chex.Array
    enemy_attack_pause_step: chex.Array
    lives: chex.Array
    player_alive: chex.Array
    player_respawn_timer: chex.Array
    score: chex.Array
    turn_step: chex.Array
    dive_probability: chex.Array
    enemy_bullet_max_cooldown: chex.Array
    enemy_attack_shot_timer: chex.Array
    enemy_attack_shots_fired: chex.Array
    enemy_attack_volley_size: chex.Array


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
        key = jax.random.PRNGKey(state.turn_step)
        do_new_dive = jax.random.uniform(key, shape=(), minval=0, maxval=100) < state.dive_probability / 10
        return jax.lax.cond(do_new_dive & jnp.any(state.enemy_grid_alive == 1), lambda state: initialise_new_dive(state), lambda state: continue_active_dives(state), state)

    @jax.jit
    def initialise_new_dive(state):
        key = jax.random.PRNGKey(state.turn_step + 101)  # currently deterministic
        key_choice, key_volley, key_shot_delay = jax.random.split(key, 3)

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
        new_shots_fired = state.enemy_attack_shots_fired.at[diver_idx].set(0)
        random_initial_delay = jax.random.randint(key_shot_delay, shape=(), minval=30, maxval=91)
        new_shot_timer = state.enemy_attack_shot_timer.at[diver_idx].set(random_initial_delay)

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

        # Richtung
        new_enemy_attack_direction = jnp.where(state.player_x < new_attack_x, -1.0, 1.0)
        new_attack_turning = state.enemy_attack_turning.at[diver_idx].set(0)
        new_attack_turn_step = state.enemy_attack_turn_step.at[diver_idx].set(0)

        # timer
        new_attack_respawn_timer = state.enemy_attack_respawn_timer.at[diver_idx].set(0)

        random_volley_size = jax.random.randint(key_volley, shape=(), minval=1, maxval=MAX_SHOTS_PER_VOLLEY + 1)
        new_volley_size = state.enemy_attack_volley_size.at[diver_idx].set(random_volley_size)

        return state._replace(
            enemy_attack_states=new_attack_states,
            enemy_attack_pos=new_attack_pos,
            enemy_attack_x=new_attack_x,
            enemy_attack_y=new_attack_y,
            enemy_grid_alive=new_grid_alive,
            enemy_attack_direction = new_enemy_attack_direction,
            enemy_attack_turning=new_attack_turning,
            enemy_attack_turn_step=new_attack_turn_step,
            enemy_attack_respawn_timer=new_attack_respawn_timer,
            enemy_attack_shots_fired=new_shots_fired,
            enemy_attack_shot_timer=new_shot_timer,
            enemy_attack_volley_size=new_volley_size
        )

    @jax.jit
    def continue_active_dives(state: GalaxianState) -> GalaxianState:
        # e.g. is_attacking: [True, False, True, False, True] (second killed by player)
        is_attacking = (state.enemy_attack_states == 1)
        target_x = state.player_x
        target_y = state.player_y

        curr_x = state.enemy_attack_x
        curr_y = state.enemy_attack_y

        def grey_dive(state: GalaxianState, i: int) -> GalaxianState:

            def update_enemy_state(state: GalaxianState) -> GalaxianState:
                new_enemy_attack_direction = jnp.where(
                    (state.enemy_attack_turning != 0) & (state.enemy_attack_turn_step == ENEMY_ATTACK_TURN_TIME), # change the direction, if the enemy is currently turning and the turn is over
                    state.enemy_attack_turning,
                    state.enemy_attack_direction
                )

                player_right = state.enemy_attack_x < state.player_x-10
                player_left = state.enemy_attack_x > state.player_x+10

                new_enemy_attack_turning = jnp.where(
                    (state.enemy_attack_turning == 0) & (state.enemy_attack_turn_step == 0) & (
                                player_right & (state.enemy_attack_direction == -1) | player_left & (state.enemy_attack_direction == 1)),
                    -state.enemy_attack_direction,  # change turning to the opposite of the current direction
                    jnp.where(
                        (state.enemy_attack_turn_step == ENEMY_ATTACK_TURN_TIME) | (state.enemy_attack_states == 0),
                        0,                         # reset turning, if turn is over or the enemy died
                        state.enemy_attack_turning)
                )

                new_enemy_attack_turn_step = jnp.where(
                    (state.enemy_attack_turn_step == ENEMY_ATTACK_TURN_TIME) | (state.enemy_attack_states == 0),
                    0,
                    jnp.where(
                        state.enemy_attack_turning != 0,
                        state.enemy_attack_turn_step + 1,
                        state.enemy_attack_turn_step
                    )
                )

                new_enemy_attack_move_step = jnp.where(
                    (state.enemy_attack_move_step == 15) | (state.enemy_attack_states == 0),
                    0,
                    jnp.where(
                        state.enemy_attack_turning == 0,
                        state.enemy_attack_move_step + 1,
                        state.enemy_attack_move_step
                    )
                )

                return state._replace(
                    enemy_attack_direction=new_enemy_attack_direction,
                    enemy_attack_turning=new_enemy_attack_turning,
                    enemy_attack_turn_step=new_enemy_attack_turn_step,
                    enemy_attack_move_step=new_enemy_attack_move_step
                )

            def update_enemy_position(state: GalaxianState) -> GalaxianState:

                delta_x = jnp.where(
                    state.enemy_attack_states == 1,
                    jnp.where(
                        state.enemy_attack_turning == 0,
                        ATTACK_MOVE_PATTERN[state.enemy_attack_move_step, 0] * state.enemy_attack_direction,
                        ATTACK_TURN_PATTERN[state.enemy_attack_turn_step, 0] * state.enemy_attack_turning,
                    ),
                    0
                )

                delta_y = jnp.where(
                    state.enemy_attack_states == 1,
                    1,
                    0
                )


                return state._replace(
                     enemy_attack_x=curr_x + delta_x,
                     enemy_attack_y=curr_y + delta_y,
                 )

            state = update_enemy_state(state)
            state = update_enemy_position(state)
            return state

        def red_dive(state: GalaxianState, i: int) -> GalaxianState:
            #jax.debug.print("red")
            return grey_dive(state, i)

        def purple_dive(state: GalaxianState, i: int) -> GalaxianState:
            #jax.debug.print("purple")
            return grey_dive(state, i)

        def white_dive(state: GalaxianState, i: int) -> GalaxianState:
            #jax.debug.print("white")
            return grey_dive(state, i)

        new_pause_step = jnp.where(
            state.enemy_attack_pause_step < 7,
            state.enemy_attack_pause_step + 1,
        0
        )

        state = state._replace(
            enemy_attack_pause_step= new_pause_step,
        )

        def dive_loop(state):
            return lax.fori_loop(
                0, MAX_DIVERS,
                lambda i, state: jax.lax.cond(
                    state.enemy_attack_pos[i, 0] == 5,
                    lambda state: white_dive(state, i),
                    lambda state: jax.lax.cond(
                        state.enemy_attack_pos[i, 0] == 4,
                        lambda state: red_dive(state, i),
                        lambda state: jax.lax.cond(
                            state.enemy_attack_pos[i, 0] == 3,
                            lambda state: purple_dive(state, i),
                            lambda state: grey_dive(state, i),
                            state
                        ),
                        state
                    ),
                    state
                ),
                state
            )

        return jax.lax.cond(
            ATTACK_PAUSE_PATTERN[new_pause_step] == 1,
            lambda state: dive_loop(state),
            lambda state: state,
            state
        )

    @jax.jit
    def respawn_finished_dives(state: GalaxianState) -> GalaxianState:
        def body(i, new_state):

            #diver unter dem player und außerhalb window werden auf respawn gesetzt
            respawn_condition = jnp.logical_and(jnp.logical_or(new_state.enemy_attack_y[i] > DIVE_KILL_Y - 30,
                                               jnp.logical_or(new_state.enemy_attack_x[i] < ENEMY_LEFT_BOUND,
                                                              new_state.enemy_attack_x[i] > ENEMY_RIGHT_BOUND)),
                                                  new_state.enemy_attack_states[i] == 1)

            new_state = jax.lax.cond(
                respawn_condition,
                lambda state: state._replace(
                    enemy_attack_states=state.enemy_attack_states.at[i].set(2),
                    enemy_attack_x=state.enemy_attack_x.at[i].set(
                        state.enemy_grid_x[state.enemy_attack_pos[i, 0], state.enemy_attack_pos[i, 1]]
                    ),
                    enemy_attack_y=state.enemy_attack_y.at[i].set(-10)
                ),
                lambda state: state,
                new_state
            )

            #continue respawnende diver
            new_state = jax.lax.cond(
                new_state.enemy_attack_states[i] == 2,
                lambda state: state._replace(
                    enemy_attack_y=state.enemy_attack_y.at[i].set(
                        lax.clamp(
                            jnp.array(-10, dtype=state.enemy_attack_y.dtype),
                            (state.enemy_attack_y[i] + 1).astype(state.enemy_attack_y.dtype),
                            state.enemy_grid_y[state.enemy_attack_pos[i, 0], state.enemy_attack_pos[i, 1]],
                        )
                    ),
                    enemy_attack_x=state.enemy_attack_x.at[i].set(
                        state.enemy_grid_x[state.enemy_attack_pos[i, 0], state.enemy_attack_pos[i, 1]]
                    )
                ),
                lambda state: state,
                new_state
            )

            # beende respawn
            new_state = jax.lax.cond(
                (new_state.enemy_attack_states[i] == 2) &
                (new_state.enemy_attack_x[i] == new_state.enemy_grid_x[
                    new_state.enemy_attack_pos[i, 0], new_state.enemy_attack_pos[i, 1]]) &
                (new_state.enemy_attack_y[i] == new_state.enemy_grid_y[
                    new_state.enemy_attack_pos[i, 0], new_state.enemy_attack_pos[i, 1]]),
                lambda state: state._replace(
                    enemy_attack_states=state.enemy_attack_states.at[i].set(0),
                    enemy_grid_alive=state.enemy_grid_alive.at[tuple(state.enemy_attack_pos[i])].set(1)
                ),
                lambda state: state,
                new_state
            )
            return new_state

        return lax.fori_loop(0, MAX_DIVERS, body, state)


    new_state = respawn_finished_dives(state)
    free_slots = jnp.sum(state.enemy_attack_states == 0)
    return jax.lax.cond(
        free_slots > 0,
        lambda state: test_for_new_dive(state),
        lambda state: continue_active_dives(state),
        new_state
    )


@jax.jit
def update_enemy_bullets(state: 'GalaxianState') -> 'GalaxianState':

    #volley timer
    new_timers = jnp.maximum(0, state.enemy_attack_shot_timer - 1)

    # continue active bullets
    is_active_mask = state.enemy_attack_bullet_y >= 0.0
    moved_y = jnp.where(is_active_mask, state.enemy_attack_bullet_y + ENEMY_ATTACK_BULLET_SPEED, -1.0)
    moved_x = jnp.where(is_active_mask, state.enemy_attack_bullet_x, -1.0)

    # despawn bullets below player
    off_screen_mask = moved_y > NATIVE_GAME_HEIGHT
    bullets_y_after_move = jnp.where(off_screen_mask, -1.0, moved_y)
    bullets_x_after_move = jnp.where(off_screen_mask, -1.0, moved_x)

    # continue volley
    can_shoot_mask = (
        (state.enemy_attack_states == 1) &
        (state.enemy_attack_shots_fired < state.enemy_attack_volley_size) &
        (new_timers <= 0)
    )

    # select new shooters
    potential_shooter_indices = jnp.where(can_shoot_mask, jnp.arange(MAX_DIVERS), MAX_DIVERS)
    sorted_shooter_indices = jnp.sort(potential_shooter_indices)


    def _spawn_one_shot(shooter_idx, carry):
        #find empty slot
        (current_x, current_y, current_shots_fired, current_timers) = carry
        available_bullet_slots = jnp.where(current_y == -1.0, size=1, fill_value=-1)[0]
        target_bullet_slot = available_bullet_slots[0]
        can_spawn = (shooter_idx < MAX_DIVERS) & (target_bullet_slot != -1)

        def _do_spawn(x, y, shots_fired, timers):
            new_x = x.at[target_bullet_slot].set(state.enemy_attack_x[shooter_idx])
            new_y = y.at[target_bullet_slot].set(state.enemy_attack_y[shooter_idx])
            new_shots_fired = shots_fired.at[shooter_idx].set(shots_fired[shooter_idx] + 1)
            new_timers = timers.at[shooter_idx].set(VOLLEY_SHOT_DELAY)
            return new_x, new_y, new_shots_fired, new_timers

        def _do_not_spawn(x, y, shots_fired, timers):
            return x, y, shots_fired, timers

        return lax.cond(
            can_spawn,
            lambda: _do_spawn(current_x, current_y, current_shots_fired, current_timers),
            lambda: _do_not_spawn(current_x, current_y, current_shots_fired, current_timers)
        )

    initial_carry = (bullets_x_after_move, bullets_y_after_move, state.enemy_attack_shots_fired, new_timers)


    final_bullet_x, final_bullet_y, final_shots_fired, final_timers = lax.fori_loop(
        0, MAX_DIVERS,
        lambda i, carry: _spawn_one_shot(sorted_shooter_indices[i], carry),
        initial_carry
    )

    return state._replace(
        enemy_attack_bullet_x=final_bullet_x,
        enemy_attack_bullet_y=final_bullet_y,
        enemy_attack_shots_fired=final_shots_fired,
        enemy_attack_shot_timer=final_timers
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
def update_player_bullet(state: GalaxianState, action: chex.Array) -> GalaxianState:
    is_shooting_action = jnp.any(
        jnp.array([
            action == Action.FIRE,
            action == Action.RIGHTFIRE,
            action == Action.LEFTFIRE,
        ])
    )

    bullet_is_inactive = state.bullet_y < 0

    def fire_new_bullet(state: GalaxianState) -> GalaxianState:
        return state._replace(
            bullet_x=state.player_x + PLAYER_BULLET_X_OFFSET,
            bullet_y=state.player_y - PLAYER_BULLET_Y_OFFSET
        )

    def move_active_bullet(state: GalaxianState) -> GalaxianState:
        new_y = state.bullet_y - BULLET_MOVE_SPEED

        return state._replace(
            bullet_x=jnp.where(new_y < 0, -1.0, state.bullet_x),
            bullet_y=jnp.where(new_y < 0, -1.0, new_y)
        )

    return lax.cond(
        bullet_is_inactive,
        lambda state: lax.cond(
            is_shooting_action,
            fire_new_bullet,
            lambda state: state,
            state
        ),
        move_active_bullet,
        state
    )


@jax.jit
def bullet_collision(state: GalaxianState) -> GalaxianState:
    x_diff = jnp.abs(state.bullet_x - state.enemy_grid_x)
    y_diff = jnp.abs(state.bullet_y - state.enemy_grid_y)

    mask = (x_diff <= 10) & (y_diff <= 10) & (state.enemy_grid_alive == 1)
    hit  = jnp.any(mask)

    def process_hit(s_and_idx):
        s, (rows, cols) = s_and_idx
        hit_rows = rows[0] # first hit
        hit_cols = cols[0]
        # setze alle getroffenen auf dead
        new_death = s.enemy_death_frame.at[hit_rows, hit_cols].set(1)

        new_alive = s.enemy_grid_alive.at[hit_rows, hit_cols].set(2)  # 2 = dying
        return s._replace(
            enemy_grid_alive=new_alive,
            enemy_death_frame=new_death,
            bullet_x=-1.0, bullet_y=-1.0,
            score=s.score + 30
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
def update_death_frames(state: GalaxianState) -> GalaxianState:
    def advance_cell(frame):
        # if in 1..5, increment or then clear
        return jnp.where(frame == 0, 0,
                         jnp.where(frame < 5, frame + 1, 0))
    new_frames = jax.vmap(jax.vmap(advance_cell))(state.enemy_death_frame)
    # once frame wraps to 0, also mark the cell fully dead
    cleared_mask = (state.enemy_death_frame == 5)
    new_alive = jnp.where(cleared_mask, 0, state.enemy_grid_alive)
    return state._replace(
        enemy_death_frame=new_frames,
        enemy_grid_alive=new_alive
    )

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
        hit_indices = jnp.where(collision_mask, size=MAX_DIVERS, fill_value=-1)[0]  # size müsste man irgendwann vllt anpassen
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
                              enemy_death_frame=jnp.zeros((GRID_ROWS, GRID_COLS), dtype=jnp.int32),
                              enemy_grid_direction=jnp.array(1),
                              enemy_attack_states=jnp.zeros(MAX_DIVERS),
                              enemy_attack_pos=jnp.full((MAX_DIVERS, 2), -1, dtype=jnp.int32),
                              enemy_attack_direction=jnp.zeros(MAX_DIVERS),
                              enemy_attack_turning=jnp.zeros(MAX_DIVERS),
                              enemy_attack_turn_step=jnp.zeros(MAX_DIVERS, dtype=jnp.int32),
                              enemy_attack_move_step=jnp.zeros(MAX_DIVERS, dtype=jnp.int32),
                              enemy_attack_x=jnp.zeros(MAX_DIVERS),
                              enemy_attack_y=jnp.zeros(MAX_DIVERS),
                              enemy_attack_respawn_timer=jnp.zeros(MAX_DIVERS),
                              enemy_attack_bullet_x=jnp.full(MAX_DIVERS, -1.0),
                              enemy_attack_bullet_y=jnp.full(MAX_DIVERS, -1.0),
                              enemy_attack_bullet_timer=jnp.zeros(MAX_DIVERS),
                              enemy_attack_pause_step=jnp.array(0),
                              lives=jnp.array(3),
                              player_alive=jnp.array(True),
                              player_respawn_timer=jnp.array(PLAYER_DEATH_DELAY),
                              score=jnp.array(0, dtype=jnp.int32),
                              enemy_attack_target_x=jnp.zeros(MAX_DIVERS),
                              enemy_attack_target_y=jnp.zeros(MAX_DIVERS),
                              turn_step=jnp.array(0),
                              dive_probability=jnp.array(BASE_DIVE_PROBABILITY),
                              enemy_bullet_max_cooldown=ENEMY_ATTACK_BULLET_DELAY,
                              enemy_attack_shot_timer=jnp.zeros(MAX_DIVERS),
                              enemy_attack_shots_fired=jnp.zeros(MAX_DIVERS, dtype=jnp.int32),
                              enemy_attack_volley_size=jnp.zeros(MAX_DIVERS, dtype=jnp.int32)
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
        new_state = update_player_bullet(new_state, action)
        new_state = update_enemy_positions(new_state)
        new_state = bullet_collision(new_state)
        new_state = bullet_collision_attack(new_state)
        new_state = update_death_frames(new_state)
        new_state = update_enemy_attack(new_state)
        new_state = update_enemy_bullets(new_state)
        new_state = check_player_death_by_enemy(new_state)
        new_state = check_player_death_by_bullet(new_state)
        new_state = new_state._replace(turn_step=new_state.turn_step + 1)

        new_state = jax.lax.cond(jnp.logical_and(jnp.logical_not(jnp.any(state.enemy_grid_alive == 1)), jnp.logical_not(jnp.any(state.enemy_attack_states != 0))), lambda new_state: enter_new_wave(new_state), lambda s: s, new_state)

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
        return state.lives <= 0

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: GalaxianState, all_rewards: chex.Array) -> GalaxianInfo:
        return GalaxianInfo(time=state.turn_step, all_rewards=all_rewards)

# helper function to normalize frame dimensions to a target shape
def normalize_frame(frame: jnp.ndarray, target_shape: Tuple[int, int, int]) -> jnp.ndarray:
    h, w, c = frame.shape
    th, tw, tc = target_shape
    assert c == tc, f"Channel mismatch: {c} vs {tc}"

    # Pad or crop vertically
    if h < th:
        top = (th - h) // 2
        bottom = th - h - top
        frame = jnp.pad(frame, ((top, bottom), (0, 0), (0, 0)), constant_values=0)
    elif h > th:
        crop = (h - th) // 2
        frame = frame[crop:crop + th, :, :]

    # Pad or crop horizontally
    if w < tw:
        left = (tw - w) // 2
        right = tw - w - left
        frame = jnp.pad(frame, ((0, 0), (left, right), (0, 0)), constant_values=0)
    elif w > tw:
        crop = (w - tw) // 2
        frame = frame[:, crop:crop + tw, :]

    return frame

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
    enemy_purple = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/galaxian/purple_blue_enemy_1.npy"),transpose=True)
    enemy_white = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/galaxian/white_enemy_1.npy"),transpose=True)
    death_enemy_1 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/galaxian/death_enemy_1.npy"),transpose=True)
    death_enemy_2 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/galaxian/death_enemy_2.npy"),transpose=True)
    death_enemy_3 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/galaxian/death_enemy_3.npy"),transpose=True)
    death_enemy_4 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/galaxian/death_enemy_4.npy"),transpose=True)
    death_enemy_5 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/galaxian/death_enemy_5.npy"),transpose=True)

    # normalize frames to the same shape
    target_shape = enemy_gray.shape
    death_enemy_1 = normalize_frame(death_enemy_1, target_shape)
    death_enemy_2 = normalize_frame(death_enemy_2, target_shape)
    death_enemy_3 = normalize_frame(death_enemy_3, target_shape)
    death_enemy_4 = normalize_frame(death_enemy_4, target_shape)
    death_enemy_5 = normalize_frame(death_enemy_5, target_shape)

    SPRITE_BG = bg[jnp.newaxis, ...]
    SPRITE_PLAYER = player[jnp.newaxis, ...]
    SPRITE_BULLET = bullet[jnp.newaxis, ...]
    SPRITE_ENEMY_GRAY = enemy_gray[jnp.newaxis, ...]
    SPRITE_ENEMY_RED = enemy_red[jnp.newaxis, ...]
    SPRITE_ENEMY_PURPLE = enemy_purple[jnp.newaxis, ...]
    SPRITE_ENEMY_WHITE = enemy_white[jnp.newaxis, ...]
    SPRITE_LIFE = life[jnp.newaxis, ...]
    SPRITE_ENEMY_BULLET = enemy_bullet[jnp.newaxis, ...]
    SPRITE_ENEMY_DEATH = jnp.stack([death_enemy_1, death_enemy_2, death_enemy_3,
                                    death_enemy_4, death_enemy_5], axis=0)
    return(
        SPRITE_BG,
        SPRITE_PLAYER,
        SPRITE_BULLET,
        SPRITE_ENEMY_GRAY,
        SPRITE_ENEMY_RED,
        SPRITE_ENEMY_PURPLE,
        SPRITE_ENEMY_WHITE,
        SPRITE_LIFE,
        SPRITE_ENEMY_BULLET,
        SPRITE_ENEMY_DEATH
    )

class GalaxianRenderer(AtraJaxisRenderer):
    def __init__(self):
        (
            self.SPRITE_BG,
            self.SPRITE_PLAYER,
            self.SPRITE_BULLET,
            self.SPRITE_ENEMY_GRAY,
            self.SPRITE_ENEMY_RED,
            self.SPRITE_ENEMY_PURPLE,
            self.SPRITE_ENEMY_WHITE,
            self.SPRITE_LIFE,
            self.SPRITE_ENEMY_BULLET,
            self.SPRITE_ENEMY_DEATH
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
        def draw_bullet_active(r):
            bullet = aj.get_sprite_frame(self.SPRITE_BULLET, 0)
            bullet_x = jnp.round(state.bullet_x).astype(jnp.int32)
            bullet_y = jnp.round(state.bullet_y).astype(jnp.int32)
            return aj.render_at(r, bullet_x, bullet_y, bullet)
        def draw_bullet_inactive(r):
            bullet = aj.get_sprite_frame(self.SPRITE_BULLET, 0)
            bullet_x = jnp.round(state.player_x + PLAYER_BULLET_X_OFFSET).astype(jnp.int32)
            bullet_y = jnp.round(state.player_y - PLAYER_BULLET_Y_OFFSET).astype(jnp.int32)
            return aj.render_at(r, bullet_x, bullet_y, bullet)
        raster = lax.cond(state.bullet_y > 0, draw_bullet_active, draw_bullet_inactive, raster)

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
                cond = state.enemy_attack_states[i] >= 1

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
                death_frame = state.enemy_death_frame[i, j].astype(jnp.int32)
                alive = (state.enemy_grid_alive[i, j] == 1)


                def draw_death(r0):
                    sprite = get_sprite_frame(self.SPRITE_ENEMY_DEATH, death_frame - 1)
                    x = jnp.round(state.enemy_grid_x[i, j]).astype(jnp.int32)
                    y = jnp.round(state.enemy_grid_y[i, j]).astype(jnp.int32)
                    return render_at(r0, x, y, sprite)


                def draw_alive(r0):
                    conds = [i == 5, i == 4, i == 3]
                    choices = [
                        get_sprite_frame(self.SPRITE_ENEMY_WHITE, 0),
                        get_sprite_frame(self.SPRITE_ENEMY_RED, 0),
                        get_sprite_frame(self.SPRITE_ENEMY_PURPLE, 0),
                    ]
                    default = get_sprite_frame(self.SPRITE_ENEMY_GRAY, 0)
                    sprite = jnp.select(conds, choices, default)
                    x = jnp.round(state.enemy_grid_x[i, j]).astype(jnp.int32)
                    y = jnp.round(state.enemy_grid_y[i, j]).astype(jnp.int32)
                    return render_at(r0, x, y, sprite)

                # choose: death‐anim if df>0; else alive‐sprite if alive; else no draw
                return lax.cond(
                    death_frame > 0,
                    draw_death,
                    lambda r0: lax.cond(alive, draw_alive, lambda r1: r1, r0),
                    r_inner
                )

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




