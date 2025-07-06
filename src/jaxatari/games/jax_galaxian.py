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
GRID_COLS = 7 #TODO needs to be 7, but >5 is clamping right
NATIVE_GAME_WIDTH = 160
NATIVE_GAME_HEIGHT = 210
PYGAME_SCALE_FACTOR = 3
PYGAME_WINDOW_WIDTH = NATIVE_GAME_WIDTH * PYGAME_SCALE_FACTOR
PYGAME_WINDOW_HEIGHT = NATIVE_GAME_HEIGHT * PYGAME_SCALE_FACTOR
ENEMY_SPACING_X = 14
ENEMY_SPACING_Y = 11
ENEMY_GRID_Y = 80
START_X = NATIVE_GAME_WIDTH // 4
START_Y = NATIVE_GAME_HEIGHT
ENEMY_ATTACK_SPEED = 2
ENEMY_ATTACK_TURN_TIME = 32
ENEMY_ATTACK_BULLET_SPEED = 5
ENEMY_ATTACK_BULLET_DELAY = 75
ENEMY_ATTACK_MAX_BULLETS = 2
LIVES = 2
PLAYER_DEATH_DELAY = 50
ENEMY_LEFT_BOUND = 17
ENEMY_RIGHT_BOUND = NATIVE_GAME_WIDTH - 25
DIVE_KILL_Y = 175
DIVE_SPEED = 0.5
MAX_DIVERS = 5
MAX_SUPPORT_CALLERS = 2
MAX_SUPPORTERS = 2
BASE_DIVE_PROBABILITY = 30 #TODO back to 5 before merge
MAX_SHOTS_PER_VOLLEY = 4
VOLLEY_SHOT_DELAY = 10
VOLLEY_PROBABILITIES = jnp.array([0.4, 0.3, 0.2, 0.1])

ERROR_VALUE = -9999

PLAYER_BULLET_Y_OFFSET = 3
PLAYER_BULLET_X_OFFSET = 3

ENEMY_GRID = jnp.array([
    [1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1],
    [0,0,1,0,1,0,0],
]).astype(jnp.float32)

TEST_GRID = jnp.array([
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [1,1,1,1,1,1,1],
    [0,0,1,0,1,0,0],
]).astype(jnp.float32)

ATTACK_MOVE_PATTERN = jnp.array([
    [
        [1,1],[0,1],[1,1],[1,1],[0,1],[1,1],[1,1],[0,1],[1,1],[1,1],[1,1],[0,1],[1,1],[1,1],[0,1]
    ],
    [
        [2,1],[2,1],[2,1],[2,1],[2,1],[2,1],[2,1],[2,1],[2,1],[2,1],[2,1],[2,1],[2,1],[2,1],[2,1]
    ]
])

ATTACK_TURN_PATTERN = jnp.array([
    [
        [-1,1],[0,1],[-1,1],[0,1],[0,1],[-1,1],[0,1],[0,1],[-1,1],[0,1],[0,1],[0,1],[-1,1],[0,1],[0,1],[0,1],
        [0,1],[0,1],[0,1],[1,1],[0,1],[0,1],[0,1],[1,1],[0,1],[0,1],[1,1],[0,1],[0,1],[1,1],[0,1],[1,1]
    ],
    [
        [-1,1],[-1,1],[-2,1],[-1,1],[-1,1],[-1,1],[-1,1],[-1,1],[-1,1],[-1,1],[0,1],[-1,1],[0,1],[0,1],[0,1],[0,1],
        [0,1],[0,1],[0,1],[0,1],[1,1],[0,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[2,1],[1,1],[1,1]
    ]
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
    enemy_grid_alive: chex.Array   # 0: dead, 1: alive, 2: attacking, 3: supporting
    enemy_death_frame_grid: chex.Array
    enemy_death_frame_attack: chex.Array
    enemy_death_frame_support: chex.Array # 0 = not-dying, 1–5 = which death sprite to show
    enemy_grid_direction: chex.Array
    enemy_attack_states: chex.Array      # 0: unused, 1: attack, 2: respawn 3: dead_caller, 4: dying
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
    enemy_attack_number: chex.Array
    enemy_attack_max: chex.Array
    enemy_support_caller_idx: chex.Array
    enemy_support_states: chex.Array  # 0: unused, 1: support, 2: respawn, 4: dying
    enemy_support_pos: chex.Array
    enemy_support_x: chex.Array
    enemy_support_y: chex.Array
    level: chex.Array
    lives: chex.Array
    player_alive: chex.Array
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
    new_x = jnp.where(
        state.player_alive == True,
        state.player_x + (press_right * 5) - (press_left * 5),
        state.player_x
    )

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
        first_available_slot = jnp.where(state.enemy_attack_states == 0, size=MAX_DIVERS, fill_value=-1)[0][0]

        return jax.lax.cond(jnp.any(state.enemy_grid_alive == 1) & (first_available_slot <= state.level) & (state.player_alive == True), lambda state: initialise_new_dive(state,first_available_slot), lambda state: state, state)

    def choose_diver(state, rows, row_number, key_choice):
        key_row, key_col = jax.random.split(key_choice, 2)

        # finde zufällige nicht-leere Reihe
        valid_rows = jnp.where(rows, size=row_number, fill_value=-1)[0]
        num_valid_rows = jnp.sum(rows)
        random_row_idx = jax.random.randint(key_row, shape=(), minval=0, maxval=num_valid_rows)
        chosen_enemy_row = valid_rows[random_row_idx]

        return choose_column(state, chosen_enemy_row, key_col)

    def choose_column(state, row, key_col):
        # zufällig entscheiden, ob erster oder letzter Feind der Reihe genommen wird
        first_or_last = jax.random.randint(key_col, shape=(), minval=0, maxval=2)
        row_alive_mask = (state.enemy_grid_alive[row] == 1)
        alive_cols = jnp.where(row_alive_mask, size=GRID_COLS, fill_value=-1)[0]
        first_alive_col = alive_cols[0]
        last_alive_col = alive_cols[jnp.sum(row_alive_mask) - 1]
        chosen_enemy_col = jnp.where(first_or_last == 0, first_alive_col, last_alive_col)

        return row, chosen_enemy_col

    def choose_diver_level_0(state, diver_idx, key_choice):
        #jax.debug.print("enemy_grid_alive: {}", state.enemy_grid_alive)
        rows_with_alive = jnp.any(state.enemy_grid_alive == 1, axis=1)

        num_valid_lower_rows = jnp.sum(rows_with_alive[:4])
        num_valid_white_rows = jnp.sum(rows_with_alive[-1])

        def lower_rows(state, key_choice):
            return choose_diver(state, rows_with_alive[:4],4, key_choice)

        def white_row(state, key_choice):
            return jax.lax.cond(
                state.enemy_support_caller_idx == -1,
                lambda state: choose_column(state,5, key_choice),
                lambda state: (-1,-1),
                state)

        def red_row(state,key_choice):
            return choose_column(state, 4, key_choice)

        return jax.lax.cond(
            num_valid_lower_rows > 0,
            lambda state: lower_rows(state, key_choice),
            lambda state: jax.lax.cond(
                num_valid_white_rows > 0,
                lambda state: white_row(state, key_choice),
                lambda state: red_row(state, key_choice),
                state
            ),
            state
        )

    def choose_diver_default(state, key_choice):
        rows_with_alive = jnp.any(state.enemy_grid_alive == 1, axis=1)
        return choose_diver(state,rows_with_alive,GRID_ROWS, key_choice)


    @jax.jit
    def initialise_new_dive(state,diver_idx):
        key = jax.random.PRNGKey(state.turn_step + 101)  # currently deterministic
        key_choice, key_volley, key_shot_delay = jax.random.split(key, 3)

        chosen_enemy_row, chosen_enemy_col = choose_diver_level_0(state, diver_idx, key_choice)


        def initialise(state, diver_idx):
            # e.g. enemy_attack_states=[1. 0. 0. 1. 1.] -> enemy_attack_states=[1. 1. 0. 1. 1.]
            new_attack_states = state.enemy_attack_states.at[diver_idx].set(1)  # 1: actively attacking
            new_shots_fired = state.enemy_attack_shots_fired.at[diver_idx].set(0)
            random_initial_delay = jax.random.randint(key_shot_delay, shape=(), minval=30, maxval=91)
            new_shot_timer = state.enemy_attack_shot_timer.at[diver_idx].set(random_initial_delay)

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

            chosen_index = jax.random.categorical(key_volley, jnp.log(VOLLEY_PROBABILITIES))
            random_volley_size = chosen_index + 1
            new_volley_size = state.enemy_attack_volley_size.at[diver_idx].set(random_volley_size)

            def initialise_supporters(state):
                new_enemy_support_caller_idx = diver_idx
                row, column = choose_column(state, 5, key_choice)

                red_row = state.enemy_grid_alive[4]

                # Finde die indizierten Werte in der Nähe der gewählten Spalte

                nearby_mask = jnp.zeros_like(red_row, dtype=jnp.bool_)
                nearby_mask = nearby_mask.at[jnp.clip(column-1, 0, GRID_COLS-1)].set(True)
                nearby_mask = nearby_mask.at[column].set(True)
                nearby_mask = nearby_mask.at[jnp.clip(column+1, 0, GRID_COLS-1)].set(True)

                # Kombiniere die Nähe-Maske mit der "ist lebendig"-Maske
                valid_support_mask = nearby_mask & (red_row == 1)

                # Finde die Indizes der ersten beiden gültigen Unterstützer, von rechts beginnend
                sorted_indices = jnp.sort(jnp.where(valid_support_mask, jnp.arange(GRID_COLS), -1))[
                                 ::-1]  # Absteigend sortieren
                support_indices = sorted_indices[:2]  # Nimm die ersten beiden (höchsten) Indizes

                # Wenn weniger als 2 Unterstützer gefunden wurden, fülle mit -1 auf
                padded_support_indices = jnp.pad(support_indices, (0, 2 - len(support_indices)), constant_values=-1)[:2]

                #jax.debug.print("support_indices: {}, padded_support_indices: {}", padded_support_indices, padded_support_indices)
                new_enemy_support_states = jnp.array([
                    jnp.where(padded_support_indices[i] != -1, 1, 0)  # 1 für aktive Unterstützer, 0 für inaktive
                    for i in range(MAX_SUPPORTERS)
                ], dtype=jnp.int32)

                new_enemy_support_pos = jnp.array([
                    [jnp.where(padded_support_indices[i] != -1, 4, -1),
                     jnp.where(padded_support_indices[i] != -1, padded_support_indices[i], -1)]
                    for i in range(MAX_SUPPORTERS)
                ], dtype=jnp.int32)

                new_support_x = jnp.array([
                    jnp.where(new_enemy_support_pos[i, 0] != -1,
                              state.enemy_grid_x[new_enemy_support_pos[i, 0], new_enemy_support_pos[i, 1]],
                              -1.0)
                    for i in range(MAX_SUPPORTERS)
                ], dtype=jnp.float32)
                new_support_y = jnp.array([
                    jnp.where(new_enemy_support_pos[i, 0] != -1,
                              state.enemy_grid_y[new_enemy_support_pos[i, 0], new_enemy_support_pos[i, 1]],
                              -1.0)
                    for i in range(MAX_SUPPORTERS)
                ], dtype=jnp.float32)

                new_enemy_grid = state.enemy_grid_alive

                for i in range(MAX_SUPPORTERS):
                    idx = padded_support_indices[i]
                    new_enemy_grid = jax.lax.cond(
                        idx != -1,
                        lambda g: g.at[4, idx].set(2),
                        lambda g: g,
                        new_enemy_grid
                    )

                return state._replace(
                    enemy_support_caller_idx=new_enemy_support_caller_idx,
                    enemy_support_states= new_enemy_support_states,
                    enemy_support_pos=new_enemy_support_pos,
                    enemy_support_x = new_support_x,
                    enemy_support_y = new_support_y,
                    enemy_grid_alive=new_enemy_grid
                )

            # init supporters, if diver is white
            state = jax.lax.cond(
                chosen_enemy_row == 5,
                lambda state: initialise_supporters(state),
                lambda state: state,
                state
            )

            # alte Position für respawn speichern
            new_attack_pos = state.enemy_attack_pos.at[diver_idx].set(
                jnp.array([chosen_enemy_row, chosen_enemy_col], dtype=jnp.int32)
            )
            new_grid_alive = state.enemy_grid_alive.at[chosen_enemy_row, chosen_enemy_col].set(2)

            #jax.debug.print("enemy_grid_alive: {}", state.enemy_grid_alive)

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

        return jax.lax.cond(
            chosen_enemy_row >-1,
            lambda state: initialise(state, diver_idx),
            lambda state: state._replace(),
            state)

    @jax.jit
    def continue_active_dives(state: GalaxianState) -> GalaxianState:
        # e.g. is_attacking: [True, False, True, False, True] (second killed by player)
        is_attacking = (state.enemy_attack_states == 1)
        target_x = state.player_x
        target_y = state.player_y

        curr_x = state.enemy_attack_x
        curr_y = state.enemy_attack_y

        def dive(state: GalaxianState) -> GalaxianState:

            def update_enemy_state(state: GalaxianState) -> GalaxianState:

                #jax.debug.print("enemy_attack_states: {}", state.enemy_attack_states)

                new_enemy_attack_direction = jnp.where(
                    (state.enemy_attack_turning != 0) & (state.enemy_attack_turn_step == ENEMY_ATTACK_TURN_TIME), # change the direction, if the enemy is currently turning and the turn is over
                    state.enemy_attack_turning,
                    state.enemy_attack_direction
                )

                player_right = state.enemy_attack_x < state.player_x-10
                player_left = state.enemy_attack_x > state.player_x+10

                new_enemy_attack_turning = jnp.where(
                    (state.enemy_attack_turn_step == ENEMY_ATTACK_TURN_TIME) | (state.enemy_attack_states == 0) | (
                                state.enemy_attack_states == 2),
                    0,  # reset turning, if turn is over or the enemy died
                    jnp.where(
                        (state.enemy_attack_turning == 0) & (state.enemy_attack_turn_step == 0) & (
                                player_right & (state.enemy_attack_direction == -1) | player_left & (
                                    state.enemy_attack_direction == 1)),
                        -state.enemy_attack_direction,  # change turning to the opposite of the current direction
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


                caller_idx = state.enemy_support_caller_idx


                # Setze zurück, falls keine Supporter mehr übrig
                new_enemy_attack_states, new_support_caller_idx = lax.cond(
                    (caller_idx != -1) & (jnp.logical_not(jnp.any(jnp.logical_or(state.enemy_support_states == 1, state.enemy_support_states == 2)))) & (state.enemy_attack_states[caller_idx] == 3),
                    lambda state: (state.enemy_attack_states.at[caller_idx].set(0), -1),
                    lambda state: (state.enemy_attack_states, state.enemy_support_caller_idx),
                    state
                )


                return state._replace(
                    enemy_attack_direction=new_enemy_attack_direction,
                    enemy_attack_turning=new_enemy_attack_turning,
                    enemy_attack_turn_step=new_enemy_attack_turn_step,
                    enemy_attack_move_step=new_enemy_attack_move_step,
                    enemy_attack_states = new_enemy_attack_states,
                    enemy_support_caller_idx = new_support_caller_idx
                )

            def update_enemy_position(state: GalaxianState) -> GalaxianState:

                delta_x = jnp.where(
                    jnp.logical_or(state.enemy_attack_states == 1, state.enemy_attack_states == 3),
                    jnp.where(
                        state.enemy_attack_turning == 0,
                        jnp.where(
                            state.enemy_attack_pos[:, 0] == 3,
                            ATTACK_MOVE_PATTERN[1, state.enemy_attack_move_step, 0] * state.enemy_attack_direction,
                            ATTACK_MOVE_PATTERN[0, state.enemy_attack_move_step, 0] * state.enemy_attack_direction
                        ),
                        jnp.where(
                            state.enemy_attack_pos[:, 0] == 3,
                            ATTACK_TURN_PATTERN[1, state.enemy_attack_turn_step, 0] * state.enemy_attack_turning,
                            ATTACK_TURN_PATTERN[0,state.enemy_attack_turn_step, 0] * state.enemy_attack_turning
                        )
                    ),
                    0
                )

                delta_y = jnp.where(
                    jnp.logical_or(state.enemy_attack_states == 1, state.enemy_attack_states == 3),
                    1,
                    0
                )

                # Falls ein weißer Angreifer aus Reihe 5 ist, bewege auch die Supporter
                support_caller_active = state.enemy_support_caller_idx != -1

                # Berechne neue Support-Positionen mit jnp.where
                curr_support_x = state.enemy_support_x
                curr_support_y = state.enemy_support_y

                # Vektorisierte Bewegungsanwendung auf Supporter
                supporter_active = state.enemy_support_states == 1
                # Nur Bewegung anwenden, wenn Support aktiv ist und Caller ein weißer Feind ist
                delta_x_support = jnp.where(supporter_active & support_caller_active,
                                              jnp.tile(delta_x[state.enemy_support_caller_idx], MAX_SUPPORTERS),
                                              jnp.zeros(MAX_SUPPORTERS))
                delta_y_support = jnp.where(supporter_active & support_caller_active,
                                              jnp.tile(delta_y[state.enemy_support_caller_idx], MAX_SUPPORTERS),
                                              jnp.zeros(MAX_SUPPORTERS))




                return state._replace(
                    enemy_attack_x=curr_x + delta_x,
                    enemy_attack_y=curr_y + delta_y,
                    enemy_support_x=curr_support_x + delta_x_support,
                    enemy_support_y=curr_support_y + delta_y_support

                 )

            state = update_enemy_state(state)
            state = update_enemy_position(state)
            return state

        # def green_dive(state: GalaxianState, i: int) -> GalaxianState:
        #     jax.debug.print("green")
        #     return dive(state, 0, i)
        #
        # def red_dive(state: GalaxianState, i: int) -> GalaxianState:
        #     jax.debug.print("red")
        #     return dive(state,0, i)
        #
        # def purple_dive(state: GalaxianState, i: int) -> GalaxianState:
        #     jax.debug.print("purple")
        #     return dive(state,1, i)
        #
        # def white_dive(state: GalaxianState, i: int) -> GalaxianState:
        #     jax.debug.print("white")
        #     return dive(state,0, i)

        new_pause_step = jnp.where(
            state.enemy_attack_pause_step < 7,
            state.enemy_attack_pause_step + 1,
        0
        )

        state = state._replace(
            enemy_attack_pause_step= new_pause_step,
        )

        return jax.lax.cond(
            ATTACK_PAUSE_PATTERN[new_pause_step] == 1,
            lambda state: dive(state),
            lambda state: state,
            state
        )

    @jax.jit
    def respawn_finished_dives(state: GalaxianState) -> GalaxianState:
        def body(i, new_state):

            #diver unter dem player und außerhalb window werden auf respawn gesetzt
            respawn_condition = jnp.logical_and(jnp.logical_or(new_state.enemy_attack_y[i] > DIVE_KILL_Y,
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

            is_caller = state.enemy_support_caller_idx == i

            #jax.debug.print("enemy_support_caller_idx: {}", state.enemy_support_caller_idx)
            # beende respawn
            new_state = jax.lax.cond(
                (new_state.enemy_attack_states[i] == 2) &
                (new_state.enemy_attack_x[i] == new_state.enemy_grid_x[
                    new_state.enemy_attack_pos[i, 0], new_state.enemy_attack_pos[i, 1]]) &
                (new_state.enemy_attack_y[i] == new_state.enemy_grid_y[
                    new_state.enemy_attack_pos[i, 0], new_state.enemy_attack_pos[i, 1]]),
                lambda state: jax.lax.cond(
                    is_caller,
                    lambda state: state._replace(
                        enemy_support_caller_idx=jnp.array(-1, dtype=jnp.int32),
                        enemy_attack_states=state.enemy_attack_states.at[i].set(0),
                        enemy_grid_alive=state.enemy_grid_alive.at[tuple(state.enemy_attack_pos[i])].set(1)
                    ),
                    lambda state: state._replace(
                        enemy_attack_states=state.enemy_attack_states.at[i].set(0),
                        enemy_grid_alive=state.enemy_grid_alive.at[tuple(state.enemy_attack_pos[i])].set(1)
                    ),
                    new_state
                ),
                lambda state: state,
                new_state
            )

            return new_state

        def body_support(i, new_state):

            #supporter unter dem player und außerhalb window werden auf respawn gesetzt
            respawn_condition = jnp.logical_and(jnp.logical_or(new_state.enemy_support_y[i] > DIVE_KILL_Y,
                                               jnp.logical_or(new_state.enemy_support_x[i] < ENEMY_LEFT_BOUND,
                                                              new_state.enemy_support_x[i] > ENEMY_RIGHT_BOUND)),
                                                  new_state.enemy_support_states[i] == 1)

            new_state = jax.lax.cond(
                respawn_condition,
                lambda state: state._replace(
                    enemy_support_states=state.enemy_support_states.at[i].set(2),
                    enemy_support_x=state.enemy_support_x.at[i].set(
                        state.enemy_grid_x[state.enemy_support_pos[i, 0], state.enemy_support_pos[i, 1]]
                    ),
                    enemy_support_y=state.enemy_support_y.at[i].set(-10)
                ),
                lambda state: state,
                new_state
            )

            #continue respawnende diver
            new_state = jax.lax.cond(
                new_state.enemy_support_states[i] == 2,
                lambda state: state._replace(
                    enemy_support_y=state.enemy_support_y.at[i].set(
                        lax.clamp(
                            jnp.array(-10, dtype=state.enemy_support_y.dtype),
                            (state.enemy_support_y[i] + 1).astype(state.enemy_support_y.dtype),
                            state.enemy_grid_y[state.enemy_support_pos[i, 0], state.enemy_support_pos[i, 1]],
                        )
                    ),
                    enemy_support_x=state.enemy_support_x.at[i].set(
                        state.enemy_grid_x[state.enemy_support_pos[i, 0], state.enemy_support_pos[i, 1]]
                    )
                ),
                lambda state: state,
                new_state
            )

            #jax.debug.print("enemy_support_states: {}", state.enemy_support_states)
            # beende respawn
            new_state = jax.lax.cond(
                (new_state.enemy_support_states[i] == 2) &
                (new_state.enemy_support_x[i] == new_state.enemy_grid_x[
                    new_state.enemy_support_pos[i, 0], new_state.enemy_support_pos[i, 1]]) &
                (new_state.enemy_support_y[i] == new_state.enemy_grid_y[
                    new_state.enemy_support_pos[i, 0], new_state.enemy_support_pos[i, 1]]),
                lambda state: state._replace(
                    enemy_support_states=state.enemy_support_states.at[i].set(0),
                    enemy_grid_alive=state.enemy_grid_alive.at[tuple(state.enemy_support_pos[i])].set(1)
                ),
                lambda state: state,
                new_state
            )
            return new_state


        state = lax.fori_loop(0, MAX_DIVERS, body, state)
        return lax.fori_loop(0, MAX_SUPPORTERS, body_support, state)


    new_state = respawn_finished_dives(state)
    new_state = test_for_new_dive(new_state)
    new_state = continue_active_dives(new_state)

    return new_state


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
            is_shooting_action & state.player_alive,
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
        new_death = s.enemy_death_frame_grid.at[hit_rows, hit_cols].set(1)

        new_alive = s.enemy_grid_alive.at[hit_rows, hit_cols].set(0)
        return s._replace(
            enemy_grid_alive=new_alive,
            enemy_death_frame_grid=new_death,
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
def update_enemy_death_frames(state: GalaxianState) -> GalaxianState:
    def advance_cell(frame):
        # if in 1..5, increment or then clear
        return jnp.where(frame == 0, 0,
                         jnp.where(frame < 5, frame + 1, 0))
    new_frames_grid = jax.vmap(jax.vmap(advance_cell))(state.enemy_death_frame_grid)
    new_frames_attack = jax.vmap(advance_cell)(state.enemy_death_frame_attack)
    new_frames_support = jax.vmap(advance_cell)(state.enemy_death_frame_support)
    # once frame wraps to 0, also mark the cell fully dead
    cleared_mask_grid = (state.enemy_death_frame_grid == 5)
    new_alive = jnp.where(cleared_mask_grid, 0, state.enemy_grid_alive)
    new_attackers = jnp.where(
        state.enemy_death_frame_attack == 5,
        jnp.where(
            jnp.arange(state.enemy_attack_states.shape[0]) == state.enemy_support_caller_idx,
            3,
            0
        ),
        state.enemy_attack_states
    )
    new_supporters = jnp.where(
        state.enemy_death_frame_support == 5,
        0,
        state.enemy_support_states
    )
    return state._replace(
        enemy_death_frame_grid=new_frames_grid,
        enemy_death_frame_attack=new_frames_attack,
        enemy_death_frame_support=new_frames_support,
        enemy_grid_alive=new_alive,
        enemy_attack_states=new_attackers,
        enemy_support_states=new_supporters,
    )

def update_player_death_frames(state: GalaxianState) -> GalaxianState:
    return state

@jax.jit
def bullet_collision_attack(state: GalaxianState) -> GalaxianState:
    # Abstände
    x_diff = jnp.abs(state.bullet_x - state.enemy_attack_x)
    y_diff = jnp.abs(state.bullet_y - state.enemy_attack_y)
    # Kollisionsmaske: innerhalb 10px und aktuell angreifend (state 1)
    mask = (x_diff <= 10) & (y_diff <= 10) & (state.enemy_attack_states == 1)
    hit = jnp.any(mask) & state.player_alive

    def process_hit(state: GalaxianState) -> GalaxianState:
        hit_indices = jnp.where(mask, size=MAX_DIVERS, fill_value=-1)[0]
        hit_idx = hit_indices[0]
        pos = state.enemy_attack_pos[hit_idx]
        new_grid = state.enemy_grid_alive.at[tuple(pos)].set(0)
        new_death = state.enemy_death_frame_attack.at[hit_idx].set(1)

        new_attack_states = state.enemy_attack_states.at[hit_idx].set(4)

        return state._replace(
            enemy_grid_alive=new_grid,
            enemy_death_frame_attack=new_death,
            bullet_x=jnp.array(-1.0, dtype=state.bullet_x.dtype),
            bullet_y=jnp.array(-1.0, dtype=state.bullet_y.dtype),
            enemy_attack_states=new_attack_states,
            score=state.score + jnp.array(50, dtype=state.score.dtype)
        )

    def no_hit(state: GalaxianState) -> GalaxianState:
        return state

    return lax.cond(hit, process_hit, no_hit, state)

def bullet_collision_support(state: GalaxianState) -> GalaxianState:
    # Abstände
    x_diff = jnp.abs(state.bullet_x - state.enemy_support_x)
    y_diff = jnp.abs(state.bullet_y - state.enemy_support_y)
    # Kollisionsmaske: innerhalb 10px und aktuell angreifend (state 1)
    mask = (x_diff <= 10) & (y_diff <= 10) & (state.enemy_support_states == 1)
    hit = jnp.any(mask) & state.player_alive

    def process_hit(state: GalaxianState) -> GalaxianState:
        hit_indices = jnp.where(mask, size=MAX_DIVERS, fill_value=-1)[0]
        hit_idx = hit_indices[0]
        pos = state.enemy_support_pos[hit_idx]
        new_grid = state.enemy_grid_alive.at[tuple(pos)].set(0)
        new_death = state.enemy_death_frame_support.at[hit_idx].set(1)

        new_support_states = state.enemy_support_states.at[hit_idx].set(4)

        return state._replace(
            enemy_grid_alive=new_grid,
            enemy_death_frame_support=new_death,
            bullet_x=jnp.array(-1.0, dtype=state.bullet_x.dtype),
            bullet_y=jnp.array(-1.0, dtype=state.bullet_y.dtype),
            enemy_support_states=new_support_states,
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
    hit = jnp.any(collision) & state.player_alive

    def process_hit(current_state):
        hit_indices = jnp.where(collision, size=MAX_DIVERS, fill_value=-1)[0]
        hit_idx = hit_indices[0]
        pos = current_state.enemy_attack_pos[hit_idx]
        new_enemy_grid_alive = current_state.enemy_grid_alive.at[tuple(pos)].set(0)
        new_death = state.enemy_death_frame_attack.at[hit_idx].set(1)
        new_attack_states = state.enemy_attack_states.at[hit_idx].set(4)


        return current_state._replace(
            player_alive = jnp.array(False),
            enemy_grid_alive=new_enemy_grid_alive,
            enemy_attack_states=new_attack_states,
            enemy_death_frame_attack=new_death,
        )

    return lax.cond(hit, process_hit, lambda s: s, state)

@jax.jit
def check_player_death_by_support(state: GalaxianState) -> GalaxianState:
    x_diff = jnp.abs(state.player_x - state.enemy_support_x)
    y_diff = jnp.abs(state.player_y - state.enemy_support_y)
    is_active = (state.enemy_support_states == 1)
    grid_alive = state.enemy_grid_alive[tuple(state.enemy_support_pos.T)] != 0

    collision = (x_diff <= 10) & (y_diff <= 10) & is_active & grid_alive
    hit = jnp.any(collision) & state.player_alive

    def process_hit(current_state):
        hit_indices = jnp.where(collision, size=MAX_DIVERS, fill_value=-1)[0]
        hit_idx = hit_indices[0]
        pos = current_state.enemy_support_pos[hit_idx]
        new_enemy_grid_alive = current_state.enemy_grid_alive.at[tuple(pos)].set(0)
        new_support_states = current_state.enemy_support_states.at[hit_idx].set(4)
        new_death = current_state.enemy_death_frame_support.at[hit_idx].set(1)

        return current_state._replace(
            player_alive= jnp.array(False),
            enemy_grid_alive=new_enemy_grid_alive,
            enemy_support_states=new_support_states,
            enemy_death_frame_support=new_death,
        )

    return lax.cond(hit, process_hit, lambda s: s, state)

@jax.jit
def check_player_death_by_bullet(state: GalaxianState) -> GalaxianState:
    x_diff = jnp.abs(state.player_x - state.enemy_attack_bullet_x)
    y_diff = jnp.abs(state.player_y - state.enemy_attack_bullet_y)

    collision_mask = (x_diff <= 10) & (y_diff <= 10) & (state.enemy_attack_bullet_y >= 0)
    hit = jnp.any(collision_mask) & state.player_alive

    def process_hit(current_state):
        # reset bullets
        hit_indices = jnp.where(collision_mask, size=MAX_DIVERS, fill_value=-1)[0]  # size müsste man irgendwann vllt anpassen
        new_bullet_x = current_state.enemy_attack_bullet_x.at[hit_indices].set(-1.0)
        new_bullet_y = current_state.enemy_attack_bullet_y.at[hit_indices].set(-1.0)

        return current_state._replace(
            player_alive=jnp.array(False),
            enemy_attack_bullet_x=new_bullet_x,
            enemy_attack_bullet_y=new_bullet_y
        )

    return lax.cond(hit, process_hit, lambda s: s, state)

def respawn_player(state: GalaxianState) -> GalaxianState:
    return state._replace(
        player_alive=jnp.array(True),
        lives = state.lives -1)

@jax.jit
def enter_new_wave(state: GalaxianState) -> GalaxianState:
    new_grid = ENEMY_GRID
    new_level = state.level + 1
    new_attack_bullet_cd = jnp.array(state.enemy_bullet_max_cooldown * 0.9, dtype=state.enemy_bullet_max_cooldown.dtype)
    return state._replace(enemy_grid_alive=new_grid,
                          level=new_level,)


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
        self.frameskip = frameskip + 1  # den stuff kp hab copy paste aus pong
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
                              enemy_grid_alive=TEST_GRID,
                              enemy_death_frame_grid=jnp.zeros((GRID_ROWS, GRID_COLS), dtype=jnp.int32),
                              enemy_death_frame_attack=jnp.zeros(MAX_DIVERS, dtype=jnp.int32),
                              enemy_death_frame_support=jnp.zeros(MAX_SUPPORTERS, dtype=jnp.int32),
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
                              enemy_attack_number=jnp.array(0),
                              enemy_attack_max=jnp.array(1),
                              enemy_support_caller_idx=jnp.array(-1, dtype=jnp.int32),
                              enemy_support_states=jnp.zeros(MAX_SUPPORTERS, dtype=jnp.int32),
                              enemy_support_pos=jnp.full((MAX_SUPPORTERS, 2), -1, dtype=jnp.int32),
                              enemy_support_x=jnp.zeros(MAX_SUPPORTERS),
                              enemy_support_y=jnp.zeros(MAX_SUPPORTERS),
                              level=jnp.array(0),
                              lives=jnp.array(LIVES),
                              player_alive=jnp.array(True),
                              score=jnp.array(0, dtype=jnp.int32),
                              enemy_attack_target_x=jnp.zeros(MAX_DIVERS),
                              enemy_attack_target_y=jnp.zeros(MAX_DIVERS),
                              turn_step=jnp.array(0),
                              dive_probability=jnp.array(BASE_DIVE_PROBABILITY),
                              enemy_bullet_max_cooldown=jnp.array(ENEMY_ATTACK_BULLET_DELAY),
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
        jax.debug.print("grid {}",state.enemy_grid_alive)
        jax.debug.print("attackers {}", state.enemy_attack_states)
        jax.debug.print("supporters {}", state.enemy_support_pos)
        jax.debug.print("support pos {}", state.enemy_support_x)
        new_state = update_player_position(state, action)
        new_state = update_player_bullet(new_state, action)
        new_state = update_enemy_positions(new_state)
        new_state = bullet_collision(new_state)
        new_state = bullet_collision_attack(new_state)
        new_state = bullet_collision_support(new_state)
        new_state = update_enemy_death_frames(new_state)
        new_state = update_enemy_attack(new_state)
        new_state = update_enemy_bullets(new_state)
        new_state = check_player_death_by_enemy(new_state)
        new_state = check_player_death_by_support(new_state)
        new_state = check_player_death_by_bullet(new_state)
        new_state = new_state._replace(turn_step=new_state.turn_step + 1)

        new_state = jax.lax.cond(jnp.logical_and(jnp.logical_not(jnp.any(state.enemy_grid_alive == 2)),state.player_alive==False),lambda new_state: respawn_player(new_state), lambda s: s, new_state)
        new_state = jax.lax.cond(jnp.logical_and(jnp.logical_not(jnp.any(state.enemy_grid_alive == 1)), jnp.logical_not(jnp.any(state.enemy_attack_states != 0))), lambda new_state: enter_new_wave(new_state), lambda s: s, new_state)

        done = self._get_done(new_state)
        env_reward = self._get_env_reward(state, new_state)
        all_rewards = self._get_all_reward(state, new_state)
        info = self._get_info(new_state, all_rewards)

        observation = self._get_observation(new_state)

        def do_reset(_):
            obs, s = self.reset()
            return obs, s, 0, False, self._get_info(s, jnp.zeros(1))

        return jax.lax.cond(
            done,
            do_reset,
            lambda _: (observation, new_state, env_reward, done, info),
            operand=None
        )


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
        return state.lives < 0

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
    enemy_attacking_facing_down_green = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/galaxian/enemy_attacking_facing_down_green.npy"),transpose=True)
    enemy_attacking_facing_down_purple = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/galaxian/enemy_attacking_facing_down_purple.npy"),transpose=True)
    enemy_attacking_facing_down_red = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/galaxian/enemy_attacking_facing_down_red.npy"),transpose=True)
    enemy_attacking_facing_down_white = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/galaxian/enemy_attacking_facing_down_white.npy"),transpose=True)
    enemy_attacking_facing_left_red = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/galaxian/enemy_attacking_facing_left_red.npy"),transpose=True)
    enemy_attacking_facing_right_red = enemy_attacking_facing_left_red[::-1,:]
    enemy_attacking_facing_left_white = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/galaxian/enemy_attacking_facing_left_white.npy"),transpose=True)
    enemy_attacking_facing_right_white = enemy_attacking_facing_left_white[::-1,:]
    enemy_attacking_facing_right_green = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/galaxian/enemy_attacking_facing_right_green.npy"),transpose=True)
    enemy_attacking_facing_left_green = enemy_attacking_facing_right_green[::-1,:]
    enemy_attacking_facing_right_purple = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/galaxian/enemy_attacking_facing_right_purple.npy"),transpose=True)
    enemy_attacking_facing_left_purple = enemy_attacking_facing_right_purple[::-1,:]

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
    SPRITE_ENEMY = jnp.stack([enemy_gray]*3+[enemy_purple,enemy_red,enemy_white], axis=0)
    SPRITE_ENEMY_DOWN = jnp.stack([enemy_attacking_facing_down_green]*3+[enemy_attacking_facing_down_purple,enemy_attacking_facing_down_red,enemy_attacking_facing_down_white])
    SPRITE_ENEMY_LEFT = jnp.stack([enemy_attacking_facing_left_green]*3+[enemy_attacking_facing_left_purple,enemy_attacking_facing_left_red,enemy_attacking_facing_left_white])
    SPRITE_ENEMY_RIGHT = jnp.stack([enemy_attacking_facing_right_green]*3+[enemy_attacking_facing_right_purple,enemy_attacking_facing_right_red,enemy_attacking_facing_right_white])
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
        SPRITE_ENEMY,
        SPRITE_ENEMY_DOWN,
        SPRITE_ENEMY_LEFT,
        SPRITE_ENEMY_RIGHT,
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
            self.SPRITE_ENEMY,
            self.SPRITE_ENEMY_DOWN,
            self.SPRITE_ENEMY_LEFT,
            self.SPRITE_ENEMY_RIGHT,
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
        raster = jnp.where(
            state.player_alive == True,
            aj.render_at(raster, px, py, player_frame),
            raster
        )

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
        raster = lax.cond(
            state.bullet_y > 0,
            draw_bullet_active,
            lambda r: lax.cond(
                state.player_alive,
                draw_bullet_inactive,
                lambda r: r,
                raster
            ),
            raster)

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

            def draw_single_attacker(r, i):
                is_alive = jnp.logical_or(state.enemy_attack_states[i] == 1, state.enemy_attack_states[i] == 2)
                is_dying = state.enemy_attack_states[i] == 4
                death_frame_attack = state.enemy_death_frame_attack[i].astype(jnp.int32)

                row = state.enemy_attack_pos[i][0]
                direction = jnp.where(
                    state.enemy_attack_turning[i] == 0,
                    jnp.where(
                        state.enemy_attack_direction[i] == -1,
                        0,
                        1,
                    ),
                    2
                )
                is_down = direction == 2

                def down(r):
                    ex = jnp.round(state.enemy_attack_x[i]).astype(jnp.int32)
                    ey = jnp.round(state.enemy_attack_y[i]).astype(jnp.int32)
                    sprite = get_sprite_frame(self.SPRITE_ENEMY_DOWN, row)

                    return aj.render_at(r, ex, ey, sprite)

                def side(r):
                    ex = jnp.round(state.enemy_attack_x[i]).astype(jnp.int32)
                    ey = jnp.round(state.enemy_attack_y[i]).astype(jnp.int32)
                    sprite = jnp.where(
                        direction == 1,
                        get_sprite_frame(self.SPRITE_ENEMY_RIGHT,row),
                        get_sprite_frame(self.SPRITE_ENEMY_LEFT,row),
                    )

                    return aj.render_at(r, ex, ey, sprite)

                def dying(r):
                    ex = jnp.round(state.enemy_attack_x[i]).astype(jnp.int32)
                    ey = jnp.round(state.enemy_attack_y[i]).astype(jnp.int32)
                    sprite = get_sprite_frame(self.SPRITE_ENEMY_DEATH, death_frame_attack - 1)
                    return aj.render_at(r, ex, ey, sprite)

                def dead(r):
                    return r

                return lax.cond(is_alive,lambda r: lax.cond(is_down,down,side,r) ,lambda r: lax.cond(is_dying,dying,dead,r), r)

            for i in range(MAX_DIVERS):
                r = draw_single_attacker(r, i)

            return r
        raster = lax.cond(jnp.any(state.enemy_attack_states != 0), draw_attackers, lambda r: r, raster)

        def draw_supporters(r):

            def draw_single_supporter(r, i):
                is_alive = jnp.logical_or(state.enemy_support_states[i] == 1, state.enemy_support_states[i] == 2)
                is_dying = state.enemy_support_states[i] == 4
                death_frame_support = state.enemy_death_frame_support[i].astype(jnp.int32)

                caller_idx = state.enemy_support_caller_idx
                direction = jnp.where(
                    state.enemy_attack_turning[caller_idx] == 0,
                    jnp.where(
                        state.enemy_attack_direction[caller_idx] == -1,
                        0,
                        1,
                    ),
                    2
                )
                is_down = direction == 2

                def down(r):
                    ex = jnp.round(state.enemy_support_x[i]).astype(jnp.int32)
                    ey = jnp.round(state.enemy_support_y[i]).astype(jnp.int32)
                    sprite = get_sprite_frame(self.SPRITE_ENEMY_DOWN, 4)

                    return aj.render_at(r, ex, ey, sprite)

                def side(r):
                    ex = jnp.round(state.enemy_support_x[i]).astype(jnp.int32)
                    ey = jnp.round(state.enemy_support_y[i]).astype(jnp.int32)
                    sprite = jnp.where(
                        direction == 1,
                        get_sprite_frame(self.SPRITE_ENEMY_RIGHT,4),
                        get_sprite_frame(self.SPRITE_ENEMY_LEFT,4),
                    )

                    return aj.render_at(r, ex, ey, sprite)

                def dying(r):
                    ex = jnp.round(state.enemy_support_x[i]).astype(jnp.int32)
                    ey = jnp.round(state.enemy_support_y[i]).astype(jnp.int32)
                    sprite = get_sprite_frame(self.SPRITE_ENEMY_DEATH,death_frame_support-1)
                    return aj.render_at(r, ex, ey, sprite)

                def dead(r):
                    return r

                return lax.cond(is_alive,lambda r: lax.cond(is_down,down,side,r) ,lambda r: lax.cond(is_dying,dying,dead,r), r)

            for i in range(MAX_SUPPORTERS):
                    r = draw_single_supporter(r, i)

            return r
        raster = lax.cond(state.enemy_support_caller_idx != -1, draw_supporters, lambda r: r, raster)

       # Feindgitter
        def row_body(i, r_acc):
            def col_body(j, r_inner):
                death_frame_grid = state.enemy_death_frame_grid[i, j].astype(jnp.int32)
                alive = (state.enemy_grid_alive[i, j] == 1)


                def draw_death(r0):
                    sprite = get_sprite_frame(self.SPRITE_ENEMY_DEATH, death_frame_grid - 1)
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
                    death_frame_grid > 0,
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
                    NATIVE_GAME_HEIGHT - self.life_h - self.life_spacing -10
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




