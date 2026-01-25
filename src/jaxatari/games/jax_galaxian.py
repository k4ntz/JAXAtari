import os
from typing import NamedTuple, Tuple, Optional
import jax.numpy as jnp
import chex
import pygame
from functools import partial
from jax import lax
import jax.lax
import jax
from flax import struct

import jaxatari.spaces as spaces

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import JAXGameRenderer
import jaxatari.rendering.jax_rendering_utils as render_utils
from jaxatari.modification import AutoDerivedConstants

"""
README
Aaron Reinhardt
Aaron Weis
Leon Denis Kristof
"""


def _get_default_asset_config() -> tuple:
    """
    Returns the default declarative asset manifest for Galaxian.
    """
    
    # Base sprites (single images)
    asset_list = [
        {'name': 'background', 'type': 'background', 'file': 'background.npy'},
        {'name': 'player', 'type': 'single', 'file': 'player.npy'},
        {'name': 'bullet', 'type': 'single', 'file': 'bullet.npy'},
        {'name': 'life', 'type': 'single', 'file': 'life.npy'},
        {'name': 'enemy_bullet', 'type': 'single', 'file': 'enemy_bullet.npy'},
    ]
    
    # Enemy sprites (for grid)
    asset_list.extend([
        {'name': 'enemy_grid_gray', 'type': 'single', 'file': 'gray_enemy_1.npy'},
        {'name': 'enemy_grid_purple', 'type': 'single', 'file': 'purple_blue_enemy_1.npy'},
        {'name': 'enemy_grid_red', 'type': 'single', 'file': 'red_orange_enemy_1.npy'},
        {'name': 'enemy_grid_white', 'type': 'single', 'file': 'white_enemy_1.npy'},
    ])
    # Enemy dive sprites (down)
    asset_list.extend([
        {'name': 'enemy_dive_down_green', 'type': 'single', 'file': 'enemy_attacking_facing_down_green.npy'},
        {'name': 'enemy_dive_down_purple', 'type': 'single', 'file': 'enemy_attacking_facing_down_purple.npy'},
        {'name': 'enemy_dive_down_red', 'type': 'single', 'file': 'enemy_attacking_facing_down_red.npy'},
        {'name': 'enemy_dive_down_white', 'type': 'single', 'file': 'enemy_attacking_facing_down_white.npy'},
    ])
    
    # Enemy dive sprites (left) - right is just a horizontal flip
    asset_list.extend([
        {'name': 'enemy_dive_left_green', 'type': 'single', 'file': 'enemy_attacking_facing_right_green.npy'}, # Note: old code had a mix-up, this seems to be the intended 'left'
        {'name': 'enemy_dive_left_purple', 'type': 'single', 'file': 'enemy_attacking_facing_right_purple.npy'},# Note: old code flipped this
        {'name': 'enemy_dive_left_red', 'type': 'single', 'file': 'enemy_attacking_facing_left_red.npy'},
        {'name': 'enemy_dive_left_white', 'type': 'single', 'file': 'enemy_attacking_facing_left_white.npy'},
    ])
    # Death animation (group for auto-padding)
    asset_list.append({
        'name': 'enemy_death', 'type': 'group', 'files': [
            'death_enemy_1.npy',
            'death_enemy_2.npy',
            'death_enemy_3.npy',
            'death_enemy_4.npy',
            'death_enemy_5.npy',
        ]
    })
    
    # Digits (for score)
    asset_list.append({
        'name': 'digits', 'type': 'digits', 'pattern': 'score_{}.npy'
    })
    
    return tuple(asset_list)


class GalaxianConstants(AutoDerivedConstants):
# -------- Game constants --------
# -------- Window constants --------
    NATIVE_GAME_WIDTH: int = struct.field(pytree_node=False, default=160)
    NATIVE_GAME_HEIGHT: int = struct.field(pytree_node=False, default=210)
    PYGAME_SCALE_FACTOR: int = struct.field(pytree_node=False, default=3)
   
    PYGAME_WINDOW_WIDTH: Optional[int] = struct.field(pytree_node=False, default=None)
    PYGAME_WINDOW_HEIGHT: Optional[int] = struct.field(pytree_node=False, default=None)
    START_X: Optional[int] = struct.field(pytree_node=False, default=None)
    START_Y: Optional[int] = struct.field(pytree_node=False, default=None)
  
    # -------- Player constants --------
    BULLET_MOVE_SPEED: int = struct.field(pytree_node=False, default=5)
    LIVES: int = struct.field(pytree_node=False, default=2)
    EXTRA_LIFE_SCORE: int = struct.field(pytree_node=False, default=7000)
    PLAYER_RESPAWN_TIME: int = struct.field(pytree_node=False, default=16)
    PLAYER_BULLET_Y_OFFSET: int = struct.field(pytree_node=False, default=3)
    PLAYER_BULLET_X_OFFSET: int = struct.field(pytree_node=False, default=3)
    # -------- Grid constants --------
    GRID_ROWS: int = struct.field(pytree_node=False, default=6)
    GRID_COLS: int = struct.field(pytree_node=False, default=7)
    ENEMY_SPACING_X: int = struct.field(pytree_node=False, default=16)
    ENEMY_SPACING_Y: int = struct.field(pytree_node=False, default=11)
    ENEMY_GRID_Y: int = struct.field(pytree_node=False, default=80)
    ENEMY_MOVE_FRAMES: int = struct.field(pytree_node=False, default=16)
    PURPLE_ROW: int = struct.field(pytree_node=False, default=3)
    RED_ROW: int = struct.field(pytree_node=False, default=4)
    WHITE_ROW: int = struct.field(pytree_node=False, default=5)
    # -------- Enemy constants --------
    ENEMY_ATTACK_SPEED: int = struct.field(pytree_node=False, default=2)
    ENEMY_ATTACK_TURN_TIME: int = struct.field(pytree_node=False, default=32)
    ENEMY_ATTACK_BULLET_SPEED: int = struct.field(pytree_node=False, default=3)
    ENEMY_ATTACK_BULLET_DELAY: int = struct.field(pytree_node=False, default=75)
    ENEMY_ATTACK_MAX_BULLETS: int = struct.field(pytree_node=False, default=2)
    ENEMY_LEFT_BOUND: int = struct.field(pytree_node=False, default=17)
    ENEMY_RIGHT_BOUND: Optional[int] = struct.field(pytree_node=False, default=None)

    DIVE_KILL_Y: int = struct.field(pytree_node=False, default=175)
    DIVE_SPEED: int = struct.field(pytree_node=False, default=0.5)
    DIRECTION_CHANGE_RANGE: int = struct.field(pytree_node=False, default=10)
    MAX_DIVERS: int = struct.field(pytree_node=False, default=5)
    MAX_SUPPORT_CALLERS: int = struct.field(pytree_node=False, default=2)
    MAX_SUPPORTERS: int = struct.field(pytree_node=False, default=2)
    VOLLEY_SHOT_DELAY: int = struct.field(pytree_node=False, default=10)
    VOLLEY_PROBABILITIES: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([0.4, 0.3, 0.2, 0.1]))
    # -------- Sprite sizes --------
    ENEMY_HEIGHT: int = struct.field(pytree_node=False, default=9)
    ENEMY_WIDTH: int = struct.field(pytree_node=False, default=6)
    ENEMY_ATTACK_HEIGHT: int = struct.field(pytree_node=False, default=10)
    ENEMY_ATTACK_WIDTH: int = struct.field(pytree_node=False, default=8)
    PLAYER_HEIGHT: int = struct.field(pytree_node=False, default=14)
    PLAYER_WIDTH: int = struct.field(pytree_node=False, default=8)
    # -------- SCORE constants --------
    SCORES: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([30]*3+[40,50,60]))
    DIVE_MULTIPLIER: int = struct.field(pytree_node=False, default=2)
    # -------- Value constants --------
    ERROR_VALUE: int = struct.field(pytree_node=False, default=-9999)
    # --- Grid states ---
    DEAD: int = struct.field(pytree_node=False, default=0)
    GRID: int = struct.field(pytree_node=False, default=1)
    ACTIVE: int = struct.field(pytree_node=False, default=2)
    # --- Attack states ---
    EMPTY: int = struct.field(pytree_node=False, default=0)
    ATTACK: int = struct.field(pytree_node=False, default=1)
    SUPPORT: int = struct.field(pytree_node=False, default=2)
    RESPAWN: int = struct.field(pytree_node=False, default=2)
    DEAD_CALLER: int = struct.field(pytree_node=False, default=3)
    DYING: int = struct.field(pytree_node=False, default=4)
    # --- Directions ---
    LEFT: int = struct.field(pytree_node=False, default=-1)
    RIGHT: int = struct.field(pytree_node=False, default=1)
    TURNING_LEFT: int = struct.field(pytree_node=False, default=-1)
    TURNING_RIGHT: int = struct.field(pytree_node=False, default=1)
    NO_TURNING: int = struct.field(pytree_node=False, default=0)
    # -------- Pattern constants --------
    ENEMY_GRID: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([
        [1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1],
        [0,1,1,1,1,1,0],
        [0,0,1,0,1,0,0],
    ]).astype(jnp.float32))

    ATTACK_MOVE_PATTERN: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([
        [
            [1,1],[0,1],[1,1],[1,1],[0,1],[1,1],[1,1],[0,1],[1,1],[1,1],[1,1],[0,1],[1,1],[1,1],[0,1]
        ],
        [
            [2,1],[2,1],[2,1],[2,1],[2,1],[2,1],[2,1],[2,1],[2,1],[2,1],[2,1],[2,1],[2,1],[2,1],[2,1]
        ]
    ]))

    ATTACK_TURN_PATTERN: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([
        [
            [-1,1],[0,1],[-1,1],[0,1],[0,1],[-1,1],[0,1],[0,1],[-1,1],[0,1],[0,1],[0,1],[-1,1],[0,1],[0,1],[0,1],
            [0,1],[0,1],[0,1],[1,1],[0,1],[0,1],[0,1],[1,1],[0,1],[0,1],[1,1],[0,1],[0,1],[1,1],[0,1],[1,1]
        ],
        [
            [-1,1],[-1,1],[-2,1],[-1,1],[-1,1],[-1,1],[-1,1],[-1,1],[-1,1],[-1,1],[0,1],[-1,1],[0,1],[0,1],[0,1],[0,1],
            [0,1],[0,1],[0,1],[0,1],[1,1],[0,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[2,1],[1,1],[1,1]
        ]
    ]))

    ATTACK_PAUSE_PATTERN: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([
        1,0,1,0,1,1,1,0
    ]))
    
    # Asset config baked into constants
    ASSET_CONFIG: tuple = struct.field(pytree_node=False, default_factory=lambda: _get_default_asset_config())
    
    def compute_derived(self):
        """Compute derived constants based on static fields."""
        return {
            'PYGAME_WINDOW_WIDTH': self.NATIVE_GAME_WIDTH * self.PYGAME_SCALE_FACTOR,
            'PYGAME_WINDOW_HEIGHT': self.NATIVE_GAME_HEIGHT * self.PYGAME_SCALE_FACTOR,
            'START_X': self.NATIVE_GAME_WIDTH // 4,
            'START_Y': self.NATIVE_GAME_HEIGHT,
            'ENEMY_RIGHT_BOUND': self.NATIVE_GAME_WIDTH - 25,
        }

@struct.dataclass
class GalaxianState:
    player_x: chex.Array
    player_y: chex.Array
    player_respawn_timer: chex.Array
    bullet_x: chex.Array
    bullet_y: chex.Array
    enemy_grid_x: chex.Array
    enemy_grid_y: chex.Array
    enemy_grid_state: chex.Array   # 0: dead, 1: alive, 2: attacking, 3: supporting
    enemy_grid_move_frames: chex.Array
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
    got_extra_life: chex.Array
    player_alive: chex.Array
    score: chex.Array
    turn_step: chex.Array
    enemy_attack_shot_timer: chex.Array
    enemy_attack_shots_fired: chex.Array
    enemy_attack_volley_size: chex.Array

@struct.dataclass
class GalaxianObservation:
    player_x: chex.Array
    player_y: chex.Array
    bullet_x: chex.Array
    bullet_y: chex.Array
    enemy_grid_x: chex.Array
    enemy_grid_y: chex.Array
    enemy_grid_state: chex.Array  # 0: dead, 1: alive, 2: attacking
    enemy_attack_pos: chex.Array
    enemy_attack_x: chex.Array
    enemy_attack_y: chex.Array
    enemy_support_pos: chex.Array
    enemy_support_x: chex.Array
    enemy_support_y: chex.Array
    enemy_attack_bullet_x: chex.Array
    enemy_attack_bullet_y: chex.Array

@struct.dataclass
class GalaxianInfo:
    time: jnp.ndarray
    lives: chex.Array
    score: chex.Array
    level: chex.Array


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


class JaxGalaxian(JaxEnvironment[GalaxianState, GalaxianObservation, GalaxianInfo, GalaxianConstants]):
    # Minimal ALE action set for Galaxian
    ACTION_SET: jnp.ndarray = jnp.array(
        [
            Action.NOOP,
            Action.FIRE,
            Action.RIGHT,
            Action.LEFT,
            Action.RIGHTFIRE,
            Action.LEFTFIRE,
        ],
        dtype=jnp.int32,
    )
    
    def __init__(self, consts: GalaxianConstants = None):
        super().__init__()
        self.consts = consts or GalaxianConstants()
        self.renderer = GalaxianRenderer()


    @partial(jax.jit, static_argnums=(0,))
    def update_player_position(self, state: GalaxianState, action) -> GalaxianState:

        press_right = jnp.any(
            jnp.array([action == Action.RIGHT, action == Action.RIGHTFIRE])
        )

        press_left = jnp.any(
            jnp.array([action == Action.LEFT, action == Action.LEFTFIRE])
        )

        # rohe neue X-Position
        new_x = jnp.where(
            state.player_alive == True,
            state.player_x + (press_right) - (press_left),
            state.player_x
        )

        # clamp zwischen 0 und (SCREEN_WIDTH − PLAYER_WIDTH)
        new_x = jnp.clip(new_x, 17, self.consts.NATIVE_GAME_WIDTH - 25)
        return state.replace(player_x=new_x)

    @partial(jax.jit, static_argnums=(0,))
    def update_enemy_positions(self, state: GalaxianState) -> GalaxianState:

        new_move_frames = jnp.where(
            state.enemy_grid_move_frames < self.consts.ENEMY_MOVE_FRAMES,
            state.enemy_grid_move_frames + 1,
            0,
        )

        new_x = jnp.where(
            new_move_frames == self.consts.ENEMY_MOVE_FRAMES,
            state.enemy_grid_x + 1 * state.enemy_grid_direction,
            state.enemy_grid_x,
        )

        # clamp auf [LEFT, RIGHT]
        new_x = jnp.clip(new_x, self.consts.ENEMY_LEFT_BOUND, self.consts.ENEMY_RIGHT_BOUND)

        # Rand-Bounce wie gehabt
        hit_left = jnp.any(new_x <= self.consts.ENEMY_LEFT_BOUND)
        hit_right = jnp.any(new_x >= self.consts.ENEMY_RIGHT_BOUND)
        new_dir = jnp.where(hit_left, 1,
                            jnp.where(hit_right, -1, state.enemy_grid_direction))

        return state.replace(
            enemy_grid_move_frames=new_move_frames,
            enemy_grid_x=new_x,
            enemy_grid_direction=new_dir
        )

    @partial(jax.jit, static_argnums=(0,))
    def update_enemy_attack(self, state: GalaxianState) -> GalaxianState:
        # random ob gedived wird
        # falls ja, gehe erst zu initialise_new_dive und im anschluss zu continue_active_dives(state)
        # falls nein, gehe direkt zu continue_active_dives(state)

        def test_for_new_dive(state):
            first_available_slot = \
            jnp.where(state.enemy_attack_states == self.consts.EMPTY, size=self.consts.MAX_DIVERS,
                      fill_value=-1)[0][0]

            return jax.lax.cond(
                jnp.any(state.enemy_grid_state == self.consts.GRID) & (first_available_slot <= state.level) & (
                            state.player_alive == True), lambda state: initialise_new_dive(state, first_available_slot),
                lambda state: state, state)

        def choose_row(state, rows, row_number, key_choice):
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
            row_alive_mask = (state.enemy_grid_state[row] == self.consts.GRID)
            alive_cols = jnp.where(row_alive_mask, size=self.consts.GRID_COLS, fill_value=-1)[0]
            first_alive_col = alive_cols[0]
            last_alive_col = alive_cols[jnp.sum(row_alive_mask) - 1]
            chosen_enemy_col = jnp.where(first_or_last == 0, first_alive_col, last_alive_col)

            return row, chosen_enemy_col

        def choose_diver(state, key_choice):
            rows_with_alive = jnp.any(state.enemy_grid_state == self.consts.GRID, axis=1)

            num_valid_lower_rows = jnp.sum(rows_with_alive[:4])
            num_valid_white_rows = jnp.sum(rows_with_alive[-1])

            def lower_rows(state, key_choice):
                return choose_row(state, rows_with_alive[:4], 4, key_choice)

            def white_row(state, key_choice):
                return jax.lax.cond(
                    state.enemy_support_caller_idx == self.consts.ERROR_VALUE,
                    lambda state: choose_column(state, self.consts.WHITE_ROW, key_choice),
                    lambda state: (self.consts.ERROR_VALUE, self.consts.ERROR_VALUE),
                    state)

            def red_row(state, key_choice):
                return choose_column(state, self.consts.RED_ROW, key_choice)

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

        def initialise_new_dive(state, diver_idx):
            key = jax.random.PRNGKey(state.turn_step + 101)  # currently deterministic
            key_choice, key_volley, key_shot_delay = jax.random.split(key, 3)

            chosen_enemy_row, chosen_enemy_col = choose_diver(state, key_choice)

            def initialise(state, diver_idx):
                # e.g. enemy_attack_states=[1. 0. 0. 1. 1.] -> enemy_attack_states=[1. 1. 0. 1. 1.]
                new_attack_states = state.enemy_attack_states.at[diver_idx].set(
                    self.consts.ATTACK)  # 1: actively attacking
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
                new_attack_turning = state.enemy_attack_turning.at[diver_idx].set(self.consts.NO_TURNING)
                new_attack_turn_step = state.enemy_attack_turn_step.at[diver_idx].set(0)

                # timer
                new_attack_respawn_timer = state.enemy_attack_respawn_timer.at[diver_idx].set(0)

                chosen_index = jax.random.categorical(key_volley, jnp.log(self.consts.VOLLEY_PROBABILITIES))
                random_volley_size = chosen_index + 1
                new_volley_size = state.enemy_attack_volley_size.at[diver_idx].set(random_volley_size)

                def initialise_supporters(state):
                    new_enemy_support_caller_idx = diver_idx
                    row, column = choose_column(state, self.consts.WHITE_ROW, key_choice)

                    red_row = state.enemy_grid_state[4]

                    # Finde die indizierten Werte in der Nähe der gewählten Spalte

                    nearby_mask = jnp.zeros_like(red_row, dtype=jnp.bool_)
                    nearby_mask = nearby_mask.at[jnp.clip(column - 1, 0, self.consts.GRID_COLS - 1)].set(True)
                    nearby_mask = nearby_mask.at[column].set(True)
                    nearby_mask = nearby_mask.at[jnp.clip(column + 1, 0, self.consts.GRID_COLS - 1)].set(True)

                    # Kombiniere die Nähe-Maske mit der "ist lebendig"-Maske
                    valid_support_mask = nearby_mask & (red_row == 1)

                    # Finde die Indizes der ersten beiden gültigen Unterstützer, von rechts beginnend
                    sorted_indices = jnp.sort(
                        jnp.where(valid_support_mask, jnp.arange(self.consts.GRID_COLS), -1))[
                                     ::-1]  # Absteigend sortieren
                    support_indices = sorted_indices[:2]  # Nimm die ersten beiden (höchsten) Indizes

                    # Wenn weniger als 2 Unterstützer gefunden wurden, fülle mit -1 auf
                    padded_support_indices = jnp.pad(support_indices, (0, 2 - len(support_indices)),
                                                     constant_values=-1)[:2]

                    new_enemy_support_states = jnp.array([
                        jnp.where(padded_support_indices[i] != -1, 1, 0)  # 1 für aktive Unterstützer, 0 für inaktive
                        for i in range(self.consts.MAX_SUPPORTERS)
                    ], dtype=jnp.int32)

                    new_enemy_support_pos = jnp.array([
                        [jnp.where(padded_support_indices[i] != -1, 4, -1),
                         jnp.where(padded_support_indices[i] != -1, padded_support_indices[i], -1)]
                        for i in range(self.consts.MAX_SUPPORTERS)
                    ], dtype=jnp.int32)

                    new_support_x = jnp.array([
                        jnp.where(new_enemy_support_pos[i, 0] != -1,
                                  state.enemy_grid_x[new_enemy_support_pos[i, 0], new_enemy_support_pos[i, 1]],
                                  -1.0)
                        for i in range(self.consts.MAX_SUPPORTERS)
                    ], dtype=jnp.float32)
                    new_support_y = jnp.array([
                        jnp.where(new_enemy_support_pos[i, 0] != -1,
                                  state.enemy_grid_y[new_enemy_support_pos[i, 0], new_enemy_support_pos[i, 1]],
                                  -1.0)
                        for i in range(self.consts.MAX_SUPPORTERS)
                    ], dtype=jnp.float32)

                    new_enemy_grid = state.enemy_grid_state

                    for i in range(self.consts.MAX_SUPPORTERS):
                        idx = padded_support_indices[i]
                        new_enemy_grid = jax.lax.cond(
                            idx != self.consts.ERROR_VALUE,
                            lambda g: g.at[4, idx].set(self.consts.ACTIVE),
                            lambda g: g,
                            new_enemy_grid
                        )

                    return state.replace(
                        enemy_support_caller_idx=new_enemy_support_caller_idx,
                        enemy_support_states=new_enemy_support_states,
                        enemy_support_pos=new_enemy_support_pos,
                        enemy_support_x=new_support_x,
                        enemy_support_y=new_support_y,
                        enemy_grid_state=new_enemy_grid
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
                new_grid_state = state.enemy_grid_state.at[chosen_enemy_row, chosen_enemy_col].set(
                    self.consts.ACTIVE)

                return state.replace(
                    enemy_attack_states=new_attack_states,
                    enemy_attack_pos=new_attack_pos,
                    enemy_attack_x=new_attack_x,
                    enemy_attack_y=new_attack_y,
                    enemy_grid_state=new_grid_state,
                    enemy_attack_direction=new_enemy_attack_direction,
                    enemy_attack_turning=new_attack_turning,
                    enemy_attack_turn_step=new_attack_turn_step,
                    enemy_attack_respawn_timer=new_attack_respawn_timer,
                    enemy_attack_shots_fired=new_shots_fired,
                    enemy_attack_shot_timer=new_shot_timer,
                    enemy_attack_volley_size=new_volley_size
                )

            return jax.lax.cond(
                chosen_enemy_row > -1,
                lambda state: initialise(state, diver_idx),
                lambda state: state.replace(),
                state)

        def continue_active_dives(state: GalaxianState) -> GalaxianState:
            curr_x = state.enemy_attack_x
            curr_y = state.enemy_attack_y

            def dive(state: GalaxianState) -> GalaxianState:
                def update_enemy_state(state: GalaxianState) -> GalaxianState:
                    new_enemy_attack_direction = jnp.where(
                        (state.enemy_attack_turning != self.consts.NO_TURNING) & (
                                    state.enemy_attack_turn_step == self.consts.ENEMY_ATTACK_TURN_TIME),
                        # change the direction, if the enemy is currently turning and the turn is over
                        state.enemy_attack_turning,
                        state.enemy_attack_direction
                    )

                    player_right = state.enemy_attack_x < state.player_x - self.consts.DIRECTION_CHANGE_RANGE
                    player_left = state.enemy_attack_x > state.player_x + self.consts.DIRECTION_CHANGE_RANGE

                    new_enemy_attack_turning = jnp.where(
                        (state.enemy_attack_turn_step == self.consts.ENEMY_ATTACK_TURN_TIME) | (
                                    state.enemy_attack_states == self.consts.EMPTY) | (
                                state.enemy_attack_states == self.consts.RESPAWN),
                        self.consts.NO_TURNING,  # reset turning, if turn is over or the enemy died
                        jnp.where(
                            (state.enemy_attack_turning == self.consts.NO_TURNING) & (
                                        state.enemy_attack_turn_step == 0) & (
                                    player_right & (
                                        state.enemy_attack_direction == self.consts.LEFT) | player_left & (
                                            state.enemy_attack_direction == self.consts.RIGHT)),
                            -state.enemy_attack_direction,  # change turning to the opposite of the current direction
                            state.enemy_attack_turning)
                    )

                    new_enemy_attack_turn_step = jnp.where(
                        (state.enemy_attack_turn_step == self.consts.ENEMY_ATTACK_TURN_TIME) | (
                                    state.enemy_attack_states == self.consts.EMPTY),
                        0,
                        jnp.where(
                            state.enemy_attack_turning != self.consts.NO_TURNING,
                            state.enemy_attack_turn_step + 1,
                            state.enemy_attack_turn_step
                        )
                    )

                    new_enemy_attack_move_step = jnp.where(
                        (state.enemy_attack_move_step == 15) | (state.enemy_attack_states == self.consts.EMPTY),
                        0,
                        jnp.where(
                            state.enemy_attack_turning == self.consts.NO_TURNING,
                            state.enemy_attack_move_step + 1,
                            state.enemy_attack_move_step
                        )
                    )

                    caller_idx = state.enemy_support_caller_idx

                    # Setze zurück, falls keine Supporter mehr übrig
                    new_enemy_attack_states, new_support_caller_idx = lax.cond(
                        (caller_idx != self.consts.ERROR_VALUE) & (jnp.logical_not(jnp.any(
                            jnp.logical_or(state.enemy_support_states == self.consts.ATTACK,
                                           state.enemy_support_states == self.consts.RESPAWN)))) & (
                                    state.enemy_attack_states[caller_idx] == self.consts.DEAD_CALLER),
                        lambda state: (state.enemy_attack_states.at[caller_idx].set(self.consts.EMPTY),
                                       self.consts.ERROR_VALUE),
                        lambda state: (state.enemy_attack_states, state.enemy_support_caller_idx),
                        state
                    )

                    return state.replace(
                        enemy_attack_direction=new_enemy_attack_direction,
                        enemy_attack_turning=new_enemy_attack_turning,
                        enemy_attack_turn_step=new_enemy_attack_turn_step,
                        enemy_attack_move_step=new_enemy_attack_move_step,
                        enemy_attack_states=new_enemy_attack_states,
                        enemy_support_caller_idx=new_support_caller_idx
                    )

                def update_enemy_position(state: GalaxianState) -> GalaxianState:
                    delta_x = jnp.where(
                        jnp.logical_or(state.enemy_attack_states == self.consts.ATTACK,
                                       state.enemy_attack_states == self.consts.DEAD_CALLER),
                        jnp.where(
                            state.enemy_attack_turning == self.consts.NO_TURNING,
                            jnp.where(
                                state.enemy_attack_pos[:, 0] == self.consts.PURPLE_ROW,
                                self.consts.ATTACK_MOVE_PATTERN[
                                    1, state.enemy_attack_move_step, 0] * state.enemy_attack_direction,
                                self.consts.ATTACK_MOVE_PATTERN[
                                    0, state.enemy_attack_move_step, 0] * state.enemy_attack_direction
                            ),
                            jnp.where(
                                state.enemy_attack_pos[:, 0] == self.consts.PURPLE_ROW,
                                self.consts.ATTACK_TURN_PATTERN[
                                    1, state.enemy_attack_turn_step, 0] * state.enemy_attack_turning,
                                self.consts.ATTACK_TURN_PATTERN[
                                    0, state.enemy_attack_turn_step, 0] * state.enemy_attack_turning
                            )
                        ),
                        0
                    )

                    delta_y = jnp.where(
                        jnp.logical_or(state.enemy_attack_states == self.consts.ATTACK,
                                       state.enemy_attack_states == self.consts.DEAD_CALLER),
                        1,
                        0
                    )

                    # Falls ein weißer Angreifer aus Reihe 5 ist, bewege auch die Supporter
                    support_caller_active = state.enemy_support_caller_idx != self.consts.ERROR_VALUE

                    # Berechne neue Support-Positionen mit jnp.where
                    curr_support_x = state.enemy_support_x
                    curr_support_y = state.enemy_support_y

                    # Vektorisierte Bewegungsanwendung auf Supporter
                    supporter_active = state.enemy_support_states == 1
                    # Nur Bewegung anwenden, wenn Support aktiv ist und Caller ein weißer Feind ist
                    delta_x_support = jnp.where(supporter_active & support_caller_active,
                                                jnp.tile(delta_x[state.enemy_support_caller_idx],
                                                         self.consts.MAX_SUPPORTERS),
                                                jnp.zeros(self.consts.MAX_SUPPORTERS))
                    delta_y_support = jnp.where(supporter_active & support_caller_active,
                                                jnp.tile(delta_y[state.enemy_support_caller_idx],
                                                         self.consts.MAX_SUPPORTERS),
                                                jnp.zeros(self.consts.MAX_SUPPORTERS))

                    return state.replace(
                        enemy_attack_x=curr_x + delta_x,
                        enemy_attack_y=curr_y + delta_y,
                        enemy_support_x=curr_support_x + delta_x_support,
                        enemy_support_y=curr_support_y + delta_y_support

                    )

                state = update_enemy_state(state)
                state = update_enemy_position(state)
                return state

            new_pause_step = jnp.where(
                state.enemy_attack_pause_step < 7,
                state.enemy_attack_pause_step + 1,
                0
            )

            state = state.replace(
                enemy_attack_pause_step=new_pause_step,
            )

            return jax.lax.cond(
                self.consts.ATTACK_PAUSE_PATTERN[new_pause_step] == 1,
                lambda state: dive(state),
                lambda state: state,
                state
            )

        def respawn_finished_dives(state: GalaxianState) -> GalaxianState:
            def body(i, new_state):
                # diver unter dem player und außerhalb window werden auf respawn gesetzt
                respawn_condition = jnp.logical_and(
                    jnp.logical_or(new_state.enemy_attack_y[i] > self.consts.DIVE_KILL_Y,
                                   jnp.logical_or(new_state.enemy_attack_x[i] < self.consts.ENEMY_LEFT_BOUND,
                                                  new_state.enemy_attack_x[i] > self.consts.ENEMY_RIGHT_BOUND)),
                    new_state.enemy_attack_states[i] == 1)

                new_state = jax.lax.cond(
                    respawn_condition,
                    lambda state: state.replace(
                        enemy_attack_states=state.enemy_attack_states.at[i].set(self.consts.RESPAWN),
                        enemy_attack_x=state.enemy_attack_x.at[i].set(
                            state.enemy_grid_x[state.enemy_attack_pos[i, 0], state.enemy_attack_pos[i, 1]]
                        ),
                        enemy_attack_y=state.enemy_attack_y.at[i].set(-10)
                    ),
                    lambda state: state,
                    new_state
                )

                # continue respawnende diver
                new_state = jax.lax.cond(
                    new_state.enemy_attack_states[i] == self.consts.RESPAWN,
                    lambda state: state.replace(
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

                # beende respawn
                new_state = jax.lax.cond(
                    (new_state.enemy_attack_states[i] == self.consts.RESPAWN) &
                    (new_state.enemy_attack_x[i] == new_state.enemy_grid_x[
                        new_state.enemy_attack_pos[i, 0], new_state.enemy_attack_pos[i, 1]]) &
                    (new_state.enemy_attack_y[i] == new_state.enemy_grid_y[
                        new_state.enemy_attack_pos[i, 0], new_state.enemy_attack_pos[i, 1]]),
                    lambda state: jax.lax.cond(
                        is_caller,
                        lambda state: state.replace(
                            enemy_support_caller_idx=jnp.array(self.consts.ERROR_VALUE, dtype=jnp.int32),
                            enemy_attack_states=state.enemy_attack_states.at[i].set(self.consts.EMPTY),
                            enemy_grid_state=state.enemy_grid_state.at[tuple(state.enemy_attack_pos[i])].set(
                                self.consts.GRID)
                        ),
                        lambda state: state.replace(
                            enemy_attack_states=state.enemy_attack_states.at[i].set(self.consts.EMPTY),
                            enemy_grid_state=state.enemy_grid_state.at[tuple(state.enemy_attack_pos[i])].set(
                                self.consts.GRID)
                        ),
                        new_state
                    ),
                    lambda state: state,
                    new_state
                )

                return new_state

            def body_support(i, new_state):
                # supporter unter dem player und außerhalb window werden auf respawn gesetzt
                respawn_condition = jnp.logical_and(
                    jnp.logical_or(new_state.enemy_support_y[i] > self.consts.DIVE_KILL_Y,
                                   jnp.logical_or(new_state.enemy_support_x[i] < self.consts.ENEMY_LEFT_BOUND,
                                                  new_state.enemy_support_x[i] > self.consts.ENEMY_RIGHT_BOUND)),
                    new_state.enemy_support_states[i] == 1)

                new_state = jax.lax.cond(
                    respawn_condition,
                    lambda state: state.replace(
                        enemy_support_states=state.enemy_support_states.at[i].set(self.consts.RESPAWN),
                        enemy_support_x=state.enemy_support_x.at[i].set(
                            state.enemy_grid_x[state.enemy_support_pos[i, 0], state.enemy_support_pos[i, 1]]
                        ),
                        enemy_support_y=state.enemy_support_y.at[i].set(-10)
                    ),
                    lambda state: state,
                    new_state
                )

                # continue respawnende diver
                new_state = jax.lax.cond(
                    new_state.enemy_support_states[i] == self.consts.RESPAWN,
                    lambda state: state.replace(
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

                # beende respawn
                new_state = jax.lax.cond(
                    (new_state.enemy_support_states[i] == self.consts.RESPAWN) &
                    (new_state.enemy_support_x[i] == new_state.enemy_grid_x[
                        new_state.enemy_support_pos[i, 0], new_state.enemy_support_pos[i, 1]]) &
                    (new_state.enemy_support_y[i] == new_state.enemy_grid_y[
                        new_state.enemy_support_pos[i, 0], new_state.enemy_support_pos[i, 1]]),
                    lambda state: state.replace(
                        enemy_support_states=state.enemy_support_states.at[i].set(self.consts.EMPTY),
                        enemy_grid_state=state.enemy_grid_state.at[tuple(state.enemy_support_pos[i])].set(
                            self.consts.GRID)
                    ),
                    lambda state: state,
                    new_state
                )
                return new_state

            def scan_body(carry, i):
                return body(i, carry), None

            def scan_body_support(carry, i):
                return body_support(i, carry), None

            state, _ = lax.scan(scan_body, state, jnp.arange(self.consts.MAX_DIVERS))
            state, _ = lax.scan(scan_body_support, state, jnp.arange(self.consts.MAX_SUPPORTERS))
            return state

        new_state = respawn_finished_dives(state)
        new_state = test_for_new_dive(new_state)
        new_state = continue_active_dives(new_state)

        return new_state

    @partial(jax.jit, static_argnums=(0,))
    def update_enemy_bullets(self, state: 'GalaxianState') -> 'GalaxianState':

        # volley timer
        new_timers = jnp.maximum(0, state.enemy_attack_shot_timer - 1)

        # continue active bullets
        is_active_mask = state.enemy_attack_bullet_y >= 0.0
        moved_y = jnp.where(is_active_mask, state.enemy_attack_bullet_y + self.consts.ENEMY_ATTACK_BULLET_SPEED,
                            -1.0)
        moved_x = jnp.where(is_active_mask, state.enemy_attack_bullet_x, -1.0)

        # despawn bullets below player
        off_screen_mask = moved_y > self.consts.NATIVE_GAME_HEIGHT
        bullets_y_after_move = jnp.where(off_screen_mask, -1.0, moved_y)
        bullets_x_after_move = jnp.where(off_screen_mask, -1.0, moved_x)

        # continue volley
        can_shoot_mask = (
                (state.enemy_attack_states == self.consts.ATTACK) &
                (state.enemy_attack_shots_fired < state.enemy_attack_volley_size) &
                (new_timers <= 0)
        )

        # select new shooters
        potential_shooter_indices = jnp.where(can_shoot_mask, jnp.arange(self.consts.MAX_DIVERS),
                                              self.consts.MAX_DIVERS)
        sorted_shooter_indices = jnp.sort(potential_shooter_indices)

        def _spawn_one_shot(shooter_idx, carry):
            # find empty slot
            (current_x, current_y, current_shots_fired, current_timers) = carry
            available_bullet_slots = jnp.where(current_y == -1.0, size=1, fill_value=-1)[0]
            target_bullet_slot = available_bullet_slots[0]
            can_spawn = (shooter_idx < self.consts.MAX_DIVERS) & (target_bullet_slot != -1)

            def _do_spawn(x, y, shots_fired, timers):
                new_x = x.at[target_bullet_slot].set(state.enemy_attack_x[shooter_idx])
                new_y = y.at[target_bullet_slot].set(state.enemy_attack_y[shooter_idx])
                new_shots_fired = shots_fired.at[shooter_idx].set(shots_fired[shooter_idx] + 1)
                new_timers = timers.at[shooter_idx].set(self.consts.VOLLEY_SHOT_DELAY)
                return new_x, new_y, new_shots_fired, new_timers

            def _do_not_spawn(x, y, shots_fired, timers):
                return x, y, shots_fired, timers

            return lax.cond(
                can_spawn,
                lambda: _do_spawn(current_x, current_y, current_shots_fired, current_timers),
                lambda: _do_not_spawn(current_x, current_y, current_shots_fired, current_timers)
            )

        initial_carry = (bullets_x_after_move, bullets_y_after_move, state.enemy_attack_shots_fired, new_timers)

        def scan_spawn_one_shot(carry, i):
            return _spawn_one_shot(sorted_shooter_indices[i], carry), None

        final_carry, _ = lax.scan(scan_spawn_one_shot, initial_carry, jnp.arange(self.consts.MAX_DIVERS))
        final_bullet_x, final_bullet_y, final_shots_fired, final_timers = final_carry

        return state.replace(
            enemy_attack_bullet_x=final_bullet_x.astype(jnp.float32),
            enemy_attack_bullet_y=final_bullet_y.astype(jnp.float32),
            enemy_attack_shots_fired=final_shots_fired,
            enemy_attack_shot_timer=final_timers
        )

    @partial(jax.jit, static_argnums=(0,))
    def update_player_bullet(self, state: GalaxianState, action: chex.Array) -> GalaxianState:
        is_shooting_action = jnp.any(
            jnp.array([
                action == Action.FIRE,
                action == Action.RIGHTFIRE,
                action == Action.LEFTFIRE,
            ])
        )

        bullet_is_inactive = state.bullet_y < 0

        def fire_new_bullet(state: GalaxianState) -> GalaxianState:
            return state.replace(
                bullet_x=state.player_x + self.consts.PLAYER_BULLET_X_OFFSET,
                bullet_y=state.player_y - self.consts.PLAYER_BULLET_Y_OFFSET
            )

        def move_active_bullet(state: GalaxianState) -> GalaxianState:
            new_y = state.bullet_y - self.consts.BULLET_MOVE_SPEED

            return state.replace(
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

    @partial(jax.jit, static_argnums=(0,))
    def update_enemy_death_frames(self, state: GalaxianState) -> GalaxianState:
        def advance_cell(frame):
            # if in 1..5, increment or then clear
            return jnp.where(frame == 0, 0,
                             jnp.where(frame < 5, frame + 1, 0))

        new_frames_grid = jax.vmap(jax.vmap(advance_cell))(state.enemy_death_frame_grid)
        new_frames_attack = jax.vmap(advance_cell)(state.enemy_death_frame_attack)
        new_frames_support = jax.vmap(advance_cell)(state.enemy_death_frame_support)
        # once frame wraps to 0, also mark the cell fully dead
        cleared_mask_grid = (state.enemy_death_frame_grid == 5)
        new_alive = jnp.where(cleared_mask_grid, 0, state.enemy_grid_state)
        new_attackers = jnp.where(
            state.enemy_death_frame_attack == 5,
            jnp.where(
                jnp.arange(state.enemy_attack_states.shape[0]) == state.enemy_support_caller_idx,
                self.consts.DEAD_CALLER,
                self.consts.DEAD
            ),
            state.enemy_attack_states
        )
        new_supporters = jnp.where(
            state.enemy_death_frame_support == 5,
            0,
            state.enemy_support_states
        )
        return state.replace(
            enemy_death_frame_grid=new_frames_grid,
            enemy_death_frame_attack=new_frames_attack,
            enemy_death_frame_support=new_frames_support,
            enemy_grid_state=new_alive,
            enemy_attack_states=new_attackers,
            enemy_support_states=new_supporters,
        )

    @partial(jax.jit, static_argnums=(0,))
    def bullet_collision(self, state: GalaxianState) -> GalaxianState:
        # Kollision mit Feinden im Gitter
        x_diff_grid = jnp.abs(state.bullet_x - (state.enemy_grid_x + self.consts.ENEMY_WIDTH / 2))
        y_diff_grid = jnp.abs(state.bullet_y - (state.enemy_grid_y + self.consts.ENEMY_HEIGHT / 2))
        mask_grid = (x_diff_grid <= self.consts.ENEMY_WIDTH / 2) & (
                    y_diff_grid <= self.consts.ENEMY_HEIGHT / 2) & (
                                state.enemy_grid_state == self.consts.GRID)
        hit_grid = jnp.any(mask_grid)

        # Kollision mit angreifenden Feinden
        x_diff_attack = jnp.abs(state.bullet_x - (state.enemy_attack_x + self.consts.ENEMY_ATTACK_WIDTH / 2))
        y_diff_attack = jnp.abs(state.bullet_y - (state.enemy_attack_y + self.consts.ENEMY_ATTACK_HEIGHT / 2))
        mask_attack = (x_diff_attack <= self.consts.ENEMY_ATTACK_WIDTH / 2) & (
                    y_diff_attack <= self.consts.ENEMY_ATTACK_HEIGHT / 2) & (
                                  state.enemy_attack_states == self.consts.ATTACK)
        hit_attack = jnp.any(mask_attack) & state.player_alive

        # Kollision mit Unterstützern
        x_diff_support = jnp.abs(state.bullet_x - (state.enemy_support_x + self.consts.ENEMY_ATTACK_WIDTH / 2))
        y_diff_support = jnp.abs(state.bullet_y - (state.enemy_support_y + self.consts.ENEMY_ATTACK_HEIGHT / 2))
        mask_support = (x_diff_support <= self.consts.ENEMY_ATTACK_WIDTH / 2) & (
                    y_diff_support <= self.consts.ENEMY_ATTACK_HEIGHT / 2) & (
                                   state.enemy_support_states == self.consts.ATTACK)
        hit_support = jnp.any(mask_support) & state.player_alive

        # Treffer-Verarbeitung für Gitter-Feinde
        def process_hit_grid(state):
            collision_indices = jnp.where(mask_grid, size=state.enemy_grid_state.size, fill_value=-1)
            hit_rows = collision_indices[0]
            hit_cols = collision_indices[1]
            hit_row = hit_rows[0]  # erster Treffer
            hit_col = hit_cols[0]

            new_death = state.enemy_death_frame_grid.at[hit_row, hit_col].set(1)
            new_alive = state.enemy_grid_state.at[hit_row, hit_col].set(self.consts.DEAD)

            return state.replace(
                enemy_grid_state=new_alive,
                enemy_death_frame_grid=new_death,
                bullet_x=jnp.array(self.consts.ERROR_VALUE, dtype=state.bullet_y.dtype),
                bullet_y=jnp.array(self.consts.ERROR_VALUE, dtype=state.bullet_y.dtype),
                score=state.score + self.consts.SCORES[hit_row]
            )

        # Treffer-Verarbeitung für angreifende Feinde
        def process_hit_attack(state):
            hit_indices = \
            jnp.where(mask_attack, size=self.consts.MAX_DIVERS, fill_value=self.consts.ERROR_VALUE)[0]
            hit_idx = hit_indices[0]
            pos = state.enemy_attack_pos[hit_idx]
            new_grid = state.enemy_grid_state.at[tuple(pos)].set(self.consts.DEAD)
            new_death = state.enemy_death_frame_attack.at[hit_idx].set(1)
            new_attack_states = state.enemy_attack_states.at[hit_idx].set(self.consts.DYING)

            score_row = pos[0]
            score_multiplier = jnp.where(state.enemy_attack_states[hit_idx] == self.consts.ATTACK,
                                         self.consts.DIVE_MULTIPLIER, 1)

            return state.replace(
                enemy_grid_state=new_grid,
                enemy_death_frame_attack=new_death,
                bullet_x=jnp.array(self.consts.ERROR_VALUE, dtype=state.bullet_y.dtype),
                bullet_y=jnp.array(self.consts.ERROR_VALUE, dtype=state.bullet_y.dtype),
                enemy_attack_states=new_attack_states,
                score=state.score + self.consts.SCORES[score_row] * score_multiplier
            )

        # Treffer-Verarbeitung für Unterstützer
        def process_hit_support(state):
            hit_indices = \
            jnp.where(mask_support, size=self.consts.MAX_SUPPORTERS, fill_value=self.consts.ERROR_VALUE)[0]
            hit_idx = hit_indices[0]
            pos = state.enemy_support_pos[hit_idx]
            new_grid = state.enemy_grid_state.at[tuple(pos)].set(self.consts.DEAD)
            new_death = state.enemy_death_frame_support.at[hit_idx].set(1)
            new_support_states = state.enemy_support_states.at[hit_idx].set(self.consts.DYING)

            return state.replace(
                enemy_grid_state=new_grid,
                enemy_death_frame_support=new_death,
                bullet_x=jnp.array(self.consts.ERROR_VALUE, dtype=state.bullet_y.dtype),
                bullet_y=jnp.array(self.consts.ERROR_VALUE, dtype=state.bullet_y.dtype),
                enemy_support_states=new_support_states,
                score=state.score + self.consts.SCORES[
                    self.consts.RED_ROW] * self.consts.DIVE_MULTIPLIER
            )

        # Verarbeite Treffer in prioritätsreihenfolge
        state = lax.cond(hit_grid, process_hit_grid, lambda s: s, state)
        state = lax.cond(hit_attack & (state.bullet_y > 0), process_hit_attack, lambda s: s, state)
        state = lax.cond(hit_support & (state.bullet_y > 0), process_hit_support, lambda s: s, state)

        return state

    @partial(jax.jit, static_argnums=(0,))
    def check_player_death_by_enemy(self, state: GalaxianState) -> GalaxianState:
        x_diff = jnp.abs(state.player_x + self.consts.PLAYER_WIDTH / 2 - (
                    state.enemy_attack_x + self.consts.ENEMY_ATTACK_WIDTH / 2))
        y_diff = jnp.abs(state.player_y + self.consts.PLAYER_HEIGHT / 2 - (
                    state.enemy_attack_y + self.consts.ENEMY_ATTACK_HEIGHT / 2))
        is_active = (state.enemy_attack_states == 1)
        grid_state = state.enemy_grid_state[tuple(state.enemy_attack_pos.T)] != 0

        collision = (x_diff <= self.consts.PLAYER_WIDTH / 2) & (
                    y_diff <= self.consts.PLAYER_HEIGHT / 2) & is_active & grid_state
        hit = jnp.any(collision) & state.player_alive

        def process_hit(current_state):
            hit_indices = jnp.where(collision, size=self.consts.MAX_DIVERS, fill_value=-1)[0]
            hit_idx = hit_indices[0]
            pos = current_state.enemy_attack_pos[hit_idx]
            new_enemy_grid_state = current_state.enemy_grid_state.at[tuple(pos)].set(self.consts.DEAD)
            new_death = state.enemy_death_frame_attack.at[hit_idx].set(1)
            new_attack_states = state.enemy_attack_states.at[hit_idx].set(self.consts.DYING)

            score_row = pos[0]
            score_multiplier = jnp.where(state.enemy_attack_states[hit_idx] == self.consts.ATTACK,
                                         self.consts.DIVE_MULTIPLIER, 1)

            return current_state.replace(
                player_alive=jnp.array(False),
                enemy_grid_state=new_enemy_grid_state,
                enemy_attack_states=new_attack_states,
                enemy_death_frame_attack=new_death,
                score=state.score + self.consts.SCORES[score_row] * score_multiplier,
            )

        return lax.cond(hit, process_hit, lambda s: s, state)

    @partial(jax.jit, static_argnums=(0,))
    def check_player_death_by_support(self, state: GalaxianState) -> GalaxianState:
        x_diff = jnp.abs(state.player_x + self.consts.PLAYER_WIDTH / 2 - (
                    state.enemy_support_x + self.consts.ENEMY_ATTACK_WIDTH / 2))
        y_diff = jnp.abs(state.player_y + self.consts.PLAYER_HEIGHT / 2 - (
                    state.enemy_support_y + self.consts.ENEMY_ATTACK_HEIGHT / 2))
        is_active = (state.enemy_support_states == 1)
        grid_state = state.enemy_grid_state[tuple(state.enemy_support_pos.T)] != 0

        collision = (x_diff <= self.consts.PLAYER_WIDTH / 2) & (
                    y_diff <= self.consts.PLAYER_HEIGHT / 2) & is_active & grid_state
        hit = jnp.any(collision) & state.player_alive

        def process_hit(current_state):
            hit_indices = jnp.where(collision, size=self.consts.MAX_DIVERS, fill_value=-1)[0]
            hit_idx = hit_indices[0]
            pos = current_state.enemy_support_pos[hit_idx]
            new_enemy_grid_state = current_state.enemy_grid_state.at[tuple(pos)].set(self.consts.DEAD)
            new_support_states = current_state.enemy_support_states.at[hit_idx].set(self.consts.DYING)
            new_death = current_state.enemy_death_frame_support.at[hit_idx].set(1)

            return current_state.replace(
                player_alive=jnp.array(False),
                enemy_grid_state=new_enemy_grid_state,
                enemy_support_states=new_support_states,
                enemy_death_frame_support=new_death,
                score=state.score + self.consts.SCORES[
                    self.consts.RED_ROW] * self.consts.DIVE_MULTIPLIER,
            )

        return lax.cond(hit, process_hit, lambda s: s, state)

    @partial(jax.jit, static_argnums=(0,))
    def check_player_death_by_bullet(self, state: GalaxianState) -> GalaxianState:
        x_diff = jnp.abs(state.player_x + self.consts.PLAYER_WIDTH / 2 - state.enemy_attack_bullet_x)
        y_diff = jnp.abs(state.player_y + self.consts.PLAYER_HEIGHT / 2 - state.enemy_attack_bullet_y)

        collision_mask = (x_diff <= self.consts.PLAYER_WIDTH / 2) & (
                    y_diff <= self.consts.PLAYER_HEIGHT / 2) & (state.enemy_attack_bullet_y >= 0)
        hit = jnp.any(collision_mask) & state.player_alive

        def process_hit(current_state):
            # reset bullets
            hit_indices = jnp.where(collision_mask, size=self.consts.MAX_DIVERS, fill_value=-1)[
                0]
            new_bullet_x = current_state.enemy_attack_bullet_x.at[hit_indices].set(-1.0)
            new_bullet_y = current_state.enemy_attack_bullet_y.at[hit_indices].set(-1.0)

            return current_state.replace(
                player_alive=jnp.array(False),
                enemy_attack_bullet_x=new_bullet_x,
                enemy_attack_bullet_y=new_bullet_y
            )

        return lax.cond(hit, process_hit, lambda s: s, state)

    @partial(jax.jit, static_argnums=(0,))
    def increase_player_respawn_timer(self, state: GalaxianState) -> GalaxianState:
        timer = jnp.where(
            state.player_alive,
            0,
            state.player_respawn_timer + 1,
        )
        return state.replace(
            player_respawn_timer=timer,
        )

    @partial(jax.jit, static_argnums=(0,))
    def try_respawn_player(self, state: GalaxianState) -> GalaxianState:
        return jax.lax.cond(
            state.player_respawn_timer >= self.consts.PLAYER_RESPAWN_TIME,
            lambda s: s.replace(
                player_alive=jnp.array(True),
                lives=state.lives - 1
            ),
            lambda s: s,
            state
        )

    @partial(jax.jit, static_argnums=(0,))
    def enter_new_wave(self, state: GalaxianState) -> GalaxianState:
        new_grid = self.consts.ENEMY_GRID
        new_level = state.level + 1
        return state.replace(enemy_grid_state=new_grid,
                              level=new_level, )

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jax.random.PRNGKey = None) -> Tuple[GalaxianObservation, GalaxianState]:
        grid_rows = self.consts.GRID_ROWS
        grid_cols = self.consts.GRID_COLS
        enemy_spacing_x = self.consts.ENEMY_SPACING_X
        start_x = self.consts.START_X

        x_positions = jnp.arange(grid_cols) * enemy_spacing_x + start_x  # arange schreibt so 0 1 2 3....
        enemy_grid = jnp.tile(x_positions, (grid_rows, 1))  # kopiert die zeile untereinander

        row_indices = jnp.arange(grid_rows).reshape(-1, 1)  # erzeugt 1. enemy jeder zeile
        enemy_y_rows = self.consts.ENEMY_GRID_Y - row_indices * self.consts.ENEMY_SPACING_Y  # jeweils y pos
        enemy_grid_y = jnp.broadcast_to(enemy_y_rows, (
            grid_rows, grid_cols))  # kopiert die werte für 1. enemy auf die rechts in der zeile




        state = GalaxianState(player_x=jnp.array(self.consts.NATIVE_GAME_WIDTH / 2.0, dtype=jnp.float32),
                              player_y=jnp.array(self.consts.NATIVE_GAME_HEIGHT - 40.0, dtype=jnp.float32),
                              player_respawn_timer=jnp.array(0),
                              bullet_x=jnp.array(-1.0,dtype=jnp.float32),
                              bullet_y=jnp.array(-1.0,dtype=jnp.float32),
                              enemy_grid_x=enemy_grid.astype(jnp.float32),
                              enemy_grid_y=enemy_grid_y.astype(jnp.float32),
                              enemy_grid_state=self.consts.ENEMY_GRID,
                              enemy_death_frame_grid=jnp.zeros((self.consts.GRID_ROWS, self.consts.GRID_COLS), dtype=jnp.int32),
                              enemy_death_frame_attack=jnp.zeros(self.consts.MAX_DIVERS, dtype=jnp.int32),
                              enemy_death_frame_support=jnp.zeros(self.consts.MAX_SUPPORTERS, dtype=jnp.int32),
                              enemy_grid_direction=jnp.array(1),
                              enemy_grid_move_frames=jnp.array(0),
                              enemy_attack_states=jnp.zeros(self.consts.MAX_DIVERS),
                              enemy_attack_pos=jnp.full((self.consts.MAX_DIVERS, 2), self.consts.ERROR_VALUE, dtype=jnp.int32),
                              enemy_attack_direction=jnp.zeros(self.consts.MAX_DIVERS),
                              enemy_attack_turning=jnp.zeros(self.consts.MAX_DIVERS),
                              enemy_attack_turn_step=jnp.zeros(self.consts.MAX_DIVERS, dtype=jnp.int32),
                              enemy_attack_move_step=jnp.zeros(self.consts.MAX_DIVERS, dtype=jnp.int32),
                              enemy_attack_x=jnp.zeros(self.consts.MAX_DIVERS, dtype=jnp.float32),
                              enemy_attack_y=jnp.zeros(self.consts.MAX_DIVERS, dtype=jnp.float32),
                              enemy_attack_respawn_timer=jnp.zeros(self.consts.MAX_DIVERS),
                              enemy_attack_bullet_x=jnp.full(self.consts.MAX_DIVERS, -1.0, dtype=jnp.float32),
                              enemy_attack_bullet_y=jnp.full(self.consts.MAX_DIVERS, -1.0, dtype=jnp.float32),
                              enemy_attack_bullet_timer=jnp.zeros(self.consts.MAX_DIVERS),
                              enemy_attack_pause_step=jnp.array(0),
                              enemy_attack_number=jnp.array(0),
                              enemy_attack_max=jnp.array(1),
                              enemy_support_caller_idx=jnp.array(self.consts.ERROR_VALUE, dtype=jnp.int32),
                              enemy_support_states=jnp.zeros(self.consts.MAX_SUPPORTERS, dtype=jnp.int32),
                              enemy_support_pos=jnp.full((self.consts.MAX_SUPPORTERS, 2), self.consts.ERROR_VALUE, dtype=jnp.int32),
                              enemy_support_x=jnp.zeros(self.consts.MAX_SUPPORTERS, dtype=jnp.float32),
                              enemy_support_y=jnp.zeros(self.consts.MAX_SUPPORTERS, dtype=jnp.float32),
                              level=jnp.array(0, dtype=jnp.int32),
                              lives=jnp.array(self.consts.LIVES, dtype=jnp.int32),
                              got_extra_life=jnp.array(False),
                              player_alive=jnp.array(True),
                              score=jnp.array(0, dtype=jnp.int32),
                              enemy_attack_target_x=jnp.zeros(self.consts.MAX_DIVERS),
                              enemy_attack_target_y=jnp.zeros(self.consts.MAX_DIVERS),
                              turn_step=jnp.array(0),
                              enemy_attack_shot_timer=jnp.zeros(self.consts.MAX_DIVERS),
                              enemy_attack_shots_fired=jnp.zeros(self.consts.MAX_DIVERS, dtype=jnp.int32),
                              enemy_attack_volley_size=jnp.zeros(self.consts.MAX_DIVERS, dtype=jnp.int32)
                             )

        initial_obs = self._get_observation(state)
        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: GalaxianState) -> GalaxianObservation:
        return GalaxianObservation(
            player_x=state.player_x,
            player_y=state.player_y,
            bullet_x=state.bullet_x,
            bullet_y=state.bullet_y,
            enemy_grid_x=state.enemy_grid_x,
            enemy_grid_y=state.enemy_grid_y,
            enemy_grid_state=state.enemy_grid_state,
            enemy_attack_pos=state.enemy_attack_pos,
            enemy_attack_x=state.enemy_attack_x,
            enemy_attack_y=state.enemy_attack_y,
            enemy_support_pos=state.enemy_support_pos,
            enemy_support_x=state.enemy_support_x,
            enemy_support_y=state.enemy_support_y,
            enemy_attack_bullet_x=state.enemy_attack_bullet_x,
            enemy_attack_bullet_y=state.enemy_attack_bullet_y,
        )

    @partial(jax.jit, static_argnums=(0,))
    def get_action_space(self):
        return jnp.array([Action.NOOP, Action.LEFT, Action.RIGHT, Action.FIRE, Action.LEFTFIRE, Action.RIGHTFIRE])


    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTION_SET))

    def observation_space(self) -> spaces.Dict:
        return spaces.Dict(
            {
                "player_x": spaces.Box(low=0, high=self.consts.NATIVE_GAME_WIDTH, shape=(), dtype=jnp.float32),
                "player_y": spaces.Box(low=0, high=self.consts.NATIVE_GAME_HEIGHT, shape=(), dtype=jnp.float32),
                "bullet_x": spaces.Box(low=self.consts.ERROR_VALUE, high=self.consts.NATIVE_GAME_WIDTH, shape=(), dtype=jnp.float32),
                "bullet_y": spaces.Box(low=self.consts.ERROR_VALUE, high=self.consts.NATIVE_GAME_HEIGHT, shape=(), dtype=jnp.float32),
                "enemy_grid_x": spaces.Box(low=0, high=self.consts.NATIVE_GAME_WIDTH, shape=(self.consts.GRID_ROWS, self.consts.GRID_COLS), dtype=jnp.float32),
                "enemy_grid_y": spaces.Box(low=0, high=self.consts.NATIVE_GAME_HEIGHT, shape=(self.consts.GRID_ROWS, self.consts.GRID_COLS), dtype=jnp.float32),
                "enemy_grid_state": spaces.Box(low=0, high=2, shape=(self.consts.GRID_ROWS, self.consts.GRID_COLS), dtype=jnp.float32),
                "enemy_attack_pos": spaces.Box(low=self.consts.ERROR_VALUE, high=max(self.consts.GRID_ROWS, self.consts.GRID_COLS), shape=(self.consts.MAX_DIVERS, 2), dtype=jnp.float32),
                "enemy_attack_x": spaces.Box(low=self.consts.ERROR_VALUE, high=self.consts.NATIVE_GAME_WIDTH, shape=(self.consts.MAX_DIVERS,), dtype=jnp.float32),
                "enemy_attack_y": spaces.Box(low=self.consts.ERROR_VALUE, high=self.consts.NATIVE_GAME_HEIGHT, shape=(self.consts.MAX_DIVERS,), dtype=jnp.float32),
                "enemy_support_pos": spaces.Box(low=self.consts.ERROR_VALUE, high=max(self.consts.GRID_ROWS, self.consts.GRID_COLS), shape=(self.consts.MAX_SUPPORTERS, 2), dtype=jnp.float32),
                "enemy_support_x": spaces.Box(low=self.consts.ERROR_VALUE, high=self.consts.NATIVE_GAME_WIDTH, shape=(self.consts.MAX_SUPPORTERS,), dtype=jnp.float32),
                "enemy_support_y": spaces.Box(low=self.consts.ERROR_VALUE, high=self.consts.NATIVE_GAME_HEIGHT, shape=(self.consts.MAX_SUPPORTERS,), dtype=jnp.float32),
                "enemy_attack_bullet_x": spaces.Box(low=self.consts.ERROR_VALUE, high=self.consts.NATIVE_GAME_WIDTH, shape=(self.consts.MAX_DIVERS,), dtype=jnp.float32),
                "enemy_attack_bullet_y": spaces.Box(low=self.consts.ERROR_VALUE, high=self.consts.NATIVE_GAME_HEIGHT, shape=(self.consts.MAX_DIVERS,), dtype=jnp.float32),
            }
        )

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(210, 160, 3),
            dtype=jnp.uint8
        )

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: GalaxianObservation) -> chex.Array:
        """Converts the observation to a flat array."""
        return jnp.concatenate(
            [
                obs.player_x.flatten(),
                obs.player_y.flatten(),
                obs.bullet_x.flatten(),
                obs.bullet_y.flatten(),
                obs.enemy_grid_x.flatten(),
                obs.enemy_grid_y.flatten(),
                obs.enemy_grid_state.flatten(),
                obs.enemy_attack_pos.flatten(),
                obs.enemy_attack_x.flatten(),
                obs.enemy_attack_y.flatten(),
                obs.enemy_support_pos.flatten(),
                obs.enemy_support_x.flatten(),
                obs.enemy_support_y.flatten(),
                obs.enemy_attack_bullet_x.flatten(),
                obs.enemy_attack_bullet_y.flatten(),
            ]
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(
            self, state: GalaxianState, action: chex.Array
    ) -> Tuple[GalaxianObservation, GalaxianState, float, bool, GalaxianInfo]:
        # Translate agent action index to ALE console action
        atari_action = jnp.take(self.ACTION_SET, jnp.asarray(action, dtype=jnp.int32))

        new_state = self.update_player_position(state, atari_action)
        new_state = self.update_player_bullet(new_state, atari_action)
        new_state = self.update_enemy_positions(new_state)
        new_state = self.bullet_collision(new_state)
        new_state = self.update_enemy_death_frames(new_state)
        new_state = self.update_enemy_attack(new_state)
        new_state = self.update_enemy_bullets(new_state)
        new_state = self.check_player_death_by_enemy(new_state)
        new_state = self.check_player_death_by_support(new_state)
        new_state = self.check_player_death_by_bullet(new_state)
        new_state = self.increase_player_respawn_timer(new_state)
        new_state = jax.lax.cond(jnp.logical_and(jnp.logical_not(jnp.any(state.enemy_grid_state == self.consts.ACTIVE)), state.player_alive == False), lambda new_state: self.try_respawn_player(new_state), lambda s: s, new_state)
        new_state = jax.lax.cond(jnp.logical_and(jnp.logical_not(jnp.any(state.enemy_grid_state == self.consts.GRID)), jnp.logical_not(jnp.any(state.enemy_attack_states != 0))), lambda new_state: self.enter_new_wave(new_state), lambda s: s, new_state)
        new_state = jax.lax.cond(jnp.logical_and(jnp.logical_not(state.got_extra_life),state.score >= self.consts.EXTRA_LIFE_SCORE), lambda s: s.replace(lives=s.lives + 1, got_extra_life=jnp.array(True)), lambda s: s, new_state)
        new_state = new_state.replace(turn_step=new_state.turn_step + 1)
        done = self._get_done(new_state)
        env_reward = self._get_reward(state, new_state)
        info = self._get_info(new_state)

        observation = self._get_observation(new_state)


        return observation, new_state, env_reward, done, info



    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: GalaxianState, state: GalaxianState):
        return state.score - previous_state.score

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: GalaxianState) -> bool:
        return state.lives < 0

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: GalaxianState) -> GalaxianInfo:
        return GalaxianInfo(
            time=state.turn_step,
            lives=state.lives,
            score=state.score,
            level=state.level)
            

    def render(self, state: GalaxianState) -> jnp.ndarray:
        return self.renderer.render(state)


class GalaxianRenderer(JAXGameRenderer):
    def __init__(self, consts: GalaxianConstants = None):
        super().__init__()
        self.consts = consts or GalaxianConstants()
        
        # 1. Configure the rendering utility
        self.config = render_utils.RendererConfig(
            game_dimensions=(self.consts.NATIVE_GAME_HEIGHT, self.consts.NATIVE_GAME_WIDTH),
            channels=3,
            #downscale=(84, 84)
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)
        # 2. Define sprite path
        MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.sprite_path = os.path.join(MODULE_DIR, "sprites/galaxian")
        # 3. Get asset config from constants
        final_asset_config = list(self.consts.ASSET_CONFIG)
        # 4. Load, process, and set up all assets
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(final_asset_config, self.sprite_path)

        # 5. Pre-stack ID masks for fast lookup, replicating old logic
        
        # Stack for grid enemies: [gray, gray, gray, purple, red, white]
        grid_sprites = [
            self.SHAPE_MASKS['enemy_grid_gray'],
            self.SHAPE_MASKS['enemy_grid_gray'],
            self.SHAPE_MASKS['enemy_grid_gray'],
            self.SHAPE_MASKS['enemy_grid_purple'],
            self.SHAPE_MASKS['enemy_grid_red'],
            self.SHAPE_MASKS['enemy_grid_white']
        ]
        self.stacked_grid_sprites = jnp.stack(grid_sprites)
        # Stack for diving-down enemies: [green, green, green, purple, red, white]
        dive_down_sprites = [
            self.SHAPE_MASKS['enemy_dive_down_green'],
            self.SHAPE_MASKS['enemy_dive_down_green'],
            self.SHAPE_MASKS['enemy_dive_down_green'],
            self.SHAPE_MASKS['enemy_dive_down_purple'],
            self.SHAPE_MASKS['enemy_dive_down_red'],
            self.SHAPE_MASKS['enemy_dive_down_white']
        ]
        self.stacked_dive_down_sprites = jnp.stack(dive_down_sprites)
        
        # Stack for diving-left enemies: [green, green, green, purple, red, white]
        # Note: The old code had a confusing mix of left/right. This implementation
        # loads the 'left' sprites and will flip them to create the 'right' view.
        dive_left_sprites = [
            self.SHAPE_MASKS['enemy_dive_left_green'],
            self.SHAPE_MASKS['enemy_dive_left_green'],
            self.SHAPE_MASKS['enemy_dive_left_green'],
            self.SHAPE_MASKS['enemy_dive_left_purple'],
            self.SHAPE_MASKS['enemy_dive_left_red'],
            self.SHAPE_MASKS['enemy_dive_left_white']
        ]
        self.stacked_dive_left_sprites = jnp.stack(dive_left_sprites)

        # Store sprite dimensions for lives
        life_mask = self.SHAPE_MASKS['life']
        self.life_h, self.life_w = life_mask.shape
        self.life_spacing = 3

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: GalaxianState):
        # 1. Start with the background ID raster
        raster = self.jr.create_object_raster(self.BACKGROUND)

        # 2. Player
        player_mask = self.SHAPE_MASKS['player']
        px = jnp.round(state.player_x).astype(jnp.int32)
        py = jnp.round(state.player_y).astype(jnp.int32)
        raster = jax.lax.cond(
            state.player_alive,
            lambda r: self.jr.render_at(r, px, py, player_mask, flip_offset=self.FLIP_OFFSETS['player']),
            lambda r: r,
            raster
        )

        # 3. Player Bullet
        bullet_mask = self.SHAPE_MASKS['bullet']
        def draw_bullet_active(r):
            bullet_x = jnp.round(state.bullet_x).astype(jnp.int32)
            bullet_y = jnp.round(state.bullet_y).astype(jnp.int32)
            return self.jr.render_at(r, bullet_x, bullet_y, bullet_mask, flip_offset=self.FLIP_OFFSETS['bullet'])

        def draw_bullet_inactive(r):
            bullet_x = jnp.round(state.player_x + self.consts.PLAYER_BULLET_X_OFFSET).astype(jnp.int32)
            bullet_y = jnp.round(state.player_y - self.consts.PLAYER_BULLET_Y_OFFSET).astype(jnp.int32)
            return self.jr.render_at(r, bullet_x, bullet_y, bullet_mask, flip_offset=self.FLIP_OFFSETS['bullet'])
        
        raster = jax.lax.cond(
            state.bullet_y > 0,
            draw_bullet_active,
            lambda r: jax.lax.cond(
                state.player_alive,
                draw_bullet_inactive,
                lambda r_in: r_in,
                r
            ),
            raster
        )

        # 4. Enemy Bullets
        enemy_bullet_mask = self.SHAPE_MASKS['enemy_bullet']
        def _draw_single_enemy_bullet(i, r_acc):
            x = jnp.round(state.enemy_attack_bullet_x[i]).astype(jnp.int32)
            y = jnp.round(state.enemy_attack_bullet_y[i]).astype(jnp.int32)
            return jax.lax.cond(
                state.enemy_attack_bullet_y[i] >= 0,  # Active bullet
                lambda r_in: self.jr.render_at_clipped(r_in, x, y, enemy_bullet_mask, flip_offset=self.FLIP_OFFSETS['enemy_bullet']),
                lambda r_in: r_in,
                r_acc
            )

        raster = jax.lax.fori_loop(0, self.consts.MAX_DIVERS, _draw_single_enemy_bullet, raster)

        # 5. Attacking Enemies
        def _draw_single_attacker(i, r_acc):
            is_alive = jnp.logical_or(state.enemy_attack_states[i] == 1, state.enemy_attack_states[i] == 2)
            is_dying = state.enemy_attack_states[i] == 4
            death_frame_attack = state.enemy_death_frame_attack[i].astype(jnp.int32)
            row = state.enemy_attack_pos[i][0]
            direction = jnp.where(
                state.enemy_attack_turning[i] == 0,
                jnp.where(state.enemy_attack_direction[i] == -1, 0, 1), # 0=left, 1=right
                2 # 2=down
            )
            is_down = direction == 2
            
            ex = jnp.round(state.enemy_attack_x[i]).astype(jnp.int32)
            ey = jnp.round(state.enemy_attack_y[i]).astype(jnp.int32)
            def down(r):
                sprite = self.stacked_dive_down_sprites[row]
                return self.jr.render_at_clipped(r, ex, ey, sprite, flip_offset=self.FLIP_OFFSETS['enemy_dive_down_green']) # Offset is same for group
            
            def side(r):
                sprite = self.stacked_dive_left_sprites[row]
                return self.jr.render_at_clipped(r, ex, ey, sprite, 
                                         flip_horizontal=(direction == 1), # Flip if moving right
                                         flip_offset=self.FLIP_OFFSETS['enemy_dive_left_green']) # Offset is same for group

            def dying(r):
                sprite = self.SHAPE_MASKS['enemy_death'][death_frame_attack - 1]
                return self.jr.render_at_clipped(r, ex, ey, sprite, flip_offset=self.FLIP_OFFSETS['enemy_death'])

            def dead(r):
                return r

            return jax.lax.cond(is_alive, lambda r: jax.lax.cond(is_down, down, side, r), lambda r: jax.lax.cond(is_dying, dying, dead, r), r_acc)

        raster = jax.lax.fori_loop(0, self.consts.MAX_DIVERS, _draw_single_attacker, raster)

        # 6. Supporter Enemies
        def _draw_single_supporter(i, r_acc):
            is_alive = jnp.logical_or(state.enemy_support_states[i] == 1, state.enemy_support_states[i] == 2)
            is_dying = state.enemy_support_states[i] == 4
            death_frame_support = state.enemy_death_frame_support[i].astype(jnp.int32)
            
            caller_idx = state.enemy_support_caller_idx
            direction = jnp.where(
                state.enemy_attack_turning[caller_idx] == 0,
                jnp.where(state.enemy_attack_direction[caller_idx] == -1, 0, 1), # 0=left, 1=right
                2 # 2=down
            )
            is_down = direction == 2
            
            ex = jnp.round(state.enemy_support_x[i]).astype(jnp.int32)
            ey = jnp.round(state.enemy_support_y[i]).astype(jnp.int32)
            
            # Supporters are always row 4 (red)
            row_idx = 4 
            def down(r):
                sprite = self.stacked_dive_down_sprites[row_idx]
                return self.jr.render_at_clipped(r, ex, ey, sprite, flip_offset=self.FLIP_OFFSETS['enemy_dive_down_green'])
            
            def side(r):
                sprite = self.stacked_dive_left_sprites[row_idx]
                return self.jr.render_at_clipped(r, ex, ey, sprite, 
                                         flip_horizontal=(direction == 1), # Flip if moving right
                                         flip_offset=self.FLIP_OFFSETS['enemy_dive_left_green'])
            
            def dying(r):
                sprite = self.SHAPE_MASKS['enemy_death'][death_frame_support - 1]
                return self.jr.render_at_clipped(r, ex, ey, sprite, flip_offset=self.FLIP_OFFSETS['enemy_death'])

            def dead(r):
                return r

            r_out = jax.lax.cond(is_alive, lambda r: jax.lax.cond(is_down, down, side, r), lambda r: jax.lax.cond(is_dying, dying, dead, r), r_acc)
            return r_out

        raster = jax.lax.cond(
            state.enemy_support_caller_idx != self.consts.ERROR_VALUE,
            lambda r: jax.lax.fori_loop(0, self.consts.MAX_SUPPORTERS, _draw_single_supporter, r),
            lambda r: r,
            raster
        )

        # 7. Enemy Grid
        def row_body(i, r_acc):
            def col_body(j, r_inner):
                death_frame_grid = state.enemy_death_frame_grid[i, j].astype(jnp.int32)
                alive = (state.enemy_grid_state[i, j] == 1)
                
                x = jnp.round(state.enemy_grid_x[i, j]).astype(jnp.int32)
                y = jnp.round(state.enemy_grid_y[i, j]).astype(jnp.int32)
                def draw_death(r0):
                    sprite = self.SHAPE_MASKS['enemy_death'][death_frame_grid - 1]
                    return self.jr.render_at_clipped(r0, x, y, sprite, flip_offset=self.FLIP_OFFSETS['enemy_death'])

                def draw_alive(r0):
                    sprite = self.stacked_grid_sprites[i] # Select sprite based on row `i`
                    return self.jr.render_at_clipped(r0, x, y, sprite, flip_offset=self.FLIP_OFFSETS['enemy_grid_gray']) # Offset is same

                return jax.lax.cond(
                    death_frame_grid > 0,
                    draw_death,
                    lambda r0: jax.lax.cond(alive, draw_alive, lambda r1: r1, r0),
                    r_inner
                )

            return jax.lax.fori_loop(0, self.consts.GRID_COLS, col_body, r_acc)
        
        raster = jax.lax.fori_loop(0, self.consts.GRID_ROWS, row_body, raster)

        # 8. Life Icons
        life_sprite = self.SHAPE_MASKS['life']
        x0 = 10 + self.life_w + self.life_spacing
        y0 = self.consts.NATIVE_GAME_HEIGHT - self.life_h - self.life_spacing - 10
        raster = self.jr.render_indicator(
            raster, x0, y0, 
            state.lives, 
            life_sprite, 
            spacing=self.life_w + self.life_spacing, 
            max_value=self.consts.LIVES
        )
        
        # 9. Score
        digits = self.jr.int_to_digits(state.score, max_digits=5)
        digit_sprites = self.SHAPE_MASKS['digits']
        
        def score_loop_body(i, r_acc):
            digit_to_render = digits[i]
            x_pos = jnp.int32(self.consts.NATIVE_GAME_WIDTH - 100 - (i + 1) * 10)
            y_pos = jnp.int32(5)
            digit_mask = digit_sprites[digit_to_render]
            return self.jr.render_at(r_acc, x_pos, y_pos, digit_mask, flip_offset=self.FLIP_OFFSETS['digits'])
            
        raster = jax.lax.fori_loop(0, 5, score_loop_body, raster)

        # 10. Final Palette Lookup
        return self.jr.render_from_palette(raster, self.PALETTE)