import os
from functools import partial
from typing import NamedTuple, Tuple
import jax.lax
import jax.numpy as jnp
import chex

import src.jaxatari.spaces as spaces
from src.jaxatari.renderers import JAXGameRenderer
from src.jaxatari.rendering import jax_rendering_utils as jr
from src.jaxatari.environment import JaxEnvironment, JAXAtariAction as Action

#TODO: Constants
class TetrisConstants(NamedTuple):
    BOARD_WIDTH: int = 10
    BOARD_HEIGHT: int = 22
    BOARD_X: int = 21  # left margin
    BOARD_Y: int = 27  # top margin
    BOARD_PADDING: int = 2
    CELL_WIDTH: int = 3
    CELL_HEIGHT: int = 7
    DIGIT_X: int = 95
    DIGIT_Y: int = 27
    # Pygame window dimensions
    WINDOW_WIDTH: int = 160 * 3
    WINDOW_HEIGHT: int = 210 * 3
    WIDTH: int = 160
    HEIGHT: int = 210
    FPS: int = 10

# Tetris pieces
TETROMINOS = jnp.array([
    [  # I
        [[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]],
        [[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0]],
        [[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]],
        [[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0]]
    ],
    [  # O
        [[0,0,0,0],[0,1,1,0],[0,1,1,0],[0,0,0,0]]
    ] * 4,
    [  # T
        [[0,0,0,0],[0,1,0,0],[1,1,1,0],[0,0,0,0]],
        [[0,1,0,0],[0,1,1,0],[0,1,0,0],[0,0,0,0]],
        [[0,0,0,0],[1,1,1,0],[0,1,0,0],[0,0,0,0]],
        [[0,1,0,0],[1,1,0,0],[0,1,0,0],[0,0,0,0]]
    ],
    [  # S
        [[0,0,0,0],[0,1,1,0],[1,1,0,0],[0,0,0,0]],
        [[0,1,0,0],[0,1,1,0],[0,0,1,0],[0,0,0,0]]
    ] * 2,
    [  # Z
        [[0,0,0,0],[1,1,0,0],[0,1,1,0],[0,0,0,0]],
        [[0,0,1,0],[0,1,1,0],[0,1,0,0],[0,0,0,0]]
    ] * 2,
    [  # J
        [[0,0,0,0],[1,0,0,0],[1,1,1,0],[0,0,0,0]],
        [[0,1,0,0],[0,1,0,0],[1,1,0,0],[0,0,0,0]],
        [[0,0,0,0],[1,1,1,0],[0,0,1,0],[0,0,0,0]],
        [[0,1,1,0],[0,1,0,0],[0,1,0,0],[0,0,0,0]]
    ],
    [  # L
        [[0,0,0,0],[0,0,1,0],[1,1,1,0],[0,0,0,0]],
        [[1,1,0,0],[0,1,0,0],[0,1,0,0],[0,0,0,0]],
        [[0,0,0,0],[1,1,1,0],[1,0,0,0],[0,0,0,0]],
        [[0,1,0,0],[0,1,0,0],[0,1,1,0],[0,0,0,0]]
    ]
], dtype=jnp.int32)

# Constants Board
BOARD_WIDTH = 10
BOARD_HEIGHT = 20

# immutable state container
class TetrisState(NamedTuple):
    board: chex.Array           # Game Board, height x width
    #TODO: how much pieces left?
    current_piece_x: chex.Array
    current_piece_y: chex.Array
    current_piece_shape: chex.Array         # given in index
    current_piece_rotation: chex.Array      # given in index
    next_piece_shape: chex.Array            # given in index
    score: chex.Array

class PieceInfo(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    shape: jnp.ndarray
    rotation: jnp.ndarray

class TetrisObservation(NamedTuple):
    board: chex.Array
    current_piece: PieceInfo
    next_piece: chex.Array
    score: chex.Array

class TetrisInfo(NamedTuple):
    time: jnp.ndarray
    all_rewards: chex.Array #TODO: for what?

@jax.jit
# TODO: logic about human Input
# TODO: use x + y or just wrapper?
# TODO: implement is_valid
def player_step(state: TetrisState, action: chex.Array) -> TetrisState:
    # 1. handle rotation
    new_rotation = jax.lax.cond(
        action == Action.FIRE,
        lambda r: (r+1) % 4,
        lambda r: r,
        operand = state.current_piece_rotation,
        # if action == Action.FIRE:
        #     new_rotation = (rotation + 1) % 4
        # else:
        #     new_rotation = rotation
    )

    # 2. handle left+right
    left = action == Action.LEFT
    right = action == Action.RIGHT
    delta_x = jnp.where(left, -1, jnp.where(right, 1, 0))
    # if move_left:
    #    delta_x = -1
    # elif move_right:
    #    delta_x = 1
    # else:
    #    delta_x = 0
    new_x = state.current_piece_x + delta_x

    # 3. check move+rotation validity
    is_move_valid = is_valid(state.board, new_x, state.current_piece_y, state.current_piece_shape, new_rotation)
    # if illegal then restore
    x = jax.lax.select(is_move_valid, new_x, state.current_piece_x)
    rotation = jax.lax.select(is_move_valid, new_rotation, state.current_piece_rotation)

    # 4. move down
    delta_y = jnp.where(action == Action.DOWN, 2, 1)
    new_y = state.current_piece_y + delta_y
    is_down_valid = is_valid(state.board, x, new_y , state.current_piece_shape, rotation) #TODO: new or old?



def piece_step(
    # TODO: logic about falling down and collision check

):

# no Enemy_step

@jax.jit
def _reset(

):
    #TODO

class JaxTetris(JaxEnvironment[TetrisState, TetrisObservation, TetrisInfo]):
    def __init__(self, reward_funcs: list[callable]=None):
        super().__init__()
        self.renderer = TetrisRenderer()#TODO
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.action_set = [
            Action.NOOP,
            Action.FIRE,
            Action.RIGHT,
            Action.LEFT,
            Action.DOWN
        ]
        self.obs_size = #TODO

    def reset(self, key=None) -> Tuple[TetrisObservation, TetrisState]:
        """
        Resets the game state to the initial state.
        Returns the initial state and the reward.
        """
        state = TetrisState(
            #TODO
        )
        initial_obs = self._get_observation(state)

        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: TetrisState, action: chex.Array) -> Tuple[TetrisObservation, TetrisState, float, bool, TetrisInfo]: #TODO: why float+bool?

    def render(self, state: TetrisState) -> jnp.ndarray:
        return self.renderer.render(state)

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: TetrisState):
        #TODO


    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: TetrisObservation) -> jnp.ndarray:
        return jnp.concatenate([
            #TODO
        ])

    def action_space(self) -> spaces.Discrete:
        """
        Returns the action space for Tetris.
        Actions are:
        0: NOOP
        1: FIRE
        2: RIGHT
        3: LEFT
        4: DOWN
        """
        return spaces.Discrete(5)

    def observation_space(self) -> spaces:
        """
        Returns the observation space for Tetris.
        The Observation contains:
         # TODO:
        """
        return spaces.Dict({
            #TODO
        })

    def image_space(self) -> spaces.Box:
        """
        Returns the image space for Tetris.
        """
        return spaces.Box(
            #TODO
        )


    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: TetrisState, all_rewards: chex.Array) -> TetrisInfo:
        #TODO


    @partial(jax.jit, static_argnums=(0,))
    def _get_env_reward(self, previous_state: TetrisState, state: TetrisState):
        #TODO



    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(self, previous_state: TetrisState, state: TetrisState):
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array(
            [reward_func(previous_state, state) for reward_func in self.reward_funcs]
        )
        return rewards


    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: TetrisState) -> bool:
        #TODO


class TetrisRenderer(JAXGameRenderer):
    def __init__(self, consts: TetrisConstants = None):
        super().__init__()
        self.consts = consts or TetrisConstants()
        (
            self.SPRITE_BG,
            self.SPRITE_BOARD,
            self.SCORE_DIGIT_SPRITES,
            self.SPRITE_ROW_COLORS,
        ) = self.load_sprites()

    def load_sprites(self):
        """Load all sprites required for Tetris rendering."""
        MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

        # Load sprites
        bg = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/tetris/background.npy"), transpose=True)
        board = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/tetris/board.npy"), transpose=True)

        # Convert all sprites to the expected format (add frame dimension)
        SPRITE_BG = jnp.expand_dims(bg, axis=0)
        SPRITE_BOARD = jnp.expand_dims(board, axis=0)

        # Load digits for scores
        SCORE_DIGIT_SPRITES = jr.load_and_pad_digits(
            os.path.join(MODULE_DIR, "sprites/tetris/score/score_{}.npy"),
            num_chars=10,
        )

        # Colors for tetris pieces on the board
        row_squares = []
        for i in range(22):  # assuming 22 rows
            sprite = jr.loadFrame(os.path.join(MODULE_DIR, f"sprites/tetris/height_colors/h_{i}.npy"))
            row_squares.append(sprite)

        SPRITE_ROW_COLORS = jnp.stack(row_squares, axis=0)  # Shape: (22, H, W, 4)

        return (
            SPRITE_BG,
            SPRITE_BOARD,
            SCORE_DIGIT_SPRITES,
            SPRITE_ROW_COLORS
        )

    def render(self, state):
        raster = jr.create_initial_frame(width=160, height=210)

        frame_bg = jr.get_sprite_frame(self.SPRITE_BG, 0)
        raster = jr.render_at(raster, 0, 0, frame_bg)

        frame_board = jr.get_sprite_frame(self.SPRITE_BOARD, 0)
        raster = jr.render_at(raster, self.BOARD_X, self.BOARD_Y, frame_board)

        board = state.board

        num_rows = board.shape[0] # 22
        num_cols = board.shape[1] # 10

        def render_board_row(row_idx, raster):
            row = board[row_idx]
            sprite = self.SPRITE_ROW_COLORS[row_idx % len(self.SPRITE_ROW_COLORS)]

            def render_col(col_idx, raster):
                val = row[col_idx]

                def draw_sprite(r):
                    x = self.BOARD_X + self.BOARD_PADDING + col_idx * (self.CELL_WIDTH + 1)
                    y = self.BOARD_Y + row_idx * (self.CELL_HEIGHT + 1)
                    return jr.render_at(r, x, y, sprite)

                return jax.lax.cond(jnp.equal(val, 1), draw_sprite, lambda r: r, raster)

            return jax.lax.fori_loop(0, num_cols, render_col, raster)

        raster = jax.lax.fori_loop(0, num_rows, render_board_row, raster)

        #render current falling piece using row sprites
        piece = self.get_piece_shape(int(state.current_piece), int(state.current_rotation))  # shape (4, 4)
        pos_y, pos_x = state.current_position  # shape (2,)

        def render_piece_cell(i, raster):
            y = i // 4
            x = i % 4

            val = piece[y, x]

            def draw_piece(r):
                board_y = pos_y + y
                board_x = pos_x + x

                in_bounds_y = jnp.logical_and(board_y >= 0, board_y < num_rows)
                in_bounds_x = jnp.logical_and(board_x >= 0, board_x < num_cols)
                in_bounds = jnp.logical_and(in_bounds_y, in_bounds_x)

                def render_pixel(r):
                    sprite = self.SPRITE_ROW_COLORS[board_y % len(self.SPRITE_ROW_COLORS)]
                    px = self.BOARD_X + self.BOARD_PADDING + board_x * (self.CELL_WIDTH + 1)
                    py = self.BOARD_Y + board_y * (self.CELL_HEIGHT + 1)
                    return jr.render_at(r, px, py, sprite)

                return jax.lax.cond(in_bounds, render_pixel, lambda r: r, r)

            return jax.lax.cond(val == 1, draw_piece, lambda r: r, raster)

        raster = jax.lax.fori_loop(0, 16, render_piece_cell, raster)


        # get score digits with zero-padding (always 4 digits)
        score_digits = jr.int_to_digits(state.score, max_digits=4)
        raster = jr.render_label_selective(
            raster,
            95,  # x position for the most left digit
            27,  # y position
            score_digits,
            self.SCORE_DIGIT_SPRITES,
            start_index=0,
            num_to_render=4,
            spacing=16  # each digit offset by 16 px
        )

        return raster