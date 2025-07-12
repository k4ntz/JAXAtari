# Author: Nil Erdal
# Author: Emir Tavukcu
# Author: Fanwei Kong

import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple
import chex
from jax import lax
import os

from src.jaxatari.renderers import JAXGameRenderer
from src.jaxatari.rendering import jax_rendering_utils as jr
from src.jaxatari.environment import JaxEnvironment, JAXAtariAction as Action

# Game Constants
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

# immutable state container
class TetrisState(NamedTuple):
    board: chex.Array
    current_piece: chex.Array
    current_position: chex.Array
    current_rotation: chex.Array
    next_piece: chex.Array
    score: chex.Array
    is_game_over: chex.Array
    rng_key: chex.Array

class TetrisObservation(NamedTuple):
    board: jnp.ndarray
    current_piece: jnp.ndarray
    current_position: jnp.ndarray
    current_rotation: jnp.ndarray
    next_piece: jnp.ndarray

class TetrisInfo(NamedTuple):
    score: jnp.ndarray
    is_game_over: jnp.ndarray


class JaxTetris(JaxEnvironment[TetrisState, TetrisObservation, TetrisInfo, TetrisConstants]):
    def __init__(self, consts: TetrisConstants = None, reward_funcs: list[callable] = None):
        consts = consts or TetrisConstants()
        super().__init__(consts)
        self.renderer = TetrisRenderer(self.consts)
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
        self.obs_size = 3 * 4 + 1 + 1

    def reset(self, rng_key: jax.random.PRNGKey) -> TetrisObservation:
        self.rng_key = rng_key
        self.state = self.init_state(self.rng_key)
        return self._get_observation()

    def step(self, action: jnp.ndarray, rotate: jnp.bool_) -> Tuple[TetrisState, TetrisObservation, jnp.float_, jnp.bool_, TetrisInfo]:
        self.state, reward, done, info = self.tetris_step(self.state, action, rotate)
        obs = self._get_observation()
        return self.state, obs, reward, done, info

    def _get_observation(self) -> TetrisObservation:
        return TetrisObservation(
            board=self.state.board,
            current_piece=self.state.current_piece,
            current_position=self.state.current_position,
            current_rotation=self.state.current_rotation,
            next_piece=self.state.next_piece
        )

    def init_state(self, rng_key):
        """
        Initialize the Tetris game state.

        Args:
            rng_key (jax.random.PRNGKey): Random number generator key.

        Returns:
            TetrisState: The initial game state containing an empty board, current piece,
                         position, rotation, next piece, score, game over flag, and RNG key.
        """
        board = jnp.zeros((self.BOARD_HEIGHT, self.BOARD_WIDTH), dtype=jnp.int32)
        piece_type, pos, rot, rng_key = self.spawn_piece(rng_key)
        next_piece, _, _, rng_key = self.spawn_piece(rng_key)
        return TetrisState(board, piece_type, pos, rot, next_piece, jnp.array(0), jnp.array(False), rng_key)

    def spawn_piece(self, rng_key):
        """
        Spawn a new tetromino piece.

        Args:
            rng_key (jax.random.PRNGKey): Random number generator key.

        Returns:
            piece_type (int): Index of the tetromino type.
            pos (jnp.ndarray): Initial position of the piece (y, x).
            rotation (int): Initial rotation index (0-3).
            rng_key (jax.random.PRNGKey): Updated RNG key after splitting.
        """
        rng_key, subkey = jax.random.split(rng_key)
        piece_type = jax.random.randint(subkey, (), 0, len(TETROMINOS))
        rotation = jnp.array(0)
        pos = jnp.array([0, 3])
        return piece_type, pos, rotation, rng_key

    def get_piece_shape(self, piece_type, rotation):
        """
        Get the shape matrix of the current tetromino piece given its type and rotation.

        Args:
            piece_type (int): Index of the tetromino type.
            rotation (int): Rotation index (0-3).

        Returns:
            jnp.ndarray: 4x4 binary matrix representing the tetromino shape.
        """
        return TETROMINOS[piece_type, rotation % 4]

    def check_collision(self, board, piece, position):
        """
        Check if the given piece at the specified position collides with the board boundaries
        or existing locked blocks.

        Args:
            board (jnp.ndarray): Current game board.
            piece (jnp.ndarray): 4x4 matrix representing the tetromino shape.
            position (jnp.ndarray): Position (y, x) to check.

        Returns:
            bool: True if collision occurs, False otherwise.
        """

        def check_cell(i, result):
            y, x = divmod(i, 4)
            piece_val = piece[y, x]
            py = position[0] + y
            px = position[1] + x
            cond = jnp.logical_and(piece_val == 1, (
                    (py >= self.BOARD_HEIGHT) | (px < 0) | (px >= self.BOARD_WIDTH) |
                    jnp.logical_and(py >= 0, board[py, px] == 1)
            ))
            return result | cond

        return jax.lax.fori_loop(0, 16, check_cell, False)

    def lock_piece(self, board, piece, position):
        """
        Lock a tetromino piece into the board by setting the occupied cells.

        Args:
            board (jnp.ndarray): Current game board.
            piece (jnp.ndarray): 4x4 matrix representing the tetromino shape.
            position (jnp.ndarray): Position (y, x) to lock the piece.

        Returns:
            jnp.ndarray: Updated board with the piece locked in place.
        """

        def update_cell(i, board):
            y, x = divmod(i, 4)
            py = position[0] + y
            px = position[1] + x
            return jax.lax.cond(
                piece[y, x] == 1,
                lambda b: b.at[py, px].set(1),
                lambda b: b,
                board
            )

        return jax.lax.fori_loop(0, 16, update_cell, board)

    def clear_lines(self, board):
        """
        Clear completed lines from the board and shift remaining rows down.

        Args:
            board (jnp.ndarray): Current game board.

        Returns:
            tuple:
                new_board (jnp.ndarray): Board after line clears.
                num_cleared (int): Number of lines cleared.
        """
        full_rows = jnp.all(board != 0, axis=1)
        num_cleared = jnp.sum(full_rows)

        def compress_board(board, full_rows):
            new_board = jnp.zeros_like(board)
            write_idx = board.shape[0] - 1

            def body(i, val):
                write_idx, new_board = val
                row_idx = board.shape[0] - 1 - i
                row_is_full = full_rows[row_idx]

                # skip the row if it is full
                def skip():
                    return write_idx, new_board

                def copy():
                    new_board_updated = new_board.at[write_idx].set(board[row_idx])
                    return write_idx - 1, new_board_updated

                return lax.cond(row_is_full, skip, copy)

            write_idx, new_board = lax.fori_loop(0, board.shape[0], body, (write_idx, new_board))
            return new_board

        new_board = compress_board(board, full_rows)
        return new_board, num_cleared

    @jax.jit
    def tetris_step(self, state: TetrisState, action: jnp.ndarray, rotate: jnp.array) -> Tuple[
        TetrisState, jnp.float_, jnp.bool_, TetrisInfo]:
        """
        Perform a single Tetris game step by applying action and rotation inputs.

        Args:
            state (TetrisState): Current game state.
            action (jnp.ndarray): Movement action as [dy, dx], e.g., down, left, right.
            rotate (bool): Whether to rotate the piece.

        Returns:
            tuple:
                new_state (TetrisState): Updated game state after the step.
                reward (float): Reward obtained in this step (lines cleared).
                done (bool): Whether the game is over.
                info (TetrisInfo): Additional info such as score and game-over flag.
        """
        dy, dx = action

        def apply_rotation(state):
            new_rotation = (state.current_rotation + 1) % 4
            rotated_piece = self.get_piece_shape(state.current_piece, new_rotation)
            collides = self.check_collision(state.board, rotated_piece, state.current_position)
            return lax.cond(collides,
                            lambda _: state,
                            lambda _: state._replace(current_rotation=new_rotation),
                            operand=None)

        state = lax.cond(rotate, apply_rotation, lambda s: s, state)

        piece = self.get_piece_shape(state.current_piece, state.current_rotation)
        new_pos = state.current_position + jnp.array([dy, dx])
        collides = self.check_collision(state.board, piece, new_pos)

        def try_move_down(state):
            vertical_pos = state.current_position + jnp.array([1, 0])
            piece = self.get_piece_shape(state.current_piece, state.current_rotation)
            collides_vert = self.check_collision(state.board, piece, vertical_pos)

            def lock_and_spawn(_):
                board = self.lock_piece(state.board, piece, state.current_position)
                board, cleared = self.clear_lines(board)
                score = jax.lax.cond(
                    cleared == 5,
                    lambda _: state.score + 10,
                    lambda _: state.score + cleared,
                    operand=None
                )
                reward = score * 1.0
                next_piece, _, _, new_rng = self.spawn_piece(state.rng_key)
                piece_type, pos, rot, new_rng = self.spawn_piece(new_rng)
                game_over = self.check_collision(board, self.get_piece_shape(piece_type, rot), pos)
                new_state = TetrisState(
                    board=board,
                    current_piece=piece_type,
                    current_position=pos,
                    current_rotation=rot,
                    next_piece=next_piece,
                    score=score,
                    is_game_over=game_over,
                    rng_key=new_rng
                )
                return new_state, reward, game_over

            def no_lock(_):
                return state, 0.0, False

            return lax.cond(collides_vert, lock_and_spawn, no_lock, operand=None)

        def move_piece(state):
            return state._replace(current_position=new_pos), 0.0, False

        state, reward, done = lax.cond(collides, try_move_down, move_piece, state)
        info = TetrisInfo(score=state.score, is_game_over=state.is_game_over)
        return state, reward, done, info



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