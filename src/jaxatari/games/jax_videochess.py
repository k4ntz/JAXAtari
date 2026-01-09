import os
from functools import partial
from typing import NamedTuple, Tuple, Callable, Optional
import importlib

import chex
import jax
import jax.numpy as jnp

import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.spaces import Space



class VideoChessConstants:
    # Piece encoding
    EMPTY = 0

    W_PAWN = 1
    W_KNIGHT = 2
    W_BISHOP = 3
    W_ROOK = 4
    W_QUEEN = 5
    W_KING = 6

    B_PAWN = 7
    B_KNIGHT = 8
    B_BISHOP = 9
    B_ROOK = 10
    B_QUEEN = 11
    B_KING = 12

    CURSOR = 13  # for rendering cursor as a "piece"

    COLOUR_WHITE = 0
    COLOUR_BLACK = 1

    WIDTH = 160
    HEIGHT = 210

    NUM_RANKS = 8
    NUM_FILES = 8

    OFFSET_X_BOARD = 12
    OFFSET_Y_BOARD = 50

    PHASE_SELECT_PIECE = 0
    PHASE_SELECT_TARGET = 1
    PHASE_GAME_OVER = 2

    ANIMATION_FRAME_RATE = 30

    MOVE_DELAY_FRAMES = 5  # legacy, not used at the moment
    LAST_MOVE_BLINK_FRAMES = 90  # ~3 seconds at 30 FPS
    BLINK_PERIOD = 6  # frames for cursor/piece blinking

    # reuse a very simple asset config: one board + a group of pieces
    ASSET_CONFIG = (
        {"name": "board", "type": "single", "file": "background.npy", "transpose": True},
        {"name": "pieces", "type": "group", "files": [f"pieces/{i}.npy" for i in range(14)]},
    )



class VideoChessState(NamedTuple):
    board: chex.Array          # (8,8) int32
    to_move: int               # 0=white,1=black
    game_phase: int
    cursor_pos: chex.Array     # (2,) row,col
    selected_square: chex.Array  # (2,) or [-1,-1]
    frame_counter: chex.Array
    winner: int                # -1, 0,1
    rng_key: chex.PRNGKey
    last_move_target: chex.Array  # (2,) coordinates of the last moved-to square, or [-1,-1]
    last_move_timer: chex.Array   # int: frames remaining for last-move blinking
    last_was_move: chex.Array  # boolean flag: was the last applied action a move?
    last_was_fire: chex.Array  # boolean flag: was the last applied action FIRE?


class VideoChessObservation(NamedTuple):
    board: chex.Array
    cursor_pos: chex.Array
    selected_square: chex.Array
    to_move: chex.Array


class VideoChessInfo(NamedTuple):
    pass



class BoardHandler:

    @staticmethod
    def reset_board() -> jnp.ndarray:
        """Standard chess start position."""
        c = VideoChessConstants
        board = jnp.zeros((c.NUM_RANKS, c.NUM_FILES), dtype=jnp.int32)

        # White pieces (your side, rendered as white sprites 1–6)
        white_back = jnp.array(
            [c.W_ROOK, c.W_KNIGHT, c.W_BISHOP, c.W_QUEEN,
             c.W_KING, c.W_BISHOP, c.W_KNIGHT, c.W_ROOK],
            dtype=jnp.int32,
        )
        white_pawns = jnp.full((c.NUM_FILES,), c.W_PAWN, dtype=jnp.int32)

        # Black pieces (opponent, rendered as orange sprites 7–12)
        black_back = jnp.array(
            [c.B_ROOK, c.B_KNIGHT, c.B_BISHOP, c.B_QUEEN,
             c.B_KING, c.B_BISHOP, c.B_KNIGHT, c.B_ROOK],
            dtype=jnp.int32,
        )
        black_pawns = jnp.full((c.NUM_FILES,), c.B_PAWN, dtype=jnp.int32)

        # Place black (opponent, orange) at the TOP (ranks 0 and 1)
        board = board.at[0, :].set(black_back)
        board = board.at[1, :].set(black_pawns)

        # Place white (YOU) at the BOTTOM (ranks 7 and 6)
        board = board.at[6, :].set(white_pawns)
        board = board.at[7, :].set(white_back)

        return board

    @staticmethod
    def in_bounds(row, col) -> bool:
        c = VideoChessConstants
        return (0 <= row) & (row < c.NUM_RANKS) & (0 <= col) & (col < c.NUM_FILES)

    @staticmethod
    def is_empty(board: jnp.ndarray, row, col) -> bool:
        return board[row, col] == VideoChessConstants.EMPTY

    @staticmethod
    def is_white(piece) -> bool:
        c = VideoChessConstants
        return (piece >= c.W_PAWN) & (piece <= c.W_KING)

    @staticmethod
    def is_black(piece) -> bool:
        c = VideoChessConstants
        return (piece >= c.B_PAWN) & (piece <= c.B_KING)

    @staticmethod
    def is_same_colour(piece, colour: int) -> bool:
        c = VideoChessConstants
        return jax.lax.cond(
            colour == c.COLOUR_WHITE,
            lambda: BoardHandler.is_white(piece),
            lambda: BoardHandler.is_black(piece),
        )

    @staticmethod
    def is_opponent(piece, colour: int) -> bool:
        c = VideoChessConstants
        return jax.lax.cond(
            colour == c.COLOUR_WHITE,
            lambda: BoardHandler.is_black(piece),
            lambda: BoardHandler.is_white(piece),
        )

    @staticmethod
    def legal_moves_for_colour(board: jnp.ndarray, from_sq: chex.Array, colour: chex.Array) -> jnp.ndarray:
        """Legal targets for `from_sq` for the given colour (padded to (56,2))."""
        c = VideoChessConstants

        row = from_sq[0]
        col = from_sq[1]

        piece = board[row, col]
        colour_scalar = jnp.int32(colour)

        def no_moves() -> jnp.ndarray:
            return jnp.full((56, 2), -1, dtype=jnp.int32)

        # Helper for sliding pieces: one ray in a given direction
        def ray_moves(dir_row: chex.Array, dir_col: chex.Array) -> jnp.ndarray:
            steps = jnp.arange(1, 8, dtype=jnp.int32)  # 1..7

            r2 = row + dir_row * steps
            c2 = col + dir_col * steps

            inb = BoardHandler.in_bounds(r2, c2)

            safe_r2 = jnp.clip(r2, 0, c.NUM_RANKS - 1)
            safe_c2 = jnp.clip(c2, 0, c.NUM_FILES - 1)
            target_all = board[safe_r2, safe_c2]
            target = jnp.where(inb, target_all, jnp.int32(c.EMPTY))

            same_colour = BoardHandler.is_same_colour(target, colour_scalar)
            any_piece = target != jnp.int32(c.EMPTY)

            cum = jnp.cumsum(any_piece.astype(jnp.int32))  # inclusive
            exclusive_cum = jnp.concatenate(
                [jnp.array([0], dtype=jnp.int32), cum[:-1]],
                axis=0,
            )
            blocked_before = (exclusive_cum > 0)

            legal = inb & (~blocked_before) & (~same_colour)

            coords = jnp.stack([r2, c2], axis=1)
            coords_masked = jnp.where(
                legal[:, None],
                coords.astype(jnp.int32),
                jnp.array([-1, -1], dtype=jnp.int32),
            )
            return coords_masked  # (7, 2)

        # Individual piece move generators
        def pawn_moves() -> jnp.ndarray:
            # White pawns move "up" (towards smaller row), black "down".
            is_white_turn = (colour_scalar == jnp.int32(c.COLOUR_WHITE))

            fwd = jax.lax.cond(
                is_white_turn,
                lambda: jnp.int32(-1),
                lambda: jnp.int32(1),
            )

            start_rank_row = jax.lax.cond(
                is_white_turn,
                lambda: jnp.int32(6),  # white pawns on row 6
                lambda: jnp.int32(1),  # black pawns on row 1
            )
            start_rank = (row == start_rank_row)

            moves = []

            f_row1 = row + fwd
            f_col = col
            inb1 = BoardHandler.in_bounds(f_row1, f_col)
            target_f1 = jax.lax.cond(
                inb1,
                lambda: board[f_row1, f_col],
                lambda: jnp.int32(c.EMPTY),
            )
            empty1 = (target_f1 == jnp.int32(c.EMPTY))

            move_f1 = jax.lax.cond(
                inb1 & empty1,
                lambda: jnp.stack([f_row1, f_col]).astype(jnp.int32),
                lambda: jnp.array([-1, -1], dtype=jnp.int32),
            )
            moves.append(move_f1)

            f_row2 = row + fwd * jnp.int32(2)
            inb2 = BoardHandler.in_bounds(f_row2, f_col)

            def can_two_step():
                target_f2 = board[f_row2, f_col]
                return empty1 & (target_f2 == jnp.int32(c.EMPTY))

            empty2 = jax.lax.cond(inb2 & start_rank, can_two_step, lambda: False)

            move_f2 = jax.lax.cond(
                inb2 & start_rank & empty2,
                lambda: jnp.stack([f_row2, f_col]).astype(jnp.int32),
                lambda: jnp.array([-1, -1], dtype=jnp.int32),
            )
            moves.append(move_f2)

            cap_l_row = row + fwd
            cap_l_col = col - jnp.int32(1)
            inb_cap_l = BoardHandler.in_bounds(cap_l_row, cap_l_col)
            target_l = jax.lax.cond(
                inb_cap_l,
                lambda: board[cap_l_row, cap_l_col],
                lambda: jnp.int32(c.EMPTY),
            )
            can_cap_l = inb_cap_l & BoardHandler.is_opponent(target_l, colour_scalar)

            move_cap_l = jax.lax.cond(
                can_cap_l,
                lambda: jnp.stack([cap_l_row, cap_l_col]).astype(jnp.int32),
                lambda: jnp.array([-1, -1], dtype=jnp.int32),
            )
            moves.append(move_cap_l)

            cap_r_row = row + fwd
            cap_r_col = col + jnp.int32(1)
            inb_cap_r = BoardHandler.in_bounds(cap_r_row, cap_r_col)
            target_r = jax.lax.cond(
                inb_cap_r,
                lambda: board[cap_r_row, cap_r_col],
                lambda: jnp.int32(c.EMPTY),
            )
            can_cap_r = inb_cap_r & BoardHandler.is_opponent(target_r, colour_scalar)

            move_cap_r = jax.lax.cond(
                can_cap_r,
                lambda: jnp.stack([cap_r_row, cap_r_col]).astype(jnp.int32),
                lambda: jnp.array([-1, -1], dtype=jnp.int32),
            )
            moves.append(move_cap_r)

            moves4 = jnp.stack(moves, axis=0)  # (4, 2)
            pad = jnp.full((56 - 4, 2), -1, dtype=jnp.int32)
            return jnp.concatenate([moves4, pad], axis=0)  # (56, 2)

        def knight_moves() -> jnp.ndarray:
            deltas = jnp.array(
                [
                    [-2, -1], [-2, 1],
                    [-1, -2], [-1, 2],
                    [1, -2],  [1, 2],
                    [2, -1],  [2, 1],
                ],
                dtype=jnp.int32,
            )

            def one(delta):
                r2 = row + delta[0]
                c2 = col + delta[1]

                inb = BoardHandler.in_bounds(r2, c2)
                target = jax.lax.cond(
                    inb,
                    lambda: board[r2, c2],
                    lambda: jnp.int32(c.EMPTY),
                )
                legal = inb & (~BoardHandler.is_same_colour(target, colour_scalar))

                return jax.lax.cond(
                    legal,
                    lambda: jnp.stack([r2, c2]).astype(jnp.int32),
                    lambda: jnp.array([-1, -1], dtype=jnp.int32),
                )

            moves8 = jax.vmap(one)(deltas)  # (8, 2)
            pad = jnp.full((56 - 8, 2), -1, dtype=jnp.int32)
            return jnp.concatenate([moves8, pad], axis=0)

        def bishop_moves() -> jnp.ndarray:
            dirs = [
                jnp.int32(-1), jnp.int32(-1),
                jnp.int32(-1), jnp.int32(1),
                jnp.int32(1),  jnp.int32(-1),
                jnp.int32(1),  jnp.int32(1),
            ]
            rays = []
            for i in range(0, 8, 2):
                dr = dirs[i]
                dc = dirs[i + 1]
                rays.append(ray_moves(dr, dc))  # each (7, 2)
            rays_cat = jnp.concatenate(rays, axis=0)  # (28, 2)
            pad = jnp.full((56 - 28, 2), -1, dtype=jnp.int32)
            return jnp.concatenate([rays_cat, pad], axis=0)

        def rook_moves() -> jnp.ndarray:
            dirs = [
                jnp.int32(-1), jnp.int32(0),
                jnp.int32(1),  jnp.int32(0),
                jnp.int32(0),  jnp.int32(-1),
                jnp.int32(0),  jnp.int32(1),
            ]
            rays = []
            for i in range(0, 8, 2):
                dr = dirs[i]
                dc = dirs[i + 1]
                rays.append(ray_moves(dr, dc))  # each (7, 2)
            rays_cat = jnp.concatenate(rays, axis=0)  # (28, 2)
            pad = jnp.full((56 - 28, 2), -1, dtype=jnp.int32)
            return jnp.concatenate([rays_cat, pad], axis=0)

        def queen_moves() -> jnp.ndarray:
            b = bishop_moves()[:28]  # (28,2)
            r = rook_moves()[:28]    # (28,2)
            return jnp.concatenate([b, r], axis=0)  # (56,2)

        def king_moves() -> jnp.ndarray:
            directions = jnp.array(
                [[-1, -1], [-1, 0], [-1, 1],
                 [0, -1],           [0, 1],
                 [1, -1],  [1, 0],  [1, 1]],
                dtype=jnp.int32,
            )

            def one_dir(delta):
                r2 = row + delta[0]
                c2 = col + delta[1]

                inb = BoardHandler.in_bounds(r2, c2)

                target = jax.lax.cond(
                    inb,
                    lambda: board[r2, c2],
                    lambda: jnp.int32(c.EMPTY),
                )

                legal = inb & (~BoardHandler.is_same_colour(target, colour_scalar))

                return jax.lax.cond(
                    legal,
                    lambda: jnp.stack([r2, c2]).astype(jnp.int32),
                    lambda: jnp.array([-1, -1], dtype=jnp.int32),
                )

            moves8 = jax.vmap(one_dir)(directions)  # (8,2)
            pad = jnp.full((56 - 8, 2), -1, dtype=jnp.int32)
            return jnp.concatenate([moves8, pad], axis=0)

        # Dispatch by piece code
        is_own = BoardHandler.is_same_colour(piece, colour_scalar)

        cW, cB = c, c  # shorthand

        is_pawn = (piece == jnp.int32(cW.W_PAWN)) | (piece == jnp.int32(cB.B_PAWN))
        is_knight = (piece == jnp.int32(cW.W_KNIGHT)) | (piece == jnp.int32(cB.B_KNIGHT))
        is_bishop = (piece == jnp.int32(cW.W_BISHOP)) | (piece == jnp.int32(cB.B_BISHOP))
        is_rook = (piece == jnp.int32(cW.W_ROOK)) | (piece == jnp.int32(cB.B_ROOK))
        is_queen = (piece == jnp.int32(cW.W_QUEEN)) | (piece == jnp.int32(cB.B_QUEEN))
        is_king = (piece == jnp.int32(cW.W_KING)) |(piece == jnp.int32(cB.B_KING))

        def moves_if_own():
            def for_pawn():
                return pawn_moves()

            def for_knight():
                return knight_moves()

            def for_bishop():
                return bishop_moves()

            def for_rook():
                return rook_moves()

            def for_queen():
                return queen_moves()

            def for_king():
                return king_moves()

            return jax.lax.cond(
                is_pawn,
                for_pawn,
                lambda: jax.lax.cond(
                    is_knight,
                    for_knight,
                    lambda: jax.lax.cond(
                        is_bishop,
                        for_bishop,
                        lambda: jax.lax.cond(
                            is_rook,
                            for_rook,
                            lambda: jax.lax.cond(
                                is_queen,
                                for_queen,
                                lambda: jax.lax.cond(
                                    is_king,
                                    for_king,
                                    no_moves,
                                ),
                            ),
                        ),
                    ),
                ),
            )

        return jax.lax.cond(
            is_own,
            moves_if_own,
            no_moves,
        )

    @staticmethod
    def legal_moves_for_black_piece(board: jnp.ndarray, from_sq: chex.Array) -> jnp.ndarray:
        """
        Backward-compatible wrapper for existing code that expects
        `legal_moves_for_black_piece`. Internally uses the colour-generic
        `legal_moves_for_colour` with COLOUR_BLACK.
        """
        return BoardHandler.legal_moves_for_colour(
            board,
            from_sq,
            jnp.int32(VideoChessConstants.COLOUR_BLACK),
        )

    @staticmethod
    def apply_move(board: jnp.ndarray, from_sq: chex.Array, to_sq: chex.Array) -> jnp.ndarray:
        """Move a piece and handle simple pawn promotion to a queen for both colours."""
        c = VideoChessConstants

        from_row = from_sq[0]
        from_col = from_sq[1]
        to_row = to_sq[0]
        to_col = to_sq[1]

        piece = board[from_row, from_col]

        # Check promotion conditions (JAX-friendly, no Python ifs on traced values)
        is_black_pawn_promo = (piece == jnp.int32(c.B_PAWN)) & (to_row == jnp.int32(0))
        is_white_pawn_promo = (piece == jnp.int32(c.W_PAWN)) & (to_row == jnp.int32(c.NUM_RANKS - 1))

        def promote_black():
            return jnp.int32(c.B_QUEEN)

        def promote_white():
            return jnp.int32(c.W_QUEEN)

        def keep_piece():
            return piece

        new_piece = jax.lax.cond(
            is_black_pawn_promo,
            lambda: promote_black(),
            lambda: jax.lax.cond(
                is_white_pawn_promo,
                lambda: promote_white(),
                keep_piece,
            ),
        )

        new_board = board.at[from_row, from_col].set(c.EMPTY)
        new_board = new_board.at[to_row, to_col].set(new_piece)
        return new_board


    @staticmethod
    def king_present(board: jnp.ndarray, colour: int) -> bool:
        """Return True if the given colour's king is still on the board."""
        c = VideoChessConstants
        king_code = jax.lax.cond(
            colour == c.COLOUR_WHITE,
            lambda: jnp.int32(c.W_KING),
            lambda: jnp.int32(c.B_KING),
        )
        return jnp.any(board == king_code)

    @staticmethod
    def has_any_legal_moves_for_colour(board: jnp.ndarray, colour: chex.Array) -> bool:
        """Return True if at least one piece of the given colour has any legal move.

        Uses legal_moves_for_colour() as a backend.
        """
        c = VideoChessConstants

        def body_fun(idx, carry):
            found = carry
            row = idx // c.NUM_FILES
            col = idx % c.NUM_FILES
            piece = board[row, col]

            def check_square(_):
                moves = BoardHandler.legal_moves_for_colour(
                    board,
                    jnp.array([row, col], dtype=jnp.int32),
                    colour,
                )
                # Any move where row >= 0 is considered a real move
                has_move = jnp.any(moves[:, 0] >= jnp.int32(0))
                return found | has_move

            return jax.lax.cond(
                BoardHandler.is_same_colour(piece, colour),
                check_square,
                lambda _ : found,
                operand=None,
            )

        total_squares = VideoChessConstants.NUM_RANKS * VideoChessConstants.NUM_FILES
        return jax.lax.fori_loop(
            0,
            total_squares,
            body_fun,
            jnp.bool_(False),
        )

    @staticmethod
    def has_any_legal_moves_for_black(board: jnp.ndarray) -> bool:
        """Backward-compatible wrapper for existing black-only callers."""
        return BoardHandler.has_any_legal_moves_for_colour(
            board,
            jnp.int32(VideoChessConstants.COLOUR_BLACK),
        )



class JaxVideoChess(
    JaxEnvironment[VideoChessState, VideoChessObservation, VideoChessInfo, VideoChessConstants]
):
    @partial(jax.jit, static_argnums=(0,))
    def _update_game_over(self, state: VideoChessState) -> VideoChessState:
        """End-of-game handling."""
        c = self.consts
        board = state.board

        white_king_present = BoardHandler.king_present(board, c.COLOUR_WHITE)
        black_king_present = BoardHandler.king_present(board, c.COLOUR_BLACK)

        def some_end(s):
            # Decide winner when at least one king is missing
            def compute_result():
                # Black wins if white king is gone and black king remains
                def black_wins():
                    return jnp.int32(c.COLOUR_BLACK)

                # White wins if black king is gone and white king remains
                def white_wins():
                    return jnp.int32(c.COLOUR_WHITE)

                # Otherwise (both gone) -> draw
                def draw():
                    return jnp.int32(-1)

                return jax.lax.cond(
                    (~white_king_present) & black_king_present,
                    lambda: black_wins(),
                    lambda: jax.lax.cond(
                        (~black_king_present) & white_king_present,
                        lambda: white_wins(),
                        draw,
                    ),
                )

            winner = compute_result()
            return s._replace(
                game_phase=c.PHASE_GAME_OVER,
                winner=winner,
            )

        kings_missing = (~white_king_present) | (~black_king_present)

        # If both kings are present, we may still have a stalemate-like situation for black
        def check_stalemate(s):
            has_moves_black = BoardHandler.has_any_legal_moves_for_black(board)
            return jax.lax.cond(
                has_moves_black,
                lambda ss: ss,
                lambda ss: ss._replace(
                    game_phase=c.PHASE_GAME_OVER,
                    winner=jnp.int32(-1),
                ),
                s,
            )

        return jax.lax.cond(
            kings_missing,
            some_end,
            check_stalemate,
            state,
        )
    def __init__(
        self,
        consts: VideoChessConstants = None,
        bot_mode: Optional[str] = None,
        bot_module: Optional[str] = None,
    ):
        consts = consts or VideoChessConstants()
        super().__init__(consts)
        self.renderer = VideoChessRenderer(self.consts)

        self.action_set = {
            Action.NOOP,
            Action.FIRE,
            Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT,
            Action.UPLEFT, Action.UPRIGHT, Action.DOWNLEFT, Action.DOWNRIGHT,
        }

        # Optional bot hook (used by mods or local testing)
        self.bot_mode = bot_mode  # None | "random" | "mods"

        @partial(jax.jit, static_argnums=(0,))
        def _bot_noop(_self, s: VideoChessState) -> VideoChessState:
            return s

        self._bot_step: Callable[[VideoChessState], VideoChessState] = lambda s: s

        # Built-in random bot (JAX-native)
        if bot_mode == "random":
            self._bot_step = lambda s: self._bot_random_move(s)

        # External bot (mods) loaded from a module, e.g. `jaxatari.games.videochess_mods`
        if bot_mode == "mods" or bot_module is not None:
            module_name = bot_module or "jaxatari.games.videochess_mods"
            mod = importlib.import_module(module_name)

            # Preferred: factory that can close over constants
            if hasattr(mod, "make_bot"):
                bot_fn = mod.make_bot(self.consts)
            elif hasattr(mod, "bot_step"):
                bot_fn = mod.bot_step
            else:
                raise AttributeError(
                    f"{module_name} must define either make_bot(consts) or bot_step(state, consts)"
                )

            # Normalize signature: bot_fn(state) -> state
            if hasattr(mod, "make_bot"):
                self._bot_step = bot_fn
            else:
                # Expect bot_step(state, consts)
                self._bot_step = lambda s: bot_fn(s, self.consts)

        # If no bot selected, keep a no-op (so step stays simple)
        if self.bot_mode is None:
            self._bot_step = lambda s: _bot_noop(self, s)
    @partial(jax.jit, static_argnums=(0,))
    def _bot_random_move(self, state: VideoChessState) -> VideoChessState:
        """One random move for the side to move."""
        c = self.consts

        # Only act in normal play and only when nothing is selected.
        can_act = (
            (state.game_phase == c.PHASE_SELECT_PIECE)
            & (state.selected_square[0] < 0)
        )

        def do_bot(s: VideoChessState) -> VideoChessState:
            key, key_sample = jax.random.split(s.rng_key)

            # Generate all 64 from-squares.
            idx = jnp.arange(c.NUM_RANKS * c.NUM_FILES, dtype=jnp.int32)
            from_rows = idx // c.NUM_FILES
            from_cols = idx % c.NUM_FILES
            from_sq = jnp.stack([from_rows, from_cols], axis=1)  # (64,2)

            # Compute legal move targets for each square (64,56,2)
            moves = jax.vmap(lambda fs: BoardHandler.legal_moves_for_colour(s.board, fs, s.to_move))(from_sq)

            # A move is valid if target row >= 0
            valid = moves[..., 0] >= jnp.int32(0)  # (64,56)

            # Flatten all candidates.
            moves_flat = moves.reshape((-1, 2))          # (64*56,2)
            valid_flat = valid.reshape((-1,))            # (64*56,)

            # We also need the matching from-square for each candidate.
            from_rep = jnp.repeat(from_sq, repeats=moves.shape[1], axis=0)  # (64*56,2)

            valid_f = valid_flat.astype(jnp.float32)
            total = jnp.sum(valid_f)

            def no_move(ss: VideoChessState) -> VideoChessState:
                # No legal moves: just advance RNG.
                return ss._replace(rng_key=key)

            def pick_and_apply(ss: VideoChessState) -> VideoChessState:
                probs = valid_f / total
                pick = jax.random.choice(key_sample, probs.shape[0], p=probs, shape=())

                fs = from_rep[pick].astype(jnp.int32)
                ts = moves_flat[pick].astype(jnp.int32)

                new_board = BoardHandler.apply_move(ss.board, fs, ts)

                return ss._replace(
                    board=new_board,
                    to_move=jnp.int32(1) - jnp.int32(ss.to_move),
                    game_phase=c.PHASE_SELECT_PIECE,
                    selected_square=jnp.array([-1, -1], dtype=jnp.int32),
                    last_move_target=ts,
                    last_move_timer=jnp.int32(c.LAST_MOVE_BLINK_FRAMES),
                    rng_key=key,
                )

            return jax.lax.cond(total <= 0.0, no_move, pick_and_apply, s)

        return jax.lax.cond(can_act, do_bot, lambda ss: ss, state)

    @partial(jax.jit, static_argnums=(0,))
    def _enumerate_all_legal_moves(self, state: VideoChessState):
        """Enumerate all legal moves for `state.to_move` (flattened + mask)."""
        c = self.consts

        idx = jnp.arange(c.NUM_RANKS * c.NUM_FILES, dtype=jnp.int32)
        from_rows = idx // c.NUM_FILES
        from_cols = idx % c.NUM_FILES
        from_sq = jnp.stack([from_rows, from_cols], axis=1)  # (64,2)

        moves = jax.vmap(lambda fs: BoardHandler.legal_moves_for_colour(state.board, fs, state.to_move))(from_sq)  # (64,56,2)
        valid = moves[..., 0] >= jnp.int32(0)  # (64,56)

        to_flat = moves.reshape((-1, 2))
        valid_flat = valid.reshape((-1,))
        from_rep = jnp.repeat(from_sq, repeats=moves.shape[1], axis=0)

        return from_rep, to_flat, valid_flat

    @partial(jax.jit, static_argnums=(0,))
    def _bot_greedy_move(self, state: VideoChessState) -> VideoChessState:
        """Greedy move: prefer high-value captures, else random."""
        c = self.consts

        can_act = (
            (state.game_phase == c.PHASE_SELECT_PIECE)
            & (state.selected_square[0] < 0)
        )

        # Piece values indexed by piece-id 0..12
        # (values are intentionally simple; tune later)
        piece_vals = jnp.array(
            [0, 1, 3, 3, 5, 9, 200, 1, 3, 3, 5, 9, 200],
            dtype=jnp.int32,
        )

        def do_bot(s: VideoChessState) -> VideoChessState:
            key, k_noise, k_fallback = jax.random.split(s.rng_key, 3)

            from_rep, to_flat, valid_flat = self._enumerate_all_legal_moves(s)

            # Compute capture score for each candidate
            r = jnp.clip(to_flat[:, 0], 0, c.NUM_RANKS - 1)
            cc = jnp.clip(to_flat[:, 1], 0, c.NUM_FILES - 1)
            target_piece = s.board[r, cc]

            # Only count captures of opponent pieces
            is_cap = BoardHandler.is_opponent(target_piece, s.to_move)
            cap_val = piece_vals[target_piece]
            cap_score = jnp.where(is_cap, cap_val, jnp.int32(0))

            # Mask invalid moves
            cap_score = jnp.where(valid_flat, cap_score, jnp.int32(-1))

            best_cap = jnp.max(cap_score)
            has_capture = best_cap > jnp.int32(0)

            def apply_move(ss: VideoChessState, fs: chex.Array, ts: chex.Array, out_key: chex.PRNGKey) -> VideoChessState:
                new_board = BoardHandler.apply_move(ss.board, fs, ts)
                return ss._replace(
                    board=new_board,
                    to_move=jnp.int32(1) - jnp.int32(ss.to_move),
                    game_phase=c.PHASE_SELECT_PIECE,
                    selected_square=jnp.array([-1, -1], dtype=jnp.int32),
                    last_move_target=ts,
                    last_move_timer=jnp.int32(c.LAST_MOVE_BLINK_FRAMES),
                    rng_key=out_key,
                )

            def pick_best_capture(ss: VideoChessState) -> VideoChessState:
                # Add tiny noise to break ties deterministically but not repetitively.
                noise = (jax.random.uniform(k_noise, cap_score.shape, minval=0.0, maxval=1.0) * 1e-3)
                score_f = cap_score.astype(jnp.float32) + noise
                pick = jnp.argmax(score_f)
                fs = from_rep[pick].astype(jnp.int32)
                ts = to_flat[pick].astype(jnp.int32)
                return apply_move(ss, fs, ts, key)

            def fallback_random(ss: VideoChessState) -> VideoChessState:
                # Uniform over all valid moves
                valid_f = valid_flat.astype(jnp.float32)
                total = jnp.sum(valid_f)

                def no_move(sss):
                    return sss._replace(rng_key=key)

                def pick(sss):
                    probs = valid_f / total
                    pick_idx = jax.random.choice(k_fallback, probs.shape[0], p=probs, shape=())
                    fs = from_rep[pick_idx].astype(jnp.int32)
                    ts = to_flat[pick_idx].astype(jnp.int32)
                    return apply_move(sss, fs, ts, key)

                return jax.lax.cond(total <= 0.0, no_move, pick, ss)

            return jax.lax.cond(has_capture, pick_best_capture, fallback_random, s)

        return jax.lax.cond(can_act, do_bot, lambda ss: ss, state)

    @partial(jax.jit, static_argnums=(0,))
    def greedy_black_reply(self, prev_state: VideoChessState, new_state: VideoChessState) -> VideoChessState:
        """Reply move for BLACK (greedy)."""
        c = self.consts

        player_just_moved = (
            (prev_state.to_move == c.COLOUR_WHITE)
            & (new_state.to_move == c.COLOUR_BLACK)
            & (new_state.game_phase == c.PHASE_SELECT_PIECE)
            & (new_state.selected_square[0] < 0)
        )

        def do_bot(s: VideoChessState) -> VideoChessState:
            forced = s._replace(to_move=jnp.int32(c.COLOUR_BLACK))
            moved = self._bot_greedy_move(forced)
            # Keep human cursor position (bot moves invisibly)
            return moved._replace(to_move=jnp.int32(c.COLOUR_WHITE), cursor_pos=s.cursor_pos)

        return jax.lax.cond(player_just_moved, do_bot, lambda s: s, new_state)

    @partial(jax.jit, static_argnums=(0,))
    def random_black_reply(self, prev_state: VideoChessState, new_state: VideoChessState) -> VideoChessState:
        """Reply move for BLACK (random)."""
        c = self.consts

        player_just_moved = (
            (prev_state.to_move == c.COLOUR_WHITE)
            & (new_state.to_move == c.COLOUR_BLACK)
            & (new_state.game_phase == c.PHASE_SELECT_PIECE)
            & (new_state.selected_square[0] < 0)
        )

        def do_bot(s: VideoChessState) -> VideoChessState:
            # Reuse the random-move sampler, but force it to act as BLACK.
            # (We temporarily treat it as BLACK-to-move, then ensure turn returns to WHITE.)
            forced = s._replace(to_move=jnp.int32(c.COLOUR_BLACK))
            moved = self._bot_random_move(forced)

            # If a move happened, _bot_random_move will have toggled to_move.
            # We want: after bot move -> WHITE to play.
            # Keep the human cursor where it was (don't reveal bot move via cursor).
            return moved._replace(to_move=jnp.int32(c.COLOUR_WHITE), cursor_pos=s.cursor_pos)

        return jax.lax.cond(player_just_moved, do_bot, lambda s: s, new_state)

    def render(self, state: VideoChessState) -> jnp.ndarray:
        return self.renderer.render(state)

    # ---- reset ----
    def reset(self, key: jax.random.PRNGKey = jax.random.PRNGKey(0)) \
            -> Tuple[VideoChessObservation, VideoChessState]:
        c = self.consts
        board = BoardHandler.reset_board()
        state = VideoChessState(
            board=board,
            to_move=c.COLOUR_WHITE,
            game_phase=c.PHASE_SELECT_PIECE,
            cursor_pos=jnp.array([4, 3], dtype=jnp.int32),
            selected_square=jnp.array([-1, -1], dtype=jnp.int32),
            frame_counter=jnp.array(0, dtype=jnp.int32),
            winner=-1,
            rng_key=key,
            last_move_target=jnp.array([-1, -1], dtype=jnp.int32),
            last_move_timer=jnp.array(0, dtype=jnp.int32),
            last_was_move=jnp.array(False, dtype=jnp.bool_),
            last_was_fire=jnp.array(False, dtype=jnp.bool_),
        )

        obs = VideoChessObservation(
            board=state.board,
            cursor_pos=state.cursor_pos,
            selected_square=state.selected_square,
            to_move=jnp.array(state.to_move, dtype=jnp.int32),
        )

        return obs, state

    # ------------------------------------------------------------------
    #  Cursor movement helper: 8 directions (UP/DOWN/LEFT/RIGHT + diagonals)
    # ------------------------------------------------------------------
    @partial(jax.jit, static_argnums=(0,))
    def _move_cursor_basic(self, state: VideoChessState, action: chex.Array) -> VideoChessState:
        """Move cursor one square in 8 directions based on Atari actions."""
        up = (action == Action.UP) | (action == Action.UPLEFT) | (action == Action.UPRIGHT)
        down = (action == Action.DOWN) | (action == Action.DOWNLEFT) | (action == Action.DOWNRIGHT)
        left = (action == Action.LEFT) | (action == Action.UPLEFT) | (action == Action.DOWNLEFT)
        right = (action == Action.RIGHT) | (action == Action.UPRIGHT) | (action == Action.DOWNRIGHT)

        drow = jax.lax.cond(
            up,
            lambda: jnp.int32(-1),
            lambda: jax.lax.cond(down, lambda: jnp.int32(1), lambda: jnp.int32(0)),
        )
        dcol = jax.lax.cond(
            left,
            lambda: jnp.int32(-1),
            lambda: jax.lax.cond(right, lambda: jnp.int32(1), lambda: jnp.int32(0)),
        )

        any_move = up | down | left | right

        row = state.cursor_pos[0]
        col = state.cursor_pos[1]

        new_row = row + drow
        new_col = col + dcol

        in_bounds = BoardHandler.in_bounds(new_row, new_col)

        new_cursor = jax.lax.cond(
            any_move & in_bounds,
            lambda: jnp.stack([new_row, new_col]).astype(jnp.int32),
            lambda: state.cursor_pos,
        )

        return state._replace(cursor_pos=new_cursor)

    # ---- main step ----
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: VideoChessState, action: chex.Array):
        """Step with edge-triggered input (held keys don’t repeat instantly)."""
        def run_phase(s):
            return jax.lax.switch(
                s.game_phase,
                [
                    lambda st: self._step_select_piece_phase(st, action),   # phase 0
                    lambda st: self._step_select_target_phase(st, action),  # phase 1
                    lambda st: self._step_game_over_phase(st, action),      # phase 2
                ],
                s,
            )

        is_fire = (action == Action.FIRE)
        is_move = (action != Action.NOOP) & (action != Action.FIRE)

        def apply_step(s):
            ns = run_phase(s)
            return ns._replace(
                last_was_move=is_move,
                last_was_fire=is_fire,
            )

        def maybe_apply_non_fire(s):
            # For non-FIRE actions: apply only if this is a move and the last applied action was not a move
            cond = is_move & (~s.last_was_move)
            return jax.lax.cond(
                cond,
                apply_step,
                lambda x: x,
                s,
            )

        # FIRE is applied only if previous applied action was not FIRE
        def fire_or_not(s):
            cond = is_fire & (~s.last_was_fire)
            return jax.lax.cond(
                cond,
                apply_step,
                maybe_apply_non_fire,
                s,
            )

        new_state = fire_or_not(state)

        player_just_moved = (
            (state.to_move == self.consts.COLOUR_WHITE)
            & (new_state.to_move == self.consts.COLOUR_BLACK)
            & (new_state.game_phase == self.consts.PHASE_SELECT_PIECE)
            & (new_state.selected_square[0] < 0)
        )

        new_state = jax.lax.cond(
            player_just_moved,
            lambda s: self._bot_step(s),
            lambda s: s,
            new_state,
        )

        def reset_flags(s):
            return s._replace(
                last_was_move=jnp.array(False, dtype=jnp.bool_),
                last_was_fire=jnp.array(False, dtype=jnp.bool_),
            )

        new_state = jax.lax.cond(
            action == Action.NOOP,
            reset_flags,
            lambda s: s,
            new_state,
        )

        new_state = new_state._replace(
            frame_counter=(new_state.frame_counter + 1) % self.consts.ANIMATION_FRAME_RATE
        )

        new_state = new_state._replace(
            last_move_timer=jax.lax.max(
                jnp.int32(0),
                new_state.last_move_timer - jnp.int32(1),
            )
        )

        new_state = self._update_game_over(new_state)

        done = self._get_done(new_state)
        reward = self._get_env_reward(state, new_state)
        info = self._get_info(new_state)
        obs = self._get_observation(new_state)

        return obs, new_state, reward, done, info

    # ---- phases ----
    @partial(jax.jit, static_argnums=(0,))
    def _step_select_piece_phase(self, state: VideoChessState, action: chex.Array) -> VideoChessState:
        """
        Phase 0: move cursor and select which piece (of the side to move) to select.
        """
        def select_piece(s: VideoChessState) -> VideoChessState:
            row, col = s.cursor_pos
            piece = s.board[row, col]
            is_own = BoardHandler.is_same_colour(piece, s.to_move)
            return jax.lax.cond(
                is_own,
                lambda st: st._replace(
                    selected_square=st.cursor_pos,
                    game_phase=self.consts.PHASE_SELECT_TARGET,
                ),
                lambda st: st,
                s,
            )

        def move_cursor(s: VideoChessState) -> VideoChessState:
            return self._move_cursor_basic(s, action)

        return jax.lax.cond(
            action == Action.FIRE,
            select_piece,
            move_cursor,
            state
        )

    @partial(jax.jit, static_argnums=(0,))
    def _step_select_target_phase(self, state: VideoChessState, action: chex.Array) -> VideoChessState:
        """
        Phase 1: a piece is selected; move cursor to target square and FIRE to move.
        FIRE on the original square deselects.
        """
        def try_move(st: VideoChessState) -> VideoChessState:
            legal_targets = BoardHandler.legal_moves_for_colour(
                st.board,
                st.selected_square,
                st.to_move,
            )
            is_legal = jnp.any(jnp.all(legal_targets == st.cursor_pos, axis=1))

            def do_move():
                from_sq = st.selected_square
                to_sq = st.cursor_pos

                new_board = BoardHandler.apply_move(st.board, from_sq, to_sq)
                return st._replace(
                    board=new_board,
                    selected_square=jnp.array([-1, -1], dtype=jnp.int32),
                    game_phase=self.consts.PHASE_SELECT_PIECE,
                    to_move=jnp.int32(1) - jnp.int32(st.to_move),
                    cursor_pos=from_sq,
                    last_move_target=to_sq,
                    last_move_timer=jnp.int32(self.consts.LAST_MOVE_BLINK_FRAMES),
                )

            return jax.lax.cond(is_legal, do_move, lambda: st)

        def deselect_or_stay(st: VideoChessState) -> VideoChessState:
            return st._replace(
                selected_square=jnp.array([-1, -1], dtype=jnp.int32),
                game_phase=self.consts.PHASE_SELECT_PIECE,
            )

        def apply_or_deselect(st: VideoChessState) -> VideoChessState:
            same = jnp.all(st.cursor_pos == st.selected_square)

            # Check if the cursor is currently on one of our own pieces (side to move)
            row, col = st.cursor_pos
            piece = st.board[row, col]
            is_own = BoardHandler.is_same_colour(piece, st.to_move)

            def reselect(st2: VideoChessState) -> VideoChessState:
                # Simply change which piece is selected, stay in target phase
                return st2._replace(selected_square=st2.cursor_pos)

            # If cursor is back on the original square: deselect.
            # Otherwise: if on our own piece, reselect that piece; else, attempt a move.
            return jax.lax.cond(
                same,
                deselect_or_stay,
                lambda st2: jax.lax.cond(
                    is_own,
                    reselect,
                    try_move,
                    st2,
                ),
                st,
            )

        def move_cursor(s: VideoChessState) -> VideoChessState:
            return self._move_cursor_basic(s, action)

        return jax.lax.cond(
            action == Action.FIRE,
            apply_or_deselect,
            move_cursor,
            state
        )

    @staticmethod
    def _step_game_over_phase(state: VideoChessState, action: chex.Array) -> VideoChessState:
        del action
        return state

    # ---- spaces & helpers ----
    def action_space(self) -> Space:
        # we now have NOOP + FIRE + 8 directions = 10 actions
        return spaces.Discrete(10)

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0, high=255,
            shape=(self.consts.HEIGHT, self.consts.WIDTH, 3),
            dtype=jnp.uint8,
        )

    def observation_space(self) -> Space:
        c = self.consts
        return spaces.Dict({
            "board": spaces.Box(low=0, high=13, shape=(c.NUM_RANKS, c.NUM_FILES), dtype=jnp.int32),
            "cursor_pos": spaces.Box(low=0, high=7, shape=(2,), dtype=jnp.int32),
            "selected_square": spaces.Box(low=-1, high=7, shape=(2,), dtype=jnp.int32),
            "to_move": spaces.Discrete(2),
        })

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: VideoChessState) -> VideoChessObservation:
        return VideoChessObservation(
            board=state.board,
            cursor_pos=state.cursor_pos,
            selected_square=state.selected_square,
            to_move=jnp.array(state.to_move, dtype=jnp.int32),
        )

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: VideoChessObservation) -> jnp.ndarray:
        return jnp.concatenate([
            obs.board.flatten(),
            obs.cursor_pos.flatten(),
            obs.selected_square.flatten(),
            obs.to_move[None],
        ])

    @partial(jax.jit, static_argnums=(0,))
    def _get_env_reward(self, prev_state: VideoChessState, state: VideoChessState) -> float:
        del prev_state, state
        return jnp.float32(0.0)  # no reward shaping yet

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, prev_state: VideoChessState, state: VideoChessState) -> float:
        return self._get_env_reward(prev_state, state)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: VideoChessState) -> bool:
        return state.game_phase == self.consts.PHASE_GAME_OVER

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: VideoChessState) -> VideoChessInfo:
        del state
        return VideoChessInfo()
    
    



class VideoChessRenderer(JAXGameRenderer):
    def __init__(self, consts: VideoChessConstants = None):
        super().__init__()
        self.consts = consts or VideoChessConstants()

        self.config = render_utils.RendererConfig(
            game_dimensions=(self.consts.HEIGHT, self.consts.WIDTH),
            channels=3,
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)

        sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/videochess"

        asset_config = list(self.consts.ASSET_CONFIG)

        background_sprite = jnp.array([[[0, 0, 170, 255]]], dtype=jnp.uint8)
        asset_config.insert(0, {"name": "background", "type": "background", "data": background_sprite})

        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS,
        ) = self.jr.load_and_setup_assets(asset_config, sprite_path)

        bg_h, bg_w = self.BACKGROUND.shape
        target_h, target_w = self.config.game_dimensions
        if bg_h != target_h or bg_w != target_w:
            bg_color_id = self.BACKGROUND[0, 0]
            self.BACKGROUND = jnp.full((target_h, target_w), bg_color_id, dtype=self.BACKGROUND.dtype)

        self.PIECE_STACK = self.SHAPE_MASKS["pieces"]
        self.BOARD_MASK = self.SHAPE_MASKS["board"]

        self.PRE_RENDERED_BOARD = self.jr.render_at(
            self.BACKGROUND,
            self.consts.OFFSET_X_BOARD,
            self.consts.OFFSET_Y_BOARD,
            self.BOARD_MASK,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _calculate_piece_grid(self, state: VideoChessState) -> jnp.ndarray:
        """Piece grid with blinking for selection + last move."""
        grid = state.board
        c = self.consts

        # Same blink pattern as overlays (fast, 50% duty cycle)
        period = jnp.int32(c.BLINK_PERIOD)
        half = jnp.int32(c.BLINK_PERIOD // 2)
        blink_on = (state.frame_counter % period) < half

        in_target_phase = (state.game_phase == c.PHASE_SELECT_TARGET)
        sel_row = state.selected_square[0]
        sel_col = state.selected_square[1]
        has_selection = (sel_row >= 0) & (sel_col >= 0)

        same_square = (state.cursor_pos[0] == sel_row) & (state.cursor_pos[1] == sel_col)

        def maybe_blink(g):
            def do_blink(g2):
                # When blinking off, hide the piece by setting the cell to EMPTY.
                # When blinking on, leave the piece as drawn in the board.
                return jax.lax.cond(
                    blink_on,
                    lambda gg: gg,
                    lambda gg: gg.at[sel_row, sel_col].set(c.EMPTY),
                    g2,
                )

            return jax.lax.cond(same_square & has_selection, do_blink, lambda gg: gg, g)

        # First, apply blinking for the selected piece (only in target phase)
        grid_after_select = jax.lax.cond(in_target_phase, maybe_blink, lambda gg: gg, grid)

        # Then, optionally blink the last moved-to square while last_move_timer > 0
        lm_row = state.last_move_target[0]
        lm_col = state.last_move_target[1]
        has_last = (lm_row >= 0) & (lm_col >= 0)
        last_active = state.last_move_timer > 0

        def blink_last(g3):
            # When blink_on: leave piece visible; when blink_off: hide it
            return jax.lax.cond(
                blink_on,
                lambda gg: gg,
                lambda gg: gg.at[lm_row, lm_col].set(c.EMPTY),
                g3,
            )

        return jax.lax.cond(last_active & has_last, blink_last, lambda gg: gg, grid_after_select)
    @partial(jax.jit, static_argnums=(0,))
    def _render_overlays(self, state: VideoChessState, raster: jnp.ndarray) -> jnp.ndarray:
        """Draw cursor/ghost overlays on top of the board render."""
        c = self.consts

        # Blink pattern: fast, 50% duty cycle (3 frames on, 3 frames off at 30 FPS)
        period = jnp.int32(c.BLINK_PERIOD)
        half = jnp.int32(c.BLINK_PERIOD // 2)
        blink_on = (state.frame_counter % period) < half

        # Selection / phase flags
        in_select_phase = (state.game_phase == c.PHASE_SELECT_PIECE)
        in_target_phase = (state.game_phase == c.PHASE_SELECT_TARGET)

        sel_row = state.selected_square[0]
        sel_col = state.selected_square[1]
        has_selection = (sel_row >= 0) & (sel_col >= 0)

        # Selected piece sprite index (used for ghost at cursor square)
        sel_piece_index = jax.lax.cond(
            has_selection,
            lambda: state.board[sel_row, sel_col],
            lambda: jnp.int32(VideoChessConstants.EMPTY),
        )

        def render_cell(row, col, r):
            piece = state.board[row, col]

            is_cursor = (state.cursor_pos[0] == row) & (state.cursor_pos[1] == col)
            is_selected_square = (state.selected_square[0] == row) & (state.selected_square[1] == col)

            # "Own" here means: belongs to the side whose turn it is.
            is_own = BoardHandler.is_same_colour(piece, state.to_move)

            same_square = is_cursor & is_selected_square

            def draw_cursor(r_in):
                mask = self.PIECE_STACK[c.CURSOR]
                x = c.OFFSET_X_BOARD + 4 + col * 17
                y = c.OFFSET_Y_BOARD + 2 + row * 13
                return self.jr.render_at(r_in, x, y, mask)

            def draw_sel_piece(r_in):
                mask = self.PIECE_STACK[sel_piece_index]
                x = c.OFFSET_X_BOARD + 4 + col * 17
                y = c.OFFSET_Y_BOARD + 2 + row * 13
                return self.jr.render_at(r_in, x, y, mask)

            # --- SELECT_PIECE phase: cursor always visible; blink only when on own piece ---
            def handle_select_phase(r_in):
                """In piece-selection phase:
                - Cursor is always visible on its square.
                - If it is hovering over one of our own pieces (side to move), the cursor blinks
                  (i.e., only drawn on blink_on), while the piece beneath never disappears.
                - On non-own squares, the cursor is drawn every frame (no blinking), so it
                  is easy to track.
                """

                def when_cursor(rr):
                    # If cursor is on an own piece: blink cursor
                    def over_own(r_own):
                        return jax.lax.cond(
                            blink_on,
                            draw_cursor,
                            lambda rrr: rrr,
                            r_own,
                        )

                    # If cursor is on a non-own square: always draw cursor
                    def over_other(r_other):
                        return draw_cursor(r_other)

                    return jax.lax.cond(
                        is_own,
                        over_own,
                        over_other,
                        rr,
                    )

                # If this cell is not the cursor position, do nothing
                return jax.lax.cond(
                    is_cursor,
                    when_cursor,
                    lambda rr: rr,
                    r_in,
                )

            # --- SELECT_TARGET phase: source cursor + target ghost alternation ---
            def handle_target_phase(r_in):
                # Case 1: cursor still on the selected square:
                # the piece itself blinks (handled in _calculate_piece_grid), no cursor overlay.
                def when_same(rr):
                    return rr

                # Case 2: cursor on a different square than selected:
                # - On original selected square: cursor blinks on top.
                # - On cursor square: ghost of the selected piece appears when blink_on.
                def when_diff(rr):
                    def handle_origin(r_origin):
                        # origin square: show cursor only when blink_on
                        cond_origin = blink_on & has_selection
                        return jax.lax.cond(cond_origin, draw_cursor, lambda rrr: rrr, r_origin)

                    def handle_cursor_square(r_cursor):
                        # cursor square: show ghost piece only when blink_on
                        # (so the piece "blinks" at the cursor position)
                        cond_ghost = blink_on & has_selection
                        return jax.lax.cond(cond_ghost, draw_sel_piece, lambda rrr: rrr, r_cursor)

                    return jax.lax.cond(
                        is_selected_square,
                        handle_origin,
                        lambda r_mid: jax.lax.cond(
                            is_cursor,
                            handle_cursor_square,
                            lambda rrr: rrr,
                            r_mid,
                        ),
                        rr,
                    )

                return jax.lax.cond(same_square, when_same, when_diff, r_in)

            # Default: no overlays (other phases)
            def handle_default(r_in):
                # Also allow a non-blinking cursor overlay in non-special states if desired.
                # Here we do nothing to keep it simple.
                return r_in

            # Phase switch
            return jax.lax.cond(
                in_select_phase,
                handle_select_phase,
                lambda r_mid: jax.lax.cond(
                    in_target_phase,
                    handle_target_phase,
                    handle_default,
                    r_mid,
                ),
                r,
            )

        def render_row(row, r):
            return jax.lax.fori_loop(
                0, c.NUM_FILES,
                lambda col, r_in: render_cell(row, col, r_in),
                r,
            )

        return jax.lax.fori_loop(0, c.NUM_RANKS, render_row, raster)

    @partial(jax.jit, static_argnums=(0,))
    def _render_pieces_on_board(self, piece_grid: jnp.ndarray, raster: jnp.ndarray) -> jnp.ndarray:
        c = self.consts

        def render_piece(row, col, r):
            piece_index = piece_grid[row, col]

            def draw(r_in):
                mask = self.PIECE_STACK[piece_index]
                x = c.OFFSET_X_BOARD + 4 + col * 17
                y = c.OFFSET_Y_BOARD + 2 + row * 13
                return self.jr.render_at(r_in, x, y, mask)

            return jax.lax.cond(piece_index != c.EMPTY, draw, lambda r_in: r_in, r)

        def render_row(row, r):
            return jax.lax.fori_loop(
                0, c.NUM_FILES,
                lambda col, r_in: render_piece(row, col, r_in),
                r,
            )

        return jax.lax.fori_loop(0, c.NUM_RANKS, render_row, raster)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: VideoChessState) -> jnp.ndarray:
        raster = self.PRE_RENDERED_BOARD
        piece_grid = self._calculate_piece_grid(state)
        raster = self._render_pieces_on_board(piece_grid, raster)
        raster = self._render_overlays(state, raster)
        return self.jr.render_from_palette(raster, self.PALETTE)