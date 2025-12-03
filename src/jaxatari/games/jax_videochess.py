import os
from functools import partial
from typing import NamedTuple, Tuple

import chex
import jax
import jax.numpy as jnp

import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.spaces import Space


# -------------------- CONSTANTS --------------------

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
    COLOUR_BLACK = 1  # we play as black

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

    MOVE_DELAY_FRAMES = 5  # process movement actions every 4 frames
    LAST_MOVE_BLINK_FRAMES = 90  # ~3 seconds at 30 FPS

    # reuse a very simple asset config: one board + a group of pieces
    ASSET_CONFIG = (
        {"name": "board", "type": "single", "file": "background.npy", "transpose": True},
        {"name": "pieces", "type": "group", "files": [f"pieces/{i}.npy" for i in range(14)]},
    )


# -------------------- STATE / OBS / INFO --------------------

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


# -------------------- BOARD HANDLER (VERY MINIMAL RULES) --------------------

class BoardHandler:

    @staticmethod
    def reset_board() -> jnp.ndarray:
        """Standard chess start position, encoded with VideoChessConstants."""
        c = VideoChessConstants
        board = jnp.zeros((c.NUM_RANKS, c.NUM_FILES), dtype=jnp.int32)

        white_back = jnp.array(
            [c.W_ROOK, c.W_KNIGHT, c.W_BISHOP, c.W_QUEEN,
             c.W_KING, c.W_BISHOP, c.W_KNIGHT, c.W_ROOK],
            dtype=jnp.int32,
        )
        white_pawns = jnp.full((c.NUM_FILES,), c.W_PAWN, dtype=jnp.int32)

        black_back = jnp.array(
            [c.B_ROOK, c.B_KNIGHT, c.B_BISHOP, c.B_QUEEN,
             c.B_KING, c.B_BISHOP, c.B_KNIGHT, c.B_ROOK],
            dtype=jnp.int32,
        )
        black_pawns = jnp.full((c.NUM_FILES,), c.B_PAWN, dtype=jnp.int32)

        board = board.at[0, :].set(white_back)
        board = board.at[1, :].set(white_pawns)
        board = board.at[6, :].set(black_pawns)
        board = board.at[7, :].set(black_back)

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
    def legal_moves_for_black_piece(board: jnp.ndarray, from_sq: chex.Array) -> jnp.ndarray:
        """Legal moves for a black piece (chess-like, only black side for now).

        Implemented:
        - Black pawn: forward 1/2 from start, diagonal captures.
        - Knight: L-shaped jumps, max 8.
        - Bishop: sliding diagonals.
        - Rook: sliding orthogonals.
        - Queen: bishop + rook rays.
        - King: one step in any direction.

        Returns:
            jnp.ndarray of shape (28, 2) with target squares, padded with [-1, -1].
        """
        c = VideoChessConstants

        # Keep row/col as JAX scalars (no int())
        row = from_sq[0]
        col = from_sq[1]

        piece = board[row, col]

        # Helper: a canonical "no-move" array
        def no_moves_28() -> jnp.ndarray:
            return jnp.full((28, 2), -1, dtype=jnp.int32)

        # ----------------------
        # Pawn moves (black pawns)
        # ----------------------
        def pawn_moves() -> jnp.ndarray:
            # Candidates: forward 1, forward 2 (from start), capture left, capture right
            # We'll build 4 candidates and then pad to (28, 2).

            # 1) One step forward (row - 1, same col), only if empty
            f_row1 = row - jnp.int32(1)
            f_col = col
            inb1 = BoardHandler.in_bounds(f_row1, f_col)
            empty1 = jax.lax.cond(
                inb1,
                lambda: BoardHandler.is_empty(board, f_row1, f_col),
                lambda: False,
            )
            move_f1 = jax.lax.cond(
                inb1 & empty1,
                lambda: jnp.stack([f_row1, f_col]).astype(jnp.int32),
                lambda: jnp.array([-1, -1], dtype=jnp.int32),
            )

            # 2) Two steps forward from start rank (row == 6)
            f_row2 = row - jnp.int32(2)
            start_rank = (row == jnp.int32(6))
            inb2 = BoardHandler.in_bounds(f_row2, f_col)

            def can_two_step():
                return (
                    BoardHandler.is_empty(board, f_row1, f_col)
                    & BoardHandler.is_empty(board, f_row2, f_col)
                )

            empty2 = jax.lax.cond(inb2 & start_rank, can_two_step, lambda: False)

            move_f2 = jax.lax.cond(
                inb2 & start_rank & empty2,
                lambda: jnp.stack([f_row2, f_col]).astype(jnp.int32),
                lambda: jnp.array([-1, -1], dtype=jnp.int32),
            )

            # 3) Capture diagonally up-left (row-1, col-1)
            cap_l_row = row - jnp.int32(1)
            cap_l_col = col - jnp.int32(1)
            inb_cap_l = BoardHandler.in_bounds(cap_l_row, cap_l_col)
            target_l = jax.lax.cond(
                inb_cap_l,
                lambda: board[cap_l_row, cap_l_col],
                lambda: jnp.int32(c.EMPTY),
            )
            can_cap_l = inb_cap_l & BoardHandler.is_opponent(target_l, c.COLOUR_BLACK)
            move_cap_l = jax.lax.cond(
                can_cap_l,
                lambda: jnp.stack([cap_l_row, cap_l_col]).astype(jnp.int32),
                lambda: jnp.array([-1, -1], dtype=jnp.int32),
            )

            # 4) Capture diagonally up-right (row-1, col+1)
            cap_r_row = row - jnp.int32(1)
            cap_r_col = col + jnp.int32(1)
            inb_cap_r = BoardHandler.in_bounds(cap_r_row, cap_r_col)
            target_r = jax.lax.cond(
                inb_cap_r,
                lambda: board[cap_r_row, cap_r_col],
                lambda: jnp.int32(c.EMPTY),
            )
            can_cap_r = inb_cap_r & BoardHandler.is_opponent(target_r, c.COLOUR_BLACK)
            move_cap_r = jax.lax.cond(
                can_cap_r,
                lambda: jnp.stack([cap_r_row, cap_r_col]).astype(jnp.int32),
                lambda: jnp.array([-1, -1], dtype=jnp.int32),
            )

            # Stack the 4 pawn candidates
            pawn_moves_4 = jnp.stack([move_f1, move_f2, move_cap_l, move_cap_r], axis=0)  # (4,2)

            # Pad up to (28,2) with [-1,-1]
            pad = jnp.full((28 - 4, 2), -1, dtype=jnp.int32)
            return jnp.concatenate([pawn_moves_4, pad], axis=0)

        # ----------------------
        # Knight moves (L-shaped jumps)
        # ----------------------
        def knight_moves() -> jnp.ndarray:
            deltas = jnp.array([
                [-2, -1], [-2, 1],
                [-1, -2], [-1, 2],
                [1, -2],  [1, 2],
                [2, -1],  [2, 1],
            ], dtype=jnp.int32)

            def one(delta):
                r2 = row + delta[0]
                c2 = col + delta[1]
                inb = BoardHandler.in_bounds(r2, c2)

                target = jax.lax.cond(
                    inb,
                    lambda: board[r2, c2],
                    lambda: jnp.int32(c.EMPTY),
                )

                legal = inb & (~BoardHandler.is_same_colour(target, c.COLOUR_BLACK))

                return jax.lax.cond(
                    legal,
                    lambda: jnp.stack([r2, c2]).astype(jnp.int32),
                    lambda: jnp.array([-1, -1], dtype=jnp.int32),
                )

            moves8 = jax.vmap(one)(deltas)  # (8,2)
            pad = jnp.full((28 - 8, 2), -1, dtype=jnp.int32)
            return jnp.concatenate([moves8, pad], axis=0)

        # ----------------------
        # Sliding moves helper (for bishop/rook/queen)
        # ----------------------
        def ray_moves(dir_row, dir_col, max_steps: int = 7) -> jnp.ndarray:
            """Generate up to max_steps squares along a single ray.

            Stops when leaving board or hitting any piece; if it hits an
            opponent, that square is included; if it hits own piece, it is blocked.
            Returns shape (max_steps, 2), padded with [-1, -1].
            """
            def body(i, carry):
                moves_arr, blocked = carry
                # step index i -> distance (i+1)
                r2 = row + dir_row * (i + 1)
                c2 = col + dir_col * (i + 1)
                inb = BoardHandler.in_bounds(r2, c2)

                def when_in_bounds(carry_in):
                    moves_arr_, blocked_ = carry_in

                    def if_blocked(c_block):
                        # Already blocked in this direction: do nothing
                        return c_block

                    def if_not_blocked(c_not_block):
                        moves_arr_nb, blocked_nb = c_not_block
                        target = board[r2, c2]
                        is_own = BoardHandler.is_same_colour(target, c.COLOUR_BLACK)
                        is_empty = (target == jnp.int32(c.EMPTY))
                        is_opp = BoardHandler.is_opponent(target, c.COLOUR_BLACK)

                        # Own piece: cannot move there, ray stops
                        def own_case():
                            return moves_arr_nb, True

                        # Opponent piece: can capture, then ray stops
                        def opp_case():
                            ma = moves_arr_nb.at[i].set(jnp.stack([r2, c2]).astype(jnp.int32))
                            return ma, True

                        # Empty square: can move, ray continues
                        def empty_case():
                            ma = moves_arr_nb.at[i].set(jnp.stack([r2, c2]).astype(jnp.int32))
                            return ma, False

                        # Use explicit operand for all jax.lax.cond calls:
                        return jax.lax.cond(
                            is_own,
                            lambda _: own_case(),
                            lambda _: jax.lax.cond(
                                is_opp,
                                lambda __: opp_case(),
                                lambda __: jax.lax.cond(
                                    is_empty,
                                    lambda ___: empty_case(),
                                    lambda ___: (moves_arr_nb, True),
                                    (moves_arr_nb, blocked_nb),
                                ),
                                (moves_arr_nb, blocked_nb),
                            ),
                            (moves_arr_nb, blocked_nb),
                        )

                    return jax.lax.cond(blocked_, if_blocked, if_not_blocked, (moves_arr_, blocked_))

                def when_out_of_bounds(carry_in):
                    moves_arr_, _ = carry_in
                    # Once we leave the board, ray is blocked for further squares
                    return moves_arr_, True

                return jax.lax.cond(inb, when_in_bounds, when_out_of_bounds, (moves_arr, blocked))

            init_moves = jnp.full((max_steps, 2), -1, dtype=jnp.int32)
            init_blocked = False
            moves_arr, _ = jax.lax.fori_loop(0, max_steps, body, (init_moves, init_blocked))
            return moves_arr

        # Bishop: 4 diagonal rays (4 * 7 = 28)
        def bishop_moves() -> jnp.ndarray:
            dirs = jnp.array([
                [-1, -1],
                [-1,  1],
                [ 1, -1],
                [ 1,  1],
            ], dtype=jnp.int32)

            def one(d):
                return ray_moves(d[0], d[1], 7)

            rays = jax.vmap(one)(dirs)  # (4,7,2)
            return rays.reshape((28, 2))

        # Rook: 4 orthogonal rays (4 * 7 = 28)
        def rook_moves() -> jnp.ndarray:
            dirs = jnp.array([
                [-1,  0],
                [ 1,  0],
                [ 0, -1],
                [ 0,  1],
            ], dtype=jnp.int32)

            def one(d):
                return ray_moves(d[0], d[1], 7)

            rays = jax.vmap(one)(dirs)  # (4,7,2)
            return rays.reshape((28, 2))

        # Queen: bishop + rook rays, compressed into 28 slots.
        def queen_moves() -> jnp.ndarray:
            b28 = bishop_moves()   # (28,2) with -1 padding
            r28 = rook_moves()     # (28,2) with -1 padding
            both = jnp.concatenate([b28, r28], axis=0)  # (56,2)

            def pack_56_to_28(moves56: jnp.ndarray) -> jnp.ndarray:
                def body(i, carry):
                    res, count = carry
                    move = moves56[i]
                    is_valid = (move[0] >= jnp.int32(0))

                    def add(carry2):
                        res2, count2 = carry2

                        def really_add(c3):
                            res3, count3 = c3
                            res3 = res3.at[count3].set(move)
                            return res3, count3 + jnp.int32(1)

                        def no_add(c3):
                            return c3

                        return jax.lax.cond(count2 < jnp.int32(28), really_add, no_add, (res2, count2))

                    def skip(carry2):
                        return carry2

                    return jax.lax.cond(is_valid, add, skip, (res, count))

                init_res = jnp.full((28, 2), -1, dtype=jnp.int32)
                init_count = jnp.int32(0)
                res_final, _ = jax.lax.fori_loop(0, 56, body, (init_res, init_count))
                return res_final

            return pack_56_to_28(both)

        # King: 1-step in 8 directions
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

                legal = inb & (~BoardHandler.is_same_colour(target, c.COLOUR_BLACK))

                return jax.lax.cond(
                    legal,
                    lambda: jnp.stack([r2, c2]).astype(jnp.int32),
                    lambda: jnp.array([-1, -1], dtype=jnp.int32),
                )

            moves_8 = jax.vmap(one_dir)(directions)  # (8,2)
            pad = jnp.full((28 - 8, 2), -1, dtype=jnp.int32)
            return jnp.concatenate([moves_8, pad], axis=0)

        # ----------------------
        # Piece-type dispatch
        # ----------------------
        def moves_if_black_piece() -> jnp.ndarray:
            return jax.lax.cond(
                piece == jnp.int32(c.B_PAWN),
                pawn_moves,
                lambda: jax.lax.cond(
                    piece == jnp.int32(c.B_KNIGHT),
                    knight_moves,
                    lambda: jax.lax.cond(
                        piece == jnp.int32(c.B_BISHOP),
                        bishop_moves,
                        lambda: jax.lax.cond(
                            piece == jnp.int32(c.B_ROOK),
                            rook_moves,
                            lambda: jax.lax.cond(
                                piece == jnp.int32(c.B_QUEEN),
                                queen_moves,
                                lambda: jax.lax.cond(
                                    piece == jnp.int32(c.B_KING),
                                    king_moves,
                                    no_moves_28,
                                ),
                            ),
                        ),
                    ),
                ),
            )

        # If the selected square is not a black piece, return "no moves"
        return jax.lax.cond(
            BoardHandler.is_black(piece),
            moves_if_black_piece,
            no_moves_28,
        )

    @staticmethod
    def apply_move(board: jnp.ndarray, from_sq: chex.Array, to_sq: chex.Array) -> jnp.ndarray:
        """Move piece, with simple pawn promotion to queen for both colours.

        - Black pawn promotes on rank 0 (top of the board) to B_QUEEN.
        - White pawn promotes on rank 7 (bottom of the board) to W_QUEEN.
        """
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
    def has_any_legal_moves_for_black(board: jnp.ndarray) -> bool:
        """Return True if at least one black piece has any legal move.

        Uses the existing legal_moves_for_black_piece() as a backend.
        """
        c = VideoChessConstants

        def body_fun(idx, carry):
            found = carry
            row = idx // c.NUM_FILES
            col = idx % c.NUM_FILES
            piece = board[row, col]

            def check_square(_):
                moves = BoardHandler.legal_moves_for_black_piece(
                    board,
                    jnp.array([row, col], dtype=jnp.int32),
                )
                # Any move where row >= 0 is considered a real move
                has_move = jnp.any(moves[:, 0] >= jnp.int32(0))
                return found | has_move

            return jax.lax.cond(
                BoardHandler.is_black(piece),
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


# -------------------- ENVIRONMENT --------------------

class JaxVideoChess(
    JaxEnvironment[VideoChessState, VideoChessObservation, VideoChessInfo, VideoChessConstants]
):
    @partial(jax.jit, static_argnums=(0,))
    def _update_game_over(self, state: VideoChessState) -> VideoChessState:
        """Update game_phase and winner based on basic end conditions.

        - If one king is missing -> the other side wins.
        - If both kings are missing -> draw (winner = -1).
        - If both kings are present but black has no legal moves -> treat as stalemate (draw).
        """
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
    def __init__(self, consts: VideoChessConstants = None):
        consts = consts or VideoChessConstants()
        super().__init__(consts)
        self.renderer = VideoChessRenderer(self.consts)

        # Allow 4-way + diagonal movement + FIRE + NOOP
        self.action_set = {
            Action.NOOP,
            Action.FIRE,
            Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT,
            Action.UPLEFT, Action.UPRIGHT, Action.DOWNLEFT, Action.DOWNRIGHT,
        }

    def render(self, state: VideoChessState) -> jnp.ndarray:
        return self.renderer.render(state)

    # ---- reset ----
    def reset(self, key: jax.random.PRNGKey = jax.random.PRNGKey(0)) \
            -> Tuple[VideoChessObservation, VideoChessState]:
        c = self.consts
        board = BoardHandler.reset_board()
        state = VideoChessState(
            board=board,
            to_move=c.COLOUR_BLACK,
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

        # **DO NOT** call self._get_observation here – build it directly:
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
        """
        Move cursor one square in 8 directions based on Atari actions.
        """
        # Direction flags
        up = (action == Action.UP) | (action == Action.UPLEFT) | (action == Action.UPRIGHT)
        down = (action == Action.DOWN) | (action == Action.DOWNLEFT) | (action == Action.DOWNRIGHT)
        left = (action == Action.LEFT) | (action == Action.UPLEFT) | (action == Action.DOWNLEFT)
        right = (action == Action.RIGHT) | (action == Action.UPRIGHT) | (action == Action.DOWNRIGHT)

        # Compute deltas
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
        """Step: FIRE and movement are edge-triggered (one effect per keypress).

        - Movement actions: applied only if last_was_move is False.
        - FIRE: applied only if last_was_fire is False.
        This prevents a held FIRE from instantly re-selecting a moved piece.
        """
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

        # If NOOP, reset both last_was_move and last_was_fire (release input edge detection)
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

        # Advance frame counter for animations
        new_state = new_state._replace(
            frame_counter=(new_state.frame_counter + 1) % self.consts.ANIMATION_FRAME_RATE
        )

        # Countdown for last-move blinking
        new_state = new_state._replace(
            last_move_timer=jax.lax.max(
                jnp.int32(0),
                new_state.last_move_timer - jnp.int32(1),
            )
        )

        # Check for basic end conditions (king captured or no black moves -> draw)
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
        Phase 0: move cursor and select which black piece to move.
        """
        def select_piece(s: VideoChessState) -> VideoChessState:
            row, col = s.cursor_pos
            piece = s.board[row, col]
            is_own = BoardHandler.is_same_colour(piece, self.consts.COLOUR_BLACK)
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
            legal_targets = BoardHandler.legal_moves_for_black_piece(st.board, st.selected_square)
            is_legal = jnp.any(jnp.all(legal_targets == st.cursor_pos, axis=1))

            def do_move():
                # Original and target squares
                from_sq = st.selected_square
                to_sq = st.cursor_pos

                new_board = BoardHandler.apply_move(st.board, from_sq, to_sq)
                return st._replace(
                    board=new_board,
                    selected_square=jnp.array([-1, -1], dtype=jnp.int32),
                    game_phase=self.consts.PHASE_SELECT_PIECE,
                    # After move, cursor jumps back to original square
                    cursor_pos=from_sq,
                    # Remember where we just moved to, so that piece can blink
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

            # Check if the cursor is currently on one of our own (black) pieces
            row, col = st.cursor_pos
            piece = st.board[row, col]
            is_own = BoardHandler.is_same_colour(piece, self.consts.COLOUR_BLACK)

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
    
    


# -------------------- RENDERER (REUSE CHECKERS SPRITES) --------------------

class VideoChessRenderer(JAXGameRenderer):
    def __init__(self, consts: VideoChessConstants = None):
        super().__init__()
        self.consts = consts or VideoChessConstants()

        self.config = render_utils.RendererConfig(
            game_dimensions=(self.consts.HEIGHT, self.consts.WIDTH),
            channels=3,
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)

        # IMPORTANT: currently reusing the VideoCheckers sprite folder
        sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/videochess"

        asset_config = list(self.consts.ASSET_CONFIG)

        # simple 1x1 background colour (dark blue)
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
        """Return the piece grid, with optional blinking of the selected piece
        when the cursor is still on its original square in the target phase,
        and also blink the last moved-to piece while last_move_timer > 0.
        """
        grid = state.board
        c = self.consts

        # Same blink pattern as overlays (fast, 50% duty cycle)
        blink_period = jnp.int32(6)
        blink_on = (state.frame_counter % blink_period) < (blink_period // 2)

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
        """
        Draw cursor and ghost-piece overlays on top of the already rendered pieces.
        - In SELECT_PIECE phase: hovering over own piece -> blinking cursor on top.
        - In SELECT_TARGET phase:
          * If cursor is still on selected square: no cursor overlay; piece itself blinks.
          * If cursor is on a different square than selected:
              - Original selected square: cursor blinks on top.
              - Cursor square: ghost of selected piece appears when blink_on; empty when blink_off.
        """
        c = self.consts

        # Blink pattern: fast, 50% duty cycle (3 frames on, 3 frames off at 30 FPS)
        blink_period = jnp.int32(6)
        blink_on = (state.frame_counter % blink_period) < (blink_period // 2)

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

            is_own = BoardHandler.is_same_colour(piece, VideoChessConstants.COLOUR_BLACK)

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
                - If it is hovering over one of our own (black) pieces, the cursor blinks
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
        # 1) Draw all pieces according to the board only
        piece_grid = self._calculate_piece_grid(state)
        raster = self._render_pieces_on_board(piece_grid, raster)
        # 2) Draw cursor / ghost overlays on top
        raster = self._render_overlays(state, raster)
        # 3) Convert from palette to RGB
        return self.jr.render_from_palette(raster, self.PALETTE)