import os
from functools import partial
from typing import NamedTuple, Tuple, Dict, Any, List, Optional
import jax
import jax.lax
import jax.numpy as jnp
import chex
from flax import struct

import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action, EnvObs
from jaxatari.spaces import Space
from jaxatari.modification import AutoDerivedConstants

#
# by Tim Morgner and Jan Larionow
#

# IMPORTANT: WE, THE PLAYER, PLAY AS BLACK, THE OPPONENT IS WHITE.

# Transitions between game phases
# SELECT_PIECE -> MOVE_PIECE: # Player selects a piece to move
# MOVE_PIECE -> SHOW_OPPONENT_MOVE: # Player moves the piece with no further jumps available
# MOVE_PIECE -> MOVE_PIECE: # Player moves the piece with further jumps available
# SHOW_OPPONENT_MOVE -> SELECT_PIECE: # Player makes an input to select a piece after the opponent's move


def _get_default_asset_config() -> tuple:
    """
    Returns the default declarative asset manifest for VideoCheckers.
    Kept immutable (tuple of dicts) to fit NamedTuple defaults.
    """
    # Pieces: 0=EMPTY, 1=WHITE_PIECE, 2=BLACK_PIECE, 3=WHITE_KING, 4=BLACK_KING, 5=BLACK_CURSOR, 6=WHITE_CURSOR
    piece_files = [
        f"pieces/{i}.npy" for i in range(7)
    ]
    
    # Text sprites: 0-11
    text_files = [f"text/{i}.npy" for i in range(12)]
    
    return (
        # The checkerboard pattern
        {'name': 'board', 'type': 'single', 'file': 'background.npy', 'transpose': True},
        
        # Group for all piece types
        {'name': 'pieces', 'type': 'group', 'files': piece_files},
        
        # Group for text sprites
        {'name': 'text', 'type': 'group', 'files': text_files},
    )


class VideoCheckersConstants(AutoDerivedConstants):
    MAX_PIECES: int = struct.field(pytree_node=False, default=12)

    COLOUR_WHITE: int = struct.field(pytree_node=False, default=0)
    COLOUR_BLACK: int = struct.field(pytree_node=False, default=1)

    WIDTH: int = struct.field(pytree_node=False, default=160)
    HEIGHT: int = struct.field(pytree_node=False, default=210)

    SCALING_FACTOR: int = struct.field(pytree_node=False, default=3)

    OFFSET_X_BOARD: int = struct.field(pytree_node=False, default=12)
    OFFSET_Y_BOARD: int = struct.field(pytree_node=False, default=50)

    # MOVES array order: 0=DownRight, 1=UpRight, 2=UpLeft, 3=DownLeft
    # This order is used by AI heuristics and must match the order expected by get_global_legality
    MOVES: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array([
        [1, 1],   # 0: DownRight (row+1, col+1)
        [-1, 1],  # 1: UpRight (row-1, col+1)
        [-1, -1], # 2: UpLeft (row-1, col-1)
        [1, -1],  # 3: DownLeft (row+1, col-1)
    ]))

    # Opponent move scoring
    CAPTURE_W: float = struct.field(pytree_node=False, default=10.0)  # capturing opponent piece
    UPGRADE_W: int = struct.field(pytree_node=False, default=5)  # upgrading piece
    ADVANCE_W: float = struct.field(pytree_node=False, default=1.0)  # moving forward
    CENTER_FWD: float = struct.field(pytree_node=False, default=0.5)  # moving towards center
    CENTER_BWD: float = struct.field(pytree_node=False, default=-2.0)  # moving away from center
    NO_MOVE: float = struct.field(pytree_node=False, default=-jnp.inf)  # avoid standing still

    EMPTY_TILE: int = struct.field(pytree_node=False, default=0)
    WHITE_PIECE: int = struct.field(pytree_node=False, default=1)
    BLACK_PIECE: int = struct.field(pytree_node=False, default=2)
    WHITE_KING: int = struct.field(pytree_node=False, default=3)
    BLACK_KING: int = struct.field(pytree_node=False, default=4)
    WHITE_CURSOR: int = struct.field(pytree_node=False, default=5)
    BLACK_CURSOR: int = struct.field(pytree_node=False, default=6)

    NUM_FIELDS_X: int = struct.field(pytree_node=False, default=8)
    NUM_FIELDS_Y: int = struct.field(pytree_node=False, default=8)
    CENTER: float = struct.field(pytree_node=False, default=3.5)

    SELECT_PIECE_PHASE: int = struct.field(pytree_node=False, default=0)
    MOVE_PIECE_PHASE: int = struct.field(pytree_node=False, default=1)
    SHOW_OPPONENT_MOVE_PHASE: int = struct.field(pytree_node=False, default=2)
    GAME_OVER_PHASE: int = struct.field(pytree_node=False, default=3)

    ANIMATION_FRAME_RATE: int = struct.field(pytree_node=False, default=30)

    # Asset config baked into constants (immutable default) for asset overrides
    ASSET_CONFIG: tuple[dict, ...] = struct.field(pytree_node=False, default=_get_default_asset_config())

    def compute_derived(self) -> dict[str, Any]:
        return {
            'WINDOW_WIDTH': self.WIDTH * self.SCALING_FACTOR,
            'WINDOW_HEIGHT': self.HEIGHT * self.SCALING_FACTOR,
        }

@struct.dataclass
class OpponentMove:
    start_pos: chex.Array  # Start position of the opponent's piece
    end_pos: chex.Array  # End position of the opponent's piece
    piece_type: int  # Type of the piece at the end position (king or normal)
    captured_positions: chex.Array  # Array of positions of captured pieces
    resulting_board: chex.Array  # New board with all moves applied


@struct.dataclass
class LegalityMap:
    """Pre-computed legality information for the entire board."""
    can_move: chex.Array  # (8, 8) bool - can move from this square
    can_jump: chex.Array  # (8, 8) bool - can jump from this square
    any_jump: bool  # Whether any jump is available
    move_dirs: chex.Array  # (4, 8, 8) bool - which of the 4 directions are valid moves
    jump_dirs: chex.Array  # (4, 8, 8) bool - which of the 4 directions are valid jumps
    moves: chex.Array  # (4, 8, 8, 2) - stacked [move_dirs, jump_dirs] for easy indexing


@struct.dataclass
class VideoCheckersState:
    board: chex.Array  # Shape (NUM_FIELDS_Y, NUM_FIELDS_X)
    game_phase: int
    cursor_pos: chex.Array
    has_jumped: chex.Array  # True if the selected piece has already jumped, so the player cant deselect it.
    additional_jump: bool  # True if in the MOVE_PIECE_PHASE a there has already been a jump, so the player can jump again. This prevents the player from deselecting the piece.
    selected_piece: chex.Array
    frame_counter: chex.Array
    opponent_move: OpponentMove
    winner: int  # -1 if no winner, COLOUR_WHITE if white won, COLOUR_BLACK if black won.
    must_jump: bool  # True if player must jump (computed once per step, cached in state)
    rng_key: chex.PRNGKey


@struct.dataclass
class VideoCheckersObservation:
    board: chex.Array
    start_pos: chex.Array
    end_pos: chex.Array
    must_jump: chex.Array
    cursor_pos: chex.Array


@struct.dataclass
class VideoCheckersInfo:
    pass

class JaxVideoCheckers(
    JaxEnvironment[VideoCheckersState, VideoCheckersObservation, VideoCheckersInfo, VideoCheckersConstants]):
    # Minimal ALE action set for Video Checkers (from scripts/action_space_helper.py)
    # Note: NOOP is NOT in the ALE action set for this game
    ACTION_SET: jnp.ndarray = jnp.array(
        [
            Action.FIRE,
            Action.UPRIGHT,
            Action.UPLEFT,
            Action.DOWNRIGHT,
            Action.DOWNLEFT,
        ],
        dtype=jnp.int32,
    )

    def __init__(self, consts: VideoCheckersConstants = None):
        consts = consts or VideoCheckersConstants()
        super().__init__(consts)
        self.renderer = VideoCheckersRenderer(self.consts)

    def render(self, state: VideoCheckersState) -> jnp.ndarray:
        return self.renderer.render(state)

    # BoardHandler methods moved into class
    def reset_board(self):
        """
        Returns a clean board with all pieces in their initial position.
        Returns:
            new board with all pieces in their initial position.
        """
        # Initialize the board with pieces, this is a placeholder
        board = jnp.zeros((self.consts.NUM_FIELDS_X,
                           self.consts.NUM_FIELDS_Y), dtype=jnp.int32)
        # Set up the initial pieces on the board
        white_rows = jnp.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        white_cols = jnp.array([1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7])

        black_rows = jnp.array([5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7])
        black_cols = jnp.array([0, 2, 4, 6, 1, 3, 5, 7, 0, 2, 4, 6])

        board = board.at[white_rows, white_cols].set(self.consts.WHITE_PIECE)
        board = board.at[black_rows, black_cols].set(self.consts.BLACK_PIECE)

        return board

    def move_piece(self, row, col, drow, dcol, board) -> (jnp.ndarray, int, bool, int, int):
        """
        Moves a piece and handles all side effects (upgrading pieces, capturing).
        Args:
            row: row that the pieces is in
            col: column that the pieces is in
            drow: movement in a row
            dcol: movement in a column
            board: current game board
        Returns:
             - new board with the move applied,
             - the type of the piece after the move (e.g. if piece was upgraded) and
             - the coordinates of the captured piece (-1,-1 if nothing was captured)
        """
        # 1. move & upgrade piece
        piece = board[row, col]
        new_row = row + drow
        new_col = col + dcol

        upgrade_white = (piece == self.consts.WHITE_PIECE) & (new_row == 7)
        upgrade_black = (piece == self.consts.BLACK_PIECE) & (new_row == 0)
        new_piece = jax.lax.cond(
            upgrade_white,
            lambda: self.consts.WHITE_KING,
            lambda: jax.lax.cond(
                upgrade_black,
                lambda: self.consts.BLACK_KING,
                lambda: piece)
        )

        new_board = (board
                     .at[(row, col)].set(self.consts.EMPTY_TILE)
                     .at[(new_row, new_col)].set(new_piece))

        # 2. handle capture
        is_jump = (jnp.abs(drow) == 2) & (jnp.abs(dcol) == 2)

        def _handle_capture(board):
            captured_row = row + drow // 2
            captured_col = col + dcol // 2
            return (board.at[(captured_row, captured_col)].set(self.consts.EMPTY_TILE),
                    captured_row, captured_col)

        new_board, captured_row, captured_col = jax.lax.cond(
            is_jump,
            _handle_capture,
            lambda b: (b, -1, -1),
            new_board
        )

        return new_board, new_piece, is_jump, captured_row, captured_col

    def tile_is_free(self, row, col, board):
        """
        Args:
            row: row of tile to check
            col: column of tile to check
            board: current game board
        """
        return board[row, col] == self.consts.EMPTY_TILE

    def move_in_bounds(self, row, col, drow, dcol):
        """
        Checks if move can be made in the given direction.
        Args:
            row: row index of the piece
            col: column index of the piece
            drow: movement in y direction
            dcol: movement in x direction

        Returns: True, if cursor can be moved in the given direction, False otherwise.
        """
        return (((0 <= row + drow) & (row + drow < self.consts.NUM_FIELDS_Y))
                & ((0 <= col + dcol) & (col + dcol < self.consts.NUM_FIELDS_X)))

    def get_possible_moves_for_piece(self, row, col, board: chex.Array):
        """
        Get all possible moves for a piece at position (row,col)
        Args:
           row: row index of the piece
           col: column index of the piece
           board: current game board

        Returns: array of all possible moves. If a move in a given direction is not possible, it returns [0,0]
        """

        is_not_a_piece = (self.tile_is_free(row, col, board)) | (row == -1)

        def _get_moves():
            def prune_possible_moves_for_jumps(possible_moves):
                """
                Prune the possible moves for jumps, set the rest to [0 0].
                """

                return jax.lax.fori_loop(
                    lower=0,
                    upper=possible_moves.shape[0],
                    body_fun=lambda i, moves: jax.lax.cond(
                        jnp.all(jnp.abs(moves[i]) == 2),  # Check if the move is a jump
                        lambda moves: moves.at[i].set(moves[i]),  # Keep the jump move
                        lambda moves: moves.at[i].set(jnp.array([0, 0])),  # Set non-jump move to [0, 0]
                        operand=moves
                    ),
                    init_val=possible_moves
                )

            def check_move(move):
                drow, dcol = move
                jump_available = self.move_is_available(row=row, col=col, drow=2 * drow, dcol=2 * dcol,
                                                                board=board)  # check jump
                move_available = self.move_is_available(row=row, col=col, drow=drow, dcol=dcol,
                                                                board=board)  # check normal move

                # Return jump move if available, else normal move if available, else [0,0]
                return jax.lax.cond(
                    jump_available,
                    lambda: move * 2,
                    lambda: jax.lax.cond(
                        move_available,
                        lambda: move,
                        lambda: jnp.array([0, 0]),
                    ),
                )

            possible_moves = jax.vmap(check_move)(self.consts.MOVES)
            # if there is a jump in the possible moves, make non jump moves [0,0]
            has_jump = jnp.any(jnp.all(jnp.abs(possible_moves) == 2, axis=1))
            possible_moves = jax.lax.cond(
                has_jump,
                lambda: prune_possible_moves_for_jumps(possible_moves),
                lambda: possible_moves,

            )
            return possible_moves

        return jax.lax.cond(is_not_a_piece, lambda: jnp.zeros((4, 2), dtype=jnp.int32), _get_moves)

    def count_pieces(self, board: jnp.ndarray) -> (float, float):
        cnt_white = jnp.sum(board == 1) + jnp.sum(board == 3)
        cnt_black = jnp.sum(board == 2) + jnp.sum(board == 4)
        return cnt_white, cnt_black

    def move_is_available(self, row, col, drow, dcol, board: chex.Array):
        """
        Checks if a piece can be moved in the given direction. Checks for both, simple moves and jumps.
        Refactored to use boolean math instead of lax.cond to speed up compilation.
        Args:
            row: row index of the piece
            col: column index of the piece
            drow: movement in y direction
            dcol: movement in x direction
            board: current game board

        Returns:
            True, if a piece can be moved in the given direction, False otherwise.
        """
        new_row, new_col = row + drow, col + dcol
        in_bounds = (new_row >= 0) & (new_row < self.consts.NUM_FIELDS_Y) & (new_col >= 0) & (new_col < self.consts.NUM_FIELDS_X)
        
        piece = board[row, col]
        # Use simple clipping to prevent OOB access during check, relies on in_bounds to validate result
        check_row = jnp.clip(new_row, 0, 7)
        check_col = jnp.clip(new_col, 0, 7)
        target_tile = board[check_row, check_col]
        
        is_white = (piece == self.consts.WHITE_PIECE) | (piece == self.consts.WHITE_KING)
        is_king = (piece == self.consts.WHITE_KING) | (piece == self.consts.BLACK_KING)
        
        is_jump = (jnp.abs(drow) == 2) & (jnp.abs(dcol) == 2)
        # Fix logic direction: White(0-2) moves + (down?), Black(5-7) moves - (up?) 
        # Note: In reset_board white is 0-2, black is 5-7.
        # If White moves 0->7, drow is positive.
        expected_drow_jump = jnp.where(is_white, 2, -2)
        expected_drow_move = jnp.where(is_white, 1, -1)
        
        is_fwd_jump = is_jump & (drow == expected_drow_jump)
        is_fwd_move = (~is_jump) & (drow == expected_drow_move)
        can_move_dir = is_king | is_fwd_jump | is_fwd_move

        # Jump check
        j_row, j_col = row + drow // 2, col + dcol // 2
        j_in_bounds = (j_row >= 0) & (j_row < 8) & (j_col >= 0) & (j_col < 8)
        check_j_row = jnp.clip(j_row, 0, 7)
        check_j_col = jnp.clip(j_col, 0, 7)
        jumped_piece = board[check_j_row, check_j_col]

        is_opp = jnp.where(is_white, 
                           (jumped_piece == self.consts.BLACK_PIECE) | (jumped_piece == self.consts.BLACK_KING),
                           (jumped_piece == self.consts.WHITE_PIECE) | (jumped_piece == self.consts.WHITE_KING))

        valid_jump = is_jump & is_opp & (target_tile == self.consts.EMPTY_TILE) & j_in_bounds
        valid_move = (~is_jump) & (target_tile == self.consts.EMPTY_TILE)

        return in_bounds & can_move_dir & (valid_jump | valid_move)

    def get_movable_pieces(self, colour, board: chex.Array) -> (jnp.ndarray, bool):
        """
        Refactored: Uses get_global_legality instead of expensive per-piece vmaps.
        """
        leg = self.get_global_legality(board, colour)
        
        # Select mask based on rule "Must Jump"
        movable_mask = jnp.where(leg.any_jump, leg.can_jump, leg.can_move)
        
        # Get indices
        rows, cols = jnp.where(movable_mask, size=self.consts.MAX_PIECES, fill_value=-1)
        movable_positions = jnp.stack([rows, cols], axis=1)
        
        return movable_positions, leg.any_jump

    def is_movable_piece(self, colour, position, board: chex.Array):
        """
        check if position is in return set of get_movable_pieces. This is used to check if a piece can be selected in the select piece phase.
        Args:
            colour: Colour of the side to check for. 0 for white, 1 for black.
            position: Position of the piece to check
            board: Current board
        Returns:
            True, if the piece is movable, False otherwise.
        """
        movable_pieces, _ = self.get_movable_pieces(colour, board)
        is_movable = jnp.any(jnp.all(movable_pieces == position, axis=1))
        return is_movable

    @partial(jax.jit, static_argnums=(0,))
    def get_global_legality(self, board: chex.Array, colour: int) -> LegalityMap:
        """
        Calculates legality for the ENTIRE board in one SIMD sweep.
        No indices, no size=MAX_PIECES, no vmap overhead.
        This replaces multiple calls to get_movable_pieces with a single pre-computation.
        """
        rows, cols = jnp.indices((self.consts.NUM_FIELDS_Y, self.consts.NUM_FIELDS_X))
        
        # Get piece mask for this colour
        own_pieces_mask = jnp.where(
            colour == self.consts.COLOUR_WHITE,
            (board == self.consts.WHITE_PIECE) | (board == self.consts.WHITE_KING),
            (board == self.consts.BLACK_PIECE) | (board == self.consts.BLACK_KING)
        )
        
        # Check all 4 directions for every tile at once
        def check_dir(drow, dcol):
            # Check 1-step move
            can_move_1 = jax.vmap(jax.vmap(
                lambda r, c: self.move_is_available(r, c, drow, dcol, board)
            ))(rows, cols)
            # Check 2-step jump
            can_jump_2 = jax.vmap(jax.vmap(
                lambda r, c: self.move_is_available(r, c, 2*drow, 2*dcol, board)
            ))(rows, cols)
            return can_move_1, can_jump_2
        
        # Vectorize over the 4 move directions
        move_results, jump_results = jax.vmap(
            lambda d: check_dir(d[0], d[1])
        )(self.consts.MOVES)
        
        # Extract move and jump masks (shape: (4, 8, 8))
        move_dirs = move_results  # (4, 8, 8) bool
        jump_dirs = jump_results  # (4, 8, 8) bool
        
        # Mask by own pieces
        move_dirs = move_dirs & own_pieces_mask[None, :, :]  # (4, 8, 8)
        jump_dirs = jump_dirs & own_pieces_mask[None, :, :]  # (4, 8, 8)
        
        # Aggregate: can move/jump from any direction
        can_move_mask = jnp.any(move_dirs, axis=0)  # (8, 8)
        can_jump_mask = jnp.any(jump_dirs, axis=0)  # (8, 8)
        any_jump = jnp.any(can_jump_mask)
        
        # Stack move and jump directions for easy indexing: (4, 8, 8, 2)
        # moves[..., 0] = move_dirs, moves[..., 1] = jump_dirs
        moves_stacked = jnp.stack([move_dirs, jump_dirs], axis=-1)
        
        return LegalityMap(
            can_move=can_move_mask,
            can_jump=can_jump_mask,
            any_jump=any_jump,
            move_dirs=move_dirs,
            jump_dirs=jump_dirs,
            moves=moves_stacked
        )

    # OpponentMoveHandler methods moved into class
    def add_captured_position(self, opponent_move: OpponentMove, position: chex.Array) -> OpponentMove:
        # 1. find empty slot in array to store the new position
        matches = jnp.all(opponent_move.captured_positions == jnp.array([-1, -1]), axis=1)
        first_index = jnp.argmax(matches)
        has_match = jnp.any(matches)

        # 2. get new opponent move, if an empty slot was found (guaranteed), do nothing if not
        return jax.lax.cond(
            has_match,
            lambda o: o.replace(captured_positions=o.captured_positions.at[first_index].set(position)),
            lambda o: o,
            opponent_move
        )

    def clear_captured_positions(self, opponent_move: OpponentMove) -> OpponentMove:
        """
        Resets the array containing the captured positions back to only sentinel positions.
        Args:
            opponent_move: opponent move with the captured positions array to be reset.
        Returns:
            new OpponentMove with a clean captures positions array.
        """
        return opponent_move.replace(captured_positions=jnp.full(opponent_move.captured_positions.shape, -1))

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: VideoCheckersObservation) -> jnp.ndarray:
        return jnp.concatenate([
          obs.board.flatten(),
            obs.start_pos.flatten(),
            obs.end_pos.flatten(),
            obs.must_jump.flatten(),
            obs.cursor_pos.flatten()
        ])

    def reset(self, key: jax.random.PRNGKey = jax.random.PRNGKey(0)) \
            -> Tuple[VideoCheckersObservation, VideoCheckersState]:
        """
        Resets the game state to the initial state.
        Returns the initial state and the reward (i.e. 0)

        Args:
            key: Random key for generating the initial state.
        Returns:
            initial_obs: Initial observation of the game.
            state: Initial game state.
        """
        # Initialize the board with pieces, this is a placeholder
        board = self.reset_board()
        
        # Calculate initial must_jump using get_global_legality
        leg = self.get_global_legality(board, self.consts.COLOUR_BLACK)

        # Default state
        state = VideoCheckersState(
            cursor_pos=jnp.array([6, 7]), board=board, game_phase=self.consts.SELECT_PIECE_PHASE,
            selected_piece=jnp.array([-1, -1]), frame_counter=jnp.array(0), winner=-1,
            additional_jump=False,
            opponent_move=OpponentMove(
                start_pos=jnp.array([-1, -1]), end_pos=jnp.array([-1, -1]), piece_type=-1,
                captured_positions=jnp.full((12, 2), -1), resulting_board=board
            ),
            rng_key=key, has_jumped=False, must_jump=leg.any_jump
        )
        """
        testboard = jnp.zeros((self.consts.NUM_FIELDS_Y, self.consts.NUM_FIELDS_X), dtype=jnp.int32)
        # Debug
        pos_black = (4, 3)
        pos_white = (3, 4)
        testboard = (testboard.at[pos_white].set(self.consts.WHITE_PIECE)
                     .at[(1, 6)].set(self.consts.WHITE_PIECE)
                     .at[pos_black].set(self.consts.BLACK_PIECE))
        state = VideoCheckersState(cursor_pos=jnp.array(pos_black), board=testboard,
                                   game_phase=self.consts.MOVE_PIECE_PHASE,
                                   selected_piece=jnp.array(pos_black), frame_counter=jnp.array(0), winner=-1,
                                   additional_jump=False,
                                   opponent_move=OpponentMove(start_pos=jnp.array([-1, -1]),
                                                              end_pos=jnp.array([-1, -1]),
                                                              piece_type=-1,
                                                              captured_positions=jnp.array([[-1, -1]])
                                                              )
                                   , rng_key=key)
        """
        # Debug state show opponent move phase
        """testboard = jnp.zeros((self.consts.NUM_FIELDS_Y, self.consts.NUM_FIELDS_X), dtype=jnp.int32)
        testboard = testboard.at[1, 0].set(self.consts.WHITE_PIECE)
        testboard = testboard.at[2, 1].set(self.consts.BLACK_PIECE)
        testboard = testboard.at[4, 3].set(self.consts.BLACK_PIECE)
        testboard = testboard.at[6, 5].set(self.consts.BLACK_PIECE)
        testboard = testboard.at[0,1].set(self.consts.WHITE_PIECE)
        testboard = testboard.at[0,3].set(self.consts.WHITE_PIECE)
        testboard = testboard.at[0,5].set(self.consts.WHITE_PIECE)
        testboard = testboard.at[0,7].set(self.consts.WHITE_PIECE)
        testboard = testboard.at[7,0].set(self.consts.BLACK_PIECE)
        testboard = testboard.at[7,2].set(self.consts.BLACK_PIECE)
        testboard = testboard.at[6,1].set(self.consts.BLACK_PIECE)
        state = VideoCheckersState(cursor_pos=jnp.array([4, 3]), board=testboard, game_phase=self.consts.SHOW_OPPONENT_MOVE_PHASE,
                                      selected_piece=jnp.array([-1, -1]), frame_counter=jnp.array(0), winner=-1, additional_jump=False,
                                      opponent_move=OpponentMove(start_pos=jnp.array([1, 0]),
                                                                  end_pos=jnp.array([7, 6]),
                                                                  piece_type=self.consts.WHITE_KING,
                                                                  captured_positions=jnp.array([[2, 1], [4, 3], [6, 5]]),
                                                                  resulting_board=board
                                                                  ), has_jumped=False, rng_key=key)"""

        # Debug state game over phase
        """state = VideoCheckersState(cursor_pos=jnp.array([4, 1]), board=board, game_phase=GAME_OVER_PHASE,
                                        selected_piece=jnp.array([-1, -1]), frame_counter=jnp.array(0), winner=COLOUR_BLACK, additional_jump=False,
                                        opponent_move=OpponentMove(start_pos=jnp.array([-1, -1]),
                                                                    end_pos=jnp.array([-1, -1]),
                                                                    piece_type=-1,
                                                                    captured_positions=jnp.array([[-1, -1]])
                                                                    ))"""

        initial_obs = self._get_observation(state)

        return initial_obs, state



    @partial(jax.jit, static_argnums=(0,))
    def step_select_piece_phase(self, state: VideoCheckersState, action: chex.Array, leg: LegalityMap) -> VideoCheckersState:
        """
        Handles moving the cursor and selecting a piece in the select piece phase.
        After a piece is selected, the game phase changes to MOVE_PIECE_PHASE.
        Args:
            state: The current game state.
            action: The action taken by the player.
            leg: Pre-computed legality map for the current board.
        Returns:
            VideoCheckersState: The new game state after the action.
        """

        def select_piece(s):
            r, c = s.cursor_pos
            # CRITICAL FIX: Ensure we aren't accessing out of bounds (though cursor should be bounded)
            # leg.can_jump is (8, 8) bool.
            valid_selection = jnp.where(leg.any_jump, leg.can_jump[r, c], leg.can_move[r, c])
            
            # Also check if there is actually a piece there belonging to us (BLACK)
            # The legality map handles piece ownership, so valid_selection is enough.
            return jax.lax.cond(
                valid_selection,
                lambda: s.replace(selected_piece=s.cursor_pos, game_phase=self.consts.MOVE_PIECE_PHASE),
                lambda: s
            )

        def move_cursor(s, act):
            # Reverted to explicit direction logic for reliability
            up = (act == Action.UPLEFT) | (act == Action.UPRIGHT)
            right = (act == Action.DOWNRIGHT) | (act == Action.UPRIGHT)
            
            drow = jnp.where(up, -1, 1)
            dcol = jnp.where(right, 1, -1)
            
            # Calculate new pos
            nr = s.cursor_pos[0] + drow
            nc = s.cursor_pos[1] + dcol
            
            # Bounds check
            in_bounds = (nr >= 0) & (nr < self.consts.NUM_FIELDS_Y) & (nc >= 0) & (nc < self.consts.NUM_FIELDS_X)
            
            # Prevent movement on FIRE or NOOP
            is_move_action = (act == Action.UPLEFT) | (act == Action.UPRIGHT) | \
                             (act == Action.DOWNRIGHT) | (act == Action.DOWNLEFT)

            return jax.lax.cond(
                in_bounds & is_move_action,
                lambda: s.replace(cursor_pos=jnp.array([nr, nc])),
                lambda: s
            )

        new_state = jax.lax.cond(
            action == Action.FIRE,
            lambda s: select_piece(s),
            lambda s: move_cursor(s, action),
            operand=state)

        return new_state

    @partial(jax.jit, static_argnums=(0,))
    def _compute_heuristic_scores(self, state: VideoCheckersState, leg: LegalityMap) -> jnp.ndarray:
        """
        Computes scores for all 8x8x4 possible moves in a single tensor sweep.
        No loops, no branching, near-instant compilation.
        Returns: (4, 8, 8) score tensor
        """
        rows, cols = jnp.indices((self.consts.NUM_FIELDS_Y, self.consts.NUM_FIELDS_X))
        board = state.board
        
        # 1. Capture Score
        # leg.moves[..., 1] is the jump-legality mask (shape: 4, 8, 8)
        capture_scores = leg.moves[..., 1].astype(jnp.float32) * self.consts.CAPTURE_W

        # 2. Advance Score (White pieces want to increase row index, so positive drow)
        # d_rows is the first element of our direction vectors
        d_rows = self.consts.MOVES[:, 0]  # Shape (4,)
        advance_mask = d_rows[:, None, None] > 0  # Shape (4, 1, 1) -> broadcasts to (4, 8, 8)
        advance_scores = advance_mask.astype(jnp.float32) * self.consts.ADVANCE_W

        # 3. Toward-Center Score
        # Distance from center for all 64 tiles
        dist_sq = (rows - self.consts.CENTER)**2 + (cols - self.consts.CENTER)**2  # (8, 8)
        
        # Calculate new distance for every possible move direction
        # We add direction vectors to current indices
        new_rows = rows[None, :, :] + self.consts.MOVES[:, 0, None, None]  # (4, 8, 8)
        new_cols = cols[None, :, :] + self.consts.MOVES[:, 1, None, None]  # (4, 8, 8)
        
        # Clip indices to keep them in-bounds for the distance lookup
        new_rows_clipped = jnp.clip(new_rows, 0, self.consts.NUM_FIELDS_Y - 1)
        new_cols_clipped = jnp.clip(new_cols, 0, self.consts.NUM_FIELDS_X - 1)
        
        new_dist_sq = (new_rows_clipped - self.consts.CENTER)**2 + (new_cols_clipped - self.consts.CENTER)**2
        toward_center = new_dist_sq < dist_sq[None, :, :]  # (4, 8, 8)
        center_scores = jnp.where(toward_center, self.consts.CENTER_FWD, self.consts.CENTER_BWD)

        # 4. Upgrade Score
        # If a white piece reaches row 7 (last row for white)
        is_white_piece = (board == self.consts.WHITE_PIECE)[None, :, :]  # (1, 8, 8) -> broadcasts
        reaches_edge = (new_rows == (self.consts.NUM_FIELDS_Y - 1)) & is_white_piece  # (4, 8, 8)
        upgrade_scores = reaches_edge.astype(jnp.float32) * self.consts.UPGRADE_W

        # 5. Total Score & Masking
        # Sum all tensors (Broadcasting handles the (4, 8, 8) shapes)
        total_scores = capture_scores + advance_scores + center_scores + upgrade_scores
        
        # Critically: Mask out illegal moves and non-AI pieces
        is_ai = ((board == self.consts.WHITE_PIECE) | (board == self.consts.WHITE_KING))[None, :, :]  # (1, 8, 8)
        # Legal moves: if any_jump, use jump_dirs, else use move_dirs
        legal_mask = jnp.where(
            leg.any_jump,
            leg.moves[..., 1],  # jump_dirs (4, 8, 8)
            leg.moves[..., 0]   # move_dirs (4, 8, 8)
        )
        
        final_mask = is_ai & legal_mask  # (4, 8, 8)
        return jnp.where(final_mask, total_scores, -1e9)

    def calculate_best_move_per_piece(self, piece: chex.Array, key, board: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """
        For a given piece, returns the best move and its score.
        Args:
            piece: array containing the pieces row and col
            key: an RNG key used for breaking ties
            board: current game board
        Returns:
            - best move (drow, dcol)
            - best move score
        """
        row, col = piece
        piece_type = board[row, col]

        moves = self.get_possible_moves_for_piece(row, col, board)
        drow, dcol = moves[:, 0], moves[:, 1]
        is_jump = jnp.all(jnp.abs(moves) == 2, axis=1)

        # “Forward” move for regular pieces, opposite for kings
        white_piece = piece_type == self.consts.WHITE_PIECE
        white_king = piece_type == self.consts.WHITE_KING
        advance = jnp.where(
            white_piece, drow == -1,
            jnp.where(white_king, drow == 1, False)
        )

        # Toward-centre test using squared distance (avoids sqrt)
        new_row = row + drow
        new_col = col + dcol
        old_d2 = (row - self.consts.CENTER) ** 2 + (col - self.consts.CENTER) ** 2
        new_d2 = (new_row - self.consts.CENTER) ** 2 + (new_col - self.consts.CENTER) ** 2
        toward = new_d2 < old_d2

        # Reaches edge
        reaches_upgrade = (new_row == 7) & white_piece

        # no move punish
        no_move = (drow == 0) | (dcol == 0)

        scores = (
                self.consts.CAPTURE_W * is_jump.astype(jnp.float32)
                + self.consts.ADVANCE_W * advance.astype(jnp.float32)
                + self.consts.UPGRADE_W * reaches_upgrade.astype(jnp.float32)
                + self.consts.CENTER_FWD * toward.astype(jnp.float32)
                + self.consts.CENTER_BWD * (~toward).astype(jnp.float32)
        )

        # Apply mask via where (avoid mixing -inf into sums)
        scores = jnp.where(~no_move, scores, -jnp.inf)

        # Tie-breaker noise only on legal entries; keep masked as -inf
        key, subkey = jax.random.split(key)
        noise = 0.01 * jax.random.normal(subkey, scores.shape, dtype=scores.dtype)
        scores = jnp.where(~no_move, scores + noise, scores)

        all_illegal = jnp.all(no_move)
        # When all illegal, argmax will be 0; preserve score -inf and return dummy move
        best_idx = jnp.argmax(scores)
        best_move = jnp.where(all_illegal, jnp.array([0, 0], moves.dtype), moves[best_idx])
        best_score = jnp.where(all_illegal, -jnp.inf, scores[best_idx])

        return best_move, best_score

    def calculate_best_first_opponent_move(self, movable_pieces: chex.Array, state: VideoCheckersState) \
            -> VideoCheckersState:
        """
        Computes the first move (of potentially multiple) using vectorized heuristic scoring.
        Args:
            movable_pieces: array of size MAX_PIECES containing row and col position for movable pieces and -1,-1 for imovable
            state: current game state
        Returns:
            new state with updated opponent_move and has_jumped fields
        """
        # Pre-compute legality for AI (white)
        leg = self.get_global_legality(state.board, self.consts.COLOUR_WHITE)
        
        # Compute scores for all moves at once
        scores = self._compute_heuristic_scores(state, leg)  # (4, 8, 8)
        
        # Add tie-breaker noise
        rng_key, subkey = jax.random.split(state.rng_key)
        noise = 0.01 * jax.random.normal(subkey, scores.shape, dtype=scores.dtype)
        scores = scores + noise
        
        # Find best move across all (direction, row, col)
        flat_idx = jnp.argmax(scores.flatten())
        best_dir, best_row, best_col = jnp.unravel_index(flat_idx, scores.shape)
        
        move_vec = self.consts.MOVES[best_dir]
        is_jump = leg.jump_dirs[best_dir, best_row, best_col]
        
        # Check validity logic handled by -1e9 mask in _compute_heuristic_scores
        final_move = jnp.where(leg.any_jump & is_jump, move_vec * 2, move_vec)
        start_pos = jnp.array([best_row, best_col])

        # Execute
        new_board, new_piece, did_jump, c_r, c_c = self.move_piece(
            best_row, best_col, final_move[0], final_move[1], state.board)
        
        opp_move = state.opponent_move.replace(
            start_pos=start_pos, end_pos=start_pos + final_move, piece_type=new_piece, resulting_board=new_board
        )
        
        opp_move = jax.lax.cond(did_jump, 
            lambda: self.add_captured_position(opp_move, jnp.array([c_r, c_c])),
            lambda: opp_move)
        
        return state.replace(has_jumped=did_jump, opponent_move=opp_move, rng_key=rng_key)

    def calculate_best_further_opponent_move(self, state: VideoCheckersState) \
            -> VideoCheckersState:
        """
        Handles the case where the opponent has already jumped and can jump again.
        Uses static unrolled loop (max 4 jumps) for faster compilation.
        Args:
            state: current game state
        Returns:
            new state with updated opponent_move and has_jumped fields
        """
        def jump_step(i, s: VideoCheckersState) -> VideoCheckersState:
            # Only process if we still have a jump available
            def process_jump(s_inner):
                # Calculate legality for the current board state
                leg = self.get_global_legality(s_inner.opponent_move.resulting_board, self.consts.COLOUR_WHITE)
                
                # Only check the piece that just moved
                curr_pos = s_inner.opponent_move.end_pos
                can_jump_again = leg.can_jump[curr_pos[0], curr_pos[1]]
                
                def perform_jump():
                    # Create a temporary state for scoring
                    temp_state = s_inner.replace(board=s_inner.opponent_move.resulting_board)
                    scores = self._compute_heuristic_scores(temp_state, leg)  # (4, 8, 8)
                    
                    # Mask to only the current piece position
                    piece_scores = scores[:, curr_pos[0], curr_pos[1]]  # (4,)
                    # Only consider jump moves
                    jump_scores = jnp.where(leg.jump_dirs[:, curr_pos[0], curr_pos[1]], piece_scores, -1e9)
                    
                    # Add noise for tie-breaking
                    rng_key, subkey = jax.random.split(s_inner.rng_key)
                    noise = 0.01 * jax.random.normal(subkey, jump_scores.shape, dtype=jump_scores.dtype)
                    jump_scores = jump_scores + noise
                    
                    best_dir = jnp.argmax(jump_scores)
                    best_move = self.consts.MOVES[best_dir] * 2  # Jump is 2x move
                    
                    new_board, new_piece, is_jump, c_row, c_col = self.move_piece(
                        row=curr_pos[0], col=curr_pos[1], drow=best_move[0], dcol=best_move[1],
                        board=s_inner.opponent_move.resulting_board)
                    new_pos = curr_pos + best_move

                    new_opponent_move = s_inner.opponent_move.replace(end_pos=new_pos, piece_type=new_piece,
                                                                     resulting_board=new_board)
                    new_opponent_move = self.add_captured_position(new_opponent_move,
                                                                  jnp.array((c_row, c_col), dtype=jnp.int32))
                    return s_inner.replace(
                        has_jumped=is_jump,
                        opponent_move=new_opponent_move,
                        rng_key=rng_key
                    )
                
                def skip_jump():
                    return s_inner.replace(has_jumped=False)
                
                return jax.lax.cond(can_jump_again, perform_jump, skip_jump)
            
            def skip_step(s_inner):
                return s_inner.replace(has_jumped=False)
            
            return jax.lax.cond(s.has_jumped, process_jump, skip_step, s)

        return jax.lax.fori_loop(0, 4, jump_step, state)

    def step_move_piece_phase(self, state: VideoCheckersState, action: chex.Array, leg: LegalityMap) -> VideoCheckersState:
        """
        Handles moving a piece in the move piece phase.
        Args:
            state: The current game state.
            action: The action taken by the player.
        Returns:
            VideoCheckersState: The new game state after the action.
        """

        def prepare_opponent_move(state: VideoCheckersState) -> VideoCheckersState:
            """
            Prepares the opponent's move by changing the game phase to SHOW_OPPONENT_MOVE_PHASE.
            This is called when the player has moved a piece and no further jumps are available.
            In this function we determine the opponent's move.
            The resulting opponent move is stored in the state with
            the start position, end position, captured positions and what type the moved piece is at the end position.
            the calculate_best_opponent_move(board,allowed_moves) function is used to determine the opponent's move.
            """

            def moves_available(moveable_pieces: chex.Array, state: VideoCheckersState):
                new_state = self.calculate_best_first_opponent_move(moveable_pieces, state)
                # if the opponent has jumped, we need to check calculate_best_further_opponent_move
                opponent_move = jax.lax.cond(
                    new_state.has_jumped,
                    lambda: self.calculate_best_further_opponent_move(new_state).opponent_move,
                    lambda: new_state.opponent_move,
                )

                return state.replace(
                    game_phase=self.consts.SHOW_OPPONENT_MOVE_PHASE,
                    has_jumped=False,
                    opponent_move=opponent_move,
                    selected_piece=jnp.array([-1, -1]),  # Reset the selected piece
                    rng_key=new_state.rng_key,
                )

            # For AI (white), still use get_movable_pieces since it's called less frequently
            # The player legality is pre-computed, but AI moves are calculated on-demand
            movable_pieces, _ = self.get_movable_pieces(self.consts.COLOUR_WHITE, state.board)

            return jax.lax.cond(
                jnp.all(movable_pieces == jnp.array([-1, -1])),
                lambda: state.replace(game_phase=self.consts.GAME_OVER_PHASE, winner=self.consts.COLOUR_BLACK),
                lambda: moves_available(movable_pieces, state)
            )

        def move_cursor_logic(s, act):
            is_on_selected = jnp.all(s.selected_piece == s.cursor_pos)
            
            def move_back(s_in, action):
                # Calculate vector back to selected piece
                diff = s_in.selected_piece - s_in.cursor_pos
                
                # Determine input vector based on Action
                input_vec = jnp.array([0, 0])
                input_vec = jnp.where(action == Action.UPRIGHT, jnp.array([-1, 1]), input_vec)
                input_vec = jnp.where(action == Action.UPLEFT, jnp.array([-1, -1]), input_vec)
                input_vec = jnp.where(action == Action.DOWNRIGHT, jnp.array([1, 1]), input_vec)
                input_vec = jnp.where(action == Action.DOWNLEFT, jnp.array([1, -1]), input_vec)

                # Normalize diff to unit direction (1 step)
                # If diff is +/- 2 (jump), we divide by 2. If +/- 1, divide by 1.
                dist = jnp.max(jnp.abs(diff)) # Will be 1 or 2
                norm_diff = diff // dist 
                
                # Check if input matches the direction back
                is_correct_dir = jnp.all(input_vec == norm_diff) & (jnp.any(input_vec != 0))
                
                return jax.lax.cond(is_correct_dir, lambda: s_in.replace(cursor_pos=s_in.selected_piece), lambda: s_in)

            def move_away(s_in, action):
                # FIX: Manual mapping of Action -> Vector
                # Row is Y (Down is +), Col is X (Right is +)
                # UPRIGHT: Row -1, Col +1
                # UPLEFT:  Row -1, Col -1
                # DOWNRIGHT: Row +1, Col +1
                # DOWNLEFT:  Row +1, Col -1
                
                vec_move = jnp.array([0, 0]) # Default
                vec_move = jnp.where(action == Action.UPRIGHT, jnp.array([-1, 1]), vec_move)
                vec_move = jnp.where(action == Action.UPLEFT, jnp.array([-1, -1]), vec_move)
                vec_move = jnp.where(action == Action.DOWNRIGHT, jnp.array([1, 1]), vec_move)
                vec_move = jnp.where(action == Action.DOWNLEFT, jnp.array([1, -1]), vec_move)
                
                is_move_action = jnp.any(vec_move != 0)
                
                # Check legality via the map
                # We need to find WHICH index in leg.moves corresponds to this vector
                # The MOVES constant is: [[1, 1], [-1, 1], [-1, -1], [1, -1]]
                # 0: DownRight, 1: UpRight, 2: UpLeft, 3: DownLeft
                
                # Let's map Action -> Index manually
                idx = -1
                idx = jnp.where(action == Action.DOWNRIGHT, 0, idx)
                idx = jnp.where(action == Action.UPRIGHT, 1, idx)
                idx = jnp.where(action == Action.UPLEFT, 2, idx)
                idx = jnp.where(action == Action.DOWNLEFT, 3, idx)
                
                r, c = s_in.selected_piece
                
                # If idx is -1 (invalid action), valid_* will be false (safe access via clip or condition)
                # Safe access trick: use idx 0 if invalid, but gate with is_move_action
                safe_idx = jnp.maximum(idx, 0)
                
                can_move_dir = leg.moves[safe_idx, r, c, 0]
                can_jump_dir = leg.moves[safe_idx, r, c, 1]
                
                # Apply
                new_pos_move = s_in.cursor_pos + vec_move
                new_pos_jump = s_in.cursor_pos + (vec_move * 2)
                
                return jax.lax.cond(
                    is_move_action & (idx != -1) & can_jump_dir,
                    lambda: s_in.replace(cursor_pos=new_pos_jump),
                    lambda: jax.lax.cond(
                        is_move_action & (idx != -1) & can_move_dir,
                        lambda: s_in.replace(cursor_pos=new_pos_move),
                        lambda: s_in
                    )
                )

            return jax.lax.cond(is_on_selected, lambda: move_away(s, act), lambda: move_back(s, act))

        def place_piece(s):
            cursor_on_sel = jnp.all(s.selected_piece == s.cursor_pos)
            
            def deselect(s_in):
                return jax.lax.cond(
                    s_in.has_jumped, 
                    lambda: s_in, 
                    lambda: s_in.replace(selected_piece=jnp.array([-1,-1]), game_phase=self.consts.SELECT_PIECE_PHASE)
                )
            
            def commit_move(s_in):
                piece = s_in.board[s_in.selected_piece[0], s_in.selected_piece[1]]
                move = s_in.cursor_pos - s_in.selected_piece
                is_jump = jnp.max(jnp.abs(move)) > 1
                
                # Crown King logic
                is_king_row = s_in.cursor_pos[0] == 0
                # Ensure we only crown WHITE_PIECE (1) -> WHITE_KING (3) or BLACK_PIECE (2) -> BLACK_KING (4)
                # Here we are Black player (row 0 is our target)
                new_piece = jnp.where(
                    (piece == self.consts.BLACK_PIECE) & is_king_row, 
                    self.consts.BLACK_KING, 
                    piece
                )
                
                new_board = s_in.board.at[tuple(s_in.selected_piece)].set(0).at[tuple(s_in.cursor_pos)].set(new_piece)
                
                # Handle capture
                cap_pos = s_in.selected_piece + (move // 2)
                new_board = jax.lax.cond(is_jump, lambda: new_board.at[tuple(cap_pos)].set(0), lambda: new_board)
                
                # Check for double jump possibility
                new_leg = self.get_global_legality(new_board, self.consts.COLOUR_BLACK)
                can_jump_again = new_leg.can_jump[s_in.cursor_pos[0], s_in.cursor_pos[1]]
                
                return jax.lax.cond(
                    is_jump & can_jump_again,
                    lambda: s_in.replace(board=new_board, selected_piece=s_in.cursor_pos, has_jumped=True, must_jump=True),
                    lambda: prepare_opponent_move(s_in.replace(board=new_board))
                )

            return jax.lax.cond(cursor_on_sel, lambda: deselect(s), lambda: commit_move(s))

        new_state = jax.lax.cond(
            action == Action.FIRE,
            lambda s: place_piece(s),
            lambda s: move_cursor_logic(s, action),
            operand=state
        )

        return new_state

    def step_show_opponent_move_phase(self, state, action):
        # Applies stored move
        return jax.lax.cond(
            action == Action.NOOP,
            lambda: state,
            lambda: jax.lax.cond(
                self.count_pieces(state.opponent_move.resulting_board)[1] == 0, # Check black count
                lambda: state.replace(board=state.opponent_move.resulting_board, game_phase=self.consts.GAME_OVER_PHASE, winner=self.consts.COLOUR_WHITE),
                lambda: state.replace(board=state.opponent_move.resulting_board, 
                                      opponent_move=self.clear_captured_positions(state.opponent_move),
                                      game_phase=self.consts.SELECT_PIECE_PHASE, selected_piece=jnp.array([-1,-1]))
            )
        )

    def step_game_over_phase(self, state: VideoCheckersState, action: chex.Array) -> VideoCheckersState:
        """
        Handles the game over phase, where the game is finished and no further actions are taken.
        Args:
            state: The current game state.
            action: The action taken by the player (ignored in this phase).
        Returns:
            VideoCheckersState: The new game state after the action.
        """
        return state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: VideoCheckersState, action: chex.Array) -> Tuple[
        VideoCheckersObservation, VideoCheckersState, float, bool, VideoCheckersInfo]:
        """
        Takes a step in the game environment based on the action taken.
        Args:
            state: The current game state.
            action: The action taken by the player.
        Returns:
            observation: The new observation of the game state.
            new_state: The new game state after taking the action.
            reward: The reward received after taking the action.
            done: A boolean indicating if the game is over.
            info: Additional information about the game state.
        """
        atari_action = jnp.take(self.ACTION_SET, action.astype(jnp.int32))
        
        # Calculate legality ONCE at top of step
        leg = self.get_global_legality(state.board, self.consts.COLOUR_BLACK)
        
        def process_frame(s):
            return jax.lax.switch(s.game_phase, [
                lambda: self.step_select_piece_phase(s, atari_action, leg),
                lambda: self.step_move_piece_phase(s, atari_action, leg),
                lambda: self.step_show_opponent_move_phase(s, atari_action),
                lambda: s # Game over
            ])

        should_act = (state.frame_counter == (self.consts.ANIMATION_FRAME_RATE - 1)) & (atari_action != Action.NOOP)
        new_state = jax.lax.cond(should_act, lambda: process_frame(state), lambda: state)
        
        new_state = new_state.replace(frame_counter=(new_state.frame_counter + 1) % self.consts.ANIMATION_FRAME_RATE)
        
        # Update must_jump for obs
        # If board changed, strictly we should re-calc legality, but for Observation purposes 
        # using the legality of the start of next frame is correct.
        # We can optimize by only calc'ing if phase is SELECT or MOVE
        final_leg = self.get_global_legality(new_state.board, self.consts.COLOUR_BLACK)
        new_state = new_state.replace(must_jump=final_leg.any_jump)
        
        done = new_state.game_phase == self.consts.GAME_OVER_PHASE
        reward = self._get_env_reward(state, new_state)
        
        return self._get_observation(new_state), new_state, reward, done, self._get_info(new_state)

    def action_space(self):
        """
        Returns the action space of the game environment.
        Returns:
            action_space: The action space of the game environment.
        """
        return spaces.Discrete(len(self.ACTION_SET))

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.consts.HEIGHT, self.consts.WIDTH, 3),
            dtype=jnp.uint8,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: VideoCheckersState) -> VideoCheckersInfo:
        """
        Returns additional information about the game state.
        Args:
            state: The current game state.
        Returns:
            VideoCheckersInfo: Additional information about the game state.
        """
        return VideoCheckersInfo()

    def observation_space(self) -> spaces.Dict:
        """
        Returns the observation space of the game environment.
        The observation contains:
        - board: array of shape (8, 8) representing the game board
        - start_pos: array of shape (2,) representing the starting position of the selected piece
        - end_pos: array of shape (2,) representing the ending position of the selected piece
        - must_jump: boolean indicating if the player must jump
        - cursor_pos: array of shape (2,) representing the position of the cursor
        """
        return spaces.Dict({
            'board': spaces.Box(low=0,high=160, shape=(8, 8), dtype=jnp.int32),
            'start_pos': spaces.Box(low=-1,high=7, shape=(2,), dtype=jnp.int32),
            'end_pos': spaces.Box(low=-1,high=7, shape=(2,), dtype=jnp.int32),
            'must_jump': spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
            'cursor_pos': spaces.Box(low=0, high=7, shape=(2,), dtype=jnp.int32),
        })

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: VideoCheckersState):
        return VideoCheckersObservation(board=state.board, start_pos=state.cursor_pos,
                                        end_pos=state.selected_piece, must_jump=state.must_jump.astype(jnp.int32),
                                        cursor_pos=state.cursor_pos)

    @partial(jax.jit, static_argnums=(0,))
    def _get_env_reward(self, previous_state: VideoCheckersState, state: VideoCheckersState) -> float:
        p_c = self.count_pieces(previous_state.board)
        n_c = self.count_pieces(state.board)
        return (n_c[1] - n_c[0]) - (p_c[1] - p_c[0])

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: VideoCheckersState, state: VideoCheckersState) -> float:
        """
        Calculates the reward from the environment state.
        Args:
            previous_state: The previous environment state.
            state: The environment state.
        Returns: reward
        """
        return self._get_env_reward(previous_state, state)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: VideoCheckersState) -> bool:
        """
        Returns whether the game is done based on the game state.
        Args:
            state: The current game state.
        """
        return state.game_phase == self.consts.GAME_OVER_PHASE




class VideoCheckersRenderer(JAXGameRenderer):
    def __init__(self, consts: VideoCheckersConstants = None):
        super().__init__()
        self.consts = consts or VideoCheckersConstants()
        
        # 1. Configure the renderer
        self.config = render_utils.RendererConfig(
            game_dimensions=(self.consts.HEIGHT, self.consts.WIDTH),
            channels=3,
            #downscale=(84, 84)
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)

        # 2. Define sprite path
        sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/videocheckers"
        
        # 3. Start from (possibly modded) asset config provided via constants
        final_asset_config = list(self.consts.ASSET_CONFIG)
        
        # 4. Create procedural assets using modded constants
        background_sprite = jnp.array([[[160, 96, 64, 255]]], dtype=jnp.uint8)
        
        # 5. Append procedural assets
        final_asset_config.insert(0, {'name': 'background', 'type': 'background', 'data': background_sprite})
        
        # 6. Load all assets, create palette, and generate ID masks
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(final_asset_config, sprite_path)
        
        # 5. Expand background to full size if it's procedural (1x1)
        # The procedural background creates a 1x1 raster, but we need full game dimensions
        bg_h, bg_w = self.BACKGROUND.shape
        # Determine the correct target size (use downscaled dimensions if downscaling is enabled)
        if self.config.downscale:
            target_h, target_w = self.config.downscale
        else:
            target_h, target_w = self.config.game_dimensions
        if bg_h != target_h or bg_w != target_w:
            # Get the background color ID from the 1x1 background
            bg_color_id = self.BACKGROUND[0, 0]
            # Create full-size background filled with that color ID
            self.BACKGROUND = jnp.full((target_h, target_w), bg_color_id, dtype=self.BACKGROUND.dtype)
        
        # 7. Pre-compute/cache values for rendering
        self._cache_sprite_stacks()
        self.PRE_RENDERED_BOARD = self._precompute_static_board()

    def _cache_sprite_stacks(self):
        """Caches the sprite stacks for easy access."""
        self.PIECE_STACK = self.SHAPE_MASKS['pieces']
        self.TEXT_STACK = self.SHAPE_MASKS['text']

    def _precompute_static_board(self) -> jnp.ndarray:
        """Pre-renders the static board onto the solid background color."""
        # self.BACKGROUND is already the solid color ID mask
        board_mask = self.SHAPE_MASKS['board']
        return self.jr.render_at(
            self.BACKGROUND,
            self.consts.OFFSET_X_BOARD,
            self.consts.OFFSET_Y_BOARD,
            board_mask
        )

    # --- Vectorized Logic for Piece Calculation ---
    
    @partial(jax.jit, static_argnums=(0,))
    def _calculate_piece_grid(self, state: VideoCheckersState) -> jnp.ndarray:
        """
        Replaced 64-way vmap+switch with global board masking.
        """
        rows, cols = jnp.indices((self.consts.NUM_FIELDS_Y, self.consts.NUM_FIELDS_X))
        board = state.board
        
        # Common masks
        is_cursor = (rows == state.cursor_pos[0]) & (cols == state.cursor_pos[1])
        is_selected = (rows == state.selected_piece[0]) & (cols == state.selected_piece[1])
        flash_on = state.frame_counter < (self.consts.ANIMATION_FRAME_RATE // 2)
        
        # SELECT PHASE LOGIC
        select_grid = jnp.where(is_cursor & flash_on, self.consts.BLACK_CURSOR, board)

        # MOVE PHASE LOGIC
        # Flash the cursor position with the piece that is currently being "held"
        # Safe access: check if selected_piece is valid (not [-1, -1])
        selected_valid = (state.selected_piece[0] >= 0) & (state.selected_piece[1] >= 0)
        selected_piece_type = jnp.where(
            selected_valid,
            board[state.selected_piece[0], state.selected_piece[1]],
            self.consts.EMPTY_TILE
        )
        move_grid = jnp.where(is_selected & flash_on & selected_valid, self.consts.BLACK_CURSOR, board)
        move_grid = jnp.where(is_cursor & flash_on & selected_valid, selected_piece_type, move_grid)
        # If cursor is ON selected piece (unmoved), handle special triple-flash or similar
        is_unmoved = is_selected & is_cursor
        frame_mod = state.frame_counter % self.consts.ANIMATION_FRAME_RATE
        is_flashing_off_unmoved = ((frame_mod >= 5) & (frame_mod < 10)) | (frame_mod >= 15)
        move_grid = jnp.where(is_unmoved & is_flashing_off_unmoved, self.consts.EMPTY_TILE, move_grid)

        # OPPONENT MOVE LOGIC
        opp = state.opponent_move
        is_start = (rows == opp.start_pos[0]) & (cols == opp.start_pos[1])
        is_end = (rows == opp.end_pos[0]) & (cols == opp.end_pos[1])
        
        # Check all captured slots at once
        captured_mask = jnp.any(
            (opp.captured_positions[:, 0, None, None] == rows) & 
            (opp.captured_positions[:, 1, None, None] == cols), axis=0
        )

        opp_grid_before = jnp.where(is_start, self.consts.WHITE_CURSOR, 
                                   jnp.where(is_end, self.consts.EMPTY_TILE, board))
        opp_grid_after = jnp.where(is_start, self.consts.EMPTY_TILE,
                                  jnp.where(is_end, opp.piece_type,
                                           jnp.where(captured_mask, self.consts.BLACK_CURSOR, board)))
        
        opp_grid = jnp.where(flash_on, opp_grid_before, opp_grid_after)

        # Switch ONCE for the whole board
        return jax.lax.switch(
            state.game_phase,
            [lambda: select_grid, lambda: move_grid, lambda: opp_grid, lambda: board]
        )

    # --- JIT-ted Render Helpers ---

    @partial(jax.jit, static_argnums=(0,))
    def _render_pieces_on_board(self, piece_grid: jnp.ndarray, raster: jnp.ndarray) -> jnp.ndarray:
        """
        OPTIMIZED: Instead of 64 loops, we only loop over actually occupied dark squares.
        """
        # Flatten the grid for easier iteration
        flat_grid = piece_grid.flatten()
        
        # Checkered pattern mask
        rows, cols = jnp.indices((self.consts.NUM_FIELDS_Y, self.consts.NUM_FIELDS_X))
        dark_squares = ((rows + cols) % 2 == 1).flatten()
        
        def draw_step(i, current_raster):
            row, col = i // self.consts.NUM_FIELDS_X, i % self.consts.NUM_FIELDS_X
            piece_idx = flat_grid[i]
            
            # Use data-driven check instead of nested lax.cond where possible
            should_draw = dark_squares[i] & (piece_idx != self.consts.EMPTY_TILE)
            
            def perform_draw(r):
                x = self.consts.OFFSET_X_BOARD + 4 + col * 17
                y = self.consts.OFFSET_Y_BOARD + 2 + row * 13
                return self.jr.render_at(r, x, y, self.PIECE_STACK[piece_idx])
            
            return jax.lax.cond(should_draw, perform_draw, lambda r: r, current_raster)

        # Still a loop, but much lighter because the logic inside perform_draw 
        # is now a simple array update rather than complex logic.
        total_squares = self.consts.NUM_FIELDS_Y * self.consts.NUM_FIELDS_X
        return jax.lax.fori_loop(0, total_squares, draw_step, raster)

    @partial(jax.jit, static_argnums=(0,))
    def _render_jump_indicator(self, must_jump: bool, raster: jnp.ndarray) -> jnp.ndarray:
        """Renders the 'JP' (must jump) indicator."""
        
        def _render(r):
            x_offset = 100
            j_sprite = self.TEXT_STACK[10] # 'J'
            p_sprite = self.TEXT_STACK[11] # 'P'
            r = self.jr.render_at(r, x_offset, 20, j_sprite)
            r = self.jr.render_at(r, x_offset + 10, 20, p_sprite)
            return r

        return jax.lax.cond(must_jump, _render, lambda r: r, raster)

    # --- Main Render Function ---

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: VideoCheckersState):
        """
        Renders the current game state using JAX operations.
        """
        # 1. Start with the pre-rendered static board
        raster = self.PRE_RENDERED_BOARD

        # 2. Calculate the 8x8 grid of piece indices to draw
        # This is where all the complex logic happens (vectorized)
        piece_grid = self._calculate_piece_grid(state)

        # 3. Render the pieces based on the calculated grid
        raster = self._render_pieces_on_board(piece_grid, raster)
        
        # 4. Render the jump indicator (read from state instead of recalculating)
        raster = self._render_jump_indicator(state.must_jump, raster)

        # 5. Final conversion from palette IDs to RGB
        return self.jr.render_from_palette(raster, self.PALETTE)
