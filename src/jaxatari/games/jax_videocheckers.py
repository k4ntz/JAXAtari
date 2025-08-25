import os
from functools import partial
from typing import NamedTuple, Tuple
import jax.lax
import jax.numpy as jnp
import chex

import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as jr
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action, EnvObs
from jaxatari.spaces import Space


#
# by Tim Morgner and Jan Larionow
#

# IMPORTANT: WE, THE PLAYER, PLAY AS BLACK, THE OPPONENT IS WHITE.

# Transitions between game phases
# SELECT_PIECE -> MOVE_PIECE: # Player selects a piece to move
# MOVE_PIECE -> SHOW_OPPONENT_MOVE: # Player moves the piece with no further jumps available
# MOVE_PIECE -> MOVE_PIECE: # Player moves the piece with further jumps available
# SHOW_OPPONENT_MOVE -> SELECT_PIECE: # Player makes an input to select a piece after the opponent's move


class VideoCheckersConstants:
    MAX_PIECES = 12

    COLOUR_WHITE: int = 0
    COLOUR_BLACK: int = 1

    WIDTH: int = 160
    HEIGHT: int = 210

    SCALING_FACTOR = 3
    WINDOW_WIDTH = WIDTH * SCALING_FACTOR
    WINDOW_HEIGHT = HEIGHT * SCALING_FACTOR

    OFFSET_X_BOARD = 12
    OFFSET_Y_BOARD = 50

    MOVES = jnp.array([
        [1, 1],  # UPRIGHT
        [-1, 1],  # DOWNRIGHT
        [-1, -1],  # DOWNLEFT
        [1, -1],  # UPLEFT
    ])

    # Opponent move scoring
    CAPTURE_W = 10.0  # capturing opponent piece
    UPGRADE_W = 5  # upgrading piece
    ADVANCE_W = 1.0  # moving forward
    CENTER_FWD = 0.5  # moving towards center
    CENTER_BWD = -2.0  # moving away from center
    NO_MOVE = -jnp.inf  # avoid standing still

    EMPTY_TILE = 0
    WHITE_PIECE = 1
    BLACK_PIECE = 2
    WHITE_KING = 3
    BLACK_KING = 4
    WHITE_CURSOR = 5
    BLACK_CURSOR = 6

    NUM_FIELDS_X = 8
    NUM_FIELDS_Y = 8
    CENTER = 3.5

    SELECT_PIECE_PHASE = 0
    MOVE_PIECE_PHASE = 1
    SHOW_OPPONENT_MOVE_PHASE = 2
    GAME_OVER_PHASE = 3

    ANIMATION_FRAME_RATE = 30


class OpponentMove(NamedTuple):
    start_pos: chex.Array  # Start position of the opponent's piece
    end_pos: chex.Array  # End position of the opponent's piece
    piece_type: int  # Type of the piece at the end position (king or normal)
    captured_positions: chex.Array  # Array of positions of captured pieces
    resulting_board: chex.Array  # New board with all moves applied


class OpponentMoveHandler:
    """
    Handles modifications to the opponent's move. All methods should be of type OpponentMove, args -> OpponentMove.
    """

    @staticmethod
    def add_captured_position(opponent_move: OpponentMove, position: chex.Array) -> OpponentMove:
        # 1. find empty slot in array to store the new position
        matches = jnp.all(opponent_move.captured_positions == jnp.array([-1, -1]), axis=1)
        first_index = jnp.argmax(matches)
        has_match = jnp.any(matches)

        # 2. get new opponent move, if an empty slot was found (guaranteed), do nothing if not
        return jax.lax.cond(
            has_match,
            lambda o: o._replace(captured_positions=o.captured_positions.at[first_index].set(position)),
            lambda o: o,
            opponent_move
        )

    @staticmethod
    def clear_captured_positions(opponent_move: OpponentMove) -> OpponentMove:
        """
        Resets the array containing the captured positions back to only sentinel positions.
        Args:
            opponent_move: opponent move with the captured positions array to be reset.
        Returns:
            new OpponentMove with a clean captures positions array.
        """
        return opponent_move._replace(captured_positions=jnp.full(opponent_move.captured_positions.shape, -1))


class VideoCheckersState(NamedTuple):
    board: chex.Array  # Shape (NUM_FIELDS_Y, NUM_FIELDS_X)
    game_phase: int
    cursor_pos: chex.Array
    has_jumped: chex.Array  # True if the selected piece has already jumped, so the player cant deselect it.
    additional_jump: bool  # True if in the MOVE_PIECE_PHASE a there has already been a jump, so the player can jump again. This prevents the player from deselecting the piece.
    selected_piece: chex.Array
    frame_counter: chex.Array
    opponent_move: OpponentMove
    winner: int  # -1 if no winner, COLOUR_WHITE if white won, COLOUR_BLACK if black won.
    rng_key: chex.PRNGKey


class VideoCheckersObservation(NamedTuple):
    board: chex.Array
    start_pos: chex.Array
    end_pos: chex.Array
    must_jump: chex.Array
    cursor_pos: chex.Array


class VideoCheckersInfo(NamedTuple):
    all_rewards: chex.Array


class BoardHandler:
    """Handles modifications to the board, as well as board reliant operations."""

    @staticmethod
    def reset_board():
        """
        Returns a clean board with all pieces in their initial position.
        Returns:
            new board with all pieces in their initial position.
        """
        # Initialize the board with pieces, this is a placeholder
        board = jnp.zeros((VideoCheckersConstants.NUM_FIELDS_X,
                           VideoCheckersConstants.NUM_FIELDS_Y), dtype=jnp.int32)
        # Set up the initial pieces on the board
        white_rows = jnp.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        white_cols = jnp.array([1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7])

        black_rows = jnp.array([5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7])
        black_cols = jnp.array([0, 2, 4, 6, 1, 3, 5, 7, 0, 2, 4, 6])

        board = board.at[white_rows, white_cols].set(VideoCheckersConstants.WHITE_PIECE)
        board = board.at[black_rows, black_cols].set(VideoCheckersConstants.BLACK_PIECE)

        return board

    @staticmethod
    def move_piece(row, col, drow, dcol, board) -> (jnp.ndarray, int, bool, int, int):
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

        upgrade_white = piece == VideoCheckersConstants.WHITE_PIECE & (new_row == 7)
        upgrade_black = piece == VideoCheckersConstants.BLACK_PIECE & (new_row == 0)
        new_piece = jax.lax.cond(
            upgrade_white,
            lambda: VideoCheckersConstants.WHITE_KING,
            lambda: jax.lax.cond(
                upgrade_black,
                lambda: VideoCheckersConstants.BLACK_KING,
                lambda: piece)
        )

        new_board = (board
                     .at[(row, col)].set(VideoCheckersConstants.EMPTY_TILE)
                     .at[(new_row, new_col)].set(new_piece))

        # 2. handle capture
        is_jump = (jnp.abs(drow) == 2) & (jnp.abs(dcol) == 2)

        def _handle_capture(board):
            captured_row = row + drow // 2
            captured_col = col + dcol // 2
            return (board.at[(captured_row, captured_col)].set(VideoCheckersConstants.EMPTY_TILE),
                    captured_row, captured_col)

        new_board, captured_row, captured_col = jax.lax.cond(
            is_jump,
            _handle_capture,
            lambda b: (b, -1, -1),
            new_board
        )

        return new_board, new_piece, is_jump, captured_row, captured_col

    @staticmethod
    def tile_is_free(row, col, board):
        """
        Args:
            row: row of tile to check
            col: column of tile to check
            board: current game board
        """
        return board[row, col] == VideoCheckersConstants.EMPTY_TILE

    @staticmethod
    def move_in_bounds(row, col, drow, dcol):
        """
        Checks if move can be made in the given direction.
        Args:
            row: row index of the piece
            col: column index of the piece
            drow: movement in y direction
            dcol: movement in x direction

        Returns: True, if cursor can be moved in the given direction, False otherwise.
        """
        return (((0 <= row + drow) & (row + drow < VideoCheckersConstants.NUM_FIELDS_Y))
                & ((0 <= col + dcol) & (col + dcol < VideoCheckersConstants.NUM_FIELDS_X)))

    @staticmethod
    def get_possible_moves_for_piece(row, col, board: chex.Array):
        """
        Get all possible moves for a piece at position (row,col)
        Args:
           row: row index of the piece
           col: column index of the piece
           board: current game board

        Returns: array of all possible moves. If a move in a given direction is not possible, it returns [0,0]
        """

        is_not_a_piece = (BoardHandler.tile_is_free(row, col, board)) | (row == -1)

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
                jump_available = BoardHandler.move_is_available(row=row, col=col, drow=2 * drow, dcol=2 * dcol,
                                                                board=board)  # check jump
                move_available = BoardHandler.move_is_available(row=row, col=col, drow=drow, dcol=dcol,
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

            possible_moves = jax.vmap(check_move)(VideoCheckersConstants.MOVES)
            # if there is a jump in the possible moves, make non jump moves [0,0]
            has_jump = jnp.any(jnp.all(jnp.abs(possible_moves) == 2, axis=1))
            possible_moves = jax.lax.cond(
                has_jump,
                lambda: prune_possible_moves_for_jumps(possible_moves),
                lambda: possible_moves,

            )
            return possible_moves

        return jax.lax.cond(is_not_a_piece, lambda: jnp.zeros((4, 2), dtype=jnp.int32), _get_moves)

    @staticmethod
    def count_pieces(board: jnp.ndarray) -> (float, float):
        cnt_white = jnp.sum(board == 1) + jnp.sum(board == 3)
        cnt_black = jnp.sum(board == 2) + jnp.sum(board == 4)
        return cnt_white, cnt_black

    @staticmethod
    def move_is_available(row, col, drow, dcol, board: chex.Array):
        """
    Checks if a piece can be moved in the given direction. Checks for both, simple moves and jumps.
    Simple move is available if
            - destination is in bounds
            - destination is free
            - destination is forwards (unless king piece)
    Args:
        row: row index of the piece
        col: column index of the piece
        drow: movement in y direction
        dcol: movement in x direction
        board: current game board

    Returns:
        True, if a piece can be moved in the given direction, False otherwise.
    """
        landing_in_bounds = BoardHandler.move_in_bounds(row=row, col=col, drow=drow, dcol=dcol)

        def handle_jump():
            """
            Handle moves with |dx|=2 and |dy|=2
            jump is not possible if the jumped piece is of the same color as the jumping one
            BE SURE not just to check if the jumped pice is the same piece, but the same colour. (A white piece still cant jump a white king)
            jump movement for normal, non-king pieces is only possible forwards (row - 2 for black, row + 2 for white)
            Returns:
                True if that movement is available, False otherwise.
            """
            piece = board[row, col]
            jumped_piece = board[row + drow // 2, col + dcol // 2]
            piece_is_king = (piece == VideoCheckersConstants.WHITE_KING) | (piece == VideoCheckersConstants.BLACK_KING)
            drow_forward = jax.lax.cond(
                (piece == VideoCheckersConstants.WHITE_PIECE) | (piece == VideoCheckersConstants.WHITE_KING),
                lambda: 2, lambda: -2)
            is_forward = (drow == drow_forward)
            can_jump_in_direction = piece_is_king | is_forward
            tile_is_free = BoardHandler.tile_is_free(row + drow, col + dcol, board)
            jumped_piece_is_opponent = jax.lax.cond(
                jnp.logical_or(piece == VideoCheckersConstants.WHITE_PIECE, piece == VideoCheckersConstants.WHITE_KING),
                lambda: jnp.logical_or(jumped_piece == VideoCheckersConstants.BLACK_PIECE,
                                       jumped_piece == VideoCheckersConstants.BLACK_KING),
                lambda: jnp.logical_or(jumped_piece == VideoCheckersConstants.WHITE_PIECE,
                                       jumped_piece == VideoCheckersConstants.WHITE_KING),
            )
            return landing_in_bounds & tile_is_free & jumped_piece_is_opponent & can_jump_in_direction

        def handle_move():
            """
            Handle moves with |dx|=1 and |dy|=1
            Returns: True if that movement is available, False otherwise.
            """
            piece = board[row, col]
            piece_is_king = (piece == VideoCheckersConstants.WHITE_KING) | (piece == VideoCheckersConstants.BLACK_KING)

            drow_forward = jax.lax.cond(
                (piece == VideoCheckersConstants.WHITE_PIECE) | (piece == VideoCheckersConstants.WHITE_KING),
                lambda: 1, lambda: -1)
            is_forward = (drow == drow_forward)
            can_move_in_direction = piece_is_king | is_forward

            tile_is_free = BoardHandler.tile_is_free(row + drow, col + dcol, board)
            return landing_in_bounds & tile_is_free & can_move_in_direction

        is_jump = (jnp.abs(dcol) == 2) & (jnp.abs(drow) == 2)
        return jax.lax.cond(is_jump, handle_jump, handle_move)

    @staticmethod
    def get_movable_pieces(colour, board: chex.Array) -> (jnp.ndarray, bool):
        """
        For the given colour, return the position of pieces that can perform a legal move. This method therefore enforces
        the "must jump if possible" rule, returning only the positions of pieces with a jump available.
        Args:
            colour: Piece's colour
            board: Current game board

        Returns:
            Array of size (MAX_PIECES, 2), containing the positions of pieces that can perform a legal move. If no legal
            move is available for a piece, it is instead padded with [-1, -1]. Also returns a flag if any of the pieces
            can jump.
        """
        own_pieces = jax.lax.cond(colour == VideoCheckersConstants.COLOUR_WHITE,
                                  lambda: [VideoCheckersConstants.WHITE_PIECE, VideoCheckersConstants.WHITE_KING],
                                  lambda: [VideoCheckersConstants.BLACK_PIECE, VideoCheckersConstants.BLACK_KING], )
        own_pieces_mask = jnp.zeros_like(board, dtype=bool)
        for piece in own_pieces:
            own_pieces_mask |= (board == piece)

        # get positions of own pieces. static output shape, which is required for jit compilation.
        rows, cols = jnp.where(own_pieces_mask, size=VideoCheckersConstants.MAX_PIECES, fill_value=-1)
        positions = jnp.stack([rows, cols], axis=1)

        # vectorise function and apply to all positions
        vmapped_get_possible_moves = jax.vmap(BoardHandler.get_possible_moves_for_piece, in_axes=(0, 0, None))
        all_possible_moves = vmapped_get_possible_moves(rows, cols, board)

        # masks for each piece
        can_move_mask = jnp.any(all_possible_moves != 0, axis=(1, 2))  # any move available
        can_jump_mask = jnp.any(jnp.all(jnp.abs(all_possible_moves) == 2, axis=2), axis=1)  # jump available
        any_jump_available = jnp.any(can_jump_mask)

        movable_mask = jnp.where(any_jump_available,
                                 can_jump_mask,
                                 can_move_mask)  # this is just an if statement (cond, x ,y)

        movable_positions = jnp.where(
            movable_mask[:, None],  # Reshape mask to (MAX_PIECES, 1) for broadcasting
            positions,
            jnp.array([-1, -1])
        )

        return movable_positions, any_jump_available

    @staticmethod
    def is_movable_piece(colour, position, board: chex.Array):
        """
        check if position is in return set of get_movable_pieces. This is used to check if a piece can be selected in the select piece phase.
        Args:
            colour: Colour of the side to check for. 0 for white, 1 for black.
            position: Position of the piece to check
            board: Current board
        Returns:
            True, if the piece is movable, False otherwise.
        """
        movable_pieces, _ = BoardHandler.get_movable_pieces(colour, board)
        is_movable = jnp.any(jnp.all(movable_pieces == position, axis=1))
        return is_movable


class JaxVideoCheckers(
    JaxEnvironment[VideoCheckersState, VideoCheckersObservation, VideoCheckersInfo, VideoCheckersConstants]):
    def __init__(self, consts: VideoCheckersConstants = None, reward_funcs: list[callable] = None):
        consts = consts or VideoCheckersConstants()
        super().__init__(consts)
        self.renderer = VideoCheckersRenderer(self.consts)
        self.frame_stack_size = 4
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.action_set = {
            Action.FIRE,
            Action.UPRIGHT,
            Action.UPLEFT,
            Action.DOWNRIGHT,
            Action.DOWNLEFT
        }

    def render(self, state: VideoCheckersState) -> jnp.ndarray:
        return self.renderer.render(state)

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
        board = BoardHandler.reset_board()

        # Default state
        state = VideoCheckersState(cursor_pos=jnp.array([6, 7]), board=board, game_phase=self.consts.SELECT_PIECE_PHASE,
                                   selected_piece=jnp.array([-1, -1]), frame_counter=jnp.array(0), winner=-1,
                                   additional_jump=False,
                                   opponent_move=OpponentMove(start_pos=jnp.array([-1, -1]),
                                                              end_pos=jnp.array([-1, -1]),
                                                              piece_type=-1,
                                                              captured_positions=jnp.full((12, 2), -1),
                                                              # total of 12 pieces per side
                                                              resulting_board=board
                                                              ),
                                   rng_key=key,
                                   has_jumped=False)
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

        # if the phase is not SELECT_PIECE_PHASE, print a debug message
        jax.lax.cond(
            state.game_phase != self.consts.SELECT_PIECE_PHASE,
            lambda: jax.debug.print("Warning: Game phase is not SELECT_PIECE_PHASE, it is {game_phase}",
                                    game_phase=state.game_phase),
            lambda: None,

        )

        initial_obs = self._get_observation(state)

        def expand_and_copy(x):
            x_expanded = jnp.expand_dims(x, axis=0)
            return jnp.concatenate([x_expanded] * self.frame_stack_size, axis=0)

        # Apply transformation to each leaf in the pytree
        initial_obs = jax.tree.map(expand_and_copy, initial_obs)

        return initial_obs, state



    @partial(jax.jit, static_argnums=(0,))
    def step_select_piece_phase(self, state: VideoCheckersState, action: chex.Array) -> VideoCheckersState:
        """
        Handles moving the cursor and selecting a piece in the select piece phase.
        After a piece is selected, the game phase changes to MOVE_PIECE_PHASE.
        Args:
            state: The current game state.
            action: The action taken by the player.
        Returns:
            VideoCheckersState: The new game state after the action.
        """

        def select_piece(state: VideoCheckersState) -> VideoCheckersState:
            """
            Selects a piece at the current cursor position and changes the game phase to MOVE_PIECE_PHASE.
            """
            row, col = state.cursor_pos
            piece = state.board[row, col]
            return jax.lax.cond(
                (piece != self.consts.EMPTY_TILE)
                & BoardHandler.is_movable_piece(self.consts.COLOUR_BLACK, state.cursor_pos, state.board),
                lambda s: s._replace(
                    selected_piece=s.cursor_pos,
                    game_phase=self.consts.MOVE_PIECE_PHASE,
                ),
                lambda s: s,
                operand=state
            )

        def move_cursor(state: VideoCheckersState, action: chex.Array) -> VideoCheckersState:
            """
            Moves the cursor based on the action taken.
            """
            up = jnp.logical_or(action == Action.UPLEFT, action == Action.UPRIGHT)
            right = jnp.logical_or(action == Action.DOWNRIGHT, action == Action.UPRIGHT)
            donw_left = (
                    action == Action.DOWNLEFT)  # this is to prevent illegal pure ordinal inputs to move it down left.

            drow = jax.lax.cond(up, lambda: -1, lambda: 1, )  # -1 for up, 1 for down
            dcol = jax.lax.cond(right, lambda: 1, lambda: -1, )  # 1 for right, -1 for left

            new_cursor_pos = state.cursor_pos + jnp.array([drow, dcol])
            # Check if the new position is within bounds
            in_bounds = BoardHandler.move_in_bounds(row=state.cursor_pos[0],
                                                    col=state.cursor_pos[1],
                                                    drow=drow,
                                                    dcol=dcol)
            new_cursor_pos = jax.lax.cond(
                in_bounds & jnp.logical_or(up, jnp.logical_or(right, donw_left)),
                lambda: new_cursor_pos,
                lambda: state.cursor_pos,

            )
            return state._replace(cursor_pos=new_cursor_pos)

        new_state = jax.lax.cond(
            action == Action.FIRE,
            lambda s: select_piece(s),
            lambda s: move_cursor(s, action),
            operand=state)

        return new_state

    @staticmethod
    def calculate_best_move_per_piece(self, piece: chex.Array, key, board: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """
        For a given piece, returns the best move and its score.
        Args:
            self: the current game (JaxVideoCheckers instance)
            piece: array containing the pieces row and col
            key: an RNG key used for breaking ties
            board: current game board
        Returns:
            - best move (drow, dcol)
            - best move score
        """
        row, col = piece
        piece_type = board[row, col]

        moves = BoardHandler.get_possible_moves_for_piece(row, col, board)
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

        all_illegal = jnp.all(~~no_move)
        # When all illegal, argmax will be 0; preserve score -inf and return dummy move
        best_idx = jnp.argmax(scores)
        best_move = jnp.where(all_illegal, jnp.array([0, 0], moves.dtype), moves[best_idx])
        best_score = jnp.where(all_illegal, -jnp.inf, scores[best_idx])

        jax.debug.print("Piece {p} best move {m} score {s}", p=piece, m=best_move, s=best_score)

        return best_move, best_score

    @staticmethod
    def calculate_best_first_opponent_move(self, movable_pieces: chex.Array, state: VideoCheckersState) \
            -> VideoCheckersState:
        """
        Computes the first move (of potentially multiple) based on simple heuristics. Doesn't apply them (this gets done
        in step_show_opponent_move_phase)
        Args:
             self: the current game (JaxVideoCheckers instance)
             movable_pieces: array of size MAX_PIECES containing row and col position for movable pieces and -1,-1 for imovable
             state: current game state
         Returns:
             new state with updated opponent_move and has_jumped fields
        """

        # Supply an independent RNG key per piece
        rng_key, splitkey = jax.random.split(state.rng_key)
        best_moves, best_scores = (jax.vmap(self.calculate_best_move_per_piece, in_axes=(None, 0, None, None)
                                            )(self, movable_pieces, splitkey, state.board))

        # Choose the overall best move among all pieces
        top_piece_idx = jnp.argmax(best_scores)
        piece_to_move = movable_pieces[top_piece_idx]  # (row, col)
        move_to_play = best_moves[top_piece_idx]  # (drow, dcol)

        # Record the move and return the new state
        new_pos = piece_to_move + move_to_play
        new_board, new_piece, is_jump, c_row, c_col = BoardHandler.move_piece(
            row=piece_to_move[0], col=piece_to_move[1], drow=move_to_play[0], dcol=move_to_play[1], board=state.board)

        new_opponent_move = state.opponent_move._replace(start_pos=piece_to_move,
                                                         end_pos=new_pos,
                                                         piece_type=new_piece,
                                                         resulting_board=new_board)

        newest_opponent_move = jax.lax.cond(is_jump,
                                            lambda: OpponentMoveHandler.add_captured_position(new_opponent_move,
                                                                                              jnp.array((c_row, c_col),
                                                                                                        dtype=jnp.int32)),
                                            lambda: new_opponent_move)

        return state._replace(has_jumped=is_jump, opponent_move=newest_opponent_move)

    def calculate_best_further_opponent_move(self, state: VideoCheckersState) \
            -> VideoCheckersState:
        """
        Handles the case where the opponent has already jumped and can jump again. He can only jump again with the same piece.
        So only check if the piece on end_pos can jump again.
        Args:
            state: current game state
        Returns:
            new state with updated opponent_move and has_jumped fields
        """
        jax.debug.print("Computing further opponent move...")

        def loop_body(i, state: VideoCheckersState) -> VideoCheckersState:
            """
            Loop body for the fori_loop, which calculates the best further opponent move.
            This is called when the opponent has already jumped and can jump again.
            """

            def calc_further_move(state: VideoCheckersState) -> VideoCheckersState:
                """
                Get all moves for this piece and check if there is a jump available.
                If so use that jump, otherwise return the state unchanged except has_jumped is set to False.
                Use the function calculate_best_move_per_piece to get the best move for the piece.
                """
                piece = state.opponent_move.end_pos
                rng_key, splitkey = jax.random.split(state.rng_key)
                board = state.opponent_move.resulting_board

                best_move, best_score = self.calculate_best_move_per_piece(self, piece, splitkey, board)
                jax.debug.print("Best move for piece {piece}: {best_move}, score: {best_score}",
                                piece=piece, best_move=best_move, best_score=best_score)
                new_board, new_piece, is_jump, c_row, c_col = BoardHandler.move_piece(
                    row=piece[0], col=piece[1], drow=best_move[0], dcol=best_move[1], board=board)
                new_pos = piece + best_move

                new_opponent_move = state.opponent_move._replace(end_pos=new_pos, piece_type=new_piece,
                                                                 resulting_board=new_board)
                new_opponent_move = OpponentMoveHandler.add_captured_position(new_opponent_move,
                                                                              jnp.array((c_row, c_col),
                                                                                        dtype=jnp.int32))
                return jax.lax.cond(
                    is_jump,
                    lambda: state._replace(
                        has_jumped=True,
                        opponent_move=new_opponent_move,
                        rng_key=rng_key
                    ),
                    lambda: state._replace(
                        has_jumped=False,
                        rng_key=rng_key
                    )
                )

            return jax.lax.cond(
                state.has_jumped,
                lambda s: calc_further_move(s),
                lambda s: s,
                state
            )

        return jax.lax.fori_loop(
            lower=0,
            upper=12,
            body_fun=loop_body,
            init_val=state
        )

    def step_move_piece_phase(self, state: VideoCheckersState, action: chex.Array) -> VideoCheckersState:
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
                new_state = self.calculate_best_first_opponent_move(self, moveable_pieces, state)
                # if the opponent has jumped, we need to check calculate_best_further_opponent_move
                opponent_move = jax.lax.cond(
                    new_state.has_jumped,
                    lambda: self.calculate_best_further_opponent_move(new_state).opponent_move,
                    lambda: new_state.opponent_move,
                )

                return state._replace(
                    game_phase=self.consts.SHOW_OPPONENT_MOVE_PHASE,
                    has_jumped=False,
                    opponent_move=opponent_move,
                    selected_piece=jnp.array([-1, -1]),  # Reset the selected piece
                    rng_key=new_state.rng_key,
                )

            movable_pieces, _ = BoardHandler.get_movable_pieces(self.consts.COLOUR_WHITE, state.board)

            return jax.lax.cond(
                jnp.all(movable_pieces == jnp.array([-1, -1])),
                lambda: state._replace(game_phase=self.consts.GAME_OVER_PHASE, winner=self.consts.COLOUR_BLACK),
                lambda: moves_available(movable_pieces, state)
            )

        def move_cursor(state: VideoCheckersState, action: chex.Array) -> VideoCheckersState:
            """
            Moves the cursor based on the action taken.
            a normal piece can either move forward (upleft/upright) one tile or jump over an opponent's piece up two tiles.
            a king can move in all four directions one tile or jump over an opponent's piece up two tiles.
            (this is returend by get_possible_moves_for_piece)
            when the cursor is not on the selected piece the only valid move is back to the selected piece.
            The check if the only move is to return to the selected piece is NOT DONE in get_possible_moves_for_piece, we have to do it here.
            """

            def _move_cursor_back(state: VideoCheckersState, action: chex.Array) -> VideoCheckersState:
                """
                Moves the cursor back to the selected piece.
                For this we need to check what direction it is from the cursor to the selected piece.
                Then check if the user input is in the same direction.
                """
                row_diff = state.selected_piece[0] - state.cursor_pos[0]
                col_diff = state.selected_piece[1] - state.cursor_pos[1]

                # Prüfe, ob die Aktion in Richtung des ausgewählten Steins geht
                up = jnp.logical_or(action == Action.UPLEFT, action == Action.UPRIGHT)
                right = jnp.logical_or(action == Action.DOWNRIGHT, action == Action.UPRIGHT)

                is_jump = (jnp.abs(row_diff) == 2) & (jnp.abs(col_diff) == 2)
                drow = jax.lax.cond(is_jump,
                                    lambda: jax.lax.cond(up, lambda: -2, lambda: 2),
                                    lambda: jax.lax.cond(up, lambda: -1, lambda: 1),
                                    )
                dcol = jax.lax.cond(is_jump,
                                    lambda: jax.lax.cond(right, lambda: 2, lambda: -2, ),
                                    lambda: jax.lax.cond(right, lambda: 1, lambda: -1, ),
                                    )

                # Prüfe, ob die Richtung ein Sprung ist (2 Felder Unterschied)
                is_correct_direction = (row_diff == drow) & (col_diff == dcol)

                # If the action is in the correct direction, move the cursor back to the selected piece
                return jax.lax.cond(
                    is_correct_direction,
                    lambda s: s._replace(cursor_pos=s.selected_piece),
                    lambda s: s,  # If not in the correct direction, do nothing
                    operand=state
                )

            def _move_cursor_away(state: VideoCheckersState, action: chex.Array) -> VideoCheckersState:
                """
                Moves the cursor away from the selected piece. THIS CAN BE JUMPS.
                For the we need to use get_possible_moves_for_piece!
                Action can be one of the four directions (UPRIGHT, UPLEFT, DOWNRIGHT, DOWNLEFT). which are not Movement vectors but just enumerated values.
                """

                # Ermittle die möglichen Züge für die ausgewählte Figur
                possible_moves = BoardHandler.get_possible_moves_for_piece(state.selected_piece[0],
                                                                           state.selected_piece[1],
                                                                           state.board)

                move_vector, jump_vector = jax.lax.cond(
                    action == Action.UPRIGHT,
                    lambda: (jnp.array([-1, 1]), jnp.array([-2, 2])),
                    lambda: jax.lax.cond(
                        action == Action.UPLEFT,
                        lambda: (jnp.array([-1, -1]), jnp.array([-2, -2])),
                        lambda: jax.lax.cond(
                            action == Action.DOWNRIGHT,
                            lambda: (jnp.array([1, 1]), jnp.array([2, 2])),
                            lambda: jax.lax.cond(
                                action == Action.DOWNLEFT,
                                lambda: (jnp.array([1, -1]), jnp.array([2, -2])),
                                lambda: (jnp.array([0, 0]), jnp.array([0, 0])),  # Default case
                            ),
                        ),
                    ),
                )

                # Prüfe, ob der Zug gültig ist
                is_valid_move = jnp.logical_or(
                    jnp.any(jnp.all(possible_moves == move_vector, axis=1)),
                    jnp.any(jnp.all(possible_moves == jump_vector, axis=1))
                )

                jump_available = jnp.any(jnp.all(possible_moves == jump_vector, axis=1))

                # If the move is valid, update the cursor position
                return jax.lax.cond(
                    is_valid_move,
                    lambda s: jax.lax.cond(
                        jump_available,
                        lambda s: s._replace(cursor_pos=s.cursor_pos + jump_vector),
                        lambda s: s._replace(cursor_pos=s.cursor_pos + move_vector),
                        operand=s),
                    lambda s: s,  # If not a valid move, do nothing
                    operand=state
                )

            def _move_cursor(state: VideoCheckersState, action: chex.Array) -> VideoCheckersState:
                return jax.lax.cond(
                    jnp.all(state.selected_piece == state.cursor_pos),
                    lambda s: _move_cursor_away(s, action),
                    lambda s: _move_cursor_back(s, action),
                    operand=state
                )

            return jax.lax.cond(
                action == Action.NOOP,
                lambda s: s,  # No action taken, return the same state
                lambda s: _move_cursor(s, action),
                operand=state
            )

        def place_piece(state: VideoCheckersState) -> VideoCheckersState:
            """
            Places the selected piece at the destination and updates the game phase.
            Updating the game phase can be:
            A. If no further jumps are available, change to SHOW_OPPONENT_MOVE_PHASE.
            B. If further jumps are available, stay in MOVE_PIECE_PHASE but the destination is reset to [-1, -1].
            Or if the piece has not been moved (put back down), return to the select piece phase.
            """

            def _deselect_piece(state: VideoCheckersState) -> VideoCheckersState:
                """
                Only allows deselecting the piece if it has not jumped.
                """
                return jax.lax.cond(
                    state.has_jumped,
                    lambda s: state,
                    lambda s: s._replace(selected_piece=jnp.array([-1, -1]), game_phase=self.consts.SELECT_PIECE_PHASE),
                    operand=state
                )

            def _place_piece(state: VideoCheckersState) -> VideoCheckersState:
                piece_type = state.board[state.selected_piece[0], state.selected_piece[1]]
                move = state.cursor_pos - state.selected_piece
                jumped = (jnp.abs(move[0]) > 1) & (jnp.abs(move[1]) > 1)

                new_piece = jax.lax.cond((state.cursor_pos[0] == 0),
                                         lambda: self.consts.BLACK_KING,
                                         lambda: piece_type)

                # move piece
                new_board = state.board \
                    .at[tuple(state.selected_piece)].set(self.consts.EMPTY_TILE) \
                    .at[tuple(state.cursor_pos)].set(new_piece)

                # capture piece (determine the piece between state.selected_piece and state.cursor_pos
                captured_piece = state.selected_piece + (move // 2)
                new_board = jax.lax.cond(
                    jumped,
                    lambda: new_board.at[tuple(captured_piece)].set(self.consts.EMPTY_TILE),
                    lambda: new_board
                )

                # get new state
                new_state = state._replace(board=new_board)
                captured_piece = state.selected_piece + (
                        move // 2)  # mid position for jumped piece if jumps are available from new pos
                new_moves = BoardHandler.get_possible_moves_for_piece(row=new_state.cursor_pos[0],
                                                                      col=new_state.cursor_pos[1],
                                                                      board=new_state.board)
                move_distances = jnp.abs(new_moves)  # Shape: (n_moves, 2)
                max_distances = jnp.max(move_distances, axis=1)  # Max distance per move
                has_jump_from_new_pos = jnp.any(max_distances > 1)

                # stay in same phase and reselect piece if jumped and can continue jumping, change phase if not
                return jax.lax.cond(
                    jumped & has_jump_from_new_pos,
                    lambda s: s._replace(selected_piece=s.cursor_pos, has_jumped=True),
                    lambda s: prepare_opponent_move(s),
                    new_state
                )

            cursor_on_selected_piece = jnp.all(state.selected_piece == state.cursor_pos)

            return jax.lax.cond(cursor_on_selected_piece,
                                lambda s: _deselect_piece(s),
                                # deselect piece
                                lambda s: _place_piece(s),  # move piece + side effects
                                state
                                )

        new_state = jax.lax.cond(
            action == Action.FIRE,
            lambda s: place_piece(s),
            lambda s: move_cursor(s, action),
            operand=state
        )

        return new_state

    def step_show_opponent_move_phase(self, state: VideoCheckersState, action: chex.Array) -> VideoCheckersState:
        """
        Handles showing the opponent's move in the show opponent move phase.
        This is interrupted by the player making any input, which then returns to the select piece phase.
        Args:
            state: The current game state.
            action: The action taken by the player.
        Returns:
            VideoCheckersState: The new game state after the action.
        """

        def apply_opponent_move(state: VideoCheckersState) -> VideoCheckersState:
            """
            Applies the opponent's move to the game state.
            """
            # Get the opponent's move
            opponent_move = state.opponent_move

            # Update the game state with the new board and reset cursor position and opponent move
            return state._replace(
                board=opponent_move.resulting_board,
                opponent_move=OpponentMoveHandler.clear_captured_positions(opponent_move),
                game_phase=self.consts.SELECT_PIECE_PHASE,  # Change phase back to select piece phase
                selected_piece=jnp.array([-1, -1]),  # Reset selected piece
            )

        new_state = jax.lax.cond(
            action == Action.NOOP,
            lambda s: s,  # No action taken, return the same state
            apply_opponent_move,
            operand=state
        )

        # check if player lost
        new_state = jax.lax.cond(
            BoardHandler.count_pieces(new_state.board)[1] == 0,
            lambda s: s._replace(game_phase=self.consts.GAME_OVER_PHASE, winner=self.consts.COLOUR_WHITE),
            lambda s: s,
            new_state
        )

        return new_state

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
        # Switch between game phases to choose which function handles the step
        # So separate function for each game phase
        new_state = jax.lax.cond(
            (state.frame_counter == (self.consts.ANIMATION_FRAME_RATE - 1)) & (action != Action.NOOP),
            lambda: jax.lax.cond(
                state.game_phase == self.consts.SELECT_PIECE_PHASE,
                lambda: self.step_select_piece_phase(state, action),
                lambda: jax.lax.cond(
                    state.game_phase == self.consts.MOVE_PIECE_PHASE,
                    lambda: self.step_move_piece_phase(state, action),
                    lambda: jax.lax.cond(
                        state.game_phase == self.consts.SHOW_OPPONENT_MOVE_PHASE,
                        lambda: self.step_show_opponent_move_phase(state, action),
                        lambda: self.step_game_over_phase(state, action),
                    ),
                ),
            ),
            lambda: state,
        )

        # if the new_state phase is not equal to the old state phase,
        # print what the old phase was and what the new phase is

        jax.lax.cond(
            new_state.game_phase != state.game_phase,
            lambda: jax.debug.print("Game phase changed from {old_phase} to {new_phase}",
                                    old_phase=state.game_phase, new_phase=new_state.game_phase),
            lambda: None)

        new_state = new_state._replace(frame_counter=(new_state.frame_counter + 1) % self.consts.ANIMATION_FRAME_RATE)

        done = self._get_done(new_state)
        env_reward = self._get_env_reward(state, new_state)
        all_rewards = self._get_all_reward(state, new_state)
        info = self._get_info(new_state, all_rewards)

        observation = self._get_observation(new_state)

        return observation, new_state, env_reward, done, info

    def action_space(self):
        """
        Returns the action space of the game environment.
        Returns:
            action_space: The action space of the game environment.
        """
        return spaces.Discrete(5)

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.consts.HEIGHT, self.consts.WIDTH, 3),
            dtype=jnp.uint8,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: VideoCheckersState, all_rewards: chex.Array) -> VideoCheckersInfo:
        """
        Returns additional information about the game state.
        Args:
            state: The current game state.
            all_rewards: The rewards received after taking the action.
        Returns:
            VideoCheckersInfo: Additional information about the game state.
        """
        return VideoCheckersInfo(all_rewards=all_rewards)

    def observation_space(self) -> spaces:
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
        """
        Returns the observation of the game state.
        Args:
            state: The current game state.
        Returns:
            VideoCheckersObservation: The observation of the game state.
        """
        _, must_jump = BoardHandler.get_movable_pieces(self.consts.COLOUR_BLACK, state.board)
        return VideoCheckersObservation(board=state.board,
                                        start_pos=state.cursor_pos,
                                        end_pos=state.selected_piece,
                                        must_jump=must_jump.astype(jnp.int32),
                                        cursor_pos=state.cursor_pos)

    @partial(jax.jit, static_argnums=(0,))
    def _get_env_reward(self, previous_state: VideoCheckersState, state: VideoCheckersState) -> float:
        """
        Returns the environment reward based on the game state.
        Args:
            previous_state: The previous game state.
        """
        previous_counts = BoardHandler.count_pieces(previous_state.board)
        previous_lead_black = previous_counts[1] - previous_counts[0]
        new_counts = BoardHandler.count_pieces(state.board)
        new_lead_black = new_counts[1] - new_counts[0]
        return new_lead_black - previous_lead_black

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(self, previous_state: VideoCheckersState, state: VideoCheckersState):
        """
        Returns all rewards based on the game state.
        Args:
            previous_state: The previous game state.
            state: The current game state.
        Returns:
            rewards: The rewards received after taking the action.
        """
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array(
            [reward_func(previous_state, state) for reward_func in self.reward_funcs]
        )
        return rewards

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: VideoCheckersState) -> bool:
        """
        Returns whether the game is done based on the game state.
        Args:
            state: The current game state.
        """
        return state.game_phase == VideoCheckersConstants.GAME_OVER_PHASE


def load_sprites():
    """
    Load all sprites required for Flag Capture rendering.
    Returns:
        Sprite arrays (BG, pieces, text)
    """
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

    background = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/videocheckers/background.npy"), transpose=True)

    # Convert all sprites to the expected format (add frame dimension)
    SPRITE_BG = jnp.expand_dims(background, axis=0)
    SPRITE_PIECES = jr.load_and_pad_digits(os.path.join(MODULE_DIR, "sprites/videocheckers/pieces/{}.npy"),
                                           num_chars=7)
    SPRITE_TEXT = jr.load_and_pad_digits(os.path.join(MODULE_DIR, "sprites/videocheckers/text/{}.npy"),
                                         num_chars=12)

    return (
        SPRITE_BG,
        SPRITE_PIECES,
        SPRITE_TEXT,
    )


class VideoCheckersRenderer(JAXGameRenderer):
    def __init__(self, consts: VideoCheckersConstants = None):
        super().__init__()
        self.consts = consts or VideoCheckersConstants()
        (
            self.SPRITE_BG,
            self.SPRITE_PIECES,
            self.SPRITE_TEXT,
        ) = load_sprites()

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        """
        Renders the current game state using JAX operations.

        Args:
            state: A VideoCheckersState object containing the current game state.

        Returns:
            A JAX array representing the rendered frame.
        """

        raster = jr.create_initial_frame(width=self.consts.WIDTH, height=self.consts.HEIGHT)

        def determine_piece_type_select_phase(row, col, state: VideoCheckersState):
            """
            Determines the piece type to render in the select piece phase.
            This means rendering a cursor if frame_counter is under half
            Args:
                row: Row index of the piece.
                col: Column index of the piece.
                state: Current game state.
            Returns:
                The piece type to render.
            """
            return jax.lax.cond(
                (state.frame_counter < (self.consts.ANIMATION_FRAME_RATE / 2)) & (state.cursor_pos[0] == row) & (
                        state.cursor_pos[1] == col),
                lambda: self.consts.BLACK_CURSOR,
                lambda: state.board[row, col],
            )

        def determine_piece_type_move_phase(row, col, state: VideoCheckersState):
            """
            Determines the piece type to render in the move piece phase.
            We have a selected piece and destination.
            if the destination is on the selected piece the piece should be fast blinking (render the piece sprite if frame_counter is not 5 to 10 or 15 to 20)
            if the destination is not on the selected piece we have two animation states. one for < 30 and one for >= 30.
            all pieces on the board should be rendered as normal. exceptions are only the selected piece and the destination.
            for the < 30 state, the selected piece should be rendered as BLACK_CURSOR and the destination as the selected pieces tile sprite.
            For the >= 30 state, the selected piece should be rendered as its original sprite and the destination as nothing (empty tile).
            if the piece to be determined is not the selected piece or destination, it should be rendered as normal.
            Args:
                row: Row index of the piece.
                col: Column index of the piece.
                state: Current game state.
            Returns:
                The piece type to render.
            """
            is_selected_piece = jnp.all(state.selected_piece == jnp.array([row, col]))
            is_cursor_pos = jnp.all(state.cursor_pos == jnp.array([row, col]))
            is_unmoved = jnp.all(is_selected_piece & is_cursor_pos)

            def f_umoved():
                # If the piece is unmoved, blink fast
                return jax.lax.cond(
                    (state.frame_counter % self.consts.ANIMATION_FRAME_RATE < 5) | (
                            state.frame_counter % self.consts.ANIMATION_FRAME_RATE >= 10) & (
                            state.frame_counter % self.consts.ANIMATION_FRAME_RATE < 15),
                    lambda: self.consts.EMPTY_TILE,
                    lambda: state.board[row, col],
                )

            def f_selected():
                # If the piece is selected, render it as BLACK_CURSOR if frame_counter < 30, else render it as its original sprite
                return jax.lax.cond(
                    state.frame_counter < (self.consts.ANIMATION_FRAME_RATE / 2),
                    lambda: self.consts.BLACK_CURSOR,
                    lambda: state.board[row, col],
                )

            def f_cursor():
                # If the piece is the destination, render it as the selected piece's tile sprite if frame_counter < 30, else render it as EMPTY_TILE
                return jax.lax.cond(
                    state.frame_counter < (self.consts.ANIMATION_FRAME_RATE / 2),
                    lambda: state.board[state.selected_piece[0], state.selected_piece[1]],
                    # Use selected piece's tile sprite
                    lambda: self.consts.EMPTY_TILE,
                )

            return jax.lax.cond(
                is_unmoved,
                f_umoved,
                lambda: jax.lax.cond(
                    is_selected_piece,
                    f_selected,
                    lambda: jax.lax.cond(
                        is_cursor_pos,
                        f_cursor,
                        lambda: state.board[row, col],
                    ),

                ),
            )

        def determine_piece_type_show_opponent_move_phase(row, col, state: VideoCheckersState):
            """
            Determines the piece type to render in the show opponent move phase.
            We have two animation states. one for < 30, let call it "before move" and one for >= 30, let call it "after move".
            In the "before move" state, the opponent_move.start_pos should be rendered as WHITE_CURSOR and the opponent_move.end_pos as an empty tile.
            The captured positions should be rendered as their original piece type.
            In the "after move" state, the opponent_move.start_pos should be rendered as an empty tile, the opponent_move.end_pos as the opponent_move.piece_type and the captured positions as BLACK_CURSOR.
            Args:
                row: Row index of the piece.
                col: Column index of the piece.
                state: Current game state.
            Returns:
                The piece type to render.
            """

            is_start_pos = jnp.all(state.opponent_move.start_pos == jnp.array([row, col]))
            is_end_pos = jnp.all(state.opponent_move.end_pos == jnp.array([row, col]))
            is_captured_pos = jnp.any(jnp.all(state.opponent_move.captured_positions == jnp.array([row, col]), axis=1))

            def f_before_move(s):
                return jax.lax.cond(
                    is_start_pos,
                    lambda s: self.consts.WHITE_CURSOR,
                    lambda s: jax.lax.cond(
                        is_end_pos,
                        lambda s: self.consts.EMPTY_TILE,
                        lambda s: s.board[row, col],
                        operand=s
                    ),
                    operand=s
                )

            def f_after_move(s):
                return jax.lax.cond(
                    is_start_pos,
                    lambda s: self.consts.EMPTY_TILE,
                    lambda s: jax.lax.cond(
                        is_end_pos,
                        lambda s: state.opponent_move.piece_type,  # Render end position as the opponent's piece type
                        lambda s: jax.lax.cond(
                            is_captured_pos,
                            lambda s: self.consts.BLACK_CURSOR,  # Render captured positions as BLACK_CURSOR
                            lambda s: s.board[row, col],
                            operand=s
                        ),
                        operand=s
                    ),
                    operand=s
                )

            return jax.lax.cond(
                state.frame_counter < (self.consts.ANIMATION_FRAME_RATE / 2),
                f_before_move,
                f_after_move,
                operand=state
            )

        def determine_piece_type_game_over_phase(row, col, state: VideoCheckersState):
            return state.board[row, col]

        def render_pieces_on_board(raster, state: VideoCheckersState):
            def render_piece(row, col, raster):
                # call 4 different function to determine which piece to render depending on the phase of the game. No logic just call the 4 functions
                piece_type = jax.lax.cond(
                    state.game_phase == self.consts.SELECT_PIECE_PHASE,
                    lambda: determine_piece_type_select_phase(row, col, state),
                    lambda: jax.lax.cond(
                        state.game_phase == self.consts.MOVE_PIECE_PHASE,
                        lambda: determine_piece_type_move_phase(row, col, state),
                        lambda: jax.lax.cond(
                            state.game_phase == self.consts.SHOW_OPPONENT_MOVE_PHASE,
                            lambda: determine_piece_type_show_opponent_move_phase(row, col, state),
                            lambda: determine_piece_type_game_over_phase(row, col, state),
                        ),
                    ),
                )

                piece_frame = jr.get_sprite_frame(self.SPRITE_PIECES, piece_type)
                return jax.lax.cond(
                    (piece_frame is not None) & ((row + col) % 2 == 1),  # Only render on dark squares
                    lambda: jr.render_at(
                        raster,
                        self.consts.OFFSET_X_BOARD + 4 + col * 17,  # Calculate the position on the board
                        self.consts.OFFSET_Y_BOARD + 2 + row * 13,
                        piece_frame,
                    ),
                    lambda: raster,
                )

            def render_row(row, raster):
                return jax.lax.fori_loop(
                    0, self.consts.NUM_FIELDS_X, lambda col, raster: render_piece(row, col, raster), raster
                )

            return jax.lax.fori_loop(0, self.consts.NUM_FIELDS_Y, render_row, raster)

        def render_jump_indicator(raster, state: VideoCheckersState):
            _, must_jump = BoardHandler.get_movable_pieces(VideoCheckersConstants.BLACK_PIECE, state.board)

            def _render(raster):
                x_offset = 100
                j_sprite = jr.get_sprite_frame(self.SPRITE_TEXT, 10)
                p_sprite = jr.get_sprite_frame(self.SPRITE_TEXT, 11)
                pos_xj = x_offset
                pos_xp = x_offset + 10
                pos_y = 20
                raster = jr.render_at(raster, pos_xj, pos_y, j_sprite)
                raster = jr.render_at(raster, pos_xp, pos_y, p_sprite)

                return raster

            new_raster = jax.lax.cond(
                must_jump,
                _render,
                lambda r: r,
                raster
            )
            return new_raster

        frame_bg = jr.get_sprite_frame(self.SPRITE_BG, 0)
        # fill background with [160, 96, 64]
        raster = raster.at[:, :].set(jnp.array([160, 96, 64], dtype=jnp.uint8))
        raster = jr.render_at(raster, self.consts.OFFSET_X_BOARD, self.consts.OFFSET_Y_BOARD, frame_bg)
        raster = render_pieces_on_board(raster, state)
        raster = render_jump_indicator(raster, state)

        return raster
