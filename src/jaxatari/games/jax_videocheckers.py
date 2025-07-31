import os
from functools import partial
from typing import NamedTuple, Tuple
import jax.lax
import jax.numpy as jnp
import chex


import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as jr
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action

#
# by Tim Morgner and Jan Larionow
#

# IMPORTANT: WE, THE PLAYER, PLAY AS BLACK, THE OPPONENT IS WHITE.

# region Constants
# Constants for game environment
WIDTH = 160
HEIGHT = 210
# endregion

# region Pygame Constants
# Pygame window dimensions
SCALING_FACTOR = 3
WINDOW_WIDTH = WIDTH * SCALING_FACTOR
WINDOW_HEIGHT = HEIGHT * SCALING_FACTOR
# endregion

# region Offsets
OFFSET_X_BOARD = 12
OFFSET_Y_BOARD = 50
# endregion

# region Moves
# in (y, x) notation
MOVES = jnp.array([
    [1, 1],  # UPRIGHT
    [-1, 1],  # DOWNRIGHT
    [-1, -1],  # DOWNLEFT
    [1, -1],  # UPLEFT
])
# endregion

# region Checker pieces
EMPTY_TILE = 0
WHITE_PIECE = 1
BLACK_PIECE = 2
WHITE_KING = 3
BLACK_KING = 4
WHITE_CURSOR = 5
BLACK_CURSOR = 6
# endregion

# region Board dimensions
NUM_FIELDS_X = 8
NUM_FIELDS_Y = 8
# endregion

# Transitions between game phases
# SELECT_PIECE -> MOVE_PIECE: # Player selects a piece to move
# MOVE_PIECE -> SHOW_OPPONENT_MOVE: # Player moves the piece with no further jumps available
# MOVE_PIECE -> MOVE_PIECE: # Player moves the piece with further jumps available
# SHOW_OPPONENT_MOVE -> SELECT_PIECE: # Player makes an input to select a piece after the opponent's move
# TODO: GAME OVER STATES
# region Game Phases
SELECT_PIECE_PHASE = 0
MOVE_PIECE_PHASE = 1
SHOW_OPPONENT_MOVE_PHASE = 2
GAME_OVER_PHASE = 3
# endregion

# region rendering constants
# Constants for rendering
ANIMATION_FRAME_RATE = 6 if os.environ.get("JAX_DISABLE_JIT") == "1" else 60  # Workaround for JAX disabling JIT compilation
# endregion

# region Colours
COLOUR_WHITE = 0
COLOUR_BLACK = 1
# endregion

MAX_PIECES = 12

class VideoCheckersConstants(NamedTuple):
    TODO=1
    #TODO

class OpponentMove(NamedTuple):
    start_pos: chex.Array  # Start position of the opponent's piece
    end_pos: chex.Array  # End position of the opponent's piece
    piece_type: int  # Type of the piece at the end position (king or normal)
    captured_positions: chex.Array  # Array of positions of captured pieces


class VideoCheckersState(NamedTuple):
    board: chex.Array  # Shape (NUM_FIELDS_Y, NUM_FIELDS_X)
    game_phase: int
    cursor_pos: chex.Array
    additional_jump: bool  # True if in the MOVE_PIECE_PHASE a there has already been a jump, so the player can jump again. This prevents the player from deselecting the piece.
    selected_piece: chex.Array
    frame_counter: chex.Array
    opponent_move: OpponentMove
    winner: int  # -1 if no winner, COLOUR_WHITE if white won, COLOUR_BLACK if black won.


class VideoCheckersObservation(NamedTuple):
    board: chex.Array  # All animation is already baked into the board observation
    start_pos: chex.Array  # This is for the move number display
    end_pos: chex.Array  # This is for the piece number display
    must_jump: chex.Array  # This is for the must jump display, if a jump is available
    # TODO: rework observation to include more information, e.g. cursor position, selected piece, etc.


class VideoCheckersInfo(NamedTuple):
    all_rewards: chex.Array


@partial(jax.jit, static_argnums=(0,))
def get_possible_moves_for_piece(position, state: VideoCheckersState):
    """
    Get all possible moves for a piece at position (y,x)
    Args:
        position: array containing x,y coordinates in that order
        state: current game state

    Returns: array of all possible moves. If a move in a given direction is not possible, it returns [0,0]
    """

    row, col = position[0], position[1]
    current_piece = state.board[row, col]
    is_not_a_piece = (current_piece == EMPTY_TILE) | (row == -1)

    def _get_moves():
        def check_move(move):
            drow, dcol = move

            piece = state.board[row, col]
            piece_is_king = (current_piece == WHITE_KING) | (current_piece == BLACK_KING)

            dy_forward = jax.lax.cond((piece == WHITE_PIECE) | (piece == WHITE_KING), lambda: 1, lambda: -1)
            is_forward = (drow == dy_forward)
            can_move_in_direction = piece_is_king | is_forward

            def get_valid_move_for_direction():
                jump_available = move_is_available(row=row, col=col, drow=2 * drow, dcol=2 * dcol,
                                                   state=state)  # check jump
                move_available = move_is_available(row=row, col=col, drow=drow, dcol=dcol,
                                                   state=state)  # check normal move

                # Return jump move if available, else normal move if available, else [0,0]
                return jax.lax.cond(
                    jump_available,
                    lambda s: move * 2,
                    lambda s: jax.lax.cond(
                        move_available,
                        lambda s: move,
                        lambda s: jnp.array([0, 0]),
                        operand=None),
                    operand=None
                )

            return jax.lax.cond(can_move_in_direction,
                                get_valid_move_for_direction,
                                lambda: jnp.array([0, 0]))

        possible_moves = jax.vmap(check_move)(MOVES)
        return possible_moves

    return jax.lax.cond(is_not_a_piece, lambda: jnp.zeros((4, 2), dtype=jnp.int32), _get_moves)


@partial(jax.jit, static_argnums=(0,))
def move_in_bounds(row, col, drow, dcol, state: VideoCheckersState):
    """
    Checks if move can be made in the given direction.
    Args:
        row: row index of the piece
        col: column index of the piece
        drow: movement in y direction
        dcol: movement in x direction
        state: state of the game, containing current cursor position.

    Returns: True, if cursor can be moved in the given direction, False otherwise.

    """
    return (jnp.logical_and(
        (0 <= row + drow),
        (row + drow < NUM_FIELDS_Y)
    ) & jnp.logical_and(
        (0 <= col + dcol),
        (col + dcol < NUM_FIELDS_X)
    ))


@partial(jax.jit, static_argnums=(0,))
def move_is_available(row, col, drow, dcol, state: VideoCheckersState):
    """
    Checks if a piece can be moved in the given direction. Checks for both, simple moves and jumps.
    Args:
        row: row index of the piece
        col: column index of the piece
        drow: movement in y direction
        dcol: movement in x direction
        state: state of the game, containing position of the current piece and the board-state.

    Returns:
        True, if a piece can be moved in the given direction, False otherwise.
    """
    landing_in_bounds = move_in_bounds(row=row, col=col, drow=drow, dcol=dcol, state=state)
    board = state.board

    def handle_jump():
        """
        Handle moves with |dx|=2 and |dy|=2
        Returns: True if that movement is available, False otherwise.
        """
        own_colour = state.board[row, col]
        jumped_col = col + (dcol // 2)
        jumped_row = row + (drow // 2)
        return jax.lax.cond(
            landing_in_bounds,
            lambda s: (board[jumped_row, jumped_col] != EMPTY_TILE) &  # jumped-tile is not empty
                      (board[jumped_row, jumped_col] != own_colour) &  # jumped-tile is not of same colour
                      (board[row + 2 * drow, col + 2 * dcol] == EMPTY_TILE),  # landing tile is empty
            lambda s: False,
            operand=None
        )

    def handle_move():
        """
        Handle moves with |dx|=1 and |dy|=1
        Returns: True if that movement is available, False otherwise.
        """
        return landing_in_bounds & (board[row + drow, col + dcol] == EMPTY_TILE)

    is_jump = (jnp.abs(dcol) == 2) & (jnp.abs(drow) == 2)
    return jax.lax.cond(is_jump, handle_jump, handle_move)


def is_movable_piece(colour, position, state: VideoCheckersState):
    """
    check if position is in return set of get_movable_pieces. This is used to check if a piece can be selected in the select piece phase.
    Args:
        colour: Colour of the side to check for. 0 for white, 1 for black.
        position: Position of the piece to check
        state: Current state of the game
    Returns: True, if the piece is movable, False otherwise.
    """
    movable_pieces = get_movable_pieces(colour, state)
    is_movable = jnp.any(jnp.all(movable_pieces == position, axis=1))
    jax.debug.print("Is position {position} movable: {is_movable}", position=position, is_movable=is_movable)
    return is_movable


def get_movable_pieces(colour, state: VideoCheckersState) -> jnp.ndarray:
    """
    For the given colour, return the position of pieces that can perform a legal move. This method therefore enforces
    the "must jump if possible" rule, returning only the positions of pieces with a jump available.
    Args:
        colour: Piece's colour
        state: Current state of the game

    Returns: Array of size (MAX_PIECES, 2), containing the positions of pieces that can perform a legal move. If no legal
    move is available for a piece, it is instead padded with [-1, -1].
    """
    own_pieces = jax.lax.cond(colour == COLOUR_WHITE, lambda s: [WHITE_PIECE, WHITE_KING],
                              lambda s: [BLACK_PIECE, BLACK_KING], operand=None)
    own_pieces_mask = jnp.zeros_like(state.board, dtype=bool)
    for piece in own_pieces:
        own_pieces_mask |= (state.board == piece)

    # get positions of own pieces. static output shape, which is required for jit compilation.
    rows, cols = jnp.where(own_pieces_mask, size=MAX_PIECES, fill_value=-1)
    positions = jnp.stack([rows, cols], axis=1)

    # jax.debug.print("Positions of own pieces: {positions}", positions=positions)
    # DEBUG NOTES: positions works correctly

    # vectorise function and apply to all positions
    vmapped_get_possible_moves = jax.vmap(get_possible_moves_for_piece, in_axes=(0, None))
    all_possible_moves = vmapped_get_possible_moves(positions, state)

    # jax.debug.print("All possible moves for all pieces: {all_possible_moves}", all_possible_moves=all_possible_moves)

    # masks for each piece
    can_move_mask = jnp.any(all_possible_moves != 0, axis=(1, 2))  # any move available
    can_jump_mask = jnp.any(jnp.all(jnp.abs(all_possible_moves) == 2, axis=2), axis=1)  # jump available
    any_jump_available = jnp.any(can_jump_mask)

    movable_mask = jnp.where(any_jump_available, can_jump_mask,
                             can_move_mask)  # this is just an if statement (cond, x ,y)

    movable_positions = jnp.where(
        movable_mask[:, None],  # Reshape mask to (MAX_PIECES, 1) for broadcasting
        positions,
        jnp.array([-1, -1])
    )

    return movable_positions


class JaxVideoCheckers(JaxEnvironment[VideoCheckersState, VideoCheckersObservation, VideoCheckersInfo,VideoCheckersConstants]):
    def __init__(self, reward_funcs: list[callable] = None):
        super().__init__()
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

    def reset(self, key: jax.random.PRNGKey = jax.random.PRNGKey(187)) -> Tuple[
        VideoCheckersObservation, VideoCheckersState]:
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
        board = jnp.zeros((NUM_FIELDS_X, NUM_FIELDS_Y), dtype=jnp.int32)
        # Set up the initial pieces on the board

        board = board.at[0, 1].set(WHITE_PIECE)
        board = board.at[0, 3].set(WHITE_PIECE)
        board = board.at[0, 5].set(WHITE_PIECE)
        board = board.at[0, 7].set(WHITE_PIECE)
        board = board.at[1, 0].set(WHITE_PIECE)
        board = board.at[1, 2].set(WHITE_PIECE)
        board = board.at[1, 4].set(WHITE_PIECE)
        board = board.at[1, 6].set(WHITE_PIECE)
        board = board.at[2, 1].set(WHITE_PIECE)
        board = board.at[2, 3].set(WHITE_PIECE)
        board = board.at[2, 5].set(WHITE_PIECE)
        board = board.at[2, 7].set(WHITE_PIECE)

        board = board.at[5, 0].set(BLACK_PIECE)
        board = board.at[5, 2].set(BLACK_PIECE)
        board = board.at[5, 4].set(BLACK_PIECE)
        board = board.at[5, 6].set(BLACK_PIECE)
        board = board.at[6, 1].set(BLACK_PIECE)
        board = board.at[6, 3].set(BLACK_PIECE)
        board = board.at[6, 5].set(BLACK_PIECE)
        board = board.at[6, 7].set(BLACK_PIECE)
        board = board.at[7, 0].set(BLACK_PIECE)
        board = board.at[7, 2].set(BLACK_PIECE)
        board = board.at[7, 4].set(BLACK_PIECE)
        board = board.at[7, 6].set(BLACK_PIECE)

        # Default state
        state = VideoCheckersState(cursor_pos=jnp.array([6, 7]), board=board, game_phase=SELECT_PIECE_PHASE,
                                   selected_piece=jnp.array([-1, -1]), frame_counter=jnp.array(0), winner=-1,
                                   additional_jump=False,
                                   opponent_move=OpponentMove(start_pos=jnp.array([-1, -1]),
                                                              end_pos=jnp.array([-1, -1]),
                                                              piece_type=-1,
                                                              captured_positions=jnp.array([[-1, -1]])
                                                              ))

        # Debug state move piece phase
        """state = VideoCheckersState(cursor_pos=jnp.array([4, 1]), board=board, game_phase=MOVE_PIECE_PHASE,
                                      selected_piece=jnp.array([5, 0]), frame_counter=jnp.array(0), winner=-1, additional_jump=False,
                                      opponent_move=OpponentMove(start_pos=jnp.array([-1, -1]),
                                                                  end_pos=jnp.array([-1, -1]),
                                                                  piece_type=-1,
                                                                  captured_positions=jnp.array([[-1, -1]])
                                                                  ))"""

        # Debug state show opponent move phase
        """
        testboard = jnp.zeros((NUM_FIELDS_Y, NUM_FIELDS_X), dtype=jnp.int32)
        testboard = testboard.at[1, 0].set(WHITE_PIECE)
        testboard = testboard.at[2, 1].set(BLACK_PIECE)
        testboard = testboard.at[4, 3].set(BLACK_PIECE)
        testboard = testboard.at[6, 5].set(BLACK_PIECE)
        testboard = testboard.at[0,1].set(WHITE_PIECE)
        testboard = testboard.at[0,3].set(WHITE_PIECE)
        testboard = testboard.at[0,5].set(WHITE_PIECE)
        testboard = testboard.at[0,7].set(WHITE_PIECE)
        testboard = testboard.at[7,0].set(BLACK_PIECE)
        testboard = testboard.at[7,2].set(BLACK_PIECE)
        testboard = testboard.at[6,1].set(BLACK_PIECE)
        state = VideoCheckersState(cursor_pos=jnp.array([4, 3]), board=testboard, game_phase=SHOW_OPPONENT_MOVE_PHASE,
                                      selected_piece=jnp.array([-1, -1]), frame_counter=jnp.array(0), winner=-1, additional_jump=False,
                                      opponent_move=OpponentMove(start_pos=jnp.array([1, 0]),
                                                                  end_pos=jnp.array([7, 6]),
                                                                  piece_type=WHITE_KING,
                                                                  captured_positions=jnp.array([[2, 1], [4, 3], [6, 5]])
                                                                  ))"""

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
            state.game_phase != SELECT_PIECE_PHASE,
            lambda _: jax.debug.print("Warning: Game phase is not SELECT_PIECE_PHASE, it is {game_phase}",
                                      game_phase=state.game_phase),
            lambda _: None,
            operand=None
        )

        initial_obs = self._get_observation(state)

        def expand_and_copy(x):
            x_expanded = jnp.expand_dims(x, axis=0)
            return jnp.concatenate([x_expanded] * self.frame_stack_size, axis=0)

        # Apply transformation to each leaf in the pytree
        initial_obs = jax.tree.map(expand_and_copy, initial_obs)

        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: VideoCheckersState):
        """
        Returns the observation of the game state.
        Args:
            state: The current game state.
        Returns:
            VideoCheckersObservation: The observation of the game state.
        """
        return VideoCheckersObservation(board=state.board,
                                        start_pos=state.cursor_pos,
                                        end_pos=state.selected_piece,
                                        must_jump=jnp.array(True, dtype=jnp.bool_))
        # TODO generate valid observation instead of placeholder

    # Important for phase changes. The fields used by the next phase must be reset. This means select_piece, etc.

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
                (piece != EMPTY_TILE) & is_movable_piece(COLOUR_BLACK, state.cursor_pos, state),
                lambda s: s._replace(
                    selected_piece=s.cursor_pos,
                    game_phase=MOVE_PIECE_PHASE,
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
            donw_left = (action == Action.DOWNLEFT) # this is to prevent illegal pure ordinal inputs to move it down left.

            drow = jax.lax.cond(up, lambda _: -1, lambda _: 1, operand=None)  # -1 for up, 1 for down
            dcol = jax.lax.cond(right, lambda _: 1, lambda _: -1, operand=None)  # 1 for right, -1 for left

            new_cursor_pos = state.cursor_pos + jnp.array([drow, dcol])
            # Check if the new position is within bounds
            in_bounds = move_in_bounds(row=state.cursor_pos[0],
                                       col=state.cursor_pos[1],
                                       drow=drow,
                                       dcol=dcol,
                                       state=state)
            new_cursor_pos = jax.lax.cond(
                in_bounds & jnp.logical_or(up, jnp.logical_or(right, donw_left)),
                lambda _: new_cursor_pos,
                lambda _: state.cursor_pos,
                operand=None
            )
            # jax.debug.print("New cursor position row: {new_cursor_pos[0]}, col: {new_cursor_pos[1]}", new_cursor_pos=new_cursor_pos)
            return state._replace(cursor_pos=new_cursor_pos)

        new_state = jax.lax.cond(
            action == Action.FIRE,
            lambda s: select_piece(s),
            lambda s: move_cursor(s, action),
            operand=state)

        return new_state

    def step_move_piece_phase(self, state: VideoCheckersState, action: chex.Array) -> VideoCheckersState:
        """
        Handles moving a piece in the move piece phase.
        Args:
            state: The current game state.
            action: The action taken by the player.
        Returns:
            VideoCheckersState: The new game state after the action.
        """

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

                # Check if the action is in the direction of the selected piece
                up = jnp.logical_or(action == Action.UPLEFT, action == Action.UPRIGHT)
                right = jnp.logical_or(action == Action.DOWNRIGHT, action == Action.UPRIGHT)

                drow = jax.lax.cond(up, lambda _: -1, lambda _: 1, operand=None)
                dcol = jax.lax.cond(right, lambda _: 1, lambda _: -1, operand=None)

                # Check if the action is in the direction of the selected piece
                is_correct_direction = jax.lax.cond(
                    (row_diff == drow) & (col_diff == dcol),
                    lambda _: True,
                    lambda _: False,
                    operand=None
                )

                # If the action is in the correct direction, move the cursor to the selected piece
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
                possible_moves = get_possible_moves_for_piece(state.selected_piece, state)

                move_vector, jump_vector = jax.lax.cond(
                    action == Action.UPRIGHT,
                    lambda _: (jnp.array([-1, 1]), jnp.array([-2, 2])),
                    lambda _: jax.lax.cond(
                        action == Action.UPLEFT,
                        lambda _: (jnp.array([-1, -1]), jnp.array([-2, -2])),
                        lambda _: jax.lax.cond(
                            action == Action.DOWNRIGHT,
                            lambda _: (jnp.array([1, 1]), jnp.array([2, 2])),
                            lambda _: jax.lax.cond(
                                action == Action.DOWNLEFT,
                                lambda _: (jnp.array([1, -1]), jnp.array([2, -2])),
                                lambda _: (jnp.array([0, 0]), jnp.array([0, 0])),  # Default case
                                operand=None
                            ),
                            operand=None
                        ),
                        operand=None
                    ),
                    operand=None
                )

                # Prüfe, ob der Zug gültig ist
                is_valid_move = jnp.logical_or(
                    jnp.any(jnp.all(possible_moves == move_vector, axis=1)),
                    jnp.any(jnp.all(possible_moves == jump_vector, axis=1))
                )

                # If the move is valid, update the cursor position
                return jax.lax.cond(
                    is_valid_move,
                    lambda s: s._replace(cursor_pos=s.cursor_pos + move_vector),
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

            def _place_piece(state: VideoCheckersState) -> VideoCheckersState:
                piece_type = state.board[state.selected_piece[0], state.selected_piece[1]]
                move = state.cursor_pos - state.selected_piece
                jumped = (jnp.abs(move[0]) > 2) & (jnp.abs(move[1]) > 2)

                jax.debug.print(
                    "Moving piece type {piece_type} from {selected_piece} to {cursor_pos}",
                    piece_type=piece_type,
                    selected_piece=state.selected_piece,
                    cursor_pos=state.cursor_pos)

                # move piece
                new_board = state.board.at[tuple(state.selected_piece)].set(EMPTY_TILE)
                new_board = new_board.at[tuple(state.cursor_pos)].set(piece_type)

                # get new state
                new_state = state._replace(board=new_board)

                # check for moved piece if jumps are available from new pos
                new_moves = get_possible_moves_for_piece(new_state.cursor_pos, new_state)
                move_distances = jnp.abs(new_moves - new_state.cursor_pos)  # Shape: (n_moves, 2)
                max_distances = jnp.max(move_distances, axis=1)  # Max distance per move
                has_jump_from_new_pos = jnp.any(max_distances > 1)

                # stay in same phase and reselect piece if jumped and can continue jumping, change phase if not
                return jax.lax.cond(
                    jumped & has_jump_from_new_pos,
                    lambda s: s._replace(selected_piece=s.cursor_pos),
                    # TODO check if resetting destination to -1 -1 is necessary if can make another jump
                    lambda s: s._replace(game_phase=SHOW_OPPONENT_MOVE_PHASE),
                    new_state
                )

            cursor_on_selected_piece = jnp.all(state.selected_piece == state.cursor_pos)

            return jax.lax.cond(
                cursor_on_selected_piece,
                lambda s: s._replace(selected_piece=jnp.array([-1, -1]), game_phase=SELECT_PIECE_PHASE),
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

            # Update the board with the opponent's move
            new_board = state.board.at[tuple(opponent_move.start_pos)].set(EMPTY_TILE)
            new_board = new_board.at[tuple(opponent_move.end_pos)].set(opponent_move.piece_type)

            # Remove captured pieces from the board
            new_board = jax.lax.fori_loop(
                0,
                opponent_move.captured_positions.shape[0],
                lambda i, board: board.at[tuple(opponent_move.captured_positions[i])].set(EMPTY_TILE),
                new_board
            )

            # Update the game state with the new board and reset cursor position
            return state._replace(
                board=new_board,
                game_phase=SELECT_PIECE_PHASE,  # Change phase back to select piece phase
                selected_piece=jnp.array([-1, -1]),  # Reset selected piece
            )

        new_state = jax.lax.cond(
            action == Action.NOOP,
            lambda s: s,  # No action taken, return the same state
            apply_opponent_move,
            operand=state
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
            (state.frame_counter == (ANIMATION_FRAME_RATE-1)) & (action != Action.NOOP),
            lambda _: jax.lax.cond(
                state.game_phase == SELECT_PIECE_PHASE,
                lambda _: self.step_select_piece_phase(state, action),
                lambda _: jax.lax.cond(
                    state.game_phase == MOVE_PIECE_PHASE,
                    lambda _: self.step_move_piece_phase(state, action),
                    lambda _: jax.lax.cond(
                        state.game_phase == SHOW_OPPONENT_MOVE_PHASE,
                        lambda _: self.step_show_opponent_move_phase(state, action),
                        lambda _: self.step_game_over_phase(state, action),
                        operand=None
                    ),
                    operand=None
                ),
                operand=None
            ),
            lambda _: state,
            operand=None
        )

        new_state = new_state._replace(frame_counter=(new_state.frame_counter + 1) % ANIMATION_FRAME_RATE)

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
        return jnp.array(list(self.action_set), dtype=jnp.int32)

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

    @partial(jax.jit, static_argnums=(0,))
    def _get_env_reward(self, previous_state: VideoCheckersState, state: VideoCheckersState):
        """
        Returns the environment reward based on the game state.
        Args:
            previous_state: The previous game state.
        """
        return 0  # TODO: Implement environment reward logic

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
        return False  # TODO: Implement game over logic


def load_sprites():
    """
    Load all sprites required for Flag Capture rendering.
    Returns:
        TODO
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
    def __init__(self):
        super().__init__()
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
        raster: jnp.ndarray = jnp.full((WIDTH, HEIGHT, 3), jnp.array([160, 96, 64], dtype=jnp.uint8))

        frame_bg = jr.get_sprite_frame(self.SPRITE_BG, 0)
        raster = jr.render_at(raster, OFFSET_X_BOARD, OFFSET_Y_BOARD, frame_bg)
        """
        jax.debug.print("Cursor position: {state.cursor_pos}", state=state)
        jax.debug.print("Phase: {state.game_phase}", state=state)
        jax.debug.print("Selected piece: {state.selected_piece}", state=state)
        """

        def determine_piece_type_select_phase(row, col, state):
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
                (state.frame_counter < (ANIMATION_FRAME_RATE / 2)) & (state.cursor_pos[0] == row) & (
                        state.cursor_pos[1] == col),
                lambda _: BLACK_CURSOR,
                lambda _: state.board[row, col],
                operand=None
            )

        def determine_piece_type_move_phase(row, col, state):
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

            def f_umoved(_):
                # If the piece is unmoved, blink fast
                return jax.lax.cond(
                    (state.frame_counter % 5 < 2) | (state.frame_counter % 5 > 3),
                    lambda _: EMPTY_TILE,
                    lambda _: state.board[row, col],
                    operand=None
                )

            def f_selected(_):
                # If the piece is selected, render it as BLACK_CURSOR if frame_counter < 30, else render it as its original sprite
                return jax.lax.cond(
                    state.frame_counter < (ANIMATION_FRAME_RATE / 2),
                    lambda _: BLACK_CURSOR,
                    lambda _: state.board[row, col],
                    operand=None
                )

            def f_cursor(_):
                # If the piece is the destination, render it as the selected piece's tile sprite if frame_counter < 30, else render it as EMPTY_TILE
                return jax.lax.cond(
                    state.frame_counter < (ANIMATION_FRAME_RATE / 2),
                    lambda _: state.board[state.selected_piece[0], state.selected_piece[1]],
                    # Use selected piece's tile sprite
                    lambda _: EMPTY_TILE,
                    operand=None
                )

            return jax.lax.cond(
                is_unmoved,
                f_umoved,
                lambda _: jax.lax.cond(
                    is_selected_piece,
                    f_selected,
                    lambda _: jax.lax.cond(
                        is_cursor_pos,
                        f_cursor,
                        lambda _: state.board[row, col],
                        operand=None
                    ),
                    operand=None
                ),
                operand=None)

        def determine_piece_type_show_opponent_move_phase(row, col, state):
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
                    lambda _: WHITE_CURSOR,
                    lambda _: jax.lax.cond(
                        is_end_pos,
                        lambda _: EMPTY_TILE,
                        lambda _: s.board[row, col],
                        operand=s
                    ),
                    operand=s
                )

            def f_after_move(s):
                return jax.lax.cond(
                    is_start_pos,
                    lambda _: EMPTY_TILE,
                    lambda s: jax.lax.cond(
                        is_end_pos,
                        lambda _: state.opponent_move.piece_type,  # Render end position as the opponent's piece type
                        lambda s: jax.lax.cond(
                            is_captured_pos,
                            lambda _: BLACK_CURSOR,  # Render captured positions as BLACK_CURSOR
                            lambda s: s.board[row, col],
                            operand=s
                        ),
                        operand=s
                    ),
                    operand=s
                )

            return jax.lax.cond(
                state.frame_counter < (ANIMATION_FRAME_RATE / 2),
                f_before_move,
                f_after_move,
                operand=state
            )

        def determine_piece_type_game_over_phase(row, col, state):
            # TODO This is either like the show opponent move phase or showing the last move of the player.
            return state.board[row, col]  # TODO

        def render_pieces_on_board(state, raster):
            def render_piece(row, col, raster):
                # call 4 different function to determine which piece to render depending on the phase of the game. No logic just call the 4 functions
                piece_type = jax.lax.cond(
                    state.game_phase == SELECT_PIECE_PHASE,
                    lambda _: determine_piece_type_select_phase(row, col, state),
                    lambda _: jax.lax.cond(
                        state.game_phase == MOVE_PIECE_PHASE,
                        lambda _: determine_piece_type_move_phase(row, col, state),
                        lambda _: jax.lax.cond(
                            state.game_phase == SHOW_OPPONENT_MOVE_PHASE,
                            lambda _: determine_piece_type_show_opponent_move_phase(row, col, state),
                            lambda _: determine_piece_type_game_over_phase(row, col, state),
                            operand=None
                        ),
                        operand=None
                    ),
                    operand=None
                )

                piece_frame = jr.get_sprite_frame(self.SPRITE_PIECES, piece_type)
                return jax.lax.cond(
                    (piece_frame is not None) & ((row + col) % 2 == 1),  # Only render on dark squares
                    lambda _: jr.render_at(
                        raster,
                        OFFSET_X_BOARD + 4 + col * 17,  # Calculate the position on the board
                        OFFSET_Y_BOARD + 2 + row * 13,
                        piece_frame,
                    ),
                    lambda _: raster,
                    None,
                )

            def render_row(row, raster):
                return jax.lax.fori_loop(
                    0, NUM_FIELDS_X, lambda col, raster: render_piece(row, col, raster), raster
                )

            return jax.lax.fori_loop(0, NUM_FIELDS_Y, render_row, raster)

        raster = render_pieces_on_board(state, raster)

        return raster
