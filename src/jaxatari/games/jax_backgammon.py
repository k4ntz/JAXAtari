import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple
from functools import partial

# Constants for game Environment
NUM_POINTS = 24  # Number of points on the board
NUM_CHECKERS = 15  # Number of checkers per player
BAR_INDEX = 24  # Index for the bar (captured pieces)
HOME_INDEX = 25  # Index for pieces that reached home
MAX_DICE = 2  # Number of dice

# Player constants
WHITE = 1  # Player 1 (white checkers)
BLACK = -1  # Player 2 (black checkers)


class BackgammonState(NamedTuple):
    """State representation for a backgammon game.

    Attributes:
        board: A 2x26 array representing the board positions:
               - First dimension: [0] for WHITE pieces, [1] for BLACK pieces
               - Second dimension: Points 0-23, bar (24), and home (25)
        dice: Current dice values (1-6)
        current_player: Current player (WHITE or BLACK)
        is_game_over: Whether the game is over
    """
    board: jnp.ndarray  # Shape (2, 26): [player, position]
    dice: jnp.ndarray  # Shape (2,): two dice values
    current_player: int  # WHITE or BLACK
    is_game_over: bool  # Whether the game is over


def init_state() -> BackgammonState:
    """Initialize the backgammon state with standard starting position."""
    # Initialize the board with zeros - a 2x26 array
    # First dimension: [0] for WHITE pieces, [1] for BLACK pieces
    # Second dimension: Points 0-23, bar (24), and home (25)
    board = jnp.zeros((2, 26), dtype=jnp.int32)

    # Set up WHITE pieces (player 1)
    board = board.at[0, 0].set(2)  # 2 pieces on point 0
    board = board.at[0, 11].set(5)  # 5 pieces on point 11
    board = board.at[0, 16].set(3)  # 3 pieces on point 16
    board = board.at[0, 18].set(5)  # 5 pieces on point 18

    # Set up BLACK pieces (player 2)
    board = board.at[1, 23].set(2)  # 2 pieces on point 23
    board = board.at[1, 12].set(5)  # 5 pieces on point 12
    board = board.at[1, 7].set(3)  # 3 pieces on point 7
    board = board.at[1, 5].set(5)  # 5 pieces on point 5

    # Start with no dice rolled
    dice = jnp.zeros(2, dtype=jnp.int32)

    # Set the current player to WHITE at the beginning of the game
    return BackgammonState(
        board=board,
        dice=dice,
        current_player=WHITE,
        is_game_over=False
    )


def reset(key: jax.Array) -> BackgammonState:
    """Reset the game to the initial state and roll initial dice."""
    state = init_state()
    dice, key = roll_dice(key)
    return state._replace(dice=dice)


def roll_dice(key: jax.Array) -> Tuple[jnp.ndarray, jax.Array]:
    """Roll two dice and return the results and the next key."""
    key, subkey = jax.random.split(key)
    dice = jax.random.randint(subkey, (MAX_DICE,), 1, 7)
    return dice, key

def get_player_index(player: int) -> int:
    return jax.lax.cond(player == WHITE, lambda _: 0, lambda _: 1, operand=None)

def is_valid_move(state: BackgammonState, from_point: int, to_point: int) -> bool:
    """Check if a move from from_point to to_point is valid."""
    board = state.board
    player = state.current_player
    player_idx = get_player_index(player)
    opponent_idx = 1 - player_idx

    # first condition: check if the player has a piece on the from_point
    has_piece = board[player_idx, from_point] > 0

    # second condition: check if the move is within the board limits
    in_bounds = (0 <= from_point) & (from_point < 24) & (0 <= to_point) & (to_point < 24)

    # Check if the player is moving from the bar
    # is_moving_from_bar = from_point == BAR_INDEX

    # third condition: check if the player is moving in the correct direction
    distance = jax.lax.cond(
        player == WHITE,
        lambda _: from_point - to_point,
        lambda _: to_point - from_point,
        operand=None
    )
    correct_direction = distance > 0

    # fourth condition: check if the move is valid according to the dice
    dice_match = (state.dice[0] == distance) | (state.dice[1] == distance)       #| (state.dice[0] + state.dice[1] == distance)

    # fifth condition: check if the destination point is not blocked
    not_blocked = board[opponent_idx, to_point] <= 1

    return has_piece & in_bounds & correct_direction & dice_match & not_blocked # & is_moving_from_bar


@jax.jit
def step(state: BackgammonState, action: Tuple[int, int], key: jax.Array) -> Tuple[BackgammonState, jax.Array]:
    """Execute a move action and update the game state."""
    from_point, to_point = action
    board = state.board
    player = state.current_player
    player_idx = get_player_index(player)
    opponent_idx = 1 - player_idx

    is_valid = is_valid_move(state, from_point, to_point)

    def execute_move(board):

        # Special case: moving from the bar
        board = jax.lax.cond(
            from_point == BAR_INDEX,
            lambda b: b.at[player_idx, BAR_INDEX].add(-1),
            lambda b: b.at[player_idx, from_point].add(-1),
            operand=board
        )

        # Special case: bearing off
        opponent_pieces = board[opponent_idx, to_point]
        board = jax.lax.cond(
            to_point == HOME_INDEX,
            lambda b: b.at[player_idx, HOME_INDEX].add(1),
            lambda b: jax.lax.cond(
                opponent_pieces == 1,
                lambda b: b.at[opponent_idx, to_point].set(0).at[opponent_idx, BAR_INDEX].add(1),
                lambda b: b,
                operand=board
            ),
            board
        )

        #remove piece from from_point(source)
        board = board.at[player_idx, from_point].add(-1)

        # Place piece at destination
        board = board.at[player_idx, to_point].add(1)

        return board

    # Execute the move if valid, otherwise keep the board unchanged
    new_board = jax.lax.cond(
        is_valid,
        execute_move,
        lambda b: b,
        operand=board
    )

    # Update the dice by removing the used dice
    distance = jax.lax.cond(
        from_point == BAR_INDEX,
        lambda a: jax.lax.cond(
            a[0] == WHITE,
            lambda b: 25 - b[1],  # Distance from bar for WHITE
            lambda b: b[1],  # Distance from bar for BLACK
            operand=(a[0], a[2])
        ),
        lambda a: jax.lax.cond(
            a[0] == WHITE,
            lambda b: b[1] - b[2],  # Regular move distance for WHITE
            lambda b: b[2] - b[1],  # Regular move distance for BLACK
            operand=(a[0], a[1], a[2])
        ),
        operand=(player, from_point, to_point)
    )

    """Update the dice based on the distance moved (especially sum of two dice)"""
    new_dice = jax.lax.cond(
        is_valid & (state.dice[0] == distance),
        lambda d: jnp.array([0, d[1]]),
        lambda d: jax.lax.cond(
            is_valid & (state.dice[1] == distance),
            lambda d2: jnp.array([d2[0], 0]),
            lambda d2: d2,
            operand=d
        ),
        operand=state.dice
    )

    """Check if the player is still able to move after using one dice"""
    # Check if all dice are used
    all_dice_used = (new_dice[0] == 0) & (new_dice[1] == 0)

    # Roll new dice and switch player if all dice are used
    def next_player_turn(key):
        next_dice, next_key = roll_dice(key)
        return next_dice, -state.current_player, next_key

    def same_player_turn(key):
        return new_dice, state.current_player, key

    """change the condition to see if there is still an available move depending on the dices"""
    next_dice, next_player, next_key = jax.lax.cond(
        all_dice_used,
        next_player_turn,
        same_player_turn,
        key
    )

    # Check if the game is over (all pieces at home)
    white_won = new_board[0, HOME_INDEX] == NUM_CHECKERS
    black_won = new_board[1, HOME_INDEX] == NUM_CHECKERS
    game_over = white_won | black_won

    # Create new state
    new_state = BackgammonState(
        board=new_board,
        dice=next_dice,
        current_player=next_player,
        is_game_over=game_over
    )

    return new_state, next_key


def get_valid_moves(state: BackgammonState) -> jnp.ndarray:
    """Return a matrix of valid moves for the current state.

    Returns:
        A NUM_POINTS x NUM_POINTS boolean matrix where True indicates a valid move
        from row index to column index.
    """
    valid_moves = jnp.zeros((NUM_POINTS, NUM_POINTS), dtype=bool)

    # For each possible starting point
    for from_point in range(NUM_POINTS):
        # For each possible destination
        for to_point in range(NUM_POINTS):
            valid_moves = valid_moves.at[from_point, to_point].set(
                is_valid_move(state, from_point, to_point)
            )

    return valid_moves


def render(state: BackgammonState) -> str:
    """Render the current state of the game in ASCII format. Used ChatGPT to generate this function."""
    board = state.board

    # Create a string buffer for the output
    output = []

    # Add board header
    output.append("  12 11 10  9  8  7  |   6  5  4  3  2  1")
    output.append("  -------------------------------------------------")

    # Render the board top row (points 12-1)
    top_row = ["  "]
    for i in range(12, 0, -1):
        point_idx = i - 1
        white = board[0, point_idx]
        black = board[1, point_idx]

        if white > 0:
            top_row.append("W{} ".format(white))
        elif black > 0:
            top_row.append("B{} ".format(black))
        else:
            top_row.append("•  ")

        if i == 7:
            top_row.append(" |   ")
    output.append("".join(top_row))

    # Add middle divider
    output.append("  -------------------------------------------------")

    # Render the board bottom row (points 13-24)
    bottom_row = ["  "]
    for i in range(13, 25):
        point_idx = i - 1
        white = board[0, point_idx]
        black = board[1, point_idx]

        if white > 0:
            bottom_row.append("W{} ".format(white))
        elif black > 0:
            bottom_row.append("B{} ".format(black))
        else:
            bottom_row.append("•  ")

        if i == 18:
            bottom_row.append(" |   ")
    output.append("".join(bottom_row))

    # Add board footer
    output.append("  -------------------------+------------------------")
    output.append("  13 14 15 16 17 18  |   19 20 21 22 23 24")

    # Add bar and home information
    output.append("")
    output.append("Bar: White: {}, Black: {}".format(board[0, BAR_INDEX], board[1, BAR_INDEX]))
    output.append("Home: White: {}, Black: {}".format(board[0, HOME_INDEX], board[1, HOME_INDEX]))

    # Add current player and dice information
    output.append("")
    output.append("Current player: {}".format("White" if state.current_player == WHITE else "Black"))
    output.append("Dice: {}".format(state.dice))
    output.append("Game over: {}".format(state.is_game_over))

    return "\n".join(output)


def run_game(key: jax.Array, max_steps=20):
    """Run a simple game with random moves for demonstration."""
    state = reset(key)
    print(render(state))

    for i in range(max_steps):
        if state.is_game_over:
            break

        # Find valid moves
        valid_moves = get_valid_moves(state)
        valid_coords = jnp.argwhere(valid_moves)

        # If no valid moves, roll dice and continue
        if len(valid_coords) == 0:
            dice, key = roll_dice(key)
            state = state._replace(
                dice=dice,
                current_player=-state.current_player
            )
            print("\nNo valid moves. Switching players.")
            print(render(state))
            continue

        # Choose a random valid move
        key, move_key = jax.random.split(key)
        move_idx = jax.random.randint(move_key, (), 0, len(valid_coords))
        from_point, to_point = valid_coords[move_idx]

        print(f"\nMove: {from_point} → {to_point}")

        # Execute the move
        state, key = step(state, (from_point, to_point), key)
        print(render(state))

    if state.is_game_over:
        print("\nGame over!")
    else:
        print("\nReached maximum steps.")

    return state


def main():
    key = jax.random.PRNGKey(0)
    run_game(key, max_steps=10)


if __name__ == "__main__":
    main()