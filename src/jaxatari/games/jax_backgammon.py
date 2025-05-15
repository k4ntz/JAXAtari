import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple, Any, List
from ..environment import JaxEnvironment, EnvState
from ..renderers import AtraJaxisRenderer

# Constants for game Environment
NUM_POINTS = 24
NUM_CHECKERS = 15
BAR_INDEX = 24
HOME_INDEX = 25
MAX_DICE = 2

WHITE = 1
BLACK = -1


class BackgammonState(NamedTuple):
    board: jnp.ndarray  # (2, 26)
    dice: jnp.ndarray  # (2,)
    current_player: int
    is_game_over: bool


class BackgammonInfo(NamedTuple):
    time: jnp.ndarray


class BackgammonObservation(NamedTuple):
    bar_counts: jnp.ndarray
    home_counts: jnp.ndarray


import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple, Any, List
from functools import partial
from ..environment import JaxEnvironment, EnvState
from ..renderers import AtraJaxisRenderer

# Constants for game Environment
NUM_POINTS = 24
NUM_CHECKERS = 15
BAR_INDEX = 24
HOME_INDEX = 25
MAX_DICE = 2

WHITE = 1
BLACK = -1


class BackgammonState(NamedTuple):
    board: jnp.ndarray  # (2, 26)
    dice: jnp.ndarray  # (2,)
    current_player: int
    is_game_over: bool


class BackgammonInfo(NamedTuple):
    time: jnp.ndarray


class BackgammonObservation(NamedTuple):
    bar_counts: jnp.ndarray
    home_counts: jnp.ndarray


class JaxBackgammonEnv(JaxEnvironment[BackgammonState, jnp.ndarray, dict]):
    def __init__(self, key: jax.Array):
        super().__init__()
        self.key = key
        # Pre-compute all possible moves for fast validation
        self.action_space = jnp.array([(i, j) for i in range(26) for j in range(26)], dtype=jnp.int32)

    @staticmethod
    @jax.jit
    def init_state_impl() -> BackgammonState:
        board = jnp.zeros((2, 26), dtype=jnp.int32)
        board = board.at[0, 0].set(2).at[0, 11].set(5).at[0, 16].set(3).at[0, 18].set(5)
        board = board.at[1, 23].set(2).at[1, 12].set(5).at[1, 7].set(3).at[1, 5].set(5)
        dice = jnp.zeros(2, dtype=jnp.int32)
        return BackgammonState(board=board, dice=dice, current_player=WHITE, is_game_over=False)

    def init_state(self) -> BackgammonState:
        return JaxBackgammonEnv.init_state_impl()

    @partial(jax.jit, static_argnums=(0,))
    def reset_impl(self, key):
        state = self.init_state()
        dice, new_key = JaxBackgammonEnv.roll_dice(key)
        state = state._replace(dice=dice)
        return self._get_observation(state), state, new_key

    def reset(self) -> Tuple[jnp.ndarray, BackgammonState]:
        obs, state, self.key = self.reset_impl(self.key)
        return obs, state

    @staticmethod
    @jax.jit
    def roll_dice(key: jax.Array) -> Tuple[jnp.ndarray, jax.Array]:
        key, subkey = jax.random.split(key)
        dice = jax.random.randint(subkey, (MAX_DICE,), 1, 7)
        return dice, key

    @staticmethod
    @jax.jit
    def get_player_index(player: int) -> int:
        return jax.lax.cond(player == WHITE, lambda _: 0, lambda _: 1, operand=None)

    @staticmethod
    @jax.jit
    def is_valid_move(state: BackgammonState, move: Tuple[int, int]) -> bool:
        from_point, to_point = move[0], move[1]
        board = state.board
        player = state.current_player
        player_idx = JaxBackgammonEnv.get_player_index(player)
        opponent_idx = 1 - player_idx

        has_piece = board[player_idx, from_point] > 0
        in_bounds = (0 <= from_point) & (from_point < 24) & (0 <= to_point) & (to_point < 24)

        # Efficient move distance calculation
        distance = jax.lax.select(
            player == WHITE,
            from_point - to_point,
            to_point - from_point
        )

        correct_direction = distance > 0
        dice_match = (state.dice[0] == distance) | (state.dice[1] == distance)
        not_blocked = board[opponent_idx, to_point] <= 1
        return has_piece & in_bounds & correct_direction & dice_match & not_blocked

    @staticmethod
    @jax.jit
    def execute_move(board, player_idx, opponent_idx, from_point, to_point):
        """Execute a single move on the board."""
        opponent_pieces = board[opponent_idx, to_point]

        # Handle home point or capturing
        board = jax.lax.cond(
            to_point == HOME_INDEX,
            lambda b: b.at[player_idx, HOME_INDEX].add(1),
            lambda b: jax.lax.cond(
                opponent_pieces == 1,
                lambda b: b.at[opponent_idx, to_point].set(0).at[opponent_idx, BAR_INDEX].add(1),
                lambda b: b,
                operand=b
            ),
            board
        )

        # Move the piece
        board = board.at[player_idx, from_point].add(-1)
        board = board.at[player_idx, to_point].add(1)
        return board

    @staticmethod
    @jax.jit
    def compute_distance(player, from_point, to_point):
        """Compute move distance based on player and points."""
        return jax.lax.cond(
            from_point == BAR_INDEX,
            lambda a: jax.lax.select(a[0] == WHITE, 25 - a[1], a[1]),
            lambda a: jax.lax.select(a[0] == WHITE, a[1] - a[2], a[2] - a[1]),
            (player, from_point, to_point)
        )

    @staticmethod
    @jax.jit
    def update_dice(dice, is_valid, distance):
        """Update dice after a move."""
        return jax.lax.cond(
            is_valid & (dice[0] == distance),
            lambda d: jnp.array([0, d[1]]),
            lambda d: jax.lax.cond(
                is_valid & (d[1] == distance),
                lambda d2: jnp.array([d2[0], 0]),
                lambda d2: d2,
                d
            ),
            dice
        )

    @partial(jax.jit, static_argnums=(0,))
    def step_impl(self, state: BackgammonState, action: Tuple[int, int], key: jax.Array) -> Tuple[
        jnp.ndarray, BackgammonState, float, bool, dict, jax.Array]:
        from_point, to_point = action
        board = state.board
        player = state.current_player
        player_idx = JaxBackgammonEnv.get_player_index(player)
        opponent_idx = 1 - player_idx

        # Use the vectorized is_valid_move function with a single move
        is_valid = JaxBackgammonEnv.is_valid_move(state, jnp.array([from_point, to_point]))

        # Execute the move if valid
        new_board = jax.lax.cond(
            is_valid,
            lambda _: JaxBackgammonEnv.execute_move(board, player_idx, opponent_idx, from_point, to_point),
            lambda _: board,
            operand=None
        )

        # Calculate move distance
        distance = JaxBackgammonEnv.compute_distance(player, from_point, to_point)

        # Update dice based on the move
        new_dice = JaxBackgammonEnv.update_dice(state.dice, is_valid, distance)

        # Check if all dice are used
        all_dice_used = (new_dice[0] == 0) & (new_dice[1] == 0)

        # Prepare for next turn if necessary
        def next_turn(k):
            next_dice, new_key = JaxBackgammonEnv.roll_dice(k)
            return next_dice, -state.current_player, new_key

        def same_turn(k):
            return new_dice, state.current_player, k

        next_dice, next_player, new_key = jax.lax.cond(all_dice_used, next_turn, same_turn, key)

        # Check for game over
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

        # Prepare return values
        obs = self._get_observation(new_state)
        reward = self._get_reward(state, new_state)
        done = self._get_done(new_state)
        info = self._get_info(new_state)

        return obs, new_state, reward, done, info, new_key

    def step(self, state: BackgammonState, action: Tuple[int, int]) -> Tuple[
        jnp.ndarray, BackgammonState, float, bool, dict]:
        obs, new_state, reward, done, info, self.key = self.step_impl(state, action, self.key)
        return obs, new_state, reward, done, info

    def get_action_space(self) -> jnp.ndarray:
        return self.action_space

    def get_observation_space(self) -> Tuple:
        return (2, 26), (2,), (), ()

    @staticmethod
    @jax.jit
    def _get_observation(state: BackgammonState) -> jnp.ndarray:
        """Convert state to observation vector."""
        return jnp.concatenate([
            state.board.flatten(),
            state.dice,
            jnp.array([state.current_player], dtype=jnp.int32),
            jnp.array([jnp.where(state.is_game_over, 1, 0)], dtype=jnp.int32)
        ])

    @staticmethod
    @jax.jit
    def _get_info(state: BackgammonState) -> dict:
        return {"player": state.current_player, "dice": state.dice}

    @staticmethod
    @jax.jit
    def _get_reward(prev: BackgammonState, state: BackgammonState) -> float:
        return jax.lax.select(
            state.is_game_over,
            jax.lax.select(state.current_player != prev.current_player, 1.0, -1.0),
            0.0
        )

    @staticmethod
    @jax.jit
    def _get_done(state: BackgammonState) -> bool:
        return state.is_game_over

    @partial(jax.jit, static_argnums=(0,))
    def _check_all_moves(self, state: BackgammonState) -> jnp.ndarray:
        """Create a boolean mask of valid moves."""
        batch_is_valid = jax.vmap(lambda move: JaxBackgammonEnv.is_valid_move(state, move))
        return batch_is_valid(self.action_space)

    def get_valid_moves_jit(self, state: BackgammonState) -> jnp.ndarray:
        """Get all valid moves for the current state using JAX."""
        # Get boolean mask of valid moves
        valid_mask = self._check_all_moves(state)
        # Use numpy to perform the indexing outside of JIT
        return jnp.array(self.action_space)[jnp.array(valid_mask)]

    def get_valid_moves(self, state: BackgammonState) -> List[Tuple[int, int]]:
        """Convert valid moves from array to list of tuples."""
        valid_moves_array = self.get_valid_moves_jit(state)
        return [tuple(map(int, move)) for move in valid_moves_array]

    def render(self, state: EnvState) -> Tuple[jnp.ndarray]:
        return

class BackgammonRenderer(AtraJaxisRenderer):
    def __init__(self, env: JaxBackgammonEnv):
        self.env = env

    def render(self, state: BackgammonState) -> str:
        """Render the current state of the game in ASCII format."""
        board = state.board
        output = []

        # Board header
        output.append("  12 11 10  9  8  7  |   6  5  4  3  2  1")
        output.append("  -------------------------------------------------")

        def render_row(indices: range, split_at: int):
            row = ["  "]
            for i in indices:
                point_idx = i - 1
                white = board[0, point_idx]
                black = board[1, point_idx]

                if white > 0:
                    row.append(f"W{white} ")
                elif black > 0:
                    row.append(f"B{black} ")
                else:
                    row.append("•  ")

                if i == split_at:
                    row.append(" |   ")
            return "".join(row)

        # Top row (points 12–1)
        output.append(render_row(range(12, 0, -1), 7))

        # Middle divider
        output.append("  -------------------------------------------------")

        # Bottom row (points 13–24)
        output.append(render_row(range(13, 25), 18))

        # Footer
        output.append("  -------------------------+------------------------")
        output.append("  13 14 15 16 17 18  |   19 20 21 22 23 24")

        # Bar and Home
        output.append("")
        output.append(f"Bar: White: {board[0, BAR_INDEX]}, Black: {board[1, BAR_INDEX]}")
        output.append(f"Home: White: {board[0, HOME_INDEX]}, Black: {board[1, HOME_INDEX]}")

        # Game status
        output.append("")
        output.append(f"Current player: {'White' if state.current_player == WHITE else 'Black'}")
        output.append(f"Dice: {state.dice}")
        output.append(f"Game over: {state.is_game_over}")

        return "\n".join(output)

    def display(self, state: BackgammonState) -> None:
        """Print the rendered state to the console."""
        print(self.render(state))

    def close(self):
        pass


def get_user_move(state: BackgammonState, env: JaxBackgammonEnv) -> Tuple[int, int]:
    """Get a move from the user via keyboard input."""
    valid_moves = env.get_valid_moves(state)

    if not valid_moves:
        print("No valid moves available. Press Enter to continue to next player's turn.")
        input()
        return None

    # Display valid moves with numbers for selection
    print("\nValid moves:")
    for i, move in enumerate(valid_moves):
        from_point, to_point = move
        from_display = "BAR" if from_point == BAR_INDEX else str(from_point + 1)
        to_display = "HOME" if to_point == HOME_INDEX else str(to_point + 1)
        print(f"{i + 1}: {from_display} → {to_display}")

    # Get user input
    while True:
        try:
            choice = input("\nEnter move number (or 'q' to quit): ")
            if choice.lower() == 'q':
                return 'quit'

            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(valid_moves):
                return valid_moves[choice_idx]
            else:
                print(f"Please enter a number between 1 and {len(valid_moves)}")
        except ValueError:
            print("Please enter a valid number")

def run_game_without_input(key: jax.Array, max_steps=20):
    env = JaxBackgammonEnv(key)
    obs, state = env.reset()
    renderer = BackgammonRenderer(env)
    env.renderer = renderer

    for i in range(max_steps):
        if state.is_game_over:
            print("\nGame Over!")
            break

        valid_moves = env.get_valid_moves(state)

        if len(valid_moves) == 0:
            state = state._replace(
                dice=env._roll_dice(),
                current_player=-state.current_player
            )
            env.render(state)
            continue

        # Choose random move
        env.key, move_key = jax.random.split(env.key)
        idx = jax.random.randint(move_key, (), 0, len(valid_moves))
        action = valid_moves[idx]

        renderer.display(state)
        print(f"\nMove: {action[0] + 1} → {action[1] + 1}")
        obs, state, reward, done, info = env.step(state, action)

        if done:
            print("\nGame Over!")
            break

    return state

def run_game_with_input(key: jax.Array, max_steps=100):
    """Run the backgammon game with keyboard input for moves."""
    env = JaxBackgammonEnv(key)
    # Create and attach the renderer
    renderer = BackgammonRenderer(env)
    env.renderer = renderer

    obs, state = env.reset()

    print("\n==== Welcome to JAX Backgammon ====")
    print("Points are numbered 1-24, with 1-6 on the white home board.")
    print("BAR refers to the bar, and HOME refers to moving pieces off the board.")

    step_count = 0
    while step_count < max_steps and not state.is_game_over:
        # Use the renderer to display the board
        renderer.display(state)

        # Get move from the user
        action = get_user_move(state, env)

        if action == 'quit':
            print("\nGame ended by user.")
            break

        if action is None:
            # No valid moves, roll new dice and switch players
            new_dice, new_key = env.roll_dice(env.key)
            env.key = new_key
            state = state._replace(
                dice=new_dice,
                current_player=-state.current_player
            )
            continue

        # Execute the move
        from_display = "BAR" if action[0] == BAR_INDEX else str(action[0] + 1)
        to_display = "HOME" if action[1] == HOME_INDEX else str(action[1] + 1)
        print(f"\nExecuting move: {from_display} → {to_display}")
        obs, state, reward, done, info = env.step(state, action)

        step_count += 1

        if done:
            renderer.display(state)
            winner = "White" if state.board[0, HOME_INDEX] == NUM_CHECKERS else "Black"
            print(f"\n==== Game Over! {winner} wins! ====")

    if step_count >= max_steps:
        print("\nMaximum steps reached. Game ended.")

    renderer.close()
    return state

def main():
    key = jax.random.PRNGKey(0)
    # run_game_without_input(key)
    run_game_with_input(key)


if __name__ == "__main__":
    main()
