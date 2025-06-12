import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple, Tuple, Any, List
from jaxatari.environment import JaxEnvironment, EnvState
from jaxatari.renderers import AtraJaxisRenderer

"""
Contribuors: Ayush Bansal, Mahta Mollaeian, Anh Tuan Nguyen, Abdallah Siwar  

Game: JAX Backgammon

This module defines a JAX-accelerated backgammon environment for reinforcement learning and simulation.
It includes the environment class, state structures, move validation and execution logic, rendering, and user interaction.
"""

# Constants for game Environment
NUM_POINTS = 24
NUM_CHECKERS = 15
BAR_INDEX = 24
HOME_INDEX = 25
MAX_DICE = 2
WHITE_HOME = jnp.array(range(19, 25))
BLACK_HOME = jnp.array(range(0, 6))


WHITE = 1
BLACK = -1


class BackgammonState(NamedTuple):
    """Represents the complete state of a backgammon game."""
    board: jnp.ndarray  # (2, 26)
    dice: jnp.ndarray  # (4,)
    current_player: int
    is_game_over: bool


class BackgammonInfo(NamedTuple):
    """Contains auxiliary information about the environment (e.g., timing or metadata)."""
    time: jnp.ndarray


class BackgammonObservation(NamedTuple):
    """Simplified observation structure containing counts on bar and home."""
    bar_counts: jnp.ndarray
    home_counts: jnp.ndarray

class JaxBackgammonEnv(JaxEnvironment[BackgammonState, jnp.ndarray, dict]):
    """
    JAX-based backgammon environment supporting JIT compilation and vectorized operations.
    Provides functionality for state initialization, step transitions, valid move evaluation, and observation generation.
    """
    def __init__(self, key: jax.Array):
        super().__init__()
        self.key = key
        # Pre-compute all possible moves for fast validation
        self.action_space = jnp.array([(i, j) for i in range(26) for j in range(26)], dtype=jnp.int32)

    @staticmethod
    @jax.jit
    def init_state(key) -> BackgammonState:
        board = jnp.zeros((2, 26), dtype=jnp.int32)
        board = board.at[0, 0].set(2).at[0, 11].set(5).at[0, 16].set(3).at[0, 18].set(5)
        board = board.at[1, 23].set(2).at[1, 12].set(5).at[1, 7].set(3).at[1, 5].set(5)
        dice = jnp.zeros(4, dtype=jnp.int32)

        #The condition for the while loop
        def cond_fun(carry):
            white_roll, black_roll, key = carry
            return white_roll == black_roll
        #The code to be run in the while loop
        def body_fun(carry):
            _, _, key = carry
            key, subkey1, subkey2 = jax.random.split(key, 3)
            white_roll = jax.random.randint(subkey1, (), 1, 7)
            black_roll = jax.random.randint(subkey2, (), 1, 7)
            return (white_roll, black_roll, key)

        # Generate the first dice throw
        key, subkey1, subkey2 = jax.random.split(key, 3)
        white_roll = jax.random.randint(subkey1, (), 1, 7)
        black_roll = jax.random.randint(subkey2, (), 1, 7)
        carry = (white_roll, black_roll, key)

        white_roll, black_roll, key = jax.lax.while_loop(cond_fun, body_fun, carry)

        # Set the player who rolled higher
        current_player = jax.lax.cond(
        white_roll > black_roll,
        lambda _: WHITE,
        lambda _: BLACK,
        operand=None
        )

        # Prepare initial dice values for that player
        first_die = jax.lax.cond(current_player == WHITE, lambda _: white_roll, lambda _: black_roll, operand=None)
        second_die = jax.lax.cond(current_player == WHITE, lambda _: black_roll, lambda _: white_roll, operand=None)

        is_double = first_die == second_die
        dice = jax.lax.cond(
            is_double,
            lambda _: jnp.array([first_die] * 4),
            lambda _: jnp.array([first_die, second_die, 0, 0]),
            operand=None
        )

        return BackgammonState(
            board=board,
            dice=dice,
            current_player=current_player,
            is_game_over=False
        )

    def reset(self) -> Tuple[jnp.ndarray, BackgammonState]:
        state = self.init_state(self.key)
        dice, self.key = self.roll_dice(self.key)
        state = state._replace(dice=dice)
        return self._get_observation(state), state

    @staticmethod
    @jax.jit
    def roll_dice(key: jax.Array) -> Tuple[jnp.ndarray, jax.Array]:
        """
        Roll two dice and expand to shape (4,):
        - If not a double: [d1, d2, 0, 0]
        - If a double:     [d, d, d, d]
        """
        key, subkey = jax.random.split(key)
        dice = jax.random.randint(subkey, (2,), 1, 7)
        is_double = dice[0] == dice[1]

        expanded_dice = jax.lax.cond(
            is_double,
            lambda d: jnp.array([d[0], d[0], d[0], d[0]]),  # use all 4 moves
            lambda d: jnp.array([d[0], d[1], 0, 0]),         # only 2 dice used
            operand=dice
        )

        return expanded_dice, key

    @staticmethod
    @jax.jit
    def get_player_index(player: int) -> int:
        return jax.lax.cond(player == WHITE, lambda _: 0, lambda _: 1, operand=None)

    @staticmethod
    @jax.jit
    def is_valid_move(state: BackgammonState, move: Tuple[int, int]) -> bool:
        from_point, to_point = move
        board = state.board
        player = state.current_player
        player_idx = JaxBackgammonEnv.get_player_index(player)
        opponent_idx = 1 - player_idx

        in_bounds = (0 <= from_point) & (from_point < 25) & (0 <= to_point) & (to_point <= HOME_INDEX)
        same_point = from_point == to_point
        has_bar_checkers = board[player_idx, BAR_INDEX] > 0
        moving_from_bar = from_point == BAR_INDEX
        must_move_from_bar = (~moving_from_bar) & has_bar_checkers

        early_invalid = (~in_bounds) | must_move_from_bar | same_point

        def return_false(_):
            return False

        def continue_check(_):
            def bar_case(_):
                def is_valid_entry(die_val: int) -> bool:
                    expected_entry = jax.lax.select(player == WHITE, die_val - 1, 24 - die_val)
                    matches_entry = to_point == expected_entry
                    entry_open = board[opponent_idx, expected_entry] <= 1
                    return matches_entry & entry_open

                bar_has_checker = board[player_idx, BAR_INDEX] > 0
                bar_entry_valid = jnp.any(jax.vmap(is_valid_entry)(state.dice))
                return bar_has_checker & bar_entry_valid

            def bearing_off_case(_):
                can_bear_off = JaxBackgammonEnv.check_bearing_off(state, player)
                has_piece = board[player_idx, from_point] > 0

                bearing_off_distance = jax.lax.cond(
                    player == WHITE,
                    lambda _: HOME_INDEX - from_point,
                    lambda _: from_point + 1,
                    operand=None
                )

                dice_match = jnp.any(state.dice == bearing_off_distance)

                point_indices = jnp.arange(24)

                higher_mask = jax.lax.cond(
                    player == WHITE,
                    lambda _: point_indices > from_point,
                    lambda _: point_indices < from_point,
                    operand=None
                )

                higher_checkers = jnp.any(jnp.where(higher_mask, board[player_idx, :24], 0) > 0)
                larger_die = jnp.any(state.dice > bearing_off_distance)

                valid_bear = dice_match | ((~higher_checkers) & larger_die)

                return can_bear_off & has_piece & valid_bear

            def normal_case(_):
                has_piece = board[player_idx, from_point] > 0
                not_blocked = board[opponent_idx, to_point] <= 1
                base_distance = jax.lax.select(player == WHITE, to_point - from_point, from_point - to_point)
                correct_direction = base_distance > 0
                dice_match = jnp.any(state.dice == base_distance)
                return has_piece & not_blocked & correct_direction & dice_match

            return jax.lax.cond(
                moving_from_bar,
                bar_case,
                lambda _: jax.lax.cond(
                    to_point == HOME_INDEX,
                    bearing_off_case,
                    normal_case,
                    operand=None
                ),
                operand=None
            )

        return jax.lax.cond(early_invalid, return_false, continue_check, operand=None)

    @staticmethod
    def check_bearing_off(state: BackgammonState, player: int) -> bool:
        """check for bearing off using lax.cond instead of if statements."""
        board = state.board
        player_idx = jax.lax.cond(player == WHITE, lambda _: 0, lambda _: 1, operand=None)

        home_range = jax.lax.cond(player == WHITE, lambda _: WHITE_HOME, lambda _: BLACK_HOME, operand=None)

        all_in_home_board = jnp.sum(board[player_idx, home_range]) == NUM_CHECKERS
        none_on_bar = board[player_idx, BAR_INDEX] == 0

        return all_in_home_board & none_on_bar

    @staticmethod
    @jax.jit
    def execute_move(board, player_idx, opponent_idx, from_point, to_point):
        """Apply a move to the board, updating for possible hits or bearing off."""
        opponent_pieces = board[opponent_idx, to_point]

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

        board = board.at[player_idx, from_point].add(-1)
        board = board.at[player_idx, to_point].add(1)
        return board

    @staticmethod
    @jax.jit
    def compute_distance(player, from_point, to_point):
        """Compute move distance based on player and points, including bearing off."""
        is_from_bar = from_point == BAR_INDEX

        bar_distance = jax.lax.cond(
            player == WHITE,
            lambda _: to_point + 1,          
            lambda _: 24 - to_point,
            operand=None
        )

        regular_distance = jax.lax.cond(
            to_point == HOME_INDEX,
            lambda _: jax.lax.cond(
                player == WHITE,
                lambda _: HOME_INDEX - from_point,
                lambda _: from_point + 1,
                operand=None
            ),
            lambda _: jax.lax.cond(
                player == WHITE,
                lambda _: to_point - from_point,
                lambda _: from_point - to_point,
                operand=None
            ),
            operand=None
        )

        return jax.lax.cond(is_from_bar, lambda _: bar_distance, lambda _: regular_distance, operand=None)
    
    @staticmethod
    @jax.jit
    def update_dice(dice: jnp.ndarray, is_valid: bool, distance: int) -> jnp.ndarray:
        """Consume one matching die (only the first match). Works with up to 4 dice."""

        def consume_one(dice):
            def scan_fn(carry, i):
                d, used = carry
                should_consume = (~used) & (d[i] == distance)
                new_d = jax.lax.cond(
                    should_consume,
                    lambda _: d.at[i].set(0),
                    lambda _: d,
                    operand=None
                )
                new_used = used | should_consume
                return (new_d, new_used), None

            (updated_dice, _), _ = jax.lax.scan(scan_fn, (dice, False), jnp.arange(4))
            return updated_dice

        return jax.lax.cond(is_valid, consume_one, lambda d: d, dice)

    @partial(jax.jit, static_argnums=(0,))
    def step_impl(self, state: BackgammonState, action: Tuple[int, int], key: jax.Array) -> Tuple[jnp.ndarray, BackgammonState, float, bool, dict, jax.Array]:
        """Perform a step in the environment, applying the action and returning the new state."""
        from_point, to_point = action
        board = state.board
        player = state.current_player
        player_idx = JaxBackgammonEnv.get_player_index(player)
        opponent_idx = 1 - player_idx

        # use the vectorized is_valid_move function with a single move
        is_valid = JaxBackgammonEnv.is_valid_move(state, jnp.array([from_point, to_point]))

        # execute the move if valid
        new_board = jax.lax.cond(
            is_valid,
            lambda _: JaxBackgammonEnv.execute_move(board, player_idx, opponent_idx, from_point, to_point),
            lambda _: board,
            operand=None
        )

        # calculate move distance
        distance = JaxBackgammonEnv.compute_distance(player, from_point, to_point)

        # update dice based on the move
        new_dice = JaxBackgammonEnv.update_dice(state.dice, is_valid, distance)

        # check if all dice are used
        all_dice_used = jnp.all(new_dice == 0)

        # prepare for next turn based on the outcome
        def next_turn(k):
            """ Switch to the next player and roll new dice."""
            next_dice, new_key = JaxBackgammonEnv.roll_dice(k)
            return next_dice, -state.current_player, new_key

        def same_turn(k):
            """ Continue with the same player."""
            return new_dice, state.current_player, k

        next_dice, next_player, new_key = jax.lax.cond(all_dice_used, next_turn, same_turn, key)

        # check winner conditions
        white_won = new_board[0, HOME_INDEX] == NUM_CHECKERS
        black_won = new_board[1, HOME_INDEX] == NUM_CHECKERS
        game_over = white_won | black_won

        # update game state
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

    def step(self, state: BackgammonState, action: Tuple[int, int]) -> Tuple[jnp.ndarray, BackgammonState, float, bool, dict]:
        """Perform a step in the environment, applying the action and returning the new state."""
        obs, new_state, reward, done, info, self.key = self.step_impl(state, action, self.key)
        return obs, new_state, reward, done, info

    def get_action_space(self) -> jnp.ndarray:
        """Return the action space for the environment."""
        return self.action_space

    def get_observation_space(self) -> Tuple:
        """Return the observation space for the environment."""
        return (2, 26), (4,), (), ()

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
        """Return auxiliary information about the environment."""
        return {"player": state.current_player, "dice": state.dice}

    @staticmethod
    @jax.jit
    def _get_reward(prev: BackgammonState, state: BackgammonState) -> float:
        """Calculate the reward based on the game state."""
        return jax.lax.select(
            state.is_game_over,
            jax.lax.select(state.current_player != prev.current_player, 1.0, -1.0),
            0.0
        )

    @staticmethod
    @jax.jit
    def _get_done(state: BackgammonState) -> bool:
        """Check if the game is over."""
        return state.is_game_over

    def get_valid_moves(self, state: BackgammonState) -> List[Tuple[int, int]]:
        @jax.jit
        def _check_all_moves(state):
            return jax.vmap(lambda move: self.is_valid_move(state, move))(self.action_space)

        valid_mask = _check_all_moves(state)
        valid_moves_array = self.action_space[jnp.array(valid_mask)]
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

    print("\nValid moves:")
    for i, move in enumerate(valid_moves):
        from_point, to_point = move
        from_display = "BAR" if from_point == BAR_INDEX else str(from_point + 1)
        to_display = "HOME" if to_point == HOME_INDEX else str(to_point + 1)
        print(f"{i + 1}: {from_display} → {to_display}")

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

def run_game_without_input(key: jax.Array, max_steps=200):
    """Run the backgammon game without user input, using random moves."""
    env = JaxBackgammonEnv(key)
    obs, state = env.reset()
    renderer = BackgammonRenderer(env)
    env.renderer = renderer
    env.key = key

    print(f"Initial roll: White {state.dice[0]}, Black {state.dice[1]}")
    print(f"{'White' if state.current_player == WHITE else 'Black'} will start the game!")

    for i in range(max_steps):
        if state.is_game_over:
            print("\nGame Over!")
            break

        valid_moves = env.get_valid_moves(state)
        new_dice, new_key  = env.roll_dice(key) # change the RNG key to get better randomness
        env.key = new_key

        if len(valid_moves) == 0:
            state = state._replace(
                dice=new_dice,
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

    print(f"Initial roll: White {state.dice[0]}, Black {state.dice[1]}")
    print(f"{'White' if state.current_player == WHITE else 'Black'} will start the game!")
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
