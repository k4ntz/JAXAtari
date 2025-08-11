# Third party imports
import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple, Tuple, Any, List
from jax import Array
import os
from pathlib import Path

# Project imports
from jaxatari.environment import JaxEnvironment
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as jr
import jaxatari.spaces as spaces

"""
Contribuors: Ayush Bansal, Mahta Mollaeian, Anh Tuan Nguyen, Abdallah Siwar  

Game: JAX Backgammon

This module defines a JAX-accelerated backgammon environment for reinforcement learning and simulation.
It includes the environment class, state structures, move validation and execution logic, rendering, and user interaction.
"""
class BackgammonConstants(NamedTuple):
# Constants for game Environment
    NUM_POINTS = 24
    NUM_CHECKERS = 15
    BAR_INDEX = 24
    HOME_INDEX = 25
    MAX_DICE = 2
    WHITE_HOME = jnp.array(range(18, 24))
    BLACK_HOME = jnp.array(range(0, 6))


    WHITE = 1
    BLACK = -1


class BackgammonState(NamedTuple):
    """Represents the complete state of a backgammon game."""
    board: jnp.ndarray  # (2, 26)
    dice: jnp.ndarray  # (4,)
    current_player: int
    is_game_over: bool
    key: jax.random.PRNGKey


class BackgammonInfo(NamedTuple):
    """Contains auxiliary information about the environment (e.g., timing or metadata)."""
    time: jnp.ndarray


class BackgammonObservation(NamedTuple):
    """Simplified observation structure containing counts on bar and home."""
    bar_counts: jnp.ndarray
    home_counts: jnp.ndarray

class JaxBackgammonEnv(JaxEnvironment[BackgammonState, jnp.ndarray, dict, BackgammonConstants]):
    """
    JAX-based backgammon environment supporting JIT compilation and vectorized operations.
    Provides functionality for state initialization, step transitions, valid move evaluation, and observation generation.
    """
    def __init__(self,consts: BackgammonConstants = None):
        consts = consts or BackgammonConstants()
        super().__init__(consts)

        # Pre-compute all possible moves (indexed as a scalar in the framework)
        self._action_pairs = jnp.array([(i, j) for i in range(26) for j in range(26)], dtype=jnp.int32)
        self.renderer = BackgammonRenderer(self)

    @partial(jax.jit, static_argnums=(0,))
    def init_state(self,key) -> BackgammonState:
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
        lambda _: self.consts.WHITE,
        lambda _: self.consts.BLACK,
        operand=None
        )

        # Prepare initial dice values for that player
        first_die = jax.lax.cond(current_player == self.consts.WHITE, lambda _: white_roll, lambda _: black_roll, operand=None)
        second_die = jax.lax.cond(current_player == self.consts.WHITE, lambda _: black_roll, lambda _: white_roll, operand=None)

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
            is_game_over=False,
            key = key
        )

    def reset(self, key: jax.random.PRNGKey = None) -> Tuple[jnp.ndarray, BackgammonState]:
        print(key)
        key = jax.lax.cond(
            key is None,
            lambda _: jax.random.PRNGKey(0),
            lambda _: key,
            operand = None
        )
        state = self.init_state(key)
        dice, key = self.roll_dice(key)
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

    @partial(jax.jit, static_argnums=(0,))
    def get_player_index(self, player: int) -> int:
        return jax.lax.cond(player == self.consts.WHITE, lambda _: 0, lambda _: 1, operand=None)

    @partial(jax.jit, static_argnums=(0,))
    def is_valid_move(self, state: BackgammonState, move: Tuple[int, int]) -> bool:
        from_point, to_point = move
        board = state.board
        player = state.current_player
        player_idx = self.get_player_index(player)
        opponent_idx = 1 - player_idx

        in_bounds = ((0 <= from_point) & (from_point <= 24) & (0 <= to_point) & (to_point <= self.consts.HOME_INDEX) & (to_point != self.consts.BAR_INDEX))
        # Convert from_point and to_point to JAX arrays to support JIT
        from_point = jnp.asarray(from_point)
        to_point = jnp.asarray(to_point)

        # Logical flags
        same_point = from_point == to_point
        has_bar_checkers = board[player_idx, self.consts.BAR_INDEX] > 0
        moving_from_bar = from_point == self.consts.BAR_INDEX
        must_move_from_bar = jnp.logical_not(moving_from_bar) & has_bar_checkers
        moving_to_bar = to_point == self.consts.BAR_INDEX

        # Early rejection
        early_invalid = jnp.logical_not(in_bounds) | must_move_from_bar | same_point | moving_to_bar


        def return_false(_):
            return False

        def continue_check(_):
            def bar_case(_):
                def is_valid_entry(die_val: int) -> bool:
                    expected_entry = jax.lax.select(player == self.consts.WHITE, die_val - 1, 24 - die_val)
                    matches_entry = to_point == expected_entry
                    entry_open = board[opponent_idx, expected_entry] <= 1
                    return matches_entry & entry_open

                bar_has_checker = board[player_idx, self.consts.BAR_INDEX] > 0
                bar_entry_valid = jnp.any(jax.vmap(is_valid_entry)(state.dice))
                return bar_has_checker & bar_entry_valid

            def bearing_off_case(_):
                can_bear_off = self.check_bearing_off(state, player)

                bearing_off_distance = jax.lax.cond(
                    player == self.consts.WHITE,
                    lambda _: self.consts.HOME_INDEX - from_point - 1,
                    lambda _: from_point + 1,
                    operand=None
                )

                dice_match = jnp.any(state.dice == bearing_off_distance)

                def white_check():
                    slice_start = 18
                    slice_len = 6  # White home points: 18–23
                    full_home = jax.lax.dynamic_slice(board[player_idx], (slice_start,), (slice_len,))

                    # Create a mask: only keep points strictly above from_point
                    mask = jnp.arange(18, 24) < from_point
                    return jnp.any(full_home * mask > 0)



                def black_check():
                    slice_start = 0
                    slice_len = 6  # Black home points: 0–5
                    full_home = jax.lax.dynamic_slice(board[player_idx], (slice_start,), (slice_len,))

                    # Keep only points strictly above from_point
                    mask = jnp.arange(0, 6) > from_point
                    return jnp.any(full_home * mask > 0)


                higher_checkers_exist = jax.lax.cond(
                    player == self.consts.WHITE,
                    lambda _: white_check(),
                    lambda _: black_check(),
                    operand=None
                )

                # Larger die than needed is allowed only if no higher checkers
                larger_die_available = jnp.any(state.dice > bearing_off_distance)

                # Checker must be present at the from_point
                has_piece = board[player_idx, from_point] > 0

                valid_bear = has_piece & (dice_match | ((~higher_checkers_exist) & larger_die_available))

                return jax.lax.cond(
                    can_bear_off,
                    lambda _: valid_bear,
                    lambda _: False,
                    operand=None
                )

            def normal_case(_):
                has_piece = board[player_idx, from_point] > 0
                not_blocked = board[opponent_idx, to_point] <= 1
                base_distance = jax.lax.select(player == self.consts.WHITE, to_point - from_point, from_point - to_point)
                correct_direction = base_distance > 0
                dice_match = jnp.any(state.dice == base_distance)
                not_moving_to_bar = to_point != self.consts.BAR_INDEX
                return has_piece & not_blocked & correct_direction & dice_match & not_moving_to_bar

            return jax.lax.cond(
                moving_from_bar,
                bar_case,
                lambda _: jax.lax.cond(
                    to_point == self.consts.HOME_INDEX,
                    bearing_off_case,
                    normal_case,
                    operand=None
                ),
                operand=None
            )

        return jax.lax.cond(early_invalid, return_false, continue_check, operand=None)

    @partial(jax.jit, static_argnums=(0,))
    def check_bearing_off(self, state: BackgammonState, player: int) -> bool:
        """check for bearing off using lax.cond instead of if statements."""
        board = state.board
        player_idx = self.get_player_index(player)

        # Full 0–23 range (playable points)
        point_indices = jnp.arange(24)

        # Mask for non-home points
        non_home_mask = jnp.where(player == self.consts.WHITE,
                               point_indices < 18,   # Points 0–17 (before 19)
                               point_indices > 5)

        in_play = board[player_idx, :24]
        outside_home_checkers = jnp.sum(jnp.where(non_home_mask, in_play, 0))
        on_bar = board[player_idx, self.consts.BAR_INDEX]
        return (outside_home_checkers == 0) & (on_bar == 0)

    @partial(jax.jit, static_argnums=(0,))
    def execute_move(self, board, player_idx, opponent_idx, from_point, to_point):
        """Apply a move to the board, updating for possible hits or bearing off."""
        # Remove checker from source first
        board = board.at[player_idx, from_point].add(-1)

        # If hitting opponent, update opponent’s bar and clear their point
        board = jax.lax.cond(
            (to_point != self.consts.HOME_INDEX) & (board[opponent_idx, to_point] == 1),
            lambda b: b.at[opponent_idx, to_point].set(0).at[opponent_idx, self.consts.BAR_INDEX].add(1),
            lambda b: b,
            operand=board
        )

        # Add to destination: either to_point or HOME_INDEX
        board = jax.lax.cond(
            to_point == self.consts.HOME_INDEX,
            lambda b: b.at[player_idx, self.consts.HOME_INDEX].add(1),
            lambda b: b.at[player_idx, to_point].add(1),
            operand=board
        )
        return board

    @partial(jax.jit, static_argnums=(0,))
    def compute_distance(self, player, from_point, to_point):
        """Compute move distance based on player and points, including bearing off."""
        is_from_bar = from_point == self.consts.BAR_INDEX

        bar_distance = jax.lax.cond(
            player == self.consts.WHITE,
            lambda _: to_point + 1,
            lambda _: 24 - to_point,
            operand=None
        )

        regular_distance = jax.lax.cond(
            to_point == self.consts.HOME_INDEX,
            lambda _: jax.lax.cond(
                player == self.consts.WHITE,
                lambda _: to_point - from_point - 1, #needs to be checked
                lambda _: from_point + 1,
                operand=None
            ),
            lambda _: jax.lax.cond(
                player == self.consts.WHITE,
                lambda _: to_point - from_point,
                lambda _: from_point - to_point,
                operand=None
            ),
            operand=None
        )

        return jax.lax.cond(is_from_bar, lambda _: bar_distance, lambda _: regular_distance, operand=None)

    @staticmethod
    @jax.jit
    def update_dice(dice: jnp.ndarray, is_valid: bool, distance: int, allow_oversized: bool = False) -> jnp.ndarray:
        """Consume one matching die (only the first match). Works with up to 4 dice."""

        def consume_one(dice):
            def scan_match_exact(carry, i):
                dice, usedDice = carry
                match_exact = (dice[i] == distance)
                should_consume = (~usedDice) & match_exact

                new_d = jax.lax.cond(
                    should_consume,
                    lambda _: dice.at[i].set(0),
                    lambda _: dice,
                    operand=None
                )
                new_used = usedDice | should_consume
                return (new_d, new_used), None

            def scan_match_oversized(carry, i):
                dice, consumed_dice = carry
                match_oversized = (dice[i] > distance)
                should_consume = (~consumed_dice) & match_oversized

                new_d = jax.lax.cond(
                    should_consume,
                    lambda _: dice.at[i].set(0),
                    lambda _: dice,
                    operand=None
                )
                new_used = consumed_dice | should_consume
                return (new_d, new_used), None

            def scan_match_fallback(carry, i):
                dice, consumed_dice = carry
                max_die_val = jnp.max(dice)
                match_fallback = (dice[i] == max_die_val) & (max_die_val < distance)
                should_consume = (~consumed_dice) & match_fallback

                new_d = jax.lax.cond(
                    should_consume,
                    lambda _: dice.at[i].set(0),
                    lambda _: dice,
                    operand=None
                )
                new_used = consumed_dice | should_consume
                return (new_d, new_used), None

            (new_dice, consumed_dice), _ = jax.lax.scan(scan_match_exact, (dice, False), jnp.arange(4))
            (new_dice, consumed_dice), _ = jax.lax.scan(scan_match_oversized,
                                                        (new_dice, consumed_dice | (~ allow_oversized)), jnp.arange(4))
            (new_dice, consumed_dice), _ = jax.lax.scan(scan_match_fallback, (new_dice, consumed_dice), jnp.arange(4))
            return new_dice

        return jax.lax.cond(is_valid, consume_one, lambda d: d, dice)

    @partial(jax.jit, static_argnums=(0,))
    def step_impl(self, state: BackgammonState, action: Tuple[int, int], key: jax.Array) -> Tuple[jnp.ndarray, BackgammonState, float, bool, dict, jax.Array]:
        """Perform a step in the environment, applying the action and returning the new state."""
        from_point, to_point = action
        board = state.board
        player = state.current_player
        player_idx = self.get_player_index(player)
        opponent_idx = 1 - player_idx

        # use the vectorized is_valid_move function with a single move
        is_valid = self.is_valid_move(state, jnp.array([from_point, to_point]))

        # execute the move if valid
        new_board = jax.lax.cond(
            is_valid,
            lambda _: self.execute_move(board, player_idx, opponent_idx, from_point, to_point),
            lambda _: board,
            operand=None
        )

        # calculate move distance
        distance = self.compute_distance(player, from_point, to_point)
        allow_oversized = (to_point == self.consts.HOME_INDEX)
        new_dice = JaxBackgammonEnv.update_dice(state.dice, is_valid, distance, allow_oversized)


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
        white_won = new_board[0, self.consts.HOME_INDEX] == self.consts.NUM_CHECKERS
        black_won = new_board[1, self.consts.HOME_INDEX] == self.consts.NUM_CHECKERS
        game_over = white_won | black_won

        # update game state
        new_state = BackgammonState(
            board=new_board,
            dice=next_dice,
            current_player=next_player,
            is_game_over=game_over,
            key = new_key
        )

        # Prepare return values
        obs = self._get_observation(new_state)
        reward = self._get_reward(state, new_state)
        done = self._get_done(new_state)
        info = self._get_info(new_state)

        return obs, new_state, reward, done, info, new_key

    def step(self, state: BackgammonState, action: jnp.ndarray) -> Tuple[jnp.ndarray, BackgammonState, float, bool, dict]:
        """Perform a step in the environment using a scalar action index."""
        # Map the scalar action index to a (from_point, to_point) pair
        move = tuple(self._action_pairs[action])
        obs, new_state, reward, done, info, new_key = self.step_impl(state, move, state.key)
        new_state = new_state._replace(key=new_key)
        return obs, new_state, reward, done, info

    def action_space(self) -> spaces.Discrete:
        """Return the discrete action space (scalar index into move list)."""
        return spaces.Discrete(self._action_pairs.shape[0])

    def observation_space(self) -> spaces.Box:
        """Return the observation space for the environment."""
        shape = (2 * 26 + 4 + 1 + 1,)  # = (58,)
        # current_player can be -1 or 1 → low must include -1
        return spaces.Box(low=-1, high=self.consts.NUM_CHECKERS, shape=shape, dtype=jnp.int32)

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
        player = state.current_player


        @jax.jit
        def _check_all_moves(state):
            return jax.vmap(lambda move: self.is_valid_move(state, move))(self.action_space)

        valid_mask = _check_all_moves(state)
        valid_moves_array = self.action_space[jnp.array(valid_mask)]
        return [tuple(map(int, move)) for move in valid_moves_array]

    def render(self, state: BackgammonState) -> Tuple[jnp.ndarray]:
        self.renderer.render(state)


class BackgammonRenderer(JAXGameRenderer):
    def __init__(self, env=None):
        super().__init__(env)
        asset_dir = Path("src/jaxatari/games/sprites/backgammon/")

        # All sprites should be RGBA (H, W, 4)
        self.sprite_board = jr.loadFrame(asset_dir / "backgammon_board.npy")
        self.sprite_checker_white = jr.loadFrame(asset_dir / "white_checker.npy")
        self.sprite_checker_black = jr.loadFrame(asset_dir / "black_checker.npy")

        self.point_positions = self._compute_point_positions()

    import jax.numpy as jnp

    def _compute_point_positions(self):
        """Return a JAX array of shape (26, 2) with (x, y) coords."""
        positions = []
        for i in range(13):
            x = 50 + i * 60
            y = 50
            positions.append((x, y))
        for i in range(13):
            x = 50 + (12 - i) * 60
            y = 400
            positions.append((x, y))
        return jnp.array(positions, dtype=jnp.int32)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        """
        Render the current Backgammon state to an RGBA (H, W, 4) frame.
        Fully JIT-compatible.
        """
        # Create a blank RGBA frame matching board dimensions
        frame = jr.create_initial_frame(
            self.sprite_board.shape[0],  # height
            self.sprite_board.shape[1],  # width
        )

        # Draw the board first
        frame = jr.render_at(frame, 0, 0, self.sprite_board)

        def draw_point(point_idx, fr):
            pos_x, pos_y = self.point_positions[point_idx]

            # White checkers (player 0 row)
            count_white = jnp.int32(state.board[0, point_idx])

            def draw_white(i, f):
                return jr.render_at(f, pos_x, pos_y + i * 20, self.sprite_checker_white)

            fr = jax.lax.fori_loop(0, count_white, draw_white, fr)

            # Black checkers (player 1 row)
            count_black = jnp.int32(state.board[1, point_idx])

            def draw_black(i, f):
                return jr.render_at(f, pos_x, pos_y - i * 20, self.sprite_checker_black)

            fr = jax.lax.fori_loop(0, count_black, draw_black, fr)

            return fr

        # Loop over all 26 points
        frame = jax.lax.fori_loop(0, 26, draw_point, frame)

        return frame


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
        from_display = "BAR" if from_point == env.consts.BAR_INDEX else str(from_point + 1)
        to_display = "HOME" if to_point == env.consts.HOME_INDEX else str(to_point + 1)
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

def run_game_without_input(key: jax.Array, max_steps=400):
    """Run the backgammon game without user input, using random moves."""
    env = JaxBackgammonEnv()
    obs, state = env.reset(key)
    renderer = BackgammonRenderer(env)
    env.renderer = renderer

    print(f"Initial roll: White {state.dice[0]}, Black {state.dice[1]}")
    print(f"{'White' if state.current_player == env.consts.WHITE else 'Black'} will start the game!")

    for i in range(max_steps):
        if state.is_game_over:
            print("\nGame Over!")
            break

        valid_moves = env.get_valid_moves(state)
        new_dice, new_key  = env.roll_dice(state.key) # change the RNG key to get better randomness
        state = state._replace(key = new_key)

        if len(valid_moves) == 0:
            state = state._replace(
                dice=new_dice,
                current_player=-state.current_player
            )
            env.render(state)
            continue

        # Choose random move
        new_key, move_key = jax.random.split(state.key)
        state = state._replace(key=new_key)
        idx = jax.random.randint(move_key, (), 0, len(valid_moves))
        action = valid_moves[idx]

        renderer.display(state)
        print(f"\nMove: {action[0] + 1} → {action[1] + 1}")
        obs, state, reward, done, info = env.step(state, action)

        if done:
            renderer.display(state)
            white_home = state.board[0, env.consts.HOME_INDEX]
            black_home = state.board[1, env.consts.HOME_INDEX]
            winner = "White" if white_home == env.consts.NUM_CHECKERS else "Black"
            print(f"\n==== Game Over! {winner} wins! ====")
            print(f"White home: {white_home}, Black home: {black_home}")
            break

    return state

def run_game_with_input(key: jax.Array, max_steps=200):
    """Run the backgammon game with keyboard input for moves."""
    env = JaxBackgammonEnv()
    # Create and attach the renderer
    renderer = BackgammonRenderer(env)
    env.renderer = renderer
    print('key')
    print(key)
    obs, state = env.reset(key)

    print("\n==== Welcome to JAX Backgammon ====")
    print("Points are numbered 1-24, with 1-6 on the white home board.")
    print("BAR refers to the bar, and HOME refers to moving pieces off the board.")

    print(f"Initial roll: White {state.dice[0]}, Black {state.dice[1]}")
    print(f"{'White' if state.current_player == env.consts.WHITE else 'Black'} will start the game!")
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
            new_dice, new_key = env.roll_dice(state.key)
            state = state._replace(
                dice=new_dice,
                current_player=-state.current_player,
                key = new_key
            )
            continue

        # Execute the move
        from_display = "BAR" if action[0] == env.consts.BAR_INDEX else str(action[0] + 1)
        to_display = "HOME" if action[1] == env.consts.HOME_INDEX else str(action[1] + 1)
        print(f"\nExecuting move: {from_display} → {to_display}")
        obs, state, reward, done, info = env.step(state, action)

        step_count += 1

        if done:
            renderer.display(state)
            winner = "White" if state.board[0, env.consts.HOME_INDEX] == env.consts.NUM_CHECKERS else "Black"
            print(f"\n==== Game Over! {winner} wins! ====")

    if step_count >= max_steps:
        print("\nMaximum steps reached. Game ended.")

    renderer.close()
    return state

def main():
    key = jax.random.PRNGKey(0)
    # run_game_without_input(key)
    run_game_without_input(key)


if __name__ == "__main__":
    main()
