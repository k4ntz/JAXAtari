# Third party imports
import chex
import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple, Tuple, Any, List, Optional
from jax import Array
import os
from pathlib import Path
from enum import IntEnum

# Project imports
from jaxatari.environment import JaxEnvironment
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as jr
import jaxatari.spaces as spaces

"""
Contributors: Ayush Bansal, Mahta Mollaeian, Anh Tuan Nguyen, Abdallah Siwar  

Game: JAX Backgammon

This module defines a JAX-accelerated backgammon environment for reinforcement learning and simulation.
It includes the environment class, state structures, move validation and execution logic, rendering, and user interaction.
"""

class BackgammonConstants(NamedTuple):
    """Constants for game Environment"""
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
    last_move: Tuple[int, int] = (-1, -1)
    last_dice: int = -1

class BackgammonInfo(NamedTuple):
    """Contains auxiliary information about the environment (e.g., timing or metadata)."""
    time: jnp.ndarray
    all_rewards: chex.Array

class BackgammonObservation(NamedTuple):
    """Complete backgammon observation structure for object-centric observations."""
    board: jnp.ndarray  # (2, 26) - full board state [white_checkers, black_checkers]
    dice: jnp.ndarray   # (4,) - available dice values
    current_player: jnp.ndarray  # (1,) - current player (-1 for black, 1 for white)
    is_game_over: jnp.ndarray    # (1,) - game over flag
    bar_counts: jnp.ndarray      # (2,) - checkers on bar [white, black]
    home_counts: jnp.ndarray     # (2,) - checkers borne off [white, black]

class GamePhase(IntEnum):
    """Phases of the interactive gameplay."""
    WAITING_FOR_ROLL = 0  # Waiting for space to roll dice
    SELECTING_CHECKER = 1  # Moving cursor to select a checker
    MOVING_CHECKER = 2     # Checker picked up, moving to destination
    TURN_COMPLETE = 3      # All moves done, waiting for space to end turn

class InteractiveState(NamedTuple):
    """State for interactive gameplay."""
    game_phase: int
    cursor_position: int  # Current cursor position (0-25)
    picked_checker_from: int  # Where we picked up a checker from (-1 if none)
    current_die_index: int  # Which die we're using (0-3)
    moves_made: jnp.ndarray  # Track which dice have been used

WHITE_PATH = [0,1,2,3,4,5, 6,7,8,9,10,11, 12,13,14,15,16,17, 18,19,20,21,22,23, 24, 25]
BLACK_PATH = [23,22,21,20,19,18, 17,16,15,14,13,12, 11,10,9,8,7,6, 5,4,3,2,1,0, 24, 25]

class JaxBackgammonEnv(JaxEnvironment[BackgammonState, jnp.ndarray, dict, BackgammonConstants]):
    """
    JAX-based backgammon environment supporting JIT compilation and vectorized operations.
    Provides functionality for state initialization, step transitions, valid move evaluation, and observation generation.
    """
    
    def __init__(self, consts: BackgammonConstants = None, reward_funcs: list[callable] = None):
        consts = consts or BackgammonConstants()
        super().__init__(consts)

        # Pre-compute all possible moves (indexed as a scalar in the framework)
        self._action_pairs = jnp.array([(i, j) for i in range(26) for j in range(26)], dtype=jnp.int32)
        
        # Reserve one extra action index for "roll dice"
        self._roll_action_index = self._action_pairs.shape[0]

        self.renderer = BackgammonRenderer(self)
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs

    @partial(jax.jit, static_argnums=(0,))
    def init_state(self, key) -> BackgammonState:
        board = jnp.zeros((2, 26), dtype=jnp.int32)
        # White (player 0)
        board = board.at[0, 0].set(2)  # point 24
        board = board.at[0, 11].set(5)  # point 13
        board = board.at[0, 16].set(3)  # point 8
        board = board.at[0, 18].set(5)  # point 6

        # Black (player 1)
        board = board.at[1, 23].set(2)  # point 1
        board = board.at[1, 12].set(5)  # point 12
        board = board.at[1, 7].set(3)  # point 17
        board = board.at[1, 5].set(5)  # point 19

        dice = jnp.zeros(4, dtype=jnp.int32)

        # The condition for the while loop
        def cond_fun(carry):
            white_roll, black_roll, key = carry
            return white_roll == black_roll
        
        # The code to be run in the while loop
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
        first_dice = jax.lax.cond(current_player == self.consts.WHITE, lambda _: white_roll, lambda _: black_roll, operand=None)
        second_dice = jax.lax.cond(current_player == self.consts.WHITE, lambda _: black_roll, lambda _: white_roll, operand=None)

        is_double = first_dice == second_dice
        dice = jax.lax.cond(
            is_double,
            lambda _: jnp.array([first_dice] * 4),
            lambda _: jnp.array([first_dice, second_dice, 0, 0]),
            operand=None
        )

        return BackgammonState(
            board=board,
            dice=dice,
            current_player=current_player,
            is_game_over=False,
            key=key
        )

    def reset(self, key: jax.random.PRNGKey = None) -> Tuple[jnp.ndarray, BackgammonState]:
        print(key)
        key = jax.lax.cond(
            key is None,
            lambda _: jax.random.PRNGKey(0),
            lambda _: key,
            operand=None
        )
        state = self.init_state(key)
        dice, key = self.roll_dice(key)
        state = state._replace(dice=dice, key=key)
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

        in_bounds = ((0 <= from_point) & (from_point <= 24) & 
                    (0 <= to_point) & (to_point <= self.consts.HOME_INDEX) & 
                    (to_point != self.consts.BAR_INDEX))
        
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
                def is_valid_entry(dice_val: int) -> bool:
                    expected_entry = jax.lax.select(player == self.consts.WHITE, dice_val - 1, 24 - dice_val)
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

                # Larger dice than needed is allowed only if no higher checkers
                larger_dice_available = jnp.any(state.dice > bearing_off_distance)

                # Checker must be present at the from_point
                has_piece = board[player_idx, from_point] > 0

                valid_bear = has_piece & (dice_match | ((~higher_checkers_exist) & larger_dice_available))

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
        """Check for bearing off using lax.cond instead of if statements."""
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

        # If hitting opponent, update opponent's bar and clear their point
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
                lambda _: self.consts.HOME_INDEX - from_point,
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
        """Consume one matching dice (only the first match). Works with up to 4 dice."""

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
                max_dice_val = jnp.max(dice)
                match_fallback = (dice[i] == max_dice_val) & (max_dice_val < distance)
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
    def step_impl(self, state: BackgammonState, action: Tuple[int, int], key: jax.Array):
        from_point, to_point = action
        board = state.board
        player = state.current_player
        player_idx = self.get_player_index(player)
        opponent_idx = 1 - player_idx

        is_valid = self.is_valid_move(state, jnp.array([from_point, to_point]))

        new_board = jax.lax.cond(
            is_valid,
            lambda _: self.execute_move(board, player_idx, opponent_idx, from_point, to_point),
            lambda _: board,
            operand=None,
        )

        distance = self.compute_distance(player, from_point, to_point)
        allow_oversized = (to_point == self.consts.HOME_INDEX)

        # Figure out which dice was used (first matching index, or -1)
        def find_dice(dice, distance):
            matches = jnp.where(dice == distance, 1, 0)
            idx = jnp.argmax(matches)  # gives first match
            return jnp.where(jnp.any(matches), idx, -1)

        used_dice = find_dice(state.dice, distance)

        new_dice = JaxBackgammonEnv.update_dice(state.dice, is_valid, distance, allow_oversized)

        all_dice_used = jnp.all(new_dice == 0)

        def next_turn(k):
            next_dice, new_key = JaxBackgammonEnv.roll_dice(k)
            return next_dice, -state.current_player, new_key

        def same_turn(k):
            return new_dice, state.current_player, k

        next_dice, next_player, new_key = jax.lax.cond(all_dice_used, next_turn, same_turn, key)

        white_won = new_board[0, self.consts.HOME_INDEX] == self.consts.NUM_CHECKERS
        black_won = new_board[1, self.consts.HOME_INDEX] == self.consts.NUM_CHECKERS
        game_over = white_won | black_won

        new_state = BackgammonState(
            board=new_board,
            dice=next_dice,
            current_player=next_player,
            is_game_over=game_over,
            key=new_key,
            last_move=(from_point, to_point),
            last_dice=used_dice
        )

        obs = self._get_observation(new_state)
        reward = self._get_reward(state, new_state)
        all_rewards = self._get_all_reward(state, new_state)
        done = self._get_done(new_state)
        info = self._get_info(new_state, all_rewards)

        return obs, new_state, reward, done, info, new_key

    def step(self, state: BackgammonState, action: jnp.ndarray) -> Tuple[jnp.ndarray, BackgammonState, float, bool, dict]:
        """Perform a step in the environment using a scalar action index."""
        def do_roll(_):
            dice, key = self.roll_dice(state.key)
            new_state = state._replace(dice=dice, key=key)
            return (
                self._get_observation(new_state),
                new_state,
                0.0,
                False,
                self._get_info(new_state)
            )

        def do_move(act):
            move = tuple(self._action_pairs[act])
            obs, new_state, reward, done, info, new_key = self.step_impl(state, move, state.key)
            new_state = new_state._replace(key=new_key)
            return obs, new_state, reward, done, info

        return jax.lax.cond(
            action == self._roll_action_index,
            do_roll,
            do_move,
            operand=action
        )

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: BackgammonObservation) -> jnp.ndarray:
        """Convert object-centric observation to flat array."""
        return jnp.concatenate([
            obs.board.flatten(),
            obs.dice.flatten(),
            obs.current_player.flatten(),
            obs.is_game_over.flatten(),
            obs.bar_counts.flatten(),  # 2 elements
            obs.home_counts.flatten()
        ]).astype(jnp.int32)

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(self, previous_state: BackgammonState, state: BackgammonState):
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array(
            [reward_func(previous_state, state) for reward_func in self.reward_funcs]
        )
        return rewards

    def image_space(self) -> spaces.Box:
        """Returns the image space for rendered frames."""
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.renderer.frame_height, self.renderer.frame_width, 3),
            dtype=jnp.uint8
        )

    def action_space(self) -> spaces.Discrete:
        """Return the discrete action space (scalar index into move list)."""
        return spaces.Discrete(self._action_pairs.shape[0] + 1)  # +1 for roll action

    def observation_space(self) -> spaces.Dict:
        """Return the observation space for the environment."""
        return spaces.Dict({
            "board": spaces.Box(
                low=0,
                high=self.consts.NUM_CHECKERS,
                shape=(2, 26),
                dtype=jnp.int32
            ),
            "dice": spaces.Box(
                low=0,
                high=6,
                shape=(4,),
                dtype=jnp.int32
            ),
            "current_player": spaces.Box(
                low=-1,  # BLACK = -1
                high=1,  # WHITE = 1
                shape=(1,),
                dtype=jnp.int32
            ),
            "is_game_over": spaces.Box(
                low=0,
                high=1,
                shape=(1,),
                dtype=jnp.int32
            ),
            "bar_counts": spaces.Box(
                low=0,
                high=self.consts.NUM_CHECKERS,
                shape=(2,),
                dtype=jnp.int32
            ),
            "home_counts": spaces.Box(
                low=0,
                high=self.consts.NUM_CHECKERS,
                shape=(2,),
                dtype=jnp.int32
            ),
        })

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: BackgammonState) -> BackgammonObservation:
        """Convert state to object-centric observation."""
        return BackgammonObservation(
            board=state.board,
            dice=state.dice,
            current_player=jnp.array([state.current_player], dtype=jnp.int32),
            is_game_over=jnp.array([jnp.where(state.is_game_over, 1, 0)], dtype=jnp.int32),
            bar_counts=jnp.array([state.board[0, 24], state.board[1, 24]], dtype=jnp.int32),
            home_counts=jnp.array([state.board[0, 25], state.board[1, 25]], dtype=jnp.int32)
        )

    @staticmethod
    @jax.jit
    def _get_info(state: BackgammonState, all_rewards: chex.Array = None) -> dict:
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
            return jax.vmap(lambda move: self.is_valid_move(state, move))(self._action_pairs)

        valid_mask = _check_all_moves(state)
        valid_moves_array = self._action_pairs[valid_mask]
        return [tuple(map(int, move)) for move in valid_moves_array]

    def render(self, state: BackgammonState) -> Tuple[jnp.ndarray]:
        return self.renderer.render(state)


class BackgammonRenderer(JAXGameRenderer):
    def __init__(self, env=None):
        super().__init__(env)

        self.frame_height = 210
        self.frame_width = 160
        self.color_background      = jnp.array([0, 0, 0], dtype=jnp.uint8)      # black background
        self.color_board           = jnp.array([0, 0, 0], dtype=jnp.uint8)      # black board
        self.color_triangle_light  = jnp.array([0, 0, 255], dtype=jnp.uint8)    # blue points
        self.color_triangle_dark   = jnp.array([0, 200, 0], dtype=jnp.uint8)    # green points
        self.color_white_checker   = jnp.array([255, 255, 255], dtype=jnp.uint8)  # white checkers
        self.color_black_checker   = jnp.array([255, 0, 0], dtype=jnp.uint8)     # red checkers
        self.color_border          = jnp.array([0, 200, 0], dtype=jnp.uint8)     # green bar
        
        self.top_margin_for_dice = 25  # pixels reserved for dice row

        # Geometry
        self.board_margin = 8
        self.triangle_length = 60
        self.triangle_thickness = 12
        self.bar_thickness = 14
        self.checker_width = 8       # rectangular checkers
        self.checker_height = 4      # smaller height
        self.checker_stack_offset = 5  # vertical stacking distance
        self.triangle_positions = self._compute_triangle_positions()
        self.bar_y = self.top_margin_for_dice + self.frame_height // 2 - self.bar_thickness // 2 - 10
        self.bar_x = self.board_margin
        self.bar_width = self.frame_width - 2 * self.board_margin

    def _compute_triangle_positions(self):
        """Return an (24,2) int32 array of (x,y) top-left coords for each triangle.
        Layout matches traditional backgammon:
          - Top row: points 13-18 (left), 19-24 (right)  
          - Bottom row: points 12-7 (left), 6-1 (right)
        """
        positions: List[Tuple[int, int]] = []

        left_x = self.board_margin
        right_x = self.frame_width - self.board_margin - self.triangle_length

        # Top row - left side (points 13-18) - blue, green, blue, green, blue, green
        for i in range(6):
            y = self.top_margin_for_dice + self.board_margin + i * self.triangle_thickness
            positions.append((left_x, y))

        # Top row - right side (points 19-24) - green, blue, green, blue, green, blue  
        for i in range(6):
            y = self.top_margin_for_dice + self.board_margin + i * self.triangle_thickness
            positions.append((right_x, y))

        # Bottom row - right side (points 6-1) - blue, green, blue, green, blue, green
        for i in range(6):
            y = self.frame_height - self.board_margin - (i + 1) * self.triangle_thickness
            positions.append((right_x, y))

        # Bottom row - left side (points 12-7) - green, blue, green, blue, green, blue
        for i in range(6):
            y = self.frame_height - self.board_margin - (i + 1) * self.triangle_thickness
            positions.append((left_x, y))

        return jnp.array(positions, dtype=jnp.int32)

    @partial(jax.jit, static_argnums=(0,))
    def _draw_rectangle(self, frame, x, y, width, height, color):
        yy, xx = jnp.mgrid[0:self.frame_height, 0:self.frame_width]
        mask = (xx >= x) & (xx < (x + width)) & (yy >= y) & (yy < (y + height))
        return jnp.where(mask[..., None], color, frame)

    @partial(jax.jit, static_argnums=(0,))
    def _draw_triangle(self, frame, x, y, length, thickness, color, point_right=True):
        """
        Draw an isosceles triangle whose rectangular bounding box is:
           x <= xx < x+length,   y <= yy < y+thickness
        If point_right==True, the triangle's tip is at (x+length, center_y) (points right).
        If point_right==False, the tip is at (x, center_y) (points left).
        """
        yy, xx = jnp.mgrid[0:self.frame_height, 0:self.frame_width]
        xx_f = xx.astype(jnp.float32)
        yy_f = yy.astype(jnp.float32)
        x_f = jnp.asarray(x, dtype=jnp.float32)
        y_f = jnp.asarray(y, dtype=jnp.float32)
        length_f = jnp.asarray(length, dtype=jnp.float32)
        thickness_f = jnp.asarray(thickness, dtype=jnp.float32)

        center_y = y_f + thickness_f / 2.0

        t = jax.lax.select(point_right,(xx_f - x_f) / length_f,(x_f + length_f - xx_f) / length_f)
        half_width = (1.0 - t) * (thickness_f / 2.0)
        in_bbox = (xx >= x) & (xx < (x + length)) & (yy >= y) & (yy < (y + thickness))
        valid_t = (t >= 0.0) & (t <= 1.0)
        within_profile = jnp.abs(yy_f - center_y) <= half_width
        mask = in_bbox & valid_t & within_profile

        return jnp.where(mask[..., None], color, frame)

    @partial(jax.jit, static_argnums=(0,))
    def _draw_circle(self, frame, cx, cy, radius, color):
        yy, xx = jnp.mgrid[0:self.frame_height, 0:self.frame_width]
        cx_f = jnp.asarray(cx, dtype=jnp.float32)
        cy_f = jnp.asarray(cy, dtype=jnp.float32)
        xx_f = xx.astype(jnp.float32)
        yy_f = yy.astype(jnp.float32)
        mask = (xx_f - cx_f) ** 2 + (yy_f - cy_f) ** 2 <= (radius ** 2)
        return jnp.where(mask[..., None], color, frame)

    @partial(jax.jit, static_argnums=(0,))
    def _draw_board_outline(self, frame):
        frame = self._draw_rectangle(frame, 0, 0, self.frame_width, self.frame_height, self.color_background)
        board_x = self.board_margin - 6
        # shift board *down* by the dice margin
        board_y = self.top_margin_for_dice + self.board_margin - 6
        board_w = self.frame_width - 2 * (self.board_margin - 6)
        # reduce height accordingly
        board_h = self.frame_height - self.top_margin_for_dice - 2 * (self.board_margin - 6)

        frame = self._draw_rectangle(frame, board_x, board_y, board_w, board_h, self.color_board)
        frame = self._draw_rectangle(frame, self.bar_x, self.bar_y, self.bar_width, self.bar_thickness, self.color_border)
        return frame

    @partial(jax.jit, static_argnums=(0,))
    def _draw_triangles(self, frame):
        left_x = self.board_margin

        def draw_triangle_at_index(i, fr):
            pos = self.triangle_positions[i]
            x = pos[0]; y = pos[1]

            group  = i // 6        # 0: TL, 1: TR, 2: BR, 3: BL
            within = i % 6         # 0..5 inside each group

            # Left groups (0,3) start with light; right groups (1,2) start with dark
            start_light = jnp.logical_or(group == 0, group == 3)
            use_light   = jnp.where(start_light, (within % 2 == 0), (within % 2 == 1))

            color = jax.lax.select(use_light, self.color_triangle_light, self.color_triangle_dark)
            point_right = (x == left_x)  # left columns point right, right columns point left
            return self._draw_triangle(fr, x, y, self.triangle_length, self.triangle_thickness, color, point_right)

        return jax.lax.fori_loop(0, 24, draw_triangle_at_index, frame)

    @partial(jax.jit, static_argnums=(0,))
    def _draw_checkers_on_point(self, frame, point_idx, white_count, black_count):
        """
        Draw Atari-like stacks: tiny rectangles, 2-high, extending HORIZONTALLY
        along the triangle toward its tip. Top halves attach to the top edge; bottom
        halves attach to the bottom edge.
        """
        pos = self.triangle_positions[point_idx]
        x = pos[0]; y = pos[1]

        # left columns have x == board_margin → stacks go left→right
        left_x = self.board_margin
        is_left_column = (x == left_x)
        is_top_half    = (point_idx < 12)

        # horizontal direction (+1 for left columns, -1 for right columns)
        dir_sign = jnp.where(is_left_column, 1, -1)

        # starting x near the base of the triangle (not the tip)
        base_x = jnp.where(
            is_left_column,
            x + 2,                                    # left columns start near left edge
            x + self.triangle_length - self.checker_width - 2  # right columns start near right edge
        )

        # two rows: row 0 and row 1 (2-high); attach to top or bottom band
        # vertical origin for the two rows
        row0_y = jnp.where(
            is_top_half,
            y + 1,                                    # top halves stick to top edge
            y + self.triangle_thickness - 2*self.checker_height - 1  # bottom halves stick to bottom edge
        )
        row1_y = row0_y + (self.checker_height + 1)

        # helper to draw 'count' rectangles in 2-high columns, marching HORIZONTALLY
        def draw_stack(fr, count, color):
            def draw_single(i, f):
                col = i // 2               # column index (0,1,2,...) along the triangle
                row = i & 1                # 0 or 1 (which row of the 2-high stack)

                # x marches toward the triangle tip
                cx = base_x + dir_sign * (col * (self.checker_width + 1))
                cy = jnp.where(row == 0, row0_y, row1_y)

                return self._draw_rectangle(f, cx, cy, self.checker_width, self.checker_height, color)
            return jax.lax.fori_loop(0, count, draw_single, fr)

        # draw white then black on top (matches your layering approach)
        frame = draw_stack(frame, white_count, self.color_white_checker)
        frame = draw_stack(frame, black_count, self.color_black_checker)
        return frame

    @partial(jax.jit, static_argnums=(0,))
    def _draw_bar_checkers(self, frame, white_count, black_count):
        """Draw checkers on the horizontal bar in 2x2 stacking pattern."""
        cx = self.frame_width // 2
        cy = self.frame_height // 2

        def draw_stack(fr, count, color, x_offset):
            def draw_single(i, f):
                row = i // 2
                col = i % 2
                
                checker_x = cx + x_offset + col * (self.checker_width + 1)
                checker_y = cy - self.checker_height//2 + row * self.checker_stack_offset
                
                return self._draw_rectangle(f, checker_x, checker_y,
                                          self.checker_width, self.checker_height, color)
            return jax.lax.fori_loop(0, count, draw_single, fr)

        # White to the left, black to the right
        frame = draw_stack(frame, white_count, self.color_white_checker, -25)
        frame = draw_stack(frame, black_count, self.color_black_checker, 10)
        return frame

    @partial(jax.jit, static_argnums=(0,))
    def _draw_home_checkers(self, frame, white_count, black_count):
        """Draw home stacks near bottom-center in 2x2 pattern."""
        cy = self.frame_height - 18
        cx_center = self.frame_width // 2

        def draw_stack(fr, count, color, x_offset):
            def draw_single(i, f):
                row = i // 2
                col = i % 2
                
                checker_x = cx_center + x_offset + col * (self.checker_width + 1)
                checker_y = cy - self.checker_height//2 + row * self.checker_stack_offset
                
                return self._draw_rectangle(f, checker_x, checker_y,
                                          self.checker_width, self.checker_height, color)
            return jax.lax.fori_loop(0, count, draw_single, fr)

        frame = draw_stack(frame, white_count, self.color_white_checker, -30)
        frame = draw_stack(frame, black_count, self.color_black_checker, 10)
        return frame

    @partial(jax.jit, static_argnums=(0,))
    def _draw_dice(self, frame, dice):
        """Draw dice above the board."""
        dice_size = 12
        total_width = 4 * (dice_size + 3)
        start_x = self.frame_width // 2 - total_width // 2 + 12
        dice_y = self.board_margin

        def draw_single(i, fr):
            val = dice[i]
            dx = start_x + i * (dice_size + 3)
            
            def draw_val(_):
                fr2 = self._draw_rectangle(fr, dx, dice_y, dice_size, dice_size, jnp.array([240, 240, 240], dtype=jnp.uint8))
                center_x = dx + dice_size // 2
                center_y = dice_y + dice_size // 2
                pip = jnp.array([0, 0, 0], dtype=jnp.uint8)
                r = 1
                
                def p1(_): return self._draw_circle(fr2, center_x, center_y, r, pip)
                def p2(_):
                    fr3 = self._draw_circle(fr2, center_x - 3, center_y - 3, r, pip)
                    return self._draw_circle(fr3, center_x + 3, center_y + 3, r, pip)
                def p3(_):
                    fr3 = self._draw_circle(fr2, center_x - 3, center_y - 3, r, pip)
                    fr3 = self._draw_circle(fr3, center_x, center_y, r, pip)
                    return self._draw_circle(fr3, center_x + 3, center_y + 3, r, pip)
                def p4(_):
                    fr4 = self._draw_circle(fr2, center_x - 3, center_y - 3, r, pip)
                    fr4 = self._draw_circle(fr4, center_x + 3, center_y - 3, r, pip)
                    fr4 = self._draw_circle(fr4, center_x - 3, center_y + 3, r, pip)
                    return self._draw_circle(fr4, center_x + 3, center_y + 3, r, pip)
                def p5(_):
                    fr5 = p4(None)
                    return self._draw_circle(fr5, center_x, center_y, r, pip)
                def p6(_):
                    fr6 = self._draw_circle(fr2, center_x - 4, center_y - 3, r, pip)
                    fr6 = self._draw_circle(fr6, center_x + 4, center_y - 3, r, pip)
                    fr6 = self._draw_circle(fr6, center_x - 4, center_y + 3, r, pip)
                    fr6 = self._draw_circle(fr6, center_x + 4, center_y + 3, r, pip)
                    return fr6
                    
                funcs = [p1, p2, p3, p4, p5, p6]
                return jax.lax.switch(jnp.clip(val - 1, 0, 5), funcs, operand=None)
            
            return jax.lax.cond(val > 0, draw_val, lambda _: fr, operand=None)
        
        return jax.lax.fori_loop(0, 4, draw_single, frame)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: BackgammonState):
        frame = jnp.zeros((self.frame_height, self.frame_width, 3), dtype=jnp.uint8)

        frame = self._draw_board_outline(frame)
        frame = self._draw_triangles(frame)

        def draw_point_checkers(point_idx, fr):
            white_count = jnp.maximum(state.board[0, point_idx], 0)
            black_count = jnp.maximum(state.board[1, point_idx], 0)
            return self._draw_checkers_on_point(fr, point_idx, white_count, black_count)

        frame = jax.lax.fori_loop(0, 24, draw_point_checkers, frame)

        # Bar and home stacks
        frame = self._draw_bar_checkers(frame, jnp.maximum(state.board[0, 24], 0), jnp.maximum(state.board[1, 24], 0))
        frame = self._draw_home_checkers(frame, jnp.maximum(state.board[0, 25], 0), jnp.maximum(state.board[1, 25], 0))

        # Dice
        frame = self._draw_dice(frame, state.dice)

        return frame


class BackgammonInteractiveWrapper:
    """
    Wrapper around JaxBackgammonEnv to handle interactive gameplay with cursor.
    This handles the state machine for picking up and dropping checkers.
    """
    
    def __init__(self, env: JaxBackgammonEnv):
        self.env = env
        self.reset_interactive_state()
        
    def reset_interactive_state(self):
        """Reset the interactive state machine."""
        self.game_phase = GamePhase.SELECTING_CHECKER
        self.cursor_position = 0
        self.picked_checker_from = -1
        self.current_die_index = 0
        self.moves_made = jnp.zeros(4, dtype=jnp.bool_)
        
    def reset(self, key=None):
        """Reset both the environment and interactive state."""
        obs, state = self.env.reset(key)
        self.reset_interactive_state()
        self.game_phase = GamePhase.SELECTING_CHECKER
        return obs, state
        
    def handle_input(self, state: BackgammonState, action: str) -> Tuple[BackgammonState, bool]:
        """
        Handle keyboard input and update state accordingly.
        
        Args:
            state: Current game state
            action: One of 'left', 'right', 'space'
            
        Returns:
            Updated state and whether the action was valid
        """
        
        if action == 'left':
            return self._handle_cursor_left(state)
        elif action == 'right':
            return self._handle_cursor_right(state)
        elif action == 'space':
            return self._handle_space(state)
        else:
            return state, False
            
    def _move_cursor(self, state: BackgammonState, direction: int):
        """Move cursor forward (right) or backward (left) along the correct player path."""
        path = WHITE_PATH if state.current_player == 1 else BLACK_PATH
        idx = path.index(self.cursor_position)
        new_idx = (idx + direction) % len(path)
        self.cursor_position = path[new_idx]
        return state, True

    def _handle_cursor_left(self, state: BackgammonState):
        if self.game_phase in [GamePhase.SELECTING_CHECKER, GamePhase.MOVING_CHECKER]:
            return self._move_cursor(state, -1)
        return state, False

    def _handle_cursor_right(self, state: BackgammonState):
        if self.game_phase in [GamePhase.SELECTING_CHECKER, GamePhase.MOVING_CHECKER]:
            return self._move_cursor(state, +1)
        return state, False
        
    def _handle_space(self, state: BackgammonState) -> Tuple[BackgammonState, bool]:
        """Handle space bar press based on current game phase."""
        
        if self.game_phase == GamePhase.WAITING_FOR_ROLL:
            # Roll dice to start turn
            dice, key = self.env.roll_dice(state.key)
            new_state = state._replace(dice=dice, key=key)
            self.game_phase = GamePhase.SELECTING_CHECKER
            self.moves_made = jnp.zeros(4, dtype=jnp.bool_)
            return new_state, True
            
        elif self.game_phase == GamePhase.SELECTING_CHECKER:
            # Check if we need to end turn (no valid moves or all dice used)
            if self._all_dice_used(state) or not self._has_valid_moves(state):
                # End turn and switch players
                dice, key = self.env.roll_dice(state.key)
                new_state = state._replace(
                    dice=dice,
                    current_player=-state.current_player,
                    key=key
                )
                self.moves_made = jnp.zeros(4, dtype=jnp.bool_)
                return new_state, True
            
            # Try to pick up a checker
            player_idx = self.env.get_player_index(state.current_player)
            
            # Check if there's a checker at cursor position
            if self.cursor_position < 26 and state.board[player_idx, self.cursor_position] > 0:
                # Check if this position has valid moves
                valid_moves = self._get_valid_moves_from_position(state, self.cursor_position)
                
                if len(valid_moves) > 0:
                    self.picked_checker_from = self.cursor_position
                    self.game_phase = GamePhase.MOVING_CHECKER
                    return state, True
                    
        elif self.game_phase == GamePhase.MOVING_CHECKER:
            # Try to drop the checker
            move = (self.picked_checker_from, self.cursor_position)
            
            # Check if this is a valid move
            if self.env.is_valid_move(state, move):
                # Execute the move
                obs, new_state, reward, done, info, key = self.env.step_impl(
                    state, move, state.key
                )
                new_state = new_state._replace(key=key)
                
                # Mark die as used
                distance = self._calculate_move_distance(state, move)
                self._mark_die_used(state, distance)
                
                # Return to selecting phase
                self.game_phase = GamePhase.SELECTING_CHECKER
                self.picked_checker_from = -1
                    
                return new_state, True
            else:
                # Invalid drop location - return to selecting
                self.game_phase = GamePhase.SELECTING_CHECKER
                self.picked_checker_from = -1
                return state, False
                
        return state, False
        
    def _get_valid_moves_from_position(self, state: BackgammonState, from_pos: int) -> list:
        """Get all valid moves from a given position."""
        valid_moves = []
        for to_pos in range(26):
            if self.env.is_valid_move(state, (from_pos, to_pos)):
                valid_moves.append(to_pos)
        return valid_moves
        
    def _calculate_move_distance(self, state: BackgammonState, move: Tuple[int, int]) -> int:
        """Calculate the distance of a move."""
        return self.env.compute_distance(state.current_player, move[0], move[1])
        
    def _mark_die_used(self, state: BackgammonState, distance: int):
        """Mark a die as used based on the distance moved."""
        for i in range(4):
            if not self.moves_made[i] and state.dice[i] == distance:
                self.moves_made = self.moves_made.at[i].set(True)
                break
                
    def _all_dice_used(self, state: BackgammonState) -> bool:
        """Check if all available dice have been used."""
        available_dice = jnp.sum(state.dice > 0)
        used_dice = jnp.sum(self.moves_made)
        return used_dice >= available_dice
        
    def _has_valid_moves(self, state: BackgammonState) -> bool:
        """Check if the current player has any valid moves."""
        return len(self.env.get_valid_moves(state)) > 0
        
    def render(self, state: BackgammonState) -> jnp.ndarray:
        """Render the game with cursor overlay."""
        # Get base render from environment
        frame = self.env.render(state)
        
        # Add cursor visualization
        frame = self._draw_cursor(frame, state)
        
        # Add phase indicator
        frame = self._draw_phase_indicator(frame)
        
        return frame
        
    def _draw_cursor(self, frame: jnp.ndarray, state: BackgammonState) -> jnp.ndarray:
        """Draw cursor on the board."""
        if self.game_phase in [GamePhase.SELECTING_CHECKER, GamePhase.MOVING_CHECKER]:
            # Calculate cursor position on screen
            if self.cursor_position < 24:
                # Cursor on a point
                pos = self.env.renderer.triangle_positions[self.cursor_position]
                cursor_x = int(pos[0] + 5)
                cursor_y = int(pos[1] - 5 if self.cursor_position < 12 else pos[1] + 15)
            elif self.cursor_position == 24:
                # Cursor on bar
                cursor_x = frame.shape[1] // 2
                cursor_y = frame.shape[0] // 2
            else:
                # Cursor on home
                cursor_x = frame.shape[1] // 2
                cursor_y = frame.shape[0] - 25
                
            # Draw cursor (yellow square)
            cursor_color = jnp.array([255, 255, 0], dtype=jnp.uint8)
            frame = self._draw_cursor_box(frame, cursor_x, cursor_y, cursor_color)
            
            # If holding a checker, show it following the cursor
            if self.game_phase == GamePhase.MOVING_CHECKER:
                player_idx = self.env.get_player_index(state.current_player)
                checker_color = self.env.renderer.color_white_checker if player_idx == 0 else self.env.renderer.color_black_checker
                frame = self._draw_floating_checker(frame, cursor_x, cursor_y - 10, checker_color)
                
        return frame
        
    def _draw_cursor_box(self, frame: jnp.ndarray, x: int, y: int, color: jnp.ndarray) -> jnp.ndarray:
        """Draw a cursor box at the given position."""
        # Draw a hollow square for the cursor
        size = 10
        thickness = 2
        
        # Top and bottom edges
        frame = self.env.renderer._draw_rectangle(frame, x-size//2, y-size//2, size, thickness, color)
        frame = self.env.renderer._draw_rectangle(frame, x-size//2, y+size//2-thickness, size, thickness, color)
        
        # Left and right edges
        frame = self.env.renderer._draw_rectangle(frame, x-size//2, y-size//2, thickness, size, color)
        frame = self.env.renderer._draw_rectangle(frame, x+size//2-thickness, y-size//2, thickness, size, color)
        
        return frame
        
    def _draw_floating_checker(self, frame: jnp.ndarray, x: int, y: int, color: jnp.ndarray) -> jnp.ndarray:
        """Draw a checker floating above the cursor."""
        return self.env.renderer._draw_rectangle(
            frame, x-4, y-2, 
            self.env.renderer.checker_width, 
            self.env.renderer.checker_height, 
            color
        )
        
    def _draw_phase_indicator(self, frame: jnp.ndarray) -> jnp.ndarray:
        """Draw text indicating current game phase."""
        # This is simplified - in a real implementation you'd want proper text rendering
        # For now, we'll just use colored indicators
        indicator_x = 5
        indicator_y = 5
        
        if self.game_phase == GamePhase.WAITING_FOR_ROLL:
            color = jnp.array([255, 0, 0], dtype=jnp.uint8)  # Red
        elif self.game_phase == GamePhase.SELECTING_CHECKER:
            color = jnp.array([0, 255, 0], dtype=jnp.uint8)  # Green
        elif self.game_phase == GamePhase.MOVING_CHECKER:
            color = jnp.array([255, 255, 0], dtype=jnp.uint8)  # Yellow
        else:  # TURN_COMPLETE
            color = jnp.array([0, 0, 255], dtype=jnp.uint8)  # Blue
            
        return self.env.renderer._draw_rectangle(frame, indicator_x, indicator_y, 5, 5, color)


# Main game execution
def play_interactive_game():
    """Main function to run the interactive backgammon game."""
    import pygame
    import numpy as np
    
    # Initialize Pygame
    pygame.init()
    
    # Set up display
    SCALE = 4  # Scale factor for the display
    env = JaxBackgammonEnv()
    wrapper = BackgammonInteractiveWrapper(env)
    
    WINDOW_WIDTH = env.renderer.frame_width * SCALE
    WINDOW_HEIGHT = env.renderer.frame_height * SCALE
    
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("JAX Backgammon - Interactive")
    
    # Initialize game
    key = jax.random.PRNGKey(42)
    obs, state = wrapper.reset(key)
    
    # Game loop
    running = True
    clock = pygame.time.Clock()
    
    # Font for displaying text
    font = pygame.font.Font(None, 24)
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                valid_move = False
                
                if event.key == pygame.K_LEFT:
                    state, valid_move = wrapper.handle_input(state, 'left')
                elif event.key == pygame.K_RIGHT:
                    state, valid_move = wrapper.handle_input(state, 'right')
                elif event.key == pygame.K_SPACE:
                    state, valid_move = wrapper.handle_input(state, 'space')
                elif event.key == pygame.K_ESCAPE:
                    running = False
                    
                # Debug output
                if valid_move:
                    print(f"Phase: {wrapper.game_phase}, Cursor: {wrapper.cursor_position}, Player: {state.current_player}")
                    print(f"Dice: {state.dice}")
        
        # Render the game
        frame = wrapper.render(state)
        
        # Convert JAX array to numpy then to pygame surface
        frame_np = np.array(frame)
        frame_surface = pygame.surfarray.make_surface(frame_np.transpose((1, 0, 2)))
        frame_scaled = pygame.transform.scale(frame_surface, (WINDOW_WIDTH, WINDOW_HEIGHT))
        
        # Draw to screen
        screen.blit(frame_scaled, (0, 0))
        
        # Add text overlays
        current_player = "White" if state.current_player == 1 else "Black"
        player_text = font.render(f"Current: {current_player}", True, (255, 255, 255))
        screen.blit(player_text, (10, WINDOW_HEIGHT - 30))
        
        # Show game phase
        phase_names = {
            GamePhase.WAITING_FOR_ROLL: "Press SPACE to roll",
            GamePhase.SELECTING_CHECKER: "Select a checker",
            GamePhase.MOVING_CHECKER: "Move to destination",
            GamePhase.TURN_COMPLETE: "Press SPACE to end turn"
        }
        phase_text = font.render(phase_names.get(wrapper.game_phase, ""), True, (255, 255, 255))
        screen.blit(phase_text, (10, WINDOW_HEIGHT - 60))
        
        # Show dice values
        dice_str = f"Dice: {' '.join(str(d) for d in state.dice if d > 0)}"
        dice_text = font.render(dice_str, True, (255, 255, 255))
        screen.blit(dice_text, (10, WINDOW_HEIGHT - 90))
        
        # Check for game over
        if state.is_game_over:
            winner = "White" if state.board[0, 25] == 15 else "Black"
            winner_text = font.render(f"Game Over! {winner} wins!", True, (255, 255, 0))
            text_rect = winner_text.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2))
            screen.blit(winner_text, text_rect)
        
        pygame.display.flip()
        clock.tick(30)
    
    pygame.quit()


# Alternative: Simple test without pygame
def test_game_without_pygame():
    """Test the game logic without pygame (console-based)."""
    env = JaxBackgammonEnv()
    wrapper = BackgammonInteractiveWrapper(env)
    
    key = jax.random.PRNGKey(42)
    obs, state = wrapper.reset(key)
    
    print("Backgammon Game Started!")
    print("Commands: 'l' (left), 'r' (right), 's' (space), 'q' (quit)")
    print("-" * 50)
    
    while not state.is_game_over:
        # Display current state
        player = "White" if state.current_player == 1 else "Black"
        print(f"\nCurrent Player: {player}")
        print(f"Dice: {[d for d in state.dice if d > 0]}")
        print(f"Cursor Position: {wrapper.cursor_position}")
        print(f"Phase: {wrapper.game_phase}")
        
        # Get input
        cmd = input("Command: ").strip().lower()
        
        if cmd == 'q':
            break
        elif cmd == 'l':
            state, valid = wrapper.handle_input(state, 'left')
        elif cmd == 'r':
            state, valid = wrapper.handle_input(state, 'right')
        elif cmd == 's':
            state, valid = wrapper.handle_input(state, 'space')
        else:
            print("Invalid command!")
            continue
            
        # Show board state (simplified)
        print("\nBoard State:")
        print("White:", [(i, int(state.board[0, i])) for i in range(26) if state.board[0, i] > 0])
        print("Black:", [(i, int(state.board[1, i])) for i in range(26) if state.board[1, i] > 0])
    
    if state.is_game_over:
        winner = "White" if state.board[0, 25] == 15 else "Black"
        print(f"\nGame Over! {winner} wins!")


if __name__ == "__main__":
    # Choose which version to run
    try:
        import pygame
        play_interactive_game()
    except ImportError:
        print("Pygame not installed. Running console version...")
        test_game_without_pygame()