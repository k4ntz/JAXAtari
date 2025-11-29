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
from jaxatari.environment import JaxEnvironment, JAXAtariAction
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
    DOUBLING_CUBE = 1  # 1x by default; set to 2, 4, 8... if you add a doubling cube UI


class BackgammonState(NamedTuple):
    """Represents the complete state of a backgammon game."""
    board: jnp.ndarray  # (2, 26)
    dice: jnp.ndarray  # (4,)
    current_player: int
    is_game_over: bool
    key: jax.random.PRNGKey
    last_move: Tuple[int, int] = (-1, -1)
    last_dice: int = -1
    cursor_position: int = 0
    picked_checker_from: int = -1
    game_phase: int = 0  # 0=WAITING_FOR_ROLL, 1=SELECTING_CHECKER, 2=MOVING_CHECKER
    last_action: int = JAXAtariAction.NOOP  # Store last action for keyup handling
    await_keyup: bool = False
    last_valid_drop: int = -1  # -1 means no persistent highlight
    picked_bar_side: int = -1  # 24 (left), 26 (right), or -1 if not from bar



class BackgammonInfo(NamedTuple):
    """Contains auxiliary information about the environment (e.g., timing or metadata)."""
    player: jnp.ndarray
    dice: jnp.ndarray
    all_rewards: chex.Array


class BackgammonObservation(NamedTuple):
    """Complete backgammon observation structure for object-centric observations."""
    board: jnp.ndarray  # (2, 26) - full board state [white_checkers, black_checkers]
    dice: jnp.ndarray  # (4,) - available dice values
    current_player: jnp.ndarray  # (1,) - current player (-1 for black, 1 for white)
    is_game_over: jnp.ndarray  # (1,) - game over flag
    bar_counts: jnp.ndarray  # (2,) - checkers on bar [white, black]
    home_counts: jnp.ndarray  # (2,) - checkers borne off [white, black]


class GamePhase(IntEnum):
    """Phases of the interactive gameplay."""
    WAITING_FOR_ROLL = 0  # Waiting for space to roll dice
    SELECTING_CHECKER = 1  # Moving cursor to select a checker
    MOVING_CHECKER = 2  # Checker picked up, moving to destination
    TURN_COMPLETE = 3  # All moves done, waiting for space to end turn


class InteractiveState(NamedTuple):
    """State for interactive gameplay."""
    game_phase: int
    cursor_position: int  # Current cursor position (0-25)
    picked_checker_from: int  # Where we picked up a checker from (-1 if none)
    current_die_index: int  # Which die we're using (0-3)
    moves_made: jnp.ndarray  # Track which dice have been used


WHITE_PATH = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
BLACK_PATH = [23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 24, 25]


class JaxBackgammonEnv(JaxEnvironment[BackgammonState, jnp.ndarray, dict, BackgammonConstants]):
    """
    JAX-based backgammon environment supporting JIT compilation and vectorized operations.
    Provides functionality for state initialization, step transitions, valid move evaluation, and observation generation.
    """

    def __init__(self, consts: BackgammonConstants = None, reward_funcs: list[callable] = None):
        consts = consts or BackgammonConstants()
        super().__init__(consts)

        # Pre-compute all possible moves (from_point, to_point) for points 0..25
        # Shape: (676, 2) = 26×26 combinations
        # This will be used for vectorized move validation in Phase 3 (sequence generation)
        NUM_ACTION_PAIRS = 26 * 26
        self._action_pairs = jnp.array([(i, j) for i in range(26) for j in range(26)], dtype=jnp.int32)
        assert self._action_pairs.shape == (NUM_ACTION_PAIRS, 2), \
            f"Action pairs shape mismatch: expected ({NUM_ACTION_PAIRS}, 2), got {self._action_pairs.shape}"

        # Special action indices for interactive play
        self._roll_action_index = self._action_pairs.shape[0]

        self.renderer = BackgammonRenderer(self)
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs

        # Define action set for jaxatari compatibility (interactive mode)
        # NOTE: For AI agent training (PPO/PQN), you can either:
        #   1. Use these interactive actions (LEFT/RIGHT/FIRE/NOOP) to simulate cursor movement
        #   2. Or add a direct move API: step_move(state, move_index) where move_index ∈ [0, 675]
        # Current implementation: Interactive mode (ALE-compatible)
        self.action_set = [
            JAXAtariAction.LEFT,  # Move cursor left
            JAXAtariAction.RIGHT,  # Move cursor right
            JAXAtariAction.FIRE,  # Space (select/drop/roll)
            JAXAtariAction.NOOP  # No-op (do nothing)
        ]

    @partial(jax.jit, static_argnums=(0,))
    def init_state(self, key) -> BackgammonState:
        board = jnp.zeros((2, 26), dtype=jnp.int32)
        # White (player 0)
        board = board.at[0, 0].set(2)   # 24
        board = board.at[0, 11].set(5)  # 13
        board = board.at[0, 16].set(3)  # 8
        board = board.at[0, 18].set(5)  # 6
        # Black (player 1)
        board = board.at[1, 23].set(2)  # 1
        board = board.at[1, 12].set(5)  # 12
        board = board.at[1, 7].set(3)   # 17
        board = board.at[1, 5].set(5)   # 19

        # roll until not equal
        def cond_fun(c):
            w, b, k = c
            return w == b

        def body_fun(c):
            _, _, k = c
            k, k1, k2 = jax.random.split(k, 3)
            w = jax.random.randint(k1, (), 1, 7)
            b = jax.random.randint(k2, (), 1, 7)
            return (w, b, k)

        key, k1, k2 = jax.random.split(key, 3)
        w0 = jax.random.randint(k1, (), 1, 7)
        b0 = jax.random.randint(k2, (), 1, 7)
        w, b, key = jax.lax.while_loop(cond_fun, body_fun, (w0, b0, key))

        # who starts?
        current_player = jax.lax.cond(
            w > b, lambda _: self.consts.WHITE, lambda _: self.consts.BLACK, operand=None
        )

        # starting dice: winner’s die first; doubles expand to 4
        first  = jax.lax.cond(current_player == self.consts.WHITE, lambda _: w, lambda _: b, operand=None)
        second = jax.lax.cond(current_player == self.consts.WHITE, lambda _: b, lambda _: w, operand=None)
        is_double = first == second
        dice = jax.lax.cond(
            is_double,
            lambda _: jnp.array([first, first, first, first]),
            lambda _: jnp.array([first, second, 0, 0]),
            operand=None
        )

        # starting cursor
        initial_cursor = jax.lax.cond(
            current_player == self.consts.WHITE, lambda _: 0, lambda _: 23, operand=None
        )

        return BackgammonState(
            board=board,
            dice=dice,
            current_player=current_player,
            is_game_over=False,
            key=key,
            last_move=(-1, -1),
            last_dice=-1,
            cursor_position=initial_cursor,
            picked_checker_from=-1,
            game_phase=jnp.int32(1),              # SELECTING_CHECKER
            last_action=JAXAtariAction.NOOP,
            await_keyup=False,
            last_valid_drop=-1,
        )

    def reset(self, key: jax.random.PRNGKey = None) -> Tuple[jnp.ndarray, BackgammonState]:
        """Reset the environment. The initial roll happens inside init_state now."""
        if key is None:
            key = jax.random.PRNGKey(0)
        state = self.init_state(key)   # already rolled & selected starter
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
            lambda d: jnp.array([d[0], d[1], 0, 0]),  # only 2 dice used
            operand=dice
        )

        return expanded_dice, key

    @partial(jax.jit, static_argnums=(0,))
    def get_player_index(self, player: int) -> int:
        return jax.lax.cond(player == self.consts.WHITE, lambda _: 0, lambda _: 1, operand=None)

    @partial(jax.jit, static_argnums=(0,))
    def _is_valid_move_basic(self, state: BackgammonState, move: Tuple[int, int]) -> bool:
        from_point, to_point = move
        board = state.board
        player = state.current_player
        player_idx = self.get_player_index(player)
        opponent_idx = 1 - player_idx

        in_bounds = ((0 <= from_point) & (from_point <= 24) &
                    (0 <= to_point) & (to_point <= self.consts.HOME_INDEX) &
                    (to_point != self.consts.BAR_INDEX))

        from_point = jnp.asarray(from_point)
        to_point   = jnp.asarray(to_point)

        same_point         = from_point == to_point
        has_bar_checkers   = board[player_idx, self.consts.BAR_INDEX] > 0
        moving_from_bar    = from_point == self.consts.BAR_INDEX
        must_move_from_bar = (~moving_from_bar) & has_bar_checkers
        moving_to_bar      = to_point == self.consts.BAR_INDEX

        early_invalid = (~in_bounds) | must_move_from_bar | same_point | moving_to_bar
        def return_false(_): return jnp.bool_(False)

        def continue_check(_):
            def bar_case(_):
                def is_valid_entry(dv) -> bool:
                    expected_entry = jax.lax.select(player == self.consts.WHITE, dv - 1, 24 - dv)
                    matches_entry  = (to_point == expected_entry)
                    entry_open     = board[opponent_idx, expected_entry] <= 1
                    return matches_entry & entry_open
                bar_has  = board[player_idx, self.consts.BAR_INDEX] > 0
                valid_en = jnp.any(jax.vmap(is_valid_entry)(state.dice))
                return bar_has & valid_en

            def bearing_off_case(_):
                can_bear_off = self.check_bearing_off(state, player)
                
                # ALE-konforme Bearing-Off-Distanz:
                # White: Point 18=6-pip, 19=5-pip, ..., 23=1-pip → distance = 24 - from_point
                # Black: Point 5=6-pip, 4=5-pip, ..., 0=1-pip   → distance = from_point + 1
                bearing_off_distance = jax.lax.cond(
                    player == self.consts.WHITE,
                    lambda _: 24 - from_point,  # FIX: war HOME_INDEX - from - 1
                    lambda _: from_point + 1,
                    operand=None
                )
                dice_match = jnp.any(state.dice == bearing_off_distance)

                # Prüfe ob höhere Steine existieren (für Oversize-Regel)
                # "Höher" bedeutet: WEITER vom Home entfernt (größere Distanz)
                def white_check():
                    # White: Point 23=1-pip (näheste), 22=2-pip, ..., 18=6-pip (weiteste)
                    # Höher = größere Distanz = kleinere Point-Number
                    # Prüfe ob Steine auf Points < from_point existieren (weiter entfernt)
                    full_home = jax.lax.dynamic_slice(board[player_idx], (18,), (6,))
                    mask = (jnp.arange(18, 24) < from_point)
                    return jnp.any(full_home * mask > 0)
                
                def black_check():
                    # Black: Point 0=1-pip (näheste), 1=2-pip, ..., 5=6-pip (weiteste)
                    # Höher = größere Distanz = größere Point-Number
                    # Prüfe ob Steine auf Points > from_point existieren (weiter entfernt)
                    full_home = jax.lax.dynamic_slice(board[player_idx], (0,), (6,))
                    mask = (jnp.arange(0, 6) > from_point)
                    return jnp.any(full_home * mask > 0)

                higher_exists = jax.lax.cond(player == self.consts.WHITE, lambda _: white_check(), lambda _: black_check(), operand=None)
                larger_ok     = jnp.any(state.dice > bearing_off_distance)
                has_piece     = board[player_idx, from_point] > 0
                valid_bear    = has_piece & (dice_match | ((~higher_exists) & larger_ok))
                return jax.lax.cond(can_bear_off, lambda _: valid_bear, lambda _: jnp.bool_(False), operand=None)

            def normal_case(_):
                has_piece   = board[player_idx, from_point] > 0
                not_blocked = board[opponent_idx, to_point] <= 1
                base_dist   = jax.lax.select(player == self.consts.WHITE, to_point - from_point, from_point - to_point)
                correct_dir = base_dist > 0
                dice_match  = jnp.any(state.dice == base_dist)
                return has_piece & not_blocked & correct_dir & dice_match & (to_point != self.consts.BAR_INDEX)

            return jax.lax.cond(
                moving_from_bar,
                bar_case,
                lambda _: jax.lax.cond(to_point == self.consts.HOME_INDEX, bearing_off_case, normal_case, operand=None),
                operand=None
            )

        return jax.lax.cond(early_invalid, return_false, continue_check, operand=None)

    @partial(jax.jit, static_argnums=(0,))
    def _distinct_nonzero_dice(self, dice: jnp.ndarray):
        nz = dice * (dice > 0)                   
        hi = jnp.max(nz).astype(jnp.int32)
        lo = jnp.max(jnp.where((nz > 0) & (nz < hi), nz, 0)).astype(jnp.int32)
        has_two = (lo > 0) & (hi > lo)
        return has_two, lo, hi

    @partial(jax.jit, static_argnums=(0,))
    def _any_move_with_single_die(self, state: BackgammonState, die_value) -> bool:
        die = jnp.asarray(die_value, dtype=jnp.int32)
        test_state = state._replace(dice=jnp.array([die, 0, 0, 0], dtype=jnp.int32))
        mask = jax.vmap(lambda mv: self._is_valid_move_basic(test_state, mv))(self._action_pairs)
        return jnp.any(mask)

    @partial(jax.jit, static_argnums=(0,))
    def is_valid_move(self, state: BackgammonState, move: Tuple[int, int]) -> bool:
        basic_ok = self._is_valid_move_basic(state, move)

        def enforce(_):
            has_two, lo, hi = self._distinct_nonzero_dice(state.dice)  # JAX scalars

            can_hi = self._any_move_with_single_die(state, hi)
            can_lo = jax.lax.cond(
                has_two,
                lambda __: self._any_move_with_single_die(state, lo),
                lambda __: jnp.bool_(False),
                operand=None
            )

            # Höherer-Würfel-Regel: Wenn nur EINER der Würfel geht, muss der HÖHERE verwendet werden
            # need_rule = True wenn: 2 verschiedene Würfel UND höherer geht UND niedrigerer NICHT geht
            need_rule = has_two & can_hi & (~can_lo)

            def must_use_hi(_):
                # Prüfe ob aktueller Move den höheren Würfel benutzt
                state_hi = state._replace(dice=jnp.array([hi, 0, 0, 0], dtype=jnp.int32))
                uses_hi = self._is_valid_move_basic(state_hi, move)
                
                # Wenn Regel aktiv: Move ist nur legal wenn er höheren Würfel benutzt
                return uses_hi

            ok2 = jax.lax.cond(need_rule, must_use_hi, lambda __: jnp.bool_(True), operand=None)
            return basic_ok & ok2

        return jax.lax.cond(basic_ok, enforce, lambda _: jnp.bool_(False), operand=None)

    @partial(jax.jit, static_argnums=(0,))
    def check_bearing_off(self, state: BackgammonState, player: int) -> bool:
        """Check for bearing off using lax.cond instead of if statements."""
        board = state.board
        player_idx = self.get_player_index(player)

        # Full 0–23 range (playable points)
        point_indices = jnp.arange(24)

        # Mask for non-home points
        non_home_mask = jnp.where(player == self.consts.WHITE,
                                  point_indices < 18,  # Points 0–17 (before 19)
                                  point_indices > 5)

        in_play = board[player_idx, :24]
        outside_home_checkers = jnp.sum(jnp.where(non_home_mask, in_play, 0))
        on_bar = board[player_idx, self.consts.BAR_INDEX]
        return (outside_home_checkers == 0) & (on_bar == 0)

    @partial(jax.jit, static_argnums=(0,))
    def has_any_legal_move(self, state: BackgammonState) -> jnp.ndarray:
        mask = jax.vmap(lambda mv: self.is_valid_move(state, mv))(self._action_pairs)
        return jnp.any(mask)

    @partial(jax.jit, static_argnums=(0,))
    def _auto_pass_if_stuck(self, state: BackgammonState) -> BackgammonState:
        """
        If the current player has NO legal move with the current dice, auto-pass:
        - switch player
        - roll new dice for the opponent
        - reset phase & cursor for the new player
        Otherwise, return state unchanged.
        """
        def do_pass(_):
            next_dice, new_key = self.roll_dice(state.key)
            next_player = -state.current_player
            next_cursor = jax.lax.cond(
                next_player == self.consts.WHITE, lambda _: jnp.int32(0), lambda _: jnp.int32(23), operand=None
            )
            return state._replace(
                dice=next_dice,
                key=new_key,
                current_player=next_player,
                game_phase=jnp.int32(1),          # SELECTING_CHECKER
                cursor_position=next_cursor,
                picked_checker_from=jnp.int32(-1),
                last_move=state.last_move,
                last_dice=jnp.int32(-1)
            )

        return jax.lax.cond(self.has_any_legal_move(state), lambda _: state, do_pass, operand=None)

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
    def _loser_has_checker_in_winner_home(self, state: BackgammonState, winner: int) -> bool:
        # winner == WHITE(1) → check BLACK on 18..23
        # winner == BLACK(-1) → check WHITE on 0..5
        def white_wins(_):
            return jnp.sum(state.board[1, 18:24]) > 0  # black in white home
        def black_wins(_):
            return jnp.sum(state.board[0, 0:6]) > 0    # white in black home
        return jax.lax.cond(winner == self.consts.WHITE, white_wins, black_wins, operand=None)

    @partial(jax.jit, static_argnums=(0,))
    def compute_outcome_multiplier(self, state: BackgammonState) -> jax.Array:
        """
        Returns 1 (single), 2 (gammon), 3 (backgammon) × DOUBLING_CUBE as int32.
        Gammon: loser has borne off 0.
        Backgammon: loser has borne off 0 AND has checkers on bar or in winner's home.
        """
        white_off = state.board[0, self.consts.HOME_INDEX]
        black_off = state.board[1, self.consts.HOME_INDEX]
        white_bar = state.board[0, self.consts.BAR_INDEX]
        black_bar = state.board[1, self.consts.BAR_INDEX]

        white_won = (white_off == self.consts.NUM_CHECKERS)
        black_won = (black_off == self.consts.NUM_CHECKERS)

        def when_white_wins(_):
            loser_off = black_off
            loser_bar = black_bar
            in_winner_home = self._loser_has_checker_in_winner_home(state, self.consts.WHITE)
            is_backgammon = (loser_off == 0) & ((loser_bar > 0) | in_winner_home)
            is_gammon     = (loser_off == 0) & (~is_backgammon)
            mul = jax.lax.select(is_backgammon, jnp.int32(3),
                jax.lax.select(is_gammon,     jnp.int32(2), jnp.int32(1)))
            return (mul * jnp.int32(self.consts.DOUBLING_CUBE)).astype(jnp.int32)

        def when_black_wins(_):
            loser_off = white_off
            loser_bar = white_bar
            in_winner_home = self._loser_has_checker_in_winner_home(state, self.consts.BLACK)
            is_backgammon = (loser_off == 0) & ((loser_bar > 0) | in_winner_home)
            is_gammon     = (loser_off == 0) & (~is_backgammon)
            mul = jax.lax.select(is_backgammon, jnp.int32(3),
                jax.lax.select(is_gammon,     jnp.int32(2), jnp.int32(1)))
            return (mul * jnp.int32(self.consts.DOUBLING_CUBE)).astype(jnp.int32)

        def no_winner(_):
            return jnp.int32(1)

        return jax.lax.cond(
            white_won, when_white_wins,
            lambda _: jax.lax.cond(black_won, when_black_wins, no_winner, operand=None),
            operand=None
        )

    @partial(jax.jit, static_argnums=(0,))
    def compute_distance(self, player, from_point, to_point):
        """Compute move distance based on player and points, including bearing off."""
        is_from_bar = from_point == self.consts.BAR_INDEX

        # Distance when entering from BAR
        bar_distance = jax.lax.cond(
            player == self.consts.WHITE,
            lambda _: to_point + 1,         # WHITE enters on 0..5 → die = to_point+1
            lambda _: 24 - to_point,        # BLACK enters on 23..18 → die = 24-to_point
            operand=None
        )

        # Regular / bearing-off distance
        regular_distance = jax.lax.cond(
            to_point == self.consts.HOME_INDEX,
            # FIX: bearing-off distance
            # WHITE home is 18..23, with point 23 = 1-pip → distance = 24 - from_point
            # BLACK home is 0..5,   with point 0  = 1-pip → distance = from_point + 1
            lambda _: jax.lax.cond(
                player == self.consts.WHITE,
                lambda _: jnp.int32(24) - from_point,   
                lambda _: from_point + 1,
                operand=None
            ),
            # Normal board move
            lambda _: jax.lax.cond(
                player == self.consts.WHITE,
                lambda _: to_point - from_point,        # WHITE moves upward in index
                lambda _: from_point - to_point,        # BLACK moves downward in index
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
        
        new_state = self._auto_pass_if_stuck(new_state)

        obs = self._get_observation(new_state)
        reward = self._get_reward(state, new_state)
        all_rewards = self._get_all_reward(state, new_state)
        done = self._get_done(new_state)
        info = self._get_info(new_state, all_rewards)
        return obs, new_state, reward, done, info, new_key


    def step(self, state: BackgammonState, action: jnp.ndarray):
        """Interactive step with JAX-safe debounce: each press triggers once; holding does nothing until NOOP arrives."""

        # Action flags (JAX bool scalars)
        is_left  = action == JAXAtariAction.LEFT
        is_right = action == JAXAtariAction.RIGHT
        is_fire  = action == JAXAtariAction.FIRE
        is_noop  = action == JAXAtariAction.NOOP

        # ---------- helpers (JAX-safe) ----------
        def handle_cursor_move(s, direction):
            """
            Cursor moves on the rotated ring:

            LEFT  : 0→1→2→3→4→5→26→6→7→8→9→10→11→12→13→14→15→16→17→24→18→19→20→21→22→23
            RIGHT : reverse of the above.

            Rules:
            - Bars (24, 26) are cursor-only (you can stand on them but not drop there).
            - No wrap 0↔23 ever.
            - While MOVING at 0/23:
                * at 0  : RIGHT does nothing (unless legal bear-off jump to 25), LEFT is allowed.
                * at 23 : LEFT  does nothing (unless legal bear-off jump to 25), RIGHT is allowed.
            - Leaving HOME (25) only back to its visual edge (25+LEFT→23 for White, 25+RIGHT→0 for Black).
            """
            can_move = (s.game_phase == 1) | (s.game_phase == 2)
            pos      = s.cursor_position

            is_left  = jnp.array(direction == -1, dtype=jnp.bool_)
            is_right = jnp.array(direction ==  1, dtype=jnp.bool_)

            at_home       = (pos == jnp.int32(self.consts.HOME_INDEX))   # 25
            at_left_edge  = (pos == jnp.int32(0))
            at_right_edge = (pos == jnp.int32(23))

            is_moving = (s.game_phase == jnp.int32(2))
            is_white  = (s.current_player == self.consts.WHITE)

            # Rotated ring order (26 = right bar, 24 = left bar)
            ring = jnp.array(
                [0, 1, 2, 3, 4, 5, 26, 6, 7, 8, 9, 10, 11,
                12, 13, 14, 15, 16, 17, 24, 18, 19, 20, 21, 22, 23],
                dtype=jnp.int32
            )
            ring_len = ring.shape[0]

            # Precompute next maps for LEFT/RIGHT on the ring
            def build_maps(_):
                next_left  = jnp.arange(27, dtype=jnp.int32)
                next_right = jnp.arange(27, dtype=jnp.int32)
                def body(i, carry):
                    nL, nR = carry
                    a = ring[i]
                    b = ring[(i + 1) % ring_len]  # LEFT goes forward in ring array
                    c = ring[(i - 1) % ring_len]  # RIGHT goes backward in ring array
                    nL = nL.at[a].set(b)
                    nR = nR.at[a].set(c)
                    return (nL, nR)
                return jax.lax.fori_loop(0, ring_len, body, (next_left, next_right))

            next_left, next_right = build_maps(None)

            # Candidate target from ring movement (no special cases yet)
            ring_target = jax.lax.select(is_left, next_left[pos],
                            jax.lax.select(is_right, next_right[pos], pos))

            # ---- Bear-off jump to HOME (only when MOVING and bearing-off is allowed) ----
            not_from_bar     = (s.picked_checker_from != jnp.int32(self.consts.BAR_INDEX))
            can_bear_off_now = self.check_bearing_off(s, s.current_player)
            # Your desired directions:
            #  - from 23 with LEFT  → 25 (White)
            #  - from  0 with RIGHT → 25 (Black)
            jump_home_white  = is_moving & is_left  & at_right_edge & is_white
            jump_home_black  = is_moving & is_right & at_left_edge  & (~is_white)
            to_home_attempt  = (jump_home_white | jump_home_black) & not_from_bar & can_bear_off_now
            home_target      = jnp.int32(self.consts.HOME_INDEX)

            # ---- Block wrap 0↔23 always ----
            block_wrap = (at_left_edge & is_right) | (at_right_edge & is_left)
            ring_target = jax.lax.select(block_wrap, pos, ring_target)

            # ---- While MOVING on edges, block only the forbidden direction (don't "stick") ----
            # At 0  : block RIGHT (unless home jump)
            # At 23 : block LEFT  (unless home jump)
            block_edge_dir = is_moving & (
                (at_left_edge  & is_right & (~to_home_attempt)) |
                (at_right_edge & is_left  & (~to_home_attempt))
            )
            ring_target = jax.lax.select(block_edge_dir, pos, ring_target)

            # ---- Leaving HOME only to the correct edge ----
            from_home_target = jax.lax.cond(
                is_white,
                lambda _: jax.lax.select(is_left,  jnp.int32(23), pos),  # White: 25 + LEFT  → 23
                lambda _: jax.lax.select(is_right, jnp.int32(0),  pos),  # Black: 25 + RIGHT → 0
                operand=None
            )

            # Final resolution:
            # 1) If at HOME → restrict leaving
            # 2) Else if a legal HOME jump now → 25
            # 3) Else follow the (possibly blocked) ring target
            target_if_move = jax.lax.cond(
                at_home,
                lambda _: from_home_target,
                lambda _: jax.lax.select(to_home_attempt, home_target, ring_target),
                operand=None
            )

            new_cursor = jax.lax.cond(can_move, lambda _: target_if_move, lambda _: pos, operand=None)
            ns = s._replace(cursor_position=new_cursor)
            return self._get_observation(ns), ns, 0.0, False, self._get_info(ns)


        def handle_space(s):
            def do_roll(ss):
                dice, key = self.roll_dice(ss.key)
                ns = ss._replace(dice=dice, key=key, game_phase=1)  # SELECTING_CHECKER
                # Auto-pass if no legal move with these dice
                ns = self._auto_pass_if_stuck(ns)
                return self._get_observation(ns), ns, 0.0, False, self._get_info(ns)

            def do_select(ss):
                player_idx = self.get_player_index(ss.current_player)
                pos = ss.cursor_position
                is_bar_cursor = (pos == jnp.int32(self.consts.BAR_INDEX)) | (pos == jnp.int32(26))

                selectable = jax.lax.cond(
                    is_bar_cursor,
                    lambda _: ss.board[player_idx, self.consts.BAR_INDEX],   # real bar store
                    lambda _: ss.board[player_idx, pos],
                    operand=None
                )
                has_checker = selectable > 0
                picked_from = jax.lax.select(is_bar_cursor, jnp.int32(self.consts.BAR_INDEX), pos)
                picked_bar_side = jax.lax.select(is_bar_cursor, pos, ss.picked_bar_side)
                
                ns = jax.lax.cond(
                    has_checker,
                    # enter MOVING and remember which bar half we picked from
                    lambda s2: s2._replace(
                        picked_checker_from=picked_from,
                        picked_bar_side=picked_bar_side,
                        game_phase=2
                    ),
                    lambda s2: s2,
                    operand=ss
                )
                return self._get_observation(ns), ns, 0.0, False, self._get_info(ns)

            def do_drop(ss):
                move = (ss.picked_checker_from, ss.cursor_position)
                is_valid = self.is_valid_move(ss, move)

                def execute_valid_move(s2):
                    obs, ns, reward, done, info, key = self.step_impl(s2, move, s2.key)
                    all_dice_used = jnp.all(ns.dice == 0)
                    next_phase = jax.lax.cond(all_dice_used, lambda _: 0, lambda _: 1, operand=None)
                    next_cursor = jax.lax.cond(
                        ns.current_player != s2.current_player,
                        lambda _: jax.lax.cond(ns.current_player == self.consts.WHITE, lambda _: 0, lambda _: 23, operand=None),
                        lambda _: s2.cursor_position,
                        operand=None
                    )
                    fs = ns._replace(
                        picked_checker_from=-1,
                        picked_bar_side=-1,                   
                        game_phase=next_phase,
                        cursor_position=next_cursor,
                        key=key,
                        last_valid_drop=s2.cursor_position
                    )
                    return obs, fs, reward, done, info

                def invalid_drop(s2):
                    # Checker jumps back: move cursor to origin, return to SELECTING phase
                    # FIX: Bei Bar-Drops zurück zur ursprünglichen Bar-Seite (picked_bar_side), nicht nur BAR_INDEX
                    was_from_bar = (s2.picked_checker_from == jnp.int32(self.consts.BAR_INDEX))
                    fallback_cursor = jax.lax.cond(
                        was_from_bar & (s2.picked_bar_side >= 0),
                        lambda _: s2.picked_bar_side,  # Zurück zur korrekten Bar-Hälfte
                        lambda _: s2.picked_checker_from,  # Normaler Fall: zurück zum Punkt
                        operand=None
                    )
                    ns = s2._replace(
                        picked_checker_from=-1,
                        picked_bar_side=-1,                   
                        game_phase=1,
                        cursor_position=fallback_cursor
                    )
                    return self._get_observation(ns), ns, 0.0, False, self._get_info(ns)

                return jax.lax.cond(is_valid, execute_valid_move, invalid_drop, operand=ss)

            return jax.lax.switch(state.game_phase, [do_roll, do_select, do_drop], operand=state)

        def process_action_once(s):
            """Select exactly one action branch using nested lax.cond (priority: LEFT, RIGHT, FIRE, else NOOP/idle)."""
            # LEFT
            return jax.lax.cond(
                is_left,
                lambda ss: handle_cursor_move(ss, -1),
                lambda ss: jax.lax.cond(
                    is_right,
                    lambda ss2: handle_cursor_move(ss2, 1),
                    lambda ss2: jax.lax.cond(
                        is_fire,
                        handle_space,
                        # default/NOOP: no state change
                        lambda ss3: (self._get_observation(ss3), ss3, 0.0, False, self._get_info(ss3)),
                        operand=ss2
                    ),
                    operand=ss
                ),
                operand=s
            )

        # When awaiting key-up: only NOOP clears; everything else is ignored.
        def handle_when_blocked(s):
            ns = jax.lax.cond(
                is_noop,
                lambda _: s._replace(await_keyup=False, last_action=JAXAtariAction.NOOP),
                lambda _: s,
                operand=None
            )
            return self._get_observation(ns), ns, 0.0, False, self._get_info(ns)

        # When free: process one action and then arm debounce for LEFT/RIGHT/FIRE.
        def handle_when_free(s):
            obs, ns, reward, done, info = process_action_once(s)
            should_arm = jnp.logical_or(jnp.logical_or(is_left, is_right), is_fire)  # arm for left/right/space
            ns2 = jax.lax.cond(
                should_arm,
                lambda _: ns._replace(await_keyup=True, last_action=action),
                lambda _: ns,
                operand=None
            )
            return obs, ns2, reward, done, info

        return jax.lax.cond(state.await_keyup, handle_when_blocked, handle_when_free, operand=state)


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

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: BackgammonState, all_rewards: chex.Array = None) -> BackgammonInfo:
        """Extract info from state with consistent JAX types."""
        if all_rewards is None:
            # keep shape stable across the codebase (1,) float32 by default
            all_rewards = jnp.zeros((1,), dtype=jnp.float32)

        return BackgammonInfo(
            player=jnp.asarray(state.current_player, dtype=jnp.int32),
            dice=jnp.asarray(state.dice, dtype=jnp.int32),
            all_rewards=jnp.asarray(all_rewards, dtype=jnp.float32),
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, prev: BackgammonState, state: BackgammonState) -> jax.Array:
        """+1 winner, -1 loser (× gammon/backgammon)."""
        white_off = state.board[0, self.consts.HOME_INDEX]
        black_off = state.board[1, self.consts.HOME_INDEX]

        white_won = (white_off == self.consts.NUM_CHECKERS)
        black_won = (black_off == self.consts.NUM_CHECKERS)

        winner_sign = jnp.where(white_won, 1.0,
                        jnp.where(black_won, -1.0, 0.0)).astype(jnp.float32)

        mult = self.compute_outcome_multiplier(state).astype(jnp.float32)  # or jnp.float32(1.0) for ±1 only

        return (winner_sign * mult).astype(jnp.float32)

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

    def render(self, state: BackgammonState) -> jnp.ndarray:
        return self.renderer.render(state)


class BackgammonRenderer(JAXGameRenderer):
    def __init__(self, env=None):
        super().__init__()
        self.env = env
        # Initialize all rendering parameters
        self.frame_height = 210
        self.frame_width = 160
        self.color_background = jnp.array([0, 0, 0], dtype=jnp.uint8)  # black background
        self.color_board = jnp.array([0, 0, 0], dtype=jnp.uint8)  # black board
        self.color_triangle_light = jnp.array([42, 44, 168], dtype=jnp.uint8)  # blue points
        self.color_triangle_dark = jnp.array([87, 147, 74], dtype=jnp.uint8)  # green points
        self.color_white_checker = jnp.array([255, 255, 255], dtype=jnp.uint8)  # white checkers
        self.color_black_checker = jnp.array([230, 88, 83], dtype=jnp.uint8)  # red checkers
        self.color_border = jnp.array([81, 146, 119], dtype=jnp.uint8)  # green bar

        self.top_margin_for_dice = 25  # pixels reserved for dice row

        # Geometry
        self.board_margin = 8
        self.triangle_length = 60
        self.triangle_thickness = 12
        self.bar_thickness = 14
        # Chip geometry
        self.checker_width  = 4
        self.checker_height = 4

        # Horizontal distance of the first column from the triangle BASE edge
        self.base_margin = 2          # pixels

        # vertical & horizontal spacing controls
        self.band_top_margin    = 1.2   # distance from top band edge
        self.band_bottom_margin = 1   # distance from bottom band edge
        self.chip_gap_y         = 2   # vertical gap between the two chip rows
        self.chip_gap_x         = 4   # horizontal gap between columns (used in cx step)

        self.checker_stack_offset = self.checker_height + self.chip_gap_y

        self.edge_line_thickness = 2
        self.edge_line_color     = jnp.array([81, 146, 119], dtype=jnp.uint8)

        self.bar_y = self.top_margin_for_dice + self.frame_height // 2 - self.bar_thickness // 2 - 10
        self.bar_x = self.board_margin
        self.bar_width = self.frame_width - 2 * self.board_margin
        self.bar_edge_padding = 2 
        self.bar_vertical_padding = 2  
        self.triangle_positions = self._compute_triangle_positions()

    def _compute_triangle_positions(self):
        # Screen anchors
        left_x  = self.board_margin
        right_x = self.frame_width - self.board_margin - self.triangle_length

        y_top    = self.top_margin_for_dice + self.board_margin
        y_bottom = self.frame_height - self.board_margin
        band_h   = 6 * self.triangle_thickness

        positions = []

        # 0..5  (lower-right)  bottom -> top   (0 lowest, 5 highest)
        for i in range(6):
            y = y_bottom - (i + 1) * self.triangle_thickness
            positions.append((right_x, y))

        # 6..11 (upper-right)  bottom -> top   (6 lowest near center, 11 highest)
        for i in range(6):
            y = y_top + (5 - i) * self.triangle_thickness
            positions.append((right_x, y))

        # 12..17 (upper-left)  top -> bottom   (12 highest, 17 lowest)
        for i in range(6):
            y = y_top + i * self.triangle_thickness
            positions.append((left_x, y))

        # 18..23 (lower-left)  top -> bottom   (18 highest near center, 23 lowest)  <-- FIX
        lower_left_top_y = y_bottom - band_h  # top edge of the lower band
        for i in range(6):
            y = lower_left_top_y + i * self.triangle_thickness
            positions.append((left_x, y))

        # bar-left (24): center of left half of the bar (logic unchanged)
        bar_left_x   = self.bar_x + (self.bar_width // 4)
        bar_center_y = self.bar_y + self.bar_thickness // 2
        positions.append((bar_left_x, bar_center_y))

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

        t = jax.lax.select(point_right, (xx_f - x_f) / length_f, (x_f + length_f - xx_f) / length_f)
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
        board_y = self.top_margin_for_dice + self.board_margin - 6
        board_w = self.frame_width - 2 * (self.board_margin - 6)
        board_h = self.frame_height - self.top_margin_for_dice - 2 * (self.board_margin - 6)

        frame = self._draw_rectangle(frame, board_x, board_y, board_w, board_h, self.color_board)

        # Split bar into two half rectangles (left for Red, right for White)
        half_w = self.bar_width // 2
        # left half
        frame = self._draw_rectangle(frame, self.bar_x, self.bar_y, half_w, self.bar_thickness, self.color_border)
        # right half
        frame = self._draw_rectangle(frame, self.bar_x + half_w, self.bar_y, self.bar_width - half_w, self.bar_thickness, self.color_border)
        return frame

    @partial(jax.jit, static_argnums=(0,))
    def _draw_triangles(self, frame):
        """
        Draw triangles using the rotated layout.
        - Left column triangles point RIGHT (toward center)
        - Right column triangles point LEFT (toward center)
        - Alternate colors by column and row to get the classic pattern.
        """
        left_x  = self.board_margin
        right_x = self.frame_width - self.board_margin - self.triangle_length

        def draw_triangle_at_index(i, fr):
            pos = self.triangle_positions[i]
            x = pos[0]; y = pos[1]

            # Column flags
            is_left_column  = (x == left_x)

            # Row index within its column band (0..5):
            #  - lower-right:  i in 0..5
            #  - upper-right:  i in 6..11
            #  - upper-left:   i in 12..17
            #  - lower-left:   i in 18..23
            band_idx = jnp.where(i < 6, i,
                        jnp.where(i < 12, i - 6,
                        jnp.where(i < 18, i - 12, i - 18)))

            # Base start color by column & band (your original logic)
            is_upper_band = ((i >= 6) & (i < 12)) | ((i >= 12) & (i < 18))
            start_light = jnp.where(
                is_left_column,
                jnp.where(is_upper_band, True, False),   # left column: upper starts LIGHT, lower starts DARK
                jnp.where(is_upper_band, False, True)    # right column: upper starts DARK,  lower starts LIGHT
            )

            # --- Flip alternation for specific bands: upper-right (6..11) and lower-left (18..23) ---
            # band_id: 0=lower-right, 1=upper-right, 2=upper-left, 3=lower-left
            band_id = jnp.where(i < 6, 0,
                        jnp.where(i < 12, 1,
                        jnp.where(i < 18, 2, 3)))
            flip = (band_id == 1) | (band_id == 3)
            start_light = jnp.logical_xor(start_light, flip)

            # Alternate color within the band
            use_light = jnp.where(start_light, (band_idx % 2 == 0), (band_idx % 2 == 1))
            color = jax.lax.select(use_light, self.color_triangle_light, self.color_triangle_dark)

            # Tip direction: left column points right; right column points left.
            point_right = is_left_column

            return self._draw_triangle(fr, x, y, self.triangle_length, self.triangle_thickness, color, point_right)

        return jax.lax.fori_loop(0, 24, draw_triangle_at_index, frame)


    @partial(jax.jit, static_argnums=(0,))
    def _draw_edge_lines(self, frame):
        # triangle columns start/end
        left_x  = self.board_margin
        right_x = self.frame_width - self.board_margin - self.triangle_length
        tri_span_x = left_x
        tri_span_w = (right_x + self.triangle_length) - left_x  # ends exactly where triangles end

        # vertical reference: triangles occupy [y_top_tri, y_bottom_tri)
        y_top_tri    = self.top_margin_for_dice + self.board_margin
        y_bottom_tri = self.frame_height - self.board_margin

        # tweak thickness/offsets
        thickness     = self.edge_line_thickness               # e.g., 3
        bottom_offset = 2                                       # <- push the lower band a bit lower

        # positions
        top_y    = y_top_tri - thickness                        # just above top triangles
        bottom_y = y_bottom_tri + bottom_offset                  # a bit below bottom triangles

        # draw
        frame = self._draw_rectangle(frame, tri_span_x, top_y,    tri_span_w, thickness, self.edge_line_color)
        frame = self._draw_rectangle(frame, tri_span_x, bottom_y, tri_span_w, thickness, self.edge_line_color)
        return frame


    @partial(jax.jit, static_argnums=(0,))
    def _draw_checkers_on_point(self, frame, point_idx, white_count, black_count):
        """
        Draw 2-high stacks that fill from BOTTOM to TOP inside each triangle band.
        Spacing is controlled by:
        - base_margin:   horizontal padding from the triangle base edge
        - chip_gap_x:    horizontal gap between columns
        - band_top_margin / band_bottom_margin: vertical paddings in the band
        - checker_stack_offset: vertical distance between the two rows
        """
        pos = self.triangle_positions[point_idx]
        x, y = pos[0], pos[1]

        is_left_column = (x == self.board_margin)
        # march direction (+1 on left column, -1 on right column)
        dir_sign = jnp.where(is_left_column, 1, -1)

        # start near the triangle BASE edge, not the tip
        base_x = jnp.where(
            is_left_column,
            x + self.base_margin,
            x + self.triangle_length - self.checker_width - self.base_margin
        )

        # band vertical limits with adjustable paddings
        band_top    = y + self.band_top_margin
        band_bottom = y + self.triangle_thickness - self.band_bottom_margin

        # bottom row sits against band_bottom; second row above it
        row0_y = jnp.int32(band_bottom - self.checker_height)
        row1_y = jnp.int32(row0_y - self.checker_stack_offset)

        # horizontal step between columns uses chip_gap_x
        col_step = self.checker_width + self.chip_gap_x

        def draw_stack(fr, count, color):
            def draw_single(i, f):
                col = i // 2     # column index along the triangle
                row = i & 1      # 0 or 1 (bottom row first, then above)
                cx  = base_x + dir_sign * (col * col_step)
                cy  = jnp.where(row == 0, row0_y, row1_y)
                return self._draw_rectangle(f, cx, cy, self.checker_width, self.checker_height, color)
            return jax.lax.fori_loop(0, count, draw_single, fr)

        frame = draw_stack(frame, white_count, self.color_white_checker)
        frame = draw_stack(frame, black_count, self.color_black_checker)
        return frame

    @partial(jax.jit, static_argnums=(0,))
    def _draw_bar_checkers(self, frame, white_count, black_count):
        """
        Bar stacks anchored to OUTER edges, growing bottom→top.
        Uses chip_gap_x for horizontal spacing and checker_stack_offset for vertical.
        """
        col_step = self.checker_width + self.chip_gap_x
        row_step = self.checker_stack_offset
        two_cols_w = 2 * self.checker_width + self.chip_gap_x  # total width of two columns with one gap

        # leftmost and rightmost anchors inside the bar
        base_x_left  = self.bar_x + self.bar_edge_padding
        base_x_right = self.bar_x + self.bar_width - self.bar_edge_padding - two_cols_w

        # bottom y inside the bar
        base_y_bottom = (
            self.bar_y + self.bar_thickness
            - self.bar_vertical_padding - self.checker_height
        )

        def draw_stack(fr, count, base_x, color):
            def draw_single(i, f):
                row = i // 2
                col = i % 2
                x = base_x + col * col_step
                y = base_y_bottom - row * row_step   # bottom→top stacking
                return self._draw_rectangle(f, x, y, self.checker_width, self.checker_height, color)
            return jax.lax.fori_loop(0, count, draw_single, fr)

        # Red (black player) on the far LEFT, White on the far RIGHT
        frame = draw_stack(frame, black_count, base_x_left,  self.color_black_checker)
        frame = draw_stack(frame, white_count, base_x_right, self.color_white_checker)
        return frame

    @partial(jax.jit, static_argnums=(0,))
    def _draw_dice(self, frame, dice, current_player):
        """Draw dice above the board, colored by the player with the turn."""
        dice_size = 12
        total_width = 4 * (dice_size + 3)
        start_x = self.frame_width // 2 - total_width // 2 + 12
        dice_y = self.board_margin

        # player-colored background, contrasting pips
        die_bg = jax.lax.cond(
            current_player == self.env.consts.WHITE,
            lambda _: self.color_white_checker,   # white
            lambda _: self.color_black_checker,   # red
            operand=None
        )
        pip = jax.lax.cond(
            current_player == self.env.consts.WHITE,
            lambda _: jnp.array([0, 0, 0], dtype=jnp.uint8),       # black pips on white
            lambda _: jnp.array([255, 255, 255], dtype=jnp.uint8), # white pips on red
            operand=None
        )

        def draw_single(i, fr):
            val = dice[i]
            dx = start_x + i * (dice_size + 3)

            def draw_val(_):
                fr2 = self._draw_rectangle(fr, dx, dice_y, dice_size, dice_size, die_bg)
                center_x = dx + dice_size // 2
                center_y = dice_y + dice_size // 2

                pip_size = 2

                def dot(f, x, y):
                    return self._draw_rectangle(f, x - pip_size // 2, y - pip_size // 2,
                                                pip_size, pip_size, pip)

                def p1(_): return dot(fr2, center_x, center_y)
                def p2(_):
                    fr3 = dot(fr2, center_x - 3, center_y - 3)
                    return dot(fr3, center_x + 3, center_y + 3)
                def p3(_):
                    fr3 = dot(fr2, center_x - 3, center_y - 3)
                    fr3 = dot(fr3, center_x, center_y)
                    return dot(fr3, center_x + 3, center_y + 3)
                def p4(_):
                    fr4 = dot(fr2, center_x - 3, center_y - 3)
                    fr4 = dot(fr4, center_x + 3, center_y - 3)
                    fr4 = dot(fr4, center_x - 3, center_y + 3)
                    return dot(fr4, center_x + 3, center_y + 3)
                def p5(_):
                    fr5 = p4(None)
                    return dot(fr5, center_x, center_y)
                def p6(_):
                    fr6 = dot(fr2, center_x - 3, center_y - 3)
                    fr6 = dot(fr6, center_x - 3, center_y)
                    fr6 = dot(fr6, center_x - 3, center_y + 3)
                    fr6 = dot(fr6, center_x + 3, center_y - 3)
                    fr6 = dot(fr6, center_x + 3, center_y)
                    return dot(fr6, center_x + 3, center_y + 3)

                funcs = [p1, p2, p3, p4, p5, p6]
                return jax.lax.switch(jnp.clip(val - 1, 0, 5), funcs, operand=None)

            return jax.lax.cond(val > 0, draw_val, lambda _: fr, operand=None)

        return jax.lax.fori_loop(0, 4, draw_single, frame)


    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: BackgammonState):
        frame = jnp.zeros((self.frame_height, self.frame_width, 3), dtype=jnp.uint8)

        frame = self._draw_board_outline(frame)
        frame = self._draw_triangles(frame)

        frame = self._draw_edge_lines(frame)

        # Draw highlight: under cursor in SELECTING, pinned to source in MOVING
        def draw_cursor_highlight(f):
            # Color used for the highlight overlay
            highlight_color = jnp.array([199, 172, 91], dtype=jnp.uint8)

            def which_index(_):
                # When MOVING, highlight the origin. If origin is BAR (24), use the remembered bar half (24 or 26).
                def moving_origin(_):
                    return jax.lax.cond(
                        state.picked_checker_from == jnp.int32(self.env.consts.BAR_INDEX),
                        # If we picked from bar, choose left(24) or right(26) based on picked_bar_side.
                        lambda __: jax.lax.select(
                            state.picked_bar_side == jnp.int32(26),
                            jnp.int32(26),
                            jnp.int32(self.env.consts.BAR_INDEX)
                        ),
                        # Otherwise just highlight the original triangle index.
                        lambda __: jnp.int32(state.picked_checker_from),
                        operand=None
                    )

                return jax.lax.cond(
                    state.game_phase == jnp.int32(2),   # MOVING
                    moving_origin,
                    lambda __: jax.lax.cond(
                        state.game_phase == jnp.int32(1),  # SELECTING
                        lambda ___: jnp.int32(state.cursor_position),
                        # IDLE (phase 0 or others): show last valid drop if any, else -1
                        lambda ___: jnp.int32(state.last_valid_drop),
                        operand=None
                    ),
                    operand=None
                )

            hi = which_index(None)
            no_hi = (hi < 0)

            def add_highlight(frame):
                # Triangle highlight (0..23)
                def on_triangle(fr):
                    return self._draw_triangle(
                        fr,
                        self.triangle_positions[hi][0],
                        self.triangle_positions[hi][1],
                        self.triangle_length,
                        self.triangle_thickness,
                        highlight_color,
                        self.triangle_positions[hi][0] == self.board_margin
                    )

                # Bar highlight halves
                def on_bar_left(fr):
                    half_w = self.bar_width // 2
                    return self._draw_rectangle(fr, self.bar_x, self.bar_y, half_w, self.bar_thickness, highlight_color)

                def on_bar_right(fr):
                    half_w = self.bar_width // 2
                    return self._draw_rectangle(fr, self.bar_x + half_w, self.bar_y,
                                                self.bar_width - half_w, self.bar_thickness, highlight_color)

                # Home (25) or anything else: no highlight
                def noop(fr): 
                    return fr

                return jax.lax.cond(
                    hi < jnp.int32(24),
                    on_triangle,
                    lambda fr2: jax.lax.cond(
                        hi == jnp.int32(self.env.consts.BAR_INDEX), on_bar_left,
                        lambda fr3: jax.lax.cond(hi == jnp.int32(26), on_bar_right, noop, operand=fr3),
                        operand=fr2
                    ),
                    operand=frame
                )

            return jax.lax.cond(no_hi, lambda fr: fr, add_highlight, operand=f)


        frame = draw_cursor_highlight(frame)

        # Draw checkers (with one removed if picked)
        def draw_point_checkers(point_idx, fr):
            player_idx = self.env.get_player_index(state.current_player)
            white_count = state.board[0, point_idx]
            black_count = state.board[1, point_idx]

            # Subtract one from picked position if in moving phase
            white_count = jax.lax.cond(
                (state.game_phase == 2) & (state.picked_checker_from == point_idx) & (player_idx == 0),
                lambda c: jnp.maximum(c - 1, 0),
                lambda c: c,
                operand=white_count
            )

            black_count = jax.lax.cond(
                (state.game_phase == 2) & (state.picked_checker_from == point_idx) & (player_idx == 1),
                lambda c: jnp.maximum(c - 1, 0),
                lambda c: c,
                operand=black_count
            )

            return self._draw_checkers_on_point(fr, point_idx, white_count, black_count)

        frame = jax.lax.fori_loop(0, 24, draw_point_checkers, frame)

        # Bar and home stacks
        player_idx = self.env.get_player_index(state.current_player)
        picked_from_bar = (state.game_phase == 2) & (state.picked_checker_from == self.env.consts.BAR_INDEX)

        # subtract 1 from the current player's bar if we picked the checker from BAR
        white_bar = state.board[0, self.env.consts.BAR_INDEX] - jnp.where(picked_from_bar & (player_idx == 0), 1, 0)
        black_bar = state.board[1, self.env.consts.BAR_INDEX] - jnp.where(picked_from_bar & (player_idx == 1), 1, 0)

        white_bar = jnp.maximum(white_bar, 0)
        black_bar = jnp.maximum(black_bar, 0)

        frame = self._draw_bar_checkers(frame, white_bar, black_bar)

        def draw_floating_checker(f):
            player_idx = self.env.get_player_index(state.current_player)
            color = jax.lax.cond(
                player_idx == 0,
                lambda _: self.color_white_checker,
                lambda _: self.color_black_checker,
                operand=None
            )

            pos = state.cursor_position
            is_home      = (pos == self.env.consts.HOME_INDEX)                 # 25
            is_bar_left  = (pos == jnp.int32(self.env.consts.BAR_INDEX))       # 24
            is_bar_right = (pos == jnp.int32(26))                               # split-bar right half (cursor-only)

            # Choose edge triangle for HOME anchor:
            # If picked from WHITE home (18..23) → use point 23; else use point 0.
            src = state.picked_checker_from
            use_right_edge = (src >= 18)  # white home points
            edge_idx = jax.lax.select(use_right_edge, jnp.int32(23), jnp.int32(0))

            # Triangle X center at chosen edge
            tri_x = self.triangle_positions[edge_idx][0] + self.triangle_length // 2

            # Place HOME just outside the correct band for the rotated layout:
            # - if edge_idx is in top half (6..17) → just above the top band
            # - else (0..5 or 18..23)              → just below the bottom band
            y_top_tri    = self.top_margin_for_dice + self.board_margin
            y_bottom_tri = self.frame_height - self.board_margin
            edge_is_top_half = ( (edge_idx >= 6) & (edge_idx <= 17) )

            home_cx = tri_x
            home_cy = jax.lax.select(
                edge_is_top_half,
                y_top_tri - (self.checker_height // 2) - 1,   # above top band
                y_bottom_tri + (self.checker_height // 2) + 1 # below bottom band
            )

            # Centers for split bar halves
            bar_left_cx  = self.bar_x + (self.bar_width // 4)
            bar_right_cx = self.bar_x + (3 * self.bar_width // 4)
            bar_cy       = self.bar_y + self.bar_thickness // 2

            # Compute draw position
            cx = jax.lax.cond(
                is_home,
                lambda _: home_cx,
                lambda _: jax.lax.cond(
                    pos < 24,  # triangles 0..23
                    lambda _: self.triangle_positions[pos][0] + self.triangle_length // 2,
                    lambda _: jax.lax.cond(
                        is_bar_left,
                        lambda _: bar_left_cx,          # left bar (24)
                        lambda _: jax.lax.cond(
                            is_bar_right,
                            lambda _: bar_right_cx,     # right bar (26)
                            lambda _: self.frame_width // 2,  # fallback (shouldn't hit)
                            operand=None
                        ),
                        operand=None
                    ),
                    operand=None
                ),
                operand=None
            )

            cy = jax.lax.cond(
                is_home,
                lambda _: home_cy,
                lambda _: jax.lax.cond(
                    pos < 24,  # triangles 0..23
                    lambda _: self.triangle_positions[pos][1] + self.triangle_thickness // 2,
                    lambda _: jax.lax.cond(
                        is_bar_left | is_bar_right,
                        lambda _: bar_cy,               # both bar halves
                        lambda _: self.bar_y + self.bar_thickness // 2,  # fallback
                        operand=None
                    ),
                    operand=None
                ),
                operand=None
            )

            return self._draw_rectangle(
                f,
                cx - self.checker_width // 2,
                cy - self.checker_height // 2,
                self.checker_width,
                self.checker_height,
                color
            )

        frame = jax.lax.cond(
            state.game_phase == 2,  # Only in MOVING_CHECKER phase
            draw_floating_checker,
            lambda f: f,
            operand=frame
        )

        frame = self._draw_dice(frame, state.dice, state.current_player)
        return frame
