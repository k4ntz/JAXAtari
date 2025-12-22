# Third party imports
import chex
import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple, Tuple, List, Optional
import os
from enum import IntEnum

# Project imports
from jaxatari.environment import JaxEnvironment, JAXAtariAction
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as jr
import jaxatari.spaces as spaces

"""
Contributors: Ayush Bansal, Mahta Mollaeian, Anh Tuan Nguyen, Abdallah Siwar, Pascha Sobouti  

Game: JAX Backgammon

This module defines a JAX-accelerated backgammon environment for reinforcement learning and simulation.
It includes the environment class, state structures, move validation and execution logic, rendering, and user interaction.
"""


class BackgammonConstants(NamedTuple):
    """Constants for game Environment.
    
    Board layout:
    - Points 0-23: Playable triangles
    - Point 24 (BAR_INDEX): Bar for hit checkers
    - Point 25 (HOME_INDEX): Borne-off checkers
    
    Home boards (for bearing off):
    - White home: Points 18-23 (WHITE_HOME_RANGE)
    - Black home: Points 0-5 (BLACK_HOME_RANGE)
    """
    NUM_POINTS = 24
    NUM_CHECKERS = 15
    BAR_INDEX = 24
    HOME_INDEX = 25
    # Home board ranges - used for bearing-off eligibility checks
    WHITE_HOME_RANGE = jnp.array(range(18, 24))
    BLACK_HOME_RANGE = jnp.array(range(0, 6))
    WHITE = 1
    BLACK = -1
    DOUBLING_CUBE = 1  # Multiplier for gammon/backgammon scoring


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
    """Phases of the interactive gameplay.
    
    Note: This enum is kept for documentation purposes. Inside @jit functions,
    we use jnp.int32 literals (0, 1, 2) directly because JAX traces don't 
    always handle Python enums correctly. The enum values here serve as the
    authoritative reference for what each phase number means.
    
    Phase flow:
    0 (WAITING_FOR_ROLL) -> FIRE -> 1 (SELECTING_CHECKER)
    1 (SELECTING_CHECKER) -> FIRE on checker -> 2 (MOVING_CHECKER)  
    2 (MOVING_CHECKER) -> FIRE on valid dest -> 1 or 0 (if turn ends)
    
    TURN_COMPLETE (3) was planned but not implemented - the game auto-advances
    to the next player when all dice are consumed.
    """
    WAITING_FOR_ROLL = 0
    SELECTING_CHECKER = 1
    MOVING_CHECKER = 2
    TURN_COMPLETE = 3  # Reserved for future use


# Movement paths for each player (point indices in logical move order)
# White moves: 0 -> 23 -> bear off (ascending point numbers)
# Black moves: 23 -> 0 -> bear off (descending point numbers)
WHITE_PATH = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
BLACK_PATH = [23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 24, 25]

# ============================================================================
# PRE-COMPUTED CURSOR NAVIGATION MAPS (computed at module load time)
# ============================================================================
# Rotated ring order: 0→1→2→3→4→5→26→6→7→8→9→10→11→12→13→14→15→16→17→24→18→19→20→21→22→23
_CURSOR_RING = jnp.array(
    [0, 1, 2, 3, 4, 5, 26, 6, 7, 8, 9, 10, 11,
     12, 13, 14, 15, 16, 17, 24, 18, 19, 20, 21, 22, 23],
    dtype=jnp.int32
)

def _build_cursor_maps():
    """Pre-compute LEFT/RIGHT navigation maps for the cursor ring."""
    ring = _CURSOR_RING
    ring_len = ring.shape[0]
    next_left = jnp.arange(27, dtype=jnp.int32)
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

# Build maps once at module load time
_CURSOR_NEXT_LEFT, _CURSOR_NEXT_RIGHT = _build_cursor_maps()


# Available board color themes
BACKGAMMON_THEMES = ["classic", "brown", "blue"]


class JaxBackgammonEnv(JaxEnvironment[BackgammonState, jnp.ndarray, dict, BackgammonConstants]):
    """
    JAX-based backgammon environment supporting JIT compilation and vectorized operations.
    Provides functionality for state initialization, step transitions, valid move evaluation, and observation generation.
    
    Args:
        consts: Game constants (optional, uses defaults if None)
        reward_funcs: Custom reward functions (optional)
        theme: Board color theme - one of "classic" (green), "brown" (wooden), "blue" (tournament)
    """

    def __init__(self, consts: BackgammonConstants = None, reward_funcs: list[callable] = None, theme: str = "classic"):
        consts = consts or BackgammonConstants()
        super().__init__(consts)
        
        # Validate and store theme
        if theme not in BACKGAMMON_THEMES:
            raise ValueError(f"Invalid theme '{theme}'. Must be one of: {BACKGAMMON_THEMES}")
        self.theme = theme

        # Pre-compute all possible moves (from_point, to_point) for points 0..25
        # Shape: (676, 2) = 26×26 combinations
        # This will be used for vectorized move validation in Phase 3 (sequence generation)
        NUM_ACTION_PAIRS = 26 * 26
        self._action_pairs = jnp.array([(i, j) for i in range(26) for j in range(26)], dtype=jnp.int32)
        assert self._action_pairs.shape == (NUM_ACTION_PAIRS, 2), \
            f"Action pairs shape mismatch: expected ({NUM_ACTION_PAIRS}, 2), got {self._action_pairs.shape}"

        # Special action indices for interactive play
        self._roll_action_index = self._action_pairs.shape[0]

        self.renderer = BackgammonRenderer(self, theme=theme)
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs

        # Define action set for jaxatari compatibility (interactive mode)
        # NOTE: For AI agent training (PPO/PQN), you can either:
        #   1. Use these interactive actions (LEFT/RIGHT/FIRE/NOOP) to simulate cursor movement
        #   2. Or add a direct move API: step_move(state, move_index) where move_index ∈ [0, 675]
        # Current implementation: Interactive mode
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
                
                # Bearing off distance calculation:
                # White: Point 18=6-pip, 19=5-pip, ..., 23=1-pip → distance = 24 - from_point
                # Black: Point 5=6-pip, 4=5-pip, ..., 0=1-pip   → distance = from_point + 1
                bearing_off_distance = jax.lax.cond(
                    player == self.consts.WHITE,
                    lambda _: 24 - from_point,  # FIX: war HOME_INDEX - from - 1
                    lambda _: from_point + 1,
                    operand=None
                )
                dice_match = jnp.any(state.dice == bearing_off_distance)

                # Check if higher checkers exist (for oversize rule)
                # "Higher" means: FURTHER from home (greater distance)
                def white_check():
                    # White: Point 23=1-pip (nearest), 22=2-pip, ..., 18=6-pip (farthest)
                    # Higher = greater distance = smaller point number
                    # Check if checkers exist on points < from_point (further away)
                    full_home = jax.lax.dynamic_slice(board[player_idx], (18,), (6,))
                    mask = (jnp.arange(18, 24) < from_point)
                    return jnp.any(full_home * mask > 0)
                
                def black_check():
                    # Black: Point 0=1-pip (nearest), 1=2-pip, ..., 5=6-pip (farthest)
                    # Higher = greater distance = greater point number
                    # Check if checkers exist on points > from_point (further away)
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

            # Higher-die rule: If only ONE die can be used, the HIGHER must be used
            # need_rule = True if: 2 different dice AND higher can be used AND lower cannot be used
            need_rule = has_two & can_hi & (~can_lo)

            def must_use_hi(_):
                # Check if current move uses the higher die
                state_hi = state._replace(dice=jnp.array([hi, 0, 0, 0], dtype=jnp.int32))
                uses_hi = self._is_valid_move_basic(state_hi, move)
                
                # If rule active: Move is only legal if it uses the higher die
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

    # ============================================================================
    # DIRECT MOVE API (For RL Training - bypasses cursor-based interaction)
    # ============================================================================

    @partial(jax.jit, static_argnums=(0,))
    def step_move(self, state: BackgammonState, move_idx: int) -> Tuple[jnp.ndarray, BackgammonState, float, bool, dict, jax.Array]:
        """
        Direct move API for RL agents - bypasses cursor-based interaction.
        
        Args:
            state: Current game state
            move_idx: Index into _action_pairs (0-675) for moves, or 676 for roll/pass
        
        Returns:
            Tuple of (observation, new_state, reward, done, info, new_key)
        
        Usage for RL:
            valid_mask = env.get_valid_action_mask(state)  # shape (677,)
            action = agent.select_action(obs, valid_mask)   # masked policy
            obs, state, reward, done, info, key = env.step_move(state, action)
        
        Move indexing:
            - move_idx 0-675: (from_point, to_point) pairs where from/to ∈ [0, 25]
            - move_idx 676: Roll dice (if in phase 0) or pass/confirm
        """
        is_roll_action = (move_idx == self._roll_action_index)
        
        def do_roll(_):
            # Roll dice and switch to selecting phase
            new_dice, new_key = self.roll_dice(state.key)
            new_state = state._replace(
                dice=new_dice,
                key=new_key,
                game_phase=jnp.int32(1),  # SELECTING_CHECKER
            )
            new_state = self._auto_pass_if_stuck(new_state)
            obs = self._get_observation(new_state)
            reward = self._get_reward(state, new_state)
            all_rewards = self._get_all_reward(state, new_state)
            done = self._get_done(new_state)
            info = self._get_info(new_state, all_rewards)
            return obs, new_state, reward, done, info, new_key
        
        def do_move(_):
            # Execute the move directly via step_impl
            move = self._action_pairs[move_idx]
            return self.step_impl(state, (move[0], move[1]), state.key)
        
        return jax.lax.cond(is_roll_action, do_roll, do_move, operand=None)

    @partial(jax.jit, static_argnums=(0,))
    def get_valid_action_mask(self, state: BackgammonState) -> jnp.ndarray:
        """
        Returns a boolean mask of valid actions for the current state.
        
        Shape: (677,) where:
            - indices 0-675: Valid (from, to) move pairs
            - index 676: Roll action (valid only in phase 0)
        
        Usage:
            mask = env.get_valid_action_mask(state)
            # In your agent: logits[~mask] = -inf before softmax
        """
        # Check all move pairs
        move_mask = jax.vmap(lambda mv: self.is_valid_move(state, mv))(self._action_pairs)
        
        # Roll action is valid only in phase 0 (IDLE/waiting for roll)
        roll_valid = (state.game_phase == jnp.int32(0))
        
        return jnp.concatenate([move_mask, jnp.array([roll_valid])])

    @partial(jax.jit, static_argnums=(0,))
    def get_legal_move_indices(self, state: BackgammonState) -> jnp.ndarray:
        """
        Returns indices of valid moves (not a mask, but actual indices).
        Useful for random action selection or debugging.
        
        Returns:
            Array of valid move indices. Padded with -1 for fixed shape.
        """
        mask = self.get_valid_action_mask(state)
        indices = jnp.arange(677)
        # Return indices where mask is True, padded with -1
        valid_indices = jnp.where(mask, indices, -1)
        return valid_indices

    @partial(jax.jit, static_argnums=(0,))
    def sample_random_action(self, state: BackgammonState, key: jax.Array) -> Tuple[int, jax.Array]:
        """
        Sample a random valid action uniformly.
        
        Args:
            state: Current game state
            key: JAX random key
        
        Returns:
            (action_index, new_key)
        """
        mask = self.get_valid_action_mask(state)
        key, subkey = jax.random.split(key)
        
        # Sample uniformly from valid actions
        # Use categorical with logits: valid=0, invalid=-inf
        logits = jnp.where(mask, 0.0, -jnp.inf)
        action = jax.random.categorical(subkey, logits)
        
        return action, key

    @property
    def num_actions(self) -> int:
        """Total number of possible actions (676 moves + 1 roll)."""
        return self._action_pairs.shape[0] + 1

    # ============================================================================
    # DECOMPOSED STEP HELPERS (Design Guide Section 4: Modular, JIT-compatible)
    # ============================================================================

    @partial(jax.jit, static_argnums=(0,))
    def _handle_cursor_move(self, state: BackgammonState, direction: int) -> BackgammonState:
        """
        Handle cursor movement (LEFT=-1 or RIGHT=1).
        
        Cursor moves on the rotated ring:
        LEFT  : 0→1→2→3→4→5→26→6→7→8→9→10→11→12→13→14→15→16→17→24→18→19→20→21→22→23
        RIGHT : reverse of the above.
        
        Rules:
        - Bars (24, 26) are cursor-only (you can stand on them but not drop there).
        - No wrap 0↔23 ever.
        - While MOVING at 0/23: blocked unless legal bear-off jump to HOME (25).
        - Leaving HOME (25) only back to its visual edge.
        
        Returns: Updated state with new cursor_position.
        """
        can_move = (state.game_phase == 1) | (state.game_phase == 2)
        pos = state.cursor_position

        is_left = jnp.array(direction == -1, dtype=jnp.bool_)
        is_right = jnp.array(direction == 1, dtype=jnp.bool_)

        at_home = (pos == jnp.int32(self.consts.HOME_INDEX))  # 25
        at_left_edge = (pos == jnp.int32(0))
        at_right_edge = (pos == jnp.int32(23))

        is_moving = (state.game_phase == jnp.int32(2))
        is_white = (state.current_player == self.consts.WHITE)

        # Use pre-computed module-level navigation maps
        next_left = _CURSOR_NEXT_LEFT
        next_right = _CURSOR_NEXT_RIGHT

        # Candidate target from ring movement
        ring_target = jax.lax.select(is_left, next_left[pos],
                        jax.lax.select(is_right, next_right[pos], pos))

        # ---- Bear-off jump to HOME (only when MOVING and bearing-off is allowed) ----
        not_from_bar = (state.picked_checker_from != jnp.int32(self.consts.BAR_INDEX))
        can_bear_off_now = self.check_bearing_off(state, state.current_player)
        jump_home_white = is_moving & is_left & at_right_edge & is_white
        jump_home_black = is_moving & is_right & at_left_edge & (~is_white)
        to_home_attempt = (jump_home_white | jump_home_black) & not_from_bar & can_bear_off_now
        home_target = jnp.int32(self.consts.HOME_INDEX)

        # ---- Block wrap 0↔23 always ----
        block_wrap = (at_left_edge & is_right) | (at_right_edge & is_left)
        ring_target = jax.lax.select(block_wrap, pos, ring_target)

        # ---- While MOVING on edges, block forbidden direction (unless home jump) ----
        block_edge_dir = is_moving & (
            (at_left_edge & is_right & (~to_home_attempt)) |
            (at_right_edge & is_left & (~to_home_attempt))
        )
        ring_target = jax.lax.select(block_edge_dir, pos, ring_target)

        # ---- Leaving HOME only to the correct edge ----
        from_home_target = jax.lax.cond(
            is_white,
            lambda _: jax.lax.select(is_left, jnp.int32(23), pos),  # White: 25 + LEFT → 23
            lambda _: jax.lax.select(is_right, jnp.int32(0), pos),   # Black: 25 + RIGHT → 0
            operand=None
        )

        # Final resolution
        target_if_move = jax.lax.cond(
            at_home,
            lambda _: from_home_target,
            lambda _: jax.lax.select(to_home_attempt, home_target, ring_target),
            operand=None
        )

        new_cursor = jax.lax.cond(can_move, lambda _: target_if_move, lambda _: pos, operand=None)
        return state._replace(cursor_position=new_cursor)

    @partial(jax.jit, static_argnums=(0,))
    def _handle_roll_dice(self, state: BackgammonState) -> BackgammonState:
        """
        Handle dice roll action (FIRE in phase 0).
        Rolls dice and transitions to SELECTING_CHECKER phase.
        Auto-passes if no legal moves available.
        
        Returns: Updated state with new dice and game_phase.
        """
        dice, key = self.roll_dice(state.key)
        new_state = state._replace(dice=dice, key=key, game_phase=1)
        # Auto-pass if no legal move with these dice
        return self._auto_pass_if_stuck(new_state)

    @partial(jax.jit, static_argnums=(0,))
    def _handle_pick_checker(self, state: BackgammonState) -> BackgammonState:
        """
        Handle checker selection (FIRE in phase 1).
        Picks up a checker at cursor position if player owns one there.
        
        Returns: Updated state with picked_checker_from set and phase=2 if valid.
        """
        player_idx = self.get_player_index(state.current_player)
        pos = state.cursor_position
        is_bar_cursor = (pos == jnp.int32(self.consts.BAR_INDEX)) | (pos == jnp.int32(26))

        selectable = jax.lax.cond(
            is_bar_cursor,
            lambda _: state.board[player_idx, self.consts.BAR_INDEX],
            lambda _: state.board[player_idx, pos],
            operand=None
        )
        has_checker = selectable > 0
        picked_from = jax.lax.select(is_bar_cursor, jnp.int32(self.consts.BAR_INDEX), pos)
        picked_bar_side = jax.lax.select(is_bar_cursor, pos, state.picked_bar_side)

        return jax.lax.cond(
            has_checker,
            lambda s: s._replace(
                picked_checker_from=picked_from,
                picked_bar_side=picked_bar_side,
                game_phase=2
            ),
            lambda s: s,
            operand=state
        )

    @partial(jax.jit, static_argnums=(0,))
    def _handle_drop_checker(self, state: BackgammonState) -> Tuple[BackgammonState, float, bool]:
        """
        Handle checker drop (FIRE in phase 2).
        Attempts to drop the picked checker at cursor position.
        If valid: executes move via step_impl, updates phase.
        If invalid: returns checker to origin, stays in phase 1.
        
        Returns: (new_state, reward, done)
        """
        move = (state.picked_checker_from, state.cursor_position)
        is_valid = self.is_valid_move(state, move)

        def execute_valid_move(s):
            obs, ns, reward, done, info, key = self.step_impl(s, move, s.key)
            all_dice_used = jnp.all(ns.dice == 0)
            next_phase = jax.lax.cond(all_dice_used, lambda _: 0, lambda _: 1, operand=None)
            next_cursor = jax.lax.cond(
                ns.current_player != s.current_player,
                lambda _: jax.lax.cond(ns.current_player == self.consts.WHITE, lambda _: 0, lambda _: 23, operand=None),
                lambda _: s.cursor_position,
                operand=None
            )
            final_state = ns._replace(
                picked_checker_from=-1,
                picked_bar_side=-1,
                game_phase=next_phase,
                cursor_position=next_cursor,
                key=key,
                last_valid_drop=s.cursor_position
            )
            return final_state, reward, done

        def invalid_drop(s):
            # Return checker to origin
            was_from_bar = (s.picked_checker_from == jnp.int32(self.consts.BAR_INDEX))
            fallback_cursor = jax.lax.cond(
                was_from_bar & (s.picked_bar_side >= 0),
                lambda _: s.picked_bar_side,
                lambda _: s.picked_checker_from,
                operand=None
            )
            ns = s._replace(
                picked_checker_from=-1,
                picked_bar_side=-1,
                game_phase=1,
                cursor_position=fallback_cursor
            )
            return ns, 0.0, False

        return jax.lax.cond(is_valid, execute_valid_move, invalid_drop, operand=state)

    @partial(jax.jit, static_argnums=(0,))
    def _handle_fire_action(self, state: BackgammonState) -> Tuple[BackgammonState, float, bool]:
        """
        Handle FIRE action by dispatching to the appropriate phase handler.
        Phase 0: Roll dice
        Phase 1: Pick checker
        Phase 2: Drop checker
        
        Returns: (new_state, reward, done)
        """
        def do_roll(s):
            ns = self._handle_roll_dice(s)
            return ns, 0.0, False

        def do_pick(s):
            ns = self._handle_pick_checker(s)
            return ns, 0.0, False

        def do_drop(s):
            return self._handle_drop_checker(s)

        return jax.lax.switch(state.game_phase, [do_roll, do_pick, do_drop], operand=state)

    @partial(jax.jit, static_argnums=(0,))
    def _process_action(self, state: BackgammonState, action: jnp.ndarray) -> Tuple[BackgammonState, float, bool]:
        """
        Process a single action and return new state.
        Dispatches to appropriate handler based on action type.
        
        Returns: (new_state, reward, done)
        """
        is_left = action == JAXAtariAction.LEFT
        is_right = action == JAXAtariAction.RIGHT
        is_fire = action == JAXAtariAction.FIRE

        def handle_left(s):
            ns = self._handle_cursor_move(s, -1)
            return ns, 0.0, False

        def handle_right(s):
            ns = self._handle_cursor_move(s, 1)
            return ns, 0.0, False

        def handle_noop(s):
            return s, 0.0, False

        # Nested dispatch: LEFT > RIGHT > FIRE > NOOP
        return jax.lax.cond(
            is_left,
            handle_left,
            lambda s: jax.lax.cond(
                is_right,
                handle_right,
                lambda s2: jax.lax.cond(
                    is_fire,
                    self._handle_fire_action,
                    handle_noop,
                    operand=s2
                ),
                operand=s
            ),
            operand=state
        )

    @partial(jax.jit, static_argnums=(0,))
    def _apply_debounce(self, state: BackgammonState, action: jnp.ndarray, 
                        new_state: BackgammonState) -> BackgammonState:
        """
        Apply debounce logic: arm await_keyup for LEFT/RIGHT/FIRE actions.
        
        Returns: State with debounce flags updated.
        """
        is_left = action == JAXAtariAction.LEFT
        is_right = action == JAXAtariAction.RIGHT
        is_fire = action == JAXAtariAction.FIRE
        should_arm = is_left | is_right | is_fire
        
        return jax.lax.cond(
            should_arm,
            lambda s: s._replace(await_keyup=True, last_action=action),
            lambda s: s,
            operand=new_state
        )

    @partial(jax.jit, static_argnums=(0,))
    def _handle_blocked_input(self, state: BackgammonState, action: jnp.ndarray) -> BackgammonState:
        """
        Handle input when awaiting key-up (debounce active).
        Only NOOP clears the debounce; other actions are ignored.
        
        Returns: State with await_keyup potentially cleared.
        """
        is_noop = action == JAXAtariAction.NOOP
        return jax.lax.cond(
            is_noop,
            lambda s: s._replace(await_keyup=False, last_action=JAXAtariAction.NOOP),
            lambda s: s,
            operand=state
        )

    # ============================================================================
    # MAIN STEP FUNCTION (Orchestrator Pattern - Design Guide Section 5)
    # ============================================================================

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: BackgammonState, action: jnp.ndarray):
        """
        Interactive step with JAX-safe debounce.
        
        This is the main entry point for gameplay. It follows the orchestrator pattern:
        - Delegates to helper functions for specific logic
        - Handles debounce for key-press/release behavior
        - Returns (observation, new_state, reward, done, info)
        
        Each key press triggers once; holding does nothing until NOOP arrives.
        """
        # Branch 1: Debounce active - wait for key release (NOOP)
        def when_blocked(s):
            ns = self._handle_blocked_input(s, action)
            return self._get_observation(ns), ns, 0.0, False, self._get_info(ns)

        # Branch 2: Ready for input - process action and arm debounce
        def when_free(s):
            ns, reward, done = self._process_action(s, action)
            ns = self._apply_debounce(s, action, ns)
            obs = self._get_observation(ns)
            info = self._get_info(ns)
            return obs, ns, reward, done, info

        return jax.lax.cond(state.await_keyup, when_blocked, when_free, operand=state)


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
        """Direct Move API (RL-optimized) — Use 677 actions"""
        return spaces.Discrete(self.num_actions)  # = 677
        """Interactive mode (cursor-based) — Use 4 actions"""
        #return spaces.Discrete(len(self.action_set))  # LEFT, RIGHT, FIRE, NOOP

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


# ============================================================================
# RENDERER (Using JaxRenderingUtils - Design Guide + Renderer Guide)
# ============================================================================

class BackgammonRenderer(JAXGameRenderer):
    """
    JAX-native Backgammon renderer using JaxRenderingUtils.
    
    Follows the palette-based rendering pattern from Renderer Guide:
    1. Load sprites and create palette in __init__
    2. render() creates object_raster, stamps sprites, converts to RGB
    
    Supports multiple color themes:
    - "classic": Original green theme
    - "brown": Wooden/brown theme  
    - "blue": Tournament/blue theme
    """
    
    def __init__(self, env=None, theme: str = "classic"):
        super().__init__()
        self.env = env
        self.theme = theme
        
        # Frame dimensions
        self.frame_height = 210
        self.frame_width = 160
        
        # Geometry constants (same as original)
        self.top_margin_for_dice = 25
        self.board_margin = 8
        self.triangle_length = 60
        self.triangle_thickness = 12
        self.bar_thickness = 14
        self.checker_width = 4
        self.checker_height = 4
        self.base_margin = 2
        self.band_top_margin = 1.2
        self.band_bottom_margin = 1
        self.chip_gap_y = 2
        self.chip_gap_x = 4
        self.checker_stack_offset = self.checker_height + self.chip_gap_y
        self.bar_edge_padding = 2
        self.bar_vertical_padding = 2
        self.dice_size = 12
        self.pip_size = 2
        
        # Computed positions
        self.bar_y = self.top_margin_for_dice + self.frame_height // 2 - self.bar_thickness // 2 - 10
        self.bar_x = self.board_margin
        self.bar_width = self.frame_width - 2 * self.board_margin
        
        # Pre-compute triangle positions
        self.triangle_positions = self._compute_triangle_positions()
        
        # Setup JaxRenderingUtils
        from jaxatari.rendering import jax_rendering_utils as render_utils
        
        self.config = render_utils.RendererConfig(
            game_dimensions=(self.frame_height, self.frame_width),
            channels=3,
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)
        
        # Load assets from theme folder
        base_sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/backgammon"
        sprite_path = f"{base_sprite_path}/themes/{self.theme}"
        
        # Fallback to base folder if theme folder doesn't exist
        if not os.path.exists(sprite_path):
            print(f"Warning: Theme folder '{self.theme}' not found, using base sprites")
            sprite_path = base_sprite_path
            
        asset_config = self._get_asset_config()
        
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(asset_config, sprite_path)
    
    def _get_asset_config(self) -> list:
        """Returns the declarative manifest of all assets for backgammon."""
        return [
            {'name': 'background', 'type': 'background', 'file': 'background.npy'},
            # Checkers
            {'name': 'white_checker', 'type': 'single', 'file': 'white_checker.npy'},
            {'name': 'black_checker', 'type': 'single', 'file': 'black_checker.npy'},
            {'name': 'highlight_checker', 'type': 'single', 'file': 'highlight_checker.npy'},
            # Triangles (6 variants: light/dark × left/right + highlight × left/right)
            {'name': 'triangle_light_right', 'type': 'single', 'file': 'triangle_light_right.npy'},
            {'name': 'triangle_dark_right', 'type': 'single', 'file': 'triangle_dark_right.npy'},
            {'name': 'triangle_light_left', 'type': 'single', 'file': 'triangle_light_left.npy'},
            {'name': 'triangle_dark_left', 'type': 'single', 'file': 'triangle_dark_left.npy'},
            {'name': 'triangle_highlight_right', 'type': 'single', 'file': 'triangle_highlight_right.npy'},
            {'name': 'triangle_highlight_left', 'type': 'single', 'file': 'triangle_highlight_left.npy'},
            # Bar highlights
            {'name': 'bar_highlight_left', 'type': 'single', 'file': 'bar_highlight_left.npy'},
            {'name': 'bar_highlight_right', 'type': 'single', 'file': 'bar_highlight_right.npy'},
            # Dice
            {'name': 'die_white', 'type': 'single', 'file': 'die_white.npy'},
            {'name': 'die_red', 'type': 'single', 'file': 'die_red.npy'},
            {'name': 'pip_black', 'type': 'single', 'file': 'pip_black.npy'},
            {'name': 'pip_white', 'type': 'single', 'file': 'pip_white.npy'},
        ]
    
    def _compute_triangle_positions(self):
        """Compute (x, y) positions for all 24 triangles + bar positions."""
        left_x = self.board_margin
        right_x = self.frame_width - self.board_margin - self.triangle_length
        
        y_top = self.top_margin_for_dice + self.board_margin
        y_bottom = self.frame_height - self.board_margin
        band_h = 6 * self.triangle_thickness
        
        positions = []
        
        # 0..5 (lower-right) bottom -> top
        for i in range(6):
            y = y_bottom - (i + 1) * self.triangle_thickness
            positions.append((right_x, y))
        
        # 6..11 (upper-right) bottom -> top
        for i in range(6):
            y = y_top + (5 - i) * self.triangle_thickness
            positions.append((right_x, y))
        
        # 12..17 (upper-left) top -> bottom
        for i in range(6):
            y = y_top + i * self.triangle_thickness
            positions.append((left_x, y))
        
        # 18..23 (lower-left) top -> bottom
        lower_left_top_y = y_bottom - band_h
        for i in range(6):
            y = lower_left_top_y + i * self.triangle_thickness
            positions.append((left_x, y))
        
        # bar-left (24): center of left half of the bar
        bar_left_x = self.bar_x + (self.bar_width // 4)
        bar_center_y = self.bar_y + self.bar_thickness // 2
        positions.append((bar_left_x, bar_center_y))
        
        return jnp.array(positions, dtype=jnp.int32)
    
    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: BackgammonState) -> jnp.ndarray:
        """
        Render the current game state using palette-based rendering.
        
        Pattern: create_object_raster -> render_at (sprites) -> render_from_palette
        """
        # 1. Start with background
        raster = self.jr.create_object_raster(self.BACKGROUND)
        
        # 2. Draw triangles
        raster = self._draw_all_triangles(raster, state)
        
        # 3. Draw highlight (cursor or picked position)
        raster = self._draw_highlight(raster, state)
        
        # 4. Draw checkers on all points
        raster = self._draw_all_checkers(raster, state)
        
        # 5. Draw bar checkers
        raster = self._draw_bar_checkers(raster, state)
        
        # 6. Draw floating checker (if in MOVING phase)
        raster = self._draw_floating_checker(raster, state)
        
        # 7. Draw dice
        raster = self._draw_dice(raster, state)
        
        # 8. Convert to RGB
        return self.jr.render_from_palette(raster, self.PALETTE)
    
    @partial(jax.jit, static_argnums=(0,))
    def _draw_all_triangles(self, raster, state):
        """Draw all 24 triangles with correct colors."""
        left_x = self.board_margin
        
        def draw_triangle_at_index(i, r):
            pos = self.triangle_positions[i]
            x, y = pos[0], pos[1]
            
            is_left_column = (x == left_x)
            band_idx = jnp.where(i < 6, i,
                        jnp.where(i < 12, i - 6,
                        jnp.where(i < 18, i - 12, i - 18)))
            band_id = jnp.where(i < 6, 0,
                        jnp.where(i < 12, 1,
                        jnp.where(i < 18, 2, 3)))
            
            is_upper_band = ((i >= 6) & (i < 12)) | ((i >= 12) & (i < 18))
            start_light = jnp.where(
                is_left_column,
                jnp.where(is_upper_band, True, False),
                jnp.where(is_upper_band, False, True)
            )
            flip = (band_id == 1) | (band_id == 3)
            start_light = jnp.logical_xor(start_light, flip)
            use_light = jnp.where(start_light, (band_idx % 2 == 0), (band_idx % 2 == 1))
            
            # Select triangle mask based on color and direction
            # Left column: point_right=True, Right column: point_right=False
            mask = jax.lax.cond(
                is_left_column,
                lambda _: jax.lax.select(use_light, 
                    self.SHAPE_MASKS["triangle_light_right"],
                    self.SHAPE_MASKS["triangle_dark_right"]),
                lambda _: jax.lax.select(use_light,
                    self.SHAPE_MASKS["triangle_light_left"],
                    self.SHAPE_MASKS["triangle_dark_left"]),
                operand=None
            )
            
            return self.jr.render_at(r, x, y, mask)
        
        return jax.lax.fori_loop(0, 24, draw_triangle_at_index, raster)
    
    @partial(jax.jit, static_argnums=(0,))
    def _draw_highlight(self, raster, state):
        """Draw highlight on cursor position or picked checker origin."""
        
        def get_highlight_index(_):
            def moving_origin(_):
                return jax.lax.cond(
                    state.picked_checker_from == jnp.int32(self.env.consts.BAR_INDEX),
                    lambda __: jax.lax.select(
                        state.picked_bar_side == jnp.int32(26),
                        jnp.int32(26),
                        jnp.int32(self.env.consts.BAR_INDEX)
                    ),
                    lambda __: jnp.int32(state.picked_checker_from),
                    operand=None
                )
            
            return jax.lax.cond(
                state.game_phase == jnp.int32(2),
                moving_origin,
                lambda __: jax.lax.cond(
                    state.game_phase == jnp.int32(1),
                    lambda ___: jnp.int32(state.cursor_position),
                    lambda ___: jnp.int32(state.last_valid_drop),
                    operand=None
                ),
                operand=None
            )
        
        hi = get_highlight_index(None)
        no_hi = (hi < 0)
        
        def add_highlight(r):
            def on_triangle(r2):
                pos = self.triangle_positions[hi]
                is_left = (pos[0] == self.board_margin)
                mask = jax.lax.select(is_left,
                    self.SHAPE_MASKS["triangle_highlight_right"],
                    self.SHAPE_MASKS["triangle_highlight_left"])
                return self.jr.render_at(r2, pos[0], pos[1], mask)
            
            def on_bar_left(r2):
                return self.jr.render_at(r2, self.bar_x, self.bar_y, 
                    self.SHAPE_MASKS["bar_highlight_left"])
            
            def on_bar_right(r2):
                half_w = self.bar_width // 2
                return self.jr.render_at(r2, self.bar_x + half_w, self.bar_y,
                    self.SHAPE_MASKS["bar_highlight_right"])
            
            def noop(r2):
                return r2
            
            return jax.lax.cond(
                hi < jnp.int32(24),
                on_triangle,
                lambda r2: jax.lax.cond(
                    hi == jnp.int32(self.env.consts.BAR_INDEX),
                    on_bar_left,
                    lambda r3: jax.lax.cond(hi == jnp.int32(26), on_bar_right, noop, operand=r3),
                    operand=r2
                ),
                operand=r
            )
        
        return jax.lax.cond(no_hi, lambda r: r, add_highlight, operand=raster)
    
    @partial(jax.jit, static_argnums=(0,))
    def _draw_all_checkers(self, raster, state):
        """Draw checkers on all 24 points."""
        
        def draw_point_checkers(point_idx, r):
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
            
            return self._draw_checkers_on_point(r, point_idx, white_count, black_count)
        
        return jax.lax.fori_loop(0, 24, draw_point_checkers, raster)
    
    @partial(jax.jit, static_argnums=(0,))
    def _draw_checkers_on_point(self, raster, point_idx, white_count, black_count):
        """Draw checker stacks on a single point."""
        pos = self.triangle_positions[point_idx]
        x, y = pos[0], pos[1]
        
        is_left_column = (x == self.board_margin)
        dir_sign = jnp.where(is_left_column, 1, -1)
        
        base_x = jnp.where(
            is_left_column,
            x + self.base_margin,
            x + self.triangle_length - self.checker_width - self.base_margin
        )
        
        band_bottom = y + self.triangle_thickness - self.band_bottom_margin
        row0_y = jnp.int32(band_bottom - self.checker_height)
        row1_y = jnp.int32(row0_y - self.checker_stack_offset)
        col_step = self.checker_width + self.chip_gap_x
        
        def draw_stack(r, count, mask):
            def draw_single(i, f):
                col = i // 2
                row = i & 1
                cx = base_x + dir_sign * (col * col_step)
                cy = jnp.where(row == 0, row0_y, row1_y)
                return self.jr.render_at(f, cx, cy, mask)
            return jax.lax.fori_loop(0, count, draw_single, r)
        
        raster = draw_stack(raster, white_count, self.SHAPE_MASKS["white_checker"])
        raster = draw_stack(raster, black_count, self.SHAPE_MASKS["black_checker"])
        return raster
    
    @partial(jax.jit, static_argnums=(0,))
    def _draw_bar_checkers(self, raster, state):
        """Draw checkers on the bar."""
        player_idx = self.env.get_player_index(state.current_player)
        picked_from_bar = (state.game_phase == 2) & (state.picked_checker_from == self.env.consts.BAR_INDEX)
        
        white_bar = state.board[0, self.env.consts.BAR_INDEX] - jnp.where(picked_from_bar & (player_idx == 0), 1, 0)
        black_bar = state.board[1, self.env.consts.BAR_INDEX] - jnp.where(picked_from_bar & (player_idx == 1), 1, 0)
        white_bar = jnp.maximum(white_bar, 0)
        black_bar = jnp.maximum(black_bar, 0)
        
        col_step = self.checker_width + self.chip_gap_x
        row_step = self.checker_stack_offset
        two_cols_w = 2 * self.checker_width + self.chip_gap_x
        
        base_x_left = self.bar_x + self.bar_edge_padding
        base_x_right = self.bar_x + self.bar_width - self.bar_edge_padding - two_cols_w
        base_y_bottom = self.bar_y + self.bar_thickness - self.bar_vertical_padding - self.checker_height
        
        def draw_stack(r, count, base_x, mask):
            def draw_single(i, f):
                row = i // 2
                col = i % 2
                x = base_x + col * col_step
                y = base_y_bottom - row * row_step
                return self.jr.render_at(f, x, y, mask)
            return jax.lax.fori_loop(0, count, draw_single, r)
        
        raster = draw_stack(raster, black_bar, base_x_left, self.SHAPE_MASKS["black_checker"])
        raster = draw_stack(raster, white_bar, base_x_right, self.SHAPE_MASKS["white_checker"])
        return raster
    
    @partial(jax.jit, static_argnums=(0,))
    def _draw_floating_checker(self, raster, state):
        """Draw the checker being moved (floating at cursor position)."""
        
        def draw_checker(r):
            player_idx = self.env.get_player_index(state.current_player)
            mask = jax.lax.select(player_idx == 0,
                self.SHAPE_MASKS["white_checker"],
                self.SHAPE_MASKS["black_checker"])
            
            pos = state.cursor_position
            is_home = (pos == self.env.consts.HOME_INDEX)
            is_bar_left = (pos == jnp.int32(self.env.consts.BAR_INDEX))
            is_bar_right = (pos == jnp.int32(26))
            
            # Compute position
            y_top_tri = self.top_margin_for_dice + self.board_margin
            y_bottom_tri = self.frame_height - self.board_margin
            
            src = state.picked_checker_from
            use_right_edge = (src >= 18)
            edge_idx = jax.lax.select(use_right_edge, jnp.int32(23), jnp.int32(0))
            tri_x = self.triangle_positions[edge_idx][0] + self.triangle_length // 2
            
            edge_is_top_half = ((edge_idx >= 6) & (edge_idx <= 17))
            home_cx = tri_x
            home_cy = jax.lax.select(
                edge_is_top_half,
                y_top_tri - (self.checker_height // 2) - 1,
                y_bottom_tri + (self.checker_height // 2) + 1
            )
            
            bar_left_cx = self.bar_x + (self.bar_width // 4)
            bar_right_cx = self.bar_x + (3 * self.bar_width // 4)
            bar_cy = self.bar_y + self.bar_thickness // 2
            
            cx = jax.lax.cond(
                is_home,
                lambda _: home_cx,
                lambda _: jax.lax.cond(
                    pos < 24,
                    lambda _: self.triangle_positions[pos][0] + self.triangle_length // 2,
                    lambda _: jax.lax.cond(
                        is_bar_left,
                        lambda _: bar_left_cx,
                        lambda _: jax.lax.cond(is_bar_right, lambda _: bar_right_cx, 
                            lambda _: self.frame_width // 2, operand=None),
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
                    pos < 24,
                    lambda _: self.triangle_positions[pos][1] + self.triangle_thickness // 2,
                    lambda _: jax.lax.cond(
                        is_bar_left | is_bar_right,
                        lambda _: bar_cy,
                        lambda _: self.bar_y + self.bar_thickness // 2,
                        operand=None
                    ),
                    operand=None
                ),
                operand=None
            )
            
            return self.jr.render_at(r, cx - self.checker_width // 2, cy - self.checker_height // 2, mask)
        
        return jax.lax.cond(
            state.game_phase == 2,
            draw_checker,
            lambda r: r,
            operand=raster
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def _draw_dice(self, raster, state):
        """Draw dice above the board."""
        total_width = 4 * (self.dice_size + 3)
        start_x = self.frame_width // 2 - total_width // 2 + 12
        dice_y = self.board_margin
        
        is_white_player = (state.current_player == self.env.consts.WHITE)
        die_mask = jax.lax.select(is_white_player,
            self.SHAPE_MASKS["die_white"],
            self.SHAPE_MASKS["die_red"])
        pip_mask = jax.lax.select(is_white_player,
            self.SHAPE_MASKS["pip_black"],
            self.SHAPE_MASKS["pip_white"])
        
        def draw_single_die(i, r):
            val = state.dice[i]
            dx = start_x + i * (self.dice_size + 3)
            
            def draw_val(_):
                # Draw die background
                r2 = self.jr.render_at(r, dx, dice_y, die_mask)
                
                # Draw pips based on value
                center_x = dx + self.dice_size // 2
                center_y = dice_y + self.dice_size // 2
                
                def dot(f, x, y):
                    return self.jr.render_at(f, x - self.pip_size // 2, y - self.pip_size // 2, pip_mask)
                
                # Pip patterns for values 1-6
                def p1(_): return dot(r2, center_x, center_y)
                def p2(_):
                    f = dot(r2, center_x - 3, center_y - 3)
                    return dot(f, center_x + 3, center_y + 3)
                def p3(_):
                    f = dot(r2, center_x - 3, center_y - 3)
                    f = dot(f, center_x, center_y)
                    return dot(f, center_x + 3, center_y + 3)
                def p4(_):
                    f = dot(r2, center_x - 3, center_y - 3)
                    f = dot(f, center_x + 3, center_y - 3)
                    f = dot(f, center_x - 3, center_y + 3)
                    return dot(f, center_x + 3, center_y + 3)
                def p5(_):
                    f = p4(None)
                    return dot(f, center_x, center_y)
                def p6(_):
                    f = dot(r2, center_x - 3, center_y - 3)
                    f = dot(f, center_x - 3, center_y)
                    f = dot(f, center_x - 3, center_y + 3)
                    f = dot(f, center_x + 3, center_y - 3)
                    f = dot(f, center_x + 3, center_y)
                    return dot(f, center_x + 3, center_y + 3)
                
                funcs = [p1, p2, p3, p4, p5, p6]
                return jax.lax.switch(jnp.clip(val - 1, 0, 5), funcs, operand=None)
            
            return jax.lax.cond(val > 0, draw_val, lambda _: r, operand=None)
        
        return jax.lax.fori_loop(0, 4, draw_single_die, raster)
