import chex
import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple, List, Optional, Any
import os
from flax import struct

# Project imports
from jaxatari.environment import JaxEnvironment, JAXAtariAction, ObjectObservation
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as jr
from jaxatari.modification import AutoDerivedConstants
import jaxatari.spaces as spaces

"""
Contributors: Ayush Bansal, Mahta Mollaeian, Anh Tuan Nguyen, Abdallah Siwar, Pascha Sobouti  

Game: JAX Backgammon

This module defines a JAX-accelerated backgammon environment for reinforcement learning and simulation.
It includes the environment class, state structures, move validation and execution logic, rendering, and user interaction.
"""


def _get_asset_config() -> tuple:
    """Returns the declarative manifest of all assets for backgammon."""
    return (
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
    )


_RIGHT_BAR_INDEX = 26

class BackgammonConstants(AutoDerivedConstants):
    """Constants for game Environment.
    
    Board layout:
    - Points 0-23: Playable triangles
    - Point 24 (BAR_INDEX): Bar for hit checkers
    - Point 25 (HOME_INDEX): Borne-off checkers
    
    Home boards (for bearing off):
    - White home: Points 18-23 (WHITE_HOME_RANGE)
    - Black home: Points 0-5 (BLACK_HOME_RANGE)
    """
    NUM_POINTS: int = struct.field(pytree_node=False, default=24)
    NUM_CHECKERS: int = struct.field(pytree_node=False, default=15)
    BAR_INDEX: int = struct.field(pytree_node=False, default=24)
    RIGHT_BAR_INDEX: int = struct.field(pytree_node=False, default=_RIGHT_BAR_INDEX)
    HOME_INDEX: int = struct.field(pytree_node=False, default=25)
    # Home board ranges - used for bearing-off eligibility checks
    WHITE_HOME_RANGE: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array(range(18, 24)))
    BLACK_HOME_RANGE: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array(range(0, 6)))
    WHITE: int = struct.field(pytree_node=False, default=1)
    BLACK: int = struct.field(pytree_node=False, default=-1)
    DOUBLING_CUBE: int = struct.field(pytree_node=False, default=1)  # Multiplier for gammon/backgammon scoring

    # Theme for board colors: "classic", "brown", "blue"
    THEME: str = struct.field(pytree_node=False, default="classic")

    # Frame dimensions
    FRAME_HEIGHT: int = struct.field(pytree_node=False, default=210)
    FRAME_WIDTH: int = struct.field(pytree_node=False, default=160)

    # Geometry constants (same as original)
    TOP_MARGIN_FOR_DICE: int = struct.field(pytree_node=False, default=25)
    BOARD_MARGIN: int = struct.field(pytree_node=False, default=8)
    TRIANGLE_LENGTH: int = struct.field(pytree_node=False, default=60)
    TRIANGLE_THICKNESS: int = struct.field(pytree_node=False, default=12)
    BAR_THICKNESS: int = struct.field(pytree_node=False, default=14)
    CHECKER_WIDTH: int = struct.field(pytree_node=False, default=4)
    CHECKER_HEIGHT: int = struct.field(pytree_node=False, default=4)
    BASE_MARGIN: int = struct.field(pytree_node=False, default=2)
    BAND_TOP_MARGIN: float = struct.field(pytree_node=False, default=1.2)
    BAND_BOTTOM_MARGIN: int = struct.field(pytree_node=False, default=1)
    CHIP_GAP_Y: int = struct.field(pytree_node=False, default=2)
    CHIP_GAP_X: int = struct.field(pytree_node=False, default=4)
    BAR_EDGE_PADDING: int = struct.field(pytree_node=False, default=2)
    BAR_VERTICAL_PADDING: int = struct.field(pytree_node=False, default=2)
    DICE_SIZE: int = struct.field(pytree_node=False, default=12)
    PIP_SIZE: int = struct.field(pytree_node=False, default=2)

    # Derived constants (computed from base constants via compute_derived)
    CHECKER_STACK_OFFSET: Optional[int] = struct.field(pytree_node=False, default=None)
    BAR_Y: Optional[int] = struct.field(pytree_node=False, default=None)
    BAR_X: Optional[int] = struct.field(pytree_node=False, default=None)
    BAR_WIDTH: Optional[int] = struct.field(pytree_node=False, default=None)

    ASSET_CONFIG: tuple = struct.field(pytree_node=False, default_factory=_get_asset_config)

    def compute_derived(self):
        """Compute derived constants from base geometry constants."""
        return {
            'CHECKER_STACK_OFFSET': self.CHECKER_HEIGHT + self.CHIP_GAP_Y,
            'BAR_Y': self.TOP_MARGIN_FOR_DICE + self.FRAME_HEIGHT // 2 - self.BAR_THICKNESS // 2 - 10,
            'BAR_X': self.BOARD_MARGIN,
            'BAR_WIDTH': self.FRAME_WIDTH - 2 * self.BOARD_MARGIN,
        }


@struct.dataclass
class BackgammonState:
    """Represents the complete state of a backgammon game."""
    board: jnp.ndarray  # (2, 26)
    dice: jnp.ndarray  # (4,) - remaining moves available
    original_dice: jnp.ndarray  # (2,) - original roll values for display (ALE shows these)
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
    picked_bar_side: int = -1  # BAR_INDEX (left), RIGHT_BAR_INDEX (right), or -1 if not from bar
    move_repeat_timer: int = 0  # Frames since last cursor move (for hold-to-repeat)


@struct.dataclass
class BackgammonInfo:
    """Contains auxiliary information about the environment (e.g., timing or metadata)."""
    player: jnp.ndarray
    dice: jnp.ndarray
    all_rewards: chex.Array

@struct.dataclass
class BackgammonObservation:
    """Object-centric styled observation of the game."""
    board: jnp.ndarray
    dice: jnp.ndarray
    current_player: jnp.ndarray
    is_game_over: jnp.ndarray
    bar_counts: jnp.ndarray
    home_counts: jnp.ndarray
    cursor_position: jnp.ndarray
    game_phase: jnp.ndarray
    picked_checker_from: jnp.ndarray
    last_valid_drop: jnp.ndarray


# ============================================================================
# PRE-COMPUTED CURSOR NAVIGATION MAPS (computed at module load time)
# ============================================================================
# Rotated ring order: 0→1→2→3→4→5→RIGHT_BAR_INDEX→6→7→8→9→10→11→12→13→14→15→16→17→24→18→19→20→21→22→23
_CURSOR_RING = jnp.array(
    [0, 1, 2, 3, 4, 5, _RIGHT_BAR_INDEX, 6, 7, 8, 9, 10, 11,
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


class JaxBackgammonEnv(JaxEnvironment[BackgammonState, BackgammonObservation, BackgammonInfo, BackgammonConstants]):
    """
    JAX-based backgammon environment supporting JIT compilation and vectorized operations.
    Provides functionality for state initialization, step transitions, valid move evaluation, and observation generation.
    
    Args:
        consts: Game constants (optional, uses defaults if None)
        reward_funcs: Custom reward functions (optional)
        theme: Board color theme - one of "classic" (green), "brown" (wooden), "blue" (tournament)
               If None, uses consts.THEME (which defaults to "classic")
    """

    ACTION_SET: jnp.ndarray = jnp.array(
        [
            JAXAtariAction.NOOP,
            JAXAtariAction.FIRE,
            JAXAtariAction.RIGHT,
            JAXAtariAction.LEFT,
        ],
        dtype=jnp.int32,
    )

    def __init__(self, consts: BackgammonConstants = None, config: jr.RendererConfig = None):
        self.consts = consts or BackgammonConstants()
        super().__init__(self.consts)

        # Pre-compute all possible moves (from_point, to_point) for points 0..25
        # Shape: (676, 2) = 26×26 combinations
        # This will be used for vectorized move validation
        NUM_ACTION_PAIRS = 26 * 26
        self._action_pairs = jnp.array([(i, j) for i in range(26) for j in range(26)], dtype=jnp.int32)
        assert self._action_pairs.shape == (NUM_ACTION_PAIRS, 2), \
            f"Action pairs shape mismatch: expected ({NUM_ACTION_PAIRS}, 2), got {self._action_pairs.shape}"

        # Special action indices for interactive play
        self._roll_action_index = self._action_pairs.shape[0]

        self.renderer = BackgammonRenderer(self, config=config)
        self.reward_funcs = None

        # Lowercase alias retained for compatibility with older helper scripts.
        self.action_set = list(self.ACTION_SET)

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

        # Store original dice for display (always 2 values)
        original_dice = jnp.array([first, second], dtype=jnp.int32)

        return BackgammonState(
            board=board,
            dice=dice,
            original_dice=original_dice,
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
            move_repeat_timer=0,
        )

    def reset(self, key: jax.random.PRNGKey = None) -> Tuple[BackgammonObservation, BackgammonState]:
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
        return jnp.where(player == self.consts.WHITE, 0, 1)

    @partial(jax.jit, static_argnums=(0,))
    def _is_valid_move_basic(self, state: BackgammonState, move: Tuple[int, int]) -> bool:
        """Branchless move validation — uses jnp.where instead of nested lax.cond
        to avoid 2^N path explosion under vmap."""
        from_point = jnp.asarray(move[0], dtype=jnp.int32)
        to_point   = jnp.asarray(move[1], dtype=jnp.int32)
        board      = state.board
        player_idx = jnp.where(state.current_player == self.consts.WHITE, 0, 1)
        opp_idx    = 1 - player_idx
        is_white   = (state.current_player == self.consts.WHITE)

        has_bar         = board[player_idx, 24] > 0
        moving_from_bar = (from_point == 24)
        to_home         = (to_point == 25)

        # ── fast-reject without branching ────────────────────────────────────────
        in_bounds  = (from_point >= 0) & (from_point <= 24) & (to_point >= 0) & \
                     (to_point <= 25) & (to_point != 24)
        must_bar   = has_bar & (~moving_from_bar)
        early_fail = (~in_bounds) | must_bar | (from_point == to_point)

        # ── compute ALL three case results unconditionally ────────────────────────

        # Case A: bar entry
        fp_safe     = jnp.clip(from_point, 0, 23)
        tp_safe     = jnp.clip(to_point, 0, 23)
        def is_valid_bar_entry(dv):
            expected_entry = jnp.where(is_white, dv - 1, jnp.int32(24) - dv)
            matches_entry  = (to_point == expected_entry)
            safe_entry     = jnp.clip(expected_entry, 0, 23)
            entry_open     = board[opp_idx, safe_entry] <= 1
            return matches_entry & entry_open
        bar_valid   = (board[player_idx, 24] > 0) & jnp.any(jax.vmap(is_valid_bar_entry)(state.dice))

        # Case B: normal move
        has_piece   = board[player_idx, fp_safe] > 0
        raw_dist    = jnp.where(is_white, to_point - from_point, from_point - to_point)
        dest_in_bd  = (to_point >= 0) & (to_point <= 23)
        dest_open   = jnp.where(dest_in_bd, board[opp_idx, tp_safe] <= 1, jnp.bool_(False))
        dice_match  = jnp.any(state.dice == raw_dist)
        normal_valid = has_piece & dest_in_bd & dest_open & (raw_dist > 0) & dice_match

        # Case C: bearing off
        can_bear    = self.check_bearing_off(state, state.current_player)
        bear_dist   = jnp.where(is_white, jnp.int32(24) - from_point, from_point + jnp.int32(1))
        exact_die   = jnp.any(state.dice == bear_dist)

        # higher_exists: vectorised prefix/suffix, no cond
        wh          = board[player_idx, 18:24]
        w_higher    = jnp.any(wh * (jnp.arange(18, 24) < from_point) > 0)
        bh          = board[player_idx, 0:6]
        b_higher    = jnp.any(bh * (jnp.arange(0, 6) > from_point) > 0)
        higher      = jnp.where(is_white, w_higher, b_higher)

        oversize    = (~higher) & jnp.any(state.dice > bear_dist)
        in_home     = jnp.where(is_white, (from_point >= 18) & (from_point <= 23),
                                           (from_point >= 0)  & (from_point <= 5))
        bear_valid  = can_bear & has_piece & in_home & (exact_die | oversize)

        # ── select result with jnp.where (no cond nesting) ───────────────────────
        result = jnp.where(moving_from_bar, bar_valid,
                   jnp.where(to_home, bear_valid, normal_valid))

        return (~early_fail) & result

    @partial(jax.jit, static_argnums=(0,))
    def _distinct_nonzero_dice(self, dice: jnp.ndarray):
        nz = dice * (dice > 0)                   
        hi = jnp.max(nz).astype(jnp.int32)
        lo = jnp.max(jnp.where((nz > 0) & (nz < hi), nz, 0)).astype(jnp.int32)
        has_two = (lo > 0) & (hi > lo)
        return has_two, lo, hi

    @partial(jax.jit, static_argnums=(0,))
    def _candidate_to_points_for_die(self, state: BackgammonState, from_point: jnp.ndarray, die: jnp.ndarray):
        """Return at most two candidate destinations for a single die move from one source point.

        Candidate 0: normal destination (or bar entry destination when moving from bar)
        Candidate 1: optional HOME destination for bearing off
        """
        from_point = jnp.asarray(from_point, dtype=jnp.int32)
        die = jnp.asarray(die, dtype=jnp.int32)
        player = state.current_player

        normal_to = jax.lax.cond(
            player == self.consts.WHITE,
            lambda _: from_point + die,
            lambda _: from_point - die,
            operand=None,
        )
        bar_to = jax.lax.cond(
            player == self.consts.WHITE,
            lambda _: die - 1,
            lambda _: 24 - die,
            operand=None,
        )

        is_from_bar = from_point == jnp.int32(self.consts.BAR_INDEX)
        primary_to = jax.lax.select(is_from_bar, bar_to, normal_to)

        in_white_home = (from_point >= 18) & (from_point <= 23)
        in_black_home = (from_point >= 0) & (from_point <= 5)
        in_home_board = jax.lax.select(player == self.consts.WHITE, in_white_home, in_black_home)
        can_bear_off_here = (~is_from_bar) & in_home_board & self.check_bearing_off(state, player)

        secondary_to = jnp.int32(self.consts.HOME_INDEX)
        secondary_valid = can_bear_off_here & (primary_to != secondary_to)

        return jnp.array([primary_to, secondary_to], dtype=jnp.int32), jnp.array([True, secondary_valid], dtype=jnp.bool_)

    @partial(jax.jit, static_argnums=(0,))
    def _any_move_with_single_die(self, state: BackgammonState, die_value) -> bool:
        """Branchless check: can any checker move with the given die value?
        Uses jnp.where instead of lax.cond to avoid path explosion under vmap."""
        die = jnp.asarray(die_value, dtype=jnp.int32)
        no_die = die <= 0
        test_state = state.replace(dice=jnp.array([die, 0, 0, 0], dtype=jnp.int32))

        player_idx = jnp.where(state.current_player == self.consts.WHITE, 0, 1)
        from_points = jnp.arange(self.consts.BAR_INDEX + 1, dtype=jnp.int32)  # 0..24
        has_piece = state.board[player_idx, from_points] > 0

        # If player has checkers on bar, only bar moves are legal.
        has_bar_checker = state.board[player_idx, self.consts.BAR_INDEX] > 0
        bar_gate = jnp.where(
            has_bar_checker,
            from_points == self.consts.BAR_INDEX,
            jnp.ones_like(from_points, dtype=jnp.bool_),
        )
        allowed_from = has_piece & bar_gate

        def eval_from(fp, allowed):
            to_points, to_valid = self._candidate_to_points_for_die(test_state, fp, die)

            def eval_to(tp, valid):
                mv = jnp.array([fp, tp], dtype=jnp.int32)
                result = self._is_valid_move_basic(test_state, mv)
                return valid & result

            move_ok = jnp.any(jax.vmap(eval_to)(to_points, to_valid))
            return allowed & move_ok

        has_any = jnp.any(jax.vmap(eval_from)(from_points, allowed_from))
        return jnp.where(no_die, jnp.bool_(False), has_any)

    @partial(jax.jit, static_argnums=(0,))
    def _precompute_dice_enforcement(self, state: BackgammonState):
        """Compute higher-die enforcement state once per board position.

        Returns:
            need_rule: True when higher-die rule must be enforced
            state_hi: state with only higher die active (for uses_hi check)
        """
        has_two, lo, hi = self._distinct_nonzero_dice(state.dice)
        can_hi = self._any_move_with_single_die(state, hi)
        can_lo = jax.lax.cond(
            has_two,
            lambda __: self._any_move_with_single_die(state, lo),
            lambda __: jnp.bool_(False),
            operand=None,
        )
        need_rule = has_two & can_hi & (~can_lo)
        state_hi = state.replace(dice=jnp.array([hi, 0, 0, 0], dtype=jnp.int32))
        return need_rule, state_hi

    @partial(jax.jit, static_argnums=(0,))
    def _is_valid_move_with_enforcement(self, state: BackgammonState, move: Tuple[int, int],
                                        need_rule: jnp.ndarray, state_hi: BackgammonState) -> bool:
        """Branchless move validation with higher-die enforcement."""
        basic_ok = self._is_valid_move_basic(state, move)
        uses_hi = self._is_valid_move_basic(state_hi, move)
        enforcement_ok = jnp.where(need_rule, uses_hi, jnp.bool_(True))
        return basic_ok & enforcement_ok

    @partial(jax.jit, static_argnums=(0,))
    def is_valid_move(self, state: BackgammonState, move: Tuple[int, int]) -> bool:
        need_rule, state_hi = self._precompute_dice_enforcement(state)
        return self._is_valid_move_with_enforcement(state, move, need_rule, state_hi)

    @partial(jax.jit, static_argnums=(0,))
    def check_bearing_off(self, state: BackgammonState, player: int) -> bool:
        """Branchless bearing-off eligibility check."""
        board = state.board
        player_idx = jnp.where(player == self.consts.WHITE, 0, 1)

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
        """Branchless legal move check."""
        has_two, lo, hi = self._distinct_nonzero_dice(state.dice)

        can_hi = self._any_move_with_single_die(state, hi)
        # Always compute can_lo (both branches run under vmap anyway)
        can_lo = self._any_move_with_single_die(state, lo)
        # Only count can_lo if we actually have two distinct dice
        return can_hi | (has_two & can_lo)

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
            # New roll means new original_dice for display
            new_original = jnp.array([next_dice[0], next_dice[1]], dtype=jnp.int32)
            next_player = -state.current_player
            next_cursor = jnp.where(
                next_player == self.consts.WHITE, jnp.int32(0), jnp.int32(23)
            )
            return state.replace(
                dice=next_dice,
                original_dice=new_original,
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
        """Apply a move to the board — branchless using jnp.where."""
        # Remove checker from source
        board = board.at[player_idx, from_point].add(-1)

        # Hit detection: opponent has exactly 1 checker at destination (not HOME)
        is_hit = (to_point != self.consts.HOME_INDEX) & (board[opponent_idx, to_point] == 1)
        # Compute hit board unconditionally, select with where
        hit_board = board.at[opponent_idx, to_point].set(0).at[opponent_idx, self.consts.BAR_INDEX].add(1)
        board = jax.tree.map(lambda h, o: jnp.where(is_hit, h, o), hit_board, board)

        # Add to destination: HOME_INDEX or to_point
        is_home = (to_point == self.consts.HOME_INDEX)
        board_home = board.at[player_idx, self.consts.HOME_INDEX].add(1)
        board_dest = board.at[player_idx, to_point].add(1)
        board = jax.tree.map(lambda h, d: jnp.where(is_home, h, d), board_home, board_dest)
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
        """Compute move distance — branchless using jnp.where."""
        is_white    = (player == self.consts.WHITE)
        is_from_bar = (from_point == self.consts.BAR_INDEX)
        is_to_home  = (to_point == self.consts.HOME_INDEX)

        # Distance when entering from BAR
        bar_distance = jnp.where(is_white, to_point + 1, jnp.int32(24) - to_point)

        # Bearing-off distance
        bear_distance = jnp.where(is_white, jnp.int32(24) - from_point, from_point + jnp.int32(1))

        # Normal board move distance
        normal_distance = jnp.where(is_white, to_point - from_point, from_point - to_point)

        # Select: bar > bear-off > normal
        regular_distance = jnp.where(is_to_home, bear_distance, normal_distance)
        return jnp.where(is_from_bar, bar_distance, regular_distance)

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
            # New turn gets new original_dice for display
            new_original = jnp.array([next_dice[0], next_dice[1]], dtype=jnp.int32)
            return next_dice, new_original, -state.current_player, new_key

        def same_turn(k):
            # Keep the same original_dice during the turn
            return new_dice, state.original_dice, state.current_player, k

        next_dice, next_original, next_player, new_key = jax.lax.cond(all_dice_used, next_turn, same_turn, key)

        white_won = new_board[0, self.consts.HOME_INDEX] == self.consts.NUM_CHECKERS
        black_won = new_board[1, self.consts.HOME_INDEX] == self.consts.NUM_CHECKERS
        game_over = white_won | black_won

        new_state = BackgammonState(
            board=new_board,
            dice=next_dice,
            original_dice=next_original,
            current_player=next_player,
            is_game_over=game_over,
            key=new_key,
            last_move=(from_point, to_point),
            last_dice=used_dice
        )
        
        # Only check auto-pass when we just switched players (all dice used)
        # and the game isn't over — avoids expensive has_any_legal_move call
        # on every single step
        new_state = jax.lax.cond(
            all_dice_used & (~game_over),
            lambda s: self._auto_pass_if_stuck(s),
            lambda s: s,
            operand=new_state
        )

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
    def step_move(self, state: BackgammonState, move_idx: int) -> Tuple[BackgammonObservation, BackgammonState, float, bool, BackgammonInfo, jax.Array]:
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
            new_state = state.replace(
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
        # Precompute higher-die enforcement once per state and reuse for all 676 moves.
        need_rule, state_hi = self._precompute_dice_enforcement(state)
        move_mask = jax.vmap(
            lambda mv: self._is_valid_move_with_enforcement(state, mv, need_rule, state_hi)
        )(self._action_pairs)
        
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
        LEFT  : 0→1→2→3→4→5→RIGHT_BAR_INDEX→6→7→8→9→10→11→12→13→14→15→16→17→24→18→19→20→21→22→23
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
        return state.replace(cursor_position=new_cursor)

    @partial(jax.jit, static_argnums=(0,))
    def _handle_roll_dice(self, state: BackgammonState) -> BackgammonState:
        """
        Handle dice roll action (FIRE in phase 0).
        Rolls dice and transitions to SELECTING_CHECKER phase.
        Auto-passes if no legal moves available.
        
        Returns: Updated state with new dice, original_dice, and game_phase.
        """
        dice, key = self.roll_dice(state.key)
        # Store original roll for display (ALE always shows 2 dice with original values)
        # For doubles, both display dice show the same value
        original_dice = jnp.array([dice[0], dice[1]], dtype=jnp.int32)
        new_state = state.replace(dice=dice, original_dice=original_dice, key=key, game_phase=1)
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
        is_bar_cursor = (pos == jnp.int32(self.consts.BAR_INDEX)) | (pos == jnp.int32(self.consts.RIGHT_BAR_INDEX))

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
            lambda s: s.replace(
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
            final_state = ns.replace(
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
            ns = s.replace(
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

    # Movement repeat delay disabled: process held LEFT/RIGHT every step.
    MOVE_REPEAT_DELAY: int = 0
    MOVE_INITIAL_DELAY: int = 0

    @partial(jax.jit, static_argnums=(0,))
    def _apply_debounce(self, state: BackgammonState, action: jnp.ndarray, 
                        new_state: BackgammonState) -> BackgammonState:
        """
        Apply debounce logic: arm await_keyup for FIRE only.
        LEFT/RIGHT update last_action and reset timer.
        
        Returns: State with debounce flags updated.
        """
        is_left = action == JAXAtariAction.LEFT
        is_right = action == JAXAtariAction.RIGHT
        is_fire = action == JAXAtariAction.FIRE
        is_movement = is_left | is_right
        
        # Only FIRE needs full debounce (wait for key release)
        # Movement actions reset the repeat timer instead
        return jax.lax.cond(
            is_fire,
            lambda s: s.replace(await_keyup=True, last_action=action, move_repeat_timer=0),
            lambda s: jax.lax.cond(
                is_movement,
                lambda s2: s2.replace(last_action=action, move_repeat_timer=0),
                lambda s2: s2,
                operand=s
            ),
            operand=new_state
        )

    @partial(jax.jit, static_argnums=(0,))
    def _handle_blocked_input(self, state: BackgammonState, action: jnp.ndarray) -> BackgammonState:
        """
        Handle input when awaiting key-up (debounce active for FIRE).
        NOOP clears the debounce; LEFT/RIGHT can still move.
        
        Returns: State with await_keyup potentially cleared.
        """
        is_noop = action == JAXAtariAction.NOOP
        return jax.lax.cond(
            is_noop,
            lambda s: s.replace(await_keyup=False, last_action=JAXAtariAction.NOOP, move_repeat_timer=0),
            lambda s: s,
            operand=state
        )

    # ============================================================================
    # MAIN STEP FUNCTION (Orchestrator Pattern - Design Guide Section 5)
    # ============================================================================

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: BackgammonState, action: jnp.ndarray):
        """
        Interactive step with JAX-safe debounce and continuous movement.
        
        This is the main entry point for gameplay. It follows the orchestrator pattern:
        - Delegates to helper functions for specific logic
        - Handles debounce for FIRE key-press/release behavior
        - LEFT/RIGHT are processed each step (no repeat delay)
        - Returns (observation, new_state, reward, done, info)
        
        FIRE triggers once per press; LEFT/RIGHT repeat when held.
        """
        # Translate policy action index to ALE-style action constant.
        atari_action = jnp.take(self.ACTION_SET, action.astype(jnp.int32))

        is_left = atari_action == JAXAtariAction.LEFT
        is_right = atari_action == JAXAtariAction.RIGHT
        is_movement = is_left | is_right
        
        # Branch 1: Debounce active (FIRE held) - wait for key release, but allow movement
        def when_blocked(s):
            # Allow movement even when FIRE is blocked
            def do_movement(s2):
                def process_repeat(s3):
                    ns, _, _ = self._process_action(s3, atari_action)
                    return ns.replace(last_action=atari_action, move_repeat_timer=0)
                return process_repeat(s2)
            
            ns = jax.lax.cond(
                is_movement,
                do_movement,
                lambda s2: self._handle_blocked_input(s2, action),
                operand=s
            )
            return self._get_observation(ns), ns, 0.0, False, self._get_info(ns)

        # Branch 2: Ready for input - process action
        def when_free(s):
            # For movement: process each LEFT/RIGHT step immediately.
            def handle_movement(s2):
                def process_move(s3):
                    ns, reward, done = self._process_action(s3, atari_action)
                    ns = ns.replace(last_action=atari_action, move_repeat_timer=0)
                    return ns, reward, done
                return process_move(s2)
            
            def handle_other(s2):
                ns, reward, done = self._process_action(s2, atari_action)
                ns = self._apply_debounce(s, atari_action, ns)
                return ns, reward, done
            
            ns, reward, done = jax.lax.cond(is_movement, handle_movement, handle_other, operand=s)
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
            obs.bar_counts.flatten(),
            obs.home_counts.flatten(),
            obs.cursor_position.flatten(),
            obs.game_phase.flatten(),
            obs.picked_checker_from.flatten(),
            obs.last_valid_drop.flatten(),
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
        """Interactive ALE-style action space used by wrappers and benchmark training."""
        return spaces.Discrete(len(self.ACTION_SET))

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
            "cursor_position": spaces.Box(
                low=0,
                high=26,
                shape=(1,),
                dtype=jnp.int32
            ),
            "game_phase": spaces.Box(
                low=0,
                high=2,
                shape=(1,),
                dtype=jnp.int32
            ),
            "picked_checker_from": spaces.Box(
                low=-1,
                high=26,
                shape=(1,),
                dtype=jnp.int32
            ),
            "last_valid_drop": spaces.Box(
                low=-1,
                high=26,
                shape=(1,),
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
            home_counts=jnp.array([state.board[0, 25], state.board[1, 25]], dtype=jnp.int32),
            cursor_position=jnp.array([state.cursor_position], dtype=jnp.int32),
            game_phase=jnp.array([state.game_phase], dtype=jnp.int32),
            picked_checker_from=jnp.array([state.picked_checker_from], dtype=jnp.int32),
            last_valid_drop=jnp.array([state.last_valid_drop], dtype=jnp.int32),
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
        """Python utility; use get_valid_action_mask() for JIT-compatible pipelines."""
        valid_mask = jax.device_get(self.get_valid_action_mask(state)[: self._action_pairs.shape[0]])
        valid_indices = jnp.where(valid_mask)[0]
        return [tuple(map(int, self._action_pairs[int(i)])) for i in valid_indices]

    def render(self, state: BackgammonState) -> jnp.ndarray:
        return self.renderer.render(state)


# ============================================================================
# RENDERER (Using JaxRenderingUtils - Design Guide + Renderer Guide)
# ============================================================================

class BackgammonRenderer(JAXGameRenderer):
    """
    JAX-native Backgammon renderer using JaxRenderingUtils.
    
    Supports multiple color themes:
    - "classic": Original green theme
    - "brown": Wooden/brown theme  
    - "blue": Tournament/blue theme
    """
    
    def __init__(self, env: "JaxBackgammonEnv" = None, config: Optional[jr.RendererConfig] = None, consts: Optional[Any] = None):
        # Allow being called without env (as in some wrapper tests or modifications)
        # but ensure config and consts are handled.
        if config is None:
            # Create a default config if not provided
            from jaxatari.rendering.jax_rendering_utils import RendererConfig
            config = RendererConfig()
            
        super().__init__(env, config=config)
        self.env = env
        
        # Determine consts from env or argument
        self.consts = consts if consts is not None else (env.consts if env else None)
        
        # Determine theme from env or consts
        if env is not None:
            self.theme = env.consts.THEME
        elif self.consts is not None:
            self.theme = getattr(self.consts, "THEME", "classic")
        else:
            self.theme = "classic"
        
        # Frame dimensions
        self.frame_height = 210
        self.frame_width = 160
        
        # Geometry constants (updated to match extracted ALE sprites)
        self.top_margin_for_dice = 15
        self.board_margin = 24
        self.triangle_length = 48
        self.triangle_thickness = 11
        self.bar_thickness = 11
        self.checker_width = 4          # ALE checker is 4x4 pixels
        self.checker_height = 4
        self.base_margin = 2
        self.band_top_margin = 0.5
        self.band_bottom_margin = 0.5
        self.chip_gap_y = 1              # Gap between checker rows (ALE: row0 at y=39, row1 at y=44 -> offset=5)
        self.chip_gap_x = 4              # Gap between checker columns (ALE: col_step=8 = 4+4)
        self.checker_stack_offset = self.checker_height + self.chip_gap_y
        self.bar_edge_padding = 2
        self.bar_vertical_padding = 2
        self.dice_width = 14 
        self.dice_height = 17
        self.pip_size = 2
        
        # Computed positions (ALE bar is at y=110-120)
        self.bar_y = 110  # Fixed to match ALE
        self.bar_x = self.board_margin
        self.bar_width = self.frame_width - 2 * self.board_margin - 7
        
        # Pre-compute triangle positions
        self.triangle_positions = self._compute_triangle_positions()
        
        # Setup JaxRenderingUtils
        from jaxatari.rendering import jax_rendering_utils as render_utils
        
        self.config = config or render_utils.RendererConfig(
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
            
        asset_config = list(_get_asset_config())
        
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(asset_config, sprite_path)
        
        # Pre-bake the static triangle raster (triangles never change)
        self._prebaked_triangle_raster = self._prebake_triangle_raster()
        
        # Pre-compute a 4-triangle mask stack for branchless selection:
        # [light_right, dark_right, light_left, dark_left]
        self._triangle_mask_stack = jnp.stack([
            self.SHAPE_MASKS["triangle_light_right"],
            self.SHAPE_MASKS["triangle_dark_right"],
            self.SHAPE_MASKS["triangle_light_left"],
            self.SHAPE_MASKS["triangle_dark_left"],
        ])
        
        # Pre-compute pip bitmask table for dice rendering (values 1-6)
        # 7 pip positions: TL=0, TR=1, ML=2, C=3, MR=4, BL=5, BR=6
        # Each row is a bool mask for which pips to draw
        self._pip_table = jnp.array([
            [0, 0, 0, 1, 0, 0, 0],  # 1: center
            [1, 0, 0, 0, 0, 0, 1],  # 2: TL, BR
            [1, 0, 0, 1, 0, 0, 1],  # 3: TL, C, BR
            [1, 1, 0, 0, 0, 1, 1],  # 4: TL, TR, BL, BR
            [1, 1, 0, 1, 0, 1, 1],  # 5: TL, TR, C, BL, BR
            [1, 1, 1, 0, 1, 1, 1],  # 6: TL, TR, ML, MR, BL, BR
        ], dtype=jnp.bool_)
        
        # Pip position offsets within a die (local x, y)
        self._pip_positions = jnp.array([
            [2, 2],    # TL
            [10, 2],   # TR
            [2, 7],    # ML
            [6, 7],    # C
            [10, 7],   # MR
            [2, 12],   # BL
            [10, 12],  # BR
        ], dtype=jnp.int32)
    
    def _get_asset_config(self) -> list:
        """Backwards-compatible accessor for the shared module-level manifest."""
        return list(_get_asset_config())
    
    def _compute_triangle_positions(self):
        """Compute (x, y) positions for all 24 triangles + bar positions.
        
        ALE layout (verified from frame analysis):
        - Upper band: y=38-108 (6 triangles, each 11 rows with 1-row gaps)
        - Bar: y=110-120 (teal)
        - Lower band: y=122-192 (6 triangles, each 11 rows with 1-row gaps)
        - Triangle stride: 12 rows (11 triangle + 1 gap)
        """
        left_x = self.board_margin
        right_x = self.frame_width - self.board_margin - self.triangle_length
        
        # ALE-accurate positions
        upper_band_start = 38  # First upper triangle starts here
        lower_band_start = 122  # First lower triangle starts here
        triangle_stride = 12   # 11 rows + 1 gap between triangles
        
        positions = []
        
        # 0..5 (lower-right) bottom -> top
        # Point 0 is at bottom (y=182), point 5 is at top (y=122)
        for i in range(6):
            y = lower_band_start + (5 - i) * triangle_stride
            positions.append((right_x, y))
        
        # 6..11 (upper-right) bottom -> top
        # Point 6 is at bottom (y=98), point 11 is at top (y=38)
        for i in range(6):
            y = upper_band_start + (5 - i) * triangle_stride
            positions.append((right_x, y))
        
        # 12..17 (upper-left) top -> bottom
        # Point 12 is at top (y=38), point 17 is at bottom (y=98)
        for i in range(6):
            y = upper_band_start + i * triangle_stride
            positions.append((left_x, y))
        
        # 18..23 (lower-left) top -> bottom
        # Point 18 is at top (y=122), point 23 is at bottom (y=182)
        for i in range(6):
            y = lower_band_start + i * triangle_stride
            positions.append((left_x, y))
        
        # bar-left (24): center of left half of the bar
        bar_left_x = self.bar_x + (self.bar_width // 4)
        bar_center_y = self.bar_y + self.bar_thickness // 2
        positions.append((bar_left_x, bar_center_y))
        
        return jnp.array(positions, dtype=jnp.int32)
    
    def _get_player_index(self, player):
        """Helper to get player index without env dependency."""
        return jnp.where(player == self.consts.WHITE, 0, 1)

    def _prebake_triangle_raster(self):
        """Pre-render all 24 triangles onto the background. Called once at init."""
        raster = self.jr.create_object_raster(self.BACKGROUND)
        left_x = self.board_margin
        
        # Python loop at init time — not JIT-compiled, runs once
        for i in range(24):
            pos = self.triangle_positions[i]
            x, y = int(pos[0]), int(pos[1])
            
            is_left_column = (x == left_x)
            band_idx = i % 6
            band_id = i // 6
            
            is_upper_band = (6 <= i < 12) or (12 <= i < 18)
            if is_left_column:
                start_light = is_upper_band
            else:
                start_light = not is_upper_band
            if band_id in (1, 3):
                start_light = not start_light
            use_light = (band_idx % 2 == 0) if start_light else (band_idx % 2 == 1)
            
            if is_left_column:
                key = "triangle_light_right" if use_light else "triangle_dark_right"
            else:
                key = "triangle_light_left" if use_light else "triangle_dark_left"
            
            raster = self.jr.render_at(raster, x, y, self.SHAPE_MASKS[key])
        
        return raster

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: BackgammonState) -> jnp.ndarray:
        """
        Render the current game state using palette-based rendering.
        
        Optimized: uses pre-baked triangle raster as starting point.
        """
        # 1. Start with pre-baked background + triangles (no per-frame cost)
        raster = self._prebaked_triangle_raster.copy()
        
        # 2. Draw highlight (cursor or picked position)
        raster = self._draw_highlight(raster, state)
        
        # 3. Draw checkers on all points
        raster = self._draw_all_checkers(raster, state)
        
        # 4. Draw bar checkers
        raster = self._draw_bar_checkers(raster, state)
        
        # 5. Draw floating checker (if in MOVING phase)
        raster = self._draw_floating_checker(raster, state)
        
        # 6. Draw dice
        raster = self._draw_dice(raster, state)
        
        # 7. Convert to RGB
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
        """Branchless highlight rendering."""
        # Compute highlight index without nested lax.cond
        is_moving = (state.game_phase == jnp.int32(2))
        is_selecting = (state.game_phase == jnp.int32(1))
        
        # Moving phase: use picked_checker_from (with bar side logic)
        is_bar_pick = (state.picked_checker_from == jnp.int32(self.consts.BAR_INDEX))
        moving_hi = jnp.where(
            is_bar_pick,
            jnp.where(state.picked_bar_side == jnp.int32(self.consts.RIGHT_BAR_INDEX),
                      jnp.int32(self.consts.RIGHT_BAR_INDEX), jnp.int32(self.consts.BAR_INDEX)),
            jnp.int32(state.picked_checker_from)
        )
        
        hi = jnp.where(is_moving, moving_hi,
               jnp.where(is_selecting,
                         jnp.int32(state.cursor_position),
                         jnp.int32(state.last_valid_drop)))
        
        no_hi = (hi < 0)
        
        # Select which mask and position to use — compute all, select with where
        safe_hi = jnp.clip(hi, 0, 23)  # safe index for triangle_positions
        tri_pos = self.triangle_positions[safe_hi]
        is_left = (tri_pos[0] == self.board_margin)
        tri_mask = jax.lax.select(is_left,
            self.SHAPE_MASKS["triangle_highlight_right"],
            self.SHAPE_MASKS["triangle_highlight_left"])
        
        # Compute all possible render targets
        # Triangle render
        raster_tri = self.jr.render_at(raster, tri_pos[0], tri_pos[1], tri_mask)
        # Bar left render
        raster_bar_l = self.jr.render_at(raster, self.bar_x, self.bar_y,
            self.SHAPE_MASKS["bar_highlight_left"])
        # Bar right render
        half_w = self.bar_width // 2
        raster_bar_r = self.jr.render_at(raster, self.bar_x + half_w, self.bar_y,
            self.SHAPE_MASKS["bar_highlight_right"])
        
        # Select the right result with jnp.where over the raster arrays
        is_tri = (hi < jnp.int32(24))
        is_bar_l = (hi == jnp.int32(self.consts.BAR_INDEX))  # 24
        is_bar_r = (hi == jnp.int32(self.consts.RIGHT_BAR_INDEX))
        
        result = jnp.where(no_hi, raster,
                   jnp.where(is_tri, raster_tri,
                     jnp.where(is_bar_l, raster_bar_l,
                       jnp.where(is_bar_r, raster_bar_r, raster))))
        return result
    
    @partial(jax.jit, static_argnums=(0,))
    def _draw_all_checkers(self, raster, state):
        """Draw checkers on all 24 points — branchless count adjustment."""
        player_idx = self._get_player_index(state.current_player)
        is_picking = (state.game_phase == 2)
        
        def draw_point_checkers(point_idx, r):
            white_count = state.board[0, point_idx]
            black_count = state.board[1, point_idx]
            
            # Subtract one from picked position — branchless
            pick_here = is_picking & (state.picked_checker_from == point_idx)
            white_count = jnp.where(pick_here & (player_idx == 0),
                                    jnp.maximum(white_count - 1, 0), white_count)
            black_count = jnp.where(pick_here & (player_idx == 1),
                                    jnp.maximum(black_count - 1, 0), black_count)
            
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
        
        # ALE stacks checkers from TIP toward BASE:
        # For upper triangles (pointing DOWN): tip is at bottom (higher y), base at top (lower y)
        # For lower triangles (pointing UP): tip is at top (lower y), base at bottom (higher y)
        # First checker (i=0, row=0) goes near the TIP (y + 1 + offset)
        # Second checker (i=1, row=1) goes near the BASE (y + 1)
        # For upper triangles (y=38): TIP row at y=44, BASE row at y=39
        near_base_y = jnp.int32(y + 1)  # Row closer to wide part of triangle
        near_tip_y = jnp.int32(near_base_y + self.checker_stack_offset)  # Row closer to triangle point
        col_step = self.checker_width + self.chip_gap_x
        
        def draw_stack(r, count, mask):
            def draw_single(i, f):
                col = i // 2
                row = i & 1
                cx = base_x + dir_sign * (col * col_step)
                # row=0 (first/single checker) goes near tip, row=1 (second) goes near base
                cy = jnp.where(row == 0, near_tip_y, near_base_y)
                return self.jr.render_at(f, cx, cy, mask)
            return jax.lax.fori_loop(0, count, draw_single, r)
        
        raster = draw_stack(raster, white_count, self.SHAPE_MASKS["white_checker"])
        raster = draw_stack(raster, black_count, self.SHAPE_MASKS["black_checker"])
        return raster
    
    @partial(jax.jit, static_argnums=(0,))
    def _draw_bar_checkers(self, raster, state):
        """Draw checkers on the bar.
        
        ALE layout:
        - White checkers on LEFT side of bar, starting at x=26, expanding RIGHT
        - Black/Red checkers on RIGHT side of bar, starting at x=130, expanding LEFT
        - Stacking order (same as triangles): Base first, then Top
          - i=0: col 0, row 0 (y=116, base/bottom)
          - i=1: col 0, row 1 (y=111, top)
          - i=2: col 1, row 0 (y=116, base/bottom)
          - i=3: col 1, row 1 (y=111, top)
        """
        player_idx = self._get_player_index(state.current_player)
        picked_from_bar = (state.game_phase == 2) & (state.picked_checker_from == self.consts.BAR_INDEX)
        
        white_bar = state.board[0, self.consts.BAR_INDEX] - jnp.where(picked_from_bar & (player_idx == 0), 1, 0)
        black_bar = state.board[1, self.consts.BAR_INDEX] - jnp.where(picked_from_bar & (player_idx == 1), 1, 0)
        white_bar = jnp.maximum(white_bar, 0)
        black_bar = jnp.maximum(black_bar, 0)
        
        col_step = self.checker_width + self.chip_gap_x  # 4 + 4 = 8
        row_step = self.checker_stack_offset  # 5
        
        # White starts at x=26 (left edge), expands right
        base_x_white = self.bar_x + 2  # x=26
        
        # Black starts at x=130 (right edge), expands left
        base_x_black = self.bar_x + 106  # x=130 (24 + 106)
        
        # Y positions: base (bottom) at y=116, top at y=111
        # Bar is at y=110
        base_y = self.bar_y + 6  # y=116 for row 0 (base/bottom)
        top_y = self.bar_y + 1   # y=111 for row 1 (top)
        
        def draw_white_stack(r, count, mask):
            """White expands left to right (positive X direction)"""
            def draw_single(i, f):
                col = i // 2
                row = i % 2
                x = base_x_white + col * col_step
                y = jnp.where(row == 0, base_y, top_y)
                return self.jr.render_at(f, x, y, mask)
            return jax.lax.fori_loop(0, count, draw_single, r)
        
        def draw_black_stack(r, count, mask):
            """Black expands right to left (negative X direction)"""
            def draw_single(i, f):
                col = i // 2
                row = i % 2
                x = base_x_black - col * col_step  # Subtract to go left
                y = jnp.where(row == 0, base_y, top_y)
                return self.jr.render_at(f, x, y, mask)
            return jax.lax.fori_loop(0, count, draw_single, r)
        
        raster = draw_white_stack(raster, white_bar, self.SHAPE_MASKS["white_checker"])
        raster = draw_black_stack(raster, black_bar, self.SHAPE_MASKS["black_checker"])
        return raster
    
    @partial(jax.jit, static_argnums=(0,))
    def _draw_floating_checker(self, raster, state):
        """Draw the floating checker — branchless position calculation."""
        player_idx = self._get_player_index(state.current_player)
        mask = jax.lax.select(player_idx == 0,
            self.SHAPE_MASKS["white_checker"],
            self.SHAPE_MASKS["black_checker"])
        
        pos = state.cursor_position
        is_home = (pos == jnp.int32(self.consts.HOME_INDEX))
        is_bar_left = (pos == jnp.int32(self.consts.BAR_INDEX))
        is_bar_right = (pos == jnp.int32(self.consts.RIGHT_BAR_INDEX))
        is_on_board = (pos >= 0) & (pos < 24)
        
        # Home position
        src = state.picked_checker_from
        use_right_edge = (src >= 18)
        edge_idx = jax.lax.select(use_right_edge, jnp.int32(23), jnp.int32(0))
        tri_x = self.triangle_positions[edge_idx][0] + self.triangle_length // 2
        y_top_tri = self.top_margin_for_dice + self.board_margin
        y_bottom_tri = self.frame_height - self.board_margin
        edge_is_top_half = (edge_idx >= 6) & (edge_idx <= 17)
        home_cx = tri_x
        home_cy = jax.lax.select(edge_is_top_half,
            y_top_tri - (self.checker_height // 2) - 1,
            y_bottom_tri + (self.checker_height // 2) + 1)
        
        # Board position (safe index)
        safe_pos = jnp.clip(pos, 0, 23)
        board_cx = self.triangle_positions[safe_pos][0] + self.triangle_length // 2
        board_cy = self.triangle_positions[safe_pos][1] + self.triangle_thickness // 2
        
        # Bar position
        bar_left_cx = self.bar_x + (self.bar_width // 4)
        bar_right_cx = self.bar_x + (3 * self.bar_width // 4)
        bar_cy = self.bar_y + self.bar_thickness // 2
        
        # Select with chained jnp.where — no nested lax.cond
        cx = jnp.where(is_home, home_cx,
               jnp.where(is_on_board, board_cx,
                 jnp.where(is_bar_left, bar_left_cx,
                   jnp.where(is_bar_right, bar_right_cx,
                     self.frame_width // 2))))
        
        cy = jnp.where(is_home, home_cy,
               jnp.where(is_on_board, board_cy,
                 jnp.where(is_bar_left | is_bar_right, bar_cy,
                   self.bar_y + self.bar_thickness // 2)))
        
        # Always compute the rendered raster, then select with where
        rendered = self.jr.render_at(raster, cx - self.checker_width // 2,
                                    cy - self.checker_height // 2, mask)
        
        should_draw = (state.game_phase == 2) & (state.cursor_position != self.consts.HOME_INDEX)
        return jnp.where(should_draw, rendered, raster)
    
    @partial(jax.jit, static_argnums=(0,))
    def _draw_dice(self, raster, state):
        """Draw dice — uses pre-computed pip bitmask table instead of lax.switch."""
        dice_gap = 6
        start_x = 64
        dice_y = 17
        
        is_white_player = (state.current_player == self.consts.WHITE)
        die_mask = jax.lax.select(is_white_player,
            self.SHAPE_MASKS["die_white"],
            self.SHAPE_MASKS["die_red"])
        pip_mask = jax.lax.select(is_white_player,
            self.SHAPE_MASKS["pip_black"],
            self.SHAPE_MASKS["pip_white"])
        
        display_dice = state.original_dice
        pip_table = self._pip_table      # (6, 7) bool
        pip_positions = self._pip_positions  # (7, 2) int32
        
        def draw_single_die(i, r):
            val = display_dice[i]
            dx = start_x + i * (self.dice_width + dice_gap)
            
            # Draw die background
            r_with_die = self.jr.render_at(r, dx, dice_y, die_mask)
            
            # Look up active pips from bitmask table
            pip_active = pip_table[jnp.clip(val - 1, 0, 5)]  # (7,) bool
            
            # Draw all 7 possible pip positions, masking by active flag
            def draw_pip(j, f):
                px = dx + pip_positions[j, 0]
                py = dice_y + pip_positions[j, 1]
                rendered = self.jr.render_at(f, px, py, pip_mask)
                return jnp.where(pip_active[j], rendered, f)
            
            r_with_pips = jax.lax.fori_loop(0, 7, draw_pip, r_with_die)
            
            # Only draw if val > 0
            return jnp.where(val > 0, r_with_pips, r)
        
        return jax.lax.fori_loop(0, 2, draw_single_die, raster)