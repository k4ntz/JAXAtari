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
    last_move: Tuple[int, int] = (-1, -1)   # NEW
    last_dice: int = -1                      # NEW


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
            return jax.vmap(lambda move: self.is_valid_move(state, move))(self._action_pairs)

        valid_mask = _check_all_moves(state)
        valid_moves_array = self._action_pairs[valid_mask]
        return [tuple(map(int, move)) for move in valid_moves_array]

    def render(self, state: BackgammonState) -> Tuple[jnp.ndarray]:
        self.renderer.render(state)


class BackgammonRenderer(JAXGameRenderer):
    def __init__(self, env=None):
        super().__init__(env)

        # Frame geometry - smaller to account for 4x upscale in play.py
        self.frame_height = 210  # Standard Atari height
        self.frame_width = 180  # Standard Atari width

        # Colors (RGB)
        self.color_background = jnp.array([34, 139, 34], dtype=jnp.uint8)  # Forest green
        self.color_board = jnp.array([139, 69, 19], dtype=jnp.uint8)  # Saddle brown
        self.color_triangle_light = jnp.array([222, 184, 135], dtype=jnp.uint8)  # Burlywood
        self.color_triangle_dark = jnp.array([160, 82, 45], dtype=jnp.uint8)  # Saddle brown
        self.color_white_checker = jnp.array([255, 255, 255], dtype=jnp.uint8)  # White
        self.color_black_checker = jnp.array([50, 50, 50], dtype=jnp.uint8)  # Dark gray
        self.color_border = jnp.array([101, 67, 33], dtype=jnp.uint8)  # Dark brown

        # Game geometry - scaled for 160x210 canvas
        self.board_margin = 8
        self.triangle_height = 60
        self.triangle_width = 12
        self.bar_width = 16
        self.checker_radius = 5
        self.checker_stack_offset = 8

        # Precompute positions
        self.triangle_positions = self._compute_triangle_positions()
        self.bar_position = (self.frame_width // 2 - self.bar_width // 2, self.board_margin)
        self.home_position = (self.frame_width - 80, self.frame_height // 2)

    def _compute_triangle_positions(self):
        """Compute the base positions for all 24 triangles"""
        positions = []

        # Left side triangles (points 12-7 top, 13-18 bottom)
        left_start = self.board_margin
        for i in range(6):
            x = left_start + i * self.triangle_width
            # Top triangles (points 12-7, displayed right to left)
            positions.append((x, self.board_margin))

        # Right side triangles (points 6-1 top, 19-24 bottom)
        right_start = self.board_margin + 6 * self.triangle_width + self.bar_width
        for i in range(6):
            x = right_start + i * self.triangle_width
            # Top triangles (points 6-1, displayed right to left)
            positions.append((x, self.board_margin))

        # Bottom triangles - reverse order
        for i in range(6):
            x = right_start + (5 - i) * self.triangle_width
            # Bottom triangles (points 19-24)
            positions.append((x, self.frame_height - self.board_margin - self.triangle_height))

        for i in range(6):
            x = left_start + (5 - i) * self.triangle_width
            # Bottom triangles (points 13-18)
            positions.append((x, self.frame_height - self.board_margin - self.triangle_height))

        return jnp.array(positions, dtype=jnp.int32)

    @partial(jax.jit, static_argnums=(0,))
    def _draw_rectangle(self, frame, x, y, width, height, color):
        """Draw a filled rectangle"""
        x, y = jnp.clip(x, 0, self.frame_width), jnp.clip(y, 0, self.frame_height)
        x2 = jnp.clip(x + width, 0, self.frame_width)
        y2 = jnp.clip(y + height, 0, self.frame_height)

        # Create coordinate grids
        yy, xx = jnp.mgrid[0:self.frame_height, 0:self.frame_width]
        mask = (xx >= x) & (xx < x2) & (yy >= y) & (yy < y2)

        # Apply color where mask is True
        frame = jnp.where(mask[..., None], color, frame)
        return frame

    @partial(jax.jit, static_argnums=(0,))
    def _draw_triangle(self, frame, x, y, width, height, color, point_up=True):
        """Draw a triangle"""
        yy, xx = jnp.mgrid[0:self.frame_height, 0:self.frame_width]

        def triangle_pointing_up():
            # Triangle pointing up (for bottom row)
            left_edge = yy >= (y + height) - ((xx - x) * height / width)
            right_edge = yy >= (y + height) - ((x + width - xx) * height / width)
            bottom_edge = yy <= y + height
            top_edge = yy >= y
            return left_edge & right_edge & bottom_edge & top_edge & (xx >= x) & (xx < x + width)

        def triangle_pointing_down():
            # Triangle pointing down (for top row)
            left_edge = yy <= y + ((xx - x) * height / width)
            right_edge = yy <= y + ((x + width - xx) * height / width)
            bottom_edge = yy <= y + height
            top_edge = yy >= y
            return left_edge & right_edge & bottom_edge & top_edge & (xx >= x) & (xx < x + width)

        mask = jax.lax.cond(point_up, triangle_pointing_up, triangle_pointing_down)
        frame = jnp.where(mask[..., None], color, frame)
        return frame

    @partial(jax.jit, static_argnums=(0,))
    def _draw_circle(self, frame, center_x, center_y, radius, color):
        """Draw a filled circle"""
        yy, xx = jnp.mgrid[0:self.frame_height, 0:self.frame_width]
        distance_sq = (xx - center_x) ** 2 + (yy - center_y) ** 2
        mask = distance_sq <= radius ** 2
        frame = jnp.where(mask[..., None], color, frame)
        return frame

    @partial(jax.jit, static_argnums=(0,))
    def _draw_board_outline(self, frame):
        """Draw the board background and outline"""
        # Background
        frame = self._draw_rectangle(frame, 0, 0, self.frame_width, self.frame_height,
                                     self.color_background)

        # Board area
        board_x = self.board_margin - 10
        board_y = self.board_margin - 10
        board_w = self.frame_width - 2 * (self.board_margin - 10)
        board_h = self.frame_height - 2 * (self.board_margin - 10)

        frame = self._draw_rectangle(frame, board_x, board_y, board_w, board_h,
                                     self.color_board)

        # Bar
        bar_x = self.frame_width // 2 - self.bar_width // 2
        bar_y = self.board_margin
        bar_h = self.frame_height - 2 * self.board_margin

        frame = self._draw_rectangle(frame, bar_x, bar_y, self.bar_width, bar_h,
                                     self.color_border)

        return frame

    @partial(jax.jit, static_argnums=(0,))
    def _draw_triangles(self, frame):
        """Draw all 24 triangles"""

        def draw_triangle_at_index(i, fr):
            pos = self.triangle_positions[i]
            x, y = pos[0], pos[1]

            # Alternate colors
            color = jax.lax.select(i % 2 == 0,
                                   self.color_triangle_light,
                                   self.color_triangle_dark)

            # Top triangles point down, bottom triangles point up
            point_up = i >= 12

            return self._draw_triangle(fr, x, y, self.triangle_width,
                                       self.triangle_height, color, point_up)

        return jax.lax.fori_loop(0, 24, draw_triangle_at_index, frame)

    @partial(jax.jit, static_argnums=(0,))
    def _draw_checkers_on_point(self, frame, point_idx, white_count, black_count):
        """Draw checkers on a specific point"""
        pos = self.triangle_positions[point_idx]
        base_x = pos[0] + self.triangle_width // 2
        base_y = pos[1]

        # Adjust base_y for bottom triangles
        base_y = jax.lax.select(point_idx >= 12,
                                base_y + self.triangle_height - self.checker_radius,
                                base_y + self.checker_radius)

        # Direction of stacking
        stack_direction = jax.lax.select(point_idx >= 12, -1, 1)

        def draw_checker_stack(fr, count, color, start_offset):
            def draw_single_checker(i, frame_inner):
                y_offset = start_offset + i * self.checker_stack_offset * stack_direction
                checker_y = base_y + y_offset
                return self._draw_circle(frame_inner, base_x, checker_y,
                                         self.checker_radius, color)

            return jax.lax.fori_loop(0, count, draw_single_checker, fr)

        # Draw white checkers first, then black checkers on top
        frame = draw_checker_stack(frame, white_count, self.color_white_checker, 0)
        white_offset = white_count * self.checker_stack_offset * stack_direction
        frame = draw_checker_stack(frame, black_count, self.color_black_checker, white_offset)

        return frame

    @partial(jax.jit, static_argnums=(0,))
    def _draw_bar_checkers(self, frame, white_count, black_count):
        """Draw checkers on the bar"""
        bar_center_x = self.frame_width // 2
        bar_center_y = self.frame_height // 2

        def draw_bar_stack(fr, count, color, y_offset):
            def draw_single_checker(i, frame_inner):
                checker_y = bar_center_y + y_offset + i * self.checker_stack_offset
                return self._draw_circle(frame_inner, bar_center_x, checker_y,
                                         self.checker_radius, color)

            return jax.lax.fori_loop(0, count, draw_single_checker, fr)

        # White checkers above center, black checkers below
        frame = draw_bar_stack(frame, white_count, self.color_white_checker, -25)
        frame = draw_bar_stack(frame, black_count, self.color_black_checker, 10)

        return frame

    @partial(jax.jit, static_argnums=(0,))
    def _draw_home_checkers(self, frame, white_count, black_count):
        """Draw checkers in the home area"""
        home_x = self.frame_width - 20

        def draw_home_stack(fr, count, color, y_start):
            def draw_single_checker(i, frame_inner):
                checker_y = y_start + i * self.checker_stack_offset
                return self._draw_circle(frame_inner, home_x, checker_y,
                                         self.checker_radius, color)

            return jax.lax.fori_loop(0, count, draw_single_checker, fr)

        # White checkers at bottom, black checkers at top
        frame = draw_home_stack(frame, white_count, self.color_white_checker, 150)
        frame = draw_home_stack(frame, black_count, self.color_black_checker, 40)

        return frame

    @partial(jax.jit, static_argnums=(0,))
    def _draw_dice(self, frame, dice):
        """Draw the current dice"""
        dice_size = 12
        dice_x_start = self.frame_width // 2 - 15
        dice_y = self.frame_height // 2 - dice_size // 2

        def draw_single_dice(i, fr):
            dice_value = dice[i]
            dice_x = dice_x_start + i * 15

            def draw_active_dice(_):
                # Draw dice background (white square)
                fr_with_bg = self._draw_rectangle(fr, dice_x, dice_y, dice_size, dice_size,
                                                  jnp.array([240, 240, 240], dtype=jnp.uint8))

                # Draw dice border
                border_width = 1
                fr_with_border = self._draw_rectangle(fr_with_bg, dice_x - border_width,
                                                      dice_y - border_width,
                                                      dice_size + 2 * border_width,
                                                      dice_size + 2 * border_width,
                                                      self.color_border)
                fr_with_border = self._draw_rectangle(fr_with_border, dice_x, dice_y,
                                                      dice_size, dice_size,
                                                      jnp.array([240, 240, 240], dtype=jnp.uint8))

                # Draw pips based on dice value
                pip_color = jnp.array([0, 0, 0], dtype=jnp.uint8)  # Black pips
                pip_radius = 1
                center_x = dice_x + dice_size // 2
                center_y = dice_y + dice_size // 2

                def draw_pips_1(_):
                    return self._draw_circle(fr_with_border, center_x, center_y, pip_radius, pip_color)

                def draw_pips_2(_):
                    fr2 = self._draw_circle(fr_with_border, center_x - 3, center_y - 3, pip_radius, pip_color)
                    return self._draw_circle(fr2, center_x + 3, center_y + 3, pip_radius, pip_color)

                def draw_pips_3(_):
                    fr3 = self._draw_circle(fr_with_border, center_x - 3, center_y - 3, pip_radius, pip_color)
                    fr3 = self._draw_circle(fr3, center_x, center_y, pip_radius, pip_color)
                    return self._draw_circle(fr3, center_x + 3, center_y + 3, pip_radius, pip_color)

                def draw_pips_4(_):
                    fr4 = self._draw_circle(fr_with_border, center_x - 3, center_y - 3, pip_radius, pip_color)
                    fr4 = self._draw_circle(fr4, center_x + 3, center_y - 3, pip_radius, pip_color)
                    fr4 = self._draw_circle(fr4, center_x - 3, center_y + 3, pip_radius, pip_color)
                    return self._draw_circle(fr4, center_x + 3, center_y + 3, pip_radius, pip_color)

                def draw_pips_5(_):
                    fr5 = self._draw_circle(fr_with_border, center_x - 3, center_y - 3, pip_radius, pip_color)
                    fr5 = self._draw_circle(fr5, center_x + 3, center_y - 3, pip_radius, pip_color)
                    fr5 = self._draw_circle(fr5, center_x, center_y, pip_radius, pip_color)
                    fr5 = self._draw_circle(fr5, center_x - 3, center_y + 3, pip_radius, pip_color)
                    return self._draw_circle(fr5, center_x + 3, center_y + 3, pip_radius, pip_color)

                def draw_pips_6(_):
                    fr6 = self._draw_circle(fr_with_border, center_x - 3, center_y - 3, pip_radius, pip_color)
                    fr6 = self._draw_circle(fr6, center_x + 3, center_y - 3, pip_radius, pip_color)
                    fr6 = self._draw_circle(fr6, center_x - 3, center_y, pip_radius, pip_color)
                    fr6 = self._draw_circle(fr6, center_x + 3, center_y, pip_radius, pip_color)
                    fr6 = self._draw_circle(fr6, center_x - 3, center_y + 3, pip_radius, pip_color)
                    return self._draw_circle(fr6, center_x + 3, center_y + 3, pip_radius, pip_color)

                def draw_nothing(_):
                    return fr_with_border

                # Switch based on dice value
                return jax.lax.switch(dice_value - 1,
                                      [draw_pips_1, draw_pips_2, draw_pips_3,
                                       draw_pips_4, draw_pips_5, draw_pips_6],
                                      operand=None)

            def skip_dice(_):
                return fr

            return jax.lax.cond(dice_value > 0, draw_active_dice, skip_dice, operand=None)

        return jax.lax.fori_loop(0, 4, draw_single_dice, frame)

    @partial(jax.jit, static_argnums=(0,))
    def _highlight_checker(self, frame, point_idx, state, color=jnp.array([255, 0, 0], dtype=jnp.uint8)):
        """Highlight the *visible* checker at a given point (topmost one)."""
        pos = self.triangle_positions[point_idx]
        base_x = pos[0] + self.triangle_width // 2
        base_y = pos[1]

        white_count = state.board[0, point_idx]
        black_count = state.board[1, point_idx]

        # Which color occupies the point?
        is_white = white_count > 0
        checker_color = jax.lax.cond(is_white, lambda _: self.color_white_checker, lambda _: self.color_black_checker, operand=None)
        count = jax.lax.cond(is_white, lambda _: white_count, lambda _: black_count, operand=None)

        # Top vs bottom orientation
        is_bottom = point_idx >= 12
        stack_direction = jax.lax.select(is_bottom, -1, 1)

        base_y = jax.lax.select(is_bottom, base_y + self.triangle_height - self.checker_radius, base_y + self.checker_radius,)

        # Position of topmost checker in the stack
        y_offset = (count - 1) * self.checker_stack_offset * stack_direction
        checker_y = base_y + y_offset

        # Draw the checker normally, then outline it
        frame = self._draw_circle(frame, base_x, checker_y, self.checker_radius, checker_color)

        # Outline ring
        outer_r = self.checker_radius + 2
        inner_r = self.checker_radius
        frame = self._draw_circle(frame, base_x, checker_y, outer_r, color)
        frame = self._draw_circle(frame, base_x, checker_y, inner_r, checker_color)
        return frame

    @partial(jax.jit, static_argnums=(0,))
    def _highlight_dice(self, frame, dice_index, color=jnp.array([255, 0, 0], dtype=jnp.uint8)):
        """Draw only an outline border around a dice."""
        dice_size = 12
        dice_x_start = self.frame_width // 2 - 15
        dice_y = self.frame_height // 2 - dice_size // 2
        dice_x = dice_x_start + dice_index * 15

        border_width = 2

        # Outline only — outer rect in highlight color, then redraw inner in transparent way
        frame = self._draw_rectangle(
            frame,
            dice_x - border_width,
            dice_y - border_width,
            dice_size + 2 * border_width,
            border_width,  # top
            color,
        )
        frame = self._draw_rectangle(
            frame,
            dice_x - border_width,
            dice_y + dice_size,
            dice_size + 2 * border_width,
            border_width,  # bottom
            color,
        )
        frame = self._draw_rectangle(
            frame,
            dice_x - border_width,
            dice_y,
            border_width,
            dice_size,
            color,
        )
        frame = self._draw_rectangle(
            frame,
            dice_x + dice_size,
            dice_y,
            border_width,
            dice_size,
            color,
        )

        return frame

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

        frame = self._draw_bar_checkers(frame,
                                        jnp.maximum(state.board[0, 24], 0),
                                        jnp.maximum(state.board[1, 24], 0))
        frame = self._draw_home_checkers(frame,
                                         jnp.maximum(state.board[0, 25], 0),
                                         jnp.maximum(state.board[1, 25], 0))

        frame = self._draw_dice(frame, state.dice)

        # Always highlight last move & dice if available
        from_point, to_point = state.last_move

        frame = jax.lax.cond(
            from_point >= 0,
            lambda fr: self._highlight_checker(fr, from_point, state, jnp.array([255, 0, 0], dtype=jnp.uint8)),
            lambda fr: fr,
            frame,
        )
        frame = jax.lax.cond(
            to_point >= 0,
            lambda fr: self._highlight_checker(fr, to_point, state, jnp.array([0, 255, 0], dtype=jnp.uint8)),
            lambda fr: fr,
            frame,
        )
        frame = jax.lax.cond(
            state.last_dice >= 0,
            lambda fr: self._highlight_dice(fr, state.last_dice),
            lambda fr: fr,
            frame,
        )

        return frame