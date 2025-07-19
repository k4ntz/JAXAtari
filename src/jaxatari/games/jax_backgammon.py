# Third party imports
import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple, Tuple, Any, List
import pygame
from jax import Array
import os
from pathlib import Path

# Project imports
from src.jaxatari.environment import JaxEnvironment
from src.jaxatari.renderers import JAXGameRenderer
from src.jaxatari.rendering import jax_rendering_utils as jr

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

        # Pre-compute all possible moves for fast validation
        self.action_space = jnp.array([(i, j) for i in range(26) for j in range(26)], dtype=jnp.int32)
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

    def step(self, state: BackgammonState, action: Tuple[int, int]) -> Tuple[jnp.ndarray, BackgammonState, float, bool, dict]:
        """Perform a step in the environment, applying the action and returning the new state."""
        obs, new_state, reward, done, info, new_key = self.step_impl(state, action, state.key)
        new_state = new_state._replace(key=new_key)
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
    def __init__(self, env: JaxBackgammonEnv):
        super().__init__()
        self.asset_dir = Path("src/jaxatari/games/sprites/backgammon")

        self.SPRITE_BOARD = jnp.expand_dims(jr.loadFrame(self.asset_dir / "board.npy"), axis=0)
        self.SPRITE_WHITE = jnp.expand_dims(jr.loadFrame(self.asset_dir / "white_checker.npy"), axis=0)
        self.SPRITE_BLACK = jnp.expand_dims(jr.loadFrame(self.asset_dir / "black_checker.npy"), axis=0)
        self.SPRITE_DICE = [
            jnp.expand_dims(jr.loadFrame(self.asset_dir / f"dice_{i}.npy"), axis=0)
            for i in range(1, 7)
        ]
    def load_and_pad_png_sprites(folder_path: str, filenames: list[str]) -> jnp.ndarray:
        """Loads .png sprites from disk and pads them to match dimensions."""
        sprites = []
        for name in filenames:
            img = Image.open(os.path.join(folder_path, name)).convert("RGBA")
            arr = np.array(img)
            sprites.append(jnp.array(arr))

        padded_sprites, _ = jr.pad_to_match(sprites)
        return jnp.stack(padded_sprites)


    def text_to_indices(self, text: str) -> jnp.ndarray:
        """Convert a string to indices in self.char_sprites"""
        indices = [self.char_map[c] for c in text if c in self.char_map]
        return jnp.array(indices, dtype=jnp.int32)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        raster = jr.create_initial_frame(width=1024, height=640)
        raster = jr.render_at(raster, 0, 0, jr.get_sprite_frame(self.SPRITE_BOARD, 0))

        for point in range(24):
            for player in [0, 1]:
                count = state.board[player, point]
                for i in range(count):
                    sprite = self.SPRITE_WHITE if player == 0 else self.SPRITE_BLACK
                    x, y = self.get_checker_position(point, i, player)
                    raster = jr.render_at(raster, x, y, jr.get_sprite_frame(sprite, 0))

        for i in range(2):
            die_val = state.dice[i]
            if die_val > 0:
                die_sprite = self.SPRITE_DICE[int(die_val) - 1]
                raster = jr.render_at(raster, 480 + i * 60, 10, jr.get_sprite_frame(die_sprite, 0))

        # Score display
        white_digits = jr.int_to_digits(state.board[0, 25], max_digits=2)
        black_digits = jr.int_to_digits(state.board[1, 25], max_digits=2)

        raster = jr.render_label(raster, 60, 10, white_digits, self.char_sprites)
        raster = jr.render_label(raster, 860, 10, black_digits, self.char_sprites)

        # Current player label
        player_text = "WHITE" if state.current_player == 1 else "BLACK"
        label_indices = self.text_to_indices(player_text)
        raster = jr.render_label(raster, 440, 580, label_indices, self.char_sprites)

        # Game over text
        if state.is_game_over:
            over_indices = self.text_to_indices("GAMEOVER")
            raster = jr.render_label(raster, 400, 300, over_indices, self.char_sprites)

        return raster

    def get_checker_position(self, point: int, index: int, player: int) -> tuple[int, int]:
        triangle_width = 40
        spacing = 15
        base_y_top = 30
        base_y_bottom = 400

        if point < 12:
            x = 40 + point * triangle_width
            y = base_y_bottom - index * spacing
        else:
            x = 40 + (23 - point) * triangle_width
            y = base_y_top + index * spacing

        return int(x), int(y)



# class BackgammonRenderer(JAXGameRenderer):
#     def __init__(self, env: JaxBackgammonEnv):
#         self.env = env
#
#     def render(self, state: BackgammonState) -> str:
#         """Render the current state of the game in ASCII format."""
#         board = state.board
#         output = []
#
#         # Board header
#         output.append("  12 11 10  9  8  7  |   6  5  4  3  2  1")
#         output.append("  -------------------------------------------------")
#
#         def render_row(indices: range, split_at: int):
#             row = ["  "]
#             for i in indices:
#                 point_idx = i - 1
#                 white = board[0, point_idx]
#                 black = board[1, point_idx]
#
#                 if white > 0:
#                     row.append(f"W{white} ")
#                 elif black > 0:
#                     row.append(f"B{black} ")
#                 else:
#                     row.append("•  ")
#
#                 if i == split_at:
#                     row.append(" |   ")
#             return "".join(row)
#
#         # Top row (points 12–1)
#         output.append(render_row(range(12, 0, -1), 7))
#
#         # Middle divider
#         output.append("  -------------------------------------------------")
#
#         # Bottom row (points 13–24)
#         output.append(render_row(range(13, 25), 18))
#
#         # Footer
#         output.append("  -------------------------+------------------------")
#         output.append("  13 14 15 16 17 18  |   19 20 21 22 23 24")
#
#         # Bar and Home
#         output.append("")
#         output.append(f"Bar: White: {board[0, self.env.consts.BAR_INDEX]}, Black: {board[1, self.env.consts.BAR_INDEX]}")
#         output.append(f"Home: White: {board[0, self.env.consts.HOME_INDEX]}, Black: {board[1, self.env.consts.HOME_INDEX]}")
#
#         # Game status
#         output.append("")
#         output.append(f"Current player: {'White' if state.current_player == self.env.consts.WHITE else 'Black'}")
#         output.append(f"Dice: {state.dice}")
#         output.append(f"Game over: {state.is_game_over}")
#
#         return "\n".join(output)
#
#     def display(self, state: BackgammonState) -> None:
#         """Print the rendered state to the console."""
#         print(self.render(state))
#
#     def close(self):
#         pass


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
