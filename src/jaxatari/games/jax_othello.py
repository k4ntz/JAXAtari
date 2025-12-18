from jax._src.pjit import JitWrapped
import os
from functools import partial
from typing import NamedTuple, Tuple
import jax.lax
import jax.numpy as jnp
import chex

import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action


# --- Othello Konstanten ---
class OthelloConstants(NamedTuple):
    WIDTH: int = 160
    HEIGHT: int = 210

    BOARD_SIZE: int = 8 
    EMPTY: int = 0
    PLAYER_1: int = 1
    PLAYER_2: int = 2

    BACKGROUND_COLOR: Tuple[int, int, int] = (0, 80, 0)
    BOARD_COLOR: Tuple[int, int, int] = (0, 0, 0)
    PLAYER_1_COLOR: Tuple[int, int, int] = (255, 255, 255)
    PLAYER_2_COLOR: Tuple[int, int, int] = (0, 0, 0)
    CURSOR_COLOR: Tuple[int, int, int] = (255, 0, 0)

    INPUT_DELAY: int = 12

    START_BOARD: chex.Array = jnp.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 2, 1, 0, 0, 0],
        [0, 0, 0, 1, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=jnp.int32)

    BOARD_TOP_LEFT_X: int = 18
    BOARD_TOP_LEFT_Y: int = 22
    CELL_SIZE_hor: int = 16
    CELL_SIZE_ver: int = 22

    # ------Score---------#
    SCORE_P1_X: int = 32
    SCORE_P1_Y: int = 4
    SCORE_P2_X: int = 112
    SCORE_P2_Y: int = 4

    ENEMY_WEIGHTS: chex.Array = jnp.array([
        [64,   -25,    10,      4,      4,      10,      -25,    64],
        [-25,   -25,    -8,      -8,      -8,      -8,      -25,    -25],
        [10,    -8,      4,      0,      0,      4,      -8,      10],
        [4,      -8,      0,      0,      0,      0,      -8,      4],
        [4,      -8,      0,      0,      0,      0,      -8,      4],
        [10,    -8,      4,      0,      0,      4,      -8,      10],
        [-25,   -25,    -8,      -8,      -8,      -8,      -25,    -25],
        [64,   -25,    10,      4,      4,      10,      -25,    64]
    ], dtype=jnp.int32)

    # --- TIMING CONSTANTS ---
    FRAMES_TO_PLACE: int = 0
    FRAMES_TO_FIRST_FLIP: int = 30
    FRAMES_BETWEEN_FLIPS: int = 32
    FRAMES_AFTER_FLIP: int = 28 
    
    CURSOR_START_FRAME: int = 2

    PHASE_PLAY: int = 0
    PHASE_ANIMATION: int = 1

    SUBPHASE_NONE: int = 0
    SUBPHASE_INITIAL_PLACE: int = 1
    SUBPHASE_FLIPPING: int = 2
    SUBPHASE_END_BUFFER: int = 3 


# --- Othello Spielzustand ---
class OthelloState(NamedTuple):
    board: chex.Array
    cursor_x: chex.Array
    cursor_y: chex.Array
    player_1_score: chex.Array
    player_2_score: chex.Array
    current_player: chex.Array
    passes: chex.Array
    step_counter: chex.Array
    input_timer: chex.Array
    phase: chex.Array
    pieces_to_flip: chex.Array
    animation_timer: chex.Array
    target_player: chex.Array
    target_x: chex.Array
    target_y: chex.Array
    animation_sub_phase: chex.Array
    lfsr_state: chex.Array
    turn_start_frame: chex.Array
    hide_cursor: chex.Array


# --- Othello Beobachtung ---
class OthelloObservation(NamedTuple):
    board: jnp.ndarray
    cursor_x: jnp.ndarray
    cursor_y: jnp.ndarray
    score_player_1: jnp.ndarray
    score_player_2: jnp.ndarray
    current_player: jnp.ndarray


# --- Othello Info ---
class OthelloInfo(NamedTuple):
    time: jnp.ndarray
    legal_moves: jnp.ndarray


class JaxOthello(JaxEnvironment[OthelloState, OthelloObservation, OthelloInfo, OthelloConstants]):
    def __init__(self, consts: OthelloConstants = None, reward_funcs: list[callable] = None):
        consts = consts or OthelloConstants()
        super().__init__(consts)
        self.renderer = OthelloRenderer(self.consts)

        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs

        self.action_set = [
            Action.NOOP, Action.FIRE, Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT,
            Action.UPRIGHT, Action.UPLEFT, Action.DOWNRIGHT, Action.DOWNLEFT,
        ]
        self.obs_size = (self.consts.BOARD_SIZE * self.consts.BOARD_SIZE) + 2 + 2 + 1

    def _player_step(self, state: OthelloState, action: chex.Array) -> OthelloState:
        up = (action == Action.UP) | (action == Action.UPRIGHT) | (action == Action.UPLEFT)
        down = (action == Action.DOWN) | (action == Action.DOWNRIGHT) | (action == Action.DOWNLEFT)
        left = (action == Action.LEFT) | (action == Action.UPLEFT) | (action == Action.DOWNLEFT)
        right = (action == Action.RIGHT) | (action == Action.UPRIGHT) | (action == Action.DOWNRIGHT)
        is_moving = (up | down | left | right)

        can_move = jnp.logical_and(state.input_timer <= 0, state.phase == self.consts.PHASE_PLAY)

        def _move():
            ny = (state.cursor_y - up + down) % self.consts.BOARD_SIZE
            nx = (state.cursor_x - left + right) % self.consts.BOARD_SIZE
            return ny, nx

        new_cursor_y, new_cursor_x = jax.lax.cond(
            jnp.logical_and(is_moving, can_move),
            _move,
            lambda: (state.cursor_y, state.cursor_x)
        )

        def _update_timer():
            return jax.lax.select(
                can_move,
                self.consts.INPUT_DELAY,
                state.input_timer - 1
            )

        new_timer = jax.lax.cond(
            is_moving,
            _update_timer,
            lambda: self.consts.INPUT_DELAY 
        )
        new_timer = jnp.maximum(new_timer, 0)

        return state._replace(
            cursor_x=new_cursor_x,
            cursor_y=new_cursor_y,
            input_timer=new_timer
        )

    # --- Helper methods ---
    @partial(jax.jit, static_argnums=(0,))
    def _count_pieces(self, board: chex.Array) -> Tuple[chex.Array, chex.Array]:
        score_1 = jnp.sum(board == self.consts.PLAYER_1)
        score_2 = jnp.sum(board == self.consts.PLAYER_2)
        return score_1, score_2

    @partial(jax.jit, static_argnums=(0,))
    def _is_valid_move(self, board: chex.Array, y: chex.Array, x: chex.Array, player: chex.Array) -> bool:
        opponent = jnp.where(player == self.consts.PLAYER_1, self.consts.PLAYER_2, self.consts.PLAYER_1)
        is_empty = (board[y, x] == self.consts.EMPTY)
        directions = jnp.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]], dtype=jnp.int32)
        init_val = False

        def _check_dir_loop(i, is_valid_so_far):
            def _check_this_dir():
                dy, dx = directions[i]
                dir_init_val = 0
                def _dir_scan_loop(j, loop_state):
                    cy = y + (j + 1) * dy
                    cx = x + (j + 1) * dx
                    def _check_cell():
                        in_bounds = jnp.logical_and(
                            jnp.logical_and(cy >= 0, cy < self.consts.BOARD_SIZE),
                            jnp.logical_and(cx >= 0, cx < self.consts.BOARD_SIZE)
                        )
                        def _in_bounds_check():
                            cell = board[cy, cx]
                            is_opp = (cell == opponent)
                            is_player = (cell == player)
                            state_0_logic = jnp.where(is_opp, 1, 3)
                            state_1_logic = jnp.where(is_opp, 1, jnp.where(is_player, 2, 3))
                            return jax.lax.switch(loop_state, [lambda: state_0_logic, lambda: state_1_logic])
                        return jnp.where(in_bounds, _in_bounds_check(), 3)
                    return jnp.where(jnp.logical_or(loop_state == 2, loop_state == 3), loop_state, _check_cell())
                final_state = jax.lax.fori_loop(0, self.consts.BOARD_SIZE, _dir_scan_loop, dir_init_val)
                return (final_state == 2)
            is_this_dir_valid = jax.lax.cond(is_valid_so_far, lambda: False, _check_this_dir)
            return jnp.logical_or(is_valid_so_far, is_this_dir_valid)
        is_any_dir_valid = jax.lax.fori_loop(0, 8, _check_dir_loop, init_val)
        return jnp.logical_and(is_empty, is_any_dir_valid)

    @partial(jax.jit, static_argnums=(0,))
    def _has_any_valid_move(self, board: chex.Array, player: chex.Array) -> bool:
        def _check_pos(i, found_valid):
            return jax.lax.cond(
                found_valid,
                lambda: True,
                lambda: self._is_valid_move(board, i // self.consts.BOARD_SIZE, i % self.consts.BOARD_SIZE, player)
            )
        return jax.lax.fori_loop(0, self.consts.BOARD_SIZE * self.consts.BOARD_SIZE, _check_pos, False)

    @partial(jax.jit, static_argnums=(0,))
    def _get_flip_mask(self, board: chex.Array, y: chex.Array, x: chex.Array, player: chex.Array) -> chex.Array:
        opponent = jnp.where(player == self.consts.PLAYER_1, self.consts.PLAYER_2, self.consts.PLAYER_1)
        directions = jnp.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]], dtype=jnp.int32)
        mask = jnp.zeros_like(board, dtype=jnp.bool_)
        init_val = mask

        def _flip_dir_loop(i, current_mask):
            dy, dx = directions[i]
            scan_init_val = (0, 0)
            def _dir_scan_loop(j, val):
                loop_state, num_flips = val
                cy = y + (j + 1) * dy
                cx = x + (j + 1) * dx
                def _check_cell():
                    in_bounds = jnp.logical_and(
                        jnp.logical_and(cy >= 0, cy < self.consts.BOARD_SIZE),
                        jnp.logical_and(cx >= 0, cx < self.consts.BOARD_SIZE)
                    )
                    def _in_bounds_check():
                        cell = board[cy, cx]
                        is_opp = (cell == opponent)
                        is_player = (cell == player)
                        state_0_logic = jax.lax.cond(is_opp, lambda: (1, 1), lambda: (3, 0))
                        state_1_logic = jax.lax.cond(is_opp, lambda: (1, num_flips + 1),
                                                     lambda: jax.lax.cond(is_player, lambda: (2, num_flips),
                                                                          lambda: (3, 0)))
                        return jax.lax.switch(loop_state, [lambda: state_0_logic, lambda: state_1_logic])
                    return jax.lax.cond(in_bounds, _in_bounds_check, lambda: (3, 0))
                condition = jnp.logical_or(loop_state == 2, loop_state == 3)
                return jax.lax.cond(condition, lambda: (loop_state, num_flips), _check_cell)
            final_state, num_to_flip = jax.lax.fori_loop(0, self.consts.BOARD_SIZE, _dir_scan_loop, scan_init_val)
            def _mark_flips():
                flip_init_val = current_mask
                def _mark_loop(k, mask_to_update):
                    fy = y + (k + 1) * dy
                    fx = x + (k + 1) * dx
                    return mask_to_update.at[fy, fx].set(True)
                return jax.lax.fori_loop(0, num_to_flip, _mark_loop, flip_init_val)
            return jax.lax.cond(final_state == 2, _mark_flips, lambda: current_mask)
        final_mask = jax.lax.fori_loop(0, 8, _flip_dir_loop, init_val)
        return final_mask


    # @partial(jax.jit, static_argnums=(0,))
    # def _ai_turn(self, state: OthelloState) -> OthelloState:
    #     player = self.consts.PLAYER_2
    #     init_val = (-9999, -1, state.lfsr_state)
    #
    #     def _scan_loop(i, carry):
    #         best_s, best_idx, curr_lfsr = carry
    #         idx = 63 - i
    #         y = idx // self.consts.BOARD_SIZE
    #         x = idx % self.consts.BOARD_SIZE
    #         is_valid = self._is_valid_move(state.board, y, x, player)
    #
    #         def _eval_move(lfsr):
    #             mask = self._get_flip_mask(state.board, y, x, player)
    #             flips = jnp.sum(mask).astype(jnp.int32)
    #             total_pieces = state.player_1_score + state.player_2_score
    #             score_flips = jax.lax.select(
    #                 (total_pieces > 19) & (total_pieces < 66),
    #                 -flips,
    #                 flips
    #             )
    #             score_pos = self._get_dynamic_position_score(state.board, y, x)
    #             final_score = score_pos + score_flips
    #             is_better = (final_score > best_s)
    #             is_equal = (final_score == best_s)
    #             next_lfsr = self._atari_lfsr_step(lfsr)
    #             should_swap = is_better | (is_equal & ((next_lfsr & 3) == 0))
    #             new_best_s = jax.lax.select(should_swap, final_score, best_s)
    #             new_best_idx = jax.lax.select(should_swap, idx, best_idx)
    #             return new_best_s, new_best_idx, next_lfsr
    #
    #         next_lfsr_dummy = self._atari_lfsr_step(curr_lfsr)
    #         return jax.lax.cond(
    #             is_valid,
    #             lambda: _eval_move(curr_lfsr),
    #             lambda: (best_s, best_idx, next_lfsr_dummy)
    #         )
    #
    #     best_score_final, best_idx_final, final_lfsr = jax.lax.fori_loop(0, 64, _scan_loop, init_val)
    #
    #     def _execute_move():
    #         y = best_idx_final // self.consts.BOARD_SIZE
    #         x = best_idx_final % self.consts.BOARD_SIZE
    #         flip_mask = self._get_flip_mask(state.board, y, x, player)
    #
    #         return state._replace(
    #             cursor_y=jnp.array(y, dtype=jnp.int32),
    #             phase=jnp.array(self.consts.PHASE_ANIMATION, dtype=jnp.int32),
    #             animation_sub_phase=jnp.array(self.consts.SUBPHASE_INITIAL_PLACE, dtype=jnp.int32),
    #             pieces_to_flip=flip_mask,
    #             target_player=jnp.array(player, dtype=jnp.int32),
    #             target_x=jnp.array(x, dtype=jnp.int32),
    #             target_y=jnp.array(y, dtype=jnp.int32),
    #             animation_timer=jnp.array(self.consts.FRAMES_TO_PLACE, dtype=jnp.int32),
    #             passes=jnp.array(0, dtype=jnp.int32),
    #             lfsr_state=final_lfsr
    #         )
    #
    #     def _pass_turn():
    #         return state._replace(
    #             current_player=jnp.array(self.consts.PLAYER_1, dtype=jnp.int32),
    #             passes=state.passes + 1,
    #             lfsr_state=final_lfsr,
    #             turn_start_frame=state.step_counter + 1
    #         )
    #
    #     return jax.lax.cond(best_idx_final != -1, _execute_move, _pass_turn)

    @partial(jax.jit, static_argnums=(0,))
    def _ai_turn(self, state: OthelloState) -> OthelloState:
        # --- KONFIGURATION ---
        SEARCH_DEPTH = 2  # Suchtiefe (2 = KI Zug + Gegner Antwort)
        AI_PLAYER = self.consts.PLAYER_2
        OPPONENT = self.consts.PLAYER_1

        # --- 1. BEWERTUNG (Score als Float!) ---
        def _evaluate_board(board):
            p2_count = jnp.sum(board == AI_PLAYER)
            p1_count = jnp.sum(board == OPPONENT)
            # FIX: astype(jnp.float32), damit es kompatibel zu den 1e9 Werten ist!
            return (p2_count - p1_count).astype(jnp.float32)

        # --- 2. ZUG SIMULIEREN ---
        def _simulate_move_logic(board, y, x, player):
            flip_mask = self._get_flip_mask(board, y, x, player)
            board_next = board.at[y, x].set(player)
            return jnp.where(flip_mask, player, board_next)

        # --- 3. MINIMAX (Rekursiv via Loop) ---
        def run_minimax(board, current_player, depth, maximizing):
            if depth == 0:
                return _evaluate_board(board)

            # Wir scannen alle 64 Felder
            def _scan_moves(carry, idx):
                y, x = idx // self.consts.BOARD_SIZE, idx % self.consts.BOARD_SIZE
                is_valid = self._is_valid_move(board, y, x, current_player)

                def _do_branch():
                    new_board = _simulate_move_logic(board, y, x, current_player)
                    next_player = jnp.where(current_player == AI_PLAYER, OPPONENT, AI_PLAYER)
                    return run_minimax(new_board, next_player, depth - 1, not maximizing)

                # Wenn ungültig: -1e9 (Float)
                invalid_val = -1e9 if maximizing else 1e9
                # Jetzt geben beide Zweige floats zurück -> Kein Crash mehr!
                val = jax.lax.cond(is_valid, _do_branch, lambda: invalid_val)

                # Maximize oder Minimize update
                new_best = jnp.maximum(carry, val) if maximizing else jnp.minimum(carry, val)
                return new_best, None

            init_val = -1e9 if maximizing else 1e9
            best_val, _ = jax.lax.scan(_scan_moves, init_val, jnp.arange(64))

            # Fallback falls kein Zug möglich (Passen): Board jetzt bewerten
            has_moves = self._has_any_valid_move(board, current_player)
            return jax.lax.select(has_moves, best_val, _evaluate_board(board))

        # --- 4. BESTEN ZUG FINDEN (Root) ---

        def _get_best_move_idx():
            def _score_root_move(idx):
                y, x = idx // self.consts.BOARD_SIZE, idx % self.consts.BOARD_SIZE
                is_valid = self._is_valid_move(state.board, y, x, AI_PLAYER)

                def _eval():
                    new_board = _simulate_move_logic(state.board, y, x, AI_PLAYER)
                    # Tiefe - 1, Gegner ist dran (minimizing)
                    return run_minimax(new_board, OPPONENT, SEARCH_DEPTH - 1, False)

                # Hier auch wichtig: lambda muss float zurückgeben (-1e9 ist float)
                return jax.lax.cond(is_valid, _eval, lambda: -1e9)

            # Berechne Scores für alle 64 Felder parallel
            all_scores = jax.vmap(_score_root_move)(jnp.arange(64))

            # --- NEU: ZUFALL BEI GLEICHSTAND (Tie-Breaker) ---
            # Wir nutzen den lfsr_state als Seed für einen RNG Key
            rng_key = jax.random.PRNGKey(state.lfsr_state)

            # Erzeuge zufälliges Rauschen für jedes der 64 Felder (z.B. zwischen 0.0 und 0.1)
            noise = jax.random.uniform(rng_key, shape=(64,), minval=0.0, maxval=0.1)

            # Addiere das Rauschen auf die Scores.
            # Da ungültige Moves -1e9 haben, macht +0.1 sie nicht plötzlich gültig.
            # Aber bei validen Moves (z.B. beide Score 5.0) entscheidet jetzt das Rauschen.
            scores_with_noise = all_scores + noise

            # Wichtig: Index basiert auf Noise-Score, aber der Rückgabewert 'best_score'
            # sollte der echte Score (ohne Noise) sein, damit die Logik unten sauber bleibt.
            return jnp.argmax(scores_with_noise), jnp.max(all_scores)

        best_idx, best_score = _get_best_move_idx()
        next_lfsr = self._atari_lfsr_step(state.lfsr_state)

        # --- 5. EXECUTE ---
        def _execute_move():
            y = best_idx // self.consts.BOARD_SIZE
            x = best_idx % self.consts.BOARD_SIZE
            flip_mask = self._get_flip_mask(state.board, y, x, AI_PLAYER)

            return state._replace(
                cursor_y=jnp.array(y, dtype=jnp.int32),
                cursor_x=jnp.array(x, dtype=jnp.int32),  
                
                phase=jnp.array(self.consts.PHASE_ANIMATION, dtype=jnp.int32),
                animation_sub_phase=jnp.array(self.consts.SUBPHASE_INITIAL_PLACE, dtype=jnp.int32),
                pieces_to_flip=flip_mask,
                target_player=jnp.array(AI_PLAYER, dtype=jnp.int32),
                target_x=jnp.array(x, dtype=jnp.int32),
                target_y=jnp.array(y, dtype=jnp.int32),
                animation_timer=jnp.array(self.consts.FRAMES_TO_PLACE, dtype=jnp.int32),
                passes=jnp.array(0, dtype=jnp.int32),
                lfsr_state=next_lfsr
            )

        def _pass_turn():
            return state._replace(
                current_player=jnp.array(self.consts.PLAYER_1, dtype=jnp.int32),
                passes=state.passes + 1,
                lfsr_state=next_lfsr,
                turn_start_frame=state.step_counter + 1
            )

        return jax.lax.cond(best_score > -0.9e9, _execute_move, _pass_turn)

    @partial(jax.jit, static_argnums=(0,))
    def _atari_lfsr_step(self, lfsr_state: chex.Array) -> chex.Array:
        d9 = (lfsr_state >> 8) & 0xFF
        d8 = lfsr_state & 0xFF
        bVar1 = (d9 >> 7) & 1
        xor_val = (d8 ^ d9) & 0xFF
        feedback = (xor_val >> 6) & 1
        new_d9 = ((d9 << 1) & 0xFF) | feedback
        new_d8 = ((d8 << 1) & 0xFF) | bVar1
        return ((new_d9 << 8) | new_d8).astype(jnp.int32)

    @partial(jax.jit, static_argnums=(0,))
    def _get_dynamic_position_score(self, board, y, x):
        raw_weight = self.consts.ENEMY_WEIGHTS[y, x]
        def _check_corner():
            cy = jax.lax.select(y < 4, 0, 7)
            cx = jax.lax.select(x < 4, 0, 7)
            corner_taken = (board[cy, cx] != self.consts.EMPTY)
            return jax.lax.select(corner_taken, 10, raw_weight)
        return jax.lax.cond(raw_weight == -25, _check_corner, lambda: raw_weight)

    @partial(jax.jit, static_argnums=(0,))
    def _attempt_player_move(self, state: OthelloState) -> Tuple[OthelloState, bool]:
        y, x = state.cursor_y, state.cursor_x
        player = state.current_player
        is_valid = self._is_valid_move(state.board, y, x, player)

        def _initiate_move():
            flip_mask = self._get_flip_mask(state.board, y, x, player)
            
            new_state = state._replace(
                phase=jnp.array(self.consts.PHASE_ANIMATION, dtype=jnp.int32),
                animation_sub_phase=jnp.array(self.consts.SUBPHASE_INITIAL_PLACE, dtype=jnp.int32),
                pieces_to_flip=flip_mask,
                target_player=player,
                target_x=x,
                target_y=y,
                animation_timer=jnp.array(self.consts.FRAMES_TO_PLACE, dtype=jnp.int32),
                passes=jnp.array(0, dtype=jnp.int32)
            )
            return new_state._replace(hide_cursor=jnp.array(False, dtype=jnp.bool_)), True
        
        def _fail_move():
            return state._replace(hide_cursor=jnp.array(True, dtype=jnp.bool_)), False

        return jax.lax.cond(is_valid, _initiate_move, _fail_move)

    def _game_logic_step(self, state: OthelloState, action: chex.Array) -> OthelloState:
        state = state._replace(lfsr_state=self._atari_lfsr_step(state.lfsr_state))
        
        # --- PHASE 1: SPIELZUG LOGIK ---
        def _play_phase(s):
            is_ai = (s.current_player == self.consts.PLAYER_2)

            def _human_turn(current_s):
                has_moves = self._has_any_valid_move(current_s.board, current_s.current_player)
                def _process_input(s_in):
                    is_fire = (action == Action.FIRE)
                    
                    s_reset = s_in._replace(hide_cursor=jnp.array(False, dtype=jnp.bool_))
                    
                    return jax.lax.cond(
                        is_fire,
                        self._attempt_player_move,
                        lambda _s: (_s, False),
                        s_reset
                    )
                
                def _auto_pass(s_in):
                    next_player = jnp.where(s_in.current_player == self.consts.PLAYER_1, 
                                            self.consts.PLAYER_2, self.consts.PLAYER_1)
                    return s_in._replace(
                        current_player=next_player,
                        passes=s_in.passes + 1,
                        turn_start_frame=s_in.step_counter + 1
                    ), True 

                return jax.lax.cond(has_moves, _process_input, _auto_pass, current_s)

            def _ai_turn_wrapper(current_s):
                return self._ai_turn(current_s), True

            new_s, move_made = jax.lax.cond(
                is_ai,
                _ai_turn_wrapper,
                _human_turn,
                s
            )
            return new_s

        # --- PHASE 2: ANIMATION ---
        def _animation_phase(s):
            timer_done = (s.animation_timer <= 0)
            
            # --- SUBPHASE 1: INITIAL PLACE ---
            def _initial_place_logic(current_s):
                def _do_place_and_score():
                    new_board = current_s.board.at[current_s.target_y, current_s.target_x].set(current_s.target_player)
                    
                    is_p1 = (current_s.target_player == self.consts.PLAYER_1)
                    s1 = current_s.player_1_score + jnp.where(is_p1, 1, 0)
                    s2 = current_s.player_2_score + jnp.where(is_p1, 0, 1)
                    
                    return current_s._replace(
                        board=new_board,
                        player_1_score=s1,
                        player_2_score=s2,
                        animation_sub_phase=jnp.array(self.consts.SUBPHASE_FLIPPING, dtype=jnp.int32),
                        animation_timer=jnp.array(self.consts.FRAMES_TO_FIRST_FLIP, dtype=jnp.int32)
                    )
                
                def _wait():
                    return current_s._replace(animation_timer=current_s.animation_timer - 1)
                    
                return jax.lax.cond(timer_done, _do_place_and_score, _wait)

            # --- SUBPHASE 2: FLIPPING ---
            def _flipping_logic(current_s):
                def _process_flip():
                    flat_mask = current_s.pieces_to_flip.flatten()
                    idx = jnp.argmax(flat_mask) 
                    has_flip = flat_mask[idx]
                    
                    def _do_flip():
                        y = idx // self.consts.BOARD_SIZE
                        x = idx % self.consts.BOARD_SIZE
                        
                        # Board Update
                        new_board = current_s.board.at[y, x].set(current_s.target_player)
                        
                        # Score Update
                        is_p1 = (current_s.target_player == self.consts.PLAYER_1)
                        s1 = current_s.player_1_score + jnp.where(is_p1, 1, -1)
                        s2 = current_s.player_2_score + jnp.where(is_p1, -1, 1)
                        
                        # Mask Update
                        new_mask = current_s.pieces_to_flip.at[y, x].set(False)
                        
                        # Prüfen, ob dies der letzte Flip war
                        is_last_flip = ~jnp.any(new_mask)
                        
                        def _prepare_buffer():
                            # Wenn es der letzte Flip war: SOFORT Spieler wechseln
                            next_player = jnp.where(current_s.target_player == self.consts.PLAYER_1, 
                                                    self.consts.PLAYER_2, self.consts.PLAYER_1)
                            
                            return current_s._replace(
                                board=new_board,
                                player_1_score=s1,
                                player_2_score=s2,
                                pieces_to_flip=new_mask,
                                current_player=next_player, 
                                animation_sub_phase=jnp.array(self.consts.SUBPHASE_END_BUFFER, dtype=jnp.int32),
                                # Timer für den Puffer setzen
                                animation_timer=jnp.array(self.consts.FRAMES_BETWEEN_FLIPS + self.consts.FRAMES_AFTER_FLIP, dtype=jnp.int32),
                                # CHANGE: Reset blink rhythm hier, damit er genau bei Frame 0 des Zyklus beginnt!
                                turn_start_frame=current_s.step_counter + 1
                            )
                        
                        def _continue_flipping():
                            return current_s._replace(
                                board=new_board,
                                player_1_score=s1,
                                player_2_score=s2,
                                pieces_to_flip=new_mask,
                                animation_timer=jnp.array(self.consts.FRAMES_BETWEEN_FLIPS, dtype=jnp.int32)
                            )

                        return jax.lax.cond(is_last_flip, _prepare_buffer, _continue_flipping)
                    
                    def _finish_fallback():
                         return current_s._replace(
                            phase=jnp.array(self.consts.PHASE_PLAY, dtype=jnp.int32),
                            animation_sub_phase=jnp.array(self.consts.SUBPHASE_NONE, dtype=jnp.int32),
                            turn_start_frame=current_s.step_counter + 1
                        )

                    return jax.lax.cond(has_flip, _do_flip, _finish_fallback)

                def _wait():
                    return current_s._replace(animation_timer=current_s.animation_timer - 1)

                return jax.lax.cond(timer_done, _process_flip, _wait)
            
            # --- SUBPHASE 3: END BUFFER ---
            def _end_buffer_logic(current_s):
                def _finish_buffer():
                     return current_s._replace(
                        phase=jnp.array(self.consts.PHASE_PLAY, dtype=jnp.int32),
                        animation_sub_phase=jnp.array(self.consts.SUBPHASE_NONE, dtype=jnp.int32),
                        turn_start_frame=current_s.step_counter + 1
                    )
                def _wait():
                    return current_s._replace(animation_timer=current_s.animation_timer - 1)
                
                return jax.lax.cond(timer_done, _finish_buffer, _wait)

            return jax.lax.switch(
                s.animation_sub_phase,
                [lambda _s: _s, _initial_place_logic, _flipping_logic, _end_buffer_logic],
                s
            )

        new_state = jax.lax.switch(
            state.phase,
            [_play_phase, _animation_phase],
            state
        )
        
        return new_state._replace(step_counter=state.step_counter + 1)

    # ... [Reset / Step / Obs wie gehabt] ...
    def reset(self, key=None) -> Tuple[OthelloObservation, OthelloState]:
        lfsr_seed = jax.random.randint(key, (), 0, 65535).astype(jnp.int32)
        state = OthelloState(
            board=self.consts.START_BOARD,
            cursor_x=jnp.array(self.consts.BOARD_SIZE - 1, dtype=jnp.int32),
            cursor_y=jnp.array(self.consts.BOARD_SIZE - 1, dtype=jnp.int32),
            player_1_score=jnp.array(2, dtype=jnp.int32),
            player_2_score=jnp.array(2, dtype=jnp.int32),
            current_player=jnp.array(self.consts.PLAYER_1, dtype=jnp.int32),
            passes=jnp.array(0, dtype=jnp.int32),
            step_counter=jnp.array(0, dtype=jnp.int32),
            input_timer=jnp.array(self.consts.INPUT_DELAY, dtype=jnp.int32),
            phase=jnp.array(self.consts.PHASE_PLAY, dtype=jnp.int32),
            pieces_to_flip=jnp.zeros((self.consts.BOARD_SIZE, self.consts.BOARD_SIZE), dtype=jnp.bool_),
            animation_timer=jnp.array(0, dtype=jnp.int32),
            target_player=jnp.array(0, dtype=jnp.int32),
            target_x=jnp.array(0, dtype=jnp.int32),
            target_y=jnp.array(0, dtype=jnp.int32),
            animation_sub_phase=jnp.array(self.consts.SUBPHASE_NONE, dtype=jnp.int32),
            lfsr_state=lfsr_seed,
            turn_start_frame=jnp.array(self.consts.CURSOR_START_FRAME, dtype=jnp.int32),
            hide_cursor=jnp.array(False, dtype=jnp.bool_)
        )
        initial_obs = self._get_observation(state)
        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: OthelloState, action: chex.Array) -> Tuple[
        OthelloObservation, OthelloState, float, bool, OthelloInfo]:
        previous_state = state
        state_after_cursor = self._player_step(state, action)
        state_after_logic = self._game_logic_step(state_after_cursor, action)
        state = state_after_logic
        done = self._get_done(state)
        env_reward = self._get_reward(previous_state, state)
        info = self._get_info(state)
        observation = self._get_observation(state)
        return observation, state, env_reward, done, info

    def render(self, state: OthelloState) -> jnp.ndarray:
        return self.renderer.render(state)

    def _get_observation(self, state: OthelloState) -> OthelloObservation:
        return OthelloObservation(
            board=state.board,
            cursor_x=state.cursor_x,
            cursor_y=state.cursor_y,
            score_player_1=state.player_1_score,
            score_player_2=state.player_2_score,
            current_player=state.current_player
        )
        
    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: OthelloObservation) -> jnp.ndarray:
        return jnp.concatenate([
            obs.board.flatten(),
            jnp.array([obs.cursor_x]),
            jnp.array([obs.cursor_y]),
            jnp.array([obs.score_player_1]),
            jnp.array([obs.score_player_2]),
            jnp.array([obs.current_player]),
        ])

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))

    def observation_space(self) -> spaces.Dict:
        return spaces.Dict({
            "board": spaces.Box(low=0, high=2, shape=(self.consts.BOARD_SIZE, self.consts.BOARD_SIZE), dtype=jnp.int32),
            "cursor_x": spaces.Box(low=0, high=self.consts.BOARD_SIZE - 1, shape=(), dtype=jnp.int32),
            "cursor_y": spaces.Box(low=0, high=self.consts.BOARD_SIZE - 1, shape=(), dtype=jnp.int32),
            "score_player_1": spaces.Box(low=0, high=64, shape=(), dtype=jnp.int32),
            "score_player_2": spaces.Box(low=0, high=64, shape=(), dtype=jnp.int32),
            "current_player": spaces.Box(low=1, high=2, shape=(), dtype=jnp.int32),
        })

    def image_space(self) -> spaces.Box:
        return spaces.Box(low=0, high=255, shape=(self.consts.HEIGHT, self.consts.WIDTH, 3), dtype=jnp.uint8)

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: OthelloState) -> OthelloInfo:
        return OthelloInfo(time=state.step_counter, legal_moves=jnp.zeros((8, 8)))

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: OthelloState, state: OthelloState) -> float:
        reward = (state.player_1_score - state.player_2_score) - \
                 (previous_state.player_1_score - previous_state.player_2_score)
        return reward.astype(jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: OthelloState) -> bool:
        board_full = (state.player_1_score + state.player_2_score) >= (self.consts.BOARD_SIZE * self.consts.BOARD_SIZE)
        no_moves_left = (state.passes >= 2)
        return jnp.logical_or(board_full, no_moves_left)


# --- Othello Renderer ---
class OthelloRenderer(JAXGameRenderer):
    def __init__(self, consts: OthelloConstants = None):
        super().__init__()
        self.consts = consts or OthelloConstants()
        self.config = render_utils.RendererConfig(
            game_dimensions=(self.consts.HEIGHT, self.consts.WIDTH),
            channels=3,
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)

        asset_config = self._get_asset_config()
        sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/othello"
        (
            self.PALETTE, self.SHAPE_MASKS, self.BACKGROUND, self.COLOR_TO_ID, self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(asset_config, sprite_path)

    def _get_asset_config(self) -> list:
        return [
            {'name': 'background', 'type': 'background', 'file': 'background.npy'},
            {'name': 'piece_white', 'type': 'single', 'file': 'piece_white.npy'},
            {'name': 'piece_black', 'type': 'single', 'file': 'piece_black.npy'},
            {'name': 'cursor', 'type': 'single', 'file': 'cursor.npy'},
            {'name': 'empty_cursor', 'type': 'single', 'file': 'empty_cursor.npy'},
            {'name': 'score_0_player', 'type': 'single', 'file': 'player_score_0.npy'}, 
            {'name': 'score_1_player', 'type': 'single', 'file': 'player_score_1.npy'},
            {'name': 'score_2_player', 'type': 'single', 'file': 'player_score_2.npy'},
            {'name': 'score_3_player', 'type': 'single', 'file': 'player_score_3.npy'},
            {'name': 'score_4_player', 'type': 'single', 'file': 'player_score_4.npy'},
            {'name': 'score_5_player', 'type': 'single', 'file': 'player_score_5.npy'},
            {'name': 'score_6_player', 'type': 'single', 'file': 'player_score_6.npy'},
            {'name': 'score_7_player', 'type': 'single', 'file': 'player_score_7.npy'},
            {'name': 'score_8_player', 'type': 'single', 'file': 'player_score_8.npy'},
            {'name': 'score_9_player', 'type': 'single', 'file': 'player_score_9.npy'},
            {'name': 'score_0_enemy', 'type': 'single', 'file': 'enemy_score_0.npy'},
            {'name': 'score_1_enemy', 'type': 'single', 'file': 'enemy_score_1.npy'},
            {'name': 'score_2_enemy', 'type': 'single', 'file': 'enemy_score_2.npy'},
            {'name': 'score_3_enemy', 'type': 'single', 'file': 'enemy_score_3.npy'},
            {'name': 'score_4_enemy', 'type': 'single', 'file': 'enemy_score_4.npy'},
            {'name': 'score_5_enemy', 'type': 'single', 'file': 'enemy_score_5.npy'},
            {'name': 'score_6_enemy', 'type': 'single', 'file': 'enemy_score_6.npy'},
            {'name': 'score_7_enemy', 'type': 'single', 'file': 'enemy_score_7.npy'},
            {'name': 'score_8_enemy', 'type': 'single', 'file': 'enemy_score_8.npy'},
            {'name': 'score_9_enemy', 'type': 'single', 'file': 'enemy_score_9.npy'},
        ]

    @partial(jax.jit, static_argnums=(0,))
    def _draw_pieces(self, state: OthelloState, raster: chex.Array) -> chex.Array:
        mask_p1 = self.SHAPE_MASKS["piece_white"]
        mask_p2 = self.SHAPE_MASKS["piece_black"]

        def _draw_cell_loop(i, current_raster):
            x = i % self.consts.BOARD_SIZE
            y = i // self.consts.BOARD_SIZE
            px = self.consts.BOARD_TOP_LEFT_X + (x * self.consts.CELL_SIZE_hor)
            py = self.consts.BOARD_TOP_LEFT_Y + (y * self.consts.CELL_SIZE_ver)
            cell_state = state.board[y, x]
            current_raster = jax.lax.switch(
                cell_state,
                [
                    lambda r: r,
                    lambda r: self.jr.render_at(r, px, py, mask_p1),
                    lambda r: self.jr.render_at(r, px, py, mask_p2)
                ],
                current_raster
            )
            return current_raster

        raster = jax.lax.fori_loop(0, self.consts.BOARD_SIZE * self.consts.BOARD_SIZE, _draw_cell_loop, raster)
        return raster

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: OthelloState):
        raster = self.jr.create_object_raster(self.BACKGROUND)
        raster = self._draw_pieces(state, raster)

        # --- CURSOR & BLINK LOGIK ---
        show_cursor_globally = (state.step_counter >= self.consts.CURSOR_START_FRAME)

        cursor_mask = self.SHAPE_MASKS["cursor"]
        piece_black_mask = self.SHAPE_MASKS["piece_black"]
        empty_cursor_mask = self.SHAPE_MASKS["empty_cursor"]
        
        cursor_pixel_x = self.consts.BOARD_TOP_LEFT_X + state.cursor_x * self.consts.CELL_SIZE_hor
        cursor_pixel_y = self.consts.BOARD_TOP_LEFT_Y + state.cursor_y * self.consts.CELL_SIZE_ver
        
        BLINK_SPEED = 1
        
        # Zyklus 0, 1, 2, 3
        blink_cycle = (jnp.maximum(0, state.step_counter - state.turn_start_frame) // BLINK_SPEED) % 4
        
        cell_under_cursor = state.board[state.cursor_y, state.cursor_x]
        current_player = state.current_player
        
        def _get_draw_flags():
            default_normal = (blink_cycle == 0)
            default_empty = False
            
            # P1 (Weiß/Spieler) Logic
            # Wenn auf Gegner-Feld (Switch Moment): default_normal greift (1x an, 3x aus)
            def _p1_logic():
                is_own = (cell_under_cursor == self.consts.PLAYER_1)
                p1_normal = jax.lax.cond(is_own, lambda: blink_cycle != 1, lambda: default_normal)
                p1_empty = jax.lax.cond(is_own, lambda: blink_cycle == 1, lambda: default_empty)
                return p1_normal, p1_empty

            # P2 (Schwarz/Enemy) Logic
            def _p2_logic():
                is_white = (cell_under_cursor == self.consts.PLAYER_1) 
                is_black = (cell_under_cursor == self.consts.PLAYER_2)
                
                # P2 auf Weiß: 1x Schwarz (Frame 0), 3x Weiß (Frame 1-3)
                case_white_normal = (blink_cycle == 0) 
                case_white_empty = False
                
                # P2 auf Schwarz (der gerade gesetzte Stein): 
                # Soll: 3x Schwarz, 1x Grün
                # Frames 0, 1, 2 -> Zeige Schwarz (normal)
                # Frame 3 -> Zeige Leer (empty/grün)
                case_black_normal = (blink_cycle != 3) 
                case_black_empty = (blink_cycle == 3)
                
                return jax.lax.cond(
                    is_white,
                    lambda: (case_white_normal, case_white_empty),
                    lambda: jax.lax.cond(is_black, lambda: (case_black_normal, case_black_empty), lambda: (default_normal, default_empty))
                )

            return jax.lax.cond(current_player == self.consts.PLAYER_1, _p1_logic, _p2_logic)

        is_visible_global = (state.step_counter >= self.consts.CURSOR_START_FRAME) & (~state.hide_cursor)

        should_draw_normal, should_draw_empty = jax.lax.cond(
            is_visible_global,
            _get_draw_flags,
            lambda: (False, False)
        )

        def _draw_normal_cursor(r):
            mask = jax.lax.select(
                state.current_player == self.consts.PLAYER_1,
                cursor_mask,
                piece_black_mask
            )
            return self.jr.render_at(r, cursor_pixel_x, cursor_pixel_y, mask)

        raster = jax.lax.cond(
            should_draw_normal,
            _draw_normal_cursor,
            lambda r: r,
            raster
        )
        
        raster = jax.lax.cond(
            should_draw_empty,
            lambda r: self.jr.render_at(r, cursor_pixel_x, cursor_pixel_y, empty_cursor_mask),
            lambda r: r,
            raster
        )
        
        # --- SCORE RENDERING ---
        DIGIT_WIDTH = 16 
        p1_digit_masks = jnp.stack([self.SHAPE_MASKS[f"score_{i}_player"] for i in range(10)])
        p2_digit_masks = jnp.stack([self.SHAPE_MASKS[f"score_{i}_enemy"] for i in range(10)])

        def _draw_score_for_player(r, score, base_x, base_y, masks):
            ones = score % 10
            r = self.jr.render_at(r, base_x, base_y, masks[ones])
            tens = (score // 10) % 10
            def _draw_tens(curr_r):
                return self.jr.render_at(curr_r, base_x - DIGIT_WIDTH, base_y, masks[tens])
            r = jax.lax.cond(score >= 10, _draw_tens, lambda curr_r: curr_r, r)
            return r

        raster = _draw_score_for_player(raster, state.player_1_score, self.consts.SCORE_P1_X, self.consts.SCORE_P1_Y, p1_digit_masks)
        raster = _draw_score_for_player(raster, state.player_2_score, self.consts.SCORE_P2_X, self.consts.SCORE_P2_Y, p2_digit_masks)

        img = self.jr.render_from_palette(raster, self.PALETTE)

        # --- GAME OVER FILTER ---
        board_full = (state.player_1_score + state.player_2_score) >= (self.consts.BOARD_SIZE * self.consts.BOARD_SIZE)
        no_moves_left = (state.passes >= 2)
        is_done = jnp.logical_or(board_full, no_moves_left)

        def _apply_filter(image):
             filter_color = jnp.array([196, 212, 210], dtype=jnp.uint16)
             return (image.astype(jnp.uint16) * filter_color // 255).astype(jnp.uint8)

        return jax.lax.cond(is_done, _apply_filter, lambda x: x, img)