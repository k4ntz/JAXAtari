import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple
from jaxatari.games.jax_othello import OthelloState
from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin

class MinMaxAIMod(JaxAtariInternalModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def _ai_turn(self, state):
        AI_PLAYER = self._env.consts.PLAYER_2
        OPPONENT = self._env.consts.PLAYER_1
        BOARD_DIM = self._env.consts.BOARD_SIZE
        COUNT_FIELDS = BOARD_DIM * BOARD_DIM

        def _simulate_move(board, y, x, player):
            mask = self._env._get_flip_mask(board, y, x, player)
            nb = board.at[y, x].set(player)
            nb = jnp.where(mask, player, nb)
            return nb

        def _evaluate_board(board, last_y, last_x):
            pos_score = self._env._get_dynamic_position_score(board, last_y, last_x)
            p1_count, p2_count = self._env._count_pieces(board)
            material_score = p2_count - p1_count
            return (material_score + pos_score).astype(jnp.float32)

        def _score_root_move(idx):
            y, x = idx // BOARD_DIM, idx % BOARD_DIM
            is_valid = self._env._is_valid_move(state.board, y, x, AI_PLAYER)

            def _get_score():
                my_board = _simulate_move(state.board, y, x, AI_PLAYER)
                return _evaluate_board(my_board, y, x)

            return jax.lax.cond(is_valid, _get_score, lambda: -1e9)

        all_scores = jax.vmap(_score_root_move)(jnp.arange(COUNT_FIELDS))

        rng_key = jax.random.PRNGKey(state.lfsr_state)
        noise = jax.random.uniform(rng_key, shape=(COUNT_FIELDS,), minval=0.0, maxval=0.1)
        best_idx = jnp.argmax(all_scores + noise)
        best_score = jnp.max(all_scores)
        
        next_lfsr = self._env._atari_lfsr_step(state.lfsr_state)

        def _execute_move():
            y, x = best_idx // BOARD_DIM, best_idx % BOARD_DIM
            flip_mask = self._env._get_flip_mask(state.board, y, x, AI_PLAYER)
            
            return state.replace(
                cursor_y=y.astype(jnp.int32),
                cursor_x=x.astype(jnp.int32),
                phase=jnp.array(self._env.consts.PHASE_ANIMATION, dtype=jnp.int32),
                animation_sub_phase=jnp.array(self._env.consts.SUBPHASE_INITIAL_PLACE, dtype=jnp.int32),
                pieces_to_flip=flip_mask,
                target_player=jnp.array(AI_PLAYER, dtype=jnp.int32),
                target_x=x.astype(jnp.int32),
                target_y=y.astype(jnp.int32),
                animation_timer=jnp.array(self._env.consts.FRAMES_TO_PLACE, dtype=jnp.int32),
                passes=jnp.array(0, dtype=jnp.int32),
                lfsr_state=next_lfsr
            )

        def _pass_turn():
            return state.replace(
                current_player=jnp.array(OPPONENT, dtype=jnp.int32),
                passes=state.passes + 1,
                lfsr_state=next_lfsr,
                turn_start_frame=state.step_counter + 1
            )

        return jax.lax.cond(best_score > -0.9e9, _execute_move, _pass_turn)

class BombMod(JaxAtariInternalModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def _attempt_player_move(self, state: OthelloState) -> Tuple[OthelloState, bool]:
        y, x = state.cursor_y, state.cursor_x
        player = state.current_player
        is_valid_place = self._env._is_valid_move(state.board, y, x, player)
        is_piece = state.board[y, x] != self._env.consts.EMPTY

        def _initiate_place():
            flip_mask = self._env._get_flip_mask(state.board, y, x, player)
            
            new_state = state.replace(
                phase=jnp.array(self._env.consts.PHASE_ANIMATION, dtype=jnp.int32),
                animation_sub_phase=jnp.array(self._env.consts.SUBPHASE_INITIAL_PLACE, dtype=jnp.int32),
                pieces_to_flip=flip_mask,
                target_player=player,
                target_x=x,
                target_y=y,
                animation_timer=jnp.array(self._env.consts.FRAMES_TO_PLACE, dtype=jnp.int32),
                passes=jnp.array(0, dtype=jnp.int32)
            )
            return new_state.replace(hide_cursor=jnp.array(False, dtype=jnp.bool_)), True

        def _initiate_destroy():
            new_board = state.board.at[y, x].set(self._env.consts.EMPTY)
            p1_count = jnp.sum(new_board == self._env.consts.PLAYER_1)
            p2_count = jnp.sum(new_board == self._env.consts.PLAYER_2)
            next_player = jnp.where(player == self._env.consts.PLAYER_1, self._env.consts.PLAYER_2, self._env.consts.PLAYER_1)
            
            new_state = state.replace(
                board=new_board,
                player_1_score=p1_count,
                player_2_score=p2_count,
                current_player=next_player,
                passes=jnp.array(0, dtype=jnp.int32),
                turn_start_frame=state.step_counter + 1,
                hide_cursor=jnp.array(False, dtype=jnp.bool_)
            )
            return new_state, True

        def _fail_move():
            return state.replace(hide_cursor=jnp.array(True, dtype=jnp.bool_)), False

        return jax.lax.cond(
            is_valid_place,
            _initiate_place,
            lambda: jax.lax.cond(is_piece, _initiate_destroy, _fail_move)
        )


_large_start_board = jnp.zeros((10, 10), dtype=jnp.int32)
_large_start_board = _large_start_board.at[4, 4].set(2)
_large_start_board = _large_start_board.at[4, 5].set(1)
_large_start_board = _large_start_board.at[5, 4].set(1)
_large_start_board = _large_start_board.at[5, 5].set(2)

_large_weights = jnp.ones((10, 10), dtype=jnp.int32) * 5
_large_weights = _large_weights.at[0, 0].set(64)
_large_weights = _large_weights.at[0, 9].set(64)
_large_weights = _large_weights.at[9, 0].set(64)
_large_weights = _large_weights.at[9, 9].set(64)


class LargeBoardMod(JaxAtariInternalModPlugin):
    constants_overrides = {
        "WIDTH": 192,
        "HEIGHT": 254,
        "BOARD_SIZE": 10,
        "START_BOARD": _large_start_board,
        "ENEMY_WEIGHTS": _large_weights,
        "BOARD_TOP_LEFT_X": 18,
        "BOARD_TOP_LEFT_Y": 22, 
    }
    asset_overrides = {
        "background": {"name": "background", "type": "background", "file": "big_background.npy"},
    }


_small_start_board = jnp.zeros((6, 6), dtype=jnp.int32)
_small_start_board = _small_start_board.at[2, 2].set(2)
_small_start_board = _small_start_board.at[2, 3].set(1)
_small_start_board = _small_start_board.at[3, 2].set(1)
_small_start_board = _small_start_board.at[3, 3].set(2)

_small_weights = jnp.ones((6, 6), dtype=jnp.int32) * 5
_small_weights = _small_weights.at[0, 0].set(64)
_small_weights = _small_weights.at[0, 5].set(64)
_small_weights = _small_weights.at[5, 0].set(64)
_small_weights = _small_weights.at[5, 5].set(64)

class SmallBoardMod(JaxAtariInternalModPlugin):
    constants_overrides = {
        "BOARD_SIZE": 6,
        "START_BOARD": _small_start_board,
        "ENEMY_WEIGHTS": _small_weights,
        "BOARD_TOP_LEFT_X": 34,
        "BOARD_TOP_LEFT_Y": 44,
    }
    asset_overrides = {
        "background": {"name": "background", "type": "background", "file": "small_background.npy"},
    }
