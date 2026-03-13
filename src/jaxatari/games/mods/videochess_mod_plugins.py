from functools import partial

import jax
import jax.numpy as jnp

from jaxatari.modification import JaxAtariPostStepModPlugin

class RandomBotBlackMod(JaxAtariPostStepModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        env = self._env
        for _ in range(6):
            if hasattr(env, "_env"):
                env = env._env
            else:
                break

        return env.random_black_reply(prev_state, new_state)


class GreedyBotBlackMod(JaxAtariPostStepModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        env = self._env
        for _ in range(6):
            if hasattr(env, "_env"):
                env = env._env
            else:
                break

        return env.greedy_black_reply(prev_state, new_state)


class MinimaxBotBlackMod(JaxAtariPostStepModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        env = self._env
        for _ in range(6):
            if hasattr(env, "_env"):
                env = env._env
            else:
                break

        return env.minimax_black_reply(prev_state, new_state)
    

def _unwrap_to_base_env(env, max_depth: int = 8):
    # unwrap controller/wrapper chain until we reach the actual game env
    for _ in range(max_depth):
        if hasattr(env, "_env"):
            env = env._env
        else:
            break
    return env


def _empty_board(c):
    return jnp.zeros((c.NUM_RANKS, c.NUM_FILES), dtype=jnp.int32)


def _place_kings(board, c):
    # Keep standard chess king squares: black top, white bottom
    board = board.at[0, 4].set(jnp.int32(c.B_KING))
    board = board.at[7, 4].set(jnp.int32(c.W_KING))
    return board


def _reset_state_with_board(state, board, c):
    # Keep cursor as-is, but clear selection / blink state
    return state._replace(
        board=board,
        to_move=jnp.int32(c.COLOUR_WHITE),
        game_phase=jnp.int32(c.PHASE_SELECT_PIECE),
        selected_square=jnp.array([-1, -1], dtype=jnp.int32),
        last_move_target=jnp.array([-1, -1], dtype=jnp.int32),
        last_move_timer=jnp.int32(0),
    )


class PawnsOnlyMod(JaxAtariPostStepModPlugin):
    """All non-king pieces are pawns."""

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        env = _unwrap_to_base_env(self._env)
        c = env.consts

        board = _empty_board(c)

        # Fill the same squares that would normally be occupied at game start:
        # Black pieces on ranks 0-1, white pieces on ranks 6-7.
        board = board.at[0, :].set(jnp.int32(c.B_PAWN))
        board = board.at[1, :].set(jnp.int32(c.B_PAWN))
        board = board.at[6, :].set(jnp.int32(c.W_PAWN))
        board = board.at[7, :].set(jnp.int32(c.W_PAWN))

        # Keep exactly one king per side on the standard squares.
        board = _place_kings(board, c)

        state = _reset_state_with_board(state, board, c)
        obs = obs._replace(board=state.board) if hasattr(obs, "_replace") else obs
        return obs, state


class QueensOnlyMod(JaxAtariPostStepModPlugin):
    """All non-king pieces are queens."""

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        env = _unwrap_to_base_env(self._env)
        c = env.consts

        board = _empty_board(c)

        # Fill the same squares that would normally be occupied at game start:
        # Black pieces on ranks 0-1, white pieces on ranks 6-7.
        board = board.at[0, :].set(jnp.int32(c.B_QUEEN))
        board = board.at[1, :].set(jnp.int32(c.B_QUEEN))
        board = board.at[6, :].set(jnp.int32(c.W_QUEEN))
        board = board.at[7, :].set(jnp.int32(c.W_QUEEN))

        # Keep exactly one king per side on the standard squares.
        board = _place_kings(board, c)

        state = _reset_state_with_board(state, board, c)
        obs = obs._replace(board=state.board) if hasattr(obs, "_replace") else obs
        return obs, state


class KingsOnlyMod(JaxAtariPostStepModPlugin):
    """Only kings."""
    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        env = _unwrap_to_base_env(self._env)
        c = env.consts

        board = _empty_board(c)

        # Same occupied squares as normal chess start, but everything is a king.
        board = board.at[0, :].set(jnp.int32(c.B_KING))
        board = board.at[1, :].set(jnp.int32(c.B_KING))
        board = board.at[6, :].set(jnp.int32(c.W_KING))
        board = board.at[7, :].set(jnp.int32(c.W_KING))

        state = _reset_state_with_board(state, board, c)
        obs = obs._replace(board=state.board) if hasattr(obs, "_replace") else obs
        return obs, state
