from functools import partial

import jax
import jax.numpy as jnp

from jaxtari.modification import JaxtariInternalModPlugin, JaxtariPostStepModPlugin
from jaxtari.games.jax_videochess import BoardHandler


def _unwrap_to_base_env(env, max_depth: int = 8):
    for _ in range(max_depth):
        if hasattr(env, "_env"):
            env = env._env
        else:
            break
    return env


def _empty_board(c):
    return jnp.zeros((c.NUM_RANKS, c.NUM_FILES), dtype=jnp.int32)


def _place_kings(board, c):
    board = board.at[0, 4].set(jnp.int32(c.B_KING))
    board = board.at[7, 4].set(jnp.int32(c.W_KING))
    return board


def _reset_state_with_board(state, board, c):
    return state.replace(
        board=board,
        to_move=jnp.int32(c.COLOUR_WHITE),
        game_phase=jnp.int32(c.PHASE_SELECT_PIECE),
        selected_square=jnp.array([-1, -1], dtype=jnp.int32),
        last_move_target=jnp.array([-1, -1], dtype=jnp.int32),
        last_move_timer=jnp.int32(0),
        highlight_squares=jnp.full((c.MAX_MOVES_PER_PIECE, 2), -1, dtype=jnp.int32),
    )


# ---------------------------------------------------------------------------
# Opponent / control mods (constants → BLACK_BOT_KIND)
# Default env: BLACK_BOT_KIND=3 (minimax). play_both_sides turns the bot off.
# ---------------------------------------------------------------------------

class PlayBothSidesMod(JaxtariInternalModPlugin):
    """Human controls both white and black (no automatic black reply)."""

    constants_overrides = {"BLACK_BOT_KIND": 0}


class RandomBotBlackMod(JaxtariInternalModPlugin):
    """Black replies with a random legal move after white moves."""

    constants_overrides = {"BLACK_BOT_KIND": 1}


class GreedyBotBlackMod(JaxtariInternalModPlugin):
    """Black replies with a greedy material-capture move after white moves."""

    constants_overrides = {"BLACK_BOT_KIND": 2}


class MinimaxBotBlackMod(JaxtariInternalModPlugin):
    """Black replies with depth-2 minimax (this is the unmodded default)."""

    constants_overrides = {"BLACK_BOT_KIND": 3}


class InstantMovementMod(JaxtariInternalModPlugin):
    """Held direction inputs repeat every frame (FIRE stays edge-triggered)."""

    constants_overrides = {"INPUT_EDGE_TRIGGERED": False}


class LegalMovesDisplayMod(JaxtariPostStepModPlugin):
    """Show legal-move indicator dots for the currently selected piece."""

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        env = _unwrap_to_base_env(self._env)
        c = env.consts

        in_target = new_state.game_phase == c.PHASE_SELECT_TARGET
        sel = new_state.selected_square
        has_sel = (sel[0] >= 0) & (sel[1] >= 0)

        highlights = jax.lax.cond(
            in_target & has_sel,
            lambda: BoardHandler.legal_moves_for_colour(
                new_state.board, sel, new_state.to_move,
                new_state.en_passant_sq, new_state.castling_rights,
            ),
            lambda: jnp.full((c.MAX_MOVES_PER_PIECE, 2), -1, dtype=jnp.int32),
        )
        return new_state.replace(highlight_squares=highlights)


# ---------------------------------------------------------------------------
# Board setup mods
# ---------------------------------------------------------------------------

class PawnsOnlyMod(JaxtariPostStepModPlugin):
    """All non-king pieces are pawns."""

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        env = _unwrap_to_base_env(self._env)
        c = env.consts
        board = _empty_board(c)
        board = board.at[0, :].set(jnp.int32(c.B_PAWN))
        board = board.at[1, :].set(jnp.int32(c.B_PAWN))
        board = board.at[6, :].set(jnp.int32(c.W_PAWN))
        board = board.at[7, :].set(jnp.int32(c.W_PAWN))
        board = _place_kings(board, c)
        state = _reset_state_with_board(state, board, c)
        obs = obs.replace(board=state.board) if hasattr(obs, "replace") else obs
        return obs, state


class QueensOnlyMod(JaxtariPostStepModPlugin):
    """All non-king pieces are queens."""

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        env = _unwrap_to_base_env(self._env)
        c = env.consts
        board = _empty_board(c)
        board = board.at[0, :].set(jnp.int32(c.B_QUEEN))
        board = board.at[1, :].set(jnp.int32(c.B_QUEEN))
        board = board.at[6, :].set(jnp.int32(c.W_QUEEN))
        board = board.at[7, :].set(jnp.int32(c.W_QUEEN))
        board = _place_kings(board, c)
        state = _reset_state_with_board(state, board, c)
        obs = obs.replace(board=state.board) if hasattr(obs, "replace") else obs
        return obs, state


class RooksOnlyMod(JaxtariPostStepModPlugin):
    """All non-king pieces are rooks."""

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        env = _unwrap_to_base_env(self._env)
        c = env.consts
        board = _empty_board(c)
        board = board.at[0, :].set(jnp.int32(c.B_ROOK))
        board = board.at[1, :].set(jnp.int32(c.B_ROOK))
        board = board.at[6, :].set(jnp.int32(c.W_ROOK))
        board = board.at[7, :].set(jnp.int32(c.W_ROOK))
        board = _place_kings(board, c)
        state = _reset_state_with_board(state, board, c)
        obs = obs.replace(board=state.board) if hasattr(obs, "replace") else obs
        return obs, state


class KnightsOnlyMod(JaxtariPostStepModPlugin):
    """All non-king pieces are knights."""

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        env = _unwrap_to_base_env(self._env)
        c = env.consts
        board = _empty_board(c)
        board = board.at[0, :].set(jnp.int32(c.B_KNIGHT))
        board = board.at[1, :].set(jnp.int32(c.B_KNIGHT))
        board = board.at[6, :].set(jnp.int32(c.W_KNIGHT))
        board = board.at[7, :].set(jnp.int32(c.W_KNIGHT))
        board = _place_kings(board, c)
        state = _reset_state_with_board(state, board, c)
        obs = obs.replace(board=state.board) if hasattr(obs, "replace") else obs
        return obs, state


class BishopsOnlyMod(JaxtariPostStepModPlugin):
    """All non-king pieces are bishops."""

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        env = _unwrap_to_base_env(self._env)
        c = env.consts
        board = _empty_board(c)
        board = board.at[0, :].set(jnp.int32(c.B_BISHOP))
        board = board.at[1, :].set(jnp.int32(c.B_BISHOP))
        board = board.at[6, :].set(jnp.int32(c.W_BISHOP))
        board = board.at[7, :].set(jnp.int32(c.W_BISHOP))
        board = _place_kings(board, c)
        state = _reset_state_with_board(state, board, c)
        obs = obs.replace(board=state.board) if hasattr(obs, "replace") else obs
        return obs, state


class CheckmateTestMod(JaxtariPostStepModPlugin):
    """Sparse board for testing checkmate: white king + 2 queens vs. lone black king."""

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        env = _unwrap_to_base_env(self._env)
        c = env.consts
        board = _empty_board(c)
        board = board.at[0, 0].set(jnp.int32(c.B_KING))
        board = board.at[7, 7].set(jnp.int32(c.W_KING))
        board = board.at[7, 5].set(jnp.int32(c.W_QUEEN))
        board = board.at[6, 6].set(jnp.int32(c.W_QUEEN))
        state = _reset_state_with_board(state, board, c)
        obs = obs.replace(board=state.board) if hasattr(obs, "replace") else obs
        return obs, state
