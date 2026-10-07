"""
Backgammon Modification Plugins

This module provides mods using the JaxAtari plugin system:
- JaxAtariInternalModPlugin: For modifying constants/attributes at construction
- JaxAtariPostStepModPlugin: For modifying state after each step

Available Mods:
1. Theme Mods (simple mods) - change visual theme:
    - BrownThemeMod
    - BlueThemeMod
    - ClassicThemeMod (default)
2. NoHitsMod (complex mod) - disables hitting for simpler learning
3. RewardShapingMod (complex mod) - adds intermediate rewards for better RL training
4. ShortGameMod (simple mod) - starts from endgame position for rapid iteration
5. SimplifyBackgammonMod (simple mod) - fewer checkers for faster episodes
6. ALEControlsMod (complex mod) - implements original ALE controls (hold-to-scroll, release-to-drop)
7. HighlightLegalMovesMod (simple mod) - highlights legal drop targets when picking a checker
8. SetupModeMod (simple mod) - starts from custom board setups for curriculum learning
"""

from functools import partial
from typing import Tuple, Any, Optional

import jax
import jax.numpy as jnp

from jaxatari.modification import JaxAtariPostStepModPlugin, JaxAtariInternalModPlugin
from jaxatari.games.jax_backgammon import BackgammonState, JaxBackgammonEnv, BackgammonRenderer
from jaxatari.environment import JAXAtariAction


@partial(jax.jit, static_argnums=())
def _compute_pip_count_for_player(board: jnp.ndarray, player_idx: int) -> jax.Array:
    """Compute pip count for player index (lower is better)."""
    points_board = board[player_idx, :24]

    white_distances = 24 - jnp.arange(24)
    black_distances = jnp.arange(24) + 1

    distances = jax.lax.cond(
        player_idx == 0,
        lambda _: white_distances,
        lambda _: black_distances,
        operand=None,
    )

    pip_count = jnp.sum(points_board * distances)
    bar_checkers = board[player_idx, 24]
    pip_count = pip_count + bar_checkers * 25
    return pip_count.astype(jnp.float32)


@partial(jax.jit, static_argnums=())
def _count_safe_points_for_player(board: jnp.ndarray, player_idx: int) -> jax.Array:
    points = board[player_idx, :24]
    return jnp.sum(points >= 2)


class BrownThemeMod(JaxAtariInternalModPlugin):
    """
    Changes the visual theme to 'brown' color scheme.
    """
    constants_overrides = {
        "THEME": "brown"
    }


class BlueThemeMod(JaxAtariInternalModPlugin):
    """
    Changes the visual theme to 'blue' color scheme.
    """
    constants_overrides = {
        "THEME": "blue"
    }


class ClassicThemeMod(JaxAtariInternalModPlugin):
    """
    Changes the visual theme to 'classic' color scheme (default).
    """
    constants_overrides = {
        "THEME": "classic"
    }


class NoHitsMod(JaxAtariPostStepModPlugin):
    """
    Disables hitting in Backgammon for simplified learning.
    
    When enabled, landing on an opponent's blot does NOT send them to the bar.
    The opponent's checker is restored to its original position after any hit.
    
    This creates a simpler game where pieces can coexist on points
    (but still can't land on blocked points with 2+ opponent checkers).
    
    Useful for initial RL training before introducing full complexity.
    """
    
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: BackgammonState, new_state: BackgammonState) -> BackgammonState:
        """
        Undo any hits that occurred during the step.
        Called after the main step is complete.
        """
        # Determine opponent based on who just moved
        opponent_idx = jax.lax.cond(
            prev_state.current_player == 1,
            lambda _: 1,  # White's opponent is Black (idx 1)
            lambda _: 0,  # Black's opponent is White (idx 0)
            operand=None
        )
        
        prev_opp_bar = prev_state.board[opponent_idx, 24]
        new_opp_bar = new_state.board[opponent_idx, 24]
        
        # Number of hits to undo
        hits_to_undo = new_opp_bar - prev_opp_bar
        
        # Find where the hit occurred (the destination of the last move)
        last_to = new_state.last_move[1]
        
        def undo_hits(args):
            board, hits, dest, opp_idx = args
            # Remove from bar
            new_bar = board[opp_idx, 24] - hits
            board = board.at[opp_idx, 24].set(new_bar)
            # Add back to destination (they were hit from there)
            board = board.at[opp_idx, dest].set(board[opp_idx, dest] + hits)
            return board
        
        def no_op(args):
            return args[0]
        
        new_board = jax.lax.cond(
            hits_to_undo > 0,
            undo_hits,
            no_op,
            operand=(new_state.board, hits_to_undo, last_to, opponent_idx)
        )
        
        return new_state.replace(board=new_board)


class RewardShapingMod(JaxAtariInternalModPlugin):
    """
    Adds intermediate rewards to Backgammon for better RL training.
    
    Reward components:
    1. Pip count improvement: Reward for moving checkers closer to home
    2. Bearing off bonus: Extra reward for bearing off checkers
    3. Hit bonus: Small reward for hitting opponent's blots
    4. Safety bonus: Reward for making points (2+ checkers)
    
    This helps RL agents learn faster by providing denser reward signals.
    """
    
    # Reward shaping weights (can be overridden via constants_overrides)
    pip_weight: float = 0.01
    bear_off_bonus: float = 0.1
    hit_bonus: float = 0.05
    safety_weight: float = 0.02
    
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        """Patch env.step to return reward + shaping bonus."""
        obs, new_state, reward, done, info = JaxBackgammonEnv.step(self._env, state, action)

        # Determine which player just moved (current player in prev_state)
        player_idx = jax.lax.cond(
            state.current_player == 1,
            lambda _: 0,
            lambda _: 1,
            operand=None,
        )
        opponent_idx = 1 - player_idx

        prev_pip = _compute_pip_count_for_player(state.board, player_idx)
        curr_pip = _compute_pip_count_for_player(new_state.board, player_idx)
        pip_improvement = prev_pip - curr_pip

        prev_borne = state.board[player_idx, 25]
        curr_borne = new_state.board[player_idx, 25]
        new_borne = curr_borne - prev_borne

        prev_opp_bar = state.board[opponent_idx, 24]
        curr_opp_bar = new_state.board[opponent_idx, 24]
        hits = curr_opp_bar - prev_opp_bar

        prev_safe = _count_safe_points_for_player(state.board, player_idx)
        curr_safe = _count_safe_points_for_player(new_state.board, player_idx)
        new_safe = curr_safe - prev_safe

        # Did the agent successfully pick up a checker? (Phase 1 -> Phase 2)
        picked_up = (state.game_phase == 1) & (new_state.game_phase == 2)
        pick_bonus = jnp.where(picked_up, 0.01, 0.0)

        # Did the agent successfully drop a checker on a valid point? 
        # (Phase 2 -> Phase 1 or Phase 0)
        valid_drop = (state.game_phase == 2) & ((new_state.game_phase == 1) | (new_state.game_phase == 0))
        drop_bonus = jnp.where(valid_drop, 0.05, 0.0)

        # Calculate final bonus including UI interactions
        bonus = (
            self.pip_weight * pip_improvement +
            self.bear_off_bonus * new_borne +
            self.hit_bonus * jnp.maximum(0, hits) +
            self.safety_weight * new_safe +
            pick_bonus + 
            drop_bonus
        ).astype(jnp.float32)

        shaped_reward = jnp.asarray(reward, dtype=jnp.float32) + bonus
        return obs, new_state, shaped_reward, done, info


class ShortGameMod(JaxAtariPostStepModPlugin):
    """
    Creates very short backgammon games for rapid iteration.
    
    Starts with all checkers in the home board, ready to bear off.
    Games typically last only a few moves.
    
    Useful for testing bearing-off logic and end-game scenarios.
    """
    
    @partial(jax.jit, static_argnums=(0,))
    def _create_endgame_board(self) -> jnp.ndarray:
        """Create a board with all checkers in home, ready to bear off."""
        board = jnp.zeros((2, 26), dtype=jnp.int32)
        
        # White (player 0): All 15 checkers distributed in home (points 18-23)
        board = board.at[0, 18].set(3)
        board = board.at[0, 19].set(3)
        board = board.at[0, 20].set(3)
        board = board.at[0, 21].set(2)
        board = board.at[0, 22].set(2)
        board = board.at[0, 23].set(2)
        
        # Black (player 1): All 15 checkers in home (points 0-5)
        board = board.at[1, 5].set(3)
        board = board.at[1, 4].set(3)
        board = board.at[1, 3].set(3)
        board = board.at[1, 2].set(2)
        board = board.at[1, 1].set(2)
        board = board.at[1, 0].set(2)
        
        return board
    
    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state: BackgammonState) -> Tuple[Any, BackgammonState]:
        """Replace initial board with endgame position."""
        endgame_board = self._create_endgame_board()
        new_state = state.replace(board=endgame_board)
        # Note: obs should be recomputed by environment if needed
        return obs, new_state


class SimplifyBackgammonMod(JaxAtariPostStepModPlugin):
    """
    Simplifies Backgammon for faster RL training.
    
    Modifications:
    - Fewer checkers per player (default: 5 instead of 15)
    - Modified starting positions for shorter games
    - Same rules, just faster episodes
        
    Note: num_checkers can be configured via constants_overrides.
    Default is 5 checkers per player.
    """
    
    # Can be overridden via constants_overrides
    num_checkers: int = 5
    
    constants_overrides = {
        "NUM_CHECKERS": 5
    }
    
    @partial(jax.jit, static_argnums=(0,))
    def _create_simplified_board(self) -> jnp.ndarray:
        """Create a simplified starting board with fewer checkers."""
        board = jnp.zeros((2, 26), dtype=jnp.int32)
        
        # Distribution: split between far point and mid point
        far_count = self.num_checkers // 2
        near_count = self.num_checkers - far_count
        
        # White (player 0): checkers on point 0 and point 11
        board = board.at[0, 0].set(far_count)
        board = board.at[0, 11].set(near_count)
        
        # Black (player 1): checkers on point 23 and point 12 (mirror)
        board = board.at[1, 23].set(far_count)
        board = board.at[1, 12].set(near_count)
        
        return board
    
    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state: BackgammonState) -> Tuple[Any, BackgammonState]:
        """Replace initial board with simplified position."""
        simplified_board = self._create_simplified_board()
        new_state = state.replace(board=simplified_board)
        return obs, new_state
    
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: BackgammonState, new_state: BackgammonState) -> BackgammonState:
        """
        Override win detection for fewer checkers.
        Game ends when all num_checkers have been borne off.
        """
        white_off = new_state.board[0, 25]
        black_off = new_state.board[1, 25]
        
        game_over = (white_off >= self.num_checkers) | (black_off >= self.num_checkers)
        
        return new_state.replace(is_game_over=game_over)


class ALEControlsMod(JaxAtariInternalModPlugin):
    """
    Implements original ALE Backgammon controls: "hold-to-scroll, release-to-drop".
    
    In the original ALE version:
    - Hold LEFT/RIGHT to scroll through positions
    - Release the button to drop the checker at current position
    - No FIRE confirmation needed for placement
    
    This is harder to control but matches the original ALE behavior.
    """
    
    # Mark as conflicting with the default cursor-based controls
    conflicts_with = ["DefaultControlsMod"]
    
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        """Patch env.step to auto-drop on button release (NOOP) during moving phase."""
        obs, new_state, reward, done, info = JaxBackgammonEnv.step(self._env, state, action)

        was_scrolling = (
            (state.last_action == JAXAtariAction.LEFT) |
            (state.last_action == JAXAtariAction.RIGHT)
        )
        is_release = (action == JAXAtariAction.NOOP) & was_scrolling

        # During phase 2: releasing the joystick drops the checker
        can_drop = is_release & (state.game_phase == jnp.int32(2)) & (new_state.picked_checker_from >= 0)

        def do_drop(_):
            dropped_state, drop_reward, drop_done = self._env._handle_drop_checker(new_state)
            drop_obs = self._env._get_observation(dropped_state)
            drop_info = self._env._get_info(dropped_state)
            return drop_obs, dropped_state, drop_reward, drop_done, drop_info

        def skip_drop(_):
            return obs, new_state, reward, done, info

        obs_after_drop, state_after_drop, reward_after_drop, done_after_drop, info_after_drop = jax.lax.cond(
            can_drop, do_drop, skip_drop, operand=None
        )

        # During phase 1: releasing the joystick picks up the checker
        # Using original state.game_phase ensures we don't pick up in the exact frame we just dropped
        can_pick = is_release & (state.game_phase == jnp.int32(1))

        def do_pick(_):
            picked_state = self._env._handle_pick_checker(state_after_drop)
            pick_obs = self._env._get_observation(picked_state)
            pick_info = self._env._get_info(picked_state)
            return pick_obs, picked_state, reward_after_drop, done_after_drop, pick_info

        def skip_pick(_):
            return obs_after_drop, state_after_drop, reward_after_drop, done_after_drop, info_after_drop

        final_obs, final_state, final_reward, final_done, final_info = jax.lax.cond(
            can_pick, do_pick, skip_pick, operand=None
        )

        # Clear last_action so that holding a NOOP doesn't trigger releases infinitely
        final_state = jax.lax.cond(
            is_release,
            lambda s: s.replace(last_action=JAXAtariAction.NOOP),
            lambda s: s,
            operand=final_state
        )

        return final_obs, final_state, final_reward, final_done, final_info


class HighlightLegalMovesMod(JaxAtariInternalModPlugin):
    """Highlights all legal drop targets while a checker is picked (game_phase == 2)."""

    @partial(jax.jit, static_argnums=(0,))
    def _draw_highlight(self, raster, state):
        renderer = self._env.renderer

        # Keep default single-highlight behavior first.
        raster = BackgammonRenderer._draw_highlight(renderer, raster, state)

        should_draw = (state.game_phase == jnp.int32(2)) & (state.picked_checker_from >= 0)

        def draw_legal_targets(r):
            from_point = jnp.int32(state.picked_checker_from)
            to_points = jnp.arange(renderer.consts.HOME_INDEX + 1, dtype=jnp.int32)  # 0..25

            valid = jax.vmap(
                lambda tp: self._env.is_valid_move(state, jnp.array([from_point, tp], dtype=jnp.int32))
            )(to_points)

            def draw_one(i, rr):
                tp = to_points[i]
                ok = valid[i]

                def draw_valid(r2):
                    def draw_triangle(r3):
                        pos = renderer.triangle_positions[tp]
                        is_left = (pos[0] == renderer.board_margin)
                        mask = jax.lax.select(
                            is_left,
                            renderer.SHAPE_MASKS["triangle_highlight_right"],
                            renderer.SHAPE_MASKS["triangle_highlight_left"],
                        )
                        return renderer.jr.render_at(r3, pos[0], pos[1], mask)

                    def draw_bar_left(r3):
                        return renderer.jr.render_at(
                            r3,
                            renderer.bar_x,
                            renderer.bar_y,
                            renderer.SHAPE_MASKS["bar_highlight_left"],
                        )

                    return jax.lax.cond(
                        tp < jnp.int32(24),
                        draw_triangle,
                        lambda r3: jax.lax.cond(tp == jnp.int32(24), draw_bar_left, lambda x: x, operand=r3),
                        operand=r2,
                    )

                return jax.lax.cond(ok, draw_valid, lambda x: x, operand=rr)

            return jax.lax.fori_loop(0, renderer.consts.HOME_INDEX + 1, draw_one, r)

        return jax.lax.cond(should_draw, draw_legal_targets, lambda x: x, operand=raster)


class SetupModeMod(JaxAtariPostStepModPlugin):
    """Start from a custom board setup for curriculum learning."""

    # Optional user-defined board. If None, a default curriculum setup is used.
    setup_board: Optional[jnp.ndarray] = None

    @partial(jax.jit, static_argnums=(0,))
    def _default_setup_board(self) -> jnp.ndarray:
        board = jnp.zeros((2, 26), dtype=jnp.int32)
        # White checkpoints toward bear-off.
        board = board.at[0, 18].set(5)
        board = board.at[0, 20].set(5)
        board = board.at[0, 22].set(5)
        # Black mirrored setup.
        board = board.at[1, 5].set(5)
        board = board.at[1, 3].set(5)
        board = board.at[1, 1].set(5)
        return board

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state: BackgammonState) -> Tuple[Any, BackgammonState]:
        board = self._default_setup_board() if self.setup_board is None else jnp.asarray(self.setup_board, dtype=jnp.int32)
        new_state = state.replace(
            board=board,
            game_phase=jnp.int32(1),
            picked_checker_from=jnp.int32(-1),
            last_valid_drop=jnp.int32(-1),
        )
        new_obs = self._env._get_observation(new_state)
        return new_obs, new_state

