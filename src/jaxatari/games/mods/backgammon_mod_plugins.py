"""
Backgammon Modification Plugins

This module provides mods using the JaxAtari plugin system:
- JaxAtariInternalModPlugin: For modifying constants/attributes at construction
- JaxAtariPostStepModPlugin: For modifying state after each step

Available Mods:
1. SimplifyBackgammonMod - reduced checker count (complex mod)
2. RewardShapingMod - intermediate rewards (complex mod)
3. NoHitsMod - no hitting variant (complex mod)
4. ShortGameMod - endgame starting position (complex mod)
5. ThemeMod - visual theme selection (simple mod)
6. ALEControlsMod - original ALE hold-to-scroll release-to-drop controls (complex mod)
"""

from functools import partial
from typing import Tuple, Any

import jax
import jax.numpy as jnp

from jaxatari.modification import JaxAtariPostStepModPlugin, JaxAtariInternalModPlugin
from jaxatari.games.jax_backgammon import BackgammonState


class BrownThemeMod(JaxAtariInternalModPlugin):
    """
    Changes the visual theme to 'brown' color scheme.
    Simple mod - overrides the THEME constant in BackgammonConstants.
    """
    constants_overrides = {
        "THEME": "brown"
    }


class BlueThemeMod(JaxAtariInternalModPlugin):
    """
    Changes the visual theme to 'blue' color scheme.
    Simple mod - overrides the THEME constant in BackgammonConstants.
    """
    constants_overrides = {
        "THEME": "blue"
    }


class ClassicThemeMod(JaxAtariInternalModPlugin):
    """
    Changes the visual theme to 'classic' color scheme (default).
    Simple mod - overrides the THEME constant in BackgammonConstants.
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
    
    Complex mod - modifies game state after each step.
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
        
        return new_state._replace(board=new_board)


class RewardShapingMod(JaxAtariPostStepModPlugin):
    """
    Adds intermediate rewards to Backgammon for better RL training.
    
    Reward components:
    1. Pip count improvement: Reward for moving checkers closer to home
    2. Bearing off bonus: Extra reward for bearing off checkers
    3. Hit bonus: Small reward for hitting opponent's blots
    4. Safety bonus: Reward for making points (2+ checkers)
    
    This helps RL agents learn faster by providing denser reward signals.
    
    Complex mod - modifies reward calculation after each step.
    
    Note: This mod modifies the reward returned by step(). Since plugins
    run after step(), we store the shaped reward in the state for the
    wrapper to pick up. Currently this is experimental.
    """
    
    # Reward shaping weights (can be overridden via constants_overrides)
    pip_weight: float = 0.01
    bear_off_bonus: float = 0.1
    hit_bonus: float = 0.05
    safety_weight: float = 0.02
    
    @partial(jax.jit, static_argnums=(0,))
    def _compute_pip_count(self, board: jnp.ndarray, player_idx: int) -> float:
        """Compute pip count for a player (lower is better)."""
        points_board = board[player_idx, :24]
        
        white_distances = 24 - jnp.arange(24)
        black_distances = jnp.arange(24) + 1
        
        distances = jax.lax.cond(
            player_idx == 0,
            lambda _: white_distances,
            lambda _: black_distances,
            operand=None
        )
        
        pip_count = jnp.sum(points_board * distances)
        bar_checkers = board[player_idx, 24]
        pip_count = pip_count + bar_checkers * 25
        
        return pip_count.astype(jnp.float32)
    
    @partial(jax.jit, static_argnums=(0,))
    def _count_safe_points(self, board: jnp.ndarray, player_idx: int) -> int:
        """Count number of points with 2+ checkers (safe points)."""
        points = board[player_idx, :24]
        return jnp.sum(points >= 2)
    
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: BackgammonState, new_state: BackgammonState) -> BackgammonState:
        """
        Compute shaped reward components based on state changes.
        
        Note: The actual reward modification happens in the wrapper layer.
        This plugin calculates the bonus and could store it in state if needed.
        For now, this is a reference implementation.
        """
        # Determine which player just moved
        player_idx = jax.lax.cond(
            prev_state.current_player == 1, 
            lambda _: 0, 
            lambda _: 1, 
            operand=None
        )
        opponent_idx = 1 - player_idx
        
        # 1. Pip count improvement
        prev_pip = self._compute_pip_count(prev_state.board, player_idx)
        curr_pip = self._compute_pip_count(new_state.board, player_idx)
        pip_improvement = prev_pip - curr_pip
        
        # 2. Bearing off bonus
        prev_borne = prev_state.board[player_idx, 25]
        curr_borne = new_state.board[player_idx, 25]
        new_borne = curr_borne - prev_borne
        
        # 3. Hit bonus
        prev_opp_bar = prev_state.board[opponent_idx, 24]
        curr_opp_bar = new_state.board[opponent_idx, 24]
        hits = curr_opp_bar - prev_opp_bar
        
        # 4. Safety bonus
        prev_safe = self._count_safe_points(prev_state.board, player_idx)
        curr_safe = self._count_safe_points(new_state.board, player_idx)
        new_safe = curr_safe - prev_safe
        
        # Calculate total bonus (stored for reference, actual reward mod in wrapper)
        # bonus = (self.pip_weight * pip_improvement + 
        #          self.bear_off_bonus * new_borne +
        #          self.hit_bonus * jnp.maximum(0, hits) +
        #          self.safety_weight * new_safe)
        
        # For now, return state unchanged - reward shaping needs wrapper integration
        return new_state


class ShortGameMod(JaxAtariPostStepModPlugin):
    """
    Creates very short backgammon games for rapid iteration.
    
    Starts with all checkers in the home board, ready to bear off.
    Games typically last only a few moves.
    
    Useful for testing bearing-off logic and end-game scenarios.
    
    Complex mod - modifies initial state on reset.
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
        new_state = state._replace(board=endgame_board)
        # Note: obs should be recomputed by environment if needed
        return obs, new_state


class SimplifyBackgammonMod(JaxAtariPostStepModPlugin):
    """
    Simplifies Backgammon for faster RL training.
    
    Modifications:
    - Fewer checkers per player (default: 5 instead of 15)
    - Modified starting positions for shorter games
    - Same rules, just faster episodes
    
    Complex mod - modifies initial state and win detection.
    
    Note: num_checkers can be configured via constants_overrides.
    Default is 5 checkers per player.
    """
    
    # Can be overridden via constants_overrides
    num_checkers: int = 5
    
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
        new_state = state._replace(board=simplified_board)
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
        
        return new_state._replace(is_game_over=game_over)


class ALEControlsMod(JaxAtariPostStepModPlugin):
    """
    Implements original ALE Backgammon controls: "hold-to-scroll, release-to-drop".
    
    In the original ALE version:
    - Hold LEFT/RIGHT to scroll through positions
    - Release the button to drop the checker at current position
    - No FIRE confirmation needed for placement
    
    This is harder to control but matches the original ALE behavior.
    
    Complex mod - provides backward compatibility with original ALE interface.
    
    Note: This is a complex mod that requires integration with step() logic.
    Implementation TODO: Needs to track button hold state and trigger drop on release.
    """
    
    # Mark as conflicting with the default cursor-based controls
    conflicts_with = ["DefaultControlsMod"]
    
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: BackgammonState, new_state: BackgammonState) -> BackgammonState:
        """
        Implement release-to-drop logic.
        
        If we were scrolling (LEFT/RIGHT held) and now action is NOOP,
        that means the button was released -> drop the checker.
        
        TODO: This requires tracking the previous action, which is not
        currently stored in state. Full implementation needs:
        1. Add 'last_action' to BackgammonState
        2. Detect NOOP after LEFT/RIGHT
        3. Auto-trigger drop (like FIRE in phase 2)
        """
        # Placeholder - full implementation requires state modification
        return new_state
