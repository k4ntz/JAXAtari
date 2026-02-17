"""
Backgammon Modification Wrappers

This module provides wrappers to modify the Backgammon environment for:
1. SimplifyBackgammonWrapper - reduced checker count
2. RewardShapingWrapper - intermediate rewards
3. NoHitsBackgammonWrapper - no hitting variant
4. ShortGameWrapper - endgame starting position

Usage:
    from jaxatari.games.mods.backgammon_mods import SimplifyBackgammonWrapper
    
    env = JaxBackgammonEnv()
    env = SimplifyBackgammonWrapper(env, num_checkers=5)
"""

import functools
from typing import Any, Dict, Tuple, Union

import chex
import jax
import jax.numpy as jnp
from jaxatari.wrappers import JaxatariWrapper
from jaxatari.games.jax_backgammon import BackgammonState, BackgammonConstants


class SimplifyBackgammonWrapper(JaxatariWrapper):
    """
    Simplifies Backgammon for faster RL training.
    
    Modifications:
    - Fewer checkers per player (default: 5 instead of 15)
    - Modified starting positions for shorter games
    - Same rules, just faster episodes
    
    This demonstrates the "modifiability" criterion from Design Guide Section 6.
    
    Args:
        env: The BackgammonEnv to wrap
        num_checkers: Number of checkers per player (default: 5)
    """
    
    def __init__(self, env, num_checkers: int = 5):
        super().__init__(env)
        self.num_checkers = num_checkers
        
        # Validate
        if num_checkers < 2 or num_checkers > 15:
            raise ValueError(f"num_checkers must be between 2 and 15, got {num_checkers}")
    
    def _create_simplified_board(self) -> jnp.ndarray:
        """
        Create a simplified starting board with fewer checkers.
        
        Distribution for 5 checkers:
        - White: 2 on point 0, 3 on point 11 (in opponent's territory)
        - Black: 2 on point 23, 3 on point 12 (mirror of white)
        """
        board = jnp.zeros((2, 26), dtype=jnp.int32)
        
        if self.num_checkers == 5:
            # White (player 0): 2 on point 0, 3 on point 11
            board = board.at[0, 0].set(2)
            board = board.at[0, 11].set(3)
            # Black (player 1): 2 on point 23, 3 on point 12
            board = board.at[1, 23].set(2)
            board = board.at[1, 12].set(3)
        elif self.num_checkers == 3:
            # Ultra-simple: 1 on far point, 2 on mid point
            board = board.at[0, 0].set(1)
            board = board.at[0, 11].set(2)
            board = board.at[1, 23].set(1)
            board = board.at[1, 12].set(2)
        else:
            # Generic distribution: split between two positions
            far_count = self.num_checkers // 2
            near_count = self.num_checkers - far_count
            board = board.at[0, 0].set(far_count)
            board = board.at[0, 11].set(near_count)
            board = board.at[1, 23].set(far_count)
            board = board.at[1, 12].set(near_count)
        
        return board
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Any, BackgammonState]:
        """Reset with simplified board."""
        # Get the base reset
        obs, state = self._env.reset(key)
        
        # Replace board with simplified version
        simplified_board = self._create_simplified_board()
        new_state = state._replace(board=simplified_board)
        
        # Update observation
        new_obs = self._env._get_observation(new_state)
        
        return new_obs, new_state
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state: BackgammonState, action: Union[int, float]) -> Tuple[Any, BackgammonState, float, bool, Dict]:
        """Step with adjusted win detection for fewer checkers."""
        obs, new_state, reward, done, info = self._env.step(state, action)
        
        # Override win detection: check if all num_checkers have been borne off
        # HOME_INDEX is 25 (where borne-off checkers go)
        white_off = new_state.board[0, 25]
        black_off = new_state.board[1, 25]
        
        white_won = white_off >= self.num_checkers
        black_won = black_off >= self.num_checkers
        game_over = white_won | black_won
        
        # Calculate reward based on winner
        # White = player 1 (+1), Black = player -1 (-1 for white perspective)
        new_reward = jax.lax.cond(
            game_over,
            lambda _: jnp.where(white_won, 1.0, -1.0),
            lambda _: reward,
            operand=None
        )
        
        # Update state with correct game_over flag
        final_state = new_state._replace(is_game_over=game_over)
        
        # Update done flag
        new_done = game_over
        
        return obs, final_state, new_reward, new_done, info


class RewardShapingWrapper(JaxatariWrapper):
    """
    Adds intermediate rewards to Backgammon for better RL training.
    
    Reward components:
    1. Pip count improvement: Reward for moving checkers closer to home
    2. Bearing off bonus: Extra reward for bearing off checkers
    3. Hit bonus: Small reward for hitting opponent's blots
    4. Safety bonus: Reward for making points (2+ checkers)
    
    This helps RL agents learn faster by providing denser reward signals.
    
    Args:
        env: The BackgammonEnv to wrap
        pip_weight: Weight for pip count improvement (default: 0.01)
        bear_off_bonus: Bonus for each checker borne off (default: 0.1)
        hit_bonus: Bonus for hitting opponent (default: 0.05)
        safety_weight: Weight for making safe points (default: 0.02)
    """
    
    def __init__(self, env, 
                 pip_weight: float = 0.01,
                 bear_off_bonus: float = 0.1,
                 hit_bonus: float = 0.05,
                 safety_weight: float = 0.02):
        super().__init__(env)
        self.pip_weight = pip_weight
        self.bear_off_bonus = bear_off_bonus
        self.hit_bonus = hit_bonus
        self.safety_weight = safety_weight
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def _compute_pip_count(self, board: jnp.ndarray, player_idx: int) -> float:
        """
        Compute pip count for a player (lower is better).
        Pip count = sum of (checkers on point * distance to bear off)
        """
        # For white (idx 0): distance = 24 - point_idx for points 0-23
        # For black (idx 1): distance = point_idx + 1 for points 0-23
        
        points_board = board[player_idx, :24]  # Only board points, not bar/home
        
        white_distances = 24 - jnp.arange(24)
        black_distances = jnp.arange(24) + 1
        
        distances = jax.lax.cond(
            player_idx == 0,
            lambda _: white_distances,
            lambda _: black_distances,
            operand=None
        )
        
        pip_count = jnp.sum(points_board * distances)
        
        # Add bar checkers (max distance = 25)
        bar_checkers = board[player_idx, 24]
        pip_count = pip_count + bar_checkers * 25
        
        return pip_count.astype(jnp.float32)
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def _count_safe_points(self, board: jnp.ndarray, player_idx: int) -> int:
        """Count number of points with 2+ checkers (safe points)."""
        points = board[player_idx, :24]
        return jnp.sum(points >= 2)
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def _compute_shaped_reward(self, prev_state: BackgammonState, 
                                state: BackgammonState,
                                base_reward: float) -> float:
        """Compute the shaped reward with all components."""
        
        # Determine which player just moved (prev_state.current_player)
        player = prev_state.current_player
        player_idx = jax.lax.cond(player == 1, lambda _: 0, lambda _: 1, operand=None)
        opponent_idx = 1 - player_idx
        
        shaped_reward = base_reward
        
        # 1. Pip count improvement
        prev_pip = self._compute_pip_count(prev_state.board, player_idx)
        curr_pip = self._compute_pip_count(state.board, player_idx)
        pip_improvement = prev_pip - curr_pip  # Lower pip is better
        shaped_reward = shaped_reward + self.pip_weight * pip_improvement
        
        # 2. Bearing off bonus
        prev_borne = prev_state.board[player_idx, 25]
        curr_borne = state.board[player_idx, 25]
        new_borne = curr_borne - prev_borne
        shaped_reward = shaped_reward + self.bear_off_bonus * new_borne
        
        # 3. Hit bonus (opponent gained bar checkers)
        prev_opp_bar = prev_state.board[opponent_idx, 24]
        curr_opp_bar = state.board[opponent_idx, 24]
        hits = curr_opp_bar - prev_opp_bar
        shaped_reward = shaped_reward + self.hit_bonus * jnp.maximum(0, hits)
        
        # 4. Safety bonus (making points)
        prev_safe = self._count_safe_points(prev_state.board, player_idx)
        curr_safe = self._count_safe_points(state.board, player_idx)
        new_safe = curr_safe - prev_safe
        shaped_reward = shaped_reward + self.safety_weight * new_safe
        
        return shaped_reward.astype(jnp.float32)
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state: BackgammonState, action: Union[int, float]) -> Tuple[Any, BackgammonState, float, bool, Dict]:
        """Step with shaped rewards."""
        obs, new_state, base_reward, done, info = self._env.step(state, action)
        
        # Compute shaped reward
        shaped_reward = self._compute_shaped_reward(state, new_state, base_reward)
        
        return obs, new_state, shaped_reward, done, info


class NoHitsBackgammonWrapper(JaxatariWrapper):
    """
    Disables hitting in Backgammon for simplified learning.
    
    When enabled, landing on an opponent's blot does NOT send them to the bar.
    This creates a simpler game where pieces can pass through each other
    (but still can't land on blocked points with 2+ opponent checkers).
    
    Useful for initial RL training before introducing full complexity.
    """
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state: BackgammonState, action: Union[int, float]) -> Tuple[Any, BackgammonState, float, bool, Dict]:
        """Step without allowing hits - restore opponent blots after move."""
        # Store opponent's bar count before move
        prev_board = state.board
        
        # Execute normal step
        obs, new_state, reward, done, info = self._env.step(state, action)
        
        # If opponent gained bar checkers, undo the hit
        # by restoring the blot to its original position
        opponent_idx = jax.lax.cond(
            state.current_player == 1,
            lambda _: 1,  # White's opponent is Black (idx 1)
            lambda _: 0,  # Black's opponent is White (idx 0)
            operand=None
        )
        
        prev_opp_bar = prev_board[opponent_idx, 24]
        new_opp_bar = new_state.board[opponent_idx, 24]
        
        # Number of hits to undo
        hits_to_undo = new_opp_bar - prev_opp_bar
        
        # Find where the hit occurred (the destination of the last move)
        last_to = new_state.last_move[1]
        
        # Restore: remove hit checkers from bar, put back on destination
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
        
        final_state = new_state._replace(board=new_board)
        
        return obs, final_state, reward, done, info


class ShortGameWrapper(JaxatariWrapper):
    """
    Creates very short backgammon games for rapid iteration.
    
    Starts with all checkers in the home board, ready to bear off.
    Games typically last only a few moves.
    
    Useful for testing bearing-off logic and end-game scenarios.
    
    Args:
        env: The BackgammonEnv to wrap
    """
    
    def _create_endgame_board(self) -> jnp.ndarray:
        """Create a board with all checkers in home, ready to bear off."""
        board = jnp.zeros((2, 26), dtype=jnp.int32)
        
        # White (player 0): All 15 checkers distributed in home (points 18-23)
        # Points 18-23 are White's home board
        board = board.at[0, 18].set(3)  # 3 on 6-point
        board = board.at[0, 19].set(3)  # 3 on 5-point
        board = board.at[0, 20].set(3)  # 3 on 4-point
        board = board.at[0, 21].set(2)  # 2 on 3-point
        board = board.at[0, 22].set(2)  # 2 on 2-point
        board = board.at[0, 23].set(2)  # 2 on 1-point
        
        # Black (player 1): All 15 checkers in home (points 0-5)
        board = board.at[1, 5].set(3)   # 3 on 6-point (Black's perspective)
        board = board.at[1, 4].set(3)   # 3 on 5-point
        board = board.at[1, 3].set(3)   # 3 on 4-point
        board = board.at[1, 2].set(2)   # 2 on 3-point
        board = board.at[1, 1].set(2)   # 2 on 2-point
        board = board.at[1, 0].set(2)   # 2 on 1-point
        
        return board
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Any, BackgammonState]:
        """Reset with endgame board position."""
        obs, state = self._env.reset(key)
        
        endgame_board = self._create_endgame_board()
        new_state = state._replace(board=endgame_board)
        new_obs = self._env._get_observation(new_state)
        
        return new_obs, new_state
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state: BackgammonState, action: Union[int, float]) -> Tuple[Any, BackgammonState, float, bool, Dict]:
        """Pass through to base environment."""
        return self._env.step(state, action)
