"""
3D Tic-Tac-Toe for JAXAtari
JAX Implementation (JIT-compatible)

Based on JAXAtari Design Guide
Inherits from JaxEnvironment base class
"""

from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
from jax import lax
import jax.random as jrandom
import numpy as np

# Import the base class
from jaxatari.environment import JaxEnvironment, JAXAtariAction
from jaxatari.spaces import Discrete, Dict as DictSpace, Box

# ============================================================================
# DATA STRUCTURES (NamedTuples) - Must work with JAX
# ============================================================================

class TicTactoe3DConstants(NamedTuple):
    """Static game parameters"""
    BOARD_SIZE: int = 4  # 4x4x4 cube
    SCREEN_WIDTH: int = 210
    SCREEN_HEIGHT: int = 160
    FRAME_SKIP: int = 1
    
    # Colors (RGB)
    BACKGROUND_COLOR: Tuple[int, int, int] = (0, 0, 0)
    PLAYER_COLOR: Tuple[int, int, int] = (0, 255, 0)  # Green
    AI_COLOR: Tuple[int, int, int] = (255, 0, 0)      # Red
    GRID_COLOR: Tuple[int, int, int] = (100, 100, 100)
    EMPTY_COLOR: Tuple[int, int, int] = (50, 50, 50)


class TicTactoe3DState(NamedTuple):
    """Dynamic game state (changes each step) - JAX version"""
    board: jnp.ndarray  # Shape (4, 4, 4), dtype=int8
    current_player: jnp.ndarray  # int8 (1 or -1)
    game_over: jnp.ndarray  # bool (0 or 1)
    winner: jnp.ndarray  # int8 (0, 1, -1, or 2)
    player_score: jnp.ndarray  # int32
    ai_score: jnp.ndarray  # int32
    move_count: jnp.ndarray  # int32
    step_counter: jnp.ndarray  # int32


class TicTactoe3DObservation(NamedTuple):
    """Object-centric observation for the agent"""
    board: jnp.ndarray  # Shape (4, 4, 4), dtype=int8
    player_to_move: jnp.ndarray  # int8
    game_over: jnp.ndarray  # bool
    winner: jnp.ndarray  # int8


class TicTactoe3DInfo(NamedTuple):
    """Auxiliary information"""
    move_count: jnp.ndarray  # int32
    winner: jnp.ndarray  # int8
    reason: int  # encoded as int (0=ongoing, 1=player_won, -1=ai_won, 2=draw)


# ============================================================================
# 3D TIC-TAC-TOE ENVIRONMENT - JAX VERSION
# ============================================================================

class JaxTicTactoe3D(JaxEnvironment):
    """3D Tic-Tac-Toe Environment for JAXAtari - JAX Implementation
    
    Inherits from JaxEnvironment base class to match JAXAtari API.
    """
    
    def __init__(self, consts: TicTactoe3DConstants = None):
        """Initialize the environment (CPU-only, not JIT-compiled)"""
        super().__init__(consts)
        self.consts = consts if consts is not None else TicTactoe3DConstants()
        
        # Pre-compute winning lines as JAX array for efficiency
        self.winning_lines = self._precompute_winning_lines_jax()
        
        # Pre-compute line checking array for fast vectorized checking
        self.num_winning_lines = len(self.winning_lines)
    
    def _precompute_winning_lines_jax(self):
        """Pre-compute all possible winning lines in 4x4x4 cube"""
        lines_list = []
        board_size = self.consts.BOARD_SIZE
        
        # Lines along x-axis
        for y in range(board_size):
            for z in range(board_size):
                lines_list.append([(x, y, z) for x in range(board_size)])
        
        # Lines along y-axis
        for x in range(board_size):
            for z in range(board_size):
                lines_list.append([(x, y, z) for y in range(board_size)])
        
        # Lines along z-axis
        for x in range(board_size):
            for y in range(board_size):
                lines_list.append([(x, y, z) for z in range(board_size)])
        
        # Main diagonals (x-y plane for each z)
        for z in range(board_size):
            lines_list.append([(i, i, z) for i in range(board_size)])
            lines_list.append([(i, board_size-1-i, z) for i in range(board_size)])
        
        # Main diagonals (x-z plane for each y)
        for y in range(board_size):
            lines_list.append([(i, y, i) for i in range(board_size)])
            lines_list.append([(i, y, board_size-1-i) for i in range(board_size)])
        
        # Main diagonals (y-z plane for each x)
        for x in range(board_size):
            lines_list.append([(x, i, i) for i in range(board_size)])
            lines_list.append([(x, i, board_size-1-i) for i in range(board_size)])
        
        # Space diagonals (corner to corner)
        lines_list.append([(i, i, i) for i in range(board_size)])
        lines_list.append([(i, i, board_size-1-i) for i in range(board_size)])
        lines_list.append([(i, board_size-1-i, i) for i in range(board_size)])
        lines_list.append([(i, board_size-1-i, board_size-1-i) for i in range(board_size)])
        
        return lines_list
    
    def reset(self, key: jrandom.PRNGKey = None) -> Tuple[TicTactoe3DObservation, TicTactoe3DState]:
        """Reset to initial state (JIT-compatible)"""
        board = jnp.zeros((4, 4, 4), dtype=jnp.int8)
        state = TicTactoe3DState(
            board=board,
            current_player=jnp.array(1, dtype=jnp.int8),
            game_over=jnp.array(False, dtype=jnp.bool_),
            winner=jnp.array(0, dtype=jnp.int8),
            player_score=jnp.array(0, dtype=jnp.int32),
            ai_score=jnp.array(0, dtype=jnp.int32),
            move_count=jnp.array(0, dtype=jnp.int32),
            step_counter=jnp.array(0, dtype=jnp.int32)
        )
        obs = self._get_observation(state)
        return obs, state
    
    @staticmethod
    def _check_winner_np(board_np, winning_lines_list) -> int:
        """Check winner using numpy (CPU-side helper)"""
        for line in winning_lines_list:
            values = [board_np[pos] for pos in line]
            if values[0] != 0 and all(v == values[0] for v in values):
                return int(values[0])
        return 0
    
    def _check_winner_jax(self, board: jnp.ndarray) -> jnp.int8:
        """Check for winner using pure JAX operations (JIT-compatible)"""
        # Convert winning_lines to JAX array for vectorization
        winning_lines_array = jnp.array(self.winning_lines, dtype=jnp.int32)
        
        def check_single_line(line_indices):
            """Check if a single line is a winner"""
            x_indices = line_indices[:, 0]
            y_indices = line_indices[:, 1]
            z_indices = line_indices[:, 2]
            
            # Get values at these positions
            values = board[x_indices, y_indices, z_indices]
            
            # Check if all are equal and non-zero
            first_val = values[0]
            all_equal = jnp.all(values == first_val)
            non_zero = first_val != 0
            is_winner = all_equal & non_zero
            
            return jnp.where(is_winner, first_val, jnp.int8(0))
        
        # Check all lines and return first winner
        winner_values = jax.vmap(check_single_line)(winning_lines_array)
        
        # Return first non-zero value, or 0
        # Use reduce to get the first non-zero element
        def pick_first_winner(carry, val):
            return lax.cond(carry != 0, lambda _: carry, lambda _: val, None), None
        
        result, _ = lax.scan(pick_first_winner, jnp.int8(0), winner_values)
        return result


    def step(self, state: TicTactoe3DState, action: int) -> Tuple[TicTactoe3DObservation, TicTactoe3DState, float, bool, TicTactoe3DInfo]:
        """Execute one game step - Fully JAX-compatible with deterministic AI"""
        
        # Unpack action into coordinates
        x = action // 16
        y = (action % 16) // 4
        z = action % 4
        
        # Check if move is valid
        is_valid = state.board[x, y, z] == 0
        
        # Apply player move
        new_board_with_player = state.board.at[x, y, z].set(jnp.int8(1))
        
         # Check if player wins (using pure JAX)
        player_winner = self._check_winner_jax(new_board_with_player)
        
        # Find first empty position for AI
        flat_board = new_board_with_player.flatten()
        empty_positions = jnp.where(flat_board == 0, size=64, fill_value=64)[0]
        ai_action = empty_positions[0]
        ai_x = ai_action // 16
        ai_y = (ai_action % 16) // 4
        ai_z = ai_action % 4
        
        # Apply AI move
        new_board_with_ai = new_board_with_player.at[ai_x, ai_y, ai_z].set(jnp.int8(-1))
        
        # Check if AI wins (using pure JAX)
        ai_winner = self._check_winner_jax(new_board_with_ai)
        
        # Check if board is full after player move
        player_board_full = (flat_board == 0).sum() == 0
        
        # Now use lax.cond only for branching logic
        def game_over_case(_):
            obs = self._get_observation(state)
            info = self._get_info(state)
            return obs, state, jnp.array(0.0), jnp.array(True), info
        
        def invalid_move_case(_):
            obs = self._get_observation(state)
            info = self._get_info(state)
            return obs, state, jnp.array(-0.1), jnp.array(False), info
        
        def player_wins_case(_):
            new_state = TicTactoe3DState(
                board=new_board_with_player,
                current_player=jnp.array(1, dtype=jnp.int8),
                game_over=jnp.array(True, dtype=jnp.bool_),
                winner=jnp.array(1, dtype=jnp.int8),
                player_score=state.player_score + 1,
                ai_score=state.ai_score,
                move_count=state.move_count + 1,
                step_counter=state.step_counter + 1
            )
            obs = self._get_observation(new_state)
            info = self._get_info(new_state)
            return obs, new_state, jnp.array(1.0), jnp.array(True), info
        
        def player_board_full_case(_):
            new_state = TicTactoe3DState(
                board=new_board_with_player,
                current_player=jnp.array(1, dtype=jnp.int8),
                game_over=jnp.array(True, dtype=jnp.bool_),
                winner=jnp.array(2, dtype=jnp.int8),
                player_score=state.player_score,
                ai_score=state.ai_score,
                move_count=state.move_count + 1,
                step_counter=state.step_counter + 1
            )
            obs = self._get_observation(new_state)
            info = self._get_info(new_state)
            return obs, new_state, jnp.array(0.0), jnp.array(True), info
        
        def ai_wins_case(_):
            new_state = TicTactoe3DState(
                board=new_board_with_ai,
                current_player=jnp.array(1, dtype=jnp.int8),
                game_over=jnp.array(True, dtype=jnp.bool_),
                winner=jnp.array(-1, dtype=jnp.int8),
                player_score=state.player_score,
                ai_score=state.ai_score + 1,
                move_count=state.move_count + 2,
                step_counter=state.step_counter + 2
            )
            obs = self._get_observation(new_state)
            info = self._get_info(new_state)
            return obs, new_state, jnp.array(-1.0), jnp.array(True), info
        
        def game_continues_case(_):
            new_state = TicTactoe3DState(
                board=new_board_with_ai,
                current_player=jnp.array(1, dtype=jnp.int8),
                game_over=jnp.array(False, dtype=jnp.bool_),
                winner=jnp.array(0, dtype=jnp.int8),
                player_score=state.player_score,
                ai_score=state.ai_score,
                move_count=state.move_count + 2,
                step_counter=state.step_counter + 2
            )
            obs = self._get_observation(new_state)
            info = self._get_info(new_state)
            return obs, new_state, jnp.array(0.0), jnp.array(False), info
        
        # Decision tree: game_over → invalid → player_winner → board_full → ai_winner → continue
        def handle_valid_move(_):
            def handle_player_move(_):
                def handle_after_player(_):
                    return lax.cond(
                        ai_winner != 0,
                        ai_wins_case,
                        game_continues_case,
                        None
                    )
                return lax.cond(
                    player_board_full,
                    player_board_full_case,
                    handle_after_player,
                    None
                )
            return lax.cond(
                player_winner != 0,
                player_wins_case,
                handle_player_move,
                None
            )
        
        # Main control flow
        result = lax.cond(
            state.game_over,
            game_over_case,
            lambda _: lax.cond(
                is_valid,
                handle_valid_move,
                invalid_move_case,
                None
            ),
            None
        )
        
        return result



    
    def render(self, state: TicTactoe3DState) -> jnp.ndarray:
        """Render the game state as RGB image"""
        screen = jnp.zeros((self.consts.SCREEN_HEIGHT, self.consts.SCREEN_WIDTH, 3), dtype=jnp.uint8)
        
        # Fill background
        screen = screen + jnp.array(self.consts.BACKGROUND_COLOR, dtype=jnp.uint8)
        
        cell_size = 20
        margin = 10
        
        # Simple rendering (not optimized yet)
        for layer in range(4):
            for y in range(4):
                for x in range(4):
                    sx = margin + layer * (4 * cell_size + 5) + x * cell_size
                    sy = margin + y * cell_size
                    
                    cell = state.board[x, y, layer]
                    
                    color = jnp.where(
                        cell == 1,
                        jnp.array(self.consts.PLAYER_COLOR, dtype=jnp.uint8),
                        jnp.where(
                            cell == -1,
                            jnp.array(self.consts.AI_COLOR, dtype=jnp.uint8),
                            jnp.array(self.consts.EMPTY_COLOR, dtype=jnp.uint8)
                        )
                    )
                    
                    screen = screen.at[sy:sy+cell_size, sx:sx+cell_size].set(color)
        
        return screen
    
    def action_space(self) -> Discrete:
        """Returns the action space"""
        return Discrete(64)
    
    def observation_space(self) -> DictSpace:
        """Returns the observation space"""
        return DictSpace({
        "board": Box(low=-1, high=1, shape=(4, 4, 4), dtype=np.int16),
        "player_to_move": Box(low=-1, high=1, shape=(), dtype=np.int16),
        "game_over": Box(low=0, high=1, shape=(), dtype=np.int16),
        "winner": Box(low=-1, high=2, shape=(), dtype=np.int16)
                            })



    
    def image_space(self) -> Box:
        """Returns the image space"""
        return Box(low=0, high=255, shape=(self.consts.SCREEN_HEIGHT, self.consts.SCREEN_WIDTH, 3), dtype=np.uint8)
    
    def _get_observation(self, state: TicTactoe3DState) -> TicTactoe3DObservation:
        """Convert state to observation"""
        return TicTactoe3DObservation(
        board=jnp.array(state.board, dtype=jnp.int16),  
        player_to_move=jnp.array(state.current_player, dtype=jnp.int16),
        game_over=jnp.array(state.game_over, dtype=jnp.int16),
        winner=jnp.array(state.winner, dtype=jnp.int16)
    )

    
    def obs_to_flat_array(self, obs: TicTactoe3DObservation) -> jnp.ndarray:
        """Convert observation to flat array"""
        flat = jnp.concatenate([
            obs.board.flatten(),
            jnp.array([obs.player_to_move, jnp.uint8(obs.game_over), obs.winner], dtype=jnp.int8)
        ])
        return flat
    
    def _get_info(self, state: TicTactoe3DState, all_rewards: jnp.ndarray = None) -> TicTactoe3DInfo:
        """Get auxiliary information - JAX-compatible version"""
        return TicTactoe3DInfo(
            move_count=state.move_count,
            winner=state.winner,
            reason=0  # Keep as Python int (NamedTuple field, not used in step())
        )

    
    def _get_reward(self, previous_state: TicTactoe3DState, state: TicTactoe3DState) -> float:
        """Calculate reward from transition"""
        return 0.0
    
    def _get_done(self, state: TicTactoe3DState) -> jnp.bool_:
        """Determine if game is over"""
        return state.game_over  

