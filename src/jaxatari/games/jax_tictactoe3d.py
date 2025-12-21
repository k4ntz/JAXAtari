"""
3D Tic-Tac-Toe for JAXAtari
JAX Implementation (JIT-compatible & Vectorized)

"""

from typing import NamedTuple, Tuple
import os
from functools import partial
import jax
import jax.numpy as jnp
import chex
import jax.lax 
import jaxatari.spaces as spaces
import numpy as np
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action

# --- Helper to generate masks once (Module Level) ---
def _generate_win_masks():
    """Generates the 76 winning line masks for 4x4x4."""
    masks = []
    
    # Helper to generate a single mask 4x4x4
    def get_mask(coords):
        m = np.zeros((4, 4, 4), dtype=np.int32)
        for z, y, x in coords:
            m[z, y, x] = 1
        return m

    # 1. Rows (x-axis) - 16 lines
    for z in range(4):
        for y in range(4):
            masks.append(get_mask([(z, y, x) for x in range(4)]))
            
    # 2. Cols (y-axis) - 16 lines
    for z in range(4):
        for x in range(4):
            masks.append(get_mask([(z, y, x) for y in range(4)]))
            
    # 3. Pillars (z-axis) - 16 lines
    for y in range(4):
        for x in range(4):
            masks.append(get_mask([(z, y, x) for z in range(4)]))
            
    # 4. Plane Diagonals (z-fixed) - 8 lines
    for z in range(4):
        masks.append(get_mask([(z, i, i) for i in range(4)]))
        masks.append(get_mask([(z, i, 3-i) for i in range(4)]))
        
    # 5. Plane Diagonals (y-fixed) - 8 lines
    for y in range(4):
        masks.append(get_mask([(i, y, i) for i in range(4)]))
        masks.append(get_mask([(3-i, y, i) for i in range(4)]))

    # 6. Plane Diagonals (x-fixed) - 8 lines
    for x in range(4):
        masks.append(get_mask([(i, i, x) for i in range(4)]))
        masks.append(get_mask([(3-i, i, x) for i in range(4)]))

    # 7. Space Diagonals - 4 lines
    masks.append(get_mask([(i, i, i) for i in range(4)]))
    masks.append(get_mask([(i, i, 3-i) for i in range(4)]))
    masks.append(get_mask([(i, 3-i, i) for i in range(4)]))
    masks.append(get_mask([(i, 3-i, 3-i) for i in range(4)]))
    
    return jnp.array(masks, dtype=jnp.int32)

# Pre-compute masks (Executed once at import time)
WIN_MASKS_ARRAY = _generate_win_masks()

# --- CALIBRATED COORDINATE GENERATOR ---
# Based on User Measurements:
# L0 Center: 61, 21  -> TopLeft approx 58, 18
# L1 Center: 61, 67  -> Diff Y = 46
# L2 Center: 61, 113 -> Diff Y = 46
# L3 Center: 61, 159 -> Diff Y = 46
# X is constant ~61 for the first column.

def _generate_pixel_coords():
    """Generates the pixel grid based on calibrated offsets."""
    coords = np.zeros((4, 4, 4, 2), dtype=np.int32)
    
    start_x = 58
    start_y = 18
    
    level_step_y = 46  # Huge jump between boards
    level_step_x = 0   # Vertically aligned
    
    cell_step_x = 8
    cell_step_y = 10
    row_skew_x = 5     # Diagonal perspective within a board
    
    for z in range(4): # Levels
        level_base_x = start_x + (z * level_step_x)
        level_base_y = start_y + (z * level_step_y)
        
        for y in range(4): # Rows
            for x in range(4): # Columns
                # Skew X by Y (isometric-ish look)
                px = level_base_x + (x * cell_step_x) + (y * row_skew_x)
                py = level_base_y + (y * cell_step_y)
                coords[z, y, x] = [px, py]
                
    return jnp.array(coords, dtype=jnp.int32)

PIXEL_COORDS_GENERATED = _generate_pixel_coords()


def _get_default_asset_config() -> tuple:
    """Default asset configuration for TicTacToe3D."""
    return (
        {'name': 'background', 'type': 'background', 'file': 'background.npy'},
        {'name': 'x', 'type': 'single', 'file': 'X.npy'},
        {'name': 'o', 'type': 'single', 'file': 'O.npy'},
        {'name': 'cursor', 'type': 'single', 'file': 'cursor.npy'},
    )
    
class TicTacToe3DConstants(NamedTuple):
    """Game constants."""
    BOARD_SIZE: int = 4
    EMPTY: int = 0
    PLAYER_X: int = 1
    PLAYER_O: int = 2
    FIRST_PLAYER: int = 1  # Player X starts
    
    # Store win masks in constants for easy JIT access
    WIN_MASKS: chex.Array = WIN_MASKS_ARRAY
    
    # Screen dimensions
    HEIGHT: int = 207
    WIDTH: int = 156
    NUM_ACTIONS: int = 8
    
    # Cursor settings
    BLINK_PERIOD: int = 15
    MOVE_COOLDOWN: int = 6
    
    # Cell rendering
    # ZERO OFFSET: Aligns 7x7 cursor perfectly with 8x10 grid cell
    CELL_CENTER_X: int = 0
    CELL_CENTER_Y: int = 0
    
    # Use the dynamically generated coordinates
    PIXEL_COORDS: chex.Array = PIXEL_COORDS_GENERATED

    ASSET_CONFIG: tuple = _get_default_asset_config()

class TicTacToe3DState(NamedTuple):
    board: jnp.ndarray          # (4, 4, 4), uint8 or int32
    current_player: jnp.ndarray # scalar int32
    game_over: jnp.ndarray      # scalar bool_
    winner: jnp.ndarray         # scalar int32
    move_count: jnp.ndarray     # scalar int32
    cursor_x: jnp.ndarray
    cursor_y: jnp.ndarray
    cursor_z: jnp.ndarray
    frame: jnp.ndarray

class TicTacToe3DObservation(NamedTuple):
    """Object-centric observation exposed to agent."""
    board: jnp.ndarray          # Shape (4,4,4): game state
    current_player: jnp.ndarray # Scalar: whose turn it logically is
    valid_moves: jnp.ndarray    # Shape (64,): bool mask of legal moves
    game_over: jnp.ndarray      # Scalar bool
    winner: jnp.ndarray         # Scalar int32

class TicTacToe3DInfo(NamedTuple):
    """Auxiliary diagnostic information."""
    move_count: jnp.ndarray
    game_phase: jnp.ndarray     # 0=ongoing, 1=x_won, 2=o_won, 3=draw
    last_move_player: jnp.ndarray  # Who made last move
    last_move_action: jnp.ndarray  # What action was taken

class JaxTicTacToe3DEnvironment(JaxEnvironment):
    """3D Tic-Tac-Toe environment for JAXAtari (Vectorized)."""
    def __init__(self, consts: TicTacToe3DConstants | None = None):
        self.consts = consts or TicTacToe3DConstants()
        super().__init__(self.consts)
        self.renderer = TicTacToe3DRenderer(self.consts)

        # Discrete action space indices 0..7 map to ALE actions
        self.action_set = [
            Action.NOOP,   # 0
            Action.FIRE,   # 1  SPACE
            Action.UP,     # 2  ↑
            Action.RIGHT,  # 3  →
            Action.LEFT,   # 4  ←
            Action.DOWN,   # 5  ↓
        ]
        self.ACTION_MAP = jnp.array(
            [int(a) for a in self.action_set],
            dtype=jnp.int32,
        )

        h, w = self.renderer.BACKGROUND.shape[:2]
        self._image_space = spaces.Box(low=0, high=255, shape=(h, w, 3), dtype=jnp.uint8)

    def action_space(self):
        return spaces.Discrete(len(self.action_set)) 

    def observation_space(self) -> spaces.Space:
        """Object-centric observation space."""
        return spaces.Dict({
            "board": spaces.Box(0, 2, shape=(4, 4, 4), dtype=jnp.int32),
            "current_player": spaces.Discrete(3),  # 1 or 2
            "valid_moves": spaces.Box(0, 1, shape=(64,), dtype=jnp.bool_),
            "game_over": spaces.Box(0, 1, shape=(), dtype=jnp.bool_),
            "winner": spaces.Box(0, 2, shape=(), dtype=jnp.int32),
        })
    
    def image_space(self) -> spaces.Space:
        """Rendered image space."""
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.consts.HEIGHT, self.consts.WIDTH, 3),
            dtype=jnp.uint8
        )

    def render(self, state: TicTacToe3DState) -> jnp.ndarray:
        return self.renderer.render(state)

    def reset(self, key: chex.PRNGKey):
        state = TicTacToe3DState(
            board=jnp.zeros((4, 4, 4), dtype=jnp.uint8),
            current_player=jnp.int32(self.consts.FIRST_PLAYER),
            game_over=jnp.bool_(False),
            winner=jnp.int32(0),
            move_count=jnp.int32(0),
            cursor_x=jnp.int32(0),
            cursor_y=jnp.int32(0),
            cursor_z=jnp.int32(0),
            frame=jnp.int32(0),
        )

        observation = state.board
        return observation, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: TicTacToe3DState, action: jnp.ndarray):
        """Execute one game step."""
        
        # --- Move cursor based on action ---
        cursor_array = jnp.array([state.cursor_x, state.cursor_y, state.cursor_z])
        new_cursor = self._move_cursor(cursor_array, action, state.frame)
        cx, cy, cz = new_cursor[0], new_cursor[1], new_cursor[2]
        
        board = state.board
        player = state.current_player
        
        # --- PLACE action ---
        is_place = action == Action.FIRE
        is_empty = board[cz, cy, cx] == self.consts.EMPTY
        can_place = jnp.logical_and(is_place, is_empty)
        can_place = jnp.logical_and(can_place, jnp.logical_not(state.game_over))  # Can't play if game over
        
        def place_mark(b):
            return b.at[cz, cy, cx].set(player.astype(b.dtype))
        
        new_board = jax.lax.cond(
            can_place,
            place_mark,
            lambda b: b,
            board
        )
        
        # --- Check for winner (Vectorized) ---
        winner = self._check_winner(new_board)
        game_over = jnp.logical_or(winner != self.consts.EMPTY, state.move_count >= 63)
        
        # --- Switch player only if placed ---
        next_player = jnp.where(
            player == self.consts.PLAYER_X,
            self.consts.PLAYER_O,
            self.consts.PLAYER_X
        )
        
        new_player = jax.lax.cond(
            can_place,
            lambda _: next_player,
            lambda _: player,
            operand=None
        )
        
        new_move_count = jax.lax.cond(
            can_place,
            lambda c: c + 1,
            lambda c: c,
            state.move_count
        )
        
        # --- Build new state ---
        new_state = state._replace(
            board=new_board,
            current_player=new_player,
            move_count=new_move_count,
            cursor_x=cx,
            cursor_y=cy,
            cursor_z=cz,
            frame=state.frame + 1,
            winner=winner,      
            game_over=game_over
        )
        
        # --- Compute observation ---
        valid_moves = (new_board == self.consts.EMPTY).reshape(64)
        observation = new_state.board
        
        # --- Compute reward ---
        reward = self._get_reward(state, new_state)
        
        # --- Compute done ---
        done = self._get_done(new_state)
        
        # --- Compute info ---
        info = self._get_info(state, new_state, action)
        
        return observation, new_state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _move_cursor(self, cursor, ale_action, step_count):
        """
        Move cursor based on action with Layer Navigation.
        - Horizontal: Wraps or Clips on current layer.
        - Vertical: Moving past top/bottom edge switches layers (z-axis).
        """
        x, y, z = cursor[0], cursor[1], cursor[2]
        
        can_move = (step_count % self.consts.MOVE_COOLDOWN) == 0

        # --- Horizontal Move (Simple Clamp) ---
        dx = jnp.where(ale_action == int(Action.LEFT), -1,
             jnp.where(ale_action == int(Action.RIGHT), 1, 0))
        
        new_x = jnp.clip(x + dx, 0, self.consts.BOARD_SIZE - 1)
        new_x = jnp.where(can_move, new_x, x)

        # --- Vertical Move (Layer Transition Logic) ---
        dy = jnp.where(ale_action == int(Action.UP), -1,
             jnp.where(ale_action == int(Action.DOWN), 1, 0))
        
        raw_y = y + dy
        
        # Check transitions
        # Moving DOWN past bottom (y=3) -> go to next layer (z+1), reset y to 0
        move_down_layer = jnp.logical_and(dy == 1, raw_y > 3)
        # Moving UP past top (y=0) -> go to prev layer (z-1), reset y to 3
        move_up_layer = jnp.logical_and(dy == -1, raw_y < 0)
        
        # Calculate potential new Z
        z_down = jnp.clip(z + 1, 0, 3)
        z_up = jnp.clip(z - 1, 0, 3)
        
        new_z = jnp.where(move_down_layer, z_down,
                jnp.where(move_up_layer, z_up, z))
        
        # Check if Z actually changed (to avoid wrapping at the very top/bottom of entire stack)
        z_changed = new_z != z
        
        # Determine new Y
        new_y_if_down = 0
        new_y_if_up = 3
        new_y_standard = jnp.clip(raw_y, 0, 3)
        
        # If we changed layer, snap Y. If not, clip Y.
        new_y = jnp.where(jnp.logical_and(move_down_layer, z_changed), new_y_if_down,
                jnp.where(jnp.logical_and(move_up_layer, z_changed), new_y_if_up,
                new_y_standard))

        # Apply cooldown check
        final_y = jnp.where(can_move, new_y, y)
        final_z = jnp.where(can_move, new_z, z)
        # Preserve X when switching layers (standard Atari behavior)
        final_x = new_x

        return jnp.array([final_x, final_y, final_z], dtype=jnp.int32)

    @partial(jax.jit, static_argnums=(0,))
    def _check_winner(self, board: jnp.ndarray) -> jnp.ndarray:
        """
        100% vectorized win check using tensor contraction.
        Complexity: O(1) ops, no Python loops.
        """
        # Create boolean masks for each player
        is_x = (board == self.consts.PLAYER_X).astype(jnp.int32)
        is_o = (board == self.consts.PLAYER_O).astype(jnp.int32)
        
        # Tensor dot product against the 76 winning masks
        # Masks: (76, 4, 4, 4), Board: (4, 4, 4)
        # Result: (76,) containing the count of markers in each winning line
        scores_x = jnp.tensordot(self.consts.WIN_MASKS, is_x, axes=((1, 2, 3), (0, 1, 2)))
        scores_o = jnp.tensordot(self.consts.WIN_MASKS, is_o, axes=((1, 2, 3), (0, 1, 2)))
        
        # Check if any line has sum == 4
        x_wins = jnp.any(scores_x == 4)
        o_wins = jnp.any(scores_o == 4)
        
        return jnp.where(x_wins, self.consts.PLAYER_X, 
               jnp.where(o_wins, self.consts.PLAYER_O, self.consts.EMPTY))
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(
        self, previous_state: TicTacToe3DState, state: TicTacToe3DState
    ) -> jnp.ndarray:
        """Calculate reward for state transition."""
        player_won = state.winner == self.consts.PLAYER_X
        opponent_won = state.winner == self.consts.PLAYER_O
        return jnp.where(player_won, 1.0, jnp.where(opponent_won, -1.0, 0.0))
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_info(
        self,
        previous_state: TicTacToe3DState,
        state: TicTacToe3DState,
        action: jnp.ndarray
    ) -> TicTacToe3DInfo:
        """Extract diagnostic information from transition."""
        game_phase = jnp.where(
            state.winner == self.consts.PLAYER_X,
            1,  # X won
            jnp.where(
                state.winner == self.consts.PLAYER_O,
                2,  # O won
                jnp.where(
                    state.move_count >= 64,  # 4*4*4=64
                    3,  # Draw
                    0   # Ongoing
                )
            )
        )
        
        return TicTacToe3DInfo(
            move_count=state.move_count,
            game_phase=jnp.int32(game_phase),
            last_move_player=previous_state.current_player,  # Who made the move
            last_move_action=action,  # What action was taken
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: TicTacToe3DState) -> jnp.ndarray:
        return jnp.logical_or(
            state.game_over,
            state.move_count >= 64
        )


class TicTacToe3DRenderer(JAXGameRenderer):
    def __init__(self, consts: TicTacToe3DConstants | None = None):
        self.consts = consts or TicTacToe3DConstants()
        super().__init__(self.consts)

        self.config = render_utils.RendererConfig(
            game_dimensions=(self.consts.HEIGHT, self.consts.WIDTH),
            channels=3,
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)

        # 1) Start from constants asset config
        final_asset_config = list(self.consts.ASSET_CONFIG)

        # 2) Sprite path
        sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/tictactoe3d"

        # 3) Bake assets once
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS,
        ) = self.jr.load_and_setup_assets(final_asset_config, sprite_path)

        self.x_mask = self.SHAPE_MASKS["x"]
        self.o_mask = self.SHAPE_MASKS["o"]
        self.cursor_mask = self.SHAPE_MASKS["cursor"]

    def cell_to_pixel(self, x, y, z):
        """Get pixel coordinates - JAX-native indexing."""
        px = self.consts.PIXEL_COORDS[z, y, x, 0]
        py = self.consts.PIXEL_COORDS[z, y, x, 1]
        return px, py

    def cell_center_to_pixel(self, x, y, z):
        """Get pixel coordinates for cell center (with offset for cursor)."""
        px, py = self.cell_to_pixel(x, y, z) 
        # Apply offset if defined, otherwise align with X/O
        cpx = px + self.consts.CELL_CENTER_X
        cpy = py + self.consts.CELL_CENTER_Y
        return cpx, cpy

    def render(self, state):
        """
        Decoupled Renderer: Eager execution.
        Safely runs outside JIT to prevent pure_callback crashes on macOS.
        """
        raster = self.jr.create_object_raster(self.BACKGROUND)
        
        # Access data directly (works for both JIT traces and concrete values)
        board = state.board
        
        # Using standard loops here allows this to run safely in eager mode
        # without building a massive graph or using pure_callback
        for idx in range(64):
            z = idx // 16
            y = (idx % 16) // 4
            x = idx % 4
            
            cell_val = board[z, y, x]
            px, py = self.cell_to_pixel(x, y, z)
            
            # Use lax.cond to remain compatible if user mistakenly JITs this,
            # but relies on standard loop structure to avoid OOM.
            raster = jax.lax.cond(
                cell_val == self.consts.PLAYER_X,
                lambda r: self.jr.render_at(r, px, py, self.x_mask),
                lambda r: r,
                raster
            )
            
            raster = jax.lax.cond(
                cell_val == self.consts.PLAYER_O,
                lambda r: self.jr.render_at(r, px, py, self.o_mask),
                lambda r: r,
                raster
            )

        # Draw Cursor
        cpx, cpy = self.cell_center_to_pixel(state.cursor_x, state.cursor_y, state.cursor_z)
        blink_on = (state.frame // 15) % 2 == 0
        
        raster = jax.lax.cond(
            blink_on,
            lambda r: self.jr.render_at(r, cpx, cpy, self.cursor_mask),
            lambda r: r,
            raster
        )
        
        return self.jr.render_from_palette(raster, self.PALETTE)