"""
3D Tic-Tac-Toe for JAXAtari
JAX Implementation (JIT-compatible)

Based on JAXAtari Design Guide
Inherits from JaxEnvironment base class
"""

from typing import NamedTuple, Tuple
import os
from functools import partial
import jax
import jax.numpy as jnp
import chex
import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action


def _get_default_asset_config() -> tuple:
    """
    Default asset configuration for TicTacToe3D. """
    return (
        {'name': 'background', 'type': 'background', 'file': 'background.npy'},
        {'name': 'X', 'type': 'single', 'file': 'X.npy'},
        {'name': 'O', 'type': 'single', 'file': 'O.npy'},
    )
    
class TicTacToe3DConstants(NamedTuple):
    
     # --- Board ---
    BOARD_SIZE: int = 4
    
    # --- Cell encoding ---
    EMPTY: int = 0
    PLAYER_X: int = 1
    PLAYER_O: int = 2
    
     # --- Rendering layout  ---
    ORIGIN_X: int = 20
    ORIGIN_Y: int = 18
    CELL_W: int = 18
    CELL_H: int = 18
    
    LAYER_OFFSETS: Tuple[Tuple[int, int], ...] = (
        (0, 0),
        (10, 18),
        (20, 36),
        (30, 54),
    )
    
    ASSET_CONFIG: tuple = _get_default_asset_config()
        
        
        
class TicTacToe3DState(NamedTuple):
    board: jnp.ndarray          # (4, 4, 4) shape, values 0/1/2
    current_player: int         # 1 for X, 2 for O
    game_over: bool
    winner: int                 # 0=no winner, 1=X wins, 2=O wins
    move_count: int

    
    

class JaxTicTacToe3D(JaxEnvironment):
    def __init__(self):
        self.consts = TicTacToe3DConstants()
        self.renderer = TicTacToe3DRenderer(self.consts)
        super().__init__()
    
    def reset(self):
        """Reset game to initial state."""
        board = jnp.zeros((4, 4, 4), dtype=jnp.int32)
        return TicTacToe3DState(
            board=board,
            current_player=1,  # X starts
            game_over=False,
            winner=0,
            move_count=0,
        )
    
    def step(self, state, action):
        """Execute one game action."""
        # action should encode (x, y, z) position
        # Returns: next_state, reward, done, info
        pass
    
    def render(self, state):
        """Render current game state."""
        return self.renderer.render(state)
    
    def _check_winner(self, board):
        """Check if there's a winner on the 3D board."""
        # Check all lines (rows, columns, diagonals, verticals across layers)
        pass
    @property
    def observation_space(self):
        """Board state: (4,4,4) with values 0/1/2."""
        return spaces.Box(0, 2, shape=(4, 4, 4), dtype=jnp.uint8)

    @property
    def action_space(self):
        """64 possible moves (4×4×4 cells)."""
        return spaces.Discrete(64)


    
    

class TicTacToe3DRenderer(JAXGameRenderer):
    def __init__(self, consts : TicTacToe3DConstants= None):
        super().__init__(consts)
        self.consts = consts or TicTacToe3DConstants()
        h, w, _ = self.BACKGROUND.shape
        self.config = render_utils.RendererConfig(
            game_dimensions=(h, w),
            channels=3,
        )
        self.jr = render_utils.JaxRenderingUtils(self.config) 
        final_asset_config = list(self.consts.ASSET_CONFIG)
        sprite_path =f"{os.path.dirname(os.path.abspath(__file__))}/sprites/tictactoe3d"
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS,
        ) = self.jr.load_and_setup_assets(final_asset_config,sprite_path)
        
        self.ORIGIN_X = 20
        self.ORIGIN_Y = 18

        # Cell spacing within a single 4x4 layer
        self.CELL_W = 18
        self.CELL_H = 18

        # Layer offsets to create the "stacked 3D" illusion (z=0..3)
        # (dx, dy) per layer
        self.LAYER_OFFSETS = jnp.array(
            [
                [0, 0],
                [10, 18],
                [20, 36],
                [30, 54],
            ],
            dtype=jnp.int32,
        )
    def cell_to_pixel(self, x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Convert board coordinates (x,y,z) in [0..3] to screen pixel (px,py).
        JIT-safe (pure JAX ops).
        """
        dz = self.LAYER_OFFSETS[z]  # shape (2,)
        px = jnp.int32(self.ORIGIN_X) + jnp.int32(x) * jnp.int32(self.CELL_W) + dz[0]
        py = jnp.int32(self.ORIGIN_Y) + jnp.int32(y) * jnp.int32(self.CELL_H) + dz[1]
        return px, py    

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        """
        Returns an RGB frame (H, W, 3) uint8.
        """
        # Start from baked background as an object raster
        raster = self.jr.create_object_raster(self.BACKGROUND)

        x_mask = self.SHAPE_MASKS["x"]
        o_mask = self.SHAPE_MASKS["o"]
        board = state.board  # (4,4,4)

        def render_one_cell(r, idx):
            # idx in [0..63]
            z = idx // 16
            rem = idx - z * 16
            y = rem // 4
            x = rem - y * 4

            v = board[z, y, x]  # 0/1/2

            px, py = self.cell_to_pixel(x, y, z)

            def draw_x(rr):
                return self.jr.render_at(rr, px, py, x_mask)

            def draw_o(rr):
                return self.jr.render_at(rr, px, py, o_mask)

            # v == 0 -> no-op, v == 1 -> X, v == 2 -> O
            r = jax.lax.cond(v == 1, draw_x, lambda rr: rr, r)
            r = jax.lax.cond(v == 2, draw_o, lambda rr: rr, r)
            return r, None

        raster, _ = jax.lax.scan(render_one_cell, raster, jnp.arange(64, dtype=jnp.int32))

        # Convert palette raster to final RGB image
        return self.jr.render_from_palette(raster, self.PALETTE)
        
