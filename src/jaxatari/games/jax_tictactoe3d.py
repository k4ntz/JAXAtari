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
        {'name': 'x', 'type': 'single', 'file': 'X.npy'},
        {'name': 'o', 'type': 'single', 'file': 'O.npy'},
    )
    
class TicTacToe3DConstants(NamedTuple):
    
     # --- Board ---
    BOARD_SIZE: int = 4
    
    # --- Cell encoding ---
    EMPTY: int = 0
    PLAYER_X: int = 1
    PLAYER_O: int = 2
    FIRST_PLAYER: int = PLAYER_X
    
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
    board: jnp.ndarray          # (4, 4, 4), uint8 or int32
    current_player: jnp.ndarray # scalar int32
    game_over: jnp.ndarray     # scalar bool_
    winner: jnp.ndarray         # scalar int32
    move_count: jnp.ndarray     # scalar int32

    
    

class JaxTicTacToe3D(JaxEnvironment):
    def __init__(self, consts: TicTacToe3DConstants=None):
        self.consts = consts or TicTacToe3DConstants()
        super().__init__(self.consts)
        self.renderer = TicTacToe3DRenderer(self.consts)
        
        
    def step(self, state: TicTacToe3DState, action: jnp.ndarray):
     # Decode action -> (x, y, z)
          pass
    

    
    def reset(self):
        """Reset game state to initial state"""
        return TicTacToe3DState(
        board=jnp.zeros((4, 4, 4), dtype=jnp.uint8),
        current_player=jnp.int32(self.consts.FIRST_PLAYER),
        game_over=jnp.bool_(False),
        winner=jnp.int32(0),
        move_count=jnp.int32(0),
    )

    
    
    
    def render(self, state):
        """Render current game state."""
        return self.renderer.render(state)
    
    def _check_winner(self, board):
        """Check if there's a winner on the 3D board."""
        # Check all lines (rows, columns, diagonals, verticals across layers)
        pass
    def observation_space(self):
        """Board state: (4,4,4) with values 0/1/2."""
        return spaces.Box(0, 2, shape=(4, 4, 4), dtype=jnp.uint8)

    def action_space(self):
        """64 possible moves (4×4×4 cells)."""
        return spaces.Discrete(64)
    
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, prev_state, state):
       return jnp.int32(0)

    
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state):
       return {}

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: TicTacToe3DState) -> jnp.ndarray:
        return jnp.logical_or(
        state.game_over,
        state.move_count >= 64
    )

    
    
class TicTacToe3DRenderer(JAXGameRenderer):
    def __init__(self, consts: TicTacToe3DConstants = None):
        self.consts = consts or TicTacToe3DConstants()   
        super().__init__(self.consts)                   

        self.config = render_utils.RendererConfig(
            game_dimensions=(210, 160),
            channels=3,
            #downscale=(84, 84)
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
        self.ORIGIN_X = self.consts.ORIGIN_X
        self.ORIGIN_Y = self.consts.ORIGIN_Y

        # Cell spacing within a single 4x4 layer
        self.CELL_W = self.consts.CELL_W
        self.CELL_H = self.consts.CELL_H

        # Layer offsets to create the "stacked 3D" illusion (z=0..3)
        # (dx, dy) per layer
        self.LAYER_OFFSETS = jnp.array(consts.LAYER_OFFSETS, dtype=jnp.int32)

        
    def cell_to_pixel(self, x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Convert board coordinates x,y,z) in [0..3] to screen pixel (px,py).
        
        """
        consts= self.consts
        
        dx, dy = self.LAYER_OFFSETS[z]
        px = jnp.int32(consts.ORIGIN_X) + x * jnp.int32(consts.CELL_W) + dx
        py = jnp.int32(consts.ORIGIN_Y) + y * jnp.int32(consts.CELL_H) + dy
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

        def render_one_cell(idx, r):
            # board size is always 4x4x4
            # idx in [0..63] since 64 cells in a board
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

            # v == 0 -> nothing, v == 1 -> X, v == 2 -> O
            r = jax.lax.cond(v == self.consts.PLAYER_X, draw_x, lambda rr: rr, r)
            r = jax.lax.cond(v == self.consts.PLAYER_O, draw_o, lambda rr: rr, r)
            return r

        raster = jax.lax.fori_loop(0, 64, render_one_cell , raster)

        # Convert palette raster to final RGB image
        return self.jr.render_from_palette(raster, self.PALETTE)
        
