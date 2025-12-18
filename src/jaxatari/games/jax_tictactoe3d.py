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
        {'name': 'cursor', 'type': 'single', 'file': 'cursor.npy'},
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
    ORIGIN_X: int = 40
    ORIGIN_Y: int = 20
    CELL_W: int = 18
    CELL_H: int = 18
    PLACE: int = 7
    
    LAYER_OFFSETS: Tuple[Tuple[int, int], ...] = (
        (0, 0),
        (6, 6),
        (12, 12),
        (18, 18),
        )
    CURSOR_OFFSET_X: int = 2
    CURSOR_OFFSET_Y: int = 2

    
    ASSET_CONFIG: tuple = _get_default_asset_config()
        
        
        
class TicTacToe3DState(NamedTuple):
    board: jnp.ndarray          # (4, 4, 4), uint8 or int32
    current_player: jnp.ndarray # scalar int32
    game_over: jnp.ndarray     # scalar bool_
    winner: jnp.ndarray         # scalar int32
    move_count: jnp.ndarray     # scalar int32
    cursor_x: jnp.ndarray
    cursor_y: jnp.ndarray
    cursor_z: jnp.ndarray
    frame: jnp.ndarray
    
    

class JaxTicTacToe3D(JaxEnvironment):
    def __init__(self, consts: TicTacToe3DConstants=None):
        self.consts = consts or TicTacToe3DConstants()
        super().__init__(self.consts)
        self.renderer = TicTacToe3DRenderer(self.consts)
        self.action_set = [
                Action.NOOP,
                Action.LEFT,
                Action.RIGHT,
                Action.UP,
                Action.DOWN,
                Action.FIRE,
]

        

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: TicTacToe3DState, action: jnp.ndarray):

    # --- Cursor position ---
        cx, cy, cz = state.cursor_x, state.cursor_y, state.cursor_z
        
        cx = jax.lax.select(action == Action.LEFT,  jnp.maximum(cx - 1, 0), cx)
        cx = jax.lax.select(action == Action.RIGHT, jnp.minimum(cx + 1, 3), cx)
        cy = jax.lax.select(action == Action.UP,    jnp.maximum(cy - 1, 0), cy)
        cy = jax.lax.select(action == Action.DOWN,  jnp.minimum(cy + 1, 3), cy)


        board = state.board
        player = state.current_player

    # --- PLACE action ---
        is_place = action == Action.FIRE

        is_empty = board[cz, cy, cx] == self.consts.EMPTY
        can_place = jnp.logical_and(is_place, is_empty)

        def place_mark(b):
            return b.at[cz, cy, cx].set(player.astype(b.dtype))

        new_board = jax.lax.cond(
            can_place,
            place_mark,
            lambda b: b,
            board,)

    # --- Switch player only if placed ---
        next_player = jnp.where(
            player == self.consts.PLAYER_X,
            self.consts.PLAYER_O,
            self.consts.PLAYER_X,
    )

        new_player = jax.lax.cond(
            can_place,
            lambda _: next_player,
            lambda _: player,
            operand=None,
    )

        new_move_count = jax.lax.cond(
            can_place,
            lambda c: c + 1,
            lambda c: c,
            state.move_count,
    )
        new_state = state._replace(
            board=new_board,
            current_player=new_player,
            move_count=new_move_count,
            cursor_x=cx,
            cursor_y=cy,
            cursor_z=cz,
            frame=state.frame + 1
    )

        observation = new_state.board
        reward = jnp.int32(0)
        done = self._get_done(new_state)
        info = self._get_info(state, new_state, action)

        return observation, new_state, reward, done, info

    

    
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
        return spaces.Discrete(len(self.action_set))


    
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state, state):
       return jnp.int32(0)

    
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, previous_state,state, action ):
       return {}

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: TicTacToe3DState) -> jnp.ndarray:
        return jnp.logical_or(
        state.game_over,
        state.move_count >= 64
    )

    
    
class TicTacToe3DRenderer(JAXGameRenderer):
    def __init__(self, consts: TicTacToe3DConstants=None):
        self.consts = consts or TicTacToe3DConstants()   
        super().__init__(self.consts)                   

        self.config = render_utils.RendererConfig(
            game_dimensions=(210, 160),
            channels=3,
            #downscale=(84, 84)
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)  
        self.LAYER_OFFSETS= jnp.array(self.consts.LAYER_OFFSETS, dtype=jnp.int32)
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
    # 1) Start from background
        raster = self.jr.create_object_raster(self.BACKGROUND)

        board = state.board
        x_mask = self.SHAPE_MASKS["x"]
        o_mask = self.SHAPE_MASKS["o"]
        cursor_mask = self.SHAPE_MASKS["cursor"]

        # 2) DEBUG: draw cursor at ORIGIN to locate grid start
        raster = self.jr.render_at(
            raster,
            self.consts.ORIGIN_X,
            self.consts.ORIGIN_Y,
            cursor_mask,
        )

        # 3) Draw pieces only
        def render_one_cell(idx, r):
            z = idx // 16
            rem = idx - z * 16
            y = rem // 4
            x = rem - y * 4

            v = board[z, y, x]
            px, py = self.cell_to_pixel(x, y, z)

            r = jax.lax.cond(
                v == self.consts.PLAYER_X,
                lambda rr: self.jr.render_at(rr, px, py, x_mask),
                lambda rr: rr,
                r,
            )
            r = jax.lax.cond(
                v == self.consts.PLAYER_O,
                lambda rr: self.jr.render_at(rr, px, py, o_mask),
                lambda rr: rr,
                r,
            )
            return r

        raster = jax.lax.fori_loop(0, 64, render_one_cell, raster)

        # 4) Draw actual cursor
        cx, cy, cz = state.cursor_x, state.cursor_y, state.cursor_z
        px, py = self.cell_to_pixel(cx, cy, cz)

        raster = self.jr.render_at(raster, px, py, cursor_mask)

        return self.jr.render_from_palette(raster, self.PALETTE)
