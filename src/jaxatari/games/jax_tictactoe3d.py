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
import jax.lax 
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
    BOARD_SIZE: int = 4

    # --- Mapping anchor: "upper left of highest level" (z=0, y=0, x=0)
    # These numbers come from the rules you posted.
    ANCHOR_X: int = 58
    ANCHOR_Y: int = 17

    # --- Per-step deltas (from your measurement/rules)
    # Moving within a level:
    STEP_RIGHT_X: int = 8
    STEP_RIGHT_Y: int = 0

    ROW_DOWN_X: int = 5
    ROW_DOWN_Y: int = 10

    # Switching levels (z increasing = going "down" visually)
    LEVEL_DOWN_X: int = -15
    LEVEL_DOWN_Y: int = 16

    # Cursor blink
    BLINK_PERIOD: int = 15

    # Screen dimensions (ARL Atari dims usually 210x160 or 160x210 depending on repo)
    # Your background.npy likely already matches correct size; we’ll read it.
    # Keep these only for spaces/config.
    WIDTH: int = 160
    HEIGHT: int = 210
    NUM_ACTIONS: int = 8
    PIXEL_COORDS: chex.Array = jnp.array([
        # Level 0 (top)
        [
            [[58, 17], [66, 17], [74, 17], [82, 17]],
            [[63, 27], [71, 27], [79, 27], [87, 27]],
            [[68, 37], [76, 37], [84, 37], [92, 37]],
            [[73, 47], [81, 47], [89, 47], [97, 47]],
        ],
        # Level 1
        [
            [[43, 33], [51, 33], [59, 33], [67, 33]],
            [[48, 43], [56, 43], [64, 43], [72, 43]],
            [[53, 53], [61, 53], [69, 53], [77, 53]],
            [[58, 63], [66, 63], [74, 63], [82, 63]],
        ],
        # Level 2
        [
            [[28, 49], [36, 49], [44, 49], [52, 49]],
            [[33, 59], [41, 59], [49, 59], [57, 59]],
            [[38, 69], [46, 69], [54, 69], [62, 69]],
            [[43, 79], [51, 79], [59, 79], [67, 79]],
        ],
        # Level 3 (bottom)
        [
            [[13, 65], [21, 65], [29, 65], [37, 65]],
            [[18, 75], [26, 75], [34, 75], [42, 75]],
            [[23, 85], [31, 85], [39, 85], [47, 85]],
            [[28, 95], [36, 95], [44, 95], [52, 95]],
        ],
    ], dtype=jnp.int32)

    ASSET_CONFIG: tuple = _get_default_asset_config()

class TicTacToe3DState(NamedTuple):
    board: chex.Array  # (4,4,4) uint8
    cursor: chex.Array  # (3,) int32  (x,y,z)
    player: chex.Array  # scalar uint8  (1 or 2)
    done: chex.Array    # scalar bool
    winner: chex.Array  # scalar uint8  (0/1/2)
    step_count: chex.Array  # scalar int32
    
    #Winning lines
class JaxTicTacToe3DEnvironment(JaxEnvironment):
    """
    Minimal environment wrapper around the pure state transition.
    Not focused on RL, but keeps the standard JAXAtari interface.
    """
    def __init__(self, consts: TicTacToe3DConstants | None = None):
        self.consts = consts or TicTacToe3DConstants()
        super().__init__(self.consts)
        self.renderer = TicTacToe3DRenderer(self.consts)
        self.action_set = [
            Action.NOOP,
            Action.LEFT,
            Action.RIGHT,
            Action.UP,
            Action.DOWN,
            Action.FIRE,       # place
            Action.LEFTFIRE,   # level up (z-1)
            Action.RIGHTFIRE,  # level down (z+1)
        ]
        self._action_space = spaces.Discrete(len(self.action_set))
        # Observation: we return the rendered frame
        h, w, _ = self.renderer.BACKGROUND.shape[:2]
        self._image_space = spaces.Box(low=0, high=255, shape=(h, w, 3), dtype=jnp.uint8)
    def action_space(self):
        return self._action_space
    def image_space(self):
        return self._image_space
    def observation_space(self):
        return self._image_space
    def render(self, state: TicTacToe3DState) -> jnp.ndarray:
        return self.renderer.render(state)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey = jax.random.PRNGKey(0)):
        state = TicTacToe3DState(
        board=jnp.zeros((4,4,4), dtype=jnp.uint8), # 4x4x4 board
        cursor=jnp.array([0, 0, 0], dtype=jnp.int32), # in pos x,y,z
        player=jnp.uint8(1),
        done=jnp.bool_(False),
        winner=jnp.uint8(0),
        step_count=jnp.int32(0),
    )
        obs = self.renderer.render(state)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: TicTacToe3DState, action: chex.Array):
        new_state = self._transition(state, action)
        obs = self.renderer.render(new_state)
        reward = jnp.float32(0.0)
        done = new_state.done
        info = {}
        return obs, new_state, reward, done, info


    
    
    def _move_cursor(self, cursor: chex.Array, action: chex.Array) -> chex.Array:
        def clamp(v):
            return jnp.clip(v, 0, self.consts.BOARD_SIZE - 1).astype(jnp.int32)

        return jax.lax.switch(
            action,
            [
            lambda: cursor,  # NOOP
            lambda: jnp.array([clamp(cursor[0] - 1), cursor[1], cursor[2]]),
            lambda: jnp.array([clamp(cursor[0] + 1), cursor[1], cursor[2]]),
            lambda: jnp.array([cursor[0], clamp(cursor[1] - 1), cursor[2]]),
            lambda: jnp.array([cursor[0], clamp(cursor[1] + 1), cursor[2]]),
            lambda: jnp.array([cursor[0], cursor[1], clamp(cursor[2] - 1)]),
            lambda: jnp.array([cursor[0], cursor[1], clamp(cursor[2] + 1)]),
            lambda: cursor,  # PLACE
            ],
        )

    def _place_piece(self, board, cursor, player):
        z, y, x = cursor[2], cursor[1], cursor[0]
        new_board = board.at[z, y, x].set(player)
        winner = _check_winner(new_board)
        done = winner != 0
        next_player = jnp.uint8(3 - player)
        return new_board, next_player, done, winner
    def _check_winner(board):
       return jnp.uint8(0)
   def  transition(self, s: TicTacToe3DState, a: chex.Array) -> TicTacToe3DState:
    # --- Cursor update ---
        new_cursor = self._move_cursor(s.cursor, a)

        # --- Place conditions ---
        cell_empty = s.board[
            s.cursor[2], s.cursor[1], s.cursor[0]
        ] == 0

        do_place = (a == 7) & cell_empty & (~s.done)

        def place_fn(_):
            return self._place_piece(s.board, s.cursor, s.player)

        def no_place_fn(_):
            return s.board, s.player, s.done, s.winner

        board, player, done, winner = jax.lax.cond(
            do_place,
            place_fn,
            no_place_fn,
            operand=None,
        )

        return TicTacToe3DState(
            board=board,
            cursor=new_cursor,
            player=player,
            done=done,
            winner=winner,
            step_count=s.step_count + 1,
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

        final_asset_config = list(self.consts.ASSET_CONFIG)
        sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/tictactoe3d"

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
        px, py = self.consts.PIXEL_COORDS[z, y, x]
        return px - 3, py - 3

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: TicTacToe3DState):
        raster = self.jr.create_object_raster(self.BACKGROUND)

        board = state.board
        cursor = state.cursor
        step_count = state.step_count

        def draw_cell(r, idx):
            z = idx // 16
            rem = idx - z * 16
            y = rem // 4
            x = rem - y * 4

            v = board[z, y, x]
            px, py = self.cell_to_pixel(x, y, z)

            r = jax.lax.cond(v == 1, lambda rr: self.jr.render_at(rr, px, py, self.x_mask), lambda rr: rr, r)
            r = jax.lax.cond(v == 2, lambda rr: self.jr.render_at(rr, px, py, self.o_mask), lambda rr: rr, r)
            return r, None

        raster, _ = jax.lax.scan(draw_cell, raster, jnp.arange(64))

        blink_on = (step_count // self.consts.BLINK_PERIOD) % 2 == 0
        cx, cy, cz = cursor
        cpx, cpy = self.cell_to_pixel(cx, cy, cz)

        raster = jax.lax.cond(
            blink_on & (~state.done),
            lambda r: self.jr.render_at(r, cpx, cpy, self.cursor_mask),
            lambda r: r,
            raster,
        )

        return self.jr.render_from_palette(raster, self.PALETTE)
