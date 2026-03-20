from functools import partial
from typing import NamedTuple, Tuple
import os

import jax
import jax.numpy as jnp
import chex
from jax import lax
from jax import random as jrandom
from flax import struct

import jaxatari.spaces as spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action, ObjectObservation
from jaxatari.rendering import jax_rendering_utils as render_utils
import numpy as np
from jaxatari.renderers import JAXGameRenderer
from jaxatari.modification import AutoDerivedConstants

def _get_default_asset_config() -> tuple:
    """
    Returns the default declarative asset manifest for Tetris.
    Kept immutable (tuple of dicts) to fit NamedTuple defaults.
    """
    return (
        {'name': 'background', 'type': 'background', 'file': 'background.npy'},
        {'name': 'board_overlay', 'type': 'single', 'file': 'board.npy'},
        {'name': 'score_digits', 'type': 'digits', 'pattern': 'score/score_{}.npy'},
        {'name': 'banner_one', 'type': 'single', 'file': 'text_one.npy'},
        {'name': 'banner_two', 'type': 'single', 'file': 'text_two.npy'},
        {'name': 'banner_triple', 'type': 'single', 'file': 'text_triple.npy'},
        {'name': 'banner_tetris', 'type': 'single', 'file': 'text_tetris.npy'},
    )

class TetrisConstants(AutoDerivedConstants):
    # logical grid (Board)
    BOARD_WIDTH: int = struct.field(pytree_node=False, default=10)
    BOARD_HEIGHT: int = struct.field(pytree_node=False, default=22)

    # Env timing – in "env frames" (number of step() calls)
    GRAVITY_FRAMES: int = struct.field(pytree_node=False, default=30)      # auto-fall cadence
    DAS_FRAMES: int = struct.field(pytree_node=False, default=10)          # delayed auto shift for horiz
    ARR_FRAMES: int = struct.field(pytree_node=False, default=3)           # auto-repeat rate for horiz
    ROT_DAS_FRAMES: int = struct.field(pytree_node=False, default=12)      # rotate auto-repeat cadence
    SOFT_PACE_FRAMES: int = struct.field(pytree_node=False, default=4)     # paced soft-drop while held
    SOFT_DROP_SCORE_PER_CELL: int = struct.field(pytree_node=False, default=1)
    LINE_CLEAR_SCORE: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([0, 1, 2, 3, 4], dtype=jnp.int32))  # 0 -> if no row is cleared, 1 -> only 1 row, 2 -> 2 rows, 3-> 3 rows, 4 -> 4 rows, just like the original game

    # Render tiling
    BOARD_X: int = struct.field(pytree_node=False, default=21)       # left margin
    BOARD_Y: int = struct.field(pytree_node=False, default=27)       # top margin
    BOARD_PADDING: int = struct.field(pytree_node=False, default=2)
    CELL_WIDTH: int = struct.field(pytree_node=False, default=3)
    CELL_HEIGHT: int = struct.field(pytree_node=False, default=7)
    DIGIT_X: int = struct.field(pytree_node=False, default=95)
    DIGIT_Y: int = struct.field(pytree_node=False, default=27)
    WINDOW_WIDTH: int = struct.field(pytree_node=False, default=160 * 3)
    WINDOW_HEIGHT: int = struct.field(pytree_node=False, default=210 * 3)
    TETROMINOS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([
        # I
        [
            [[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]],
            [[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]],
        ],
        # O
        [
            [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
        ],
        # T
        [
            [[0, 0, 0, 0], [0, 1, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0]],
            [[0, 1, 0, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [1, 1, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
            [[0, 1, 0, 0], [1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
        ],
        # S
        [
            [[0, 0, 0, 0], [0, 1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]],
            [[0, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]],
            [[0, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
        ],
        # Z
        [
            [[0, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
            [[0, 0, 1, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
            [[0, 0, 1, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
        ],
        # J
        [
            [[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0]],
            [[0, 1, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [1, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
            [[0, 1, 1, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
        ],
        # L
        [
            [[0, 0, 0, 0], [0, 0, 1, 0], [1, 1, 1, 0], [0, 0, 0, 0]],
            [[1, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [1, 1, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
        ],
    ], dtype=jnp.int32))
    RESET: int = struct.field(pytree_node=False, default=Action.DOWNLEFTFIRE)  # keep a reserved reset action

    # Asset config baked into constants (immutable default) for asset overrides
    ASSET_CONFIG: tuple = struct.field(pytree_node=False, default=_get_default_asset_config())

# ======================== State/Obs/Info =================
@struct.dataclass
class TetrisState:
    """ Environment state, Fields are given in arrays to keep everything compatible"""
    board: chex.Array        # (H,W) int32 {0,1}
    piece_type: chex.Array   # () int32 0..6
    pos: chex.Array          # (2,) int32 [y,x]
    rot: chex.Array          # () int32 0..3
    next_piece: chex.Array   # () int32 0..6
    score: chex.Array        # () int32
    game_over: chex.Array    # () bool
    key: chex.Array          # PRNGKey
    tick: chex.Array         # () int32 (env frames)

    # Held-key repeat timers (env-managed DAS/ARR)
    das_timer: chex.Array    # () int32
    arr_timer: chex.Array    # () int32
    move_dir: chex.Array     # () int32 -1/0/+1

    rot_timer: chex.Array    # () int32 (rotation repeat)
    soft_timer: chex.Array   # () int32 (paced soft drop)

    last_action: chex.Array  # () int32 (Atari code)

    banner_timer: chex.Array  # fields for the one, two, triple, tetris sprites
    banner_code: chex.Array

@struct.dataclass
class TetrisObservation:
    board: chex.Array
    active_piece: ObjectObservation
    next_piece: ObjectObservation
    score: chex.Array
    game_over: chex.Array

@struct.dataclass
class TetrisInfo:
    score: chex.Array
    cleared: chex.Array
    game_over: chex.Array

# ======================= Environment =====================

class JaxTetris(JaxEnvironment[TetrisState, TetrisObservation, TetrisInfo, TetrisConstants]):
    # ALE minimal action set: [NOOP, FIRE, RIGHT, LEFT, DOWN]
    ACTION_SET: jnp.ndarray = jnp.array([
        Action.NOOP,
        Action.FIRE,
        Action.RIGHT,
        Action.LEFT,
        Action.DOWN
    ], dtype=jnp.int32)

    def __init__(self, consts: TetrisConstants = None):
        """ Initialize the JaxTetris environment"""

        consts = consts or TetrisConstants()
        super().__init__(consts)
        self.renderer = TetrisRenderer(self.consts)
        self.instant_drop = False

    # ----- Helpers -----
    @partial(jax.jit, static_argnums=0)
    def piece_grid(self, piece_type: chex.Array, rot: chex.Array) -> chex.Array:
        """
        Return the 4x4 grid for a given tetromino type and rotation.
        """
        # Select the correct rotation for the given piece type
        return self.consts.TETROMINOS[piece_type, (rot & 3)]

    @partial(jax.jit, static_argnums=0)
    def spawn_piece(self, key: chex.Array) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        """
        Spawn a new random tetromino, returning its type, position, rotation, and updated PRNG key.
        """
        key, sub = jrandom.split(key)  # Split PRNG key for reproducibility
        p = jrandom.randint(sub, (), 0, 7, dtype=jnp.int32)  # Random piece type
        pos = jnp.array([0, 3], dtype=jnp.int32)  # Spawn at top center
        rot = jnp.int32(0)  # Initial rotation
        return p, pos, rot, key

    @partial(jax.jit, static_argnums=0)
    def check_collision(self, board: chex.Array, grid4: chex.Array, pos: chex.Array) -> chex.Array:
        """
        Check if the tetromino at the given position collides with the board or is out of bounds.

        blocks touching boundaries count as a collision
        """
        H = jnp.int32(self.consts.BOARD_HEIGHT)
        W = jnp.int32(self.consts.BOARD_WIDTH)

        ys = jnp.arange(4, dtype=jnp.int32)
        xs = jnp.arange(4, dtype=jnp.int32)
        yy, xx = jnp.meshgrid(ys, xs, indexing="ij")  # (4,4)
        on = (grid4 == 1)
        py = pos[0] + yy
        px = pos[1] + xx

        inb_side = (px >= 0) & (px < W) & (py < H)
        readable = inb_side & (py >= 0)
        pyc = jnp.clip(py, 0, H - 1)
        pxc = jnp.clip(px, 0, W - 1)
        occ = jnp.where(readable, board[pyc, pxc] == 1, False)
        side_or_bottom_oob = on & (~inb_side) & (py >= 0)
        coll = on & (occ | side_or_bottom_oob)
        return jnp.any(coll)

    @partial(jax.jit, static_argnums=0)
    def lock_piece(self, board: chex.Array, grid4: chex.Array, pos: chex.Array) -> chex.Array:
        """
        Lock the current tetromino onto the board at the given position.
        """
        H = self.consts.BOARD_HEIGHT  # avoid tracer shapes
        W = self.consts.BOARD_WIDTH

        # Build an overlay mask over the whole board where piece occupies
        by = jnp.arange(H, dtype=jnp.int32)[:, None]
        bx = jnp.arange(W, dtype=jnp.int32)[None, :]

        # Map board coords to piece-local coords and test membership
        rel_y = by - pos[0]
        rel_x = bx - pos[1]
        within_piece = (rel_y >= 0) & (rel_y < 4) & (rel_x >= 0) & (rel_x < 4)
        rel_yc = jnp.clip(rel_y, 0, 3)
        rel_xc = jnp.clip(rel_x, 0, 3)
        cover = within_piece & (grid4[rel_yc, rel_xc] == 1)
        # Only lock for board cells with non-negative piece rows (top negative rows are ignored)
        cover = cover & (rel_y >= 0)
        return jnp.where(cover, jnp.int32(1), board)

    @partial(jax.jit, static_argnums=0)
    def clear_lines(self, board: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """
        Clear all full lines from the board.
        Returns the new board and the number of lines cleared.
        """
        full = jnp.all(board == 1, axis=1)  # (H,)
        H = self.consts.BOARD_HEIGHT  # python int for shape ops

        idx = jnp.arange(H, dtype=jnp.int32)
        keep_key = (~full).astype(jnp.int32)
        # Composite key makes all full rows (0) come first, then non-full (1), each in original order
        key = keep_key * H + idx
        perm = jnp.argsort(key)
        board_sorted = board[perm]

        cleared = jnp.sum(full).astype(jnp.int32)
        # Zero out the top `cleared` rows
        row_ids = jnp.arange(H, dtype=jnp.int32)
        is_top_cleared = row_ids < cleared
        nb = jnp.where(is_top_cleared[:, None], jnp.int32(0), board_sorted)
        return nb, cleared

    @partial(jax.jit, static_argnums=0)
    def try_rotate(self, board: chex.Array, piece_type: chex.Array, pos: chex.Array, rot: chex.Array):
        """
        Try to rotate the tetromino (clockwise) in place (no wall kick).
        Returns the new position and rotation if successful, otherwise the original.
        """
        new_rot = (rot + 1) & 3
        rotated_grid = self.piece_grid(piece_type, new_rot)

        # I piece symmetric tiny wall-kick: try [(0,0), (0,+1), (0,-1)]
        is_I = (piece_type == jnp.int32(0))

        def rotate_I(_):
            # Detect second from right for current I piece
            current_grid = self.piece_grid(piece_type, rot)
            xs = jnp.arange(4, dtype=jnp.int32)
            ys = jnp.arange(4, dtype=jnp.int32)
            yy, xx = jnp.meshgrid(ys, xs, indexing="ij")
            on = (current_grid == 1)
            world_x = pos[1] + xx
            W = jnp.int32(self.consts.BOARD_WIDTH)
            max_px = jnp.max(jnp.where(on, world_x, jnp.int32(-1)))
            # Forbid when at rightmost or second-from-right
            # Apply this block ONLY when the current I is vertical, so horizontal behavior is unaffected
            is_vertical_now = ((rot & jnp.int32(1)) == jnp.int32(1))
            is_right_blocked = is_vertical_now & (max_px >= (W - jnp.int32(2)))
            # +forbid when touching leftmost column while vertical
            min_px = jnp.min(jnp.where(on, world_x, jnp.int32(1 << 30)))
            is_left_blocked = is_vertical_now & (min_px == jnp.int32(0))

            pos0 = pos
            pos1 = pos + jnp.array([jnp.int32(0), jnp.int32(1)], dtype=jnp.int32)
            pos2 = pos + jnp.array([jnp.int32(0), jnp.int32(-1)], dtype=jnp.int32)

            feas0 = ~self.check_collision(board, rotated_grid, pos0)
            feas1 = ~self.check_collision(board, rotated_grid, pos1)
            feas2 = ~self.check_collision(board, rotated_grid, pos2)

            # Pick first feasible in order 0,1,2
            feas = jnp.stack([feas0, feas1, feas2])
            idxs = jnp.array([0, 1, 2], dtype=jnp.int32)
            sentinel = jnp.int32(3)
            keys = jnp.where(feas, idxs, sentinel)
            best = jnp.argmin(keys)
            success = jnp.any(feas) & (~is_right_blocked) & (~is_left_blocked)

            pos_best = jnp.where(best == 0, pos0, jnp.where(best == 1, pos1, pos2))
            pos_out = jnp.where(success, pos_best, pos)
            rot_out = jnp.where(success, new_rot, rot)
            return pos_out, rot_out

        def rotate_default(_):
            can_in_place = ~self.check_collision(board, rotated_grid, pos)
            pos_out = jnp.where(can_in_place, pos, pos)
            rot_out = jnp.where(can_in_place, new_rot, rot)
            return pos_out, rot_out

        return lax.cond(is_I, rotate_I, rotate_default, operand=None)

    @partial(jax.jit, static_argnums=0)
    def _lock_spawn(self, s: TetrisState, grid: chex.Array, tick_next: chex.Array, soft_points: chex.Array):
        """
        Lock the current piece, clear lines, update score, and spawn the next piece.
        Returns the new state, reward, game over flag, and info.
        """
        board_locked = self.lock_piece(s.board, grid, s.pos)  # Lock piece
        board_cleared, lines_cleared = self.clear_lines(board_locked)  # Clear lines
        line_clear_score = jnp.array(self.consts.LINE_CLEAR_SCORE, dtype=jnp.int32)[jnp.clip(lines_cleared, 0, 4)]  # Score for lines
        total_score = s.score + line_clear_score + soft_points  # Update score

        # Banner logic for line clear
        
        show_frames_by_lines = jnp.array([0,60,60,60,60], dtype = jnp.int32)
        new_banner_timer = show_frames_by_lines[lines_cleared]
        new_banner_code = lines_cleared
        banner_timer = jnp.where(lines_cleared > 0, new_banner_timer, s.banner_timer)
        banner_code = jnp.where(lines_cleared > 0, new_banner_code, s.banner_code)

        current_piece = s.next_piece
        pos, rot = jnp.array([0, 3], jnp.int32), jnp.int32(0)
        new_piece_grid = self.piece_grid(current_piece, rot)
        game_over = self.check_collision(board_cleared, new_piece_grid, pos)  # Check if game over
        next_piece, _, _, key2 = self.spawn_piece(s.key)

        new_state = s.replace(board=board_cleared, piece_type=current_piece, pos=pos, rot=rot,
                        next_piece=next_piece, score=total_score, game_over=game_over, key=key2, tick=tick_next,
                        banner_timer=banner_timer,  # new row for banners
                        banner_code=banner_code
                        )

        reward = (lines_cleared > 0).astype(jnp.float32) * line_clear_score.astype(jnp.float32)
        info = TetrisInfo(score=new_state.score, cleared=lines_cleared, game_over=game_over)
        return new_state, reward, game_over, info

    # ----- Spaces -----
    def action_space(self) -> spaces.Discrete:
        """
        Return the action space for the environment.
        """
        return spaces.Discrete(len(self.ACTION_SET))

    def observation_space(self) -> spaces.Dict:
        c = self.consts
        # Logical grid dimensions
        h = int(c.BOARD_HEIGHT)
        w = int(c.BOARD_WIDTH)
        grid_size = (h, w)
        
        single_obj = spaces.get_object_space(n=None, screen_size=grid_size)
        
        return spaces.Dict({
            "board": spaces.Box(low=0, high=1, shape=(h, w), dtype=jnp.int32),
            "active_piece": single_obj,
            "next_piece": single_obj, # Represents type via visual_id
            "score": spaces.Box(low=0, high=999999, shape=(), dtype=jnp.int32),
            "game_over": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
        })

    def image_space(self) -> spaces.Box:
        """
        Return the image space for rendering.
        """
        return spaces.Box(low=0, high=255, shape=(210, 160, 3), dtype=jnp.uint8)

    # ----- Public API -----
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jrandom.PRNGKey = None) -> Tuple[TetrisObservation, TetrisState]:
        """
        Reset the environment and return the initial observation and state.
        - Clear board
        """
        key = jrandom.PRNGKey(0) if key is None else key
        board = jnp.zeros((self.consts.BOARD_HEIGHT, self.consts.BOARD_WIDTH), dtype=jnp.int32)
        cur, pos, rot, key = self.spawn_piece(key)
        nxt, _, _, key = self.spawn_piece(key)
        state = TetrisState(
            board=board, piece_type=cur, pos=pos, rot=rot, next_piece=nxt,
            score=jnp.int32(0), game_over=jnp.bool_(False), key=key,
            tick=jnp.int32(0),
            das_timer=jnp.int32(0), arr_timer=jnp.int32(0), move_dir=jnp.int32(0),
            rot_timer=jnp.int32(0), soft_timer=jnp.int32(0),
            last_action=jnp.int32(Action.NOOP),
            banner_timer =jnp.int32(0),
            banner_code = jnp.int32(0)
        )
        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def step(self, state: TetrisState, action: chex.Array) -> Tuple[
        TetrisObservation, TetrisState, float, bool, TetrisInfo]:
        """
        Take one step in the environment.
        Returns the new observation, state, reward, done flag, and info.
        """
        previous_state = state
        # Translate agent action (0,1,2,3,4) to ALE action
        atari_action = jnp.take(self.ACTION_SET, action)

        # Decode inputs
        is_left = (atari_action == Action.LEFT) | (atari_action == Action.UPLEFT) | (atari_action == Action.DOWNLEFT)
        is_right = (atari_action == Action.RIGHT) | (atari_action == Action.UPRIGHT) | (atari_action == Action.DOWNRIGHT)
        is_up = (atari_action == Action.UP) | (atari_action == Action.UPLEFT) | (atari_action == Action.UPRIGHT)
        is_down = (atari_action == Action.DOWN) | (atari_action == Action.DOWNLEFT) | (atari_action == Action.DOWNRIGHT)
        is_fire = (atari_action == Action.FIRE) | (atari_action == Action.DOWNFIRE) | (atari_action == Action.UPFIRE) \
                  | (atari_action == Action.LEFTFIRE) | (atari_action == Action.RIGHTFIRE) \
                  | (atari_action == Action.UPLEFTFIRE) | (atari_action == Action.UPRIGHTFIRE) \
                  | (atari_action == Action.DOWNLEFTFIRE) | (atari_action == Action.DOWNRIGHTFIRE)

        # Allow only single-key actions: if multiple logical keys pressed, treat as NOOP
        pressed_count = (
            is_left.astype(jnp.int32)
            + is_right.astype(jnp.int32)
            + is_up.astype(jnp.int32)
            + is_down.astype(jnp.int32)
            + is_fire.astype(jnp.int32)
        )
        exactly_one = (pressed_count == 1)
        is_left = is_left & exactly_one
        is_right = is_right & exactly_one
        is_up = is_up & exactly_one
        is_down = is_down & exactly_one
        is_fire = is_fire & exactly_one

        do_reset = (atari_action == self.consts.RESET)

        tick_next = state.tick + jnp.int32(1)
        gravity_drop = (tick_next % jnp.int32(self.consts.GRAVITY_FRAMES) == 0).astype(jnp.int32)

        # ---- Held-key timers (horizontal) ----
        new_move_dir = jnp.where(is_left, -1, jnp.where(is_right, 1, state.move_dir * 0))
        das = jnp.where(new_move_dir != 0,
                        jnp.where(state.move_dir == 0, jnp.int32(self.consts.DAS_FRAMES),
                                  jnp.maximum(0, state.das_timer - 1)),
                        jnp.int32(0))
        arr = jnp.where(new_move_dir != 0,
                        jnp.where(das == 0, jnp.maximum(0, state.arr_timer - 1),
                                  jnp.int32(self.consts.ARR_FRAMES)),
                        jnp.int32(0))
        do_move_now = (new_move_dir != 0) & ((state.move_dir == 0) | ((das == 0) & (arr == 0)))

        # ---- Rotation repeat gating ----
        # Allow FIRE to rotate only when instant_drop is disabled
        allow_fire_rotate = jnp.logical_not(jnp.bool_(self.instant_drop))
        is_up_single = (atari_action == Action.UP)
        is_fire_single = (atari_action == Action.FIRE)
        rotate_pressed = is_up_single | (allow_fire_rotate & is_fire_single)
        last_was_rotate = ((state.last_action == Action.UP) |
                           (allow_fire_rotate & (state.last_action == Action.FIRE)))

        # countdown while held; reset to 0 when neither is held
        rot_timer = jnp.where(rotate_pressed, jnp.maximum(0, state.rot_timer - 1), jnp.int32(0))

        # edge-trigger or repeat
        do_rotate_now = rotate_pressed & ((~last_was_rotate) | (rot_timer == 0))

        # ---- Soft drop timer ----
        soft_timer = jnp.where(is_down, jnp.maximum(0, state.soft_timer - 1), jnp.int32(0))
        do_soft_now = is_down & (soft_timer == 0)

        # ---- Hard drop (SPACE) edge when instant_drop=True ----
        do_hard_now = (jnp.bool_(self.instant_drop) & is_fire & (state.last_action != Action.FIRE))

        # ---- Apply rotate ----
        pos_r, rot_r = lax.cond(
            do_rotate_now,
            lambda _: self.try_rotate(state.board, state.piece_type, state.pos, state.rot),
            lambda _: (state.pos, state.rot),
            operand=None
        )
        state = state.replace(pos=pos_r, rot=rot_r)

        # ---- Horizontal move ----
        grid = self.piece_grid(state.piece_type, state.rot)
        pos_h = state.pos + jnp.array([0, jnp.int32(jnp.where(do_move_now, new_move_dir, 0))], dtype=jnp.int32)
        coll_h = self.check_collision(state.board, grid, pos_h)
        state = lax.cond(coll_h, lambda s: s, lambda s: s.replace(pos=pos_h), state)

        # ---- Vertical movement / lock ----
        def do_hard(s: TetrisState):
            g = self.piece_grid(s.piece_type, s.rot)

            def cond_fun(t):
                pnext = t.pos + jnp.array([1, 0], jnp.int32)
                return ~self.check_collision(t.board, g, pnext)

            def body_fun(t):
                return t.replace(pos=t.pos + jnp.array([1, 0], jnp.int32))

            s2 = lax.while_loop(cond_fun, body_fun, s)
            return self._lock_spawn(s2, g, tick_next, soft_points=jnp.int32(0))

        def do_soft_or_gravity(s: TetrisState):
            dy = jnp.clip(do_soft_now.astype(jnp.int32) | gravity_drop, 0, 1)
            pos_v = s.pos + jnp.array([dy, 0], dtype=jnp.int32)
            coll_v = self.check_collision(s.board, grid, pos_v)
            return lax.cond(
                coll_v,
                lambda ss: self._lock_spawn(ss, grid, tick_next, soft_points=jnp.int32(0)),
                lambda ss: (ss.replace(pos=pos_v, tick=tick_next),
                            jnp.float32(0.0), jnp.bool_(False),
                            TetrisInfo(score=ss.score, cleared=jnp.int32(0), game_over=ss.game_over)),
                s
            )

        def do_env_reset(_):
            obs, st0 = self.reset(state.key)
            return st0, jnp.float32(0.0), jnp.bool_(False), TetrisInfo(score=st0.score, cleared=jnp.int32(0),
                                                                       game_over=st0.game_over)

        state, _reward, done, info = lax.cond(
            do_reset, do_env_reset,
            lambda _: lax.cond(do_hard_now, do_hard, do_soft_or_gravity, state),
            operand=None
        )

        # If game over, auto-reset
        def after_over(ss):
            obs2, st2 = self.reset(ss.key)
            return st2, jnp.float32(0.0), jnp.bool_(False), TetrisInfo(score=st2.score, cleared=jnp.int32(0),
                                                                       game_over=jnp.bool_(False))

        state, reward, done, info = lax.cond(state.game_over, after_over, lambda s: (s, _reward, done, info), state)

        # ---- Timers & book-keeping ----
        next_banner_timer = jnp.maximum(0, state.banner_timer - 1)

        state = state.replace(
            das_timer=das,
            arr_timer=arr,
            move_dir=new_move_dir,
            rot_timer=jnp.where(do_rotate_now, jnp.int32(self.consts.ROT_DAS_FRAMES), rot_timer),
            soft_timer=jnp.where(do_soft_now, jnp.int32(self.consts.SOFT_PACE_FRAMES), soft_timer),
            last_action=atari_action,
            banner_timer=next_banner_timer,
            banner_code=jnp.where(next_banner_timer == 0, jnp.int32(0), state.banner_code)
        )

        # ---- Outputs ----
        obs = self._get_observation(state)
        reward = self._get_reward(previous_state, state)
        done = self._get_done(state)
        info = self._get_info(state)
        return obs, state, reward, done, info

    # ----- Helpers used inside step -----
    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: TetrisState) -> TetrisObservation:
        c = self.consts
        w, h = int(c.BOARD_WIDTH), int(c.BOARD_HEIGHT)
        
        # --- Active Piece ---
        # Piece position is (row, col) = (y, x) in grid coords
        # Rotation 0..3 -> 0, 90, 180, 270
        rot_deg = (state.rot * 90.0).astype(jnp.float32)
        
        active_piece = ObjectObservation.create(
            x=jnp.clip(state.pos[1], 0, w), # pos[1] is x/col
            y=jnp.clip(state.pos[0], 0, h), # pos[0] is y/row
            width=jnp.array(4, dtype=jnp.int32), # All pieces are 4x4 grids
            height=jnp.array(4, dtype=jnp.int32),
            active=jnp.array(1, dtype=jnp.int32),
            visual_id=state.piece_type, # Type determines color/shape
            orientation=jnp.array(rot_deg, dtype=jnp.float32)
        )

        # --- Next Piece ---
        # Not on board, so position 0,0 inactive or just metadata?
        # Standardize as an object with valid ID but perhaps off-board coordinates 
        # or just visually distinct. Let's keep it 'active' for metadata access.
        next_piece = ObjectObservation.create(
            x=jnp.array(0, dtype=jnp.int32),
            y=jnp.array(0, dtype=jnp.int32),
            width=jnp.array(4, dtype=jnp.int32),
            height=jnp.array(4, dtype=jnp.int32),
            active=jnp.array(1, dtype=jnp.int32),
            visual_id=state.next_piece
        )

        return TetrisObservation(
            board=state.board,
            active_piece=active_piece,
            next_piece=next_piece,
            score=state.score,
            game_over=state.game_over.astype(jnp.int32)
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: TetrisState) -> TetrisInfo:
        """
        Extract info (score, cleared lines, game over) from the state.
        """
        return TetrisInfo(score=state.score, cleared=jnp.int32(0), game_over=state.game_over)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: TetrisState, state: TetrisState) -> float:
        """
        Compute the reward for the transition from previous_state to state.
        Only nonzero when lines are cleared.
        """
        cleared = state.score - previous_state.score
        return jnp.where(cleared > 0, cleared.astype(jnp.float32), jnp.float32(0.0))

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: TetrisState) -> bool:
        """
        Check if the game is over.
        """
        return state.game_over

    def render(self, state: TetrisState) -> jnp.ndarray:
        """
        Render the current game state to an image.
        """
        return self.renderer.render(state)

# ======================= Renderer (pure JAX) =============
class TetrisRenderer(JAXGameRenderer):
    def __init__(self, consts: TetrisConstants = None, config: render_utils.RendererConfig = None):
        self.consts = consts or TetrisConstants()
        super().__init__(self.consts)
        
        # Use injected config if provided, else default
        if config is None:
            self.config = render_utils.RendererConfig(
                game_dimensions=(210, 160),
                channels=3,
                downscale=None
            )
        else:
            self.config = config
        self.jr = render_utils.JaxRenderingUtils(self.config)

        # 1. Start from (possibly modded) asset config provided via constants
        final_asset_config = list(self.consts.ASSET_CONFIG)
        
        # 2. Create procedural assets using modded constants
        # Procedurally generate 1x1 pixel sprites for each of the 22 row colors.
        # This ensures they are all included in the final color palette.
        sprite_path = os.path.join(render_utils.get_base_sprite_dir(), "tetris")
        for i in range(22):
            # The file is loaded here to get the color, then converted to a procedural sprite.
            color_rgba = self.jr.loadFrame(
                f"{sprite_path}/height_colors/h_{i}.npy"
            )[0, 0] # Get the color from the top-left pixel
            final_asset_config.append({
                'name': f'row_color_{i}',
                'type': 'procedural',
                'data': color_rgba.reshape(1, 1, 4)
            })
        
        # 3. Load all assets, create palette, and generate ID masks
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS,
        ) = self.jr.load_and_setup_assets(final_asset_config, sprite_path)

        # 4. Precompute the color map for the board grid
        self.BOARD_COLOR_MAP = self._precompute_board_color_map()

    def _precompute_board_color_map(self) -> jnp.ndarray:
        """Creates a lookup table mapping a row's object ID to its color ID."""
        # Object ID 0 is transparent/background.
        # Object ID 1 will map to the color of row_color_0, ID 2 to row_color_1, and so on.
        color_ids = [self.jr.TRANSPARENT_ID]
        for i in range(22):
            # Load the color, convert to a tuple, and look up its ID in the generated map.
            color_rgba = self.jr.loadFrame(
                f"{os.path.join(render_utils.get_base_sprite_dir(), 'tetris')}/height_colors/h_{i}.npy"
            )[0, 0]
            rgb_tuple = tuple(np.array(color_rgba[:3]))
            color_ids.append(self.COLOR_TO_ID[rgb_tuple])
        return jnp.array(color_ids)

    @partial(jax.jit, static_argnums=(0,))
    def get_piece_shape(self, piece_idx: chex.Array, rotation_idx: chex.Array) -> chex.Array:
        return self.consts.TETROMINOS[piece_idx, rotation_idx]

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: TetrisState):
        # --- 1. Initialize Raster & Static Elements ---
        raster = self.jr.create_object_raster(self.BACKGROUND)
        raster = self.jr.render_at(
            raster, self.consts.BOARD_X, self.consts.BOARD_Y, self.SHAPE_MASKS['board_overlay']
        )

        # Get the shape of the current piece
        piece_shape = self.get_piece_shape(state.piece_type, state.rot)
        H_board, W_board = state.board.shape

        # Create coordinate grids for the entire board
        board_rows, board_cols = jnp.mgrid[:H_board, :W_board]

        # For each board cell, calculate its coordinate relative to the piece's top-left corner
        relative_rows = board_rows - state.pos[0]
        relative_cols = board_cols - state.pos[1]

        # Create a mask for which board cells are covered by the piece's 4x4 grid
        in_piece_grid = (relative_rows >= 0) & (relative_rows < 4) & \
                        (relative_cols >= 0) & (relative_cols < 4)

        # Safely look up the piece shape for the covered cells
        safe_rel_rows = jnp.clip(relative_rows, 0, 3)
        safe_rel_cols = jnp.clip(relative_cols, 0, 3)
        piece_values = piece_shape[safe_rel_rows, safe_rel_cols]

        # Create the final "piece layer" by applying the mask
        piece_layer = jnp.where(in_piece_grid, piece_values, 0)
        
        # Combine the static board with the new piece layer
        combined_board_state = jnp.logical_or(state.board, piece_layer).astype(jnp.int32)
        
        # Create a grid of object IDs based on the row index (from bottom up)
        row_indices = jnp.arange(H_board, 0, -1)[:, None] # Shape (H, 1) -> [[22], [21], ..., [1]]
        object_id_grid = row_indices * combined_board_state
        
        # Render the entire grid in one go
        raster = self.jr.render_grid_inverse(
            raster,
            grid_state=object_id_grid,
            grid_origin=(
                self.consts.BOARD_X + self.consts.BOARD_PADDING,
                self.consts.BOARD_Y
            ),
            cell_size=(self.consts.CELL_WIDTH, self.consts.CELL_HEIGHT),
            color_map=self.BOARD_COLOR_MAP,
            cell_padding=(1, 1) # 1 pixel padding between cells
        )

        # --- 3. Render UI (Score and Banners) ---
        score_digits = self.jr.int_to_digits(state.score, max_digits=4)
        raster = self.jr.render_label(
            raster, 95, 27, score_digits, self.SHAPE_MASKS['score_digits'], spacing=16, max_digits=4
        )

        # Draw the ONE/TWO/THREE/TETRIS banner if active
        def draw_banner(r_):
            # Use switch over equal-shaped outputs by returning the raster
            def drw(mask_key):
                return lambda r: self.jr.render_at(r, 95, 122, self.SHAPE_MASKS[mask_key])

            branches = [
                drw('banner_one'),
                drw('banner_two'),
                drw('banner_triple'),
                drw('banner_tetris'),
            ]
            idx = jnp.clip(state.banner_code - 1, 0, 3)
            return jax.lax.switch(idx, branches, r_)

        raster = jax.lax.cond(
            (state.banner_timer > 0) & (state.banner_code > 0),
            draw_banner,
            lambda r_: r_,
            raster
        )

        # --- 4. Finalize Frame ---
        return self.jr.render_from_palette(raster, self.PALETTE)