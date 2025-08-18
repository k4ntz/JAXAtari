from functools import partial
from typing import NamedTuple, Tuple
import os

import jax
import jax.numpy as jnp
import chex
from jax import lax
from jax import random as jrandom

import jaxatari.spaces as spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.rendering import jax_rendering_utils as jr
from jaxatari.renderers import JAXGameRenderer

class TetrisConstants(NamedTuple):
    BOARD_WIDTH: int = 10
    BOARD_HEIGHT: int = 22

    # Env timing – in "env frames" (number of step() calls)
    GRAVITY_FRAMES: int = 30      # auto-fall cadence
    DAS_FRAMES: int = 10          # delayed auto shift for horiz
    ARR_FRAMES: int = 3           # auto-repeat rate for horiz
    ROT_DAS_FRAMES: int = 12      # rotate auto-repeat cadence
    SOFT_PACE_FRAMES: int = 4     # paced soft-drop while held
    SOFT_DROP_SCORE_PER_CELL = 1
    LINE_CLEAR_SCORE = (0, 1, 2, 3, 4)  # 0 -> if no row is cleared, 1 -> only 1 row, 2 -> 2 rows, 3-> 3 rows, 4 -> 4 rows, just like the original game

    # Render tiling
    BOARD_X: int = 21  # left margin
    BOARD_Y: int = 27  # top margin
    BOARD_PADDING: int = 2
    CELL_WIDTH: int = 3
    CELL_HEIGHT: int = 7
    DIGIT_X: int = 95
    DIGIT_Y: int = 27
    WINDOW_WIDTH: int = 160 * 3
    WINDOW_HEIGHT: int = 210 * 3
    TETROMINOS = jnp.array([
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
    ], dtype=jnp.int32)
    RESET: int = Action.DOWNLEFTFIRE  # keep a reserved reset action

# ======================== State/Obs/Info =================
class TetrisState(NamedTuple):
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

class TetrisObservation(NamedTuple):
    board: chex.Array
    piece_type: chex.Array
    pos: chex.Array
    rot: chex.Array
    next_piece: chex.Array

class TetrisInfo(NamedTuple):
    score: chex.Array
    cleared: chex.Array
    game_over: chex.Array

# ======================= Environment =====================

class JaxTetris(JaxEnvironment[TetrisState, TetrisObservation, TetrisInfo, TetrisConstants]):
    def __init__(self, consts: TetrisConstants = None, reward_funcs: list[callable]=None, instant_drop: bool = False):
        consts = consts or TetrisConstants()
        super().__init__(consts)
        self.renderer = TetrisRenderer(self.consts)
        self.instant_drop = instant_drop
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)

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
        Try to rotate the tetromino in place (no wall kick).
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
            # Forbid when at rightmost or second-from-right; allow from third-from-right and leftwards
            # Apply this block ONLY when the current I is vertical, so horizontal behavior is unaffected
            is_vertical_now = ((rot & jnp.int32(1)) == jnp.int32(1))
            is_right_blocked = is_vertical_now & (max_px >= (W - jnp.int32(2)))

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
            success = jnp.any(feas) & (~is_right_blocked)

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

        new_state = s._replace(board=board_cleared, piece_type=current_piece, pos=pos, rot=rot,
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
        return spaces.Discrete(5)

    def observation_space(self) -> spaces.Dict:
        """
        Return the observation space for the environment.
        """
        return spaces.Dict({
            "board": spaces.Box(low=0, high=1, shape=(self.consts.BOARD_HEIGHT, self.consts.BOARD_WIDTH), dtype=jnp.int32),
            "piece_type": spaces.Box(low=0, high=6, shape=(), dtype=jnp.int32),
            "pos": spaces.Box(low=0, high=max(self.consts.BOARD_HEIGHT, self.consts.BOARD_WIDTH), shape=(2,), dtype=jnp.int32),
            "rot": spaces.Box(low=0, high=3, shape=(), dtype=jnp.int32),
            "next_piece": spaces.Box(low=0, high=6, shape=(), dtype=jnp.int32),
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
        a = action.astype(jnp.int32)

        # Decode inputs
        is_left = (a == Action.LEFT) | (a == Action.UPLEFT) | (a == Action.DOWNLEFT)
        is_right = (a == Action.RIGHT) | (a == Action.UPRIGHT) | (a == Action.DOWNRIGHT)
        is_up = (a == Action.UP) | (a == Action.UPLEFT) | (a == Action.UPRIGHT)
        is_down = (a == Action.DOWN) | (a == Action.DOWNLEFT) | (a == Action.DOWNRIGHT)
        is_fire = (a == Action.FIRE) | (a == Action.DOWNFIRE) | (a == Action.UPFIRE) \
                  | (a == Action.LEFTFIRE) | (a == Action.RIGHTFIRE) \
                  | (a == Action.UPLEFTFIRE) | (a == Action.UPRIGHTFIRE) \
                  | (a == Action.DOWNLEFTFIRE) | (a == Action.DOWNRIGHTFIRE)

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

        do_reset = (a == self.consts.RESET)

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
        is_up_single = (a == Action.UP)
        is_fire_single = (a == Action.FIRE)
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
        state = state._replace(pos=pos_r, rot=rot_r)

        # ---- Horizontal move ----
        grid = self.piece_grid(state.piece_type, state.rot)
        pos_h = state.pos + jnp.array([0, jnp.int32(jnp.where(do_move_now, new_move_dir, 0))], dtype=jnp.int32)
        coll_h = self.check_collision(state.board, grid, pos_h)
        state = lax.cond(coll_h, lambda s: s, lambda s: s._replace(pos=pos_h), state)

        # ---- Vertical movement / lock ----
        def do_hard(s: TetrisState):
            g = self.piece_grid(s.piece_type, s.rot)

            def cond_fun(t):
                pnext = t.pos + jnp.array([1, 0], jnp.int32)
                return ~self.check_collision(t.board, g, pnext)

            def body_fun(t):
                return t._replace(pos=t.pos + jnp.array([1, 0], jnp.int32))

            s2 = lax.while_loop(cond_fun, body_fun, s)
            return self._lock_spawn(s2, g, tick_next, soft_points=jnp.int32(0))

        def do_soft_or_gravity(s: TetrisState):
            dy = jnp.clip(do_soft_now.astype(jnp.int32) | gravity_drop, 0, 1)
            pos_v = s.pos + jnp.array([dy, 0], dtype=jnp.int32)
            coll_v = self.check_collision(s.board, grid, pos_v)
            return lax.cond(
                coll_v,
                lambda ss: self._lock_spawn(ss, grid, tick_next, soft_points=jnp.int32(0)),
                lambda ss: (ss._replace(pos=pos_v, tick=tick_next),
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

        state = state._replace(
            das_timer=das,
            arr_timer=arr,
            move_dir=new_move_dir,
            rot_timer=jnp.where(do_rotate_now, jnp.int32(self.consts.ROT_DAS_FRAMES), rot_timer),
            soft_timer=jnp.where(do_soft_now, jnp.int32(self.consts.SOFT_PACE_FRAMES), soft_timer),
            last_action=a,
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
        """
        Convert the state to an observation.
        """
        return TetrisObservation(
            board=state.board,
            piece_type=state.piece_type,
            pos=state.pos,
            rot=state.rot,
            next_piece=state.next_piece,
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

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: TetrisObservation) -> chex.Array:
        """
        Flatten the observation to a 1D array (for vectorization/testing).
        """
        return jnp.concatenate([obs.board.flatten(), obs.piece_type[None], obs.pos, obs.rot[None], obs.next_piece[None]]).astype(jnp.int32)

    def render(self, state: TetrisState) -> jnp.ndarray:
        """
        Render the current game state to an image.
        """
        return self.renderer.render(state)

# ======================= Renderer (pure JAX) =============
class TetrisRenderer(JAXGameRenderer):
    def __init__(self, consts: TetrisConstants = None):
        super().__init__()
        self.consts = consts or TetrisConstants()
        (
            self.SPRITE_BG,
            self.SPRITE_BOARD,
            self.SCORE_DIGIT_SPRITES,
            self.SPRITE_ROW_COLORS,
            self.SPRITE_ONE, #new banner for one
            self.SPRITE_TWO, #new banner for two
            self.SPRITE_THREE, #new banner for triple
            self.SPRITE_TETRIS, #new banner for tetris
        ) = self.load_sprites()

        self.N_COLOR_ROWS = int(self.SPRITE_ROW_COLORS.shape[0])

    def load_sprites(self):
        """Load all sprites required for Tetris rendering."""
        MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

        # Load background sprites
        bg = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/tetris/background.npy"))
        board = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/tetris/board.npy"))

        # Convert all sprites to the expected format (add frame dimension)
        SPRITE_BG = jnp.expand_dims(bg, axis=0)
        SPRITE_BOARD = jnp.expand_dims(board, axis=0)

        # Load digits for scores
        SCORE_DIGIT_SPRITES = jr.load_and_pad_digits(
            os.path.join(MODULE_DIR, "sprites/tetris/score/score_{}.npy"),
            num_chars=10,
        )

        # Colors for tetris pieces on the board
        row_squares = []
        for i in range(22):  # 22 rows
            sprite = jr.loadFrame(os.path.join(MODULE_DIR, f"sprites/tetris/height_colors/h_{i}.npy"))
            row_squares.append(sprite)

        SPRITE_ROW_COLORS = jnp.stack(row_squares, axis=0)  # Shape: (22, H, W, 4)

        # Sprites for banners: when a row is cleared a message
        # on the right side of the board is shown
        # Load banner sprites shown when rows are cleared.
        # Each .npy file stores an RGBA image (H, W, 4) for ONE/TWO/TRIPLE/TETRIS
        one = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/tetris/text_one.npy"))
        two = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/tetris/text_two.npy"))
        three = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/tetris/text_triple.npy"))
        tetris = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/tetris/text_tetris.npy"))


        # uint8 -> everything into 8 bits. It also works without the conversion so idk
        # SPRITE_ONE = jnp.expand_dims(one.astype(jnp.uint8), axis=0)

        # expand dimensions so that shape (H, W, 4) becomes (1, H, W, 4),
        # making them compatible with sprite-handling code that expects a
        # sequence of frames: (NumFrames, Height, Width, Channels).
        SPRITE_ONE = jnp.expand_dims(jnp.array(one, dtype=jnp.uint8), axis=0)
        SPRITE_TWO = jnp.expand_dims(jnp.array(two, dtype=jnp.uint8), axis=0)
        SPRITE_THREE = jnp.expand_dims(jnp.array(three, dtype=jnp.uint8), axis=0)
        SPRITE_TETRIS = jnp.expand_dims(jnp.array( tetris, dtype=jnp.uint8), axis=0)


        return (
            SPRITE_BG,
            SPRITE_BOARD,
            SCORE_DIGIT_SPRITES,
            SPRITE_ROW_COLORS,
            SPRITE_ONE,
            SPRITE_TWO,
            SPRITE_THREE,
            SPRITE_TETRIS
        )

    @partial(jax.jit, static_argnums=(0,))
    def get_piece_shape(self, piece_idx: chex.Array, rotation_idx: chex.Array) -> chex.Array:
        return self.consts.TETROMINOS[piece_idx, rotation_idx]

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        # Base raster and static layers
        raster = jr.create_initial_frame(width=160, height=210)

        frame_bg = jr.get_sprite_frame(self.SPRITE_BG, 0)
        raster = jr.render_at(raster, 0, 0, frame_bg)

        frame_board = jr.get_sprite_frame(self.SPRITE_BOARD, 0)
        raster = jr.render_at(raster, self.consts.BOARD_X, self.consts.BOARD_Y, frame_board)

        # Vectorized board + piece rasterization (no loops)
        board = state.board  # (H_board, W_board) {0,1}
        H_board = board.shape[0]
        W_board = board.shape[1]

        # Raster grid (xy indexing)
        H_out, W_out, _ = raster.shape
        xx, yy = jnp.meshgrid(jnp.arange(W_out, dtype=jnp.int32),
                              jnp.arange(H_out, dtype=jnp.int32), indexing="xy")

        # Geometry
        cell_w = jnp.int32(self.consts.CELL_WIDTH)
        cell_h = jnp.int32(self.consts.CELL_HEIGHT)
        stride_x = cell_w + jnp.int32(1)
        stride_y = cell_h + jnp.int32(1)
        board_x0 = jnp.int32(self.consts.BOARD_X + self.consts.BOARD_PADDING)
        board_y0 = jnp.int32(self.consts.BOARD_Y)

        # Local board-plane coordinates
        local_x = xx - board_x0
        local_y = yy - board_y0

        in_region = (
            (local_x >= 0)
            & (local_y >= 0)
            & (local_x < stride_x * W_board)
            & (local_y < stride_y * H_board)
        )

        # Cell indices and intra-cell offsets
        cell_c = local_x // stride_x
        cell_r = local_y // stride_y
        off_x = local_x - cell_c * stride_x
        off_y = local_y - cell_r * stride_y

        inside_tile = (off_x < cell_w) & (off_y < cell_h)

        # Safe indices for board gather
        cell_r_c = jnp.clip(cell_r, 0, H_board - 1)
        cell_c_c = jnp.clip(cell_c, 0, W_board - 1)

        # Board occupancy per pixel
        board_on = (board[cell_r_c, cell_c_c] == 1)

        # Piece occupancy per pixel
        piece = self.get_piece_shape(state.piece_type, state.rot)  # (4,4)
        pos_y, pos_x = state.pos
        rel_y = cell_r - jnp.asarray(pos_y, jnp.int32)
        rel_x = cell_c - jnp.asarray(pos_x, jnp.int32)
        in_piece = (rel_y >= 0) & (rel_y < 4) & (rel_x >= 0) & (rel_x < 4)
        rel_y_c = jnp.clip(rel_y, 0, 3)
        rel_x_c = jnp.clip(rel_x, 0, 3)
        piece_on = (piece[rel_y_c, rel_x_c] == 1) & in_piece

        # Active pixels to draw (non-overlapping by construction)
        active = in_region & inside_tile & (board_on | piece_on)

        # Select row-colored sprite pixel per raster pixel
        row_sel = ((H_board - 1 - cell_r_c) % jnp.int32(self.N_COLOR_ROWS))
        off_y_c = jnp.clip(off_y, 0, cell_h - 1)
        off_x_c = jnp.clip(off_x, 0, cell_w - 1)
        sprite_px_rgba = self.SPRITE_ROW_COLORS[row_sel, off_y_c, off_x_c]  # (...,4)

        # Alpha blend in one pass for all active pixels
        sprite_rgb = sprite_px_rgba[..., :3].astype(jnp.float32)
        sprite_a = (sprite_px_rgba[..., 3].astype(jnp.float32) / 255.0)
        base_rgb = raster[..., :3].astype(jnp.float32)

        a_mask = sprite_a * active.astype(jnp.float32)
        blended_rgb = sprite_rgb * a_mask[..., None] + base_rgb * (1.0 - a_mask[..., None])
        raster = jnp.where(active[..., None], blended_rgb.astype(raster.dtype), raster)

        # score (unchanged)
        score_digits = jr.int_to_digits(state.score, max_digits=4)
        raster = jr.render_label_selective(
            raster,
            95, 27,
            score_digits,
            self.SCORE_DIGIT_SPRITES,
            start_index=0,
            num_to_render=4,
            spacing=16
        )

        # -------- NEW: draw ONE / TWO / THREE / TETRIS while banner is active --------
        # compute a position to the right of the board
        label_x = 95
        label_y = 122

        def draw_with(sprite, r_):
            frame = jr.get_sprite_frame(sprite, 0)
            return jr.render_at(r_, label_x, label_y, frame)

        def draw_none(r_):   return r_

        def draw_one(r_):    return draw_with(self.SPRITE_ONE, r_)

        def draw_two(r_):    return draw_with(self.SPRITE_TWO, r_)

        def draw_three(r_):  return draw_with(self.SPRITE_THREE, r_)

        def draw_tetris(r_): return draw_with(self.SPRITE_TETRIS, r_)

        def draw_banner(r_):
            # banner_code: 0 none, 1 ONE, 2 TWO, 3 THREE, 4 TETRIS
            idx = jnp.clip(state.banner_code, jnp.int32(0), jnp.int32(4))
            fns = [draw_none, draw_one, draw_two, draw_three, draw_tetris]
            return jax.lax.switch(idx, fns, r_)

        raster = jax.lax.cond(
            (state.banner_timer > jnp.int32(0)) & (state.banner_code > jnp.int32(0)),
            draw_banner,
            lambda r_: r_,
            raster
        )

        return raster