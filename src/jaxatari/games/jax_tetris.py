from __future__ import annotations
from functools import partial
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
import chex
from jax import lax
from jax import random as jrandom

import jaxatari.spaces as spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import JAXGameRenderer

# ===================== Static Config =====================
BOARD_W: int = 10
BOARD_H: int = 20

# Env timing – in "env frames" (number of step() calls)
GRAVITY_FRAMES: int = 30      # auto-fall cadence
DAS_FRAMES: int = 10          # delayed auto shift for horiz
ARR_FRAMES: int = 3           # auto-repeat rate for horiz
ROT_DAS_FRAMES: int = 12      # rotate auto-repeat cadence
SOFT_PACE_FRAMES: int = 4     # paced soft-drop while held

# Scoring (custom rules)
LINE_CLEAR_SCORE = (0, 100, 300, 500, 800)  # kept for structure but not used for tally
SOFT_DROP_SCORE_PER_CELL = 1                # +1 when DOWN moves one cell
SCORE_PER_LINE = 100                        # +100 per cleared line
SCORE_ON_LOCK = 20                          # +20 when a piece locks

# Colors (no assets)
BG_COLOR     = jnp.array([18, 18, 22], dtype=jnp.uint8)
COLOR_EMPTY  = jnp.array([25, 25, 30], dtype=jnp.uint8)
COLOR_LOCKED = jnp.array([90, 210, 255], dtype=jnp.uint8)
COLOR_ACTIVE = jnp.array([220, 70, 100], dtype=jnp.uint8)
COLOR_GHOST  = jnp.array([70, 70, 90], dtype=jnp.uint8)
COLOR_SCORE  = jnp.array([245, 245, 245], dtype=jnp.uint8)  # scoreboard digits

# Render tiling
CELL: int = 5
MARGIN: int = 2

# ---- Scoreboard layout (in grid cells to the RIGHT of the board) ----
DIGIT_W: int = 3   # 3 cells wide per digit
DIGIT_H: int = 5   # 5 cells high per digit
DIGIT_SPACE: int = 1  # 1 empty cell between digits
MAX_SCORE_DIGITS: int = 6
HUD_W_CELLS: int = MAX_SCORE_DIGITS * DIGIT_W + (MAX_SCORE_DIGITS - 1) * DIGIT_SPACE + 2  # +2 padding

IMG_H = BOARD_H * (CELL + MARGIN) + MARGIN
IMG_W = (BOARD_W + HUD_W_CELLS) * (CELL + MARGIN) + MARGIN  # <- extra space on the RIGHT for score


def _tile(v3: jnp.ndarray) -> jnp.ndarray:
    return jnp.broadcast_to(v3.reshape(1,1,3), (CELL, CELL, 3))
TILE_EMPTY  = _tile(COLOR_EMPTY)
TILE_LOCKED = _tile(COLOR_LOCKED)
TILE_ACTIVE = _tile(COLOR_ACTIVE)
TILE_GHOST  = _tile(COLOR_GHOST)
TILE_SCORE  = _tile(COLOR_SCORE)

# ======================== Tetrominoes ====================
TETROMINOS = jnp.array([
    # I
    [
        [[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]],
        [[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0]],
        [[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]],
        [[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0]],
    ],
    # O
    [
        [[0,0,0,0],[0,1,1,0],[0,1,1,0],[0,0,0,0]],
        [[0,0,0,0],[0,1,1,0],[0,1,1,0],[0,0,0,0]],
        [[0,0,0,0],[0,1,1,0],[0,1,1,0],[0,0,0,0]],
        [[0,0,0,0],[0,1,1,0],[0,1,1,0],[0,0,0,0]],
    ],
    # T
    [
        [[0,0,0,0],[0,1,0,0],[1,1,1,0],[0,0,0,0]],
        [[0,1,0,0],[0,1,1,0],[0,1,0,0],[0,0,0,0]],
        [[0,0,0,0],[1,1,1,0],[0,1,0,0],[0,0,0,0]],
        [[0,1,0,0],[1,1,0,0],[0,1,0,0],[0,0,0,0]],
    ],
    # S
    [
        [[0,0,0,0],[0,1,1,0],[1,1,0,0],[0,0,0,0]],
        [[0,1,0,0],[0,1,1,0],[0,0,1,0],[0,0,0,0]],
        [[0,0,0,0],[0,1,1,0],[1,1,0,0],[0,0,0,0]],
        [[0,1,0,0],[0,1,1,0],[0,0,1,0],[0,0,0,0]],
    ],
    # Z
    [
        [[0,0,0,0],[1,1,0,0],[0,1,1,0],[0,0,0,0]],
        [[0,0,1,0],[0,1,1,0],[0,1,0,0],[0,0,0,0]],
        [[0,0,0,0],[1,1,0,0],[0,1,1,0],[0,0,0,0]],
        [[0,0,1,0],[0,1,1,0],[0,1,0,0],[0,0,0,0]],
    ],
    # J
    [
        [[0,0,0,0],[1,0,0,0],[1,1,1,0],[0,0,0,0]],
        [[0,1,0,0],[0,1,0,0],[1,1,0,0],[0,0,0,0]],
        [[0,0,0,0],[1,1,1,0],[0,0,1,0],[0,0,0,0]],
        [[0,1,1,0],[0,1,0,0],[0,1,0,0],[0,0,0,0]],
    ],
    # L
    [
        [[0,0,0,0],[0,0,1,0],[1,1,1,0],[0,0,0,0]],
        [[1,1,0,0],[0,1,0,0],[0,1,0,0],[0,0,0,0]],
        [[0,0,0,0],[1,1,1,0],[1,0,0,0],[0,0,0,0]],
        [[0,1,0,0],[0,1,0,0],[0,1,1,0],[0,0,0,0]],
    ],
], dtype=jnp.int32)  # (7,4,4,4)

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

# ======================== Pure JAX helpers =================
@jax.jit
def piece_grid(piece_type: chex.Array, rot: chex.Array) -> chex.Array:
    return TETROMINOS[piece_type, (rot & 3)]

@jax.jit
def spawn_piece(key: chex.Array) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    key, sub = jrandom.split(key)
    p = jrandom.randint(sub, (), 0, 7, dtype=jnp.int32)
    pos = jnp.array([0, 3], dtype=jnp.int32)
    rot = jnp.int32(0)
    return p, pos, rot, key

@jax.jit
def check_collision(board: chex.Array, grid4: chex.Array, pos: chex.Array) -> chex.Array:
    H = jnp.int32(BOARD_H); W = jnp.int32(BOARD_W)
    def body(i, acc):
        y, x = jnp.divmod(i, 4)
        on = grid4[y, x] == 1
        py = pos[0] + y; px = pos[1] + x
        inb_side = (px >= 0) & (px < W) & (py < H)
        readable = inb_side & (py >= 0)
        pyc = jnp.clip(py, 0, H - 1); pxc = jnp.clip(px, 0, W - 1)
        occ = jnp.where(readable, board[pyc, pxc] == 1, False)
        side_or_bottom_oob = on & (~inb_side) & (py >= 0)
        return acc | (on & (occ | side_or_bottom_oob))
    return lax.fori_loop(0, 16, body, False)

@jax.jit
def lock_piece(board: chex.Array, grid4: chex.Array, pos: chex.Array) -> chex.Array:
    H = jnp.int32(BOARD_H); W = jnp.int32(BOARD_W)
    def body(i, b):
        y, x = jnp.divmod(i, 4)
        on = grid4[y, x] == 1
        py = pos[0] + y; px = pos[1] + x
        inb = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        pyc = jnp.clip(py, 0, H - 1); pxc = jnp.clip(px, 0, W - 1)
        return lax.cond(on & inb, lambda bb: bb.at[pyc, pxc].set(1), lambda bb: bb, b)
    return lax.fori_loop(0, 16, body, board)

@jax.jit
def clear_lines(board: chex.Array) -> Tuple[chex.Array, chex.Array]:
    full = jnp.all(board == 1, axis=1)
    def scan_row(carry, y):
        nb, wy, cnt = carry
        row = board[y]; is_full = full[y]
        def write_row(c):
            _nb, _wy, _cnt = c
            _nb = _nb.at[_wy].set(row)
            return _nb, _wy - 1, _cnt
        def skip_row(c):
            _nb, _wy, _cnt = c
            return _nb, _wy, _cnt + 1
        return lax.cond(~is_full, write_row, skip_row, (nb, wy, cnt)), None
    init = (jnp.zeros_like(board), jnp.int32(BOARD_H - 1), jnp.int32(0))
    (nb, _wy, cleared), _ = lax.scan(scan_row, init, jnp.arange(BOARD_H - 1, -1, -1))
    return nb, cleared

@jax.jit
def try_rotate_with_kick(board: chex.Array, piece_type: chex.Array, pos: chex.Array, rot: chex.Array):
    """CW rotate; if collides, try L, R, then Up (all 1-cell)."""
    new_rot = (rot + 1) & 3
    g2 = piece_grid(piece_type, new_rot)
    pos0 = pos
    posL = pos + jnp.array([0, -1], jnp.int32)
    posR = pos + jnp.array([0,  1], jnp.int32)
    posU = pos + jnp.array([-1, 0], jnp.int32)

    # variables kept for structure (future SRS I-kicks)
    posL2 = pos + jnp.array([0, -2], jnp.int32)
    posR2 = pos + jnp.array([0, 2], jnp.int32)
    is_I = (piece_type == jnp.int32(0))
    _ = (posL2, posR2, is_I)

    ok0 = ~check_collision(board, g2, pos0)
    okL = ~check_collision(board, g2, posL)
    okR = ~check_collision(board, g2, posR)
    okU = ~check_collision(board, g2, posU)

    pos_final = jnp.where(ok0, pos0,
                  jnp.where(okL, posL,
                  jnp.where(okR, posR,
                  jnp.where(okU, posU, pos))))
    rot_final = jnp.where(ok0 | okL | okR | okU, new_rot, rot)
    return pos_final, rot_final

@jax.jit
def ghost_pos(board: chex.Array, piece_type: chex.Array, pos: chex.Array, rot: chex.Array):
    g = piece_grid(piece_type, rot)
    def cond_fn(p):
        p_next = p + jnp.array([1, 0], jnp.int32)
        return ~check_collision(board, g, p_next)
    def body_fn(p):
        return p + jnp.array([1, 0], jnp.int32)
    return lax.while_loop(cond_fn, body_fn, pos)


# ======================= Score glyphs (3x5) =====================
# Each digit is a 5x3 binary grid; 1-cells are drawn with TILE_SCORE
DIGIT_GLYPHS = jnp.array([
    # 0
    [[1,1,1],
     [1,0,1],
     [1,0,1],
     [1,0,1],
     [1,1,1]],
    # 1
    [[0,1,0],
     [1,1,0],
     [0,1,0],
     [0,1,0],
     [1,1,1]],
    # 2
    [[1,1,1],
     [0,0,1],
     [1,1,1],
     [1,0,0],
     [1,1,1]],
    # 3
    [[1,1,1],
     [0,0,1],
     [1,1,1],
     [0,0,1],
     [1,1,1]],
    # 4
    [[1,0,1],
     [1,0,1],
     [1,1,1],
     [0,0,1],
     [0,0,1]],
    # 5
    [[1,1,1],
     [1,0,0],
     [1,1,1],
     [0,0,1],
     [1,1,1]],
    # 6
    [[1,1,1],
     [1,0,0],
     [1,1,1],
     [1,0,1],
     [1,1,1]],
    # 7
    [[1,1,1],
     [0,0,1],
     [0,1,0],
     [0,1,0],
     [0,1,0]],
    # 8
    [[1,1,1],
     [1,0,1],
     [1,1,1],
     [1,0,1],
     [1,1,1]],
    # 9
    [[1,1,1],
     [1,0,1],
     [1,1,1],
     [0,0,1],
     [1,1,1]],
], dtype=jnp.uint8)  # (10, 5, 3)


# ======================= Renderer (pure JAX) =====================
class TetrisRenderer(JAXGameRenderer):
    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: TetrisState) -> chex.Array:
        def _blit(im, top, left, tile):
            def row(dy, m):
                def col(dx, r):
                    py = top + dy; px = left + dx
                    return r.at[py, px].set(tile[dy, dx])
                m = lax.fori_loop(0, CELL, col, m)
                return m
            return lax.fori_loop(0, CELL, row, im)

        @jax.jit
        def _draw_cell(img, y, x, tile):
            top = y * (CELL + MARGIN) + MARGIN
            left = x * (CELL + MARGIN) + MARGIN
            return _blit(img, top, left, tile)

        # ---- canvas ----
        img = jnp.ones((IMG_H, IMG_W, 3), dtype=jnp.uint8) * BG_COLOR

        # ---- playfield (left) ----
        def fill_cell(i, im):
            y = i // BOARD_W; x = i % BOARD_W
            return _draw_cell(im, y, x, TILE_EMPTY)
        img = lax.fori_loop(0, BOARD_H * BOARD_W, fill_cell, img)

        # ---- ghost ----
        g = piece_grid(state.piece_type, state.rot)
        gp = ghost_pos(state.board, state.piece_type, state.pos, state.rot)
        def draw_ghost(i, im):
            gy = i // 4; gx = i % 4
            on = g[gy, gx] == 1
            py = gp[0] + gy; px = gp[1] + gx
            inb = (py >= 0) & (py < BOARD_H) & (px >= 0) & (px < BOARD_W)
            return lax.cond(on & inb, lambda m: _draw_cell(m, py, px, TILE_GHOST), lambda m: m, im)
        img = lax.fori_loop(0, 16, draw_ghost, img)

        # ---- locked blocks ----
        def draw_locked(i, im):
            y = i // BOARD_W; x = i % BOARD_W
            return lax.cond(state.board[y, x] == 1, lambda m: _draw_cell(m, y, x, TILE_LOCKED), lambda m: m, im)
        img = lax.fori_loop(0, BOARD_H * BOARD_W, draw_locked, img)

        # ---- active piece ----
        def draw_active(i, im):
            gy = i // 4; gx = i % 4
            on = g[gy, gx] == 1
            py = state.pos[0] + gy; px = state.pos[1] + gx
            inb = (py >= 0) & (py < BOARD_H) & (px >= 0) & (px < BOARD_W)
            return lax.cond(on & inb, lambda m: _draw_cell(m, py, px, TILE_ACTIVE), lambda m: m, im)
        img = lax.fori_loop(0, 16, draw_active, img)

        # ---- scoreboard background (right sidebar) ----
        HUD_X0 = BOARD_W  # first column right after the playfield
        HUD_W = HUD_W_CELLS
        def fill_hud(i, im):
            y = i // HUD_W; x = i % HUD_W
            return _draw_cell(im, y, HUD_X0 + x, TILE_EMPTY)
        img = lax.fori_loop(0, BOARD_H * HUD_W, fill_hud, img)

        # ---- convert state.score to digits (LSB first) ----
        def int_to_digits(n, maxd):
            def body(carry, i):
                val, out = carry
                q = val // 10
                r = val - q * 10
                out = out.at[i].set(r.astype(jnp.int32))
                return (q, out), None
            out0 = jnp.zeros((maxd,), dtype=jnp.int32)
            (q_final, out_digits), _ = lax.scan(body, (jnp.int32(n), out0), jnp.arange(maxd))
            return out_digits  # [0]=ones, ...

        digits_lsd = int_to_digits(state.score, MAX_SCORE_DIGITS)

        # number of meaningful digits (1..MAX)
        nd = jnp.where(state.score < 10, 1,
             jnp.where(state.score < 100, 2,
             jnp.where(state.score < 1000, 3,
             jnp.where(state.score < 10000, 4,
             jnp.where(state.score < 100000, 5, MAX_SCORE_DIGITS)))))

        # total width in cells occupied by the number
        total_w = nd * DIGIT_W + jnp.maximum(0, nd - 1) * DIGIT_SPACE
        start_x = HUD_X0 + HUD_W - total_w - 1  # right align with 1-cell right margin
        start_y = 1  # top padding

        # Draw digits left->right across the HUD area
        def draw_digit_slot(i, im):
            # i = 0..MAX_SCORE_DIGITS-1 across the HUD row
            # Only draw when i is within the last 'nd' slots
            draw_this = i >= (MAX_SCORE_DIGITS - nd)
            # Which actual digit index (0..nd-1) does this slot correspond to?
            # Map slots so the rightmost slot is the ones place.
            k = i - (MAX_SCORE_DIGITS - nd)                # 0..nd-1
            # Fetch digit value from LSD-first buffer: value at index (nd-1-k)
            val_idx = jnp.maximum(0, nd - 1 - k)
            dval = digits_lsd[val_idx]

            # compute top-left cell for this digit block
            x0 = start_x + k * (DIGIT_W + DIGIT_SPACE)
            y0 = start_y

            def draw_digit(im2):
                # 5x3 glyph
                def cell_loop(j, im3):
                    gy = j // DIGIT_W
                    gx = j % DIGIT_W
                    on = DIGIT_GLYPHS[dval, gy, gx] == 1
                    return lax.cond(on, lambda m: _draw_cell(m, y0 + gy, x0 + gx, TILE_SCORE), lambda m: m, im3)
                return lax.fori_loop(0, DIGIT_W * DIGIT_H, cell_loop, im2)

            return lax.cond(draw_this, draw_digit, lambda m: m, im)

        img = lax.fori_loop(0, MAX_SCORE_DIGITS, draw_digit_slot, img)

        return img

# ======================= Environment =====================
class TetrisConstants(NamedTuple):
    RESET: int = Action.DOWNLEFTFIRE  # keep a reserved reset action
    ACTION_SET: tuple = (
        Action.NOOP,      # 0
        Action.LEFT,      # 1
        Action.RIGHT,     # 2
        Action.UP,        # 3  -> rotate
        Action.DOWN,      # 4  -> soft drop (paced)
        Action.FIRE,      # 5  -> hard drop
        # also accept diagonals naturally via Atari codes
    )

class JaxTetris(JaxEnvironment[TetrisState, TetrisObservation, TetrisInfo, TetrisConstants]):
    def __init__(self, consts: TetrisConstants | None = None):
        super().__init__(consts or TetrisConstants())
        self.renderer = TetrisRenderer()
        self.action_set = jnp.array(list(self.consts.ACTION_SET), dtype=jnp.int32)

    # ----- Spaces -----
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.consts.ACTION_SET))

    def observation_space(self) -> spaces.Dict:
        return spaces.Dict({
            "board": spaces.Box(low=0, high=1, shape=(BOARD_H, BOARD_W), dtype=jnp.int32),
            "piece_type": spaces.Box(low=0, high=6, shape=(), dtype=jnp.int32),
            "pos": spaces.Box(low=0, high=max(BOARD_H, BOARD_W), shape=(2,), dtype=jnp.int32),
            "rot": spaces.Box(low=0, high=3, shape=(), dtype=jnp.int32),
            "next_piece": spaces.Box(low=0, high=6, shape=(), dtype=jnp.int32),
        })

    def image_space(self) -> spaces.Box:
        return spaces.Box(low=0, high=255, shape=(IMG_H, IMG_W, 3), dtype=jnp.uint8)

    # ----- Public API -----
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jrandom.PRNGKey = None) -> Tuple[TetrisObservation, TetrisState]:
        key = jrandom.PRNGKey(0) if key is None else key
        board = jnp.zeros((BOARD_H, BOARD_W), dtype=jnp.int32)
        cur, pos, rot, key = spawn_piece(key)
        nxt, _, _, key = spawn_piece(key)
        state = TetrisState(
            board=board, piece_type=cur, pos=pos, rot=rot, next_piece=nxt,
            score=jnp.int32(0), game_over=jnp.bool_(False), key=key,
            tick=jnp.int32(0),
            das_timer=jnp.int32(0), arr_timer=jnp.int32(0), move_dir=jnp.int32(0),
            rot_timer=jnp.int32(0), soft_timer=jnp.int32(0),
            last_action=jnp.int32(Action.NOOP)
        )
        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def step(self, state: TetrisState, action: chex.Array) -> Tuple[
        TetrisObservation, TetrisState, float, bool, TetrisInfo]:
        a = action.astype(jnp.int32)
        is_left = (a == Action.LEFT)
        is_right = (a == Action.RIGHT)
        is_up = (a == Action.UP)
        is_down = (a == Action.DOWN)
        is_fire = (a == Action.FIRE)

        do_reset = (a == self.consts.RESET)

        tick_next = state.tick + jnp.int32(1)
        gravity_drop = (tick_next % jnp.int32(GRAVITY_FRAMES) == 0).astype(jnp.int32)

        # Held-key timers
        new_move_dir = jnp.where(is_left, -1, jnp.where(is_right, 1, state.move_dir * 0))
        das = jnp.where(new_move_dir != 0,
                        jnp.where(state.move_dir == 0, jnp.int32(DAS_FRAMES), jnp.maximum(0, state.das_timer - 1)),
                        jnp.int32(0))
        arr = jnp.where(new_move_dir != 0,
                        jnp.where(das == 0, jnp.maximum(0, state.arr_timer - 1), jnp.int32(ARR_FRAMES)),
                        jnp.int32(0))
        # ✅ fixed parentheses here:
        do_move_now = (new_move_dir != 0) & ((state.move_dir == 0) | ((das == 0) & (arr == 0)))

        rot_timer = jnp.where(is_up, jnp.maximum(0, state.rot_timer - 1), jnp.int32(0))
        do_rotate_now = is_up & ((state.last_action != Action.UP) | (rot_timer == 0))

        soft_timer = jnp.where(is_down, jnp.maximum(0, state.soft_timer - 1), jnp.int32(0))
        do_soft_now = is_down & (soft_timer == 0)

        do_hard_now = is_fire & (state.last_action != Action.FIRE)

        # Rotate
        pos_r, rot_r = lax.cond(
            do_rotate_now,
            lambda _: try_rotate_with_kick(state.board, state.piece_type, state.pos, state.rot),
            lambda _: (state.pos, state.rot),
            operand=None
        )
        state = state._replace(pos=pos_r, rot=rot_r)

        # Horizontal
        grid = piece_grid(state.piece_type, state.rot)
        pos_h = state.pos + jnp.array([0, jnp.int32(jnp.where(do_move_now, new_move_dir, 0))], dtype=jnp.int32)
        coll_h = check_collision(state.board, grid, pos_h)
        state = lax.cond(coll_h, lambda s: s, lambda s: s._replace(pos=pos_h), state)

        # Vertical
        def do_hard(s: TetrisState):
            g = piece_grid(s.piece_type, s.rot)

            def cond_fun(t):
                pnext = t.pos + jnp.array([1, 0], jnp.int32)
                return ~check_collision(t.board, g, pnext)

            def body_fun(t):
                return t._replace(pos=t.pos + jnp.array([1, 0], jnp.int32))

            s2 = lax.while_loop(cond_fun, body_fun, s)
            return _lock_spawn(s2, g, tick_next, soft_points=jnp.int32(0))

        def do_soft_or_gravity(s: TetrisState):
            # Compute requested movement
            dy_soft = do_soft_now.astype(jnp.int32)       # 1 when DOWN triggers a paced soft step this frame
            dy_grav = gravity_drop                         # 1 when gravity tick triggers
            dy = jnp.clip(dy_soft | dy_grav, 0, 1)        # actual attempted move (0/1)

            pos_v = s.pos + jnp.array([dy, 0], dtype=jnp.int32)
            coll_v = check_collision(s.board, grid, pos_v)

            # +1 only for a successful soft step (ignore gravity)
            soft_points = SOFT_DROP_SCORE_PER_CELL * (1 - coll_v.astype(jnp.int32)) * dy_soft

            return lax.cond(
                coll_v,
                lambda ss: _lock_spawn(ss, grid, tick_next, soft_points),
                lambda ss: (ss._replace(pos=pos_v, score=ss.score + soft_points, tick=tick_next),
                            jnp.float32(0.0), jnp.bool_(False),
                            TetrisInfo(score=ss.score + soft_points, cleared=jnp.int32(0), game_over=ss.game_over)),
                s
            )

        def do_env_reset(_):
            obs, st0 = self.reset(state.key)
            return st0, jnp.float32(0.0), jnp.bool_(False), TetrisInfo(score=st0.score, cleared=jnp.int32(0),
                                                                       game_over=st0.game_over)

        state, reward, done, info = lax.cond(
            do_reset, do_env_reset,
            lambda _: lax.cond(do_hard_now, do_hard, do_soft_or_gravity, state),
            operand=None
        )

        def after_over(ss):
            obs2, st2 = self.reset(ss.key)
            return st2, jnp.float32(0.0), jnp.bool_(False), TetrisInfo(score=st2.score, cleared=jnp.int32(0),
                                                                       game_over=jnp.bool_(False))

        state, reward, done, info = lax.cond(state.game_over, after_over, lambda s: (s, reward, done, info), state)

        state = state._replace(
            das_timer=das,
            arr_timer=arr,
            move_dir=new_move_dir,
            rot_timer=jnp.where(do_rotate_now, jnp.int32(ROT_DAS_FRAMES), rot_timer),
            soft_timer=jnp.where(do_soft_now, jnp.int32(SOFT_PACE_FRAMES), soft_timer),
            last_action=a
        )

        obs = self._get_observation(state)
        return obs, state, reward, jnp.bool_(False), info

    # ----- Helpers used inside step -----
    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: TetrisState) -> TetrisObservation:
        return TetrisObservation(
            board=state.board,
            piece_type=state.piece_type,
            pos=state.pos,
            rot=state.rot,
            next_piece=state.next_piece,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: TetrisState) -> TetrisInfo:
        return TetrisInfo(score=state.score, cleared=jnp.int32(0), game_over=state.game_over)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: TetrisState, state: TetrisState) -> float:
        return (state.score - previous_state.score).astype(jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: TetrisState) -> bool:
        # unused (auto-restart), keep for API completeness
        return state.game_over

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: TetrisObservation) -> chex.Array:
        return jnp.concatenate([obs.board.flatten(), obs.piece_type[None], obs.pos, obs.rot[None], obs.next_piece[None]]).astype(jnp.int32)

    def render(self, state: TetrisState) -> jnp.ndarray:
        return self.renderer.render(state)

# ---------- lock/spawn core (kept close to env to use constants) ----------
@partial(jax.jit, static_argnums=())
def _lock_spawn(s: TetrisState, grid: chex.Array, tick_next: chex.Array, soft_points: chex.Array):
    b2 = lock_piece(s.board, grid, s.pos)
    b3, cleared = clear_lines(b2)

    # scoring: +20 for lock, +100 per cleared line, +soft_points passed in
    add_lines = cleared * jnp.int32(SCORE_PER_LINE)
    add_lock  = jnp.int32(SCORE_ON_LOCK)
    add_total = add_lines + add_lock + soft_points

    score2 = s.score + add_total
    cur = s.next_piece
    pos, rot = jnp.array([0, 3], jnp.int32), jnp.int32(0)
    g_new = piece_grid(cur, rot)
    over = check_collision(b3, g_new, pos)
    nxt, _, _, key2 = spawn_piece(s.key)
    s2 = s._replace(board=b3, piece_type=cur, pos=pos, rot=rot,
                    next_piece=nxt, score=score2, game_over=over, key=key2, tick=tick_next)

    # Reward equals score gained on this event (lock or line clear)
    reward = (score2 - s.score).astype(jnp.float32)
    info = TetrisInfo(score=s2.score, cleared=cleared, game_over=over)
    return s2, reward, over, info