from __future__ import annotations
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
    LINE_CLEAR_SCORE = (0, 1, 3, 5, 8)  # 0..4

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
            [[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]],
            [[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]],
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
    def __init__(self, consts: TetrisConstants = None, reward_funcs: list[callable]=None):
        consts = consts or TetrisConstants()
        super().__init__(consts)
        self.renderer = TetrisRenderer(self.consts)
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.action_set = [
            Action.NOOP,
            Action.FIRE,
            Action.RIGHT,
            Action.LEFT,
            Action.DOWN
        ]
        self.obs_size = 3 * 4 + 1 + 1

    # ----- Helpers -----
    @partial(jax.jit, static_argnums=0)
    def piece_grid(self, piece_type: chex.Array, rot: chex.Array) -> chex.Array:
        return self.consts.TETROMINOS[piece_type, (rot & 3)]

    @partial(jax.jit, static_argnums=0)
    def spawn_piece(self, key: chex.Array) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        key, sub = jrandom.split(key)
        p = jrandom.randint(sub, (), 0, 7, dtype=jnp.int32)
        pos = jnp.array([0, 3], dtype=jnp.int32)
        rot = jnp.int32(0)
        return p, pos, rot, key

    @partial(jax.jit, static_argnums=0)
    def check_collision(self, board: chex.Array, grid4: chex.Array, pos: chex.Array) -> chex.Array:
        H = jnp.int32(self.consts.BOARD_HEIGHT)
        W = jnp.int32(self.consts.BOARD_WIDTH)

        def body(i, acc):
            y, x = jnp.divmod(i, 4)
            on = grid4[y, x] == 1
            py = pos[0] + y
            px = pos[1] + x
            inb_side = (px >= 0) & (px < W) & (py < H)
            readable = inb_side & (py >= 0)
            pyc = jnp.clip(py, 0, H - 1)
            pxc = jnp.clip(px, 0, W - 1)
            occ = jnp.where(readable, board[pyc, pxc] == 1, False)
            side_or_bottom_oob = on & (~inb_side) & (py >= 0)
            return acc | (on & (occ | side_or_bottom_oob))

        return lax.fori_loop(0, 16, body, False)

    @partial(jax.jit, static_argnums=0)
    def lock_piece(self, board: chex.Array, grid4: chex.Array, pos: chex.Array) -> chex.Array:
        H = jnp.int32(self.consts.BOARD_HEIGHT)
        W = jnp.int32(self.consts.BOARD_WIDTH)

        def body(i, b):
            y, x = jnp.divmod(i, 4)
            on = grid4[y, x] == 1
            py = pos[0] + y
            px = pos[1] + x
            inb = (px >= 0) & (px < W) & (py >= 0) & (py < H)
            pyc = jnp.clip(py, 0, H - 1)
            pxc = jnp.clip(px, 0, W - 1)
            return lax.cond(on & inb, lambda bb: bb.at[pyc, pxc].set(1), lambda bb: bb, b)

        return lax.fori_loop(0, 16, body, board)

    @partial(jax.jit, static_argnums=0)
    def clear_lines(self, board: chex.Array) -> Tuple[chex.Array, chex.Array]:
        full = jnp.all(board == 1, axis=1)

        def scan_row(carry, y):
            nb, wy, cnt = carry
            row = board[y]
            is_full = full[y]

            def write_row(c):
                _nb, _wy, _cnt = c
                _nb = _nb.at[_wy].set(row)
                return _nb, _wy - 1, _cnt

            def skip_row(c):
                _nb, _wy, _cnt = c
                return _nb, _wy, _cnt + 1

            return lax.cond(~is_full, write_row, skip_row, (nb, wy, cnt)), None

        init = (jnp.zeros_like(board), jnp.int32(self.consts.BOARD_HEIGHT - 1), jnp.int32(0))
        (nb, _wy, cleared), _ = lax.scan(scan_row, init, jnp.arange(self.consts.BOARD_HEIGHT - 1, -1, -1))
        return nb, cleared

    @partial(jax.jit, static_argnums=0)
    def try_rotate_with_kick(self, board: chex.Array, piece_type: chex.Array, pos: chex.Array, rot: chex.Array):
        """CW rotate; if collides, try L, R, then Up (all 1-cell)."""
        new_rot = (rot + 1) & 3
        g2 = self.piece_grid(piece_type, new_rot)
        pos0 = pos
        posL = pos + jnp.array([0, -1], jnp.int32)
        posR = pos + jnp.array([0, 1], jnp.int32)
        posU = pos + jnp.array([-1, 0], jnp.int32)

        # extra kicks for I piece (index 0)
        posL2 = pos + jnp.array([0, -2], jnp.int32)
        posR2 = pos + jnp.array([0, 2], jnp.int32)
        is_I = (piece_type == jnp.int32(0))

        ok0 = ~self.check_collision(board, g2, pos0)
        okL = ~self.check_collision(board, g2, posL)
        okR = ~self.check_collision(board, g2, posR)
        okU = ~self.check_collision(board, g2, posU)

        pos_final = jnp.where(ok0, pos0,
                              jnp.where(okL, posL,
                                        jnp.where(okR, posR,
                                                  jnp.where(okU, posU, pos))))
        rot_final = jnp.where(ok0 | okL | okR | okU, new_rot, rot)
        return pos_final, rot_final

    @partial(jax.jit, static_argnums=0)
    def _lock_spawn(self, s: TetrisState, grid: chex.Array, tick_next: chex.Array, soft_points: chex.Array):
        b2 = self.lock_piece(s.board, grid, s.pos)
        b3, cleared = self.clear_lines(b2)
        add = jnp.array(self.consts.LINE_CLEAR_SCORE, dtype=jnp.int32)[jnp.clip(cleared, 0, 4)]
        score2 = s.score + add + soft_points
        cur = s.next_piece
        pos, rot = jnp.array([0, 3], jnp.int32), jnp.int32(0)
        g_new = self.piece_grid(cur, rot)
        over = self.check_collision(b3, g_new, pos)
        nxt, _, _, key2 = self.spawn_piece(s.key)
        s2 = s._replace(board=b3, piece_type=cur, pos=pos, rot=rot,
                        next_piece=nxt, score=score2, game_over=over, key=key2, tick=tick_next)
        reward = (cleared > 0).astype(jnp.float32) * add.astype(jnp.float32)
        info = TetrisInfo(score=s2.score, cleared=cleared, game_over=over)
        return s2, reward, over, info

    # ----- Spaces -----
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(5)

    def observation_space(self) -> spaces.Dict:
        return spaces.Dict({
            "board": spaces.Box(low=0, high=1, shape=(self.consts.BOARD_HEIGHT, self.consts.BOARD_WIDTH), dtype=jnp.int32),
            "piece_type": spaces.Box(low=0, high=6, shape=(), dtype=jnp.int32),
            "pos": spaces.Box(low=0, high=max(self.consts.BOARD_HEIGHT, self.consts.BOARD_WIDTH), shape=(2,), dtype=jnp.int32),
            "rot": spaces.Box(low=0, high=3, shape=(), dtype=jnp.int32),
            "next_piece": spaces.Box(low=0, high=6, shape=(), dtype=jnp.int32),
        })

    def image_space(self) -> spaces.Box:
        return spaces.Box(low=0, high=255, shape=(self.consts.IMG_H, self.consts.IMG_W, 3), dtype=jnp.uint8)

    # ----- Public API -----
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jrandom.PRNGKey = None) -> Tuple[TetrisObservation, TetrisState]:
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
            last_action=jnp.int32(Action.NOOP)
        )
        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def step(self, state: TetrisState, action: chex.Array) -> Tuple[
        TetrisObservation, TetrisState, float, bool, TetrisInfo]:
        previous_state = state
        a = action.astype(jnp.int32)
        is_left = (a == Action.LEFT) | (a == Action.UPLEFT) | (a == Action.DOWNLEFT)
        is_right = (a == Action.RIGHT) | (a == Action.UPRIGHT) | (a == Action.DOWNRIGHT)
        is_up = (a == Action.UP) | (a == Action.UPLEFT) | (a == Action.UPRIGHT)
        is_down = (a == Action.DOWN) | (a == Action.DOWNLEFT) | (a == Action.DOWNRIGHT)
        is_fire = (a == Action.FIRE) | (a == Action.DOWNFIRE) | (a == Action.UPFIRE) \
                  | (a == Action.LEFTFIRE) | (a == Action.RIGHTFIRE) \
                  | (a == Action.UPLEFTFIRE) | (a == Action.UPRIGHTFIRE) \
                  | (a == Action.DOWNLEFTFIRE) | (a == Action.DOWNRIGHTFIRE)

        do_reset = (a == self.consts.RESET)

        tick_next = state.tick + jnp.int32(1)
        gravity_drop = (tick_next % jnp.int32(self.consts.GRAVITY_FRAMES) == 0).astype(jnp.int32)

        # Held-key timers
        new_move_dir = jnp.where(is_left, -1, jnp.where(is_right, 1, state.move_dir * 0))
        das = jnp.where(new_move_dir != 0,
                        jnp.where(state.move_dir == 0, jnp.int32(self.consts.DAS_FRAMES), jnp.maximum(0, state.das_timer - 1)),
                        jnp.int32(0))
        arr = jnp.where(new_move_dir != 0,
                        jnp.where(das == 0, jnp.maximum(0, state.arr_timer - 1), jnp.int32(self.consts.ARR_FRAMES)),
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
            lambda _: self.try_rotate_with_kick(state.board, state.piece_type, state.pos, state.rot),
            lambda _: (state.pos, state.rot),
            operand=None
        )
        state = state._replace(pos=pos_r, rot=rot_r)

        # Horizontal
        grid = self.piece_grid(state.piece_type, state.rot)
        pos_h = state.pos + jnp.array([0, jnp.int32(jnp.where(do_move_now, new_move_dir, 0))], dtype=jnp.int32)
        coll_h = self.check_collision(state.board, grid, pos_h)
        state = lax.cond(coll_h, lambda s: s, lambda s: s._replace(pos=pos_h), state)

        # Vertical
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
                lambda ss: self._lock_spawn(ss, grid, tick_next, soft_points = jnp.int32(0)),
                lambda ss: (ss._replace(pos=pos_v, tick=tick_next),
                            jnp.float32(0.0), jnp.bool_(False),
                            TetrisInfo(score=ss.score , cleared=jnp.int32(0), game_over=ss.game_over)),
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

        def after_over(ss):
            obs2, st2 = self.reset(ss.key)
            return st2, jnp.float32(0.0), jnp.bool_(False), TetrisInfo(score=st2.score, cleared=jnp.int32(0),
                                                                       game_over=jnp.bool_(False))

        state, _reward, done, info = lax.cond(state.game_over, after_over, lambda s: (s, _reward, done, info), state)

        state = state._replace(
            das_timer=das,
            arr_timer=arr,
            move_dir=new_move_dir,
            rot_timer=jnp.where(do_rotate_now, jnp.int32(self.consts.ROT_DAS_FRAMES), rot_timer),
            soft_timer=jnp.where(do_soft_now, jnp.int32(self.consts.SOFT_PACE_FRAMES), soft_timer),
            last_action=a
        )

        obs = self._get_observation(state)
        reward = self._get_reward(previous_state, state)
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
        # only reward when line is cleared, otherwise 0 
        cleared = state.score - previous_state.score
        return jnp.where(cleared > 0, cleared.astype(jnp.float32), jnp.float32(0.0))

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: TetrisState) -> bool:
        # unused (auto-restart), keep for API completeness
        return state.game_over

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: TetrisObservation) -> chex.Array:
        return jnp.concatenate([obs.board.flatten(), obs.piece_type[None], obs.pos, obs.rot[None], obs.next_piece[None]]).astype(jnp.int32)

    def render(self, state: TetrisState) -> jnp.ndarray:
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
        ) = self.load_sprites()

        self.N_COLOR_ROWS = int(self.SPRITE_ROW_COLORS.shape[0])

    def load_sprites(self):
        """Load all sprites required for Tetris rendering."""
        MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

        # Load sprites
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

        return (
            SPRITE_BG,
            SPRITE_BOARD,
            SCORE_DIGIT_SPRITES,
            SPRITE_ROW_COLORS
        )

    @partial(jax.jit, static_argnums=(0,))
    def get_piece_shape(self, piece_idx: chex.Array, rotation_idx: chex.Array) -> chex.Array:
        return self.consts.TETROMINOS[piece_idx, rotation_idx]

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        raster = jr.create_initial_frame(width=160, height=210)

        frame_bg = jr.get_sprite_frame(self.SPRITE_BG, 0)
        raster = jr.render_at(raster, 0, 0, frame_bg)

        frame_board = jr.get_sprite_frame(self.SPRITE_BOARD, 0)
        raster = jr.render_at(raster, self.consts.BOARD_X, self.consts.BOARD_Y, frame_board)

        board = state.board

        num_rows = board.shape[0] # 22
        num_cols = board.shape[1] # 10

        def render_board_row(row_idx, raster):
            row = board[row_idx]
            sprite = self.SPRITE_ROW_COLORS[row_idx % len(self.SPRITE_ROW_COLORS)]

            def render_col(col_idx, raster):
                val = row[col_idx]

                def draw_sprite(r):
                    x = self.consts.BOARD_X + self.consts.BOARD_PADDING + col_idx * (self.consts.CELL_WIDTH + 1)
                    y = self.consts.BOARD_Y + row_idx * (self.consts.CELL_HEIGHT + 1)
                    return jr.render_at(r, x, y, sprite)

                return jax.lax.cond(jnp.equal(val, 1), draw_sprite, lambda r: r, raster)

            return jax.lax.fori_loop(0, num_cols, render_col, raster)

        raster = jax.lax.fori_loop(0, num_rows, render_board_row, raster)

        piece = self.get_piece_shape(state.piece_type, state.rot)  # (4,4)
        pos_y, pos_x = state.pos

        def render_piece_cell(i, raster):
            y = i // 4
            x = i % 4
            val = piece[y, x]

            def draw_piece(r):
                board_y = pos_y + y
                board_x = pos_x + x

                in_bounds_y = jnp.logical_and(board_y >= 0, board_y < num_rows)
                in_bounds_x = jnp.logical_and(board_x >= 0, board_x < num_cols)
                in_bounds = jnp.logical_and(in_bounds_y, in_bounds_x)

                def render_pixel(r):
                    sprite = self.SPRITE_ROW_COLORS[board_y % self.N_COLOR_ROWS]
                    px = self.consts.BOARD_X + self.consts.BOARD_PADDING + board_x * (self.consts.CELL_WIDTH + 1)
                    py = self.consts.BOARD_Y + board_y * (self.consts.CELL_HEIGHT + 1)
                    return jr.render_at(r, px, py, sprite)

                return jax.lax.cond(in_bounds, render_pixel, lambda r: r, r)

            return jax.lax.cond(jnp.equal(val, 1), draw_piece, lambda r: r, raster)

        raster = jax.lax.fori_loop(0, 16, render_piece_cell, raster)

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
        return raster
