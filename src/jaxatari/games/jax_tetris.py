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
    def __init__(self, consts: TetrisConstants = None, reward_funcs: list[callable]=None):
        consts = consts or TetrisConstants()
        super().__init__(consts)
        self.renderer = TetrisRenderer(self.consts)
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

        def body(i, acc):
            y, x = jnp.divmod(i, 4)
            on = grid4[y, x] == 1
            py = pos[0] + y
            px = pos[1] + x
            inb_side = (px >= 0) & (px < W) & (py < H)
            readable = inb_side & (py >= 0)
            pyc = jnp.clip(py, 0, H - 1)
            pxc = jnp.clip(px, 0, W - 1)
            occ = jnp.where(readable, board[pyc, pxc] == 1, False)  # Is cell already filled?
            side_or_bottom_oob = on & (~inb_side) & (py >= 0)  # Out of bounds at side or bottom
            return acc | (on & (occ | side_or_bottom_oob))  # check Any collision?

        return lax.fori_loop(0, 16, body, False)

    @partial(jax.jit, static_argnums=0)
    def lock_piece(self, board: chex.Array, grid4: chex.Array, pos: chex.Array) -> chex.Array:
        """
        Lock the current tetromino onto the board at the given position.
        """
        H = jnp.int32(self.consts.BOARD_HEIGHT)
        W = jnp.int32(self.consts.BOARD_WIDTH)

        def body(i, b):
            y, x = jnp.divmod(i, 4)
            on = grid4[y, x] == 1
            py = pos[0] + y
            px = pos[1] + x
            inb = (px >= 0) & (px < W) & (py >= 0) & (py < H)  # In board
            pyc = jnp.clip(py, 0, H - 1)
            pxc = jnp.clip(px, 0, W - 1)
            # Set cell to 1 if in bounds and occupied by piece
            return lax.cond(on & inb, lambda bb: bb.at[pyc, pxc].set(1), lambda bb: bb, b)

        return lax.fori_loop(0, 16, body, board)

    @partial(jax.jit, static_argnums=0)
    def clear_lines(self, board: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """
        Clear all full lines from the board.
        Returns the new board and the number of lines cleared.
        """
        full = jnp.all(board == 1, axis=1)  # check which rows are full

        def scan_row(carry, y):
            nb, wy, cnt = carry
            row = board[y]
            is_full = full[y]

            def write_row(c):
                _nb, _wy, _cnt = c
                _nb = _nb.at[_wy].set(row)  # Copy row down
                return _nb, _wy - 1, _cnt

            def skip_row(c):
                _nb, _wy, _cnt = c
                return _nb, _wy, _cnt + 1  # Count cleared line

            return lax.cond(~is_full, write_row, skip_row, (nb, wy, cnt)), None

        init = (jnp.zeros_like(board), jnp.int32(self.consts.BOARD_HEIGHT - 1), jnp.int32(0))
        (nb, _wy, cleared), _ = lax.scan(scan_row, init, jnp.arange(self.consts.BOARD_HEIGHT - 1, -1, -1))
        return nb, cleared

    @partial(jax.jit, static_argnums=0)
    def try_rotate(self, board: chex.Array, piece_type: chex.Array, pos: chex.Array, rot: chex.Array):
        """
        Try to rotate the tetromino in place (no wall kick).
        Returns the new position and rotation if successful, otherwise the original.
        """
        new_rot = (rot + 1) & 3  # Next rotation
        rotated_grid = self.piece_grid(piece_type, new_rot)  # Rotated grid
        original_pos = pos  # Try original position
        can_rotate_in_place = ~self.check_collision(board, rotated_grid, original_pos)  # Is rotation valid?
        pos_final = jnp.where(can_rotate_in_place, original_pos, pos)
        rot_final = jnp.where(can_rotate_in_place, new_rot, rot)
        return pos_final, rot_final

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

        do_move_now = (new_move_dir != 0) & ((state.move_dir == 0) | ((das == 0) & (arr == 0)))

        rot_timer = jnp.where(is_up, jnp.maximum(0, state.rot_timer - 1), jnp.int32(0))
        do_rotate_now = is_up & ((state.last_action != Action.UP) | (rot_timer == 0))

        soft_timer = jnp.where(is_down, jnp.maximum(0, state.soft_timer - 1), jnp.int32(0))
        do_soft_now = is_down & (soft_timer == 0)

        do_hard_now = is_fire & (state.last_action != Action.FIRE)

        # Rotate
        pos_r, rot_r = lax.cond(
            do_rotate_now,
            lambda _: self.try_rotate(state.board, state.piece_type, state.pos, state.rot),
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

        state, reward, done, info = lax.cond(state.game_over, after_over, lambda s: (s, _reward, done, info), state)

        next_banner_timer = jnp.maximum(0, state.banner_timer - 1) # new row for banners

        state = state._replace(
            das_timer=das,
            arr_timer=arr,
            move_dir=new_move_dir,
            rot_timer=jnp.where(do_rotate_now, jnp.int32(self.consts.ROT_DAS_FRAMES), rot_timer),
            soft_timer=jnp.where(do_soft_now, jnp.int32(self.consts.SOFT_PACE_FRAMES), soft_timer),
            last_action=a,
            banner_timer = next_banner_timer, # new row for banners
            banner_code = jnp.where(next_banner_timer == 0, jnp.int32(0), state.banner_code) #new row for banners
        )

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


        ######################## I'M NOT SURE IF WE HAVE TO IMPLEMENT BANNERS LIKE THIS ####################

        one = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/tetris/text_one.npy"))
        two = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/tetris/text_two.npy"))
        three = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/tetris/text_triple.npy"))
        tetris = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/tetris/text_tetris.npy"))


        # uint8 -> everything into 8 bits. It also works without the conversion so idk
        # SPRITE_ONE = jnp.expand_dims(one.astype(jnp.uint8), axis=0)

        #
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