import os
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import NamedTuple, Tuple, Optional, Callable, Sequence

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as jr
import jaxatari.spaces as spaces


class SurroundConstants(NamedTuple):
    """Parameters defining the Surround grid and visuals."""

    # Playfield layout
    GRID_WIDTH: int = 40
    GRID_HEIGHT: int = 24

    # Mapping from grid cells to screen pixels
    CELL_SIZE: Tuple[int, int] = (4, 8)  # (width, height)

    # Atari-typische Bildschirmgröße (W,H)
    SCREEN_SIZE: Tuple[int, int] = (160, 210)

    # Colors
    P1_TRAIL_COLOR: Tuple[int, int, int] = (255, 102, 204)  # Border color
    P2_TRAIL_COLOR: Tuple[int, int, int] = (255, 102, 204)  # Border color
    BACKGROUND_COLOR: Tuple[int, int, int] = (153, 153, 255)  # Blau-Lila Hintergrund
    # Head colors (small square on top of the trail)
    P1_HEAD_COLOR: Tuple[int, int, int] = (221, 51, 136)    # yellow (score color)
    P2_HEAD_COLOR: Tuple[int, int, int] = (255, 221, 51)    # magenta (score color)
    HEAD_SCALE: float = 0.5  # fraction of the cell size (0< scale ≤1)

    # Border 
    BORDER_CELLS_X: int = 2    # linke/rechte Dicke in Zellen
    BORDER_CELLS_Y: int = 1    # obere/untere Dicke in Zellen
    BORDER_COLOR: Tuple[int, int, int] = (255, 102, 204)

    # Divider stripes (thin red lines across the middle of each occupied cell)
    DIVIDER_COLOR: Tuple[int, int, int] = (153, 153, 255)   # Match playfield background color
    DIVIDER_THICKNESS: int = 1  # pixels (in screen space)

    # Starting positions (x, y) - snapped to nearest rectangle (cell) on the field
    # These should be integers and not between cells. Adjusted to be inside the playfield, not on borders.
    # Middle of the playfield, within a rectangle (cell)
    # Set to the exact center row of the grid
    P1_START_POS: Tuple[int, int] = (4, 10)  # left side, vertical center
    P2_START_POS: Tuple[int, int] = (35, 10) # right side, vertical center

    # Starting directions
    P1_START_DIR: int = Action.RIGHT
    P2_START_DIR: int = Action.LEFT

    # Rules
    ALLOW_REVERSE: bool = False

    # Maximum number of environment steps before truncation
    MAX_STEPS: int = 1000

    # Game rules
    WIN_SCORE: int = 10

    # Rough logic rate control when caller steps at ~60 FPS
    # Move only every N calls to step (e.g., 60/4 = 15 for ~4 Hz)
    MOVE_EVERY_N_STEPS: int = 15


class SurroundState(NamedTuple):
    """Immutable game state container."""

    pos0: jnp.ndarray  # (x, y)
    pos1: jnp.ndarray  # (x, y)
    dir0: jnp.ndarray  # () int32
    dir1: jnp.ndarray  # () int32
    trail: jnp.ndarray  # (GRID_WIDTH, GRID_HEIGHT)
    border: jnp.ndarray  # (GRID_WIDTH, GRID_HEIGHT) bool mask
    terminated: jnp.ndarray  # () bool
    time: jnp.ndarray  # step counter
    score0: jnp.ndarray  # () int32
    score1: jnp.ndarray  # () int32
    # Round management
    pending_reset: jnp.ndarray  # () bool, perform reset on next step
    substep: jnp.ndarray  # () int32, frame-skip counter


class SurroundObservation(NamedTuple):
    """Observation returned to the agent."""

    grid: jnp.ndarray  # (GRID_WIDTH, GRID_HEIGHT) int32
    pos0: jnp.ndarray  # (2,) int32
    pos1: jnp.ndarray  # (2,) int32
    agent_id: jnp.ndarray  # () int32


class SurroundInfo(NamedTuple):
    """Additional environment information."""

    time: jnp.ndarray
    all_rewards: jnp.ndarray


def create_border_mask(consts: SurroundConstants) -> jnp.ndarray:
    mask = jnp.zeros((consts.GRID_WIDTH, consts.GRID_HEIGHT), dtype=jnp.bool_)
    bx, by = consts.BORDER_CELLS_X, consts.BORDER_CELLS_Y
    mask = mask.at[:bx, :].set(True)
    mask = mask.at[-bx:, :].set(True)
    mask = mask.at[:, :by].set(True)
    mask = mask.at[:, -by:].set(True)
    return mask


class SurroundRenderer(JAXGameRenderer):
    """Very small dummy renderer used for tests."""

    def __init__(self, consts: Optional[SurroundConstants] = None):
        consts = consts or SurroundConstants()
        super().__init__(consts)
        self.consts = consts

        module_dir = os.path.dirname(os.path.abspath(__file__))
        digit_path = os.path.join(module_dir, "sprites/seaquest/digits/{}" + ".npy")
        digits = jr.load_and_pad_digits(digit_path)
        p1_color = jnp.array(self.consts.P1_HEAD_COLOR, dtype=jnp.uint8)
        p2_color = jnp.array(self.consts.P2_HEAD_COLOR, dtype=jnp.uint8)
        self.p1_digits = digits.at[..., :3].set(jnp.where(digits[..., 3:] > 0, p1_color, 0))
        self.p2_digits = digits.at[..., :3].set(jnp.where(digits[..., 3:] > 0, p2_color, 0))

    def render(self, state: SurroundState) -> jnp.ndarray:  # pragma: no cover - visual
        bg = jnp.array(self.consts.BACKGROUND_COLOR, dtype=jnp.uint8)
        width, height = self.consts.SCREEN_SIZE
        img = jnp.ones((height, width, 3), dtype=jnp.uint8) * bg

        # Playfield-Geometrie
        cell_w, cell_h = self.consts.CELL_SIZE
        field_h = self.consts.GRID_HEIGHT * cell_h
        field_w = self.consts.GRID_WIDTH * cell_w
        slack = height - field_h
        y_off = (slack // cell_h) * cell_h  # snap offset to cell size for grid alignment

        playfield = jnp.ones((field_h, field_w, 3), dtype=jnp.uint8) * bg

        # Trails (upscale aus Zellen)
        def upscale(mask):
            return jnp.repeat(jnp.repeat(mask, cell_h, axis=0), cell_w, axis=1)

        p1_color = jnp.array(self.consts.P1_TRAIL_COLOR, dtype=jnp.uint8)
        p2_color = jnp.array(self.consts.P2_TRAIL_COLOR, dtype=jnp.uint8)

        p1_mask = upscale((state.trail == 1).T)[..., None]
        p2_mask = upscale((state.trail == 2).T)[..., None]
        playfield = jnp.where(p1_mask, p1_color, playfield)
        playfield = jnp.where(p2_mask, p2_color, playfield)

        # Border
        bx = self.consts.BORDER_CELLS_X * cell_w
        by = self.consts.BORDER_CELLS_Y * cell_h
        border_color = jnp.array(self.consts.BORDER_COLOR, dtype=jnp.uint8)
        playfield = playfield.at[:by, :, :].set(border_color)
        playfield = playfield.at[-by:, :, :].set(border_color)
        playfield = playfield.at[:, :bx, :].set(border_color)
        playfield = playfield.at[:, -bx:, :].set(border_color)

        # Divider stripes over trails and border (horizontal midline per cell)
        trail_any = upscale((state.trail != 0).T)
        border_up = upscale(state.border.T)
        occupied = jnp.logical_or(trail_any, border_up)
        ys = jnp.arange(field_h)
        mid = cell_h // 2
        band = (ys % cell_h >= mid) & (ys % cell_h < mid + max(1, self.consts.DIVIDER_THICKNESS))
        band_2d = jnp.broadcast_to(band[:, None], (field_h, field_w))
        divider_mask = jnp.logical_and(band_2d, occupied)[..., None]
        divider_col = jnp.array(self.consts.DIVIDER_COLOR, dtype=jnp.uint8)
        playfield = jnp.where(divider_mask, divider_col, playfield)

        # Köpfe (ohne Python-int()) — draw after divider so heads remain solid
        p1x = (state.pos0[0] * cell_w).astype(jnp.int32)
        p1y = (state.pos0[1] * cell_h).astype(jnp.int32)
        p2x = (state.pos1[0] * cell_w).astype(jnp.int32)
        p2y = (state.pos1[1] * cell_h).astype(jnp.int32)

        head_patch1 = jnp.ones((cell_h, cell_w, 3), dtype=jnp.uint8) * p1_color
        head_patch2 = jnp.ones((cell_h, cell_w, 3), dtype=jnp.uint8) * p2_color
        playfield = jax.lax.dynamic_update_slice(playfield, head_patch1, (p1y, p1x, 0))
        playfield = jax.lax.dynamic_update_slice(playfield, head_patch2, (p2y, p2x, 0))

        # ---- Head fills the entire cell, colored as in the score display ----
        head_patch1 = jnp.ones((cell_h, cell_w, 3), dtype=jnp.uint8) * jnp.array(self.consts.P1_HEAD_COLOR, dtype=jnp.uint8)
        head_patch2 = jnp.ones((cell_h, cell_w, 3), dtype=jnp.uint8) * jnp.array(self.consts.P2_HEAD_COLOR, dtype=jnp.uint8)
        playfield = jax.lax.dynamic_update_slice(playfield, head_patch1, (p1y, p1x, 0))
        playfield = jax.lax.dynamic_update_slice(playfield, head_patch2, (p2y, p2x, 0))
        # Playfield ins Bild
        img = img.at[y_off:y_off + field_h, :field_w, :].set(playfield)

        # Scores: directly above the box surrounding the playfield
        idx0 = jnp.clip(state.score0 % 10, 0, 9)
        idx1 = jnp.clip(state.score1 % 10, 0, 9)
        digit_p1 = jr.get_sprite_frame(self.p1_digits, idx0)
        digit_p2 = jr.get_sprite_frame(self.p2_digits, idx1)
        
        # Größer skalieren
        scale_digits = 2
        digit_p1 = jnp.kron(digit_p1, jnp.ones((scale_digits, scale_digits, 1), dtype=jnp.uint8))
        digit_p2 = jnp.kron(digit_p2, jnp.ones((scale_digits, scale_digits, 1), dtype=jnp.uint8))

        # Y-Position (wie gehabt)
        border_y = self.consts.BORDER_CELLS_Y * self.consts.CELL_SIZE[1]
        score_y = max(0, y_off + border_y - digit_p1.shape[0] - 8)

        # X-Position: näher zusammengerückt
        padding_x = 30
        img = jr.render_at(img, padding_x, score_y, digit_p1)
        img = jr.render_at(img, width - padding_x - digit_p2.shape[1], score_y, digit_p2)

        return img


class JaxSurround(
    JaxEnvironment[SurroundState, SurroundObservation, SurroundInfo, SurroundConstants]
):
    """A very small two player Surround implementation."""

    def __init__(
        self,
        consts: Optional[SurroundConstants] = None,
        reward_funcs: Optional[Sequence[Callable[[SurroundState, SurroundState], jnp.ndarray]]] = None,
    ):
        consts = consts or SurroundConstants()
        super().__init__(consts)
        self.renderer = SurroundRenderer(self.consts)
        self.action_set = [
            Action.NOOP,
            Action.FIRE,
            Action.UP,
            Action.RIGHT,
            Action.LEFT,
            Action.DOWN,
        ]
        # Wichtig: reward_funcs für _get_all_rewards speichern.
        # Bleibt während der Laufzeit statisch -> JAX-jit-freundlich.
        self.reward_funcs = reward_funcs

    # --- Internal AI helper for P1 (left player) ---

    def _dir_left(self, d: jnp.ndarray) -> jnp.ndarray:
        # Map für alle 0..5 (NOOP, FIRE, UP, RIGHT, LEFT, DOWN)
        left_map = jnp.array([
            0,  # NOOP -> NOOP
            1,  # FIRE -> FIRE
            4,  # UP   -> LEFT
            2,  # RIGHT-> UP
            5,  # LEFT -> DOWN
            3,  # DOWN -> RIGHT
        ], dtype=jnp.int32)
        return left_map[d]

    def _dir_right(self, d: jnp.ndarray) -> jnp.ndarray:
        right_map = jnp.array([
            0,  # NOOP -> NOOP
            1,  # FIRE -> FIRE
            3,  # UP   -> RIGHT
            5,  # RIGHT-> DOWN
            2,  # LEFT -> UP
            4,  # DOWN -> LEFT
        ], dtype=jnp.int32)
        return right_map[d]

    def _dir_offset(self, d: jnp.ndarray) -> jnp.ndarray:
        # Indexierbares Offsets-Array (NOOP, FIRE, UP, RIGHT, LEFT, DOWN)
        offsets = jnp.array([
            [0,  0],   # NOOP
            [0,  0],   # FIRE (keine Bewegung)
            [0, -1],   # UP
            [1,  0],   # RIGHT
            [-1, 0],   # LEFT
            [0,  1],   # DOWN
        ], dtype=jnp.int32)
        return offsets[d]

    def _is_blocked(self, state, pos_xy: jnp.ndarray, action_dir: jnp.ndarray) -> jnp.ndarray:
        off = self._dir_offset(action_dir)           # (2,)
        nxt = pos_xy + off                           # (2,)
        x, y = nxt[0], nxt[1]
        # Bounds-Check
        out = jnp.logical_or(
            jnp.logical_or(x < 0, x >= self.consts.GRID_WIDTH),
            jnp.logical_or(y < 0, y >= self.consts.GRID_HEIGHT),
        )
        # Grid-/Border-Check (nur prüfen, wenn inbounds)
        hit_border = jax.lax.cond(
            out, lambda: False, lambda: state.border[x, y]
        )
        hit_trail = jax.lax.cond(
            out, lambda: False, lambda: (state.trail[x, y] != 0)
        )
        return jnp.logical_or(out, jnp.logical_or(hit_border, hit_trail))

    def _opponent_policy(self, state) -> jnp.ndarray:
        """Wählt eine Richtung für P1: bevorzugt 'geradeaus', sonst links, dann rechts."""
        curr = state.dir0.astype(jnp.int32)

        keep  = curr
        left  = self._dir_left(curr)
        right = self._dir_right(curr)

        cand_dirs = jnp.stack([keep, left, right], axis=0)  # (3,)
        # Für jede Kandidatenrichtung prüfen, ob blockiert
        def chk(d):
            return self._is_blocked(state, state.pos0, d)
        blocked = jax.vmap(chk)(cand_dirs)                  # (3,) bool

        free_mask = jnp.logical_not(blocked).astype(jnp.int32)
        # Index des ersten True in free_mask bestimmen:
        # argmax liefert das erste Maximum; falls alles 0 ist → 0, das behandeln wir separat
        idx_first = jnp.argmax(free_mask)
        any_free = jnp.any(free_mask == 1)

        chosen = cand_dirs[idx_first]
        # Wenn keiner frei, NOOP
        noop = jnp.array(int(Action.NOOP), dtype=jnp.int32)
        return jax.lax.select(any_free, chosen, noop)


    def reset(
            self,
            key: Optional[jax.random.PRNGKey] = None,
            scores: Optional[Tuple[int, int]] = None,
        ) -> Tuple[SurroundObservation, SurroundState]:
            del key
            # Clamp start positions to inner playfield (never on border bricks)
            p0_start = jnp.array((
                jnp.clip(self.consts.P1_START_POS[0], self.consts.BORDER_CELLS_X, self.consts.GRID_WIDTH  - self.consts.BORDER_CELLS_X - 1),
                jnp.clip(self.consts.P1_START_POS[1], self.consts.BORDER_CELLS_Y, self.consts.GRID_HEIGHT - self.consts.BORDER_CELLS_Y - 1),
            ), dtype=jnp.int32)
            p1_start = jnp.array((
                jnp.clip(self.consts.P2_START_POS[0], self.consts.BORDER_CELLS_X, self.consts.GRID_WIDTH  - self.consts.BORDER_CELLS_X - 1),
                jnp.clip(self.consts.P2_START_POS[1], self.consts.BORDER_CELLS_Y, self.consts.GRID_HEIGHT - self.consts.BORDER_CELLS_Y - 1),
            ), dtype=jnp.int32)
            grid = jnp.zeros((self.consts.GRID_WIDTH, self.consts.GRID_HEIGHT), dtype=jnp.int32)
            border = create_border_mask(self.consts)

            # keep scores from previous round if provided
            if scores is None:
                s0 = jnp.array(0, dtype=jnp.int32)
                s1 = jnp.array(0, dtype=jnp.int32)
            else:
                s0 = jnp.array(int(scores[0]), dtype=jnp.int32)
                s1 = jnp.array(int(scores[1]), dtype=jnp.int32)

            state = SurroundState(
                p0_start,
                p1_start,
                jnp.array(self.consts.P1_START_DIR, dtype=jnp.int32),
                jnp.array(self.consts.P2_START_DIR, dtype=jnp.int32),
                grid,
                border,
                jnp.array(False, dtype=jnp.bool_),
                jnp.array(0, dtype=jnp.int32),
                s0,
                s1,
                jnp.array(False, dtype=jnp.bool_),  # pending_reset
                jnp.array(0, dtype=jnp.int32),      # substep
            )
            return self._get_observation(state), state    

    
    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, state: SurroundState, actions: jnp.ndarray | tuple | list
    ) -> Tuple[SurroundObservation, SurroundState, jnp.ndarray, bool, SurroundInfo]:
        """Advance the game by at most one logic move.
        - If only a single action is provided, it's interpreted as P2's (human) action;
          P1 (left) is controlled by a simple opponent policy.
        - On collision, we update the score and set `pending_reset=True`, but we do NOT set done.
          On the *next* call to step, we reset positions/trail while preserving scores.
        - `done` is True only when a player reaches WIN_SCORE or MAX_STEPS is exceeded.
        """
        # --- Helper: internal hard reset keeping scores ---
        def _round_reset(state: SurroundState):
            p0_start = jnp.array((
                jnp.clip(self.consts.P1_START_POS[0], self.consts.BORDER_CELLS_X, self.consts.GRID_WIDTH  - self.consts.BORDER_CELLS_X - 1),
                jnp.clip(self.consts.P1_START_POS[1], self.consts.BORDER_CELLS_Y, self.consts.GRID_HEIGHT - self.consts.BORDER_CELLS_Y - 1),
            ), dtype=jnp.int32)
            p1_start = jnp.array((
                jnp.clip(self.consts.P2_START_POS[0], self.consts.BORDER_CELLS_X, self.consts.GRID_WIDTH  - self.consts.BORDER_CELLS_X - 1),
                jnp.clip(self.consts.P2_START_POS[1], self.consts.BORDER_CELLS_Y, self.consts.GRID_HEIGHT - self.consts.BORDER_CELLS_Y - 1),
            ), dtype=jnp.int32)
            grid = jnp.zeros((self.consts.GRID_WIDTH, self.consts.GRID_HEIGHT), dtype=jnp.int32)
            return state._replace(
                pos0=p0_start,
                pos1=p1_start,
                dir0=jnp.array(self.consts.P1_START_DIR, dtype=jnp.int32),
                dir1=jnp.array(self.consts.P2_START_DIR, dtype=jnp.int32),
                trail=grid,
                terminated=jnp.array(False, dtype=jnp.bool_),
                pending_reset=jnp.array(False, dtype=jnp.bool_),
                # keep scores and time, reset substep so movement resumes deterministically
                substep=jnp.array(0, dtype=jnp.int32),
            )

        # If a reset was pending (we showed the scored frame), perform it now.
        state = jax.lax.cond(state.pending_reset, lambda: _round_reset(state), lambda: state)

        # --- Frame skip / logic gating ---
        # Only move every N steps to emulate a logic fps when the caller runs at high fps.
        substep = state.substep + 1
        do_logic = (substep % jnp.maximum(self.consts.MOVE_EVERY_N_STEPS, 1)) == 0

        # Parse action(s)
        actions = jnp.asarray(actions, dtype=jnp.int32)
        # Scalar -> treat as P2 only (human); compute AI for P1
        def _joint_from_scalar(a_scalar):
            ai = self._opponent_policy(state)
            return jnp.stack([ai, a_scalar], axis=0)
        def _joint_from_array(a_array):
            a_array = jnp.reshape(a_array, (-1,))
            return jnp.where(a_array.shape[0] == 2, a_array, _joint_from_scalar(a_array[0]))
        joint_action = jax.lax.cond(actions.ndim == 0, lambda: _joint_from_scalar(actions), lambda: _joint_from_array(actions))

        # Movement vectors for each direction
        offsets = jnp.array(
            [
                [0, 0],   # NOOP
                [0, 0],   # FIRE -> no movement
                [0, -1],  # UP
                [1, 0],   # RIGHT
                [-1, 0],  # LEFT
                [0, 1],   # DOWN
            ],
            dtype=jnp.int32,
        )

        # Update direction (no reverse if disallowed)
        def update_dir(curr_dir, action):
            is_move = jnp.logical_and(action >= Action.UP, action <= Action.DOWN)
            candidate = jax.lax.select(is_move, action, curr_dir)
            if not self.consts.ALLOW_REVERSE:
                opp = jnp.array([
                    Action.NOOP,   # NOOP
                    Action.NOOP,   # FIRE
                    Action.DOWN,   # UP -> DOWN
                    Action.LEFT,   # RIGHT -> LEFT
                    Action.RIGHT,  # LEFT -> RIGHT
                    Action.UP,     # DOWN -> UP
                ], dtype=jnp.int32)
                candidate = jax.lax.cond(candidate == opp[curr_dir], lambda: curr_dir, lambda: candidate)
            return candidate

        new_dir0 = update_dir(state.dir0, joint_action[0])
        new_dir1 = update_dir(state.dir1, joint_action[1])

        state_no_move = state._replace(dir0=new_dir0, dir1=new_dir1, substep=substep)

        def _skip_move(_):
            # Kein Logik-Tick: nur Blickrichtung aktualisiert zurückgeben
            obs = self._get_observation(state_no_move)
            # "Keine Bewegung" -> Rewards relativ zum alten Zustand berechnen
            all_rewards = self._get_all_rewards(state, state_no_move)
            info = self._get_info(state_no_move, all_rewards)
            reward = jnp.array(0, dtype=jnp.int32)
            done = jnp.array(False, dtype=jnp.bool_)
            return obs, state_no_move, reward, done, info

        def _logic_move(_):
            # --- ab hier deine bisherige "Logik-Tick"-Berechnung ---
            offsets = jnp.array(
                [
                    [0, 0],   # NOOP
                    [0, 0],   # FIRE
                    [0, -1],  # UP
                    [1, 0],   # RIGHT
                    [-1, 0],  # LEFT
                    [0, 1],   # DOWN
                ],
                dtype=jnp.int32,
            )

            # neue Positionen
            offset_p0 = offsets[new_dir0]
            offset_p1 = offsets[new_dir1]
            new_p0 = state.pos0 + offset_p0
            new_p1 = state.pos1 + offset_p1

            grid_w = self.consts.GRID_WIDTH
            grid_h = self.consts.GRID_HEIGHT

            def out_of_bounds(pos):
                return jnp.logical_or(
                    jnp.logical_or(pos[0] < 0, pos[0] >= grid_w),
                    jnp.logical_or(pos[1] < 0, pos[1] >= grid_h),
                )

            out0 = out_of_bounds(new_p0)
            out1 = out_of_bounds(new_p1)

            head_on = jnp.all(new_p0 == new_p1)

            hit_p0 = jax.lax.cond(
                out0,
                lambda: True,
                lambda: jnp.logical_or(state.border[tuple(new_p0)], state.trail[tuple(new_p0)] != 0),
            )
            hit_p1 = jax.lax.cond(
                out1,
                lambda: True,
                lambda: jnp.logical_or(state.border[tuple(new_p1)], state.trail[tuple(new_p1)] != 0),
            )
            hit_p0 = jnp.logical_or(hit_p0, head_on)
            hit_p1 = jnp.logical_or(hit_p1, head_on)

            # Trail aktualisieren (mit alten Köpfen)
            grid0 = state.trail.at[tuple(state.pos0)].set(1)
            grid = grid0.at[tuple(state.pos1)].set(2)

            # OOB -> Kopf bleibt zum Rendern stehen
            new_p0 = jax.lax.select(out0, state.pos0, new_p0)
            new_p1 = jax.lax.select(out1, state.pos1, new_p1)

            p0_only_crashed = jnp.logical_and(hit_p0, jnp.logical_not(hit_p1))
            p1_only_crashed = jnp.logical_and(hit_p1, jnp.logical_not(hit_p0))
            new_score0 = state.score0 + jnp.where(p1_only_crashed, 1, 0)
            new_score1 = state.score1 + jnp.where(p0_only_crashed, 1, 0)

            win_score = self.consts.WIN_SCORE
            game_over = jnp.logical_or(new_score0 >= win_score, new_score1 >= win_score)
            time_exceeded = (state.time + 1) >= self.consts.MAX_STEPS
            round_over = jnp.logical_or(hit_p0, hit_p1)

            next_state = state._replace(
                pos0=new_p0,
                pos1=new_p1,
                dir0=new_dir0,
                dir1=new_dir1,
                trail=grid,
                time=state.time + 1,
                score0=new_score0,
                score1=new_score1,
                terminated=jnp.array(False, dtype=jnp.bool_),
                pending_reset=jnp.logical_and(round_over, jnp.logical_not(game_over)),
                substep=jnp.array(0, dtype=jnp.int32),
            )

            reward = self._get_reward(state, next_state)
            done = jnp.logical_or(game_over, time_exceeded)
            next_state = next_state._replace(terminated=jnp.array(done, dtype=jnp.bool_))

            obs = self._get_observation(next_state)
            info = self._get_info(next_state, self._get_all_rewards(state, next_state))
            return obs, next_state, reward, done, info

        # WICHTIG: JAX-kompatible Verzweigung ohne Python-`if`
        obs, next_state, reward, done, info = jax.lax.cond(
            do_logic, _logic_move, _skip_move, operand=None
        )
        return obs, next_state, reward, done, info



    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: SurroundState) -> SurroundObservation:
        grid = state.trail
        grid = grid.at[tuple(state.pos0)].set(1)
        grid = grid.at[tuple(state.pos1)].set(2)
        return SurroundObservation(
            grid=grid,
            pos0=state.pos0.astype(jnp.int32),
            pos1=state.pos1.astype(jnp.int32),
            agent_id=jnp.array(0, dtype=jnp.int32),
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: SurroundState, all_rewards: jnp.ndarray) -> SurroundInfo:
        return SurroundInfo(time=state.time, all_rewards=all_rewards)   

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: SurroundState, state: SurroundState) -> jnp.ndarray:
        previous_diff = previous_state.score0 - previous_state.score1
        diff = state.score0 - state.score1
        return diff - previous_diff
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_all_rewards(self, previous_state: SurroundState, state: SurroundState) -> jnp.ndarray:
        # Statische Verzweigung: self ist per static_argnums=(0,) statisch.
        if self.reward_funcs is None:
            return jnp.zeros((1,), dtype=jnp.float32)
        # reward_funcs ist eine Sequenz von Callables: rf(prev, curr) -> scalar/array
        rewards = [rf(previous_state, state) for rf in self.reward_funcs]
        return jnp.asarray(rewards, dtype=jnp.float32)

    
    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: SurroundState) -> jnp.ndarray:
        reached_score = jnp.logical_or(state.score0 >= self.consts.WIN_SCORE, state.score1 >= self.consts.WIN_SCORE)
        time_exceeded = state.time >= self.consts.MAX_STEPS
        done = jnp.logical_or(reached_score, time_exceeded)
        return done.astype(jnp.bool_)


    def action_space(self) -> spaces.Discrete:
        """Returns the action space for the controllable player."""
        return spaces.Discrete(len(self.action_set))

    def observation_space(self) -> spaces.Dict:
        # Prefer per-dimension bounds; fall back to scalar bounds if unsupported by spaces.Box
        try:
            pos_low = jnp.array([0, 0], dtype=jnp.int32)
            pos_high = jnp.array([self.consts.GRID_WIDTH - 1, self.consts.GRID_HEIGHT - 1], dtype=jnp.int32)
            pos_box0 = spaces.Box(low=pos_low, high=pos_high, shape=(2,), dtype=jnp.int32)
            pos_box1 = spaces.Box(low=pos_low, high=pos_high, shape=(2,), dtype=jnp.int32)
        except Exception:
            pos_box0 = spaces.Box(0, self.consts.GRID_WIDTH, shape=(2,), dtype=jnp.int32)
            pos_box1 = spaces.Box(0, self.consts.GRID_WIDTH, shape=(2,), dtype=jnp.int32)

        return spaces.Dict({
            "grid": spaces.Box(
                low=0,
                high=2,
                shape=(self.consts.GRID_WIDTH, self.consts.GRID_HEIGHT),
                dtype=jnp.int32,
            ),
            "pos0": pos_box0,
            "pos1": pos_box1,
            "agent_id": spaces.Box(0, 1, shape=(), dtype=jnp.int32),
        })

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.consts.SCREEN_SIZE[1], self.consts.SCREEN_SIZE[0], 3),
            dtype=jnp.uint8,
        )

    def render(self, state: SurroundState) -> jnp.ndarray:
        return self.renderer.render(state)

    def obs_to_flat_array(self, obs: SurroundObservation) -> jnp.ndarray:
        flat = [obs.grid.reshape(-1), obs.pos0.reshape(-1), obs.pos1.reshape(-1), jnp.array([obs.agent_id], dtype=jnp.int32)]
        return jnp.concatenate(flat).astype(jnp.int32)


def _pygame_action() -> int:
    """Map pressed keys to a Surround action."""
    import pygame

    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        return Action.UP
    if keys[pygame.K_RIGHT]:
        return Action.RIGHT
    if keys[pygame.K_LEFT]:
        return Action.LEFT
    if keys[pygame.K_DOWN]:
        return Action.DOWN
    if keys[pygame.K_SPACE]:
        return Action.FIRE
    return Action.NOOP


def main():
    import pygame
    import jax.numpy as jnp
    from jaxatari.environment import JAXAtariAction as Action
    from jaxatari.games.jax_surround import JaxSurround

    env = JaxSurround()
    _obs, state = env.reset()

    pygame.init()
    scale = 4
    W, H = env.consts.SCREEN_SIZE
    screen = pygame.display.set_mode((W * scale, H * scale))
    pygame.display.set_caption("JAX Surround")
    clock = pygame.time.Clock()

    # ---------- WICHTIG: JIT WARMUP ----------
    # Einmal step + render ausführen, damit JIT vor Spielstart kompiliert.
    warmup_action = jnp.array([Action.NOOP, Action.NOOP], dtype=jnp.int32)
    _o, state, _r, _d, _i = env.step(state, warmup_action)
    _ = env.render(state)
    clock.tick(0)     # dt zurücksetzen
    # -----------------------------------------

    LOGIC_HZ = 4                # 4 Zellen pro Sekunde
    RENDER_HZ = 60
    STEP_MS = 1000 // LOGIC_HZ
    acc_ms = 0
    running = True
    latest_action = Action.NOOP

    while running:
        # feste Render-FPS
        dt = clock.tick(RENDER_HZ)
        acc_ms += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Eingabe (immer lesen, aber erst beim nächsten Logikstep anwenden)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            latest_action = Action.UP
        elif keys[pygame.K_RIGHT]:
            latest_action = Action.RIGHT
        elif keys[pygame.K_LEFT]:
            latest_action = Action.LEFT
        elif keys[pygame.K_DOWN]:
            latest_action = Action.DOWN
        elif keys[pygame.K_SPACE]:
            latest_action = Action.FIRE

        # ---- feste Logikrate: max. 1 Step pro Frame (Clamping) ----
        if acc_ms >= STEP_MS:
            acc_ms -= STEP_MS
            joint_action = jnp.array([latest_action, Action.NOOP], dtype=jnp.int32)
            _obs, state, reward, done, _info = env.step(state, joint_action)
            if bool(done):
                _obs, state = env.reset()
                latest_action = Action.NOOP
                acc_ms = 0
        # -----------------------------------------------------------
if __name__ == "__main__":
    main()
