import os
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import NamedTuple, Tuple, Optional, Callable, Sequence
from flax import struct

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
import jaxatari.spaces as spaces
from jaxatari.modification import AutoDerivedConstants

def _get_default_asset_config() -> tuple:
    """
    Returns the default declarative asset manifest for Surround.
    Kept immutable (tuple of dicts) to fit NamedTuple defaults.
    Note: Surround uses mostly procedural sprites, so this is empty.
    """
    return ()

class SurroundConstants(AutoDerivedConstants):
    """Parameters defining the Surround grid and visuals."""

    # Playfield layout
    GRID_WIDTH: int = struct.field(pytree_node=False, default=40)
    GRID_HEIGHT: int = struct.field(pytree_node=False, default=24)

    # Mapping from grid cells to screen pixels
    CELL_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: (4, 8))  # (width, height)

    # Atari-typische Bildschirmgröße (W,H)
    SCREEN_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: (160, 210))

    # Colors
    P1_TRAIL_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default_factory=lambda: (255, 102, 204))  # Border color
    P2_TRAIL_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default_factory=lambda: (255, 102, 204))  # Border color
    BACKGROUND_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default_factory=lambda: (153, 153, 255))  # Blau-Lila Hintergrund
    # Head colors (small square on top of the trail)
    P1_HEAD_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default_factory=lambda: (221, 51, 136))    # yellow (score color)
    P2_HEAD_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default_factory=lambda: (255, 221, 51))    # magenta (score color)
    HEAD_SCALE: float = struct.field(pytree_node=False, default=0.5)  # fraction of the cell size (0< scale ≤1)

    # Border 
    BORDER_CELLS_X: int = struct.field(pytree_node=False, default=2)    # linke/rechte Dicke in Zellen
    BORDER_CELLS_Y: int = struct.field(pytree_node=False, default=1)    # obere/untere Dicke in Zellen
    BORDER_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default_factory=lambda: (255, 102, 204))

    # Divider stripes (thin red lines across the middle of each occupied cell)
    DIVIDER_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default_factory=lambda: (153, 153, 255))   # Match playfield background color
    DIVIDER_THICKNESS: int = struct.field(pytree_node=False, default=1)  # pixels (in screen space)

    # Starting positions (x, y) - snapped to nearest rectangle (cell) on the field
    # These should be integers and not between cells. Adjusted to be inside the playfield, not on borders.
    # Middle of the playfield, within a rectangle (cell)
    # Set to the exact center row of the grid
    P1_START_POS: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: (4, 10))  # left side, vertical center
    P2_START_POS: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: (35, 10)) # right side, vertical center

    # Starting directions
    P1_START_DIR: int = struct.field(pytree_node=False, default=Action.RIGHT)
    P2_START_DIR: int = struct.field(pytree_node=False, default=Action.LEFT)

    # Rules
    ALLOW_REVERSE: bool = struct.field(pytree_node=False, default=False)

    # Maximum number of environment steps before truncation
    MAX_STEPS: int = struct.field(pytree_node=False, default=1000)

    # Game rules
    WIN_SCORE: int = struct.field(pytree_node=False, default=10)

    # Rough logic rate control when caller steps at ~60 FPS
    # Move only every N calls to step (e.g., 60/4 = 15 for ~4 Hz)
    MOVE_EVERY_N_STEPS: int = struct.field(pytree_node=False, default=15)

    # Asset config baked into constants (immutable default) for asset overrides
    ASSET_CONFIG: tuple = struct.field(pytree_node=False, default=_get_default_asset_config())
    # --- Speed-up schedule (game accelerates over time) ---
    # Every SPEEDUP_STEPS logic ticks, reduce the effective MOVE_EVERY by SPEEDUP_DELTA, but not below MIN_MOVE_EVERY.
    # SPEEDUP_STEPS: int = 200          # how many logic ticks until next speed bump
    # SPEEDUP_DELTA: int = 1            # how much to reduce the period each bump
    # MIN_MOVE_EVERY_N_STEPS: int = 4   # lower bound (higher speed)


@struct.dataclass
class SurroundState:
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


@struct.dataclass
class SurroundObservation:
    """Observation returned to the agent."""

    grid: jnp.ndarray  # (GRID_WIDTH, GRID_HEIGHT) int32
    pos0: jnp.ndarray  # (2,) int32
    pos1: jnp.ndarray  # (2,) int32
    agent_id: jnp.ndarray  # () int32


@struct.dataclass
class SurroundInfo:
    """Additional environment information."""

    step_counter: jnp.ndarray


def create_border_mask(consts: SurroundConstants) -> jnp.ndarray:
    mask = jnp.zeros((consts.GRID_WIDTH, consts.GRID_HEIGHT), dtype=jnp.bool_)
    bx, by = consts.BORDER_CELLS_X, consts.BORDER_CELLS_Y
    mask = mask.at[:bx, :].set(True)
    mask = mask.at[-bx:, :].set(True)
    mask = mask.at[:, :by].set(True)
    mask = mask.at[:, -by:].set(True)
    return mask


class JaxSurround(
    JaxEnvironment[SurroundState, SurroundObservation, SurroundInfo, SurroundConstants]
):
    """A very small two player Surround implementation."""
    
    # Minimal ALE action set for Surround (from scripts/action_space_helper.py)
    # Note: FIRE is NOT in the ALE action set for this game
    ACTION_SET: jnp.ndarray = jnp.array(
        [
            Action.NOOP,
            Action.UP,
            Action.RIGHT,
            Action.LEFT,
            Action.DOWN,
        ],
        dtype=jnp.int32,
    )

    def __init__(
        self,
        consts: Optional[SurroundConstants] = None,
    ):
        consts = consts or SurroundConstants()
        super().__init__(consts)
        self.renderer = SurroundRenderer(self.consts)

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

    def _open_steps_ahead(self, state, pos_xy: jnp.ndarray, action_dir: jnp.ndarray, k: int = 3) -> jnp.ndarray:
        """Lightweight Heuristik: zähle freie Felder in Richtung action_dir bis max. k."""
        off = self._dir_offset(action_dir)  # (2,)
        def step_once(carry, _):
            p = carry
            nxt = p + off
            x, y = nxt[0], nxt[1]
            out = jnp.logical_or(
                jnp.logical_or(x < 0, x >= self.consts.GRID_WIDTH),
                jnp.logical_or(y < 0, y >= self.consts.GRID_HEIGHT),
            )
            # belegt?
            occ = jax.lax.cond(
                out, lambda: True, lambda: (state.border[x, y] | (state.trail[x, y] != 0))
            )
            nxt_ok = jnp.logical_not(jnp.logical_or(out, occ))
            # Wenn blockiert, bleibe stehen (keine weiteren Felder)
            nxt_pos = jax.lax.select(nxt_ok, nxt, p)
            return nxt_pos, nxt_ok.astype(jnp.int32)
        _, seen = jax.lax.scan(step_once, pos_xy, xs=None, length=k)
        return jnp.sum(seen).astype(jnp.int32)

    def _neighbor_free_count(self, state, pos_xy: jnp.ndarray) -> jnp.ndarray:
        """Zähle freie Nachbarzellen (UP/RIGHT/LEFT/DOWN)."""
        dirs = jnp.array([Action.UP, Action.RIGHT, Action.LEFT, Action.DOWN], dtype=jnp.int32)
        def free(d):
            return jnp.logical_not(self._is_blocked(state, pos_xy, d)).astype(jnp.int32)
        return jnp.sum(jax.vmap(free)(dirs))
    
    def _opponent_policy(self, state) -> jnp.ndarray:
        """Heuristik: keep > left > right, aber mit Scoring:
        - blockiert = sehr schlecht
        - mehr freie Schritte voraus (k=3) = besser
        - mehr freie Nachbarn nach dem Zug = besser
        - kleine Präferenz für 'geradeaus' (ruhiges Fahren)"""
        curr  = state.dir0.astype(jnp.int32)
        keep  = curr
        left  = self._dir_left(curr)
        right = self._dir_right(curr)
        cand_dirs = jnp.stack([keep, left, right], axis=0)  # (3,)

        def score_dir(d):
            blocked = self._is_blocked(state, state.pos0, d)
            # Position nach einer Schritt- hypothetisch
            nxt = state.pos0 + self._dir_offset(d)
            # Feature 1: freie Schritte voraus
            open_ahead = self._open_steps_ahead(state, state.pos0, d, k=3)
            # Feature 2: lokale Bewegungsfreiheit am Ziel
            local_free = self._neighbor_free_count(state, nxt)
            # Basis-Score
            base = 0
            base += 10 * open_ahead
            base += 2 * local_free
            # leichte Präferenz geradeaus
            base += jnp.where(d == keep, 1, 0)
            # harter Penalty wenn blockiert
            return jnp.where(blocked, -1_000_000, base)
        
        scores = jax.vmap(score_dir)(cand_dirs)  # (3,)
        # Bei Punktegleichheit Reihenfolge: keep, left, right
        # -> wir fügen winzige Tiebreaker abnehmend hinzu
        tie = jnp.array([3e-4, 2e-4, 1e-4], dtype=jnp.float32)
        scores = scores.astype(jnp.float32) + tie
        idx = jnp.argmax(scores)
        choice = cand_dirs[idx]
        # Falls wirklich alles blockiert -> NOOP
        all_blocked = jnp.all(scores < -999999)
        return jax.lax.select(all_blocked, jnp.array(int(Action.NOOP), jnp.int32), choice)


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
            return state.replace(
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

        # --- Frame skip / logic gating (+ dynamic speed-up) ---
        # Effective period decreases over time according to SPEEDUP_* constants.
        substep = state.substep + 1
        # bumps = (state.time // jnp.maximum(self.consts.SPEEDUP_STEPS, 1)) * self.consts.SPEEDUP_DELTA
        # eff_period = self.consts.MOVE_EVERY_N_STEPS - bumps
        # eff_period = jnp.maximum(self.consts.MIN_MOVE_EVERY_N_STEPS, eff_period)
        # eff_period = jnp.maximum(eff_period, 1)
        # do_logic = (substep % eff_period) == 0
        do_logic = (substep % jnp.maximum(self.consts.MOVE_EVERY_N_STEPS, 1)) == 0


        # Parse action(s) - translate compact agent action indices to ALE console actions
        actions = jnp.asarray(actions, dtype=jnp.int32)
        atari_actions = jnp.take(self.ACTION_SET, actions)
        
        # Scalar -> treat as P2 only (human); compute AI for P1
        def _joint_from_scalar(a_scalar):
            ai = self._opponent_policy(state)
            return jnp.stack([ai, a_scalar], axis=0)
        def _joint_from_array(a_array):
            a_array = jnp.reshape(a_array, (-1,))
            return jnp.where(a_array.shape[0] == 2, a_array, _joint_from_scalar(a_array[0]))
        joint_action = jax.lax.cond(atari_actions.ndim == 0, lambda: _joint_from_scalar(atari_actions), lambda: _joint_from_array(atari_actions))

        # Movement vectors for each direction (mapped to Action constants)
        # Index mapping: 0=NOOP, 1=UP, 2=RIGHT, 3=LEFT, 4=DOWN
        offsets = jnp.array(
            [
                [0, 0],   # NOOP
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
                # Map Action constants to their opposites (for reverse prevention)
                # Check if candidate is opposite of current direction
                is_opposite = jnp.logical_or(
                    jnp.logical_and(curr_dir == Action.UP, candidate == Action.DOWN),
                    jnp.logical_or(
                        jnp.logical_and(curr_dir == Action.DOWN, candidate == Action.UP),
                        jnp.logical_or(
                            jnp.logical_and(curr_dir == Action.LEFT, candidate == Action.RIGHT),
                            jnp.logical_and(curr_dir == Action.RIGHT, candidate == Action.LEFT)
                        )
                    )
                )
                candidate = jax.lax.cond(is_opposite, lambda: curr_dir, lambda: candidate)
            return candidate

        new_dir0 = update_dir(state.dir0, joint_action[0])
        new_dir1 = update_dir(state.dir1, joint_action[1])

        state_no_move = state.replace(dir0=new_dir0, dir1=new_dir1, substep=substep)

        def _skip_move(_):
            # Kein Logik-Tick: nur Blickrichtung aktualisiert zurückgeben
            obs = self._get_observation(state_no_move)
            # "Keine Bewegung" -> Rewards relativ zum alten Zustand berechnen
            info = self._get_info(state_no_move)
            reward = jnp.array(0, dtype=jnp.int32)
            done = jnp.array(False, dtype=jnp.bool_)
            return obs, state_no_move, reward, done, info

        def _logic_move(_):
            # --- ab hier deine bisherige "Logik-Tick"-Berechnung ---
            # Use _dir_offset which correctly maps Action constants to offsets
            # neue Positionen
            offset_p0 = self._dir_offset(new_dir0)
            offset_p1 = self._dir_offset(new_dir1)
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

            # 1) Kopf auf dasselbe Feld
            head_on = jnp.all(new_p0 == new_p1)
            # 2) Swap-Collision (Spieler tauschen die Felder im selben Tick)
            swap_collision = jnp.all(new_p0 == state.pos1) & jnp.all(new_p1 == state.pos0)

            # Basistreffer (Border/Trail oder OOB), noch ohne head_on/swap
            base_hit_p0 = jax.lax.cond(
                out0,
                lambda: True,
                lambda: jnp.logical_or(state.border[tuple(new_p0)], state.trail[tuple(new_p0)] != 0),
            )
            base_hit_p1 = jax.lax.cond(
                out1,
                lambda: True,
                lambda: jnp.logical_or(state.border[tuple(new_p1)], state.trail[tuple(new_p1)] != 0),
            )

            # Endgültige Treffer (funktional, ohne spätere += / Re-Zuweisung)
            hit_extra = jnp.logical_or(head_on, swap_collision)
            hit_p0 = jnp.logical_or(base_hit_p0, hit_extra)
            hit_p1 = jnp.logical_or(base_hit_p1, hit_extra)


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

            next_state = state.replace(
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
            next_state = next_state.replace(terminated=jnp.array(done, dtype=jnp.bool_))

            obs = self._get_observation(next_state)
            info = self._get_info(next_state)
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
    def _get_info(self, state: SurroundState) -> SurroundInfo:
        return SurroundInfo(step_counter=state.time)


    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: SurroundState, state: SurroundState) -> jnp.ndarray:
        previous_diff = previous_state.score0 - previous_state.score1
        diff = state.score0 - state.score1
        return diff - previous_diff

    
    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: SurroundState) -> jnp.ndarray:
        reached_score = jnp.logical_or(state.score0 >= self.consts.WIN_SCORE, state.score1 >= self.consts.WIN_SCORE)
        time_exceeded = state.time >= self.consts.MAX_STEPS
        done = jnp.logical_or(reached_score, time_exceeded)
        return done.astype(jnp.bool_)


    def action_space(self) -> spaces.Discrete:
        """Returns the action space for the controllable player."""
        return spaces.Discrete(len(self.ACTION_SET))

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


class SurroundRenderer(JAXGameRenderer):
    def __init__(self, consts: Optional[SurroundConstants] = None):
        super().__init__()
        self.consts = consts or SurroundConstants()
        self.config = render_utils.RendererConfig(
            game_dimensions=(self.consts.SCREEN_SIZE[1], self.consts.SCREEN_SIZE[0]),
            channels=3,
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)

        self.P1_HEAD_COLOR_TUPLE = (214, 214, 42)    # Yellow
        self.P2_HEAD_COLOR_TUPLE = (198, 89, 179)    # Red/Pink
        self.PLAYFIELD_COLOR_TUPLE = (181, 119, 181) # Lavender
        self.BORDER_COLOR_TUPLE = (214, 92, 92)      # Pink
        self.DIVIDER_COLOR_TUPLE = (142, 142, 142)   # Grey

        # 1. Start from (possibly modded) asset config provided via constants
        final_asset_config = list(self.consts.ASSET_CONFIG)
        
        # 2. Create procedural assets using modded constants
        procedural_sprites = {
            'p1_head': jnp.array(list(self.P1_HEAD_COLOR_TUPLE) + [255], dtype=jnp.uint8).reshape(1, 1, 4),
            'p2_head': jnp.array(list(self.P2_HEAD_COLOR_TUPLE) + [255], dtype=jnp.uint8).reshape(1, 1, 4),
            'p1_trail': jnp.array(list(self.consts.P1_TRAIL_COLOR) + [255], dtype=jnp.uint8).reshape(1, 1, 4),
            'p2_trail': jnp.array(list(self.consts.P2_TRAIL_COLOR) + [255], dtype=jnp.uint8).reshape(1, 1, 4),
            'border': jnp.array(list(self.BORDER_COLOR_TUPLE) + [255], dtype=jnp.uint8).reshape(1, 1, 4),
            'divider': jnp.array(list(self.DIVIDER_COLOR_TUPLE) + [255], dtype=jnp.uint8).reshape(1, 1, 4),
            'playfield': jnp.array(list(self.PLAYFIELD_COLOR_TUPLE) + [255], dtype=jnp.uint8).reshape(1, 1, 4),
        }

        # Create full-screen background using BACKGROUND_COLOR from constants
        bg_color = jnp.array(list(self.consts.BACKGROUND_COLOR) + [255], dtype=jnp.uint8)
        background_rgba = jnp.tile(bg_color.reshape(1, 1, 4), (self.consts.SCREEN_SIZE[1], self.consts.SCREEN_SIZE[0], 1))
        final_asset_config.append({'name': 'background', 'type': 'background', 'data': background_rgba})

        for name, data in procedural_sprites.items():
            final_asset_config.append({'name': name, 'type': 'procedural', 'data': data})

        module_dir = os.path.dirname(os.path.abspath(__file__))
        digit_path = os.path.join(module_dir, "sprites/seaquest/digits/{}" + ".npy")
        base_digits = self.jr.load_and_pad_digits(digit_path)
        scale = 2
        scaled_digits_list = [jnp.kron(d, jnp.ones((scale, scale, 1), dtype=jnp.uint8)) for d in base_digits]
        scaled_digits = jnp.stack(scaled_digits_list)

        alpha_mask = scaled_digits[..., 3:] > 128
        p1_digits_rgba = jnp.concatenate([jnp.where(alpha_mask, jnp.array(self.P1_HEAD_COLOR_TUPLE), 0), scaled_digits[..., 3:]], axis=-1)
        p2_digits_rgba = jnp.concatenate([jnp.where(alpha_mask, jnp.array(self.P2_HEAD_COLOR_TUPLE), 0), scaled_digits[..., 3:]], axis=-1)

        final_asset_config.append({'name': 'p1_digits', 'type': 'procedural', 'data': p1_digits_rgba})
        final_asset_config.append({'name': 'p2_digits', 'type': 'procedural', 'data': p2_digits_rgba})
        
        # 3. Load all assets, create palette, and generate ID masks
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS,
        ) = self.jr.load_and_setup_assets(final_asset_config, "")

        self.TRAIL_COLOR_MAP = jnp.array([
            self.jr.TRANSPARENT_ID,
            self.COLOR_TO_ID[self.consts.P1_TRAIL_COLOR],
            self.COLOR_TO_ID[self.consts.P2_TRAIL_COLOR],
        ])

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: SurroundState) -> jnp.ndarray:
        raster = self.jr.create_object_raster(self.BACKGROUND)
        cell_w, cell_h = self.consts.CELL_SIZE
        field_h = self.consts.GRID_HEIGHT * cell_h
        field_w = self.consts.GRID_WIDTH * cell_w
        slack = self.consts.SCREEN_SIZE[1] - field_h
        y_off = (slack // cell_h) * cell_h

        raster = self.jr.draw_rects(
            raster,
            positions=jnp.array([[0, y_off]]),
            sizes=jnp.array([[field_w, field_h]]),
            color_id=self.COLOR_TO_ID[self.PLAYFIELD_COLOR_TUPLE]
        )

        raster = self.jr.render_grid_inverse(
            raster, state.trail.T, (0, y_off), (cell_w, cell_h), self.TRAIL_COLOR_MAP
        )

        bx = self.consts.BORDER_CELLS_X * cell_w
        by = self.consts.BORDER_CELLS_Y * cell_h
        border_positions = jnp.array([[0, y_off], [0, y_off + field_h - by], [0, y_off], [field_w - bx, y_off]])
        border_sizes = jnp.array([[field_w, by], [field_w, by], [bx, field_h], [bx, field_h]])
        raster = self.jr.draw_rects(raster, border_positions, border_sizes, self.COLOR_TO_ID[self.BORDER_COLOR_TUPLE])

        occupied_grid = jnp.logical_or(state.trail != 0, state.border).T.astype(jnp.int32)
        temp_raster = jnp.zeros_like(raster, dtype=bool)
        grid_mask_raster = self.jr.render_grid_inverse(
            temp_raster, occupied_grid, (0, y_off), (cell_w, cell_h), jnp.array([False, True])
        )
        yy = self.jr._yy
        relative_y = yy - y_off
        mid = cell_h // 2
        divider_thickness = max(1, self.consts.DIVIDER_THICKNESS)
        band_mask = (relative_y % cell_h >= mid) & (relative_y % cell_h < mid + divider_thickness)
        final_divider_mask = jnp.logical_and(grid_mask_raster, band_mask)
        raster = jnp.where(final_divider_mask, self.COLOR_TO_ID[self.DIVIDER_COLOR_TUPLE], raster)

        p1x = state.pos0[0] * cell_w
        p1y = state.pos0[1] * cell_h + y_off
        p2x = state.pos1[0] * cell_w
        p2y = state.pos1[1] * cell_h + y_off

        p1_trail_mask = jnp.ones((cell_h, cell_w), dtype=jnp.uint8) * self.COLOR_TO_ID[self.consts.P1_TRAIL_COLOR]
        p2_trail_mask = jnp.ones((cell_h, cell_w), dtype=jnp.uint8) * self.COLOR_TO_ID[self.consts.P2_TRAIL_COLOR]
        p1_head_mask = jnp.ones((cell_h, cell_w), dtype=jnp.uint8) * self.COLOR_TO_ID[self.P1_HEAD_COLOR_TUPLE]
        p2_head_mask = jnp.ones((cell_h, cell_w), dtype=jnp.uint8) * self.COLOR_TO_ID[self.P2_HEAD_COLOR_TUPLE]

        raster = self.jr.render_at(raster, p1x, p1y, p1_trail_mask)
        raster = self.jr.render_at(raster, p2x, p2y, p2_trail_mask)
        raster = self.jr.render_at(raster, p1x, p1y, p1_head_mask)
        raster = self.jr.render_at(raster, p2x, p2y, p2_head_mask)

        border_y_abs = y_off + self.consts.BORDER_CELLS_Y * cell_h
        digit_h = self.SHAPE_MASKS['p1_digits'].shape[1]
        score_y = max(0, border_y_abs - digit_h - 8)
        padding_x = 30
        p1_digit_val = jnp.clip(state.score0 % 10, 0, 9)
        p2_digit_val = jnp.clip(state.score1 % 10, 0, 9)

        raster = self.jr.render_at(raster, padding_x, score_y, self.SHAPE_MASKS['p1_digits'][p1_digit_val])
        raster = self.jr.render_at(raster, self.consts.SCREEN_SIZE[0] - padding_x - self.SHAPE_MASKS['p2_digits'].shape[2], score_y, self.SHAPE_MASKS['p2_digits'][p2_digit_val])

        return self.jr.render_from_palette(raster, self.PALETTE)