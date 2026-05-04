"""
jax_bowling.py — A JAXAtari implementation of Atari Bowling.

Layout (top-down, like the original ROM):

    Bowler (left)  ─────────► Pins (right)

                                 7  8  9  10        (back row)
                                  4  5  6
                                   2  3
                                    1                (front pin)

Phases per roll
---------------
    WAITING   : player can move UP/DOWN, FIRE launches the ball
    ROLLING   : ball travels right; UP/DOWN apply a curve
    RESETTING : short delay so knocked pins are visible
    GAME_OVER : terminal

Game length: 10 frames × 2 rolls (simplified scoring = total pins knocked).

The class follows the JAXAtari pattern (JaxEnvironment + JAXGameRenderer).
Sprites are built programmatically (RGBA jnp arrays) so no .npy assets are
required.  The env is fully jitted and ready to be wrapped by the existing
NUDGE / VIPER / NeuralPPO training harnesses later.
"""

import os
from functools import partial
from typing import Tuple, NamedTuple, List, Dict, Any, Optional

import chex
import jax
import jax.numpy as jnp

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
import jaxatari.rendering.jax_rendering_utils as jr


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Phase enum (kept as plain ints because NamedTuples + JIT prefer ints)
PHASE_WAITING:   int = 0
PHASE_ROLLING:   int = 1
PHASE_RESETTING: int = 2
PHASE_GAME_OVER: int = 3


class BowlingConstants(NamedTuple):
    # Screen — matches the ALE Bowling render (210×160, HWC)
    screen_width:  int = 160
    screen_height: int = 210

    # Lane bounds in screen coordinates.  In the original ROM the lane
    # spans roughly y=104..170, x=0..160, with horizontal blue lines at
    # y≈102 and y≈172.  We treat the inner band as the playable area.
    lane_top:    int = 105
    lane_bottom: int = 170
    lane_left:   int = 8
    lane_right:  int = 156

    # Player (sprite shape extracted from the ROM is 32 tall × 9 wide).
    # The bowler in the reset frame stands at (x=18, y=139), so we keep
    # the same x and let player_y move freely along the lane.
    player_x:      int = 18
    player_width:  int = 9
    player_height: int = 32
    # Allow the head (top of sprite) to range so feet stay inside the lane.
    player_y_min:  int = 110
    player_y_max:  int = 138       # 138 + 32 = 170 (lane_bottom)
    player_y_init: int = 139       # ROM default

    # Ball — extracted sprite is 10 tall × 6 wide.
    ball_width:   int   = 6
    ball_height:  int   = 10
    ball_speed_x: float = 2.5      # px / step
    curve_dy:     float = 0.35     # UP/DOWN curve while rolling
    curve_max:    float = 2.0      # |vy| clamp

    # Pins (10, triangle pointing toward bowler).  Sprite is 4 tall × 2 wide.
    num_pins:   int = 10
    pin_width:  int = 2
    pin_height: int = 4

    # Pin (x, y) coordinates in screen space — these are the ACTUAL ROM
    # positions taken from the captured reset frame.  Apex points left
    # toward the bowler.
    pin_x: Tuple[int, ...] = (
        121,                       # apex (front, 1 pin)
        125, 125,                  # row 2 (2 pins)
        129, 129, 129,             # row 3 (3 pins)
        133, 133, 133, 133,        # back row (4 pins)
    )
    pin_y: Tuple[int, ...] = (
        137,                       # apex
        131, 143,                  # row 2
        125, 137, 149,             # row 3
        119, 131, 143, 155,        # back row
    )

    # Phase timing
    reset_delay: int = 24

    # Game length
    n_frames:        int = 10
    rolls_per_frame: int = 2


# ─────────────────────────────────────────────────────────────────────────────
# Pytrees: state / observation / info
# ─────────────────────────────────────────────────────────────────────────────

class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray


class BowlingState(NamedTuple):
    """Full simulation state.  Pin positions are static (in consts);
    only `pins_standing` is dynamic."""
    player_y:       chex.Array  # int32
    ball_x:         chex.Array  # float32
    ball_y:         chex.Array  # float32
    ball_vx:        chex.Array  # float32
    ball_vy:        chex.Array  # float32
    pins_standing:  chex.Array  # bool   [num_pins]
    score:          chex.Array  # int32  (running pins knocked)
    frame_idx:      chex.Array  # int32  (0..n_frames-1, then n_frames = done)
    roll_idx:       chex.Array  # int32  (0 or 1 within the frame)
    phase:          chex.Array  # int32  (PHASE_*)
    phase_timer:    chex.Array  # int32
    time:           chex.Array  # int32  (global step counter)
    game_over:      chex.Array  # bool


class BowlingObservation(NamedTuple):
    player: EntityPosition
    ball:   EntityPosition
    # pins: shape (num_pins, 5) -> [x, y, w, h, standing(0/1)]
    pins:      jnp.ndarray
    score:     jnp.ndarray
    frame_idx: jnp.ndarray
    roll_idx:  jnp.ndarray
    phase:     jnp.ndarray


class BowlingInfo(NamedTuple):
    time:        jnp.ndarray
    all_rewards: jnp.ndarray


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

class JaxBowling(JaxEnvironment[BowlingState, BowlingObservation, BowlingInfo, BowlingConstants]):
    """Atari-flavoured Bowling, fully jitted."""

    def __init__(self, consts: BowlingConstants = None, reward_funcs: list[callable] = None):
        if consts is None:
            consts = BowlingConstants()
        super().__init__(consts)
        self.reward_funcs = tuple(reward_funcs) if reward_funcs is not None else None
        self.renderer = BowlingRenderer(self.consts)
        self.state = self.reset()

    # ── reset ─────────────────────────────────────────────────────────────
    def reset(self, key: jax.random.PRNGKey = None) -> Tuple[BowlingObservation, BowlingState]:
        c = self.consts
        state = BowlingState(
            player_y=jnp.array(c.player_y_init, dtype=jnp.int32),
            ball_x=jnp.array(c.player_x + c.player_width, dtype=jnp.float32),
            ball_y=jnp.array(c.player_y_init, dtype=jnp.float32),
            ball_vx=jnp.array(0.0, dtype=jnp.float32),
            ball_vy=jnp.array(0.0, dtype=jnp.float32),
            pins_standing=jnp.ones((c.num_pins,), dtype=jnp.bool_),
            score=jnp.array(0, dtype=jnp.int32),
            frame_idx=jnp.array(0, dtype=jnp.int32),
            roll_idx=jnp.array(0, dtype=jnp.int32),
            phase=jnp.array(PHASE_WAITING, dtype=jnp.int32),
            phase_timer=jnp.array(0, dtype=jnp.int32),
            time=jnp.array(0, dtype=jnp.int32),
            game_over=jnp.array(False, dtype=jnp.bool_),
        )
        return self._get_observation(state), state

    # ── step ──────────────────────────────────────────────────────────────
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: BowlingState, action: int
             ) -> Tuple[BowlingObservation, BowlingState, float, bool, BowlingInfo]:
        c = self.consts

        action = jnp.asarray(action, dtype=jnp.int32)

        is_waiting   = state.phase == PHASE_WAITING
        is_rolling   = state.phase == PHASE_ROLLING
        is_resetting = state.phase == PHASE_RESETTING
        is_done      = state.phase == PHASE_GAME_OVER

        # ── 1) WAITING: player moves; FIRE launches the ball ──────────────
        dy_player = jnp.where(action == Action.UP,   -1,
                     jnp.where(action == Action.DOWN, 1, 0))
        new_player_y = jnp.where(
            is_waiting,
            jnp.clip(state.player_y + dy_player, c.player_y_min, c.player_y_max),
            state.player_y,
        ).astype(jnp.int32)

        fire_pressed = jnp.logical_and(is_waiting, action == Action.FIRE)

        # Ball spawns just to the right of the player, aligned with player_y
        ball_x_after_fire = jnp.where(
            fire_pressed,
            jnp.float32(c.player_x + c.player_width + 1),
            state.ball_x,
        )
        ball_y_after_fire = jnp.where(
            fire_pressed, new_player_y.astype(jnp.float32), state.ball_y
        )
        ball_vx_after_fire = jnp.where(
            fire_pressed, jnp.float32(c.ball_speed_x), state.ball_vx
        )
        ball_vy_after_fire = jnp.where(fire_pressed, jnp.float32(0.0), state.ball_vy)

        phase_after_fire = jnp.where(
            fire_pressed, jnp.int32(PHASE_ROLLING), state.phase
        )

        # ── 2) ROLLING: integrate ball, apply curve, check pin collisions ─
        in_rolling_now = phase_after_fire == PHASE_ROLLING

        curve = jnp.where(action == Action.UP,   -c.curve_dy,
                jnp.where(action == Action.DOWN,  c.curve_dy, 0.0))
        ball_vy_curved = jnp.where(in_rolling_now,
                                    ball_vy_after_fire + curve,
                                    ball_vy_after_fire)
        ball_vy_curved = jnp.clip(ball_vy_curved, -c.curve_max, c.curve_max)

        new_ball_x = jnp.where(in_rolling_now,
                                ball_x_after_fire + ball_vx_after_fire,
                                ball_x_after_fire)
        new_ball_y = jnp.where(in_rolling_now,
                                ball_y_after_fire + ball_vy_curved,
                                ball_y_after_fire)

        # Pin collisions: AABB centred on pin_x/pin_y vs ball_x/ball_y
        pin_xs = jnp.array(c.pin_x, dtype=jnp.float32)
        pin_ys = jnp.array(c.pin_y, dtype=jnp.float32)
        bw = jnp.float32(c.ball_width)
        bh = jnp.float32(c.ball_height)
        pw = jnp.float32(c.pin_width)
        ph = jnp.float32(c.pin_height)

        overlap_x = (jnp.abs(new_ball_x - pin_xs) * 2.0) < (bw + pw)
        overlap_y = (jnp.abs(new_ball_y - pin_ys) * 2.0) < (bh + ph)
        hit = overlap_x & overlap_y & state.pins_standing & in_rolling_now

        new_pins_standing = state.pins_standing & ~hit
        pins_knocked_delta = jnp.sum(hit).astype(jnp.int32)

        # ── 3) Roll-end detection ────────────────────────────────────────
        ball_off_right = new_ball_x > jnp.float32(c.lane_right)
        ball_in_gutter = jnp.logical_or(
            new_ball_y < jnp.float32(c.lane_top),
            new_ball_y > jnp.float32(c.lane_bottom),
        )
        all_pins_down = jnp.sum(new_pins_standing) == 0
        roll_ended = jnp.logical_and(
            in_rolling_now,
            jnp.logical_or(jnp.logical_or(ball_off_right, ball_in_gutter),
                           all_pins_down),
        )

        phase_after_roll = jnp.where(
            roll_ended, jnp.int32(PHASE_RESETTING), phase_after_fire
        )
        timer_after_roll = jnp.where(roll_ended, jnp.int32(0), state.phase_timer)

        # ── 4) RESETTING: tick timer, then advance roll/frame ────────────
        in_resetting_now = phase_after_roll == PHASE_RESETTING
        new_phase_timer = jnp.where(
            in_resetting_now, timer_after_roll + 1, timer_after_roll
        )

        reset_done = jnp.logical_and(in_resetting_now,
                                      new_phase_timer >= c.reset_delay)

        # Did we just complete a strike on the first roll?
        is_strike  = jnp.logical_and(state.roll_idx == 0,
                                      jnp.sum(new_pins_standing) == 0)
        # Frame ends if strike (skip 2nd roll) OR we just finished the 2nd roll
        frame_ends = jnp.logical_or(is_strike, state.roll_idx == 1)

        # Advance roll / frame indices (only on reset_done)
        new_roll_idx = jnp.where(
            reset_done,
            jnp.where(frame_ends, jnp.int32(0), state.roll_idx + 1),
            state.roll_idx,
        ).astype(jnp.int32)

        new_frame_idx = jnp.where(
            jnp.logical_and(reset_done, frame_ends),
            state.frame_idx + 1,
            state.frame_idx,
        ).astype(jnp.int32)

        # Reset pins at the end of a frame
        pins_reset_now = jnp.logical_and(reset_done, frame_ends)
        new_pins_standing = jnp.where(
            pins_reset_now,
            jnp.ones((c.num_pins,), dtype=jnp.bool_),
            new_pins_standing,
        )

        # Game-over check
        new_game_over = jnp.logical_or(
            state.game_over, new_frame_idx >= c.n_frames
        )

        # Phase transition out of RESETTING
        new_phase = jnp.where(
            reset_done,
            jnp.where(new_game_over,
                       jnp.int32(PHASE_GAME_OVER),
                       jnp.int32(PHASE_WAITING)),
            phase_after_roll,
        ).astype(jnp.int32)

        # Reset ball back near player when going to WAITING
        going_to_wait = jnp.logical_and(reset_done, ~new_game_over)
        final_ball_x = jnp.where(
            going_to_wait,
            jnp.float32(c.player_x + c.player_width + 1),
            new_ball_x,
        )
        final_ball_y = jnp.where(
            going_to_wait, new_player_y.astype(jnp.float32), new_ball_y
        )
        final_ball_vx = jnp.where(going_to_wait, jnp.float32(0.0),
                                   ball_vx_after_fire)
        final_ball_vy = jnp.where(going_to_wait, jnp.float32(0.0),
                                   ball_vy_curved)

        # Frozen state once game over: actions are no-ops
        final_player_y = jnp.where(is_done, state.player_y, new_player_y)
        final_phase    = jnp.where(is_done, state.phase,    new_phase)

        # ── 5) Score & bookkeeping ───────────────────────────────────────
        new_score = state.score + pins_knocked_delta

        new_state = BowlingState(
            player_y=final_player_y.astype(jnp.int32),
            ball_x=final_ball_x.astype(jnp.float32),
            ball_y=final_ball_y.astype(jnp.float32),
            ball_vx=final_ball_vx.astype(jnp.float32),
            ball_vy=final_ball_vy.astype(jnp.float32),
            pins_standing=new_pins_standing,
            score=new_score.astype(jnp.int32),
            frame_idx=new_frame_idx.astype(jnp.int32),
            roll_idx=new_roll_idx.astype(jnp.int32),
            phase=final_phase.astype(jnp.int32),
            phase_timer=new_phase_timer.astype(jnp.int32),
            time=(state.time + 1).astype(jnp.int32),
            game_over=new_game_over,
        )

        obs         = self._get_observation(new_state)
        env_reward  = self._get_reward(state, new_state)
        done        = self._get_done(new_state)
        all_rewards = self._get_all_reward(state, new_state)
        info        = self._get_info(new_state, all_rewards)

        return obs, new_state, env_reward, done, info

    # ── observation / reward / info ──────────────────────────────────────
    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: BowlingState) -> BowlingObservation:
        c = self.consts

        player = EntityPosition(
            x=jnp.array(c.player_x,      dtype=jnp.int32),
            y=state.player_y,
            width=jnp.array(c.player_width,  dtype=jnp.int32),
            height=jnp.array(c.player_height, dtype=jnp.int32),
        )

        ball = EntityPosition(
            x=jnp.round(state.ball_x).astype(jnp.int32),
            y=jnp.round(state.ball_y).astype(jnp.int32),
            width=jnp.array(c.ball_width,  dtype=jnp.int32),
            height=jnp.array(c.ball_height, dtype=jnp.int32),
        )

        pin_xs = jnp.array(c.pin_x, dtype=jnp.int32)
        pin_ys = jnp.array(c.pin_y, dtype=jnp.int32)
        ws     = jnp.full((c.num_pins,), c.pin_width,  dtype=jnp.int32)
        hs     = jnp.full((c.num_pins,), c.pin_height, dtype=jnp.int32)
        st     = state.pins_standing.astype(jnp.int32)
        # Stack into [num_pins, 5]
        pins = jnp.stack([pin_xs, pin_ys, ws, hs, st], axis=1)

        return BowlingObservation(
            player=player,
            ball=ball,
            pins=pins,
            score=state.score,
            frame_idx=state.frame_idx,
            roll_idx=state.roll_idx,
            phase=state.phase,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, prev: BowlingState, state: BowlingState) -> jnp.ndarray:
        # Per-step reward = newly knocked pins this step.
        return (state.score - prev.score).astype(jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(self, prev: BowlingState, state: BowlingState) -> jnp.ndarray:
        if self.reward_funcs is None:
            return jnp.zeros(1)
        return jnp.array([f(prev, state) for f in self.reward_funcs])

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: BowlingState, all_rewards: chex.Array = None) -> BowlingInfo:
        return BowlingInfo(time=state.time, all_rewards=all_rewards)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: BowlingState) -> jnp.ndarray:
        return state.game_over

    # ── spaces ───────────────────────────────────────────────────────────
    def action_space(self) -> spaces.Discrete:
        """
        Bowling uses 4 raw Atari action codes:
            0: NOOP
            1: FIRE   (launch the ball when in WAITING)
            2: UP     (move player up / curve ball up while rolling)
            5: DOWN   (move player down / curve ball down while rolling)
        We follow the freeway convention: return Discrete(N) for the count,
        while the env itself accepts the raw atari ints above.
        """
        return spaces.Discrete(4)

    def observation_space(self) -> spaces.Dict:
        c = self.consts
        return spaces.Dict({
            "player": spaces.Dict({
                "x":      spaces.Box(low=0, high=c.screen_width,  shape=(), dtype=jnp.int32),
                "y":      spaces.Box(low=0, high=c.screen_height, shape=(), dtype=jnp.int32),
                "width":  spaces.Box(low=0, high=c.screen_width,  shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=c.screen_height, shape=(), dtype=jnp.int32),
            }),
            "ball": spaces.Dict({
                "x":      spaces.Box(low=0, high=c.screen_width,  shape=(), dtype=jnp.int32),
                "y":      spaces.Box(low=0, high=c.screen_height, shape=(), dtype=jnp.int32),
                "width":  spaces.Box(low=0, high=c.screen_width,  shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=c.screen_height, shape=(), dtype=jnp.int32),
            }),
            "pins":      spaces.Box(low=0, high=c.screen_height, shape=(c.num_pins, 5), dtype=jnp.int32),
            "score":     spaces.Box(low=0, high=300, shape=(), dtype=jnp.int32),
            "frame_idx": spaces.Box(low=0, high=c.n_frames, shape=(), dtype=jnp.int32),
            "roll_idx":  spaces.Box(low=0, high=2,        shape=(), dtype=jnp.int32),
            "phase":     spaces.Box(low=0, high=3,        shape=(), dtype=jnp.int32),
        })

    def image_space(self) -> spaces.Box:
        return spaces.Box(low=0, high=255,
                          shape=(self.consts.screen_height, self.consts.screen_width, 3),
                          dtype=jnp.uint8)

    def render(self, state: BowlingState) -> jnp.ndarray:
        return self.renderer.render(state)

    def obs_to_flat_array(self, obs: BowlingObservation) -> jnp.ndarray:
        parts = [
            obs.player.x.reshape(-1),  obs.player.y.reshape(-1),
            obs.player.width.reshape(-1), obs.player.height.reshape(-1),
            obs.ball.x.reshape(-1),    obs.ball.y.reshape(-1),
            obs.ball.width.reshape(-1),  obs.ball.height.reshape(-1),
            obs.pins.reshape(-1),
            obs.score.reshape(-1),
            obs.frame_idx.reshape(-1),
            obs.roll_idx.reshape(-1),
            obs.phase.reshape(-1),
        ]
        return jnp.concatenate(parts).astype(jnp.int32)


# ─────────────────────────────────────────────────────────────────────────────
# Renderer — uses sprites extracted from the original Atari Bowling ROM.
#
# Sprites live in:
#   JAXAtari/src/jaxatari/games/sprites/bowling/
#       background.npy   solid lane orange (210, 160, 4)
#       player.npy       (32,  9, 4)  bowler in pink/blue with black shoes
#       ball.npy         (10,  6, 4)  blue lozenge
#       pin_up.npy       ( 4,  2, 4)  blue standing pin
#       pin_down.npy     ( 2,  2, 4)  flattened, semi-transparent fallen pin
#
# To regenerate: run scripts/extract_bowling_frames.py then
# scripts/extract_bowling_sprites.py.
# ─────────────────────────────────────────────────────────────────────────────


class BowlingRenderer(JAXGameRenderer):
    """Atari-Bowling-faithful renderer.  Loads RGBA sprites from disk."""

    def __init__(self, consts: BowlingConstants = None):
        super().__init__()
        self.consts = consts or BowlingConstants()
        self.sprites = self._load_sprites()

    def _load_sprites(self) -> Dict[str, jnp.ndarray]:
        """Load the .npy sprite assets ripped from the Atari ROM."""
        module_dir = os.path.dirname(os.path.abspath(__file__))
        sprite_dir = os.path.join(module_dir, "mods", "bowling","sprites")

        names = ["background", "player", "ball", "pin_up", "pin_down"]
        out: Dict[str, jnp.ndarray] = {}
        for name in names:
            path = os.path.join(sprite_dir, f"{name}.npy")
            frame = jr.loadFrame(path).astype(jnp.uint8)   # (H, W, 4)
            out[name] = frame
        return out

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: BowlingState) -> jnp.ndarray:
        c = self.consts
        raster = jr.create_initial_frame(width=c.screen_width, height=c.screen_height)

        # ── Background: full-frame lane orange (matches the ROM exactly) ─
        raster = jr.render_at(raster, 0, 0, self.sprites["background"])

        # ── Lane border lines (the two horizontal blue stripes) ──────────
        line_color = jnp.array([45, 50, 184], dtype=jnp.uint8)
        raster = raster.at[c.lane_top - 1 : c.lane_top + 1,
                           c.lane_left    : c.lane_right, :].set(line_color)
        raster = raster.at[c.lane_bottom  : c.lane_bottom + 2,
                           c.lane_left    : c.lane_right, :].set(line_color)

        # ── Pins (loop unrolled at trace time) ───────────────────────────
        pin_up   = self.sprites["pin_up"]
        pin_down = self.sprites["pin_down"]
        for i in range(c.num_pins):
            sprite = jax.lax.cond(
                state.pins_standing[i],
                lambda: pin_up,
                lambda: pin_down,
            )
            raster = jr.render_at(raster, c.pin_x[i], c.pin_y[i], sprite)

        # ── Player ───────────────────────────────────────────────────────
        raster = jr.render_at(
            raster, c.player_x, state.player_y, self.sprites["player"]
        )

        # ── Ball (always drawn — sits next to the player while waiting) ─
        ball_x_int = jnp.round(state.ball_x).astype(jnp.int32)
        ball_y_int = jnp.round(state.ball_y).astype(jnp.int32)
        raster = jr.render_at(raster, ball_x_int, ball_y_int, self.sprites["ball"])

        # ── Lightweight HUD: a yellow score bar (max 100 = 1 strike-out
        # game), plus 10 frame ticks across the top edge.
        bar_w_max = c.screen_width - 16
        score_capped = jnp.minimum(state.score, 100)
        bar_w = (score_capped.astype(jnp.float32) / 100.0 * bar_w_max).astype(jnp.int32)
        bar_w = jnp.clip(bar_w, 0, bar_w_max)
        col_idx  = jnp.arange(bar_w_max)
        bar_mask = col_idx < bar_w
        bar_row  = jnp.where(
            bar_mask[None, :, None],
            jnp.array([240, 240,  0], dtype=jnp.uint8)[None, None, :],
            jnp.array([ 40,  40, 40], dtype=jnp.uint8)[None, None, :],
        )
        bar_row = jnp.broadcast_to(bar_row, (4, bar_w_max, 3))
        raster = raster.at[8 : 12, 8 : 8 + bar_w_max, :].set(bar_row.astype(jnp.uint8))

        for f in range(c.n_frames):
            x0 = 8 + f * 14
            color = jax.lax.cond(
                state.frame_idx > f,
                lambda: jnp.array([100, 220, 100], dtype=jnp.uint8),
                lambda: jax.lax.cond(
                    state.frame_idx == f,
                    lambda: jnp.array([220, 220,  60], dtype=jnp.uint8),
                    lambda: jnp.array([ 80,  80,  80], dtype=jnp.uint8),
                ),
            )
            raster = raster.at[2 : 5, x0 : x0 + 10, :].set(color)

        return raster
