"""
jax_lostluggage.py
===================
Action set (9 actions, matching the log output):
  0 NOOP  1 UP  2 RIGHT  3 LEFT  4 DOWN
  5 UPRIGHT  6 UPLEFT  7 DOWNRIGHT  8 DOWNLEFT
"""

import os
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import chex
from jax import lax
from jax import random as jrandom
from flax import struct

import jaxatari.spaces as spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action, ObjectObservation
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

class LostLuggageConstants(struct.PyTreeNode):
    # ── Display ──────────────────────────────────────────────────────────────
    NATIVE_W: int = struct.field(pytree_node=False, default=160)
    NATIVE_H: int = struct.field(pytree_node=False, default=210)

    # ── Borders (px) ──────────────────────────────────────────────────
    BORDER_LEFT:   int = struct.field(pytree_node=False, default=12)
    BORDER_RIGHT:  int = struct.field(pytree_node=False, default=148)
    PLAYER_Y_TOP:  int = struct.field(pytree_node=False, default=95)
    PLAYER_Y_BOT:  int = struct.field(pytree_node=False, default=170)

    # ── Suitcase geometry ────────────────────────────────────────────────────
    SUIT_W:       int = struct.field(pytree_node=False, default=6)
    SUIT_H:       int = struct.field(pytree_node=False, default=7)
    SUIT_Y_SPAWN: int = struct.field(pytree_node=False, default=67)
    SUIT_Y_FLOOR: int = struct.field(pytree_node=False, default=170)

    SUIT_SPAWN_XS: chex.Array = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.arange(56, 105, 4, dtype=jnp.int32)  # shape (13,)
    )

    # ── Round / spawn schedule ───────────────────────────────────────────────
    SUITS_PER_ROUND:         int = struct.field(pytree_node=False, default=25)
    SPAWN_INTERVAL_START:    float = struct.field(pytree_node=False, default=13)
    SPAWN_DECREASE_EVERY:    float = struct.field(pytree_node=False, default=2)
    SPAWN_DECREASE_AMOUNT:   float = struct.field(pytree_node=False, default=1)
    SPAWN_INTERVAL_MIN:     float = struct.field(pytree_node=False, default=5)
    MAX_ACTIVE_SUITS:        int = struct.field(pytree_node=False, default=25)

    # ── Gravity ──────────────────────────────────────────────────────────────
    # A suitcase drops 2 native-px every GRAVITY_FRAMES steps.
    GRAVITY_FRAMES: int = struct.field(pytree_node=False, default=2) # speed of drop

     # ── Delay between / before rounds (fps) ─────────
    ROUND_DELAY_TICKS: int = struct.field(pytree_node=False, default=75)

    # ── Player geometry ──────────────────────────────────────────────────────
    # Two passengers sharing one anchor_x, size 6×17
    # Gap between A's right edge (anchor_x+5) and B's left edge (anchor_x+17) = 11 px.
    PASS_A_REL_X:    int = struct.field(pytree_node=False, default=0)
    PASS_B_REL_X:    int = struct.field(pytree_node=False, default=17)
    PASS_W:          int = struct.field(pytree_node=False, default=6)
    PASS_H:          int = struct.field(pytree_node=False, default=17)
    PLAYER_START_X:  int = struct.field(pytree_node=False, default=72)
    PLAYER_Y:        int = struct.field(pytree_node=False, default=153)
    PLAYER_SPEED:    int = struct.field(pytree_node=False, default=3) # changed from 2 to 3
    PLAYER_VERT_SPEED: int = struct.field(pytree_node=False, default=1.5) # changed from 1 to 1.5
    # How many frames to hold the passenger in a standing pose
    PASS_POSE_HOLD_FRAMES: int = struct.field(pytree_node=False, default=8)
    
    # Plane sprite behaviour
    PLANE_START_X: int = struct.field(pytree_node=False, default=0)
    PLANE_START_Y: int = struct.field(pytree_node=False, default=5)
    PLANE_W: int = struct.field(pytree_node=False, default=8)
    PLANE_H: int = struct.field(pytree_node=False, default=8)
    # How many frames for the plane to cross the screen
    PLANE_TRIP_FRAMES: int = struct.field(pytree_node=False, default=600)

    # ── Score / lives ────────────────────────────────────────────────────────
    INITIAL_LIVES:   int = struct.field(pytree_node=False, default=3)
    MAX_LIVES:       int = struct.field(pytree_node=False, default=10)


    # Extra-life thresholds: award a life at score%1000 == 400 and score%1000 == 800.
    # Stored as a 2-element array for vectorised checks.
    EXTRA_LIFE_THRESHOLDS: chex.Array = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array([400, 800], dtype=jnp.int32)
    )

    # ── Angle probability table ──────────────────────────────────────────────
    ANGLE_CUM_PROBS: chex.Array = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array([
            [500, 750, 1000, 1000, 1000, 1000, 1000],
            [450, 675,  900,  950, 1000, 1000, 1000],
            [400, 600,  800,  900, 1000, 1000, 1000],
            [350, 525,  700,  850, 1000, 1000, 1000],
            [250, 375,  500,  625, 750,  875,  1000]
        ], dtype=jnp.int32)
    )
    PRESS_HOLD_THRESHOLD: int = struct.field(pytree_node=False, default=5)
    POSE_DURATION: int = struct.field(pytree_node=False, default=3)

    # ── HUD positions ────────────────────────────────────────────────────────
    SCORE_X:  int = struct.field(pytree_node=False, default=88)
    SCORE_Y:  int = struct.field(pytree_node=False, default=186)
    LIFE_X:   int = struct.field(pytree_node=False, default=28)
    LIFE_Y:   int = struct.field(pytree_node=False, default=171)
    LIFE_SPACING: int = struct.field(pytree_node=False, default=9)

    # Flashing light during active rounds
    FLASH_X: int = struct.field(pytree_node=False, default=79)
    FLASH_Y: int = struct.field(pytree_node=False, default=43)
    FLASH_W: int = struct.field(pytree_node=False, default=2)
    FLASH_H: int = struct.field(pytree_node=False, default=2)
    FLASH_TOGGLE_FRAMES: int = struct.field(pytree_node=False, default=5)

    # ── Asset config ─────────────────────────────────────────────────────────
    ASSET_CONFIG: tuple = struct.field(pytree_node=False, default_factory=lambda: (
        {'name': 'background', 'type': 'background', 'file': 'background.npy'},
        {'name': 'digits',     'type': 'digits',     'pattern': 'digits/{}.npy'},
        {'name': 'suitcase',   'type': 'single',     'file': 'suitcase.npy'},
        {'name': 'life',       'type': 'single',     'file': 'life.npy'},
        {'name': 'passenger',  'type': 'single',     'file': 'passenger.npy'},
        {'name': 'passenger_standing','type':'single', 'file':'passenger_standing.npy'},
        {'name': 'plane_0',    'type': 'single',     'file': 'plane_0.npy'},
        {'name': 'flashing_light','type':'single','file':'flashing_light.npy'},
        {'name': 'suit_green',   'type': 'single', 'file': 'green_suitcase.npy'},
        {'name': 'suit_grey',    'type': 'single', 'file': 'grey_suitcase.npy'},
        {'name': 'suit_orange',  'type': 'single', 'file': 'orange_suitcase.npy'},
        {'name': 'suit_pink',    'type': 'single', 'file': 'pink_suitcase.npy'},
        {'name': 'suit_purple',  'type': 'single', 'file': 'purple_suitcase.npy'},
        {'name': 'suit_swamp',   'type': 'single', 'file': 'swamp_suitcase.npy'},
        {'name': 'suit_blue',    'type': 'single', 'file': 'blue_suitcase.npy'},
    ))

    # Ordered list of suitcase sprite name keys (should match ASSET_CONFIG order)
    SUIT_SPRITE_NAMES: tuple = struct.field(pytree_node=False, default_factory=lambda: (
        'suit_green', 'suit_grey', 'suit_orange', 'suit_pink',
        'suit_purple', 'suit_swamp', 'suit_blue',
    ))

# ─────────────────────────────────────────────────────────────────────────────
#  STATE
# ─────────────────────────────────────────────────────────────────────────────

@struct.dataclass
class LostLuggageState:
    # ── Player ───────────────────────────────────────────────────────────────
    player_x: chex.Array        
    player_y: chex.Array
    # ── Suitcases (fixed-size arrays, MAX_ACTIVE_SUITS=25 slots) ─────────────
    suit_x_fp: chex.Array       # (N,) int32  x * 16
    suit_y_fp: chex.Array       # (N,) int32  y * 16
    # Horizontal drift per vertical-pixel drop, stored as int32 × 1000
    suit_dx_fp: chex.Array      # (N,) int32  tan(angle) * 1000  (signed)
    suit_active: chex.Array     # (N,) bool   slot in use
    suit_color:   chex.Array    # (N,) int32  colour index 0-6

    # ── Round / spawn bookkeeping ─────────────────────────────────────────────
    round_num:       chex.Array  # () int32
    suits_spawned:   chex.Array  # () int32  spawned this round
    next_spawn_step: chex.Array  # () int32  step at which to spawn next suit
    round_failed:    chex.Array  # () bool

    # ── Inter-round delay ────────────────────────────────────────────────────
    # Counts down from ROUND_DELAY_TICKS to 0. While > 0 the game is frozen
    # (no gravity, no spawning). Set at reset and after each round end.
    delay_ticks: chex.Array     # () int32

    # ── Scoring / lives ───────────────────────────────────────────────────────
    score:           chex.Array  # () int32
    lives:           chex.Array  # () int32
    # Extra-life flags: shape (MAX_EXTRA_LIFE_BANDS, 2) bool
    # bands 0..MAX_EXTRA_LIFE_BANDS-1, threshold index 0=400 1=800
    extra_life_flags: chex.Array  # (20, 2) bool

    # ── Global tick ──────────────────────────────────────────────────────────
    tick: chex.Array             # () int32

    # ── PRNG key ─────────────────────────────────────────────────────────────
    key: chex.Array              # PRNGKey
    # ── Passenger animation state
    passenger_pose_timer: chex.Array  # () int32  countdown while standing pose is held
    passenger_pose_index: chex.Array  # () int32  0=default, 1=standing
    # ── Plane state
    plane_frame: chex.Array   # () int32  current frame within trip (0..PLANE_TRIP_FRAMES-1)
    plane_x_fp: chex.Array    # () int32  fixed-point x (×16)
    plane_y: chex.Array       # () int32  y position (native pixels)
    passenger_hold_counter: chex.Array  # () int32
    passenger_pose_cycle: chex.Array  # () int32


@struct.dataclass
class LostLuggageObservation:
    player:    ObjectObservation
    suitcases: ObjectObservation
    score:     chex.Array   # () int32
    lives:     chex.Array   # () int32
    round_num: chex.Array   # () int32


@struct.dataclass
class LostLuggageInfo:
    score:     chex.Array
    lives:     chex.Array
    round_num: chex.Array
    tick:      chex.Array


# ─────────────────────────────────────────────────────────────────────────────
#  FIXED-POINT HELPERS
# ─────────────────────────────────────────────────────────────────────────────
# Drift is stored as tan(angle)×1000.
#
# 1 native px down  →  x shifts by  tan(angle_deg) * 1  native px
#                   →  suit_x_fp shifts by  dx_fp = round(tan(angle_deg)*1000) * 16 / 1000
#                                         = dx_fp per gravity tick

_TAN_X16 = jnp.array([0, 10, 20, 30], dtype=jnp.int32)  

# Screen-half threshold: x < SCREEN_HALF → drift left; x >= SCREEN_HALF → drift right
SCREEN_HALF = 81


def _sample_suit(key: chex.Array, round_num: chex.Array,
                 consts: LostLuggageConstants) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    """
    Sample spawn x and angle for one suitcase.
    """
    key, k1, k2 = jrandom.split(key, 3)

    # --- Spawn x ---
    xi = jrandom.randint(k1, (), 0, 13, dtype=jnp.int32)  # 13 possible positions
    x = consts.SUIT_SPAWN_XS[xi]

    # --- Angle ---
    row = jnp.minimum(round_num - 1, jnp.int32(4))
    cum = consts.ANGLE_CUM_PROBS[row]
    u   = jrandom.randint(k2, (), 0, 1000, dtype=jnp.int32)

    slot = jnp.sum((cum <= u).astype(jnp.int32))
    slot = jnp.minimum(slot, jnp.int32(6))

    mag_idx = jnp.array([0, 1, 1, 2, 2, 3, 3], dtype=jnp.int32)[slot]

    dx = _TAN_X16[mag_idx]

    # Determine sign based on spawn position
    sign = jnp.where(x < jnp.int32(81), -1, 1)

    dx_per_drop = dx * sign

    x_fp = x * jnp.int32(16)
    y_fp = jnp.int32(consts.SUIT_Y_SPAWN) * jnp.int32(16)

    return x_fp, y_fp, dx_per_drop, key


def _compute_spawn_interval(round_num: chex.Array, consts: LostLuggageConstants) -> chex.Array:
    """
    Compute the spawn interval (in gravity-tick units) for the given round number.
    Ensures interval >= 1.
    """
    start  = jnp.float32(consts.SPAWN_INTERVAL_START)
    every  = jnp.int32(consts.SPAWN_DECREASE_EVERY)
    amount = jnp.float32(consts.SPAWN_DECREASE_AMOUNT)
    min_iv = jnp.float32(consts.SPAWN_INTERVAL_MIN)

    decreases = jnp.floor_divide(
        jnp.maximum(round_num - jnp.int32(1), jnp.int32(0)),
        every
    )

    interval = start - decreases.astype(jnp.float32) * amount

    # Clamp to minimum interval
    interval = jnp.maximum(interval, min_iv)

    return interval.astype(jnp.int32)


# ─────────────────────────────────────────────────────────────────────────────
#  ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────

class JaxLostLuggage(JaxEnvironment[LostLuggageState, LostLuggageObservation,
                                    LostLuggageInfo, LostLuggageConstants]):

    ACTION_SET: jnp.ndarray = jnp.array([
        Action.NOOP,
        Action.UP,
        Action.RIGHT,
        Action.LEFT,
        Action.DOWN,
        Action.UPRIGHT,
        Action.UPLEFT,
        Action.DOWNRIGHT,
        Action.DOWNLEFT,
    ], dtype=jnp.int32)

    def __init__(self, consts: LostLuggageConstants = None, config=None):
        self.consts = consts or LostLuggageConstants()
        super().__init__(self.consts)

        if config is None:
            self.config = render_utils.RendererConfig(
                game_dimensions=(210, 160),
                channels=3,
                downscale=None
            )
        else:
            self.config = config

        self.jr = render_utils.JaxRenderingUtils(self.config)
        self.renderer = JAXGameRenderer(self)
        final_asset_config = list(self.consts.ASSET_CONFIG)
        # sprite_path = os.path.join(render_utils.get_base_sprite_dir(), "lost_luggage")
        sprite_path = os.path.join(os.path.dirname(__file__), "sprites", "lostluggage")

        (self.PALETTE, self.SHAPE_MASKS, self.BACKGROUND,
         self.COLOR_TO_ID, self.FLIP_OFFSETS) = \
            self.jr.load_and_setup_assets(final_asset_config, sprite_path)
        
        suit_masks = [self.SHAPE_MASKS[name] for name in self.consts.SUIT_SPRITE_NAMES]
        self.SUIT_MASKS_STACKED = jnp.stack(suit_masks, axis=0)   # (7, H, W)

    # ── Action space ─────────────────────────────────────────────────────────
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTION_SET))

    def observation_space(self) -> spaces.Dict:
        N = self.consts.MAX_ACTIVE_SUITS
        screen = (self.consts.NATIVE_H, self.consts.NATIVE_W)
        return spaces.Dict({
            "player":    spaces.get_object_space(n=None, screen_size=screen),
            "suitcases": spaces.get_object_space(n=N,    screen_size=screen),
            "score":     spaces.Box(low=0, high=999999, shape=(), dtype=jnp.int32),
            "lives":     spaces.Box(low=0, high=self.consts.MAX_LIVES, shape=(), dtype=jnp.int32),
            "round_num": spaces.Box(low=1, high=99,     shape=(), dtype=jnp.int32),
        })

    def image_space(self) -> spaces.Box:
        return spaces.Box(low=0, high=255,
                          shape=(self.consts.NATIVE_H, self.consts.NATIVE_W, 3),
                          dtype=jnp.uint8)

    # ── Reset ─────────────────────────────────────────────────────────────────
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key=None):
        key = jrandom.PRNGKey(0) if key is None else key
        N = self.consts.MAX_ACTIVE_SUITS
        # initial spawn interval (gravity-tick units)
        init_spawn_interval = _compute_spawn_interval(jnp.int32(1), self.consts) * jnp.int32(self.consts.GRAVITY_FRAMES)
        state = LostLuggageState(
            player_x=jnp.int32(self.consts.PLAYER_START_X),
            player_y=jnp.int32(self.consts.PLAYER_Y),
            suit_x_fp=jnp.zeros(N, dtype=jnp.int32),
            suit_y_fp=jnp.zeros(N, dtype=jnp.int32),
            suit_dx_fp=jnp.zeros(N, dtype=jnp.int32),
            suit_color = jnp.zeros(N, dtype=jnp.int32),
            suit_active=jnp.zeros(N, dtype=jnp.bool_),
            round_num=jnp.int32(1),
            suits_spawned=jnp.int32(0),
            next_spawn_step  = jnp.int32(self.consts.ROUND_DELAY_TICKS) + init_spawn_interval,
            round_failed = jnp.bool_(False),
            delay_ticks = jnp.int32(self.consts.ROUND_DELAY_TICKS),
            passenger_pose_timer = jnp.int32(0),
            passenger_pose_index = jnp.int32(0),
            passenger_hold_counter = jnp.int32(0),
            passenger_pose_cycle = jnp.int32(0),
            plane_frame = jnp.int32(0),
            plane_x_fp = jnp.int32(self.consts.PLANE_START_X * 16),
            plane_y = jnp.int32(self.consts.PLANE_START_Y),
            score=jnp.int32(0),
            lives=jnp.int32(self.consts.INITIAL_LIVES),
            extra_life_flags=jnp.zeros((20,2), dtype=jnp.bool_),
            tick=jnp.int32(0),
            key=key
        )
        obs = self._get_observation(state)
        return obs, state

    # ── Step ─────────────────────────────────────────────────────────────────
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: LostLuggageState, action: chex.Array):
        atari_action = jnp.take(self.ACTION_SET, jnp.asarray(action, dtype=jnp.int32))

        # preserve previous score to compute per-step reward (score delta)
        prev_score_for_reward = state.score

         # ── Delay countdown ──────────────────────────────────────────────────
        # While delay_ticks > 0 the game is frozen; only the counter ticks down.
        is_delayed = state.delay_ticks > jnp.int32(0)
        new_delay  = jnp.maximum(state.delay_ticks - jnp.int32(1), jnp.int32(0))
        state      = state.replace(delay_ticks=new_delay)

        is_left  = ((atari_action == Action.LEFT) | (atari_action == Action.UPLEFT) | (atari_action == Action.DOWNLEFT))
        is_right = ((atari_action == Action.RIGHT)| (atari_action == Action.UPRIGHT)| (atari_action == Action.DOWNRIGHT))
        is_up    = ((atari_action == Action.UP)   | (atari_action == Action.UPRIGHT)  | (atari_action == Action.UPLEFT))
        is_down  = ((atari_action == Action.DOWN) | (atari_action == Action.DOWNRIGHT)| (atari_action == Action.DOWNLEFT))

        # Clamp horizontal
        player_width = jnp.int32(self.consts.PASS_B_REL_X + self.consts.PASS_W)
        min_px = jnp.int32(self.consts.BORDER_LEFT)
        max_px = jnp.int32(self.consts.BORDER_RIGHT - player_width)

        dx = jnp.where(is_left, -self.consts.PLAYER_SPEED,
             jnp.where(is_right, self.consts.PLAYER_SPEED, 0))
        new_px = jnp.clip(state.player_x + dx, min_px, max_px)

        # Vertical movement
        dy = jnp.where(is_up, -self.consts.PLAYER_VERT_SPEED,
             jnp.where(is_down, self.consts.PLAYER_VERT_SPEED, 0))
        max_py = jnp.int32(self.consts.PLAYER_Y_BOT - self.consts.PASS_H)
        new_py = jnp.clip(state.player_y + dy,
                        jnp.int32(self.consts.PLAYER_Y_TOP),
                        max_py)

        state = state.replace(player_x=new_px, player_y=new_py)

        # Passenger pose handling: when a movement input occurs and the
        # pose-timer is zero, set the standing pose and start the hold timer.
        pressed = (is_left | is_right | is_up | is_down)
        threshold = jnp.int32(self.consts.PRESS_HOLD_THRESHOLD)
        duration  = jnp.int32(self.consts.POSE_DURATION)
        cycle_len = threshold + duration  # frames before first standing + standing duration

        # Increment cycle counter if pressed, else reset
        new_cycle = jnp.where(pressed,
                            (state.passenger_pose_cycle + 1) % (cycle_len * 2),
                            jnp.int32(0))

        # Determine pose index based on cycle:
        # - First threshold frames → default
        # - Next duration frames → standing
        # - Repeat
        pose_index = jnp.where(
            new_cycle >= threshold,
            1,  # standing pose
            0   # default pose
        )

        # Save state
        state = state.replace(
            passenger_pose_cycle=new_cycle,
            passenger_pose_index=pose_index
        )

        def active_step(s: LostLuggageState) -> LostLuggageState:
 
            # 1. Spawn
            should_spawn = (s.tick >= s.next_spawn_step) & \
                           (s.suits_spawned < jnp.int32(self.consts.SUITS_PER_ROUND))
            s = lax.cond(should_spawn, self._spawn_one_suitcase, lambda x: x, s)
 
            # 2. Gravity
            tick_next    = s.tick + jnp.int32(1)
            gravity_drop = (tick_next % jnp.int32(self.consts.GRAVITY_FRAMES) == 0)
            s = lax.cond(gravity_drop, self._apply_gravity, lambda x: x, s)
 
            # 3. Collisions / escape
            s = self._process_collisions(s)
 
            # 4. Round transitions
            #    a) Round FAILED: a suitcase escaped → reset same round
            #    b) Round WON:    all 25 caught → advance round
            round_won  = (s.suits_spawned >= jnp.int32(self.consts.SUITS_PER_ROUND)) & \
                         (~jnp.any(s.suit_active)) & (~s.round_failed)
 
            s = lax.cond(s.round_failed, self._reset_round,   lambda x: x, s)
            s = lax.cond(round_won,      self._advance_round,  lambda x: x, s)
 
            return s
        
        state = lax.cond(is_delayed, lambda s: s, active_step, state)
 
        # Advance tick unconditionally
        state = state.replace(tick=state.tick + jnp.int32(1))

        # Advance plane position: compute frame within trip and derive x_fp.
        trip = jnp.int32(self.consts.PLANE_TRIP_FRAMES)
        frame_next = (state.plane_frame + jnp.int32(1)) % trip
        # start_x in native px, convert to fixed-point ×16
        start_x_fp = jnp.int32(self.consts.PLANE_START_X) * jnp.int32(16)
        total_native_dx = jnp.int32(self.consts.NATIVE_W) + jnp.int32(16) 
        dx_fp = (total_native_dx * jnp.int32(16)) // trip
        plane_x_fp_next = start_x_fp + frame_next * dx_fp

        state = state.replace(plane_frame=frame_next, plane_x_fp=plane_x_fp_next)

        obs = self._get_observation(state)
        reward = (state.score - prev_score_for_reward).astype(jnp.float32)
        done = self._get_done(state)
        info = self._get_info(state)
        return obs, state, reward, done, info

    # ── Internal helpers ──────────────────────────────────────────────────────

    @partial(jax.jit, static_argnums=(0,))
    def _spawn_one_suitcase(self, state: LostLuggageState) -> LostLuggageState:
        free_mask = ~state.suit_active
        has_free  = jnp.any(free_mask)

        def spawn(s):
            free_slots = jnp.where(free_mask,
                                jnp.arange(self.consts.MAX_ACTIVE_SUITS),
                                self.consts.MAX_ACTIVE_SUITS)
            slot = jnp.min(free_slots)

            x_fp, y_fp, dx, new_key = _sample_suit(s.key, s.round_num, self.consts)

            num_colors = len(self.consts.SUIT_SPRITE_NAMES)
            color_idx = jrandom.randint(new_key, (), 0, num_colors, dtype=jnp.int32)

            spawn_int = _compute_spawn_interval(s.round_num, self.consts)
            next_spawn = s.tick + (spawn_int * jnp.int32(self.consts.GRAVITY_FRAMES))

            return s.replace(
                suit_x_fp       = s.suit_x_fp.at[slot].set(x_fp),
                suit_y_fp       = s.suit_y_fp.at[slot].set(y_fp),
                suit_dx_fp      = s.suit_dx_fp.at[slot].set(dx),
                suit_color      = s.suit_color.at[slot].set(color_idx),
                suit_active     = s.suit_active.at[slot].set(True),
                suits_spawned   = s.suits_spawned + jnp.int32(1),
                next_spawn_step = next_spawn,
                key             = new_key,
            )

        return lax.cond(has_free, spawn, lambda s: s, state)

    @partial(jax.jit, static_argnums=(0,))
    def _apply_gravity(self, state: LostLuggageState) -> LostLuggageState:
        """
        Drop every active suitcase by 1 native px vertically,
        accumulating horizontal drift according to its angle.
        """
        drop = jnp.int32(16 + state.round_num * 2)  # speed of drop
        new_y_fp = jnp.where(state.suit_active,
                             state.suit_y_fp + drop,
                             state.suit_y_fp)
        new_x_fp = jnp.where(state.suit_active,
                             state.suit_x_fp + state.suit_dx_fp,
                             state.suit_x_fp)
        return state.replace(suit_x_fp=new_x_fp, suit_y_fp=new_y_fp)

    @partial(jax.jit, static_argnums=(0,))
    def _process_collisions(self, state: LostLuggageState) -> LostLuggageState:
        sx = state.suit_x_fp >> 4
        sy = state.suit_y_fp >> 4
        px = state.player_x
        py = state.player_y

        a_left  = px + jnp.int32(self.consts.PASS_A_REL_X)
        a_right = a_left + jnp.int32(self.consts.PASS_W)
        b_left  = px + jnp.int32(self.consts.PASS_B_REL_X)
        b_right = b_left + jnp.int32(self.consts.PASS_W)

        p_top = py
        p_bot = py + jnp.int32(self.consts.PASS_H)

        def overlap_x(suit_x, p_left, p_right):
            return (suit_x + jnp.int32(self.consts.SUIT_W) > p_left) & (suit_x < p_right)

        def overlap_y(suit_y, p_top_, p_bot_):
            return (suit_y + jnp.int32(self.consts.SUIT_H) > p_top_) & (suit_y < p_bot_)

        hit_a = overlap_x(sx, a_left, a_right) & overlap_y(sy, p_top, p_bot)
        hit_b = overlap_x(sx, b_left, b_right) & overlap_y(sy, p_top, p_bot)
        caught = (hit_a | hit_b) & state.suit_active

        hit_left_wall  = (sx < self.consts.BORDER_LEFT) & state.suit_active
        hit_right_wall = (sx + self.consts.SUIT_W > self.consts.BORDER_RIGHT + 1) & state.suit_active
        hit_floor      = (sy + self.consts.SUIT_H > self.consts.SUIT_Y_FLOOR) & state.suit_active
        escaped = (hit_left_wall | hit_right_wall | hit_floor) & (~caught)

        pts_each  = jnp.int32(2) + state.round_num
        n_caught  = jnp.sum(caught.astype(jnp.int32))
        score_inc = n_caught * pts_each
        prev_score = state.score
        new_score  = prev_score + score_inc

        # Extra life logic
        thresholds = jnp.array([400, 800], dtype=jnp.int32)
        band_before = prev_score // 1000
        band_after  = new_score // 1000

        bands = jnp.clip(jnp.array([band_before, band_after]), 0, 19)

        def check_threshold(flags, band, t_idx):
            milestone = band * 1000 + thresholds[t_idx]
            crossed = (prev_score < milestone) & (new_score >= milestone)
            already = flags[band, t_idx]
            earn = crossed & (~already)
            flags = flags.at[band, t_idx].set(already | earn)
            return flags, earn.astype(jnp.int32)

        new_flags = state.extra_life_flags
        bonus_lives = jnp.int32(0)

        # iterate over the two band entries (before/after)
        for i in range(2):
            band = bands[i]
            for t_idx in range(2):
                new_flags, earned = check_threshold(new_flags, band, t_idx)
                bonus_lives += earned

        n_escaped  = jnp.sum(escaped.astype(jnp.int32))
        new_lives  = jnp.clip(state.lives + bonus_lives - n_escaped,
                            0, self.consts.MAX_LIVES)

        new_active = state.suit_active & ~caught & ~escaped

        any_escaped = n_escaped > jnp.int32(0)
        round_failed = state.round_failed | any_escaped

        return state.replace(
            suit_active      = new_active,
            score            = new_score,
            lives            = new_lives,
            extra_life_flags = new_flags,
            round_failed     = round_failed,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _reset_round(self, state: LostLuggageState) -> LostLuggageState:
        """
        A suitcase escaped: restart the same round (same round_num, same pts/speed).
        Deactivate all suitcases, reset spawn counters, start delay.
        """
        N = self.consts.MAX_ACTIVE_SUITS
        spawn_int = _compute_spawn_interval(state.round_num, self.consts) * jnp.int32(self.consts.GRAVITY_FRAMES)
        next_spawn = state.tick + jnp.int32(self.consts.ROUND_DELAY_TICKS) + spawn_int

        return state.replace(
            suit_x_fp       = jnp.zeros(N, dtype=jnp.int32),
            suit_y_fp       = jnp.zeros(N, dtype=jnp.int32),
            suit_dx_fp      = jnp.zeros(N, dtype=jnp.int32),
            suit_color      = jnp.zeros(N, dtype=jnp.int32),
            suit_active     = jnp.zeros(N, dtype=jnp.bool_),
            suits_spawned   = jnp.int32(0),
            next_spawn_step = next_spawn,
            round_failed    = jnp.bool_(False),
            delay_ticks     = jnp.int32(self.consts.ROUND_DELAY_TICKS),
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def _advance_round(self, state: LostLuggageState) -> LostLuggageState:
        """All 25 caught: move to the next round, 5-second delay."""
        N = self.consts.MAX_ACTIVE_SUITS
        # compute spawn interval for the upcoming round (state.round_num + 1)
        spawn_int = _compute_spawn_interval(state.round_num + jnp.int32(1), self.consts) * jnp.int32(self.consts.GRAVITY_FRAMES)
        next_spawn = state.tick + jnp.int32(self.consts.ROUND_DELAY_TICKS) + spawn_int

        return state.replace(
            round_num       = state.round_num + jnp.int32(1),
            suit_x_fp       = jnp.zeros(N, dtype=jnp.int32),
            suit_y_fp       = jnp.zeros(N, dtype=jnp.int32),
            suit_dx_fp      = jnp.zeros(N, dtype=jnp.int32),
            suit_color      = jnp.zeros(N, dtype=jnp.int32),
            suit_active     = jnp.zeros(N, dtype=jnp.bool_),
            suits_spawned   = jnp.int32(0),
            next_spawn_step = next_spawn,
            round_failed    = jnp.bool_(False),
            delay_ticks     = jnp.int32(self.consts.ROUND_DELAY_TICKS),
        )

    # ── Observation / reward / done / info ────────────────────────────────────

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: LostLuggageState) -> LostLuggageObservation:
        # Player: treat the two-passenger unit as one object (bounding box)
        player = ObjectObservation.create(
            x      = state.player_x,
            y      = state.player_y,
            width  = jnp.int32(self.consts.PASS_B_REL_X + self.consts.PASS_W),
            height = jnp.int32(self.consts.PASS_H),
            active = jnp.int32(1),
        )
        sx = state.suit_x_fp >> 4
        sy = state.suit_y_fp >> 4
        suitcases = ObjectObservation.create(
            x      = sx,
            y      = sy,
            width  = jnp.full(self.consts.MAX_ACTIVE_SUITS, self.consts.SUIT_W,  dtype=jnp.int32),
            height = jnp.full(self.consts.MAX_ACTIVE_SUITS, self.consts.SUIT_H,  dtype=jnp.int32),
            active = state.suit_active.astype(jnp.int32),
        )

        return LostLuggageObservation(
            player    = player,
            suitcases = suitcases,
            score     = state.score,
            lives     = state.lives,
            round_num = state.round_num,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, state: LostLuggageState) -> float:
        return jnp.float32(state.score)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: LostLuggageState) -> bool:
        return state.lives <= jnp.int32(0)

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: LostLuggageState) -> LostLuggageInfo:
        return LostLuggageInfo(
            score     = state.score,
            lives     = state.lives,
            round_num = state.round_num,
            tick      = state.tick,
        )

    def get_renderer(self):
        return self.renderer

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        raster = self.jr.create_object_raster(self.BACKGROUND)
        # --- Plane ---
        plane_mask = self.SHAPE_MASKS.get('plane_0', None)
        if plane_mask is not None:
            plane_x = state.plane_x_fp >> jnp.int32(4)
            plane_y = state.plane_y
            raster = self.jr.render_at(raster, plane_x, plane_y, plane_mask)

        # --- Player ---
        pm_default = self.SHAPE_MASKS["passenger"]
        pm_stand = self.SHAPE_MASKS.get("passenger_standing", pm_default)

        # choose mask: standing if pose index==1 else default
        chosen_mask = jnp.where(state.passenger_pose_index == jnp.int32(1), pm_stand, pm_default)

        raster = self.jr.render_at(raster, state.player_x, state.player_y, chosen_mask)
        raster = self.jr.render_at(raster, state.player_x + self.consts.PASS_B_REL_X, state.player_y, chosen_mask)

        # --- Suitcases ---

        # suit_masks_stacked shape: (7, H_s, W_s)
        suit_masks = self.SUIT_MASKS_STACKED

        sx = state.suit_x_fp >> 4
        sy = state.suit_y_fp >> 4

        def render_one(i, r):
            active    = state.suit_active[i]
            x         = sx[i]
            y         = sy[i]
            color_idx = state.suit_color[i]
            mask      = suit_masks[color_idx]          # (H_s, W_s)
            return lax.cond(
                active,
                lambda r_: self.jr.render_at(r_, x, y, mask),
                lambda r_: r_,
                r,
            )

        raster = jax.lax.fori_loop(
            0,
            self.consts.MAX_ACTIVE_SUITS,
            render_one,
            raster
        )

        # --- Flashing light ---
        flash_mask = self.SHAPE_MASKS.get('flashing_light', None)
        if flash_mask is not None:
            # Show only during active round
            active_round = (state.delay_ticks == jnp.int32(0)) & (~state.round_failed)
            # toggle every FLASH_TOGGLE_FRAMES frames
            period = jnp.int32(self.consts.FLASH_TOGGLE_FRAMES)
            visible = ((state.tick // period) % jnp.int32(2)) == jnp.int32(0)
            show = active_round & visible
            def draw(r):
                return self.jr.render_at(r,
                                         jnp.int32(self.consts.FLASH_X),
                                         jnp.int32(self.consts.FLASH_Y),
                                         flash_mask)
            raster = lax.cond(show, draw, lambda r: r, raster)

        # --- Lives icons ---
        # Render lives starting at the left border (left-to-right). 
        life_mask = self.SHAPE_MASKS["life"]

        def render_life(i, raster):
            return jax.lax.cond(
                i < state.lives,
                lambda r: self.jr.render_at(
                    r,
                    jnp.int32(self.consts.LIFE_X) + i * jnp.int32(self.consts.LIFE_SPACING),
                    jnp.int32(self.consts.LIFE_Y) + jnp.int32(2),
                    life_mask
                ),
                lambda r: r,
                raster
            )

        raster = jax.lax.fori_loop(
            0,
            self.consts.MAX_LIVES,
            render_life,
            raster
        )
        
        # --- Score digits ---
        # Render right-aligned so digits grow to the left.
        max_digits = 6
        spacing = 8
        digits = self.jr.int_to_digits(state.score, max_digits=max_digits)
        digit_masks = self.SHAPE_MASKS["digits"]

        # Find first non-zero digit index (most significant). If all zeros,
        # render exactly one zero digit (start at last index).
        nonzero_mask = (digits != 0)
        any_nonzero = jnp.any(nonzero_mask)
        first_nonzero = jnp.argmax(nonzero_mask)
        start_index = jnp.where(any_nonzero, first_nonzero, jnp.int32(max_digits - 1))
        num_to_render = jnp.int32(max_digits) - start_index

        # Right-align: compute x so the rightmost digit sits where SCORE_X would be
        right_x = jnp.int32(self.consts.SCORE_X) + (jnp.int32(max_digits) - num_to_render) * jnp.int32(spacing)

        raster = self.jr.render_label_selective(
            raster,
            right_x,
            jnp.int32(self.consts.SCORE_Y),
            digits,
            digit_masks,
            start_index,
            num_to_render,
            spacing=spacing,
            max_digits_to_render=max_digits,
        )

        return self.jr.render_from_palette(raster, self.PALETTE)
