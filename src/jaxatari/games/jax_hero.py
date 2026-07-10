"""
JAX implementation of H.E.R.O. (Helicopter Emergency Rescue Operation) —
Activision, 1984. Levels 1-3, measured pixel-for-pixel from the real ROM.

Ground truth / provenance
-------------------------
Everything gameplay-visible here was measured directly from the Activision
H.E.R.O. ROM running under ALE (see scripts/hero_record_level.py and the
capture notes below): the per-room background pixels, wall collision
rectangles, palette, HUD layout, player/creature/miner sprites, physics
constants, weapon behaviour and room-transition model. The measured data
lives in hero_levels.py (palette-indexed RLE screens + object tables).

Screen-flip world model (measured)
----------------------------------
The original game does NOT scroll: each level is a stack of STATIC screens
("rooms", 142 rows of cave + fixed HUD below). The player's screen y spans
~8..144; crossing the bottom flips to the next room with the player placed
just above the top (and vice versa flying up). Level 1 has 2 rooms, level 2
has 4, level 3 has 6. Rescuing the miner in the last room advances to the
next level (score/lives carry over, power and dynamite refill); rescuing the
LAST level's miner completes the episode.

Measured mechanics
------------------
  * Fall: constant 1 px/frame (no acceleration). Thrust (UP): the fall stops
    immediately, then rise ramps ~0.125 px/frame^2 up to 2 px/frame.
  * Walk: exactly 1 px/frame.
  * Laser: a 1-px-tall beam anchored at the player's head that EXTENDS while
    fire is held (~4 px/frame, max 26 px) and vanishes on release. Kills
    creatures only (+50); it does not affect walls.
  * Dynamite: laid with DOWN (a stick at the feet), fuse 60 frames (~2 s,
    per the manual; playability fix over the ~26-frame ROM reading), then a
    brief blast that kills creatures (+50) and breaks a destructible wall
    (+75). Blasting the opening-room pillar is the way down through each
    level; level 2 room 1's pillar foot is breakable too. (Project decision
    2026-07-10: dynamite is the wall-breaker, the classic H.E.R.O. mechanic,
    rather than the raw-ROM laser-melt.)
  * Death (creature touch, blast, power depletion): lose a life and respawn
    at the top of the CURRENT room (measured); power refills.
  * Miner rescue: +1000 plus the remaining power as bonus points.
  * Lives: start 4 (measured), +1 every 20000 points (max 6, per manual).

Known approximations (non-measured details)
-------------------------------------------
Leg/rotor animation frames, the explosion flash sprite, the score digit font
for digits not observed in captures, and the ACTIVISION logo lettering are
hand-drawn approximations in the measured palette. The power drain rate uses
a fine-grained internal gauge (4000 units at 1/frame) whose HUD bar and
end-of-level bonus magnitude match the original's behaviour.

Conventions follow the other games in this package (see jax_freeway.py):
constants subclass AutoDerivedConstants; state/observation/info are
flax.struct dataclass pytrees; the env subclasses JaxEnvironment and the
renderer subclasses JAXGameRenderer.
"""

import os
from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, List
from flax import struct

from jaxatari.environment import JaxEnvironment, ObjectObservation, JAXAtariAction as Action
import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.modification import AutoDerivedConstants
from jaxatari.games import hero_levels as HL


# ---------------------------------------------------------------------------
# Pack the measured level data into fixed-shape (padded) arrays indexed by
# (level, room). Unused slots are marked invalid so they never collide/draw.
# ---------------------------------------------------------------------------
_NUM_LEVELS = HL.NUM_LEVELS
_ROOMS = HL.ROOMS_PER_LEVEL
_MAX_ROOMS = max(_ROOMS)
_RECTS = [HL.WALL_RECTS_L1, HL.WALL_RECTS_L2, HL.WALL_RECTS_L3]
# +2 spare slots: carving the destructible zones out of the static rects can
# split one rect into two (see _build_level_arrays)
_MAX_WALLS = max(len(r) for lv in _RECTS for r in lv) + 2
_MAX_SPIDERS = max(len(s) for s in HL.SPIDERS)
_MAX_DWALLS = max(len(d) for d in HL.DESTRUCTIBLE)


def _build_level_arrays():
    nL, nR, nW, nS = _NUM_LEVELS, _MAX_ROOMS, _MAX_WALLS, _MAX_SPIDERS
    walls = np.zeros((nL, nR, nW, 4), np.int32)
    wall_valid = np.zeros((nL, nR, nW), bool)
    rooms_n = np.zeros((nL,), np.int32)
    miner = np.zeros((nL, 3), np.int32)           # room, x, y
    sp_room = np.zeros((nL, nS), np.int32)
    sp_x = np.zeros((nL, nS), np.int32)
    sp_y = np.zeros((nL, nS), np.int32)
    sp_patrol = np.zeros((nL, nS), np.int32)
    sp_valid = np.zeros((nL, nS), bool)
    nD = _MAX_DWALLS
    dw = np.zeros((nL, nD, 6), np.int32)          # room, x, y, w, h, dyn_ok
    dw_valid = np.zeros((nL, nD), bool)
    for li in range(nL):
        rooms_n[li] = _ROOMS[li]
        for ri, rects in enumerate(_RECTS[li]):
            for wi, (x, y, w, h) in enumerate(rects):
                walls[li, ri, wi] = (x, y, w, h)
                wall_valid[li, ri, wi] = True
        miner[li] = HL.MINER_POS[li]
        for si, (rm, x, y, patrol) in enumerate(HL.SPIDERS[li]):
            sp_room[li, si], sp_x[li, si], sp_y[li, si] = rm, x, y
            sp_patrol[li, si] = patrol
            sp_valid[li, si] = True
        for di, (rm, x, y, w, h, dyn_ok) in enumerate(HL.DESTRUCTIBLE[li]):
            dw[li, di] = (rm, x, y, w, h, dyn_ok)
            dw_valid[li, di] = True
            # carve the destructible zone out of the STATIC rects: every rect
            # overlapping the zone (the decomposition may have split a pillar
            # into several columns) loses the zone's y-range
            for wi in range(walls.shape[2]):
                wx, wy, ww, wh = walls[li, rm, wi]
                if not wall_valid[li, rm, wi]:
                    continue
                if wx >= x + w or wx + ww <= x:      # no x overlap
                    continue
                if wy >= y + h or wy + wh <= y:      # no y overlap
                    continue
                assert wx >= x and wx + ww <= x + w, \
                    f"static rect {wx, wy, ww, wh} extends beyond the zone"
                above_h = max(0, y - wy)
                below_h = max(0, wy + wh - (y + h))
                if above_h > 0:
                    walls[li, rm, wi] = (wx, wy, ww, above_h)
                else:
                    wall_valid[li, rm, wi] = False
                if below_h > 0:
                    free = int(np.argmin(wall_valid[li, rm]))
                    assert not wall_valid[li, rm, free], "no spare wall slot"
                    walls[li, rm, free] = (wx, y + h, ww, below_h)
                    wall_valid[li, rm, free] = True
    return dict(walls=walls, wall_valid=wall_valid, rooms_n=rooms_n, miner=miner,
                sp_room=sp_room, sp_x=sp_x, sp_y=sp_y, sp_patrol=sp_patrol,
                sp_valid=sp_valid, dw=dw, dw_valid=dw_valid)


_LV = _build_level_arrays()


class HeroConstants(AutoDerivedConstants):
    # --- Screen (ALE native) ---
    screen_width: int = struct.field(pytree_node=False, default=160)
    screen_height: int = struct.field(pytree_node=False, default=210)
    cave_bottom: int = struct.field(pytree_node=False, default=142)

    num_levels: int = struct.field(pytree_node=False, default=_NUM_LEVELS)
    max_rooms: int = struct.field(pytree_node=False, default=_MAX_ROOMS)

    # --- Player: 6-px collision box; the drawn sprite is 9x24 and centred
    # over the box (measured box; the wider art matches the running pose). ---
    player_width: int = struct.field(pytree_node=False, default=6)
    player_height: int = struct.field(pytree_node=False, default=24)
    spawn_x: int = struct.field(pytree_node=False, default=HL.SPAWN[0])
    spawn_y: int = struct.field(pytree_node=False, default=HL.SPAWN[1])
    respawn_x: int = struct.field(pytree_node=False, default=HL.RESPAWN[0])
    respawn_y: int = struct.field(pytree_node=False, default=HL.RESPAWN[1])

    # Room flip thresholds (player TOP y; the flip happens when the measured
    # centre row crosses the screen edge: centre = top + 11).
    flip_bottom_y: int = struct.field(pytree_node=False, default=133)
    flip_enter_top_y: int = struct.field(pytree_node=False, default=-3)

    # --- Flight physics (measured) ---
    fall_speed: float = struct.field(pytree_node=False, default=1.0)
    thrust_accel: float = struct.field(pytree_node=False, default=0.125)
    thrust_spinup: int = struct.field(pytree_node=False, default=16)
    max_rise_speed: float = struct.field(pytree_node=False, default=-2.0)
    move_speed: int = struct.field(pytree_node=False, default=1)

    # --- Laser (measured: extends while held) ---
    laser_growth: int = struct.field(pytree_node=False, default=4)
    laser_max_length: int = struct.field(pytree_node=False, default=26)
    laser_height: int = struct.field(pytree_node=False, default=1)

    # --- Dynamite ---
    # Fuse: 60 frames (~2 s, per the manual). The raw ROM capture measured
    # ~26 frames, but at 1 px/frame walk speed that leaves a ~2 px escape
    # margin — unplayable for a human (playability fix, 2026-07-10: fleeing
    # right after planting must clear the blast comfortably).
    dyn_width: int = struct.field(pytree_node=False, default=4)
    dyn_height: int = struct.field(pytree_node=False, default=10)
    dyn_fuse: int = struct.field(pytree_node=False, default=60)
    explosion_frames: int = struct.field(pytree_node=False, default=4)
    explosion_radius: int = struct.field(pytree_node=False, default=16)
    starting_dynamite: int = struct.field(pytree_node=False, default=6)

    # --- Power / lives / scoring ---
    max_power: int = struct.field(pytree_node=False, default=4000)
    power_drain_per_frame: int = struct.field(pytree_node=False, default=1)
    # brief creature-proof grace after a respawn (the respawn spot can sit
    # inside a creature's patrol zone, as on the real console)
    respawn_invuln: int = struct.field(pytree_node=False, default=60)
    starting_lives: int = struct.field(pytree_node=False, default=4)
    max_lives: int = struct.field(pytree_node=False, default=6)
    extra_life_score: int = struct.field(pytree_node=False, default=20000)
    creature_points: int = struct.field(pytree_node=False, default=50)
    wall_points: int = struct.field(pytree_node=False, default=75)
    miner_points: int = struct.field(pytree_node=False, default=1000)

    # --- Spiders (sprite 7x11: 6 thread rows + 5 body rows; bob +-5;
    # level-3 critters additionally patrol horizontally, measured range) ---
    num_spiders: int = struct.field(pytree_node=False, default=_MAX_SPIDERS)
    spider_width: int = struct.field(pytree_node=False, default=7)
    spider_height: int = struct.field(pytree_node=False, default=11)
    spider_body_top: int = struct.field(pytree_node=False, default=6)
    spider_bob_amp: int = struct.field(pytree_node=False, default=5)
    spider_bob_half_period: int = struct.field(pytree_node=False, default=20)
    spider_patrol_half_period: int = struct.field(pytree_node=False, default=40)

    # --- Miner (sprite 8x12) ---
    miner_width: int = struct.field(pytree_node=False, default=8)
    miner_height: int = struct.field(pytree_node=False, default=12)

    num_walls: int = struct.field(pytree_node=False, default=_MAX_WALLS)

    # --- Level-indexed measured data ---
    LEVEL_ROOMS: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["rooms_n"], dtype=jnp.int32))
    ROOM_WALLS: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["walls"], dtype=jnp.int32))
    ROOM_WALL_VALID: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["wall_valid"], dtype=jnp.bool_))
    LEVEL_MINER: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["miner"], dtype=jnp.int32))
    SPIDER_ROOM: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["sp_room"], dtype=jnp.int32))
    SPIDER_X: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["sp_x"], dtype=jnp.int32))
    SPIDER_Y: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["sp_y"], dtype=jnp.int32))
    SPIDER_PATROL: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["sp_patrol"], dtype=jnp.int32))
    SPIDER_VALID: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["sp_valid"], dtype=jnp.bool_))
    # Destructible walls: (room, x, y, w, h, dynamite_ok) per slot. A dynamite
    # blast destroys a dyn_ok wall outright (the laser does not affect walls).
    num_dwalls: int = struct.field(pytree_node=False, default=_MAX_DWALLS)
    DESTRUCT: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["dw"], dtype=jnp.int32))
    DESTRUCT_VALID: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["dw_valid"], dtype=jnp.bool_))

    # --- HUD layout (measured from the ROM frame) ---
    hud_top: int = struct.field(pytree_node=False, default=142)
    hud_bottom: int = struct.field(pytree_node=False, default=189)
    power_bar_x: int = struct.field(pytree_node=False, default=49)
    power_bar_y: int = struct.field(pytree_node=False, default=145)
    power_bar_width: int = struct.field(pytree_node=False, default=78)
    power_bar_height: int = struct.field(pytree_node=False, default=5)
    lives_x: int = struct.field(pytree_node=False, default=81)
    lives_y: int = struct.field(pytree_node=False, default=152)
    lives_spacing: int = struct.field(pytree_node=False, default=8)
    dyn_icons_x: int = struct.field(pytree_node=False, default=59)
    dyn_icons_y: int = struct.field(pytree_node=False, default=166)
    dyn_icons_spacing: int = struct.field(pytree_node=False, default=8)
    score_right_x: int = struct.field(pytree_node=False, default=103)
    score_y: int = struct.field(pytree_node=False, default=179)
    score_digit_pitch: int = struct.field(pytree_node=False, default=8)

    # --- Colors (measured) ---
    bg_color: Tuple[int, int, int] = struct.field(pytree_node=False, default=(0, 0, 0))
    hud_color: Tuple[int, int, int] = struct.field(pytree_node=False, default=(111, 111, 111))
    laser_color: Tuple[int, int, int] = struct.field(pytree_node=False, default=(232, 232, 74))
    power_color: Tuple[int, int, int] = struct.field(pytree_node=False, default=(232, 232, 74))
    text_color: Tuple[int, int, int] = struct.field(pytree_node=False, default=(214, 214, 214))
    explosion_color: Tuple[int, int, int] = struct.field(pytree_node=False, default=(252, 232, 120))

    def compute_derived(self):
        return {}


# ---------------------------------------------------------------------------
# State / Observation / Info
# ---------------------------------------------------------------------------
@struct.dataclass
class HeroState:
    player_x: chex.Array          # sprite-left, screen px
    player_y: chex.Array          # sprite-top, screen px (room-local)
    player_vy: chex.Array         # <0 rising, >0 falling
    thrust_timer: chex.Array      # frames UP has been held (rotor spin-up)
    room: chex.Array              # current room within the level
    facing: chex.Array            # -1 left, +1 right
    walk_timer: chex.Array
    has_moved: chex.Array         # power only drains after the first move
    laser_len: chex.Array         # 0 = off; grows while fire held
    power: chex.Array
    lives: chex.Array
    score: chex.Array
    level: chex.Array
    dynamite_count: chex.Array
    dyn_active: chex.Array
    dyn_x: chex.Array
    dyn_y: chex.Array
    dyn_room: chex.Array
    dyn_fuse: chex.Array
    explosion_timer: chex.Array
    spider_alive: chex.Array      # (num_spiders,)
    wall_stage: chex.Array        # (num_dwalls,) 0 intact / 2 blasted away
    invuln_timer: chex.Array      # creature-proof frames after a respawn
    miner_rescued: chex.Array
    level_complete: chex.Array
    step_counter: chex.Array
    game_over: chex.Array
    rng_key: chex.PRNGKey


@struct.dataclass
class HeroObservation:
    player: ObjectObservation
    laser: ObjectObservation
    spiders: ObjectObservation
    miner: ObjectObservation
    dynamite: ObjectObservation
    walls: ObjectObservation
    power: chex.Array
    lives: chex.Array
    score: chex.Array
    dynamite_count: chex.Array
    level: chex.Array
    room: chex.Array


@struct.dataclass
class HeroInfo:
    time: chex.Array
    power: chex.Array
    lives: chex.Array
    dynamite_count: chex.Array
    all_rewards: chex.Array


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class JaxHero(JaxEnvironment[HeroState, HeroObservation, HeroInfo, HeroConstants]):
    # Compact agent action set -> ALE console actions.
    # DOWN (and DOWNFIRE) lays dynamite, as on the real console. DOWNLEFT /
    # DOWNRIGHT plant while already running, so "plant and flee" works as a
    # single held input (a human holding LEFT and tapping DOWN must not stall).
    ACTION_SET: jnp.ndarray = jnp.array([
        Action.NOOP,
        Action.FIRE,
        Action.UP,
        Action.RIGHT,
        Action.LEFT,
        Action.DOWN,
        Action.UPRIGHT,
        Action.UPLEFT,
        Action.RIGHTFIRE,
        Action.LEFTFIRE,
        Action.UPFIRE,
        Action.DOWNFIRE,
        Action.DOWNRIGHT,
        Action.DOWNLEFT,
    ], dtype=jnp.int32)

    def __init__(self, consts: HeroConstants = None):
        if consts is None:
            consts = HeroConstants()
        super().__init__(consts)
        self.renderer = HeroRenderer(self.consts)

    # --- reset ------------------------------------------------------------
    def reset(self, key: jax.random.PRNGKey = None) -> Tuple[HeroObservation, HeroState]:
        if key is None:
            key = jax.random.PRNGKey(0)
        c = self.consts
        state = HeroState(
            player_x=jnp.array(c.spawn_x, dtype=jnp.int32),
            player_y=jnp.array(c.spawn_y, dtype=jnp.int32),
            player_vy=jnp.array(0.0, dtype=jnp.float32),
            thrust_timer=jnp.array(0, dtype=jnp.int32),
            room=jnp.array(0, dtype=jnp.int32),
            facing=jnp.array(1, dtype=jnp.int32),
            walk_timer=jnp.array(0, dtype=jnp.int32),
            has_moved=jnp.array(False, dtype=jnp.bool_),
            laser_len=jnp.array(0, dtype=jnp.int32),
            power=jnp.array(c.max_power, dtype=jnp.int32),
            lives=jnp.array(c.starting_lives, dtype=jnp.int32),
            score=jnp.array(0, dtype=jnp.int32),
            level=jnp.array(0, dtype=jnp.int32),
            dynamite_count=jnp.array(c.starting_dynamite, dtype=jnp.int32),
            dyn_active=jnp.array(False, dtype=jnp.bool_),
            dyn_x=jnp.array(0, dtype=jnp.int32),
            dyn_y=jnp.array(0, dtype=jnp.int32),
            dyn_room=jnp.array(0, dtype=jnp.int32),
            dyn_fuse=jnp.array(0, dtype=jnp.int32),
            explosion_timer=jnp.array(0, dtype=jnp.int32),
            spider_alive=self.consts.SPIDER_VALID[0],
            wall_stage=jnp.zeros((c.num_dwalls,), dtype=jnp.int32),
            invuln_timer=jnp.array(0, dtype=jnp.int32),
            miner_rescued=jnp.array(False, dtype=jnp.bool_),
            level_complete=jnp.array(False, dtype=jnp.bool_),
            step_counter=jnp.array(0, dtype=jnp.int32),
            game_over=jnp.array(False, dtype=jnp.bool_),
            rng_key=key,
        )
        return self._get_observation(state), state

    # --- helpers ----------------------------------------------------------
    @staticmethod
    def _aabb(ax, ay, aw, ah, bx, by, bw, bh):
        return ((ax < bx + bw) & (ax + aw > bx) &
                (ay < by + bh) & (ay + ah > by))

    def _spider_pos(self, state):
        """Current (x, y) of every spider slot of the level: the measured
        vertical bob on the thread plus the measured horizontal patrol for
        the level-3 critters (patrol halfwidth 0 = pure bobber)."""
        c = self.consts
        lvl = state.level
        t = state.step_counter % (2 * c.spider_bob_half_period)
        bob = jnp.abs(t - c.spider_bob_half_period) * (2 * c.spider_bob_amp) \
            // c.spider_bob_half_period - c.spider_bob_amp
        tp = state.step_counter % (2 * c.spider_patrol_half_period)
        tri = jnp.abs(tp - c.spider_patrol_half_period) * 2 \
            - c.spider_patrol_half_period          # -P .. +P triangle
        sweep = (c.SPIDER_PATROL[lvl] * tri) // c.spider_patrol_half_period
        return c.SPIDER_X[lvl] + sweep, c.SPIDER_Y[lvl] + bob

    def _dwall_rects(self, state):
        """Solid rect of every destructible wall of the level: intact = the
        full rect, blasted (stage 2) = returned with w=0 so it never collides
        and the passage behind it opens."""
        c = self.consts
        dw = c.DESTRUCT[state.level]                    # (nD, 6)
        broken = state.wall_stage >= 2
        w = jnp.where(broken, 0, dw[:, 3])
        solid = c.DESTRUCT_VALID[state.level] & (~broken)
        return dw[:, 0], dw[:, 1], dw[:, 2], w, dw[:, 4], solid

    def _hits_wall(self, state, px, py):
        """Player box vs the current room's measured wall rects (+ the
        remaining parts of the destructible walls). The top 2 rows of the
        sprite are the 1-px rotor shaft and do not collide (the blast hole in
        level 2 is 23 px tall and the real player fits through it)."""
        c = self.consts
        box_y, box_h = py + 2, c.player_height - 2
        lvl, room = state.level, state.room
        walls = c.ROOM_WALLS[lvl, room]
        valid = c.ROOM_WALL_VALID[lvl, room]
        hit = jnp.any(self._aabb(px, box_y, c.player_width, box_h,
                                 walls[:, 0], walls[:, 1], walls[:, 2], walls[:, 3]) & valid)
        d_room, dx, dy_, dwid, dh, solid = self._dwall_rects(state)
        d_hit = jnp.any(solid & (d_room == room) &
                        self._aabb(px, box_y, c.player_width, box_h,
                                   dx, dy_, dwid, dh))
        return hit | d_hit

    # --- step -------------------------------------------------------------
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: HeroState, action: int) -> Tuple[HeroObservation, HeroState, float, bool, HeroInfo]:
        c = self.consts
        atari_action = jnp.take(self.ACTION_SET, action)
        lvl = state.level

        up = jnp.isin(atari_action, jnp.array(
            [Action.UP, Action.UPRIGHT, Action.UPLEFT, Action.UPFIRE], dtype=jnp.int32))
        left = jnp.isin(atari_action, jnp.array(
            [Action.LEFT, Action.UPLEFT, Action.LEFTFIRE, Action.DOWNLEFT], dtype=jnp.int32))
        right = jnp.isin(atari_action, jnp.array(
            [Action.RIGHT, Action.UPRIGHT, Action.RIGHTFIRE, Action.DOWNRIGHT], dtype=jnp.int32))
        down = jnp.isin(atari_action, jnp.array(
            [Action.DOWN, Action.DOWNFIRE, Action.DOWNLEFT, Action.DOWNRIGHT], dtype=jnp.int32))
        laser_fire = jnp.isin(atari_action, jnp.array(
            [Action.FIRE, Action.UPFIRE, Action.LEFTFIRE, Action.RIGHTFIRE], dtype=jnp.int32))

        # --- lay dynamite (DOWN, as on the console) ---
        can_place = down & (state.dynamite_count > 0) & (~state.dyn_active) & (state.explosion_timer <= 0)
        dyn_active = state.dyn_active | can_place
        dyn_x = jnp.where(can_place, state.player_x - 2, state.dyn_x).astype(jnp.int32)
        dyn_y = jnp.where(can_place, state.player_y + c.player_height - c.dyn_height, state.dyn_y).astype(jnp.int32)
        dyn_room = jnp.where(can_place, state.room, state.dyn_room).astype(jnp.int32)
        dyn_fuse = jnp.where(can_place, c.dyn_fuse, state.dyn_fuse).astype(jnp.int32)
        dynamite_count = jnp.where(can_place, state.dynamite_count - 1, state.dynamite_count).astype(jnp.int32)

        # --- horizontal movement (measured 1 px/frame) ---
        dx = (right.astype(jnp.int32) - left.astype(jnp.int32)) * c.move_speed
        cand_x = jnp.clip(state.player_x + dx, 8, c.screen_width - 8 - c.player_width).astype(jnp.int32)
        x_blocked = self._hits_wall(state, cand_x, state.player_y)
        new_x = jnp.where(x_blocked, state.player_x, cand_x).astype(jnp.int32)
        new_facing = jnp.where(right, 1, jnp.where(left, -1, state.facing)).astype(jnp.int32)

        # --- vertical: constant fall; thrust cancels the fall instantly,
        # hovers through the measured 16-frame rotor spin-up, then ramps ---
        thrust_timer = jnp.where(up, state.thrust_timer + 1, 0).astype(jnp.int32)
        vy_up = jnp.where(thrust_timer <= c.thrust_spinup, 0.0,
                          jnp.maximum(jnp.minimum(state.player_vy, 0.0) - c.thrust_accel,
                                      c.max_rise_speed))
        new_vy = jnp.where(up, vy_up, c.fall_speed).astype(jnp.float32)
        dy = jnp.round(new_vy).astype(jnp.int32)
        cand_y = (state.player_y + dy).astype(jnp.int32)
        # ceiling of the topmost room: don't fly above the level start screen
        cand_y = jnp.where((state.room == 0) & (cand_y < 8), jnp.int32(8), cand_y)
        y_blocked = self._hits_wall(state, new_x, cand_y)
        new_y = jnp.where(y_blocked, state.player_y, cand_y).astype(jnp.int32)
        new_vy = jnp.where(y_blocked, 0.0, new_vy).astype(jnp.float32)

        # --- room flips (measured screen-flip model) ---
        n_rooms = c.LEVEL_ROOMS[lvl]
        flip_down = (new_y > c.flip_bottom_y) & (state.room < n_rooms - 1)
        flip_up = (new_y < c.flip_enter_top_y) & (state.room > 0)
        new_room = (state.room + flip_down.astype(jnp.int32) - flip_up.astype(jnp.int32)).astype(jnp.int32)
        new_y = jnp.where(flip_down, c.flip_enter_top_y,
                          jnp.where(flip_up, c.flip_bottom_y, new_y)).astype(jnp.int32)
        # keep the player inside the last room's floor
        new_y = jnp.clip(new_y, -24, c.cave_bottom - 2)

        moved_input = up | left | right | down
        has_moved = state.has_moved | moved_input
        moved_h = new_x != state.player_x
        walk_timer = jnp.where(moved_h, state.walk_timer + 1, 0).astype(jnp.int32)

        # --- laser: extends while held, vanishes on release (measured) ---
        laser_len = jnp.where(laser_fire,
                              jnp.minimum(state.laser_len + c.laser_growth, c.laser_max_length),
                              0).astype(jnp.int32)
        laser_on = laser_len > 0
        lx = jnp.where(new_facing < 0, new_x - laser_len, new_x + c.player_width).astype(jnp.int32)
        ly = (new_y + 1).astype(jnp.int32)

        # --- dynamite fuse / explosion ---
        new_fuse = jnp.where(dyn_active, dyn_fuse - 1, dyn_fuse).astype(jnp.int32)
        explode_now = dyn_active & (new_fuse <= 0)
        explosion_timer = jnp.where(explode_now, c.explosion_frames,
                                    jnp.maximum(0, state.explosion_timer - 1)).astype(jnp.int32)
        dyn_active = dyn_active & (~explode_now)

        r = c.explosion_radius
        ex, ey, ew, eh = dyn_x - r, dyn_y - r, 2 * r + c.dyn_width, 2 * r + c.dyn_height
        blast_here = explode_now & (dyn_room == new_room)

        # --- destructible walls: a dynamite blast destroys a dyn_ok wall
        # outright (the laser does not affect walls) ---
        dw = c.DESTRUCT[lvl]                              # (nD, 6)
        d_room, d_x, d_y, d_w, d_h, d_solid = self._dwall_rects(state)
        dyn_break = (d_solid & (dw[:, 5] > 0) & explode_now & (d_room == dyn_room) &
                     self._aabb(ex, ey, ew, eh, d_x, d_y, d_w, d_h))
        wall_stage = jnp.where(dyn_break, 2, state.wall_stage).astype(jnp.int32)
        # a wall removed this frame scores once (+75)
        walls_broken = jnp.sum(((state.wall_stage < 2) & (wall_stage >= 2)).astype(jnp.int32))

        # --- spiders: bob on their thread; killed by laser or blast ---
        sp_x, sp_y = self._spider_pos(state)
        sp_room = c.SPIDER_ROOM[lvl]
        sp_here = state.spider_alive & (sp_room == new_room)
        body_y = sp_y + c.spider_body_top
        spider_laser = (laser_on & sp_here &
                        self._aabb(lx, ly, laser_len, c.laser_height,
                                   sp_x, body_y, c.spider_width, c.spider_height - c.spider_body_top))
        spider_blast = (state.spider_alive & (sp_room == dyn_room) & explode_now &
                        self._aabb(ex, ey, ew, eh, sp_x, body_y,
                                   c.spider_width, c.spider_height - c.spider_body_top))
        spider_kill = spider_laser | spider_blast
        spider_alive = state.spider_alive & (~spider_kill)
        creatures_killed = jnp.sum(spider_kill.astype(jnp.int32))

        # --- player death conditions (creatures can't kill during the brief
        # post-respawn grace) ---
        died_spider = ((state.invuln_timer <= 0) &
                       jnp.any(sp_here & (~spider_kill) &
                               self._aabb(new_x, new_y, c.player_width, c.player_height,
                                          sp_x, body_y, c.spider_width,
                                          c.spider_height - c.spider_body_top)))
        died_blast = blast_here & self._aabb(new_x, new_y, c.player_width, c.player_height,
                                             ex, ey, ew, eh)

        # --- power drain (starts after the first move) ---
        drain = jnp.where(has_moved, c.power_drain_per_frame, 0)
        new_power = jnp.maximum(0, state.power - drain).astype(jnp.int32)
        died_power = (new_power <= 0) & (state.power > 0)

        # --- miner rescue (last room of the level) ---
        m = c.LEVEL_MINER[lvl]
        touch_miner = ((~state.miner_rescued) & (new_room == m[0]) &
                       self._aabb(new_x, new_y, c.player_width, c.player_height,
                                  m[1], m[2], c.miner_width, c.miner_height))
        power_bonus = jnp.where(touch_miner, new_power, 0).astype(jnp.int32)

        is_last = lvl >= (c.num_levels - 1)
        finish = touch_miner & is_last
        advance = touch_miner & (~is_last)

        # --- scoring (+ extra life every 20000 points, manual) ---
        new_score = (state.score
                     + creatures_killed * c.creature_points
                     + walls_broken * c.wall_points
                     + touch_miner.astype(jnp.int32) * c.miner_points
                     + power_bonus).astype(jnp.int32)
        extra_lives = (new_score // c.extra_life_score - state.score // c.extra_life_score).astype(jnp.int32)

        # --- death / lives / respawn (top of the CURRENT room, measured) ---
        died = (died_blast | died_spider | died_power) & (~touch_miner)
        new_lives = jnp.clip(state.lives - died.astype(jnp.int32) + extra_lives,
                             0, c.max_lives).astype(jnp.int32)
        respawned = died & (new_lives > 0)

        next_lvl = jnp.clip(lvl + advance.astype(jnp.int32), 0, c.num_levels - 1)
        # The measured respawn spot can sit inside a wall in some rooms (e.g.
        # level 2 room 1's pillar): fall back to the spawn column, then to
        # the descent-shaft column, whichever is free in this room.
        rs_state = state.replace(room=new_room)
        rs_bad_1 = self._hits_wall(rs_state, jnp.int32(c.respawn_x), jnp.int32(c.respawn_y))
        rs_bad_2 = self._hits_wall(rs_state, jnp.int32(c.spawn_x), jnp.int32(c.spawn_y))
        rs_x = jnp.where(rs_bad_1, jnp.where(rs_bad_2, 76, c.spawn_x), c.respawn_x)
        rs_y = jnp.where(rs_bad_1, jnp.where(rs_bad_2, 8, c.spawn_y), c.respawn_y)
        final_x = jnp.where(advance, c.spawn_x,
                            jnp.where(respawned, rs_x, new_x)).astype(jnp.int32)
        final_y = jnp.where(advance, c.spawn_y,
                            jnp.where(respawned, rs_y, new_y)).astype(jnp.int32)
        final_room = jnp.where(advance, 0, new_room).astype(jnp.int32)

        reset_pose = respawned | advance
        final_vy = jnp.where(reset_pose, 0.0, new_vy).astype(jnp.float32)
        final_thrust = jnp.where(reset_pose, 0, thrust_timer).astype(jnp.int32)
        final_facing = jnp.where(reset_pose, 1, new_facing).astype(jnp.int32)
        final_walk_timer = jnp.where(reset_pose, 0, walk_timer).astype(jnp.int32)
        final_has_moved = has_moved & (~reset_pose)
        final_laser = jnp.where(reset_pose, 0, laser_len).astype(jnp.int32)
        final_dyn_active = dyn_active & (~reset_pose)
        final_explosion = jnp.where(reset_pose, 0, explosion_timer).astype(jnp.int32)
        final_dyn_fuse = jnp.where(reset_pose, 0, new_fuse).astype(jnp.int32)
        # Power refills on respawn or advance; dynamite refills on a new level.
        final_power = jnp.where(reset_pose, c.max_power, new_power).astype(jnp.int32)
        final_dyn_count = jnp.where(advance, c.starting_dynamite, dynamite_count).astype(jnp.int32)

        final_spider_alive = jnp.where(advance, c.SPIDER_VALID[next_lvl], spider_alive)
        # walls reset intact on a new level; a respawn keeps a blasted wall gone
        final_wall_stage = jnp.where(advance, 0, wall_stage).astype(jnp.int32)
        final_invuln = jnp.where(respawned, c.respawn_invuln,
                                 jnp.maximum(0, state.invuln_timer - 1)).astype(jnp.int32)

        final_level = next_lvl.astype(jnp.int32)
        final_miner_rescued = jnp.where(advance, False, state.miner_rescued | touch_miner)
        level_complete = state.level_complete | finish

        game_over = state.game_over | (died & (new_lives <= 0))
        new_step = (state.step_counter + 1).astype(jnp.int32)

        new_state = HeroState(
            player_x=final_x,
            player_y=final_y,
            player_vy=final_vy,
            thrust_timer=final_thrust,
            room=final_room,
            facing=final_facing,
            walk_timer=final_walk_timer,
            has_moved=final_has_moved,
            laser_len=final_laser,
            power=final_power,
            lives=new_lives,
            score=new_score,
            level=final_level,
            dynamite_count=final_dyn_count,
            dyn_active=final_dyn_active,
            dyn_x=dyn_x,
            dyn_y=dyn_y,
            dyn_room=dyn_room,
            dyn_fuse=final_dyn_fuse,
            explosion_timer=final_explosion,
            spider_alive=final_spider_alive,
            wall_stage=final_wall_stage,
            invuln_timer=final_invuln,
            miner_rescued=final_miner_rescued,
            level_complete=level_complete,
            step_counter=new_step,
            game_over=game_over,
            rng_key=state.rng_key,
        )

        done = self._get_done(new_state)
        reward = self._get_reward(state, new_state)
        all_rewards = self._get_all_rewards(state, new_state)
        obs = self._get_observation(new_state)
        info = self._get_info(new_state, all_rewards)
        return obs, new_state, reward, done, info

    # --- observation / info / reward / done -------------------------------
    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: HeroState) -> HeroObservation:
        c = self.consts
        lvl = state.level
        player = ObjectObservation.create(
            x=state.player_x, y=state.player_y,
            width=jnp.array(c.player_width, jnp.int32),
            height=jnp.array(c.player_height, jnp.int32),
            active=jnp.array(True, jnp.bool_),
            orientation=jnp.where(state.facing < 0, 270, 90).astype(jnp.int32),
        )

        laser_on = state.laser_len > 0
        lx = jnp.where(state.facing < 0, state.player_x - state.laser_len,
                       state.player_x + c.player_width).astype(jnp.int32)
        laser = ObjectObservation.create(
            x=lx, y=(state.player_y + 1).astype(jnp.int32),
            width=state.laser_len.astype(jnp.int32),
            height=jnp.array(c.laser_height, jnp.int32),
            active=laser_on,
            orientation=jnp.where(state.facing < 0, 270, 90).astype(jnp.int32),
        )

        sp_x, sp_y = self._spider_pos(state)
        spiders = ObjectObservation.create(
            x=sp_x.astype(jnp.int32),
            y=(sp_y + c.spider_body_top).astype(jnp.int32),
            width=jnp.full((c.num_spiders,), c.spider_width, jnp.int32),
            height=jnp.full((c.num_spiders,), c.spider_height - c.spider_body_top, jnp.int32),
            active=state.spider_alive & (c.SPIDER_ROOM[lvl] == state.room),
            visual_id=jnp.arange(c.num_spiders, dtype=jnp.int32),
        )

        m = c.LEVEL_MINER[lvl]
        miner = ObjectObservation.create(
            x=m[1].astype(jnp.int32), y=m[2].astype(jnp.int32),
            width=jnp.array(c.miner_width, jnp.int32),
            height=jnp.array(c.miner_height, jnp.int32),
            active=(~state.miner_rescued) & (state.room == m[0]),
        )

        dynamite = ObjectObservation.create(
            x=state.dyn_x, y=state.dyn_y,
            width=jnp.array(c.dyn_width, jnp.int32),
            height=jnp.array(c.dyn_height, jnp.int32),
            active=((state.dyn_active | (state.explosion_timer > 0)) &
                    (state.dyn_room == state.room)),
            state=state.explosion_timer,
        )

        walls_r = c.ROOM_WALLS[lvl, state.room]
        walls = ObjectObservation.create(
            x=walls_r[:, 0], y=walls_r[:, 1], width=walls_r[:, 2], height=walls_r[:, 3],
            active=c.ROOM_WALL_VALID[lvl, state.room],
        )

        return HeroObservation(
            player=player, laser=laser, spiders=spiders,
            miner=miner, dynamite=dynamite, walls=walls,
            power=state.power, lives=state.lives, score=state.score,
            dynamite_count=state.dynamite_count, level=state.level, room=state.room,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: HeroState, all_rewards: jnp.ndarray = None) -> HeroInfo:
        if all_rewards is None:
            all_rewards = jnp.zeros(1, dtype=jnp.float32)
        return HeroInfo(
            time=state.step_counter, power=state.power, lives=state.lives,
            dynamite_count=state.dynamite_count, all_rewards=all_rewards,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: HeroState, state: HeroState) -> float:
        return (state.score - previous_state.score).astype(jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_rewards(self, previous_state: HeroState, state: HeroState) -> jnp.ndarray:
        return jnp.array([self._get_reward(previous_state, state)], dtype=jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: HeroState) -> bool:
        return state.game_over | state.level_complete

    # --- spaces -----------------------------------------------------------
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTION_SET))

    def observation_space(self) -> spaces.Dict:
        c = self.consts
        screen = (c.screen_height, c.screen_width)
        return spaces.Dict({
            "player": spaces.get_object_space(n=None, screen_size=screen),
            "laser": spaces.get_object_space(n=None, screen_size=screen),
            "spiders": spaces.get_object_space(n=c.num_spiders, screen_size=screen),
            "miner": spaces.get_object_space(n=None, screen_size=screen),
            "dynamite": spaces.get_object_space(n=None, screen_size=screen),
            "walls": spaces.get_object_space(n=c.num_walls, screen_size=screen),
            "power": spaces.Box(low=0, high=c.max_power, shape=(), dtype=jnp.int32),
            "lives": spaces.Box(low=0, high=c.max_lives, shape=(), dtype=jnp.int32),
            "score": spaces.Box(low=0, high=jnp.iinfo(jnp.int32).max, shape=(), dtype=jnp.int32),
            "dynamite_count": spaces.Box(low=0, high=c.starting_dynamite, shape=(), dtype=jnp.int32),
            "level": spaces.Box(low=0, high=c.num_levels, shape=(), dtype=jnp.int32),
            "room": spaces.Box(low=0, high=c.max_rooms, shape=(), dtype=jnp.int32),
        })

    def image_space(self) -> spaces.Box:
        c = self.consts
        return spaces.Box(low=0, high=255,
                          shape=(c.screen_height, c.screen_width, 3), dtype=jnp.uint8)

    def render(self, state: HeroState) -> jnp.ndarray:
        return self.renderer.render(state)


# ---------------------------------------------------------------------------
# Renderer — measured sprites and HUD over the measured room backgrounds.
# ---------------------------------------------------------------------------
# Pixel-art tables transcribed from ROM frame captures. Characters map to RGB
# via _ART_PALETTE ('.' = transparent).
_ART_PALETTE = {
    '.': None,
    'Y': (232, 232, 74),      # rotor / dynamite-icon fuse (yellow)
    'R': (184, 50, 50),       # helmet red
    'r': (200, 72, 72),       # helmet highlight
    'B': (84, 138, 210),      # suit blue
    'W': (214, 214, 214),     # legs / dynamite stick / digits (white)
    'S': (170, 170, 170),     # spider thread / miner girder (silver)
    '1': (223, 183, 85),      # spider body gradient (measured warm ramp)
    '2': (210, 164, 74),
    '3': (195, 144, 61),
    '4': (180, 122, 48),
    '5': (162, 98, 33),
    'P': (228, 111, 111),     # miner head pink
    'G': (92, 186, 92),       # miner body green
    'y': (252, 252, 84),      # miner lamp spark
    'b': (45, 87, 176),       # HUD mini-hero suit blue
    'K': (50, 132, 50),       # level-2 breakable pillar tone
}

# Roderick (facing right), 9 wide x 24 tall. Built from stacked sections:
# rotor (3) + helmet/prop-pack (5) + suit (6) + legs (10). The visual sprite
# is wider than the 6-px collision box (player_width) and is drawn centred
# over it (see the render call). While walking the legs animate through a
# running stride (wide<->narrow stance); while airborne the rotor spins and
# the legs dangle together.
_HERO_TOP_BLADE = [          # spinning rotor (airborne)
    "YYYYYYYY.",
    "....Y....",
    "....Y....",
]
_HERO_TOP_T = [              # rotor at rest: the wide T of the reference art
    ".YYYYY...",
    "...Y.....",
    "...Y.....",
]
_HERO_HEAD = [               # red helmet (left) + prop-pack "c" (right)
    ".RR..RRR.",
    ".RR..R...",
    ".RR..RRR.",
    ".RR......",
    ".RR......",
]
_HERO_SUIT = [               # blue body with the arms notched out
    "BBBB.BBBB",
    "BBBBBBBBB",
    "BBBBBBBBB",
    ".BBBBBBB.",
    ".BB...BB.",
    "..B...B..",
]
_HERO_LEGS_CONTACT = [       # run, contact pose: front leg planted forward,
    "...WWW...",             # back leg trailing bent up behind (the pose of
    "..WWWWW..",             # the reference art)
    ".WWW.WW..",
    "WWW..WW..",
    "WW....WW.",
    "......WW.",
    "......WW.",
    ".....WW..",
    ".....WW..",
    ".....WWWW",
]
_HERO_LEGS_PASS = [          # run, passing pose: planted leg under the hip,
    "...WWW...",             # other knee lifted (alternates with contact)
    "..WWWWW..",
    "..WW.WW..",
    "..WW.WWW.",
    "..WW..WW.",
    "..WW.....",
    "..WW.....",
    "..WW.....",
    "..WW.....",
    "..WWWW...",
]
_HERO_LEGS_TOGETHER = [      # dangling (flight / idle)
    "...WWW...",
    "...WWW...",
    "..WWWWW..",
    "..WW.WW..",
    "..WW.WW..",
    "..WW.WW..",
    "..WW.WW..",
    "..WW.WW..",
    "..WW.WW..",
    "..WWWWW..",
]
# Ground poses show the resting T-rotor and run through the two stride
# poses; airborne poses spin the rotor and dangle the legs.
_PLAYER_IDLE = _HERO_TOP_T + _HERO_HEAD + _HERO_SUIT + _HERO_LEGS_TOGETHER
_PLAYER_WALK0 = _HERO_TOP_T + _HERO_HEAD + _HERO_SUIT + _HERO_LEGS_CONTACT
_PLAYER_WALK1 = _HERO_TOP_T + _HERO_HEAD + _HERO_SUIT + _HERO_LEGS_PASS
_PLAYER_FLY0 = _HERO_TOP_T + _HERO_HEAD + _HERO_SUIT + _HERO_LEGS_TOGETHER
_PLAYER_FLY1 = _HERO_TOP_BLADE + _HERO_HEAD + _HERO_SUIT + _HERO_LEGS_TOGETHER

# Hanging spider: 6 thread rows + 5 body rows (measured).
_SPIDER_ART = [
    "...S...",
    "...S...",
    "...S...",
    "...S...",
    "...S...",
    "...S...",
    ".1.1.1.",
    "2.222.2",
    "3.333.3",
    "4..4..4",
    "5.....5",
]

# Trapped miner, 8 wide x 12 tall (measured).
_MINER_ART = [
    "...y....",
    "..SSS...",
    "..PP....",
    "..PP....",
    "..P.....",
    ".GGG....",
    ".G.G....",
    ".G.G....",
    ".G.G.G..",
    ".SS.SSS.",
    ".SSSS.S.",
    "..SS..SS",
]

# Dynamite stick (measured: slanted white stick).
_DYN_ART = [
    ".WW.",
    ".WW.",
    ".W..",
    ".W..",
    "..W.",
    "..W.",
    "..W.",
    "..WW",
    "..WW",
    "...W",
]

# (Destructible walls stay baked in the room backgrounds; the renderer
# stamps black over the melted/blasted slices — see _build_dwall_stamps.)

# HUD mini-hero life icon, 5 wide x 13 tall (measured).
_LIFE_ICON_ART = [
    "YYY..",
    ".Y...",
    ".R.RR",
    ".R.RR",
    ".b.b.",
    ".bb.b",
    "..b.b",
    "..b.b",
    "...W.",
    "...W.",
    "..WW.",
    "..W..",
    "..W..",
]

# HUD dynamite stick icon, 3 wide x 11 tall (measured).
_DYN_ICON_ART = [".YY"] * 5 + ["RRR"] * 6

# Activision-style 8x6 score digit font ('0' verified against the ROM frame).
_DIGIT_FONT_8 = {
    0: ["011110", "110011", "110011", "110011", "110011", "110011", "110011", "011110"],
    1: ["001100", "011100", "001100", "001100", "001100", "001100", "001100", "111111"],
    2: ["011110", "110011", "000011", "000110", "001100", "011000", "110000", "111111"],
    3: ["011110", "110011", "000011", "001110", "000011", "110011", "110011", "011110"],
    4: ["000110", "001110", "011110", "110110", "111111", "000110", "000110", "000110"],
    5: ["111111", "110000", "111110", "000011", "000011", "110011", "110011", "011110"],
    6: ["011110", "110000", "110000", "111110", "110011", "110011", "110011", "011110"],
    7: ["111111", "000011", "000110", "001100", "011000", "011000", "011000", "011000"],
    8: ["011110", "110011", "110011", "011110", "110011", "110011", "110011", "011110"],
    9: ["011110", "110011", "110011", "011111", "000011", "000011", "110011", "011110"],
}

# 5x3 letter font for the HUD "POWER" label and the logo lettering.
_LETTER_FONT = {
    'P': ["111", "101", "111", "100", "100"],
    'O': ["111", "101", "101", "101", "111"],
    'W': ["101", "101", "101", "111", "101"],
    'E': ["111", "100", "111", "100", "111"],
    'R': ["111", "101", "111", "110", "101"],
    'A': ["111", "101", "111", "101", "101"],
    'C': ["111", "100", "100", "100", "111"],
    'T': ["111", "010", "010", "010", "010"],
    'I': ["111", "010", "010", "010", "111"],
    'V': ["101", "101", "101", "101", "010"],
    'S': ["111", "100", "111", "001", "111"],
    'N': ["101", "111", "111", "111", "101"],
    ' ': ["000", "000", "000", "000", "000"],
}


class HeroRenderer(JAXGameRenderer):
    """Draws the measured per-room backgrounds (decoded from hero_levels.py)
    with dynamic sprites and the measured HUD on top. Rooms are static
    screens; state.room selects which background is shown (screen-flip)."""

    def __init__(self, consts: HeroConstants = None, config: render_utils.RendererConfig = None):
        self.consts = consts or HeroConstants()
        super().__init__(self.consts)
        c = self.consts

        if config is None:
            self.config = render_utils.RendererConfig(
                game_dimensions=(c.screen_height, c.screen_width), channels=3, downscale=None)
        else:
            self.config = config
        self.jr = render_utils.JaxRenderingUtils(self.config)

        palettes = [HL.PALETTE_L1, HL.PALETTE_L2, HL.PALETTE_L3]
        blobs = [HL.BG_RLE_L1, HL.BG_RLE_L2, HL.BG_RLE_L3]

        asset_config = [
            {'name': 'background', 'type': 'background', 'data': self._build_background()},
            {'name': 'player_idle', 'type': 'procedural', 'data': self._sprite(_PLAYER_IDLE)},
            {'name': 'player_walk0', 'type': 'procedural', 'data': self._sprite(_PLAYER_WALK0)},
            {'name': 'player_walk1', 'type': 'procedural', 'data': self._sprite(_PLAYER_WALK1)},
            {'name': 'player_fly0', 'type': 'procedural', 'data': self._sprite(_PLAYER_FLY0)},
            {'name': 'player_fly1', 'type': 'procedural', 'data': self._sprite(_PLAYER_FLY1)},
            {'name': 'spider', 'type': 'procedural', 'data': self._sprite(_SPIDER_ART)},
            {'name': 'miner', 'type': 'procedural', 'data': self._sprite(_MINER_ART)},
            {'name': 'dynamite', 'type': 'procedural', 'data': self._sprite(_DYN_ART)},
            {'name': 'explosion', 'type': 'procedural', 'data': self._build_explosion(c.explosion_radius)},
            {'name': 'laser_unit', 'type': 'procedural', 'data': self._solid(1, 2, c.laser_color)},
            {'name': 'power_unit', 'type': 'procedural', 'data': self._solid(c.power_bar_height, 1, c.power_color)},
            {'name': 'life_icon', 'type': 'procedural', 'data': self._sprite(_LIFE_ICON_ART)},
            {'name': 'dyn_icon', 'type': 'procedural', 'data': self._sprite(_DYN_ICON_ART)},
            {'name': 'digits', 'type': 'digits', 'data': self._build_digits(c.text_color)},
        ]
        # measured per-room cave backgrounds (black-padded for short levels)
        for li in range(c.num_levels):
            for ri in range(c.max_rooms):
                if ri < len(blobs[li]):
                    img = HL.decode_bg(blobs[li][ri], palettes[li])
                else:
                    img = np.zeros((c.cave_bottom, c.screen_width, 3), np.uint8)
                rgba = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
                asset_config.append({'name': f'bg_{li}_{ri}', 'type': 'procedural',
                                     'data': jnp.asarray(rgba)})
        # black stamps for the destructible walls: per (level, slot, stage),
        # padded to the largest wall. wall_stage is 0 (intact, draws nothing)
        # or 2 (blasted, blanks the whole wall); stage 1 is unused.
        max_dw_h = max(int(v) for v in _LV["dw"][:, :, 4].flatten()) or 1
        max_dw_w = max(int(v) for v in _LV["dw"][:, :, 3].flatten()) or 1
        for li in range(c.num_levels):
            for di in range(c.num_dwalls):
                _, _, _, w, h, _ = (int(v) for v in _LV["dw"][li, di])
                for stage in range(3):
                    stamp = np.zeros((max_dw_h, max_dw_w, 4), np.uint8)
                    if _LV["dw_valid"][li, di] and stage >= 2:
                        stamp[:h, :w, 3] = 255          # opaque black over the wall
                    asset_config.append({'name': f'dwall_{li}_{di}_{stage}',
                                         'type': 'procedural', 'data': jnp.asarray(stamp)})

        sprite_path = os.path.join(render_utils.get_base_sprite_dir(), "hero")
        (
            self.PALETTE, self.SHAPE_MASKS, self.BACKGROUND,
            self.COLOR_TO_ID, self.FLIP_OFFSETS,
        ) = self.jr.load_and_setup_assets(asset_config, sprite_path)

        self.PLAYER_FRAMES = jnp.stack([
            self.SHAPE_MASKS["player_idle"],
            self.SHAPE_MASKS["player_walk0"],
            self.SHAPE_MASKS["player_walk1"],
            self.SHAPE_MASKS["player_fly0"],
            self.SHAPE_MASKS["player_fly1"],
        ])
        self.BGS = jnp.stack([
            jnp.stack([self.SHAPE_MASKS[f"bg_{li}_{ri}"] for ri in range(c.max_rooms)])
            for li in range(c.num_levels)
        ])  # (nL, nR, 142, 160)
        self.DWALL_STAMPS = jnp.stack([
            jnp.stack([
                jnp.stack([self.SHAPE_MASKS[f"dwall_{li}_{di}_{st}"] for st in range(3)])
                for di in range(c.num_dwalls)])
            for li in range(c.num_levels)
        ])  # (nL, nD, 3, H, W)

    # --- procedural asset builders ----------------------------------------
    @staticmethod
    def _solid(h: int, w: int, color: Tuple[int, int, int]) -> jnp.ndarray:
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, 0:3] = np.array(color, dtype=np.uint8)
        rgba[:, :, 3] = 255
        return jnp.asarray(rgba)

    @staticmethod
    def _sprite(art: List[str]) -> jnp.ndarray:
        h, w = len(art), len(art[0])
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        for r, row in enumerate(art):
            for col, ch in enumerate(row):
                color = _ART_PALETTE[ch]
                if color is not None:
                    rgba[r, col, 0:3] = np.array(color, dtype=np.uint8)
                    rgba[r, col, 3] = 255
        return jnp.asarray(rgba)

    def _build_explosion(self, radius: int) -> jnp.ndarray:
        """Brief radial flash (the real blast lasts 1-2 frames; approximate)."""
        c = self.consts
        size = 2 * radius
        yy, xx = np.mgrid[0:size, 0:size]
        dist = np.sqrt((xx - radius + 0.5) ** 2 + (yy - radius + 0.5) ** 2)
        rgba = np.zeros((size, size, 4), dtype=np.uint8)
        outer = dist <= radius
        core = dist <= radius * 0.5
        rgba[outer, 0:3] = np.array((232, 232, 74), dtype=np.uint8)
        rgba[outer, 3] = 255
        rgba[core, 0:3] = np.array(c.explosion_color, dtype=np.uint8)
        return jnp.asarray(rgba)

    def _build_digits(self, color: Tuple[int, int, int]) -> jnp.ndarray:
        digits = np.zeros((10, 8, 6, 4), dtype=np.uint8)
        for d, rows in _DIGIT_FONT_8.items():
            for r, row in enumerate(rows):
                for col, ch in enumerate(row):
                    if ch == "1":
                        digits[d, r, col, 0:3] = np.array(color, dtype=np.uint8)
                        digits[d, r, col, 3] = 255
        return jnp.asarray(digits)

    def _build_background(self) -> jnp.ndarray:
        """Static screen layer: black cave area (rooms composited per frame)
        + the measured HUD: gray panel rows 142-188 (x8+), POWER label,
        black band, and the ACTIVISION wordmark rows 194-200."""
        c = self.consts
        bg = np.zeros((c.screen_height, c.screen_width, 4), dtype=np.uint8)
        bg[:, :, 3] = 255
        bg[c.hud_top:c.hud_bottom, 8:c.screen_width, 0:3] = np.array(c.hud_color, dtype=np.uint8)

        def text(s, x, y, color):
            cx = x
            for ch in s:
                glyph = _LETTER_FONT.get(ch, _LETTER_FONT[' '])
                for r, row in enumerate(glyph):
                    for cc, v in enumerate(row):
                        if v == '1':
                            bg[y + r, cx + cc, 0:3] = np.array(color, dtype=np.uint8)
                cx += 4
            return cx

        text("POWER", 26, c.power_bar_y, c.text_color)

        # ACTIVISION wordmark (approximate lettering; measured row tints).
        row_tints = [(184, 50, 50), (180, 122, 48), (210, 210, 64),
                     (110, 156, 66), (45, 50, 184)]
        word = "ACTIVISION"
        x0 = (c.screen_width - len(word) * 4) // 2
        cx = x0
        for ch in word:
            glyph = _LETTER_FONT[ch]
            for r, row in enumerate(glyph):
                for cc, v in enumerate(row):
                    if v == '1':
                        bg[195 + r, cx + cc, 0:3] = np.array((214, 214, 214), dtype=np.uint8)
            cx += 4
        bg[194, x0:cx, 0:3] = np.array(row_tints[0], dtype=np.uint8)
        bg[200, x0:cx, 0:3] = np.array(row_tints[4], dtype=np.uint8)
        return jnp.asarray(bg)

    # --- render -----------------------------------------------------------
    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: HeroState) -> jnp.ndarray:
        c = self.consts
        lvl, room = state.level, state.room

        raster = self.jr.create_object_raster(self.BACKGROUND)
        cave = self.BGS[lvl, room]
        raster = jax.lax.dynamic_update_slice(raster, cave, (0, 0))

        def maybe(cond, x, y, mask, ras, flip=False):
            return jax.lax.cond(
                cond,
                lambda r: self.jr.render_at_clipped(
                    r, jnp.asarray(x, jnp.int32), jnp.asarray(y, jnp.int32),
                    mask, flip_horizontal=flip),
                lambda r: r,
                ras,
            )

        # destructible walls: baked in the background; stamp black over the
        # melted/blasted slices per the wall's current stage
        for di in range(c.num_dwalls):
            dwr = c.DESTRUCT[lvl, di]
            stage = jnp.clip(state.wall_stage[di], 0, 2)
            raster = maybe(c.DESTRUCT_VALID[lvl, di] & (room == dwr[0]),
                           dwr[1], dwr[2], self.DWALL_STAMPS[lvl, di, stage], raster)

        # miner (last room)
        m = c.LEVEL_MINER[lvl]
        raster = maybe((~state.miner_rescued) & (room == m[0]),
                       m[1], m[2], self.SHAPE_MASKS["miner"], raster)

        # spiders: bob on their threads (+ measured horizontal patrol on L3)
        t = state.step_counter % (2 * c.spider_bob_half_period)
        bob = jnp.abs(t - c.spider_bob_half_period) * (2 * c.spider_bob_amp) \
            // c.spider_bob_half_period - c.spider_bob_amp
        tp = state.step_counter % (2 * c.spider_patrol_half_period)
        tri = jnp.abs(tp - c.spider_patrol_half_period) * 2 - c.spider_patrol_half_period
        sweep = (c.SPIDER_PATROL[lvl] * tri) // c.spider_patrol_half_period
        sp_room = c.SPIDER_ROOM[lvl]
        for i in range(c.num_spiders):
            raster = maybe(state.spider_alive[i] & (sp_room[i] == room),
                           c.SPIDER_X[lvl, i] + sweep[i], c.SPIDER_Y[lvl, i] + bob,
                           self.SHAPE_MASKS["spider"], raster)

        # dynamite + explosion flash
        dyn_here = state.dyn_room == room
        raster = maybe(state.dyn_active & dyn_here, state.dyn_x, state.dyn_y,
                       self.SHAPE_MASKS["dynamite"], raster)
        raster = maybe((state.explosion_timer > 0) & dyn_here,
                       state.dyn_x - c.explosion_radius, state.dyn_y - c.explosion_radius,
                       self.SHAPE_MASKS["explosion"], raster)

        # player: pose = idle / walking (2 frames) / airborne (rotor spin);
        # hovering during the thrust spin-up also shows the spinning rotor
        airborne = (jnp.abs(state.player_vy) > 0.5) | (state.thrust_timer > 0)
        walk_frame = 1 + (state.walk_timer // 4) % 2
        fly_frame = 3 + (state.step_counter // 2) % 2
        frame = jnp.where(airborne, fly_frame,
                          jnp.where(state.walk_timer > 0, walk_frame, 0))
        # the 9-px sprite is drawn centred over the 6-px collision box
        sprite_dx = (self.PLAYER_FRAMES.shape[2] - c.player_width) // 2
        raster = self.jr.render_at_clipped(
            raster, state.player_x - sprite_dx, state.player_y,
            self.PLAYER_FRAMES[frame],
            flip_horizontal=(state.facing < 0))

        # laser: growing 1-px beam anchored at the head (drawn in 2px units)
        lx0 = jnp.where(state.facing < 0, state.player_x - state.laser_len,
                        state.player_x + c.player_width)
        ly = state.player_y + 1

        def draw_seg(i, ras):
            return jax.lax.cond(
                i * 2 < state.laser_len,
                lambda r: self.jr.render_at_clipped(
                    r, lx0 + i * 2, ly, self.SHAPE_MASKS["laser_unit"]),
                lambda r: r,
                ras,
            )
        raster = jax.lax.fori_loop(0, c.laser_max_length // 2, draw_seg, raster)

        # keep sprites out of the HUD band
        raster = raster.at[c.cave_bottom:, :].set(self.BACKGROUND[c.cave_bottom:, :])

        # --- HUD (measured layout) ---
        power_units = (state.power * c.power_bar_width) // c.max_power

        def draw_power(i, ras):
            return jax.lax.cond(
                i < power_units,
                lambda r: self.jr.render_at_clipped(
                    r, jnp.array(c.power_bar_x) + i, jnp.array(c.power_bar_y),
                    self.SHAPE_MASKS["power_unit"]),
                lambda r: r,
                ras,
            )
        raster = jax.lax.fori_loop(0, c.power_bar_width, draw_power, raster)

        # reserve lives as mini-heroes (lives - 1, measured 3 icons at start)
        raster = self.jr.render_indicator(raster, c.lives_x, c.lives_y,
                                          jnp.maximum(0, state.lives - 1),
                                          self.SHAPE_MASKS["life_icon"],
                                          spacing=c.lives_spacing,
                                          max_value=c.max_lives - 1)

        # dynamite sticks remaining
        raster = self.jr.render_indicator(raster, c.dyn_icons_x, c.dyn_icons_y,
                                          state.dynamite_count, self.SHAPE_MASKS["dyn_icon"],
                                          spacing=c.dyn_icons_spacing,
                                          max_value=c.starting_dynamite)

        # score: right-aligned, no leading zeros (measured position)
        score_digits = self.jr.int_to_digits(state.score, max_digits=6)
        n_digits = jnp.maximum(1, (jnp.floor(
            jnp.log10(jnp.maximum(state.score, 1).astype(jnp.float32))) + 1).astype(jnp.int32))
        start = 6 - n_digits
        x_start = c.score_right_x - n_digits * c.score_digit_pitch
        raster = self.jr.render_label_selective(
            raster, x_start, c.score_y, score_digits,
            self.SHAPE_MASKS["digits"], start, n_digits,
            spacing=c.score_digit_pitch, max_digits_to_render=6)

        return self.jr.render_from_palette(raster, self.PALETTE)
