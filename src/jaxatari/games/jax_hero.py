"""
JAX implementation of H.E.R.O. (Helicopter Emergency Rescue Operation) — Activision, 1983.

Roderick Hero descends into a mine with a helicopter backpack to rescue a trapped
miner. This file implements a complete, JIT-compatible **Level 1** following the
provided level specification:

Active objects
--------------
  * Roderick Hero (player): walks left/right, flies up (gravity/drift momentum),
    shoots a horizontal laser, and places dynamite.
  * The trapped miner: sits still at the bottom; touching him ends the level.
  * Spiders: hang on strings and move up/down along a fixed vertical line.
  * Walls: solid blocks that block movement; one wall is breakable with dynamite.

Level 1 rules (implemented)
---------------------------
  * Starting inventory: 4 lives (1 active + 3 reserve), 6 dynamites, 100% power.
  * Cannot pass through solid walls / floors.
  * Power bar drains 1 unit every frame *after the player first moves*; 0 = death.
  * Collisions:
      - Player touches spider          -> player dies (lose 1 life, restart at top)
      - Player inside dynamite blast    -> player dies
      - Laser hits spider               -> spider gone, +50 points
      - Dynamite blast hits breakable wall -> wall gone, +75 points
      - Player touches miner            -> level ends, remaining power -> points

Conventions follow the other games in this package (see jax_freeway.py):
constants subclass AutoDerivedConstants; state/observation/info are
flax.struct.dataclass pytrees; the env subclasses JaxEnvironment and the
renderer subclasses JAXGameRenderer. Graphics use procedural colored sprites so
the environment runs without sprite files; real Atari sprites can be dropped into
~/.local/share/jaxatari/sprites/hero/ later by swapping the asset entries.
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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Level 1 wall layout. Each row is (x, y, w, h). The breakable wall is index 6.
# Non-breakable walls are baked into the background; the breakable one is drawn
# dynamically so it can vanish.
_WALLS = [
    (8,   16, 28, 28),   # 0 top-left block (hangs from ceiling)
    (72,  16, 20, 36),   # 1 top-center pillar
    (124, 16, 28, 28),   # 2 top-right block
    (8,   76, 80, 22),   # 3 left ledge
    (112, 76, 40, 22),   # 4 right ledge   (descent gap is x in [88, 112])
    (120, 112, 32, 20),  # 5 lower-right block
    (8,   120, 32, 10),  # 6 BREAKABLE cap sealing the miner's pocket
    (40,  120, 8, 28),   # 7 pocket right wall
    (8,   148, 144, 16), # 8 floor
]
_WALL_BREAKABLE = [0, 0, 0, 0, 0, 0, 1, 0, 0]
_BREAKABLE_IDX = 6


class HeroConstants(AutoDerivedConstants):
    # --- Screen ---
    screen_width: int = struct.field(pytree_node=False, default=160)
    screen_height: int = struct.field(pytree_node=False, default=210)

    # Playable cave bounds (top banner above, HUD below).
    play_top: int = struct.field(pytree_node=False, default=16)
    play_bottom: int = struct.field(pytree_node=False, default=164)
    play_left: int = struct.field(pytree_node=False, default=8)
    play_right: int = struct.field(pytree_node=False, default=152)

    # --- Player ---
    player_width: int = struct.field(pytree_node=False, default=6)
    player_height: int = struct.field(pytree_node=False, default=13)
    player_start_x: int = struct.field(pytree_node=False, default=44)
    player_start_y: int = struct.field(pytree_node=False, default=24)

    # --- Flight physics (prop thrust vs. gravity) ---
    gravity: float = struct.field(pytree_node=False, default=0.20)
    thrust: float = struct.field(pytree_node=False, default=-0.45)
    max_fall_speed: float = struct.field(pytree_node=False, default=2.5)
    max_rise_speed: float = struct.field(pytree_node=False, default=-2.5)
    move_speed: int = struct.field(pytree_node=False, default=1)

    # --- Laser ---
    laser_length: int = struct.field(pytree_node=False, default=24)
    laser_height: int = struct.field(pytree_node=False, default=2)
    laser_duration: int = struct.field(pytree_node=False, default=6)

    # --- Dynamite ---
    dyn_width: int = struct.field(pytree_node=False, default=3)
    dyn_height: int = struct.field(pytree_node=False, default=6)
    dyn_fuse: int = struct.field(pytree_node=False, default=60)
    explosion_frames: int = struct.field(pytree_node=False, default=8)
    explosion_radius: int = struct.field(pytree_node=False, default=14)
    starting_dynamite: int = struct.field(pytree_node=False, default=6)

    # --- Power / lives / scoring ---
    max_power: int = struct.field(pytree_node=False, default=4000)
    power_drain_per_frame: int = struct.field(pytree_node=False, default=1)
    starting_lives: int = struct.field(pytree_node=False, default=4)
    spider_points: int = struct.field(pytree_node=False, default=50)
    wall_points: int = struct.field(pytree_node=False, default=75)

    # --- Spiders ---
    num_spiders: int = struct.field(pytree_node=False, default=2)
    spider_width: int = struct.field(pytree_node=False, default=7)
    spider_height: int = struct.field(pytree_node=False, default=7)
    spider_move_period: int = struct.field(pytree_node=False, default=3)
    SPIDER_X: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array([96, 64], dtype=jnp.int32))
    SPIDER_Y_MIN: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array([100, 100], dtype=jnp.int32))
    SPIDER_Y_MAX: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array([140, 138], dtype=jnp.int32))
    SPIDER_START_Y: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array([120, 110], dtype=jnp.int32))
    SPIDER_START_DIR: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array([1, -1], dtype=jnp.int32))

    # --- Miner ---
    miner_x: int = struct.field(pytree_node=False, default=14)
    miner_y: int = struct.field(pytree_node=False, default=134)
    miner_width: int = struct.field(pytree_node=False, default=8)
    miner_height: int = struct.field(pytree_node=False, default=12)

    # --- Walls ---
    num_walls: int = struct.field(pytree_node=False, default=len(_WALLS))
    breakable_idx: int = struct.field(pytree_node=False, default=_BREAKABLE_IDX)
    WALLS: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_WALLS, dtype=jnp.int32))
    WALL_BREAKABLE: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_WALL_BREAKABLE, dtype=jnp.bool_))

    # --- HUD layout ---
    power_bar_x: int = struct.field(pytree_node=False, default=44)
    power_bar_y: int = struct.field(pytree_node=False, default=172)
    power_bar_width: int = struct.field(pytree_node=False, default=100)
    power_bar_height: int = struct.field(pytree_node=False, default=5)

    # --- Colors (RGB) ---
    bg_color: Tuple[int, int, int] = struct.field(pytree_node=False, default=(0, 0, 0))
    wall_color: Tuple[int, int, int] = struct.field(pytree_node=False, default=(150, 100, 60))
    breakable_color: Tuple[int, int, int] = struct.field(pytree_node=False, default=(180, 120, 72))
    banner_color: Tuple[int, int, int] = struct.field(pytree_node=False, default=(236, 236, 236))
    hud_color: Tuple[int, int, int] = struct.field(pytree_node=False, default=(45, 45, 45))
    player_color: Tuple[int, int, int] = struct.field(pytree_node=False, default=(132, 144, 252))
    laser_color: Tuple[int, int, int] = struct.field(pytree_node=False, default=(252, 224, 112))
    spider_color: Tuple[int, int, int] = struct.field(pytree_node=False, default=(214, 214, 214))
    miner_color: Tuple[int, int, int] = struct.field(pytree_node=False, default=(66, 200, 66))
    dyn_color: Tuple[int, int, int] = struct.field(pytree_node=False, default=(228, 60, 60))
    explosion_color: Tuple[int, int, int] = struct.field(pytree_node=False, default=(252, 160, 40))
    power_color: Tuple[int, int, int] = struct.field(pytree_node=False, default=(252, 200, 72))
    text_color: Tuple[int, int, int] = struct.field(pytree_node=False, default=(236, 236, 236))

    def compute_derived(self):
        return {}


# ---------------------------------------------------------------------------
# State / Observation / Info
# ---------------------------------------------------------------------------
@struct.dataclass
class HeroState:
    player_x: chex.Array
    player_y: chex.Array
    player_vy: chex.Array
    facing: chex.Array            # -1 left, +1 right
    has_moved: chex.Array         # power only drains after first move
    laser_timer: chex.Array
    power: chex.Array
    lives: chex.Array
    score: chex.Array
    level: chex.Array
    dynamite_count: chex.Array
    dyn_active: chex.Array
    dyn_x: chex.Array
    dyn_y: chex.Array
    dyn_fuse: chex.Array
    explosion_timer: chex.Array
    spider_y: chex.Array          # (num_spiders,)
    spider_dir: chex.Array        # (num_spiders,)
    spider_alive: chex.Array      # (num_spiders,)
    wall_active: chex.Array       # (num_walls,)
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
    # DOWNFIRE places dynamite; the other *FIRE variants shoot the laser.
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
            player_x=jnp.array(c.player_start_x, dtype=jnp.int32),
            player_y=jnp.array(c.player_start_y, dtype=jnp.int32),
            player_vy=jnp.array(0.0, dtype=jnp.float32),
            facing=jnp.array(1, dtype=jnp.int32),
            has_moved=jnp.array(False, dtype=jnp.bool_),
            laser_timer=jnp.array(0, dtype=jnp.int32),
            power=jnp.array(c.max_power, dtype=jnp.int32),
            lives=jnp.array(c.starting_lives, dtype=jnp.int32),
            score=jnp.array(0, dtype=jnp.int32),
            level=jnp.array(0, dtype=jnp.int32),
            dynamite_count=jnp.array(c.starting_dynamite, dtype=jnp.int32),
            dyn_active=jnp.array(False, dtype=jnp.bool_),
            dyn_x=jnp.array(0, dtype=jnp.int32),
            dyn_y=jnp.array(0, dtype=jnp.int32),
            dyn_fuse=jnp.array(0, dtype=jnp.int32),
            explosion_timer=jnp.array(0, dtype=jnp.int32),
            spider_y=c.SPIDER_START_Y.astype(jnp.int32),
            spider_dir=c.SPIDER_START_DIR.astype(jnp.int32),
            spider_alive=jnp.ones((c.num_spiders,), dtype=jnp.bool_),
            wall_active=jnp.ones((c.num_walls,), dtype=jnp.bool_),
            miner_rescued=jnp.array(False, dtype=jnp.bool_),
            level_complete=jnp.array(False, dtype=jnp.bool_),
            step_counter=jnp.array(0, dtype=jnp.int32),
            game_over=jnp.array(False, dtype=jnp.bool_),
            rng_key=key,
        )
        return self._get_observation(state), state

    # --- collision helpers ------------------------------------------------
    @staticmethod
    def _aabb(ax, ay, aw, ah, bx, by, bw, bh):
        """Axis-aligned bounding-box overlap (broadcasts over arrays)."""
        return ((ax < bx + bw) & (ax + aw > bx) &
                (ay < by + bh) & (ay + ah > by))

    def _hits_wall(self, px, py, wall_active):
        """True if a player-sized box at (px, py) overlaps any active wall."""
        c = self.consts
        w = c.WALLS
        overlap = self._aabb(px, py, c.player_width, c.player_height,
                             w[:, 0], w[:, 1], w[:, 2], w[:, 3]) & wall_active
        return jnp.any(overlap)

    # --- step -------------------------------------------------------------
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: HeroState, action: int) -> Tuple[HeroObservation, HeroState, float, bool, HeroInfo]:
        c = self.consts
        atari_action = jnp.take(self.ACTION_SET, action)

        up = jnp.isin(atari_action, jnp.array(
            [Action.UP, Action.UPRIGHT, Action.UPLEFT, Action.UPFIRE], dtype=jnp.int32))
        left = jnp.isin(atari_action, jnp.array(
            [Action.LEFT, Action.UPLEFT, Action.LEFTFIRE], dtype=jnp.int32))
        right = jnp.isin(atari_action, jnp.array(
            [Action.RIGHT, Action.UPRIGHT, Action.RIGHTFIRE], dtype=jnp.int32))
        down = jnp.isin(atari_action, jnp.array(
            [Action.DOWN, Action.DOWNFIRE], dtype=jnp.int32))
        laser_fire = jnp.isin(atari_action, jnp.array(
            [Action.FIRE, Action.UPFIRE, Action.LEFTFIRE, Action.RIGHTFIRE], dtype=jnp.int32))
        dyn_fire = (atari_action == Action.DOWNFIRE)

        # --- place dynamite (uses current player position) ---
        can_place = dyn_fire & (state.dynamite_count > 0) & (~state.dyn_active) & (state.explosion_timer <= 0)
        dyn_active = jnp.where(can_place, True, state.dyn_active)
        dyn_x = jnp.where(can_place, state.player_x + c.player_width // 2, state.dyn_x).astype(jnp.int32)
        dyn_y = jnp.where(can_place, state.player_y + c.player_height - c.dyn_height, state.dyn_y).astype(jnp.int32)
        dyn_fuse = jnp.where(can_place, c.dyn_fuse, state.dyn_fuse).astype(jnp.int32)
        dynamite_count = jnp.where(can_place, state.dynamite_count - 1, state.dynamite_count).astype(jnp.int32)

        # --- horizontal movement with wall collision ---
        dx = (right.astype(jnp.int32) - left.astype(jnp.int32)) * c.move_speed
        cand_x = jnp.clip(state.player_x + dx, c.play_left, c.play_right - c.player_width).astype(jnp.int32)
        x_blocked = self._hits_wall(cand_x, state.player_y, state.wall_active)
        new_x = jnp.where(x_blocked, state.player_x, cand_x).astype(jnp.int32)
        new_facing = jnp.where(right, 1, jnp.where(left, -1, state.facing)).astype(jnp.int32)

        # --- vertical flight with wall collision ---
        accel = jnp.where(up, c.thrust, c.gravity)
        accel = jnp.where(down, accel + c.gravity, accel)
        new_vy = jnp.clip(state.player_vy + accel, c.max_rise_speed, c.max_fall_speed).astype(jnp.float32)
        dy = jnp.round(new_vy).astype(jnp.int32)
        cand_y = jnp.clip(state.player_y + dy, c.play_top, c.play_bottom - c.player_height).astype(jnp.int32)
        y_blocked = self._hits_wall(new_x, cand_y, state.wall_active) | (cand_y != state.player_y + dy)
        new_y = jnp.where(y_blocked, state.player_y, cand_y).astype(jnp.int32)
        new_vy = jnp.where(y_blocked, 0.0, new_vy).astype(jnp.float32)

        moved_input = up | left | right | down
        has_moved = state.has_moved | moved_input

        # --- dynamite fuse / explosion ---
        new_fuse = jnp.where(dyn_active, dyn_fuse - 1, dyn_fuse).astype(jnp.int32)
        explode_now = dyn_active & (new_fuse <= 0)
        explosion_timer = jnp.where(explode_now, c.explosion_frames,
                                    jnp.maximum(0, state.explosion_timer - 1)).astype(jnp.int32)
        dyn_active = jnp.where(explode_now, False, dyn_active)

        # Explosion box (centered on the dynamite).
        r = c.explosion_radius
        ex, ey, ew, eh = dyn_x - r, dyn_y - r, 2 * r, 2 * r

        # Breakable wall destroyed by blast.
        wall_hit = (explode_now &
                    self._aabb(ex, ey, ew, eh,
                               c.WALLS[:, 0], c.WALLS[:, 1], c.WALLS[:, 2], c.WALLS[:, 3]) &
                    c.WALL_BREAKABLE & state.wall_active)
        wall_active = state.wall_active & (~wall_hit)
        walls_broken = jnp.sum(wall_hit.astype(jnp.int32))

        # Player caught in the blast.
        died_blast = explode_now & self._aabb(
            new_x, new_y, c.player_width, c.player_height, ex, ey, ew, eh)

        # --- spiders move up/down on their vertical line ---
        move = (state.step_counter % c.spider_move_period) == 0
        sy = state.spider_y + jnp.where(move, state.spider_dir, 0)
        hit_top = sy <= c.SPIDER_Y_MIN
        hit_bot = sy >= c.SPIDER_Y_MAX
        new_spider_dir = jnp.where(hit_top, 1, jnp.where(hit_bot, -1, state.spider_dir)).astype(jnp.int32)
        new_spider_y = jnp.clip(sy, c.SPIDER_Y_MIN, c.SPIDER_Y_MAX).astype(jnp.int32)

        # --- laser ---
        new_laser_timer = jnp.where(
            laser_fire & (state.laser_timer <= 0),
            c.laser_duration,
            jnp.maximum(0, state.laser_timer - 1),
        ).astype(jnp.int32)
        laser_on = new_laser_timer > 0
        lx = jnp.where(new_facing < 0, new_x - c.laser_length, new_x + c.player_width).astype(jnp.int32)
        ly = (new_y + c.player_height // 3).astype(jnp.int32)

        # Laser kills spiders it overlaps.
        laser_hit = (laser_on & state.spider_alive &
                     self._aabb(lx, ly, c.laser_length, c.laser_height,
                                c.SPIDER_X, new_spider_y, c.spider_width, c.spider_height))
        spider_alive = state.spider_alive & (~laser_hit)
        spiders_killed = jnp.sum(laser_hit.astype(jnp.int32))

        # --- player touches spider ---
        player_spider = (spider_alive &
                         self._aabb(new_x, new_y, c.player_width, c.player_height,
                                    c.SPIDER_X, new_spider_y, c.spider_width, c.spider_height))
        died_spider = jnp.any(player_spider)

        # --- power drain ---
        drain = jnp.where(has_moved, c.power_drain_per_frame, 0)
        new_power = jnp.maximum(0, state.power - drain).astype(jnp.int32)
        died_power = (new_power <= 0) & (state.power > 0)

        # --- miner rescue ---
        touch_miner = (~state.miner_rescued) & self._aabb(
            new_x, new_y, c.player_width, c.player_height,
            c.miner_x, c.miner_y, c.miner_width, c.miner_height)
        miner_rescued = state.miner_rescued | touch_miner
        level_complete = state.level_complete | touch_miner
        power_bonus = jnp.where(touch_miner, (new_power * 100) // c.max_power, 0).astype(jnp.int32)

        # --- scoring ---
        new_score = (state.score
                     + spiders_killed * c.spider_points
                     + walls_broken * c.wall_points
                     + power_bonus).astype(jnp.int32)

        # --- death / lives / respawn ---
        died = (died_blast | died_spider | died_power) & (~touch_miner)
        new_lives = (state.lives - died.astype(jnp.int32)).astype(jnp.int32)
        respawned = died & (new_lives > 0)

        def sel(a, b):
            return jnp.where(respawned, a, b)

        final_x = sel(jnp.array(c.player_start_x, jnp.int32), new_x).astype(jnp.int32)
        final_y = sel(jnp.array(c.player_start_y, jnp.int32), new_y).astype(jnp.int32)
        final_vy = jnp.where(respawned, 0.0, new_vy).astype(jnp.float32)
        final_facing = sel(jnp.array(1, jnp.int32), new_facing).astype(jnp.int32)
        final_has_moved = has_moved & (~respawned)
        final_power = sel(jnp.array(c.max_power, jnp.int32), new_power).astype(jnp.int32)
        final_laser = jnp.where(respawned, 0, new_laser_timer).astype(jnp.int32)
        final_dyn_active = dyn_active & (~respawned)
        final_explosion = jnp.where(respawned, 0, explosion_timer).astype(jnp.int32)
        final_dyn_fuse = jnp.where(respawned, 0, new_fuse).astype(jnp.int32)

        game_over = state.game_over | (died & (new_lives <= 0))
        new_step = (state.step_counter + 1).astype(jnp.int32)

        new_state = HeroState(
            player_x=final_x,
            player_y=final_y,
            player_vy=final_vy,
            facing=final_facing,
            has_moved=final_has_moved,
            laser_timer=final_laser,
            power=final_power,
            lives=new_lives,
            score=new_score,
            level=state.level,
            dynamite_count=dynamite_count,
            dyn_active=final_dyn_active,
            dyn_x=dyn_x,
            dyn_y=dyn_y,
            dyn_fuse=final_dyn_fuse,
            explosion_timer=final_explosion,
            spider_y=new_spider_y,
            spider_dir=new_spider_dir,
            spider_alive=spider_alive,
            wall_active=wall_active,
            miner_rescued=miner_rescued,
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
        player = ObjectObservation.create(
            x=state.player_x, y=state.player_y,
            width=jnp.array(c.player_width, jnp.int32),
            height=jnp.array(c.player_height, jnp.int32),
            active=jnp.array(True, jnp.bool_),
            orientation=jnp.where(state.facing < 0, 270, 90).astype(jnp.int32),
        )

        laser_active = state.laser_timer > 0
        lx = jnp.where(state.facing < 0, state.player_x - c.laser_length,
                       state.player_x + c.player_width).astype(jnp.int32)
        laser = ObjectObservation.create(
            x=lx, y=(state.player_y + c.player_height // 3).astype(jnp.int32),
            width=jnp.array(c.laser_length, jnp.int32),
            height=jnp.array(c.laser_height, jnp.int32),
            active=laser_active,
            orientation=jnp.where(state.facing < 0, 270, 90).astype(jnp.int32),
        )

        spiders = ObjectObservation.create(
            x=c.SPIDER_X.astype(jnp.int32),
            y=state.spider_y.astype(jnp.int32),
            width=jnp.full((c.num_spiders,), c.spider_width, jnp.int32),
            height=jnp.full((c.num_spiders,), c.spider_height, jnp.int32),
            active=state.spider_alive,
            visual_id=jnp.arange(c.num_spiders, dtype=jnp.int32),
        )

        miner = ObjectObservation.create(
            x=jnp.array(c.miner_x, jnp.int32), y=jnp.array(c.miner_y, jnp.int32),
            width=jnp.array(c.miner_width, jnp.int32),
            height=jnp.array(c.miner_height, jnp.int32),
            active=(~state.miner_rescued),
        )

        dynamite = ObjectObservation.create(
            x=state.dyn_x, y=state.dyn_y,
            width=jnp.array(c.dyn_width, jnp.int32),
            height=jnp.array(c.dyn_height, jnp.int32),
            active=(state.dyn_active | (state.explosion_timer > 0)),
            state=state.explosion_timer,
        )

        walls = ObjectObservation.create(
            x=c.WALLS[:, 0], y=c.WALLS[:, 1], width=c.WALLS[:, 2], height=c.WALLS[:, 3],
            active=state.wall_active,
            visual_id=c.WALL_BREAKABLE.astype(jnp.int32),
        )

        return HeroObservation(
            player=player, laser=laser, spiders=spiders, miner=miner,
            dynamite=dynamite, walls=walls,
            power=state.power, lives=state.lives, score=state.score,
            dynamite_count=state.dynamite_count, level=state.level,
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
            "lives": spaces.Box(low=0, high=c.starting_lives, shape=(), dtype=jnp.int32),
            "score": spaces.Box(low=0, high=jnp.iinfo(jnp.int32).max, shape=(), dtype=jnp.int32),
            "dynamite_count": spaces.Box(low=0, high=c.starting_dynamite, shape=(), dtype=jnp.int32),
            "level": spaces.Box(low=0, high=jnp.iinfo(jnp.int32).max, shape=(), dtype=jnp.int32),
        })

    def image_space(self) -> spaces.Box:
        c = self.consts
        return spaces.Box(low=0, high=255,
                          shape=(c.screen_height, c.screen_width, 3), dtype=jnp.uint8)

    def render(self, state: HeroState) -> jnp.ndarray:
        return self.renderer.render(state)


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------
# 5x3 bitmap font for digits 0-9 (used for the score and the level banner).
_DIGIT_FONT = {
    0: ["111", "101", "101", "101", "111"],
    1: ["010", "110", "010", "010", "111"],
    2: ["111", "001", "111", "100", "111"],
    3: ["111", "001", "111", "001", "111"],
    4: ["101", "101", "111", "001", "001"],
    5: ["111", "100", "111", "001", "111"],
    6: ["111", "100", "111", "101", "111"],
    7: ["111", "001", "010", "010", "010"],
    8: ["111", "101", "111", "101", "111"],
    9: ["111", "101", "111", "001", "111"],
}


class HeroRenderer(JAXGameRenderer):
    """
    Procedural placeholder renderer. Non-breakable walls, the floor, the banner
    and the HUD band are baked into the static background; dynamic objects are
    drawn each frame with solid colored sprites. Replace the 'procedural' asset
    entries with real .npy sprites (see jax_freeway.py) when art is available.
    """

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

        walls_np = np.array(c.WALLS)
        breakable_np = np.array(c.WALL_BREAKABLE)
        bw = int(walls_np[c.breakable_idx, 2])
        bh = int(walls_np[c.breakable_idx, 3])

        asset_config = [
            {'name': 'background', 'type': 'background',
             'data': self._build_background(walls_np, breakable_np)},
            {'name': 'player', 'type': 'procedural',
             'data': self._solid(c.player_height, c.player_width, c.player_color)},
            {'name': 'laser', 'type': 'procedural',
             'data': self._solid(c.laser_height, c.laser_length, c.laser_color)},
            {'name': 'spider', 'type': 'procedural',
             'data': self._solid(c.spider_height, c.spider_width, c.spider_color)},
            {'name': 'miner', 'type': 'procedural',
             'data': self._solid(c.miner_height, c.miner_width, c.miner_color)},
            {'name': 'dynamite', 'type': 'procedural',
             'data': self._solid(c.dyn_height, c.dyn_width, c.dyn_color)},
            {'name': 'explosion', 'type': 'procedural',
             'data': self._solid(2 * c.explosion_radius, 2 * c.explosion_radius, c.explosion_color)},
            {'name': 'breakable', 'type': 'procedural',
             'data': self._solid(bh, bw, c.breakable_color)},
            {'name': 'power_unit', 'type': 'procedural',
             'data': self._solid(c.power_bar_height, 1, c.power_color)},
            {'name': 'dyn_icon', 'type': 'procedural',
             'data': self._solid(6, 2, c.dyn_color)},
            {'name': 'life_icon', 'type': 'procedural',
             'data': self._solid(8, 4, c.player_color)},
            {'name': 'digits', 'type': 'digits', 'data': self._build_digits(c.text_color)},
        ]
        sprite_path = os.path.join(render_utils.get_base_sprite_dir(), "hero")

        (
            self.PALETTE, self.SHAPE_MASKS, self.BACKGROUND,
            self.COLOR_TO_ID, self.FLIP_OFFSETS,
        ) = self.jr.load_and_setup_assets(asset_config, sprite_path)

    # --- procedural asset builders ----------------------------------------
    @staticmethod
    def _solid(h: int, w: int, color: Tuple[int, int, int]) -> jnp.ndarray:
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3] = color[0], color[1], color[2], 255
        return jnp.asarray(rgba)

    def _build_background(self, walls_np, breakable_np) -> jnp.ndarray:
        c = self.consts
        bg = np.zeros((c.screen_height, c.screen_width, 4), dtype=np.uint8)
        bg[:, :, 3] = 255
        bg[:, :, 0:3] = np.array(c.bg_color, dtype=np.uint8)

        # Top banner band and bottom HUD band.
        bg[0:c.play_top, :, 0:3] = np.array(c.banner_color, dtype=np.uint8)
        bg[c.play_bottom:, :, 0:3] = np.array(c.hud_color, dtype=np.uint8)

        # Bake all non-breakable walls.
        for i in range(walls_np.shape[0]):
            if breakable_np[i]:
                continue
            x, y, w, h = (int(v) for v in walls_np[i])
            bg[y:y + h, x:x + w, 0:3] = np.array(c.wall_color, dtype=np.uint8)
        return jnp.asarray(bg)

    def _build_digits(self, color: Tuple[int, int, int]) -> jnp.ndarray:
        digits = np.zeros((10, 5, 3, 4), dtype=np.uint8)
        for d, rows in _DIGIT_FONT.items():
            for r, row in enumerate(rows):
                for col, ch in enumerate(row):
                    if ch == "1":
                        digits[d, r, col, 0:3] = np.array(color, dtype=np.uint8)
                        digits[d, r, col, 3] = 255
        return jnp.asarray(digits)

    # --- render -----------------------------------------------------------
    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: HeroState) -> jnp.ndarray:
        c = self.consts
        raster = self.jr.create_object_raster(self.BACKGROUND)

        def maybe(cond, x, y, mask, ras):
            return jax.lax.cond(
                cond,
                lambda r: self.jr.render_at_clipped(r, x.astype(jnp.int32), y.astype(jnp.int32), mask),
                lambda r: r,
                ras,
            )

        # Breakable wall.
        bx, by = c.WALLS[c.breakable_idx, 0], c.WALLS[c.breakable_idx, 1]
        raster = maybe(state.wall_active[c.breakable_idx], bx, by, self.SHAPE_MASKS["breakable"], raster)

        # Miner.
        raster = maybe(~state.miner_rescued, jnp.array(c.miner_x), jnp.array(c.miner_y),
                       self.SHAPE_MASKS["miner"], raster)

        # Spiders.
        for i in range(c.num_spiders):
            raster = maybe(state.spider_alive[i], c.SPIDER_X[i], state.spider_y[i],
                           self.SHAPE_MASKS["spider"], raster)

        # Dynamite.
        raster = maybe(state.dyn_active, state.dyn_x, state.dyn_y, self.SHAPE_MASKS["dynamite"], raster)

        # Explosion.
        raster = maybe(state.explosion_timer > 0,
                       state.dyn_x - c.explosion_radius, state.dyn_y - c.explosion_radius,
                       self.SHAPE_MASKS["explosion"], raster)

        # Player.
        raster = self.jr.render_at_clipped(raster, state.player_x, state.player_y, self.SHAPE_MASKS["player"])

        # Laser.
        laser_on = state.laser_timer > 0
        lx = jnp.where(state.facing < 0, state.player_x - c.laser_length, state.player_x + c.player_width)
        ly = state.player_y + c.player_height // 3
        raster = maybe(laser_on, lx, ly, self.SHAPE_MASKS["laser"], raster)

        # --- HUD ---
        # Power bar (filled width proportional to remaining power).
        power_units = (state.power * c.power_bar_width) // c.max_power

        def draw_unit(i, ras):
            return jax.lax.cond(
                i < power_units,
                lambda r: self.jr.render_at_clipped(
                    r, jnp.array(c.power_bar_x) + i, jnp.array(c.power_bar_y), self.SHAPE_MASKS["power_unit"]),
                lambda r: r,
                ras,
            )

        raster = jax.lax.fori_loop(0, c.power_bar_width, draw_unit, raster)

        # Dynamite count + lives indicators.
        raster = self.jr.render_indicator(raster, c.power_bar_x, c.power_bar_y + 10,
                                          state.dynamite_count, self.SHAPE_MASKS["dyn_icon"],
                                          spacing=5, max_value=c.starting_dynamite)
        raster = self.jr.render_indicator(raster, 8, c.power_bar_y,
                                          state.lives, self.SHAPE_MASKS["life_icon"],
                                          spacing=6, max_value=c.starting_lives)

        # Score (5 digits, leading zeros).
        score_digits = self.jr.int_to_digits(state.score, max_digits=5)
        raster = self.jr.render_label_selective(
            raster, c.power_bar_x + 40, c.power_bar_y + 10, score_digits,
            self.SHAPE_MASKS["digits"], 0, 5, spacing=4, max_digits_to_render=5)

        # Level banner number (centered) -> shows level+1.
        level_digit = self.jr.int_to_digits(state.level + 1, max_digits=1)
        raster = self.jr.render_label_selective(
            raster, c.screen_width // 2 - 1, 5, level_digit,
            self.SHAPE_MASKS["digits"], 0, 1, spacing=4, max_digits_to_render=1)

        return self.jr.render_from_palette(raster, self.PALETTE)
