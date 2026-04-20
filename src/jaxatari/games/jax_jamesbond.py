"""JAXAtari implementation of James Bond 007 (Atari 2600 / ALE).

The ALE version of James Bond 007 is a horizontal side-scroller in which the
player's vehicle (a car/boat/submarine hybrid loosely inspired by Bond's Lotus
Esprit) appears on the left side of the screen.  The world scrolls from right
to left and enemies come from the right in three distinct vertical zones:

  * Sky zone   (top)    — helicopter/jet enemies
  * Sea zone   (middle) — speedboat enemies on the water surface
  * Sub zone   (bottom) — submarine enemies underwater

The player can move up and down to change zones and fires bullets to the right.
Enemy vehicles also fire back.  The episode ends when the player is hit and all
lives are lost.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple, Tuple

import chex
import jax
import jax.numpy as jnp

import jaxatari.spaces as spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils


# ---------------------------------------------------------------------------
# Enemy type constants (integer tags stored in state arrays)
# ---------------------------------------------------------------------------
ENEMY_JET  = 0   # sky zone
ENEMY_BOAT = 1   # sea zone
ENEMY_SUB  = 2   # underwater zone


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class JamesBondConstants(NamedTuple):
    # Screen
    WIDTH:  int = 160
    HEIGHT: int = 210

    # HUD / status bars
    HUD_TOP_H:    int = 16   # height of top score bar
    HUD_BOTTOM_H: int = 18   # height of bottom lives bar

    # Zone y-centres (vertical centre of each gameplay zone)
    SKY_Y:  int = 48
    SEA_Y:  int = 100
    SUB_Y:  int = 155

    # Zone boundary lines (used for background drawing)
    HORIZON_Y:   int = 76   # top of sea surface (sky / sea boundary)
    SEABED_Y:    int = 130  # top of underwater zone (sea / sub boundary)

    # Player
    PLAYER_START_X: int = 20
    PLAYER_W:        int = 16
    PLAYER_H:        int = 8
    PLAYER_SPEED_Y:  int = 3   # pixels per frame (up / down)
    PLAYER_SPEED_X:  int = 2   # small horizontal drift
    PLAYER_MIN_X:    int = 0
    PLAYER_MAX_X:    int = 36  # player stays on left side

    # Player y bounds (top of sprite must stay inside playfield)
    PLAYER_MIN_Y: int = 16    # just below HUD
    PLAYER_MAX_Y: int = 182   # just above bottom HUD

    # Player bullets
    MAX_BULLETS:   int = 3
    BULLET_W:      int = 4
    BULLET_H:      int = 2
    BULLET_SPEED:  int = 6

    # Enemy bullets
    MAX_E_BULLETS:    int = 4
    E_BULLET_W:       int = 2
    E_BULLET_H:       int = 2
    E_BULLET_SPEED:   int = 2
    ENEMY_FIRE_PROB:  float = 0.015

    # Enemies
    MAX_ENEMIES:           int   = 6
    ENEMY_W:               int   = 16
    ENEMY_H:               int   = 8
    ENEMY_SPEED:           int   = 1
    ENEMY_SPAWN_INTERVAL:  int   = 45   # frames between enemy spawns
    ENEMY_SPAWN_X:         int   = 152  # x at which enemies first appear

    # Scroll
    SCROLL_SPEED: int = 1           # pixels per frame
    BG_STRIPE_W:  int = 8           # width of alternating cloud / wave stripe

    # Lives
    NUM_LIVES: int = 3

    # Rewards
    SCORE_PER_KILL: int = 100



# ---------------------------------------------------------------------------
# State / Observation / Info
# ---------------------------------------------------------------------------

class JamesBondState(NamedTuple):
    player_x:       chex.Array   # scalar int32
    player_y:       chex.Array   # scalar int32

    # Player bullets — shape (MAX_BULLETS,)
    bullet_x:       chex.Array
    bullet_y:       chex.Array
    bullet_active:  chex.Array   # bool

    # Enemies — shape (MAX_ENEMIES,)
    enemy_x:        chex.Array
    enemy_y:        chex.Array
    enemy_type:     chex.Array   # 0=jet, 1=boat, 2=sub
    enemy_active:   chex.Array   # bool

    # Enemy bullets — shape (MAX_E_BULLETS,)
    e_bullet_x:     chex.Array
    e_bullet_y:     chex.Array
    e_bullet_active: chex.Array  # bool

    scroll_offset:  chex.Array   # for background animation
    spawn_timer:    chex.Array   # counts down to next enemy spawn
    score:          chex.Array
    lives:          chex.Array
    step_count:     chex.Array
    key:            chex.PRNGKey


class JamesBondObservation(NamedTuple):
    player_x:     chex.Array
    player_y:     chex.Array
    enemy_x:      chex.Array
    enemy_y:      chex.Array
    enemy_type:   chex.Array
    enemy_active: chex.Array
    bullet_x:     chex.Array
    bullet_y:     chex.Array
    bullet_active: chex.Array
    score:        chex.Array
    lives:        chex.Array


class JamesBondInfo(NamedTuple):
    score:      chex.Array
    lives:      chex.Array
    step_count: chex.Array


# ---------------------------------------------------------------------------
# Helper: axis-aligned rectangle overlap test (vectorised over one axis)
# ---------------------------------------------------------------------------

def _rect_overlap(ax, ay, aw, ah, bx, by, bw, bh):
    """Returns True if rectangle A overlaps rectangle B (broadcast-safe)."""
    return (ax < bx + bw) & (ax + aw > bx) & (ay < by + bh) & (ay + ah > by)


# ---------------------------------------------------------------------------
# Main environment
# ---------------------------------------------------------------------------

class JaxJamesbond(JaxEnvironment[JamesBondState, JamesBondObservation, JamesBondInfo, JamesBondConstants]):
    """
    JAX implementation of the ALE James Bond 007 game.

    Gameplay (matching the Atari 2600 / ALE version):
      - Horizontal side-scroller, world moves right → left.
      - Player vehicle on the left; can move up/down (and slightly left/right).
      - Three vertical zones: sky (jets), sea surface (boats), underwater (subs).
      - Player fires bullets to the right.
      - Enemy vehicles fire back towards the left.
      - Destroying an enemy scores SCORE_PER_KILL points.
      - Being hit by an enemy or enemy bullet costs a life.
      - Episode ends when all lives are gone.
    """

    def __init__(self, consts: JamesBondConstants | None = None):
        consts = consts or JamesBondConstants()
        super().__init__(consts)
        self.renderer = JamesBondRenderer(self.consts)
        # Full 18-action set, matching ALE JamesBond-v5
        self.action_set = [
            Action.NOOP,
            Action.FIRE,
            Action.UP,
            Action.RIGHT,
            Action.LEFT,
            Action.DOWN,
            Action.UPRIGHT,
            Action.UPLEFT,
            Action.DOWNRIGHT,
            Action.DOWNLEFT,
            Action.UPFIRE,
            Action.RIGHTFIRE,
            Action.LEFTFIRE,
            Action.DOWNFIRE,
            Action.UPRIGHTFIRE,
            Action.UPLEFTFIRE,
            Action.DOWNRIGHTFIRE,
            Action.DOWNLEFTFIRE,
        ]

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self, key: chex.PRNGKey = jax.random.PRNGKey(42)) -> Tuple[JamesBondObservation, JamesBondState]:
        n_e  = self.consts.MAX_ENEMIES
        n_b  = self.consts.MAX_BULLETS
        n_eb = self.consts.MAX_E_BULLETS

        state = JamesBondState(
            player_x=jnp.array(self.consts.PLAYER_START_X, dtype=jnp.int32),
            player_y=jnp.array(self.consts.SEA_Y - self.consts.PLAYER_H // 2, dtype=jnp.int32),

            bullet_x=jnp.zeros((n_b,),  dtype=jnp.int32),
            bullet_y=jnp.zeros((n_b,),  dtype=jnp.int32),
            bullet_active=jnp.zeros((n_b,), dtype=bool),

            enemy_x=jnp.zeros((n_e,), dtype=jnp.int32),
            enemy_y=jnp.zeros((n_e,), dtype=jnp.int32),
            enemy_type=jnp.zeros((n_e,), dtype=jnp.int32),
            enemy_active=jnp.zeros((n_e,), dtype=bool),

            e_bullet_x=jnp.zeros((n_eb,), dtype=jnp.int32),
            e_bullet_y=jnp.zeros((n_eb,), dtype=jnp.int32),
            e_bullet_active=jnp.zeros((n_eb,), dtype=bool),

            scroll_offset=jnp.array(0, dtype=jnp.int32),
            spawn_timer=jnp.array(self.consts.ENEMY_SPAWN_INTERVAL, dtype=jnp.int32),
            score=jnp.array(0, dtype=jnp.int32),
            lives=jnp.array(self.consts.NUM_LIVES, dtype=jnp.int32),
            step_count=jnp.array(0, dtype=jnp.int32),
            key=key,
        )
        return self._get_observation(state), state

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: JamesBondState,
        action: chex.Array,
    ) -> Tuple[JamesBondObservation, JamesBondState, float, bool, JamesBondInfo]:

        # 1. Move player
        state = self._step_player(state, action)

        # 2. Fire player bullet
        state = self._step_player_fire(state, action)

        # 3. Move existing player bullets
        state = self._step_bullets(state)

        # 4. Spawn + move enemies
        state = self._step_enemies(state)

        # 5. Enemy fire
        state = self._step_enemy_fire(state)

        # 6. Move enemy bullets
        state = self._step_enemy_bullets(state)

        # 7. Collision: player bullets vs enemies
        state, kills = self._resolve_bullet_enemy_collisions(state)

        # 8. Collision: player vs enemies / enemy bullets
        state, hit = self._resolve_player_hit(state)

        # 9. Score, lives, done
        score = (state.score + kills * self.consts.SCORE_PER_KILL).astype(jnp.int32)
        lives = (state.lives - hit.astype(jnp.int32)).astype(jnp.int32)
        lives = jnp.maximum(lives, 0)

        state = state._replace(
            score=score,
            lives=lives,
            step_count=(state.step_count + 1).astype(jnp.int32),
            scroll_offset=(
                (state.scroll_offset + self.consts.SCROLL_SPEED) % self.consts.WIDTH
            ).astype(jnp.int32),
        )

        reward = (kills * self.consts.SCORE_PER_KILL).astype(jnp.float32)
        done   = state.lives <= 0
        obs    = self._get_observation(state)
        info   = self._get_info(state)
        return obs, state, reward, done, info

    # ------------------------------------------------------------------
    # Player movement
    # ------------------------------------------------------------------

    def _step_player(self, state: JamesBondState, action: chex.Array) -> JamesBondState:
        # Decode directional bits from action
        move_up    = (
            (action == Action.UP)          | (action == Action.UPRIGHT)   |
            (action == Action.UPLEFT)      | (action == Action.UPFIRE)    |
            (action == Action.UPRIGHTFIRE) | (action == Action.UPLEFTFIRE)
        )
        move_down  = (
            (action == Action.DOWN)          | (action == Action.DOWNRIGHT)   |
            (action == Action.DOWNLEFT)      | (action == Action.DOWNFIRE)    |
            (action == Action.DOWNRIGHTFIRE) | (action == Action.DOWNLEFTFIRE)
        )
        move_right = (
            (action == Action.RIGHT)         | (action == Action.UPRIGHT)     |
            (action == Action.DOWNRIGHT)     | (action == Action.RIGHTFIRE)   |
            (action == Action.UPRIGHTFIRE)   | (action == Action.DOWNRIGHTFIRE)
        )
        move_left  = (
            (action == Action.LEFT)          | (action == Action.UPLEFT)      |
            (action == Action.DOWNLEFT)      | (action == Action.LEFTFIRE)    |
            (action == Action.UPLEFTFIRE)    | (action == Action.DOWNLEFTFIRE)
        )

        dy = jnp.where(move_up,   -self.consts.PLAYER_SPEED_Y, 0)
        dy = jnp.where(move_down,  self.consts.PLAYER_SPEED_Y, dy)
        dx = jnp.where(move_right, self.consts.PLAYER_SPEED_X, 0)
        dx = jnp.where(move_left, -self.consts.PLAYER_SPEED_X, dx)

        new_y = jnp.clip(
            state.player_y + dy.astype(jnp.int32),
            self.consts.PLAYER_MIN_Y,
            self.consts.PLAYER_MAX_Y,
        ).astype(jnp.int32)
        new_x = jnp.clip(
            state.player_x + dx.astype(jnp.int32),
            self.consts.PLAYER_MIN_X,
            self.consts.PLAYER_MAX_X,
        ).astype(jnp.int32)

        return state._replace(player_x=new_x, player_y=new_y)

    # ------------------------------------------------------------------
    # Player firing
    # ------------------------------------------------------------------

    def _step_player_fire(self, state: JamesBondState, action: chex.Array) -> JamesBondState:
        fired = (
            (action == Action.FIRE)         | (action == Action.UPFIRE)        |
            (action == Action.DOWNFIRE)     | (action == Action.RIGHTFIRE)     |
            (action == Action.LEFTFIRE)     | (action == Action.UPRIGHTFIRE)   |
            (action == Action.UPLEFTFIRE)   | (action == Action.DOWNRIGHTFIRE) |
            (action == Action.DOWNLEFTFIRE)
        )

        # Find a free bullet slot
        free_mask = ~state.bullet_active
        any_free  = jnp.any(free_mask)
        slot      = jnp.argmax(free_mask.astype(jnp.int32))

        spawn_x = (state.player_x + self.consts.PLAYER_W).astype(jnp.int32)
        spawn_y = (state.player_y + self.consts.PLAYER_H // 2 - self.consts.BULLET_H // 2).astype(jnp.int32)

        should_spawn = fired & any_free

        new_bx = jnp.where(
            jnp.arange(self.consts.MAX_BULLETS) == slot,
            jnp.where(should_spawn, spawn_x, state.bullet_x),
            state.bullet_x,
        )
        new_by = jnp.where(
            jnp.arange(self.consts.MAX_BULLETS) == slot,
            jnp.where(should_spawn, spawn_y, state.bullet_y),
            state.bullet_y,
        )
        new_ba = jnp.where(
            jnp.arange(self.consts.MAX_BULLETS) == slot,
            jnp.where(should_spawn, True, state.bullet_active),
            state.bullet_active,
        )

        return state._replace(bullet_x=new_bx, bullet_y=new_by, bullet_active=new_ba)

    # ------------------------------------------------------------------
    # Move player bullets (to the right)
    # ------------------------------------------------------------------

    def _step_bullets(self, state: JamesBondState) -> JamesBondState:
        new_bx = (state.bullet_x + state.bullet_active.astype(jnp.int32) * self.consts.BULLET_SPEED).astype(jnp.int32)
        # Deactivate bullets that left the screen
        still_on = new_bx < self.consts.WIDTH
        new_ba   = state.bullet_active & still_on
        new_bx   = jnp.where(new_ba, new_bx, 0)
        new_by   = jnp.where(new_ba, state.bullet_y, 0)
        return state._replace(bullet_x=new_bx, bullet_y=new_by, bullet_active=new_ba)

    # ------------------------------------------------------------------
    # Spawn and move enemies
    # ------------------------------------------------------------------

    def _step_enemies(self, state: JamesBondState) -> JamesBondState:
        # Move active enemies to the left
        new_ex = (state.enemy_x - state.enemy_active.astype(jnp.int32) * self.consts.ENEMY_SPEED).astype(jnp.int32)

        # Deactivate enemies that scrolled off the left edge
        still_on     = new_ex + self.consts.ENEMY_W > 0
        enemy_active = state.enemy_active & still_on
        new_ex       = jnp.where(enemy_active, new_ex, 0)
        new_ey       = jnp.where(enemy_active, state.enemy_y, 0)
        new_et       = jnp.where(enemy_active, state.enemy_type, 0)

        # Countdown spawn timer
        new_timer = (state.spawn_timer - 1).astype(jnp.int32)

        # Spawn a new enemy when timer reaches zero
        key, type_key, y_key = jax.random.split(state.key, 3)

        # Randomly choose enemy type (0=jet, 1=boat, 2=sub)
        etype    = jax.random.randint(type_key, (), 0, 3, dtype=jnp.int32)
        ey_table = jnp.array(
            [
                self.consts.SKY_Y - self.consts.ENEMY_H // 2,
                self.consts.SEA_Y - self.consts.ENEMY_H // 2,
                self.consts.SUB_Y - self.consts.ENEMY_H // 2,
            ],
            dtype=jnp.int32,
        )
        ey = ey_table[etype]

        # Find free enemy slot
        free_mask = ~enemy_active
        any_free  = jnp.any(free_mask)
        slot      = jnp.argmax(free_mask.astype(jnp.int32))

        should_spawn = (new_timer <= 0) & any_free
        new_timer    = jnp.where(new_timer <= 0, self.consts.ENEMY_SPAWN_INTERVAL, new_timer).astype(jnp.int32)

        indices = jnp.arange(self.consts.MAX_ENEMIES)
        is_slot = indices == slot

        new_ex = jnp.where(is_slot & should_spawn, self.consts.ENEMY_SPAWN_X, new_ex)
        new_ey = jnp.where(is_slot & should_spawn, ey,                         new_ey)
        new_et = jnp.where(is_slot & should_spawn, etype,                      new_et)
        enemy_active = jnp.where(is_slot & should_spawn, True,                 enemy_active)

        return state._replace(
            enemy_x=new_ex.astype(jnp.int32),
            enemy_y=new_ey.astype(jnp.int32),
            enemy_type=new_et.astype(jnp.int32),
            enemy_active=enemy_active,
            spawn_timer=new_timer,
            key=key,
        )

    # ------------------------------------------------------------------
    # Enemies fire at the player
    # ------------------------------------------------------------------

    def _step_enemy_fire(self, state: JamesBondState) -> JamesBondState:
        key, roll_key = jax.random.split(state.key)

        rolls = jax.random.uniform(roll_key, (self.consts.MAX_ENEMIES,))
        wants_fire = state.enemy_active & (rolls < self.consts.ENEMY_FIRE_PROB)

        # For each enemy that fires, try to place a bullet in a free slot
        free_mask = ~state.e_bullet_active

        def fire_one(carry, i):
            bx, by, ba = carry
            free_slots = ~ba
            any_free   = jnp.any(free_slots)
            slot       = jnp.argmax(free_slots.astype(jnp.int32))
            should     = wants_fire[i] & any_free
            spawn_bx   = state.enemy_x[i]
            spawn_by   = (state.enemy_y[i] + self.consts.ENEMY_H // 2 - self.consts.E_BULLET_H // 2).astype(jnp.int32)
            indices    = jnp.arange(self.consts.MAX_E_BULLETS)
            bx = jnp.where((indices == slot) & should, spawn_bx, bx)
            by = jnp.where((indices == slot) & should, spawn_by, by)
            ba = jnp.where((indices == slot) & should, True,      ba)
            return (bx, by, ba), None

        (new_ebx, new_eby, new_eba), _ = jax.lax.scan(
            fire_one,
            (state.e_bullet_x, state.e_bullet_y, state.e_bullet_active),
            jnp.arange(self.consts.MAX_ENEMIES),
        )

        return state._replace(
            e_bullet_x=new_ebx.astype(jnp.int32),
            e_bullet_y=new_eby.astype(jnp.int32),
            e_bullet_active=new_eba,
            key=key,
        )

    # ------------------------------------------------------------------
    # Move enemy bullets (to the left)
    # ------------------------------------------------------------------

    def _step_enemy_bullets(self, state: JamesBondState) -> JamesBondState:
        new_ebx = (state.e_bullet_x - state.e_bullet_active.astype(jnp.int32) * self.consts.E_BULLET_SPEED).astype(jnp.int32)
        still_on = new_ebx + self.consts.E_BULLET_W > 0
        new_eba  = state.e_bullet_active & still_on
        new_ebx  = jnp.where(new_eba, new_ebx, 0)
        new_eby  = jnp.where(new_eba, state.e_bullet_y, 0)
        return state._replace(e_bullet_x=new_ebx, e_bullet_y=new_eby, e_bullet_active=new_eba)

    # ------------------------------------------------------------------
    # Player bullet vs enemy collisions
    # ------------------------------------------------------------------

    def _resolve_bullet_enemy_collisions(
        self, state: JamesBondState
    ) -> Tuple[JamesBondState, chex.Array]:
        """Returns updated state and the number of kills this frame."""

        # Build hit matrix: (MAX_BULLETS, MAX_ENEMIES)
        bx = state.bullet_x[:, None]   # (B, 1)
        by = state.bullet_y[:, None]
        ba = state.bullet_active[:, None]

        ex = state.enemy_x[None, :]    # (1, E)
        ey = state.enemy_y[None, :]
        ea = state.enemy_active[None, :]

        hit = (
            ba & ea &
            _rect_overlap(bx, by, self.consts.BULLET_W, self.consts.BULLET_H,
                           ex, ey, self.consts.ENEMY_W,  self.consts.ENEMY_H)
        )  # (B, E)

        # A bullet is consumed if it hit any enemy
        bullet_hit = jnp.any(hit, axis=1)  # (B,)
        # An enemy is killed if hit by any bullet
        enemy_hit  = jnp.any(hit, axis=0)  # (E,)

        kills = enemy_hit.astype(jnp.int32).sum()

        new_ba = state.bullet_active & ~bullet_hit
        new_bx = jnp.where(new_ba, state.bullet_x, 0)
        new_by = jnp.where(new_ba, state.bullet_y, 0)

        new_ea = state.enemy_active & ~enemy_hit
        new_ex = jnp.where(new_ea, state.enemy_x, 0)
        new_ey = jnp.where(new_ea, state.enemy_y, 0)
        new_et = jnp.where(new_ea, state.enemy_type, 0)

        state = state._replace(
            bullet_active=new_ba, bullet_x=new_bx, bullet_y=new_by,
            enemy_active=new_ea,  enemy_x=new_ex,  enemy_y=new_ey, enemy_type=new_et,
        )
        return state, kills

    # ------------------------------------------------------------------
    # Player hit detection
    # ------------------------------------------------------------------

    def _resolve_player_hit(
        self, state: JamesBondState
    ) -> Tuple[JamesBondState, chex.Array]:
        """Returns updated state and bool (was player hit this frame?)."""
        px, py = state.player_x, state.player_y
        pw, ph = self.consts.PLAYER_W, self.consts.PLAYER_H

        # Hit by enemy body
        enemy_hit = _rect_overlap(
            px, py, pw, ph,
            state.enemy_x, state.enemy_y, self.consts.ENEMY_W, self.consts.ENEMY_H,
        ) & state.enemy_active          # (E,)

        # Hit by enemy bullet
        ebullet_hit = _rect_overlap(
            px, py, pw, ph,
            state.e_bullet_x, state.e_bullet_y, self.consts.E_BULLET_W, self.consts.E_BULLET_H,
        ) & state.e_bullet_active       # (EB,)

        hit = jnp.any(enemy_hit) | jnp.any(ebullet_hit)

        # Clear the bullet that caused the hit
        new_eba = jnp.where(ebullet_hit, False, state.e_bullet_active)
        state = state._replace(e_bullet_active=new_eba)

        return state, hit

    # ------------------------------------------------------------------
    # Observations / info
    # ------------------------------------------------------------------

    def _get_observation(self, state: JamesBondState) -> JamesBondObservation:
        return JamesBondObservation(
            player_x=state.player_x,
            player_y=state.player_y,
            enemy_x=state.enemy_x,
            enemy_y=state.enemy_y,
            enemy_type=state.enemy_type,
            enemy_active=state.enemy_active,
            bullet_x=state.bullet_x,
            bullet_y=state.bullet_y,
            bullet_active=state.bullet_active,
            score=state.score,
            lives=state.lives,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: JamesBondState) -> JamesBondInfo:
        return JamesBondInfo(
            score=state.score,
            lives=state.lives,
            step_count=state.step_count,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, prev: JamesBondState, state: JamesBondState) -> float:
        return ((state.score - prev.score)).astype(jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: JamesBondState) -> bool:
        return state.lives <= 0

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: JamesBondObservation) -> jnp.ndarray:
        return jnp.concatenate([
            obs.player_x.flatten().astype(jnp.float32),
            obs.player_y.flatten().astype(jnp.float32),
            obs.enemy_x.flatten().astype(jnp.float32),
            obs.enemy_y.flatten().astype(jnp.float32),
            obs.enemy_type.flatten().astype(jnp.float32),
            obs.enemy_active.flatten().astype(jnp.float32),
            obs.bullet_x.flatten().astype(jnp.float32),
            obs.bullet_y.flatten().astype(jnp.float32),
            obs.bullet_active.flatten().astype(jnp.float32),
            obs.score.flatten().astype(jnp.float32),
            obs.lives.flatten().astype(jnp.float32),
        ])

    def render(self, state: JamesBondState) -> jnp.ndarray:
        return self.renderer.render(state)

    # ------------------------------------------------------------------
    # Spaces
    # ------------------------------------------------------------------

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(18)

    def observation_space(self) -> spaces.Space:
        n_e = self.consts.MAX_ENEMIES
        n_b = self.consts.MAX_BULLETS
        W, H = self.consts.WIDTH, self.consts.HEIGHT
        return spaces.Dict({
            "player_x":     spaces.Box(0, W, shape=(), dtype=jnp.int32),
            "player_y":     spaces.Box(0, H, shape=(), dtype=jnp.int32),
            "enemy_x":      spaces.Box(0, W, shape=(n_e,), dtype=jnp.int32),
            "enemy_y":      spaces.Box(0, H, shape=(n_e,), dtype=jnp.int32),
            "enemy_type":   spaces.Box(0, 3, shape=(n_e,), dtype=jnp.int32),
            "enemy_active": spaces.Box(0, 1, shape=(n_e,), dtype=bool),
            "bullet_x":     spaces.Box(0, W, shape=(n_b,), dtype=jnp.int32),
            "bullet_y":     spaces.Box(0, H, shape=(n_b,), dtype=jnp.int32),
            "bullet_active": spaces.Box(0, 1, shape=(n_b,), dtype=bool),
            "score":        spaces.Box(0, 999999, shape=(), dtype=jnp.int32),
            "lives":        spaces.Box(0, self.consts.NUM_LIVES, shape=(), dtype=jnp.int32),
        })

    def image_space(self) -> spaces.Box:
        return spaces.Box(0, 255, shape=(self.consts.HEIGHT, self.consts.WIDTH, 3), dtype=jnp.uint8)


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class JamesBondRenderer(JAXGameRenderer):
    """
    Simple rectangle-based renderer approximating the ALE visual layout.

    Background zones:
      - Black HUD strip at top and bottom
      - Sky (light blue) from HUD to horizon
      - Sea surface (dark blue) from horizon to seabed
      - Underwater (deep blue/green) below seabed

    Sprites are coloured rectangles:
      - Player      → yellow
      - Jet         → red
      - Boat        → orange
      - Sub         → green
      - Player bullet → white
      - Enemy bullet  → light orange
    """

    # Palette index constants (kept here so NamedTuple fields stay valid identifiers)
    _ID_HUD      = 0
    _ID_SKY      = 1
    _ID_SEA_SURF = 2
    _ID_SEA_DEEP = 3
    _ID_HORIZON  = 4
    _ID_SEABED   = 5
    _ID_PLAYER   = 6
    _ID_JET      = 7
    _ID_BOAT     = 8
    _ID_SUB      = 9
    _ID_BULLET   = 10
    _ID_E_BULLET = 11

    # RGB colours (approximate ALE palette)
    _COLOR_HUD      = (0,   0,   0)
    _COLOR_SKY      = (80,  128, 192)
    _COLOR_SEA_SURF = (0,   72,  148)
    _COLOR_SEA_DEEP = (0,   48,  108)
    _COLOR_HORIZON  = (200, 220, 180)
    _COLOR_SEABED   = (120, 100, 60)
    _COLOR_PLAYER   = (210, 210, 64)
    _COLOR_JET      = (200, 72,  72)
    _COLOR_BOAT     = (210, 140, 60)
    _COLOR_SUB      = (84,  184, 84)
    _COLOR_BULLET   = (255, 255, 255)
    _COLOR_E_BULLET = (255, 180, 100)

    def __init__(self, consts: JamesBondConstants | None = None):
        self.consts = consts or JamesBondConstants()
        super().__init__(self.consts)

        self.config = render_utils.RendererConfig(
            game_dimensions=(self.consts.HEIGHT, self.consts.WIDTH),
            channels=3,
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)

        # Palette: index → RGB (order must match _ID_* constants above)
        self.PALETTE = jnp.array(
            [
                self._COLOR_HUD,        # 0
                self._COLOR_SKY,        # 1
                self._COLOR_SEA_SURF,   # 2
                self._COLOR_SEA_DEEP,   # 3
                self._COLOR_HORIZON,    # 4
                self._COLOR_SEABED,     # 5
                self._COLOR_PLAYER,     # 6
                self._COLOR_JET,        # 7
                self._COLOR_BOAT,       # 8
                self._COLOR_SUB,        # 9
                self._COLOR_BULLET,     # 10
                self._COLOR_E_BULLET,   # 11
            ],
            dtype=jnp.uint8,
        )

        # Pre-build the static background raster (does not change with game state)
        self._STATIC_BG = self._build_background()

    def _build_background(self) -> jnp.ndarray:
        """Builds the static zone background as a palette-index raster (H, W)."""
        c = self.consts
        H, W = c.HEIGHT, c.WIDTH

        # Start with sky colour everywhere
        bg = jnp.full((H, W), self._ID_SKY, dtype=jnp.uint8)

        rows = jnp.arange(H)[:, None]   # (H, 1) broadcasts over width

        # Top HUD bar
        bg = jnp.where(rows < c.HUD_TOP_H, self._ID_HUD, bg)

        # Horizon line (2 px thick)
        bg = jnp.where((rows >= c.HORIZON_Y) & (rows < c.HORIZON_Y + 2), self._ID_HORIZON, bg)

        # Sea surface zone (below horizon)
        bg = jnp.where(rows >= c.HORIZON_Y + 2, self._ID_SEA_SURF, bg)

        # Seabed transition line (2 px)
        bg = jnp.where((rows >= c.SEABED_Y) & (rows < c.SEABED_Y + 2), self._ID_SEABED, bg)

        # Deep water / underwater zone
        bg = jnp.where(rows >= c.SEABED_Y + 2, self._ID_SEA_DEEP, bg)

        # Bottom HUD bar
        bg = jnp.where(rows >= H - c.HUD_BOTTOM_H, self._ID_HUD, bg)

        return bg.astype(jnp.uint8)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: JamesBondState) -> jnp.ndarray:
        c = self.consts

        # --- Background (with animated scroll stripes) ---
        raster = self._draw_scroll_stripes(self._STATIC_BG, state.scroll_offset)

        # --- Player ---
        raster = self.jr.draw_rects(
            raster,
            jnp.stack([state.player_x, state.player_y])[None, :],
            jnp.array([[c.PLAYER_W, c.PLAYER_H]], dtype=jnp.int32),
            self._ID_PLAYER,
        )

        # --- Enemies (grouped by type to keep color_id a Python int) ---
        for etype, color_id in (
            (ENEMY_JET,  self._ID_JET),
            (ENEMY_BOAT, self._ID_BOAT),
            (ENEMY_SUB,  self._ID_SUB),
        ):
            # Collect positions for enemies of this type; hide others off-screen
            mask = state.enemy_active & (state.enemy_type == etype)
            ex = jnp.where(mask, state.enemy_x, jnp.full_like(state.enemy_x, -c.ENEMY_W))
            ey = state.enemy_y
            positions = jnp.stack([ex, ey], axis=1)          # (MAX_ENEMIES, 2)
            sizes     = jnp.tile(jnp.array([[c.ENEMY_W, c.ENEMY_H]], dtype=jnp.int32),
                                 (c.MAX_ENEMIES, 1))
            raster = self.jr.draw_rects(raster, positions, sizes, color_id)

        # --- Player bullets ---
        bx = jnp.where(state.bullet_active, state.bullet_x,
                       jnp.full_like(state.bullet_x, -c.BULLET_W))
        b_positions = jnp.stack([bx, state.bullet_y], axis=1)
        b_sizes     = jnp.tile(jnp.array([[c.BULLET_W, c.BULLET_H]], dtype=jnp.int32),
                               (c.MAX_BULLETS, 1))
        raster = self.jr.draw_rects(raster, b_positions, b_sizes, self._ID_BULLET)

        # --- Enemy bullets ---
        ebx = jnp.where(state.e_bullet_active, state.e_bullet_x,
                        jnp.full_like(state.e_bullet_x, -c.E_BULLET_W))
        eb_positions = jnp.stack([ebx, state.e_bullet_y], axis=1)
        eb_sizes     = jnp.tile(jnp.array([[c.E_BULLET_W, c.E_BULLET_H]], dtype=jnp.int32),
                                (c.MAX_E_BULLETS, 1))
        raster = self.jr.draw_rects(raster, eb_positions, eb_sizes, self._ID_E_BULLET)

        return self.jr.render_from_palette(raster, self.PALETTE)

    @partial(jax.jit, static_argnums=(0,))
    def _draw_scroll_stripes(
        self, bg: jnp.ndarray, scroll_offset: chex.Array
    ) -> jnp.ndarray:
        """
        Adds animated scrolling stripes:
          - sparse horizontal dashes in the sky zone (simulate clouds/distance marks)
          - sparse ripple lines in the sea zone (simulate waves)
        """
        c = self.consts
        H, W = c.HEIGHT, c.WIDTH
        cols = jnp.arange(W)[None, :]   # (1, W)
        rows = jnp.arange(H)[:, None]   # (H, 1)

        shifted_col = (cols + scroll_offset) % (c.BG_STRIPE_W * 2)

        # Sky dashes
        in_sky       = (rows >= c.HUD_TOP_H) & (rows < c.HORIZON_Y)
        cloud_stripe = in_sky & (shifted_col < c.BG_STRIPE_W) & (rows % 10 < 2)
        raster = jnp.where(cloud_stripe, jnp.uint8(self._ID_HORIZON), bg)

        # Sea ripples
        in_sea       = (rows >= c.HORIZON_Y + 2) & (rows < c.SEABED_Y)
        wave_stripe  = in_sea & (shifted_col < 2) & (rows % 8 == 0)
        raster = jnp.where(wave_stripe, jnp.uint8(self._ID_HORIZON), raster)

        return raster.astype(jnp.uint8)
