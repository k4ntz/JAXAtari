"""
Boxing mod plugins.

Simple mods (constant overrides / no-ops):
- StaticEnemyMod: CPU opponent stands still (no movement AI)
- FastStunMod: Stun duration halved (recover faster from hits)
- SlowStunMod: Stun duration doubled (longer knockback animation)
- InfiniteTimeMod: Clock never counts down – the round goes on forever
- OneHitKOMod: A single landed punch triggers a KO
- DoubleSpeedMod: Both boxers move at double speed

Hard mods (method overrides with custom logic):
- RandomWalkEnemyMod: CPU wanders randomly instead of tracking the player
- MirrorEnemyMod: CPU mirrors the player's vertical position and slowly
  closes the horizontal gap
- BodyHitsScoreMod: All hits score (removes the fine vertical alignment
  check so body hits count, not just head-level)
"""

import jax
import jax.numpy as jnp
from dataclasses import replace
from functools import partial

from jaxatari.games.jax_boxing import BoxingState
from jaxatari.modification import (
    JaxAtariInternalModPlugin,
    JaxAtariPostStepModPlugin,
)


# ──────────────────────────────────────────────────────────────────────
# 1. CPU movement mods
# ──────────────────────────────────────────────────────────────────────

class StaticEnemyMod(JaxAtariInternalModPlugin):
    """CPU boxer does not move at all – only punches."""

    @partial(jax.jit, static_argnums=(0,))
    def _cpu_movement_step(self, state: BoxingState) -> BoxingState:
        # No-op: return state unchanged (CPU stays where it is)
        return state


class RandomWalkEnemyMod(JaxAtariInternalModPlugin):
    """CPU walks randomly instead of tracking the player."""

    @partial(jax.jit, static_argnums=(0,))
    def _cpu_movement_step(self, state: BoxingState) -> BoxingState:
        is_stunned = jnp.logical_and(
            state.hit_boxer_stun_timer > 0,
            state.hit_boxer_index == 1,
        )

        key, k1, k2 = jax.random.split(state.key, 3)
        dx = jax.random.randint(k1, (), -1, 2)  # -1, 0, or 1
        dy = jax.random.randint(k2, (), -1, 2)

        consts = self._env.consts
        new_x = jnp.clip(
            state.right_boxer_x + dx, consts.XMIN_BOXER, consts.XMAX_BOXER
        ).astype(jnp.int32)
        new_y = jnp.clip(
            state.right_boxer_y + dy, consts.YMIN, consts.YMAX
        ).astype(jnp.int32)

        # Block movement while stunned
        new_x = jnp.where(is_stunned, state.right_boxer_x, new_x)
        new_y = jnp.where(is_stunned, state.right_boxer_y, new_y)

        return replace(state, right_boxer_x=new_x, right_boxer_y=new_y, key=key)


class MirrorEnemyMod(JaxAtariInternalModPlugin):
    """CPU mirrors the player's vertical position and slowly closes the
    horizontal gap, creating a "shadow boxer" that tracks vertically
    but advances at a fixed pace."""

    @partial(jax.jit, static_argnums=(0,))
    def _cpu_movement_step(self, state: BoxingState) -> BoxingState:
        is_stunned = jnp.logical_and(
            state.hit_boxer_stun_timer > 0,
            state.hit_boxer_index == 1,
        )
        consts = self._env.consts

        # Vertical: snap to player's Y position
        mirror_y = jnp.clip(
            state.left_boxer_y, consts.YMIN, consts.YMAX
        ).astype(jnp.int32)

        # Horizontal: slowly creep toward the player (1 px every 4 frames)
        should_advance = (state.step_counter % 4 == 0)
        dx = jnp.sign(state.left_boxer_x - state.right_boxer_x).astype(jnp.int32)
        new_x = jnp.where(
            should_advance,
            state.right_boxer_x + dx,
            state.right_boxer_x,
        )
        new_x = jnp.clip(new_x, consts.XMIN_BOXER, consts.XMAX_BOXER).astype(jnp.int32)

        # Block movement while stunned
        new_x = jnp.where(is_stunned, state.right_boxer_x, new_x)
        mirror_y = jnp.where(is_stunned, state.right_boxer_y, mirror_y)

        return replace(state, right_boxer_x=new_x, right_boxer_y=mirror_y)


# ──────────────────────────────────────────────────────────────────────
# 2. Stun-timing mods (constant overrides – no method patches needed)
# ──────────────────────────────────────────────────────────────────────

class FastStunMod(JaxAtariInternalModPlugin):
    """Halve the stun duration so boxers recover faster."""
    constants_overrides = {"STUN_DURATION": 7}


class SlowStunMod(JaxAtariInternalModPlugin):
    """Double the stun duration for a longer knockback animation."""
    constants_overrides = {"STUN_DURATION": 30}


# ──────────────────────────────────────────────────────────────────────
# 3. Timer / round mods
# ──────────────────────────────────────────────────────────────────────

class InfiniteTimeMod(JaxAtariInternalModPlugin):
    """Clock never counts down – the round goes on forever."""

    @partial(jax.jit, static_argnums=(0,))
    def _timer_step(self, state: BoxingState) -> BoxingState:
        return state  # no-op


# ──────────────────────────────────────────────────────────────────────
# 4. KO threshold mods
# ──────────────────────────────────────────────────────────────────────

class OneHitKOMod(JaxAtariInternalModPlugin):
    """A single landed punch triggers a KO."""
    constants_overrides = {"MAX_SCORE": 1}


# ──────────────────────────────────────────────────────────────────────
# 5. Movement speed mods
# ──────────────────────────────────────────────────────────────────────

class DoubleSpeedMod(JaxAtariInternalModPlugin):
    """Both boxers move at double speed."""
    constants_overrides = {"MOVE_SPEED": 2}


# ──────────────────────────────────────────────────────────────────────
# 6. Scoring mods
# ──────────────────────────────────────────────────────────────────────

class BodyHitsScoreMod(JaxAtariInternalModPlugin):
    """
    Remove the fine vertical-alignment requirement so *any* body-level
    hit scores, not just head-level punches.

    Overrides both _hit_detection_step (player → CPU) and
    _cpu_hit_detection_step (CPU → player).
    """

    @partial(jax.jit, static_argnums=(0,))
    def _hit_detection_step(self, state: BoxingState) -> BoxingState:
        """Player hit detection – body hits score."""
        horiz_dist = jnp.abs(state.left_boxer_x - state.right_boxer_x)
        vert_dist = jnp.abs(state.left_boxer_y - state.right_boxer_y)

        in_horiz_range = horiz_dist <= self._env.consts.HIT_DISTANCE_HORIZONTAL
        in_vert_range = vert_dist < self._env.consts.HIT_DISTANCE_VERTICAL

        just_reached_max = state.extended_arm_maximum[0] == 1
        punch_active = state.left_boxer_punch_active > 0
        punch_not_landed_yet = state.left_boxer_punch_landed == 0

        # No fine_vert_aligned check – all body hits count
        hit_landed = jnp.logical_and(
            jnp.logical_and(punch_active, just_reached_max),
            jnp.logical_and(
                jnp.logical_and(in_horiz_range, in_vert_range),
                punch_not_landed_yet,
            ),
        )

        opponent_not_stunned = jnp.logical_or(
            state.hit_boxer_stun_timer == 0,
            state.hit_boxer_index != 1,
        )
        valid_hit = jnp.logical_and(hit_landed, opponent_not_stunned)

        new_punch_landed = jnp.where(valid_hit, 1, state.left_boxer_punch_landed).astype(jnp.int32)
        new_score = jnp.where(valid_hit, state.left_boxer_score + 1, state.left_boxer_score).astype(jnp.int32)
        new_stun_timer = jnp.where(valid_hit, self._env.consts.STUN_DURATION, state.hit_boxer_stun_timer).astype(jnp.int32)
        new_hit_index = jnp.where(valid_hit, 1, state.hit_boxer_index).astype(jnp.int32)
        new_punching_arm = jnp.where(valid_hit, state.left_boxer_last_arm, state.punching_arm_index).astype(jnp.int32)
        new_dancing = jnp.where(valid_hit, 57, state.cpu_dancing_value).astype(jnp.int32)

        return replace(
            state,
            left_boxer_score=new_score,
            hit_boxer_stun_timer=new_stun_timer,
            hit_boxer_index=new_hit_index,
            left_boxer_punch_landed=new_punch_landed,
            punching_arm_index=new_punching_arm,
            cpu_dancing_value=new_dancing,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _cpu_hit_detection_step(self, state: BoxingState) -> BoxingState:
        """CPU hit detection – body hits score."""
        horiz_dist = jnp.abs(state.left_boxer_x - state.right_boxer_x)
        vert_dist = jnp.abs(state.left_boxer_y - state.right_boxer_y)

        in_horiz_range = horiz_dist <= self._env.consts.HIT_DISTANCE_HORIZONTAL
        in_vert_range = vert_dist < self._env.consts.HIT_DISTANCE_VERTICAL

        just_reached_max = state.extended_arm_maximum[1] == 1
        punch_active = state.right_boxer_punch_active > 0
        punch_not_landed_yet = state.right_boxer_punch_landed == 0

        # No fine_vert_aligned check
        hit_landed = jnp.logical_and(
            jnp.logical_and(punch_active, just_reached_max),
            jnp.logical_and(
                jnp.logical_and(in_horiz_range, in_vert_range),
                punch_not_landed_yet,
            ),
        )

        player_not_stunned = jnp.logical_or(
            state.hit_boxer_stun_timer == 0,
            state.hit_boxer_index != 0,
        )
        valid_hit = jnp.logical_and(hit_landed, player_not_stunned)

        new_punch_landed = jnp.where(valid_hit, 1, state.right_boxer_punch_landed).astype(jnp.int32)
        new_score = jnp.where(valid_hit, state.right_boxer_score + 1, state.right_boxer_score).astype(jnp.int32)
        new_stun_timer = jnp.where(valid_hit, self._env.consts.STUN_DURATION, state.hit_boxer_stun_timer).astype(jnp.int32)
        new_hit_index = jnp.where(valid_hit, 0, state.hit_boxer_index).astype(jnp.int32)
        new_punching_arm = jnp.where(valid_hit, state.right_boxer_last_arm + 2, state.punching_arm_index).astype(jnp.int32)
        new_dancing = jnp.where(valid_hit, 57, state.cpu_dancing_value).astype(jnp.int32)

        return replace(
            state,
            right_boxer_score=new_score,
            hit_boxer_stun_timer=new_stun_timer,
            hit_boxer_index=new_hit_index,
            right_boxer_punch_landed=new_punch_landed,
            punching_arm_index=new_punching_arm,
            cpu_dancing_value=new_dancing,
        )
