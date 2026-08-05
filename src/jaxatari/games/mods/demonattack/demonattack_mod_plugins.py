from functools import partial

import jax
import jax.numpy as jnp

from jaxatari.games.jax_demonattack import (
    DEMON_STATUS_NORMAL,
    DEMON_STATUS_SMALL,
    DemonAttackState,
)
from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin


def _demon_ids(env) -> jax.Array:
    return jnp.arange(env.consts.MAX_DEMONS, dtype=jnp.int32)


def _ready_demons(state: DemonAttackState) -> jax.Array:
    return jnp.logical_and(state.spawn_anim_timer <= 0, state.spawn_pause_timer <= 0)


def _active_demons(state: DemonAttackState) -> jax.Array:
    return jnp.logical_and(state.demons_alive, _ready_demons(state))


def _normal_active_demons(state: DemonAttackState) -> jax.Array:
    return jnp.logical_and(
        _active_demons(state),
        state.demon_status == DEMON_STATUS_NORMAL,
    )


def _clear_bombs(env, state: DemonAttackState) -> DemonAttackState:
    return state.replace(
        bomb_active=jnp.zeros_like(state.bomb_active, dtype=jnp.bool_),
        bomb_burst_step=jnp.asarray(env.consts.BOMB_BURST_RATES, dtype=state.bomb_burst_step.dtype),
        bomb_burst_length=jnp.zeros_like(state.bomb_burst_length),
        bomb_burst_timer=jnp.zeros_like(state.bomb_burst_timer),
        bomb_action_counter=jnp.zeros_like(state.bomb_action_counter),
    )


def _restore_lives(env, state: DemonAttackState) -> DemonAttackState:
    return state.replace(
        lives=jnp.asarray(env.consts.MAX_BUNKERS, dtype=state.lives.dtype),
        game_over=jnp.asarray(False, dtype=state.game_over.dtype),
    )


class FastPlayerMod(JaxAtariInternalModPlugin):
    """Moves the player twice as fast."""

    constants_overrides = {
        "PLAYER_SPEED": 2,
    }


class FastLaserMod(JaxAtariInternalModPlugin):
    """Makes the player laser travel faster on every wave pattern."""

    constants_overrides = {
        "WAVE_LASER_SPEED_TABLE": (6, 7, 8, 8, 9, 9),
    }


class SlowEnemyShotsMod(JaxAtariInternalModPlugin):
    """Keeps enemy shots at the slowest speed for all difficulty levels."""

    constants_overrides = {
        "ENEMY_SHOT_SPEED_TABLE": (1, 1, 1, 1, 1, 1),
    }


class ShortWavesMod(JaxAtariInternalModPlugin):
    """Reduces the number of demons required to finish each wave."""

    constants_overrides = {
        "WAVE_TOTAL_DEMONS": 3,
    }


class NoEnemyShotsMod(JaxAtariPostStepModPlugin):
    """Removes active enemy shots after each reset and step."""

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: DemonAttackState, new_state: DemonAttackState) -> DemonAttackState:
        return _clear_bombs(self._env, new_state)

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state: DemonAttackState):
        state = _clear_bombs(self._env, state)
        return self._env._get_observation(state), state


class InfiniteLivesMod(JaxAtariPostStepModPlugin):
    """Restores the bunker/life count after each reset and step."""

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: DemonAttackState, new_state: DemonAttackState) -> DemonAttackState:
        return _restore_lives(self._env, new_state)

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state: DemonAttackState):
        state = _restore_lives(self._env, state)
        return self._env._get_observation(state), state


class RelentlessWavesMod(JaxAtariInternalModPlugin):
    """Increases pressure with quicker respawns and more frequent full bursts."""

    constants_overrides = {
        "RESPAWN_DELAY": 10,
        "SPAWN_MOVE_PAUSE": 4,
        "WAVE_TOTAL_DEMONS": 12,
        "ENEMY_SHOT_ACTION_TABLE": (4, 4, 3, 3, 3, 2, 3, 2, 3, 2, 3, 2),
        "BOMB_PRE_FIRE_PAUSE": 8,
        "BOMB_BURST_LENGTH_OPTIONS": (5, 7, 7, 7),
        "BOMB_JITTER_X_TABLE": (-1, 0, 1, 0, -1, 0, 1),
    }


class LateWaveStartMod(JaxAtariPostStepModPlugin):
    """Starts each reset on wave 8 so tracking projectiles and late sprites are active."""

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state: DemonAttackState):
        state = self._env._initialize_wave_state(
            state,
            jnp.asarray(self._env.consts.TRACKING_PROJECTILES_START_WAVE, dtype=state.wave_number.dtype),
        )
        return self._env._get_observation(state), state


class PlayerGuidedLaserMod(JaxAtariPostStepModPlugin):
    """Lets the player steer an active laser horizontally after firing."""

    conflicts_with = ["homing_laser"]

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: DemonAttackState, new_state: DemonAttackState) -> DemonAttackState:
        guided_x = new_state.player_x + self._env.consts.PLAYER_SIZE[1] // 2
        return new_state.replace(
            laser_x=jnp.where(new_state.laser_active, guided_x, new_state.laser_x),
        )


class HomingLaserMod(JaxAtariPostStepModPlugin):
    """Steers the active player laser toward the nearest hittable demon."""

    conflicts_with = ["player_guided_laser"]

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: DemonAttackState, new_state: DemonAttackState) -> DemonAttackState:
        consts = self._env.consts
        hittable = _active_demons(new_state)
        demon_center_x = new_state.demons_x + consts.DEMON_SIZE[1] // 2
        target_idx = jnp.argmin(jnp.where(
            hittable,
            jnp.abs(demon_center_x - new_state.laser_x),
            10_000,
        ))
        target_x = demon_center_x[target_idx] - consts.LASER_SIZE[1] // 2
        laser_delta = jnp.clip(target_x - new_state.laser_x, -2, 2)
        laser_x = jnp.clip(
            new_state.laser_x + laser_delta,
            consts.DEMON_MIN_X,
            consts.DEMON_MAX_X,
        )
        should_home = jnp.logical_and(new_state.laser_active, jnp.any(hittable))
        return new_state.replace(
            laser_x=jnp.where(should_home, laser_x, new_state.laser_x),
        )


class TeleportingDemonsMod(JaxAtariPostStepModPlugin):
    """Blinks one active demon in place and then teleports it."""

    BLINK_FRAMES = 28
    TELEPORT_INTERVAL = 192

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: DemonAttackState, new_state: DemonAttackState) -> DemonAttackState:
        consts = self._env.consts
        ids = _demon_ids(self._env)
        warning_active = new_state.spawn_pause_timer > consts.SPAWN_MOVE_PAUSE
        finish_warning = jnp.logical_and(
            prev_state.spawn_pause_timer > consts.SPAWN_MOVE_PAUSE,
            new_state.spawn_pause_timer <= consts.SPAWN_MOVE_PAUSE,
        )
        teleport_busy = jnp.logical_or(
            jnp.any(warning_active),
            jnp.any(prev_state.spawn_pause_timer > consts.SPAWN_MOVE_PAUSE),
        )
        eligible = _active_demons(new_state)
        eligible = jnp.logical_and(
            eligible,
            new_state.demon_status != DEMON_STATUS_SMALL,
        )
        eligible = jnp.logical_and(eligible, jnp.logical_not(teleport_busy))
        due = jnp.mod(new_state.step_counter, self.TELEPORT_INTERVAL) == 0
        desired_slot = jnp.mod(
            new_state.step_counter // self.TELEPORT_INTERVAL + new_state.wave_number,
            consts.MAX_DEMONS,
        )
        candidate_order = jnp.mod(desired_slot + ids, consts.MAX_DEMONS)
        ordered_eligible = eligible[candidate_order]
        first_ordered_idx = jnp.argmax(ordered_eligible.astype(jnp.int32))
        selected_slot = candidate_order[first_ordered_idx]
        has_target = jnp.any(eligible)
        start_warning = jnp.logical_and(
            due,
            jnp.logical_and(has_target, ids == selected_slot),
        )

        # When interrupted, it doesnt teleport anymore and only spawns split demons like before
        uninterrupted_normal = jnp.logical_and(
            prev_state.demon_status == DEMON_STATUS_NORMAL,
            new_state.demon_status == DEMON_STATUS_NORMAL,
        )
        teleport = jnp.logical_and(finish_warning, uninterrupted_normal)
        x_span = (
            consts.DEMON_MAX_X
            - consts.DEMON_MIN_X
            - consts.DEMON_SIZE[1]
        )
        min_gap = consts.DEMON_MIN_VERTICAL_DISTANCE
        lane_min_y = jnp.asarray((
            consts.DEMON_MIN_Y,
            new_state.demons_y[0] + min_gap,
            new_state.demons_y[1] + min_gap,
        ), dtype=jnp.int32)
        lane_max_y = jnp.asarray((
            new_state.demons_y[1] - min_gap,
            new_state.demons_y[2] - min_gap,
            consts.DEMON_MAX_Y,
        ), dtype=jnp.int32)
        lane_center_y = jnp.asarray((
            (consts.DEMON_MIN_Y + new_state.demons_y[1]) // 2,
            (new_state.demons_y[0] + new_state.demons_y[2]) // 2,
            (new_state.demons_y[1] + consts.DEMON_MAX_Y) // 2,
        ), dtype=jnp.int32)
        seed = new_state.step_counter + new_state.wave_number * 13 + ids * 29
        target_x = consts.DEMON_MIN_X + jnp.mod(seed * 11, x_span)
        target_y = jnp.clip(
            lane_center_y + jnp.mod(seed * 7, 5) - 2,
            lane_min_y,
            lane_max_y,
        )
        return new_state.replace(
            demons_x=jnp.where(teleport, target_x, new_state.demons_x),
            demons_y=jnp.where(teleport, target_y, new_state.demons_y),
            spawn_pause_timer=jnp.where(
                start_warning,
                consts.SPAWN_MOVE_PAUSE + self.BLINK_FRAMES,
                jnp.where(teleport, 0, new_state.spawn_pause_timer),
            ),
            demon_moving_right=jnp.where(
                teleport,
                target_x < new_state.player_x,
                new_state.demon_moving_right,
            ),
            demon_moving_down=jnp.where(teleport, True, new_state.demon_moving_down),
        )


class SideStepLowestDemonsMod(JaxAtariPostStepModPlugin):
    """Makes the lowest normal demon sidestep away from near misses until no other demon is alive."""

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: DemonAttackState, new_state: DemonAttackState) -> DemonAttackState:
        consts = self._env.consts
        ids = _demon_ids(self._env)
        active = _normal_active_demons(new_state)
        multiple_demons_alive = jnp.sum(new_state.demons_alive.astype(jnp.int32)) > 1
        bottom_slot = jnp.asarray(consts.MAX_DEMONS - 1, dtype=jnp.int32)
        moved_this_step = jnp.logical_or(
            prev_state.demons_x != new_state.demons_x,
            prev_state.demons_y != new_state.demons_y,
        )
        active = jnp.logical_and(
            active,
            jnp.logical_and(multiple_demons_alive, ids == bottom_slot),
        )
        active = jnp.logical_and(active, moved_this_step)

        demon_center = new_state.demons_x + consts.DEMON_SIZE[1] // 2
        laser_distance = jnp.abs(demon_center - new_state.laser_x)
        laser_near_x = laser_distance <= 16
        laser_in_lane = jnp.logical_and(
            new_state.laser_y <= new_state.demons_y + consts.DEMON_SIZE[0] + 12,
            new_state.laser_y >= new_state.demons_y - 28,
        )
        threatened = jnp.logical_and(
            active,
            jnp.logical_and(new_state.laser_active, jnp.logical_and(laser_near_x, laser_in_lane)),
        )
        dodge_active = threatened
        dodge_dir = jnp.where(
            new_state.laser_x < demon_center,
            2,
            -2,
        )
        demons_x = jnp.clip(
            new_state.demons_x + jnp.where(dodge_active, dodge_dir, 0),
            consts.DEMON_MIN_X,
            consts.DEMON_MAX_X - consts.DEMON_SIZE[1],
        )
        return new_state.replace(
            demons_x=demons_x,
            demon_moving_right=jnp.where(dodge_active, dodge_dir > 0, new_state.demon_moving_right),
        )


class ZigZagMovementDemonsMod(JaxAtariPostStepModPlugin):
    """Adds a zigzag movement pattern to normal demons."""

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: DemonAttackState, new_state: DemonAttackState) -> DemonAttackState:
        consts = self._env.consts
        ids = _demon_ids(self._env)
        normal_active = _normal_active_demons(new_state)
        source_firing = jnp.logical_and(
            ids == new_state.bomb_source_idx,
            jnp.logical_or(new_state.bomb_burst_length > 0, jnp.any(new_state.bomb_active)),
        )
        normal_active = jnp.logical_and(normal_active, jnp.logical_not(source_firing))
        phase = jnp.mod(new_state.step_counter + ids * 13, 48)
        zigzag = phase < 24
        x_step = jnp.where(zigzag, 1, -1)
        y_pulse = phase == 0
        y_step = jnp.where(new_state.demon_moving_down, 1, -1)
        demons_x = jnp.clip(
            jnp.where(normal_active, prev_state.demons_x + x_step, new_state.demons_x),
            consts.DEMON_MIN_X,
            consts.DEMON_MAX_X - consts.DEMON_SIZE[1],
        )
        demons_y = jnp.clip(
            jnp.where(
                jnp.logical_and(normal_active, y_pulse),
                prev_state.demons_y + y_step,
                new_state.demons_y,
            ),
            consts.DEMON_MIN_Y,
            consts.DEMON_MAX_Y - consts.DEMON_SIZE[0],
        )
        hit_x_edge = jnp.logical_or(
            demons_x <= consts.DEMON_MIN_X,
            demons_x >= consts.DEMON_MAX_X - consts.DEMON_SIZE[1],
        )
        return new_state.replace(
            demons_x=demons_x,
            demons_y=demons_y,
            demon_moving_right=jnp.where(
                jnp.logical_and(normal_active, hit_x_edge),
                jnp.logical_not(new_state.demon_moving_right),
                new_state.demon_moving_right,
            ),
            demon_moving_down=jnp.where(
                jnp.logical_and(normal_active, y_pulse),
                jnp.logical_not(new_state.demon_moving_down),
                new_state.demon_moving_down,
            ),
        )
