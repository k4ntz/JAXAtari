from functools import partial
import chex
import jax
import jax.numpy as jnp

from jaxatari.games.jax_upndown import UpNDownState
from jaxatari.modification import JaxAtariInternalModPlugin


class RemoveStepRoadsMod(JaxAtariInternalModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def _is_steep_road_segment(self, current_road: chex.Array, road_index_A: chex.Array, road_index_B: chex.Array) -> chex.Array:
        return jnp.array(False)


class HigherPlayerSpeedMod(JaxAtariInternalModPlugin):
    constants_overrides = {
        "MAX_SPEED": 9,
    }


class MoreCollectiblesMod(JaxAtariInternalModPlugin):
    constants_overrides = {
        "MAX_ACTIVE_COLLECTIBLES": 4,
        "COLLECTIBLE_SPAWN_INTERVAL": 120,
    }


class MinCarSpawnGapMod(JaxAtariInternalModPlugin):
    conflicts_with = ["progressive_car_spawn_rate"]
    constants_overrides = {
        "ENEMY_SPAWN_INTERVAL_BASE": 50,
    }


class AllowJumpBackwardsMod(JaxAtariInternalModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def _jump_speed_allows_start(self, player_speed: chex.Array) -> chex.Array:
        return jnp.array(True)


class SingleLaneCarSpawnMod(JaxAtariInternalModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def _sample_enemy_spawn_road(self, rng_key: chex.PRNGKey) -> chex.Array:
        return jnp.array(1, dtype=jnp.int32)


class ProgressiveCarSpawnRateMod(JaxAtariInternalModPlugin):
    conflicts_with = ["minimum_car_spawn_gap"]

    @partial(jax.jit, static_argnums=(0,))
    def _adjust_enemy_spawn_timer(self, state: UpNDownState, spawn_timer: chex.Array) -> chex.Array:
        start_interval = jnp.int32(self._env.consts.ENEMY_SPAWN_INTERVAL_BASE)
        min_interval = jnp.int32(8)
        horizon = jnp.float32(1800.0)

        progress = jnp.clip(state.movement_steps.astype(jnp.float32) / horizon, 0.0, 1.0)
        decayed_interval = jnp.round(
            start_interval.astype(jnp.float32) - progress * (start_interval.astype(jnp.float32) - min_interval.astype(jnp.float32))
        ).astype(jnp.int32)

        target_interval = jnp.maximum(min_interval, decayed_interval)
        return jnp.minimum(spawn_timer, target_interval)

    @partial(jax.jit, static_argnums=(0,))
    def _on_level_completed(self, state: UpNDownState) -> UpNDownState:
        return state._replace(
            movement_steps=jnp.array(0, dtype=jnp.int32),
            enemy_spawn_timer=jnp.array(self._env.consts.ENEMY_SPAWN_INTERVAL_BASE, dtype=jnp.int32),
        )


class TimeDecayCollectibleValueMod(JaxAtariInternalModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def _on_level_completed(self, state: UpNDownState) -> UpNDownState:
        return state._replace(movement_steps=jnp.array(0, dtype=jnp.int32))

    @partial(jax.jit, static_argnums=(0,))
    def _collectible_score_values(self, state: UpNDownState, collectible_type_ids: chex.Array) -> chex.Array:
        base_scores = self._env.consts.COLLECTIBLE_SCORES[collectible_type_ids]
        elapsed_decay = jnp.floor(state.movement_steps.astype(jnp.float32) / 200.0).astype(jnp.int32)
        min_scores = jnp.maximum(jnp.int32(10), base_scores // 3)
        return jnp.maximum(base_scores - elapsed_decay, min_scores)
