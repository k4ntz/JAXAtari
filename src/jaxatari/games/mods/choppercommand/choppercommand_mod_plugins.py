import chex
import jax
import jax.numpy as jnp
from functools import partial

from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin


class AlwaysCenteredMod(JaxAtariPostStepModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        pause = jnp.where(jnp.equal(prev_state.pause_timer, self._env.consts.DEATH_PAUSE_FRAMES + 2), self._env.consts.DEATH_PAUSE_FRAMES + 1, new_state.pause_timer)
        return new_state.replace(local_player_offset=jnp.array(0), pause_timer=pause)

class MoreTrucksMod (JaxAtariInternalModPlugin):
    constants_overrides = {
        "MAX_TRUCKS": 36,
        "MIDDLE_TRUCK_INDICES": (4, 13, 22, 31),
        "ENEMY_OUT_OF_CYCLE_RIGHT": 144,
        "ENEMY_OUT_OF_CYCLE_LEFT": 144,
    }

    @partial(jax.jit, static_argnums=(0,))
    def initialize_truck_positions(self) -> chex.Array:
        # Gaps: 56 if i%9 == 0 else 32
        indices = jnp.arange(self._env.consts.MAX_TRUCKS)
        gaps = jnp.where(indices % 9 == 0, 56.0, 32.0)

        # Cumulative sum starting at -748 (first truck at -748 + 248 = -500, etc.)
        x_positions = -748.0 + jnp.cumsum(gaps)

        return jnp.stack([
            x_positions,
            jnp.full(self._env.consts.MAX_TRUCKS, 156.0),
            jnp.full(self._env.consts.MAX_TRUCKS, -1.0),
            jnp.full(self._env.consts.MAX_TRUCKS, self._env.consts.FRAMES_DEATH_ANIMATION_TRUCK + 1.0)
        ], axis=1)

    @partial(jax.jit, static_argnums=(0,))
    def initialize_enemy_positions(self, init_rng: chex.PRNGKey, ref_x: chex.Array) -> tuple[chex.Array, chex.Array]:
        """
        Spawn enemy fleets, but wrap each fleet's anchor so it is near ref_x.
        This prevents origin-spawn artifacts during the no-move pause and keeps
        the minimap consistent immediately after respawn.
        """
        total_enemies = 12
        fleet_idx = jnp.arange(total_enemies) // 3
        unit_idx = jnp.arange(total_enemies) % 3

        # Base math for anchor positions
        base_anchors = -876.0 + 312.0 * fleet_idx.astype(jnp.float32)

        # Wrap helper inline: put x within +-624 of ref
        period = jnp.asarray(1248.0, dtype=jnp.float32)
        half = jnp.asarray(624.0, dtype=jnp.float32)
        ref_x = jnp.asarray(ref_x, dtype=jnp.float32)
        anchor_x = ref_x + jnp.mod(base_anchors - ref_x + half, period) - half

        # RNG Splits
        key_dir, key_chop, key_off = jax.random.split(init_rng, 3)

        directions = jax.random.choice(
            key_dir,
            jnp.array([-1.0, 1.0], dtype=jnp.float32),
            shape=(total_enemies,),
            replace=True
        )
        x_offsets = jax.random.randint(
            key_off,
            (total_enemies,),
            -self._env.consts.ENEMY_MAXIMUM_SPAWN_OFFSET + 5,
            self._env.consts.ENEMY_MAXIMUM_SPAWN_OFFSET - 5
        ).astype(jnp.float32)

        # Generate chopper counts per fleet (shape: 4) and expand to (12)
        chopper_counts = jax.random.randint(key_chop, (4,), 0, 4)
        is_chopper = unit_idx < chopper_counts[fleet_idx]

        # Assemble final positions
        vertical_spacing = 30
        units_per_fleet = 3
        y_start = self._env.consts.HEIGHT_ONLY_PLAYING_FIELD // 2 - (units_per_fleet // 2) * vertical_spacing
        y_positions = jnp.full(total_enemies, y_start, dtype=jnp.float32) + unit_idx.astype(
            jnp.float32) * vertical_spacing

        positions = jnp.stack([
            anchor_x + x_offsets,
            y_positions,
            directions,
            jnp.full(total_enemies, self._env.consts.FRAMES_DEATH_ANIMATION_ENEMY + 5.0, dtype=jnp.float32)
        ], axis=1)

        # Distribute into jets vs choppers (jets at slots where ~is_chopper, choppers where is_chopper)
        zeros = jnp.zeros_like(positions)
        chopper_positions = jnp.where(is_chopper[:, None], positions, zeros)
        jet_positions = jnp.where(~is_chopper[:, None], positions, zeros)

        return jet_positions, chopper_positions