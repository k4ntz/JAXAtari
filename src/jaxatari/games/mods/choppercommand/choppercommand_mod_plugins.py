import chex
import jax
import jax.numpy as jnp
from functools import partial

from jaxatari.environment import JAXAtariAction as Action
from jaxatari.games.jax_choppercommand import ChopperCommandState
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

class HomingPlayerMissileMod(JaxAtariInternalModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def player_missile_step(
            self,
            state: ChopperCommandState,
            curr_player_x,
            curr_player_y,
            action: chex.Array,
    ):
        fire = jnp.any(
            jnp.array([
                action == Action.FIRE,
                action == Action.UPRIGHTFIRE,
                action == Action.UPLEFTFIRE,
                action == Action.DOWNFIRE,
                action == Action.DOWNRIGHTFIRE,
                action == Action.DOWNLEFTFIRE,
                action == Action.RIGHTFIRE,
                action == Action.LEFTFIRE,
                action == Action.UPFIRE,
            ])
        )

        missile_y = curr_player_y + 6
        cooldown = jnp.maximum(state.player_missile_cooldown - 1, 0)

        def try_spawn(missiles):
            spawn_x = jnp.where(
                state.player_facing_direction == -1,
                curr_player_x - self._env.consts.PLAYER_MISSILE_WIDTH,
                curr_player_x + self._env.consts.PLAYER_SIZE[0],
            )
            new_missile = jnp.array([
                spawn_x,
                missile_y,
                state.player_facing_direction,
                spawn_x
            ], dtype=jnp.int32)

            # Find first free slot (direction == 0 means inactive)
            free = missiles[:, 2] == 0
            first_free_mask = (jnp.cumsum(free.astype(jnp.int32)) == 1) & free

            # Spawn in first free slot; when MAX_PLAYER_MISSILES==1 this is just slot 0
            updated = jnp.where(
                first_free_mask[:, None],
                jnp.broadcast_to(new_missile, missiles.shape),
                missiles
            )
            did_spawn = jnp.any(first_free_mask)
            return updated, did_spawn

        def spawn_if_possible(missiles):
            def do_spawn(_):
                return try_spawn(missiles)

            def skip_spawn(_):
                return missiles, False

            return jax.lax.cond(
                jnp.logical_and(jnp.logical_and(fire, state.pause_timer > self._env.consts.DEATH_PAUSE_FRAMES),
                                cooldown == 0), do_spawn, skip_spawn, operand=None)

        def update_missile(missile):
            exists = missile[2] != 0
            # compute next x (as before)
            new_x = missile[0] + missile[2] * self._env.consts.MISSILE_SPEED + state.player_velocity_x

            # Gather all enemies into a single array (shape: N,4)
            enemies = jnp.concatenate([state.jet_positions, state.chopper_positions], axis=0)

            # Active enemy mask (direction != 0 and not in death animation)
            enemy_active = jnp.logical_and(enemies[:, 2] != 0, enemies[:, 3] > self._env.consts.FRAMES_DEATH_ANIMATION_ENEMY)

            # Work in float for distance math
            missile_x_f = missile[0].astype(jnp.float32)
            missile_y_f = missile[1].astype(jnp.float32)
            enemy_x = enemies[:, 0].astype(jnp.float32)
            enemy_y = enemies[:, 1].astype(jnp.float32)

            # Consider only enemies that lie in the direction the missile moves (ahead in x)
            # missile[2] is direction: -1 (left) or 1 (right)
            dir_f = missile[2].astype(jnp.float32)
            ahead_mask = jnp.logical_and(enemy_active, ((enemy_x - missile_x_f) * dir_f) > 0.0)

            # If there are any enemies ahead, pick the one with smallest horizontal distance
            def pick_target_y():
                # distances only for ahead candidates; others get a large sentinel
                dists = jnp.where(ahead_mask, jnp.abs(enemy_x - missile_x_f), jnp.full(enemy_x.shape, 1e9, dtype=jnp.float32))
                idx = jnp.argmin(dists)
                return enemy_y[idx]

            any_ahead = jnp.any(ahead_mask)
            target_y_f = jax.lax.cond(any_ahead, lambda _: pick_target_y(), lambda _: missile_y_f, operand=None)

            # Move missile Y by at most 1 pixel toward target each frame (integer step)
            # If no target ahead, target_y_f == missile_y_f and Y doesn't change.
            y_delta = jnp.sign(target_y_f - missile_y_f).astype(jnp.int32)
            new_y = (missile_y_f.astype(jnp.int32) + y_delta).astype(jnp.int32)

            updated = jnp.array([
                new_x,  # updated x
                new_y,  # updated y (homing)
                missile[2],  # direction stays
                missile[3]  # x_spawn stays
            ], dtype=jnp.int32)

            chopper_pos = (self._env.consts.WIDTH // 2) - 8 + state.local_player_offset + (
                        state.player_velocity_x * self._env.consts.DISTANCE_WHEN_FLYING)
            left_bound = state.player_x - chopper_pos - self._env.consts.PLAYER_MISSILE_WIDTH
            right_bound = state.player_x + (self._env.consts.WIDTH - chopper_pos)

            out_of_bounds = jnp.logical_or(updated[0] < left_bound, updated[0] > right_bound)
            return jnp.where(jnp.logical_and(exists, ~out_of_bounds), updated, jnp.array([0, 0, 0, 0], dtype=jnp.int32))

        updated_missiles = jax.vmap(update_missile)(state.player_missile_positions)
        # jax.debug.print("{}", updated_missiles)
        updated_missiles, did_spawn = spawn_if_possible(updated_missiles)
        new_cooldown = jnp.where(did_spawn, self._env.consts.MISSILE_COOLDOWN_FRAMES, cooldown)

        return updated_missiles, new_cooldown