import jax
import jax.numpy as jnp
from functools import partial
from jaxatari.modification import JaxAtariPostStepModPlugin, JaxAtariInternalModPlugin

class NoLastLineMod(JaxAtariPostStepModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        # Lane is index 4, active_flag is index 5
        # lane 3 is the lowest one (index 3 in 0-3 range)
        is_last_lane = new_state.enemies[:, 4] == 3
        new_enemies = new_state.enemies.at[:, 5].set(
            jnp.where(is_last_lane, 0, new_state.enemies[:, 5])
        )
        return new_state.replace(enemies=new_enemies)

class JetsOnlyMod(JaxAtariInternalModPlugin):
    constants_overrides = {
        "enemy_probabilities": jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)
    }

class RandomEnemiesMod(JaxAtariInternalModPlugin):
    constants_overrides = {
        "enemy_probabilities": jnp.array([1/3, 1/3, 1/3], dtype=jnp.float32)
    }

from jaxatari.games.jax_atlantis import AtlantisState

class AtlantisSpeedMod(JaxAtariInternalModPlugin):
    def __init__(self, speed):
        super().__init__()
        self.speed = speed

    @partial(jax.jit, static_argnums=(0,))
    def _move_enemies(self, state: AtlantisState) -> AtlantisState:
        cfg = self._env.consts
        enemies = state.enemies
        x_pos = enemies[:, 0]
        dx_vel = enemies[:, 2]
        enemy_ids = enemies[:, 3]
        lane_indices = enemies[:, 4]
        is_active = enemies[:, 5] == 1
        number_lanes = cfg.enemy_paths.shape[0]

        speed_modded_dx_vel = jnp.sign(dx_vel) * self.speed
        new_pos = x_pos + speed_modded_dx_vel

        on_screen = (new_pos + cfg.enemy_width[enemy_ids] > 0) & (
            new_pos < cfg.screen_width
        )

        off_screen_enemies = ~on_screen
        inactive_enemies = ~is_active

        respawn_x = jnp.where(
            dx_vel < 0, cfg.screen_width, -cfg.enemy_width[enemy_ids]
        )

        next_lanes = jnp.where(is_active, lane_indices + 1, 0)

        next_lane_free = (
            inactive_enemies
            | (next_lanes >= number_lanes)
            | ((next_lanes < number_lanes) & state.lanes_free[next_lanes])
        )

        updated_x = jnp.where(
            (off_screen_enemies & next_lane_free & is_active),
            respawn_x,
            new_pos,
        )

        updated_lanes = jnp.where(
            (off_screen_enemies & next_lane_free),
            lane_indices + 1,
            lane_indices,
        )

        flags = is_active & (updated_lanes < number_lanes)

        lane_y_positions = jnp.where(
            updated_lanes < number_lanes,
            cfg.enemy_paths[updated_lanes],
            -cfg.enemy_height[enemy_ids],
        )

        updated_enemies = enemies.at[:, 0].set(updated_x)
        updated_enemies = updated_enemies.at[:, 1].set(lane_y_positions)
        updated_enemies = updated_enemies.at[:, 2].set(
            jnp.where(flags, speed_modded_dx_vel, enemies[:, 2])
        )
        updated_enemies = updated_enemies.at[:, 4].set(updated_lanes)
        updated_enemies = updated_enemies.at[:, 5].set(flags)

        lane_masks = []
        for i in range(len(cfg.enemy_paths)):
            lane_mask = (updated_enemies[:, 4] == i) & flags
            lane_is_occupied = jnp.any(lane_mask)
            lane_masks.append(~lane_is_occupied)

        free_lanes = jnp.array(lane_masks)
        return state.replace(enemies=updated_enemies, lanes_free=free_lanes)

class SpeedModeSlowMod(AtlantisSpeedMod):
    def __init__(self):
        super().__init__(speed=2)

class SpeedModeMediumMod(AtlantisSpeedMod):
    def __init__(self):
        super().__init__(speed=4)

class SpeedModeFastMod(AtlantisSpeedMod):
    def __init__(self):
        super().__init__(speed=6)
