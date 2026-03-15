import jax
import jax.numpy as jnp
from functools import partial
from jaxatari.games.jax_battlezone import (BattlezoneState, BattlezoneObservation,
                                           BattlezoneConstants, Enemy, Projectile)
from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action, ObjectObservation
import chex



class GlobalVisionMod(JaxAtariInternalModPlugin):
    #conflicts_with = ["small_fov"]

    def _get_observation(self, state: BattlezoneState):
        # -------------------------------enemies----------------------------------------------
        enemies_u, _ = self._env.world_cords_to_viewport_cords_arr(state.enemies.x, state.enemies.z,
                                                          self._env.consts.CAMERA_FOCAL_LENGTH)
        zoom_factor = jnp.clip(((-0.15 * (state.enemies.distance) + 21.0) / 20.0), 0.0, 1.0)
        pixels_deleted_due_to_zoom = (jnp.round(1.0 / zoom_factor) + 1)
        enemies_width = self._env.consts.ENEMY_WIDTHS[state.enemies.enemy_type] - pixels_deleted_due_to_zoom
        enemies_heights = self._env.consts.ENEMY_HEIGHTS[state.enemies.enemy_type] - pixels_deleted_due_to_zoom
        enemy_mask = state.enemies.active
        enemies_u = jnp.where(enemy_mask, enemies_u - (enemies_width / 2), -100)
        enemies = ObjectObservation.create(
            x=enemies_u,
            y=jnp.where(enemy_mask, jnp.full((len(enemies_u),),
                                             self._env.consts.ENEMY_POS_Y - (enemies_heights / 2)), -100),
            width=jnp.where(enemy_mask, enemies_width, -100),
            height=jnp.where(enemy_mask, enemies_heights, -100),
        )

        # ---------------------------------projectiles------------------------------------------------
        enemy_projectiles_u, enemy_projectiles_v = self._env.world_cords_to_viewport_cords_arr(state.enemy_projectiles.x,
                                                                                      state.enemy_projectiles.z,
                                                                                      self._env.consts.CAMERA_FOCAL_LENGTH)
        enemy_projectiles_mask = state.enemy_projectiles.active
        player_projectiles_u, player_projectiles_v = self._env.world_cords_to_viewport_cords_arr(state.player_projectile.x,
                                                                                        state.player_projectile.z,
                                                                                        self._env.consts.CAMERA_FOCAL_LENGTH)
        projectiles_x = jnp.concatenate([
            jnp.where(state.player_projectile.active, jnp.atleast_1d(player_projectiles_u - 1), -100),
            jnp.where(enemy_projectiles_mask, enemy_projectiles_u - 1, -100)
        ])
        projectiles_y = jnp.concatenate([
            jnp.where(state.player_projectile.active, jnp.atleast_1d(player_projectiles_v - 1), -100),
            jnp.where(enemy_projectiles_mask, enemy_projectiles_v - 1, -100)
        ])
        projectiles = ObjectObservation.create(
            x=projectiles_x,
            y=projectiles_y,
            width=jnp.full((len(projectiles_x),), 2),
            height=jnp.full((len(projectiles_x),), 3),
        )
        # -----------------------------radar----------------------------------------
        # Check if enemy in radar radius
        # Scale to radar size
        scale_val = self._env.consts.RADAR_RADIUS / self._env.consts.RADAR_MAX_SCAN_RADIUS
        radar_enemies_x = state.enemies.x * scale_val
        radar_enemies_z = state.enemies.z * scale_val * (-1)
        # Offset to radar center
        radar_enemies_x = jnp.round(radar_enemies_x + self._env.consts.RADAR_CENTER_X).astype(jnp.int32)
        radar_enemies_z = jnp.round(radar_enemies_z + self._env.consts.RADAR_CENTER_Y).astype(jnp.int32)
        # Only allow in range enemies
        radar_dots = ObjectObservation.create(
            x=radar_enemies_x,
            y=radar_enemies_z,
            width=jnp.full((len(radar_enemies_x),), 1),
            height=jnp.full((len(radar_enemies_x),), 1)
        )
        # ----------------------------------------------------------------------------
        return BattlezoneObservation(
            enemies=enemies,
            radar_dots=radar_dots,
            projectiles=projectiles,
            score=state.score,
            life=state.life,
            enemy_types=jnp.where(enemy_mask, state.enemies.enemy_type, -1),
        )


class SmallFOVMod(JaxAtariInternalModPlugin):
    # conflicts_with = ["global_vision"]
    constants_overrides = {
        "CAMERA_FOCAL_LENGTH": 120,
    }


class AfkEnemiesMod(JaxAtariInternalModPlugin):
    def enemy_movement(self, enemy: Enemy, projectile: Projectile):
        return enemy, projectile

