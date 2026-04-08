import jax
import jax.numpy as jnp
from functools import partial

from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin
from jaxatari.games.montezuma2.core import Montezuma2State

# --- Gameplay & Ability Mods ---

class InfiniteAmuletMod(JaxAtariPostStepModPlugin):
    """
    Post-step mod to keep the amulet active forever.
    """
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: Montezuma2State, new_state: Montezuma2State):
        return new_state.replace(
            amulet_time=jnp.array(660, dtype=jnp.int32),
            inventory=new_state.inventory.at[3].set(1)
        )

class SuperJumpMod(JaxAtariInternalModPlugin):
    """
    Internal mod to increase jump height.
    """
    constants_overrides = {
        "JUMP_Y_OFFSETS": jnp.array([3, 3, 3, 3, 2, 2, 2, 1, 1, 0, 0, 0, 0, 0, -1, -1, -2, -2, -2, -3, -3, -3, -3], dtype=jnp.int32)
    }

class FastPlayerMod(JaxAtariInternalModPlugin):
    """
    Internal mod to increase player speed.
    """
    constants_overrides = {
        "PLAYER_SPEED": 2
    }

class NoFallDamageMod(JaxAtariInternalModPlugin):
    """
    Internal mod to disable fall damage.
    """
    constants_overrides = {
        "MAX_FALL_DISTANCE": 255
    }

# --- Utility & Visual Mods ---

class RevealMapMod(JaxAtariInternalModPlugin):
    """
    Internal mod to make dark rooms always visible, by granting the player 
    the torch effect during rendering.
    """
    @partial(jax.jit, static_argnums=(0,))
    def _render_hook_pre_render(self, state: Montezuma2State) -> Montezuma2State:
        return state.replace(
            inventory=state.inventory.at[2].set(1)
        )

class DebugHudMod(JaxAtariInternalModPlugin):
    """
    Internal mod to display debug info (Room ID, X, Y) on the HUD.
    """
    @partial(jax.jit, static_argnums=(0,))
    def _render_hook_post_ui(self, raster: jnp.ndarray, state: Montezuma2State) -> jnp.ndarray:
        jr = self._env.renderer.jr
        masks = self._env.renderer.digit_masks
        
        # In Montezuma2, masks[0] is 'digit_none', masks[1] is '0', etc.
        # So we add 1 to the digits to get the correct sprite index.
        
        # Room ID
        room_digits = jr.int_to_digits(state.room_id, max_digits=2) + 1
        raster = jr.render_label(raster, 10, 10, room_digits, masks, 7, 2)
        
        # Player X
        x_digits = jr.int_to_digits(state.player_x, max_digits=3) + 1
        raster = jr.render_label(raster, 10, 20, x_digits, masks, 7, 3)
        
        # Player Y
        y_digits = jr.int_to_digits(state.player_y, max_digits=3) + 1
        raster = jr.render_label(raster, 10, 30, y_digits, masks, 7, 3)
        
        return raster

class NoEnemiesMod(JaxAtariPostStepModPlugin):
    """
    Post-step mod to immediately remove any enemies in the current room.
    """
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: Montezuma2State, new_state: Montezuma2State):
        return new_state.replace(
            enemies_active=jnp.zeros_like(new_state.enemies_active)
        )

class CenterBouncingSkullMod(JaxAtariPostStepModPlugin):
    """
    Post-step mod to make the rolling skull (type 1) jump vertically at the center of the screen.
    Forces its X position to 77, enables bouncing, and removes horizontal direction.
    """
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: Montezuma2State, new_state: Montezuma2State):
        is_rolling_skull = new_state.enemies_type == 1
        
        # Center X is approximately 77 (160 / 2 - 6 / 2)
        new_enemies_x = jnp.where(is_rolling_skull, 77, new_state.enemies_x)
        
        # Enable bouncing for vertical jumping
        new_enemies_bouncing = jnp.where(is_rolling_skull, 1, new_state.enemies_bouncing)
        
        # Disable horizontal movement
        new_enemies_direction = jnp.where(is_rolling_skull, 0, new_state.enemies_direction)
        
        return new_state.replace(
            enemies_x=new_enemies_x,
            enemies_bouncing=new_enemies_bouncing,
            enemies_direction=new_enemies_direction
        )

class RollingSkullsMod(JaxAtariPostStepModPlugin):
    """
    Post-step mod to make the two skulls that usually bump (bounce) roll instead,
    and slightly augment the space between them.
    """
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: Montezuma2State, new_state: Montezuma2State):
        is_room_5 = new_state.room_id == 5
        
        # We only want to apply the initial position change when entering the room
        just_entered_room_5 = jnp.logical_and(is_room_5, prev_state.room_id != 5)
        
        # Modify enemies in room 5
        # Index 0: 112 -> 120, Index 1: 95 -> 87 (augmenting space from 17 to 33)
        new_enemies_x = jnp.where(just_entered_room_5, 
                                  new_state.enemies_x.at[0].set(120).at[1].set(87), 
                                  new_state.enemies_x)
        
        # Disable bouncing for rolling animation and no vertical jump
        # This can be applied every step when in room 5 to ensure they stay rolling
        new_enemies_bouncing = jnp.where(is_room_5, 
                                         new_state.enemies_bouncing.at[0].set(0).at[1].set(0), 
                                         new_state.enemies_bouncing)
        
        return new_state.replace(
            enemies_x=new_enemies_x,
            enemies_bouncing=new_enemies_bouncing
        )

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state: Montezuma2State):
        is_room_5 = state.room_id == 5
        
        new_enemies_x = jnp.where(is_room_5, 
                                  state.enemies_x.at[0].set(120).at[1].set(87), 
                                  state.enemies_x)
        
        new_enemies_bouncing = jnp.where(is_room_5, 
                                         state.enemies_bouncing.at[0].set(0).at[1].set(0), 
                                         state.enemies_bouncing)
        
        state = state.replace(
            enemies_x=new_enemies_x,
            enemies_bouncing=new_enemies_bouncing
        )
        return obs, state


class MovingSnakesMod(JaxAtariPostStepModPlugin):
    """
    Post-step mod to make snakes move horizontally in all rooms.
    They will move on the x axis from 20 to 140.
    """
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: Montezuma2State, new_state: Montezuma2State):
        is_snake = new_state.enemies_type == 4
        
        new_enemies_direction = jnp.where(is_snake,
                                          jnp.where(new_state.enemies_direction == 0, 1, new_state.enemies_direction),
                                          new_state.enemies_direction)
        
        new_eminx = jnp.where(is_snake, 20, new_state.enemies_min_x)
        new_emaxx = jnp.where(is_snake, 140, new_state.enemies_max_x)
        
        # Ensure their X position is strictly within bounds so they don't get stuck bouncing on speed=0 frames
        new_enemies_x = jnp.where(is_snake, 
                                  jnp.clip(new_state.enemies_x, 21, 139), 
                                  new_state.enemies_x)

        return new_state.replace(
            enemies_direction=new_enemies_direction,
            enemies_min_x=new_eminx,
            enemies_max_x=new_emaxx,
            enemies_x=new_enemies_x
        )

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state: Montezuma2State):
        is_snake = state.enemies_type == 4
        
        new_enemies_direction = jnp.where(is_snake,
                                          jnp.where(state.enemies_direction == 0, 1, state.enemies_direction),
                                          state.enemies_direction)
        
        new_eminx = jnp.where(is_snake, 20, state.enemies_min_x)
        new_emaxx = jnp.where(is_snake, 140, state.enemies_max_x)

        new_enemies_x = jnp.where(is_snake, 
                                  jnp.clip(state.enemies_x, 21, 139), 
                                  state.enemies_x)

        state = state.replace(
            enemies_direction=new_enemies_direction,
            enemies_min_x=new_eminx,
            enemies_max_x=new_emaxx,
            enemies_x=new_enemies_x
        )
        return obs, state


class JumpingSpidersMod(JaxAtariPostStepModPlugin):
    """
    Post-step mod to make all spiders (type 3) jump (bounce).
    """
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: Montezuma2State, new_state: Montezuma2State):
        is_spider = new_state.enemies_type == 3
        
        new_enemies_bouncing = jnp.where(is_spider, 1, new_state.enemies_bouncing)
        
        return new_state.replace(
            enemies_bouncing=new_enemies_bouncing
        )

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state: Montezuma2State):
        is_spider = state.enemies_type == 3
        
        new_enemies_bouncing = jnp.where(is_spider, 1, state.enemies_bouncing)
        
        state = state.replace(
            enemies_bouncing=new_enemies_bouncing
        )
        return obs, state
