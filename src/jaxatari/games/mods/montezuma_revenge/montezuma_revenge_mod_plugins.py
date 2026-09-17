import jax
import jax.numpy as jnp
from functools import partial

from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin
from jaxatari.games.montezuma_revenge.core import MontezumaRevengeState

# --- Gameplay & Ability Mods ---

class InfiniteAmuletMod(JaxAtariPostStepModPlugin):
    """
    Post-step mod to keep the amulet active forever.
    """
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: MontezumaRevengeState, new_state: MontezumaRevengeState):
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
    def _render_hook_pre_render(self, state: MontezumaRevengeState) -> MontezumaRevengeState:
        return state.replace(
            inventory=state.inventory.at[2].set(1)
        )

class DebugHudMod(JaxAtariInternalModPlugin):
    """
    Internal mod to display debug info (Room ID, X, Y) on the HUD.
    """
    @partial(jax.jit, static_argnums=(0,))
    def _render_hook_post_ui(self, raster: jnp.ndarray, state: MontezumaRevengeState) -> jnp.ndarray:
        jr = self._env.renderer.jr
        masks = self._env.renderer.digit_masks
        
        # In MontezumaRevenge, masks[0] is 'digit_none', masks[1] is '0', etc.
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
    def run(self, prev_state: MontezumaRevengeState, new_state: MontezumaRevengeState):
        return new_state.replace(
            enemies_active=jnp.zeros_like(new_state.enemies_active)
        )

class CenterBouncingSkullMod(JaxAtariPostStepModPlugin):
    """
    Post-step mod to make the rolling skull (type 1) jump vertically at the center of the screen.
    Forces its X position to 77, enables bouncing, and removes horizontal direction.
    Only applies if there is exactly one skull of type 1 in the room.
    """
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: MontezumaRevengeState, new_state: MontezumaRevengeState):
        is_skull_type_1 = new_state.enemies_type == 1
        num_skulls = jnp.sum(jnp.logical_and(new_state.enemies_active == 1, is_skull_type_1))
        is_single_skull_room = num_skulls == 1
        
        is_target = jnp.logical_and(is_single_skull_room, is_skull_type_1)
        
        # Center X is approximately 77 (160 / 2 - 6 / 2)
        new_enemies_x = jnp.where(is_target, 77, new_state.enemies_x)
        
        # Enable bouncing for vertical jumping
        new_enemies_bouncing = jnp.where(is_target, 1, new_state.enemies_bouncing)
        
        # Disable horizontal movement
        new_enemies_direction = jnp.where(is_target, 0, new_state.enemies_direction)
        
        return new_state.replace(
            enemies_x=new_enemies_x,
            enemies_bouncing=new_enemies_bouncing,
            enemies_direction=new_enemies_direction
        )

class RollingSkullsMod(JaxAtariPostStepModPlugin):
    """
    Post-step mod to make the two skulls that usually bump (bounce) roll instead,
    and slightly augment the space between them.
    Only applies if there are exactly two skulls of type 1 in the room.
    """
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: MontezumaRevengeState, new_state: MontezumaRevengeState):
        is_skull_type_1 = new_state.enemies_type == 1
        num_skulls = jnp.sum(jnp.logical_and(new_state.enemies_active == 1, is_skull_type_1))
        is_double_skull_room = num_skulls == 2
        
        is_target = jnp.logical_and(is_double_skull_room, is_skull_type_1)
        
        # Apply initial position change when entering a room with 2 skulls
        just_entered_room = new_state.room_id != prev_state.room_id
        should_shift = jnp.logical_and(just_entered_room, is_double_skull_room)
        
        # General shift logic: move them further apart by 8 pixels each
        x0 = new_state.enemies_x[0]
        x1 = new_state.enemies_x[1]
        new_x0 = jnp.where(x0 < x1, x0 - 8, x0 + 8)
        new_x1 = jnp.where(x1 < x0, x1 - 8, x1 + 8)
        
        # We assume the skulls are at index 0 and 1 (standard in M2)
        shifted_x = new_state.enemies_x.at[0].set(new_x0).at[1].set(new_x1)
        new_enemies_x = jnp.where(should_shift, shifted_x, new_state.enemies_x)
        
        # Disable bouncing for rolling animation and no vertical jump
        new_enemies_bouncing = jnp.where(is_target, 0, new_state.enemies_bouncing)
        
        return new_state.replace(
            enemies_x=new_enemies_x,
            enemies_bouncing=new_enemies_bouncing
        )

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state: MontezumaRevengeState):
        is_skull_type_1 = state.enemies_type == 1
        num_skulls = jnp.sum(jnp.logical_and(state.enemies_active == 1, is_skull_type_1))
        is_double_skull_room = num_skulls == 2
        
        is_target = jnp.logical_and(is_double_skull_room, is_skull_type_1)
        
        x0 = state.enemies_x[0]
        x1 = state.enemies_x[1]
        new_x0 = jnp.where(x0 < x1, x0 - 8, x0 + 8)
        new_x1 = jnp.where(x1 < x0, x1 - 8, x1 + 8)
        
        new_enemies_x = jnp.where(is_double_skull_room, 
                                  state.enemies_x.at[0].set(new_x0).at[1].set(new_x1), 
                                  state.enemies_x)
        
        new_enemies_bouncing = jnp.where(is_target, 0, state.enemies_bouncing)
        
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
    def run(self, prev_state: MontezumaRevengeState, new_state: MontezumaRevengeState):
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
    def after_reset(self, obs, state: MontezumaRevengeState):
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
    def run(self, prev_state: MontezumaRevengeState, new_state: MontezumaRevengeState):
        is_spider = new_state.enemies_type == 3
        
        new_enemies_bouncing = jnp.where(is_spider, 1, new_state.enemies_bouncing)
        
        return new_state.replace(
            enemies_bouncing=new_enemies_bouncing
        )

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state: MontezumaRevengeState):
        is_spider = state.enemies_type == 3
        
        new_enemies_bouncing = jnp.where(is_spider, 1, state.enemies_bouncing)
        
        state = state.replace(
            enemies_bouncing=new_enemies_bouncing
        )
        return obs, state

class SwordKillBonusMod(JaxAtariInternalModPlugin):
    """
    Internal mod to provide bonus points if the player kills an enemy with the sword.
    It overrides the KILL_ENEMY_REWARD constant to 300.
    """
    constants_overrides = {
        "KILL_ENEMY_REWARD": 300
    }

class ThreeSwordsMod(JaxAtariPostStepModPlugin):
    """
    Post-step mod to start the player with 3 swords.
    Since the base game now natively supports up to 3 swords in the inventory,
    we simply set it at reset.
    """
    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state: MontezumaRevengeState):
        # Give 3 swords natively
        new_inventory = state.inventory.at[1].set(3)
        
        state = state.replace(
            inventory=new_inventory
        )
        return obs, state
