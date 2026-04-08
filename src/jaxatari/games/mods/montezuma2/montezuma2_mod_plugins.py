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
