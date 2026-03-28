import jax
import jax.numpy as jnp
from functools import partial
import chex

from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin
from jaxatari.games.jax_frostbite import FrostbiteState

class NoEnemiesMod(JaxAtariInternalModPlugin):
    """
    Internal mod to disable all enemies (polar bear, geese, crabs, clams) 
    and only keep the fishes.
    """
    
    @partial(jax.jit, static_argnums=(0,))
    def _update_polar_grizzly(self, state: FrostbiteState):
        """Disable polar grizzly bear."""
        return state.replace(polar_grizzly_active=jnp.array(0, dtype=jnp.int32))

    @partial(jax.jit, static_argnums=(0,))
    def _spawn_obstacles_vec(self, state: FrostbiteState, spawn_mask: jnp.ndarray) -> FrostbiteState:
        """Override to only spawn fishes."""
        from jaxatari.games.jax_frostbite import JaxFrostbite
        new_state = JaxFrostbite._spawn_obstacles_vec(self._env, state, spawn_mask)
        
        is_spawned = spawn_mask.astype(jnp.bool_)
        forced_type = jnp.int32(1) # ID_FISH
        
        new_type = jnp.where(is_spawned, forced_type, new_state.obstacle_types)
        max_copies = new_state.obstacle_max_copies
        initial_fish_mask = (jnp.int32(1) << max_copies) - jnp.int32(1)
        new_fish_mask = jnp.where(is_spawned, initial_fish_mask, new_state.fish_alive_mask)
        
        return new_state.replace(
            obstacle_types=new_type,
            fish_alive_mask=new_fish_mask
        )

class RedIceMod(JaxAtariInternalModPlugin):
    """
    Changes the color of the ice blocks to red.
    """
    # Frostbite defines colors as Atari hex codes.
    # We still override the logic colors so they match expected behavior,
    # but we ALSO provide the RGB tuples to actually tint the sprites!
    constants_overrides = {
        "COLOR_ICE_WHITE": 0x44, # Red 
        "COLOR_ICE_BLUE": 0x42,  # Darker Red
        "RGB_ICE_WHITE": (255, 50, 50),
        "RGB_ICE_BLUE": (200, 0, 0),
    }

class _StaticIceMod(JaxAtariInternalModPlugin):
    """
    Sets ice block horizontal speeds to zero.
    """
    @partial(jax.jit, static_argnums=(0,))
    def _calculate_speeds_for_level(self, level: jnp.ndarray):
        from jaxatari.games.jax_frostbite import JaxFrostbite
        speeds = JaxFrostbite._calculate_speeds_for_level(self._env, level)
        speeds['ice_speed_whole'] = jnp.zeros_like(speeds['ice_speed_whole'])
        speeds['ice_speed_frac'] = jnp.zeros_like(speeds['ice_speed_frac'])
        return speeds

class _MisalignedIceMod(JaxAtariPostStepModPlugin):
    """
    Makes the ice blocks vertically aligned with each other but off-center (staggered gaps).
    """
    @partial(jax.jit, static_argnums=(0,))
    def _apply_alignment(self, state: FrostbiteState) -> FrostbiteState:
        # 68 perfectly misaligns them (vertically aligned with each other but off center)
        misaligned_ice_x = jnp.array([68, 68, 68, 68], dtype=jnp.int32)
        
        # Re-init positions based on this anchor
        new_positions, new_counts = self._env._init_block_positions(
            misaligned_ice_x, state.level, state.ice_fine_motion_index
        )
        
        return state.replace(
            ice_x=misaligned_ice_x,
            ice_block_positions=new_positions,
            ice_block_counts=new_counts
        )
        
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: FrostbiteState, new_state: FrostbiteState) -> FrostbiteState:
        return self._apply_alignment(new_state)

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state: FrostbiteState):
        new_state = self._apply_alignment(state)
        new_obs = self._env._get_observation(new_state)
        return new_obs, new_state


class _AlignedIceMod(JaxAtariPostStepModPlugin):
    """
    Makes the ice blocks perfectly centered.
    """
    @partial(jax.jit, static_argnums=(0,))
    def _apply_alignment(self, state: FrostbiteState) -> FrostbiteState:
        level_idx = jnp.maximum(state.level - 1, 0)
        is_breathing_level = (state.level >= self._env.consts.ICE_BREATH_MIN_LEVEL) & ((state.level & 1) == 1)
        base_use_narrow = jnp.where(level_idx < 4, False, jnp.where(level_idx < 8, True, (level_idx // 2) % 2 == 1))
        use_narrow = jnp.where(is_breathing_level, False, base_use_narrow)
        
        ideal_ice_x = jnp.where(use_narrow, jnp.int32(18), jnp.int32(24))
        aligned_ice_x_arr = jnp.full((4,), ideal_ice_x, dtype=jnp.int32)
        
        new_positions, new_counts = self._env._init_block_positions(
            aligned_ice_x_arr, state.level, state.ice_fine_motion_index
        )
        
        return state.replace(
            ice_x=aligned_ice_x_arr,
            ice_block_positions=new_positions,
            ice_block_counts=new_counts
        )
        
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: FrostbiteState, new_state: FrostbiteState) -> FrostbiteState:
        return self._apply_alignment(new_state)

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state: FrostbiteState):
        new_state = self._apply_alignment(state)
        new_obs = self._env._get_observation(new_state)
        return new_obs, new_state
