import jax
import jax.numpy as jnp
from functools import partial

from jaxatari.modification import JaxAtariInternalModPlugin
from jaxatari.games.jax_frostbite import FrostbiteState

class NoEnemiesMod(JaxAtariInternalModPlugin):
    """
    Internal mod to disable all enemies (polar bear, geese, crabs, clams) 
    and only keep the fishes.
    """
    
    @partial(jax.jit, static_argnums=(0,))
    def _update_polar_grizzly(self, state: FrostbiteState):
        """Disable polar grizzly bear."""
        # Return state with polar_grizzly_active set to 0. 
        # This prevents the bear from rendering or triggering collisions.
        return state.replace(polar_grizzly_active=jnp.array(0, dtype=jnp.int32))

    @partial(jax.jit, static_argnums=(0,))
    def _spawn_obstacles_vec(self, state: FrostbiteState, spawn_mask: jnp.ndarray) -> FrostbiteState:
        """Override to only spawn fishes."""
        # Call original function from the environment class to progress RNG
        # and handle the core spawning logic (x positions, patterns, etc)
        from jaxatari.games.jax_frostbite import JaxFrostbite
        new_state = JaxFrostbite._spawn_obstacles_vec(self._env, state, spawn_mask)
        
        is_spawned = spawn_mask.astype(jnp.bool_)
        
        # ID_FISH is 1 in Frostbite
        forced_type = jnp.int32(1)
        
        # Override the spawned obstacle type
        new_type = jnp.where(is_spawned, forced_type, new_state.obstacle_types)
        
        # Update fish alive mask to be consistent with the forced type
        max_copies = new_state.obstacle_max_copies
        initial_fish_mask = (jnp.int32(1) << max_copies) - jnp.int32(1)
        
        new_fish_mask = jnp.where(is_spawned, initial_fish_mask, new_state.fish_alive_mask)
        
        return new_state.replace(
            obstacle_types=new_type,
            fish_alive_mask=new_fish_mask
        )