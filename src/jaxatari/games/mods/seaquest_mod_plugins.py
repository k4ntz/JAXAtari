import jax
import jax.numpy as jnp
from functools import partial

from jaxatari.modification import JaxAtariPostStepModPlugin
from jaxatari.games.jax_seaquest import SeaquestState


class DisableEnemiesMod(JaxAtariPostStepModPlugin):
    """Disable enemies in the environment."""
    
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        """
        This function is called by the wrapper *after*
        the main step is complete.
        Access the environment via self._env (set by JaxAtariModWrapper).
        """
        # Zero out all enemy positions
        return new_state._replace(
            shark_positions=jnp.zeros_like(new_state.shark_positions),
            sub_positions=jnp.zeros_like(new_state.sub_positions),
            enemy_missile_positions=jnp.zeros_like(new_state.enemy_missile_positions),
            surface_sub_position=jnp.zeros_like(new_state.surface_sub_position)
        )

