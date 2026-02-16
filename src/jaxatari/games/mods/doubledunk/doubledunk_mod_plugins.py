import jax
import jax.numpy as jnp
from functools import partial
from jaxatari.modification import JaxAtariInternalModPlugin
from jaxatari.games.jax_doubledunk import DunkGameState

class TimerMod(JaxAtariInternalModPlugin):
    """
    Ends the game after 1 minute (3600 frames at 60fps)
    instead of the default score limit (24).
    """
    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: DunkGameState) -> bool:
        return state.step_counter >= 3600