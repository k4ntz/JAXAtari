import chex
import jax
import jax.numpy as jnp
from functools import partial

from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin


class AlwaysCenteredMod(JaxAtariPostStepModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        pause = jnp.where(jnp.equal(prev_state.pause_timer, self._env.consts.DEATH_PAUSE_FRAMES + 2), self._env.consts.DEATH_PAUSE_FRAMES + 1, new_state.pause_timer)
        return new_state.replace(local_player_offset=jnp.array(0), pause_timer=pause)