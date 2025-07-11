import functools
from typing import Any, Dict, Tuple, Union


import chex
import jax
import jax.numpy as jnp
from jaxatari.games.jax_seaquest import SeaquestState

from jaxatari.wrappers import JaxatariWrapper

class DisableEnemiesWrapper(JaxatariWrapper):
    """Disable enemies in the environment."""
    @functools.partial(jax.jit, static_argnums=(0,))
    def disable_enemies(self, state: SeaquestState) -> SeaquestState:
        new_state = state._replace(
            shark_positions=jnp.zeros_like(state.shark_positions),
            sub_positions=jnp.zeros_like(state.sub_positions),
            enemy_missile_positions=jnp.zeros_like(state.enemy_missile_positions),
            surface_sub_position=jnp.zeros_like(state.surface_sub_position)
        )
        return new_state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state: SeaquestState, action: Union[int, float]) -> Tuple[chex.Array, SeaquestState, float, bool, Dict[Any, Any]]:
        new_obs, state, reward, done, info = self._env.step(state, action)
        new_state = self.disable_enemies(state)

        return new_obs, new_state, reward, done, info