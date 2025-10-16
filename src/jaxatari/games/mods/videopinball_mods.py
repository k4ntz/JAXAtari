import functools
from typing import Any, Dict, Tuple, Union


import chex
import jax
import jax.numpy as jnp
from jaxatari.games.jax_videopinball import VideoPinballState

from jaxatari.wrappers import JaxatariWrapper


class NeverActivateTiltMode(JaxatariWrapper):
    """Prevent entering tilt mode by resetting the tilt counter."""

    @functools.partial(jax.jit, static_argnums=(0,))
    def prevent_tilt(self, state: VideoPinballState) -> VideoPinballState:
        new_state = state._replace(
            tilt_mode_active=jnp.array(False, dtype=jnp.bool_),
        )
        return new_state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self, state: VideoPinballState, action: Union[int, float]
    ) -> Tuple[chex.Array, VideoPinballState, float, bool, Dict[Any, Any]]:
        new_obs, new_state, reward, done, info = self._env.step(state, action)
        new_state = self.prevent_tilt(new_state)

        return new_obs, new_state, reward, done, info


class NoScoringCooldown(JaxatariWrapper):
    """Remove the cooldown period after hitting a target."""

    @functools.partial(jax.jit, static_argnums=(0,))
    def remove_cooldown(self, state: VideoPinballState) -> VideoPinballState:
        new_state = state._replace(
            target_cooldown=jnp.array(-1, dtype=jnp.int32),
            special_target_cooldown=jnp.array(-1, dtype=jnp.int32),
            active_targets=jnp.array([True, True, True, True], dtype=jnp.bool_),
            rollover_enabled=jnp.array(True, dtype=jnp.bool_),
        )

        return new_state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self, state: VideoPinballState, action: Union[int, float]
    ) -> Tuple[chex.Array, VideoPinballState, float, bool, Dict[Any, Any]]:
        new_obs, new_state, reward, done, info = self._env.step(state, action)
        new_state = self.remove_cooldown(new_state)

        return new_obs, new_state, reward, done, info
