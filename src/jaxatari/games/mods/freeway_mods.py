import functools
from typing import Any, Dict, Tuple, Union


import chex
import jax
import jax.numpy as jnp
from jaxatari.games.jax_freeway import FreewayState

from jaxatari.wrappers import JaxatariWrapper

class StopAllCars(JaxatariWrapper):
    """Stops all cars randomly with probability 0.4"""
    @functools.partial(jax.jit, static_argnums=(0,))
    def stop_cars(self, prev_state: FreewayState, state: FreewayState) -> FreewayState:
        key = jax.random.PRNGKey(state.time)
        chance = 0.4
        random_bool = jax.random.bernoulli(key, chance)
        
        new_cars = jax.lax.cond(
            random_bool,
            lambda x: prev_state.cars,  # Keep the previous cars' x positions if the random condition is met
            lambda x: state.cars,  # Otherwise, use the current cars' x positions
            operand=None
        )

        new_state = state._replace(
            cars=new_cars
        )

        return new_state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state: FreewayState, action: Union[int, float]) -> Tuple[chex.Array, FreewayState, float, bool, Dict[Any, Any]]:
        new_obs, new_state, reward, done, info = self._env.step(state, action)
        new_state = self.stop_cars(state, new_state)

        return new_obs, new_state, reward, done, info

class AlwaysStopAllCars(JaxatariWrapper):
    """Stops all cars after spawning"""
    @functools.partial(jax.jit, static_argnums=(0,))
    def stop_cars(self, prev_state: FreewayState, state: FreewayState) -> FreewayState:
        new_state = state._replace(
            cars=prev_state.cars  # Always keep the previous cars' x positions
        )

        return new_state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state: FreewayState, action: Union[int, float]) -> Tuple[chex.Array, FreewayState, float, bool, Dict[Any, Any]]:
        new_obs, new_state, reward, done, info = self._env.step(state, action)
        new_state = self.stop_cars(state, new_state)

        return new_obs, new_state, reward, done, info

class SpeedMode(JaxatariWrapper):
    """Increase speed of all cars by a factor of 2"""

    @functools.partial(jax.jit, static_argnums=(0,))
    def speed_up(self, prev_state: FreewayState, state: FreewayState) -> FreewayState:
        # look one step in the future 
        _, future_state, _, _, _ = self._env.step(state, 0)

        new_state = state._replace(
            cars=future_state.cars  # Update the cars' x positions to be twice as fast
        )

        return new_state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state: FreewayState, action: Union[int, float]) -> Tuple[chex.Array, FreewayState, float, bool, Dict[Any, Any]]:
        new_obs, new_state, reward, done, info = self._env.step(state, action)
        new_state = self.speed_up(state, new_state)
        return new_obs, new_state, reward, done, info

        