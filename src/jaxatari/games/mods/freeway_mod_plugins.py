import jax
import jax.numpy as jnp
from functools import partial

from jaxatari.modification import JaxAtariPostStepModPlugin, JaxAtariInternalModPlugin
from jaxatari.games.jax_freeway import FreewayState


class StopAllCarsMod(JaxAtariPostStepModPlugin):
    """Stops all cars randomly with probability 0.4"""
    
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: FreewayState, new_state: FreewayState) -> FreewayState:
        """
        This function is called by the wrapper *after*
        the main step is complete.
        Access the environment via self._env (set by JaxAtariModWrapper).
        """
        key = jax.random.PRNGKey(new_state.time)
        chance = 0.4
        random_bool = jax.random.bernoulli(key, chance)
        
        new_cars = jax.lax.cond(
            random_bool,
            lambda _: prev_state.cars,  # Keep the previous cars' x positions if the random condition is met
            lambda _: new_state.cars,  # Otherwise, use the current cars' x positions
            operand=None
        )
        
        return new_state._replace(cars=new_cars)


class AlwaysStopAllCarsMod(JaxAtariPostStepModPlugin):
    """Stops all cars after spawning"""
    
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: FreewayState, new_state: FreewayState) -> FreewayState:
        """
        This function is called by the wrapper *after*
        the main step is complete.
        Access the environment via self._env (set by JaxAtariModWrapper).
        """
        # Always keep the previous cars' x positions
        return new_state._replace(cars=prev_state.cars)


class SpeedModeMod(JaxAtariInternalModPlugin):
    """
    Doubles the speed of all cars by making them "teleport" twice as far
    on each update, via multiplying their update step by 2.
    This is done by overriding the `car_update` constant.
    """
    @staticmethod
    def double_update_arr(arr):
        arr = jnp.array(arr)
        # Multiply each timing by 2
        return [int(x * 2) for x in arr]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        base_update = self._env.consts.car_update
        update_list = list(base_update)
        self.constants_overrides = {
            "car_update": SpeedModeMod.double_update_arr(update_list)
        }


class BlackCarsMod(JaxAtariInternalModPlugin):
    """Makes all cars black by overriding CAR_COLORS constant"""
    
    # Override CAR_COLORS to make all cars black
    # Using (0, 0, 0) for pure black. None means use original color.
    constants_overrides = {
        "CAR_COLORS": [
            (0, 0, 0),  # Lane 0 - black
            (0, 0, 0),  # Lane 1 - black
            (0, 0, 0),  # Lane 2 - black
            (0, 0, 0),  # Lane 3 - black
            (0, 0, 0),  # Lane 4 - black
            (0, 0, 0),  # Lane 5 - black
            (0, 0, 0),  # Lane 6 - black
            (0, 0, 0),  # Lane 7 - black
            (0, 0, 0),  # Lane 8 - black
            (0, 0, 0),  # Lane 9 - black
        ]
    }


class CenterCarsOnResetMod(JaxAtariPostStepModPlugin):
    """
    Positions all cars in the center of the screen when the environment resets.
    """
    
    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state: FreewayState):
        """
        Called after reset to modify initial state.
        Positions all cars in the center of the screen horizontally.
        """
        # Calculate center x position
        center_x = self._env.consts.screen_width // 2
        
        # Create new cars array with all cars at center x position
        # Keep the original y positions (lane positions)
        centered_cars = state.cars.at[:, 0].set(center_x)
        
        # Return modified observation and state
        modified_state = state._replace(cars=centered_cars)
        return obs, modified_state
