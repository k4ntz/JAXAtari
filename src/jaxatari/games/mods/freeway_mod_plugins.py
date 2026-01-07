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


class StaticCarsMod(JaxAtariPostStepModPlugin):
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

class HallOfFameMod(JaxAtariInternalModPlugin):
    """
    Spawns the cars to make a "hall of fame" formation.
    """
    constants_overrides = {
        "lane_phase_offset": [
            -101, -101, -101, -101, -101, 
            35, 35, 35, 35, 35
            ]
    }

class SlowCarsMod(JaxAtariInternalModPlugin):
    """
    Halves the speed of all cars by making them move twice as far
    on each update, via multiplying their update step by 2.
    This is done by overriding the `CAR_UPDATES` constant.
    """
    constants_overrides = {
        "CAR_UPDATES": 
        [
            -10,  # Lane 0
            -8,   # Lane 1
            -6,   # Lane 2
            -4,   # Lane 3
            -2,   # Lane 4
            2,    # Lane 5
            4,    # Lane 6
            6,    # Lane 7
            8,    # Lane 8
            10,   # Lane 9
        ]
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

class InvertSpeed(JaxAtariInternalModPlugin):
    """Inverts the speed of all cars by overriding CAR_UPDATES constant"""
    
    # Override CAR_UPDATES to invert the directions of all cars
    constants_overrides = {
        "CAR_UPDATES": [
            5,  # Lane 0
            4,  # Lane 1
            3,  # Lane 2
            2,  # Lane 3
            1,  # Lane 4
            -1,   # Lane 5
            -2,   # Lane 6
            -3,   # Lane 7
            -4,   # Lane 8
            -5,   # Lane 9
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
