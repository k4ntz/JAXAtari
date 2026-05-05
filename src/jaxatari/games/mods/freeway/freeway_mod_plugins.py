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
        
        return new_state.replace(cars=new_cars)


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
        return new_state.replace(cars=prev_state.cars)

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
        modified_state = state.replace(cars=centered_cars)
        return obs, modified_state


import os
from jaxatari.rendering.jax_rendering_utils import JaxRenderingUtils, RendererConfig, get_base_sprite_dir

# Initialize utilities
_jr = JaxRenderingUtils(RendererConfig())
_bike_path = os.path.join(get_base_sprite_dir(), "freeway", "bike.npy")
_bike_array = _jr.loadFrame(_bike_path)

# Define distinct color pairs for 10 lanes (Biker, Motorbike)
_color_pairs = [
    ((255, 0, 0), (0, 0, 255)),       # Lane 0: Red / Blue
    ((0, 255, 0), (255, 255, 0)),     # Lane 1: Green / Yellow
    ((255, 0, 255), (0, 255, 255)),   # Lane 2: Magenta / Cyan
    ((255, 128, 0), (128, 0, 255)),   # Lane 3: Orange / Purple
    ((255, 255, 255), (0, 0, 0)),     # Lane 4: White / Black
    ((0, 0, 255), (255, 0, 0)),       # Lane 5: Blue / Red
    ((255, 255, 0), (0, 255, 0)),     # Lane 6: Yellow / Green
    ((0, 255, 255), (255, 0, 255)),   # Lane 7: Cyan / Magenta
    ((128, 0, 255), (255, 128, 0)),   # Lane 8: Purple / Orange
    ((0, 0, 0), (255, 255, 255)),     # Lane 9: Black / White
]

_recolored_bikes = []
for _biker_color, _motorbike_color in _color_pairs:
    _rule = [
        {'source': (80, 184, 57), 'target': _biker_color},
        {'source': (32, 167, 32), 'target': _biker_color},
        {'source': (234, 61, 49), 'target': _motorbike_color},
        {'source': (255, 32, 32), 'target': _motorbike_color}
    ]
    _recolored_bikes.append(_jr.perform_recoloring(_bike_array, _rule))


class FrogMod(JaxAtariInternalModPlugin):
    """Replaces the player sprites with frog sprites."""
    asset_overrides = {
        "player": {
            'name': 'player', 'type': 'group',
            'files': ['frog_hit.npy', 'frog_walk.npy', 'frog_idle.npy']
        }
    }


class BikesMod(JaxAtariInternalModPlugin):
    """Replaces all cars with uniquely colored bike sprites."""
    asset_overrides = {
        'car_dark_red': {'name': 'car_dark_red', 'type': 'procedural', 'data': _recolored_bikes[0]},
        'car_light_green': {'name': 'car_light_green', 'type': 'procedural', 'data': _recolored_bikes[1]},
        'car_dark_green': {'name': 'car_dark_green', 'type': 'procedural', 'data': _recolored_bikes[2]},
        'car_light_red': {'name': 'car_light_red', 'type': 'procedural', 'data': _recolored_bikes[3]},
        'car_blue': {'name': 'car_blue', 'type': 'procedural', 'data': _recolored_bikes[4]},
        'car_brown': {'name': 'car_brown', 'type': 'procedural', 'data': _recolored_bikes[5]},
        'car_light_blue': {'name': 'car_light_blue', 'type': 'procedural', 'data': _recolored_bikes[6]},
        'car_red': {'name': 'car_red', 'type': 'procedural', 'data': _recolored_bikes[7]},
        'car_green': {'name': 'car_green', 'type': 'procedural', 'data': _recolored_bikes[8]},
        'car_yellow': {'name': 'car_yellow', 'type': 'procedural', 'data': _recolored_bikes[9]},
    }


_bg_path = os.path.join(get_base_sprite_dir(), "freeway", "background.npy")
_bg_array = _jr.loadFrame(_bg_path)

_lane_color_rule = [
    {'source': (214, 214, 214), 'target': (0, 0, 0)},       # Lane separation black
    {'source': (252, 252, 84), 'target': (255, 0, 0)}       # Double lane separation red
]
_recolored_bg = _jr.perform_recoloring(_bg_array, _lane_color_rule)

class NewLaneColorsMod(JaxAtariInternalModPlugin):
    """Makes the lane separation black and the double lane separation red."""
    asset_overrides = {
        'background': {
            'name': 'background',
            'type': 'background',
            'data': _recolored_bg
        }
    }

_score_paths = [os.path.join(get_base_sprite_dir(), "freeway", f"score_{i}.npy") for i in range(10)]
_score_array = _jr._load_and_pad_digits_from_paths(_score_paths)
_green_score_rule = [{'source': (228, 111, 111), 'target': (0, 255, 0)}]
_recolored_score = _jr.perform_recoloring(_score_array, _green_score_rule)

class GreenScoreMod(JaxAtariInternalModPlugin):
    """Makes the score digits green."""
    asset_overrides = {
        'score_digits': {
            'name': 'score_digits',
            'type': 'digits',
            'data': _recolored_score
        }
    }

class TooClosePenaltyMod(JaxAtariInternalModPlugin):
    """Provides a -1 reward if the chicken is too close to a car on the same line."""
    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: FreewayState, state: FreewayState):
        base_reward = state.score - previous_state.score

        chicken_x = self._env.consts.chicken_x
        chicken_y_top = state.chicken_y - self._env.consts.chicken_height
        chicken_y_bottom = state.chicken_y

        cars_x = state.cars[:, 0]
        cars_y_top = state.cars[:, 1] - self._env.consts.car_height
        cars_y_bottom = state.cars[:, 1]

        y_overlap = jnp.logical_and(
            chicken_y_top < cars_y_bottom,
            chicken_y_bottom > cars_y_top
        )

        x_close = jnp.abs(chicken_x - cars_x) < 10

        too_close_any = jnp.any(jnp.logical_and(y_overlap, x_close))
        penalty = jnp.where(too_close_any, 1.0, 0.0)

        return base_reward.astype(jnp.float32) - penalty
