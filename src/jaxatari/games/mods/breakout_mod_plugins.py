import jax
import jax.numpy as jnp
from functools import partial

from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin
from jaxatari.games.jax_breakout import BreakoutState


class SpeedModeMod(JaxAtariInternalModPlugin):
    """Increase speed to maximum at all time steps."""
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_ball_velocity(self, speed_idx, direction_idx, step_counter):
        """Returns the ball's velocity based on the speed and direction indices."""
        # Override to always return maximum speed
        direction = self._env.consts.BALL_DIRECTIONS[direction_idx]
        speed = 3
        return speed * direction[0], speed * direction[1]


class SmallPaddleMod(JaxAtariInternalModPlugin):
    """Always use a small paddle."""
    
    constants_overrides = {
        "PLAYER_SIZE": (4, 4),
        "PLAYER_SIZE_SMALL": (4, 4),
    }


class BigPaddleMod(JaxAtariInternalModPlugin):
    """Always use a bigger paddle."""
    
    constants_overrides = {
        "PLAYER_SIZE": (40, 4),
        "PLAYER_SIZE_SMALL": (40, 4),
    }


class BallDriftMod(JaxAtariPostStepModPlugin):
    """Consistently drift the ball to the right."""
    
    # Default: drift every 4 steps, direction 1 (right)
    _drift_buffer = 4
    _direction = 1
    
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: BreakoutState, new_state: BreakoutState) -> BreakoutState:
        """
        This function is called by the wrapper *after*
        the main step is complete.
        """
        # Drift the ball one step every drift_buffer steps
        # This affects the next step
        return jax.lax.cond(
            new_state.step_counter % self._drift_buffer == 0,
            lambda s: s._replace(ball_x=s.ball_x + self._direction),
            lambda s: s,
            operand=new_state
        )


class BallGravityMod(JaxAtariPostStepModPlugin):
    """
    Pulls the ball down.
    The ball will be pulled down by 1 every gravity_buffer steps. The direction is 1 for down, -1 for up.
    """
    
    # Default: gravity every 4 steps, direction -1 (down, but negative because y increases downward)
    _gravity_buffer = 4
    _direction = -1
    
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: BreakoutState, new_state: BreakoutState) -> BreakoutState:
        """
        This function is called by the wrapper *after*
        the main step is complete.
        """
        # Pull the ball down by 1 every gravity_buffer steps
        # This affects the next step
        return jax.lax.cond(
            new_state.step_counter % self._gravity_buffer == 0,
            lambda s: s._replace(ball_y=s.ball_y + self._direction),
            lambda s: s,
            operand=new_state
        )


def recolor_4d_sprite(sprite_array: jnp.ndarray, new_rgb_color: jnp.ndarray) -> jnp.ndarray:
    """
    Recolors the non-transparent pixels of a 4D RGBA sprite array.

    Args:
        sprite_array: The input array with shape (Frame, H, W, 4).
        new_rgb_color: A 3-element array for the new RGB color.

    Returns:
        A new array with the sprite recolored.
    """
    # Create a mask from the alpha channel (the 4th channel)
    is_visible = sprite_array[:, :, :, 3] > 0

    # Use the mask to set the RGB values (:3) of visible pixels
    recolored_array = sprite_array.at[is_visible, :3].set(new_rgb_color)
    
    return recolored_array


class BallColorMod(JaxAtariInternalModPlugin):
    """Changes the balls color to a set color"""
    
    # Default color is yellow, but this can be overridden via constants
    constants_overrides = {
        "BALL_COLOR": (255, 255, 0),  # Yellow by default
    }
    


class BlockColorMod(JaxAtariInternalModPlugin):
    """Changes the blocks color to a set color"""
    
    # Default color is yellow
    constants_overrides = {
        "BLOCK_COLORS": [
            (255, 255, 0),  # All rows yellow
            (255, 255, 0),
            (255, 255, 0),
            (255, 255, 0),
            (255, 255, 0),
            (255, 255, 0),
        ]
    }


class PlayerColorMod(JaxAtariInternalModPlugin):
    """Changes the player color to a set color"""
    
    # Default color is yellow
    constants_overrides = {
        "PLAYER_COLOR": (255, 255, 0),  # Yellow by default
    }

