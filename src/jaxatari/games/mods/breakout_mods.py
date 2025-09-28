import functools
from typing import Optional
import chex
import jax
import jax.numpy as jnp

from jaxatari.wrappers import JaxatariWrapper

class SpeedMode(JaxatariWrapper):
    """Increase speed to maximum at all time steps."""
    def __init__(self, env):
        super().__init__(env)
        self._env = env
        # Overrides get_ball_velocity from env
        self._env.get_ball_velocity = self.get_ball_velocity.__get__(self._env, self._env.__class__) 

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_ball_velocity(self, speed_idx, direction_idx, step_counter):
        """Returns the ball's velocity based on the speed and direction indices."""
        # Overrides the default function from the env
        direction = self._env.consts.BALL_DIRECTIONS[direction_idx]
        speed = 3
        return speed * direction[0], speed * direction[1]

class SmallPaddle(JaxatariWrapper):
    """Always use a small paddle."""
    def __init__(self, env):
        super().__init__(env)
        self._env = env
        self._env.consts.PLAYER_SIZE = (4, 4)
        self._env.consts.PLAYER_SIZE_SMALL = (4, 4)

class BigPaddle(JaxatariWrapper):
    """Always use a bigger paddle."""
    def __init__(self, env):
        super().__init__(env)
        self._env = env
        self._env.consts.PLAYER_SIZE = (40, 4)
        self._env.consts.PLAYER_SIZE_SMALL = (40, 4)

class BallDrift(JaxatariWrapper):
    """Consistently drift the ball to the right."""
    def __init__(self, env, drift_buffer: int = 4, direction: int = 1):
        super().__init__(env)
        self._drift_buffer = drift_buffer
        self._direction = direction
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        # drift the ball first, then step the environment to make use of existing reset and bounding logic
        # drift one step every drift_buffer steps
        state = jax.lax.cond(
            state.step_counter % self._drift_buffer == 0,
            lambda: state._replace(
                ball_x = state.ball_x + self._direction,
            ),
            lambda: state,
        )
        new_obs, new_state, reward, done, info = self._env.step(state, action)
        return new_obs, new_state, reward, done, info
    

class BallGravity(JaxatariWrapper):
    """
    Pulls the ball down.
    The ball will be pulled down by 1 every gravity_buffer steps. The direction is 1 for down, -1 for up.
    """
    def __init__(self, env, gravity_buffer: int = 4, direction: int = -1):
        super().__init__(env)
        self._gravity_buffer = gravity_buffer
        self._direction = direction
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        # pull the ball down first, then step the environment to make use of existing reset and bounding logic
        # pull the ball down by 1 every gravity_buffer steps
        state = jax.lax.cond(
            state.step_counter % self._gravity_buffer == 0,
            lambda: state._replace(
                ball_y = state.ball_y + self._direction,
            ),
            lambda: state,
        )
        new_obs, new_state, reward, done, info = self._env.step(state, action)
        return new_obs, new_state, reward, done, info

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

class BallColor(JaxatariWrapper):
    """Changes the balls color to a set color"""
    def __init__(self, env, color: tuple[int, int, int] = (255, 255, 0)):
        super().__init__(env)
        # get the current ball sprite and change all colored pixels to the given color
        self._env.renderer.SPRITE_BALL = recolor_4d_sprite(self._env.renderer.SPRITE_BALL, jnp.array(color, dtype=jnp.uint8))

class BlockColor(JaxatariWrapper):
    """Changes the blocks color to a set color"""
    def __init__(self, env, color: tuple[int, int, int] = (255, 255, 0)):
        super().__init__(env)
        
        num_rows = self._env.renderer.consts.NUM_ROWS
        single_color_array = jnp.array(color, dtype=jnp.uint8)

        # Create the full palette by repeating the single color for each row.
        new_palette = jnp.tile(single_color_array, (num_rows, 1))
        
        # Overwrite the renderer's block colors.
        self._env.renderer.BLOCK_COLORS = new_palette


class PlayerColor(JaxatariWrapper):
    """Changes the player color to a set color"""
    def __init__(self, env, color: tuple[int, int, int] = (255, 255, 0)):
        super().__init__(env)
        # get the current player sprite and change all colored pixels to the given color
        self._env.renderer.SPRITE_PLAYER = recolor_4d_sprite(self._env.renderer.SPRITE_PLAYER, jnp.array(color, dtype=jnp.uint8))