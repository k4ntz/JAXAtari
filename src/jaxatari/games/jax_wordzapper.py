import os
from functools import partial
from typing import NamedTuple, Tuple
import jax.lax
import jax.numpy as jnp
import chex
import pygame
from gymnax.environments import spaces

from jaxatari.renderers import AtraJaxisRenderer
from jaxatari.rendering import atraJaxis as aj
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action

from enum import IntEnum


# TODO : remove unnecessary constants

# Constants for game environment
MAX_SPEED = 12
BALL_SPEED = jnp.array([-1, 1])  # Ball speed in x and y direction
ENEMY_STEP_SIZE = 2
WIDTH = 160
HEIGHT = 210

# Constants for ball physics
BASE_BALL_SPEED = 1
BALL_MAX_SPEED = 4  # Maximum ball speed cap

# constants for paddle speed influence
MIN_BALL_SPEED = 1

PLAYER_ACCELERATION = jnp.array([6, 3, 1, -1, 1, -1, 0, 0, 1, 0, -1, 0, 1])

BALL_START_X = jnp.array(78)
BALL_START_Y = jnp.array(115)

# Background color and object colors
BACKGROUND_COLOR = 144, 72, 17
PLAYER_COLOR = 92, 186, 92
ENEMY_COLOR = 213, 130, 74
BALL_COLOR = 236, 236, 236  # White ball
WALL_COLOR = 236, 236, 236  # White walls
SCORE_COLOR = 236, 236, 236  # White score

# Player and enemy paddle positions
PLAYER_X = 140
ENEMY_X = 16

# Object sizes (width, height)
PLAYER_SIZE = (4, 16)
BALL_SIZE = (2, 4)
ENEMY_SIZE = (4, 16)
WALL_TOP_Y = 24
WALL_TOP_HEIGHT = 10
WALL_BOTTOM_Y = 194
WALL_BOTTOM_HEIGHT = 16

# Pygame window dimensions
WINDOW_WIDTH = 160 * 3
WINDOW_HEIGHT = 210 * 3

# define the positions of the state information
# define the positions of the state information
STATE_TRANSLATOR: dict = {
    #TODO
}


def get_human_action() -> chex.Array:
    """
    Records if UP or DOWN is being pressed and returns the corresponding action.

    Returns:
        action: int, action taken by the player (LEFT, RIGHT, FIRE, LEFTFIRE, RIGHTFIRE, UP, DOWN, NOOP, UPFIRE, DOWNFIRE).
    """
    keys = pygame.key.get_pressed()
    if keys[pygame.K_a] and keys[pygame.K_SPACE]:
        return jnp.array(Action.LEFTFIRE)
    elif keys[pygame.K_d] and keys[pygame.K_SPACE]:
        return jnp.array(Action.RIGHTFIRE)
    elif keys[pygame.K_a]:
        return jnp.array(Action.LEFT)
    elif keys[pygame.K_d]:
        return jnp.array(Action.RIGHT)
    elif keys[pygame.K_SPACE]:
        return jnp.array(Action.FIRE)
    elif keys[pygame.K_w] and keys[pygame.K_SPACE]:  # W + SPACE for UPFIRE
        return jnp.array(Action.UPFIRE)
    elif keys[pygame.K_s] and keys[pygame.K_SPACE]:  # S + SPACE for DOWNFIRE
        return jnp.array(Action.DOWNFIRE)
    elif keys[pygame.K_w]:  # W key for UP
        return jnp.array(Action.UP)
    elif keys[pygame.K_s]:  # S key for DOWN
        return jnp.array(Action.DOWN)
    else:
        return jnp.array(Action.NOOP)
    


class WordZapperState(NamedTuple):
    #TODO
    pass


class EntityPosition(NamedTuple):
    #TODO : review
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray

    

class WordZapperObservation(NamedTuple):
    #TODO
    pass


class WordZapperInfo(NamedTuple):
    #TODO : review
    time: jnp.ndarray
    all_rewards: chex.Array


@jax.jit
def player_step():
    #TODO
    pass


def ball_step():
    #TODO
    pass


def enemy_step():
    #TODO
    pass


@jax.jit
def _reset_ball_after_goal():
    #TODO : give a better name 
    #TODO
    pass



def load_sprites():
    """Load all sprites required for Word Zapper rendering."""
    #TODO
    pass

class JaxWordZapper() :
    #TODO
    pass



class WordZapperRenderer() :
    #TODO
    pass




#TODO : review, remove unrelevant lines
if __name__ == "__main__":
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Word Zapper")
    clock = pygame.time.Clock()

    game = JaxWordZapper()

    # Create the JAX renderer
    renderer = WordZapperRenderer()

    # Get jitted functions
    jitted_step = jax.jit(game.step)
    jitted_reset = jax.jit(game.reset)

    obs, curr_state = jitted_reset()

    # Game loop
    running = True
    frame_by_frame = False
    frameskip = 1
    counter = 1

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    frame_by_frame = not frame_by_frame
            elif event.type == pygame.KEYDOWN or (
                    event.type == pygame.KEYUP and event.key == pygame.K_n
            ):
                if event.key == pygame.K_n and frame_by_frame:
                    if counter % frameskip == 0:
                        action = get_human_action()
                        obs, curr_state, reward, done, info = jitted_step(
                            curr_state, action
                        )

        if not frame_by_frame:
            if counter % frameskip == 0:
                action = get_human_action()
                obs, curr_state, reward, done, info = jitted_step(curr_state, action)

        # Render and display
        raster = renderer.render(curr_state)

        aj.update_pygame(screen, raster, 3, WIDTH, HEIGHT)

        counter += 1
        clock.tick(60)

    pygame.quit()