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


def get_human_action() -> list:
    """
    Records if multiple keys are being pressed and returns the corresponding actions.

    Returns:
        actions: A list of actions taken by the player (e.g., LEFT, RIGHT, UP, DOWN, FIRE, etc.).
    """
    keys = pygame.key.get_pressed()
    actions = []

    # Movement keys
    if keys[pygame.K_a]:
        actions.append("LEFT")
    if keys[pygame.K_d]:
        actions.append("RIGHT")
    if keys[pygame.K_w]:
        actions.append("UP")
    if keys[pygame.K_s]:
        actions.append("DOWN")

    # Firing keys with diagonal combinations
    if keys[pygame.K_SPACE]:
        if keys[pygame.K_a] and keys[pygame.K_w]:
            actions.append("UPLEFTFIRE")
        elif keys[pygame.K_d] and keys[pygame.K_w]:
            actions.append("UPRIGHTFIRE")
        elif keys[pygame.K_a] and keys[pygame.K_s]:
            actions.append("DOWNLEFTFIRE")
        elif keys[pygame.K_d] and keys[pygame.K_s]:
            actions.append("DOWNRIGHTFIRE")
        elif keys[pygame.K_a]:
            actions.append("LEFTFIRE")
        elif keys[pygame.K_d]:
            actions.append("RIGHTFIRE")
        elif keys[pygame.K_w]:
            actions.append("UPFIRE")
        elif keys[pygame.K_s]:
            actions.append("DOWNFIRE")
        else:
            actions.append("FIRE")

    # If no keys are pressed, return NOOP
    if not actions:
        actions.append("NOOP")

    return actions
    


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



class WordZapperRenderer(AtraJaxisRenderer):
    def __init__(self, screen):
        """
        Initialize the renderer with all necessary rectangles and colors.
        """
        super().__init__()  # No arguments passed to AtraJaxisRenderer's __init__

        # Store the screen (Pygame surface)
        self.screen = screen

        # Define colors
        self.background_color = (0, 0, 0)  # Black
        self.spaceship_color = (255, 0, 0)  # Red
        self.title_color = (0, 0, 255)  # Blue
        self.text_color = (255, 255, 255)  # White

        # Define rectangles
        self.wordzapper_rect = jnp.array([0, 0, 800, 100])  # Title bar
        self.spaceship_rect = jnp.array([111, 365, 50, 30])  # Spaceship

    def render(self, spaceship_rect, current_time):
        """
        Render all game elements on the screen.
        """
        # Fill the screen with the background color
        self.screen.fill(self.background_color)  # Use Pygame's fill method

        # Render the title bar
        pygame.draw.rect(self.screen, self.title_color, self.wordzapper_rect)

        # Render the spaceship
        pygame.draw.rect(self.screen, self.spaceship_color, spaceship_rect)

        # Render the time
        font = pygame.font.SysFont(None, 36)
        time_surf = font.render(f"TIME: {current_time}", True, self.text_color)
        self.screen.blit(time_surf, (400, 50))

        # Update the display
        pygame.display.update()  # Use Pygame's display update method


if __name__ == "__main__":
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Word Zapper")
    clock = pygame.time.Clock()

    # Initialize the renderer
    renderer = WordZapperRenderer(screen)

    # Define the spaceship rectangle
    spaceship_rect = pygame.Rect(111, 365, 50, 30)  # Use Pygame Rect for rendering

    # Game loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get player actions
        actions = get_human_action()
        for action in actions:
            if action == "LEFT":
                spaceship_rect.x -= 5
            elif action == "RIGHT":
                spaceship_rect.x += 5
            elif action == "UP":
                spaceship_rect.y -= 5
            elif action == "DOWN":
                spaceship_rect.y += 5
            elif action == "FIRE":
                print("FIRE")
            elif action == "UPFIRE":
                print("UPFIRE")
            elif action == "DOWNFIRE":
                print("DOWNFIRE")
            elif action == "LEFTFIRE":
                print("LEFTFIRE")
            elif action == "RIGHTFIRE":
                print("RIGHTFIRE")
            elif action == "UPLEFTFIRE":
                spaceship_rect.x -= 2.5
                spaceship_rect.y -= 2.5
                print("UPLEFTFIRE")
            elif action == "UPRIGHTFIRE":
                spaceship_rect.x += 2.5
                spaceship_rect.y -= 2.5
                print("UPRIGHTFIRE")
            elif action == "DOWNLEFTFIRE":
                spaceship_rect.x -= 2.5
                spaceship_rect.y += 2.5
                print("DOWNLEFTFIRE")
            elif action == "DOWNRIGHTFIRE":
                spaceship_rect.x += 2.5
                spaceship_rect.y += 2.5
                print("DOWNRIGHTFIRE")

        # Prevent the spaceship from going out of bounds
        if spaceship_rect.x < 0:
            spaceship_rect.x = 0
        if spaceship_rect.x > WINDOW_WIDTH - spaceship_rect.width:
            spaceship_rect.x = WINDOW_WIDTH - spaceship_rect.width
        if spaceship_rect.y < 0:
            spaceship_rect.y = 0
        if spaceship_rect.y > WINDOW_HEIGHT - spaceship_rect.height:
            spaceship_rect.y = WINDOW_HEIGHT - spaceship_rect.height

        # Calculate the current time
        current_time = 90 - pygame.time.get_ticks() // 1000

        # Render the game
        renderer.render(spaceship_rect, current_time)

        # Control the frame rate
        clock.tick(60)

    pygame.quit()