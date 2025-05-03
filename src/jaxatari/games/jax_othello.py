import os
from functools import partial
from typing import NamedTuple, Tuple
import jax.lax
import jax.numpy as jnp
import chex
import pygame
from gymnax.environments import spaces

from jaxatari.rendering import atraJaxis as aj
from jaxatari.environment import JaxEnvironment


# Game Environment
WIDTH = 160
HEIGHT = 210

# Pygame window dimensions
WINDOW_WIDTH = 160 * 3
WINDOW_HEIGHT = 210 * 3


def load_sprites():
    """Load all sprites required for Pong rendering."""
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Load sprites
    player = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/othello/player_white_disc.npy"), transpose=True)
    enemy = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/othello/enemy_black_disc.npy"), transpose=True)

    bg = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/othello/othello_background.npy"), transpose=True)

    # TODO: get a correctly sized background image / resize the saved image..
    #bg = jax.image.resize(bg, (WIDTH, HEIGHT, 4), method='bicubic')

    # Convert all sprites to the expected format (add frame dimension)
    SPRITE_BG = jnp.expand_dims(bg, axis=0)
    SPRITE_PLAYER = jnp.expand_dims(player, axis=0)
    SPRITE_ENEMY = jnp.expand_dims(enemy, axis=0)
    # SPRITE_BALL = jnp.expand_dims(ball, axis=0)

    # Load digits for scores
    # PLAYER_DIGIT_SPRITES = aj.load_and_pad_digits(
    #     os.path.join(MODULE_DIR, "sprites/pong/player_score_{}.npy"),
    #     num_chars=10,
    # )
    # ENEMY_DIGIT_SPRITES = aj.load_and_pad_digits(
    #     os.path.join(MODULE_DIR, "sprites/pong/enemy_score_{}.npy"),
    #     num_chars=10,
    # )

    return (
        SPRITE_BG,
        SPRITE_PLAYER,
        SPRITE_ENEMY,
        # SPRITE_BALL,
        # PLAYER_DIGIT_SPRITES,
        # ENEMY_DIGIT_SPRITES
    )


class Renderer_AtraJaxisOthello:

    def __init__(self):
        (
            self.SPRITE_BG,
            self.SPRITE_PLAYER,
            self.SPRITE_ENEMY,
            # self.SPRITE_BALL,
            # self.PLAYER_DIGIT_SPRITES,
            # self.ENEMY_DIGIT_SPRITES,
        ) = load_sprites()

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):

        # Create empty raster with CORRECT orientation for atraJaxis framework
        # Note: For pygame, the raster is expected to be (width, height, channels)
        # where width corresponds to the horizontal dimension of the screen
        raster = jnp.zeros((WIDTH, HEIGHT, 3))

        # Render Background - (0, 0) is top-left corner
        frame_bg = aj.get_sprite_frame(self.SPRITE_BG, 0)
        raster = aj.render_at(raster, 0, 0, frame_bg)

        frame_player = aj.get_sprite_frame(self.SPRITE_PLAYER, 0)
        raster = aj.render_at(raster, 0, 0, frame_player)

        frame_enemy = aj.get_sprite_frame(self.SPRITE_ENEMY, 0)
        raster = aj.render_at(raster, 50, 50, frame_enemy)

        return raster


if __name__ == "__main__":
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Othello Game")

    # Create the JAX renderer
    renderer = Renderer_AtraJaxisOthello()


    # Game Loop
    running = True

    while running:


        # Render and display
        raster = renderer.render(None)
        aj.update_pygame(screen, raster, 3, WIDTH, HEIGHT)

    pygame.quit()