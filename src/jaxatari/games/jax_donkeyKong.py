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


WIDTH = 160
HEIGHT = 210

# Pygame window dimensions
WINDOW_WIDTH = 160 * 3
WINDOW_HEIGHT = 210 * 3

# Donkey Kong position
DONKEYKONG_X = 33
DONKEYKONG_Y = 14

# Girlfriend position
GIRLFRIEND_X = 62
GIRLFRIEND_Y = 17

# Life Bar positions
LIFE_BAR_1_X = 116
LIFE_BAR_2_X = 124
LIFE_BAR_Y = 23



def load_sprites():
    """Load all sprites required for Pong rendering."""
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Load sprites
    bg = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/donkeyKong_background.npy"), transpose=True)

    donkeyKong_pose_1 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/donkeyKong1.npy"), transpose=True)
    donkeyKong_pose_2 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/donkeyKong2.npy"), transpose=True)

    girlfriend = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/girlfriend.npy"), transpose=True)

    life_bar = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/life_bar.npy"), transpose=True)

    mario_standing_right = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/mario_standing_right.npy"), transpose=True)
    mario_standing_left = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/mario_standing_left.npy"), transpose=True)
    mario_jumping_right = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/mario_jumping_right.npy"), transpose=True)
    mario_jumping_left = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/mario_jumping_left.npy"), transpose=True)

    # Convert all sprites to the expected format (add frame dimension)
    SPRITE_BG = jnp.expand_dims(bg, axis=0)
    SPRITES_DONKEYKONG = jnp.stack([donkeyKong_pose_1, donkeyKong_pose_2], axis=0)
    SPRITE_GIRLFRIEND = jnp.expand_dims(girlfriend, axis=0)
    SPRITE_LIFEBAR = jnp.expand_dims(life_bar, axis=0)
    SPRITES_MARIO = jnp.stack([mario_standing_right, mario_standing_left, mario_jumping_right], axis=0)

    SPRITES_BARREL = aj.load_and_pad_digits(
        os.path.join(MODULE_DIR, "sprites/donkeyKong/barrel{}.npy"),
        num_chars=3,
    )

    # SPRITE_PLAYER = jnp.expand_dims(player, axis=0)
    # SPRITE_ENEMY = jnp.expand_dims(enemy, axis=0)
    # SPRITE_BALL = jnp.expand_dims(ball, axis=0)

    return (
        SPRITE_BG,
        SPRITES_DONKEYKONG,
        SPRITE_GIRLFRIEND,
        SPRITES_BARREL,
        SPRITE_LIFEBAR,
        SPRITES_MARIO,
        # SPRITE_PLAYER,
        # SPRITE_ENEMY,
        # SPRITE_BALL,
        # PLAYER_DIGIT_SPRITES,
        # ENEMY_DIGIT_SPRITES
    )


class DonkeyKongRenderer(AtraJaxisRenderer):
    """JAX-based Pong game renderer, optimized with JIT compilation."""

    def __init__(self):
        (
            self.SPRITE_BG,
            self.SPRITES_DONKEYKONG,
            self.SPRITE_GIRLFRIEND,
            self.SPRITES_BARREL,
            self.SPRITE_LIFEBAR,
            self.SPRITES_MARIO,
            # self.SPRITE_PLAYER,
            # self.SPRITE_ENEMY,
            # self.SPRITE_BALL,
            # self.PLAYER_DIGIT_SPRITES,
            # self.ENEMY_DIGIT_SPRITES,
        ) = load_sprites()

    
    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        """
        Renders the current game state using JAX operations.

        Args:
            state: A DonkeyKongState object containing the current game state.

        Returns:
            A JAX array representing the rendered frame.
        """
        raster = jnp.zeros((WIDTH, HEIGHT, 3))

        # Background raster
        frame_bg = aj.get_sprite_frame(self.SPRITE_BG, 0)
        raster = aj.render_at(raster, 0, 0, frame_bg)

        # DonkeyKong
        frame_donkeyKong = aj.get_sprite_frame(self.SPRITES_DONKEYKONG, 0)
        raster = aj.render_at(raster, DONKEYKONG_X, DONKEYKONG_Y, frame_donkeyKong)

        # Girlfriend
        frame_girlfriend = aj.get_sprite_frame(self.SPRITE_GIRLFRIEND, 0)
        raster = aj.render_at(raster, GIRLFRIEND_X, GIRLFRIEND_Y, frame_girlfriend)

        # Life Bars - depending if lifes are still given 
        frame_life_bar = aj.get_sprite_frame(self.SPRITE_LIFEBAR, 0)
        raster = aj.render_at(raster, LIFE_BAR_1_X, LIFE_BAR_Y, frame_life_bar)
        raster = aj.render_at(raster, LIFE_BAR_2_X, LIFE_BAR_Y, frame_life_bar)


        # Barrels - example for now
        frame_barrel = aj.get_sprite_frame(self.SPRITES_BARREL, 0)
        raster = aj.render_at(raster, 5, 5, frame_barrel)
        frame_barrel = aj.get_sprite_frame(self.SPRITES_BARREL, 1)
        raster = aj.render_at(raster, 5, 15, frame_barrel)
        frame_barrel = aj.get_sprite_frame(self.SPRITES_BARREL, 2)
        raster = aj.render_at(raster, 5, 25, frame_barrel)

        # Mario
        frame_mario = aj.get_sprite_frame(self.SPRITES_MARIO, 0)
        raster = aj.render_at(raster, 5, 35, frame_mario)
        frame_mario = aj.get_sprite_frame(self.SPRITES_MARIO, 1)
        raster = aj.render_at(raster, 15, 35, frame_mario)
        frame_mario = aj.get_sprite_frame(self.SPRITES_MARIO, 2)
        raster = aj.render_at(raster, 5, 55, frame_mario)
        # frame_mario = aj.get_sprite_frame(self.SPRITES_MARIO, 3)
        # raster = aj.render_at(raster, 15, 55, frame_mario)

        return raster


if __name__ == "__main__":
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Donkey Kong Game")
    clock = pygame.time.Clock()

    # Create the JAX renderer
    renderer = DonkeyKongRenderer()




    # Game Loop
    running = True


    while running:
        # Render and Display
        raster = renderer.render(state=None)
        aj.update_pygame(screen, raster, 3, WIDTH, HEIGHT)

    pygame.quit()