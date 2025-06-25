import os
from functools import partial
import pygame
import chex
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Tuple, NamedTuple, List, Dict, Optional, Any
import pygame
import jaxatari.rendering.atraJaxis as aj
import numpy as np
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import AtraJaxisRenderer


@dataclass
class GameConfig:
    """Game configuration parameters"""

    screen_width: int = 160
    screen_height: int = 210
    shark_speeds: List[float] = None
    fish_speed: float = 1
    fish_x: List[int] = None
    fish_y: List[int] = None
    top_border: int = 30
    bottom_border: int = 180
    num_rows: int = 6
    row_spacing: int = 16
    row_borders: List[int] = None


    def __post_init__(self):
        if self.row_borders is None:
            self.row_borders = [
                self.top_border + i * self.row_spacing for i in range(self.num_rows)
            ]


class GameState(NamedTuple):
        """Game state representation"""
        player1_rod_x: chex.Array
        player1_rod_y: chex.Array
        player2_rod_x: chex.Array
        player2_rod_y: chex.Array
        score: chex.Array
        shark_x: chex.Array
        fish_x: chex.Array
        time = chex.Array

class EntityPositions(NamedTuple):
    """Positions of entities in the game"""
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray

class FishingDerbyObservation(NamedTuple):
    """Observation structure for Fishing Derby"""
    player1_rod: EntityPositions
    player2_rod: EntityPositions
    shark: EntityPositions
    fish: EntityPositions
    score: jnp.ndarray

class FishingDerbyInfo(NamedTuple):
    time: jnp.ndarray

class FishingDerby(JaxEnvironment[GameState, FishingDerbyObservation, FishingDerbyInfo]):
    def __init__(self):
        super().__init__()
        self.config = GameConfig()
        self.state = self.reset()

    def reset(self) -> Tuple[FishingDerbyObservation, GameState]:

        for row in range(self.config.num_rows):
            row_y = self.config.fish_y[row]

        state = GameState(
            player1_rod_x=jnp.array(0.0),
            player1_rod_y=jnp.array(0.0),
            player2_rod_x=jnp.array(0.0),
            player2_rod_y=jnp.array(0.0),
            score=jnp.array([0, 0]),
            shark_x=jnp.array([self.config.screen_width // 2]),
            fish_x=jnp.array(self.config.fish_x or [self.config.screen_width // 2]),
        )
        return self._get_observation(state), state
class FishingDerbyRenderer(AtraJaxisRenderer):
    def __init__(self):
        super().__init__()
        self.sprites = self._load_sprites()
        self.game_config = GameConfig()

    def load_sprites(self):
        MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
        # Load sprites
        bg = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/fishingderby/background.npy"))
        sky = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/fishingderby/sky.npy"))
        pl1 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/fishingderby/player1.npy"))
        pl2 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/fishingderby/player2.npy"))
        shark1 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/fishingderby/shark1.npy"))
        shark2 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/fishingderby/shark2.npy"))
        fish1 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/fishingderby/fish1.npy"))
        fish2 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/fishingderby/fish2.npy"))

        # padding sprites to match dimensions
        shark_sub_sprites = aj.pad_to_match([shark1, shark2])
        fish_sub_sprites = aj.pad_to_match([fish1, fish2])

        SPRITE_BG = jnp.expand_dims(bg, axis=0)

        SPRITE_SKY = jnp.expand_dims(sky, axis=0)

        # fish sprites (animation frames)
        SPRITE_FISH = jnp.concatenate(
            [
                jnp.repeat(fish_sub_sprites[0][None], 4, axis=0),
                jnp.repeat(fish_sub_sprites[1][None], 4, axis=0),
            ]
        )
        # shark sprites (animation frames)
        SPRITE_SHARK = jnp.concatenate(
            [
                jnp.repeat(shark_sub_sprites[0][None], 4, axis=0),
                jnp.repeat(shark_sub_sprites[1][None], 4, axis=0),
            ]
        )

        DIGITS = aj.load_and_pad_digits(os.path.join(MODULE_DIR, "sprites/fishingderby/digits/{}.npy"))

        return (
            SPRITE_BG,
            SPRITE_SKY,
            SPRITE_FISH,
            SPRITE_SHARK,
            DIGITS
        )

    # Load sprites once at module level
    (
        SPRITE_BG,
        SPRITE_SKY,
        SPRITE_FISH,
        SPRITE_SHARK,
        DIGITS
    ) = load_sprites()


if __name__ == "__main__":
    # Initialize game and renderer
    game = FishingDerby()
    pygame.init()
    screen = pygame.display.set_mode((game.config.screen_width, game.config.screen_height))
    renderer_Atrajaxis = FishingDerbyRenderer()

    # Get jitted functions
    jitted_step = jax.jit(game.step)
    jitted_reset = jax.jit(game.reset)

    curr_obs, curr_state = jitted_reset()

    # Game loop with rendering
    running = True
    frame_by_frame = False
    frameskip = 1
    counter = 1

