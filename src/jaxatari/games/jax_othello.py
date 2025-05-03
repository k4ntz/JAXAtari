import os
from functools import partial
from typing import NamedTuple, Tuple
import jax.lax
import jax.numpy as jnp
import chex
import pygame
import enum
from gymnax.environments import spaces

from jaxatari.rendering import atraJaxis as aj
from jaxatari.environment import JaxEnvironment


# Game Environment
WIDTH = 160
HEIGHT = 210

# Pygame window dimensions
WINDOW_WIDTH = 160 * 3
WINDOW_HEIGHT = 210 * 3

# Actions constants
NOOP = 0
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4
PLACE = 5
DIFFICULTY = 6
RESET = 7


def get_human_action() -> chex.Array:
    keys = pygame.key.get_pressed()
    if keys[pygame.K_1]:
        return jnp.array(DIFFICULTY)
    elif keys[pygame.K_2]:
        return jnp.array(RESET)
    elif keys[pygame.K_w]:
        return jnp.array(UP)
    elif keys[pygame.K_s]:
        return jnp.array(DOWN)
    elif keys[pygame.K_a]:
        return jnp.array(LEFT)
    elif keys[pygame.K_d]:
        return jnp.array(RIGHT)
    elif keys[pygame.K_LCTRL]:
        return jnp.array(PLACE)
    else:
        return jnp.array(NOOP)



# state container
class FieldColor(enum.IntEnum):
    EMPTY = 0
    WHITE = 1
    BLACK = 2

class Field(NamedTuple):
    filed_id: chex.Array
    field_color: chex.Array

class OthelloState(NamedTuple):
    player_score: chex.Array
    enemy_score: chex.Array
    step_counter: chex.Array
    field: Field
    field_place: chex.Array

class OthelloObservation(NamedTuple):
    field: Field
    score_player: jnp.ndarray
    score_enemy: jnp.ndarray

class OthelloInfo(NamedTuple):
    time: jnp.ndarray
    all_rewards: chex.Array


class JaxOthello(JaxEnvironment[OthelloState, OthelloObservation, OthelloInfo]):
    def __init__(self, frameskip: int = 0, reward_funcs: list[callable]=None):
        super().__init__()
        self.frameskip = frameskip + 1
        self.frame_stack_size = 4
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.action_set = set(range(64)) | {NOOP}
        self.obs_size = 130


    def reset(self) -> OthelloState:
        """ Reset the game state to the initial state """
        field_color_init = jnp.full((8, 8), FieldColor.EMPTY.value, dtype=jnp.int32)
        field_color_init = field_color_init.at[3,3].set(FiledColor.BLACK.value)
        field_color_init = field_color_init.at[4,3].set(FiledColor.WHITE.value)
        field_color_init = field_color_init.at[3,4].set(FiledColor.WHITE.value)
        field_color_init = field_color_init.at[4,4].set(FiledColor.BLACK.value)

        state = OthelloState(
            player_score = jnp.array(2).astype(jnp.int32),
            enemy_score = jnp.array(2).astype(jnp.int32),
            step_counter =jnp.array(0).astype(jnp.int32),
            field = Field(
                field_id = jnp.array(64, dtype=jnp.int32).reshape((8,8)),
                field_color = field_color_init
            ),
            field_place = jnp.array(63).astype(jnp.int32)
        )
        initial_obs = self._get_observation(state)


    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: OthelloState):
        return OthelloObservation(
            field=state.field,
            score_enemy=field.score_enemy,
            score_player=field.score_player
        )




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
        # PLAYER_DIGIT_SPRITES,
        # ENEMY_DIGIT_SPRITES
    )


class Renderer_AtraJaxisOthello:

    def __init__(self):
        (
            self.SPRITE_BG,
            self.SPRITE_PLAYER,
            self.SPRITE_ENEMY,
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

    game = JaxOthello(frameskip=1)

    # Create the JAX renderer
    renderer = Renderer_AtraJaxisOthello()




    # Game Loop
    running = True
    frameskip = 4
    counter = 1

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False     
        
        if counter % frameskip == 0:
            action = get_human_action()


        # Render and display
        raster = renderer.render(None)
        aj.update_pygame(screen, raster, 3, WIDTH, HEIGHT)

    pygame.quit()