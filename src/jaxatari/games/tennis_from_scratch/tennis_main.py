import pygame
import tennis_renderer as renderer
from jaxatari.rendering import atraJaxis as aj
from typing import NamedTuple
import chex
import jax.lax
from functools import partial
import jax.numpy as jnp

FRAME_WIDTH = 152
FRAME_HEIGHT = 206
# top: (left_x = 32, right_x = 111, width = right_x - left_x = 79)
FIELD_WIDTH_TOP = 79
# bottom: (left_x = 16, right_x = 127, width = right_x - left_x = 111)
FIELD_WIDTH_BOTTOM = 111
# top_y = 44, bottom_y = 174, height = bottom_y - top_y = 130
FIELD_HEIGHT = 130

# these values are used for the actual gameplay calculations
GAME_OFFSET_LEFT_BOTTOM = 15 + 1 # don't use 16, because that is on the line and playing on the line still counts todo: check if that is actually true
GAME_OFFSET_TOP = 43 # don't use 44, because that is on the line and playing on the line still counts todo: check if that is actually true
GAME_WIDTH = FIELD_WIDTH_BOTTOM
GAME_HEIGHT = FIELD_HEIGHT

class TennisState(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    ball_x: chex.Array
    ball_y: chex.Array
    ball_z: chex.Array = 0.0
    ball_direction: chex.Array = 0
    counter : chex.Array = 0


#@partial(jax.jit, static_argnums=(0,))
def tennis_step(state: TennisState) -> TennisState:
    # new_player_x = jnp.where(state.player_x < FRAME_WIDTH, state.player_x + 1, state.player_x - 1)
    new_ball_x = jnp.where(state.ball_direction == 0, state.ball_x + 1, state.ball_x - 1)

    new_direction = jnp.where(state.ball_x >= GAME_WIDTH, 1, state.ball_direction)
    new_direction = jnp.where(state.ball_x <= 0, 0, new_direction)

    new_ball_y = jnp.where(state.counter % 4 == 0, state.ball_y + 1, state.ball_y)

    return TennisState(state.player_x, state.player_y, new_ball_x, new_ball_y, ball_direction=new_direction, counter=state.counter + 1)

def tennis_reset() -> TennisState:
    return TennisState(0.0, 100.0, 0.0, 0.0)

if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((FRAME_WIDTH * 3, FRAME_HEIGHT * 3))
    pygame.display.set_caption("Tennis Game")
    clock = pygame.time.Clock()

    renderer = renderer.TennisRenderer()

    jitted_step = jax.jit(tennis_step)
    jitted_reset = jax.jit(tennis_reset)

    current_state = jitted_reset()

    running = True
    while running:
        # Inputs
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Update logic
        current_state = jitted_step(current_state)

        # Render and display
        raster = renderer.render(current_state)

        aj.update_pygame(screen, raster, 3, FRAME_WIDTH, FRAME_HEIGHT)

        clock.tick(30)

    pygame.quit()
