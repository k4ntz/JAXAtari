import pygame
import tennis_renderer as renderer
from jaxatari.rendering import atraJaxis as aj
from typing import NamedTuple
import chex
import jax.lax
from functools import partial
import jax.numpy as jnp

WIDTH = 152
HEIGHT = 206

class TennisState(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    ball_x: chex.Array
    ball_y: chex.Array


#@partial(jax.jit, static_argnums=(0,))
def tennis_step(state: TennisState) -> TennisState:
    new_player_x = jnp.where(state.player_x < WIDTH, state.player_x + 1, state.player_x - 1)

    return TennisState(new_player_x, state.player_y, state.ball_y, state.ball_y)

def tennis_reset() -> TennisState:
    return TennisState(0, 100, 0, 0)

if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((WIDTH * 3, HEIGHT * 3))
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

        aj.update_pygame(screen, raster, 3, WIDTH, HEIGHT)

        clock.tick(30)

    pygame.quit()
