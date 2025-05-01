import os
from functools import partial
from typing import NamedTuple, Tuple
from jaxatari.environment import JaxEnvironment, EnvState, EnvObs, EnvInfo
import jax
import jax.numpy as jnp
import chex
import numpy as np
import pygame
import jaxatari.rendering.atraJaxis as aj
from jaxatari.games.jax_tennis import COURT_WIDTH, COURT_HEIGHT, JaxTennis, Renderer_AJ, AnimatorState
from util import *

BG, PL_R, BAT_R, PL_B, BAT_B, BALL, BALL_SHADE, DIGITS_R, DIGITS_B = load_sprites()

if __name__ == "__main__":
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((COURT_WIDTH * 4, COURT_HEIGHT * 4))
    pygame.display.set_caption("Tennis Game")
    clock = pygame.time.Clock()

    # Create game instance
    game = JaxTennis(frameskip=1)

    # Initialize renderer
    renderer = Renderer_AJ()
    animator_state = AnimatorState(
        r_x=0, r_y=0, r_f=12, r_bat_f=0, b_x=0, b_y=0, b_f=12, b_bat_f=0
    )

    # JIT compile main functions
    jitted_step = jax.jit(game.step)
    jitted_reset = jax.jit(game.reset)

    # Main game loop structure
    curr_state, curr_obs = jitted_reset()
    running = True
    frame_by_frame = False
    frameskip = game.frameskip
    counter = 1

    list_of_y = []
    list_of_z = []

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
                        curr_state, obs, reward, done, info = jitted_step(curr_state, action)
                        # print the current game scores
                        print(f"Player: {curr_state.player_score} - Enemy: {curr_state.enemy_score}")

        if not frame_by_frame:
            if counter % frameskip == 0:
                # Get action (to be implemented with proper controls)
                action = get_human_action()
                curr_state, obs, reward, done, info = jitted_step(curr_state, action)

        raster, animator_state = renderer.render(curr_state, animator_state)
        aj.update_pygame(screen, raster, 4, COURT_WIDTH, COURT_HEIGHT)

        counter += 1
        clock.tick(30)

    pygame.quit()