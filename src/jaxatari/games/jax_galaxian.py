import os
from functools import partial
from typing import NamedTuple, Tuple
import jax.lax
import jax.numpy as jnp
import chex
import pygame
from gymnax.environments import spaces

from jaxatari.games.jax_kangaroo import SCREEN_WIDTH

class GalaxianState(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    player_shooting_cooldown: chex.Array
    enemy_grid_x: chex.Array
    enemy_grid_y: chex.Array
    enemy_grid_direction: chex.Array

class Action(NamedTuple):
    player_move_dir: chex.Array


def update_player_position(state: GalaxianState, action: Action) -> GalaxianState:
    new_x = state.player_x + action.player_move_dir * 5
    return GalaxianState(
        player_x=new_x,
        player_y=state.player_y,
        player_shooting_cooldown=state.player_shooting_cooldown,
        enemy_grid_x=state.enemy_grid_x,
        enemy_grid_y=state.enemy_grid_y,
        enemy_grid_direction=state.enemy_grid_direction
    )



def step(state: GalaxianState, action: Action) -> GalaxianState:
    newState = update_player_position(state, action)
    return  newState




def init_state():
    return GalaxianState(player_x=jnp.array(50),
                         player_y=jnp.array(20),
                         player_shooting_cooldown=jnp.array(0),
                         enemy_grid_x=jnp.array(50),
                         enemy_grid_y=jnp.array(300),
                         enemy_grid_direction=jnp.array(20))



def get_action_from_keyboard():
    keys = pygame.key.get_pressed()
    left = keys[pygame.K_a] or keys[pygame.K_LEFT]
    right = keys[pygame.K_d] or keys[pygame.K_RIGHT]
    if left:
        return Action(player_move_dir=jnp.array(-1))
    elif right:
        return Action(player_move_dir=jnp.array(1))
    else:
        return Action(player_move_dir=jnp.array(0))


if __name__ == "__main__":  #run with: python -m jaxatari.games.jax_galaxian
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Galaxian")
    clock = pygame.time.Clock()

    running = True
    state = init_state()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.fill((0, 0, 0))
        pygame.display.flip()
        action = get_action_from_keyboard()
        state = step(state, action)
        clock.tick(60)
        print("X:", state.player_x)


    pygame.quit()