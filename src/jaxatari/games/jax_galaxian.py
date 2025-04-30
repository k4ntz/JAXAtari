import os
from functools import partial
from typing import NamedTuple, Tuple
import jax.lax
import jax.numpy as jnp
import chex
import pygame
from gymnax.environments import spaces

from jaxatari.games.jax_kangaroo import SCREEN_WIDTH


# -------- Game constants --------
SHOOTING_COOLDOWN = 20
ENEMY_MOVE_SPEED = 0.01

class GalaxianState(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    player_shooting: chex.Array
    player_shooting_cooldown: chex.Array
    enemy_grid: chex.Array
    enemy_grid_alive: chex.Array
    enemy_grid_direction: chex.Array

class Action(NamedTuple):
    player_move_dir: chex.Array
    player_shooting: chex.Array


def update_player_position(state: GalaxianState, action: Action) -> GalaxianState:
    new_x = state.player_x + action.player_move_dir * 5
    return state._replace(player_x=new_x)

# TODO implement direction change at border
def update_enemy_positions(state: GalaxianState) -> GalaxianState:
    new_enemy_grid = state.enemy_grid + ENEMY_MOVE_SPEED * state.enemy_grid_direction
    return state._replace(enemy_grid=new_enemy_grid)


def init_state():
    grid_rows = 5
    grid_cols = 7
    enemy_spacing_x = 20
    start_x = 100

    x_positions = jnp.arange(grid_cols) * enemy_spacing_x + start_x #arange schreibt so 0 1 2 3....
    enemy_grid = jnp.tile(x_positions, (grid_rows, 1))    #kopiert die zeile untereinander
    enemy_alive = jnp.ones((grid_rows, grid_cols), dtype=bool) #alles auf 1

    return GalaxianState(player_x=jnp.array(50),
                         player_y=jnp.array(20),
                         player_shooting_cooldown=jnp.array(0),
                         player_shooting=jnp.array(0),
                         enemy_grid=enemy_grid,
                         enemy_grid_alive=enemy_alive,
                         enemy_grid_direction=jnp.array(20))



def get_action_from_keyboard():
    keys = pygame.key.get_pressed()
    left = keys[pygame.K_a] or keys[pygame.K_LEFT]
    right = keys[pygame.K_d] or keys[pygame.K_RIGHT]
    shoot = keys[pygame.K_SPACE]
    move_dir = jnp.array(0)

    if left and not right:
        move_dir = jnp.array(-1)
    elif right and not left:
        move_dir = jnp.array(1)
    else:
        move_dir = jnp.array(0)

    if (shoot):
        shoot = jnp.array(1)
    else :
        shoot = jnp.array(0)

    return Action(player_move_dir=move_dir, player_shooting=shoot)


# TODO implement bullet
def handleShooting(state: GalaxianState, action: Action) -> GalaxianState:
    if action.player_shooting and state.player_shooting_cooldown == 0:
        state = state._replace(player_shooting_cooldown=jnp.array(SHOOTING_COOLDOWN))
    elif state.player_shooting_cooldown > 0:
        state = state._replace(player_shooting_cooldown=state.player_shooting_cooldown - 1)
    return state

def draw(screen, state: GalaxianState):
    player_rect = pygame.Rect(int(state.player_x), int(600 - state.player_y), 20, 10)
    pygame.draw.rect(screen, (0, 255, 0), player_rect)

    for i in range(state.enemy_grid.shape[0]):
        for j in range(state.enemy_grid.shape[1]):
            if state.enemy_grid_alive[i, j]:
                x = int(state.enemy_grid[i, j])
                y = 100 + i * 30
                enemy_rect = pygame.Rect(x, y, 15, 10)
                pygame.draw.rect(screen, (255, 0, 0), enemy_rect)


def step(state: GalaxianState, action: Action) -> GalaxianState:
    newState = update_player_position(state, action)
    newState = handleShooting(newState, action)
    newState = update_enemy_positions(newState)
    return  newState


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
        draw(screen, state)
        pygame.display.flip()
        action = get_action_from_keyboard()
        state = step(state, action)
        clock.tick(60)


    pygame.quit()


