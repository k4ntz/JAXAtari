import os
from functools import partial
from typing import NamedTuple, Tuple
import jax.lax
import jax.numpy as jnp
import chex
import pygame
from gymnax.environments import spaces

from src.jaxatari.games.jax_kangaroo import SCREEN_HEIGHT, SCREEN_WIDTH

#from jaxatari.games.jax_kangaroo import SCREEN_WIDTH


# -------- Game constants --------
SHOOTING_COOLDOWN = 20
ENEMY_MOVE_SPEED = 1
BULLET_MOVE_SPEED = 2
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 800
GRID_ROWS = 5
GRID_COLS = 7
ENEMY_SPACING_X = 20
ENEMY_SPACING_Y = 20
ENEMY_GRID_Y = 300
START_X = 100

class GalaxianState(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    player_shooting: chex.Array
    player_shooting_cooldown: chex.Array
    bullet_x: chex.Array
    bullet_y: chex.Array
    enemy_grid: chex.Array
    enemy_grid_alive: chex.Array
    enemy_grid_direction: chex.Array

class Action(NamedTuple):
    player_move_dir: chex.Array
    player_shooting: chex.Array


def update_player_position(state: GalaxianState, action: Action) -> GalaxianState:
    new_x = jnp.clip(state.player_x + action.player_move_dir * 5, 0, SCREEN_WIDTH)
    return state._replace(player_x=new_x)


def update_enemy_positions(state: GalaxianState) -> GalaxianState:
    if state.enemy_grid[0, state.enemy_grid.shape[1]] > SCREEN_WIDTH:
        new_enemy_grid_direction = -1
    elif state.enemy_grid[0, 0] < 0:
        new_enemy_grid_direction = 1
    else:
        new_enemy_grid_direction = state.enemy_grid_direction

    new_enemy_grid = state.enemy_grid + ENEMY_MOVE_SPEED * state.enemy_grid_direction
    #print (new_enemy_grid)
    #print (new_enemy_grid_direction)
    return state._replace(enemy_grid=new_enemy_grid, enemy_grid_direction=new_enemy_grid_direction)


def init_state():
    grid_rows = GRID_ROWS
    grid_cols = GRID_COLS
    enemy_spacing_x = ENEMY_SPACING_X
    start_x = START_X

    x_positions = jnp.arange(grid_cols) * enemy_spacing_x + start_x #arange schreibt so 0 1 2 3....
    enemy_grid = jnp.tile(x_positions, (grid_rows, 1))    #kopiert die zeile untereinander
    enemy_alive = jnp.ones((grid_rows, grid_cols), dtype=bool) #alles auf 1

    return GalaxianState(player_x=jnp.array(SCREEN_WIDTH / 2),
                         player_y=jnp.array(20),
                         player_shooting_cooldown=jnp.array(0),
                         player_shooting=jnp.array(0),
                         bullet_x=jnp.array([]),
                         bullet_y=jnp.array([]),
                         enemy_grid=enemy_grid,
                         enemy_grid_alive=enemy_alive,
                         enemy_grid_direction=jnp.array(1))



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

    if shoot:
        shoot = jnp.array(1)
    else:
        shoot = jnp.array(0)

    return Action(player_move_dir=move_dir, player_shooting=shoot)



def handleShooting(state: GalaxianState, action: Action) -> GalaxianState:
    if action.player_shooting and state.player_shooting_cooldown == 0:
        new_bullet_x = jnp.append(state.bullet_x, state.player_x)
        new_bullet_y = jnp.append(state.bullet_y, state.player_y)
        return state._replace(
            bullet_x=new_bullet_x,
            bullet_y=new_bullet_y,
            player_shooting_cooldown=jnp.array(SHOOTING_COOLDOWN)
        )
    elif state.player_shooting_cooldown > 0:
        return state._replace(player_shooting_cooldown=state.player_shooting_cooldown - 1)
    return state



def updateBullets(state: GalaxianState) -> GalaxianState:
    new_bullets_y = jax.vmap(lambda bullet_y: (bullet_y + BULLET_MOVE_SPEED))
    return state._replace(bullet_y=new_bullets_y(state.bullet_y))

def removeBullets(state: GalaxianState) -> GalaxianState:
    top_cutoff = state.bullet_y <= 500
    new_bullet_y = state.bullet_y[top_cutoff]
    new_bullet_x = state.bullet_x[top_cutoff]
    return state._replace(bullet_y=new_bullet_y, bullet_x=new_bullet_x)

def bulletCollision(state: GalaxianState) -> GalaxianState:
    for i in range(state.bullet_x.shape[0]):
        for j in range(state.enemy_grid.shape[0]):
            for k in range(state.enemy_grid.shape[1]):
                if abs(state.bullet_x[i] - state.enemy_grid[j, k]) <= 10 and abs(state.bullet_y[i] - (ENEMY_GRID_Y - j * ENEMY_SPACING_Y)) <= 10 and state.enemy_grid_alive[j, k]:
                    new_enemy_grid_alive = state.enemy_grid_alive.at[j, k].set(False)
                    new_bullet_x = jnp.delete(state.bullet_x, i)
                    new_bullet_y = jnp.delete(state.bullet_y, i)
                    return state._replace(enemy_grid_alive=new_enemy_grid_alive, bullet_x=new_bullet_x, bullet_y=new_bullet_y)
    return state

def draw(screen, state: GalaxianState):
    player_rect = pygame.Rect(int(state.player_x), int(600 - state.player_y), 20, 10)
    pygame.draw.rect(screen, (0, 255, 0), player_rect)

    for i in range(state.bullet_x.shape[0]):
        if state.bullet_x[i] > 0:
            bullet_rect = pygame.Rect(int(state.bullet_x[i]), int(600 - state.bullet_y[i]), 5, 10)
            pygame.draw.rect(screen, (255, 255, 0), bullet_rect)

    for i in range(state.enemy_grid.shape[0]):
        for j in range(state.enemy_grid.shape[1]):
            if state.enemy_grid_alive[i, j]:
                x = int(state.enemy_grid[i, j])
                y = ENEMY_GRID_Y + i * ENEMY_SPACING_Y
                enemy_rect = pygame.Rect(x, y, 15, 10)
                pygame.draw.rect(screen, (255, 0, 0), enemy_rect)


def step(state: GalaxianState, action: Action) -> GalaxianState:
    newState = update_player_position(state, action)
    newState = handleShooting(newState, action)
    newState = update_enemy_positions(newState)
    newState = updateBullets(newState)
    newState = removeBullets(newState)
    newState = bulletCollision(newState)
    return  newState


if __name__ == "__main__":  #run with: python -m jaxatari.games.jax_galaxian
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
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


