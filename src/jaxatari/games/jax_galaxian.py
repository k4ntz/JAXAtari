from typing import NamedTuple, Tuple
import jax.numpy as jnp
import chex
import pygame
from jax import lax
import jax.lax






# -------- Game constants --------
SHOOTING_COOLDOWN = 80
ENEMY_MOVE_SPEED = 10
BULLET_MOVE_SPEED = 2
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 800
GRID_ROWS = 6
GRID_COLS = 7
ENEMY_SPACING_X = 20
ENEMY_SPACING_Y = 20
ENEMY_GRID_Y = 300
START_X = 100

class GalaxianState(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    player_shooting_cooldown: chex.Array
    bullet_x: chex.Array
    bullet_y: chex.Array
    enemy_grid_x: chex.Array
    enemy_grid_y: chex.Array
    enemy_grid_alive: chex.Array
    enemy_grid_direction: chex.Array

class Action(NamedTuple):
    player_move_dir: chex.Array
    player_shooting: chex.Array

@jax.jit
def update_player_position(state: GalaxianState, action: Action) -> GalaxianState:
    new_x = jnp.clip(state.player_x + action.player_move_dir * 5, 0, SCREEN_WIDTH)
    return state._replace(player_x=new_x)

@jax.jit
def update_enemy_positions(state: GalaxianState) -> GalaxianState:
    def move_left():
        return -1

    def move_right():
        return 1

    def keep_direction():
        return state.enemy_grid_direction

    new_enemy_grid_direction = lax.cond(state.enemy_grid_x[0, state.enemy_grid_x.shape[1]] > SCREEN_WIDTH, move_left, lambda: lax.cond(state.enemy_grid_x[0, 0] < 0, move_right, keep_direction))
    new_enemy_grid_x = state.enemy_grid_x + ENEMY_MOVE_SPEED * state.enemy_grid_direction
    return state._replace(enemy_grid_x=new_enemy_grid_x, enemy_grid_direction=new_enemy_grid_direction)

def init_action():
    return Action(player_move_dir=jnp.array(0),
                  player_shooting=jnp.array(0))

def init_galaxian_state():
    grid_rows = GRID_ROWS
    grid_cols = GRID_COLS
    enemy_spacing_x = ENEMY_SPACING_X
    start_x = START_X

    x_positions = jnp.arange(grid_cols) * enemy_spacing_x + start_x #arange schreibt so 0 1 2 3....
    enemy_grid = jnp.tile(x_positions, (grid_rows, 1))    #kopiert die zeile untereinander

    row_indices = jnp.arange(grid_rows).reshape(-1, 1)  # erzeugt 1. enemy jeder zeile
    enemy_y_rows = ENEMY_GRID_Y - row_indices * ENEMY_SPACING_Y  # jeweils y pos
    enemy_grid_y = jnp.broadcast_to(enemy_y_rows, (
    grid_rows, grid_cols))  # kopiert die werte für 1. enemy auf die rechts in der zeile

    enemy_alive = jnp.ones((grid_rows, grid_cols), dtype=bool) #alles auf 1

    return GalaxianState(player_x=jnp.array(SCREEN_WIDTH / 2),
                         player_y=jnp.array(20),
                         player_shooting_cooldown=jnp.array(0),
                         bullet_x=jnp.array(-1),
                         bullet_y=jnp.array(-1),
                         enemy_grid_x=enemy_grid,
                         enemy_grid_y=enemy_grid_y,
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



def handle_shooting(state: GalaxianState, action: Action) -> GalaxianState:
    if action.player_shooting and state.player_shooting_cooldown == 0:
        new_bullet_x = jnp.array(state.player_x)
        new_bullet_y = jnp.array(state.player_y)
        return state._replace(
            bullet_x=new_bullet_x,
            bullet_y=new_bullet_y,
            player_shooting_cooldown=jnp.array(SHOOTING_COOLDOWN)
        )
    elif state.player_shooting_cooldown > 0:
        return state._replace(player_shooting_cooldown=state.player_shooting_cooldown - 1)
    return state


def update_bullets(state: GalaxianState) -> GalaxianState:
    new_bullets_y = jnp.array(state.bullet_y + BULLET_MOVE_SPEED)
    return state._replace(bullet_y=new_bullets_y)


def remove_bullets(state: GalaxianState) -> GalaxianState:
    if state.bullet_y > 500:
        new_bullet_y = jnp.array(-1)
        new_bullet_x = jnp.array(-1)
        return state._replace(bullet_y=new_bullet_y, bullet_x=new_bullet_x)
    return state

@jax.jit
def bullet_collision(state: GalaxianState) -> GalaxianState:
    x_diff = jnp.abs(state.bullet_x - state.enemy_grid_x)
    y_diff = jnp.abs(state.bullet_y - state.enemy_grid_y)

    collision_mask = (x_diff <= 10) & (y_diff <= 10) & state.enemy_grid_alive
    max_collisions_size = state.enemy_grid_alive.size
    collision_indices = jnp.where(
        collision_mask,
        size=max_collisions_size,
        fill_value=-1
    )

    hit = jnp.any(collision_mask)

    def process_hit(operands):
        current_state, indices = operands
        row_indices, col_indices = indices
        new_enemy_grid_alive = current_state.enemy_grid_alive.at[row_indices, col_indices].set(False)
        new_bullet_x = jnp.array(-1, dtype=state.bullet_x.dtype)
        new_bullet_y = jnp.array(-1, dtype=state.bullet_y.dtype)

        return state._replace(
            enemy_grid_alive=new_enemy_grid_alive,
            bullet_x=new_bullet_x,
            bullet_y=new_bullet_y
        )


    def process_none(operands):
        current_state, _ = operands
        return current_state

    return lax.cond(hit, process_hit, process_none, operand=(state, collision_indices))



def draw(screen, state: GalaxianState):
    player_rect = pygame.Rect(int(state.player_x), int(600 - state.player_y), 20, 10)
    pygame.draw.rect(screen, (0, 255, 0), player_rect)


    if state.bullet_x > 0:
        bullet_rect = pygame.Rect(int(state.bullet_x), int(600 - state.bullet_y), 5, 10)
        pygame.draw.rect(screen, (255, 255, 0), bullet_rect)

    for i in range(state.enemy_grid_x.shape[0]):
        for j in range(state.enemy_grid_x.shape[1]):
            if state.enemy_grid_alive[i, j]:
                x = int(state.enemy_grid_x[i, j])
                y = ENEMY_GRID_Y + i * ENEMY_SPACING_Y
                enemy_rect = pygame.Rect(x, y, 15, 10)
                pygame.draw.rect(screen, (255, 0, 0), enemy_rect)


def step(state: GalaxianState, action: Action) -> GalaxianState:
    newState = update_player_position(state, action)
    newState = handle_shooting(newState, action)
    newState = update_enemy_positions(newState)
    newState = update_bullets(newState)
    newState = remove_bullets(newState)
    newState = bullet_collision(newState)
    return  newState


if __name__ == "__main__":  #run with: python -m jaxatari.games.jax_galaxian
    pygame.init()
    state = init_galaxian_state()
    # jit causes first shot to lag, might need better solution
    dummy_action = init_action()._replace(player_shooting=jnp.array(1))
    state = step(state, dummy_action)
    state = init_galaxian_state()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Galaxian")
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.fill((0, 0, 0))
        draw(screen, state)
        pygame.display.flip()
        action = get_action_from_keyboard()
        state = step(state, action)
        clock.tick(30)


    pygame.quit()


