from typing import NamedTuple, Tuple
import jax.numpy as jnp
import chex
import pygame
from absl.testing.parameterized import parameters
from jax import lax
import jax.lax
from pandas.core.interchange import column

# -------- Game constants --------
SHOOTING_COOLDOWN = 80
ENEMY_MOVE_SPEED = 2
BULLET_MOVE_SPEED = 5
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 800
GRID_ROWS = 6
GRID_COLS = 7
ENEMY_SPACING_X = 20
ENEMY_SPACING_Y = 20
ENEMY_GRID_Y = 300
START_X = 100
ENEMY_ATTACK_SPEED = 5

class GalaxianState(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    player_shooting_cooldown: chex.Array
    bullet_x: chex.Array
    bullet_y: chex.Array
    enemy_grid_x: chex.Array
    enemy_grid_y: chex.Array
    enemy_grid_alive: chex.Array        # 0: dead, 1: alive, 2: attacking
    enemy_grid_direction: chex.Array
    enemy_attack_state: chex.Array      # 0: init, 1: attack, 2: respawn
    enemy_attack_pos: chex.Array
    enemy_attack_x: chex.Array
    enemy_attack_y: chex.Array
    enemy_attack_respawn_timer: chex.Array
    random_key: chex.Array

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

# @jax.jit
# def update_enemy_attack_state(state: GalaxianState) -> GalaxianState:
#
#   #  enemy_attack_state = state.enemy_attack_state # 0: init, 1: attack, 2: respawn
#   #  enemy_attack_pos = state.enemy_attack_pos
#   #  enemy_attack_x = state.enemy_attack_x
#   #  enemy_attack_y = state.enemy_attack_y
#     enemy_attack_respawn_timer = state.enemy_attack_respawn_timer
#     enemy_grid_alive = state.enemy_grid_alive
#     enemy_grid_x = state.enemy_grid_x
#     enemy_grid_y = state.enemy_grid_y
#
#     def start_attack(s):
#         enemy_attack_state, enemy_attack_pos, enemy_attack_x, enemy_attack_y, enemy_attack_respawn_timer, enemy_grid_alive, enemy_grid_x, enemy_grid_y = s
#
#         row = 0
#         column = 0
#
#         lax.while_loop(
#             lambda s: enemy_grid_alive[s[0],s[1]] != 1,
#             lambda s: jax.lax.cond(
#                 s[0] >= GRID_ROWS,
#                 lambda u: (s[0] + 1, 0),
#                 lambda u: (s[0], s[1] + 1),
#                 s
#             ),
#             (row, column)
#         )
#
#         return 1, jnp.array([row, column]), enemy_grid_x[row, column], enemy_grid_y[row, column], enemy_attack_respawn_timer, enemy_grid_alive.at[row, column].set(2)
#
#     new_enemy_attack_state, new_enemy_attack_pos, new_enemy_attack_x, new_enemy_attack_y, new_enemy_attack_respawn_timer, new_enemy_grid_alive = jax.lax.cond(
#         enemy_attack_state == 0,
#         start_attack,
#         lambda s: jax.lax.cond(
#             enemy_attack_state == 1,
#             lambda t: jax.lax.cond(
#                 enemy_grid_alive[enemy_attack_pos] == 0,
#                 lambda u: (0, t[1], t[2], t[3], 20, t[5]),
#                 lambda u: lax.cond(
#                     enemy_attack_y > SCREEN_HEIGHT,
#                     lambda v: (2, u[1], u[2], u[3], u[4], u[5].at[0,0].set(1)),
#                     lambda v: (u[0], u[1],u[2], u[3] + ENEMY_ATTACK_SPEED, u[4], u[5]),
#             u
#                 ),
#                 t,
#             ),
#             # enemy attack state == 2
#             lambda t: jax.lax.cond(
#                 enemy_attack_respawn_timer <= 0,
#                 lambda u: (0, t[1], t[2], t[3], 20, t[5]),
#                 lambda u: (t[0], t[1], t[2], t[3], t[4] - 1, t[5]),
#                 t
#             ),
#             s
#         ),
#         (enemy_attack_state, enemy_attack_pos, enemy_attack_x, enemy_attack_y, enemy_attack_respawn_timer, enemy_grid_alive, enemy_grid_x, enemy_grid_y)
#     )
#
#     return state._replace(
#         enemy_attack_state=new_enemy_attack_state,
#         enemy_attack_pos=new_enemy_attack_pos,
#         enemy_attack_x=new_enemy_attack_x,
#         enemy_attack_y=new_enemy_attack_y,
#         enemy_grid_alive=new_enemy_grid_alive,
#         enemy_attack_respawn_timer=new_enemy_attack_respawn_timer
#     )

@jax.jit
def update_enemy_attack(state: GalaxianState) -> GalaxianState:

    jax.debug.print("grid: {}", state.enemy_grid_alive[tuple(state.enemy_attack_pos)])
    jax.debug.print("state: {}", state.enemy_attack_state)
    new_enemy_attack_state = jnp.where(
        # enemy was killed, and state has not been reset yet, transitions to state 0
        jnp.logical_and(state.enemy_grid_alive[tuple(state.enemy_attack_pos)] == 0,state.enemy_attack_state != 0),
        0,
        jnp.where(
            # state 0, instantly transitions to state 1
            state.enemy_attack_state == 0,
            1,
            jnp.where(
                # state 1, transitions to state 2 if out of screen, otherwise transitions to state 1
                jnp.logical_and(state.enemy_attack_state == 1, state.enemy_attack_y > SCREEN_HEIGHT),
                2,
                jnp.where(
                    # state 2, transitions to state 0 if respawn timer is 0, otherwise transitions to state 2
                    jnp.logical_and(state.enemy_attack_state == 2, state.enemy_attack_respawn_timer <= 0),
                    0,
                    state.enemy_attack_state,
                )
            )
        )
    )

    def determine_enemy_pos(_):
        row = 0
        column = 0
        jax.debug.print("grid determine: {}", state.enemy_grid_alive[0,0])
        position = lax.while_loop(
            lambda s: state.enemy_grid_alive[s] != 1,
            lambda s: jax.lax.cond(
                s[1] >= GRID_ROWS,
                lambda u: (s[0] + 1, 0),
                lambda u: (s[0], s[1] + 1),
                s
            ),
            (row, column)
        )

        jax.debug.print("new pos: {}, {}", row, column)
        return jnp.array(position)

    new_enemy_attack_pos = lax.cond(
        state.enemy_attack_state == 0,
        determine_enemy_pos,
        lambda s: s,
        state.enemy_attack_pos
    )


    new_enemy_attack_x, new_enemy_attack_y = lax.cond(
        state.enemy_attack_state == 0,
        lambda _: (state.enemy_grid_x[tuple(new_enemy_attack_pos)], state.enemy_grid_y[tuple(new_enemy_attack_pos)]),
        lambda _: (state.enemy_attack_x, state.enemy_attack_y),
        operand=None
    )

    delta_x = jnp.where(
        state.enemy_attack_state == 1,
        jnp.where(
            state.enemy_attack_x < state.player_x,
            ENEMY_ATTACK_SPEED,
            -ENEMY_ATTACK_SPEED
        ),
        0
    )

    delta_y = jnp.where(
        state.enemy_attack_state == 1,
        1,
        0
    )

    new_enemy_attack_respawn_timer = jnp.where(
        state.enemy_attack_state == 2,
        state.enemy_attack_respawn_timer -1,
        20
    )


    new_enemy_grid_alive = jnp.where(
        state.enemy_attack_state == 0,
        state.enemy_grid_alive.at[tuple(new_enemy_attack_pos)].set(2),
        jnp.where(
            jnp.logical_and(state.enemy_attack_state == 1, state.enemy_attack_y > SCREEN_HEIGHT),
            state.enemy_grid_alive.at[tuple(new_enemy_attack_pos)].set(1),
            state.enemy_grid_alive
        )
    )

    return state._replace(
        enemy_attack_state=new_enemy_attack_state,
        enemy_attack_x=new_enemy_attack_x + delta_x,
        enemy_attack_y=new_enemy_attack_y + delta_y,
        enemy_attack_pos=new_enemy_attack_pos,
        enemy_attack_respawn_timer=new_enemy_attack_respawn_timer,
        enemy_grid_alive=new_enemy_grid_alive
    )


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

    enemy_alive = jnp.ones((grid_rows, grid_cols)) #alles auf 1


    return GalaxianState(player_x=jnp.array(SCREEN_WIDTH / 2),
                         player_y=jnp.array(SCREEN_HEIGHT - 20),
                         player_shooting_cooldown=jnp.array(0),
                         bullet_x=jnp.array(-1),
                         bullet_y=jnp.array(-1),
                         enemy_grid_x=enemy_grid,
                         enemy_grid_y=enemy_grid_y,
                         enemy_grid_alive=enemy_alive,
                         enemy_grid_direction=jnp.array(1),
                         enemy_attack_state=jnp.array(0),
                         enemy_attack_pos=jnp.array((-1,-1)),
                         enemy_attack_x=jnp.array(-1),
                         enemy_attack_y=jnp.array(-1),
                         enemy_attack_respawn_timer=jnp.array(20),
                         random_key=jax.random.PRNGKey(0))


def get_action_from_keyboard():
    keys = pygame.key.get_pressed()
    left = keys[pygame.K_a] or keys[pygame.K_LEFT]
    right = keys[pygame.K_d] or keys[pygame.K_RIGHT]
    shooting = keys[pygame.K_SPACE]
    move_dir = jnp.array(0)

    left_cond =  jnp.logical_and(left, not right)
    right_cond = jnp.logical_and(right, not left)

    move_dir = lax.cond(
        left_cond,
        lambda _: jnp.array(-1),
        lambda _: lax.cond(
            right_cond,
            lambda _: jnp.array(1),
            lambda _: jnp.array(0),
            operand=None
        ),
        operand=None
    )

    shoot = lax.cond(shooting, lambda _: jnp.array(1), lambda _: jnp.array(0), operand=None)

    return Action(player_move_dir=move_dir, player_shooting=shoot)


@jax.jit
def handle_shooting(state: GalaxianState, action: Action) -> GalaxianState:
    def shoot(_):
        return state._replace(
            bullet_x=jnp.array(state.player_x, dtype=state.bullet_x.dtype),
            bullet_y=jnp.array(state.player_y),
            player_shooting_cooldown=jnp.array(SHOOTING_COOLDOWN)
        )

    def decrease_cooldown(_):
        return state._replace(
            player_shooting_cooldown=state.player_shooting_cooldown - 1
        )

    def idle(_):
        return state

    can_shoot = jnp.logical_and(action.player_shooting, state.player_shooting_cooldown == 0)
    on_cooldown = jnp.logical_not(can_shoot) & (state.player_shooting_cooldown > 0)

    return lax.cond(
        can_shoot,
        shoot,
        lambda _: lax.cond(on_cooldown, decrease_cooldown, idle, operand=None),
        operand=None
    )


@jax.jit
def update_bullets(state: GalaxianState) -> GalaxianState:
    new_bullets_y = jnp.array(state.bullet_y - BULLET_MOVE_SPEED)
    return state._replace(bullet_y=new_bullets_y)


@jax.jit
def remove_bullets(state: GalaxianState) -> GalaxianState:
    def reset_bullet(_):
        new_bullet_y = jnp.array(-1)
        new_bullet_x = jnp.array(-1)
        return state._replace(bullet_y=new_bullet_y, bullet_x=new_bullet_x)

    def idle(_):
        return state

    return lax.cond(state.bullet_y < 100, reset_bullet, idle, operand=None)


@jax.jit
def bullet_collision(state: GalaxianState) -> GalaxianState:
    x_diff = jnp.abs(state.bullet_x - state.enemy_grid_x)
    y_diff = jnp.abs(state.bullet_y - state.enemy_grid_y)

    collision_mask = (x_diff <= 10) & (y_diff <= 10) & (state.enemy_grid_alive == 1)
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
        new_enemy_grid_alive = current_state.enemy_grid_alive.at[row_indices, col_indices].set(0)
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

@jax.jit
def bullet_collision_attack(state: GalaxianState) -> GalaxianState:
    x_diff = jnp.abs(state.bullet_x - state.enemy_attack_x)
    y_diff = jnp.abs(state.bullet_y - state.enemy_attack_y)
    collision = (x_diff <= 10) & (y_diff <= 10)
    hit = jnp.any(collision)

    def process_hit(operands):
        current_state = operands
        enemy_attack_pos = current_state.enemy_attack_pos
        new_enemy_grid_alive = current_state.enemy_grid_alive.at[tuple(enemy_attack_pos)].set(0)
        new_bullet_x = jnp.array(-1, dtype=state.bullet_x.dtype)
        new_bullet_y = jnp.array(-1, dtype=state.bullet_y.dtype)
        return state._replace(
            enemy_grid_alive=new_enemy_grid_alive,
            bullet_x=new_bullet_x,
            bullet_y=new_bullet_y
        )
    def process_none(operands):
        current_state = operands
        return current_state

    return lax.cond(hit, process_hit, process_none, operand=state)


def draw(screen, state: GalaxianState):
    player_rect = pygame.Rect(int(state.player_x), int(state.player_y), 20, 10)
    pygame.draw.rect(screen, (0, 255, 0), player_rect)


    if state.bullet_x > 0:
        bullet_rect = pygame.Rect(int(state.bullet_x), int(state.bullet_y), 5, 10)
        pygame.draw.rect(screen, (255, 255, 0), bullet_rect)

    if state.enemy_attack_state.any() == 1:
        enemy_attack_rect = pygame.Rect(int(state.enemy_attack_x),int(state.enemy_attack_y), 15,10)
        pygame.draw.rect(screen, (255,0,0), enemy_attack_rect)

    for i in range(state.enemy_grid_x.shape[0]):
        for j in range(state.enemy_grid_x.shape[1]):
            if state.enemy_grid_alive[i, j] == 1:
                x = int(state.enemy_grid_x[i, j])
                y = ENEMY_GRID_Y - i * ENEMY_SPACING_Y
                enemy_rect = pygame.Rect(x, y, 15, 10)
                pygame.draw.rect(screen, (255, 0, 0), enemy_rect)


@jax.jit
def step(state: GalaxianState, action: Action) -> GalaxianState:
    newState = update_player_position(state, action)
    newState = handle_shooting(newState, action)
    newState = update_enemy_positions(newState)
    newState = update_bullets(newState)
    newState = remove_bullets(newState)
    newState = bullet_collision(newState)
    newState = bullet_collision_attack(newState)
    newState = update_enemy_attack(newState)
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


