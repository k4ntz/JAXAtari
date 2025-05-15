import os
from typing import NamedTuple, Tuple
import jax.numpy as jnp
import chex
import pygame
from functools import partial
from jax import lax
import jax.lax
from gymnax.environments import spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action

from jaxatari.games.jax_seaquest import SPRITE_BG
from jaxatari.renderers import AtraJaxisRenderer
from jaxatari.rendering import atraJaxis as aj

from src.jaxatari.environment import JaxEnvironment
from jaxatari.environment import JaxEnvironment

"""
# Action constants
NOOP = 0
FIRE = 1
RIGHT = 2
LEFT = 3
"""

if not pygame.get_init():
    pygame.init()

# -------- Game constants --------
SHOOTING_COOLDOWN = 80
ENEMY_MOVE_SPEED = 0.5
BULLET_MOVE_SPEED = 2
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 800
GRID_ROWS = 6
GRID_COLS = 7
NATIVE_GAME_WIDTH = 224
NATIVE_GAME_HEIGHT = 288
PYGAME_SCALE_FACTOR = 3
PYGAME_WINDOW_WIDTH = NATIVE_GAME_WIDTH * PYGAME_SCALE_FACTOR
PYGAME_WINDOW_HEIGHT = NATIVE_GAME_HEIGHT * PYGAME_SCALE_FACTOR
ENEMY_SPACING_X = 7
ENEMY_SPACING_Y = 7
ENEMY_GRID_Y = 40
START_X = NATIVE_GAME_WIDTH // 4
START_Y = NATIVE_GAME_HEIGHT - 20
ENEMY_ATTACK_SPEED = 1

ENEMY_SPACING_X = 20
ENEMY_SPACING_Y = 20
ENEMY_GRID_Y = 300
START_X = 100
ENEMY_ATTACK_SPEED = 2
ENEMY_ATTACK_TURN_TIME = 30
ENEMY_ATTACK_BULLET_SPEED = 10
ENEMY_ATTACK_BULLET_DELAY = 50
ENEMY_ATTACK_MAX_BULLETS = 2
LIVES = 3
PLAYER_DEATH_DELAY = 50

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
    enemy_attack_direction: chex.Array  # -1: left, 1: right
    enemy_attack_turning: chex.Array    # -1: turning left, 1: turning right, 0: no turning
    enemy_attack_turn_timer: chex.Array
    enemy_attack_respawn_timer: chex.Array
    enemy_attack_bullet_x: chex.Array
    enemy_attack_bullet_y: chex.Array
    enemy_attack_bullet_timer: chex.Array
    lives: chex.Array
    player_alive: chex.Array
    player_respawn_timer: chex.Array

    random_key: chex.Array
    step_counter: chex.Array
    #obs_stack: chex.ArrayTree


@jax.jit
def update_player_position(state: GalaxianState, action) -> GalaxianState:

    press_right = jnp.any(
        jnp.array([action == Action.RIGHT, action == Action.RIGHTFIRE])
    )

    press_left = jnp.any(
        jnp.array([action == Action.LEFT, action == Action.LEFTFIRE])
    )

    new_x = jnp.clip(state.player_x + press_right * 5, 0, SCREEN_WIDTH)
    new_x = jnp.clip(new_x - press_left * 5, 0, SCREEN_WIDTH)
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



@jax.jit
def update_enemy_attack(state: GalaxianState) -> GalaxianState:

    enemy_out_of_bounds = (state.enemy_attack_y > SCREEN_HEIGHT) | (state.enemy_attack_x <= -10) | (state.enemy_attack_x >= SCREEN_WIDTH + 10)
    #jax.debug.print("grid: {}", state.enemy_grid_alive[tuple(state.enemy_attack_pos)])
    #jax.debug.print("state: {}", state.enemy_attack_state)
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
        #jax.debug.print("grid determine: {}", state.enemy_grid_alive[0,0])
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

        #jax.debug.print("new pos: {}, {}", row, column)
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

    #jax.debug.print("direction: {}",state.enemy_attack_direction)
    #jax.debug.print("turning: {}", state.enemy_attack_turning)
    #jax.debug.print("timer: {}", state.enemy_attack_turn_timer)
    new_enemy_attack_direction = jnp.where(
        (state.enemy_attack_turning != 0) & (state.enemy_attack_turn_timer == 0),
        state.enemy_attack_turning,
        state.enemy_attack_direction
    )


    player_right = state.enemy_attack_x < state.player_x
    player_left = state.enemy_attack_x > state.player_x
    new_enemy_attack_turning = jnp.where(
        (state.enemy_attack_turning == 0) & (state.enemy_attack_turn_timer == ENEMY_ATTACK_TURN_TIME) & ((player_right) & (state.enemy_attack_direction == -1) | (player_left) & (state.enemy_attack_direction == 1)),
        -state.enemy_attack_direction,
        jnp.where(
            (state.enemy_attack_turn_timer == 0) | (state.enemy_attack_state == 0),
            0,
            state.enemy_attack_turning)
    )


    new_enemy_attack_turn_timer = jnp.where(
        (state.enemy_attack_turn_timer == 0) | (state.enemy_attack_state == 0),
        ENEMY_ATTACK_TURN_TIME,
        jnp.where(
            state.enemy_attack_turning != 0,
            state.enemy_attack_turn_timer -1,
            state.enemy_attack_turn_timer
        )
    )

    delta_x = jnp.where(
        state.enemy_attack_state == 1,
        ENEMY_ATTACK_SPEED * state.enemy_attack_direction,
        0
    )

    delta_y = jnp.where(
        state.enemy_attack_state == 1,
        int(ENEMY_ATTACK_SPEED/2),
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

    new_enemy_attack_bullet_timer = jnp.where(
        state.enemy_attack_bullet_timer <= 0,
        ENEMY_ATTACK_BULLET_DELAY,
        state.enemy_attack_bullet_timer -1,
    )

    return state._replace(
        enemy_attack_state=new_enemy_attack_state,
        enemy_attack_x=new_enemy_attack_x + delta_x,
        enemy_attack_y=new_enemy_attack_y + delta_y,
        enemy_attack_pos=new_enemy_attack_pos,
        enemy_attack_direction=new_enemy_attack_direction,
        enemy_attack_turning=new_enemy_attack_turning,
        enemy_attack_turn_timer=new_enemy_attack_turn_timer,
        enemy_attack_respawn_timer=new_enemy_attack_respawn_timer,
        enemy_grid_alive=new_enemy_grid_alive,
        enemy_attack_bullet_timer = new_enemy_attack_bullet_timer,
    )

def update_enemy_bullets(state: GalaxianState) -> GalaxianState:

    bullet_available = (state.enemy_attack_bullet_timer == 0) & (state.enemy_attack_bullet_y == -1)
    new_enemy_attack_bullet_x = jnp.where(
        bullet_available,
        state.enemy_attack_x,
        state.enemy_attack_bullet_x
    )
    new_enemy_attack_bullet_y = jnp.where(
        bullet_available,
        state.enemy_attack_y,
        state.enemy_attack_bullet_y
    )

    bullet_exists = (state.enemy_attack_bullet_y != -1)
    new_enemy_attack_bullet_y = jnp.where(
        bullet_exists,
        state.enemy_attack_bullet_y + ENEMY_ATTACK_BULLET_SPEED,
        new_enemy_attack_bullet_y
    )

    bullet_out_of_bounds = (state.enemy_attack_bullet_y > SCREEN_HEIGHT)
    new_enemy_attack_bullet_x = jnp.where(
        bullet_out_of_bounds,
        -1,
        new_enemy_attack_bullet_x
    )
    new_enemy_attack_bullet_y = jnp.where(
        bullet_out_of_bounds,
        -1,
        new_enemy_attack_bullet_y
    )

    return state._replace(
        enemy_attack_bullet_x=new_enemy_attack_bullet_x,
        enemy_attack_bullet_y=new_enemy_attack_bullet_y,
    )



def get_action_from_keyboard():
    keys = pygame.key.get_pressed()
    left = keys[pygame.K_a] or keys[pygame.K_LEFT]
    right = keys[pygame.K_d] or keys[pygame.K_RIGHT]
    shooting = keys[pygame.K_SPACE]

    left_only = left and not right
    right_only = right and not left

    if shooting:
        if left_only:
            return Action.LEFTFIRE
        elif right_only:
            return Action.RIGHTFIRE
        else:
            return Action.FIRE
    else:
        if left_only:
            return Action.LEFT
        elif right_only:
            return Action.RIGHT
        else:
            return Action.NOOP



@jax.jit
def handle_shooting(state: GalaxianState, action) -> GalaxianState:
    def shoot(_):
        return state._replace(
            bullet_x=jnp.array(state.player_x, dtype=state.bullet_x.dtype),
            bullet_y=jnp.array(state.player_y, dtype=state.bullet_y.dtype),
            player_shooting_cooldown=jnp.array(SHOOTING_COOLDOWN)
        )

    def decrease_cooldown(_):
        return state._replace(
            player_shooting_cooldown=state.player_shooting_cooldown - 1
        )

    def idle(_):
        return state


    shooting = jnp.any(
        jnp.array(
            [
                action == Action.FIRE,
                action == Action.RIGHTFIRE,
                action == Action.LEFTFIRE,
            ]
        )
    )

    can_shoot = jnp.logical_and(shooting, state.player_shooting_cooldown == 0)
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
        new_bullet_y = jnp.array(-1,dtype=jnp.float32)
        new_bullet_x = jnp.array(-1,dtype=jnp.float32)
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

@jax.jit
def check_player_death_by_enemy(state: GalaxianState) -> GalaxianState:

    x_diff = jnp.abs(state.player_x - state.enemy_attack_x)
    y_diff = jnp.abs(state.player_y - state.enemy_attack_y)

    collision = (x_diff <= 10) & (y_diff <= 10) & (state.enemy_grid_alive[tuple(state.enemy_attack_pos)] != 0)
    hit = jnp.any(collision)

    def process_hit(operands):
        current_state = operands
        new_lives = current_state.lives - 1
        new_enemy_grid_alive = current_state.enemy_grid_alive.at[tuple(current_state.enemy_attack_pos)].set(0)

        return state._replace(lives=new_lives, enemy_grid_alive=new_enemy_grid_alive)

    return lax.cond(hit, process_hit, lambda _: state, operand=state)

def check_player_death_by_bullet(state: GalaxianState) -> GalaxianState:

    x_diff = jnp.abs(state.player_x - state.enemy_attack_bullet_x)
    y_diff = jnp.abs(state.player_y - state.enemy_attack_bullet_y)

    collision = (x_diff <= 10) & (y_diff <= 10)
    hit = jnp.any(collision)

    def process_hit(operands):
        current_state = operands
        new_lives = current_state.lives - 1
        new_enemy_attack_bullet_x = jnp.array(-1)
        new_enemy_attack_bullet_y = jnp.array(-1)

        return state._replace(lives=new_lives, enemy_attack_bullet_x=new_enemy_attack_bullet_x,enemy_attack_bullet_y=new_enemy_attack_bullet_y)

    return lax.cond(hit, process_hit, lambda _: state, operand=state)

def draw(screen, state):
    # Spieler zeichnen
    player_rect = pygame.Rect(int(state.player_x), int(state.player_y), 20, 10)
    pygame.draw.rect(screen, (0, 255, 0), player_rect)

    # Kugel zeichnen
    if state.bullet_x > -1:
        bullet_rect = pygame.Rect(int(state.bullet_x - 2.5), int(state.bullet_y - 5), 5, 10)
        pygame.draw.rect(screen, (255, 255, 0), bullet_rect)

    # Angreifenden Feind zeichnen
    if jnp.all(state.enemy_attack_pos >= 0):
        enemy_attack_rect = pygame.Rect(int(state.enemy_attack_x - 7.5), int(state.enemy_attack_y - 5), 15, 10)
        pygame.draw.rect(screen, (255, 0, 0), enemy_attack_rect)

   # Feindliche Kugel zeichnen
    if state.enemy_attack_bullet_x > -1:
       bullet_rect = pygame.Rect(int(state.enemy_attack_bullet_x - 2.5), int(state.enemy_attack_bullet_y - 5), 5, 10)
       pygame.draw.rect(screen, (255, 255, 0), bullet_rect)

    # Feindgitter zeichnen
    for i in range(state.enemy_grid_x.shape[0]):
        for j in range(state.enemy_grid_x.shape[1]):
            if state.enemy_grid_alive[i, j] == 1:
                x = int(state.enemy_grid_x[i, j] - 7.5)
                y = int(state.enemy_grid_y[i, j] - 5)
                enemy_rect = pygame.Rect(x, y, 15, 10)
                pygame.draw.rect(screen, (255, 0, 0), enemy_rect)

    # Leben zeichnen
    for i in range(state.lives):
        x = SCREEN_WIDTH - 20 - i * 20
        y = SCREEN_HEIGHT - 20
        pygame.draw.rect(screen, (100, 255, 100), pygame.Rect(x, y, 10, 20))




class GalaxianObservation(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    bullet_x: chex.Array
    bullet_y: chex.Array
    enemy_grid_x: chex.Array
    enemy_grid_y: chex.Array
    enemy_grid_alive: chex.Array  # 0: dead, 1: alive, 2: attacking
    enemy_attack_pos: chex.Array
    enemy_attack_x: chex.Array
    enemy_attack_y: chex.Array

class GalaxianInfo(NamedTuple):
    time: jnp.ndarray
    all_rewards: chex.Array

class JaxGalaxian(JaxEnvironment[GalaxianState, GalaxianObservation, GalaxianInfo]):
    def __init__(self, frameskip: int = 0, reward_funcs: list[callable]=None):
        super().__init__()
        self.frameskip = frameskip + 1  #den stuff kp hab copy paste aus pong
        self.frame_stack_size = 4
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.action_set = {
            Action.NOOP,
            Action.FIRE,
            Action.RIGHT,
            Action.LEFT,
            Action.RIGHTFIRE,
            Action.LEFTFIRE
        }
        self.obs_size = 3*2+1
        #Größe der Beobachtungsdaten
        #3*4 = 3 entitys mit je 2 werten: enemy, player, bullet -> x,y
        # +1 ein extra wert für den score
        # sind erstmal nur temporary values

    def reset(self, key = None) -> Tuple[GalaxianObservation, GalaxianState]:
        grid_rows = GRID_ROWS
        grid_cols = GRID_COLS
        enemy_spacing_x = ENEMY_SPACING_X
        start_x = START_X

        x_positions = jnp.arange(grid_cols) * enemy_spacing_x + start_x  # arange schreibt so 0 1 2 3....
        enemy_grid = jnp.tile(x_positions, (grid_rows, 1))  # kopiert die zeile untereinander

        row_indices = jnp.arange(grid_rows).reshape(-1, 1)  # erzeugt 1. enemy jeder zeile
        enemy_y_rows = ENEMY_GRID_Y - row_indices * ENEMY_SPACING_Y  # jeweils y pos
        enemy_grid_y = jnp.broadcast_to(enemy_y_rows, (
            grid_rows, grid_cols))  # kopiert die werte für 1. enemy auf die rechts in der zeile

        enemy_alive = jnp.ones((grid_rows, grid_cols))  # alles auf 1



        state = GalaxianState(player_x=jnp.array(NATIVE_GAME_WIDTH / 2.0),
                             player_y=jnp.array(NATIVE_GAME_HEIGHT - 20.0),
                             player_shooting_cooldown=jnp.array(0),
                             bullet_x=jnp.array(-1.0,dtype=jnp.float32),
                             bullet_y=jnp.array(-1.0,dtype=jnp.float32),
                             enemy_grid_x=enemy_grid.astype(jnp.float32),
                             enemy_grid_y=enemy_grid_y.astype(jnp.float32),
                             enemy_grid_alive=enemy_alive,
                             enemy_grid_direction=jnp.array(1),
                             enemy_attack_state=jnp.array(0),
                             enemy_attack_pos=jnp.array((-1, -1)),

                             enemy_attack_direction=jnp.array(1),
                             enemy_attack_turning=jnp.array(0),
                             enemy_attack_turn_timer=jnp.array(ENEMY_ATTACK_TURN_TIME),
                             enemy_attack_x=jnp.array(-1.0, dtype=jnp.float32),
                             enemy_attack_y=jnp.array(-1.0, dtype=jnp.float32),
                             enemy_attack_respawn_timer=jnp.array(20),
                             enemy_attack_bullet_x=jnp.array(-1),
                             enemy_attack_bullet_y=jnp.array(-1),
                             #enemy_attack_bullet_x=jnp.full((ENEMY_ATTACK_MAX_BULLETS,), -1),
                             #enemy_attack_bullet_y=jnp.full((ENEMY_ATTACK_MAX_BULLETS,), -1),
                             enemy_attack_bullet_timer = jnp.array(ENEMY_ATTACK_BULLET_DELAY),
                             lives=jnp.array(3),
                             player_alive=jnp.array(True),
                             player_respawn_timer=jnp.array(PLAYER_DEATH_DELAY),
                             random_key=jax.random.PRNGKey(0),
                             step_counter=jnp.array(0),
                             )

        initial_obs = self._get_observation(state)


        # def expand_and_copy(x):
        #     x_expanded = jnp.expand_dims(x, axis=0)
        #     return jnp.concatenate([x_expanded] * self.frame_stack_size, axis=0)
        #
        # # Apply transformation to each leaf in the pytree
        # stacked_obs = jax.tree.map(expand_and_copy, initial_obs)
        #
        # new_state = state._replace(obs_stack=stacked_obs)
        return initial_obs, state

        #alles aus galaxianState was man aus Spielersicht wahrnimmt
    def _get_observation(self, state: GalaxianState) -> GalaxianObservation:
        return GalaxianObservation(
            player_x=state.player_x,
            player_y=state.player_y,
            bullet_x=state.bullet_x,
            bullet_y=state.bullet_y,
            enemy_grid_x=state.enemy_grid_x,
            enemy_grid_y=state.enemy_grid_y,
            enemy_grid_alive=state.enemy_grid_alive,
            enemy_attack_pos=state.enemy_attack_pos,
            enemy_attack_x=state.enemy_attack_x,
            enemy_attack_y=state.enemy_attack_y,
        )

    @partial(jax.jit, static_argnums=(0,))
    def get_action_space(self):
        return jnp.array([Action.NOOP, Action.LEFT, Action.RIGHT])


    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))

    def observation_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=None,
            dtype=jnp.uint8,
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(
            self, state: GalaxianState, action: chex.Array
    ) -> Tuple[GalaxianObservation, GalaxianState, float, bool, GalaxianInfo]:
        #TODO: refactor like the other games
        # create new state instead of replacing old state step by step
        new_state = update_player_position(state, action)
        new_state = handle_shooting(new_state, action)
        new_state = update_enemy_positions(new_state)
        new_state = update_bullets(new_state)
        new_state = remove_bullets(new_state)
        new_state = bullet_collision(new_state)
        new_state = bullet_collision_attack(new_state)
        new_state = update_enemy_attack(new_state)
        new_state = update_enemy_bullets(new_state)
        new_state = check_player_death_by_enemy(new_state)
        new_state = check_player_death_by_bullet(new_state)
        new_state = update_enemy_attack(new_state)  # This was missing from your step
        new_state = new_state._replace(step_counter=new_state.step_counter + 1)

        done = self._get_done(new_state)
        env_reward = self._get_env_reward(state, new_state)  # previous_state is 'state' here
        all_rewards = self._get_all_reward(state, new_state)  # previous_state is 'state' here
        info = self._get_info(new_state, all_rewards)

        observation = self._get_observation(new_state)
        # jax.debug.print("obs: {}", observation) # Be careful with debug prints in jitted functions
        #jax.debug.print("obs: {}", observation)

        # stack the new observation, remove the oldest one
        # observation = jax.tree.map(lambda stack, obs: jnp.concatenate([stack[1:], jnp.expand_dims(obs, axis=0)], axis=0), new_state.obs_stack, observation)
        # new_state = new_state._replace(obs_stack=observation)

        return observation, new_state, env_reward, done, info


    @partial(jax.jit, static_argnums=(0,))
    def _get_env_reward(self, previous_state: GalaxianState, state: GalaxianState):
        return 420 #TODO: implement reward function (after score is implemented)


    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(self, previous_state: GalaxianState, state: GalaxianState) -> chex.Array:
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array(
            [reward_func(previous_state, state) for reward_func in self.reward_funcs]
        )
        return rewards

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: GalaxianState) -> bool:
        return False #TODO: implement done function (test if game over)

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: GalaxianState, all_rewards: chex.Array) -> GalaxianInfo:
        return GalaxianInfo(time=state.step_counter, all_rewards=all_rewards)


def load_sprites():
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

    # load sprites
    bg = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/galaxian/background.npy"),transpose=True)
    player = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/galaxian/player.npy"),transpose=True)
    bullet = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/galaxian/bullet.npy"),transpose=True)
    enemy_white = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/galaxian/white_enemy_1.npy"),transpose=True)  # Assuming you have enemy.npy

    SPRITE_BG = jnp.expand_dims(bg, axis = 0)
    SPRITE_PLAYER = jnp.expand_dims(player, axis = 0)
    SPRITE_BULLET = jnp.expand_dims(bullet, axis = 0)
    SPRITE_ENEMY_WHITE = jnp.expand_dims(enemy_white, axis=0)

    return(
        SPRITE_BG,
        SPRITE_PLAYER,
        SPRITE_BULLET,
        SPRITE_ENEMY_WHITE
    )

class GalaxianRenderer(AtraJaxisRenderer):
    def __init__(self):
        (
        self.SPRITE_BG,
        self.SPRITE_PLAYER,
        self.SPRITE_BULLET,
        self.SPRITE_ENEMY_WHITE
        ) = load_sprites()

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: GalaxianState):  # Added type hint for clarity
        raster = jnp.zeros((NATIVE_GAME_WIDTH, NATIVE_GAME_HEIGHT, 3))

        # Render background
        frame_bg = aj.get_sprite_frame(self.SPRITE_BG, 0)
        raster = aj.render_at(raster, 0, 0, frame_bg)

        # Render player spaceship
        frame_player = aj.get_sprite_frame(self.SPRITE_PLAYER, 0)
        # Ensure player_x and player_y are scalars or 0-dim arrays for render_at
        raster = aj.render_at(raster, jnp.round(state.player_x).astype(jnp.int32),
                              jnp.round(state.player_y).astype(jnp.int32), frame_player)

        # Render bullet
        # Ensure bullet_x and bullet_y are scalars or 0-dim arrays
        # You might need to adjust sprite centering by subtracting half its width/height from pos
        def render_bullet_fn(raster, state):
            frame_bullet = aj.get_sprite_frame(self.SPRITE_BULLET, 0)
            return aj.render_at(raster, jnp.round(state.bullet_x).astype(jnp.int32),
                                jnp.round(state.bullet_y).astype(jnp.int32), frame_bullet)

        def identity_fn(raster, state):
            return raster

        # Conditionally render bullet if it's active (e.g., bullet_y > 0)
        raster = lax.cond(
            state.bullet_y > 0,  # Condition for bullet being active
            render_bullet_fn,
            identity_fn,
            raster,
            state
        )

        # Render enemy grid (this is more complex due to the loop)
        # For jitted rendering of multiple enemies, you'd typically use lax.scan or vmap if possible,
        # or unroll the loop if the number of enemies is fixed and small.
        # A simpler, but potentially less performant if not optimized by JAX, way is a loop:
        # However, Python loops are not JIT-compatible in this way directly for changing raster.
        # For full JAX compatibility, you'd need a more advanced approach.
        # For now, let's demonstrate one enemy for simplicity or how you might structure it.

        # --- Rendering attacking enemy ---
        def render_attacking_enemy_fn(raster, state):
            frame_enemy = aj.get_sprite_frame(self.SPRITE_ENEMY_WHITE, 0)  # Assuming you have an enemy sprite
            return aj.render_at(raster, jnp.round(state.enemy_attack_x).astype(jnp.int32),
                                jnp.round(state.enemy_attack_y).astype(jnp.int32), frame_enemy)

        raster = lax.cond(
            jnp.all(state.enemy_attack_pos >= 0),  # If an enemy is attacking
            render_attacking_enemy_fn,
            identity_fn,  # identity_fn needs to accept raster and state
            raster,
            state
        )

        # --- Rendering enemy grid (conceptual) ---
        # This part is tricky to do efficiently and correctly in a JIT-compiled render function
        # with a dynamic number of alive enemies.
        # One approach for a fixed grid size:
        # Create a mask of alive enemies.
        # Get coordinates of all enemies.
        # Use lax.fori_loop to iterate and render.

        # A simplified version for demonstration (might be slow or not fully JIT-friendly without care):
        # For rendering the grid, you would ideally use lax.fori_loop or lax.scan
        # This is a complex part. For a start, you might focus on one or a few enemies.
        # A full grid rendering with JAX requires careful construction.
        # Example of rendering just the first enemy in the grid if alive:
        def render_first_grid_enemy(params):
            raster_in, state_in, i, j = params
            frame_enemy = aj.get_sprite_frame(self.SPRITE_ENEMY_WHITE, 0)

            # Example: render enemy at grid cell (i,j) if alive
            def render_this_enemy(r):
                return aj.render_at(r,
                                    jnp.round(state_in.enemy_grid_x[i, j]).astype(jnp.int32),
                                    jnp.round(state_in.enemy_grid_y[i, j]).astype(jnp.int32),
                                    frame_enemy)

            def do_nothing(r):
                return r

            return lax.cond(state_in.enemy_grid_alive[i, j] == 1,
                            render_this_enemy(raster_in),
                            do_nothing(raster_in))

        # To iterate over the grid (this is illustrative and needs to be done carefully for JIT)
        # for i in range(GRID_ROWS): # Python loops are problematic for JIT
        #    for j in range(GRID_COLS):
        #        raster = lax.cond(state.enemy_grid_alive[i,j] == 1,
        #                          lambda r: aj.render_at(r, state.enemy_grid_x[i,j], state.enemy_grid_y[i,j], enemy_frame), # pseudo
        #                          lambda r: r,
        #                          raster)

        # A more JAX-idiomatic way for the grid would involve `lax.fori_loop` for rows and columns
        # or `vmap` if positions and sprites can be batched.

        def row_loop_body(i, current_raster):
            def col_loop_body(j, inner_raster):
                frame_enemy = aj.get_sprite_frame(self.SPRITE_ENEMY_WHITE, 0)

                def render_it(r):
                    return aj.render_at(r,
                                        jnp.round(state.enemy_grid_x[i, j]).astype(jnp.int32),
                                        jnp.round(state.enemy_grid_y[i, j]).astype(jnp.int32),
                                        frame_enemy)

                def skip_it(r):
                    return r

                return lax.cond(state.enemy_grid_alive[i, j] == 1,
                                render_it,
                                skip_it,
                                inner_raster)

            return lax.fori_loop(0, GRID_COLS, col_loop_body, current_raster)

        raster = lax.fori_loop(0, GRID_ROWS, row_loop_body, raster)

        return raster


# run with: python -m jaxatari.games.jax_galaxian
if __name__ == "__main__":
    pygame.init()

    game = JaxGalaxian(frameskip=1)
    jitted_step = jax.jit(game.step)
    jitted_reset = jax.jit(game.reset)
    initial_observation, state  = jitted_reset()  # Unpack the tuple

    screen = pygame.display.set_mode((PYGAME_WINDOW_WIDTH, PYGAME_WINDOW_HEIGHT))
    pygame.display.set_caption("Galaxian")
    clock = pygame.time.Clock()

    renderer = GalaxianRenderer()

    # Get jitted functions (reset doesn't need to be jitted if only called once)
    # state, initial_observation = game.reset() # Call reset directly
    state, initial_observation = jax.jit(game.reset)() # If you want to jit it

    jitted_step = game.step # Already jitted using partial in class

    running = True
    while running:
        action_int = Action.NOOP # Default action
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.fill((0, 0, 0))
        draw(screen, state)
        pygame.display.flip()
        action = get_action_from_keyboard()
        state, observation, reward, done, info = jitted_step(state, action)  # Unpack the tuple
        render_output = renderer.render(state)
        aj.update_pygame(screen, render_output, PYGAME_SCALE_FACTOR, NATIVE_GAME_WIDTH,
                         NATIVE_GAME_HEIGHT)  # Assuming scale factor 1

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()




