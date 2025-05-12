import pygame
import tennis_renderer as renderer
from jaxatari.environment import JAXAtariAction
from jaxatari.rendering import atraJaxis as aj
from typing import NamedTuple
import chex
import jax.lax
from functools import partial
import jax.numpy as jnp
import jax.random as random

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

PLAYER_WIDTH = 13 # player flips side so total covered x section is greater
PLAYER_HEIGHT = 23

PLAYER_MIN_X = 10#17
PLAYER_MAX_X = 130#142

# lower y-axis values are towards the top in our case, opposite in original game
PLAYER_Y_LOWER_BOUND_BOTTOM = 180#206-2-PLAYER_HEIGHT
PLAYER_Y_UPPER_BOUND_BOTTOM = 109#206-53-PLAYER_HEIGHT
PLAYER_Y_LOWER_BOUND_TOP = 72#206-91-PLAYER_HEIGHT
PLAYER_Y_UPPER_BOUND_TOP = 0#206-148-PLAYER_HEIGHT
PLAYER_START_X = 20
PLAYER_START_Y = 20
PLAYER_START_DIRECTION = 1 # 1 right, -1 left
PLAYER_START_FIELD = 1 #1 top, -1 bottom

rand_key = random.PRNGKey(0)

class BallState(NamedTuple):
    ball_x: chex.Array
    ball_y: chex.Array
    ball_z: chex.Array
    ball_z_fp: chex.Array
    ball_velocity_z_fp: chex.Array
    ball_hit_start_x: chex.Array
    ball_hit_start_y: chex.Array
    ball_hit_target_x: chex.Array
    ball_hit_target_y: chex.Array

class PlayerState(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    player_direction: chex.Array
    player_field: chex.Array #top or bottom field

class TennisState(NamedTuple):
    player_state: PlayerState = PlayerState(jnp.array(PLAYER_START_X), jnp.array(PLAYER_START_Y), jnp.array(PLAYER_START_DIRECTION), jnp.array(PLAYER_START_FIELD))
    ball_state: BallState = BallState(jnp.array(GAME_WIDTH / 2.0 - 2.5), jnp.array(0.0), jnp.array(0.0), jnp.array(0.0), jnp.array(0.0), jnp.array(0.0), jnp.array(0.0), jnp.array(GAME_WIDTH / 2.0 - 2.5), jnp.array(0.0))
    counter : chex.Array = jnp.array(0)

#@partial(jax.jit, static_argnums=(0,))
def tennis_step(state: TennisState, action) -> TennisState:
    new_ball_state = ball_step(state, action)
    new_player_state = player_step(state, action)

    return TennisState(new_player_state, new_ball_state, state.counter + 1)
    # new_player_x = jnp.where(state.player_x < FRAME_WIDTH, state.player_x + 1, state.player_x - 1)
    #new_ball_x = jnp.where(state.ball_direction == 0, state.ball_x + 1, state.ball_x - 1)
    #new_ball_z = jnp.where(state.ball_z_direction == 0, state.ball_z + 1, state.ball_z - 1)

    #new_z_direction = jnp.where(state.ball_z >= 10, 1, state.ball_z_direction)
    #new_z_direction = jnp.where(state.ball_z <= 0, 0, new_z_direction)

    #new_direction = jnp.where(state.ball_x >= GAME_WIDTH, 1, state.ball_direction)
    #new_direction = jnp.where(state.ball_x <= 0, 0, new_direction)

    #new_ball_y = jnp.where(state.counter % 4 == 0, state.ball_y + 1, state.ball_y)

    #return TennisState(state.player_x, state.player_y, new_ball_x, new_ball_y, ball_z=new_ball_z, ball_direction=new_direction, ball_z_direction=new_z_direction, counter=state.counter + 1)

def player_step(state: TennisState, action: chex.Array) -> PlayerState:
    """Update the player position based on coodinates and action"""
    player_state = update_player_pos(state.player_state, action)
    # todo add turning etc.
    return player_state


def update_player_pos(state: PlayerState, action: chex.Array) -> PlayerState:
    up = jnp.any(jnp.array([action == JAXAtariAction.UP, action == JAXAtariAction.UPRIGHT, action == JAXAtariAction.UPLEFT, action == JAXAtariAction.UPFIRE, action == JAXAtariAction.UPRIGHTFIRE,
                            action == JAXAtariAction.UPLEFTFIRE]))
    down = jnp.any(jnp.array(
        [action == JAXAtariAction.DOWN, action == JAXAtariAction.DOWNRIGHT, action == JAXAtariAction.DOWNLEFT, action == JAXAtariAction.DOWNFIRE, action == JAXAtariAction.DOWNRIGHTFIRE,
         action == JAXAtariAction.DOWNLEFTFIRE]))
    left = jnp.any(jnp.array(
        [action == JAXAtariAction.LEFT, action == JAXAtariAction.UPLEFT, action == JAXAtariAction.DOWNLEFT, action == JAXAtariAction.LEFTFIRE, action == JAXAtariAction.UPLEFTFIRE,
         action == JAXAtariAction.DOWNLEFTFIRE]))
    right = jnp.any(jnp.array(
        [action == JAXAtariAction.RIGHT, action == JAXAtariAction.UPRIGHT, action == JAXAtariAction.DOWNRIGHT, action == JAXAtariAction.RIGHTFIRE, action == JAXAtariAction.UPRIGHTFIRE,
         action == JAXAtariAction.DOWNRIGHTFIRE]))

    # check if the player is trying to move left
    player_x = jnp.where(
        left,
        state.player_x - 1,
        state.player_x,
    )
    # check if the player is trying to move right
    player_x = jnp.where(
        right,
        state.player_x + 1,
        player_x,
    )

    player_x = jnp.clip(player_x, PLAYER_MIN_X, PLAYER_MAX_X)

    # check if the player is trying to move up
    player_y = jnp.where(
        up,
        state.player_y - 1,
        state.player_y,
    )

    # check if the player is trying to move down
    player_y = jnp.where(
        down,
        state.player_y + 1,
        player_y,
    )
    player_y = jnp.where(state.player_field == 1 , jnp.clip(player_y, PLAYER_Y_UPPER_BOUND_TOP, PLAYER_Y_LOWER_BOUND_TOP), jnp.clip(player_y, PLAYER_Y_UPPER_BOUND_BOTTOM, PLAYER_Y_LOWER_BOUND_BOTTOM))
    return PlayerState(player_x, player_y, state.player_direction, state.player_field)


def ball_step(state: TennisState, action) -> BallState:
    # 2.2 is initial velocity, 0.11 is gravity per frame
    ball_state = state.ball_state
    new_ball_velocity_z_fp = jnp.where(ball_state.ball_z == 0, 21 + random.uniform(rand_key) * 2, ball_state.ball_velocity_z_fp - 1.1)

    new_ball_z_fp = ball_state.ball_z_fp + new_ball_velocity_z_fp
    new_ball_z = new_ball_z_fp // 10

    new_ball_z = jnp.where(new_ball_z <= 0, 0, new_ball_z)
    new_ball_z_fp = jnp.where(new_ball_z <= 0, 0, new_ball_z_fp)

    dx = ball_state.ball_hit_target_x - ball_state.ball_x
    dy = ball_state.ball_hit_target_y - ball_state.ball_y
    dist = jnp.sqrt(dx**2 + dy**2) + 1e-8  # Add epsilon to avoid divide-by-zero

    norm_dx = dx / dist
    norm_dy = dy / dist

    new_ball_x = jnp.where(ball_state.ball_x != ball_state.ball_hit_target_x, ball_state.ball_x + norm_dx, ball_state.ball_x)
    new_ball_y = jnp.where(ball_state.ball_y != ball_state.ball_hit_target_y, ball_state.ball_y + norm_dy,
                           ball_state.ball_y)
    #new_ball_state = jnp.where(action == JAXAtariAction.FIRE, handle_ball_fire(state), state.ball_state)
    #BallState(new_ball_state.ball_x, new_ball_state.ball_y, new_ball_z, new_ball_z_fp, new_ball_velocity_z_fp, new_ball_state.new_ball_hit_start_x, new_ball_state.new_ball_hit_start_y, new_ball_state.new_ball_hit_target_x, new_ball_state.new_ball_hit_target_y)

    # todo fix hardcoded values (2 is ball width, 5 is player width)
    player_state = state.player_state

    player_overlap_ball_x = jnp.logical_or(
        jnp.logical_or(
            jnp.logical_and(
                player_state.player_x >= ball_state.ball_x - 1,
                player_state.player_x <= ball_state.ball_x + 2 + 1
            ),
            jnp.logical_and(
                ball_state.ball_x >= player_state.player_x - 1,
                ball_state.ball_x <= player_state.player_x + 5 + 1
            )
        ),
        jnp.logical_or(
            jnp.logical_and(
                player_state.player_x + 5 >= ball_state.ball_x - 1,
                player_state.player_x + 5 <= ball_state.ball_x + 2 + 1
            ),
            jnp.logical_and(
                ball_state.ball_x + 2 >= player_state.player_x - 1,
                ball_state.ball_x + 2 <= player_state.player_x + 5 + 1
            )
        )
    )
    should_fire = jnp.logical_and(
        action == JAXAtariAction.FIRE,
        player_overlap_ball_x
    )

    return jax.lax.cond(
        should_fire, lambda _: handle_ball_fire(state), lambda _: BallState(new_ball_x, new_ball_y, new_ball_z, new_ball_z_fp, new_ball_velocity_z_fp, ball_state.ball_hit_start_x, ball_state.ball_hit_start_y, ball_state.ball_hit_target_x, ball_state.ball_hit_target_y), None
    )


def handle_ball_fire(state: TennisState) -> BallState:
    #new_ball_x = state.player_x
    #new_ball_y = state.player_y

    new_ball_hit_start_x = state.ball_state.ball_x
    new_ball_hit_start_y = state.ball_state.ball_y

    # todo fix hardcoded values
    player_width = 5.0
    ball_width = 2.0
    max_dist = player_width / 2 + ball_width / 2

    angle = -1 * ((state.player_state.player_x - state.ball_state.ball_x) / max_dist)
    # calc x landing position depending on player hit angle
    #angle = 0 # neutral angle, between -1...1
    left_offset = -39
    right_offset = 39
    offset = ((angle + 1) / 2) * (right_offset - left_offset) + left_offset

    new_ball_hit_target_x = new_ball_hit_start_x + offset
    new_ball_hit_target_y = new_ball_hit_start_y + 80

    return BallState(state.ball_state.ball_x, state.ball_state.ball_y, state.ball_state.ball_z, state.ball_state.ball_z_fp, state.ball_state.ball_velocity_z_fp, new_ball_hit_start_x, new_ball_hit_start_y, new_ball_hit_target_x, new_ball_hit_target_y)

def tennis_reset() -> TennisState:
    player_state = PlayerState(jnp.array(PLAYER_START_X), jnp.array(PLAYER_START_Y), jnp.array(PLAYER_START_DIRECTION), jnp.array(PLAYER_START_FIELD))
    ball_state = BallState(jnp.array(GAME_WIDTH / 2.0 - 2.5), jnp.array(0.0), jnp.array(0.0), jnp.array(0.0), jnp.array(0.0), jnp.array(0.0), jnp.array(0.0), jnp.array(GAME_WIDTH / 2.0 - 2.5), jnp.array(0.0))
    return TennisState(player_state, ball_state, jnp.array(0))

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
        current_state = jitted_step(current_state, 0)

        # Render and display
        raster = renderer.render(current_state)

        aj.update_pygame(screen, raster, 3, FRAME_WIDTH, FRAME_HEIGHT)

        clock.tick(30)

    pygame.quit()
