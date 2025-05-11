import pygame
import tennis_renderer as renderer
from jaxatari.rendering import atraJaxis as aj
from jaxatari.environment import JAXAtariAction
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

rand_key = random.PRNGKey(0)

class BallState(NamedTuple):
    ball_x: chex.Array
    ball_y: chex.Array
    ball_z: chex.Array
    ball_z_fp: chex.Array
    ball_velocity_z_fp: chex.Array

class TennisState(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    ball_state: chex.Array = BallState(50, 0, 0, 0, 0)
    counter : chex.Array = 0

#@partial(jax.jit, static_argnums=(0,))
def tennis_step(state: TennisState, action) -> TennisState:
    new_ball_state = ball_step(state.ball_state)
    new_state_after_player_step = player_step(state, action)

    return TennisState(new_state_after_player_step.player_x, new_state_after_player_step.player_y, new_ball_state, state.counter + 1)
    # new_player_x = jnp.where(state.player_x < FRAME_WIDTH, state.player_x + 1, state.player_x - 1)
    #new_ball_x = jnp.where(state.ball_direction == 0, state.ball_x + 1, state.ball_x - 1)
    #new_ball_z = jnp.where(state.ball_z_direction == 0, state.ball_z + 1, state.ball_z - 1)

    #new_z_direction = jnp.where(state.ball_z >= 10, 1, state.ball_z_direction)
    #new_z_direction = jnp.where(state.ball_z <= 0, 0, new_z_direction)

    #new_direction = jnp.where(state.ball_x >= GAME_WIDTH, 1, state.ball_direction)
    #new_direction = jnp.where(state.ball_x <= 0, 0, new_direction)

    #new_ball_y = jnp.where(state.counter % 4 == 0, state.ball_y + 1, state.ball_y)

    #return TennisState(state.player_x, state.player_y, new_ball_x, new_ball_y, ball_z=new_ball_z, ball_direction=new_direction, ball_z_direction=new_z_direction, counter=state.counter + 1)

def ball_step(state: BallState) -> BallState:
    """
    Generate a symmetrical Atari-style jump arc using fixed-point math.

    Parameters:
    - v0_fp: initial velocity in float (e.g., 2.2)
    - gravity_fp: gravity per frame in float (e.g., 0.12)
    - total_frames: number of frames to simulate (default 42)
    - scale: fixed-point scale factor (e.g., 10 means 1.0 = 10)

    Returns:
    - List of integer Z positions for each frame, ending with two 0s
    """
    # 2.2 is initial velocity, 0.11 is gravity per frame
    new_ball_velocity_z_fp = jnp.where(state.ball_z == 0, 21 + random.uniform(rand_key) * 2, state.ball_velocity_z_fp - 1.1)

    new_ball_z_fp = state.ball_z_fp + new_ball_velocity_z_fp
    new_ball_z = new_ball_z_fp // 10

    new_ball_z = jnp.where(new_ball_z <= 0, 0, new_ball_z)
    new_ball_z_fp = jnp.where(new_ball_z <= 0, 0, new_ball_z_fp)

    return BallState(state.ball_x, state.ball_y, new_ball_z, new_ball_z_fp, new_ball_velocity_z_fp)

def player_step(state: TennisState, action) -> TennisState:
    should_move_right = jnp.logical_and(
        action == JAXAtariAction.RIGHT,
        state.player_x <= GAME_WIDTH,
    )

    should_move_left = jnp.logical_and(
        action == JAXAtariAction.LEFT,
        state.player_x >= 0,
    )

    new_player_x = jnp.where(should_move_right, state.player_x + 1, state.player_x)
    new_player_x = jnp.where(should_move_left, state.player_x - 1, new_player_x)

    should_move_up = jnp.logical_and(
        action == JAXAtariAction.UP,
        state.player_y >= 0,
    )

    should_move_down = jnp.logical_and(
        action == JAXAtariAction.DOWN,
        state.player_y<= GAME_HEIGHT,
    )

    new_player_y = jnp.where(should_move_up, state.player_y - 1, state.player_y)
    new_player_y = jnp.where(should_move_down, state.player_y + 1, new_player_y)

    return TennisState(new_player_x, new_player_y, state.ball_state, state.counter)

def tennis_reset() -> TennisState:
    return TennisState(0.0, 100.0)

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
