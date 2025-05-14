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

# frame (window) constants
FRAME_WIDTH = 152
FRAME_HEIGHT = 206

# field constants (the actual tennis court)
# top: (left_x = 32, right_x = 111, width = right_x - left_x = 79)
FIELD_WIDTH_TOP = 79
# bottom: (left_x = 16, right_x = 127, width = right_x - left_x = 111)
FIELD_WIDTH_BOTTOM = 111
# top_y = 44, bottom_y = 174, height = bottom_y - top_y = 130
FIELD_HEIGHT = 130

# game constants (these values are used for the actual gameplay calculations)
GAME_OFFSET_LEFT_BOTTOM = 15 + 1 # don't use 16, because that is on the line and playing on the line still counts todo: check if that is actually true
GAME_OFFSET_TOP = 43.0 # don't use 44, because that is on the line and playing on the line still counts todo: check if that is actually true
GAME_WIDTH = FIELD_WIDTH_BOTTOM
GAME_HEIGHT = FIELD_HEIGHT

# player constants
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

# ball constants
BALL_GRAVITY_PER_FRAME = 1.1
BALL_SERVING_BOUNCE_VELOCITY_BASE = 21
BALL_SERVING_BOUNCE_VELOCITY_RANDOM_OFFSET = 2
BALL_WIDTH = 2.0

rand_key = random.PRNGKey(0)

class BallState(NamedTuple):
    ball_x: chex.Array # x-coordinate of the ball
    ball_y: chex.Array # y-coordinate of the ball
    ball_z: chex.Array # z-coordinate of the ball
    ball_z_fp: chex.Array # z-coordinate of the ball with exactly one point (effectively ball_z * 10, used for calculations)
    ball_velocity_z_fp: chex.Array # z-velocity of the ball with exactly one point
    ball_hit_start_x: chex.Array # x-coordinate of the location where the ball was last hit
    ball_hit_start_y: chex.Array # y-coordinate of the location where the ball was last hit
    ball_hit_target_x: chex.Array # x-coordinate of the location where the ball was last aimed towards
    ball_hit_target_y: chex.Array # y-coordinate of the location where the ball was last aimed towards
    move_x: chex.Array # Normalized distance from ball_x to ball_hit_target_x, updated on hit
    move_y: chex.Array # Normalized distance from ball_y to ball_hit_target_y, updated on hit

class PlayerState(NamedTuple):
    player_x: chex.Array # x-coordinate of the player
    player_y: chex.Array # y-coordinate of the player
    player_direction: chex.Array # direction the player is currently facing in todo explain which values can happen here in what they mean
    player_field: chex.Array #top or bottom field

class EnemyState(NamedTuple):
    enemy_x: chex.Array # x-coordinate of the enemy
    enemy_y: chex.Array # y-coordinate of the enemy

class TennisState(NamedTuple):
    is_serving: chex.Array = False # whether the game is currently in serving state (ball bouncing on one side until player hits)
    player_state: PlayerState = PlayerState( # all player-related data
        jnp.array(PLAYER_START_X),
        jnp.array(PLAYER_START_Y),
        jnp.array(PLAYER_START_DIRECTION),
        jnp.array(PLAYER_START_FIELD)
    )
    enemy_state: EnemyState = EnemyState( # all enemy-related data
        jnp.array(0),
        jnp.array(0)
    )
    ball_state: BallState = BallState( # all ball-related data
        jnp.array(GAME_WIDTH / 2.0 - 2.5),
        jnp.array(GAME_OFFSET_TOP),
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(GAME_WIDTH / 2.0 - 2.5),
        jnp.array(GAME_OFFSET_TOP),
        0.0,
        0.0
    )
    counter: chex.Array = jnp.array(0) # not currently used, just a counter that is increased by one each frame todo evaluate if we can remove this

def tennis_step(state: TennisState, action) -> TennisState:
    """
    Updates the entire state of the game by calling all step functions.

    Args:
        state (TennisState): The current state of the game.
        action: The action to apply.

    Returns:
        TennisState: The updated state of the game.
    """

    new_ball_state = ball_step(state, action)
    new_player_state = player_step(state, action)

    return TennisState(state.is_serving, new_player_state, state.enemy_state, new_ball_state, state.counter + 1)

# todo needs docs
def player_step(state: TennisState, action: chex.Array) -> PlayerState:
    player_state = update_player_pos(state.player_state, action)
    # todo add turning etc.
    return player_state

def update_player_pos(state: PlayerState, action: chex.Array) -> PlayerState:
    """
    Updates player position based on provided action and applies bounding box.

    Args:
        state (PlayerState): The current player state.
        action (chex.Array): The action to apply.

    Returns:
        PlayerState: The updated player state.
    """

    # does the action contain UP
    up = jnp.any(
        jnp.array(
            [action == JAXAtariAction.UP, action == JAXAtariAction.UPRIGHT, action == JAXAtariAction.UPLEFT, action == JAXAtariAction.UPFIRE, action == JAXAtariAction.UPRIGHTFIRE,
                            action == JAXAtariAction.UPLEFTFIRE]
        )
    )
    # does the action contain DOWN
    down = jnp.any(
        jnp.array(
            [action == JAXAtariAction.DOWN, action == JAXAtariAction.DOWNRIGHT, action == JAXAtariAction.DOWNLEFT, action == JAXAtariAction.DOWNFIRE, action == JAXAtariAction.DOWNRIGHTFIRE,
             action == JAXAtariAction.DOWNLEFTFIRE]
        )
    )
    # does the action contain LEFT
    left = jnp.any(
        jnp.array(
            [action == JAXAtariAction.LEFT, action == JAXAtariAction.UPLEFT, action == JAXAtariAction.DOWNLEFT, action == JAXAtariAction.LEFTFIRE, action == JAXAtariAction.UPLEFTFIRE,
             action == JAXAtariAction.DOWNLEFTFIRE]
        )
    )
    # does the action contain RIGHT
    right = jnp.any(
        jnp.array(
            [action == JAXAtariAction.RIGHT, action == JAXAtariAction.UPRIGHT, action == JAXAtariAction.DOWNRIGHT, action == JAXAtariAction.RIGHTFIRE, action == JAXAtariAction.UPRIGHTFIRE,
             action == JAXAtariAction.DOWNRIGHTFIRE]
        )
    )

    # move left if the player is trying to move left
    player_x = jnp.where(
        left,
        state.player_x - 1,
        state.player_x,
    )
    # move right if the player is trying to move right
    player_x = jnp.where(
        right,
        state.player_x + 1,
        player_x,
    )
    # apply X bounding box
    player_x = jnp.clip(player_x, PLAYER_MIN_X, PLAYER_MAX_X)

    # move up if the player is trying to move up
    player_y = jnp.where(
        up,
        state.player_y - 1,
        state.player_y,
    )
    # move down if the player is trying to move down
    player_y = jnp.where(
        down,
        state.player_y + 1,
        player_y,
    )
    # apply Y bounding box
    player_y = jnp.where(
        state.player_field == 1,
        jnp.clip(player_y, PLAYER_Y_UPPER_BOUND_TOP, PLAYER_Y_LOWER_BOUND_TOP),
        jnp.clip(player_y, PLAYER_Y_UPPER_BOUND_BOTTOM, PLAYER_Y_LOWER_BOUND_BOTTOM)
    )

    return PlayerState(
        player_x,
        player_y,
        state.player_direction,
        state.player_field
    )

def ball_step(state: TennisState, action) -> BallState:
    """
    Updates ball position by applying velocity and gravity. Also handles player-ball collisions
    and fires the ball if the provided action contains FIRE.

    Args:
        state (TennisState): The current state of the game.
        action (chex.Array): The action to apply.

    Returns:
        BallState: The updated ball state.
    """

    def get_serving_bounce_velocity() -> int:
        """
        Applies a random offset to the base serving bounce velocity.

        Returns:
            int: The calculated bounce velocity
        """
        return BALL_SERVING_BOUNCE_VELOCITY_BASE + random.uniform(rand_key) * BALL_SERVING_BOUNCE_VELOCITY_RANDOM_OFFSET

    ball_state = state.ball_state

    # update the fixed-point z velocity value by applying either upward velocity or gravity
    new_ball_velocity_z_fp = jnp.where(
        ball_state.ball_z == 0,
        get_serving_bounce_velocity(),
        ball_state.ball_velocity_z_fp - BALL_GRAVITY_PER_FRAME
    )

    # update the fixed-point z value by applying current velocity (can be positive or negative)
    new_ball_z_fp = ball_state.ball_z_fp + new_ball_velocity_z_fp
    # calculate actual z value using floor division by 10, because fixed-point value has exactly one point
    new_ball_z = new_ball_z_fp // 10

    # apply lower bounding box
    new_ball_z = jnp.clip(new_ball_z, 0, new_ball_z)
    new_ball_z_fp = jnp.clip(new_ball_z_fp, 0, new_ball_z_fp)

    # ball movement in x/y direction is linear, no velocity involved
    new_ball_x = jnp.where(
        ball_state.ball_x != ball_state.ball_hit_target_x,
        ball_state.ball_x + ball_state.move_x,
        ball_state.ball_x
    )
    new_ball_y = jnp.where(
        ball_state.ball_y != ball_state.ball_hit_target_y,
        ball_state.ball_y + ball_state.move_y,
        ball_state.ball_y
    )

    player_state = state.player_state

    # check player-ball collisions
    player_end = player_state.player_x + PLAYER_WIDTH
    ball_end = ball_state.ball_x + BALL_WIDTH
    player_overlap_ball_x = jnp.logical_not(
        jnp.logical_or(
            player_end <= ball_state.ball_x,
            ball_end <= player_state.player_x
        )
    )

    # check if ball should be served or hit regularly todo is there even a difference?
    should_serve = jnp.logical_and(
        action == JAXAtariAction.FIRE, # todo allow for other FIRE actions, like LEFTFIRE
        state.is_serving
    )

    return jax.lax.cond(
        player_overlap_ball_x,
        lambda _: jax.lax.cond(
            should_serve, lambda _: handle_ball_serve(state), lambda _: handle_ball_fire(state), None
        ),
        lambda _: BallState(
            new_ball_x,
            new_ball_y,
            new_ball_z,
            new_ball_z_fp,
            new_ball_velocity_z_fp,
            ball_state.ball_hit_start_x,
            ball_state.ball_hit_start_y,
            ball_state.ball_hit_target_x,
            ball_state.ball_hit_target_y,
            ball_state.move_x,
            ball_state.move_y
        ),
        None
    )

# todo needs docs
def handle_ball_fire(state: TennisState) -> BallState:
    new_ball_hit_start_x = state.ball_state.ball_x
    new_ball_hit_start_y = state.ball_state.ball_y

    # todo fix hardcoded values
    # todo this is incorrect, it assumes the player_x is in the center of the player
    ball_width = 2.0
    max_dist = PLAYER_WIDTH / 2 + ball_width / 2

    angle = -1 * (((state.player_state.player_x + PLAYER_WIDTH / 2) - (state.ball_state.ball_x + 2 / 2)) / max_dist)
    # calc x landing position depending on player hit angle
    # angle = 0 # neutral angle, between -1...1
    left_offset = -39
    right_offset = 39
    offset = ((angle + 1) / 2) * (right_offset - left_offset) + left_offset

    new_ball_hit_target_x = new_ball_hit_start_x + offset
    new_ball_hit_target_y = new_ball_hit_start_y + 91

    hit_vel = 24.0

    dx = new_ball_hit_target_x - state.ball_state.ball_x
    dy = new_ball_hit_target_y - state.ball_state.ball_y
    #dist = jnp.sqrt(dx**2 + dy**2) + 1e-8  # Add epsilon to avoid divide-by-zero
    dist = jnp.linalg.norm(jnp.array([dx, dy])) + 1e-8

    norm_dx = dx / dist
    norm_dy = dy / dist

    move_x = norm_dx * 2.1
    move_y = norm_dy * 2.1

    return BallState(
        state.ball_state.ball_x,
        state.ball_state.ball_y,
        14.0,
        140.0,
        hit_vel,
        new_ball_hit_start_x,
        new_ball_hit_start_y,
        new_ball_hit_target_x,
        new_ball_hit_target_y,
        move_x,
        move_y
    )

# todo needs docs
def handle_ball_serve(state: TennisState) -> BallState:
    #new_ball_x = state.player_x
    #new_ball_y = state.player_y

    new_ball_hit_start_x = state.ball_state.ball_x
    new_ball_hit_start_y = state.ball_state.ball_y

    # todo fix hardcoded values
    # todo this is incorrect, it assumes the player_x is in the center of the player
    ball_width = 2.0
    max_dist = PLAYER_WIDTH / 2 + ball_width / 2

    angle = -1 * (((state.player_state.player_x + PLAYER_WIDTH / 2) - (state.ball_state.ball_x + 2 / 2)) / max_dist)
    # calc x landing position depending on player hit angle
    #angle = 0 # neutral angle, between -1...1
    left_offset = -39
    right_offset = 39
    offset = ((angle + 1) / 2) * (right_offset - left_offset) + left_offset

    new_ball_hit_target_x = new_ball_hit_start_x + offset
    new_ball_hit_target_y = new_ball_hit_start_y + 91

    return BallState(
        state.ball_state.ball_x,
        state.ball_state.ball_y,
        state.ball_state.ball_z,
        state.ball_state.ball_z_fp,
        state.ball_state.ball_velocity_z_fp,
        new_ball_hit_start_x,
        new_ball_hit_start_y,
        new_ball_hit_target_x,
        new_ball_hit_target_y,
        state.ball_state.move_x,
        state.ball_state.move_y
    )

def tennis_reset() -> TennisState:
    """
    Provides the initial state for the game. For that purpose, we use the default values assigned in TennisState.

    Returns:
        TennisState: The initial state of the game.
    """
    return TennisState()

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
