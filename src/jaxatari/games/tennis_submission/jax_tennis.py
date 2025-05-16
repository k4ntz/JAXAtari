import pygame
from jaxatari.environment import JAXAtariAction
from jaxatari.rendering import atraJaxis as aj
from typing import NamedTuple
import chex
import jax.lax
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
GAME_OFFSET_LEFT_BOTTOM = 15 + 1  # don't use 16, because that is on the line and playing on the line still counts todo: check if that is actually true
GAME_OFFSET_TOP = 43.0  # don't use 44, because that is on the line and playing on the line still counts todo: check if that is actually true
GAME_WIDTH = FIELD_WIDTH_BOTTOM
GAME_HEIGHT = FIELD_HEIGHT
GAME_MIDDLE = GAME_OFFSET_TOP + 0.5 * GAME_HEIGHT
GAME_OFFSET_BOTTOM = GAME_OFFSET_TOP + GAME_HEIGHT
PAUSE_DURATION = 100

# player constants
PLAYER_WIDTH = 13  # player flips side so total covered x section is greater
PLAYER_HEIGHT = 23
PLAYER_MIN_X = 10  # 17
PLAYER_MAX_X = 130  # 142
# lower y-axis values are towards the top in our case, opposite in original game
PLAYER_Y_LOWER_BOUND_BOTTOM = 180  # 206-2-PLAYER_HEIGHT
PLAYER_Y_UPPER_BOUND_BOTTOM = 109  # 206-53-PLAYER_HEIGHT
PLAYER_Y_LOWER_BOUND_TOP = 72  # 206-91-PLAYER_HEIGHT
PLAYER_Y_UPPER_BOUND_TOP = 0  # 206-148-PLAYER_HEIGHT

START_X = GAME_OFFSET_LEFT_BOTTOM + 0.5 * GAME_WIDTH - 0.5 * PLAYER_WIDTH
PLAYER_START_Y = GAME_OFFSET_TOP - PLAYER_HEIGHT
ENEMY_START_Y = GAME_OFFSET_BOTTOM - PLAYER_HEIGHT

PLAYER_START_DIRECTION = 1  # 1 right, -1 left
PLAYER_START_FIELD = 1  # 1 top, -1 bottom

# ball constants
BALL_GRAVITY_PER_FRAME = 1.1
BALL_SERVING_BOUNCE_VELOCITY_BASE = 21
BALL_SERVING_BOUNCE_VELOCITY_RANDOM_OFFSET = 2
BALL_WIDTH = 2.0

rand_key = random.key(0)


class BallState(NamedTuple):
    ball_x: chex.Array  # x-coordinate of the ball
    ball_y: chex.Array  # y-coordinate of the ball
    ball_z: chex.Array  # z-coordinate of the ball
    ball_z_fp: chex.Array  # z-coordinate of the ball with exactly one point (effectively ball_z * 10, used for calculations)
    ball_velocity_z_fp: chex.Array  # z-velocity of the ball with exactly one point
    ball_hit_start_x: chex.Array  # x-coordinate of the location where the ball was last hit
    ball_hit_start_y: chex.Array  # y-coordinate of the location where the ball was last hit
    ball_hit_target_x: chex.Array  # x-coordinate of the location where the ball was last aimed towards
    ball_hit_target_y: chex.Array  # y-coordinate of the location where the ball was last aimed towards
    move_x: chex.Array  # Normalized distance from ball_x to ball_hit_target_x, updated on hit
    move_y: chex.Array  # Normalized distance from ball_y to ball_hit_target_y, updated on hit
    bounces: chex.Array  # how many times the ball has hit the ground since it was last hit by an entity


class PlayerState(NamedTuple):
    player_x: chex.Array  # x-coordinate of the player
    player_y: chex.Array  # y-coordinate of the player
    player_direction: chex.Array  # direction the player is currently facing in todo explain which values can happen here in what they mean
    player_field: chex.Array  # top or bottom field


class EnemyState(NamedTuple):
    enemy_x: chex.Array  # x-coordinate of the enemy
    enemy_y: chex.Array  # y-coordinate of the enemy


class GameState(NamedTuple):
    is_serving: chex.Array  # whether the game is currently in serving state (ball bouncing on one side until player hits)
    pause_counter: chex.Array  # delay between restart of game
    player_score: chex.Array
    enemy_score: chex.Array


class TennisState(NamedTuple):
    player_state: PlayerState = PlayerState(  # all player-related data
        jnp.array(START_X),
        jnp.array(PLAYER_START_Y),
        jnp.array(PLAYER_START_DIRECTION),
        jnp.array(PLAYER_START_FIELD)
    )
    enemy_state: EnemyState = EnemyState(  # all enemy-related data
        jnp.array(START_X),
        jnp.array(ENEMY_START_Y)
    )
    ball_state: BallState = BallState(  # all ball-related data
        jnp.array(GAME_WIDTH / 2.0 - 2.5),
        jnp.array(GAME_OFFSET_TOP),
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(GAME_WIDTH / 2.0 - 2.5),
        jnp.array(GAME_OFFSET_TOP),
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(0),
    )
    game_state: GameState = GameState(
        jnp.array(True),
        jnp.array(0),
        jnp.array(0),
        jnp.array(0)
    )
    counter: chex.Array = jnp.array(
        0)  # not currently used, just a counter that is increased by one each frame todo evaluate if we can remove this

@jax.jit
def normal_step(state: TennisState, action) -> TennisState:
    new_state_after_score_check = check_score(state)
    new_state_after_ball_step = ball_step(new_state_after_score_check, action)
    new_player_state = player_step(new_state_after_ball_step, action)
    new_enemy_state = enemy_step(new_state_after_ball_step)
    return TennisState(new_player_state, new_enemy_state, new_state_after_ball_step.ball_state,
                       new_state_after_ball_step.game_state, new_state_after_ball_step.counter + 1)

@jax.jit
def tennis_step(state: TennisState, action) -> TennisState:
    """
    Updates the entire state of the game by calling all step functions.

    Args:
        state (TennisState): The current state of the game.
        action: The action to apply.

    Returns:
        TennisState: The updated state of the game.
    """

    return jax.lax.cond(state.game_state.pause_counter > 0,
                        lambda _: TennisState(state.player_state, state.enemy_state, state.ball_state,
                                              GameState(
                                                  state.game_state.is_serving,
                                                  state.game_state.pause_counter - 1,
                                                  state.game_state.player_score,
                                                  state.game_state.enemy_score),
                                              state.counter + 1),
                        lambda _: normal_step(state, action),
                        None
                        )

@jax.jit
def check_score(state: TennisState) -> TennisState:
    """
    Checks whether a point was scored.
    """

    new_bounces = jnp.where(
        jnp.logical_and(state.ball_state.ball_z <= 0, jnp.logical_not(state.game_state.is_serving)),
        # ball is at z=0 and not serving
        state.ball_state.bounces + 1,
        state.ball_state.bounces
    )

    # update the scores and start pause of game
    new_game_state = jax.lax.cond(
        jnp.logical_or(
            jnp.logical_and(  # If player is top field and ball is bottom field the player scores
                state.ball_state.ball_y >= GAME_MIDDLE,
                state.player_state.player_field == 1
            ),
            jnp.logical_and(  # If player is bottom field and ball is top field the player scores
                state.ball_state.ball_y <= GAME_MIDDLE,
                state.player_state.player_field == -1
            )
        ),
        lambda _: GameState(jnp.array(True), jnp.array(PAUSE_DURATION), state.game_state.player_score + 1,
                            state.game_state.enemy_score),
        lambda _: GameState(jnp.array(True), jnp.array(PAUSE_DURATION), state.game_state.player_score,
                            state.game_state.enemy_score + 1),
        None
    )

    return jax.lax.cond(
        new_bounces > 2,
        lambda _: TennisState(
            PlayerState(
                jnp.array(START_X),
                jnp.where(state.player_state.player_field == 1, jnp.array(PLAYER_START_Y),
                          jnp.array(ENEMY_START_Y)),
                jnp.array(PLAYER_START_DIRECTION),
                state.player_state.player_field
            ),
            EnemyState(
                jnp.array(START_X),
                jnp.where(state.player_state.player_field == 1, jnp.array(ENEMY_START_Y),
                          jnp.array(PLAYER_START_Y))
            ),
            BallState(
                jnp.array(GAME_WIDTH / 2.0 - 2.5),
                jnp.array(GAME_OFFSET_TOP),
                jnp.array(0.0),
                jnp.array(0.0),
                jnp.array(0.0),
                jnp.array(0.0),
                jnp.array(0.0),
                jnp.array(GAME_WIDTH / 2.0 - 2.5),
                jnp.array(GAME_OFFSET_TOP),
                jnp.array(0.0),
                jnp.array(0.0),
                jnp.array(0)
            ),
            new_game_state,
            state.counter
        ),
        lambda _: TennisState(state.player_state, state.enemy_state, BallState(  # no one has scored yet
            state.ball_state.ball_x,
            state.ball_state.ball_y,
            state.ball_state.ball_z,
            state.ball_state.ball_z_fp,
            state.ball_state.ball_velocity_z_fp,
            state.ball_state.ball_hit_start_x,
            state.ball_state.ball_hit_start_y,
            state.ball_state.ball_hit_target_x,
            state.ball_state.ball_hit_target_y,
            state.ball_state.move_x,
            state.ball_state.move_y,
            new_bounces,
        ), state.game_state, state.counter),
        None
    )


# todo needs docs
@jax.jit
def player_step(state: TennisState, action: chex.Array) -> PlayerState:
    player_state = update_player_pos(state.player_state, action)
    # todo add turning etc.
    return player_state

@jax.jit
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
            [action == JAXAtariAction.UP, action == JAXAtariAction.UPRIGHT, action == JAXAtariAction.UPLEFT,
             action == JAXAtariAction.UPFIRE, action == JAXAtariAction.UPRIGHTFIRE,
             action == JAXAtariAction.UPLEFTFIRE]
        )
    )
    # does the action contain DOWN
    down = jnp.any(
        jnp.array(
            [action == JAXAtariAction.DOWN, action == JAXAtariAction.DOWNRIGHT, action == JAXAtariAction.DOWNLEFT,
             action == JAXAtariAction.DOWNFIRE, action == JAXAtariAction.DOWNRIGHTFIRE,
             action == JAXAtariAction.DOWNLEFTFIRE]
        )
    )
    # does the action contain LEFT
    left = jnp.any(
        jnp.array(
            [action == JAXAtariAction.LEFT, action == JAXAtariAction.UPLEFT, action == JAXAtariAction.DOWNLEFT,
             action == JAXAtariAction.LEFTFIRE, action == JAXAtariAction.UPLEFTFIRE,
             action == JAXAtariAction.DOWNLEFTFIRE]
        )
    )
    # does the action contain RIGHT
    right = jnp.any(
        jnp.array(
            [action == JAXAtariAction.RIGHT, action == JAXAtariAction.UPRIGHT, action == JAXAtariAction.DOWNRIGHT,
             action == JAXAtariAction.RIGHTFIRE, action == JAXAtariAction.UPRIGHTFIRE,
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

@jax.jit
def enemy_step(state: TennisState) -> EnemyState:
    new_enemy_x = jnp.where(
        state.enemy_state.enemy_x + PLAYER_WIDTH / 2 < state.ball_state.ball_x,
        state.enemy_state.enemy_x + 1,
        state.enemy_state.enemy_x
    )

    new_enemy_x = jnp.where(
        state.enemy_state.enemy_x + PLAYER_WIDTH / 2 > state.ball_state.ball_x,
        state.enemy_state.enemy_x - 1,
        new_enemy_x
    )

    return EnemyState(
        new_enemy_x,
        state.enemy_state.enemy_y
    )

@jax.jit
def ball_step(state: TennisState, action) -> TennisState:
    """
    Updates ball position by applying velocity and gravity. Also handles player-ball collisions
    and fires the ball if the provided action contains FIRE.

    Args:
        state (TennisState): The current state of the game.
        action (chex.Array): The action to apply.

    Returns:
        BallState: The updated ball state.
    """

    @jax.jit
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

    # apply lower bounding box (500 is effectively no MAX bound)d
    new_ball_z = jnp.clip(new_ball_z, 0, 500)
    new_ball_z_fp = jnp.clip(new_ball_z_fp, 0, 500)

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
    enemy_state = state.enemy_state

    upper_entity_x = jnp.where(
        player_state.player_field == 1,
        player_state.player_x,
        enemy_state.enemy_x
    )
    upper_entity_y = jnp.where(
        player_state.player_field == 1,
        player_state.player_y,
        enemy_state.enemy_y
    )

    lower_entity_x = jnp.where(
        player_state.player_field == -1,
        player_state.player_x,
        enemy_state.enemy_x
    )
    lower_entity_y = jnp.where(
        player_state.player_field == -1,
        player_state.player_y,
        enemy_state.enemy_y
    )

    upper_entity_overlapping_ball = jnp.logical_and(
        is_overlapping(
            upper_entity_x,
            PLAYER_WIDTH,
            0,  # this is z pos
            PLAYER_HEIGHT,
            ball_state.ball_x,
            BALL_WIDTH,
            ball_state.ball_z,
            BALL_WIDTH  # todo rename to BALL_SIZE because ball is square
        ),
        jnp.absolute(upper_entity_y + PLAYER_HEIGHT - ball_state.ball_y) <= 3
    )

    lower_entity_overlapping_ball = jnp.logical_and(
        is_overlapping(
            lower_entity_x,
            PLAYER_WIDTH,
            0,
            PLAYER_HEIGHT,
            ball_state.ball_x,
            BALL_WIDTH,
            ball_state.ball_z,
            BALL_WIDTH
        ),
        jnp.absolute(lower_entity_y + PLAYER_HEIGHT - ball_state.ball_y) <= 3
    )

    # check if fire is pressed
    fire = jnp.any(jnp.array(
        [action == JAXAtariAction.FIRE, action == JAXAtariAction.LEFTFIRE, action == JAXAtariAction.DOWNLEFTFIRE,
         action == JAXAtariAction.DOWNFIRE,
         action == JAXAtariAction.DOWNRIGHTFIRE, action == JAXAtariAction.RIGHTFIRE,
         action == JAXAtariAction.UPRIGHTFIRE, action == JAXAtariAction.UPFIRE,
         action == JAXAtariAction.UPLEFTFIRE]))

    any_collision = jnp.logical_or(
        upper_entity_overlapping_ball,
        lower_entity_overlapping_ball
    )
    should_hit = jnp.logical_and(any_collision, jnp.logical_or(jnp.logical_not(state.game_state.is_serving), fire))
    new_is_serving = jnp.where(should_hit, False, state.game_state.is_serving)

    # no need to check whether the lower entity is actually overlapping because this variable won't be used if it isn't
    ball_fire_direction = jnp.where(
        upper_entity_overlapping_ball,
        1,
        -1
    )
    # no need to check whether the lower entity is actually overlapping because this variable won't be used if it isn't
    hitting_entity_x = jnp.where(
        upper_entity_overlapping_ball,
        upper_entity_x,
        lower_entity_x
    )

    ball_state_after_fire = jax.lax.cond(
        should_hit,
        lambda _: handle_ball_fire(state, hitting_entity_x, ball_fire_direction),
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
            ball_state.move_y,
            ball_state.bounces
        ),
        None
    )

    return TennisState(
        player_state,
        enemy_state,
        BallState(
            new_ball_x,
            new_ball_y,
            ball_state_after_fire.ball_z,
            ball_state_after_fire.ball_z_fp,
            ball_state_after_fire.ball_velocity_z_fp,
            ball_state_after_fire.ball_hit_start_x,
            ball_state_after_fire.ball_hit_start_y,
            ball_state_after_fire.ball_hit_target_x,
            ball_state_after_fire.ball_hit_target_y,
            ball_state_after_fire.move_x,
            ball_state_after_fire.move_y,
            ball_state_after_fire.bounces
        ),
        GameState(
            new_is_serving,
            state.game_state.pause_counter,
            state.game_state.player_score,
            state.game_state.enemy_score,
        ),
        state.counter
    )

@jax.jit
def is_overlapping(entity1_x, entity1_w, entity1_y, entity1_h, entity2_x, entity2_w, entity2_y,
                   entity2_h) -> chex.Array:
    entity1_end_x = entity1_x + entity1_w
    entity2_end_x = entity2_x + entity2_w
    is_overlapping_x = jnp.logical_not(
        jnp.logical_or(
            entity1_end_x <= entity2_x,
            entity2_end_x <= entity1_x
        )
    )

    entity1_end_y = entity1_y + entity1_h
    entity2_end_y = entity2_y + entity2_h
    is_overlapping_y = jnp.logical_not(
        jnp.logical_or(
            entity1_end_y <= entity2_y,
            entity2_end_y <= entity1_y
        )
    )

    return jnp.logical_and(is_overlapping_x, is_overlapping_y)


# todo needs docs
@jax.jit
def handle_ball_fire(state: TennisState, hitting_entity_x, direction) -> BallState:
    # direction = 1 from top side to bottom
    # direction = -1 from bottom side to top
    # direction = 0 (dont do this)
    new_ball_hit_start_x = state.ball_state.ball_x
    new_ball_hit_start_y = state.ball_state.ball_y

    # todo fix hardcoded values
    # todo this is incorrect, it assumes the player_x is in the center of the player
    ball_width = 2.0
    max_dist = PLAYER_WIDTH / 2 + ball_width / 2

    angle = -1 * (((hitting_entity_x + PLAYER_WIDTH / 2) - (state.ball_state.ball_x + 2 / 2)) / max_dist) * direction
    # calc x landing position depending on player hit angle
    # angle = 0 # neutral angle, between -1...1
    left_offset = -39
    right_offset = 39
    offset = ((angle + 1) / 2) * (right_offset - left_offset) + left_offset

    new_ball_hit_target_y = new_ball_hit_start_y + (91 * direction)
    field_min_x = 32
    field_max_x = 32 + FIELD_WIDTH_TOP
    new_ball_hit_target_x = jnp.clip(new_ball_hit_start_x + offset, field_min_x, field_max_x)

    hit_vel = 24.0

    dx = new_ball_hit_target_x - state.ball_state.ball_x
    dy = new_ball_hit_target_y - state.ball_state.ball_y
    # dist = jnp.sqrt(dx**2 + dy**2) + 1e-8  # Add epsilon to avoid divide-by-zero
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
        move_y,
        jnp.array(0)  # ball has not bounced after last hit
    )


# todo needs docs
@jax.jit
def handle_ball_serve(state: TennisState) -> BallState:
    # new_ball_x = state.player_x
    # new_ball_y = state.player_y

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
        state.ball_state.move_y,
        jnp.array(0)
    )

@jax.jit
def tennis_reset() -> TennisState:
    """
    Provides the initial state for the game. For that purpose, we use the default values assigned in TennisState.

    Returns:
        TennisState: The initial state of the game.
    """
    return TennisState()

"""
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

    pygame.quit()"""

# ---------------------- RENDERER --------------------------------------------------------------#

from functools import partial
import jax
import jax.numpy as jnp
import jaxatari.rendering.atraJaxis as aj
import os
# Todo remove
import csv, uuid

def load_sprites():
    MODULE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    BG = jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "tennis_submission/sprites/background.npy")), axis=0)
    BALL = jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "tennis_submission/sprites/ball.npy")), axis=0)
    BALL_SHADOW = jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "tennis_submission/sprites/ball_shadow.npy")), axis=0)

    return BG, BALL, BALL_SHADOW


# ToDo remove
def get_bounding_box(width, height, border_color=(255, 0, 0, 255), border_thickness=1):
    """
    Creates a bounding box sprite with transparent interior and colored border.

    Args:
        width: Width of the bounding box (static)
        height: Height of the bounding box (static)
        border_color: RGBA tuple for border color (default: red)
        border_thickness: Pixel width of the border (default: 1)

    Returns:
        Array of shape (height, width, 4) with RGBA values
    """
    # Create empty RGBA array
    sprite = jnp.zeros((width, height, 4), dtype=jnp.uint8)

    # Create border mask
    border_mask = jnp.zeros((width, height), dtype=bool)

    # Set border pixels to True
    border_mask = border_mask.at[:border_thickness, :].set(True)  # Top border
    border_mask = border_mask.at[-border_thickness:, :].set(True)  # Bottom border
    border_mask = border_mask.at[:, :border_thickness].set(True)  # Left border
    border_mask = border_mask.at[:, -border_thickness:].set(True)  # Right border

    # Apply border color where mask is True
    sprite = jnp.where(
        border_mask[..., None],  # Expand for RGBA channels
        jnp.array(border_color, dtype=jnp.uint8),
        sprite
    )

    return sprite[jnp.newaxis, ...]


def perspective_transform(x, y, apply_offsets = True, width_top = 79.0, width_bottom = 111.0, height = 130.0):
    # Normalize y: 0 at top (far), 1 at bottom (near)
    y_norm = y / height

    # Interpolate width at this y level
    current_width = width_top * (1 - y_norm) + width_bottom * y_norm

    # Horizontal offset to center the perspective slice
    offset = (width_bottom - current_width) / 2
    #offset = 0

    # Normalize x based on bottom width (field space)
    x_norm = x / width_bottom

    # Compute final x position
    x_screen = offset + x_norm * current_width
    y_screen = y  # No vertical scaling

    if apply_offsets:
        return x_screen + GAME_OFFSET_LEFT_BOTTOM, y_screen + GAME_OFFSET_TOP
    return x_screen, y_screen



class TennisRenderer:

    def __init__(self):
        (self.BG, self.BALL, self.BALL_SHADOW) = load_sprites()
        # use bounding box as mockup
        self.PLAYER = get_bounding_box(PLAYER_WIDTH, PLAYER_HEIGHT)

    #@partial(jax.jit, static_argnums=(0,))
    def render(self, state: TennisState) -> jnp.ndarray:
        raster = jnp.zeros((FRAME_WIDTH, FRAME_HEIGHT, 3))

        frame_bg = aj.get_sprite_frame(self.BG, 0)
        raster = aj.render_at(raster, 0, 0, frame_bg)

        frame_ball_shadow = aj.get_sprite_frame(self.BALL_SHADOW, 0)
        # calculate screen coordinates of ball
        #ball_screen_x, ball_screen_y = self.perspective_transform(state.ball_state.ball_x, state.ball_state.ball_y)
        raster = aj.render_at(raster, state.ball_state.ball_x, state.ball_state.ball_y, frame_ball_shadow)

        frame_ball = aj.get_sprite_frame(self.BALL, 0)
        # apply flat y offset depending on z value
        raster = aj.render_at(raster, state.ball_state.ball_x, state.ball_state.ball_y - state.ball_state.ball_z, frame_ball)

        # Prepare CSV output
        #os.makedirs("saved_traces", exist_ok=True)
        #csv_path = os.path.join("saved_traces", f"{os.getpid()}.csv")

        #with open(csv_path, "a", newline="") as csvfile:
        #    writer = csv.writer(csvfile)
        #    writer.writerow([state.ball_state.ball_x, state.ball_state.ball_y, state.ball_state.ball_z])

        #print("x ball: {:0.2f}., y ball: {:0.2f}, z ball: {:0.2f}, vel: {:0.2f}".format(state.ball_state.ball_x, state.ball_state.ball_y, state.ball_state.ball_z, state.ball_state.ball_velocity_z_fp))

        #player_screen_x, player_screen_y = self.perspective_transform(state.player_state.player_x, state.player_state.player_y)
        #player_screen_x, player_screen_y = self.perspective_transform(0, -20)
        frame_player = aj.get_sprite_frame(self.PLAYER, 0)
        raster = aj.render_at(raster, state.player_state.player_x, state.player_state.player_y, frame_player)

        raster = aj.render_at(raster, state.enemy_state.enemy_x, state.enemy_state.enemy_y, frame_player)

        player_x_rec = jnp.zeros((2, 2, 4))
        player_x_rec = player_x_rec.at[:, :, 1].set(255)  # Yellow
        player_x_rec = player_x_rec.at[:, :, 2].set(255)  # Yellow
        player_x_rec = player_x_rec.at[:, :, 3].set(255)  # Alpha

        raster = aj.render_at(raster, state.player_state.player_x, state.player_state.player_y, player_x_rec)

        #print(state.player_state.player_x, state.player_state.player_y)
        #frame_player = aj.get_sprite_frame(self.PLAYER, 0)
        #raster = aj.render_at(raster, 0, 0, frame_player)

        # visualize perspective transform
        """for i in range(0, GAME_HEIGHT):
            rec_left, _ = self.perspective_transform2(0, i)
            rec_right, _ = self.perspective_transform2(GAME_WIDTH, i)
            rec_width = rec_right - rec_left
            rectangle = jnp.zeros((int(rec_width), 1, 4))
            rectangle = rectangle.at[:, :, 0].set(255)  # Red
            rectangle = rectangle.at[:, :, 3].set(255)  # Alpha

            raster = aj.render_at(raster, i + GAME_OFFSET_TOP, rec_left + 2, rectangle)"""

        #raster = aj.render_at(raster, screen_x, screen_y, frame_ball)
        """
        player_rec = jnp.zeros((5, 5, 4))
        player_rec = player_rec.at[:, :, 1].set(255)  # Yellow
        player_rec = player_rec.at[:, :, 0].set(255)  # Yellow
        player_rec = player_rec.at[:, :, 3].set(255)  # Alpha
        player_coords = self.perspective_transform(state.player_x, state.player_y)

        raster = aj.render_at(raster, player_coords[0], player_coords[1], player_rec)"""

        top_left_rec = jnp.zeros((2, 2, 4))
        top_left_rec = top_left_rec.at[:, :, 1].set(255)  # Yellow
        top_left_rec = top_left_rec.at[:, :, 0].set(255)  # Yellow
        top_left_rec = top_left_rec.at[:, :, 3].set(255)  # Alpha
        top_left_corner_coords = perspective_transform(0, 0)

        raster = aj.render_at(raster, top_left_corner_coords[0], top_left_corner_coords[1], top_left_rec)

        top_right_rec = jnp.zeros((2, 2, 4))
        top_right_rec = top_right_rec.at[:, :, 2].set(255)  # Blue
        top_right_rec = top_right_rec.at[:, :, 3].set(255)  # Alpha
        top_right_corner_coords = perspective_transform(GAME_WIDTH, 0)

        raster = aj.render_at(raster, top_right_corner_coords[0], top_right_corner_coords[1], top_right_rec)

        bottom_left_rec = jnp.zeros((2, 2, 4))
        bottom_left_rec = bottom_left_rec.at[:, :, 1].set(255)  # Green
        bottom_left_rec = bottom_left_rec.at[:, :, 3].set(255)  # Alpha
        bottom_left_corner_coords = perspective_transform(0, GAME_HEIGHT)

        raster = aj.render_at(raster, bottom_left_corner_coords[0], bottom_left_corner_coords[1], bottom_left_rec)

        bottom_right_rec = jnp.zeros((2, 2, 4))
        bottom_right_rec = bottom_right_rec.at[:, :, 0].set(255)  # Red
        bottom_right_rec = bottom_right_rec.at[:, :, 3].set(255)  # Alpha
        bottom_right_corner_coords = perspective_transform(GAME_WIDTH, GAME_HEIGHT)

        raster = aj.render_at(raster, bottom_right_corner_coords[0], bottom_right_corner_coords[1], bottom_right_rec)

        #raster = aj.render_at(raster, 0, 0, rectangle)

        return raster

    # we always use coordinates including the lines


# ---------------JAX ENVIRONMENT----------------------------------------------------------------------------------------

from jaxatari.environment import JaxEnvironment, EnvState, EnvObs, EnvInfo
from typing import NamedTuple, Tuple
import jax.numpy as jnp

from jaxatari.renderers import AtraJaxisRenderer

renderer = TennisRenderer()

class TennisObs(NamedTuple):
    some_shit: jnp.ndarray = 0

class TennisInfo(NamedTuple):
    pass

class AtraJaxisTennisRenderer(AtraJaxisRenderer):

    def render(self, state: TennisState):
        return renderer.render(state)

class TennisJaxEnv(JaxEnvironment[TennisState, TennisObs, TennisInfo]):

    def reset(self, key) -> Tuple[TennisObs, TennisState]:
        reset_state = tennis_reset()
        reset_obs = self._get_observation(reset_state)

        return reset_obs, reset_state

    def step( self, state: TennisState, action) -> Tuple[TennisObs, TennisState, float, bool, TennisInfo]:
        new_state = tennis_step(state, action)
        new_obs = self._get_observation(new_state)
        reward = self._get_reward(state, new_state)
        done = self._get_done(new_state)
        info = self._get_info(new_state)

        return new_obs, new_state, reward, done, info

    def render(self, state: TennisState) -> Tuple[jnp.ndarray]:
        return renderer.render(state)

    def get_action_space(self) -> jnp.ndarray:
        return jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])

    def get_observation_space(self) -> Tuple:
        pass

    def _get_observation(self, state: TennisState) -> TennisObs:
        return TennisObs()

    def _get_info(self, state: TennisState) -> TennisInfo:
        return TennisInfo()

    def _get_reward(self, previous_state: TennisState, state: TennisState) -> float:
        return 0.0

    def _get_done(self, state: TennisState) -> bool:
        return False

