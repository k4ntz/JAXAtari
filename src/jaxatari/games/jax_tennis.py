import pygame
from jaxatari.environment import JAXAtariAction
import chex
import jax.lax
import jax.random as random
import jax
import jaxatari.rendering.jax_rendering_utils as aj
import os
import chex
from jaxatari.environment import JaxEnvironment, EnvState, EnvObs, EnvInfo
from typing import NamedTuple, Tuple
import jax.numpy as jnp
import jaxatari.spaces as spaces

from jaxatari.renderers import JAXGameRenderer

# frame (window) constants
FRAME_WIDTH = 160  # 152
FRAME_HEIGHT = 210  # 206

# field constants (the actual tennis court)
# top: (left_x = 32, right_x = 111, width = right_x - left_x = 79)
FIELD_WIDTH_TOP = 79
# bottom: (left_x = 16, right_x = 127, width = right_x - left_x = 111)
FIELD_WIDTH_BOTTOM = 111
# top_y = 44, bottom_y = 174, height = bottom_y - top_y = 130
FIELD_HEIGHT = 130

# game constants (these values are used for the actual gameplay calculations)
GAME_OFFSET_LEFT_BOTTOM = 15 + 1  # don't use 16, because that is on the line and playing on the line still counts
GAME_OFFSET_TOP = 43.0  # don't use 44, because that is on the line and playing on the line still counts
GAME_WIDTH = FIELD_WIDTH_BOTTOM
GAME_HEIGHT = FIELD_HEIGHT
GAME_MIDDLE = GAME_OFFSET_TOP + 0.5 * GAME_HEIGHT
GAME_OFFSET_BOTTOM = GAME_OFFSET_TOP + GAME_HEIGHT
PAUSE_DURATION = 100

# player constants
PLAYER_CONST = 0
PLAYER_WIDTH = 13  # player flips side so total covered x section is greater
PLAYER_HEIGHT = 23
PLAYER_MIN_X = 8  # left movement restriction
PLAYER_MAX_X = 144  # right movement restriction
# lower y-axis values are towards the top in our case, opposite in original game
PLAYER_Y_LOWER_BOUND_BOTTOM = 160  # 180  # 206-2-PLAYER_HEIGHT
PLAYER_Y_UPPER_BOUND_BOTTOM = 109  # 206-53-PLAYER_HEIGHT
PLAYER_Y_LOWER_BOUND_TOP = 70  # 72  # 206-91-PLAYER_HEIGHT
PLAYER_Y_UPPER_BOUND_TOP = 15  # 35  # 206-148-PLAYER_HEIGHT

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
LONG_HIT_THRESHOLD_TOP = 52
LONG_HIT_THRESHOLD_BOTTOM = 121
LONG_HIT_DISTANCE = 91  # previously 91
SHORT_HIT_DISTANCE = 40

# enemy constants
ENEMY_CONST = 1

GAME_MIDDLE_HORIZONTAL = (FRAME_WIDTH - PLAYER_WIDTH) / 2

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
    last_hit: chex.Array  # 0 if last hit was performed by player, 1 if last hit was by enemy


class PlayerState(NamedTuple):
    player_x: chex.Array  # x-coordinate of the player
    player_y: chex.Array  # y-coordinate of the player
    player_direction: chex.Array  # direction the player is currently facing in (-1: look towards left, 1: look towards right)
    player_field: chex.Array  # top (1) or bottom (-1) field
    player_serving: chex.Array  # true: Player is serving, false: Enemy is serving


class EnemyState(NamedTuple):
    enemy_x: chex.Array  # x-coordinate of the enemy
    enemy_y: chex.Array  # y-coordinate of the enemy
    prev_walking_direction: chex.Array  # previous walking direction (in x-direction) of the enemy, -1 = towards x=min, 1 = towards x=max
    enemy_direction: chex.Array
    y_movement_direction: chex.Array  # direction in which the enemy is moving until it hits the ball, -1 towards net 1 away from net


class GameState(NamedTuple):
    is_serving: chex.Array  # whether the game is currently in serving state (ball bouncing on one side until player hits)
    pause_counter: chex.Array  # delay between restart of game
    player_score: chex.Array  # The score line within the current set (goes up in increments of 1, instead of traditional tennis counting)
    enemy_score: chex.Array
    player_game_score: chex.Array  # Number of won sets
    enemy_game_score: chex.Array
    is_finished: chex.Array  # True if the game is finished (Player or enemy has won the game)


class AnimatorState(NamedTuple):
    player_frame: chex.Array = 0
    enemy_frame: chex.Array = 0
    player_racket_frame: chex.Array = 0
    player_racket_animation: chex.Array = False
    enemy_racket_frame: chex.Array = 0
    enemy_racket_animation: chex.Array = False


class TennisState(NamedTuple):
    player_state: PlayerState = PlayerState(  # all player-related data
        jnp.array(START_X),
        jnp.array(PLAYER_START_Y),
        jnp.array(PLAYER_START_DIRECTION),
        jnp.array(PLAYER_START_FIELD),
        jnp.array(True)
    )
    enemy_state: EnemyState = EnemyState(  # all enemy-related data
        jnp.array(START_X),
        jnp.array(ENEMY_START_Y),
        jnp.array(0.0),
        jnp.array(PLAYER_START_DIRECTION),  # could create ENEMY_START_DIRECTION constant
        jnp.array(1)
    )
    ball_state: BallState = BallState(  # all ball-related data
        jnp.array(FRAME_WIDTH / 2.0 - BALL_WIDTH / 2),
        jnp.array(GAME_OFFSET_TOP),
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(FRAME_WIDTH / 2.0 - BALL_WIDTH / 2),
        jnp.array(GAME_OFFSET_TOP),
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(0),
        jnp.array(-1)
    )
    game_state: GameState = GameState(
        jnp.array(True),
        jnp.array(0),
        jnp.array(0),
        jnp.array(0),
        jnp.array(0),
        jnp.array(0),
        jnp.array(False),
    )
    counter: chex.Array = jnp.array(
        0)
    animator_state: AnimatorState = AnimatorState()


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

    return jax.lax.cond(state.game_state.is_finished,
                        lambda _: state,
                        lambda _: jax.lax.cond(state.game_state.pause_counter > 0,
                                               lambda _: TennisState(state.player_state, state.enemy_state,
                                                                     state.ball_state,
                                                                     GameState(
                                                                         state.game_state.is_serving,
                                                                         state.game_state.pause_counter - 1,
                                                                         state.game_state.player_score,
                                                                         state.game_state.enemy_score,
                                                                         state.game_state.player_game_score,
                                                                         state.game_state.enemy_game_score,
                                                                         state.game_state.is_finished,
                                                                     ),
                                                                     state.counter + 1),
                                               lambda _: normal_step(state, action),
                                               None
                                               ),
                        None
                        )


@jax.jit
def normal_step(state: TennisState, action) -> TennisState:
    """
    Updates the entire state of the game by calling all step functions. Should be used when game is not paused.

    Args:
        state (TennisState): The current state of the game.
        action: The action to apply.

    Returns:
        TennisState: The updated state of the game.
    """

    new_state_after_score_check = check_score(state)
    new_state_after_ball_step = ball_step(new_state_after_score_check, action)
    new_player_state = player_step(new_state_after_ball_step, action)
    new_enemy_state = enemy_step(new_state_after_ball_step)
    new_animator_state = animator_step(state, new_player_state, new_enemy_state)

    return TennisState(new_player_state, new_enemy_state, new_state_after_ball_step.ball_state,
                       new_state_after_ball_step.game_state, state.counter + 1, new_animator_state)


@jax.jit
def animator_step(state: TennisState, new_player_state, new_enemy_state) -> AnimatorState:
    """
    Updates the animator state depending on the current player and enemy state.

    Args:
        state (TennisState): The current state of the game.
        new_player_state (PlayerState): The new player state, recorded 1 frame after `state`.
        new_enemy_state (EnemyState): The enew enemy state, recorded 1 frame after `state`.

    Returns:
        AnimatorState: The updated animator state.
    """

    @jax.jit
    def check_has_moved(prev_x, cur_x, prev_y, cur_y):
        return jnp.logical_or(
            prev_x != cur_x,
            prev_y != cur_y,
        )

    # Animations for player
    has_player_moved = check_has_moved(state.player_state.player_x, new_player_state.player_x,
                                       state.player_state.player_y, new_player_state.player_y)
    new_player_frame = jnp.where(
        # only update if player has moved, otherwise reset to frame 0
        has_player_moved,
        jnp.where(
            # only update every 4 ticks
            state.counter % 4 == 0,
            state.animator_state.player_frame + 1,
            state.animator_state.player_frame,
        ),
        0
    )

    # if ball is at x/y coordinates of hit location, it was hit in this frame
    was_ball_hit = jnp.logical_and(
        state.ball_state.ball_x == state.ball_state.ball_hit_start_x,
        state.ball_state.ball_y == state.ball_state.ball_hit_start_y,
    )
    player_hit = jnp.logical_or(
        jnp.logical_and(
            state.ball_state.ball_y < GAME_MIDDLE,
            state.player_state.player_field == 1
        ),
        jnp.logical_and(
            state.ball_state.ball_y > GAME_MIDDLE,
            state.player_state.player_field == -1
        )
    )

    # set player_racket_animation to True if ball was hit
    new_player_racket_animation = jnp.logical_or(
        jnp.logical_and(
            was_ball_hit,
            player_hit
        ),
        state.animator_state.player_racket_animation,
    )
    new_enemy_racket_animation = jnp.logical_or(
        jnp.logical_and(
            was_ball_hit,
            jnp.logical_not(player_hit)
        ),
        state.animator_state.enemy_racket_animation,
    )

    new_player_racket_frame = jnp.where(
        jnp.logical_and(
            new_player_racket_animation,
            state.counter % 4 == 0,
        ),
        (state.animator_state.player_racket_frame + 1) % 4,
        state.animator_state.player_racket_frame,
    )

    new_enemy_racket_frame = jnp.where(
        jnp.logical_and(
            new_enemy_racket_animation,
            state.counter % 4 == 0,
        ),
        (state.animator_state.enemy_racket_frame + 1) % 4,
        state.animator_state.enemy_racket_frame,
    )

    # if animation is over (has reached start frame again), set player_racket_animation to False
    new_player_racket_animation = jnp.where(
        jnp.logical_and(
            new_player_racket_frame == 0,
            state.counter % 4 == 0,
        ),
        False,
        new_player_racket_animation,
    )

    new_enemy_racket_animation = jnp.where(
        jnp.logical_and(
            new_enemy_racket_frame == 0,
            state.counter % 4 == 0,
        ),
        False,
        new_enemy_racket_animation,
    )

    # Animations for enemy
    has_enemy_moved = check_has_moved(state.enemy_state.enemy_x, new_enemy_state.enemy_x, state.enemy_state.enemy_y,
                                      new_enemy_state.enemy_y)
    new_enemy_frame = jnp.where(
        # only update if enemy has moved, otherwise reset to frame 0
        has_enemy_moved,
        jnp.where(
            # only update every 4 ticks
            state.counter % 4 == 0,
            state.animator_state.enemy_frame + 1,
            state.animator_state.enemy_frame,
        ),
        0
    )

    new_animator_state = AnimatorState(new_player_frame % 4, new_enemy_frame % 4, new_player_racket_frame,
                                       new_player_racket_animation, new_enemy_racket_frame, new_enemy_racket_animation)

    return new_animator_state


@jax.jit
def check_score(state: TennisState) -> TennisState:
    """
    Checks whether a point was scored and updates the state accordingly. Handles re-spawning, changing sides and all necessary restart behavior.

    Args:
        state (TennisState): The current state of the game.

    Returns:
        TennisState: The updated state of the game.
    """

    new_bounces = jnp.where(
        jnp.logical_and(state.ball_state.ball_z <= 0, jnp.logical_not(state.game_state.is_serving)),
        # ball is at z=0 and not serving
        state.ball_state.bounces + 1,
        state.ball_state.bounces
    )

    # update the scores and start pause of game
    after_score_update_game_state = jax.lax.cond(
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
                            state.game_state.enemy_score, state.game_state.player_game_score,
                            state.game_state.enemy_game_score, state.game_state.is_finished),
        lambda _: GameState(jnp.array(True), jnp.array(PAUSE_DURATION), state.game_state.player_score,
                            state.game_state.enemy_score + 1, state.game_state.player_game_score,
                            state.game_state.enemy_game_score, state.game_state.is_finished),
        None
    )

    # Check if a set has ended and if the game has ended
    after_set_check_game_state = check_end(check_set(after_score_update_game_state))
    game_point_scored = jnp.logical_or(
        after_set_check_game_state.player_game_score != after_score_update_game_state.player_game_score,
        after_set_check_game_state.enemy_game_score != after_score_update_game_state.enemy_game_score
    )
    should_change_sides = jnp.logical_and(
        game_point_scored,
        (after_set_check_game_state.player_game_score + after_set_check_game_state.enemy_game_score) % 2 != 0
    )

    new_player_field = jnp.where(
        should_change_sides,
        jnp.where(
            state.player_state.player_field == 1,
            -1,
            1
        ),
        state.player_state.player_field
    )

    new_player_serving = jnp.where(
        game_point_scored,
        jnp.where(
            state.player_state.player_serving == True,
            False,
            True
        ),
        state.player_state.player_serving
    )

    ball_spawn_y = jnp.where(
        new_player_serving == True,
        jnp.where(
            new_player_field == 1,
            GAME_OFFSET_TOP,
            GAME_OFFSET_TOP + GAME_HEIGHT
        ),
        jnp.where(
            new_player_field == 1,
            GAME_OFFSET_TOP + GAME_HEIGHT,
            GAME_OFFSET_TOP
        )
    )

    return jax.lax.cond(
        new_bounces >= 2,
        lambda _: TennisState(
            PlayerState(
                jnp.array(START_X),
                jnp.where(new_player_field == 1, jnp.array(PLAYER_START_Y),
                          jnp.array(ENEMY_START_Y)),
                jnp.array(PLAYER_START_DIRECTION),
                new_player_field,
                new_player_serving
            ),
            EnemyState(
                jnp.array(START_X),
                jnp.where(new_player_field == 1, jnp.array(ENEMY_START_Y),
                          jnp.array(PLAYER_START_Y)),
                jnp.array(0.0),
                jnp.array(PLAYER_START_DIRECTION),  # could create ENEMY_START_DIRECTION
                jnp.array(1)
            ),
            BallState(
                jnp.array(FRAME_WIDTH / 2.0 - BALL_WIDTH / 2),
                ball_spawn_y,
                jnp.array(0.0),
                jnp.array(0.0),
                jnp.array(0.0),
                jnp.array(0.0),
                jnp.array(0.0),
                jnp.array(FRAME_WIDTH / 2.0 - BALL_WIDTH / 2),
                jnp.array(GAME_OFFSET_TOP),
                jnp.array(0.0),
                jnp.array(0.0),
                jnp.array(0),
                jnp.array(-1)
            ),
            after_set_check_game_state,
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
            state.ball_state.last_hit
        ), state.game_state, state.counter),
        None
    )


@jax.jit
def check_set(state: GameState) -> GameState:
    """
    Checks whether the current set has ended and updates the score accordingly.

    Args:
        state (GameState): The current state of the game.

    Returns:
        GameState: The updated state of the game.
    """
    player_won_set = jnp.logical_and(state.player_score >= 4, state.player_score >= state.enemy_score + 2)
    enemy_won_set = jnp.logical_and(state.enemy_score >= 4, state.enemy_score >= state.player_score + 2)

    return jax.lax.cond(
        # Check if set has ended
        jnp.logical_or(player_won_set, enemy_won_set),
        # Set has ended
        lambda _: jax.lax.cond(
            player_won_set,
            # Player has won set
            lambda _: GameState(state.is_serving, state.pause_counter, jnp.array(0), jnp.array(0),
                                state.player_game_score + 1, state.enemy_game_score, state.is_finished),
            # Enemy has won set
            lambda _: GameState(state.is_serving, state.pause_counter, jnp.array(0), jnp.array(0),
                                state.player_game_score, state.enemy_game_score + 1, state.is_finished),
            None
        ),
        # Set is still ongoing
        lambda _: state,
        None
    )


@jax.jit
def check_end(state: GameState) -> GameState:
    """
    Checks whether the entire game is over and updates the score accordingly.

    Args:
        state (GameState): The current state of the game.

    Returns:
        GameState: The updated state of the game.
    """

    player_won = jnp.logical_and(state.player_game_score >= 6, state.player_game_score >= state.enemy_game_score + 2)
    enemy_won = jnp.logical_and(state.enemy_game_score >= 6, state.enemy_game_score >= state.player_game_score + 2)
    is_finished = jnp.where(jnp.logical_or(player_won, enemy_won), True, False)
    return GameState(state.is_serving, state.pause_counter, state.player_score, state.enemy_score,
                     state.player_game_score, state.enemy_game_score, is_finished)


@jax.jit
def player_step(state: TennisState, action: chex.Array) -> PlayerState:
    """
    Updates player position based on provided action and applies bounding box.

    Args:
        state (PlayerState): The current player state.
        action (chex.Array): The action to apply.

    Returns:
        PlayerState: The updated player state.
    """

    player_state = state.player_state

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
        player_state.player_x - 1,
        player_state.player_x,
    )
    # move right if the player is trying to move right
    player_x = jnp.where(
        right,
        player_state.player_x + 1,
        player_x,
    )
    # apply X bounding box
    player_x = jnp.clip(player_x, PLAYER_MIN_X, PLAYER_MAX_X)

    # move up if the player is trying to move up
    player_y = jnp.where(
        jnp.logical_and(
            up,
            jnp.logical_not(state.game_state.is_serving)  # not allowed to change y position while someone is serving
        ),
        player_state.player_y - 1,
        player_state.player_y,
    )

    # move down if the player is trying to move down
    player_y = jnp.where(
        jnp.logical_and(
            down,
            jnp.logical_not(state.game_state.is_serving)  # not allowed to change y position while someone is serving
        ),
        player_state.player_y + 1,
        player_y,
    )
    # apply Y bounding box
    player_y = jnp.where(
        player_state.player_field == 1,
        jnp.clip(player_y, PLAYER_Y_UPPER_BOUND_TOP, PLAYER_Y_LOWER_BOUND_TOP),
        jnp.clip(player_y, PLAYER_Y_UPPER_BOUND_BOTTOM, PLAYER_Y_LOWER_BOUND_BOTTOM)
    )

    new_player_direction = jnp.where(
        player_state.player_x > state.ball_state.ball_x,
        -1,
        player_state.player_direction
    )

    new_player_direction = jnp.where(
        player_state.player_x < state.ball_state.ball_x,
        1,
        new_player_direction
    )

    return PlayerState(
        player_x,
        player_y,
        new_player_direction,
        player_state.player_field,
        player_state.player_serving
    )


@jax.jit
def enemy_step(state: TennisState) -> EnemyState:
    """
    Updates enemy position by following the enemy strategy explained below.

    Enemy strategy:
        x coordinate:
            - just follow x of the ball
            - keep ball at center of sprite so that sprite does not flip often
        y coordinate:
            - rush towards net after hitting ball
            - stay at net for some time
            - turning point (does not quite line up with player hitting ball, usually slightly earlier)
            - move as far away from net as possible
            - after reaching limit sometimes starts moving towards net again

    Args:
        state (TennisState): The current state of the game.

    Returns:
        EnemyState: The updated enemy state.
    """

    # x-coordinate
    enemy_hit_offset = state.enemy_state.prev_walking_direction * 5 * -1 * 0  # this is always 0
    enemy_x_hit_point = state.enemy_state.enemy_x + PLAYER_WIDTH / 2 + enemy_hit_offset

    player_x_hit_point = state.player_state.player_x + PLAYER_WIDTH / 2
    ball_tracking_tolerance = 1
    x_tracking_tolerance = 2

    # move to middle
    def move_x_to_middle():
        middle_step_x = jnp.where(jnp.less_equal(state.enemy_state.enemy_x, GAME_MIDDLE_HORIZONTAL),
                                  state.enemy_state.enemy_x + 1,
                                  state.enemy_state.enemy_x - 1)
        return jnp.where(jnp.abs(state.enemy_state.enemy_x - GAME_MIDDLE_HORIZONTAL) > 1, middle_step_x,
                         state.enemy_state.enemy_x)

    def track_ball_x():
        enemy_aiming_x_offset = jnp.where(
            player_x_hit_point < FRAME_WIDTH / 2,
            5,
            -15
        )
        diff = state.ball_state.ball_x - (enemy_x_hit_point + enemy_aiming_x_offset)

        # move right if ball is sufficiently to the right
        new_enemy_x = jnp.where(
            diff > x_tracking_tolerance,
            state.enemy_state.enemy_x + 1,
            state.enemy_state.enemy_x
        )

        # move left if ball is sufficiently to the left
        new_enemy_x = jnp.where(
            diff < -x_tracking_tolerance,
            state.enemy_state.enemy_x - 1,
            new_enemy_x
        )
        return new_enemy_x

    new_enemy_x = jax.lax.cond(state.ball_state.last_hit == 1, move_x_to_middle, track_ball_x)

    cur_walking_direction = jnp.where(
        new_enemy_x - state.enemy_state.enemy_x < 0,
        -1,
        state.enemy_state.prev_walking_direction
    )
    cur_walking_direction = jnp.where(
        new_enemy_x - state.enemy_state.enemy_x > 0,
        1,
        cur_walking_direction
    )

    # jnp.clip(new_enemy_x - state.enemy_state.enemy_x, -1, 1))

    should_perform_direction_change = jnp.logical_or(jnp.logical_or(
        jnp.abs((enemy_x_hit_point) - state.ball_state.ball_x) >= ball_tracking_tolerance,
        # state.enemy_state.prev_walking_direction == cur_walking_direction
        False
    ), state.ball_state.last_hit == 1)

    new_enemy_x = jnp.where(should_perform_direction_change, new_enemy_x, state.enemy_state.enemy_x)

    # y-coordinate

    def enemy_y_step():
        state_after_y = jax.lax.cond(
            state.ball_state.last_hit == 1,
            # last hit was enemy rush net
            lambda _: EnemyState(state.enemy_state.enemy_x,
                                 jnp.where(jnp.logical_and(state.enemy_state.enemy_y != PLAYER_Y_LOWER_BOUND_TOP,
                                                           state.enemy_state.enemy_y != PLAYER_Y_UPPER_BOUND_BOTTOM),
                                           state.enemy_state.enemy_y - state.player_state.player_field,
                                           state.enemy_state.enemy_y
                                           ), state.enemy_state.prev_walking_direction,
                                 state.enemy_state.enemy_direction, jnp.array(1)),
            # last hit was player move away from net until baseline is hit then move towards ball
            lambda _: EnemyState(state.enemy_state.enemy_x,
                                 jnp.where(state.player_state.player_field == 1,
                                           jax.numpy.clip(state.enemy_state.enemy_y +
                                                          state.player_state.player_field * state.enemy_state.y_movement_direction,
                                                          PLAYER_Y_UPPER_BOUND_BOTTOM,
                                                          PLAYER_Y_LOWER_BOUND_BOTTOM),
                                           jax.numpy.clip(state.enemy_state.enemy_y +
                                                          state.player_state.player_field * state.enemy_state.y_movement_direction,
                                                          PLAYER_Y_UPPER_BOUND_TOP,
                                                          PLAYER_Y_LOWER_BOUND_TOP)),
                                 state.enemy_state.prev_walking_direction,
                                 state.enemy_state.enemy_direction,
                                 jnp.where(jnp.logical_or(state.enemy_state.enemy_y == PLAYER_Y_UPPER_BOUND_TOP,
                                                          state.enemy_state.enemy_y == PLAYER_Y_LOWER_BOUND_BOTTOM),
                                           jnp.array(-1),
                                           state.enemy_state.y_movement_direction)), operand=None)

        return jax.lax.cond(
            state.game_state.is_serving,
            lambda _: state.enemy_state,
            lambda _: state_after_y,
            operand=None
        )

    enemy_state_after_y_step = enemy_y_step()

    new_enemy_direction = jnp.where(
        state.enemy_state.enemy_x > state.ball_state.ball_x,
        -1,
        state.enemy_state.enemy_direction
    )

    new_enemy_direction = jnp.where(
        state.enemy_state.enemy_x < state.ball_state.ball_x,
        1,
        new_enemy_direction
    )

    return EnemyState(
        new_enemy_x,
        enemy_state_after_y_step.enemy_y,
        cur_walking_direction,
        new_enemy_direction,
        enemy_state_after_y_step.y_movement_direction
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

    # apply lower bounding box (500 is effectively no MAX bound)
    new_ball_z = jnp.clip(new_ball_z, 0, 500)
    new_ball_z_fp = jnp.clip(new_ball_z_fp, 0, 500)

    # ball movement in x/y direction is linear, no velocity involved
    new_ball_x = ball_state.ball_x + ball_state.move_x
    new_ball_y = ball_state.ball_y + ball_state.move_y

    player_state = state.player_state
    enemy_state = state.enemy_state

    # check player-ball collisions
    player_end = player_state.player_x + PLAYER_WIDTH
    ball_end = ball_state.ball_x + BALL_WIDTH
    player_overlap_ball_x = jnp.logical_not(
        jnp.logical_or(
            player_end <= ball_state.ball_x,
            ball_end <= player_state.player_x
        )
    )

    # Correct X positions based on facing direction
    corrected_player_x = jnp.where(
        player_state.player_direction == 1,
        player_state.player_x,
        player_state.player_x - PLAYER_WIDTH + 2
    )

    corrected_enemy_x = jnp.where(
        enemy_state.enemy_direction == 1,
        enemy_state.enemy_x,
        enemy_state.enemy_x - PLAYER_WIDTH + 2
    )

    upper_entity_x = jnp.where(
        player_state.player_field == 1,
        corrected_player_x,
        corrected_enemy_x
    )
    upper_entity_y = jnp.where(
        player_state.player_field == 1,
        player_state.player_y,
        enemy_state.enemy_y
    )

    lower_entity_x = jnp.where(
        player_state.player_field == -1,
        corrected_player_x,
        corrected_enemy_x
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
            0,  # the original game ignores height when checking collision
            BALL_WIDTH  # could be renamed to to BALL_SIZE because ball is square
        ),
        jnp.absolute(upper_entity_y + PLAYER_HEIGHT - ball_state.ball_y) <= 3
    )

    lower_entity_overlapping_ball = jnp.logical_and(
        is_overlapping(
            lower_entity_x,
            PLAYER_WIDTH,
            0,  # this is z pos
            PLAYER_HEIGHT,
            ball_state.ball_x,
            BALL_WIDTH,
            0,  # the original game ignores height when checking collision
            BALL_WIDTH  # could be renamed to to BALL_SIZE because ball is square
        ),
        jnp.absolute(lower_entity_y + PLAYER_HEIGHT - ball_state.ball_y) <= 3
    )

    lower_entity_performed_last_hit = jnp.logical_or(
        jnp.logical_and(
            state.player_state.player_field == -1, state.ball_state.last_hit != ENEMY_CONST
        ),
        jnp.logical_and(
            state.player_state.player_field == 1, state.ball_state.last_hit != PLAYER_CONST
        )
    )

    upper_entity_performed_last_hit = jnp.logical_or(
        jnp.logical_and(
            state.player_state.player_field == 1, state.ball_state.last_hit != ENEMY_CONST
        ),
        jnp.logical_and(
            state.player_state.player_field == -1, state.ball_state.last_hit != PLAYER_CONST
        )
    )

    # check if fire is pressed
    fire = jnp.any(jnp.array(
        [action == JAXAtariAction.FIRE, action == JAXAtariAction.LEFTFIRE, action == JAXAtariAction.DOWNLEFTFIRE,
         action == JAXAtariAction.DOWNFIRE,
         action == JAXAtariAction.DOWNRIGHTFIRE, action == JAXAtariAction.RIGHTFIRE,
         action == JAXAtariAction.UPRIGHTFIRE, action == JAXAtariAction.UPFIRE,
         action == JAXAtariAction.UPLEFTFIRE]))

    any_entity_ready_to_fire = jnp.logical_or(
        jnp.logical_and(
            upper_entity_overlapping_ball,
            lower_entity_performed_last_hit
        ),
        jnp.logical_and(
            lower_entity_overlapping_ball,
            upper_entity_performed_last_hit
        )
    )

    should_hit = jnp.logical_and(any_entity_ready_to_fire, jnp.logical_or(
        jnp.logical_not(jnp.logical_and(state.player_state.player_serving, state.game_state.is_serving)),
        fire
    ))
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

    hitting_entity_y = jnp.where(
        upper_entity_overlapping_ball,
        upper_entity_y,
        lower_entity_y
    )

    # record which entity hit the ball most recently
    new_last_hit = jnp.where(should_hit,
                             # player hit
                             jnp.where(
                                 jnp.logical_or(
                                     jnp.logical_and(upper_entity_overlapping_ball, player_state.player_field == 1),
                                     jnp.logical_and(lower_entity_overlapping_ball, player_state.player_field == -1)
                                 ),
                                 0,
                                 1
                             ),
                             state.ball_state.last_hit
                             )

    ball_state_after_fire = jax.lax.cond(
        should_hit,
        lambda _: handle_ball_fire(state, hitting_entity_x, hitting_entity_y, ball_fire_direction),
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
            ball_state.bounces,
            ball_state.last_hit,
        ),
        None
    )

    return TennisState(
        player_state,
        enemy_state,
        BallState(
            ball_state_after_fire.ball_x,
            ball_state_after_fire.ball_y,
            ball_state_after_fire.ball_z,
            ball_state_after_fire.ball_z_fp,
            ball_state_after_fire.ball_velocity_z_fp,
            ball_state_after_fire.ball_hit_start_x,
            ball_state_after_fire.ball_hit_start_y,
            ball_state_after_fire.ball_hit_target_x,
            ball_state_after_fire.ball_hit_target_y,
            ball_state_after_fire.move_x,
            ball_state_after_fire.move_y,
            ball_state_after_fire.bounces,
            new_last_hit,
        ),
        GameState(
            new_is_serving,
            state.game_state.pause_counter,
            state.game_state.player_score,
            state.game_state.enemy_score,
            state.game_state.player_game_score,
            state.game_state.enemy_game_score,
            state.game_state.is_finished
        ),
        state.counter
    )


@jax.jit
def is_overlapping(entity1_x, entity1_w, entity1_y, entity1_h, entity2_x, entity2_w, entity2_y,
                   entity2_h) -> chex.Array:
    """
    Checks if two entities are overlapping, given their positions and size.

    Args:
        entity1_x: The x coordinate of the first entity.
        entity1_w: The width of the first entity.
        entity1_y: The y coordinate of the first entity.
        entity1_h: The height of the first entity.
        entity2_x: The x coordinate of the second entity.
        entity2_w: The width of the second entity.
        entity2_y: The y coordinate of the second entity.
        entity2_h: The height of the second entity.

    Returns:
        chex.Array: True/False depending on whether the two entities are overlapping.
    """

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


@jax.jit
def handle_ball_fire(state: TennisState, hitting_entity_x, hitting_entity_y, direction) -> BallState:
    """
    Adds velocity to the ball based on the hitting entity position.

    Args:
        state (TennisState): Current state of the game.
        hitting_entity_x: X coordinate of the hitting entity.
        hitting_entity_y: Y coordinate of the hitting entity.
        direction: Direction of the hitting entity.

    Returns:
        BallState: New ball state after ball was hit.
    """

    # direction = 1 from top side to bottom
    # direction = -1 from bottom side to top
    # direction = 0 (dont do this)
    new_ball_hit_start_x = state.ball_state.ball_x
    new_ball_hit_start_y = state.ball_state.ball_y

    # hardcoded values could be replaced with constants, but won't make a difference since they shouldn't be changed anyway
    ball_width = 2.0
    max_dist = PLAYER_WIDTH / 2 + ball_width / 2

    angle = -1 * (((hitting_entity_x + PLAYER_WIDTH / 2) - (state.ball_state.ball_x + 2 / 2)) / max_dist) * direction

    hitting_entity_field = jnp.where(
        hitting_entity_y < GAME_MIDDLE,
        1,
        -1
    )
    angle *= hitting_entity_field  # adjust angle in case entity is in bottom field (* -1 will reverse angle)
    # calc x landing position depending on player hit angle
    # angle = 0 is neutral angle, between -1...1
    left_offset = -39
    right_offset = 39
    offset = ((angle + 1) / 2) * (right_offset - left_offset) + left_offset

    y_dist = jnp.where(
        direction > 0,
        # hitting from top to bottom
        jnp.where(
            hitting_entity_y > LONG_HIT_THRESHOLD_TOP,
            SHORT_HIT_DISTANCE,
            LONG_HIT_DISTANCE
        ),
        # hitting from bottom to top
        jnp.where(
            hitting_entity_y < LONG_HIT_THRESHOLD_BOTTOM,
            SHORT_HIT_DISTANCE,
            LONG_HIT_DISTANCE
        )
    )

    new_ball_hit_target_y = new_ball_hit_start_y + (y_dist * direction)
    field_min_x = 32
    field_max_x = 32 + FIELD_WIDTH_TOP
    new_ball_hit_target_x = jnp.clip(new_ball_hit_start_x + offset, field_min_x, field_max_x)

    @jax.jit
    def compute_ball_landing_frame(z_fp0, v0_fp, gravity_fp):
        """
        Computes the number of frames until the ball hits the ground (z == 0).

        Args:
            z_fp0: Initial z fixed-point position (e.g., 0)
            v0_fp: Initial z fixed-point velocity (e.g., 140)
            gravity_fp: Gravity per frame in fixed-point (e.g., 4)

        Returns:
            jnp.ndarray: Number of frames until ball hits the ground
        """
        A = -gravity_fp
        B = 2 * v0_fp + gravity_fp
        C = 2 * (z_fp0 - 10)  # '10' comes from floor(z_fp / 10) == 0  → z_fp < 10

        discriminant = B ** 2 - 4 * A * C
        sqrt_discriminant = jnp.sqrt(discriminant)

        # Only consider the positive root (when the ball actually hits the ground)
        t = (-B - sqrt_discriminant) / (2 * A)

        # Floor to get the last integer frame before hitting ground
        t_landing = jnp.floor(t).astype(jnp.int32)

        return t_landing

    hit_vel = 24.0

    dx = new_ball_hit_target_x - state.ball_state.ball_x
    dy = new_ball_hit_target_y - state.ball_state.ball_y

    steps = compute_ball_landing_frame(14, 24.0, BALL_GRAVITY_PER_FRAME)

    move_x = dx / steps
    move_y = dy / steps

    return BallState(
        state.ball_state.ball_x,
        state.ball_state.ball_y,
        jnp.array(14.0),
        jnp.array(140.0),
        jnp.array(hit_vel),
        new_ball_hit_start_x,
        new_ball_hit_start_y,
        new_ball_hit_target_x,
        new_ball_hit_target_y,
        move_x,
        move_y,
        jnp.array(0),  # ball has not bounced after last hit
        state.ball_state.last_hit
    )


@jax.jit
def tennis_reset() -> TennisState:
    """
    Provides the initial state for the game. For that purpose, we use the default values assigned in TennisState.

    Returns:
        TennisState: The initial state of the game.
    """
    return TennisState()


def switch_blue_and_red(sprite, blue_color=[117, 128, 240, 255], red_color=[240, 128, 128, 255]):
    # Convert color constants to jax arrays of the same dtype
    blue_color = jnp.array(blue_color, dtype=sprite.dtype)
    red_color = jnp.array(red_color, dtype=sprite.dtype)

    # Create a mask: shape (H, W), where each pixel matches the blue color
    mask_blue = jnp.all(sprite == blue_color, axis=-1)  # shape: (H, W)
    mask_red = jnp.all(sprite == red_color, axis=-1)  # shape: (H, W)

    # Find the indices of the pixels to replace
    indices_blue = jnp.argwhere(mask_blue)  # shape: (N, 2)
    indices_red = jnp.argwhere(mask_red)  # shape: (N, 2)

    # Replace each matching pixel using .at[].set()
    for idx in indices_blue:
        sprite = sprite.at[tuple(idx)].set(red_color)
    for idx in indices_red:
        sprite = sprite.at[tuple(idx)].set(blue_color)

    return sprite


def load_sprites():
    MODULE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    BG = jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "games/sprites/tennis/background.npy")), axis=0)
    BALL = jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "games/sprites/tennis/ball.npy")), axis=0)
    BALL_SHADOW = jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "games/sprites/tennis/ball_shadow.npy")),
                                  axis=0)
    PLAYER_0 = jnp.expand_dims(
        aj.loadFrame(os.path.join(MODULE_DIR, "games/sprites/tennis/player_no_racket.npy")), axis=0)
    PLAYER_1 = jnp.expand_dims(
        aj.loadFrame(os.path.join(MODULE_DIR, "games/sprites/tennis/player_no_racket_1.npy")), axis=0)
    PLAYER_2 = jnp.expand_dims(
        aj.loadFrame(os.path.join(MODULE_DIR, "games/sprites/tennis/player_no_racket_2.npy")), axis=0)
    PLAYER_3 = jnp.expand_dims(
        aj.loadFrame(os.path.join(MODULE_DIR, "games/sprites/tennis/player_no_racket_3.npy")), axis=0)
    UI_DEUCE = jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "games/sprites/tennis/ui_blue_9.npy")),
                               axis=0)
    UI_AD_IN = jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "games/sprites/tennis/ui_blue_9.npy")),
                               axis=0)
    UI_AD_OUT = jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "games/sprites/tennis/ui_blue_9.npy")),
                                axis=0)

    def recolor_rgba(stack, src_rgba, dst_rgba):
        # stack shape: (10, 1, H, W, 4) or (10, H, W, 4); dtype uint8
        src = jnp.asarray(src_rgba, dtype=stack.dtype)
        dst = jnp.asarray(dst_rgba, dtype=stack.dtype)
        mask = jnp.all(stack == src, axis=-1, keepdims=True)  # shape (...,1)
        return jnp.where(mask, dst, stack)

    RACKETS_RED = aj.load_and_pad_digits(
        os.path.join(MODULE_DIR, "games/sprites/tennis/racket_red_{}.npy"), num_chars=4
    )
    RACKETS_BLUE = recolor_rgba(RACKETS_RED, [240, 128, 128, 255], [117, 128, 240, 255])

    UI_NUMBERS_BLUE = aj.load_and_pad_digits(
        os.path.join(MODULE_DIR, "games/sprites/tennis/ui_blue_{}.npy"), num_chars=10
    )
    UI_NUMBERS_RED = recolor_rgba(UI_NUMBERS_BLUE, [117, 128, 240, 255], [240, 128, 128, 255])

    return (BG, switch_blue_and_red(BG), BALL, BALL_SHADOW,
            jnp.asarray([switch_blue_and_red(PLAYER_0), switch_blue_and_red(PLAYER_1), switch_blue_and_red(PLAYER_2),
                         switch_blue_and_red(PLAYER_3)]), jnp.asarray([PLAYER_0, PLAYER_1, PLAYER_2, PLAYER_3]),
            RACKETS_BLUE,
            RACKETS_RED,
            UI_NUMBERS_BLUE,
            UI_NUMBERS_RED,
            UI_DEUCE,
            UI_AD_IN,
            UI_AD_OUT)


class TennisRenderer:

    def __init__(self):
        (self.BG_TOP_RED, self.BG_TOP_BLUE, self.BALL, self.BALL_SHADOW, self.PLAYER_BLUE, self.PLAYER_RED,
         self.RACKET_BLUE, self.RACKET_RED, self.UI_NUMBERS_BLUE,
         self.UI_NUMBERS_RED, self.UI_DEUCE, self.UI_AD_IN, self.UI_AD_OUT) = load_sprites()

    def render_number_centered(self, raster, number, position, red=False):
        digits = aj.int_to_digits(number, max_digits=2)  # or 2
        sprites = self.UI_NUMBERS_RED if red else self.UI_NUMBERS_BLUE

        n = jnp.asarray(number, jnp.int32)

        # is the value 1 digit?
        is_single = n < 10

        # For max_digits = 2:
        start_idx = jax.lax.select(is_single, 1, 0)  # skip leading zero when single digit
        count = jax.lax.select(is_single, 1, 2)

        raster = aj.render_label_selective(
            raster, position[0], position[1],
            digits, sprites,
            start_idx, count,
            spacing=7
        )

        return raster

    def render(self, state: TennisState) -> jnp.ndarray:
        raster = aj.create_initial_frame(width=FRAME_WIDTH, height=FRAME_HEIGHT)

        # render background
        bg_top_red = jnp.where(state.player_state.player_field == 1, True, False)

        raster = jax.lax.cond(
            bg_top_red,
            lambda r: aj.render_at(
                r,
                0,
                0,
                aj.get_sprite_frame(self.BG_TOP_RED, 0),
            ),
            lambda r: aj.render_at(
                r,
                0,
                0,
                aj.get_sprite_frame(self.BG_TOP_BLUE, 0),
            ),
            raster,
        )

        # render ball
        frame_ball_shadow = aj.get_sprite_frame(self.BALL_SHADOW, 0)
        raster = aj.render_at(raster, state.ball_state.ball_x, state.ball_state.ball_y, frame_ball_shadow)

        frame_ball = aj.get_sprite_frame(self.BALL, 0)
        # apply flat y offset depending on z value
        raster = aj.render_at(raster, state.ball_state.ball_x, state.ball_state.ball_y - state.ball_state.ball_z,
                              frame_ball)

        # render player & enemy
        frame_player = aj.get_sprite_frame(self.PLAYER_RED[state.animator_state.player_frame], 0)
        frame_enemy = aj.get_sprite_frame(self.PLAYER_BLUE[state.animator_state.enemy_frame], 0)

        player_pos = jnp.where(state.player_state.player_direction == 1,
                               state.player_state.player_x - 2,
                               state.player_state.player_x - 4)

        racket_offset_x = jnp.asarray([0, 1, 2, 2])
        player_racket_pos = jnp.where(state.player_state.player_direction == 1,
                                      state.player_state.player_x - 4 + racket_offset_x[
                                          state.animator_state.player_racket_frame],
                                      state.player_state.player_x - racket_offset_x[
                                          state.animator_state.player_racket_frame] - 2)

        raster = aj.render_at(raster, player_pos, state.player_state.player_y, frame_player,
                              flip_horizontal=jnp.where(state.player_state.player_direction == -1, True, False))

        racket_offset = jnp.asarray([1, 8, 8, 4])

        frame_racket_player = aj.get_sprite_frame(self.RACKET_RED, state.animator_state.player_racket_frame)

        raster = aj.render_at(raster, player_racket_pos + state.player_state.player_direction * 8,
                              state.player_state.player_y + racket_offset[state.animator_state.player_racket_frame],
                              frame_racket_player,
                              flip_horizontal=jnp.where(state.player_state.player_direction == -1, True, False))

        enemy_pos = jnp.where(state.enemy_state.enemy_direction == 1,
                              state.enemy_state.enemy_x - 2,
                              state.enemy_state.enemy_x - 4)
        enemy_racket_pos = jnp.where(state.enemy_state.enemy_direction == 1,
                                     state.enemy_state.enemy_x - 4 + racket_offset_x[
                                         state.animator_state.enemy_racket_frame],
                                     state.enemy_state.enemy_x - racket_offset_x[
                                         state.animator_state.enemy_racket_frame] - 2)

        raster = aj.render_at(raster, enemy_pos, state.enemy_state.enemy_y, frame_enemy,
                              flip_horizontal=jnp.where(state.enemy_state.enemy_direction == -1, True, False))

        frame_racket_enemy = aj.get_sprite_frame(self.RACKET_BLUE, state.animator_state.enemy_racket_frame)

        raster = aj.render_at(raster, enemy_racket_pos + state.enemy_state.enemy_direction * 8,
                              state.enemy_state.enemy_y + racket_offset[state.animator_state.enemy_racket_frame],
                              frame_racket_enemy,
                              flip_horizontal=jnp.where(state.enemy_state.enemy_direction == -1, True, False))

        # render score UI
        should_display_overall_score = jnp.logical_and((
                                                               state.game_state.player_game_score + state.game_state.enemy_game_score) > 0,
                                                       jnp.logical_and(state.game_state.player_score == 0,
                                                                       state.game_state.enemy_score == 0))

        def render_overall(r):
            # display overall score
            r = self.render_number_centered(r, state.game_state.player_game_score, [FRAME_WIDTH / 4, 2],
                                            red=True)
            r = self.render_number_centered(r, state.game_state.enemy_game_score, [(FRAME_WIDTH / 4) * 3, 2])
            return r

        def render_current_score(raster):
            tennis_scores = jnp.array([0, 15, 30, 40], dtype=jnp.int32)

            ps = state.game_state.player_score
            es = state.game_state.enemy_score
            serving = state.player_state.player_serving  # bool

            # "deuce-like" phase once both reached 40 (i.e., >= 3 points)
            deuce_like = (ps >= 3) & (es >= 3)

            def render_deuce(raster_in):
                # deuce if tied; otherwise advantage
                is_tied = (ps == es)

                def do_deuce(r):
                    ui = aj.get_sprite_frame(self.UI_DEUCE, 0)
                    x = (FRAME_WIDTH // 4) - (ui.shape[0] // 2)
                    return aj.render_at(r, x, 2, ui)

                def do_adv(r):
                    # AD-IN if leader is the server; otherwise AD-OUT
                    ad_in = ((ps > es) & serving) | ((es > ps) & (~serving))

                    def render_ad_in(rr):
                        ui = aj.get_sprite_frame(self.UI_AD_IN, 0)
                        x = (FRAME_WIDTH // 4) - (ui.shape[0] // 2)
                        return aj.render_at(rr, x, 2, ui)

                    def render_ad_out(rr):
                        ui = aj.get_sprite_frame(self.UI_AD_OUT, 0)
                        x = (FRAME_WIDTH // 4) - (ui.shape[0] // 2)
                        return aj.render_at(rr, x, 2, ui)

                    return jax.lax.cond(ad_in, render_ad_in, render_ad_out, r)

                return jax.lax.cond(is_tied, do_deuce, do_adv, raster_in)

            def render_regular(raster_in):
                # clip score indices to [0..3] and map to 0/15/30/40
                pid = jnp.minimum(3, ps)
                eid = jnp.minimum(3, es)
                pnum = tennis_scores[pid]
                enum = tennis_scores[eid]

                r = self.render_number_centered(raster_in, pnum, [FRAME_WIDTH // 4, 2], red=True)
                r = self.render_number_centered(r, enum, [(FRAME_WIDTH // 4) * 3, 2])
                return r

            raster = jax.lax.cond(deuce_like, render_deuce, render_regular, raster)
            return raster

        raster = jax.lax.cond(
            should_display_overall_score,
            lambda r: render_overall(r),
            lambda r: render_current_score(r),
            raster,
        )

        return raster


renderer = TennisRenderer()


class PlayerObs(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    player_direction: chex.Array
    player_field: chex.Array  # top (1) or bottom (-1) field
    player_serving: chex.Array  # true: Player is serving, false: Enemy is serving


class EnemyObs(NamedTuple):
    enemy_x: chex.Array
    enemy_y: chex.Array
    enemy_direction: chex.Array


class BallObs(NamedTuple):
    ball_x: chex.Array  # x-coordinate of the ball
    ball_y: chex.Array  # y-coordinate of the ball
    ball_z: chex.Array  # z-coordinate of the ball
    bounces: chex.Array  # how many times the ball has hit the ground since it was last hit by an entity
    last_hit: chex.Array  # 0 if last hit was performed by player, 1 if last hit was by enemy


class TennisObs(NamedTuple):
    player: PlayerObs
    enemy: EnemyObs
    ball: BallObs
    is_serving_state: chex.Array  # whether the game is currently in serving state (ball bouncing on one side until player hits)
    player_points: chex.Array  # The score line within the current set (goes up in increments of 1, instead of traditional tennis counting)
    enemy_points: chex.Array
    player_sets: chex.Array  # Number of won sets
    enemy_sets: chex.Array


class TennisInfo(NamedTuple):
    all_rewards: jnp.ndarray


class AtraJaxisTennisRenderer(JAXGameRenderer):

    def render(self, state: TennisState):
        return renderer.render(state)


class TennisConstants(NamedTuple):
    pass


class TennisJaxEnv(JaxEnvironment[TennisState, TennisObs, TennisInfo, TennisConstants]):

    def __init__(self, consts: TennisConstants = None, reward_funcs: list[callable] = None):
        if consts is None:
            consts = TennisConstants()
        self.consts = consts
        self.reward_funcs = reward_funcs

    def _get_all_reward(self, previous_state: TennisState, state: TennisState):
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array(
            [reward_func(previous_state, state) for reward_func in self.reward_funcs]
        )
        return rewards

    def reset(self, key) -> Tuple[TennisObs, TennisState]:
        reset_state = tennis_reset()
        reset_obs = self._get_observation(reset_state)

        return reset_obs, reset_state

    def step(self, state: TennisState, action) -> Tuple[TennisObs, TennisState, float, bool, TennisInfo]:
        new_state = tennis_step(state, action)
        new_obs = self._get_observation(new_state)
        reward = self._get_reward(state, new_state)
        all_rewards = self._get_all_reward(state, new_state)
        done = self._get_done(new_state)
        info = self._get_info(new_state, all_rewards=all_rewards)

        return new_obs, new_state, reward, done, info

    def render(self, state: TennisState) -> Tuple[jnp.ndarray]:
        return renderer.render(state)

    def action_space(self) -> jnp.ndarray:
        """self.action_set = [
            Action.NOOP,
            Action.FIRE,
            Action.UP,
            Action.RIGHT,
            Action.LEFT,
            Action.DOWN,
            Action.UPRIGHT,
            Action.UPLEFT,
            Action.DOWNRIGHT,
            Action.DOWNLEFT,
            Action.UPFIRE,
            Action.RIGHTFIRE,
            Action.LEFTFIRE,
            Action.DOWNFIRE,
            Action.UPRIGHTFIRE,
            Action.UPLEFTFIRE,
            Action.DOWNRIGHTFIRE,
            Action.DOWNLEFTFIRE
        ]"""

        return spaces.Discrete(18)

    def flatten_player_obs(self, player_obs: PlayerObs):
        return jnp.concatenate([jnp.array([player_obs.player_x]), jnp.array([player_obs.player_y]),
                                jnp.array([player_obs.player_direction]), jnp.array([player_obs.player_field]),
                                jnp.array([player_obs.player_serving])])

    def flatten_enemy_obs(self, enemy_obs: EnemyObs):
        return jnp.concatenate(
            [jnp.array([enemy_obs.enemy_x]), jnp.array([enemy_obs.enemy_y]), jnp.array([enemy_obs.enemy_direction])])

    def flatten_ball_obs(self, ball_obs: BallObs):
        return jnp.concatenate(
            [jnp.array([ball_obs.ball_x]), jnp.array([ball_obs.ball_y]), jnp.array([ball_obs.ball_z]),
             jnp.array([ball_obs.bounces]), jnp.array([ball_obs.last_hit])])

    def observation_space(self) -> Tuple:
        return spaces.Dict({
            "player": spaces.Dict({
                "player_x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.float32),
                "player_y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.float32),
                "player_direction": spaces.Box(low=-1, high=1, shape=(), dtype=jnp.int32),
                "player_field": spaces.Box(low=-1, high=1, shape=(), dtype=jnp.int32),
                "player_serving": spaces.Box(low=0, high=1, shape=(), dtype=jnp.uint8),
            }),
            "enemy": spaces.Dict({
                "enemy_x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.float32),
                "enemy_y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.float32),
                "enemy_direction": spaces.Box(low=-1, high=1, shape=(), dtype=jnp.int32),
            }),
            "ball": spaces.Dict({
                "ball_x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.float32),
                "ball_y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.float32),
                "ball_z": spaces.Box(low=0, high=999, shape=(), dtype=jnp.float32),
                # no theoretical upper limit, but usually won't go above 50
                "bounces": spaces.Box(low=0, high=999, shape=(), dtype=jnp.int32),
                # no theoretical upper limit, but usually won't go above 2 since game is restarted at that point
                "last_hit": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
            }),
            "is_serving_state": spaces.Box(low=0, high=1, shape=(), dtype=jnp.uint8),
            "player_points": spaces.Box(low=0, high=45, shape=(), dtype=jnp.int32),
            "player_sets": spaces.Box(low=0, high=999, shape=(), dtype=jnp.int32),  # no theoretical upper limit
            "enemy_points": spaces.Box(low=0, high=45, shape=(), dtype=jnp.int32),
            "enemy_sets": spaces.Box(low=0, high=999, shape=(), dtype=jnp.int32),  # no theoretical upper limit
        })

    def image_space(self) -> spaces.Box:
        return spaces.Box(low=0, high=255, shape=(FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=jnp.uint8)

    def obs_to_flat_array(self, obs):
        return jnp.concatenate([
            self.flatten_player_obs(obs.player),
            self.flatten_enemy_obs(obs.enemy),
            self.flatten_ball_obs(obs.ball),
            obs.is_serving_state.flatten(),
            obs.player_points.flatten(),
            obs.player_sets.flatten(),
            obs.enemy_points.flatten(),
            obs.enemy_sets.flatten()
        ]).astype(jnp.float64)

#jnp.array(x, dtype=jnp.float32)

    def _get_observation(self, state: TennisState) -> TennisObs:
        return TennisObs(player=PlayerObs(state.player_state.player_x.astype(jnp.float32), state.player_state.player_y.astype(jnp.float32),
                                          state.player_state.player_direction.astype(jnp.int32),
                                          state.player_state.player_field.astype(jnp.int32),
                                          jnp.where(state.player_state.player_serving, 1, 0).astype(jnp.uint8)),
                         enemy=EnemyObs(state.enemy_state.enemy_x.astype(jnp.float32), state.enemy_state.enemy_y.astype(jnp.float32),
                                        state.enemy_state.enemy_direction.astype(jnp.int32)),
                         ball=BallObs(state.ball_state.ball_x.astype(jnp.float32), state.ball_state.ball_y.astype(jnp.float32), state.ball_state.ball_z.astype(jnp.float32),
                                      state.ball_state.bounces.astype(jnp.int32), state.ball_state.last_hit.astype(jnp.int32)),
                         is_serving_state=jnp.where(state.game_state.is_serving, 1, 0).astype(jnp.uint8),
                         player_points=state.game_state.player_score.astype(jnp.int32),
                         player_sets=state.game_state.player_game_score.astype(jnp.int32),
                         enemy_points=state.game_state.enemy_score.astype(jnp.int32),
                         enemy_sets=state.game_state.enemy_game_score.astype(jnp.int32))

    def _get_info(self, state: TennisState, all_rewards: jnp.ndarray = None) -> TennisInfo:
        return TennisInfo(all_rewards=all_rewards)

    def _get_reward(self, previous_state: TennisState, state: TennisState) -> float:
        return 0.0

    def _get_done(self, state: TennisState) -> bool:
        return state.game_state.is_finished
