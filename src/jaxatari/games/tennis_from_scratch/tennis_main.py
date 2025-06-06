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
GAME_OFFSET_LEFT_BOTTOM = 15 + 1  # don't use 16, because that is on the line and playing on the line still counts todo: check if that is actually true
GAME_OFFSET_TOP = 43.0  # don't use 44, because that is on the line and playing on the line still counts todo: check if that is actually true
GAME_WIDTH = FIELD_WIDTH_BOTTOM
GAME_HEIGHT = FIELD_HEIGHT
GAME_MIDDLE = GAME_OFFSET_TOP + 0.5 * GAME_HEIGHT
GAME_OFFSET_BOTTOM = GAME_OFFSET_TOP + GAME_HEIGHT
PAUSE_DURATION = 100

# player constants
PLAYER_CONST = 0
PLAYER_WIDTH = 13  # player flips side so total covered x section is greater
PLAYER_HEIGHT = 23
PLAYER_MIN_X = 10  # 17
PLAYER_MAX_X = 130  # 142
# lower y-axis values are towards the top in our case, opposite in original game
PLAYER_Y_LOWER_BOUND_BOTTOM = 160 #180  # 206-2-PLAYER_HEIGHT
PLAYER_Y_UPPER_BOUND_BOTTOM = 109  # 206-53-PLAYER_HEIGHT
PLAYER_Y_LOWER_BOUND_TOP = 70 #72  # 206-91-PLAYER_HEIGHT
PLAYER_Y_UPPER_BOUND_TOP = 15 #35  # 206-148-PLAYER_HEIGHT

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
LONG_HIT_THRESHOLD_TOP = 30
LONG_HIT_THRESHOLD_BOTTOM = 30

# enemy constants
ENEMY_CONST = 1

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
    player_direction: chex.Array  # direction the player is currently facing in todo explain which values can happen here in what they mean
    player_field: chex.Array  # top or bottom field


class EnemyState(NamedTuple):
    enemy_x: chex.Array  # x-coordinate of the enemy
    enemy_y: chex.Array  # y-coordinate of the enemy
    prev_walking_direction: chex.Array # previous walking direction (in x-direction) of the enemy, -1 = towards x=min, 1 = towards x=max


class GameState(NamedTuple):
    is_serving: chex.Array  # whether the game is currently in serving state (ball bouncing on one side until player hits)
    pause_counter: chex.Array  # delay between restart of game
    player_score: chex.Array  # The score line within the current set
    enemy_score: chex.Array
    player_game_score: chex.Array  # Number of won sets
    enemy_game_score: chex.Array
    is_finished: chex.Array  # True if the game is finished (Player or enemy has won the game)


class TennisState(NamedTuple):
    player_state: PlayerState = PlayerState(  # all player-related data
        jnp.array(START_X),
        jnp.array(PLAYER_START_Y),
        jnp.array(PLAYER_START_DIRECTION),
        jnp.array(PLAYER_START_FIELD)
    )
    enemy_state: EnemyState = EnemyState(  # all enemy-related data
        jnp.array(START_X),
        jnp.array(ENEMY_START_Y),
        jnp.array(0.0)
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
    increased_score_state = jax.lax.cond(
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
                          jnp.array(PLAYER_START_Y)),
                jnp.array(0.0)
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
                jnp.array(0),
                jnp.array(-1)
            ),
            # Check if a set has ended and if the game has ended
            check_end(check_set(increased_score_state)),
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


# Cheks whether the current set has ended and updates the score accordingly
@jax.jit
def check_set(state: GameState) -> GameState:
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


# Checks whether the entire game is over
@jax.jit
def check_end(state: GameState) -> GameState:
    player_won = jnp.logical_and(state.player_game_score >= 6, state.player_game_score >= state.enemy_game_score + 2)
    enemy_won = jnp.logical_and(state.enemy_game_score >= 6, state.enemy_game_score >= state.player_game_score + 2)
    is_finished = jnp.where(jnp.logical_or(player_won, enemy_won), True, False)
    return GameState(state.is_serving, state.pause_counter, state.player_score, state.enemy_score,
                     state.player_game_score, state.enemy_game_score, is_finished)


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


"""
enemy strategy:
x - coordinate:
    - just follow x of the ball
    - keep ball at center of sprite so that sprite does not flip often
y - coordinate:
    - rush towards net after hitting ball
    - stay at net for some time
    - turning point (does not quite line up with player hitting ball, usually slightly earlier)
    - move as far away from net as possible
    - after reaching limit sometimes starts moving towards net again


"""


@jax.jit
def enemy_step(state: TennisState) -> EnemyState:
    # x-coordinate
    # simply track balls x-coordinate
    ball_tracking_tolerance = 1
    enemy_hit_offset = state.enemy_state.prev_walking_direction * 5 * -1 * 0
    enemy_x_hit_point = state.enemy_state.enemy_x + PLAYER_WIDTH / 2 + enemy_hit_offset
    new_enemy_x = jnp.where(
        enemy_x_hit_point < state.ball_state.ball_x,
        state.enemy_state.enemy_x + 1,
        state.enemy_state.enemy_x
    )

    new_enemy_x = jnp.where(
        enemy_x_hit_point > state.ball_state.ball_x,
        state.enemy_state.enemy_x - 1,
        new_enemy_x
    )

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

        #jnp.clip(new_enemy_x - state.enemy_state.enemy_x, -1, 1))

    should_perform_direction_change = jnp.logical_or(
        jnp.abs((enemy_x_hit_point) - state.ball_state.ball_x) >= ball_tracking_tolerance,
        #state.enemy_state.prev_walking_direction == cur_walking_direction
        False
    )

    new_enemy_x = jnp.where(should_perform_direction_change, new_enemy_x, state.enemy_state.enemy_x)

    # y-coordinate

    normal_step_y = jnp.where(
        state.ball_state.last_hit == 1,
        # last hit was enemy, rush the net
        jnp.where(jnp.logical_and(state.enemy_state.enemy_y != PLAYER_Y_LOWER_BOUND_TOP,
                                  state.enemy_state.enemy_y != PLAYER_Y_UPPER_BOUND_BOTTOM),
                  state.enemy_state.enemy_y - state.player_state.player_field,
                  state.enemy_state.enemy_y
                  ),
        # last hit was player, move away from net
        jnp.where(jnp.logical_and(state.enemy_state.enemy_y != PLAYER_Y_UPPER_BOUND_TOP,
                                  state.enemy_state.enemy_y != PLAYER_Y_LOWER_BOUND_BOTTOM),
                  state.enemy_state.enemy_y + state.player_state.player_field,
                  state.enemy_state.enemy_y
                  )
    )

    new_enemy_y = jnp.where(state.game_state.is_serving,
                            state.enemy_state.enemy_y,
                            normal_step_y,
                            #jnp.clip(normal_step_y, 0, ENEMY_START_Y)
                            )

    return EnemyState(
        new_enemy_x,
        new_enemy_y,
        cur_walking_direction,
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
            0, # this is z pos
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
            0, # this is z pos
            PLAYER_HEIGHT,
            ball_state.ball_x,
            BALL_WIDTH,
            ball_state.ball_z,
            BALL_WIDTH
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


    should_hit = jnp.logical_and(any_entity_ready_to_fire, jnp.logical_or(jnp.logical_not(state.game_state.is_serving), fire))
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
        upper_entity_x,
        upper_entity_y
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
            new_last_hit,
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
def handle_ball_fire(state: TennisState, hitting_entity_x, hitting_entity_y, direction) -> BallState:
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

    #y_distance = jnp.where(
    #    direction ,
   # )

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
        jnp.array(0),
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
