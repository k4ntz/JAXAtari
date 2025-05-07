import os
from functools import partial
from typing import NamedTuple, Tuple
import jax.lax
import jax.numpy as jnp
import chex
import pygame
from gymnax.environments import spaces

from jaxatari.rendering import atraJaxis as aj
from jaxatari.environment import JaxEnvironment

# Constants for game environment
WIDTH = 160
HEIGHT = 210

# Action constants
NOOP = 0
FIRE = 1
RIGHT = 2
LEFT = 3
# TODO: What are these actions?
RIGHTFIRE = 4
LEFTFIRE = 5

# Physics constants
# TODO: check if these are correct
GRAVITY = 3  # 0.12
BALL_MAX_SPEED = 6.0
FLIPPER_STRENGTH = 4
PLUNGER_MAX_STRENGTH = 8

# Game layout constants
# TODO: check if these are correct
BALL_SIZE = (4, 4)
FLIPPER_LEFT_POS = (30, 180)
FLIPPER_RIGHT_POS = (110, 180)
PLUNGER_POS = (150, 120)
PLUNGER_MAX_HEIGHT = 20  # Taken from RAM values (67-87)


# Background color and object colors
BACKGROUND_COLOR = 0, 0, 0
BALL_COLOR = 255, 255, 255  # ball
FLIPPER_COLOR = 255, 0, 0  # flipper
TEXT_COLOR = 255, 255, 255  # text

# Starting at plunger position
# TODO: check if these are correct
BALL_START_X = jnp.array(150)
BALL_START_Y = jnp.array(120)
BALL_START_DIRECTION = jnp.array(0)

# Pygame window dimensions
WINDOW_WIDTH = 160 * 3
WINDOW_HEIGHT = 210 * 3

# TODO: check if these are correct
WALL_TOP_Y = 16
WALL_LEFT_X = 0
WALL_RIGHT_X = 160
WALL_BOTTOM_Y = 200

# TODO: check if these are correct
WALL_TOP_HEIGHT = 5
WALL_BOTTOM_HEIGHT = 5
WALL_LEFT_HEIGHT = 5
WALL_RIGHT_HEIGHT = 5

# define the positions of the state information
STATE_TRANSLATOR: dict = {
    0: "ball_x",
    1: "ball_y",
    2: "ball_vel_x",
    3: "ball_vel_y",
    4: "left_flipper_angle",
    5: "right_flipper_angle",
    6: "plunger_position",
    7: "score",
    8: "lives",
    9: "bonus_multiplier",
    10: "bumpers_active",
    11: "targets_hit",
    12: "step_counter",
    13: "ball_in_play",
}


def get_human_action() -> chex.Array:
    """
    Records if UP or DOWN is being pressed and returns the corresponding action.

    Returns:
        action: int, action taken by the player (LEFT, RIGHT, FIRE, LEFTFIRE, RIGHTFIRE, NOOP).
    """
    keys = pygame.key.get_pressed()
    if keys[pygame.K_a] and keys[pygame.K_SPACE]:
        return jnp.array(LEFTFIRE)
    elif keys[pygame.K_d] and keys[pygame.K_SPACE]:
        return jnp.array(RIGHTFIRE)
    elif keys[pygame.K_a]:
        return jnp.array(LEFT)
    elif keys[pygame.K_d]:
        return jnp.array(RIGHT)
    elif keys[pygame.K_SPACE]:
        return jnp.array(FIRE)
    else:
        return jnp.array(NOOP)


# immutable state container
class VideoPinballState(NamedTuple):
    ball_x: chex.Array
    ball_y: chex.Array
    ball_vel_x: chex.Array
    ball_vel_y: chex.Array
    ball_direction: chex.Array  # 0: left/up, 1:left/down , 2: right/up, 3: right/down
    left_flipper_angle: chex.Array
    right_flipper_angle: chex.Array
    plunger_position: (
        chex.Array
    )  # Value between 0 and 20 where 20 means that the plunger is fully pulled
    plunger_power: (
        chex.Array
    )  # Should be 2 * plunger_position and only be set when the plunger is released
    score: chex.Array
    lives: chex.Array
    bonus_multiplier: chex.Array
    bumpers_active: chex.Array
    targets_hit: chex.Array
    step_counter: chex.Array
    ball_in_play: chex.Array
    obs_stack: chex.ArrayTree


class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray


class VideoPinballObservation(NamedTuple):
    ball: EntityPosition
    left_flipper: EntityPosition
    right_flipper: EntityPosition
    plunger: EntityPosition
    bumpers: jnp.ndarray  # bumper states array
    targets: jnp.ndarray  # target states array
    score: jnp.ndarray
    lives: jnp.ndarray
    bonus_multiplier: jnp.ndarray


class VideoPinballInfo(NamedTuple):
    time: jnp.ndarray
    all_rewards: chex.Array


@jax.jit
def plunger_step(state_plunger_position: chex.Array, action: chex.Array) -> chex.Array:
    """
    Update the plunger position based on the current state and action.
    And set the plunger power to 2 * plunger_position.
    """
    # check if the plunger is pulled
    # TODO: We also need to check if it allowed to fire
    # TODO: Also if the ball lands in the plunger area during play it is allowed to fire again even if no reset occured
    plunger_pulled = jnp.logical_or(action == FIRE, action == LEFTFIRE)

    # Reset plunger power
    plunger_power = jnp.array(0)  # Reset after each step

    # Calculate the plunger power by its position
    # Formula is plunger_power = 2 * plunger_position

    # update the plunger position
    # 0 after plunger release

    return plunger_position, plunger_power


@jax.jit
def player_step(
    state_player_y, state_player_speed, acceleration_counter, action: chex.Array
):
    # check if one of the buttons is pressed
    up = jnp.logical_or(action == LEFT, action == LEFTFIRE)
    down = jnp.logical_or(action == RIGHT, action == RIGHTFIRE)

    # get the current acceleration
    acceleration = PLAYER_ACCELERATION[acceleration_counter]

    # perform the deceleration checks first, since in the base game
    # on a direction switch the player is first decelerated and then accelerated in the new direction
    # check if the player touches a wall
    touches_wall = jnp.logical_or(
        state_player_y < WALL_TOP_Y,
        state_player_y + PLAYER_SIZE[1] > WALL_BOTTOM_Y,
    )

    player_speed = state_player_speed

    # if no button was clicked OR the paddle touched a wall and there is a speed, apply deceleration (halfing the speed every tick)
    player_speed = jax.lax.cond(
        jnp.logical_or(jnp.logical_not(jnp.logical_or(up, down)), touches_wall),
        lambda s: jnp.round(s / 2).astype(jnp.int32),
        lambda s: s,
        operand=player_speed,
    )

    direction_change_up = jnp.logical_and(up, state_player_speed > 0)
    # also apply deceleration if the direction is changed
    player_speed = jax.lax.cond(
        direction_change_up,
        lambda s: 0,
        lambda s: s,
        operand=player_speed,
    )
    direction_change_down = jnp.logical_and(down, state_player_speed < 0)

    player_speed = jax.lax.cond(
        direction_change_down,
        lambda s: 0,
        lambda s: s,
        operand=player_speed,
    )

    # reset the acceleration counter on a direction change
    direction_change = jnp.logical_or(direction_change_up, direction_change_down)
    acceleration_counter = jax.lax.cond(
        direction_change,
        lambda _: 0,
        lambda s: s,
        operand=acceleration_counter,
    )

    # add the current acceleration to the speed (positive if up, negative if down)
    player_speed = jax.lax.cond(
        up,
        lambda s: jnp.maximum(s - acceleration, -MAX_SPEED),
        lambda s: s,
        operand=player_speed,
    )

    player_speed = jax.lax.cond(
        down,
        lambda s: jnp.minimum(s + acceleration, MAX_SPEED),
        lambda s: s,
        operand=player_speed,
    )

    # reset or increment the acceleration counter here
    new_acceleration_counter = jax.lax.cond(
        jnp.logical_or(up, down),  # If moving in either direction
        lambda s: jnp.minimum(s + 1, 15),  # Increment counter
        lambda s: 0,  # Reset if no movement
        operand=acceleration_counter,
    )

    # calculate the new player position
    player_y = jnp.clip(
        state_player_y + player_speed,
        WALL_TOP_Y + WALL_TOP_HEIGHT - 10,
        WALL_BOTTOM_Y - 4,
    )
    return player_y, player_speed, new_acceleration_counter


@jax.jit
def hit_obstacle_to_left(
    ball_x: chex.Array, ball_y: chex.Array, obstacle_: chex.Array
) -> bool:
    """
    Check if the ball is hitting an obstacle to the left.
    """
    return jnp.logical_and(
        ball_x < obstacle_x,
        jnp.logical_and(
            ball_y > WALL_TOP_Y,
            ball_y < WALL_BOTTOM_Y,
        ),
    )


def ball_step(
    state: VideoPinballState,
    action,
):
    """
    Update the pinballs position and velocity based on the current state and action.
    """

    ball_vel_x = state.ball_vel_x
    ball_vel_y = state.ball_vel_y
    ball_direction = state.ball_direction

    """
    Plunger calculation
    """
    # Add plunger power to the ball velocity, only set to non-zero value once fired
    ball_vel_y = jax.lax.cond(
        state.plunger_power > 0,
        ball_vel_y + state.plunger_power,
        ball_vel_y,
    )

    """
    Paddle calculation
    """
    # Check if the ball is hitting a paddle

    """
    Obstacle hit calculation
    """
    # Iterate over all other rigid objects in the game and check if the ball is hitting them

    """
    Gravity calculation
    """
    # TODO: Test if the ball ever has velocity of state.plunger_power because right now we always
    # immediately deduct the gravity from the velocity
    # Direction has to be figured into the gravity calculation
    gravity_delta = jnp.where(
        jnp.logical_or(ball_direction == 0, ball_direction == 2), -GRAVITY, GRAVITY
    )  # Subtract gravity if the ball is moving up otherwise add it
    ball_vel_y = jnp.where(state.ball_in_play, ball_vel_y + gravity_delta, ball_vel_y)
    ball_direction = jnp.where(
        ball_vel_y < 0,
        ball_direction + 1,
        ball_direction,
    )  # Change y direction if the y velocity should be negative i.e. ball now falling
    ball_vel_y = jnp.where(
        ball_vel_y < 0, -ball_vel_y, ball_vel_y
    )  # Make sure y velocity is positive

    """
    Ball movement calculation observing its direction 
    """
    should_invert_x_vel = jnp.logical_or(ball_direction == 2, ball_direction == 3)
    should_invert_y_vel = jnp.logical_or(ball_direction == 1, ball_direction == 3)
    signed_ball_vel_x = jnp.where(should_invert_x_vel, -ball_vel_x, ball_vel_x)
    signed_ball_vel_y = jnp.where(should_invert_y_vel, -ball_vel_y, ball_vel_y)

    # Only change position, direction and velocity if the ball is in play
    ball_x = jnp.where(
        state.ball_in_play, state.ball_x + signed_ball_vel_x, BALL_START_X
    )
    ball_y = jnp.where(
        state.ball_in_play, state.ball_y + signed_ball_vel_y, BALL_START_Y
    )
    # Clip the ball velocity to the maximum speed
    ball_vel_x = jnp.clip(ball_vel_x, 0, BALL_MAX_SPEED)
    ball_vel_y = jnp.clip(ball_vel_y, 0, BALL_MAX_SPEED)

    """
    Check if ball is in play if not ignore the calculations
    """
    # TODO: Maybe do the stuff above in a function that is called if we are in play
    ball_direction = jnp.where(state.ball_in_play, ball_direction, BALL_START_DIRECTION)
    ball_vel_x = jnp.where(state.ball_in_play, ball_vel_x, jnp.array(0))
    ball_vel_y = jnp.where(state.ball_in_play, ball_vel_y, jnp.array(0))

    return ball_x, ball_y, ball_direction, ball_vel_x, ball_vel_y


def enemy_step(state, step_counter, ball_y, ball_speed_y):
    # Skip movement every 8th step
    should_move = step_counter % 8 != 0

    # Calculate direction (-1 for up, 0 for stay, 1 for down)
    direction = jnp.sign(ball_y - state.enemy_y)

    # Calculate new position
    new_y = state.enemy_y + (direction * ENEMY_STEP_SIZE).astype(jnp.int32)
    # Return either new position or current position based on should_move
    return jax.lax.cond(
        should_move, lambda _: new_y, lambda _: state.enemy_y, operand=None
    )


@jax.jit
def _reset_ball_after_goal(
    state_and_goal: Tuple[VideoPinballState, bool],
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    """
    Determines new ball position and velocity after a goal.
    Args:
        state_and_goal: Tuple of (current state, whether goal was scored on right side)
    Returns:
        Tuple of (ball_x, ball_y, ball_vel_x, ball_vel_y) as int32 arrays
    """
    state, scored_right = state_and_goal

    # Determine Y velocity direction based on ball position
    ball_vel_y = jnp.where(
        state.ball_y > BALL_START_Y,
        1,  # Ball was in lower half, go down
        -1,  # Ball was in upper half, go up
    ).astype(jnp.int32)

    # X velocity is always towards the side that just got scored on
    ball_vel_x = jnp.where(
        scored_right, 1, -1  # Ball moves right  # Ball moves left
    ).astype(jnp.int32)

    return (
        BALL_START_X.astype(jnp.int32),
        BALL_START_Y.astype(jnp.int32),
        ball_vel_x.astype(jnp.int32),
        ball_vel_y.astype(jnp.int32),
    )


class JaxPong(
    JaxEnvironment[VideoPinballState, VideoPinballObservation, VideoPinballInfo]
):
    def __init__(self, frameskip: int = 0, reward_funcs: list[callable] = None):
        super().__init__()
        self.frameskip = frameskip + 1
        self.frame_stack_size = 4
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.action_set = {
            NOOP,
            FIRE,
            RIGHT,
            LEFT,
        }
        self.obs_size = 3 * 4 + 1 + 1

    def reset(self) -> VideoPinballState:
        """
        Resets the game state to the initial state.
        Returns the initial state and the reward (i.e. 0)
        """
        state = VideoPinballState(
            player_y=jnp.array(96).astype(jnp.int32),
            player_speed=jnp.array(0.0).astype(jnp.int32),
            ball_x=jnp.array(78).astype(jnp.int32),
            ball_y=jnp.array(115).astype(jnp.int32),
            enemy_y=jnp.array(115).astype(jnp.int32),
            enemy_speed=jnp.array(0.0).astype(jnp.int32),
            ball_vel_x=BALL_SPEED[0].astype(jnp.int32),
            ball_vel_y=BALL_SPEED[1].astype(jnp.int32),
            player_score=jnp.array(0).astype(jnp.int32),
            enemy_score=jnp.array(0).astype(jnp.int32),
            step_counter=jnp.array(0).astype(jnp.int32),
            acceleration_counter=jnp.array(0).astype(jnp.int32),
            buffer=jnp.array(96).astype(jnp.int32),
            obs_stack=None,
        )
        initial_obs = self._get_observation(state)

        def expand_and_copy(x):
            x_expanded = jnp.expand_dims(x, axis=0)
            return jnp.concatenate([x_expanded] * self.frame_stack_size, axis=0)

        # Apply transformation to each leaf in the pytree
        initial_obs = jax.tree.map(expand_and_copy, initial_obs)

        new_state = state._replace(obs_stack=initial_obs)
        return new_state, initial_obs

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, state: VideoPinballState, action: chex.Array
    ) -> Tuple[
        VideoPinballState, VideoPinballObservation, float, bool, VideoPinballInfo
    ]:
        # chex provides jax with additional debug/testing functionality.
        # Probably best to use it instead of simply jnp.array

        # Step 1: Update player position and speed
        # only execute player step on even steps (base implementation only moves the player every second tick)
        new_player_y, player_speed_b, new_acceleration_counter = player_step(
            state.player_y, state.player_speed, state.acceleration_counter, action
        )

        new_player_y, player_speed, new_acceleration_counter = jax.lax.cond(
            state.step_counter % 2 == 0,
            lambda _: (new_player_y, player_speed_b, new_acceleration_counter),
            lambda _: (state.player_y, state.player_speed, state.acceleration_counter),
            operand=None,
        )

        buffer = jax.lax.cond(
            jax.lax.eq(state.buffer, state.player_y),
            lambda _: new_player_y,
            lambda _: state.buffer,
            operand=None,
        )
        player_y = state.buffer

        enemy_y = enemy_step(state, state.step_counter, state.ball_y, state.ball_y)

        # Step 2: Update ball position and velocity
        ball_x, ball_y, ball_vel_x, ball_vel_y = ball_step(state, action)

        # Step 3: Score and goal detection
        player_goal = ball_x < 4
        enemy_goal = ball_x > 156
        ball_reset = jnp.logical_or(enemy_goal, player_goal)

        # Step 4: Update scores
        player_score = jax.lax.cond(
            player_goal,
            lambda s: s + 1,
            lambda s: s,
            operand=state.player_score,
        )
        enemy_score = jax.lax.cond(
            enemy_goal,
            lambda s: s + 1,
            lambda s: s,
            operand=state.enemy_score,
        )

        # Step 5: Reset ball if goal was scored
        current_values = (
            ball_x.astype(jnp.int32),
            ball_y.astype(jnp.int32),
            ball_vel_x.astype(jnp.int32),
            ball_vel_y.astype(jnp.int32),
        )
        ball_x_final, ball_y_final, ball_vel_x_final, ball_vel_y_final = jax.lax.cond(
            ball_reset,
            lambda x: _reset_ball_after_goal((state, enemy_goal)),
            lambda x: x,
            operand=current_values,
        )

        # Step 6: Update step counter for game freeze after goal
        step_counter = jax.lax.cond(
            ball_reset,
            lambda s: jnp.array(0),
            lambda s: s + 1,
            operand=state.step_counter,
        )

        # Step 7: Update enemy position and speed

        # Step 8: Reset enemy position on goal
        enemy_y_final = jax.lax.cond(
            ball_reset,
            lambda s: BALL_START_Y.astype(jnp.int32),
            lambda s: enemy_y.astype(jnp.int32),
            operand=None,
        )

        # Step 9: Handle ball position during game freeze
        ball_x_final = jax.lax.cond(
            step_counter < 60,
            lambda s: BALL_START_X.astype(jnp.int32),
            lambda s: s,
            operand=ball_x_final,
        )
        ball_y_final = jax.lax.cond(
            step_counter < 60,
            lambda s: BALL_START_Y.astype(jnp.int32),
            lambda s: s,
            operand=ball_y_final,
        )

        new_state = PongState(
            player_y=player_y,
            player_speed=player_speed,
            ball_x=ball_x_final,
            ball_y=ball_y_final,
            enemy_y=enemy_y_final,
            enemy_speed=0,
            ball_vel_x=ball_vel_x_final,
            ball_vel_y=ball_vel_y_final,
            player_score=player_score,
            enemy_score=enemy_score,
            step_counter=step_counter,
            acceleration_counter=new_acceleration_counter,
            buffer=buffer,
            obs_stack=state.obs_stack,  # old for now
        )

        done = self._get_done(new_state)
        env_reward = self._get_env_reward(state, new_state)
        all_rewards = self._get_all_reward(state, new_state)
        info = self._get_info(new_state, all_rewards)

        observation = self._get_observation(new_state)
        # stack the new observation, remove the oldest one
        observation = jax.tree.map(
            lambda stack, obs: jnp.concatenate(
                [stack[1:], jnp.expand_dims(obs, axis=0)], axis=0
            ),
            new_state.obs_stack,
            observation,
        )
        new_state = new_state._replace(obs_stack=observation)

        return new_state, new_state.obs_stack, env_reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: PongState):
        # create player
        player = EntityPosition(
            x=jnp.array(PLAYER_X),
            y=state.player_y,
            width=jnp.array(PLAYER_SIZE[0]),
            height=jnp.array(PLAYER_SIZE[1]),
        )

        # create enemy
        enemy = EntityPosition(
            x=jnp.array(ENEMY_X),
            y=state.enemy_y,
            width=jnp.array(ENEMY_SIZE[0]),
            height=jnp.array(ENEMY_SIZE[1]),
        )

        ball = EntityPosition(
            x=state.ball_x,
            y=state.ball_y,
            width=jnp.array(BALL_SIZE[0]),
            height=jnp.array(BALL_SIZE[1]),
        )
        return PongObservation(
            player=player,
            enemy=enemy,
            ball=ball,
            score_player=state.player_score,
            score_enemy=state.enemy_score,
        )

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: PongObservation) -> jnp.ndarray:
        return jnp.concatenate(
            [
                obs.player.x.flatten(),
                obs.player.y.flatten(),
                obs.player.height.flatten(),
                obs.player.width.flatten(),
                obs.enemy.x.flatten(),
                obs.enemy.y.flatten(),
                obs.enemy.height.flatten(),
                obs.enemy.width.flatten(),
                obs.ball.x.flatten(),
                obs.ball.y.flatten(),
                obs.ball.height.flatten(),
                obs.ball.width.flatten(),
                obs.score_player.flatten(),
                obs.score_enemy.flatten(),
            ]
        )

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
    def _get_info(self, state: PongState, all_rewards: chex.Array) -> PongInfo:
        return PongInfo(time=state.step_counter, all_rewards=all_rewards)

    @partial(jax.jit, static_argnums=(0,))
    def _get_env_reward(self, previous_state: PongState, state: PongState):
        return (state.player_score - state.enemy_score) - (
            previous_state.player_score - previous_state.enemy_score
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(self, previous_state: PongState, state: PongState):
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array(
            [reward_func(previous_state, state) for reward_func in self.reward_funcs]
        )
        return rewards

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: PongState) -> bool:
        return jnp.logical_or(
            jnp.greater_equal(state.player_score, 20),
            jnp.greater_equal(state.enemy_score, 20),
        )


def load_sprites():
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Define the base directory for sprites relative to the script
    SPRITES_BASE_DIR = os.path.join(MODULE_DIR, "sprites/videopinball") # Assuming sprites are in a 'sprites/videopinball' subdirectory


    # Load sprites
    sprite_background = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "Background.npy"), transpose=True)
    sprite_ball = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "Ball.npy"), transpose=True)

    sprite_atari_logo = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "AtariLogo.npy"), transpose=True)
    sprite_x = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "X.npy"), transpose=True)
    sprite_yellow_diamond_bottom = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "YellowDiamondBottom.npy"), transpose=True)
    sprite_yellow_diamond_top = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "YellowDiamondTop.npy"), transpose=True)

    # sprite_wall_bottom_left_square = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "WallBottomLeftSquare.npy"), transpose=True)
    # sprite_wall_bumper = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "WallBumper.npy"), transpose=True)
    # sprite_wall_dropper = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "WallDropper.npy"), transpose=True)
    # sprite_wall_left_l = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "WallLeftL.npy"), transpose=True)
    # sprite_wall_outer = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "WallOuter.npy"), transpose=True)
    # sprite_wall_right_l = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "WallRightL.npy"), transpose=True)
    # sprite_wall_small_horizontal = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "WallSmallHorizontal.npy"), transpose=True)
    sprite_walls = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "Walls.npy"), transpose=True)

    # Animated sprites
    sprite_spinner0 = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "SpinnerBottom.npy"), transpose=True)
    sprite_spinner1 = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "SpinnerRight.npy"), transpose=True)
    sprite_spinner2 = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "SpinnerTop.npy"), transpose=True)
    sprite_spinner3 = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "SpinnerLeft.npy"), transpose=True)

    sprite_launcher0 = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "Launcher0.npy"), transpose=True)
    sprite_launcher1 = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "Launcher1.npy"), transpose=True)
    sprite_launcher2 = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "Launcher2.npy"), transpose=True)
    sprite_launcher3 = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "Launcher3.npy"), transpose=True)
    sprite_launcher4 = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "Launcher4.npy"), transpose=True)
    sprite_launcher5 = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "Launcher4.npy"), transpose=True)
    sprite_launcher6 = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "Launcher6.npy"), transpose=True)
    sprite_launcher7 = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "Launcher7.npy"), transpose=True)
    sprite_launcher8 = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "Launcher8.npy"), transpose=True)
    sprite_launcher9 = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "Launcher9.npy"), transpose=True)
    sprite_launcher10 = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "Launcher10.npy"), transpose=True)
    sprite_launcher11 = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "Launcher11.npy"), transpose=True)
    sprite_launcher12 = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "Launcher12.npy"), transpose=True)
    sprite_launcher13 = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "Launcher13.npy"), transpose=True)
    sprite_launcher14 = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "Launcher14.npy"), transpose=True)
    sprite_launcher15 = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "Launcher15.npy"), transpose=True)
    sprite_launcher16 = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "Launcher16.npy"), transpose=True)
    sprite_launcher17 = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "Launcher17.npy"), transpose=True)
    sprite_launcher18 = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "Launcher18.npy"), transpose=True)

    sprite_flipper_left0 = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "FlipperLeft0.npy"), transpose=True)
    sprite_flipper_left1 = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "FlipperLeft1.npy"), transpose=True)
    sprite_flipper_left2 = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "FlipperLeft2.npy"), transpose=True)
    sprite_flipper_left3 = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "FlipperLeft3.npy"), transpose=True)
    sprite_flipper_right0 = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "FlipperRight0.npy"), transpose=True)
    sprite_flipper_right1 = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "FlipperRight1.npy"), transpose=True)
    sprite_flipper_right2 = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "FlipperRight2.npy"), transpose=True)
    sprite_flipper_right3 = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "FlipperRight3.npy"), transpose=True)

    sprites_spinner = aj.pad_to_match([sprite_spinner0, sprite_spinner1, sprite_spinner2, sprite_spinner3])
    sprites_spinner = jnp.concatenate([
        jnp.repeat(sprites_spinner[0][None], 2, axis=0),
        jnp.repeat(sprites_spinner[1][None], 2, axis=0),
        jnp.repeat(sprites_spinner[2][None], 2, axis=0),
        jnp.repeat(sprites_spinner[3][None], 2, axis=0)
    ])


    sprites_plunger = aj.pad_to_match_top([sprite_launcher0, sprite_launcher1, sprite_launcher2, sprite_launcher3,
                                        sprite_launcher4, sprite_launcher5, sprite_launcher6, sprite_launcher7,
                                        sprite_launcher8, sprite_launcher9, sprite_launcher10, sprite_launcher11,
                                        sprite_launcher12, sprite_launcher13, sprite_launcher14, sprite_launcher15,
                                        sprite_launcher16, sprite_launcher17, sprite_launcher18])



    # sprites_plunger = jnp.concatenate([
    #     jnp.repeat(sprites_plunger[0][None], 1, axis=0),
    #     jnp.repeat(sprites_plunger[1][None], 1, axis=0),
    #     jnp.repeat(sprites_plunger[2][None], 1, axis=0),
    #     jnp.repeat(sprites_plunger[3][None], 1, axis=0),
    #     jnp.repeat(sprites_plunger[4][None], 1, axis=0),
    #     jnp.repeat(sprites_plunger[5][None], 1, axis=0),
    #     jnp.repeat(sprites_plunger[6][None], 1, axis=0),
    #     jnp.repeat(sprites_plunger[7][None], 1, axis=0),
    #     jnp.repeat(sprites_plunger[8][None], 1, axis=0),
    #     jnp.repeat(sprites_plunger[9][None], 1, axis=0),
    #     jnp.repeat(sprites_plunger[10][None], 1, axis=0),
    #     jnp.repeat(sprites_plunger[11][None], 1, axis=0),
    #     jnp.repeat(sprites_plunger[12][None], 1, axis=0),
    #     jnp.repeat(sprites_plunger[13][None], 1, axis=0),
    #     jnp.repeat(sprites_plunger[14][None], 1, axis=0),
    #     jnp.repeat(sprites_plunger[15][None], 1, axis=0),
    #     jnp.repeat(sprites_plunger[16][None], 1, axis=0),
    #     jnp.repeat(sprites_plunger[17][None], 1, axis=0),
    #     jnp.repeat(sprites_plunger[18][None], 1, axis=0)
    # ])

    sprites_flipper_left = aj.pad_to_match([sprite_flipper_left0, sprite_flipper_left1,
                                            sprite_flipper_left2, sprite_flipper_left3])

    # sprites_flipper_left = jnp.concatenate([
    #     jnp.repeat(sprites_flipper_left[0][None], 2, axis=0),
    #     jnp.repeat(sprites_flipper_left[1][None], 2, axis=0),
    #     jnp.repeat(sprites_flipper_left[2][None], 2, axis=0),
    #     jnp.repeat(sprites_flipper_left[3][None], 2, axis=0)
    # ])

    sprites_flipper_right = aj.pad_to_match([sprite_flipper_right0, sprite_flipper_right1,
                                             sprite_flipper_right2, sprite_flipper_right3])

    # sprites_flipper_right = jnp.concatenate([
    #     jnp.repeat(sprites_flipper_right[0][None], 2, axis=0),
    #     jnp.repeat(sprites_flipper_right[1][None], 2, axis=0),
    #     jnp.repeat(sprites_flipper_right[2][None], 2, axis=0),
    #     jnp.repeat(sprites_flipper_right[3][None], 2, axis=0)
    # ])

    sprites_plunger = jnp.stack(sprites_plunger, axis=0)
    sprites_flipper_left = jnp.stack(sprites_flipper_left, axis=0)
    sprites_flipper_right = jnp.stack(sprites_flipper_right, axis=0)

    # Load number sprites
    sprites_score_numbers = aj.load_and_pad_digits(
        os.path.join(SPRITES_BASE_DIR, "ScoreNumber{}.npy"),
        num_chars=10,  # For digits 0 through 9
    )

    # TODO: Check if this works, might require a dummy fieldnumber0 sprite
    sprites_field_numbers = aj.load_and_pad_digits(
        os.path.join(SPRITES_BASE_DIR, "FieldNumber{}.npy"),
        num_chars=10,  # Load 0-9, even if you only use 1-9
    )

    sprite_background =  jnp.expand_dims(sprite_background, axis=0)
    sprite_ball = jnp.expand_dims(sprite_ball, axis=0)
    sprite_walls = jnp.expand_dims(sprite_walls, axis=0)

    sprite_atari_logo = jnp.expand_dims(sprite_atari_logo, axis=0)
    sprite_x = jnp.expand_dims(sprite_x, axis=0)
    sprite_yellow_diamond_bottom = jnp.expand_dims(sprite_yellow_diamond_bottom, axis=0)
    sprite_yellow_diamond_top = jnp.expand_dims(sprite_yellow_diamond_top, axis=0)

    # This was commented in Pong, no idea if its needed (probably not)
    # sprite_background = jax.image.resize(sprite_background, (WIDTH, HEIGHT, 4), method='bicubic')


    return {
        "atari_logo": sprite_atari_logo,
        "background": sprite_background,
        "ball": sprite_ball,
        "spinner": sprites_spinner,
        "x": sprite_x,
        "yellow_diamond_bottom": sprite_yellow_diamond_bottom,
        "yellow_diamond_top": sprite_yellow_diamond_top,

        # "wall_bottom_left_square": sprite_wall_bottom_left_square,
        # "wall_bumper": sprite_wall_bumper,
        # "wall_dropper": sprite_wall_dropper,
        # "wall_left_l": sprite_wall_left_l,
        # "wall_outer": sprite_wall_outer,
        # "wall_right_l": sprite_wall_right_l,
        # "wall_small_horizontal": sprite_wall_small_horizontal,
        "walls": sprite_walls,

        # Animated sprites
        "flipper_left": sprites_flipper_left,
        "flipper_right": sprites_flipper_right,
        "plunger": sprites_plunger,

        # Digit sprites
        "score_number_digits": sprites_score_numbers,
        "field_number_digits": sprites_field_numbers,
    }


class Renderer_AtraJaxisVideoPinball:
    """JAX-based Video Pinball game renderer, optimized with JIT compilation."""

    def __init__(self):
        self.sprites = load_sprites()

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        """
        Renders the current game state using JAX operations.

        Args:
            state: A VideoPinballState object containing the current game state.

        Returns:
            A JAX array representing the rendered frame.
        """
        # Create empty raster with CORRECT orientation for atraJaxis framework
        # Note: For pygame, the raster is expected to be (width, height, channels)
        # where width corresponds to the horizontal dimension of the screen
        raster = jnp.zeros((WIDTH, HEIGHT, 3))

        # Render static objects
        frame_bg = aj.get_sprite_frame(self.sprites["background"], 0)
        raster = aj.render_at(raster, 0, 0, frame_bg)

        frame_walls = aj.get_sprite_frame(self.sprites["walls"], 0)
        raster = aj.render_at(raster, 16, 0, frame_walls)


        # Render animated objects TODO: (unfinished, game_state implementation needed)
        frame_flipper_left = aj.get_sprite_frame(self.sprites["flipper_left"], 0)
        raster = aj.render_at(raster, 184, 64, frame_flipper_left)

        frame_flipper_right = aj.get_sprite_frame(self.sprites["flipper_right"], 0)
        raster = aj.render_at(raster, 184, 83, frame_flipper_right)

        frame_plunger = aj.get_sprite_frame(self.sprites["plunger"], 0)
        raster = aj.render_at(raster, 133, 148, frame_plunger)

        frame_spinner = aj.get_sprite_frame(self.sprites["spinner"], state.step_counter % 8)
        raster = aj.render_at(raster, 90, 30, frame_spinner)
        raster = aj.render_at(raster, 90, 126, frame_spinner)

        frame_ball = aj.get_sprite_frame(self.sprites["ball"], 0)
        raster = aj.render_at(raster, state.ball_y, state.ball_x, frame_ball)


        # Render score TODO: (unfinished, game_state implementation needed)
        frame_unknown = aj.get_sprite_frame(self.sprites["score_number_digits"], 1)
        raster = aj.render_at(raster, 3, 4, frame_unknown)

        frame_ball_count = aj.get_sprite_frame(self.sprites["score_number_digits"], 1)
        raster = aj.render_at(raster, 3, 36, frame_ball_count)

        frame_score1 = aj.get_sprite_frame(self.sprites["score_number_digits"], 0)
        raster = aj.render_at(raster, 3, 64, frame_score1)
        frame_score2 = aj.get_sprite_frame(self.sprites["score_number_digits"], 0)
        raster = aj.render_at(raster, 3, 80, frame_score2)
        frame_score3 = aj.get_sprite_frame(self.sprites["score_number_digits"], 0)
        raster = aj.render_at(raster, 3, 96, frame_score3)
        frame_score4 = aj.get_sprite_frame(self.sprites["score_number_digits"], 0)
        raster = aj.render_at(raster, 3, 112, frame_score4)
        frame_score5 = aj.get_sprite_frame(self.sprites["score_number_digits"], 0)
        raster = aj.render_at(raster, 3, 128, frame_score5)
        frame_score6 = aj.get_sprite_frame(self.sprites["score_number_digits"], 0)
        raster = aj.render_at(raster, 3, 144, frame_score6)




        # Render special yellow field objects TODO: (unfinished, game_state implementation needed)
        frame_bumper_left = aj.get_sprite_frame(self.sprites["field_number_digits"], 1)
        raster = aj.render_at(raster, 122, 46, frame_bumper_left)
        frame_bumper_middle = aj.get_sprite_frame(self.sprites["field_number_digits"], 1)
        raster = aj.render_at(raster, 58, 78, frame_bumper_middle)
        frame_bumper_right = aj.get_sprite_frame(self.sprites["field_number_digits"], 1)
        raster = aj.render_at(raster, 122, 110, frame_bumper_right)

        frame_dropper_left = aj.get_sprite_frame(self.sprites["field_number_digits"], 1)
        raster = aj.render_at(raster, 58, 46, frame_dropper_left)
        frame_dropper_right = aj.get_sprite_frame(self.sprites["atari_logo"], 0)
        raster = aj.render_at(raster, 58, 109, frame_dropper_right)

        frame_diamond = aj.get_sprite_frame(self.sprites["yellow_diamond_top"], 0)
        raster = aj.render_at(raster, 24, 60, frame_diamond)
        raster = aj.render_at(raster, 24, 76, frame_diamond)
        raster = aj.render_at(raster, 24, 92, frame_diamond)

        frame_special_diamond = aj.get_sprite_frame(self.sprites["yellow_diamond_bottom"], 0)
        raster = aj.render_at(raster, 120, 76, frame_special_diamond)


        return raster




if __name__ == "__main__":
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Pong Game")
    clock = pygame.time.Clock()

    game = JaxPong(frameskip=1)

    # Create the JAX renderer
    renderer = Renderer_AtraJaxisVideoPinball()

    # Get jitted functions
    jitted_step = jax.jit(game.step)
    jitted_reset = jax.jit(game.reset)

    curr_state, obs = jitted_reset()

    # Game loop
    running = True
    frame_by_frame = False
    frameskip = game.frameskip
    counter = 1

    while running:
        # Event loop that checks display settings (QUIT, frame-by-frame-mode toggled, next for frame-by-frame mode)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    frame_by_frame = not frame_by_frame
            elif event.type == pygame.KEYDOWN or (
                event.type == pygame.KEYUP and event.key == pygame.K_n
            ):
                if event.key == pygame.K_n and frame_by_frame:
                    if counter % frameskip == 0:
                        # If frame-by-frame mode activated and next frame is requested,
                        # get human (game) action and perform step
                        action = get_human_action()
                        curr_state, obs, reward, done, info = jitted_step(
                            curr_state, action
                        )

        if not frame_by_frame:
            # If not in frame-by-frame mode perform step at each clock-tick
            # i.e. get human (game) action
            if counter % frameskip == 0:
                action = get_human_action()
                # Update game step
                curr_state, obs, reward, done, info = jitted_step(curr_state, action)

        # Render and display
        raster = renderer.render(curr_state)

        aj.update_pygame(screen, raster, 3, WIDTH, HEIGHT)

        counter += 1
        clock.tick(60)

    pygame.quit()
