import os
from functools import partial
from typing import NamedTuple, Tuple
import jax.lax
import jax.numpy as jnp
import chex
import pygame
from gymnax.environments import spaces

from jaxatari.rendering import atraJaxis as aj
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action

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
FLIPPER_MAX_STRENGTH = 3
PLUNGER_MAX_POSITION = 20

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
BALL_START_X = jnp.array(149)
BALL_START_Y = jnp.array(129)
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
    ball_direction: chex.Array  # 0: left/up, 1:left/down , 2: right/up, 3: right/down (Shouldn't this be a function?)
    left_flipper_angle: chex.Array
    right_flipper_angle: chex.Array
    plunger_position: chex.Array  # Value between 0 and 20 where 20 means that the plunger is fully pulled
    score: chex.Array
    lives: chex.Array
    bonus_multiplier: chex.Array
    bumpers_active: chex.Array
    targets_hit: chex.Array
    step_counter: chex.Array
    ball_in_play: chex.Array
    # obs_stack: chex.ArrayTree     What is this for? Pong doesnt have this right?


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
def plunger_step(state: VideoPinballState, action: chex.Array) -> chex.Array:
    """
    Update the plunger position based on the current state and action.
    And set the plunger power to 2 * plunger_position.
    """

    # if ball is not in play and DOWN was clicked, move plunger down
    plunger_position = jax.lax.cond(
        jnp.logical_and(state.plunger_position < PLUNGER_MAX_POSITION, jnp.logical_and(action == Action.DOWN, jnp.logical_not(state.ball_in_play))),
        lambda s: s + 1,
        lambda s: s,
        operand=state.plunger_position
    )

    # same for UP
    plunger_position = jax.lax.cond(
        jnp.logical_and(state.plunger_position > 0, jnp.logical_and(action == Action.UP, jnp.logical_not(state.ball_in_play))),
        lambda s: s - 1,
        lambda s: s,
        operand=state.plunger_position
    )

    # If FIRE
    plunger_power = jax.lax.cond(
        jnp.logical_and(action == Action.FIRE, jnp.logical_not(state.ball_in_play)),
        lambda s: s * 2,
        lambda s: 0,
        operand=state.plunger_position
    )

    return plunger_position, plunger_power


@jax.jit
def flipper_step(
    state: VideoPinballState, action: chex.Array
):

    left_flipper_angle = jax.lax.cond(
        jnp.logical_and(action == Action.LEFT, state.left_flipper_angle < FLIPPER_MAX_STRENGTH),
        lambda a: a + 1,
        lambda a: a,
        operand=state.left_flipper_angle
    )

    right_flipper_angle = jax.lax.cond(
        jnp.logical_and(action == Action.RIGHT, state.right_flipper_angle < FLIPPER_MAX_STRENGTH),
        lambda a: a + 1,
        lambda a: a,
        operand=state.right_flipper_angle
    )

    # TODO update angles based on step phase?

    # TODO ball acceleration should be computed from current and new plunger/flipper states or some other way
    # _ball_property_ = ...


    
    return left_flipper_angle, right_flipper_angle


@jax.jit
def hit_obstacle_to_left(
    ball_x: chex.Array, ball_y: chex.Array, obstacle_: chex.Array
) -> bool:
    """
    Check if the ball is hitting an obstacle to the left.
    """
    return jnp.logical_and(
        ball_x < 7, # I changed this obstacle_x variable so the code runs
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
    ball_vel_y = jnp.where(
        state.plunger_position * 2 > 0,
        ball_vel_y + state.plunger_position * 2,
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

    # TODO add ball_in_play to return and if plunger hit set to True
    return ball_x, ball_y, ball_direction, ball_vel_x, ball_vel_y





@jax.jit
def _reset_ball_after_losing_life(state: VideoPinballState):
    """
    When the ball goes into the gutter and you lose a life, respawn the ball on the launcher
    and possibly adjust the ball/life counter
    """
    return BALL_START_X, BALL_START_Y, 0, 0


class JaxVideoPinball(
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

    def reset(self) -> Tuple[VideoPinballState, VideoPinballObservation]:
        """
        Resets the game state to the initial state.
        Returns the initial state and the reward (i.e. 0)
        """
        state = VideoPinballState(
            ball_x=jnp.array(BALL_START_X).astype(jnp.int32),
            ball_y=jnp.array(BALL_START_Y).astype(jnp.int32),
            ball_vel_x=jnp.array(0).astype(jnp.int32),
            ball_vel_y=jnp.array(0).astype(jnp.int32),
            ball_direction=jnp.array(0).astype(jnp.int32), # Necessary?
            left_flipper_angle=jnp.array(0).astype(jnp.int32),
            right_flipper_angle=jnp.array(0).astype(jnp.int32),
            plunger_position=jnp.array(0).astype(jnp.int32),
            score=jnp.array(0).astype(jnp.int32),
            lives=jnp.array(1).astype(jnp.int32),
            bonus_multiplier=jnp.array(1).astype(jnp.int32),
            bumpers_active=jnp.array([1,1,1]).astype(jnp.int32),
            targets_hit=jnp.array([1,1,1]).astype(jnp.int32),
            step_counter=jnp.array(0).astype(jnp.int32),
            ball_in_play=jnp.array(True).astype(jnp.bool),
        )
        initial_obs = self._get_observation(state)
        return state, initial_obs #, initial_obs



    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, state: VideoPinballState, action: chex.Array
    ) -> Tuple[
        VideoPinballState, VideoPinballObservation, float, bool, VideoPinballInfo
    ]:
        # chex provides jax with additional debug/testing functionality.
        # Probably best to use it instead of simply jnp.array

        # Step 1: Update ball position and velocity
        ball_x, ball_y, ball_direction, ball_vel_x, ball_vel_y = ball_step(state, action)

        # Step 2: Check if ball is in the gutter
        ball_reset = ball_y > 192

        # Step 3: Update Plunger and Flippers
        plunger_position, plunger_strength = plunger_step(state, action)
        left_flipper_angle, right_flipper_angle = flipper_step(state, action)

        # Step 4: Update scores
        score = jnp.array(0).astype(jnp.int32)
        bonus_multiplier = jnp.array(1).astype(jnp.int32)

        # Step 5: Update Objects on Hit (like Bumpers and Targets)
        bumpers_active = jnp.array([1, 1, 1]).astype(jnp.int32)
        targets_hit = jnp.array([1, 1, 1]).astype(jnp.int32)

        # Step 5: Reset ball if it went down the gutter
        current_values = (
            ball_x.astype(jnp.int32),
            ball_y.astype(jnp.int32),
            ball_vel_x.astype(jnp.int32),
            ball_vel_y.astype(jnp.int32),
        )
        ball_x_final, ball_y_final, ball_vel_x_final, ball_vel_y_final = jax.lax.cond(
            ball_reset,
            lambda x: _reset_ball_after_losing_life(state),
            lambda x: x,
            operand=current_values,
        )

        lives = jax.lax.cond(
            ball_reset,
            lambda x: x + 1, #Because it's not really lives but more like a ball count? You start at 1 and it goes up to 3
            lambda x: x,
            operand=state.lives,
        )


        new_state = VideoPinballState(
            ball_x=ball_x_final,
            ball_y=ball_y_final,
            ball_vel_x=ball_vel_x_final,
            ball_vel_y=ball_vel_y_final,
            ball_direction=ball_direction,  # Necessary?
            left_flipper_angle=left_flipper_angle,
            right_flipper_angle=right_flipper_angle,
            plunger_position=plunger_position,
            score=score,
            lives=lives,
            bonus_multiplier=bonus_multiplier,
            bumpers_active=bumpers_active,
            targets_hit=targets_hit,
            step_counter=jnp.array(state.step_counter + 1).astype(jnp.int32),
            ball_in_play=True, # Necessary?
            # obs_stack=None,
        )

        done = self._get_done(new_state)
        env_reward = self._get_env_reward(state, new_state)
        all_rewards = self._get_all_reward(state, new_state)
        info = self._get_info(new_state, all_rewards)

        observation = self._get_observation(new_state)
        # stack the new observation, remove the oldest one
        # observation = jax.tree.map(
        #     lambda stack, obs: jnp.concatenate(
        #         [stack[1:], jnp.expand_dims(obs, axis=0)], axis=0
        #     ),
        #     new_state.obs_stack,
        #     observation,
        # )
        # new_state = new_state._replace(obs_stack=observation)

        return new_state, observation, env_reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: VideoPinballState):

        ball = EntityPosition(
            x=state.ball_x,
            y=state.ball_y,
            width=jnp.array(BALL_SIZE[0]),
            height=jnp.array(BALL_SIZE[1]),
        )


        # TODO: Implement proper positions and sizes for flippers and plunger
        left_flipper = EntityPosition(
            x=state.ball_x,
            y=state.ball_y,
            width=jnp.array(BALL_SIZE[0]),
            height=jnp.array(BALL_SIZE[1]),
        )

        right_flipper = EntityPosition(
            x=state.ball_x,
            y=state.ball_y,
            width=jnp.array(BALL_SIZE[0]),
            height=jnp.array(BALL_SIZE[1]),
        )

        plunger = EntityPosition(
            x=state.ball_x,
            y=state.ball_y,
            width=jnp.array(BALL_SIZE[0]),
            height=jnp.array(BALL_SIZE[1]),
        )

        return VideoPinballObservation(
            ball = ball,
            left_flipper = left_flipper,
            right_flipper = right_flipper,
            plunger = plunger,
            bumpers = state.bumpers_active,  # bumper states array
            targets = state.targets_hit, # target states array
            score = state.score,
            lives = state.lives,
            bonus_multiplier = state.bonus_multiplier
        )

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: VideoPinballObservation) -> jnp.ndarray:
        return jnp.concatenate(
            [
                obs.ball.x.flatten(),
                obs.ball.y.flatten(),
                obs.ball.height.flatten(),
                obs.ball.width.flatten(),
                obs.left_flipper.x.flatten(),
                obs.left_flipper.y.flatten(),
                obs.left_flipper.height.flatten(),
                obs.left_flipper.width.flatten(),
                obs.right_flipper.x.flatten(),
                obs.right_flipper.y.flatten(),
                obs.right_flipper.height.flatten(),
                obs.right_flipper.width.flatten(),
                obs.bumpers.flatten(),
                obs.targets.flatten(),
                obs.score.flatten(),
                obs.lives.flatten(),
                obs.bonus_multiplier.flatten(),


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
    def _get_info(self, state: VideoPinballState, all_rewards: chex.Array) -> VideoPinballInfo:
        return VideoPinballInfo(time=state.step_counter, all_rewards=all_rewards)

    @partial(jax.jit, static_argnums=(0,))
    def _get_env_reward(self, previous_state: VideoPinballState, state: VideoPinballState):
        return 0

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(self, previous_state: VideoPinballState, state: VideoPinballState):
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array(
            [reward_func(previous_state, state) for reward_func in self.reward_funcs]
        )
        return rewards

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: VideoPinballState) -> bool:
        return False


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
        raster = aj.render_at(raster, 0, 16, frame_walls)


        # Render animated objects TODO: (unfinished, game_state implementation needed)
        frame_flipper_left = aj.get_sprite_frame(self.sprites["flipper_left"], 0)
        raster = aj.render_at(raster, 64, 184, frame_flipper_left)

        frame_flipper_right = aj.get_sprite_frame(self.sprites["flipper_right"], 0)
        raster = aj.render_at(raster, 83, 184, frame_flipper_right)

        frame_plunger = aj.get_sprite_frame(self.sprites["plunger"], 0)
        raster = aj.render_at(raster, 148, 133, frame_plunger)

        frame_spinner = aj.get_sprite_frame(self.sprites["spinner"], state.step_counter % 8)
        raster = aj.render_at(raster, 30, 90, frame_spinner)
        raster = aj.render_at(raster, 126, 90, frame_spinner)

        frame_ball = aj.get_sprite_frame(self.sprites["ball"], 0)
        raster = aj.render_at(raster, state.ball_x, state.ball_y, frame_ball)


        # Render score TODO: (unfinished, game_state implementation needed)
        frame_unknown = aj.get_sprite_frame(self.sprites["score_number_digits"], 1)
        raster = aj.render_at(raster, 4, 3, frame_unknown)

        frame_ball_count = aj.get_sprite_frame(self.sprites["score_number_digits"], 1)
        raster = aj.render_at(raster, 36, 3, frame_ball_count)

        frame_score1 = aj.get_sprite_frame(self.sprites["score_number_digits"], 0)
        raster = aj.render_at(raster, 64, 3, frame_score1)
        frame_score2 = aj.get_sprite_frame(self.sprites["score_number_digits"], 0)
        raster = aj.render_at(raster, 80, 3, frame_score2)
        frame_score3 = aj.get_sprite_frame(self.sprites["score_number_digits"], 0)
        raster = aj.render_at(raster, 96, 3, frame_score3)
        frame_score4 = aj.get_sprite_frame(self.sprites["score_number_digits"], 0)
        raster = aj.render_at(raster, 112, 3, frame_score4)
        frame_score5 = aj.get_sprite_frame(self.sprites["score_number_digits"], 0)
        raster = aj.render_at(raster, 128, 3, frame_score5)
        frame_score6 = aj.get_sprite_frame(self.sprites["score_number_digits"], 0)
        raster = aj.render_at(raster, 144, 3, frame_score6)




        # Render special yellow field objects TODO: (unfinished, game_state implementation needed)
        frame_bumper_left = aj.get_sprite_frame(self.sprites["field_number_digits"], 1)
        raster = aj.render_at(raster, 46, 122, frame_bumper_left)
        frame_bumper_middle = aj.get_sprite_frame(self.sprites["field_number_digits"], 1)
        raster = aj.render_at(raster, 78, 58, frame_bumper_middle)
        frame_bumper_right = aj.get_sprite_frame(self.sprites["field_number_digits"], 1)
        raster = aj.render_at(raster, 110, 122, frame_bumper_right)

        frame_dropper_left = aj.get_sprite_frame(self.sprites["field_number_digits"], 1)
        raster = aj.render_at(raster, 46, 58, frame_dropper_left)
        frame_dropper_right = aj.get_sprite_frame(self.sprites["atari_logo"], 0)
        raster = aj.render_at(raster, 109, 58, frame_dropper_right)

        frame_diamond = aj.get_sprite_frame(self.sprites["yellow_diamond_top"], 0)
        raster = aj.render_at(raster, 60, 24, frame_diamond)
        raster = aj.render_at(raster, 76, 24, frame_diamond)
        raster = aj.render_at(raster, 92, 24, frame_diamond)

        frame_special_diamond = aj.get_sprite_frame(self.sprites["yellow_diamond_bottom"], 0)
        raster = aj.render_at(raster, 76, 120, frame_special_diamond)


        return raster




if __name__ == "__main__":
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Pong Game")
    clock = pygame.time.Clock()

    game = JaxVideoPinball(frameskip=1)

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
