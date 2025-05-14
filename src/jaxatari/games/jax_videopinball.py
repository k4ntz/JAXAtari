"""
Project: JAXAtari VideoPinball
Description: Brief description of the module.

Authors:
    - Team Alpha <team.alpha@example.com>
    - Team Beta <team.beta@example.com>
    - John Doe <johndoe@example.com>
"""

import os
from functools import partial
from typing import NamedTuple, Tuple
import jax.lax
import jax.numpy as jnp
import jax.random as jrandom
import chex
import pygame
from gymnax.environments import spaces

from jaxatari.rendering import atraJaxis as aj
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action

# Constants for game environment
WIDTH = 160
HEIGHT = 210

# Physics constants
# TODO: check if these are correct
GRAVITY = 0.12
BALL_MAX_SPEED = 6.0
FLIPPER_MAX_ANGLE = 3
FLIPPER_ANIMATION_Y_OFFSETS = jnp.array([0, 0, 3, 7]) # This is a little scuffed, it would be cleaner to just fix the sprites but this works fine
FLIPPER_ANIMATION_X_OFFSETS = jnp.array([0, 0, 0, 1]) # Only for the right flipper
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

# fixed position to ensure ball stays at rest until launched, down the drain still needs to be tested
BALL_START_X = jnp.array(149)
BALL_START_Y = jnp.array(129)  
BALL_START_DIRECTION = jnp.array(0)

# Pygame window dimensions
WINDOW_WIDTH = 160 * 3
WINDOW_HEIGHT = 210 * 3

GAME_BOTTOM_Y = 191

TOP_WALL_LEFT_X_OFFSET = 0
TOP_WALL_TOP_Y_OFFSET = 16
RIGHT_WALL_LEFT_X_OFFSET = 152
BOTTOM_WALL_TOP_Y_OFFSET = 184

OUTER_WALL_THICKNESS = 8

# define fixed dimensions for all scene objects to ensure consistency
WALL_WIDTH = 8
WALL_HEIGHT = 176
HORIZONTAL_WALL_WIDTH = 160
HORIZONTAL_WALL_HEIGHT = 8

# can be modified to ensure all bounding boxes have the SAME exact shape
UNIFORM_BOX_SIZE = (10, 10)  

TOP_WALL_BOUNDING_BOX = jnp.ones(UNIFORM_BOX_SIZE).astype(jnp.bool_)
TOP_WALL_OFFSET = jnp.array([TOP_WALL_LEFT_X_OFFSET, TOP_WALL_TOP_Y_OFFSET])

BOTTOM_WALL_BOUNDING_BOX = jnp.ones(UNIFORM_BOX_SIZE).astype(jnp.bool_)
BOTTOM_WALL_OFFSET = jnp.array([TOP_WALL_LEFT_X_OFFSET, BOTTOM_WALL_TOP_Y_OFFSET])

LEFT_WALL_BOUNDING_BOX = jnp.ones(UNIFORM_BOX_SIZE).astype(jnp.bool_)
LEFT_WALL_OFFSET = jnp.array([TOP_WALL_LEFT_X_OFFSET, TOP_WALL_TOP_Y_OFFSET])

RIGHT_WALL_BOUNDING_BOX = jnp.ones(UNIFORM_BOX_SIZE).astype(jnp.bool_)
RIGHT_WALL_OFFSET = jnp.array([RIGHT_WALL_LEFT_X_OFFSET, TOP_WALL_TOP_Y_OFFSET])

WALL_CORNER_BLOCK_WIDTH = 4
WALL_CORNER_BLOCK_HEIGHT = 8

INNER_WALL_TOP_Y = 56
INNER_WALL_TOP_X = 12

INNER_WALL_THICKNESS = 4

MIDDLE_BAR_Y = 72
MIDDLE_BAR_X = 104
MIDDLE_BAR_WIDTH = 16
MIDDLE_BAR_HEIGHT = 8

#need to check if this is correct, not able to find make the score work
# Define constants for scoring events
BUMPER_HIT_POINTS = 10
TARGET_HIT_POINTS = 50
SPINNER_POINTS = 25
ROLLOVER_POINTS = 15
BONUS_MULTIPLIER_POINTS = 100
FLIPPER_HIT_POINTS = 5
WALL_BOUNCE_POINTS = 2


@chex.dataclass
class BallMovement:
    old_ball_x: chex.Array
    old_ball_y: chex.Array
    new_ball_x: chex.Array
    new_ball_y: chex.Array


@chex.dataclass
class SceneObject:
    hit_box_matrix: chex.Array
    hit_box_offset: chex.Array
    reflecting: chex.Array  # 0: no reflection, 1: reflection


TOP_WALL_SCENE_OBJECT = SceneObject(
    hit_box_matrix=TOP_WALL_BOUNDING_BOX,
    hit_box_offset=TOP_WALL_OFFSET,
    reflecting=jnp.array(1),
)
BOTTOM_WALL_SCENE_OBJECT = SceneObject(
    hit_box_matrix=BOTTOM_WALL_BOUNDING_BOX,
    hit_box_offset=BOTTOM_WALL_OFFSET,
    reflecting=jnp.array(1),
)
LEFT_WALL_SCENE_OBJECT = SceneObject(
    hit_box_matrix=LEFT_WALL_BOUNDING_BOX,
    hit_box_offset=LEFT_WALL_OFFSET,
    reflecting=jnp.array(1),
)
RIGHT_WALL_SCENE_OBJECT = SceneObject(
    hit_box_matrix=RIGHT_WALL_BOUNDING_BOX,
    hit_box_offset=RIGHT_WALL_OFFSET,
    reflecting=jnp.array(1),
)

SCENE_OBJECT_LIST = [
    TOP_WALL_SCENE_OBJECT,
    BOTTOM_WALL_SCENE_OBJECT,
    LEFT_WALL_SCENE_OBJECT,
    RIGHT_WALL_SCENE_OBJECT,
]

# comment out the scene objects stacking for now to check if the scene objects themselves work
SCENE_OBJECTS_STACKED = None
# SCENE_OBJECTS_STACKED = SceneObject(
#     hit_box_matrix=jnp.stack([obj.hit_box_matrix for obj in SCENE_OBJECT_LIST]),
#     hit_box_offset=jnp.stack([obj.hit_box_offset for obj in SCENE_OBJECT_LIST]),
#     reflecting=jnp.stack([obj.reflecting for obj in SCENE_OBJECT_LIST]),
# )

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
    if keys[pygame.K_a]:
        return jnp.array(Action.LEFT)
    elif keys[pygame.K_d]:
        return jnp.array(Action.RIGHT)
    elif keys[pygame.K_SPACE]:
        return jnp.array(Action.FIRE)
    elif keys[pygame.K_w]:
        return jnp.array(Action.UP)
    elif keys[pygame.K_s]:
        return jnp.array(Action.DOWN)
    else:
        return jnp.array(Action.NOOP)


# immutable state container
class VideoPinballState(NamedTuple):
    ball_x: chex.Array
    ball_y: chex.Array
    ball_vel_x: chex.Array
    ball_vel_y: chex.Array
    ball_direction: (
        chex.Array
    )  # 0: left/up, 1:left/down , 2: right/up, 3: right/down (Shouldn't this be a function?)
    left_flipper_angle: chex.Array
    right_flipper_angle: chex.Array
    plunger_position: (
        chex.Array
    )  # Value between 0 and 20 where 20 means that the plunger is fully pulled
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
def plunger_step(state: VideoPinballState, action: chex.Array) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """
    Update the plunger position based on the current state and action.
    """
    # Only move plunger if ball is not in play
    if_not_in_play = jnp.logical_not(state.ball_in_play)
    
    # if ball is not in play and DOWN was clicked, move plunger down
    plunger_position = jax.lax.cond(
        jnp.logical_and(
            state.plunger_position < PLUNGER_MAX_POSITION,
            jnp.logical_and(action == Action.DOWN, if_not_in_play),
        ),
        lambda s: s + 1,
        lambda s: s,
        operand=state.plunger_position,
    )

    # same for UP
    plunger_position = jax.lax.cond(
        jnp.logical_and(
            state.plunger_position > 0,
            jnp.logical_and(action == Action.UP, if_not_in_play),
        ),
        lambda s: s - 1,
        lambda s: s,
        operand=plunger_position,
    )

    # check if FIRE is pressed and we're ready to launch
    fire_pressed = action == Action.FIRE
    can_launch = jnp.logical_and(plunger_position > 0, if_not_in_play)
    should_launch = jnp.logical_and(fire_pressed, can_launch)
    
    # Calculate plunger power based on position when firing
    plunger_power = jax.lax.cond(
        should_launch,
        lambda s: s * 0.5,  # Convert position to power
        lambda s: jnp.array(0.0),
        operand=jnp.array(plunger_position, dtype=jnp.float32),
    )

    # Set ball in play if we're launching
    ball_in_play = jax.lax.cond(
        should_launch,
        lambda _: jnp.array(True),
        lambda _: state.ball_in_play,
        operand=None,
    )

    # Reset plunger position if fired
    plunger_position = jax.lax.cond(
        should_launch,
        lambda _: jnp.array(0),
        lambda _: plunger_position,
        operand=None,
    )

    return plunger_position, plunger_power, ball_in_play


@jax.jit
def flipper_step(state: VideoPinballState, action: chex.Array):

    left_flipper_angle = jax.lax.cond(
        jnp.logical_and(
            action == Action.LEFT, state.left_flipper_angle < FLIPPER_MAX_ANGLE
        ),
        lambda a: a + 1,
        lambda a: a,
        operand=state.left_flipper_angle,
    )

    right_flipper_angle = jax.lax.cond(
        jnp.logical_and(
            action == Action.RIGHT, state.right_flipper_angle < FLIPPER_MAX_ANGLE
        ),
        lambda a: a + 1,
        lambda a: a,
        operand=state.right_flipper_angle,
    )

    left_flipper_angle = jax.lax.cond(
        jnp.logical_and(
            jnp.logical_not(action == Action.LEFT), state.left_flipper_angle > 0
        ),
        lambda a: a - 1,
        lambda a: a,
        operand=left_flipper_angle,
    )

    right_flipper_angle = jax.lax.cond(
        jnp.logical_and(
            jnp.logical_not(action == Action.RIGHT), state.right_flipper_angle > 0
        ),
        lambda a: a - 1,
        lambda a: a,
        operand=right_flipper_angle,
    )

    # TODO update angles based on step phase?

    # TODO ball acceleration should be computed from current and new plunger/flipper states or some other way
    # _ball_property_ = ...

    return left_flipper_angle, right_flipper_angle


@jax.jit
def check_flipper_collision(
    state: VideoPinballState,
    ball_x: chex.Array,
    ball_y: chex.Array,
    ball_vel_x: chex.Array,
    ball_vel_y: chex.Array
) -> Tuple[chex.Array, chex.Array]:
    """
    check if ball is colliding with either flipper and calculate new velocity.
    Returns updated velocity values.
    """
    # Define flipper collision zones based on flipper position and angle
    left_flipper_x_min = 60
    left_flipper_x_max = 78
    left_flipper_y_base = 184 - FLIPPER_ANIMATION_Y_OFFSETS[state.left_flipper_angle]
    left_flipper_y_min = left_flipper_y_base - 6
    left_flipper_y_max = left_flipper_y_base + 4
    
    right_flipper_x_min = 79
    right_flipper_x_max = 97 + FLIPPER_ANIMATION_X_OFFSETS[state.right_flipper_angle]
    right_flipper_y_base = 184 - FLIPPER_ANIMATION_Y_OFFSETS[state.right_flipper_angle]
    right_flipper_y_min = right_flipper_y_base - 6
    right_flipper_y_max = right_flipper_y_base + 4
    
    # check if ball is in collision region of left flipper
    left_flipper_hit = jnp.logical_and(
        jnp.logical_and(ball_x >= left_flipper_x_min, ball_x <= left_flipper_x_max),
        jnp.logical_and(ball_y >= left_flipper_y_min, ball_y <= left_flipper_y_max)
    )
    
    # check if ball is in collision region of right flipper
    right_flipper_hit = jnp.logical_and(
        jnp.logical_and(ball_x >= right_flipper_x_min, ball_x <= right_flipper_x_max),
        jnp.logical_and(ball_y >= right_flipper_y_min, ball_y <= right_flipper_y_max)
    )
    
    # calculate flipper angle influence on ball direction
    # higher angle = more vertical boost, also consider which side of flipper was hit
    left_flipper_bounce_x = jnp.array([-2.0, -1.0, 0.0, 1.0])
    left_flipper_bounce_y = jnp.array([-1.0, -2.0, -3.0, -4.0])
    right_flipper_bounce_x = jnp.array([2.0, 1.0, 0.0, -1.0])
    right_flipper_bounce_y = jnp.array([-1.0, -2.0, -3.0, -4.0])
    
    # Select the appropriate bounce values based on flipper angle
    left_bounce_x = left_flipper_bounce_x[state.left_flipper_angle]
    left_bounce_y = left_flipper_bounce_y[state.left_flipper_angle]
    right_bounce_x = right_flipper_bounce_x[state.right_flipper_angle]
    right_bounce_y = right_flipper_bounce_y[state.right_flipper_angle]
    
    # apply velocity changes based on flipper hit
    # if flipper is moving (angle > 0), apply full force, otherwise apply reduced bounce
    left_flipper_moving = state.left_flipper_angle > 0
    right_flipper_moving = state.right_flipper_angle > 0
    
    new_vel_x = jnp.where(
        jnp.logical_and(left_flipper_hit, left_flipper_moving),
        left_bounce_x,  
        jnp.where(
            jnp.logical_and(right_flipper_hit, right_flipper_moving),
            right_bounce_x,  
            jnp.where(
                left_flipper_hit,
                -ball_vel_x * 0.8,  
                jnp.where(
                    right_flipper_hit,
                    -ball_vel_x * 0.8,  
                    ball_vel_x
                )
            )
        )
    )
    
    new_vel_y = jnp.where(
        jnp.logical_and(left_flipper_hit, left_flipper_moving),
        left_bounce_y,  
        
        # Full force when flipper is moving
        jnp.where(
            jnp.logical_and(right_flipper_hit, right_flipper_moving),
            right_bounce_y,  
            
            # Full force when flipper is moving
            jnp.where(
                jnp.logical_or(left_flipper_hit, right_flipper_hit),
                
                # Simple bounce when hitting stationary flipper
                -ball_vel_y * 0.8,  
                ball_vel_y
            )
        )
    )
    
    # Return updated velocities
    return new_vel_x, new_vel_y


@jax.jit
def check_bumper_collision(
    state: VideoPinballState, 
    ball_x: chex.Array, 
    ball_y: chex.Array,
    ball_vel_x: chex.Array,
    ball_vel_y: chex.Array
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """
    check if ball is colliding with any bumper and calculate new velocity.
    Returns updated velocity values and whether a bumper was hit.
    """
    # Define bumper positions (coordinates are approximate based on the sprites)
    bumper_positions = [
        (46, 122, 10),  # Left bumper (x, y, radius)
        (78, 58, 10),   # Middle bumper
        (110, 122, 10)  # Right bumper
    ]
    
    # check collision with each bumper
    bumper_hit = jnp.array(False)
    new_vx = ball_vel_x
    new_vy = ball_vel_y
    
    for i in range(3):
        # Extract bumper position
        bx, by, radius = bumper_positions[i]
        
        # Calculate distance from ball to bumper center
        dx = ball_x - bx
        dy = ball_y - by
        distance_squared = dx*dx + dy*dy
        
        # If within radius, collision occurred
        collision = distance_squared < (radius * radius)
        
        # Calculate bounce direction (away from bumper center)
        bounce_angle = jnp.arctan2(dy, dx)
        bounce_speed = 4.0  # Strong bounce from bumpers
        
        # Compute new velocity components if collision occurred
        collision_vx = bounce_speed * jnp.cos(bounce_angle)
        collision_vy = bounce_speed * jnp.sin(bounce_angle)
        
        # Update velocities if collision happened
        new_vx = jnp.where(
            collision,
            collision_vx,
            new_vx
        )
        
        new_vy = jnp.where(
            collision,
            collision_vy,
            new_vy
        )
        
        # Update bumper hit flag
        bumper_hit = jnp.logical_or(bumper_hit, collision)
    
    return new_vx, new_vy, bumper_hit


@jax.jit
def check_spinner_collision(
    state: VideoPinballState,
    ball_x: chex.Array,
    ball_y: chex.Array,
    ball_vel_x: chex.Array,
    ball_vel_y: chex.Array
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """
    check if ball is colliding with spinners.
    Returns updated velocity and whether a spinner was hit.
    """
    # Define spinner areas
    left_spinner_hit = jnp.logical_and(
        jnp.logical_and(ball_x > 25, ball_x < 35),
        jnp.logical_and(ball_y > 85, ball_y < 95)
    )
    
    right_spinner_hit = jnp.logical_and(
        jnp.logical_and(ball_x > 121, ball_x < 135),
        jnp.logical_and(ball_y > 85, ball_y < 95)
    )
    
    spinner_hit = jnp.logical_or(left_spinner_hit, right_spinner_hit)
    
    new_vx = jnp.where(
        spinner_hit,
        ball_vel_x * 1.2,
        ball_vel_x
    )
    
    new_vy = jnp.where(
        spinner_hit, 
        ball_vel_y * 1.2,
        ball_vel_y
    )
    
    return new_vx, new_vy, spinner_hit


@jax.jit
def ball_step(
    state: VideoPinballState,
    plunger_power,
    ball_in_play,
    action,
):
    """
    Update ball position and velocity based on current state, plunger power, and physics.
    Returns new ball position, direction, and velocity.
    """
    # check if we need to launch the ball (plunger power > 0 and ball not in play)
    launch_ball = jnp.logical_and(plunger_power > 0, jnp.logical_not(state.ball_in_play))
    
    # Launch ball logic
    def launch():
        # launch with velocity proportional to plunger power
        random_x_component = jnp.array([-0.5, 0.5], dtype=jnp.float32)[state.step_counter % 2]
        vx = random_x_component  
        vy = -plunger_power - jnp.array(2.0, dtype=jnp.float32)  
        x = BALL_START_X.astype(jnp.float32)
        y = BALL_START_Y.astype(jnp.float32)
        return x, y, vx, vy, jnp.array(True, dtype=jnp.bool_)
    
    # Ball physics when already in play
    def update_physics():
        # apply gravity to current velocity
        new_vy = state.ball_vel_y + GRAVITY
        
        y_to_x_influence = jnp.sign(state.ball_vel_y) * 0.02
        new_vx = state.ball_vel_x + y_to_x_influence
        
        # update position with velocity
        new_x = state.ball_x + new_vx
        new_y = state.ball_y + new_vy
        
        # define wall boundaries with explicit values matching the visual boundaries
        left_wall_x = 10.0
        right_wall_x = 150.0
        top_wall_y = 20.0
        bottom_wall_y = 183.0 
        
        # detect collisions more precisely
        hit_left = new_x <= jnp.array(left_wall_x, dtype=jnp.float32)
        hit_right = new_x >= jnp.array(right_wall_x, dtype=jnp.float32)
        hit_top = new_y <= jnp.array(top_wall_y, dtype=jnp.float32)
        hit_bottom = new_y >= jnp.array(bottom_wall_y, dtype=jnp.float32)

        #  detect whether the ball is in the flipper drain
        drain_gap_left  = 70.0   
        drain_gap_right = 90.0
        hits_bottom_row = new_y >= bottom_wall_y
        in_gap_x  = jnp.logical_and(new_x >= drain_gap_left,
                                    new_x <= drain_gap_right)
        in_drain  = jnp.logical_and(hits_bottom_row, in_gap_x)
        
        # calc new velocities after boundary collisions with better bounce physics
        energy_preservation = jnp.array(0.98, dtype=jnp.float32)
        
        new_vx = jnp.where(
            jnp.logical_or(hit_left, hit_right),
            -new_vx * energy_preservation,
            new_vx
        )
        
        # bottom collision bounce to make it more responsive, needs to be testsed
        bottom_bounce_energy = jnp.array(0.98, dtype=jnp.float32)  # tncreased bounce energy, needs to be experimented with, im not sure
        top_bounce_energy = jnp.array(0.92, dtype=jnp.float32)
        
        new_vy = jnp.where(
            in_drain,
            new_vy, 
            jnp.where(
                hit_bottom,
                -jnp.abs(new_vy) * bottom_bounce_energy,
                jnp.where(
                    hit_top,
                    jnp.abs(new_vy) * top_bounce_energy,
                    new_vy
                )
            )
        )
        
        # correct positions after collisions to prevent getting stuck in walls
        corrected_x = jnp.where(
            hit_left, 
            jnp.array(left_wall_x + 1.5, dtype=jnp.float32),  
            # Push further away from wall, might need to tweak this
            jnp.where(
                hit_right,
                
                # Push further away from wall, might need to tweak this
                jnp.array(right_wall_x - 1.5, dtype=jnp.float32),  
                new_x
            )
        )
        
        corrected_y = jnp.where(
            hit_top,
            jnp.array(top_wall_y + 1.5, dtype=jnp.float32),
            jnp.where(
                jnp.logical_and(hit_bottom, jnp.logical_not(in_drain)),
                jnp.array(bottom_wall_y - 2.0, dtype=jnp.float32),
                new_y
            )
        )
        
        # Apply velocity limits
        clamped_vx = jnp.clip(new_vx, -BALL_MAX_SPEED, BALL_MAX_SPEED)
        clamped_vy = jnp.clip(new_vy, -BALL_MAX_SPEED, BALL_MAX_SPEED)
        
        # check for flipper collisions
        clamped_vx, clamped_vy = check_flipper_collision(
            state, corrected_x, corrected_y, clamped_vx, clamped_vy
        )
        
        # check for bumper collisions
        bumper_vx, bumper_vy, bumper_hit = check_bumper_collision(
            state, corrected_x, corrected_y, clamped_vx, clamped_vy
        )
        clamped_vx = jnp.where(bumper_hit, bumper_vx, clamped_vx)
        clamped_vy = jnp.where(bumper_hit, bumper_vy, clamped_vy)
        
        # check for spinner collisions
        spinner_vx, spinner_vy, spinner_hit = check_spinner_collision(
            state, corrected_x, corrected_y, clamped_vx, clamped_vy
        )
        clamped_vx = jnp.where(spinner_hit, spinner_vx, clamped_vx)
        clamped_vy = jnp.where(spinner_hit, spinner_vy, clamped_vy)

        center_target_hit = jnp.logical_and(
            jnp.logical_and(corrected_x > 76-5, corrected_x < 76+10),
            jnp.logical_and(corrected_y > 120-5, corrected_y < 120+10)
        )
        
        # bounce slightly differently from center target testing purpose, need to confirm this later
        target_vx = -clamped_vx * 1.2
        target_vy = -clamped_vy * 1.2
        
        clamped_vx = jnp.where(center_target_hit, target_vx, clamped_vx)
        clamped_vy = jnp.where(center_target_hit, target_vy, clamped_vy)

        left_gutter = corrected_x < jnp.array(60.0, dtype=jnp.float32)
        right_gutter = corrected_x > jnp.array(100.0, dtype=jnp.float32)
        
        gutter_zone = jnp.logical_or(left_gutter, right_gutter)
        below_flippers = corrected_y >= jnp.array(188.0, dtype=jnp.float32)
        in_gutter = in_drain
        
        # maintain ball state unless it's in gutter
        return jax.lax.cond(
            in_gutter,
            lambda _: (
                BALL_START_X.astype(jnp.float32), 
                BALL_START_Y.astype(jnp.float32), 
                jnp.array(0.0, dtype=jnp.float32),  
                jnp.array(0.0, dtype=jnp.float32), 
                jnp.array(False, dtype=jnp.bool_)   
            ),
            lambda _: (corrected_x, corrected_y, clamped_vx, clamped_vy, state.ball_in_play),
            operand=None
        )
        
    # ball at rest at plunger position
    def at_rest():
        return (
            BALL_START_X.astype(jnp.float32),
            BALL_START_Y.astype(jnp.float32),
            jnp.array(0.0, dtype=jnp.float32),  # zero velocity, also, will have to find a way to test this
            jnp.array(0.0, dtype=jnp.float32),
            jnp.array(False, dtype=jnp.bool_)   # explicitly set to False
        )
    
    # use conditional logic to determine which state applies
    x, y, vx, vy, ball_in_play_new = jax.lax.cond(
        launch_ball,
        lambda _: launch(),
        lambda _: jax.lax.cond(
            state.ball_in_play,
            lambda _: update_physics(),
            lambda _: at_rest(),
            operand=None
        ),
        operand=None
    )
    
    # Calculate ball direction based on velocity
    direction_right = vx > 0
    direction_down = vy > 0
    
    ball_direction = jnp.where(
        direction_right,
        # right/down: 3, right/up: 2
        jnp.where(direction_down, 3, 2),  
        
        # left/down: 1, left/up: 0
        jnp.where(direction_down, 1, 0)   
    )
    
    return (
        x,
        y,
        ball_direction.astype(jnp.int32),
        vx,
        vy,
        ball_in_play_new,      
    )


@jax.jit
def _calc_hit_point(
    ball_movement: BallMovement,
    scene_object: SceneObject,
) -> chex.Array:
    """
    Calculate the hit point of the ball with the bounding box.
    Uses the slab method also known as ray AABB collision.

    Returns:
        hit_point: jnp.ndarray, the time and hit point of the ball with the bounding box.
        hit_point[0]: jnp.ndarray, the time of entry
        hit_point[1]: jnp.ndarray, the x position of the hit point
        hit_point[2]: jnp.ndarray, the y position of the hit point
    """

    # Calculate trajectory of the ball in x and y direction
    trajectory_x = ball_movement.new_ball_x - ball_movement.old_ball_x
    trajectory_y = ball_movement.new_ball_y - ball_movement.old_ball_y

    # Force non-zero trajectory values to avoid division by zero
    trajectory_x = jax.lax.cond(
        trajectory_x == 0, lambda x: jnp.array(1e-8), lambda x: x, operand=trajectory_x
    )
    trajectory_y = jax.lax.cond(
        trajectory_y == 0, lambda x: jnp.array(1e-8), lambda x: x, operand=trajectory_y
    )

    tx1 = (scene_object.hit_box_offset[0] - ball_movement.old_ball_x) / trajectory_x
    tx2 = (
        scene_object.hit_box_offset[0]
        + scene_object.hit_box_matrix.shape[-2]
        - ball_movement.old_ball_x
    ) / trajectory_x
    ty1 = (scene_object.hit_box_offset[1] - ball_movement.old_ball_y) / trajectory_y
    ty2 = (
        scene_object.hit_box_offset[1]
        + scene_object.hit_box_matrix.shape[-1]
        - ball_movement.old_ball_y
    ) / trajectory_y

    # Calculate the time of intersection with the bounding box
    tmin_x = jnp.minimum(tx1, tx2)
    tmax_x = jnp.maximum(tx1, tx2)
    tmin_y = jnp.minimum(ty1, ty2)
    tmax_y = jnp.maximum(ty1, ty2)

    t_entry = jnp.maximum(tmin_x, tmin_y)
    t_exit = jnp.minimum(tmax_x, tmax_y)

    no_collision = jnp.logical_or(t_entry > t_exit, t_entry > 1)
    no_collision = jnp.logical_or(no_collision, t_entry < 0)

    hit_point = jnp.array(
        [
            t_entry,
            ball_movement.old_ball_x + t_entry * trajectory_x,
            ball_movement.old_ball_y + t_entry * trajectory_y,
        ]
    )

    return jax.lax.cond(
        no_collision, lambda _: jnp.array([9999, -1, -1]), lambda _: hit_point
    )


@jax.jit
def _check_all_obstacle_hits(
    old_ball_x: chex.Array,
    old_ball_y: chex.Array,
    new_ball_x: chex.Array,
    new_ball_y: chex.Array,
    ball_direction: chex.Array,
) -> bool:
    """
    check if the ball is hitting an obstacle.
    """
    # Temporarily disable collision checking until the stacked scene objects are fixed
    return False


@jax.jit
def _reset_ball(state: VideoPinballState):
    """
    When the ball goes into the gutter or into the plunger hole,
    respawn the ball on the launcher.
    """
    return (
        BALL_START_X.astype(jnp.float32),
        BALL_START_Y.astype(jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
    )


def _calculate_score(state, ball_x, ball_y, ball_vel_x, ball_vel_y):
    """
    calculate score based on game events
    """
    # Lower minimum velocity required to score to make scoring easier
    min_velocity = 0.02  
    total_velocity = jnp.sqrt(ball_vel_x**2 + ball_vel_y**2)
    is_moving = total_velocity > min_velocity
    
    left_wall_hit = ball_x <= 15
    right_wall_hit = ball_x >= 145
    top_wall_hit = ball_y <= 25
    bottom_wall_hit = ball_y >= 175
    
    wall_hit = jnp.logical_or(
        jnp.logical_or(left_wall_hit, right_wall_hit),
        jnp.logical_or(top_wall_hit, bottom_wall_hit)
    )
    
    # always give points for wall hits when the ball is moving
    wall_points = jnp.where(wall_hit, WALL_BOUNCE_POINTS, 0)
    
    # check bumper collisions with larger hit areas for easier scoring
    left_bumper_hit = jnp.logical_and(
        jnp.logical_and(ball_x > 35, ball_x < 57),
        jnp.logical_and(ball_y > 110, ball_y < 132)
    )
    
    middle_bumper_hit = jnp.logical_and(
        jnp.logical_and(ball_x > 65, ball_x < 92),
        jnp.logical_and(ball_y > 45, ball_y < 72)
    )
    
    right_bumper_hit = jnp.logical_and(
        jnp.logical_and(ball_x > 95, ball_x < 125),
        jnp.logical_and(ball_y > 110, ball_y < 132)
    )
    
    bumper_hit = jnp.logical_or(
        jnp.logical_or(left_bumper_hit, middle_bumper_hit),
        right_bumper_hit
    )
    bumper_points = jnp.where(bumper_hit, BUMPER_HIT_POINTS, 0)
    
    # check spinners with wider detection zones
    left_spinner_hit = jnp.logical_and(
        jnp.logical_and(ball_x > 20, ball_x < 42),
        jnp.logical_and(ball_y > 80, ball_y < 102)
    )
    
    right_spinner_hit = jnp.logical_and(
        jnp.logical_and(ball_x > 115, ball_x < 142),
        jnp.logical_and(ball_y > 80, ball_y < 102)
    )
    
    spinner_hit = jnp.logical_or(left_spinner_hit, right_spinner_hit)
    spinner_points = jnp.where(jnp.logical_and(spinner_hit, is_moving), SPINNER_POINTS, 0)
    
    # Center diamond target with larger hit area
    center_target_hit = jnp.logical_and(
        jnp.logical_and(ball_x > 67, ball_x < 92),
        jnp.logical_and(ball_y > 110, ball_y < 132)
    )
    target_points = jnp.where(jnp.logical_and(center_target_hit, is_moving), TARGET_HIT_POINTS, 0)
    
    # Top diamonds
    top_diamonds_hit = jnp.logical_and(
        jnp.logical_and(ball_x > 50, ball_x < 105),
        jnp.logical_and(ball_y > 15, ball_y < 40)
    )
    top_points = jnp.where(jnp.logical_and(top_diamonds_hit, is_moving), BONUS_MULTIPLIER_POINTS, 0)
    
    # Flipper hit detection
    flipper_hit = jnp.logical_and(
        jnp.logical_and(ball_x > 55, ball_x < 105),
        jnp.logical_and(ball_y > 170, ball_y < 190)
    )
    flipper_moving = jnp.logical_or(state.left_flipper_angle > 0, state.right_flipper_angle > 0)
    flipper_points = jnp.where(
        jnp.logical_and(jnp.logical_and(flipper_hit, flipper_moving), is_moving),
        FLIPPER_HIT_POINTS, 
        0
    )
    
    # Calculate total score increment
    score_increment = (bumper_points + spinner_points + target_points + wall_points + flipper_points + top_points)
    
    # Add minimum score to always make progress
    minimum_score = jnp.where(is_moving, 1, 0)
    score_increment = jnp.where(score_increment == 0,
                                jnp.where(wall_hit | bumper_hit, 1, 0),
                                score_increment)
    
    # Apply bonus multiplier
    score_increment = score_increment * state.bonus_multiplier
    
    return score_increment


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
            Action.NOOP,
            Action.FIRE,
            Action.RIGHT,
            Action.LEFT,
            Action.UP,
            Action.DOWN,
        }
        self.obs_size = 3 * 4 + 1 + 1

    def reset(self, prng_key) -> Tuple[VideoPinballState, VideoPinballObservation]:
        """
        resets the game state to the initial state
        """
        state = VideoPinballState(
            ball_x=jnp.array(BALL_START_X).astype(jnp.float32),
            ball_y=jnp.array(BALL_START_Y).astype(jnp.float32),
            ball_vel_x=jnp.array(0.0).astype(jnp.float32),
            ball_vel_y=jnp.array(0.0).astype(jnp.float32),
            ball_direction=jnp.array(0).astype(jnp.int32),
            left_flipper_angle=jnp.array(0).astype(jnp.int32),
            right_flipper_angle=jnp.array(0).astype(jnp.int32),
            plunger_position=jnp.array(0).astype(jnp.int32),
            score=jnp.array(0).astype(jnp.int32),
            lives=jnp.array(1).astype(jnp.int32),
            bonus_multiplier=jnp.array(1).astype(jnp.int32),
            bumpers_active=jnp.array([1, 1, 1]).astype(jnp.int32),
            targets_hit=jnp.array([1, 1, 1]).astype(jnp.int32),
            step_counter=jnp.array(0).astype(jnp.int32),
            ball_in_play=jnp.array(False).astype(jnp.bool),
        )
        initial_obs = self._get_observation(state)
        return initial_obs, state  

    def step(
        self, state: VideoPinballState, action: chex.Array
    ) -> Tuple[
        VideoPinballObservation, VideoPinballState, float, bool, VideoPinballInfo
    ]:
        """
        Execute one step of the environment's dynamics.
        
        Args:
            state: Current state of the game
            action: Action to take
            
        Returns:
            observation: Next observation
            new_state: Next state
            reward: Reward from this step
            done: Whether the episode is done
            info: Additional information
        """
        # Update plunger position and get plunger power
        plunger_position, plunger_power, ball_in_play = plunger_step(state, action)
        
        # Update flipper positions
        left_flipper_angle, right_flipper_angle = flipper_step(state, action)
        
        # Update ball physics
        (ball_x, ball_y, ball_direction,
         ball_vel_x, ball_vel_y,
         ball_in_play) = ball_step(     
            state,
            plunger_power,
            ball_in_play,
            action,
        )
        
        # Calculate score increment based on game events
        score_increment = _calculate_score(state, ball_x, ball_y, ball_vel_x, ball_vel_y)
        score = state.score + score_increment

        # Update Objects on Hit
        bumpers_active = jnp.array([1, 1, 1]).astype(jnp.int32) 
        targets_hit = jnp.array([1, 1, 1]).astype(jnp.int32)
        bonus_multiplier = state.bonus_multiplier

        new_state = VideoPinballState(
            ball_x=ball_x, 
            ball_y=ball_y, 
            ball_vel_x=ball_vel_x, 
            ball_vel_y=ball_vel_y, 
            ball_direction=ball_direction,
            left_flipper_angle=left_flipper_angle,
            right_flipper_angle=right_flipper_angle,
            plunger_position=plunger_position,
            score=score,
            lives=state.lives,
            bonus_multiplier=bonus_multiplier,
            bumpers_active=bumpers_active,
            targets_hit=targets_hit,
            step_counter=state.step_counter + 1,
            ball_in_play=ball_in_play,
        )

        done = self._get_done(new_state)
        env_reward = self._get_env_reward(state, new_state)
        all_rewards = self._get_all_reward(state, new_state)
        info = self._get_info(new_state, all_rewards)

        observation = self._get_observation(new_state)
        
        return observation, new_state, env_reward, done, info

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
            ball=ball,
            left_flipper=left_flipper,
            right_flipper=right_flipper,
            plunger=plunger,
            bumpers=state.bumpers_active,  # bumper states array
            targets=state.targets_hit,  # target states array
            score=state.score,
            lives=state.lives,
            bonus_multiplier=state.bonus_multiplier,
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

    # def action_space(self) -> spaces.Discrete:
    #     return spaces.Discrete(len(self.action_set))

    def get_action_space(self):
        return jnp.array(list(self.action_set))

    def observation_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=None,
            dtype=jnp.uint8,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(
        self, state: VideoPinballState, all_rewards: chex.Array
    ) -> VideoPinballInfo:
        return VideoPinballInfo(time=state.step_counter, all_rewards=all_rewards)

    @partial(jax.jit, static_argnums=(0,))
    def _get_env_reward(
        self, previous_state: VideoPinballState, state: VideoPinballState
    ):
        return state.score - previous_state.score

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(
        self, previous_state: VideoPinballState, state: VideoPinballState
    ):
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
    SPRITES_BASE_DIR = os.path.join(
        MODULE_DIR, "sprites/videopinball"
    )  # Assuming sprites are in a 'sprites/videopinball' subdirectory

    # Load sprites
    sprite_background = aj.loadFrame(
        os.path.join(SPRITES_BASE_DIR, "Background.npy"), transpose=True
    )
    sprite_ball = aj.loadFrame(
        os.path.join(SPRITES_BASE_DIR, "Ball.npy"), transpose=True
    )

    sprite_atari_logo = aj.loadFrame(
        os.path.join(SPRITES_BASE_DIR, "AtariLogo.npy"), transpose=True
    )
    sprite_x = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "X.npy"), transpose=True)
    sprite_yellow_diamond_bottom = aj.loadFrame(
        os.path.join(SPRITES_BASE_DIR, "YellowDiamondBottom.npy"), transpose=True
    )
    sprite_yellow_diamond_top = aj.loadFrame(
        os.path.join(SPRITES_BASE_DIR, "YellowDiamondTop.npy"), transpose=True
    )

    # sprite_wall_bottom_left_square = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "WallBottomLeftSquare.npy"), transpose=True)
    # sprite_wall_bumper = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "WallBumper.npy"), transpose=True)
    # sprite_wall_dropper = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "WallDropper.npy"), transpose=True)
    # sprite_wall_left_l = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "WallLeftL.npy"), transpose=True)
    # sprite_wall_outer = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "WallOuter.npy"), transpose=True)
    # sprite_wall_right_l = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "WallRightL.npy"), transpose=True)
    # sprite_wall_small_horizontal = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "WallSmallHorizontal.npy"), transpose=True)
    sprite_walls = aj.loadFrame(
        os.path.join(SPRITES_BASE_DIR, "Walls.npy"), transpose=True
    )

    # Animated sprites
    sprite_spinner0 = aj.loadFrame(
        os.path.join(SPRITES_BASE_DIR, "SpinnerBottom.npy"), transpose=True
    )
    sprite_spinner1 = aj.loadFrame(
        os.path.join(SPRITES_BASE_DIR, "SpinnerRight.npy"), transpose=True
    )
    sprite_spinner2 = aj.loadFrame(
        os.path.join(SPRITES_BASE_DIR, "SpinnerTop.npy"), transpose=True
    )
    sprite_spinner3 = aj.loadFrame(
        os.path.join(SPRITES_BASE_DIR, "SpinnerLeft.npy"), transpose=True
    )

    sprite_launcher0 = aj.loadFrame(
        os.path.join(SPRITES_BASE_DIR, "Launcher0.npy"), transpose=True
    )
    sprite_launcher1 = aj.loadFrame(
        os.path.join(SPRITES_BASE_DIR, "Launcher1.npy"), transpose=True
    )
    sprite_launcher2 = aj.loadFrame(
        os.path.join(SPRITES_BASE_DIR, "Launcher2.npy"), transpose=True
    )
    sprite_launcher3 = aj.loadFrame(
        os.path.join(SPRITES_BASE_DIR, "Launcher3.npy"), transpose=True
    )
    sprite_launcher4 = aj.loadFrame(
        os.path.join(SPRITES_BASE_DIR, "Launcher4.npy"), transpose=True
    )
    sprite_launcher5 = aj.loadFrame(
        os.path.join(SPRITES_BASE_DIR, "Launcher4.npy"), transpose=True
    )
    sprite_launcher6 = aj.loadFrame(
        os.path.join(SPRITES_BASE_DIR, "Launcher6.npy"), transpose=True
    )
    sprite_launcher7 = aj.loadFrame(
        os.path.join(SPRITES_BASE_DIR, "Launcher7.npy"), transpose=True
    )
    sprite_launcher8 = aj.loadFrame(
        os.path.join(SPRITES_BASE_DIR, "Launcher8.npy"), transpose=True
    )
    sprite_launcher9 = aj.loadFrame(
        os.path.join(SPRITES_BASE_DIR, "Launcher9.npy"), transpose=True
    )
    sprite_launcher10 = aj.loadFrame(
        os.path.join(SPRITES_BASE_DIR, "Launcher10.npy"), transpose=True
    )
    sprite_launcher11 = aj.loadFrame(
        os.path.join(SPRITES_BASE_DIR, "Launcher11.npy"), transpose=True
    )
    sprite_launcher12 = aj.loadFrame(
        os.path.join(SPRITES_BASE_DIR, "Launcher12.npy"), transpose=True
    )
    sprite_launcher13 = aj.loadFrame(
        os.path.join(SPRITES_BASE_DIR, "Launcher13.npy"), transpose=True
    )
    sprite_launcher14 = aj.loadFrame(
        os.path.join(SPRITES_BASE_DIR, "Launcher14.npy"), transpose=True
    )
    sprite_launcher15 = aj.loadFrame(
        os.path.join(SPRITES_BASE_DIR, "Launcher15.npy"), transpose=True
    )
    sprite_launcher16 = aj.loadFrame(
        os.path.join(SPRITES_BASE_DIR, "Launcher16.npy"), transpose=True
    )
    sprite_launcher17 = aj.loadFrame(
        os.path.join(SPRITES_BASE_DIR, "Launcher17.npy"), transpose=True
    )
    sprite_launcher18 = aj.loadFrame(
        os.path.join(SPRITES_BASE_DIR, "Launcher18.npy"), transpose=True
    )

    sprite_flipper_left0 = aj.loadFrame(
        os.path.join(SPRITES_BASE_DIR, "FlipperLeft0.npy"), transpose=True
    )
    sprite_flipper_left1 = aj.loadFrame(
        os.path.join(SPRITES_BASE_DIR, "FlipperLeft1.npy"), transpose=True
    )
    sprite_flipper_left2 = aj.loadFrame(
        os.path.join(SPRITES_BASE_DIR, "FlipperLeft2.npy"), transpose=True
    )
    sprite_flipper_left3 = aj.loadFrame(
        os.path.join(SPRITES_BASE_DIR, "FlipperLeft3.npy"), transpose=True
    )
    sprite_flipper_right0 = aj.loadFrame(
        os.path.join(SPRITES_BASE_DIR, "FlipperRight0.npy"), transpose=True
    )
    sprite_flipper_right1 = aj.loadFrame(
        os.path.join(SPRITES_BASE_DIR, "FlipperRight1.npy"), transpose=True
    )
    sprite_flipper_right2 = aj.loadFrame(
        os.path.join(SPRITES_BASE_DIR, "FlipperRight2.npy"), transpose=True
    )
    sprite_flipper_right3 = aj.loadFrame(
        os.path.join(SPRITES_BASE_DIR, "FlipperRight3.npy"), transpose=True
    )

    sprites_spinner = aj.pad_to_match(
        [sprite_spinner0, sprite_spinner1, sprite_spinner2, sprite_spinner3]
    )
    sprites_spinner = jnp.concatenate(
        [
            jnp.repeat(sprites_spinner[0][None], 2, axis=0),
            jnp.repeat(sprites_spinner[1][None], 2, axis=0),
            jnp.repeat(sprites_spinner[2][None], 2, axis=0),
            jnp.repeat(sprites_spinner[3][None], 2, axis=0),
        ]
    )

    sprites_plunger = aj.pad_to_match_top(
        [
            sprite_launcher0,
            sprite_launcher1,
            sprite_launcher2,
            sprite_launcher3,
            sprite_launcher4,
            sprite_launcher5,
            sprite_launcher6,
            sprite_launcher7,
            sprite_launcher8,
            sprite_launcher9,
            sprite_launcher10,
            sprite_launcher11,
            sprite_launcher12,
            sprite_launcher13,
            sprite_launcher14,
            sprite_launcher15,
            sprite_launcher16,
            sprite_launcher17,
            sprite_launcher18,
        ]
    )

    sprites_plunger = jnp.concatenate(
        [
            jnp.repeat(sprites_plunger[0][None], 3, axis=0),
            jnp.repeat(sprites_plunger[1][None], 2, axis=0),
            jnp.repeat(sprites_plunger[2][None], 1, axis=0),
            jnp.repeat(sprites_plunger[3][None], 1, axis=0),
            jnp.repeat(sprites_plunger[4][None], 1, axis=0),
            jnp.repeat(sprites_plunger[5][None], 1, axis=0),
            jnp.repeat(sprites_plunger[6][None], 1, axis=0),
            jnp.repeat(sprites_plunger[7][None], 1, axis=0),
            jnp.repeat(sprites_plunger[8][None], 1, axis=0),
            jnp.repeat(sprites_plunger[9][None], 1, axis=0),
            jnp.repeat(sprites_plunger[10][None], 1, axis=0),
            jnp.repeat(sprites_plunger[11][None], 1, axis=0),
            jnp.repeat(sprites_plunger[12][None], 1, axis=0),
            jnp.repeat(sprites_plunger[13][None], 1, axis=0),
            jnp.repeat(sprites_plunger[14][None], 1, axis=0),
            jnp.repeat(sprites_plunger[15][None], 1, axis=0),
            jnp.repeat(sprites_plunger[16][None], 1, axis=0),
            jnp.repeat(sprites_plunger[17][None], 1, axis=0),
            jnp.repeat(sprites_plunger[18][None], 1, axis=0),
        ]
    )

    sprites_flipper_left = aj.pad_to_match(
        [
            sprite_flipper_left0,
            sprite_flipper_left1,
            sprite_flipper_left2,
            sprite_flipper_left3,
        ]
    )

    # sprites_flipper_left = jnp.concatenate([
    #     jnp.repeat(sprites_flipper_left[0][None], 2, axis=0),
    #     jnp.repeat(sprites_flipper_left[1][None], 2, axis=0),
    #     jnp.repeat(sprites_flipper_left[2][None], 2, axis=0),
    #     jnp.repeat(sprites_flipper_left[3][None], 2, axis=0)
    # ])

    sprites_flipper_right = aj.pad_to_match(
        [
            sprite_flipper_right0,
            sprite_flipper_right1,
            sprite_flipper_right2,
            sprite_flipper_right3,
        ]
    )

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

    sprite_background = jnp.expand_dims(sprite_background, axis=0)
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
        frame_flipper_left = aj.get_sprite_frame(
            self.sprites["flipper_left"], state.left_flipper_angle
        )
        raster = aj.render_at(raster, 64, 184 - FLIPPER_ANIMATION_Y_OFFSETS[state.left_flipper_angle], frame_flipper_left)

        frame_flipper_right = aj.get_sprite_frame(
            self.sprites["flipper_right"], state.right_flipper_angle
        )
        raster = aj.render_at(raster, 83 + FLIPPER_ANIMATION_X_OFFSETS[state.right_flipper_angle], 184 - FLIPPER_ANIMATION_Y_OFFSETS[state.right_flipper_angle], frame_flipper_right)

        frame_plunger = aj.get_sprite_frame(
            self.sprites["plunger"], state.plunger_position
        )  # Still slightly inaccurate
        raster = aj.render_at(raster, 148, 133, frame_plunger)

        frame_spinner = aj.get_sprite_frame(
            self.sprites["spinner"], state.step_counter % 8
        )
        raster = aj.render_at(raster, 30, 90, frame_spinner)
        raster = aj.render_at(raster, 126, 90, frame_spinner)

        frame_ball = aj.get_sprite_frame(self.sprites["ball"], 0)
        raster = aj.render_at(raster, state.ball_x, state.ball_y, frame_ball)

        # render score digits dynamically based on the current score
        score_value = state.score.astype(int)
        score_str = str(score_value).zfill(6) 
        
        for i, digit in enumerate(score_str):
            try:
                digit_value = int(digit)
                frame_score_digit = aj.get_sprite_frame(self.sprites["score_number_digits"], digit_value)
                raster = aj.render_at(raster, 64 + i * 16, 3, frame_score_digit)
            except ValueError:
                frame_score_digit = aj.get_sprite_frame(self.sprites["score_number_digits"], 0)
                raster = aj.render_at(raster, 64 + i * 16, 3, frame_score_digit)

        # Render special yellow field objects TODO: (unfinished, game_state implementation needed)
        frame_bumper_left = aj.get_sprite_frame(self.sprites["field_number_digits"], 1)
        raster = aj.render_at(raster, 46, 122, frame_bumper_left)
        frame_bumper_middle = aj.get_sprite_frame(
            self.sprites["field_number_digits"], 1
        )
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

        frame_special_diamond = aj.get_sprite_frame(
            self.sprites["yellow_diamond_bottom"], 0
        )
        raster = aj.render_at(raster, 76, 120, frame_special_diamond)

        return raster


if __name__ == "__main__":
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("VideoPinball Game")
    clock = pygame.time.Clock()
    seed = 42
    prng_key = jrandom.PRNGKey(seed)

    game = JaxVideoPinball(frameskip=1)

    # Create the JAX renderer
    renderer = Renderer_AtraJaxisVideoPinball()

    # Get jitted functions
    jitted_step = game.step
    jitted_reset = game.reset

    obs, curr_state = jitted_reset(prng_key=prng_key)

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
                        action = int(get_human_action())  # Convert from JAX array to Python int
                        obs, curr_state, reward, done, info = jitted_step(
                            curr_state, action
                        )

        if not frame_by_frame:
            # If not in frame-by-frame mode perform step at each clock-tick
            # i.e. get human (game) action
            if counter % frameskip == 0:
                action = int(get_human_action())  # Convert from JAX array to Python int
                # Update game step
                obs, curr_state, reward, done, info = jitted_step(curr_state, action)

        # Render and display
        raster = renderer.render(curr_state)

        aj.update_pygame(screen, raster, 3, WIDTH, HEIGHT)

        counter += 1
        clock.tick(60)

    pygame.quit()