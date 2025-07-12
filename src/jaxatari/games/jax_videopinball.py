"""
Project: JAXAtari VideoPinball
Description: Our team's JAX implementation of Video Pinball.

Authors:
    - Michael Olenberger <michael.olenberger@stud.tu-darmstadt.de>
    - Maximilian Roth <maximilian.roth@stud.tu-darmstadt.de>
    - Jonas Neumann <jonas.neumann@stud.tu-darmstadt.de>
    - Yuddhish Chooah <yuddhish.chooah@stud.tu-darmstadt.de>

We recorded a sequence of game play and have added the interpretations of the RAM values as
a CSV file with the RAM registers and their meaning and possible values.

Implemented features:
- Working Game state, reset() and step() functions
- Plunger and Flipper movement logic
- Plunger physics
- Accurate and jit-compatible rendering for implemented mechanics
- Ball respawning upon failure and life counter
- Wall collisions
- Rudimentary ball physics (IMPORTANT: Pull the plunger back all the way when testing)

Why the ball physics are not yet perfect:
Video Pinball has extremely complicated ball physics. Launch angles, when hitting (close to) corners are
seemingly random and velocity calculation has a variety of strange quirks like strong spontaneous acceleration
when a slow balls hit walls at certain angles, etc...
These properties are impossible to read from the RAM state and need to be investigated
frame by frame in various scenarios. Thus, the physics are far from perfect.
There is still a physics bug when calculating multiple wall collisions during a single step which unfortunately
gets triggered when the plunger is pulled all the way down. When testing, pulling it all the way down, this is the only
way we found so far to mitigate this issue until we match the physics to the Atari original.

Additional notes:
The renderer requires a custom function that was implemented in atraJaxis.py
If the game.py files are tested separately, this function needs to be included manually:

@jax.jit
def pad_to_match_top(sprites):
    max_height = max(sprite.shape[0] for sprite in sprites)
    max_width = max(sprite.shape[1] for sprite in sprites)

    def pad_sprite(sprite):
        pad_height = max_height - sprite.shape[0]
        pad_width = max_width - sprite.shape[1]
        return jnp.pad(
            sprite,
            ((pad_height, 0), (pad_width, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    return [pad_sprite(sprite) for sprite in sprites]
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

from jaxatari.renderers import AtraJaxisRenderer
from jaxatari.rendering import atraJaxis as aj
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action

# Constants for game environment
WIDTH = 160
HEIGHT = 210


# Physics constants
# TODO: check if these are correct
GRAVITY = 0.5  # 0.12
VELOCITY_DAMPENING_VALUE = 0  # 24
BALL_MAX_SPEED = 10.0
FLIPPER_MAX_ANGLE = 3
FLIPPER_ANIMATION_Y_OFFSETS = jnp.array(
    [0, 0, 3, 7]
)  # This is a little scuffed, it would be cleaner to just fix the sprites but this works fine
FLIPPER_ANIMATION_X_OFFSETS = jnp.array([0, 0, 0, 1])  # Only for the right flipper
PLUNGER_MAX_POSITION = 20

# Game logic constants
T_ENTRY_NO_COLLISION = 9999
VELOCITY_RETENTION_FACTOR = 1.0

# Game layout constants
# TODO: check if these are correct
BALL_SIZE = (4, 4)
FLIPPER_LEFT_POS = (30, 180)
FLIPPER_RIGHT_POS = (110, 180)
PLUNGER_POS = (150, 120)
PLUNGER_MAX_HEIGHT = 20  # Taken from RAM values (67-87)
INVISIBLE_BLOCK_REFLECTION_FACTOR = (
    2  # 8 times the plunger power is added to ball_vel_x
)


# Background color and object colors
BACKGROUND_COLOR = 0, 0, 0
BALL_COLOR = 255, 255, 255  # ball
FLIPPER_COLOR = 255, 0, 0  # flipper
TEXT_COLOR = 255, 255, 255  # text


# Pygame window dimensions
WINDOW_WIDTH = 160 * 3
WINDOW_HEIGHT = 210 * 3


# Positions/dimensions
BALL_START_X = jnp.array(149.0)
BALL_START_Y = jnp.array(129.0)
BALL_START_DIRECTION = jnp.array(0)

GAME_BOTTOM_Y = 191

TOP_WALL_LEFT_X_OFFSET = 0
TOP_WALL_TOP_Y_OFFSET = 16
RIGHT_WALL_LEFT_X_OFFSET = 152
BOTTOM_WALL_TOP_Y_OFFSET = 184

INVISIBLE_BLOCK_LEFT_X_OFFSET = 149
INVISIBLE_BLOCK_TOP_Y_OFFSET = 36

INNER_WALL_TOP_Y_OFFSET = 56
LEFT_INNER_WALL_TOP_X_OFFSET = 12
RIGHT_INNER_WALL_TOP_X_OFFSET = 144

QUADRUPLE_STEP_Y_OFFSET = 152
TRIPLE_STEP_Y_OFFSET = 160
DOUBLE_STEP_Y_OFFSET = 168
SINGLE_STEP_Y_OFFSET = 176

LEFT_QUADRUPLE_STEP_X_OFFSET = 16
RIGHT_QUADRUPLE_STEP_X_OFFSET = 140
LEFT_TRIPLE_STEP_X_OFFSET = 20
RIGHT_TRIPLE_STEP_X_OFFSET = 136
LEFT_DOUBLE_STEP_X_OFFSET = 24
RIGHT_DOUBLE_STEP_X_OFFSET = 132
LEFT_SINGLE_STEP_X_OFFSET = 28
RIGHT_SINGLE_STEP_X_OFFSET = 128

OUTER_WALL_THICKNESS = 8
INNER_WALL_THICKNESS = 4
WALL_CORNER_BLOCK_WIDTH = 4
WALL_CORNER_BLOCK_HEIGHT = 8
STEP_HEIGHT = 8
STEP_WIDTH = 4

TOP_WALL_BOUNDING_BOX = jnp.ones((OUTER_WALL_THICKNESS, 160)).astype(jnp.bool)
TOP_WALL_OFFSET = jnp.array([TOP_WALL_LEFT_X_OFFSET, TOP_WALL_TOP_Y_OFFSET])

BOTTOM_WALL_BOUNDING_BOX = jnp.ones((OUTER_WALL_THICKNESS, 160)).astype(jnp.bool)
BOTTOM_WALL_OFFSET = jnp.array([TOP_WALL_LEFT_X_OFFSET, BOTTOM_WALL_TOP_Y_OFFSET])

LEFT_WALL_BOUNDING_BOX = jnp.ones((176, OUTER_WALL_THICKNESS)).astype(jnp.bool)
LEFT_WALL_OFFSET = jnp.array([TOP_WALL_LEFT_X_OFFSET, TOP_WALL_TOP_Y_OFFSET])

RIGHT_WALL_BOUNDING_BOX = jnp.ones((176, OUTER_WALL_THICKNESS)).astype(jnp.bool)
RIGHT_WALL_OFFSET = jnp.array([RIGHT_WALL_LEFT_X_OFFSET, TOP_WALL_TOP_Y_OFFSET])

LEFT_INNER_WALL_OFFSET = jnp.array(
    [LEFT_INNER_WALL_TOP_X_OFFSET, INNER_WALL_TOP_Y_OFFSET]
)
LEFT_INNER_WALL_BOUNDING_BOX = jnp.ones((135, INNER_WALL_THICKNESS)).astype(
    jnp.bool
)  # 135 = 191 - 56 (height of inner wall)

RIGHT_INNER_WALL_OFFSET = jnp.array(
    [RIGHT_INNER_WALL_TOP_X_OFFSET, INNER_WALL_TOP_Y_OFFSET]
)
RIGHT_INNER_WALL_BOUNDING_BOX = jnp.ones((135, INNER_WALL_THICKNESS)).astype(
    jnp.bool
)  # 135 = 191 - 56 (height of inner wall)

QUADRUPLE_STEP_BOUNDING_BOX = jnp.ones((4 * STEP_HEIGHT, STEP_WIDTH)).astype(
    jnp.bool
)  # (4*8) x 4
LEFT_QUADRUPLE_STEP_OFFSET = jnp.array(
    [LEFT_QUADRUPLE_STEP_X_OFFSET, QUADRUPLE_STEP_Y_OFFSET]
)
RIGHT_QUADRUPLE_STEP_OFFSET = jnp.array(
    [RIGHT_QUADRUPLE_STEP_X_OFFSET, QUADRUPLE_STEP_Y_OFFSET]
)

TRIPLE_STEP_BOUNDING_BOX = jnp.ones((3 * STEP_HEIGHT, STEP_WIDTH)).astype(
    jnp.bool
)  # (3*8) x 4
LEFT_TRIPLE_STEP_OFFSET = jnp.array([LEFT_TRIPLE_STEP_X_OFFSET, TRIPLE_STEP_Y_OFFSET])
RIGHT_TRIPLE_STEP_OFFSET = jnp.array([RIGHT_TRIPLE_STEP_X_OFFSET, TRIPLE_STEP_Y_OFFSET])

DOUBLE_STEP_BOUNDING_BOX = jnp.ones((2 * STEP_HEIGHT, STEP_WIDTH)).astype(
    jnp.bool
)  # (2*8) x 4
LEFT_DOUBLE_STEP_OFFSET = jnp.array([LEFT_DOUBLE_STEP_X_OFFSET, DOUBLE_STEP_Y_OFFSET])
RIGHT_DOUBLE_STEP_OFFSET = jnp.array([RIGHT_DOUBLE_STEP_X_OFFSET, DOUBLE_STEP_Y_OFFSET])

SINGLE_STEP_BOUNDING_BOX = jnp.ones((1 * STEP_HEIGHT, STEP_WIDTH)).astype(
    jnp.bool
)  # (1*8) x 4
LEFT_SINGLE_STEP_OFFSET = jnp.array([LEFT_SINGLE_STEP_X_OFFSET, SINGLE_STEP_Y_OFFSET])
RIGHT_SINGLE_STEP_OFFSET = jnp.array([RIGHT_SINGLE_STEP_X_OFFSET, SINGLE_STEP_Y_OFFSET])

INVISIBLE_BLOCK_BOUNDING_BOX = jnp.ones((2, INNER_WALL_THICKNESS)).astype(jnp.bool)
INVISIBLE_BLOCK_OFFSET = jnp.array(
    [INVISIBLE_BLOCK_LEFT_X_OFFSET, INVISIBLE_BLOCK_TOP_Y_OFFSET]
)

MIDDLE_BAR_Y = 72
MIDDLE_BAR_X = 104
MIDDLE_BAR_WIDTH = 16
MIDDLE_BAR_HEIGHT = 8


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
    horizontal: chex.Array  # 0: vertical, 1: horizontal


INVISIBLE_BLOCK_SCENE_OBJECT = SceneObject(
    hit_box_matrix=INVISIBLE_BLOCK_BOUNDING_BOX,
    hit_box_offset=INVISIBLE_BLOCK_OFFSET,
    reflecting=jnp.array(1),
    horizontal=jnp.array(1),
)


TOP_WALL_SCENE_OBJECT = SceneObject(
    hit_box_matrix=TOP_WALL_BOUNDING_BOX,
    hit_box_offset=TOP_WALL_OFFSET,
    reflecting=jnp.array(1),
    horizontal=jnp.array(1),
)
BOTTOM_WALL_SCENE_OBJECT = SceneObject(
    hit_box_matrix=BOTTOM_WALL_BOUNDING_BOX,
    hit_box_offset=BOTTOM_WALL_OFFSET,
    reflecting=jnp.array(1),
    horizontal=jnp.array(1),
)
LEFT_WALL_SCENE_OBJECT = SceneObject(
    hit_box_matrix=LEFT_WALL_BOUNDING_BOX,
    hit_box_offset=LEFT_WALL_OFFSET,
    reflecting=jnp.array(1),
    horizontal=jnp.array(0),
)
RIGHT_WALL_SCENE_OBJECT = SceneObject(
    hit_box_matrix=RIGHT_WALL_BOUNDING_BOX,
    hit_box_offset=RIGHT_WALL_OFFSET,
    reflecting=jnp.array(1),
    horizontal=jnp.array(0),
)

LEFT_INNER_WALL_SCENE_OBJECT = SceneObject(
    hit_box_matrix=LEFT_INNER_WALL_BOUNDING_BOX,
    hit_box_offset=LEFT_INNER_WALL_OFFSET,
    reflecting=jnp.array(1),
    horizontal=jnp.array(0),
)

RIGHT_INNER_WALL_SCENE_OBJECT = SceneObject(
    hit_box_matrix=RIGHT_INNER_WALL_BOUNDING_BOX,
    hit_box_offset=RIGHT_INNER_WALL_OFFSET,
    reflecting=jnp.array(1),
    horizontal=jnp.array(0),
)

LEFT_QUADRUPLE_STEP_SCENE_OBJECT = SceneObject(
    hit_box_matrix=QUADRUPLE_STEP_BOUNDING_BOX,
    hit_box_offset=LEFT_QUADRUPLE_STEP_OFFSET,
    reflecting=jnp.array(1),
    horizontal=jnp.array(0),
)
RIGHT_QUADRUPLE_STEP_SCENE_OBJECT = SceneObject(
    hit_box_matrix=QUADRUPLE_STEP_BOUNDING_BOX,
    hit_box_offset=RIGHT_QUADRUPLE_STEP_OFFSET,
    reflecting=jnp.array(1),
    horizontal=jnp.array(0),
)
LEFT_TRIPLE_STEP_SCENE_OBJECT = SceneObject(
    hit_box_matrix=TRIPLE_STEP_BOUNDING_BOX,
    hit_box_offset=LEFT_TRIPLE_STEP_OFFSET,
    reflecting=jnp.array(1),
    horizontal=jnp.array(0),
)
RIGHT_TRIPLE_STEP_SCENE_OBJECT = SceneObject(
    hit_box_matrix=TRIPLE_STEP_BOUNDING_BOX,
    hit_box_offset=RIGHT_TRIPLE_STEP_OFFSET,
    reflecting=jnp.array(1),
    horizontal=jnp.array(0),
)
LEFT_DOUBLE_STEP_SCENE_OBJECT = SceneObject(
    hit_box_matrix=DOUBLE_STEP_BOUNDING_BOX,
    hit_box_offset=LEFT_DOUBLE_STEP_OFFSET,
    reflecting=jnp.array(1),
    horizontal=jnp.array(0),
)
RIGHT_DOUBLE_STEP_SCENE_OBJECT = SceneObject(
    hit_box_matrix=DOUBLE_STEP_BOUNDING_BOX,
    hit_box_offset=RIGHT_DOUBLE_STEP_OFFSET,
    reflecting=jnp.array(1),
    horizontal=jnp.array(0),
)
LEFT_SINGLE_STEP_SCENE_OBJECT = SceneObject(
    hit_box_matrix=SINGLE_STEP_BOUNDING_BOX,
    hit_box_offset=LEFT_SINGLE_STEP_OFFSET,
    reflecting=jnp.array(1),
    horizontal=jnp.array(0),
)
RIGHT_SINGLE_STEP_SCENE_OBJECT = SceneObject(
    hit_box_matrix=SINGLE_STEP_BOUNDING_BOX,
    hit_box_offset=RIGHT_SINGLE_STEP_OFFSET,
    reflecting=jnp.array(1),
    horizontal=jnp.array(0),
)

REFLECTING_SCENE_OBJECT_LIST = [
    TOP_WALL_SCENE_OBJECT,
    BOTTOM_WALL_SCENE_OBJECT,
    LEFT_WALL_SCENE_OBJECT,
    RIGHT_WALL_SCENE_OBJECT,
    LEFT_INNER_WALL_SCENE_OBJECT,
    RIGHT_INNER_WALL_SCENE_OBJECT,
    LEFT_QUADRUPLE_STEP_SCENE_OBJECT,
    RIGHT_QUADRUPLE_STEP_SCENE_OBJECT,
    LEFT_TRIPLE_STEP_SCENE_OBJECT,
    RIGHT_TRIPLE_STEP_SCENE_OBJECT,
    LEFT_DOUBLE_STEP_SCENE_OBJECT,
    RIGHT_DOUBLE_STEP_SCENE_OBJECT,
    LEFT_SINGLE_STEP_SCENE_OBJECT,
    RIGHT_SINGLE_STEP_SCENE_OBJECT,
    # INVISIBLE_BLOCK_SCENE_OBJECT,
]
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


# Todo: Switch to a data class
@chex.dataclass
class HitPoint:
    t_entry: chex.Array
    x: chex.Array
    y: chex.Array


def get_human_action() -> chex.Array:
    """
    Records if UP or DOWN is being pressed and returns the corresponding action.

    Returns:
        action: int, action taken by the player (LEFT, RIGHT, FIRE, LEFTFIRE, RIGHTFIRE, NOOP).
    """
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        return jnp.array(Action.LEFT)
    elif keys[pygame.K_RIGHT]:
        return jnp.array(Action.RIGHT)
    elif keys[pygame.K_SPACE]:
        return jnp.array(Action.FIRE)
    elif keys[pygame.K_UP]:
        return jnp.array(Action.UP)
    elif keys[pygame.K_DOWN]:
        return jnp.array(Action.DOWN)
    else:
        return jnp.array(Action.NOOP)


# immutable state container
class VideoPinballState(NamedTuple):
    ball_x: chex.Array
    ball_y: chex.Array
    ball_vel_x: chex.Array
    ball_vel_y: chex.Array
    ball_direction: (chex.Array)  # 0: left/up, 1:left/down , 2: right/up, 3: right/down (Shouldn't this be a function?)
    left_flipper_angle: chex.Array
    right_flipper_angle: chex.Array
    plunger_position: (chex.Array)  # Value between 0 and 20 where 20 means that the plunger is fully pulled
    plunger_power: (chex.Array)  # 2 * plunger_position, only set to non-zero value once fired, reset after hitting invisible block
    score: chex.Array
    lives: chex.Array
    bumper_multiplier: chex.Array
    active_targets: chex.Array
    target_cooldown: chex.Array
    special_target_cooldown: chex.Array
    atari_symbols: chex.Array
    dropper_counter: chex.Array
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
    bumper_multiplier: jnp.ndarray
    active_targets: chex.Array
    atari_symbols: chex.Array
    dropper_counter: chex.Array


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
        jnp.logical_and(
            state.plunger_position < PLUNGER_MAX_POSITION,
            jnp.logical_and(action == Action.DOWN, jnp.logical_not(state.ball_in_play)),
        ),
        lambda s: s + 1,
        lambda s: s,
        operand=state.plunger_position,
    )

    # same for UP
    plunger_position = jax.lax.cond(
        jnp.logical_and(
            state.plunger_position > 0,
            jnp.logical_and(action == Action.UP, jnp.logical_not(state.ball_in_play)),
        ),
        lambda s: s - 1,
        lambda s: s,
        operand=plunger_position,
    )

    # If FIRE
    plunger_power = jax.lax.cond(
        jnp.logical_and(action == Action.FIRE, jnp.logical_not(state.ball_in_play)),
        lambda s: jnp.round(s / PLUNGER_MAX_POSITION * BALL_MAX_SPEED),
        lambda s: 0.0,
        operand=plunger_position,
    )

    plunger_position = jax.lax.cond(
        plunger_power > 0, lambda p: 0, lambda p: p, operand=plunger_position
    )

    return plunger_position, plunger_power


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
        + scene_object.hit_box_matrix.shape[1]
        - ball_movement.old_ball_x
    ) / trajectory_x
    ty1 = (scene_object.hit_box_offset[1] - ball_movement.old_ball_y) / trajectory_y
    ty2 = (
        scene_object.hit_box_offset[1]
        + scene_object.hit_box_matrix.shape[0]
        - ball_movement.old_ball_y
    ) / trajectory_y

    # Calculate the time of intersection with the bounding box
    tmin_x = jnp.minimum(tx1, tx2)
    tmax_x = jnp.maximum(tx1, tx2)
    tmin_y = jnp.minimum(ty1, ty2)
    tmax_y = jnp.maximum(ty1, ty2)

    # Calculate the time of entry and exit
    t_entry = jnp.maximum(tmin_x, tmin_y)
    t_exit = jnp.minimum(tmax_x, tmax_y)

    # t_entry > t_exit means that the ball is not colliding with the bounding box, because it has already passed it
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
        no_collision,
        lambda: jnp.array([T_ENTRY_NO_COLLISION, -1.0, -1.0]),
        lambda: hit_point,
    )


@jax.jit
def _reflect_ball(
    ball_movement: BallMovement,
    hit_point: chex.Array,
    scene_object_horizontal: chex.Array,
) -> tuple[chex.Array]:
    """
    From the previous ball position, a hit point, the ball_direction, and the velocity,
    computes the new ball position and direction after reflection off of the obstacle
    hit.
    """
    velocity_x = ball_movement.new_ball_x - ball_movement.old_ball_x
    velocity_y = ball_movement.new_ball_y - ball_movement.old_ball_y

    # Calculate the trajectory of the ball to the hit point
    trajectory_to_hit_point_x = ball_movement.new_ball_x - hit_point[1]
    trajectory_to_hit_point_y = ball_movement.new_ball_y - hit_point[2]

    # Calculate the surface normal of the hit point
    surface_normal_x = jnp.where(scene_object_horizontal, jnp.array(0), jnp.array(1))
    surface_normal_y = jnp.where(scene_object_horizontal, jnp.array(1), jnp.array(0))

    # Calculate the dot product of the velocity and the surface normal
    velocity_normal_prod = velocity_x * surface_normal_x + velocity_y * surface_normal_y
    velocity_normal_prod = (
        velocity_normal_prod - VELOCITY_DAMPENING_VALUE
    )  # Dampen the velocity a bit (value taken from RAM values)

    reflected_velocity_x = velocity_x - 2 * velocity_normal_prod * surface_normal_x
    reflected_velocity_y = velocity_y - 2 * velocity_normal_prod * surface_normal_y

    d_hit_point = jnp.sqrt(
        jnp.square(trajectory_to_hit_point_x) + jnp.square(trajectory_to_hit_point_y)
    )
    d_trajectory = jnp.sqrt(jnp.square(velocity_x) + jnp.square(velocity_y))

    r = d_hit_point / d_trajectory

    new_ball_x = hit_point[1] + r * reflected_velocity_x
    new_ball_y = hit_point[2] + r * reflected_velocity_y

    return jnp.round(new_ball_x), jnp.round(new_ball_y)


@jax.jit
def _check_reflecting_obstacle_hits(
    ball_movement: BallMovement,
) -> tuple[chex.Array, SceneObject]:
    """
    Check if the ball is hitting an obstacle.

    # top, left, right bottom, outer walls

    # left, right inner walls
    # middle bar
    # corner boxes top left, right, bottom left, right
    # bumper
    # drop targets
    # lit up targets
    # rollovers (left & atari)
    # spinner
    """
    # Jax should be able to optimize this (?) (also with a for loop ???)
    hit_points = jnp.stack(
        [
            _calc_hit_point(ball_movement, TOP_WALL_SCENE_OBJECT),
            _calc_hit_point(ball_movement, RIGHT_WALL_SCENE_OBJECT),
            _calc_hit_point(ball_movement, BOTTOM_WALL_SCENE_OBJECT),
            _calc_hit_point(ball_movement, LEFT_WALL_SCENE_OBJECT),
            _calc_hit_point(ball_movement, LEFT_INNER_WALL_SCENE_OBJECT),
            _calc_hit_point(ball_movement, RIGHT_INNER_WALL_SCENE_OBJECT),
            _calc_hit_point(ball_movement, LEFT_QUADRUPLE_STEP_SCENE_OBJECT),
            _calc_hit_point(ball_movement, RIGHT_QUADRUPLE_STEP_SCENE_OBJECT),
            _calc_hit_point(ball_movement, LEFT_TRIPLE_STEP_SCENE_OBJECT),
            _calc_hit_point(ball_movement, RIGHT_TRIPLE_STEP_SCENE_OBJECT),
            _calc_hit_point(ball_movement, LEFT_DOUBLE_STEP_SCENE_OBJECT),
            _calc_hit_point(ball_movement, RIGHT_DOUBLE_STEP_SCENE_OBJECT),
            _calc_hit_point(ball_movement, LEFT_SINGLE_STEP_SCENE_OBJECT),
            _calc_hit_point(ball_movement, RIGHT_SINGLE_STEP_SCENE_OBJECT),
        ],
        axis=0,
    )

    # Get and return first object hit (argmin entry time)
    lowest_entry_time_index = jnp.argmin(hit_points[:, 0])
    hit_point = hit_points[lowest_entry_time_index]

    scene_object = jax.lax.switch(
        lowest_entry_time_index,
        (
            lambda: TOP_WALL_SCENE_OBJECT.horizontal,
            lambda: RIGHT_WALL_SCENE_OBJECT.horizontal,
            lambda: BOTTOM_WALL_SCENE_OBJECT.horizontal,
            lambda: LEFT_WALL_SCENE_OBJECT.horizontal,
            lambda: LEFT_INNER_WALL_SCENE_OBJECT.horizontal,
            lambda: RIGHT_INNER_WALL_SCENE_OBJECT.horizontal,
            lambda: LEFT_QUADRUPLE_STEP_SCENE_OBJECT.horizontal,
            lambda: RIGHT_QUADRUPLE_STEP_SCENE_OBJECT.horizontal,
            lambda: LEFT_TRIPLE_STEP_SCENE_OBJECT.horizontal,
            lambda: RIGHT_TRIPLE_STEP_SCENE_OBJECT.horizontal,
            lambda: LEFT_DOUBLE_STEP_SCENE_OBJECT.horizontal,
            lambda: RIGHT_DOUBLE_STEP_SCENE_OBJECT.horizontal,
            lambda: LEFT_SINGLE_STEP_SCENE_OBJECT.horizontal,
            lambda: RIGHT_SINGLE_STEP_SCENE_OBJECT.horizontal,
        ),
    )
    return hit_point, scene_object


@jax.jit
def _get_signed_ball_directions(ball_direction) -> tuple[chex.Array, chex.Array]:
    x_sign = jnp.where(
        jnp.logical_or(ball_direction == 2, ball_direction == 3),
        jnp.array(1.0),
        jnp.array(-1.0),
    )
    y_sign = jnp.where(
        jnp.logical_or(ball_direction == 0, ball_direction == 2),
        jnp.array(-1.0),
        jnp.array(1.0),
    )
    return x_sign, y_sign


@jax.jit
def _get_ball_direction(signed_vel_x, signed_vel_y) -> chex.Array:
    # If both values are negative, we move closer to (0, 0) in the top left corner and fly in direction 0
    top_left = jnp.logical_and(signed_vel_x <= 0, signed_vel_y <= 0)  # 0
    top_right = jnp.logical_and(signed_vel_x > 0, signed_vel_y <= 0)  # 2
    bottom_right = jnp.logical_and(signed_vel_x > 0, signed_vel_y > 0)  # 3
    bottom_left = jnp.logical_and(signed_vel_x <= 0, signed_vel_y > 0)  # 1

    bool_array = jnp.array([top_left, bottom_left, top_right, bottom_right])
    return jnp.argmax(bool_array)


@jax.jit
def _calc_ball_change(ball_x, ball_y, ball_vel_x, ball_vel_y, ball_direction):
    sign_x, sign_y = _get_signed_ball_directions(ball_direction)
    ball_vel_x = jnp.clip(ball_vel_x, 0, BALL_MAX_SPEED)
    ball_vel_y = jnp.clip(ball_vel_y, 0, BALL_MAX_SPEED)
    signed_ball_vel_x = sign_x * ball_vel_x
    signed_ball_vel_y = sign_y * ball_vel_y
    # Only change position, direction and velocity if the ball is in play
    # TODO override ball_x, ball_y if obstacle hit (_reflect_ball)
    ball_x = ball_x + signed_ball_vel_x
    ball_y = ball_y + signed_ball_vel_y
    return ball_x, ball_y, ball_vel_x, ball_vel_y, signed_ball_vel_x, signed_ball_vel_y


@jax.jit
def ball_step(
    state: VideoPinballState,
    plunger_power,
    action,
):
    """
    Update the pinballs position and velocity based on the current state and action.
    """
    ball_x = state.ball_x
    ball_y = state.ball_y
    ball_vel_x = state.ball_vel_x
    ball_vel_y = state.ball_vel_y
    ball_direction = state.ball_direction
    ball_in_play = state.ball_in_play
    jax.debug.print(
        "Ball Step, Initial Ball x {ball_x}, y , {ball_y}, vel_x {ball_vel_x}, vel_y {ball_vel_y}, direction {ball_direction}",
        ball_x=ball_x,
        ball_y=ball_y,
        ball_vel_x=ball_vel_x,
        ball_vel_y=ball_vel_y,
        ball_direction=ball_direction,
    )
    """
    Plunger calculation
    """
    # Add plunger power to the ball velocity, only set to non-zero value once fired, reset after hitting invisible block
    ball_direction = jnp.where(
        plunger_power > 0,
        jnp.array(0),
        ball_direction,
    )  # Set direction to 0 if the ball is fired
    ball_vel_y = jnp.where(
        plunger_power > 0,
        ball_vel_y + plunger_power,
        ball_vel_y,
    )
    """
    Flipper calculation
    """
    # Check if the ball is hitting a paddle

    """
    Gravity calculation
    """
    # TODO: Test if the ball ever has velocity of state.plunger_power because right now we always
    # immediately deduct the gravity from the velocity
    # Direction has to be figured into the gravity calculation
    gravity_delta = jnp.where(
        jnp.logical_or(ball_direction == 0, ball_direction == 2), -GRAVITY, GRAVITY
    )  # Subtract gravity if the ball is moving up otherwise add it
    ball_vel_y = ball_vel_y + gravity_delta
    ball_direction = jnp.where(
        ball_vel_y < 0,
        ball_direction + 1,  # if ball direction was towards upper left
        ball_direction,
    )
    ball_vel_y = jnp.abs(ball_vel_y)
    jax.debug.print(
        "Ball Step, After Gravity Ball x {ball_x}, y , {ball_y}, vel_x {ball_vel_x}, vel_y {ball_vel_y}, direction {ball_direction}",
        ball_x=ball_x,
        ball_y=ball_y,
        ball_vel_x=ball_vel_x,
        ball_vel_y=ball_vel_y,
        ball_direction=ball_direction,
    )

    """
    Ball movement calculation observing its direction 
    """
    ball_x, ball_y, ball_vel_x, ball_vel_y, signed_ball_vel_x, signed_ball_vel_y = (
        _calc_ball_change(
            state.ball_x, state.ball_y, ball_vel_x, ball_vel_y, ball_direction
        )
    )
    jax.debug.print(
        "Ball Step, After Ball Update x {ball_x}, y , {ball_y}, vel_x {ball_vel_x}, vel_y {ball_vel_y}, direction {ball_direction}",
        ball_x=ball_x,
        ball_y=ball_y,
        ball_vel_x=ball_vel_x,
        ball_vel_y=ball_vel_y,
        ball_direction=ball_direction,
    )
    """
    Check if the ball is hitting the invisible block at the plunger hole
    """
    ball_movement = BallMovement(
        old_ball_x=state.ball_x,
        old_ball_y=state.ball_y,
        new_ball_x=ball_x,
        new_ball_y=ball_y,
    )
    invisible_block_hit_data = _calc_hit_point(
        ball_movement, INVISIBLE_BLOCK_SCENE_OBJECT
    )
    is_invisible_block_hit = jnp.logical_and(
        jnp.logical_not(ball_in_play),
        invisible_block_hit_data[0] != T_ENTRY_NO_COLLISION,
    )
    jax.debug.print(
        "Ball Step, invisible ball hit {invis_ball_hit}, ball in play {bip}, plunger_power {pp}",
        invis_ball_hit=is_invisible_block_hit,
        bip=ball_in_play,
        pp=plunger_power,
    )
    # set x vel to y vel and divide vel y by 5
    ball_vel_x = jnp.where(
        is_invisible_block_hit,
        ball_vel_y,
        ball_vel_x,
    )
    ball_vel_y = jnp.where(is_invisible_block_hit, ball_vel_y / 5, ball_vel_y)
    # set ball_x, ball_y to below invisible element and proceed with new ball_movement
    ball_x = jnp.where(
        is_invisible_block_hit,
        jnp.array(INVISIBLE_BLOCK_LEFT_X_OFFSET, dtype=jnp.float32),
        ball_x,
    )
    ball_y = jnp.where(
        is_invisible_block_hit,
        jnp.array(INVISIBLE_BLOCK_TOP_Y_OFFSET, dtype=jnp.float32),
        ball_y,
    )
    new_ball_direction = jnp.where(
        (plunger_power // 2) % 2 == 0, 0, 1  # semi random y direction
    )
    ball_direction = jnp.where(
        is_invisible_block_hit, new_ball_direction, ball_direction
    )
    sign_x, sign_y = _get_signed_ball_directions(ball_direction)
    signed_ball_vel_x = sign_x * ball_vel_x
    signed_ball_vel_y = sign_y * ball_vel_y
    ball_in_play = jnp.logical_or(ball_in_play, is_invisible_block_hit)

    """
    Obstacle hit calculation
    """
    # Calculate whether and where obstacles are hit
    # If a reflecting obstacle is hit, reflect the ball
    # If a non-reflecting obstacle is hit, proceed with usual ball position calculation
    # new_ball_direction, _ball_vel_x, _ball_vel_y = _get_obstacle_hit_direction()
    ball_movement = BallMovement(
        old_ball_x=state.ball_x,
        old_ball_y=state.ball_y,
        new_ball_x=ball_x,
        new_ball_y=ball_y,
    )

    hit_data, scene_object_horizontal = _check_reflecting_obstacle_hits(ball_movement)
    # TODO what happens if the ball is reflected into another reflecting object => maybe a while-loop
    reflected_ball_x, reflected_ball_y = _reflect_ball(
        ball_movement, hit_data, scene_object_horizontal
    )

    # vector from hit point to end point of reflected trajectory
    reflection_trajectory_x = reflected_ball_x - hit_data[1]
    reflection_trajectory_y = reflected_ball_y - hit_data[2]

    ball_x = jnp.where(hit_data[0] == T_ENTRY_NO_COLLISION, ball_x, reflected_ball_x)
    ball_y = jnp.where(hit_data[0] == T_ENTRY_NO_COLLISION, ball_y, reflected_ball_y)
    ball_trajectory_x = jnp.where(
        hit_data[0] == T_ENTRY_NO_COLLISION, signed_ball_vel_x, reflection_trajectory_x
    )
    ball_trajectory_y = jnp.where(
        hit_data[0] == T_ENTRY_NO_COLLISION, signed_ball_vel_y, reflection_trajectory_y
    )
    jax.debug.print(
        "Signed Ball Vel y {signed_ball_vel_y}, Reflect Trajectory y {reflection_trajectory_y}",
        signed_ball_vel_y=signed_ball_vel_y,
        reflection_trajectory_y=reflection_trajectory_y,
    )
    jax.debug.print(
        "Ball Step, After Obstacle Update x {ball_x}, y , {ball_y}, vel_x {ball_vel_x}, vel_y {ball_vel_y}, direction {ball_direction}",
        ball_x=ball_x,
        ball_y=ball_y,
        ball_vel_x=ball_vel_x,
        ball_vel_y=ball_vel_y,
        ball_direction=ball_direction,
    )

    """
    Some final calculations
    """
    jax.debug.print(
        "Ball Trajectory x, Ball Trajectory y {ball_trajectory_x}, {ball_trajectory_y}",
        ball_trajectory_x=ball_trajectory_x,
        ball_trajectory_y=ball_trajectory_y,
    )
    ball_direction = _get_ball_direction(ball_trajectory_x, ball_trajectory_y)
    # Clip the ball velocity to the maximum speed
    ball_vel_x = jnp.clip(jnp.abs(signed_ball_vel_x), 0, BALL_MAX_SPEED)
    ball_vel_y = jnp.clip(jnp.abs(signed_ball_vel_y), 0, BALL_MAX_SPEED)

    """
    Check if ball is in play if not ignore the calculations
    """
    # TODO: Maybe do the stuff above in a function that is called if we are in play
    # ball_direction = jnp.where(ball_in_play, ball_direction, BALL_START_DIRECTION)
    # ball_x = jnp.where(ball_in_play, ball_x, BALL_START_X)
    # ball_y = jnp.where(ball_in_play, ball_y, BALL_START_Y)
    # ball_vel_x = jnp.where(ball_in_play, ball_vel_x, jnp.array(0.0))
    # ball_vel_y = jnp.where(ball_in_play, ball_vel_y, jnp.array(0.0))
    jax.debug.print(
        "Ball Step, Final Ball x {ball_x}, y , {ball_y}, vel_x {ball_vel_x}, vel_y {ball_vel_y}, direction {ball_direction}",
        ball_x=ball_x,
        ball_y=ball_y,
        ball_vel_x=ball_vel_x,
        ball_vel_y=ball_vel_y,
        ball_direction=ball_direction,
    )
    return (ball_x, ball_y, ball_direction, ball_vel_x, ball_vel_y, ball_in_play)


@jax.jit
def _reset_ball(state: VideoPinballState):
    """
    When the ball goes into the gutter or into the plunger hole,
    respawn the ball on the launcher.
    """

    return BALL_START_X, BALL_START_Y, jnp.array(0.0), jnp.array(0.0)


def _handle_ball_in_gutter(state: VideoPinballState, score, atari_symbols, dropper_counter):
    lives = jax.lax.cond(
        atari_symbols < 4,
        lambda x: x + 1,
        lambda x: x,
        operand=state.lives,
    )

    # TODO: This should slowly happen one by one but this will do for now
    score = score + ((dropper_counter - 1) * atari_symbols)
    dropper_counter = jnp.array(1).astype(jnp.int32)

    atari_symbols = jnp.array(0).astype(jnp.int32)

    bumper_multiplier = jnp.array(1).astype(jnp.int32)
    active_targets = jnp.array([1, 1, 1, 0]).astype(jnp.int32)

    return score, active_targets, atari_symbols, dropper_counter, bumper_multiplier, lives



@jax.jit
def process_objects_hit(state: VideoPinballState, objects_hit):
    # Bumpers: Give points
    # Targets: Make them disappear, give points
    # Targets: Check if all hit, increase multiplier
    # BonusTarget: Give points, make screen flash, something else?
    # Dropper: Give points, increase number
    # Atari: Give points, make Atari symbol at bottom appear
    # Assume objects_hit is list: [LeftBumper, MiddleBumper, RightBumper, LeftTarget, MiddleTarget, RightTarget,
    #                              BonusTarget, Dropper, AtariSymbol, Spinner]

    # ball_vel_y = jnp.where(
    #     plunger_power > 0,
    #     ball_vel_y + plunger_power,
    #     ball_vel_y,
    # )

    # plunger_position = jax.lax.cond(
    #     jnp.logical_and(
    #         state.plunger_position < PLUNGER_MAX_POSITION,
    #         jnp.logical_and(action == Action.DOWN, jnp.logical_not(state.ball_in_play)),
    #     ),
    #     lambda s: s + 1,
    #     lambda s: s,
    #     operand=state.plunger_position,
    # )

    # Bumper points
    score = state.score
    active_targets = state.active_targets
    atari_symbols = state.atari_symbols
    dropper_counter = state.dropper_counter

    score += jnp.where(objects_hit[0],
        100 * state.bumper_multiplier,
        0,
    )
    score += jnp.where(
        objects_hit[1],
        100 * state.bumper_multiplier,
        0,
    )
    score += jnp.where(
        objects_hit[2],
        100 * state.bumper_multiplier,
        0,
    )

    # Give points for targets hit
    score += jnp.where(objects_hit[3],100,0)
    score += jnp.where(objects_hit[4],100,0)
    score += jnp.where(objects_hit[5],100,0)

    # Make hit targets disappear
    active_targets = jax.lax.cond(
        objects_hit[3],
        lambda s: jnp.array([0, s[1], s[2], s[3]]).astype(jnp.int32),
        lambda s: s,
        operand=active_targets,
    )

    active_targets = jax.lax.cond(
        objects_hit[4],
        lambda s: jnp.array([s[0], 0, s[2], s[3]]).astype(jnp.int32),
        lambda s: s,
        operand=active_targets,
    )

    active_targets = jax.lax.cond(
        objects_hit[5],
        lambda s: jnp.array([s[0], s[1], 0, s[3]]).astype(jnp.int32),
        lambda s: s,
        operand=active_targets,
    )

    # Bottom Bonus Target
    score += jnp.where(objects_hit[6],1100,0)
    active_targets = jax.lax.cond(
        objects_hit[6],
        lambda s: jnp.array([s[0], s[1], s[2], 0]).astype(jnp.int32),
        lambda s: s,
        operand=active_targets,
    )

    # Give score for hitting the dropper and increase its number
    score += jnp.where(objects_hit[7],100,0)
    #TODO: Check if dropper really goes up further beyond 9 even though the number isnt displayed
    dropper_counter = jax.lax.cond(
        objects_hit[7],
        lambda s: s + 1,
        lambda s: s,
        operand=dropper_counter,
    )

    # Give score for hitting the Atari symbol and make a symbol appear at the bottom
    score += jnp.where(objects_hit[8], 100, 0)
    atari_symbols = jax.lax.cond(
        jnp.logical_and(objects_hit[8], atari_symbols < 4),
        lambda s: s + 1,
        lambda s: s,
        operand=atari_symbols,
    )

    # Give 1 point for hitting a spinner
    score += jnp.where(objects_hit[9],1,0)

    return score, active_targets, atari_symbols, dropper_counter



@jax.jit
def handle_target_cooldowns(state: VideoPinballState, previous_active_targets):

    # 2 second cooldown after hitting all targets until they respawn
    target_cooldown, active_targets, increase_bm = jax.lax.cond(
        jnp.logical_and(
            jnp.logical_and(previous_active_targets[0] == 0, state.target_cooldown == -1),
            jnp.logical_and(previous_active_targets[1] == 0, previous_active_targets[2] == 0)
        ),
        lambda cd, a: (60, jnp.array([1, 1, 1, a[3]]).astype(jnp.int32), True),
        lambda cd, a: (cd, a, False),
        state.target_cooldown, previous_active_targets)

    # Increase Bumper multiplier if all targets got hit
    bumper_multiplier = jax.lax.cond(jnp.logical_and(increase_bm, state.bumper_multiplier < 9),
                                           lambda s: s + 1,
                                           lambda s: s,
                                           operand=state.bumper_multiplier)

    # count down the cooldown timer
    target_cooldown = jax.lax.cond(
        jnp.logical_and(
            jnp.logical_and(active_targets[0] == 0, target_cooldown != -1),
            jnp.logical_and(active_targets[1] == 0, active_targets[2] == 0)
        ),
        lambda s: s - 1,
        lambda s: s,
        operand=target_cooldown)

    # count down the despawn cooldown timer
    special_target_cooldown = jax.lax.cond(jnp.logical_and(active_targets[3] > 0, state.ball_in_play),
                                           lambda s: s - 1,
                                           lambda s: s,
                                           operand=state.special_target_cooldown)

    # count up the respawn cooldown timer
    special_target_cooldown = jax.lax.cond(jnp.logical_and(active_targets[3] < -1, state.ball_in_play),
                                           lambda s: s + 1,
                                           lambda s: s,
                                           operand=special_target_cooldown)

    # despawn the special target
    special_target_cooldown, active_targets = jax.lax.cond(
        jnp.logical_and(active_targets[3] == 0, state.ball_in_play),
        lambda cd, a: (cd - 600, a.at[3].set(0)),  # Check how the real cooldown works
        lambda cd, a: (cd, a),
        special_target_cooldown, active_targets)

    # spawn the special target
    special_target_cooldown, active_targets = jax.lax.cond(
        jnp.logical_and(active_targets[3] == -1, state.ball_in_play),
        lambda cd, a: (cd + 181, a.at[3].set(1)),
        lambda cd, a: (cd, a),
        special_target_cooldown, active_targets)


    return active_targets, target_cooldown, special_target_cooldown, bumper_multiplier



def _split_integer(number: jnp.ndarray, max_digits: int = 6) -> jnp.ndarray:
    """
    Splits an integer into a JAX array of its individual digits.

    The output array will have a fixed size determined by `max_digits`.
    If the input number has fewer digits than `max_digits`, it will be
    padded with leading zeros.

    Args:
        number: The integer to split. Should be a non-negative integer.
                Can be a Python int or a JAX array.
        max_digits: The maximum number of digits expected. The output array
                    will have this many elements.

    Returns:
        A 1D JAX array where each element is a digit of the input number.

    Example:
        split_integer(247900, max_digits=6) == jnp.array([2, 4, 7, 9, 0, 0])
        split_integer(123, max_digits=6)    == jnp.array([0, 0, 0, 1, 2, 3])
        split_integer(0, max_digits=3)      == jnp.array([0, 0, 0])
    """
    # Ensure the input number is a JAX array.
    # This handles both Python integers and existing JAX arrays.
    number = jnp.asarray(number, dtype=jnp.int32)

    # Create an array of powers of 10 to extract digits from left to right.
    # For max_digits=6, this will be [10^5, 10^4, 10^3, 10^2, 10^1, 10^0]
    powers_of_10 = 10 ** jnp.arange(max_digits - 1, -1, -1, dtype=jnp.int32)

    # Perform integer division by powers of 10, then take modulo 10
    # to isolate each digit.
    # Example:
    # (247900 // 100000) % 10 = 2
    # (247900 // 10000) % 10 = 4
    # ...
    # (247900 // 1) % 10 = 0
    digits = (number // powers_of_10) % 10

    return digits


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
        Resets the game state to the initial state.
        Returns the initial state and the reward (i.e. 0)
        """
        state = VideoPinballState(
            ball_x=jnp.array(BALL_START_X).astype(jnp.float32),
            ball_y=jnp.array(BALL_START_Y).astype(jnp.float32),
            ball_vel_x=jnp.array(0.0),
            ball_vel_y=jnp.array(0.0),
            ball_direction=jnp.array(0).astype(jnp.int32),
            left_flipper_angle=jnp.array(0).astype(jnp.int32),
            right_flipper_angle=jnp.array(0).astype(jnp.int32),
            plunger_position=jnp.array(0).astype(jnp.int32),
            plunger_power=jnp.array(0).astype(jnp.float32),
            score=jnp.array(0).astype(jnp.int32),
            lives=jnp.array(1).astype(jnp.int32),
            bumper_multiplier=jnp.array(1).astype(jnp.int32),
            active_targets=jnp.array([1, 1, 1, 0]).astype(jnp.int32),
            target_cooldown=jnp.array(-1).astype(jnp.int32),
            special_target_cooldown=jnp.array(120).astype(jnp.int32),
            atari_symbols=jnp.array(0).astype(jnp.int32),
            dropper_counter=jnp.array(1).astype(jnp.int32),
            step_counter=jnp.array(0).astype(jnp.int32),
            ball_in_play=jnp.array(False).astype(jnp.bool),
        )

        initial_obs = self._get_observation(state)
        return initial_obs, state  # , initial_obs

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, state: VideoPinballState, action: chex.Array
    ) -> Tuple[
        VideoPinballState, VideoPinballObservation, float, bool, VideoPinballInfo
    ]:
        # chex provides jax with additional debug/testing functionality.
        # Probably best to use it instead of simply jnp.array

        # Step 1: Update Plunger and Flippers
        plunger_position, new_plunger_power = plunger_step(state, action)
        # Update plunger power only if it is > 0
        plunger_power = jax.lax.cond(
            new_plunger_power > 0,
            lambda x: x,
            lambda _: state.plunger_power,
            operand=new_plunger_power,
        )
        left_flipper_angle, right_flipper_angle = flipper_step(state, action)

        # Print new_plunger_power and plunger_power
        jax.debug.print(
            "Plunger Step, Plunger Power {plunger_power}, New Plunger Power {new_plunger_power}",
            plunger_power=plunger_power,
            new_plunger_power=new_plunger_power,
        )

        # Step 2: Update ball position and velocity
        ball_x, ball_y, ball_direction, ball_vel_x, ball_vel_y, ball_in_play = (
            ball_step(
                state,
                new_plunger_power,
                action,
            )
        )

        # Step 3: Check if ball is in the gutter or in plunger hole
        ball_in_gutter = ball_y > 209
        # TODO if ball is back in plunger hole reset, not instantly
        ball_reset = jnp.logical_or(
            ball_in_gutter,
            jnp.logical_and(ball_x > 148, ball_y > 129),
        )

        # Step 4: Update scores and handle special objects
        # TODO: Input the list of objects hit once collisions are done
        score, active_targets, atari_symbols, dropper_counter = process_objects_hit(state, [False, False, False, False, False, False, False, False, False, False])
        active_targets, target_cooldown, special_target_cooldown, bumper_multiplier = handle_target_cooldowns(state, active_targets)


        # Step 5: Reset ball if it went down the gutter
        current_values = (
            ball_x,
            ball_y,
            ball_vel_x,
            ball_vel_y,
        )

        ball_x_final, ball_y_final, ball_vel_x_final, ball_vel_y_final = jax.lax.cond(
            ball_reset,
            lambda x: _reset_ball(state),
            lambda x: x,
            operand=current_values,
        )

        score, active_targets, atari_symbols, dropper_counter, bumper_multiplier, lives = jax.lax.cond(
            ball_in_gutter,
            lambda s, at, asym, dc, bm: _handle_ball_in_gutter(state, s, asym, dc),
            lambda s, at, asym, dc, bm: (s, at, asym, dc, bm, state.lives),
            score, active_targets, atari_symbols, dropper_counter, bumper_multiplier
        )

        ball_in_play = jnp.where(ball_reset, jnp.array(False), ball_in_play)

        new_state = VideoPinballState(
            ball_x=ball_x_final,
            ball_y=ball_y_final,
            ball_vel_x=ball_vel_x_final,
            ball_vel_y=ball_vel_y_final,
            ball_direction=ball_direction,
            left_flipper_angle=left_flipper_angle,
            right_flipper_angle=right_flipper_angle,
            plunger_position=plunger_position,
            plunger_power=plunger_power,
            score=score,
            lives=lives,
            bumper_multiplier=bumper_multiplier,
            active_targets=active_targets,
            target_cooldown=target_cooldown,
            special_target_cooldown=special_target_cooldown,
            atari_symbols=atari_symbols,
            dropper_counter=dropper_counter,
            step_counter=jnp.array(state.step_counter + 1).astype(jnp.int32),
            ball_in_play=ball_in_play,
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
        jax.debug.print("------------------------------------------")
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
            bumpers=state.bumper_multiplier,  # bumper states array
            targets=state.active_targets,  # target states array
            score=state.score,
            lives=state.lives,
            bumper_multiplier=state.bumper_multiplier,
            active_targets=state.active_targets,
            atari_symbols=state.atari_symbols,
            dropper_counter=state.dropper_counter
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
                obs.bumper_multiplier.flatten(),
            ]
        )

    def action_space(self) -> spaces.Discrete:
        """Returns the action space for Pong.
        Actions are:
        0: NOOP
        1: FIRE
        2: RIGHT
        3: LEFT
        4: RIGHTFIRE
        5: LEFTFIRE
        """
        return spaces.Discrete(len(self.action_set))

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
        return 0

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

@jax.jit
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


class VideoPinballRenderer(AtraJaxisRenderer):
    """JAX-based Video Pinball game renderer, optimized with JIT compilation."""

    def __init__(self):
        self.sprites = load_sprites()

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: VideoPinballState):
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
        raster = aj.render_at(
            raster,
            64,
            184 - FLIPPER_ANIMATION_Y_OFFSETS[state.left_flipper_angle],
            frame_flipper_left,
        )

        frame_flipper_right = aj.get_sprite_frame(
            self.sprites["flipper_right"], state.right_flipper_angle
        )
        raster = aj.render_at(
            raster,
            83 + FLIPPER_ANIMATION_X_OFFSETS[state.right_flipper_angle],
            184 - FLIPPER_ANIMATION_Y_OFFSETS[state.right_flipper_angle],
            frame_flipper_right,
        )

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

        # Render score TODO: (unfinished, game_state implementation needed)
        frame_unknown = aj.get_sprite_frame(self.sprites["score_number_digits"], 1)
        raster = aj.render_at(raster, 4, 3, frame_unknown)

        frame_ball_count = aj.get_sprite_frame(self.sprites["score_number_digits"], state.lives)
        raster = aj.render_at(raster, 36, 3, frame_ball_count)

        numbers = _split_integer(state.score)
        frame_score1 = aj.get_sprite_frame(self.sprites["score_number_digits"], numbers[0])
        raster = aj.render_at(raster, 64, 3, frame_score1)
        frame_score2 = aj.get_sprite_frame(self.sprites["score_number_digits"], numbers[1])
        raster = aj.render_at(raster, 80, 3, frame_score2)
        frame_score3 = aj.get_sprite_frame(self.sprites["score_number_digits"], numbers[2])
        raster = aj.render_at(raster, 96, 3, frame_score3)
        frame_score4 = aj.get_sprite_frame(self.sprites["score_number_digits"], numbers[3])
        raster = aj.render_at(raster, 112, 3, frame_score4)
        frame_score5 = aj.get_sprite_frame(self.sprites["score_number_digits"], numbers[4])
        raster = aj.render_at(raster, 128, 3, frame_score5)
        frame_score6 = aj.get_sprite_frame(self.sprites["score_number_digits"], numbers[5])
        raster = aj.render_at(raster, 144, 3, frame_score6)

        # Render special yellow field objects TODO: (unfinished, game_state implementation needed)
        frame_bumper_left = aj.get_sprite_frame(self.sprites["field_number_digits"], state.bumper_multiplier)
        raster = aj.render_at(raster, 46, 122, frame_bumper_left)
        frame_bumper_middle = aj.get_sprite_frame(self.sprites["field_number_digits"], state.bumper_multiplier)
        raster = aj.render_at(raster, 78, 58, frame_bumper_middle)
        frame_bumper_right = aj.get_sprite_frame(self.sprites["field_number_digits"], state.bumper_multiplier)
        raster = aj.render_at(raster, 110, 122, frame_bumper_right)

        displayed_dropper_number = jax.lax.cond(
            state.dropper_counter > 9,
            lambda c: 9,
            lambda c: c,
            operand=state.dropper_counter
        )
        frame_dropper_left = aj.get_sprite_frame(self.sprites["field_number_digits"], displayed_dropper_number)
        raster = aj.render_at(raster, 46, 58, frame_dropper_left)
        frame_atari_logo = aj.get_sprite_frame(self.sprites["atari_logo"], 0)
        raster = aj.render_at(raster, 109, 58, frame_atari_logo)

        frame_target = aj.get_sprite_frame(self.sprites["yellow_diamond_top"], 0)
        raster = jax.lax.cond(
            state.active_targets[0],
            lambda r: aj.render_at(raster, 60, 24, frame_target),
            lambda r: raster,
            operand=raster
        )

        raster = jax.lax.cond(
            state.active_targets[1],
            lambda r: aj.render_at(raster, 76, 24, frame_target),
            lambda r: raster,
            operand=raster
        )

        raster = jax.lax.cond(
            state.active_targets[2],
            lambda r: aj.render_at(raster, 92, 24, frame_target),
            lambda r: raster,
            operand=raster
        )


        frame_special_target = aj.get_sprite_frame(self.sprites["yellow_diamond_bottom"], 0)
        raster = jax.lax.cond(
            state.active_targets[3],
            lambda r: aj.render_at(raster, 76, 120, frame_special_target),
            lambda r: raster,
            operand=raster
        )

        # Render Atari Logos and the X
        raster = jax.lax.cond(
            state.atari_symbols > 0,
            lambda r: aj.render_at(raster, 60, 154, frame_atari_logo),
            lambda r: raster,
            operand=raster
        )

        raster = jax.lax.cond(
            jnp.logical_or(state.atari_symbols == 2, state.atari_symbols == 3),
            lambda r: aj.render_at(raster, 76, 154, frame_atari_logo),
            lambda r: raster,
            operand=raster
        )

        raster = jax.lax.cond(
            state.atari_symbols > 2,
            lambda r: aj.render_at(raster, 90, 154, frame_atari_logo),
            lambda r: raster,
            operand=raster
        )

        frame_X = aj.get_sprite_frame(self.sprites["x"], 0)
        raster = jax.lax.cond(
            state.atari_symbols == 4,
            lambda r: aj.render_at(raster, 76, 157, frame_X),
            lambda r: raster,
            operand=raster
        )

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
    renderer = VideoPinballRenderer()

    # Get jitted functions
    jitted_step = jax.jit(game.step)
    jitted_reset = jax.jit(game.reset)

    obs, curr_state = jitted_reset(prng_key=prng_key)

    # Game loop
    running = True
    frame_by_frame = False
    frameskip = game.frameskip
    counter = 1
    reset = False

    while running:
        reset = False
        # Event loop that checks display settings (QUIT, frame-by-frame-mode toggled, next for frame-by-frame mode)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    frame_by_frame = not frame_by_frame
                if event.key == pygame.K_r:
                    counter = 1
                    prng_key, key = jrandom.split(prng_key)
                    obs, curr_state = jitted_reset(prng_key=key)
                    reset = True
            elif event.type == pygame.KEYDOWN or (
                event.type == pygame.KEYUP and event.key == pygame.K_n
            ):
                if event.key == pygame.K_n and frame_by_frame:
                    if counter % frameskip == 0:
                        # If frame-by-frame mode activated and next frame is requested,
                        # get human (game) action and perform step
                        action = get_human_action()
                        obs, curr_state, reward, done, info = jitted_step(
                            curr_state, action
                        )

        if not frame_by_frame or reset:
            # If not in frame-by-frame mode perform step at each clock-tick
            # i.e. get human (game) action
            if counter % frameskip == 0:
                action = get_human_action()
                # Update game step
                obs, curr_state, reward, done, info = jitted_step(curr_state, action)

        # Render and display
        raster = renderer.render(curr_state)

        aj.update_pygame(screen, raster, 3, WIDTH, HEIGHT)

        counter += 1
        clock.tick(60)

    pygame.quit()
