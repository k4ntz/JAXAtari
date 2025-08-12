"""
Project: JAXAtari VideoPinball
Description: Our team's JAX implementation of Video Pinball.

Authors:
    - Michael Olenberger <michael.olenberger@stud.tu-darmstadt.de>
    - Maximilian Roth <maximilian.roth@stud.tu-darmstadt.de>
    - Jonas Neumann <jonas.neumann@stud.tu-darmstadt.de>
    - Yuddhish Chooah <yuddhish.chooah@stud.tu-darmstadt.de>


Implemented features:
- Plunger and Flipper movement logic
- Plunger physics
- Jit-compatible rendering
- Special object logic (like yellow targets, rollovers, etc.)
- Scoring
- Ball respawning upon hitting the gutter and life counter
- Basic collisions and ball physics

Why the ball physics are not yet perfect:
Video Pinball has extremely complicated ball physics. Launch angles when hitting (close to) corners are
seemingly random and velocity calculation has a variety of strange quirks like strong spontaneous acceleration
when a slow balls hit walls at certain angles, etc...
These properties are impossible to read from the RAM state and need to be investigated
frame by frame in various scenarios. Thus, the physics are far from perfect.
Collisions are mostly implemented apart from the flippers but non-reflective collisions
such as when hitting a target are still missing, even though the logic for these objects such as the targets
(like increasing bumper multiplier and respawn cooldown etc.) are already implemented so the game
is a little more complete than it seems.

Additional notes:
The renderer requires a custom function that was implemented in atraJaxis.py
We were not working on the game for almost 2 months as most of the team debated whether they
should quit the project or not, which is why the game is still in such an incomplete state.
However, we are now determined to continue.

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
GRAVITY = 0.03  # 0.12
VELOCITY_DAMPENING_VALUE = 0.1  # 24
BALL_MAX_SPEED = 3
FLIPPER_MAX_ANGLE = 3
FLIPPER_ANIMATION_Y_OFFSETS = jnp.array(
    [0, 0, 3, 7]
)  # This is a little scuffed, it would be cleaner to just fix the sprites but this works fine
FLIPPER_ANIMATION_X_OFFSETS = jnp.array([0, 0, 0, 1])  # Only for the right flipper
PLUNGER_MAX_POSITION = 20

# Game logic constants
T_ENTRY_NO_COLLISION = 9999
TARGET_RESPAWN_COOLDOWN = 16
SPECIAL_TARGET_ACTIVE_DURATION = 257
SPECIAL_TARGET_INACTIVE_DURATION = 787
TARGET_RESPAWN_COOLDOWN = 16
SPECIAL_TARGET_ACTIVE_DURATION = 257
SPECIAL_TARGET_INACTIVE_DURATION = 787

# Game layout constants
# TODO: check if these are correct
BALL_SIZE = (2, 4)
FLIPPER_LEFT_POS = (30, 180)
FLIPPER_RIGHT_POS = (110, 180)
PLUNGER_POS = (150, 120)
PLUNGER_MAX_HEIGHT = 20  # Taken from RAM values (67-87)
INVISIBLE_BLOCK_REFLECTION_FACTOR = (
    2  # 8 times the plunger power is added to ball_vel_x
)


# Background color and object colors
BG_COLOR = 0, 0, 0
TILT_MODE_COLOR = 167, 26, 26
BACKGROUND_COLOR = jnp.array([0, 0, 0], dtype=jnp.uint8)
WALL_COLOR = jnp.array([104, 72, 198], dtype=jnp.uint8)
GROUP3_COLOR = jnp.array([187, 159, 71], dtype=jnp.uint8)
GROUP4_COLOR = jnp.array([210, 164, 74], dtype=jnp.uint8)
GROUP5_COLOR = jnp.array([236, 236, 236], dtype=jnp.uint8)

# Color cycling arrays (each inner tuple converted to a list with alpha)
BACKGROUND_COLOR_CYCLING = jnp.array(
    [
        [74, 74, 74],
        [111, 111, 111],
        [142, 142, 142],
        [170, 170, 170],
        [192, 192, 192],
        [214, 214, 214],
        [236, 236, 236],
        [72, 72, 0],
    ],
    dtype=jnp.uint8,
)

WALL_COLOR_CYCLING = jnp.array(
    [
        [78, 50, 181],
        [51, 26, 163],
        [20, 0, 144],
        [188, 144, 252],
        [169, 128, 240],
        [149, 111, 227],
        [127, 92, 213],
        [146, 70, 192],
    ],
    dtype=jnp.uint8,
)

GROUP3_COLOR_CYCLING = jnp.array(
    [
        [210, 182, 86],
        [232, 204, 99],
        [252, 224, 112],
        [72, 44, 0],
        [105, 77, 20],
        [134, 106, 38],
        [162, 134, 56],
        [160, 171, 79],
    ],
    dtype=jnp.uint8,
)

GROUP4_COLOR_CYCLING = jnp.array(
    [
        [195, 144, 61],
        [236, 200, 96],
        [223, 183, 85],
        [144, 72, 17],
        [124, 44, 0],
        [180, 122, 48],
        [162, 98, 33],
        [227, 151, 89],
    ],
    dtype=jnp.uint8,
)

GROUP5_COLOR_CYCLING = jnp.array(
    [
        [214, 214, 214],
        [192, 192, 192],
        [170, 170, 170],
        [142, 142, 142],
        [111, 111, 111],
        [74, 74, 74],
        [0, 0, 0],
        [252, 252, 84],
    ],
    dtype=jnp.uint8,
)

# BACKGROUND_COLOR = 0, 0, 0
# WALL_COLOR = 104, 72, 198
# GROUP3_COLOR = 187, 159, 71
# GROUP4_COLOR = 210, 164, 74
# GROUP5_COLOR = 236, 236, 236
# TILT_MODE_COLOR = 167, 26, 26
# BACKGROUND_COLOR_CYCLING = jnp.array([(74, 74, 74), (111, 111, 111), (142, 142, 142), (170, 170, 170),
#                                       (192, 192, 192), (214, 214, 214), (236, 236, 236), (72, 72, 0)])
# WALL_COLOR_CYCLING = jnp.array([(78, 50, 181), (51, 26, 163), (20, 0, 144), (188, 144, 252),
#                                 (169, 128, 240), (149, 111, 227), (127, 92, 213), (146, 70, 192)])
# GROUP3_COLOR_CYCLING = jnp.array([(210, 182, 86), (232, 204, 99), (252, 224, 112), (72, 44, 0),
#                                   (105, 77, 20), (134, 106, 38), (162, 134, 56), (160, 171, 79)])
# GROUP4_COLOR_CYCLING = jnp.array([(195, 144, 61), (236, 200, 96), (223, 183, 85), (144, 72, 17),
#                                   (124, 44, 0), (180, 122, 48), (162, 98, 33), (227, 151, 89)])
# GROUP5_COLOR_CYCLING = jnp.array([(214, 214, 214), (192, 192, 192), (170, 170, 170), (142, 142, 142),
#                                   (111, 111, 111), (74, 74, 74), (0, 0, 0), (252, 252, 84)])
BG_COLOR = 0, 0, 0
TILT_MODE_COLOR = 167, 26, 26
BACKGROUND_COLOR = jnp.array([0, 0, 0], dtype=jnp.uint8)
WALL_COLOR = jnp.array([104, 72, 198], dtype=jnp.uint8)
GROUP3_COLOR = jnp.array([187, 159, 71], dtype=jnp.uint8)
GROUP4_COLOR = jnp.array([210, 164, 74], dtype=jnp.uint8)
GROUP5_COLOR = jnp.array([236, 236, 236], dtype=jnp.uint8)

# Color cycling arrays (each inner tuple converted to a list with alpha)
BACKGROUND_COLOR_CYCLING = jnp.array(
    [
        [74, 74, 74],
        [111, 111, 111],
        [142, 142, 142],
        [170, 170, 170],
        [192, 192, 192],
        [214, 214, 214],
        [236, 236, 236],
        [72, 72, 0],
    ],
    dtype=jnp.uint8,
)

WALL_COLOR_CYCLING = jnp.array(
    [
        [78, 50, 181],
        [51, 26, 163],
        [20, 0, 144],
        [188, 144, 252],
        [169, 128, 240],
        [149, 111, 227],
        [127, 92, 213],
        [146, 70, 192],
    ],
    dtype=jnp.uint8,
)

GROUP3_COLOR_CYCLING = jnp.array(
    [
        [210, 182, 86],
        [232, 204, 99],
        [252, 224, 112],
        [72, 44, 0],
        [105, 77, 20],
        [134, 106, 38],
        [162, 134, 56],
        [160, 171, 79],
    ],
    dtype=jnp.uint8,
)

GROUP4_COLOR_CYCLING = jnp.array(
    [
        [195, 144, 61],
        [236, 200, 96],
        [223, 183, 85],
        [144, 72, 17],
        [124, 44, 0],
        [180, 122, 48],
        [162, 98, 33],
        [227, 151, 89],
    ],
    dtype=jnp.uint8,
)

GROUP5_COLOR_CYCLING = jnp.array(
    [
        [214, 214, 214],
        [192, 192, 192],
        [170, 170, 170],
        [142, 142, 142],
        [111, 111, 111],
        [74, 74, 74],
        [0, 0, 0],
        [252, 252, 84],
    ],
    dtype=jnp.uint8,
)

# BACKGROUND_COLOR = 0, 0, 0
# WALL_COLOR = 104, 72, 198
# GROUP3_COLOR = 187, 159, 71
# GROUP4_COLOR = 210, 164, 74
# GROUP5_COLOR = 236, 236, 236
# TILT_MODE_COLOR = 167, 26, 26
# BACKGROUND_COLOR_CYCLING = jnp.array([(74, 74, 74), (111, 111, 111), (142, 142, 142), (170, 170, 170),
#                                       (192, 192, 192), (214, 214, 214), (236, 236, 236), (72, 72, 0)])
# WALL_COLOR_CYCLING = jnp.array([(78, 50, 181), (51, 26, 163), (20, 0, 144), (188, 144, 252),
#                                 (169, 128, 240), (149, 111, 227), (127, 92, 213), (146, 70, 192)])
# GROUP3_COLOR_CYCLING = jnp.array([(210, 182, 86), (232, 204, 99), (252, 224, 112), (72, 44, 0),
#                                   (105, 77, 20), (134, 106, 38), (162, 134, 56), (160, 171, 79)])
# GROUP4_COLOR_CYCLING = jnp.array([(195, 144, 61), (236, 200, 96), (223, 183, 85), (144, 72, 17),
#                                   (124, 44, 0), (180, 122, 48), (162, 98, 33), (227, 151, 89)])
# GROUP5_COLOR_CYCLING = jnp.array([(214, 214, 214), (192, 192, 192), (170, 170, 170), (142, 142, 142),
#                                   (111, 111, 111), (74, 74, 74), (0, 0, 0), (252, 252, 84)])


# Pygame window dimensions
WINDOW_WIDTH = 160 * 3
WINDOW_HEIGHT = 210 * 3


# Outer objects (walls etc.) Positions/dimensions
BALL_START_X = jnp.array(149.0)
BALL_START_Y = jnp.array(129.0)
BALL_START_DIRECTION = jnp.array(0)

GAME_BOTTOM_Y = 191

TOP_WALL_LEFT_X_OFFSET = 0
TOP_WALL_TOP_Y_OFFSET = 16
RIGHT_WALL_LEFT_X_OFFSET = 152
BOTTOM_WALL_LEFT_X_OFFSET = 12
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

# Inner Objects Positions and Dimensions

VERTICAL_BAR_HEIGHT = 30
VERTICAL_BAR_WIDTH = 3

ROLLOVER_BAR_DISTANCE = 12  # Distance between the left and right rollover bar

BUMPER_WIDTH = 15
BUMPER_HEIGHT = 30

LEFT_COLUMN_X_OFFSET = 40
MIDDLE_COLUMN_X_OFFSET = 72
RIGHT_COLUMN_X_OFFSET = 104
TOP_ROW_Y_OFFSET = 49
MIDDLE_ROW_Y_OFFSET = 113
BOTTOM_ROW_Y_OFFSET = 177

MIDDLE_BAR_X = 72
MIDDLE_BAR_Y = 104
MIDDLE_BAR_WIDTH = 16
MIDDLE_BAR_HEIGHT = 8

# Flipper Parts Positions and Dimensions

# Flipper bounding boxes
# Position 0 - Downn (RAM state 0)
# Written as height x width
BOUNDING_BOX_1_2 = jnp.ones((1, 2)).astype(jnp.bool)
BOUNDING_BOX_2_3 = jnp.ones((2, 3)).astype(jnp.bool)
BOUNDING_BOX_2_4 = jnp.ones((2, 4)).astype(jnp.bool)
BOUDNING_BOX_1_5 = jnp.ones((1, 5)).astype(jnp.bool)
BOUNDING_BOX_1_4 = jnp.ones((1, 4)).astype(jnp.bool)
BOUNDING_BOX_2_5 = jnp.ones((2, 5)).astype(jnp.bool)
BOUNDING_BOX_2_6 = jnp.ones((2, 6)).astype(jnp.bool)

BOUNDING_BOX_1_10 = jnp.ones((1, 10)).astype(jnp.bool)
BOUNDING_BOX_2_13 = jnp.ones((2, 13)).astype(jnp.bool)

# Position 32 - Mid-Down (RAM state 32)
BOUNDING_BOX_1_6  = jnp.ones((1, 6)).astype(jnp.bool)
BOUNDING_BOX_1_7  = jnp.ones((1, 7)).astype(jnp.bool)
BOUNDING_BOX_1_9  = jnp.ones((1, 9)).astype(jnp.bool)

# Position 48 - Mid-Down (RAM state 48)
BOUNDING_BOX_1_1  = jnp.ones((1, 1)).astype(jnp.bool)
BOUNDING_BOX_1_3  = jnp.ones((1, 3)).astype(jnp.bool)

# Spinner Bounding Boxes

# Bounding boxes of the spinner if the joined (single larger) spinner part is on the top or bottom
SPINNER_TOP_BOTTOM_LARGE_HEIGHT = jnp.array(2)
SPINNER_TOP_BOTTOM_LARGE_WIDTH = jnp.array(3)
SPINNER_TOP_BOTTOM_SMALL_HEIGHT = jnp.array(1)
SPINNER_TOP_BOTTOM_SMALL_WIDTH = jnp.array(2)

# Bounding boxes of the spinner if the joined (single larger) spinner part is on the left or right
SPINNER_LEFT_RIGHT_LARGE_HEIGHT = jnp.array(1)
SPINNER_LEFT_RIGHT_LARGE_WIDTH = jnp.array(4)
SPINNER_LEFT_RIGHT_SMALL_HEIGHT = jnp.array(1)
SPINNER_LEFT_RIGHT_SMALL_WIDTH = jnp.array(2)


FLIPPER_OFFSETS_STATE_0_LEFT = jnp.array(
    [
        [64, 183],  
        [64, 184],  
        [64, 185],  
        [64, 186],  
        [67, 187],  
        [68, 187],  
        [69, 188],  
    ],
    dtype=jnp.int32,
)

FLIPPER_OFFSETS_STATE_0_RIGHT = jnp.array(
    [
        [94, 183],  
        [93, 184],  
        [92, 185],  
        [91, 186],  
        [89, 187],  
        [87, 187],  
        [85, 188],  
    ],
    dtype=jnp.int32,
)

# Offsets for the flipper in the first mid‑down position (RAM state 16).
FLIPPER_OFFSETS_STATE_16_LEFT = jnp.array(
    [
        [64, 184],  
        [64, 185],  
    ],
    dtype=jnp.int32,
)

FLIPPER_OFFSETS_STATE_16_RIGHT = jnp.array(
    [
        [86, 184],  
        [83, 185],  
    ],
    dtype=jnp.int32,
)

# Position 16 - Mid-Down (RAM state 16)
# Offsets for the flipper in the second mid‑angle position (RAM state 32).
FLIPPER_OFFSETS_STATE_32_LEFT = jnp.array(
    [
        [71, 181],  
        [67, 182],  
        [67, 182],  
    ],
    dtype=jnp.int32,
)

# Position 32 - Mid-Down (RAM state 32)
FLIPPER_OFFSETS_STATE_32_RIGHT = jnp.array(
    [
        [83, 181],  
        [86, 182],   
        [84, 182],  
    ],
    dtype=jnp.int32,
)

# Offsets for the flipper in the fully upright position (RAM state 48).
FLIPPER_OFFSETS_STATE_48_LEFT = jnp.array(
    [
        [74, 176],  
        [73, 177],  
    ],
    dtype=jnp.int32,
)

# Position 48 - Mid-Down (RAM state 48)
FLIPPER_OFFSETS_STATE_48_RIGHT = jnp.array(
    [
        [85, 176],  
        [84, 177],  
    ],
    dtype=jnp.int32,
)

# A convenience mapping from RAM state to the corresponding offset arrays.
#FLIPPER_OFFSETS = {
#    0: {
#        "left": FLIPPER_OFFSETS_STATE_0_LEFT,
#        "right": FLIPPER_OFFSETS_STATE_0_RIGHT,
#    },
#    16: {
#        "left": FLIPPER_OFFSETS_STATE_16_LEFT,
#        "right": FLIPPER_OFFSETS_STATE_16_RIGHT,
#    },
#    32: {
#        "left": FLIPPER_OFFSETS_STATE_32_LEFT,
#        "right": FLIPPER_OFFSETS_STATE_32_RIGHT,
#    },
#    48: {
#        "left": FLIPPER_OFFSETS_STATE_48_LEFT,
#        "right": FLIPPER_OFFSETS_STATE_48_RIGHT,
#    },
#}

@chex.dataclass
class BallMovement:
    old_ball_x: chex.Array
    old_ball_y: chex.Array
    new_ball_x: chex.Array
    new_ball_y: chex.Array


@chex.dataclass
class SceneObject:
    hit_box_width: chex.Array
    hit_box_height: chex.Array
    hit_box_x_offset: chex.Array
    hit_box_y_offset: chex.Array
    reflecting: chex.Array  # 0: no reflection, 1: reflection
    score_type: (
        chex.Array
    )  # 0: no score, 1: Bumper, 2: Spinner, 3: Left Rollover, 4: Atari Rollover, 5: Special Lit Up Target,
    # 6: Left Lit Up Target, 7:Middle Lit Up Target, 8: Right Lit Up Target


# Instantiate a SceneObject like this:
INVISIBLE_BLOCK_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(2),  # type: ignore
    hit_box_width=jnp.array(INNER_WALL_THICKNESS),  # type: ignore
    hit_box_x_offset=jnp.array(INVISIBLE_BLOCK_LEFT_X_OFFSET),  # type: ignore
    hit_box_y_offset=jnp.array(INVISIBLE_BLOCK_TOP_Y_OFFSET),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(0),  # type: ignore
)

# Wall Scene Objects

TOP_WALL_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(OUTER_WALL_THICKNESS),  # type: ignore
    hit_box_width=jnp.array(160),  # type: ignore
    hit_box_x_offset=jnp.array(TOP_WALL_LEFT_X_OFFSET),  # type: ignore
    hit_box_y_offset=jnp.array(TOP_WALL_TOP_Y_OFFSET),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(0),  # type: ignore
)
BOTTOM_WALL_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(OUTER_WALL_THICKNESS),  # type: ignore
    hit_box_width=jnp.array(160),  # type: ignore
    hit_box_x_offset=jnp.array(BOTTOM_WALL_LEFT_X_OFFSET),  # type: ignore
    hit_box_y_offset=jnp.array(BOTTOM_WALL_TOP_Y_OFFSET),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(0),  # type: ignore
)
LEFT_WALL_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(176),  # type: ignore
    hit_box_width=jnp.array(OUTER_WALL_THICKNESS),  # type: ignore
    hit_box_x_offset=jnp.array(TOP_WALL_LEFT_X_OFFSET),  # type: ignore
    hit_box_y_offset=jnp.array(TOP_WALL_TOP_Y_OFFSET),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(0),  # type: ignore
)
RIGHT_WALL_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(176),  # type: ignore
    hit_box_width=jnp.array(OUTER_WALL_THICKNESS),  # type: ignore
    hit_box_x_offset=jnp.array(RIGHT_WALL_LEFT_X_OFFSET),  # type: ignore
    hit_box_y_offset=jnp.array(TOP_WALL_TOP_Y_OFFSET),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(0),  # type: ignore
)

LEFT_INNER_WALL_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(135),  # type: ignore
    hit_box_width=jnp.array(INNER_WALL_THICKNESS),  # type: ignore
    hit_box_x_offset=jnp.array(LEFT_INNER_WALL_TOP_X_OFFSET),  # type: ignore
    hit_box_y_offset=jnp.array(INNER_WALL_TOP_Y_OFFSET),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(0),  # type: ignore
)

RIGHT_INNER_WALL_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(135),  # type: ignore
    hit_box_width=jnp.array(INNER_WALL_THICKNESS),  # type: ignore
    hit_box_x_offset=jnp.array(RIGHT_INNER_WALL_TOP_X_OFFSET),  # type: ignore
    hit_box_y_offset=jnp.array(INNER_WALL_TOP_Y_OFFSET),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(0),  # type: ignore
)

# Steps (Stairway left and right) Scene Objects

LEFT_QUADRUPLE_STEP_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(4 * STEP_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(STEP_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(LEFT_QUADRUPLE_STEP_X_OFFSET),  # type: ignore
    hit_box_y_offset=jnp.array(QUADRUPLE_STEP_Y_OFFSET),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(0),  # type: ignore
)
RIGHT_QUADRUPLE_STEP_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(4 * STEP_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(STEP_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(RIGHT_QUADRUPLE_STEP_X_OFFSET),  # type: ignore
    hit_box_y_offset=jnp.array(QUADRUPLE_STEP_Y_OFFSET),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(0),  # type: ignore
)
LEFT_TRIPLE_STEP_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(3 * STEP_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(STEP_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(LEFT_TRIPLE_STEP_X_OFFSET),  # type: ignore
    hit_box_y_offset=jnp.array(TRIPLE_STEP_Y_OFFSET),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(0),  # type: ignore
)
RIGHT_TRIPLE_STEP_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(3 * STEP_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(STEP_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(RIGHT_TRIPLE_STEP_X_OFFSET),  # type: ignore
    hit_box_y_offset=jnp.array(TRIPLE_STEP_Y_OFFSET),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(0),  # type: ignore
)
LEFT_DOUBLE_STEP_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(2 * STEP_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(STEP_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(LEFT_DOUBLE_STEP_X_OFFSET),  # type: ignore
    hit_box_y_offset=jnp.array(DOUBLE_STEP_Y_OFFSET),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(0),  # type: ignore
)
RIGHT_DOUBLE_STEP_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(2 * STEP_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(STEP_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(RIGHT_DOUBLE_STEP_X_OFFSET),  # type: ignore
    hit_box_y_offset=jnp.array(DOUBLE_STEP_Y_OFFSET),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(0),  # type: ignore
)
LEFT_SINGLE_STEP_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(1 * STEP_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(STEP_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(LEFT_SINGLE_STEP_X_OFFSET),  # type: ignore
    hit_box_y_offset=jnp.array(SINGLE_STEP_Y_OFFSET),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(0),  # type: ignore
)
RIGHT_SINGLE_STEP_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(1 * STEP_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(STEP_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(RIGHT_SINGLE_STEP_X_OFFSET),  # type: ignore
    hit_box_y_offset=jnp.array(SINGLE_STEP_Y_OFFSET),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(0),  # type: ignore
)
TOP_LEFT_STEP_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(1 * STEP_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(1 * STEP_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(8),  # type: ignore
    hit_box_y_offset=jnp.array(24),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(0),  # type: ignore
)
TOP_RIGHT_STEP_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(1 * STEP_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(1 * STEP_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(148),  # type: ignore
    hit_box_y_offset=jnp.array(24),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(0),  # type: ignore
)

# Rollover Bars Scene Objects

LEFT_ROLLOVER_LEFT_BAR_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(VERTICAL_BAR_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(VERTICAL_BAR_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(LEFT_COLUMN_X_OFFSET),  # type: ignore
    hit_box_y_offset=jnp.array(TOP_ROW_Y_OFFSET),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(0),  # type: ignore
)

LEFT_ROLLOVER_RIGHT_BAR_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(VERTICAL_BAR_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(VERTICAL_BAR_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(LEFT_COLUMN_X_OFFSET + ROLLOVER_BAR_DISTANCE),  # type: ignore
    hit_box_y_offset=jnp.array(TOP_ROW_Y_OFFSET),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(0),  # type: ignore
)

ATARI_ROLLOVER_LEFT_BAR_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(VERTICAL_BAR_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(VERTICAL_BAR_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(RIGHT_COLUMN_X_OFFSET),  # type: ignore
    hit_box_y_offset=jnp.array(TOP_ROW_Y_OFFSET),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(0),  # type: ignore
)

ATARI_ROLLOVER_RIGHT_BAR_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(VERTICAL_BAR_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(VERTICAL_BAR_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(RIGHT_COLUMN_X_OFFSET + ROLLOVER_BAR_DISTANCE),  # type: ignore
    hit_box_y_offset=jnp.array(TOP_ROW_Y_OFFSET),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(0),  # type: ignore
)

# Bumper Scene Objects

TOP_BUMPER_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(BUMPER_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(BUMPER_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(MIDDLE_COLUMN_X_OFFSET),  # type: ignore
    hit_box_y_offset=jnp.array(TOP_ROW_Y_OFFSET),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(1),  # type: ignore
)

LEFT_BUMPER_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(BUMPER_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(BUMPER_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(LEFT_COLUMN_X_OFFSET),  # type: ignore
    hit_box_y_offset=jnp.array(MIDDLE_ROW_Y_OFFSET),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(1),  # type: ignore
)
RIGHT_BUMPER_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(BUMPER_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(BUMPER_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(RIGHT_COLUMN_X_OFFSET),  # type: ignore
    hit_box_y_offset=jnp.array(MIDDLE_ROW_Y_OFFSET),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(1),  # type: ignore
)

# Left Spinner Scene Objects

# Left Spinner Bottom Position

LEFT_SPINNER_BOTTOM_POSITION_JOINED_PART_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(SPINNER_TOP_BOTTOM_LARGE_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(SPINNER_TOP_BOTTOM_LARGE_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(32),  # type: ignore
    hit_box_y_offset=jnp.array(100),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(2),  # type: ignore
)

LEFT_SPINNER_BOTTOM_POSITION_LEFT_PART_1_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(SPINNER_TOP_BOTTOM_SMALL_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(SPINNER_TOP_BOTTOM_SMALL_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(30),  # type: ignore
    hit_box_y_offset=jnp.array(94),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(2),  # type: ignore
)

LEFT_SPINNER_BOTTOM_POSITION_LEFT_PART_2_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(SPINNER_TOP_BOTTOM_SMALL_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(SPINNER_TOP_BOTTOM_SMALL_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(31),  # type: ignore
    hit_box_y_offset=jnp.array(92),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(2),  # type: ignore
)

LEFT_SPINNER_BOTTOM_POSITION_RIGHT_PART_1_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(SPINNER_TOP_BOTTOM_SMALL_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(SPINNER_TOP_BOTTOM_SMALL_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(35),  # type: ignore
    hit_box_y_offset=jnp.array(94),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(2),  # type: ignore
)

LEFT_SPINNER_BOTTOM_POSITION_RIGHT_PART_2_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(SPINNER_TOP_BOTTOM_SMALL_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(SPINNER_TOP_BOTTOM_SMALL_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(34),  # type: ignore
    hit_box_y_offset=jnp.array(92),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(2),  # type: ignore
)

# Left Spinner Top Position
LEFT_SPINNER_TOP_POSITION_JOINED_PART_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(SPINNER_TOP_BOTTOM_LARGE_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(SPINNER_TOP_BOTTOM_LARGE_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(32),  # type: ignore
    hit_box_y_offset=jnp.array(90),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(2),  # type: ignore
)
LEFT_SPINNER_TOP_POSITION_LEFT_PART_1_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(SPINNER_TOP_BOTTOM_SMALL_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(SPINNER_TOP_BOTTOM_SMALL_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(30),  # type: ignore
    hit_box_y_offset=jnp.array(96),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(2),  # type: ignore
)

LEFT_SPINNER_TOP_POSITION_LEFT_PART_2_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(SPINNER_TOP_BOTTOM_SMALL_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(SPINNER_TOP_BOTTOM_SMALL_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(31),  # type: ignore
    hit_box_y_offset=jnp.array(98),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(2),  # type: ignore
)

LEFT_SPINNER_TOP_POSITION_RIGHT_PART_1_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(SPINNER_TOP_BOTTOM_SMALL_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(SPINNER_TOP_BOTTOM_SMALL_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(35),  # type: ignore
    hit_box_y_offset=jnp.array(96),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(2),  # type: ignore
)

LEFT_SPINNER_TOP_POSITION_RIGHT_PART_2_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(SPINNER_TOP_BOTTOM_SMALL_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(SPINNER_TOP_BOTTOM_SMALL_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(34),  # type: ignore
    hit_box_y_offset=jnp.array(98),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(2),  # type: ignore
)

# Left Spinner Left Position
LEFT_SPINNER_LEFT_POSITION_JOINED_PART_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(SPINNER_LEFT_RIGHT_LARGE_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(SPINNER_LEFT_RIGHT_LARGE_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(30),  # type: ignore
    hit_box_y_offset=jnp.array(94),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(2),  # type: ignore
)

LEFT_SPINNER_LEFT_POSITION_LEFT_PART_1_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(SPINNER_LEFT_RIGHT_SMALL_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(SPINNER_LEFT_RIGHT_SMALL_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(33),  # type: ignore
    hit_box_y_offset=jnp.array(90),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(2),  # type: ignore
)

LEFT_SPINNER_LEFT_POSITION_LEFT_PART_2_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(SPINNER_LEFT_RIGHT_SMALL_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(SPINNER_LEFT_RIGHT_SMALL_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(34),  # type: ignore
    hit_box_y_offset=jnp.array(92),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(2),  # type: ignore
)

LEFT_SPINNER_LEFT_POSITION_RIGHT_PART_1_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(SPINNER_LEFT_RIGHT_SMALL_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(SPINNER_LEFT_RIGHT_SMALL_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(33),  # type: ignore
    hit_box_y_offset=jnp.array(100),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(2),  # type: ignore
)

LEFT_SPINNER_LEFT_POSITION_RIGHT_PART_2_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(SPINNER_LEFT_RIGHT_SMALL_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(SPINNER_LEFT_RIGHT_SMALL_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(34),  # type: ignore
    hit_box_y_offset=jnp.array(98),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(2),  # type: ignore
)

# Left Spinner Right Position
LEFT_SPINNER_RIGHT_POSITION_JOINED_PART_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(SPINNER_LEFT_RIGHT_LARGE_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(SPINNER_LEFT_RIGHT_LARGE_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(35),  # type: ignore
    hit_box_y_offset=jnp.array(94),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(2),  # type: ignore
)

LEFT_SPINNER_RIGHT_POSITION_LEFT_PART_1_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(SPINNER_LEFT_RIGHT_SMALL_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(SPINNER_LEFT_RIGHT_SMALL_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(32),  # type: ignore
    hit_box_y_offset=jnp.array(100),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(2),  # type: ignore
)

LEFT_SPINNER_RIGHT_POSITION_LEFT_PART_2_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(SPINNER_LEFT_RIGHT_SMALL_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(SPINNER_LEFT_RIGHT_SMALL_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(31),  # type: ignore
    hit_box_y_offset=jnp.array(98),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(2),  # type: ignore
)

LEFT_SPINNER_RIGHT_POSITION_RIGHT_PART_1_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(SPINNER_LEFT_RIGHT_SMALL_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(SPINNER_LEFT_RIGHT_SMALL_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(32),  # type: ignore
    hit_box_y_offset=jnp.array(90),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(2),  # type: ignore
)

LEFT_SPINNER_RIGHT_POSITION_RIGHT_PART_2_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(SPINNER_LEFT_RIGHT_SMALL_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(SPINNER_LEFT_RIGHT_SMALL_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(31),  # type: ignore
    hit_box_y_offset=jnp.array(92),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(2),  # type: ignore
)

# Right Spinner Scene Objects

# Right Spinner Bottom Position
RIGHT_SPINNER_BOTTOM_POSITION_JOINED_PART_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(SPINNER_TOP_BOTTOM_LARGE_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(SPINNER_TOP_BOTTOM_LARGE_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(128),  # type: ignore
    hit_box_y_offset=jnp.array(100),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(2),  # type: ignore
)

RIGHT_SPINNER_BOTTOM_POSITION_LEFT_PART_1_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(SPINNER_TOP_BOTTOM_SMALL_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(SPINNER_TOP_BOTTOM_SMALL_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(126),  # type: ignore
    hit_box_y_offset=jnp.array(94),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(2),  # type: ignore
)

RIGHT_SPINNER_BOTTOM_POSITION_LEFT_PART_2_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(SPINNER_TOP_BOTTOM_SMALL_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(SPINNER_TOP_BOTTOM_SMALL_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(127),  # type: ignore
    hit_box_y_offset=jnp.array(92),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(2),  # type: ignore
)

RIGHT_SPINNER_BOTTOM_POSITION_RIGHT_PART_1_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(SPINNER_TOP_BOTTOM_SMALL_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(SPINNER_TOP_BOTTOM_SMALL_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(131),  # type: ignore
    hit_box_y_offset=jnp.array(94),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(2),  # type: ignore
)

RIGHT_SPINNER_BOTTOM_POSITION_RIGHT_PART_2_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(SPINNER_TOP_BOTTOM_SMALL_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(SPINNER_TOP_BOTTOM_SMALL_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(130),  # type: ignore
    hit_box_y_offset=jnp.array(92),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(2),  # type: ignore
)

# Right Spinner Top Position
RIGHT_SPINNER_TOP_POSITION_JOINED_PART_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(SPINNER_TOP_BOTTOM_LARGE_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(SPINNER_TOP_BOTTOM_LARGE_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(128),  # type: ignore
    hit_box_y_offset=jnp.array(90),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(2),  # type: ignore
)

RIGHT_SPINNER_TOP_POSITION_LEFT_PART_1_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(SPINNER_TOP_BOTTOM_SMALL_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(SPINNER_TOP_BOTTOM_SMALL_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(131),  # type: ignore
    hit_box_y_offset=jnp.array(96),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(2),  # type: ignore
)

RIGHT_SPINNER_TOP_POSITION_LEFT_PART_2_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(SPINNER_TOP_BOTTOM_SMALL_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(SPINNER_TOP_BOTTOM_SMALL_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(130),  # type: ignore
    hit_box_y_offset=jnp.array(98),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(2),  # type: ignore
)

RIGHT_SPINNER_TOP_POSITION_RIGHT_PART_1_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(SPINNER_TOP_BOTTOM_SMALL_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(SPINNER_TOP_BOTTOM_SMALL_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(126),  # type: ignore
    hit_box_y_offset=jnp.array(96),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(2),  # type: ignore
)

RIGHT_SPINNER_TOP_POSITION_RIGHT_PART_2_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(SPINNER_TOP_BOTTOM_SMALL_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(SPINNER_TOP_BOTTOM_SMALL_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(127),  # type: ignore
    hit_box_y_offset=jnp.array(98),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(2),  # type: ignore
)

# Right Spinner Left Position
RIGHT_SPINNER_LEFT_POSITION_JOINED_PART_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(SPINNER_LEFT_RIGHT_LARGE_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(SPINNER_LEFT_RIGHT_LARGE_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(126),  # type: ignore
    hit_box_y_offset=jnp.array(94),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(2),  # type: ignore
)

RIGHT_SPINNER_LEFT_POSITION_LEFT_PART_1_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(SPINNER_LEFT_RIGHT_SMALL_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(SPINNER_LEFT_RIGHT_SMALL_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(129),  # type: ignore
    hit_box_y_offset=jnp.array(90),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(2),  # type: ignore
)

RIGHT_SPINNER_LEFT_POSITION_LEFT_PART_2_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(SPINNER_LEFT_RIGHT_SMALL_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(SPINNER_LEFT_RIGHT_SMALL_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(130),  # type: ignore
    hit_box_y_offset=jnp.array(92),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(2),  # type: ignore
)

RIGHT_SPINNER_LEFT_POSITION_RIGHT_PART_1_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(SPINNER_LEFT_RIGHT_SMALL_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(SPINNER_LEFT_RIGHT_SMALL_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(129),  # type: ignore
    hit_box_y_offset=jnp.array(100),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(2),  # type: ignore
)

RIGHT_SPINNER_LEFT_POSITION_RIGHT_PART_2_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(SPINNER_LEFT_RIGHT_SMALL_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(SPINNER_LEFT_RIGHT_SMALL_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(130),  # type: ignore
    hit_box_y_offset=jnp.array(98),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(2),  # type: ignore
)

# Right Spinner Right Position
RIGHT_SPINNER_RIGHT_POSITION_JOINED_PART_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(SPINNER_LEFT_RIGHT_LARGE_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(SPINNER_LEFT_RIGHT_LARGE_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(131),  # type: ignore
    hit_box_y_offset=jnp.array(94),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(2),  # type: ignore
)

RIGHT_SPINNER_RIGHT_POSITION_LEFT_PART_1_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(SPINNER_LEFT_RIGHT_SMALL_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(SPINNER_LEFT_RIGHT_SMALL_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(128),  # type: ignore
    hit_box_y_offset=jnp.array(100),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(2),  # type: ignore
)

RIGHT_SPINNER_RIGHT_POSITION_LEFT_PART_2_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(SPINNER_LEFT_RIGHT_SMALL_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(SPINNER_LEFT_RIGHT_SMALL_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(127),  # type: ignore
    hit_box_y_offset=jnp.array(98),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(2),  # type: ignore
)

RIGHT_SPINNER_RIGHT_POSITION_RIGHT_PART_1_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(SPINNER_LEFT_RIGHT_SMALL_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(SPINNER_LEFT_RIGHT_SMALL_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(128),  # type: ignore
    hit_box_y_offset=jnp.array(90),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(2),  # type: ignore
)

RIGHT_SPINNER_RIGHT_POSITION_RIGHT_PART_2_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(SPINNER_LEFT_RIGHT_SMALL_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(SPINNER_LEFT_RIGHT_SMALL_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(127),  # type: ignore
    hit_box_y_offset=jnp.array(92),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(2),  # type: ignore
)

DIAMOND_VERTICAL_RECTANGLE_BOUNDING_BOX_HEIGHT = jnp.array(10)
DIAMOND_VERTICAL_RECTANGLE_BOUNDING_BOX_WIDTH = jnp.array(3)
DIAMOND_HORIZONTAL_RECTANGLE_BOUNDING_BOX_HEIGHT = jnp.array(6)
DIAMOND_HORIZONTAL_RECTANGLE_BOUNDING_BOX_WIDTH = jnp.array(5)
DIAMOND_SMALL_RECTANGLE_BOUNDING_BOX_HEIGHT = jnp.array(2)
DIAMOND_SMALL_RECTANGLE_BOUNDING_BOX_WIDTH = jnp.array(1)

LEFT_LIT_UP_TARGET_LARGE_VERTICAL_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(DIAMOND_VERTICAL_RECTANGLE_BOUNDING_BOX_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(DIAMOND_VERTICAL_RECTANGLE_BOUNDING_BOX_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(62),  # type: ignore
    hit_box_y_offset=jnp.array(26),  # type: ignore
    reflecting=jnp.array(0),  # type: ignore
    score_type=jnp.array(6),  # type: ignore
)

LEFT_LIT_UP_TARGET_LARGE_HORIZONTAL_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(DIAMOND_HORIZONTAL_RECTANGLE_BOUNDING_BOX_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(DIAMOND_HORIZONTAL_RECTANGLE_BOUNDING_BOX_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(60),  # type: ignore
    hit_box_y_offset=jnp.array(30),  # type: ignore
    reflecting=jnp.array(0),  # type: ignore
    score_type=jnp.array(6),  # type: ignore
)

MIDDLE_LIT_UP_TARGET_LARGE_VERTICAL_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(DIAMOND_VERTICAL_RECTANGLE_BOUNDING_BOX_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(DIAMOND_VERTICAL_RECTANGLE_BOUNDING_BOX_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(78),  # type: ignore
    hit_box_y_offset=jnp.array(26),  # type: ignore
    reflecting=jnp.array(0),  # type: ignore
    score_type=jnp.array(7),  # type: ignore
)

MIDDLE_LIT_UP_TARGET_LARGE_HORIZONTAL_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(DIAMOND_HORIZONTAL_RECTANGLE_BOUNDING_BOX_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(DIAMOND_HORIZONTAL_RECTANGLE_BOUNDING_BOX_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(77),  # type: ignore
    hit_box_y_offset=jnp.array(28),  # type: ignore
    reflecting=jnp.array(0),  # type: ignore
    score_type=jnp.array(7),  # type: ignore
)

RIGHT_LIT_UP_TARGET_LARGE_VERTICAL_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(DIAMOND_VERTICAL_RECTANGLE_BOUNDING_BOX_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(DIAMOND_VERTICAL_RECTANGLE_BOUNDING_BOX_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(94),  # type: ignore
    hit_box_y_offset=jnp.array(26),  # type: ignore
    reflecting=jnp.array(0),  # type: ignore
    score_type=jnp.array(8),  # type: ignore
)

RIGHT_LIT_UP_TARGET_LARGE_HORIZONTAL_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(DIAMOND_HORIZONTAL_RECTANGLE_BOUNDING_BOX_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(DIAMOND_HORIZONTAL_RECTANGLE_BOUNDING_BOX_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(93),  # type: ignore
    hit_box_y_offset=jnp.array(28),  # type: ignore
    reflecting=jnp.array(0),  # type: ignore
    score_type=jnp.array(8),  # type: ignore
)

SPECIAL_LIT_UP_TARGET_LARGE_VERTICAL_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(DIAMOND_VERTICAL_RECTANGLE_BOUNDING_BOX_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(DIAMOND_VERTICAL_RECTANGLE_BOUNDING_BOX_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(79),  # type: ignore
    hit_box_y_offset=jnp.array(122),  # type: ignore
    reflecting=jnp.array(0),  # type: ignore
    score_type=jnp.array(5),  # type: ignore
)

SPECIAL_LIT_UP_TARGET_LARGE_HORIZONTAL_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(DIAMOND_HORIZONTAL_RECTANGLE_BOUNDING_BOX_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(DIAMOND_HORIZONTAL_RECTANGLE_BOUNDING_BOX_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(78),  # type: ignore
    hit_box_y_offset=jnp.array(124),  # type: ignore
    reflecting=jnp.array(0),  # type: ignore
    score_type=jnp.array(5),  # type: ignore
)

LEFT_ROLLOVER_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(12),  # type: ignore
    hit_box_width=jnp.array(6),  # type: ignore
    hit_box_x_offset=jnp.array(44),  # type: ignore
    hit_box_y_offset=jnp.array(58),  # type: ignore
    reflecting=jnp.array(0),  # type: ignore
    score_type=jnp.array(3),  # type: ignore
)

ATARI_ROLLOVER_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(12),  # type: ignore
    hit_box_width=jnp.array(6),  # type: ignore
    hit_box_x_offset=jnp.array(108),  # type: ignore
    hit_box_y_offset=jnp.array(58),  # type: ignore
    reflecting=jnp.array(0),  # type: ignore
    score_type=jnp.array(4),  # type: ignore
)

# Middle Bar Scene Object

MIDDLE_BAR_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(MIDDLE_BAR_HEIGHT),  # type: ignore
    hit_box_width=jnp.array(MIDDLE_BAR_WIDTH),  # type: ignore
    hit_box_x_offset=jnp.array(MIDDLE_BAR_X),  # type: ignore
    hit_box_y_offset=jnp.array(MIDDLE_BAR_Y),  # type: ignore
    reflecting=jnp.array(1),  # type: ignore
    score_type=jnp.array(0),  # type: ignore
)

# Left Flipper SceneObjects for different states
LEFT_FLIPPER_STATE_0_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(6),  
    hit_box_width=jnp.array(6),   
    hit_box_x_offset=jnp.array(64),
    hit_box_y_offset=jnp.array(183),
    reflecting=jnp.array(1),      
    score_type=jnp.array(0), #no score 
)

LEFT_FLIPPER_STATE_16_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(2),  
    hit_box_width=jnp.array(1),   
    hit_box_x_offset=jnp.array(64),
    hit_box_y_offset=jnp.array(184),
    reflecting=jnp.array(1),
    score_type=jnp.array(0),
)

LEFT_FLIPPER_STATE_32_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(2),  
    hit_box_width=jnp.array(5),   
    hit_box_x_offset=jnp.array(67),
    hit_box_y_offset=jnp.array(181),
    reflecting=jnp.array(1),
    score_type=jnp.array(0),
)

LEFT_FLIPPER_STATE_48_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(2),  
    hit_box_width=jnp.array(2),   
    hit_box_x_offset=jnp.array(73),
    hit_box_y_offset=jnp.array(176),
    reflecting=jnp.array(1),
    score_type=jnp.array(0),
)

# Right Flipper SceneObjects for different states
RIGHT_FLIPPER_STATE_0_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(6),  
    hit_box_width=jnp.array(10),  
    hit_box_x_offset=jnp.array(85),
    hit_box_y_offset=jnp.array(183),
    reflecting=jnp.array(1),
    score_type=jnp.array(0),
)

RIGHT_FLIPPER_STATE_16_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(2),  
    hit_box_width=jnp.array(4),   
    hit_box_x_offset=jnp.array(83),
    hit_box_y_offset=jnp.array(184),
    reflecting=jnp.array(1),
    score_type=jnp.array(0),
)

RIGHT_FLIPPER_STATE_32_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(2),  
    hit_box_width=jnp.array(4),   
    hit_box_x_offset=jnp.array(83),
    hit_box_y_offset=jnp.array(181),
    reflecting=jnp.array(1),
    score_type=jnp.array(0),
)

RIGHT_FLIPPER_STATE_48_SCENE_OBJECT = SceneObject(
    hit_box_height=jnp.array(2),  
    hit_box_width=jnp.array(2),   
    hit_box_x_offset=jnp.array(84),
    hit_box_y_offset=jnp.array(176),
    reflecting=jnp.array(1),
    score_type=jnp.array(0),
)

# If you edit these make sure that the indexing over it is still correct
# we do this in: _check_obstacle_hits
ALL_SCENE_OBJECTS_LIST = [
    # Non-reflective scene objects
    LEFT_LIT_UP_TARGET_LARGE_VERTICAL_SCENE_OBJECT,  # 0
    LEFT_LIT_UP_TARGET_LARGE_HORIZONTAL_SCENE_OBJECT,  # 1
    MIDDLE_LIT_UP_TARGET_LARGE_VERTICAL_SCENE_OBJECT,  # 2
    MIDDLE_LIT_UP_TARGET_LARGE_HORIZONTAL_SCENE_OBJECT,  # 3
    RIGHT_LIT_UP_TARGET_LARGE_VERTICAL_SCENE_OBJECT,  # 4
    RIGHT_LIT_UP_TARGET_LARGE_HORIZONTAL_SCENE_OBJECT,  # 5
    SPECIAL_LIT_UP_TARGET_LARGE_VERTICAL_SCENE_OBJECT,  # 6
    SPECIAL_LIT_UP_TARGET_LARGE_HORIZONTAL_SCENE_OBJECT,  # 7
    LEFT_ROLLOVER_SCENE_OBJECT,  # 8
    ATARI_ROLLOVER_SCENE_OBJECT,  # 9
    # Reflective scene objects
    TOP_WALL_SCENE_OBJECT,  # 10
    BOTTOM_WALL_SCENE_OBJECT,  # 11
    LEFT_WALL_SCENE_OBJECT,  # 12
    RIGHT_WALL_SCENE_OBJECT,  # 13
    LEFT_INNER_WALL_SCENE_OBJECT,  # 14
    RIGHT_INNER_WALL_SCENE_OBJECT,  # 15
    LEFT_QUADRUPLE_STEP_SCENE_OBJECT,  # 16
    RIGHT_QUADRUPLE_STEP_SCENE_OBJECT,  # 17
    LEFT_TRIPLE_STEP_SCENE_OBJECT,  # 18
    RIGHT_TRIPLE_STEP_SCENE_OBJECT,  # 19
    LEFT_DOUBLE_STEP_SCENE_OBJECT,  # 20
    RIGHT_DOUBLE_STEP_SCENE_OBJECT,  # 21
    LEFT_SINGLE_STEP_SCENE_OBJECT,  # 22
    RIGHT_SINGLE_STEP_SCENE_OBJECT,  # 23
    TOP_LEFT_STEP_SCENE_OBJECT,  # 24
    TOP_RIGHT_STEP_SCENE_OBJECT,  # 25
    LEFT_ROLLOVER_LEFT_BAR_SCENE_OBJECT,  # 26
    LEFT_ROLLOVER_RIGHT_BAR_SCENE_OBJECT,  # 27
    ATARI_ROLLOVER_LEFT_BAR_SCENE_OBJECT,  # 28
    ATARI_ROLLOVER_RIGHT_BAR_SCENE_OBJECT,  # 29
    TOP_BUMPER_SCENE_OBJECT,  # 30
    LEFT_BUMPER_SCENE_OBJECT,  # 31
    RIGHT_BUMPER_SCENE_OBJECT,  # 32
    LEFT_SPINNER_BOTTOM_POSITION_JOINED_PART_SCENE_OBJECT,  # 33
    LEFT_SPINNER_BOTTOM_POSITION_LEFT_PART_1_SCENE_OBJECT,  # 34
    LEFT_SPINNER_BOTTOM_POSITION_LEFT_PART_2_SCENE_OBJECT,  # 35
    LEFT_SPINNER_BOTTOM_POSITION_RIGHT_PART_1_SCENE_OBJECT,  # 36
    LEFT_SPINNER_BOTTOM_POSITION_RIGHT_PART_2_SCENE_OBJECT,  # 37
    LEFT_SPINNER_RIGHT_POSITION_JOINED_PART_SCENE_OBJECT,  # 38
    LEFT_SPINNER_RIGHT_POSITION_LEFT_PART_1_SCENE_OBJECT,  # 39
    LEFT_SPINNER_RIGHT_POSITION_LEFT_PART_2_SCENE_OBJECT,  # 40
    LEFT_SPINNER_RIGHT_POSITION_RIGHT_PART_1_SCENE_OBJECT,  # 41
    LEFT_SPINNER_RIGHT_POSITION_RIGHT_PART_2_SCENE_OBJECT,  # 42
    LEFT_SPINNER_TOP_POSITION_JOINED_PART_SCENE_OBJECT,  # 43
    LEFT_SPINNER_TOP_POSITION_LEFT_PART_1_SCENE_OBJECT,  # 44
    LEFT_SPINNER_TOP_POSITION_LEFT_PART_2_SCENE_OBJECT,  # 45
    LEFT_SPINNER_TOP_POSITION_RIGHT_PART_1_SCENE_OBJECT,  # 46
    LEFT_SPINNER_TOP_POSITION_RIGHT_PART_2_SCENE_OBJECT,  # 47
    LEFT_SPINNER_LEFT_POSITION_JOINED_PART_SCENE_OBJECT,  # 48
    LEFT_SPINNER_LEFT_POSITION_LEFT_PART_1_SCENE_OBJECT,  # 49
    LEFT_SPINNER_LEFT_POSITION_LEFT_PART_2_SCENE_OBJECT,  # 50
    LEFT_SPINNER_LEFT_POSITION_RIGHT_PART_1_SCENE_OBJECT,  # 51
    LEFT_SPINNER_LEFT_POSITION_RIGHT_PART_2_SCENE_OBJECT,  # 52
    RIGHT_SPINNER_BOTTOM_POSITION_JOINED_PART_SCENE_OBJECT,  # 53
    RIGHT_SPINNER_BOTTOM_POSITION_LEFT_PART_1_SCENE_OBJECT,  # 54
    RIGHT_SPINNER_BOTTOM_POSITION_LEFT_PART_2_SCENE_OBJECT,  # 55
    RIGHT_SPINNER_BOTTOM_POSITION_RIGHT_PART_1_SCENE_OBJECT,  # 56
    RIGHT_SPINNER_BOTTOM_POSITION_RIGHT_PART_2_SCENE_OBJECT,  # 57
    RIGHT_SPINNER_RIGHT_POSITION_JOINED_PART_SCENE_OBJECT,  # 58
    RIGHT_SPINNER_RIGHT_POSITION_LEFT_PART_1_SCENE_OBJECT,  # 59
    RIGHT_SPINNER_RIGHT_POSITION_LEFT_PART_2_SCENE_OBJECT,  # 60
    RIGHT_SPINNER_RIGHT_POSITION_RIGHT_PART_1_SCENE_OBJECT,  # 61
    RIGHT_SPINNER_RIGHT_POSITION_RIGHT_PART_2_SCENE_OBJECT,  # 62
    RIGHT_SPINNER_TOP_POSITION_JOINED_PART_SCENE_OBJECT,  # 63
    RIGHT_SPINNER_TOP_POSITION_LEFT_PART_1_SCENE_OBJECT,  # 64
    RIGHT_SPINNER_TOP_POSITION_LEFT_PART_2_SCENE_OBJECT,  # 65
    RIGHT_SPINNER_TOP_POSITION_RIGHT_PART_1_SCENE_OBJECT,  # 66
    RIGHT_SPINNER_TOP_POSITION_RIGHT_PART_2_SCENE_OBJECT,  # 67
    RIGHT_SPINNER_LEFT_POSITION_JOINED_PART_SCENE_OBJECT,  # 68
    RIGHT_SPINNER_LEFT_POSITION_LEFT_PART_1_SCENE_OBJECT,  # 69
    RIGHT_SPINNER_LEFT_POSITION_LEFT_PART_2_SCENE_OBJECT,  # 70
    RIGHT_SPINNER_LEFT_POSITION_RIGHT_PART_1_SCENE_OBJECT,  # 71
    RIGHT_SPINNER_LEFT_POSITION_RIGHT_PART_2_SCENE_OBJECT,  # 72
    MIDDLE_BAR_SCENE_OBJECT,  # 73
    LEFT_FLIPPER_STATE_0_SCENE_OBJECT,  # 74
    LEFT_FLIPPER_STATE_16_SCENE_OBJECT,  # 75
    LEFT_FLIPPER_STATE_32_SCENE_OBJECT,  # 76
    LEFT_FLIPPER_STATE_48_SCENE_OBJECT,  # 77
    RIGHT_FLIPPER_STATE_0_SCENE_OBJECT,  # 78
    RIGHT_FLIPPER_STATE_16_SCENE_OBJECT,  # 79
    RIGHT_FLIPPER_STATE_32_SCENE_OBJECT,  # 80
    RIGHT_FLIPPER_STATE_48_SCENE_OBJECT,  # 81
]

REFLECTING_SCENE_OBJECTS = jnp.stack(
    [
        jnp.array(
            [
                scene_object.hit_box_height,
                scene_object.hit_box_width,
                scene_object.hit_box_x_offset,
                scene_object.hit_box_y_offset,
                scene_object.reflecting,
                scene_object.score_type,
            ]
        )
        for scene_object in ALL_SCENE_OBJECTS_LIST
    ]
).squeeze()
NON_REFLECTING_SCENE_OBJECTS = jnp.stack(
   [
       jnp.array([
           scene_object.hit_box_height,
           scene_object.hit_box_width,
           scene_object.hit_box_x_offset,
           scene_object.hit_box_y_offset,
           scene_object.reflecting,
           scene_object.score_type
       ]) for scene_object in ALL_SCENE_OBJECTS_LIST
       if scene_object.reflecting == 0
   ]
).squeeze()

# define the positions of the state information
STATE_TRANSLATOR: dict = {
    0: "ball_x",
    1: "ball_y",
    2: "ball_vel_x",
    3: "ball_vel_y",
    4: "ball_direction",
    5: "left_flipper_angle",
    6: "right_flipper_angle",
    7: "plunger_position",
    8: "plunger_power",
    9: "score",
    10: "lives",
    11: "bumper_multiplier",
    12: "active_targets",
    13: "target_cooldown",
    14: "special_target_cooldown",
    15: "atari_symbols",
    16: "rollover_counter",
    17: "step_counter",
    18: "ball_in_play",
    19: "respawn_timer",
    20: "color_cycling",
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
    ball_direction: (
        chex.Array
    )  # 0: left/up, 1:left/down , 2: right/up, 3: right/down (Shouldn't this be a function?)
    left_flipper_angle: chex.Array
    right_flipper_angle: chex.Array
    plunger_position: (
        chex.Array
    )  # Value between 0 and 20 where 20 means that the plunger is fully pulled
    plunger_power: (
        chex.Array
    )  # 2 * plunger_position, only set to non-zero value once fired, reset after hitting invisible block
    score: chex.Array
    lives: chex.Array
    bumper_multiplier: chex.Array
    active_targets: chex.Array  # Left diamond, Middle diamond, Right diamond, Special target
    target_cooldown: chex.Array
    special_target_cooldown: chex.Array
    atari_symbols: chex.Array
    rollover_counter: chex.Array
    step_counter: chex.Array
    ball_in_play: chex.Array
    respawn_timer: chex.Array
    color_cycling: chex.Array
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
    rollover_counter: chex.Array


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
        lambda s: s / PLUNGER_MAX_POSITION * BALL_MAX_SPEED,
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
    scene_object: chex.Array,
) -> chex.Array:
    """
    Calculate the hit point of the ball with the bounding box.
    Uses the slab method also known as ray AABB collision.

    scene_object is an array of the form
    [
        SceneObject.hit_box_height,
        SceneObject.hit_box_width,
        SceneObject.hit_box_x_offset,
        SceneObject.hit_box_y_offset,
        SceneObject.reflecting,
        SceneObject.score_type
    ]

    Returns:
        hit_point: jnp.ndarray, the time and hit point of the ball with the bounding box.
        hit_point[0]: jnp.ndarray, the time of entry
        hit_point[1]: jnp.ndarray, the x position of the hit point
        hit_point[2]: jnp.ndarray, the y position of the hit point
        hit_point[3]: jnp.ndarray, whether the obstacle was hit horizontally
        hit_point[4,5]: scene_object properties: reflecting, score_type
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

    tx1 = (scene_object[2] - ball_movement.old_ball_x) / trajectory_x
    tx2 = (scene_object[2] + scene_object[1] - ball_movement.old_ball_x) / trajectory_x
    ty1 = (scene_object[3] - ball_movement.old_ball_y) / trajectory_y
    ty2 = (scene_object[3] + scene_object[0] - ball_movement.old_ball_y) / trajectory_y

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

    hit_point_x = ball_movement.old_ball_x + t_entry * trajectory_x
    hit_point_y = ball_movement.old_ball_y + t_entry * trajectory_y

    # determine on which side the ball has hit the obstacle
    scene_object_half_height = scene_object[0] / 2.0
    scene_object_half_width = scene_object[1] / 2.0
    scene_object_middle_point_y = scene_object[3] + scene_object_half_height
    scene_object_middle_point_x = scene_object[2] + scene_object_half_width

    # distance of ball y to middle point of scene object
    d_middle_point_ball_y = jnp.abs(scene_object_middle_point_y - hit_point_y)
    d_middle_point_ball_x = jnp.abs(scene_object_middle_point_x - hit_point_x)

    # if ball hit the scene object to the top/bottom, this distance should be around half height of the scene object
    hit_horizontal = jnp.abs(d_middle_point_ball_y - scene_object_half_height) < 1e-1
    hit_vertical = jnp.abs(d_middle_point_ball_x - scene_object_half_width) < 1e-1

    hit_point = jnp.array(
        [
            t_entry,
            hit_point_x,
            hit_point_y,
            hit_horizontal,
            hit_vertical,
            scene_object[4],
            scene_object[5],
        ]
    )

    hit_point = jax.lax.cond(
        no_collision,
        lambda: jnp.array([T_ENTRY_NO_COLLISION, -1.0, -1.0, 0.0, 0.0, 1.0, 0.0]),
        lambda: hit_point,
    )

    return hit_point


@jax.jit
def _reflect_ball(
    ball_movement: BallMovement,
    hit_point: chex.Array,
) -> tuple[chex.Array]:
    """
    From the previous ball position, a hit point, the ball_direction, and the velocity,
    computes the new ball position and direction after reflection off of the obstacle
    hit.
    """
    velocity_x = ball_movement.new_ball_x - ball_movement.old_ball_x
    velocity_y = ball_movement.new_ball_y - ball_movement.old_ball_y

    # Calculate the trajectory of the ball to the hit point
    trajectory_to_hit_point_x = hit_point[1] - ball_movement.old_ball_x
    trajectory_to_hit_point_y = hit_point[2] - ball_movement.old_ball_y

    # Calculate the surface normal of the hit point
    surface_normal_x = jnp.where(hit_point[3], jnp.array(0), jnp.array(1))
    surface_normal_y = jnp.where(hit_point[3], jnp.array(1), jnp.array(0))

    # Calculate the dot product of the velocity and the surface normal
    velocity_normal_prod = velocity_x * surface_normal_x + velocity_y * surface_normal_y
    velocity_normal_prod = (
        velocity_normal_prod * (1 - VELOCITY_DAMPENING_VALUE)
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

    # If the ball hit a corner, reflect ball in the direction it came from instead of
    # via surface normal of the object
    new_ball_x = jnp.where(jnp.logical_and(hit_point[3], hit_point[4]), hit_point[1] - r * velocity_x, new_ball_x)
    new_ball_y = jnp.where(jnp.logical_and(hit_point[3], hit_point[4]), hit_point[2] - r * velocity_y, new_ball_y)

    return new_ball_x, new_ball_y

@jax.jit
def _check_obstacle_hits(
    state: VideoPinballState, ball_movement: BallMovement, scoring_list: chex.Array
) -> tuple[chex.Array, SceneObject]:
    """
    Check if the ball is hitting an obstacle.
    """
    reflecting_hit_points = jax.vmap(
        lambda scene_objects: _calc_hit_point(ball_movement, scene_objects)
    )(REFLECTING_SCENE_OBJECTS)
    non_reflecting_hit_points = jax.vmap(
        lambda scene_objects: _calc_hit_point(ball_movement, scene_objects)
    )(NON_REFLECTING_SCENE_OBJECTS) 
    """
    TODO FOR MAX:

    IDEA:
        We do not want to consider objects to be hit if they are not at the right position.
        I.e. if the flipper is currently on the lowest position, the highest one should be ignored for hits.
        The same goes for spinners, lit up special target, the lit up targets (diamonds).
        
    HOW WE DO IT:
        Manually set the entry time for all the components of these objects to a very high number.
        This will then ignore them for reflections and point scoring as only the lowest entry time is to be considered.
    """

    is_left_lit_up_target_active = state.active_targets[0]
    is_middle_lit_up_target_active = state.active_targets[1]
    is_right_lit_up_target_active = state.active_targets[2]
    is_special_lit_up_target_active = state.active_targets[3]
    # Disable inactive lit up targets (diamonds)
    # hit_points = jax.lax.cond(is_left_lit_up_target_active, lambda a: a, lambda a: a.at[0, 0].set(T_ENTRY_NO_COLLISION), non_reflecting_hit_points)
    # hit_points = jax.lax.cond(is_left_lit_up_target_active, lambda a: a, lambda a: a.at[1, 0].set(T_ENTRY_NO_COLLISION), non_reflecting_hit_points)

    # hit_points = jax.lax.cond(is_middle_lit_up_target_active, lambda a: a, lambda a: a.at[2, 0].set(T_ENTRY_NO_COLLISION), non_reflecting_hit_points)
    # hit_points = jax.lax.cond(is_middle_lit_up_target_active, lambda a: a, lambda a: a.at[3, 0].set(T_ENTRY_NO_COLLISION), non_reflecting_hit_points)

    # hit_points = jax.lax.cond(is_right_lit_up_target_active, lambda a: a, lambda a: a.at[4, 0].set(T_ENTRY_NO_COLLISION), non_reflecting_hit_points)
    # hit_points = jax.lax.cond(is_right_lit_up_target_active, lambda a: a, lambda a: a.at[5, 0].set(T_ENTRY_NO_COLLISION), non_reflecting_hit_points)

    # # Disable inactive special lit up target
    # hit_points = jax.lax.cond(is_special_lit_up_target_active, lambda a: a, lambda a: a.at[6, 0].set(T_ENTRY_NO_COLLISION), non_reflecting_hit_points)
    # hit_points = jax.lax.cond(is_special_lit_up_target_active, lambda a: a, lambda a: a.at[7, 0].set(T_ENTRY_NO_COLLISION), non_reflecting_hit_points)

    # # Disable inactive spinner parts
    # # We do this by setting the entry time of inactive spinner parts to a high value
    # spinner_state = state.step_counter % 8  # 0: Bottom, 1: Right, 2: Top, 3: Left
    # # Left Spinner SceneObject indices
    # # Bottom: 33,34,35,36,37
    # # Right:  38,39,40,41,42
    # # Top:    43,44,45,46,47
    # # Left:   48,49,50,51,52
    # initial_left_spinner_indices = jnp.array([33, 38, 43, 48]).astype(jnp.int32)
    # current_initial_left_spinner_index = initial_left_spinner_indices[spinner_state]
    # # Right Spinner SceneObject indices
    # # Bottom: 53,54,55,56,57
    # # Right:  58,59,60,61,62
    # # Top:    63,64,65,66,67
    # # Left:   68,69,70,71,72
    # initial_right_spinner_indices = jnp.array([53, 58, 63, 68]).astype(jnp.int32)
    # current_initial_right_spinner_index = initial_right_spinner_indices[spinner_state]

    # # Disable all the spinner parts not matching the current step
    # hit_points = jax.lax.cond(23 == current_initial_left_spinner_index, lambda a: a, lambda a: a.at[23:28, 0].set(T_ENTRY_NO_COLLISION), reflecting_hit_points)
    # hit_points = jax.lax.cond(28 == current_initial_left_spinner_index, lambda a: a, lambda a: a.at[28:33, 0].set(T_ENTRY_NO_COLLISION), reflecting_hit_points)
    # hit_points = jax.lax.cond(33 == current_initial_left_spinner_index, lambda a: a, lambda a: a.at[33:38, 0].set(T_ENTRY_NO_COLLISION), reflecting_hit_points)
    # hit_points = jax.lax.cond(38 == current_initial_left_spinner_index, lambda a: a, lambda a: a.at[38:43, 0].set(T_ENTRY_NO_COLLISION), reflecting_hit_points)
    # hit_points = jax.lax.cond(43 == current_initial_right_spinner_index, lambda a: a, lambda a: a.at[43:48, 0].set(T_ENTRY_NO_COLLISION), reflecting_hit_points)
    # hit_points = jax.lax.cond(48 == current_initial_right_spinner_index, lambda a: a, lambda a: a.at[48:53, 0].set(T_ENTRY_NO_COLLISION), reflecting_hit_points)
    # hit_points = jax.lax.cond(53 == current_initial_right_spinner_index, lambda a: a, lambda a: a.at[53:58, 0].set(T_ENTRY_NO_COLLISION), reflecting_hit_points)
    # hit_points = jax.lax.cond(58 == current_initial_right_spinner_index, lambda a: a, lambda a: a.at[58:63, 0].set(T_ENTRY_NO_COLLISION), reflecting_hit_points)

    # # Disable inactive flipper parts
    # left_flipper_angle = state.left_flipper_angle
    # right_flipper_angle = state.right_flipper_angle
    
    # # TODO: @Max implements after @Kun finishes the scene objects
    

    # Get and return first object hit (argmin entry time)
    lowest_reflecting_object_entry_time_index = jnp.argmin(reflecting_hit_points[:, 0])
    hit_point = reflecting_hit_points[lowest_reflecting_object_entry_time_index]
    lowest_entry_time = hit_point[0]

    # Probably could do this in a pythonic for-loop since jax unrolls those in jitted functions
    # reflecting scoring objects (only one possible)
    scoring_list_0 = scoring_list[0]
    scoring_list_1 = scoring_list[1]
    scoring_list_2 = scoring_list[2]
    scoring_list_3 = scoring_list[3]
    scoring_list_4 = scoring_list[4]
    scoring_list_5 = scoring_list[5]
    scoring_list_6 = scoring_list[6]
    scoring_list_7 = scoring_list[7]
    scoring_list_8 = scoring_list[8]

    scoring_list_0 = jnp.logical_or(scoring_list_0, jnp.where(hit_point[6] == 0, 1, 0))
    scoring_list_1 = jnp.logical_or(scoring_list_1, jnp.where(hit_point[6] == 1, 1, 0))
    scoring_list_2 = jnp.logical_or(scoring_list_2, jnp.where(hit_point[6] == 2, 1, 0))
    scoring_list_3 = jnp.logical_or(scoring_list_3, jnp.where(hit_point[6] == 3, 1, 0))
    scoring_list_4 = jnp.logical_or(scoring_list_4, jnp.where(hit_point[6] == 4, 1, 0))
    scoring_list_5 = jnp.logical_or(scoring_list_5, jnp.where(hit_point[6] == 5, 1, 0))
    scoring_list_6 = jnp.logical_or(scoring_list_6, jnp.where(hit_point[6] == 6, 1, 0))
    scoring_list_7 = jnp.logical_or(scoring_list_7, jnp.where(hit_point[6] == 7, 1, 0))
    scoring_list_8 = jnp.logical_or(scoring_list_8, jnp.where(hit_point[6] == 8, 1, 0))

    # non-reflecting scoring objects (multiple possible but only one of each type)
    # hit_before_reflection calculates for each object whether it was hit before hitting any rigid object, i.e. shape (, n_non_reflecting_objects)
    # the jnp.where(...) statement checks whether any non_reflecting_object of a specific scoring_type was hit
    hit_before_reflection = jnp.logical_and(non_reflecting_hit_points[:,0] < hit_point[0], non_reflecting_hit_points[:,0] != T_ENTRY_NO_COLLISION)
    scoring_list_0 = jnp.logical_or(scoring_list_0, jnp.where(jnp.any(jnp.logical_and(hit_before_reflection, non_reflecting_hit_points[:,6] == 0), axis=0), 1, 0))
    scoring_list_1 = jnp.logical_or(scoring_list_1, jnp.where(jnp.any(jnp.logical_and(hit_before_reflection, non_reflecting_hit_points[:,6] == 1), axis=0), 1, 0))
    scoring_list_2 = jnp.logical_or(scoring_list_2, jnp.where(jnp.any(jnp.logical_and(hit_before_reflection, non_reflecting_hit_points[:,6] == 2), axis=0), 1, 0))
    scoring_list_3 = jnp.logical_or(scoring_list_3, jnp.where(jnp.any(jnp.logical_and(hit_before_reflection, non_reflecting_hit_points[:,6] == 3), axis=0), 1, 0))
    scoring_list_4 = jnp.logical_or(scoring_list_4, jnp.where(jnp.any(jnp.logical_and(hit_before_reflection, non_reflecting_hit_points[:,6] == 4), axis=0), 1, 0))
    scoring_list_5 = jnp.logical_or(scoring_list_5, jnp.where(jnp.any(jnp.logical_and(hit_before_reflection, non_reflecting_hit_points[:,6] == 5), axis=0), 1, 0))
    scoring_list_6 = jnp.logical_or(scoring_list_6, jnp.where(jnp.any(jnp.logical_and(hit_before_reflection, non_reflecting_hit_points[:,6] == 6), axis=0), 1, 0))
    scoring_list_7 = jnp.logical_or(scoring_list_7, jnp.where(jnp.any(jnp.logical_and(hit_before_reflection, non_reflecting_hit_points[:,6] == 7), axis=0), 1, 0))
    scoring_list_8 = jnp.logical_or(scoring_list_8, jnp.where(jnp.any(jnp.logical_and(hit_before_reflection, non_reflecting_hit_points[:,6] == 8), axis=0), 1, 0))
    
    scoring_list = jnp.stack([
        scoring_list_0,
        scoring_list_1,
        scoring_list_2,
        scoring_list_3,
        scoring_list_4,
        scoring_list_5,
        scoring_list_6,
        scoring_list_7,
        scoring_list_8,
    ], axis=0)

    return hit_point, scoring_list


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
def _calc_ball_collision_loop(state: VideoPinballState, ball_movement: BallMovement):
    @jax.jit
    def _body_fun(
        args: tuple[VideoPinballState, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array],
    ):
        state, old_ball_x, old_ball_y, new_ball_x, new_ball_y, scoring_list, compute_flag = (
            args
        )

        @jax.jit
        def _compute_ball_collision(
            state, old_ball_x, old_ball_y, new_ball_x, new_ball_y, scoring_list
        ):
            _ball_movement = BallMovement(
                old_ball_x=old_ball_x,
                old_ball_y=old_ball_y,
                new_ball_x=new_ball_x,
                new_ball_y=new_ball_y,
            )
            hit_data, scoring_list = _check_obstacle_hits(
                state, _ball_movement, scoring_list
            )
            reflected_ball_x, reflected_ball_y = _reflect_ball(_ball_movement, hit_data)
            # if there was a collision, returned BallMovement is from hit point to reflected position
            # if there was no collision, returned BallMovement will be the original BallMovement
            no_collision = hit_data[0] == T_ENTRY_NO_COLLISION
            collision = jnp.logical_not(no_collision)

            # jax.debug.print("ballmovement: old x: {}, y: {}, new x: {}, y: {} | hitpoint x: {}, y: {}, horizontal: {} | reflected x: {}, y: {} | collision: {} | moved: {} | trajectory from hit to old: ({}, {}), ", old_ball_x, old_ball_y, new_ball_x, new_ball_y, hit_data[1], hit_data[2], hit_data[3], reflected_ball_x, reflected_ball_y, collision, ball_moved_after_collision, ball_trajectory_from_hit_point_to_old_x, ball_trajectory_from_hit_point_to_old_y)
            old_ball_x = jnp.where(collision, hit_data[1], old_ball_x)
            old_ball_y = jnp.where(collision, hit_data[2], old_ball_y)
            new_ball_x = jnp.where(collision, reflected_ball_x, new_ball_x)
            new_ball_y = jnp.where(collision, reflected_ball_y, new_ball_y)

            return (
                state,
                old_ball_x,
                old_ball_y,
                new_ball_x,
                new_ball_y,
                scoring_list,
                collision,
            )

        # Where compute_flag is true, compute ball collisions along the trajectory of BallMovement
        # The returned BallMovement is either the original BallMovement or a new BallMovement, from the
        # obstacle hit to the new location of the ball after reflecting it from that obstacle
        state, old_ball_x, old_ball_y, new_ball_x, new_ball_y, scoring_list, collision = (
            jax.lax.cond(
                compute_flag,
                _compute_ball_collision,
                lambda state, old_ball_x, old_ball_y, new_ball_x, new_ball_y, scoring_list: (
                    state,
                    old_ball_x,
                    old_ball_y,
                    new_ball_x,
                    new_ball_y,
                    scoring_list,
                    compute_flag,
                ),
                state,
                old_ball_x,
                old_ball_y,
                new_ball_x,
                new_ball_y,
                scoring_list,
            )
        )
        # If there was no collision, set compute_flag to False
        compute_flag = jnp.logical_and(compute_flag, collision)
        # "or" the scoring_lists together
        return (
            state,
            old_ball_x,
            old_ball_y,
            new_ball_x,
            new_ball_y,
            scoring_list,
            compute_flag,
        )

    @jax.jit
    def _cond_fun(
        args: tuple[VideoPinballState, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array],
    ):
        state, old_ball_x, old_ball_y, new_ball_x, new_ball_y, scoring_list, compute_flag = (
            args
        )
        # jax.debug.print("Old ball x: {}, y: {}, New ball x: {}, y: {}.  Compute: {}", old_ball_x, old_ball_y, new_ball_x, new_ball_y, compute_flag)
        return jnp.any(compute_flag)

    scoring_list = jnp.stack([jnp.zeros_like(ball_movement.old_ball_x, dtype=jnp.bool) for i in range(9)])
    compute_flag = jnp.ones_like(ball_movement.new_ball_x, dtype=jnp.bool)
    state, old_ball_x, old_ball_y, new_ball_x, new_ball_y, scoring_list, _ = (
        jax.lax.while_loop(
            # while there are new BallMovements along whose trajectory there might be collisions (compute_flag set to True),
            # compute whether there are collisions and if so, calculate a new BallMovement and keep
            # corresponding compute_flag set to True, else set it to False to keep the BallMovement
            _cond_fun,
            _body_fun,
            (
                state,
                ball_movement.old_ball_x,
                ball_movement.old_ball_y,
                ball_movement.new_ball_x,
                ball_movement.new_ball_y,
                scoring_list,
                compute_flag,
            ),
        )
    )

    return (
        BallMovement(
            old_ball_x=old_ball_x,
            old_ball_y=old_ball_y,
            new_ball_x=new_ball_x,
            new_ball_y=new_ball_y,
        ),
        scoring_list,
    )


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

    """
    Ball movement calculation observing its direction 
    """
    ball_x, ball_y, ball_vel_x, ball_vel_y, signed_ball_vel_x, signed_ball_vel_y = (
        _calc_ball_change(
            state.ball_x, state.ball_y, ball_vel_x, ball_vel_y, ball_direction
        )
    )

    """
    Check if the ball is hitting the invisible block at the plunger hole
    """
    ball_movement = BallMovement(
        old_ball_x=state.ball_x,  # type: ignore
        old_ball_y=state.ball_y,  # type: ignore
        new_ball_x=ball_x,  # type: ignore
        new_ball_y=ball_y,  # type: ignore
    )
    invisible_block_hit_data = _calc_hit_point(
        ball_movement,
        jnp.array(
            [
                INVISIBLE_BLOCK_SCENE_OBJECT.hit_box_height,
                INVISIBLE_BLOCK_SCENE_OBJECT.hit_box_width,
                INVISIBLE_BLOCK_SCENE_OBJECT.hit_box_x_offset,
                INVISIBLE_BLOCK_SCENE_OBJECT.hit_box_y_offset,
                INVISIBLE_BLOCK_SCENE_OBJECT.reflecting,
                INVISIBLE_BLOCK_SCENE_OBJECT.score_type,
            ]
        ),
    )
    is_invisible_block_hit = jnp.logical_and(
        jnp.logical_not(ball_in_play),
        invisible_block_hit_data[0] != T_ENTRY_NO_COLLISION,
    )
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
        old_ball_x=state.ball_x,  # type: ignore
        old_ball_y=state.ball_y,  # type: ignore
        new_ball_x=ball_x,  # type: ignore
        new_ball_y=ball_y,  # type: ignore
    )

    ball_movement, scoring_list = _calc_ball_collision_loop(state, ball_movement)

    ball_trajectory_x = ball_movement.new_ball_x - ball_movement.old_ball_x
    ball_trajectory_y = ball_movement.new_ball_y - ball_movement.old_ball_y

    ball_x = ball_movement.new_ball_x
    ball_y = ball_movement.new_ball_y

    """
    Some final calculations
    """
    ball_direction = _get_ball_direction(ball_trajectory_x, ball_trajectory_y)
    # Clip the ball velocity to the maximum speed
    ball_vel_x = jnp.clip(jnp.abs(signed_ball_vel_x), 0, BALL_MAX_SPEED)
    ball_vel_y = jnp.clip(jnp.abs(signed_ball_vel_y), 0, BALL_MAX_SPEED)

    return (
        ball_x,
        ball_y,
        ball_direction,
        ball_vel_x,
        ball_vel_y,
        ball_in_play,
        scoring_list,
    )


@jax.jit
def _reset_ball(state: VideoPinballState):
    """
    When the ball goes into the gutter or into the plunger hole,
    respawn the ball on the launcher.
    """

    return BALL_START_X, BALL_START_Y, jnp.array(0.0), jnp.array(0.0)


@jax.jit
def _ball_enters_gutter(state: VideoPinballState):

    respawn_timer = jnp.where(
        state.rollover_counter > 1,
        jnp.array((state.rollover_counter - 1) * 16).astype(jnp.int32),
        jnp.array(1).astype(jnp.int32),
    )

    return respawn_timer


@jax.jit
def handle_ball_in_gutter(
    rt,
    rollover_counter,
    score,
    atari_symbols,
    lives,
    active_targets,
    special_target_cooldown,
):

    multiplier = jnp.clip(atari_symbols + 1, max=4)
    score = jnp.where(rt % 16 == 15, score + 1000 * multiplier, score)
    rollover_counter = jnp.where(rt % 16 == 15, rollover_counter - 1, rollover_counter)
    respawn_timer = jnp.where(rt > 0, rt - 1, rt)

    lives, active_targets, atari_symbols, special_target_cooldown = jax.lax.cond(
        respawn_timer == 0,
        lambda l, asym: _reset_stuff_and_handle_lives(l, asym),
        lambda l, asym: (l, active_targets, asym, special_target_cooldown),
        lives,
        atari_symbols,
    )

    return (
        respawn_timer,
        rollover_counter,
        score,
        atari_symbols,
        lives,
        active_targets,
        special_target_cooldown,
    )


@jax.jit
def _reset_stuff_and_handle_lives(lives, atari_symbols):
    lives = jax.lax.cond(
        atari_symbols < 4,
        lambda x: x + 1,
        lambda x: x,
        operand=lives,
    )

    active_targets = jnp.array([True, True, True, False]).astype(jnp.bool)
    atari_symbols = jnp.array(0).astype(jnp.int32)
    special_target_cooldown = jnp.array(0).astype(jnp.int32)

    # TODO: Whoever implements tilt mode: Turn off tilt mode here in this function

    return lives, active_targets, atari_symbols, special_target_cooldown


@jax.jit
def process_objects_hit(state: VideoPinballState, objects_hit):
    # Bumpers: Give points
    # Targets: Make them disappear, give points
    # Targets: Check if all hit, increase multiplier
    # BonusTarget: Give points, make screen flash, something else?
    # Rollover: Give points, increase number
    # Atari: Give points, make Atari symbol at bottom appear
    # Assume objects_hit is list:
    # [0: no score, 1: Bumper, 2: Spinner, 3: Left Rollover, 4: Atari Rollover,
    # 5: Special Lit Up Target, 6: Left Lit Up Target, 7: Middle Lit Up Target, 8: Right Lit Up Target]

    # Bumper points
    score = state.score
    active_targets = state.active_targets
    atari_symbols = state.atari_symbols
    rollover_counter = state.rollover_counter

    score += jnp.where(
        objects_hit[1],
        100 * state.bumper_multiplier,
        0,
    )

    # Give points for targets hit
    score += jnp.where(objects_hit[6], 100, 0)
    score += jnp.where(objects_hit[7], 100, 0)
    score += jnp.where(objects_hit[8], 100, 0)

    # Make hit targets disappear
    active_targets = jax.lax.cond(
        objects_hit[6],
        lambda s: jnp.array([False, s[1], s[2], s[3]]).astype(jnp.bool),
        lambda s: s,
        operand=active_targets,
    )

    active_targets = jax.lax.cond(
        objects_hit[7],
        lambda s: jnp.array([s[0], False, s[2], s[3]]).astype(jnp.bool),
        lambda s: s,
        operand=active_targets,
    )

    active_targets = jax.lax.cond(
        objects_hit[8],
        lambda s: jnp.array([s[0], s[1], False, s[3]]).astype(jnp.bool),
        lambda s: s,
        operand=active_targets,
    )

    # Bottom Bonus Target
    score += jnp.where(objects_hit[6], 1100, 0)
    active_targets, color_cycling = jax.lax.cond(
        objects_hit[5],
        lambda s, cc: (
            jnp.array([s[0], s[1], s[2], False]).astype(jnp.bool),
            jnp.array(30).astype(jnp.int32),
        ),
        lambda s, cc: (s, cc),
        active_targets,
        state.color_cycling,
    )

    # Give score for hitting the rollover and increase its number
    score += jnp.where(objects_hit[3], 100, 0)
    rollover_counter = jax.lax.cond(
        objects_hit[7],
        lambda s: s + 1,
        lambda s: s,
        operand=rollover_counter,
    )

    # Give score for hitting the Atari symbol and make a symbol appear at the bottom
    score += jnp.where(objects_hit[4], 100, 0)
    atari_symbols = jax.lax.cond(
        jnp.logical_and(objects_hit[8], atari_symbols < 4),
        lambda s: s + 1,
        lambda s: s,
        operand=atari_symbols,
    )

    # Do color cycling when the fourth Atari symbol has been hit
    color_cycling = jnp.where(
        jnp.logical_and(state.atari_symbols == 3, atari_symbols == 4),
        jnp.array(30).astype(jnp.int32),
        color_cycling,
    )

    # Give 1 point for hitting a spinner
    score += jnp.where(objects_hit[2], 1, 0)

    return score, active_targets, atari_symbols, rollover_counter, color_cycling


@jax.jit
def handle_target_cooldowns(
    state: VideoPinballState, previous_active_targets, color_cycling
):

    targets_are_inactive = jnp.logical_and(
        jnp.logical_not(previous_active_targets[0]),
        jnp.logical_and(
            jnp.logical_not(previous_active_targets[1]),
            jnp.logical_not(previous_active_targets[2]),
        ),
    )

    # Start 2 second cooldown after hitting all targets until they respawn
    target_cooldown, increase_bm, color_cycling = jax.lax.cond(
        jnp.logical_and(targets_are_inactive, state.target_cooldown == -1),
        lambda cd, cc: (
            jnp.array(TARGET_RESPAWN_COOLDOWN).astype(jnp.int32),
            True,
            jnp.array(-9),
        ),
        lambda cd, cc: (cd, False, cc),
        state.target_cooldown,
        color_cycling,
    )

    # Increase Bumper multiplier if all targets got hit
    bumper_multiplier = jax.lax.cond(
        jnp.logical_and(increase_bm, state.bumper_multiplier < 9),
        lambda s: s + 1,
        lambda s: s,
        operand=state.bumper_multiplier,
    )

    # count down the cooldown timer
    target_cooldown = jax.lax.cond(
        jnp.logical_and(targets_are_inactive, target_cooldown != -1),
        lambda s: s - 1,
        lambda s: s,
        operand=target_cooldown,
    )

    # After the cooldown, respawn the targets
    target_cooldown, active_targets = jax.lax.cond(
        jnp.logical_and(targets_are_inactive, target_cooldown == 0),
        lambda tc, pat: (
            jnp.array(-1).astype(jnp.int32),
            jnp.array([True, True, True, pat[3]]).astype(jnp.bool),
        ),
        lambda tc, pat: (tc, pat),
        target_cooldown,
        previous_active_targets,
    )

    # count down the despawn cooldown timer
    special_target_cooldown = jax.lax.cond(
        jnp.logical_and(state.special_target_cooldown > 0, state.ball_in_play),
        lambda s: s - 1,
        lambda s: s,
        operand=state.special_target_cooldown,
    )

    # count up the respawn cooldown timer
    special_target_cooldown = jax.lax.cond(
        jnp.logical_and(special_target_cooldown < -1, state.ball_in_play),
        lambda s: s + 1,
        lambda s: s,
        operand=special_target_cooldown,
    )

    # despawn the special target
    special_target_cooldown, active_targets = jax.lax.cond(
        jnp.logical_and(special_target_cooldown == 0, state.ball_in_play),
        lambda cd, a: (
            cd - SPECIAL_TARGET_INACTIVE_DURATION,
            a.at[3].set(False),
        ),  # Check how the real cooldown works
        lambda cd, a: (cd, a),
        special_target_cooldown,
        active_targets,
    )

    # spawn the special target
    special_target_cooldown, active_targets = jax.lax.cond(
        jnp.logical_and(special_target_cooldown == -1, state.ball_in_play),
        lambda cd, a: (cd + SPECIAL_TARGET_ACTIVE_DURATION, a.at[3].set(True)),
        lambda cd, a: (cd, a),
        special_target_cooldown,
        active_targets,
    )

    return (
        active_targets,
        target_cooldown,
        special_target_cooldown,
        bumper_multiplier,
        color_cycling,
    )


@jax.jit
def color_cycler(color):

    new_color = jax.lax.cond(
        color > 0,  # Condition 1: is value > 0?
        lambda x: color - 1,  # True branch for Condition 1
        lambda x: jnp.where(  # False branch for Condition 1 (nested cond)
            color < 0,  # Condition 2: is value < 0?
            color + 1,  # True branch for Condition 2
            color,  # False branch for Condition 2 (must be == 0)
        ),
        None,  # Operand for outer cond (not used by branch funcs)
    )

    new_color = jnp.where(new_color == -1, jnp.array(14).astype(jnp.int32), new_color)

    return new_color


@jax.jit
def handle_color_cycling(raster, color):

    bg_color_mask = jnp.all(raster == BACKGROUND_COLOR, axis=-1, keepdims=True)
    wall_color_mask = jnp.all(raster == WALL_COLOR, axis=-1, keepdims=True)
    group3_color_mask = jnp.all(raster == GROUP3_COLOR, axis=-1, keepdims=True)
    group4_color_mask = jnp.all(raster == GROUP4_COLOR, axis=-1, keepdims=True)
    group5_color_mask = jnp.all(raster == GROUP5_COLOR, axis=-1, keepdims=True)

    raster = jnp.where(bg_color_mask, BACKGROUND_COLOR_CYCLING[color], raster)
    raster = jnp.where(wall_color_mask, WALL_COLOR_CYCLING[color], raster)
    raster = jnp.where(group3_color_mask, GROUP3_COLOR_CYCLING[color], raster)
    raster = jnp.where(group4_color_mask, GROUP4_COLOR_CYCLING[color], raster)
    raster = jnp.where(group5_color_mask, GROUP5_COLOR_CYCLING[color], raster)

    raster = raster.at[0:8, 175:176, :].set(BG_COLOR)
    raster = raster.at[:, 191:, :].set(BG_COLOR)

    return raster


@jax.jit
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
            active_targets=jnp.array([True, True, True, False]).astype(jnp.bool),
            target_cooldown=jnp.array(-1).astype(jnp.int32),
            special_target_cooldown=jnp.array(-120).astype(jnp.int32),
            atari_symbols=jnp.array(0).astype(jnp.int32),
            rollover_counter=jnp.array(1).astype(jnp.int32),
            step_counter=jnp.array(0).astype(jnp.int32),
            ball_in_play=jnp.array(False).astype(jnp.bool),
            respawn_timer=jnp.array(0).astype(jnp.int32),
            color_cycling=jnp.array(0).astype(jnp.int32),
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
        # jax.debug.print(
        #    "Plunger Step, Plunger Power {plunger_power}, New Plunger Power {new_plunger_power}",
        #    plunger_power=plunger_power,
        #    new_plunger_power=new_plunger_power,
        # )

        # Step 2: Update ball position and velocity
        (
            ball_x,
            ball_y,
            ball_direction,
            ball_vel_x,
            ball_vel_y,
            ball_in_play,
            scoring_list,
        ) = ball_step(
            state,
            new_plunger_power,
            action,
        )

        # Step 3: Check if ball is in the gutter or in plunger hole
        ball_in_gutter = ball_y > 192
        # TODO if ball is back in plunger hole reset, not instantly
        ball_reset = jnp.logical_or(
            ball_in_gutter,
            jnp.logical_and(ball_x > 148, ball_y > 129),
        )

        # Step 4: Update scores and handle special objects
        # TODO: Input the list of objects hit once collisions are done
        score, active_targets, atari_symbols, rollover_counter, color_cycling = (
            process_objects_hit(
                state,
                scoring_list,
            )
        )
        (
            active_targets,
            target_cooldown,
            special_target_cooldown,
            bumper_multiplier,
            color_cycling,
        ) = handle_target_cooldowns(state, active_targets, color_cycling)

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

        respawn_timer = jax.lax.cond(
            ball_in_gutter,
            lambda at: _ball_enters_gutter(state),
            lambda at: state.respawn_timer,
            active_targets,
        )

        # TODO: Whoever implements tilt mode: properly define this tilt_mode_active boolean
        #  (this needs to be exactly here, don't move it)
        tilt_mode_active = False
        score = jnp.where(tilt_mode_active, state.score, score)

        (
            respawn_timer,
            rollover_counter,
            score,
            atari_symbols,
            lives,
            active_targets,
            special_target_cooldown,
        ) = jax.lax.cond(
            respawn_timer > 0,
            lambda rt, rc, s, asym, l, at, stc: handle_ball_in_gutter(
                rt, rc, s, asym, l, at, stc
            ),
            lambda rt, rc, s, asym, l, at, stc: (rt, rc, s, asym, l, at, stc),
            respawn_timer,
            rollover_counter,
            score,
            atari_symbols,
            state.lives,
            active_targets,
            special_target_cooldown,
        )

        ball_in_play = jnp.where(
            jnp.logical_or(ball_reset, respawn_timer > 0),
            jnp.array(False),
            ball_in_play,
        )
        # ball_in_play = jnp.where(ball_reset, jnp.array(False), ball_in_play)

        color_cycling = color_cycler(color_cycling)

        # TEST: This makes the rollover increase and make the ball hit the gutter after a short time
        # rollover_counter = jnp.where(state.step_counter % 400 == 399, jnp.array(3).astype(jnp.int32), rollover_counter)
        # ball_y_final = jnp.where(state.step_counter % 420 == 419, jnp.array(194).astype(jnp.int32), ball_y_final)

        # active_targets = jnp.where(state.step_counter % 420 == 419, jnp.array([False, False, False, False]).astype(jnp.bool), active_targets)

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
            rollover_counter=rollover_counter,
            step_counter=jnp.array(state.step_counter + 1).astype(jnp.int32),
            ball_in_play=ball_in_play,
            respawn_timer=respawn_timer,
            color_cycling=color_cycling,
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
        # jax.debug.print("------------------------------------------")

        # Check if all lives are lost and the game should reset
        observation, new_state = jax.lax.cond(
            jnp.logical_and(lives > 3, respawn_timer == 0),
            lambda ob, ns: self.reset(
                jax.random.PRNGKey(score + special_target_cooldown + env_reward)
            ),
            lambda ob, ns: (ob, ns),
            observation,
            new_state,
        )

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
            rollover_counter=state.rollover_counter,
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
    # sprite_wall_rollover = aj.loadFrame(os.path.join(SPRITES_BASE_DIR, "Wallrollover.npy"), transpose=True)
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
        # "wall_rollover": sprite_wall_rollover,
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


@jax.jit
def render_tilt_mode(r):
    r = r.at[:, 0:16, :].set(TILT_MODE_COLOR)
    r = r.at[36:40, 184:192, :].set(BG_COLOR)
    r = r.at[120:124, 184:192, :].set(BG_COLOR)

    return r


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

        # TODO: Whoever implements tilt mode: properly define this tilt_mode_active boolean
        tilt_mode_active = False
        raster = jnp.where(tilt_mode_active, render_tilt_mode(raster), raster)

        # Render animated objects
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

        # Render score
        frame_unknown = aj.get_sprite_frame(self.sprites["score_number_digits"], 1)
        raster = aj.render_at(raster, 4, 3, frame_unknown)

        displayed_lives = jnp.clip(state.lives, max=3)
        frame_ball_count = aj.get_sprite_frame(
            self.sprites["score_number_digits"], displayed_lives
        )
        raster = aj.render_at(raster, 36, 3, frame_ball_count)

        numbers = _split_integer(state.score)
        frame_score1 = aj.get_sprite_frame(
            self.sprites["score_number_digits"], numbers[0]
        )
        raster = aj.render_at(raster, 64, 3, frame_score1)
        frame_score2 = aj.get_sprite_frame(
            self.sprites["score_number_digits"], numbers[1]
        )
        raster = aj.render_at(raster, 80, 3, frame_score2)
        frame_score3 = aj.get_sprite_frame(
            self.sprites["score_number_digits"], numbers[2]
        )
        raster = aj.render_at(raster, 96, 3, frame_score3)
        frame_score4 = aj.get_sprite_frame(
            self.sprites["score_number_digits"], numbers[3]
        )
        raster = aj.render_at(raster, 112, 3, frame_score4)
        frame_score5 = aj.get_sprite_frame(
            self.sprites["score_number_digits"], numbers[4]
        )
        raster = aj.render_at(raster, 128, 3, frame_score5)
        frame_score6 = aj.get_sprite_frame(
            self.sprites["score_number_digits"], numbers[5]
        )
        raster = aj.render_at(raster, 144, 3, frame_score6)

        # Render special yellow field objects
        frame_bumper_left = aj.get_sprite_frame(
            self.sprites["field_number_digits"], state.bumper_multiplier
        )
        raster = aj.render_at(raster, 46, 122, frame_bumper_left)
        frame_bumper_middle = aj.get_sprite_frame(
            self.sprites["field_number_digits"], state.bumper_multiplier
        )
        raster = aj.render_at(raster, 78, 58, frame_bumper_middle)
        frame_bumper_right = aj.get_sprite_frame(
            self.sprites["field_number_digits"], state.bumper_multiplier
        )
        raster = aj.render_at(raster, 110, 122, frame_bumper_right)

        displayed_rollover_number = state.rollover_counter % 9
        frame_rollover_left = aj.get_sprite_frame(
            self.sprites["field_number_digits"], displayed_rollover_number
        )
        raster = aj.render_at(raster, 46, 58, frame_rollover_left)
        frame_atari_logo = aj.get_sprite_frame(self.sprites["atari_logo"], 0)
        raster = aj.render_at(raster, 109, 58, frame_atari_logo)

        frame_target = aj.get_sprite_frame(self.sprites["yellow_diamond_top"], 0)
        raster = jax.lax.cond(
            state.active_targets[0],
            lambda r: aj.render_at(raster, 60, 24, frame_target),
            lambda r: raster,
            operand=raster,
        )

        raster = jax.lax.cond(
            state.active_targets[1],
            lambda r: aj.render_at(raster, 76, 24, frame_target),
            lambda r: raster,
            operand=raster,
        )

        raster = jax.lax.cond(
            state.active_targets[2],
            lambda r: aj.render_at(raster, 92, 24, frame_target),
            lambda r: raster,
            operand=raster,
        )

        frame_special_target = aj.get_sprite_frame(
            self.sprites["yellow_diamond_bottom"], 0
        )
        raster = jax.lax.cond(
            state.active_targets[3],
            lambda r: aj.render_at(raster, 76, 120, frame_special_target),
            lambda r: raster,
            operand=raster,
        )

        # Render Atari Logos and the X
        raster = jax.lax.cond(
            jnp.logical_and(state.atari_symbols > 0, state.respawn_timer == 0),
            lambda r: aj.render_at(raster, 60, 154, frame_atari_logo),
            lambda r: raster,
            operand=raster,
        )

        raster = jax.lax.cond(
            jnp.logical_and(
                jnp.logical_or(state.atari_symbols == 2, state.atari_symbols == 3),
                state.respawn_timer == 0,
            ),
            lambda r: aj.render_at(raster, 76, 154, frame_atari_logo),
            lambda r: raster,
            operand=raster,
        )

        raster = jax.lax.cond(
            jnp.logical_and(state.atari_symbols > 2, state.respawn_timer == 0),
            lambda r: aj.render_at(raster, 90, 154, frame_atari_logo),
            lambda r: raster,
            operand=raster,
        )

        frame_X = aj.get_sprite_frame(self.sprites["x"], 0)
        raster = jax.lax.cond(
            jnp.logical_and(state.atari_symbols == 4, state.respawn_timer == 0),
            lambda r: aj.render_at(raster, 76, 157, frame_X),
            lambda r: raster,
            operand=raster,
        )

        # Handle color cycling
        color = jnp.ceil(state.color_cycling / jnp.array(4.0)).astype(jnp.int32)
        color = jnp.clip(color, min=0)

        raster = jax.lax.cond(
            color > 0, lambda r: handle_color_cycling(r, color), lambda r: r, raster
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
