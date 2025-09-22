# Team Mario Bros: Sai Aakaash, Sam Diehl, Jonas Rudolph, Harshith Salanke

import os
from functools import partial
from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
import chex
from jax import lax, debug
from gymnax.environments import spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action

# --- Screen params---
SCREEN_WIDTH = 160
SCREEN_HEIGHT = 210
WINDOW_SCALE = 3

# --- Physik params ---
MOVE_SPEED = 0.5
ASCEND_VY = -1.0  # ↑ 2 px / frame
DESCEND_VY = 1.0  # ↓ 2 px / frame
ASCEND_FRAMES = 21 * 2  # 42 px tall jump (21 × 2)
# Enemy horizontal speed in pixels per frame (1 px every 8 frames = 0.125 px/frame)
ENEMY_MOVE_SPEED = 0.125

# -------- Movement params ----------------------------------
movement_pattern = jnp.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0], dtype=jnp.float32)
pat_len = movement_pattern.shape[0]
# -------- Break params ----------------------------------
BRAKE_DURATION = 10 * 2# in Frames
BRAKE_TOTAL_DISTANCE = 7.0 * 2# in Pixels
BRAKE_SPEED = jnp.array(BRAKE_TOTAL_DISTANCE / BRAKE_DURATION, dtype=jnp.float32)  # ≈ 0.7 px/frame

# --- Player params ---------------
PLAYER_SIZE = (9, 23)  # w, h
PLAYER_START_X, PLAYER_START_Y = 15, 140
PLAYER_RESPWAN_XY = jnp.array([78, 36], dtype=jnp.float32)
PALETTE = jnp.array([
    [0, 0, 0],        # 0: transparent
    [104, 72, 198],   # 1: blue, mario hat
    [252, 188, 116],  # 2: skin
    [181, 83, 40],    # 3: brown, mario overalls
    [134, 106, 38],   # 4: green mario shoes
    [152, 163, 74],   # 5: enemy olive
    [227, 151, 89],   # 6: fireball orange
], dtype=jnp.uint8)

PLAYER_STANDING_RIGHT= jnp.array([
    [0,0,0,1,1,0,0,0], 
    [0,0,1,1,1,0,0,0], 
    [0,1,1,1,1,1,1,0], 
    [0,0,0,2,2,0,0,0], 
    [0,2,0,2,0,0,0,0], 
    [0,2,0,2,2,2,2,0], 
    [0,2,0,2,0,2,2,2], 
    [0,2,2,2,0,0,0,0], 
    [0,2,2,2,2,2,2,0], 
    [0,0,2,2,2,2,0,0], 
    [0,0,0,2,2,0,0,0], 
    [0,0,3,3,3,3,0,0], 
    [0,3,3,3,3,3,0,3], 
    [3,3,0,3,3,3,3,3], 
    [3,3,0,3,3,3,3,0], 
    [0,0,1,1,1,1,1,0], 
    [0,0,1,1,1,1,1,0], 
    [0,0,1,1,1,1,1,0], 
    [0,0,1,1,1,1,1,0], 
    [0,0,1,1,0,1,1,0], 
    [0,0,1,1,0,1,1,0], 
    [0,0,4,4,0,4,4,0], 
    [0,0,4,4,4,0,4,4],  
], dtype=jnp.uint8)

PLAYER_STANDING_LEFT = jnp.array([
    [0,0,0,1,1,0,0,0],
    [0,0,0,1,1,1,0,0],
    [0,1,1,1,1,1,1,0],
    [0,0,0,2,2,0,0,0],
    [0,0,0,0,2,0,2,0],
    [0,2,2,2,2,0,2,0],
    [2,2,2,0,2,0,2,0],
    [0,0,0,0,2,2,2,0],
    [0,2,2,2,2,2,2,0],
    [0,0,2,2,2,2,0,0],
    [0,0,0,2,2,0,0,0],
    [0,0,3,3,3,3,0,0],
    [3,0,3,3,3,3,3,0],
    [3,3,3,3,3,0,3,3],
    [0,3,3,3,3,0,3,3],
    [0,1,1,1,1,1,0,0],
    [0,1,1,1,1,1,0,0],
    [0,1,1,1,1,1,0,0],
    [0,1,1,1,1,1,0,0],
    [0,1,1,0,1,1,0,0],
    [0,1,1,0,1,1,0,0],
    [0,4,4,0,4,4,0,0],
    [4,4,0,4,4,4,0,0],
], dtype=jnp.uint8)

PLAYER_WALK_RIGHT_1 = jnp.array([
    [0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 2, 2, 0, 0, 0],
    [0, 2, 0, 2, 0, 0, 0, 0],
    [0, 2, 0, 2, 2, 2, 2, 0],
    [0, 2, 0, 2, 0, 2, 2, 2],
    [0, 2, 2, 2, 0, 0, 0, 0],
    [0, 2, 2, 2, 2, 2, 2, 0],
    [0, 0, 2, 2, 2, 2, 0, 0],
    [0, 0, 0, 2, 2, 0, 0, 0],
    [0, 0, 3, 3, 3, 3, 0, 0],
    [0, 3, 3, 3, 3, 3, 0, 3],
    [0, 3, 3, 0, 3, 3, 3, 3],
    [0, 0, 0, 0, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 0, 0, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0],
    [4, 0, 0, 0, 0, 0, 0, 0]
], dtype=jnp.uint8)

PLAYER_WALK_RIGHT_2= jnp.array([
    [0, 0, 1, 1, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 2, 2, 0, 0, 0, 0],
    [2, 0, 2, 0, 0, 0, 0, 0],
    [2, 0, 2, 2, 2, 2, 0, 0],
    [2, 0, 2, 0, 2, 2, 2, 0],
    [2, 2, 2, 0, 0, 0, 0, 0],
    [2, 2, 2, 2, 2, 2, 0, 0],
    [0, 2, 2, 2, 2, 0, 0, 0],
    [0, 0, 2, 2, 0, 0, 0, 0],
    [0, 3, 3, 3, 3, 0, 0, 0],
    [3, 3, 3, 3, 3, 0, 0, 0],
    [3, 3, 0, 3, 3, 3, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 0],
    [0, 4, 0, 0, 0, 4, 4, 0],
    [0, 0, 0, 4, 0, 4, 4, 4]
], dtype=jnp.uint8)

PLAYER_WALK_LEFT_1 = jnp.array([
    [0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 2, 2, 0, 0, 0],
    [0, 0, 0, 0, 2, 0, 2, 0],
    [0, 2, 2, 2, 2, 0, 2, 0],
    [2, 2, 2, 0, 2, 0, 2, 0],
    [0, 0, 0, 0, 2, 2, 2, 0],
    [0, 2, 2, 2, 2, 2, 2, 0],
    [0, 0, 2, 2, 2, 2, 0, 0],
    [0, 0, 0, 2, 2, 0, 0, 0],
    [0, 0, 3, 3, 3, 3, 0, 0],
    [3, 0, 3, 3, 3, 3, 3, 0],
    [3, 3, 3, 3, 0, 3, 3, 0],
    [3, 3, 3, 3, 0, 3, 3, 0],
    [0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 0],
    [1, 0, 1, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 4],
], dtype=jnp.uint8)

PLAYER_WALK_LEFT_2 = jnp.array([
    [0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 2, 2, 0, 0],
    [0, 0, 0, 0, 2, 0, 2],
    [0, 2, 2, 2, 2, 0, 2],
    [2, 2, 2, 0, 2, 0, 2],
    [0, 0, 0, 0, 2, 2, 2],
    [0, 2, 2, 2, 2, 2, 2],
    [0, 0, 2, 2, 2, 2, 0],
    [0, 0, 0, 2, 2, 0, 0],
    [0, 0, 3, 3, 3, 3, 0],
    [0, 0, 3, 3, 3, 3, 3],
    [0, 3, 3, 3, 0, 3, 3],
    [0, 3, 3, 3, 0, 3, 3],
    [0, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0],
    [4, 4, 0, 0, 0, 4, 0],
], dtype=jnp.uint8)

PLAYER_JUMP_LEFT = jnp.array([
    [0, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 2, 2, 0, 0, 0],
    [0, 0, 0, 0, 2, 0, 2, 0],
    [0, 2, 2, 2, 2, 0, 2, 0],
    [2, 2, 2, 0, 2, 0, 2, 0],
    [0, 0, 0, 0, 2, 2, 2, 0],
    [0, 2, 2, 2, 2, 2, 2, 0],
    [2, 0, 2, 2, 2, 2, 0, 0],
    [2, 0, 0, 2, 2, 0, 0, 0],
    [3, 3, 3, 3, 3, 3, 3, 0],
    [0, 3, 3, 3, 3, 3, 3, 3],
    [0, 0, 3, 3, 3, 0, 3, 3],
    [3, 0, 3, 3, 3, 0, 3, 3],
    [1, 0, 1, 1, 1, 1, 0, 0],
    [1, 1, 0, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 1],
], dtype=jnp.uint8)

PLAYER_JUMP_RIGHT = jnp.array([
    [0, 0, 1, 1, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 2, 2, 0, 0, 0],
    [0, 2, 0, 2, 0, 0, 0, 0],
    [0, 2, 0, 2, 2, 2, 2, 0],
    [0, 2, 0, 2, 0, 2, 2, 2],
    [0, 2, 2, 2, 0, 0, 0, 0],
    [0, 2, 2, 2, 2, 2, 2, 0],
    [0, 0, 2, 2, 2, 2, 0, 2],
    [0, 0, 0, 2, 2, 0, 0, 2],
    [0, 3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 0],
    [3, 3, 0, 3, 3, 3, 0, 0],
    [3, 3, 0, 3, 3, 3, 0, 3],
    [0, 0, 1, 1, 1, 1, 0, 1],
    [0, 0, 1, 1, 1, 0, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0],
], dtype=jnp.uint8)

FIREBALL_1 = jnp.array([
    [0,0,0,6,6,0,0],
    [0,0,0,0,0,6,0],
    [0,6,6,6,0,0,0],
    [0,6,6,6,0,6,6],
    [6,6,6,6,6,0,0],
    [6,6,6,6,6,0,6],
    [6,6,6,6,6,0,6],
    [6,6,6,6,0,6,6],
    [6,6,6,6,6,6,6],
    [0,6,6,6,6,6,0],
    [0,6,6,6,6,6,0],
    [0,0,6,6,6,0,0],
], dtype=jnp.uint8)

FIREBALL_2 = jnp.array([
    [0,0,6,6,0,0,0,0],
    [0,6,6,6,6,0,0,0],
    [6,6,6,6,6,0,0,0],
    [6,6,6,6,6,6,0,0],
    [6,6,6,6,6,6,0,6],
    [6,6,6,6,6,6,0,0],
    [6,6,6,6,6,6,0,0],
    [6,6,6,6,6,6,0,6],
    [6,6,0,6,6,0,0,6],
    [0,6,6,0,0,0,6,0],
    [0,6,6,6,6,6,6,0],
    [0,0,6,6,6,0,0,0],
], dtype=jnp.uint8)

FIREBALL_3 = jnp.array([
    [0,0,6,6,6,6,0,0],
    [0,6,6,0,6,6,6,0],
    [0,0,0,0,0,6,6,0],
    [6,0,0,0,0,0,6,6],
    [6,0,0,6,6,0,6,6],
    [0,0,6,6,6,6,0,6],
    [6,0,6,6,6,6,6,6],
    [0,0,6,6,6,6,6,6],
    [0,0,6,6,6,6,6,6],
    [0,0,6,6,6,6,6,0],
    [0,0,6,6,6,6,6,0],
    [0,0,0,6,6,6,0,0],
], dtype=jnp.uint8)

FIREBALL_4 = jnp.array([
    [0,0,0,6,6,0,0],
    [0,6,6,6,6,6,0],
    [0,6,6,6,6,6,0],
    [0,6,0,6,6,6,6],
    [6,6,0,6,6,6,6],
    [6,0,6,6,6,6,6],
    [6,0,6,6,6,6,6],
    [6,0,6,6,6,6,6],
    [6,6,0,6,6,6,0],
    [0,6,0,0,0,0,0],
    [0,6,6,6,0,0,0],
    [0,0,0,6,6,6,0],
], dtype=jnp.uint8)


# --- Enemies params ---
ENEMY_SIZE = (8, 8)  # w, h
ENEMY_HOLD    = jnp.int32(16) # number of frames to hold each enemy walking frame

ENEMY_WALK_RIGHT_1= jnp.array([
    [0,0,0,0,0,0,5,0],
    [0,0,0,0,0,5,0,5],
    [0,0,0,0,0,5,5,0],
    [0,0,0,0,0,0,5,5],
    [0,0,5,5,0,0,5,0],
    [0,5,5,5,5,0,5,0],
    [5,5,5,5,5,5,5,0],
    [5,5,5,5,5,5,0,0],
    [0,5,0,0,5,0,0,0],
    [0,5,0,0,5,0,0,0],
], dtype=jnp.uint8)

ENEMY_WALK_RIGHT_2 = jnp.array([
    [0,0,0,0,0,0,5,0],
    [0,0,0,0,0,5,0,5],
    [0,0,5,5,0,5,5,5],
    [0,5,5,5,5,0,5,0],
    [5,5,5,5,5,5,0,0],
    [5,5,5,5,5,5,0,0],
    [0,5,5,5,5,0,0,0],
    [0,5,0,0,5,0,0,0],
    [5,0,0,5,0,0,0,0],
], dtype=jnp.uint8)

ENEMY_WALK_LEFT_1 = jnp.array([
    [0,5,0,0,0,0,0,0],
    [5,0,5,0,0,0,0,0],
    [0, 5, 5, 0, 0, 0, 0, 0],
    [5, 5, 0, 0, 0, 0, 0, 0],
    [0, 5, 0, 0, 5, 5, 0, 0],
    [0, 5, 0, 5, 5, 5, 5, 0],
    [0, 5, 5, 5, 5, 5, 5, 5],
    [0, 0, 5, 5, 5, 5, 5, 5],
    [0, 0, 0, 5, 0, 0, 5, 0],
    [0, 0, 0, 5, 0, 0, 5, 0]], jnp.uint8)

ENEMY_WALK_LEFT_2 = jnp.array([
    [0, 5, 0, 0, 0, 0, 0, 0],
       [5, 0, 5, 0, 0, 0, 0, 0],
       [5, 5, 5, 0, 5, 5, 0, 0],
       [0, 5, 0, 5, 5, 5, 5, 0],
       [0, 0, 5, 5, 5, 5, 5, 5],
       [0, 0, 5, 5, 5, 5, 5, 5],
       [0, 0, 0, 5, 5, 5, 5, 0],
       [0, 0, 0, 5, 0, 0, 5, 0],
       [0, 0, 0, 0, 5, 0, 0, 5]], jnp.uint8)

ENEMY_STUNNED_1 = jnp.array([
    [0,0,0,0,5,0,0,0],
    [0,0,0,0,5,0,0,0],
    [0,5,0,0,5,0,0,0],
    [0,5,0,0,5,0,0,0],
    [0,5,0,5,5,0,5,5],
    [0,5,5,5,5,5,5,0],
    [5,5,5,5,0,5,0,5],
    [5,5,5,5,5,0,5,0],
    [5,5,5,5,5,0,0,0],
    [0,5,5,5,0,0,0,0],
], dtype=jnp.uint8)

ENEMY_STUNNED_2 = jnp.array([
    [0,0,5,5,5,0,0,0],
    [0,0,5,0,5,0,0,0],
    [0,0,0,5,0,0,0,0],
    [5,0,0,5,0,0,5,0],
    [0,0,5,5,5,0,0,0],
    [0,5,5,5,5,5,0,0],
    [0,5,5,5,5,5,0,0],
    [0,5,5,5,5,5,0,0],
    [0,0,5,5,5,0,0,0],
    [0,0,5,0,5,0,0,0],
    [0,0,5,0,5,0,0,0],
    [0,5,5,0,5,5,0,0],
], dtype=jnp.uint8)


# --- Fireball params ---
FIREBALL_SIZE = (9, 14)
FIREBALL_MOVEMENT_Start = jnp.array([32, 32, 32, 24], dtype=jnp.float32)
FIREBALL_MOVEMENT = jnp.array([4, 6, 6], dtype=jnp.float32)
FIREBALL_RESTART = 120
FIREBALL_Y = jnp.array([175 - 28, 135 - 28, 95 - 28, 57 - 28])
FIREBALL_X = jnp.array([4, 142])
FIREBALL_INIT_XY = jnp.array([FIREBALL_X[1], FIREBALL_Y[1]])
FIREBALL_DIR = jnp.array([1, -1])
FIREBALL_HOLD = jnp.int32(4)  # change to taste

# --- Object Colors ---
PLAYER_COLOR = jnp.array([0, 255, 0], dtype=jnp.uint8)
ENEMY_COLOR = jnp.array([255, 0, 0], dtype=jnp.uint8)
FIREBALL_COLOR = jnp.array([255, 255, 0], dtype=jnp.uint8)
PLATFORM_COLOR = jnp.array([228, 111, 111], dtype=jnp.uint8)
GROUND_COLOR = jnp.array([181, 83, 40], dtype=jnp.uint8)
POW_COLOR = jnp.array([201, 164, 74], dtype=jnp.uint8)

# --- Platform params---
PLATFORMS = jnp.array([
    [0, 175, 160, 23],  # Ground x, y, w, h
    [0, 57, 64, 3],  # 3.FLoor Left
    [96, 57, 68, 3],  # 3.Floor Right
    [31, 95, 97, 3],  # 2.Floor Middle
    [0, 95, 16, 3],  # 2.Floor Left
    [144, 95, 18, 3],  # 2.Floor Right
    [0, 135, 48, 3],  # 1.Floor Left
    [112, 135, 48, 3]  # 1.Floor Right
])

# --- Pow_Block params ---
POW_BLOCK = jnp.array([[72, 141, 16, 7]])  # x, y, w, h

# Each number maps to 7-segment display (a,b,c,d,e,f,g)
DIGIT_SEGMENTS = jnp.array([
    [1,1,1,1,1,1,0],  # 0
    [0,1,1,0,0,0,0],  # 1
    [1,1,0,1,1,0,1],  # 2
    [1,1,1,1,0,0,1],  # 3
    [0,1,1,0,0,1,1],  # 4
    [1,0,1,1,0,1,1],  # 5
    [1,0,1,1,1,1,1],  # 6
    [1,1,1,0,0,0,0],  # 7
    [1,1,1,1,1,1,1],  # 8
    [1,1,1,1,0,1,1],  # 9
], dtype=jnp.int32)


class Fireball(NamedTuple):
    pos: jnp.ndarray
    start_pat: jnp.ndarray
    move_pat: jnp.ndarray
    count: chex.Array
    state: chex.Array
    dir: chex.Array
    rnd: chex.Array
    ani: chex.Array

class Enemy(NamedTuple):
    enemy_pos: jnp.ndarray  # shape (N,2): x/y positions
    enemy_vel: jnp.ndarray  # shape (N,2): x/y velocities
    enemy_platform_idx: jnp.ndarray  # shape (N,): index of current platform the enemy is on
    enemy_timer: jnp.ndarray  # shape (N,): frame count until next patrol/teleport decision
    enemy_weak_timer: jnp.ndarray  # shape (N,): frames remaining while weak (counts down from 776)
    enemy_initial_sides: jnp.ndarray  # shape (N,): 0=spawned on left, 1=spawned on right
    enemy_active: jnp.ndarray  # shape (N,), int32: 1=active/spawned, 0=inactive slot
    enemy_init_positions: jnp.ndarray  # shape (N,2): prototype spawn positions for enemies (use [0]=right, [1]=left)
    enemy_status: jnp.ndarray  # shape (N,)

class GameState(NamedTuple):  # Enemy movement + global game fields
    enemy: Enemy
    pow_block_counter: int
    score: int
    fireball: Fireball
    game_over: bool
    next_spawn_side: jnp.int32  # 1 = right next, 0 = left next

class PlayerState(NamedTuple):  # Player movement
    pos: jnp.ndarray
    vel: jnp.ndarray
    on_ground: bool
    jump_phase: chex.Array
    ascend_frames: chex.Array
    idx_right: chex.Array
    idx_left: chex.Array
    jump: chex.Array
    walk_anim: chex.Array
    is_walking: chex.Array
    move: chex.Array
    jumpL: chex.Array
    jumpR: chex.Array
    last_dir: chex.Array
    brake_frames_left: chex.Array
    bumped_idx: chex.Array
    pow_bumped: chex.Array
    safe: bool



class MarioBrosState(NamedTuple):
    player: PlayerState
    game: GameState
    lives: chex.Array


class MarioBrosObservation(NamedTuple):  # Copied from jax_kangaroo.py ln.166-168
    player_pos: chex.Array
    player_on_ground: chex.Array
    player_brake_frames_left: chex.Array
    player_lives: chex.Array
    enemy_active: chex.Array
    enemy_pos: chex.Array
    enemy_state: chex.Array
    fireball_pos: chex.Array
    fireball_dir: chex.Array
    pow_block_counter: chex.Array
    pow_block_pos: chex.Array
    plattforms_pos: chex.Array


class MarioBrosInfo(NamedTuple):  # Copied from jax_kangaroo.py ln.186-187
    score: chex.Array

class MarioBrosConstants(NamedTuple):
    SCREEN_WIDTH: int = 160
    SCREEN_HEIGHT: int = 210


def check_collision(pos: jnp.ndarray, vel: jnp.ndarray, platforms: jnp.ndarray, pow_block: jnp.ndarray, pow_block_counter:int):
    x, y = pos
    vx, vy = vel
    w, h = PLAYER_SIZE

    left, right = x, x + w
    top, bottom = y, y + h

    # Platforms
    px, py, pw, ph = platforms[:, 0], platforms[:, 1], platforms[:, 2], platforms[:, 3]
    p_left, p_right = px, px + pw
    p_top, p_bottom = py, py + ph

    overlap_x = (right > p_left) & (left < p_right)
    overlap_y = (bottom > p_top) & (top < p_bottom)
    collided = overlap_x & overlap_y

    prev_bottom = bottom - vy
    next_bottom = bottom

    landed = collided & (vy > 0) & (bottom - vy <= p_top)
    landing_cross = overlap_x & (vy > 0) & (prev_bottom <= p_top) & (next_bottom >= p_top)
    EPS = 1.0
    resting_touch = overlap_x & (vy == 0) & (jnp.abs(bottom - p_top) <= EPS)
    on_platform_mask = landing_cross | resting_touch
    on_platform = jnp.any(on_platform_mask)
    bumped = collided & (vy < 0) & (top - vy >= p_bottom)

    # Landing and bump height adjustments
    landing_y = jnp.where(on_platform_mask, p_top - h, jnp.inf)
    bumping_y = jnp.where(bumped, p_bottom, -jnp.inf)
    new_y_land = jnp.min(landing_y)
    new_y_bump = jnp.max(bumping_y)

    # POW block (only bump from below)
    pow_x, pow_y, pow_w, pow_h = pow_block[0]
    pow_left, pow_right = pow_x, pow_x + pow_w
    pow_top, pow_bottom = pow_y, pow_y + pow_h

    pow_overlap_x = (right > pow_left) & (left < pow_right)

    def compute_pow_bump(_):
        # only run when counter > 0
        return (
            pow_overlap_x &
            (vy < 0) &
            (top - vy >= pow_bottom) &
            (top <= pow_bottom)
        )
    
    # if counter ≤ 0, always False
    pow_bumped = lax.cond(
        pow_block_counter > 0,
        compute_pow_bump,
        lambda _: False,
        operand=None,
    )
    
    pow_bump_y = jnp.where(pow_bumped, pow_bottom, -jnp.inf)
    pow_y_new = jnp.max(pow_bump_y)

    # Calculate bumped_idx (index of bumped platform or -1 if none)
    bumped_indices = jnp.where(bumped, jnp.arange(len(bumped)), 1_000_000)
    min_idx = jnp.min(bumped_indices)
    bumped_idx = jnp.where(min_idx == 1_000_000, -1, min_idx)

    return (jnp.any(on_platform),
            jnp.any(landed),
            jnp.any(bumped | pow_bumped),
            new_y_land,
            jnp.maximum(new_y_bump, pow_y_new),
            pow_bumped,
            bumped_idx)

def check_enemy_collision(player_pos, enemy_pos):
    # checks Player collision with enemys

    px, py = player_pos
    pw, ph = PLAYER_SIZE
    ex, ey = enemy_pos[:, 0], enemy_pos[:, 1]
    ew, eh = ENEMY_SIZE

    overlap_x = (px < ex + ew) & (px + pw > ex)
    overlap_y = (py < ey + eh) & (py + ph > ey)
    return jnp.any(overlap_x & overlap_y)

def fireball_new_start_pos(rng):
    rng, subkey1 = jax.random.split(rng)
    height = jax.random.randint(subkey1, (), 0, 4)

    rng, subkey2 = jax.random.split(rng)
    side = jax.random.randint(subkey2, (), 0, 2)
    return jnp.array([FIREBALL_X[side], FIREBALL_Y[height]]), FIREBALL_DIR[side], rng

def fireball_step(fb:Fireball):
    def start(f: Fireball):
        arr = f.start_pat
        first_nonzero_index = jnp.argmax(arr != 0)
        arr = arr.at[first_nonzero_index].set(arr[first_nonzero_index] - 1)
        new_x = f.pos[0] + f.dir
        last_is_zero = arr[first_nonzero_index] == 0
    
        new_pos= jnp.array([jnp.where(arr[first_nonzero_index] == 0, new_x, f.pos[0]), f.pos[1]]) 
        new_state= jnp.where(last_is_zero, 1, 0)
        new_start_pat= jnp.where(last_is_zero, FIREBALL_MOVEMENT_Start, arr)
        return Fireball(
            pos= new_pos,
            start_pat= new_start_pat,
            move_pat= f.move_pat,
            count= f.count,
            state= new_state,
            dir= f.dir,
            rnd= f.rnd,
            ani= f.ani
        )
        
    def move(f: Fireball):
        arr = f.move_pat
        first_nonzero_index = jnp.argmax(arr != 0)
        arr = arr.at[first_nonzero_index].set(arr[first_nonzero_index] - 1)
        new_x = f.pos[0] + f.dir
        last_is_zero = arr[first_nonzero_index] == 0
        
        new_pos= jnp.array([jnp.where(arr[first_nonzero_index] == 0, new_x, f.pos[0]), f.pos[1]])
        new_state = jnp.where(
            (new_pos[0] < 4) | (new_pos[0] > SCREEN_WIDTH - 18),
            2,  # despawn
            1   # bewegen
        )
        new_move_pat= jnp.where(last_is_zero, FIREBALL_MOVEMENT, arr)
        changed = jnp.any(f.pos != new_pos)
        return Fireball(
            pos= new_pos,
            start_pat= f.start_pat,
            move_pat= new_move_pat,
            count= f.count,
            state= new_state,
            dir= f.dir,
            rnd= f.rnd,
            ani= lax.cond(changed, lambda _: (f.ani % 3) + 1, lambda _: f.ani, operand=None)
        )
    def wait(f: Fireball):
        def stay(ff: Fireball):
            return Fireball(
            pos= ff.pos,
            start_pat= ff.start_pat,
            move_pat= ff.move_pat,
            count= ff.count - 1,
            state= ff.state,
            dir= ff.dir,
            rnd= ff.rnd,
            ani= ff.ani
        )
        def end(ff: Fireball):
            new_pos, new_dir, rng = fireball_new_start_pos(ff.rnd)
            return Fireball(
            pos= new_pos,
            start_pat= ff.start_pat,
            move_pat= ff.move_pat,
            count= 60,
            state= 0,
            dir= new_dir,
            rnd= rng,
            ani= ff.ani
        )
        return lax.cond(f.count > 0, stay, end, f)
    return lax.switch(fb.state, [start, move, wait], fb)

@jax.jit
def enemy_step(
    enemy_pos,
    enemy_vel,
    enemy_platform_idx,
    enemy_timer,
    platforms,
    initial_sides,
    active_mask,
    init_positions,
    enemy_status,   # added status
):
    ew, eh = ENEMY_SIZE
    pw = platforms[:, 2]
    enemy_init_platform_idx = jnp.array([1, 2, 1])  # same as initial enemy_platform_idx
    TELEPORT_DELAY_FRAMES = 5 * 60  # e.g. 5 seconds at 60 FPS; adjust as needed


    def step_enemy(pos, vel, p_idx, timer, side, i, status):
        x, y = pos
        vx, vy = vel
        # Handle dead (status == 3) enemy with respawn delay
        # Handle dead (status == 3) enemy -> permanently offscreen
        def dead_logic():
            invisible_pos = jnp.array([-1000.0, -1000.0])
            invisible_vel = jnp.array([0.0, 0.0])
            return invisible_pos, invisible_vel, p_idx, timer, side, status

        # If enemy is weak (status==1) or dead (status==3), velocity is zero
        is_weak_or_dead = (status == 1) | (status == 3)
        vx = jnp.where(is_weak_or_dead, 0.0, vx)
        vy = jnp.where(is_weak_or_dead, 0.0, vy)

        # Fix: If recovering from weak state (status == 2), resume movement
        recovering = status == 2
        vx = jnp.where(
            recovering & (vx == 0.0),
            jnp.where(side == 1, -ENEMY_MOVE_SPEED, ENEMY_MOVE_SPEED),
            vx
        )

        plat = platforms[p_idx]
        plat_x, plat_y, plat_w, plat_h = plat
        plat_left = plat_x
        plat_right = plat_x + plat_w

        EPS = 0.5
        at_left_edge = x <= plat_left + EPS
        at_right_edge = x + ew >= plat_right - EPS
        at_edge = at_left_edge | at_right_edge

        # Check platform below at next x position (x + vx)
        def platform_below_y(pos_x):
            px, py, pw, ph = platforms[:, 0], platforms[:, 1], platforms[:, 2], platforms[:, 3]
            left = px
            right = px + pw
            supported_x = (pos_x + ew > left) & (pos_x < right)
            below_y = py > y
            candidates = jnp.where(supported_x & below_y, py, jnp.inf)
            return jnp.min(candidates)

        min_platform_below_y = platform_below_y(x + vx)
        has_platform_below = min_platform_below_y != jnp.inf

        def teleport_down(pos_x, vx, side, ew, eh, platforms, min_platform_below_y):
            same_y_mask = platforms[:, 1] == min_platform_below_y
            plat_lefts = platforms[:, 0]
            plat_rights = platforms[:, 0] + platforms[:, 2]
            inside_x_mask = (pos_x + ew > plat_lefts) & (pos_x < plat_rights)
            valid_mask = same_y_mask & inside_x_mask

            idx_below = jnp.argmax(valid_mask.astype(jnp.int32))

            num_enemies = enemy_platform_idx.shape[0]
            indices = jnp.arange(num_enemies, dtype=jnp.int32)
            blocked = jnp.any((enemy_platform_idx == idx_below) & (active_mask != 0) & (indices != i))

            def blocked_branch():
                new_pos_x = jnp.array(x, dtype=jnp.float32)
                new_pos_y = jnp.array(y, dtype=jnp.float32)
                new_vx_down = jnp.array(-vx, dtype=jnp.float32)
                new_timer = timer + 1
                new_side = jnp.array(1 - side, dtype=jnp.int32)
                return (
                    jnp.array([new_pos_x, new_pos_y]),
                    jnp.array([new_vx_down, vy]),
                    p_idx,
                    new_timer,
                    new_side,
                    status,
                )

            def fall_branch():
                new_pos_x = jnp.array(pos_x, dtype=jnp.float32)
                new_pos_y = jnp.array(platforms[idx_below, 1] - eh, dtype=jnp.float32)
                new_vx_down = vx
                new_timer = jnp.array(0, dtype=jnp.int32)
                new_side = jnp.array(side, dtype=jnp.int32)
                return (
                    jnp.array([new_pos_x, new_pos_y]),
                    jnp.array([new_vx_down, vy]),
                    idx_below,
                    new_timer,
                    new_side,
                    status,
                )

            return jax.lax.cond(blocked, blocked_branch, fall_branch)

        def stay_on():
            new_x = x + vx

            # Wrap around screen
            new_x = jnp.where(new_x < 0, SCREEN_WIDTH - ew, new_x)
            new_x = jnp.where(new_x + ew > SCREEN_WIDTH, 0.0, new_x)

            return (
                jnp.array([new_x, y]),
                jnp.array([vx, vy]),
                p_idx,
                timer + 1,
                side,
                status,
            )

        # Handle dead status first (so it overrides normal move)
        # Handle dead status first (so it overrides normal move)
        result = jax.lax.cond(
            status == 3,
            dead_logic,
            lambda: (
                jax.lax.cond(
                    (at_edge & has_platform_below) & (p_idx == 0),
                    # If on ground floor and reach edge -> vanish (mark dead)
                    lambda: (
                        jnp.array([-1000.0, -1000.0]),
                        jnp.array([0.0, 0.0]),
                        p_idx,
                        timer,
                        side,
                        jnp.array(3, dtype=jnp.int32)  # permanently dead
                    ),
                    lambda: jax.lax.cond(
                        at_edge & has_platform_below,
                        lambda: teleport_down(x + vx, vx, side, ew, eh, platforms, min_platform_below_y),
                        stay_on
                    ),
                )
            )
        )

        return result

    def conditional_step(pos, vel, idx, timer, side, i, active, status):
        return jax.lax.cond(
            active,
            lambda _: step_enemy(pos, vel, idx, timer, side, i, status),
            lambda _: (pos, vel, idx, timer, side, status),
            operand=None
        )

    indices = jnp.arange(enemy_pos.shape[0], dtype=jnp.int32)

    new_pos, new_vel, new_idx, new_timer, new_sides, new_status = jax.vmap(
        conditional_step,
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0)
    )(enemy_pos, enemy_vel, enemy_platform_idx, enemy_timer, initial_sides, indices, active_mask, enemy_status)

    return new_pos, new_vel, new_idx, new_timer, new_sides, new_status


@jax.jit
def movement(state: PlayerState, game_state:GameState) -> PlayerState:    # Calculates movement of Player based on given state and action taken
    
    def normal(x):

        move = state.move * MOVE_SPEED
        jump_btn = state.jump
        vx = move
        # -------- phase / frame bookkeeping --------------------------
        start_jump = (jump_btn == 1) & state.on_ground & (state.jump_phase == 0)

        jump_phase = jnp.where(start_jump, 1, state.jump_phase)
        asc_left = jnp.where(start_jump, ASCEND_FRAMES, state.ascend_frames)

        # vertical speed for this frame
        vy = jnp.where(
            jump_phase == 1, ASCEND_VY,
            jnp.where(jump_phase == 2, DESCEND_VY,
                    jnp.where(state.on_ground, 0.0, DESCEND_VY))
        )

        # integrate position
        new_pos = state.pos + jnp.array([vx, vy])

        on_platform, landed, bumped, y_land, y_bump, pow_bumped, bumped_idx = check_collision(new_pos, jnp.array([vx, vy]), PLATFORMS,
                                                                                POW_BLOCK, game_state.pow_block_counter)

        new_y = jnp.where(on_platform, y_land,
                        jnp.where(bumped, y_bump, new_pos[1]))

        # ---------- update phases after collision & time -------------
        # decrement ascend frames while ascending
        asc_left = jnp.where(jump_phase == 1, jnp.maximum(asc_left - 1, 0), asc_left)
        # switch to descend when ascent finished
        jump_phase = jnp.where((jump_phase == 1) & (asc_left == 0), 2, jump_phase)
        # head bump → descend immediately
        jump_phase = jnp.where(bumped & (vy < 0), 2, jump_phase)
        asc_left = jnp.where(bumped & (vy < 0), 0, asc_left)
        # landing → reset
        jump_phase = jnp.where(on_platform, 0, jump_phase)
        asc_left = jnp.where(on_platform, 0, asc_left)
        # walked off ledge → fall
        jump_phase = jnp.where((jump_phase == 0) & (~on_platform), 2, jump_phase)

        vy_final = jnp.where(
            jump_phase == 1, ASCEND_VY,
            jnp.where(jump_phase == 2, DESCEND_VY, 0.0)
        )

        #new_x = jnp.clip(new_pos[0], 0, SCREEN_WIDTH - PLAYER_SIZE[0])
        ew = PLAYER_SIZE[0]  # enemy/player width
        new_x = jnp.where(new_pos[0] < 0, SCREEN_WIDTH - ew, new_pos[0])
        new_x = jnp.where(new_x + ew > SCREEN_WIDTH, 0.0, new_x)

        return PlayerState(
            pos=jnp.array([new_x, new_y]),
            vel=jnp.array([vx, vy_final]),
            on_ground=on_platform,
            jump_phase=jump_phase.astype(jnp.int32),
            ascend_frames=asc_left.astype(jnp.int32),
            idx_right=state.idx_right,
            idx_left=state.idx_left,
            jump=state.jump,
            walk_anim=state.walk_anim,
            is_walking=state.is_walking,
            move=state.move,
            jumpL=state.jumpL,
            jumpR=state.jumpR,
            last_dir=state.last_dir,
            brake_frames_left=state.brake_frames_left,
            bumped_idx=bumped_idx,
            pow_bumped=pow_bumped,
            safe= state.safe
        )
    return lax.cond(state.safe, lambda x: x, normal , state)




@jax.jit
def player_step(state: PlayerState, action: chex.Array, game_state: GameState) -> PlayerState:
    def death_step(state: PlayerState) -> PlayerState:
        press_fire = (action == Action.FIRE) | (action == Action.LEFTFIRE) | (action == Action.RIGHTFIRE)
        press_right = (action == Action.RIGHT) | (action == Action.RIGHTFIRE)
        press_left = (action == Action.LEFT) | (action == Action.LEFTFIRE)

        cond = (press_fire | press_right | press_left) & jnp.all(state.pos == PLAYER_RESPWAN_XY)
        state = state._replace(safe=jnp.where(cond, False, state.safe))

        
        pos = state.pos
        pos = jnp.array([pos[0], pos[1] + DESCEND_VY], dtype=jnp.float32)
        new_pos = lax.cond(
            pos[1] > SCREEN_HEIGHT,
            lambda _: PLAYER_RESPWAN_XY,
            lambda _: pos,
            operand=None
        )
        state = state._replace(pos= new_pos)
        
        return state

    def step(state: PlayerState) -> PlayerState:
        # 1) decode buttons
        press_fire = (action == Action.FIRE) | (action == Action.LEFTFIRE) | (action == Action.RIGHTFIRE)
        press_right = (action == Action.RIGHT) | (action == Action.RIGHTFIRE)
        press_left = (action == Action.LEFT) | (action == Action.LEFTFIRE)

        state = state._replace(safe=jnp.where((press_fire | press_right | press_left), False, state.safe))
        # 2) reset horizontal/jump input on ground
        state0 = lax.cond(
            state.on_ground,
            lambda s: s._replace(move=0.0, jump=0, jumpL=False, jumpR=False),
            lambda s: s,
            state
        )


        # 3) only allow inputs when not jumping(player stutters when state.on_ground is used for check)
        press_fire = press_fire & (state0.jump == 0)
        press_right = press_right & (state0.jump == 0)
        press_left = press_left & (state0.jump == 0)
        

        # 3) set jump flag
        state1 = state0._replace(jump=jnp.where(press_fire, 1, state0.jump))

        # 4) walking/braking vs. jumping
        def walk_or_brake(s):
            # move right
            def mr(ss):
                return ss._replace(
                    move=movement_pattern[ss.idx_right],
                    idx_right=(ss.idx_right + 1) % pat_len,
                    idx_left=0,
                    last_dir=1,
                    brake_frames_left=0
                )

            # move left
            def ml(ss):
                return ss._replace(
                    move=-movement_pattern[ss.idx_left],
                    idx_left=(ss.idx_left + 1) % pat_len,
                    idx_right=0,
                    last_dir=-1,
                    brake_frames_left=0
                )

            # apply brake
            def br(ss):
                ss2 = ss._replace(
                    brake_frames_left=jnp.where((jnp.abs(ss.move) > 0) & (ss.brake_frames_left == 0),
                                                BRAKE_DURATION, ss.brake_frames_left)
                )

                def do_brake(x):
                    nb = x.brake_frames_left - 1
                    return x._replace(
                        move=x.last_dir * BRAKE_SPEED,
                        brake_frames_left=nb,
                        last_dir=x.last_dir
                    )

                return lax.cond(ss2.brake_frames_left > 0, do_brake, lambda x: x._replace(move=0.0), ss2)

            return lax.cond(press_right, mr,
                            lambda ss: lax.cond(press_left, ml, br, ss), s)

        def jump_move(s):
            # mid-air jumping momentum
            def jr(ss):
                return ss._replace(
                    move=movement_pattern[ss.idx_right],
                    idx_right=(ss.idx_right + 1) % pat_len,
                    idx_left=0,
                    jumpR=True,
                    brake_frames_left=0,
                    last_dir=1
                )

            def jl(ss):
                return ss._replace(
                    move=-movement_pattern[ss.idx_left],
                    idx_left=(ss.idx_left + 1) % pat_len,
                    idx_right=0,
                    jumpL=True,
                    brake_frames_left=0,
                    last_dir=-1
                )

            def js(ss):
                return ss._replace(
                    idx_left=0,
                    idx_right=0,
                    brake_frames_left=0,
                    last_dir=ss.last_dir
                )

            condR = press_right | s.jumpR
            condL = press_left | s.jumpL
            return lax.cond(condR, jr,
                            lambda ss: lax.cond(condL, jl, js, ss), s)

        state2 = lax.cond(state1.jump == 0, walk_or_brake, jump_move, state1)

        # 5) apply physics
        s_phys = movement(state2, game_state)

        # 6) stable walk animation state
        walking_intent = (press_left | press_right | (s_phys.brake_frames_left > 0))
        is_walking = walking_intent & s_phys.on_ground & (s_phys.jump == 0)
        next_walk_anim = jnp.where(is_walking, (state.walk_anim + 1) % 8, jnp.int32(0))

        new_state = s_phys._replace(walk_anim=next_walk_anim, is_walking=is_walking)
        return new_state
    cond = state.safe & jnp.any(state.pos != PLAYER_RESPWAN_XY)
    return lax.cond(cond, death_step, step, state)
    


def draw_rect(image, x, y, w, h, color):
    y0 = jnp.clip(jnp.floor(y), 0, SCREEN_HEIGHT - 1).astype(jnp.int32)
    y1 = jnp.clip(jnp.floor(y + h), 0, SCREEN_HEIGHT).astype(jnp.int32)
    x0 = jnp.clip(jnp.floor(x), 0, SCREEN_WIDTH - 1).astype(jnp.int32)
    x1 = jnp.clip(jnp.floor(x + w), 0, SCREEN_WIDTH).astype(jnp.int32)

    mask_y = (jnp.arange(SCREEN_HEIGHT) >= y0) & (jnp.arange(SCREEN_HEIGHT) < y1)
    mask_x = (jnp.arange(SCREEN_WIDTH) >= x0) & (jnp.arange(SCREEN_WIDTH) < x1)
    mask = jnp.outer(mask_y, mask_x)

    color_arr = jnp.array(color, dtype=image.dtype).reshape(1, 1, 3)
    new_image = jnp.where(mask[:, :, None], color_arr, image)
    return new_image

def draw_digit(img, digit, x, y, size=6, color=(255, 255, 255)):
    seg = DIGIT_SEGMENTS[digit]

    def segment_rect(index):
        w, h = size, size // 2
        return [
            (x + 0, y + 0, w, h),                  # a
            (x + w, y + 0, h, w),                  # b
            (x + w, y + w, h, w),                  # c
            (x + 0, y + 2 * w, w, h),              # d
            (x - h, y + w, h, w),                  # e
            (x - h, y + 0, h, w),                  # f
            (x + 0, y + w, w, h),                  # g
        ][index]

    for i in range(7):
        img = lax.cond(
            seg[i] == 1,
            lambda im: draw_rect(im, *segment_rect(i), color),
            lambda im: im,
            img
        )
    return img


def indices_to_rgb_and_mask(idx, palette):
    # idx: (H, W) uint8, 0=transparent
    mask = idx != 0
    clamped = jnp.clip(idx, 0, palette.shape[0] - 1)
    rgb = palette[clamped]            # (H, W, 3)
    return rgb, mask

import jax.numpy as jnp
from jax import lax

def draw_sprite_rgb(img, x, y, sprite_rgb, sprite_mask):
    """
    Blit a colored sprite with transparency.
    img: (H,W,3) uint8
    x,y: top-left screen coords (floats ok)
    sprite_rgb: (h,w,3) uint8
    sprite_mask: (h,w) bool
    """
    H, W, _ = img.shape
    h, w, _ = sprite_rgb.shape

    # integer destination box (still traced scalars)
    x0 = jnp.clip(jnp.floor(x), 0, W - 1).astype(jnp.int32)
    y0 = jnp.clip(jnp.floor(y), 0, H - 1).astype(jnp.int32)
    x1 = jnp.clip(jnp.floor(x) + w, 0, W).astype(jnp.int32)
    y1 = jnp.clip(jnp.floor(y) + h, 0, H).astype(jnp.int32)

    # traced boolean, OK for lax.cond (but not for Python if/or)
    valid = (x1 > x0) & (y1 > y0)

    def do_draw(im):
        yy = jnp.arange(H)[:, None]
        xx = jnp.arange(W)[None, :]

        in_x = (xx >= x0) & (xx < x1)
        in_y = (yy >= y0) & (yy < y1)
        in_box = in_x & in_y

        # map dest → source indices
        src_x = (xx - x0).clip(0, w - 1)
        src_y = (yy - y0).clip(0, h - 1)

        src_rgb = sprite_rgb[src_y, src_x]
        src_a = sprite_mask[src_y, src_x]  # bool

        m = in_box & src_a
        return jnp.where(m[:, :, None], src_rgb, im)

    # if not valid, return img unchanged — but do it with JAX
    return lax.cond(valid, do_draw, lambda im: im, img)

PLAYER_STANDING_RIGHT_RGB, PLAYER_STANDING_RIGHT_MASK = indices_to_rgb_and_mask(PLAYER_STANDING_RIGHT, PALETTE)
PLAYER_STANDING_LEFT_RGB, PLAYER_STANDING_LEFT_MASK = indices_to_rgb_and_mask(PLAYER_STANDING_LEFT, PALETTE)
PLAYER_WALK_RIGHT_1_RGB, PLAYER_WALK_RIGHT_1_MASK = indices_to_rgb_and_mask(PLAYER_WALK_RIGHT_1, PALETTE)
PLAYER_WALK_RIGHT_2_RGB, PLAYER_WALK_RIGHT_2_MASK = indices_to_rgb_and_mask(PLAYER_WALK_RIGHT_2, PALETTE)
PLAYER_WALK_LEFT_1_RGB, PLAYER_WALK_LEFT_1_MASK = indices_to_rgb_and_mask(PLAYER_WALK_LEFT_1, PALETTE)
PLAYER_WALK_LEFT_2_RGB, PLAYER_WALK_LEFT_2_MASK = indices_to_rgb_and_mask(PLAYER_WALK_LEFT_2, PALETTE)
PLAYER_JUMP_LEFT_RGB, PLAYER_JUMP_LEFT_MASK = indices_to_rgb_and_mask(PLAYER_JUMP_LEFT, PALETTE)
PLAYER_JUMP_RIGHT_RGB, PLAYER_JUMP_RIGHT_MASK = indices_to_rgb_and_mask(PLAYER_JUMP_RIGHT, PALETTE)
FIREBALL_1_RGB, FIREBALL_1_MASK = indices_to_rgb_and_mask(FIREBALL_1, PALETTE)
FIREBALL_2_RGB, FIREBALL_2_MASK = indices_to_rgb_and_mask(FIREBALL_2, PALETTE)
FIREBALL_3_RGB, FIREBALL_3_MASK = indices_to_rgb_and_mask(FIREBALL_3, PALETTE)
FIREBALL_4_RGB, FIREBALL_4_MASK = indices_to_rgb_and_mask(FIREBALL_4, PALETTE)
ENEMY_WALK_RIGHT_1_RGB, ENEMY_WALK_RIGHT_1_MASK = indices_to_rgb_and_mask(ENEMY_WALK_RIGHT_1, PALETTE)
ENEMY_WALK_RIGHT_2_RGB, ENEMY_WALK_RIGHT_2_MASK = indices_to_rgb_and_mask(ENEMY_WALK_RIGHT_2, PALETTE)
ENEMY_WALK_LEFT_1_RGB, ENEMY_WALK_LEFT_1_MASK = indices_to_rgb_and_mask(ENEMY_WALK_LEFT_1, PALETTE)
ENEMY_WALK_LEFT_2_RGB, ENEMY_WALK_LEFT_2_MASK = indices_to_rgb_and_mask(ENEMY_WALK_LEFT_2, PALETTE)
ENEMY_STUNNED_1_RGB, ENEMY_STUNNED_1_MASK = indices_to_rgb_and_mask(ENEMY_STUNNED_1, PALETTE)
ENEMY_STUNNED_2_RGB, ENEMY_STUNNED_2_MASK = indices_to_rgb_and_mask(ENEMY_STUNNED_2, PALETTE)




import jax.numpy as jnp
from jax import lax


import jaxatari.rendering.jax_rendering_utils as ru
from jaxatari.renderers import JAXGameRenderer

import jax.numpy as jnp
from jax import lax

# Tunables
WALK_HOLD = jnp.int32(5)   # frames one walking frame is held

def draw_player_by_state(image, p, px, py):
    """
    Show standing/ walking / jumping::
    - Holding RIGHT/LEFT: walk right/left,
    - Jump + RIGHT/LEFT held: use jump sprite for that side.
    - Jump without side held: use jump sprite based on previous standing direction (last_dir).
    - After moving right/left and stopping: show standing_right/standing_left respectively.
    """
    # --- phases ---
    jumping   = (p.jump_phase != 0) & (~p.on_ground)      # in-air state
    walking   = p.is_walking & (~jumping)  
    standing  = (~jumping) & (~walking) & p.on_ground

    # --- interchanging walk frames ---
    walk_frame_is_1 = ((p.walk_anim // WALK_HOLD) % 2) == 0 

    # --- directions ---
    # Use last_dir as primary facing
    vx = p.vel[0]
    vel_sign = jnp.sign(vx).astype(jnp.int32)       # -1, 0, +1
    base_dir = jnp.where(p.last_dir != 0, p.last_dir, vel_sign)

    # Jump direction
    jump_right = p.jumpR | ((~p.jumpR) & (~p.jumpL) & (base_dir == 1))
    jump_left  = p.jumpL | ((~p.jumpR) & (~p.jumpL) & (base_dir == -1))

    # Standing direction
    stand_right = (base_dir != -1)  # right if base_dir==1 or 0
    stand_left  = (base_dir == -1)

    # Walking direction: use last_dir (how you set it on key press)
    walk_right_dir = (base_dir == 1)
    walk_left_dir  = (base_dir == -1)

    # --- sprite branches ---
    def stand_r(im): return draw_sprite_rgb(im, px, py, PLAYER_STANDING_RIGHT_RGB, PLAYER_STANDING_RIGHT_MASK)
    def stand_l(im): return draw_sprite_rgb(im, px, py, PLAYER_STANDING_LEFT_RGB,  PLAYER_STANDING_LEFT_MASK)

    def walk_r1(im): return draw_sprite_rgb(im, px, py, PLAYER_WALK_RIGHT_1_RGB,   PLAYER_WALK_RIGHT_1_MASK)
    def walk_r2(im): return draw_sprite_rgb(im, px, py, PLAYER_WALK_RIGHT_2_RGB,   PLAYER_WALK_RIGHT_2_MASK)
    def walk_l1(im): return draw_sprite_rgb(im, px, py, PLAYER_WALK_LEFT_1_RGB,    PLAYER_WALK_LEFT_1_MASK)
    def walk_l2(im): return draw_sprite_rgb(im, px, py, PLAYER_WALK_LEFT_2_RGB,    PLAYER_WALK_LEFT_2_MASK)

    def jump_r(im): return draw_sprite_rgb(im, px, py, PLAYER_JUMP_RIGHT_RGB,      PLAYER_JUMP_RIGHT_MASK)
    def jump_l(im): return draw_sprite_rgb(im, px, py, PLAYER_JUMP_LEFT_RGB,       PLAYER_JUMP_LEFT_MASK)

    # --- phase dispatch ---
    def draw_jump(im):
        return lax.cond(jump_right, jump_r,
                        lambda im2: lax.cond(jump_left, jump_l, jump_r, im2),
                        im)

    def draw_walk(im):
        def walk_right(im2):
            return lax.cond(walk_frame_is_1, walk_r1, walk_r2, im2)
        def walk_left(im2):
            return lax.cond(walk_frame_is_1, walk_l1, walk_l2, im2)
        return lax.cond(walk_right_dir, walk_right,
                        lambda im3: lax.cond(walk_left_dir, walk_left,
                                             lambda im4: lax.cond(stand_right, stand_r, stand_l, im4),
                                             im3),
                        im)

    def draw_stand(im):
        return lax.cond(stand_right, stand_r, stand_l, im)
    # draw jump if jumping, else walk if walking, else stand
    return lax.cond(jumping, draw_jump,
                    lambda im: lax.cond(walking, draw_walk, draw_stand, im),
                    image)

ENEMY_WEAK_TOTAL = jnp.int32(776)

def draw_enemy_by_state(image, ex, ey, vx, timer, status, weak_timer):
    """
    - status: 1=weak(stunned), 2=strong/recovering, 3=dead
    - weak_timer: frames remaining in weak state (counts down from 776)
    If stunned: show STUNNED_1 for first half, STUNNED_2 for second half.
    Else: animate walk based on horizontal direction.
    """
    # enemy stunned
    def draw_stunned(im):
        first_half = weak_timer > (ENEMY_WEAK_TOTAL // 2)
        def s1(ii): return draw_sprite_rgb(ii, ex, ey, ENEMY_STUNNED_1_RGB, ENEMY_STUNNED_1_MASK)
        def s2(ii): return draw_sprite_rgb(ii, ex, ey-2, ENEMY_STUNNED_2_RGB, ENEMY_STUNNED_2_MASK) # second frame is a bit larger (ey-2)
        return lax.cond(first_half, s1, s2, im)

    # enemy is walking
    frame1 = ((timer // ENEMY_HOLD) % 2) == 0

    def r1(im): return draw_sprite_rgb(im, ex, ey, ENEMY_WALK_RIGHT_1_RGB, ENEMY_WALK_RIGHT_1_MASK)
    def r2(im): return draw_sprite_rgb(im, ex, ey, ENEMY_WALK_RIGHT_2_RGB, ENEMY_WALK_RIGHT_2_MASK)
    def l1(im): return draw_sprite_rgb(im, ex, ey, ENEMY_WALK_LEFT_1_RGB,  ENEMY_WALK_LEFT_1_MASK)
    def l2(im): return draw_sprite_rgb(im, ex, ey, ENEMY_WALK_LEFT_2_RGB,  ENEMY_WALK_LEFT_2_MASK)

    def draw_right(im): return lax.cond(frame1, r1, r2, im)
    def draw_left(im):  return lax.cond(frame1, l1, l2, im)

    dir_right = vx >= 0.0

    def draw_walk(im):
        return lax.cond(dir_right, draw_right, draw_left, im)

    is_stunned = (status == 1)
    return lax.cond(is_stunned, draw_stunned, draw_walk, image)


def draw_fireball(image, f, fx, fy):

    def fireball_1(im): return draw_sprite_rgb(im, fx, fy, FIREBALL_1_RGB, FIREBALL_1_MASK)
    def fireball_2(im): return draw_sprite_rgb(im, fx, fy, FIREBALL_2_RGB, FIREBALL_2_MASK)
    def fireball_3(im): return draw_sprite_rgb(im, fx, fy, FIREBALL_3_RGB, FIREBALL_3_MASK)
    def fireball_4(im): return draw_sprite_rgb(im, fx, fy, FIREBALL_4_RGB, FIREBALL_4_MASK)
    return lax.switch(f.ani - 1, [fireball_1, fireball_2, fireball_3, fireball_4], image)

class MarioBrosRenderer(JAXGameRenderer):
    def __init__(self):
        pass

    def render(self, state: MarioBrosState) -> jnp.ndarray:
        def white_screen(_: any) -> jnp.ndarray:
            return jnp.ones((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=jnp.uint8) * 255
        def continue_render(_: any) -> jnp.ndarray:
            image = jnp.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=jnp.uint8)

            # --- Draw player ---
            px, py = state.player.pos
            player_color = lax.cond(
                state.player.brake_frames_left > 0,
                lambda _: ENEMY_COLOR,
                lambda _: PLAYER_COLOR,
                operand=None
            )

            image = draw_player_by_state(image, state.player, px, py)


            # --- Draw enemies ---
            def draw_enemy(i, img):
                active_i = state.game.enemy.enemy_active[i]

                def do_draw(_):
                    ex, ey = state.game.enemy.enemy_pos[i]
                    vx, vy = state.game.enemy.enemy_vel[i]
                    timer  = state.game.enemy.enemy_timer[i]
                    status = state.game.enemy.enemy_status[i]
                    wtime  = state.game.enemy.enemy_weak_timer[i]
                    return draw_enemy_by_state(img, ex, ey-2, vx, timer, status, wtime)

                def skip_draw(_):
                    return img

                return lax.cond(state.game.enemy.enemy_active[i] == 1, do_draw, skip_draw, operand=None)

            image = lax.fori_loop(0, state.game.enemy.enemy_pos.shape[0], draw_enemy, image)

            # --- Draw fireball ---
            fx, fy = state.game.fireball.pos
            image = draw_fireball(image, state.game.fireball, fx, fy)
            # --- Draw platforms ---
            def draw_platform(i, img):
                plat = PLATFORMS[i]
                color = lax.cond(i == 0, lambda _: GROUND_COLOR, lambda _: PLATFORM_COLOR, operand=None)
                return draw_rect(img, plat[0], plat[1], plat[2], plat[3], color)

            image = lax.fori_loop(0, PLATFORMS.shape[0], draw_platform, image)

            # --- Draw POW block only if hits < 3 ---
            def draw_pow(img):
                x, y, w, h = POW_BLOCK[0]
                return draw_rect(img, x, y, w, h, POW_COLOR)

            image = lax.cond(state.game.pow_block_counter > 0, draw_pow, lambda img: img, image)

            # --- Draw lives ---
            def draw_life(i, img):
                x = 5 + i * 12
                y = 5
                return draw_rect(img, x, y, 10, 10, (228, 111, 111))

            image = lax.fori_loop(0, state.lives, draw_life, image)

            # --- Draw score using 7-segment digits ---
            score = state.game.score
            digit_positions = 5  # Max digits to display

            def draw_score_digit(i, img):
                power = digit_positions - 1 - i
                divisor = 10 ** power
                digit = (score // divisor) % 10
                x = SCREEN_WIDTH - (digit_positions - i) * 12 - 5
                y = 5
                return draw_digit(img, digit, x, y)

            image = lax.fori_loop(0, digit_positions, draw_score_digit, image)

            return image
        
        return lax.cond(state.game.game_over, white_screen, continue_render, operand=None)

class JaxMarioBros(JaxEnvironment[
                       MarioBrosState, MarioBrosObservation, MarioBrosInfo, MarioBrosConstants]):  # copied and adapted from jax_kangaroo.py ln.1671
    # holds reset and main step function

    def __init__(self):
        self.renderer = MarioBrosRenderer()
        self.action_set = [
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
        ]

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))

    def get_action_space(self) -> jnp.ndarray:
        return jnp.array(self.action_set)

    def _get_observation(self, state: MarioBrosState) -> MarioBrosObservation:
        obs = MarioBrosObservation(
            player_pos= state.player.pos,
            player_on_ground= state.player.on_ground,
            player_brake_frames_left= state.player.brake_frames_left,
            player_lives= state.lives,
            enemy_active= state.game.enemy.enemy_active,
            enemy_pos= state.game.enemy.enemy_pos,
            enemy_state= state.game.enemy.enemy_status,
            fireball_pos= state.game.fireball.pos,
            fireball_dir= state.game.fireball.dir,
            pow_block_counter= state.game.pow_block_counter,
            pow_block_pos= POW_BLOCK,
            plattforms_pos= PLATFORMS
        )
        return obs

    def obs_to_flat_array(self, obs: MarioBrosObservation) -> chex.Array:
        return jnp.concatenate(
            [
            obs.plattforms_pos.flatten(),
            obs.player_on_ground.flatten(),
            obs.player_brake_frames_left.flatten(),
            obs.player_lives.flatten(),
            obs.enemy_active.flatten(),
            obs.enemy_pos.flatten(),
            obs.enemy_state.flatten(),
            obs.fireball_pos.flatten(),
            obs.fireball_dir.flatten(),
            obs.pow_block_counter.flatten(),
            obs.pow_block_pos.flatten(),
            obs.plattforms_pos.flatten(),
            ]
        )

    def observation_space(self) -> spaces.Dict:
        return spaces.Dict({
        # --- Player ---
        "player_pos": spaces.Box(low=0, high=jnp.array([SCREEN_WIDTH, SCREEN_HEIGHT]), shape=(2,), dtype=jnp.float32),
        "player_on_ground": spaces.Discrete(2),  # 0=False, 1=True
        "player_brake_frames_left": spaces.Box(low=0, high=20, shape=(), dtype=jnp.int32),  # max Brake Frames ggf. anpassen
        "player_lives": spaces.Box(low=0, high=4, shape=(), dtype=jnp.int32),

        # --- Enemy ---
        "enemy_active": spaces.Box(low=0, high=2, shape=(3,), dtype=jnp.int32),  # 0=inactive,1=active,2=pending
        "enemy_pos": spaces.Box(low=0, high=jnp.array([SCREEN_WIDTH, SCREEN_HEIGHT]), shape=(3, 2), dtype=jnp.float32),
        "enemy_state": spaces.Box(low=1, high=3, shape=(3,), dtype=jnp.int32),  # 1=weak,2=strong,3=dead

        # --- Fireball ---
        "fireball_pos": spaces.Box(low=0, high=jnp.array([SCREEN_WIDTH, SCREEN_HEIGHT]), shape=(2,), dtype=jnp.float32),
        "fireball_dir": spaces.Box(low=-1, high=1, shape=(), dtype=jnp.int32),

        # --- POW block ---
        "pow_block_counter": spaces.Box(low=0, high=3, shape=(), dtype=jnp.int32),
        "pow_block_pos": spaces.Box(low=0, high=jnp.array([SCREEN_WIDTH, SCREEN_HEIGHT]), shape=(1, 4), dtype=jnp.int32),

        # --- Platforms ---
        "plattforms_pos": spaces.Box(low=0, high=jnp.array([SCREEN_WIDTH, SCREEN_HEIGHT]), shape=(PLATFORMS.shape[0], 4), dtype=jnp.int32),
    })

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(210, 160, 3),
            dtype=jnp.uint8
        )

    def _get_info(self, state: MarioBrosState) -> MarioBrosInfo:
        return MarioBrosInfo(
            score=0
        )

    def _get_reward(
        self, previous_state: MarioBrosState, state: MarioBrosState
    ) -> float:
        return state.game.score - previous_state.game.score
    
    def _get_done(self, state: MarioBrosState) -> bool:
        return jnp.logical_or(state.lives < 0, state.game.score > 9999)

    def render(self, state: MarioBrosState) -> jnp.ndarray:
        return self.renderer.render(state)

    def reset(self, key=None) -> Tuple[MarioBrosObservation, MarioBrosState]:
        game = self.reset_game()
        obs = self._get_observation(game)
        return obs, game

    def reset_game(self) -> MarioBrosState:
        p1_y = PLATFORMS[1, 1]
        p2_y = PLATFORMS[2, 1]
        enemy_status = jnp.array([2, 2, 2])

        # Enemy 1 position (used also for enemy 3)
        enemy1_pos = jnp.array([5.0, p1_y - ENEMY_SIZE[1]])
        enemy2_pos = jnp.array([130.0, p2_y - ENEMY_SIZE[1]])

        new_state = MarioBrosState(
            player=PlayerState(
                pos=jnp.array([PLAYER_START_X, PLAYER_START_Y], dtype=jnp.float32),
                vel=jnp.array([0.0, 0.0]),
                on_ground=False,
                jump_phase=jnp.int32(0),
                ascend_frames=jnp.int32(0),
                idx_right=jnp.int32(0),
                idx_left=jnp.int32(0),
                jump=jnp.int32(0),
                walk_anim=jnp.int32(0),
                is_walking=False,
                move=jnp.array(0.0, dtype=jnp.float32),
                jumpL=False,
                jumpR=False,
                last_dir=jnp.int32(0),
                brake_frames_left=jnp.int32(0),
                bumped_idx=0,
                pow_bumped=False,
                safe= False
            ),
            game=GameState(
                enemy=Enemy(
                    enemy_pos=jnp.array([enemy1_pos, enemy2_pos, enemy1_pos]),  # 3rd enemy uses enemy1 prototype
                    enemy_vel=jnp.array([[ENEMY_MOVE_SPEED, 0.0], [-ENEMY_MOVE_SPEED, 0.0], [ENEMY_MOVE_SPEED, 0.0]]),
                    enemy_platform_idx=jnp.array([1, 2, 1]),
                    enemy_timer=jnp.array([0, 0, 0]),
                    enemy_weak_timer=jnp.array([0, 0, 0]),  # keep zeros for each enemy at start
                    enemy_initial_sides=jnp.array([0, 1, 0]),
                    enemy_active=jnp.array([1, 0, 0], dtype=jnp.int32),  # only first enemy active at start
                    enemy_init_positions=jnp.array([enemy1_pos, enemy2_pos, enemy1_pos]),
                    enemy_status=enemy_status
                ),
                pow_block_counter=jnp.int32(3),
                score = jnp.int32(0),
                fireball=Fireball(
                    pos= jnp.array(FIREBALL_INIT_XY),
                    start_pat= FIREBALL_MOVEMENT_Start,
                    move_pat= FIREBALL_MOVEMENT,
                    count= FIREBALL_RESTART,
                    state= 0,
                    dir= -1,
                    rnd= jax.random.PRNGKey(0),
                    ani= 1
                ),
                game_over= False,
                next_spawn_side=jnp.int32(1),
        ),
            lives=jnp.int32(4)
        )

        return new_state

    

    def old_step(self, state: MarioBrosState, action: chex.Array) -> Tuple[
        MarioBrosObservation, MarioBrosState, float, bool, MarioBrosInfo]:
        # calls player_step function and check for collision with enemy
        def enemy_collision(s):
            obs, rS = self.reset()
            return obs, rS, 0.0, True, self._get_info(rS)

        def no_enemy_collision(s):
            return self._get_observation(state), MarioBrosState(
                player=s,
                game=state.game,
                lives=state.lives
            ), 0.0, True, self._get_info(state)

        return jax.lax.cond(check_enemy_collision(state.player.pos, state.game.enemy.enemy_pos), enemy_collision,
                            no_enemy_collision, player_step(state.player, action))

    from functools import partial
    @partial(jax.jit, static_argnums=0)
    def step(self, state: MarioBrosState, action: chex.Array) -> Tuple[
        MarioBrosObservation, MarioBrosState, float, bool, MarioBrosInfo]:
        """
        Performs one step of the environment given a state and an action.
        Handles player movement, enemy collisions, enemy patrol updates,
        POW hits, and enemy status changes.

        Args:
            state: Current MarioBrosState.
            action: Action chosen by the agent.

        Returns:
            obs: MarioBrosObservation (player position).
            new_state: Updated MarioBrosState.
            reward: Float reward (0.0 here).
            done: Boolean flag if episode ended.
            info: MarioBrosInfo.
        """

        def game_over(state: MarioBrosState) -> MarioBrosState:
            obs, new_state = self.reset()
            obs = self._get_observation(state)
            info = self._get_info(state)
            reward = self._get_reward(state, new_state)
            return obs, new_state, reward, True, info
           

        def step(_) -> MarioBrosState:
            # 1) Advance player state given action
            new_player = player_step(state.player, action, state.game)
            bumped_idx = new_player.bumped_idx
            pow_bumped = new_player.pow_bumped
            # 2) Detect collisions between player and enemies
            def check_enemy_collision_per_enemy(player_pos, enemy_positions):
                px, py = player_pos
                pw, ph = PLAYER_SIZE
                ex, ey = enemy_positions[:, 0], enemy_positions[:, 1]
                ew, eh = ENEMY_SIZE

                overlap_x = (px < ex + ew) & (px + pw > ex)
                overlap_y = (py < ey + eh) & (py + ph > ey)
                return overlap_x & overlap_y  # bool array per enemy

            def check_fireball_collision(fb: Fireball, player_pos) -> bool:
                px, py = player_pos
                pw, ph = PLAYER_SIZE
                fx, fy = fb.pos
                fw, fh = FIREBALL_SIZE
                
                overlap_x = (px < fx + fw) & (px + pw > fx)
                overlap_y = (py < fy + fh) & (py + ph > fy)
                
                return overlap_x & overlap_y

            collided_mask = check_enemy_collision_per_enemy(new_player.pos, state.game.enemy.enemy_pos) & (state.game.enemy.enemy_active != 0)

            fireball_hit = check_fireball_collision(state.game.fireball, new_player.pos)
            # Check if any collided enemy is strong (status==2)
            strong_enemy_hit = jnp.any(collided_mask & (state.game.enemy.enemy_status == 2))

            # --- Handling collision with strong enemy: reset Mario position only ---
            def on_hit(_):
                # Spielerleben um 1 reduzieren
                new_lives = state.lives - 1
                
                # Mario an Respawn-Position setzen, falls Leben >= 0
                new_player_updated = new_player._replace(safe= True, jump_phase= jnp.int32(0))
                
                # Neuer Zustand mit reduziertem Leben und neuer Position
                new_game = state.game._replace(game_over=jnp.where(state.lives > 0, False, True))

                new_state = MarioBrosState(
                    player=new_player_updated,
                    game=new_game,
                    lives=new_lives
                )
                
                
                # Beobachtung und Info aus dem endgültigen Zustand holen
                obs = self._get_observation(new_state)
                info = self._get_info(new_state)
                reward = self._get_reward(state, new_state)
                done = self._get_done(new_state)
                return obs, new_state, reward, done, info

            # --- No strong enemy collision: normal game progress ---
            def on_no_hit(_):
                # 3) Update enemy patrol movements & dynamic spawn logic (pending spawn state)
                enemy = state.game.enemy

                # active==1 means fully active; 2 means pending (spawned this frame, not yet occupying for checks); 0 means free
                active_now = (enemy.enemy_active == 1)

                TOP_PLATFORM_IDX = jnp.array([1, 2], dtype=jnp.int32)
                is_on_top = jnp.logical_and(
                    jnp.isin(enemy.enemy_platform_idx, TOP_PLATFORM_IDX),
                    active_now
                )
                top_present = jnp.any(is_on_top)

                # find first free slot (enemy_active == 0)
                free_mask = (enemy.enemy_active == 0)
                any_free = jnp.any(free_mask)
                idx_to_spawn = jnp.argmax(free_mask.astype(jnp.int32))

                right_proto = enemy.enemy_init_positions[0]
                left_proto = enemy.enemy_init_positions[1]

                spawn_side = state.game.next_spawn_side
                spawn_pos = jnp.where(spawn_side == 1, right_proto, left_proto)
                spawn_vel = jnp.where(spawn_side == 1,
                                      jnp.array([ENEMY_MOVE_SPEED, 0.0]),
                                      jnp.array([-ENEMY_MOVE_SPEED, 0.0]))

                def do_spawn(ep, ev, pidx, timer, init_sides, active_arr, status, next_side):
                    ep2 = ep.at[idx_to_spawn].set(spawn_pos)
                    ev2 = ev.at[idx_to_spawn].set(spawn_vel)
                    pidx2 = pidx.at[idx_to_spawn].set(1)
                    timer2 = timer.at[idx_to_spawn].set(0)
                    init_sides2 = init_sides.at[idx_to_spawn].set(spawn_side)
                    # mark pending (2) — not drawn / not counted as occupying top
                    active2 = active_arr.at[idx_to_spawn].set(jnp.int32(2))
                    status2 = status.at[idx_to_spawn].set(jnp.int32(2))
                    next_side2 = jnp.where(next_side == 1, jnp.int32(0), jnp.int32(1))
                    return ep2, ev2, pidx2, timer2, init_sides2, active2, status2, next_side2

                def no_spawn(ep, ev, pidx, timer, init_sides, active_arr, status, next_side):
                    return ep, ev, pidx, timer, init_sides, active_arr, status, next_side

                # don't spawn if there's any active or pending enemy on top
                pending_mask = (enemy.enemy_active == 2)
                top_pending = jnp.any(
                    jnp.logical_and(
                        jnp.isin(enemy.enemy_platform_idx, TOP_PLATFORM_IDX),
                        pending_mask
                    )
                )
                spawn_cond = jnp.logical_and(jnp.logical_not(jnp.logical_or(top_present, top_pending)), any_free)

                (ep2, ev2, pidx2, timer2, init_sides2, active2, status2, next_spawn_side) = jax.lax.cond(
                    spawn_cond,
                    do_spawn,
                    no_spawn,
                    enemy.enemy_pos, enemy.enemy_vel, enemy.enemy_platform_idx,
                    enemy.enemy_timer, enemy.enemy_initial_sides,
                    enemy.enemy_active, enemy.enemy_status,
                    state.game.next_spawn_side
                )

                # enemy_step expects mask==True for INACTIVE slots (legacy). Pass True where slot is inactive (active2 == 0).
                mask_for_step = (active2 == 1)

                ep, ev, idx, timer, sides, status = enemy_step(
                    ep2, ev2, pidx2, timer2,
                    PLATFORMS,
                    init_sides2,
                    mask_for_step,
                    enemy.enemy_init_positions,
                    status2
                )

                # If we spawned this frame, preserve spawn position/platform for this frame (prevent repeated spawning)
                idx = jax.lax.cond(spawn_cond, lambda arr: arr.at[idx_to_spawn].set(1), lambda arr: arr, idx)


                # Maintain 0=free,1=active,2=pending semantics
                new_enemy_active = active2

                # Promote pending (2) -> active (1) only when no *other* enemy (active or pending) is on top.
                idx_top_mask = jnp.logical_and(
                    jnp.isin(idx, TOP_PLATFORM_IDX),
                    new_enemy_active != 0
                )
                total_top = jnp.sum(idx_top_mask.astype(jnp.int32))
                self_on_top = idx_top_mask.astype(jnp.int32)
                others_on_top = total_top - self_on_top
                promote_mask = jnp.logical_and(new_enemy_active == 2, others_on_top == 0)
                new_enemy_active = jnp.where(promote_mask, jnp.int32(1), new_enemy_active)

                # Deactivate permanently-dead enemies (status == 3)
                new_enemy_active = jnp.where(status == 3, jnp.int32(0), new_enemy_active)

                # 4) Update fireball AFTER enemy_step
                new_fireball = fireball_step(state.game.fireball)


                # 5) POW hit logic and platform bump detection (unchanged)
                pow_hit = pow_bumped & (state.game.pow_block_counter > 0)
                new_pow_block_counter = jnp.maximum(state.game.pow_block_counter - pow_hit, 0)
                bumped_idx_final = jnp.where(pow_hit, -2, bumped_idx)

                # 6) Toggle enemy status for bumped platforms/POW (strong <-> weak)
                def toggle_status(old_status):
                    return jnp.where(old_status == 2, 1,
                                     jnp.where(old_status == 1, 2, old_status))

                TOGGLE_RANGE = 10
                player_x = new_player.pos[0]
                enemy_x = ep[:, 0]
                nearby_mask = jnp.abs(enemy_x - player_x) <= TOGGLE_RANGE
                toggle_mask = (idx == bumped_idx_final) & nearby_mask

                new_enemy_status = jax.lax.cond(
                    bumped_idx_final >= 0,
                    lambda old_status: jnp.where(toggle_mask, toggle_status(old_status), old_status),
                    lambda old_status: old_status,
                    status
                )

                new_enemy_status = jax.lax.cond(
                    bumped_idx_final == -2,
                    lambda old_status: toggle_status(old_status),
                    lambda old_status: old_status,
                    new_enemy_status
                )

                # --- Collision logic with enemies after enemy step ---
                # Use fully-active check (==1) so pending enemies are not hittable/drawn
                collided_mask = check_enemy_collision_per_enemy(new_player.pos, ep) & (new_enemy_active == 1)
                fireball_hit = check_fireball_collision(new_fireball, new_player.pos)

                enemy_status_after_hit = jnp.where(
                    (collided_mask) & (new_enemy_status == 1),
                    3,
                    new_enemy_status
                )

                # --- Weak-state countdown logic (start/reset when enemy becomes weak, count down while weak) ---
                # prev_weak / prev_status are from the previous frame (before this frame's toggles/hits)
                prev_weak = enemy.enemy_weak_timer
                prev_status = enemy.enemy_status

                # Use the *final* status for this frame (after bump/toggle/hit) to decide timer behavior.
                final_status = enemy_status_after_hit

                # If an enemy just became weak this frame (was not 1, now 1), set timer to 776.
                start_weak_mask = (prev_status != 1) & (final_status == 1)
                weak_timer = jnp.where(start_weak_mask, jnp.array(776, dtype=jnp.int32), prev_weak)

                # For enemies already weak, decrement timer each frame (clamp at 0).
                weak_timer = jnp.where(final_status == 1, jnp.maximum(weak_timer - 1, 0), weak_timer)

                # If timer reached 0 while still weak, revert to strong (status==2).
                revert_mask = (final_status == 1) & (weak_timer == 0)
                final_status = jnp.where(revert_mask, jnp.array(2, dtype=jnp.int32), final_status)

                # Replace enemy_status_after_hit with the possibly-updated final_status
                enemy_status_after_hit = final_status
                # -------------------------------------------------------------------------------

                # --- Score tracking: reward for defeating weak enemies ---
                was_weak = (collided_mask) & (new_enemy_status == 1)
                now_dead = (collided_mask) & (enemy_status_after_hit == 3)
                newly_killed = was_weak & now_dead

                num_killed = jnp.sum(newly_killed.astype(jnp.int32))
                score_gain = num_killed * 800
                new_score = state.game.score + score_gain

                # If collided enemy is strong (status==2), reset Mario position locally (extra safety)
                mario_pos_reset = jnp.any(((collided_mask) & (new_enemy_status == 2)) | fireball_hit)
                ORIGINAL_MARIO_POS = jnp.array([100, 100], dtype=jnp.float32)

                # Build updated enemy and game state (store active mask as enemy_active)
                new_enemy = Enemy(
                    enemy_pos=ep,
                    enemy_vel=ev,
                    enemy_platform_idx=idx,
                    enemy_timer=timer,
                    enemy_weak_timer=weak_timer,
                    enemy_initial_sides=sides,
                    enemy_active=new_enemy_active,
                    enemy_init_positions=state.game.enemy.enemy_init_positions,
                    enemy_status=enemy_status_after_hit,
                )

                new_game = GameState(
                    enemy=new_enemy,
                    pow_block_counter=new_pow_block_counter,
                    score=new_score,
                    fireball=new_fireball,
                    game_over=state.game.game_over,
                    next_spawn_side=next_spawn_side,
                )

                # Final state and observation
                new_state = MarioBrosState(
                    player=new_player,
                    game=new_game,
                    lives=state.lives
                )

                obs = self._get_observation(new_state)
                reward = self._get_reward(state, new_state)
                done = self._get_done(new_state)
                return obs, new_state, reward, done, self._get_info(new_state)

            # 10) Return based on whether strong enemy was hit
            cond = (strong_enemy_hit | fireball_hit) & jnp.logical_not(new_player.safe)
            return jax.lax.cond(cond, on_hit, on_no_hit, new_player)
        return lax.cond(self._get_done(state), game_over, step, state)


# run game with: python scripts\play.py --game mariobros
if __name__ == "__main__":
    pass