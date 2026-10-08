import jax.numpy as jnp
from typing import NamedTuple
import chex

# --------------------  Constants --------------------
MAX_LADDERS = 6
MAX_PLATFORMS = 20
MAX_FRUITS = 3
MAX_BELLS = 1
MAX_CHILD = 1
MAX_OBJECTS = MAX_LADDERS + MAX_PLATFORMS + MAX_FRUITS + MAX_BELLS + MAX_CHILD

class LevelConstants(NamedTuple):
    ladder_positions: chex.Array
    ladder_sizes: chex.Array
    platform_positions: chex.Array
    platform_sizes: chex.Array
    fruit_positions: chex.Array
    bell_position: chex.Array
    child_position: chex.Array


LADDER_HEIGHT = jnp.array(36)
LADDER_WIDTH = jnp.array(8)
P_HEIGHT = jnp.array(4)

# -------------------- Level 1 --------------------

LEVEL_1_LADDERS_POS = jnp.array(
    [
        [132, 132],  # L1L1
        [20, 84],  # L1L2
        [132, 36],  # L1L3
    ]
)

LEVEL_1_LADDERS_SIZE = jnp.array(
    [
        [LADDER_WIDTH, LADDER_HEIGHT],
        [LADDER_WIDTH, LADDER_HEIGHT],
        [LADDER_WIDTH, LADDER_HEIGHT],
    ]
)

LEVEL_1_PLATFORMS_POS = jnp.array(
    [
        [16, 172],  # L1P1
        [16, 124],  # L1P2
        [16, 76],  # L1P3
        [16, 28],  # L1P4
    ]
)

LEVEL_1_PLATFORMS_SIZE = jnp.array(
    [
        [128, P_HEIGHT],
        [128, P_HEIGHT],
        [128, P_HEIGHT],
        [128, P_HEIGHT],
    ]
)

LEVEL_1_FRUITS_POS = jnp.array([[119, 108], [39, 84], [59, 60]])

LEVEL_1_BELL_POS = jnp.array([93, 36])

LEVEL_1_CHILD_POS = jnp.array([121, 13])

Kangaroo_Level_1 = LevelConstants(
    ladder_positions=LEVEL_1_LADDERS_POS,
    ladder_sizes=LEVEL_1_LADDERS_SIZE,
    platform_positions=LEVEL_1_PLATFORMS_POS,
    platform_sizes=LEVEL_1_PLATFORMS_SIZE,
    fruit_positions=LEVEL_1_FRUITS_POS,
    bell_position=LEVEL_1_BELL_POS,
    child_position=LEVEL_1_CHILD_POS,
)

# -------------------- Level 2 --------------------

LEVEL_2_LADDERS_POS = jnp.array(
    [
        [120, 132],  # L2L1
        [24, 116],  # L2L2
        [128, 36],  # L2L3
    ]
)

LEVEL_2_LADDERS_SIZE = jnp.array(
    [
        [LADDER_WIDTH, 4],
        [LADDER_WIDTH, 4],
        [LADDER_WIDTH, 4],
    ]
)

LEVEL_2_PLATFORMS_POS = jnp.array(
    [
        [16, 172],  # L2P1
        [16, 28],  # L2P2
        [16, 124],  # L2P3
        [52, 124],  # L2P4
        [16, 76],  # L2P5
        [84, 76],  # L2P6
        [28, 164],  # L2P7
        [112, 84],  # L2P8
        [120, 44],  # L2P9
        [48, 156],  # L2P10
        [76, 148],  # L2P11
        [104, 140],  # L2P12
        [16, 108],  # L2P13
        [56, 100],  # L2P14
        [84, 92],  # L2P15
        [64, 60],  # L2P16
        [92, 52],  # L2P17
        [28, 68],  # L2P18
    ]
)

LEVEL_2_PLATFORMS_SIZE = jnp.array(
    [
        [128, P_HEIGHT],  # L2P1
        [128, P_HEIGHT],  # L2P2
        [28, P_HEIGHT],  # L2P3
        [92, P_HEIGHT],  # L2P4
        [60, P_HEIGHT],  # L2P5
        [60, P_HEIGHT],  # L2P6
        [24, P_HEIGHT],  # L2P7
        [24, P_HEIGHT],  # L2P8
        [24, P_HEIGHT],  # L2P9
        [32, P_HEIGHT],  # L2P10
        [32, P_HEIGHT],  # L2P11
        [32, P_HEIGHT],  # L2P12
        [32, P_HEIGHT],  # L2P13
        [20, P_HEIGHT],  # L2P14
        [20, P_HEIGHT],  # L2P15
        [20, P_HEIGHT],  # L2P16
        [20, P_HEIGHT],  # L2P17
        [28, P_HEIGHT],  # L2P18
    ]
)

# LEVEL_2_FRUITS_POS
LEVEL_2_FRUITS_POS = jnp.array([[44, 68], [124, 92], [94, 140]])
# LEVEL_2_BELL_POS
LEVEL_2_BELL_POS = jnp.array([31, 36])
# LEVEL_2_CHILD_POS
LEVEL_2_CHILD_POS = jnp.array([121, 13])

Kangaroo_Level_2 = LevelConstants(
    ladder_positions=LEVEL_2_LADDERS_POS,
    ladder_sizes=LEVEL_2_LADDERS_SIZE,
    platform_positions=LEVEL_2_PLATFORMS_POS,
    platform_sizes=LEVEL_2_PLATFORMS_SIZE,
    fruit_positions=LEVEL_2_FRUITS_POS,
    bell_position=LEVEL_2_BELL_POS,
    child_position=LEVEL_2_CHILD_POS,
)

# -------------------- Level 3 --------------------

LEVEL_3_LADDERS_POS = jnp.array(
    [
        [20, 36],  # L3L1
        [20, 148],  # L3L2
        [36, 116],  # L3L3
        [104, 36],  # L3L4
        [120, 68],  # L3L5
        [132, 84],  # L3L6
    ]
)

LEVEL_3_LADDERS_SIZE = jnp.array(
    [
        [LADDER_WIDTH, 28],  # L3L1
        [LADDER_WIDTH, 4],  # L3L2
        [LADDER_WIDTH, 20],  # L3L3
        [LADDER_WIDTH, 20],  # L3L4
        [LADDER_WIDTH, 4],  # L3L5
        [LADDER_WIDTH, 4],  # L3L6
    ]
)

LEVEL_3_PLATFORMS_POS = jnp.array(
    [
        [16, 172],  # L3P1
        [16, 28],  # L3P2
        [88, 140],  # L3P3
        [64, 148],  # L3P4
        [100, 116],  # L3P5
        [48, 100],  # L3P6
        [76, 52],  # L3P7
        [80, 36],  # L3P8
        [104, 132],  # L3P9
        [84, 156],  # L3P10
        [124, 124],  # L3P11
        [52, 84],  # L3P12
        [108, 164],  # L3P13
        [16, 108],  # L3P14
        [16, 92],  # L3P15
        [76, 92],  # L3P16
        [16, 140],  # L3P17
        [96, 60],  # L3P18
        [100, 76],  # L3P19
        [60, 44],  # L3P20
    ]
)

LEVEL_3_PLATFORMS_SIZE = jnp.array(
    [
        [128, P_HEIGHT],  # L3P1
        [128, P_HEIGHT],  # L3P2
        [16, P_HEIGHT],  # L3P3
        [16, P_HEIGHT],  # L3P4
        [16, P_HEIGHT],  # L3P5
        [16, P_HEIGHT],  # L3P6
        [16, P_HEIGHT],  # L3P7
        [16, P_HEIGHT],  # L3P8
        [20, P_HEIGHT],  # L3P9
        [20, P_HEIGHT],  # L3P10
        [20, P_HEIGHT],  # L3P11
        [20, P_HEIGHT],  # L3P12
        [36, P_HEIGHT],  # L3P13
        [80, P_HEIGHT],  # L3P14
        [28, P_HEIGHT],  # L3P15
        [69, P_HEIGHT],  # L3P16
        [32, P_HEIGHT],  # L3P17
        [36, P_HEIGHT],  # L3P18
        [44, P_HEIGHT],  # L3P19
        [12, P_HEIGHT],  # L3P20
    ]
)

# LEVEL_3_FRUITS_POS
LEVEL_3_FRUITS_POS = jnp.array([[124, 92], [89, 116], [18, 60]])
# LEVEL_3_BELL_POS
LEVEL_3_BELL_POS = jnp.array([130, 36])
# LEVEL_3_CHILD_POS
LEVEL_3_CHILD_POS = jnp.array([121, 13])

Kangaroo_Level_3 = LevelConstants(
    ladder_positions=LEVEL_3_LADDERS_POS,
    ladder_sizes=LEVEL_3_LADDERS_SIZE,
    platform_positions=LEVEL_3_PLATFORMS_POS,
    platform_sizes=LEVEL_3_PLATFORMS_SIZE,
    fruit_positions=LEVEL_3_FRUITS_POS,
    bell_position=LEVEL_3_BELL_POS,
    child_position=LEVEL_3_CHILD_POS,
)
