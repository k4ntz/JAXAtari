import jax.numpy as jnp
# Action constants
NOOP = 0
FIRE = 1
UP = 2
RIGHT = 3
LEFT = 4
DOWN = 5
UPRIGHT = 6
UPLEFT = 7
DOWNRIGHT = 8
DOWNLEFT = 9
UPFIRE = 10
RIGHTFIRE = 11
LEFTFIRE = 12
DOWNFIRE = 13
UPRIGHTFIRE = 14
UPLEFTFIRE = 15
DOWNRIGHTFIRE = 16
DOWNLEFTFIRE = 17

# Game constants (placeholders - to be adjusted according to ALE)
COURT_WIDTH = 160
COURT_HEIGHT = 210
TOP_START_X = 71
TOP_START_Y = 24
BOT_START_X = 71
BOT_START_Y = 160
BALL_START_X = 75  # taken from the RAM extraction script (initial ram 77 - 2)
BALL_START_Y = 44  # taken from the RAM extraction script (189 - initial ram 145)
BALL_START_Z = 7  # of course there is no real z, but using the shadow it is suggested that there is a z
PLAYER_WIDTH = 13
PLAYER_HEIGHT = 23
BALL_SIZE = 2

WAIT_AFTER_GOAL = 0  # number of ticks that are waited after a goal was scored

Z_DERIVATIVES = jnp.array([
    # Bounce recovery from 0
    3, 2, 3, 2, 2, 2,

    # This is where we join during serve (height ~14)
    2, 2, 2, 2, 2, 1, 2, 1, 1, 2, 1, 0, 1, 1, 0, 1,

    # Peak plateau at 38
    0, 0, 0, 0, 0, -1, 0, -1, -1, 0, -1,

    # Main descent
    -2, -1, -1, -2, -1, -2, -2, -2, -2, -2,

    # Final drop to 0
    -3, -2, -3, -3, -2, -3, -1
])

# Y-movement derivatives (excluding net jump)
Y_DERIVATIVES = jnp.array([
    # Initial oscillation
    0, -2, 0, 0, 0, 2, 0, 0, 0, 2,

    # Gradual rise with alternating speeds
    0, 2, 0, 2, 0, 2, 2, 1, 3, 2,

    # Steady rise
    2, 2, 2, 2, 4, 2, 4, 2, 4, 2,

    # Fast rise before net
    4, 4, 4, 2, 4,

    # After net (landed around y=114)
    4, 4, 6, 4, 6, 2, -2, 0, 0, 0,

    # Plateau with slight adjustments
    0, 0, 0, 0, 2, 0, 0, 2, 0, 2,

    # Final movement
    0, 2, 8, -4
])

# all x patterns as global constants
PATTERN_1 = jnp.array([1])  # Edge right
PATTERN_2 = jnp.array([1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0])  # Near right
PATTERN_3 = jnp.array([0, 1, 1, 0, 1, 1, 0, 1, 1])  # Right 3-4
PATTERN_4 = jnp.array([1, 0, 0, 1, 0, 0, 0])  # Right 5
PATTERN_5 = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])  # Right 6-7
PATTERN_6 = jnp.array([-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # Position 8 & left 7-8
PATTERN_7 = jnp.array([-1, 0, 0, -1, 0, 0, 0])  # Right 9
PATTERN_8 = jnp.array([-1, 0])  # Right 10 & left 5
PATTERN_9 = jnp.array([-1, -1, 0])  # Right 11-13, left 10, & left 1-4
PATTERN_10 = jnp.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1])  # Left 9
PATTERN_11 = jnp.array([-1, 0, 0])  # Left 6

# Create a 2D array where each row is a pattern, padded to the same length
max_length = max(len(p) for p in [
    PATTERN_1, PATTERN_2, PATTERN_3, PATTERN_4, PATTERN_5,
    PATTERN_6, PATTERN_7, PATTERN_8, PATTERN_9, PATTERN_10, PATTERN_11
])

# Pad each pattern to max_length
def pad_pattern(pattern, length):
    return jnp.pad(pattern, (0, length - len(pattern)), mode="wrap")


# Create a single 2D array of patterns
STACKED_X_PATTERNS = jnp.stack([
    pad_pattern(PATTERN_1, max_length),
    pad_pattern(PATTERN_2, max_length),
    pad_pattern(PATTERN_3, max_length),
    pad_pattern(PATTERN_4, max_length),
    pad_pattern(PATTERN_5, max_length),
    pad_pattern(PATTERN_6, max_length),
    pad_pattern(PATTERN_7, max_length),
    pad_pattern(PATTERN_8, max_length),
    pad_pattern(PATTERN_9, max_length),
    pad_pattern(PATTERN_10, max_length),
    pad_pattern(PATTERN_11, max_length)
])


# Store the length of each pattern for proper cycling
PATTERN_LENGTHS = jnp.array([
    len(PATTERN_1), len(PATTERN_2), len(PATTERN_3),
    len(PATTERN_4), len(PATTERN_5), len(PATTERN_6),
    len(PATTERN_7), len(PATTERN_8), len(PATTERN_9),
    len(PATTERN_10), len(PATTERN_11)
])

# Pre-compute first non-zero values for each pattern
X_DIRECTION_ARRAY = jnp.array([
    1,   # PATTERN_1 starts with 1
    1,   # PATTERN_2 starts with 1
    0,   # PATTERN_3 starts with 0, but first non-zero is 1
    1,   # PATTERN_4 starts with 1
    0,   # PATTERN_5 starts with many 0s, first non-zero is 1
    -1,  # PATTERN_6 starts with -1
    -1,  # PATTERN_7 starts with -1
    -1,  # PATTERN_8 starts with -1
    -1,  # PATTERN_9 starts with -1
    0,   # PATTERN_10 starts with 0s, first non-zero is 1
    -1   # PATTERN_11 starts with -1
])

TOPSIDE_STARTING_Z = 28
BOTSIDE_STARTING_Z = 2 # approx
TOPSIDE_BOUNCE = jnp.array([0,0,0,0,0,0,0,0,0,2,0,0,2,-2,-4,-4,-6,-4,-4,-4,-2,-4,-4,-4,-2,-4,-2,-4,-2,-2,-2,-2,-2,-2])
BOTSIDE_BOUNCE = jnp.array([0,0,0,0,0,0,0,0,0,2,0,0,2,0,2])


# Index to jump to after crossing net (skip the 18 value)
NET_CROSS_INDEX = 35

SERVE_INDEX = 6  # Index where height is 14 during upward movement

# X movement patterns # TODO reimplement these
LEFT_X_PATTERN = jnp.array([-1, 0, -1, 0])
RIGHT_X_PATTERN = jnp.array([1, 0, 1, 0])

# game constrains (i.e. the net)
NET_RANGE = (98, 113)

# TODO: define the constraints of everyone (player, ball, enemy) according to the base implementation
TOP_ENTITY_MAX_LEFT = 4
TOP_ENTITY_MAX_RIGHT = 142
TOP_ENTITY_MAX_TOP = 18
TOP_ENTITY_MAX_BOTTOM = 75

BOTTOM_ENTITY_MAX_LEFT = 4
BOTTOM_ENTITY_MAX_RIGHT = 142
BOTTOM_ENTITY_MAX_TOP = 113
BOTTOM_ENTITY_MAX_BOTTOM = 178

NET_TOP_LEFT = (40, 48)  # (0, 48)  # x,y top left corner
NET_TOP_RIGHT = (120, 48)  # (120, 48)
NET_BOTTOM_LEFT = (24, 178)
NET_BOTTOM_RIGHT = (136, 178)