import os
from functools import partial
from typing import NamedTuple, Tuple
from jaxatari.environment import JaxEnvironment, EnvState, EnvObs, EnvInfo
import jax
import jax.numpy as jnp
import chex
import numpy as np
import pygame
import jaxatari.rendering.atraJaxis as aj
from jaxatari.games.jax_kangaroo import ENEMY_WIDTH, ENEMY_HEIGHT


def load_sprites():
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
    BG = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/tennis/bg/1.npy"))
    
    frames_pl_r = []
    for i in range(1, 5):
        frame = aj.loadFrame(os.path.join(MODULE_DIR, f"sprites/tennis/pl_red/{i}.npy"))
        frames_pl_r.append(frame)
    PL_R = jnp.array(aj.pad_to_match(frames_pl_r))
    
    frames_bat_r = []
    for i in range(1, 5):
        frame = aj.loadFrame(os.path.join(MODULE_DIR, f"sprites/tennis/bat_r/{i}.npy"))
        frames_bat_r.append(frame)
    BAT_R = jnp.array(aj.pad_to_match(frames_bat_r))
    
    frames_pl_b = []
    for i in range(1, 5):
        frame = aj.loadFrame(os.path.join(MODULE_DIR, f"sprites/tennis/pl_blue/{i}.npy"))
        frames_pl_b.append(frame)
    PL_B = jnp.array(aj.pad_to_match(frames_pl_b))
    
    frames_bat_b = []
    for i in range(1, 5):
        frame = aj.loadFrame(os.path.join(MODULE_DIR, f"sprites/tennis/bat_b/{i}.npy"))
        frames_bat_b.append(frame)
    BAT_B = jnp.array(aj.pad_to_match(frames_bat_b))
    
    BALL = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/tennis/ball/1.npy"))
    BALL_SHADE = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/tennis/ball_shade/1.npy"))
    
    DIGITS_R = aj.load_and_pad_digits(os.path.join(MODULE_DIR, "sprites/tennis/digits_r/{}.npy"))
    DIGITS_B = aj.load_and_pad_digits(os.path.join(MODULE_DIR, "sprites/tennis/digits_b/{}.npy"))


    return BG, PL_R, BAT_R, PL_B, BAT_B, BALL, BALL_SHADE, DIGITS_R, DIGITS_B

BG, PL_R, BAT_R, PL_B, BAT_B, BALL, BALL_SHADE, DIGITS_R, DIGITS_B = load_sprites()

# TODO: remove, for debugging purposes only
def jaxprint(arg1, arg2=None):
    jax.debug.print("{x}: {y}", x=arg1, y=arg2)


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

# TODO: due to the movement properties of the ball in Tennis, the velocity values should probably represent how frequent ticks are skipped..
# Define state container
class TennisState(NamedTuple):
    player_x: chex.Array  # Player x position
    player_y: chex.Array  # Player y position
    player_direction: chex.Array  # Player direction (0 for right, 1 for left)
    enemy_x: chex.Array  # Enemy x position
    enemy_y: chex.Array  # Enemy y position
    enemy_direction: chex.Array  # Enemy direction (0 for right, 1 for left)
    ball_x: chex.Array  # Ball x position
    ball_y: chex.Array  # Ball y position
    ball_z: chex.Array  # Ball height/shadow
    ball_curve_counter: chex.Array
    ball_x_dir: chex.Array  # Ball x direction
    ball_y_dir: chex.Array  # Ball y direction
    ball_movement_tick: chex.Array  # Ball movement tick
    ball_curve: chex.Array
    ball_start: chex.Array
    ball_end: chex.Array
    shadow_x: chex.Array  # Shadow x position
    shadow_y: chex.Array  # Shadow y position
    player_round_score: chex.Array
    enemy_round_score: chex.Array
    player_score: chex.Array
    enemy_score: chex.Array
    round_overtime: chex.Array  # boolean array that is only true if the round is in overtime
    game_overtime: chex.Array # boolean array that is only true if the game is in overtime
    serving: chex.Array  # boolean for serve state
    side_switch_counter: chex.Array # tracks side switch cycle position
    just_hit: chex.Array # boolean for just hit state
    player_side: chex.Array  # 0 if player on top side; 1 if player on bottom side
    ball_was_infield: chex.Array
    current_tick: chex.Array
    ball_y_tick: chex.Array
    ball_x_pattern_idx: chex.Array  # Array holding the x-movement pattern
    ball_x_counter: chex.Array  # Current index in the pattern
    player_hit: chex.Array
    enemy_hit: chex.Array


# Observation and Info containers
class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray

class TennisObservation(NamedTuple):
    player: EntityPosition
    enemy: EntityPosition
    ball: EntityPosition
    ball_shadow: EntityPosition
    player_round_score: jnp.ndarray
    enemy_round_score: jnp.ndarray
    player_score: jnp.ndarray
    enemy_score: jnp.ndarray

class TennisInfo(NamedTuple):
    serving: jnp.ndarray
    player_side: jnp.ndarray
    current_tick: jnp.ndarray
    ball_direction: jnp.ndarray


def get_human_action() -> chex.Array:
    """Get human action from keyboard with support for diagonal movement and combined fire"""
    keys = pygame.key.get_pressed()
    up = keys[pygame.K_UP] or keys[pygame.K_w]
    down = keys[pygame.K_DOWN] or keys[pygame.K_s]
    left = keys[pygame.K_LEFT] or keys[pygame.K_a]
    right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
    fire = keys[pygame.K_SPACE]

    # Diagonal movements with fire
    if up and right and fire:
        return jnp.array(UPRIGHTFIRE)
    if up and left and fire:
        return jnp.array(UPLEFTFIRE)
    if down and right and fire:
        return jnp.array(DOWNRIGHTFIRE)
    if down and left and fire:
        return jnp.array(DOWNLEFTFIRE)

    # Cardinal directions with fire
    if up and fire:
        return jnp.array(UPFIRE)
    if down and fire:
        return jnp.array(DOWNFIRE)
    if left and fire:
        return jnp.array(LEFTFIRE)
    if right and fire:
        return jnp.array(RIGHTFIRE)

    # Diagonal movements
    if up and right:
        return jnp.array(UPRIGHT)
    if up and left:
        return jnp.array(UPLEFT)
    if down and right:
        return jnp.array(DOWNRIGHT)
    if down and left:
        return jnp.array(DOWNLEFT)

    # Cardinal directions
    if up:
        return jnp.array(UP)
    if down:
        return jnp.array(DOWN)
    if left:
        return jnp.array(LEFT)
    if right:
        return jnp.array(RIGHT)
    if fire:
        return jnp.array(FIRE)

    return jnp.array(NOOP)


@partial(jax.jit, static_argnums=())
def get_ball_x_pattern(impact_point: chex.Array, player_direction: chex.Array) -> Tuple[chex.Array, chex.Array]:
    """
    Determines the ball's x-direction pattern based on where it hit the player's paddle.

    Args:
        impact_point: Position where the ball hit the paddle (0 is leftmost, 12 is rightmost)
        player_direction: Direction player is facing (0 for right, 1 for left)

    Returns:
        Tuple containing:
            chex.Array: X-pattern array for the ball movement
            chex.Array: Index for tracking position in the pattern
    """
    EDGE_RIGHT = 0
    NEAR_RIGHT = 1
    RIGHT_3_4 = 2
    RIGHT_5 = 3
    RIGHT_6_7 = 4
    RIGHT_LEFT_8 = 5
    RIGHT_9 = 6
    RIGHT_10_LEFT_5 = 7
    RIGHT_11_13_LEFT_1_4_10 = 8
    LEFT_9 = 9
    LEFT_6 = 10

    # Function to select pattern index when player is facing right
    def select_right_facing_pattern_index(point):
        return jax.lax.switch(
            point,
            [
                lambda: EDGE_RIGHT,                # 0: Rightmost edge
                lambda: NEAR_RIGHT,                # 1: 1-2 pixels from right
                lambda: NEAR_RIGHT,                # 2: 1-2 pixels from right
                lambda: RIGHT_3_4,                 # 3: 3 pixels from right
                lambda: RIGHT_3_4,                 # 4: 4 pixels from right
                lambda: RIGHT_5,                   # 5: 5 pixels from right
                lambda: RIGHT_6_7,                 # 6: 6-7 pixels from right
                lambda: RIGHT_6_7,                 # 7: 6-7 pixels from right
                lambda: RIGHT_LEFT_8,              # 8: 8 pixels from right
                lambda: RIGHT_9,                   # 9: 9 pixels from right
                lambda: RIGHT_10_LEFT_5,           # 10: 10 pixels from right
                lambda: RIGHT_11_13_LEFT_1_4_10,   # 11: 11-13 pixels from right
                lambda: RIGHT_11_13_LEFT_1_4_10,   # 12: 11-13 pixels from right
            ],
        )

    # Function to select pattern index when player is facing left
    def select_left_facing_pattern_index(point):
        return jax.lax.switch(
            point,
            [
                lambda: RIGHT_11_13_LEFT_1_4_10,   # 0: 1-4 pixels from left
                lambda: RIGHT_11_13_LEFT_1_4_10,   # 1: 1-4 pixels from left
                lambda: RIGHT_11_13_LEFT_1_4_10,   # 2: 1-4 pixels from left
                lambda: RIGHT_11_13_LEFT_1_4_10,   # 3: 1-4 pixels from left
                lambda: RIGHT_10_LEFT_5,           # 4: 5 pixels from left
                lambda: LEFT_6,                    # 5: 6 pixels from left
                lambda: RIGHT_LEFT_8,              # 6: 7-8 pixels from left
                lambda: RIGHT_LEFT_8,              # 7: 7-8 pixels from left
                lambda: LEFT_9,                    # 8: 9 pixels from left
                lambda: RIGHT_11_13_LEFT_1_4_10,   # 9: 10 pixels from left
                lambda: RIGHT_11_13_LEFT_1_4_10,   # 10: After switch point
                lambda: RIGHT_11_13_LEFT_1_4_10,   # 11: After switch point
                lambda: RIGHT_11_13_LEFT_1_4_10,   # 12: After switch point
            ],
        )

    # Select pattern index based on player direction and impact point
    pattern_index = jax.lax.cond(
        player_direction == 0,
        lambda: select_right_facing_pattern_index(impact_point.astype(jnp.int32)),
        lambda: select_left_facing_pattern_index(impact_point.astype(jnp.int32))
    )

    return pattern_index


@partial(jax.jit, static_argnums=())
def update_z_position(z_pos: chex.Array, current_tick: chex.Array, serve_hit: chex.Array) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """
    Updates Z position using a continuous cycle pattern.
    """

    # Get current derivative from pattern
    z_derivative = Z_DERIVATIVES[current_tick % Z_DERIVATIVES.shape[0]]

    # Update position
    new_z = z_pos + z_derivative

    # On teleport to 14 (serve), jump to the correct point in the cycle
    # Otherwise just continue the cycle
    new_tick = jnp.where(
        serve_hit,
        jnp.array(SERVE_INDEX - 1),
        (current_tick + 1) % Z_DERIVATIVES.shape[0]
    )

    # Ensure z never goes below 0
    new_z = jnp.maximum(new_z, 0)

    # overwrite new z with 14 which seems to be the standard start point of the z movement
    final_new_z = jnp.where(serve_hit, 14, new_z)

    # on serve, set the z_deriv to 0
    final_z_deriv = jnp.where(serve_hit, 0, z_derivative)

    return final_new_z, final_z_deriv, new_tick


@partial(jax.jit, static_argnums=())
def update_y_position(y_pos: chex.Array, current_tick: chex.Array, direction: chex.Array) -> Tuple[
    chex.Array, chex.Array]:
    """
    Updates Y position using the derivative pattern.

    Args:
        y_pos: Current Y position
        current_tick: Current index in pattern
        direction: Direction of movement (-1 for down, 1 for up)

    Returns:
        new_y_pos
    """
    # Get current derivative
    y_derivative = Y_DERIVATIVES[current_tick % Y_DERIVATIVES.shape[0]]

    y_derivative = jnp.where(
        direction == 1,
        y_derivative,
        -y_derivative
    )

    # Update position
    new_y = y_pos + y_derivative

    # check if the ball is currently crossing the net, i.e. if the y is in the net range
    crossing_net = jnp.logical_and(
        jnp.greater_equal(new_y, NET_RANGE[0]),
        jnp.less_equal(new_y, NET_RANGE[1])
    )

    # if y_ball_tick is negative, the ball is currently crossing the net. If so, continue counting it down and wait for 4 ticks before teleporting the ball
    new_tick = jnp.where(
        crossing_net,
        jnp.maximum(current_tick - 1, 0),
        (current_tick + 1) % Y_DERIVATIVES.shape[0]
    )

    # if the new_tick is between -4 and 0, set y to 0 (masking)
    new_y = jnp.where(
        jnp.logical_and(jnp.less(new_tick, 0), jnp.greater_equal(new_tick, -4)),
        0,
        new_y
    )

    # if new_tick is smaller than -4, teleport the ball to the other side of the net
    new_y = jnp.where(
        jnp.less(new_tick, -4),
        NET_RANGE[1] + 2,
        new_y
    )

    return new_y, new_tick



def ball_step(state: TennisState, top_collision: chex.Array,
              bottom_collision: chex.Array, action: chex.Array) -> Tuple[
    chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
    """Update ball state for one step.
    Returns:
        Tuple of (new_x, new_y, new_z, x_dir, y_dir, serving, ball_movement_tick)
    """
    serving_side = (state.side_switch_counter >= 2).astype(jnp.int32)
    player_is_server = jnp.equal(state.player_side, serving_side)

    serve_started = jnp.logical_or(
        jnp.logical_and(action == FIRE, player_is_server),  # Player serves with FIRE
        jnp.logical_and(~player_is_server, state.current_tick % 60 == 0)
        # Enemy tries to serve automatically every ~60 ticks
    )
    serve_hit = jnp.logical_and(jnp.logical_or(top_collision, bottom_collision), serve_started)

    # Update Z position first
    new_z, delta_z, new_ball_movement_tick = update_z_position(state.ball_z, state.ball_movement_tick, serve_hit)

    def get_directions():
        # Find out in which direction the ball should go using top_collision and bottom_collision
        y_direction = jnp.where(
            top_collision, 1, jnp.where(bottom_collision, -1, state.ball_y_dir)
        )

        # Determine which player is involved in the collision
        is_player_collision = jnp.logical_xor(
            jnp.logical_and(top_collision, state.player_side == 0),
            jnp.logical_and(bottom_collision, state.player_side == 1)
        )

        # Get the relevant player position and direction based on who was hit
        player_pos = jnp.where(is_player_collision, state.player_x, state.enemy_x)

        # Use the actual stored direction of the player/enemy that was hit
        hitting_entity_dir = jnp.where(
            is_player_collision,
            state.player_direction,  # Use player's current direction
            state.enemy_direction,  # Use enemy's current direction
        )

        # impact point calculation
        # - For right-facing (0): measure from RIGHT edge (player_pos + 13)
        # - For left-facing (1): measure from LEFT edge (player_pos)
        impact_point = jnp.where(
            hitting_entity_dir == 0,
            # Facing right: measure from right edge
            (player_pos + PLAYER_WIDTH) - state.ball_x,  # Pixels from right edge
            # Facing left: measure from left edge
            state.ball_x - player_pos,  # Pixels from left edge
        )

        # Get the pattern index for the appropriate x-movement pattern
        pattern_idx = get_ball_x_pattern(impact_point, hitting_entity_dir)

        # Get the first value from the pattern for immediate use
        x_direction = jnp.where(
            jnp.logical_or(top_collision, bottom_collision),
            X_DIRECTION_ARRAY[pattern_idx],
            state.ball_x_dir,
        )
        paddle_center_x = player_pos + PLAYER_WIDTH / 2  # Get player center
        offset_from_center = (paddle_center_x - (COURT_WIDTH // 2)) / 30
        offset_from_player = (state.ball_x - paddle_center_x) / (PLAYER_WIDTH / 2)

        x_direction =  (offset_from_center / -2) + (offset_from_player/2)

        return x_direction, y_direction, pattern_idx, 0

    def handle_serve():
        serving_entity_x = jnp.where(player_is_server, state.player_x, state.enemy_x)

        serve_x = jnp.where(
            serve_hit,
            serving_entity_x + (PLAYER_WIDTH / 2),  # Center of server paddle
            state.ball_x
        )

        serve_y = jnp.where(
            serve_hit,
            jnp.where(
                serving_side == 0,
                24,  # Top serve position
                159  # Bottom serve position
            ),
            state.ball_y
        )

        x_direction, y_direction, new_x_pattern, _ = jax.lax.cond(
            serve_hit,
            lambda: get_directions(),
            lambda: (0.0, 0, state.ball_x_pattern_idx, 0),
        )

        new_x_counter_idx = jnp.where(serve_hit, 0, state.ball_x_counter)

        return (
            serve_x.astype(jnp.float32),
            serve_y.astype(jnp.float32),
            new_z.astype(jnp.float32),
            delta_z,
            x_direction,
            y_direction,
            ~serve_hit,
            new_ball_movement_tick,
            state.ball_y_tick.astype(jnp.int8),
            new_x_pattern,
            new_x_counter_idx,
            jnp.array(0),
            jnp.array(0.0),
            state.ball_start,
            state.ball_end
        )

    def handle_normal_play():
        # Increment pattern index
        new_x_counter_idx = state.ball_x_counter + 1

        new_y = state.ball_y + state.ball_y_dir * 2
        new_x = state.ball_x + state.ball_x_dir - state.ball_curve

        ball_arc = -3 * jnp.abs(
            jnp.sin((jnp.pi / 70) * state.ball_z)
        )  # Creates slight curve effect
        # Apply arc effect to ball position
        new_x = new_x + ball_arc
        new_y = new_y

        ball_curve = ball_arc
        ball_curve_counter = state.ball_curve_counter + 1

        # if there was a collision, get the new directions
        x_direction, y_direction, new_ball_x_pattern_idx, ball_curve_counter = (
            jax.lax.cond(
                jnp.logical_or(top_collision, bottom_collision),
                lambda: get_directions(),
                lambda: (
                    state.ball_x_dir,
                    state.ball_y_dir,
                    state.ball_x_pattern_idx,
                    ball_curve_counter,
                ),
            )
        )

        ball_start, ball_end = jax.lax.cond(
            jnp.logical_or(top_collision, bottom_collision),
            lambda: (state.enemy_y, state.player_y),
            lambda: (state.ball_start, state.ball_end),
        )

        # in case of collision reset z to 14, set the new_ball_movement_tick to SERVE_INDEX - 1 and the ball_y_tick to 0
        final_new_z = jnp.where(
            jnp.logical_or(top_collision, bottom_collision), 14, new_z
        )

        final_new_ball_movement_tick = jnp.where(
            jnp.logical_or(top_collision, bottom_collision),
            SERVE_INDEX - 1,
            new_ball_movement_tick,
        )

        return (
            new_x.astype(jnp.float32),
            new_y.astype(jnp.float32),
            final_new_z.astype(jnp.float32),
            delta_z,
            x_direction,
            y_direction,
            state.serving,
            final_new_ball_movement_tick,
            jnp.array(0).astype(jnp.int8),
            new_ball_x_pattern_idx,
            new_x_counter_idx,
            ball_curve_counter,
            ball_curve,
            ball_start,
            ball_end
        )

    return jax.lax.cond(
        state.serving, lambda _: handle_serve(), lambda _: handle_normal_play(), None
    )


def player_step(
    state_player_x: chex.Array,
    state_player_y: chex.Array,
    state_player_direction: chex.Array,
    action: chex.Array,
    ball_x: chex.Array,
    side: chex.Array,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """
    Updates player position based on current position and action.

    Args:
        state_player_x: Current x coordinate of player
        state_player_y: Current y coordinate of player
        action: Current player action

    Returns:
        Tuple containing:
            chex.Array: New player x position
            chex.Array: New player y position
    """

    player_direction = state_player_direction

    # Calculate if we need to switch direction (turn around)
    # Switch when player is facing right (0) and ball is too far left
    ball_x = ball_x.astype(jnp.int32)
    should_turn_left = jnp.logical_and(
        player_direction == 0,  # Currently facing right
        ball_x == state_player_x,  # Ball is at the exact same position
    )

    # Switch when player is facing left (1) and ball is too far right
    should_turn_right = jnp.logical_and(
        player_direction == 1, ball_x == state_player_x + 9  # Currently facing left
    )

    # Update direction
    new_direction = jnp.where(
        should_turn_left,
        1,  # Switch to facing left
        jnp.where(
            should_turn_right,
            0,  # Switch to facing right
            player_direction,  # Keep current direction
        ),
    )

    # Teleport player when turning
    new_player_x = jnp.where(
        should_turn_left,
        state_player_x - 8,  # Teleport left by 7 pixels
        jnp.where(
            should_turn_right,
            state_player_x + 8,  # Teleport right by 7 pixels
            state_player_x,  # Keep current position
        ),
    )

    # TODO: adjust the borders of the game according to base implementation
    player_max_left, player_max_right, player_max_top, player_max_bottom = jax.lax.cond(
        jnp.equal(side, 0),
        lambda _: (
            TOP_ENTITY_MAX_LEFT,
            TOP_ENTITY_MAX_RIGHT,
            TOP_ENTITY_MAX_TOP,
            TOP_ENTITY_MAX_BOTTOM
        ),
        lambda _: (
            BOTTOM_ENTITY_MAX_LEFT,
            BOTTOM_ENTITY_MAX_RIGHT,
            BOTTOM_ENTITY_MAX_TOP,
            BOTTOM_ENTITY_MAX_BOTTOM
        ),
        operand=None
    )

    # handle diagonal movement by setting left/right and top/down variables
    up = jnp.any(jnp.array([action == UP, action == UPRIGHT, action == UPLEFT, action == UPFIRE, action == UPRIGHTFIRE,
                            action == UPLEFTFIRE]))
    down = jnp.any(jnp.array(
        [action == DOWN, action == DOWNRIGHT, action == DOWNLEFT, action == DOWNFIRE, action == DOWNRIGHTFIRE,
         action == DOWNLEFTFIRE]))
    left = jnp.any(jnp.array(
        [action == LEFT, action == UPLEFT, action == DOWNLEFT, action == LEFTFIRE, action == UPLEFTFIRE,
         action == DOWNLEFTFIRE]))
    right = jnp.any(jnp.array(
        [action == RIGHT, action == UPRIGHT, action == DOWNRIGHT, action == RIGHTFIRE, action == UPRIGHTFIRE,
         action == DOWNRIGHTFIRE]))

    # check if the player is trying to move left
    player_x = jnp.where(
        jnp.logical_and(left, new_player_x > 0),
        new_player_x - 1,
        new_player_x,
    )

    # check if the player is trying to move right
    player_x = jnp.where(
        jnp.logical_and(right, new_player_x < COURT_WIDTH - 13),
        new_player_x + 1,
        player_x,
    )

    player_x = jnp.clip(player_x, player_max_left, player_max_right)

    # check if the player is trying to move up
    player_y = jnp.where(
        jnp.logical_and(up, state_player_y > 0),
        state_player_y - 1,
        state_player_y,
    )

    # check if the player is trying to move down
    player_y = jnp.where(
        jnp.logical_and(down, state_player_y < COURT_HEIGHT - 23),
        state_player_y + 1,
        player_y,
    )

    player_y = jnp.clip(player_y, player_max_top, player_max_bottom)

    return player_x, player_y, new_direction


def enemy_step(
    state_enemy_x: chex.Array,
    state_enemy_y: chex.Array,
    state_enemy_direction: chex.Array,
    ball_x: chex.Array,
    ball_y: chex.Array,
    side: chex.Array,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """
    Updates enemy position based on current position, direction and ball position.

    Args:
        state_enemy_x: Current x coordinate of enemy
        state_enemy_y: Current y coordinate of enemy
        state_enemy_direction: Current direction of enemy
        ball_x: Current x coordinate of ball
        ball_y: Current y coordinate of ball
        side: Which side the enemy is on

    Returns:
        Tuple containing:
            chex.Array: New enemy x position
            chex.Array: New enemy y position
            chex.Array: New enemy direction
    """

    enemy_direction = state_enemy_direction

    # Calculate if enemy needs to switch direction (turn around)
    # Switch when enemy is facing right (0) and ball is at the exact same position
    should_turn_left = jnp.logical_and(
        enemy_direction == 0,  # Currently facing right
        ball_x == state_enemy_x,  # Ball is at the exact same position
    )

    # Switch when enemy is facing left (1) and ball is at specific offset
    should_turn_right = jnp.logical_and(
        enemy_direction == 1, ball_x == state_enemy_x + 9  # Currently facing left
    )

    # Update direction
    new_direction = jnp.where(
        should_turn_left,
        1,  # Switch to facing left
        jnp.where(
            should_turn_right,
            0,  # Switch to facing right
            enemy_direction,  # Keep current direction
        ),
    )

    # Teleport enemy when turning
    new_enemy_x = jnp.where(
        should_turn_left,
        state_enemy_x - 8,  # Teleport left by 8 pixels
        jnp.where(
            should_turn_right,
            state_enemy_x + 8,  # Teleport right by 8 pixels
            state_enemy_x,  # Keep current position
        ),
    )

    # Get appropriate bounds based on side
    enemy_max_left, enemy_max_right, enemy_max_bottom, enemy_max_top = jax.lax.cond(
        jnp.equal(side, 1),  # Opposite of player's side
        lambda _: (
            TOP_ENTITY_MAX_LEFT,
            TOP_ENTITY_MAX_RIGHT,
            TOP_ENTITY_MAX_BOTTOM,
            TOP_ENTITY_MAX_TOP,
        ),
        lambda _: (
            BOTTOM_ENTITY_MAX_LEFT,
            BOTTOM_ENTITY_MAX_RIGHT,
            BOTTOM_ENTITY_MAX_BOTTOM,
            BOTTOM_ENTITY_MAX_TOP,
        ),
        operand=None,
    )

    ball_x = (
        ball_x.astype(jnp.int32) - 6
    )  # force the enemy to move the ball to its center

    # Basic AI movement - move towards the ball
    enemy_x = jnp.where(
        ball_x < new_enemy_x,
        new_enemy_x - 1,
        jnp.where(ball_x > new_enemy_x, new_enemy_x + 1, new_enemy_x),
    )

    # Apply bounds
    enemy_x = jnp.clip(enemy_x, enemy_max_left, enemy_max_right)

    # For now, maintain Y position
    enemy_y = state_enemy_y

    return enemy_x, enemy_y, new_direction

def check_if_round_over(player_round_score, enemy_round_score, round_overtime) -> Tuple[chex.Array, chex.Array]:
    """
    A round is over if:
    1. we are not in round_overtime and one player has more than 40 points whilst the other has less than 40
    2. we are in round_overtime and one player has a score difference of 2
    Args:
        player_round_score: Current player score
        enemy_round_score: Current enemy score
        round_overtime: Boolean array that is only true if the round is in overtime

    Returns:
        Tuple containing:
            bool: True if the player won the round; False otherwise
            bool: True if the enemy won the round; False otherwise
    """
    # check if the score is 40-30 or 30-40
    clear_winner_player = jnp.logical_and(
        jnp.logical_not(round_overtime),
        jnp.logical_and(
            jnp.greater(player_round_score, 40),
            jnp.less(enemy_round_score, 40),
        ),
    )

    clear_winner_enemy = jnp.logical_and(
        jnp.logical_not(round_overtime),
        jnp.logical_and(
            jnp.greater(enemy_round_score, 40),
            jnp.less(player_round_score, 40),
        )
    )

    # if its overtime, check if the score difference is 2
    player_win_in_overtime = jnp.logical_and(
        round_overtime,
        player_round_score - enemy_round_score >= 2,
    )

    enemy_win_in_overtime = jnp.logical_and(
        round_overtime,
        enemy_round_score - player_round_score >= 2,
    )

    return jnp.logical_or(clear_winner_player, player_win_in_overtime), jnp.logical_or(clear_winner_enemy, enemy_win_in_overtime)

def check_scoring(state: TennisState) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, bool, chex.Array, chex.Array, chex.Array]:
    """
    Checks if a point was scored and updates the score accordingly.

    Args:
        state: Current game state containing ball and player positions and scores

    Returns:
        Tuple containing:
            chex.Array: Updated player score
            chex.Array: Updated enemy score
            bool: Whether a point was scored in this step
    """
    def get_next_score(current_score, round_score_intervals):
        """Get the next score value from the round_score_intervals array."""
        # Find the current score index in the intervals
        current_index = jnp.argmax(jnp.array(round_score_intervals) == current_score)
        # Ensure we don't go beyond the array bounds
        next_index = jnp.minimum(current_index + 1, len(round_score_intervals) - 1)
        return round_score_intervals[next_index]

    round_score_intervals = jnp.array([0, 15, 30, 40, 41])

    ball_in_field, side = check_ball_in_field(state)

    player_round_score, enemy_round_score, add_point = jax.lax.cond(
        jnp.logical_or(ball_in_field, state.serving),
        lambda _: (state.player_round_score, state.enemy_round_score, False),
        lambda _: (state.player_round_score, state.enemy_round_score, True),
        operand=None,
    )

    # Ball left on player's side, enemy scores
    enemy_scores = jnp.logical_and(add_point, jnp.equal(side, state.player_side))

    # Ball left on enemy's side, player scores
    player_scores = jnp.logical_and(add_point, jnp.not_equal(side, state.player_side))

    # Update scores
    player_round_score = jnp.where(
        player_scores,
        get_next_score(state.player_round_score, round_score_intervals),
        state.player_round_score
    )

    enemy_round_score = jnp.where(
        enemy_scores,
        get_next_score(state.enemy_round_score, round_score_intervals),
        state.enemy_round_score
    )

    # if we are in round_overtime, dont set the round_scores to the next step but simply add 1 to it (we are only looking at differences now)
    player_round_score = jnp.where(
        jnp.logical_and(player_scores, state.round_overtime),
        state.player_round_score + 1,
        player_round_score
    )

    enemy_round_score = jnp.where(
        jnp.logical_and(enemy_scores, state.round_overtime),
        state.enemy_round_score + 1,
        enemy_round_score
    )

    # recheck the round_overtime value
    round_overtime = jnp.logical_or(
        state.round_overtime,
        jnp.logical_and(
            jnp.greater_equal(player_round_score, 40),
            jnp.greater_equal(enemy_round_score, 40),
        )
    )

    # check if one of the two won this round
    player_won_round, enemy_won_round = check_if_round_over(player_round_score, enemy_round_score, round_overtime)

    # if either player or enemy won a round, reset the scores
    player_round_score = jnp.where(
        jnp.logical_or(player_won_round, enemy_won_round),
        0,
        player_round_score,
    )

    enemy_round_score = jnp.where(
        jnp.logical_or(player_won_round, enemy_won_round),
        0,
        enemy_round_score,
    )

    # if the round was won, we set the round_overtime to false
    round_overtime = jnp.where(
        jnp.logical_or(player_won_round, enemy_won_round),
        False,
        round_overtime,
    )

    newly_round_overtime = jnp.logical_and(
        round_overtime,
        jnp.logical_not(state.round_overtime)
    )

    # if we are newly overtime set the scores to 0
    player_round_score = jnp.where(
        newly_round_overtime,
        0,
        player_round_score
    )

    enemy_round_score = jnp.where(
        newly_round_overtime,
        0,
        enemy_round_score
    )

    # Increment side_switch_counter when a round is won (circuit through 0-3)
    new_side_switch_counter = jnp.where(
        jnp.logical_or(player_won_round, enemy_won_round),
        jnp.mod(state.side_switch_counter + 1, 4),  # Cycle through 0,1,2,3
        state.side_switch_counter
    )

    new_player_side = jnp.mod(new_side_switch_counter, 2) # 0 for top, 1 for bottom

    # increase the overall game scoring accordingly
    player_score = jnp.where(player_won_round, state.player_score + 1, state.player_score)

    enemy_score = jnp.where(enemy_won_round, state.enemy_score + 1, state.enemy_score)

    return player_round_score, enemy_round_score, player_score, enemy_score, add_point, round_overtime, new_side_switch_counter, new_player_side


def check_ball_in_field(state: TennisState) -> Tuple[chex.Array, chex.Array]:
    """
    Checks if the ball is in the field
    Args:
        state: Current game state containing ball and player positions and scores

    Returns:
        Tuple containing:
            bool: True if ball is in the field; False otherwise
            int: 0 if ball left the ball in the top part of the field; 1 if the ball left on the bottom part of the field
    """
    thresh_hold = 1e-9
    left_line_val = (NET_BOTTOM_LEFT[0] - NET_TOP_LEFT[0]) * (
        state.ball_y - NET_TOP_LEFT[1]
    ) - (NET_BOTTOM_LEFT[1] - NET_TOP_LEFT[1]) * (state.ball_x - NET_TOP_LEFT[0])
    right_line_val = (NET_BOTTOM_RIGHT[0] - NET_TOP_RIGHT[0]) * (
        state.ball_y - NET_TOP_RIGHT[1]
    ) - (NET_BOTTOM_RIGHT[1] - NET_TOP_RIGHT[1]) * (state.ball_x - NET_TOP_RIGHT[0])

    in_field_sides = jnp.logical_and(
        jnp.less_equal(left_line_val, -thresh_hold),
        jnp.greater_equal(right_line_val, thresh_hold),
    )

    in_field_top = (state.ball_y + state.ball_z) >= NET_TOP_LEFT[1]

    in_field_bottom = (state.ball_y + state.ball_z) <= NET_BOTTOM_LEFT[1]

    side = jax.lax.cond(
        jnp.less_equal((state.ball_y + state.ball_z), NET_RANGE[0]),
        lambda _: 0,
        lambda _: 1,
        operand=None,
    )

    return (
        jnp.logical_or(
            jnp.logical_and(
                jnp.logical_and(in_field_sides, in_field_top), in_field_bottom
            ),
            jnp.greater(state.ball_z, 0),
        ),
        side,
    )


def check_collision(
    player_x: chex.Array,
    player_y: chex.Array,
    state: TennisState,
    just_hit: chex.Array,
    serving: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """
    Checks if a collision occurred between the ball and the players,
    categorizing collisions as top or bottom based on players' positions
    relative to the net.

    Args:
        player_x: Current x coordinate of player
        player_y: Current y coordinate of player
        state: Current game state
        just_hit: Whether the ball was just hit
        serving: Whether we're in serving state

    Returns:
        Tuple containing:
            chex.Array: Whether a collision occurred with the player above the net
            chex.Array: Whether a collision occurred with the player below the net
            chex.Array: Updated just_hit state
    """
    # Define valid hit zones
    TOP_VALID_Y = (20, 113)
    BOTTOM_VALID_Y = (113, 178)
    TOP_VALID_Z = (0, 30)
    BOTTOM_VALID_Z = (0, 30)

    def check_hit_zone(
        y_pos: chex.Array, z_pos: chex.Array, valid_y: tuple, valid_z: tuple
    ) -> chex.Array:
        y_valid = jnp.logical_and(y_pos >= valid_y[0], y_pos <= valid_y[1])
        z_valid = jnp.logical_and(z_pos >= valid_z[0], z_pos <= valid_z[1])
        return jnp.logical_and(y_valid, z_valid)

    # Check basic overlaps
    player_overlap = jnp.logical_and(
        jnp.logical_and(
            state.ball_x < player_x + PLAYER_WIDTH, state.ball_x + BALL_SIZE > player_x
        ),
        jnp.logical_and(
            state.ball_y < player_y + PLAYER_HEIGHT, state.ball_y + BALL_SIZE > player_y
        ),
    )

    enemy_overlap = jnp.logical_and(
        jnp.logical_and(
            state.ball_x < state.enemy_x + PLAYER_WIDTH,
            state.ball_x + BALL_SIZE > state.enemy_x,
        ),
        jnp.logical_and(
            state.ball_y < state.enemy_y + PLAYER_HEIGHT,
            state.ball_y + BALL_SIZE > state.enemy_y,
        ),
    )

    # Combine with valid hit zones based on player side
    player_hit_zone = jnp.where(
        state.player_side == 0,
        check_hit_zone(state.ball_y, state.ball_z, TOP_VALID_Y, TOP_VALID_Z),
        check_hit_zone(state.ball_y, state.ball_z, BOTTOM_VALID_Y, BOTTOM_VALID_Z)
    )

    enemy_hit_zone = jnp.where(
        state.player_side == 0,
        check_hit_zone(state.ball_y, state.ball_z, BOTTOM_VALID_Y, BOTTOM_VALID_Z),
        check_hit_zone(state.ball_y, state.ball_z, TOP_VALID_Y, TOP_VALID_Z)
    )

    # Check collisions with direction constraints
    # Direction checks also need to be flipped based on player side
    player_dir_valid = jnp.where(
        state.player_side == 0,
        jnp.less_equal(state.ball_y_dir, 0),
        jnp.greater_equal(state.ball_y_dir, 0),
    )

    enemy_dir_valid = jnp.where(
        state.player_side == 0,
        jnp.greater_equal(state.ball_y_dir, 0),
        jnp.less_equal(state.ball_y_dir, 0),
    )

    player_collision = jnp.logical_and(
        jnp.logical_and(player_overlap, player_hit_zone), player_dir_valid
    )

    enemy_collision = jnp.logical_and(
        jnp.logical_and(enemy_overlap, enemy_hit_zone), enemy_dir_valid
    )

    # Return collisions in proper order (top, bottom) based on player side
    top_collision = jnp.where(state.player_side == 0, player_collision, enemy_collision)

    bottom_collision = jnp.where(
        state.player_side == 0, enemy_collision, player_collision
    )

    return top_collision, bottom_collision


def before_serve(state: TennisState) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """
    Plays the idle animation of the ball before serving.

    Args:
        state: Current game state

    Returns:
        chex.Array: Updated ball z position
        chex.Array: Updated ball movement tick
        chex.Array: Updated delta z
    """

    # idle movement of the ball
    idle_movement = jnp.array(
        [
            # Initial fast ascent (7 frames)
            3,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            # Transition and slowdown (7 frames)
            1,
            1,
            2,
            1,
            1,
            1,
            0,
            # Peak hover (9 frames)
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            # Initial descent (6 frames)
            -1,
            -1,
            0,
            -1,
            -1,
            -1,
            # Transition (3 frames)
            -2,
            -1,
            -1,
            # Fast descent (7 frames)
            -2,
            -2,
            -2,
            -2,
            -2,
            -2,
            -2,
            # Final drop (2 frames)
            -3,
        ]
    )

    # Idle animation direction depends on serving side
    serving_side = (state.side_switch_counter >= 2).astype(jnp.int32)

    # Possibly flip animation pattern based on serving side
    delta_z = idle_movement[state.ball_movement_tick % idle_movement.shape[0]]

    # get the speed of the next z movement using the idle movement pattern
    new_z = state.ball_z + delta_z

    # update the ball_movement_tick
    new_ball_movement_tick = state.ball_movement_tick + 1

    # Reset movement tick at pattern end to maintain synchronization
    new_ball_movement_tick = new_ball_movement_tick % idle_movement.shape[0]

    return new_z, new_ball_movement_tick, delta_z


class JaxTennis(JaxEnvironment[TennisState, TennisObservation, TennisInfo]):
    def __init__(self):
        super().__init__()

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(
        self,
        state: TennisState
    ) -> TennisObservation:
        # create player
        player = EntityPosition(
            x=state.player_x,
            y=state.player_y,
            width=jnp.array(PLAYER_WIDTH),
            height=jnp.array(PLAYER_HEIGHT),
        )

        # create enemy
        enemy = EntityPosition(
            x=state.enemy_x,
            y=state.enemy_y,
            width=jnp.array(PLAYER_WIDTH),
            height=jnp.array(PLAYER_HEIGHT),
        )

        # create ball
        ball = EntityPosition(
            x=state.ball_x,
            y=state.ball_y,
            width=jnp.array(BALL_SIZE),
            height=jnp.array(BALL_SIZE),
        )

        # create shadow
        shadow = EntityPosition(
            x=state.shadow_x,
            y=state.shadow_y,
            width=jnp.array(BALL_SIZE),
            height=jnp.array(BALL_SIZE),
        )

        # return the obs object
        return TennisObservation(
            player=player,
            enemy=enemy,
            ball=ball,
            ball_shadow = shadow,
            player_round_score=state.player_round_score,
            enemy_round_score=state.enemy_round_score,
            player_score=state.player_score,
            enemy_score=state.enemy_score
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(
        self,
        state: TennisState
    ) -> TennisInfo:
        return TennisInfo(
            serving=state.serving,
            player_side=state.player_side,
            current_tick=state.current_tick,
            ball_direction=state.ball_y_dir
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(
        self,
        previous_state: TennisState,
        state: TennisState
    ) -> float:
        return (state.player_score - state.enemy_score) - (previous_state.player_score - previous_state.enemy_score)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(
        self,
        state: TennisState
    ) -> chex.Array:
        """
        Returns true only if:
        1. one score is 6 and the other score is lower than 5
        2. the game_overtime is true and the score difference is == 2
        Args:
            state: Current game state

        Returns: boolean indicating if the game is over
        """

        # check if the score is 6-4 or 4-6
        clear_winner = jnp.logical_and(
            jnp.logical_not(state.game_overtime),
            jnp.logical_or(
                jnp.logical_and(
                    jnp.greater_equal(state.player_score, 6),
                    jnp.less(state.enemy_score, 5),
                ),
                jnp.logical_and(
                    jnp.greater_equal(state.enemy_score, 6),
                    jnp.less(state.player_score, 5),
                ),
            ),
        )

        # if its overtime, check if the score difference is 2
        win_in_overtime = jnp.logical_and(
            state.game_overtime,
            jnp.abs(state.player_score - state.enemy_score) >= 2,
        )

        return jnp.logical_or(clear_winner, win_in_overtime)

    @partial(jax.jit, static_argnums=(0,))
    def calculate_player_side(self, counter: chex.Array) -> chex.Array:
        """Calculate player side based on counter value (0-3)
        Returns 0 for top, 1 for bottom
        """
        return jnp.mod(counter, 2)  # Even = top, Odd = bottom

    @partial(jax.jit, static_argnums=(0,))
    def calculate_serving_side(self, counter: chex.Array) -> chex.Array:
        """Calculate serving side based on counter value (0-3)
        Returns 0 for top serve, 1 for bottom serve
        """
        return jnp.where(counter < 2, 0, 1)  # First two states = top serve

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key=None) -> Tuple[TennisObservation, TennisState]:
        # Use provided internal_counter or default to 0
        internal_counter = jnp.array(0)

        # Calculate sides based on internal_counter
        player_side = self.calculate_player_side(internal_counter)
        serving_side = self.calculate_serving_side(internal_counter)

        # Set positions based on sides
        player_x = jnp.array(TOP_START_X).astype(jnp.int32)
        player_y = jnp.where(
            player_side == 0,
            jnp.array(TOP_START_Y),
            jnp.array(BOT_START_Y)
        ).astype(jnp.int32)

        enemy_x = jnp.array(BOT_START_X).astype(jnp.int32)
        enemy_y = jnp.where(
            player_side == 0,
            jnp.array(BOT_START_Y),
            jnp.array(TOP_START_Y)
        ).astype(jnp.int32)

        # Initialize ball based on serving side
        ball_y = jnp.where(
            serving_side == 0,
            jnp.array(BALL_START_Y),
            jnp.array(COURT_HEIGHT - BALL_START_Y)
        ).astype(jnp.float32)

        """Resets game to initial state."""
        reset_state = TennisState(
            player_x=player_x,
            player_y=player_y,
            player_direction=jnp.array(0).astype(jnp.int32),
            enemy_x=enemy_x,
            enemy_y=enemy_y,
            enemy_direction=jnp.array(0).astype(jnp.int32),
            ball_x=jnp.array(BALL_START_X).astype(jnp.float32),
            ball_y=ball_y,
            ball_z=jnp.array(BALL_START_Z).astype(jnp.float32),
            ball_curve_counter=jnp.array(0),
            ball_x_dir=jnp.array(1).astype(jnp.float32),
            ball_y_dir=jnp.array(1).astype(jnp.int32),
            shadow_x=jnp.array(0).astype(jnp.int32),
            shadow_y=jnp.array(0).astype(jnp.int32),
            ball_movement_tick=jnp.array(0).astype(jnp.int32),
            player_score=jnp.array(0).astype(jnp.int32),
            enemy_score=jnp.array(0).astype(jnp.int32),
            player_round_score=jnp.array(0).astype(jnp.int32),
            enemy_round_score=jnp.array(0).astype(jnp.int32),
            serving=jnp.array(1).astype(jnp.bool),  # boolean for serve state
            just_hit=jnp.array(0).astype(jnp.bool),  # boolean for just hit state
            player_side=player_side,
            ball_was_infield=jnp.array(0).astype(jnp.bool),
            current_tick=jnp.array(0).astype(jnp.int32),
            ball_y_tick=jnp.array(0).astype(jnp.int8),
            ball_x_pattern_idx=jnp.array(
                -1
            ),  # can be any value since this should be overwritten before being used (in case its not, it will throw an error now)
            ball_x_counter=jnp.array(0),
            ball_curve=jnp.array(0.0),
            round_overtime=jnp.array(0).astype(jnp.bool),
            game_overtime=jnp.array(0).astype(jnp.bool),
            ball_start=jnp.array(TOP_START_Y).astype(jnp.int32),
            ball_end=jnp.array(BOT_START_Y).astype(jnp.int32),
            side_switch_counter=jnp.array(0).astype(jnp.int32),
            player_hit=jnp.array(False),
            enemy_hit=jnp.array(False),
        )
        return self._get_observation(reset_state), reset_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: TennisState, action: chex.Array) -> Tuple[
        TennisObservation, TennisState, float, bool, TennisInfo]:
        def normal_play():
            """Executes one game step."""
            # Update player position
            player_x, player_y, new_player_direction = player_step(
                state.player_x,
                state.player_y,
                state.player_direction,
                action,
                state.ball_x,
                state.player_side,
            )

            # check if there was a collision
            top_collision, bottom_collision = check_collision(
                player_x,
                player_y,
                state,
                state.just_hit,
                state.serving,
            )

            # Get information about serving
            serving_side = (state.side_switch_counter >= 2).astype(jnp.int32)
            player_is_server = jnp.equal(state.player_side, serving_side)

            # For serving state, only register a hit when serve is executed
            serve_started = jnp.logical_or(
                jnp.logical_and(action == FIRE, player_is_server),  # Player serves with FIRE
                jnp.logical_and(~player_is_server, state.current_tick % 60 == 0)  # Enemy auto-serve
            )
            serve_hit = jnp.logical_and(jnp.logical_or(top_collision, bottom_collision), serve_started)

            # Determine hit variables differently based on serve state
            if_serving_player_hit = jnp.logical_and(serve_hit, player_is_server)
            if_serving_enemy_hit = jnp.logical_and(serve_hit, ~player_is_server)

            # For normal play, detect hits based on collisions and player position
            if_normal_player_hit = jnp.logical_and(
                ~state.serving,
                jnp.logical_xor(
                    jnp.logical_and(top_collision, state.player_side == 0),
                    jnp.logical_and(bottom_collision, state.player_side == 1)
                )
            )

            if_normal_enemy_hit = jnp.logical_and(
                ~state.serving,
                jnp.logical_xor(
                    jnp.logical_and(top_collision, state.player_side == 1),
                    jnp.logical_and(bottom_collision, state.player_side == 0)
                )
            )

            # Combine serve and normal play conditions
            player_hit = jnp.logical_or(if_serving_player_hit, if_normal_player_hit)
            enemy_hit = jnp.logical_or(if_serving_enemy_hit, if_normal_enemy_hit)

            # Update ball position and velocity
            (
                ball_x,
                ball_y,
                ball_z,
                delta_z,
                new_ball_x_dir,
                new_ball_y_dir,
                serve,
                updated_ball_movement_tick,
                new_ball_y_tick,
                new_x_ball_pattern_idx,
                new_x_ball_id,
                ball_curve_counter,
                ball_curve,
                ball_start,
                ball_end,
            ) = ball_step(
                state,
                top_collision,
                bottom_collision,
                action,
            )

            # if nothing is happening, play the idle animation of the ball
            ball_z, new_ball_movement_tick, delta_z = jax.lax.cond(
                serve,
                lambda: before_serve(state),
                lambda: (ball_z, updated_ball_movement_tick, delta_z),
            )

            # calculate the z into the ball_y
            ball_y = jnp.where(serve, ball_y - delta_z, ball_y)

            ball_was_infield = jax.lax.cond(
                jnp.logical_or(
                    state.ball_was_infield,
                    jnp.logical_and(
                        jnp.greater_equal(ball_y, NET_TOP_LEFT[1]),
                        jnp.less_equal(ball_y, NET_BOTTOM_LEFT[1]),
                    ),
                ),
                lambda _: True,
                lambda _: False,
                operand=None,
            )

            # Check scoring
            player_round_score, enemy_round_score, player_game_score, enemy_game_score, point_scored, round_overtime, new_side_switch_counter, new_player_side = check_scoring(
                state)

            enemy_x, enemy_y, new_enemy_direction = enemy_step(
                state.enemy_x,
                state.enemy_y,
                state.enemy_direction,
                ball_x,
                ball_y,
                state.player_side,
            )

            newly_overtime = jnp.logical_and(
                ~state.round_overtime,
                jnp.logical_and(
                    jnp.equal(player_game_score, 6),
                    jnp.equal(enemy_game_score, 6),
                )
            )

            # check if the game is in overtime (i.e. if the score is 6-6 set the flag to true)
            game_overtime = jnp.logical_or(
                state.game_overtime,
                jnp.logical_and(
                    jnp.equal(player_game_score, 6),
                    jnp.equal(enemy_game_score, 6),
                )
            )

            # in case its newly overtime, reset the _game_scores to 0
            player_game_score = jnp.where(
                newly_overtime,
                0,
                player_game_score
            )

            enemy_game_score = jnp.where(
                newly_overtime,
                0,
                enemy_game_score
            )

            _, reset_state = self.reset()

            # Check if player side has changed
            side_changed = jnp.not_equal(state.player_side, new_player_side)

            # When sides change after round ends, reset player and enemy positions
            player_y = jnp.where(
                side_changed,
                jnp.where(
                    new_player_side == 0,
                    jnp.array(TOP_START_Y),  # Reset to top position
                    jnp.array(BOT_START_Y)  # Reset to bottom position
                ),
                player_y
            )

            enemy_y = jnp.where(
                side_changed,
                jnp.where(
                    new_player_side == 0,
                    jnp.array(BOT_START_Y),  # Reset to bottom position
                    jnp.array(TOP_START_Y)  # Reset to top position
                ),
                enemy_y
            )

            # if its serve, block the y movement of player and enemy (depending on the side)
            player_y = jnp.where(serve, jnp.where(new_player_side == 0, TOP_START_Y, BOT_START_Y), player_y)

            # if its serve, block the y movement of player and enemy
            enemy_y = jnp.where(serve, jnp.where(new_player_side == 0, BOT_START_Y, TOP_START_Y), enemy_y)

            # if the game is frozen, return the current state
            serve = jax.lax.cond(
                jnp.logical_or(serve, jnp.logical_and(point_scored, ball_was_infield)),
                lambda _: True,
                lambda _: False,
                operand=None,
            )

            new_serving_side = (new_side_switch_counter >= 2).astype(jnp.int32)
            # Calculate correct ball position based on new serving side
            side_corrected_y = jnp.where(
                new_serving_side == 0,
                jnp.array(BALL_START_Y),  # Top position
                jnp.array(COURT_HEIGHT - BALL_START_Y)  # Bottom position
            ).astype(jnp.float32)

            (
                ball_x,
                ball_y,
                ball_z,
                player_x,
                player_y,
                enemy_x,
                enemy_y,
                ball_was_infield,
                current_tick,
                new_ball_movement_tick,
                new_ball_y_tick,
            ) = jax.lax.cond(
                jnp.logical_and(point_scored, ball_was_infield),
                lambda _: (
                    reset_state.ball_x,
                    side_corrected_y,
                    reset_state.ball_z,
                    reset_state.player_x,
                    reset_state.player_y,
                    reset_state.enemy_x,
                    reset_state.enemy_y,
                    False,
                    -1,
                    reset_state.ball_movement_tick,
                    reset_state.ball_y_tick,
                ),
                lambda _: (
                    ball_x,
                    ball_y,
                    ball_z,
                    player_x,
                    player_y,
                    enemy_x,
                    enemy_y,
                    ball_was_infield,
                    state.current_tick,
                    new_ball_movement_tick,
                    new_ball_y_tick,
                ),
                operand=None,
            )

            # make sure the ball_y is not going negative, if it does, reset it to 0
            ball_y = jnp.maximum(ball_y, 0)

            calculated_state = TennisState(
                player_x=player_x,
                player_y=player_y,
                player_direction=new_player_direction,
                enemy_x=enemy_x,
                enemy_y=enemy_y,
                enemy_direction=new_enemy_direction,
                ball_x=ball_x,
                ball_y=ball_y,
                ball_z=ball_z,
                ball_curve_counter=ball_curve_counter,
                ball_x_dir=new_ball_x_dir,
                ball_y_dir=new_ball_y_dir,
                shadow_x=ball_x.astype(jnp.int32),
                shadow_y=(((ball_y + ball_z + 1) // 2) * 2).astype(jnp.int32),
                ball_movement_tick=new_ball_movement_tick,
                player_round_score=player_round_score,
                enemy_round_score=enemy_round_score,
                player_score=player_game_score,
                enemy_score=enemy_game_score,
                serving=serve,
                just_hit=jnp.array(False),
                player_side=new_player_side,
                ball_was_infield=ball_was_infield,
                current_tick=current_tick + 1,
                ball_y_tick=new_ball_y_tick.astype(jnp.int8),
                ball_x_pattern_idx=new_x_ball_pattern_idx,
                ball_x_counter=new_x_ball_id,
                ball_curve=ball_curve,
                round_overtime=round_overtime,
                game_overtime=game_overtime,
                ball_start=ball_start,
                ball_end=ball_end,
                side_switch_counter=new_side_switch_counter,
                player_hit=player_hit,
                enemy_hit=enemy_hit
            )

            returned_state = jax.lax.cond(
                state.current_tick < WAIT_AFTER_GOAL,
                lambda: state._replace(current_tick=state.current_tick + 1),
                lambda: calculated_state,
            )

            return  self._get_observation(returned_state), returned_state, self._get_reward(state, returned_state), self._get_done(returned_state), self._get_info(returned_state)

        def game_over_freeze():
            """Freezes the game after it's over."""
            return self._get_observation(state), state, self._get_reward(state, state), jnp.bool(True), self._get_info(
                state)

        return jax.lax.cond(
            self._get_done(state),
            lambda: game_over_freeze(),
            lambda: normal_play(),
        )

class AnimatorState(NamedTuple):
    r_x: chex.Array
    r_y: chex.Array
    r_f: chex.Array
    r_bat_f: chex.Array
    b_x: chex.Array
    b_y: chex.Array
    b_f: chex.Array
    b_bat_f: chex.Array


OFFSET_BAT_Y = jnp.array([7, 7, 5, 3])

from jaxatari.renderers import AtraJaxisRenderer

class TennisRenderer(AtraJaxisRenderer):

    @partial(jax.jit, static_argnums=(0,))
    def next_body_frame(self, diff_x, diff_y, frame):

        # Condition 1: r_x - state.player_x > 0 or r_y - state.player_y > 0
        condition1 = (diff_x > 0) | (diff_y > 0)

        # Condition 2: r_x - state.player_x == 0 and r_y - state.player_y == 0
        condition2 = (diff_x == 0) & (diff_y == 0)

        # Calculate next frame based on conditions
        next_frame = jnp.where(
            condition1, (frame + 1) % 16, jnp.where(condition2, 12, (frame - 1) % 16)
        )

        return next_frame

    @partial(jax.jit, static_argnums=(0,))
    def bat_position(self, body_x, body_y, body_direction, frame):
        key_frame = frame // 4
        offset_x = 8

        offset_y = OFFSET_BAT_Y[key_frame]
        bat_x = jnp.where(body_direction, body_x - offset_x, body_x + offset_x)
        bat_y = body_y + offset_y
        return bat_x, bat_y

    @partial(jax.jit, static_argnums=(0,))
    def next_bat_frame(self, frame, hit):
        cond = hit | (frame != 0)
        return jnp.where(cond, (frame + 1) % 16, 0)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state, animator_state):

        # render background
        raster = jnp.zeros((COURT_WIDTH, COURT_HEIGHT, 3))
        raster = aj.render_at(raster, 0, 0, BG)

        flip_player = state.ball_x <= state.player_x
        flip_enemy = state.ball_x <= state.enemy_x
        # render player
        raster = aj.render_at(
            raster,
            state.player_x,
            state.player_y,
            PL_R[animator_state.r_f // 4],
            flip_horizontal=flip_player,
        )

        # TODO remove
        def get_bounding_box(width, height, border_color=(255, 0, 0, 255), border_thickness=1):
            """
            Creates a bounding box sprite with transparent interior and colored border.

            Args:
                width: Width of the bounding box (static)
                height: Height of the bounding box (static)
                border_color: RGBA tuple for border color (default: red)
                border_thickness: Pixel width of the border (default: 1)

            Returns:
                Array of shape (height, width, 4) with RGBA values
            """
            # Create empty RGBA array
            sprite = jnp.zeros((width, height, 4), dtype=jnp.uint8)

            # Create border mask
            border_mask = jnp.zeros((width, height), dtype=bool)

            # Set border pixels to True
            border_mask = border_mask.at[:border_thickness, :].set(True)  # Top border
            border_mask = border_mask.at[-border_thickness:, :].set(True)  # Bottom border
            border_mask = border_mask.at[:, :border_thickness].set(True)  # Left border
            border_mask = border_mask.at[:, -border_thickness:].set(True)  # Right border

            # Apply border color where mask is True
            sprite = jnp.where(
                border_mask[..., None],  # Expand for RGBA channels
                jnp.array(border_color, dtype=jnp.uint8),
                sprite
            )

            return sprite


        raster = aj.render_at(
            raster,
            state.player_y,
            state.player_x,
            get_bounding_box(PLAYER_WIDTH, PLAYER_HEIGHT),
        )

        # render enemy
        raster = aj.render_at(
            raster,
            state.enemy_x,
            state.enemy_y,
            PL_B[animator_state.b_f // 4],
            flip_horizontal=flip_enemy,
        )

        raster = aj.render_at(
            raster,
            state.enemy_y,
            state.enemy_x,
            get_bounding_box(ENEMY_WIDTH, ENEMY_HEIGHT),
        )

        # render ball
        raster = aj.render_at(raster, state.ball_x, state.ball_y, BALL)

        # render ball shade

        raster = aj.render_at(raster, state.shadow_x, state.shadow_y, BALL_SHADE)

        # render player bat
        r_bat_x, r_bat_y = self.bat_position(
            state.player_x,
            state.player_y,
            flip_player,
            animator_state.r_bat_f,
        )

        raster = aj.render_at(
            raster,
            r_bat_x,
            r_bat_y,
            BAT_R[animator_state.r_bat_f // 4],
            flip_horizontal=flip_player,
        )

        # render enemy bat

        b_bat_x, b_bat_y = self.bat_position(
            state.enemy_x, state.enemy_y, flip_enemy, animator_state.b_bat_f
        )
        raster = aj.render_at(
            raster,
            b_bat_x,
            b_bat_y,
            BAT_B[animator_state.b_bat_f // 4],
            flip_horizontal=flip_enemy,
        )

        # render scores
        
        r_score_array = aj.int_to_digits(state.player_round_score, max_digits = 2)
        b_score_array = aj.int_to_digits(state.enemy_round_score, max_digits = 2)
        
        raster = aj.render_label(raster, 60, 10, r_score_array, DIGITS_R, spacing=7)
        raster = aj.render_label(raster, 90, 10, b_score_array, DIGITS_B, spacing=7)

        # state transition

        next_r_f = self.next_body_frame(
            state.player_x - animator_state.r_x,
            state.player_y - animator_state.r_y,
            animator_state.r_f,
        )
        next_b_f = self.next_body_frame(
            state.enemy_x - animator_state.b_x,
            state.enemy_y - animator_state.b_y,
            animator_state.b_f,
        )

        new_animator_state = AnimatorState(
            r_x=state.player_x,
            r_y=state.player_y,
            r_f=next_r_f,
            r_bat_f=self.next_bat_frame(animator_state.r_bat_f, state.player_hit),
            b_x=state.enemy_x,
            b_y=state.enemy_y,
            b_f=next_b_f,
            b_bat_f=self.next_bat_frame(animator_state.b_bat_f, state.enemy_hit),
        )

        return raster, new_animator_state


if __name__ == "__main__":
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((COURT_WIDTH * 4, COURT_HEIGHT * 4))
    pygame.display.set_caption("Tennis Game")
    clock = pygame.time.Clock()

    # Create game instance
    game = JaxTennis()

    # Initialize renderer
    renderer = TennisRenderer()
    animator_state = AnimatorState(
        r_x=0, r_y=0, r_f=12, r_bat_f=0, b_x=0, b_y=0, b_f=12, b_bat_f=0
    )

    # JIT compile main functions
    jitted_step = jax.jit(game.step)
    jitted_reset = jax.jit(game.reset)

    # Main game loop structure
    curr_obs, curr_state = jitted_reset()
    running = True
    frame_by_frame = False
    frameskip = 1
    counter = 1

    list_of_y = []
    list_of_z = []

    while running:
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
                        action = get_human_action()
                        obs, curr_state, reward, done, info = jitted_step(curr_state, action)
                        # print the current game scores
                        print(f"Player: {curr_state.player_score} - Enemy: {curr_state.enemy_score}")

        if not frame_by_frame:
            if counter % frameskip == 0:
                # Get action (to be implemented with proper controls)
                action = get_human_action()
                obs, curr_state, reward, done, info = jitted_step(curr_state, action)

        raster, animator_state = renderer.render(curr_state, animator_state)
        aj.update_pygame(screen, raster, 4, COURT_WIDTH, COURT_HEIGHT)

        counter += 1
        clock.tick(30)

    pygame.quit()
