from functools import partial
from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
import chex
import pygame

from jaxatari.environment import JaxEnvironment

# Constants for game environment
WINDOW_WIDTH = 160 * 3
WINDOW_HEIGHT = 210 * 3

# Colors from the original game
BACKGROUND_COLOR = (0, 0, 0)  # Black background
PLAYER_COLOR = (200, 72, 72)  # Red paddle
BALL_COLOR = (200, 72, 72)  # Red ball
WALL_COLOR = (142, 142, 142)  # Grey walls

# Block colors in order from top to bottom
BLOCK_COLORS = [
    (200, 72, 72),  # Red
    (198, 108, 58),  # Light Red
    (180, 122, 48),  # Orange
    (162, 162, 42),  # Yellow
    (72, 160, 72),  # Green
    (66, 72, 200),  # Blue
]

# Object sizes and positions
PLAYER_SIZE = (16, 4)  # Width, Height of paddle
PLAYER_SIZE_SMALL = (12, 4)
BALL_SIZE = (2, 4)  # Width, Height of ball
BLOCK_SIZE = (8, 6)  # Width, Height of blocks

# Wall positions and sizes
WALL_TOP_Y = 17
WALL_TOP_HEIGHT = 15
WALL_SIDE_WIDTH = 8

# Initial positions
PLAYER_START_X = 99
PLAYER_START_Y = 189
BALL_START_X = jnp.array([16, 78, 80, 142])
BALL_START_Y = 122

# Game boundaries
PLAYER_X_MIN = WALL_SIDE_WIDTH
PLAYER_X_MAX = 160 - PLAYER_SIZE[0]

# Player speed and acceleration
PLAYER_MAX_SPEED = 6
PLAYER_ACCELERATION = jnp.array([3, 2, -1, 1, 1])
PLAYER_WALL_ACCELERATION = jnp.array([1, 2, 1, 1, 1])

# Block layout
BLOCKS_PER_ROW = 18
NUM_ROWS = 6
BLOCK_START_Y = 57  # Starting Y position of first row
BLOCK_START_X = 8  # Starting X position of blocks

NUM_LIVES = 5

# Ball speed
BALL_VELOCITIES_ABS = jnp.array([
    [[1, 1], [1, 1]],  # Base speed
    [[2, 1], [1, 1]],  # Left hit or Right hit
    [[1, 2], [1, 1]],  # Middle left, middle, middle right hit
    [[2, 2], [2, 2]],  # Speed after 12th consecutive hit
    [[2, 3], [2, 3]]   # Speed after hitting speed block (upper three layers)
])

BALL_DIRECTIONS = jnp.array([
    [1, 1],      # Down right
    [-1, 1],     # Down left
    [1, -1],     # Up right
    [-1, -1],    # Up left
])

# Lookup tables for reversing ball direction
REVERSE_X = jnp.array([1, 0, 3, 2])
REVERSE_Y = jnp.array([2, 3, 0, 1])



class EntityPosition(NamedTuple):
    x: chex.Array
    y: chex.Array
    width: chex.Array
    height: chex.Array

class BreakoutObservation(NamedTuple):
    player: EntityPosition
    ball: EntityPosition
    blocks: chex.Array
    score: chex.Array
    lives: chex.Array

class BreakoutInfo(NamedTuple):
    time: chex.Array
    wall_resets: chex.Array


# Game state container
class State(NamedTuple):
    player_x: chex.Array
    player_speed: chex.Array
    small_paddle: chex.Array
    ball_x: chex.Array
    ball_y: chex.Array
    ball_vel_x: chex.Array
    ball_vel_y: chex.Array
    ball_speed_idx: chex.Array
    ball_direction_idx: chex.Array
    consecutive_paddle_hits: chex.Array
    blocks: chex.Array
    score: chex.Array
    lives: chex.Array
    step_counter: chex.Array
    acceleration_counter: chex.Array
    game_started: chex.Array
    blocks_hittable: chex.Array
    wall_resets: chex.Array
    all_blocks_cleared: chex.Array


# Actions
NOOP = 0
FIRE = 1
RIGHT = 2
LEFT = 3

def get_human_action() -> chex.Array:
    """Records keyboard input and returns the corresponding action."""
    keys = pygame.key.get_pressed()
    if keys[pygame.K_a]:
        return jnp.array(LEFT)
    elif keys[pygame.K_d]:
        return jnp.array(RIGHT)
    elif keys[pygame.K_SPACE]:
        return jnp.array(FIRE)
    else:
        return jnp.array(NOOP)

@jax.jit
def player_step(
    state_player_x: chex.Array,
    state_player_speed: chex.Array,
    acceleration_counter: chex.Array,
    action: chex.Array,
) -> (chex.Array, chex.Array, chex.Array):
    """Updates the player position based on the action."""
    left = action == LEFT
    right = action == RIGHT

    # Check if the paddle is touching the left or right wall.
    touches_wall = jnp.logical_or(
        state_player_x <= PLAYER_X_MIN, state_player_x >= PLAYER_X_MAX
    )

    # Get the acceleration schedule based on whether the paddle is at a wall.
    # If touching a wall, use PLAYER_WALL_ACCELERATION, otherwise use PLAYER_ACCELERATION.
    acceleration = jax.lax.cond(
        touches_wall,
        lambda _: PLAYER_WALL_ACCELERATION[acceleration_counter],
        lambda _: PLAYER_ACCELERATION[acceleration_counter],
        operand=None,
    )

    # Apply deceleration if no button is pressed or if the paddle touches a wall.
    player_speed = jax.lax.cond(
        jnp.logical_or(jnp.logical_not(jnp.logical_or(left, right)), touches_wall),
        lambda s: jnp.round(s / 2).astype(jnp.int32),
        lambda s: s,
        operand=state_player_speed,
    )

    # If the paddle is moving to the right but the player pressed left, reset speed.
    direction_change_left = jnp.logical_and(left, state_player_speed > 0)
    player_speed = jax.lax.cond(
        direction_change_left,
        lambda s: 0,
        lambda s: s,
        operand=player_speed,
    )

    # Likewise, if moving left but the player pressed right, reset speed.
    direction_change_right = jnp.logical_and(right, state_player_speed < 0)
    player_speed = jax.lax.cond(
        direction_change_right,
        lambda s: 0,
        lambda s: s,
        operand=player_speed,
    )

    # If a direction change occurred, reset the acceleration counter.
    direction_change = jnp.logical_or(direction_change_left, direction_change_right)
    acceleration_counter = jax.lax.cond(
        direction_change,
        lambda _: 0,
        lambda s: s,
        operand=acceleration_counter,
    )

    # Apply acceleration for the pressed direction:
    # For left, we subtract the acceleration (making speed more negative), clamped to -MAX_SPEED.
    player_speed = jax.lax.cond(
        left,
        lambda s: jnp.maximum(s - acceleration, -PLAYER_MAX_SPEED),
        lambda s: s,
        operand=player_speed,
    )

    # For right, we add the acceleration (making speed more positive), clamped to MAX_SPEED.
    player_speed = jax.lax.cond(
        right,
        lambda s: jnp.minimum(s + acceleration, PLAYER_MAX_SPEED),
        lambda s: s,
        operand=player_speed,
    )

    # Update the acceleration counter: increment if a directional button is held, otherwise reset.
    new_acceleration_counter = jax.lax.cond(
        jnp.logical_or(left, right),
        lambda s: jnp.minimum(s + 1, PLAYER_ACCELERATION.size - 1),
        lambda s: 0,
        operand=acceleration_counter,
    )

    # Update the paddle's horizontal position and clamp it within the game boundaries.
    player_x = jnp.clip(state_player_x + player_speed, PLAYER_X_MIN, PLAYER_X_MAX)

    return player_x, player_speed, new_acceleration_counter

@jax.jit
def get_ball_velocity(speed_idx, direction_idx, step_counter):
    """Returns the ball's velocity based on the speed and direction indices."""
    sub_idx = step_counter % 2
    abs_speed = BALL_VELOCITIES_ABS[speed_idx, sub_idx, :]
    direction = BALL_DIRECTIONS[direction_idx]
    return abs_speed[0] * direction[0], abs_speed[1] * direction[1]

@jax.jit
def detect_paddle_hit(ball_x, ball_y, player_x, small_paddle):
    """Detects if the ball has hit the paddle."""
    paddle_width = jnp.where(small_paddle, PLAYER_SIZE_SMALL[0], PLAYER_SIZE[0])

    paddle_hit_y = jnp.logical_and(
        ball_y + BALL_SIZE[1] > PLAYER_START_Y,
        ball_y + BALL_SIZE[1] <= PLAYER_START_Y + PLAYER_SIZE[1]
    )

    paddle_hit_x = jnp.logical_and(
        ball_x + BALL_SIZE[0] >= player_x,
        ball_x <= player_x + paddle_width
    )

    return jnp.logical_and(paddle_hit_x, paddle_hit_y)

@jax.jit
def ball_step(state, game_started, player_x):
    """Updates the ball's position, handles wall collisions, and paddle bounces."""
    # Compute spawn index and spawn position
    idx = state.step_counter % 4
    ball_start_x = BALL_START_X[idx]
    ball_start_y = BALL_START_Y

    def not_started_fn(_):
        # Before the game starts, use the spawn position and spawn velocity without
        # collision handling.
        new_direction_idx = jnp.where(jnp.logical_or(idx == 0, idx == 2), 0, 1)
        new_speed_idx = 0
        vel_x, vel_y = get_ball_velocity(new_speed_idx, new_direction_idx, state.step_counter)
        return (ball_start_x, ball_start_y, vel_x, vel_y, new_speed_idx, new_direction_idx,
                jnp.array(0), jnp.array(True), jnp.array(False))

    def started_fn(_):
        # Once the game has started, update ball position normally
        ball_x = state.ball_x + state.ball_vel_x
        ball_y = state.ball_y + state.ball_vel_y

        ball_speed_idx = state.ball_speed_idx
        ball_direction_idx = state.ball_direction_idx

        hit_wall_or_paddle = False

        # Collision handling with side walls
        left_collision = ball_x <= WALL_SIDE_WIDTH
        right_collision = ball_x >= 160 - WALL_SIDE_WIDTH - BALL_SIZE[0]
        ball_x = jnp.where(left_collision, WALL_SIDE_WIDTH, ball_x)
        ball_x = jnp.where(right_collision, 160 - WALL_SIDE_WIDTH - BALL_SIZE[0], ball_x)
        ball_direction_idx = jnp.where(
            jnp.logical_or(left_collision, right_collision),
            REVERSE_X[ball_direction_idx],
            ball_direction_idx
        )

        hit_side_wall = jnp.logical_or(left_collision, right_collision)
        ball_in_block_area = jnp.logical_and(
            ball_y >= BLOCK_START_Y,
            ball_y <= BLOCK_START_Y + NUM_ROWS * BLOCK_SIZE[1]
        )
        hit_wall_or_paddle = jnp.logical_or(hit_wall_or_paddle, jnp.logical_and(hit_side_wall, jnp.logical_not(ball_in_block_area)))

        # Collision handling with top wall
        top_collision = ball_y <= WALL_TOP_Y + WALL_TOP_HEIGHT
        ball_y = jnp.where(top_collision, WALL_TOP_Y + WALL_TOP_HEIGHT, ball_y)
        ball_direction_idx = jnp.where(
            top_collision,
            REVERSE_Y[ball_direction_idx],
            ball_direction_idx
        )

        hit_wall_or_paddle = jnp.logical_or(hit_wall_or_paddle, top_collision)

        # Set small_paddle to True when ball hits top wall
        small_paddle = jnp.logical_or(state.small_paddle, top_collision)

        # Detect a paddle hit
        paddle_hit = detect_paddle_hit(ball_x, ball_y, player_x, small_paddle)

        hit_wall_or_paddle = jnp.logical_or(hit_wall_or_paddle, paddle_hit)

        section_width = PLAYER_SIZE[0] / 5 # Divide section into 5 equal parts
        hit_section = jnp.where(
            paddle_hit,
            jnp.floor((ball_x - player_x) / section_width).astype(jnp.int32),
            -1
        )

        # hit_section 0, left
        ball_speed_idx = jax.lax.select(
            jnp.logical_and(paddle_hit, jnp.logical_and(hit_section == 0, ball_speed_idx != 4)),
            1,
            ball_speed_idx
        )
        ball_direction_idx = jax.lax.select(
            jnp.logical_and(paddle_hit, hit_section == 0),
            jnp.array(3, dtype=ball_direction_idx.dtype),
            ball_direction_idx
        )
        # hit_section 1, middle-left
        ball_speed_idx = jax.lax.select(
            jnp.logical_and(paddle_hit, jnp.logical_and(hit_section == 1, ball_speed_idx != 4)),
            2,
            ball_speed_idx
        )
        ball_direction_idx = jax.lax.select(
            jnp.logical_and(paddle_hit, hit_section == 1),
            jnp.array(3, dtype=ball_direction_idx.dtype),
            ball_direction_idx
        )
        # hit_section 2, middle (x direction maintains)
        new_dir_for_hit2 = jnp.where(state.ball_vel_x > 0, 2, 3)
        ball_speed_idx = jax.lax.select(
            jnp.logical_and(paddle_hit, jnp.logical_and(hit_section == 2, ball_speed_idx != 4)),
            2,
            ball_speed_idx
        )
        ball_direction_idx = jax.lax.select(
            jnp.logical_and(paddle_hit, hit_section == 2),
            new_dir_for_hit2,
            ball_direction_idx
        )
        # hit_section 3, middle-right
        ball_speed_idx = jax.lax.select(
            jnp.logical_and(paddle_hit, jnp.logical_and(hit_section == 3, ball_speed_idx != 4)),
            2,
            ball_speed_idx
        )
        ball_direction_idx = jax.lax.select(
            jnp.logical_and(paddle_hit, hit_section == 3),
            jnp.array(2, dtype=ball_direction_idx.dtype),
            ball_direction_idx
        )
        # hit_section 4, right
        ball_speed_idx = jax.lax.select(
            jnp.logical_and(paddle_hit, jnp.logical_and(hit_section == 4, ball_speed_idx != 4)),
            1,
            ball_speed_idx
        )
        ball_direction_idx = jax.lax.select(
            jnp.logical_and(paddle_hit, hit_section == 4),
            jnp.array(2, dtype=ball_direction_idx.dtype),
            ball_direction_idx
        )

        new_vel_x, new_vel_y = get_ball_velocity(ball_speed_idx, ball_direction_idx, state.step_counter)

        paddle_hit_for_hit_counter = jnp.logical_and(paddle_hit, jnp.logical_and(state.ball_vel_y > 0, new_vel_y < 0))

        # update consecutive paddle hits counter
        new_consecutive_hits = jax.lax.cond(
            jnp.logical_and(paddle_hit_for_hit_counter, state.ball_vel_y > 0),
            lambda _: state.consecutive_paddle_hits + 1,
            lambda _: state.consecutive_paddle_hits,
            operand=None
        )
        # if 12th consecutive hits, increase ball speed
        ball_speed_idx = jax.lax.cond(
            jnp.logical_and(new_consecutive_hits >= 12, ball_speed_idx != 4),
            lambda _: jnp.array(3, dtype=ball_speed_idx.dtype),
            lambda _: ball_speed_idx,
            operand=None
        )

        # Update ball speed
        new_vel_x, new_vel_y = get_ball_velocity(ball_speed_idx, ball_direction_idx, state.step_counter)

        # Reset blocks_hittable if ball hit wall or paddle
        blocks_hittable = jnp.where(hit_wall_or_paddle, True, state.blocks_hittable)

        return (ball_x, ball_y, new_vel_x, new_vel_y, ball_speed_idx, ball_direction_idx,
                new_consecutive_hits, blocks_hittable, small_paddle)

    # Use a conditional: if game_started is true, run the normal update branch;
    # otherwise, use the spawn values.
    (ball_x, ball_y, ball_vel_x, ball_vel_y, ball_speed_idx, ball_direction_idx,
     new_consecutive_hits, blocks_hittable, small_paddle) = jax.lax.cond(
        game_started, started_fn, not_started_fn, operand=None
    )

    return (ball_x, ball_y, ball_vel_x, ball_vel_y, ball_speed_idx, ball_direction_idx,
            new_consecutive_hits, blocks_hittable, small_paddle)


@jax.jit
def check_block_collision(state, ball_x, ball_y, ball_speed_idx, ball_direction_idx, consecutive_hits):
    """Checks for block collisions and updates the state using vectorized operations."""

    # Get state variables
    blocks = state.blocks
    score = state.score
    blocks_hittable = state.blocks_hittable

    # Define function to handle case when blocks are not hittable
    def no_hit_path(_):
        new_vel_x, new_vel_y = get_ball_velocity(ball_speed_idx, ball_direction_idx, state.step_counter)
        all_blocks_cleared = jnp.all(blocks == 0)
        return (blocks, score, ball_x, ball_y, new_vel_x, new_vel_y,
                ball_speed_idx, ball_direction_idx, consecutive_hits, blocks_hittable, all_blocks_cleared)

    # Define function to handle case when blocks are hittable
    def hit_path(_):
        # Create a grid of all possible block positions
        row_indices, col_indices = jnp.mgrid[:NUM_ROWS, :BLOCKS_PER_ROW]

        # Calculate positions for all blocks at once
        block_xs = BLOCK_START_X + col_indices * BLOCK_SIZE[0]
        block_ys = BLOCK_START_Y + row_indices * BLOCK_SIZE[1]

        # Only consider active blocks
        active_blocks = blocks == 1

        # Check horizontal collision for all blocks
        x_collision = jnp.logical_and(
            ball_x < block_xs + BLOCK_SIZE[0],
            ball_x + BALL_SIZE[0] > block_xs
        )

        # Check vertical collision for all blocks
        y_collision = jnp.logical_and(
            ball_y <= block_ys + BLOCK_SIZE[1],
            ball_y + BALL_SIZE[1] >= block_ys
        )

        # Overall collision mask
        collision_mask = jnp.logical_and(
            active_blocks,
            jnp.logical_and(x_collision, y_collision)
        )

        # Check for bottom edge collision (for bounce)
        bounce_condition = jnp.abs((block_ys + BLOCK_SIZE[1]) - ball_y) <= 4
        bounce_mask = jnp.logical_and(collision_mask, bounce_condition)

        # Check if any block was hit
        any_hit = jnp.any(bounce_mask)

        # Find the hit with highest priority (process blocks top-to-bottom, left-to-right)
        # Use a single priority value: row * BLOCKS_PER_ROW + col
        priority_grid = row_indices * BLOCKS_PER_ROW + col_indices

        # Mask priorities - only consider blocks with bounce
        masked_priorities = jnp.where(
            bounce_mask,
            priority_grid,
            jnp.full_like(priority_grid, NUM_ROWS * BLOCKS_PER_ROW + 1)  # High value for non-hits
        )

        # Find the block with minimum priority value (highest priority)
        flat_priorities = masked_priorities.reshape(-1)
        hit_idx = jnp.argmin(flat_priorities)
        hit_priority = flat_priorities[hit_idx]

        # Extract row and column from priority
        hit_row = hit_priority // BLOCKS_PER_ROW
        hit_col = hit_priority % BLOCKS_PER_ROW

        # Check if we have a valid hit (within grid bounds)
        valid_hit = hit_priority < (NUM_ROWS * BLOCKS_PER_ROW)

        # Determine points based on row
        points = jnp.where(
            hit_row >= 4,
            1,  # Bottom rows
            jnp.where(
                hit_row >= 2,
                4,  # Middle rows
                7  # Top rows
            )
        )

        # Update blocks - clear the hit block
        updated_blocks = jnp.where(
            jnp.logical_and(any_hit, valid_hit),
            blocks.at[hit_row, hit_col].set(0),
            blocks
        )

        # Update score
        updated_score = jnp.where(
            jnp.logical_and(any_hit, valid_hit),
            score + points,
            score
        )

        # Update ball direction for bounce
        updated_direction_idx = jnp.where(
            jnp.logical_and(any_hit, valid_hit),
            REVERSE_Y[ball_direction_idx],
            ball_direction_idx
        )

        # Check if we need to accelerate the ball (upper three rows)
        accelerate = jnp.logical_and(
            jnp.logical_and(any_hit, valid_hit),
            hit_row < 3
        )

        # Update ball speed
        updated_speed_idx = jnp.where(
            jnp.logical_and(accelerate, ball_speed_idx != 4),
            jnp.array(4, dtype=ball_speed_idx.dtype),
            ball_speed_idx
        )

        # Calculate new velocities
        new_vel_x, new_vel_y = get_ball_velocity(updated_speed_idx, updated_direction_idx, state.step_counter)

        # Update blocks_hittable
        updated_blocks_hittable = jnp.where(
            jnp.logical_and(any_hit, valid_hit),
            False,
            blocks_hittable
        )

        # Check if all blocks are cleared
        all_blocks_cleared = jnp.all(updated_blocks == 0)

        return (updated_blocks, updated_score, ball_x, ball_y, new_vel_x, new_vel_y,
                updated_speed_idx, updated_direction_idx, consecutive_hits,
                updated_blocks_hittable, all_blocks_cleared)

    # Execute the appropriate path based on blocks_hittable
    return jax.lax.cond(
        blocks_hittable,
        hit_path,
        no_hit_path,
        operand=None
    )

class JaxBreakout(JaxEnvironment[State, BreakoutObservation, BreakoutInfo]):
    def __init__(self):
        super().__init__()

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key = None) -> tuple[BreakoutObservation, State]:
        """Initialize game state"""
        init_speed_idx = 0
        init_direction_idx = 0
        init_vel_x, init_vel_y = get_ball_velocity(init_speed_idx, init_direction_idx, 0)

        state =  State(
            player_x=jnp.array(PLAYER_START_X),
            player_speed=jnp.array(0),
            small_paddle=jnp.array(False),
            ball_x=jnp.array(BALL_START_X[0]),
            ball_y=jnp.array(BALL_START_Y),
            ball_vel_x=init_vel_x,
            ball_vel_y=init_vel_y,
            ball_speed_idx=jnp.array(init_speed_idx),
            ball_direction_idx=jnp.array(init_direction_idx),
            consecutive_paddle_hits=jnp.array(0),
            blocks_hittable=jnp.array(True),
            blocks=jnp.ones((NUM_ROWS, BLOCKS_PER_ROW), dtype=jnp.int32),
            score=jnp.array(0),
            lives=jnp.array(NUM_LIVES),
            step_counter=jnp.array(0),
            acceleration_counter=jnp.array(0).astype(jnp.int32),
            game_started=jnp.array(0),
            wall_resets=jnp.array(0),
            all_blocks_cleared=jnp.array(False),
        )

        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: State, action: chex.Array) -> Tuple[
        BreakoutObservation, State, chex.Array, chex.Array, BreakoutInfo]:
        prev_score = state.score
        new_state = self._step(state, action)
        obs = self._get_observation(new_state)
        reward = self._get_reward(prev_score, new_state.score)
        done = self._get_done(new_state)
        info = self._get_info(new_state)
        return obs, new_state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _step(self, state: State, action: chex.Array) -> State:
        # Update player position
        new_player_x, new_paddle_v, new_acceleration_counter = player_step(
            state.player_x, state.player_speed, state.acceleration_counter, action
        )

        game_started = jnp.logical_or(state.game_started, action == FIRE)

        # Update ball, check collisions, etc., as before, but now pass new_player_x
        (ball_x, ball_y, ball_vel_x, ball_vel_y, ball_speed_idx, ball_direction_idx,
         consecutive_hits, blocks_hittable, small_paddle) = ball_step(
            state, game_started, new_player_x
        )

        # Detect paddle hit (for resetting the wall)
        paddle_hit = detect_paddle_hit(ball_x, ball_y, new_player_x, small_paddle)

        # Check for block collisions
        (new_blocks, new_score, ball_x, ball_y, ball_vel_x, ball_vel_y, ball_speed_idx,
         ball_direction_idx, consecutive_hits, blocks_hittable, all_blocks_cleared) = check_block_collision(
            state._replace(blocks_hittable=blocks_hittable), ball_x, ball_y, ball_speed_idx, ball_direction_idx, consecutive_hits
        )

        # Reset wall if paddle hit occurs after all blocks were cleared and we haven't reset the wall already
        should_reset_wall = jnp.logical_and(
            jnp.logical_and(state.all_blocks_cleared, paddle_hit),
            state.wall_resets < 1
        )
        new_blocks = jnp.where(should_reset_wall, jnp.ones_like(new_blocks), new_blocks)
        new_wall_resets = jnp.where(should_reset_wall, state.wall_resets + 1, state.wall_resets)

        # Update all_blocks_cleared status
        new_all_blocks_cleared = jnp.logical_and(
            jnp.logical_or(state.all_blocks_cleared, all_blocks_cleared),
            jnp.logical_not(should_reset_wall)  # Reset to False if we just reset the wall
        )

        # Handle life loss, etc.
        life_lost = ball_y >= WINDOW_HEIGHT // 3
        ball_x = jnp.where(life_lost, new_player_x + 7, ball_x)
        ball_y = jnp.where(life_lost, BALL_START_Y, ball_y)
        ball_speed_idx = jnp.where(life_lost, 0, ball_speed_idx)
        ball_direction_idx = jnp.where(life_lost, 0, ball_direction_idx)
        ball_vel_x, ball_vel_y = get_ball_velocity(ball_speed_idx, ball_direction_idx, state.step_counter)
        consecutive_hits = jnp.where(life_lost, 0, consecutive_hits)
        blocks_hittable = jnp.where(life_lost, True, blocks_hittable)
        game_started = jnp.where(life_lost, jnp.array(0), game_started)
        new_lives = jnp.where(life_lost, state.lives - 1, state.lives)
        small_paddle = jnp.where(life_lost, jnp.array(False), small_paddle)

        return State(
            player_x=new_player_x,
            player_speed=new_paddle_v,
            small_paddle=small_paddle,
            ball_x=ball_x,
            ball_y=ball_y,
            ball_vel_x=ball_vel_x,
            ball_vel_y=ball_vel_y,
            ball_speed_idx=ball_speed_idx,
            ball_direction_idx=ball_direction_idx,
            consecutive_paddle_hits=consecutive_hits,
            blocks_hittable=blocks_hittable,
            blocks=new_blocks,
            score=new_score,
            lives=new_lives,
            step_counter=state.step_counter + 1,
            acceleration_counter=new_acceleration_counter,
            game_started=game_started,
            wall_resets=new_wall_resets,
            all_blocks_cleared=new_all_blocks_cleared,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: State) -> BreakoutObservation:
        paddle_width = jnp.where(state.small_paddle, PLAYER_SIZE_SMALL[0], PLAYER_SIZE[0])

        player = EntityPosition(
            x=state.player_x,
            y=jnp.array(PLAYER_START_Y),
            width=paddle_width,
            height=jnp.array(PLAYER_SIZE[1]),
        )

        ball = EntityPosition(
            x=state.ball_x,
            y=state.ball_y,
            width=jnp.array(BALL_SIZE[0]),
            height=jnp.array(BALL_SIZE[1]),
        )

        return BreakoutObservation(
            player=player,
            ball=ball,
            blocks=state.blocks,
            score=state.score,
            lives=state.lives,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: State) -> BreakoutInfo:
        return BreakoutInfo(
            time=state.step_counter,
            wall_resets=state.wall_resets
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_score: chex.Array, current_score: chex.Array) -> chex.Array:
        return current_score - previous_score

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: State) -> chex.Array:
        return state.lives <= 0


class Renderer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Breakout")
        self.clock = pygame.time.Clock()

    def render(self, state: State):
        # Clear screen
        self.screen.fill(BACKGROUND_COLOR)

        # Draw walls
        # Top wall
        pygame.draw.rect(
            self.screen,
            WALL_COLOR,
            (0, WALL_TOP_Y * 3, WINDOW_WIDTH, WALL_TOP_HEIGHT * 3),
        )

        # Left wall
        pygame.draw.rect(
            self.screen,
            WALL_COLOR,
            (0, WALL_TOP_Y * 3, WALL_SIDE_WIDTH * 3, (196 - 17) * 3),
        )

        # Right wall
        pygame.draw.rect(
            self.screen,
            WALL_COLOR,
            (
                WINDOW_WIDTH - WALL_SIDE_WIDTH * 3,
                WALL_TOP_Y * 3,
                WALL_SIDE_WIDTH * 3,
                (196 - 18) * 3,
            ),
        )

        # Draw blocks
        for row in range(NUM_ROWS):
            for col in range(BLOCKS_PER_ROW):
                if state.blocks[row, col] == 1:
                    block_rect = pygame.Rect(
                        (BLOCK_START_X + col * BLOCK_SIZE[0]) * 3,
                        (BLOCK_START_Y + row * BLOCK_SIZE[1]) * 3,
                        BLOCK_SIZE[0] * 3,
                        BLOCK_SIZE[1] * 3,
                    )
                    pygame.draw.rect(self.screen, BLOCK_COLORS[row], block_rect)

        # Draw player paddle
        paddle_width = PLAYER_SIZE_SMALL[0] if state.small_paddle else PLAYER_SIZE[0]
        player_rect = pygame.Rect(
            int(state.player_x) * 3,
            PLAYER_START_Y * 3,
            paddle_width * 3,
            PLAYER_SIZE[1] * 3,
        )
        pygame.draw.rect(self.screen, PLAYER_COLOR, player_rect)

        # Draw ball only if the game has started
        if state.game_started and state.ball_y < 197:
            ball_rect = pygame.Rect(
                int(state.ball_x) * 3,
                int(state.ball_y) * 3,
                BALL_SIZE[0] * 3,
                BALL_SIZE[1] * 3,
            )
            pygame.draw.rect(self.screen, BALL_COLOR, ball_rect)

        # Draw score and lives
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {state.score}", True, (255, 255, 255))
        lives_text = font.render(f"Lives: {state.lives}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(lives_text, (WINDOW_WIDTH - 100, 10))

        pygame.display.flip()


if __name__ == "__main__":
    # Initialize game and renderer
    game = JaxBreakout()
    renderer = Renderer()

    # Get jitted functions
    jitted_step = jax.jit(game.step)
    jitted_reset = jax.jit(game.reset)

    obs, curr_state = jitted_reset()

    # Game loop
    running = True
    frameskip = 1
    frame_by_frame = False
    counter = 1

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
                        if done:
                            running = False

        if not frame_by_frame:
            if counter % frameskip == 0:
                action = get_human_action()
                obs, curr_state, reward, done, info = jitted_step(curr_state, action)
                if done:
                    running = False

        renderer.render(curr_state)
        counter += 1
        renderer.clock.tick(60)

    pygame.quit()
