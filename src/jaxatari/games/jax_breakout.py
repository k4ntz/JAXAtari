from functools import partial
import os
from typing import NamedTuple, Tuple
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
import chex
import pygame

from jaxatari.environment import JaxEnvironment, ObjectObservation, JAXAtariAction as Action
import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
import jaxatari.rendering.jax_rendering_utils as render_utils

class BreakoutConstants(struct.PyTreeNode):
    WINDOW_WIDTH: int = struct.field(pytree_node=False, default=160)
    WINDOW_HEIGHT: int = struct.field(pytree_node=False, default=210)
    BACKGROUND_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default_factory=lambda: (0, 0, 0))
    PLAYER_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default_factory=lambda: (200, 72, 72))
    BALL_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default_factory=lambda: (200, 72, 72))
    WALL_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default_factory=lambda: (142, 142, 142))
    BLOCK_COLORS: Tuple[Tuple[int, int, int], ...] = struct.field(pytree_node=False, default_factory=lambda: tuple([
        (200, 72, 72),
        (198, 108, 58),
        (180, 122, 48), (162, 162, 42), (72, 160, 72), (66, 72, 200),
    ]))
    PLAYER_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: (16, 4))
    PLAYER_SIZE_SMALL: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: (12, 4))
    BALL_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: (2, 4))
    BLOCK_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: (8, 6))
    WALL_TOP_Y: int = struct.field(pytree_node=False, default=17)
    WALL_TOP_HEIGHT: int = struct.field(pytree_node=False, default=15)
    WALL_SIDE_WIDTH: int = struct.field(pytree_node=False, default=8)
    PLAYER_START_X: int = struct.field(pytree_node=False, default=99)
    PLAYER_START_Y: int = struct.field(pytree_node=False, default=189)
    BALL_START_X: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([16, 78, 80, 142]))
    BALL_START_Y: int = struct.field(pytree_node=False, default=122)
    PLAYER_X_MIN: int = struct.field(pytree_node=False, default=8)
    # PLAYER_X_MAX is calculated dynamically based on paddle width to support mods
    # It will be computed as WINDOW_WIDTH - WALL_SIDE_WIDTH - max(PLAYER_SIZE[0], PLAYER_SIZE_SMALL[0])
    PLAYER_MAX_SPEED: int = struct.field(pytree_node=False, default=6)
    PLAYER_ACCELERATION: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([3, 2, -1, 1, 1]))
    PLAYER_WALL_ACCELERATION: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([1, 2, 1, 1, 1]))
    BLOCKS_PER_ROW: int = struct.field(pytree_node=False, default=18)
    NUM_ROWS: int = struct.field(pytree_node=False, default=6)
    BLOCK_START_Y: int = struct.field(pytree_node=False, default=57)
    BLOCK_START_X: int = struct.field(pytree_node=False, default=8)
    NUM_LIVES: int = struct.field(pytree_node=False, default=5)
    BALL_VELOCITIES_ABS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([
        [[1, 1], [1, 1]],
        [[2, 1], [1, 1]],
        [[1, 2], [1, 1]],
        [[2, 2], [2, 2]],
        [[2, 3], [2, 3]]
    ]))
    BALL_DIRECTIONS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([
        [1, 1],
        [-1, 1],
        [1, -1],
        [-1, -1],
    ]))
    REVERSE_X: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([1, 0, 3, 2]))
    REVERSE_Y: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([2, 3, 0, 1]))

@struct.dataclass
class BreakoutObservation:
    player: ObjectObservation
    ball: ObjectObservation
    blocks: jnp.ndarray
    lives: jnp.ndarray
    score: jnp.ndarray

@struct.dataclass
class BreakoutInfo:
    time: chex.Array
    wall_resets: chex.Array


# Game state container
@struct.dataclass
class BreakoutState:
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

class JaxBreakout(JaxEnvironment[BreakoutState, BreakoutObservation, BreakoutInfo, BreakoutConstants]):
    # Minimal ALE action set for Breakout: 0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT
    ACTION_SET: jnp.ndarray = jnp.array(
        [Action.NOOP, Action.FIRE, Action.RIGHT, Action.LEFT],
        dtype=jnp.int32,
    )
    
    def __init__(self, consts: BreakoutConstants = None):
        consts = consts or BreakoutConstants()
        super().__init__(consts)
        self.renderer = BreakoutRenderer(self.consts)
    
    def _get_player_x_max(self) -> int:
        """Calculate the maximum X position for the player paddle.
        This ensures the paddle doesn't extend beyond the right wall, accounting for variable paddle sizes."""
        max_paddle_width = max(self.consts.PLAYER_SIZE[0], self.consts.PLAYER_SIZE_SMALL[0])
        return self.consts.WINDOW_WIDTH - self.consts.WALL_SIDE_WIDTH - max_paddle_width 

    def get_human_action(self) -> chex.Array:
        """Records keyboard input and returns the corresponding action."""
        keys = pygame.key.get_pressed()
        if keys[pygame.K_a]:
            return jnp.array(Action.LEFT)
        elif keys[pygame.K_d]:
            return jnp.array(Action.RIGHT)
        elif keys[pygame.K_SPACE]:
            return jnp.array(Action.FIRE)
        else:
            return jnp.array(Action.NOOP)

    @partial(jax.jit, static_argnums=(0,))
    def _player_step(
        self,
        state_player_x: chex.Array,
        state_player_speed: chex.Array,
        acceleration_counter: chex.Array,
        atari_action: chex.Array,
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Updates the player position based on the action."""
        left = atari_action == Action.LEFT
        right = atari_action == Action.RIGHT

        # Calculate the maximum X position based on paddle width
        # Use the maximum paddle width to ensure it works for both normal and small paddles
        max_paddle_width = max(self.consts.PLAYER_SIZE[0], self.consts.PLAYER_SIZE_SMALL[0])
        player_x_max = self.consts.WINDOW_WIDTH - self.consts.WALL_SIDE_WIDTH - max_paddle_width
        
        # Check if the paddle is touching the left or right wall.
        touches_wall = jnp.logical_or(
            state_player_x <= self.consts.PLAYER_X_MIN, state_player_x >= player_x_max
        )

        # Get the acceleration schedule based on whether the paddle is at a wall.
        # If touching a wall, use PLAYER_WALL_ACCELERATION, otherwise use PLAYER_ACCELERATION.
        acceleration = jax.lax.cond(
            touches_wall,
            lambda _: self.consts.PLAYER_WALL_ACCELERATION[acceleration_counter],
            lambda _: self.consts.PLAYER_ACCELERATION[acceleration_counter],
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
            lambda s: jnp.maximum(s - acceleration, -self.consts.PLAYER_MAX_SPEED),
            lambda s: s,
            operand=player_speed,
        )

        # For right, we add the acceleration (making speed more positive), clamped to MAX_SPEED.
        player_speed = jax.lax.cond(
            right,
            lambda s: jnp.minimum(s + acceleration, self.consts.PLAYER_MAX_SPEED),
            lambda s: s,
            operand=player_speed,
        )

        # Update the acceleration counter: increment if a directional button is held, otherwise reset.
        new_acceleration_counter = jax.lax.cond(
            jnp.logical_or(left, right),
            lambda s: jnp.minimum(s + 1, self.consts.PLAYER_ACCELERATION.size - 1),
            lambda s: 0,
            operand=acceleration_counter,
        )

        # Update the paddle's horizontal position and clamp it within the game boundaries.
        # player_x_max was already calculated at the beginning of this function
        player_x = jnp.clip(state_player_x + player_speed, self.consts.PLAYER_X_MIN, player_x_max)

        return player_x, player_speed, new_acceleration_counter

    @partial(jax.jit, static_argnums=(0,))
    def _get_ball_velocity(self, speed_idx, direction_idx, step_counter):
        """Returns the ball's velocity based on the speed and direction indices."""
        sub_idx = step_counter % 2
        abs_speed = self.consts.BALL_VELOCITIES_ABS[speed_idx, sub_idx, :]
        direction = self.consts.BALL_DIRECTIONS[direction_idx]
        return abs_speed[0] * direction[0], abs_speed[1] * direction[1]

    @partial(jax.jit, static_argnums=(0,))
    def _detect_paddle_hit(self, ball_x, ball_y, player_x, small_paddle):
        """Detects if the ball has hit the paddle."""
        paddle_width = jnp.where(small_paddle, self.consts.PLAYER_SIZE_SMALL[0], self.consts.PLAYER_SIZE[0])

        # Check for collision from above (ball hitting paddle top)
        paddle_hit_y_from_above = jnp.logical_and(
            ball_y + self.consts.BALL_SIZE[1] > self.consts.PLAYER_START_Y,
            ball_y + self.consts.BALL_SIZE[1] <= self.consts.PLAYER_START_Y + self.consts.PLAYER_SIZE[1]
        )

        # Check for collision from sides (ball hitting paddle left or right edge)
        paddle_hit_x_from_left = jnp.logical_and(
            ball_x + self.consts.BALL_SIZE[0] >= player_x,
            ball_x + self.consts.BALL_SIZE[0] <= player_x + 2  # Small tolerance for side hit
        )
        
        paddle_hit_x_from_right = jnp.logical_and(
            ball_x <= player_x + paddle_width,
            ball_x >= player_x + paddle_width - 2  # Small tolerance for side hit
        )

        # X-axis collision check (ball overlaps with paddle horizontally)
        paddle_hit_x_overlap = jnp.logical_and(
            ball_x + self.consts.BALL_SIZE[0] >= player_x,
            ball_x <= player_x + paddle_width
        )

        # Y-axis collision check (ball overlaps with paddle vertically)
        paddle_hit_y_overlap = jnp.logical_and(
            ball_y + self.consts.BALL_SIZE[1] >= self.consts.PLAYER_START_Y,
            ball_y <= self.consts.PLAYER_START_Y + self.consts.PLAYER_SIZE[1]
        )

        # Combine all collision types
        hit_from_above = jnp.logical_and(paddle_hit_x_overlap, paddle_hit_y_from_above)
        hit_from_left = jnp.logical_and(paddle_hit_x_from_left, paddle_hit_y_overlap)
        hit_from_right = jnp.logical_and(paddle_hit_x_from_right, paddle_hit_y_overlap)

        return jnp.logical_or(jnp.logical_or(hit_from_above, hit_from_left), hit_from_right)

    @partial(jax.jit, static_argnums=(0,))
    def _ball_step(self, state, game_started, player_x):
        """Updates the ball's position, handles wall collisions, and paddle bounces."""
        # Compute spawn index and spawn position
        idx = state.step_counter % 4
        ball_start_x = self.consts.BALL_START_X[idx]
        ball_start_y = self.consts.BALL_START_Y

        def not_started_fn(_):
            # Before the game starts, use the spawn position and spawn velocity without
            # collision handling.
            new_direction_idx = jnp.where(jnp.logical_or(idx == 0, idx == 2), 0, 1)
            new_speed_idx = 0
            vel_x, vel_y = self._get_ball_velocity(new_speed_idx, new_direction_idx, state.step_counter)
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
            left_collision = ball_x <= self.consts.WALL_SIDE_WIDTH
            right_collision = ball_x >= 160 - self.consts.WALL_SIDE_WIDTH - self.consts.BALL_SIZE[0]
            ball_x = jnp.where(left_collision, self.consts.WALL_SIDE_WIDTH, ball_x)
            ball_x = jnp.where(right_collision, 160 - self.consts.WALL_SIDE_WIDTH - self.consts.BALL_SIZE[0], ball_x)
            ball_direction_idx = jnp.where(
                jnp.logical_or(left_collision, right_collision),
                self.consts.REVERSE_X[ball_direction_idx],
                ball_direction_idx
            )

            hit_side_wall = jnp.logical_or(left_collision, right_collision)
            ball_in_block_area = jnp.logical_and(
                ball_y >= self.consts.BLOCK_START_Y,
                ball_y <= self.consts.BLOCK_START_Y + self.consts.NUM_ROWS * self.consts.BLOCK_SIZE[1]
            )
            hit_wall_or_paddle = jnp.logical_or(hit_wall_or_paddle, jnp.logical_and(hit_side_wall, jnp.logical_not(ball_in_block_area)))

            # Collision handling with top wall
            top_collision = ball_y <= self.consts.WALL_TOP_Y + self.consts.WALL_TOP_HEIGHT
            ball_y = jnp.where(top_collision, self.consts.WALL_TOP_Y + self.consts.WALL_TOP_HEIGHT, ball_y)
            ball_direction_idx = jnp.where(
                top_collision,
                self.consts.REVERSE_Y[ball_direction_idx],
                ball_direction_idx
            )

            hit_wall_or_paddle = jnp.logical_or(hit_wall_or_paddle, top_collision)

            # Set small_paddle to True when ball hits top wall
            small_paddle = jnp.logical_or(state.small_paddle, top_collision)

            # Detect a paddle hit
            paddle_hit = self._detect_paddle_hit(ball_x, ball_y, player_x, small_paddle)

            hit_wall_or_paddle = jnp.logical_or(hit_wall_or_paddle, paddle_hit)

            section_width = self.consts.PLAYER_SIZE[0] / 5 # Divide section into 5 equal parts
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

            new_vel_x, new_vel_y = self._get_ball_velocity(ball_speed_idx, ball_direction_idx, state.step_counter)

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
            new_vel_x, new_vel_y = self._get_ball_velocity(ball_speed_idx, ball_direction_idx, state.step_counter)

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


    @partial(jax.jit, static_argnums=(0,))
    def _check_block_collision(self, state, ball_x, ball_y, ball_speed_idx, ball_direction_idx, consecutive_hits):
        """Checks for block collisions and updates the state using vectorized operations."""

        # Get state variables
        blocks = state.blocks
        score = state.score
        blocks_hittable = state.blocks_hittable

        # Define function to handle case when blocks are not hittable
        def no_hit_path(_):
            new_vel_x, new_vel_y = self._get_ball_velocity(ball_speed_idx, ball_direction_idx, state.step_counter)
            all_blocks_cleared = jnp.all(blocks == 0)
            return (blocks, score, ball_x, ball_y, new_vel_x, new_vel_y,
                    ball_speed_idx, ball_direction_idx, consecutive_hits, blocks_hittable, all_blocks_cleared)

        # Define function to handle case when blocks are hittable
        def hit_path(_):
            # Create a grid of all possible block positions
            row_indices, col_indices = jnp.mgrid[:self.consts.NUM_ROWS, :self.consts.BLOCKS_PER_ROW]

            # Calculate positions for all blocks at once
            block_xs = self.consts.BLOCK_START_X + col_indices * self.consts.BLOCK_SIZE[0]
            block_ys = self.consts.BLOCK_START_Y + row_indices * self.consts.BLOCK_SIZE[1]

            # Only consider active blocks
            active_blocks = blocks == 1

            # Check horizontal collision for all blocks
            x_collision = jnp.logical_and(
                ball_x < block_xs + self.consts.BLOCK_SIZE[0],
                ball_x + self.consts.BALL_SIZE[0] > block_xs
            )

            # Check vertical collision for all blocks
            y_collision = jnp.logical_and(
                ball_y <= block_ys + self.consts.BLOCK_SIZE[1],
                ball_y + self.consts.BALL_SIZE[1] >= block_ys
            )

            # Overall collision mask
            collision_mask = jnp.logical_and(
                active_blocks,
                jnp.logical_and(x_collision, y_collision)
            )

            # Check for bottom edge collision (for bounce)
            bounce_condition = jnp.abs((block_ys + self.consts.BLOCK_SIZE[1]) - ball_y) <= 4
            bounce_mask = jnp.logical_and(collision_mask, bounce_condition)

            # Check if any block was hit
            any_hit = jnp.any(bounce_mask)

            # Find the hit with highest priority (process blocks top-to-bottom, left-to-right)
            # Use a single priority value: row * BLOCKS_PER_ROW + col
            priority_grid = row_indices * self.consts.BLOCKS_PER_ROW + col_indices

            # Mask priorities - only consider blocks with bounce
            masked_priorities = jnp.where(
                bounce_mask,
                priority_grid,
                jnp.full_like(priority_grid, self.consts.NUM_ROWS * self.consts.BLOCKS_PER_ROW + 1)  # High value for non-hits
            )

            # Find the block with minimum priority value (highest priority)
            flat_priorities = masked_priorities.reshape(-1)
            hit_idx = jnp.argmin(flat_priorities)
            hit_priority = flat_priorities[hit_idx]

            # Extract row and column from priority
            hit_row = hit_priority // self.consts.BLOCKS_PER_ROW
            hit_col = hit_priority % self.consts.BLOCKS_PER_ROW

            # Check if we have a valid hit (within grid bounds)
            valid_hit = hit_priority < (self.consts.NUM_ROWS * self.consts.BLOCKS_PER_ROW)

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
                self.consts.REVERSE_Y[ball_direction_idx],
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
            new_vel_x, new_vel_y = self._get_ball_velocity(updated_speed_idx, updated_direction_idx, state.step_counter)

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

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey = None) -> tuple[BreakoutObservation, BreakoutState]:
        """Initialize game state"""
        init_speed_idx = 0
        init_direction_idx = 0
        init_vel_x, init_vel_y = self._get_ball_velocity(init_speed_idx, init_direction_idx, 0)

        state =  BreakoutState(
            player_x=jnp.array(self.consts.PLAYER_START_X),
            player_speed=jnp.array(0),
            small_paddle=jnp.array(False),
            ball_x=jnp.array(self.consts.BALL_START_X[0]),
            ball_y=jnp.array(self.consts.BALL_START_Y),
            ball_vel_x=init_vel_x,
            ball_vel_y=init_vel_y,
            ball_speed_idx=jnp.array(init_speed_idx),
            ball_direction_idx=jnp.array(init_direction_idx),
            consecutive_paddle_hits=jnp.array(0),
            blocks_hittable=jnp.array(True),
            blocks=jnp.ones((self.consts.NUM_ROWS, self.consts.BLOCKS_PER_ROW), dtype=jnp.int32),
            score=jnp.array(0),
            lives=jnp.array(self.consts.NUM_LIVES),
            step_counter=jnp.array(0),
            acceleration_counter=jnp.array(0).astype(jnp.int32),
            game_started=jnp.array(0),
            wall_resets=jnp.array(0),
            all_blocks_cleared=jnp.array(False),
        )

        return self._get_observation(state), state

    def render(self, state: BreakoutState) -> jnp.ndarray:
        return self.renderer.render(state)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: BreakoutState, action: chex.Array) -> Tuple[
        BreakoutObservation, BreakoutState, chex.Array, chex.Array, BreakoutInfo]:
        new_state = self._step(state, action)
        obs = self._get_observation(new_state)
        env_reward = self._get_reward(state, new_state)
        done = self._get_done(new_state)
        info = self._get_info(new_state)
        return obs, new_state, env_reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _step(self, state: BreakoutState, action: chex.Array) -> BreakoutState:
        # Translate agent action index to ALE console action
        atari_action = jnp.take(self.ACTION_SET, action.astype(jnp.int32))
        # Update player position
        new_player_x, new_paddle_v, new_acceleration_counter = self._player_step(
            state.player_x, state.player_speed, state.acceleration_counter, atari_action
        )
        #TODO: this is a hack -> always fire
        game_started = jnp.logical_or(state.game_started, True)
        # game_started = jnp.logical_or(state.game_started, atari_action == Action.FIRE)

        # Update ball, check collisions, etc., as before, but now pass new_player_x
        (ball_x, ball_y, ball_vel_x, ball_vel_y, ball_speed_idx, ball_direction_idx,
         consecutive_hits, blocks_hittable, small_paddle) = self._ball_step(
            state, game_started, new_player_x
        )

        # Detect paddle hit (for resetting the wall)
        paddle_hit = self._detect_paddle_hit(ball_x, ball_y, new_player_x, small_paddle)

        # Check for block collisions
        (new_blocks, new_score, ball_x, ball_y, ball_vel_x, ball_vel_y, ball_speed_idx,
         ball_direction_idx, consecutive_hits, blocks_hittable, all_blocks_cleared) = self._check_block_collision(
            state.replace(blocks_hittable=blocks_hittable), ball_x, ball_y, ball_speed_idx, ball_direction_idx, consecutive_hits
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
        life_lost = ball_y >= self.consts.WINDOW_HEIGHT
        ball_x = jnp.where(life_lost, new_player_x + 7, ball_x)
        ball_y = jnp.where(life_lost, self.consts.BALL_START_Y, ball_y)
        ball_speed_idx = jnp.where(life_lost, 0, ball_speed_idx)
        ball_direction_idx = jnp.where(life_lost, 0, ball_direction_idx)
        ball_vel_x, ball_vel_y = self._get_ball_velocity(ball_speed_idx, ball_direction_idx, state.step_counter)
        consecutive_hits = jnp.where(life_lost, 0, consecutive_hits)
        blocks_hittable = jnp.where(life_lost, True, blocks_hittable)
        game_started = jnp.where(life_lost, jnp.array(0), game_started)
        new_lives = jnp.where(life_lost, state.lives - 1, state.lives)
        small_paddle = jnp.where(life_lost, jnp.array(False), small_paddle)

        return BreakoutState(
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
    def _get_observation(self, state: BreakoutState) -> BreakoutObservation:
        # --- Player ---
        paddle_width = jnp.where(state.small_paddle, self.consts.PLAYER_SIZE_SMALL[0], self.consts.PLAYER_SIZE[0])
        
        # Clip coordinates to screen bounds 
        p_x = jnp.clip(state.player_x, 0, self.consts.WINDOW_WIDTH)
        p_y = jnp.clip(jnp.array(self.consts.PLAYER_START_Y, dtype=jnp.int32), 0, self.consts.WINDOW_HEIGHT)
        
        player = ObjectObservation.create(
            x=p_x,
            y=p_y,
            width=paddle_width.astype(jnp.int32),
            height=jnp.array(self.consts.PLAYER_SIZE[1], dtype=jnp.int32),
            active=jnp.array(1, dtype=jnp.int32)
        )

        # --- Ball ---
        # Calculate orientation from velocity vector (0-360 degrees)
        ball_orientation = jnp.mod(jnp.degrees(jnp.arctan2(state.ball_vel_y, state.ball_vel_x)), 360.0)
        
        b_x = jnp.clip(state.ball_x, 0, self.consts.WINDOW_WIDTH)
        b_y = jnp.clip(state.ball_y, 0, self.consts.WINDOW_HEIGHT)

        ball = ObjectObservation.create(
            x=b_x,
            y=b_y,
            width=jnp.array(self.consts.BALL_SIZE[0], dtype=jnp.int32),
            height=jnp.array(self.consts.BALL_SIZE[1], dtype=jnp.int32),
            orientation=ball_orientation,
            active=jnp.array(1, dtype=jnp.int32)
        )
        
        # --- Blocks ---
        # Pass the grid directly as a dense array
        blocks = state.blocks.astype(jnp.int32)

        return BreakoutObservation(
            player=player,
            ball=ball,
            blocks=blocks,
            lives=state.lives,
            score=state.score
        )

    
    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: BreakoutState) -> BreakoutInfo:
        return BreakoutInfo(
            time=state.step_counter,
            wall_resets=state.wall_resets,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: BreakoutState, current_state: BreakoutState) -> chex.Array:
        return current_state.score - previous_state.score

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: BreakoutState) -> chex.Array:
        return jnp.logical_or(state.lives <= 0, state.all_blocks_cleared)

    def action_space(self) -> spaces.Discrete:
        """Returns the action space for Breakout.
        Actions are:
        0: NOOP
        1: FIRE
        2: RIGHT
        3: LEFT
        """
        return spaces.Discrete(len(self.ACTION_SET))

    def observation_space(self) -> spaces.Dict:
        """Returns the observation space for Breakout.
        The observation contains:
        - player: ObjectObservation (x, y, width, height)
        - ball: ObjectObservation (x, y, width, height)
        - blocks: jnp.ndarray (6, 18) with 0/1 values for each block
        - lives: jnp.ndarray (1) with the number of lives
        - score: jnp.ndarray (1) with the score
        """
        return spaces.Dict({
            "player": spaces.get_object_space(n=None, screen_size=(self.consts.WINDOW_HEIGHT, self.consts.WINDOW_WIDTH)),
            "ball": spaces.get_object_space(n=None, screen_size=(self.consts.WINDOW_HEIGHT, self.consts.WINDOW_WIDTH)),
            "blocks": spaces.Box(low=0, high=1, shape=(self.consts.NUM_ROWS, self.consts.BLOCKS_PER_ROW), dtype=jnp.int32),
            "lives": spaces.Box(low=0, high=self.consts.NUM_LIVES, shape=(), dtype=jnp.int32),
            "score": spaces.Box(low=0, high=jnp.iinfo(jnp.int32).max, shape=(), dtype=jnp.int32),
        })

    def image_space(self) -> spaces.Box:
        """Returns the image space for Breakout.
        The image is a RGB image with shape (210, 160, 3).
        """
        return spaces.Box(
            low=0,
            high=255,
            shape=(210, 160, 3),
            dtype=jnp.uint8
        )


class BreakoutRenderer(JAXGameRenderer):
    def __init__(self, consts: BreakoutConstants = None, config: render_utils.RendererConfig = None):
        self.consts = consts or BreakoutConstants()
        super().__init__(self.consts)
        
        # Use injected config if provided, else default
        if config is None:
            self.config = render_utils.RendererConfig(
                game_dimensions=(210, 160),
                channels=3,
                downscale=None
            )
        else:
            self.config = config
        self.jr = render_utils.JaxRenderingUtils(self.config)

        # 1. Standard API setup from the new renderer
        procedural_sprites = self._create_procedural_sprites()
        asset_config = self._get_asset_config(procedural_sprites)
        sprite_path = os.path.join(render_utils.get_base_sprite_dir(), "breakout")
        
        # Add ball and player colors to palette as 1x1 procedural sprites if they differ from defaults
        # This ensures the colors are in the palette before we recolor
        default_ball_color = (200, 72, 72)
        default_player_color = (200, 72, 72)
        
        if self.consts.BALL_COLOR != default_ball_color:
            ball_color_rgba = jnp.array(list(self.consts.BALL_COLOR) + [255], dtype=jnp.uint8).reshape(1, 1, 4)
            asset_config.append({
                'name': 'ball_color_override',
                'type': 'procedural',
                'data': ball_color_rgba
            })
        
        if self.consts.PLAYER_COLOR != default_player_color:
            player_color_rgba = jnp.array(list(self.consts.PLAYER_COLOR) + [255], dtype=jnp.uint8).reshape(1, 1, 4)
            asset_config.append({
                'name': 'player_color_override',
                'type': 'procedural',
                'data': player_color_rgba
            })
        
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(asset_config, sprite_path)

        PALETTE_ID_BLACK = 0 # in this case 0 is black which is the background color. In other cases this might need to be adjusted.
        self.COLOR_MAP = jnp.array([
            PALETTE_ID_BLACK,
            self.COLOR_TO_ID[self.consts.BLOCK_COLORS[0]], # ID 1 -> Red
            self.COLOR_TO_ID[self.consts.BLOCK_COLORS[1]], # ID 2 -> Orange
            self.COLOR_TO_ID[self.consts.BLOCK_COLORS[2]], # ID 3 -> Yellow
            self.COLOR_TO_ID[self.consts.BLOCK_COLORS[3]], # ID 4 -> Green
            self.COLOR_TO_ID[self.consts.BLOCK_COLORS[4]], # ID 5 -> Blue
            self.COLOR_TO_ID[self.consts.BLOCK_COLORS[5]], # ID 6 -> Indigo
        ])
        
        # Recolor ball and player sprites if colors differ from defaults
        self._recolor_ball_and_player(sprite_path)
    
    @partial(jax.jit, static_argnums=(0,))
    def _render_blocks_inverse(self, raster: jnp.ndarray, blocks_state: jnp.ndarray) -> jnp.ndarray:
        """Renders the block grid using a highly efficient inverse mapping technique."""
        xx, yy = self.jr._xx, self.jr._yy

        # --- 1. Inverse Mapping Calculation (No changes here) ---
        start_x = self.consts.BLOCK_START_X * self.jr.config.width_scaling
        start_y = self.consts.BLOCK_START_Y * self.jr.config.height_scaling
        block_w = self.consts.BLOCK_SIZE[0] * self.jr.config.width_scaling
        block_h = self.consts.BLOCK_SIZE[1] * self.jr.config.height_scaling

        col_idx = jnp.floor((xx - start_x) / block_w).astype(jnp.int32)
        row_idx = jnp.floor((yy - start_y) / block_h).astype(jnp.int32)

        # --- 2. Create Masks and Get Data (Changes are here) ---
        grid_mask = (row_idx >= 0) & (row_idx < self.consts.NUM_ROWS) & \
                    (col_idx >= 0) & (col_idx < self.consts.BLOCKS_PER_ROW)

        safe_row = jnp.clip(row_idx, 0, self.consts.NUM_ROWS - 1)
        safe_col = jnp.clip(col_idx, 0, self.consts.BLOCKS_PER_ROW - 1)
        active_block_map = blocks_state[safe_row, safe_col]

        # directly use the array of 6 color IDs.
        color_id_map = self.ALL_BLOCK_COLOR_IDS[safe_row]

        # --- 3. Apply to Raster (No changes here) ---
        final_block_mask = grid_mask & (active_block_map == 1)

        return jnp.where(final_block_mask, color_id_map, raster)

    def _precompute_block_masks(self):
        """Pre-computes a stack of monochrome masks, one for each possible block."""
        def draw_single_mask(x, y):
            blank_raster = jnp.zeros((210, 160), dtype=jnp.uint8)
            patch = jnp.ones((self.consts.BLOCK_SIZE[1], self.consts.BLOCK_SIZE[0]), dtype=jnp.uint8)
            return jax.lax.dynamic_update_slice(blank_raster, patch, (y, x))

        rows, cols = jnp.mgrid[:self.consts.NUM_ROWS, :self.consts.BLOCKS_PER_ROW]
        all_xs = (self.consts.BLOCK_START_X + cols * self.consts.BLOCK_SIZE[0]).flatten()
        all_ys = (self.consts.BLOCK_START_Y + rows * self.consts.BLOCK_SIZE[1]).flatten()

        return jax.vmap(draw_single_mask)(all_xs, all_ys)

    def _recolor_3d_sprite(self, sprite_array: jnp.ndarray, new_rgb_color: jnp.ndarray) -> jnp.ndarray:
        """
        Recolors the non-transparent pixels of a 3D RGBA sprite array.
        
        Args:
            sprite_array: The input array with shape (H, W, 4).
            new_rgb_color: A 3-element array for the new RGB color.
            
        Returns:
            A new array with the sprite recolored.
        """
        # Create a mask from the alpha channel (the 4th channel)
        is_visible = sprite_array[:, :, 3] > 0
        
        # Use the mask to set the RGB values (:3) of visible pixels
        recolored_array = sprite_array.at[is_visible, :3].set(new_rgb_color)
        
        return recolored_array
    
    def _recolor_ball_and_player(self, sprite_path: str):
        """
        Recolors ball and player sprites based on BALL_COLOR and PLAYER_COLOR in constants.
        Note: If player sprite is procedural (due to custom PLAYER_SIZE), it's already created
        with the correct color, so we skip recoloring it.
        """
        default_ball_color = (200, 72, 72)
        default_player_color = (200, 72, 72)
        default_player_size = (16, 4)
        
        # Recolor ball if color differs from default
        if self.consts.BALL_COLOR != default_ball_color:
            # Load the original ball sprite
            ball_sprite_file = os.path.join(sprite_path, 'ball.npy')
            original_ball_sprite = self.jr.loadFrame(ball_sprite_file)
            
            # Recolor the sprite
            new_ball_color = jnp.array(self.consts.BALL_COLOR, dtype=jnp.uint8)
            recolored_ball_sprite = self._recolor_3d_sprite(original_ball_sprite, new_ball_color)
            
            # Create a new mask from the recolored sprite
            new_ball_mask = self.jr._create_id_mask(recolored_ball_sprite, self.COLOR_TO_ID)
            self.SHAPE_MASKS["ball"] = new_ball_mask
        
        # Recolor player if color differs from default AND player size is default
        # (If player size is custom, the procedural sprite already has the correct color)
        if self.consts.PLAYER_COLOR != default_player_color and self.consts.PLAYER_SIZE == default_player_size:
            # Load the original player sprite
            player_sprite_file = os.path.join(sprite_path, 'player.npy')
            original_player_sprite = self.jr.loadFrame(player_sprite_file)
            
            # Recolor the sprite
            new_player_color = jnp.array(self.consts.PLAYER_COLOR, dtype=jnp.uint8)
            recolored_player_sprite = self._recolor_3d_sprite(original_player_sprite, new_player_color)
            
            # Create a new mask from the recolored sprite
            new_player_mask = self.jr._create_id_mask(recolored_player_sprite, self.COLOR_TO_ID)
            self.SHAPE_MASKS["player"] = new_player_mask

    def _create_procedural_sprites(self) -> dict:
        """Procedurally creates RGBA sprites for blocks, the bottom bar, and player paddle if size differs from default."""
        # Create a (6, 1, 1, 4) stack of single-pixel sprites, one for each block color.
        # This will add the block colors to our palette.
        block_colors_rgba = jnp.array(self.consts.BLOCK_COLORS, dtype=jnp.uint8)
        block_pixels = jnp.c_[block_colors_rgba, jnp.full((6, 1), 255, dtype=jnp.uint8)]
        block_sprites = block_pixels.reshape(6, 1, 1, 4)

        # Create the black bar for the bottom of the screen.
        bar_w, bar_h = 160, 14
        bottom_bar_sprite = jnp.zeros((bar_h, bar_w, 4), dtype=jnp.uint8).at[:, :, 3].set(255)

        result = {
            'block_colors': block_sprites,
            'bottom_bar': bottom_bar_sprite
        }
        
        # Create procedural player sprite if PLAYER_SIZE differs from default (16, 4)
        default_player_size = (16, 4)
        if self.consts.PLAYER_SIZE != default_player_size:
            player_w, player_h = self.consts.PLAYER_SIZE
            # Create a solid rectangle sprite with the player color
            player_sprite = jnp.zeros((player_h, player_w, 4), dtype=jnp.uint8)
            # Set RGB to player color and alpha to 255 (fully opaque)
            player_color_rgba = jnp.array(list(self.consts.PLAYER_COLOR) + [255], dtype=jnp.uint8)
            player_sprite = player_sprite.at[:, :, :].set(player_color_rgba)
            result['player'] = player_sprite

        return result

    def _get_asset_config(self, procedural_sprites: dict) -> list:
        """Returns the declarative manifest of all assets for the game."""
        asset_config = [
            {'name': 'background', 'type': 'background', 'file': 'background.npy'},
            {'name': 'ball', 'type': 'single', 'file': 'ball.npy'},
            {'name': 'score_digits', 'type': 'digits', 'pattern': 'score_{}.npy'},
            
            # Add the procedurally created sprites to the manifest
            {'name': 'block_colors', 'type': 'procedural', 'data': procedural_sprites['block_colors']},
            {'name': 'bottom_bar', 'type': 'procedural', 'data': procedural_sprites['bottom_bar']},
        ]
        
        # Use procedural player sprite if available (when PLAYER_SIZE differs from default), otherwise use file
        if 'player' in procedural_sprites:
            asset_config.append({'name': 'player', 'type': 'procedural', 'data': procedural_sprites['player']})
        else:
            asset_config.append({'name': 'player', 'type': 'single', 'file': 'player.npy'})
        
        return asset_config

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: BreakoutState) -> jnp.ndarray:
        """Renders the game state using the highly parallel einsum approach."""
        # --- 1. Initialize Raster & Draw Non-Block Sprites ---
        # Start with the background palette IDs
        raster = self.BACKGROUND

        # 2. Draw player and ball using the standard API
        raster = self.jr.render_at(raster, state.player_x, self.consts.PLAYER_START_Y, self.SHAPE_MASKS["player"])
        ball_x = jnp.where(state.game_started, state.ball_x, -10)
        ball_y = jnp.where(state.game_started, state.ball_y, -10)
        raster = self.jr.render_at(raster, ball_x, ball_y, self.SHAPE_MASKS["ball"])

        # --- 3. Draw blocks ---
        # Create a grid of object IDs (1-6) based on the row.
        row_ids = jnp.arange(1, self.consts.NUM_ROWS + 1)[:, None]

        # Multiply by the existing block state (0s and 1s) to make empty cells 0.
        object_id_grid = row_ids * state.blocks

        # The render call is now more general:
        raster = self.jr.render_grid_inverse(
            raster,
            grid_state=object_id_grid,
            grid_origin=(self.consts.BLOCK_START_X, self.consts.BLOCK_START_Y),
            cell_size=self.consts.BLOCK_SIZE,
            color_map=self.COLOR_MAP
        )
        
        # --- 4. Draw UI and Finalize ---
        player_score_digits = self.jr.int_to_digits(state.score, max_digits=3)
        player_lifes_digit = self.jr.int_to_digits(state.lives, max_digits=1)
        number_players_digit = self.jr.int_to_digits(1, max_digits=1)

        raster = self.jr.render_label_selective(raster, 36, 5,
                                    player_score_digits, self.SHAPE_MASKS['score_digits'],
                                    0, 3,
                                    spacing=16)

        raster = self.jr.render_label_selective(raster, 100, 5,
                                    player_lifes_digit, self.SHAPE_MASKS['score_digits'],
                                    0, 1,
                                    spacing=16)
        
        # 3. Render number of players
        raster = self.jr.render_label_selective(raster, 132, 5,
                                            number_players_digit, self.SHAPE_MASKS['score_digits'],
                                            0, 1,
                                            spacing=16)

        raster = self.jr.render_at(raster, 0, 196, self.SHAPE_MASKS['bottom_bar'])

        # --- 5. Final Palette Lookup ---
        return self.jr.render_from_palette(raster, self.PALETTE)