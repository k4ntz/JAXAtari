from functools import partial
import os
from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
import chex
import pygame
# moin
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
import jaxatari.rendering.jax_rendering_utils as jr

class BreakoutConstants(NamedTuple):
    WINDOW_WIDTH: int = 160
    WINDOW_HEIGHT: int = 210
    BACKGROUND_COLOR: Tuple[int, int, int] = (0, 0, 0)
    PLAYER_COLOR: Tuple[int, int, int] = (200, 72, 72)
    BALL_COLOR: Tuple[int, int, int] = (200, 72, 72)
    WALL_COLOR: Tuple[int, int, int] = (142, 142, 142)
    BLOCK_COLORS: list = [
        (200, 72, 72),
        (198, 108, 58),
        (180, 122, 48),
        (162, 162, 42),
        (72, 160, 72),
        (66, 72, 200),
    ]
    PLAYER_SIZE: Tuple[int, int] = (16, 4)
    PLAYER_SIZE_SMALL: Tuple[int, int] = (12, 4)
    BALL_SIZE: Tuple[int, int] = (2, 4)
    BLOCK_SIZE: Tuple[int, int] = (8, 6)
    WALL_TOP_Y: int = 17
    WALL_TOP_HEIGHT: int = 15
    WALL_SIDE_WIDTH: int = 8
    PLAYER_START_X: int = 99
    PLAYER_START_Y: int = 189
    BALL_START_X: chex.Array = jnp.array([16, 78, 80, 142])
    BALL_START_Y: int = 122
    PLAYER_X_MIN: int = 8
    PLAYER_X_MAX: int = 160 - 16
    PLAYER_MAX_SPEED: int = 6
    PLAYER_ACCELERATION: chex.Array = jnp.array([3, 2, -1, 1, 1])
    PLAYER_WALL_ACCELERATION: chex.Array = jnp.array([1, 2, 1, 1, 1])
    BLOCKS_PER_ROW: int = 18
    NUM_ROWS: int = 6
    BLOCK_START_Y: int = 57
    BLOCK_START_X: int = 8
    NUM_LIVES: int = 5
    BALL_VELOCITIES_ABS: chex.Array = jnp.array([
        [[1, 1], [1, 1]],
        [[2, 1], [1, 1]],
        [[1, 2], [1, 1]],
        [[2, 2], [2, 2]],
        [[2, 3], [2, 3]]
    ])
    BALL_DIRECTIONS: chex.Array = jnp.array([
        [1, 1],
        [-1, 1],
        [1, -1],
        [-1, -1],
    ])
    REVERSE_X: chex.Array = jnp.array([1, 0, 3, 2])
    REVERSE_Y: chex.Array = jnp.array([2, 3, 0, 1])


class EntityPosition(NamedTuple):
    x: chex.Array
    y: chex.Array
    width: chex.Array
    height: chex.Array

class BreakoutObservation(NamedTuple):
    player: EntityPosition
    ball: EntityPosition
    blocks: chex.Array
    # TODO: move this into info??
    score: chex.Array
    lives: chex.Array

class BreakoutInfo(NamedTuple):
    time: chex.Array
    wall_resets: chex.Array
    all_rewards: chex.Array


# Game state container
class BreakoutState(NamedTuple):
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
    def __init__(self, consts: BreakoutConstants = None, reward_funcs: list[callable]=None):
        consts = consts or BreakoutConstants()
        super().__init__(consts)
        self.renderer = BreakoutRenderer(self.consts)
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs) 
        self.reward_funcs = reward_funcs 

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
        action: chex.Array,
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Updates the player position based on the action."""
        left = action == Action.LEFT
        right = action == Action.RIGHT

        # Check if the paddle is touching the left or right wall.
        touches_wall = jnp.logical_or(
            state_player_x <= self.consts.PLAYER_X_MIN, state_player_x >= self.consts.PLAYER_X_MAX
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
        player_x = jnp.clip(state_player_x + player_speed, self.consts.PLAYER_X_MIN, self.consts.PLAYER_X_MAX)

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
        all_rewards = self._get_all_reward(state, new_state)
        info = self._get_info(new_state, all_rewards)
        return obs, new_state, env_reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _step(self, state: BreakoutState, action: chex.Array) -> BreakoutState:
        # Update player position
        new_player_x, new_paddle_v, new_acceleration_counter = self._player_step(
            state.player_x, state.player_speed, state.acceleration_counter, action
        )
        #TODO: this is a hack -> always fire
        game_started = jnp.logical_or(state.game_started, True)
        # game_started = jnp.logical_or(state.game_started, action == Action.FIRE)

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
        paddle_width = jnp.where(state.small_paddle, self.consts.PLAYER_SIZE_SMALL[0], self.consts.PLAYER_SIZE[0])

        player = EntityPosition(
            x=state.player_x,
            y=jnp.array(self.consts.PLAYER_START_Y),
            width=paddle_width,
            height=jnp.array(self.consts.PLAYER_SIZE[1]),
        )

        ball = EntityPosition(
            x=state.ball_x,
            y=state.ball_y,
            width=jnp.array(self.consts.BALL_SIZE[0]),
            height=jnp.array(self.consts.BALL_SIZE[1]),
        )

        return BreakoutObservation(
            player=player,
            ball=ball,
            blocks=state.blocks,
            score=state.score,
            lives=state.lives,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: BreakoutState, all_rewards: chex.Array = None) -> BreakoutInfo:
        return BreakoutInfo(
            time=state.step_counter,
            wall_resets=state.wall_resets,
            all_rewards=all_rewards
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: BreakoutState, current_state: BreakoutState) -> chex.Array:
        return current_state.score - previous_state.score

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(self, previous_state: BreakoutState, state: BreakoutState):
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array(
            [reward_func(previous_state, state) for reward_func in self.reward_funcs]
        )
        return rewards

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: BreakoutState) -> chex.Array:
        return jnp.logical_or(state.lives <= 0, jnp.logical_or(state.all_blocks_cleared, state.step_counter >= 5000))

    def action_space(self) -> spaces.Discrete:
        """Returns the action space for Breakout.
        Actions are:
        0: NOOP
        1: FIRE
        2: UP
        3: RIGHT
        4: LEFT
        """
        #TODO: in ALE, this is 4: NOOP, FIRE, RIGHT, LEFT
        # But since actions are currently directly mapped from digits
        # return Discrete(4) would lead to not being able to use the left action
        return spaces.Discrete(5)

    def observation_space(self) -> spaces.Dict:
        """Returns the observation space for Breakout.
        The observation contains:
        - player: EntityPosition (x, y, width, height)
        - ball: EntityPosition (x, y, width, height)
        - blocks: array of shape (6, 18) with 0/1 values for each block
        - score: int (0-999999)
        - lives: int (0-5)
        """
        return spaces.Dict({
            "player": spaces.Dict({
                "x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
            }),
            "ball": spaces.Dict({
                "x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
            }),
            "blocks": spaces.Box(low=0, high=1, shape=(self.consts.NUM_ROWS, self.consts.BLOCKS_PER_ROW), dtype=jnp.int32),
            "score": spaces.Box(low=0, high=999999, shape=(), dtype=jnp.int32),
            "lives": spaces.Box(low=0, high=5, shape=(), dtype=jnp.int32),
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


    def obs_to_flat_array(self, obs: BreakoutObservation) -> jnp.ndarray:
        return jnp.concatenate([
            obs.player.x.flatten(),
            obs.player.y.flatten(),
            obs.player.width.flatten(),
            obs.player.height.flatten(),
            obs.ball.x.flatten(),
            obs.ball.y.flatten(),
            obs.ball.width.flatten(),
            obs.ball.height.flatten(),
            obs.blocks.flatten(),
            obs.score.flatten(),
            obs.lives.flatten(),
        ])
    

class BreakoutRenderer(JAXGameRenderer):
    def __init__(self, consts: BreakoutConstants = None):
        super().__init__()
        self.consts = consts or BreakoutConstants()
        self.SPRITE_BG, self.SPRITE_PLAYER, self.SPRITE_BALL, self.DIGIT_SPRITES = self.load_sprites()
        self.BLOCK_COLORS = jnp.array(self.consts.BLOCK_COLORS, dtype=jnp.uint8)

        self.BLOCK_SIZE = (self.consts.BLOCK_SIZE[0], self.consts.BLOCK_SIZE[1])

        # Pre-compute the monochrome masks. Keep color adjustable
        self.ALL_POSSIBLE_BLOCK_MASKS = self._precompute_block_masks()

    def load_sprites(self):
        """Load all sprites required for Pong rendering."""
        MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

        # Load sprites
        player = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/breakout/player.npy"))
        ball = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/breakout/ball.npy"))

        bg = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/breakout/background.npy"))

        # Convert all sprites to the expected format (add frame dimension)
        SPRITE_BG = jnp.expand_dims(bg, axis=0)
        SPRITE_PLAYER = jnp.expand_dims(player, axis=0)
        SPRITE_BALL = jnp.expand_dims(ball, axis=0)

        # Load digits for scores
        DIGIT_SPRITES = jr.load_and_pad_digits(
            os.path.join(MODULE_DIR, "sprites/breakout/score_{}.npy"),
            num_chars=10,
        )

        return (
            SPRITE_BG,
            SPRITE_PLAYER,
            SPRITE_BALL,
            DIGIT_SPRITES
        )
    
    
    def _precompute_block_masks(self):
        """
        Calculates a stack of MONOCHROME MASKS, one for each possible block position.
        """
        # This helper function now draws a block of 1s on a background of 0s.
        def draw_single_mask(x, y):
            # Canvas shape is now (Height, Width)
            blank_raster = jnp.zeros((210, 160), dtype=jnp.uint8)
            # Patch is now a monochrome block of 1s
            patch = jnp.ones(self.BLOCK_SIZE[::-1], dtype=jnp.uint8) # Use [::-1] for (H, W)
            return jax.lax.dynamic_update_slice(blank_raster, patch, (y, x))

        # Generate coordinates for ALL possible block positions (colors no longer needed here)
        rows_grid, cols_grid = jnp.meshgrid(
            jnp.arange(self.consts.NUM_ROWS, dtype=jnp.int32),
            jnp.arange(self.consts.BLOCKS_PER_ROW, dtype=jnp.int32),
            indexing='ij'
        )
        all_xs = (self.consts.BLOCK_START_X + cols_grid * self.BLOCK_SIZE[0]).flatten()
        all_ys = (self.consts.BLOCK_START_Y + rows_grid * self.BLOCK_SIZE[1]).flatten()

        # Vectorize the drawing function over all possible blocks.
        # The output is now a stack of masks.
        # Shape: (NUM_BLOCKS, HEIGHT, WIDTH)
        return jax.vmap(draw_single_mask, in_axes=(0, 0))(all_xs, all_ys)


    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        """
        Renders the current game state using JAX operations.

        Args:
            state: A PongState object containing the current game state.

        Returns:
            A JAX array representing the rendered frame.
        """
        # Create empty raster with correct orientation
        # Frame shape should be (Height, Width, Channels) = (210, 160, 3)
        raster = jr.create_initial_frame(width=160, height=210)

        # Render background - (0, 0) is top-left corner
        frame_bg = jr.get_sprite_frame(self.SPRITE_BG, 0)
        raster = jr.render_at(raster, 0, 0, frame_bg)

        # Render player paddle
        frame_player = jr.get_sprite_frame(self.SPRITE_PLAYER, 0)
        raster = jr.render_at(raster, state.player_x, self.consts.PLAYER_START_Y, frame_player)

        # Render ball - ball position is (ball_x, ball_y)
        # Move ball outside visible area when game hasn't started
        ball_x = jnp.where(state.game_started, state.ball_x, -10)
        ball_y = jnp.where(state.game_started, state.ball_y, -10)
        frame_ball = jr.get_sprite_frame(self.SPRITE_BALL, 0)
        raster = jr.render_at(raster, ball_x, ball_y, frame_ball)

        # 1. Create a mask for currently active blocks from the game state.
        active_mask = (state.blocks == 1).flatten().astype(jnp.float32)

        # 2. Apply the mask to the pre-computed rasters to zero out inactive blocks.
        # The mask is reshaped to allow broadcasting across the raster dimensions.
        rows_grid, _ = jnp.meshgrid(
            jnp.arange(self.consts.NUM_ROWS),
            jnp.arange(self.consts.BLOCKS_PER_ROW),
            indexing='ij'
        )
        all_colors = self.BLOCK_COLORS[rows_grid.flatten()]

        # 3. Combine masks, colors, and the active_mask to create the final layer.
        # `einsum` provides a clean way to multiply and sum along the correct axes.
        # 'b' = block, 'h' = height, 'w' = width, 'c' = channel
        # This multiplies each block's mask (bhw) by its color (bc) and the
        # activity scalar (b), then sums along the 'b' axis.
        blocks_layer = jnp.einsum(
            'b,bhw,bc->hwc',
            active_mask,
            self.ALL_POSSIBLE_BLOCK_MASKS,
            all_colors
        )

        # 4. Add the block layer onto the main raster.
        raster += blocks_layer

        # 1. Get digit array
        player_score_digits = jr.int_to_digits(state.score, max_digits=3)
        player_lifes_digit = jr.int_to_digits(state.lives, max_digits=1)
        number_players_digit = jr.int_to_digits(1, max_digits=1)

        # score starts at 36, 5
        # number of lives at 100, 5
        # number players at 132, 5 (always 1 for us)

        # 1. Render score
        raster = jr.render_label_selective(raster, 36, 5,
                                            player_score_digits, self.DIGIT_SPRITES,
                                            0, 3,
                                            spacing=16)
        
        # 2. Render number of lives
        raster = jr.render_label_selective(raster, 100, 5,
                                            player_lifes_digit, self.DIGIT_SPRITES,
                                            0, 1,
                                            spacing=16)
        
        # 3. Render number of players
        raster = jr.render_label_selective(raster, 132, 5,
                                            number_players_digit, self.DIGIT_SPRITES,
                                            0, 1,
                                            spacing=16)

        # after y=196 til y=210 render a black rectangle (its blocking the view of the ball)
        # Frame is (Height, Width, Channels) so we index as [y_range, x_range, :]
        # Force the last 14 rows (y=196 to y=210) to be black
        raster = raster.at[196:210, :, :].set(0)

        return raster
