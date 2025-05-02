from tennis_states import *
import chex
from typing import Tuple
import jax
import jax.numpy as jnp
from tennis_constants import *
from functools import partial
from jaxatari.environment import JaxEnvironment
import jax
import chex

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
    #TOP_VALID_Z = (0, 30)
    TOP_VALID_Z = (0, 300)
    #BOTTOM_VALID_Z = (0, 30)
    BOTTOM_VALID_Z = (0, 300)

    def check_hit_zone(
        y_pos: chex.Array, z_pos: chex.Array, valid_y: tuple, valid_z: tuple
    ) -> chex.Array:
        y_valid = jnp.logical_and(y_pos >= valid_y[0], y_pos <= valid_y[1])
        z_valid = jnp.logical_and(z_pos >= valid_z[0], z_pos <= valid_z[1])
        return jnp.logical_and(y_valid, z_valid)

    # Check basic overlaps
    """ 
    player_overlap = jnp.logical_and(
        jnp.logical_and(
            state.ball_x < player_x + PLAYER_WIDTH, state.ball_x + BALL_SIZE > player_x
        ),
        jnp.logical_and(
            state.ball_y < player_y + PLAYER_HEIGHT, state.ball_y + BALL_SIZE > player_y
        ),
    ) 
    """

    player_overlap = jnp.logical_and(
        # check x axis
        jnp.logical_or(
            # check left side of ball inside player bounds
            jnp.logical_and(
                state.ball_x > player_x, state.ball_x < player_x + PLAYER_WIDTH
            ),
            # check right side of ball inside player bounds
            jnp.logical_and(
                state.ball_x + BALL_SIZE > player_x, state.ball_x + BALL_SIZE < player_x + PLAYER_WIDTH
            )
        ),
        # check y axis
        jnp.logical_or(
            # check top side of ball inside player bounds
            jnp.logical_and(
                state.ball_y > player_y, state.ball_y < player_y + PLAYER_HEIGHT
            ),
            # check bottom side of ball inside player bounds
            jnp.logical_and(
                state.ball_y + BALL_SIZE > player_y, state.ball_y + BALL_SIZE < player_y + PLAYER_HEIGHT
            )
        )
    )

    """
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
    """

    enemy_overlap = jnp.logical_and(
        # check x axis
        jnp.logical_or(
            # check left side of ball inside enemy bounds
            jnp.logical_and(
                state.ball_x > state.enemy_x, state.ball_x < state.enemy_x + PLAYER_WIDTH
            ),
            # check right side of ball inside enemy bounds
            jnp.logical_and(
                state.ball_x + BALL_SIZE > state.enemy_x, state.ball_x + BALL_SIZE < state.enemy_x + PLAYER_WIDTH
            )
        ),
        # check y axis
        jnp.logical_or(
            # check top side of ball inside enemy bounds
            jnp.logical_and(
                state.ball_y > state.enemy_y, state.ball_y < state.enemy_y + PLAYER_HEIGHT
            ),
            # check bottom side of ball inside enemy bounds
            jnp.logical_and(
                state.ball_y + BALL_SIZE > state.enemy_y, state.ball_y + BALL_SIZE < state.enemy_y + PLAYER_HEIGHT
            )
        )
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