from functools import partial
from typing import Tuple
from jaxatari.environment import JaxEnvironment
import jax
import jax.numpy as jnp
import chex

from jaxatari.games.tennis_rebuild.tennis_states import TennisState, TennisObservation, TennisInfo, EntityPosition
from tennis_constants import *

class JaxTennis(JaxEnvironment[TennisState, TennisObservation, TennisInfo]):
    def __init__(self, frameskip=0):
        super().__init__()
        self.frameskip = frameskip

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
    def reset(self) -> Tuple[TennisState, TennisObservation]:
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
        return reset_state, self._get_observation(reset_state)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: TennisState, action: chex.Array) -> Tuple[
        TennisState, TennisObservation, float, bool, TennisInfo]:
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

            reset_state, _ = self.reset()

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

            return returned_state, self._get_observation(returned_state), self._get_reward(state, returned_state), self._get_done(returned_state), self._get_info(returned_state)

        def game_over_freeze():
            """Freezes the game after it's over."""
            return state, self._get_observation(state), self._get_reward(state, state), jnp.bool(True), self._get_info(
                state)

        return jax.lax.cond(
            self._get_done(state),
            lambda: game_over_freeze(),
            lambda: normal_play(),
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
