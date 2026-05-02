import functools
from typing import Any, Dict, Tuple, Union
import chex
import jax
import jax.numpy as jnp
from jax import lax
from jaxatari.games.jax_tennis import TennisState, EnemyState

from jaxatari.modification import JaxAtariPostStepModPlugin, JaxAtariInternalModPlugin

class RandomBallSpeedWrapper(JaxAtariPostStepModPlugin):
    """Ball has random speed after every hit."""
    @functools.partial(jax.jit, static_argnums=(0,))
    def make_random(self, prev_state: TennisState, state: TennisState) -> TennisState:
        was_ball_hit = jnp.logical_and(
            prev_state.ball_state.ball_x == prev_state.ball_state.ball_hit_start_x,
            prev_state.ball_state.ball_y == prev_state.ball_state.ball_hit_start_y,
        )

        key, subkey = jax.random.split(state.random_key)
        ball_speed_modifier = jax.random.uniform(subkey, shape=()) + 0.5

        new_move_x = state.ball_state.move_x * ball_speed_modifier
        new_move_y = state.ball_state.move_y * ball_speed_modifier

        new_state = state.replace(
            ball_state = state.ball_state.replace(
                move_x=new_move_x,
                move_y=new_move_y,
            ),
            random_key = key,
        )

        return lax.cond(
            was_ball_hit,
            lambda _: new_state,
            lambda _: state,
            operand=None
        )

    def run(self, prev_state: TennisState, new_state: TennisState) -> TennisState:
        return self.make_random(prev_state, new_state)

class RandomWalkSpeedWrapper(JaxAtariPostStepModPlugin):
    """Player has random walk speed everytime they start moving."""
    @functools.partial(jax.jit, static_argnums=(0,))
    def make_random(self, prev_state: TennisState, state: TennisState) -> TennisState:
        diff_x = state.player_state.player_x - prev_state.player_state.player_x
        diff_y = state.player_state.player_y - prev_state.player_state.player_y

        did_not_walk = jnp.logical_and(diff_x <= 0, diff_y <= 0)

        key, subkey = jax.random.split(state.random_key)
        new_walk_speed = jax.random.uniform(subkey, shape=()) + 0.5

        new_state = state.replace(
            player_state = state.player_state.replace(
                player_walk_speed=new_walk_speed,
            ),
            random_key = key,
        )

        return lax.cond(
            did_not_walk,
            lambda _: new_state,
            lambda _: state,
            operand=None
        )

    def run(self, prev_state: TennisState, new_state: TennisState) -> TennisState:
        return self.make_random(prev_state, new_state)


class FastPlayerMod(JaxAtariPostStepModPlugin):
    """
    Increases player walk speed.
    """
    def after_reset(self, obs, state: TennisState) -> Tuple[Any, TennisState]:
        return obs, state.replace(player_state=state.player_state.replace(player_walk_speed=jnp.array(2.0)))

    def run(self, prev_state: TennisState, new_state: TennisState) -> TennisState:
        # Also ensure it stays fast if something else resets it
        return new_state.replace(player_state=new_state.player_state.replace(player_walk_speed=jnp.array(2.0)))


class SuperGravityMod(JaxAtariInternalModPlugin):
    """
    Increases ball gravity.
    """
    constants_overrides = {
        "BALL_GRAVITY_PER_FRAME": 2.2,
    }


class LazyEnemyMod(JaxAtariInternalModPlugin):
    """
    Enemy moves slower.
    """
    @functools.partial(jax.jit, static_argnums=(0,))
    def _enemy_step(self, state: TennisState) -> EnemyState:
        # Re-implementation of enemy step with slower movement
        enemy_x_hit_point = state.enemy_state.enemy_x + self._env.consts.PLAYER_WIDTH / 2
        player_x_hit_point = state.player_state.player_x + self._env.consts.PLAYER_WIDTH / 2
        ball_tracking_tolerance = 1
        x_tracking_tolerance = 2

        def move_x_to_middle():
            middle_step_x = jnp.where(jnp.less_equal(state.enemy_state.enemy_x, self._env.consts.GAME_MIDDLE_HORIZONTAL),
                                      state.enemy_state.enemy_x + 0.5,
                                      state.enemy_state.enemy_x - 0.5)
            return jnp.where(jnp.abs(state.enemy_state.enemy_x - self._env.consts.GAME_MIDDLE_HORIZONTAL) > 0.5, middle_step_x,
                             state.enemy_state.enemy_x)

        def track_ball_x():
            enemy_aiming_x_offset = jnp.where(
                player_x_hit_point < self._env.consts.FRAME_WIDTH / 2,
                5,
                -15
            )
            diff = state.ball_state.ball_x - (enemy_x_hit_point + enemy_aiming_x_offset)

            # move right if ball is sufficiently to the right
            new_enemy_x = jnp.where(
                diff > x_tracking_tolerance,
                state.enemy_state.enemy_x + 0.5,
                state.enemy_state.enemy_x
            )

            # move left if ball is sufficiently to the left
            new_enemy_x = jnp.where(
                diff < -x_tracking_tolerance,
                state.enemy_state.enemy_x - 0.5,
                new_enemy_x
            )
            return new_enemy_x

        new_enemy_x = jax.lax.cond(state.ball_state.last_hit == 1, move_x_to_middle, track_ball_x)

        cur_walking_direction = jnp.where(
            new_enemy_x - state.enemy_state.enemy_x < 0,
            -1,
            state.enemy_state.prev_walking_direction
        )
        cur_walking_direction = jnp.where(
            new_enemy_x - state.enemy_state.enemy_x > 0,
            1,
            cur_walking_direction
        )

        should_perform_direction_change = jnp.logical_or(
            jnp.abs((enemy_x_hit_point) - state.ball_state.ball_x) >= ball_tracking_tolerance,
            state.ball_state.last_hit == 1
        )
        new_enemy_x = jnp.where(should_perform_direction_change, new_enemy_x, state.enemy_state.enemy_x)

        def enemy_y_step():
            # Enemy moves slower in Y as well
            y_speed = 0.5
            state_after_y = jax.lax.cond(
                state.ball_state.last_hit == 1,
                lambda _: EnemyState(state.enemy_state.enemy_x,
                                     jnp.where(jnp.logical_and(state.enemy_state.enemy_y != self._env.consts.PLAYER_Y_LOWER_BOUND_TOP,
                                                               state.enemy_state.enemy_y != self._env.consts.PLAYER_Y_UPPER_BOUND_BOTTOM),
                                               state.enemy_state.enemy_y - state.player_state.player_field * y_speed,
                                               state.enemy_state.enemy_y
                                               ), state.enemy_state.prev_walking_direction,
                                     state.enemy_state.enemy_direction, jnp.array(1)),
                lambda _: EnemyState(state.enemy_state.enemy_x,
                                     jnp.where(state.player_state.player_field == 1,
                                               jnp.clip(state.enemy_state.enemy_y +
                                                              state.player_state.player_field * state.enemy_state.y_movement_direction * y_speed,
                                                              self._env.consts.PLAYER_Y_UPPER_BOUND_BOTTOM,
                                                              self._env.consts.PLAYER_Y_LOWER_BOUND_BOTTOM),
                                               jnp.clip(state.enemy_state.enemy_y +
                                                              state.player_state.player_field * state.enemy_state.y_movement_direction * y_speed,
                                                              self._env.consts.PLAYER_Y_UPPER_BOUND_TOP,
                                                              self._env.consts.PLAYER_Y_LOWER_BOUND_TOP)),
                                     state.enemy_state.prev_walking_direction,
                                     state.enemy_state.enemy_direction,
                                     jnp.where(jnp.logical_or(state.enemy_state.enemy_y == self._env.consts.PLAYER_Y_UPPER_BOUND_TOP,
                                                              state.enemy_state.enemy_y == self._env.consts.PLAYER_Y_LOWER_BOUND_BOTTOM),
                                               jnp.array(-1),
                                               state.enemy_state.y_movement_direction)), operand=None)

            return jax.lax.cond(
                state.game_state.is_serving,
                lambda _: state.enemy_state,
                lambda _: state_after_y,
                operand=None
            )

        enemy_state_after_y_step = enemy_y_step()
        new_enemy_direction = jnp.where(state.enemy_state.enemy_x > state.ball_state.ball_x, -1, state.enemy_state.enemy_direction)
        new_enemy_direction = jnp.where(state.enemy_state.enemy_x < state.ball_state.ball_x, 1, new_enemy_direction)

        return EnemyState(
            new_enemy_x,
            enemy_state_after_y_step.enemy_y,
            cur_walking_direction,
            new_enemy_direction,
            enemy_state_after_y_step.y_movement_direction
        )


class HighBounceMod(JaxAtariInternalModPlugin):
    """
    Increases ball bounce velocity.
    """
    constants_overrides = {
        "BALL_SERVING_BOUNCE_VELOCITY_BASE": 30.0,
    }

class FastEnemyMod(JaxAtariInternalModPlugin):
    """
    Enemy moves faster.
    """
    @functools.partial(jax.jit, static_argnums=(0,))
    def _enemy_step(self, state: TennisState) -> EnemyState:
        # Re-implementation of enemy step with faster movement
        enemy_x_hit_point = state.enemy_state.enemy_x + self._env.consts.PLAYER_WIDTH / 2
        player_x_hit_point = state.player_state.player_x + self._env.consts.PLAYER_WIDTH / 2
        ball_tracking_tolerance = 1
        x_tracking_tolerance = 2

        def move_x_to_middle():
            middle_step_x = jnp.where(jnp.less_equal(state.enemy_state.enemy_x, self._env.consts.GAME_MIDDLE_HORIZONTAL),
                                      state.enemy_state.enemy_x + 2,
                                      state.enemy_state.enemy_x - 2)
            return jnp.where(jnp.abs(state.enemy_state.enemy_x - self._env.consts.GAME_MIDDLE_HORIZONTAL) > 2, middle_step_x,
                             state.enemy_state.enemy_x)

        def track_ball_x():
            enemy_aiming_x_offset = jnp.where(
                player_x_hit_point < self._env.consts.FRAME_WIDTH / 2,
                5,
                -15
            )
            diff = state.ball_state.ball_x - (enemy_x_hit_point + enemy_aiming_x_offset)

            # move right if ball is sufficiently to the right
            new_enemy_x = jnp.where(
                diff > x_tracking_tolerance,
                state.enemy_state.enemy_x + 2,
                state.enemy_state.enemy_x
            )

            # move left if ball is sufficiently to the left
            new_enemy_x = jnp.where(
                diff < -x_tracking_tolerance,
                state.enemy_state.enemy_x - 2,
                new_enemy_x
            )
            return new_enemy_x

        new_enemy_x = jax.lax.cond(state.ball_state.last_hit == 1, move_x_to_middle, track_ball_x)

        cur_walking_direction = jnp.where(
            new_enemy_x - state.enemy_state.enemy_x < 0,
            -1,
            state.enemy_state.prev_walking_direction
        )
        cur_walking_direction = jnp.where(
            new_enemy_x - state.enemy_state.enemy_x > 0,
            1,
            cur_walking_direction
        )

        should_perform_direction_change = jnp.logical_or(
            jnp.abs((enemy_x_hit_point) - state.ball_state.ball_x) >= ball_tracking_tolerance,
            state.ball_state.last_hit == 1
        )
        new_enemy_x = jnp.where(should_perform_direction_change, new_enemy_x, state.enemy_state.enemy_x)

        def enemy_y_step():
            y_speed = 2.0
            state_after_y = jax.lax.cond(
                state.ball_state.last_hit == 1,
                lambda _: EnemyState(state.enemy_state.enemy_x,
                                     jnp.where(jnp.logical_and(state.enemy_state.enemy_y != self._env.consts.PLAYER_Y_LOWER_BOUND_TOP,
                                                               state.enemy_state.enemy_y != self._env.consts.PLAYER_Y_UPPER_BOUND_BOTTOM),
                                               state.enemy_state.enemy_y - state.player_state.player_field * y_speed,
                                               state.enemy_state.enemy_y
                                               ), state.enemy_state.prev_walking_direction,
                                     state.enemy_state.enemy_direction, jnp.array(1)),
                lambda _: EnemyState(state.enemy_state.enemy_x,
                                     jnp.where(state.player_state.player_field == 1,
                                               jnp.clip(state.enemy_state.enemy_y +
                                                              state.player_state.player_field * state.enemy_state.y_movement_direction * y_speed,
                                                              self._env.consts.PLAYER_Y_UPPER_BOUND_BOTTOM,
                                                              self._env.consts.PLAYER_Y_LOWER_BOUND_BOTTOM),
                                               jnp.clip(state.enemy_state.enemy_y +
                                                              state.player_state.player_field * state.enemy_state.y_movement_direction * y_speed,
                                                              self._env.consts.PLAYER_Y_UPPER_BOUND_TOP,
                                                              self._env.consts.PLAYER_Y_LOWER_BOUND_TOP)),
                                     state.enemy_state.prev_walking_direction,
                                     state.enemy_state.enemy_direction,
                                     jnp.where(jnp.logical_or(state.enemy_state.enemy_y == self._env.consts.PLAYER_Y_UPPER_BOUND_TOP,
                                                              state.enemy_state.enemy_y == self._env.consts.PLAYER_Y_LOWER_BOUND_BOTTOM),
                                               jnp.array(-1),
                                               state.enemy_state.y_movement_direction)), operand=None)

            return jax.lax.cond(
                state.game_state.is_serving,
                lambda _: state.enemy_state,
                lambda _: state_after_y,
                operand=None
            )

        enemy_state_after_y_step = enemy_y_step()
        new_enemy_direction = jnp.where(state.enemy_state.enemy_x > state.ball_state.ball_x, -1, state.enemy_state.enemy_direction)
        new_enemy_direction = jnp.where(state.enemy_state.enemy_x < state.ball_state.ball_x, 1, new_enemy_direction)

        return EnemyState(
            new_enemy_x,
            enemy_state_after_y_step.enemy_y,
            cur_walking_direction,
            new_enemy_direction,
            enemy_state_after_y_step.y_movement_direction
        )
