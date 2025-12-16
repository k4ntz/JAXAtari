from typing import Tuple
import jax
import jax.numpy as jnp
from functools import partial
from jaxatari.games.jax_miniature_golf import MiniatureGolfState, MiniatureGolfObservation, MiniatureGolfInfo
from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin
import chex

# --- 1. Individual Mod Plugins ---
class LargeHoleMod(JaxAtariInternalModPlugin):
    asset_overrides = {
        "hole": {
            'name': 'hole',
            'type': 'single',
            'file': 'hole_large.npy',
        }
    }
    constants_overrides = {
        "HOLE_SIZE": (9, 12),
    }


class MovingHoleMod(JaxAtariInternalModPlugin):
    HOLE_MIN_X: chex.Array = jnp.array([8, 40, 70, 75, 148, 128, 143, 25, 19])
    HOLE_MAX_X: chex.Array = jnp.array([30, 100, 90, 82, 153, 148, 153, 33, 34])
    HOLE_MIN_Y: chex.Array = jnp.array([170, 49, 123, 89, 160, 111, 55, 97, 46])
    HOLE_MAX_Y: chex.Array = jnp.array([190, 59, 140, 115, 189, 126, 70, 111, 56])

    def _hole_step(self, state: MiniatureGolfState) -> MiniatureGolfState:
        """Move the hole in a clockwise loop."""
        min_x = jax.lax.select_n(
            state.level,
            self.HOLE_MIN_X[0], self.HOLE_MIN_X[1], self.HOLE_MIN_X[2],
            self.HOLE_MIN_X[3], self.HOLE_MIN_X[4], self.HOLE_MIN_X[5],
            self.HOLE_MIN_X[6], self.HOLE_MIN_X[7], self.HOLE_MIN_X[8],
        )
        max_x = jax.lax.select_n(
            state.level,
            self.HOLE_MAX_X[0], self.HOLE_MAX_X[1], self.HOLE_MAX_X[2],
            self.HOLE_MAX_X[3], self.HOLE_MAX_X[4], self.HOLE_MAX_X[5],
            self.HOLE_MAX_X[6], self.HOLE_MAX_X[7], self.HOLE_MAX_X[8],
        )
        min_y = jax.lax.select_n(
            state.level,
            self.HOLE_MIN_Y[0], self.HOLE_MIN_Y[1], self.HOLE_MIN_Y[2],
            self.HOLE_MIN_Y[3], self.HOLE_MIN_Y[4], self.HOLE_MIN_Y[5],
            self.HOLE_MIN_Y[6], self.HOLE_MIN_Y[7], self.HOLE_MIN_Y[8],
        )
        max_y = jax.lax.select_n(
            state.level,
            self.HOLE_MAX_Y[0], self.HOLE_MAX_Y[1], self.HOLE_MAX_Y[2],
            self.HOLE_MAX_Y[3], self.HOLE_MAX_Y[4], self.HOLE_MAX_Y[5],
            self.HOLE_MAX_Y[6], self.HOLE_MAX_Y[7], self.HOLE_MAX_Y[8],
        )

        hole_x_new = state.hole_x
        hole_y_new = state.hole_y
        hole_x_new = jnp.where(
            jnp.logical_and(state.hole_y == min_y, state.hole_x < max_x),
            hole_x_new + 1,
            hole_x_new
        )
        hole_x_new = jnp.where(
            jnp.logical_and(state.hole_y == max_y, state.hole_x > min_x),
            hole_x_new - 1,
            hole_x_new
        )
        hole_y_new = jnp.where(
            jnp.logical_and(state.hole_x == max_x, state.hole_y < max_y),
            hole_y_new + 1,
            hole_y_new
        )
        hole_y_new = jnp.where(
            jnp.logical_and(state.hole_x == min_x, state.hole_y > min_y),
            hole_y_new - 1,
            hole_y_new
        )
        # ONLY DEBUG: REMOVE THIS:
        hole_x_new = jnp.where(state.fire_prev, state.ball_x, hole_x_new)
        hole_y_new = jnp.where(state.fire_prev, state.ball_y, hole_y_new)
        #END DEBUG


        return MiniatureGolfState(
            player_x=state.player_x,
            player_y=state.player_y,
            ball_x=state.ball_x,
            ball_y=state.ball_y,
            ball_x_subpixel=state.ball_x_subpixel,
            ball_y_subpixel=state.ball_y_subpixel,
            ball_vel_x=state.ball_vel_x,
            ball_vel_y=state.ball_vel_y,
            hole_x=hole_x_new,
            hole_y=hole_y_new,
            obstacle_x=state.obstacle_x,
            obstacle_y=state.obstacle_y,
            obstacle_dir=state.obstacle_dir,
            shot_count=state.shot_count,
            level=state.level,
            wall_layout=state.wall_layout,
            acceleration_threshold=state.acceleration_threshold,
            acceleration_counter=state.acceleration_counter,
            mod_4_counter=state.mod_4_counter,
            fire_prev=state.fire_prev,
            right_number=state.right_number,
        )


class MultipleHolesMod(JaxAtariInternalModPlugin):
    # is this even possible, when MiniatureGolfObservation has only one hole?
    pass


class PermeableObstacleMod(JaxAtariInternalModPlugin):
    def _obstacle_step(self, state: MiniatureGolfState) -> MiniatureGolfState:
        min_x = jax.lax.select_n(
            state.level,
            self._env.consts.OBSTACLE_MIN_X[0],
            self._env.consts.OBSTACLE_MIN_X[1],
            self._env.consts.OBSTACLE_MIN_X[2],
            self._env.consts.OBSTACLE_MIN_X[3],
            self._env.consts.OBSTACLE_MIN_X[4],
            self._env.consts.OBSTACLE_MIN_X[5],
            self._env.consts.OBSTACLE_MIN_X[6],
            self._env.consts.OBSTACLE_MIN_X[7],
            self._env.consts.OBSTACLE_MIN_X[8],
        )
        max_x = jax.lax.select_n(
            state.level,
            self._env.consts.OBSTACLE_MAX_X[0],
            self._env.consts.OBSTACLE_MAX_X[1],
            self._env.consts.OBSTACLE_MAX_X[2],
            self._env.consts.OBSTACLE_MAX_X[3],
            self._env.consts.OBSTACLE_MAX_X[4],
            self._env.consts.OBSTACLE_MAX_X[5],
            self._env.consts.OBSTACLE_MAX_X[6],
            self._env.consts.OBSTACLE_MAX_X[7],
            self._env.consts.OBSTACLE_MAX_X[8],
        )
        min_y = jax.lax.select_n(
            state.level,
            self._env.consts.OBSTACLE_MIN_Y[0],
            self._env.consts.OBSTACLE_MIN_Y[1],
            self._env.consts.OBSTACLE_MIN_Y[2],
            self._env.consts.OBSTACLE_MIN_Y[3],
            self._env.consts.OBSTACLE_MIN_Y[4],
            self._env.consts.OBSTACLE_MIN_Y[5],
            self._env.consts.OBSTACLE_MIN_Y[6],
            self._env.consts.OBSTACLE_MIN_Y[7],
            self._env.consts.OBSTACLE_MIN_Y[8],
        )
        max_y = jax.lax.select_n(
            state.level,
            self._env.consts.OBSTACLE_MAX_Y[0],
            self._env.consts.OBSTACLE_MAX_Y[1],
            self._env.consts.OBSTACLE_MAX_Y[2],
            self._env.consts.OBSTACLE_MAX_Y[3],
            self._env.consts.OBSTACLE_MAX_Y[4],
            self._env.consts.OBSTACLE_MAX_Y[5],
            self._env.consts.OBSTACLE_MAX_Y[6],
            self._env.consts.OBSTACLE_MAX_Y[7],
            self._env.consts.OBSTACLE_MAX_Y[8],
        )

        obstacle_moves_horizontally = jnp.equal(min_y, max_y)
        obstacle_x_new = jnp.where(
            obstacle_moves_horizontally,
            state.obstacle_x + 1 - 2 * state.obstacle_dir,
            state.obstacle_x
        )
        obstacle_y_new = jnp.where(
            jnp.logical_not(obstacle_moves_horizontally),
            state.obstacle_y + 2 - 4 * state.obstacle_dir,  # y-axis is scaled by 2 in the original game
            state.obstacle_y
        )
        flip_direction = jnp.logical_or(
            jnp.logical_and(
                obstacle_moves_horizontally,
                jnp.logical_or(jnp.equal(obstacle_x_new, min_x), jnp.equal(obstacle_x_new, max_x))
            ),
            jnp.logical_and(
                jnp.logical_not(obstacle_moves_horizontally),
                jnp.logical_or(jnp.equal(obstacle_y_new, min_y), jnp.equal(obstacle_y_new, max_y))
            ),
        )
        obstacle_dir_new = jnp.where(flip_direction, 1 - state.obstacle_dir, state.obstacle_dir)

        # handle special case of level 8
        obstacle_y_new = jnp.where(state.level == 7, jnp.mod(state.obstacle_y + 1, 256), obstacle_y_new)
        obstacle_dir_new = jnp.where(state.level == 7, jnp.array(0), obstacle_dir_new)

        return MiniatureGolfState(
            player_x=state.player_x,
            player_y=state.player_y,
            ball_x=state.ball_x,
            ball_y=state.ball_y,
            ball_x_subpixel=state.ball_x_subpixel,
            ball_y_subpixel=state.ball_y_subpixel,
            ball_vel_x=state.ball_vel_x,
            ball_vel_y=state.ball_vel_y,
            hole_x=state.hole_x,
            hole_y=state.hole_y,
            obstacle_x=obstacle_x_new,
            obstacle_y=obstacle_y_new,
            obstacle_dir=obstacle_dir_new,
            shot_count=state.shot_count,
            level=state.level,
            wall_layout=state.wall_layout,
            acceleration_threshold=state.acceleration_threshold,
            acceleration_counter=state.acceleration_counter,
            mod_4_counter=state.mod_4_counter,
            fire_prev=state.fire_prev,
            right_number=state.right_number,
        )


class PermeableWallMod(JaxAtariInternalModPlugin):
    def _ball_step(self, state: MiniatureGolfState) -> MiniatureGolfState:
        ball_x_subpixel_new = state.ball_x_subpixel + state.ball_vel_x
        ball_y_subpixel_new = state.ball_y_subpixel + state.ball_vel_y
        ball_delta_x, ball_x_subpixel_new = jnp.divmod(ball_x_subpixel_new, 16)
        ball_delta_y, ball_y_subpixel_new = jnp.divmod(ball_y_subpixel_new, 16)
        ball_x_new = state.ball_x + ball_delta_x
        ball_y_new = state.ball_y + ball_delta_y * 2
        ball_x_new = jnp.mod(ball_x_new, self._env.consts.WIDTH)
        ball_y_new = jnp.mod(ball_y_new, self._env.consts.HEIGHT)

        return MiniatureGolfState(
            player_x=state.player_x,
            player_y=state.player_y,
            ball_x=ball_x_new,
            ball_y=ball_y_new,
            ball_x_subpixel=ball_x_subpixel_new,
            ball_y_subpixel=ball_y_subpixel_new,
            ball_vel_x=state.ball_vel_x,
            ball_vel_y=state.ball_vel_y,
            hole_x=state.hole_x,
            hole_y=state.hole_y,
            obstacle_x=state.obstacle_x,
            obstacle_y=state.obstacle_y,
            obstacle_dir=state.obstacle_dir,
            shot_count=state.shot_count,
            level=state.level,
            wall_layout=state.wall_layout,
            acceleration_threshold=state.acceleration_threshold,
            acceleration_counter=state.acceleration_counter,
            mod_4_counter=state.mod_4_counter,
            fire_prev=state.fire_prev,
            right_number=state.right_number,
        )


class SoftShotRequiredMod(JaxAtariInternalModPlugin):
    SPEED_THRESHOLD = 10  # determines how 'soft' the shot must be to get into the hole
    def _is_ball_in_hole(self, state: MiniatureGolfState):
        # check that the ball is not moving too fast
        slow =jnp.logical_and(
            jnp.less(jnp.abs(state.ball_vel_x), self.SPEED_THRESHOLD),
            jnp.less(jnp.abs(state.ball_vel_y), self.SPEED_THRESHOLD)
        )
        overlap =  self._env._is_overlapping(state.ball_x, state.ball_y, self._env.consts.BALL_SIZE[0], self._env.consts.BALL_SIZE[1],
                                             state.hole_x, state.hole_y, self._env.consts.HOLE_SIZE[0], self._env.consts.HOLE_SIZE[1])
        return jnp.logical_and(slow, overlap)


class StationaryObstacleMod(JaxAtariPostStepModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        """
        This function is called by the wrapper *after*
        the main step is complete.
        Access the environment via self._env (set by JaxAtariModWrapper).
        """
        return new_state._replace(
            obstacle_x=jnp.where(prev_state.level == new_state.level, prev_state.obstacle_x, new_state.obstacle_x),
            obstacle_y=jnp.where(prev_state.level == new_state.level, prev_state.obstacle_y, new_state.obstacle_y),
        )


class AlwaysZeroShotsMod(JaxAtariPostStepModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        """
        This function is called by the wrapper *after*
        the main step is complete.
        Access the environment via self._env (set by JaxAtariModWrapper).
        """
        return new_state._replace(
            shot_count=jnp.array(0, dtype=jnp.int32),
        )