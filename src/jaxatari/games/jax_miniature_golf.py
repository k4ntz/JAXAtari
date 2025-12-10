from functools import partial
import os
from typing import NamedTuple, Tuple
import jax.lax
import jax.numpy as jnp
import chex

import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action

class MiniatureGolfConstants(NamedTuple):
    WIDTH: int = 160
    HEIGHT: int = 210
    BALL_START_X: chex.Array = jnp.array([133, 78, 6, 8, 26, 8, 8, 138, 128])
    BALL_START_Y: chex.Array = jnp.array([179, 189, 49, 147, 37, 111, 55, 49, 133])
    HOLE_X: chex.Array = jnp.array([8, 83, 83, 82, 148, 148, 153, 29, 19])
    HOLE_Y: chex.Array = jnp.array([190, 49, 123, 89, 189, 111, 55, 111, 46])
    BACKGROUND_COLOR: Tuple[int, int, int] = (92, 186, 92)
    PLAYER_COLOR: Tuple[int, int, int] = (66, 72, 200)
    OBSTACLE_COLOR: Tuple[int, int, int] = (214, 92, 92)
    OBSTACLE_MIN_X: chex.Array = jnp.array([1, 1, 1, 55, 69, 67, 78, 254, 1])               # special case level 8:
    OBSTACLE_MAX_X: chex.Array = jnp.array([35, 107, 35, 103, 69, 67, 78, 254, 109])        # barrier y counts down
    OBSTACLE_MIN_Y: chex.Array = jnp.array([155, 155, 133, 81, 197, 219, 211, 0, 155])      # from 255 to 0
    OBSTACLE_MAX_Y: chex.Array = jnp.array([155, 155, 133, 81, 121, 61, 91, 255, 155])      # wrapping back to 255
    HOLE_COLOR: Tuple[int, int, int] = (66, 72, 200)
    BALL_COLOR: Tuple[int, int, int] = (210, 210, 64)
    WALL_COLOR: Tuple[int, int, int] = (210, 210, 64)
    SCORE_COLOR: Tuple[int, int, int] = (66, 72, 200)
    PLAYER_START_X: chex.Array = jnp.array([133, 78, 6, 8, 26, 8, 8, 138, 128])
    PLAYER_START_Y: chex.Array = jnp.array([175, 185, 45, 143, 33, 107, 51, 45, 129])
    PAR_VALUES: chex.Array = jnp.array([4, 3, 4, 4, 4, 3, 7, 3, 4])
    PLAYER_SIZE: Tuple[int, int] = (4, 8)
    BALL_SIZE: Tuple[int, int] = (2, 4)
    HOLE_SIZE: Tuple[int, int] = (3, 4)
    OBSTACLE_SIZE: Tuple[int, int] = (8, 16)
    DIGIT_SIZE: Tuple[int, int] = (12, 10)
    SCORE_POS_TENS_DIGIT: Tuple[int, int] = (16, 9)
    SCORE_POS_ONES_DIGIT: Tuple[int, int] = (32, 9)
    PAR_POS: Tuple[int, int] = (111, 9)

    LEVEL_1: chex.Array = jnp.load(f"{os.path.dirname(os.path.abspath(__file__))}/sprites/miniature_golf/level_1.npy")
    LEVEL_2: chex.Array = jnp.load(f"{os.path.dirname(os.path.abspath(__file__))}/sprites/miniature_golf/level_2.npy")
    LEVEL_3: chex.Array = jnp.load(f"{os.path.dirname(os.path.abspath(__file__))}/sprites/miniature_golf/level_3.npy")
    LEVEL_4: chex.Array = jnp.load(f"{os.path.dirname(os.path.abspath(__file__))}/sprites/miniature_golf/level_4.npy")
    LEVEL_5: chex.Array = jnp.load(f"{os.path.dirname(os.path.abspath(__file__))}/sprites/miniature_golf/level_5.npy")
    LEVEL_6: chex.Array = jnp.load(f"{os.path.dirname(os.path.abspath(__file__))}/sprites/miniature_golf/level_6.npy")
    LEVEL_7: chex.Array = jnp.load(f"{os.path.dirname(os.path.abspath(__file__))}/sprites/miniature_golf/level_7.npy")
    LEVEL_8: chex.Array = jnp.load(f"{os.path.dirname(os.path.abspath(__file__))}/sprites/miniature_golf/level_8.npy")
    LEVEL_9: chex.Array = jnp.load(f"{os.path.dirname(os.path.abspath(__file__))}/sprites/miniature_golf/level_9.npy")

    WALL_LAYOUT_LEVEL_1: chex.Array = (LEVEL_1[:,:,:3] == jnp.array(WALL_COLOR))[:,:,0].astype(jnp.int4)
    WALL_LAYOUT_LEVEL_2: chex.Array = (LEVEL_2[:,:,:3] == jnp.array(WALL_COLOR))[:,:,0].astype(jnp.int4)
    WALL_LAYOUT_LEVEL_3: chex.Array = (LEVEL_3[:,:,:3] == jnp.array(WALL_COLOR))[:,:,0].astype(jnp.int4)
    WALL_LAYOUT_LEVEL_4: chex.Array = (LEVEL_4[:,:,:3] == jnp.array(WALL_COLOR))[:,:,0].astype(jnp.int4)
    WALL_LAYOUT_LEVEL_5: chex.Array = (LEVEL_5[:,:,:3] == jnp.array(WALL_COLOR))[:,:,0].astype(jnp.int4)
    WALL_LAYOUT_LEVEL_6: chex.Array = (LEVEL_6[:,:,:3] == jnp.array(WALL_COLOR))[:,:,0].astype(jnp.int4)
    WALL_LAYOUT_LEVEL_7: chex.Array = (LEVEL_7[:,:,:3] == jnp.array(WALL_COLOR))[:,:,0].astype(jnp.int4)
    WALL_LAYOUT_LEVEL_8: chex.Array = (LEVEL_8[:,:,:3] == jnp.array(WALL_COLOR))[:,:,0].astype(jnp.int4)
    WALL_LAYOUT_LEVEL_9: chex.Array = (LEVEL_9[:,:,:3] == jnp.array(WALL_COLOR))[:,:,0].astype(jnp.int4)


# immutable state container
class MiniatureGolfState(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    ball_x: chex.Array
    ball_y: chex.Array
    ball_x_subpixel: chex.Array  # see original ROM, memory address $9e
    ball_y_subpixel: chex.Array  # see original ROM, memory address $98
    ball_vel_x: chex.Array
    ball_vel_y: chex.Array
    hole_x: chex.Array
    hole_y: chex.Array
    obstacle_x: chex.Array
    obstacle_y: chex.Array
    obstacle_dir: chex.Array
    shot_count: chex.Array
    level: chex.Array
    wall_layout: chex.Array
    acceleration_threshold: chex.Array
    acceleration_counter: chex.Array
    mod_4_counter: chex.Array
    fire_prev: chex.Array


class EntityPosition(NamedTuple):
    x: chex.Array
    y: chex.Array
    width: chex.Array
    height: chex.Array


class MiniatureGolfObservation(NamedTuple):
    player: EntityPosition
    hole: EntityPosition
    ball: EntityPosition
    obstacle: EntityPosition
    shot_count: chex.Array
    wall_layout: chex.Array


class MiniatureGolfInfo(NamedTuple):
    pass


class JaxMiniatureGolf(JaxEnvironment[MiniatureGolfState, MiniatureGolfObservation, MiniatureGolfInfo, MiniatureGolfConstants]):
    def __init__(self, consts: MiniatureGolfConstants = None, reward_funcs: list[callable]=None):
        consts = consts or MiniatureGolfConstants()
        super().__init__(consts)
        self.renderer = MiniatureGolfRenderer(self.consts)
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.action_set = [
            Action.NOOP,
            Action.FIRE,
            Action.RIGHT,
            Action.LEFT,
            Action.UP,
            Action.DOWN,
        ]

    def _overlaps_wall(self, wall_layout: chex.Array, x: chex.Array, y: chex.Array):
        rows, cols = jnp.mgrid[:self.consts.HEIGHT, :self.consts.WIDTH]

        mask = jnp.logical_and(
            rows == y,
            cols == x
        )

        return jnp.any(
            jnp.logical_and(
                wall_layout == 1,
                mask
            )
        )

    def _any_corners_overlap_wall(self, wall_layout: chex.Array, x: chex.Array, y: chex.Array):
        overlap_top_left_corner = self._overlaps_wall(wall_layout, x, y)
        overlap_top_right_corner = self._overlaps_wall(wall_layout, x + self.consts.BALL_SIZE[0] - 1, y)
        overlap_bottom_left_corner = self._overlaps_wall(wall_layout, x, y + self.consts.BALL_SIZE[1] - 1)
        overlap_bottom_right_corner = self._overlaps_wall(wall_layout, x + self.consts.BALL_SIZE[0] - 1, y + self.consts.BALL_SIZE[1] - 1)
        return jnp.any(jnp.logical_or(
            jnp.logical_or(overlap_top_left_corner, overlap_top_right_corner),
            jnp.logical_or(overlap_bottom_left_corner, overlap_bottom_right_corner),
        ))

    def _is_overlapping(self, x1, y1, w1, h1, x2, y2, w2, h2):
        rows, cols = jnp.mgrid[:self.consts.HEIGHT, :self.consts.WIDTH]

        mask_first = jnp.logical_and(
            jnp.logical_and(
                cols >= x1,
                cols < x1 + w1
            ),
            jnp.logical_and(
                rows >= y1,
                rows < y1 + h1
            )
        )

        mask_second = jnp.logical_and(
            jnp.logical_and(
                cols >= x2,
                cols < x2 + w2
            ),
            jnp.logical_and(
                rows >= y2,
                rows < y2 + h2
            )
        )

        return jnp.any(jnp.logical_and(mask_first, mask_second))


    def _ball_step(self, state: MiniatureGolfState) -> MiniatureGolfState:
        ball_x_subpixel_new = state.ball_x_subpixel + state.ball_vel_x
        ball_y_subpixel_new = state.ball_y_subpixel + state.ball_vel_y
        ball_delta_x, ball_x_subpixel_new = jnp.divmod(ball_x_subpixel_new, 16)
        ball_delta_y, ball_y_subpixel_new = jnp.divmod(ball_y_subpixel_new, 16)
        ball_x_new = state.ball_x + ball_delta_x
        ball_y_new = state.ball_y + ball_delta_y

        overlap_top_left_corner = self._overlaps_wall(state.wall_layout, ball_x_new, ball_y_new)
        overlap_top_right_corner = self._overlaps_wall(state.wall_layout, ball_x_new + self.consts.BALL_SIZE[0] - 1, ball_y_new)
        overlap_bottom_left_corner = self._overlaps_wall(state.wall_layout, ball_x_new, ball_y_new + self.consts.BALL_SIZE[1] - 1)
        overlap_bottom_right_corner = self._overlaps_wall(state.wall_layout, ball_x_new + self.consts.BALL_SIZE[0] - 1, ball_y_new + self.consts.BALL_SIZE[1] - 1)

        collision_x = jnp.logical_or(
            jnp.logical_and(overlap_top_left_corner, overlap_bottom_left_corner),
            jnp.logical_and(overlap_top_right_corner, overlap_bottom_right_corner)
        )
        collision_y = jnp.logical_or(
            jnp.logical_and(overlap_top_left_corner, overlap_top_right_corner),
            jnp.logical_and(overlap_bottom_left_corner, overlap_bottom_right_corner)
        )

        # handle special case: all corners overlap
        all_corners = jnp.logical_and(
            jnp.logical_and(overlap_top_left_corner, overlap_bottom_left_corner),
            jnp.logical_and(overlap_top_right_corner, overlap_bottom_right_corner)
        )

        # handle special case: only one corner overlaps
        single_corner = jnp.logical_and(
            jnp.logical_and(
                jnp.logical_not(collision_x),
                jnp.logical_not(collision_y),
            ),
            jnp.logical_or(
                jnp.logical_or(
                    overlap_top_left_corner,
                    overlap_top_right_corner,
                ),
                jnp.logical_or(
                    overlap_bottom_left_corner,
                    overlap_bottom_right_corner,
                )
            )
        )

        overlap_only_x_change = self._any_corners_overlap_wall(state.wall_layout, ball_x_new, state.ball_y)
        overlap_only_y_change = self._any_corners_overlap_wall(state.wall_layout, state.ball_x, ball_y_new)

        collision_x = jnp.logical_or(
            collision_x,
            jnp.logical_and(
                single_corner,
                overlap_only_x_change,
            )
        )
        collision_y = jnp.logical_or(
            collision_y,
            jnp.logical_and(
                single_corner,
                overlap_only_y_change,
            )
        )

        collision_x = jnp.where(all_corners, overlap_only_x_change, collision_x)
        collision_y = jnp.where(all_corners, overlap_only_y_change, collision_y)

        # negate velocity in case of overlap
        ball_vel_x_new = jnp.where(
            collision_x,
            -state.ball_vel_x,
            state.ball_vel_x
        )
        ball_vel_y_new = jnp.where(
            collision_y,
            -state.ball_vel_y,
            state.ball_vel_y
        )

        return MiniatureGolfState(
            player_x=state.player_x,
            player_y=state.player_y,
            ball_x=ball_x_new,
            ball_y=ball_y_new,
            ball_x_subpixel=ball_x_subpixel_new,
            ball_y_subpixel=ball_y_subpixel_new,
            ball_vel_x=ball_vel_x_new,
            ball_vel_y=ball_vel_y_new,
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
        )


    def _update_velocity(self, state: MiniatureGolfState) -> MiniatureGolfState:
        acceleration_counter = jnp.where(state.acceleration_counter >= 0,
                                         state.acceleration_counter - 1,
                                         state.acceleration_counter + 1)
        decelerate = jnp.equal(acceleration_counter, 0)
        ball_vel_x_new = jnp.where(state.ball_vel_x > 0, state.ball_vel_x - 1, jnp.where(
            state.ball_vel_x < 0, state.ball_vel_x + 1, state.ball_vel_x
        ))
        ball_vel_y_new = jnp.where(state.ball_vel_y > 0, state.ball_vel_y - 1, jnp.where(
            state.ball_vel_y < 0, state.ball_vel_y + 1, state.ball_vel_y
        ))
        ball_vel_x_new = jnp.where(decelerate, ball_vel_x_new, state.ball_vel_x)
        ball_vel_y_new = jnp.where(decelerate, ball_vel_y_new, state.ball_vel_y)
        acceleration_threshold_new = jnp.where(decelerate, 1, state.acceleration_threshold)
        acceleration_counter_new = jnp.where(decelerate, acceleration_threshold_new, acceleration_counter)

        return MiniatureGolfState(
            player_x=state.player_x,
            player_y=state.player_y,
            ball_x=state.ball_x,
            ball_y=state.ball_y,
            ball_x_subpixel=state.ball_x_subpixel,
            ball_y_subpixel=state.ball_y_subpixel,
            ball_vel_x=ball_vel_x_new,
            ball_vel_y=ball_vel_y_new,
            hole_x=state.hole_x,
            hole_y=state.hole_y,
            obstacle_x=state.obstacle_x,
            obstacle_y=state.obstacle_y,
            obstacle_dir=state.obstacle_dir,
            shot_count=state.shot_count,
            level=state.level,
            wall_layout=state.wall_layout,
            acceleration_threshold=acceleration_threshold_new,
            acceleration_counter=acceleration_counter_new,
            mod_4_counter=state.mod_4_counter,
            fire_prev=state.fire_prev,
        )


    def _player_step(self, state: MiniatureGolfState, action: chex.Array) -> MiniatureGolfState:
        up = jnp.equal(action, Action.UP)
        down = jnp.equal(action, Action.DOWN)
        right = jnp.equal(action, Action.RIGHT)
        left = jnp.equal(action, Action.LEFT)
        # as in the original game, only move every fourth frame when player is close to the ball
        player_close_to_ball = jnp.logical_and(
            jnp.abs(state.ball_x - state.player_x) < 0x15,
            jnp.abs(state.ball_y - state.player_y) < 0x0b
        )
        should_move = jnp.logical_or(jnp.logical_not(player_close_to_ball), jnp.equal(state.mod_4_counter, 0))
        up = jnp.logical_and(up, should_move)
        down = jnp.logical_and(down, should_move)
        right = jnp.logical_and(right, should_move)
        left = jnp.logical_and(left, should_move)

        ball_stationary = jnp.logical_and(jnp.equal(state.ball_vel_x, 0), jnp.equal(state.ball_vel_y, 0))
        temporary_state = self._update_velocity(state)
        ball_vel_x_new = jnp.where(ball_stationary, 0, temporary_state.ball_vel_x)
        ball_vel_y_new = jnp.where(ball_stationary, 0, temporary_state.ball_vel_y)
        fire = jnp.logical_and(ball_stationary, jnp.equal(action, Action.FIRE))
        # only count FIRE if not pressed the previous frame
        fire = jnp.logical_and(fire, jnp.logical_not(state.fire_prev))
        shot_count_new = jnp.where(fire, state.shot_count + 1, state.shot_count)

        ball_vel_x_new = jnp.where(fire, (state.ball_x - state.player_x) // 4, ball_vel_x_new)
        ball_vel_y_new = jnp.where(fire, (state.ball_y - state.player_y) // 4, ball_vel_y_new)
        ball_stationary_now = jnp.logical_and(jnp.equal(ball_vel_x_new, 0), jnp.equal(ball_vel_y_new, 0))
        fire_had_effect = jnp.logical_and(ball_stationary, jnp.logical_not(ball_stationary_now))

        v_abs_x = jnp.abs(ball_vel_x_new)
        v_abs_y = jnp.abs(ball_vel_y_new) * 2
        tempo = (jnp.where(v_abs_x > v_abs_y, v_abs_x, v_abs_y) + 0x74) // 2

        acceleration_threshold_new = jnp.where(fire_had_effect, tempo, temporary_state.acceleration_threshold)
        acceleration_counter_new = jnp.where(fire_had_effect, tempo, temporary_state.acceleration_counter)

        player_y_dec = jnp.where(up, 1, 0)
        player_y_inc = jnp.where(down, 1, 0)
        player_x_inc = jnp.where(right, 1, 0)
        player_x_dec = jnp.where(left, 1, 0)
        player_x_without_fire = jnp.clip(state.player_x + player_x_inc - player_x_dec, 1, self.consts.WIDTH - self.consts.PLAYER_SIZE[0] - 1)
        player_y_without_fire = jnp.clip(state.player_y + player_y_inc - player_y_dec, 1, self.consts.HEIGHT - self.consts.PLAYER_SIZE[1] - 1)
        player_x_new = jnp.where(fire, state.ball_x, player_x_without_fire)
        player_y_new = jnp.where(fire, state.ball_y, player_y_without_fire)

        return MiniatureGolfState(
            player_x=player_x_new,
            player_y=player_y_new,
            ball_x=state.ball_x,
            ball_y=state.ball_y,
            ball_x_subpixel=state.ball_x_subpixel,
            ball_y_subpixel=state.ball_y_subpixel,
            ball_vel_x=ball_vel_x_new,
            ball_vel_y=ball_vel_y_new,
            hole_x=state.hole_x,
            hole_y=state.hole_y,
            obstacle_x=state.obstacle_x,
            obstacle_y=state.obstacle_y,
            obstacle_dir=state.obstacle_dir,
            shot_count=shot_count_new,
            level=state.level,
            wall_layout=state.wall_layout,
            acceleration_threshold=acceleration_threshold_new,
            acceleration_counter=acceleration_counter_new,
            mod_4_counter=state.mod_4_counter,
            fire_prev=jnp.equal(action, Action.FIRE),
        )

    def _score_and_reset(self, state: MiniatureGolfState) -> MiniatureGolfState:
        player_goal = self._is_overlapping(state.ball_x, state.ball_y, self.consts.BALL_SIZE[0], self.consts.BALL_SIZE[1],
                                           state.hole_x, state.hole_y, self.consts.HOLE_SIZE[0], self.consts.HOLE_SIZE[1])

        level_new = jnp.where(player_goal,
            state.level + 1,
            state.level
        )

        hole_x_new = jax.lax.select_n(
            level_new,
            self.consts.HOLE_X[0],
            self.consts.HOLE_X[1],
            self.consts.HOLE_X[2],
            self.consts.HOLE_X[3],
            self.consts.HOLE_X[4],
            self.consts.HOLE_X[5],
            self.consts.HOLE_X[6],
            self.consts.HOLE_X[7],
            self.consts.HOLE_X[8],
        )
        hole_x_new = jnp.where(player_goal, hole_x_new, state.hole_x)

        hole_y_new =jax.lax.select_n(
            level_new,
            self.consts.HOLE_Y[0],
            self.consts.HOLE_Y[1],
            self.consts.HOLE_Y[2],
            self.consts.HOLE_Y[3],
            self.consts.HOLE_Y[4],
            self.consts.HOLE_Y[5],
            self.consts.HOLE_Y[6],
            self.consts.HOLE_Y[7],
            self.consts.HOLE_Y[8],
        )
        hole_y_new = jnp.where(player_goal, hole_y_new, state.hole_y)

        player_x_new = jax.lax.select_n(
            level_new,
            self.consts.PLAYER_START_X[0],
            self.consts.PLAYER_START_X[1],
            self.consts.PLAYER_START_X[2],
            self.consts.PLAYER_START_X[3],
            self.consts.PLAYER_START_X[4],
            self.consts.PLAYER_START_X[5],
            self.consts.PLAYER_START_X[6],
            self.consts.PLAYER_START_X[7],
            self.consts.PLAYER_START_X[8],
        )
        player_x_new = jnp.where(player_goal, player_x_new, state.player_x)

        player_y_new = jax.lax.select_n(
            level_new,
            self.consts.PLAYER_START_Y[0],
            self.consts.PLAYER_START_Y[1],
            self.consts.PLAYER_START_Y[2],
            self.consts.PLAYER_START_Y[3],
            self.consts.PLAYER_START_Y[4],
            self.consts.PLAYER_START_Y[5],
            self.consts.PLAYER_START_Y[6],
            self.consts.PLAYER_START_Y[7],
            self.consts.PLAYER_START_Y[8],
        )
        player_y_new = jnp.where(player_goal, player_y_new, state.player_y)

        ball_x_new =  jax.lax.select_n(
            level_new,
            self.consts.BALL_START_X[0],
            self.consts.BALL_START_X[1],
            self.consts.BALL_START_X[2],
            self.consts.BALL_START_X[3],
            self.consts.BALL_START_X[4],
            self.consts.BALL_START_X[5],
            self.consts.BALL_START_X[6],
            self.consts.BALL_START_X[7],
            self.consts.BALL_START_X[8],
        )
        ball_x_new = jnp.where(player_goal, ball_x_new, state.ball_x)

        ball_y_new = jax.lax.select_n(
            level_new,
            self.consts.BALL_START_Y[0],
            self.consts.BALL_START_Y[1],
            self.consts.BALL_START_Y[2],
            self.consts.BALL_START_Y[3],
            self.consts.BALL_START_Y[4],
            self.consts.BALL_START_Y[5],
            self.consts.BALL_START_Y[6],
            self.consts.BALL_START_Y[7],
            self.consts.BALL_START_Y[8],
        )
        ball_y_new = jnp.where(player_goal, ball_y_new, state.ball_y)

        ball_vel_x_new = jnp.where(player_goal, 0, state.ball_vel_x)
        ball_vel_y_new = jnp.where(player_goal, 0, state.ball_vel_y)

        obstacle_x_new = jax.lax.select_n(
            level_new,
            self.consts.OBSTACLE_MIN_X[0],
            self.consts.OBSTACLE_MIN_X[1],
            self.consts.OBSTACLE_MIN_X[2],
            self.consts.OBSTACLE_MIN_X[3],
            self.consts.OBSTACLE_MIN_X[4],
            self.consts.OBSTACLE_MIN_X[5],
            self.consts.OBSTACLE_MIN_X[6],
            self.consts.OBSTACLE_MIN_X[7],
            self.consts.OBSTACLE_MIN_X[8],
        )
        obstacle_x_new = jnp.where(player_goal, obstacle_x_new, state.obstacle_x)

        obstacle_y_new = jax.lax.select_n(
            level_new,
            self.consts.OBSTACLE_MIN_Y[0],
            self.consts.OBSTACLE_MIN_Y[1],
            self.consts.OBSTACLE_MIN_Y[1],
            self.consts.OBSTACLE_MIN_Y[1],
            self.consts.OBSTACLE_MIN_Y[1],
            self.consts.OBSTACLE_MIN_Y[1],
            self.consts.OBSTACLE_MIN_Y[1],
            self.consts.OBSTACLE_MIN_Y[1],
            self.consts.OBSTACLE_MIN_Y[1],
        )
        obstacle_y_new = jnp.where(player_goal, obstacle_y_new, state.obstacle_y)

        obstacle_dir_new = jnp.where(player_goal, 0, state.obstacle_dir)

        shot_count_new = state.shot_count

        wall_layout_new = jax.lax.select_n(
            level_new,
            self.consts.WALL_LAYOUT_LEVEL_1,
            self.consts.WALL_LAYOUT_LEVEL_2,
            self.consts.WALL_LAYOUT_LEVEL_3,
            self.consts.WALL_LAYOUT_LEVEL_4,
            self.consts.WALL_LAYOUT_LEVEL_5,
            self.consts.WALL_LAYOUT_LEVEL_6,
            self.consts.WALL_LAYOUT_LEVEL_7,
            self.consts.WALL_LAYOUT_LEVEL_8,
            self.consts.WALL_LAYOUT_LEVEL_9,
        )
        wall_layout_new = jnp.where(player_goal, wall_layout_new, state.wall_layout)

        return MiniatureGolfState(
            player_x=player_x_new,
            player_y=player_y_new,
            ball_x=ball_x_new,
            ball_y=ball_y_new,
            ball_x_subpixel=state.ball_x_subpixel,  # as in the original, subpixel values are not reset
            ball_y_subpixel=state.ball_y_subpixel,
            ball_vel_x=ball_vel_x_new,
            ball_vel_y=ball_vel_y_new,
            hole_x=hole_x_new,
            hole_y=hole_y_new,
            obstacle_x=obstacle_x_new,
            obstacle_y=obstacle_y_new,
            obstacle_dir=obstacle_dir_new,
            shot_count=shot_count_new,
            level=level_new,
            wall_layout=wall_layout_new,
            acceleration_threshold=state.acceleration_threshold,
            acceleration_counter=state.acceleration_counter,
            mod_4_counter=jnp.mod(state.mod_4_counter + 1, 4),
            fire_prev=state.fire_prev,
        )

    def _obstacle_step(self, state: MiniatureGolfState) -> MiniatureGolfState:
        min_x = jax.lax.select_n(
            state.level,
            self.consts.OBSTACLE_MIN_X[0],
            self.consts.OBSTACLE_MIN_X[1],
            self.consts.OBSTACLE_MIN_X[2],
            self.consts.OBSTACLE_MIN_X[3],
            self.consts.OBSTACLE_MIN_X[4],
            self.consts.OBSTACLE_MIN_X[5],
            self.consts.OBSTACLE_MIN_X[6],
            self.consts.OBSTACLE_MIN_X[7],
            self.consts.OBSTACLE_MIN_X[8],
        )
        max_x = jax.lax.select_n(
            state.level,
            self.consts.OBSTACLE_MAX_X[0],
            self.consts.OBSTACLE_MAX_X[1],
            self.consts.OBSTACLE_MAX_X[2],
            self.consts.OBSTACLE_MAX_X[3],
            self.consts.OBSTACLE_MAX_X[4],
            self.consts.OBSTACLE_MAX_X[5],
            self.consts.OBSTACLE_MAX_X[6],
            self.consts.OBSTACLE_MAX_X[7],
            self.consts.OBSTACLE_MAX_X[8],
        )
        min_y = jax.lax.select_n(
            state.level,
            self.consts.OBSTACLE_MIN_Y[0],
            self.consts.OBSTACLE_MIN_Y[1],
            self.consts.OBSTACLE_MIN_Y[2],
            self.consts.OBSTACLE_MIN_Y[3],
            self.consts.OBSTACLE_MIN_Y[4],
            self.consts.OBSTACLE_MIN_Y[5],
            self.consts.OBSTACLE_MIN_Y[6],
            self.consts.OBSTACLE_MIN_Y[7],
            self.consts.OBSTACLE_MIN_Y[8],
        )
        max_y = jax.lax.select_n(
            state.level,
            self.consts.OBSTACLE_MAX_Y[0],
            self.consts.OBSTACLE_MAX_Y[1],
            self.consts.OBSTACLE_MAX_Y[2],
            self.consts.OBSTACLE_MAX_Y[3],
            self.consts.OBSTACLE_MAX_Y[4],
            self.consts.OBSTACLE_MAX_Y[5],
            self.consts.OBSTACLE_MAX_Y[6],
            self.consts.OBSTACLE_MAX_Y[7],
            self.consts.OBSTACLE_MAX_Y[8],
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
        )

    def reset(self, key=None) -> Tuple[MiniatureGolfObservation, MiniatureGolfState]:
        state = MiniatureGolfState(
            player_x=self.consts.PLAYER_START_X[0],
            player_y=self.consts.PLAYER_START_Y[0],
            ball_x=self.consts.BALL_START_X[0],
            ball_y=self.consts.BALL_START_Y[0],
            ball_x_subpixel=jnp.array(0),
            ball_y_subpixel=jnp.array(0),
            ball_vel_x=jnp.array(0),
            ball_vel_y=jnp.array(0),
            hole_x=self.consts.HOLE_X[0],
            hole_y=self.consts.HOLE_Y[0],
            obstacle_x=self.consts.OBSTACLE_MIN_X[0],
            obstacle_y=self.consts.OBSTACLE_MIN_Y[0],
            obstacle_dir=jnp.array(0),
            shot_count=jnp.array(1),           # as in the original game, we start at shot 1 TODO: actually, we start at 0, but 1 is displayed anyway
            level=jnp.array(0),
            wall_layout=self.consts.WALL_LAYOUT_LEVEL_1,
            acceleration_threshold=jnp.array(0),
            acceleration_counter=jnp.array(0),
            mod_4_counter=jnp.array(0),
            fire_prev=jnp.array(0),
        )
        initial_obs = self._get_observation(state)

        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: MiniatureGolfState, action: chex.Array) -> Tuple[MiniatureGolfObservation, MiniatureGolfState, float, bool, MiniatureGolfInfo]:
        previous_state = state
        state = self._ball_step(state)
        state = self._player_step(state, action)
        state = self._obstacle_step(state)
        state = self._score_and_reset(state)
        # TODO: check for collision with barrier

        done = self._get_done(state)
        env_reward = self._get_reward(previous_state, state)
        info = self._get_info(state)
        observation = self._get_observation(state)

        return observation, state, env_reward, done, info


    def render(self, state: MiniatureGolfState) -> jnp.ndarray:
        return self.renderer.render(state)

    def _get_observation(self, state: MiniatureGolfState):
        player = EntityPosition(
            x=state.player_x,
            y=state.player_y,
            width=jnp.array(self.consts.PLAYER_SIZE[0]),
            height=jnp.array(self.consts.PLAYER_SIZE[1]),
        )

        hole = EntityPosition(
            x=state.hole_x,
            y=state.hole_y,
            width=jnp.array(self.consts.HOLE_SIZE[0]),
            height=jnp.array(self.consts.HOLE_SIZE[1]),
        )

        ball = EntityPosition(
            x=state.ball_x,
            y=state.ball_y,
            width=jnp.array(self.consts.BALL_SIZE[0]),
            height=jnp.array(self.consts.BALL_SIZE[1]),
        )

        obstacle = EntityPosition(
            x=state.obstacle_x,
            y=state.obstacle_y,
            width=jnp.array(self.consts.OBSTACLE_SIZE[0]),
            height=jnp.array(self.consts.OBSTACLE_SIZE[1]),
        )

        return MiniatureGolfObservation(
            player=player,
            hole=hole,
            ball=ball,
            obstacle=obstacle,
            shot_count=state.shot_count,
            wall_layout=state.wall_layout,
        )

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: MiniatureGolfObservation) -> jnp.ndarray:
        return jnp.concatenate([
            obs.player.x.flatten(),
            obs.player.y.flatten(),
            obs.player.height.flatten(),
            obs.player.width.flatten(),
            obs.hole.x.flatten(),
            obs.hole.y.flatten(),
            obs.hole.height.flatten(),
            obs.hole.width.flatten(),
            obs.ball.x.flatten(),
            obs.ball.y.flatten(),
            obs.ball.height.flatten(),
            obs.ball.width.flatten(),
            obs.obstacle.x.flatten(),
            obs.obstacle.y.flatten(),
            obs.obstacle.height.flatten(),
            obs.obstacle.width.flatten(),
            obs.shot_count.flatten(),
            obs.wall_layout.flatten()
        ]
        )

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(6)

    def observation_space(self) -> spaces:
        return spaces.Dict({
            "player": spaces.Dict({
                "x": spaces.Box(low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.int32),
            }),
            "hole": spaces.Dict({
                "x": spaces.Box(low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.int32),
            }),
            "ball": spaces.Dict({
                "x": spaces.Box(low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.int32),
            }),
            "obstacle": spaces.Dict({
                "x": spaces.Box(low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.int32),
            }),
            "shot_count": spaces.Box(low=0, high=99, shape=(), dtype=jnp.int32),
            "wall_layout": spaces.Box(low=0, high=1, shape=(self.consts.HEIGHT, self.consts.WIDTH), dtype=jnp.int4),
        })

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.consts.HEIGHT, self.consts.WIDTH, 3),
            dtype=jnp.uint8
        )

    @partial(jax.jit, static_argnums=(0,))
    def _reward(self, state: MiniatureGolfState):
        return state.level * 100 - state.shot_count

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: MiniatureGolfState) -> MiniatureGolfInfo:
        return MiniatureGolfInfo()

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: MiniatureGolfState, state: MiniatureGolfState):
        return self._reward(state) - self._reward(previous_state)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: MiniatureGolfState) -> bool:
        return jnp.equal(state.level, 9)

class MiniatureGolfRenderer(JAXGameRenderer):
    def __init__(self, consts: MiniatureGolfConstants = None):
        super().__init__()
        self.consts = consts or MiniatureGolfConstants()
        self.config = render_utils.RendererConfig(
            game_dimensions=(self.consts.HEIGHT, self.consts.WIDTH),
            channels=3,
            #downscale=(84, 84)
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)

        # 2. Update asset config to include both walls
        asset_config = self._get_asset_config()
        sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/miniature_golf"

        # 3. Make a single call to the setup function
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(asset_config, sprite_path)

    def _get_asset_config(self) -> list:
        """Returns the declarative manifest of all assets for the game."""
        return [
            {'name': 'background', 'type': 'background', 'file': 'background.npy'},
            {'name': 'player', 'type': 'single', 'file': 'player.npy'},
            {'name': 'ball', 'type': 'single', 'file': 'ball.npy'},
            {'name': 'hole', 'type': 'single', 'file': 'hole.npy'},
            {'name': 'obstacle', 'type': 'single', 'file': 'obstacle.npy'},
            {'name': 'left_digits', 'type': 'digits', 'pattern': 'left_{}.npy'},
            {'name': 'right_digits', 'type': 'digits', 'pattern': 'right_{}.npy'},
            {'name': 'level_1', 'type': 'single', 'file': 'level_1.npy'},
            {'name': 'level_2', 'type': 'single', 'file': 'level_2.npy'},
            {'name': 'level_3', 'type': 'single', 'file': 'level_3.npy'},
            {'name': 'level_4', 'type': 'single', 'file': 'level_4.npy'},
            {'name': 'level_5', 'type': 'single', 'file': 'level_5.npy'},
            {'name': 'level_6', 'type': 'single', 'file': 'level_6.npy'},
            {'name': 'level_7', 'type': 'single', 'file': 'level_7.npy'},
            {'name': 'level_8', 'type': 'single', 'file': 'level_8.npy'},
            {'name': 'level_9', 'type': 'single', 'file': 'level_9.npy'},
        ]

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        raster = self.jr.create_object_raster(jax.lax.select_n(state.level,
                                                               self.SHAPE_MASKS["level_1"],
                                                               self.SHAPE_MASKS["level_2"],
                                                               self.SHAPE_MASKS["level_3"],
                                                               self.SHAPE_MASKS["level_4"],
                                                               self.SHAPE_MASKS["level_5"],
                                                               self.SHAPE_MASKS["level_6"],
                                                               self.SHAPE_MASKS["level_7"],
                                                               self.SHAPE_MASKS["level_8"],
                                                               self.SHAPE_MASKS["level_9"]))

        ball_mask = self.SHAPE_MASKS["ball"]
        raster = self.jr.render_at(raster, state.ball_x, state.ball_y, ball_mask)

        hole_mask = self.SHAPE_MASKS["hole"]
        raster = self.jr.render_at(raster, state.hole_x, state.hole_y, hole_mask)

        player_mask = self.SHAPE_MASKS["player"]
        raster = self.jr.render_at(raster, state.player_x, state.player_y, player_mask)

        obstacle_mask = self.SHAPE_MASKS["obstacle"]
        raster = self.jr.render_at(raster, state.obstacle_x, state.obstacle_y, obstacle_mask)

        # TODO: implement it such that changing the colors in self.consts actually has an effect
        # (i.e. with the ID mapping)

        # Stamp Score using the label utility
        left_digits = self.jr.int_to_digits(state.shot_count, max_digits=2)
        right_digits = self.jr.int_to_digits(jnp.array(0), max_digits=2) # TODO: make par count appear no shot fired in current level

        # Note: The logic for single/double digits is complex for a jitted function.
        left_digit_masks = self.SHAPE_MASKS["left_digits"] # Assumes single color
        right_digit_masks = self.SHAPE_MASKS["right_digits"] # Assumes single color

        left_single_digit = state.shot_count < 10
        left_start_index = jax.lax.select(left_single_digit, 1, 0)
        left_num_to_render = jax.lax.select(left_single_digit, 1, 2)
        left_render_x = jax.lax.select(left_single_digit,
                                         self.consts.SCORE_POS_ONES_DIGIT[0],
                                         self.consts.SCORE_POS_TENS_DIGIT[0])
        spacing = self.consts.SCORE_POS_ONES_DIGIT[0] - self.consts.SCORE_POS_TENS_DIGIT[0]

        raster = self.jr.render_label_selective(raster, left_render_x, self.consts.SCORE_POS_ONES_DIGIT[1], left_digits,
                                                left_digit_masks, left_start_index, left_num_to_render, spacing=spacing)

        right_single_digit = 0 < 10 # TODO
        right_start_index = jax.lax.select(right_single_digit, 1, 0)
        right_num_to_render = jax.lax.select(right_single_digit, 1, 2)
        right_render_x = jax.lax.select(right_single_digit,
                                        self.consts.PAR_POS[0],
                                        self.consts.PAR_POS[0] - spacing - self.consts.DIGIT_SIZE[0])

        raster = self.jr.render_label_selective(raster, right_render_x, self.consts.PAR_POS[1], right_digits,
                                                right_digit_masks, right_start_index, right_num_to_render,
                                                spacing=spacing)

        return self.jr.render_from_palette(raster, self.PALETTE)
