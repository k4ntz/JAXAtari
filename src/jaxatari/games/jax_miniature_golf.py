from collections import deque
from functools import partial
import os
from typing import Tuple
import jax.lax
import jax.numpy as jnp
import chex
import numpy as np
from flax import struct

import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action, ObjectObservation
from jaxatari.modification import AutoDerivedConstants


_SPRITE_DIR = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/miniature_golf"
_WALL_RGB = np.array([210, 210, 64], dtype=np.int32)

HOLE_X: Tuple[int, int, int, int, int, int, int, int, int] = (8, 83, 83, 82, 148, 148, 153, 29, 19)
HOLE_Y: Tuple[int, int, int, int, int, int, int, int, int] = (190, 49, 123, 89, 189, 111, 55, 111, 46)
HOLE_SIZE: Tuple[int, int] = (3, 4)
BALL_SIZE: Tuple[int, int] = (2, 4)


def _get_default_asset_config() -> tuple:
    """Returns the declarative manifest of all default assets for the game."""
    return (
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
    )


def _get_score_mask(
    wall_layout: np.ndarray,
    hole_x: int,
    hole_y: int,
    hole_w: int,
    hole_h: int,
    ball_width: int,
    ball_height: int,
) -> np.ndarray:
    """CPU BFS distance-to-hole potential used by ManhattanRewardMod."""
    wall_layout = np.asarray(wall_layout)
    dist = np.full(wall_layout.shape, np.inf, dtype=np.float64)
    q: deque[Tuple[int, int]] = deque()
    for y in range(hole_y - ball_height + 1, hole_y + hole_h):
        for x in range(hole_x - ball_width + 1, hole_x + hole_w):
            dist[y, x] = 0
            q.append((x, y))

    height, width = wall_layout.shape
    while q:
        x, y = q.popleft()
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < width and 0 <= ny < height and dist[ny, nx] > dist[y, x] + 1 and wall_layout[ny, nx] != 1:
                dist[ny, nx] = dist[y, x] + 1
                q.append((nx, ny))

    return (1.0 / (1.0 + dist)).astype(np.float32)


def _load_level_collision_data() -> Tuple[chex.Array, chex.Array]:
    """Load wall layouts / score masks on CPU, then upload once as stacked arrays."""
    walls = []
    for i in range(1, 10):
        level = np.load(f"{_SPRITE_DIR}/level_{i}.npy")
        walls.append(np.all(level[:, :, :3] == _WALL_RGB, axis=-1).astype(np.int32))
    wall_layouts_np = np.stack(walls, axis=0)
    score_masks_np = np.stack(
        [
            _get_score_mask(
                wall_layouts_np[i],
                HOLE_X[i],
                HOLE_Y[i],
                HOLE_SIZE[0],
                HOLE_SIZE[1],
                BALL_SIZE[0],
                BALL_SIZE[1],
            )
            for i in range(9)
        ],
        axis=0,
    )
    return jnp.asarray(wall_layouts_np), jnp.asarray(score_masks_np)


_WALL_LAYOUTS, _SCORE_MASKS = _load_level_collision_data()


class MiniatureGolfConstants(AutoDerivedConstants):
    WIDTH: int = struct.field(pytree_node=False, default=160)
    HEIGHT: int = struct.field(pytree_node=False, default=210)
    BALL_START_X: Tuple[int, int, int, int, int, int, int, int, int] = struct.field(pytree_node=False, default=(133, 78, 6, 8, 26, 8, 8, 138, 128))
    BALL_START_Y: Tuple[int, int, int, int, int, int, int, int, int] = struct.field(pytree_node=False, default=(179, 189, 49, 147, 37, 111, 55, 49, 133))
    HOLE_X: Tuple[int, int, int, int, int, int, int, int, int] = struct.field(pytree_node=False, default=HOLE_X)
    HOLE_Y: Tuple[int, int, int, int, int, int, int, int, int] = struct.field(pytree_node=False, default=HOLE_Y)
    BACKGROUND_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(92, 186, 92))
    PLAYER_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(66, 72, 200))
    OBSTACLE_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(214, 92, 92))
    OBSTACLE_MIN_X: Tuple[int, int, int, int, int, int, int, int, int] = struct.field(pytree_node=False, default=(1, 1, 1, 55, 69, 67, 78, 26, 1))
    OBSTACLE_MAX_X: Tuple[int, int, int, int, int, int, int, int, int] = struct.field(pytree_node=False, default=(35, 148, 35, 103, 69, 67, 78, 26, 109))
    OBSTACLE_MIN_Y: Tuple[int, int, int, int, int, int, int, int, int] = struct.field(pytree_node=False, default=(121, 121, 99, 47, 87, 27, 57, 0, 121))
    OBSTACLE_MAX_Y: Tuple[int, int, int, int, int, int, int, int, int] = struct.field(pytree_node=False, default=(121, 121, 99, 47, 163, 185, 177, 255, 121))
    HOLE_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(66, 72, 200))
    BALL_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(210, 210, 64))
    WALL_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(210, 210, 64))
    SCORE_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(66, 72, 200))
    PLAYER_START_X: Tuple[int, int, int, int, int, int, int, int, int] = struct.field(pytree_node=False, default=(133, 78, 6, 8, 26, 8, 8, 138, 128))
    PLAYER_START_Y: Tuple[int, int, int, int, int, int, int, int, int] = struct.field(pytree_node=False, default=(175, 185, 45, 143, 33, 107, 51, 45, 129))
    PLAYER_MIN_Y: int = struct.field(pytree_node=False, default=23)
    PLAYER_MAX_Y: int = struct.field(pytree_node=False, default=195)
    PAR_VALUES: Tuple[int, int, int, int, int, int, int, int, int] = struct.field(pytree_node=False, default=(4, 3, 4, 4, 4, 3, 7, 3, 4))
    PLAYER_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default=(4, 8))
    BALL_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default=BALL_SIZE)
    HOLE_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default=HOLE_SIZE)
    OBSTACLE_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default=(8, 16))
    DIGIT_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default=(12, 10))
    SCORE_POS_TENS_DIGIT: Tuple[int, int] = struct.field(pytree_node=False, default=(16, 9))
    SCORE_POS_ONES_DIGIT: Tuple[int, int] = struct.field(pytree_node=False, default=(32, 9))
    PAR_POS: Tuple[int, int] = struct.field(pytree_node=False, default=(112, 9))
    NUM_LEVELS: int = struct.field(pytree_node=False, default=9)
    # 0-indexed hole to start on (0 = hole 1). Overridable via consts or start_level_* mods.
    START_LEVEL: int = struct.field(pytree_node=False, default=0)
    # ALE exposes diagonal actions but resolves them to horizontal only; True enables true diagonals.
    ALLOW_DIAGONAL_MOVEMENT: bool = struct.field(pytree_node=False, default=False)

    # Stacked (NUM_LEVELS, H, W) for O(1) level lookup instead of select_n over 9 large arrays.
    WALL_LAYOUTS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: _WALL_LAYOUTS)
    SCORE_MASKS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: _SCORE_MASKS)

    ASSET_CONFIG: tuple = struct.field(pytree_node=False, default_factory=_get_default_asset_config)


# immutable state container
class MiniatureGolfState(struct.PyTreeNode):
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
    acceleration_threshold: chex.Array
    acceleration_counter: chex.Array
    mod_4_counter: chex.Array
    fire_prev: chex.Array
    right_number: chex.Array


class MiniatureGolfObservation(struct.PyTreeNode):
    player: ObjectObservation
    hole: ObjectObservation
    ball: ObjectObservation
    obstacle: ObjectObservation
    shot_count: chex.Array


class MiniatureGolfInfo(struct.PyTreeNode):
    pass


class JaxMiniatureGolf(JaxEnvironment[MiniatureGolfState, MiniatureGolfObservation, MiniatureGolfInfo, MiniatureGolfConstants]):
    # Full ALE action set. Diagonals exist but move horizontally only unless ALLOW_DIAGONAL_MOVEMENT.
    ACTION_SET: jnp.ndarray = jnp.array(
        [
            Action.NOOP, Action.FIRE, Action.UP, Action.RIGHT, Action.LEFT, Action.DOWN,
            Action.UPRIGHT, Action.UPLEFT, Action.DOWNRIGHT, Action.DOWNLEFT,
            Action.UPFIRE, Action.RIGHTFIRE, Action.LEFTFIRE, Action.DOWNFIRE,
            Action.UPRIGHTFIRE, Action.UPLEFTFIRE, Action.DOWNRIGHTFIRE, Action.DOWNLEFTFIRE,
        ],
        dtype=jnp.int32,
    )

    def __init__(self, consts: MiniatureGolfConstants = None, reward_funcs: list[callable]=None):
        consts = consts or MiniatureGolfConstants()
        super().__init__(consts)
        self.renderer = MiniatureGolfRenderer(self.consts)
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs

    def _wall_layout_for_level(self, level: chex.Array) -> chex.Array:
        return self.consts.WALL_LAYOUTS[jnp.clip(level, max=self.consts.NUM_LEVELS - 1)]

    def _score_mask_for_level(self, level: chex.Array) -> chex.Array:
        return self.consts.SCORE_MASKS[jnp.clip(level, max=self.consts.NUM_LEVELS - 1)]

    @staticmethod
    def _is_fire_action(action: chex.Array) -> chex.Array:
        return (
            (action == Action.FIRE)
            | (action == Action.UPFIRE)
            | (action == Action.RIGHTFIRE)
            | (action == Action.LEFTFIRE)
            | (action == Action.DOWNFIRE)
            | (action == Action.UPRIGHTFIRE)
            | (action == Action.UPLEFTFIRE)
            | (action == Action.DOWNRIGHTFIRE)
            | (action == Action.DOWNLEFTFIRE)
        )

    def _overlaps_wall(self, wall_layout: chex.Array, x: chex.Array, y: chex.Array):
        return wall_layout[y, x] == 1

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
        """Check if rectangles with width wi, height hi and upper-left corner at (xi, yi) overlap."""
        x_no_overlap = jnp.logical_or(
            x1 + w1 <= x2,
            x2 + w2 <= x1
        )
        y_no_overlap = jnp.logical_or(
            y1 + h1 <= y2,
            y2 + h2 <= y1
        )
        return jnp.logical_not(jnp.logical_or(x_no_overlap, y_no_overlap))


    def _ball_step(self, state: MiniatureGolfState) -> MiniatureGolfState:
        ball_x_subpixel_new = state.ball_x_subpixel + state.ball_vel_x
        ball_y_subpixel_new = state.ball_y_subpixel + state.ball_vel_y
        ball_delta_x, ball_x_subpixel_new = jnp.divmod(ball_x_subpixel_new, 16)
        ball_delta_y, ball_y_subpixel_new = jnp.divmod(ball_y_subpixel_new, 16)
        ball_x_new = state.ball_x + ball_delta_x
        ball_y_new = state.ball_y + ball_delta_y * 2

        wall_layout = self._wall_layout_for_level(state.level)

        overlap_top_left_corner = self._overlaps_wall(wall_layout, ball_x_new, ball_y_new)
        overlap_top_right_corner = self._overlaps_wall(wall_layout, ball_x_new + self.consts.BALL_SIZE[0] - 1, ball_y_new)
        overlap_bottom_left_corner = self._overlaps_wall(wall_layout, ball_x_new, ball_y_new + self.consts.BALL_SIZE[1] - 1)
        overlap_bottom_right_corner = self._overlaps_wall(wall_layout, ball_x_new + self.consts.BALL_SIZE[0] - 1, ball_y_new + self.consts.BALL_SIZE[1] - 1)

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

        overlap_only_x_change = self._any_corners_overlap_wall(wall_layout, ball_x_new, state.ball_y)
        overlap_only_y_change = self._any_corners_overlap_wall(wall_layout, state.ball_x, ball_y_new)

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

        # prevent ball from getting stuck in the wall
        x_positions_to_check = ball_x_new + jnp.arange(self.consts.BALL_SIZE[0])
        y_positions_to_check = ball_y_new + jnp.arange(self.consts.BALL_SIZE[1])
        vmap = jax.vmap(jax.vmap(self._overlaps_wall, in_axes=(None, None, 0)), in_axes=(None, 0, None))
        no_overlap = jnp.logical_not(vmap(wall_layout, x_positions_to_check, y_positions_to_check))
        x_no_overlap, y_no_overlap = jnp.nonzero(no_overlap, size=self.consts.BALL_SIZE[0] * self.consts.BALL_SIZE[1],
                                                 fill_value=jnp.nan)
        x_min = jnp.nanmin(x_no_overlap).astype(jnp.int32)
        x_max = jnp.nanmax(x_no_overlap).astype(jnp.int32) + 1
        y_min = jnp.nanmin(y_no_overlap).astype(jnp.int32)
        y_max = jnp.nanmax(y_no_overlap).astype(jnp.int32) + 1

        def get_x_when_stuck_in_wall():
            # since the ball's width is only 2, it gets stuck in the wall within a single frame occasionally
            x_positions_to_check_ = ball_x_new + jnp.arange(-2, self.consts.BALL_SIZE[0] + 2)
            no_overlap_ = jnp.logical_not(vmap(wall_layout, x_positions_to_check_, y_positions_to_check))
            x_no_overlap_, _ = jnp.nonzero(
                no_overlap_, size=(4 + self.consts.BALL_SIZE[0]) * self.consts.BALL_SIZE[1], fill_value=jnp.nan
            )
            x_min_ = jnp.nanmin(x_no_overlap_).astype(jnp.int32) - 2
            x_max_ = jnp.nanmax(x_no_overlap_).astype(jnp.int32) + 1
            return ball_x_new + (x_min_ + 2) - (self.consts.BALL_SIZE[0] + 2 - x_max_)

        ball_x_new = jnp.where(
            jnp.any(no_overlap),
            ball_x_new + (x_min - 0) - (self.consts.BALL_SIZE[0] - x_max),
            get_x_when_stuck_in_wall()
        )
        ball_y_new = jnp.where(
            jnp.any(no_overlap),
            ball_y_new + (y_min - 0) - (self.consts.BALL_SIZE[1] - y_max),
            ball_y_new
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
            acceleration_threshold=state.acceleration_threshold,
            acceleration_counter=state.acceleration_counter,
            mod_4_counter=state.mod_4_counter,
            fire_prev=state.fire_prev,
            right_number=state.right_number,
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
        acceleration_counter_new = jnp.where(decelerate, state.acceleration_threshold, acceleration_counter)

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
            acceleration_threshold=acceleration_threshold_new,
            acceleration_counter=acceleration_counter_new,
            mod_4_counter=state.mod_4_counter,
            fire_prev=state.fire_prev,
            right_number=state.right_number,
        )


    def _player_step(self, state: MiniatureGolfState, action: chex.Array) -> MiniatureGolfState:
        # Cardinal vertical / horizontal (+ fire combos).
        up = (action == Action.UP) | (action == Action.UPFIRE)
        down = (action == Action.DOWN) | (action == Action.DOWNFIRE)
        right = (
            (action == Action.RIGHT)
            | (action == Action.RIGHTFIRE)
            | (action == Action.UPRIGHT)
            | (action == Action.UPRIGHTFIRE)
            | (action == Action.DOWNRIGHT)
            | (action == Action.DOWNRIGHTFIRE)
        )
        left = (
            (action == Action.LEFT)
            | (action == Action.LEFTFIRE)
            | (action == Action.UPLEFT)
            | (action == Action.UPLEFTFIRE)
            | (action == Action.DOWNLEFT)
            | (action == Action.DOWNLEFTFIRE)
        )
        # ALE: diagonal inputs only affect the horizontal axis. Opt-in true diagonals via mod.
        up_diag = (
            (action == Action.UPLEFT)
            | (action == Action.UPLEFTFIRE)
            | (action == Action.UPRIGHT)
            | (action == Action.UPRIGHTFIRE)
        )
        down_diag = (
            (action == Action.DOWNLEFT)
            | (action == Action.DOWNLEFTFIRE)
            | (action == Action.DOWNRIGHT)
            | (action == Action.DOWNRIGHTFIRE)
        )
        allow_diag = self.consts.ALLOW_DIAGONAL_MOVEMENT
        up = up | (up_diag & allow_diag)
        down = down | (down_diag & allow_diag)
        # as in the original game, only move every fourth frame when player is close to the ball
        player_close_to_ball = jnp.logical_and(
            jnp.abs(state.ball_x - state.player_x) < 0x15,
            jnp.abs(state.ball_y - state.player_y) < 2 * 0x0b
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
        fire_pressed = self._is_fire_action(action)
        fire = jnp.logical_and(ball_stationary, fire_pressed)
        # only count FIRE if not pressed the previous frame
        fire = jnp.logical_and(fire, jnp.logical_not(state.fire_prev))

        ball_vel_x_new = jnp.where(fire, (state.ball_x - state.player_x) // 4, ball_vel_x_new)
        ball_vel_y_new = jnp.where(fire, (state.ball_y - state.player_y) // 8, ball_vel_y_new)
        ball_stationary_now = jnp.logical_and(jnp.equal(ball_vel_x_new, 0), jnp.equal(ball_vel_y_new, 0))
        fire_had_effect = jnp.logical_and(jnp.logical_and(ball_stationary, fire), jnp.logical_not(ball_stationary_now))
        shot_count_new = jnp.where(fire_had_effect, state.shot_count + 1, state.shot_count)

        v_abs_x = jnp.abs(ball_vel_x_new)
        v_abs_y = jnp.abs(ball_vel_y_new) * 2
        tempo = jnp.where(v_abs_x > v_abs_y, v_abs_x, v_abs_y) + 0x74
        tempo = jnp.where(tempo >= 0x80, 0x100 - tempo, tempo) // 4  # see ROM, arithmetic in two's complement

        acceleration_threshold_new = jnp.where(fire_had_effect, tempo, temporary_state.acceleration_threshold)
        acceleration_counter_new = jnp.where(fire_had_effect, tempo, temporary_state.acceleration_counter)

        player_y_dec = jnp.where(up, 2, 0)
        player_y_inc = jnp.where(down, 2, 0)
        player_x_inc = jnp.where(right, 1, 0)
        player_x_dec = jnp.where(left, 1, 0)
        player_x_without_fire = jnp.clip(state.player_x + player_x_inc - player_x_dec, 1, self.consts.WIDTH - self.consts.PLAYER_SIZE[0] - 1)
        player_y_without_fire = jnp.clip(state.player_y + player_y_inc - player_y_dec, self.consts.PLAYER_MIN_Y, self.consts.PLAYER_MAX_Y)
        player_x_new = jnp.where(fire, state.ball_x, player_x_without_fire)
        player_y_new = jnp.where(fire, state.ball_y - 4, player_y_without_fire)

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
            acceleration_threshold=acceleration_threshold_new,
            acceleration_counter=acceleration_counter_new,
            mod_4_counter=state.mod_4_counter,
            fire_prev=fire_pressed,
            right_number=jnp.where(fire_had_effect, 0, state.right_number),
        )

    def _is_ball_in_hole(self, state: MiniatureGolfState):
        return self._is_overlapping(state.ball_x, state.ball_y, self.consts.BALL_SIZE[0], self.consts.BALL_SIZE[1],
                                    state.hole_x, state.hole_y, self.consts.HOLE_SIZE[0], self.consts.HOLE_SIZE[1])

    def _score_and_reset(self, state: MiniatureGolfState) -> MiniatureGolfState:
        player_goal = self._is_ball_in_hole(state)

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
            self.consts.OBSTACLE_MIN_Y[2],
            self.consts.OBSTACLE_MIN_Y[3],
            self.consts.OBSTACLE_MIN_Y[4],
            self.consts.OBSTACLE_MIN_Y[5],
            self.consts.OBSTACLE_MIN_Y[6],
            self.consts.OBSTACLE_MIN_Y[7] + 221,  # special case
            self.consts.OBSTACLE_MIN_Y[8],
        )
        obstacle_y_new = jnp.where(player_goal, obstacle_y_new, state.obstacle_y)

        obstacle_dir_new = jnp.where(player_goal, 0, state.obstacle_dir)

        right_number_new = jax.lax.select_n(
            level_new,
            self.consts.PAR_VALUES[0],
            self.consts.PAR_VALUES[1],
            self.consts.PAR_VALUES[2],
            self.consts.PAR_VALUES[3],
            self.consts.PAR_VALUES[4],
            self.consts.PAR_VALUES[5],
            self.consts.PAR_VALUES[6],
            self.consts.PAR_VALUES[7],
            self.consts.PAR_VALUES[8],
        )
        right_number_new = jnp.where(player_goal, right_number_new, state.right_number)
        shot_count_new = jnp.where(player_goal, level_new + 1, state.shot_count)

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
            acceleration_threshold=state.acceleration_threshold,
            acceleration_counter=state.acceleration_counter,
            mod_4_counter=jnp.mod(state.mod_4_counter + 1, 4),
            fire_prev=state.fire_prev,
            right_number=right_number_new,
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
        obstacle_y_new = jnp.where(state.level == 7, jnp.mod(state.obstacle_y + 2, 256), obstacle_y_new)
        obstacle_dir_new = jnp.where(state.level == 7, jnp.array(0), obstacle_dir_new)

        # handle ball - obstacle collision
        ball_obstacle_collide = self._is_overlapping(
            state.ball_x, state.ball_y, self.consts.BALL_SIZE[0], self.consts.BALL_SIZE[1],
            state.obstacle_x, state.obstacle_y, self.consts.OBSTACLE_SIZE[0], self.consts.OBSTACLE_SIZE[1],
        )
        ball_stationary = jnp.logical_and(
            jnp.equal(state.ball_vel_x, 0),
            jnp.equal(state.ball_vel_y, 0),
        )
        bounce_horizontally = jnp.greater(
            2 * jnp.abs(state.ball_x + self.consts.BALL_SIZE[0]//2 - obstacle_x_new - self.consts.OBSTACLE_SIZE[0]//2),
            jnp.abs(state.ball_y + self.consts.BALL_SIZE[1]//2 - obstacle_y_new - self.consts.OBSTACLE_SIZE[1]//2),
        )

        ball_vel_x_new = jnp.where(
            ball_obstacle_collide,
            jnp.where(
                ball_stationary,
                jnp.where(
                    obstacle_moves_horizontally,
                    jnp.where(
                        jnp.equal(obstacle_dir_new, 0),
                        40,
                        -40,
                    ),
                    0x0a
                ),
                jnp.where(
                    bounce_horizontally,
                    -state.ball_vel_x,
                    state.ball_vel_x
                )
            ),
            state.ball_vel_x
        )
        ball_vel_y_new = jnp.where(
            ball_obstacle_collide,
            jnp.where(
                ball_stationary,
                jnp.where(
                    obstacle_moves_horizontally,
                    0x0a,
                    jnp.where(
                        jnp.equal(obstacle_dir_new, 0),
                        40,
                        -40,
                    ),
                ),
                jnp.where(
                    bounce_horizontally,
                    state.ball_vel_y,
                    -state.ball_vel_y
                )
            ),
            state.ball_vel_y
        )

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
            obstacle_x=obstacle_x_new,
            obstacle_y=obstacle_y_new,
            obstacle_dir=obstacle_dir_new,
            shot_count=state.shot_count,
            level=state.level,
            acceleration_threshold=state.acceleration_threshold,
            acceleration_counter=state.acceleration_counter,
            mod_4_counter=state.mod_4_counter,
            fire_prev=state.fire_prev,
            right_number=state.right_number,
        )

    def _hole_step(self, state: MiniatureGolfState) -> MiniatureGolfState:
        """Has no effect unless overridden by mods."""
        return state

    def reset(self, key=None) -> Tuple[MiniatureGolfObservation, MiniatureGolfState]:
        level = int(self.consts.START_LEVEL)
        level = max(0, min(level, self.consts.NUM_LEVELS - 1))
        obstacle_y0 = self.consts.OBSTACLE_MIN_Y[level]
        if level == 7:
            obstacle_y0 = self.consts.OBSTACLE_MIN_Y[7] + 221  # matches _score_and_reset special case
        state = MiniatureGolfState(
            player_x=jnp.array(self.consts.PLAYER_START_X[level]).astype(jnp.int32),
            player_y=jnp.array(self.consts.PLAYER_START_Y[level]).astype(jnp.int32),
            ball_x=jnp.array(self.consts.BALL_START_X[level]).astype(jnp.int32),
            ball_y=jnp.array(self.consts.BALL_START_Y[level]).astype(jnp.int32),
            ball_x_subpixel=jnp.array(0),
            ball_y_subpixel=jnp.array(0),
            ball_vel_x=jnp.array(0),
            ball_vel_y=jnp.array(0),
            hole_x=jnp.array(self.consts.HOLE_X[level]).astype(jnp.int32),
            hole_y=jnp.array(self.consts.HOLE_Y[level]).astype(jnp.int32),
            obstacle_x=jnp.array(self.consts.OBSTACLE_MIN_X[level]).astype(jnp.int32),
            obstacle_y=jnp.array(obstacle_y0).astype(jnp.int32),
            obstacle_dir=jnp.array(0),
            shot_count=jnp.array(0),
            level=jnp.array(level),
            acceleration_threshold=jnp.array(0),
            acceleration_counter=jnp.array(0),
            mod_4_counter=jnp.array(0),
            fire_prev=jnp.array(False),
            right_number=jnp.array(self.consts.PAR_VALUES[level]).astype(jnp.int32),
        )
        initial_obs = self._get_observation(state)

        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: MiniatureGolfState, action: chex.Array) -> Tuple[MiniatureGolfObservation, MiniatureGolfState, float, bool, MiniatureGolfInfo]:
        atari_action = jnp.take(self.ACTION_SET, action.astype(jnp.int32))
        previous_state = state
        state = self._ball_step(state)
        state = self._player_step(state, atari_action)
        state = self._obstacle_step(state)
        state = self._hole_step(state)
        state = self._score_and_reset(state)

        done = self._get_done(state)
        env_reward = self._get_reward(previous_state, state)
        info = self._get_info(state)
        observation = self._get_observation(state)

        return observation, state, env_reward, done, info

    def render(self, state: MiniatureGolfState) -> jnp.ndarray:
        return self.renderer.render(state)

    def _get_observation(self, state: MiniatureGolfState):
        player = ObjectObservation.create(
            x=state.player_x,
            y=state.player_y,
            width=jnp.array(self.consts.PLAYER_SIZE[0]),
            height=jnp.array(self.consts.PLAYER_SIZE[1]),
        )

        hole = ObjectObservation.create(
            x=state.hole_x,
            y=state.hole_y,
            width=jnp.array(self.consts.HOLE_SIZE[0]),
            height=jnp.array(self.consts.HOLE_SIZE[1]),
        )

        ball = ObjectObservation.create(
            x=state.ball_x,
            y=state.ball_y,
            width=jnp.array(self.consts.BALL_SIZE[0]),
            height=jnp.array(self.consts.BALL_SIZE[1]),
        )

        obstacle = ObjectObservation.create(
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
        )

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTION_SET))

    def observation_space(self) -> spaces:
        object_space = spaces.get_object_space(n=None, screen_size=(self.consts.HEIGHT, self.consts.WIDTH))

        return spaces.Dict({
            "player": object_space,
            "hole": object_space,
            "ball": object_space,
            "obstacle": object_space,
            "shot_count": spaces.Box(low=0, high=99, shape=(), dtype=jnp.int32),
        })

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.consts.HEIGHT, self.consts.WIDTH, 3),
            dtype=jnp.uint8
        )

    @partial(jax.jit, static_argnums=(0,))
    def _manhattan_reward_potential(self, state: MiniatureGolfState):
        score_mask = self._score_mask_for_level(state.level)
        return state.level * 1e4 + score_mask[state.ball_y, state.ball_x]

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: MiniatureGolfState) -> MiniatureGolfInfo:
        return MiniatureGolfInfo()

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: MiniatureGolfState, state: MiniatureGolfState):
        level_completed = state.level > previous_state.level
        par = jax.lax.select_n(
            previous_state.level,
            self.consts.PAR_VALUES[0],
            self.consts.PAR_VALUES[1],
            self.consts.PAR_VALUES[2],
            self.consts.PAR_VALUES[3],
            self.consts.PAR_VALUES[4],
            self.consts.PAR_VALUES[5],
            self.consts.PAR_VALUES[6],
            self.consts.PAR_VALUES[7],
            self.consts.PAR_VALUES[8],
        )
        reward = (par - previous_state.shot_count).astype(jnp.float32)
        return jnp.where(level_completed, reward, jnp.array(0.0, dtype=jnp.float32))

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: MiniatureGolfState) -> bool:
        return jnp.equal(state.level, self.consts.NUM_LEVELS)

class MiniatureGolfRenderer(JAXGameRenderer):
    def __init__(self, consts: MiniatureGolfConstants = None, config: render_utils.RendererConfig = None):
        super().__init__()
        self.consts = consts or MiniatureGolfConstants()

        if config is None:
            self.config = render_utils.RendererConfig(
                game_dimensions=(self.consts.HEIGHT, self.consts.WIDTH),
                channels=3,
                downscale=None,
            )
        else:
            self.config = config

        self.jr = render_utils.JaxRenderingUtils(self.config)

        # 2. Update asset config to include both walls
        sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/miniature_golf"

        # 3. Make a single call to the setup function
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(self.consts.ASSET_CONFIG, sprite_path)
        self.LEVEL_MASKS = jnp.stack(
            [self.SHAPE_MASKS[f"level_{i}"] for i in range(1, 10)],
            axis=0,
        )

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        level_idx = jnp.clip(state.level, max=self.consts.NUM_LEVELS - 1)
        raster = self.jr.create_object_raster(self.LEVEL_MASKS[level_idx])

        ball_mask = self.SHAPE_MASKS["ball"]
        raster = self.jr.render_at(raster, state.ball_x, state.ball_y, ball_mask)

        hole_mask = self.SHAPE_MASKS["hole"]
        raster = self.jr.render_at(raster, state.hole_x, state.hole_y, hole_mask)

        player_mask = self.SHAPE_MASKS["player"]
        raster = self.jr.render_at(raster, state.player_x, state.player_y, player_mask)

        obstacle_mask = self.SHAPE_MASKS["obstacle"]
        raster = self.jr.render_at_clipped(raster, state.obstacle_x, state.obstacle_y, obstacle_mask)

        # TODO: implement it such that changing the colors in self.consts actually has an effect
        # (i.e. with the ID mapping)

        # Stamp Score using the label utility
        shot_count_to_render = jnp.mod(jnp.clip(state.shot_count, min=1), 100)
        left_digits = self.jr.int_to_digits(shot_count_to_render, max_digits=2)
        right_digits = self.jr.int_to_digits(state.right_number, max_digits=2)

        # Note: The logic for single/double digits is complex for a jitted function.
        left_digit_masks = self.SHAPE_MASKS["left_digits"] # Assumes single color
        right_digit_masks = self.SHAPE_MASKS["right_digits"] # Assumes single color

        left_single_digit = shot_count_to_render < 10
        left_start_index = jax.lax.select(left_single_digit, 1, 0)
        left_num_to_render = jax.lax.select(left_single_digit, 1, 2)
        left_render_x = jax.lax.select(left_single_digit,
                                         self.consts.SCORE_POS_ONES_DIGIT[0],
                                         self.consts.SCORE_POS_TENS_DIGIT[0])
        spacing = self.consts.SCORE_POS_ONES_DIGIT[0] - self.consts.SCORE_POS_TENS_DIGIT[0]

        raster = self.jr.render_label_selective(raster, left_render_x, self.consts.SCORE_POS_ONES_DIGIT[1], left_digits,
                                                left_digit_masks, left_start_index, left_num_to_render, spacing=spacing)

        right_single_digit = state.right_number < 10
        right_start_index = jax.lax.select(right_single_digit, 1, 0)
        right_num_to_render = jax.lax.select(right_single_digit, 1, 2)
        right_render_x = jax.lax.select(right_single_digit,
                                        self.consts.PAR_POS[0],
                                        self.consts.PAR_POS[0] - spacing - self.consts.DIGIT_SIZE[0])

        raster = self.jr.render_label_selective(raster, right_render_x, self.consts.PAR_POS[1], right_digits,
                                                right_digit_masks, right_start_index, right_num_to_render,
                                                spacing=spacing)

        return self.jr.render_from_palette(raster, self.PALETTE)
