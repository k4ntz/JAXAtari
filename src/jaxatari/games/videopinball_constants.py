import jax.numpy as jnp
import chex

from typing import NamedTuple
from flax import struct
from jaxatari.modification import AutoDerivedConstants

@struct.dataclass
class BallMovement:
    old_ball_x: chex.Array
    old_ball_y: chex.Array
    new_ball_x: chex.Array
    new_ball_y: chex.Array


@struct.dataclass
class SceneObject:
    hit_box_width: chex.Array
    hit_box_height: chex.Array
    hit_box_x_offset: chex.Array
    hit_box_y_offset: chex.Array
    reflecting: chex.Array  # 0: no reflection, 1: reflection
    score_type: (
        chex.Array
    )  # 0: no score, 1: Bumper, 2: Spinner, 3: Left Rollover, 4: Atari Rollover, 5: Special Lit Up Target,
    # 6: Left Lit Up Target, 7:Middle Lit Up Target, 8: Right Lit Up Target, 9: Left Flipper, 10: Right Flipper, 11: Tilt Mode Hole Plug
    variant: (
        chex.Array
    )  # a more general property: Used along with score_type to identify the exact SceneObject for a specific game state.


# Todo: Switch to a data class
@struct.dataclass
class HitPoint:
    t_entry: chex.Array
    x: chex.Array
    y: chex.Array


# TODO will have to bring this into VideoPinballConstants or refactor another way.
@struct.dataclass
class HitPointSelector:
    # Hit point properties
    T_ENTRY: int = 0  # time at which the ball intersects with the scene object
    X: int = (
        1  # x position of the point where the ball intersects with the scene object
    )
    Y: int = 2  # y position
    RX: int = 3  # reflected ball x position
    RY: int = 4  # reflected ball y position
    OBJECT_WIDTH: int = 5  # Scene Object properties (of the object that was hit)
    OBJECT_HEIGHT: int = 6
    OBJECT_X: int = 7
    OBJECT_Y: int = 8
    OBJECT_REFLECTING: int = 9
    OBJECT_SCORE_TYPE: int = 10
    OBJECT_VARIANT: int = 11


# immutable state container
@struct.dataclass
class VideoPinballState:
    ball_x: chex.Array
    ball_y: chex.Array
    ball_vel_x: chex.Array
    ball_vel_y: chex.Array
    ball_direction: (
        chex.Array
    )  # 0: left/up, 1:left/down , 2: right/up, 3: right/down (Shouldn't this be a function?)

    left_flipper_angle: chex.Array
    right_flipper_angle: chex.Array
    left_flipper_counter: chex.Array
    right_flipper_counter: chex.Array
    left_flipper_active: chex.Array
    right_flipper_active: chex.Array

    plunger_position: (
        chex.Array
    )  # Value between 0 and 20 where 20 means that the plunger is fully pulled
    plunger_power: (
        chex.Array
    )  # 2 * plunger_position, only set to non-zero value once fired, reset after hitting invisible block

    score: chex.Array
    lives_lost: chex.Array
    bumper_multiplier: chex.Array
    active_targets: (
        chex.Array
    )  # Left diamond, Middle diamond, Right diamond, Special target
    target_cooldown: chex.Array
    special_target_cooldown: chex.Array
    atari_symbols: chex.Array
    rollover_counter: chex.Array
    rollover_enabled: chex.Array

    step_counter: chex.Array
    ball_in_play: chex.Array
    respawn_timer: chex.Array
    color_cycling: chex.Array
    tilt_mode_active: chex.Array
    tilt_counter: chex.Array
    rng_key: chex.Array
    # obs_stack: chex.ArrayTree     What is this for? Pong doesnt have this right?


@struct.dataclass
class EntityState:
    x: chex.Array  # x position of the entity (upper left)
    y: chex.Array  # y position of the entity (upper left)
    w: chex.Array  # width of the entity
    h: chex.Array  # height of the entity
    active: chex.Array  # whether this entity is currently active


@struct.dataclass
class VideoPinballObservation:
    ball: jnp.ndarray
    spinners: jnp.ndarray
    flippers: jnp.ndarray
    plunger: jnp.ndarray
    targets: jnp.ndarray
    bumpers: jnp.ndarray
    rollovers: jnp.ndarray
    tilt_mode_hole_plugs: jnp.ndarray
    score: jnp.ndarray
    lives_lost: jnp.ndarray
    atari_symbols: jnp.ndarray
    bumper_multiplier: jnp.ndarray
    rollover_counter: jnp.ndarray
    color_cycling: jnp.ndarray
    tilt_mode_active: jnp.ndarray


@struct.dataclass
class VideoPinballInfo:
    time: chex.Array
    plunger_power: chex.Array
    target_cooldown: chex.Array
    special_target_cooldown: chex.Array
    rollover_enabled: chex.Array
    step_counter: chex.Array
    ball_in_play: chex.Array
    respawn_timer: chex.Array
    tilt_counter: chex.Array


class VideoPinballConstants(AutoDerivedConstants):
    # Constants for game environment
    WIDTH: int = struct.field(pytree_node=False, default=160)
    HEIGHT: int = struct.field(pytree_node=False, default=210)

    # Physics constants
    GRAVITY: float = struct.field(pytree_node=False, default=0.03)
    VELOCITY_DAMPENING_VALUE: float = struct.field(pytree_node=False, default=0.065)
    VELOCITY_ACCELERATION_VALUE: float = struct.field(pytree_node=False, default=0.15)
    MAX_REFLECTIONS_PER_GAMESTEP: int = struct.field(pytree_node=False, default=3)  # max collisions to process per timestep - this significantly effects game performance
    BALL_MAX_SPEED: float = struct.field(pytree_node=False, default=4)
    BALL_MIN_SPEED: float = struct.field(pytree_node=False, default=0.3)
    NUDGE_EFFECT_INTERVAL: int = struct.field(pytree_node=False, default=2)  # Num steps in between nudge changing
    NUDGE_EFFECT_AMOUNT: float = struct.field(pytree_node=False, default=0.75)  # Amount of nudge effect applied to the ball's velocity
    TILT_COUNT_INCREASE_INTERVAL: int = struct.field(pytree_node=False, default=4)  # Number of steps after which the tilt counter increases
    TILT_COUNT_DECREASE_INTERVAL: int = struct.field(pytree_node=False, default=4)
    TILT_COUNT_TILT_MODE_ACTIVE: int = struct.field(pytree_node=False, default=512)
    FLIPPER_MAX_ANGLE: int = struct.field(pytree_node=False, default=3)
    FLIPPER_ANIMATION_Y_OFFSETS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([0, 0, 3, 7], dtype=jnp.int32))
    FLIPPER_ANIMATION_X_OFFSETS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([0, 0, 0, 1], dtype=jnp.int32))  # Only for the right flipper
    PLUNGER_MAX_POSITION: int = struct.field(pytree_node=False, default=20)

    # Game logic constants
    T_ENTRY_NO_COLLISION: int = struct.field(pytree_node=False, default=9999)
    TARGET_RESPAWN_COOLDOWN: int = struct.field(pytree_node=False, default=16)
    SPECIAL_TARGET_ACTIVE_DURATION: int = struct.field(pytree_node=False, default=257)
    SPECIAL_TARGET_INACTIVE_DURATION: int = struct.field(pytree_node=False, default=787)

    # Game layout constants
    BALL_SIZE: list[int] = struct.field(pytree_node=False, default_factory=lambda: [2, 4])
    FLIPPER_LEFT_POS: list[int] = struct.field(pytree_node=False, default_factory=lambda: [30, 180])
    FLIPPER_RIGHT_POS: list[int] = struct.field(pytree_node=False, default_factory=lambda: [110, 180])
    PLUNGER_POS: list[int] = struct.field(pytree_node=False, default_factory=lambda: [150, 120])
    PLUNGER_MAX_HEIGHT: int = struct.field(pytree_node=False, default=20)  # Taken from RAM values (67-87)
    INVISIBLE_BLOCK_MEAN_REFLECTION_FACTOR: float = struct.field(pytree_node=False, default=0.001)  # 8 times the plunger power is added to ball_vel_x

    # Background color and object colors
    BG_COLOR: list[int] = struct.field(pytree_node=False, default_factory=lambda: [0, 0, 0])
    TILT_MODE_COLOR: list[int] = struct.field(pytree_node=False, default_factory=lambda: [167, 26, 26])
    BACKGROUND_COLOR: list[int] = struct.field(pytree_node=False, default_factory=lambda: [0, 0, 0])
    BOUNDARY_COLOR: list[int] = struct.field(pytree_node=False, default_factory=lambda: [0, 255, 0])
    WALL_COLOR: list[int] = struct.field(pytree_node=False, default_factory=lambda: [104, 72, 198])
    GROUP3_COLOR: list[int] = struct.field(pytree_node=False, default_factory=lambda: [187, 159, 71])
    GROUP4_COLOR: list[int] = struct.field(pytree_node=False, default_factory=lambda: [210, 164, 74])
    GROUP5_COLOR: list[int] = struct.field(pytree_node=False, default_factory=lambda: [236, 236, 236])

    # Color cycling arrays (each inner tuple converted to a list with alpha)
    BACKGROUND_COLOR_CYCLING: list[list[int]] = struct.field(pytree_node=False, default_factory=lambda:
        [
            [74, 74, 74],
            [111, 111, 111],
            [142, 142, 142],
            [170, 170, 170],
            [192, 192, 192],
            [214, 214, 214],
            [236, 236, 236],
            [72, 72, 0],
        ])

    WALL_COLOR_CYCLING: list[list[int]] = struct.field(pytree_node=False, default_factory=lambda:
        [
            [78, 50, 181],
            [51, 26, 163],
            [20, 0, 144],
            [188, 144, 252],
            [169, 128, 240],
            [149, 111, 227],
            [127, 92, 213],
            [146, 70, 192],
        ])

    GROUP3_COLOR_CYCLING: list[list[int]] = struct.field(pytree_node=False, default_factory=lambda:
        [
            [210, 182, 86],
            [232, 204, 99],
            [252, 224, 112],
            [72, 44, 0],
            [105, 77, 20],
            [134, 106, 38],
            [162, 134, 56],
            [160, 171, 79],
        ])

    GROUP4_COLOR_CYCLING: list[list[int]] = struct.field(pytree_node=False, default_factory=lambda:
        [
            [195, 144, 61],
            [236, 200, 96],
            [223, 183, 85],
            [144, 72, 17],
            [124, 44, 0],
            [180, 122, 48],
            [162, 98, 33],
            [227, 151, 89],
        ])

    GROUP5_COLOR_CYCLING: list[list[int]] = struct.field(pytree_node=False, default_factory=lambda:
        [
            [214, 214, 214],
            [192, 192, 192],
            [170, 170, 170],
            [142, 142, 142],
            [111, 111, 111],
            [74, 74, 74],
            [0, 0, 0],
            [252, 252, 84],
        ])

    # Pygame window dimensions
    WINDOW_WIDTH: int = struct.field(pytree_node=False, default=160 * 3)
    WINDOW_HEIGHT: int = struct.field(pytree_node=False, default=210 * 3)

    # Objects (walls etc.) Positions/dimensions
    BALL_START_X: float = struct.field(pytree_node=False, default=149.0)
    BALL_START_Y: float = struct.field(pytree_node=False, default=129.0)
    BALL_START_DIRECTION: int = struct.field(pytree_node=False, default=0)

    GAME_BOTTOM_Y: int = struct.field(pytree_node=False, default=191)

    TOP_WALL_LEFT_X_OFFSET: int = struct.field(pytree_node=False, default=0)
    TOP_WALL_TOP_Y_OFFSET: int = struct.field(pytree_node=False, default=16)
    RIGHT_WALL_LEFT_X_OFFSET: int = struct.field(pytree_node=False, default=152)
    BOTTOM_WALL_LEFT_X_OFFSET: int = struct.field(pytree_node=False, default=12)
    BOTTOM_WALL_TOP_Y_OFFSET: int = struct.field(pytree_node=False, default=184)

    INVISIBLE_BLOCK_LEFT_X_OFFSET: int = struct.field(pytree_node=False, default=129)
    INVISIBLE_BLOCK_TOP_Y_OFFSET: int = struct.field(pytree_node=False, default=36)

    INNER_WALL_TOP_Y_OFFSET: int = struct.field(pytree_node=False, default=56)
    LEFT_INNER_WALL_TOP_X_OFFSET: int = struct.field(pytree_node=False, default=12)
    RIGHT_INNER_WALL_TOP_X_OFFSET: int = struct.field(pytree_node=False, default=144)

    QUADRUPLE_STEP_Y_OFFSET: int = struct.field(pytree_node=False, default=152)
    TRIPLE_STEP_Y_OFFSET: int = struct.field(pytree_node=False, default=160)
    DOUBLE_STEP_Y_OFFSET: int = struct.field(pytree_node=False, default=168)
    SINGLE_STEP_Y_OFFSET: int = struct.field(pytree_node=False, default=176)

    LEFT_QUADRUPLE_STEP_X_OFFSET: int = struct.field(pytree_node=False, default=16)
    RIGHT_QUADRUPLE_STEP_X_OFFSET: int = struct.field(pytree_node=False, default=140)
    LEFT_TRIPLE_STEP_X_OFFSET: int = struct.field(pytree_node=False, default=20)
    RIGHT_TRIPLE_STEP_X_OFFSET: int = struct.field(pytree_node=False, default=136)
    LEFT_DOUBLE_STEP_X_OFFSET: int = struct.field(pytree_node=False, default=24)
    RIGHT_DOUBLE_STEP_X_OFFSET: int = struct.field(pytree_node=False, default=132)
    LEFT_SINGLE_STEP_X_OFFSET: int = struct.field(pytree_node=False, default=28)
    RIGHT_SINGLE_STEP_X_OFFSET: int = struct.field(pytree_node=False, default=128)

    OUTER_WALL_THICKNESS: int = struct.field(pytree_node=False, default=8)
    INNER_WALL_THICKNESS: int = struct.field(pytree_node=False, default=4)
    WALL_CORNER_BLOCK_WIDTH: int = struct.field(pytree_node=False, default=4)
    WALL_CORNER_BLOCK_HEIGHT: int = struct.field(pytree_node=False, default=8)
    STEP_HEIGHT: int = struct.field(pytree_node=False, default=8)
    STEP_WIDTH: int = struct.field(pytree_node=False, default=4)

    # Inner Objects Positions and Dimensions

    VERTICAL_BAR_HEIGHT: int = struct.field(pytree_node=False, default=32)
    VERTICAL_BAR_WIDTH: int = struct.field(pytree_node=False, default=4)

    ROLLOVER_BAR_DISTANCE: int = struct.field(pytree_node=False, default=12)  # Distance between the left and right rollover bar

    BUMPER_WIDTH: int = struct.field(pytree_node=False, default=16)
    BUMPER_HEIGHT: int = struct.field(pytree_node=False, default=32)

    LEFT_COLUMN_X_OFFSET: int = struct.field(pytree_node=False, default=40)
    MIDDLE_COLUMN_X_OFFSET: int = struct.field(pytree_node=False, default=72)
    RIGHT_COLUMN_X_OFFSET: int = struct.field(pytree_node=False, default=104)
    TOP_ROW_Y_OFFSET: int = struct.field(pytree_node=False, default=48)
    MIDDLE_ROW_Y_OFFSET: int = struct.field(pytree_node=False, default=112)
    BOTTOM_ROW_Y_OFFSET: int = struct.field(pytree_node=False, default=177)

    MIDDLE_BAR_X: int = struct.field(pytree_node=False, default=72)
    MIDDLE_BAR_Y: int = struct.field(pytree_node=False, default=104)
    MIDDLE_BAR_WIDTH: int = struct.field(pytree_node=False, default=16)
    MIDDLE_BAR_HEIGHT: int = struct.field(pytree_node=False, default=8)

    # Flipper Parts Positions and Dimensions

    # Flipper bounding boxes - these are a special case as they are not used for AABB collision,
    # but for their own special kind of collision
    FLIPPER_LEFT_PIVOT_X: int = struct.field(pytree_node=False, default=64)
    FLIPPER_RIGHT_PIVOT_X: int = struct.field(pytree_node=False, default=96)  # pixel + 1 to not leave a gap
    FLIPPER_PIVOT_Y_TOP: int = struct.field(pytree_node=False, default=184)
    FLIPPER_PIVOT_Y_BOT: int = struct.field(pytree_node=False, default=190)  # pixel + 1

    FLIPPER_00_TOP_END_Y: int = struct.field(pytree_node=False, default=190)
    FLIPPER_00_BOT_END_Y: int = struct.field(pytree_node=False, default=192)  # pixel + 1
    FLIPPER_16_TOP_END_Y: int = struct.field(pytree_node=False, default=186)
    FLIPPER_16_BOT_END_Y: int = struct.field(pytree_node=False, default=188)  # pixel + 1
    FLIPPER_32_TOP_END_Y: int = struct.field(pytree_node=False, default=181)
    FLIPPER_32_BOT_END_Y: int = struct.field(pytree_node=False, default=183)  # pixel + 1
    FLIPPER_48_TOP_END_Y: int = struct.field(pytree_node=False, default=177)
    FLIPPER_48_BOT_END_Y: int = struct.field(pytree_node=False, default=179)  # pixel + 1

    FLIPPER_LEFT_00_END_X: int = struct.field(pytree_node=False, default=77)  # pixel + 1
    FLIPPER_LEFT_16_END_X: int = struct.field(pytree_node=False, default=77)  # pixel + 1
    FLIPPER_LEFT_32_END_X: int = struct.field(pytree_node=False, default=77)  # pixel + 1
    FLIPPER_LEFT_48_END_X: int = struct.field(pytree_node=False, default=76)  # pixel + 1
    FLIPPER_RIGHT_00_END_X: int = struct.field(pytree_node=False, default=83)
    FLIPPER_RIGHT_16_END_X: int = struct.field(pytree_node=False, default=83)
    FLIPPER_RIGHT_32_END_X: int = struct.field(pytree_node=False, default=83)
    FLIPPER_RIGHT_48_END_X: int = struct.field(pytree_node=False, default=84)

    # Spinner Bounding Boxes

    # Bounding boxes of the spinner if the joined (single larger) spinner part is on the top or bottom
    SPINNER_TOP_BOTTOM_LARGE_HEIGHT: int = struct.field(pytree_node=False, default=2)
    SPINNER_TOP_BOTTOM_LARGE_WIDTH: int = struct.field(pytree_node=False, default=2)
    SPINNER_TOP_BOTTOM_SMALL_HEIGHT: int = struct.field(pytree_node=False, default=2)
    SPINNER_TOP_BOTTOM_SMALL_WIDTH: int = struct.field(pytree_node=False, default=1)

    # Bounding boxes of the spinner if the joined (single larger) spinner part is on the left or right
    SPINNER_LEFT_RIGHT_LARGE_HEIGHT: int = struct.field(pytree_node=False, default=2)
    SPINNER_LEFT_RIGHT_LARGE_WIDTH: int = struct.field(pytree_node=False, default=1)
    SPINNER_LEFT_RIGHT_SMALL_HEIGHT: int = struct.field(pytree_node=False, default=2)
    SPINNER_LEFT_RIGHT_SMALL_WIDTH: int = struct.field(pytree_node=False, default=1)

    LEFT_SPINNER_MIDDLE_POINT: tuple[int, int] = struct.field(pytree_node=False, default=(33, 97))
    RIGHT_SPINNER_MIDDLE_POINT: tuple[int, int] = struct.field(pytree_node=False, default=(129, 97))

    # SceneObjects that depend on base constants - computed in compute_derived()
    INVISIBLE_BLOCK_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    TOP_WALL_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    BOTTOM_WALL_SCENE_OBJECT_1: SceneObject = struct.field(pytree_node=False, default=None)
    BOTTOM_WALL_SCENE_OBJECT_2: SceneObject = struct.field(pytree_node=False, default=None)
    BOTTOM_WALL_SCENE_OBJECT_3: SceneObject = struct.field(pytree_node=False, default=None)
    BOTTOM_WALL_SCENE_OBJECT_4: SceneObject = struct.field(pytree_node=False, default=None)
    TILT_MODE_HOLE_PLUG_LEFT: SceneObject = struct.field(pytree_node=False, default=None)
    TILT_MODE_HOLE_PLUG_RIGHT: SceneObject = struct.field(pytree_node=False, default=None)
    LEFT_WALL_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    RIGHT_WALL_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    LEFT_INNER_WALL_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    RIGHT_INNER_WALL_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    LEFT_QUADRUPLE_STEP_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    RIGHT_QUADRUPLE_STEP_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    LEFT_TRIPLE_STEP_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    RIGHT_TRIPLE_STEP_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    LEFT_DOUBLE_STEP_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    RIGHT_DOUBLE_STEP_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    LEFT_SINGLE_STEP_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    RIGHT_SINGLE_STEP_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    TOP_LEFT_STEP_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    TOP_RIGHT_STEP_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)

    # Rollover Bars Scene Objects

    LEFT_ROLLOVER_LEFT_BAR_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    LEFT_ROLLOVER_RIGHT_BAR_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    ATARI_ROLLOVER_LEFT_BAR_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    ATARI_ROLLOVER_RIGHT_BAR_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)

    # Bumper Scene Objects

    TOP_BUMPER_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    LEFT_BUMPER_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    RIGHT_BUMPER_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)

    # Left Spinner Scene Objects

    # Left Spinner Bottom Position

    LEFT_SPINNER_BOTTOM_POSITION_JOINED_PART_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    LEFT_SPINNER_BOTTOM_POSITION_LEFT_PART_1_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    LEFT_SPINNER_BOTTOM_POSITION_LEFT_PART_2_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    LEFT_SPINNER_BOTTOM_POSITION_RIGHT_PART_1_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    LEFT_SPINNER_BOTTOM_POSITION_RIGHT_PART_2_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)

    # Left Spinner Top Position
    LEFT_SPINNER_TOP_POSITION_JOINED_PART_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    LEFT_SPINNER_TOP_POSITION_LEFT_PART_1_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    LEFT_SPINNER_TOP_POSITION_LEFT_PART_2_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    LEFT_SPINNER_TOP_POSITION_RIGHT_PART_1_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    LEFT_SPINNER_TOP_POSITION_RIGHT_PART_2_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)

    # Left Spinner Left Position
    LEFT_SPINNER_LEFT_POSITION_JOINED_PART_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    LEFT_SPINNER_LEFT_POSITION_LEFT_PART_1_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    LEFT_SPINNER_LEFT_POSITION_LEFT_PART_2_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    LEFT_SPINNER_LEFT_POSITION_RIGHT_PART_1_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    LEFT_SPINNER_LEFT_POSITION_RIGHT_PART_2_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)

    # Left Spinner Right Position
    LEFT_SPINNER_RIGHT_POSITION_JOINED_PART_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    LEFT_SPINNER_RIGHT_POSITION_LEFT_PART_1_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    LEFT_SPINNER_RIGHT_POSITION_LEFT_PART_2_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    LEFT_SPINNER_RIGHT_POSITION_RIGHT_PART_1_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    LEFT_SPINNER_RIGHT_POSITION_RIGHT_PART_2_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)

    # Right Spinner Scene Objects

    # Right Spinner Bottom Position
    RIGHT_SPINNER_BOTTOM_POSITION_JOINED_PART_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    RIGHT_SPINNER_BOTTOM_POSITION_LEFT_PART_1_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    RIGHT_SPINNER_BOTTOM_POSITION_LEFT_PART_2_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    RIGHT_SPINNER_BOTTOM_POSITION_RIGHT_PART_1_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    RIGHT_SPINNER_BOTTOM_POSITION_RIGHT_PART_2_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)

    # Right Spinner Top Position
    RIGHT_SPINNER_TOP_POSITION_JOINED_PART_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    RIGHT_SPINNER_TOP_POSITION_LEFT_PART_1_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    RIGHT_SPINNER_TOP_POSITION_LEFT_PART_2_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    RIGHT_SPINNER_TOP_POSITION_RIGHT_PART_1_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    RIGHT_SPINNER_TOP_POSITION_RIGHT_PART_2_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)

    # Right Spinner Left Position
    RIGHT_SPINNER_LEFT_POSITION_JOINED_PART_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    RIGHT_SPINNER_LEFT_POSITION_LEFT_PART_1_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    RIGHT_SPINNER_LEFT_POSITION_LEFT_PART_2_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    RIGHT_SPINNER_LEFT_POSITION_RIGHT_PART_1_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    RIGHT_SPINNER_LEFT_POSITION_RIGHT_PART_2_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)

    # Right Spinner Right Position
    RIGHT_SPINNER_RIGHT_POSITION_JOINED_PART_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    RIGHT_SPINNER_RIGHT_POSITION_LEFT_PART_1_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    RIGHT_SPINNER_RIGHT_POSITION_LEFT_PART_2_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    RIGHT_SPINNER_RIGHT_POSITION_RIGHT_PART_1_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    RIGHT_SPINNER_RIGHT_POSITION_RIGHT_PART_2_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)

    DIAMOND_VERTICAL_RECTANGLE_BOUNDING_BOX_HEIGHT: chex.Array = struct.field(pytree_node=False, default=None)
    DIAMOND_VERTICAL_RECTANGLE_BOUNDING_BOX_WIDTH: chex.Array = struct.field(pytree_node=False, default=None)
    DIAMOND_HORIZONTAL_RECTANGLE_BOUNDING_BOX_HEIGHT: chex.Array = struct.field(pytree_node=False, default=None)
    DIAMOND_HORIZONTAL_RECTANGLE_BOUNDING_BOX_WIDTH: chex.Array = struct.field(pytree_node=False, default=None)
    DIAMOND_SMALL_RECTANGLE_BOUNDING_BOX_HEIGHT: chex.Array = struct.field(pytree_node=False, default=None)
    DIAMOND_SMALL_RECTANGLE_BOUNDING_BOX_WIDTH: chex.Array = struct.field(pytree_node=False, default=None)

    LEFT_LIT_UP_TARGET_LARGE_VERTICAL_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    LEFT_LIT_UP_TARGET_LARGE_HORIZONTAL_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    LEFT_LIT_UP_TARGET_SMALL_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    MIDDLE_LIT_UP_TARGET_LARGE_VERTICAL_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    MIDDLE_LIT_UP_TARGET_LARGE_HORIZONTAL_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    MIDDLE_LIT_UP_TARGET_SMALL_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    RIGHT_LIT_UP_TARGET_LARGE_VERTICAL_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    RIGHT_LIT_UP_TARGET_LARGE_HORIZONTAL_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    RIGHT_LIT_UP_TARGET_SMALL_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    SPECIAL_LIT_UP_TARGET_LARGE_VERTICAL_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    SPECIAL_LIT_UP_TARGET_LARGE_HORIZONTAL_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    LEFT_ROLLOVER_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    ATARI_ROLLOVER_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    MIDDLE_BAR_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)

    LEFT_FLIPPER_00_BOT_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    LEFT_FLIPPER_16_BOT_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    LEFT_FLIPPER_32_BOT_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    LEFT_FLIPPER_48_BOT_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    LEFT_FLIPPER_00_TOP_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    LEFT_FLIPPER_16_TOP_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    LEFT_FLIPPER_32_TOP_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    LEFT_FLIPPER_48_TOP_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    RIGHT_FLIPPER_00_BOT_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    RIGHT_FLIPPER_16_BOT_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    RIGHT_FLIPPER_32_BOT_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    RIGHT_FLIPPER_48_BOT_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    RIGHT_FLIPPER_00_TOP_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    RIGHT_FLIPPER_16_TOP_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    RIGHT_FLIPPER_32_TOP_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)
    RIGHT_FLIPPER_48_TOP_SCENE_OBJECT: SceneObject = struct.field(pytree_node=False, default=None)

    ALL_SCENE_OBJECTS_LIST: list = struct.field(pytree_node=False, default=None)
    REFLECTING_SCENE_OBJECTS: chex.Array = struct.field(pytree_node=False, default=None)
    NON_REFLECTING_SCENE_OBJECTS: chex.Array = struct.field(pytree_node=False, default=None)
    TOTAL_SCORE_TYPES: chex.Array = struct.field(pytree_node=False, default=None)
    _FLIPPERS_SORTED: tuple = struct.field(pytree_node=False, default=None)
    FLIPPERS: chex.Array = struct.field(pytree_node=False, default=None)
    FLIPPER_SEGMENTS_SORTED: chex.Array = struct.field(pytree_node=False, default=None)

    def compute_derived(self) -> dict:
        """Compute all SceneObjects and derived arrays from base constants."""
        # Diamond constants
        diamond_vertical_height = jnp.array(10)
        diamond_vertical_width = jnp.array(3)
        diamond_horizontal_height = jnp.array(6)
        diamond_horizontal_width = jnp.array(5)
        diamond_small_height = jnp.array(2)
        diamond_small_width = jnp.array(1)

        # Compute wall and step SceneObjects that depend on base constants
        invisible_block = SceneObject(
            hit_box_height=jnp.array(2),
            hit_box_width=jnp.array(self.INNER_WALL_THICKNESS + 20),
            hit_box_x_offset=jnp.array(self.INVISIBLE_BLOCK_LEFT_X_OFFSET),
            hit_box_y_offset=jnp.array(self.INVISIBLE_BLOCK_TOP_Y_OFFSET),
            reflecting=jnp.array(1),
            score_type=jnp.array(0),
            variant=jnp.array(-1),
        )

        top_wall = SceneObject(
            hit_box_height=jnp.array(self.OUTER_WALL_THICKNESS),
            hit_box_width=jnp.array(160),
            hit_box_x_offset=jnp.array(self.TOP_WALL_LEFT_X_OFFSET),
            hit_box_y_offset=jnp.array(self.TOP_WALL_TOP_Y_OFFSET),
            reflecting=jnp.array(1),
            score_type=jnp.array(0),
            variant=jnp.array(-1),
        )
        bottom_wall_1 = SceneObject(
            hit_box_height=jnp.array(self.OUTER_WALL_THICKNESS),
            hit_box_width=jnp.array(24),
            hit_box_x_offset=jnp.array(self.BOTTOM_WALL_LEFT_X_OFFSET),
            hit_box_y_offset=jnp.array(self.BOTTOM_WALL_TOP_Y_OFFSET),
            reflecting=jnp.array(1),
            score_type=jnp.array(0),
            variant=jnp.array(-1),
        )
        bottom_wall_2 = SceneObject(
            hit_box_height=jnp.array(self.OUTER_WALL_THICKNESS),
            hit_box_width=jnp.array(24),
            hit_box_x_offset=jnp.array(40),
            hit_box_y_offset=jnp.array(self.BOTTOM_WALL_TOP_Y_OFFSET),
            reflecting=jnp.array(1),
            score_type=jnp.array(0),
            variant=jnp.array(-1),
        )
        bottom_wall_3 = SceneObject(
            hit_box_height=jnp.array(self.OUTER_WALL_THICKNESS),
            hit_box_width=jnp.array(24),
            hit_box_x_offset=jnp.array(96),
            hit_box_y_offset=jnp.array(self.BOTTOM_WALL_TOP_Y_OFFSET),
            reflecting=jnp.array(1),
            score_type=jnp.array(0),
            variant=jnp.array(-1),
        )
        bottom_wall_4 = SceneObject(
            hit_box_height=jnp.array(self.OUTER_WALL_THICKNESS),
            hit_box_width=jnp.array(24),
            hit_box_x_offset=jnp.array(124),
            hit_box_y_offset=jnp.array(self.BOTTOM_WALL_TOP_Y_OFFSET),
            reflecting=jnp.array(1),
            score_type=jnp.array(0),
            variant=jnp.array(-1),
        )

        tilt_mode_hole_plug_left = SceneObject(
            hit_box_height=jnp.array(self.OUTER_WALL_THICKNESS),
            hit_box_width=jnp.array(4),
            hit_box_x_offset=jnp.array(36),
            hit_box_y_offset=jnp.array(self.BOTTOM_WALL_TOP_Y_OFFSET),
            reflecting=jnp.array(1),
            score_type=jnp.array(11),
            variant=jnp.array(-1),
        )
        tilt_mode_hole_plug_right = SceneObject(
            hit_box_height=jnp.array(self.OUTER_WALL_THICKNESS),
            hit_box_width=jnp.array(4),
            hit_box_x_offset=jnp.array(120),
            hit_box_y_offset=jnp.array(self.BOTTOM_WALL_TOP_Y_OFFSET),
            reflecting=jnp.array(1),
            score_type=jnp.array(11),
            variant=jnp.array(-1),
        )

        left_wall = SceneObject(
            hit_box_height=jnp.array(176),
            hit_box_width=jnp.array(self.OUTER_WALL_THICKNESS),
            hit_box_x_offset=jnp.array(self.TOP_WALL_LEFT_X_OFFSET),
            hit_box_y_offset=jnp.array(self.TOP_WALL_TOP_Y_OFFSET),
            reflecting=jnp.array(1),
            score_type=jnp.array(0),
            variant=jnp.array(-1),
        )
        right_wall = SceneObject(
            hit_box_height=jnp.array(176),
            hit_box_width=jnp.array(self.OUTER_WALL_THICKNESS),
            hit_box_x_offset=jnp.array(self.RIGHT_WALL_LEFT_X_OFFSET),
            hit_box_y_offset=jnp.array(self.TOP_WALL_TOP_Y_OFFSET),
            reflecting=jnp.array(1),
            score_type=jnp.array(0),
            variant=jnp.array(-1),
        )

        left_inner_wall = SceneObject(
            hit_box_height=jnp.array(135),
            hit_box_width=jnp.array(self.INNER_WALL_THICKNESS),
            hit_box_x_offset=jnp.array(self.LEFT_INNER_WALL_TOP_X_OFFSET),
            hit_box_y_offset=jnp.array(self.INNER_WALL_TOP_Y_OFFSET),
            reflecting=jnp.array(1),
            score_type=jnp.array(0),
            variant=jnp.array(-1),
        )
        right_inner_wall = SceneObject(
            hit_box_height=jnp.array(135),
            hit_box_width=jnp.array(self.INNER_WALL_THICKNESS),
            hit_box_x_offset=jnp.array(self.RIGHT_INNER_WALL_TOP_X_OFFSET),
            hit_box_y_offset=jnp.array(self.INNER_WALL_TOP_Y_OFFSET),
            reflecting=jnp.array(1),
            score_type=jnp.array(0),
            variant=jnp.array(-1),
        )

        left_quadruple_step = SceneObject(
            hit_box_height=jnp.array(4 * self.STEP_HEIGHT),
            hit_box_width=jnp.array(self.STEP_WIDTH),
            hit_box_x_offset=jnp.array(self.LEFT_QUADRUPLE_STEP_X_OFFSET),
            hit_box_y_offset=jnp.array(self.QUADRUPLE_STEP_Y_OFFSET),
            reflecting=jnp.array(1),
            score_type=jnp.array(0),
            variant=jnp.array(-1),
        )
        right_quadruple_step = SceneObject(
            hit_box_height=jnp.array(4 * self.STEP_HEIGHT),
            hit_box_width=jnp.array(self.STEP_WIDTH),
            hit_box_x_offset=jnp.array(self.RIGHT_QUADRUPLE_STEP_X_OFFSET),
            hit_box_y_offset=jnp.array(self.QUADRUPLE_STEP_Y_OFFSET),
            reflecting=jnp.array(1),
            score_type=jnp.array(0),
            variant=jnp.array(-1),
        )
        left_triple_step = SceneObject(
            hit_box_height=jnp.array(3 * self.STEP_HEIGHT),
            hit_box_width=jnp.array(self.STEP_WIDTH),
            hit_box_x_offset=jnp.array(self.LEFT_TRIPLE_STEP_X_OFFSET),
            hit_box_y_offset=jnp.array(self.TRIPLE_STEP_Y_OFFSET),
            reflecting=jnp.array(1),
            score_type=jnp.array(0),
            variant=jnp.array(-1),
        )
        right_triple_step = SceneObject(
            hit_box_height=jnp.array(3 * self.STEP_HEIGHT),
            hit_box_width=jnp.array(self.STEP_WIDTH),
            hit_box_x_offset=jnp.array(self.RIGHT_TRIPLE_STEP_X_OFFSET),
            hit_box_y_offset=jnp.array(self.TRIPLE_STEP_Y_OFFSET),
            reflecting=jnp.array(1),
            score_type=jnp.array(0),
            variant=jnp.array(-1),
        )
        left_double_step = SceneObject(
            hit_box_height=jnp.array(2 * self.STEP_HEIGHT),
            hit_box_width=jnp.array(self.STEP_WIDTH),
            hit_box_x_offset=jnp.array(self.LEFT_DOUBLE_STEP_X_OFFSET),
            hit_box_y_offset=jnp.array(self.DOUBLE_STEP_Y_OFFSET),
            reflecting=jnp.array(1),
            score_type=jnp.array(0),
            variant=jnp.array(-1),
        )
        right_double_step = SceneObject(
            hit_box_height=jnp.array(2 * self.STEP_HEIGHT),
            hit_box_width=jnp.array(self.STEP_WIDTH),
            hit_box_x_offset=jnp.array(self.RIGHT_DOUBLE_STEP_X_OFFSET),
            hit_box_y_offset=jnp.array(self.DOUBLE_STEP_Y_OFFSET),
            reflecting=jnp.array(1),
            score_type=jnp.array(0),
            variant=jnp.array(-1),
        )
        left_single_step = SceneObject(
            hit_box_height=jnp.array(1 * self.STEP_HEIGHT),
            hit_box_width=jnp.array(self.STEP_WIDTH),
            hit_box_x_offset=jnp.array(self.LEFT_SINGLE_STEP_X_OFFSET),
            hit_box_y_offset=jnp.array(self.SINGLE_STEP_Y_OFFSET),
            reflecting=jnp.array(1),
            score_type=jnp.array(0),
            variant=jnp.array(-1),
        )

        # Compute all SceneObjects
        right_single_step = SceneObject(
            hit_box_height=jnp.array(1 * self.STEP_HEIGHT),
            hit_box_width=jnp.array(self.STEP_WIDTH),
            hit_box_x_offset=jnp.array(self.RIGHT_SINGLE_STEP_X_OFFSET),
            hit_box_y_offset=jnp.array(self.SINGLE_STEP_Y_OFFSET),
            reflecting=jnp.array(1),
            score_type=jnp.array(0),
            variant=jnp.array(-1),
        )
        top_left_step = SceneObject(
            hit_box_height=jnp.array(1 * self.STEP_HEIGHT),
            hit_box_width=jnp.array(1 * self.STEP_WIDTH),
            hit_box_x_offset=jnp.array(8),
            hit_box_y_offset=jnp.array(24),
            reflecting=jnp.array(1),
            score_type=jnp.array(0),
            variant=jnp.array(-1),
        )
        top_right_step = SceneObject(
            hit_box_height=jnp.array(1 * self.STEP_HEIGHT),
            hit_box_width=jnp.array(1 * self.STEP_WIDTH),
            hit_box_x_offset=jnp.array(148),
            hit_box_y_offset=jnp.array(24),
            reflecting=jnp.array(1),
            score_type=jnp.array(0),
            variant=jnp.array(-1),
        )

        left_rollover_left_bar = SceneObject(
            hit_box_height=jnp.array(self.VERTICAL_BAR_HEIGHT),
            hit_box_width=jnp.array(self.VERTICAL_BAR_WIDTH),
            hit_box_x_offset=jnp.array(self.LEFT_COLUMN_X_OFFSET),
            hit_box_y_offset=jnp.array(self.TOP_ROW_Y_OFFSET),
            reflecting=jnp.array(1),
            score_type=jnp.array(0),
            variant=jnp.array(-1),
        )
        left_rollover_right_bar = SceneObject(
            hit_box_height=jnp.array(self.VERTICAL_BAR_HEIGHT),
            hit_box_width=jnp.array(self.VERTICAL_BAR_WIDTH),
            hit_box_x_offset=jnp.array(self.LEFT_COLUMN_X_OFFSET + self.ROLLOVER_BAR_DISTANCE),
            hit_box_y_offset=jnp.array(self.TOP_ROW_Y_OFFSET),
            reflecting=jnp.array(1),
            score_type=jnp.array(0),
            variant=jnp.array(-1),
        )
        atari_rollover_left_bar = SceneObject(
            hit_box_height=jnp.array(self.VERTICAL_BAR_HEIGHT),
            hit_box_width=jnp.array(self.VERTICAL_BAR_WIDTH),
            hit_box_x_offset=jnp.array(self.RIGHT_COLUMN_X_OFFSET),
            hit_box_y_offset=jnp.array(self.TOP_ROW_Y_OFFSET),
            reflecting=jnp.array(1),
            score_type=jnp.array(0),
            variant=jnp.array(-1),
        )
        atari_rollover_right_bar = SceneObject(
            hit_box_height=jnp.array(self.VERTICAL_BAR_HEIGHT),
            hit_box_width=jnp.array(self.VERTICAL_BAR_WIDTH),
            hit_box_x_offset=jnp.array(self.RIGHT_COLUMN_X_OFFSET + self.ROLLOVER_BAR_DISTANCE),
            hit_box_y_offset=jnp.array(self.TOP_ROW_Y_OFFSET),
            reflecting=jnp.array(1),
            score_type=jnp.array(0),
            variant=jnp.array(-1),
        )

        top_bumper = SceneObject(
            hit_box_height=jnp.array(self.BUMPER_HEIGHT),
            hit_box_width=jnp.array(self.BUMPER_WIDTH),
            hit_box_x_offset=jnp.array(self.MIDDLE_COLUMN_X_OFFSET),
            hit_box_y_offset=jnp.array(self.TOP_ROW_Y_OFFSET),
            reflecting=jnp.array(1),
            score_type=jnp.array(1),
            variant=jnp.array(-1),
        )
        left_bumper = SceneObject(
            hit_box_height=jnp.array(self.BUMPER_HEIGHT),
            hit_box_width=jnp.array(self.BUMPER_WIDTH),
            hit_box_x_offset=jnp.array(self.LEFT_COLUMN_X_OFFSET),
            hit_box_y_offset=jnp.array(self.MIDDLE_ROW_Y_OFFSET),
            reflecting=jnp.array(1),
            score_type=jnp.array(1),
            variant=jnp.array(-1),
        )
        right_bumper = SceneObject(
            hit_box_height=jnp.array(self.BUMPER_HEIGHT),
            hit_box_width=jnp.array(self.BUMPER_WIDTH),
            hit_box_x_offset=jnp.array(self.RIGHT_COLUMN_X_OFFSET),
            hit_box_y_offset=jnp.array(self.MIDDLE_ROW_Y_OFFSET),
            reflecting=jnp.array(1),
            score_type=jnp.array(1),
            variant=jnp.array(-1),
        )

        # Left Spinner Scene Objects
        left_spinner_bottom_joined = SceneObject(
            hit_box_height=jnp.array(self.SPINNER_TOP_BOTTOM_LARGE_HEIGHT),
            hit_box_width=jnp.array(self.SPINNER_TOP_BOTTOM_LARGE_WIDTH),
            hit_box_x_offset=jnp.array(32),
            hit_box_y_offset=jnp.array(100),
            reflecting=jnp.array(1),
            score_type=jnp.array(2),
            variant=jnp.array(0),
        )
        left_spinner_bottom_left_1 = SceneObject(
            hit_box_height=jnp.array(self.SPINNER_TOP_BOTTOM_SMALL_HEIGHT),
            hit_box_width=jnp.array(self.SPINNER_TOP_BOTTOM_SMALL_WIDTH),
            hit_box_x_offset=jnp.array(30),
            hit_box_y_offset=jnp.array(94),
            reflecting=jnp.array(1),
            score_type=jnp.array(2),
            variant=jnp.array(0),
        )
        left_spinner_bottom_left_2 = SceneObject(
            hit_box_height=jnp.array(self.SPINNER_TOP_BOTTOM_SMALL_HEIGHT),
            hit_box_width=jnp.array(self.SPINNER_TOP_BOTTOM_SMALL_WIDTH),
            hit_box_x_offset=jnp.array(31),
            hit_box_y_offset=jnp.array(92),
            reflecting=jnp.array(1),
            score_type=jnp.array(2),
            variant=jnp.array(0),
        )
        left_spinner_bottom_right_1 = SceneObject(
            hit_box_height=jnp.array(self.SPINNER_TOP_BOTTOM_SMALL_HEIGHT),
            hit_box_width=jnp.array(self.SPINNER_TOP_BOTTOM_SMALL_WIDTH),
            hit_box_x_offset=jnp.array(35),
            hit_box_y_offset=jnp.array(94),
            reflecting=jnp.array(1),
            score_type=jnp.array(2),
            variant=jnp.array(0),
        )
        left_spinner_bottom_right_2 = SceneObject(
            hit_box_height=jnp.array(self.SPINNER_TOP_BOTTOM_SMALL_HEIGHT),
            hit_box_width=jnp.array(self.SPINNER_TOP_BOTTOM_SMALL_WIDTH),
            hit_box_x_offset=jnp.array(34),
            hit_box_y_offset=jnp.array(92),
            reflecting=jnp.array(1),
            score_type=jnp.array(2),
            variant=jnp.array(0),
        )

        left_spinner_top_joined = SceneObject(
            hit_box_height=jnp.array(self.SPINNER_TOP_BOTTOM_LARGE_HEIGHT),
            hit_box_width=jnp.array(self.SPINNER_TOP_BOTTOM_LARGE_WIDTH),
            hit_box_x_offset=jnp.array(32),
            hit_box_y_offset=jnp.array(90),
            reflecting=jnp.array(1),
            score_type=jnp.array(2),
            variant=jnp.array(1),
        )
        left_spinner_top_left_1 = SceneObject(
            hit_box_height=jnp.array(self.SPINNER_TOP_BOTTOM_SMALL_HEIGHT),
            hit_box_width=jnp.array(self.SPINNER_TOP_BOTTOM_SMALL_WIDTH),
            hit_box_x_offset=jnp.array(30),
            hit_box_y_offset=jnp.array(96),
            reflecting=jnp.array(1),
            score_type=jnp.array(2),
            variant=jnp.array(1),
        )
        left_spinner_top_left_2 = SceneObject(
            hit_box_height=jnp.array(self.SPINNER_TOP_BOTTOM_SMALL_HEIGHT),
            hit_box_width=jnp.array(self.SPINNER_TOP_BOTTOM_SMALL_WIDTH),
            hit_box_x_offset=jnp.array(31),
            hit_box_y_offset=jnp.array(98),
            reflecting=jnp.array(1),
            score_type=jnp.array(2),
            variant=jnp.array(1),
        )
        left_spinner_top_right_1 = SceneObject(
            hit_box_height=jnp.array(self.SPINNER_TOP_BOTTOM_SMALL_HEIGHT),
            hit_box_width=jnp.array(self.SPINNER_TOP_BOTTOM_SMALL_WIDTH),
            hit_box_x_offset=jnp.array(35),
            hit_box_y_offset=jnp.array(96),
            reflecting=jnp.array(1),
            score_type=jnp.array(2),
            variant=jnp.array(1),
        )
        left_spinner_top_right_2 = SceneObject(
            hit_box_height=jnp.array(self.SPINNER_TOP_BOTTOM_SMALL_HEIGHT),
            hit_box_width=jnp.array(self.SPINNER_TOP_BOTTOM_SMALL_WIDTH),
            hit_box_x_offset=jnp.array(34),
            hit_box_y_offset=jnp.array(98),
            reflecting=jnp.array(1),
            score_type=jnp.array(2),
            variant=jnp.array(1),
        )

        left_spinner_left_joined = SceneObject(
            hit_box_height=jnp.array(self.SPINNER_LEFT_RIGHT_LARGE_HEIGHT),
            hit_box_width=jnp.array(self.SPINNER_LEFT_RIGHT_LARGE_WIDTH),
            hit_box_x_offset=jnp.array(30),
            hit_box_y_offset=jnp.array(94),
            reflecting=jnp.array(1),
            score_type=jnp.array(2),
            variant=jnp.array(2),
        )
        left_spinner_left_left_1 = SceneObject(
            hit_box_height=jnp.array(self.SPINNER_LEFT_RIGHT_SMALL_HEIGHT),
            hit_box_width=jnp.array(self.SPINNER_LEFT_RIGHT_SMALL_WIDTH),
            hit_box_x_offset=jnp.array(33),
            hit_box_y_offset=jnp.array(90),
            reflecting=jnp.array(1),
            score_type=jnp.array(2),
            variant=jnp.array(2),
        )
        left_spinner_left_left_2 = SceneObject(
            hit_box_height=jnp.array(self.SPINNER_LEFT_RIGHT_SMALL_HEIGHT),
            hit_box_width=jnp.array(self.SPINNER_LEFT_RIGHT_SMALL_WIDTH),
            hit_box_x_offset=jnp.array(34),
            hit_box_y_offset=jnp.array(92),
            reflecting=jnp.array(1),
            score_type=jnp.array(2),
            variant=jnp.array(2),
        )
        left_spinner_left_right_1 = SceneObject(
            hit_box_height=jnp.array(self.SPINNER_LEFT_RIGHT_SMALL_HEIGHT),
            hit_box_width=jnp.array(self.SPINNER_LEFT_RIGHT_SMALL_WIDTH),
            hit_box_x_offset=jnp.array(33),
            hit_box_y_offset=jnp.array(100),
            reflecting=jnp.array(1),
            score_type=jnp.array(2),
            variant=jnp.array(2),
        )
        left_spinner_left_right_2 = SceneObject(
            hit_box_height=jnp.array(self.SPINNER_LEFT_RIGHT_SMALL_HEIGHT),
            hit_box_width=jnp.array(self.SPINNER_LEFT_RIGHT_SMALL_WIDTH),
            hit_box_x_offset=jnp.array(34),
            hit_box_y_offset=jnp.array(98),
            reflecting=jnp.array(1),
            score_type=jnp.array(2),
            variant=jnp.array(2),
        )

        left_spinner_right_joined = SceneObject(
            hit_box_height=jnp.array(self.SPINNER_LEFT_RIGHT_LARGE_HEIGHT),
            hit_box_width=jnp.array(self.SPINNER_LEFT_RIGHT_LARGE_WIDTH),
            hit_box_x_offset=jnp.array(35),
            hit_box_y_offset=jnp.array(94),
            reflecting=jnp.array(1),
            score_type=jnp.array(2),
            variant=jnp.array(3),
        )
        left_spinner_right_left_1 = SceneObject(
            hit_box_height=jnp.array(self.SPINNER_LEFT_RIGHT_SMALL_HEIGHT),
            hit_box_width=jnp.array(self.SPINNER_LEFT_RIGHT_SMALL_WIDTH),
            hit_box_x_offset=jnp.array(32),
            hit_box_y_offset=jnp.array(100),
            reflecting=jnp.array(1),
            score_type=jnp.array(2),
            variant=jnp.array(3),
        )
        left_spinner_right_left_2 = SceneObject(
            hit_box_height=jnp.array(self.SPINNER_LEFT_RIGHT_SMALL_HEIGHT),
            hit_box_width=jnp.array(self.SPINNER_LEFT_RIGHT_SMALL_WIDTH),
            hit_box_x_offset=jnp.array(31),
            hit_box_y_offset=jnp.array(98),
            reflecting=jnp.array(1),
            score_type=jnp.array(2),
            variant=jnp.array(3),
        )
        left_spinner_right_right_1 = SceneObject(
            hit_box_height=jnp.array(self.SPINNER_LEFT_RIGHT_SMALL_HEIGHT),
            hit_box_width=jnp.array(self.SPINNER_LEFT_RIGHT_SMALL_WIDTH),
            hit_box_x_offset=jnp.array(32),
            hit_box_y_offset=jnp.array(90),
            reflecting=jnp.array(1),
            score_type=jnp.array(2),
            variant=jnp.array(3),
        )
        left_spinner_right_right_2 = SceneObject(
            hit_box_height=jnp.array(self.SPINNER_LEFT_RIGHT_SMALL_HEIGHT),
            hit_box_width=jnp.array(self.SPINNER_LEFT_RIGHT_SMALL_WIDTH),
            hit_box_x_offset=jnp.array(31),
            hit_box_y_offset=jnp.array(92),
            reflecting=jnp.array(1),
            score_type=jnp.array(2),
            variant=jnp.array(3),
        )

        # Right Spinner Scene Objects
        right_spinner_bottom_joined = SceneObject(
            hit_box_height=jnp.array(self.SPINNER_TOP_BOTTOM_LARGE_HEIGHT),
            hit_box_width=jnp.array(self.SPINNER_TOP_BOTTOM_LARGE_WIDTH),
            hit_box_x_offset=jnp.array(128),
            hit_box_y_offset=jnp.array(100),
            reflecting=jnp.array(1),
            score_type=jnp.array(2),
            variant=jnp.array(0),
        )
        right_spinner_bottom_left_1 = SceneObject(
            hit_box_height=jnp.array(self.SPINNER_TOP_BOTTOM_SMALL_HEIGHT),
            hit_box_width=jnp.array(self.SPINNER_TOP_BOTTOM_SMALL_WIDTH),
            hit_box_x_offset=jnp.array(126),
            hit_box_y_offset=jnp.array(94),
            reflecting=jnp.array(1),
            score_type=jnp.array(2),
            variant=jnp.array(0),
        )
        right_spinner_bottom_left_2 = SceneObject(
            hit_box_height=jnp.array(self.SPINNER_TOP_BOTTOM_SMALL_HEIGHT),
            hit_box_width=jnp.array(self.SPINNER_TOP_BOTTOM_SMALL_WIDTH),
            hit_box_x_offset=jnp.array(127),
            hit_box_y_offset=jnp.array(92),
            reflecting=jnp.array(1),
            score_type=jnp.array(2),
            variant=jnp.array(0),
        )
        right_spinner_bottom_right_1 = SceneObject(
            hit_box_height=jnp.array(self.SPINNER_TOP_BOTTOM_SMALL_HEIGHT),
            hit_box_width=jnp.array(self.SPINNER_TOP_BOTTOM_SMALL_WIDTH),
            hit_box_x_offset=jnp.array(131),
            hit_box_y_offset=jnp.array(94),
            reflecting=jnp.array(1),
            score_type=jnp.array(2),
            variant=jnp.array(0),
        )
        right_spinner_bottom_right_2 = SceneObject(
            hit_box_height=jnp.array(self.SPINNER_TOP_BOTTOM_SMALL_HEIGHT),
            hit_box_width=jnp.array(self.SPINNER_TOP_BOTTOM_SMALL_WIDTH),
            hit_box_x_offset=jnp.array(130),
            hit_box_y_offset=jnp.array(92),
            reflecting=jnp.array(1),
            score_type=jnp.array(2),
            variant=jnp.array(0),
        )

        right_spinner_top_joined = SceneObject(
            hit_box_height=jnp.array(self.SPINNER_TOP_BOTTOM_LARGE_HEIGHT),
            hit_box_width=jnp.array(self.SPINNER_TOP_BOTTOM_LARGE_WIDTH),
            hit_box_x_offset=jnp.array(128),
            hit_box_y_offset=jnp.array(90),
            reflecting=jnp.array(1),
            score_type=jnp.array(2),
            variant=jnp.array(1),
        )
        right_spinner_top_left_1 = SceneObject(
            hit_box_height=jnp.array(self.SPINNER_TOP_BOTTOM_SMALL_HEIGHT),
            hit_box_width=jnp.array(self.SPINNER_TOP_BOTTOM_SMALL_WIDTH),
            hit_box_x_offset=jnp.array(131),
            hit_box_y_offset=jnp.array(96),
            reflecting=jnp.array(1),
            score_type=jnp.array(2),
            variant=jnp.array(1),
        )
        right_spinner_top_left_2 = SceneObject(
            hit_box_height=jnp.array(self.SPINNER_TOP_BOTTOM_SMALL_HEIGHT),
            hit_box_width=jnp.array(self.SPINNER_TOP_BOTTOM_SMALL_WIDTH),
            hit_box_x_offset=jnp.array(130),
            hit_box_y_offset=jnp.array(98),
            reflecting=jnp.array(1),
            score_type=jnp.array(2),
            variant=jnp.array(1),
        )
        right_spinner_top_right_1 = SceneObject(
            hit_box_height=jnp.array(self.SPINNER_TOP_BOTTOM_SMALL_HEIGHT),
            hit_box_width=jnp.array(self.SPINNER_TOP_BOTTOM_SMALL_WIDTH),
            hit_box_x_offset=jnp.array(126),
            hit_box_y_offset=jnp.array(96),
            reflecting=jnp.array(1),
            score_type=jnp.array(2),
            variant=jnp.array(1),
        )
        right_spinner_top_right_2 = SceneObject(
            hit_box_height=jnp.array(self.SPINNER_TOP_BOTTOM_SMALL_HEIGHT),
            hit_box_width=jnp.array(self.SPINNER_TOP_BOTTOM_SMALL_WIDTH),
            hit_box_x_offset=jnp.array(127),
            hit_box_y_offset=jnp.array(98),
            reflecting=jnp.array(1),
            score_type=jnp.array(2),
            variant=jnp.array(1),
        )

        right_spinner_left_joined = SceneObject(
            hit_box_height=jnp.array(self.SPINNER_LEFT_RIGHT_LARGE_HEIGHT),
            hit_box_width=jnp.array(self.SPINNER_LEFT_RIGHT_LARGE_WIDTH),
            hit_box_x_offset=jnp.array(126),
            hit_box_y_offset=jnp.array(94),
            reflecting=jnp.array(1),
            score_type=jnp.array(2),
            variant=jnp.array(2),
        )
        right_spinner_left_left_1 = SceneObject(
            hit_box_height=jnp.array(self.SPINNER_LEFT_RIGHT_SMALL_HEIGHT),
            hit_box_width=jnp.array(self.SPINNER_LEFT_RIGHT_SMALL_WIDTH),
            hit_box_x_offset=jnp.array(129),
            hit_box_y_offset=jnp.array(90),
            reflecting=jnp.array(1),
            score_type=jnp.array(2),
            variant=jnp.array(2),
        )
        right_spinner_left_left_2 = SceneObject(
            hit_box_height=jnp.array(self.SPINNER_LEFT_RIGHT_SMALL_HEIGHT),
            hit_box_width=jnp.array(self.SPINNER_LEFT_RIGHT_SMALL_WIDTH),
            hit_box_x_offset=jnp.array(130),
            hit_box_y_offset=jnp.array(92),
            reflecting=jnp.array(1),
            score_type=jnp.array(2),
            variant=jnp.array(2),
        )
        right_spinner_left_right_1 = SceneObject(
            hit_box_height=jnp.array(self.SPINNER_LEFT_RIGHT_SMALL_HEIGHT),
            hit_box_width=jnp.array(self.SPINNER_LEFT_RIGHT_SMALL_WIDTH),
            hit_box_x_offset=jnp.array(129),
            hit_box_y_offset=jnp.array(100),
            reflecting=jnp.array(1),
            score_type=jnp.array(2),
            variant=jnp.array(2),
        )
        right_spinner_left_right_2 = SceneObject(
            hit_box_height=jnp.array(self.SPINNER_LEFT_RIGHT_SMALL_HEIGHT),
            hit_box_width=jnp.array(self.SPINNER_LEFT_RIGHT_SMALL_WIDTH),
            hit_box_x_offset=jnp.array(130),
            hit_box_y_offset=jnp.array(98),
            reflecting=jnp.array(1),
            score_type=jnp.array(2),
            variant=jnp.array(2),
        )

        right_spinner_right_joined = SceneObject(
            hit_box_height=jnp.array(self.SPINNER_LEFT_RIGHT_LARGE_HEIGHT),
            hit_box_width=jnp.array(self.SPINNER_LEFT_RIGHT_LARGE_WIDTH),
            hit_box_x_offset=jnp.array(131),
            hit_box_y_offset=jnp.array(94),
            reflecting=jnp.array(1),
            score_type=jnp.array(2),
            variant=jnp.array(3),
        )
        right_spinner_right_left_1 = SceneObject(
            hit_box_height=jnp.array(self.SPINNER_LEFT_RIGHT_SMALL_HEIGHT),
            hit_box_width=jnp.array(self.SPINNER_LEFT_RIGHT_SMALL_WIDTH),
            hit_box_x_offset=jnp.array(128),
            hit_box_y_offset=jnp.array(100),
            reflecting=jnp.array(1),
            score_type=jnp.array(2),
            variant=jnp.array(3),
        )
        right_spinner_right_left_2 = SceneObject(
            hit_box_height=jnp.array(self.SPINNER_LEFT_RIGHT_SMALL_HEIGHT),
            hit_box_width=jnp.array(self.SPINNER_LEFT_RIGHT_SMALL_WIDTH),
            hit_box_x_offset=jnp.array(127),
            hit_box_y_offset=jnp.array(98),
            reflecting=jnp.array(1),
            score_type=jnp.array(2),
            variant=jnp.array(3),
        )
        right_spinner_right_right_1 = SceneObject(
            hit_box_height=jnp.array(self.SPINNER_LEFT_RIGHT_SMALL_HEIGHT),
            hit_box_width=jnp.array(self.SPINNER_LEFT_RIGHT_SMALL_WIDTH),
            hit_box_x_offset=jnp.array(128),
            hit_box_y_offset=jnp.array(90),
            reflecting=jnp.array(1),
            score_type=jnp.array(2),
            variant=jnp.array(3),
        )
        right_spinner_right_right_2 = SceneObject(
            hit_box_height=jnp.array(self.SPINNER_LEFT_RIGHT_SMALL_HEIGHT),
            hit_box_width=jnp.array(self.SPINNER_LEFT_RIGHT_SMALL_WIDTH),
            hit_box_x_offset=jnp.array(127),
            hit_box_y_offset=jnp.array(92),
            reflecting=jnp.array(1),
            score_type=jnp.array(2),
            variant=jnp.array(3),
        )

        # Lit up targets
        left_lit_up_target_large_vertical = SceneObject(
            hit_box_height=diamond_vertical_height,
            hit_box_width=diamond_vertical_width,
            hit_box_x_offset=jnp.array(62),
            hit_box_y_offset=jnp.array(26),
            reflecting=jnp.array(0),
            score_type=jnp.array(6),
            variant=jnp.array(0),
        )
        left_lit_up_target_large_horizontal = SceneObject(
            hit_box_height=diamond_horizontal_height,
            hit_box_width=diamond_horizontal_width,
            hit_box_x_offset=jnp.array(61),
            hit_box_y_offset=jnp.array(28),
            reflecting=jnp.array(0),
            score_type=jnp.array(6),
            variant=jnp.array(0),
        )
        left_lit_up_target_small = SceneObject(
            hit_box_height=diamond_small_height,
            hit_box_width=diamond_small_width,
            hit_box_x_offset=jnp.array(63),
            hit_box_y_offset=jnp.array(30),
            reflecting=jnp.array(1),
            score_type=jnp.array(6),
            variant=jnp.array(0),
        )

        middle_lit_up_target_large_vertical = SceneObject(
            hit_box_height=diamond_vertical_height,
            hit_box_width=diamond_vertical_width,
            hit_box_x_offset=jnp.array(78),
            hit_box_y_offset=jnp.array(26),
            reflecting=jnp.array(0),
            score_type=jnp.array(7),
            variant=jnp.array(1),
        )
        middle_lit_up_target_large_horizontal = SceneObject(
            hit_box_height=diamond_horizontal_height,
            hit_box_width=diamond_horizontal_width,
            hit_box_x_offset=jnp.array(77),
            hit_box_y_offset=jnp.array(28),
            reflecting=jnp.array(0),
            score_type=jnp.array(7),
            variant=jnp.array(1),
        )
        middle_lit_up_target_small = SceneObject(
            hit_box_height=diamond_small_height,
            hit_box_width=diamond_small_width,
            hit_box_x_offset=jnp.array(79),
            hit_box_y_offset=jnp.array(30),
            reflecting=jnp.array(1),
            score_type=jnp.array(7),
            variant=jnp.array(1),
        )

        right_lit_up_target_large_vertical = SceneObject(
            hit_box_height=diamond_vertical_height,
            hit_box_width=diamond_vertical_width,
            hit_box_x_offset=jnp.array(94),
            hit_box_y_offset=jnp.array(26),
            reflecting=jnp.array(0),
            score_type=jnp.array(8),
            variant=jnp.array(2),
        )
        right_lit_up_target_large_horizontal = SceneObject(
            hit_box_height=diamond_horizontal_height,
            hit_box_width=diamond_horizontal_width,
            hit_box_x_offset=jnp.array(93),
            hit_box_y_offset=jnp.array(28),
            reflecting=jnp.array(0),
            score_type=jnp.array(8),
            variant=jnp.array(2),
        )
        right_lit_up_target_small = SceneObject(
            hit_box_height=diamond_small_height,
            hit_box_width=diamond_small_width,
            hit_box_x_offset=jnp.array(95),
            hit_box_y_offset=jnp.array(30),
            reflecting=jnp.array(1),
            score_type=jnp.array(8),
            variant=jnp.array(2),
        )

        special_lit_up_target_large_vertical = SceneObject(
            hit_box_height=diamond_vertical_height,
            hit_box_width=diamond_vertical_width,
            hit_box_x_offset=jnp.array(79),
            hit_box_y_offset=jnp.array(122),
            reflecting=jnp.array(0),
            score_type=jnp.array(5),
            variant=jnp.array(3),
        )
        special_lit_up_target_large_horizontal = SceneObject(
            hit_box_height=diamond_horizontal_height,
            hit_box_width=diamond_horizontal_width,
            hit_box_x_offset=jnp.array(78),
            hit_box_y_offset=jnp.array(124),
            reflecting=jnp.array(0),
            score_type=jnp.array(5),
            variant=jnp.array(3),
        )

        left_rollover = SceneObject(
            hit_box_height=jnp.array(12),
            hit_box_width=jnp.array(10),
            hit_box_x_offset=jnp.array(43),
            hit_box_y_offset=jnp.array(58),
            reflecting=jnp.array(0),
            score_type=jnp.array(3),
            variant=jnp.array(-1),
        )
        atari_rollover = SceneObject(
            hit_box_height=jnp.array(12),
            hit_box_width=jnp.array(10),
            hit_box_x_offset=jnp.array(107),
            hit_box_y_offset=jnp.array(58),
            reflecting=jnp.array(0),
            score_type=jnp.array(4),
            variant=jnp.array(-1),
        )

        middle_bar = SceneObject(
            hit_box_height=jnp.array(self.MIDDLE_BAR_HEIGHT),
            hit_box_width=jnp.array(self.MIDDLE_BAR_WIDTH),
            hit_box_x_offset=jnp.array(self.MIDDLE_BAR_X),
            hit_box_y_offset=jnp.array(self.MIDDLE_BAR_Y),
            reflecting=jnp.array(1),
            score_type=jnp.array(0),
            variant=jnp.array(-1),
        )

        # Flipper Scene Objects
        left_flipper_00_bot = SceneObject(
            hit_box_height=jnp.abs(self.FLIPPER_PIVOT_Y_BOT - self.FLIPPER_00_BOT_END_Y),
            hit_box_width=jnp.abs(self.FLIPPER_LEFT_00_END_X - self.FLIPPER_LEFT_PIVOT_X),
            hit_box_x_offset=self.FLIPPER_LEFT_PIVOT_X,
            hit_box_y_offset=jnp.min(jnp.array([self.FLIPPER_PIVOT_Y_BOT, self.FLIPPER_00_BOT_END_Y])),
            reflecting=jnp.array(1),
            score_type=jnp.array(9),
            variant=jnp.array(0),
        )
        left_flipper_16_bot = SceneObject(
            hit_box_height=jnp.abs(self.FLIPPER_PIVOT_Y_BOT - self.FLIPPER_16_BOT_END_Y),
            hit_box_width=jnp.abs(self.FLIPPER_LEFT_16_END_X - self.FLIPPER_LEFT_PIVOT_X),
            hit_box_x_offset=self.FLIPPER_LEFT_PIVOT_X,
            hit_box_y_offset=jnp.min(jnp.array([self.FLIPPER_PIVOT_Y_BOT, self.FLIPPER_16_BOT_END_Y])),
            reflecting=jnp.array(1),
            score_type=jnp.array(9),
            variant=jnp.array(1),
        )
        left_flipper_32_bot = SceneObject(
            hit_box_height=jnp.abs(self.FLIPPER_PIVOT_Y_BOT - self.FLIPPER_32_BOT_END_Y),
            hit_box_width=jnp.abs(self.FLIPPER_LEFT_32_END_X - self.FLIPPER_LEFT_PIVOT_X),
            hit_box_x_offset=self.FLIPPER_LEFT_PIVOT_X,
            hit_box_y_offset=jnp.min(jnp.array([self.FLIPPER_PIVOT_Y_BOT, self.FLIPPER_32_BOT_END_Y])),
            reflecting=jnp.array(1),
            score_type=jnp.array(9),
            variant=jnp.array(2),
        )
        left_flipper_48_bot = SceneObject(
            hit_box_height=jnp.abs(self.FLIPPER_PIVOT_Y_BOT - self.FLIPPER_48_BOT_END_Y),
            hit_box_width=jnp.abs(self.FLIPPER_LEFT_48_END_X - self.FLIPPER_LEFT_PIVOT_X),
            hit_box_x_offset=self.FLIPPER_LEFT_PIVOT_X,
            hit_box_y_offset=jnp.min(jnp.array([self.FLIPPER_PIVOT_Y_BOT, self.FLIPPER_48_BOT_END_Y])),
            reflecting=jnp.array(1),
            score_type=jnp.array(9),
            variant=jnp.array(3),
        )

        left_flipper_00_top = SceneObject(
            hit_box_height=jnp.abs(self.FLIPPER_PIVOT_Y_TOP - self.FLIPPER_00_TOP_END_Y),
            hit_box_width=jnp.abs(self.FLIPPER_LEFT_00_END_X - self.FLIPPER_LEFT_PIVOT_X),
            hit_box_x_offset=self.FLIPPER_LEFT_PIVOT_X,
            hit_box_y_offset=jnp.min(jnp.array([self.FLIPPER_PIVOT_Y_TOP, self.FLIPPER_00_TOP_END_Y])),
            reflecting=jnp.array(1),
            score_type=jnp.array(9),
            variant=jnp.array(4),
        )
        left_flipper_16_top = SceneObject(
            hit_box_height=jnp.abs(self.FLIPPER_PIVOT_Y_TOP - self.FLIPPER_16_TOP_END_Y),
            hit_box_width=jnp.abs(self.FLIPPER_LEFT_16_END_X - self.FLIPPER_LEFT_PIVOT_X),
            hit_box_x_offset=self.FLIPPER_LEFT_PIVOT_X,
            hit_box_y_offset=jnp.min(jnp.array([self.FLIPPER_PIVOT_Y_TOP, self.FLIPPER_16_TOP_END_Y])),
            reflecting=jnp.array(1),
            score_type=jnp.array(9),
            variant=jnp.array(5),
        )
        left_flipper_32_top = SceneObject(
            hit_box_height=jnp.abs(self.FLIPPER_PIVOT_Y_TOP - self.FLIPPER_32_TOP_END_Y),
            hit_box_width=jnp.abs(self.FLIPPER_LEFT_32_END_X - self.FLIPPER_LEFT_PIVOT_X),
            hit_box_x_offset=self.FLIPPER_LEFT_PIVOT_X,
            hit_box_y_offset=jnp.min(jnp.array([self.FLIPPER_PIVOT_Y_TOP, self.FLIPPER_32_TOP_END_Y])),
            reflecting=jnp.array(1),
            score_type=jnp.array(9),
            variant=jnp.array(6),
        )
        left_flipper_48_top = SceneObject(
            hit_box_height=jnp.abs(self.FLIPPER_PIVOT_Y_TOP - self.FLIPPER_48_TOP_END_Y),
            hit_box_width=jnp.abs(self.FLIPPER_LEFT_48_END_X - self.FLIPPER_LEFT_PIVOT_X),
            hit_box_x_offset=self.FLIPPER_LEFT_PIVOT_X,
            hit_box_y_offset=jnp.min(jnp.array([self.FLIPPER_PIVOT_Y_TOP, self.FLIPPER_48_TOP_END_Y])),
            reflecting=jnp.array(1),
            score_type=jnp.array(9),
            variant=jnp.array(7),
        )

        right_flipper_00_bot = SceneObject(
            hit_box_height=jnp.abs(self.FLIPPER_PIVOT_Y_BOT - self.FLIPPER_00_BOT_END_Y),
            hit_box_width=jnp.abs(self.FLIPPER_RIGHT_PIVOT_X - self.FLIPPER_RIGHT_00_END_X),
            hit_box_x_offset=jnp.min(jnp.array([self.FLIPPER_RIGHT_PIVOT_X, self.FLIPPER_RIGHT_00_END_X])),
            hit_box_y_offset=jnp.min(jnp.array([self.FLIPPER_PIVOT_Y_BOT, self.FLIPPER_00_BOT_END_Y])),
            reflecting=jnp.array(1),
            score_type=jnp.array(10),
            variant=jnp.array(8),
        )
        right_flipper_16_bot = SceneObject(
            hit_box_height=jnp.abs(self.FLIPPER_PIVOT_Y_BOT - self.FLIPPER_16_BOT_END_Y),
            hit_box_width=jnp.abs(self.FLIPPER_RIGHT_PIVOT_X - self.FLIPPER_RIGHT_16_END_X),
            hit_box_x_offset=jnp.min(jnp.array([self.FLIPPER_RIGHT_PIVOT_X, self.FLIPPER_RIGHT_16_END_X])),
            hit_box_y_offset=jnp.min(jnp.array([self.FLIPPER_PIVOT_Y_BOT, self.FLIPPER_16_BOT_END_Y])),
            reflecting=jnp.array(1),
            score_type=jnp.array(10),
            variant=jnp.array(9),
        )
        right_flipper_32_bot = SceneObject(
            hit_box_height=jnp.abs(self.FLIPPER_PIVOT_Y_BOT - self.FLIPPER_32_BOT_END_Y),
            hit_box_width=jnp.abs(self.FLIPPER_RIGHT_PIVOT_X - self.FLIPPER_RIGHT_32_END_X),
            hit_box_x_offset=jnp.min(jnp.array([self.FLIPPER_RIGHT_PIVOT_X, self.FLIPPER_RIGHT_32_END_X])),
            hit_box_y_offset=jnp.min(jnp.array([self.FLIPPER_PIVOT_Y_BOT, self.FLIPPER_32_BOT_END_Y])),
            reflecting=jnp.array(1),
            score_type=jnp.array(10),
            variant=jnp.array(10),
        )
        right_flipper_48_bot = SceneObject(
            hit_box_height=jnp.abs(self.FLIPPER_PIVOT_Y_BOT - self.FLIPPER_48_BOT_END_Y),
            hit_box_width=jnp.abs(self.FLIPPER_RIGHT_PIVOT_X - self.FLIPPER_RIGHT_48_END_X),
            hit_box_x_offset=jnp.min(jnp.array([self.FLIPPER_RIGHT_PIVOT_X, self.FLIPPER_RIGHT_48_END_X])),
            hit_box_y_offset=jnp.min(jnp.array([self.FLIPPER_PIVOT_Y_BOT, self.FLIPPER_48_BOT_END_Y])),
            reflecting=jnp.array(1),
            score_type=jnp.array(10),
            variant=jnp.array(11),
        )

        right_flipper_00_top = SceneObject(
            hit_box_height=jnp.abs(self.FLIPPER_PIVOT_Y_TOP - self.FLIPPER_00_TOP_END_Y),
            hit_box_width=jnp.abs(self.FLIPPER_RIGHT_PIVOT_X - self.FLIPPER_RIGHT_00_END_X),
            hit_box_x_offset=jnp.min(jnp.array([self.FLIPPER_RIGHT_PIVOT_X, self.FLIPPER_RIGHT_00_END_X])),
            hit_box_y_offset=jnp.min(jnp.array([self.FLIPPER_PIVOT_Y_TOP, self.FLIPPER_00_TOP_END_Y])),
            reflecting=jnp.array(1),
            score_type=jnp.array(10),
            variant=jnp.array(12),
        )
        right_flipper_16_top = SceneObject(
            hit_box_height=jnp.abs(self.FLIPPER_PIVOT_Y_TOP - self.FLIPPER_16_TOP_END_Y),
            hit_box_width=jnp.abs(self.FLIPPER_RIGHT_PIVOT_X - self.FLIPPER_RIGHT_16_END_X),
            hit_box_x_offset=jnp.min(jnp.array([self.FLIPPER_RIGHT_PIVOT_X, self.FLIPPER_RIGHT_16_END_X])),
            hit_box_y_offset=jnp.min(jnp.array([self.FLIPPER_PIVOT_Y_TOP, self.FLIPPER_16_TOP_END_Y])),
            reflecting=jnp.array(1),
            score_type=jnp.array(10),
            variant=jnp.array(13),
        )
        right_flipper_32_top = SceneObject(
            hit_box_height=jnp.abs(self.FLIPPER_PIVOT_Y_TOP - self.FLIPPER_32_TOP_END_Y),
            hit_box_width=jnp.abs(self.FLIPPER_RIGHT_PIVOT_X - self.FLIPPER_RIGHT_32_END_X),
            hit_box_x_offset=jnp.min(jnp.array([self.FLIPPER_RIGHT_PIVOT_X, self.FLIPPER_RIGHT_32_END_X])),
            hit_box_y_offset=jnp.min(jnp.array([self.FLIPPER_PIVOT_Y_TOP, self.FLIPPER_32_TOP_END_Y])),
            reflecting=jnp.array(1),
            score_type=jnp.array(10),
            variant=jnp.array(14),
        )
        right_flipper_48_top = SceneObject(
            hit_box_height=jnp.abs(self.FLIPPER_PIVOT_Y_TOP - self.FLIPPER_48_TOP_END_Y),
            hit_box_width=jnp.abs(self.FLIPPER_RIGHT_PIVOT_X - self.FLIPPER_RIGHT_48_END_X),
            hit_box_x_offset=jnp.min(jnp.array([self.FLIPPER_RIGHT_PIVOT_X, self.FLIPPER_RIGHT_48_END_X])),
            hit_box_y_offset=jnp.min(jnp.array([self.FLIPPER_PIVOT_Y_TOP, self.FLIPPER_48_TOP_END_Y])),
            reflecting=jnp.array(1),
            score_type=jnp.array(10),
            variant=jnp.array(15),
        )

        # Build the list of all scene objects
        all_scene_objects_list = [
            left_lit_up_target_large_vertical,
            left_lit_up_target_large_horizontal,
            left_lit_up_target_small,
            middle_lit_up_target_large_vertical,
            middle_lit_up_target_large_horizontal,
            middle_lit_up_target_small,
            right_lit_up_target_large_vertical,
            right_lit_up_target_large_horizontal,
            right_lit_up_target_small,
            special_lit_up_target_large_vertical,
            special_lit_up_target_large_horizontal,
            left_rollover,
            atari_rollover,
            top_wall,
            bottom_wall_1,
            bottom_wall_2,
            bottom_wall_3,
            bottom_wall_4,
            tilt_mode_hole_plug_left,
            tilt_mode_hole_plug_right,
            left_wall,
            right_wall,
            left_inner_wall,
            right_inner_wall,
            left_quadruple_step,
            right_quadruple_step,
            left_triple_step,
            right_triple_step,
            left_double_step,
            right_double_step,
            left_single_step,
            right_single_step,
            top_left_step,
            top_right_step,
            left_rollover_left_bar,
            left_rollover_right_bar,
            atari_rollover_left_bar,
            atari_rollover_right_bar,
            top_bumper,
            left_bumper,
            right_bumper,
            left_spinner_bottom_joined,
            left_spinner_bottom_left_1,
            left_spinner_bottom_left_2,
            left_spinner_bottom_right_1,
            left_spinner_bottom_right_2,
            left_spinner_right_joined,
            left_spinner_right_left_1,
            left_spinner_right_left_2,
            left_spinner_right_right_1,
            left_spinner_right_right_2,
            left_spinner_top_joined,
            left_spinner_top_left_1,
            left_spinner_top_left_2,
            left_spinner_top_right_1,
            left_spinner_top_right_2,
            left_spinner_left_joined,
            left_spinner_left_left_1,
            left_spinner_left_left_2,
            left_spinner_left_right_1,
            left_spinner_left_right_2,
            right_spinner_bottom_joined,
            right_spinner_bottom_left_1,
            right_spinner_bottom_left_2,
            right_spinner_bottom_right_1,
            right_spinner_bottom_right_2,
            right_spinner_right_joined,
            right_spinner_right_left_1,
            right_spinner_right_left_2,
            right_spinner_right_right_1,
            right_spinner_right_right_2,
            right_spinner_top_joined,
            right_spinner_top_left_1,
            right_spinner_top_left_2,
            right_spinner_top_right_1,
            right_spinner_top_right_2,
            right_spinner_left_joined,
            right_spinner_left_left_1,
            right_spinner_left_left_2,
            right_spinner_left_right_1,
            right_spinner_left_right_2,
            middle_bar,
            left_flipper_00_bot,
            left_flipper_00_top,
            left_flipper_16_bot,
            left_flipper_16_top,
            left_flipper_32_bot,
            left_flipper_32_top,
            left_flipper_48_bot,
            left_flipper_48_top,
            right_flipper_00_bot,
            right_flipper_00_top,
            right_flipper_16_bot,
            right_flipper_16_top,
            right_flipper_32_bot,
            right_flipper_32_top,
            right_flipper_48_bot,
            right_flipper_48_top,
        ]

        # Compute derived arrays
        reflecting_scene_objects = jnp.stack([
            jnp.array([
                scene_object.hit_box_width,
                scene_object.hit_box_height,
                scene_object.hit_box_x_offset,
                scene_object.hit_box_y_offset,
                scene_object.reflecting,
                scene_object.score_type,
                scene_object.variant,
            ], dtype=jnp.int32)
            for scene_object in all_scene_objects_list
            if scene_object.reflecting == 1
        ]).squeeze()

        non_reflecting_scene_objects = jnp.stack([
            jnp.array([
                scene_object.hit_box_width,
                scene_object.hit_box_height,
                scene_object.hit_box_x_offset,
                scene_object.hit_box_y_offset,
                scene_object.reflecting,
                scene_object.score_type,
                scene_object.variant,
            ], dtype=jnp.int32)
            for scene_object in all_scene_objects_list
            if scene_object.reflecting == 0
        ]).squeeze()

        total_score_types = (
            jnp.max(jnp.concat([reflecting_scene_objects, non_reflecting_scene_objects], axis=0)[:, 5]) + 1
        )

        flippers_sorted = (
            left_flipper_00_bot,
            left_flipper_16_bot,
            left_flipper_32_bot,
            left_flipper_48_bot,
            left_flipper_00_top,
            left_flipper_16_top,
            left_flipper_32_top,
            left_flipper_48_top,
            right_flipper_00_bot,
            right_flipper_16_bot,
            right_flipper_32_bot,
            right_flipper_48_bot,
            right_flipper_00_top,
            right_flipper_16_top,
            right_flipper_32_top,
            right_flipper_48_top,
        )

        flippers = jnp.stack([
            jnp.array([
                scene_object.hit_box_width,
                scene_object.hit_box_height,
                scene_object.hit_box_x_offset,
                scene_object.hit_box_y_offset,
                scene_object.reflecting,
                scene_object.score_type,
                scene_object.variant,
            ], dtype=jnp.int32)
            for scene_object in flippers_sorted
        ]).squeeze()

        flipper_segments_sorted = jnp.array([
            [[self.FLIPPER_LEFT_PIVOT_X, self.FLIPPER_PIVOT_Y_BOT], [self.FLIPPER_LEFT_00_END_X, self.FLIPPER_00_BOT_END_Y]],
            [[self.FLIPPER_LEFT_PIVOT_X, self.FLIPPER_PIVOT_Y_BOT], [self.FLIPPER_LEFT_16_END_X, self.FLIPPER_16_BOT_END_Y]],
            [[self.FLIPPER_LEFT_PIVOT_X, self.FLIPPER_PIVOT_Y_BOT], [self.FLIPPER_LEFT_32_END_X, self.FLIPPER_32_BOT_END_Y]],
            [[self.FLIPPER_LEFT_PIVOT_X, self.FLIPPER_PIVOT_Y_BOT], [self.FLIPPER_LEFT_48_END_X, self.FLIPPER_48_BOT_END_Y]],
            [[self.FLIPPER_LEFT_PIVOT_X, self.FLIPPER_PIVOT_Y_TOP], [self.FLIPPER_LEFT_00_END_X, self.FLIPPER_00_TOP_END_Y]],
            [[self.FLIPPER_LEFT_PIVOT_X, self.FLIPPER_PIVOT_Y_TOP], [self.FLIPPER_LEFT_16_END_X, self.FLIPPER_16_TOP_END_Y]],
            [[self.FLIPPER_LEFT_PIVOT_X, self.FLIPPER_PIVOT_Y_TOP], [self.FLIPPER_LEFT_32_END_X, self.FLIPPER_32_TOP_END_Y]],
            [[self.FLIPPER_LEFT_PIVOT_X, self.FLIPPER_PIVOT_Y_TOP], [self.FLIPPER_LEFT_48_END_X, self.FLIPPER_48_TOP_END_Y]],
            [[self.FLIPPER_RIGHT_PIVOT_X, self.FLIPPER_PIVOT_Y_BOT], [self.FLIPPER_RIGHT_00_END_X, self.FLIPPER_00_BOT_END_Y]],
            [[self.FLIPPER_RIGHT_PIVOT_X, self.FLIPPER_PIVOT_Y_BOT], [self.FLIPPER_RIGHT_16_END_X, self.FLIPPER_16_BOT_END_Y]],
            [[self.FLIPPER_RIGHT_PIVOT_X, self.FLIPPER_PIVOT_Y_BOT], [self.FLIPPER_RIGHT_32_END_X, self.FLIPPER_32_BOT_END_Y]],
            [[self.FLIPPER_RIGHT_PIVOT_X, self.FLIPPER_PIVOT_Y_BOT], [self.FLIPPER_RIGHT_48_END_X, self.FLIPPER_48_BOT_END_Y]],
            [[self.FLIPPER_RIGHT_PIVOT_X, self.FLIPPER_PIVOT_Y_TOP], [self.FLIPPER_RIGHT_00_END_X, self.FLIPPER_00_TOP_END_Y]],
            [[self.FLIPPER_RIGHT_PIVOT_X, self.FLIPPER_PIVOT_Y_TOP], [self.FLIPPER_RIGHT_16_END_X, self.FLIPPER_16_TOP_END_Y]],
            [[self.FLIPPER_RIGHT_PIVOT_X, self.FLIPPER_PIVOT_Y_TOP], [self.FLIPPER_RIGHT_32_END_X, self.FLIPPER_32_TOP_END_Y]],
            [[self.FLIPPER_RIGHT_PIVOT_X, self.FLIPPER_PIVOT_Y_TOP], [self.FLIPPER_RIGHT_48_END_X, self.FLIPPER_48_TOP_END_Y]],
        ])

        return {
            'INVISIBLE_BLOCK_SCENE_OBJECT': invisible_block,
            'TOP_WALL_SCENE_OBJECT': top_wall,
            'BOTTOM_WALL_SCENE_OBJECT_1': bottom_wall_1,
            'BOTTOM_WALL_SCENE_OBJECT_2': bottom_wall_2,
            'BOTTOM_WALL_SCENE_OBJECT_3': bottom_wall_3,
            'BOTTOM_WALL_SCENE_OBJECT_4': bottom_wall_4,
            'TILT_MODE_HOLE_PLUG_LEFT': tilt_mode_hole_plug_left,
            'TILT_MODE_HOLE_PLUG_RIGHT': tilt_mode_hole_plug_right,
            'LEFT_WALL_SCENE_OBJECT': left_wall,
            'RIGHT_WALL_SCENE_OBJECT': right_wall,
            'LEFT_INNER_WALL_SCENE_OBJECT': left_inner_wall,
            'RIGHT_INNER_WALL_SCENE_OBJECT': right_inner_wall,
            'LEFT_QUADRUPLE_STEP_SCENE_OBJECT': left_quadruple_step,
            'RIGHT_QUADRUPLE_STEP_SCENE_OBJECT': right_quadruple_step,
            'LEFT_TRIPLE_STEP_SCENE_OBJECT': left_triple_step,
            'RIGHT_TRIPLE_STEP_SCENE_OBJECT': right_triple_step,
            'LEFT_DOUBLE_STEP_SCENE_OBJECT': left_double_step,
            'RIGHT_DOUBLE_STEP_SCENE_OBJECT': right_double_step,
            'LEFT_SINGLE_STEP_SCENE_OBJECT': left_single_step,
            'DIAMOND_VERTICAL_RECTANGLE_BOUNDING_BOX_HEIGHT': diamond_vertical_height,
            'DIAMOND_VERTICAL_RECTANGLE_BOUNDING_BOX_WIDTH': diamond_vertical_width,
            'DIAMOND_HORIZONTAL_RECTANGLE_BOUNDING_BOX_HEIGHT': diamond_horizontal_height,
            'DIAMOND_HORIZONTAL_RECTANGLE_BOUNDING_BOX_WIDTH': diamond_horizontal_width,
            'DIAMOND_SMALL_RECTANGLE_BOUNDING_BOX_HEIGHT': diamond_small_height,
            'DIAMOND_SMALL_RECTANGLE_BOUNDING_BOX_WIDTH': diamond_small_width,
            'RIGHT_SINGLE_STEP_SCENE_OBJECT': right_single_step,
            'TOP_LEFT_STEP_SCENE_OBJECT': top_left_step,
            'TOP_RIGHT_STEP_SCENE_OBJECT': top_right_step,
            'LEFT_ROLLOVER_LEFT_BAR_SCENE_OBJECT': left_rollover_left_bar,
            'LEFT_ROLLOVER_RIGHT_BAR_SCENE_OBJECT': left_rollover_right_bar,
            'ATARI_ROLLOVER_LEFT_BAR_SCENE_OBJECT': atari_rollover_left_bar,
            'ATARI_ROLLOVER_RIGHT_BAR_SCENE_OBJECT': atari_rollover_right_bar,
            'TOP_BUMPER_SCENE_OBJECT': top_bumper,
            'LEFT_BUMPER_SCENE_OBJECT': left_bumper,
            'RIGHT_BUMPER_SCENE_OBJECT': right_bumper,
            'LEFT_SPINNER_BOTTOM_POSITION_JOINED_PART_SCENE_OBJECT': left_spinner_bottom_joined,
            'LEFT_SPINNER_BOTTOM_POSITION_LEFT_PART_1_SCENE_OBJECT': left_spinner_bottom_left_1,
            'LEFT_SPINNER_BOTTOM_POSITION_LEFT_PART_2_SCENE_OBJECT': left_spinner_bottom_left_2,
            'LEFT_SPINNER_BOTTOM_POSITION_RIGHT_PART_1_SCENE_OBJECT': left_spinner_bottom_right_1,
            'LEFT_SPINNER_BOTTOM_POSITION_RIGHT_PART_2_SCENE_OBJECT': left_spinner_bottom_right_2,
            'LEFT_SPINNER_TOP_POSITION_JOINED_PART_SCENE_OBJECT': left_spinner_top_joined,
            'LEFT_SPINNER_TOP_POSITION_LEFT_PART_1_SCENE_OBJECT': left_spinner_top_left_1,
            'LEFT_SPINNER_TOP_POSITION_LEFT_PART_2_SCENE_OBJECT': left_spinner_top_left_2,
            'LEFT_SPINNER_TOP_POSITION_RIGHT_PART_1_SCENE_OBJECT': left_spinner_top_right_1,
            'LEFT_SPINNER_TOP_POSITION_RIGHT_PART_2_SCENE_OBJECT': left_spinner_top_right_2,
            'LEFT_SPINNER_LEFT_POSITION_JOINED_PART_SCENE_OBJECT': left_spinner_left_joined,
            'LEFT_SPINNER_LEFT_POSITION_LEFT_PART_1_SCENE_OBJECT': left_spinner_left_left_1,
            'LEFT_SPINNER_LEFT_POSITION_LEFT_PART_2_SCENE_OBJECT': left_spinner_left_left_2,
            'LEFT_SPINNER_LEFT_POSITION_RIGHT_PART_1_SCENE_OBJECT': left_spinner_left_right_1,
            'LEFT_SPINNER_LEFT_POSITION_RIGHT_PART_2_SCENE_OBJECT': left_spinner_left_right_2,
            'LEFT_SPINNER_RIGHT_POSITION_JOINED_PART_SCENE_OBJECT': left_spinner_right_joined,
            'LEFT_SPINNER_RIGHT_POSITION_LEFT_PART_1_SCENE_OBJECT': left_spinner_right_left_1,
            'LEFT_SPINNER_RIGHT_POSITION_LEFT_PART_2_SCENE_OBJECT': left_spinner_right_left_2,
            'LEFT_SPINNER_RIGHT_POSITION_RIGHT_PART_1_SCENE_OBJECT': left_spinner_right_right_1,
            'LEFT_SPINNER_RIGHT_POSITION_RIGHT_PART_2_SCENE_OBJECT': left_spinner_right_right_2,
            'RIGHT_SPINNER_BOTTOM_POSITION_JOINED_PART_SCENE_OBJECT': right_spinner_bottom_joined,
            'RIGHT_SPINNER_BOTTOM_POSITION_LEFT_PART_1_SCENE_OBJECT': right_spinner_bottom_left_1,
            'RIGHT_SPINNER_BOTTOM_POSITION_LEFT_PART_2_SCENE_OBJECT': right_spinner_bottom_left_2,
            'RIGHT_SPINNER_BOTTOM_POSITION_RIGHT_PART_1_SCENE_OBJECT': right_spinner_bottom_right_1,
            'RIGHT_SPINNER_BOTTOM_POSITION_RIGHT_PART_2_SCENE_OBJECT': right_spinner_bottom_right_2,
            'RIGHT_SPINNER_TOP_POSITION_JOINED_PART_SCENE_OBJECT': right_spinner_top_joined,
            'RIGHT_SPINNER_TOP_POSITION_LEFT_PART_1_SCENE_OBJECT': right_spinner_top_left_1,
            'RIGHT_SPINNER_TOP_POSITION_LEFT_PART_2_SCENE_OBJECT': right_spinner_top_left_2,
            'RIGHT_SPINNER_TOP_POSITION_RIGHT_PART_1_SCENE_OBJECT': right_spinner_top_right_1,
            'RIGHT_SPINNER_TOP_POSITION_RIGHT_PART_2_SCENE_OBJECT': right_spinner_top_right_2,
            'RIGHT_SPINNER_LEFT_POSITION_JOINED_PART_SCENE_OBJECT': right_spinner_left_joined,
            'RIGHT_SPINNER_LEFT_POSITION_LEFT_PART_1_SCENE_OBJECT': right_spinner_left_left_1,
            'RIGHT_SPINNER_LEFT_POSITION_LEFT_PART_2_SCENE_OBJECT': right_spinner_left_left_2,
            'RIGHT_SPINNER_LEFT_POSITION_RIGHT_PART_1_SCENE_OBJECT': right_spinner_left_right_1,
            'RIGHT_SPINNER_LEFT_POSITION_RIGHT_PART_2_SCENE_OBJECT': right_spinner_left_right_2,
            'RIGHT_SPINNER_RIGHT_POSITION_JOINED_PART_SCENE_OBJECT': right_spinner_right_joined,
            'RIGHT_SPINNER_RIGHT_POSITION_LEFT_PART_1_SCENE_OBJECT': right_spinner_right_left_1,
            'RIGHT_SPINNER_RIGHT_POSITION_LEFT_PART_2_SCENE_OBJECT': right_spinner_right_left_2,
            'RIGHT_SPINNER_RIGHT_POSITION_RIGHT_PART_1_SCENE_OBJECT': right_spinner_right_right_1,
            'RIGHT_SPINNER_RIGHT_POSITION_RIGHT_PART_2_SCENE_OBJECT': right_spinner_right_right_2,
            'LEFT_LIT_UP_TARGET_LARGE_VERTICAL_SCENE_OBJECT': left_lit_up_target_large_vertical,
            'LEFT_LIT_UP_TARGET_LARGE_HORIZONTAL_SCENE_OBJECT': left_lit_up_target_large_horizontal,
            'LEFT_LIT_UP_TARGET_SMALL_SCENE_OBJECT': left_lit_up_target_small,
            'MIDDLE_LIT_UP_TARGET_LARGE_VERTICAL_SCENE_OBJECT': middle_lit_up_target_large_vertical,
            'MIDDLE_LIT_UP_TARGET_LARGE_HORIZONTAL_SCENE_OBJECT': middle_lit_up_target_large_horizontal,
            'MIDDLE_LIT_UP_TARGET_SMALL_SCENE_OBJECT': middle_lit_up_target_small,
            'RIGHT_LIT_UP_TARGET_LARGE_VERTICAL_SCENE_OBJECT': right_lit_up_target_large_vertical,
            'RIGHT_LIT_UP_TARGET_LARGE_HORIZONTAL_SCENE_OBJECT': right_lit_up_target_large_horizontal,
            'RIGHT_LIT_UP_TARGET_SMALL_SCENE_OBJECT': right_lit_up_target_small,
            'SPECIAL_LIT_UP_TARGET_LARGE_VERTICAL_SCENE_OBJECT': special_lit_up_target_large_vertical,
            'SPECIAL_LIT_UP_TARGET_LARGE_HORIZONTAL_SCENE_OBJECT': special_lit_up_target_large_horizontal,
            'LEFT_ROLLOVER_SCENE_OBJECT': left_rollover,
            'ATARI_ROLLOVER_SCENE_OBJECT': atari_rollover,
            'MIDDLE_BAR_SCENE_OBJECT': middle_bar,
            'LEFT_FLIPPER_00_BOT_SCENE_OBJECT': left_flipper_00_bot,
            'LEFT_FLIPPER_16_BOT_SCENE_OBJECT': left_flipper_16_bot,
            'LEFT_FLIPPER_32_BOT_SCENE_OBJECT': left_flipper_32_bot,
            'LEFT_FLIPPER_48_BOT_SCENE_OBJECT': left_flipper_48_bot,
            'LEFT_FLIPPER_00_TOP_SCENE_OBJECT': left_flipper_00_top,
            'LEFT_FLIPPER_16_TOP_SCENE_OBJECT': left_flipper_16_top,
            'LEFT_FLIPPER_32_TOP_SCENE_OBJECT': left_flipper_32_top,
            'LEFT_FLIPPER_48_TOP_SCENE_OBJECT': left_flipper_48_top,
            'RIGHT_FLIPPER_00_BOT_SCENE_OBJECT': right_flipper_00_bot,
            'RIGHT_FLIPPER_16_BOT_SCENE_OBJECT': right_flipper_16_bot,
            'RIGHT_FLIPPER_32_BOT_SCENE_OBJECT': right_flipper_32_bot,
            'RIGHT_FLIPPER_48_BOT_SCENE_OBJECT': right_flipper_48_bot,
            'RIGHT_FLIPPER_00_TOP_SCENE_OBJECT': right_flipper_00_top,
            'RIGHT_FLIPPER_16_TOP_SCENE_OBJECT': right_flipper_16_top,
            'RIGHT_FLIPPER_32_TOP_SCENE_OBJECT': right_flipper_32_top,
            'RIGHT_FLIPPER_48_TOP_SCENE_OBJECT': right_flipper_48_top,
            'ALL_SCENE_OBJECTS_LIST': all_scene_objects_list,
            'REFLECTING_SCENE_OBJECTS': reflecting_scene_objects,
            'NON_REFLECTING_SCENE_OBJECTS': non_reflecting_scene_objects,
            'TOTAL_SCORE_TYPES': total_score_types,
            '_FLIPPERS_SORTED': flippers_sorted,
            'FLIPPERS': flippers,
            'FLIPPER_SEGMENTS_SORTED': flipper_segments_sorted,
        }
