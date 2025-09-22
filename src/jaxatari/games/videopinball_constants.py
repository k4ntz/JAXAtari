import jax.numpy as jnp
import chex

from typing import NamedTuple


@chex.dataclass
class BallMovement:
    old_ball_x: chex.Array
    old_ball_y: chex.Array
    new_ball_x: chex.Array
    new_ball_y: chex.Array


@chex.dataclass
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
@chex.dataclass
class HitPoint:
    t_entry: chex.Array
    x: chex.Array
    y: chex.Array


# TODO will have to bring this into VideoPinballConstants or refactor another way.
class HitPointSelector(NamedTuple):
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
class VideoPinballState(NamedTuple):
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
    lives: chex.Array
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


class EntityState(NamedTuple):
    x: chex.Array  # x position of the entity (upper left)
    y: chex.Array  # y position of the entity (upper left)
    w: chex.Array  # width of the entity
    h: chex.Array  # height of the entity
    active: chex.Array  # whether this entity is currently active


class VideoPinballObservation(NamedTuple):
    ball: jnp.ndarray
    spinners: jnp.ndarray
    flippers: jnp.ndarray
    plunger: jnp.ndarray
    targets: jnp.ndarray
    bumpers: jnp.ndarray
    rollovers: jnp.ndarray
    tilt_mode_hole_plugs: jnp.ndarray
    score: jnp.ndarray
    lives: jnp.ndarray
    atari_symbols: jnp.ndarray
    bumper_multiplier: jnp.ndarray
    rollover_counter: jnp.ndarray
    color_cycling: jnp.ndarray
    tilt_mode_active: jnp.ndarray


class VideoPinballInfo(NamedTuple):
    time: chex.Array
    plunger_power: chex.Array
    target_cooldown: chex.Array
    special_target_cooldown: chex.Array
    rollover_enabled: chex.Array
    step_counter: chex.Array
    ball_in_play: chex.Array
    respawn_timer: chex.Array
    tilt_counter: chex.Array
    all_rewards: chex.Array


# TODO: This Constants class is required for the game to run with the new environment
# TODO: We should add all of our constants into this class and to do that we must add type hints...
# TODO: ...like in the example with the WIDTH here or else it wont work
# TODO: Finally we need to update all the Constants calls to self.consts.<const_name>
class VideoPinballConstants(NamedTuple):
    # Constants for game environment
    WIDTH = jnp.array(160)
    HEIGHT = jnp.array(210)

    # Physics constants
    GRAVITY = jnp.array(0.03)
    VELOCITY_DAMPENING_VALUE = jnp.array(0.075)  # 24
    VELOCITY_ACCELERATION_VALUE = jnp.array(0.125)
    MAX_REFLECTIONS_PER_GAMESTEP = jnp.array(
        3
    )  # max collisions to process per timestep
    BALL_MAX_SPEED = jnp.array(3.5)
    BALL_MIN_SPEED = jnp.array(0.35)
    NUDGE_EFFECT_INTERVAL = jnp.array(2)  # Num steps in between nudge changing
    NUDGE_EFFECT_AMOUNT = jnp.array(
        0.1
    )  # Amount of nudge effect applied to the ball's velocity
    TILT_COUNT_INCREASE_INTERVAL = jnp.array(
        4
    )  # Number of steps after which the tilt counter increases
    TILT_COUNT_DECREASE_INTERVAL = jnp.array(4)
    TILT_COUNT_TILT_MODE_ACTIVE = jnp.array(512)
    FLIPPER_MAX_ANGLE = jnp.array(3)
    FLIPPER_ANIMATION_Y_OFFSETS = jnp.array(
        [0, 0, 3, 7]
    )  # This is a little scuffed, it would be cleaner to just fix the sprites but this works fine
    FLIPPER_ANIMATION_X_OFFSETS = jnp.array([0, 0, 0, 1])  # Only for the right flipper
    PLUNGER_MAX_POSITION = jnp.array(20)

    # Game logic constants
    T_ENTRY_NO_COLLISION = jnp.array(9999)
    TARGET_RESPAWN_COOLDOWN = jnp.array(16)
    SPECIAL_TARGET_ACTIVE_DURATION = jnp.array(257)
    SPECIAL_TARGET_INACTIVE_DURATION = jnp.array(787)

    # Game layout constants
    BALL_SIZE = jnp.array([2, 4])
    FLIPPER_LEFT_POS = jnp.array([30, 180])
    FLIPPER_RIGHT_POS = jnp.array([110, 180])
    PLUNGER_POS = jnp.array([150, 120])
    PLUNGER_MAX_HEIGHT = jnp.array(20)  # Taken from RAM values (67-87)
    INVISIBLE_BLOCK_MEAN_REFLECTION_FACTOR = jnp.array(
        0.001
    )  # 8 times the plunger power is added to ball_vel_x

    # Background color and object colors
    BG_COLOR = jnp.array([0, 0, 0], dtype=jnp.uint8)
    TILT_MODE_COLOR = jnp.array([167, 26, 26], dtype=jnp.uint8)
    BACKGROUND_COLOR = jnp.array([0, 0, 0], dtype=jnp.uint8)
    BOUNDARY_COLOR = jnp.array([0, 255, 0], dtype=jnp.uint8)
    WALL_COLOR = jnp.array([104, 72, 198], dtype=jnp.uint8)
    GROUP3_COLOR = jnp.array([187, 159, 71], dtype=jnp.uint8)
    GROUP4_COLOR = jnp.array([210, 164, 74], dtype=jnp.uint8)
    GROUP5_COLOR = jnp.array([236, 236, 236], dtype=jnp.uint8)

    # Color cycling arrays (each inner tuple converted to a list with alpha)
    BACKGROUND_COLOR_CYCLING = jnp.array(
        [
            [74, 74, 74],
            [111, 111, 111],
            [142, 142, 142],
            [170, 170, 170],
            [192, 192, 192],
            [214, 214, 214],
            [236, 236, 236],
            [72, 72, 0],
        ],
        dtype=jnp.uint8,
    )

    WALL_COLOR_CYCLING = jnp.array(
        [
            [78, 50, 181],
            [51, 26, 163],
            [20, 0, 144],
            [188, 144, 252],
            [169, 128, 240],
            [149, 111, 227],
            [127, 92, 213],
            [146, 70, 192],
        ],
        dtype=jnp.uint8,
    )

    GROUP3_COLOR_CYCLING = jnp.array(
        [
            [210, 182, 86],
            [232, 204, 99],
            [252, 224, 112],
            [72, 44, 0],
            [105, 77, 20],
            [134, 106, 38],
            [162, 134, 56],
            [160, 171, 79],
        ],
        dtype=jnp.uint8,
    )

    GROUP4_COLOR_CYCLING = jnp.array(
        [
            [195, 144, 61],
            [236, 200, 96],
            [223, 183, 85],
            [144, 72, 17],
            [124, 44, 0],
            [180, 122, 48],
            [162, 98, 33],
            [227, 151, 89],
        ],
        dtype=jnp.uint8,
    )

    GROUP5_COLOR_CYCLING = jnp.array(
        [
            [214, 214, 214],
            [192, 192, 192],
            [170, 170, 170],
            [142, 142, 142],
            [111, 111, 111],
            [74, 74, 74],
            [0, 0, 0],
            [252, 252, 84],
        ],
        dtype=jnp.uint8,
    )

    # Pygame window dimensions
    WINDOW_WIDTH = jnp.array(160) * 3
    WINDOW_HEIGHT = jnp.array(210) * 3

    # Objects (walls etc.) Positions/dimensions
    BALL_START_X = jnp.array(149.0)
    BALL_START_Y = jnp.array(129.0)
    BALL_START_DIRECTION = jnp.array(0)

    GAME_BOTTOM_Y = jnp.array(191)

    TOP_WALL_LEFT_X_OFFSET = jnp.array(0)
    TOP_WALL_TOP_Y_OFFSET = jnp.array(16)
    RIGHT_WALL_LEFT_X_OFFSET = jnp.array(152)
    BOTTOM_WALL_LEFT_X_OFFSET = jnp.array(12)
    BOTTOM_WALL_TOP_Y_OFFSET = jnp.array(184)

    INVISIBLE_BLOCK_LEFT_X_OFFSET = jnp.array(149)
    INVISIBLE_BLOCK_TOP_Y_OFFSET = jnp.array(36)

    INNER_WALL_TOP_Y_OFFSET = jnp.array(56)
    LEFT_INNER_WALL_TOP_X_OFFSET = jnp.array(12)
    RIGHT_INNER_WALL_TOP_X_OFFSET = jnp.array(144)

    QUADRUPLE_STEP_Y_OFFSET = jnp.array(152)
    TRIPLE_STEP_Y_OFFSET = jnp.array(160)
    DOUBLE_STEP_Y_OFFSET = jnp.array(168)
    SINGLE_STEP_Y_OFFSET = jnp.array(176)

    LEFT_QUADRUPLE_STEP_X_OFFSET = jnp.array(16)
    RIGHT_QUADRUPLE_STEP_X_OFFSET = jnp.array(140)
    LEFT_TRIPLE_STEP_X_OFFSET = jnp.array(20)
    RIGHT_TRIPLE_STEP_X_OFFSET = jnp.array(136)
    LEFT_DOUBLE_STEP_X_OFFSET = jnp.array(24)
    RIGHT_DOUBLE_STEP_X_OFFSET = jnp.array(132)
    LEFT_SINGLE_STEP_X_OFFSET = jnp.array(28)
    RIGHT_SINGLE_STEP_X_OFFSET = jnp.array(128)

    OUTER_WALL_THICKNESS = jnp.array(8)
    INNER_WALL_THICKNESS = jnp.array(4)
    WALL_CORNER_BLOCK_WIDTH = jnp.array(4)
    WALL_CORNER_BLOCK_HEIGHT = jnp.array(8)
    STEP_HEIGHT = jnp.array(8)
    STEP_WIDTH = jnp.array(4)

    # Inner Objects Positions and Dimensions

    VERTICAL_BAR_HEIGHT = jnp.array(32)
    VERTICAL_BAR_WIDTH = jnp.array(4)

    ROLLOVER_BAR_DISTANCE = jnp.array(
        12
    )  # Distance between the left and right rollover bar

    BUMPER_WIDTH = jnp.array(16)
    BUMPER_HEIGHT = jnp.array(32)

    LEFT_COLUMN_X_OFFSET = jnp.array(40)
    MIDDLE_COLUMN_X_OFFSET = jnp.array(72)
    RIGHT_COLUMN_X_OFFSET = jnp.array(104)
    TOP_ROW_Y_OFFSET = jnp.array(48)
    MIDDLE_ROW_Y_OFFSET = jnp.array(112)
    BOTTOM_ROW_Y_OFFSET = jnp.array(177)

    MIDDLE_BAR_X = jnp.array(72)
    MIDDLE_BAR_Y = jnp.array(104)
    MIDDLE_BAR_WIDTH = jnp.array(16)
    MIDDLE_BAR_HEIGHT = jnp.array(8)

    # Flipper Parts Positions and Dimensions

    # Flipper bounding boxes - these are a special case as they are not used for AABB collision,
    # but for their own special kind of collision
    FLIPPER_LEFT_PIVOT_X = jnp.array(64)
    FLIPPER_RIGHT_PIVOT_X = jnp.array(96)  # pixel + 1 to not leave a gap
    FLIPPER_PIVOT_Y_TOP = jnp.array(184)
    FLIPPER_PIVOT_Y_BOT = jnp.array(190)  # pixel + 1

    FLIPPER_00_TOP_END_Y = jnp.array(190)
    FLIPPER_00_BOT_END_Y = jnp.array(192)  # pixel + 1
    FLIPPER_16_TOP_END_Y = jnp.array(186)
    FLIPPER_16_BOT_END_Y = jnp.array(188)  # pixel + 1
    FLIPPER_32_TOP_END_Y = jnp.array(181)
    FLIPPER_32_BOT_END_Y = jnp.array(183)  # pixel + 1
    FLIPPER_48_TOP_END_Y = jnp.array(177)
    FLIPPER_48_BOT_END_Y = jnp.array(179)  # pixel + 1

    FLIPPER_LEFT_00_END_X = FLIPPER_LEFT_16_END_X = FLIPPER_LEFT_32_END_X = jnp.array(
        77
    )  # pixel + 1
    FLIPPER_LEFT_48_END_X = jnp.array(76)  # pixel + 1
    FLIPPER_RIGHT_00_END_X = FLIPPER_RIGHT_16_END_X = FLIPPER_RIGHT_32_END_X = (
        jnp.array(83)
    )
    FLIPPER_RIGHT_48_END_X = jnp.array(84)

    # Spinner Bounding Boxes

    # Bounding boxes of the spinner if the joined (single larger) spinner part is on the top or bottom
    SPINNER_TOP_BOTTOM_LARGE_HEIGHT = jnp.array(2)
    SPINNER_TOP_BOTTOM_LARGE_WIDTH = jnp.array(2)
    SPINNER_TOP_BOTTOM_SMALL_HEIGHT = jnp.array(2)
    SPINNER_TOP_BOTTOM_SMALL_WIDTH = jnp.array(1)

    # Bounding boxes of the spinner if the joined (single larger) spinner part is on the left or right
    SPINNER_LEFT_RIGHT_LARGE_HEIGHT = jnp.array(2)
    SPINNER_LEFT_RIGHT_LARGE_WIDTH = jnp.array(1)
    SPINNER_LEFT_RIGHT_SMALL_HEIGHT = jnp.array(2)
    SPINNER_LEFT_RIGHT_SMALL_WIDTH = jnp.array(1)

    LEFT_SPINNER_MIDDLE_POINT = jnp.array(33), jnp.array(97)
    RIGHT_SPINNER_MIDDLE_POINT = jnp.array(129), jnp.array(97)

    # Instantiate a SceneObject like this:
    INVISIBLE_BLOCK_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(2),  # type: ignore
        hit_box_width=jnp.array(INNER_WALL_THICKNESS),  # type: ignore
        hit_box_x_offset=jnp.array(INVISIBLE_BLOCK_LEFT_X_OFFSET),  # type: ignore
        hit_box_y_offset=jnp.array(INVISIBLE_BLOCK_TOP_Y_OFFSET),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(0),  # type: ignore
        variant=jnp.array(-1),
    )

    # Wall Scene Objects

    TOP_WALL_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(OUTER_WALL_THICKNESS),  # type: ignore
        hit_box_width=jnp.array(160),  # type: ignore
        hit_box_x_offset=jnp.array(TOP_WALL_LEFT_X_OFFSET),  # type: ignore
        hit_box_y_offset=jnp.array(TOP_WALL_TOP_Y_OFFSET),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(0),  # type: ignore
        variant=jnp.array(-1),
    )
    BOTTOM_WALL_SCENE_OBJECT_1 = SceneObject(
        hit_box_height=jnp.array(OUTER_WALL_THICKNESS),  # type: ignore
        hit_box_width=jnp.array(24),  # type: ignore
        hit_box_x_offset=jnp.array(BOTTOM_WALL_LEFT_X_OFFSET),  # type: ignore
        hit_box_y_offset=jnp.array(BOTTOM_WALL_TOP_Y_OFFSET),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(0),  # type: ignore
        variant=jnp.array(-1),
    )
    BOTTOM_WALL_SCENE_OBJECT_2 = SceneObject(
        hit_box_height=jnp.array(OUTER_WALL_THICKNESS),  # type: ignore
        hit_box_width=jnp.array(24),  # type: ignore
        hit_box_x_offset=jnp.array(40),  # type: ignore
        hit_box_y_offset=jnp.array(BOTTOM_WALL_TOP_Y_OFFSET),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(0),  # type: ignore
        variant=jnp.array(-1),
    )

    BOTTOM_WALL_SCENE_OBJECT_3 = SceneObject(
        hit_box_height=jnp.array(OUTER_WALL_THICKNESS),  # type: ignore
        hit_box_width=jnp.array(24),  # type: ignore
        hit_box_x_offset=jnp.array(96),  # type: ignore
        hit_box_y_offset=jnp.array(BOTTOM_WALL_TOP_Y_OFFSET),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(0),  # type: ignore
        variant=jnp.array(-1),
    )
    BOTTOM_WALL_SCENE_OBJECT_4 = SceneObject(
        hit_box_height=jnp.array(OUTER_WALL_THICKNESS),  # type: ignore
        hit_box_width=jnp.array(24),  # type: ignore
        hit_box_x_offset=jnp.array(124),  # type: ignore
        hit_box_y_offset=jnp.array(BOTTOM_WALL_TOP_Y_OFFSET),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(0),  # type: ignore
        variant=jnp.array(-1),
    )

    TILT_MODE_HOLE_PLUG_LEFT = SceneObject(
        hit_box_height=jnp.array(OUTER_WALL_THICKNESS),  # type: ignore
        hit_box_width=jnp.array(4),  # type: ignore
        hit_box_x_offset=jnp.array(36),  # type: ignore
        hit_box_y_offset=jnp.array(BOTTOM_WALL_TOP_Y_OFFSET),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(11),  # type: ignore
        variant=jnp.array(-1),
    )

    TILT_MODE_HOLE_PLUG_RIGHT = SceneObject(
        hit_box_height=jnp.array(OUTER_WALL_THICKNESS),  # type: ignore
        hit_box_width=jnp.array(4),  # type: ignore
        hit_box_x_offset=jnp.array(120),  # type: ignore
        hit_box_y_offset=jnp.array(BOTTOM_WALL_TOP_Y_OFFSET),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(11),  # type: ignore
        variant=jnp.array(-1),
    )

    LEFT_WALL_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(176),  # type: ignore
        hit_box_width=jnp.array(OUTER_WALL_THICKNESS),  # type: ignore
        hit_box_x_offset=jnp.array(TOP_WALL_LEFT_X_OFFSET),  # type: ignore
        hit_box_y_offset=jnp.array(TOP_WALL_TOP_Y_OFFSET),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(0),  # type: ignore
        variant=jnp.array(-1),
    )
    RIGHT_WALL_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(176),  # type: ignore
        hit_box_width=jnp.array(OUTER_WALL_THICKNESS),  # type: ignore
        hit_box_x_offset=jnp.array(RIGHT_WALL_LEFT_X_OFFSET),  # type: ignore
        hit_box_y_offset=jnp.array(TOP_WALL_TOP_Y_OFFSET),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(0),  # type: ignore
        variant=jnp.array(-1),
    )

    LEFT_INNER_WALL_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(135),  # type: ignore
        hit_box_width=jnp.array(INNER_WALL_THICKNESS),  # type: ignore
        hit_box_x_offset=jnp.array(LEFT_INNER_WALL_TOP_X_OFFSET),  # type: ignore
        hit_box_y_offset=jnp.array(INNER_WALL_TOP_Y_OFFSET),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(0),  # type: ignore
        variant=jnp.array(-1),
    )

    RIGHT_INNER_WALL_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(135),  # type: ignore
        hit_box_width=jnp.array(INNER_WALL_THICKNESS),  # type: ignore
        hit_box_x_offset=jnp.array(RIGHT_INNER_WALL_TOP_X_OFFSET),  # type: ignore
        hit_box_y_offset=jnp.array(INNER_WALL_TOP_Y_OFFSET),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(0),  # type: ignore
        variant=jnp.array(-1),
    )

    # Steps (Stairway left and right) Scene Objects

    LEFT_QUADRUPLE_STEP_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(4 * STEP_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(STEP_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(LEFT_QUADRUPLE_STEP_X_OFFSET),  # type: ignore
        hit_box_y_offset=jnp.array(QUADRUPLE_STEP_Y_OFFSET),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(0),  # type: ignore
        variant=jnp.array(-1),
    )
    RIGHT_QUADRUPLE_STEP_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(4 * STEP_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(STEP_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(RIGHT_QUADRUPLE_STEP_X_OFFSET),  # type: ignore
        hit_box_y_offset=jnp.array(QUADRUPLE_STEP_Y_OFFSET),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(0),  # type: ignore
        variant=jnp.array(-1),
    )
    LEFT_TRIPLE_STEP_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(3 * STEP_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(STEP_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(LEFT_TRIPLE_STEP_X_OFFSET),  # type: ignore
        hit_box_y_offset=jnp.array(TRIPLE_STEP_Y_OFFSET),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(0),  # type: ignore
        variant=jnp.array(-1),
    )
    RIGHT_TRIPLE_STEP_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(3 * STEP_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(STEP_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(RIGHT_TRIPLE_STEP_X_OFFSET),  # type: ignore
        hit_box_y_offset=jnp.array(TRIPLE_STEP_Y_OFFSET),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(0),  # type: ignore
        variant=jnp.array(-1),
    )
    LEFT_DOUBLE_STEP_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(2 * STEP_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(STEP_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(LEFT_DOUBLE_STEP_X_OFFSET),  # type: ignore
        hit_box_y_offset=jnp.array(DOUBLE_STEP_Y_OFFSET),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(0),  # type: ignore
        variant=jnp.array(-1),
    )
    RIGHT_DOUBLE_STEP_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(2 * STEP_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(STEP_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(RIGHT_DOUBLE_STEP_X_OFFSET),  # type: ignore
        hit_box_y_offset=jnp.array(DOUBLE_STEP_Y_OFFSET),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(0),  # type: ignore
        variant=jnp.array(-1),
    )
    LEFT_SINGLE_STEP_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(1 * STEP_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(STEP_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(LEFT_SINGLE_STEP_X_OFFSET),  # type: ignore
        hit_box_y_offset=jnp.array(SINGLE_STEP_Y_OFFSET),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(0),  # type: ignore
        variant=jnp.array(-1),
    )
    RIGHT_SINGLE_STEP_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(1 * STEP_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(STEP_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(RIGHT_SINGLE_STEP_X_OFFSET),  # type: ignore
        hit_box_y_offset=jnp.array(SINGLE_STEP_Y_OFFSET),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(0),  # type: ignore
        variant=jnp.array(-1),
    )
    TOP_LEFT_STEP_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(1 * STEP_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(1 * STEP_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(8),  # type: ignore
        hit_box_y_offset=jnp.array(24),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(0),  # type: ignore
        variant=jnp.array(-1),
    )
    TOP_RIGHT_STEP_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(1 * STEP_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(1 * STEP_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(148),  # type: ignore
        hit_box_y_offset=jnp.array(24),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(0),  # type: ignore
        variant=jnp.array(-1),
    )

    # Rollover Bars Scene Objects

    LEFT_ROLLOVER_LEFT_BAR_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(VERTICAL_BAR_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(VERTICAL_BAR_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(LEFT_COLUMN_X_OFFSET),  # type: ignore
        hit_box_y_offset=jnp.array(TOP_ROW_Y_OFFSET),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(0),  # type: ignore
        variant=jnp.array(-1),
    )

    LEFT_ROLLOVER_RIGHT_BAR_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(VERTICAL_BAR_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(VERTICAL_BAR_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(LEFT_COLUMN_X_OFFSET + ROLLOVER_BAR_DISTANCE),  # type: ignore
        hit_box_y_offset=jnp.array(TOP_ROW_Y_OFFSET),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(0),  # type: ignore
        variant=jnp.array(-1),
    )

    ATARI_ROLLOVER_LEFT_BAR_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(VERTICAL_BAR_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(VERTICAL_BAR_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(RIGHT_COLUMN_X_OFFSET),  # type: ignore
        hit_box_y_offset=jnp.array(TOP_ROW_Y_OFFSET),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(0),  # type: ignore
        variant=jnp.array(-1),
    )

    ATARI_ROLLOVER_RIGHT_BAR_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(VERTICAL_BAR_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(VERTICAL_BAR_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(RIGHT_COLUMN_X_OFFSET + ROLLOVER_BAR_DISTANCE),  # type: ignore
        hit_box_y_offset=jnp.array(TOP_ROW_Y_OFFSET),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(0),  # type: ignore
        variant=jnp.array(-1),
    )

    # Bumper Scene Objects

    TOP_BUMPER_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(BUMPER_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(BUMPER_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(MIDDLE_COLUMN_X_OFFSET),  # type: ignore
        hit_box_y_offset=jnp.array(TOP_ROW_Y_OFFSET),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(1),  # type: ignore
        variant=jnp.array(-1),
    )

    LEFT_BUMPER_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(BUMPER_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(BUMPER_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(LEFT_COLUMN_X_OFFSET),  # type: ignore
        hit_box_y_offset=jnp.array(MIDDLE_ROW_Y_OFFSET),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(1),  # type: ignore
        variant=jnp.array(-1),
    )
    RIGHT_BUMPER_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(BUMPER_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(BUMPER_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(RIGHT_COLUMN_X_OFFSET),  # type: ignore
        hit_box_y_offset=jnp.array(MIDDLE_ROW_Y_OFFSET),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(1),  # type: ignore
        variant=jnp.array(-1),
    )

    # Left Spinner Scene Objects

    # Left Spinner Bottom Position

    LEFT_SPINNER_BOTTOM_POSITION_JOINED_PART_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(SPINNER_TOP_BOTTOM_LARGE_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(SPINNER_TOP_BOTTOM_LARGE_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(32),  # type: ignore
        hit_box_y_offset=jnp.array(100),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(2),  # type: ignore
        variant=jnp.array(0),
    )

    LEFT_SPINNER_BOTTOM_POSITION_LEFT_PART_1_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(SPINNER_TOP_BOTTOM_SMALL_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(SPINNER_TOP_BOTTOM_SMALL_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(30),  # type: ignore
        hit_box_y_offset=jnp.array(94),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(2),  # type: ignore
        variant=jnp.array(0),
    )

    LEFT_SPINNER_BOTTOM_POSITION_LEFT_PART_2_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(SPINNER_TOP_BOTTOM_SMALL_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(SPINNER_TOP_BOTTOM_SMALL_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(31),  # type: ignore
        hit_box_y_offset=jnp.array(92),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(2),  # type: ignore
        variant=jnp.array(0),
    )

    LEFT_SPINNER_BOTTOM_POSITION_RIGHT_PART_1_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(SPINNER_TOP_BOTTOM_SMALL_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(SPINNER_TOP_BOTTOM_SMALL_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(35),  # type: ignore
        hit_box_y_offset=jnp.array(94),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(2),  # type: ignore
        variant=jnp.array(0),
    )

    LEFT_SPINNER_BOTTOM_POSITION_RIGHT_PART_2_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(SPINNER_TOP_BOTTOM_SMALL_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(SPINNER_TOP_BOTTOM_SMALL_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(34),  # type: ignore
        hit_box_y_offset=jnp.array(92),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(2),  # type: ignore
        variant=jnp.array(0),
    )

    # Left Spinner Top Position
    LEFT_SPINNER_TOP_POSITION_JOINED_PART_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(SPINNER_TOP_BOTTOM_LARGE_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(SPINNER_TOP_BOTTOM_LARGE_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(32),  # type: ignore
        hit_box_y_offset=jnp.array(90),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(2),  # type: ignore
        variant=jnp.array(1),
    )
    LEFT_SPINNER_TOP_POSITION_LEFT_PART_1_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(SPINNER_TOP_BOTTOM_SMALL_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(SPINNER_TOP_BOTTOM_SMALL_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(30),  # type: ignore
        hit_box_y_offset=jnp.array(96),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(2),  # type: ignore
        variant=jnp.array(1),
    )

    LEFT_SPINNER_TOP_POSITION_LEFT_PART_2_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(SPINNER_TOP_BOTTOM_SMALL_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(SPINNER_TOP_BOTTOM_SMALL_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(31),  # type: ignore
        hit_box_y_offset=jnp.array(98),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(2),  # type: ignore
        variant=jnp.array(1),
    )

    LEFT_SPINNER_TOP_POSITION_RIGHT_PART_1_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(SPINNER_TOP_BOTTOM_SMALL_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(SPINNER_TOP_BOTTOM_SMALL_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(35),  # type: ignore
        hit_box_y_offset=jnp.array(96),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(2),  # type: ignore
        variant=jnp.array(1),
    )

    LEFT_SPINNER_TOP_POSITION_RIGHT_PART_2_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(SPINNER_TOP_BOTTOM_SMALL_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(SPINNER_TOP_BOTTOM_SMALL_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(34),  # type: ignore
        hit_box_y_offset=jnp.array(98),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(2),  # type: ignore
        variant=jnp.array(1),
    )

    # Left Spinner Left Position
    LEFT_SPINNER_LEFT_POSITION_JOINED_PART_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(SPINNER_LEFT_RIGHT_LARGE_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(SPINNER_LEFT_RIGHT_LARGE_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(30),  # type: ignore
        hit_box_y_offset=jnp.array(94),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(2),  # type: ignore
        variant=jnp.array(2),
    )

    LEFT_SPINNER_LEFT_POSITION_LEFT_PART_1_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(SPINNER_LEFT_RIGHT_SMALL_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(SPINNER_LEFT_RIGHT_SMALL_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(33),  # type: ignore
        hit_box_y_offset=jnp.array(90),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(2),  # type: ignore
        variant=jnp.array(2),
    )

    LEFT_SPINNER_LEFT_POSITION_LEFT_PART_2_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(SPINNER_LEFT_RIGHT_SMALL_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(SPINNER_LEFT_RIGHT_SMALL_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(34),  # type: ignore
        hit_box_y_offset=jnp.array(92),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(2),  # type: ignore
        variant=jnp.array(2),
    )

    LEFT_SPINNER_LEFT_POSITION_RIGHT_PART_1_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(SPINNER_LEFT_RIGHT_SMALL_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(SPINNER_LEFT_RIGHT_SMALL_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(33),  # type: ignore
        hit_box_y_offset=jnp.array(100),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(2),  # type: ignore
        variant=jnp.array(2),
    )

    LEFT_SPINNER_LEFT_POSITION_RIGHT_PART_2_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(SPINNER_LEFT_RIGHT_SMALL_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(SPINNER_LEFT_RIGHT_SMALL_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(34),  # type: ignore
        hit_box_y_offset=jnp.array(98),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(2),  # type: ignore
        variant=jnp.array(2),
    )

    # Left Spinner Right Position
    LEFT_SPINNER_RIGHT_POSITION_JOINED_PART_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(SPINNER_LEFT_RIGHT_LARGE_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(SPINNER_LEFT_RIGHT_LARGE_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(35),  # type: ignore
        hit_box_y_offset=jnp.array(94),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(2),  # type: ignore
        variant=jnp.array(3),
    )

    LEFT_SPINNER_RIGHT_POSITION_LEFT_PART_1_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(SPINNER_LEFT_RIGHT_SMALL_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(SPINNER_LEFT_RIGHT_SMALL_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(32),  # type: ignore
        hit_box_y_offset=jnp.array(100),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(2),  # type: ignore
        variant=jnp.array(3),
    )

    LEFT_SPINNER_RIGHT_POSITION_LEFT_PART_2_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(SPINNER_LEFT_RIGHT_SMALL_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(SPINNER_LEFT_RIGHT_SMALL_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(31),  # type: ignore
        hit_box_y_offset=jnp.array(98),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(2),  # type: ignore
        variant=jnp.array(3),
    )

    LEFT_SPINNER_RIGHT_POSITION_RIGHT_PART_1_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(SPINNER_LEFT_RIGHT_SMALL_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(SPINNER_LEFT_RIGHT_SMALL_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(32),  # type: ignore
        hit_box_y_offset=jnp.array(90),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(2),  # type: ignore
        variant=jnp.array(3),
    )

    LEFT_SPINNER_RIGHT_POSITION_RIGHT_PART_2_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(SPINNER_LEFT_RIGHT_SMALL_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(SPINNER_LEFT_RIGHT_SMALL_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(31),  # type: ignore
        hit_box_y_offset=jnp.array(92),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(2),  # type: ignore
        variant=jnp.array(3),
    )

    # Right Spinner Scene Objects

    # Right Spinner Bottom Position
    RIGHT_SPINNER_BOTTOM_POSITION_JOINED_PART_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(SPINNER_TOP_BOTTOM_LARGE_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(SPINNER_TOP_BOTTOM_LARGE_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(128),  # type: ignore
        hit_box_y_offset=jnp.array(100),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(2),  # type: ignore
        variant=jnp.array(0),
    )

    RIGHT_SPINNER_BOTTOM_POSITION_LEFT_PART_1_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(SPINNER_TOP_BOTTOM_SMALL_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(SPINNER_TOP_BOTTOM_SMALL_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(126),  # type: ignore
        hit_box_y_offset=jnp.array(94),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(2),  # type: ignore
        variant=jnp.array(0),
    )

    RIGHT_SPINNER_BOTTOM_POSITION_LEFT_PART_2_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(SPINNER_TOP_BOTTOM_SMALL_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(SPINNER_TOP_BOTTOM_SMALL_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(127),  # type: ignore
        hit_box_y_offset=jnp.array(92),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(2),  # type: ignore
        variant=jnp.array(0),
    )

    RIGHT_SPINNER_BOTTOM_POSITION_RIGHT_PART_1_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(SPINNER_TOP_BOTTOM_SMALL_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(SPINNER_TOP_BOTTOM_SMALL_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(131),  # type: ignore
        hit_box_y_offset=jnp.array(94),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(2),  # type: ignore
        variant=jnp.array(0),
    )

    RIGHT_SPINNER_BOTTOM_POSITION_RIGHT_PART_2_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(SPINNER_TOP_BOTTOM_SMALL_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(SPINNER_TOP_BOTTOM_SMALL_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(130),  # type: ignore
        hit_box_y_offset=jnp.array(92),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(2),  # type: ignore
        variant=jnp.array(0),
    )

    # Right Spinner Top Position
    RIGHT_SPINNER_TOP_POSITION_JOINED_PART_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(SPINNER_TOP_BOTTOM_LARGE_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(SPINNER_TOP_BOTTOM_LARGE_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(128),  # type: ignore
        hit_box_y_offset=jnp.array(90),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(2),  # type: ignore
        variant=jnp.array(1),
    )

    RIGHT_SPINNER_TOP_POSITION_LEFT_PART_1_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(SPINNER_TOP_BOTTOM_SMALL_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(SPINNER_TOP_BOTTOM_SMALL_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(131),  # type: ignore
        hit_box_y_offset=jnp.array(96),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(2),  # type: ignore
        variant=jnp.array(1),
    )

    RIGHT_SPINNER_TOP_POSITION_LEFT_PART_2_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(SPINNER_TOP_BOTTOM_SMALL_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(SPINNER_TOP_BOTTOM_SMALL_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(130),  # type: ignore
        hit_box_y_offset=jnp.array(98),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(2),  # type: ignore
        variant=jnp.array(1),
    )

    RIGHT_SPINNER_TOP_POSITION_RIGHT_PART_1_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(SPINNER_TOP_BOTTOM_SMALL_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(SPINNER_TOP_BOTTOM_SMALL_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(126),  # type: ignore
        hit_box_y_offset=jnp.array(96),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(2),  # type: ignore
        variant=jnp.array(1),
    )

    RIGHT_SPINNER_TOP_POSITION_RIGHT_PART_2_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(SPINNER_TOP_BOTTOM_SMALL_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(SPINNER_TOP_BOTTOM_SMALL_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(127),  # type: ignore
        hit_box_y_offset=jnp.array(98),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(2),  # type: ignore
        variant=jnp.array(1),
    )

    # Right Spinner Left Position
    RIGHT_SPINNER_LEFT_POSITION_JOINED_PART_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(SPINNER_LEFT_RIGHT_LARGE_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(SPINNER_LEFT_RIGHT_LARGE_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(126),  # type: ignore
        hit_box_y_offset=jnp.array(94),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(2),  # type: ignore
        variant=jnp.array(2),
    )

    RIGHT_SPINNER_LEFT_POSITION_LEFT_PART_1_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(SPINNER_LEFT_RIGHT_SMALL_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(SPINNER_LEFT_RIGHT_SMALL_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(129),  # type: ignore
        hit_box_y_offset=jnp.array(90),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(2),  # type: ignore
        variant=jnp.array(2),
    )

    RIGHT_SPINNER_LEFT_POSITION_LEFT_PART_2_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(SPINNER_LEFT_RIGHT_SMALL_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(SPINNER_LEFT_RIGHT_SMALL_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(130),  # type: ignore
        hit_box_y_offset=jnp.array(92),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(2),  # type: ignore
        variant=jnp.array(2),
    )

    RIGHT_SPINNER_LEFT_POSITION_RIGHT_PART_1_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(SPINNER_LEFT_RIGHT_SMALL_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(SPINNER_LEFT_RIGHT_SMALL_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(129),  # type: ignore
        hit_box_y_offset=jnp.array(100),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(2),  # type: ignore
        variant=jnp.array(2),
    )

    RIGHT_SPINNER_LEFT_POSITION_RIGHT_PART_2_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(SPINNER_LEFT_RIGHT_SMALL_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(SPINNER_LEFT_RIGHT_SMALL_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(130),  # type: ignore
        hit_box_y_offset=jnp.array(98),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(2),  # type: ignore
        variant=jnp.array(2),
    )

    # Right Spinner Right Position
    RIGHT_SPINNER_RIGHT_POSITION_JOINED_PART_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(SPINNER_LEFT_RIGHT_LARGE_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(SPINNER_LEFT_RIGHT_LARGE_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(131),  # type: ignore
        hit_box_y_offset=jnp.array(94),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(2),  # type: ignore
        variant=jnp.array(3),
    )

    RIGHT_SPINNER_RIGHT_POSITION_LEFT_PART_1_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(SPINNER_LEFT_RIGHT_SMALL_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(SPINNER_LEFT_RIGHT_SMALL_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(128),  # type: ignore
        hit_box_y_offset=jnp.array(100),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(2),  # type: ignore
        variant=jnp.array(3),
    )

    RIGHT_SPINNER_RIGHT_POSITION_LEFT_PART_2_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(SPINNER_LEFT_RIGHT_SMALL_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(SPINNER_LEFT_RIGHT_SMALL_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(127),  # type: ignore
        hit_box_y_offset=jnp.array(98),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(2),  # type: ignore
        variant=jnp.array(3),
    )

    RIGHT_SPINNER_RIGHT_POSITION_RIGHT_PART_1_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(SPINNER_LEFT_RIGHT_SMALL_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(SPINNER_LEFT_RIGHT_SMALL_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(128),  # type: ignore
        hit_box_y_offset=jnp.array(90),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(2),  # type: ignore
        variant=jnp.array(3),
    )

    RIGHT_SPINNER_RIGHT_POSITION_RIGHT_PART_2_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(SPINNER_LEFT_RIGHT_SMALL_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(SPINNER_LEFT_RIGHT_SMALL_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(127),  # type: ignore
        hit_box_y_offset=jnp.array(92),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(2),  # type: ignore
        variant=jnp.array(3),
    )

    DIAMOND_VERTICAL_RECTANGLE_BOUNDING_BOX_HEIGHT = jnp.array(10)
    DIAMOND_VERTICAL_RECTANGLE_BOUNDING_BOX_WIDTH = jnp.array(3)
    DIAMOND_HORIZONTAL_RECTANGLE_BOUNDING_BOX_HEIGHT = jnp.array(6)
    DIAMOND_HORIZONTAL_RECTANGLE_BOUNDING_BOX_WIDTH = jnp.array(5)
    DIAMOND_SMALL_RECTANGLE_BOUNDING_BOX_HEIGHT = jnp.array(2)
    DIAMOND_SMALL_RECTANGLE_BOUNDING_BOX_WIDTH = jnp.array(1)

    LEFT_LIT_UP_TARGET_LARGE_VERTICAL_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(DIAMOND_VERTICAL_RECTANGLE_BOUNDING_BOX_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(DIAMOND_VERTICAL_RECTANGLE_BOUNDING_BOX_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(62),  # type: ignore
        hit_box_y_offset=jnp.array(26),  # type: ignore
        reflecting=jnp.array(0),  # type: ignore
        score_type=jnp.array(6),  # type: ignore
        variant=jnp.array(0),
    )

    LEFT_LIT_UP_TARGET_LARGE_HORIZONTAL_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(DIAMOND_HORIZONTAL_RECTANGLE_BOUNDING_BOX_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(DIAMOND_HORIZONTAL_RECTANGLE_BOUNDING_BOX_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(61),  # type: ignore
        hit_box_y_offset=jnp.array(28),  # type: ignore
        reflecting=jnp.array(0),  # type: ignore
        score_type=jnp.array(6),  # type: ignore
        variant=jnp.array(0),
    )

    MIDDLE_LIT_UP_TARGET_LARGE_VERTICAL_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(DIAMOND_VERTICAL_RECTANGLE_BOUNDING_BOX_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(DIAMOND_VERTICAL_RECTANGLE_BOUNDING_BOX_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(78),  # type: ignore
        hit_box_y_offset=jnp.array(26),  # type: ignore
        reflecting=jnp.array(0),  # type: ignore
        score_type=jnp.array(7),  # type: ignore
        variant=jnp.array(1),
    )

    MIDDLE_LIT_UP_TARGET_LARGE_HORIZONTAL_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(DIAMOND_HORIZONTAL_RECTANGLE_BOUNDING_BOX_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(DIAMOND_HORIZONTAL_RECTANGLE_BOUNDING_BOX_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(77),  # type: ignore
        hit_box_y_offset=jnp.array(28),  # type: ignore
        reflecting=jnp.array(0),  # type: ignore
        score_type=jnp.array(7),  # type: ignore
        variant=jnp.array(1),
    )

    RIGHT_LIT_UP_TARGET_LARGE_VERTICAL_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(DIAMOND_VERTICAL_RECTANGLE_BOUNDING_BOX_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(DIAMOND_VERTICAL_RECTANGLE_BOUNDING_BOX_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(94),  # type: ignore
        hit_box_y_offset=jnp.array(26),  # type: ignore
        reflecting=jnp.array(0),  # type: ignore
        score_type=jnp.array(8),  # type: ignore
        variant=jnp.array(2),
    )

    RIGHT_LIT_UP_TARGET_LARGE_HORIZONTAL_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(DIAMOND_HORIZONTAL_RECTANGLE_BOUNDING_BOX_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(DIAMOND_HORIZONTAL_RECTANGLE_BOUNDING_BOX_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(93),  # type: ignore
        hit_box_y_offset=jnp.array(28),  # type: ignore
        reflecting=jnp.array(0),  # type: ignore
        score_type=jnp.array(8),  # type: ignore
        variant=jnp.array(2),
    )

    SPECIAL_LIT_UP_TARGET_LARGE_VERTICAL_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(DIAMOND_VERTICAL_RECTANGLE_BOUNDING_BOX_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(DIAMOND_VERTICAL_RECTANGLE_BOUNDING_BOX_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(79),  # type: ignore
        hit_box_y_offset=jnp.array(122),  # type: ignore
        reflecting=jnp.array(0),  # type: ignore
        score_type=jnp.array(5),  # type: ignore
        variant=jnp.array(3),
    )

    SPECIAL_LIT_UP_TARGET_LARGE_HORIZONTAL_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(DIAMOND_HORIZONTAL_RECTANGLE_BOUNDING_BOX_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(DIAMOND_HORIZONTAL_RECTANGLE_BOUNDING_BOX_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(78),  # type: ignore
        hit_box_y_offset=jnp.array(124),  # type: ignore
        reflecting=jnp.array(0),  # type: ignore
        score_type=jnp.array(5),  # type: ignore
        variant=jnp.array(3),
    )

    LEFT_ROLLOVER_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(12),  # type: ignore
        hit_box_width=jnp.array(10),  # type: ignore
        hit_box_x_offset=jnp.array(43),  # type: ignore
        hit_box_y_offset=jnp.array(58),  # type: ignore
        reflecting=jnp.array(0),  # type: ignore
        score_type=jnp.array(3),  # type: ignore
        variant=jnp.array(-1),
    )

    ATARI_ROLLOVER_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(12),  # type: ignore
        hit_box_width=jnp.array(10),  # type: ignore
        hit_box_x_offset=jnp.array(107),  # type: ignore
        hit_box_y_offset=jnp.array(58),  # type: ignore
        reflecting=jnp.array(0),  # type: ignore
        score_type=jnp.array(4),  # type: ignore
        variant=jnp.array(-1),
    )

    # Middle Bar Scene Object

    MIDDLE_BAR_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.array(MIDDLE_BAR_HEIGHT),  # type: ignore
        hit_box_width=jnp.array(MIDDLE_BAR_WIDTH),  # type: ignore
        hit_box_x_offset=jnp.array(MIDDLE_BAR_X),  # type: ignore
        hit_box_y_offset=jnp.array(MIDDLE_BAR_Y),  # type: ignore
        reflecting=jnp.array(1),  # type: ignore
        score_type=jnp.array(0),  # type: ignore
        variant=jnp.array(-1),
    )

    LEFT_FLIPPER_00_BOT_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.abs(FLIPPER_PIVOT_Y_BOT - FLIPPER_00_BOT_END_Y),
        hit_box_width=jnp.abs(FLIPPER_LEFT_00_END_X - FLIPPER_LEFT_PIVOT_X),
        hit_box_x_offset=FLIPPER_LEFT_PIVOT_X,
        hit_box_y_offset=jnp.min(jnp.array([FLIPPER_PIVOT_Y_BOT, FLIPPER_00_BOT_END_Y])),
        reflecting=jnp.array(1),
        score_type=jnp.array(9),
        variant=jnp.array(0),
    )
    LEFT_FLIPPER_16_BOT_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.abs(FLIPPER_PIVOT_Y_BOT - FLIPPER_16_BOT_END_Y),
        hit_box_width=jnp.abs(FLIPPER_LEFT_16_END_X - FLIPPER_LEFT_PIVOT_X),
        hit_box_x_offset=FLIPPER_LEFT_PIVOT_X,
        hit_box_y_offset=jnp.min(jnp.array([FLIPPER_PIVOT_Y_BOT, FLIPPER_16_BOT_END_Y])),
        reflecting=jnp.array(1),
        score_type=jnp.array(9),
        variant=jnp.array(1),
    )
    LEFT_FLIPPER_32_BOT_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.abs(FLIPPER_PIVOT_Y_BOT - FLIPPER_32_BOT_END_Y),
        hit_box_width=jnp.abs(FLIPPER_LEFT_32_END_X - FLIPPER_LEFT_PIVOT_X),
        hit_box_x_offset=FLIPPER_LEFT_PIVOT_X,
        hit_box_y_offset=jnp.min(jnp.array([FLIPPER_PIVOT_Y_BOT, FLIPPER_32_BOT_END_Y])),
        reflecting=jnp.array(1),
        score_type=jnp.array(9),
        variant=jnp.array(2),
    )
    LEFT_FLIPPER_48_BOT_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.abs(FLIPPER_PIVOT_Y_BOT - FLIPPER_48_BOT_END_Y),
        hit_box_width=jnp.abs(FLIPPER_LEFT_48_END_X - FLIPPER_LEFT_PIVOT_X),
        hit_box_x_offset=FLIPPER_LEFT_PIVOT_X,
        hit_box_y_offset=jnp.min(jnp.array([FLIPPER_PIVOT_Y_BOT, FLIPPER_48_BOT_END_Y])),
        reflecting=jnp.array(1),
        score_type=jnp.array(9),
        variant=jnp.array(3),
    )

    LEFT_FLIPPER_00_TOP_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.abs(FLIPPER_PIVOT_Y_TOP - FLIPPER_00_TOP_END_Y),
        hit_box_width=jnp.abs(FLIPPER_LEFT_00_END_X - FLIPPER_LEFT_PIVOT_X),
        hit_box_x_offset=FLIPPER_LEFT_PIVOT_X,
        hit_box_y_offset=jnp.min(jnp.array([FLIPPER_PIVOT_Y_TOP, FLIPPER_00_TOP_END_Y])),
        reflecting=jnp.array(1),
        score_type=jnp.array(9),
        variant=jnp.array(4),
    )
    LEFT_FLIPPER_16_TOP_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.abs(FLIPPER_PIVOT_Y_TOP - FLIPPER_16_TOP_END_Y),
        hit_box_width=jnp.abs(FLIPPER_LEFT_16_END_X - FLIPPER_LEFT_PIVOT_X),
        hit_box_x_offset=FLIPPER_LEFT_PIVOT_X,
        hit_box_y_offset=jnp.min(jnp.array([FLIPPER_PIVOT_Y_TOP, FLIPPER_16_TOP_END_Y])),
        reflecting=jnp.array(1),
        score_type=jnp.array(9),
        variant=jnp.array(5),
    )
    LEFT_FLIPPER_32_TOP_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.abs(FLIPPER_PIVOT_Y_TOP - FLIPPER_32_TOP_END_Y),
        hit_box_width=jnp.abs(FLIPPER_LEFT_32_END_X - FLIPPER_LEFT_PIVOT_X),
        hit_box_x_offset=FLIPPER_LEFT_PIVOT_X,
        hit_box_y_offset=jnp.min(jnp.array([FLIPPER_PIVOT_Y_TOP, FLIPPER_32_TOP_END_Y])),
        reflecting=jnp.array(1),
        score_type=jnp.array(9),
        variant=jnp.array(6),
    )
    LEFT_FLIPPER_48_TOP_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.abs(FLIPPER_PIVOT_Y_TOP - FLIPPER_48_TOP_END_Y),
        hit_box_width=jnp.abs(FLIPPER_LEFT_48_END_X - FLIPPER_LEFT_PIVOT_X),
        hit_box_x_offset=FLIPPER_LEFT_PIVOT_X,
        hit_box_y_offset=jnp.min(jnp.array([FLIPPER_PIVOT_Y_TOP, FLIPPER_48_TOP_END_Y])),
        reflecting=jnp.array(1),
        score_type=jnp.array(9),
        variant=jnp.array(7),
    )

    RIGHT_FLIPPER_00_BOT_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.abs(FLIPPER_PIVOT_Y_BOT - FLIPPER_00_BOT_END_Y),
        hit_box_width=jnp.abs(FLIPPER_RIGHT_PIVOT_X - FLIPPER_RIGHT_00_END_X),
        hit_box_x_offset=jnp.min(jnp.array([FLIPPER_RIGHT_PIVOT_X, FLIPPER_RIGHT_00_END_X])),
        hit_box_y_offset=jnp.min(jnp.array([FLIPPER_PIVOT_Y_BOT, FLIPPER_00_BOT_END_Y])),
        reflecting=jnp.array(1),
        score_type=jnp.array(10),
        variant=jnp.array(8),
    )
    RIGHT_FLIPPER_16_BOT_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.abs(FLIPPER_PIVOT_Y_BOT - FLIPPER_16_BOT_END_Y),
        hit_box_width=jnp.abs(FLIPPER_RIGHT_PIVOT_X - FLIPPER_RIGHT_16_END_X),
        hit_box_x_offset=jnp.min(jnp.array([FLIPPER_RIGHT_PIVOT_X, FLIPPER_RIGHT_16_END_X])),
        hit_box_y_offset=jnp.min(jnp.array([FLIPPER_PIVOT_Y_BOT, FLIPPER_16_BOT_END_Y])),
        reflecting=jnp.array(1),
        score_type=jnp.array(10),
        variant=jnp.array(9),
    )
    RIGHT_FLIPPER_32_BOT_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.abs(FLIPPER_PIVOT_Y_BOT - FLIPPER_32_BOT_END_Y),
        hit_box_width=jnp.abs(FLIPPER_RIGHT_PIVOT_X - FLIPPER_RIGHT_32_END_X),
        hit_box_x_offset=jnp.min(jnp.array([FLIPPER_RIGHT_PIVOT_X, FLIPPER_RIGHT_32_END_X])),
        hit_box_y_offset=jnp.min(jnp.array([FLIPPER_PIVOT_Y_BOT, FLIPPER_32_BOT_END_Y])),
        reflecting=jnp.array(1),
        score_type=jnp.array(10),
        variant=jnp.array(10),
    )
    RIGHT_FLIPPER_48_BOT_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.abs(FLIPPER_PIVOT_Y_BOT - FLIPPER_48_BOT_END_Y),
        hit_box_width=jnp.abs(FLIPPER_RIGHT_PIVOT_X - FLIPPER_RIGHT_48_END_X),
        hit_box_x_offset=jnp.min(jnp.array([FLIPPER_RIGHT_PIVOT_X, FLIPPER_RIGHT_48_END_X])),
        hit_box_y_offset=jnp.min(jnp.array([FLIPPER_PIVOT_Y_BOT, FLIPPER_48_BOT_END_Y])),
        reflecting=jnp.array(1),
        score_type=jnp.array(10),
        variant=jnp.array(11),
    )

    RIGHT_FLIPPER_00_TOP_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.abs(FLIPPER_PIVOT_Y_TOP - FLIPPER_00_TOP_END_Y),
        hit_box_width=jnp.abs(FLIPPER_RIGHT_PIVOT_X - FLIPPER_RIGHT_00_END_X),
        hit_box_x_offset=jnp.min(jnp.array([FLIPPER_RIGHT_PIVOT_X, FLIPPER_RIGHT_00_END_X])),
        hit_box_y_offset=jnp.min(jnp.array([FLIPPER_PIVOT_Y_TOP, FLIPPER_00_TOP_END_Y])),
        reflecting=jnp.array(1),
        score_type=jnp.array(10),
        variant=jnp.array(12),
    )
    RIGHT_FLIPPER_16_TOP_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.abs(FLIPPER_PIVOT_Y_TOP - FLIPPER_16_TOP_END_Y),
        hit_box_width=jnp.abs(FLIPPER_RIGHT_PIVOT_X - FLIPPER_RIGHT_16_END_X),
        hit_box_x_offset=jnp.min(jnp.array([FLIPPER_RIGHT_PIVOT_X, FLIPPER_RIGHT_16_END_X])),
        hit_box_y_offset=jnp.min(jnp.array([FLIPPER_PIVOT_Y_TOP, FLIPPER_16_TOP_END_Y])),
        reflecting=jnp.array(1),
        score_type=jnp.array(10),
        variant=jnp.array(13),
    )
    RIGHT_FLIPPER_32_TOP_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.abs(FLIPPER_PIVOT_Y_TOP - FLIPPER_32_TOP_END_Y),
        hit_box_width=jnp.abs(FLIPPER_RIGHT_PIVOT_X - FLIPPER_RIGHT_32_END_X),
        hit_box_x_offset=jnp.min(jnp.array([FLIPPER_RIGHT_PIVOT_X, FLIPPER_RIGHT_32_END_X])),
        hit_box_y_offset=jnp.min(jnp.array([FLIPPER_PIVOT_Y_TOP, FLIPPER_32_TOP_END_Y])),
        reflecting=jnp.array(1),
        score_type=jnp.array(10),
        variant=jnp.array(14),
    )
    RIGHT_FLIPPER_48_TOP_SCENE_OBJECT = SceneObject(
        hit_box_height=jnp.abs(FLIPPER_PIVOT_Y_TOP - FLIPPER_48_TOP_END_Y),
        hit_box_width=jnp.abs(FLIPPER_RIGHT_PIVOT_X - FLIPPER_RIGHT_48_END_X),
        hit_box_x_offset=jnp.min(jnp.array([FLIPPER_RIGHT_PIVOT_X, FLIPPER_RIGHT_48_END_X])),
        hit_box_y_offset=jnp.min(jnp.array([FLIPPER_PIVOT_Y_TOP, FLIPPER_48_TOP_END_Y])),
        reflecting=jnp.array(1),
        score_type=jnp.array(10),
        variant=jnp.array(15),
    )

    _ALL_SCENE_OBJECTS_LIST = [
        # Non-reflective scene objects
        LEFT_LIT_UP_TARGET_LARGE_VERTICAL_SCENE_OBJECT,  # 0
        LEFT_LIT_UP_TARGET_LARGE_HORIZONTAL_SCENE_OBJECT,  # 1
        MIDDLE_LIT_UP_TARGET_LARGE_VERTICAL_SCENE_OBJECT,  # 2
        MIDDLE_LIT_UP_TARGET_LARGE_HORIZONTAL_SCENE_OBJECT,  # 3
        RIGHT_LIT_UP_TARGET_LARGE_VERTICAL_SCENE_OBJECT,  # 4
        RIGHT_LIT_UP_TARGET_LARGE_HORIZONTAL_SCENE_OBJECT,  # 5
        SPECIAL_LIT_UP_TARGET_LARGE_VERTICAL_SCENE_OBJECT,  # 6
        SPECIAL_LIT_UP_TARGET_LARGE_HORIZONTAL_SCENE_OBJECT,  # 7
        LEFT_ROLLOVER_SCENE_OBJECT,  # 8
        ATARI_ROLLOVER_SCENE_OBJECT,  # 9
        # Reflective scene objects
        TOP_WALL_SCENE_OBJECT,  # 10
        BOTTOM_WALL_SCENE_OBJECT_1,  # 11
        BOTTOM_WALL_SCENE_OBJECT_2,
        BOTTOM_WALL_SCENE_OBJECT_3,
        BOTTOM_WALL_SCENE_OBJECT_4,
        TILT_MODE_HOLE_PLUG_LEFT,
        TILT_MODE_HOLE_PLUG_RIGHT,
        LEFT_WALL_SCENE_OBJECT,  # 12
        RIGHT_WALL_SCENE_OBJECT,  # 13
        LEFT_INNER_WALL_SCENE_OBJECT,  # 14
        RIGHT_INNER_WALL_SCENE_OBJECT,  # 15
        LEFT_QUADRUPLE_STEP_SCENE_OBJECT,  # 16
        RIGHT_QUADRUPLE_STEP_SCENE_OBJECT,  # 17
        LEFT_TRIPLE_STEP_SCENE_OBJECT,  # 18
        RIGHT_TRIPLE_STEP_SCENE_OBJECT,  # 19
        LEFT_DOUBLE_STEP_SCENE_OBJECT,  # 20
        RIGHT_DOUBLE_STEP_SCENE_OBJECT,  # 21
        LEFT_SINGLE_STEP_SCENE_OBJECT,  # 22
        RIGHT_SINGLE_STEP_SCENE_OBJECT,  # 23
        TOP_LEFT_STEP_SCENE_OBJECT,  # 24
        TOP_RIGHT_STEP_SCENE_OBJECT,  # 25
        LEFT_ROLLOVER_LEFT_BAR_SCENE_OBJECT,  # 26
        LEFT_ROLLOVER_RIGHT_BAR_SCENE_OBJECT,  # 27
        ATARI_ROLLOVER_LEFT_BAR_SCENE_OBJECT,  # 28
        ATARI_ROLLOVER_RIGHT_BAR_SCENE_OBJECT,  # 29
        TOP_BUMPER_SCENE_OBJECT,  # 30
        LEFT_BUMPER_SCENE_OBJECT,  # 31
        RIGHT_BUMPER_SCENE_OBJECT,  # 32
        LEFT_SPINNER_BOTTOM_POSITION_JOINED_PART_SCENE_OBJECT,  # 33
        LEFT_SPINNER_BOTTOM_POSITION_LEFT_PART_1_SCENE_OBJECT,  # 34
        LEFT_SPINNER_BOTTOM_POSITION_LEFT_PART_2_SCENE_OBJECT,  # 35
        LEFT_SPINNER_BOTTOM_POSITION_RIGHT_PART_1_SCENE_OBJECT,  # 36
        LEFT_SPINNER_BOTTOM_POSITION_RIGHT_PART_2_SCENE_OBJECT,  # 37
        LEFT_SPINNER_RIGHT_POSITION_JOINED_PART_SCENE_OBJECT,  # 38
        LEFT_SPINNER_RIGHT_POSITION_LEFT_PART_1_SCENE_OBJECT,  # 39
        LEFT_SPINNER_RIGHT_POSITION_LEFT_PART_2_SCENE_OBJECT,  # 40
        LEFT_SPINNER_RIGHT_POSITION_RIGHT_PART_1_SCENE_OBJECT,  # 41
        LEFT_SPINNER_RIGHT_POSITION_RIGHT_PART_2_SCENE_OBJECT,  # 42
        LEFT_SPINNER_TOP_POSITION_JOINED_PART_SCENE_OBJECT,  # 43
        LEFT_SPINNER_TOP_POSITION_LEFT_PART_1_SCENE_OBJECT,  # 44
        LEFT_SPINNER_TOP_POSITION_LEFT_PART_2_SCENE_OBJECT,  # 45
        LEFT_SPINNER_TOP_POSITION_RIGHT_PART_1_SCENE_OBJECT,  # 46
        LEFT_SPINNER_TOP_POSITION_RIGHT_PART_2_SCENE_OBJECT,  # 47
        LEFT_SPINNER_LEFT_POSITION_JOINED_PART_SCENE_OBJECT,  # 48
        LEFT_SPINNER_LEFT_POSITION_LEFT_PART_1_SCENE_OBJECT,  # 49
        LEFT_SPINNER_LEFT_POSITION_LEFT_PART_2_SCENE_OBJECT,  # 50
        LEFT_SPINNER_LEFT_POSITION_RIGHT_PART_1_SCENE_OBJECT,  # 51
        LEFT_SPINNER_LEFT_POSITION_RIGHT_PART_2_SCENE_OBJECT,  # 52
        RIGHT_SPINNER_BOTTOM_POSITION_JOINED_PART_SCENE_OBJECT,  # 53
        RIGHT_SPINNER_BOTTOM_POSITION_LEFT_PART_1_SCENE_OBJECT,  # 54
        RIGHT_SPINNER_BOTTOM_POSITION_LEFT_PART_2_SCENE_OBJECT,  # 55
        RIGHT_SPINNER_BOTTOM_POSITION_RIGHT_PART_1_SCENE_OBJECT,  # 56
        RIGHT_SPINNER_BOTTOM_POSITION_RIGHT_PART_2_SCENE_OBJECT,  # 57
        RIGHT_SPINNER_RIGHT_POSITION_JOINED_PART_SCENE_OBJECT,  # 58
        RIGHT_SPINNER_RIGHT_POSITION_LEFT_PART_1_SCENE_OBJECT,  # 59
        RIGHT_SPINNER_RIGHT_POSITION_LEFT_PART_2_SCENE_OBJECT,  # 60
        RIGHT_SPINNER_RIGHT_POSITION_RIGHT_PART_1_SCENE_OBJECT,  # 61
        RIGHT_SPINNER_RIGHT_POSITION_RIGHT_PART_2_SCENE_OBJECT,  # 62
        RIGHT_SPINNER_TOP_POSITION_JOINED_PART_SCENE_OBJECT,  # 63
        RIGHT_SPINNER_TOP_POSITION_LEFT_PART_1_SCENE_OBJECT,  # 64
        RIGHT_SPINNER_TOP_POSITION_LEFT_PART_2_SCENE_OBJECT,  # 65
        RIGHT_SPINNER_TOP_POSITION_RIGHT_PART_1_SCENE_OBJECT,  # 66
        RIGHT_SPINNER_TOP_POSITION_RIGHT_PART_2_SCENE_OBJECT,  # 67
        RIGHT_SPINNER_LEFT_POSITION_JOINED_PART_SCENE_OBJECT,  # 68
        RIGHT_SPINNER_LEFT_POSITION_LEFT_PART_1_SCENE_OBJECT,  # 69
        RIGHT_SPINNER_LEFT_POSITION_LEFT_PART_2_SCENE_OBJECT,  # 70
        RIGHT_SPINNER_LEFT_POSITION_RIGHT_PART_1_SCENE_OBJECT,  # 71
        RIGHT_SPINNER_LEFT_POSITION_RIGHT_PART_2_SCENE_OBJECT,  # 72
        MIDDLE_BAR_SCENE_OBJECT,  # 73
        LEFT_FLIPPER_00_BOT_SCENE_OBJECT,
        LEFT_FLIPPER_00_TOP_SCENE_OBJECT,
        LEFT_FLIPPER_16_BOT_SCENE_OBJECT,
        LEFT_FLIPPER_16_TOP_SCENE_OBJECT,
        LEFT_FLIPPER_32_BOT_SCENE_OBJECT,
        LEFT_FLIPPER_32_TOP_SCENE_OBJECT,
        LEFT_FLIPPER_48_BOT_SCENE_OBJECT,
        LEFT_FLIPPER_48_TOP_SCENE_OBJECT,
        RIGHT_FLIPPER_00_BOT_SCENE_OBJECT,
        RIGHT_FLIPPER_00_TOP_SCENE_OBJECT,
        RIGHT_FLIPPER_16_BOT_SCENE_OBJECT,
        RIGHT_FLIPPER_16_TOP_SCENE_OBJECT,
        RIGHT_FLIPPER_32_BOT_SCENE_OBJECT,
        RIGHT_FLIPPER_32_TOP_SCENE_OBJECT,
        RIGHT_FLIPPER_48_BOT_SCENE_OBJECT,
        RIGHT_FLIPPER_48_TOP_SCENE_OBJECT,
    ]

    # For reference
    # class SceneObject:
    #     hit_box_width
    #     hit_box_height
    #     hit_box_x_offset
    #     hit_box_y_offset
    #     reflecting
    #     score_type
    #     variant

    REFLECTING_SCENE_OBJECTS = jnp.stack(
        [
            jnp.array(
                [
                    scene_object.hit_box_width,
                    scene_object.hit_box_height,
                    scene_object.hit_box_x_offset,
                    scene_object.hit_box_y_offset,
                    scene_object.reflecting,
                    scene_object.score_type,
                    scene_object.variant,
                ],
                dtype=jnp.int32,
            )
            for scene_object in _ALL_SCENE_OBJECTS_LIST
            if scene_object.reflecting == 1
        ]
    ).squeeze()
    NON_REFLECTING_SCENE_OBJECTS = jnp.stack(
        [
            jnp.array(
                [
                    scene_object.hit_box_width,
                    scene_object.hit_box_height,
                    scene_object.hit_box_x_offset,
                    scene_object.hit_box_y_offset,
                    scene_object.reflecting,
                    scene_object.score_type,
                    scene_object.variant,
                ],
                dtype=jnp.int32,
            )
            for scene_object in _ALL_SCENE_OBJECTS_LIST
            if scene_object.reflecting == 0
        ]
    ).squeeze()
    _FLIPPERS_SORTED = (
        # sorted in a way that allows finding/identifying
        # the corresponding scene object by index
        LEFT_FLIPPER_00_BOT_SCENE_OBJECT,
        LEFT_FLIPPER_16_BOT_SCENE_OBJECT,
        LEFT_FLIPPER_32_BOT_SCENE_OBJECT,
        LEFT_FLIPPER_48_BOT_SCENE_OBJECT,
        LEFT_FLIPPER_00_TOP_SCENE_OBJECT,
        LEFT_FLIPPER_16_TOP_SCENE_OBJECT,
        LEFT_FLIPPER_32_TOP_SCENE_OBJECT,
        LEFT_FLIPPER_48_TOP_SCENE_OBJECT,
        RIGHT_FLIPPER_00_BOT_SCENE_OBJECT,
        RIGHT_FLIPPER_16_BOT_SCENE_OBJECT,
        RIGHT_FLIPPER_32_BOT_SCENE_OBJECT,
        RIGHT_FLIPPER_48_BOT_SCENE_OBJECT,
        RIGHT_FLIPPER_00_TOP_SCENE_OBJECT,
        RIGHT_FLIPPER_16_TOP_SCENE_OBJECT,
        RIGHT_FLIPPER_32_TOP_SCENE_OBJECT,
        RIGHT_FLIPPER_48_TOP_SCENE_OBJECT,
    )
    FLIPPER_SEGMENTS_SORTED = jnp.array([
        # (px, py), (ex, ey)
        [[FLIPPER_LEFT_PIVOT_X, FLIPPER_PIVOT_Y_BOT], [FLIPPER_LEFT_00_END_X, FLIPPER_00_BOT_END_Y]],
        [[FLIPPER_LEFT_PIVOT_X, FLIPPER_PIVOT_Y_BOT], [FLIPPER_LEFT_16_END_X, FLIPPER_16_BOT_END_Y]],
        [[FLIPPER_LEFT_PIVOT_X, FLIPPER_PIVOT_Y_BOT], [FLIPPER_LEFT_32_END_X, FLIPPER_32_BOT_END_Y]],
        [[FLIPPER_LEFT_PIVOT_X, FLIPPER_PIVOT_Y_BOT], [FLIPPER_LEFT_48_END_X, FLIPPER_48_BOT_END_Y]],
        [[FLIPPER_LEFT_PIVOT_X, FLIPPER_PIVOT_Y_TOP], [FLIPPER_LEFT_00_END_X, FLIPPER_00_TOP_END_Y]],
        [[FLIPPER_LEFT_PIVOT_X, FLIPPER_PIVOT_Y_TOP], [FLIPPER_LEFT_16_END_X, FLIPPER_16_TOP_END_Y]],
        [[FLIPPER_LEFT_PIVOT_X, FLIPPER_PIVOT_Y_TOP], [FLIPPER_LEFT_32_END_X, FLIPPER_32_TOP_END_Y]],
        [[FLIPPER_LEFT_PIVOT_X, FLIPPER_PIVOT_Y_TOP], [FLIPPER_LEFT_48_END_X, FLIPPER_48_TOP_END_Y]],
        
        [[FLIPPER_RIGHT_PIVOT_X, FLIPPER_PIVOT_Y_BOT], [FLIPPER_RIGHT_00_END_X, FLIPPER_00_BOT_END_Y]],
        [[FLIPPER_RIGHT_PIVOT_X, FLIPPER_PIVOT_Y_BOT], [FLIPPER_RIGHT_16_END_X, FLIPPER_16_BOT_END_Y]],
        [[FLIPPER_RIGHT_PIVOT_X, FLIPPER_PIVOT_Y_BOT], [FLIPPER_RIGHT_32_END_X, FLIPPER_32_BOT_END_Y]],
        [[FLIPPER_RIGHT_PIVOT_X, FLIPPER_PIVOT_Y_BOT], [FLIPPER_RIGHT_48_END_X, FLIPPER_48_BOT_END_Y]],
        [[FLIPPER_RIGHT_PIVOT_X, FLIPPER_PIVOT_Y_TOP], [FLIPPER_RIGHT_00_END_X, FLIPPER_00_TOP_END_Y]],
        [[FLIPPER_RIGHT_PIVOT_X, FLIPPER_PIVOT_Y_TOP], [FLIPPER_RIGHT_16_END_X, FLIPPER_16_TOP_END_Y]],
        [[FLIPPER_RIGHT_PIVOT_X, FLIPPER_PIVOT_Y_TOP], [FLIPPER_RIGHT_32_END_X, FLIPPER_32_TOP_END_Y]],
        [[FLIPPER_RIGHT_PIVOT_X, FLIPPER_PIVOT_Y_TOP], [FLIPPER_RIGHT_48_END_X, FLIPPER_48_TOP_END_Y]],
    ])
    FLIPPERS = jnp.stack(
        [
            jnp.array(
                [
                    scene_object.hit_box_width,
                    scene_object.hit_box_height,
                    scene_object.hit_box_x_offset,
                    scene_object.hit_box_y_offset,
                    scene_object.reflecting,
                    scene_object.score_type,
                    scene_object.variant,
                ],
                dtype=jnp.int32,
            )
            for scene_object in _FLIPPERS_SORTED
        ]
    ).squeeze()
