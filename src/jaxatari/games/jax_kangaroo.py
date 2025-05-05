import os
from functools import partial
from typing import NamedTuple, Tuple, Dict, Any, Optional
import jax
import jax.numpy as jnp
import chex
import pygame
from jax import Array
from gymnax.environments import spaces
from jaxatari.environment import JaxEnvironment

from jaxatari.games.kangaroo_levels import (
    LevelConstants,
    Kangaroo_Level_1,
    Kangaroo_Level_2,
    Kangaroo_Level_3,
)

# -------- Action constants --------
NOOP, FIRE, UP, RIGHT, LEFT, DOWN, UPRIGHT, UPLEFT, DOWNRIGHT, DOWNLEFT = range(10)
(
    UPFIRE,
    RIGHTFIRE,
    LEFTFIRE,
    DOWNFIRE,
    UPRIGHTFIRE,
    UPLEFTFIRE,
    DOWNRIGHTFIRE,
    DOWNLEFTFIRE,
) = range(10, 18)
RESET = 18

# -------- Game constants --------
RENDER_SCALE_FACTOR = 4
SCREEN_WIDTH, SCREEN_HEIGHT = 160, 210
PLAYER_WIDTH, PLAYER_HEIGHT = 8, 24
ENEMY_WIDTH, ENEMY_HEIGHT = 8, 24
FRUIT_WIDTH = 8
FRUIT_HEIGHT = 12

BELL_WIDTH = 6
BELL_HEIGHT = 11

CHILD_WIDTH = 8
CHILD_HEIGHT = 15

MONKEY_WIDTH = 6
MONKEY_HEIGHT = 15
MONKEY_COLOR = (227, 159, 89)

BACKGROUND_COLOR = (80, 0, 132)
PLAYER_COLOR = (223, 183, 85)
ENEMY_COLOR = (227, 151, 89)
FRUIT_COLOR_STATE_1 = (214, 92, 92)
FRUIT_COLOR_STATE_2 = (230, 250, 92)
FRUIT_COLOR_STATE_3 = (255, 92, 250)
FRUIT_COLOR_STATE_4 = (0, 92, 250)
FRUIT_COLOR = [
    FRUIT_COLOR_STATE_1,
    FRUIT_COLOR_STATE_2,
    FRUIT_COLOR_STATE_3,
    FRUIT_COLOR_STATE_4,
]
COCONUT_COLOR = PLATFORM_COLOR = (162, 98, 33)
LADDER_COLOR = (129, 78, 26)
BELL_COLOR = (210, 164, 74)

PLAYER_START_X, PLAYER_START_Y = 23, 148
MOVEMENT_SPEED = 1

LEFT_CLIP = 16
RIGHT_CLIP = 144

COCONUT_WIDTH = 3
COCONUT_HEIGHT = 4


# -------- Entity Classes --------
class Entity(NamedTuple):
    x: chex.Array
    y: chex.Array
    w: chex.Array
    h: chex.Array


class PlayerState(NamedTuple):
    # Player position
    x: chex.Array
    y: chex.Array
    vel_x: chex.Array
    orientation: chex.Array
    height: chex.Array
    # crouching
    is_crouching: chex.Array
    # jumping
    is_jumping: chex.Array
    jump_base_y: chex.Array
    jump_counter: chex.Array
    jump_orientation: chex.Array
    landing_base_y: chex.Array
    # climbing
    is_climbing: chex.Array
    climb_base_y: chex.Array
    climb_counter: chex.Array
    cooldown_counter: chex.Array
    # other
    is_crashing: chex.Array
    chrash_timer: chex.Array
    punch_left: chex.Array
    punch_right: chex.Array
    last_stood_on_platform_y: chex.Array
    walk_animation: chex.Array


class LevelState(NamedTuple):
    """All level related state variables."""

    timer: chex.Array
    platform_positions: chex.Array
    platform_sizes: chex.Array
    ladder_positions: chex.Array
    ladder_sizes: chex.Array
    fruit_positions: chex.Array
    fruit_actives: chex.Array
    fruit_stages: chex.Array
    bell_position: chex.Array
    bell_timer: chex.Array
    child_position: chex.Array
    child_velocity: chex.Array
    child_timer: chex.Array
    falling_coco_position: chex.Array
    falling_coco_dropping: chex.Array
    falling_coco_counter: chex.Array
    falling_coco_skip_update: chex.Array
    step_counter: chex.Array
    monkey_states: chex.Array
    """
    - 0: non-existent
    - 1: moving down
    - 2: moving left
    - 3: throwing
    - 4: moving right
    - 5: moving up
    """
    monkey_positions: chex.Array
    """2D array: [monkey_index, [x, y]]"""
    monkey_throw_timers: chex.Array
    spawn_protection: chex.Array
    coco_positions: chex.Array
    coco_states: chex.Array
    """
    - 0: non existent
    - 1: charging
    - 2: throwing
    """
    spawn_position: chex.Array
    """
    - 0: foot
    - 1: head
    """
    bell_animation: chex.Array


class KangarooState(NamedTuple):
    player: PlayerState
    level: LevelState
    score: chex.Array
    current_level: chex.Array
    level_finished: chex.Array
    levelup_timer: chex.Array
    reset_coords: chex.Array
    levelup: chex.Array
    lives: chex.Array
    obs_stack: chex.ArrayTree


class KangarooObservation(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    player_o: chex.Array
    platform_positions: chex.Array
    platform_sizes: chex.Array
    ladder_positions: chex.Array
    ladder_sizes: chex.Array
    fruit_positions: chex.Array
    fruit_actives: chex.Array
    fruit_stages: chex.Array
    bell_position: chex.Array
    child_position: chex.Array
    falling_coco_position: chex.Array
    monkey_states: chex.Array
    monkey_positions: chex.Array
    coco_positions: chex.Array
    coco_states: chex.Array


class KangarooInfo(NamedTuple):
    score: chex.Array
    level: chex.Array
    all_rewards: chex.Array

# Level Constants
LADDER_HEIGHT = jnp.array(35)
LADDER_WIDTH = jnp.array(8)
P_HEIGHT = jnp.array(4)

LEVEL_1 = Kangaroo_Level_1
LEVEL_2 = Kangaroo_Level_2
LEVEL_3 = Kangaroo_Level_3


# -------- Keyboard Inputs --------
def get_human_action() -> chex.Array:
    keys = pygame.key.get_pressed()
    up = keys[pygame.K_w] or keys[pygame.K_UP]
    down = keys[pygame.K_s] or keys[pygame.K_DOWN]
    left = keys[pygame.K_a] or keys[pygame.K_LEFT]
    right = keys[pygame.K_d] or keys[pygame.K_RIGHT]
    fire = keys[pygame.K_SPACE]
    reset = keys[pygame.K_r]

    if reset:
        return jnp.array(RESET)

    x, y = 0, 0
    if up and not down:
        y = 1
    elif not up and down:
        y = -1

    if left and not right:
        x = -1
    elif not left and right:
        x = 1

    if fire:
        if x == -1 and y == -1:
            return jnp.array(DOWNLEFTFIRE)
        elif x == -1 and y == 1:
            return jnp.array(UPLEFTFIRE)
        elif x == 1 and y == -1:
            return jnp.array(DOWNRIGHTFIRE)
        elif x == 1 and y == 1:
            return jnp.array(UPRIGHTFIRE)
        elif x == 0 and y == -1:
            return jnp.array(DOWNFIRE)
        else:
            return jnp.array(FIRE)
    else:
        if x == -1 and y == -1:
            return jnp.array(DOWNLEFT)
        elif x == -1 and y == 1:
            return jnp.array(UPLEFT)
        elif x == 1 and y == -1:
            return jnp.array(DOWNRIGHT)
        elif x == 1 and y == 1:
            return jnp.array(UPRIGHT)
        elif x == -1:
            return jnp.array(LEFT)
        elif x == 1:
            return jnp.array(RIGHT)
        elif y == -1:
            return jnp.array(DOWN)
        elif y == 1:
            return jnp.array(UP)

    return jnp.array(NOOP)


@partial(jax.jit, static_argnums=())
def get_valid_platforms(level_constants: LevelConstants) -> chex.Array:
    """Check if a platform position is valid (not padding)."""
    return level_constants.platform_positions[:, 0] != -1


@partial(jax.jit, static_argnums=(1), donate_argnums=(0))
def get_platforms_below_player(state: KangarooState, y_offset=0) -> chex.Array:
    """Returns array of booleans indicating if player is on a platform."""
    player_x = state.player.x
    player_y = state.player.y + y_offset
    player_bottom_y = player_y + state.player.height

    level_constants: LevelConstants = get_level_constants(state.current_level)

    platform_positions = level_constants.platform_positions  # [N, 2]
    platform_sizes = level_constants.platform_sizes  # [N, 2]

    # Extract platform coordinates
    platform_x = platform_positions[:, 0]
    platform_y = platform_positions[:, 1]
    platform_width = platform_sizes[:, 0]

    # Vectorized checks
    player_is_within_platform_x = jnp.logical_and(
        (player_x + PLAYER_WIDTH) >= platform_x,
        player_x <= (platform_x + platform_width),
    )

    platform_is_below_player = player_bottom_y <= platform_y

    # Calculate vertical distances
    diff_to_platforms = jnp.where(
        platform_is_below_player, platform_y - player_bottom_y, 1000
    )

    # Check which platforms are valid
    valid_platforms = get_valid_platforms(level_constants)

    # Combine all conditions for candidate platforms
    candidate_platforms = (
        player_is_within_platform_x & platform_is_below_player & valid_platforms
    )

    # Set distances for non-candidate platforms to a large value
    masked_diffs = jnp.where(candidate_platforms, diff_to_platforms, 1000)

    # Find the closest platform
    closest_platform_idx = jnp.argmin(masked_diffs)
    min_diff = masked_diffs[closest_platform_idx]

    # Create result array with True only for the closest valid platform
    result = jnp.zeros_like(platform_x, dtype=bool)
    return result.at[closest_platform_idx].set(min_diff < 1000)


@partial(jax.jit, static_argnums=())
def entities_collide_with_threshold(
    e1_x: chex.Array,
    e1_y: chex.Array,
    e1_w: chex.Array,
    e1_h: chex.Array,
    e2_x: chex.Array,
    e2_y: chex.Array,
    e2_w: chex.Array,
    e2_h: chex.Array,
    threshold: chex.Array,
) -> chex.Array:
    """Returns True if rectangles overlap by at least threshold fraction. This only Checks for overlap in the x dimension."""
    overlap_start_x = jnp.maximum(e1_x, e2_x)
    overlap_end_x = jnp.minimum(e1_x + e1_w, e2_x + e2_w)
    overlap_start_y = jnp.maximum(e1_y, e2_y)
    overlap_end_y = jnp.minimum(e1_y + e1_h, e2_y + e2_h)

    # Calculate dimensions of overlap region
    overlap_width = overlap_end_x - overlap_start_x
    overlap_height = overlap_end_y - overlap_start_y

    # Calculate minimum required overlap area based on threshold
    min_required_overlap = e1_w * threshold

    # Check if overlap exceeds required threshold
    meets_threshold = overlap_width >= min_required_overlap

    return jnp.where((overlap_width < 0) | (overlap_height < 0), False, meets_threshold)


@partial(jax.jit, static_argnums=())
def entities_collide(
    e1_x: chex.Array,
    e1_y: chex.Array,
    e1_w: chex.Array,
    e1_h: chex.Array,
    e2_x: chex.Array,
    e2_y: chex.Array,
    e2_w: chex.Array,
    e2_h: chex.Array,
) -> chex.Array:
    """
    Calls do_collide_with_threshold with a threshold of 0 and checks if two rectangles overlap.
    """
    return entities_collide_with_threshold(
        e1_x, e1_y, e1_w, e1_h, e2_x, e2_y, e2_w, e2_h, 0
    )


@partial(jax.jit, static_argnums=(1, 2), donate_argnums=(0))
def player_is_above_ladder(
    state: KangarooState,
    threshold: float = 0.3,
    virtual_hitbox_height: float = 12.0,
) -> chex.Array:
    """Checks collision between a virtual hitbox below player and ladders."""

    level_constants: LevelConstants = get_level_constants(state.current_level)

    ladder_x = level_constants.ladder_positions[:, 0]
    ladder_y = level_constants.ladder_positions[:, 1]
    ladder_w = level_constants.ladder_sizes[:, 0]
    ladder_h = level_constants.ladder_sizes[:, 1]

    return jax.vmap(
        entities_collide_with_threshold,
        in_axes=(None, None, None, None, 0, 0, 0, 0, None),
    )(
        state.player.x,
        state.player.y + state.player.height,
        PLAYER_WIDTH,
        virtual_hitbox_height,
        ladder_x,
        ladder_y,
        ladder_w,
        ladder_h,
        threshold,
    )


@partial(jax.jit, static_argnums=(1), donate_argnums=(0))
def check_ladder_collisions(state: KangarooState, threshold: float = 0.3) -> chex.Array:
    """Vectorized ladder collision checking."""

    level_constants: LevelConstants = get_level_constants(state.current_level)

    ladder_x = level_constants.ladder_positions[:, 0]
    ladder_y = level_constants.ladder_positions[:, 1]
    ladder_w = level_constants.ladder_sizes[:, 0]
    ladder_h = level_constants.ladder_sizes[:, 1]

    return jax.vmap(
        entities_collide_with_threshold,
        in_axes=(None, None, None, None, 0, 0, 0, 0, None),
    )(
        state.player.x,
        state.player.y + 16,
        PLAYER_WIDTH,
        state.player.height - 16,
        ladder_x,
        ladder_y,
        ladder_w,
        ladder_h,
        threshold,
    )


@partial(jax.jit, donate_argnums=(0), static_argnums=(3))
def player_is_on_ladder(
    state: KangarooState,
    ladder_pos: chex.Array,
    ladder_size: chex.Array,
    threshold: float = 0.3,
) -> chex.Array:
    """
    Checks the collision of the player with a ladder. <threshold>% of the players surface area have to overlap with the ladder.
    """
    return entities_collide_with_threshold(
        state.player.x,
        state.player.y,
        PLAYER_WIDTH,
        state.player.height,
        ladder_pos[0],
        ladder_pos[1],
        ladder_size[0],
        ladder_size[1],
        threshold,
    )


@partial(jax.jit, donate_argnums=(0))
# -------- Jumping and Climbing --------
def player_jump_controller(
    state: KangarooState, jump_pressed: chex.Array, ladder_intersect: chex.Array
):
    """
    Schedule:
      tick  8: total offset = -12
      tick 16: still -12 (just sprite/hitbox change)
      tick 24: total offset = -24
      tick 32: total offset = -12  (moved up +12 from -24)
      tick 40: total offset =  0   (moved up +12 from -12) -> jump ends
    """
    player_y = state.player.y
    jump_counter = state.player.jump_counter
    is_jumping = state.player.is_jumping

    cooldown_condition = state.player.cooldown_counter > 0
    jump_start = jump_pressed & ~is_jumping & ~ladder_intersect & ~cooldown_condition

    # Update jump state on start
    jump_counter = jnp.where(jump_start, 0, jump_counter)
    jump_orientation = jnp.where(
        jump_start, state.player.orientation, state.player.jump_orientation
    )
    jump_base_y = jnp.where(jump_start, player_y, state.player.jump_base_y)
    new_landing_base_y = jump_base_y
    # check if player is on/above a new platform and change jump_base_y accordingly

    platform_y_below_player = get_y_of_platform_below_player(state)

    # find a new potential landing_base if player is above a higher platform
    new_landing_base_y = jnp.where(
        is_jumping
        & ((platform_y_below_player - PLAYER_HEIGHT) < jump_base_y)
        & ~jump_start,
        platform_y_below_player - PLAYER_HEIGHT,
        new_landing_base_y,
    )

    is_jumping = is_jumping | jump_start

    # Update counter if jumping
    jump_counter = jnp.where(is_jumping, jump_counter + 1, jump_counter)

    # Calculate vertical offset based on jump phase
    def offset_for(count):
        conditions = [
            (count <= 8),
            (count < 16),
            (count <= 24),
            (count <= 32),
            (count < 41),
        ]
        values = [
            -1,
            -8,
            -8,
            -16,
            -8,
        ]
        return jnp.select(conditions, values, default=0)

    # check if player is on a new platform and cancel jump if so
    jump_cancel = (
        is_jumping
        & (player_y >= new_landing_base_y)
        & (new_landing_base_y < jump_base_y)
        & (jump_counter > 32)
    )
    jump_counter = jnp.where(jump_cancel, 40, jump_counter)
    jump_base_y = jnp.where(jump_cancel, new_landing_base_y, jump_base_y)
    new_y = jnp.where(jump_cancel, new_landing_base_y, player_y)
    new_cooldown_counter = jnp.where(jump_cancel, 8, state.player.cooldown_counter)

    total_offset = offset_for(jump_counter)
    new_y = jnp.where(is_jumping & ~jump_cancel, jump_base_y + total_offset, new_y)

    # Check for jump completion
    jump_complete = jump_counter >= 41
    is_jumping = jnp.where(jump_complete, False, is_jumping)
    jump_counter = jnp.where(jump_complete, 0, jump_counter)

    return_value = (
        new_y,
        jump_counter,
        is_jumping,
        jump_base_y,
        new_landing_base_y,
        jump_orientation,
        new_cooldown_counter,
    )

    return jax.lax.cond(
        state.levelup_timer == 0,
        lambda: return_value,
        lambda: (
            state.player.y,
            state.player.jump_counter,
            state.player.is_jumping,
            state.player.jump_base_y,
            state.player.landing_base_y,
            state.player.jump_orientation,
            state.player.cooldown_counter,
        ),
    )


@partial(jax.jit, donate_argnums=(0))
def player_climb_controller(
    state: KangarooState,
    y: chex.Array,
    press_up: chex.Array,
    press_down: chex.Array,
    ladder_intersect: chex.Array,
) -> tuple[Array, Array, Array, Array, Array]:

    # Ladder Below Collision
    ladder_intersect_below = jnp.any(player_is_above_ladder(state))

    new_y = y
    is_climbing = state.player.is_climbing
    is_climbing = jnp.where(state.player.is_jumping, False, is_climbing)

    climb_counter = state.player.climb_counter

    cooldown_over = state.player.cooldown_counter <= 0

    climb_start = (
        press_up
        & ~is_climbing
        & ladder_intersect
        & ~state.player.is_jumping
        & cooldown_over
    )
    climb_start_downward = (
        press_down
        & ~is_climbing
        & ladder_intersect_below
        & ~state.player.is_jumping
        & cooldown_over
    )

    is_climbing = is_climbing | climb_start | climb_start_downward

    climb_counter = jnp.where(climb_start | climb_start_downward, 0, climb_counter)

    climb_base_y = jnp.where(climb_start, new_y, state.player.climb_base_y)

    climb_base_y = jnp.where(
        climb_start_downward,
        get_y_of_platform_below_player(state, 1) - PLAYER_HEIGHT,
        climb_base_y,
    )

    new_y = jnp.where(climb_start, new_y - 8, new_y)
    new_y = jnp.where(climb_start_downward, new_y + 8, new_y)

    climb_counter = jnp.where(is_climbing, climb_counter + 1, climb_counter)

    climb_up = jnp.logical_and(press_up, is_climbing)
    climb_down = jnp.logical_and(press_down, is_climbing)

    new_y = jnp.where(
        jnp.logical_and(climb_up, jnp.equal(climb_counter, 19)), new_y - 8, new_y
    )
    new_y = jnp.where(
        jnp.logical_and(climb_down, jnp.equal(climb_counter, 19)), new_y + 8, new_y
    )

    set_new_climb_base = (
        climb_up
        & ((get_y_of_platform_below_player(state) - state.player.height) >= new_y)
        & ladder_intersect
    )
    climb_base_y = jnp.where(
        set_new_climb_base,  # when player is on a new platform but still climbing up
        get_y_of_platform_below_player(state) - PLAYER_HEIGHT,
        climb_base_y,
    )
    # Check if not climbing anymore -> bottom of ladder
    climb_stop = is_climbing & (new_y >= climb_base_y) & ~climb_start_downward

    is_climbing = jnp.where(climb_stop, False, is_climbing)

    # Check if not climbing anymore -> top of ladder
    is_climbing = jnp.where(ladder_intersect | climb_start_downward, is_climbing, False)

    climb_counter = jnp.where(climb_counter >= 19, 0, climb_counter)
    cooldown_counter = jnp.where(
        climb_stop | set_new_climb_base,
        15,
        jnp.where(
            state.player.cooldown_counter > 0, state.player.cooldown_counter - 1, 0
        ),
    )

    return_value = (new_y, is_climbing, climb_base_y, climb_counter, cooldown_counter)

    return jax.lax.cond(
        state.levelup_timer == 0,
        lambda: return_value,
        lambda: (
            state.player.y,
            state.player.is_climbing,
            state.player.climb_base_y,
            state.player.climb_counter,
            state.player.cooldown_counter,
        ),
    )


# -------- Player Height --------
@partial(jax.jit, static_argnums=())
def player_height_controller(
    is_jumping: chex.Array,
    jump_counter: chex.Array,
    is_crouching: chex.Array,
) -> chex.Array:
    """
    Jump-based height changes:
      -  0..15 ticks => height=24 (normal)
      - 16..23 ticks => height=15 (small sprite)
      - 24..39 ticks => height=23 (stretched jump sprite)
      - >= 40        => height=24 (back on ground)

    If pressing DOWN *and not jumping*, override => 16 for ducking.
    """

    def jump_based_height(count):
        conditions = [
            (count < 8),
            (count < 16),
            (count < 24),
            (count < 40),
        ]
        values = [
            23,
            24,
            15,
            23,
        ]
        return jnp.select(conditions, values, default=24)

    candidate_height = jump_based_height(jump_counter)
    height_if_jumping = jnp.where(is_jumping, candidate_height, 24)

    is_crouching = jnp.logical_and(is_crouching, jnp.logical_not(is_jumping))

    new_height = jnp.where(is_crouching, 16, height_if_jumping)
    return new_height


@partial(jax.jit, static_argnums=(1), donate_argnums=(0))
def get_y_of_platform_below_player(state: KangarooState, y_offset=0) -> chex.Array:
    """Gets the y-position of the next platform below the player."""

    level_constants: LevelConstants = get_level_constants(state.current_level)

    # Get array with True only for closest platform below player
    platform_bands: jax.Array = get_platforms_below_player(state, y_offset)
    platform_ys = level_constants.platform_positions[:, 1]

    # Check if any platform is below player
    has_platform_below = jnp.any(platform_bands)

    # Get the y-position of the closest platform below player using element-wise multiplication
    # This works because platform_bands has at most one True value (the closest platform)
    platform_y = jnp.sum(platform_bands * platform_ys)

    # Return platform_y if any platform is below, otherwise return 1000
    return jnp.where(has_platform_below, platform_y, jnp.array(1000))


@partial(jax.jit, donate_argnums=(0))
def fruits_step(state: KangarooState) -> Tuple[chex.Array, chex.Array]:
    """Handles fruit collection and scoring."""

    fruit_x = state.level.fruit_positions[:, 0]
    fruit_y = state.level.fruit_positions[:, 1]

    def check_fruit(p_x, p_y, p_w, p_h, f_x, f_y, f_w, f_h, stage, active):
        """Returns score addition and new activation state per fruit."""
        fruit_collision = entities_collide(p_x, p_y, p_w, p_h, f_x, f_y, f_w, f_h)
        collision_condition = jnp.logical_and(fruit_collision, active)
        return jnp.where(collision_condition, 100 * (2**stage), 0), jnp.where(
            collision_condition, False, active
        )

    (score_additions, new_activations) = jax.vmap(
        check_fruit, in_axes=(None, None, None, None, 0, 0, None, None, 0, 0)
    )(
        state.player.x,
        state.player.y,
        PLAYER_WIDTH,
        state.player.height,
        fruit_x,
        fruit_y,
        FRUIT_WIDTH,
        FRUIT_HEIGHT,
        state.level.fruit_stages,
        state.level.fruit_actives,
    )
    new_score = jnp.sum(score_additions)

    # Check for bell collision
    bell_collision = entities_collide(
        state.player.x,
        state.player.y,
        PLAYER_WIDTH,
        state.player.height,
        state.level.bell_position[0],
        state.level.bell_position[1],
        BELL_WIDTH,
        BELL_HEIGHT,
    )
    bell_active = ~jnp.any(state.level.fruit_stages == 3)

    RESPAWN_AFTER_TICKS = 40

    counter = state.level.bell_timer
    counter_start = bell_collision & (counter == 0) & bell_active
    counter = jnp.where(counter_start, 1, counter)
    counter = jnp.where(counter > 0, counter + 1, counter)
    counter = jnp.where(counter == RESPAWN_AFTER_TICKS + 1, 0, counter)
    respawn_timer_done = counter == RESPAWN_AFTER_TICKS

    def get_new_stages(respawn_timer_done, active, stage):
        return jnp.where(
            respawn_timer_done & (~active),
            jnp.clip(stage + 1, 0, 3),
            stage,
        )

    new_stages = jax.vmap(get_new_stages, in_axes=(None, 0, 0))(
        respawn_timer_done, state.level.fruit_actives, state.level.fruit_stages
    )

    activations = jax.lax.cond(
        respawn_timer_done,
        lambda: jnp.less_equal(new_stages, jnp.array([3, 3, 3])),
        lambda: new_activations,
    )

    return new_score, activations, new_stages, counter


@partial(jax.jit, donate_argnums=(0))
def child_step(state: KangarooState) -> Tuple[chex.Array]:

    RESET_TIMER_AFTER = 50

    counter = state.level.child_timer
    counter = counter + 1
    counter = jnp.where(counter > RESET_TIMER_AFTER, 0, counter)
    reset = counter == RESET_TIMER_AFTER

    child_velocity = state.level.child_velocity
    new_child_velocity = jnp.where(reset, child_velocity * -1, child_velocity)

    new_child_x = jnp.where(
        state.levelup_timer == 0,
        jnp.where(
            (counter % 5) == 0,
            state.level.child_position[0] + new_child_velocity,
            state.level.child_position[0],
        ),
        state.level.child_position[0],
    )
    new_child_y = state.level.child_position[1]
    new_child_timer = counter

    return new_child_timer, new_child_x, new_child_y, new_child_velocity


def pad_array(arr: jax.Array, target_size: int):
    """Pads a 2D array with -1s to reach target size in first dimension."""
    current_size = arr.shape[0]

    return jnp.pad(
        arr,
        ((0, target_size - current_size), (0, 0)),
        mode="constant",
        constant_values=-1,
    )


def pad_to_size(level_constants: LevelConstants, max_platforms: int):
    """Pads all arrays in level constants to specified size."""
    return LevelConstants(
        ladder_positions=pad_array(level_constants.ladder_positions, max_platforms),
        ladder_sizes=pad_array(level_constants.ladder_sizes, max_platforms),
        platform_positions=pad_array(level_constants.platform_positions, max_platforms),
        platform_sizes=pad_array(level_constants.platform_sizes, max_platforms),
        fruit_positions=level_constants.fruit_positions,
        bell_position=level_constants.bell_position,
        child_position=level_constants.child_position,
    )


@partial(jax.jit, static_argnums=())
def get_level_constants(current_level: int) -> LevelConstants:
    """Returns constants for the current level."""
    max_platforms = 20

    # Pad each level's arrays to max size
    level1_padded = pad_to_size(LEVEL_1, max_platforms)
    level2_padded = pad_to_size(LEVEL_2, max_platforms)
    level3_padded = pad_to_size(LEVEL_3, max_platforms)

    return jax.lax.cond(
        current_level == 1,
        lambda _: level1_padded,
        lambda _: jax.lax.cond(
            current_level == 2,
            lambda _: level2_padded,
            lambda _: level3_padded,
            operand=None,
        ),
        operand=None,
    )


@partial(jax.jit, donate_argnums=(0))
def player_step(state: KangarooState, action: chex.Array):
    """Main player movement and state update function."""
    level_constants = get_level_constants(state.current_level)
    x, y = state.player.x, state.player.y
    old_height = state.player.height
    old_orientation = state.player.orientation

    # Get inputs
    press_right = jnp.any(
        jnp.array([action == RIGHT, action == UPRIGHT, action == DOWNRIGHT])
    )

    press_left = jnp.any(
        jnp.array([action == LEFT, action == UPLEFT, action == DOWNLEFT])
    )

    press_up = jnp.any(jnp.array([action == UP, action == UPRIGHT, action == UPLEFT]))

    press_fire = jnp.any(
        jnp.array(
            [
                action == FIRE,
                action == DOWNFIRE,
                action == UPLEFTFIRE,
                action == UPRIGHTFIRE,
                action == DOWNLEFTFIRE,
                action == DOWNRIGHTFIRE,
            ]
        )
    )

    press_down_fire = jnp.any(jnp.array(action == DOWNFIRE))

    press_down = jnp.any(
        jnp.array([action == DOWN, action == DOWNLEFT, action == DOWNRIGHT])
    )

    press_down = jnp.where(state.player.is_jumping, False, press_down)
    press_fire = jnp.where(state.player.is_jumping, False, press_fire)
    press_fire = jnp.where(state.player.is_climbing, False, press_fire)
    press_fire = jnp.where(press_down_fire, False, press_fire)

    press_up = jnp.where(press_down_fire, False, press_up)

    # Forbid left/right movement while climbing
    press_right = jnp.where(state.player.is_climbing, False, press_right)
    press_left = jnp.where(state.player.is_climbing, False, press_left)

    is_looking_left = state.player.orientation == -1
    is_looking_right = state.player.orientation == 1
    is_punching_left = (
        jnp.logical_and(press_fire, is_looking_left) & ~state.player.is_crashing
    )
    is_punching_right = (
        jnp.logical_and(press_fire, is_looking_right) & ~state.player.is_crashing
    )

    # check for any collision with standard threshold
    ladder_intersect_thresh = jnp.any(check_ladder_collisions(state))
    ladder_intersect_no_thresh = jnp.any(check_ladder_collisions(state, 0))

    ladder_intersect = jnp.where(
        state.player.is_climbing, ladder_intersect_no_thresh, ladder_intersect_thresh
    )

    # Jump controller
    (
        new_y,
        new_jump_counter,
        new_is_jumping,
        new_jump_base_y,
        new_landing_base_y,
        new_jump_orientation,
        new_cooldown_counter,
    ) = player_jump_controller(state, press_up, ladder_intersect)

    # Climb controller
    (
        new_y,
        new_is_climbing,
        new_climb_base_y,
        new_climb_counter,
        new_cooldown_counter,
    ) = player_climb_controller(state, new_y, press_up, press_down, ladder_intersect)

    new_is_crouching = press_down & ~new_is_climbing & ~new_is_jumping

    # Calculate horizontal velocity
    candidate_vel_x = jnp.where(
        new_is_crouching,
        0,
        jnp.where(
            press_left, -MOVEMENT_SPEED, jnp.where(press_right, MOVEMENT_SPEED, 0)
        ),
    )

    # Check Orientation (Left/Right)
    # if standing still, keep the old orientation
    standing_still = jnp.equal(candidate_vel_x, 0)
    new_orientation = jnp.sign(candidate_vel_x)
    new_orientation = jnp.where(standing_still, old_orientation, new_orientation)

    stop_in_air = jnp.logical_and(
        new_is_jumping, state.player.jump_orientation != new_orientation
    )

    # Stop Jump when orientation changes mid air
    vel_x = jnp.where(stop_in_air, 0, candidate_vel_x)

    # Height controller
    new_player_height = player_height_controller(
        is_jumping=new_is_jumping,
        jump_counter=new_jump_counter,
        is_crouching=new_is_crouching,
    )
    new_player_height = jnp.where(
        (state.levelup_timer > 0) | state.player.is_crashing,
        PLAYER_HEIGHT,
        new_player_height,
    )
    # If height changes, shift the player's top so the bottom remains consistent
    dy = old_height - new_player_height
    new_y = new_y + dy

    # x-axis movement
    x = jnp.where(
        state.player.is_crashing | state.levelup_timer != 0,
        x,
        jnp.clip(x + vel_x, LEFT_CLIP, RIGHT_CLIP - PLAYER_WIDTH),
    )

    # y-axis movement
    platform_bools: jax.Array = get_platforms_below_player(state)
    platform_ys: jax.Array = level_constants.platform_positions[:, 1]

    valid_platforms = get_valid_platforms(level_constants)

    # Vectorized platform-dependent y calculation
    # Create a mask for valid platforms that affect the player
    valid_and_affecting = jnp.logical_and(platform_bools, valid_platforms)

    # Calculate potential y-values for all platforms at once
    climbing_transition = ~state.player.is_climbing & new_is_climbing & press_down

    # For each platform, calculate what y would be if player is positioned on it
    platform_y_values = jnp.where(
        climbing_transition, new_y, jnp.clip(new_y, 0, platform_ys - new_player_height)
    )

    # Apply the mask to get only values for platforms that affect the player
    masked_platform_y_values = jnp.where(valid_and_affecting, platform_y_values, new_y)

    # Take the minimum valid y-value to ensure player doesn't fall through platforms
    # This works because y increases downward in screen coordinates
    platform_dependent_y = jnp.min(
        jnp.where(valid_and_affecting, masked_platform_y_values, SCREEN_HEIGHT)
    )

    y = jnp.where(
        state.player.is_crashing,
        jnp.where((y + new_player_height) > SCREEN_HEIGHT, y, y + 2),
        platform_dependent_y,
    )

    # check if player reached the final platform
    final_platform_y = 28
    player_on_last_platform = (new_y + new_player_height) == final_platform_y
    level_finished = (
        player_on_last_platform & ~state.level_finished & (state.levelup_timer == 0)
    )

    y = jnp.where(state.levelup_timer == 0, y, state.player.y)

    # Reset X and Y when going to next level
    x = jnp.where(state.reset_coords, PLAYER_START_X, x)
    y = jnp.where(state.reset_coords, PLAYER_START_Y, y)

    return (
        x,
        y,
        vel_x,
        new_is_crouching,
        new_is_jumping,
        new_is_climbing,
        new_jump_counter,
        new_orientation,
        new_jump_base_y,
        new_landing_base_y,
        new_player_height,
        new_jump_orientation,
        new_climb_base_y,
        new_climb_counter,
        is_punching_left,
        is_punching_right,
        new_cooldown_counter,
        level_finished,
    )


@partial(jax.jit, donate_argnums=(0))
def timer_controller(state: KangarooState):
    return jnp.where(
        state.level.step_counter == 255, state.level.timer - 100, state.level.timer
    )


@partial(jax.jit, donate_argnums=(0))
def next_level(state: KangarooState):

    RESET_AFTER_TICKS = 256

    counter = state.levelup_timer
    counter_start = state.level_finished & (counter == 0)
    counter = jnp.where((counter > 0) | counter_start, counter + 1, counter)
    reset_timer_done = counter == RESET_AFTER_TICKS
    counter = jnp.where(counter > RESET_AFTER_TICKS, 0, counter)

    reset_coords = jnp.where(reset_timer_done, jnp.array(True), jnp.array(False))
    levelup = jnp.where(reset_timer_done, jnp.array(True), jnp.array(False))

    current_level = jnp.where(levelup, state.current_level + 1, state.current_level)

    return current_level, counter, reset_coords, levelup


@partial(jax.jit, donate_argnums=(0))
def lives_controller(state: KangarooState):
    # timer check
    is_time_over = state.level.timer <= 0

    new_last_stood_on_platform_y = jnp.where(
        get_y_of_platform_below_player(state) == (state.player.y + state.player.height),
        get_y_of_platform_below_player(state),
        state.player.last_stood_on_platform_y,
    )

    # platform_drop_check()

    y_of_platform_below_player = get_y_of_platform_below_player(state)
    player_is_falling = (
        (state.player.y + state.player.height) == state.player.last_stood_on_platform_y
    ) & (y_of_platform_below_player > state.player.last_stood_on_platform_y)

    # monkey touch check

    def check_monkey_collision(p_x, p_y, p_w, p_h, m_x, m_y, m_w, m_h, m_state):
        return jnp.logical_and(
            entities_collide(p_x, p_y, p_w, p_h, m_x, m_y, m_w, m_h), m_state != 0
        )

    monkey_collision = jax.vmap(
        check_monkey_collision,
        in_axes=(None, None, None, None, 0, 0, None, None, 0),
    )(
        state.player.x,
        state.player.y,
        PLAYER_WIDTH,
        state.player.height,
        state.level.monkey_positions[:, 0],
        state.level.monkey_positions[:, 1],
        MONKEY_WIDTH,
        MONKEY_HEIGHT - 1,
        state.level.monkey_states,
    )

    player_collided_with_monkey = jnp.any(monkey_collision)

    # coconut touch check
    def check_collision(p_x, p_y, p_w, p_h, m_x, m_y, m_w, m_h, m_state):
        return jnp.logical_and(
            entities_collide(p_x, p_y, p_w, p_h, m_x, m_y, m_w, m_h - 1), m_state != 0
        )

    collision = jax.vmap(
        check_collision,
        in_axes=(None, None, None, None, 0, 0, None, None, 0),
    )(
        state.player.x,
        state.player.y,
        PLAYER_WIDTH,
        state.player.height,
        state.level.coco_positions[:, 0],
        state.level.coco_positions[:, 1],
        COCONUT_WIDTH,
        COCONUT_HEIGHT,
        state.level.coco_states,
    )

    player_collided_with_horizontal_coco = jnp.any(collision)

    crashed_falling_coco = entities_collide(
        state.player.x,
        state.player.y,
        PLAYER_WIDTH,
        state.player.height,
        state.level.falling_coco_position[0],
        state.level.falling_coco_position[1],
        COCONUT_WIDTH,
        COCONUT_HEIGHT,
    )

    remove_live = (
        is_time_over
        | player_is_falling
        | crashed_falling_coco
        | player_collided_with_monkey
        | player_collided_with_horizontal_coco
    ) & ~state.player.is_crashing
    new_is_crashing = jnp.where(remove_live, True, state.player.is_crashing)

    start_timer = (
        state.player.is_crashing
        & (state.player.chrash_timer == 0)
        & ((state.player.y + state.player.height) > SCREEN_HEIGHT)
    )

    # start counter
    RESPAWN_AFTER_TICKS = 40

    counter = state.player.chrash_timer
    counter_start = start_timer
    counter = jnp.where(counter_start, 1, counter)
    counter = jnp.where(counter > 0, counter + 1, counter)
    counter = jnp.where(counter == RESPAWN_AFTER_TICKS + 1, 0, counter)
    crash_timer_done = counter == RESPAWN_AFTER_TICKS

    new_is_crashing = jnp.where(crash_timer_done, False, new_is_crashing)

    return (
        jnp.where(remove_live, state.lives - 1, state.lives),
        new_is_crashing,
        counter,
        crash_timer_done,
        new_last_stood_on_platform_y,
    )


@partial(jax.jit, donate_argnums=(0))
def falling_coconut_controller(state: KangarooState):
    falling_coco_exists = (state.level.falling_coco_position[0] != 13) | (
        state.level.falling_coco_position[1] != -1
    )

    # update coco position
    spawn_new_coco = ~falling_coco_exists & (state.level.step_counter == 255)

    update_positions = ~state.level.falling_coco_skip_update & (
        ((state.level.step_counter % 8) == 0) | spawn_new_coco
    )

    # detect if coco is above player and switch from x-following state to dropping state
    coco_first_time_above_player = (
        ~state.level.falling_coco_dropping
        & falling_coco_exists
        & (
            ((state.level.falling_coco_position[0] + COCONUT_WIDTH) > state.player.x)
            & (state.level.falling_coco_position[0] < (state.player.x + PLAYER_WIDTH))
        )
        & update_positions
    )

    new_falling_coco_dropping = jnp.where(
        coco_first_time_above_player, True, state.level.falling_coco_dropping
    )

    new_falling_coco_skip_update = coco_first_time_above_player

    # reset skip update
    new_falling_coco_skip_update = jnp.where(
        state.level.falling_coco_skip_update
        & (((state.level.step_counter % 8) == 0) | spawn_new_coco),
        False,
        new_falling_coco_skip_update | state.level.falling_coco_skip_update,
    )

    new_falling_coco_counter = jnp.where(
        update_positions,
        jnp.where(
            spawn_new_coco,
            0,
            jnp.where(
                state.level.falling_coco_dropping & update_positions,
                state.level.falling_coco_counter + 1,
                state.level.falling_coco_counter,
            ),
        ),
        state.level.falling_coco_counter,
    )

    reset_coco = new_falling_coco_counter > 20
    new_falling_coco_counter = jnp.where(reset_coco, 0, new_falling_coco_counter)

    new_falling_coco_dropping = jnp.where(reset_coco, False, new_falling_coco_dropping)

    new_falling_coco_position_x = jnp.where(
        update_positions
        & ~state.level.falling_coco_dropping
        & (falling_coco_exists | spawn_new_coco),
        state.level.falling_coco_position[0] + 2,
        state.level.falling_coco_position[0],
    )
    new_falling_coco_position_y = jnp.where(
        update_positions & (falling_coco_exists | spawn_new_coco),
        8 * new_falling_coco_counter + 8,
        state.level.falling_coco_position[1],
    )

    new_falling_coco_position = jnp.where(
        reset_coco,
        jnp.array([13, -1]),
        jnp.array([new_falling_coco_position_x, new_falling_coco_position_y]),
    )

    return (
        new_falling_coco_position,
        new_falling_coco_dropping,
        new_falling_coco_counter,
        new_falling_coco_skip_update,
    )


@partial(jax.jit, donate_argnums=(0))
def monkey_controller(state: KangarooState, punching: chex.Array):
    """Monkey controller function."""

    # Count non-zero monkey states with a vectorized operation
    current_monkeys_existing = jnp.sum(state.level.monkey_states != 0)

    ## state 0 -> 1

    spawn_new_monkey = (
        ~state.level.spawn_protection
        & (current_monkeys_existing < 4)
        & (state.level.step_counter == 16)
    )

    # Vectorized approach to find first non-existing monkey
    monkey_states_is_zero = state.level.monkey_states == 0
    # Find the index of the first zero value using argmin
    first_non_existing_monkey_index = jnp.argmin(~monkey_states_is_zero)
    # Make sure we only use this index if there's at least one non-existing monkey
    first_non_existing_monkey_index = jnp.where(
        jnp.any(monkey_states_is_zero), first_non_existing_monkey_index, jnp.array(-1)
    )

    # update monkey states
    new_monkey_states = state.level.monkey_states
    new_monkey_states = jax.lax.cond(
        spawn_new_monkey,
        lambda: new_monkey_states.at[first_non_existing_monkey_index].set(1),
        lambda: new_monkey_states,
    )

    # State 1 -> 2

    # Vectorized implementation - replace the for loop with array operations
    monkey_lower_y = state.level.monkey_positions[:, 1] + MONKEY_HEIGHT
    monkey_on_p1 = monkey_lower_y == 172
    monkey_on_p2 = monkey_lower_y == 124
    monkey_on_p3 = monkey_lower_y == 76

    platform_y_under_player = get_y_of_platform_below_player(state)

    transition_1_to_2 = (
        (
            monkey_on_p1
            & (platform_y_under_player <= 172)
            & (platform_y_under_player > 124)
        )
        | (
            monkey_on_p2
            & (platform_y_under_player <= 124)
            & (platform_y_under_player > 76)
        )
        | (
            monkey_on_p3
            & (platform_y_under_player <= 76)
            & (platform_y_under_player > 28)
        )
    )

    # Apply the transition to all monkeys at once
    new_monkey_states = jnp.where(
        (new_monkey_states == 1) & transition_1_to_2, 2, new_monkey_states
    )

    ### It might be possible for the monkey to change from 1 to 5 if the player is on a higher platform...?

    # State 1 -> 5

    # Vectorized implementation for state 1 -> 5 transition
    # Create a mask for monkeys in state 1
    in_state_1 = new_monkey_states == 1

    # Check which monkeys should transition (y + height >= 172)
    should_transition = (state.level.monkey_positions[:, 1] + MONKEY_HEIGHT) >= 172

    # Combine conditions and update states
    # Only monkeys that are in state 1 AND meet the transition condition should change to state 5
    new_monkey_states = jnp.where(
        in_state_1 & should_transition,
        5,  # New state for monkeys meeting the condition
        new_monkey_states,  # Keep original state for others
    )

    # State 2 -> 3
    ## Not sure about the correct condition but something like: is the monkey close to the player (x-wise)
    ## OR the monkey is already too far left (there might be a max x coordinate == 107)
    ### If so, change state to 3
    ### If not, keep state 2

    # Vectorized implementation for state 2 -> 3 transition
    # Create a mask for monkeys in state 2
    in_state_2 = new_monkey_states == 2

    # Check which monkeys have reached the threshold x position
    monkey_x_positions = state.level.monkey_positions[:, 0]
    min_x_reached = monkey_x_positions <= 107

    # Combine conditions and update states
    # Only monkeys that are in state 2 AND meet the transition condition should change to state 3
    new_monkey_states = jnp.where(
        in_state_2 & min_x_reached,
        3,  # New state for monkeys meeting the condition
        new_monkey_states,  # Keep original state for others
    )

    # State 3 -> 4
    ## If the waiting timer is over, change state to 4

    # Create mask for monkeys in state 3
    in_state_3 = new_monkey_states == 3

    # Check which monkeys have their throw timer at 0
    timer_is_zero = state.level.monkey_throw_timers == 0

    # Combine conditions for state transition (from 3 to 4 when timer is 0)
    should_transition = in_state_3 & timer_is_zero & (state.level.monkey_states == 3)

    # Update states all at once: change to state 4 for monkeys that should transition
    new_monkey_states = jnp.where(should_transition, 4, new_monkey_states)

    # State 4 -> 5
    ## If the monkey is at x == 130 wait for 8 frames and then change state to 5

    # Vectorized implementation for state 4 -> 5 transition
    # Create a mask for monkeys in state 4
    in_state_4 = new_monkey_states == 4

    # Check which monkeys have reached the threshold x position
    monkey_x_positions = state.level.monkey_positions[:, 0]
    reached_right_position = monkey_x_positions >= 146

    # Combine conditions and update states all at once
    # Only monkeys that are in state 4 AND have reached position should change to state 5
    new_monkey_states = jnp.where(
        in_state_4 & reached_right_position,
        5,  # New state for monkeys meeting both conditions
        new_monkey_states,  # Keep original state for others
    )

    # State 5 -> 0
    ## If the monkey is at y == 5 change state to 0 and reset the position to the starting position

    # Vectorized approach for state 5 -> 0 transition
    # Create a mask for monkeys in state 5
    in_state_5 = new_monkey_states == 5

    # Check which monkeys have reached the top position for transition
    monkey_y_positions = state.level.monkey_positions[:, 1]
    reached_top_position = monkey_y_positions <= 5

    # Update states in a single vectorized operation
    new_monkey_states = jnp.where(
        in_state_5 & reached_top_position,
        0,  # New state for monkeys meeting transition conditions
        new_monkey_states,  # Keep original state for others
    )

    # Additional
    ## If monkey is punched by player (collision with player + player punch in right direction) -> change state to 0 and reset position

    fist_x = jnp.where(
        state.player.orientation > 0, state.player.x + PLAYER_WIDTH, state.player.x - 3
    )
    fist_y = state.player.y + 8
    fist_w = 3
    fist_h = 4

    def check_punch(f_x, f_y, f_w, f_h, m_x, m_y, m_w, m_h, m_state, punching):
        return jnp.logical_and(
            entities_collide(f_x, f_y, f_w, f_h, m_x, m_y, m_w, m_h),
            jnp.logical_and(m_state != 0, punching),
        )

    monkeys_punched = jax.vmap(
        check_punch,
        in_axes=(None, None, None, None, 0, 0, None, None, 0, None),
    )(
        fist_x,
        fist_y,
        fist_w,
        fist_h,
        state.level.monkey_positions[:, 0],
        state.level.monkey_positions[:, 1],
        MONKEY_WIDTH,
        MONKEY_HEIGHT,
        state.level.monkey_states,
        punching,
    )

    score_addition = jnp.sum(monkeys_punched) * 200

    new_monkey_states = jax.vmap(lambda a, b: jnp.where(b, 0, a), in_axes=(0, 0))(
        new_monkey_states, monkeys_punched
    )

    # Update monkey positions using vectorization
    def update_single_monkey_position(
        state_monkey, position_monkey, new_state_monkey, step_counter
    ):
        """Update position for a single monkey."""
        should_update = step_counter % 16 == 0

        # Calculate potential new positions for each state
        # State 0: Reset position
        pos_state_0 = jnp.array([152, 5])

        # State 1: Moving down
        pos_state_1 = jnp.where(
            state_monkey == 0,
            jnp.array([152, 5]),
            jnp.array([position_monkey[0], position_monkey[1] + 8]),
        )

        # State 2: Moving left
        pos_state_2 = jnp.array([position_monkey[0] - 3, position_monkey[1]])

        # State 3: Waiting/throwing
        pos_state_3 = position_monkey

        # State 4: Moving right
        pos_state_4 = jnp.array([position_monkey[0] + 3, position_monkey[1]])

        # State 5: Moving up
        pos_state_5 = jnp.where(
            state_monkey == 1,
            jnp.array([146, position_monkey[1]]),
            jnp.array([position_monkey[0], position_monkey[1] - 16]),
        )

        # Select new position based on monkey state
        def new_state(state_monkey):
            return jnp.array(
                [
                    (state_monkey == 0),
                    (state_monkey == 1),
                    (state_monkey == 2),
                    (state_monkey == 3),
                    (state_monkey == 4),
                    (state_monkey == 5),
                ]
            )

        new_pos = jnp.select(
            new_state(new_state_monkey),
            [
                pos_state_0,
                pos_state_1,
                pos_state_2,
                pos_state_3,
                pos_state_4,
                pos_state_5,
            ],
            default=position_monkey,
        )

        # Only apply updates when step counter allows it
        return jnp.where(should_update, new_pos, position_monkey)

    # Apply vectorized function to all monkeys at once
    new_monkey_positions = jax.vmap(
        update_single_monkey_position, in_axes=(0, 0, 0, None)
    )(
        state.level.monkey_states,
        state.level.monkey_positions,
        new_monkey_states,
        state.level.step_counter,
    )

    # update monkey throw timers

    # Vectorized update of monkey throw timers
    def update_timer(new_state, old_state, current_timer, step_counter):
        """Update a single monkey throw timer."""
        return jnp.where(
            new_state == 3,
            jnp.where(
                old_state == 2,
                4,
                jnp.where(step_counter % 16 == 0, current_timer - 1, current_timer),
            ),
            current_timer,
        )

    new_monkey_throw_timers = jax.vmap(update_timer, in_axes=(0, 0, 0, None))(
        new_monkey_states,
        state.level.monkey_states,
        state.level.monkey_throw_timers,
        state.level.step_counter,
    )

    def update_coco_state(
        old_m_state, new_m_state, old_m_timer, new_m_timer, c_state, c_pos_x
    ):
        return jnp.where(
            (old_m_state != 3) & (new_m_state == 3),
            1,
            jnp.where(
                (c_state == 1) & (old_m_timer == 3) & (new_m_timer == 2),
                2,
                jnp.where(c_pos_x <= 15, 0, c_state),
            ),
        )

    new_coco_states = jax.vmap(
        update_coco_state, in_axes=(0, 0, 0, 0, 0, 0)
    )(
        state.level.monkey_states,
        new_monkey_states,
        state.level.monkey_throw_timers,
        new_monkey_throw_timers,
        state.level.coco_states,
        state.level.coco_positions[:, 0],
    )

    def update_coco_positions(
        new_c_state, old_c_state, stepc, old_c_pos, new_m_pos
    ):
        return jnp.where(
            new_c_state == 2,
            jnp.where(
                stepc % 2 == 0,
                jnp.array([old_c_pos[0] - 2, old_c_pos[1]]),
                old_c_pos,
            ),
            jnp.where(
                (new_c_state == 1) & (old_c_state == 0),
                jnp.array(
                    [
                        new_m_pos[0] - 6,
                        jnp.where(
                            state.level.spawn_position,
                            new_m_pos[1] - 5,
                            new_m_pos[1] + MONKEY_HEIGHT - COCONUT_HEIGHT,
                        ),
                    ]
                ),
                old_c_pos,
            ),
        )

    new_coco_positions = jax.vmap(
        update_coco_positions, in_axes=(0, 0, None, 0, 0)
    )(
        new_coco_states,
        state.level.coco_states,
        state.level.step_counter,
        state.level.coco_positions,
        new_monkey_positions,
    )

    flip = jnp.any((state.level.monkey_states != 3) & (new_monkey_states == 3))

    return (
        new_monkey_states,
        new_monkey_positions,
        new_monkey_throw_timers,
        score_addition,
        new_coco_positions,
        new_coco_states,
        flip,
    )


# -------- Game Interface for Reset and Step --------
class JaxKangaroo(JaxEnvironment[KangarooState, KangarooObservation, KangarooInfo]):
    def __init__(self, frameskip: int = 1, reward_funcs: list[callable]=None):
        self.frameskip = frameskip
        self.frame_stack_size = 4
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.action_set = {
            NOOP,
            FIRE,
            UP,
            RIGHT,
            LEFT,
            DOWN,
            UPRIGHT,
            UPLEFT,
            DOWNRIGHT,
            DOWNLEFT 
        }
        self.obs_size = 205
        # self.obs_size = 3+2*2*MAX_PLATFORMS+2*2*MAX_LADDERS+2*MAX_FRUITS+MAX_FRUITS+MAX_FRUITS+2*MAX_BELLS+2*MAX_CHILD+2+4+2*4+2*4+4

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: KangarooObservation) -> chex.Array:
        """Converts the observation to a flat array."""
        obs_leaves = jax.tree.flatten(obs)[0]
        obs_flat= jnp.concatenate([jnp.ravel(leaf) for leaf in obs_leaves])
        return obs_flat

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))

    def observation_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=None,
            dtype=jnp.uint8,
        )


    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key = None) -> Tuple[KangarooObservation, KangarooState, ]:
        state = self.reset_level(1)
        return state.obs_stack, state

    @partial(jax.jit, static_argnums=(0))
    def reset_level(self, next_level=1) -> KangarooState:

        next_level = jnp.clip(next_level, 1, 3)
        level_constants: LevelConstants = get_level_constants(next_level)

        new_state = KangarooState(
            player=PlayerState(
                x=jnp.array(PLAYER_START_X),
                y=jnp.array(PLAYER_START_Y),
                vel_x=jnp.array(0),
                is_crouching=jnp.array(False),
                is_jumping=jnp.array(False),
                is_climbing=jnp.array(False),
                jump_counter=jnp.array(0),
                orientation=jnp.array(1),
                jump_base_y=jnp.array(PLAYER_START_Y),
                landing_base_y=jnp.array(PLAYER_START_Y),
                height=jnp.array(PLAYER_HEIGHT),
                jump_orientation=jnp.array(0),
                climb_base_y=jnp.array(PLAYER_START_Y),
                climb_counter=jnp.array(0),
                punch_left=jnp.array(False),
                punch_right=jnp.array(False),
                cooldown_counter=jnp.array(0),
                chrash_timer=jnp.array(0),
                is_crashing=jnp.array(False),
                last_stood_on_platform_y=jnp.array(1000),
                walk_animation=jnp.array(0),
            ),
            level=LevelState(
                bell_position=level_constants.bell_position,
                bell_timer=jnp.array(0),
                fruit_positions=level_constants.fruit_positions,
                fruit_actives=jnp.ones(3, dtype=jnp.bool_),
                fruit_stages=jnp.zeros(3, dtype=jnp.int32),
                ladder_positions=level_constants.ladder_positions,
                ladder_sizes=level_constants.ladder_sizes,
                platform_positions=level_constants.platform_positions,
                platform_sizes=level_constants.platform_sizes,
                child_position=level_constants.child_position,
                child_timer=jnp.array(0),
                child_velocity=jnp.array(1),
                timer=jnp.array(2000),
                falling_coco_position=jnp.array([13, -1]),
                falling_coco_dropping=jnp.array(False),
                falling_coco_counter=jnp.array(0),
                falling_coco_skip_update=jnp.array(False),
                step_counter=jnp.array(0),
                monkey_states=jnp.zeros(4, dtype=jnp.int32),
                monkey_positions=jnp.array([[152, 5], [152, 5], [152, 5], [152, 5]]),
                monkey_throw_timers=jnp.zeros(4, dtype=jnp.int32),
                spawn_protection=jnp.array(True),
                coco_positions=jnp.array(
                    [[-10, -10], [-10, -10], [-10, -10], [-10, -10]]
                ),
                coco_states=jnp.zeros(4, dtype=jnp.int32),
                spawn_position=jnp.array(False),
                bell_animation=jnp.array(0),
            ),
            score=jnp.array(0),
            current_level=next_level,
            level_finished=jnp.array(False),
            levelup_timer=jnp.array(0),
            reset_coords=jnp.array(False),
            levelup=jnp.array(False),
            lives=jnp.array(3),
            obs_stack=None #fill later
        )
        initial_obs = self._get_observation(new_state)

        def expand_and_copy(x):
            x_expanded = jnp.expand_dims(x, axis=0)
            return jnp.concatenate([x_expanded] * self.frame_stack_size, axis=0)

        # Apply transformation to each leaf in the pytree
        initial_obs = jax.tree.map(expand_and_copy, initial_obs)
        new_state = new_state._replace(obs_stack=initial_obs)
        return new_state


    @partial(jax.jit, static_argnums=(0), donate_argnums=(1))
    def step(
        self, state: KangarooState, action: chex.Array
    ) -> Tuple[KangarooObservation, KangarooState, float, bool, KangarooInfo]:
        reset_cond = jnp.any(jnp.array([action == RESET]))

        # Update player state
        (
            player_x,
            player_y,
            vel_x,
            is_crouching,
            is_jumping,
            is_climbing,
            jump_counter,
            orientation,
            jump_base_y,
            landing_base_y,
            new_player_height,
            new_jump_orientation,
            climb_base_y,
            climb_counter,
            punch_left,
            punch_right,
            cooldown_counter,
            level_finished,
        ) = player_step(state, action)

        new_current_level, new_levelup_timer, new_reset_coords, new_levelup = (
            next_level(state)
        )

        # Handle fruit collection
        score_addition, new_actives, new_fruit_stages, bell_timer = fruits_step(state)
        child_timer, new_child_x, new_child_y, new_child_velocity = child_step(state)

        # Handle Main Timer
        new_main_timer = timer_controller(state)

        (
            new_falling_coco_position,
            new_falling_coco_dropping,
            new_falling_coco_counter,
            new_falling_coco_skip_update,
        ) = falling_coconut_controller(state)

        # update monkeys
        (
            new_monkey_states,
            new_monkey_positions,
            new_monkey_throw_timers,
            score_addition2,
            new_coco_positions,
            new_coco_states,
            flip,
        ) = monkey_controller(state, (punch_left | punch_right))

        (
            new_lives,
            new_is_crashing,
            crash_timer,
            crash_timer_done,
            new_last_stood_on_platform_y,
        ) = lives_controller(state)

        # reset_current_level_progress()

        # add the time after finishing a level
        score_addition3 = jnp.where(level_finished, state.level.timer, 0)

        # add score if levelup from lvl3 to lvl1
        score_addition = score_addition + score_addition2 + score_addition3
        score_addition = jax.lax.cond(
            new_current_level == 4,
            lambda: score_addition + 1400,
            lambda: score_addition,
        )
        new_current_level = jnp.where(new_current_level == 4, 1, new_current_level)

        # set the bell animation counter. if the bell is rung (bell_timer > 0), set the animation to 192 and start counting down
        new_bell_animation_timer = jnp.where(
            bell_timer > 0,
            jnp.where(state.level.bell_animation == 0, 192, state.level.bell_animation),
            jnp.where(
                state.level.bell_animation > 0,
                state.level.bell_animation - 1,
                state.level.bell_animation,
            )
        )


        new_level_state = jax.lax.cond(
            new_levelup,
            lambda: self.reset_level(new_current_level).level,
            lambda: jax.lax.cond(
                crash_timer_done,
                lambda: self.reset_level(state.current_level).level,
                lambda: LevelState(
                    bell_position=state.level.bell_position,
                    fruit_positions=state.level.fruit_positions,
                    ladder_positions=state.level.ladder_positions,
                    ladder_sizes=state.level.ladder_sizes,
                    platform_positions=state.level.platform_positions,
                    platform_sizes=state.level.platform_sizes,
                    child_position=jnp.array([new_child_x, new_child_y]),
                    timer=new_main_timer,
                    bell_timer=bell_timer,
                    child_timer=child_timer,
                    child_velocity=new_child_velocity,
                    fruit_actives=new_actives,
                    fruit_stages=new_fruit_stages,
                    falling_coco_position=jnp.where(
                        state.levelup_timer == 0,
                        new_falling_coco_position,
                        state.level.falling_coco_position,
                    ),
                    falling_coco_dropping=new_falling_coco_dropping,
                    falling_coco_counter=new_falling_coco_counter,
                    falling_coco_skip_update=new_falling_coco_skip_update,
                    step_counter=(state.level.step_counter + 1) % 256,
                    monkey_positions=jnp.where(
                        state.levelup_timer == 0,
                        new_monkey_positions,
                        state.level.monkey_positions,
                    ),
                    monkey_states=new_monkey_states,
                    monkey_throw_timers=new_monkey_throw_timers,
                    spawn_protection=jnp.where(
                        (state.level.step_counter == 255)
                        & state.level.spawn_protection,
                        False,
                        state.level.spawn_protection,
                    ),
                    coco_positions=new_coco_positions,
                    coco_states=new_coco_states,
                    spawn_position=jnp.where(
                        flip,
                        ~state.level.spawn_position,
                        state.level.spawn_position,
                    ),
                    bell_animation=new_bell_animation_timer
                ),
            ),
        )

        # if one of the walk buttons is pressed, increase the walk animation
        currently_walking = jnp.logical_or(
                jnp.logical_or(
                    jnp.logical_or(action == RIGHT, action == LEFT),
                    jnp.logical_or(action == UPRIGHT, action == UPLEFT)
                ),
                jnp.logical_or(action == DOWNRIGHT, action == DOWNLEFT)
            )
        new_walk_counter = jnp.where(
            currently_walking,
            state.player.walk_animation + 1,
            0
        )

        # if the walk_animation is 16, reset to 0
        new_walk_counter = jnp.where(new_walk_counter == 16, 0, new_walk_counter)

        new_player_state = jax.lax.cond(
            crash_timer_done,
            lambda: self.reset_level(state.current_level).player,
            lambda: PlayerState(
                x=player_x,
                y=player_y,
                vel_x=vel_x,
                is_crouching=is_crouching,
                is_jumping=is_jumping,
                is_climbing=is_climbing,
                jump_counter=jump_counter,
                orientation=orientation,
                jump_base_y=jump_base_y,
                landing_base_y=landing_base_y,
                height=new_player_height,
                jump_orientation=new_jump_orientation,
                climb_base_y=climb_base_y,
                climb_counter=climb_counter,
                punch_left=punch_left,
                punch_right=punch_right,
                cooldown_counter=cooldown_counter,
                chrash_timer=crash_timer,
                is_crashing=new_is_crashing,
                last_stood_on_platform_y=new_last_stood_on_platform_y,
                walk_animation=new_walk_counter,
            ),
        )

        # new_state = jax.lax.cond(
        #     reset_cond,
        #     lambda: self.reset_level(1),
        #     lambda: jax.lax.cond(
        #         state.lives <= 0,
        #         lambda: state,
        #         lambda: KangarooState(
        #             player=new_player_state,
        #             level=new_level_state,
        #             score=state.score + score_addition,
        #             current_level=new_current_level,
        #             level_finished=level_finished,
        #             levelup_timer=new_levelup_timer,
        #             reset_coords=new_reset_coords,
        #             levelup=new_levelup,
        #             lives=new_lives,
        #             obs_stack=state.obs_stack,
        #         ),
        #     ),
        # )

        new_state = jax.lax.cond(
            reset_cond,
            lambda: self.reset_level(1),
            lambda: KangarooState(
                player=new_player_state,
                level=new_level_state,
                score=state.score + score_addition,
                current_level=new_current_level,
                level_finished=level_finished,
                levelup_timer=new_levelup_timer,
                reset_coords=new_reset_coords,
                levelup=new_levelup,
                lives=new_lives,
                obs_stack=state.obs_stack,
            ),
        )
        done = self._get_done(new_state)
        env_reward = self._get_env_reward(state, new_state)
        all_rewards = self._get_all_rewards(state, new_state)
        info = self._get_info(new_state, all_rewards)

        observation = self._get_observation(new_state)
        observation = jax.tree.map(lambda stack, obs: jnp.concatenate([stack[1:], jnp.expand_dims(obs, axis=0)], axis=0), new_state.obs_stack, observation)
        new_state = new_state._replace(obs_stack=observation)

        return new_state.obs_stack, new_state, env_reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: KangarooState) -> KangarooObservation:
        return KangarooObservation(
            player_x=state.player.x,
            player_y=state.player.y,
            player_o=state.player.orientation,
            platform_positions=state.level.platform_positions,
            platform_sizes=state.level.platform_sizes,
            ladder_positions=state.level.ladder_positions,
            ladder_sizes=state.level.ladder_sizes,
            fruit_positions=state.level.fruit_positions,
            fruit_actives=state.level.fruit_actives,
            fruit_stages=state.level.fruit_stages,
            bell_position=state.level.bell_position,
            child_position=state.level.child_position,
            falling_coco_position=state.level.falling_coco_position,
            monkey_states=state.level.monkey_states,
            monkey_positions=state.level.monkey_positions,
            coco_positions=state.level.coco_positions,
            coco_states=state.level.coco_states,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: KangarooState, all_rewards: chex.Array) -> KangarooInfo:
        return KangarooInfo(
            score=state.score,
            level=state.current_level,
            all_rewards=all_rewards,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_env_reward(self, previous_state: KangarooState, state: KangarooState) -> float:
        return state.score - previous_state.score

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_rewards(self, previous_state: KangarooState, state: KangarooState) -> chex.Array: 
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array(
            [reward_func(previous_state, state) for reward_func in self.reward_funcs]
        )
        return rewards 

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: KangarooState) -> bool:
        return state.lives <= 0

import jaxatari.rendering.atraJaxis as aj
from jaxatari.renderers import AtraJaxisRenderer

class KangarooRenderer(AtraJaxisRenderer):
    # Type hint for sprites dictionary
    sprites: Dict[str, Any]

    def __init__(self):
        """
        Initializes the renderer by loading sprites, including level backgrounds.

        Args:
            sprite_path: Path to the directory containing sprite .npy files.
        """
        self.sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/kangaroo"
        self.sprites = self._load_sprites()
        # Store background sprites directly for use in render function
        self.background_0 = self.sprites.get('background_0')
        self.background_1 = self.sprites.get('background_1')
        self.background_2 = self.sprites.get('background_2')


    def _load_sprites(self) -> dict[str, Any]:
        """Loads all necessary sprites from .npy files."""
        sprites: Dict[str, Any] = {}

        # Helper function to load a single sprite frame
        def _load_sprite_frame(name: str) -> Optional[chex.Array]:
            path = os.path.join(self.sprite_path, f'{name}.npy')
            frame = aj.loadFrame(path)
            if isinstance(frame, jnp.ndarray) and frame.ndim >= 2:
                return frame.astype(jnp.uint8)


        # --- Load Sprites ---
        # Backgrounds + Dynamic elements + UI elements
        sprite_names = [
            'background_0', 'background_1', 'background_2',
            'ape_climb_left', 'ape_climb_right', 'ape_moving', 'ape_standing',
            'bell', 'ringing_bell', 'child_jump', 'child', 'coconut', 'kangaroo',
            'kangaroo_climb', 'kangaroo_dead', 'kangaroo_ducking',
            'kangaroo_jump_high', 'kangaroo_jump', 'kangaroo_lives',
            'kangaroo_walk', 'kangaroo_boxing',
            'strawberry', 'throwing_ape', 'thrown_coconut', 'time_dash',
        ]
        for name in sprite_names:
            loaded_sprite = _load_sprite_frame(name)
            if loaded_sprite is not None:
                 sprites[name] = loaded_sprite

        # pad the kangaroo and monkey sprites since they have to be used interchangeably (and jax enforces same sizes)
        ape_sprites = aj.pad_to_match([sprites['ape_climb_left'], sprites['ape_climb_right'], sprites['ape_moving'], sprites['ape_standing'], sprites['throwing_ape']])

        sprites['ape_climb_left'] = ape_sprites[0]
        sprites['ape_climb_right'] = ape_sprites[1]
        sprites['ape_moving'] = ape_sprites[2]
        sprites['ape_standing'] = ape_sprites[3]
        sprites['throwing_ape'] = ape_sprites[4]

        # --- pad kangaroo ---
        kangaroo_sprites = aj.pad_to_match([sprites['kangaroo'], sprites['kangaroo_climb'], sprites['kangaroo_dead'], sprites['kangaroo_ducking'], sprites['kangaroo_jump_high'], sprites['kangaroo_jump'], sprites['kangaroo_walk'], sprites['kangaroo_boxing']])

        sprites['kangaroo'] = kangaroo_sprites[0]
        sprites['kangaroo_climb'] = kangaroo_sprites[1]
        sprites['kangaroo_dead'] = kangaroo_sprites[2]
        sprites['kangaroo_ducking'] = kangaroo_sprites[3]
        sprites['kangaroo_jump_high'] = kangaroo_sprites[4]
        sprites['kangaroo_jump'] = kangaroo_sprites[5]
        sprites['kangaroo_walk'] = kangaroo_sprites[6]
        sprites['kangaroo_boxing'] = kangaroo_sprites[7]

        # pad bell / ringing bell
        bell_sprites = aj.pad_to_match([sprites['bell'], sprites['ringing_bell']])

        sprites['bell'] = bell_sprites[0]
        sprites['ringing_bell'] = bell_sprites[1]

        # --- Load Digit Sprites ---
        # Score digits
        score_digit_path = os.path.join(self.sprite_path, 'score_{}.npy')
        digits = aj.load_and_pad_digits(score_digit_path, num_chars=10)
        sprites['digits'] = digits

        # Time digits
        time_digit_path = os.path.join(self.sprite_path, 'time_{}.npy')
        time_digits = aj.load_and_pad_digits(time_digit_path, num_chars=10)
        sprites['time_digits'] = time_digits

        # expand all sprites similar to the Pong/Seaquest loading
        for key in sprites.keys():
            if isinstance(sprites[key], (list, tuple)):
                sprites[key] = [jnp.expand_dims(sprite, axis=0) for sprite in sprites[key]]
            else:
                sprites[key] = jnp.expand_dims(sprites[key], axis=0)

        return sprites

    # Apply JIT compilation. static_argnums=(0,) means 'self' is static.
    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: KangarooState) -> chex.Array:
        """
        Renders the current game state to a JAX array (raster image)
        using pre-rendered backgrounds per level.

        Args:
            state: The current KangarooState.

        Returns:
            A JAX array representing the rendered game screen (HEIGHT, WIDTH, 3), dtype=uint8.
        """

        # --- Select and Render Background ---
        # Initialize raster (optional, could directly use background if it covers all)
        # Starting with zeros allows transparency in dynamic sprites if they use it.
        raster = jnp.zeros((SCREEN_WIDTH, SCREEN_HEIGHT, 3), dtype=jnp.uint8)

        # Get the current level index (ensure it's integer and within bounds 0-2)
        level_idx = state.current_level.astype(int)
        # Clamp index to be safe, although state should ideally be valid
        level_idx = jnp.clip(level_idx, 1, 3)

        selected_background = jax.lax.switch(
            level_idx,
            [
                lambda: jnp.zeros(self.background_0.shape, dtype=self.background_0.dtype),  # Level 0 (empty)
                lambda: self.background_0,  # Level 1
                lambda: self.background_1,  # Level 2
                lambda: self.background_2,  # Level 3
            ]
        )
        selected_background = aj.get_sprite_frame(selected_background, 0)

        raster = aj.render_at(raster, 0, 0, selected_background)

        # --- Removed Wall Rendering ---
        # --- Removed Platform Rendering Loop ---
        # --- Removed Ladder Rendering Loop ---

        # --- Draw fruits (Strawberries) ---
        fruit_sprite = self.sprites.get('strawberry', None)
        fruit_positions = state.level.fruit_positions
        fruit_actives = state.level.fruit_actives

        def _draw_fruit(i, current_raster):
            should_draw = jnp.logical_and(fruit_actives[i], fruit_sprite is not None)
            pos = fruit_positions[i]
            def render_fruit_sprite(raster_to_update):
                return aj.render_at(raster_to_update, pos[0].astype(int), pos[1].astype(int), aj.get_sprite_frame(fruit_sprite, 0))
            return jax.lax.cond(should_draw, render_fruit_sprite, lambda r: r, current_raster)

        num_fruits_to_draw = fruit_positions.shape[0]
        raster = jax.lax.fori_loop(0, num_fruits_to_draw, _draw_fruit, raster)

        # --- Draw Bell ---
        # if the bell_animation is: 192-176, 143-128, 95-80, 47-32 draw the alternate bell sprite
        bell_in_range_left = jnp.logical_or(
            jnp.logical_and(state.level.bell_animation <= 192, state.level.bell_animation >= 176),
            jnp.logical_and(state.level.bell_animation <= 95, state.level.bell_animation >= 80),
        )

        bell_in_range_right = jnp.logical_or(
            jnp.logical_and(state.level.bell_animation <= 143, state.level.bell_animation >= 128),
            jnp.logical_and(state.level.bell_animation <= 47, state.level.bell_animation >= 32)
        )

        bell_sprite = jax.lax.cond(
            jnp.logical_or(bell_in_range_left, bell_in_range_right),
            lambda: self.sprites.get('ringing_bell'),
            lambda: self.sprites.get('bell')
        )

        bell_pos = state.level.bell_position
        not_all_fruits_collected = ~jnp.any(state.level.fruit_stages == 3)
        bell_pos_valid = bell_pos[0] != -1
        sprite_is_valid = bell_sprite is not None
        should_draw_bell = jnp.logical_and(jnp.logical_and(not_all_fruits_collected, bell_pos_valid), sprite_is_valid)

        def draw_bell_func(current_raster):
            return aj.render_at(current_raster, bell_pos[0].astype(int), bell_pos[1].astype(int), aj.get_sprite_frame(bell_sprite, 0), flip_horizontal=bell_in_range_left)
        raster = jax.lax.cond(should_draw_bell, draw_bell_func, lambda r: r, raster)

        # --- Draw monkeys (Apes) ---
        monkey_positions = state.level.monkey_positions
        monkey_states = state.level.monkey_states

        def _draw_monkey(i, current_raster):
            state_idx = monkey_states[i].astype(int)
            pos = monkey_positions[i]
            should_draw = state_idx != 0
            """
            - 0: non-existent
            - 1: moving down
            - 2: moving left
            - 3: throwing
            - 4: moving right
            - 5: moving up
            """
            monkey_sprite = jax.lax.switch(
                state_idx,
                [
                    lambda: self.sprites.get('ape_standing'), # Case 0
                    lambda: self.sprites.get('ape_climb_left'),    # Case 1
                    lambda: self.sprites.get('ape_moving'),    # Case 2
                    lambda: self.sprites.get('throwing_ape'),  # Case 3
                    lambda: self.sprites.get('ape_moving'),    # Case 4
                    lambda: self.sprites.get('ape_climb_right'),# Case 5
                ]
            )

            # in case its state_idx 2 or 4 and the counter is % 16, use standing instead of moving
            monkey_sprite = jax.lax.cond(
                jnp.logical_and(
                    (state.level.step_counter % 32) < 16,
                    jnp.logical_or(state_idx == 2, state_idx == 4)
                ),
                lambda: self.sprites.get('ape_standing'),
                lambda: monkey_sprite
            )

            is_moving_left = (state_idx == 4)
            flip_h = is_moving_left
            sprite_is_valid = monkey_sprite is not None
            should_draw = jnp.logical_and(should_draw, sprite_is_valid)
            def render_monkey_sprite(raster_to_update):
                return aj.render_at(raster_to_update, pos[0].astype(int), pos[1].astype(int), aj.get_sprite_frame(monkey_sprite, 0), flip_horizontal=flip_h)
            return jax.lax.cond(should_draw, render_monkey_sprite, lambda r: r, current_raster)

        num_monkeys_to_draw = monkey_positions.shape[0]
        raster = jax.lax.fori_loop(0, num_monkeys_to_draw, _draw_monkey, raster)

        # --- Draw player (Kangaroo) ---
        player_pos_x = state.player.x
        player_pos_y = state.player.y
        player_orientation = state.player.orientation
        flip_player = player_orientation < 0
        sprite_lambda = jax.lax.cond(
            state.player.is_crashing, lambda: self.sprites.get('kangaroo_dead'),
            lambda: jax.lax.cond(
                state.player.is_climbing, lambda: self.sprites.get('kangaroo_climb'),
                lambda: jax.lax.cond(
                    state.player.is_crouching, lambda: self.sprites.get('kangaroo_ducking'),
                    lambda: jax.lax.cond(
                        state.player.is_jumping, lambda: self.sprites.get('kangaroo_jump'),
                        lambda: jax.lax.cond(
                            state.player.punch_left | state.player.punch_right,
                            lambda: self.sprites.get('kangaroo_boxing'),
                            lambda: self.sprites.get('kangaroo')
                        )
                    )
                )
            )
        )

        # check if player.walk_animation is between 6 and 16 in which range the kangaroo has a different animation
        player_walking_animation = jnp.logical_and(state.player.walk_animation > 6, state.player.walk_animation < 16)

        # in case the new_walk_counter is between 6 and 16, decrease player_y by 1 (its hovering slightly) TODO: does this impact hitboxes?
        player_pos_y = jnp.where(
            player_walking_animation,
            player_pos_y - 1,
            player_pos_y
        )

        sprite_lambda = jax.lax.cond(
            player_walking_animation,
            lambda: self.sprites.get('kangaroo_walk'),
            lambda: sprite_lambda
        )

        # in case the player_animation is between 17 and 25, use high jump
        sprite_lambda = jax.lax.cond(
            jnp.logical_and(state.player.jump_counter > 16, state.player.jump_counter < 25),
            lambda: self.sprites.get('kangaroo_jump_high'),
            lambda: sprite_lambda
        )

        player_sprite = sprite_lambda
        sprite_is_valid = player_sprite is not None
        def render_player_sprite(raster_to_update):
             return aj.render_at(raster_to_update, player_pos_x.astype(int), player_pos_y.astype(int), aj.get_sprite_frame(player_sprite, 0), flip_horizontal=flip_player)
        raster = jax.lax.cond(sprite_is_valid, render_player_sprite, lambda r: r, raster)

        # --- Draw Child ---
        child_pos = state.level.child_position
        is_jumping = (state.level.step_counter % 32) < 16
        # if the velocity is negative, flip horizontal
        child_flip = state.level.child_velocity > 0
        child_sprite_lambda = jax.lax.cond(
            is_jumping, lambda: self.sprites.get('child_jump'), lambda: self.sprites.get('child')
        )
        child_sprite = child_sprite_lambda
        should_draw_child = jnp.logical_and(child_pos[0] != -1, child_sprite is not None)
        def draw_child_func(current_raster):
            return aj.render_at(current_raster, child_pos[0].astype(int), child_pos[1].astype(int), aj.get_sprite_frame(child_sprite, 0), child_flip)
        raster = jax.lax.cond(should_draw_child, draw_child_func, lambda r: r, raster)

        # --- Draw falling coconut ---
        falling_coco_pos = state.level.falling_coco_position
        coco_sprite = self.sprites.get('thrown_coconut', None)
        should_draw_falling_coco = jnp.logical_and(falling_coco_pos[1] != -1, coco_sprite is not None)
        def draw_falling_coco_func(current_raster):
            return aj.render_at(current_raster, falling_coco_pos[0].astype(int), falling_coco_pos[1].astype(int), aj.get_sprite_frame(coco_sprite, 0))
        raster = jax.lax.cond(should_draw_falling_coco, draw_falling_coco_func, lambda r: r, raster)

        # --- Draw thrown coconuts ---
        coco_positions = state.level.coco_positions
        coco_states = state.level.coco_states
        coco_sprite = self.sprites.get('coconut', None)
        def _draw_coco(i, current_raster):
            should_draw = jnp.logical_and(coco_states[i] != 0, coco_sprite is not None)
            pos = coco_positions[i]
            def render_coco_sprite(raster_to_update):
                return aj.render_at(raster_to_update, pos[0].astype(int), pos[1].astype(int), aj.get_sprite_frame(coco_sprite, 0))
            return jax.lax.cond(should_draw, render_coco_sprite, lambda r: r, current_raster)
        num_cocos_to_draw = coco_positions.shape[0]
        raster = jax.lax.fori_loop(0, num_cocos_to_draw, _draw_coco, raster)

        # --- Draw UI ---
        # Score
        digit_sprites = self.sprites.get('digits', None)
        score_digits_indices = aj.int_to_digits(state.score, max_digits=6)
        raster = aj.render_label(raster, 105, 182, score_digits_indices, digit_sprites[0], spacing=8)

        # Lives
        life_sprite = self.sprites.get('kangaroo_lives', None)
        lives_count = jnp.maximum(state.lives.astype(int) - 1, 0)
        raster = aj.render_indicator(raster, 15, 182, lives_count, life_sprite[0], spacing=8)

        # Timer
        time_digit_sprites = self.sprites.get('time_digits', None)
        timer_val = jnp.maximum(state.level.timer.astype(int), 0)
        timer_digits_indices = aj.int_to_digits(timer_val, max_digits=4)
        raster = aj.render_label(raster, 80, 190, timer_digits_indices, time_digit_sprites[0], spacing=4)

        # Ensure the final raster has the correct dtype
        return raster.astype(jnp.uint8)

if __name__ == "__main__":
    pygame.init()
    game = JaxKangaroo()

    scaling = 4

    screen = pygame.display.set_mode((SCREEN_WIDTH * scaling, SCREEN_HEIGHT * scaling))
    pygame.display.set_caption("Kangaroo")
    clock = pygame.time.Clock()

    renderer = KangarooRenderer()
    jitted_step = jax.jit(game.step)
    jitted_reset = jax.jit(game.reset)
    (_, curr_state) = jitted_reset()
    running = True
    frame_by_frame = False
    frameskip = game.frameskip
    counter = 1

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    frame_by_frame = not frame_by_frame
            elif event.type == pygame.KEYDOWN or (
                event.type == pygame.KEYUP and event.key == pygame.K_n
            ):
                if event.key == pygame.K_n and frame_by_frame:
                    if counter % frameskip == 0:
                        action = get_human_action()
                        (_, curr_state, _, _, _) = jitted_step(curr_state, action)

        if not frame_by_frame:
            if counter % frameskip == 0:
                action = get_human_action()
                (_, curr_state, _, _, _) = jitted_step(curr_state, action)

        # Render and display
        raster = renderer.render(curr_state)

        aj.update_pygame(screen, raster, scaling, SCREEN_WIDTH, SCREEN_HEIGHT)

        counter += 1
        pygame.time.Clock().tick(60)

    pygame.quit()
