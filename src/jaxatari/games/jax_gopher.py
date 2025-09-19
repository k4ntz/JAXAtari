# **************************************************************************
# * Authors: Nick Rentschler, Erik Porada, Marc Neumann
# **************************************************************************


import os
from enum import Enum
from functools import partial
from typing import NamedTuple, Tuple, Dict, Any, Callable, TypeVar, Optional

import chex
import jax.lax
import jax.numpy as jnp
import numpy as np
from flax.nnx import fori_loop
from jax import config, core, jit
from jax.lax import cond as jif
from jax.lax import select as jselect
from jax.numpy import array as arr
from jax.numpy import logical_and as jand
from jax.numpy import logical_not as jnot
from jax.numpy import logical_or as jor

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as jr
import jaxatari.spaces as spaces

# ================================ UTIL ALIAS ==================================

byte_arr = partial(arr, dtype=jnp.uint8)
int_arr = partial(arr, dtype=jnp.int32)
float_arr = partial(arr, dtype=jnp.float32)
bool_arr = partial(arr, dtype=jnp.bool)
byte_val = partial(core.concrete_or_error, jnp.uint8)
int_val = partial(core.concrete_or_error, jnp.int32)
float_val = partial(core.concrete_or_error, jnp.float32)
bool_val = partial(core.concrete_or_error, jnp.bool)

ByteArray = jax.Array
ByteValue = jax.Array
IntArray = jax.Array
IntValue = jax.Array
FloatArray = jax.Array
FloatValue = jax.Array
BoolArray = jax.Array
BoolValue = jax.Array

jif_true = partial(jif, false_fun=lambda s: s)

P1 = TypeVar("P1")
P2 = TypeVar("P2")
R = TypeVar("R")


@partial(jit, static_argnames=("true_fun", "false_fun"))
def jif_p2(condition: BoolValue | bool, true_fun: Callable[[P1, P2], R], false_fun: Callable[[P1, P2], R],
           operand: tuple[P1, P2]) -> R:
    return jif(
        condition,
        lambda params: true_fun(params[0], params[1]),
        lambda params: false_fun(params[0], params[1]),
        operand=operand
    )


@partial(jit, static_argnames=("true_fun",))
def jif_p2_true(condition: BoolValue | bool, true_fun: Callable[[P1, P2], P1], operand: tuple[P1, P2]) -> P1:
    return jif(
        condition,
        lambda params: true_fun(params[0], params[1]),
        lambda params: params[0],
        operand=operand
    )


class GopherConstants:
    SCREEN_MIN_X = 0
    SCREEN_MAX_X = 159

    GOPHER_MIN_X = SCREEN_MIN_X + 3
    GOPHER_MAX_X = SCREEN_MAX_X - 11
    DUCK_MIN_X = SCREEN_MIN_X + 12
    DUCK_MAX_X = SCREEN_MAX_X - 11
    FARMER_MIN_X = SCREEN_MIN_X + 20
    FARMER_MAX_X = SCREEN_MAX_X - 11

    HOLE_0_X = 15
    HOLE_1_X = 31
    HOLE_2_X = 47
    HOLE_3_X = 111
    HOLE_4_X = 127
    HOLE_5_X = 143

    DIRT_LAYER_Y_INDICES = int_arr([0, 1, 2, 3])

    CARROT_0_X = 63
    CARROT_1_X = 79
    CARROT_2_X = 95

    SEED_GROUND_LEVEL = 107
    # Exclusive
    SEED_MAX_CATCHING_Y = 87
    # Inclusive
    SEED_MIN_CATCHING_Y = 83

    # Initial position constants
    FARMER_INIT_X = (SCREEN_MAX_X // 2) + 4
    GOPHER_INIT_X = GOPHER_MAX_X - 1
    SEED_INIT_Y = 8

    # Game State values
    GS_MAIN_GAME_LOOP = 0
    GS_GOPHER_STOLE_CARROT = 1
    GS_DUCK_WAIT = 2
    GS_RESET_AFTER_ROUND = 3
    GS_CHECK_FOR_GAME_OVER = 4
    GS_INIT_NEXT_ROUND = 5
    GS_FINISHED_NEXT_ROUND_SETUP = 6
    GS_PAUSE_FOR_ACTION_BUTTON = 7
    GS_WAIT_FOR_NEW_GAME = 8

    # Duck constants
    DUCK_INIT_ANIMATION_TIMER = 32
    DUCK_ANIMATION_DOWN_WING = DUCK_INIT_ANIMATION_TIMER - 8
    DUCK_ANIMATION_STATIONARY_WING = DUCK_ANIMATION_DOWN_WING - 8
    DUCK_ANIMATION_UP_WING = DUCK_ANIMATION_STATIONARY_WING - 8

    SEED_TARGET_X_MASK = 0b01111111

    # Gopher constants
    GOPHER_Y_TARGET_MASK = 0x0F
    GOPHER_Y_LOCKED_BIT = 0b10000000
    GOPHER_Y_DIFFICULT_TARGET_MASK = 0x08  # 0b00001000

    # 0b1000 0000 = First 1 bit means current direction of gopher, 1 = left, 0 = right
    GOPHER_X_DIR_MASK = 0b10000000
    # Which tunnel to target
    GOPHER_TUNNEL_TARGET_MASK = 0b00000111

    GOPHER_Y_UNDERGROUND = 0
    GOPHER_Y_ABOVE_GROUND = 35

    # Seed constants
    SEED_INIT_DECAYING_TIMER = 120

    POINTS_FILL_TUNNEL = 20
    POINTS_BONK_GOPHER = 100

    # Wait timer constants
    WAIT_TIME_CARROT_STOLEN = 136  # wait 119 frames ~ 2 seconds

    # Audio Value constants
    END_AUDIO_TUNE = 0
    AUDIO_DURATION_MASK = 0xE0

    # Memory addresses

    MEM_AUDIO_VALUES = 0xFD0A
    MEM_AUDIO_BONK_GOPHER = 0xFD43
    MEM_AUDIO_GOPHER_TAUNT = 0xFD4A
    MEM_AUDIO_STOLEN_CARROT = 0xFD61
    MEM_AUDIO_DIG_TUNNEL = 0xFD7A
    MEM_AUDIO_FILL_TUNNEL = 0xFD7E
    MEM_AUDIO_DUCK_QUACKING = 0xFD84
    MEM_AUDIO_GAME_OVER_THEME_1 = 0xFD8C
    MEM_AUDIO_GAME_OVER_THEME_2 = 0xFD9F

    # Data tables
    TUNNEL_X_POSITIONS = byte_arr([
        HOLE_0_X, HOLE_1_X,
        HOLE_2_X, HOLE_3_X,
        HOLE_4_X, HOLE_5_X,
        HOLE_0_X, HOLE_5_X
    ])
    CARROT_X_POSITIONS = byte_arr([
        CARROT_2_X, CARROT_1_X, CARROT_0_X
    ])
    REVERSED_CARROT_X_POSITIONS = byte_arr([
        CARROT_0_X, CARROT_1_X, CARROT_2_X
    ])

    GOPHER_Y_TAUNTING = GOPHER_Y_ABOVE_GROUND - 1
    GOPHER_Y_TARGET_UNDER_GROUND = 0

    GOPHER_Y_TARGET_POSITIONS = byte_arr([
        GOPHER_Y_UNDERGROUND,
        GOPHER_Y_UNDERGROUND + 7,
        GOPHER_Y_UNDERGROUND + 14,
        GOPHER_Y_ABOVE_GROUND - 13,
        GOPHER_Y_TAUNTING,  # taunting
        GOPHER_Y_ABOVE_GROUND,  # stealing
        GOPHER_Y_UNDERGROUND + 7,
        GOPHER_Y_UNDERGROUND + 14,

        # When score >= 10,000 always draws out of these:
        GOPHER_Y_ABOVE_GROUND - 13,
        GOPHER_Y_ABOVE_GROUND,  # stealing
        GOPHER_Y_TAUNTING,  # taunting
        GOPHER_Y_UNDERGROUND + 14,
        GOPHER_Y_ABOVE_GROUND - 13,
        GOPHER_Y_ABOVE_GROUND,  # stealing
        GOPHER_Y_TAUNTING,  # taunting
        GOPHER_Y_ABOVE_GROUND  # stealing
    ])

    AUDIO_STARTING_THEME_1 = [
        4,  # high pitch square wave pure tone
        6 << 4 | 15, 7 << 4 | 1, 7 << 4 | 3, 7 << 4 | 4, 7 << 4 | 3,
        7 << 4 | 1, 7 << 4 | 3, 7 << 4 | 10, 7 << 4 | 15, 7 << 4 | 13,
        7 << 4 | 10, 7 << 4 | 7, 7 << 4 | 10, 7 << 4 | 15, 28 << 3 | 26,
        6 << 4 | 15, 7 << 4 | 3, 6 << 4 | 15, 7 << 4 | 3, 7 << 4 | 1,
        7 << 4 | 4, 7 << 4 | 1, 7 << 4 | 10, 7 << 4 | 4, 7 << 4 | 2,
        7 << 4 | 1, 7 << 4 | 0, 16 << 3 | 15, 20 << 3 | 9, END_AUDIO_TUNE
    ]
    AUDIO_STARTING_THEME_2 = [
        12,  # lower pitch square wave sound
        7 << 4 | 1, 7 << 4 | 1, 28 << 3 | 26, 7 << 4 | 4, 7 << 4 | 4,
        7 << 4 | 1, 7 << 4 | 1, 28 << 3 | 15, 28 << 3 | 17, 7 << 4 | 1,
        7 << 4 | 4, 7 << 4 | 1, 7 << 4 | 4, 7 << 4 | 3, 7 << 4 | 7,
        7 << 4 | 3, 7 << 4 | 7, 7 << 4 | 1, 7 << 4 | 2, 7 << 4 | 3,
        7 << 4 | 6, 7 << 4 | 4, 7 << 4 | 1, 7 << 4 | 10, END_AUDIO_TUNE
    ]
    AUDIO_BONK_GOPHER = [
        12,  # lower pitch square wave sound
        1 << 4 | 10, 1 << 4 | 2, 0 << 4 | 11, 0 << 4 | 6, 0 << 4 | 1,
        END_AUDIO_TUNE]
    AUDIO_GOPHER_TAUNT = [
        4,  # high pitch square wave pure tone
        3 << 4 | 7, 1 << 4 | 0, 3 << 4 | 7, 1 << 4 | 0, 1 << 4 | 7,
        3 << 4 | 11, 1 << 4 | 3, 3 << 4 | 11, 1 << 4 | 4, 0 << 4 | 14,
        1 << 4 | 4, 0 << 4 | 14, 1 << 4 | 4, 3 << 4 | 7, 1 << 4 | 0,
        3 << 4 | 7, 1 << 4 | 0, 1 << 4 | 7, 1 << 4 | 11, 1 << 4 | 3,
        3 << 4 | 11, END_AUDIO_TUNE]
    AUDIO_STOLEN_CARROT = [
        7,  # low and buzzy
        1 << 4 | 3, 0 << 4 | 7, 1 << 4 | 3, 0 << 4 | 7, 1 << 4 | 2,
        0 << 4 | 6, 1 << 4 | 2, 0 << 4 | 6, 1 << 4 | 1, 0 << 4 | 5,
        1 << 4 | 1, 0 << 4 | 5, 1 << 4 | 0, 0 << 4 | 4, 0 << 4 | 15,
        0 << 4 | 3, 0 << 4 | 14, 0 << 4 | 2, 0 << 4 | 13, 0 << 4 | 2,
        0 << 4 | 12, 0 << 4 | 1, 7 << 4 | 2, END_AUDIO_TUNE]
    AUDIO_DIG_TUNNEL = [
        8,  # white noise
        0 << 4 | 4, 0 << 4 | 3, END_AUDIO_TUNE]
    AUDIO_FILL_TUNNEL = [
        6,  # bass sound
        0 << 4 | 1, 0 << 4 | 4, 0 << 4 | 2, 0 << 4 | 6, END_AUDIO_TUNE
    ]
    AUDIO_DUCK_QUACK = [
        1,  # saw waveform
        0 << 4 | 15, 0 << 4 | 14, 2 << 4 | 13, 2 << 4 | 12, 4 << 4 | 11,
        0 << 4 | 12, END_AUDIO_TUNE]
    AUDIO_GAME_OVER_THEME_1 = [
        4,  # high pitch square wave pure tone
        28 << 3 | 7, 28 << 3 | 11, 28 << 3 | 17, 28 << 3 | 26, 3 << 4 | 3,
        2 << 4 | 0, 3 << 4 | 3, 2 << 4 | 0, 3 << 4 | 3, 2 << 4 | 0,
        3 << 4 | 3, 2 << 4 | 0, 7 << 4 | 4, 6 << 4 | 0, 7 << 4 | 4,
        6 << 4 | 0, 7 << 4 | 15, END_AUDIO_TUNE]
    AUDIO_GAME_OVER_THEME_2 = [
        4,  # high pitch square wave pure tone
        28 << 3 | 11, 28 << 3 | 17, 28 << 3 | 26, 28 << 3 | 19, 3 << 4 | 7,
        2 << 4 | 0, 3 << 4 | 7, 2 << 4 | 0, 3 << 4 | 7, 2 << 4 | 0,
        3 << 4 | 7, 2 << 4 | 0, 7 << 4 | 10, 6 << 4 | 0, 7 << 4 | 10,
        6 << 4 | 0, 7 << 4 | 3, END_AUDIO_TUNE]

    AUDIO_VALUES = byte_arr(AUDIO_STARTING_THEME_1 + AUDIO_STARTING_THEME_2 + \
                            AUDIO_BONK_GOPHER + AUDIO_GOPHER_TAUNT + AUDIO_STOLEN_CARROT + AUDIO_DIG_TUNNEL + \
                            AUDIO_FILL_TUNNEL + AUDIO_DUCK_QUACK + AUDIO_GAME_OVER_THEME_1 + AUDIO_GAME_OVER_THEME_2)


# =============================== GAME STATE ===================================


class FarmerRenderingState(Enum):
    STANDING = 0
    DIGGING_1 = 1
    DIGGING_2 = 2


class GopherRenderState(Enum):
    DISABLED = 0
    UNDERGROUND_RUNNING_1 = 1
    UNDERGROUND_RUNNING_2 = 2
    RISING = 3
    TAUNTING_1 = 4
    TAUNTING_2 = 5
    ABOVE_GROUND_RUNNING_1 = 6
    ABOVE_GROUND_RUNNING_2 = 7


class DuckRenderState(Enum):
    DISABLED = 0
    STATIONARY = 1
    UP = 2
    DOWN = 3


class GopherState(NamedTuple):
    # Game variants/modes
    expert_difficulty: BoolValue = bool_val(True)
    is_duck_mode_enabled: BoolValue = bool_val(False)

    # Main
    score_value: IntValue = int_val(0)
    garden_dirt_flags: BoolArray = bool_arr([False] * 160)

    game_state: ByteValue = byte_val(0)
    frame_count: ByteValue = byte_val(0)

    # Duck
    duck_x: ByteValue = byte_val(0)
    duck_moving_left: BoolValue = bool_val(0)
    duck_seed_drop_target: ByteValue = byte_val(0)
    duck_animation_timer: ByteValue = byte_val(0)

    # Seed
    seed_x: ByteValue = byte_val(0)
    seed_y: ByteValue = byte_val(0)
    seed_visible: BoolValue = bool_val(0)
    held_seed_decaying_timer: ByteValue = byte_val(0)

    # Farmer
    farmer_x: ByteValue = byte_val(0)
    farmer_animation_idx: ByteValue = byte_val(0)

    # Gopher
    gopher_x: ByteValue = byte_val(0)
    gopher_y: ByteValue = byte_val(0)
    gopher_change_direction_delay: ByteValue = byte_val(0)

    gopher_animation_rate: ByteValue = byte_val(0)
    gopher_facing_left: BoolValue = bool_val(0)
    gopher_carrot_target_id: ByteValue = byte_val(0)
    gopher_tunnel_target_id: ByteValue = byte_val(0)
    gopher_targeting_carrot: BoolValue = bool_val(False)
    gopher_target_move_left: BoolValue = bool_val(False)
    gopher_y_target_id: ByteValue = byte_val(0)
    gopher_y_target_extra_flag: BoolValue = bool_val(False)
    gopher_y_locked: BoolValue = bool_val(0)
    gopher_change_direction_timer: ByteValue = byte_val(0)
    gopher_taunt_timer: ByteValue = byte_val(0)

    # Carrots
    carrots_left: BoolArray = bool_arr([False, False, False])  # Right to left

    # Random
    random_0: ByteValue = byte_val(0)
    random_1: ByteValue = byte_val(0)

    # Audio
    audio_index_values: IntArray = byte_arr([0] * 2)
    audio_duration_values: IntArray = byte_arr([0] * 2)
    audio_channel_index: ByteValue = byte_val(0)

    # Misc
    action_button_prev_pressed: BoolValue = bool_val(False)

    # Render states
    gopher_render_state: ByteValue = byte_val(GopherRenderState.DISABLED.value)
    farmer_render_state: ByteValue = byte_val(FarmerRenderingState.STANDING.value)
    duck_render_state: ByteValue = byte_val(DuckRenderState.DISABLED.value)
    carrot_render_state: BoolArray = bool_arr([False, False, False])  # Right to left

    # Input
    left_pressed: BoolValue = bool_val(False)
    right_pressed: BoolValue = bool_val(False)
    fire_pressed: BoolValue = bool_val(False)
    carry_state: ByteValue = byte_val(0)

    # Control flag
    is_first_frame: BoolValue = bool_val(True)


class EntityPosition(NamedTuple):
    x: IntValue
    y: IntValue


class GopherObservation(NamedTuple):
    farmer_x: IntValue
    farmer_animation_state: IntValue

    # TODO: It might be worth considering to add the previous location of the gopher as well
    #       to give a sense of direction to a RL agent, since it's movement is often predictable.
    gopher_position: EntityPosition

    duck_x: IntValue
    duck_moving_left: ByteValue
    duck_active: ByteValue

    seed_position: EntityPosition
    seed_active: ByteValue
    seed_held_by_farmer: ByteValue

    score: IntValue

    carrot_states: BoolArray

    # For every hole shows how many layers you can still fill up
    # 0 = No hole, 4 = Hole till underground
    hole_states: IntArray
    # TODO: It might be worth considering to add the states of all dirt tiles as well
    #       to allow an RL agent to predict possible turn-around points for the Gopher


class GopherInfo(NamedTuple):
    # Game variant information
    expert_difficulty: BoolValue
    duck_mode_enabled: BoolValue
    all_rewards: chex.Array


# ================================ JAX UTIL ====================================

@partial(jit, static_argnums=(1,))
def get_first_index(mask: jax.Array, default_value: int = -1) -> IntValue:
    # Check if we found any match
    has_match = jnp.any(mask)

    def find_first_index(mask: jax.Array):
        # Find the cumulative sum - this will be 1 at the first True value, and increase after
        cumsum = jnp.cumsum(mask)

        # This creates a mask that's True only for the first element where condition is True
        first_match_mask = cumsum == 1

        first_index = jnp.argmax(first_match_mask)
        return first_index

    return jif(
        has_match,
        find_first_index,
        lambda _: int_val(default_value),
        operand=mask
    )


@jit
def shift_right_with_carry(byte_value: ByteValue) -> tuple[ByteValue, ByteValue]:
    result = jnp.right_shift(byte_value, 1)
    carry_value = jselect(jnp.bitwise_and(byte_value, 0x1) != 0, byte_val(1), byte_val(0))
    return result, carry_value


@jit
def roll_left_with_carry(byte_value: ByteValue, carry_bit: ByteValue) -> tuple[ByteValue, ByteValue]:
    new_carry_bit = jnp.bitwise_and(jnp.right_shift(byte_value, 7), 1)
    new_byte_value = jnp.bitwise_or(jnp.bitwise_and(jnp.left_shift(byte_value, 1), 0xFF), carry_bit)
    return new_byte_value, new_carry_bit


@jit
def adc_with_carry(accumulator: ByteValue, memory_value: ByteValue, carry_flag: ByteValue) -> tuple[
    ByteValue, ByteValue]:
    # Cast to 16 bit to capture overflow
    result = accumulator.astype(jnp.uint16) + memory_value + carry_flag

    carry_flag = jselect(result > 0xFF, byte_val(1), byte_val(0))

    # Cast back to byte
    result = jnp.bitwise_and(result, 0xFF).astype(jnp.uint8)

    return result, carry_flag


# Use bitwise NOT with masking to keep only 8 bits
@jit
def flip_byte(byte_value: ByteValue) -> ByteValue:
    # ~ flips all bits, then we use & 0xFF to keep only the lowest 8 bits
    return jnp.bitwise_and(jnp.bitwise_not(byte_value), 0xFF)


# Check if left most bit (bit 7) is set.
@jit
def is_msb_set(byte_value: ByteValue) -> BoolValue:
    # Create mask with 1 in the leftmost position (2^7 = 128)
    mask = 1 << 7  # or simply use 128
    return jnp.bitwise_and(byte_value, mask) != 0


# =============================== JAXATARI ENV =================================

class JaxGopher(JaxEnvironment[GopherState, GopherObservation, GopherInfo, None]):
    def __init__(self, consts: GopherConstants | None = None,
                 reward_funcs: list[Callable[[GopherState, GopherState], chex.Array]] | None = None):
        consts = consts or GopherConstants()
        super().__init__(consts)
        self.action_set = [Action.NOOP, Action.LEFT, Action.RIGHT, Action.LEFTFIRE, Action.RIGHTFIRE, Action.FIRE]

        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs

        # Init computed constants/cache
        self.init_garden_dirt_flags = bool_arr(self.init_garden_dirt_flags())
        self.y_position_to_garden_tile_y_map = byte_arr(self.init_y_position_to_garden_tile_y_map())

        self.renderer = GopherRenderer(self.consts)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: Optional[jax.random.PRNGKey] = None, duck_enabled: chex.Array | bool = True,
              expert_difficulty: chex.Array | bool = True) -> Tuple[GopherObservation, GopherState]:
        state: GopherState = self.reset_to_main_loop(GopherState(), duck_enabled, expert_difficulty)

        if key is not None:
            # Use true random initialization, if so desired
            # Otherwise uses the deterministic initial random state the atari game originally has
            random_bytes = jax.random.randint(key, shape=(2,), minval=0, maxval=256, dtype=jnp.uint8)
            state = state._replace(
                random_0=random_bytes[0],
                random_1=random_bytes[1]
            )

        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: GopherState, action: int) -> tuple[
        GopherObservation, GopherState, FloatValue, BoolValue, GopherInfo]:
        # This does the "overscan" routine of the original atari game
        # which is supposed to happen after the rendering.
        # But since rendering only happens after the step function,
        # we do the "overscan" step from the last frame now (if not first frame)
        new_state = jif(
            state.is_first_frame,
            lambda s: s._replace(
                is_first_frame=bool_val(False)
            ),
            self.update_game_state,
            operand=state
        )

        # Input logic
        new_state = new_state._replace(
            fire_pressed=jor(
                jor(
                    jselect(
                        action == Action.LEFTFIRE, bool_val(True), bool_val(False)
                    ),
                    jselect(
                        action == Action.RIGHTFIRE, bool_val(True), bool_val(False)
                    )
                ), jselect(
                    action == Action.FIRE, bool_val(True), bool_val(False)
                )
            ),
            left_pressed=jor(
                jselect(
                    action == Action.LEFT, bool_val(True), bool_val(False)
                ),
                jselect(
                    action == Action.LEFTFIRE, bool_val(True), bool_val(False)
                )
            ),
            right_pressed=jor(
                jselect(
                    action == Action.RIGHT, bool_val(True), bool_val(False)
                ),
                jselect(
                    action == Action.RIGHTFIRE, bool_val(True), bool_val(False)
                )
            )
        )
        new_state: GopherState = self.update_game(new_state)

        done = self._get_done(new_state)
        reward = self._get_reward(state, new_state)
        obs = self._get_observation(new_state)
        all_rewards = self._get_all_reward(state, new_state)
        info = self._get_info(new_state, all_rewards)

        return obs, new_state, reward, done, info

    def render(self, state: GopherState) -> jnp.ndarray:
        return self.renderer.render(state)

    def action_space(self) -> spaces.Discrete:
        # See self.action_set for actual action space
        return spaces.Discrete(6)

    def observation_space(self) -> spaces.Space:
        return spaces.Dict({
            "farmer_x": spaces.Box(low=self.consts.FARMER_MIN_X - 1, high=self.consts.FARMER_MAX_X, shape=(),
                                   dtype=jnp.uint8),
            "farmer_animation_state": spaces.Box(low=0, high=2, shape=(), dtype=jnp.uint8),
            "gopher_position": spaces.Dict({
                "x": spaces.Box(low=self.consts.GOPHER_MIN_X, high=self.consts.GOPHER_MAX_X + 1, shape=(),
                                dtype=jnp.int32),
                "y": spaces.Box(low=self.consts.GOPHER_Y_UNDERGROUND, high=self.consts.GOPHER_Y_ABOVE_GROUND, shape=(),
                                dtype=jnp.int32)
            }),
            "duck_x": spaces.Box(low=self.consts.DUCK_MIN_X - 1, high=self.consts.DUCK_MAX_X, shape=(),
                                 dtype=jnp.uint8),
            "duck_moving_left": spaces.Box(low=0, high=1, shape=(), dtype=jnp.uint8),
            "duck_active": spaces.Box(low=0, high=1, shape=(), dtype=jnp.uint8),
            "seed_position": spaces.Dict({
                # 0 when deactivated
                "x": spaces.Box(low=min(self.consts.DUCK_MIN_X - 1, 0), high=self.consts.DUCK_MAX_X, shape=(),
                                dtype=jnp.int32),
                # 0 when deactivated
                "y": spaces.Box(low=min(self.consts.SEED_INIT_Y, 0), high=self.consts.SEED_GROUND_LEVEL, shape=(),
                                dtype=jnp.int32)
            }),
            "seed_active": spaces.Box(low=0, high=1, shape=(), dtype=jnp.uint8),
            "seed_held_by_farmer": spaces.Box(low=0, high=1, shape=(), dtype=jnp.uint8),
            "score": spaces.Box(low=0, high=999999, shape=(), dtype=jnp.int32),
            # True / False for every carrot
            "carrot_states": spaces.Box(low=0, high=1, shape=(3,), dtype=jnp.uint8),
            "hole_states": spaces.Box(low=0, high=4, shape=(6,), dtype=jnp.uint8)
        })

    def image_space(self) -> spaces.Space:
        return spaces.Box(
            low=0,
            high=255,
            shape=(210, 160, 3),
            dtype=jnp.uint8
        )

    def obs_to_flat_array(self, obs: GopherObservation) -> jnp.ndarray:
        return jnp.concatenate([
            obs.farmer_x.flatten(),
            obs.farmer_animation_state.flatten(),
            obs.gopher_position.x.flatten(),
            obs.gopher_position.y.flatten(),
            obs.duck_x.flatten(),
            obs.duck_moving_left.flatten(),
            obs.duck_active.flatten(),
            obs.seed_position.x.flatten(),
            obs.seed_position.y.flatten(),
            obs.seed_active.flatten(),
            obs.seed_held_by_farmer.flatten(),
            obs.score.flatten(),
            obs.carrot_states.flatten(),
            obs.hole_states.flatten()
        ])

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, prev_state: GopherState, state: GopherState) -> FloatValue:
        return state.score_value - prev_state.score_value

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: GopherState) -> BoolValue:
        return jnot(jnp.any(state.carrots_left))
        # Or, for more graceful exit:
        # return state.game_state == GS_WAIT_FOR_NEW_GAME

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: GopherState) -> GopherObservation:
        return GopherObservation(
            farmer_x=state.farmer_x,
            farmer_animation_state=state.farmer_render_state,

            gopher_position=EntityPosition(
                state.gopher_x,
                state.gopher_y
            ),

            duck_x=state.duck_x,
            duck_moving_left=jnp.where(state.duck_moving_left, byte_val(1), byte_val(0)),
            duck_active=jnp.where(state.duck_animation_timer != 0, byte_val(1), byte_val(0)),

            seed_position=EntityPosition(
                state.seed_x,
                state.seed_y
            ),
            seed_active=jnp.where(state.seed_visible, byte_val(1), byte_val(0)),
            seed_held_by_farmer=jnp.where(state.held_seed_decaying_timer != 0, byte_val(1), byte_val(0)),

            score=state.score_value,
            carrot_states=jnp.where(state.carrots_left, byte_val(1), byte_val(0)),
            hole_states=jax.vmap(lambda x: self.get_first_dirt_index_for_hole(state, x))(int_arr(
                [self.consts.HOLE_0_X, self.consts.HOLE_1_X, self.consts.HOLE_2_X,
                 self.consts.HOLE_3_X, self.consts.HOLE_4_X, self.consts.HOLE_5_X]
            ))
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(self, previous_state: GopherState, state: GopherState):
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array(
            [reward_func(previous_state, state) for reward_func in self.reward_funcs]
        )
        return rewards

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: GopherState, all_rewards: chex.Array = None) -> GopherInfo:
        return GopherInfo(
            expert_difficulty=state.expert_difficulty,
            duck_mode_enabled=state.is_duck_mode_enabled,
            all_rewards=all_rewards
        )

    # =========================== GAME UTIL ===================================
    def init_y_position_to_garden_tile_y_map(self):
        y_map = [3]
        for y in range(1, 7):
            y_map.append(2)
        for y in range(7, 14):
            y_map.append(1)
        for y in range(14, self.consts.GOPHER_Y_ABOVE_GROUND):
            y_map.append(0)
        return y_map

    def init_garden_dirt_flags(self):
        garden_dirt_flags = [False] * 160

        # Making space for gopher init position
        for x in range(0, 4):
            garden_dirt_flags[3 * 40 + x] = True
        for x in range(36, 40):
            garden_dirt_flags[3 * 40 + x] = True

        return garden_dirt_flags

    @partial(jit, static_argnums=(0,))
    def get_first_dirt_index_for_hole(self, s: GopherState, hole_x_position: IntValue):
        # Check for left dirt flag, exactly like the fill hole logic from the farmer

        # mapped = jax.lax.map(
        #     lambda y_pos: jnot(s.garden_dirt_flags[get_garden_flag_index(hole_x_idx, y_pos)]),
        #     byte_arr([0, 1, 2, 3])
        # )
        # first_dirt_y_pos = get_first_index(mapped, default_value=4)

        dirt_flags = jnot(
            s.garden_dirt_flags[
                self.get_garden_flag_index(self.get_garden_flag_x_index(hole_x_position),
                                           self.consts.DIRT_LAYER_Y_INDICES)]
        )

        return get_first_index(dirt_flags, default_value=4)

    # =========================== GAME LOGIC ===================================

    @partial(jit, static_argnums=(0,))
    def reset_to_main_loop(self, s: GopherState, duck_enabled: BoolValue, expert_difficulty: BoolValue) -> GopherState:
        s = s._replace(
            expert_difficulty=expert_difficulty,
            game_state=byte_val(self.consts.GS_MAIN_GAME_LOOP),
            carrots_left=bool_arr([True, True, True]),
            gopher_change_direction_delay=byte_val(15),

            gopher_y_target_extra_flag=bool_val(True),

            gopher_tunnel_target_id=byte_val(7),
            frame_count=byte_val(255),
            audio_index_values=byte_arr([56, 30]),
            carry_state=byte_val(1)
        )

        s = self.init_game_round_data(s)

        s = jif(
            duck_enabled,
            lambda s: s._replace(
                is_duck_mode_enabled=bool_val(True),
                gopher_y_target_id=byte_val(2),
                random_0=byte_val(91),
                random_1=byte_val(200),
                audio_duration_values=byte_arr([0, 12])
            ),
            lambda s: s._replace(
                gopher_y_target_id=byte_val(10),
                random_0=byte_val(131),
                random_1=byte_val(110),
                audio_duration_values=byte_arr([0, 14])
            ),
            operand=s
        )

        s = s._replace(
            gopher_target_move_left=bool_val(False),
            gopher_tunnel_target_id=byte_val(7),
            frame_count=byte_val(255),
            audio_index_values=byte_arr([56, 30]),
            carry_state=byte_val(1)
        )
        return s

    @partial(jit, static_argnums=(0,))
    def init_game_round_data(self, s: GopherState) -> GopherState:
        return self.init_garden_dirt_values(
            s._replace(
                duck_render_state=byte_val(DuckRenderState.DISABLED.value),
                seed_y=byte_val(0),
                seed_visible=bool_val(False),
                farmer_render_state=byte_val(FarmerRenderingState.STANDING.value),
                gopher_render_state=byte_val(GopherRenderState.UNDERGROUND_RUNNING_1.value),
                farmer_x=byte_val(self.consts.FARMER_INIT_X),
                gopher_x=byte_val(self.consts.GOPHER_INIT_X),
                duck_x=byte_val(self.consts.GOPHER_INIT_X),
                gopher_change_direction_timer=s.gopher_change_direction_delay,
                gopher_y=byte_val(0),
                gopher_facing_left=bool_val(True),
                held_seed_decaying_timer=byte_val(0),
                duck_animation_timer=byte_val(0)
            )
        )

    @partial(jit, static_argnums=(0,))
    def init_garden_dirt_values(self, s: GopherState):
        return s._replace(
            garden_dirt_flags=self.init_garden_dirt_flags,
            gopher_taunt_timer=byte_val(0),
            gopher_y_target_id=jnp.bitwise_and(s.random_0, self.consts.GOPHER_Y_TARGET_MASK),
            gopher_y_locked=bool_val(False),
            gopher_y_target_extra_flag=jnp.bitwise_and(s.random_0,
                                                       (
                                                               ~self.consts.GOPHER_Y_TARGET_MASK & 0xFF & ~self.consts.GOPHER_Y_LOCKED_BIT)) != 0,
            gopher_target_move_left=jnp.bitwise_and(s.random_1, self.consts.GOPHER_X_DIR_MASK) != 0,
            gopher_tunnel_target_id=jnp.bitwise_and(s.random_1, self.consts.GOPHER_TUNNEL_TARGET_MASK),
            gopher_targeting_carrot=bool_val(False),
            gopher_carrot_target_id=byte_val(0)
        )

    @partial(jit, static_argnums=(0,))
    def get_garden_flag_x_index(self, x_pos: IntValue) -> IntValue:
        return x_pos // 4

    @partial(jit, static_argnums=(0,))
    def get_garden_flag_y_index(self, y_pos: IntValue) -> IntValue:
        return self.y_position_to_garden_tile_y_map[y_pos]

    @partial(jit, static_argnums=(0,))
    def get_garden_flag_index_from_pos(self, x_pos: IntValue, y_pos: IntValue) -> IntValue:
        x_idx = self.get_garden_flag_x_index(x_pos)
        y_idx = self.get_garden_flag_y_index(y_pos)

        return self.get_garden_flag_index(x_idx, y_idx)

    @partial(jit, static_argnums=(0,))
    def get_garden_flag_index(self, x_idx: IntValue, y_idx: IntValue) -> IntValue:
        return y_idx * 40 + x_idx

    @partial(jit, static_argnums=(0,))
    def update_game(self, s: GopherState) -> GopherState:
        # Increment frameCount
        s = s._replace(frame_count=s.frame_count + 1)

        s, carry = self.play_game_audio(s, s.carry_state)
        s = self.next_random(s, carry)

        s = self.update_duck(s)
        s = self.update_seed(s)

        s = s._replace(
            carrot_render_state=s.carrots_left
        )

        return jif_true(
            s.game_state == self.consts.GS_MAIN_GAME_LOOP,
            lambda s: self.update_farmer(self.update_gopher_digging(s)),
            operand=s
        )

    @partial(jit, static_argnums=(0,))
    def update_duck(self, s: GopherState) -> GopherState:
        def update_animation(s: GopherState) -> GopherState:
            new_timer = s.duck_animation_timer - 1
            s = s._replace(
                duck_animation_timer=new_timer
            )

            s = jif_true(
                new_timer == self.consts.DUCK_ANIMATION_DOWN_WING,
                lambda s: s._replace(
                    duck_render_state=byte_val(DuckRenderState.DOWN.value)
                ),
                operand=s
            )
            s = jif_true(
                new_timer == self.consts.DUCK_ANIMATION_STATIONARY_WING,
                lambda s: s._replace(
                    duck_render_state=byte_val(DuckRenderState.STATIONARY.value)
                ),
                operand=s
            )
            s = jif_true(
                new_timer == self.consts.DUCK_ANIMATION_UP_WING,
                lambda s: s._replace(
                    duck_render_state=byte_val(DuckRenderState.UP.value)
                ),
                operand=s
            )
            return s

        def update(s: GopherState) -> GopherState:
            new_animation_timer = s.duck_animation_timer - 1

            s = jif(
                new_animation_timer == 0,
                lambda s: s._replace(
                    duck_animation_timer=byte_val(self.consts.DUCK_INIT_ANIMATION_TIMER),
                    duck_render_state=byte_val(DuckRenderState.STATIONARY.value)
                ),
                update_animation,
                operand=s
            )

            # Quack
            s = jif_true(
                jnp.bitwise_and(s.frame_count, 0x1F) == 0,
                lambda s: self.set_game_audio_values(s,
                                                     self.consts.MEM_AUDIO_DUCK_QUACKING - self.consts.MEM_AUDIO_VALUES),
                operand=s
            )

            s = jif(
                s.duck_moving_left,
                lambda s: s._replace(
                    duck_x=s.duck_x - 1
                ),
                lambda s: s._replace(
                    duck_x=s.duck_x + 1
                ),
                operand=s
            )

            return jif_true(
                jor(s.duck_x < self.consts.DUCK_MIN_X, s.duck_x >= self.consts.DUCK_MAX_X),
                lambda s: self.disable_duck(s),
                operand=s
            )

        return jif(
            s.duck_animation_timer == 0,
            lambda s: self.disable_duck(s),
            update,
            operand=s
        )

    @partial(jit, static_argnums=(0,))
    def disable_duck(self, s: GopherState) -> GopherState:
        return s._replace(
            duck_render_state=byte_val(DuckRenderState.DISABLED.value),
            duck_animation_timer=byte_val(0)
        )

    @partial(jit, static_argnums=(0,))
    def update_seed(self, s: GopherState) -> GopherState:
        def update_moving_seed(s: GopherState) -> GopherState:
            return jif(
                s.duck_moving_left,
                lambda s: s._replace(
                    seed_x=s.seed_x - 1
                ),
                lambda s: s._replace(
                    seed_x=s.seed_x + 1
                ),
                operand=s
            )

        return jif_true(
            s.seed_visible,
            lambda s: jif(
                s.held_seed_decaying_timer != 0,
                lambda s: self.update_seed_held_by_farmer(s),  # Seed is held by farmer
                lambda s: jif(
                    s.duck_seed_drop_target == s.seed_x,
                    lambda s: self.update_falling_seed(s),  # Seed reached target position
                    update_moving_seed,  # Seed is still moving with duck
                    operand=s
                ),
                operand=s
            ),
            operand=s
        )

    @partial(jit, static_argnums=(0,))
    def update_falling_seed(self, s: GopherState) -> GopherState:
        # Seed falls down
        new_seed_y = s.seed_y + 1
        s = s._replace(seed_y=new_seed_y)

        # Check if seed reached ground
        s = jif_true(
            new_seed_y == self.consts.SEED_GROUND_LEVEL,
            lambda s: s._replace(
                seed_y=byte_val(0),
                seed_visible=bool_val(False)
            ),
            operand=s
        )

        # Check if farmer caught the seed
        return jif_true(
            jand(jand(
                new_seed_y >= self.consts.SEED_MIN_CATCHING_Y,
                new_seed_y < self.consts.SEED_MAX_CATCHING_Y
            ), jnp.abs(s.seed_x.astype(jnp.int16) - s.farmer_x) < 5),
            lambda s: s._replace(
                held_seed_decaying_timer=byte_val(self.consts.SEED_INIT_DECAYING_TIMER)
            ),
            operand=s
        )

    @partial(jit, static_argnums=(0,))
    def update_seed_held_by_farmer(self, s: GopherState) -> GopherState:
        # Move seed with farmer
        new_decaying_timer = s.held_seed_decaying_timer - 1

        s = s._replace(
            seed_x=s.farmer_x,
            held_seed_decaying_timer=new_decaying_timer
        )

        # Check if seed decayed
        return jif_true(
            new_decaying_timer == 0,
            lambda s: s._replace(
                seed_y=byte_val(0),
                seed_visible=bool_val(False)
            ),
            operand=s
        )

    @partial(jit, static_argnums=(0,))
    def update_gopher_digging(self, s: GopherState) -> GopherState:
        s = jif_true(
            jand(
                s.score_value >= 10_000,
                jor(s.gopher_y_target_id != self.consts.GOPHER_Y_TARGET_UNDER_GROUND, s.gopher_y_target_extra_flag)
            ),
            lambda s: s._replace(
                gopher_y_locked=True,
                gopher_y_target_id=jnp.bitwise_or(s.gopher_y_target_id, self.consts.GOPHER_Y_DIFFICULT_TARGET_MASK)
            ),
            operand=s
        )

        return jif(
            s.gopher_y == self.consts.GOPHER_Y_UNDERGROUND,
            lambda s: self.gopher_digging(s, s.gopher_x + jselect(s.gopher_facing_left, byte_val(0), byte_val(8))),
            lambda s: jif_true(
                s.gopher_y != self.consts.GOPHER_Y_ABOVE_GROUND,
                lambda s: self.gopher_digging(s, self.consts.TUNNEL_X_POSITIONS[s.gopher_tunnel_target_id]),
                operand=s
            ),
            operand=s
        )

    @partial(jit, static_argnums=(0,))
    def gopher_digging(self, s: GopherState, x_pos: ByteValue) -> GopherState:
        def dirt_not_removed(s: GopherState, x_pos: ByteValue) -> GopherState:
            dirt_idx = self.get_garden_flag_index_from_pos(x_pos, s.gopher_y)

            s = self.set_game_audio_values(s._replace(
                garden_dirt_flags=s.garden_dirt_flags.at[dirt_idx].set(True)
            ), self.consts.MEM_AUDIO_DIG_TUNNEL - self.consts.MEM_AUDIO_VALUES)

            s = jif_p2_true(
                s.gopher_y != self.consts.GOPHER_Y_UNDERGROUND,
                lambda s, x_pos: s._replace(
                    # Up or down movement, removing adjacent dirt as well
                    garden_dirt_flags=s.garden_dirt_flags.at[
                        self.get_garden_flag_index_from_pos(x_pos + 1, s.gopher_y)].set(
                        True)
                ),
                operand=(s, x_pos)
            )

            return self.check_to_change_gopher_horizontal_direction(s)

        dirt_idx = self.get_garden_flag_index_from_pos(x_pos, s.gopher_y)

        return jif_p2_true(
            jnot(s.garden_dirt_flags[dirt_idx]),
            dirt_not_removed,
            operand=(s, x_pos)
        )

    @partial(jit, static_argnums=(0,))
    def check_to_change_gopher_horizontal_direction(self, s: GopherState) -> GopherState:
        def check(s: GopherState) -> GopherState:
            s = s._replace(
                gopher_change_direction_timer=s.gopher_change_direction_timer - 1
            )

            return jif(
                s.gopher_change_direction_timer == 0,
                lambda s: s._replace(
                    gopher_y_locked=bool_val(True),
                    gopher_change_direction_timer=s.gopher_change_direction_delay
                ),
                lambda s: jif(
                    s.gopher_y == self.consts.GOPHER_Y_UNDERGROUND,
                    lambda s: s._replace(
                        gopher_target_move_left=jnot(s.gopher_target_move_left)
                    ),
                    lambda s: s._replace(
                        gopher_y_target_extra_flag=bool_val(False),
                        gopher_y_locked=bool_val(True),
                        gopher_y_target_id=byte_val(self.consts.GOPHER_Y_TARGET_UNDER_GROUND),
                    ),
                    operand=s
                ),
                operand=s
            )

        return jif_true(
            jnot(s.gopher_y_locked),
            check,
            operand=s
        )

    @partial(jit, static_argnums=(0,))
    def update_farmer(self, s: GopherState) -> GopherState:
        def check_for_action(s: GopherState) -> GopherState:
            def check_already_pressed(s: GopherState):
                return jif_true(
                    jnot(s.action_button_prev_pressed),
                    lambda s: self.increment_farmer_animation_index(
                        s._replace(action_button_prev_pressed=bool_val(True))),
                    operand=s
                )

            return jif(
                s.fire_pressed,
                check_already_pressed,
                lambda s: s._replace(action_button_prev_pressed=bool_val(False)),
                operand=s
            )

        return jif(
            s.farmer_animation_idx != 0,
            lambda s: self.increment_farmer_animation_index(s),
            check_for_action,
            operand=s
        )

    @partial(jit, static_argnums=(0,))
    def increment_farmer_animation_index(self, s: GopherState):
        s = s._replace(farmer_animation_idx=s.farmer_animation_idx + 1)

        return jif(
            s.farmer_animation_idx == 8,
            lambda s: self.farmer_action(
                s._replace(
                    farmer_animation_idx=byte_val(0),
                    farmer_render_state=byte_val(FarmerRenderingState.STANDING.value)
                )
            ),
            lambda s: jif(
                s.farmer_animation_idx == 2,
                lambda s: s._replace(
                    farmer_render_state=byte_val(FarmerRenderingState.DIGGING_1.value)
                ),
                lambda s: jif_true(
                    s.farmer_animation_idx == 4,
                    lambda s: s._replace(
                        farmer_render_state=byte_val(FarmerRenderingState.DIGGING_2.value)
                    ),
                    operand=s
                ),
                operand=s
            ),
            operand=s
        )

    # Carrot planting or hole filling
    @partial(jit, static_argnums=(0,))
    def farmer_action(self, s: GopherState) -> GopherState:
        carrot_id = get_first_index(jnp.abs((s.farmer_x.astype(jnp.int16) - 4) - self.consts.CARROT_X_POSITIONS) < 6)

        def potentially_fill_hole(s: GopherState):
            tunnel_id = get_first_index(
                jnp.abs((s.farmer_x.astype(jnp.int16) - 4) - self.consts.TUNNEL_X_POSITIONS) < 6)
            return jif(
                tunnel_id == -1,
                lambda params: params[0],
                lambda params: self.fill_tunnel(params[0], params[1]),
                operand=(s, tunnel_id)
            )

        def potentially_plant_carrot(s: GopherState, carrot_id: IntValue):
            def plant_carrot(s: GopherState, carrot_id: IntValue):
                return s._replace(
                    carrots_left=s.carrots_left.at[carrot_id].set(bool_val(True)),
                    seed_y=byte_val(0),
                    seed_visible=bool_val(False),
                    held_seed_decaying_timer=byte_val(0)
                )

            return jif_p2_true(
                s.held_seed_decaying_timer != 0,
                plant_carrot,
                operand=(s, carrot_id)
            )

        return jif_p2(
            carrot_id == -1,
            lambda s, _: potentially_fill_hole(s),
            potentially_plant_carrot,
            operand=(s, carrot_id)
        )

    @partial(jit, static_argnums=(0,))
    def fill_tunnel(self, s: GopherState, hole_id: IntValue):
        gopher_target_x = jif(
            s.gopher_targeting_carrot,
            lambda s: self.consts.CARROT_X_POSITIONS[s.gopher_carrot_target_id],
            lambda s: self.consts.TUNNEL_X_POSITIONS[s.gopher_tunnel_target_id],
            operand=s
        )
        filling_possible = jnot(jand(gopher_target_x == self.consts.TUNNEL_X_POSITIONS[hole_id],
                                     s.gopher_y != self.consts.GOPHER_Y_UNDERGROUND))

        def fill_hole(params: tuple[GopherState, IntValue, IntValue]):
            s, hole_x_idx, target_fill_y_pos = params

            f_idx = target_fill_y_pos * 40 + hole_x_idx
            indices = jif(
                target_fill_y_pos == 3,
                lambda: int_arr([f_idx - 1, f_idx, f_idx + 1, f_idx + 2]),
                lambda: int_arr([f_idx, f_idx + 1, f_idx, f_idx + 1]),  # Repeat value, only two are relevant
            )
            new_garden_dirt_flags = s.garden_dirt_flags.at[indices].set(False)

            return self.increment_score(
                self.set_game_audio_values(
                    s._replace(
                        garden_dirt_flags=new_garden_dirt_flags
                    ),
                    self.consts.MEM_AUDIO_FILL_TUNNEL - self.consts.MEM_AUDIO_VALUES
                ),
                self.consts.POINTS_FILL_TUNNEL
            )

        def potentially_fill_hole(s: GopherState, hole_id: IntValue):
            hole_x_position = self.consts.TUNNEL_X_POSITIONS[hole_id]
            hole_x_idx = self.get_garden_flag_x_index(hole_x_position)

            first_dirt_y_pos = self.get_first_dirt_index_for_hole(s, hole_x_position)
            target_fill_y_pos = first_dirt_y_pos - 1

            return jif(
                target_fill_y_pos >= 0,
                fill_hole,
                lambda params: params[0],
                operand=(s, hole_x_idx, target_fill_y_pos)
            )

        return jif_p2_true(
            filling_possible,
            potentially_fill_hole,
            operand=(s, hole_id)
        )

    @partial(jit, static_argnums=(0, 2))
    def increment_score(self, s: GopherState, amount: int) -> GopherState:
        prev_100_digit = jnp.floor_divide(s.score_value, int_val(100))
        next_100_digit = jnp.floor_divide(s.score_value + amount, int_val(100))

        s = jif_true(
            next_100_digit != prev_100_digit,
            lambda s: self.check_to_spawn_duck(self.check_to_decrement_gopher_direction_timer(s)),
            operand=s
        )

        # Score wraps around after 1 million
        return s._replace(
            score_value=(s.score_value + amount) % 1_000_000
        )

    @partial(jit, static_argnums=(0,))
    def check_to_decrement_gopher_direction_timer(self, s: GopherState) -> GopherState:
        # Check if 000X00 digit in score is even (0) or odd (1)
        return jif_true(
            jand((s.score_value // 100) % 2 == 0, s.gopher_change_direction_delay > 1),
            # Decrement gopher change direction timer (making game harder)
            lambda s: s._replace(
                gopher_change_direction_delay=s.gopher_change_direction_delay - 1
            ),
            operand=s
        )

    @partial(jit, static_argnums=(0,))
    def check_to_spawn_duck(self, s: GopherState) -> GopherState:
        def spawn_duck(s: GopherState) -> GopherState:
            new_attributes_byte = s.random_0
            new_moving_left = is_msb_set(new_attributes_byte)
            new_seed_target_pos = jnp.bitwise_and(new_attributes_byte, self.consts.SEED_TARGET_X_MASK)

            # Spawn at left or right edge of screen
            duck_spawn_x = jselect(new_moving_left, byte_val(self.consts.DUCK_MAX_X), byte_val(self.consts.DUCK_MIN_X))
            # Spawn seed with specific offset to duck
            seed_spawn_x = jselect(new_moving_left, byte_val(self.consts.SCREEN_MAX_X - 19),
                                   byte_val(self.consts.DUCK_MIN_X + 8))

            new_seed_target_pos = jselect(new_seed_target_pos < 20, byte_val((self.consts.SCREEN_MAX_X + 1) // 2),
                                          new_seed_target_pos)

            s = s._replace(
                duck_moving_left=new_moving_left,
                duck_seed_drop_target=new_seed_target_pos,
                duck_x=duck_spawn_x,
                seed_x=seed_spawn_x
            )

            return self.init_duck_state(s)

        score_100_digit = (s.score_value // 100) % 10

        # Duck enabled
        # and not full carrots
        # and seed (with / without duck) not already visible
        return jif_true(
            jand(
                jand(
                    jand(s.is_duck_mode_enabled, jnp.any(s.carrots_left == False)),
                    jnot(s.seed_visible)
                ),
                jor(score_100_digit == 4, score_100_digit == 9)
            ),
            spawn_duck,
            operand=s
        )

    @partial(jit, static_argnums=(0,))
    def init_duck_state(self, s: GopherState) -> GopherState:
        return s._replace(
            duck_animation_timer=byte_val(self.consts.DUCK_INIT_ANIMATION_TIMER),
            duck_render_state=byte_val(DuckRenderState.STATIONARY.value),
            seed_y=byte_val(self.consts.SEED_INIT_Y),
            seed_visible=bool_val(True),
            held_seed_decaying_timer=byte_val(0)
        )

    @partial(jit, static_argnums=(0, 2))
    def set_game_audio_values(self, s: GopherState, audio_offset: int) -> GopherState:
        next_audio_channel_index = (s.audio_channel_index + 1) % 2
        return s._replace(
            audio_channel_index=next_audio_channel_index,
            audio_index_values=s.audio_index_values.at[next_audio_channel_index & 1].set(audio_offset + 1)
        )

    @partial(jit, static_argnums=(0,))
    def play_game_audio(self, s: GopherState, carry) -> tuple[GopherState, ByteValue]:
        def play_audio(s: GopherState, audio_channel_idx: ByteValue, audio_value: ByteValue) -> tuple[
            GopherState, ByteValue]:
            a = jnp.bitwise_and(audio_value, self.consts.AUDIO_DURATION_MASK)
            shift_amount = jselect(jnot(is_msb_set(a)), byte_val(4), byte_val(3))
            a = jnp.right_shift(a, shift_amount - 1)
            a, carry = shift_right_with_carry(a)

            return (s._replace(
                audio_index_values=s.audio_index_values.at[audio_channel_idx].set(
                    s.audio_index_values[audio_channel_idx] + 1
                ),
                audio_duration_values=s.audio_duration_values.at[audio_channel_idx].set(a)
            ), carry)

        def play_next_audio_value(s: GopherState, audio_channel_idx: ByteValue, carry: ByteValue) -> tuple[
            GopherState, ByteValue]:
            audio_value = self.consts.AUDIO_VALUES[s.audio_index_values[audio_channel_idx]]

            return jif(
                audio_value == self.consts.END_AUDIO_TUNE,
                lambda params: (params[0], params[2]),
                lambda params: play_audio(params[0], params[1], params[3]),
                operand=(s, audio_channel_idx, carry, audio_value)
            )

        def play_audio_channel(rev_audio_channel_idx: ByteValue, params: tuple[GopherState, ByteValue]) -> tuple[
            GopherState, ByteValue]:
            s, carry = params
            # 0 -> 1 and 1 -> 0
            audio_channel_idx = 1 - rev_audio_channel_idx

            return jif(
                s.audio_duration_values[audio_channel_idx] == 0,
                lambda params: play_next_audio_value(params[0], params[1], params[2]),
                lambda params: (params[0]._replace(
                    audio_duration_values=params[0].audio_duration_values.at[params[1]].set(
                        params[0].audio_duration_values[params[1]] - 1
                    )
                ), params[2]),
                operand=(s, audio_channel_idx, carry)
            )

        return jax.lax.fori_loop(
            0, 2, play_audio_channel, (s, carry)
        )

    @partial(jit, static_argnums=(0,))
    def next_random(self, s: GopherState, carry: ByteValue) -> GopherState:
        r1_0 = s.random_1
        r0_0 = s.random_0

        r0_1, carry = roll_left_with_carry(r0_0, carry)
        r1_1, carry = roll_left_with_carry(r1_0, carry)

        r0_2, _ = adc_with_carry(r0_1, byte_val(195), carry)

        return s._replace(
            random_0=jnp.bitwise_xor(r0_0, r0_2),
            random_1=jnp.bitwise_xor(r1_0, r1_1)
        )

    @partial(jit, static_argnums=(0,))
    def update_game_state(self, s: GopherState) -> GopherState:
        gs = s.game_state

        s = jif_true(gs == self.consts.GS_MAIN_GAME_LOOP, lambda s: self.update_main_game_loop(s), operand=s)
        s = jif_true(gs == self.consts.GS_GOPHER_STOLE_CARROT, lambda s: self.carrot_stolen_by_gopher(s), operand=s)
        s = jif_true(gs == self.consts.GS_DUCK_WAIT, lambda s: self.wait_for_duck_to_advance_game_state(s), operand=s)
        s = jif_true(jor(gs == self.consts.GS_RESET_AFTER_ROUND, gs == self.consts.GS_INIT_NEXT_ROUND),
                     lambda s: self.advance_current_game_state(self.init_game_round_data(s), byte_val(0)), operand=s)
        s = jif_true(gs == self.consts.GS_CHECK_FOR_GAME_OVER, lambda s: self.check_for_game_over_state(s), operand=s)
        s = jif_true(gs == self.consts.GS_FINISHED_NEXT_ROUND_SETUP,
                     lambda s: self.advance_current_game_state(s, byte_val(0)),
                     operand=s)
        s = jif_true(gs == self.consts.GS_PAUSE_FOR_ACTION_BUTTON,
                     lambda s: self.wait_for_action_button_to_continue_round(s),
                     operand=s)
        s = jif_true(gs == self.consts.GS_WAIT_FOR_NEW_GAME, lambda s: self.end_of_frame(s, byte_val(0)), operand=s)

        return s

    @partial(jit, static_argnums=(0,))
    def update_main_game_loop(self, s: GopherState) -> GopherState:
        s = self.update_farmer_movement(s)
        s, can_be_bonked = self.update_gopher_movement(s)

        return jif_true(
            can_be_bonked,
            lambda s: self.update_gopher_animation(
                self.update_gopher_taunt_logic(
                    self.check_for_farmer_bonking_gopher(
                        s
                    )
                )
            ),
            operand=s
        )

    @partial(jit, static_argnums=(0,))
    def update_gopher_movement(self, s: GopherState) -> tuple[GopherState, BoolValue]:
        def move_gopher(s: GopherState) -> GopherState:
            return jif(
                s.gopher_target_move_left,
                lambda s: s._replace(
                    gopher_x=jif(
                        s.gopher_x <= self.consts.GOPHER_MIN_X + 1,
                        lambda _: byte_val(self.consts.GOPHER_MAX_X),
                        lambda s: s.gopher_x - 2,
                        operand=s
                    ),
                    gopher_facing_left=bool_val(True)
                ),
                lambda s: s._replace(
                    gopher_x=jif(
                        s.gopher_x >= self.consts.GOPHER_MAX_X - 2,
                        lambda _: byte_val(self.consts.GOPHER_MIN_X),
                        lambda s: s.gopher_x + 2,
                        operand=s
                    ),
                    gopher_facing_left=bool_val(False)
                ),
                operand=s
            )

        def not_taunting(s: GopherState) -> tuple[GopherState, BoolValue]:
            x_target = jif(
                s.gopher_targeting_carrot,
                lambda s: self.consts.CARROT_X_POSITIONS[s.gopher_carrot_target_id],
                lambda s: self.consts.TUNNEL_X_POSITIONS[s.gopher_tunnel_target_id],
                operand=s
            )
            return jif(
                jnp.abs(s.gopher_x.astype(jnp.int16) - x_target) < 3,
                lambda params: self.gopher_steal_carrot_or_move_vertically(params[0], params[1]),
                lambda params: (move_gopher(params[0]), True),
                operand=(s, x_target)
            )

        return jif(
            s.gopher_taunt_timer == 0,
            not_taunting,
            lambda s: (s, bool_val(True)),
            operand=s
        )

    @partial(jit, static_argnums=(0,))
    def gopher_steal_carrot_or_move_vertically(self, s: GopherState, x_target: ByteValue) -> tuple[
        GopherState, BoolValue]:
        def steal_carrot(s: GopherState) -> GopherState:
            target_carrot = s.gopher_carrot_target_id

            return self.advance_current_game_state(
                s._replace(
                    carrots_left=s.carrots_left.at[target_carrot].set(False)
                ), byte_val(0)
            )

        return jif(
            s.gopher_y == self.consts.GOPHER_Y_ABOVE_GROUND,
            lambda params: (steal_carrot(params[0]), bool_val(False)),
            lambda params: (self.move_gopher_vertically(params[0], params[1]), bool_val(True)),
            operand=(s, x_target)
        )

    @partial(jit, static_argnums=(0,))
    def move_gopher_vertically(self, s: GopherState, x_target) -> GopherState:
        x_target = jif_true(
            jnot(s.gopher_facing_left),
            lambda x_target: x_target + 1,
            operand=x_target
        )

        # "Stick" gopher to target position
        s = s._replace(
            gopher_x=x_target
        )

        # Check if reached vertical target
        y_target = self.consts.GOPHER_Y_TARGET_POSITIONS[s.gopher_y_target_id]

        return jif_p2(
            s.gopher_y == y_target,
            lambda s, _: self.gopher_reached_vertical_target(s),
            lambda s, y_target: jif(
                s.gopher_y < y_target,
                lambda s: jif(
                    s.gopher_y + 1 == self.consts.GOPHER_Y_ABOVE_GROUND,
                    lambda s: self.set_gopher_carrot_target(
                        s._replace(
                            gopher_y=s.gopher_y + 1
                        )
                    ),
                    lambda s: s._replace(
                        gopher_y=s.gopher_y + 1
                    ),
                    operand=s
                ),
                lambda s: s._replace(
                    gopher_y=s.gopher_y - 1
                ),
                operand=s
            ),
            operand=(s, y_target)
        )

    @partial(jit, static_argnums=(0,))
    def update_gopher_animation(self, s: GopherState) -> GopherState:
        y_pos = s.gopher_y

        s = jif_true(
            y_pos >= self.consts.GOPHER_Y_UNDERGROUND + 7,
            lambda s: s._replace(
                gopher_render_state=byte_val(GopherRenderState.RISING.value)
            ),
            operand=s
        )

        s = jif_true(
            y_pos == self.consts.GOPHER_Y_ABOVE_GROUND,
            lambda s: s._replace(
                gopher_render_state=byte_val(GopherRenderState.ABOVE_GROUND_RUNNING_1.value)
            ),
            operand=s
        )

        s = jif_true(
            y_pos < self.consts.GOPHER_Y_UNDERGROUND + 7,
            lambda s: s._replace(
                gopher_render_state=byte_val(GopherRenderState.UNDERGROUND_RUNNING_1.value)
            ),
            operand=s
        )

        s = jif_true(s.gopher_taunt_timer != 0, self.animate_taunting_gopher, operand=s)

        s, carry = self.animate_crawling_gopher(s)
        return self.end_of_frame(s, carry)

    @partial(jit, static_argnums=(0,))
    def animate_taunting_gopher(self, s: GopherState) -> GopherState:
        return s._replace(
            gopher_render_state=jselect(
                jor(s.gopher_taunt_timer < 7, jand(s.gopher_taunt_timer >= 14, s.gopher_taunt_timer < 21)),
                byte_val(GopherRenderState.TAUNTING_2.value),
                byte_val(GopherRenderState.TAUNTING_1.value))
        )

    @partial(jit, static_argnums=(0,))
    def animate_crawling_gopher(self, s: GopherState) -> tuple[GopherState, ByteValue]:
        def animate(s: GopherState) -> tuple[GopherState, ByteValue]:
            s = jif_true(
                jnp.bitwise_and(s.frame_count, 3) == 0,
                lambda s: s._replace(
                    gopher_animation_rate=flip_byte(s.gopher_animation_rate)
                ),
                operand=s
            )

            return jif(
                s.gopher_animation_rate == 0,
                lambda s: (s, jselect(s.gopher_y != self.consts.GOPHER_Y_UNDERGROUND, byte_val(1), byte_val(0))),
                lambda s: (
                    s._replace(
                        gopher_render_state=jselect(s.gopher_y != self.consts.GOPHER_Y_UNDERGROUND,
                                                    byte_val(GopherRenderState.ABOVE_GROUND_RUNNING_2.value),
                                                    byte_val(GopherRenderState.UNDERGROUND_RUNNING_2.value))
                    ), byte_val(1)
                ),
                operand=s
            )

        # No crawling
        return jif(
            jand(s.gopher_y != self.consts.GOPHER_Y_UNDERGROUND, s.gopher_y != self.consts.GOPHER_Y_ABOVE_GROUND),
            lambda s: (s, byte_val(0)),
            animate,
            operand=s
        )

    @partial(jit, static_argnums=(0,))
    def set_gopher_carrot_target(self, s: GopherState) -> GopherState:
        def target_from_left(s: GopherState) -> GopherState:
            target_id = jif(
                s.carrots_left[2],
                lambda _: byte_val(2),
                lambda s: jselect(
                    s.carrots_left[1],
                    byte_val(1),
                    byte_val(0)
                ),
                operand=s
            )

            return s._replace(
                gopher_target_move_left=bool_val(False),
                gopher_tunnel_target_id=byte_val(0),
                gopher_targeting_carrot=bool_val(True),
                gopher_carrot_target_id=target_id
            )

        def target_from_right(s: GopherState) -> GopherState:
            target_id = jif(
                s.carrots_left[0],
                lambda _: byte_val(0),
                lambda s: jselect(
                    s.carrots_left[1],
                    byte_val(1),
                    byte_val(2)
                ),
                operand=s
            )

            return s._replace(
                gopher_target_move_left=bool_val(True),
                gopher_tunnel_target_id=byte_val(0),
                gopher_targeting_carrot=bool_val(True),
                gopher_carrot_target_id=target_id
            )

        return jif(
            s.gopher_x <= self.consts.SCREEN_MAX_X // 2,
            target_from_left,
            target_from_right,
            operand=s
        )

    @partial(jit, static_argnums=(0,))
    def gopher_reached_vertical_target(self, s: GopherState) -> GopherState:
        return jif(
            s.gopher_y_target_id == self.consts.GOPHER_Y_TARGET_UNDER_GROUND,
            lambda s: self.set_gopher_new_target_values(s),
            lambda s: jif(
                s.gopher_y == self.consts.GOPHER_Y_TAUNTING,
                lambda s: s._replace(
                    gopher_y_target_id=byte_val(self.consts.GOPHER_Y_TARGET_UNDER_GROUND),
                    gopher_y_locked=bool_val(True),
                    gopher_y_target_extra_flag=bool_val(False),
                    gopher_y=s.gopher_y - 1
                ),
                lambda s: s._replace(
                    gopher_y_target_id=byte_val(self.consts.GOPHER_Y_TARGET_UNDER_GROUND),
                    gopher_y_locked=bool_val(True),
                    gopher_y_target_extra_flag=bool_val(False)
                ),
                operand=s
            ),
            operand=s
        )

    @partial(jit, static_argnums=(0,))
    def set_gopher_new_target_values(self, s: GopherState) -> GopherState:
        s = self.next_random(s, carry=byte_val(1))
        s = s._replace(
            gopher_y_locked=is_msb_set(s.random_0),
            gopher_y_target_id=jnp.bitwise_and(s.random_0, self.consts.GOPHER_Y_TARGET_MASK),
            gopher_y_target_extra_flag=jnp.bitwise_and(s.random_0,
                                                       (
                                                               ~self.consts.GOPHER_Y_TARGET_MASK & 0xFF & ~self.consts.GOPHER_Y_LOCKED_BIT)) != 0,
            # Randomly choose facing direction and tunnel to target
            gopher_target_move_left=jnp.bitwise_and(s.random_1, self.consts.GOPHER_X_DIR_MASK) != 0,
            gopher_tunnel_target_id=jnp.bitwise_and(s.random_1, self.consts.GOPHER_TUNNEL_TARGET_MASK),
            gopher_targeting_carrot=bool_val(False),
            gopher_carrot_target_id=byte_val(0)
        )

        s = jif_true(
            jor(s.score_value >= 10_000, s.expert_difficulty),
            lambda s: self.smart_gopher_tunnel_targeting(s),
            operand=s
        )

        return self.normal_gopher_logic(s)

    @partial(jit, static_argnums=(0,))
    def smart_gopher_tunnel_targeting(self, s: GopherState) -> GopherState:
        s = jif_true(
            jand(s.farmer_x >= 80, s.gopher_tunnel_target_id >= 4),
            lambda s: s._replace(gopher_tunnel_target_id=s.gopher_tunnel_target_id - 4),
            operand=s
        )

        return jif_true(
            jand(
                jor(s.farmer_x < 80, jand(jnot(s.gopher_target_move_left), s.gopher_tunnel_target_id == 0)),
                s.gopher_tunnel_target_id < 4
            ),
            lambda s: s._replace(gopher_tunnel_target_id=s.gopher_tunnel_target_id + 4),
            operand=s
        )

    @partial(jit, static_argnums=(0,))
    def normal_gopher_logic(self, s: GopherState) -> GopherState:
        s = jif_true(
            s.gopher_change_direction_timer != 0,
            lambda s: s._replace(
                gopher_change_direction_timer=s.gopher_change_direction_timer - 1,
                gopher_y_locked=bool_val(False)
            ),
            operand=s
        )

        s = jif_true(
            s.gopher_change_direction_timer == 0,
            lambda s: s._replace(
                gopher_change_direction_timer=s.gopher_change_direction_delay,
                gopher_y_locked=bool_val(True)
            ),
            operand=s
        )

        return s

    @partial(jit, static_argnums=(0,))
    def update_farmer_movement(self, s: GopherState) -> GopherState:
        return jif_true(
            jor(s.left_pressed, s.right_pressed),
            lambda s: jif(
                s.right_pressed,
                lambda s: jif_true(
                    s.farmer_x < self.consts.FARMER_MAX_X,
                    lambda s: s._replace(farmer_x=s.farmer_x + 1),
                    operand=s
                ),
                lambda s: jif_true(
                    jand(s.left_pressed, s.farmer_x >= self.consts.FARMER_MIN_X),
                    lambda s: s._replace(farmer_x=s.farmer_x - 1),
                    operand=s
                ),
                operand=s
            ),
            operand=s
        )

    @partial(jit, static_argnums=(0,))
    def check_for_farmer_bonking_gopher(self, s: GopherState) -> GopherState:
        # Farmer currently in last half of animation (>= 4)
        # and Gopher is higher than or equal to the taunting position
        # and Farmer is close enough
        return jif_true(
            jand(
                s.farmer_animation_idx >= 4,
                jand(
                    s.gopher_y >= self.consts.GOPHER_Y_TAUNTING,
                    jnp.abs(s.farmer_x.astype(jnp.int16) - (s.gopher_x + 3)) < 6
                )
            ),
            lambda s: self.set_game_audio_values(self.increment_score(s, self.consts.POINTS_BONK_GOPHER),
                                                 self.consts.MEM_AUDIO_BONK_GOPHER - self.consts.MEM_AUDIO_VALUES)._replace(
                # Reset gopher
                gopher_x=byte_val(self.consts.GOPHER_INIT_X - 4),
                # Set random new horizontal direction, and new tunnel (does not target carrot)
                gopher_target_move_left=jnp.bitwise_and(s.random_0, self.consts.GOPHER_X_DIR_MASK) != 0,
                gopher_tunnel_target_id=jnp.bitwise_and(s.random_0, self.consts.GOPHER_TUNNEL_TARGET_MASK),
                gopher_targeting_carrot=bool_val(False),
                gopher_carrot_target_id=byte_val(0),

                gopher_y_target_id=byte_val(self.consts.GOPHER_Y_TARGET_UNDER_GROUND),
                gopher_y_locked=bool_val(False),
                gopher_y_target_extra_flag=bool_val(False),
                gopher_y=byte_val(0),
                gopher_taunt_timer=byte_val(0),
            ),
            operand=s
        )

    @partial(jit, static_argnums=(0,))
    def update_gopher_taunt_logic(self, s: GopherState) -> GopherState:
        s = jif_true(
            s.gopher_y == self.consts.GOPHER_Y_TAUNTING,
            lambda s: jif(
                s.gopher_taunt_timer != 0,
                lambda s: s._replace(gopher_taunt_timer=s.gopher_taunt_timer - 1),
                lambda s: self.set_game_audio_values(s,
                                                     self.consts.MEM_AUDIO_GOPHER_TAUNT - self.consts.MEM_AUDIO_VALUES)._replace(
                    gopher_taunt_timer=byte_val(28)
                ),
                operand=s
            ),
            operand=s
        )

        return jif_true(
            s.gopher_taunt_timer != 0,
            self.set_taunting_gopher_facing_direction,
            operand=s
        )

    @partial(jit, static_argnums=(0,))
    def set_taunting_gopher_facing_direction(self, s: GopherState) -> GopherState:
        return jif(
            s.farmer_x <= s.gopher_x,
            lambda s: jif_true(
                jnot(s.gopher_target_move_left),
                lambda s: s._replace(
                    gopher_target_move_left=bool_val(True),
                    gopher_facing_left=bool_val(True),
                    gopher_x=s.gopher_x - 1
                ),
                operand=s
            ),
            lambda s: jif_true(
                s.gopher_target_move_left,
                lambda s: s._replace(
                    gopher_target_move_left=bool_val(False),
                    gopher_facing_left=bool_val(False),
                    gopher_x=s.gopher_x + 1
                ),
                operand=s
            ),
            operand=s
        )

    @partial(jit, static_argnums=(0,))
    def carrot_stolen_by_gopher(self, s: GopherState) -> GopherState:
        return self.advance_current_game_state(
            self.set_game_audio_values(s._replace(
                gopher_render_state=byte_val(GopherRenderState.DISABLED.value),
                frame_count=byte_val(self.consts.WAIT_TIME_CARROT_STOLEN)
            ), self.consts.MEM_AUDIO_STOLEN_CARROT - self.consts.MEM_AUDIO_VALUES),
            carry=byte_val(0)
        )

    @partial(jit, static_argnums=(0,))
    def wait_for_duck_to_advance_game_state(self, s: GopherState) -> GopherState:
        return jif(
            s.duck_animation_timer != 0,
            lambda s: self.end_of_frame(s, carry=byte_val(0)),
            self.advance_game_state_after_frame_count_expire,
            operand=s
        )

    @partial(jit, static_argnums=(0,))
    def advance_game_state_after_frame_count_expire(self, s: GopherState) -> GopherState:
        return jif(
            s.frame_count != 255,
            lambda s: self.end_of_frame(s, carry=byte_val(0)),
            lambda s: self.advance_current_game_state(s, carry=byte_val(1)),
            operand=s
        )

    @partial(jit, static_argnums=(0,))
    def check_for_game_over_state(self, s: GopherState) -> GopherState:
        def game_over(s: GopherState):
            s = self.set_game_audio_values(s, self.consts.MEM_AUDIO_GAME_OVER_THEME_1 - self.consts.MEM_AUDIO_VALUES)
            s = self.set_game_audio_values(s, self.consts.MEM_AUDIO_GAME_OVER_THEME_2 - self.consts.MEM_AUDIO_VALUES)

            s = s._replace(game_state=byte_val(self.consts.GS_WAIT_FOR_NEW_GAME))

            return self.end_of_frame(s, carry=byte_val(0))

        return jif(
            jnp.any(s.carrots_left),
            lambda s: self.advance_current_game_state(s, carry=byte_val(0)),
            game_over,
            operand=s
        )

    @partial(jit, static_argnums=(0,))
    def wait_for_action_button_to_continue_round(self, s: GopherState) -> GopherState:
        return jif(
            s.fire_pressed,
            lambda s: s._replace(
                game_state=byte_val(self.consts.GS_MAIN_GAME_LOOP)
            ),
            lambda s: self.end_of_frame(s, carry=byte_val(0)),
            operand=s
        )

    @partial(jit, static_argnums=(0,))
    def advance_current_game_state(self, s: GopherState, carry: ByteValue) -> GopherState:
        return self.end_of_frame(
            s._replace(
                game_state=s.game_state + 1
            ),
            carry
        )

    @partial(jit, static_argnums=(0,))
    def end_of_frame(self, s: GopherState, carry) -> GopherState:
        return s._replace(
            carry_state=carry
        )


# =============================== RENDER GAME ==================================


class GopherRenderer(JAXGameRenderer):

    def __init__(self, consts: GopherConstants | None = None):
        super().__init__()
        self.consts = consts or GopherConstants()
        self.sprites = self._load_sprites()
        self.dug_dirt_display_mask = int_arr(self.init_dug_dirt_display_mask())

    def init_dug_dirt_display_mask(self):
        # dirt bit at 0,0 never gets dug, so it's safe to use
        display_mask = np.zeros([210, 160])

        for y in range(4):
            for x in range(40):
                garden_flag_idx = y * 40 + x

                # Height depends on layer that gets rendered
                if y == 3:
                    h = 12
                elif y == 0:
                    h = 8
                else:
                    h = 7

                w = 4
                display_x = x * 4

                display_y = 195 - 12 - ((3 - y) * 7) - (1 if y == 0 else 0)

                display_mask[display_y:display_y + h, display_x:display_x + w] = garden_flag_idx

        return display_mask

    # Utility function
    @partial(jax.jit, static_argnums=(0, 2))
    def render_sprite(self, raster: jax.Array, name: str, x: IntValue = int_val(0), y: IntValue = int_val(0),
                      flip_horizontal: BoolValue = bool_val(False), frame_idx: IntValue = int_val(0)):
        sprite = jr.get_sprite_frame(self.sprites[name], frame_idx)
        raster = jr.render_at(raster, x, y, sprite, flip_horizontal=flip_horizontal)
        return raster

    def _load_sprites(self):
        """Load all sprites required for Gopher rendering."""
        MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
        sprite_path = os.path.join(MODULE_DIR, "sprites/gopher/")

        sprites: Dict[str, Any] = {}

        def _load_sprite_frame(name: str) -> ByteArray:
            path = os.path.join(sprite_path, f"{name}.npy")
            frame = jr.loadFrame(path)
            return frame.astype(jnp.uint8)

        # --- Load Sprites ---
        # Backgrounds + Dynamic elements + UI elements
        sprite_names = [
            "background",
            "carrot"
        ]

        for name in sprite_names:
            loaded_sprite = _load_sprite_frame(name)
            if loaded_sprite is not None:
                sprites[name] = loaded_sprite

        # expand all sprites similar to the Pong/Seaquest loading
        for key in sprites.keys():
            if isinstance(sprites[key], (list, tuple)):
                sprites[key] = [jnp.expand_dims(sprite, axis=0) for sprite in sprites[key]]
            else:
                sprites[key] = jnp.expand_dims(sprites[key], axis=0)

        sprites["digits"] = byte_arr(
            [_load_sprite_frame(f"score_{i}") for i in range(0, 10)]
        )

        sprites["gopher_rising"] = byte_arr(
            [_load_sprite_frame(f"gopher_rising_cutoff_{i}") for i in range(19, 24)]
        )
        sprites["gopher_running"] = byte_arr(
            [_load_sprite_frame(f"gopher_running_{i}") for i in range(1, 3)]
        )
        sprites["gopher_taunting"] = byte_arr(
            [_load_sprite_frame(f"gopher_taunting_{i}") for i in range(1, 3)]
        )

        sprites["duck"] = byte_arr(
            [_load_sprite_frame(f"duck_{i}") for i in ["none", "stationary", "up", "down"]]
        )

        sprites["farmer"] = byte_arr(
            [_load_sprite_frame(f"farmer_{i}") for i in ["standing", "digging_1", "digging_2"]]
        )

        return sprites

    @partial(jax.jit, static_argnums=(0, 2))
    def render(self, state: GopherState, replicate_render_artifacts: bool = True):
        """Render the game state to a raster image."""
        raster = jnp.zeros((210, 160, 3), dtype=jnp.uint8)
        raster = self.render_sprite(raster, "background")

        # Render dug dirt
        dirt_dug_mask = jnp.broadcast_to(jnp.expand_dims(state.garden_dirt_flags[self.dug_dirt_display_mask], axis=-1),
                                         raster.shape)
        raster = jnp.where(dirt_dug_mask, byte_arr([223, 183, 85]), raster)

        # Render carrots
        def render_carrot(i, raster):
            return jif_true(
                state.carrot_render_state[i],
                lambda raster: self.render_sprite(raster, "carrot", 92 - i * 16, int_val(151)),
                operand=raster
            )

        raster = jax.lax.fori_loop(0, 3, render_carrot, raster)

        # Render gopher
        raster = self.render_gopher(raster, state)

        # Render farmer
        raster = self.render_sprite(raster, "farmer", state.farmer_x - 10, int_val(96),
                                    frame_idx=state.farmer_render_state)

        # Render duck
        raster = self.render_duck(raster, state, replicate_render_artifacts)

        # Or, instead only render duck when visible (unclear which one results in better performance):
        # raster = jif_p2_true(
        #     state.duck_render_state != DuckRenderState.DISABLED.value,
        #     lambda raster, state: self.render_duck(raster, state, replicate_render_artifacts),
        #     operand=(raster, state)
        # )

        # Render seed
        raster = jif_p2_true(
            state.seed_visible,
            lambda raster, state: self.render_seed(raster, state, replicate_render_artifacts),
            operand=(raster, state)
        )

        # Render score
        player_score_digits_indices = jr.int_to_digits(state.score_value, max_digits=6)

        def render_digit(i, raster):
            digit_value = player_score_digits_indices[i]
            sprite_to_render = self.sprites["digits"][digit_value]
            return jr.render_at(raster, 57 + i * 8, 10, sprite_to_render)

        raster = jax.lax.fori_loop(
            get_first_index(player_score_digits_indices > 0, default_value=5),
            6,
            render_digit, raster
        )

        return raster

    @partial(jax.jit, static_argnums=(0,))
    def render_gopher(self, raster: ByteArray, s: GopherState):
        raster = jif_p2_true(
            jor(s.gopher_render_state == GopherRenderState.UNDERGROUND_RUNNING_1.value,
                s.gopher_render_state == GopherRenderState.UNDERGROUND_RUNNING_2.value),
            lambda raster, s: self.render_sprite(
                raster, "gopher_running",
                s.gopher_x - jselect(s.gopher_facing_left, int_val(3), int_val(1)),
                int_val(184),
                flip_horizontal=s.gopher_facing_left,
                frame_idx=s.gopher_render_state - GopherRenderState.UNDERGROUND_RUNNING_1.value
            ),
            operand=(raster, s)
        )

        raster = jif_p2_true(
            jor(s.gopher_render_state == GopherRenderState.ABOVE_GROUND_RUNNING_1.value,
                s.gopher_render_state == GopherRenderState.ABOVE_GROUND_RUNNING_2.value),
            lambda raster, s: self.render_sprite(
                raster, "gopher_running",
                s.gopher_x - jselect(s.gopher_facing_left, int_val(3), int_val(1)),
                149 + s.gopher_render_state - GopherRenderState.ABOVE_GROUND_RUNNING_1.value,
                flip_horizontal=s.gopher_facing_left,
                frame_idx=s.gopher_render_state - GopherRenderState.ABOVE_GROUND_RUNNING_1.value
            ),
            operand=(raster, s)
        )

        max_visual_height = jselect(s.gopher_y + 12 != 34, int_val(23), int_val(22))
        climb_sprite_height = jnp.minimum(max_visual_height, s.gopher_y + 12)

        raster = jif(
            s.gopher_render_state == GopherRenderState.RISING.value,
            lambda params: self.render_sprite(
                params[0], "gopher_rising",
                params[1].gopher_x - jselect(params[1].gopher_facing_left, int_val(3), int_val(4)),
                183 - params[1].gopher_y,
                flip_horizontal=params[1].gopher_facing_left,
                frame_idx=params[2] - 19
            ),
            lambda params: params[0],
            operand=(raster, s, climb_sprite_height)
        )

        raster = jif_p2_true(
            jor(s.gopher_render_state == GopherRenderState.TAUNTING_1.value,
                s.gopher_render_state == GopherRenderState.TAUNTING_2.value),
            lambda raster, s: self.render_sprite(
                raster, "gopher_taunting",
                s.gopher_x - jselect(s.gopher_facing_left, int_val(3), int_val(4)),
                183 - s.gopher_y,
                flip_horizontal=s.gopher_facing_left,
                frame_idx=s.gopher_render_state - GopherRenderState.TAUNTING_1.value
            ),
            operand=(raster, s)
        )

        return raster

    @partial(jax.jit, static_argnums=(0, 3))
    def render_duck(self, raster: ByteArray, state: GopherState, replicate_render_artifacts: bool) -> ByteArray:
        extra_x_offset = jselect(
            jand(state.duck_render_state == DuckRenderState.STATIONARY.value,
                 jnot(state.duck_moving_left)),
            int_val(1), int_val(0)
        ) + jselect(
            jand(state.duck_render_state != DuckRenderState.DOWN.value, state.duck_moving_left),
            int_val(-1), int_val(0)
        )

        extra_x_offset += jselect(
            jand(jand(replicate_render_artifacts, jor(
                jor(jand(36 <= state.duck_x, state.duck_x < 44), jand(68 <= state.duck_x, state.duck_x < 76)),
                jor(jand(100 <= state.duck_x, state.duck_x < 108), jand(132 <= state.duck_x, state.duck_x < 140))
            )), jnot(state.duck_moving_left)),
            byte_val(1), byte_val(0)
        )

        raster = self.render_sprite(
            raster, "duck",
            state.duck_x - 12 + extra_x_offset, int_val(31),
            flip_horizontal=state.duck_moving_left,
            frame_idx=state.duck_render_state
        )

        return raster

    @partial(jax.jit, static_argnums=(0, 3))
    def render_seed(self, raster: ByteArray, s: GopherState, replicate_render_artifacts: bool):

        def target_render_artifacts(s: GopherState) -> tuple[
            ByteValue, ByteValue]:
            return jif(
                s.seed_y <= 17,
                lambda s: jif(
                    jand(39 <= s.seed_x, s.seed_x < 42),
                    lambda s: (s.seed_y + 30, byte_val(0)),  # h = 0
                    lambda s: jif(
                        s.seed_x <= 38,
                        lambda s: (s.seed_y + 30 + 1, byte_val(1)),  # display_y += 1
                        lambda s: (s.seed_y + 30, byte_val(1)),  # DEFAULT
                        operand=s
                    ),
                    operand=s
                ),
                lambda s: jif(
                    s.seed_y == 18,
                    lambda s: jif(
                        s.seed_x < 42,
                        lambda s: (s.seed_y + 30 + 1, byte_val(7 - 1)),  # h = 7, display_y += 1, h -= 1
                        lambda s: (s.seed_y + 30, byte_val(7)),  # h = 7
                        operand=s
                    ),
                    lambda s: jif(
                        jand(s.seed_x <= 26, s.seed_y == 59),
                        lambda s: (s.seed_y + 30 + 6, byte_val(2)),  # display_y += 6, h = 2
                        lambda s: jif(
                            jand(s.seed_x <= 26, s.seed_y >= 60),
                            lambda s: (s.seed_y + 30 + 6 + 1, byte_val(1)),  # display_y += 6, display_y += 1
                            lambda s: jif(
                                jand(
                                    jand(s.seed_x >= 27, s.seed_x <= 29),
                                    s.seed_y >= 60
                                ),
                                lambda s: (s.seed_y + 30 + 6, byte_val(0)),  # display_y += 6, h = 0
                                lambda s: (s.seed_y + 30 + 6, byte_val(1)),  # display_y += 6
                                operand=s
                            ),
                            operand=s
                        ),
                        operand=s
                    ),
                    operand=s
                ),
                operand=s
            )

        def moving_render_artifacts(s: GopherState) -> tuple[
            ByteValue, ByteValue]:
            h = jselect(jand(39 <= s.seed_x, s.seed_x < 42), byte_val(0), byte_val(1))
            display_y = s.seed_y + 30 + 1 - jselect(42 <= s.seed_x, byte_val(1), byte_val(0))

            return display_y, h

        def render_artifacts(raster: ByteArray, s: GopherState) -> tuple[
            ByteValue, ByteValue]:
            return jif_p2(
                s.duck_seed_drop_target == s.seed_x,
                lambda raster, s: target_render_artifacts(s),
                lambda raster, s: moving_render_artifacts(s),
                operand=(raster, s)
            )

        display_x = s.seed_x - 5,

        display_y, h = jif_p2(
            jand(s.held_seed_decaying_timer == 0, replicate_render_artifacts),
            render_artifacts,
            lambda _, s: (s.seed_y + 30, byte_val(1)),
            operand=(raster, s)
        )

        display_y, h = jif(
            s.held_seed_decaying_timer > 0,
            lambda params: jif(
                replicate_render_artifacts,
                lambda params: jif(
                    jand(19 <= params[0], params[0] <= 26),
                    lambda params: (params[1] + 6 + 1, params[2]),
                    lambda params: jif(
                        jand(27 <= params[0], params[0] <= 29),
                        lambda params: (params[1] + 6, byte_val(0)),
                        lambda params: (params[1] + 6, params[2]),
                        operand=params
                    ),
                    operand=params
                ),
                lambda params: (params[1] + 6, params[2]),
                operand=params
            ),
            lambda params: (params[1], params[2]),
            operand=(s.seed_x, display_y, h)
        )

        def render_seed_pixels(raster: ByteArray, display_x: ByteValue, display_y: ByteValue,
                               h: ByteValue) -> ByteArray:
            def draw_seed_pixel(y_offset: int, params: tuple[ByteArray, ByteValue, ByteValue, ByteValue]) -> tuple[
                ByteArray, ByteValue, ByteValue, ByteValue]:
                raster, display_x, display_y, h = params

                raster = jif(
                    y_offset < h,
                    lambda params: params[0].at[params[2], params[1]].set(byte_arr([223, 183, 85])),
                    lambda params: params[0],
                    operand=(raster, display_x, display_y + y_offset)
                )
                return raster, display_x, display_y, h

            raster, _, _, _ = fori_loop(0, 7, draw_seed_pixel, (raster, display_x, display_y, h))
            return raster

        raster = jif(
            h > 0,
            lambda params: render_seed_pixels(params[0], params[1], params[2], params[3]),
            lambda params: params[0],
            operand=(raster, display_x, display_y, h)
        )

        return raster
