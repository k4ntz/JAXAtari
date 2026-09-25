from functools import partial
from pathlib import Path

import chex
import numpy as np
import jax
import jax.numpy as jnp
from flax import struct

import jaxatari.spaces as spaces
from jaxatari.environment import (
    JaxEnvironment,
    JAXAtariAction as Action,
    ObjectObservation,
)
from jaxatari.modification import AutoDerivedConstants
from jaxatari.renderers import JAXGameRenderer


# ============================================================
# Constants
# ============================================================

PLAYER_HIDDEN = 0
PLAYER_ACTIVE = 1
PLAYER_DYING = 2
PLAYER_WAITING = 3
PLAYER_MATERIALIZING = 4

ENEMY_HIDDEN = 0
ENEMY_ACTIVE = 1
ENEMY_DYING = 2
ENEMY_MATERIALIZING = 3


class StarGunnerConstants(AutoDerivedConstants):
    # ALE image dimensions: height=210, width=160.
    WIDTH: int = struct.field(pytree_node=False, default=160)
    HEIGHT: int = struct.field(pytree_node=False, default=210)

    # OCAtari describes the player bounding box as 8x4.
    PLAYER_WIDTH: int = struct.field(pytree_node=False, default=8)
    PLAYER_HEIGHT: int = struct.field(pytree_node=False, default=4)

    PLAYER_START_X: int = struct.field(pytree_node=False, default=70)
    PLAYER_START_Y: int = struct.field(pytree_node=False, default=119)

    PLAYER_SPEED_X: int = struct.field(
        pytree_node=False,
        default=1,
    )

    PLAYER_SPEED_Y: int = struct.field(
        pytree_node=False,
        default=2,
    )

    # ============================================================
    # Player world boundaries measured from ALE
    # ============================================================

    PLAYER_MIN_Y: int = struct.field(
        pytree_node=False,
        default=57,
    )

    PLAYER_MAX_Y: int = struct.field(
        pytree_node=False,
        default=183,
    )

    PLAYER_WRAP_RIGHT_THRESHOLD_X: int = struct.field(
        pytree_node=False,
        default=130,
    )

    PLAYER_WRAP_RIGHT_DESTINATION_X: int = struct.field(
        pytree_node=False,
        default=10,
    )

    PLAYER_WRAP_LEFT_THRESHOLD_X: int = struct.field(
        pytree_node=False,
        default=9,
    )

    PLAYER_WRAP_LEFT_DESTINATION_X: int = struct.field(
        pytree_node=False,
        default=130,
    )

    PLAYER_MIN_X: int = struct.field(
        pytree_node=False,
        default=10,
    )

    PLAYER_MAX_X: int = struct.field(
        pytree_node=False,
        default=130,
    )

    PLAYER_DEATH_DURATION: int = struct.field(
        pytree_node=False,
        default=31,
    )

    PLAYER_WAIT_DURATION: int = struct.field(
        pytree_node=False,
        default=61,
    )

    PLAYER_MATERIALIZE_DURATION: int = struct.field(
        pytree_node=False,
        default=32,
    )

    # missile

    MISSILE_WIDTH: int = struct.field(
        pytree_node=False,
        default=8,
    )

    MISSILE_HEIGHT: int = struct.field(
        pytree_node=False,
        default=1,
    )

    MISSILE_SPEED: int = struct.field(
        pytree_node=False,
        default=4,
    )

    # missile offset differs for left and right side in the gym version
    MISSILE_SPAWN_OFFSET_X_RIGHT: int = struct.field(
        pytree_node=False,
        default=7,
    )

    MISSILE_SPAWN_OFFSET_X_LEFT: int = struct.field(
        pytree_node=False,
        default=-9,
    )

    MISSILE_SPAWN_OFFSET_Y: int = struct.field(
        pytree_node=False,
        default=3,
    )

    # logical borders player, rocket
    WORLD_LEFT_X: int = struct.field(
        pytree_node=False,
        default=13,
    )

    WORLD_RIGHT_X: int = struct.field(
        pytree_node=False,
        default=141,
    )

    MISSILE_MIN_X_LEFT: int = struct.field(
        pytree_node=False,
        default=9,
    )

    MISSILE_MAX_X_RIGHT: int = struct.field(
        pytree_node=False,
        default=134,
    )

    # ============================================================
    # Enemies
    # ============================================================

    # True preserves ALE's brief wrong enemy sprite after a wave change (the promised boolean-guard for the ALE bug).
    EMULATE_TRANSIENT_WRONG_ENEMY_SPRITE: bool = struct.field(
        pytree_node=False,
        default=True,
    )

    ENEMY_MIN_X: int = struct.field(
        pytree_node=False,
        default=10,
    )

    ENEMY_MAX_X: int = struct.field(
        pytree_node=False,
        default=135,
    )

    ENEMY_MIN_Y: int = struct.field(
        pytree_node=False,
        default=54,
    )

    ENEMY_MAX_Y: int = struct.field(
        pytree_node=False,
        default=172,
    )

    ENEMY_DEATH_DURATION: int = struct.field(
        pytree_node=False,
        default=32,
    )

    ENEMY_MATERIALIZE_DURATION: int = struct.field(
        pytree_node=False,
        default=32,
    )

    WAVE_TRANSITION_DURATION: int = struct.field(
        pytree_node=False,
        default=32,
    )

    # ============================================================
    # Bobo
    # ============================================================

    BOBO_MIN_X: int = struct.field(
        pytree_node=False,
        default=12,
    )

    BOBO_MAX_X: int = struct.field(
        pytree_node=False,
        default=140,
    )

    BOBO_Y: int = struct.field(
        pytree_node=False,
        default=37,
    )

    BOBO_SPEED: int = struct.field(
        pytree_node=False,
        default=2,
    )

    BOMB_START_Y: int = struct.field(
        pytree_node=False,
        default=48,
    )

    BOMB_MAX_Y: int = struct.field(
        pytree_node=False,
        default=180,
    )

    BOMB_WIDTH: int = struct.field(
        pytree_node=False,
        default=4,
    )

    # ============================================================
    # Grass
    # ============================================================

    # grass: 189...209.
    GRASS_TOP_Y: int = struct.field(
        pytree_node=False,
        default=189,
    )

    GRASS_HEIGHT: int = struct.field(
        pytree_node=False,
        default=21,
    )

    # Interval = 80
    GRASS_TILE_WIDTH: int = struct.field(
        pytree_node=False,
        default=80,
    )

    # Step = 4px.
    GRASS_SCROLL_SPEED: int = struct.field(
        pytree_node=False,
        default=4,
    )

    # ============================================================
    # Intro
    # ============================================================

    START_SCREEN_LAST_STEP: int = struct.field(
        pytree_node=False,
        default=158,
    )

    INTRO_OUT_FIRST_STEP: int = struct.field(
        pytree_node=False,
        default=160,
    )

    INTRO_OUT_LAST_STEP: int = struct.field(
        pytree_node=False,
        default=190,
    )

    PLAYER_MATERIALIZE_FIRST_STEP: int = struct.field(
        pytree_node=False,
        default=252,
    )

    PLAYER_MATERIALIZE_LAST_STEP: int = struct.field(
        pytree_node=False,
        default=283,
    )

    GAMEPLAY_START_STEP: int = struct.field(
        pytree_node=False,
        default=284,
    )

    START_STAR_X: int = struct.field(
        pytree_node=False,
        default=61,
    )

    START_STAR_Y: int = struct.field(
        pytree_node=False,
        default=83,
    )

    START_STATIC_TEXT_X: int = struct.field(
        pytree_node=False,
        default=61,
    )

    START_STATIC_TEXT_Y: int = struct.field(
        pytree_node=False,
        default=95,
    )

    INTRO_OUT_CENTER_X: int = struct.field(
        pytree_node=False,
        default=70,
    )

    INTRO_OUT_CENTER_Y: int = struct.field(
        pytree_node=False,
        default=113,
    )

    PLAYER_MATERIALIZE_TARGET_X: int = struct.field(
        pytree_node=False,
        default=70,
    )

    PLAYER_MATERIALIZE_TARGET_Y: int = struct.field(
        pytree_node=False,
        default=118,
    )

    INITIAL_SCORE: int = struct.field(pytree_node=False, default=0)
    INITIAL_LIVES: int = struct.field(pytree_node=False, default=5)
    MAX_LIVES: int = struct.field(pytree_node=False, default=255)


# ============================================================
# State, observation and info
# ============================================================

@struct.dataclass
class StarGunnerState:
    player_x: chex.Array
    player_y: chex.Array
    player_facing: chex.Array
    player_phase: chex.Array
    player_animation_age: chex.Array

    started: chex.Array
    game_over: chex.Array
    score: chex.Array
    lives: chex.Array
    step_counter: chex.Array

    fire_was_pressed: chex.Array
    missile_active: chex.Array
    missile_x: chex.Array
    missile_y: chex.Array
    missile_direction: chex.Array
    missile_age: chex.Array

    round_index: chex.Array
    wave_index: chex.Array
    wave_transition_age: chex.Array
    enemy_reserve: chex.Array

    enemy_x: chex.Array
    enemy_y: chex.Array
    enemy_direction_x: chex.Array
    enemy_direction_y: chex.Array
    enemy_axis: chex.Array
    enemy_turn_cooldown: chex.Array
    enemy_move_accumulator: chex.Array
    enemy_phase: chex.Array
    enemy_phase_age: chex.Array
    enemy_visual_type: chex.Array
    enemy_visual_delay: chex.Array
    enemy_color_counter: chex.Array
    enemy_animation_pointer: chex.Array
    enemy_animation_timer: chex.Array
    enemy_render_slot: chex.Array
    enemy_respawn_pending: chex.Array
    enemy_death_x: chex.Array
    enemy_death_y: chex.Array

    bobo_x: chex.Array
    bobo_direction: chex.Array
    bomb_active: chex.Array
    bomb_x: chex.Array
    bomb_y: chex.Array
    bomb_age: chex.Array
    bomb_cooldown: chex.Array

    rng_state: chex.Array
    rom_entropy_state: chex.Array
    rom_entropy_stars: chex.Array


@struct.dataclass
class StarGunnerObservation:
    player: ObjectObservation
    score: jnp.ndarray
    lives: jnp.ndarray


@struct.dataclass
class StarGunnerInfo:
    step_counter: jnp.ndarray
    started: jnp.ndarray
    level: jnp.ndarray
    wave: jnp.ndarray
    enemies_remaining: jnp.ndarray


# ============================================================
# Helpers
# ============================================================

def _xorshift32(value: chex.Array) -> chex.Array:
    value = value.astype(jnp.uint32)
    value = value ^ (value << jnp.uint32(13))
    value = value ^ (value >> jnp.uint32(17))
    value = value ^ (value << jnp.uint32(5))
    return value.astype(jnp.uint32)


def _rom_random_byte(
    state: chex.Array,
    carry: chex.Array,
) -> tuple[chex.Array, chex.Array, chex.Array]:
    state = state.astype(jnp.uint32)
    carry = carry.astype(jnp.uint32) & jnp.uint32(1)
    low = state & jnp.uint32(255)
    high = (state >> jnp.uint32(8)) & jnp.uint32(255)
    shifted_low = (
        (low << jnp.uint32(1)) | carry
    ) & jnp.uint32(255)
    next_high = (high + jnp.uint32(1)) & jnp.uint32(255)
    total = (low ^ shifted_low) + next_high + carry
    value = total & jnp.uint32(255)
    next_carry = (total > jnp.uint32(255)).astype(jnp.uint32)
    next_state = value | (next_high << jnp.uint32(8))
    return next_state, value, next_carry


def _rom_entropy_tick(
    state: chex.Array,
    stars: chex.Array,
    player_raw_y: chex.Array,
    scroll_right: chex.Array,
) -> tuple[chex.Array, chex.Array]:
    stars = stars.astype(jnp.uint32)
    star_a = stars[0]
    star_b = stars[1]
    star_c = stars[2]
    slots = jnp.arange(9, dtype=jnp.int32)
    rotated_slots = slots > 0

    right_a = (star_a >> jnp.uint32(1)) | (star_b & jnp.uint32(0x80))
    right_b = (
        (star_b << jnp.uint32(1)) | (star_c & jnp.uint32(1))
    ) & jnp.uint32(255)
    right_c = (
        (star_c >> jnp.uint32(1))
        | jnp.where(
            (star_a & jnp.uint32(0x10)) != 0,
            jnp.uint32(0x80),
            jnp.uint32(0),
        )
    )
    left_a = (
        (star_a << jnp.uint32(1))
        | ((star_c & jnp.uint32(0x80)) >> jnp.uint32(3))
    ) & jnp.uint32(255)
    left_b = (
        (star_b >> jnp.uint32(1)) | (star_a & jnp.uint32(0x80))
    )
    left_c = (
        (star_c << jnp.uint32(1)) | (star_b & jnp.uint32(1))
    ) & jnp.uint32(255)

    rotated_a = jnp.where(scroll_right, right_a, left_a)
    rotated_b = jnp.where(scroll_right, right_b, left_b)
    rotated_c = jnp.where(scroll_right, right_c, left_c)
    next_stars = jnp.stack(
        [
            jnp.where(rotated_slots, rotated_a, star_a),
            jnp.where(rotated_slots, rotated_b, star_b),
            jnp.where(rotated_slots, rotated_c, star_c),
        ]
    )

    star_carry = (
        scroll_right & ((star_b[1] & jnp.uint32(0x80)) != 0)
    ).astype(jnp.uint32)
    low = state.astype(jnp.uint32) & jnp.uint32(255)
    high = state.astype(jnp.uint32) & jnp.uint32(0xFF00)
    next_low = (
        low
        + (player_raw_y.astype(jnp.uint32) & jnp.uint32(15))
        + star_carry
    ) & jnp.uint32(255)
    next_state = high | next_low
    return next_state, next_stars.astype(jnp.int32)


def _enemy_type(round_index: chex.Array) -> chex.Array:
    return jnp.where(
        (round_index % 2) == 0,
        0,
        jnp.where((round_index % 4) == 1, 1, 2),
    ).astype(jnp.int32)


def _level_bonus(level: chex.Array) -> chex.Array:
    first_cycle = jnp.array(
        [400, 600, 1000, 1100],
        dtype=jnp.int32,
    )
    cycle = (level - 1) // 4
    index = (level - 1) % 4
    return (first_cycle[index] + cycle * 1000).astype(jnp.int32)


def _wave_bonus(level: chex.Array, wave: chex.Array) -> chex.Array:
    del wave
    return _level_bonus(level)


def _enemy_speed_fixed(level: chex.Array) -> chex.Array:
    integer_speed = level // 4
    fractional_speed = (level % 4) * 80
    return (integer_speed * 256 + fractional_speed).astype(jnp.int32)


# ============================================================
# Renderer
# ============================================================

class StarGunnerRenderer(JAXGameRenderer):
    def __init__(self, consts: StarGunnerConstants):
        super().__init__(consts)
        self.consts = consts

        # Game assets.
        assets_path = Path(__file__).with_name(
            "stargunner_intro_assets.npz"
        )

        with np.load(assets_path) as assets:
            self.star_frames = jnp.asarray(
                assets["star_rgb_frames"],
                dtype=jnp.uint8,
            )
            self.star_masks = jnp.asarray(
                assets["star_mask_frames"],
                dtype=jnp.bool_,
            )
            self.static_title_sprite = jnp.asarray(
                assets["static_title_rgb"],
                dtype=jnp.uint8,
            )
            self.static_title_mask = jnp.asarray(
                assets["static_title_mask"],
                dtype=jnp.bool_,
            )
            self.intro_out_colors = jnp.asarray(
                assets["intro_out_colors"],
                dtype=jnp.uint8,
            )
            self.player_in_colors = jnp.asarray(
                assets["player_in_colors"],
                dtype=jnp.uint8,
            )
            self.enemy_masks = jnp.asarray(
                assets["enemy_masks"],
                dtype=jnp.bool_,
            )
            self.enemy_phase_counts = jnp.asarray(
                assets["enemy_phase_counts"],
                dtype=jnp.int32,
            )
            self.enemy_sizes = jnp.asarray(
                assets["enemy_sizes"],
                dtype=jnp.int32,
            )
            self.bobo_masks = jnp.asarray(
                assets["bobo_masks"],
                dtype=jnp.bool_,
            )
            self.digit_masks = jnp.asarray(
                assets["digit_masks"],
                dtype=jnp.bool_,
            )
            self.round_palettes = jnp.asarray(
                assets["round_palettes"],
                dtype=jnp.uint8,
            )
            self.enemy_palette_bank = jnp.asarray(
                assets["enemy_palette_bank"],
                dtype=jnp.uint8,
            )

        self.enemy_pointer_visual_types = jnp.array(
            [0, 0, 0, 0, 2, 2, 2, 2, 1, 1, 1, 1],
            dtype=jnp.int32,
        )
        self.enemy_pointer_animation_indices = jnp.array(
            [2, 3, 0, 1, 2, 1, 0, 3, 1, 0, 3, 2],
            dtype=jnp.int32,
        )

        right_mask = jnp.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1],
            ],
            dtype=jnp.bool_,
        )

        player_colour = jnp.array(
            [214, 92, 92],
            dtype=jnp.uint8,
        )

        self.player_right = jnp.where(
            right_mask[..., None],
            player_colour,
            jnp.array(0, dtype=jnp.uint8),
        )
        self.player_left = jnp.flip(self.player_right, axis=1)
        self.player_mask_right = right_mask
        self.player_mask_left = jnp.flip(right_mask, axis=1)

        self.score_colour = jnp.array(
            [101, 160, 225],
            dtype=jnp.uint8,
        )

        self.missile_colors = jnp.array(
            [
                [144, 252, 144],
                [128, 232, 128],
                [111, 210, 111],
                [92, 186, 92],
                [72, 160, 72],
                [50, 132, 50],
                [26, 102, 26],
                [0, 68, 0],
                [132, 252, 212],
                [117, 231, 194],
                [101, 209, 174],
                [84, 184, 153],
                [66, 158, 130],
                [45, 129, 105],
                [24, 98, 78],
            ],
            dtype=jnp.uint8,
        )

        self.bobo_colors = jnp.array(
            [
                [101, 160, 225],
                [84, 138, 210],
                [66, 114, 194],
                [45, 87, 176],
                [78, 50, 181],
                [104, 72, 198],
                [127, 92, 213],
                [149, 111, 227],
                [169, 128, 240],
                [188, 144, 252],
                [213, 130, 74],
                [240, 170, 103],
                [252, 188, 116],
                [228, 111, 111],
                [200, 72, 72],
                [184, 50, 50],
            ],
            dtype=jnp.uint8,
        )

        grass_palette = jnp.array(
            [
                [0, 0, 0],
                [26, 102, 26],
                [50, 132, 50],
                [72, 160, 72],
                [92, 186, 92],
                [111, 210, 111],
            ],
            dtype=jnp.uint8,
        )

        grass_rows = (
            "00000000000000000000000000005555555555555555000000000000000000000000000000000000",
            "00000000000000000000000055555555555555555555555500000000000000000000000000000000",
            "00000000000000000000444444444444444444444444444444440000000000000000000000000000",
            "00000000000000004444444444444444444444444444444444444444000000000000000000000000",
            "00000000000033333333333333333333333333333333333333333333333300000000000000000000",
            "00000000333333333333333333333333333333333333333333333333333333330000000000000000",
            "00002222222222222222222222222222222222222222222222222222222222222222000000000000",
            "22222222222222222222222222222222222222222222222222222222222222222222222200000000",
            "11111111111111111111111111111111111111111111111111111111111111111111111111111111",
            "11111111111111111111111111111111111111111111111111111111111111111111111111111111",
            "11111111111111111111111111111111111111111111111111111111111111111111111111111111",
            "11111111111111111111111111111111111111111111111111111111111111111111111111111111",
            "11111111111111111111111111111111111111111111111111111111111111111111111111111111",
            "11111111111111111111111111111111111111111111111111111111111111111111111111111111",
            "11111111111111111111111111111111111111111111111111111111111111111111111111111111",
            "11111111111111111111111111111111111111111111111111111111111111111111111111111111",
            "11111111111111111111111111111111111111111111111111111111111111111111111111111111",
            "11111111111111111111111111111111111111111111111111111111111111111111111111111111",
            "11111111111111111111111111111111111111111111111111111111111111111111111111111111",
            "11111111111111111111111111111111111111111111111111111111111111111111111111111111",
            "11111111111111111111111111111111111111111111111111111111111111111111111111111111",
        )

        grass_indices = jnp.array(
            [
                [int(character) for character in row]
                for row in grass_rows
            ],
            dtype=jnp.int32,
        )

        self.grass_tile = grass_palette[grass_indices]
        self.grass_repeated = jnp.concatenate(
            [self.grass_tile, self.grass_tile, self.grass_tile],
            axis=1,
        )

    def _overlay_masked(
        self,
        frame: jnp.ndarray,
        sprite: jnp.ndarray,
        mask: jnp.ndarray,
        y: chex.Array,
        x: chex.Array,
    ) -> jnp.ndarray:
        height = sprite.shape[0]
        width = sprite.shape[1]
        old = jax.lax.dynamic_slice(
            frame,
            (y, x, 0),
            (height, width, 3),
        )
        composed = jnp.where(mask[..., None], sprite, old)
        return jax.lax.dynamic_update_slice(
            frame,
            composed,
            (y, x, 0),
        )

    def _draw_pixel(
        self,
        frame: jnp.ndarray,
        x: chex.Array,
        y: chex.Array,
        color: chex.Array,
    ) -> jnp.ndarray:
        valid = (
            (x >= 0)
            & (x < self.consts.WIDTH)
            & (y >= 0)
            & (y < self.consts.HEIGHT)
        )
        safe_x = jnp.clip(x, 0, self.consts.WIDTH - 1)
        safe_y = jnp.clip(y, 0, self.consts.HEIGHT - 1)
        old = frame[safe_y, safe_x]
        return frame.at[safe_y, safe_x].set(
            jnp.where(valid, color, old)
        )

    def _draw_line(
        self,
        frame: jnp.ndarray,
        x: chex.Array,
        y: chex.Array,
        width: int,
        color: chex.Array,
    ) -> jnp.ndarray:
        for offset in range(width):
            frame = self._draw_pixel(frame, x + offset, y, color)
        return frame

    def _draw_hud(self, frame: jnp.ndarray, state: StarGunnerState) -> jnp.ndarray:
        score = jnp.maximum(state.score, 0)

        # Last two digits are always zero.
        zero_mask = self.digit_masks[0]
        zero_sprite = jnp.where(
            zero_mask[..., None],
            self.score_colour,
            jnp.array(0, dtype=jnp.uint8),
        )
        for x in (82, 90):
            frame = self._overlay_masked(
                frame,
                zero_sprite,
                zero_mask,
                jnp.array(14, dtype=jnp.int32),
                jnp.array(x, dtype=jnp.int32),
            )

        score_hundreds = score // 100
        for index in range(4):
            power = 10 ** index
            digit = (score_hundreds // power) % 10
            mask = self.digit_masks[digit]
            sprite = jnp.where(
                mask[..., None],
                self.score_colour,
                jnp.array(0, dtype=jnp.uint8),
            )
            frame = jax.lax.cond(
                score_hundreds >= power,
                lambda current, sp=sprite, ma=mask, idx=index: self._overlay_masked(
                    current,
                    sp,
                    ma,
                    jnp.array(14, dtype=jnp.int32),
                    jnp.array(74 - idx * 8, dtype=jnp.int32),
                ),
                lambda current: current,
                frame,
            )

        visible_lives = jnp.minimum(state.lives, 3)
        for index in range(3):
            frame = jax.lax.cond(
                visible_lives > index,
                lambda current, idx=index: self._overlay_masked(
                    current,
                    self.player_right,
                    self.player_mask_right,
                    jnp.array(23, dtype=jnp.int32),
                    jnp.array(56 + idx * 16, dtype=jnp.int32),
                ),
                lambda current: current,
                frame,
            )
        return frame

    def _enemy_slot_color(
        self,
        state: StarGunnerState,
        slot: int,
    ) -> jnp.ndarray:
        color_counter = state.enemy_color_counter[slot]
        hue = (color_counter >> 4) & 15
        luminance = (color_counter & 15) >> 1
        return self.enemy_palette_bank[hue, luminance]

    def _draw_enemy_particles(
        self,
        frame: jnp.ndarray,
        state: StarGunnerState,
        slot: int,
        reverse: bool,
    ) -> jnp.ndarray:
        age = state.enemy_phase_age[slot]
        death_x = state.enemy_death_x[slot]
        death_y = state.enemy_death_y[slot]
        target_x = state.enemy_x[slot]
        target_y = state.enemy_y[slot]
        sprite_x_adjustment = jnp.where(
            state.enemy_visual_type[slot] == 2,
            1,
            0,
        )

        outgoing_steps = jnp.maximum(age - 1, 0)
        outgoing_start_x = death_x + 8 - sprite_x_adjustment
        first_wrap_step = (outgoing_start_x - 9) // 2 + 1
        outgoing_x = outgoing_start_x - outgoing_steps * 2
        wrapped_outgoing_x = (
            138
            - (outgoing_steps - first_wrap_step) * 2
        )
        outgoing_x = jnp.where(
            outgoing_steps >= first_wrap_step,
            wrapped_outgoing_x,
            outgoing_x,
        )

        fixed_x = (
            jnp.where(reverse, target_x, death_x)
            - sprite_x_adjustment
        )
        moving_x = jnp.where(
            reverse,
            target_x - 65 - sprite_x_adjustment + age * 2,
            outgoing_x,
        )
        upper_y = jnp.where(
            reverse,
            target_y - 63 + age * 2,
            death_y - age * 2,
        )
        lower_y = jnp.where(
            reverse,
            target_y + 65 - age * 2,
            death_y - 2 + age * 2,
        )
        color = self._enemy_slot_color(state, slot)

        for x, y in (
            (fixed_x, upper_y),
            (moving_x, upper_y),
            (fixed_x, lower_y),
            (moving_x, lower_y),
        ):
            # The enemy kernel ends before the ground scanlines.  Particle
            # coordinates keep advancing in RAM, but ALE does not draw them
            # once their raw Y reaches 156 (screen Y 188).
            visible_y = jnp.where(
                (y >= 0) & (y < self.consts.GRASS_TOP_Y - 1),
                y,
                -1,
            )
            frame = self._draw_line(frame, x, visible_y, 8, color)
        return frame

    def _draw_player_particles(
        self,
        frame: jnp.ndarray,
        state: StarGunnerState,
        reverse: bool,
    ) -> jnp.ndarray:
        age = state.player_animation_age

        distance = jnp.where(
            reverse,
            self.consts.PLAYER_MATERIALIZE_DURATION - age,
            age,
        ).astype(jnp.int32)

        color_index = jnp.minimum(
            age // 2,
            self.player_in_colors.shape[0] - 1,
        )

        color = self.player_in_colors[
            color_index
        ]

        target_x = state.player_x

        # particles are one pixel above player y
        target_y = state.player_y - 1

        moving_x = target_x + distance * 2
        top_y = target_y - distance * 2
        bottom_y = target_y + distance * 2

        for x, y in (
            (target_x, top_y),
            (moving_x, top_y),
            (moving_x, target_y),
            (target_x, bottom_y),
            (moving_x, bottom_y),
        ):
            frame = self._draw_line(
                frame,
                x,
                y,
                8,
                color,
            )

        return frame

    
    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: StarGunnerState) -> jnp.ndarray:
        frame = jnp.zeros(
            (self.consts.HEIGHT, self.consts.WIDTH, 3),
            dtype=jnp.uint8,
        )

        grass_offset = (
            state.step_counter * self.consts.GRASS_SCROLL_SPEED
        ) % self.consts.GRASS_TILE_WIDTH
        grass_sprite = jax.lax.dynamic_slice(
            self.grass_repeated,
            (0, grass_offset.astype(jnp.int32), 0),
            (
                self.consts.GRASS_HEIGHT,
                self.consts.WIDTH,
                3,
            ),
        )
        frame = jax.lax.dynamic_update_slice(
            frame,
            grass_sprite,
            (self.consts.GRASS_TOP_Y, 0, 0),
        )

        frame = self._draw_hud(frame, state)

        show_start_screen = (
            state.step_counter <= self.consts.START_SCREEN_LAST_STEP
        )
        star_index = jnp.clip(
            state.step_counter,
            0,
            self.consts.START_SCREEN_LAST_STEP,
        ).astype(jnp.int32)

        def draw_start(current_frame):
            current_frame = self._overlay_masked(
                current_frame,
                self.star_frames[star_index],
                self.star_masks[star_index],
                jnp.array(self.consts.START_STAR_Y, dtype=jnp.int32),
                jnp.array(self.consts.START_STAR_X, dtype=jnp.int32),
            )
            return self._overlay_masked(
                current_frame,
                self.static_title_sprite,
                self.static_title_mask,
                jnp.array(self.consts.START_STATIC_TEXT_Y, dtype=jnp.int32),
                jnp.array(self.consts.START_STATIC_TEXT_X, dtype=jnp.int32),
            )

        frame = jax.lax.cond(
            show_start_screen,
            draw_start,
            lambda current: current,
            frame,
        )

        show_intro_out = (
            (state.step_counter >= self.consts.INTRO_OUT_FIRST_STEP)
            & (state.step_counter <= self.consts.INTRO_OUT_LAST_STEP)
        )

        def draw_intro_out(current_frame):
            age = state.step_counter - self.consts.INTRO_OUT_FIRST_STEP
            color = self.intro_out_colors[
                jnp.minimum(age // 2, self.intro_out_colors.shape[0] - 1)
            ]
            center_x = jnp.array(self.consts.INTRO_OUT_CENTER_X, jnp.int32)
            center_y = jnp.array(self.consts.INTRO_OUT_CENTER_Y, jnp.int32)
            moving_x = center_x + age * 2
            top_y = center_y - age * 2
            bottom_y = center_y + age * 2
            for x, y in (
                (center_x, top_y),
                (moving_x, top_y),
                (moving_x, center_y),
                (center_x, bottom_y),
                (moving_x, bottom_y),
            ):
                current_frame = self._draw_line(
                    current_frame,
                    x,
                    y,
                    8,
                    color,
                )
            return current_frame

        frame = jax.lax.cond(
            show_intro_out,
            draw_intro_out,
            lambda current: current,
            frame,
        )

        show_intro_player = (
            (state.step_counter >= self.consts.PLAYER_MATERIALIZE_FIRST_STEP)
            & (state.step_counter <= self.consts.PLAYER_MATERIALIZE_LAST_STEP)
        )

        def draw_intro_player(current_frame):
            age = state.step_counter - self.consts.PLAYER_MATERIALIZE_FIRST_STEP
            distance = 32 - age
            color = self.player_in_colors[
                jnp.minimum(age // 2, self.player_in_colors.shape[0] - 1)
            ]
            target_x = jnp.array(
                self.consts.PLAYER_MATERIALIZE_TARGET_X,
                jnp.int32,
            )
            target_y = jnp.array(
                self.consts.PLAYER_MATERIALIZE_TARGET_Y,
                jnp.int32,
            )
            moving_x = target_x + distance * 2
            top_y = target_y - distance * 2
            bottom_y = target_y + distance * 2
            for x, y in (
                (target_x, top_y),
                (moving_x, top_y),
                (moving_x, target_y),
                (target_x, bottom_y),
                (moving_x, bottom_y),
            ):
                current_frame = self._draw_line(
                    current_frame,
                    x,
                    y,
                    8,
                    color,
                )
            return current_frame

        frame = jax.lax.cond(
            show_intro_player,
            draw_intro_player,
            lambda current: current,
            frame,
        )

        # Bobo.
        show_game_objects = (
            state.player_phase == PLAYER_ACTIVE
        )
        bobo_color = self.bobo_colors[
            jnp.maximum(state.step_counter - 285, 0)
            // 2
            % self.bobo_colors.shape[0]
        ]
        bobo_mask = self.bobo_masks[(state.step_counter // 8) % 2]
        bobo_sprite = jnp.where(
            bobo_mask[..., None],
            bobo_color,
            jnp.array(0, dtype=jnp.uint8),
        )
        frame = jax.lax.cond(
            show_game_objects
            & state.started
            & ~state.game_over
            & (state.step_counter >= 285),
            lambda current: self._overlay_masked(
                current,
                bobo_sprite,
                bobo_mask,
                jnp.array(self.consts.BOBO_Y, jnp.int32),
                state.bobo_x,
            ),
            lambda current: current,
            frame,
        )

        animation_pointer_index = jnp.clip(
            (state.enemy_animation_pointer - 88) // 11,
            0,
            11,
        )
        pointer_visual_type = self.enemy_pointer_visual_types[
            animation_pointer_index
        ]
        logical_visual_type = _enemy_type(state.round_index)
        sprite_visual_type = jnp.where(
            self.consts.EMULATE_TRANSIENT_WRONG_ENEMY_SPRITE,
            pointer_visual_type,
            logical_visual_type,
        )
        animation_index = self.enemy_pointer_animation_indices[
            animation_pointer_index
        ]
        mask = self.enemy_masks[sprite_visual_type, animation_index]
        # The ROM keeps advancing the old shared sprite pointer after a wave
        # change.  If that transient pointer selects the narrow enemy, TIA's
        # raw X conversion changes even though the logical enemy type has not.
        sprite_x_offset = (
            (sprite_visual_type == 2).astype(jnp.int32)
            - (logical_visual_type == 2).astype(jnp.int32)
        )
        # Pointer $84 (animation table index 4) is one scanline lower in ALE.
        sprite_y_offset = (animation_pointer_index == 4).astype(jnp.int32)

        # The ROM renders no enemies during the complete inter-wave pause,
        # including the final frame whose RAM countdown has just reached zero.
        show_enemies = show_game_objects & (state.wave_transition_age == 0)

        # Enemies.
        for slot in range(3):
            color = self._enemy_slot_color(state, slot)
            sprite = jnp.where(
                mask[..., None],
                color,
                jnp.array(0, dtype=jnp.uint8),
            )

            frame = jax.lax.cond(
                show_enemies
                & (state.enemy_render_slot == slot)
                & (
                    (state.enemy_phase[slot] == ENEMY_ACTIVE)
                    | (
                        (state.enemy_phase[slot] == ENEMY_DYING)
                        & (state.enemy_phase_age[slot] == 0)
                    )
                )
                & (state.step_counter >= 285),
                lambda current, s=slot, sp=sprite, ma=mask: self._overlay_masked(
                    current,
                    sp,
                    ma,
                    state.enemy_y[s] + sprite_y_offset,
                    state.enemy_x[s] + sprite_x_offset,
                ),
                lambda current: current,
                frame,
            )

            frame = jax.lax.cond(
                show_enemies
                & (state.enemy_render_slot == slot)
                & (state.enemy_phase[slot] == ENEMY_DYING)
                & (state.enemy_phase_age[slot] > 0),
                lambda current, s=slot: self._draw_enemy_particles(
                    current,
                    state,
                    s,
                    False,
                ),
                lambda current: current,
                frame,
            )

            frame = jax.lax.cond(
                show_enemies
                & (state.enemy_render_slot == slot)
                & (state.enemy_phase[slot] == ENEMY_MATERIALIZING),
                lambda current, s=slot: self._draw_enemy_particles(
                    current,
                    state,
                    s,
                    True,
                ),
                lambda current: current,
                frame,
            )

        # Bobo bomb.
        slot_colors_raw = jnp.stack(
            [self._enemy_slot_color(state, slot) for slot in range(3)],
            axis=0,
        )
        max_channels = jnp.maximum(
            jnp.max(slot_colors_raw.astype(jnp.int32), axis=1, keepdims=True),
            1,
        )
        slot_colors = jnp.minimum(
            slot_colors_raw.astype(jnp.int32) * 252 // max_channels,
            255,
        ).astype(jnp.uint8)
        active_colors = state.enemy_phase == ENEMY_ACTIVE
        color_count = jnp.maximum(jnp.sum(active_colors.astype(jnp.int32)), 1)
        color_rank = (state.bomb_age // 2) % color_count
        ranks = jnp.cumsum(active_colors.astype(jnp.int32)) - 1
        selected = active_colors & (ranks == color_rank)
        bomb_color = jnp.where(
            jnp.any(selected),
            jnp.sum(
                jnp.where(selected[:, None], slot_colors, 0),
                axis=0,
                dtype=jnp.uint16,
            ).astype(jnp.uint8),
            bobo_color,
        )
        bomb_sprite = jnp.broadcast_to(
            bomb_color,
            (1, self.consts.BOMB_WIDTH, 3),
        )
        frame = jax.lax.cond(
            show_game_objects
            & state.bomb_active,
            lambda current: jax.lax.dynamic_update_slice(
                current,
                bomb_sprite,
                (state.bomb_y, state.bomb_x, 0),
            ),
            lambda current: current,
            frame,
        )

        # Player missile.
        missile_color = self.missile_colors[
            (state.missile_age // 2) % self.missile_colors.shape[0]
        ]
        missile_sprite = jnp.broadcast_to(
            missile_color,
            (1, self.consts.MISSILE_WIDTH, 3),
        )
        missile_pixels = state.missile_x + jnp.arange(
            self.consts.MISSILE_WIDTH,
            dtype=jnp.int32,
        )
        missile_mask = (
            (missile_pixels >= self.consts.WORLD_LEFT_X)
            & (missile_pixels <= self.consts.WORLD_RIGHT_X)
        )
        missile_screen_y = state.missile_y - 1

        def draw_missile(current):
            old = jax.lax.dynamic_slice(
                current,
                (missile_screen_y, state.missile_x, 0),
                (1, self.consts.MISSILE_WIDTH, 3),
            )
            composed = jnp.where(
                missile_mask[None, :, None],
                missile_sprite,
                old,
            )
            return jax.lax.dynamic_update_slice(
                current,
                composed,
                (missile_screen_y, state.missile_x, 0),
            )

        frame = jax.lax.cond(
            state.missile_active,
            draw_missile,
            lambda current: current,
            frame,
        )

        player_sprite = jnp.where(
            state.player_facing == 1,
            self.player_right,
            self.player_left,
        )
        player_mask = jnp.where(
            state.player_facing == 1,
            self.player_mask_right,
            self.player_mask_left,
        )
        frame = jax.lax.cond(
            state.player_phase == PLAYER_ACTIVE,
            lambda current: self._overlay_masked(
                current,
                player_sprite,
                player_mask,
                state.player_y,
                state.player_x,
            ),
            lambda current: current,
            frame,
        )
        frame = jax.lax.cond(
            state.player_phase == PLAYER_DYING,
            lambda current: self._draw_player_particles(
                current,
                state,
                False,
            ),
            lambda current: current,
            frame,
        )
        frame = jax.lax.cond(
            state.player_phase == PLAYER_MATERIALIZING,
            lambda current: self._draw_player_particles(
                current,
                state,
                True,
            ),
            lambda current: current,
            frame,
        )

        return frame


# ============================================================
# Environment
# ============================================================

class JaxStarGunner(
    JaxEnvironment[
        StarGunnerState,
        StarGunnerObservation,
        StarGunnerInfo,
        StarGunnerConstants,
    ]
):
    ACTION_SET: jnp.ndarray = jnp.array(
        [
            Action.NOOP,
            Action.FIRE,
            Action.UP,
            Action.RIGHT,
            Action.LEFT,
            Action.DOWN,
            Action.UPRIGHT,
            Action.UPLEFT,
            Action.DOWNRIGHT,
            Action.DOWNLEFT,
            Action.UPFIRE,
            Action.RIGHTFIRE,
            Action.LEFTFIRE,
            Action.DOWNFIRE,
            Action.UPRIGHTFIRE,
            Action.UPLEFTFIRE,
            Action.DOWNRIGHTFIRE,
            Action.DOWNLEFTFIRE,
        ],
        dtype=jnp.int32,
    )

    FIRE_ACTIONS = jnp.array(
        [
            Action.FIRE,
            Action.UPFIRE,
            Action.RIGHTFIRE,
            Action.LEFTFIRE,
            Action.DOWNFIRE,
            Action.UPRIGHTFIRE,
            Action.UPLEFTFIRE,
            Action.DOWNRIGHTFIRE,
            Action.DOWNLEFTFIRE,
        ],
        dtype=jnp.int32,
    )

    UP_ACTIONS = jnp.array(
        [
            Action.UP,
            Action.UPRIGHT,
            Action.UPLEFT,
            Action.UPFIRE,
            Action.UPRIGHTFIRE,
            Action.UPLEFTFIRE,
        ],
        dtype=jnp.int32,
    )

    DOWN_ACTIONS = jnp.array(
        [
            Action.DOWN,
            Action.DOWNRIGHT,
            Action.DOWNLEFT,
            Action.DOWNFIRE,
            Action.DOWNRIGHTFIRE,
            Action.DOWNLEFTFIRE,
        ],
        dtype=jnp.int32,
    )

    RIGHT_ACTIONS = jnp.array(
        [
            Action.RIGHT,
            Action.UPRIGHT,
            Action.DOWNRIGHT,
            Action.RIGHTFIRE,
            Action.UPRIGHTFIRE,
            Action.DOWNRIGHTFIRE,
        ],
        dtype=jnp.int32,
    )

    LEFT_ACTIONS = jnp.array(
        [
            Action.LEFT,
            Action.UPLEFT,
            Action.DOWNLEFT,
            Action.LEFTFIRE,
            Action.UPLEFTFIRE,
            Action.DOWNLEFTFIRE,
        ],
        dtype=jnp.int32,
    )

    def __init__(self, consts: StarGunnerConstants | None = None):
        consts = consts or StarGunnerConstants()
        super().__init__(consts)
        self.renderer = StarGunnerRenderer(self.consts)

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self,
        key: chex.PRNGKey | None = None,
    ) -> tuple[StarGunnerObservation, StarGunnerState]:
        if key is None:
            seed = jnp.uint32(0x13579BDF)
        else:
            flat_key = jnp.ravel(jnp.asarray(key, dtype=jnp.uint32))
            seed = flat_key[0] ^ flat_key[-1] ^ jnp.uint32(0x13579BDF)

        state = StarGunnerState(
            player_x=jnp.array(self.consts.PLAYER_START_X, jnp.int32),
            player_y=jnp.array(self.consts.PLAYER_START_Y, jnp.int32),
            player_facing=jnp.array(1, jnp.int32),
            player_phase=jnp.array(PLAYER_HIDDEN, jnp.int32),
            player_animation_age=jnp.array(0, jnp.int32),
            started=jnp.array(False, jnp.bool_),
            game_over=jnp.array(False, jnp.bool_),
            score=jnp.array(self.consts.INITIAL_SCORE, jnp.int32),
            lives=jnp.array(self.consts.INITIAL_LIVES, jnp.int32),
            step_counter=jnp.array(0, jnp.int32),
            fire_was_pressed=jnp.array(False, jnp.bool_),
            missile_active=jnp.array(False, jnp.bool_),
            missile_x=jnp.array(0, jnp.int32),
            missile_y=jnp.array(0, jnp.int32),
            missile_direction=jnp.array(0, jnp.int32),
            missile_age=jnp.array(0, jnp.int32),
            round_index=jnp.array(0, jnp.int32),
            wave_index=jnp.array(0, jnp.int32),
            wave_transition_age=jnp.array(0, jnp.int32),
            enemy_reserve=jnp.array(0, jnp.int32),
            enemy_x=jnp.array([20, 20, 120], jnp.int32),
            enemy_y=jnp.array([63, 163, 63], jnp.int32),
            enemy_direction_x=jnp.zeros((3,), jnp.int32),
            enemy_direction_y=jnp.zeros((3,), jnp.int32),
            enemy_axis=jnp.zeros((3,), jnp.int32),
            enemy_turn_cooldown=jnp.zeros((3,), jnp.int32),
            enemy_move_accumulator=jnp.zeros((3,), jnp.int32),
            enemy_phase=jnp.zeros((3,), jnp.int32),
            enemy_phase_age=jnp.zeros((3,), jnp.int32),
            enemy_visual_type=jnp.zeros((3,), jnp.int32),
            enemy_visual_delay=jnp.zeros((3,), jnp.int32),
            enemy_color_counter=jnp.zeros((3,), jnp.int32),
            enemy_animation_pointer=jnp.array(88, jnp.int32),
            enemy_animation_timer=jnp.array(5, jnp.int32),
            enemy_render_slot=jnp.array(0, jnp.int32),
            enemy_respawn_pending=jnp.zeros((3,), jnp.bool_),
            enemy_death_x=jnp.array([20, 20, 120], jnp.int32),
            enemy_death_y=jnp.array([63, 163, 63], jnp.int32),
            bobo_x=jnp.array(140, jnp.int32),
            bobo_direction=jnp.array(-1, jnp.int32),
            bomb_active=jnp.array(False, jnp.bool_),
            bomb_x=jnp.array(0, jnp.int32),
            bomb_y=jnp.array(0, jnp.int32),
            bomb_age=jnp.array(0, jnp.int32),
            bomb_cooldown=jnp.array(0, jnp.int32),
            rng_state=seed,
            rom_entropy_state=jnp.array(0xFF01, jnp.uint32),
            rom_entropy_stars=jnp.array(
                [
                    [240, 240, 224, 192, 128, 0, 0, 0, 0],
                    [255, 255, 255, 255, 255, 255, 127, 63, 30],
                    [255, 63, 31, 15, 7, 3, 1, 0, 0],
                ],
                dtype=jnp.int32,
            ),
        )
        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: StarGunnerState,
        action: chex.Array,
    ) -> tuple[
        StarGunnerObservation,
        StarGunnerState,
        chex.Array,
        chex.Array,
        StarGunnerInfo,
    ]:
        action = jnp.asarray(action, dtype=jnp.int32)
        atari_action = jnp.take(self.ACTION_SET, action)

        fire_pressed = jnp.isin(atari_action, self.FIRE_ACTIONS)
        fire_edge = fire_pressed & ~state.fire_was_pressed
        pressing_up = jnp.isin(atari_action, self.UP_ACTIONS)
        pressing_down = jnp.isin(atari_action, self.DOWN_ACTIONS)
        pressing_right = jnp.isin(atari_action, self.RIGHT_ACTIONS)
        pressing_left = jnp.isin(atari_action, self.LEFT_ACTIONS)

        rng = _xorshift32(state.rng_state)
        new_step_counter = state.step_counter + 1
        gameplay_start = (
            (state.step_counter < self.consts.GAMEPLAY_START_STEP)
            & (new_step_counter >= self.consts.GAMEPLAY_START_STEP)
        )
        started = (
            (new_step_counter >= self.consts.GAMEPLAY_START_STEP)
            & ~state.game_over
        )
        # The internal countdown includes one render-order marker beyond RAM
        # $82: values above one skip gameplay, while value one produces the
        # first active frame and is stored as zero for rendering.
        wave_transition_active = state.wave_transition_age > 1
        gameplay_frame = started & ~wave_transition_active

        player_phase = state.player_phase
        player_animation_age = state.player_animation_age
        player_x = state.player_x
        player_y = state.player_y
        player_facing = state.player_facing

        player_phase = jnp.where(
            gameplay_start,
            PLAYER_ACTIVE,
            player_phase,
        )
        player_animation_age = jnp.where(
            gameplay_start,
            0,
            player_animation_age,
        )

        # Player animatio and revival logic.
        dying_done = (
            ~wave_transition_active
            & (player_phase == PLAYER_DYING)
            & (player_animation_age + 1 >= self.consts.PLAYER_DEATH_DURATION)
        )
        waiting_done = (
            ~wave_transition_active
            & (player_phase == PLAYER_WAITING)
            & (player_animation_age + 1 >= self.consts.PLAYER_WAIT_DURATION)
        )
        player_x = jnp.where(
            waiting_done,
            self.consts.PLAYER_START_X,
            player_x,
        )

        player_y = jnp.where(
            waiting_done,
            self.consts.PLAYER_START_Y,
            player_y,
        )

        player_facing = jnp.where(
            waiting_done,
            1,
            player_facing,
        )
        materialize_done = (
            ~wave_transition_active
            & (player_phase == PLAYER_MATERIALIZING)
            & (
                player_animation_age + 1
                >= self.consts.PLAYER_MATERIALIZE_DURATION
            )
        )
        player_phase = jnp.where(
            dying_done,
            jnp.where(state.lives > 0, PLAYER_WAITING, PLAYER_HIDDEN),
            player_phase,
        )
        player_phase = jnp.where(
            waiting_done,
            PLAYER_MATERIALIZING,
            player_phase,
        )
        player_phase = jnp.where(
            materialize_done,
            PLAYER_ACTIVE,
            player_phase,
        )
        reset_player_age = gameplay_start | dying_done | waiting_done | materialize_done
        player_animation_age = jnp.where(
            reset_player_age,
            0,
            jnp.where(
                (player_phase == PLAYER_DYING)
                | (player_phase == PLAYER_WAITING)
                | (player_phase == PLAYER_MATERIALIZING),
                player_animation_age + 1,
                0,
            ),
        )

        can_move = gameplay_frame & (player_phase == PLAYER_ACTIVE)
        dx = pressing_right.astype(jnp.int32) - pressing_left.astype(jnp.int32)
        dy = pressing_down.astype(jnp.int32) - pressing_up.astype(jnp.int32)
        proposed_x = player_x + dx * self.consts.PLAYER_SPEED_X
        proposed_y = player_y + dy * self.consts.PLAYER_SPEED_Y

        wrap_right = (
            can_move
            & pressing_right
            & (proposed_x >= self.consts.PLAYER_WRAP_RIGHT_THRESHOLD_X)
        )
        wrap_left = (
            can_move
            & pressing_left
            & (proposed_x <= self.consts.PLAYER_WRAP_LEFT_THRESHOLD_X)
        )
        wrapped_x = jnp.where(
            wrap_right,
            self.consts.PLAYER_WRAP_RIGHT_DESTINATION_X,
            jnp.where(
                wrap_left,
                self.consts.PLAYER_WRAP_LEFT_DESTINATION_X,
                proposed_x,
            ),
        )
        player_x = jnp.where(can_move, wrapped_x, player_x)
        player_y = jnp.where(
            can_move,
            jnp.clip(
                proposed_y,
                self.consts.PLAYER_MIN_Y,
                self.consts.PLAYER_MAX_Y,
            ),
            player_y,
        )
        proposed_facing = jnp.where(
            pressing_left,
            -1,
            jnp.where(pressing_right, 1, player_facing),
        )
        player_facing = jnp.where(can_move, proposed_facing, player_facing)

        entropy_player_raw_y = jnp.where(
            state.player_y <= self.consts.PLAYER_MIN_Y,
            23,
            jnp.where(
                state.player_y >= self.consts.PLAYER_MAX_Y,
                149,
                state.player_y - 34,
            ),
        )
        entropy_player_raw_y = jnp.where(
            state.started,
            entropy_player_raw_y,
            80,
        )
        rom_entropy_state, rom_entropy_stars = _rom_entropy_tick(
            state.rom_entropy_state,
            state.rom_entropy_stars,
            entropy_player_raw_y,
            state.player_facing > 0,
        )

        # First wave.
        enemy_phase = state.enemy_phase
        enemy_phase_age = state.enemy_phase_age
        enemy_x = state.enemy_x
        enemy_y = state.enemy_y
        enemy_direction_x = state.enemy_direction_x
        enemy_direction_y = state.enemy_direction_y
        enemy_axis = state.enemy_axis
        enemy_turn_cooldown = state.enemy_turn_cooldown
        enemy_move_accumulator = state.enemy_move_accumulator
        enemy_visual_type = state.enemy_visual_type
        enemy_visual_delay = state.enemy_visual_delay
        enemy_color_counter = state.enemy_color_counter
        enemy_animation_pointer = state.enemy_animation_pointer
        enemy_animation_timer = state.enemy_animation_timer
        enemy_render_slot = state.enemy_render_slot
        enemy_respawn_pending = state.enemy_respawn_pending
        enemy_death_x = state.enemy_death_x
        enemy_death_y = state.enemy_death_y
        enemy_reserve = state.enemy_reserve
        round_index = state.round_index
        wave_index = state.wave_index
        wave_transition_age = state.wave_transition_age

        animate_enemies = gameplay_frame & (player_phase == PLAYER_ACTIVE)
        decremented_animation_timer = enemy_animation_timer - 1
        advance_enemy_animation = animate_enemies & (
            decremented_animation_timer == 0
        )
        animation_type = _enemy_type(round_index)
        animation_start = jnp.array([88, 176, 132], jnp.int32)[
            animation_type
        ]
        animation_end = jnp.array([122, 210, 166], jnp.int32)[
            animation_type
        ]
        next_animation_pointer = enemy_animation_pointer + 11
        next_animation_pointer = jnp.where(
            next_animation_pointer < animation_end,
            next_animation_pointer,
            animation_start,
        )
        enemy_animation_pointer = jnp.where(
            advance_enemy_animation,
            next_animation_pointer,
            enemy_animation_pointer,
        )
        enemy_animation_timer = jnp.where(
            animate_enemies,
            jnp.where(
                advance_enemy_animation,
                3,
                decremented_animation_timer,
            ),
            enemy_animation_timer,
        )
        enemy_render_slot = jnp.where(
            started,
            (enemy_render_slot + 1) % (wave_index + 1),
            enemy_render_slot,
        )

        first_type = _enemy_type(round_index)
        enemy_phase = jnp.where(
            gameplay_start,
            jnp.array([ENEMY_ACTIVE, ENEMY_HIDDEN, ENEMY_HIDDEN], jnp.int32),
            enemy_phase,
        )
        enemy_phase_age = jnp.where(
            gameplay_start,
            jnp.zeros((3,), jnp.int32),
            enemy_phase_age,
        )
        enemy_visual_type = jnp.where(
            gameplay_start,
            jnp.full((3,), first_type, jnp.int32),
            enemy_visual_type,
        )
        enemy_reserve = jnp.where(gameplay_start, 10, enemy_reserve)

        enemy_direction_x = jnp.where(
            gameplay_start,
            jnp.zeros((3,), jnp.int32),
            enemy_direction_x,
        )
        enemy_direction_y = jnp.where(
            gameplay_start,
            jnp.zeros((3,), jnp.int32),
            enemy_direction_y,
        )
        enemy_axis = jnp.where(
            gameplay_start,
            jnp.zeros((3,), jnp.int32),
            enemy_axis,
        )
        enemy_turn_cooldown = jnp.where(
            gameplay_start,
            jnp.zeros((3,), jnp.int32),
            enemy_turn_cooldown,
        )
        update_enemies = (
            gameplay_frame
            & (player_phase == PLAYER_ACTIVE)
        )
        color_random_values = jnp.array(
            [
                rng & jnp.uint32(255),
                (rng >> jnp.uint32(8)) & jnp.uint32(255),
                (rng >> jnp.uint32(16)) & jnp.uint32(255),
            ],
            dtype=jnp.int32,
        )
        randomized_enemy_colors = (
            (color_random_values & 0xF0)
            + 10
            + ((color_random_values >> 7) & 1)
        ) & 0xFF
        enemy_color_counter = jnp.where(
            gameplay_start,
            randomized_enemy_colors,
            enemy_color_counter,
        )
        decremented_enemy_colors = (enemy_color_counter - 1) & 0xFF
        cycled_enemy_colors = jnp.where(
            (decremented_enemy_colors & 15) < 4,
            decremented_enemy_colors | 15,
            decremented_enemy_colors,
        )
        enemy_color_counter = jnp.where(
            update_enemies
            & ~gameplay_start
            & (enemy_phase == ENEMY_ACTIVE),
            cycled_enemy_colors,
            enemy_color_counter,
        )

        spawn_x = (
            jnp.array([20, 20, 120], jnp.int32)
            + (_enemy_type(round_index) == 2).astype(jnp.int32)
        )

        spawn_y = jnp.array(
            [63, 163, 63],
            jnp.int32,
        )

        enemy_x = jnp.where(
            materialize_done,
            spawn_x,
            enemy_x,
        )

        enemy_y = jnp.where(
            materialize_done,
            spawn_y,
            enemy_y,
        )

        enemy_direction_x = jnp.where(
            materialize_done,
            jnp.zeros((3,), jnp.int32),
            enemy_direction_x,
        )

        enemy_direction_y = jnp.where(
            materialize_done,
            jnp.zeros((3,), jnp.int32),
            enemy_direction_y,
        )

        enemy_axis = jnp.where(
            materialize_done,
            jnp.zeros((3,), jnp.int32),
            enemy_axis,
        )

        enemy_turn_cooldown = jnp.where(
            materialize_done,
            jnp.zeros((3,), jnp.int32),
            enemy_turn_cooldown,
        )

        enemy_phase = jnp.where(
            materialize_done
            & (enemy_phase != ENEMY_HIDDEN),
            ENEMY_ACTIVE,
            enemy_phase,
        )

        enemy_phase_age = jnp.where(
            materialize_done,
            jnp.zeros(
                (3,),
                jnp.int32,
            ),
            enemy_phase_age,
        )

        enemy_respawn_pending = jnp.where(
            materialize_done,
            jnp.zeros(
                (3,),
                jnp.bool_,
            ),
            enemy_respawn_pending,
        )

        enemy_death_x = jnp.where(
            materialize_done,
            spawn_x,
            enemy_death_x,
        )

        enemy_death_y = jnp.where(
            materialize_done,
            spawn_y,
            enemy_death_y,
        )

        # Enemy animation.
        death_raw_y = jnp.clip(
            enemy_death_y - 33,
            0,
            255,
        )
        outgoing_duration = jnp.minimum(
            self.consts.ENEMY_DEATH_DURATION - 1,
            jnp.minimum(
                (death_raw_y + 1) // 2,
                (256 - death_raw_y) // 2,
            ),
        )
        outgoing_duration = jnp.maximum(outgoing_duration, 1)

        # Enemy animation.
        death_finished = (
            update_enemies
            & (enemy_phase == ENEMY_DYING)
            & (enemy_phase_age >= outgoing_duration)
        )
        materialize_finished = (
            update_enemies
            & (enemy_phase == ENEMY_MATERIALIZING)
            & (
                enemy_phase_age + 1
                >= self.consts.ENEMY_MATERIALIZE_DURATION
            )
        )
        dying_color_start = (
            update_enemies
            & (enemy_phase == ENEMY_DYING)
            & (enemy_phase_age == 0)
        )
        dying_color_tick = (
            update_enemies
            & (enemy_phase == ENEMY_DYING)
            & (enemy_phase_age > 0)
            & ~death_finished
        )
        materializing_color_tick = (
            update_enemies
            & (enemy_phase == ENEMY_MATERIALIZING)
            & ~materialize_finished
        )
        enemy_color_counter = jnp.where(
            death_finished,
            0,
            jnp.where(
                dying_color_start,
                31,
                jnp.where(
                    dying_color_tick,
                    jnp.maximum(enemy_color_counter - 1, 0),
                    jnp.where(
                        materializing_color_tick,
                        jnp.minimum(enemy_color_counter + 1, 31),
                        enemy_color_counter,
                    ),
                ),
            ),
        )
        enemy_turn_cooldown = jnp.where(
            materialize_finished,
            0,
            enemy_turn_cooldown,
        )
        respawning_enemy = death_finished & enemy_respawn_pending
        enemy_phase = jnp.where(
            death_finished,
            jnp.where(
                enemy_respawn_pending,
                ENEMY_MATERIALIZING,
                ENEMY_HIDDEN,
            ),
            enemy_phase,
        )
        enemy_phase = jnp.where(
            materialize_finished,
            ENEMY_ACTIVE,
            enemy_phase,
        )
        reset_enemy_age = death_finished | materialize_finished | gameplay_start
        enemy_phase_age = jnp.where(
            reset_enemy_age,
            0,
            jnp.where(
                update_enemies
                & (enemy_phase != ENEMY_HIDDEN),
                enemy_phase_age + 1,
                enemy_phase_age,
            ),
        )
        enemy_respawn_pending = jnp.where(
            death_finished,
            False,
            enemy_respawn_pending,
        )

        # Enemy movement.
        level = round_index // 3 + 1
        logical_type = _enemy_type(round_index)
        speed_fixed = _enemy_speed_fixed(level)
        active_enemy = (
            update_enemies
            & (enemy_phase == ENEMY_ACTIVE)
            # The ROM exits the enemy routine immediately after phase FD
            # reaches the target.  Direction selection starts next frame.
            & ~materialize_finished
        )
        speed_integer = speed_fixed >> 8
        speed_fraction = speed_fixed & 0xFF
        fractional_accumulator = jnp.ravel(enemy_move_accumulator)[0]
        raw_x_adjustment = jnp.where(logical_type == 2, 9, 10)
        raw_enemy_x = enemy_x + raw_x_adjustment
        raw_enemy_y = enemy_y - 33

        enemy_rng_state = rom_entropy_state

        for slot in (2, 1, 0):
            slot_respawning = respawning_enemy[slot]
            # Phase FE always runs both respawn RNG calls.  With no reserve,
            # the ROM discards their coordinates during the wave transition.
            slot_respawn_rng_call = death_finished[slot]
            rng_after_respawn_x, respawn_x_random, _ = (
                _rom_random_byte(enemy_rng_state, jnp.uint32(0))
            )
            rng_after_respawn_y, respawn_y_random, _ = _rom_random_byte(
                rng_after_respawn_x,
                jnp.uint32(1),
            )
            enemy_rng_state = jnp.where(
                slot_respawn_rng_call,
                rng_after_respawn_y,
                enemy_rng_state,
            )
            respawn_x = (
                jnp.clip(respawn_x_random.astype(jnp.int32), 80, 140)
                - raw_x_adjustment
            )
            respawn_y = (
                (respawn_y_random.astype(jnp.int32) & 63) + 95
            )
            raw_enemy_x = raw_enemy_x.at[slot].set(
                jnp.where(
                    slot_respawning,
                    respawn_x + raw_x_adjustment,
                    raw_enemy_x[slot],
                )
            )
            raw_enemy_y = raw_enemy_y.at[slot].set(
                jnp.where(
                    slot_respawning,
                    respawn_y - 33,
                    raw_enemy_y[slot],
                )
            )

            slot_active = active_enemy[slot]
            cooldown_before = enemy_turn_cooldown[slot]
            movement_slot = slot_active & (cooldown_before > 0)
            fractional_sum = fractional_accumulator + speed_fraction
            fractional_carry = (fractional_sum >= 256).astype(jnp.int32)
            fractional_accumulator = jnp.where(
                movement_slot,
                fractional_sum & 0xFF,
                fractional_accumulator,
            )
            enemy_turn_cooldown = enemy_turn_cooldown.at[slot].set(
                jnp.where(
                    movement_slot,
                    cooldown_before - 1,
                    cooldown_before,
                )
            )

            direction_y = enemy_direction_y[slot]
            proposed_raw_y = raw_enemy_y[slot] + jnp.where(
                direction_y < 0,
                speed_integer + fractional_carry,
                jnp.where(direction_y > 0, -speed_integer, 0),
            )
            vertical_invalid = movement_slot & (
                ((direction_y > 0) & (proposed_raw_y < 23))
                | ((direction_y < 0) & (proposed_raw_y >= 140))
            )
            raw_enemy_y = raw_enemy_y.at[slot].set(
                jnp.where(
                    movement_slot & ~vertical_invalid,
                    proposed_raw_y,
                    raw_enemy_y[slot],
                )
            )

            direction_x = enemy_direction_x[slot]
            proposed_raw_x = raw_enemy_x[slot] + jnp.where(
                direction_x > 0,
                speed_integer + fractional_carry,
                jnp.where(direction_x < 0, -speed_integer, 0),
            )
            wrapped_raw_x = jnp.where(
                (direction_x > 0) & (proposed_raw_x >= 145),
                20,
                jnp.where(
                    (direction_x < 0) & (proposed_raw_x < 20),
                    145,
                    proposed_raw_x,
                ),
            )
            raw_enemy_x = raw_enemy_x.at[slot].set(
                jnp.where(
                    movement_slot & ~vertical_invalid,
                    wrapped_raw_x,
                    raw_enemy_x[slot],
                )
            )

            reinitialize_direction = slot_active & (
                (cooldown_before == 0) | vertical_invalid
            )
            initial_rng_carry = jnp.where(
                cooldown_before == 0,
                (decremented_enemy_colors[slot] & 15) >= 4,
                direction_y < 0,
            ).astype(jnp.uint32)
            rng_after_y, direction_y_random, rng_carry_y = (
                _rom_random_byte(enemy_rng_state, initial_rng_carry)
            )
            rng_after_x, direction_x_random, rng_carry_x = (
                _rom_random_byte(rng_after_y, rng_carry_y)
            )
            rng_after_meta, direction_meta_random, rng_carry_meta = (
                _rom_random_byte(rng_after_x, rng_carry_x)
            )
            enemy_rng_state = jnp.where(
                reinitialize_direction,
                rng_after_meta,
                enemy_rng_state,
            )
            masked_direction_x = direction_x_random.astype(jnp.int32) & 0x87
            masked_direction_y = direction_y_random.astype(jnp.int32) & 0x87
            next_direction_x = jnp.where(
                masked_direction_x == 0,
                0,
                jnp.where((masked_direction_x & 0x80) != 0, -1, 1),
            )
            next_direction_y = jnp.where(
                masked_direction_y == 0,
                0,
                jnp.where((masked_direction_y & 0x80) != 0, -1, 1),
            )
            next_turn_cooldown = jnp.where(
                direction_meta_random < jnp.uint32(20),
                direction_meta_random + jnp.uint32(20),
                direction_meta_random,
            )
            next_enemy_color = (
                (direction_meta_random & jnp.uint32(0xF0))
                + jnp.uint32(10)
                + rng_carry_meta
            ) & jnp.uint32(255)
            enemy_direction_x = enemy_direction_x.at[slot].set(
                jnp.where(
                    reinitialize_direction,
                    next_direction_x,
                    enemy_direction_x[slot],
                )
            )
            enemy_direction_y = enemy_direction_y.at[slot].set(
                jnp.where(
                    reinitialize_direction,
                    next_direction_y,
                    enemy_direction_y[slot],
                )
            )
            enemy_turn_cooldown = enemy_turn_cooldown.at[slot].set(
                jnp.where(
                    reinitialize_direction,
                    next_turn_cooldown,
                    enemy_turn_cooldown[slot],
                )
            )
            enemy_color_counter = enemy_color_counter.at[slot].set(
                jnp.where(
                    reinitialize_direction,
                    next_enemy_color.astype(jnp.int32),
                    enemy_color_counter[slot],
                )
            )

        rom_entropy_state = enemy_rng_state

        enemy_x = raw_enemy_x - raw_x_adjustment
        enemy_y = raw_enemy_y + 33
        enemy_move_accumulator = jnp.full(
            (3,),
            fractional_accumulator,
            dtype=jnp.int32,
        )

        # Bobo movement.
        bobo_x = state.bobo_x
        bobo_direction = state.bobo_direction
        bomb_cooldown = state.bomb_cooldown
        proposed_bobo_x = bobo_x + bobo_direction * self.consts.BOBO_SPEED
        bobo_boundary = (
            (proposed_bobo_x < self.consts.BOBO_MIN_X)
            | (proposed_bobo_x >= self.consts.BOBO_MAX_X)
        )
        bobo_initialize = (
            gameplay_frame
            & (player_phase == PLAYER_ACTIVE)
            & (new_step_counter == 285)
        )
        bobo_can_move = (
            gameplay_frame
            & (player_phase == PLAYER_ACTIVE)
            & (new_step_counter > 285)
        )
        decremented_bobo_cooldown = (bomb_cooldown - 1) & 0xFF
        bobo_timer_expired = decremented_bobo_cooldown == 0
        bobo_turn = bobo_can_move & (
            bobo_timer_expired | bobo_boundary
        )
        bobo_rng_carry = jnp.where(
            bobo_initialize | bobo_timer_expired,
            True,
            bobo_direction > 0,
        ).astype(jnp.uint32)
        bobo_rng_state, bobo_random, _ = _rom_random_byte(
            rom_entropy_state,
            bobo_rng_carry,
        )
        bobo_rng_call = bobo_initialize | bobo_turn
        rom_entropy_state = jnp.where(
            bobo_rng_call,
            bobo_rng_state,
            rom_entropy_state,
        )
        bobo_direction = jnp.where(
            bobo_turn,
            -bobo_direction,
            bobo_direction,
        )
        bobo_x = jnp.where(
            bobo_can_move,
            jnp.where(
                bobo_turn,
                bobo_x,
                jnp.clip(
                    proposed_bobo_x,
                    self.consts.BOBO_MIN_X,
                    self.consts.BOBO_MAX_X,
                ),
            ),
            bobo_x,
        )
        bomb_cooldown = jnp.where(
            bobo_rng_call,
            bobo_random.astype(jnp.int32) & 31,
            jnp.where(
                bobo_can_move,
                decremented_bobo_cooldown,
                bomb_cooldown,
            ),
        )

        # Bobo bomb.
        # Bobo bomb.
        bomb_active = state.bomb_active
        bomb_x = state.bomb_x
        bomb_y = state.bomb_y
        bomb_age = state.bomb_age
        bomb_can_move = (
            gameplay_frame
            & (player_phase == PLAYER_ACTIVE)
        )

        bomb_speed = 2 + level // 4

        moved_bomb_y = (
            bomb_y
            + bomb_speed
        )

        bomb_finished = (
            bomb_can_move
            & bomb_active
            & (
                moved_bomb_y
                > self.consts.BOMB_MAX_Y
            )
        )

        spawn_bomb = (
            bomb_can_move
            & (new_step_counter >= 286)
            & (
                ~bomb_active
                | bomb_finished
            )
        )

        bomb_active = jnp.where(
            spawn_bomb,
            True,
            jnp.where(
                bomb_finished,
                False,
                bomb_active,
            ),
        )

        bomb_x = jnp.where(
            spawn_bomb,
            bobo_x - 2,
            bomb_x,
        )

        bomb_y = jnp.where(
            spawn_bomb,
            self.consts.BOMB_START_Y,
            jnp.where(
                bomb_can_move
                & bomb_active
                & ~bomb_finished,
                moved_bomb_y,
                bomb_y,
            ),
        )

        bomb_age = jnp.where(
            spawn_bomb,
            0,
            jnp.where(
                bomb_can_move
                & bomb_active
                & ~bomb_finished,
                bomb_age + 1,
                bomb_age,
            ),
        )

        bomb_collision_start_y = jnp.where(
            spawn_bomb,
            bomb_y,
            state.bomb_y,
        )

        bomb_collision_end_y = bomb_y

        # Missile movement.
        moved_missile_x = (
            state.missile_x
            + state.missile_direction * self.consts.MISSILE_SPEED
        )
        moved_missile_inside = jnp.where(
            state.missile_direction == 1,
            moved_missile_x <= self.consts.MISSILE_MAX_X_RIGHT,
            moved_missile_x >= self.consts.MISSILE_MIN_X_LEFT,
        )
        missile_inside = wave_transition_active | moved_missile_inside
        missile_active = state.missile_active & missile_inside
        missile_x = jnp.where(
            wave_transition_active,
            state.missile_x,
            jnp.where(missile_active, moved_missile_x, 0),
        )
        missile_y = jnp.where(missile_active, state.missile_y, 0)
        missile_direction = jnp.where(
            missile_active,
            state.missile_direction,
            0,
        )
        missile_age = jnp.where(
            missile_active,
            state.missile_age + (~wave_transition_active).astype(jnp.int32),
            0,
        )

        spawn_missile = (
            gameplay_frame
            & (player_phase == PLAYER_ACTIVE)
            & fire_edge
            & ~state.missile_active
        )
        missile_offset_x = jnp.where(
            player_facing == 1,
            self.consts.MISSILE_SPAWN_OFFSET_X_RIGHT,
            self.consts.MISSILE_SPAWN_OFFSET_X_LEFT,
        )
        missile_active = missile_active | spawn_missile
        missile_x = jnp.where(
            spawn_missile,
            player_x + missile_offset_x,
            missile_x,
        )
        missile_y = jnp.where(
            spawn_missile,
            player_y + self.consts.MISSILE_SPAWN_OFFSET_Y,
            missile_y,
        )
        missile_direction = jnp.where(
            spawn_missile,
            player_facing,
            missile_direction,
        )
        missile_age = jnp.where(spawn_missile, 0, missile_age)

        # Missile collision.
        enemy_width = jnp.where(logical_type == 2, 7, 8)
        enemy_height = jnp.where(logical_type == 2, 8, 10)
        missile_left = missile_x
        # ALE's TIA collision latch includes the color-clock immediately to
        # the right of the eight visible missile pixels.
        missile_right = missile_x + self.consts.MISSILE_WIDTH
        collision_enemy = (
            gameplay_frame
            & missile_active
            & (enemy_phase == ENEMY_ACTIVE)
            & (missile_right >= enemy_x)
            & (missile_left <= enemy_x + enemy_width - 1)
            & (missile_y >= enemy_y + 2)
            & (missile_y <= enemy_y + enemy_height - 1)
        )
        enemy_hit = jnp.any(collision_enemy)
        hit_slot = jnp.argmax(collision_enemy.astype(jnp.int32))
        hit_mask = jnp.arange(3, dtype=jnp.int32) == hit_slot
        hit_mask = hit_mask & enemy_hit
        can_respawn = enemy_reserve > 0
        enemy_phase = jnp.where(hit_mask, ENEMY_DYING, enemy_phase)
        enemy_phase_age = jnp.where(hit_mask, 0, enemy_phase_age)
        enemy_death_x = jnp.where(hit_mask, enemy_x, enemy_death_x)
        enemy_death_y = jnp.where(hit_mask, enemy_y, enemy_death_y)
        enemy_respawn_pending = jnp.where(
            hit_mask,
            can_respawn,
            enemy_respawn_pending,
        )
        enemy_reserve = jnp.where(
            enemy_hit & can_respawn,
            enemy_reserve - 1,
            enemy_reserve,
        )
        missile_active = missile_active & ~enemy_hit
        missile_x = jnp.where(missile_active, missile_x, 0)
        missile_y = jnp.where(missile_active, missile_y, 0)
        missile_direction = jnp.where(
            missile_active,
            missile_direction,
            0,
        )
        missile_age = jnp.where(missile_active, missile_age, 0)

        kill_reward = (wave_index + 1) * 100
        score = state.score + jnp.where(enemy_hit, kill_reward, 0)

        # Player collision.
        player_hit_x = jnp.where(
            player_facing == 1,
            player_x + 4,
            player_x + 3,
        )
        player_hit_y = player_y + 3
        enemy_can_hit = (
            (enemy_phase == ENEMY_ACTIVE)
            | (
                (enemy_phase == ENEMY_MATERIALIZING)
                & (
                    enemy_phase_age
                    >= self.consts.ENEMY_MATERIALIZE_DURATION - 4
                )
            )
        )
        enemy_player_collision = (
            enemy_can_hit
            & (player_hit_x >= enemy_x)
            & (player_hit_x <= enemy_x + enemy_width - 1)
            & (player_hit_y >= enemy_y + 2)
            & (player_hit_y <= enemy_y + enemy_height - 1)
        )
        bomb_collision_top = jnp.minimum(
            bomb_collision_start_y,
            bomb_collision_end_y,
        )

        bomb_collision_bottom = jnp.maximum(
            bomb_collision_start_y,
            bomb_collision_end_y,
        )

        bomb_player_collision = (
            bomb_active
            & (player_hit_x >= bomb_x)
            & (
                player_hit_x
                < bomb_x + self.consts.BOMB_WIDTH
            )
            & (
                player_hit_y
                >= bomb_collision_top
            )
            & (
                player_hit_y
                <= bomb_collision_bottom
            )
        )
        enemy_collision_enabled = (
            gameplay_frame & (player_phase == PLAYER_ACTIVE)
        )
        for slot in (2, 1, 0):
            slot_collision = (
                enemy_collision_enabled & enemy_player_collision[slot]
            )
            collision_rng_state, collision_random, collision_rng_carry = (
                _rom_random_byte(rom_entropy_state, jnp.uint32(0))
            )
            rom_entropy_state = jnp.where(
                slot_collision,
                collision_rng_state,
                rom_entropy_state,
            )
            collision_raw_position = (
                (collision_random.astype(jnp.int32) & 63)
                + 20
                + collision_rng_carry.astype(jnp.int32)
            )
            enemy_x = enemy_x.at[slot].set(
                jnp.where(
                    slot_collision,
                    collision_raw_position - raw_x_adjustment,
                    enemy_x[slot],
                )
            )
            enemy_y = enemy_y.at[slot].set(
                jnp.where(
                    slot_collision,
                    collision_raw_position + 33,
                    enemy_y[slot],
                )
            )
        player_hit = (
            gameplay_frame
            & (player_phase == PLAYER_ACTIVE)
            & (jnp.any(enemy_player_collision) | bomb_player_collision)
        )
        lives = state.lives - player_hit.astype(jnp.int32)
        player_phase = jnp.where(player_hit, PLAYER_DYING, player_phase)
        player_animation_age = jnp.where(player_hit, 0, player_animation_age)
        missile_active = jnp.where(player_hit, False, missile_active)

        # Wave transition.
        active_or_animation = enemy_phase != ENEMY_HIDDEN
        wave_cleared = (
            gameplay_frame
            & (player_phase == PLAYER_ACTIVE)
            & (enemy_reserve == 0)
            & ~jnp.any(active_or_animation)
        )
        score = score + jnp.where(
            wave_cleared,
            _wave_bonus(level, wave_index),
            0,
        )
        start_next_wave = wave_cleared
        next_round = round_index + 1
        next_wave = next_round % 3
        next_type = _enemy_type(next_round)
        if not self.consts.EMULATE_TRANSIENT_WRONG_ENEMY_SPRITE:
            enemy_animation_pointer = jnp.where(
                start_next_wave,
                jnp.array([88, 176, 132], jnp.int32)[next_type],
                enemy_animation_pointer,
            )
        next_active_count = next_wave + 1
        spawn_x = (
            jnp.array([20, 20, 120], jnp.int32)
            + (next_type == 2).astype(jnp.int32)
        )
        spawn_y = jnp.array([63, 163, 63], jnp.int32)
        slot_ids = jnp.arange(3, dtype=jnp.int32)
        new_slot_mask = slot_ids == next_wave
        reset_position_mask = start_next_wave & (
            slot_ids < next_active_count
        )
        next_phase = jnp.where(
            slot_ids < next_active_count,
            ENEMY_ACTIVE,
            ENEMY_HIDDEN,
        )
        round_index = jnp.where(start_next_wave, next_round, round_index)
        wave_index = jnp.where(start_next_wave, next_wave, wave_index)
        wave_transition_age = jnp.where(
            start_next_wave,
            # ALE exposes RAM $82=30 here; the extra internal count preserves
            # the fact that its displayed zero frame is still not rendered.
            self.consts.WAVE_TRANSITION_DURATION - 1,
            jnp.maximum(wave_transition_age - 1, 0),
        )
        enemy_phase = jnp.where(start_next_wave, next_phase, enemy_phase)
        enemy_phase_age = jnp.where(
            start_next_wave,
            jnp.zeros((3,), jnp.int32),
            enemy_phase_age,
        )
        enemy_x = jnp.where(reset_position_mask, spawn_x, enemy_x)
        enemy_y = jnp.where(reset_position_mask, spawn_y, enemy_y)
        enemy_direction_x = jnp.where(
            start_next_wave & (next_wave > 0) & new_slot_mask,
            0,
            enemy_direction_x,
        )
        enemy_direction_y = jnp.where(
            start_next_wave & (next_wave > 0) & new_slot_mask,
            0,
            enemy_direction_y,
        )
        enemy_axis = jnp.where(
            start_next_wave & (next_wave > 0) & new_slot_mask,
            0,
            enemy_axis,
        )
        enemy_turn_cooldown = jnp.where(
            start_next_wave & (next_wave > 0) & new_slot_mask,
            0,
            enemy_turn_cooldown,
        )
        enemy_color_counter = jnp.where(
            start_next_wave & (next_wave > 0) & new_slot_mask,
            1,
            enemy_color_counter,
        )
        # The transient wrong active sprite comes from the shared animation
        # pointer, not from a per-slot delay.  Particle geometry follows the
        # logical round type immediately.
        enemy_visual_delay = jnp.where(
            start_next_wave,
            jnp.zeros((3,), jnp.int32),
            enemy_visual_delay,
        )
        enemy_visual_type = jnp.where(
            start_next_wave,
            jnp.full((3,), next_type, jnp.int32),
            enemy_visual_type,
        )
        enemy_respawn_pending = jnp.where(
            start_next_wave,
            jnp.zeros((3,), jnp.bool_),
            enemy_respawn_pending,
        )
        enemy_death_x = jnp.where(start_next_wave, spawn_x, enemy_death_x)
        enemy_death_y = jnp.where(start_next_wave, spawn_y, enemy_death_y)
        enemy_reserve = jnp.where(
            start_next_wave,
            (next_wave + 1) * 10,
            enemy_reserve,
        )

        # Extra life every 10000 points.
        extra_lives = score // 10000 - state.score // 10000
        lives = jnp.minimum(
            lives + jnp.maximum(extra_lives, 0),
            self.consts.MAX_LIVES,
        )

        game_over = state.game_over | (
            (lives <= 0)
            & (player_phase == PLAYER_HIDDEN)
            & started
        )
        started = started & ~game_over

        new_state = state.replace(
            player_x=player_x.astype(jnp.int32),
            player_y=player_y.astype(jnp.int32),
            player_facing=player_facing.astype(jnp.int32),
            player_phase=player_phase.astype(jnp.int32),
            player_animation_age=player_animation_age.astype(jnp.int32),
            started=started.astype(jnp.bool_),
            game_over=game_over.astype(jnp.bool_),
            score=score.astype(jnp.int32),
            lives=lives.astype(jnp.int32),
            step_counter=new_step_counter.astype(jnp.int32),
            fire_was_pressed=jnp.where(
                wave_transition_active,
                state.fire_was_pressed,
                fire_pressed,
            ).astype(jnp.bool_),
            missile_active=missile_active.astype(jnp.bool_),
            missile_x=missile_x.astype(jnp.int32),
            missile_y=missile_y.astype(jnp.int32),
            missile_direction=missile_direction.astype(jnp.int32),
            missile_age=missile_age.astype(jnp.int32),
            round_index=round_index.astype(jnp.int32),
            wave_index=wave_index.astype(jnp.int32),
            wave_transition_age=wave_transition_age.astype(jnp.int32),
            enemy_reserve=enemy_reserve.astype(jnp.int32),
            enemy_x=enemy_x.astype(jnp.int32),
            enemy_y=enemy_y.astype(jnp.int32),
            enemy_direction_x=enemy_direction_x.astype(jnp.int32),
            enemy_direction_y=enemy_direction_y.astype(jnp.int32),
            enemy_axis=enemy_axis.astype(jnp.int32),
            enemy_turn_cooldown=enemy_turn_cooldown.astype(jnp.int32),
            enemy_move_accumulator=enemy_move_accumulator.astype(jnp.int32),
            enemy_phase=enemy_phase.astype(jnp.int32),
            enemy_phase_age=enemy_phase_age.astype(jnp.int32),
            enemy_visual_type=enemy_visual_type.astype(jnp.int32),
            enemy_visual_delay=enemy_visual_delay.astype(jnp.int32),
            enemy_color_counter=enemy_color_counter.astype(jnp.int32),
            enemy_animation_pointer=enemy_animation_pointer.astype(jnp.int32),
            enemy_animation_timer=enemy_animation_timer.astype(jnp.int32),
            enemy_render_slot=enemy_render_slot.astype(jnp.int32),
            enemy_respawn_pending=enemy_respawn_pending.astype(jnp.bool_),
            enemy_death_x=enemy_death_x.astype(jnp.int32),
            enemy_death_y=enemy_death_y.astype(jnp.int32),
            bobo_x=bobo_x.astype(jnp.int32),
            bobo_direction=bobo_direction.astype(jnp.int32),
            bomb_active=bomb_active.astype(jnp.bool_),
            bomb_x=bomb_x.astype(jnp.int32),
            bomb_y=bomb_y.astype(jnp.int32),
            bomb_age=bomb_age.astype(jnp.int32),
            bomb_cooldown=bomb_cooldown.astype(jnp.int32),
            rng_state=rng.astype(jnp.uint32),
            rom_entropy_state=rom_entropy_state.astype(jnp.uint32),
            rom_entropy_stars=rom_entropy_stars.astype(jnp.int32),
        )

        observation = self._get_observation(new_state)
        reward = self._get_reward(state, new_state)
        done = self._get_done(new_state)
        info = self._get_info(new_state)
        return observation, new_state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(
        self,
        state: StarGunnerState,
    ) -> StarGunnerObservation:
        orientation = jnp.where(
            state.player_facing == 1,
            90.0,
            270.0,
        ).astype(jnp.float32)
        player = ObjectObservation.create(
            x=state.player_x.astype(jnp.int32),
            y=state.player_y.astype(jnp.int32),
            width=jnp.array(self.consts.PLAYER_WIDTH, jnp.int32),
            height=jnp.array(self.consts.PLAYER_HEIGHT, jnp.int32),
            active=(state.player_phase == PLAYER_ACTIVE).astype(jnp.int32),
            orientation=orientation,
        )
        return StarGunnerObservation(
            player=player,
            score=state.score.astype(jnp.uint32),
            lives=state.lives.astype(jnp.uint8),
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(
        self,
        previous_state: StarGunnerState,
        state: StarGunnerState,
    ) -> chex.Array:
        return (state.score - previous_state.score).astype(jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(
        self,
        state: StarGunnerState,
    ) -> chex.Array:
        return state.game_over.astype(jnp.bool_)

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(
        self,
        state: StarGunnerState,
        all_rewards: chex.Array | None = None,
    ) -> StarGunnerInfo:
        del all_rewards
        enemies_remaining = (
            state.enemy_reserve
            + jnp.sum((state.enemy_phase != ENEMY_HIDDEN).astype(jnp.int32))
        )
        return StarGunnerInfo(
            step_counter=state.step_counter,
            started=state.started,
            level=state.round_index // 3 + 1,
            wave=state.wave_index + 1,
            enemies_remaining=enemies_remaining,
        )

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTION_SET))

    def observation_space(self) -> spaces.Dict:
        player_space = spaces.Dict(
            {
                "x": spaces.Box(
                    low=self.consts.PLAYER_MIN_X,
                    high=self.consts.PLAYER_MAX_X,
                    shape=(),
                    dtype=jnp.int32,
                ),
                "y": spaces.Box(
                    low=self.consts.PLAYER_MIN_Y,
                    high=self.consts.PLAYER_MAX_Y,
                    shape=(),
                    dtype=jnp.int32,
                ),
                "width": spaces.Box(
                    low=0,
                    high=self.consts.WIDTH,
                    shape=(),
                    dtype=jnp.int32,
                ),
                "height": spaces.Box(
                    low=0,
                    high=self.consts.HEIGHT,
                    shape=(),
                    dtype=jnp.int32,
                ),
                "active": spaces.Box(
                    low=0,
                    high=1,
                    shape=(),
                    dtype=jnp.int32,
                ),
                "visual_id": spaces.Box(
                    low=0,
                    high=255,
                    shape=(),
                    dtype=jnp.int32,
                ),
                "state": spaces.Box(
                    low=0,
                    high=255,
                    shape=(),
                    dtype=jnp.int32,
                ),
                "orientation": spaces.Box(
                    low=0.0,
                    high=360.0,
                    shape=(),
                    dtype=jnp.float32,
                ),
            }
        )
        return spaces.Dict(
            {
                "player": player_space,
                "score": spaces.Box(
                    low=0,
                    high=999999,
                    shape=(),
                    dtype=jnp.uint32,
                ),
                "lives": spaces.Box(
                    low=0,
                    high=255,
                    shape=(),
                    dtype=jnp.uint8,
                ),
            }
        )

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.consts.HEIGHT, self.consts.WIDTH, 3),
            dtype=jnp.uint8,
        )

    @partial(jax.jit, static_argnums=(0,))
    def render(
        self,
        state: StarGunnerState,
    ) -> jnp.ndarray:
        return self.renderer.render(state)
