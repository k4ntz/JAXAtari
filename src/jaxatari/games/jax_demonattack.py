import os
from functools import partial
from typing import Tuple

import numpy as np
import chex
import jax.lax
import jax.numpy as jnp
from flax import struct

import jaxatari.spaces as spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action, ObjectObservation
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.modification import AutoDerivedConstants

INITIAL_WAVE_PATTERNS = 12
REPEATING_WAVE_PATTERN_START = 8
PATTERNS_PER_DIFFICULTY_ENTRY = 2
DEMON_STATUS_FREE = 0
DEMON_STATUS_SPAWNING = 1
DEMON_STATUS_NORMAL = 2
BOMB_TYPE_STANDARD = 0
BOMB_TYPE_LONG = 1
DIFFICULTY_TABLE_NAMES = (
    "ENEMY_SHOT_SPEED_TABLE",
    "WAVE_LASER_SPEED_TABLE",
)

def _create_digit_sprites(consts: "DemonAttackConstants") -> jnp.ndarray:
    digits = np.zeros((10, 8, 8, 4), dtype=np.uint8)
    color = np.array((*consts.SCORE_COLOR, 255), dtype=np.uint8)

    patterns = [
        [[1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1]],
        [[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]],
        [[1, 1, 1], [0, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1]],
        [[1, 1, 1], [0, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1]],
        [[1, 0, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1]],
        [[1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 1], [1, 1, 1]],
        [[1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 0, 1], [1, 1, 1]],
        [[1, 1, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]],
        [[1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 1, 1]],
        [[1, 1, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1]],
    ]

    for i, pattern in enumerate(patterns):
        for r, row in enumerate(pattern):
            for c, val in enumerate(row):
                if val:
                    digits[i, r + 1, c + 2] = color

    return jnp.array(digits)

def _get_default_asset_config() -> tuple:
    """
    Does not contain procedural assets. Those are generated in the init of the renderer.
    """
    return (
        {'name': 'background', 'type': 'background', 'file': 'Background.npy'},
        {'name': 'player', 'type': 'single', 'file': 'Player.npy'},
        {'name': 'player_missile', 'type': 'single', 'file': 'PlayerMissile.npy'},
        {'name': 'projectile_demon', 'type': 'single', 'file': 'Bomb_1.npy'},
        {'name': 'demon_1', 'type': 'group', 'files': [
            'Enemy_1/Enemy_1.npy',
            'Enemy_1/Enemy_2.npy',
            'Enemy_1/Enemy_3.npy',
            'Enemy_1/Enemy_4.npy',
        ]},
        {'name': 'demon_2', 'type': 'group', 'files': [
            'Enemy_2/Enemy_1.npy',
            'Enemy_2/Enemy_2.npy',
            'Enemy_2/Enemy_3.npy',
            'Enemy_2/Enemy_4.npy',
        ]},
        {'name': 'demon_3', 'type': 'group', 'files': [
            'Enemy_3/Enemy_1.npy',
            'Enemy_3/Enemy_2.npy',
            'Enemy_3/Enemy_3.npy',
            'Enemy_3/Enemy_4.npy',
        ]},
        {'name': 'demon_4', 'type': 'group', 'files': [
            'Enemy_4/Enemy_1.npy',
            'Enemy_4/Enemy_2.npy',
            'Enemy_4/Enemy_3.npy',
            'Enemy_4/Enemy_4.npy',
        ]},
        {'name': 'demon_5', 'type': 'group', 'files': [
            'Enemy_5/Enemy_1.npy',
            'Enemy_5/Enemy_2.npy',
            'Enemy_5/Enemy_3.npy',
            'Enemy_5/Enemy_4.npy',
        ]},
        {'name': 'demon_6', 'type': 'group', 'files': [
            'Enemy_6/Enemy_1.npy',
            'Enemy_6/Enemy_2.npy',
            'Enemy_6/Enemy_3.npy',
            'Enemy_6/Enemy_4.npy',
        ]},
        {'name': 'demon_7', 'type': 'group', 'files': [
            'Enemy_7/Enemy_1.npy',
            'Enemy_7/Enemy_2.npy',
            'Enemy_7/Enemy_3.npy',
            'Enemy_7/Enemy_4.npy',
        ]},
        {'name': 'demon_8', 'type': 'group', 'files': [
            'Enemy_8/Enemy_1.npy',
            'Enemy_8/Enemy_2.npy',
            'Enemy_8/Enemy_3.npy',
            'Enemy_8/Enemy_4.npy',
        ]},
        {'name': 'demon_9', 'type': 'group', 'files': [
            'Enemy_9/Enemy_1.npy',
            'Enemy_9/Enemy_2.npy',
            'Enemy_9/Enemy_3.npy',
            'Enemy_9/Enemy_4.npy',
        ]},
        {'name': 'demon_10', 'type': 'group', 'files': [
            'Enemy_10/Enemy_1.npy',
            'Enemy_10/Enemy_2.npy',
            'Enemy_10/Enemy_3.npy',
            'Enemy_10/Enemy_4.npy',
        ]},
        {'name': 'demon_11', 'type': 'group', 'files': [
            'Enemy_11/Enemy_1.npy',
            'Enemy_11/Enemy_2.npy',
            'Enemy_11/Enemy_3.npy',
            'Enemy_11/Enemy_4.npy',
        ]},
        {'name': 'demon_12', 'type': 'group', 'files': [
            'Enemy_12/Enemy_1.npy',
            'Enemy_12/Enemy_2.npy',
            'Enemy_12/Enemy_3.npy',
            'Enemy_12/Enemy_4.npy',
        ]},
        {'name': 'small_demon_5', 'type': 'group', 'files': [
            'Enemy_Small_5/Enemy_1.npy',
            'Enemy_Small_5/Enemy_2.npy',
            'Enemy_Small_5/Enemy_3.npy',
            'Enemy_Small_5/Enemy_4.npy',
        ]},
        {'name': 'small_demon_6', 'type': 'group', 'files': [
            'Enemy_Small_6/Enemy_1.npy',
            'Enemy_Small_6/Enemy_2.npy',
            'Enemy_Small_6/Enemy_3.npy',
            'Enemy_Small_6/Enemy_4.npy',
        ]},
        {'name': 'small_demon_7', 'type': 'group', 'files': [
            'Enemy_Small_7/Enemy_1.npy',
            'Enemy_Small_7/Enemy_2.npy',
            'Enemy_Small_7/Enemy_3.npy',
            'Enemy_Small_7/Enemy_4.npy',
        ]},
        {'name': 'small_demon_8', 'type': 'group', 'files': [
            'Enemy_Small_8/Enemy_1.npy',
            'Enemy_Small_8/Enemy_2.npy',
            'Enemy_Small_8/Enemy_3.npy',
            'Enemy_Small_8/Enemy_4.npy',
        ]},
        {'name': 'small_demon_9', 'type': 'group', 'files': [
            'Enemy_Small_9/Enemy_1.npy',
            'Enemy_Small_9/Enemy_2.npy',
            'Enemy_Small_9/Enemy_3.npy',
            'Enemy_Small_9/Enemy_4.npy',
        ]},
        {'name': 'small_demon_10', 'type': 'group', 'files': [
            'Enemy_Small_10/Enemy_1.npy',
            'Enemy_Small_10/Enemy_2.npy',
            'Enemy_Small_10/Enemy_3.npy',
            'Enemy_Small_10/Enemy_4.npy',
        ]},
        {'name': 'small_demon_11', 'type': 'group', 'files': [
            'Enemy_Small_11/Enemy_1.npy',
            'Enemy_Small_11/Enemy_2.npy',
            'Enemy_Small_11/Enemy_3.npy',
            'Enemy_Small_11/Enemy_4.npy',
        ]},
        {'name': 'small_demon_12', 'type': 'group', 'files': [
            'Enemy_Small_12/Enemy_1.npy',
            'Enemy_Small_12/Enemy_2.npy',
            'Enemy_Small_12/Enemy_3.npy',
            'Enemy_Small_12/Enemy_4.npy',
        ]},
        {'name': 'enemy_spawn_left', 'type': 'group', 'files': [
            'EnemySpawnAnimation/EnemySpawn_left_1.npy',
            'EnemySpawnAnimation/EnemySpawn_left_2.npy',
            'EnemySpawnAnimation/EnemySpawn_left_3.npy',
        ]},
        {'name': 'enemy_spawn_right', 'type': 'group', 'files': [
            'EnemySpawnAnimation/EnemySpawn_right_1.npy',
            'EnemySpawnAnimation/EnemySpawn_right_2.npy',
            'EnemySpawnAnimation/EnemySpawn_right_3.npy',
        ]},
        {'name': 'enemy_death_animation', 'type': 'group', 'files': [
            'EnemyDeathAnimation/EnemyPart_0.npy',
            'EnemyDeathAnimation/EnemyPart_1.npy',
            'EnemyDeathAnimation/EnemyPart_2.npy',
        ]},
        {'name': 'player_death_animation', 'type': 'group', 'files': [
            'PlayerDeathAnimation/Explode_1.npy',
            'PlayerDeathAnimation/Explode_2.npy',
            'PlayerDeathAnimation/Explode_3.npy',
            'PlayerDeathAnimation/Explode_4.npy',
            'PlayerDeathAnimation/Explode_5.npy',
            'PlayerDeathAnimation/Explode_6.npy',
            'PlayerDeathAnimation/Explode_7.npy',
        ]},
        {'name': 'bunker', 'type': 'single', 'file': 'Bunker.npy'},
    )

def _bomb_visible_repeat_window(state, consts, bomb_type):
    """Return visible repeat count and leading-repeat offset for enemy shots."""
    source_y = state.demons_y[state.bomb_source_idx] + consts.DEMON_SIZE[0]
    fallen_repeats = (state.bomb_y - source_y) // consts.BOMB_SIZE[0]
    visible_repeats = jnp.where(
        bomb_type == BOMB_TYPE_LONG,
        jnp.clip(
            fallen_repeats + 1,
            1,
            consts.LONG_BOMB_HEIGHT_MULTIPLIER,
        ),
        1,
    )
    repeat_offset = jnp.where(
        bomb_type == BOMB_TYPE_LONG,
        jnp.clip(
            fallen_repeats,
            0,
            consts.LONG_BOMB_HEIGHT_MULTIPLIER - 1,
        ),
        0,
    )
    return visible_repeats, repeat_offset

class DemonAttackConstants(AutoDerivedConstants):
    # Static Configuration
    WIDTH: int = struct.field(pytree_node=False, default=160)
    HEIGHT: int = struct.field(pytree_node=False, default=192)
    PLAYER_SPEED: int = struct.field(pytree_node=False, default=1)
    MAX_DEMONS: int = struct.field(pytree_node=False, default=3)
    RESPAWN_DELAY: int = struct.field(pytree_node=False, default=30)
    SPAWN_ANIM_FRAMES: int = struct.field(pytree_node=False, default=3)
    SPAWN_ANIM_FRAME_DURATION: int = struct.field(pytree_node=False, default=6)
    SPAWN_MOVE_PAUSE: int = struct.field(pytree_node=False, default=14)
    SPAWN_ANIM_WIDTH: int = struct.field(pytree_node=False, default=32)
    DEMON_DEATH_ANIMATION_DURATION: int = struct.field(pytree_node=False, default=18)
    WAVE_TOTAL_DEMONS: int = struct.field(pytree_node=False, default=8)
    DEMON_TELEPORT_DURATION: int = struct.field(pytree_node=False, default=44)
    DEMON_VERTICAL_MOTION_TABLE: Tuple[int, ...] = struct.field(
        pytree_node=False,
        default=(64, 128, 192, 240, 240, 192, 128, 64),
    )
    DEMON_HORIZONTAL_MOTION_TABLE: Tuple[int, ...] = struct.field(
        pytree_node=False,
        default=(255, 192, 160, 128, 128, 160, 192, 255),
    )
    DEMON_INITIAL_PHASE: Tuple[int, int, int] = struct.field(
        pytree_node=False,
        default=(1, 0, 0),
    )
    DEMON_INITIAL_Y: Tuple[int, int, int] = struct.field(
        pytree_node=False,
        default=(26, 41, 56),
    )
    DEMON_INITIAL_RANDOM: int = struct.field(pytree_node=False, default=234)
    DEMON_INITIAL_TELEPORT: int = struct.field(pytree_node=False, default=2)
    DEMON_INITIAL_TELEPORT_TIMER: int = struct.field(pytree_node=False, default=10)
    DEMON_MIN_VERTICAL_DISTANCE: int = struct.field(pytree_node=False, default=12)
    DEMON_TRACK_OFFSET: int = struct.field(pytree_node=False, default=4)
    MAX_ROM_WAVES: int = struct.field(pytree_node=False, default=84) # completing wave 84 freezes into a blank screen
    FREEZE_AFTER_MAX_ROM_WAVES: bool = struct.field(pytree_node=False, default=False)
    BLANK_SCREEN_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(0, 0, 0))
    WAVE_DEMON_TABLE: Tuple[int, ...] = struct.field(
        pytree_node=False,
        default=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
    )
    WAVE_BOMB_TYPE_TABLE: Tuple[int, ...] = struct.field(
        pytree_node=False,
        default=(BOMB_TYPE_STANDARD, BOMB_TYPE_STANDARD, BOMB_TYPE_LONG, BOMB_TYPE_LONG,
            BOMB_TYPE_STANDARD, BOMB_TYPE_STANDARD, BOMB_TYPE_LONG, BOMB_TYPE_LONG,
            BOMB_TYPE_STANDARD, BOMB_TYPE_STANDARD, BOMB_TYPE_LONG, BOMB_TYPE_LONG),
    )
    WAVE_LASER_SPEED_TABLE: Tuple[int, ...] = struct.field(pytree_node=False, default=(3, 4, 5, 5, 6, 6))
    ENEMY_SHOT_ACTION_TABLE: Tuple[int, ...] = struct.field(
        pytree_node=False,
        default=(8, 6, 6, 3, 5, 4, 5, 4, 5, 4, 5, 4),
    )
    ENEMY_SHOT_SPEED_TABLE: Tuple[int, ...] = struct.field(
        pytree_node=False,
        default=(1, 1, 2, 2, 3, 3),
    ) # TODO needs adjustments
    TRACKING_PROJECTILES_START_WAVE: int = struct.field(pytree_node=False, default=8) # starting in this wave, the demons begin using projectiles that follow the demon

    # Coordinates & Sizes. Sizes are (height, width).
    PLAYER_X: int = struct.field(pytree_node=False, default=87)
    PLAYER_Y: int = struct.field(pytree_node=False, default=174)
    PLAYER_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default=(12, 7))
    DEMON_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default=(9, 18))
    LASER_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default=(4, 1))
    PLAYER_LASER_DEPTH: int = struct.field(pytree_node=False, default=1)
    PLAYER_DEATH_ANIMATION_DURATION: int = struct.field(pytree_node=False, default=70)
    PLAYER_DEATH_FLASH_DURATION: int = struct.field(pytree_node=False, default=20)
    BOMB_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default=(4, 1))
    MAX_BOMBS: int = struct.field(pytree_node=False, default=7)
    BOMB_BURST_RATES: int = struct.field(pytree_node=False, default=4)
    BOMB_PRE_FIRE_PAUSE: int = struct.field(pytree_node=False, default=20)
    BOMB_BURST_LENGTH_OPTIONS: Tuple[int, ...] = struct.field(
        pytree_node=False,
        default=(1, 3, 5, 7),
    )
    # Assign the seven bomb slots to four timed volleys: 2 + 2 + 2 + 1.
    BOMB_BURST_RATE_BY_SLOT: Tuple[int, ...] = struct.field(
        pytree_node=False,
        default=(0, 0, 1, 1, 2, 2, 3),
    )
    BOMB_BURST_X_OFFSETS: Tuple[int, ...] = struct.field(
        pytree_node=False,
        default=(-2, 2, -2, 2, -2, 2, -1),
    )
    LONG_BOMB_BURST_X_OFFSETS: Tuple[int, ...] = struct.field(
        pytree_node=False,
        default=(-4, 4),
    )
    BOMB_JITTER_X_TABLE: Tuple[int, ...] = struct.field(
        pytree_node=False,
        default=(0, 0, 0, 0, 0, 0, 0),
    )
    LONG_BOMB_HEIGHT_MULTIPLIER: int = struct.field(pytree_node=False, default=5)
    MAX_BUNKERS: int = struct.field(pytree_node=False, default=6)
    INIT_BUNKERS: int = struct.field(pytree_node=False, default=3)
    BUNKER_X: int = struct.field(pytree_node=False, default=17)
    BUNKER_Y: int = struct.field(pytree_node=False, default=188)
    BUNKER_SPACING: int = struct.field(pytree_node=False, default=8)

    # Boundaries
    BOUNDARY = 25
    PLAYER_MIN_X: int = struct.field(pytree_node=False, default=BOUNDARY) # left boundary for player
    PLAYER_MAX_X: int = struct.field(pytree_node=False, default=None) # right boundary for player, calculated in compute_derived
    DEMON_MIN_X: int = struct.field(pytree_node=False, default=BOUNDARY)  # left boundary for demons
    DEMON_MAX_X: int = struct.field(pytree_node=False, default=None) # right boundary for demons, calculated in compute_derived
    DEMON_MIN_Y: int = struct.field(pytree_node=False, default=20)  # top boundary for demons
    DEMON_MAX_Y: int = struct.field(pytree_node=False, default=100) # bottom boundary for demons

    # Colors
    SCORE_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(194, 169, 53))

    ASSET_CONFIG: tuple = struct.field(pytree_node=False, default_factory=_get_default_asset_config)

    def compute_derived(self):
        return {
            'PLAYER_MAX_X': self.WIDTH - self.BOUNDARY,
            'DEMON_MAX_X': self.WIDTH - self.BOUNDARY,
        }

class DemonAttackState(struct.PyTreeNode):
    player_x: chex.Array
    laser_x: chex.Array
    laser_y: chex.Array
    laser_active: chex.Array

    demons_x: chex.Array
    demons_y: chex.Array  # Shape: (MAX_DEMONS,)
    demons_alive: chex.Array  # Shape: (MAX_DEMONS,) bool
    demon_x_motion_accumulator: chex.Array  # 8-bit fractional horizontal motion carry per slot
    demon_y_motion_accumulator: chex.Array  # 8-bit fractional vertical motion carry per slot
    demon_status: chex.Array  # Per-slot status: free, spawning, or normal
    demon_phase: chex.Array  # Per-slot movement phase, 0..7
    demon_moving_right: chex.Array  # Per-slot horizontal direction
    demon_moving_down: chex.Array  # Per-slot vertical direction
    demon_teleport: chex.Array  # Slot currently scheduled for spawn or spawn completion
    demon_teleport_timer: chex.Array  # Countdown controlling delayed appearance
    wave_spawned_demons: chex.Array  # Total demons that have entered the current wave
    demon_random: chex.Array  # Deterministic 8-bit generator used by movement and spawn timing
    demon_death_anim_timer: chex.Array

    bomb_x: chex.Array
    bomb_y: chex.Array
    bomb_active: chex.Array
    bomb_source_idx: chex.Array
    bomb_burst_step: chex.Array
    bomb_burst_length: chex.Array
    bomb_burst_timer: chex.Array
    bomb_action_counter: chex.Array

    score: chex.Array
    lives: chex.Array
    player_exploding: chex.Array
    explosion_timer: chex.Array
    wave_number: chex.Array # Actual attack wave: 0, 1, 2, ...
    wave_pattern: chex.Array # Level pattern: 0..11, then repeating 8..11.
    spawn_anim_timer: chex.Array
    spawn_pause_timer: chex.Array
    game_frozen: chex.Array
    game_over: chex.Array

    step_counter: chex.Array
    key: chex.PRNGKey

class DemonAttackObservation(struct.PyTreeNode):
    player: ObjectObservation
    demons: ObjectObservation
    laser: ObjectObservation
    bomb: ObjectObservation
    score: jnp.ndarray
    lives: jnp.ndarray

class DemonAttackInfo(struct.PyTreeNode):
    time: jnp.ndarray
    wave_number: jnp.ndarray
    wave_pattern: jnp.ndarray

class JaxDemonAttack(JaxEnvironment[DemonAttackState, DemonAttackObservation, DemonAttackInfo, DemonAttackConstants]):
    ACTION_SET: jnp.ndarray = jnp.array(
        [Action.NOOP, Action.FIRE, Action.RIGHT, Action.LEFT, Action.RIGHTFIRE, Action.LEFTFIRE],
        dtype=jnp.int32,
    )

    def __init__(self, consts: DemonAttackConstants = None):
        consts = consts or DemonAttackConstants()
        self._validate_wave_configuration(consts)
        super().__init__(consts)
        self.renderer = DemonAttackRenderer(self.consts)

    @staticmethod
    def _validate_wave_configuration(consts: DemonAttackConstants) -> None:
        """Fail early when custom wave tables cannot be indexed consistently."""
        expected_difficulty_entries = (
            INITIAL_WAVE_PATTERNS // PATTERNS_PER_DIFFICULTY_ENTRY
        )
        invalid_tables = [
            name
            for name in DIFFICULTY_TABLE_NAMES
            if len(getattr(consts, name)) != expected_difficulty_entries
        ]
        if invalid_tables:
            raise ValueError(
                f"Difficulty tables need {expected_difficulty_entries} entries: "
                f"{', '.join(invalid_tables)}"
            )
        if len(consts.WAVE_DEMON_TABLE) != INITIAL_WAVE_PATTERNS:
            raise ValueError(
                f"WAVE_DEMON_TABLE needs {INITIAL_WAVE_PATTERNS} pattern entries"
            )
        if len(consts.WAVE_BOMB_TYPE_TABLE) != INITIAL_WAVE_PATTERNS:
            raise ValueError(
                f"WAVE_BOMB_TYPE_TABLE needs {INITIAL_WAVE_PATTERNS} pattern entries"
            )
        if len(consts.ENEMY_SHOT_ACTION_TABLE) != INITIAL_WAVE_PATTERNS:
            raise ValueError(
                f"ENEMY_SHOT_ACTION_TABLE needs {INITIAL_WAVE_PATTERNS} pattern entries"
            )

    def _resolve_wave_pattern(self, wave_number: chex.Array) -> chex.Array:
        """Map the absolute wave number to pattern 0..11, then repeat 8..11."""
        wave_number = jnp.maximum(wave_number, 0)
        repeating_pattern_count = (
            INITIAL_WAVE_PATTERNS - REPEATING_WAVE_PATTERN_START
        )
        return jnp.where(
            wave_number < INITIAL_WAVE_PATTERNS,
            wave_number,
            REPEATING_WAVE_PATTERN_START
            + jnp.mod(wave_number - INITIAL_WAVE_PATTERNS, repeating_pattern_count),
        )

    @staticmethod
    def _difficulty_index_for_pattern(wave_pattern: chex.Array) -> chex.Array:
        """Map two consecutive patterns to one shared difficulty-table entry."""
        return wave_pattern // PATTERNS_PER_DIFFICULTY_ENTRY

    def _initial_demon_values(self):
        """Build initial per-demon movement and spawn fields."""
        zeros = jnp.zeros((self.consts.MAX_DEMONS,), dtype=jnp.int32)
        return dict(
            demons_x=zeros,
            demons_y=jnp.asarray(self.consts.DEMON_INITIAL_Y, dtype=jnp.int32),
            demons_alive=jnp.zeros((self.consts.MAX_DEMONS,), dtype=jnp.bool_),
            demon_x_motion_accumulator=zeros,
            demon_y_motion_accumulator=zeros,
            demon_status=jnp.full((self.consts.MAX_DEMONS,), DEMON_STATUS_FREE, dtype=jnp.int32),
            demon_phase=jnp.asarray(self.consts.DEMON_INITIAL_PHASE, dtype=jnp.int32),
            demon_moving_right=jnp.zeros((self.consts.MAX_DEMONS,), dtype=jnp.bool_),
            demon_moving_down=jnp.ones((self.consts.MAX_DEMONS,), dtype=jnp.bool_),
            demon_teleport=jnp.array(self.consts.DEMON_INITIAL_TELEPORT, dtype=jnp.int32),
            demon_teleport_timer=jnp.array(self.consts.DEMON_INITIAL_TELEPORT_TIMER, dtype=jnp.int32),
            wave_spawned_demons=jnp.array(0, dtype=jnp.int32),
            demon_random=jnp.array(self.consts.DEMON_INITIAL_RANDOM, dtype=jnp.int32),
            demon_death_anim_timer=zeros,
        )

    def _next_demon_random(self, random: chex.Array) -> chex.Array:
        """Advance the deterministic 8-bit demon movement pseudo-random value."""
        shifted = (random * 2) & 255
        carry = ((shifted ^ random) // 64) & 1
        return (shifted | carry).astype(jnp.int32)

    def _new_demon_y(self, demons_y: chex.Array, demon: chex.Array) -> chex.Array:
        """Choose a vertically spaced target row for a respawning demon slot."""
        def first():
            return (jnp.array(self.consts.DEMON_MIN_Y, dtype=jnp.int32) + demons_y[1]) // 2

        def second():
            return (demons_y[0] + demons_y[2]) // 2

        def third():
            return (
                demons_y[1]
                + jnp.array(self.consts.DEMON_MAX_Y, dtype=jnp.int32)
            ) // 2

        return jax.lax.switch(demon, (first, second, third)).astype(jnp.int32)

    def _difficulty_value_for_pattern(
        self,
        table: Tuple,
        wave_pattern: chex.Array,
        dtype=jnp.int32,
    ) -> chex.Array:
        """Read a value from a six-entry, pair-shared difficulty table."""
        values = jnp.asarray(table, dtype=dtype)
        index = jnp.clip(
            self._difficulty_index_for_pattern(wave_pattern),
            0,
            values.shape[0] - 1,
        )
        return values[index]

    def _bomb_type_for_wave(self, wave_pattern: chex.Array) -> chex.Array:
        bomb_types = jnp.asarray(self.consts.WAVE_BOMB_TYPE_TABLE, dtype=jnp.int32)
        pattern = jnp.clip(wave_pattern, 0, bomb_types.shape[0] - 1)
        return bomb_types[pattern]

    def _uses_long_bombs(self, wave_pattern: chex.Array) -> chex.Array:
        """Return whether the current wave pattern fires longer bombs."""
        return self._bomb_type_for_wave(wave_pattern) == BOMB_TYPE_LONG

    def _bomb_height_for_wave(self, wave_pattern: chex.Array) -> chex.Array:
        return jnp.where(
            self._uses_long_bombs(wave_pattern),
            self.consts.BOMB_SIZE[0] * self.consts.LONG_BOMB_HEIGHT_MULTIPLIER,
            self.consts.BOMB_SIZE[0],
        )

    def _bomb_visible_repeat_window(
        self, state: DemonAttackState
    ) -> Tuple[chex.Array, chex.Array]:
        bomb_type = self._bomb_type_for_wave(state.wave_pattern)
        return _bomb_visible_repeat_window(state, self.consts, bomb_type)

    def _bomb_collision_y_bounds(
        self, state: DemonAttackState
    ) -> Tuple[chex.Array, chex.Array]:
        """Return top and bottom y bounds for the currently visible enemy shots."""
        visible_repeats, repeat_offset = self._bomb_visible_repeat_window(state)
        bomb_top = state.bomb_y - repeat_offset * self.consts.BOMB_SIZE[0]
        bomb_bottom = bomb_top + visible_repeats * self.consts.BOMB_SIZE[0]
        return bomb_top, bomb_bottom

    def _bomb_burst_length_for_type(
        self, bomb_type: chex.Array, random_burst_length: chex.Array
    ) -> chex.Array:
        return jnp.where(bomb_type == BOMB_TYPE_LONG, 2, random_burst_length)

    def _bomb_jitter_for_type(
        self, bomb_type: chex.Array, standard_jitter_x: chex.Array
    ) -> chex.Array:
        return jnp.where(bomb_type == BOMB_TYPE_LONG, 0, standard_jitter_x)

    def _bomb_x_offsets_for_type(self, bomb_type: chex.Array) -> chex.Array:
        standard_offsets = jnp.asarray(
            self.consts.BOMB_BURST_X_OFFSETS,
            dtype=jnp.int32,
        )
        long_offsets = jnp.asarray(
            self.consts.LONG_BOMB_BURST_X_OFFSETS,
            dtype=jnp.int32,
        )
        long_offsets = jnp.pad(
            long_offsets,
            (0, self.consts.MAX_BOMBS - len(self.consts.LONG_BOMB_BURST_X_OFFSETS)),
        )
        return jnp.where(bomb_type == BOMB_TYPE_LONG, long_offsets, standard_offsets)

    def _bomb_sprite_repeats_for_type(self, bomb_type: chex.Array) -> chex.Array:
        return jnp.where(
            bomb_type == BOMB_TYPE_LONG,
            self.consts.LONG_BOMB_HEIGHT_MULTIPLIER,
            1,
        )

    def _spawn_target_x(self, ids: chex.Array) -> chex.Array:
        """Return evenly spaced spawn x positions for demon slot ids."""
        spacing = (self.consts.DEMON_MAX_X - self.consts.DEMON_MIN_X) // (self.consts.MAX_DEMONS + 1)
        return (self.consts.DEMON_MIN_X + (ids + 1) * spacing).astype(jnp.int32)

    def _sync_demon_status(self, state: DemonAttackState) -> DemonAttackState:
        """Derive public liveness from demon fields."""
        return state.replace(
            demons_alive=state.demon_status != DEMON_STATUS_FREE,
        )

    def _initialize_wave_state(self, state: DemonAttackState, wave_number: chex.Array) -> DemonAttackState:
        """Replace the previous wave state with a new initialized wave."""
        state = state.replace(
            wave_number=wave_number,
            wave_pattern=self._resolve_wave_pattern(wave_number),
            **self._initial_demon_values(),
            spawn_anim_timer=jnp.zeros((self.consts.MAX_DEMONS,), dtype=jnp.int32),
            spawn_pause_timer=jnp.zeros((self.consts.MAX_DEMONS,), dtype=jnp.int32),
            game_frozen=jnp.array(False, dtype=jnp.bool_),
            bomb_x=jnp.zeros((self.consts.MAX_BOMBS,), dtype=jnp.int32),
            bomb_y=jnp.zeros((self.consts.MAX_BOMBS,), dtype=jnp.int32),
            bomb_active=jnp.zeros((self.consts.MAX_BOMBS,), dtype=jnp.bool_),
            bomb_source_idx=jnp.array(0, dtype=jnp.int32),
            bomb_burst_step=jnp.array(self.consts.BOMB_BURST_RATES, dtype=jnp.int32),
            bomb_burst_length=jnp.array(0, dtype=jnp.int32),
            bomb_burst_timer=jnp.array(0, dtype=jnp.int32),
            bomb_action_counter=jnp.array(0, dtype=jnp.int32),
        )

        return self._sync_demon_status(state)

    def _advance_wave(self, state: DemonAttackState) -> DemonAttackState:
        next_wave_number = state.wave_number + 1

        should_freeze = jnp.logical_and(
            jnp.array(self.consts.FREEZE_AFTER_MAX_ROM_WAVES, dtype=jnp.bool_),
            next_wave_number >= self.consts.MAX_ROM_WAVES,
        )

        return jax.lax.cond(
            should_freeze,
            lambda s: s.replace(
                wave_number=next_wave_number,
                demons_alive=jnp.zeros((self.consts.MAX_DEMONS,), dtype=jnp.bool_),
                bomb_active=jnp.zeros((self.consts.MAX_BOMBS,), dtype=jnp.bool_),
                bomb_source_idx=jnp.array(0, dtype=jnp.int32),
                bomb_burst_step=jnp.array(self.consts.BOMB_BURST_RATES, dtype=jnp.int32),
                bomb_burst_length=jnp.array(0, dtype=jnp.int32),
                bomb_burst_timer=jnp.array(0, dtype=jnp.int32),
                bomb_action_counter=jnp.array(0, dtype=jnp.int32),
                laser_active=jnp.array(False, dtype=jnp.bool_),
                game_frozen=jnp.array(True, dtype=jnp.bool_),
            ),
            lambda s: self._initialize_wave_state(
                s.replace(
                    lives=jnp.minimum(
                        s.lives + 1,
                        jnp.array(self.consts.MAX_BUNKERS, dtype=jnp.int32)
                    )
                ),
                next_wave_number
            ),
            operand=state,
        )

    def reset(self, key: chex.PRNGKey = jax.random.PRNGKey(42)) -> Tuple[DemonAttackObservation, DemonAttackState]:
        wave_number = jnp.array(0, dtype=jnp.int32)

        state = DemonAttackState(
            player_x=jnp.array(self.consts.PLAYER_X, dtype=jnp.int32),
            laser_x=jnp.array(0, dtype=jnp.int32),
            laser_y=jnp.array(0, dtype=jnp.int32),
            laser_active=jnp.array(False, dtype=jnp.bool_),
            **self._initial_demon_values(),
            bomb_x=jnp.zeros((self.consts.MAX_BOMBS,), dtype=jnp.int32),
            bomb_y=jnp.zeros((self.consts.MAX_BOMBS,), dtype=jnp.int32),
            bomb_active=jnp.zeros((self.consts.MAX_BOMBS,), dtype=jnp.bool_),
            bomb_source_idx=jnp.array(0, dtype=jnp.int32),
            bomb_burst_step=jnp.array(self.consts.BOMB_BURST_RATES, dtype=jnp.int32),
            bomb_burst_length=jnp.array(0, dtype=jnp.int32),
            bomb_burst_timer=jnp.array(0, dtype=jnp.int32),
            bomb_action_counter=jnp.array(0, dtype=jnp.int32),
            score=jnp.array(0, dtype=jnp.int32),
            lives=jnp.array(self.consts.INIT_BUNKERS, dtype=jnp.int32),
            player_exploding=jnp.array(False, dtype=jnp.bool_),
            explosion_timer=jnp.array(0, dtype=jnp.int32),
            wave_number=wave_number,
            wave_pattern=self._resolve_wave_pattern(wave_number),
            spawn_anim_timer=jnp.zeros((self.consts.MAX_DEMONS,), dtype=jnp.int32),
            spawn_pause_timer=jnp.zeros((self.consts.MAX_DEMONS,), dtype=jnp.int32),
            game_frozen=jnp.array(False, dtype=jnp.bool_),
            game_over=jnp.array(False, dtype=jnp.bool_),
            step_counter=jnp.array(0, dtype=jnp.int32),
            key=key,
        )
        state = self._sync_demon_status(state)
        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: DemonAttackState, action: chex.Array) -> Tuple[
        DemonAttackObservation, DemonAttackState, float, bool, DemonAttackInfo]:
        atari_action = jnp.take(self.ACTION_SET, action.astype(jnp.int32))

        prev_state = state

        # Handle Explosion Timer
        def update_explosion(s):
            new_timer = s.explosion_timer - 1
            exploding = new_timer > 0
            # If timer reaches 0, player hit logic should have already reduced lives.
            # We just need to stop exploding and teleport the player back to his original x coordinate
            player_x = jnp.where(
                exploding,
                s.player_x,
                jnp.array(self.consts.PLAYER_X, dtype=jnp.int32),
            )
            return s.replace(
                player_x=player_x,
                explosion_timer=new_timer,
                player_exploding=exploding,
            )

        def explosion_step(s):
            s = update_explosion(s)
            s = self._laser_step(s, jnp.array(Action.NOOP, dtype=jnp.int32))
            s = self._update_spawn_timers(s)
            return self._demons_step(s)

        def normal_step(s, act):
            # 0. Spawn Animation Step
            s = self._update_spawn_timers(s)
            # 1. Player Step
            s = self._player_step(s, act)
            # 2. Laser Step
            s = self._laser_step(s, act)
            # 3. Demon Step
            s = self._demons_step(s)
            # 4. Bomb Step
            s = self._bomb_step(s)
            # 5. Collision Detection
            s = self._handle_collisions(s)
            return s

        state = jax.lax.cond(
            state.game_frozen,
            lambda s: s,
            lambda s: jax.lax.cond(
                s.player_exploding,
                explosion_step,
                lambda ss: normal_step(ss, atari_action),
                operand=s,
            ),
            operand=state,
        )

        _, next_key = jax.random.split(state.key)
        state = state.replace(key=next_key, step_counter=state.step_counter + 1)

        observation = self._get_observation(state)
        reward = self._get_reward(prev_state, state)
        done = self._get_done(state)
        info = self._get_info(state)
        return observation, state, reward, done, info

    def _player_step(self, state: DemonAttackState, action: chex.Array) -> DemonAttackState:
        move_right = jnp.logical_or(action == Action.RIGHT, action == Action.RIGHTFIRE)
        move_left = jnp.logical_or(action == Action.LEFT, action == Action.LEFTFIRE)

        dx = jax.lax.select(
            move_right,
            self.consts.PLAYER_SPEED,
            jax.lax.select(move_left, -self.consts.PLAYER_SPEED, 0),
        )

        new_x = jnp.clip(state.player_x + dx, self.consts.PLAYER_MIN_X, self.consts.PLAYER_MAX_X)
        return state.replace(player_x=new_x)

    def _update_spawn_timers(self, state: DemonAttackState) -> DemonAttackState:
        """Advance spawn animation and post-spawn movement-pause timers."""
        next_spawn_anim_timer = jnp.maximum(state.spawn_anim_timer - 1, 0)
        pause_can_tick = jnp.logical_and(
            state.spawn_anim_timer <= 0,
            state.spawn_pause_timer > 0,
        )

        return state.replace(
            spawn_anim_timer=next_spawn_anim_timer,
            spawn_pause_timer=jnp.where(
                pause_can_tick,
                state.spawn_pause_timer - 1,
                state.spawn_pause_timer,
            ),
            demon_death_anim_timer=jnp.maximum(state.demon_death_anim_timer - 1, 0),
        )

    def _demons_ready(self, state: DemonAttackState) -> chex.Array:
        return jnp.logical_and(
            state.demon_status == DEMON_STATUS_NORMAL,
            state.spawn_pause_timer <= 0,
        )

    def _active_demon_idx(self) -> chex.Array:
        """Return slot 3, the only demon allowed to track or fire."""
        return jnp.array(self.consts.MAX_DEMONS - 1, dtype=jnp.int32)

    def _track_demon(
        self,
        state: DemonAttackState,
        demons_x: chex.Array,
        can_track: chex.Array,
        demon_moving_right: chex.Array,
    ) -> chex.Array:
        """
        Keep the lowest demon slot beside the player. Inside the border it keeps
        its own movement; outside it returns to the nearest border.
        """
        player_left = state.player_x
        player_right = state.player_x + self.consts.PLAYER_SIZE[1]
        player_center = player_left + self.consts.PLAYER_SIZE[1] // 2
        demon_left = demons_x
        demon_right = demons_x + self.consts.DEMON_SIZE[1]
        demon_center = demon_left + self.consts.DEMON_SIZE[1] // 2
        camps_left = demon_center < player_center
        edge_gap = jnp.where(
            camps_left,
            player_left - demon_right,
            demon_left - player_right,
        )
        inside_border = jnp.abs(edge_gap) <= self.consts.DEMON_TRACK_OFFSET
        hover_direction = jnp.where(
            edge_gap <= 0,
            jnp.logical_not(camps_left),
            jnp.where(edge_gap >= self.consts.DEMON_TRACK_OFFSET, camps_left, demon_moving_right),
        )
        target_x = jnp.where(
            camps_left,
            state.player_x - self.consts.DEMON_SIZE[1] - self.consts.DEMON_TRACK_OFFSET,
            state.player_x + self.consts.PLAYER_SIZE[1] + self.consts.DEMON_TRACK_OFFSET,
        )
        tracking_direction = jnp.where(
            jnp.logical_and(can_track, jnp.logical_not(inside_border)),
            demons_x < target_x,
            jnp.where(can_track, hover_direction, demon_moving_right),
        )
        return tracking_direction

    def _laser_step(self, state: DemonAttackState, action: chex.Array) -> DemonAttackState:
        # Fire laser if not active and FIRE action
        fire = jnp.logical_or(
            jnp.logical_or(action == Action.FIRE, action == Action.RIGHTFIRE),
            action == Action.LEFTFIRE,
        )
        should_fire = jnp.logical_and(fire, jnp.logical_not(state.laser_active))

        laser_x = jax.lax.select(
            should_fire,
            state.player_x + self.consts.PLAYER_SIZE[1] // 2,
            state.laser_x,
        )
        laser_y = jax.lax.select(
            should_fire,
            jnp.array(self.consts.PLAYER_Y + self.consts.PLAYER_LASER_DEPTH, dtype=jnp.int32),
            state.laser_y,
        )
        laser_active = jnp.logical_or(should_fire, state.laser_active)

        # Move laser
        laser_speed = self._difficulty_value_for_pattern(
            self.consts.WAVE_LASER_SPEED_TABLE, state.wave_pattern
        )
        laser_y = jax.lax.select(laser_active, laser_y - laser_speed, laser_y)

        # Deactivate if out of bounds
        laser_active = jnp.logical_and(laser_active, laser_y > 0)

        return state.replace(laser_x=laser_x, laser_y=laser_y, laser_active=laser_active)

    def _demons_step(self, state: DemonAttackState) -> DemonAttackState:
        """Advance demon spawn scheduling, movement phases, and movement.

        A single selected slot is nudged toward its vertical spacing target each frame,
        free slots are scheduled through ``demon_teleport_timer``, and normal
        demons move when their 8-bit motion accumulators overflow.
        """
        state = state.replace(demon_random=self._next_demon_random(state.demon_random))
        ids = jnp.arange(self.consts.MAX_DEMONS)
        frame_mod4 = state.step_counter & 3
        selected = jnp.maximum(frame_mod4 - 1, 0)

        # Movement is globally paused while demons are not ready, and locally
        # paused for the demon currently emitting a burst.
        can_move = self._demons_ready(state)
        rate_by_slot = jnp.asarray(
            self.consts.BOMB_BURST_RATE_BY_SLOT,
            dtype=jnp.int32,
        )
        last_active_rate = rate_by_slot[jnp.maximum(state.bomb_burst_length - 1, 0)]
        burst_in_progress = jnp.logical_and(
            state.bomb_burst_length > 0,
            state.bomb_burst_step <= last_active_rate,
        )
        source_bomb_y = jnp.max(jnp.where(state.bomb_active, state.bomb_y, 0))
        source_bomb_spawn_y = state.demons_y[state.bomb_source_idx] + self.consts.DEMON_SIZE[0]
        long_burst_in_progress = jnp.logical_and(
            self._uses_long_bombs(state.wave_pattern),
            jnp.logical_and(
                jnp.any(state.bomb_active),
                source_bomb_y - source_bomb_spawn_y < (
                    self.consts.LONG_BOMB_HEIGHT_MULTIPLIER - 1
                ) * self.consts.BOMB_SIZE[0],
            ),
        )
        burst_in_progress = jnp.logical_or(burst_in_progress, long_burst_in_progress)
        source_ids = ids == state.bomb_source_idx
        can_move = jnp.logical_and(
            can_move,
            jnp.logical_not(jnp.logical_and(burst_in_progress, source_ids)),
        )

        # One slot per frame is nudged toward its spacing target. The random
        # direction flip keeps horizontal motion from becoming fully periodic.
        target_y = self._new_demon_y(state.demons_y, selected)
        selected_mask = ids == selected
        selected_active = frame_mod4 != 0
        demons_y = jnp.where(
            selected_active & selected_mask,
            state.demons_y + jnp.where(target_y >= state.demons_y[selected], 1, -1),
            state.demons_y,
        )
        tracking_demon = ids == self._active_demon_idx()
        demon_moving_right = jnp.where(
            selected_active & selected_mask & ~tracking_demon & ((state.demon_random & 7) == 0),
            jnp.logical_not(state.demon_moving_right),
            state.demon_moving_right,
        )
        phase_mask = ids == frame_mod4
        next_phase = (state.demon_phase + 1) & 7
        next_moving_down = jnp.where(
            state.demon_phase == 7,
            jnp.logical_not(state.demon_moving_down),
            state.demon_moving_down,
        )
        demon_phase = jnp.where(
            phase_mask,
            next_phase,
            state.demon_phase,
        )
        demon_moving_down = jnp.where(
            phase_mask,
            next_moving_down,
            state.demon_moving_down,
        )

        # Teleport scheduling is the spawn state machine. A free slot is chosen,
        # waits for the configured respawn delay, enters spawn animation, then
        # becomes normal.
        timer = jnp.maximum(state.demon_teleport_timer - 1, 0)
        tele_mask = ids == state.demon_teleport
        tele_status = state.demon_status[state.demon_teleport]
        can_appear = state.wave_spawned_demons < self.consts.WAVE_TOTAL_DEMONS
        start_spawn = (
                (state.demon_teleport_timer > 0)
                & (timer == 0)
                & (tele_status == DEMON_STATUS_FREE)
                & can_appear
        )
        finish_spawn = (
                (state.demon_teleport_timer > 0)
                & (timer == 0)
                & (tele_status != DEMON_STATUS_FREE)
        )
        can_schedule = (state.demon_teleport_timer == 0) & can_appear
        demon_status = state.demon_status
        free = jnp.logical_and(
            demon_status == DEMON_STATUS_FREE,
            state.demon_death_anim_timer <= 0,
        )
        scheduled = self.consts.MAX_DEMONS - 1 - jnp.argmax(free[::-1].astype(jnp.int32))
        schedule = can_schedule & jnp.any(free)

        # New demons start from the target row for their slot so the formation
        # stays vertically separated as the wave refills.
        demon_teleport = jnp.where(schedule, scheduled, state.demon_teleport)
        schedule_mask = ids == scheduled
        demons_y = jnp.where(
            schedule & schedule_mask,
            self._new_demon_y(demons_y, scheduled),
            demons_y,
        )
        demon_status = jnp.where(
            start_spawn & tele_mask,
            DEMON_STATUS_SPAWNING,
            demon_status,
        )
        demon_status = jnp.where(
            finish_spawn & tele_mask,
            DEMON_STATUS_NORMAL,
            demon_status,
        )
        demon_phase = jnp.where(finish_spawn & tele_mask, 0, demon_phase)
        demon_moving_right = jnp.where(finish_spawn & tele_mask, True, demon_moving_right)
        demon_moving_down = jnp.where(finish_spawn & tele_mask, True, demon_moving_down)

        spawn_target_x = self._spawn_target_x(ids)
        demons_x = jnp.where(
            (start_spawn | finish_spawn) & tele_mask,
            spawn_target_x,
            state.demons_x,
        )

        # Motion tables are fractional speeds. Accumulators overflow past 255
        # to produce a one-pixel step on that axis.
        y_motion_sum = (
                state.demon_y_motion_accumulator
                + jnp.asarray(self.consts.DEMON_VERTICAL_MOTION_TABLE, dtype=jnp.int32)[demon_phase]
        )
        x_motion_sum = (
                state.demon_x_motion_accumulator
                + jnp.asarray(self.consts.DEMON_HORIZONTAL_MOTION_TABLE, dtype=jnp.int32)[demon_phase]
        )
        move_y = can_move & (y_motion_sum > 255)
        move_x = can_move & (x_motion_sum > 255)

        demons_y = jnp.where(
            move_y,
            demons_y + jnp.where(demon_moving_down, 1, -1),
            demons_y,
        )

        # Horizontal boundary hits flip the direction. If the step overshot
        # the legal area, restore the previous x before continuing.
        previous_x = demons_x
        demons_x = jnp.where(
            move_x,
            demons_x + jnp.where(demon_moving_right, 1, -1),
            demons_x,
        )
        outside_x = (demons_x < self.consts.DEMON_MIN_X) | (demons_x > self.consts.DEMON_MAX_X)
        turn = can_move & (
                (demon_moving_right & (demons_x >= self.consts.DEMON_MAX_X))
                | (~demon_moving_right & (demons_x <= self.consts.DEMON_MIN_X))
        )
        demons_x = jnp.where(turn & outside_x, previous_x, demons_x)
        demon_moving_right = jnp.where(turn, jnp.logical_not(demon_moving_right), demon_moving_right)
        demon_moving_down = jnp.where(turn, True, demon_moving_down)
        demon_phase = jnp.where(turn, 1, demon_phase)
        demon_moving_right = self._track_demon(
            state,
            demons_x,
            can_move & selected_active & selected_mask & tracking_demon,
            demon_moving_right,
        )

        # Keep the three slots ordered top-to-bottom with a minimum gap. This
        # prevents the target nudges from collapsing demon rows.
        top = jnp.clip(demons_y[0], self.consts.DEMON_MIN_Y, self.consts.DEMON_MAX_Y)
        middle = jnp.maximum(
            demons_y[1],
            top + self.consts.DEMON_MIN_VERTICAL_DISTANCE,
        )
        bottom = jnp.maximum(
            demons_y[2],
            middle + self.consts.DEMON_MIN_VERTICAL_DISTANCE,
        )
        demons_y = jnp.clip(jnp.stack((
            top,
            middle,
            bottom,
        )), self.consts.DEMON_MIN_Y, self.consts.DEMON_MAX_Y).astype(jnp.int32)

        # Store the state, then derive the public alive mask and wave-spawned count.
        state = state.replace(
            demons_x=demons_x,
            demons_y=demons_y,
            demon_x_motion_accumulator=jnp.where(
                can_move,
                x_motion_sum & 255,
                state.demon_x_motion_accumulator,
            ),
            demon_y_motion_accumulator=jnp.where(
                can_move,
                y_motion_sum & 255,
                state.demon_y_motion_accumulator,
            ),
            demon_status=demon_status,
            demon_phase=demon_phase,
            demon_moving_right=demon_moving_right,
            demon_moving_down=demon_moving_down,
            demon_teleport=demon_teleport,
            demon_teleport_timer=jnp.where(
                start_spawn,
                self.consts.DEMON_TELEPORT_DURATION,
                jnp.where(
                    finish_spawn,
                    0,
                    jnp.where(
                        schedule,
                        jnp.array(self.consts.RESPAWN_DELAY, dtype=jnp.int32),
                        timer,
                    ),
                ),
            ),
            wave_spawned_demons=state.wave_spawned_demons + start_spawn.astype(jnp.int32),
            spawn_anim_timer=jnp.where(
                start_spawn & tele_mask,
                self.consts.DEMON_TELEPORT_DURATION,
                jnp.where(finish_spawn & tele_mask, 0, state.spawn_anim_timer),
            ),
            spawn_pause_timer=jnp.where(
                finish_spawn & tele_mask,
                self.consts.SPAWN_MOVE_PAUSE,
                state.spawn_pause_timer,
            ),
        )
        return self._sync_demon_status(state)

    def _bomb_step(self, state: DemonAttackState) -> DemonAttackState:
        """Advance enemy bomb movement and burst-firing state by one frame.

        Existing bombs move downward at the speed selected for the current wave
        pattern and receive their slot-specific horizontal jitter. Bombs that
        reach the bunker boundary are deactivated.

        The method also advances the enemy firing scheduler. Once the wave's
        action delay has elapsed, no previous bombs remain active, and at least
        one demon is ready, a burst begins from a selected demon. Each burst
        retains that source demon and activates the bomb slots assigned to its
        current rate after the configured interval. The burst source is released
        when all rates have been processed.
        """
        key, burst_length_key = jax.random.split(
            state.key, 2
        )
        ready_demons = self._demons_ready(state)
        shooting_demon_idx = self._active_demon_idx()

        slot_ids = jnp.arange(self.consts.MAX_BOMBS, dtype=jnp.int32)

        # First branch: wave/action timing logic
        action_limit = jnp.asarray(
            self.consts.ENEMY_SHOT_ACTION_TABLE,
            dtype=jnp.int32,
        )[state.wave_pattern]

        action_counter = state.bomb_action_counter + 1
        any_bomb_active = jnp.any(state.bomb_active)
        bomb_type = self._bomb_type_for_wave(state.wave_pattern)

        # Advance existing bombs before adding the current frame's bomb
        bomb_speed = self._difficulty_value_for_pattern(
            self.consts.ENEMY_SHOT_SPEED_TABLE,
            state.wave_pattern,
        )
        moved_y = state.bomb_y + jnp.where(state.bomb_active, bomb_speed, 0)
        bomb_despawn_y = self.consts.BUNKER_Y - self.consts.BOMB_SIZE[0]
        bomb_active_limit = jnp.where(
            bomb_type == BOMB_TYPE_LONG,
            bomb_despawn_y
            + (self.consts.LONG_BOMB_HEIGHT_MULTIPLIER - 1) * self.consts.BOMB_SIZE[0],
            bomb_despawn_y,
        )
        bomb_active = jnp.logical_and(
            state.bomb_active,
            moved_y < bomb_active_limit,
        )

        def _calc_burst_base_x(_source_idx: chex.Array, _state: DemonAttackState) -> chex.Array:
            """
            Calculate the center of the demon bomb burst.
            :param _source_idx: Demon to use as reference for where to place the burst
            :return:Array with the same x-position for each bomb
            """
            return (
                    _state.demons_x[_source_idx]
                    + self.consts.DEMON_SIZE[1] // 2
                    - self.consts.BOMB_SIZE[1] // 2
            )

        # First branch: per-slot jitter logic
        jitter_table = jnp.asarray(
            self.consts.BOMB_JITTER_X_TABLE,
            dtype=jnp.int32,
        )
        jitter_phase = jnp.mod(
            state.step_counter + slot_ids,
            len(self.consts.BOMB_JITTER_X_TABLE),
        )
        jitter_x = self._bomb_jitter_for_type(bomb_type, jitter_table[jitter_phase])

        should_use_tracking_projectiles = state.wave_number >= self.consts.TRACKING_PROJECTILES_START_WAVE

        def use_tracking_bombs(s):
            _base_x = _calc_burst_base_x(s.bomb_source_idx, s)
            bomb_type = self._bomb_type_for_wave(state.wave_pattern)
            tracked_x = _base_x + self._bomb_x_offsets_for_type(bomb_type)
            # Stop tracking once the original source demon is dead/respawning,
            # otherwise the bomb snaps to whatever new demon reuses that slot.
            source_still_ready = ready_demons[s.bomb_source_idx]
            return jnp.where(source_still_ready, tracked_x, s.bomb_x)

        def use_normal_bombs(s):
            return s.bomb_x

        x_before_jitter = jax.lax.cond(
            should_use_tracking_projectiles,
            use_tracking_bombs,
            use_normal_bombs,
            operand=state,
        )

        moved_x = x_before_jitter + jnp.where(bomb_active, jitter_x, 0)
        bomb_x = jnp.where(bomb_active, moved_x, state.bomb_x)
        bomb_y = jnp.where(bomb_active, moved_y, state.bomb_y)

        # A burst owns one demon until up to all four bomb rates have fired. New bursts wait
        # until the previous bombs have left the screen.
        burst_in_progress = state.bomb_burst_length > 0
        scheduler_idle = jnp.logical_and(
            jnp.logical_not(burst_in_progress),
            jnp.logical_not(any_bomb_active),
        )
        action_due = action_counter >= action_limit
        has_ready_demon = ready_demons[shooting_demon_idx]
        can_start_burst = jnp.logical_and(
            scheduler_idle,
            jnp.logical_and(action_due, has_ready_demon),
        )
        source_idx = jnp.where(
            can_start_burst,
            shooting_demon_idx,
            state.bomb_source_idx,
        )
        source_ready = ready_demons[source_idx]
        base_x = _calc_burst_base_x(source_idx, state)

        burst_length_idx = jax.random.randint(
            burst_length_key,
            (),
            0,
            len(self.consts.BOMB_BURST_LENGTH_OPTIONS),
            dtype=jnp.int32,
        )
        burst_length_options = jnp.asarray(
            self.consts.BOMB_BURST_LENGTH_OPTIONS,
            dtype=jnp.int32,
        )
        burst_length = burst_length_options[burst_length_idx]
        burst_length = self._bomb_burst_length_for_type(bomb_type, burst_length)
        active_burst_length = jnp.where(
            can_start_burst,
            burst_length,
            state.bomb_burst_length,
        )
        burst_step = jnp.where(
            can_start_burst,
            0,
            state.bomb_burst_step,
        )
        burst_timer = jnp.where(
            can_start_burst,
            self.consts.BOMB_PRE_FIRE_PAUSE,
            state.bomb_burst_timer,
        )
        # Activate every slot assigned to this bomb shot in one vectorized operation.
        safe_burst_step = jnp.minimum(
            burst_step,
            self.consts.BOMB_BURST_RATES - 1,
        )
        rate_by_slot = jnp.asarray(
            self.consts.BOMB_BURST_RATE_BY_SLOT,
            dtype=jnp.int32,
        )
        last_active_rate = rate_by_slot[jnp.maximum(active_burst_length - 1, 0)]
        burst_in_progress = burst_step <= last_active_rate
        fire_rate_now = jnp.logical_and(
            burst_in_progress,
            jnp.logical_and(source_ready, burst_timer <= 0),
        )
        active_burst_slots = slot_ids < active_burst_length
        slots_in_rate = jnp.logical_and(
            rate_by_slot == safe_burst_step,
            active_burst_slots,
        )

        x_offsets = self._bomb_x_offsets_for_type(bomb_type)
        fired_x = base_x + x_offsets
        fired_y = (
            state.demons_y[source_idx]
            + self.consts.DEMON_SIZE[0]
        )

        should_activate_slot = jnp.logical_and(
            jnp.logical_and(fire_rate_now, slots_in_rate),
            jnp.logical_not(bomb_active),
        )
        bomb_x = jnp.where(should_activate_slot, fired_x, bomb_x)
        bomb_y = jnp.where(should_activate_slot, fired_y, bomb_y)
        bomb_active = jnp.logical_or(bomb_active, should_activate_slot)

        # After firing, arm the delay before the next shot in the same burst.
        next_burst_step = jnp.where(fire_rate_now, burst_step + 1, burst_step)
        burst_done = next_burst_step > last_active_rate
        next_burst_timer = jnp.where(
            fire_rate_now,
            jnp.where(burst_done, 0, action_limit),
            jnp.maximum(burst_timer - 1, 0),
        )
        source_lost = jnp.logical_and(active_burst_length > 0, jnp.logical_not(source_ready))
        release_source = jnp.logical_and(
            jnp.logical_or(burst_done, source_lost),
            jnp.logical_not(jnp.any(bomb_active)),
        )
        next_burst_length = jnp.where(
            release_source,
            jnp.array(0, dtype=jnp.int32),
            active_burst_length,
        )
        next_burst_step = jnp.where(
            release_source,
            jnp.array(self.consts.BOMB_BURST_RATES, dtype=jnp.int32),
            next_burst_step,
        )
        next_burst_timer = jnp.where(
            release_source,
            jnp.array(0, dtype=jnp.int32),
            next_burst_timer,
        )
        source_idx = jnp.where(
            release_source,
            jnp.array(0, dtype=jnp.int32),
            source_idx,
        )
        action_counter = jnp.where(can_start_burst, 0, action_counter)

        return state.replace(
            key=key,
            bomb_x=bomb_x,
            bomb_y=bomb_y,
            bomb_active=bomb_active,
            bomb_source_idx=source_idx,
            bomb_burst_step=next_burst_step,
            bomb_burst_length=next_burst_length,
            bomb_burst_timer=next_burst_timer,
            bomb_action_counter=action_counter,
        )

    def _handle_collisions(self, state: DemonAttackState) -> DemonAttackState:
        # Laser vs Demons
        laser_right = state.laser_x + self.consts.LASER_SIZE[1]
        laser_bottom = state.laser_y + self.consts.LASER_SIZE[0]

        def check_demon_collision(i, carry):
            """
            The carry contains the current alive mask, score, and laser-active
            flag. A demon can only be hit after its spawn animation has ended,
            and a successful hit clears that demon, adds score, and consumes the
            laser so later demon slots in this loop cannot also be hit.
            """
            s_alive, s_score, l_active = carry

            demon_can_be_hit = jnp.logical_and(
                l_active,
                jnp.logical_and(s_alive[i], state.spawn_anim_timer[i] <= 0),
            )
            demon_right = state.demons_x[i] + self.consts.DEMON_SIZE[1]
            demon_bottom = state.demons_y[i] + self.consts.DEMON_SIZE[0]

            overlaps_horizontally = jnp.logical_and(
                laser_right > state.demons_x[i],
                state.laser_x < demon_right,
            )
            overlaps_vertically = jnp.logical_and(
                state.laser_y < demon_bottom,
                laser_bottom > state.demons_y[i],
            )
            rectangles_overlap = jnp.logical_and(
                overlaps_horizontally,
                overlaps_vertically,
            )
            demon_hit = jnp.logical_and(
                demon_can_be_hit,
                rectangles_overlap,
            )

            new_alive = s_alive.at[i].set(
                jnp.logical_and(s_alive[i], jnp.logical_not(demon_hit))
            )
            new_score = jnp.where(demon_hit, s_score + 10 + state.wave_pattern * 2, s_score)
            new_laser_active = jnp.logical_and(l_active, jnp.logical_not(demon_hit))
            return new_alive, new_score, new_laser_active

        init_carry = (
            state.demons_alive,
            state.score,
            state.laser_active,
        )
        demons_alive, score, laser_active = jax.lax.fori_loop(
            0,
            self.consts.MAX_DEMONS,
            check_demon_collision,
            init_carry,
        )

        demon_killed = jnp.any(
            jnp.logical_and(
                state.demons_alive,
                jnp.logical_not(demons_alive),
            )
        ) # boolean if at least one demon was killed

        killed = state.demons_alive & ~demons_alive # which demon was killed

        # Bomb vs Player
        bomb_width, _ = self._bomb_observation_size(state)
        player_right = state.player_x + self.consts.PLAYER_SIZE[1]
        player_bottom = self.consts.PLAYER_Y + self.consts.PLAYER_SIZE[0]
        bomb_right = state.bomb_x + bomb_width
        bomb_top, bomb_bottom = self._bomb_collision_y_bounds(state)
        player_hit = jnp.logical_and(
            state.bomb_active,
            jnp.logical_and(
                bomb_right > state.player_x,
                jnp.logical_and(
                    state.bomb_x < player_right,
                    jnp.logical_and(
                        bomb_top < player_bottom,
                        bomb_bottom > self.consts.PLAYER_Y,
                    ),
                )
            )
        )
        any_player_hit = jnp.any(player_hit)

        bunker_available = state.lives > 0
        lives = jnp.where(
            jnp.logical_and(any_player_hit, bunker_available),
            state.lives - 1,
            state.lives,
        )
        game_over = jnp.logical_or(
            state.game_over,
            jnp.logical_and(any_player_hit, jnp.logical_not(bunker_available)),
        )
        bomb_active = jnp.where(
            any_player_hit,
            jnp.zeros_like(state.bomb_active),
            state.bomb_active,
        )
        bomb_burst_step = jnp.where(
            any_player_hit,
            self.consts.BOMB_BURST_RATES,
            state.bomb_burst_step,
        )
        bomb_burst_length = jnp.where(
            any_player_hit,
            0,
            state.bomb_burst_length,
        )
        bomb_burst_timer = jnp.where(
            any_player_hit,
            0,
            state.bomb_burst_timer,
        )

        # If player hit, start explosion
        player_exploding = jnp.logical_or(state.player_exploding, any_player_hit)
        explosion_timer = jnp.where(
            any_player_hit,
            self.consts.PLAYER_DEATH_ANIMATION_DURATION,
            state.explosion_timer,
        )

        teleport_busy = state.demon_teleport_timer > 0
        reset_teleport_for_kill = jnp.logical_and(
            demon_killed,
            jnp.logical_not(teleport_busy),
        )

        state = state.replace(
            demons_alive=demons_alive,
            demon_status=jnp.where(killed, DEMON_STATUS_FREE, state.demon_status),
            demon_phase=jnp.where(killed, 0, state.demon_phase),
            demon_moving_right=jnp.where(killed, False, state.demon_moving_right),
            demon_moving_down=jnp.where(killed, True, state.demon_moving_down),
            demon_teleport=jnp.where(
                reset_teleport_for_kill,
                jnp.argmax(killed.astype(jnp.int32)),
                state.demon_teleport,
            ),
            demon_teleport_timer=jnp.where(
                reset_teleport_for_kill,
                0,
                state.demon_teleport_timer,
            ),
            demon_death_anim_timer=jnp.where(
                killed,
                self.consts.DEMON_DEATH_ANIMATION_DURATION,
                state.demon_death_anim_timer,
            ),
            score=score,
            laser_active=laser_active,
            lives=lives,
            bomb_active=bomb_active,
            bomb_burst_step=bomb_burst_step,
            bomb_burst_length=bomb_burst_length,
            bomb_burst_timer=bomb_burst_timer,
            player_exploding=player_exploding,
            explosion_timer=explosion_timer,
            game_over=game_over,
        )

        return self._advance_wave_if_complete(state)

    def _advance_wave_if_complete(
        self, state: DemonAttackState
    ) -> DemonAttackState:
        """Advance once every scheduled demon has appeared and been destroyed."""
        wave_finished = jnp.logical_and(
            state.wave_spawned_demons >= self.consts.WAVE_TOTAL_DEMONS,
            jnp.logical_and(
                jnp.logical_not(jnp.any(state.demons_alive)),
                jnp.logical_not(jnp.any(state.demon_death_anim_timer > 0)),
            ),
        )

        return jax.lax.cond(
            wave_finished,
            lambda s: self._advance_wave(s),
            lambda s: s,
            operand=state,
        )

    def _bomb_observation_size(self, state: DemonAttackState) -> Tuple[chex.Array, chex.Array]:
        """Return per-bomb width and height for the current wave's enemy shot type."""
        width = jnp.full_like(
            state.bomb_x,
            self.consts.BOMB_SIZE[1],
            dtype=jnp.int32,
        )
        height = jnp.full_like(
            state.bomb_y,
            self._bomb_height_for_wave(state.wave_pattern),
            dtype=jnp.int32,
        )
        return width, height

    def render(self, state: DemonAttackState) -> jnp.ndarray:
        return self.renderer.render(state)

    def _get_observation(self, state: DemonAttackState):
        player = ObjectObservation.create(
            x=state.player_x,
            y=jnp.array(self.consts.PLAYER_Y),
            width=jnp.array(self.consts.PLAYER_SIZE[1]),
            height=jnp.array(self.consts.PLAYER_SIZE[0]),
        )

        demons = ObjectObservation.create(
            x=state.demons_x,
            y=state.demons_y,
            width=jnp.full_like(state.demons_x, self.consts.DEMON_SIZE[1], dtype=jnp.int32),
            height=jnp.full_like(state.demons_y, self.consts.DEMON_SIZE[0], dtype=jnp.int32),
            active=state.demons_alive
        )

        laser = ObjectObservation.create(
            x=state.laser_x,
            y=state.laser_y,
            width=jnp.array(self.consts.LASER_SIZE[1]),
            height=jnp.array(self.consts.LASER_SIZE[0]),
            active=state.laser_active
        )

        bomb_width, bomb_height = self._bomb_observation_size(state)
        bomb = ObjectObservation.create(
            x=state.bomb_x,
            y=state.bomb_y,
            width=bomb_width,
            height=bomb_height,
            active=state.bomb_active
        )

        return DemonAttackObservation(
            player=player,
            demons=demons,
            laser=laser,
            bomb=bomb,
            score=state.score,
            lives=state.lives
        )

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTION_SET))

    def observation_space(self) -> spaces.Dict:
        object_space = spaces.get_object_space(n=None, screen_size=(self.consts.HEIGHT, self.consts.WIDTH))
        demons_space = spaces.get_object_space(n=self.consts.MAX_DEMONS,
                                               screen_size=(self.consts.HEIGHT, self.consts.WIDTH))

        return spaces.Dict({
            "player": object_space,
            "demons": demons_space,
            "laser": object_space,
            "bomb": spaces.get_object_space(n=self.consts.MAX_BOMBS,
                                            screen_size=(self.consts.HEIGHT, self.consts.WIDTH)),
            "score": spaces.Box(low=0, high=99999, shape=(), dtype=jnp.int32),
            "lives": spaces.Box(low=0, high=self.consts.MAX_BUNKERS, shape=(), dtype=jnp.int32),
        })

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(210, 160, 3),
            dtype=jnp.uint8
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: DemonAttackState) -> DemonAttackInfo:
        return DemonAttackInfo(
            time=state.step_counter,
            wave_number=state.wave_number,
            wave_pattern=state.wave_pattern,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: DemonAttackState, state: DemonAttackState):
        return (state.score - previous_state.score).astype(jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: DemonAttackState) -> bool:
        return jnp.logical_and(
            state.game_over,
            jnp.logical_not(state.player_exploding),
        )

class DemonAttackRenderer(JAXGameRenderer):
    def __init__(self, consts: DemonAttackConstants = None, config: render_utils.RendererConfig = None):
        super().__init__(consts)
        self.consts = consts or DemonAttackConstants()

        if config is None:
            self.config = render_utils.RendererConfig(
                game_dimensions=(210, 160),
                channels=3,
                downscale=None
            )
        else:
            self.config = config

        self.jr = render_utils.JaxRenderingUtils(self.config)

        # 1. Start from (possibly modded) asset config provided via constants
        final_asset_config = list(self.consts.ASSET_CONFIG)
        available_demon_ids = tuple(sorted(
            int(asset["name"].removeprefix("demon_"))
            for asset in final_asset_config
            if asset["name"].startswith("demon_")
        ))
        if not available_demon_ids:
            raise ValueError("ASSET_CONFIG must provide at least one demon sprite group")

        self._demon_sprite_names = tuple(
            f"demon_{demon_id}" for demon_id in available_demon_ids
        )
        if (
            min(self.consts.WAVE_DEMON_TABLE) < 0
            or max(self.consts.WAVE_DEMON_TABLE) >= len(self._demon_sprite_names)
        ):
            raise ValueError(
                "WAVE_DEMON_TABLE uses zero-based indices into the available "
                f"demon sprite groups (0..{len(self._demon_sprite_names) - 1})"
            )
        self._pattern_sprite_indices = jnp.asarray(
            self.consts.WAVE_DEMON_TABLE,
            dtype=jnp.int32,
        )

        # 2. Create procedural assets
        digit_sprites = _create_digit_sprites(self.consts)

        # Update asset config with procedural data
        final_asset_config.append({'name': 'score_digits', 'type': 'procedural', 'data': digit_sprites})

        # 3. Bake assets
        sprite_path = os.path.join(os.path.dirname(__file__), "sprites", "demonattack")
        jax.debug.print(f"Using sprites from: {sprite_path}")
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(final_asset_config, sprite_path)

    def _bomb_type_for_wave(self, wave_pattern: chex.Array) -> chex.Array:
        bomb_types = jnp.asarray(self.consts.WAVE_BOMB_TYPE_TABLE, dtype=jnp.int32)
        pattern = jnp.clip(wave_pattern, 0, bomb_types.shape[0] - 1)
        return bomb_types[pattern]

    def _bomb_sprite_repeats_for_type(self, bomb_type: chex.Array) -> chex.Array:
        return jnp.where(
            bomb_type == BOMB_TYPE_LONG,
            self.consts.LONG_BOMB_HEIGHT_MULTIPLIER,
            1,
        )

    def _blank_frame(self) -> jnp.ndarray:
        blank_color = self.consts.BLANK_SCREEN_COLOR
        if self.config.channels == 1:
            blank_color = (
                int(0.299 * blank_color[0] + 0.587 * blank_color[1] + 0.114 * blank_color[2]),
            )

        blank = jnp.ones((*self.BACKGROUND.shape, self.config.channels), dtype=jnp.uint8)
        return blank * jnp.asarray(blank_color, dtype=jnp.uint8)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: DemonAttackState):
        return jax.lax.cond(
            state.game_frozen,
            self._blank_frame,
            lambda: self._render_gameplay(state),
        )

    def _render_gameplay(self, state: DemonAttackState):
        raster = self.jr.create_object_raster(self.BACKGROUND)

        # Render bunkers as the visible life count.
        bunker_mask = self.SHAPE_MASKS["bunker"]

        def render_bunker(i, r):
            return jax.lax.cond(
                i < state.lives,
                lambda: self.jr.render_at(
                    r,
                    self.consts.BUNKER_X + i * self.consts.BUNKER_SPACING,
                    self.consts.BUNKER_Y,
                    bunker_mask,
                ),
                lambda: r,
            )

        raster = jax.lax.fori_loop(0, self.consts.MAX_BUNKERS, render_bunker, raster)

        # Render player or death animation
        death_masks = self.SHAPE_MASKS["player_death_animation"]
        death_frame = jnp.clip(
            (
                (self.consts.PLAYER_DEATH_ANIMATION_DURATION - state.explosion_timer)
                * death_masks.shape[0]
            )
            // self.consts.PLAYER_DEATH_ANIMATION_DURATION,
            0,
            death_masks.shape[0] - 1,
        )
        death_x = state.player_x + (
            self.consts.PLAYER_SIZE[1] - death_masks.shape[2]
        ) // 2
        death_y = (
            self.consts.PLAYER_Y
            + self.consts.PLAYER_SIZE[0]
            - death_masks.shape[1]
        )
        raster = jax.lax.cond(
            state.player_exploding,
            lambda: self.jr.render_at(
                raster,
                death_x,
                death_y,
                death_masks[death_frame],
            ),
            lambda: self.jr.render_at(
                raster,
                state.player_x,
                self.consts.PLAYER_Y,
                self.SHAPE_MASKS["player"],
            ),
        )

        # Render demons
        raster = self._draw_demons(raster, state)

        # Render laser
        laser_mask = self.SHAPE_MASKS["player_missile"]
        laser_render_x = jax.lax.select(
            state.laser_active,
            state.laser_x,
            state.player_x + self.consts.PLAYER_SIZE[1] // 2
        )
        laser_render_y = jax.lax.select(
            state.laser_active,
            state.laser_y,
            self.consts.PLAYER_Y + self.consts.PLAYER_LASER_DEPTH
        )
        raster = jax.lax.cond(
            jnp.logical_or(state.laser_active, jnp.logical_not(state.player_exploding)),
            lambda: self.jr.render_at(raster, laser_render_x, laser_render_y, laser_mask),
            lambda: raster,
        )

        # Render enemy shot particles.
        bomb_mask = self.SHAPE_MASKS["projectile_demon"]
        bomb_type = self._bomb_type_for_wave(state.wave_pattern)
        bomb_sprite_repeats = self._bomb_sprite_repeats_for_type(bomb_type)
        visible_bomb_repeats, bomb_repeat_offsets = _bomb_visible_repeat_window(
            state,
            self.consts,
            bomb_type,
        )

        def render_bomb(i, r):
            def render_bomb_repeat(j, rr):
                render_y = (
                    state.bomb_y[i]
                    + (j - bomb_repeat_offsets[i]) * self.consts.BOMB_SIZE[0]
                )
                return jax.lax.cond(
                    jnp.logical_and(
                        j < visible_bomb_repeats[i],
                        render_y < self.consts.BUNKER_Y - self.consts.BOMB_SIZE[0],
                    ),
                    lambda: self.jr.render_at(
                        rr,
                        state.bomb_x[i],
                        render_y,
                        bomb_mask,
                    ),
                    lambda: rr,
                )

            return jax.lax.cond(
                jnp.logical_and(state.bomb_active[i], jnp.logical_not(state.player_exploding)),
                lambda: jax.lax.fori_loop(0, bomb_sprite_repeats, render_bomb_repeat, r),
                lambda: r,
            )

        raster = jax.lax.fori_loop(0, self.consts.MAX_BOMBS, render_bomb, raster)

        # Render Score
        score_digits = self.jr.int_to_digits(state.score, max_digits=4)
        digit_masks = self.SHAPE_MASKS["score_digits"]

        is_single_digit = state.score < 10
        is_double_digit = jnp.logical_and(state.score >= 10, state.score < 100)
        is_triple_digit = jnp.logical_and(state.score >= 100, state.score < 1000)

        start_index = jax.lax.select(
            is_single_digit,
            3,
            jax.lax.select(is_double_digit, 2, jax.lax.select(is_triple_digit, 1, 0)),
        )
        num_to_render = jax.lax.select(
            is_single_digit,
            1,
            jax.lax.select(is_double_digit, 2, jax.lax.select(is_triple_digit, 3, 4)),
        )

        raster = self.jr.render_label_selective(
            raster,
            70,
            10,
            score_digits,
            digit_masks,
            start_index,
            num_to_render,
            spacing=8,
        )

        frame = self.jr.render_from_palette(raster, self.PALETTE)
        death_elapsed = (
            self.consts.PLAYER_DEATH_ANIMATION_DURATION - state.explosion_timer
        )
        flash_frames_left = jnp.clip(
            self.consts.PLAYER_DEATH_FLASH_DURATION - death_elapsed,
            0,
            self.consts.PLAYER_DEATH_FLASH_DURATION,
        )
        flash_intensity = (
            jnp.array(255, dtype=jnp.int32) * flash_frames_left
        ) // self.consts.PLAYER_DEATH_FLASH_DURATION
        flash_color = jnp.asarray(flash_intensity, dtype=jnp.uint8)
        return jnp.where(
            jnp.logical_and(
                jnp.logical_and(state.player_exploding, flash_frames_left > 0),
                jnp.all(frame == 0, axis=-1, keepdims=True),
            ),
            flash_color,
            frame,
        )

    def _draw_demons(self, raster, state):
        # Animation cycle: 4 frames, each for 8 steps. Total = 32 steps
        demon_anim_idx = (state.step_counter % 32) // 8

        # Sprite selection is per pattern, unlike pair-shared difficulty tables.
        pattern_index = jnp.clip(
            state.wave_pattern,
            0,
            len(self.consts.WAVE_DEMON_TABLE) - 1,
        )
        sprite_group_idx = self._pattern_sprite_indices[pattern_index]

        demon_masks = jax.lax.switch(
            sprite_group_idx,
            [
                lambda sprite_name=sprite_name: self.SHAPE_MASKS[sprite_name]
                for sprite_name in self._demon_sprite_names
            ],
        )

        demon_mask = demon_masks[demon_anim_idx]

        spawn_anim_total = self.consts.SPAWN_ANIM_FRAMES * self.consts.SPAWN_ANIM_FRAME_DURATION
        ids = jnp.arange(self.consts.MAX_DEMONS)
        spacing = (
            self.consts.DEMON_MAX_X - self.consts.DEMON_MIN_X
        ) // (self.consts.MAX_DEMONS + 1)
        spawn_target_x = self.consts.DEMON_MIN_X + (ids + 1) * spacing

        def render_demon(i, r):
            is_spawning = state.spawn_anim_timer[i] > 0
            is_dying = state.demon_death_anim_timer[i] > 0

            elapsed = spawn_anim_total - state.spawn_anim_timer[i]
            spawn_frame = jnp.clip(
                elapsed // self.consts.SPAWN_ANIM_FRAME_DURATION,
                0,
                self.consts.SPAWN_ANIM_FRAMES - 1,
            )

            spawn_left_mask = self.SHAPE_MASKS["enemy_spawn_left"][spawn_frame]
            spawn_right_mask = self.SHAPE_MASKS["enemy_spawn_right"][spawn_frame]

            def render_spawn():
                spawn_max_x = jnp.minimum(
                    jnp.array(self.consts.DEMON_MAX_X, dtype=jnp.int32),
                    jnp.array(self.consts.WIDTH - self.consts.SPAWN_ANIM_WIDTH, dtype=jnp.int32),
                )

                target_x = jnp.clip(
                    spawn_target_x[i] - (self.consts.SPAWN_ANIM_WIDTH - self.consts.DEMON_SIZE[1]) // 2,
                    self.consts.DEMON_MIN_X,
                    spawn_max_x,
                )

                last_step = jnp.maximum(
                    jnp.array(spawn_anim_total - 1, dtype=jnp.int32),
                    jnp.array(1, dtype=jnp.int32),
                )

                left_render_x = (
                                        self.consts.DEMON_MIN_X * (last_step - elapsed)
                                        + target_x * elapsed
                                ) // last_step

                right_render_x = (
                                         spawn_max_x * (last_step - elapsed)
                                         + target_x * elapsed
                                 ) // last_step

                spawn_raster = self.jr.render_at_clipped(
                    r,
                    left_render_x,
                    state.demons_y[i],
                    spawn_left_mask,
                )
                return self.jr.render_at_clipped(
                    spawn_raster,
                    right_render_x,
                    state.demons_y[i],
                    spawn_right_mask,
                )

            def render_normal():
                return self.jr.render_at_clipped(
                    r,
                    state.demons_x[i],
                    state.demons_y[i],
                    demon_mask,
                )

            def render_death():
                death_masks = self.SHAPE_MASKS["enemy_death_animation"]
                death_frame = jnp.clip(
                    (
                        (self.consts.DEMON_DEATH_ANIMATION_DURATION - state.demon_death_anim_timer[i])
                        * death_masks.shape[0]
                    )
                    // self.consts.DEMON_DEATH_ANIMATION_DURATION,
                    0,
                    death_masks.shape[0] - 1,
                )
                return self.jr.render_at_clipped(
                    r,
                    state.demons_x[i],
                    state.demons_y[i],
                    death_masks[death_frame],
                )

            return jax.lax.cond(
                state.demons_alive[i],
                lambda: jax.lax.cond(
                    is_spawning,
                    render_spawn,
                    render_normal,
                ),
                lambda: jax.lax.cond(is_dying, render_death, lambda: r),
            )

        return jax.lax.fori_loop(0, self.consts.MAX_DEMONS, render_demon, raster)
