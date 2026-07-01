import os

from functools import partial

from enum import IntEnum
import chex
from flax import struct
from typing import Tuple

import jax
import jax.numpy as jnp

import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action

class Level(IntEnum):
    LEVEL_1 = 1
    LEVEL_2 = 2
    LEVEL_3 = 3

class Enemy(IntEnum):
    NONE = 0
    MAD_DOCTOR_1 = 1
    CONDOR = 2
    
class TowerLevelType(IntEnum):
    FULL = 0
    MIDDLE_CUT = 1
    SIDE_CUTS = 2

class PlayerStableStates(IntEnum):
    NEUTRAL = 0
    HALF_PULL_UP = 1
    PULL_UP = 2
    REACHING = 3

@chex.dataclass
class LevelState:
    condor_active: chex.Array
    mad_doctor_active: chex.Array
    score_increment: chex.Array
    next_enemy: chex.Array
    pause_game: chex.Array
    
    @classmethod
    def new(cls, level: Level) -> chex.dataclass:
        return cls(
            condor_active = jnp.array(False),
            mad_doctor_active = jnp.array(False),
            score_increment = jnp.array(CrazyClimberConstants.SCORE_BASE_VALUE * level),
            next_enemy = jnp.array(
                jax.lax.switch(
                    level,
                    [
                        lambda: Enemy.NONE,
                        lambda: Enemy.CONDOR
                    ]
                )),
            pause_game = jnp.array(False)
        )
    
@chex.dataclass
class PlayerMoveState:
    main_state: PlayerStableStates 
    sub_step: int 
    side_step: int 
    hand_dir: int
    pos_x: int
    falling_count: int
    egg_animation_count: chex.Array
    should_fall: bool
    flicker: chex.Array

    @classmethod
    def new(cls):
        return cls(
            main_state=PlayerStableStates.NEUTRAL,
            sub_step=0,
            side_step=0,
            hand_dir=1,
            pos_x=0,
            falling_count=0,
            egg_animation_count=0,
            should_fall=False,
            flicker=jnp.array(False)
        )
    
@chex.dataclass
class TowerState:
    tower_step: int
    windows: jnp.ndarray
    spawn_probability: float
    is_falling: bool
    lowest_level: int
    levels: jnp.ndarray

    @classmethod
    def new(cls, key):
        blind_left = jnp.zeros((11, 3))
        blind_dirs_left = jax.random.choice(key, jnp.array([0, 1]), (11, 3), p=jnp.array([0.8, 0.2]))
        windows_left = jnp.stack([blind_left, blind_dirs_left], axis=2)
        windows = jnp.concatenate([windows_left, jnp.fliplr(windows_left)], axis=1)

        return cls(
            tower_step=0,
            windows=windows,
            spawn_probability=0.2,
            is_falling=False,
            lowest_level=0,
            levels=CrazyClimberConstants.TOWER1
        )

class HeliFlyAwayStates(IntEnum):
    NORMAL = 0
    ONE_ROW = 1
    BONUS_DEC = 2
    NOTHING = 3
    DONE = 4

@chex.dataclass
class HelicopterState:
    fly_away_state: HeliFlyAwayStates
    pos_x: chex.Array
    pos_y: chex.Array
    x_dir: chex.Array
    y_dir: chex.Array
    x_movement_unlocked: bool
    fly_away_step: chex.Array
    step: chex.Array

    @classmethod
    def new(cls):
        return cls(
            fly_away_state=HeliFlyAwayStates.NORMAL,
            pos_x=CrazyClimberConstants.HELICOPTER_SPAWN[0],
            pos_y=CrazyClimberConstants.HELICOPTER_SPAWN[1],
            x_dir=jnp.array(-1),
            y_dir=jnp.array(1),
            x_movement_unlocked=jnp.array(False),
            fly_away_step=jnp.array(0),
            step=jnp.array(0),
        )


@chex.dataclass
class BirdState:
    drop_egg: chex.Array
    pos_x: chex.Array
    pos_y: chex.Array
    dir: chex.Array
    stop: chex.Array
    egg_state: chex.dataclass

    @classmethod
    def new(cls):
        return cls(
            drop_egg=jnp.array(True),
            pos_x=CrazyClimberConstants.BIRD_SIZE[1],
            pos_y=CrazyClimberConstants.BIRD_Y,
            dir=jnp.array(1),
            stop=jnp.array(False),
            egg_state=EggState.new()
        )

@chex.dataclass
class EggState:
    pos_x: chex.Array
    pos_y: chex.Array
    dir: chex.Array
    vel: chex.Array
    flicker: chex.Array

    @classmethod
    def new(cls):
        return cls(
            pos_x=jnp.array(CrazyClimberConstants.BIRD_SIZE[1]),
            pos_y=jnp.array(CrazyClimberConstants.BIRD_Y),
            dir=jnp.array(1),
            vel=jnp.array(8),
            flicker=jnp.array(False)
        )

@chex.dataclass
class FlowerpotEnemyState:
    active: chex.Array
    phase: chex.Array
    phase_steps: chex.Array
    window_row: chex.Array
    window_col: chex.Array
    cycle_row: chex.Array
    drop_x_offset: chex.Array
    drop_type: chex.Array

    @classmethod
    def new(cls, active: bool, phase_steps: int, window_row, window_col, cycle_row, drop_type):
        return cls(
            active=jnp.array(active),
            phase=jnp.array(0, dtype=jnp.int32),
            phase_steps=jnp.array(phase_steps, dtype=jnp.int32),
            window_row=window_row,
            window_col=window_col,
            cycle_row=cycle_row,
            drop_x_offset=jnp.array(0, dtype=jnp.int32),
            drop_type=drop_type,
        )

class CrazyClimberState(struct.PyTreeNode):
    key: chex.PRNGKey
    step_counter: chex.Array

    score: chex.Array
    reached_apex: chex.Array
    bonus: chex.Array
    lifes: chex.Array

    player_move_state: PlayerMoveState
    bird_state: BirdState
    flowerpot_enemy_state: FlowerpotEnemyState
    tower_state: TowerState
    helicopter_state: HelicopterState
    level_state: chex.Array

    climbed_floors: chex.Array

class CrazyClimberObservation(struct.PyTreeNode):
    pass

class CrazyClimberInfo(struct.PyTreeNode):
    pass

def _create_block_sprite(color: tuple[int, int, int, int], shape: tuple[int, int]) -> jnp.ndarray:
    return jnp.tile(jnp.array(color, dtype=jnp.uint8), (*shape[:2], 1))

def _create_block_sprite_with_padding(color: tuple[int, int, int, int], shape: tuple[int, int], wanted_shape: tuple[int, int]) -> jnp.ndarray:
    padded_box = jnp.zeros((*wanted_shape, 4), dtype=jnp.uint8)
    sprite = jnp.tile(jnp.array(color, dtype=jnp.uint8), (*shape[:2], 1))
    padded_sprite = padded_box.at[0:shape[0], 0:shape[1]].set(sprite)
    return padded_sprite


def _get_default_asset_config() -> tuple:
    wall_sprite = _create_block_sprite((0, 0, 148, 255), (169, 4))
    ceiling_sprite = _create_block_sprite((0, 48, 100, 255), (5, 80))
    floor_sprite = _create_block_sprite((0, 0, 148, 255), (1, 80))
    window_sprites = jnp.array([
        _create_block_sprite_with_padding((0, 0, 148,   0), (2, 8), (8, 8)),
        _create_block_sprite_with_padding((0, 0, 148, 255), (2, 8), (8, 8)),
        _create_block_sprite_with_padding((0, 0, 148, 255), (3, 8), (8, 8)),
        _create_block_sprite_with_padding((0, 0, 148, 255), (4, 8), (8, 8)),
        _create_block_sprite_with_padding((0, 0, 148, 255), (5, 8), (8, 8)),
        _create_block_sprite_with_padding((0, 0, 148, 255), (6, 8), (8, 8)),
        _create_block_sprite_with_padding((0, 0, 148, 255), (8, 8), (8, 8)),
    ])

    return (
        {'name': 'background', 'type': 'background', 'file': 'misc/background.npy'},
        {'name': 'digits', 'type': 'digits', 'pattern': 'numbers/score_{}.npy'},
        {'name': 'life', 'type': 'single', 'file': 'misc/life.npy'},
        {'name': 'player_upwards_left_group', 'type': 'group', 'files': [
            'player/neutral_0.npy',
            'player/upwards/left_first/neutral_2.npy',
            'player/upwards/left_first/neutral_3.npy',
            'player/upwards/left_first/neutral_4.npy',
            'player/upwards/left_first/half_pull_up_0.npy',
            'player/upwards/left_first/half_pull_up_2.npy',
            'player/upwards/left_first/half_pull_up_3.npy',
            'player/upwards/left_first/half_pull_up_4.npy',
            'player/upwards/pull_up_0.npy',
            'player/upwards/pull_up_1.npy',
            'player/upwards/pull_up_4.npy',
            'player/upwards/pull_up_7.npy',
            ]},
        {'name': 'player_upwards_right_group', 'type': 'group', 'files': [
            'player/neutral_0.npy',
            'player/upwards/right_first/neutral_2.npy',
            'player/upwards/right_first/neutral_3.npy',
            'player/upwards/right_first/neutral_4.npy',
            'player/upwards/right_first/half_pull_up_0.npy',
            'player/upwards/right_first/half_pull_up_2.npy',
            'player/upwards/right_first/half_pull_up_3.npy',
            'player/upwards/right_first/half_pull_up_4.npy',
            'player/upwards/pull_up_0.npy',
            'player/upwards/pull_up_1.npy',
            'player/upwards/pull_up_4.npy',
            'player/upwards/pull_up_7.npy',
            ]},

        {'name': 'player_neutral_left_group', 'type': 'group', 'files': [
            'player/neutral_0.npy',
            'player/sideways/left/neutral_5.npy',
            'player/sideways/left/neutral_9.npy',
            ]},
        {'name': 'player_neutral_right_group', 'type': 'group', 'files': [
            'player/neutral_0.npy',
            'player/sideways/right/neutral_5.npy',
            'player/sideways/right/neutral_9.npy',
            ]},

        {'name': 'player_left_pull_up_left_group', 'type': 'group', 'files': [
            'player/upwards/left_first/half_pull_up_0.npy',
            'player/sideways/left/left_up/half_pull_up_5.npy',
            'player/sideways/left/left_up/half_pull_up_9.npy',
            ]},
        {'name': 'player_left_pull_up_right_group', 'type': 'group', 'files': [
            'player/upwards/left_first/half_pull_up_0.npy',
            'player/sideways/right/left_up/half_pull_up_5.npy',
            'player/sideways/right/left_up/half_pull_up_9.npy',
            ]},

        {'name': 'player_pull_up_left_group', 'type': 'group', 'files': [
            'player/upwards/pull_up_0.npy',
            'player/sideways/left/pull_up_5.npy',
            'player/sideways/left/pull_up_9.npy',
            ]},
        {'name': 'player_pull_up_right_group', 'type': 'group', 'files': [
            'player/upwards/pull_up_0.npy',
            'player/sideways/right/pull_up_5.npy',
            'player/sideways/right/pull_up_9.npy',
            ]},

        {'name': 'player_right_pull_up_left_group', 'type': 'group', 'files': [
            'player/upwards/right_first/half_pull_up_0.npy',
            'player/sideways/left/right_up/half_pull_up_5.npy',
            'player/sideways/left/right_up/half_pull_up_9.npy',
            ]},
        {'name': 'player_right_pull_up_right_group', 'type': 'group', 'files': [
            'player/upwards/right_first/half_pull_up_0.npy',
            'player/sideways/right/right_up/half_pull_up_5.npy',
            'player/sideways/right/right_up/half_pull_up_9.npy',
            ]},
        {'name': 'player_reaching_group', 'type': 'group', 'files': [
            'player/reaching/left.npy',
            'player/reaching/right.npy',
        ]},

        {'name': 'flowerpot_thrower_group', 'type': 'group', 'files': [
            'flowerpot_enemy/red_enemy/red_enemy_1.npy',
            'flowerpot_enemy/red_enemy/red_enemy_2.npy',
            'flowerpot_enemy/red_enemy/red_enemy_3.npy',
            'flowerpot_enemy/red_enemy/red_enemy_4.npy',
            'flowerpot_enemy/red_enemy/red_enemy_5.npy',
            ]},
        {'name': 'flowerpot_drop_group', 'type': 'group', 'files': [
            'flowerpot_enemy/blue_drop/blue_drop_1.npy',
            'flowerpot_enemy/blue_drop/blue_drop_2.npy',
            'flowerpot_enemy/blue_drop/blue_drop_3.npy',
            'flowerpot_enemy/blue_drop/blue_drop_4.npy',
            'flowerpot_enemy/blue_drop/blue_drop_5.npy',
            'flowerpot_enemy/blue_drop/blue_drop_6.npy',
            'flowerpot_enemy/blue_drop/blue_drop_7.npy',
            'flowerpot_enemy/blue_drop/blue_drop_8.npy',
            'flowerpot_enemy/blue_drop/blue_drop_9.npy',
            'flowerpot_enemy/blue_drop/blue_drop_10.npy',
            'flowerpot_enemy/blue_drop/blue_drop_11.npy',
            'flowerpot_enemy/blue_drop/blue_drop_12.npy',
            'flowerpot_enemy/blue_drop/blue_drop_13.npy',
            'flowerpot_enemy/blue_drop/blue_drop_14.npy',
            'flowerpot_enemy/blue_drop/blue_drop_15.npy',
            'flowerpot_enemy/blue_drop/blue_drop_16.npy',
            'flowerpot_enemy/blue_drop/blue_drop_17.npy',
            'flowerpot_enemy/blue_drop/blue_drop_18.npy',
            'flowerpot_enemy/blue_drop/blue_drop_19.npy',
            'flowerpot_enemy/blue_drop/blue_drop_20.npy',
            'flowerpot_enemy/blue_drop/blue_drop_21.npy',
            'flowerpot_enemy/blue_drop/blue_drop_22.npy',
            'flowerpot_enemy/blue_drop/blue_drop_23.npy',
            'flowerpot_enemy/blue_drop/blue_drop_24.npy',
            'flowerpot_enemy/blue_drop/blue_drop_25.npy',
            'flowerpot_enemy/blue_drop/blue_drop_26.npy',
            'flowerpot_enemy/blue_drop/blue_drop_27.npy',
            'flowerpot_enemy/purple_drop/purple_drop_1.npy',
            'flowerpot_enemy/purple_drop/purple_drop_2.npy',
            'flowerpot_enemy/purple_drop/purple_drop_3.npy',
            'flowerpot_enemy/purple_drop/purple_drop_4.npy',
            'flowerpot_enemy/purple_drop/purple_drop_5.npy',
            'flowerpot_enemy/purple_drop/purple_drop_6.npy',
            'flowerpot_enemy/purple_drop/purple_drop_7.npy',
            'flowerpot_enemy/purple_drop/purple_drop_8.npy',
            'flowerpot_enemy/purple_drop/purple_drop_9.npy',
            'flowerpot_enemy/purple_drop/purple_drop_10.npy',
            'flowerpot_enemy/purple_drop/purple_drop_11.npy',
            'flowerpot_enemy/purple_drop/purple_drop_12.npy',
            'flowerpot_enemy/purple_drop/purple_drop_13.npy',
            'flowerpot_enemy/purple_drop/purple_drop_14.npy',
            'flowerpot_enemy/purple_drop/purple_drop_15.npy',
            'flowerpot_enemy/purple_drop/purple_drop_16.npy',
            'flowerpot_enemy/yellow_drop/yellow_drop_1.npy',
            'flowerpot_enemy/yellow_drop/yellow_drop_2.npy',
            'flowerpot_enemy/yellow_drop/yellow_drop_3.npy',
            'flowerpot_enemy/yellow_drop/yellow_drop_4.npy',
            'flowerpot_enemy/yellow_drop/yellow_drop_5.npy',
            'flowerpot_enemy/yellow_drop/yellow_drop_6.npy',
            'flowerpot_enemy/yellow_drop/yellow_drop_7.npy',
            'flowerpot_enemy/yellow_drop/yellow_drop_8.npy',
            'flowerpot_enemy/yellow_drop/yellow_drop_9.npy',
            'flowerpot_enemy/yellow_drop/yellow_drop_10.npy',
            'flowerpot_enemy/yellow_drop/yellow_drop_11.npy',
            'flowerpot_enemy/yellow_drop/yellow_drop_12.npy',
            'flowerpot_enemy/yellow_drop/yellow_drop_13.npy',
            'flowerpot_enemy/yellow_drop/yellow_drop_14.npy',
            'flowerpot_enemy/yellow_drop/yellow_drop_15.npy',
            'flowerpot_enemy/yellow_drop/yellow_drop_16.npy',
            'flowerpot_enemy/yellow_drop/yellow_drop_17.npy',
            'flowerpot_enemy/yellow_drop/yellow_drop_18.npy',
            'flowerpot_enemy/yellow_drop/yellow_drop_19.npy',
            'flowerpot_enemy/yellow_drop/yellow_drop_20.npy',
            'flowerpot_enemy/yellow_drop/yellow_drop_21.npy',
            'flowerpot_enemy/yellow_drop/yellow_drop_22.npy',
            'flowerpot_enemy/yellow_drop/yellow_drop_23.npy',
            'flowerpot_enemy/yellow_drop/yellow_drop_24.npy',
            'flowerpot_enemy/yellow_drop/yellow_drop_25.npy',
            'flowerpot_enemy/yellow_drop/yellow_drop_26.npy',
            'flowerpot_enemy/yellow_drop/yellow_drop_27.npy',
            'flowerpot_enemy/yellow_drop/yellow_drop_28.npy',
            'flowerpot_enemy/yellow_drop/yellow_drop_29.npy',
            'flowerpot_enemy/yellow_drop/yellow_drop_30.npy',
            'flowerpot_enemy/yellow_drop/yellow_drop_31.npy',
            'flowerpot_enemy/yellow_drop/yellow_drop_32.npy',
            'flowerpot_enemy/yellow_drop/yellow_drop_33.npy',
            'flowerpot_enemy/yellow_drop/yellow_drop_34.npy',
            'flowerpot_enemy/yellow_drop/yellow_drop_35.npy',
            'flowerpot_enemy/yellow_drop/yellow_drop_36.npy',
            'flowerpot_enemy/yellow_drop/yellow_drop_37.npy',
            'flowerpot_enemy/yellow_drop/yellow_drop_38.npy',
            ]},

        {'name': 'wall', 'type': 'procedural', 'data': wall_sprite},
        {'name': 'ceiling', 'type': 'procedural', 'data': ceiling_sprite},
        {'name': 'floor', 'type': 'procedural', 'data': floor_sprite},
        {'name': 'window_blind_group', 'type': 'procedural', 'data': window_sprites},
        {'name': 'ceiling', 'type': 'procedural', 'data': ceiling_sprite},
        {'name': 'helicopter_right', 'type': 'group', 'files': [
            'helicopter/right/0+2.npy',
            'helicopter/right/1.npy',
            'helicopter/right/3.npy',
        ]},
        {'name': 'helicopter_left', 'type': 'group', 'files': [
            'helicopter/left/0+2.npy',
            'helicopter/left/1.npy',
            'helicopter/left/3.npy',
        ]},
        {'name': 'window_blind_group', 'type': 'procedural', 'data': window_sprites},

        {'name': 'bird_left', 'type': 'group', 'files': [
            'bird/left/0.npy',
            'bird/left/4.npy',
            'bird/left/8.npy',
            'bird/left/12.npy',
            'bird/left/16.npy',
        ]},
        {'name': 'bird_right', 'type': 'group', 'files': [
            'bird/right/0.npy',
            'bird/right/4.npy',
            'bird/right/8.npy',
            'bird/right/12.npy',
            'bird/right/16.npy',
        ]},

        {'name': 'egg_falling', 'type': 'group', 'files': [
            'egg/0.npy',
            'egg/1.npy',
            'egg/2.npy',
            'egg/3.npy',
            'egg/4.npy',
            'egg/5.npy',
            'egg/6.npy',
            'egg/7.npy',
            'egg/8.npy',
            'egg/9.npy',
            'egg/10.npy',
        ]}
    )

class CrazyClimberConstants(struct.PyTreeNode):
    BACKGROUND_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(0, 0, 0))
    ASSET_CONFIG: tuple = _get_default_asset_config()

    WIDTH: int = struct.field(pytree_node=False, default=160)
    HEIGHT: int = struct.field(pytree_node=False, default=210)

    PLAYER_Y: int = struct.field(pytree_node=False, default=160)
    PLAYER_POSSIBLE_X: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([40, 46, 52, 58, 64, 72, 80, 86, 92, 98, 104]))
    PLAYER_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default=(27, 16))
    PLAYER_UPWARDS_SPRITE_SEQUENCE: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([0, 0, 1, 2, 3, 4, 4, 5, 6, 7, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11]))
    PLAYER_SIDEWAYS_SPRITE_SEQUENCE: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([0, 0, 0, 0, 1, 1, 1, 1, 3, 3, 3, 3]))

    TOWER_POSSIBLE_SPRITE_CLIP: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array([0, 4, 7, 10]))
    PLAYER_Y: int = struct.field(pytree_node=False, default=156)
    PLAYER_POSSIBLE_X: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda:jnp.array([40, 46, 52, 58, 64, 72, 80, 86, 92, 98, 104]))
    TOWER_POSSIBLE_SPRITE_CLIP: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda:jnp.array([0, 4, 7, 10]))

    HELICOPTER_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default=(26, 30))
    HELICOPTER_SPAWN: Tuple[int, int] = struct.field(pytree_node=False, default=(35,67))
    HELICOPTER_BORDERS: Tuple[int, int] = struct.field(pytree_node=False, default=(10, 35+HELICOPTER_SIZE.default[1]))
    HELICOPTER_BORDERS_X: Tuple[int, int] = struct.field(pytree_node=False, default=(8, 110))
    HELICOPTER_BORDERS_Y: Tuple[int, int] = struct.field(pytree_node=False, default=(128,69))
    HELICOPTER_SPAWN_HEIGHT: int = struct.field(pytree_node=False, default=20000) # TODO: Should be set to max tower height when merged, maybe rename?
    HELICOPTER_MOVEMENT_BEGIN: int = struct.field(pytree_node=False, default=116) # TODO: value is not pixel perfect yet
    HELICOPTER_MAX_STEPS: int = struct.field(pytree_node=False, default=1540) #TODO: not precise value yet
    HELICOPTER_SEQUENCE: chex.Array = struct.field(pytree_node=False, default_factory=lambda:jnp.array([0,1,0,2]))
    HELICOPTER_SKIDS_SIZE: int = struct.field(pytree_node=False, default=22)

    SCORE_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(236, 236, 236))

    FLOWERPOT_SCORE_RANGES: jnp.ndarray = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array(
            [
                [2500, 5000],
                [10000, 12500],
            ],
            dtype=jnp.int32,
        ),
    )
    FLOWERPOT_MIN_CLIMBED_FLOORS: int = struct.field(pytree_node=False, default=25)
    FLOWERPOT_PHASE_0_STEPS: int = struct.field(pytree_node=False, default=32)
    FLOWERPOT_DROP_TYPE_COUNT: int = struct.field(pytree_node=False, default=3)
    FLOWERPOT_DROP_LOOP_LENGTHS: jnp.ndarray = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array([22, 11, 33], dtype=jnp.int32),
    )
    FLOWERPOT_DROP_SPRITE_OFFSETS: jnp.ndarray = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array([0, 27, 43], dtype=jnp.int32),
    )
    FLOWERPOT_DROP_DEFLECT_X_OFFSET: int = struct.field(pytree_node=False, default=12)
    FLOWERPOT_DROP_HITBOX_WIDTH: int = struct.field(pytree_node=False, default=7)
    FLOWERPOT_DROP_HITBOX_HEIGHT: int = struct.field(pytree_node=False, default=12)
    FLOWERPOT_CYCLE_STEPS_BY_ROW: jnp.ndarray = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array(
            [0, 0, 0, 0, 140, 128, 116, 104, 92],
            dtype=jnp.int32,
        ),
    )
    FLOWERPOT_CANDIDATE_WINDOWS: jnp.ndarray = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array(
            [
                [4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5],
                [6, 0], [6, 1], [6, 2], [6, 3], [6, 4], [6, 5],
            ],
            dtype=jnp.int32,
        ),
    )

    TOWER1 = jnp.concat([
                jnp.repeat(TowerLevelType.MIDDLE_CUT, 5),
                jnp.repeat(TowerLevelType.FULL, 9),
                jnp.repeat(TowerLevelType.MIDDLE_CUT, 13),
                jnp.repeat(TowerLevelType.FULL, 10),
                jnp.repeat(TowerLevelType.SIDE_CUTS, 10),
                jnp.repeat(TowerLevelType.FULL, 15),
                jnp.repeat(TowerLevelType.MIDDLE_CUT, 8),
                jnp.repeat(TowerLevelType.FULL, 15),
                jnp.repeat(TowerLevelType.MIDDLE_CUT, 8),
                jnp.repeat(TowerLevelType.FULL, 20),
                jnp.repeat(TowerLevelType.SIDE_CUTS, 12),
                jnp.repeat(TowerLevelType.FULL, 12),
                jnp.repeat(TowerLevelType.MIDDLE_CUT, 8),
                jnp.repeat(TowerLevelType.FULL, 18),
            ]
        )

    PIXEL_MASK_ONE_ROW = PIXEL_MASK_ONE_ROW = jnp.zeros((13, 13), dtype=bool).at[-1, :].set(True).reshape(169, 1)
    PIXEL_MASK_NOTHING = jnp.zeros((169, 1), dtype=bool)


    BIRD_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default=(12, 15))
    BIRD_Y: int = struct.field(pytree_node=False, default=49)
    BIRD_BORDERS: Tuple[int, int] = struct.field(pytree_node=False, default=(10, 35+BIRD_SIZE.default[1]))
    BIRD_SPAWN_THRESHOLD: int = struct.field(pytree_node=False, default=5000) # should be 5000 for final version
    BIRD_DESPAWN_THRESHOLD: int = struct.field(pytree_node=False, default=7500) # should be 8500 for final version
    BIRD_POSSIBLE_STEPS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([0, 4, 4, 4, 7, 7, 7, 10, 10, 10]))
    BIRD_SEQUENCE: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([0, 1, 2, 3, 4, 4, 3, 2, 1, 0]))

    EGG_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default=(8, 7))
    EGG_BORDER_BOTTOM: int = struct.field(pytree_node=False, default=210-EGG_SIZE.default[0]*2)
    EGG_FLICKER: int = struct.field(pytree_node=False, default=130)

    SCORE_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(236, 236, 236))
    SCORE_BASE_VALUE: int = struct.field(pytree_node=False, default=100)

    BONUS_DECREASE_THRESHOLD: int = struct.field(pytree_node=False, default=1229)
    BONUS_DECREASE_INTERVAL: int = struct.field(pytree_node=False, default=600)


class JaxCrazyClimber(JaxEnvironment[CrazyClimberState, CrazyClimberObservation, CrazyClimberInfo, CrazyClimberConstants]):
    ACTION_SET: jnp.ndarray = jnp.array(
        [Action.NOOP,
         Action.UP,
         Action.RIGHT,
         Action.LEFT,
         Action.DOWN,
         Action.UPRIGHT,
         Action.UPLEFT,
         Action.DOWNRIGHT,
         Action.DOWNLEFT],
        dtype=jnp.int32,
    )

    def __init__(self, consts: CrazyClimberConstants = None):
        self.consts = consts or CrazyClimberConstants()
        super().__init__(self.consts)
        self.renderer = self.CrazyClimberRenderer(consts)

    def reset(self, key: chex.PRNGKey = jax.random.PRNGKey(42)) -> (CrazyClimberObservation, CrazyClimberState):
        state_key, _step_key = jax.random.split(key)
        state = CrazyClimberState(
            key=state_key,
            step_counter=jnp.array(0).astype(jnp.int32),
            score=jnp.array(0).astype(jnp.int32),
            bonus=jnp.array(10000).astype(jnp.int32),
            lifes=jnp.array(5),
            reached_apex=jnp.array(False),
            player_move_state=PlayerMoveState.new(),
            tower_state=TowerState.new(state_key),

            bird_state=BirdState.new(),
            level_state=LevelState.new(Level.LEVEL_1),

            climbed_floors=jnp.array(0, dtype=jnp.int32),
            flowerpot_enemy_state=FlowerpotEnemyState.new(
                False,
                0,
                jnp.array(0, dtype=jnp.int32),
                jnp.array(0, dtype=jnp.int32),
                jnp.array(0, dtype=jnp.int32),
                jnp.array(0, dtype=jnp.int32),
            ),
            helicopter_state=HelicopterState.new(),
        )
        initial_obs = self._get_observation(state)

        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: CrazyClimberState, action: chex.Array) -> (CrazyClimberObservation, CrazyClimberState, float, bool, CrazyClimberInfo):
        atari_action = jnp.take(self.ACTION_SET, action.astype(jnp.int32))
        previous_state = state

        state = self._step_counter(state)
        state = self._level_step(state)
        state = self._player_step(state, atari_action)
        state = self._tower_step(state)

        state = jax.lax.cond(state.level_state.condor_active,
            lambda: self._bird_step(self._egg_step(state)),
            lambda: state)

        state = self._climbed_floors_step(state)
        state = self._flowerpot_enemy_step(state)
        state = self._flowerpot_collision_step(state)
        state = self._score_step(state)
        state = self._bonus_step(state)
        state = jax.lax.cond(state.score >= self.consts.HELICOPTER_SPAWN_HEIGHT,
            lambda: self._helicopter_step(state),
            lambda: state,
        )

        _, next_rng = jax.random.split(state.key)
        state = state.replace(key=next_rng)

        done = self._get_done(state)
        env_reward = self._get_reward(previous_state, state)
        info = self._get_info(state)
        observation = self._get_observation(state)

        return observation, state, env_reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _level_step(self, state: CrazyClimberState) -> CrazyClimberState:
        """
        Runs the correct level step method for each level
        """
        return self._level_1_step(state)

    @partial(jax.jit, static_argnums=(0,))
    def _level_1_step(self, state: CrazyClimberState) -> CrazyClimberState:
        """
        Runs the logic for Level 1 enemy activations.
        Enemies:
            - Mad Doctor Round 1
            - Condor:
                Activation: if player between threshholds and not active
                Deactivation: if player hit by egg; player outside thresholds; player fell in general
            - Mad Doctor Round 2
        """
        level_state = state.level_state

        # mad doctor logic

        # condor logic
        condor_activate = ((state.score >= self.consts.BIRD_SPAWN_THRESHOLD)
                & (state.score < self.consts.BIRD_DESPAWN_THRESHOLD)
                & (jnp.logical_not(level_state.condor_active))
                & (level_state.next_enemy == Enemy.CONDOR))
        condor_deactivate = level_state.condor_active & ((state.score > self.consts.BIRD_DESPAWN_THRESHOLD) | state.bird_state.stop)

        condor_active = jnp.where(
            condor_activate,
            True,
            jnp.where(condor_deactivate, False, level_state.condor_active)
        )

        next_enemy = jnp.where(condor_deactivate, level_state.next_enemy + 1, level_state.next_enemy)

        # mad doctor second round logic

        new_level_state = level_state.replace(
            condor_active = condor_active,
            next_enemy = next_enemy
        )

        return state.replace(level_state=new_level_state)

    @partial(jax.jit, static_argnums=(0,))
    def _step_counter(self, state: CrazyClimberState) -> CrazyClimberState:
        return state.replace(step_counter=state.step_counter + 1)

    @partial(jax.jit, static_argnums=(0,))
    def _tower_step(self, state: CrazyClimberState) -> CrazyClimberState:
        @partial(jax.jit)
        def update_blinds(windows: jnp.ndarray) -> jnp.ndarray:
            blinds_left = windows[:, :3, 0]
            blind_dirs_left = windows[:, :3, 1]

            new_blinds_left = jnp.where(
                blind_dirs_left == 1, 
                jnp.minimum(blinds_left + 1, 6), 
                blinds_left
            )

            new_blinds_left = jnp.where(
                blind_dirs_left == -1, 
                jnp.maximum(new_blinds_left - 1, 0), 
                new_blinds_left
            )

            new_blind_dirs_left = jnp.where(
                new_blinds_left == 6,
                blind_dirs_left * -1,
                blind_dirs_left  
            )
            
            new_blind_dirs_left = jnp.where(
                new_blinds_left == 0,
                jnp.zeros_like(new_blind_dirs_left),
                new_blind_dirs_left
            )
            
            windows_left = jnp.stack([new_blinds_left, new_blind_dirs_left], axis=-1)
            return jnp.concatenate([windows_left, jnp.fliplr(windows_left)], axis=1)
        
        @partial(jax.jit)
        def shift_windows(windows: jnp.ndarray, spawn_propability: float, key) -> jnp.ndarray:
            windows = jnp.roll(windows, shift=1, axis=0)
            new_blind_dirs_left = jax.random.choice(key, jnp.array([0, 1]), (1, 3), p=jnp.array([1 - spawn_propability, spawn_propability]))
            new_blinds_left = jnp.zeros((1, 3))
            new_row_left = jnp.stack([new_blinds_left, new_blind_dirs_left], axis=2)
            new_row = jnp.concatenate([new_row_left, jnp.fliplr(new_row_left)], axis=1) 
            windows = windows.at[:1, :, :].set(new_row)
            return windows
        
        @partial(jax.jit)
        def update_tower(s: TowerState) -> TowerState:
            possible_tower_steps = jnp.array([0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
            
            tower_step = jax.lax.cond(
                state.player_move_state.main_state == PlayerStableStates.PULL_UP,
                lambda: possible_tower_steps[state.player_move_state.sub_step],
                lambda: 0,
            )

            windows = jax.lax.cond(
                state.step_counter % 59 == 0,
                lambda: update_blinds(state.tower_state.windows),
                lambda: state.tower_state.windows
            )

            windows = jax.lax.cond(
                (state.player_move_state.main_state == PlayerStableStates.NEUTRAL) & state.reached_apex,
                lambda: shift_windows(windows, state.tower_state.spawn_probability, state.key),
                lambda: windows,
            )

            lowest_level = jax.lax.cond(
                (state.player_move_state.main_state == PlayerStableStates.NEUTRAL) & state.reached_apex,
                lambda: s.lowest_level + 1,
                lambda: s.lowest_level
            )

            return s.replace(tower_step=tower_step, windows=windows, lowest_level=lowest_level)

        is_falling = (state.player_move_state.falling_count == 160) | (state.tower_state.is_falling & (state.player_move_state.falling_count > 0))
        update_conds = [
            (is_falling == False) & (state.tower_state.is_falling == False),
            (is_falling == False) & (state.tower_state.is_falling == True),
            (is_falling == True) & (state.tower_state.is_falling == False),
        ]
        
        branch_idx = jnp.select(
            update_conds, 
            [0, 1, 2], 
            default=3
        )

        tower_state = jax.lax.switch(
            branch_idx,
            [
                lambda: update_tower(state.tower_state),
                lambda: TowerState.new(state.key).replace(lowest_level=state.tower_state.lowest_level),
                lambda: state.tower_state.replace(is_falling=True),
                lambda: state.tower_state,
            ]
        )
        
        return state.replace(tower_state=tower_state) 
        
    @partial(jax.jit, static_argnums=(0,))
    def _player_step(self, state: CrazyClimberState, action: chex.Array) -> CrazyClimberState:
        @partial(jax.jit)
        def is_left_hand_safe(state: CrazyClimberState) -> bool:
            player_state = state.player_move_state
            left_moving = ((player_state.sub_step > 1) & jnp.any(jnp.array([
                ((player_state.hand_dir == 1) & 
                (player_state.main_state == PlayerStableStates.NEUTRAL)),
                ((player_state.hand_dir == -1) &
                (player_state.main_state == PlayerStableStates.HALF_PULL_UP)),
            ])))

            collision_windows = jax.lax.switch(
                player_state.pos_x,
                [
                    lambda: state.tower_state.windows[9:, 0, 0],
                    lambda: state.tower_state.windows[9:, 0, 0],
                    lambda: state.tower_state.windows[9:, 1, 0],
                    lambda: state.tower_state.windows[9:, 1, 0],
                    lambda: state.tower_state.windows[9:, 2, 0],
                    lambda: state.tower_state.windows[9:, 2, 0],
                    lambda: state.tower_state.windows[9:, 3, 0],
                    lambda: state.tower_state.windows[9:, 3, 0],
                    lambda: state.tower_state.windows[9:, 4, 0],
                    lambda: state.tower_state.windows[9:, 4, 0],
                    lambda: state.tower_state.windows[9:, 5, 0],
                ],
            )
            
            return jax.lax.cond(
                left_moving,
                lambda: False,
                lambda: jax.lax.cond(
                    ((player_state.main_state == PlayerStableStates.NEUTRAL) |
                    ((player_state.main_state == PlayerStableStates.HALF_PULL_UP) & (player_state.hand_dir == -1))),
                    lambda: collision_windows[1] != 6,
                    lambda: collision_windows[0] != 6
                )
            )
            
        @partial(jax.jit)
        def is_right_hand_safe(state: CrazyClimberState) -> bool:
            player_state = state.player_move_state
            right_moving = ((player_state.sub_step > 1) & jnp.any(jnp.array([
                ((player_state.hand_dir == -1) & 
                (player_state.main_state == PlayerStableStates.NEUTRAL)),
                ((player_state.hand_dir == 1) &
                (player_state.main_state == PlayerStableStates.HALF_PULL_UP)),
            ])))

            collision_windows = jax.lax.switch(
                player_state.pos_x,
                [
                    lambda: state.tower_state.windows[9:, 0, 0],
                    lambda: state.tower_state.windows[9:, 1, 0],
                    lambda: state.tower_state.windows[9:, 1, 0],
                    lambda: state.tower_state.windows[9:, 2, 0],
                    lambda: state.tower_state.windows[9:, 2, 0],
                    lambda: state.tower_state.windows[9:, 3, 0],
                    lambda: state.tower_state.windows[9:, 3, 0],
                    lambda: state.tower_state.windows[9:, 4, 0],
                    lambda: state.tower_state.windows[9:, 4, 0],
                    lambda: state.tower_state.windows[9:, 5, 0],
                    lambda: state.tower_state.windows[9:, 5, 0],
                ],
            )

            return jax.lax.cond(
                right_moving,
                lambda: False,
                lambda: jax.lax.cond(
                    ((player_state.main_state == PlayerStableStates.NEUTRAL) |
                    ((player_state.main_state == PlayerStableStates.HALF_PULL_UP) & (player_state.hand_dir == 1))),
                    lambda: collision_windows[1] != 6,
                    lambda: collision_windows[0] != 6
                )
            )
            
        @partial(jax.jit)
        def move_upwards(s: PlayerMoveState) -> PlayerMoveState:
            # is_up_move_possible = (jax.lax.abs(s.side_step) <= 3) & (s.main_state != PlayerStableStates.REACHING)
            on_top_of_tower = state.score >= CrazyClimberConstants.HELICOPTER_SPAWN_HEIGHT
            transitioning_states = (((s.main_state != PlayerStableStates.PULL_UP) & (s.sub_step == 4)) |
                                    ((s.main_state == PlayerStableStates.PULL_UP) & (s.sub_step == 9)))
            next_state_on_transition = jnp.array([PlayerStableStates.NEUTRAL, PlayerStableStates.HALF_PULL_UP, PlayerStableStates.PULL_UP])[(s.main_state + 1) % 3]
            next_state_on_transition = jnp.where(
                on_top_of_tower,
                PlayerStableStates.REACHING,
                next_state_on_transition,
            )
            next_hand_dir = jax.lax.select(
                transitioning_states & (next_state_on_transition == PlayerStableStates.NEUTRAL),
                s.hand_dir * -1,
                s.hand_dir
            )
            return jax.lax.cond(
                transitioning_states,
                lambda _: s.replace(main_state=next_state_on_transition, sub_step=0, hand_dir=next_hand_dir, side_step=0),
                lambda s: s.replace(sub_step=s.sub_step + 1, side_step=0),
                operand=s,
            )

        @partial(jax.jit)
        def move_downwards(s: PlayerMoveState) -> PlayerMoveState:
            is_down_move_possible = ((s.main_state == PlayerStableStates.REACHING)
                                     |(
                                            # (jax.lax.abs(s.side_step) <= 3)
                                             (s.sub_step > 0)
                                             & (s.main_state != PlayerStableStates.PULL_UP)
                                     ))

            # transition flags
            is_half_pull_up_cancel = (s.main_state == PlayerStableStates.HALF_PULL_UP) & (s.sub_step == 1)
            is_reaching_cancel = s.main_state == PlayerStableStates.REACHING
            is_neutral_step1 = (s.main_state == PlayerStableStates.NEUTRAL) & (s.sub_step == 1)

            # next state logic
            next_main_state = jnp.where(
                is_half_pull_up_cancel | is_reaching_cancel,
                PlayerStableStates.NEUTRAL,
                s.main_state
            )
            next_sub_step = jnp.where(
                is_half_pull_up_cancel,
                0,
                jnp.where(
                    is_reaching_cancel,
                    4,
                    s.sub_step - 1,
                ),
            )
            next_hand_dir = jnp.where(
                is_neutral_step1 | is_half_pull_up_cancel,
                s.hand_dir * -1,
                s.hand_dir
            )

            return jax.lax.cond(
                is_down_move_possible,
                lambda: s.replace(
                    main_state=next_main_state,
                    sub_step=next_sub_step,
                    hand_dir=next_hand_dir,
                    side_step=0,
                ),
                lambda: s,
            )
        
        @partial(jax.jit)
        def move_horizontal(s: PlayerMoveState, dir: int, delta_x: int) -> PlayerMoveState:
            is_right_move_possible = s.sub_step <= 1
            return jax.lax.cond(
                is_right_move_possible,
                lambda s: jax.lax.cond(
                    (jax.lax.abs(s.side_step) >= 12) & (jax.lax.sign(s.side_step) == jax.lax.sign(dir)),
                    lambda s: s.replace(side_step=0, pos_x=jnp.clip(s.pos_x + dir * delta_x, 0, 10)),
                    lambda s: s.replace(side_step=s.side_step + dir),
                    operand=s,
                ),
                lambda s: s,
                operand=s,
            )

        POSSIBLE_X_FULL = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        POSSIBLE_X_MIDDLE_CUT = jnp.array([0, 1, 2, 8, 9, 10])
        POSSIBLE_X_SIDE_CUTS = jnp.array([4, 5, 6])

        def can_move_left(state: CrazyClimberState) -> bool:
            left_arm_up = (state.player_move_state.hand_dir == 1) & (state.player_move_state.main_state != PlayerStableStates.NEUTRAL)
            hand_offset = jnp.where(left_arm_up, 1, 0)
            next_pos_x = state.player_move_state.pos_x - 1

            can_move_left = jax.lax.switch(
                state.tower_state.levels[state.tower_state.lowest_level + 2 + hand_offset],
                [
                    lambda: jnp.any(POSSIBLE_X_FULL == next_pos_x),
                    lambda: jnp.any(POSSIBLE_X_MIDDLE_CUT == next_pos_x),
                    lambda: jnp.any(POSSIBLE_X_SIDE_CUTS == next_pos_x),
                ]
            )

            window_x = jnp.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4])[state.player_move_state.pos_x]
            can_move_left &= state.tower_state.windows[10 - hand_offset, window_x, 0] != 6
            return can_move_left
        
        def can_move_right(state: CrazyClimberState) -> bool:
            right_arm_up = (state.player_move_state.hand_dir == -1) & (state.player_move_state.main_state != PlayerStableStates.NEUTRAL)
            hand_offset = jnp.where(right_arm_up, 1, 0)
            next_pos_x = state.player_move_state.pos_x + 1

            can_move_right = jax.lax.switch(
                state.tower_state.levels[state.tower_state.lowest_level + 2 + hand_offset],
                [
                    lambda: jnp.any(POSSIBLE_X_FULL == next_pos_x),
                    lambda: jnp.any(POSSIBLE_X_MIDDLE_CUT == next_pos_x),
                    lambda: jnp.any(POSSIBLE_X_SIDE_CUTS == next_pos_x),
                ]
            )

            window_x = jnp.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5])[state.player_move_state.pos_x]
            can_move_right &= state.tower_state.windows[10 - hand_offset, window_x, 0] != 6
            return can_move_right
        
        def can_move_up(state: CrazyClimberState) -> bool:
            can_move_up = jax.lax.switch(
                state.tower_state.levels[state.tower_state.lowest_level + 3],
                [
                    lambda: jnp.any(POSSIBLE_X_FULL == state.player_move_state.pos_x),
                    lambda: jnp.any(POSSIBLE_X_MIDDLE_CUT == state.player_move_state.pos_x),
                    lambda: jnp.any(POSSIBLE_X_SIDE_CUTS == state.player_move_state.pos_x),
                ]
            )
            
            window_cols = jnp.array([[0, 0], [0, 1], [1, 1], [1, 2], [2, 2], [2, 3], [3, 3], [3, 4], [4, 4], [4, 5], [5, 5]])[state.player_move_state.pos_x]
            blinds = state.tower_state.windows[9, window_cols, 0]
            can_move_up &= ((state.player_move_state.main_state != PlayerStableStates.HALF_PULL_UP) | 
                ((state.player_move_state.main_state == PlayerStableStates.HALF_PULL_UP) & jnp.all(blinds != 6)))
            
            return can_move_up


        up = action == Action.UP
        down = action == Action.DOWN
        left = action == Action.LEFT
        right = action == Action.RIGHT

        left_hand_safe = is_left_hand_safe(state)
        right_hand_safe = is_right_hand_safe(state)

        player_move_state = state.player_move_state
        is_falling = player_move_state.falling_count > 0
        is_flying_away = (state.helicopter_state.fly_away_step > 0) | (state.helicopter_state.fly_away_state != HeliFlyAwayStates.NORMAL)
        movement_locked = jnp.logical_or(is_falling, is_flying_away)

        falling_conds = jnp.array([
            (~left_hand_safe) & (~right_hand_safe),
            (player_move_state.main_state == PlayerStableStates.NEUTRAL) & (player_move_state.hand_dir == 1) & (~right_hand_safe) & up,
            (player_move_state.main_state == PlayerStableStates.NEUTRAL) & (player_move_state.hand_dir == -1) & (~left_hand_safe) & up,
            # (player_move_state.main_state == PlayerStableStates.HALF_PULL_UP) & (player_move_state.hand_dir == 1) & (~left_hand_safe) & up,
            # (player_move_state.main_state == PlayerStableStates.HALF_PULL_UP) & (player_move_state.hand_dir == -1) & (~right_hand_safe) & up,
            (player_move_state.main_state == PlayerStableStates.PULL_UP) & (player_move_state.sub_step == 0) & down & (~(left_hand_safe & right_hand_safe))
        ])

        can_move_left = can_move_left(state)
        can_move_right = can_move_right(state)
        can_move_up = can_move_up(state)
        delta_x = jnp.where(((~can_move_left) & left) | ((~can_move_right) & right), 0, 1)

        should_fall = jnp.any(falling_conds) | player_move_state.should_fall
        pause = state.level_state.pause_game
        action_state_cases = [
            should_fall & (~movement_locked),
            up & (player_move_state.main_state != PlayerStableStates.PULL_UP) & (~movement_locked) & can_move_up,
            down & (player_move_state.main_state == PlayerStableStates.PULL_UP) & (~movement_locked),
            down & (player_move_state.main_state != PlayerStableStates.PULL_UP) & (~movement_locked),
            left & right_hand_safe & (~movement_locked),
            right & left_hand_safe & (~movement_locked),
            pause,
        ]
        
        branch_idx = jnp.select(
            action_state_cases, 
            [0, 1, 2, 3, 4, 5, 6],
            default=6
        )
        
        next_player_move_state = jax.lax.switch(
            branch_idx,
            [
                lambda s: PlayerMoveState.new().replace(falling_count=160, pos_x=s.pos_x, should_fall=False),
                lambda s: move_upwards(s),
                lambda s: move_upwards(s),
                lambda s: move_downwards(s),
                lambda s: move_horizontal(s, -1, delta_x),
                lambda s: move_horizontal(s, 1, delta_x),
                lambda s: self.update_player_move_state(s)
            ],
            operand=player_move_state
        )

        return state.replace(
            player_move_state=next_player_move_state,
        )

    @partial(jax.jit, static_argnums=(0,))
    def update_player_move_state(self, s: PlayerMoveState) -> PlayerMoveState:
        jax.debug.print("{x}", x=s.falling_count)
        state = jax.lax.cond(
            s.falling_count > 0,
            lambda: s.replace(falling_count=jnp.maximum(s.falling_count - 1, 0)),
            lambda: s
        )

        state1 = jax.lax.cond(
            state.egg_animation_count > 0,
            lambda: state.replace(egg_animation_count=jnp.maximum(state.egg_animation_count - 1, 0)),
            lambda: state
        )

        return state1

    @partial(jax.jit, static_argnums=(0,))
    def _bird_step(self, state: CrazyClimberState) -> CrazyClimberState:
        """
        Calculates new x, y coordinates and determines if bird should fly away.
        """
        bird_state = state.bird_state

        # border constraints
        hit_left_wall  = ((bird_state.pos_x < self.consts.BIRD_BORDERS[0])
            & (bird_state.dir < 0))
        hit_right_wall = ((bird_state.pos_x > self.consts.WIDTH - self.consts.BIRD_BORDERS[1])
            & (bird_state.dir > 0))

        should_move = (state.step_counter % 2 == 0)
        fly_away = (bird_state.dir < 0) & (state.score > self.consts.BIRD_DESPAWN_THRESHOLD)

        should_flip_dir = (should_move
            & (hit_right_wall | hit_left_wall)
            & ~fly_away)

        bird_state.dir = jnp.where(should_flip_dir, bird_state.dir * -1, bird_state.dir)

        bird_state.pos_x = jnp.where(
            should_move,
            bird_state.pos_x + bird_state.dir,
            bird_state.pos_x
        )

        bird_state.pos_y = jnp.where(
            state.player_move_state.main_state == PlayerStableStates.PULL_UP,
            self.consts.BIRD_Y + self.consts.BIRD_POSSIBLE_STEPS[state.player_move_state.sub_step],
            self.consts.BIRD_Y,
        )

        return state.replace(bird_state = bird_state)

    @partial(jax.jit, static_argnums=(0,))
    def _egg_step(self, state: CrazyClimberState) -> CrazyClimberState:
        """
        Calculates new coordinates for the egg and if new egg should be dropped
        """
        def new_egg(state: CrazyClimberState) -> EggState:
            """
            Resets egg coordinates
            """
            egg_state = state.bird_state.egg_state
            egg_state.pos_x = state.bird_state.pos_x
            egg_state.pos_y = 69
            egg_state.dir = state.bird_state.dir
            return egg_state

        def check_for_collision(state: CrazyClimberState) -> bool:
            """
            Checks for a collision betwenn the player and the egg
            """
            player_x = self.consts.PLAYER_POSSIBLE_X[state.player_move_state.pos_x]

            # checks if x coordinates (widths of sprites) overlap
            same_x = jnp.logical_or((((state.bird_state.egg_state.pos_x + self.consts.EGG_SIZE[1]) >= player_x)
                & ((state.bird_state.egg_state.pos_x + self.consts.EGG_SIZE[1]) <= (player_x + self.consts.PLAYER_SIZE[1]))),
                ((state.bird_state.egg_state.pos_x >= player_x)
                & (state.bird_state.egg_state.pos_x <= (player_x + self.consts.PLAYER_SIZE[1]))))

            # checks if y coordinates (heights of sprites) overlap
            same_y = jnp.logical_or((((state.bird_state.egg_state.pos_y + self.consts.EGG_SIZE[0]) >= self.consts.PLAYER_Y)
                & ((state.bird_state.egg_state.pos_y + self.consts.EGG_SIZE[0]) <= (self.consts.PLAYER_Y + self.consts.PLAYER_SIZE[0]))),
                ((state.bird_state.egg_state.pos_y >= self.consts.PLAYER_Y)
                & (state.bird_state.egg_state.pos_y <= (self.consts.PLAYER_Y + self.consts.PLAYER_SIZE[0]))))

            return jnp.logical_and(same_x, same_y)

        @partial(jax.jit)
        def egg_hit(state: CrazyClimberState) -> CrazyClimberState:
            """
            Runs logic if player got hit by egg.
            """
            # check if player is in safe position
            player_safe = jnp.logical_and(
                jnp.logical_or(state.player_move_state.main_state == PlayerStableStates.NEUTRAL,
                    state.player_move_state.main_state == PlayerStableStates.PULL_UP),
                jnp.logical_or(state.player_move_state.side_step < 4,
                    state.player_move_state.sub_step < 2))

            # if player not safe -> fall and deactivate bird else -> pause game and do egg breaking animation
            new_state = state
            new_player_state = new_state.player_move_state.replace(
                should_fall = jnp.array(True),
                flicker = jnp.array(False))

            next_state = jax.lax.cond(
                player_safe,
                lambda: break_anim(state),
                lambda: new_state.replace(player_move_state=new_player_state))

            return next_state

        def break_anim(state: CrazyClimberState) -> CrazyClimberState:
            """
            logic for the egg breaking animation and freezing of the game
            """
            anim_count = jnp.array(13)
            pause = jnp.array(True)

            level_state = state.level_state.replace(pause_game=pause)
            player_state = state.player_move_state.replace(egg_animation_count=anim_count)
            return state.replace(level_state=level_state, player_move_state=player_state)

        bird_state = state.bird_state
        egg_state = bird_state.egg_state

        egg_currently_active = ((egg_state.pos_y < self.consts.EGG_BORDER_BOTTOM))
        drop_egg = ~egg_currently_active

        state = jax.lax.cond(
            check_for_collision(state),
            lambda: egg_hit(state),
            lambda: state)

        player_state = state.player_move_state

        egg_state = jax.lax.cond(
            drop_egg,
            lambda: new_egg(state),
            lambda: egg_state
        )

        player_state.flicker = jnp.logical_and(egg_state.pos_y > self.consts.EGG_FLICKER, (state.step_counter % 2) == 0) & state.level_state.condor_active
        egg_state.flicker = jnp.logical_and(~player_state.flicker, egg_state.pos_y > self.consts.EGG_FLICKER) & state.level_state.condor_active

        egg_state.pos_y = egg_state.pos_y + 1
        egg_state.pos_x = egg_state.pos_x + (((state.step_counter % egg_state.vel) == 0).astype(int) * egg_state.dir)

        bird_state.egg_state = egg_state
        bird_state.drop_egg = drop_egg

        bird_state.stop = jnp.where(
            state.player_move_state.should_fall | state.player_move_state.falling_count == 160,
            jnp.array(True),
            state.bird_state.stop)

        return state.replace(
            player_move_state = player_state,
            bird_state = bird_state
        )

    @partial(jax.jit, static_argnums=(0,))
    def _climbed_floors_step(self, state: CrazyClimberState) -> CrazyClimberState:
        climbed_triggered = (state.player_move_state.main_state == PlayerStableStates.NEUTRAL) & state.reached_apex
        next_climbed_floors = jnp.where(
            state.tower_state.is_falling,
            0,
            jnp.where(climbed_triggered, state.climbed_floors + 1, state.climbed_floors),
        )

        return state.replace(climbed_floors=next_climbed_floors)

    @partial(jax.jit, static_argnums=(0,))
    def _score_step(self, state: CrazyClimberState) -> CrazyClimberState:
        currently_at_apex = (state.player_move_state.sub_step == 9) & (state.player_move_state.main_state == PlayerStableStates.PULL_UP)

        score_triggered = (state.player_move_state.main_state == PlayerStableStates.NEUTRAL) & state.reached_apex
        new_score = jnp.where(score_triggered, state.score + state.level_state.score_increment, state.score)

        return state.replace(
            score=new_score,
            reached_apex=currently_at_apex
        )

    @partial(jax.jit, static_argnums=(0,))
    def _flowerpot_enemy_step(self, state: CrazyClimberState) -> CrazyClimberState:
        score_ranges = self.consts.FLOWERPOT_SCORE_RANGES
        score_in_flowerpot_range = jnp.any(
            (state.score >= score_ranges[:, 0])
            & (state.score < score_ranges[:, 1])
        )
        flowerpot_area_active = (
            score_in_flowerpot_range
            & (state.climbed_floors >= self.consts.FLOWERPOT_MIN_CLIMBED_FLOORS)
        )

        def reset_flowerpot_enemy(s: CrazyClimberState) -> CrazyClimberState:
            return s.replace(
                flowerpot_enemy_state=FlowerpotEnemyState.new(
                    False,
                    -1,
                    jnp.array(0, dtype=jnp.int32),
                    jnp.array(0, dtype=jnp.int32),
                    jnp.array(0, dtype=jnp.int32),
                    jnp.array(0, dtype=jnp.int32),
                )
            )

        def protect_flowerpot_row(s: CrazyClimberState) -> CrazyClimberState:
            row = s.flowerpot_enemy_state.window_row
            windows = s.tower_state.windows
            windows = windows.at[row, :, 0].set(0)
            windows = windows.at[row, :, 1].set(0)
            return s.replace(tower_state=s.tower_state.replace(windows=windows))

        def update_active_flowerpot_enemy(s: CrazyClimberState) -> CrazyClimberState:
            climbed_triggered = (s.player_move_state.main_state == PlayerStableStates.NEUTRAL) & s.reached_apex
            phase_zero = s.flowerpot_enemy_state.phase == 0
            phase_zero_done = phase_zero & (s.flowerpot_enemy_state.phase_steps == self.consts.FLOWERPOT_PHASE_0_STEPS - 1)
            phase_one = s.flowerpot_enemy_state.phase == 1
            phase_one_steps = (
                self.consts.FLOWERPOT_CYCLE_STEPS_BY_ROW[s.flowerpot_enemy_state.cycle_row]
                - self.consts.FLOWERPOT_PHASE_0_STEPS
            )
            phase_one_done = phase_one & (s.flowerpot_enemy_state.phase_steps == phase_one_steps - 1)
            next_cycle_row = jnp.where(
                phase_zero_done,
                s.flowerpot_enemy_state.window_row,
                s.flowerpot_enemy_state.cycle_row,
            )
            next_phase = jnp.where(
                phase_zero_done,
                1,
                s.flowerpot_enemy_state.phase,
            )
            next_phase_steps = jnp.where(
                phase_zero_done,
                0,
                s.flowerpot_enemy_state.phase_steps + 1,
            )
            next_window_row = jnp.where(
                climbed_triggered,
                s.flowerpot_enemy_state.window_row + 1,
                s.flowerpot_enemy_state.window_row,
            )
            s = s.replace(
                flowerpot_enemy_state = s.flowerpot_enemy_state.replace(
                    phase=next_phase,
                    phase_steps=next_phase_steps,
                    window_row=next_window_row,
                    cycle_row=next_cycle_row,
                )
            )
            s = protect_flowerpot_row(s)
            return jax.lax.cond(
                phase_one_done,
                reset_flowerpot_enemy,
                lambda s: s,
                s,
            )

        def try_spawn_flowerpot_enemy(s: CrazyClimberState) -> CrazyClimberState:
            candidates = self.consts.FLOWERPOT_CANDIDATE_WINDOWS
            candidate_rows = candidates[:, 0]
            candidate_cols = candidates[:, 1]
            row_has_closing = jnp.any(
                s.tower_state.windows[candidate_rows, :, 0] > 0,
                axis=1,
            )
            candidate_level_types = s.tower_state.levels[
                s.tower_state.lowest_level + 12 - candidate_rows
            ]
            candidate_window_exists = (
                (candidate_level_types == TowerLevelType.FULL)
                | (
                    (candidate_level_types == TowerLevelType.MIDDLE_CUT)
                    & ((candidate_cols <= 1) | (candidate_cols >= 4))
                )
                | (
                    (candidate_level_types == TowerLevelType.SIDE_CUTS)
                    & ((candidate_cols == 2) | (candidate_cols == 3))
                )
            )
            valid_mask = ~row_has_closing & candidate_window_exists
            has_valid_candidate = jnp.any(valid_mask)

            def spawn_from_valid_candidate(s: CrazyClimberState) -> CrazyClimberState:
                random_values = jax.random.uniform(s.key, shape=(candidates.shape[0],))
                masked_random_values = jnp.where(valid_mask, random_values, -1.0)
                selected_idx = jnp.argmax(masked_random_values)
                selected_window = candidates[selected_idx]
                drop_type_key = jax.random.fold_in(s.key, 1)
                selected_drop_type = jax.random.randint(
                    drop_type_key,
                    shape=(),
                    minval=0,
                    maxval=self.consts.FLOWERPOT_DROP_TYPE_COUNT,
                    dtype=jnp.int32,
                )
                s = s.replace(
                    flowerpot_enemy_state=FlowerpotEnemyState.new(
                        True,
                        0,
                        selected_window[0],
                        selected_window[1],
                        selected_window[0],
                        selected_drop_type,
                    )
                )
                return protect_flowerpot_row(s)

            return jax.lax.cond(
                has_valid_candidate,
                spawn_from_valid_candidate,
                lambda s: s,
                s,
            )

        def run_flowerpot_area_logic(s: CrazyClimberState) -> CrazyClimberState:
            return jax.lax.cond(
                s.flowerpot_enemy_state.active,
                update_active_flowerpot_enemy,
                try_spawn_flowerpot_enemy,
                s,
            )

        return jax.lax.cond(
            flowerpot_area_active,
            run_flowerpot_area_logic,
            reset_flowerpot_enemy,
            state,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _flowerpot_collision_step(self, state: CrazyClimberState) -> CrazyClimberState:
        phase_steps = jnp.maximum(state.flowerpot_enemy_state.phase_steps, 0)

        first_cycle_offsets = jnp.array(
            [0, 0, 0, 0, 0, 0, 0, 1, 2, 4, 4, 5, 6, 7, 8, 10, 13, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23],
            dtype=jnp.int32,
        )
        loop_offsets = jnp.array(
            [0, 0, 1, 2, 4, 4, 5, 6, 7, 8, 10, 13, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23],
            dtype=jnp.int32,
        )
        drop_y_offset = jnp.where(
            phase_steps < 27,
            first_cycle_offsets[jnp.minimum(phase_steps, 26)],
            23 + ((phase_steps - 27) // 22) * 23 + loop_offsets[(phase_steps - 27) % 22],
        )

        window_local_x = jnp.array([4, 16, 28, 44, 56, 68], dtype=jnp.int32)[state.flowerpot_enemy_state.window_col]
        window_local_y = jnp.array([5, 18, 31, 44, 57, 70, 83, 96, 109, 122, 135], dtype=jnp.int32)[state.flowerpot_enemy_state.window_row]
        tower_scroll_offset = jax.lax.cond(
            ~state.tower_state.is_falling,
            lambda: self.consts.TOWER_POSSIBLE_SPRITE_CLIP[state.tower_state.tower_step],
            lambda: self.consts.TOWER_POSSIBLE_SPRITE_CLIP[state.player_move_state.falling_count % 4],
        )
        top_clip = 14 - tower_scroll_offset
        window_center_x = 40 + window_local_x + 4
        window_top_y = 44 + window_local_y - top_clip

        drop_center_x = window_center_x + state.flowerpot_enemy_state.drop_x_offset
        drop_x = drop_center_x - (self.consts.FLOWERPOT_DROP_HITBOX_WIDTH // 2)
        drop_y = window_top_y + 6 + 4 + drop_y_offset

        player_x = self.consts.PLAYER_POSSIBLE_X[state.player_move_state.pos_x]
        player_y = self.consts.PLAYER_Y
        player_width = 16
        player_height = 23

        x_overlap = (
            (drop_x < player_x + player_width)
            & (drop_x + self.consts.FLOWERPOT_DROP_HITBOX_WIDTH > player_x)
        )
        y_overlap = (
            (drop_y < player_y + player_height)
            & (drop_y + self.consts.FLOWERPOT_DROP_HITBOX_HEIGHT > player_y)
        )
        drop_collision = x_overlap & y_overlap

        player_state = state.player_move_state
        can_deflect = (
            (
                (player_state.main_state == PlayerStableStates.PULL_UP)
                & (player_state.sub_step <= 1)
            )
            | (
                (player_state.main_state == PlayerStableStates.NEUTRAL)
                & (player_state.sub_step <= 1)
            )
        ) & (jnp.abs(player_state.side_step) <= 3)

        collision_active = (
            state.flowerpot_enemy_state.active
            & (state.flowerpot_enemy_state.phase == 1)
            & (state.flowerpot_enemy_state.drop_x_offset == 0)
            & drop_collision
        )

        def deflect_drop(s: CrazyClimberState) -> CrazyClimberState:
            return s.replace(
                flowerpot_enemy_state=s.flowerpot_enemy_state.replace(
                    drop_x_offset=jnp.array(self.consts.FLOWERPOT_DROP_DEFLECT_X_OFFSET, dtype=jnp.int32),
                )
            )

        def make_player_fall(s: CrazyClimberState) -> CrazyClimberState:
            return s.replace(
                player_move_state=s.player_move_state.replace(
                    should_fall=True,
                ),
            )

        return jax.lax.cond(
            collision_active,
            lambda s: jax.lax.cond(can_deflect, deflect_drop, make_player_fall, s),
            lambda s: s,
            state,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _bonus_step(self, state: CrazyClimberState) -> CrazyClimberState: 
        bonus_condition = ((state.step_counter > (self.consts.BONUS_DECREASE_THRESHOLD - 1))
            & ((state.step_counter - self.consts.BONUS_DECREASE_THRESHOLD) % self.consts.BONUS_DECREASE_INTERVAL == 0))

        bonus = jnp.where(bonus_condition, state.bonus - 100, state.bonus)
        return state.replace(bonus=bonus)

    @partial(jax.jit, static_argnums=(0,))
    def _helicopter_step(self, state: CrazyClimberState) -> CrazyClimberState:
        @partial(jax.jit)
        def normal_step(state: CrazyClimberState) -> CrazyClimberState:
            heli_state = state.helicopter_state

            # movement specific stuff
            should_move_x = (heli_state.x_movement_unlocked & (state.step_counter % 2 == 0))
            should_move_y = True

            # border constraints
            hit_left_wall = ((heli_state.pos_x <= CrazyClimberConstants.HELICOPTER_BORDERS_X[0])
                             & (heli_state.x_dir < 0))
            hit_right_wall = ((heli_state.pos_x >= CrazyClimberConstants.HELICOPTER_BORDERS_X[1])
                              & (heli_state.x_dir > 0))
            hit_lower_wall = ((heli_state.pos_y >= CrazyClimberConstants.HELICOPTER_BORDERS_Y[0])
                              & (heli_state.y_dir < 0))
            hit_upper_wall = ((heli_state.pos_y <= CrazyClimberConstants.HELICOPTER_BORDERS_Y[1])
                              & (heli_state.y_dir > 0))

            # For some reason movement in x direction only begins after going up and down and reaching a certain threshold
            heli_state.x_movement_unlocked = jnp.where(
                heli_state.pos_y > CrazyClimberConstants.HELICOPTER_MOVEMENT_BEGIN,
                True,
                heli_state.x_movement_unlocked
            )

            # direction flip
            should_flip_hor_dir = (hit_right_wall | hit_left_wall)
            should_flip_vert_dir = (hit_upper_wall | hit_lower_wall)

            heli_state.x_dir = jnp.where(should_flip_hor_dir, heli_state.x_dir * -1, heli_state.x_dir)
            heli_state.y_dir = jnp.where(should_flip_vert_dir, heli_state.y_dir * -1,
                                               heli_state.y_dir)

            # update position
            heli_state.pos_x = jnp.where(
                should_move_x,
                heli_state.pos_x + heli_state.x_dir,
                heli_state.pos_x
            )
            heli_state.pos_y = jnp.where(
                should_move_y,
                heli_state.pos_y - heli_state.y_dir,
                heli_state.pos_y,
            )

            return state.replace(
                helicopter_state=heli_state,
            )

        @partial(jax.jit)
        def caught_helicopter_step(state: CrazyClimberState) -> CrazyClimberState:
            """
            step method when heli flies away after catching it
            first decrements the bonus and adds on to score
            then yadda yadda
            """
            bonus_not_zero = state.bonus > 0
            in_dec_state = state.helicopter_state.fly_away_state == HeliFlyAwayStates.BONUS_DEC
            should_dec = bonus_not_zero & in_dec_state

            next_bonus = jnp.where(should_dec, state.bonus - 100, state.bonus)
            next_score = state.score - next_bonus + state.bonus
            fly_away_step = state.helicopter_state.fly_away_step
            current_fly_away_state = state.helicopter_state.fly_away_state

            # transition conditions
            normal_thold = (fly_away_step > 20) & (state.helicopter_state.fly_away_state == HeliFlyAwayStates.NORMAL)
            one_row_thold = (fly_away_step > 40) & (state.helicopter_state.fly_away_state == HeliFlyAwayStates.ONE_ROW)
            bonus_dec_thold = (state.bonus <= 0) & (state.helicopter_state.fly_away_state == HeliFlyAwayStates.BONUS_DEC)
            nothing_thold = (fly_away_step > 160) & (state.helicopter_state.fly_away_state == HeliFlyAwayStates.NOTHING)

            next_fly_away_state = jnp.where(
                nothing_thold,
                HeliFlyAwayStates.DONE,
                jnp.where(
                    bonus_dec_thold,
                    HeliFlyAwayStates.NOTHING,
                    jnp.where(
                        one_row_thold,
                        HeliFlyAwayStates.BONUS_DEC,
                        jnp.where(
                            normal_thold,
                            HeliFlyAwayStates.ONE_ROW,
                            current_fly_away_state
                        )
                    )
                )
            )

            next_fly_away_step = jnp.where(next_fly_away_state != current_fly_away_state,
                0,
                fly_away_step + 1,
            )

            # if player caught, fly up for some time (need collision method for this)
            # else if state.bonus > 0 decrement bonus move in y direction
            # else done, next level
            next_helicopter_state = state.helicopter_state.replace(
                fly_away_step=next_fly_away_step,
                fly_away_state=next_fly_away_state,
            )

            return state.replace(
                bonus=next_bonus,
                score=next_score,
                helicopter_state=next_helicopter_state,
            )

        @partial(jax.jit)
        def no_time_step(state: CrazyClimberState) -> CrazyClimberState:
            """
            step method when heli flies away after time runs out
            first decrements the bonus while y movement then triggers next level
            """
            heli_state = state.helicopter_state
            # movement constraints
            hit_lower_wall = ((heli_state.pos_y >= CrazyClimberConstants.HELICOPTER_BORDERS_Y[0])
                              & (heli_state.y_dir < 0))
            hit_upper_wall = ((heli_state.pos_y <= CrazyClimberConstants.HELICOPTER_BORDERS_Y[1])
                              & (heli_state.y_dir > 0))

            should_flip_vert_dir = (hit_upper_wall | hit_lower_wall)
            next_y_dir = jnp.where(should_flip_vert_dir, heli_state.y_dir * -1,
                                         heli_state.y_dir)

            # update position
            next_pos_y = heli_state.pos_y - heli_state.y_dir

            # check for end
            done = state.bonus == 0

            # bonus decrement
            next_bonus = jnp.where(~done, state.bonus - 100, state.bonus)
            next_fly_away_step = heli_state.fly_away_step + 1

            next_helicopter_state = heli_state.replace(
                fly_away_step=next_fly_away_step,
                pos_y=next_pos_y,
                y_dir=next_y_dir,
            )

            return state.replace(
                bonus=next_bonus,
                helicopter_state=next_helicopter_state,
            )

        def check_heli_collision(state: CrazyClimberState) -> bool:
            heli_state = state.helicopter_state
            correct_height = heli_state.pos_y >= 128
            reaching = state.player_move_state.main_state == PlayerStableStates.REACHING
            right_hand_reach = state.player_move_state.hand_dir < 0
            player_x = jnp.where(right_hand_reach,
                self.consts.PLAYER_POSSIBLE_X[state.player_move_state.pos_x] + self.consts.PLAYER_SIZE[1],
                self.consts.PLAYER_POSSIBLE_X[state.player_move_state.pos_x],
            )

            heli_dir_is_left = heli_state.x_dir < 0
            heli_left_bound = jnp.where(heli_dir_is_left,
                heli_state.pos_x,
                heli_state.pos_x + CrazyClimberConstants.HELICOPTER_SIZE[0] - CrazyClimberConstants.HELICOPTER_SKIDS_SIZE,
            )
            heli_right_bound = jnp.where(heli_dir_is_left,
                heli_state.pos_x + CrazyClimberConstants.HELICOPTER_SKIDS_SIZE,
                heli_state.pos_x + CrazyClimberConstants.HELICOPTER_SIZE[0],
            )

            correct_x = (player_x > heli_left_bound) & (player_x < heli_right_bound)

            collision = correct_height & reaching & correct_x
            return collision

        heli_state = state.helicopter_state
        #TODO: needs to be adapted to work multiple times at the next level. needs to be cleared at some point (maybe when progressing to the next level)
        next_step = heli_state.step + 1

        heli_collision = check_heli_collision(state)
        time_expired = (heli_state.step >= CrazyClimberConstants.HELICOPTER_MAX_STEPS)
        branch_idx = jnp.select(
            [heli_collision, time_expired],
            [2,1],
            default=0,
        )

        next_state = jax.lax.switch(
            branch_idx,
            [
                lambda s: normal_step(s),
                lambda s: no_time_step(s),
                lambda s: caught_helicopter_step(s),
            ],
            operand=state
        )

        next_state = next_state.replace(
            helicopter_state=next_state.helicopter_state.replace(
                step=next_step,
            )
        )

        #jax.debug.print(
        #    "helicopter_state: x:{x}, y:{y}, hor_dir:{z}, vert_dir:{a}",
        #    x=next_state.helicopter_state.pos_x,
        #    y=next_state.helicopter_state.pos_y,
        #    z=next_state.helicopter_state.x_dir,
        #    a=next_state.helicopter_state.y_dir,
        #)

        return next_state

    def render(self, state: CrazyClimberState) -> jnp.ndarray:
        return self.renderer.render(state)

    # TODO: Returntype needs to be altered to match actual implementation
    def action_space(self) -> spaces.Discrete:
        pass

    # TODO: Returntype needs to be altered to match actual implementation
    def observation_space(self) -> spaces.Dict:
        object_space = spaces.get_object_space(n=None, screen_size=(self.consts.HEIGHT, self.consts.WIDTH))

        return spaces.Dict({
            "player": spaces.Box(low=0, high=21, shape=(), dtype=jnp.int32),
        })

    # TODO: Returntype needs to be altered to match actual implementation
    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(210, 160, 3),
            dtype=jnp.uint8
        ) 

    def _get_observation(self, state: CrazyClimberState) -> CrazyClimberObservation:
        pass
    
    def obs_to_flat_array(self, obs: CrazyClimberObservation) -> jnp.ndarray:
        pass

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: CrazyClimberState) -> CrazyClimberInfo:
        pass

    def _get_reward(self, previous_state: CrazyClimberState, state: CrazyClimberState) -> float:
        return 0

    def _get_done(self, state: CrazyClimberState) -> bool:
        pass

    class CrazyClimberRenderer(JAXGameRenderer):
        def __init__(self, consts: CrazyClimberConstants = None, config: render_utils.RendererConfig = None):
            self.consts = consts or CrazyClimberConstants()
            super().__init__(consts)
            self.config = render_utils.RendererConfig(
                game_dimensions=(210, 160),
                channels=3,
                downscale=None
            )
            self.jr = render_utils.JaxRenderingUtils(self.config)
            
            final_asset_config = list(self.consts.ASSET_CONFIG)

            sprite_path = os.path.join(os.path.dirname(__file__), "sprites", "crazy_climber")

            (
                self.PALETTE,
                self.SHAPE_MASKS,
                self.BACKGROUND,
                self.COLOR_TO_ID,
                self.FLIP_OFFSETS
            ) = self.jr.load_and_setup_assets(final_asset_config, sprite_path)

            self.PLAYER_UPWARDS_SPRITES = jnp.array([
                self.SHAPE_MASKS["player_upwards_left_group"],
                self.SHAPE_MASKS["player_upwards_right_group"],
            ])

            self.PLAYER_SIDEWAYS_SPRITES = jnp.array([
                jnp.array([
                    self.SHAPE_MASKS["player_neutral_left_group"],
                    self.SHAPE_MASKS["player_neutral_right_group"],
                ]),
                jnp.array([
                    self.SHAPE_MASKS["player_pull_up_left_group"],
                    self.SHAPE_MASKS["player_pull_up_right_group"],
                ]),
            ])

            self.PLAYER_SIDEWAYS_SPRITES_ARM_SPECIFIC = jnp.array([
                jnp.array([
                    self.SHAPE_MASKS["player_left_pull_up_left_group"],
                    self.SHAPE_MASKS["player_left_pull_up_right_group"],
                ]),
                jnp.array([
                    self.SHAPE_MASKS["player_right_pull_up_left_group"],
                    self.SHAPE_MASKS["player_right_pull_up_right_group"],
                ]),
            ])

            self.BIRD_SPRITES = jnp.array([
                self.SHAPE_MASKS["bird_left"],
                self.SHAPE_MASKS["bird_right"],
            ])
            self.EGG_SPRITES = self.SHAPE_MASKS["egg_falling"]

            self.TOWER_SPRITE = self._generate_tower_sprite()
            self.TOWER_CUTOUTS = self._generate_tower_cutouts()

            self.PLAYER_REACHING_SPRITES = jnp.array(
                self.SHAPE_MASKS["player_reaching_group"],
            )

            self.HELICOPTER_SPRITES = jnp.array([
                self.SHAPE_MASKS["helicopter_left"],
                self.SHAPE_MASKS["helicopter_right"],
            ])

            self.PLAYER_UPWARDS_SPRITE_SEQUENCE = jnp.array([0, 0, 1, 2, 3, 4, 4, 5, 6, 7, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11])
            self.PLAYER_SIDEWAYS_SPRITE_SEQUENCE = jnp.array([0, 0, 0, 0, 1, 1, 1, 1, 3, 3, 3, 3])

            self.FLOWERPOT_THROWER_SPRITES = self.SHAPE_MASKS["flowerpot_thrower_group"]
            _, self.FLOWERPOT_THROWER_BOTTOM_Y_OFFSETS = self._get_visible_sprite_anchors(
                self.FLOWERPOT_THROWER_SPRITES
            )
            self.FLOWERPOT_DROP_SPRITES = self.SHAPE_MASKS["flowerpot_drop_group"]
            self.FLOWERPOT_DROP_CENTER_X_OFFSETS, self.FLOWERPOT_DROP_BOTTOM_Y_OFFSETS = self._get_visible_sprite_anchors(
                self.FLOWERPOT_DROP_SPRITES
            )

            self.TOWER_SPRITE = self._generate_tower_sprite()
            self.TOWER_CUTOUTS = self._generate_tower_cutouts()


        def _get_visible_sprite_anchors(self, sprites: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            visible = sprites != self.jr.TRANSPARENT_ID
            any_visible_col = jnp.any(visible, axis=1)
            any_visible_row = jnp.any(visible, axis=2)

            left = jnp.argmax(any_visible_col, axis=1).astype(jnp.int32)
            right = (sprites.shape[2] - 1 - jnp.argmax(any_visible_col[:, ::-1], axis=1)).astype(jnp.int32)
            bottom = (sprites.shape[1] - 1 - jnp.argmax(any_visible_row[:, ::-1], axis=1)).astype(jnp.int32)

            center_x = (left + right) // 2
            return center_x, bottom

        def _generate_tower_cutouts(self) -> jnp.ndarray:
            floor_raster = self._create_raster((13, 80))
            level_full = floor_raster
            level_middle_cut = level_full.at[:, 28:52].set(0)
            level_side_cuts = level_full.at[:, 4:24].set(0).at[:, 56:76].set(0)
            return jnp.array([level_full, level_middle_cut, level_side_cuts])
            
        def _generate_tower_sprite(self) -> jnp.ndarray:
            tower_raster = self._create_raster((170, 80))

            wall_offset_x = jnp.array([0, 12, 24, 36, 40, 52, 64, 76])
            wall_offset_y = jnp.repeat(0, 8)
            wall_sprite = self.SHAPE_MASKS["wall"]
            wall_sprite_masks = jnp.repeat(wall_sprite[jnp.newaxis, :, :], 8, axis=0)
            tower_raster = self.jr.render_at_batch(
                tower_raster,
                wall_offset_x,
                wall_offset_y,
                wall_sprite_masks,
            )
            
            ceiling_offset_x = jnp.repeat(0, 13)
            ceiling_offset_y = jnp.array([0, 13, 26, 39, 52, 65, 78, 91, 104, 117, 130, 143, 156])
            ceiling_sprite = self.SHAPE_MASKS["ceiling"]
            ceiling_sprite_masks = jnp.repeat(ceiling_sprite[jnp.newaxis, :, :], 13, axis=0)
            tower_raster = self.jr.render_at_batch(
                tower_raster,
                ceiling_offset_x,
                ceiling_offset_y,
                ceiling_sprite_masks,
            )

            floor_sprite = self.SHAPE_MASKS["floor"]
            tower_raster = self.jr.render_at(tower_raster, 0, 169, floor_sprite)

            return tower_raster

        @partial(jax.jit, static_argnums=(0,1))
        def _create_raster(self, shape: tuple[int, int]) -> jnp.ndarray:
            return jnp.tile(255, shape)
        
        @partial(jax.jit, static_argnums=(0,))
        def _normalize_raster(self, raster: jnp.ndarray) -> jnp.ndarray:
            return jnp.where(raster != 255, raster, 0)
        
        @partial(jax.jit, static_argnums=(0,))
        def _clip_raster(self, base: jnp.ndarray, overlay: jnp.ndarray, offset_x: int, offset_y: int) -> jnp.ndarray:
            base_slice = jax.lax.dynamic_slice(
                base,
                (offset_y, offset_x),
                overlay.shape
            )
            
            merged_slice = jnp.where(overlay != 255, overlay, base_slice)
            base = self.jr.render_at_clipped(
                base,
                offset_x, offset_y,
                merged_slice
            )
            
            return base
        
        @partial(jax.jit, static_argnums=(0,))
        def _render_player(self, state: CrazyClimberState) -> jnp.ndarray:
            player_raster = self._create_raster(self.consts.PLAYER_SIZE)

            move_state = state.player_move_state

            sprite_index_up = self.consts.PLAYER_UPWARDS_SPRITE_SEQUENCE[5 * move_state.main_state + move_state.sub_step]
            hand_index = (move_state.hand_dir < 0).astype(int) # maps -1 -> 1 and 1 -> 0
            sprite_index_side = self.consts.PLAYER_SIDEWAYS_SPRITE_SEQUENCE[jnp.abs(move_state.side_step)]
            side_index = (move_state.side_step > 0).astype(int) # maps -1 -> 1 and 1 -> 0

            @partial(jax.jit)
            def map_player_to_sprite(sprite_index_up: int, sprite_index_side: int, hand_index: int, side_index: int) -> jnp.ndarray:
                # flags
                is_sideways = jnp.logical_and(move_state.sub_step <= 1, jnp.abs(move_state.side_step) > 3)
                is_half_pull_up = move_state.main_state == PlayerStableStates.HALF_PULL_UP
                is_reaching = move_state.main_state == PlayerStableStates.REACHING

                # sprite definitions
                sprite_reaching = self.PLAYER_REACHING_SPRITES[hand_index]
                sprite_side_arm = self.PLAYER_SIDEWAYS_SPRITES_ARM_SPECIFIC[hand_index][side_index][sprite_index_side]
                sprite_side_def = self.PLAYER_SIDEWAYS_SPRITES[move_state.main_state][side_index][sprite_index_side]
                sprite_upwards = self.PLAYER_UPWARDS_SPRITES[hand_index][sprite_index_up]

                # sprite selection
                sprite_sideways = jnp.where(is_half_pull_up, sprite_side_arm, sprite_side_def)
                sprite_motion = jnp.where(is_sideways, sprite_sideways, sprite_upwards)
                return jnp.where(is_reaching, sprite_reaching, sprite_motion)
            
            player_sprite = map_player_to_sprite(sprite_index_up, sprite_index_side, hand_index, side_index)
            player_raster = self.jr.render_at(player_raster, 0, 0, player_sprite)
            
            return player_raster
        

        @partial(jax.jit, static_argnums=(0,))
        def _render_tower(self, state: CrazyClimberState) -> jnp.ndarray:
            def clip_tower_cut(raster: jnp.ndarray, cutouts: jnp.ndarray, y: int, level_type: TowerLevelType) -> jnp.ndarray:
                raster = jax.lax.dynamic_slice(
                    raster,
                    (y, 0),
                    (13, 80)
                )

                cutout = cutouts[level_type]
                return jnp.where(cutout != 255, cutout, raster)
            
            batched_clip_tower_cut = jax.vmap(clip_tower_cut, in_axes=(None, None, 0, 0))

            tower_raster = jnp.copy(self.TOWER_SPRITE)
            window_offset_x = jnp.tile(jnp.array([4, 16, 28, 44, 56, 68]), 11)
            window_offset_y = jnp.repeat(jnp.array([5, 18, 31, 44, 57, 70, 83, 96, 109, 122, 135]), 6)
            sprites = self.SHAPE_MASKS["window_blind_group"][state.tower_state.windows[:, :, 0].astype(jnp.int32)]
            sprites = jnp.reshape(sprites, (-1, *sprites.shape[2:]))
            tower_sprite = self.jr.render_at_batch(tower_raster, window_offset_x, window_offset_y, sprites)

            level_indices = jax.lax.dynamic_slice_in_dim(
                state.tower_state.levels,
                state.tower_state.lowest_level,
                13,
                axis=0
            )
            level_offset_y = jnp.array([1, 14, 27, 40, 53, 66, 79, 92, 105, 118, 131, 144, 157])
            tower_sprite = jnp.concat(batched_clip_tower_cut(tower_sprite, self.TOWER_CUTOUTS, level_offset_y, level_indices[::-1]), axis=0)

            falling_tower_raster = jnp.copy(self.TOWER_SPRITE)
            falling_tower_sprite = jnp.concat(
                batched_clip_tower_cut(
                    falling_tower_raster, 
                    self.TOWER_CUTOUTS, 
                    level_offset_y, 
                    jnp.repeat(level_indices[0], 13)
                ), 
                axis=0
            )

            tower_sprite = jax.lax.cond(
                ~state.tower_state.is_falling,
                lambda: tower_sprite,
                lambda: falling_tower_sprite
            )

            top_clip = 14 - jax.lax.cond(
                ~state.tower_state.is_falling,
                lambda: self.consts.TOWER_POSSIBLE_SPRITE_CLIP[state.tower_state.tower_step],
                lambda: self.consts.TOWER_POSSIBLE_SPRITE_CLIP[state.player_move_state.falling_count % 4]
            )

            row_indices = state.tower_state.lowest_level + jnp.arange(13)
            max_level = CrazyClimberConstants.HELICOPTER_SPAWN_HEIGHT / 100
            valid_row_mask = row_indices < (max_level + 2) # needs to be two higher because of the unused rows at the bottom

            pixel_mask = jnp.repeat(valid_row_mask[::-1], 13)[:, None]

            fly_away_state = state.helicopter_state.fly_away_state
            nothing_cond = (fly_away_state == HeliFlyAwayStates.NOTHING) | (fly_away_state == HeliFlyAwayStates.BONUS_DEC)

            pixel_mask = jnp.where(
                fly_away_state == HeliFlyAwayStates.ONE_ROW,
                CrazyClimberConstants.PIXEL_MASK_ONE_ROW,
                jnp.where(
                    nothing_cond,
                    CrazyClimberConstants.PIXEL_MASK_NOTHING,
                    pixel_mask,
                ),
            )

            tower_sprite = jnp.where(pixel_mask, tower_sprite, 0)

            tower_raster = jax.lax.dynamic_slice_in_dim(
                tower_sprite,
                top_clip,
                156,
                axis=0
            )
            
            return tower_raster

        @partial(jax.jit, static_argnums=(0,))
        def _render_bird(self, state: CrazyClimberState) -> jnp.ndarray:
            """
            Selects correct sprites for the bird based on the sequence
            """
            bird_raster = self._create_raster(self.consts.BIRD_SIZE)
            bird_state = state.bird_state

            dir_index = (bird_state.dir > 0).astype(int)
            bird_index = self.consts.BIRD_SEQUENCE[((((bird_state.pos_x + self.consts.BIRD_SEQUENCE.size) * bird_state.dir) % 40) / 4).astype(int)]

            bird_sprite = self.BIRD_SPRITES[dir_index][bird_index]

            bird_raster = self.jr.render_at(
                bird_raster,
                0, 0,
                bird_sprite,
            )

            return bird_raster

        @partial(jax.jit, static_argnums=(0,))
        def _render_egg(self, state: CrazyClimberState) -> jnp.ndarray:
            """
            Selects correct egg sprites based on the y-position
            """
            egg_raster = self._create_raster(self.consts.EGG_SIZE)

            #egg_sprite = self.EGG_SPRITES[state.bird_state.egg_y % 11]
            egg_sprite = self.EGG_SPRITES[9]

            egg_raster = self.jr.render_at(
                egg_raster,
                0, 0,
                egg_sprite
            )

            return egg_raster

        @partial(jax.jit, static_argnums=(0,))
        def _get_window_screen_position(
            self,
            state: CrazyClimberState,
            window_row: chex.Array,
            window_col: chex.Array,
        ) -> tuple[chex.Array, chex.Array]:
            window_local_x = jnp.array([4, 16, 28, 44, 56, 68], dtype=jnp.int32)[window_col]
            window_local_y = jnp.array([5, 18, 31, 44, 57, 70, 83, 96, 109, 122, 135], dtype=jnp.int32)[window_row]

            tower_scroll_offset = jax.lax.cond(
                ~state.tower_state.is_falling,
                lambda: self.consts.TOWER_POSSIBLE_SPRITE_CLIP[state.tower_state.tower_step],
                lambda: self.consts.TOWER_POSSIBLE_SPRITE_CLIP[state.player_move_state.falling_count % 4],
            )
            top_clip = 14 - tower_scroll_offset

            screen_x = 40 + window_local_x
            screen_y = 44 + window_local_y - top_clip

            return screen_x, screen_y

        @partial(jax.jit, static_argnums=(0,))
        def _get_window_bottom_center(
            self,
            state: CrazyClimberState,
            window_row: chex.Array,
            window_col: chex.Array,
        ) -> tuple[chex.Array, chex.Array]:
            window_left_x, window_top_y = self._get_window_screen_position(state, window_row, window_col)
            return window_left_x + 4, window_top_y + 7

        @partial(jax.jit, static_argnums=(0,))
        def _render_flowerpot_thrower(self, raster: jnp.ndarray, state: CrazyClimberState) -> jnp.ndarray:
            phase = state.flowerpot_enemy_state.phase
            phase_steps = state.flowerpot_enemy_state.phase_steps

            phase_zero_sprite_idx = jnp.minimum(phase_steps // 8, 3)
            phase_one_segment = jnp.minimum(phase_steps // 8, 4)
            phase_one_sprite_idx = jnp.array([4, 3, 2, 1, 0], dtype=jnp.int32)[phase_one_segment]
            sprite_idx = jnp.where(phase == 0, phase_zero_sprite_idx, phase_one_sprite_idx)

            x_offsets = jnp.array([3, 2, 1, 1, 1], dtype=jnp.int32)
            y_offsets = jnp.array([6, 5, 3, 2, 1], dtype=jnp.int32)

            window_left_x, window_top_y = self._get_window_screen_position(
                state,
                state.flowerpot_enemy_state.window_row,
                state.flowerpot_enemy_state.window_col,
            )
            draw_x = window_left_x + x_offsets[sprite_idx]
            draw_y = window_top_y + y_offsets[sprite_idx]
            thrower_sprite = self.FLOWERPOT_THROWER_SPRITES[sprite_idx]

            thrower_visible = (
                state.flowerpot_enemy_state.active
                & ((phase == 0) | ((phase == 1) & (phase_steps < 40)))
            )

            return jax.lax.cond(
                thrower_visible,
                lambda r: self.jr.render_at(r, draw_x, draw_y, thrower_sprite),
                lambda r: r,
                raster,
            )

        @partial(jax.jit, static_argnums=(0,))
        def _render_flowerpot_drop(self, raster: jnp.ndarray, state: CrazyClimberState) -> jnp.ndarray:
            phase_steps = jnp.maximum(state.flowerpot_enemy_state.phase_steps, 0)

            drop_type = state.flowerpot_enemy_state.drop_type
            loop_length = self.consts.FLOWERPOT_DROP_LOOP_LENGTHS[drop_type]
            sprite_idx = jnp.where(
                phase_steps < 5,
                phase_steps,
                5 + ((phase_steps - 5) % loop_length),
            )
            sprite_idx = self.consts.FLOWERPOT_DROP_SPRITE_OFFSETS[drop_type] + sprite_idx

            first_cycle_offsets = jnp.array(
                [0, 0, 0, 0, 0, 0, 0, 1, 2, 4, 4, 5, 6, 7, 8, 10, 13, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23],
                dtype=jnp.int32,
            )
            loop_offsets = jnp.array(
                [0, 0, 1, 2, 4, 4, 5, 6, 7, 8, 10, 13, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23],
                dtype=jnp.int32,
            )
            y_offset = jnp.where(
                phase_steps < 27,
                first_cycle_offsets[jnp.minimum(phase_steps, 26)],
                23 + ((phase_steps - 27) // 22) * 23 + loop_offsets[(phase_steps - 27) % 22],
            )

            window_center_x, _ = self._get_window_bottom_center(
                state,
                state.flowerpot_enemy_state.window_row,
                state.flowerpot_enemy_state.window_col,
            )
            _, window_top_y = self._get_window_screen_position(
                state,
                state.flowerpot_enemy_state.window_row,
                state.flowerpot_enemy_state.window_col,
            )
            drop_sprite = self.FLOWERPOT_DROP_SPRITES[sprite_idx]
            thrower_y_offsets = jnp.array([6, 5, 3, 2, 1], dtype=jnp.int32)
            thrower_bottom_y = (
                window_top_y
                + thrower_y_offsets[4]
                + self.FLOWERPOT_THROWER_BOTTOM_Y_OFFSETS[4]
            )
            draw_x = window_center_x + state.flowerpot_enemy_state.drop_x_offset - self.FLOWERPOT_DROP_CENTER_X_OFFSETS[sprite_idx]
            draw_y = thrower_bottom_y + 4 + y_offset

            drop_visible = (
                state.flowerpot_enemy_state.active
                & (state.flowerpot_enemy_state.phase == 1)
            )

            return jax.lax.cond(
                drop_visible,
                lambda r: self.jr.render_at_clipped(r, draw_x, draw_y, drop_sprite),
                lambda r: r,
                raster,
            )

        @partial(jax.jit, static_argnums=(0,))
        def _render_helicopter(self, state: CrazyClimberState) -> jnp.ndarray:
            helicopter_raster = self._create_raster(self.consts.HELICOPTER_SIZE)

            helicopter_state = state.helicopter_state

            dir_index = (helicopter_state.x_dir > 0).astype(int)
            helicopter_index = self.consts.HELICOPTER_SEQUENCE[
                (((helicopter_state.step + self.consts.HELICOPTER_SEQUENCE.size) * helicopter_state.x_dir) % 4).astype(int)]

            helicopter_sprite = self.HELICOPTER_SPRITES[dir_index][helicopter_index]

            helicopter_raster = self.jr.render_at(
                helicopter_raster,
                0, 0,
                helicopter_sprite,
            )

            return helicopter_raster

        @partial(jax.jit, static_argnums=(0,))
        def render(self, state: CrazyClimberState) -> jnp.ndarray:
            raster = self._create_raster((210, 160))

            player_raster = self._render_player(state)
            tower_raster = self._render_tower(state)
            bird_raster = self._render_bird(state)
            egg_raster = self._render_egg(state)
            helicopter_raster = self._render_helicopter(state)
            raster = self._clip_raster(raster, tower_raster, 40, 44) # self.jr.render_at_clipped(raster, 0, 0, tower_raster)
            raster = self._clip_raster(raster, player_raster, self.consts.PLAYER_POSSIBLE_X[state.player_move_state.pos_x], self.consts.PLAYER_Y) # self.jr.render_at_clipped(raster, state.player_move_state.pos_x, self.consts.PLAYER_Y, player_raster)
            raster = jax.lax.cond(state.score >= self.consts.HELICOPTER_SPAWN_HEIGHT,
                lambda: self._clip_raster(raster, helicopter_raster, state.helicopter_state.pos_x, state.helicopter_state.pos_y),
                lambda: raster,
            )

            raster = jax.lax.cond(~state.player_move_state.flicker,
                lambda: self._clip_raster(raster, player_raster, self.consts.PLAYER_POSSIBLE_X[state.player_move_state.pos_x], self.consts.PLAYER_Y), # self.jr.render_at_clipped(raster, state.player_move_state.pos_x, self.consts.PLAYER_Y, player_raster
                lambda: raster)
            raster = jax.lax.cond(state.level_state.condor_active,
                lambda: self._clip_raster(raster, bird_raster, state.bird_state.pos_x, state.bird_state.pos_y),
                lambda: raster)
            raster = jax.lax.cond(state.level_state.condor_active & ~state.bird_state.egg_state.flicker,
                lambda: self._clip_raster(raster, egg_raster, state.bird_state.egg_state.pos_x, state.bird_state.egg_state.pos_y),
                lambda: raster)
            raster = self._render_flowerpot_thrower(raster, state)
            raster = self._render_flowerpot_drop(raster, state)

            raster = self._normalize_raster(raster)

            score_digits = self.jr.int_to_digits(state.score, max_digits=6)
            bonus_digits = self.jr.int_to_digits(state.bonus, max_digits=5)
            digit_masks = self.SHAPE_MASKS["digits"]

            life_mask = self.SHAPE_MASKS["life"]
            raster = self.jr.render_at(raster, 58, 12, life_mask)
            raster = self.jr.render_at(raster, 74, 12, life_mask)
            raster = self.jr.render_at(raster, 90, 12, life_mask)
            
            raster = self.jr.render_label_selective(raster, 57, 20, bonus_digits, digit_masks, start_index=0, num_to_render=5, spacing=8, max_digits_to_render=6)
            raster = self.jr.render_label_selective(raster, 49, 30, score_digits, digit_masks, start_index=0, num_to_render=6, spacing=8, max_digits_to_render=6)

            return self.jr.render_from_palette(raster, self.PALETTE)
