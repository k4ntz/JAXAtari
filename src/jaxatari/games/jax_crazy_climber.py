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

# Player movement states

class PlayerStableStates(IntEnum):
    NEUTRAL = 0
    HALF_PULL_UP = 1
    PULL_UP = 2
    
@chex.dataclass
class PlayerMoveState:
    main_state: PlayerStableStates 
    sub_step: int 
    side_step: int 
    hand_dir: int
    pos_x: int
    falling_count: int
    should_fall: bool

    @classmethod
    def new(cls):
        return cls(
            main_state=PlayerStableStates.NEUTRAL,
            sub_step=0,
            side_step=0,
            hand_dir=1,
            pos_x=0,
            falling_count=0,
            should_fall=False
        )

class TowerLevelType(IntEnum):
    FULL = 0
    MIDDLE_CUT = 1
    SIDE_CUTS = 2
    
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

@chex.dataclass
class FlowerpotEnemyState:
    active: chex.Array
    phase: chex.Array
    phase_steps: chex.Array
    window_row: chex.Array
    window_col: chex.Array
    cycle_row: chex.Array
    drop_x_offset: chex.Array

class CrazyClimberState(struct.PyTreeNode):
    key: chex.PRNGKey
    score: chex.Array
    bonus: chex.Array
    step_counter: chex.Array
    player_move_state: PlayerMoveState
    tower_state: TowerState
    reached_apex: chex.Array
    climbed_floors: chex.Array
    flowerpot_enemy: FlowerpotEnemyState

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
            ]},
        {'name': 'wall', 'type': 'procedural', 'data': wall_sprite},
        {'name': 'ceiling', 'type': 'procedural', 'data': ceiling_sprite},
        {'name': 'floor', 'type': 'procedural', 'data': floor_sprite},
        {'name': 'window_blind_group', 'type': 'procedural', 'data': window_sprites}
    )

class CrazyClimberConstants(struct.PyTreeNode):
    BACKGROUND_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(0, 0, 0)),
    ASSET_CONFIG: tuple = _get_default_asset_config()

    WIDTH: int = struct.field(pytree_node=False, default=160)
    HEIGHT: int = struct.field(pytree_node=False, default=210)

    PLAYER_Y: int = struct.field(pytree_node=False, default=160)
    PLAYER_POSSIBLE_X: jnp.ndarray = struct.field(pytree_node=False, default=jnp.array([40, 46, 52, 58, 64, 72, 80, 86, 92, 98, 104]))
    TOWER_POSSIBLE_SPRITE_CLIP: jnp.ndarray = struct.field(pytree_node=False, default=jnp.array([0, 4, 7, 10])) 

    SCORE_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(236, 236, 236))
    FLOWERPOT_SCORE_RANGES: jnp.ndarray = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array(
            [
                [200, 5000],
                [10000, 12500],
            ],
            dtype=jnp.int32,
        ),
    )
    FLOWERPOT_MIN_CLIMBED_FLOORS: int = struct.field(pytree_node=False, default=2)
    FLOWERPOT_PHASE_0_STEPS: int = struct.field(pytree_node=False, default=32)
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
    TOWER1: jnp.ndarray = struct.field(
        pytree_node=False, 
        default=jnp.concat([
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
    )

class JaxCrazyClimber(JaxEnvironment[CrazyClimberState, CrazyClimberObservation, CrazyClimberInfo, CrazyClimberConstants]):
    ACTION_SET: jnp.ndarray = jnp.array(
        [Action.NOOP, Action.UP, Action.RIGHT, Action.LEFT, Action.DOWN, Action.UPRIGHT, Action.UPLEFT, Action.DOWNRIGHT, Action.DOWNLEFT],
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
            score=jnp.array(0).astype(jnp.int32),
            bonus=jnp.array(10000).astype(jnp.int32),
            step_counter=jnp.array(0).astype(jnp.int32),
            player_move_state=PlayerMoveState.new(),
            tower_state=TowerState.new(state_key),
            reached_apex=jnp.array(False),
            climbed_floors=jnp.array(0, dtype=jnp.int32),
            flowerpot_enemy=FlowerpotEnemyState(
                active=jnp.array(False),
                phase=jnp.array(0, dtype=jnp.int32),
                phase_steps=jnp.array(0, dtype=jnp.int32),
                window_row=jnp.array(0, dtype=jnp.int32),
                window_col=jnp.array(0, dtype=jnp.int32),
                cycle_row=jnp.array(0, dtype=jnp.int32),
                drop_x_offset=jnp.array(0, dtype=jnp.int32),
            ),
        )
        initial_obs = self._get_observation(state)

        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: CrazyClimberState, action: chex.Array) -> (CrazyClimberObservation, CrazyClimberState, float, bool, CrazyClimberInfo):
        atari_action = jnp.take(self.ACTION_SET, action.astype(jnp.int32))
        previous_state = state
        
        state = self._step_counter(state)
        state = self._player_step(state, atari_action)
        state = self._tower_step(state)
        state = self._climbed_floors_step(state)
        state = self._flowerpot_enemy_step(state)
        state = self._flowerpot_collision_step(state)
        state = self._score_step(state)
        state = self._bonus_step(state)

        _, next_rng = jax.random.split(state.key)
        state = state.replace(key=next_rng)

        done = self._get_done(state)
        env_reward = self._get_reward(previous_state, state)
        info = self._get_info(state)
        observation = self._get_observation(state)

        return observation, state, env_reward, done, info
    
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
                lambda: TowerState.new(state.key),
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
            is_up_move_possible = (jax.lax.abs(s.side_step) <= 3)
            transitioning_states = (((s.main_state != PlayerStableStates.PULL_UP) & (s.sub_step == 4)) |
                                    ((s.main_state == PlayerStableStates.PULL_UP) & (s.sub_step == 9)))
            next_state_on_transition = jnp.array([PlayerStableStates.NEUTRAL, PlayerStableStates.HALF_PULL_UP, PlayerStableStates.PULL_UP])[(s.main_state + 1) % 3] 
            next_hand_dir = jax.lax.select(
                transitioning_states & (next_state_on_transition == PlayerStableStates.NEUTRAL),
                s.hand_dir * -1,
                s.hand_dir
            )
            return jax.lax.cond(
                is_up_move_possible,
                lambda s: jax.lax.cond(
                    transitioning_states,
                    lambda _: s.replace(main_state=next_state_on_transition, sub_step=0, hand_dir=next_hand_dir),
                    lambda s: s.replace(sub_step=s.sub_step + 1),
                    operand=s,
                ),
                lambda s: s,
                operand=s
            )

        @partial(jax.jit)
        def move_downwards(s: PlayerMoveState) -> PlayerMoveState:
            is_down_move_possible = (jax.lax.abs(s.side_step) <= 3) & (s.sub_step > 0) & (s.main_state != PlayerStableStates.PULL_UP)
            next_hand_dir = jax.lax.select(
                (s.main_state == PlayerStableStates.NEUTRAL) & (s.sub_step == 1),
                s.hand_dir * -1,
                s.hand_dir
            )
            return jax.lax.cond(
                is_down_move_possible,
                lambda s: jax.lax.cond(
                    (s.main_state == PlayerStableStates.HALF_PULL_UP) & (s.sub_step == 1),
                    lambda _: s.replace(main_state=PlayerStableStates.NEUTRAL, sub_step=0, hand_dir=s.hand_dir * -1),
                    lambda s: s.replace(sub_step=s.sub_step - 1, hand_dir=next_hand_dir),
                    operand=s
                ),
                lambda s: s,
                operand=s
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
        
        @partial(jax.jit)
        def update_player_move_state(s: PlayerMoveState) -> PlayerMoveState:
            return jax.lax.cond(
                s.falling_count > 0,
                lambda: s.replace(falling_count=jnp.maximum(s.falling_count - 1, 0)),
                lambda: s
            )
        
        def can_move_left(state: CrazyClimberState) -> bool:
            left_arm_up = (state.player_move_state.hand_dir == 1) & (state.player_move_state.main_state != PlayerStableStates.NEUTRAL)
            hand_offset = jnp.where(left_arm_up, 1, 0)
            next_pos_x = state.player_move_state.pos_x - 1
            
            possible_x_full = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            possible_x_middle_cut = jnp.array([0, 1, 2, 8, 9, 10])
            possible_x_side_cuts = jnp.array([5, 6, 7])

            can_move_left = jax.lax.switch(
                state.tower_state.levels[state.tower_state.lowest_level + 2 + hand_offset],
                [
                    lambda: jnp.any(possible_x_full == next_pos_x),
                    lambda: jnp.any(possible_x_middle_cut == next_pos_x),
                    lambda: jnp.any(possible_x_side_cuts == next_pos_x),
                ]
            )

            window_x = jnp.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4])[state.player_move_state.pos_x]
            can_move_left &= state.tower_state.windows[10 - hand_offset, window_x, 0] != 6
            return can_move_left
        
        def can_move_right(state: CrazyClimberState) -> bool:
            right_arm_up = (state.player_move_state.hand_dir == -1) & (state.player_move_state.main_state != PlayerStableStates.NEUTRAL)
            hand_offset = jnp.where(right_arm_up, 1, 0)
            next_pos_x = state.player_move_state.pos_x + 1
            
            possible_x_full = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            possible_x_middle_cut = jnp.array([0, 1, 2, 8, 9, 10])
            possible_x_side_cuts = jnp.array([5, 6, 7])

            can_move_right = jax.lax.switch(
                state.tower_state.levels[state.tower_state.lowest_level + 2 + hand_offset],
                [
                    lambda: jnp.any(possible_x_full == next_pos_x),
                    lambda: jnp.any(possible_x_middle_cut == next_pos_x),
                    lambda: jnp.any(possible_x_side_cuts == next_pos_x),
                ]
            )

            window_x = jnp.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5])[state.player_move_state.pos_x]
            can_move_right &= state.tower_state.windows[10 - hand_offset, window_x, 0] != 6
            return can_move_right
        
        def can_move_up(state: CrazyClimberState) -> bool:
            possible_up_full = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            possible_up_middle_cut = jnp.array([0, 1, 2, 8, 9, 10])
            possible_up_side_cuts = jnp.array([5, 6, 7])

            can_move_up = jax.lax.switch(
                state.tower_state.levels[state.tower_state.lowest_level + 3],
                [
                    lambda: jnp.any(possible_up_full == state.player_move_state.pos_x),
                    lambda: jnp.any(possible_up_middle_cut == state.player_move_state.pos_x),
                    lambda: jnp.any(possible_up_side_cuts == state.player_move_state.pos_x),
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
        action_state_cases = [
            should_fall & (~is_falling),
            up & (player_move_state.main_state != PlayerStableStates.PULL_UP) & (~is_falling) & can_move_up,
            down & (player_move_state.main_state == PlayerStableStates.PULL_UP) & (~is_falling),
            down & (player_move_state.main_state != PlayerStableStates.PULL_UP) & (~is_falling),
            left & right_hand_safe & (~is_falling),
            right & left_hand_safe & (~is_falling)
        ]
        
        branch_idx = jnp.select(
            action_state_cases, 
            [0, 1, 2, 3, 4, 5], 
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
                lambda s: update_player_move_state(s)
            ],
            operand=player_move_state
        )

        # jax.debug.print(
        #    "main state: {x}, sub step: {y}, side step {z}, hand dir: {w}",
        #    x=next_player_move_state.main_state,
        #    y=next_player_move_state.sub_step,
        #    z=next_player_move_state.side_step,
        #    w=next_player_move_state.hand_dir,
        #)

        return state.replace(
            player_move_state=next_player_move_state,
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

        new_score = jax.lax.select(score_triggered, state.score + 100, state.score)

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
                flowerpot_enemy=FlowerpotEnemyState(
                    active=jnp.array(False),
                    phase=jnp.array(0, dtype=jnp.int32),
                    phase_steps=jnp.array(-1, dtype=jnp.int32),
                    window_row=jnp.array(0, dtype=jnp.int32),
                    window_col=jnp.array(0, dtype=jnp.int32),
                    cycle_row=jnp.array(0, dtype=jnp.int32),
                    drop_x_offset=jnp.array(0, dtype=jnp.int32),
                )
            )

        def protect_flowerpot_row(s: CrazyClimberState) -> CrazyClimberState:
            row = s.flowerpot_enemy.window_row
            windows = s.tower_state.windows
            windows = windows.at[row, :, 0].set(0)
            windows = windows.at[row, :, 1].set(0)
            return s.replace(tower_state=s.tower_state.replace(windows=windows))

        def update_active_flowerpot_enemy(s: CrazyClimberState) -> CrazyClimberState:
            climbed_triggered = (s.player_move_state.main_state == PlayerStableStates.NEUTRAL) & s.reached_apex
            phase_zero = s.flowerpot_enemy.phase == 0
            phase_zero_done = phase_zero & (s.flowerpot_enemy.phase_steps == self.consts.FLOWERPOT_PHASE_0_STEPS - 1)
            phase_one = s.flowerpot_enemy.phase == 1
            phase_one_steps = (
                self.consts.FLOWERPOT_CYCLE_STEPS_BY_ROW[s.flowerpot_enemy.cycle_row]
                - self.consts.FLOWERPOT_PHASE_0_STEPS
            )
            phase_one_done = phase_one & (s.flowerpot_enemy.phase_steps == phase_one_steps - 1)
            next_cycle_row = jnp.where(
                phase_zero_done,
                s.flowerpot_enemy.window_row,
                s.flowerpot_enemy.cycle_row,
            )
            next_phase = jnp.where(
                phase_zero_done,
                1,
                s.flowerpot_enemy.phase,
            )
            next_phase_steps = jnp.where(
                phase_zero_done,
                0,
                s.flowerpot_enemy.phase_steps + 1,
            )
            next_window_row = jnp.where(
                climbed_triggered,
                s.flowerpot_enemy.window_row + 1,
                s.flowerpot_enemy.window_row,
            )
            s = s.replace(
                flowerpot_enemy=s.flowerpot_enemy.replace(
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
                s = s.replace(
                    flowerpot_enemy=FlowerpotEnemyState(
                        active=jnp.array(True),
                        phase=jnp.array(0, dtype=jnp.int32),
                        phase_steps=jnp.array(0, dtype=jnp.int32),
                        window_row=selected_window[0],
                        window_col=selected_window[1],
                        cycle_row=selected_window[0],
                        drop_x_offset=jnp.array(0, dtype=jnp.int32),
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
                s.flowerpot_enemy.active,
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
        phase_steps = jnp.maximum(state.flowerpot_enemy.phase_steps, 0)

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

        window_local_x = jnp.array([4, 16, 28, 44, 56, 68], dtype=jnp.int32)[state.flowerpot_enemy.window_col]
        window_local_y = jnp.array([5, 18, 31, 44, 57, 70, 83, 96, 109, 122, 135], dtype=jnp.int32)[state.flowerpot_enemy.window_row]
        tower_scroll_offset = jax.lax.cond(
            ~state.tower_state.is_falling,
            lambda: self.consts.TOWER_POSSIBLE_SPRITE_CLIP[state.tower_state.tower_step],
            lambda: self.consts.TOWER_POSSIBLE_SPRITE_CLIP[state.player_move_state.falling_count % 4],
        )
        top_clip = 14 - tower_scroll_offset
        window_center_x = 40 + window_local_x + 4
        window_top_y = 44 + window_local_y - top_clip

        drop_center_x = window_center_x + state.flowerpot_enemy.drop_x_offset
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
            state.flowerpot_enemy.active
            & (state.flowerpot_enemy.phase == 1)
            & (state.flowerpot_enemy.drop_x_offset == 0)
            & drop_collision
        )

        jax.lax.cond(
            drop_collision,
            lambda _: jax.debug.print(
                "flowerpot collision: active {a}, collision {c}, can_deflect {d}, main {m}, sub {s}, side {side}, drop_x_offset {o}",
                a=collision_active,
                c=drop_collision,
                d=can_deflect,
                m=player_state.main_state,
                s=player_state.sub_step,
                side=player_state.side_step,
                o=state.flowerpot_enemy.drop_x_offset,
            ),
            lambda _: None,
            operand=None,
        )

        def deflect_drop(s: CrazyClimberState) -> CrazyClimberState:
            return s.replace(
                flowerpot_enemy=s.flowerpot_enemy.replace(
                    drop_x_offset=jnp.array(self.consts.FLOWERPOT_DROP_DEFLECT_X_OFFSET, dtype=jnp.int32),
                )
            )

        def make_player_fall(s: CrazyClimberState) -> CrazyClimberState:
            return s.replace(
                player_move_state=PlayerMoveState.new().replace(
                    falling_count=160,
                    pos_x=s.player_move_state.pos_x,
                ),
                tower_state=s.tower_state.replace(is_falling=True),
            )

        return jax.lax.cond(
            collision_active,
            lambda s: jax.lax.cond(can_deflect, deflect_drop, make_player_fall, s),
            lambda s: s,
            state,
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def _bonus_step(self, state: CrazyClimberState) -> CrazyClimberState: 
        bonus_condition = ((state.step_counter - 1229) % 600 == 0) & (state.step_counter > 1228)

        bonus = jnp.where(bonus_condition, state.bonus - 100, state.bonus)

        return state.replace(bonus=bonus)

    
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
            player_raster = self._create_raster((23, 16))

            move_state = state.player_move_state

            sprite_index_up = self.PLAYER_UPWARDS_SPRITE_SEQUENCE[5 * move_state.main_state + move_state.sub_step]
            hand_index = (move_state.hand_dir + 2) % 3 # maps -1 -> 1, and 1 -> 0
            sprite_index_side = self.PLAYER_SIDEWAYS_SPRITE_SEQUENCE[jnp.abs(move_state.side_step)] 
            side_index = jnp.where(move_state.side_step > 0, 1, 0)

            @partial(jax.jit)
            def map_player_to_sprite(sprite_index_up: int, sprite_index_side: int, hand_index: int, side_index: int) -> jnp.ndarray:
                return jax.lax.cond(
                    jnp.logical_and(move_state.sub_step <= 1, jnp.abs(move_state.side_step) > 3),
                    lambda _: jax.lax.cond(
                        move_state.main_state == PlayerStableStates.HALF_PULL_UP,
                        lambda _: self.PLAYER_SIDEWAYS_SPRITES_ARM_SPECIFIC[hand_index][side_index][sprite_index_side],
                        lambda _: self.PLAYER_SIDEWAYS_SPRITES[move_state.main_state][side_index][sprite_index_side],
                        operand=None),
                    lambda _: self.PLAYER_UPWARDS_SPRITES[hand_index][sprite_index_up],
                    operand=None
                )
            
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
            
            tower_raster = jax.lax.dynamic_slice_in_dim(
                tower_sprite,
                top_clip,
                156,
                axis=0
            )
            
            return tower_raster

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
            phase = state.flowerpot_enemy.phase
            phase_steps = state.flowerpot_enemy.phase_steps

            phase_zero_sprite_idx = jnp.minimum(phase_steps // 8, 3)
            phase_one_segment = jnp.minimum(phase_steps // 8, 4)
            phase_one_sprite_idx = jnp.array([4, 3, 2, 1, 0], dtype=jnp.int32)[phase_one_segment]
            sprite_idx = jnp.where(phase == 0, phase_zero_sprite_idx, phase_one_sprite_idx)

            x_offsets = jnp.array([3, 2, 1, 1, 1], dtype=jnp.int32)
            y_offsets = jnp.array([6, 5, 3, 2, 1], dtype=jnp.int32)

            window_left_x, window_top_y = self._get_window_screen_position(
                state,
                state.flowerpot_enemy.window_row,
                state.flowerpot_enemy.window_col,
            )
            draw_x = window_left_x + x_offsets[sprite_idx]
            draw_y = window_top_y + y_offsets[sprite_idx]
            thrower_sprite = self.FLOWERPOT_THROWER_SPRITES[sprite_idx]

            thrower_visible = (
                state.flowerpot_enemy.active
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
            phase_steps = jnp.maximum(state.flowerpot_enemy.phase_steps, 0)

            sprite_idx = jnp.where(
                phase_steps < 5,
                phase_steps,
                5 + ((phase_steps - 5) % 22),
            )

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
                state.flowerpot_enemy.window_row,
                state.flowerpot_enemy.window_col,
            )
            _, window_top_y = self._get_window_screen_position(
                state,
                state.flowerpot_enemy.window_row,
                state.flowerpot_enemy.window_col,
            )
            drop_sprite = self.FLOWERPOT_DROP_SPRITES[sprite_idx]
            thrower_y_offsets = jnp.array([6, 5, 3, 2, 1], dtype=jnp.int32)
            thrower_bottom_y = (
                window_top_y
                + thrower_y_offsets[4]
                + self.FLOWERPOT_THROWER_BOTTOM_Y_OFFSETS[4]
            )
            draw_x = window_center_x + state.flowerpot_enemy.drop_x_offset - self.FLOWERPOT_DROP_CENTER_X_OFFSETS[sprite_idx]
            draw_y = thrower_bottom_y + 4 + y_offset

            drop_visible = (
                state.flowerpot_enemy.active
                & (state.flowerpot_enemy.phase == 1)
            )

            return jax.lax.cond(
                drop_visible,
                lambda r: self.jr.render_at_clipped(r, draw_x, draw_y, drop_sprite),
                lambda r: r,
                raster,
            )

        @partial(jax.jit, static_argnums=(0,))
        def render(self, state: CrazyClimberState) -> jnp.ndarray:
            raster = self._create_raster((210, 160))

            player_raster = self._render_player(state)
            tower_raster = self._render_tower(state)
            raster = self._clip_raster(raster, tower_raster, 40, 44) # self.jr.render_at_clipped(raster, 0, 0, tower_raster)
            raster = self._clip_raster(raster, player_raster, self.consts.PLAYER_POSSIBLE_X[state.player_move_state.pos_x], self.consts.PLAYER_Y) # self.jr.render_at_clipped(raster, state.player_move_state.pos_x, self.consts.PLAYER_Y, player_raster)
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
