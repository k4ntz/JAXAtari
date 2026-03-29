"""
authors: Paula Troszt, Ernst Christian Böhringer, Aiman Sammy Rahlf
"""

import os
from functools import partial
from typing import NamedTuple, Tuple, Dict, Any, Optional
import jax
import jax.lax
import jax.numpy as jnp
import chex
from flax import struct

import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
import jaxatari.rendering.jax_rendering_utils as render_utils
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action, ObjectObservation
from jaxatari.games.timepilot_levels import (
    LevelConstants,
    TimePilot_Level_1,
    TimePilot_Level_2,
    TimePilot_Level_3,
    TimePilot_Level_4,
    TimePilot_Level_5
)
from jaxatari.modification import AutoDerivedConstants

def _get_default_asset_config() -> tuple:
    """
    Returns the default declarative asset manifest for TimePilot.
    Kept immutable (tuple of dicts) to fit NamedTuple defaults.
    """
    # --- Define file lists for large groups ---
    # 59 files: 5 levels * (8 pos + 2 death) + (8 transition + 1 transition death)
    all_player_sprites_files = [
        # L1
        *(f'L1/L1_Player_Pos{i}.npy' for i in range(8)),
        'L1/L1_Player_Death1.npy', 'L1/L1_Player_Death2.npy',
        # L2
        *(f'L2/L2_Player_Pos{i}.npy' for i in range(8)),
        'L2/L2_Player_Death1.npy', 'L2/L2_Player_Death2.npy',
        # L3
        *(f'L3/L3_Player_Pos{i}.npy' for i in range(8)),
        'L3/L3_Player_Death1.npy', 'L3/L3_Player_Death2.npy',
        # L4
        *(f'L4/L4_Player_Pos{i}.npy' for i in range(8)),
        'L4/L4_Player_Death1.npy', 'L4/L4_Player_Death2.npy',
        # L5
        *(f'L5/L5_Player_Pos{i}.npy' for i in range(8)),
        'L5/L5_Player_Death1.npy', 'L5/L5_Player_Death2.npy',
        # Transition
        *(f'L-All/TP_Player_Pos{i}.npy' for i in range(8)),
        'L-All/TP_Player_Death.npy',
    ]

    # 75 files: 5 levels * 15 sprites (pos + death + boss)
    all_enemy_sprites_files = [
        # L1 (15 files)
        *(f'L1/L1_Enemy_Pos{i}.npy' for i in range(8)),
        'L1/L1_Enemy_Pos0.npy', 'L1/L1_Enemy_Pos1.npy', # 2 extra
        'L1/L1_Enemy_Death.npy',
        'L1/L1_Boss_Pos0.npy', 'L1/L1_Boss_Pos0.npy', # 2 boss_0
        'L1/L1_Boss_Pos1.npy', 'L1/L1_Boss_Pos1.npy', # 2 boss_1
        # L2 (15 files)
        *(f'L2/L2_Enemy_Pos{i}.npy' for i in range(8)),
        'L2/L2_Enemy_Pos0.npy', 'L2/L2_Enemy_Pos1.npy',
        'L2/L2_Enemy_Death.npy',
        'L2/L2_Boss_Pos0.npy', 'L2/L2_Boss_Pos0.npy',
        'L2/L2_Boss_Pos1.npy', 'L2/L2_Boss_Pos1.npy',
        # L3 (15 files)
        *(f'L3/L3_Enemy_Pos{i}.npy' for i in ["01", "02", "11", "12", "21", "22", "31", "32", "41", "42"]),
        'L3/L3_Enemy_Death.npy',
        *(f'L3/L3_Boss_Pos{i}.npy' for i in ["01", "02", "11", "12"]),
        # L4 (15 files)
        *(f'L4/L4_Enemy_Pos{i}.npy' for i in range(8)),
        'L4/L4_Enemy_Pos0.npy', 'L4/L4_Enemy_Pos1.npy',
        'L4/L4_Enemy_Death.npy',
        'L4/L4_Boss_Pos0.npy', 'L4/L4_Boss_Pos0.npy',
        'L4/L4_Boss_Pos1.npy', 'L4/L4_Boss_Pos1.npy',
        # L5 (15 files)
        *(f'L5/L5_Enemy_Pos{j}.npy' for i in range(5) for j in range(2)), # 10 pos
        'L5/L5_Enemy_Death.npy',
        'L5/L5_Boss_Pos0.npy', 'L5/L5_Boss_Pos0.npy',
        'L5/L5_Boss_Pos1.npy', 'L5/L5_Boss_Pos1.npy',
    ]

    return (
        # Procedural background (empty black screen)
        {'name': 'background', 'type': 'background', 'data': jnp.zeros((210, 160, 4), dtype=jnp.uint8)},
        # Procedural pixel to ensure white is in the palette
        {'name': 'white_pixel', 'type': 'procedural', 'data': jnp.array([[[255,255,255,255]]], dtype=jnp.uint8)},
        # General Sprites (Single)
        {'name': 'top_wall', 'type': 'single', 'file': 'L-All/Top.npy'},
        {'name': 'bottom_wall', 'type': 'single', 'file': 'L-All/Bottom.npy'},
        {'name': 'respawn_bottom_wall', 'type': 'single', 'file': 'L-All/Respawn_Bottom.npy'},
        {'name': 'start_screen', 'type': 'single', 'file': 'L-All/First.npy'},
        {'name': 'player_life', 'type': 'single', 'file': 'L-All/Player_Life.npy'},
        {'name': 'black_line', 'type': 'single', 'file': 'L-All/BlackLine.npy'},
        # General Sprites (Group)
        {'name': 'transition_bar', 'type': 'group', 'files': ['L-All/TeleportBar.npy', 'L-All/TeleportBar2.npy']},
        # General Sprites (Digits)
        {'name': 'digits', 'type': 'digits', 'pattern': 'L-All/Digit{}.npy'},
        # --- Level-Dependent Groups (for unified padding) ---
        {'name': 'all_clouds', 'type': 'group', 'files': [f'L{i}/L{i}_Cloud.npy' for i in range(1, 6)]},
        {'name': 'all_backgrounds', 'type': 'group', 'files': [f'L{i}/L{i}_Background.npy' for i in range(1, 6)]},
        {'name': 'all_respawn_top_walls', 'type': 'group', 'files': [f'L{i}/L{i}_Top.npy' for i in range(1, 6)]},
        {'name': 'all_player_missiles', 'type': 'group', 'files': [f'L{i}/L{i}_Player_Bullet.npy' for i in range(1, 6)]},
        {'name': 'all_enemy_missiles', 'type': 'group', 'files': [f'L{i}/L{i}_Enemy_Bullet.npy' for i in range(1, 6)]},
        {'name': 'all_enemy_remaining', 'type': 'group', 'files': [
            item for i in range(1, 6) for item in (f'L{i}/L{i}_Enemy_Life.npy', f'L{i}/L{i}_Enemy_Death_Life.npy')
        ]},
        # Massive groups
        {'name': 'all_player_sprites', 'type': 'group', 'files': all_player_sprites_files},
        {'name': 'all_enemy_sprites', 'type': 'group', 'files': all_enemy_sprites_files},
    )

class TimePilotConstants(AutoDerivedConstants):
    # Constants for game environment
    WIDTH: int = struct.field(pytree_node=False, default=160)
    HEIGHT: int = struct.field(pytree_node=False, default=210)

    # Object sizes (width, height)
    PLAYER_SIZE_PER_ROTATION: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([
        (7, 14), # up
        (8, 10),
        (8, 9), # left
        (8, 10),
        (7, 14), # down
        (8, 10),
        (8, 9), # right
        (8, 10)
    ]))
    MISSILE_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: (1, 2))

    # Rendering constants
    WALL_TOP_HEIGHT: int = struct.field(pytree_node=False, default=32)
    WALL_BOTTOM_HEIGHT: int = struct.field(pytree_node=False, default=16)
    BLACK_BORDER_TOP_HEIGHT: int = struct.field(pytree_node=False, default=0)
    BLACK_BORDER_BOTTOM_HEIGHT: int = struct.field(pytree_node=False, default=17)

    # Player constants
    PLAYER_X: int = struct.field(pytree_node=False, default=76)
    PLAYER_Y: int = struct.field(pytree_node=False, default=None)
    INITIAL_PLAYER_ROTATION: int = struct.field(pytree_node=False, default=2)
    PLAYER_SPEED_PER_ROTATION: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([
        (0, -4), # up
        (-4, -4),
        (-4, 0), # left
        (-4, 4),
        (0, 4), # right
        (4, 4),
        (4, 0), # down
        (4, -4)
    ]))
    # speed is different in every step (the pattern is recurring every eighth step)
    PLAYER_MISSILE_SPEED_PER_ROTATION: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([
        ((0, -4), (0, 4), (0, -8), (0, 4), (0, -4), (0, 4), (0, -4), (0, 0)), # up
        ((-4, -4), (4, 4), (-8, -8), (4, 4), (-4, -4), (4, 4), (-4, -4), (0, 0)),
        ((-4, 0), (4, 0), (-8, 0), (4, 0), (-4, 0), (4, 0), (-4, 0), (0, 0)), # left
        ((-4, 4), (4, -4), (-8, 8), (4, -4), (-4, 4), (4, -4), (-4, 4), (0, 0)),
        ((0, 4), (0, -4), (0, 8), (0, -4), (0, 4), (0, -4), (0, 4), (0, 0)), # down
        ((4, 4), (-4, -4), (8, 8), (-4, -4), (4, 4), (-4, -4), (4, 4), (0, 0)),
        ((4, 0), (-4, 0), (8, 0), (-4, 0), (4, 0), (-4, 0), (4, 0), (0, 0)), # right
        ((4, -4), (-4, 4), (8, -8), (-4, 4), (4, -4), (-4, 4), (4, -4), (0, 0))
    ]))

    # Cloud constants
    INITIAL_CLOUDS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([ 
        (16, 65),
        (96, 65),
        (16, 129),
        (96, 129)
    ]))

    # Game constants
    INITIAL_ENEMIES_REMAINING: int = struct.field(pytree_node=False, default=8)
    INITIAL_LIVES: int = struct.field(pytree_node=False, default=5)
    MAX_LIVES: int = struct.field(pytree_node=False, default=5)
    MAX_SCORE: int = struct.field(pytree_node=False, default=999900)

    POINTS_PER_ENEMY: int = struct.field(pytree_node=False, default=100)
    POINTS_PER_BOSS: int = struct.field(pytree_node=False, default=3000)
    POINTS_TO_GAIN_A_LIFE: int = struct.field(pytree_node=False, default=10000)
    ENEMY_KILLS_TO_ELIMINATE: int = struct.field(pytree_node=False, default=4) 

    MAX_ENTITY_X: int = struct.field(pytree_node=False, default=None)
    MAX_ENTITY_Y: int = struct.field(pytree_node=False, default=None)
    MIN_ENTITY_X: int = struct.field(pytree_node=False, default=0)
    MIN_ENTITY_Y: int = struct.field(pytree_node=False, default=None)

    # Animation and respawn delays
    START_SCREEN_DELAY: int = struct.field(pytree_node=False, default=120)
    PLAYER_DEATH_ANIMATION_DELAY: int = struct.field(pytree_node=False, default=64)
    TRANSITION_DELAY_FIRST_STAGE: int = struct.field(pytree_node=False, default=128)
    TRANSITION_DELAY_SECOND_STAGE: int = struct.field(pytree_node=False, default=128)
    TRANSITION_DELAY: int = struct.field(pytree_node=False, default=None)
    ENEMY_DEATH_ANIMATION_DELAY: int = struct.field(pytree_node=False, default=24)

    # Enemy constants
    MAX_NUMBER_OF_ENEMIES: int = struct.field(pytree_node=False, default=4)
    MAX_ENEMY_MISSILES: int = struct.field(pytree_node=False, default=2)
    INITIAL_ATTACK_DELAY: int = struct.field(pytree_node=False, default=150) # mean delay for first attack after player respawn
    ENEMY_MISSILE_SPEED_PER_ROTATION: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([
        (0, -4), # up
        (-4, -4),
        (-4, 0), # left
        (-4, 4),
        (0, 4), # down
        (4, 4),
        (4, 0), # right
        (4, -4)
    ]))
    
    # Level constants
    LEVEL_1: LevelConstants = struct.field(pytree_node=False, default=TimePilot_Level_1)
    LEVEL_2: LevelConstants = struct.field(pytree_node=False, default=TimePilot_Level_2)
    LEVEL_3: LevelConstants = struct.field(pytree_node=False, default=TimePilot_Level_3)
    LEVEL_4: LevelConstants = struct.field(pytree_node=False, default=TimePilot_Level_4)
    LEVEL_5: LevelConstants = struct.field(pytree_node=False, default=TimePilot_Level_5)

    # Asset config baked into constants (immutable default) for asset overrides
    ASSET_CONFIG: tuple = struct.field(pytree_node=False, default=_get_default_asset_config())

    def compute_derived(self):
        """Compute derived constants based on static fields."""
        return {
            'PLAYER_Y': self.WALL_TOP_HEIGHT + self.BLACK_BORDER_TOP_HEIGHT + 68,
            'MAX_ENTITY_X': self.WIDTH - 1,
            'MAX_ENTITY_Y': self.HEIGHT - self.WALL_BOTTOM_HEIGHT - self.BLACK_BORDER_BOTTOM_HEIGHT - 1,
            'MIN_ENTITY_Y': self.WALL_TOP_HEIGHT + self.BLACK_BORDER_TOP_HEIGHT + 1,
            'TRANSITION_DELAY': self.TRANSITION_DELAY_FIRST_STAGE + self.TRANSITION_DELAY_SECOND_STAGE
        }

# immutable state container
@struct.dataclass
class TimePilotState:
    player_rotation: chex.Array
    player_active: chex.Array
    player_missile_state: chex.Array # (4, ) array with (x, y, rotation, missile_step_counter)
    enemy_states: chex.Array # (4, 4) array with (x, y, rotation, active) for each enemy
    enemy_missile_states: chex.Array # (2, 4) array with (x, y, rotation, active) for each missile
    level_boss: chex.Array # index of enemy state that is used for level boss (-1 if boss has not spawned yet)
    enemy_shot_timer: chex.Array # used to time enemy attacks

    cloud_positions: chex.Array # (4, 2) array with (x, y) for each cloud
    enemy_death_timers: chex.Array # (4, ) array with death animation timer for each enemy

    level: chex.Array # 1910, 1940, 1970, 1983, 2001
    score: chex.Array
    lives: chex.Array
    enemies_remaining: chex.Array
    missile_rdy: chex.Array # tracks whether the player can fire a missile

    respawn_timer: chex.Array # used for transition animations
    next_level_transition: chex.Array # used to distinguish between player death transition and next level transition
    step_counter: chex.Array
    rng_key: chex.Array
    original_rng_key: chex.Array # keep track of original key to reuse after respawning


@struct.dataclass
class TimePilotObservation:
    player: ObjectObservation
    player_missile: ObjectObservation
    enemies: ObjectObservation  # n=4
    level_boss: ObjectObservation
    enemy_missiles: ObjectObservation  # n=2
    clouds: ObjectObservation # n=4 (Added clouds for completeness)

    level: jnp.ndarray
    score: jnp.ndarray
    lives: jnp.ndarray
    enemies_remaining: jnp.ndarray

@struct.dataclass
class TimePilotInfo:
    level: chex.Array
    score: chex.Array
    lives: chex.Array
    enemies_remaining: chex.Array

class JaxTimePilot(JaxEnvironment[TimePilotState, TimePilotObservation, TimePilotInfo, TimePilotConstants]):
    # Minimal ALE action set for Time Pilot (from scripts/action_space_helper.py)
    ACTION_SET: jnp.ndarray = jnp.array(
        [
            Action.NOOP,
            Action.FIRE,
            Action.UP,
            Action.RIGHT,
            Action.LEFT,
            Action.DOWN,
            Action.UPFIRE,
            Action.RIGHTFIRE,
            Action.LEFTFIRE,
            Action.DOWNFIRE,
        ],
        dtype=jnp.int32,
    )

    def __init__(self, consts: TimePilotConstants|None = None):
        consts = consts or TimePilotConstants()
        super().__init__(consts)
        self.obs_size = 3*6 + self.consts.MAX_NUMBER_OF_ENEMIES*6 + self.consts.MAX_ENEMY_MISSILES*6 + 4
        self.renderer = TimePilotRenderer(consts)

    @partial(jax.jit, static_argnums=(0,))
    def _get_level_constants(self, current_level: int) -> LevelConstants:
        """
        Get constants according to current level
        """
        return jax.lax.switch(
            current_level - 1,
            [
                lambda: self.consts.LEVEL_1,
                lambda: self.consts.LEVEL_2,
                lambda: self.consts.LEVEL_3,
                lambda: self.consts.LEVEL_4,
                lambda: self.consts.LEVEL_5
            ]
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def final_pos(self, min_pos, max_pos, pos):
        """
        Handles wrap-around
        """
        return ((pos - min_pos)%(max_pos - min_pos)) + min_pos

    @partial(jax.jit, static_argnums=(0,))
    def player_step(
        self,
        state_player_rotation,
        action
    ):
        # get pressed buttons
        left = jnp.logical_or(action == Action.LEFT, action == Action.LEFTFIRE)
        right = jnp.logical_or(action == Action.RIGHT, action == Action.RIGHTFIRE)
        up = jnp.logical_or(action == Action.UP, action == Action.UPFIRE)
        down = jnp.logical_or(action == Action.DOWN, action == Action.DOWNFIRE)

        # determine new rotation according to action (up = 0, left = 2, down = 4, right = 6)
        new_rotation = jax.lax.cond(
            jnp.logical_or(up, down),
            lambda: 0,
            lambda: 2
        )
        new_rotation = jax.lax.cond(
            jnp.logical_or(down, right),
            lambda: new_rotation + 4,
            lambda: new_rotation
        )

        # determine new rotation
        player_rotation = state_player_rotation

        # rotate player right
        player_rotation = jax.lax.cond(
            jnp.isin((state_player_rotation - new_rotation) % 8, jnp.array(range(1, 4))),
            lambda: (player_rotation - 1) % 8,
            lambda: player_rotation
        )
        # rotate player left
        player_rotation = jax.lax.cond(
            jnp.isin((new_rotation - state_player_rotation) % 8, jnp.array(range(1, 4))),
            lambda: (player_rotation + 1) % 8,
            lambda: player_rotation
        )
        # rotate player left if new rotation is opposite of current rotation
        player_rotation = jax.lax.cond(
            (new_rotation + 4) % 8 == state_player_rotation,
            lambda: (player_rotation + 1) % 8,
            lambda: player_rotation
        )

        return jax.lax.cond(
            jnp.logical_or(jnp.logical_or(up, down), jnp.logical_or(right, left)),
            lambda: player_rotation,
            lambda: state_player_rotation
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def enemy_step(
        self,
        level,
        level_boss,
        state_player_rotation,
        state_enemy_states,
        position_change,
        respawn_timer,
        state_rng_key
    ):     
        level_constants = self._get_level_constants(level)
        
        # move enemy without rotation
        def enemy_position_change(enemy_states):
            def update_enemy_position(enemy_state, i):
                # move enemy
                new_x = enemy_state[0] + level_constants.enemy_speed_per_rotation[enemy_state[2]][0] # x direction
                new_y = enemy_state[1] + level_constants.enemy_speed_per_rotation[enemy_state[2]][1] # y direction
                # move view
                new_x = new_x - self.consts.PLAYER_SPEED_PER_ROTATION[state_player_rotation][0] # x direction
                new_y = new_y - self.consts.PLAYER_SPEED_PER_ROTATION[state_player_rotation][1] # y direction
                
                enemy_size = jax.lax.cond(
                    i == level_boss,
                    lambda: level_constants.level_boss_size,
                    lambda: level_constants.enemy_size_per_rotation[enemy_state[2]]
                )

                # only wrap around if enemy has spawned already
                return jax.lax.cond(
                    self.entities_collide(
                        self.consts.MIN_ENTITY_X,
                        self.consts.MIN_ENTITY_Y,
                        self.consts.MAX_ENTITY_X - self.consts.MIN_ENTITY_X,
                        self.consts.MAX_ENTITY_Y - self.consts.MIN_ENTITY_Y,
                        enemy_state[0],
                        enemy_state[1],
                        enemy_size[0],
                        enemy_size[1]
                    ),
                    lambda: jnp.array([
                        self.final_pos(self.consts.MIN_ENTITY_X, self.consts.MAX_ENTITY_X, new_x),
                        self.final_pos(self.consts.MIN_ENTITY_Y, self.consts.MAX_ENTITY_Y, new_y),
                        enemy_state[2], enemy_state[3]]),
                    lambda: jnp.array([
                        self.final_pos(self.consts.MIN_ENTITY_X - 50, self.consts.MAX_ENTITY_X + 50, new_x),
                        self.final_pos(self.consts.MIN_ENTITY_Y - 50, self.consts.MAX_ENTITY_Y + 50, new_y),
                        enemy_state[2], enemy_state[3]]),
                )

            return jax.vmap(update_enemy_position)(enemy_states, jnp.array(range(self.consts.MAX_NUMBER_OF_ENEMIES)))
        
        # rotate enemy without changing position
        def enemy_rotation_change(enemy_states, rng_key):

            rng_key, subkey = jax.random.split(rng_key)
            new_rotation = jax.lax.cond(
                jax.random.randint(subkey, [], 0, 2) == 0,
                lambda: (enemy_states[0][2] - 1) % 8, # rotate right with 50% probability
                lambda: (enemy_states[0][2] + 1) % 8 # rotate left with 50% probability
            )

            rng_key, subkey = jax.random.split(rng_key)
            new_rotation = jax.lax.cond(
                # only change rotation with probability 1/enemy_rotation_probability
                jax.random.randint(subkey, [], 0, level_constants.enemy_rotation_probability) == 0, 
                lambda: new_rotation,
                lambda: enemy_states[0][2] # old rotation
            )

            def update_enemy_rotation(enemy_state):
                # set enemy_states rotation to new_rotation
                return jnp.array([enemy_state[0], enemy_state[1], 
                                                         new_rotation, enemy_state[3]])

            return jax.vmap(update_enemy_rotation)(enemy_states), rng_key
        
        # alternate changing enemy position and rotation according to step counter
        new_enemy_states, rng_key = jax.lax.cond(
            position_change, # rotation change if false
            lambda: (enemy_position_change(state_enemy_states), state_rng_key),
            lambda: enemy_rotation_change(state_enemy_states, state_rng_key)
        )

        # shake single enemy
        def shake_enemy(enemy_state, i):
            
            # shake enemy vertically
            new_y = jax.lax.cond(
                respawn_timer % 16 < 8,
                lambda: enemy_state[1] - 4,
                lambda: enemy_state[1] + 4
            )

            enemy_size = jax.lax.cond(
                    i == level_boss,
                    lambda: level_constants.level_boss_size,
                    lambda: level_constants.enemy_size_per_rotation[enemy_state[2]]
                )

            # only wrap around if enemy has spawned already
            return jax.lax.cond(
                    self.entities_collide(
                        self.consts.MIN_ENTITY_X,
                        self.consts.MIN_ENTITY_Y,
                        self.consts.MAX_ENTITY_X - self.consts.MIN_ENTITY_X,
                        self.consts.MAX_ENTITY_Y - self.consts.MIN_ENTITY_Y,
                        enemy_state[0],
                        enemy_state[1],
                        enemy_size[0],
                        enemy_size[1]
                    ),
                    lambda: jnp.array(
                        [enemy_state[0],
                         self.final_pos(self.consts.MIN_ENTITY_Y - 50, self.consts.MAX_ENTITY_Y + 50, new_y), 
                         enemy_state[2],
                         enemy_state[3]]
                    ),
                    lambda: jnp.array(
                        [enemy_state[0],
                         new_y, 
                         enemy_state[2],
                         enemy_state[3]]
                    )
                )

        # either just move enemies or add shaking after player death
        return jax.lax.cond(
            jnp.logical_and(jnp.logical_not(position_change),
                            jnp.logical_and(respawn_timer >= self.consts.TRANSITION_DELAY, 
                                            respawn_timer <= self.consts.TRANSITION_DELAY + self.consts.PLAYER_DEATH_ANIMATION_DELAY)),
            lambda: (jax.vmap(shake_enemy)(new_enemy_states, jnp.array(range(self.consts.MAX_NUMBER_OF_ENEMIES))), rng_key),
            lambda: (new_enemy_states, rng_key)
        )

    @partial(jax.jit, static_argnums=(0,))
    def cloud_step(
        self,
        state_cloud_positions,
        player_rotation,
        respawn_timer
    ):
        
        # update single cloud position
        def update_cloud(cloud_position):

            # move cloud in opposite direction of player
            new_x = cloud_position[0] - self.consts.PLAYER_SPEED_PER_ROTATION[player_rotation][0]
            new_y = cloud_position[1] - self.consts.PLAYER_SPEED_PER_ROTATION[player_rotation][1]

            return jnp.array(
                        [self.final_pos(self.consts.MIN_ENTITY_X - 32, self.consts.MAX_ENTITY_X, new_x),
                         self.final_pos(self.consts.MIN_ENTITY_Y - 12, self.consts.MAX_ENTITY_Y, new_y)]
                    )
        
        # shake single cloud
        def shake_cloud(cloud_position):

            new_y = jax.lax.cond(
                respawn_timer % 16 < 8,
                lambda: cloud_position[1] - 4,
                lambda: cloud_position[1] + 4
            )

            return jnp.array(
                        [cloud_position[0],
                         self.final_pos(self.consts.MIN_ENTITY_Y - 12, self.consts.MAX_ENTITY_Y, new_y)]
                    )

        # either move clouds according to player rotation or shake clouds after player death
        return jax.lax.cond(
            jnp.logical_and(respawn_timer >= self.consts.TRANSITION_DELAY, 
                            respawn_timer <= self.consts.TRANSITION_DELAY + self.consts.PLAYER_DEATH_ANIMATION_DELAY),
            lambda: jax.vmap(shake_cloud)(state_cloud_positions),
            lambda: jax.vmap(update_cloud)(state_cloud_positions)
        )

    @partial(jax.jit, static_argnums=(0,))
    def player_missile_step(
        self,
        player_missile_state,
        missile_rdy,
        cur_player_rotation,
        action,
        player_active,
        step_counter
    ):
        space = jnp.logical_or(action == Action.FIRE,
                              jnp.logical_or(jnp.logical_or(action == Action.RIGHTFIRE, action == Action.LEFTFIRE),
                                             jnp.logical_or(action == Action.UPFIRE, action == Action.DOWNFIRE)))
        # only allow shooting new missile every fourth step and after cooldown span of 24 steps (also making sure that space cannot be held)
        init_missile = jnp.logical_and(jnp.logical_and(jnp.logical_and(space, (step_counter + 2) % 8 == 0),
                                       jnp.logical_and(player_active,
                                                       jnp.logical_or(player_missile_state[3] <= 0,
                                                                      player_missile_state[3] >= 24))),
                                        missile_rdy)

        # move missile position
        x_offset = self.consts.PLAYER_MISSILE_SPEED_PER_ROTATION[player_missile_state[2]][player_missile_state[3] % 8][0]
        y_offset = self.consts.PLAYER_MISSILE_SPEED_PER_ROTATION[player_missile_state[2]][player_missile_state[3] % 8][1]

        player_missile_state = jax.lax.cond(
            player_missile_state[3] > 0,
            lambda: jnp.array([player_missile_state[0] + x_offset,
                               player_missile_state[1] + y_offset,
                               player_missile_state[2],
                               player_missile_state[3] + 1]),
            lambda: player_missile_state
        )

        # reset missile step counter to 0 if missile is out of screen
        player_missile_state = jax.lax.cond(
            jnp.logical_or(
                jnp.logical_or(player_missile_state[0] < self.consts.MIN_ENTITY_X, player_missile_state[0] > self.consts.MAX_ENTITY_X),
                jnp.logical_or(player_missile_state[1] < self.consts.MIN_ENTITY_Y, player_missile_state[1] > self.consts.MAX_ENTITY_Y)
                ),
            lambda: player_missile_state.at[3].set(0),
            lambda: player_missile_state
        )

        # making sure that missile is shot from the center of the player
        initial_missile_x, initial_missile_y = jax.lax.switch(
            cur_player_rotation,
            [
                lambda: (self.consts.PLAYER_X + 3, self.consts.PLAYER_Y), # up
                lambda: (self.consts.PLAYER_X, self.consts.PLAYER_Y),
                lambda: (self.consts.PLAYER_X, self.consts.PLAYER_Y + 4), # left
                lambda: (self.consts.PLAYER_X, self.consts.PLAYER_Y + 9),
                lambda: (self.consts.PLAYER_X + 3, self.consts.PLAYER_Y + 13), # down
                lambda: (self.consts.PLAYER_X + 7, self.consts.PLAYER_Y + 9),
                lambda: (self.consts.PLAYER_X + 7, self.consts.PLAYER_Y + 4), # right
                lambda: (self.consts.PLAYER_X + 7, self.consts.PLAYER_Y)
            ]
        )
        
        # either shoot new missile or move existing one (if active)
        return jax.lax.cond(
            init_missile,
            lambda: (jnp.array([initial_missile_x,
                               initial_missile_y,
                               cur_player_rotation, 1]), False),
            lambda: (player_missile_state, missile_rdy)
        )

    @partial(jax.jit, static_argnums=(0,))
    def enemy_missile_step(self, missile_states, enemy_states, enemy_attack, rng_key):

        def move_missile(state):
            
            # move missile position
            new_state = jnp.array([state[0] + self.consts.ENEMY_MISSILE_SPEED_PER_ROTATION[state[2]][0],
                                   state[1] + self.consts.ENEMY_MISSILE_SPEED_PER_ROTATION[state[2]][1],
                                   state[2],
                                   1])

            # set missile to inactive if it is out of screen
            new_state = jax.lax.cond(
                jnp.logical_or(
                    jnp.logical_or(new_state[0] < self.consts.MIN_ENTITY_X, new_state[0] > self.consts.MAX_ENTITY_X),
                    jnp.logical_or(new_state[1] < self.consts.MIN_ENTITY_Y, new_state[1] > self.consts.MAX_ENTITY_Y)
                    ),
                lambda: new_state.at[3].set(0),
                lambda: new_state
            )

            return jax.lax.cond(
                state[3] > 0,
                lambda: new_state,
                lambda: state
                )
        
        missile_states = jax.vmap(move_missile)(missile_states)

        # select enemy for shooting
        rng_key, subkey = jax.random.split(rng_key)
        shooting_enemy_idx = jax.random.randint(subkey, [], 0, self.consts.MAX_NUMBER_OF_ENEMIES)

        # do not create new missile if two already exist
        inactive_map = missile_states[:,3]<=0
        index =  jax.lax.cond(
            jnp.isin(True, inactive_map),
            lambda: jnp.argmax(inactive_map),
            lambda: -1
        )

        # possibly shoot new missile and move existing ones (if active)
        missile_states = jax.lax.cond(
            jnp.logical_and(jnp.logical_and(enemy_attack, index > -1), enemy_states[shooting_enemy_idx][3] > 0),
            lambda: missile_states.at[index].set(jnp.array([enemy_states[shooting_enemy_idx][0], 
                                                            enemy_states[shooting_enemy_idx][1], 
                                                            enemy_states[shooting_enemy_idx][2],
                                                            1])),
            lambda: missile_states
        )

        return missile_states, rng_key

    @partial(jax.jit, static_argnums=(0,))
    def is_hit(
        self,
        e_x,
        e_y,
        e_active,
        e_size,
        missile_state
    ):
        """
        Checks whether entity is hit by missile
        """
        return jax.lax.cond(
            jnp.logical_and(e_active, missile_state[3] > 0),
            lambda: self.entities_collide(e_x, e_y, e_size[0], e_size[1],
                                          missile_state[0], missile_state[1],
                                          self.consts.MISSILE_SIZE[0], self.consts.MISSILE_SIZE[1]),
            lambda: False
        )

    @partial(jax.jit, static_argnums=(0,))
    def entities_collide(
        self,
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
        Checks whether two entities are colliding
        """
        overlap_start_x = jnp.maximum(e1_x, e2_x)
        overlap_end_x = jnp.minimum(e1_x + e1_w, e2_x + e2_w)
        overlap_start_y = jnp.maximum(e1_y, e2_y)
        overlap_end_y = jnp.minimum(e1_y + e1_h, e2_y + e2_h)

        return jnp.logical_and(overlap_start_x < overlap_end_x, overlap_start_y < overlap_end_y)

    @partial(jax.jit, static_argnums=(0,))
    def player_hit(self, level, level_boss, player_rotation, player_active, enemy_states) -> int:
        """
        Returns index of enemy hit by player (or -1 if no enemy is hit)
        """

        level_constants = self._get_level_constants(level)
        
        def get_hit_enemy(enumerated_enemy_state):
            enemy_size = jax.lax.cond(
                    enumerated_enemy_state[4] == level_boss,
                    lambda: level_constants.level_boss_size,
                    lambda: level_constants.enemy_size_per_rotation[enumerated_enemy_state[2]]
                )
            
            # return i if collision of player with enemy at index i occurred, old index otherwise
            return jnp.logical_and(enumerated_enemy_state[3] > 0,
                                self.entities_collide(self.consts.PLAYER_X, self.consts.PLAYER_Y,
                                    self.consts.PLAYER_SIZE_PER_ROTATION[player_rotation][0], 
                                    self.consts.PLAYER_SIZE_PER_ROTATION[player_rotation][1],
                                    enumerated_enemy_state[0], enumerated_enemy_state[1],
                                    enemy_size[0], 
                                    enemy_size[1]))

        hit_map = jax.vmap(get_hit_enemy)(jnp.concat([enemy_states, jnp.expand_dims(jnp.arange(0, self.consts.MAX_NUMBER_OF_ENEMIES), 1)], axis=1))
        return jax.lax.cond(
            jnp.logical_and(player_active, jnp.isin(True, hit_map)),
            lambda: jnp.argmax(hit_map),
            lambda: -1
        )

    @partial(jax.jit, static_argnums=(0,))
    def player_missile_hit(
        self,
        level,
        level_boss,
        player_missile_state,
        enemy_states
    ):
        """
        Returns index of enemy hit by player missile (or -1 if no enemy is hit)
        """

        level_constants = self._get_level_constants(level)
        
        def get_hit_enemy(enumerated_enemy_state):
            enemy_size = jax.lax.cond(
                    enumerated_enemy_state[4] == level_boss,
                    lambda: level_constants.level_boss_size,
                    lambda: level_constants.enemy_size_per_rotation[enumerated_enemy_state[2]]
                )
            
            return self.is_hit(enumerated_enemy_state[0], enumerated_enemy_state[1], enumerated_enemy_state[3] > 0,
                            enemy_size, player_missile_state)

        hit_map = jax.vmap(get_hit_enemy)(jnp.concat([enemy_states, jnp.expand_dims(jnp.arange(0, self.consts.MAX_NUMBER_OF_ENEMIES), 1)], axis=1))
        return jax.lax.cond(
            jnp.isin(True, hit_map),
            lambda: jnp.argmax(hit_map),
            lambda: -1
        )

    @partial(jax.jit, static_argnums=(0,))
    def enemy_missile_hit(
        self,
        player_rotation,
        player_active,
        enemy_missile_states
    ):
        """
        Returns index of enemy missile hitting player (or -1 if player is not hit)
        """
        
        def get_hitting_missile(missile_state):
            return self.is_hit(self.consts.PLAYER_X, self.consts.PLAYER_Y, player_active, 
                            self.consts.PLAYER_SIZE_PER_ROTATION[player_rotation], missile_state)

        hit_map = jax.vmap(get_hitting_missile)(enemy_missile_states)
        return jax.lax.cond(
            jnp.isin(True, hit_map),
            lambda: jnp.argmax(hit_map),
            lambda: -1
        )
    
    def attack_delay_transform(self, value):
        """
        Returns final delay between two enemy attacks
        """
        return jnp.array((value-150)**2/75, jnp.int32) + 1

    @partial(jax.jit, static_argnums=(0,))
    def reset_level(self, old_state):
        """
        Returns level with certain values reset for usage after player death or level transition
        """
        return TimePilotState(
            player_rotation=self.consts.INITIAL_PLAYER_ROTATION,
            player_active=1,
            player_missile_state=jnp.zeros((4, )).astype(jnp.int32),
            enemy_states=self._get_level_constants(old_state.level).initial_enemies,
            enemy_missile_states=jnp.zeros((self.consts.MAX_ENEMY_MISSILES, 4)).astype(jnp.int32),
            level_boss=old_state.level_boss,
            enemy_shot_timer=self.consts.INITIAL_ATTACK_DELAY + jax.random.randint(old_state.rng_key, [], -50, 50, jnp.int32),
            cloud_positions=self.consts.INITIAL_CLOUDS,
            enemy_death_timers=jnp.zeros((4, )).astype(jnp.int32),
            missile_rdy=jnp.array(False).astype(jnp.bool),
            level=old_state.level,
            score=old_state.score,
            lives=old_state.lives,
            enemies_remaining=old_state.enemies_remaining,
            respawn_timer=old_state.respawn_timer,
            next_level_transition=old_state.next_level_transition,
            step_counter=old_state.step_counter,
            rng_key=old_state.original_rng_key,
            original_rng_key=old_state.original_rng_key
        )
    
    def reset(self, key: jax.random.PRNGKey = jax.random.PRNGKey(1234)) -> Tuple[TimePilotObservation, TimePilotState]:
        """
        Resets the game state to the initial state.
        Returns the initial state and respective observation.
        """
        state = TimePilotState(

            player_rotation=jnp.array(self.consts.INITIAL_PLAYER_ROTATION).astype(jnp.int32),
            player_active=jnp.array(1).astype(jnp.int32),
            player_missile_state=jnp.zeros((4, )).astype(jnp.int32),
            enemy_states=jnp.array(self.consts.LEVEL_1.initial_enemies).astype(jnp.int32),
            enemy_missile_states=jnp.zeros((self.consts.MAX_ENEMY_MISSILES, 4)).astype(jnp.int32),
            level_boss=jnp.array(-1).astype(jnp.int32),
            enemy_shot_timer=jnp.array(self.consts.INITIAL_ATTACK_DELAY),
            cloud_positions=jnp.array(self.consts.INITIAL_CLOUDS).astype(jnp.int32),
            enemy_death_timers=jnp.zeros((4, )).astype(jnp.int32),

            level=jnp.array(1).astype(jnp.int32),
            score=jnp.array(0).astype(jnp.int32),
            lives=jnp.array(self.consts.INITIAL_LIVES).astype(jnp.int32),
            enemies_remaining=jnp.array(self.consts.INITIAL_ENEMIES_REMAINING).astype(jnp.int32),
            missile_rdy=jnp.array(False).astype(jnp.bool),

            respawn_timer=jnp.array(self.consts.START_SCREEN_DELAY).astype(jnp.int32),
            next_level_transition=jnp.array(0).astype(jnp.int32),
            step_counter=jnp.array(0).astype(jnp.int32),
            rng_key = key,
            original_rng_key = key
        )
        initial_obs = self._get_observation(state)

        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, state: TimePilotState, action: chex.Array
    ) -> Tuple[TimePilotObservation, TimePilotState, float, bool, TimePilotInfo]:
        # Translate compact agent action index to ALE console action
        atari_action = jnp.take(self.ACTION_SET, action.astype(jnp.int32))
        
        level_constants = self._get_level_constants(state.level)
        
        # update clouds (every eighth step only)
        cloud_positions = jax.lax.cond(
            state.step_counter % 8 == 0,
            lambda: self.cloud_step(state.cloud_positions, state.player_rotation, state.respawn_timer),
            lambda: state.cloud_positions
        )

        # update player rotation (every eighth step only) if player is active
        player_rotation = jax.lax.cond(
            jnp.logical_and(jnp.logical_and(state.player_active, state.respawn_timer <= 0), 
                            (state.step_counter + 2) % 8 == 0),
            lambda: self.player_step(state.player_rotation, atari_action),
            lambda: state.player_rotation
        )

        # update player missile (every step)
        player_missile_state, missile_rdy = self.player_missile_step(
            state.player_missile_state, state.missile_rdy, player_rotation, atari_action, 
            jnp.logical_and(state.player_active, state.respawn_timer <= 0), state.step_counter
        )

        # update enemies (every eighth step only)
        enemy_states, rng_key = jax.lax.cond(
            (state.step_counter - 2) % 4 == 0,
            lambda: self.enemy_step(state.level, state.level_boss, player_rotation, state.enemy_states, 
                                    (state.step_counter - 2) % 8 == 0, state.respawn_timer, state.rng_key),
            lambda: (state.enemy_states, state.rng_key)
        )

        # enemy attacks
        # update enemy shot timer
        enemy_shot_timer = jax.lax.cond(
            state.step_counter % 4 == 0,
            lambda: state.enemy_shot_timer - 1,
            lambda: state.enemy_shot_timer
        )
        # update enemy missiles (every fourth step)
        enemy_missile_states, rng_key = jax.lax.cond(
            state.step_counter % 4 == 0,
            lambda: self.enemy_missile_step(state.enemy_missile_states, enemy_states, enemy_shot_timer <= 0, rng_key),
            lambda: (state.enemy_missile_states, rng_key)
        )
        # randomly choose delay until next enemy attack
        rng_key, subkey = jax.random.split(rng_key)
        enemy_shot_timer = jax.lax.cond(
            enemy_shot_timer <= 0,
            lambda: self.attack_delay_transform(jax.random.randint(subkey, [], level_constants.min_attack_delay, 
                                                                   level_constants.max_attack_delay)),
            lambda: enemy_shot_timer
        )

        # handle completely crazy DESTRUCTION!!!
        # get all collisions
        pe_hit = self.player_hit(state.level, state.level_boss, player_rotation, 
                                 jnp.logical_and(state.player_active, state.respawn_timer <= 0), enemy_states)
        pm_hit = self.player_missile_hit(state.level, state.level_boss, player_missile_state, enemy_states)
        em_hit = self.enemy_missile_hit(player_rotation, 
                                        jnp.logical_and(state.player_active, state.respawn_timer <= 0), enemy_missile_states)

        # making sure that enemies do not overlap after respawning
        def check_ee_collision(enumerated_state_1, enumerated_state_2, level_boss):

            enemy_size_i = jax.lax.cond(
                    enumerated_state_1[4] == level_boss,
                    lambda: level_constants.level_boss_size,
                    lambda: level_constants.enemy_size_per_rotation[enumerated_state_1[2]]
                )
            
            enemy_size_j = jax.lax.cond(
                    enumerated_state_2[4] == level_boss,
                    lambda: level_constants.level_boss_size,
                    lambda: level_constants.enemy_size_per_rotation[enumerated_state_2[2]]
                )
            
            return jax.lax.cond(
                enumerated_state_1[4] != enumerated_state_2[4],
                lambda: self.entities_collide(
                            enumerated_state_1[0],
                            enumerated_state_1[1],
                            enemy_size_i[0]*2,
                            enemy_size_i[1]*2,
                            enumerated_state_2[0],
                            enumerated_state_2[1],
                            enemy_size_j[0]*2,
                            enemy_size_j[1]*2
                        ),
                lambda: False
            )
        
        def get_ee_coll_idx(i, enemy_states, level_boss):
            enumerated_states = jnp.concat([enemy_states, jnp.expand_dims(jnp.arange(0, self.consts.MAX_NUMBER_OF_ENEMIES), 1)], axis=1)
            coll_map = jax.vmap(check_ee_collision, in_axes = (None, 0, None))(
                enumerated_states[i],
                enumerated_states,
                level_boss
            )
        
            return jax.lax.cond(
                jnp.isin(True, coll_map),
                lambda: jnp.argmax(coll_map),
                lambda: -1
            )

        # respawn enemy when its death timer is at zero
        def respawn_enemy(i, enemy_states):
            # get respawn states according to level constants as well as index of overlapping, already existing enemy (-1 if ther is no overlap)
            new_enemy_state = jax.lax.cond(
                state.enemy_death_timers[i] == 1,
                lambda: level_constants.initial_enemies[i],
                lambda: enemy_states[i]
            )

            coll_idx = jax.lax.cond(
                state.enemy_death_timers[i] == 1,
                lambda: get_ee_coll_idx(i, enemy_states.at[i].set(new_enemy_state), state.level_boss),
                lambda: -1
            )

            # get direction to move overlapping, new enemy
            mov_dir = jax.lax.cond(
                coll_idx >= 0,
                lambda: (jnp.array(
                    jnp.logical_or(new_enemy_state[1]<self.consts.MIN_ENTITY_Y,
                                   new_enemy_state[1]>self.consts.MAX_ENTITY_Y), 
                                   jnp.int32)*jnp.where(new_enemy_state[0]<enemy_states[coll_idx][0], -1, 1), 
                                   jnp.array(jnp.logical_or(new_enemy_state[0]<self.consts.MIN_ENTITY_X,
                                                            new_enemy_state[0]>self.consts.MAX_ENTITY_X), 
                                                            jnp.int32)*jnp.where(new_enemy_state[1]<enemy_states[coll_idx][1],
                                                                                  -1, 1)),
                lambda: (0, 0)
            )

            enemy_size = jax.lax.cond(
                    i == state.level_boss,
                    lambda: level_constants.level_boss_size,
                    lambda: level_constants.enemy_size_per_rotation[new_enemy_state[2]]
                )

            # determine offsets to move overlapping, new enemy
            x_offset = jax.lax.cond(
                coll_idx >= 0,
                lambda: enemy_size[0]*2*mov_dir[0],
                lambda: 0
            )
            y_offset = jax.lax.cond(
                coll_idx >= 0,
                lambda: enemy_size[1]*2*mov_dir[1],
                lambda: 0
            )

            # update enemy states accordingly
            new_enemy_state = jax.lax.fori_loop(
                0,
                self.consts.MAX_NUMBER_OF_ENEMIES,
                lambda _, e_state: jax.lax.cond(
                    (get_ee_coll_idx(i, enemy_states.at[i].set(e_state), state.level_boss) >= 0),
                    lambda: jnp.array([e_state[0] + x_offset, e_state[1] + y_offset,
                                                        e_state[2], e_state[3]]),
                    lambda: e_state
                ),
                new_enemy_state
            )

            return jax.lax.cond(
                state.enemy_death_timers[i] == 1,
                lambda: new_enemy_state,
                lambda: enemy_states[i]
            )

        enemy_states = jax.vmap(respawn_enemy, in_axes = (0, None))(jnp.arange(self.consts.MAX_NUMBER_OF_ENEMIES), enemy_states)

        # update death timer for colliding enemies
        def update_enemy_death_timer(enemy_death_timer):
            return jax.lax.cond(
                enemy_death_timer > 0,
                lambda: enemy_death_timer - 1,
                lambda: enemy_death_timer
            )
        enemy_death_timers = jax.vmap(update_enemy_death_timer)(state.enemy_death_timers)

        # handle collision of enemy with player
        enemy_states, lives, score, enemies_remaining, enemy_death_timers = jax.lax.cond(
            pe_hit > -1,
            lambda: (enemy_states.at[pe_hit].set(jnp.array([enemy_states[pe_hit][0], enemy_states[pe_hit][1], enemy_states[pe_hit][2], 0])),
                     state.lives - 1,
                     state.score + self.consts.POINTS_PER_ENEMY,
                     state.enemies_remaining - 1,
                     enemy_death_timers.at[pe_hit].set(self.consts.ENEMY_DEATH_ANIMATION_DELAY)), 
            lambda: (enemy_states, state.lives, state.score, state.enemies_remaining, enemy_death_timers)
        )

        # handle collision of enemy with player missile
        enemy_states, player_missile_state, score, enemies_remaining, enemy_death_timers = jax.lax.cond(
            pm_hit > -1,
            lambda: (enemy_states.at[pm_hit].set(jnp.array([enemy_states[pm_hit][0], enemy_states[pm_hit][1], enemy_states[pm_hit][2], 0])),
                     jnp.array([0, 0, 0, 0]),
                     score + self.consts.POINTS_PER_ENEMY,
                     enemies_remaining - 1,
                     enemy_death_timers.at[pm_hit].set(self.consts.ENEMY_DEATH_ANIMATION_DELAY)),
            lambda: (enemy_states, player_missile_state, score, enemies_remaining, enemy_death_timers)
        )

        # handle collision of player with enemy missile
        lives, enemy_missile_states = jax.lax.cond(
            em_hit > -1,
            lambda: (state.lives - 1, 
                     enemy_missile_states.at[em_hit].set(jnp.array([0, 0, 0, 0]))),
            lambda: (lives, enemy_missile_states)
        )

        # reset score if it exceeds maximum
        score = jax.lax.cond(
            score >= self.consts.MAX_SCORE,
            lambda: score - self.consts.MAX_SCORE,
            lambda: score
        )

        # spawn level boss if enemies remaining line is empty
        level_boss = jax.lax.cond(
            jnp.logical_and(enemies_remaining <= 0, state.level_boss < 0),
            lambda: jax.lax.cond(
                pe_hit > -1,
                lambda: pe_hit,
                lambda: pm_hit
            ),
            lambda: state.level_boss
        )

        # update respawn timer if it is not zero already
        respawn_timer = jax.lax.cond(
            state.respawn_timer > 0,
            lambda: state.respawn_timer - 1,
            lambda: state.respawn_timer
        )
        # set respawn timer when player is hit
        respawn_timer, player_active = jax.lax.cond(
            jnp.logical_or(pe_hit > -1, em_hit > -1),
            lambda: (self.consts.TRANSITION_DELAY + self.consts.PLAYER_DEATH_ANIMATION_DELAY, 0),
            lambda: (respawn_timer, state.player_active)
        )

        # go to next level if level boss was killed and despawn boss
        level, score, enemies_remaining, respawn_timer, level_boss, next_level_transition = jax.lax.cond(
            jnp.logical_and(state.level_boss > -1,
                            jnp.logical_or(
                                pe_hit == state.level_boss,
                                pm_hit == state.level_boss
                            )),
            lambda: (state.level + 1, 
                     score + self.consts.POINTS_PER_BOSS - self.consts.POINTS_PER_ENEMY,
                     self.consts.INITIAL_ENEMIES_REMAINING,
                     self.consts.TRANSITION_DELAY + self.consts.ENEMY_DEATH_ANIMATION_DELAY,
                     -1, 1),
            lambda: (state.level, score, enemies_remaining, respawn_timer, level_boss, state.next_level_transition)
        )
        # restart at level 1 after beating level 5
        level = jax.lax.cond(
            level > 5,
            lambda: 1,
            lambda: level
        )

        # end next level transition if respawn timer hits 0
        next_level_transition = jax.lax.cond(
            state.respawn_timer == 1,
            lambda: 0,
            lambda: next_level_transition,
        )

        # update lives
        lives = lives + score//self.consts.POINTS_TO_GAIN_A_LIFE - state.score//self.consts.POINTS_TO_GAIN_A_LIFE
        # limit lives at maximum
        lives = jax.lax.cond(
            lives > self.consts.MAX_LIVES,
            lambda: self.consts.MAX_LIVES,
            lambda: lives
        )

        # update step counter (reset if lives are zero)
        step_counter = jax.lax.cond(
            lives <= 0,
            lambda: 0,
            lambda: state.step_counter + 1
        )

        # freeze every non-shaking value during shaking
        (player_rotation, level, lives, rng_key) = jax.lax.cond(
            jnp.logical_and(jnp.logical_and(respawn_timer >= self.consts.TRANSITION_DELAY,
                                            next_level_transition <= 0),
                            respawn_timer < self.consts.TRANSITION_DELAY + self.consts.PLAYER_DEATH_ANIMATION_DELAY),
            lambda: (state.player_rotation, state.level, state.lives, state.rng_key),
            lambda: (player_rotation, level, lives, rng_key)
        )

        # determine whether new missile can be shot in next step
        missile_rdy = jnp.logical_or(missile_rdy,
                                     jnp.logical_not(
                                         jnp.logical_or(
                                             jnp.logical_or(
                                                 jnp.logical_or(
                                                     action == Action.FIRE, action == Action.UPFIRE),
                                                 action == Action.DOWNFIRE),
                                             jnp.logical_or(action == Action.RIGHTFIRE, action == Action.LEFTFIRE))))

        # do not update game state during transition screen
        new_state = jax.lax.cond(
            jnp.logical_and(respawn_timer > 1, respawn_timer < self.consts.TRANSITION_DELAY),
            lambda: TimePilotState(
                player_rotation=state.player_rotation,
                player_active=state.player_active,
                player_missile_state=state.player_missile_state,
                enemy_states=state.enemy_states,
                enemy_missile_states=state.enemy_missile_states,
                level_boss=state.level_boss,
                enemy_shot_timer=state.enemy_shot_timer,
                cloud_positions=state.cloud_positions,
                enemy_death_timers=state.enemy_death_timers,
                missile_rdy=state.missile_rdy,
                level=state.level,
                score=state.score,
                lives=state.lives,
                enemies_remaining=state.enemies_remaining,
                respawn_timer=respawn_timer,
                next_level_transition=next_level_transition,
                step_counter=step_counter,
                rng_key=state.rng_key,
                original_rng_key=state.original_rng_key
            ),
            lambda: TimePilotState(
                player_rotation=player_rotation,
                player_active=player_active,
                player_missile_state=player_missile_state,
                enemy_states=enemy_states,
                enemy_missile_states=enemy_missile_states,
                level_boss=level_boss,
                enemy_shot_timer=enemy_shot_timer,
                cloud_positions=cloud_positions,
                enemy_death_timers=enemy_death_timers,
                missile_rdy=missile_rdy,
                level=level,
                score=score,
                lives=lives,
                enemies_remaining=enemies_remaining,
                respawn_timer=respawn_timer,
                next_level_transition=next_level_transition,
                step_counter=step_counter,
                rng_key=rng_key,
                original_rng_key=state.original_rng_key
            )
        )

        # reset level after transition screen
        new_state = jax.lax.cond(
            jnp.logical_and(state.respawn_timer <= self.consts.TRANSITION_DELAY_SECOND_STAGE, state.respawn_timer > 0),
            lambda: self.reset_level(new_state),
            lambda: new_state
        )

        done = self._get_done(new_state)
        env_reward = self._get_reward(state, new_state)
        info = self._get_info(new_state)
        observation = self._get_observation(new_state)

        return observation, new_state, env_reward, done, info

    def render(self, state: TimePilotState) -> jnp.ndarray:
        return self.renderer.render(state)

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: TimePilotState) -> TimePilotObservation:
        c = self.consts
        w, h = int(c.WIDTH), int(c.HEIGHT)

        # Helper: Rotation (0..7) -> Degrees
        # 0=Up(0), 2=Left(270?), 4=Down(180), 6=Right(90)
        # 8 steps -> 45 degrees per step. 
        # But wait, logic says 0=Up. Typically 0 deg is Up or Right. 
        # Let's map directly: 0->0, 1->45... 7->315? 
        # Check logic:
        # up=0, left=2 (2*45=90? No, left is usually 270 or -90).
        # Let's stick to standard 0=Up/North clockwise or counter-clockwise convention if possible, 
        # or just map raw indices if orientation logic is complex.
        # Assuming 0=Up, clockwise: 0=0, 2=90(Right), 4=180(Down), 6=270(Left).
        # But constants comment says: 0=Up, 2=Left, 4=Down, 6=Right. So Counter-Clockwise?
        # Let's use (rot * 45.0) for now, maybe normalized later.
        def get_ori(rot):
            # Map 0->0 (Up), 2->270 (Left), 4->180 (Down), 6->90 (Right)
            # This is non-linear mapping based on typical conventions.
            return jnp.select(
                [rot == 0, rot == 2, rot == 4, rot == 6],
                [0.0, 270.0, 180.0, 90.0],
                (rot * 45.0) # Diagonal fallback
            ).astype(jnp.float32)

        # --- Player ---
        p_active = jnp.logical_and(state.player_active, state.respawn_timer <= 0).astype(jnp.int32)
        p_size = c.PLAYER_SIZE_PER_ROTATION[state.player_rotation]
        
        player = ObjectObservation.create(
            x=jnp.clip(jnp.array(c.PLAYER_X, dtype=jnp.int32), 0, w),
            y=jnp.clip(jnp.array(c.PLAYER_Y, dtype=jnp.int32), 0, h),
            width=jnp.array(p_size[0], dtype=jnp.int32),
            height=jnp.array(p_size[1], dtype=jnp.int32),
            active=p_active,
            orientation=get_ori(state.player_rotation)
        )

        # --- Player Missile ---
        pm_state = state.player_missile_state
        pm_active = (pm_state[3] > 0).astype(jnp.int32)
        
        player_missile = ObjectObservation.create(
            x=jnp.clip(pm_state[0], 0, w),
            y=jnp.clip(pm_state[1], 0, h),
            width=jnp.array(c.MISSILE_SIZE[0], dtype=jnp.int32),
            height=jnp.array(c.MISSILE_SIZE[1], dtype=jnp.int32),
            active=pm_active,
            orientation=get_ori(pm_state[2])
        )

        # --- Enemies ---
        # State: (x, y, rotation, active)
        e_states = state.enemy_states
        e_timers = state.enemy_death_timers
        
        # Calculate sizes per enemy based on rotation
        # Note: level_boss overrides size if idx match, handled below
        # For now, just use rotation lookup for standard enemies
        # This is tricky without vectorizing lookup of tuples.
        # Approximation: Use max size or average, or vmap the lookup.
        
        def get_enemy_props(i, s, t):
            # Size lookup
            # If boss:
            is_boss = (i == state.level_boss)
            # Level constants are needed for sizes. Accessing _get_level_constants inside vmap might be slow/tricky 
            # if not static. But here we are inside jitted _get_observation.
            # We can use the constants directly.
            lvl_consts = self._get_level_constants(state.level)
            
            size = jax.lax.cond(
                is_boss,
                lambda: lvl_consts.level_boss_size,
                lambda: lvl_consts.enemy_size_per_rotation[s[2]]
            )
            
            active = jnp.logical_and(
                jnp.logical_and(t <= 0, s[3]), 
                jnp.logical_not(is_boss) # Boss is separate object
            ).astype(jnp.int32)
            
            return size[0], size[1], active, get_ori(s[2])

        ew, eh, ea, eo = jax.vmap(get_enemy_props)(
            jnp.arange(c.MAX_NUMBER_OF_ENEMIES), e_states, e_timers
        )
        
        enemies = ObjectObservation.create(
            x=jnp.clip(e_states[:, 0], 0, w),
            y=jnp.clip(e_states[:, 1], 0, h),
            width=ew.astype(jnp.int32),
            height=eh.astype(jnp.int32),
            active=ea,
            orientation=eo
        )

        # --- Level Boss ---
        # Extracted from enemy list if active
        boss_idx = state.level_boss
        boss_active = (boss_idx > -1).astype(jnp.int32)
        
        # Get boss props if active, else dummy
        def get_boss_props():
            s = e_states[boss_idx]
            t = e_timers[boss_idx]
            lvl_consts = self._get_level_constants(state.level)
            size = lvl_consts.level_boss_size
            active = jnp.logical_and(t <= 0, s[3]).astype(jnp.int32)
            return s[0], s[1], size[0], size[1], active, get_ori(s[2])

        bx, by, bw, bh, ba, bo = jax.lax.cond(
            boss_active == 1,
            get_boss_props,
            lambda: (0, 0, 0, 0, 0, 0.0)
        )
        
        level_boss = ObjectObservation.create(
            x=jnp.clip(jnp.array(bx, dtype=jnp.int32), 0, w),
            y=jnp.clip(jnp.array(by, dtype=jnp.int32), 0, h),
            width=jnp.array(bw, dtype=jnp.int32),
            height=jnp.array(bh, dtype=jnp.int32),
            active=jnp.array(ba, dtype=jnp.int32),
            orientation=jnp.array(bo, dtype=jnp.float32)
        )

        # --- Enemy Missiles ---
        em_states = state.enemy_missile_states
        em_active = em_states[:, 3].astype(jnp.int32)
        
        enemy_missiles = ObjectObservation.create(
            x=jnp.clip(em_states[:, 0], 0, w),
            y=jnp.clip(em_states[:, 1], 0, h),
            width=jnp.full((c.MAX_ENEMY_MISSILES,), c.MISSILE_SIZE[0], dtype=jnp.int32),
            height=jnp.full((c.MAX_ENEMY_MISSILES,), c.MISSILE_SIZE[1], dtype=jnp.int32),
            active=em_active,
            orientation=jax.vmap(get_ori)(em_states[:, 2])
        )

        # --- Clouds ---
        cp = state.cloud_positions
        # Clouds always active? Yes, usually background.
        clouds = ObjectObservation.create(
            x=jnp.clip(cp[:, 0], 0, w),
            y=jnp.clip(cp[:, 1], 0, h),
            width=jnp.full((4,), 32, dtype=jnp.int32), # Approx cloud size?
            height=jnp.full((4,), 16, dtype=jnp.int32),
            active=jnp.ones((4,), dtype=jnp.int32),
        )

        return TimePilotObservation(
            player=player,
            player_missile=player_missile,
            enemies=enemies,
            level_boss=level_boss,
            enemy_missiles=enemy_missiles,
            clouds=clouds,
            level=state.level,
            score=state.score,
            lives=state.lives,
            enemies_remaining=state.enemies_remaining
        )

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTION_SET))

    def observation_space(self) -> spaces.Dict:
        c = self.consts
        h = int(c.HEIGHT)
        w = int(c.WIDTH)
        screen_size = (h, w)
        
        single_obj = spaces.get_object_space(n=None, screen_size=screen_size)
        
        return spaces.Dict({
            "player": single_obj,
            "player_missile": single_obj,
            "enemies": spaces.get_object_space(n=c.MAX_NUMBER_OF_ENEMIES, screen_size=screen_size),
            "level_boss": single_obj,
            "enemy_missiles": spaces.get_object_space(n=c.MAX_ENEMY_MISSILES, screen_size=screen_size),
            "clouds": spaces.get_object_space(n=4, screen_size=screen_size),
            
            "level": spaces.Box(low=1, high=5, shape=(), dtype=jnp.int32),
            "score": spaces.Box(low=0, high=c.MAX_SCORE, shape=(), dtype=jnp.int32),
            "lives": spaces.Box(low=0, high=c.MAX_LIVES, shape=(), dtype=jnp.int32),
            "enemies_remaining": spaces.Box(low=0, high=c.INITIAL_ENEMIES_REMAINING, shape=(), dtype=jnp.int32)
        })

    def image_space(self) -> spaces.Box:
        """Returns the image space for TimePilot.
        The image is a RGB image with shape (210, 160, 3).
        """
        return spaces.Box(
            low=0,
            high=255,
            shape=(210, 160, 3),
            dtype=jnp.uint8
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: TimePilotState) -> TimePilotInfo:
        return TimePilotInfo(level=state.level, score=state.score, lives= state.lives, 
                             enemies_remaining=state.enemies_remaining)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: TimePilotState, state: TimePilotState):
        return state.score - previous_state.score

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: TimePilotState) -> bool:
        return state.lives <= 0

class TimePilotRenderer(JAXGameRenderer):
    """JAX-based TimePilot game renderer, optimized with JIT compilation."""

    def __init__(self, consts: TimePilotConstants|None = None, config: render_utils.RendererConfig = None):
        """
        Initializes the renderer by loading and processing all assets.
        """
        self.consts = consts or TimePilotConstants()
        super().__init__(self.consts)
        self.sprite_path = os.path.join(render_utils.get_base_sprite_dir(), "timepilot")

        # Use injected config if provided, else default
        if config is None:
            self.config = render_utils.RendererConfig(
                game_dimensions=(self.consts.HEIGHT, self.consts.WIDTH),
                channels=3,
                downscale=None
            )
        else:
            self.config = config
        self.jr = render_utils.JaxRenderingUtils(self.config)

        # 2. Start from (possibly modded) asset config provided via constants
        final_asset_config = list(self.consts.ASSET_CONFIG)

        # 3. Load, process, and set up all assets in one call
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND, # This will be our empty (black) raster
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(final_asset_config, self.sprite_path)

        # 4. Get specific color IDs we'll need for procedural drawing
        self.BLACK_ID = self.COLOR_TO_ID.get((0, 0, 0), 0)
        self.WHITE_ID = self.COLOR_TO_ID.get((255, 255, 255), 0)

        # 5. Organize loaded masks/offsets into the nested structure
        #    the render() function expects.
        self._post_process_sprites()

    def _post_process_sprites(self):
        """
        Organizes the flat SHAPE_MASKS and FLIP_OFFSETS from setup
        into the nested list[dict] structure that the render() method expects.
        """

        # --- General Sprites ---
        self.general_sprites = {}
        self.general_offsets = {}

        # Simple 1-to-1 mappings
        simple_general_keys = [
            'top_wall', 'bottom_wall', 'respawn_bottom_wall', 'start_screen',
            'player_life', 'black_line', 'transition_bar', 'digits'
        ]
        for key in simple_general_keys:
            self.general_sprites[key] = self.SHAPE_MASKS[key]
            self.general_offsets[key] = self.FLIP_OFFSETS[key]

        # Sliced mappings
        self.general_sprites['transition_player_pos'] = self.SHAPE_MASKS['all_player_sprites'][50:58]
        self.general_sprites['transition_player_death'] = self.SHAPE_MASKS['all_player_sprites'][58]
        self.general_offsets['transition_player_pos'] = self.FLIP_OFFSETS['all_player_sprites']
        self.general_offsets['transition_player_death'] = self.FLIP_OFFSETS['all_player_sprites']

        # --- Level-Dependent Sprites ---
        self.level_sprites = []
        self.level_offsets = []

        for i in range(5):
            level_dict_masks = {}
            level_dict_offsets = {}

            # Simple group slices
            level_dict_masks['cloud'] = self.SHAPE_MASKS['all_clouds'][i]
            level_dict_offsets['cloud'] = self.FLIP_OFFSETS['all_clouds']
            
            level_dict_masks['background'] = self.SHAPE_MASKS['all_backgrounds'][i]
            level_dict_offsets['background'] = self.FLIP_OFFSETS['all_backgrounds']
            
            level_dict_masks['respawn_top_wall'] = self.SHAPE_MASKS['all_respawn_top_walls'][i]
            level_dict_offsets['respawn_top_wall'] = self.FLIP_OFFSETS['all_respawn_top_walls']
            
            level_dict_masks['player_missile'] = self.SHAPE_MASKS['all_player_missiles'][i]
            level_dict_offsets['player_missile'] = self.FLIP_OFFSETS['all_player_missiles']
            
            level_dict_masks['enemy_missile'] = self.SHAPE_MASKS['all_enemy_missiles'][i]
            level_dict_offsets['enemy_missile'] = self.FLIP_OFFSETS['all_enemy_missiles']

            # Complex group slices
            level_dict_masks['player_pos'] = self.SHAPE_MASKS['all_player_sprites'][i*10 : i*10+8]
            level_dict_masks['player_death'] = self.SHAPE_MASKS['all_player_sprites'][i*10+8 : (i+1)*10]
            level_dict_offsets['player_pos'] = self.FLIP_OFFSETS['all_player_sprites']
            level_dict_offsets['player_death'] = self.FLIP_OFFSETS['all_player_sprites']

            level_dict_masks['enemy_pos'] = self.SHAPE_MASKS['all_enemy_sprites'][i*15 : i*15+10]
            level_dict_masks['enemy_death'] = self.SHAPE_MASKS['all_enemy_sprites'][i*15+10]
            level_dict_offsets['enemy_pos'] = self.FLIP_OFFSETS['all_enemy_sprites']
            level_dict_offsets['enemy_death'] = self.FLIP_OFFSETS['all_enemy_sprites']

            # Boss sprites (4 per level)
            boss_sprites = self.SHAPE_MASKS['all_enemy_sprites'][i*15+11 : i*15+15]
            (
                level_dict_masks['level_boss_left_right'],
                level_dict_masks['level_boss_left_left'],
                level_dict_masks['level_boss_right_left'],
                level_dict_masks['level_boss_right_right']
            ) = boss_sprites
            level_dict_offsets['level_boss_left_right'] = self.FLIP_OFFSETS['all_enemy_sprites']
            level_dict_offsets['level_boss_left_left'] = self.FLIP_OFFSETS['all_enemy_sprites']
            level_dict_offsets['level_boss_right_left'] = self.FLIP_OFFSETS['all_enemy_sprites']
            level_dict_offsets['level_boss_right_right'] = self.FLIP_OFFSETS['all_enemy_sprites']

            # Enemy remaining indicators (2 per level)
            level_dict_masks['enemy_remaining'] = self.SHAPE_MASKS['all_enemy_remaining'][i*2]
            level_dict_masks['enemy_remaining_brown'] = self.SHAPE_MASKS['all_enemy_remaining'][i*2+1]
            level_dict_offsets['enemy_remaining'] = self.FLIP_OFFSETS['all_enemy_remaining']
            level_dict_offsets['enemy_remaining_brown'] = self.FLIP_OFFSETS['all_enemy_remaining']
            
            self.level_sprites.append(level_dict_masks)
            self.level_offsets.append(level_dict_offsets)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: TimePilotState) -> chex.Array:
        """
        Renders the current game state using JAX operations.

        Args:
            state: A TimePilotState object containing the current game state.
        Returns:
            A JAX array representing the rendered frame.
        """

        # --- 1. Setup ---
        # Start with the empty background raster (all BLACK_ID)
        raster = self.jr.create_object_raster(self.BACKGROUND)

        # Get general sprites
        general_sprites = self.general_sprites
        general_offsets = self.general_offsets

        # Get sprites for the current level
        # (use sprites for preceding level during first half of level transition animation)
        level_idx = state.level - 1
        level_idx_prev = jnp.maximum(0, level_idx - 1)
        
        # Select correct index for level sprites
        current_level_idx = jax.lax.cond(
            jnp.logical_and(state.next_level_transition, state.respawn_timer >= self.consts.TRANSITION_DELAY_SECOND_STAGE),
            lambda: jax.lax.switch(level_idx, [lambda: 4, lambda: 0, lambda: 1, lambda: 2, lambda: 3]), # Use prev level (5->4, 1->0, etc.)
            lambda: jax.lax.switch(level_idx, [lambda: 0, lambda: 1, lambda: 2, lambda: 3, lambda: 4]), # Use current level
        )

        # Select the correct dictionary of SHAPE_MASKS and OFFSETS
        level_sprites = jax.tree_util.tree_map(lambda *args: jnp.stack(args)[current_level_idx], *self.level_sprites)
        level_offsets = jax.tree_util.tree_map(lambda *args: jnp.stack(args)[current_level_idx], *self.level_offsets)

        # --- 2. Render Background Elements ---
        # Background texture (ground)
        raster = self.jr.render_at(
            raster, 0, self.consts.MIN_ENTITY_Y,
            level_sprites['background'],
            flip_offset=level_offsets['background']
        )

        # --- 3. Render Player ---
        # Select player sprite mask based on state
        player_sprite_mask = level_sprites['player_pos'][state.player_rotation]
        
        # Check death animation state
        is_death_frame_1 = jnp.logical_and(
                state.player_active <= 0,
                jnp.logical_or(
                jnp.isin(state.respawn_timer, jnp.array(range(317, 320))), 
                jnp.logical_or(
                    jnp.logical_or(
                        jnp.isin(state.respawn_timer, jnp.array(range(301, 309))), 
                        jnp.isin(state.respawn_timer, jnp.array(range(285, 293)))), 
                    jnp.logical_or(
                        jnp.isin(state.respawn_timer, jnp.array(range(269, 277))), 
                        jnp.isin(state.respawn_timer, jnp.array(range(256, 261)))))))
        
        is_death_frame_2 = jnp.logical_and(
                state.player_active <= 0,
                jnp.logical_or(
                    jnp.logical_or(
                        jnp.isin(state.respawn_timer, jnp.array(range(309, 317))), 
                        jnp.isin(state.respawn_timer, jnp.array(range(293, 301)))), 
                    jnp.logical_or(
                        jnp.isin(state.respawn_timer, jnp.array(range(277, 285))), 
                        jnp.isin(state.respawn_timer, jnp.array(range(261, 269))))))

        player_sprite_mask = jax.lax.cond(
            is_death_frame_1, lambda: level_sprites['player_death'][0], lambda: player_sprite_mask
        )
        player_sprite_mask = jax.lax.cond(
            is_death_frame_2, lambda: level_sprites['player_death'][1], lambda: player_sprite_mask
        )

        # Check for level transition (white player)
        in_transition_anim = jnp.logical_and(state.next_level_transition > 0, state.respawn_timer <= self.consts.TRANSITION_DELAY)
        
        player_sprite_mask = jax.lax.cond(
            in_transition_anim,
            lambda: general_sprites['transition_player_pos'][state.player_rotation],
            lambda: player_sprite_mask
        )
        player_sprite_mask = jax.lax.cond(
            jnp.logical_and(state.player_active <= 0, in_transition_anim),
            lambda: general_sprites['transition_player_death'],
            lambda: player_sprite_mask
        )

        # Get the correct flip offset (all player sprites share one)
        player_offset = level_offsets['player_pos'] # This is just self.FLIP_OFFSETS['all_player_sprites']

        # Render player (conditionally)
        raster = jax.lax.cond(
            jnp.logical_or(state.respawn_timer < self.consts.TRANSITION_DELAY_SECOND_STAGE,
                           state.respawn_timer >= self.consts.TRANSITION_DELAY),
            lambda r: self.jr.render_at(r, self.consts.PLAYER_X, self.consts.PLAYER_Y, player_sprite_mask, flip_offset=player_offset), 
            lambda r: r,
            raster
        )

        # --- 4. Render Player Missile ---
        raster = jax.lax.cond(
            state.player_missile_state[3] > 0,
            lambda r: self.jr.render_at_clipped(
                r, state.player_missile_state[0], state.player_missile_state[1],
                level_sprites['player_missile'],
                flip_offset=level_offsets['player_missile']
            ),
            lambda r: r,
            raster
        )

        # --- 5. Render Enemies ---
        # Helper functions remain inside, they are JAX-pure
        def get_plane_frame(i, current_level_sprites, current_level_offsets):
            enemy_sprite_mask = current_level_sprites['enemy_pos'][state.enemy_states[0][2]]
            enemy_sprite_mask = jax.lax.cond(
                state.level_boss == i,
                lambda: jax.lax.cond(
                    jnp.isin(state.enemy_states[i][2], jnp.array(range(1, 5))),
                    lambda: current_level_sprites['level_boss_left_left'],
                    lambda: current_level_sprites['level_boss_right_right']
                ),
                lambda: enemy_sprite_mask
            )
            enemy_sprite_mask = jax.lax.cond(
                state.enemy_death_timers[i] > 0,
                lambda: current_level_sprites['enemy_death'],
                lambda: enemy_sprite_mask
            )
            return enemy_sprite_mask, (0, 0)
        
        def get_heli_frame(i, current_level_sprites, current_level_offsets):
            frame_idx = jnp.array((state.step_counter%8) >= 4, jnp.int32)
            sprite_idx = ((state.enemy_states[i][2]%2) + 1 + jnp.array(
                state.enemy_states[i][2]>4, jnp.int32)*2)*jnp.array((state.enemy_states[i][2]%4)>0, jnp.int32)
            
            render_offset = jax.lax.cond(
                state.level == 5,
                lambda: (0, 0),
                lambda: (jnp.array(sprite_idx==0, jnp.int32)*frame_idx, 
                         (jnp.array(sprite_idx==4, jnp.int32)-jnp.array(sprite_idx==2, jnp.int32))*frame_idx)
            )
            
            enemy_sprite_mask = current_level_sprites['enemy_pos'][2*sprite_idx + frame_idx]
            
            # Boss logic
            boss_mask_A = jnp.array(current_level_sprites['level_boss_left_left']*frame_idx + 
                                  current_level_sprites['level_boss_left_right']*(1-frame_idx), jnp.uint8)
            boss_mask_B = jnp.array(current_level_sprites['level_boss_right_left']*frame_idx + 
                                  current_level_sprites['level_boss_right_right']*(1-frame_idx), jnp.uint8)
            enemy_sprite_mask, render_offset = jax.lax.cond(
                state.level_boss == i,
                lambda: (jax.lax.cond(
                    jnp.isin(state.enemy_states[i][2], jnp.array(range(1, 5))),
                    lambda: boss_mask_A,
                    lambda: boss_mask_B
                ), (0, 0)),
                lambda: (enemy_sprite_mask, render_offset)
            )
            enemy_sprite_mask = jax.lax.cond(
                state.enemy_death_timers[i] > 0,
                lambda: current_level_sprites['enemy_death'],
                lambda: enemy_sprite_mask
            )
            return enemy_sprite_mask, render_offset
        
        level_to_use_sprites_from = state.level - jnp.logical_and(state.next_level_transition,
                                                                  state.respawn_timer >= self.consts.TRANSITION_DELAY_SECOND_STAGE).astype(jnp.int32)
        def render_enemy(i, r):
            # Select frame logic
            frame_mask, render_offset_xy = jax.lax.cond(
                jnp.logical_or(level_to_use_sprites_from == 3, level_to_use_sprites_from == 5),
                lambda: get_heli_frame(i, level_sprites, level_offsets),
                lambda: get_plane_frame(i, level_sprites, level_offsets)
            )
            # All enemy sprites share the same padding/offset
            flip_offset = level_offsets['enemy_pos']
            # Height indicator
            hi_mask = general_sprites['black_line']
            hi_offset = general_offsets['black_line']
            hi_interval = (self.consts.MAX_ENTITY_Y - self.consts.MIN_ENTITY_Y)/9
            hi_rel_height = jnp.array((state.enemy_states[i][1] - self.consts.MIN_ENTITY_Y)/hi_interval, jnp.int32)
            hi_height = self.consts.MIN_ENTITY_Y + hi_rel_height*hi_interval
            # Render enemy
            ret_raster = self.jr.render_at_clipped(
                r, state.enemy_states[i][0] + render_offset_xy[0],
                state.enemy_states[i][1] + render_offset_xy[1],
                frame_mask, flip_offset=flip_offset
            )
            # Render height indicator
            ret_raster = self.jr.render_at_clipped(ret_raster, 0, hi_height, hi_mask, flip_offset=hi_offset)
            
            return jax.lax.cond(
                jnp.logical_or(state.enemy_death_timers[i] > 0, state.enemy_states[i][3] > 0),
                lambda: ret_raster,
                lambda: r
            )
        
        raster = jax.lax.fori_loop(0, self.consts.MAX_NUMBER_OF_ENEMIES, render_enemy, raster)

        # --- 6. Render Enemy Missiles ---
        missile_mask = level_sprites['enemy_missile']
        missile_offset = level_offsets['enemy_missile']
        def render_enemy_missile(i, r):
            return jax.lax.cond(
                state.enemy_missile_states[i][3] > 0,
                lambda: self.jr.render_at_clipped(
                    r, state.enemy_missile_states[i][0], state.enemy_missile_states[i][1],
                    missile_mask, flip_offset=missile_offset
                ),
                lambda: r
            )
        raster = jax.lax.fori_loop(0, self.consts.MAX_ENEMY_MISSILES, render_enemy_missile, raster)

        # --- 7. Render Clouds ---
        cloud_mask = level_sprites['cloud']
        cloud_offset = level_offsets['cloud']
        def render_cloud(i, r):
            return self.jr.render_at_clipped(
                r, state.cloud_positions[i][0], state.cloud_positions[i][1],
                cloud_mask, flip_horizontal=True, flip_offset=cloud_offset
            )
        raster = jax.lax.fori_loop(0, 4, render_cloud, raster)

        # --- 8. Render UI (Top/Bottom Walls) ---
        # Draw black/white bars using draw_rects
        raster = self.jr.draw_rects(raster, jnp.array([[0, 0]]), jnp.array([[160, self.consts.BLACK_BORDER_TOP_HEIGHT]]), self.BLACK_ID)
        raster = self.jr.draw_rects(raster, jnp.array([[0, self.consts.MAX_ENTITY_Y + 1]]), jnp.array([[160, 1]]), self.WHITE_ID)
        raster = self.jr.draw_rects(raster, jnp.array([[0, self.consts.MAX_ENTITY_Y + self.consts.WALL_BOTTOM_HEIGHT]]), jnp.array([[160, 1]]), self.WHITE_ID)
        raster = self.jr.draw_rects(raster, jnp.array([[0, self.consts.MAX_ENTITY_Y + 1 + self.consts.WALL_BOTTOM_HEIGHT]]), jnp.array([[160, 210 - (self.consts.MAX_ENTITY_Y + 1 + self.consts.WALL_BOTTOM_HEIGHT)]]), self.BLACK_ID)

        # Draw wall sprites
        raster = self.jr.render_at(
            raster, 0, self.consts.BLACK_BORDER_TOP_HEIGHT,
            general_sprites['top_wall'], flip_offset=general_offsets['top_wall']
        )
        raster = self.jr.render_at(
            raster, 0, self.consts.MAX_ENTITY_Y + 2,
            general_sprites['bottom_wall'], flip_offset=general_offsets['bottom_wall']
        )
        
        # --- 9. Render UI (Score, Lives) ---
        # Score
        digit_masks = general_sprites['digits']
        digits = self.jr.int_to_digits(state.score, max_digits = 6)
        raster = self.jr.render_label_selective(raster, 57, 7, digits, digit_masks, 0, 6, spacing = 8, max_digits_to_render=6)
        
        # Lives 
        raster = self.jr.render_indicator(
            raster, 88, 18, state.lives - 1, 
            general_sprites['player_life'], 
            spacing=-8, max_value=5
        )

        # --- 10. Handle Respawn/Transition Overlays ---
        # Respawn walls
        respawn_top_wall_mask = level_sprites['respawn_top_wall']
        respawn_top_wall_offset = level_offsets['respawn_top_wall']
        respawn_bottom_wall_mask = general_sprites['respawn_bottom_wall']
        respawn_bottom_wall_offset = general_offsets['respawn_bottom_wall']

        def render_respawn_walls(r):
            r_top = self.jr.render_at(r, 0, self.consts.BLACK_BORDER_TOP_HEIGHT, respawn_top_wall_mask, flip_offset=respawn_top_wall_offset)
            r_bottom = self.jr.render_at(r_top, 0, self.consts.MAX_ENTITY_Y + 2, respawn_bottom_wall_mask, flip_offset=respawn_bottom_wall_offset)
            return r_bottom

        raster = jax.lax.cond(
            jnp.logical_and(state.respawn_timer > 0, 
                            state.respawn_timer <= self.consts.TRANSITION_DELAY_SECOND_STAGE + self.consts.TRANSITION_DELAY_FIRST_STAGE),
            render_respawn_walls,
            lambda r: r,
            raster
        )

        # Enemies remaining line
        enemy_remaining_mask = jax.lax.cond(
            jnp.logical_and(state.respawn_timer > 0, 
                            state.respawn_timer <= self.consts.TRANSITION_DELAY_SECOND_STAGE + self.consts.TRANSITION_DELAY_FIRST_STAGE),
            lambda: level_sprites['enemy_remaining_brown'],
            lambda: level_sprites['enemy_remaining']
        )
        enemy_remaining_offset = level_offsets['enemy_remaining'] # Offset is the same

        # Do not render enemies remaining during first stage of level transition
        render_enemies_remaining = jnp.logical_not(
            jnp.logical_and(state.next_level_transition > 0, 
                            state.respawn_timer >= self.consts.TRANSITION_DELAY_SECOND_STAGE))
        
        raster = jax.lax.cond(
            jnp.logical_and(render_enemies_remaining, state.enemies_remaining > self.consts.ENEMY_KILLS_TO_ELIMINATE),
            lambda r: self.jr.render_at(r, 72, 180, enemy_remaining_mask, flip_offset=enemy_remaining_offset),
            lambda r: r,
            raster
        )
        raster = jax.lax.cond(
            jnp.logical_and(render_enemies_remaining, state.enemies_remaining > 0),
            lambda r: self.jr.render_at(r, 80, 181, enemy_remaining_mask, flip_offset=enemy_remaining_offset),
            lambda r: r,
            raster
        )

        # Next level transition animation
        def render_level_transition(r):
            # Redraw background
            r_bg = self.jr.render_at(r, 0, self.consts.MIN_ENTITY_Y, level_sprites['background'], flip_offset=level_offsets['background'])
            
            # Flash bar
            bar_mask = jax.lax.cond(
                state.respawn_timer % 8 > 3,
                lambda: general_sprites['transition_bar'][0],
                lambda: general_sprites['transition_bar'][1]
            )
            r_bar = self.jr.render_at(r_bg, 0, 103, bar_mask, flip_offset=general_offsets['transition_bar'])
            
            # Redraw (white) player on top
            r_player = self.jr.render_at(r_bar, self.consts.PLAYER_X, self.consts.PLAYER_Y, player_sprite_mask, flip_offset=player_offset)
            return r_player

        raster = jax.lax.cond(
            jnp.logical_and(state.next_level_transition, state.respawn_timer <= self.consts.TRANSITION_DELAY),
            render_level_transition,
            lambda r: r,
            raster,
        )

        # --- 11. Render Start Screen ---
        raster = jax.lax.cond(
            state.step_counter < self.consts.START_SCREEN_DELAY,
            lambda r: self.jr.render_at(
                r, 0, 0,
                general_sprites['start_screen'],
                flip_offset=general_offsets['start_screen']
            ),
            lambda r: r,
            raster
        )

        # --- 12. Final Palette Lookup ---
        return self.jr.render_from_palette(raster, self.PALETTE)