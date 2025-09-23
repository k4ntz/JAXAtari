"""
authors: Paula Troszt, Ernst Christian Böhringer, Aiman Sammy Rahlf
"""

import os
from functools import partial
from typing import NamedTuple, Tuple, Dict, Any, Optional
import jax.lax
import jax.numpy as jnp
import chex
import jaxatari.spaces as spaces

from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as jr
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.games.timepilot_levels import (
    LevelConstants,
    TimePilot_Level_1,
    TimePilot_Level_2,
    TimePilot_Level_3,
    TimePilot_Level_4,
    TimePilot_Level_5
)

class TimePilotConstants(NamedTuple):
    # Constants for game environment
    WIDTH: int = 160
    HEIGHT: int = 210

    # Object sizes (width, height)
    PLAYER_SIZE_PER_ROTATION: chex.Array = jnp.array([
        (7, 14), # up
        (8, 10),
        (8, 9), # left
        (8, 10),
        (7, 14), # down
        (8, 10),
        (8, 9), # right
        (8, 10)
    ])
    MISSILE_SIZE: Tuple[int, int] = (1, 2)

    # Rendering constants
    WALL_TOP_HEIGHT: int = 32
    WALL_BOTTOM_HEIGHT: int = 16
    BLACK_BORDER_TOP_HEIGHT: int = 0
    BLACK_BORDER_BOTTOM_HEIGHT: int = 17

    # Player constants
    PLAYER_X: int = 76
    PLAYER_Y: int = WALL_TOP_HEIGHT + BLACK_BORDER_TOP_HEIGHT + 68
    INITIAL_PLAYER_ROTATION: int = 2
    PLAYER_SPEED_PER_ROTATION: chex.Array = jnp.array([
        (0, -4), # up
        (-4, -4),
        (-4, 0), # left
        (-4, 4),
        (0, 4), # right
        (4, 4),
        (4, 0), # down
        (4, -4)
    ])
    # speed is different in every step (the pattern is recurring every eighth step)
    PLAYER_MISSILE_SPEED_PER_ROTATION: chex.Array = jnp.array([
        ((0, -4), (0, 4), (0, -8), (0, 4), (0, -4), (0, 4), (0, -4), (0, 0)), # up
        ((-4, -4), (4, 4), (-8, -8), (4, 4), (-4, -4), (4, 4), (-4, -4), (0, 0)),
        ((-4, 0), (4, 0), (-8, 0), (4, 0), (-4, 0), (4, 0), (-4, 0), (0, 0)), # left
        ((-4, 4), (4, -4), (-8, 8), (4, -4), (-4, 4), (4, -4), (-4, 4), (0, 0)),
        ((0, 4), (0, -4), (0, 8), (0, -4), (0, 4), (0, -4), (0, 4), (0, 0)), # down
        ((4, 4), (-4, -4), (8, 8), (-4, -4), (4, 4), (-4, -4), (4, 4), (0, 0)),
        ((4, 0), (-4, 0), (8, 0), (-4, 0), (4, 0), (-4, 0), (4, 0), (0, 0)), # right
        ((4, -4), (-4, 4), (8, -8), (-4, 4), (4, -4), (-4, 4), (4, -4), (0, 0))
    ])

    # Cloud constants
    INITIAL_CLOUDS: chex.Array = jnp.array([ 
        (16, 65),
        (96, 65),
        (16, 129),
        (96, 129)
    ])

    # Game constants
    INITIAL_ENEMIES_REMAINING: int = 8
    INITIAL_LIVES: int = 5
    MAX_LIVES: int = 5
    MAX_SCORE: int = 999900

    POINTS_PER_ENEMY: int = 100
    POINTS_PER_BOSS: int = 3000
    POINTS_TO_GAIN_A_LIFE: int = 10000
    ENEMY_KILLS_TO_ELIMINATE: int = 4 

    MAX_ENTITY_X: int = WIDTH - 1
    MAX_ENTITY_Y: int = HEIGHT - WALL_BOTTOM_HEIGHT - BLACK_BORDER_BOTTOM_HEIGHT - 1
    MIN_ENTITY_X: int = 0
    MIN_ENTITY_Y: int = WALL_TOP_HEIGHT + BLACK_BORDER_TOP_HEIGHT + 1

    # Animation and respawn delays
    START_SCREEN_DELAY: int = 120
    PLAYER_DEATH_ANIMATION_DELAY: int = 64
    TRANSITION_DELAY_FIRST_STAGE: int = 128
    TRANSITION_DELAY_SECOND_STAGE: int = 128
    TRANSITION_DELAY: int = TRANSITION_DELAY_FIRST_STAGE + TRANSITION_DELAY_SECOND_STAGE
    ENEMY_DEATH_ANIMATION_DELAY = 24

    # Enemy constants
    MAX_NUMBER_OF_ENEMIES: int = 4
    MAX_ENEMY_MISSILES: int = 2
    INITIAL_ATTACK_DELAY: int = 150 # mean delay for first attack after player respawn
    ENEMY_MISSILE_SPEED_PER_ROTATION = jnp.array([
        (0, -4), # up
        (-4, -4),
        (-4, 0), # left
        (-4, 4),
        (0, 4), # down
        (4, 4),
        (4, 0), # right
        (4, -4)
    ])
    
    # Level constants
    LEVEL_1: LevelConstants = TimePilot_Level_1
    LEVEL_2: LevelConstants = TimePilot_Level_2
    LEVEL_3: LevelConstants = TimePilot_Level_3
    LEVEL_4: LevelConstants = TimePilot_Level_4
    LEVEL_5: LevelConstants = TimePilot_Level_5

# immutable state container
class TimePilotState(NamedTuple):

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

class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray
    rotation: jnp.ndarray
    active: jnp.ndarray

class TimePilotObservation(NamedTuple):
    player: EntityPosition # (x, y, width, height, rotation, active)
    player_missile: EntityPosition  # (x, y, width, height, rotation, active)
    enemies: jnp.ndarray # shape (4, 6) - 4 enemies, each with (x, y, width, height, rotation, active)
    level_boss: EntityPosition # (x, y, width, height, rotation, active)
    enemy_missiles: EntityPosition  # shape (2, 6) - 4 enemy missiles, each with (x, y, width, height, rotation, active)

    level: jnp.ndarray
    score: jnp.ndarray
    lives: jnp.ndarray
    enemies_remaining: jnp.ndarray

class TimePilotInfo(NamedTuple):
    level: chex.Array
    score: chex.Array
    lives: chex.Array
    enemies_remaining: chex.Array
    all_rewards: chex.Array

class JaxTimePilot(JaxEnvironment[TimePilotState, TimePilotObservation, TimePilotInfo, TimePilotConstants]):
    def __init__(self, consts: TimePilotConstants|None = None, reward_funcs: list[callable]=None):
        consts = consts or TimePilotConstants()
        super().__init__(consts)
        self.frame_stack_size = 4
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.action_set = jnp.array([
            Action.NOOP,
            Action.FIRE,
            Action.RIGHT,
            Action.LEFT,
            Action.UP,
            Action.DOWN,
            Action.UPFIRE,
            Action.DOWNFIRE,
            Action.RIGHTFIRE,
            Action.LEFTFIRE
        ])
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
            def update_enemy_position(i, enemy_states):
                # move enemy
                new_x = enemy_states[i][0] + level_constants.enemy_speed_per_rotation[enemy_states[i][2]][0] # x direction
                new_y = enemy_states[i][1] + level_constants.enemy_speed_per_rotation[enemy_states[i][2]][1] # y direction
                # move view
                new_x = new_x - self.consts.PLAYER_SPEED_PER_ROTATION[state_player_rotation][0] # x direction
                new_y = new_y - self.consts.PLAYER_SPEED_PER_ROTATION[state_player_rotation][1] # y direction
                
                enemy_size = jax.lax.cond(
                    i == level_boss,
                    lambda: level_constants.level_boss_size,
                    lambda: level_constants.enemy_size_per_rotation[enemy_states[i][2]]
                )

                # only wrap around if enemy has spawned already
                return jax.lax.cond(
                    self.entities_collide(
                        self.consts.MIN_ENTITY_X,
                        self.consts.MIN_ENTITY_Y,
                        self.consts.MAX_ENTITY_X - self.consts.MIN_ENTITY_X,
                        self.consts.MAX_ENTITY_Y - self.consts.MIN_ENTITY_Y,
                        enemy_states[i][0],
                        enemy_states[i][1],
                        enemy_size[0],
                        enemy_size[1]
                    ),
                    lambda: enemy_states.at[i].set(jnp.array([
                        self.final_pos(self.consts.MIN_ENTITY_X, self.consts.MAX_ENTITY_X, new_x),
                        self.final_pos(self.consts.MIN_ENTITY_Y, self.consts.MAX_ENTITY_Y, new_y),
                        enemy_states[i][2], enemy_states[i][3]])),
                    lambda: enemy_states.at[i].set(jnp.array([
                        self.final_pos(self.consts.MIN_ENTITY_X - 50, self.consts.MAX_ENTITY_X + 50, new_x),
                        self.final_pos(self.consts.MIN_ENTITY_Y - 50, self.consts.MAX_ENTITY_Y + 50, new_y),
                        enemy_states[i][2], enemy_states[i][3]])),
                )

            return jax.lax.fori_loop(0, self.consts.MAX_NUMBER_OF_ENEMIES, update_enemy_position, enemy_states)
        
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

            def update_enemy_rotation(i, enemy_states):
                # set enemy_states rotation to new_rotation
                return enemy_states.at[i].set(jnp.array([enemy_states[i][0], enemy_states[i][1], 
                                                         new_rotation, enemy_states[i][3]]))

            return jax.lax.fori_loop(0, self.consts.MAX_NUMBER_OF_ENEMIES, update_enemy_rotation, enemy_states), rng_key
        
        # alternate changing enemy position and rotation according to step counter
        new_enemy_states, rng_key = jax.lax.cond(
            position_change, # rotation change if false
            lambda: (enemy_position_change(state_enemy_states), state_rng_key),
            lambda: enemy_rotation_change(state_enemy_states, state_rng_key)
        )

        # shake single enemy
        def shake_enemy(i, enemy_states):
            
            # shake enemy vertically
            new_y = jax.lax.cond(
                respawn_timer % 16 < 8,
                lambda: enemy_states[i][1] - 4,
                lambda: enemy_states[i][1] + 4
            )

            enemy_size = jax.lax.cond(
                    i == level_boss,
                    lambda: level_constants.level_boss_size,
                    lambda: level_constants.enemy_size_per_rotation[enemy_states[i][2]]
                )

            # only wrap around if enemy has spawned already
            return jax.lax.cond(
                    self.entities_collide(
                        self.consts.MIN_ENTITY_X,
                        self.consts.MIN_ENTITY_Y,
                        self.consts.MAX_ENTITY_X - self.consts.MIN_ENTITY_X,
                        self.consts.MAX_ENTITY_Y - self.consts.MIN_ENTITY_Y,
                        enemy_states[i][0],
                        enemy_states[i][1],
                        enemy_size[0],
                        enemy_size[1]
                    ),
                    lambda: enemy_states.at[i].set(
                        [enemy_states[i][0],
                         self.final_pos(self.consts.MIN_ENTITY_Y - 50, self.consts.MAX_ENTITY_Y + 50, new_y), 
                         enemy_states[i][2],
                         enemy_states[i][3]]
                    ),
                    lambda: enemy_states.at[i].set(
                        [enemy_states[i][0],
                         new_y, 
                         enemy_states[i][2],
                         enemy_states[i][3]]
                    )
                )

        # either just move enemies or add shaking after player death
        return jax.lax.cond(
            jnp.logical_and(jnp.logical_not(position_change),
                            jnp.logical_and(respawn_timer >= self.consts.TRANSITION_DELAY, 
                                            respawn_timer <= self.consts.TRANSITION_DELAY + self.consts.PLAYER_DEATH_ANIMATION_DELAY)),
            lambda: (jax.lax.fori_loop(0, self.consts.MAX_NUMBER_OF_ENEMIES, shake_enemy, new_enemy_states), rng_key),
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
        def update_cloud(i, cloud_positions):

            # move cloud in opposite direction of player
            new_x = cloud_positions[i][0] - self.consts.PLAYER_SPEED_PER_ROTATION[player_rotation][0]
            new_y = cloud_positions[i][1] - self.consts.PLAYER_SPEED_PER_ROTATION[player_rotation][1]

            return cloud_positions.at[i].set(
                        [self.final_pos(self.consts.MIN_ENTITY_X - 32, self.consts.MAX_ENTITY_X, new_x),
                         self.final_pos(self.consts.MIN_ENTITY_Y - 12, self.consts.MAX_ENTITY_Y, new_y)]
                    )
        
        # shake single cloud
        def shake_cloud(i, cloud_positions):

            new_y = jax.lax.cond(
                respawn_timer % 16 < 8,
                lambda: cloud_positions[i][1] - 4,
                lambda: cloud_positions[i][1] + 4
            )

            return cloud_positions.at[i].set(
                        [cloud_positions[i][0],
                         self.final_pos(self.consts.MIN_ENTITY_Y - 12, self.consts.MAX_ENTITY_Y, new_y)]
                    )

        # either move clouds according to player rotation or shake clouds after player death
        return jax.lax.cond(
            jnp.logical_and(respawn_timer >= self.consts.TRANSITION_DELAY, 
                            respawn_timer <= self.consts.TRANSITION_DELAY + self.consts.PLAYER_DEATH_ANIMATION_DELAY),
            lambda: jax.lax.fori_loop(0, 4, shake_cloud, state_cloud_positions),
            lambda: jax.lax.fori_loop(0, 4, update_cloud, state_cloud_positions)
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

        def move_missile(i, states):
            
            # move missile position
            new_state = jnp.array([states[i][0] + self.consts.ENEMY_MISSILE_SPEED_PER_ROTATION[states[i][2]][0],
                                   states[i][1] + self.consts.ENEMY_MISSILE_SPEED_PER_ROTATION[states[i][2]][1],
                                   states[i][2],
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
                states[i][3] > 0,
                lambda: states.at[i].set(new_state),
                lambda: states
                )
        
        missile_states = jax.lax.fori_loop(0, self.consts.MAX_ENEMY_MISSILES, move_missile, missile_states)

        # select enemy for shooting
        rng_key, subkey = jax.random.split(rng_key)
        shooting_enemy_idx = jax.random.randint(subkey, [], 0, self.consts.MAX_NUMBER_OF_ENEMIES)

        # do not shoot if enemy is inactive
        index = jax.lax.fori_loop(
            0,
            self.consts.MAX_ENEMY_MISSILES,
            lambda i, val: jax.lax.cond(jnp.logical_and(missile_states[i][3]<=0, val<0), lambda: i, lambda: val),
            -1
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

        def get_hit_enemy(i, index):

            enemy_size = jax.lax.cond(
                    i == level_boss,
                    lambda: level_constants.level_boss_size,
                    lambda: level_constants.enemy_size_per_rotation[enemy_states[i][2]]
                )
            
            # return i if collision of player with enemy at index i occurred, old index otherwise
            return jax.lax.cond(
                jnp.logical_and(enemy_states[i][3] > 0,
                                self.entities_collide(self.consts.PLAYER_X, self.consts.PLAYER_Y,
                                    self.consts.PLAYER_SIZE_PER_ROTATION[player_rotation][0], 
                                    self.consts.PLAYER_SIZE_PER_ROTATION[player_rotation][1],
                                    enemy_states[i][0], enemy_states[i][1],
                                    enemy_size[0], 
                                    enemy_size[1])),
                lambda: i,
                lambda: index
            )

        # only return collision if player is active
        return jax.lax.cond(
            player_active,
            lambda: jax.lax.fori_loop(0, self.consts.MAX_NUMBER_OF_ENEMIES, get_hit_enemy, -1),
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

        @jax.jit
        def get_hit_enemy(i, index):

            enemy_size = jax.lax.cond(
                    i == level_boss,
                    lambda: level_constants.level_boss_size,
                    lambda: level_constants.enemy_size_per_rotation[enemy_states[i][2]]
                )
            
            # return i if collision of player missile with enemy at index i occurred, old index otherwise
            return jax.lax.cond(
                self.is_hit(enemy_states[i][0], enemy_states[i][1], enemy_states[i][3] > 0,
                            enemy_size, player_missile_state),
                lambda: i,
                lambda: index)

        return jax.lax.fori_loop(0, self.consts.MAX_NUMBER_OF_ENEMIES, get_hit_enemy, -1)

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

        def get_hitting_missile(i, index):
            # return i if collision of player with enemy missile at index i occurred, old index otherwise
            return jax.lax.cond(
                self.is_hit(self.consts.PLAYER_X, self.consts.PLAYER_Y, player_active, 
                            self.consts.PLAYER_SIZE_PER_ROTATION[player_rotation], enemy_missile_states[i]),
                lambda: i,
                lambda: index)

        return jax.lax.fori_loop(0, self.consts.MAX_ENEMY_MISSILES, get_hitting_missile, -1)
    
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
            lambda: self.player_step(state.player_rotation, action),
            lambda: state.player_rotation
        )

        # update player missile (every step)
        player_missile_state, missile_rdy = self.player_missile_step(
            state.player_missile_state, state.missile_rdy, player_rotation, action, 
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
        def check_ee_collision(i, j, val, current_enemy_states, level_boss):

            enemy_size_i = jax.lax.cond(
                    i == level_boss,
                    lambda: level_constants.level_boss_size,
                    lambda: level_constants.enemy_size_per_rotation[enemy_states[i][2]]
                )
            
            enemy_size_j = jax.lax.cond(
                    j == level_boss,
                    lambda: level_constants.level_boss_size,
                    lambda: level_constants.enemy_size_per_rotation[enemy_states[j][2]]
                )
            
            return jax.lax.cond(
                jnp.logical_and(self.entities_collide(
                    current_enemy_states[i][0],
                    current_enemy_states[i][1],
                    enemy_size_i[0]*2,
                    enemy_size_i[1]*2,
                    current_enemy_states[j][0],
                    current_enemy_states[j][1],
                    enemy_size_j[0]*2,
                    enemy_size_j[1]*2
                ), i != j),
                lambda: j,
                lambda: val
            )

        # respawn enemy when its death timer is at zero
        def respawn_enemy(i, enemy_states):

            # get respawn states according to level constants as well as index of overlapping, already existing enemy (-1 if ther is no overlap)
            new_enemy_states, coll_idx = jax.lax.cond(
                state.enemy_death_timers[i] == 1,
                lambda: (enemy_states.at[i].set(level_constants.initial_enemies[i]), 
                         jax.lax.fori_loop(0, self.consts.MAX_NUMBER_OF_ENEMIES, 
                                           lambda j, val: check_ee_collision(i, j, val, enemy_states, state.level_boss), -1)),
                lambda: (enemy_states, -1)
            )

            # get direction to move overlapping, new enemy
            mov_dir = jax.lax.cond(
                coll_idx >= 0,
                lambda: (jnp.array(
                    jnp.logical_or(new_enemy_states[i][1]<self.consts.MIN_ENTITY_Y,
                                   new_enemy_states[i][1]>self.consts.MAX_ENTITY_Y), 
                                   jnp.int32)*jnp.where(new_enemy_states[i][0]<new_enemy_states[coll_idx][0], -1, 1), 
                                   jnp.array(jnp.logical_or(new_enemy_states[i][0]<self.consts.MIN_ENTITY_X,
                                                            new_enemy_states[i][0]>self.consts.MAX_ENTITY_X), 
                                                            jnp.int32)*jnp.where(new_enemy_states[i][1]<new_enemy_states[coll_idx][1],
                                                                                  -1, 1)),
                lambda: (0, 0)
            )

            enemy_size = jax.lax.cond(
                    i == state.level_boss,
                    lambda: level_constants.level_boss_size,
                    lambda: level_constants.enemy_size_per_rotation[enemy_states[i][2]]
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
            new_enemy_states = jax.lax.fori_loop(
                0,
                self.consts.MAX_NUMBER_OF_ENEMIES,
                lambda _, states: jax.lax.cond(
                    (jax.lax.fori_loop(0, self.consts.MAX_NUMBER_OF_ENEMIES, 
                                       lambda j, val: check_ee_collision(i, j, val, states, state.level_boss), 
                                       -1) >= 0),
                    lambda: states.at[i].set(jnp.array([new_enemy_states[i][0] + x_offset, new_enemy_states[i][1] + y_offset, 
                                                        new_enemy_states[i][2], new_enemy_states[i][3]])),
                    lambda: states
                ),
                new_enemy_states
            )

            return new_enemy_states

        enemy_states = jax.lax.fori_loop(0, self.consts.MAX_NUMBER_OF_ENEMIES, respawn_enemy, enemy_states)

        # update death timer for colliding enemies
        def update_enemy_death_timer(i, enemy_death_timers):
            return jax.lax.cond(
                enemy_death_timers[i] > 0,
                lambda: enemy_death_timers.at[i].set(enemy_death_timers[i] - 1),
                lambda: enemy_death_timers
            )
        enemy_death_timers = jax.lax.fori_loop(0, self.consts.MAX_NUMBER_OF_ENEMIES, update_enemy_death_timer, state.enemy_death_timers)

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
        all_rewards = self._get_all_rewards(state, new_state)
        info = self._get_info(new_state, all_rewards)
        observation = self._get_observation(new_state)

        return observation, new_state, env_reward, done, info

    def render(self, state: TimePilotState) -> jnp.ndarray:
        return self.renderer.render(state)

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: TimePilotState):

        # player
        player = EntityPosition(
            x=jnp.array(self.consts.PLAYER_X),
            y=jnp.array(self.consts.PLAYER_Y),
            width=jnp.array(self.consts.PLAYER_SIZE_PER_ROTATION[state.player_rotation][0]),
            height=jnp.array(self.consts.PLAYER_SIZE_PER_ROTATION[state.player_rotation][1]),
            rotation=state.player_rotation,
            active=jnp.logical_and(state.player_active, state.respawn_timer <= 0).astype(jnp.int32)
        )

        # player missile
        player_missile = EntityPosition(
            x=state.player_missile_state[0],
            y=state.player_missile_state[1],
            width=jnp.array(self.consts.MISSILE_SIZE[0]),
            height=jnp.array(self.consts.MISSILE_SIZE[1]),
            rotation=state.player_missile_state[2],
            active=state.player_missile_state[3]
        )

        # enemies
        def convert_enemy_state_to_entity(enemy_state, enemy_death_timer, idx):
            return jnp.array([
                enemy_state[0], # x position
                enemy_state[1], # y position
                self._get_level_constants(state.level).enemy_size_per_rotation[enemy_state[2]][0], # width
                self._get_level_constants(state.level).enemy_size_per_rotation[enemy_state[2]][1], # height
                enemy_state[2], # rotation
                jnp.logical_and(jnp.logical_and(enemy_death_timer <= 0, enemy_state[3]), 
                                jnp.logical_not(idx == state.level_boss)).astype(jnp.int32) # active flag
            ])

        enemies = jax.vmap(convert_enemy_state_to_entity)(
            state.enemy_states, state.enemy_death_timers, jnp.array(range(self.consts.MAX_NUMBER_OF_ENEMIES))
        )

        # level boss
        level_boss = jax.lax.cond(
            state.level_boss > -1,
            lambda: EntityPosition(
                x=state.enemy_states[state.level_boss][0],
                y=state.enemy_states[state.level_boss][1],
                width=jnp.array(self._get_level_constants(state.level).level_boss_size[0]),
                height=jnp.array(self._get_level_constants(state.level).level_boss_size[1]),
                rotation=state.enemy_states[state.level_boss][4],
                active=jnp.logical_and(state.enemy_death_timers[state.level_boss] <= 0, 
                                       state.enemy_states[state.level_boss][3]).astype(jnp.int32)
            ),
            lambda: EntityPosition(
                x=0,
                y=0,
                width=jnp.array(self._get_level_constants(state.level).level_boss_size[0]),
                height=jnp.array(self._get_level_constants(state.level).level_boss_size[1]),
                rotation=0,
                active=0
            )
        )

        # enemy missiles
        def convert_enemy_missile_state_to_entity(enemy_missile_state):
            return jnp.array([
                enemy_missile_state[0], # x position
                enemy_missile_state[1], # y position
                jnp.array(self.consts.MISSILE_SIZE[0]),
                jnp.array(self.consts.MISSILE_SIZE[1]),
                enemy_missile_state[2], # rotation
                enemy_missile_state[3] # active flag
            ])

        enemy_missiles = jax.vmap(convert_enemy_missile_state_to_entity)(
            state.enemy_missile_states
        )

        return TimePilotObservation(
            player=player,
            player_missile=player_missile,
            enemies=enemies,
            level_boss=level_boss,
            enemy_missiles=enemy_missiles,
            level=state.level,
            score=state.score,
            lives=state.lives,
            enemies_remaining=state.enemies_remaining
        )

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: TimePilotObservation) -> jnp.ndarray:
        """Converts the observation to a flat array."""

        def entity_pos_to_flat_array(obj: EntityPosition):
            return jnp.concatenate([
                jnp.atleast_1d(obj.x),
                jnp.atleast_1d(obj.y),
                jnp.atleast_1d(obj.width),
                jnp.atleast_1d(obj.height),
                jnp.atleast_1d(obj.rotation),
                jnp.atleast_1d(obj.active)])

        return jnp.concatenate([
            entity_pos_to_flat_array(obs.player),
            entity_pos_to_flat_array(obs.player_missile),
            obs.enemies.flatten(),
            entity_pos_to_flat_array(obs.level_boss),
            obs.enemy_missiles.flatten(),

            obs.level.flatten(),
            obs.score.flatten(),
            obs.lives.flatten(),
            obs.enemies_remaining.flatten()
        ])

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))

    def observation_space(self) -> spaces.Box: 
        """Returns the observation space for TimePilot.
        The observation contains:
        - player: EntityPosition (x, y, width, height, rotation, active)
        - player_missile: EntityPosition (x, y, width, height, rotation, active)
        - enemies: array of shape (4, 6) with (x, y, width, height, rotation, active)
        - level_boss: EntityPosition (x, y, width, height, rotation, active)
        - enemy_missiles: array of shape (2, 6) with (x, y, width, height, rotation, active)
        - level: int (1-5)
        - score: int (0-999900)
        - lives: int (0-5)
        - enemies_remaining: int (0-8)
        """
        return spaces.Dict({
            "player": spaces.Dict({
                "x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                "rotation": spaces.Box(low=0, high=8, shape=(), dtype=jnp.int32),
                "active": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
            }),
            "player_missile": spaces.Dict({
                "x": spaces.Box(low=-8, high=168, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=25, high=188, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                "rotation": spaces.Box(low=0, high=8, shape=(), dtype=jnp.int32),
                "active": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
            }),
            "enemies": spaces.Box(low=-50, high=226, shape=(self.consts.MAX_NUMBER_OF_ENEMIES, 6), dtype=jnp.int32),
            "level_boss": spaces.Dict({
                "x": spaces.Box(low=-50, high=210, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=-17, high=226, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                "rotation": spaces.Box(low=0, high=8, shape=(), dtype=jnp.int32),
                "active": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
            }),
            "enemy_missiles": spaces.Box(low=-4, high=180, shape=(self.consts.MAX_ENEMY_MISSILES, 6), dtype=jnp.int32),
            "level": spaces.Box(low=1, high=5, shape=(), dtype=jnp.int32),
            "score": spaces.Box(low=0, high=self.consts.MAX_SCORE, shape=(), dtype=jnp.int32),
            "lives": spaces.Box(low=0, high=self.consts.MAX_LIVES, shape=(), dtype=jnp.int32),
            "enemies_remaining": spaces.Box(low=0, high=self.consts.INITIAL_ENEMIES_REMAINING, shape=(), dtype=jnp.int32)
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
    def _get_info(self, state: TimePilotState, all_rewards: chex.Array = None) -> TimePilotInfo:
        return TimePilotInfo(level=state.level, score=state.score, lives= state.lives, 
                             enemies_remaining=state.enemies_remaining, all_rewards=all_rewards)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: TimePilotState, state: TimePilotState):
        return state.score - previous_state.score

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_rewards(self, previous_state: TimePilotState, state: TimePilotState):
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array(
            [reward_func(previous_state, state) for reward_func in self.reward_funcs]
        )
        return rewards

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: TimePilotState) -> bool:
        return state.lives <= 0

class TimePilotRenderer(JAXGameRenderer):
    """JAX-based TimePilot game renderer, optimized with JIT compilation."""

    def __init__(self, consts: TimePilotConstants|None = None):
        """
        Initializes the renderer by loading sprites.
        """
        super().__init__()
        self.consts = consts or TimePilotConstants()
        self.sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/timepilot"
        self.sprites = self._load_sprites()

    def _load_sprites(self) -> list[dict[str, Any]]:
        """Loads all necessary sprites from .npy files."""

        # helper function to load a single sprite frame
        def _load_sprite_frame(name: str) -> Optional[chex.Array]:
            path = os.path.join(self.sprite_path, f'{name}.npy')
            frame = jr.loadFrame(path)
            if isinstance(frame, jnp.ndarray) and frame.ndim >= 2:
                return frame.astype(jnp.uint8)

        # level independent sprites
        general_sprites: Dict[str, Any] = {}

        # decoration
        general_sprites['top_wall'] = _load_sprite_frame('L-All/Top')
        general_sprites['bottom_wall'] = _load_sprite_frame('L-All/Bottom')
        general_sprites['respawn_bottom_wall'] = _load_sprite_frame('L-All/Respawn_Bottom')
        general_sprites['start_screen'] = _load_sprite_frame('L-All/First')
        general_sprites['transition_bar'], _ = jr.pad_to_match([_load_sprite_frame('L-All/TeleportBar'),
                                                             _load_sprite_frame('L-All/TeleportBar2')])

        # digits for score
        digit_path = os.path.join(self.sprite_path, 'L-All/Digit{}.npy')
        digit_sprites = jr.load_and_pad_digits(digit_path , num_chars=10)
        general_sprites['digits'] = digit_sprites

        # lives
        general_sprites['player_life'] = _load_sprite_frame('L-All/Player_Life')

        # enemy height indicator
        general_sprites['black_line'] = _load_sprite_frame('L-All/BlackLine')

        # level dependent sprites
        sprites = []

        cloud_sprites = []
        background_sprites = []
        respawn_top_wall_sprites = []
        player_sprites = []
        enemy_sprites = []
        player_missile_sprites = []
        enemy_missile_sprites = []
        enemy_remaining_sprites = []

        # append all level dependent sprites to their respective list
        for level in ["L1", "L2", "L3", "L4", "L5"]:

            # cloud
            cloud_sprites.append(_load_sprite_frame(f'{level}/{level}_Cloud'))

            # background and decoration
            background_sprites.append(_load_sprite_frame(f'{level}/{level}_Background'))
            respawn_top_wall_sprites.append(_load_sprite_frame(f'{level}/{level}_Top'))

            # living player
            for i in range(8):
                player_sprites.append(_load_sprite_frame(f'{level}/{level}_Player_Pos{i}'))
            # player death animation
            for i in range(2):
                player_sprites.append(_load_sprite_frame(f'{level}/{level}_Player_Death{i+1}'))

            # enemy
            if level in ["L1", "L2", "L4"]:
                for i in range(8):
                    enemy_sprites.append(_load_sprite_frame(f'{level}/{level}_Enemy_Pos{i}'))
                for i in range(2): # some sprites are included multiple times to ensure equal array size for all levels
                    enemy_sprites.append(_load_sprite_frame(f'{level}/{level}_Enemy_Pos{i}'))
            if level == "L3":
                for i in ["01", "02", "11", "12", "21", "22", "31", "32", "41", "42"]:
                    enemy_sprites.append(_load_sprite_frame(f'{level}/{level}_Enemy_Pos{i}'))
            if level == "L5":
                for i in range(5): # some sprites are included multiple times to ensure equal array size for all levels
                    for j in range(2):
                        enemy_sprites.append(_load_sprite_frame(f'{level}/{level}_Enemy_Pos{j}'))

            # enemy death animation
            enemy_sprites.append(_load_sprite_frame(f'{level}/{level}_Enemy_Death'))

            # level boss
            if level in ["L1", "L2", "L4", "L5"]:
                enemy_sprites.append(_load_sprite_frame(f'{level}/{level}_Boss_Pos0')) # included twice to ensure equal array size
                enemy_sprites.append(_load_sprite_frame(f'{level}/{level}_Boss_Pos0'))
                enemy_sprites.append(_load_sprite_frame(f'{level}/{level}_Boss_Pos1')) # included twice to ensure equal array size
                enemy_sprites.append(_load_sprite_frame(f'{level}/{level}_Boss_Pos1'))
            if level == "L3":
                for i in ["01", "02", "11", "12"]:
                    enemy_sprites.append(_load_sprite_frame(f'{level}/{level}_Boss_Pos{i}'))

            # missiles
            player_missile_sprites.append(_load_sprite_frame(f'{level}/{level}_Player_Bullet'))
            enemy_missile_sprites.append(_load_sprite_frame(f'{level}/{level}_Enemy_Bullet'))

            # enemies remaining indicators
            enemy_remaining_sprites.append(_load_sprite_frame(f'{level}/{level}_Enemy_Life'))
            enemy_remaining_sprites.append(_load_sprite_frame(f'{level}/{level}_Enemy_Death_Life'))

        # add player pos and death sprites for level transition
        for i in range(8):
            player_sprites.append(_load_sprite_frame(f'L-All/TP_Player_Pos{i}'))
        player_sprites.append(_load_sprite_frame('L-All/TP_Player_Death'))
                                                          
        # pad all sprites since they have to be used interchangeably
        # (and jax enforces same sizes)
        cloud_sprites, _ = jr.pad_to_match(cloud_sprites)
        background_sprites, _ = jr.pad_to_match(background_sprites)
        respawn_top_wall_sprites, _ = jr.pad_to_match(respawn_top_wall_sprites)
        player_sprites, _ = jr.pad_to_match(player_sprites)
        enemy_sprites, _ = jr.pad_to_match(enemy_sprites)
        player_missile_sprites, _ = jr.pad_to_match(player_missile_sprites)
        enemy_missile_sprites, _ = jr.pad_to_match(enemy_missile_sprites)
        enemy_remaining_sprites, _ = jr.pad_to_match(enemy_remaining_sprites)

        # add sprites from padded lists to their respective level sprites dict
        for i in range(0, 5):

            level_sprites: Dict[str, Any] = {}

            # cloud
            level_sprites['cloud'] = cloud_sprites[i]

            # decoration
            level_sprites['background'] = background_sprites[i]
            level_sprites['respawn_top_wall'] = respawn_top_wall_sprites[i]

            # player
            level_sprites['player_pos'] = player_sprites[i*10:i*10+8]
            level_sprites['player_death'] = player_sprites[i*10+8:(i+1)*10]

            # enemy
            level_sprites['enemy_pos'] = enemy_sprites[i*15:i*15+10]
            level_sprites['enemy_death'] = enemy_sprites[i*15+10]

            # level boss
            (level_sprites['level_boss_left_right'], level_sprites['level_boss_left_left'], 
             level_sprites['level_boss_right_left'], level_sprites['level_boss_right_right']) = enemy_sprites[i*15+11:i*15+15]

            # missiles  
            level_sprites['player_missile'] = player_missile_sprites[i]
            level_sprites['enemy_missile'] = enemy_missile_sprites[i]

            # enemies remaining indicator
            level_sprites['enemy_remaining'] = enemy_remaining_sprites[i*2]
            level_sprites['enemy_remaining_brown'] = enemy_remaining_sprites[i*2+1]

            sprites.append(level_sprites)

        # player sprites for level transition
        general_sprites['transition_player_pos'] = player_sprites[50:58]
        general_sprites['transition_player_death'] = player_sprites[58]

        sprites.insert(0, general_sprites)

        # expand all sprites
        for sprite_dict in sprites:
            for key, value in sprite_dict.items():
                if isinstance(value, (list, tuple)):
                    sprite_dict[key] = jnp.array([jnp.expand_dims(sprite, axis=0) for sprite in value])
                else:
                    sprite_dict[key] = jnp.expand_dims(value, axis=0)

        return sprites

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: TimePilotState) -> chex.Array:
        """
        Renders the current game state using JAX operations.

        Args:
            state: A TimePilotState object containing the current game state.

        Returns:
            A JAX array representing the rendered frame.
        """
        # empty raster
        raster = jr.create_initial_frame(width=self.consts.WIDTH, height=self.consts.HEIGHT)

        # get sprites according to current level (use sprites for preceding level during first half of level transition animation)
        general_sprites = self.sprites[0]
        level_sprites = jax.lax.cond(
            jnp.logical_and(state.next_level_transition, state.respawn_timer >= self.consts.TRANSITION_DELAY_SECOND_STAGE),
            lambda: jax.lax.switch(
                state.level - 1,
                [
                    lambda: self.sprites[5],
                    lambda: self.sprites[1],
                    lambda: self.sprites[2],
                    lambda: self.sprites[3],
                    lambda: self.sprites[4]
                ]
            ),
            lambda: jax.lax.switch(
                state.level - 1,
                [
                    lambda: self.sprites[1],
                    lambda: self.sprites[2],
                    lambda: self.sprites[3],
                    lambda: self.sprites[4],
                    lambda: self.sprites[5]
                ]
            )
        )

        # background
        frame_bg = jr.get_sprite_frame(level_sprites['background'], 0)
        raster = jr.render_at(raster, 0, self.consts.MIN_ENTITY_Y, frame_bg)

        # set player sprite
        player_sprite = level_sprites['player_pos'][state.player_rotation]
        # use death animation instead if player has just died
        player_sprite = jax.lax.cond(
            jnp.logical_and(
                state.player_active <= 0,
                jnp.logical_or(
                jnp.isin(state.respawn_timer, jnp.array(range(317, 320))), 
                jnp.logical_or(
                    jnp.logical_or(
                        jnp.isin(state.respawn_timer, jnp.array(range(301, 309))), 
                        jnp.isin(state.respawn_timer, jnp.array(range(285, 293)))), 
                    jnp.logical_or(
                        jnp.isin(state.respawn_timer, jnp.array(range(269, 277))), 
                        jnp.isin(state.respawn_timer, jnp.array(range(256, 261))))))),
            lambda: level_sprites['player_death'][0],
            lambda: player_sprite
        )
        player_sprite = jax.lax.cond(
            jnp.logical_and(
                state.player_active <= 0,
                jnp.logical_or(
                    jnp.logical_or(
                        jnp.isin(state.respawn_timer, jnp.array(range(309, 317))), 
                        jnp.isin(state.respawn_timer, jnp.array(range(293, 301)))), 
                    jnp.logical_or(
                        jnp.isin(state.respawn_timer, jnp.array(range(277, 285))), 
                        jnp.isin(state.respawn_timer, jnp.array(range(261, 269)))))),
            lambda: level_sprites['player_death'][1],
            lambda: player_sprite
        )
        # turn player white for transition of next level
        player_sprite = jax.lax.cond(
            jnp.logical_and(state.next_level_transition > 0, state.respawn_timer <= self.consts.TRANSITION_DELAY),
            lambda: general_sprites['transition_player_pos'][state.player_rotation],
            lambda: player_sprite
        )
        # use white death animation instead if player has just died
        player_sprite = jax.lax.cond(
            jnp.logical_and(
                state.player_active <= 0,
                jnp.logical_and(state.next_level_transition > 0, 
                                state.respawn_timer <= self.consts.TRANSITION_DELAY)
            ),
            lambda: general_sprites['transition_player_death'],
            lambda: player_sprite
        )

        frame_player = jr.get_sprite_frame(player_sprite, state.step_counter)
        # do not render player during first stage of player death animation
        raster = jax.lax.cond(
            jnp.logical_or(state.respawn_timer < self.consts.TRANSITION_DELAY_SECOND_STAGE,
                           state.respawn_timer >= self.consts.TRANSITION_DELAY),
            lambda: jr.render_at(raster, self.consts.PLAYER_X, self.consts.PLAYER_Y, frame_player), 
            lambda: raster
        )

        # set player missile sprite
        frame_missile = jr.get_sprite_frame(level_sprites['player_missile'], state.step_counter)
        # only render player missile if it is active
        raster = jax.lax.cond(
            state.player_missile_state[3] > 0,
            lambda: jr.render_at(raster, state.player_missile_state[0], state.player_missile_state[1], frame_missile),
            lambda: raster
        )

        # enemies
        def get_plane_frame(i):
            """
            Get sprite and render offset for plane type enemies (levels 1, 2, 4)
            """

            # get enemy sprite
            enemy_sprite = level_sprites['enemy_pos'][state.enemy_states[0][2]]
            # use level boss sprite instead if enemy is level boss
            enemy_sprite = jax.lax.cond(
                state.level_boss == i,
                lambda: jax.lax.cond(
                    jnp.isin(state.enemy_states[i][2], jnp.array(range(1, 5))),
                    lambda: level_sprites['level_boss_left_left'],
                    lambda: level_sprites['level_boss_right_right']
                ),
                lambda: enemy_sprite
            )
            # use death animation sprite instead if enemy has just died
            enemy_sprite = jax.lax.cond(
                state.enemy_death_timers[i] > 0,
                lambda: level_sprites['enemy_death'],
                lambda: enemy_sprite
            )

            # return sprite and render offset of 0
            return jr.get_sprite_frame(enemy_sprite, 0), (0, 0)
        
        def get_heli_frame(i):
            """
            Get sprite and render offset for heli/ufo type enemies (levels 3, 5)
            """

            # get indices of current animation frame as well as sprite according to rotation
            frame_idx = jnp.array((state.step_counter%8) >= 4, jnp.int32)
            sprite_idx = ((state.enemy_states[i][2]%2) + 1 + jnp.array(
                state.enemy_states[i][2]>4, jnp.int32)*2)*jnp.array((state.enemy_states[i][2]%4)>0, jnp.int32)

            # get render offset according to current level
            render_offset = jax.lax.cond(
                state.level == 5,
                lambda: (0, 0),
                lambda: (jnp.array(sprite_idx==0, jnp.int32)*frame_idx, 
                         (jnp.array(sprite_idx==4, jnp.int32)-jnp.array(sprite_idx==2, jnp.int32))*frame_idx)
            )

            # get enemy sprite
            enemy_sprite = level_sprites['enemy_pos'][2*sprite_idx + frame_idx]
            # use level boss sprite instead if enemy is level boss
            enemy_sprite, render_offset = jax.lax.cond(
                state.level_boss == i,
                lambda: (jax.lax.cond(
                    jnp.isin(state.enemy_states[i][2], jnp.array(range(1, 5))),
                    lambda: jnp.array(level_sprites['level_boss_left_left']*frame_idx + 
                                      level_sprites['level_boss_left_right']*(1-frame_idx), jnp.uint8),
                    lambda: jnp.array(level_sprites['level_boss_right_left']*frame_idx + 
                                      level_sprites['level_boss_right_right']*(1-frame_idx), jnp.uint8)
                ), (0, 0)),
                lambda: (enemy_sprite, render_offset)
            )
            # use death animation sprite instead if enemy has just died
            enemy_sprite = jax.lax.cond(
                state.enemy_death_timers[i] > 0,
                lambda: level_sprites['enemy_death'],
                lambda: enemy_sprite
            )

            return jr.get_sprite_frame(enemy_sprite, 0), render_offset
        
        # use sprites for preceding level during first half of level transition animation
        level_to_use_sprites_from = state.level - jnp.logical_and(state.next_level_transition,
                                                                  state.respawn_timer >= self.consts.TRANSITION_DELAY_SECOND_STAGE).astype(jnp.int32)
        
        def render_enemy(i, raster):

            # use plane or heli frame according to current level
            frame, render_offset = jax.lax.cond(
                jnp.logical_or(level_to_use_sprites_from == 3, level_to_use_sprites_from == 5),
                lambda: get_heli_frame(i),
                lambda: get_plane_frame(i)
            )

            # get height indicator frame
            hi_sprite = general_sprites['black_line']
            hi_frame = jr.get_sprite_frame(hi_sprite, state.step_counter)
            # get position of height indicator belonging to enemy
            hi_interval = (self.consts.MAX_ENTITY_Y - self.consts.MIN_ENTITY_Y)/9
            hi_rel_height = jnp.array((state.enemy_states[i][1] - self.consts.MIN_ENTITY_Y)/hi_interval, jnp.int32)
            hi_height = self.consts.MIN_ENTITY_Y + hi_rel_height*hi_interval

            # render enemy
            ret_raster = jr.render_at(raster, state.enemy_states[i][0] + render_offset[0],
                                      state.enemy_states[i][1] + render_offset[1], frame)
            # render height indicator
            ret_raster = jr.render_at(ret_raster, 0, hi_height, hi_frame)
            # do not render enemy or height indicator if enemy is inactive
            return jax.lax.cond(
                jnp.logical_or(state.enemy_death_timers[i] > 0, state.enemy_states[i][3] > 0),
                lambda: ret_raster,
                lambda: raster
                )

        # render enemies
        raster = jax.lax.fori_loop(0, self.consts.MAX_NUMBER_OF_ENEMIES, render_enemy, raster)

        # enemy missiles
        frame_enemy_missile = jr.get_sprite_frame(level_sprites['enemy_missile'], state.step_counter)
        # only render enemy missile if it is active
        raster = jax.lax.fori_loop(
            0,
            self.consts.MAX_ENEMY_MISSILES,
            lambda i, val: jax.lax.cond(
                state.enemy_missile_states[i][3] > 0,
                lambda: jr.render_at(val, state.enemy_missile_states[i][0], state.enemy_missile_states[i][1], frame_enemy_missile),
                lambda: val
                ),
            raster)

        # clouds
        def render_cloud(i, raster):

            sprite = level_sprites['cloud']
            frame = jr.get_sprite_frame(sprite, state.step_counter)
            return jr.render_at(raster, state.cloud_positions[i][0], state.cloud_positions[i][1], frame, flip_horizontal=True)

        raster = jax.lax.fori_loop(0, 4, render_cloud, raster)

        # top and bottom wall
        # top
        black = jnp.array((0,0,0), dtype=jnp.uint8)
        raster = raster.at[0:self.consts.BLACK_BORDER_TOP_HEIGHT, :, :].set(black)
        frame_top_wall = jr.get_sprite_frame(general_sprites['top_wall'], state.step_counter)
        raster = jr.render_at(raster, 0, self.consts.BLACK_BORDER_TOP_HEIGHT, frame_top_wall)
        # bottom
        white = jnp.array((255,255,255), dtype=jnp.uint8)
        raster = raster.at[self.consts.MAX_ENTITY_Y + 1:self.consts.MAX_ENTITY_Y + 2, :, :].set(white)
        raster = raster.at[self.consts.MAX_ENTITY_Y + self.consts.WALL_BOTTOM_HEIGHT:self.consts.MAX_ENTITY_Y + 
                           1 + self.consts.WALL_BOTTOM_HEIGHT, :, :].set(white)
        frame_bottom_wall = jr.get_sprite_frame(general_sprites['bottom_wall'], state.step_counter)
        raster = jr.render_at(raster, 0, self.consts.MAX_ENTITY_Y + 2, frame_bottom_wall)
        raster = raster.at[self.consts.MAX_ENTITY_Y + 1 + self.consts.WALL_BOTTOM_HEIGHT:self.consts.HEIGHT, :, :].set(black)

        # score
        digit_sprites = general_sprites.get('digits', None)
        digits = jr.int_to_digits(state.score, max_digits = 6)
        raster = jr.render_label_selective(raster, 57, 7, digits, digit_sprites[0], 0, 6, spacing = 8)
        
        # lives 
        raster = jr.render_indicator(
            raster, 88, 18, state.lives - 1, 
            jr.get_sprite_frame(general_sprites['player_life'], state.step_counter), 
            spacing=-8
        )

        # use different top and bottom wall for respawn animation
        frame_respawn_top_wall = jr.get_sprite_frame(level_sprites['respawn_top_wall'], state.step_counter)
        frame_respawn_bottom_wall = jr.get_sprite_frame(general_sprites['respawn_bottom_wall'], state.step_counter)

        raster = jax.lax.cond(
            jnp.logical_and(state.respawn_timer > 0, 
                            state.respawn_timer <= self.consts.TRANSITION_DELAY_SECOND_STAGE + self.consts.TRANSITION_DELAY_FIRST_STAGE),
            lambda: jr.render_at(
                jr.render_at(raster, 0, self.consts.MAX_ENTITY_Y + 2, frame_respawn_bottom_wall), 
                0, self.consts.BLACK_BORDER_TOP_HEIGHT, frame_respawn_top_wall),
            lambda: raster
        )

        # enemies remaining line (different color during respawn animation)
        enemy_remaining_frame = jax.lax.cond(
            jnp.logical_and(state.respawn_timer > 0, 
                            state.respawn_timer <= self.consts.TRANSITION_DELAY_SECOND_STAGE + self.consts.TRANSITION_DELAY_FIRST_STAGE),
            lambda: jr.get_sprite_frame(level_sprites['enemy_remaining_brown'], state.step_counter),
            lambda: jr.get_sprite_frame(level_sprites['enemy_remaining'], state.step_counter)
        )

        # do not render enemies remaining during first stage of level transition
        render_enemies_remaining = jnp.logical_not(
            jnp.logical_and(state.next_level_transition > 0, 
                            state.respawn_timer >= self.consts.TRANSITION_DELAY_SECOND_STAGE))
        raster = jax.lax.cond(
            jnp.logical_and(render_enemies_remaining, state.enemies_remaining > self.consts.ENEMY_KILLS_TO_ELIMINATE),
            lambda: jr.render_at(raster, 72, 180, enemy_remaining_frame),
            lambda: raster
        )
        raster = jax.lax.cond(
            jnp.logical_and(render_enemies_remaining, state.enemies_remaining > 0),
            lambda: jr.render_at(raster, 80, 181, enemy_remaining_frame),
            lambda: raster
        )

        # next level transition animation
        next_level_transition_raster = jr.render_at(raster, 0, self.consts.MIN_ENTITY_Y, frame_bg)
        # flash bar behind player
        next_level_transition_raster = jax.lax.cond(
            state.respawn_timer % 8 > 3,
            lambda: jr.render_at(next_level_transition_raster, 0, 103, 
                                 jr.get_sprite_frame(general_sprites['transition_bar'][0], state.step_counter)),
            lambda: jr.render_at(next_level_transition_raster, 0, 103, 
                                 jr.get_sprite_frame(general_sprites['transition_bar'][1], state.step_counter))
        )
        next_level_transition_raster = jr.render_at(next_level_transition_raster, self.consts.PLAYER_X, 
                                                    self.consts.PLAYER_Y, frame_player)
        # use original rater outside of level transition
        raster = jax.lax.cond(
            jnp.logical_and(state.next_level_transition, state.respawn_timer <= self.consts.TRANSITION_DELAY),
            lambda: next_level_transition_raster,
            lambda: raster,
        )

        # render start screen at start of game
        start_screen_frame = jr.get_sprite_frame(general_sprites['start_screen'], state.step_counter)
        raster = jax.lax.cond(
            state.step_counter < self.consts.START_SCREEN_DELAY,
            lambda: jr.render_at(raster, 0, 0, start_screen_frame),
            lambda: raster
        )

        return raster