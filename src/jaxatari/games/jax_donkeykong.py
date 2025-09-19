import os
from functools import partial
from typing import NamedTuple, Tuple
import jax.lax

import jax.numpy as jnp
import chex
import pygame

import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as aj
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action


class DonkeyKongConstants(NamedTuple):
    # Screen dimensions
    WIDTH: int = 160
    HEIGHT: int = 210
    WINDOW_WIDTH: int = 160 * 3
    WINDOW_HEIGHT: int = 210 * 3

    # Frame rate
    FRAME_RATE: int = 30 # if more frame rate is provided, one needs to change the game behaviour

    # Donkey Kong position
    # Donkey Kong actually does nothing in game - only change sprites
    DONKEYKONG_X: int = 33
    DONKEYKONG_Y: int = 14

    # Girlfriend position - only single image of girlfriend
    GIRLFRIEND_X: int = 62
    GIRLFRIEND_Y: int = 17

    # Life Bar positions
    LEVEL_1_LIFE_BAR_1_Y: int = 116
    LEVEL_1_LIFE_BAR_2_Y: int = 124
    LEVEL_2_LIFE_BAR_1_Y: int = 112
    LEVEL_2_LIFE_BAR_2_Y: int = 120
    LIFE_BAR_X: int = 23

    # Hammer default position
    LEVEL_1_HAMMER_X: int = 68
    LEVEL_1_HAMMER_Y: int = 39
    LEVEL_1_HAMMER_SWING_X = 75
    LEVEL_2_HAMMER_X: int = 68
    LEVEL_2_HAMMER_Y: int = 78

    # Hammer Carry Duration
    HAMMER_MAX_CARRY_DURATION = 654  # maximal carry duration
    HAMMER_SWING_DURATION = 8        # if mario takes the hammer, swing up and down at each 8th frame

    # Hammer Hit Boxes
    HAMMER_HIT_BOX_X = 7
    HAMMER_HIT_BOX_Y = 4
    HAMMER_SWING_HIT_BOX_X = 6
    HAMMER_SWING_HIT_BOX_Y = 6

    # Trap positions
    TRAP_LEFT_Y: int = 52
    TRAP_RIGHT_Y: int = 104
    TRAP_FLOOR_2_X: int = 144
    TRAP_FLOOR_3_X: int = 116
    TRAP_FLOOR_4_X: int = 88
    TRAP_FLOOR_5_X: int = 60
    TRAP_WIDTH: int = 4

    # Digits position - for Game Score
    DIGIT_Y: int = 7
    FIRST_DIGIT_X: int = 96
    DISTANCE_DIGIT_X: int = 8
    NUMBER_OF_DIGITS_FOR_GAME_SCORE: int = 6
    NUMBER_OF_DIGITS_FOR_TIMER_SCORE: int = 4

    # Mario movement and physics
    LEVEL_1_MARIO_START_X = jnp.float32(176.0)
    LEVEL_1_MARIO_START_Y = jnp.float32(45.0)
    LEVEL_2_MARIO_START_X = jnp.float32(155.0)
    LEVEL_2_MARIO_START_Y = jnp.float32(32.0)
    MARIO_JUMPING_HEIGHT = jnp.float32(5.0)
    MARIO_JUMPING_FRAME_DURATION: int = 33
    MARIO_MOVING_SPEED = jnp.float32(0.335)  # pixels per frame
    MARIO_WALKING_ANIMATION_CHANGE_DURATION: int = 5
    MARIO_CLIMBING_SPEED = jnp.float32(0.333)
    MARIO_CLIMBING_ANIMATION_CHANGE_DURATION: int = 12

    # Game freeze duration if mario got hit by enemy
    GAME_FREEZE_DURATION = 70

    # If mario reaches that height on the ladder, game round is cleared
    LEVEL_1_GOAL_X = 40

    # Mario sprite indexes
    MARIO_WALK_SPRITE_0: int = 0
    MARIO_WALK_SPRITE_1: int = 1
    MARIO_WALK_SPRITE_2: int = 2
    MARIO_WALK_SPRITE_3: int = 3

    # Donkey Kong Sprite
    DONKEY_KONG_SPRITE_0: int = 0
    DONKEY_KONG_SPRITE_1: int = 1

    # Mario climbing sprite indexes
    MARIO_CLIMB_SPRITE_0: int = 0
    MARIO_CLIMB_SPRITE_1: int = 1

    # Barrel positions and sprites
    BARREL_START_X: int = 52
    BARREL_START_Y: int = 34
    BARREL_SPRITE_FALL: int = 0
    BARREL_SPRITE_RIGHT: int = 1
    BARREL_SPRITE_LEFT: int = 2

    # Barrel rolling probability
    BASE_PROBABILITY_BARREL_ROLLING_A_LADDER_DOWN_ROUND_1 = jnp.float32(0.14)
    BASE_PROBABILITY_BARREL_ROLLING_A_LADDER_DOWN_ROUND_2 = jnp.float32(0.36)
    BASE_PROBABILITY_BARREL_ROLLING_A_LADDER_DOWN_ROUND_3 = jnp.float32(0.34)
    BASE_PROBABILITY_BARREL_ROLLING_A_LADDER_DOWN_ROUND_4 = jnp.float32(0.34)
    BASE_PROBABILITY_BARREL_ROLLING_A_LADDER_DOWN_ROUND_5 = jnp.float32(0.5)
    BARREL_MOVING_SPEED: int = 1 # moving 1 pixel per frame

    # Hit boxes
    MARIO_HIT_BOX_X: int = 15
    MARIO_HIT_BOX_Y: int = 7
    BARREL_HIT_BOX_X: int = 8
    BARREL_HIT_BOX_Y: int = 8
    FIRE_HIT_BOX_X: int = 8
    FIRE_HIT_BOX_Y: int = 8

    # Fire
    FIRE_MOVING_SPEED = jnp.float32(0.49)
    FIRE_START_Y: int = 60
    FIRE_CHANGING_DIRECTION_PROB = jnp.float32(0.008)

    STAGE_2_FIRE_CHANGING_DIRECTION_DEFAULT_PROB = jnp.float32(0.0)
    STAGE_3_FIRE_CHANGING_DIRECTION_DEFAULT_PROB = jnp.float32(0.17)
    STAGE_4_FIRE_CHANGING_DIRECTION_DEFAULT_PROB = jnp.float32(0.0)
    STAGE_5_FIRE_CHANGING_DIRECTION_DEFAULT_PROB = jnp.float32(0.21)

    STAGE_2_FIRE_CHANGING_DIRECTION_INC_PROB = jnp.float32(0.008)
    STAGE_3_FIRE_CHANGING_DIRECTION_INC_PROB = jnp.float32(0.061)
    STAGE_4_FIRE_CHANGING_DIRECTION_INC_PROB = jnp.float32(0.011)
    STAGE_5_FIRE_CHANGING_DIRECTION_INC_PROB = jnp.float32(0.049)

    # Movement directions
    MOVING_UP: int = 0
    MOVING_RIGHT: int = 1
    MOVING_DOWN: int = 2
    MOVING_LEFT: int = 3

    # Bar start/end positions -- Level 1
    BAR_LEFT_Y: int = 32
    BAR_RIGHT_Y: int = 120
    BAR_1_LEFT_X: int = 193
    BAR_1_RIGHT_X: int = 193
    BAR_2_LEFT_X: int = 165 
    BAR_2_RIGHT_X: int = 172
    BAR_3_LEFT_X: int = 144
    BAR_3_RIGHT_X: int = 137
    BAR_4_LEFT_X: int = 109
    BAR_4_RIGHT_X: int = 116
    BAR_5_LEFT_X: int = 88
    BAR_5_RIGHT_X: int = 81
    BAR_6_LEFT_X: int = 60
    BAR_6_RIGHT_X: int = 60
    BAR_7_LEFT_X: int = 34
    BAR_7_RIGHT_X: int = 34

    # Bar start/end positions -- Level 2
    LEVEL_2_BAR_LEFT_Y: int = 32
    LEVEL_2_BAR_RIGHT_Y: int = 127
    LEVEL_2_BAR_1_LEFT_X: int = 172
    LEVEL_2_BAR_1_RIGHT_X: int = 172
    LEVEL_2_BAR_2_LEFT_X: int = 144
    LEVEL_2_BAR_2_RIGHT_X: int = 144
    LEVEL_2_BAR_3_LEFT_X: int = 116
    LEVEL_2_BAR_3_RIGHT_X: int = 116
    LEVEL_2_BAR_4_LEFT_X: int = 88
    LEVEL_2_BAR_4_RIGHT_X: int = 88
    LEVEL_2_BAR_5_LEFT_X: int = 60
    LEVEL_2_BAR_5_RIGHT_X: int = 60

    # Ladder
    LADDER_WIDTH: int = 4

    # Barrel spawn timing
    SPAWN_STEP_COUNTER_BARREL: int = 236

    # Scores added if mario doing following things:
    SCORE_FOR_JUMPING_OVER_ENEMY = 100
    SCORE_FOR_TRIGGERING_TRAP = 100
    SCORE_FOR_DESTROYING_BARREL = 800
    SCORE_FOR_DESTROYING_FIRE = 800
    SCORE_FOR_REACHING_GOAL_LEVEL_1 = 4100
    SCORE_FOR_REACHING_GOAL_LEVEL_2 = 3500 

    TIMER_REDUTION_DURATION = 128 # at every 128st frame, the timer will be reduced by 100, starting from 5000 "points"
    TIMER_REDUTION_AMOUNT = 100

    # Observation
    MAX_BARRELS = 4
    MAX_FIRES = 4
    MAX_TRAPS = 8
    MAX_LADDERS = 16
    
# To prevent Mario to walk outside the game spaces, set invisible wall on the left and right side of each stage
# stage means not level = 1 or 2, rather the bars on which Mario will walk during game play
class invisible_wall_each_stage(NamedTuple):
    stage: chex.Array
    left_end: chex.Array
    right_end: chex.Array

# Ladder - climbable -> some ladders are not supposed to be climbed by Mario, but barrel can roll down at those
class Ladder(NamedTuple):
    stage: chex.Array
    climbable: chex.Array
    start_x: chex.Array
    start_y: chex.Array
    end_x: chex.Array
    end_y: chex.Array

# Barrels - Level 1 Enemy
class BarrelPosition(NamedTuple):
    barrel_x: chex.Array
    barrel_y: chex.Array
    sprite: chex.Array
    moving_direction: chex.Array
    stage: chex.Array
    reached_the_end: chex.Array

# Fire - Level 2 enemy
class FirePosition(NamedTuple):
    fire_x: chex.Array
    fire_y: chex.Array
    moving_direction: chex.Array
    stage: chex.Array
    destroyed: chex.Array
    change_direction_prob: chex.Array # every fire has onw propability to change its direction

# Traps - Level 2 enemy or to Mario reaching goal
class TrapPosition(NamedTuple):
    trap_x: chex.Array
    trap_y: chex.Array
    stage: chex.Array
    triggered: chex.Array
    fall_protection: chex.Array # variable for protecting mario for the first instance after he triggers a trap; prevent mario to fall instantly

class DonkeyKongState(NamedTuple):
    game_started: chex.Array
    level: chex.Array
    game_timer: chex.Array  # needed basically to calculate the remaining time for the game
    game_remaining_time: chex.Array # if 0 Mario loses a life
    step_counter: chex.Array

    game_score: chex.Array
    game_round: chex.Array
    
    mario_x: chex.Array
    mario_y: chex.Array  
    mario_jumping: chex.Array   # jumping on spot
    mario_jumping_wide: chex.Array # jumping to left or right - Action.LEFTFIRE, Action.RIGHTFIRE
    mario_jumping_over_enemy: chex.Array
    mario_climbing: chex.Array
    start_frame_when_mario_jumped: chex.Array # variable to help calculating the frame time at which Mario can jump and reset his jump
    mario_view_direction: chex.Array
    mario_walk_frame_counter: chex.Array
    mario_climb_frame_counter: chex.Array
    mario_walk_sprite: chex.Array
    mario_climb_sprite: chex.Array
    mario_stage: chex.Array
    mario_life_counter: chex.Array

    mario_climbing_delay: chex.Array # variable needed especially for level 2 to prevent mario climbing all ladder from bottom to top without give the player the possibility to go right or left or jump

    mario_got_hit: chex.Array  # game should freeze if mario got hit by an enemy or the time is over
    game_freeze_start: chex.Array  # mario got hit, start frame
    mario_reached_goal: chex.Array

    barrels: BarrelPosition
    barrels_change_direction_prob: chex.Array
    fires: FirePosition
    traps: TrapPosition
    ladders: Ladder
    invisible_wall_each_stage: invisible_wall_each_stage
    random_key: chex.Array
    frames_since_last_barrel_spawn: chex.Array # some frames must be waited until next barrel can spawn

    hammer_x: chex.Array
    hammer_y: chex.Array
    hammer_can_hit: chex.Array
    hammer_taken: chex.Array
    hammer_carry_time: chex.Array
    hammer_usage_expired: chex.Array # only once per game round, hammer can be used
    block_jumping_and_climbing: chex.Array

    donkey_kong_sprite: chex.Array

class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray

class EntityPositionTrap(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    triggered: jnp.ndarray

class EntityPositionLadder(NamedTuple):
    start_x: jnp.ndarray
    start_y: jnp.ndarray
    end_x: jnp.ndarray
    end_y: jnp.ndarray
    width: jnp.ndarray
    

class DonkeyKongObservation(NamedTuple):
    # Scores / Timer
    total_score: jnp.ndarray
    time_remaining: jnp.ndarray
    mario_lives: jnp.ndarray
    goal_reached: jnp.ndarray
    level: jnp.ndarray

    # Player / Mario Position
    mario_position: EntityPosition # [x, y]
    mario_view_direction: jnp.ndarray # left and right
    mario_jumping: jnp.ndarray
    mario_climbing: jnp.ndarray
    
    # Hammer
    hammer_position: EntityPosition
    hammer_can_destroy_enemy: jnp.ndarray

    # Enemy
    barrels: EntityPosition
    barrel_mask: jnp.ndarray # 0 = inactive barrels, 1 = active barrels
    # fires: EntityPosition
    # fire_mask: jnp.ndarray  

    # # Traps
    # traps: EntityPositionTrap

    # # Ladders
    # ladders: EntityPositionLadder
    # ladder_mask: jnp.ndarray # 0 for some ladders for level 1 which are no ladders, its only a place holder because JAX needs consistent array sizes
                            # ladder_mask do NOT implie if those ladders are climbable or not

class DonkeyKongInfo(NamedTuple):
    time: jnp.ndarray
    all_rewards: chex.Array

class JaxDonkeyKong(JaxEnvironment[DonkeyKongState, DonkeyKongObservation, DonkeyKongInfo, DonkeyKongConstants]):
    def __init__(self, consts: DonkeyKongConstants = None, reward_funcs: list[callable]=None):
        consts = consts or DonkeyKongConstants()
        super().__init__(consts)
        self.renderer = DonkeyKongRenderer(self.consts)
        self.frame_stack_size = 4
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.action_set = [
            Action.NOOP,
            Action.FIRE,
            Action.RIGHT,
            Action.LEFT,
            Action.UP,
            Action.DOWN,
            Action.RIGHTFIRE,
            Action.LEFTFIRE,
        ]
        self.obs_size = 0

    # Bars as lienar functions - given y position of anything (can be Mario, Barrel, Fire) and the stage, it calculates the corresponding x position
    # That function is needed because some bars on level 1 are crooked
    def bar_linear_equation(self, stage, y, level=1):
        y_1, y_2 = jax.lax.cond(
            level==1,
            lambda _: (self.consts.BAR_LEFT_Y, self.consts.BAR_RIGHT_Y),
            lambda _: (self.consts.LEVEL_2_BAR_LEFT_Y, self.consts.LEVEL_2_BAR_RIGHT_Y),
            operand=None
        )

        # Bar position are measured by pixel for every single stage
        x_1_values = jax.lax.cond(
            level==1,
            lambda _: [self.consts.BAR_1_LEFT_X, self.consts.BAR_2_LEFT_X, self.consts.BAR_3_LEFT_X, self.consts.BAR_4_LEFT_X, self.consts.BAR_5_LEFT_X, self.consts.BAR_6_LEFT_X, self.consts.BAR_7_LEFT_X],
            lambda _: [self.consts.LEVEL_2_BAR_1_LEFT_X, self.consts.LEVEL_2_BAR_2_LEFT_X, self.consts.LEVEL_2_BAR_3_LEFT_X, self.consts.LEVEL_2_BAR_4_LEFT_X, self.consts.LEVEL_2_BAR_5_LEFT_X, self.consts.LEVEL_2_BAR_5_LEFT_X, self.consts.LEVEL_2_BAR_5_LEFT_X],
            operand=None
        )
        x_2_values = jax.lax.cond(
            level==1,
            lambda _: [self.consts.BAR_1_RIGHT_X, self.consts.BAR_2_RIGHT_X, self.consts.BAR_3_RIGHT_X, self.consts.BAR_4_RIGHT_X, self.consts.BAR_5_RIGHT_X, self.consts.BAR_6_RIGHT_X, self.consts.BAR_7_RIGHT_X],
            lambda _: [self.consts.LEVEL_2_BAR_1_RIGHT_X, self.consts.LEVEL_2_BAR_2_RIGHT_X, self.consts.LEVEL_2_BAR_3_RIGHT_X, self.consts.LEVEL_2_BAR_4_RIGHT_X, self.consts.LEVEL_2_BAR_5_RIGHT_X, self.consts.LEVEL_2_BAR_5_RIGHT_X, self.consts.LEVEL_2_BAR_5_RIGHT_X],
            operand=None
        )

        index = stage - 1
        branches = [lambda _, v=val: jnp.array(v) for val in x_1_values]
        x_1 = jax.lax.switch(index, branches, operand=None)
        branches = [lambda _, v=val: jnp.array(v) for val in x_2_values]
        x_2 = jax.lax.switch(index, branches, operand=None)

        m = ((x_2 - x_1) / (y_2 - y_1)).astype(jnp.float32)
        b = (x_1 - m * y_1).astype(jnp.float32)

        x = (m * y + b).astype(jnp.float32)
        return x

    @partial(jax.jit, static_argnums=(0,))
    def init_ladders_for_level(self, level: int) -> Ladder:
        # Ladder positions for level 1  --- the last 3 ladders are dummy ladders which do not exist in the real game
        # this is needed because jax needs same size of array for the Ladders to compile correctly
        Ladder_level_1 = Ladder(
            stage=jnp.array([6, 5, 5, 4, 4, 4, 3, 3, 3, 2, 2, 1, 1,                                                     -1, -1, -1], dtype=jnp.int32),
            climbable=jnp.array([True, False, True, True, True, False, False, True, True, True, True, False, True,      False, False, False]),
            start_x=jnp.array([59, 77, 74, 102, 104, 106, 134, 132, 130, 158, 161, 185, 185,                            -1, -1, -1], dtype=jnp.int32),
            start_y=jnp.array([76, 74, 106, 46, 66, 98, 62, 86, 106, 46, 78, 70, 106,                                   -1, -1, -1], dtype=jnp.int32),
            end_x=jnp.array([34, 53, 53, 79, 78, 76, 104, 106, 108, 135, 133, 161, 164,                                 -1, -1, -1], dtype=jnp.int32),
            end_y=jnp.array([76, 74, 106, 46, 66, 98, 62, 86, 106, 46, 78, 70, 106,                                     -1, -1, -1], dtype=jnp.int32),
        )

        # Ladder positions for level 2
        Ladder_level_2 = Ladder(
            stage=jnp.array([4, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1], dtype=jnp.int32),
            climbable=jnp.array([True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]),
            start_x=jnp.array([171, 171, 171, 171, 143, 143, 143, 143, 115, 115, 115, 115, 87, 87, 87, 87], dtype=jnp.int32),
            start_y=jnp.array([40, 60, 96, 116, 40, 60, 96, 116, 40, 60, 96, 116, 40, 60, 96, 116], dtype=jnp.int32),
            end_x=jnp.array([143, 143, 143, 143, 115, 115, 115, 155, 87, 87, 87, 87, 59, 59, 59, 59], dtype=jnp.int32),
            end_y=jnp.array([40, 60, 96, 116, 40, 60, 96, 116, 40, 60, 96, 116, 40, 60, 96, 116], dtype=jnp.int32),
        )

        return jax.lax.cond(
            level == 1,
            lambda _: Ladder_level_1,
            lambda _: Ladder_level_2,
            operand=None
        )

    @partial(jax.jit, static_argnums=(0,))
    def init_invisible_wall_for_level(self, level: int) -> invisible_wall_each_stage:
        # Set invisible wall depending of level
        invisible_wall_level_1 = invisible_wall_each_stage(
            stage=jnp.array([6, 5, 4, 3, 2, 1], dtype=jnp.int32),
            left_end=jnp.array([32, 37, 32, 37, 32, 37], dtype=jnp.int32),
            right_end=jnp.array([113, 120, 113, 120, 113, 120], dtype=jnp.int32),
        )
        # level 2
        invisible_wall_level_2 = invisible_wall_each_stage(
            stage=jnp.array([6, 5, 4, 3, 2, 1], dtype=jnp.int32),
            left_end=jnp.array([32, 32, 32, 32, 32, 32], dtype=jnp.int32),
            right_end=jnp.array([120, 120, 120, 120, 120, 120], dtype=jnp.int32),
        )

        return jax.lax.cond(
            level == 1,
            lambda _: invisible_wall_level_1,
            lambda _: invisible_wall_level_2,
            operand=None
        )


    # calculate if there is a collision between two object (e.g. mario and barrel)
    @staticmethod
    @jax.jit
    def _collision_between_two_objects(
        obj_a_x, obj_a_y, hit_box_a_x, hit_box_a_y,
        obj_b_x, obj_b_y, hit_box_b_x, hit_box_b_y
    ):
        a_top    = obj_a_x
        a_bottom = obj_a_x + hit_box_a_x
        a_left   = obj_a_y
        a_right  = obj_a_y + hit_box_a_y

        b_top    = obj_b_x
        b_bottom = obj_b_x + hit_box_b_x
        b_left   = obj_b_y
        b_right  = obj_b_y + hit_box_b_y

        return (
            (a_left < b_right) &
            (a_right > b_left) &
            (a_top < b_bottom) &
            (a_bottom > b_top)
        )

    # enemy step
    # level 1 --> Barrels rolling down
    # level 2 --> Fire moving left to right and vice versa + Traps
    @partial(jax.jit, static_argnums=(0,))
    def _enemy_step(self, state):

        level_1_state = self._barrel_step(state)
        level_2_state = self._fire_step(state)
        level_2_state = self._trap_step(level_2_state)

        return jax.lax.cond(
            state.level == 1,
            lambda _: level_1_state,
            lambda _: level_2_state,
            operand=None
        )

    # Level 2 enemy step
    # Fire enemy
    @partial(jax.jit, static_argnums=(0,))
    def _fire_step(self, state):
        # somehow in the original game: fires are not spanwning simultaneously, but after frame by frame
        # will affect the game only in the first 3-4 frames
        # just set the variable "destroyed" to True and rendering will create the raster properly
        def spawn_new_fires(state):
            destroyed = jax.lax.cond(
                jnp.logical_and(state.fires.destroyed[0] == False, jnp.logical_and(state.fires.destroyed[1], jnp.logical_and(state.fires.destroyed[2], state.fires.destroyed[3]))),
                lambda _: state.fires.destroyed.at[1].set(False),
                lambda _: jax.lax.cond(
                    jnp.logical_and(state.fires.destroyed[0] == False, jnp.logical_and(state.fires.destroyed[1] == False, jnp.logical_and(state.fires.destroyed[2], state.fires.destroyed[3]))),
                    lambda _: state.fires.destroyed.at[2].set(False),
                    lambda _: jax.lax.cond(
                        jnp.logical_and(state.fires.destroyed[0] == False, jnp.logical_and(state.fires.destroyed[1] == False, jnp.logical_and(state.fires.destroyed[2] == False, state.fires.destroyed[3]))),
                        lambda _: state.fires.destroyed.at[3].set(False),
                        lambda _: state.fires.destroyed,
                        operand=None
                    ),
                    operand=None
                ),
                operand=None 
            )
            fires = state.fires._replace(destroyed=destroyed)
            new_state = state._replace(fires=fires)
            return new_state
        new_state = spawn_new_fires(state)

        # implement moving of fires
        # simple: because it moves from right to left or vise versa
        # move fire in the given direction (variable moving_direction)
        def fire_move(state):
            # for loop implements movement for each fire
            def each_fire_movement(i, state):
                # movement can only be towards right or left
                new_fire_y = jax.lax.cond(
                    state.fires.moving_direction[i] == self.consts.MOVING_RIGHT,
                    lambda _: state.fires.fire_y[i] + self.consts.FIRE_MOVING_SPEED,
                    lambda _: state.fires.fire_y[i] - self.consts.FIRE_MOVING_SPEED,
                    operand=None
                )
                fire_y = state.fires.fire_y.at[i].set(new_fire_y)
                fires = state.fires._replace(fire_y = fire_y)
                new_state = state._replace(fires=fires)
                fire_can_move = jnp.logical_not(state.fires.destroyed[i])
                return jax.lax.cond(
                    fire_can_move,
                    lambda _: new_state,
                    lambda _: state,
                    operand=None
                )
            new_state = jax.lax.fori_loop(0, len(state.fires.fire_x), each_fire_movement, state)

            # fires can move only if the is not in freeze mode, so state.mario_reached_goal == False, state.mario_got_hit == False 
            return jax.lax.cond(
                jnp.logical_and(state.mario_reached_goal == False, state.mario_got_hit == False),
                lambda _: new_state,
                lambda _: state,
                operand=None
            )
        new_state = fire_move(new_state)

        # implement the change of moving direction of the fires
        # function dows not implement the actual movement
        # changing the direction is based on a propability
        def change_moving_direction_fire(state):
            def change_direction_for_each_fire(i, state):
                prob_per_sec = state.fires.change_direction_prob[i] # prob of change PER Sec
                direction_change_prob = 1.0 - (1.0 - prob_per_sec) ** (1.0 / self.consts.FRAME_RATE) # basic prop to change direction PER FRAME
                fire_y, moving_direction = jax.lax.cond(
                    state.fires.moving_direction[i] == self.consts.MOVING_RIGHT,
                    lambda _: (state.fires.fire_y[i] + self.consts.FIRE_MOVING_SPEED, state.fires.moving_direction.at[i].set(self.consts.MOVING_LEFT)),
                    lambda _: (state.fires.fire_y[i] - self.consts.FIRE_MOVING_SPEED, state.fires.moving_direction.at[i].set(self.consts.MOVING_RIGHT)),
                    operand=None
                )

                # check if fire is already at the invisible wall on right or left side, if so prop = 1.0
                direction_change_prob = jax.lax.cond(
                    state.invisible_wall_each_stage.left_end[0] >= fire_y,
                    lambda _: 1.0,
                    lambda _: direction_change_prob,
                    operand=None
                )
                direction_change_prob = jax.lax.cond(
                    state.invisible_wall_each_stage.right_end[0] <= fire_y,
                    lambda _: 1.0,
                    lambda _: direction_change_prob,
                    operand=None
                )

                # check if fire must change the direction if traps are triggered, so it can not move over it
                # also prop = 1.0 if a fire reached a triggered trap, because it can not fly over it
                def check_for_each_trap(j, has_to_change_dir):
                    fire_flying_over_trap = (
                        (state.traps.stage[j] == state.fires.stage[i])
                        & (state.fires.fire_y[i] <= state.traps.trap_y[j])
                        & ((state.fires.fire_y[i] + self.consts.FIRE_HIT_BOX_Y) >= (state.traps.trap_y[j] + self.consts.TRAP_WIDTH))
                    ) 
                    change_in_dir = jnp.logical_and(fire_flying_over_trap, state.traps.triggered[j])
                    return jax.lax.cond(
                        change_in_dir,
                        lambda _: True,
                        lambda _: has_to_change_dir,
                        operand=None
                    )
                has_to_change_dir = jax.lax.fori_loop(0, len(state.traps.trap_x), check_for_each_trap, False)
                direction_change_prob = jax.lax.cond(
                    has_to_change_dir,
                    lambda _: 1.0,
                    lambda _: direction_change_prob,
                    operand=None
                )

                # use the propability to change the direction
                # if a fire forced to change his direction --> direction_change_prob == 1.0
                def should_change(key, direction_change_prob) -> bool:
                    rnd = jax.random.uniform(key, shape=())
                    return rnd < direction_change_prob
                key = jax.random.PRNGKey(jnp.round(state.fires.fire_x[i]).astype(jnp.int32) + jnp.round(state.fires.fire_y[i]).astype(jnp.int32) + state.fires.stage[i] + state.step_counter)
                change_direction = should_change(key, direction_change_prob)

                # new_state with opposite moving direction
                fires = state.fires._replace(moving_direction=moving_direction)
                new_state = state._replace(fires=fires)
                return jax.lax.cond(
                    change_direction,
                    lambda _: new_state,
                    lambda _: state,
                    operand=None
                )
            new_state = jax.lax.fori_loop(0, len(state.fires.fire_y), change_direction_for_each_fire, state)
            return new_state
        new_state = change_moving_direction_fire(new_state)
        return new_state

    # Level 2 enemy step
    # Traps 
    @partial(jax.jit, static_argnums=(0,))
    def _trap_step(self, state):
        new_state = state
        # variable triggered will be set to True if Mario is directly over a trap
        # it does not check if Mario falls into the trap and resetting the game --> look check_if_mario_can_fall_trap
        def check_for_each_trap_if_triggered(i, state):
            # prepare new state, where the i-th trap is triggered
            triggered = state.traps.triggered.at[i].set(True)
            fall_protection = jax.lax.cond(
                state.traps.triggered[i] == False,
                lambda _: state.traps.fall_protection.at[i].set(state.mario_view_direction),
                lambda _: state.traps.fall_protection,
                operand=None
            )
            traps = state.traps._replace(triggered=triggered, fall_protection=fall_protection)
            new_state = state._replace(traps=traps)

            # check if really the i-th trap is triggered
            trap_is_triggered = (
                (state.traps.stage[i] == state.mario_stage)
                & (state.mario_y <= state.traps.trap_y[i])
                & ((state.mario_y + self.consts.MARIO_HIT_BOX_Y) >= (state.traps.trap_y[i] + self.consts.TRAP_WIDTH))
            )

            # update game score if trap is triggered for the first time
            # player can gain score points 
            game_score = state.game_score + self.consts.SCORE_FOR_TRIGGERING_TRAP
            new_state = jax.lax.cond(
                state.traps.triggered[i] == False,
                lambda _: new_state._replace(game_score=game_score),
                lambda _: new_state,
                operand=None
            )

            return jax.lax.cond(
                trap_is_triggered,
                lambda _: new_state,
                lambda _: state,
                operand=None
            )
        new_state = jax.lax.fori_loop(0, len(new_state.traps.trap_x), check_for_each_trap_if_triggered, new_state)

        # function implements the case if Mario falling into a trap or not
        def check_if_mario_can_fall_trap(i, state):
            mario_walking_over_trap = (
                (state.traps.stage[i] == state.mario_stage)
                & (state.mario_y <= state.traps.trap_y[i])
                & ((state.mario_y + self.consts.MARIO_HIT_BOX_Y) >= (state.traps.trap_y[i] + self.consts.TRAP_WIDTH))
            )

            # if Mario is in jumping state --> no fall into trap
            mario_jumping = jnp.logical_or(state.mario_jumping, state.mario_jumping_wide)

            # this variable is needed to protect player/mario to instantly fall into the trap
            # in the original game, if a trap is triggered, Mario could not fall instantly. Only when he is moving in other direction or walking over the trap in the second time
            mario_protected = state.traps.fall_protection[i] == state.mario_view_direction

            # if mario does not walk over trap --> no more protection for that trap --> reset fall_protection to -1
            fall_protection = state.traps.fall_protection.at[i].set(-1)
            traps = state.traps._replace(fall_protection=fall_protection)
            state_fall_protection_reset = state._replace(traps=traps)

            # if mario is not protected anymore (mario_protected) and he is not jumping --> mario_got_hit = True --> reset game round
            game_freeze_start = jax.lax.cond(
                state.mario_got_hit == False,
                lambda _: state.step_counter,
                lambda _: state.game_freeze_start,
                operand=None
            )
            state_mario_falls = state._replace(
                mario_got_hit = True,
                game_freeze_start = game_freeze_start,
            )

            return jax.lax.cond(
                mario_walking_over_trap == False,
                lambda _: state_fall_protection_reset,
                lambda _: jax.lax.cond(
                    jnp.logical_and(mario_protected == False, mario_jumping == False),
                    lambda _: state_mario_falls,
                    lambda _: state,
                    operand=None
                ),
                operand=None
            )
        new_state = jax.lax.fori_loop(0, len(new_state.traps.trap_x), check_if_mario_can_fall_trap, new_state)

        # check if all traps are triggered --> if so player succefully cleared the level 2 
        # set mario_reached_goal = True to indicated other function to reset the level to level 1.
        def check_all_traps_triggered(state):
            all_traps_triggered = jnp.all(state.traps.triggered)
            game_freeze_start = jax.lax.cond(
                jnp.logical_and(state.mario_reached_goal == False, all_traps_triggered == True),
                lambda _: state.step_counter,
                lambda _: state.game_freeze_start,
                operand=None
            )
            new_state = state._replace(
                mario_reached_goal = all_traps_triggered,
                game_freeze_start = game_freeze_start,
            )
            return new_state
        new_state = check_all_traps_triggered(new_state)

        return new_state


    # Level 1 enemy step
    # Barrel enemy
    @partial(jax.jit, static_argnums=(0,))
    def _barrel_step(self, state):
        step_counter = state.step_counter
        
        # pick other sprite for animation after 8 frames --> for animation
        should_pick_next_sprite = step_counter % 8 == 0
        
        new_state = state
        # calculate new position of barrels
        def update_single_barrel(x, y, direction, sprite, stage, reached_the_end):
            ladders = state.ladders

            # change sprite animation
            def flip_sprite(sprite):
                return jax.lax.cond(
                    sprite == self.consts.BARREL_SPRITE_RIGHT,
                    lambda _: self.consts.BARREL_SPRITE_LEFT,
                    lambda _: self.consts.BARREL_SPRITE_RIGHT,
                    operand=None
                )
            
            sprite = jax.lax.cond(
                jnp.logical_and(should_pick_next_sprite, direction != self.consts.MOVING_DOWN),
                lambda _: flip_sprite(sprite),
                lambda _: sprite,
                operand=None
            )

            # change x position if the barrel is still falling
            # if barrel is landed on the down stage, change the moving direction
            # barrels have variable direction which indicate if the barrels are rolling or falling
            # this function calculate the new x position if barrel is falling
            def change_x_if_barrel_is_falling(x, y, direction, sprite, stage):
                new_x = x + 2

                bar_x = jnp.round(self.bar_linear_equation(stage, y) - self.consts.BARREL_HIT_BOX_X).astype(int)
                # change the dirction to left or right (rolling) if barrel reached the next down stage/bar
                new_direction = jax.lax.cond(
                    new_x >= bar_x,
                    lambda _: jax.lax.cond(
                        stage % 2 == 0,
                        lambda _: self.consts.MOVING_RIGHT,
                        lambda _: self.consts.MOVING_LEFT,
                        operand=None
                    ),
                    lambda _: direction,
                    operand=None
                )
                new_sprite = jax.lax.cond(
                    new_x >= bar_x,
                    lambda _: self.consts.BARREL_SPRITE_RIGHT,
                    lambda _: sprite,
                    operand=None
                )

                return jax.lax.cond(
                    jnp.logical_and(jnp.logical_and(direction == self.consts.MOVING_DOWN, state.mario_got_hit == False), state.mario_reached_goal == False),
                    lambda _: (new_x, y, new_direction, new_sprite, stage),
                    lambda _: (x, y, direction, sprite, stage),
                    operand=None
                )
            x, y, direction, sprite, stage = change_x_if_barrel_is_falling(x, y, direction, sprite, stage)

            # change position
            # check if barrel can fall (ladder or end of bar)
            def check_if_barrel_will_fall(x, y, direction, sprite, stage):
                prob_barrel_rolls_down_a_ladder = state.barrels_change_direction_prob
                curr_stage = stage - 1

                # check if there is an another barrel directly under that stage
                # if so this barrel should not fall
                # prob -> 0.0 %
                def check_for_blocking_barrels(i, value):
                    blocking = curr_stage == state.barrels.stage[i]
                    return jax.lax.cond(
                        blocking,
                        lambda _: True,
                        lambda _: value,
                        operand=None
                    )
                
                # need a prop. of falling
                # it is 1.0 if surely the barrel reaches the end of a bar/stage
                # it can also fall at a ladder, but only with some constante propability and if there is no another barrel on below stage
                another_barrel_blocking_from_falling = jax.lax.fori_loop(0, len(state.barrels.barrel_x), check_for_blocking_barrels, False)
                prob_barrel_rolls_down_a_ladder = jax.lax.cond(
                    another_barrel_blocking_from_falling,
                    lambda _: 0.0,
                    lambda _: prob_barrel_rolls_down_a_ladder,
                    operand=None
                )
                
                # check first if barrel is positioned on top of a ladder
                mask = jnp.logical_and(ladders.stage == curr_stage, ladders.end_y == y)
                barrel_is_on_ladder = jnp.any(mask)
                key = jax.random.PRNGKey(jnp.round(x).astype(jnp.int32) + jnp.round(y).astype(jnp.int32) + stage + state.step_counter)
                roll_down_prob = jax.random.bernoulli(key, prob_barrel_rolls_down_a_ladder)

                new_direction = self.consts.MOVING_DOWN
                new_sprite = self.consts.BARREL_SPRITE_FALL
                new_x = x + 1
                new_stage = stage - 1

                # check secondly if barrel is positioned at the end of a bar
                bar_y = jax.lax.cond(
                    stage % 2 == 0,
                    lambda _: self.consts.BAR_RIGHT_Y,
                    lambda _: self.consts.BAR_LEFT_Y,
                    operand=None
                )
                new_direction_2 = self.consts.MOVING_DOWN
                new_stage_2 = stage - 1
                barrel_is_over_the_bar = jax.lax.cond(
                    stage % 2 == 0,
                    lambda _: jax.lax.cond(
                        y >= self.consts.BAR_RIGHT_Y,
                        lambda _: True,
                        lambda _: False,
                        operand=None
                    ),
                    lambda _: jax.lax.cond(
                        y <= self.consts.BAR_LEFT_Y,
                        lambda _: True,
                        lambda _: False,
                        operand=None
                    ),
                    operand=None
                )

                return jax.lax.cond(
                    jnp.logical_and(barrel_is_on_ladder, jnp.logical_and(direction != self.consts.MOVING_DOWN, jnp.logical_and(roll_down_prob, jnp.logical_and(state.mario_got_hit == False, state.mario_reached_goal == False)))),
                    lambda _: (new_x, y, new_direction, new_sprite, new_stage),
                    lambda _: jax.lax.cond(
                        barrel_is_over_the_bar,
                        lambda _: (x, y, new_direction_2, sprite, new_stage_2),
                        lambda _: (x, y, direction, sprite, stage),
                        operand=None
                    ),
                    operand=None
                )
            x, y, direction, sprite, stage = check_if_barrel_will_fall(x, y, direction, sprite, stage)

            # change y (x) positions when barrel is rolling on bar
            # function uses the bar_linear_equation function to calculate the proper height
            def barrel_rolling_on_a_bar(x, y, direction, sprite, stage):
                new_y = jax.lax.cond(
                    direction == self.consts.MOVING_RIGHT,
                    lambda _: y + self.consts.BARREL_MOVING_SPEED,
                    lambda _: y - self.consts.BARREL_MOVING_SPEED,
                    operand=None
                )
                new_x = jnp.round(self.bar_linear_equation(stage, new_y) - self.consts.BARREL_HIT_BOX_X).astype(int)
                return jax.lax.cond(
                    jnp.logical_and(jnp.logical_and(direction != self.consts.MOVING_DOWN, state.mario_got_hit == False), state.mario_reached_goal == False),
                    lambda _: (new_x, new_y, direction, sprite, stage),
                    lambda _: (x, y, direction, sprite, stage),
                    operand=None
                )
            x, y, direction, sprite, stage = barrel_rolling_on_a_bar(x, y, direction, sprite, stage)

            # mark x = y = -1 as a barrel reaches the end
            # marking them as -1 indicated the renderer to not render barrel into raster
            def mark_barrel_if_reached_end(x, y, direction, sprite, stage, reached_the_end):
                return jax.lax.cond(
                    jnp.logical_and(stage == 1, y <= self.consts.BAR_LEFT_Y),
                    lambda _: (-1, -1, direction, sprite, stage, True),
                    lambda _: (x, y, direction, sprite, stage, reached_the_end),
                    operand=None
                )
            x, y, direction, sprite, stage, reached_the_end = mark_barrel_if_reached_end(x, y, direction, sprite, stage, reached_the_end)

            return jax.lax.cond(
                reached_the_end == False,
                lambda _: (x, y, direction, sprite, stage, reached_the_end),
                lambda _: (-1, -1, direction, sprite, stage, reached_the_end),
                operand=None
            )
        update_all_barrels = jax.vmap(update_single_barrel) # all 4 barrels will execute function update_all_barrels

        # update barrels into new state
        barrels = new_state.barrels
        new_barrel_x, new_barrel_y, new_barrel_moving_direction, new_sprite, new_stage, new_reached_the_end = update_all_barrels(
            barrels.barrel_x, barrels.barrel_y, barrels.moving_direction, barrels.sprite, barrels.stage, barrels.reached_the_end
        )
        barrels = barrels._replace(
            barrel_x = new_barrel_x,
            barrel_y = new_barrel_y,
            moving_direction = new_barrel_moving_direction,
            sprite = new_sprite,
            stage=new_stage,
            reached_the_end=new_reached_the_end
        )
        new_state = new_state._replace(
            barrels=barrels
        )

        # new random key
        key, subkey = jax.random.split(state.random_key)
        new_state = new_state._replace(random_key=key)

        # Skip every second frame
        should_move = step_counter % 2 == 0

        # spawn a new barrel if possible
        # max 4 barrels
        # there is a constante time until next barrel can spawn
        def spawn_new_barrel(state):
            barrels = state.barrels

            # check if there are less than 4 barrels in game right here because max barrels in Donkey Kong is 4.
            def is_max_number_of_barrels_reached(i, idx):
                changable_idx = i
                return jax.lax.cond(
                    jnp.logical_and(idx == -1, barrels.reached_the_end[i] == True),
                    lambda _: changable_idx,
                    lambda _: idx,
                    operand=None
                )
            idx = jax.lax.fori_loop(0, len(barrels), is_max_number_of_barrels_reached, -1)
            # if idx != -1 means a new barrel can be theoretically spawn
            # we only now need to check if there is enough space between the new barrel and the earlier barrel
            def update_barrels(barrels, idx):
                return BarrelPosition(
                    barrel_x=barrels.barrel_x.at[idx].set(self.consts.BARREL_START_X),
                    barrel_y=barrels.barrel_y.at[idx].set(self.consts.BARREL_START_Y),
                    sprite=barrels.sprite.at[idx].set(self.consts.BARREL_SPRITE_RIGHT),
                    moving_direction=barrels.moving_direction.at[idx].set(self.consts.MOVING_RIGHT),
                    stage=barrels.stage.at[idx].set(6),
                    reached_the_end=barrels.reached_the_end.at[idx].set(False),
                )
            new_barrels = jax.lax.cond(
                idx != -1,
                lambda _: update_barrels(barrels, idx),
                lambda _: barrels,
                operand=None
            )

            new_state = state._replace(
                barrels=new_barrels,
                frames_since_last_barrel_spawn=1,
            )

            return jax.lax.cond(
                jnp.logical_and(state.frames_since_last_barrel_spawn >= self.consts.SPAWN_STEP_COUNTER_BARREL, jnp.logical_and(idx != -1, jnp.logical_and(state.mario_got_hit == False, state.mario_reached_goal == False))),
                lambda _: new_state,
                lambda _: state,
                operand=None
            )
        new_state = spawn_new_barrel(new_state)

        # return either new position or old position because of frame skip/ step counter
        return jax.lax.cond(
            should_move, lambda _: new_state, lambda _: state, operand=None
        )


    @partial(jax.jit, static_argnums=(0,))
    def _mario_step(self, state, action: chex.Array):    
        # there are multiple action which mario/player can execute

        # Jumping with Action.FIRE --> actually on the spot, there is a second function where Mario can jump wise (Action.LEFT/RIGHTFIRE)
        # several things needs to be considered --> While mario is jumping --> Action.FIRE does nothing
        # Mario is climbing --> Action.FIRE does nothing
        # Game is freezes (enemy hit or goal reached) --> Action.FIRE dows nothing
        def jumping_on_spot(state):
            new_state = state

            # Action.FIRE
            start_frame_when_mario_jumped = state.step_counter
            mario_jumping = True
            mario_x = state.mario_x - self.consts.MARIO_JUMPING_HEIGHT
            new_state = new_state._replace(
                start_frame_when_mario_jumped=start_frame_when_mario_jumped,
                mario_jumping=mario_jumping,
                mario_x=mario_x.astype(jnp.float32),
            )

            return jax.lax.cond(
                jnp.logical_and(action == Action.FIRE, jnp.logical_and(jnp.logical_and(jnp.logical_and(state.mario_climbing == False, state.mario_jumping == False), state.mario_jumping_wide == False), jnp.logical_and(state.block_jumping_and_climbing == False, state.mario_reached_goal == False))),
                lambda _: new_state,
                lambda _: state,
                operand=None
            )
        new_state = jumping_on_spot(state)

        # Jumping wide with Action.LEFTFIRE and Action.RIGHTFIRE
        # Very similar to jumping_on_spot, some prerequisites have to be considered
        def jumping_right(state):
            new_state_start_jumping = state._replace(
                start_frame_when_mario_jumped = state.step_counter,
                mario_jumping_wide = True,
                mario_view_direction = self.consts.MOVING_RIGHT,
                mario_x = (state.mario_x - self.consts.MARIO_JUMPING_HEIGHT).astype(jnp.float32),
                mario_y = (state.mario_y + self.consts.MARIO_MOVING_SPEED).astype(jnp.float32)
            )
            new_mario_x = jnp.round(self.bar_linear_equation(state.mario_stage, state.mario_y, state.level) - self.consts.MARIO_HIT_BOX_X) - 2
            new_state_already_jumping = state._replace(
                mario_y = (state.mario_y + self.consts.MARIO_MOVING_SPEED).astype(jnp.float32),
                mario_x = (new_mario_x - self.consts.MARIO_JUMPING_HEIGHT).astype(jnp.float32),
            )
            return jax.lax.cond(
                jnp.logical_and(action == Action.RIGHTFIRE, jnp.logical_and(jnp.logical_and(jnp.logical_and(state.mario_climbing == False, state.mario_jumping_wide == False), state.mario_jumping == False), jnp.logical_and(state.block_jumping_and_climbing == False, state.mario_reached_goal == False))),
                lambda _: new_state_start_jumping,
                lambda _: jax.lax.cond(
                    jnp.logical_and(state.mario_jumping_wide == True, state.mario_view_direction == self.consts.MOVING_RIGHT),
                    lambda _: new_state_already_jumping,
                    lambda _: state,
                    operand=None
                ),
                operand=None
            )
        new_state = jumping_right(new_state)

        # basically the same thing only for jumping to left
        def jumping_left(state):
            new_state_start_jumping = state._replace(
                start_frame_when_mario_jumped = state.step_counter,
                mario_jumping_wide = True,
                mario_view_direction = self.consts.MOVING_LEFT,
                mario_x = (state.mario_x - self.consts.MARIO_JUMPING_HEIGHT).astype(jnp.float32),
                mario_y = (state.mario_y - self.consts.MARIO_MOVING_SPEED).astype(jnp.float32)
            )
            new_mario_x = jnp.round(self.bar_linear_equation(state.mario_stage, state.mario_y, state.level) - self.consts.MARIO_HIT_BOX_X) - 2
            new_state_already_jumping = state._replace(
                mario_y = (state.mario_y - self.consts.MARIO_MOVING_SPEED).astype(jnp.float32),
                mario_x = (new_mario_x - self.consts.MARIO_JUMPING_HEIGHT).astype(jnp.float32),
            )
            return jax.lax.cond(
                jnp.logical_and(action == Action.LEFTFIRE, jnp.logical_and(jnp.logical_and(jnp.logical_and(jnp.logical_and(state.mario_climbing == False, state.mario_jumping_wide == False), state.mario_jumping == False), state.block_jumping_and_climbing == False), state.mario_reached_goal == False)),
                lambda _: new_state_start_jumping,
                lambda _: jax.lax.cond(
                    jnp.logical_and(state.mario_jumping_wide == True, state.mario_view_direction == self.consts.MOVING_LEFT),
                    lambda _: new_state_already_jumping,
                    lambda _: state,
                    operand=None
                ),
                operand=None
            )
        new_state = jumping_left(new_state)

        # if mario can jump successfully over an enemy --> player gets 100 additional point, therefore one has to check Mario is jumping over an enemy
        def check_mario_jumping_over_enemy(state):
            new_state = state._replace(
                mario_jumping_over_enemy = True,
            )
            mario_jumping_over_enemy = False
            mario_x = state.mario_x + self.consts.MARIO_HIT_BOX_X

            # check if Mario jumping over an enemy by set on x position of mario again to ground and check for collision
            # Here for each barrel
            def check_collision_for_each_barrel(idx, hit):
                mario_got_hit = JaxDonkeyKong._collision_between_two_objects(mario_x, state.mario_y, self.consts.MARIO_HIT_BOX_X, self.consts.MARIO_HIT_BOX_Y,
                                                                        state.barrels.barrel_x[idx], state.barrels.barrel_y[idx], self.consts.BARREL_HIT_BOX_X, self.consts.BARREL_HIT_BOX_Y)
                
                return jax.lax.cond(
                    mario_got_hit,
                    lambda _: True,
                    lambda _: hit,
                    operand=None
                )
            mario_jumping_over_enemy = jax.lax.fori_loop(0, len(state.barrels.barrel_x), check_collision_for_each_barrel, False)

            # same for every fire
            def check_collision_for_each_fire(idx, hit):
                mario_got_hit = JaxDonkeyKong._collision_between_two_objects(mario_x, state.mario_y, self.consts.MARIO_HIT_BOX_X, self.consts.MARIO_HIT_BOX_Y,
                                                                        state.fires.fire_x[idx], state.fires.fire_y[idx], self.consts.FIRE_HIT_BOX_X, self.consts.FIRE_HIT_BOX_Y)
                
                return jax.lax.cond(
                    mario_got_hit,
                    lambda _: True,
                    lambda _: hit,
                    operand=None
                )
            mario_jumping_over_enemy = jax.lax.fori_loop(0, len(state.fires.fire_x), check_collision_for_each_fire, mario_jumping_over_enemy)

            mario_is_jumping = jnp.logical_or(state.mario_jumping, state.mario_jumping_wide)
            return jax.lax.cond(
                jnp.logical_and(mario_jumping_over_enemy, mario_is_jumping),
                lambda _: new_state,
                lambda _: state,
                operand=None
            )
        new_state = check_mario_jumping_over_enemy(new_state)

        # mario climbing ladder
        # precondition, mario is already climbing --> function for STARTING climbing below
        def mario_climbing(state):
            # normal climbing upwards
            new_state_climbing_upwards = jax.lax.cond(
                state.mario_climb_frame_counter % self.consts.MARIO_CLIMBING_ANIMATION_CHANGE_DURATION == 0,
                lambda _: jax.lax.cond(
                    state.mario_climb_sprite == self.consts.MARIO_CLIMB_SPRITE_0,
                    lambda _: state._replace(
                        mario_x=(state.mario_x - self.consts.MARIO_CLIMBING_SPEED).astype(jnp.float32),
                        mario_climb_frame_counter= state.mario_climb_frame_counter + 1,
                        mario_climb_sprite=self.consts.MARIO_CLIMB_SPRITE_1,
                        donkey_kong_sprite=self.consts.DONKEY_KONG_SPRITE_0,
                    ),
                    lambda _: state._replace(
                        mario_x=(state.mario_x - self.consts.MARIO_CLIMBING_SPEED).astype(jnp.float32),
                        mario_climb_frame_counter= state.mario_climb_frame_counter + 1,
                        mario_climb_sprite=self.consts.MARIO_CLIMB_SPRITE_0,
                        donkey_kong_sprite=self.consts.DONKEY_KONG_SPRITE_1,
                    ),
                    operand=None
                ),
                lambda _: state._replace(
                    mario_x=(state.mario_x - self.consts.MARIO_CLIMBING_SPEED).astype(jnp.float32),
                    mario_climb_frame_counter= state.mario_climb_frame_counter + 1,
                ),
                operand=None
            )

            # check if mario finished climbing / reached the end of a ladder
            reached_top = new_state_climbing_upwards.mario_x <= jnp.round(self.bar_linear_equation(new_state_climbing_upwards.mario_stage + 1, new_state_climbing_upwards.mario_y, state.level) - self.consts.MARIO_HIT_BOX_X).astype(int)
            new_state_climbing_upwards = jax.lax.cond(
                reached_top,
                lambda _: state._replace(
                    mario_x = (state.mario_x - 2).astype(jnp.float32),
                    mario_climb_frame_counter= 0,
                    mario_view_direction = self.consts.MOVING_RIGHT,
                    mario_climbing = False,
                    mario_stage = state.mario_stage + 1,
                    mario_walk_frame_counter = 0,
                    mario_walk_sprite = self.consts.MARIO_WALK_SPRITE_0,
                    mario_climbing_delay = True,
                ),
                lambda _: new_state_climbing_upwards,
                operand=None
            )

            # normal climbing downwards
            new_state_climbing_downwards = jax.lax.cond(
                state.mario_climb_frame_counter % self.consts.MARIO_CLIMBING_ANIMATION_CHANGE_DURATION == 0,
                lambda _: jax.lax.cond(
                    state.mario_climb_sprite == self.consts.MARIO_CLIMB_SPRITE_0,
                    lambda _: state._replace(
                        mario_x=(state.mario_x + self.consts.MARIO_CLIMBING_SPEED).astype(jnp.float32),
                        mario_climb_frame_counter= state.mario_climb_frame_counter + 1,
                        mario_climb_sprite=self.consts.MARIO_CLIMB_SPRITE_1,
                        donkey_kong_sprite=self.consts.DONKEY_KONG_SPRITE_0,
                    ),
                    lambda _: state._replace(
                        mario_x=(state.mario_x + self.consts.MARIO_CLIMBING_SPEED).astype(jnp.float32),
                        mario_climb_frame_counter= state.mario_climb_frame_counter + 1,
                        mario_climb_sprite=self.consts.MARIO_CLIMB_SPRITE_0,
                        donkey_kong_sprite=self.consts.DONKEY_KONG_SPRITE_1,
                    ),
                    operand=None
                ),
                lambda _: state._replace(
                    mario_x=(state.mario_x + self.consts.MARIO_CLIMBING_SPEED).astype(jnp.float32),
                    mario_climb_frame_counter= state.mario_climb_frame_counter + 1,
                ),
                operand=None
            )

            # check if mario finished climbing / reached the start of a ladder
            reached_bottom = new_state_climbing_downwards.mario_x >= jnp.round(self.bar_linear_equation(new_state_climbing_downwards.mario_stage, new_state_climbing_downwards.mario_y, state.level) - self.consts.MARIO_HIT_BOX_X).astype(int)
            new_state_climbing_downwards = jax.lax.cond(
                reached_bottom,
                lambda _: state._replace(
                    mario_x = (state.mario_x - 2).astype(jnp.float32),
                    mario_climb_frame_counter= 0,
                    mario_view_direction = self.consts.MOVING_RIGHT,
                    mario_climbing = False,
                    mario_walk_frame_counter = 0,
                    mario_walk_sprite = self.consts.MARIO_WALK_SPRITE_0,
                    mario_climbing_delay = True,
                ),
                lambda _: new_state_climbing_downwards,
                operand=None
            )

            return jax.lax.cond(
                jnp.logical_and(jnp.logical_and(jnp.logical_and(jnp.logical_and(jnp.logical_and(state.mario_climbing == True, state.mario_jumping == False), state.mario_jumping_wide == False), action == Action.UP), state.block_jumping_and_climbing == False), state.mario_reached_goal == False),
                lambda _: new_state_climbing_upwards,
                lambda _: jax.lax.cond(
                    jnp.logical_and(jnp.logical_and(jnp.logical_and(jnp.logical_and(jnp.logical_and(state.mario_climbing == True, state.mario_jumping == False), state.mario_jumping_wide == False), action == Action.DOWN), state.block_jumping_and_climbing == False), state.mario_reached_goal == False),
                    lambda _: new_state_climbing_downwards,
                    lambda _: state,
                    operand=None
                ),
                operand=None
            )
        new_state = mario_climbing(new_state)

        # mario starts climbs ladder
        def mario_starts_climbing(state):
            new_state_climbing_upwards = state._replace(
                mario_view_direction=self.consts.MOVING_UP,
                mario_x=(state.mario_x + 1).astype(jnp.float32),
                mario_climbing=True,
                mario_climb_frame_counter=0,
            )
            new_state_climbing_downwards = state._replace(
                mario_view_direction=self.consts.MOVING_UP,
                mario_stage = state.mario_stage - 1,
                mario_x=(state.mario_x + 3).astype(jnp.float32),
                mario_climbing=True,
                mario_climb_frame_counter=0,
            )
            ladders = state.ladders # be careful, ladder is not the actual ladder positions but where barrel interact with the ladders

            # there ladders which can not be climbed, so check for it
            def look_for_valid_ladder_to_climb(i, value):
                mario_can_climb = value[0]
                mario_stage = value[1]
                current_ladder_climbable = (
                    (mario_stage == ladders.stage[i])
                    & (jnp.int32(jnp.round(state.mario_y) + 1) <= ladders.start_y[i]+1)
                    & ((jnp.int32(jnp.round(state.mario_y)) + self.consts.MARIO_HIT_BOX_Y - 1) >= (ladders.start_y[i]+1 + self.consts.LADDER_WIDTH))
                    & (ladders.climbable[i])
                )

                return jax.lax.cond(
                    mario_can_climb,
                    lambda _: (True, mario_stage),
                    lambda _: (current_ladder_climbable, mario_stage),
                    operand=None
                )
            mario_can_climb_upwards = jax.lax.fori_loop(0, len(ladders.stage), look_for_valid_ladder_to_climb, (False, state.mario_stage))[0]  
            mario_can_climb_downwards = jax.lax.fori_loop(0, len(ladders.stage), look_for_valid_ladder_to_climb, (False, state.mario_stage - 1))[0]        
            
            return jax.lax.cond(
                jnp.logical_and(jnp.logical_and(jnp.logical_and(jnp.logical_and(jnp.logical_and(jnp.logical_and(jnp.logical_and(mario_can_climb_upwards, state.mario_climbing == False), state.mario_jumping == False), state.mario_jumping_wide == False), action == Action.UP), state.block_jumping_and_climbing == False), state.mario_reached_goal == False), state.mario_climbing_delay == False),
                lambda _: new_state_climbing_upwards,
                lambda _: jax.lax.cond(
                    jnp.logical_and(jnp.logical_and(jnp.logical_and(jnp.logical_and(jnp.logical_and(jnp.logical_and(jnp.logical_and(mario_can_climb_downwards, state.mario_climbing == False), state.mario_jumping == False), state.mario_jumping_wide == False), action == Action.DOWN), state.block_jumping_and_climbing == False), state.mario_reached_goal == False), state.mario_climbing_delay == False),
                    lambda _: new_state_climbing_downwards,
                    lambda _: state,
                    operand=None
                ),
                operand=None
            )
        new_state = mario_starts_climbing(new_state)

        # change mario position in x direction if Action.right is chosen
        def mario_walking_to_right(state):
            last_mario_move_was_not_moving_to_right = state.mario_view_direction != self.consts.MOVING_RIGHT
            new_mario_x = jnp.round(self.bar_linear_equation(state.mario_stage, state.mario_y, state.level) - self.consts.MARIO_HIT_BOX_X).astype(jnp.float32) - 2
            
            # only difference here between two state
            # if last_mario_move_was_not_moving_to_right=True mario_walk_frame_counter = 0 --> this is needed to animate Mario sprites
            new_state = jax.lax.cond(
                last_mario_move_was_not_moving_to_right,
                lambda _: state._replace(
                    mario_x = (new_mario_x).astype(jnp.float32),
                    mario_y=(state.mario_y + self.consts.MARIO_MOVING_SPEED).astype(jnp.float32),
                    mario_view_direction=self.consts.MOVING_RIGHT,
                    mario_walk_frame_counter=0,
                    mario_climbing_delay = False,
                ),
                lambda _:state._replace(
                    mario_x = (new_mario_x).astype(jnp.float32),
                    mario_y=(state.mario_y + self.consts.MARIO_MOVING_SPEED).astype(jnp.float32),
                    mario_view_direction=self.consts.MOVING_RIGHT,
                    mario_walk_frame_counter=state.mario_walk_frame_counter + 1,
                    mario_climbing_delay = False,
                ),
                operand=None
            )

            # there are 3 Mario walking sprites, it has be choose one properly
            next_mario_walk_sprite = jax.lax.cond(
                state.mario_walk_sprite == self.consts.MARIO_WALK_SPRITE_0,
                lambda _: self.consts.MARIO_WALK_SPRITE_1,
                lambda _: jax.lax.cond(
                    state.mario_walk_sprite == self.consts.MARIO_WALK_SPRITE_1,
                    lambda _: self.consts.MARIO_WALK_SPRITE_2,
                    lambda _: jax.lax.cond(
                        state.mario_walk_sprite == self.consts.MARIO_WALK_SPRITE_2,
                        lambda _: self.consts.MARIO_WALK_SPRITE_3,
                        lambda _: self.consts.MARIO_WALK_SPRITE_0,
                        operand=None
                    ),
                    operand=None
                ),
                operand=None
            )
            change_sprite = state.mario_walk_frame_counter == self.consts.MARIO_WALKING_ANIMATION_CHANGE_DURATION
            new_state = jax.lax.cond(
                change_sprite == True,
                lambda _: new_state._replace(
                    mario_walk_frame_counter = 0,
                    mario_walk_sprite = next_mario_walk_sprite,
                ),
                lambda _: new_state,
                operand=None
            )

            # change donkey kong sprite
            new_state = new_state._replace(
                donkey_kong_sprite = self.consts.DONKEY_KONG_SPRITE_0,
            )
            return jax.lax.cond(
                jnp.logical_and(jnp.logical_and(jnp.logical_and(jnp.logical_and(state.mario_climbing == False, state.mario_jumping == False), state.mario_jumping_wide == False), action == Action.RIGHT), state.mario_reached_goal == False),
                lambda _: new_state,
                lambda _: state,
                operand=None
            )
        new_state = mario_walking_to_right(new_state)

        # similar function as mario_walking_to_right
        def mario_walking_to_left(state):
            last_mario_move_was_not_moving_to_left = state.mario_view_direction != self.consts.MOVING_LEFT
            new_mario_x = jnp.round(self.bar_linear_equation(state.mario_stage, state.mario_y, state.level) - self.consts.MARIO_HIT_BOX_X) - 2
            new_state = jax.lax.cond(
                last_mario_move_was_not_moving_to_left,
                lambda _:  state._replace(
                    mario_x = new_mario_x.astype(jnp.float32),
                    mario_y=(state.mario_y - self.consts.MARIO_MOVING_SPEED).astype(jnp.float32),
                    mario_view_direction=self.consts.MOVING_LEFT,
                    mario_walk_frame_counter=0,
                    mario_climbing_delay = False,
                ),
                lambda _:state._replace(
                    mario_x = new_mario_x.astype(jnp.float32),
                    mario_y=(state.mario_y - self.consts.MARIO_MOVING_SPEED).astype(jnp.float32),
                    mario_view_direction=self.consts.MOVING_LEFT,
                    mario_walk_frame_counter=state.mario_walk_frame_counter + 1,
                    mario_climbing_delay = False,
                ),
                operand=None
            )

            next_mario_walk_sprite = jax.lax.cond(
                state.mario_walk_sprite == self.consts.MARIO_WALK_SPRITE_0,
                lambda _: self.consts.MARIO_WALK_SPRITE_1,
                lambda _: jax.lax.cond(
                    state.mario_walk_sprite == self.consts.MARIO_WALK_SPRITE_1,
                    lambda _: self.consts.MARIO_WALK_SPRITE_2,
                    lambda _: jax.lax.cond(
                        state.mario_walk_sprite == self.consts.MARIO_WALK_SPRITE_2,
                        lambda _: self.consts.MARIO_WALK_SPRITE_3,
                        lambda _: self.consts.MARIO_WALK_SPRITE_0,
                        operand=None
                    ),
                    operand=None
                ),
                operand=None
            )
            change_sprite = state.mario_walk_frame_counter == self.consts.MARIO_WALKING_ANIMATION_CHANGE_DURATION
            new_state = jax.lax.cond(
                change_sprite == True,
                lambda _: new_state._replace(
                    mario_walk_frame_counter = 0,
                    mario_walk_sprite = next_mario_walk_sprite,
                ),
                lambda _: new_state,
                operand=None
            )

            # change donkey kong sprite
            new_state = new_state._replace(
                donkey_kong_sprite = self.consts.DONKEY_KONG_SPRITE_1,
            )
            return jax.lax.cond(
                jnp.logical_and(jnp.logical_and(jnp.logical_and(jnp.logical_and(state.mario_climbing == False, state.mario_jumping == False), state.mario_jumping_wide == False), action == Action.LEFT), state.mario_reached_goal == False),
                lambda _: new_state,
                lambda _: state,
                operand=None
            )
        new_state = mario_walking_to_left(new_state)

        # every level and each stage have a minimum and maximum position where Mario can stand --> invisble Walls
        # set Mario position to minimum or maximum if he reaches the invisible wall
        def mario_walking_into_invisible_wall(state):
            def look_for_invisible_wall_of_given_stage(i, hit):
                wall_hit = True
                wall_hit &= state.invisible_wall_each_stage.stage[i] == state.mario_stage
                mario_reaches_invisible_wall = jnp.logical_or(state.invisible_wall_each_stage.right_end[i] < state.mario_y, state.invisible_wall_each_stage.left_end[i] > state.mario_y) 
                wall_hit &= mario_reaches_invisible_wall
                return jax.lax.cond(
                    wall_hit,
                    lambda _: True,
                    lambda _: hit,
                    operand=None
                )
            hit = jax.lax.fori_loop(0, len(state.invisible_wall_each_stage.stage), look_for_invisible_wall_of_given_stage, False)

            new_state_right_wall = state._replace(
                mario_walk_sprite = self.consts.MARIO_WALK_SPRITE_0,
                mario_y = (state.mario_y - self.consts.MARIO_MOVING_SPEED).astype(jnp.float32)
            )

            new_state_left_wall = state._replace(
                mario_walk_sprite = self.consts.MARIO_WALK_SPRITE_0,
                mario_y = (state.mario_y + self.consts.MARIO_MOVING_SPEED).astype(jnp.float32)
            )

            return jax.lax.cond(
                hit,
                lambda _: jax.lax.cond(
                    state.mario_view_direction == self.consts.MOVING_RIGHT,
                    lambda _: new_state_right_wall,
                    lambda _: new_state_left_wall,
                    operand=None
                ),
                lambda _: state,
                operand=None
            )
        new_state = mario_walking_into_invisible_wall(new_state)


        # Check if mario is hitted by barrel/fire
        def mario_enemy_collision(state):
            # checking for barrels
            def check_collision_for_each_barrel(idx, state):
                mario_got_hit = JaxDonkeyKong._collision_between_two_objects(state.mario_x, state.mario_y, self.consts.MARIO_HIT_BOX_X, self.consts.MARIO_HIT_BOX_Y,
                                                                        state.barrels.barrel_x[idx], state.barrels.barrel_y[idx], self.consts.BARREL_HIT_BOX_X, self.consts.BARREL_HIT_BOX_Y)
                mario_got_hit &= jnp.logical_not(state.barrels.reached_the_end[idx])
                mario_got_hit &= jnp.logical_not(jnp.logical_or(state.mario_jumping, state.mario_jumping_wide))
                game_freeze_start = jax.lax.cond(
                    state.mario_got_hit,
                    lambda _: state.game_freeze_start,
                    lambda _: state.step_counter,
                    operand=None
                )
                new_state = state._replace(
                    mario_got_hit = True,
                    game_freeze_start = game_freeze_start,
                )
                return jax.lax.cond(
                    mario_got_hit,
                    lambda _: new_state,
                    lambda _: state,
                    operand=None
                )
            # checking for fires
            def check_collision_for_each_fire(idx, state):
                mario_got_hit = JaxDonkeyKong._collision_between_two_objects(state.mario_x, state.mario_y, self.consts.MARIO_HIT_BOX_X, self.consts.MARIO_HIT_BOX_Y,
                                                                        state.fires.fire_x[idx], state.fires.fire_y[idx], self.consts.FIRE_HIT_BOX_X, self.consts.FIRE_HIT_BOX_Y)
                mario_got_hit &= jnp.logical_not(state.fires.destroyed[idx])
                mario_got_hit &= jnp.logical_not(jnp.logical_or(state.mario_jumping, state.mario_jumping_wide))
                game_freeze_start = jax.lax.cond(
                    state.mario_got_hit,
                    lambda _: state.game_freeze_start,
                    lambda _: state.step_counter,
                    operand=None
                )
                new_state = state._replace(
                    mario_got_hit = True,
                    game_freeze_start = game_freeze_start,
                )
                return jax.lax.cond(
                    mario_got_hit,
                    lambda _: new_state,
                    lambda _: state,
                    operand=None
                )
            new_state = jax.lax.cond(
                state.level == 1,
                lambda _: jax.lax.fori_loop(0, len(state.barrels.barrel_x), check_collision_for_each_barrel, state),
                lambda _: jax.lax.fori_loop(0, len(state.fires.fire_x), check_collision_for_each_fire, state),
                operand=None
            )
            return new_state
        new_state = mario_enemy_collision(new_state)

        # reset jumping after a certain time
        def reset_jumping(state):
            # new x position of mario because the function implements the "landing" of mario on to the ground
            new_mario_x = jnp.round(self.bar_linear_equation(state.mario_stage, state.mario_y, state.level) - self.consts.MARIO_HIT_BOX_X) - 2
            new_state = state._replace(
                mario_jumping = False,
                mario_jumping_wide = False,
                mario_x = new_mario_x.astype(jnp.float32),
                mario_climbing_delay = False,
            )

            # update the score here if mario jumped over an enemy
            # if state.mario_jumping_over_enemy is set True --> mario jumped over an enemy, gives the player additional 100 score
            # CAREFUL: one step before, it was checked if there is a collision between mario and enemy
            game_score = jax.lax.cond(
                jnp.logical_and(state.mario_jumping_over_enemy == True, state.mario_got_hit == False),
                lambda _: state.game_score + self.consts.SCORE_FOR_JUMPING_OVER_ENEMY,
                lambda _: state.game_score,
                operand=None
            )
            new_state = new_state._replace(
                game_score=game_score,
                mario_jumping_over_enemy=False, # reset the variable 
            )

            return jax.lax.cond(
                jnp.logical_and(state.step_counter - state.start_frame_when_mario_jumped >= self.consts.MARIO_JUMPING_FRAME_DURATION, jnp.logical_or(state.mario_jumping == True, state.mario_jumping_wide == True)),
                lambda _: new_state,
                lambda _: state,
                operand=None
            )
        new_state = reset_jumping(new_state)

        # check if mario reached the goal -- only for level 1 climbing the ladder, for level 2 there is an another goal to be reached
        def mario_reached_goal(state):
            reached_goal = state.mario_x <= self.consts.LEVEL_1_GOAL_X
            reached_goal &= state.mario_climbing
            reached_goal &= jnp.logical_not(state.mario_reached_goal) # false if already reached the goal
            new_state = state._replace(
                game_freeze_start = state.step_counter,
                mario_reached_goal = True,
            )
            return jax.lax.cond(
                reached_goal,
                lambda _: new_state,
                lambda _: state,
                operand=None
            )
        new_state = mario_reached_goal(new_state)

        # after mario reached the goal or hit by an enemy --> reset the level, calculate score
        def reset_round_after_success_or_collision(state):
            # first reset the game state
            # and change only some variable to implement either a succefull or Mario lose state
            ladders = self.init_ladders_for_level(level=1)
            invisible_wall_each_stage = self.init_invisible_wall_for_level(level=1)
            new_state = DonkeyKongState(
                game_started = False,
                level = 1,
                step_counter=jnp.array(1).astype(jnp.int32),
                game_timer = 1,
                game_remaining_time = 5000,
                frames_since_last_barrel_spawn=jnp.array(0).astype(jnp.int32),

                mario_climbing_delay = False,

                game_score = state.game_score,
                game_round = state.game_round,

                mario_x=self.consts.LEVEL_1_MARIO_START_X,
                mario_y=self.consts.LEVEL_1_MARIO_START_Y,
                mario_jumping=False,
                mario_jumping_wide=False,
                mario_jumping_over_enemy=False,
                mario_climbing=False,
                start_frame_when_mario_jumped=-1,
                mario_view_direction=self.consts.MOVING_RIGHT,
                mario_walk_frame_counter=0,
                mario_climb_frame_counter=0,
                mario_walk_sprite=self.consts.MARIO_WALK_SPRITE_0,
                mario_climb_sprite=self.consts.MARIO_CLIMB_SPRITE_0,
                mario_stage=1,
                mario_life_counter = 2,
                mario_got_hit = False,
                game_freeze_start = -1,
                mario_reached_goal = False,

                donkey_kong_sprite = self.consts.DONKEY_KONG_SPRITE_0,

                barrels = BarrelPosition(
                    barrel_x = jnp.array([-1, -1, -1, -1]).astype(jnp.int32),
                    barrel_y = jnp.array([-1, -1, -1, -1]).astype(jnp.int32), 
                    sprite = jnp.array([self.consts.BARREL_SPRITE_RIGHT, self.consts.BARREL_SPRITE_RIGHT, self.consts.BARREL_SPRITE_RIGHT, self.consts.BARREL_SPRITE_RIGHT]).astype(jnp.int32),
                    moving_direction = jnp.array([self.consts.MOVING_RIGHT, self.consts.MOVING_RIGHT, self.consts.MOVING_RIGHT, self.consts.MOVING_RIGHT]).astype(jnp.int32),
                    stage = jnp.array([6, 6, 6, 6]).astype(jnp.int32),
                    reached_the_end=jnp.array([True, True, True, True]).astype(bool)
                ),
                barrels_change_direction_prob = state.barrels_change_direction_prob,

                fires = FirePosition(
                    fire_x = jnp.array([-1., -1., -1., -1.]).astype(jnp.float32),
                    fire_y = jnp.array([-1., -1., -1., -1.]).astype(jnp.float32),
                    moving_direction = jnp.array([self.consts.MOVING_RIGHT, self.consts.MOVING_LEFT, self.consts.MOVING_RIGHT, self.consts.MOVING_LEFT]).astype(jnp.int32),
                    stage = jnp.array([5, 4, 3, 2]).astype(jnp.int32),
                    destroyed=jnp.array([True, True, True, True]).astype(bool),
                    change_direction_prob=state.fires.change_direction_prob,
                ),

                traps = TrapPosition(
                    trap_x = jnp.array([self.consts.TRAP_FLOOR_5_X, self.consts.TRAP_FLOOR_4_X, self.consts.TRAP_FLOOR_3_X, self.consts.TRAP_FLOOR_2_X, self.consts.TRAP_FLOOR_5_X, self.consts.TRAP_FLOOR_4_X, self.consts.TRAP_FLOOR_3_X, self.consts.TRAP_FLOOR_2_X]).astype(jnp.int32),
                    trap_y = jnp.array([self.consts.TRAP_LEFT_Y, self.consts.TRAP_LEFT_Y, self.consts.TRAP_LEFT_Y, self.consts.TRAP_LEFT_Y, self.consts.TRAP_RIGHT_Y, self.consts.TRAP_RIGHT_Y, self.consts.TRAP_RIGHT_Y, self.consts.TRAP_RIGHT_Y]).astype(jnp.int32),
                    stage = jnp.array([5, 4, 3, 2, 5, 4, 3, 2]).astype(jnp.int32),
                    triggered = jnp.array([False, False, False, False, False, False, False, False]).astype(bool),
                    fall_protection = jnp.array([-1, -1, -1, -1, -1, -1, -1, -1]).astype(jnp.int32), # -1 indicates no protection, otherwise safe the view direction of mario; e.g. : if mario triggers a trap looking right, he is protected while he is moving right. But if he turns left, protection is instantly gone
                ),

                ladders=ladders,
                invisible_wall_each_stage=invisible_wall_each_stage,
                random_key = state.random_key,

                hammer_x = self.consts.LEVEL_1_HAMMER_X,
                hammer_y = self.consts.LEVEL_1_HAMMER_Y,
                hammer_can_hit = False,
                hammer_taken = False,
                hammer_carry_time = 0,
                block_jumping_and_climbing = False,
                hammer_usage_expired = False,
            )

            ladder = self.init_ladders_for_level(state.level)
            invisible_wall = self.init_invisible_wall_for_level(state.level)
            mario_x, mario_y, hammer_x, hammer_y = jax.lax.cond(
                state.level == 1,
                lambda _: (self.consts.LEVEL_1_MARIO_START_X, self.consts.LEVEL_1_MARIO_START_Y, self.consts.LEVEL_1_HAMMER_X, self.consts.LEVEL_1_HAMMER_Y),
                lambda _: (self.consts.LEVEL_2_MARIO_START_X, self.consts.LEVEL_2_MARIO_START_Y, self.consts.LEVEL_2_HAMMER_X, self.consts.LEVEL_2_HAMMER_Y),
                operand=None
            )
            # new_state_life_loose --> Mario got hit by an enemy, Mario's life counter decrement
            new_state_life_loose = new_state._replace(
                mario_life_counter = state.mario_life_counter - 1,
                level = state.level,
                ladders = ladder,
                invisible_wall_each_stage = invisible_wall,
                mario_x = mario_x.astype(jnp.float32),
                mario_y = mario_y.astype(jnp.float32),
                hammer_x = hammer_x,
                hammer_y = hammer_y,
            )
            level = jax.lax.cond(
                state.level == 1,
                lambda _: 2,
                lambda _: 1,
                operand=None
            )


            ladder = self.init_ladders_for_level(level)
            invisible_wall = self.init_invisible_wall_for_level(level)
            mario_x, mario_y, hammer_x, hammer_y = jax.lax.cond(
                level == 1,
                lambda _: (self.consts.LEVEL_1_MARIO_START_X, self.consts.LEVEL_1_MARIO_START_Y, self.consts.LEVEL_1_HAMMER_X, self.consts.LEVEL_1_HAMMER_Y),
                lambda _: (self.consts.LEVEL_2_MARIO_START_X, self.consts.LEVEL_2_MARIO_START_Y, self.consts.LEVEL_2_HAMMER_X, self.consts.LEVEL_2_HAMMER_Y),
                operand=None
            )
            game_score = jax.lax.cond(
                state.level == 1,
                lambda _: new_state.game_score + self.consts.SCORE_FOR_REACHING_GOAL_LEVEL_1,
                lambda _: new_state.game_score + self.consts.SCORE_FOR_REACHING_GOAL_LEVEL_2,
                operand=None
            )
            # new_state_clear_level --> Mario reached the next level state
            def update_enemy_probability(game_round):
                # change the prob fpr barrel falling the ladder correspoding to the given game round
                barrels_change_direction_prob = jax.lax.cond(
                    game_round == 2,
                    lambda _: self.consts.BASE_PROBABILITY_BARREL_ROLLING_A_LADDER_DOWN_ROUND_2,
                    lambda _: jax.lax.cond(
                        game_round == 3,
                        lambda _: self.consts.BASE_PROBABILITY_BARREL_ROLLING_A_LADDER_DOWN_ROUND_3,
                        lambda _: jax.lax.cond(
                            game_round >= 4,
                            lambda _: self.consts.BASE_PROBABILITY_BARREL_ROLLING_A_LADDER_DOWN_ROUND_4,
                            lambda _: self.consts.BASE_PROBABILITY_BARREL_ROLLING_A_LADDER_DOWN_ROUND_1,
                            operand=None
                        ),
                        operand=None
                    ),
                    operand=None
                )

                fires_change_direction_prob = jax.lax.cond(
                    game_round == 1,
                    lambda _: jnp.array([self.consts.STAGE_5_FIRE_CHANGING_DIRECTION_DEFAULT_PROB, self.consts.STAGE_4_FIRE_CHANGING_DIRECTION_DEFAULT_PROB, self.consts.STAGE_3_FIRE_CHANGING_DIRECTION_DEFAULT_PROB, self.consts.STAGE_2_FIRE_CHANGING_DIRECTION_DEFAULT_PROB]),
                    lambda _: jnp.array([self.consts.STAGE_5_FIRE_CHANGING_DIRECTION_DEFAULT_PROB + self.consts.STAGE_5_FIRE_CHANGING_DIRECTION_INC_PROB * game_round, self.consts.STAGE_4_FIRE_CHANGING_DIRECTION_DEFAULT_PROB + self.consts.STAGE_4_FIRE_CHANGING_DIRECTION_INC_PROB * game_round, self.consts.STAGE_3_FIRE_CHANGING_DIRECTION_DEFAULT_PROB + self.consts.STAGE_3_FIRE_CHANGING_DIRECTION_INC_PROB * game_round, self.consts.STAGE_2_FIRE_CHANGING_DIRECTION_DEFAULT_PROB + self.consts.STAGE_2_FIRE_CHANGING_DIRECTION_INC_PROB * game_round]),
                    operand=None
                )

                return barrels_change_direction_prob, fires_change_direction_prob

            game_round = jax.lax.cond(
                state.level == 2,
                lambda _: state.game_round + 1,
                lambda _: state.game_round,
                operand=None
            )
            barrels_change_direction_prob, fires_change_direction_prob = update_enemy_probability(game_round)
            fires = new_state.fires._replace(change_direction_prob = fires_change_direction_prob)
            new_state_clear_level = new_state._replace(
                level=level,
                ladders=ladder,
                invisible_wall_each_stage=invisible_wall,
                mario_x = mario_x.astype(jnp.float32),
                mario_y = mario_y.astype(jnp.float32),
                hammer_x = hammer_x,
                hammer_y = hammer_y,
                game_score = game_score,
                game_round = game_round,
                mario_life_counter = state.mario_life_counter,
                barrels_change_direction_prob=barrels_change_direction_prob,
                fires = fires,
            )            

            game_freeze_over = self.consts.GAME_FREEZE_DURATION < (state.step_counter - state.game_freeze_start)

            return jax.lax.cond(
                jnp.logical_and(game_freeze_over, state.mario_got_hit),
                lambda _: (new_state_life_loose, True),
                lambda _: jax.lax.cond(
                    jnp.logical_and(game_freeze_over, state.mario_reached_goal == True),
                    lambda _: (new_state_clear_level, True),
                    lambda _: (state, False),
                    operand=None
                ),
                operand=None
            )    
        new_state, game_can_be_resetted = reset_round_after_success_or_collision(new_state)    

        return jax.lax.cond(
            jnp.logical_and(state.mario_got_hit, game_can_be_resetted == False),
            lambda _: state,
            lambda _: jax.lax.cond(
                jnp.logical_and(state.mario_reached_goal, game_can_be_resetted == False),
                lambda _: state,
                lambda _: new_state,
                operand=None
            ),
            operand=None
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def _hammer_step(self, state, action: chex.Array):
        # calculate the position of the hammer if active
        def calculate_hammer_pos_relative_to_mario(state):
            new_state_mario_views_right_swing = state._replace(
                hammer_x = self.consts.LEVEL_1_HAMMER_SWING_X,
                hammer_y = (state.mario_y + self.consts.MARIO_HIT_BOX_Y + 1).astype(jnp.int32),
            )
            new_state_mario_views_right_no_swing = state._replace(
                hammer_x = self.consts.LEVEL_1_HAMMER_X,
                hammer_y = (state.mario_y + self.consts.MARIO_HIT_BOX_Y + 1).astype(jnp.int32),
            )

            new_state_mario_jumping_right_swing = state._replace(
                hammer_x = self.consts.LEVEL_1_HAMMER_SWING_X,
                hammer_y = (state.mario_y + self.consts.MARIO_HIT_BOX_Y + 1).astype(jnp.int32),
            )
            new_state_mario_jumping_right_no_swing = state._replace(
                hammer_x = self.consts.LEVEL_1_HAMMER_X,
                hammer_y = (state.mario_y + self.consts.MARIO_HIT_BOX_Y + 1).astype(jnp.int32),
            )

            new_state_mario_views_left_swing = state._replace(
                hammer_x = self.consts.LEVEL_1_HAMMER_SWING_X,
                hammer_y = (state.mario_y - self.consts.HAMMER_SWING_HIT_BOX_Y).astype(jnp.int32),
            )
            new_state_mario_views_left_no_swing = state._replace(
                hammer_x = self.consts.LEVEL_1_HAMMER_X,
                hammer_y = (state.mario_y - self.consts.HAMMER_HIT_BOX_Y).astype(jnp.int32),
            )

            new_state_mario_jumping_left_swing = state._replace(
                hammer_x = self.consts.LEVEL_1_HAMMER_SWING_X,
                hammer_y = (state.mario_y - self.consts.HAMMER_SWING_HIT_BOX_Y).astype(jnp.int32),
            )
            new_state_mario_jumping_left_no_swing = state._replace(
                hammer_x = self.consts.LEVEL_1_HAMMER_X,
                hammer_y = (state.mario_y - self.consts.HAMMER_HIT_BOX_Y).astype(jnp.int32),
            )


            mario_is_jumping = jnp.logical_or(state.mario_jumping, state.mario_jumping_wide)

            new_state = state
            new_state = jax.lax.cond(
                jnp.logical_and(new_state.mario_view_direction == self.consts.MOVING_RIGHT, jnp.logical_and(mario_is_jumping, new_state.hammer_can_hit)),
                lambda _: new_state_mario_jumping_right_swing,
                lambda _: new_state,
                operand=None
            )
            new_state = jax.lax.cond(
                jnp.logical_and(new_state.mario_view_direction == self.consts.MOVING_RIGHT, jnp.logical_and(mario_is_jumping, jnp.logical_not(new_state.hammer_can_hit))),
                lambda _: new_state_mario_jumping_right_no_swing,
                lambda _: new_state,
                operand=None
            )
            new_state = jax.lax.cond(
                jnp.logical_and(new_state.mario_view_direction == self.consts.MOVING_RIGHT, jnp.logical_and(jnp.logical_not(mario_is_jumping), new_state.hammer_can_hit)),
                lambda _: new_state_mario_views_right_swing,
                lambda _: new_state,
                operand=None
            )
            new_state = jax.lax.cond(
                jnp.logical_and(new_state.mario_view_direction == self.consts.MOVING_RIGHT, jnp.logical_and(jnp.logical_not(mario_is_jumping), jnp.logical_not(new_state.hammer_can_hit))),
                lambda _: new_state_mario_views_right_no_swing,
                lambda _: new_state,
                operand=None
            )
            new_state = jax.lax.cond(
                jnp.logical_and(new_state.mario_view_direction == self.consts.MOVING_LEFT, jnp.logical_and(mario_is_jumping, new_state.hammer_can_hit)),
                lambda _: new_state_mario_jumping_left_swing,
                lambda _: new_state,
                operand=None
            )
            new_state = jax.lax.cond(
                jnp.logical_and(new_state.mario_view_direction == self.consts.MOVING_LEFT, jnp.logical_and(mario_is_jumping, jnp.logical_not(new_state.hammer_can_hit))),
                lambda _: new_state_mario_jumping_left_no_swing,
                lambda _: new_state,
                operand=None
            )
            new_state = jax.lax.cond(
                jnp.logical_and(new_state.mario_view_direction == self.consts.MOVING_LEFT, jnp.logical_and(jnp.logical_not(mario_is_jumping), new_state.hammer_can_hit)),
                lambda _: new_state_mario_views_left_swing,
                lambda _: new_state,
                operand=None
            )
            new_state = jax.lax.cond(
                jnp.logical_and(new_state.mario_view_direction == self.consts.MOVING_LEFT, jnp.logical_and(jnp.logical_not(mario_is_jumping), jnp.logical_not(new_state.hammer_can_hit))),
                lambda _: new_state_mario_views_left_no_swing,
                lambda _: new_state,
                operand=None
            )
            return new_state

        # mario can take the hammer by jumping and if they collide
        def hammer_is_taken_by_mario(state):
            new_state = state._replace(
                hammer_taken = True,
                hammer_can_hit = True,
                hammer_carry_time = -1,
                block_jumping_and_climbing = True,
            )

            # if there is a collision between the hammer and Mario -> Hammer can be taken
            collision_mario_hammer = JaxDonkeyKong._collision_between_two_objects(state.mario_x, state.mario_y, self.consts.MARIO_HIT_BOX_X, self.consts.MARIO_HIT_BOX_Y, 
                                                                                state.hammer_x, state.hammer_y, self.consts.HAMMER_HIT_BOX_X, self.consts.HAMMER_HIT_BOX_Y)

            new_state = calculate_hammer_pos_relative_to_mario(new_state)

            # Original game: Mario can only take the hammer if he reached the hammer by jumping --> Jumping = True
            mario_is_jumping = jnp.logical_or(state.mario_jumping, state.mario_jumping_wide)
            return jax.lax.cond(
                jnp.logical_and(state.hammer_taken == False, jnp.logical_and(collision_mario_hammer, mario_is_jumping)),
                lambda _: new_state,
                lambda _: state,
                operand=None
            )
        new_state = hammer_is_taken_by_mario(state)

        # Hammer swings have swing time/duration
        def hammer_swing(state):
            # Hammer has Swing duration to change sprites and only during the swing, hammer is able to destroy barrel
            hammer_carry_time, hammer_can_hit = jax.lax.cond(
                (state.hammer_carry_time + 1) % self.consts.HAMMER_SWING_DURATION == 0,
                lambda _: (state.hammer_carry_time + 1, jnp.logical_not(state.hammer_can_hit)),
                lambda _: (state.hammer_carry_time + 1, state.hammer_can_hit),
                operand=None
            )         
            # Hammer lasts for self.consts.HAMMER_MAX_CARRY_DURATION
            hammer_can_hit, block_jumping_and_climbing, hammer_usage_expired = jax.lax.cond(
                state.hammer_carry_time >= self.consts.HAMMER_MAX_CARRY_DURATION,
                lambda _: (False, False, True),
                lambda _: (hammer_can_hit, state.block_jumping_and_climbing, state.hammer_usage_expired),
                operand=None
            )
            new_state = state._replace(
                hammer_carry_time = hammer_carry_time,
                hammer_can_hit = hammer_can_hit,
                block_jumping_and_climbing = block_jumping_and_climbing,
                hammer_usage_expired = hammer_usage_expired,
            ) 
            new_state = calculate_hammer_pos_relative_to_mario(new_state)
            return jax.lax.cond(
                state.hammer_taken,
                lambda _: new_state,
                lambda _: state,
                operand=None
            )
        new_state = hammer_swing(new_state)


        # Check if hammer hits barrel; if so, barrel will be "destroyed"
        def hammer_barrel_collision(state):
            def check_collision_for_each_barrel(barrel_idx, state):
                # collision checks for valid collision
                collision = JaxDonkeyKong._collision_between_two_objects( state.barrels.barrel_x[barrel_idx], state.barrels.barrel_y[barrel_idx], self.consts.BARREL_HIT_BOX_X, self.consts.BARREL_HIT_BOX_Y,
                                                            state.hammer_x, state.hammer_y, self.consts.HAMMER_SWING_HIT_BOX_X, self.consts.HAMMER_SWING_HIT_BOX_Y)
                collision &= state.hammer_can_hit
                collision &= jnp.logical_not(state.barrels.reached_the_end[barrel_idx])

                # if collision, mark the barrel as destroyed and increase the game score
                barrels = state.barrels
                new_state = state._replace(
                    barrels = BarrelPosition(
                        barrel_x=barrels.barrel_x.at[barrel_idx].set(-1),
                        barrel_y=barrels.barrel_y.at[barrel_idx].set(-1),
                        sprite=barrels.sprite.at[barrel_idx].set(self.consts.BARREL_SPRITE_RIGHT),
                        moving_direction=barrels.moving_direction.at[barrel_idx].set(self.consts.MOVING_RIGHT),
                        stage=barrels.stage.at[barrel_idx].set(6),
                        reached_the_end=barrels.reached_the_end.at[barrel_idx].set(True),
                    ),
                    game_score = state.game_score + self.consts.SCORE_FOR_DESTROYING_BARREL,
                )

                return jax.lax.cond(
                    collision,
                    lambda _: new_state,
                    lambda _: state,
                    operand=None
                )
            new_state = jax.lax.fori_loop(0, len(state.barrels.barrel_x), check_collision_for_each_barrel, state)
            return new_state
        new_state = hammer_barrel_collision(new_state)

        # Check if hammer hits fire on stage 4; if so, fire will be "destroyed"
        # in donkeyKong Level 2: only fire on stage 4 can be destroyed
        def hammer_fire_stage_4_collision(state):
            stage_4_fire_idx = 1
            collision = JaxDonkeyKong._collision_between_two_objects( state.fires.fire_x[stage_4_fire_idx], state.fires.fire_y[stage_4_fire_idx], self.consts.FIRE_HIT_BOX_X, self.consts.FIRE_HIT_BOX_Y,
                                                            state.hammer_x, state.hammer_y, self.consts.HAMMER_SWING_HIT_BOX_X, self.consts.HAMMER_SWING_HIT_BOX_Y)
            collision &= state.hammer_can_hit
            collision &= jnp.logical_not(state.fires.destroyed[stage_4_fire_idx])

            destroyed = state.fires.destroyed.at[stage_4_fire_idx].set(True)
            fires = state.fires._replace(destroyed=destroyed)
            new_state = state._replace(fires=fires, game_score = state.game_score + self.consts.SCORE_FOR_DESTROYING_FIRE)

            return jax.lax.cond(
                collision,
                lambda _: new_state,
                lambda _: state,
                operand=None
            )
        new_state = hammer_fire_stage_4_collision(new_state)

        return new_state


    def reset(self, key = [0,0]) -> Tuple[DonkeyKongObservation, DonkeyKongState]:
        """
        Resets the game state to the initial state.
        Returns the initial state and the reward (i.e. 0)
        """
        ladders = self.init_ladders_for_level(level=1)
        invisible_wall_each_stage = self.init_invisible_wall_for_level(level=1)
        state = DonkeyKongState(
            game_started = False,
            level = 1,
            step_counter=jnp.array(1).astype(jnp.int32),
            game_timer = 1,
            game_remaining_time = 5000,
            frames_since_last_barrel_spawn=jnp.array(0).astype(jnp.int32),

            mario_climbing_delay = False,

            game_score = 0,
            game_round = 1,

            mario_x=self.consts.LEVEL_1_MARIO_START_X,
            mario_y=self.consts.LEVEL_1_MARIO_START_Y,
            mario_jumping=False,
            mario_jumping_wide=False,
            mario_jumping_over_enemy=False,
            mario_climbing=False,
            start_frame_when_mario_jumped=-1,
            mario_view_direction=self.consts.MOVING_RIGHT,
            mario_walk_frame_counter=0,
            mario_climb_frame_counter=0,
            mario_walk_sprite=self.consts.MARIO_WALK_SPRITE_0,
            mario_climb_sprite=self.consts.MARIO_CLIMB_SPRITE_0,
            mario_stage=1,
            mario_life_counter = 2,
            mario_got_hit = False,
            game_freeze_start = -1,
            mario_reached_goal = False,

            donkey_kong_sprite = self.consts.DONKEY_KONG_SPRITE_0,

            barrels = BarrelPosition(
                barrel_x = jnp.array([-1, -1, -1, -1]).astype(jnp.int32),
                barrel_y = jnp.array([-1, -1, -1, -1]).astype(jnp.int32), 
                sprite = jnp.array([self.consts.BARREL_SPRITE_RIGHT, self.consts.BARREL_SPRITE_RIGHT, self.consts.BARREL_SPRITE_RIGHT, self.consts.BARREL_SPRITE_RIGHT]).astype(jnp.int32),
                moving_direction = jnp.array([self.consts.MOVING_RIGHT, self.consts.MOVING_RIGHT, self.consts.MOVING_RIGHT, self.consts.MOVING_RIGHT]).astype(jnp.int32),
                stage = jnp.array([6, 6, 6, 6]).astype(jnp.int32),
                reached_the_end=jnp.array([True, True, True, True]).astype(bool)
            ),
            barrels_change_direction_prob = self.consts.BASE_PROBABILITY_BARREL_ROLLING_A_LADDER_DOWN_ROUND_1,

            fires = FirePosition(
                fire_x = jnp.array([-1., -1., -1., -1.]).astype(jnp.float32),
                fire_y = jnp.array([-1., -1., -1., -1.]).astype(jnp.float32),
                moving_direction = jnp.array([self.consts.MOVING_RIGHT, self.consts.MOVING_LEFT, self.consts.MOVING_RIGHT, self.consts.MOVING_LEFT]).astype(jnp.int32),
                stage = jnp.array([5, 4, 3, 2]).astype(jnp.int32),
                destroyed=jnp.array([True, True, True, True]).astype(bool),
                change_direction_prob=jnp.array([self.consts.STAGE_5_FIRE_CHANGING_DIRECTION_DEFAULT_PROB, self.consts.STAGE_4_FIRE_CHANGING_DIRECTION_DEFAULT_PROB, self.consts.STAGE_3_FIRE_CHANGING_DIRECTION_DEFAULT_PROB, self.consts.STAGE_2_FIRE_CHANGING_DIRECTION_DEFAULT_PROB])
            ),

            traps = TrapPosition(
                trap_x = jnp.array([self.consts.TRAP_FLOOR_5_X, self.consts.TRAP_FLOOR_4_X, self.consts.TRAP_FLOOR_3_X, self.consts.TRAP_FLOOR_2_X, self.consts.TRAP_FLOOR_5_X, self.consts.TRAP_FLOOR_4_X, self.consts.TRAP_FLOOR_3_X, self.consts.TRAP_FLOOR_2_X]).astype(jnp.int32),
                trap_y = jnp.array([self.consts.TRAP_LEFT_Y, self.consts.TRAP_LEFT_Y, self.consts.TRAP_LEFT_Y, self.consts.TRAP_LEFT_Y, self.consts.TRAP_RIGHT_Y, self.consts.TRAP_RIGHT_Y, self.consts.TRAP_RIGHT_Y, self.consts.TRAP_RIGHT_Y]).astype(jnp.int32),
                stage = jnp.array([5, 4, 3, 2, 5, 4, 3, 2]).astype(jnp.int32),
                triggered = jnp.array([False, False, False, False, False, False, False, False]).astype(bool),
                fall_protection = jnp.array([-1, -1, -1, -1, -1, -1, -1, -1]).astype(jnp.int32), # -1 indicates no protection, otherwise safe the view direction of mario; e.g. : if mario triggers a trap looking right, he is protected while he is moving right. But if he turns left, protection is instantly gone
            ),

            ladders=ladders,
            invisible_wall_each_stage=invisible_wall_each_stage,
            random_key = jax.random.PRNGKey(key[0]),

            hammer_x = self.consts.LEVEL_1_HAMMER_X,
            hammer_y = self.consts.LEVEL_1_HAMMER_Y,
            hammer_can_hit = False,
            hammer_taken = False,
            hammer_carry_time = 0,
            block_jumping_and_climbing = False,
            hammer_usage_expired = False,
        )
        initial_obs = self._get_observation(state)

        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: DonkeyKongState, action: chex.Array) -> Tuple[DonkeyKongObservation, DonkeyKongState, float, bool, DonkeyKongInfo]:
        # First search for colision
        new_state = state

        # If there is no colision: game will continue
        # enemy_step --> maybe later write a enemy_step function which calls eighter barrel_step oder fire_step
        new_state = self._enemy_step(new_state)

        # mario step / player step
        new_state = self._mario_step(new_state, action)

        # hammer step
        new_state = self._hammer_step(new_state, action)

        # increase timer / frame counter
        game_remaining_time = jax.lax.cond(
            new_state.game_timer % self.consts.TIMER_REDUTION_DURATION == 0,
            lambda _: new_state.game_remaining_time - self.consts.TIMER_REDUTION_AMOUNT,
            lambda _: new_state.game_remaining_time,
            operand=None
        )
        mario_got_hit, game_freeze_start = jax.lax.cond(
            jnp.logical_and(new_state.game_remaining_time == 0, new_state.mario_got_hit == False),
            lambda _: (True, new_state.step_counter),
            lambda _: (new_state.mario_got_hit, new_state.game_freeze_start),
            operand=None
        )
        new_state = new_state._replace(
            step_counter=new_state.step_counter+1,
            frames_since_last_barrel_spawn=new_state.frames_since_last_barrel_spawn+1,
            game_remaining_time=game_remaining_time,
            game_timer=new_state.game_timer+1,
            mario_got_hit = mario_got_hit,
            game_freeze_start=game_freeze_start,
        )
        
        # Check if game was even started --> with human_action FIRE
        def start_game():
            started_state_level_1 = state._replace(
                game_started = True,
                barrels = BarrelPosition(
                    barrel_x = jnp.array([self.consts.BARREL_START_X, -1, -1, -1]).astype(jnp.int32),
                    barrel_y = jnp.array([self.consts.BARREL_START_Y, -1, -1, -1]).astype(jnp.int32), 
                    sprite = jnp.array([self.consts.BARREL_SPRITE_RIGHT, self.consts.BARREL_SPRITE_RIGHT, self.consts.BARREL_SPRITE_RIGHT, self.consts.BARREL_SPRITE_RIGHT]).astype(jnp.int32),
                    moving_direction = jnp.array([self.consts.MOVING_RIGHT, self.consts.MOVING_RIGHT, self.consts.MOVING_RIGHT, self.consts.MOVING_RIGHT]).astype(jnp.int32),
                    stage = jnp.array([6, 6, 6, 6]).astype(jnp.int32),
                    reached_the_end=jnp.array([False, True, True, True]).astype(bool)
                ),
            )
            started_state_level_2 = state._replace(
                game_started = True,
                fires = FirePosition(
                    fire_x = jnp.array([self.bar_linear_equation(5, self.consts.FIRE_START_Y, 2) - self.consts.FIRE_HIT_BOX_X, self.bar_linear_equation(4, self.consts.FIRE_START_Y, 2) - self.consts.FIRE_HIT_BOX_X, self.bar_linear_equation(3, self.consts.FIRE_START_Y, 2) - self.consts.FIRE_HIT_BOX_X, self.bar_linear_equation(2, self.consts.FIRE_START_Y, 2) - self.consts.FIRE_HIT_BOX_X]).astype(jnp.float32),
                    fire_y = jnp.array([self.consts.FIRE_START_Y, self.consts.FIRE_START_Y, self.consts.FIRE_START_Y, self.consts.FIRE_START_Y]).astype(jnp.float32),
                    moving_direction = jnp.array([self.consts.MOVING_RIGHT, self.consts.MOVING_LEFT, self.consts.MOVING_RIGHT, self.consts.MOVING_LEFT]).astype(jnp.int32),
                    stage = jnp.array([5, 4, 3, 2]).astype(jnp.int32),
                    destroyed=jnp.array([False, True, True, True]).astype(bool),
                    change_direction_prob = state.fires.change_direction_prob,
                )
            )
            return jax.lax.cond(
                jnp.logical_or(action == Action.FIRE, jnp.logical_or(action == Action.RIGHTFIRE, action == Action.LEFTFIRE)),
                lambda _: jax.lax.cond(
                    state.level == 1,
                    lambda _: started_state_level_1,
                    lambda _: started_state_level_2,
                    operand=None
                ),
                lambda _: state,
                operand=None
            )
        new_state = jax.lax.cond(
            state.game_started == False,
            lambda _: start_game(),
            lambda _: new_state,
            operand=None
        )

        done = self._get_done(new_state)
        env_reward = self._get_reward(state, new_state)
        all_rewards = self._get_all_reward(state, new_state)
        info = self._get_info(new_state, all_rewards)
        observation = self._get_observation(new_state)
        return observation, new_state, env_reward, done, info

    
    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: DonkeyKongState):
        mario_position = EntityPosition(
            x = jnp.round(state.mario_x).astype(jnp.int32),
            y = jnp.round(state.mario_y).astype(jnp.int32),
            width = self.consts.MARIO_HIT_BOX_Y,
            height = self.consts.MARIO_HIT_BOX_X
        )
        mario_jumping = jnp.logical_or(state.mario_jumping, state.mario_jumping_wide)
        hammer_position = EntityPosition(
            x = state.hammer_x,
            y = state.hammer_y,
            width = self.consts.HAMMER_HIT_BOX_Y,
            height = self.consts.HAMMER_HIT_BOX_X,
        )

        nums_barrels = state.barrels.barrel_x.shape[0]
        barrels = EntityPosition(
            x = state.barrels.barrel_x,
            y = state.barrels.barrel_y,
            width = jnp.full((nums_barrels,), self.consts.BARREL_HIT_BOX_Y),
            height = jnp.full((nums_barrels,), self.consts.BARREL_HIT_BOX_X),
        )
        barrel_mask = jnp.where(state.barrels.reached_the_end, 0, 1)
        
        # nums_fires = state.fires.fire_x.shape[0]
        # fires = EntityPosition(
        #     x = state.fires.fire_x,
        #     y = state.fires.fire_y,
        #     width = jnp.full((nums_fires,), self.consts.FIRE_HIT_BOX_Y),
        #     height = jnp.full((nums_fires,), self.consts.FIRE_HIT_BOX_X),
        # )
        # fire_mask = jnp.where(state.fires.destroyed, 0, 1)
        # nums_traps = state.traps.trap_x.shape[0]
        # traps = EntityPositionTrap(
        #     x = state.traps.trap_x,
        #     y = state.traps.trap_y,
        #     width = jnp.full((nums_traps,), self.consts.TRAP_WIDTH),
        #     triggered = state.traps.triggered,
        # )
        # nums_ladders = state.ladders.start_x.shape[0]
        # ladders = EntityPositionLadder(
        #     start_x = state.ladders.start_x,
        #     start_y = state.ladders.start_y,
        #     end_x = state.ladders.end_x,
        #     end_y = state.ladders.end_y,
        #     width = jnp.full((nums_ladders,), self.consts.LADDER_WIDTH),
        # )
        # ladder_mask = jnp.where(state.ladders.start_x != -1, 1, 0)
        
        return DonkeyKongObservation(
            total_score = state.game_score,
            time_remaining = state.game_remaining_time,
            mario_lives = state.mario_life_counter, 
            goal_reached = state.mario_reached_goal,
            level = state.level,
            mario_position = mario_position,
            mario_view_direction = state.mario_view_direction,
            mario_jumping = mario_jumping,
            mario_climbing = state.mario_climbing,
            hammer_position = hammer_position,
            hammer_can_destroy_enemy = state.hammer_can_hit,
            barrels = barrels,
            barrel_mask = barrel_mask,
            # fires = fires,
            # fire_mask = fire_mask,
            # traps = traps,
            # ladders = ladders,
            # ladder_mask = ladder_mask,
        )

    def render(self, state: DonkeyKongState) -> jnp.ndarray:
        return self.renderer.render(state)

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: DonkeyKongObservation) -> jnp.ndarray:
        return jnp.concatenate([
            # --- Meta ---
            obs.total_score.flatten(),
            obs.time_remaining.flatten(),
            obs.mario_lives.flatten(),
            obs.goal_reached.flatten(),
            obs.level.flatten(),

            # --- Mario ---
            obs.mario_position.x.flatten(),
            obs.mario_position.y.flatten(),
            obs.mario_position.width.flatten(),
            obs.mario_position.height.flatten(),
            obs.mario_view_direction.flatten(),
            obs.mario_jumping.flatten(),
            obs.mario_climbing.flatten(),

            # --- Hammer ---
            obs.hammer_position.x.flatten(),
            obs.hammer_position.y.flatten(),
            obs.hammer_position.width.flatten(),
            obs.hammer_position.height.flatten(),
            obs.hammer_can_destroy_enemy.flatten(),

            # --- Barrels ---         
            obs.barrels.x,
            obs.barrels.y,
            obs.barrels.width,
            obs.barrels.height,
            obs.barrel_mask,

            # --- Fires ---
            # obs.fires.x.flatten(),
            # obs.fires.y.flatten(),
            # obs.fires.width.flatten(),
            # obs.fires.height.flatten(),
            # obs.fire_mask.flatten(),

            # # --- Traps ---
            # obs.traps.x.flatten(),
            # obs.traps.y.flatten(),
            # obs.traps.width.flatten(),
            # obs.traps.triggered.flatten(),

            # # --- Ladders ---
            # obs.ladders.start_x.flatten(),
            # obs.ladders.start_y.flatten(),
            # obs.ladders.end_x.flatten(),
            # obs.ladders.end_y.flatten(),
            # obs.ladders.width.flatten(),
            # obs.ladder_mask.flatten(),
        ])

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(8)

    def observation_space(self) -> spaces.Box:
        MAX_BARRELS = self.consts.MAX_BARRELS
        MAX_FIRES = self.consts.MAX_FIRES
        MAX_TRAPS = self.consts.MAX_TRAPS
        MAX_LADDERS = self.consts.MAX_LADDERS
        
        return spaces.Dict({
            "total_score": spaces.Box(low=0, high=1e6, shape=(), dtype=jnp.int32),
            "time_remaining": spaces.Box(low=0, high=9999, shape=(), dtype=jnp.int32),
            "mario_lives": spaces.Box(low=0, high=2, shape=(), dtype=jnp.int32),
            "goal_reached": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
            "level": spaces.Discrete(2), 

            # Mario
            "mario_position": spaces.Dict({
                "x": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),    
                "width": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
            }),
            "mario_view_direction": spaces.Box(low=-1, high=1, shape=(), dtype=jnp.int32),
            "mario_jumping": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
            "mario_climbing": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),

            # Hammer
            "hammer_position": spaces.Dict({
                "x": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),    
                "width": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
            }),
            "hammer_can_destroy_enemy": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),

            # Barrels
            "barrels": spaces.Dict({
                "x": spaces.Box(low=0, high=210, shape=(MAX_BARRELS,), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=160, shape=(MAX_BARRELS, ), dtype=jnp.int32),    
                "width": spaces.Box(low=0, high=160, shape=(MAX_BARRELS, ), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=210, shape=(MAX_BARRELS, ), dtype=jnp.int32),
            }),
            "barrel_mask": spaces.Box(low=0, high=1, shape=(MAX_BARRELS,), dtype=jnp.int32),

            # # Fire
            # "fires": spaces.Dict({
            #     "x": spaces.Box(low=0, high=210, shape=(MAX_FIRES,), dtype=jnp.int32),
            #     "y": spaces.Box(low=0, high=160, shape=(MAX_FIRES, ), dtype=jnp.int32),    
            #     "width": spaces.Box(low=0, high=160, shape=(MAX_FIRES, ), dtype=jnp.int32),
            #     "height": spaces.Box(low=0, high=210, shape=(MAX_FIRES, ), dtype=jnp.int32),
            # }),
            # "fire_mask": spaces.Box(low=0, high=1, shape=(MAX_BARRELS,), dtype=jnp.int32),

            # # Traps
            # "traps": spaces.Dict({
            #     "x": spaces.Box(low=0, high=210, shape=(MAX_TRAPS,), dtype=jnp.int32),
            #     "y": spaces.Box(low=0, high=160, shape=(MAX_TRAPS,), dtype=jnp.int32),
            #     "width": spaces.Box(low=0, high=160, shape=(MAX_TRAPS,), dtype=jnp.int32),
            #     "triggered": spaces.Box(low=0, high=1, shape=(MAX_TRAPS,), dtype=jnp.int32),
            # }),

            # # Ladders
            # "ladders": spaces.Dict({
            #     "start_x": spaces.Box(low=0, high=210, shape=(MAX_LADDERS,), dtype=jnp.int32),
            #     "start_y": spaces.Box(low=0, high=160, shape=(MAX_LADDERS,), dtype=jnp.int32),
            #     "end_x": spaces.Box(low=0, high=210, shape=(MAX_LADDERS,), dtype=jnp.int32),
            #     "end_y": spaces.Box(low=0, high=160, shape=(MAX_LADDERS,), dtype=jnp.int32),
            #     "width": spaces.Box(low=0, high=160, shape=(MAX_LADDERS,), dtype=jnp.int32),
            # }),
            # "ladder_mask": spaces.Box(low=0, high=1, shape=(MAX_LADDERS,), dtype=jnp.int32),
        })

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(210, 160, 3),
            dtype=jnp.uint8
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: DonkeyKongState, all_rewards: chex.Array = None) -> DonkeyKongInfo:
        return DonkeyKongInfo(time=state.step_counter, all_rewards=all_rewards)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: DonkeyKongState, state: DonkeyKongState):
        return state.game_score - previous_state.game_score

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(self, previous_state: DonkeyKongState, state: DonkeyKongState):
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array(
            [reward_func(previous_state, state) for reward_func in self.reward_funcs]
        )
        return rewards

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state) -> bool:
        return state.mario_life_counter < 0


class DonkeyKongRenderer(JAXGameRenderer):
    """JAX-based Pong game renderer, optimized with JIT compilation."""

    def __init__(self, consts: DonkeyKongConstants = None):
        super().__init__()
        self.consts = consts or DonkeyKongConstants()
        (
            self.SPRITES_BG,
            self.SPRITES_DONKEYKONG,
            self.SPRITE_GIRLFRIEND,
            self.SPRITES_BARREL,
            self.SPRITES_LIFEBAR,
            self.SPRITES_MARIO_STANDING,
            self.SPRITES_MARIO_JUMPING,
            self.SPRITES_MARIO_WALKING_1,
            self.SPRITES_MARIO_WALKING_2,
            self.SPRITES_MARIO_CLIMBING,
            self.SPRITES_HAMMER_UP,
            self.SPRITES_HAMMER_DOWN,
            self.SPRITE_FIRE,
            self.SPRITE_DROP_PIT,
            self.SPRITES_BLUE_DIGITS,
            self.SPRITES_YELLOW_DIGITS,
        ) = self.load_sprites()

    
    def load_sprites(self):
        """Load all sprites required for Pong rendering."""
        MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

        # Load sprites
        bg_level_1 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/donkeyKong_background_level_1.npy"), transpose=False)
        bg_level_2 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/donkeyKong_background_level_2.npy"), transpose=False)

        donkeyKong_pose_1 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/donkeyKong1.npy"), transpose=False)
        donkeyKong_pose_2 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/donkeyKong2.npy"), transpose=False)

        girlfriend = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/girlfriend.npy"), transpose=False)

        level_1_life_bar = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/level_1_life_bar.npy"), transpose=False)
        level_2_life_bar = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/level_2_life_bar.npy"), transpose=False)

        mario_standing_right = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/mario_standing_right.npy"), transpose=False)
        mario_standing_left = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/mario_standing_left.npy"), transpose=False)
        mario_jumping_right = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/mario_jumping_right.npy"), transpose=False)
        mario_jumping_left = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/mario_jumping_left.npy"), transpose=False)
        mario_walking_1_right = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/mario_walking_1_right.npy"), transpose=False)
        mario_walking_1_left = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/mario_walking_1_left.npy"), transpose=False)
        mario_walking_2_right = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/mario_walking_2_right.npy"), transpose=False)
        mario_walking_2_left = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/mario_walking_2_left.npy"), transpose=False)
        mario_climbing_right = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/mario_climbing_right.npy"), transpose=False)
        mario_climbing_left = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/mario_climbing_left.npy"), transpose=False)

        hammer_up_level_1 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/hammer_up_level_1.npy"), transpose=False)
        hammer_up_level_2 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/hammer_up_level_2.npy"), transpose=False)
        hammer_down_right_level_1 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/hammer_down_right_level_1.npy"), transpose=False)
        hammer_down_left_level_1 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/hammer_down_left_level_1.npy"), transpose=False)
        hammer_down_right_level_2 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/hammer_down_right_level_2.npy"), transpose=False)
        hammer_down_left_level_2 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/hammer_down_left_level_2.npy"), transpose=False)

        fire = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/fire.npy"), transpose=False)

        drop_pit = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/drop_pit.npy"), transpose=False)

        # Convert all sprites to the expected format (add frame dimension)
        SPRITES_BG = jnp.stack([bg_level_1, bg_level_2], axis=0)
        SPRITES_DONKEYKONG = jnp.stack([donkeyKong_pose_1, donkeyKong_pose_2], axis=0)
        SPRITE_GIRLFRIEND = jnp.expand_dims(girlfriend, axis=0)
        SPRITES_LIFEBAR = jnp.stack([level_1_life_bar, level_2_life_bar], axis=0)
        SPRITES_MARIO_STANDING = jnp.stack([mario_standing_right, mario_standing_left], axis=0)
        SPRITES_MARIO_JUMPING = jnp.stack([mario_jumping_right, mario_jumping_left], axis=0)
        SPRITES_MARIO_WALKING_1 = jnp.stack([mario_walking_1_right, mario_walking_1_left], axis=0)
        SPRITES_MARIO_WALKING_2 = jnp.stack([mario_walking_2_right, mario_walking_2_left], axis=0)
        SPRITES_MARIO_CLIMBING = jnp.stack([mario_climbing_right, mario_climbing_left], axis=0)
        SPRITES_HAMMER_UP = jnp.stack([hammer_up_level_1, hammer_up_level_2], axis=0)
        SPRITES_HAMMER_DOWN = jnp.stack([hammer_down_right_level_1, hammer_down_left_level_1, hammer_down_right_level_2, hammer_down_left_level_2], axis=0)
        SPRITE_FIRE = jnp.expand_dims(fire, axis=0)
        SPRITE_DROP_PIT = jnp.expand_dims(drop_pit, axis=0)

        SPRITES_BARREL = aj.load_and_pad_digits(
            os.path.join(MODULE_DIR, "sprites/donkeyKong/barrel{}.npy"),
            num_chars=3,
        )

        SPRITES_BLUE_DIGITS = aj.load_and_pad_digits(
            os.path.join(MODULE_DIR, "sprites/donkeyKong/digits/blue_score_{}.npy"),
            num_chars=10,
        )
        SPRITES_YELLOW_DIGITS = aj.load_and_pad_digits(
            os.path.join(MODULE_DIR, "sprites/donkeyKong/digits/yellow_score_{}.npy"),
            num_chars=10,
        )

        return (
            SPRITES_BG,
            SPRITES_DONKEYKONG,
            SPRITE_GIRLFRIEND,
            SPRITES_BARREL,
            SPRITES_LIFEBAR,
            SPRITES_MARIO_STANDING,
            SPRITES_MARIO_JUMPING,
            SPRITES_MARIO_WALKING_1,
            SPRITES_MARIO_WALKING_2,
            SPRITES_MARIO_CLIMBING,
            SPRITES_HAMMER_UP,
            SPRITES_HAMMER_DOWN,
            SPRITE_FIRE,
            SPRITE_DROP_PIT,
            SPRITES_BLUE_DIGITS,
            SPRITES_YELLOW_DIGITS,
        )

    
    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        """
        Renders the current game state using JAX operations.

        Args:
            state: A DonkeyKongState object containing the current game state.

        Returns:
            A JAX array representing the rendered frame.
        """

        def render_at_transparent(raster, x, y, sprite):
            if sprite.shape[-1] > 3:
                sprite = sprite[:, :, :3]
            
            h, w, _ = sprite.shape
            sub_raster = jax.lax.dynamic_slice(raster, (x, y, 0), (h, w, 3))

            # Transparent Pixel = every channel with value 0
            mask = jnp.any(sprite != 0, axis=-1, keepdims=True)
            mask = jnp.broadcast_to(mask, sprite.shape)

            blended = jnp.where(mask, sprite, sub_raster)

            return jax.lax.dynamic_update_slice(raster, blended, (x, y, 0))

        def create_bg_raster_for_level_2_regarding_drop_pits(raster):
            frame_bg = aj.get_sprite_frame(self.SPRITES_BG, 1)
            raster = aj.render_at(raster, 0, 0, frame_bg)
            frame_drop_pit = aj.get_sprite_frame(self.SPRITE_DROP_PIT, 0)

            def look_for_triggered_traps(i, raster):
                return jax.lax.cond(
                    state.traps.triggered[i] == False,
                    lambda _: aj.render_at(raster, state.traps.trap_y[i], state.traps.trap_x[i], frame_drop_pit),
                    lambda _: raster,
                    operand=None
                )
            raster = jax.lax.fori_loop(0, len(state.traps.trap_x), look_for_triggered_traps, raster)

            return raster            

        raster = jnp.zeros((self.consts.HEIGHT, self.consts.WIDTH, 3))

        # Background raster
        level = state.level
        raster = jax.lax.cond(
            level == 1,
            lambda _: aj.render_at(raster, 0, 0, aj.get_sprite_frame(self.SPRITES_BG, 0)),
            lambda x: create_bg_raster_for_level_2_regarding_drop_pits(x),
            raster 
        )

        # DonkeyKong
        frame_donkeyKong = aj.get_sprite_frame(self.SPRITES_DONKEYKONG, state.donkey_kong_sprite)
        raster = aj.render_at(raster, self.consts.DONKEYKONG_X, self.consts.DONKEYKONG_Y, frame_donkeyKong)

        # Girlfriend
        frame_girlfriend = aj.get_sprite_frame(self.SPRITE_GIRLFRIEND, 0)
        raster = aj.render_at(raster, self.consts.GIRLFRIEND_X, self.consts.GIRLFRIEND_Y, frame_girlfriend)

        # Life Bars - depending if lifes are still given 
        frame_life_bar = jax.lax.cond(
            state.level == 1,
            lambda _: aj.get_sprite_frame(self.SPRITES_LIFEBAR, 0),
            lambda _: aj.get_sprite_frame(self.SPRITES_LIFEBAR, 1),
            operand=None
        )
        life_bar_1_y = jax.lax.cond(
            state.level == 1,
            lambda _: self.consts.LEVEL_1_LIFE_BAR_1_Y,
            lambda _: self.consts.LEVEL_2_LIFE_BAR_1_Y,
            operand=None
        )
        life_bar_2_y = jax.lax.cond(
            state.level == 1,
            lambda _: self.consts.LEVEL_1_LIFE_BAR_2_Y,
            lambda _: self.consts.LEVEL_2_LIFE_BAR_2_Y,
            operand=None
        )
        raster = jax.lax.cond(
            state.mario_life_counter == 2,
            lambda _: aj.render_at(raster, life_bar_1_y, self.consts.LIFE_BAR_X, frame_life_bar),
            lambda _: raster,
            operand=None
        )
        raster = jax.lax.cond(
            state.mario_life_counter >= 1,
            lambda _: aj.render_at(raster, life_bar_2_y, self.consts.LIFE_BAR_X, frame_life_bar),
            lambda _: raster,
            operand=None
        )

        # Mario
        frame_mario_right_side_walk_0 = aj.get_sprite_frame(self.SPRITES_MARIO_STANDING, 0)
        frame_mario_right_side_walk_1 = aj.get_sprite_frame(self.SPRITES_MARIO_WALKING_1, 0)
        frame_mario_right_side_walk_2 = aj.get_sprite_frame(self.SPRITES_MARIO_WALKING_2, 0)
        frame_mario_left_side_walk_0 = aj.get_sprite_frame(self.SPRITES_MARIO_STANDING, 1)
        frame_mario_left_side_walk_1 = aj.get_sprite_frame(self.SPRITES_MARIO_WALKING_1, 1)
        frame_mario_left_side_walk_2 = aj.get_sprite_frame(self.SPRITES_MARIO_WALKING_2, 1)
        frame_mario_jumping_right = aj.get_sprite_frame(self.SPRITES_MARIO_JUMPING, 0)
        frame_mario_jumping_left = aj.get_sprite_frame(self.SPRITES_MARIO_JUMPING, 1)
        frame_mario_climbing_left = aj.get_sprite_frame(self.SPRITES_MARIO_CLIMBING, 0)
        frame_mario_climbing_right = aj.get_sprite_frame(self.SPRITES_MARIO_CLIMBING, 1)
        mario_x = state.mario_x
        mario_y = state.mario_y
        raster = jax.lax.cond(
            jnp.logical_and(jnp.logical_or(state.mario_jumping, state.mario_jumping_wide), state.mario_view_direction == self.consts.MOVING_RIGHT),
            lambda _: render_at_transparent(raster, jnp.int32(jnp.round(mario_x)), jnp.int32(jnp.round(mario_y)), frame_mario_jumping_right),
            lambda _: jax.lax.cond(
                jnp.logical_and(jnp.logical_or(state.mario_jumping, state.mario_jumping_wide), state.mario_view_direction == self.consts.MOVING_LEFT),
                lambda _: render_at_transparent(raster, jnp.int32(jnp.round(mario_x)), jnp.int32(jnp.round(mario_y)), frame_mario_jumping_left),
                lambda _: jax.lax.cond(
                    jnp.logical_and(state.mario_view_direction==self.consts.MOVING_RIGHT, jnp.logical_or(state.mario_walk_sprite == self.consts.MARIO_WALK_SPRITE_0, state.mario_walk_sprite == self.consts.MARIO_WALK_SPRITE_2)),
                    lambda _: render_at_transparent(raster, jnp.int32(jnp.round(mario_x)), jnp.int32(jnp.round(mario_y)), frame_mario_right_side_walk_0),
                    lambda _: jax.lax.cond(
                        jnp.logical_and(state.mario_view_direction==self.consts.MOVING_RIGHT, state.mario_walk_sprite == self.consts.MARIO_WALK_SPRITE_1),
                        lambda _: render_at_transparent(raster, jnp.int32(jnp.round(mario_x)), jnp.int32(jnp.round(mario_y)), frame_mario_right_side_walk_1),
                        lambda _: jax.lax.cond(
                            jnp.logical_and(state.mario_view_direction==self.consts.MOVING_RIGHT, state.mario_walk_sprite == self.consts.MARIO_WALK_SPRITE_3),
                            lambda _: render_at_transparent(raster, jnp.int32(jnp.round(mario_x))+1, jnp.int32(jnp.round(mario_y)), frame_mario_right_side_walk_2),
                            lambda _: jax.lax.cond(
                                jnp.logical_and(state.mario_view_direction==self.consts.MOVING_LEFT, jnp.logical_or(state.mario_walk_sprite == self.consts.MARIO_WALK_SPRITE_0, state.mario_walk_sprite == self.consts.MARIO_WALK_SPRITE_2)),
                                lambda _: render_at_transparent(raster, jnp.int32(jnp.round(mario_x)), jnp.int32(jnp.round(mario_y)), frame_mario_left_side_walk_0),
                                lambda _: jax.lax.cond(
                                    jnp.logical_and(state.mario_view_direction==self.consts.MOVING_LEFT, state.mario_walk_sprite == self.consts.MARIO_WALK_SPRITE_1),
                                    lambda _: render_at_transparent(raster, jnp.int32(jnp.round(mario_x)), jnp.int32(jnp.round(mario_y)), frame_mario_left_side_walk_1),
                                    lambda _: jax.lax.cond(
                                        jnp.logical_and(state.mario_view_direction==self.consts.MOVING_LEFT, state.mario_walk_sprite == self.consts.MARIO_WALK_SPRITE_3),
                                        lambda _: render_at_transparent(raster, jnp.int32(jnp.round(mario_x))+1, jnp.int32(jnp.round(mario_y)), frame_mario_left_side_walk_2),
                                        lambda _: jax.lax.cond(
                                            jnp.logical_and(state.mario_view_direction==self.consts.MOVING_UP, state.mario_climb_sprite==self.consts.MARIO_CLIMB_SPRITE_0),
                                            lambda _: render_at_transparent(raster, jnp.int32(jnp.round(mario_x)), jnp.int32(jnp.round(mario_y)), frame_mario_climbing_left),
                                            lambda _: jax.lax.cond(
                                                jnp.logical_and(state.mario_view_direction==self.consts.MOVING_UP, state.mario_climb_sprite==self.consts.MARIO_CLIMB_SPRITE_1),
                                                lambda _: render_at_transparent(raster, jnp.int32(jnp.round(mario_x)), jnp.int32(jnp.round(mario_y)), frame_mario_climbing_right),
                                                lambda _: raster,
                                                operand=None
                                            ),
                                            operand=None
                                        ),
                                        operand=None
                                    ),
                                    operand=None
                                ),
                                operand=None
                            ),
                            operand=None
                        ),
                        operand=None
                    ),
                    operand=None
                ),
                operand=None
            ),
            operand=None
        )

        # Barrels if there are some on the field
        barrels = state.barrels
        for barrel_x, barrel_y, sprite_id, reached_the_end in zip(barrels.barrel_x, barrels.barrel_y, barrels.sprite, barrels.reached_the_end):
            frame_barrel = aj.get_sprite_frame(self.SPRITES_BARREL, sprite_id)
            raster = jax.lax.cond(
                reached_the_end,
                lambda _: raster,
                lambda _: render_at_transparent(raster, barrel_x, barrel_y, frame_barrel),
                operand=None
            )

        # Fires if there are some on the field
        fires = state.fires
        fire_sprite = aj.get_sprite_frame(self.SPRITE_FIRE, 0)
        for fire_x, fire_y, destroyed in zip(fires.fire_x, fires.fire_y, fires.destroyed):
            raster = jax.lax.cond(
                destroyed,
                lambda _: raster,
                lambda _: render_at_transparent(raster, jnp.round(fire_x).astype(int), jnp.round(fire_y).astype(int), fire_sprite),
                operand=None 
            )

        # Hammer
        frame_hammer = jax.lax.cond(
            state.level == 1,
            lambda _: aj.get_sprite_frame(self.SPRITES_HAMMER_UP, 0),
            lambda _: aj.get_sprite_frame(self.SPRITES_HAMMER_UP, 1),
            operand=None
        )
        frame_hammer_down = jax.lax.cond(
            state.mario_view_direction == self.consts.MOVING_RIGHT,
            lambda _: jax.lax.cond(
                state.level == 1,
                lambda _: aj.get_sprite_frame(self.SPRITES_HAMMER_DOWN, 0),
                lambda _: aj.get_sprite_frame(self.SPRITES_HAMMER_DOWN, 2),
                operand=None
            ),
            lambda _: jax.lax.cond(
                state.level == 1,
                lambda _: aj.get_sprite_frame(self.SPRITES_HAMMER_DOWN, 1),
                lambda _: aj.get_sprite_frame(self.SPRITES_HAMMER_DOWN, 3),
                operand=None
            ),
            operand=None
        )
        raster = jax.lax.cond(
            state.hammer_usage_expired == False,
            lambda _: jax.lax.cond(
                state.hammer_can_hit,
                lambda _: aj.render_at(raster, state.hammer_y, state.hammer_x, frame_hammer_down),
                lambda _: aj.render_at(raster, state.hammer_y, state.hammer_x, frame_hammer),
                operand=None
            ),
            lambda _: raster,
            operand=None 
        )

        # Scores/ Timer
        show_game_score = jnp.logical_or(jnp.logical_not(state.game_started), state.mario_reached_goal)
        score = jax.lax.cond(
            show_game_score,
            lambda _: state.game_score,
            lambda _: state.game_remaining_time,
            operand=None
        )
        def create_score_in_raster(i, raster):
            digit = score // (10 ** i)
            pos_x = self.consts.FIRST_DIGIT_X - self.consts.DISTANCE_DIGIT_X * i
            pos_y = self.consts.DIGIT_Y
            return jax.lax.cond(
                show_game_score == True,
                lambda _: aj.render_at(raster, pos_x, pos_y, aj.get_sprite_frame(self.SPRITES_BLUE_DIGITS, digit)),
                lambda _: aj.render_at(raster, pos_x, pos_y, aj.get_sprite_frame(self.SPRITES_YELLOW_DIGITS, digit)),
                operand=None
            )
        raster = jax.lax.cond(
            show_game_score == True,
            lambda x: jax.lax.fori_loop(0, self.consts.NUMBER_OF_DIGITS_FOR_GAME_SCORE, create_score_in_raster, x),
            lambda x: jax.lax.fori_loop(0, self.consts.NUMBER_OF_DIGITS_FOR_TIMER_SCORE, create_score_in_raster, x),
            raster
        )

        return raster.astype(jnp.uint8)
