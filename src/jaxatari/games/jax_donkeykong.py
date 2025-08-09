import os
from functools import partial
from typing import NamedTuple, Tuple
import jax.lax
import jax.numpy as jnp
import chex
import pygame
from gymnax.environments import spaces

from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as aj
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action


class DonkeyKongConstants(NamedTuple):
    # Screen dimensions
    WIDTH: int = 160
    HEIGHT: int = 210
    WINDOW_WIDTH: int = 160 * 3
    WINDOW_HEIGHT: int = 210 * 3

    # Donkey Kong position
    DONKEYKONG_X: int = 33
    DONKEYKONG_Y: int = 14

    # Girlfriend position
    GIRLFRIEND_X: int = 62
    GIRLFRIEND_Y: int = 17

    # Life Bar positions
    LEVEL_1_LIFE_BAR_1_X: int = 116
    LEVEL_1_LIFE_BAR_2_X: int = 124
    LEVEL_2_LIFE_BAR_1_X: int = 112
    LEVEL_2_LIFE_BAR_2_X: int = 120
    LIFE_BAR_Y: int = 23

    # Hammer default position
    LEVEL_1_HAMMER_X: int = 39
    LEVEL_1_HAMMER_Y: int = 68
    LEVEL_2_HAMMER_X: int = 78
    LEVEL_2_HAMMER_Y: int = 68

    # Drop Pit positions
    DP_LEFT_X: int = 52
    DP_RIGHT_X: int = 104
    DP_FLOOR_2_Y: int = 144
    DP_FLOOR_3_Y: int = 116
    DP_FLOOR_4_Y: int = 88
    DP_FLOOR_5_Y: int = 60

    # Digits position
    DIGIT_Y: int = 7
    FIRST_DIGIT_X: int = 96
    DISTANCE_DIGIT_X: int = 8
    NUMBER_OF_DIGITS_FOR_GAME_SCORE: int = 6
    NUMBER_OF_DIGITS_FOR_TIMER_SCORE: int = 4

    # Mario movement and physics
    MARIO_START_X: float = 176.0
    MARIO_START_Y: float = 45.0
    MARIO_JUMPING_HEIGHT: float = 5.0
    MARIO_JUMPING_FRAME_DURATION: int = 33
    MARIO_MOVING_SPEED: float = 0.335  # pixels per frame
    MARIO_WALKING_ANIMATION_CHANGE_DURATION: int = 5
    MARIO_CLIMBING_SPEED: float = 0.333
    MARIO_CLIMBING_ANIMATION_CHANGE_DURATION: int = 12

    # Mario sprite indexes
    MARIO_WALK_SPRITE_0: int = 0
    MARIO_WALK_SPRITE_1: int = 1
    MARIO_WALK_SPRITE_2: int = 2
    MARIO_WALK_SPRITE_3: int = 3

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
    BASE_PROBABILITY_BARREL_ROLLING_A_LADDER_DOWN: float = 0.8

    # Hit boxes
    MARIO_HIT_BOX_X: int = 15
    MARIO_HIT_BOX_Y: int = 7
    BARREL_HIT_BOX_X: int = 8
    BARREL_HIT_BOX_Y: int = 8

    # Movement directions
    MOVING_UP: int = 0
    MOVING_RIGHT: int = 1
    MOVING_DOWN: int = 2
    MOVING_LEFT: int = 3

    # Bar start/end positions
    BAR_LEFT_Y: int = 32
    BAR_RIGHT_Y: int = 120
    BAR_1_LEFT_X: int = 185 +8
    BAR_1_RIGHT_X: int = 185 +8
    BAR_2_LEFT_X: int = 157 +8
    BAR_2_RIGHT_X: int = 164 +8
    BAR_3_LEFT_X: int = 136 +8
    BAR_3_RIGHT_X: int = 129 +8
    BAR_4_LEFT_X: int = 101 +8
    BAR_4_RIGHT_X: int = 108 +8
    BAR_5_LEFT_X: int = 80 +8
    BAR_5_RIGHT_X: int = 73 +8
    BAR_6_LEFT_X: int = 52 +8
    BAR_6_RIGHT_X: int = 52 +8

    # Ladder
    LADDER_WIDTH: int = 4

    # Barrel spawn timing
    SPAWN_STEP_COUNTER_BARREL: int = 236

class InvisibleWallEachStage(NamedTuple):
    stage: chex.Array
    left_end: chex.Array
    right_end: chex.Array

class Ladder(NamedTuple):
    stage: chex.Array
    climbable: chex.Array
    start_x: chex.Array
    start_y: chex.Array
    end_x: chex.Array
    end_y: chex.Array

class BarrelPosition(NamedTuple):
    barrel_x: chex.Array
    barrel_y: chex.Array
    sprite: chex.Array
    moving_direction: chex.Array
    stage: chex.Array
    reached_the_end: chex.Array

class DonkeyKongState(NamedTuple):
    game_started: bool
    level: chex.Array
    step_counter: chex.Array
    
    mario_x: float
    mario_y: float  
    mario_jumping: bool   # jumping on spot
    mario_jumping_wide: bool
    mario_climbing: bool
    start_frame_when_mario_jumped: int
    mario_view_direction: int
    mario_walk_frame_counter: int
    mario_climb_frame_counter: int
    mario_walk_sprite: int
    mario_climb_sprite: int
    mario_stage: int

    barrels: BarrelPosition
    ladders: Ladder
    invisibleWallEachStage: InvisibleWallEachStage
    random_key: int
    frames_since_last_barrel_spawn: int

class DonkeyKongObservation(NamedTuple):
    total_score: jnp.ndarray

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

    # Bars as lienar functions
    def bar_linear_equation(self, stage, y):
        y_1 = self.consts.BAR_LEFT_Y
        y_2 = self.consts.BAR_RIGHT_Y

        x_1_values = [self.consts.BAR_1_LEFT_X, self.consts.BAR_2_LEFT_X, self.consts.BAR_3_LEFT_X, self.consts.BAR_4_LEFT_X, self.consts.BAR_5_LEFT_X, self.consts.BAR_6_LEFT_X]
        x_2_values = [self.consts.BAR_1_RIGHT_X, self.consts.BAR_2_RIGHT_X, self.consts.BAR_3_RIGHT_X, self.consts.BAR_4_RIGHT_X, self.consts.BAR_5_RIGHT_X, self.consts.BAR_6_RIGHT_X]

        index = stage - 1
        branches = [lambda _, v=val: jnp.array(v) for val in x_1_values]
        x_1 = jax.lax.switch(index, branches, operand=None)
        branches = [lambda _, v=val: jnp.array(v) for val in x_2_values]
        x_2 = jax.lax.switch(index, branches, operand=None)

        m = (x_2 - x_1) / (y_2 - y_1)
        b = x_1 - m * y_1

        x = m * y + b
        return x

    @partial(jax.jit, static_argnums=(0,))
    def init_ladders_for_level(self, level: int) -> Ladder:
        # Ladder positions for level 1
        Ladder_level_1 = Ladder(
            stage=jnp.array([5, 5, 4, 4, 4, 3, 3, 3, 2, 2, 1, 1], dtype=jnp.int32),
            climbable=jnp.array([False, True, True, True, False, False, True, True, True, True, False, True]),
            start_x=jnp.array([77, 74, 102, 104, 106, 134, 132, 130, 158, 161, 185, 185], dtype=jnp.int32),
            start_y=jnp.array([74, 106, 46, 66, 98, 62, 86, 106, 46, 78, 70, 106], dtype=jnp.int32),
            end_x=jnp.array([53, 53, 79, 78, 76, 104, 106, 108, 135, 133, 161, 164], dtype=jnp.int32),
            end_y=jnp.array([74, 106, 46, 66, 98, 62, 86, 106, 46, 78, 70, 106], dtype=jnp.int32),
        )

        # Ladder positions for level 2
        Ladder_level_2 = Ladder_level_1

        return jax.lax.cond(
            level == 1,
            lambda _: Ladder_level_1,
            lambda _: Ladder_level_2,
            operand=None
        )

    @partial(jax.jit, static_argnums=(0,))
    def init_invisible_wall_for_level(self, level: int) -> InvisibleWallEachStage:
        # Set invisible wall depending of level
        invisible_wall_level_1 = InvisibleWallEachStage(
            stage=jnp.array([6, 5, 4, 3, 2, 1], dtype=jnp.int32),
            left_end=jnp.array([32, 37, 32, 37, 32, 37], dtype=jnp.int32),
            right_end=jnp.array([113, 120, 113, 120, 113, 120], dtype=jnp.int32),
        )

        invisible_wall_level_2 = invisible_wall_level_1

        return jax.lax.cond(
            level == 1,
            lambda _: invisible_wall_level_1,
            lambda _: invisible_wall_level_2,
            operand=None
        )


    @partial(jax.jit, static_argnums=(0,))
    def _barrel_step(self, state):
        step_counter = state.step_counter
        
        # pick other sprite for animation after 8 frames
        should_pick_next_sprite = step_counter % 8 == 0
        
        new_state = state
        # calculate new position
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
            def change_x_if_barrel_is_falling(x, y, direction, sprite, stage):
                new_x = x + 2

                bar_x = jnp.round(self.bar_linear_equation(stage, y) - self.consts.BARREL_HIT_BOX_X).astype(int)
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
                    direction == self.consts.MOVING_DOWN,
                    lambda _: (new_x, y, new_direction, new_sprite, stage),
                    lambda _: (x, y, direction, sprite, stage),
                    operand=None
                )
            x, y, direction, sprite, stage = change_x_if_barrel_is_falling(x, y, direction, sprite, stage)

            # change position
            # check if barrel can fall (ladder or end of bar)
            def check_if_barrel_will_fall(x, y, direction, sprite, stage):        
                prob_barrel_rolls_down_a_ladder = self.consts.BASE_PROBABILITY_BARREL_ROLLING_A_LADDER_DOWN
                
                # check first if barrel is positioned on top of a ladder
                curr_stage = stage - 1
                mask = jnp.logical_and(ladders.stage == curr_stage, ladders.end_y == y)
                barrel_is_on_ladder = jnp.any(mask)
                roll_down_prob = jax.random.bernoulli(state.random_key, prob_barrel_rolls_down_a_ladder)

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
                    jnp.logical_and(barrel_is_on_ladder, jnp.logical_and(direction != self.consts.MOVING_DOWN, roll_down_prob)),
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
            def barrel_rolling_on_a_bar(x, y, direction, sprite, stage):
                new_y = jax.lax.cond(
                    direction == self.consts.MOVING_RIGHT,
                    lambda _: y + 1,
                    lambda _: y - 1,
                    operand=None
                )
                new_x = jnp.round(self.bar_linear_equation(stage, new_y) - self.consts.BARREL_HIT_BOX_X).astype(int)
                return jax.lax.cond(
                    direction != self.consts.MOVING_DOWN,
                    lambda _: (new_x, new_y, direction, sprite, stage),
                    lambda _: (x, y, direction, sprite, stage),
                    operand=None
                )
            x, y, direction, sprite, stage = barrel_rolling_on_a_bar(x, y, direction, sprite, stage)

            # mark x = y = -1 as a barrel reaches the end
            def mark_barrel_if_cheached_end(x, y, direction, sprite, stage, reached_the_end):
                return jax.lax.cond(
                    jnp.logical_and(stage == 1, y <= self.consts.BAR_LEFT_Y),
                    lambda _: (-1, -1, direction, sprite, stage, True),
                    lambda _: (x, y, direction, sprite, stage, reached_the_end),
                    operand=None
                )
            x, y, direction, sprite, stage, reached_the_end = mark_barrel_if_cheached_end(x, y, direction, sprite, stage, reached_the_end)

            return jax.lax.cond(
                reached_the_end == False,
                lambda _: (x, y, direction, sprite, stage, reached_the_end),
                lambda _: (-1, -1, direction, sprite, stage, reached_the_end),
                operand=None
            )
        update_all_barrels = jax.vmap(update_single_barrel)

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
                jnp.logical_and(state.frames_since_last_barrel_spawn >= self.consts.SPAWN_STEP_COUNTER_BARREL, idx != -1),
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
        # need some frame skip here because mario is much slower than fires and barrels
        

        # there are multiple action which mario/player can execute

        # Jumping with Action.FIRE
        # one things needs to be considered --> While mario is jumping --> Action.FIRE does nothing
        def jumping_on_spot(state):
            new_state = state

            # Action.FIRE
            start_frame_when_mario_jumped = state.step_counter
            mario_jumping = True
            mario_x = state.mario_x - self.consts.MARIO_JUMPING_HEIGHT
            new_state = new_state._replace(
                start_frame_when_mario_jumped=start_frame_when_mario_jumped,
                mario_jumping=mario_jumping,
                mario_x=mario_x,
            )

            return jax.lax.cond(
                jnp.logical_and(action == Action.FIRE, jnp.logical_and(jnp.logical_and(state.mario_climbing == False, state.mario_jumping == False), state.mario_jumping_wide == False)),
                lambda _: new_state,
                lambda _: state,
                operand=None
            )
        new_state = jumping_on_spot(state)

        # Jumping wide with Action.LEFTFIRE and Action.RIGHTFIRE
        def jumping_right(state):
            new_state = state._replace(
                start_frame_when_mario_jumped = state.step_counter,
                mario_jumping_wide = True,
                mario_view_direction = self.consts.MOVING_RIGHT,
                mario_x = state.mario_x - self.consts.MARIO_JUMPING_HEIGHT,
                mario_y = state.mario_y + self.consts.MARIO_MOVING_SPEED
            )
            new_state_2 = state._replace(
                mario_y = state.mario_y + self.consts.MARIO_MOVING_SPEED
            )
            return jax.lax.cond(
                jnp.logical_and(action == Action.RIGHTFIRE, jnp.logical_and(jnp.logical_and(state.mario_climbing == False, state.mario_jumping_wide == False), state.mario_jumping == False)),
                lambda _: new_state,
                lambda _: jax.lax.cond(
                    jnp.logical_and(state.mario_jumping_wide == True, state.mario_view_direction == self.consts.MOVING_RIGHT),
                    lambda _: new_state_2,
                    lambda _: state,
                    operand=None
                ),
                operand=None
            )
        new_state = jumping_right(new_state)

        def jumping_left(state):
            new_state = state._replace(
                start_frame_when_mario_jumped = state.step_counter,
                mario_jumping_wide = True,
                mario_view_direction = self.consts.MOVING_LEFT,
                mario_x = state.mario_x - self.consts.MARIO_JUMPING_HEIGHT,
                mario_y = state.mario_y - self.consts.MARIO_MOVING_SPEED
            )
            new_state_2 = state._replace(
                mario_y = state.mario_y - self.consts.MARIO_MOVING_SPEED
            )
            return jax.lax.cond(
                jnp.logical_and(action == Action.LEFTFIRE, jnp.logical_and(jnp.logical_and(state.mario_climbing == False, state.mario_jumping_wide == False), state.mario_jumping == False)),
                lambda _: new_state,
                lambda _: jax.lax.cond(
                    jnp.logical_and(state.mario_jumping_wide == True, state.mario_view_direction == self.consts.MOVING_LEFT),
                    lambda _: new_state_2,
                    lambda _: state,
                    operand=None
                ),
                operand=None
            )
        new_state = jumping_left(new_state)

        # reset jumping after a certain time
        def reset_jumping(state):
            new_state = state._replace(
                mario_jumping = False,
                mario_jumping_wide = False,
                mario_x = state.mario_x + self.consts.MARIO_JUMPING_HEIGHT,
            )

            return jax.lax.cond(
                jnp.logical_and(state.step_counter - state.start_frame_when_mario_jumped >= self.consts.MARIO_JUMPING_FRAME_DURATION, jnp.logical_or(state.mario_jumping == True, state.mario_jumping_wide == True)),
                lambda _: new_state,
                lambda _: state,
                operand=None
            )
        new_state = reset_jumping(new_state)

        # mario climbing ladder
        # precondition, mario is already climbing --> function for STARTING climbing below
        def mario_climbing(state):
            # normal climbing upwards
            new_state_climbing_upwards = jax.lax.cond(
                state.mario_climb_frame_counter % self.consts.MARIO_CLIMBING_ANIMATION_CHANGE_DURATION == 0,
                lambda _: jax.lax.cond(
                    state.mario_climb_sprite == self.consts.MARIO_CLIMB_SPRITE_0,
                    lambda _: state._replace(
                        mario_x=state.mario_x - self.consts.MARIO_CLIMBING_SPEED,
                        mario_climb_frame_counter= state.mario_climb_frame_counter + 1,
                        mario_climb_sprite=self.consts.MARIO_CLIMB_SPRITE_1
                    ),
                    lambda _: state._replace(
                        mario_x=state.mario_x - self.consts.MARIO_CLIMBING_SPEED,
                        mario_climb_frame_counter= state.mario_climb_frame_counter + 1,
                        mario_climb_sprite=self.consts.MARIO_CLIMB_SPRITE_0
                    ),
                    operand=None
                ),
                lambda _: state._replace(
                    mario_x=state.mario_x - self.consts.MARIO_CLIMBING_SPEED,
                    mario_climb_frame_counter= state.mario_climb_frame_counter + 1,
                ),
                operand=None
            )

            # check if mario finished climbing / reached the end of a ladder
            reached_top = new_state_climbing_upwards.mario_x <= jnp.round(self.bar_linear_equation(new_state_climbing_upwards.mario_stage + 1, new_state_climbing_upwards.mario_y) - self.consts.MARIO_HIT_BOX_X).astype(int)
            new_state_climbing_upwards = jax.lax.cond(
                reached_top,
                lambda _: state._replace(
                    mario_x = state.mario_x - 2,
                    mario_climb_frame_counter= 0,
                    mario_view_direction = self.consts.MOVING_RIGHT,
                    mario_climbing = False,
                    mario_stage = state.mario_stage + 1,
                    mario_walk_frame_counter = 0,
                    mario_walk_sprite = self.consts.MARIO_WALK_SPRITE_0,
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
                        mario_x=state.mario_x + self.consts.MARIO_CLIMBING_SPEED,
                        mario_climb_frame_counter= state.mario_climb_frame_counter + 1,
                        mario_climb_sprite=self.consts.MARIO_CLIMB_SPRITE_1
                    ),
                    lambda _: state._replace(
                        mario_x=state.mario_x + self.consts.MARIO_CLIMBING_SPEED,
                        mario_climb_frame_counter= state.mario_climb_frame_counter + 1,
                        mario_climb_sprite=self.consts.MARIO_CLIMB_SPRITE_0
                    ),
                    operand=None
                ),
                lambda _: state._replace(
                    mario_x=state.mario_x + self.consts.MARIO_CLIMBING_SPEED,
                    mario_climb_frame_counter= state.mario_climb_frame_counter + 1,
                ),
                operand=None
            )

            # check if mario finished climbing / reached the start of a ladder
            reached_bottom = new_state_climbing_downwards.mario_x >= jnp.round(self.bar_linear_equation(new_state_climbing_downwards.mario_stage, new_state_climbing_downwards.mario_y) - self.consts.MARIO_HIT_BOX_X).astype(int)
            new_state_climbing_downwards = jax.lax.cond(
                reached_bottom,
                lambda _: state._replace(
                    mario_x = state.mario_x - 2,
                    mario_climb_frame_counter= 0,
                    mario_view_direction = self.consts.MOVING_RIGHT,
                    mario_climbing = False,
                    mario_walk_frame_counter = 0,
                    mario_walk_sprite = self.consts.MARIO_WALK_SPRITE_0,
                ),
                lambda _: new_state_climbing_downwards,
                operand=None
            )

            return jax.lax.cond(
                jnp.logical_and(jnp.logical_and(jnp.logical_and(state.mario_climbing == True, state.mario_jumping == False), state.mario_jumping_wide == False), action == Action.UP),
                lambda _: new_state_climbing_upwards,
                lambda _: jax.lax.cond(
                    jnp.logical_and(jnp.logical_and(jnp.logical_and(state.mario_climbing == True, state.mario_jumping == False), state.mario_jumping_wide == False), action == Action.DOWN),
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
                mario_x=state.mario_x + 1,
                mario_climbing=True,
                mario_climb_frame_counter=0,
            )
            new_state_climbing_downwards = state._replace(
                mario_view_direction=self.consts.MOVING_UP,
                mario_stage = state.mario_stage - 1,
                mario_x=state.mario_x + 3,
                mario_climbing=True,
                mario_climb_frame_counter=0,
            )
            ladders = state.ladders # be careful, ladder is not the actual ladder positions but where barrel interact with the ladders

            def look_for_valid_ladder_to_climb(i, value):
                mario_can_climb = value[0]
                mario_stage = value[1]
                current_ladder_climbable = True
                current_ladder_climbable &= mario_stage == ladders.stage[i]
                current_ladder_climbable &= state.mario_y <= ladders.start_y[i]
                current_ladder_climbable &= state.mario_y + self.consts.MARIO_HIT_BOX_Y - 1 > ladders.start_y[i] + self.consts.LADDER_WIDTH

                return jax.lax.cond(
                    mario_can_climb,
                    lambda _: (True, mario_stage),
                    lambda _: (current_ladder_climbable, mario_stage),
                    operand=None
                )
            mario_can_climb_upwards = jax.lax.fori_loop(0, len(ladders.stage), look_for_valid_ladder_to_climb, (False, state.mario_stage))[0]  
            mario_can_climb_downwards = jax.lax.fori_loop(0, len(ladders.stage), look_for_valid_ladder_to_climb, (False, state.mario_stage - 1))[0]        
            
            return jax.lax.cond(
                jnp.logical_and(jnp.logical_and(jnp.logical_and(jnp.logical_and(mario_can_climb_upwards, state.mario_climbing == False), state.mario_jumping == False), state.mario_jumping_wide == False), action == Action.UP),
                lambda _: new_state_climbing_upwards,
                lambda _: jax.lax.cond(
                    jnp.logical_and(jnp.logical_and(jnp.logical_and(jnp.logical_and(mario_can_climb_downwards, state.mario_climbing == False), state.mario_jumping == False), state.mario_jumping_wide == False), action == Action.DOWN),
                    lambda _: new_state_climbing_downwards,
                    lambda _: state,
                    operand=None
                ),
                operand=None
            )
        new_state = mario_starts_climbing(new_state)

        # change mario position in x direction if Action.right or Action.left is chosen
        def mario_walking_to_right(state):
            last_mario_move_was_moving_to_right = state.mario_view_direction != self.consts.MOVING_RIGHT
            new_mario_x = jnp.round(self.bar_linear_equation(state.mario_stage, state.mario_y) - self.consts.MARIO_HIT_BOX_X) - 2
            
            new_state = jax.lax.cond(
                last_mario_move_was_moving_to_right,
                lambda _:  state._replace(
                    mario_x = new_mario_x,
                    mario_y=state.mario_y + self.consts.MARIO_MOVING_SPEED,
                    mario_view_direction=self.consts.MOVING_RIGHT,
                    mario_walk_frame_counter=0,
                ),
                lambda _:state._replace(
                    mario_x = new_mario_x,
                    mario_y=state.mario_y + self.consts.MARIO_MOVING_SPEED,
                    mario_view_direction=self.consts.MOVING_RIGHT,
                    mario_walk_frame_counter=state.mario_walk_frame_counter + 1,
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
            return jax.lax.cond(
                jnp.logical_and(jnp.logical_and(jnp.logical_and(state.mario_climbing == False, state.mario_jumping == False), state.mario_jumping_wide == False), action == Action.RIGHT),
                lambda _: new_state,
                lambda _: state,
                operand=None
            )
        new_state = mario_walking_to_right(new_state)

        # similar function as mario_walking_to_right
        def mario_walking_to_left(state):
            last_mario_move_was_moving_to_left = state.mario_view_direction != self.consts.MOVING_LEFT
            new_mario_x = jnp.round(self.bar_linear_equation(state.mario_stage, state.mario_y) - self.consts.MARIO_HIT_BOX_X) - 2
            new_state = jax.lax.cond(
                last_mario_move_was_moving_to_left,
                lambda _:  state._replace(
                    mario_x = new_mario_x,
                    mario_y=state.mario_y - self.consts.MARIO_MOVING_SPEED,
                    mario_view_direction=self.consts.MOVING_LEFT,
                    mario_walk_frame_counter=0,
                ),
                lambda _:state._replace(
                    mario_x = new_mario_x,
                    mario_y=state.mario_y - self.consts.MARIO_MOVING_SPEED,
                    mario_view_direction=self.consts.MOVING_LEFT,
                    mario_walk_frame_counter=state.mario_walk_frame_counter + 1,
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
            return jax.lax.cond(
                jnp.logical_and(jnp.logical_and(jnp.logical_and(state.mario_climbing == False, state.mario_jumping == False), state.mario_jumping_wide == False), action == Action.LEFT),
                lambda _: new_state,
                lambda _: state,
                operand=None
            )
        new_state = mario_walking_to_left(new_state)

        return new_state
    

    def reset(self, key = [0,0]) -> Tuple[DonkeyKongObservation, DonkeyKongState]:
        """
        Resets the game state to the initial state.
        Returns the initial state and the reward (i.e. 0)
        """
        ladders = self.init_ladders_for_level(level=1)
        invisibleWallEachStage = self.init_invisible_wall_for_level(level=1)
        state = DonkeyKongState(
            game_started = False,
            level = 1,
            step_counter=jnp.array(1).astype(jnp.int32),
            frames_since_last_barrel_spawn=jnp.array(0).astype(jnp.int32),

            mario_x=self.consts.MARIO_START_X,
            mario_y=self.consts.MARIO_START_Y,
            mario_jumping=False,
            mario_jumping_wide=False,
            mario_climbing=False,
            start_frame_when_mario_jumped=-1,
            mario_view_direction=self.consts.MOVING_RIGHT,
            mario_walk_frame_counter=0,
            mario_climb_frame_counter=0,
            mario_walk_sprite=self.consts.MARIO_WALK_SPRITE_0,
            mario_climb_sprite=self.consts.MARIO_CLIMB_SPRITE_0,
            mario_stage=1,

            barrels = BarrelPosition(
                barrel_x = jnp.array([-1, -1, -1, -1]).astype(jnp.int32),
                barrel_y = jnp.array([-1, -1, -1, -1]).astype(jnp.int32), 
                sprite = jnp.array([self.consts.BARREL_SPRITE_RIGHT, self.consts.BARREL_SPRITE_RIGHT, self.consts.BARREL_SPRITE_RIGHT, self.consts.BARREL_SPRITE_RIGHT]).astype(jnp.int32),
                moving_direction = jnp.array([self.consts.MOVING_RIGHT, self.consts.MOVING_RIGHT, self.consts.MOVING_RIGHT, self.consts.MOVING_RIGHT]).astype(jnp.int32),
                stage = jnp.array([6, 6, 6, 6]).astype(jnp.int32),
                reached_the_end=jnp.array([True, True, True, True]).astype(bool)
            ),

            ladders=ladders,
            invisibleWallEachStage=invisibleWallEachStage,
            random_key = jax.random.PRNGKey(key[0]),
        )
        initial_obs = self._get_observation(state)

        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: DonkeyKongState, action: chex.Array) -> Tuple[DonkeyKongObservation, DonkeyKongState, float, bool, DonkeyKongInfo]:
        # First search for colision

        new_state = state

        # If there is no colision: game will continue
        # enemy_step --> maybe later write a enemy_step function which calls eighter barrel_step oder fire_step
        new_state = self._barrel_step(state)
  
        # mario step / player step
        new_state = self._mario_step(new_state, action)

        # increase timer / frame counter
        new_state = new_state._replace(
            step_counter=new_state.step_counter+1,
            frames_since_last_barrel_spawn=new_state.frames_since_last_barrel_spawn+1,
        )
        
        # Check if game was even started --> with human_action FIRE
        def start_game():
            started_state = state._replace(
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
            return jax.lax.cond(
                action == Action.FIRE,
                lambda _: started_state,
                lambda _: state,
                operand=None
            )
        new_state = jax.lax.cond(
            state.game_started == False,
            lambda _: start_game(),
            lambda _: new_state,
            operand=None
        )

        observation = self._get_observation(new_state)
        return observation, new_state, 0, False, None

    
    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: DonkeyKongState):
        
        return DonkeyKongObservation(
            total_score = 0,
        )

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(8)


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

            # some drop pits might be already triggered - in that case, drop pits at those position will not be rendered
            raster = aj.render_at(raster, self.consts.DP_LEFT_X, self.consts.DP_FLOOR_2_Y, frame_drop_pit)
            raster = aj.render_at(raster, self.consts.DP_LEFT_X, self.consts.DP_FLOOR_3_Y, frame_drop_pit)
            raster = aj.render_at(raster, self.consts.DP_LEFT_X, self.consts.DP_FLOOR_4_Y, frame_drop_pit)
            raster = aj.render_at(raster, self.consts.DP_LEFT_X, self.consts.DP_FLOOR_5_Y, frame_drop_pit)
            raster = aj.render_at(raster, self.consts.DP_RIGHT_X, self.consts.DP_FLOOR_2_Y, frame_drop_pit)
            raster = aj.render_at(raster, self.consts.DP_RIGHT_X, self.consts.DP_FLOOR_3_Y, frame_drop_pit)
            raster = aj.render_at(raster, self.consts.DP_RIGHT_X, self.consts.DP_FLOOR_4_Y, frame_drop_pit)
            raster = aj.render_at(raster, self.consts.DP_RIGHT_X, self.consts.DP_FLOOR_5_Y, frame_drop_pit)
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
        frame_donkeyKong = aj.get_sprite_frame(self.SPRITES_DONKEYKONG, 0)
        raster = aj.render_at(raster, self.consts.DONKEYKONG_X, self.consts.DONKEYKONG_Y, frame_donkeyKong)

        # Girlfriend
        frame_girlfriend = aj.get_sprite_frame(self.SPRITE_GIRLFRIEND, 0)
        raster = aj.render_at(raster, self.consts.GIRLFRIEND_X, self.consts.GIRLFRIEND_Y, frame_girlfriend)

        # Life Bars - depending if lifes are still given 
        frame_life_bar = aj.get_sprite_frame(self.SPRITES_LIFEBAR, 0)
        raster = aj.render_at(raster, self.consts.LEVEL_1_LIFE_BAR_1_X, self.consts.LIFE_BAR_Y, frame_life_bar)
        raster = aj.render_at(raster, self.consts.LEVEL_1_LIFE_BAR_2_X, self.consts.LIFE_BAR_Y, frame_life_bar)

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

        # Hammer
        frame_hammer = aj.get_sprite_frame(self.SPRITES_HAMMER_UP, 0)
        raster = aj.render_at(raster, self.consts.LEVEL_1_HAMMER_X, self.consts.LEVEL_1_HAMMER_Y, frame_hammer)


        # Scores
        score = 5000
        show_game_score = False
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

        # Barrels - example for now
        frame_barrel = aj.get_sprite_frame(self.SPRITES_BARREL, 0)
        raster = aj.render_at(raster, 5, 5, frame_barrel)
        frame_barrel = aj.get_sprite_frame(self.SPRITES_BARREL, 1)
        raster = aj.render_at(raster, 5, 15, frame_barrel)
        frame_barrel = aj.get_sprite_frame(self.SPRITES_BARREL, 2)
        raster = aj.render_at(raster, 5, 25, frame_barrel)

        # Mario - example raster on the left of the screen
        frame_mario = aj.get_sprite_frame(self.SPRITES_MARIO_STANDING, 0)
        raster = aj.render_at(raster, 5, 35, frame_mario)
        frame_mario = aj.get_sprite_frame(self.SPRITES_MARIO_STANDING, 1)
        raster = aj.render_at(raster, 15, 35, frame_mario)

        frame_mario = aj.get_sprite_frame(self.SPRITES_MARIO_JUMPING, 0)
        raster = aj.render_at(raster, 5, 55, frame_mario)
        frame_mario = aj.get_sprite_frame(self.SPRITES_MARIO_JUMPING, 1)
        raster = aj.render_at(raster, 15, 55, frame_mario)

        frame_mario = aj.get_sprite_frame(self.SPRITES_MARIO_WALKING_1, 0)
        raster = aj.render_at(raster, 5, 75, frame_mario)
        frame_mario = aj.get_sprite_frame(self.SPRITES_MARIO_WALKING_1, 1)
        raster = aj.render_at(raster, 15, 75, frame_mario)

        frame_mario = aj.get_sprite_frame(self.SPRITES_MARIO_WALKING_2, 0)
        raster = aj.render_at(raster, 5, 95, frame_mario)
        frame_mario = aj.get_sprite_frame(self.SPRITES_MARIO_WALKING_2, 1)
        raster = aj.render_at(raster, 15, 95, frame_mario)

        frame_mario = aj.get_sprite_frame(self.SPRITES_MARIO_CLIMBING, 0)
        raster = aj.render_at(raster, 5, 115, frame_mario)
        frame_mario = aj.get_sprite_frame(self.SPRITES_MARIO_CLIMBING, 1)
        raster = aj.render_at(raster, 15, 115, frame_mario)

        # Hammer down examples
        frame_hammer_down = aj.get_sprite_frame(self.SPRITES_HAMMER_DOWN, 0)
        raster = aj.render_at(raster, 5, 135, frame_hammer_down)
        frame_hammer_down = aj.get_sprite_frame(self.SPRITES_HAMMER_DOWN, 1)
        raster = aj.render_at(raster, 15, 135, frame_hammer_down)
        frame_hammer_up_level_2 = aj.get_sprite_frame(self.SPRITES_HAMMER_UP, 1)
        raster = aj.render_at(raster, 5, 145, frame_hammer_up_level_2)
        frame_hammer_down = aj.get_sprite_frame(self.SPRITES_HAMMER_DOWN, 2)
        raster = aj.render_at(raster, 5, 155, frame_hammer_down)
        frame_hammer_down = aj.get_sprite_frame(self.SPRITES_HAMMER_DOWN, 3)
        raster = aj.render_at(raster, 15, 155, frame_hammer_down)

        # Fire
        frame_fire = aj.get_sprite_frame(self.SPRITE_FIRE, 0)
        raster = aj.render_at(raster, 5, 165, frame_fire)

        return raster
