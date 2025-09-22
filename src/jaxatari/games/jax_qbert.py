#! /usr/bin/python3
# -*- coding: utf-8 -*-
#
# JAX Q*bert
#
# Simulates the Atari Q*bert game
#
# Authors:
# - Xarion99
# - Keksmo
# - Embuer
# - Snocember
from typing import NamedTuple, Tuple
from functools import partial

import os
import chex
import jax
import jax.numpy as jnp

from jaxatari.environment import JAXAtariAction as Action
from jaxatari.environment import JaxEnvironment
from jaxatari.spaces import Space, Discrete, Box, Dict
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as aj


class QbertConstants(NamedTuple):
    WIDTH = 160
    HEIGHT = 210
    DIFFICULTY = 1
    """ Changes the difficulty of the game: 0 = no red balls will spawn, 1 = red balls will spawn """
    ENEMY_MOVE_TICK = jnp.array([65, 60, 55, 50, 45]).astype(jnp.int32)
    """ The number of ticks it take to move an enemy according to the current level """
    #TODO

class QbertState(NamedTuple):
    player_score: chex.Numeric
    """ The player_score """
    lives: chex.Numeric
    """ The number of qberts left over """
    pyramid: chex.Array
    """ A 8x8 matrix representing the color of every cube in the pyramid: 0 = start color, 1 = intermediate color, 2 = destination color
    [
             []
             [0]
            [0, 1]
           [0, 1, 2]
          [0, 1, 2, 3]
         [0, 1, 2, 3, 4]
        [0, 1, 2, 3, 4, 5]
        []
    ] 
    The outer space of the pyramid is marked with -1 and flying discs are marked with -2
    [
        [-1, -1, -1, -1, -1, -1, -1, -1]
        [-1,  0, -1, -1, -1, -1, -1, -1]
        [-1,  0,  1, -1, -1, -1, -1, -1]
        [-1,  0,  1,  2, -1, -1, -1, -1]
        [-1,  0,  1,  2,  3, -1, -1, -1]
        [-1,  0,  1,  2,  3,  4, -1, -1]
        [-1,  0,  1,  2,  3,  4,  5, -1]
        [-1, -1, -1, -1, -1, -1, -1, -1]
    ]
    """
    last_pyramid: chex.Array
    """ An array representing the last look of the pyramid """
    player_position: chex.Array
    """ An array representing the position of the player: (x, y) """
    player_last_position: chex.Array
    """ An array representing the last position of the player: (x, y) """
    player_direction: chex.Numeric
    """ The direction the player is looking: 0 = left_up, 1 = left_down, 2 = right_down, 3 = right_up """
    is_player_moving: chex.Numeric
    """ Tells whether the player is currently moving: 0 = no, 1 = yes """
    player_moving_counter: chex.Numeric
    """ The current step of the player moving animation """
    player_position_category: chex.Numeric
    """ Gives information of the type of the field the player is moving to: 0 = normal field, 1 = disc bottom, 2 = disc top, 3 = out of pyramid high 1, 4 = out of pyramid high 2, 5 = out of pyramid high 3, 6 = out of pyramid high 4, 7 = out of pyramid high 5, 8 = out of pyramid high 6 """
    level_number: chex.Numeric
    """ The number of the current level """
    round_number: chex.Numeric
    """ The number of the current round """
    green_ball_freeze_step: chex.Numeric
    """ The step number to which the enemies are frozen """
    enemy_moving_counter: chex.Numeric
    """ The current step of the enemies moving animation """
    red_ball_positions: chex.Array
    """ The positions of the red balls as a matrix ([-1, -1] if currently not on the pyramid)
    [
        [x, y]
        [x, y]
        [x, y]
    ]
    """
    purple_ball_position: chex.Array
    """ The positions of the purple balls as a matrix ([-1, -1] if currently not on the pyramid)
    [x, y]
    """
    snake_position: chex.Array
    """ The positions of the snakes as a matrix ([-1, -1] if currently not on the pyramid)
    [x, y]
    """
    green_ball_position: chex.Array
    """ The positions of the green balls as a matrix ([-1, -1] if currently not on the pyramid)
    [x, y]
    """
    sam_position: chex.Array
    """ The positions of the sams as a matrix ([-1, -1] if currently not on the pyramid)
    [x, y]
    """
    step_counter: chex.Numeric
    """ Counts the number of steps """

    just_spawned: chex.Numeric

    snake_lock: chex.Array
    #TODO

    #action_mapping: chex.Array

    #the state used for the random number generator
    prng_state: chex.Numeric

class QbertObservation(NamedTuple):
    player_score: jnp.ndarray
    lives: jnp.ndarray
    pyramid: jnp.ndarray
    player_position: jnp.ndarray
    level_number: jnp.ndarray
    round_number: jnp.ndarray
    red_ball_positions: jnp.ndarray
    purple_ball_position: jnp.ndarray
    snake_position: jnp.ndarray
    green_ball_position: jnp.ndarray
    sam_position: jnp.ndarray
    #TODO

class QbertInfo(NamedTuple):
    time: jnp.ndarray
    all_rewards: jnp.ndarray

class JaxQbert(JaxEnvironment[QbertState, QbertObservation, QbertInfo, QbertConstants]):
    def __init__(self, consts: QbertConstants = None, reward_funcs: list[callable] = None):
        super().__init__()
        self.frame_stack_size = 4
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.action_set = [
            Action.NOOP,
            Action.UP,
            Action.DOWN,
            Action.LEFT,
            Action.RIGHT
        ]
        self.obs_size = 11 #TODO
        self.consts = consts or QbertConstants()
        self.renderer = QbertRenderer(consts)
        self.action_mapping = jnp.array([[0, 0],[0,0],[0,-1],[1,1],[-1,-1],[0,1]]).astype(jnp.int32)

    def reset(self, key=jax.random.PRNGKey(int.from_bytes(os.urandom(3), byteorder='big'))) -> Tuple[QbertObservation, QbertState]:
        state = QbertState(
            player_score=jnp.array(0).astype(jnp.int32),
            lives=jnp.array(3).astype(jnp.int32),
            pyramid=jnp.array([
                [-1, -1, -1, -1, -1, -1, -1, -1],
                [-1,  0, -1, -1, -1, -1, -1, -1],
                [-1,  0,  0, -1, -1, -1, -1, -1],
                [-1,  0,  0,  0, -1, -1, -1, -1],
                [-2,  0,  0,  0,  0, -2, -1, -1],
                [-1,  0,  0,  0,  0,  0, -1, -1],
                [-1,  0,  0,  0,  0,  0,  0, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1]
            ]).astype(jnp.int32),
            last_pyramid=jnp.array([
                [-1, -1, -1, -1, -1, -1, -1, -1],
                [-1,  0, -1, -1, -1, -1, -1, -1],
                [-1,  0,  0, -1, -1, -1, -1, -1],
                [-1,  0,  0,  0, -1, -1, -1, -1],
                [-2,  0,  0,  0,  0, -2, -1, -1],
                [-1,  0,  0,  0,  0,  0, -1, -1],
                [-1,  0,  0,  0,  0,  0,  0, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1]
            ]).astype(jnp.int32),
            player_position=jnp.array([1, 1]).astype(jnp.int32),
            player_last_position=jnp.array([1, 1]).astype(jnp.int32),
            player_direction=jnp.array(1).astype(jnp.int32),
            is_player_moving=jnp.array(0).astype(jnp.int32),
            player_moving_counter=jnp.array(0).astype(jnp.int32),
            player_position_category=jnp.array(0).astype(jnp.int32),
            level_number=jnp.array(1).astype(jnp.int32),
            round_number=jnp.array(1).astype(jnp.int32),
            green_ball_freeze_step=jnp.array(0).astype(jnp.int32),
            enemy_moving_counter=jnp.array(0).astype(jnp.int32),
            red_ball_positions=jnp.array([[-1, -1], [-1, -1], [-1, -1]]).astype(jnp.int32),
            purple_ball_position=jnp.array([-1, -1]).astype(jnp.int32),
            snake_position=jnp.array([-1, -1]).astype(jnp.int32),
            green_ball_position=jnp.array([-1, -1]).astype(jnp.int32),
            sam_position=jnp.array([-1, -1]).astype(jnp.int32),
            step_counter=jnp.array(0).astype(jnp.int32),
            prng_state=jax.random.uniform(key, (), minval=1, maxval=256).astype(jnp.uint8),
            just_spawned=jnp.array(1).astype(jnp.int32),
            snake_lock=jnp.array([-1,-1]).astype(jnp.int32),
        )

        initial_obs = self._get_observation(state)

        #TODO
        return initial_obs, state
    


    @partial(jax.jit, static_argnums=(0,))
    def checkField(self, state: QbertState, character: chex.Numeric, number: chex.Numeric, position: chex.Array):
        pyra, pos, lives, points = jax.lax.cond(
            pred=jnp.logical_and(state.pyramid[position[1]][position[0]] == -2, character == 0),
            true_fun=lambda vals: self.stepped_on_disk(vals[0]),
            false_fun=lambda vals: jax.lax.cond(
                pred=vals[0].pyramid[vals[1][1]][vals[1][0]] == -1,
                true_fun=lambda vals2: self.stepped_out_map(vals2[0],vals2[2]),
                false_fun=lambda vals2: (vals2[0].pyramid,vals2[1],vals2[0].lives,jnp.array(0).astype(jnp.int32)),
                operand=vals
            ),
            operand=(state, position, character, number)
        )
        return pyra, pos, lives, points

    @partial(jax.jit, static_argnums=(0,))
    def changeColors(self, playerPos: chex.Array, samPos: chex.Array, pyramid: chex.Array, level: chex.Numeric, spawned: chex.Numeric):
        old_color=pyramid[playerPos[1]][playerPos[0]]
        pyramid=jax.lax.cond(
            pred=jnp.logical_and(samPos[0]!= -1, samPos[1] != -1),
            true_fun=lambda vals: vals[0].at[vals[1][1],vals[1][0]].set(0),
            false_fun=lambda vals: vals[0],
            operand=(pyramid,samPos)
        )
        pyramid=jax.lax.cond(
            pred=jnp.logical_or(jnp.logical_or(playerPos[0] != 1, playerPos[1] != 1), spawned != 1),
            true_fun=lambda val:
                val[0].at[val[1][1],val[1][0]].set(jax.lax.switch(
                index=level - 1,
                branches=[
                    lambda vals: 2,
                    lambda vals: jnp.minimum(vals[0][vals[1][1]][vals[1][0]] + 1, 2),
                    lambda vals: vals[0][vals[1][1]][vals[1][0]] + 2 % 4,
                    lambda vals: jax.lax.cond(
                        pred=vals[0][vals[1][1]][vals[1][0]] == 2,
                        true_fun=lambda vals2: 1,
                        false_fun=lambda vals2: jnp.minimum(vals2 + 1, 2),
                        operand=vals[0][vals[1][1]][vals[1][0]]
                    ),
                    lambda vals: vals[0][vals[1][1]][vals[1][0]] + 1 % 3,
                ],
                operand=val)),
            false_fun=lambda val: val[0],
            operand=(pyramid,playerPos)
            )
        points=jax.lax.cond(
            pred=jnp.logical_and(pyramid[playerPos[1]][playerPos[0]] == 2, old_color != 2),
            true_fun=lambda i: 25,
            false_fun=lambda i: 0,
            operand=None
        )


        return pyramid,points
    @partial(jax.jit, static_argnums=(0,))
    def checkCollisions(self, lives: chex.Numeric, playerPos: chex.Array,
                            redPos: chex.Array, purplePos: chex.Array,
                            greenPos: chex.Array, snakePos: chex.Array,
                            samPos: chex.Array, green_ball_freeze_step: chex.Numeric):
            # Build enemies as an (8, 2) int32 array; last row is a sentinel.
            enemies = jnp.stack(
                [
                    redPos[0],         # 0
                    redPos[1],         # 1
                    redPos[2],         # 2
                    purplePos,         # 3
                    greenPos,          # 4
                    snakePos,          # 5
                    samPos,            # 6
                    jnp.array([100, 100], dtype=jnp.int32)  # 7 (sentinel)
                ],
                axis=0
            ).astype(jnp.int32)
            # Row-wise equality mask (8,): True where enemy position equals playerPos.
            mask = jnp.all(enemies == playerPos, axis=1)
            collision = jnp.any(mask)
            index_if_hit = jnp.argmax(mask).astype(jnp.int32)
            # If collision: lose a life unless friendly (index 4 or 6); else index=7 sentinel.
            def _true_fun(vals):
                enemies_, lives_, idx_ = vals
                is_friendly = (idx_ == jnp.int32(4)) | (idx_ == jnp.int32(6))
                lives_new = lives_ - jnp.where(is_friendly, jnp.int32(0), jnp.int32(1))
                return lives_new, idx_
            def _false_fun(vals):
                enemies_, lives_, idx_ = vals
                return lives_, jnp.int32(7)
            lives, index = jax.lax.cond(
                collision,
                _true_fun,
                _false_fun,
                operand=(enemies, lives, index_if_hit)
            )
            purple_before = enemies[3]    
            # Mark the collided enemy (or the sentinel row if no collision) as removed.
            enemies = enemies.at[index].set(jnp.array([-1, -1], dtype=jnp.int32))
            # Special swap/removal for enemies 3 and 5 depending on purple's y == 7.
            def _swap_true(data):
                e, p = data
                # Remove purple; move snake to purple's captured position.
                return jnp.array([-1, -1], dtype=jnp.int32), p

            def _swap_false(data):
                e, p = data
                # No change.
                return e[3], e[5]

            new3, new5 = jax.lax.cond(
                pred=(purple_before[1] == jnp.int32(6)),
                true_fun=_swap_true,
                false_fun=_swap_false,
                operand=(enemies, purple_before)
            )


            enemies = enemies.at[3].set(new3)
            enemies = enemies.at[5].set(new5)
            # Points: 100 for green (4), 300 for sam (6), else 0.
            points = jax.lax.cond(
                pred=(index == jnp.int32(4)),
                true_fun=lambda i: jnp.int32(100),
                false_fun=lambda i: jax.lax.cond(
                    pred=(i == jnp.int32(6)),
                    true_fun=lambda j: jnp.int32(300),
                    false_fun=lambda j: jnp.int32(0),
                    operand=i
                ),
                operand=index
            )

            green_ball_freeze_step = green_ball_freeze_step + 206 * (index == 4)

            return (
                lives,
                [enemies[0], enemies[1], enemies[2]],
                enemies[3],
                enemies[4],
                enemies[5],
                enemies[6],
                points,
                green_ball_freeze_step
            )


    @partial(jax.jit, static_argnums=(0,))
    def stepped_out_map(self, state: QbertState, character : chex.Numeric):
        pos,player_lives,points=jax.lax.cond(
            pred=character == 0,
            true_fun=lambda state2: (jnp.array([1,1]).astype(jnp.int32), state2[0].lives - 1, jnp.array(0).astype(jnp.int32)),
            false_fun=lambda state2: jax.lax.cond(
                pred=state[1] == 3,
                true_fun=lambda vals: (jnp.array([-1,-1]).astype(jnp.int32), 0, jnp.array(500).astype(jnp.int32)),
                false_fun=lambda vals: (jnp.array([-1,-1]).astype(jnp.int32), 0, jnp.array(0).astype(jnp.int32)),
                operand=state2
            ),
            operand=(state,character)
        )
        return state.pyramid, pos,player_lives, points


    @partial(jax.jit, static_argnums=(0,))
    def stepped_on_disk(self, state: QbertState):
        pyramid=jax.lax.cond(
            pred=state.player_position[0] > 1,
            true_fun=lambda state2: state2.pyramid.at[state2.player_position[1] - 1,state2.player_position[0]].set(-1),
            false_fun=lambda state2: state2.pyramid.at[state2.player_position[1] - 1, state2.player_position[0] - 1].set(-1),
            operand=state
        )
        return pyramid, jnp.array([1,1]).astype(jnp.int32), state.lives, jnp.array(0).astype(jnp.int32)


    @partial(jax.jit, static_argnums=(0,))
    def move(self, state: QbertState, action: chex.Array, position: chex.Array):
        new_pos = jax.lax.cond(
            pred=jnp.logical_and(position[0]!=-1, action > 1),
            true_fun=lambda vals: jnp.array([vals[2][0] + vals[0][vals[1]][0], vals[2][1] + vals[0][vals[1]][1]]).astype(jnp.int32),
            false_fun=lambda vals: jnp.array([vals[2][0], vals[2][1]]).astype(jnp.int32),
            operand=(self.action_mapping, action, position)
        )

        return new_pos

    @partial(jax.jit, static_argnums=(0,))
    def move_purple_ball(self,state: QbertState):
        pos = jax.lax.cond(
            pred = jax.random.uniform(jax.random.PRNGKey(state.prng_state + 1), (), minval=1, maxval=256).astype(jnp.uint8) < 128,
            true_fun=lambda state2: self.move(state2,Action.DOWN, state2.purple_ball_position),
            false_fun=lambda state2: self.move(state2,Action.RIGHT, state2.purple_ball_position),
            operand=state
        )

        return pos

    @partial(jax.jit, static_argnums=(0,))
    def move_green_ball(self,state: QbertState):
        pos = jax.lax.cond(
            pred = jax.random.uniform(jax.random.PRNGKey(state.prng_state + 2), (), minval=1, maxval=256).astype(jnp.uint8) < 128,
            true_fun=lambda state2: self.move(state2,Action.DOWN, state2.green_ball_position),
            false_fun=lambda state2: self.move(state2,Action.RIGHT, state2.green_ball_position),
            operand=state
        )

        return pos

    @partial(jax.jit, static_argnums=(0,))
    def move_sam(self,state: QbertState):
        pos = jax.lax.cond(
            pred = jax.random.uniform(jax.random.PRNGKey(state.prng_state + 3), (), minval=1, maxval=256).astype(jnp.uint8) < 128,
            true_fun=lambda state2: self.move(state2,Action.DOWN, state2.sam_position),
            false_fun=lambda state2: self.move(state2,Action.RIGHT, state2.sam_position),
            operand=state
        )

        return pos

    @partial(jax.jit, static_argnums=(0,))
    def move_red_balls(self,state: QbertState):
        pos1 = jax.lax.cond(
            pred = jax.random.uniform(jax.random.PRNGKey(state.prng_state + 4), (), minval=1, maxval=256).astype(jnp.uint8) < 128,
            true_fun=lambda state2: self.move(state2,Action.DOWN, state2.red_ball_positions[0]),
            false_fun=lambda state2: self.move(state2,Action.RIGHT, state2.red_ball_positions[0]),
            operand=state
        )
        pos2 = jax.lax.cond(
            pred = jax.random.uniform(jax.random.PRNGKey(state.prng_state + 5), (), minval=1, maxval=256).astype(jnp.uint8) < 128,
            true_fun=lambda state2: self.move(state2,Action.DOWN, state2.red_ball_positions[1]),
            false_fun=lambda state2: self.move(state2,Action.RIGHT, state2.red_ball_positions[1]),
                operand=state
        )
        pos3 = jax.lax.cond(
            pred = jax.random.uniform(jax.random.PRNGKey(state.prng_state + 6), (), minval=1, maxval=256).astype(jnp.uint8) < 128,
            true_fun=lambda state2: self.move(state2,Action.DOWN, state2.red_ball_positions[2]),
            false_fun=lambda state2: self.move(state2,Action.RIGHT, state2.red_ball_positions[2]),
            operand=state
        )

        return jnp.array([pos1,pos2,pos3]).astype(jnp.int32)

    @partial(jax.jit, static_argnums=(0,))
    def move_snake(self, state: QbertState, target: chex.Array):
        distances=(target[0] - state.snake_position[0], target[1] - state.snake_position[1])
        direction=self.closest_direction(distances, state.prng_state + 7, directions=jnp.array([[0,-1],[1,1],[-1,-1],[0,1]]))
        direction=jax.lax.cond(
            pred=jnp.logical_or(state.pyramid[state.snake_position[1] + direction[1]][state.snake_position[0] + direction[0]] >= 0, jnp.logical_and(state.snake_position[0] + direction[0] == target[0], state.snake_position[1] + direction[1] == target[1])),
            true_fun=lambda d: d,
            false_fun=lambda d: jnp.array([0,0]),
            operand=direction
        )
        return jnp.array([state.snake_position[0] + direction[0], state.snake_position[1] + direction[1]])

    @partial(jax.jit, static_argnums=(0,))
    def closest_direction(self, vec, key, directions, tol=1e-6):
        v = jnp.asarray(vec, jnp.float32)
        d = jnp.asarray(directions, jnp.float32)

        eps = 1e-12
        v_hat = v / jnp.maximum(jnp.linalg.norm(v), eps)
        d_hat = d / jnp.maximum(jnp.linalg.norm(d, axis=1, keepdims=True), eps)

        sims = d_hat @ v_hat                      # cosine similarities
        m = sims.max()
        tied_mask = jnp.isclose(sims, m, atol=tol)

        # fixed-size indices for JIT using size=number_of_dirs
        idxs = jnp.flatnonzero(tied_mask, size=d.shape[0], fill_value=0)
        count = tied_mask.sum()

        choice = jax.random.randint(jax.random.PRNGKey(key), (), 0, jnp.maximum(count, 1))
        idx = idxs[choice]

        return directions[idx]

    @partial(jax.jit, static_argnums=(0,))
    def spawnCreatures(self, seed : chex.Numeric, red_pos: chex.Array, purple_pos: chex.Array, green_pos: chex.Array, sam_pos: chex.Array, snake_pos:chex.Array, difficulty:chex.Numeric):
        random = jax.random.uniform(jax.random.PRNGKey(seed + 8), (), minval=1, maxval=4000).astype(jnp.uint16)
        creatureIndex=jax.lax.cond(
            pred=random < 2,
            true_fun=lambda i: 2,
            false_fun=lambda i: jax.lax.cond(
                pred=i < 4,
                true_fun=lambda j: 3,
                false_fun=lambda j: jax.lax.cond(
                    pred=j < 8,
                    true_fun=lambda k: 0,
                    false_fun=lambda k: jax.lax.cond(
                        pred=k < 15,
                        true_fun=lambda l: 1,
                        false_fun=lambda l: 4,
                        operand=k
                    ),
                    operand=j
                ),
                operand=i
            ),
            operand=random
        )
        location=jax.lax.cond(
            pred=jax.random.uniform(jax.random.PRNGKey(seed + 9), (), minval=1, maxval=256).astype(jnp.uint8) < 128,
            true_fun=lambda i: jnp.array([1,2]),
            false_fun=lambda i: jnp.array([2,2]),
            operand=None
        )

        redPos,purplePos,greenPos,samPos=jax.lax.switch(
            index=creatureIndex,
            branches=[
                lambda vals: jax.lax.cond(
                    pred=vals[6] == 1,
                    true_fun=lambda vals2: (self.redSpawn(vals2[5],vals2[0]),vals2[1],vals2[2],vals2[3]),
                    false_fun=lambda vals2: (vals2[0],vals2[1],vals[2],vals[3]),
                    operand=vals
                ),
                lambda vals: jax.lax.cond(
                    pred=jnp.logical_and(jnp.logical_and(vals[1][0] == -1, vals[1][0] == -1),jnp.logical_and(vals[4][0] == -1, vals[4][1] == -1)),
                    true_fun= lambda vals2: (vals2[0],vals2[5], vals2[2], vals2[3]),
                    false_fun= lambda vals2: (vals2[0], vals2[1], vals2[2], vals2[3]),
                    operand=vals
                ),
                lambda vals: jax.lax.cond(
                    pred=jnp.logical_and(vals[2][0] == -1, vals[2][1] == -1),
                    true_fun= lambda vals2: (vals2[0],vals2[1], vals2[5], vals2[3]),
                    false_fun= lambda vals2: (vals2[0], vals2[1], vals2[2], vals2[3]),
                    operand=vals
                ),
                lambda vals: jax.lax.cond(
                    pred=jnp.logical_and(vals[3][0] == -1, vals[3][1] == -1),
                    true_fun= lambda vals2: (vals2[0],vals2[1], vals2[2], vals2[5]),
                    false_fun= lambda vals2: (vals2[0], vals2[1], vals2[2], vals2[3]),
                    operand=vals
                ),
                lambda vals: (vals[0],vals[1],vals[2],vals[3]),
            ],
            operand=(red_pos,purple_pos,green_pos,sam_pos,snake_pos,location, difficulty)
        )

        return redPos, purplePos, greenPos, samPos

    @partial(jax.jit, static_argnums=(0,))
    def redSpawn(self, location: chex.Array, redPos: chex.Array):
        redPos=jax.lax.cond(
            pred=jnp.logical_and(redPos[0][0] == -1, redPos[0][1] == -1),
            true_fun=lambda red: jnp.array([location,red[0][1],red[0][2]]),
            false_fun=lambda red: red[0],
            operand=(redPos,location)
        )
        redPos=jax.lax.cond(
            pred=jnp.logical_and(redPos[1][0] == -1, redPos[1][1] == -1),
            true_fun=lambda red: jnp.array([red[0][0],location,red[0][2]]),
            false_fun=lambda red: red[0],
            operand=(redPos,location)
        )
        redPos=jax.lax.cond(
            pred=jnp.logical_and(redPos[2][0] == -1, redPos[2][1] == -1),
            true_fun=lambda red: jnp.array([red[0][0],red[0][1],location]),
            false_fun=lambda red: red[0],
            operand=(redPos,location)
        )
        return jnp.array(redPos)

    @partial(jax.jit, static_argnums=(0,))
    def nextRound(self, state: QbertState, pyramid: chex.Array, round: chex.Numeric, level: chex.Numeric, player_position: chex.Array, spawned: chex.Numeric, green_ball_freeze_step: chex.Numeric):
        complete=jnp.all(jnp.isin(pyramid, jnp.array([-2, -1, 2], dtype=pyramid.dtype)))
        return jax.lax.cond(
            pred=complete,
            true_fun=lambda vals: (jax.lax.cond(
                pred=jax.random.uniform(jax.random.PRNGKey(vals[3] + 10), (), minval=1, maxval=256).astype(jnp.uint8) < 128,
                true_fun=lambda vals2: jnp.array([
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                    [-1,  0, -1, -1, -1, -1, -1, -1],
                    [-1,  0,  0, -1, -1, -1, -1, -1],
                    [-1,  0,  0,  0, -1, -1, -1, -1],
                    [-2,  0,  0,  0,  0, -2, -1, -1],
                    [-1,  0,  0,  0,  0,  0, -1, -1],
                    [-1,  0,  0,  0,  0,  0,  0, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1]
                ]).astype(jnp.int32),
                false_fun=lambda vals2: jnp.array([
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                    [-1,  0, -1, -1, -1, -1, -1, -1],
                    [-2,  0,  0, -2, -1, -1, -1, -1],
                    [-1,  0,  0,  0, -1, -1, -1, -1],
                    [-1,  0,  0,  0,  0, -1, -1, -1],
                    [-1,  0,  0,  0,  0,  0, -1, -1],
                    [-1,  0,  0,  0,  0,  0,  0, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1]

                ]),
                operand=None
            ),jnp.array(vals[1] % 5 + 1).astype(jnp.int32) , jnp.minimum(5, vals[2] + jnp.floor(vals[1]/4)).astype(jnp.int32), jnp.array(3100).astype(jnp.int32), jnp.array([1,1]), jnp.array(1).astype(jnp.int32), vals[5]),
            false_fun=lambda vals: (vals[0],vals[1],vals[2], jnp.array(0).astype(jnp.int32), player_position, vals[4], vals[6]),
            operand=(pyramid,round,level,state.prng_state,spawned, state.step_counter, green_ball_freeze_step)
        )

    @partial(jax.jit, static_argnums=(0,))
    def extraLives(self, round: chex.Numeric, level: chex.Numeric, lives: chex.Numeric):
        return jax.lax.cond(
            pred=jnp.logical_and(level >= 2, round == 2),
            true_fun=lambda live: live + 1,
            false_fun= lambda live: live,
            operand=lives
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: QbertState, action: chex.Array) -> Tuple[QbertObservation, QbertState, float, bool, QbertInfo]:
        tick_counter_reset = jnp.array([31, 227, 144, 124, 110, 95, 81, 66, 52]).astype(jnp.int32)

        # Handle player movement
        is_player_moving = jnp.where(jnp.logical_or(jnp.logical_and(state.player_moving_counter == 0, action != Action.NOOP), (state.player_moving_counter + 1) % tick_counter_reset[state.player_position_category] > 1), 1, 0)
        player_moving_counter = jnp.where(state.is_player_moving == 1, (state.player_moving_counter + 1) % tick_counter_reset[state.player_position_category], state.player_moving_counter)
        player_last_position = jnp.where(player_moving_counter != 0, state.player_last_position, state.player_position)
        player_position = jnp.where(jnp.logical_and(state.is_player_moving == 0, action != Action.NOOP), self.move(state, action, state.player_position), state.player_position)
        player_direction = jnp.select(
            condlist=[
                jnp.logical_and(state.is_player_moving == 0, action == Action.RIGHT),
                jnp.logical_and(state.is_player_moving == 0, action == Action.LEFT),
                jnp.logical_and(state.is_player_moving == 0, action == Action.UP),
                jnp.logical_and(state.is_player_moving == 0, action == Action.DOWN),
            ],
            choicelist=[2, 0, 3, 1],
            default=state.player_direction
        ).astype(jnp.int32)

        player_position_category = jnp.where(player_moving_counter == 0, jnp.select(
            condlist=[
                state.pyramid[player_position[1]][player_position[0]] >= 0,
                jnp.logical_and(state.pyramid[player_position[1]][player_position[0]] == -2, player_position[1] == 4),
                jnp.logical_and(state.pyramid[player_position[1]][player_position[0]] == -2, player_position[1] == 2),
                jnp.logical_and(state.pyramid[player_position[1]][player_position[0]] == -1, player_position[1] == 0),
                jnp.logical_and(state.pyramid[player_position[1]][player_position[0]] == -1, player_position[1] == 1),
                jnp.logical_and(state.pyramid[player_position[1]][player_position[0]] == -1, player_position[1] == 2),
                jnp.logical_and(state.pyramid[player_position[1]][player_position[0]] == -1, player_position[1] == 3),
                jnp.logical_and(state.pyramid[player_position[1]][player_position[0]] == -1, player_position[1] == 4),
                jnp.logical_and(state.pyramid[player_position[1]][player_position[0]] == -1, player_position[1] == 5),
                jnp.logical_and(state.pyramid[player_position[1]][player_position[0]] == -1, player_position[1] == 7),
            ],
            choicelist=[0, 1, 2, 3, 4, 5, 6, 7, 8, 8],
            default=0,
        ), state.player_position_category)

        pyramid,player_position,lives, trash=self.checkField(state,0,0,player_position)

        spawned=jax.lax.cond(
            pred=jnp.logical_or(player_position[0] != 1, player_position[1] != 1),
            true_fun=lambda i: 0,
            false_fun=lambda i: i,
            operand=state.just_spawned
        )
        # Increase enemy moving counter depending on current level
        enemy_moving_counter = jnp.where(state.step_counter == state.green_ball_freeze_step, (state.enemy_moving_counter + 1) % self.consts.ENEMY_MOVE_TICK[state.level_number], state.enemy_moving_counter)

        # Handle red ball movement
        red_pos = jnp.where(enemy_moving_counter == 0, self.move_red_balls(state), state.red_ball_positions)
        trash1, red0_pos, trash2, trash3 = jax.lax.cond(
            pred=enemy_moving_counter == 0,
            true_fun=lambda s: self.checkField(s, 1, 0, red_pos[0]),
            false_fun=lambda s: (state.pyramid, state.red_ball_positions[0], state.lives, jnp.array(0).astype(jnp.int32)),
            operand=state
        )
        trash1, red1_pos, trash2, trash3 = jax.lax.cond(
            pred=enemy_moving_counter == 0,
            true_fun=lambda s: self.checkField(s, 1, 1, red_pos[1]),
            false_fun=lambda s: (state.pyramid, state.red_ball_positions[1], state.lives, jnp.array(0).astype(jnp.int32)),
            operand=state
        )
        trash1, red2_pos, trash2, trash3 = jax.lax.cond(
            pred=enemy_moving_counter == 0,
            true_fun=lambda s: self.checkField(s, 1, 2, red_pos[2]),
            false_fun=lambda s: (state.pyramid, state.red_ball_positions[2], state.lives, jnp.array(0).astype(jnp.int32)),
            operand=state
        )

        # Handle purple ball movement
        trash1, purple_pos, trash2, trash3 = jax.lax.cond(
            pred=enemy_moving_counter == 0,
            true_fun=lambda s: self.checkField(s, 2, 0, self.move_purple_ball(s)),
            false_fun=lambda s: (state.pyramid, state.purple_ball_position, state.lives, jnp.array(0).astype(jnp.int32)),
            operand=state
        )
        temp = jnp.abs(player_position[0] - player_last_position[0]) + jnp.abs(player_position[1] - player_last_position[1])
        snake_lock=jax.lax.cond(
            pred=jnp.logical_and(jnp.logical_and(temp > 2, jnp.abs(state.snake_position[0] - player_last_position[0]) + jnp.abs(state.snake_position[1] - player_last_position[1]) < 4) , state.snake_lock[0] == -1),
            true_fun=lambda i: jax.lax.cond(
                pred=i.player_last_position[0] > 1,
                true_fun=lambda j: jnp.array([j.player_last_position[0], j.player_last_position[1] - 1]),
                false_fun=lambda j: jnp.array([j.player_last_position[0] - 1, j.player_last_position[1] - 1]),
                operand=i
            ),
            false_fun=lambda i: i.snake_lock,
            operand=state
        )

        target=jax.lax.cond(
            pred=snake_lock[0] == -1,
            true_fun=lambda i: i[0],
            false_fun=lambda i: i[1],
            operand=(player_position,snake_lock)

        )
        jax.debug.print("{a}", a=target)

        # Handle snake movement
        trash1, snake_pos, trash2, points4 = jax.lax.cond(
            pred=jnp.logical_and(enemy_moving_counter == 0, state.snake_position[0] != -1),
            true_fun=lambda s: self.checkField(s[0], 3, 0, self.move_snake(s[0], s[1])),
            false_fun=lambda s: ((s[0].pyramid, s[0].snake_position, s[0].lives, jnp.array(0).astype(jnp.int32))),
            operand=(state, target),
        )


        # Handle green ball movement
        trash1, green_pos, trash2, trash3 = jax.lax.cond(
            pred=enemy_moving_counter == 0,
            true_fun=lambda s: self.checkField(s, 4, 0, self.move_green_ball(s)),
            false_fun=lambda s: (state.pyramid, state.green_ball_position, state.lives, jnp.array(0).astype(jnp.int32)),
            operand=state
        )

        # Handle sam movement
        trash1, sam_pos, trash2, trash3 = jax.lax.cond(
            pred=enemy_moving_counter == 0,
            true_fun=lambda s: self.checkField(s, 5, 0, self.move_sam(s)),
            false_fun=lambda s: (state.pyramid, state.sam_position, state.lives, jnp.array(0).astype(jnp.int32)),
            operand=state
        )

        snake_lock=jax.lax.cond(
            pred=snake_pos[0]==-1,
            true_fun=lambda i: jnp.array([-1,-1]),
            false_fun=lambda i: i,
            operand=snake_lock
        )

        lives,red_pos,purple_pos,green_pos,snake_pos,sam_pos,points1, green_ball_freeze_step=self.checkCollisions(lives,player_last_position,jnp.array([red0_pos,red1_pos,red2_pos]),purple_pos,green_pos,snake_pos,sam_pos, state.green_ball_freeze_step)
        pyramid,points2=self.changeColors(player_position,sam_pos,pyramid,state.level_number, spawned)

        red_pos, purple_pos, green_pos, sam_pos = jax.lax.cond(
            pred=state.step_counter == green_ball_freeze_step,
            true_fun=lambda op: self.spawnCreatures(op[0], op[1], op[2], op[3], op[4], op[5], op[6]),
            false_fun=lambda op: (op[1], op[2], op[3], op[4]),
            operand=(state.prng_state, jnp.array(red_pos), jnp.array(purple_pos) , jnp.array(green_pos), jnp.array(sam_pos), jnp.array(snake_pos), QbertConstants.DIFFICULTY)
        )

        pyramid, round, level, points3, player_position, spawned, green_ball_freeze_step = jax.lax.cond(
            pred=jnp.logical_and(is_player_moving == 0, player_moving_counter == 0),
            true_fun=lambda op: self.nextRound(op[0], op[1], op[2], op[3], op[4], op[5], op[6]),
            false_fun=lambda op: (op[1], op[2], op[3], 0, op[4], op[5], op[6]),
            operand=(state, pyramid, state.round_number, state.level_number, player_position, spawned, green_ball_freeze_step)
        )

        lives=jnp.where(jnp.logical_and(is_player_moving == 0, player_moving_counter == 0), self.extraLives(round,level,lives), state.lives)

        # Updates last pyramid
        last_pyramid = jnp.where(jnp.logical_or(jnp.logical_and(jnp.logical_or(player_position_category == 1, player_position_category == 2), player_moving_counter == 45), jnp.logical_and(jnp.logical_and(player_position_category != 1, player_position_category != 2), player_moving_counter == 27)), state.pyramid, state.last_pyramid)

        jax.debug.print("is_player_moving: {a}, player_moving_counter: {b}, player_last_position: {c}, player_position: {d}, player_direction: {e}, lives: {f}, step_counter: {g}, freeze: {h}", a=is_player_moving, b=player_moving_counter, c=player_last_position, d = player_position, e=player_direction, f=state.lives, g = state.step_counter, h=green_ball_freeze_step)

        red_pos, purple_pos, green_pos, snake_pos, sam_pos=jax.lax.cond(
            pred=jnp.logical_or(round != state.round_number, lives < state.lives),
            true_fun=lambda c: (jnp.array([[-1, -1], [-1, -1], [-1, -1]]).astype(jnp.int32), jnp.array([-1,-1]), jnp.array([-1,-1]), jnp.array([-1,-1]),  jnp.array([-1,-1])),
            false_fun=lambda c: (c[0], c[1], c[2], c[3], c[4]),
            operand=(red_pos, purple_pos, green_pos, snake_pos, sam_pos)
        )

        new_state = QbertState(
            player_score=state.player_score + points1 + points2 + points3 + points4,
            lives=lives,
            pyramid=pyramid,
            last_pyramid=last_pyramid,
            player_position=player_position,
            player_last_position=player_last_position,
            player_direction=player_direction,
            player_position_category=player_position_category,
            is_player_moving=is_player_moving,
            player_moving_counter=player_moving_counter,
            level_number=level,
            round_number=round,
            green_ball_freeze_step=jnp.where(state.step_counter == green_ball_freeze_step, green_ball_freeze_step + 1, green_ball_freeze_step),
            red_ball_positions=red_pos,
            enemy_moving_counter=enemy_moving_counter,
            purple_ball_position=purple_pos,
            snake_position=snake_pos,
            green_ball_position=green_pos,
            sam_position=sam_pos,
            step_counter=state.step_counter + 1,
            prng_state=state.prng_state + 15,
            just_spawned=spawned,
            snake_lock=jnp.array([snake_lock[0], snake_lock[1]]),
        )

        #TODO
        done = self._get_done(new_state)
        env_reward = self._get_reward(state, new_state)
        all_rewards = self._get_all_reward(state, new_state)
        info = self._get_info(new_state, all_rewards)
        observation = self._get_observation(new_state)

        return observation, new_state, env_reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: QbertState, state: QbertState):
        return state.player_score - previous_state.player_score

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(self, previous_state: QbertState, state: QbertState):
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array(
            [reward_func(previous_state, state)
             for reward_func in self.reward_funcs]
        )
        return rewards

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: QbertState) -> jnp.ndarray:
        # TODO
        return self.renderer.render(state)

    def action_space(self) -> Space:
        return Discrete(len(self.action_set))

    def image_space(self) -> Space:
        return Box(0, 255, shape=(self.consts.HEIGHT, self.consts.WIDTH, 3), dtype=jnp.uint8)

    def observation_space(self) -> Space:
        return Dict({
            "player_score": Box(0, 99999, (), jnp.int32),
            "lives": Box(-1, 9, (), jnp.int32),
            "pyramid": Box(-2, 3, (8, 8), jnp.int32),
            "player_position": Box(0, 7, (2,), jnp.int32),
            "level_number": Box(1, 5, (), jnp.int32),
            "round_number": Box(1, 6, (), jnp.int32),
            "red_ball_positions": Box(0, 7, (2,), jnp.int32),
            "purple_ball_position": Box(0, 7, (2,), jnp.int32),
            "snake_position": Box(0, 7, (2,), jnp.int32),
            "green_ball_position": Box(0, 7, (2,), jnp.int32),
            "sam_position": Box(0, 7, (2,), jnp.int32)
        })

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: QbertObservation) -> jnp.ndarray:
        return jnp.concatenate([
            obs.player_score.flatten(),
            obs.lives.flatten(),
            obs.pyramid.flatten(),
            obs.player_position.flatten(),
            obs.level_number.flatten(),
            obs.round_number.flatten(),
            obs.red_ball_positions.flatten(),
            obs.purple_ball_position.flatten(),
            obs.snake_position.flatten(),
            obs.green_ball_position.flatten(),
            obs.sam_position.flatten()
        ])

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: QbertState):
        #TODO
        return QbertObservation(
            player_score=state.player_score,
            lives=state.lives,
            pyramid=state.pyramid,
            player_position=state.player_position,
            level_number=state.level_number,
            round_number=state.round_number,
            red_ball_positions=state.red_ball_positions,
            purple_ball_position=state.purple_ball_position,
            snake_position=state.snake_position,
            green_ball_position=state.green_ball_position,
            sam_position=state.sam_position
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: QbertState, all_rewards: jnp.ndarray = None) -> QbertInfo:
        return QbertInfo(state.step_counter, all_rewards)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: QbertState) -> bool:
        player_has_no_lives = state.lives == -1
        player_reaches_end_of_life = jnp.logical_and(state.level_number == 5, state.round_number == 6)
        return jnp.logical_or(player_has_no_lives, player_reaches_end_of_life)

def load_sprites():
    """ Load all sprites required for Qbert rendering """
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

    BACKGROUND = aj.get_sprite_frame(jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "sprites/qbert/background.npy"), transpose=False), axis=0), 0)
    FREEZE = aj.get_sprite_frame(jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "sprites/qbert/freeze.npy"), transpose=False), axis=0), 0)

    PLAYER_SCORE_DIGIT = aj.load_and_pad_digits(os.path.join(MODULE_DIR, "sprites/qbert/score/score_{}.npy"), num_chars=10)

    QBERT_LEFT_DOWN = aj.get_sprite_frame(jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "sprites/qbert/qbert/qbert_left_down.npy"), transpose=False), axis=0), 0)
    QBERT_LEFT_UP = aj.get_sprite_frame(jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "sprites/qbert/qbert/qbert_left_up.npy"), transpose=False), axis=0), 0)
    QBERT_RIGHT_DOWN = aj.get_sprite_frame(jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "sprites/qbert/qbert/qbert_right_down.npy"), transpose=False), axis=0), 0)
    QBERT_RIGHT_UP = aj.get_sprite_frame(jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "sprites/qbert/qbert/qbert_right_up.npy"), transpose=False), axis=0), 0)

    CUBE_SHADOW_LEFT = jnp.array([
        aj.get_sprite_frame(jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "sprites/qbert/cube/cube_shadow_left_1.npy"), transpose=False), axis=0), 0),
        aj.get_sprite_frame(jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "sprites/qbert/cube/cube_shadow_left_2.npy"), transpose=False), axis=0), 0),
        aj.get_sprite_frame(jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "sprites/qbert/cube/cube_shadow_left_3.npy"), transpose=False), axis=0), 0),
        aj.get_sprite_frame(jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "sprites/qbert/cube/cube_shadow_left_4.npy"), transpose=False), axis=0), 0),
        aj.get_sprite_frame(jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "sprites/qbert/cube/cube_shadow_left_5.npy"), transpose=False), axis=0), 0),
    ]).astype(jnp.int32)
    CUBE_SHADOW_RIGHT = jnp.array([
        aj.get_sprite_frame(jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "sprites/qbert/cube/cube_shadow_right_1.npy"), transpose=False), axis=0), 0),
        aj.get_sprite_frame(jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "sprites/qbert/cube/cube_shadow_right_2.npy"), transpose=False), axis=0), 0),
        aj.get_sprite_frame(jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "sprites/qbert/cube/cube_shadow_right_3.npy"), transpose=False), axis=0), 0),
        aj.get_sprite_frame(jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "sprites/qbert/cube/cube_shadow_right_4.npy"), transpose=False), axis=0), 0),
        aj.get_sprite_frame(jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "sprites/qbert/cube/cube_shadow_right_5.npy"), transpose=False), axis=0), 0),
    ]).astype(jnp.int32)

    DESTINATION_COLOR = aj.get_sprite_frame(jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "sprites/qbert/color/color_destination.npy"), transpose=False), axis=0), 0)
    INTERMEDIATE_COLOR = aj.get_sprite_frame(jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "sprites/qbert/color/color_intermediate.npy"), transpose=False), axis=0), 0)
    START_COLOR = aj.get_sprite_frame(jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "sprites/qbert/color/color_start.npy"), transpose=False), axis=0), 0)

    QBERT_LIVE = aj.get_sprite_frame(jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "sprites/qbert/qbert_live.npy"), transpose=False), axis=0), 0)
    DISC = aj.get_sprite_frame(jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "sprites/qbert/disc.npy"), transpose=False), axis=0), 0)

    SAM = aj.get_sprite_frame(jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "sprites/qbert/enemies/sam.npy"), transpose=False), axis=0), 0)
    GREEN_BALL = jnp.array([
        aj.get_sprite_frame(jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "sprites/qbert/enemies/green_ball_1.npy"), transpose=False), axis=0), 0),
        aj.get_sprite_frame(jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "sprites/qbert/enemies/green_ball_2.npy"), transpose=False), axis=0), 0)
    ]).astype(jnp.int32)
    PURPLE_BALL = jnp.array([
        aj.get_sprite_frame(jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "sprites/qbert/enemies/purple_ball_1.npy"), transpose=False), axis=0), 0),
        aj.get_sprite_frame(jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "sprites/qbert/enemies/purple_ball_2.npy"), transpose=False), axis=0), 0)
    ]).astype(jnp.int32)
    RED_BALL = jnp.array([
        aj.get_sprite_frame(jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "sprites/qbert/enemies/red_ball_1.npy"), transpose=False), axis=0), 0),
        aj.get_sprite_frame(jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "sprites/qbert/enemies/red_ball_2.npy"), transpose=False), axis=0), 0)
    ]).astype(jnp.int32)
    SNAKE = jnp.array([
        aj.get_sprite_frame(jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "sprites/qbert/enemies/snake_1.npy"), transpose=False), axis=0), 0),
        aj.get_sprite_frame(jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "sprites/qbert/enemies/snake_2.npy"), transpose=False), axis=0), 0),
        aj.get_sprite_frame(jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "sprites/qbert/enemies/snake_3.npy"), transpose=False), axis=0), 0),
        aj.get_sprite_frame(jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "sprites/qbert/enemies/snake_2.npy"), transpose=False), axis=0), 0),
        aj.get_sprite_frame(jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "sprites/qbert/enemies/snake_1.npy"), transpose=False), axis=0), 0),
    ]).astype(jnp.int32)

    return (
        BACKGROUND,
        FREEZE,
        PLAYER_SCORE_DIGIT,
        QBERT_LEFT_DOWN,
        QBERT_LEFT_UP,
        QBERT_RIGHT_DOWN,
        QBERT_RIGHT_UP,
        CUBE_SHADOW_LEFT,
        CUBE_SHADOW_RIGHT,
        DESTINATION_COLOR,
        INTERMEDIATE_COLOR,
        START_COLOR,
        QBERT_LIVE,
        DISC,
        SAM,
        GREEN_BALL,
        PURPLE_BALL,
        RED_BALL,
        SNAKE
    )


class QbertRenderer(JAXGameRenderer):
    def __init__(self, consts: QbertConstants = None):
        super().__init__()
        self.consts = consts or QbertConstants()
        (
            self.BACKGROUND,
            self.FREEZE,
            self.PLAYER_SCORE_DIGIT,
            self.QBERT_LEFT_DOWN,
            self.QBERT_LEFT_UP,
            self.QBERT_RIGHT_DOWN,
            self.QBERT_RIGHT_UP,
            self.CUBE_SHADOW_LEFT,
            self.CUBE_SHADOW_RIGHT,
            self.DESTINATION_COLOR,
            self.INTERMEDIATE_COLOR,
            self.START_COLOR,
            self.QBERT_LIVE,
            self.DISC,
            self.SAM,
            self.GREEN_BALL,
            self.PURPLE_BALL,
            self.RED_BALL,
            self.SNAKE
        ) = load_sprites()

        self.COLOR_POSITIONS = jnp.array([[68, 33], [56, 62], [80, 62], [44, 91], [68, 91], [92, 91], [32, 120], [56, 120], [80, 120], [104, 120], [20, 149], [44, 149], [68, 149], [92, 149], [116, 149], [8, 178], [32, 178], [56, 178], [80, 178], [104, 178], [128, 178]]).astype(jnp.int32)
        self.LIVE_POSITIONS = jnp.array([[33, 16], [41, 16], [49, 16], [57, 16], [65, 16], [73, 16], [81, 16], [89, 16], [97, 16]]).astype(jnp.int32)
        self.QBERT_POSITIONS = jnp.array([[74, 18], [62, 47], [86, 47], [50, 76], [74, 76], [98, 76], [38, 105], [62, 105], [86, 105], [110, 105], [26, 134], [50, 134], [74, 134], [98, 134], [122, 134], [14, 163], [38, 163], [62, 163], [86, 163], [110, 163], [134, 163]]).astype(jnp.int32)
        self.QBERT_MOVE_RIGHT_DOWN = jnp.array([[0, 0], [1, -1], [2, -2], [3, -3], [4, -4], [5, -5], [6, -6], [7, -5], [8, -4], [9, -3], [10, -2], [11, -1], [12, -0], [12, 1], [12, 3], [12, 5], [12, 8], [12, 10], [12, 12], [12, 14], [12, 17], [12, 19], [12, 21], [12, 23], [12, 25], [12, 27], [12, 29], [12, 29], [12, 29], [12, 29]]).astype(jnp.int32)
        self.QBERT_MOVE_RIGHT_UP = jnp.array([[0, 0], [0, -2], [0, -4], [0, -6], [0, -8], [0, -10], [0, -12], [0, -15], [0, -17], [0, -19], [0, -21], [0, -24], [0, -26], [0, -28], [0, -29], [1, -30], [2, -31], [3, -32], [4, -33], [5, -34], [6, -35], [7, -34], [8, -33], [9, -32], [10, -31], [11, -30], [12, -29], [12, -29], [12, -29], [12, -29]]).astype(jnp.int32)
        self.QBERT_MOVE_LEFT_DOWN = jnp.array([[0, 0], [-1, -1], [-2, -2], [-3, -3], [-4, -4], [-5, -5], [-6, -6], [-7, -5], [-8, -4], [-9, -3], [-10, -2], [-11, -1], [-12, -0], [-12, 1], [-12, 3], [-12, 5], [-12, 8], [-12, 10], [-12, 12], [-12, 14], [-12, 17], [-12, 19], [-12, 21], [-12, 23], [-12, 25], [-12, 27], [-12, 29], [-12, 29], [-12, 29], [-12, 29]]).astype(jnp.int32)
        self.QBERT_MOVE_LEFT_UP = jnp.array([[0, 0], [0, -2], [0, -4], [0, -6], [0, -8], [0, -10], [0, -12], [0, -15], [0, -17], [0, -19], [0, -21], [0, -24], [0, -26], [0, -28], [0, -29], [-1, -30], [-2, -31], [-3, -32], [-4, -33], [-5, -34], [-6, -35], [-7, -34], [-8, -33], [-9, -32], [-10, -31], [-11, -30], [-12, -29], [-12, -29], [-12, -29], [-12, -29]]).astype(jnp.int32)
        self.QBERT_MOVE_DISC_LEFT_BOTTOM = jnp.array([[0, 0], [0, -2], [0, -4], [0, -6], [0, -8], [0, -10], [0, -12], [0, -14], [0, -16], [0, -18], [0, -20], [0, -22], [0, -24], [0, -26], [0, -28], [-1, -29], [-2, -30], [-3, -31], [-4, -32], [-5, -33], [-6, -34], [-7, -33], [-8, -32], [-9, -31], [-10, -30], [-11, -29], [-12, -28], [-12, -27], [-12, -26], [-12, -25], [-12, -24], [-12, -23], [-12, -22], [-12, -21], [-12, -20], [-12, -19], [-12, -18], [-12, -17], [-12, -16], [-12, -15], [-12, -14], [-12, -13], [-12, -12], [-12, -11], [-12, -10], [-12, -10], [-12, -11], [-12, -12], [-12, -13], [-12, -14], [-12, -15], [-12, -16], [-12, -17], [-12, -18], [-12, -19], [-12, -20], [-12, -21], [-12, -22], [-12, -23], [-12, -24], [-12, -25], [-12, -26], [-12, -27], [-12, -28], [-12, -29], [-12, -30], [-12, -31], [-12, -32], [-12, -33], [-12, -34], [-12, -35], [-12, -36], [-12, -37], [-12, -38], [-12, -39], [-12, -40], [-12, -41], [-12, -42], [-12, -43], [-12, -44], [-12, -45], [-12, -46], [-12, -47], [-12, -48], [-12, -49], [-12, -50], [-12, -51], [-12, -52], [-12, -53], [-12, -54], [-12, -55], [-12, -56], [-12, -57], [-12, -58], [-12, -59], [-12, -60], [-12, -61], [-12, -62], [-12, -63], [-12, -64], [-12, -65], [-12, -66], [-12, -67], [-12, -68], [-12, -69], [-12, -70], [-12, -71], [-12, -72], [-12, -73], [-12, -74], [-12, -75], [-12, -76], [-12, -77], [-12, -78], [-12, -79], [-12, -80], [-12, -81], [-12, -82], [-12, -83], [-12, -84], [-12, -85], [-12, -86], [-12, -87], [-12, -88], [-12, -89], [-12, -90], [-12, -91], [-12, -92], [-12, -93], [-12, -94], [-12, -95], [-12, -96], [-12, -97], [-12, -98], [-12, -99], [-12, -100], [-12, -101], [-12, -102], [-12, -103], [-12, -104], [-12, -105], [-12, -106], [-12, -107], [-12, -108], [-12, -109], [-12, -110], [-12, -111], [-12, -112], [-12, -113], [-12, -114], [-12, -115], [-12, -116], [-12, -117], [-12, -118], [-12, -119], [-12, -120], [-12, -121], [-12, -122], [-12, -123], [-12, -124], [-11, -124], [-10, -124], [-9, -124], [-8, -124], [-7, -124], [-6, -124], [-5, -124], [-4, -124], [-3, -124], [-2, -124], [-1, -124], [0, -124], [1, -124], [2, -124], [3, -124], [4, -124], [5, -124], [6, -124], [7, -124], [8, -124], [9, -124], [10, -124], [11, -124], [12, -124], [13, -124], [14, -124], [15, -124], [16, -124], [17, -124], [18, -124], [19, -124], [20, -124], [21, -124], [22, -124], [23, -124], [24, -124], [25, -124], [26, -124], [27, -124], [28, -124], [29, -124], [30, -124], [31, -124], [32, -124], [33, -124], [34, -124], [35, -124], [36, -124], [37, -124], [38, -124], [39, -124], [40, -124], [41, -124], [42, -124], [43, -124], [44, -124], [45, -124], [46, -124], [47, -124], [48, -124], [48, -123], [48, -122], [48, -121], [48, -120], [48, -119], [48, -118], [48, -117]]).astype(jnp.int32)
        self.QBERT_MOVE_DISC_RIGHT_BOTTOM = jnp.array([[0, 0], [0, -2], [0, -4], [0, -6], [0, -8], [0, -10], [0, -12], [0, -14], [0, -16], [0, -18], [0, -20], [0, -22], [0, -24], [0, -26], [0, -28], [1, -29], [2, -30], [3, -31], [4, -32], [5, -33], [6, -34], [7, -33], [8, -32], [9, -31], [10, -30], [11, -29], [12, -28], [12, -27], [12, -26], [12, -25], [12, -24], [12, -23], [12, -22], [12, -21], [12, -20], [12, -19], [12, -18], [12, -17], [12, -16], [12, -15], [12, -14], [12, -13], [12, -12], [12, -11], [12, -10], [12, -10], [12, -11], [12, -12], [12, -13], [12, -14], [12, -15], [12, -16], [12, -17], [12, -18], [12, -19], [12, -20], [12, -21], [12, -22], [12, -23], [12, -24], [12, -25], [12, -26], [12, -27], [12, -28], [12, -29], [12, -30], [12, -31], [12, -32], [12, -33], [12, -34], [12, -35], [12, -36], [12, -37], [12, -38], [12, -39], [12, -40], [12, -41], [12, -42], [12, -43], [12, -44], [12, -45], [12, -46], [12, -47], [12, -48], [12, -49], [12, -50], [12, -51], [12, -52], [12, -53], [12, -54], [12, -55], [12, -56], [12, -57], [12, -58], [12, -59], [12, -60], [12, -61], [12, -62], [12, -63], [12, -64], [12, -65], [12, -66], [12, -67], [12, -68], [12, -69], [12, -70], [12, -71], [12, -72], [12, -73], [12, -74], [12, -75], [12, -76], [12, -77], [12, -78], [12, -79], [12, -80], [12, -81], [12, -82], [12, -83], [12, -84], [12, -85], [12, -86], [12, -87], [12, -88], [12, -89], [12, -90], [12, -91], [12, -92], [12, -93], [12, -94], [12, -95], [12, -96], [12, -97], [12, -98], [12, -99], [12, -100], [12, -101], [12, -102], [12, -103], [12, -104], [12, -105], [12, -106], [12, -107], [12, -108], [12, -109], [12, -110], [12, -111], [12, -112], [12, -113], [12, -114], [12, -115], [12, -116], [12, -117], [12, -118], [12, -119], [12, -120], [12, -121], [12, -122], [12, -123], [12, -124], [11, -124], [10, -124], [9, -124], [8, -124], [7, -124], [6, -124], [5, -124], [4, -124], [3, -124], [2, -124], [1, -124], [0, -124], [-0, -124], [-1, -124], [-2, -124], [-3, -124], [-4, -124], [-5, -124], [-6, -124], [-7, -124], [-8, -124], [-9, -124], [-10, -124], [-11, -124], [-12, -124], [-13, -124], [-14, -124], [-15, -124], [-16, -124], [-17, -124], [-18, -124], [-19, -124], [-20, -124], [-21, -124], [-22, -124], [-23, -124], [-24, -124], [-25, -124], [-26, -124], [-27, -124], [-28, -124], [-29, -124], [-30, -124], [-31, -124], [-32, -124], [-33, -124], [-34, -124], [-35, -124], [-36, -124], [-37, -124], [-38, -124], [-39, -124], [-40, -124], [-41, -124], [-42, -124], [-43, -124], [-44, -124], [-45, -124], [-46, -124], [-47, -124], [-48, -124], [-48, -123], [-48, -122], [-48, -121], [-48, -120], [-48, -119], [-48, -118], [-48, -117]]).astype(jnp.int32)
        self.QBERT_MOVE_DISC_LEFT_TOP = jnp.array([[0, 0], [0, -2], [0, -4], [0, -6], [0, -8], [0, -10], [0, -12], [0, -14], [0, -16], [0, -18], [0, -20], [0, -22], [0, -24], [0, -26], [0, -28], [-1, -29], [-2, -30], [-3, -31], [-4, -32], [-5, -33], [-6, -34], [-7, -33], [-8, -32], [-9, -31], [-10, -30], [-11, -29], [-12, -28], [-12, -27], [-12, -26], [-12, -25], [-12, -24], [-12, -23], [-12, -22], [-12, -21], [-12, -20], [-12, -19], [-12, -18], [-12, -17], [-12, -16], [-12, -15], [-12, -14], [-12, -13], [-12, -12], [-12, -11], [-12, -10], [-12, -10], [-12, -11], [-12, -12], [-12, -13], [-12, -14], [-12, -15], [-12, -16], [-12, -17], [-12, -18], [-12, -19], [-12, -20], [-12, -21], [-12, -22], [-12, -23], [-12, -24], [-12, -25], [-12, -26], [-12, -27], [-12, -28], [-12, -29], [-12, -30], [-12, -31], [-12, -32], [-12, -33], [-12, -34], [-12, -35], [-12, -36], [-12, -37], [-12, -38], [-12, -39], [-12, -40], [-12, -41], [-12, -42], [-12, -43], [-12, -44], [-12, -45], [-12, -46], [-12, -47], [-12, -48], [-12, -49], [-12, -50], [-12, -51], [-12, -52], [-12, -53], [-12, -54], [-12, -55], [-12, -56], [-12, -57], [-12, -58], [-12, -59], [-12, -60], [-12, -61], [-12, -62], [-12, -63], [-12, -64], [-12, -65], [-12, -66], [-12, -66], [-11, -66], [-10, -66], [-9, -66], [-8, -66], [-7, -66], [-6, -66], [-5, -66], [-4, -66], [-3, -66], [-2, -66], [-1, -66], [0, -66], [1, -66], [2, -66], [3, -66], [4, -66], [5, -66], [6, -66], [7, -66], [8, -66], [9, -66], [10, -66], [11, -66], [12, -66], [13, -66], [14, -66], [15, -66], [16, -66], [17, -66], [18, -66], [19, -66], [20, -66], [21, -66], [22, -66], [23, -66], [24, -66], [24, -65], [24, -64], [24, -63], [24, -62], [24, -61], [24, -60], [24, -59]]).astype(jnp.int32)
        self.QBERT_MOVE_DISC_RIGHT_TOP = jnp.array([[0, 0], [0, -2], [0, -4], [0, -6], [0, -8], [0, -10], [0, -12], [0, -14], [0, -16], [0, -18], [0, -20], [0, -22], [0, -24], [0, -26], [0, -28], [1, -29], [2, -30], [3, -31], [4, -32], [5, -33], [6, -34], [7, -33], [8, -32], [9, -31], [10, -30], [11, -29], [12, -28], [12, -27], [12, -26], [12, -25], [12, -24], [12, -23], [12, -22], [12, -21], [12, -20], [12, -19], [12, -18], [12, -17], [12, -16], [12, -15], [12, -14], [12, -13], [12, -12], [12, -11], [12, -10], [12, -10], [12, -11], [12, -12], [12, -13], [12, -14], [12, -15], [12, -16], [12, -17], [12, -18], [12, -19], [12, -20], [12, -21], [12, -22], [12, -23], [12, -24], [12, -25], [12, -26], [12, -27], [12, -28], [12, -29], [12, -30], [12, -31], [12, -32], [12, -33], [12, -34], [12, -35], [12, -36], [12, -37], [12, -38], [12, -39], [12, -40], [12, -41], [12, -42], [12, -43], [12, -44], [12, -45], [12, -46], [12, -47], [12, -48], [12, -49], [12, -50], [12, -51], [12, -52], [12, -53], [12, -54], [12, -55], [12, -56], [12, -57], [12, -58], [12, -59], [12, -60], [12, -61], [12, -62], [12, -63], [12, -64], [12, -65], [12, -66], [12, -66], [11, -66], [10, -66], [9, -66], [8, -66], [7, -66], [6, -66], [5, -66], [4, -66], [3, -66], [2, -66], [1, -66], [0, -66], [-1, -66], [-2, -66], [-3, -66], [-4, -66], [-5, -66], [-6, -66], [-7, -66], [-8, -66], [-9, -66], [-10, -66], [-11, -66], [-12, -66], [-13, -66], [-14, -66], [-15, -66], [-16, -66], [-17, -66], [-18, -66], [-19, -66], [-20, -66], [-21, -66], [-22, -66], [-23, -66], [-24, -66], [-24, -65], [-24, -64], [-24, -63], [-24, -62], [-24, -61], [-24, -60], [-24, -59]]).astype(jnp.int32)
        self.BALL_MOVE = jnp.array([[1, 2], [1, 12]])
        self.SNAKE_MOVE = jnp.array([[0, -3], [0, -1], [0, 2], [0, -1], [0, -3]])
        self.QBERT_MOVE_OUT_PYRAMID_1 = jnp.array([[0, 0], [0, -1], [0, -2], [0, -3], [0, -4], [0, -5], [0, -6], [0, -7], [0, -8], [-1, -9], [-2, -10], [-3, -11], [-4, -12], [-5, -13], [-6, -14], [-7, -13], [-8, -12], [-9, -11], [-10, -10], [-11, -9], [-12, -8], [-12, -7], [-12, -6], [-12, -5], [-12, -4], [-12, -3], [-12, -2], [-12, -1], [-12, 0], [-12, 2], [-12, 4], [-12, 6], [-12, 8], [-12, 10], [-12, 12], [-12, 14], [-12, 16], [-12, 18], [-12, 20], [-12, 22], [-12, 24], [-12, 26], [-12, 28], [-12, 30], [-12, 32], [-12, 34], [-12, 36], [-12, 38], [-12, 40], [-12, 42], [-12, 44], [-12, 46], [-12, 48], [-12, 50], [-12, 52], [-12, 54], [-12, 56], [-12, 58], [-12, 60], [-12, 62], [-12, 64], [-12, 66], [-12, 68], [-12, 70], [-12, 72], [-12, 74], [-12, 76], [-12, 78], [-12, 80], [-12, 82], [-12, 84], [-12, 86], [-12, 88], [-12, 90], [-12, 92], [-12, 94], [-12, 96], [-12, 98], [-12, 100], [-12, 102], [-12, 104], [-12, 106], [-12, 108], [-12, 110], [-12, 112], [-12, 114], [-12, 116], [-12, 118], [-12, 120], [-12, 122], [-12, 124], [-12, 126], [-12, 128], [-12, 130], [-12, 132], [-12, 134], [-12, 136], [-12, 138], [-12, 140], [-12, 142], [-12, 144], [-12, 146], [-12, 148], [-12, 150], [-12, 152], [-12, 154], [-12, 156], [-12, 158], [-12, 160], [-12, 162], [-12, 164], [-12, 166], [-12, 168], [-12, 170], [-12, 172], [-12, 174], [-12, 176], [-12, 178], [-12, 180], [-12, 182], [-12, 184], [-12, 186], [-12, 188], [-12, 190], [-12, 192]]).astype(jnp.int32)
        self.QBERT_MOVE_OUT_PYRAMID_2 = jnp.array([[0, 0], [0, -1], [0, -2], [0, -3], [0, -4], [0, -5], [0, -6], [0, -7], [0, -8], [-1, -9], [-2, -10], [-3, -11], [-4, -12], [-5, -13], [-6, -14], [-7, -13], [-8, -12], [-9, -11], [-10, -10], [-11, -9], [-12, -8], [-12, -7], [-12, -6], [-12, -5], [-12, -4], [-12, -3], [-12, -2], [-12, -1], [-12, 0], [-12, 2], [-12, 4], [-12, 6], [-12, 8], [-12, 10], [-12, 12], [-12, 14], [-12, 16], [-12, 18], [-12, 20], [-12, 22], [-12, 24], [-12, 26], [-12, 28], [-12, 30], [-12, 32], [-12, 34], [-12, 36], [-12, 38], [-12, 40], [-12, 42], [-12, 44], [-12, 46], [-12, 48], [-12, 50], [-12, 52], [-12, 54], [-12, 56], [-12, 58], [-12, 60], [-12, 62], [-12, 64], [-12, 66], [-12, 68], [-12, 70], [-12, 72], [-12, 74], [-12, 76], [-12, 78], [-12, 80], [-12, 82], [-12, 84], [-12, 86], [-12, 88], [-12, 90], [-12, 92], [-12, 94], [-12, 96], [-12, 98], [-12, 100], [-12, 102], [-12, 104], [-12, 106], [-12, 108], [-12, 110], [-12, 112], [-12, 114], [-12, 116], [-12, 118], [-12, 120], [-12, 122], [-12, 124], [-12, 126], [-12, 128], [-12, 130], [-12, 132], [-12, 134], [-12, 136], [-12, 138], [-12, 140], [-12, 142], [-12, 144], [-12, 146], [-12, 148], [-12, 150], [-12, 152], [-12, 154], [-12, 156], [-12, 158], [-12, 160], [-12, 162], [-12, 164]]).astype(jnp.int32)
        self.QBERT_MOVE_OUT_PYRAMID_3 = jnp.array([[0, 0], [0, -1], [0, -2], [0, -3], [0, -4], [0, -5], [0, -6], [0, -7], [0, -8], [-1, -9], [-2, -10], [-3, -11], [-4, -12], [-5, -13], [-6, -14], [-7, -13], [-8, -12], [-9, -11], [-10, -10], [-11, -9], [-12, -8], [-12, -7], [-12, -6], [-12, -5], [-12, -4], [-12, -3], [-12, -2], [-12, -1], [-12, 0], [-12, 2], [-12, 4], [-12, 6], [-12, 8], [-12, 10], [-12, 12], [-12, 14], [-12, 16], [-12, 18], [-12, 20], [-12, 22], [-12, 24], [-12, 26], [-12, 28], [-12, 30], [-12, 32], [-12, 34], [-12, 36], [-12, 38], [-12, 40], [-12, 42], [-12, 44], [-12, 46], [-12, 48], [-12, 50], [-12, 52], [-12, 54], [-12, 56], [-12, 58], [-12, 60], [-12, 62], [-12, 64], [-12, 66], [-12, 68], [-12, 70], [-12, 72], [-12, 74], [-12, 76], [-12, 78], [-12, 80], [-12, 82], [-12, 84], [-12, 86], [-12, 88], [-12, 90], [-12, 92], [-12, 94], [-12, 96], [-12, 98], [-12, 100], [-12, 102], [-12, 104], [-12, 106], [-12, 108], [-12, 110], [-12, 112], [-12, 114], [-12, 116], [-12, 118], [-12, 120], [-12, 122], [-12, 124], [-12, 126], [-12, 128], [-12, 130], [-12, 132]]).astype(jnp.int32)
        self.QBERT_MOVE_OUT_PYRAMID_4 = jnp.array([[0, 0], [0, -1], [0, -2], [0, -3], [0, -4], [0, -5], [0, -6], [0, -7], [0, -8], [-1, -9], [-2, -10], [-3, -11], [-4, -12], [-5, -13], [-6, -14], [-7, -13], [-8, -12], [-9, -11], [-10, -10], [-11, -9], [-12, -8], [-12, -7], [-12, -6], [-12, -5], [-12, -4], [-12, -3], [-12, -2], [-12, -1], [-12, 0], [-12, 2], [-12, 4], [-12, 6], [-12, 8], [-12, 10], [-12, 12], [-12, 14], [-12, 16], [-12, 18], [-12, 20], [-12, 22], [-12, 24], [-12, 26], [-12, 28], [-12, 30], [-12, 32], [-12, 34], [-12, 36], [-12, 38], [-12, 40], [-12, 42], [-12, 44], [-12, 46], [-12, 48], [-12, 50], [-12, 52], [-12, 54], [-12, 56], [-12, 58], [-12, 60], [-12, 62], [-12, 64], [-12, 66], [-12, 68], [-12, 70], [-12, 72], [-12, 74], [-12, 76], [-12, 78], [-12, 80], [-12, 82], [-12, 84], [-12, 86], [-12, 88], [-12, 90], [-12, 92], [-12, 94], [-12, 96], [-12, 98], [-12, 100], [-12, 102], [-12, 104], [-12, 106]]).astype(jnp.int32)
        self.QBERT_MOVE_OUT_PYRAMID_5 = jnp.array([[0, 0], [0, -1], [0, -2], [0, -3], [0, -4], [0, -5], [0, -6], [0, -7], [0, -8], [-1, -9], [-2, -10], [-3, -11], [-4, -12], [-5, -13], [-6, -14], [-7, -13], [-8, -12], [-9, -11], [-10, -10], [-11, -9], [-12, -8], [-12, -7], [-12, -6], [-12, -5], [-12, -4], [-12, -3], [-12, -2], [-12, -1], [-12, 0], [-12, 2], [-12, 4], [-12, 6], [-12, 8], [-12, 10], [-12, 12], [-12, 14], [-12, 16], [-12, 18], [-12, 20], [-12, 22], [-12, 24], [-12, 26], [-12, 28], [-12, 30], [-12, 32], [-12, 34], [-12, 36], [-12, 38], [-12, 40], [-12, 42], [-12, 44], [-12, 46], [-12, 48], [-12, 50], [-12, 52], [-12, 54], [-12, 56], [-12, 58], [-12, 60], [-12, 62], [-12, 64], [-12, 66], [-12, 68], [-12, 70], [-12, 72], [-12, 74], [-12, 76]]).astype(jnp.int32)
        self.QBERT_MOVE_OUT_PYRAMID_6 = jnp.array([[0, 0], [0, -1], [0, -2], [0, -3], [0, -4], [0, -5], [0, -6], [0, -7], [0, -8], [-1, -9], [-2, -10], [-3, -11], [-4, -12], [-5, -13], [-6, -14], [-7, -13], [-8, -12], [-9, -11], [-10, -10], [-11, -9], [-12, -8], [-12, -7], [-12, -6], [-12, -5], [-12, -4], [-12, -3], [-12, -2], [-12, -1], [-12, 0], [-12, 2], [-12, 4], [-12, 6], [-12, 8], [-12, 10], [-12, 12], [-12, 14], [-12, 16], [-12, 18], [-12, 20], [-12, 22], [-12, 24], [-12, 26], [-12, 28], [-12, 30], [-12, 32], [-12, 34], [-12, 36], [-12, 38], [-12, 40], [-12, 42], [-12, 44], [-12, 46]]).astype(jnp.int32)

        #TODO

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: QbertState) -> jnp.ndarray:
        """ Responsible for the graphical representation of the game """

        raster = jnp.zeros((self.consts.HEIGHT, self.consts.WIDTH, 3))

        # Render background
        raster = aj.render_at(raster, 0, 0, jnp.where(state.step_counter == state.green_ball_freeze_step, self.BACKGROUND, self.FREEZE))


        # Render cubes in pyramid
        raster = aj.render_at(raster, 68, 40, self.CUBE_SHADOW_RIGHT[state.round_number - 1])

        raster = aj.render_at(raster, 56, 69, self.CUBE_SHADOW_RIGHT[state.round_number - 1])
        raster = aj.render_at(raster, 80, 69, self.CUBE_SHADOW_LEFT[state.round_number - 1])

        raster = aj.render_at(raster, 44, 98, self.CUBE_SHADOW_RIGHT[state.round_number - 1])
        raster = aj.render_at(raster, 68, 98, self.CUBE_SHADOW_RIGHT[state.round_number - 1])
        raster = aj.render_at(raster, 92, 98, self.CUBE_SHADOW_LEFT[state.round_number - 1])

        raster = aj.render_at(raster, 32, 127, self.CUBE_SHADOW_RIGHT[state.round_number - 1])
        raster = aj.render_at(raster, 56, 127, self.CUBE_SHADOW_RIGHT[state.round_number - 1])
        raster = aj.render_at(raster, 80, 127, self.CUBE_SHADOW_LEFT[state.round_number - 1])
        raster = aj.render_at(raster, 104, 127, self.CUBE_SHADOW_LEFT[state.round_number - 1])

        raster = aj.render_at(raster, 20, 156, self.CUBE_SHADOW_RIGHT[state.round_number - 1])
        raster = aj.render_at(raster, 44, 156, self.CUBE_SHADOW_RIGHT[state.round_number - 1])
        raster = aj.render_at(raster, 68, 156, self.CUBE_SHADOW_RIGHT[state.round_number - 1])
        raster = aj.render_at(raster, 92, 156, self.CUBE_SHADOW_LEFT[state.round_number - 1])
        raster = aj.render_at(raster, 116, 156, self.CUBE_SHADOW_LEFT[state.round_number - 1])

        raster = aj.render_at(raster, 8, 185, self.CUBE_SHADOW_RIGHT[state.round_number - 1])
        raster = aj.render_at(raster, 32, 185, self.CUBE_SHADOW_RIGHT[state.round_number - 1])
        raster = aj.render_at(raster, 56, 185, self.CUBE_SHADOW_RIGHT[state.round_number - 1])
        raster = aj.render_at(raster, 80, 185, self.CUBE_SHADOW_LEFT[state.round_number - 1])
        raster = aj.render_at(raster, 104, 185, self.CUBE_SHADOW_LEFT[state.round_number - 1])
        raster = aj.render_at(raster, 128, 185, self.CUBE_SHADOW_LEFT[state.round_number - 1])

        # Render flying discs
        raster = jnp.where(state.last_pyramid[2][0] == -2, aj.render_at(raster, 38, 84, self.DISC), raster)
        raster = jnp.where(state.last_pyramid[2][3] == -2, aj.render_at(raster, 110, 84, self.DISC), raster)
        raster = jnp.where(state.last_pyramid[4][0] == -2, aj.render_at(raster, 14, 142, self.DISC), raster)
        raster = jnp.where(state.last_pyramid[4][5] == -2, aj.render_at(raster, 134, 142, self.DISC), raster)

        # Render the color on top of the cubes
        pyra = state.pyramid.at[state.player_position[1], state.player_position[0]].set(state.last_pyramid[state.player_position[1]][state.player_position[0]])
        raster = jax.lax.fori_loop(
            lower = 1,
            upper = 7,
            body_fun = lambda i, val: jax.lax.fori_loop(
                lower = 1,
                upper = i + 1,
                body_fun = lambda j, val2: jnp.select(
                    condlist=[
                        pyra[i, j] == 0,
                        pyra[i, j] == 1,
                        pyra[i, j] == 2
                    ],
                    choicelist=[
                        aj.render_at(val2, self.COLOR_POSITIONS[jnp.array(i * (i - 1) / 2 + (j - 1)).astype(jnp.int32)][0], self.COLOR_POSITIONS[jnp.array(i * (i - 1) / 2 + (j - 1)).astype(jnp.int32)][1] + 2, self.START_COLOR),
                        aj.render_at(val2, self.COLOR_POSITIONS[jnp.array(i * (i - 1) / 2 + (j - 1)).astype(jnp.int32)][0], self.COLOR_POSITIONS[jnp.array(i * (i - 1) / 2 + (j - 1)).astype(jnp.int32)][1] + 2, self.INTERMEDIATE_COLOR),
                        aj.render_at(val2, self.COLOR_POSITIONS[jnp.array(i * (i - 1) / 2 + (j - 1)).astype(jnp.int32)][0], self.COLOR_POSITIONS[jnp.array(i * (i - 1) / 2 + (j - 1)).astype(jnp.int32)][1] + 2, self.DESTINATION_COLOR)
                    ],
                    default=val2
                ),
                init_val=val
            ),
            init_val = raster,
        )

        # Render the player score only if player is not the first or second layer of the pyramid
        player_score_digits = aj.int_to_digits(state.player_score, max_digits=5)
        raster = jnp.where(state.player_position[1] >= 3, aj.render_label_selective(raster, 34, 6, player_score_digits, self.PLAYER_SCORE_DIGIT, 0, 5, spacing=8), raster)

        # Render players remaining lives
        raster = jax.lax.fori_loop(
            lower = 0,
            upper = state.lives,
            body_fun = lambda i, val: jnp.where(state.player_position[1] >= 3, aj.render_at(val, self.LIVE_POSITIONS[i][0], self.LIVE_POSITIONS[i][1], self.QBERT_LIVE), val),
            init_val=raster
        )

        # Render qbert
        qbert_j = state.player_last_position[0]
        qbert_i = state.player_last_position[1]
        move_positions = jnp.array([self.QBERT_MOVE_LEFT_UP, self.QBERT_MOVE_LEFT_DOWN, self.QBERT_MOVE_RIGHT_DOWN, self.QBERT_MOVE_RIGHT_UP]).astype(jnp.int32)
        qbert_sprites = jnp.array([self.QBERT_LEFT_UP, self.QBERT_LEFT_DOWN, self.QBERT_RIGHT_DOWN, self.QBERT_RIGHT_UP]).astype(jnp.int32)
        raster = jax.lax.switch(
            index=state.player_position_category,
            branches=[
                # Normal on pyramid
                lambda state: aj.render_at(raster,
                                           self.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][0] + move_positions[state.player_direction][state.player_moving_counter][0],
                                           self.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][1] + move_positions[state.player_direction][state.player_moving_counter][1],
                                           qbert_sprites[state.player_direction]),
                # On disc bottom
                lambda state: aj.render_at(raster,
                                           self.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][0] + self.QBERT_MOVE_DISC_LEFT_BOTTOM[state.player_moving_counter][0] * (state.player_direction == 0) + self.QBERT_MOVE_DISC_RIGHT_BOTTOM[state.player_moving_counter][0] * (state.player_direction == 3),
                                           self.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][1] + self.QBERT_MOVE_DISC_LEFT_BOTTOM[state.player_moving_counter][1] * (state.player_direction == 0) + self.QBERT_MOVE_DISC_RIGHT_BOTTOM[state.player_moving_counter][1] * (state.player_direction == 3),
                                           qbert_sprites[state.player_direction]),
                # On disc top
                lambda state: aj.render_at(raster,
                                           self.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][0] + self.QBERT_MOVE_DISC_LEFT_TOP[state.player_moving_counter][0] * (state.player_direction == 0) + self.QBERT_MOVE_DISC_RIGHT_TOP[state.player_moving_counter][0] * (state.player_direction == 3),
                                           self.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][1] + self.QBERT_MOVE_DISC_LEFT_TOP[state.player_moving_counter][1] * (state.player_direction == 0) + self.QBERT_MOVE_DISC_RIGHT_TOP[state.player_moving_counter][1] * (state.player_direction == 3),
                                           qbert_sprites[state.player_direction]),
                # Out of pyramid height 1
                lambda state: aj.render_at(raster,
                                           self.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][0] + self.QBERT_MOVE_OUT_PYRAMID_1[state.player_moving_counter][0] * (- 1 * (state.player_direction > 1)) + self.QBERT_MOVE_OUT_PYRAMID_6[state.player_moving_counter][0] * (state.player_direction < 1),
                                           self.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][1] + self.QBERT_MOVE_OUT_PYRAMID_1[state.player_moving_counter][1],
                                           qbert_sprites[state.player_direction]),
                # Out of pyramid height 2
                lambda state: aj.render_at(raster,
                                           self.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][0] + self.QBERT_MOVE_OUT_PYRAMID_2[state.player_moving_counter][0] * (- 1 * (state.player_direction > 1)) + self.QBERT_MOVE_OUT_PYRAMID_6[state.player_moving_counter][0] * (state.player_direction < 1),
                                           self.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][1] + self.QBERT_MOVE_OUT_PYRAMID_2[state.player_moving_counter][1],
                                           qbert_sprites[state.player_direction]),
                # Out of pyramid height 3
                lambda state: aj.render_at(raster,
                                           self.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][0] + self.QBERT_MOVE_OUT_PYRAMID_3[state.player_moving_counter][0] * (- 1 * (state.player_direction > 1)) + self.QBERT_MOVE_OUT_PYRAMID_6[state.player_moving_counter][0] * (state.player_direction < 1),
                                           self.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][1] + self.QBERT_MOVE_OUT_PYRAMID_3[state.player_moving_counter][1],
                                           qbert_sprites[state.player_direction]),
                # Out of pyramid height 4
                lambda state: aj.render_at(raster,
                                           self.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][0] + self.QBERT_MOVE_OUT_PYRAMID_4[state.player_moving_counter][0] * (- 1 * (state.player_direction > 1)) + self.QBERT_MOVE_OUT_PYRAMID_6[state.player_moving_counter][0] * (state.player_direction < 1),
                                           self.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][1] + self.QBERT_MOVE_OUT_PYRAMID_4[state.player_moving_counter][1],
                                           qbert_sprites[state.player_direction]),
                # Out of pyramid height 5
                lambda state: aj.render_at(raster,
                                           self.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][0] + self.QBERT_MOVE_OUT_PYRAMID_5[state.player_moving_counter][0] * (- 1 * (state.player_direction > 1)) + self.QBERT_MOVE_OUT_PYRAMID_6[state.player_moving_counter][0] * (state.player_direction < 1),
                                           self.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][1] + self.QBERT_MOVE_OUT_PYRAMID_5[state.player_moving_counter][1],
                                           qbert_sprites[state.player_direction]),
                # Out of pyramid height 6
                lambda state: aj.render_at(raster,
                                           self.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][0] + self.QBERT_MOVE_OUT_PYRAMID_6[state.player_moving_counter][0] * (- 1 * (state.player_direction > 1)) + self.QBERT_MOVE_OUT_PYRAMID_6[state.player_moving_counter][0] * (state.player_direction < 1),
                                           self.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][1] + self.QBERT_MOVE_OUT_PYRAMID_6[state.player_moving_counter][1],
                                           qbert_sprites[state.player_direction]),
            ],
            operand=state
        )

        # Render Sam
        sam_j = state.sam_position[0]
        sam_i = state.sam_position[1]
        raster = jnp.where(jnp.logical_and(state.sam_position[0] == -1, state.sam_position[1] == -1), raster, aj.render_at(raster,
                                                                                                                           self.QBERT_POSITIONS[jnp.array(sam_i * (sam_i - 1) / 2 + (sam_j - 1)).astype(jnp.int32)][0],
                                                                                                                           self.QBERT_POSITIONS[jnp.array(sam_i * (sam_i - 1) / 2 + (sam_j - 1)).astype(jnp.int32)][1] + 1,
                                                                                                                           self.SAM))

        # Render green ball
        green_ball_j = state.green_ball_position[0]
        green_ball_i = state.green_ball_position[1]
        green_ball_index = jnp.floor(state.enemy_moving_counter / jnp.ceil(self.consts.ENEMY_MOVE_TICK[state.level_number] / 2).astype(jnp.int32)).astype(jnp.int32)
        raster = jnp.where(jnp.logical_and(state.green_ball_position[0] == -1, state.green_ball_position[1] == -1), raster, aj.render_at(raster,
                                                                                                                                         self.QBERT_POSITIONS[jnp.array(green_ball_i * (green_ball_i - 1) / 2 + (green_ball_j - 1)).astype(jnp.int32)][0] + self.BALL_MOVE[green_ball_index][0],
                                                                                                                                         self.QBERT_POSITIONS[jnp.array(green_ball_i * (green_ball_i - 1) / 2 + (green_ball_j - 1)).astype(jnp.int32)][1] + self.BALL_MOVE[green_ball_index][1],
                                                                                                                                         self.GREEN_BALL[green_ball_index]))

        # Render purple ball
        purple_ball_j = state.purple_ball_position[0]
        purple_ball_i = state.purple_ball_position[1]
        purple_ball_index = jnp.floor(state.enemy_moving_counter / jnp.ceil(self.consts.ENEMY_MOVE_TICK[state.level_number] / 2).astype(jnp.int32)).astype(jnp.int32)
        raster = jnp.where(jnp.logical_and(state.purple_ball_position[0] == -1, state.purple_ball_position[1] == -1), raster, aj.render_at(raster,
                                                                                                                                           self.QBERT_POSITIONS[jnp.array(purple_ball_i * (purple_ball_i - 1) / 2 + (purple_ball_j - 1)).astype(jnp.int32)][0] + self.BALL_MOVE[purple_ball_index][0],
                                                                                                                                           self.QBERT_POSITIONS[jnp.array(purple_ball_i * (purple_ball_i - 1) / 2 + (purple_ball_j - 1)).astype(jnp.int32)][1] + self.BALL_MOVE[purple_ball_index][1],
                                                                                                                                           self.PURPLE_BALL[purple_ball_index]))
        # Render snake
        snake_j = state.snake_position[0]
        snake_i = state.snake_position[1]
        snake_index = jnp.floor(state.enemy_moving_counter / jnp.ceil(self.consts.ENEMY_MOVE_TICK[state.level_number] / 5).astype(jnp.int32)).astype(jnp.int32)
        raster = jnp.where(jnp.logical_and(state.snake_position[0] == -1, state.snake_position[1] == -1), raster, aj.render_at(raster,
                                                                                                                               self.QBERT_POSITIONS[jnp.array(snake_i * (snake_i - 1) / 2 + (snake_j - 1)).astype(jnp.int32)][0] + self.SNAKE_MOVE[snake_index][0],
                                                                                                                               self.QBERT_POSITIONS[jnp.array(snake_i * (snake_i - 1) / 2 + (snake_j - 1)).astype(jnp.int32)][1] + self.SNAKE_MOVE[snake_index][1],
                                                                                                                               self.SNAKE[snake_index]))
        # Render red balls
        def render_red_ball(i, r):
            red_ball_j = state.red_ball_positions[i][0]
            red_ball_i = state.red_ball_positions[i][1]
            red_ball_index = jnp.floor(state.enemy_moving_counter / jnp.ceil(self.consts.ENEMY_MOVE_TICK[state.level_number] / 2).astype(jnp.int32)).astype(jnp.int32)
            r = jnp.where(jnp.logical_and(state.red_ball_positions[0][0] == -1, state.red_ball_positions[0][1] == -1), raster, aj.render_at(r,
                                                                                                                                            self.QBERT_POSITIONS[jnp.array(red_ball_i * (red_ball_i - 1) / 2 + (red_ball_j - 1)).astype(jnp.int32)][0] + self.BALL_MOVE[red_ball_index][0],
                                                                                                                                            self.QBERT_POSITIONS[jnp.array(red_ball_i * (red_ball_i - 1) / 2 + (red_ball_j - 1)).astype(jnp.int32)][1] + self.BALL_MOVE[red_ball_index][1],
                                                                                                                                            self.RED_BALL[red_ball_index]))
            return r

        raster = jax.lax.fori_loop(
            lower=0,
            upper=3,
            body_fun=render_red_ball,
            init_val=raster
        )

        #TODO
        return raster
