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
import os
from typing import Tuple
from functools import partial

import chex
import jax
import jax.numpy as jnp
from flax import struct

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action, ObjectObservation
import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils


def _get_default_asset_config() -> tuple:
    return (
        {'name': 'background', 'type': 'background', 'file': 'background.npy'},
        {'name': 'freeze', 'type': 'single', 'file': 'freeze.npy'},
        {'name': 'score_digits', 'type': 'digits', 'pattern': 'score/score_{}.npy'},
        {'name': 'qbert_sprites', 'type': 'group', 'files': [
            'qbert/qbert_left_up.npy',
            'qbert/qbert_left_down.npy',
            'qbert/qbert_right_down.npy',
            'qbert/qbert_right_up.npy',
        ]},
        {'name': 'cube_shadow_left', 'type': 'group', 'files': [
            'cube/cube_shadow_left_1.npy',
            'cube/cube_shadow_left_2.npy',
            'cube/cube_shadow_left_3.npy',
            'cube/cube_shadow_left_4.npy',
            'cube/cube_shadow_left_5.npy',
        ]},
        {'name': 'cube_shadow_right', 'type': 'group', 'files': [
            'cube/cube_shadow_right_1.npy',
            'cube/cube_shadow_right_2.npy',
            'cube/cube_shadow_right_3.npy',
            'cube/cube_shadow_right_4.npy',
            'cube/cube_shadow_right_5.npy',
        ]},
        {'name': 'color_start', 'type': 'single', 'file': 'color/color_start.npy'},
        {'name': 'color_intermediate', 'type': 'single', 'file': 'color/color_intermediate.npy'},
        {'name': 'color_destination', 'type': 'single', 'file': 'color/color_destination.npy'},
        {'name': 'win_animation', 'type': 'group', 'files': [
            f'win_animation/win_animation_{str(i).zfill(2)}.npy' for i in range(32)
        ]},
        {'name': 'qbert_live', 'type': 'single', 'file': 'qbert_live.npy'},
        {'name': 'disc', 'type': 'single', 'file': 'disc.npy'},
        {'name': 'sam', 'type': 'single', 'file': 'enemies/sam.npy'},
        {'name': 'green_ball', 'type': 'group', 'files': [
            'enemies/green_ball_1.npy',
            'enemies/green_ball_2.npy',
        ]},
        {'name': 'purple_ball', 'type': 'group', 'files': [
            'enemies/purple_ball_1.npy',
            'enemies/purple_ball_2.npy',
        ]},
        {'name': 'red_ball', 'type': 'group', 'files': [
            'enemies/red_ball_1.npy',
            'enemies/red_ball_2.npy',
        ]},
        {'name': 'snake', 'type': 'group', 'files': [
            'enemies/snake_1.npy',
            'enemies/snake_2.npy',
            'enemies/snake_3.npy',
            'enemies/snake_2.npy',
            'enemies/snake_1.npy',
        ]},
        {'name': 'dead', 'type': 'single', 'file': 'death.npy'},
    )


class QbertConstants(struct.PyTreeNode):
    WIDTH: int = struct.field(pytree_node=False, default=160)
    HEIGHT: int = struct.field(pytree_node=False, default=210)
    DIFFICULTY: int = struct.field(pytree_node=False, default=1)
    ENEMY_MOVE_TICK: Tuple[int, ...] = struct.field(pytree_node=False, default=(65, 60, 55, 50, 45))
    ENEMY_SPAWN: Tuple[int, ...] = struct.field(pytree_node=False, default=(300, 600, 2000, 3000))
    COLLISION_CONFIRM_STEPS: int = struct.field(pytree_node=False, default=3)
    FIRST_SPAWN_DELAY: int = struct.field(pytree_node=False, default=115)
    GREEN_ONLY_WINDOW: int = struct.field(pytree_node=False, default=200)
    COILY_DELAY: int = struct.field(pytree_node=False, default=200)
    SPAWN_INTERVAL: int = struct.field(pytree_node=False, default=250)
    ASSET_CONFIG: tuple = struct.field(pytree_node=False, default_factory=_get_default_asset_config)


class QbertState(struct.PyTreeNode):
    player_score: chex.Array
    lives: chex.Array
    pyramid: chex.Array
    last_pyramid: chex.Array
    player_position: chex.Array
    player_last_position: chex.Array
    player_direction: chex.Array
    is_player_moving: chex.Array
    player_moving_counter: chex.Array
    player_position_category: chex.Array
    level_number: chex.Array
    round_number: chex.Array
    green_ball_freeze_step: chex.Array
    enemy_moving_counter: chex.Array
    red_ball_positions: chex.Array
    purple_ball_position: chex.Array
    snake_position: chex.Array
    green_ball_position: chex.Array
    sam_position: chex.Array
    step_counter: chex.Array
    next_round_animation_counter: chex.Array
    dead_animation_counter: chex.Array
    just_spawned: chex.Array
    snake_lock: chex.Array
    same_cell_frames: chex.Array
    last_spawn_step: chex.Array
    enemy_spawn_count: chex.Array
    prng_state: chex.PRNGKey


class QbertObservation(struct.PyTreeNode):
    player: ObjectObservation
    red_balls: ObjectObservation
    purple_ball: ObjectObservation
    snake: ObjectObservation
    green_ball: ObjectObservation
    sam: ObjectObservation
    player_score: chex.Array
    lives: chex.Array
    pyramid: chex.Array
    level_number: chex.Array
    round_number: chex.Array


class QbertInfo(struct.PyTreeNode):
    time: chex.Array

class JaxQbert(JaxEnvironment[QbertState, QbertObservation, QbertInfo, QbertConstants]):
    ACTION_SET: jnp.ndarray = jnp.array([
        Action.NOOP,
        Action.FIRE,
        Action.UP,
        Action.RIGHT,
        Action.LEFT,
        Action.DOWN,
    ], dtype=jnp.int32)

    def __init__(self, consts: QbertConstants = None):
        consts = consts or QbertConstants()
        super().__init__(consts)

        self.renderer = QbertRenderer(consts)
        self.action_mapping = jnp.array([[0, 0],[0,0],[0,-1],[1,1],[-1,-1],[0,1]]).astype(jnp.int32)
        self._enemy_move_tick = jnp.array(consts.ENEMY_MOVE_TICK, dtype=jnp.int32)

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
            prng_state=jax.random.split(key, 10),
            just_spawned=jnp.array(1).astype(jnp.int32),
            snake_lock=jnp.array([-1,-1]).astype(jnp.int32),
            same_cell_frames=jnp.array(0).astype(jnp.int32),
            last_spawn_step=jnp.array(0).astype(jnp.int32),
            enemy_spawn_count=jnp.array(0).astype(jnp.int32),
            next_round_animation_counter=jnp.array(0).astype(jnp.int32),
            dead_animation_counter=jnp.array(0).astype(jnp.int32),
        )

        initial_obs = self._get_observation(state)

        return initial_obs, state
    

    @partial(jax.jit, static_argnums=(0,))
    def checkField(self, state: QbertState, character: chex.Numeric, number: chex.Numeric, position: chex.Array):
        """ checks if a charcter stepped outside the map or on a disk

            :param state: the state of the game
            :param character: the character to check 0=player, 1=red, 2=purple, 3=snake, 4=green, 5=sam
            :param number: if the character is red, which index of red should be used otherwise arbitrary
            :param position: the position of the character
            """
        # Snake (Coily) stepping onto a disk: remove snake, award 500 points if lives==3.
        snake_on_disk = jnp.logical_and(character == 3, state.pyramid[position[1], position[0]] == -2)
        pyra, pos, lives, points = jax.lax.cond(
            pred=snake_on_disk,
            true_fun=lambda vals: (
                vals[0].pyramid,
                jnp.array([-1, -1], dtype=jnp.int32),
                vals[0].lives,
                jnp.where(vals[0].lives == 3, jnp.array(500, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32)),
            ),
            false_fun=lambda vals: jax.lax.cond(
                pred=jnp.logical_and(vals[0].pyramid[vals[1][1], vals[1][0]] == -2, vals[2] == 0),
                true_fun=lambda v: self.stepped_on_disk(v[0]),
                false_fun=lambda v: jax.lax.cond(
                    pred=v[0].pyramid[v[1][1]][v[1][0]] == -1,
                    true_fun=lambda v2: self.stepped_out_map(v2[0], v2[2]),
                    false_fun=lambda v2: (v2[0].pyramid, v2[1], v2[0].lives, jnp.array(0).astype(jnp.int32)),
                    operand=v
                ),
                operand=vals
            ),
            operand=(state, position, character, number)
        )
        return pyra, pos, lives, points

    @partial(jax.jit, static_argnums=(0,))
    def changeColors(self,state: QbertState, playerPos: chex.Array, samPos: chex.Array, pyramid: chex.Array, level: chex.Numeric, spawned: chex.Numeric):
        """ changes to colors of the pyramid to the appropiate values

            :param state: the state of the game
            :param playerPos: the position of the player
            :param samPos: the position of sam
            :param pyramid: the pyramid to change
            :param level: the new level
            :param spawned: whether the player just spawned or if he moved already
            """
        old_color=pyramid[playerPos[1]][playerPos[0]]
        pyramid=jax.lax.cond(
            pred=jnp.logical_and(samPos[0]!= -1, samPos[1] != -1),
            true_fun=lambda vals: vals[0].at[vals[1][1],vals[1][0]].set(0),
            false_fun=lambda vals: vals[0],
            operand=(pyramid,samPos)
        )
        pyramid=jax.lax.cond(
            pred=jnp.logical_and(jnp.logical_or(jnp.logical_or(playerPos[0] != 1, playerPos[1] != 1), spawned != 1), jnp.logical_or(playerPos[0] != state.player_position[0], playerPos[1] != state.player_position[1])),
            true_fun=lambda val:
                val[0].at[val[1][1],val[1][0]].set(jax.lax.switch(
                index=level - 1,
                branches=[
                    lambda vals: 2,
                    lambda vals: jnp.minimum(vals[0][vals[1][1]][vals[1][0]] + 1, jnp.array(2).astype(jnp.int32)),
                    lambda vals: jnp.mod(vals[0][vals[1][1]][vals[1][0]] + 2, 4),
                    lambda vals: jax.lax.cond(
                        pred=vals[0][vals[1][1]][vals[1][0]] == 2,
                        true_fun=lambda vals2: 1,
                        false_fun=lambda vals2: jnp.minimum(vals2 + 1, 2),
                        operand=vals[0][vals[1][1]][vals[1][0]]
                    ),
                    lambda vals: jnp.mod(vals[0][vals[1][1]][vals[1][0]] + 1 , 3),
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
                            samPos: chex.Array, green_ball_freeze_step: chex.Numeric,
                            same_cell_frames: chex.Numeric):
            """ checks whether a snake hit the bottom row or whether the player has collided with one of the other characters and adjusts the values accordingly.
            Death requires sustained overlap (COLLISION_CONFIRM_STEPS).
            :param lives: the new lives value
            :param playerPos: the position of the player
            :param redPos: the position of red
            :param purplePos: the position of purple
            :param greenPos: the position of green
            :param snakePos: the position of the snake
            :param green_ball_freeze_step: the steps until which the enemies are frozen
            """
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
            is_friendly = (index_if_hit == jnp.int32(4)) | (index_if_hit == jnp.int32(6))
            collision_hostile = collision & (~is_friendly)
            same_cell_frames_new = jnp.where(collision_hostile, same_cell_frames + 1, 0)
            confirm_kill = collision_hostile & (same_cell_frames_new >= self.consts.COLLISION_CONFIRM_STEPS)
            # Lose a life only when hostile and sustained overlap (confirm_kill).
            def _true_fun(vals):
                enemies_, lives_, idx_ = vals
                is_friendly_ = (idx_ == jnp.int32(4)) | (idx_ == jnp.int32(6))
                deduct = (~is_friendly_) & (same_cell_frames_new >= self.consts.COLLISION_CONFIRM_STEPS)
                lives_new = lives_ - jnp.where(deduct, jnp.int32(1), jnp.int32(0))
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
            same_cell_frames_out = jnp.where(confirm_kill, 0, same_cell_frames_new)
            purple_before = enemies[3]
            # Only remove enemy when friendly (always) or hostile and confirm_kill; else Coily stays until we actually deduct life.
            is_friendly_idx = (index == jnp.int32(4)) | (index == jnp.int32(6))
            should_remove = is_friendly_idx | confirm_kill
            enemies_removed = enemies.at[index].set(jnp.array([-1, -1], dtype=jnp.int32))
            enemies = jnp.where(should_remove, enemies_removed, enemies)
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

            # Freeze effect from green ball (index 4), Sam gives bonus points only.
            green_ball_freeze_step = green_ball_freeze_step + 206 * (index == 4)

            return (
                lives,
                [enemies[0], enemies[1], enemies[2]],
                enemies[3],
                enemies[4],
                enemies[5],
                enemies[6],
                points,
                green_ball_freeze_step,
                same_cell_frames_out
            )

    @partial(jax.jit, static_argnums=(0,))
    def stepped_out_map(self, state: QbertState, character : chex.Numeric):
        """ function used to update the values if a character stepped out of the map
            :param state: the state of the game
            :param character: the character that stepped out of map (number correspond the same way as checkField())
            """
        pos,player_lives,points=jax.lax.cond(
            pred=character == 0,
            true_fun=lambda state2: (jnp.array([1,1]).astype(jnp.int32), jnp.array(state2[0].lives - 1).astype(jnp.int32), jnp.array(0).astype(jnp.int32)),
            false_fun=lambda state2: jax.lax.cond(
                pred=state.lives == 3,
                true_fun=lambda vals: (jnp.array([-1,-1]).astype(jnp.int32), 0, jnp.array(500).astype(jnp.int32)),
                false_fun=lambda vals: (jnp.array([-1,-1]).astype(jnp.int32), 0, jnp.array(0).astype(jnp.int32)),
                operand=state2
            ),
            operand=(state,character)
        )
        return state.pyramid, pos,player_lives, points

    @partial(jax.jit, static_argnums=(0,))
    def stepped_on_disk(self, state: QbertState):
        """ function used to update the values if a player stepped on a disk
            :param state: the state of the game
            """
        pyramid=jax.lax.cond(
            pred=state.player_position[0] > 1,
            true_fun=lambda state2: state2.pyramid.at[state2.player_position[1] - 1,state2.player_position[0]].set(-1),
            false_fun=lambda state2: state2.pyramid.at[state2.player_position[1] - 1, state2.player_position[0] - 1].set(-1),
            operand=state
        )
        return pyramid, jnp.array([1,1]).astype(jnp.int32), state.lives, jnp.array(0).astype(jnp.int32)

    @partial(jax.jit, static_argnums=(0,))
    def move(self, state: QbertState, action: chex.Array, position: chex.Array):
        """ universal function to move a character
            :param state: the state of the game
            :param action: the action to execute
            :param position: the position of the character
            """
        new_pos = jax.lax.cond(
            pred=jnp.logical_and(position[0]!=-1, action > 1),
            true_fun=lambda vals: jnp.array([vals[2][0] + vals[0][vals[1]][0], vals[2][1] + vals[0][vals[1]][1]]).astype(jnp.int32),
            false_fun=lambda vals: jnp.array([vals[2][0], vals[2][1]]).astype(jnp.int32),
            operand=(self.action_mapping, action, position)
        )

        return new_pos

    @partial(jax.jit, static_argnums=(0,))
    def move_purple_ball(self,state: QbertState):
        """ function used to move the purple balls
            :param state: the state of the game
            """
        pos = jax.lax.cond(
            pred = jax.random.uniform(state.prng_state[0], (), minval=1, maxval=256).astype(jnp.uint8) < 128,
            true_fun=lambda state2: self.move(state2,Action.DOWN, state2.purple_ball_position),
            false_fun=lambda state2: self.move(state2,Action.RIGHT, state2.purple_ball_position),
            operand=state
        )

        return pos

    @partial(jax.jit, static_argnums=(0,))
    def move_green_ball(self,state: QbertState):
        """ function used to move the green balls
            :param state: the state of the game
            """
        pos = jax.lax.cond(
            pred = jax.random.uniform(state.prng_state[1], (), minval=1, maxval=256).astype(jnp.uint8) < 128,
            true_fun=lambda state2: self.move(state2,Action.DOWN, state2.green_ball_position),
            false_fun=lambda state2: self.move(state2,Action.RIGHT, state2.green_ball_position),
            operand=state
        )

        return pos

    @partial(jax.jit, static_argnums=(0,))
    def move_sam(self,state: QbertState):
        """ function used to move the sams
            :param state: the state of the game
            """
        pos = jax.lax.cond(
            pred = jax.random.uniform(state.prng_state[2], (), minval=1, maxval=256).astype(jnp.uint8) < 128,
            true_fun=lambda state2: self.move(state2,Action.DOWN, state2.sam_position),
            false_fun=lambda state2: self.move(state2,Action.RIGHT, state2.sam_position),
            operand=state
        )

        return pos

    @partial(jax.jit, static_argnums=(0,))
    def move_red_balls(self,state: QbertState):
        """ function used to move the red balls
            :param state: the state of the game
            """
        pos1 = jax.lax.cond(
            pred = jax.random.uniform(state.prng_state[3], (), minval=1, maxval=256).astype(jnp.uint8) < 128,
            true_fun=lambda state2: self.move(state2,Action.DOWN, state2.red_ball_positions[0]),
            false_fun=lambda state2: self.move(state2,Action.RIGHT, state2.red_ball_positions[0]),
            operand=state
        )
        pos2 = jax.lax.cond(
            pred = jax.random.uniform(state.prng_state[4], (), minval=1, maxval=256).astype(jnp.uint8) < 128,
            true_fun=lambda state2: self.move(state2,Action.DOWN, state2.red_ball_positions[1]),
            false_fun=lambda state2: self.move(state2,Action.RIGHT, state2.red_ball_positions[1]),
                operand=state
        )
        pos3 = jax.lax.cond(
            pred = jax.random.uniform(state.prng_state[5], (), minval=1, maxval=256).astype(jnp.uint8) < 128,
            true_fun=lambda state2: self.move(state2,Action.DOWN, state2.red_ball_positions[2]),
            false_fun=lambda state2: self.move(state2,Action.RIGHT, state2.red_ball_positions[2]),
            operand=state
        )

        return jnp.array([pos1,pos2,pos3]).astype(jnp.int32)

    @partial(jax.jit, static_argnums=(0,))
    def move_snake(self, state: QbertState, target: chex.Array):
        """ function used to move the snake towards the target
            :param state: the current state
            :param target: what the snake should target
        """
        snake_pos = state.snake_position
        # When target is a disk (-2), allow direct step onto it so Coily always jumps off when player is on disk.
        target_is_disk = (target[0] >= 0) & (target[0] < 8) & (target[1] >= 0) & (target[1] < 8) & (state.pyramid[target[1], target[0]] == -2)
        diff = target - snake_pos
        adjacent_to_disk = target_is_disk & (jnp.abs(diff[0]) <= 1) & (jnp.abs(diff[1]) <= 1) & (jnp.any(diff != 0))
        result_direct = jnp.where(adjacent_to_disk, target, jnp.array([snake_pos[0], snake_pos[1]]))

        directions = jnp.array([[0, 1], [0, -1], [1, 1], [-1, -1]])
        # 1. Determine all four potential next positions
        candidate_positions = snake_pos + directions
        rows, cols = candidate_positions[:, 0], candidate_positions[:, 1]
        # 2. Build a validity mask to filter moves
        pyramid_values = state.pyramid[cols, rows]
        tile_mask = (pyramid_values != -1)
        is_target_mask = (rows == target[0]) & (cols == target[1])
        valid_mask = (tile_mask | is_target_mask)

        directions = jax.lax.fori_loop(
            lower=0,
            upper=4,
            body_fun=lambda i, d: d.at[i].set(jnp.where(valid_mask[i] == 1, d[i], jnp.array([100, 100]))),
            init_val=directions
        )

        direction = self.closest_direction(state.snake_position, directions, target)
        result_diagonal = jnp.array([state.snake_position[0] + direction[0], state.snake_position[1] + direction[1]])

        result = jnp.where(adjacent_to_disk, result_direct, result_diagonal)
        return result

    @partial(jax.jit, static_argnums=(0,))
    def closest_direction(self, pos, directions, target):
        """ Uses the euclidean distance to find the best direction
            :param pos: the current position
            :param directions: the array of vectors
            :param target: the target position
        """

        next_positions = pos + directions
        distance_vector_next_positions_target = target - next_positions[:]

        sim = jnp.linalg.norm(distance_vector_next_positions_target, axis=1)

        index= jnp.argmin(sim)

        return directions[index]

    @partial(jax.jit, static_argnums=(0,))
    def spawnCreatures(self, key: chex.PRNGKey, red_pos: chex.Array, purple_pos: chex.Array, green_pos: chex.Array, sam_pos: chex.Array, snake_pos: chex.Array, difficulty: chex.Numeric, step_counter: chex.Numeric, enemy_spawn_count: chex.Numeric, level: chex.Numeric):
        """Spawn enemies with timing: first spawn Sam, force Coily (via purple ball) as second spawn on levels 1 and 2, and no spawns when Coily is on board."""
        random = jax.random.uniform(key[0], (), minval=1, maxval=4000).astype(jnp.uint16)
        snake_on_board = jnp.logical_or(snake_pos[0] != -1, snake_pos[1] != -1)
        no_enemy = (jnp.all(red_pos[:, 0] == -1) & (purple_pos[0] == -1) & (green_pos[0] == -1) & (sam_pos[0] == -1))
        first_spawn_window = (step_counter >= self.consts.FIRST_SPAWN_DELAY) & (step_counter < self.consts.FIRST_SPAWN_DELAY + self.consts.GREEN_ONLY_WINDOW)
        # For levels 1 and 2, force first enemy to be Sam and the second to be Coily (via purple ball).
        in_low_levels = level <= 2
        is_first_spawn = (enemy_spawn_count == 0) & in_low_levels & no_enemy
        is_second_spawn = (enemy_spawn_count == 1) & in_low_levels
        # First spawn = Sam (green alien), not green ball.
        force_sam = is_first_spawn & first_spawn_window
        coily_available = step_counter >= self.consts.FIRST_SPAWN_DELAY + self.consts.COILY_DELAY

        # When Coily is on board, suppress all spawns (not just purple/red).
        roll_index = jax.lax.cond(
            pred=random < self.consts.ENEMY_SPAWN[0],
            true_fun=lambda i: 2,
            false_fun=lambda i: jax.lax.cond(
                pred=i < self.consts.ENEMY_SPAWN[1],
                true_fun=lambda j: 3,
                false_fun=lambda j: jax.lax.cond(
                    pred=j < self.consts.ENEMY_SPAWN[2],
                    true_fun=lambda k: jax.lax.cond(
                        pred=snake_on_board,
                        true_fun=lambda _: 4,
                        false_fun=lambda _: jnp.where(coily_available, 1, 4),
                        operand=None,
                    ),
                    false_fun=lambda k: jax.lax.cond(
                        pred=k < self.consts.ENEMY_SPAWN[3],
                        true_fun=lambda l: jax.lax.cond(
                            pred=snake_on_board,
                            true_fun=lambda _: 4,
                            false_fun=lambda _: jnp.where(coily_available, 0, 4),
                            operand=None,
                        ),
                        false_fun=lambda l: 4,
                        operand=k,
                    ),
                    operand=j,
                ),
                operand=i,
            ),
            operand=random,
        )
        # Force second spawn on low levels to be purple (which will later turn into Coily) if possible.
        force_purple = is_second_spawn & (~snake_on_board)
        creatureIndex = jnp.where(
            force_sam,
            3,  # Sam
            jnp.where(
                force_purple,
                1,  # purple ball -> Coily
                jnp.where(snake_on_board, 4, roll_index),
            ),
        )
        location=jax.lax.cond(
            pred=jax.random.uniform(key[1], (), minval=1, maxval=256).astype(jnp.uint8) < 128,
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
                    false_fun=lambda vals2: (vals2[0],vals2[1],vals2[2],vals2[3]),
                    operand=vals
                ),
                lambda vals: jax.lax.cond(
                    pred=jnp.logical_and(jnp.logical_and(vals[1][0] == -1, vals[1][1] == -1),jnp.logical_and(vals[4][0] == -1, vals[4][1] == -1)),
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
        """ subroutine used to spawn the red balls
            :param location: the location where to spawn the ball
            :param redPos: the position of the red balls
            """
        redPos=jax.lax.cond(
            pred=jnp.logical_and(redPos[0][0] == -1, redPos[0][1] == -1),
            true_fun=lambda red: jnp.array([location,red[0][1],red[0][2]]),
            false_fun=lambda red: jax.lax.cond(
                pred=jnp.logical_and(redPos[1][0] == -1, redPos[1][1] == -1),
                true_fun=lambda red: jnp.array([red[0][0],location,red[0][2]]),
                false_fun=lambda red: jax.lax.cond(
                    pred=jnp.logical_and(redPos[2][0] == -1, redPos[2][1] == -1),
                    true_fun=lambda red: jnp.array([red[0][0],red[0][1],location]),
                    false_fun=lambda red: red[0],
                    operand=(redPos,location)
                ),
                operand=(redPos,location)
            ),
            operand=(redPos,location)
        )

        return jnp.array(redPos)

    @partial(jax.jit, static_argnums=(0,))
    def nextRound(self, state: QbertState, pyramid: chex.Array, round: chex.Numeric, level: chex.Numeric, player_position: chex.Array, spawned: chex.Numeric, green_ball_freeze_step: chex.Numeric):
        """ function used to start the next round and adjust all values accordingly if all fields are colored with D
            :param state: the state of the game
            :param pyramid: the new pyramid
            :param round: the new round
            :param level: the new level
            :param player_position: the new player position
            :param spawned: whether the player just spawned
            :param green_ball_freeze_step: until which step to freeze the green ball
            """
        complete=jnp.all(jnp.isin(pyramid, jnp.array([-2, -1, 2], dtype=pyramid.dtype)))
        return jax.lax.cond(
            pred=complete,
            true_fun=lambda vals: (jax.lax.cond(
                pred=jax.random.uniform(vals[3], (), minval=1, maxval=256).astype(jnp.uint8) < 128,
                true_fun=lambda vals2: jnp.array([
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, 0, -1, -1, -1, -1, -1, -1],
                    [-1, 0, 0, -1, -1, -1, -1, -1],
                    [-1, 0, 0, 0, -1, -1, -1, -1],
                    [-2, 0, 0, 0, 0, -2, -1, -1],
                    [-1, 0, 0, 0, 0, 0, -1, -1],
                    [-1, 0, 0, 0, 0, 0, 0, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1]
                ]).astype(jnp.int32),
                false_fun=lambda vals2: jnp.array([
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, 0, -1, -1, -1, -1, -1, -1],
                    [-2, 0, 0, -2, -1, -1, -1, -1],
                    [-1, 0, 0, 0, -1, -1, -1, -1],
                    [-1, 0, 0, 0, 0, -1, -1, -1],
                    [-1, 0, 0, 0, 0, 0, -1, -1],
                    [-1, 0, 0, 0, 0, 0, 0, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1]

                ]),
                operand=None
            ),jnp.array(jnp.mod(vals[1], 5) + 1).astype(jnp.int32) , jax.lax.cond(vals[1] == 4, lambda l: jnp.minimum(5, l + 1), lambda l: l, vals[2]).astype(jnp.int32), jnp.array(3100).astype(jnp.int32), jnp.array([1,1]), jnp.array(1).astype(jnp.int32), vals[5]),
            false_fun=lambda vals: (vals[0],vals[1],vals[2], jnp.array(0).astype(jnp.int32), player_position, vals[4], vals[6]),
            operand=(pyramid,round,level,state.prng_state[6],spawned, state.step_counter, green_ball_freeze_step)
        )

    @partial(jax.jit, static_argnums=(0,))
    def extraLives(self, round: chex.Numeric, level: chex.Numeric, lives: chex.Numeric):
        """ function called at the start of a new round to get an extra live if the round is 2 (except for in level 1)
            :param round: the new round
            :param level: the new level
            :param lives: the amount of lives
            """
        return jax.lax.cond(
            pred=jnp.logical_and(level >= 2, round == 2),
            true_fun=lambda live: live + 1,
            false_fun= lambda live: live,
            operand=lives
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: QbertState, action: chex.Array) -> Tuple[QbertObservation, QbertState, float, bool, QbertInfo]:
        action = jnp.take(self.ACTION_SET, action.astype(jnp.int32))

        # Handle player movement
        tick_counter_reset = jnp.array([31, 227, 144, 124, 110, 95, 81, 66, 52]).astype(jnp.int32)
        is_player_moving = jnp.where(jnp.logical_and(state.dead_animation_counter == 0, jnp.logical_and(state.next_round_animation_counter == 0, jnp.logical_or(jnp.logical_and(state.player_moving_counter == 0, action != Action.NOOP), (state.player_moving_counter + 1) % tick_counter_reset[state.player_position_category] > 1))), 1, 0)
        player_moving_counter = jnp.where(state.is_player_moving == 1, (state.player_moving_counter + 1) % tick_counter_reset[state.player_position_category], state.player_moving_counter)
        player_last_position = jnp.where(jnp.logical_or(state.dead_animation_counter != 0, jnp.logical_or(state.next_round_animation_counter != 0, player_moving_counter != 0)), state.player_last_position, state.player_position)
        player_position = jnp.where(jnp.logical_and(state.dead_animation_counter == 0, jnp.logical_and(state.next_round_animation_counter == 0, jnp.logical_and(state.is_player_moving == 0, action != Action.NOOP))), self.move(state, action, state.player_position), state.player_position)
        player_direction = jnp.select(
            condlist=[
                jnp.logical_and(state.dead_animation_counter == 0, jnp.logical_and(state.next_round_animation_counter == 0, jnp.logical_and(state.is_player_moving == 0, action == Action.RIGHT))),
                jnp.logical_and(state.dead_animation_counter == 0, jnp.logical_and(state.next_round_animation_counter == 0, jnp.logical_and(state.is_player_moving == 0, action == Action.LEFT))),
                jnp.logical_and(state.dead_animation_counter == 0, jnp.logical_and(state.next_round_animation_counter == 0, jnp.logical_and(state.is_player_moving == 0, action == Action.UP))),
                jnp.logical_and(state.dead_animation_counter == 0, jnp.logical_and(state.next_round_animation_counter == 0, jnp.logical_and(state.is_player_moving == 0, action == Action.DOWN)))
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
        enemy_moving_counter = jnp.where(jnp.logical_and(state.dead_animation_counter == 0, jnp.logical_and(state.next_round_animation_counter == 0, state.step_counter == state.green_ball_freeze_step)), (state.enemy_moving_counter + 1) % self._enemy_move_tick[state.level_number], state.enemy_moving_counter)

        # Handle red ball movement
        red_pos = jnp.where(jnp.logical_and(state.dead_animation_counter == 0, jnp.logical_and(state.next_round_animation_counter == 0, enemy_moving_counter == 0)), self.move_red_balls(state), state.red_ball_positions)
        trash1, red0_pos, trash2, trash3 = jax.lax.cond(
            pred=jnp.logical_and(state.dead_animation_counter == 0, jnp.logical_and(state.next_round_animation_counter == 0, enemy_moving_counter == 0)),
            true_fun=lambda s: self.checkField(s, 1, 0, red_pos[0]),
            false_fun=lambda s: (state.pyramid, state.red_ball_positions[0], state.lives, jnp.array(0).astype(jnp.int32)),
            operand=state
        )
        trash1, red1_pos, trash2, trash3 = jax.lax.cond(
            pred=jnp.logical_and(state.dead_animation_counter == 0, jnp.logical_and(state.next_round_animation_counter == 0, enemy_moving_counter == 0)),
            true_fun=lambda s: self.checkField(s, 1, 1, red_pos[1]),
            false_fun=lambda s: (state.pyramid, state.red_ball_positions[1], state.lives, jnp.array(0).astype(jnp.int32)),
            operand=state
        )
        trash1, red2_pos, trash2, trash3 = jax.lax.cond(
            pred=jnp.logical_and(state.dead_animation_counter == 0, jnp.logical_and(state.next_round_animation_counter == 0, enemy_moving_counter == 0)),
            true_fun=lambda s: self.checkField(s, 1, 2, red_pos[2]),
            false_fun=lambda s: (state.pyramid, state.red_ball_positions[2], state.lives, jnp.array(0).astype(jnp.int32)),
            operand=state
        )

        # Handle purple ball movement
        trash1, purple_pos, trash2, trash3 = jax.lax.cond(
            pred=jnp.logical_and(state.dead_animation_counter == 0, jnp.logical_and(state.next_round_animation_counter == 0, enemy_moving_counter == 0)),
            true_fun=lambda s: self.checkField(s, 2, 0, self.move_purple_ball(s)),
            false_fun=lambda s: (state.pyramid, state.purple_ball_position, state.lives, jnp.array(0).astype(jnp.int32)),
            operand=state
        )
        temp = jnp.abs(player_position[0] - player_last_position[0]) + jnp.abs(player_position[1] - player_last_position[1])

        snake_lock=jax.lax.cond(
            pred=jnp.logical_and(jnp.logical_and(jnp.logical_and(temp > 2, jnp.abs(state.snake_position[0] - player_last_position[0]) + jnp.abs(state.snake_position[1] - player_last_position[1]) < 4) , state.snake_lock[0] == -1), state.snake_position[0] != -1),
            true_fun=lambda i: jax.lax.cond(
                pred=i.player_last_position[0] > 1,
                true_fun=lambda j: jnp.array([j.player_last_position[0], j.player_last_position[1] - 1]),
                false_fun=lambda j: jnp.array([j.player_last_position[0] - 1, j.player_last_position[1] - 1]),
                operand=i
            ),
            false_fun=lambda i: i.snake_lock,
            operand=state
        )

        # When player is on a disk (category 1 or 2), Coily should target the disk cell so he jumps off.
        disk_target = jnp.where(
            player_position_category == 1,
            jnp.where(player_last_position[0] < 3, jnp.array([0, 4], dtype=jnp.int32), jnp.array([5, 4], dtype=jnp.int32)),
            jnp.where(
                player_position_category == 2,
                jnp.where(player_last_position[0] < 2, jnp.array([0, 2], dtype=jnp.int32), jnp.array([3, 2], dtype=jnp.int32)),
                player_position,
            ),
        )
        target = jnp.where(
            jnp.logical_or(player_position_category == 1, player_position_category == 2),
            disk_target,
            jax.lax.cond(
                pred=snake_lock[0] == -1,
                true_fun=lambda i: i[0],
                false_fun=lambda i: i[1],
                operand=(player_position, snake_lock),
            ),
        )

        # Handle snake movement
        trash1, snake_pos, trash2, points4 = jax.lax.cond(
            pred=jnp.logical_and(jnp.logical_and(state.dead_animation_counter == 0, jnp.logical_and(state.next_round_animation_counter == 0, enemy_moving_counter == 0)), state.snake_position[0] != -1),
            true_fun=lambda s: self.checkField(s[0], 3, 0, self.move_snake(s[0], s[1])),
            false_fun=lambda s: ((s[0].pyramid, s[0].snake_position, s[0].lives, jnp.array(0).astype(jnp.int32))),
            operand=(state, target),
        )


        # Handle green ball movement
        trash1, green_pos, trash2, trash3 = jax.lax.cond(
            pred=jnp.logical_and(state.dead_animation_counter == 0, jnp.logical_and(state.next_round_animation_counter == 0, enemy_moving_counter == 0)),
            true_fun=lambda s: self.checkField(s, 4, 0, self.move_green_ball(s)),
            false_fun=lambda s: (state.pyramid, state.green_ball_position, state.lives, jnp.array(0).astype(jnp.int32)),
            operand=state
        )

        # Handle sam movement
        trash1, sam_pos, trash2, trash3 = jax.lax.cond(
            pred=jnp.logical_and(state.dead_animation_counter == 0, jnp.logical_and(state.next_round_animation_counter == 0, enemy_moving_counter == 0)),
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

        tmp_lives = lives

        # Checks for collisions of the player with the enemies
        lives, red_pos, purple_pos, green_pos, snake_pos, sam_pos, points1, green_ball_freeze_step, same_cell_frames = self.checkCollisions(
            lives,
            jnp.where(jnp.logical_and(player_position_category != 1, player_position_category != 2), player_last_position, player_position),
            jnp.array([red0_pos, red1_pos, red2_pos]), purple_pos, green_pos, snake_pos, sam_pos,
            state.green_ball_freeze_step,
            state.same_cell_frames
        )

        # Increase dead_animation_counter if player lost a live
        dead_animation_counter = jnp.where(jnp.logical_or(state.dead_animation_counter != 0, tmp_lives != lives), (state.dead_animation_counter + 1) % 128, state.dead_animation_counter)

        # Changes the colors of the pyramid depending on the player position and the position of sam
        pyramid, points2 = self.changeColors(state, player_position, sam_pos, pyramid, state.level_number, spawned)

        # Spawn new enemies: only after FIRST_SPAWN_DELAY and when SPAWN_INTERVAL has elapsed (or first spawn).
        spawn_allowed = (
            jnp.logical_and(state.dead_animation_counter == 0,
            jnp.logical_and(state.next_round_animation_counter == 0,
            jnp.logical_and(enemy_moving_counter == 0,
            jnp.logical_and(state.step_counter >= self.consts.FIRST_SPAWN_DELAY,
            jnp.logical_or(state.step_counter - state.last_spawn_step >= self.consts.SPAWN_INTERVAL, state.last_spawn_step == 0)))))
        )
        before_red = jnp.array(red_pos)
        before_purple = jnp.array(purple_pos)
        before_green = jnp.array(green_pos)
        before_sam = jnp.array(sam_pos)
        red_pos, purple_pos, green_pos, sam_pos = jax.lax.cond(
            pred=spawn_allowed,
            true_fun=lambda op: self.spawnCreatures(op[0], op[1], op[2], op[3], op[4], op[5], op[6], op[7], op[8], op[9]),
            false_fun=lambda op: (op[1], op[2], op[3], op[4]),
            operand=(
                jnp.array([state.prng_state[7], state.prng_state[8]]),
                jnp.array(red_pos),
                jnp.array(purple_pos),
                jnp.array(green_pos),
                jnp.array(sam_pos),
                jnp.array(snake_pos),
                self.consts.DIFFICULTY,
                state.step_counter,
                state.enemy_spawn_count,
                state.level_number,
            ),
        )
        red_pos_arr = jnp.array(red_pos)
        spawned_this_tick = (jnp.any(red_pos_arr != before_red) | jnp.any(jnp.array(purple_pos) != before_purple) | jnp.any(jnp.array(green_pos) != before_green) | jnp.any(jnp.array(sam_pos) != before_sam))
        last_spawn_step = jnp.where(spawned_this_tick, state.step_counter, state.last_spawn_step)

        # Calculates values at the end of the round
        pyramid, round, level, points3, player_position, spawned, green_ball_freeze_step = jax.lax.cond(
            pred=jnp.logical_and(is_player_moving == 0, player_moving_counter == 0),
            true_fun=lambda op: self.nextRound(op[0], op[1], op[2], op[3], op[4], op[5], op[6]),
            false_fun=lambda op: (op[1], op[2], op[3], 0, op[4], op[5], op[6]),
            operand=(state, pyramid, state.round_number, state.level_number, player_position, spawned, green_ball_freeze_step)
        )

        # Increases next_round_animation_counter if end of round is reached
        next_round_animation_counter = jnp.where(jnp.logical_or(state.next_round_animation_counter != 0, round != state.round_number), (state.next_round_animation_counter + 1) % 160, state.next_round_animation_counter)

        # Gains an extra live at the end of a round
        lives=jnp.where(jnp.logical_and(jnp.logical_and(is_player_moving == 0, player_moving_counter == 0), round != state.round_number), self.extraLives(round,level,lives), lives)

        # Updates last pyramid
        last_pyramid = jnp.where(jnp.logical_or(state.next_round_animation_counter != 0, jnp.logical_or(jnp.logical_and(jnp.logical_or(player_position_category == 1, player_position_category == 2), player_moving_counter == 45), jnp.logical_and(jnp.logical_and(player_position_category != 1, player_position_category != 2), player_moving_counter == 27))), state.pyramid, state.last_pyramid)

        red_pos, purple_pos, green_pos, snake_pos, sam_pos = jax.lax.cond(
            pred=jnp.logical_or(round != state.round_number, lives < state.lives),
            true_fun=lambda c: (jnp.array([[-1, -1], [-1, -1], [-1, -1]]).astype(jnp.int32), jnp.array([-1, -1]), jnp.array([-1, -1]), jnp.array([-1, -1]), jnp.array([-1, -1])),
            false_fun=lambda c: (c[0], c[1], c[2], c[3], c[4]),
            operand=(red_pos, purple_pos, green_pos, snake_pos, sam_pos)
        )
        same_cell_frames = jnp.where(jnp.logical_or(round != state.round_number, lives < state.lives), 0, same_cell_frames)
        last_spawn_step = jnp.where(round != state.round_number, 0, last_spawn_step)

        # Track how many enemies have spawned in the current level:
        # increment when something spawned this tick, reset when level changes.
        enemy_spawn_count = jnp.where(spawned_this_tick, state.enemy_spawn_count + 1, state.enemy_spawn_count)
        enemy_spawn_count = jnp.where(level != state.level_number, jnp.array(0, dtype=jnp.int32), enemy_spawn_count)

        new_state = state.replace(
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
            prng_state=jax.random.split(state.prng_state[9], 10),
            just_spawned=spawned,
            snake_lock=jnp.array([snake_lock[0], snake_lock[1]]),
            same_cell_frames=same_cell_frames,
            last_spawn_step=last_spawn_step,
            enemy_spawn_count=enemy_spawn_count,
            next_round_animation_counter=next_round_animation_counter,
            dead_animation_counter=dead_animation_counter,
        )

        done = self._get_done(new_state)
        env_reward = self._get_reward(state, new_state)
        info = self._get_info(new_state)
        observation = self._get_observation(new_state)

        return observation, new_state, env_reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: QbertState, state: QbertState):
        return state.player_score - previous_state.player_score

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: QbertState) -> jnp.ndarray:
        return self.renderer.render(state)

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTION_SET))

    def image_space(self) -> spaces.Box:
        return spaces.Box(0, 255, shape=(self.consts.HEIGHT, self.consts.WIDTH, 3), dtype=jnp.uint8)

    def observation_space(self) -> spaces.Space:
        screen_size = (self.consts.HEIGHT, self.consts.WIDTH)
        single_obj = spaces.get_object_space(n=None, screen_size=screen_size)

        return spaces.Dict({
            "player": single_obj,
            "red_balls": spaces.get_object_space(n=3, screen_size=screen_size),
            "purple_ball": single_obj,
            "snake": single_obj,
            "green_ball": single_obj,
            "sam": single_obj,
            "player_score": spaces.Box(0, 99999, (), jnp.int32),
            "lives": spaces.Box(-1, 9, (), jnp.int32),
            "pyramid": spaces.Box(-2, 3, (8, 8), jnp.int32),
            "level_number": spaces.Box(1, 5, (), jnp.int32),
            "round_number": spaces.Box(1, 6, (), jnp.int32),
        })

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: QbertState):
        player = ObjectObservation.create(
            x=state.player_position[0],
            y=state.player_position[1],
            width=jnp.array(1, dtype=jnp.int32),
            height=jnp.array(1, dtype=jnp.int32),
        )
        red_balls = ObjectObservation.create(
            x=state.red_ball_positions[:, 0],
            y=state.red_ball_positions[:, 1],
            width=jnp.ones(3, dtype=jnp.int32),
            height=jnp.ones(3, dtype=jnp.int32),
            active=(state.red_ball_positions[:, 0] != -1).astype(jnp.int32),
        )
        purple_ball = ObjectObservation.create(
            x=state.purple_ball_position[0],
            y=state.purple_ball_position[1],
            width=jnp.array(1, dtype=jnp.int32),
            height=jnp.array(1, dtype=jnp.int32),
            active=(state.purple_ball_position[0] != -1).astype(jnp.int32),
        )
        snake = ObjectObservation.create(
            x=state.snake_position[0],
            y=state.snake_position[1],
            width=jnp.array(1, dtype=jnp.int32),
            height=jnp.array(1, dtype=jnp.int32),
            active=(state.snake_position[0] != -1).astype(jnp.int32),
        )
        green_ball = ObjectObservation.create(
            x=state.green_ball_position[0],
            y=state.green_ball_position[1],
            width=jnp.array(1, dtype=jnp.int32),
            height=jnp.array(1, dtype=jnp.int32),
            active=(state.green_ball_position[0] != -1).astype(jnp.int32),
        )
        sam = ObjectObservation.create(
            x=state.sam_position[0],
            y=state.sam_position[1],
            width=jnp.array(1, dtype=jnp.int32),
            height=jnp.array(1, dtype=jnp.int32),
            active=(state.sam_position[0] != -1).astype(jnp.int32),
        )
        return QbertObservation(
            player=player,
            red_balls=red_balls,
            purple_ball=purple_ball,
            snake=snake,
            green_ball=green_ball,
            sam=sam,
            player_score=state.player_score,
            lives=state.lives,
            pyramid=state.pyramid,
            level_number=state.level_number,
            round_number=state.round_number,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: QbertState) -> QbertInfo:
        return QbertInfo(state.step_counter)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: QbertState) -> bool:
        player_has_no_lives = state.lives == -1
        return player_has_no_lives

class QbertRenderer(JAXGameRenderer):
    def __init__(self, consts: QbertConstants = None, config: render_utils.RendererConfig = None):
        super().__init__(consts)
        self.consts = consts or QbertConstants()

        if config is None:
            self.config = render_utils.RendererConfig(
                game_dimensions=(self.consts.HEIGHT, self.consts.WIDTH),
                channels=3,
            )
        else:
            self.config = config

        self.jr = render_utils.JaxRenderingUtils(self.config)

        final_asset_config = list(self.consts.ASSET_CONFIG)
        sprite_path = os.path.join(render_utils.get_base_sprite_dir(), "qbert")

        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS,
        ) = self.jr.load_and_setup_assets(final_asset_config, sprite_path)

        freeze_rgba = self.jr.loadFrame(os.path.join(sprite_path, "freeze.npy"))
        self.FREEZE_BG = self.jr._create_background_raster(freeze_rgba, self.COLOR_TO_ID)

        self._enemy_move_tick = jnp.array(self.consts.ENEMY_MOVE_TICK, dtype=jnp.int32)

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

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: QbertState) -> jnp.ndarray:
        jr = self.jr
        M = self.SHAPE_MASKS

        raster = jnp.where(state.step_counter == state.green_ball_freeze_step, self.BACKGROUND, self.FREEZE_BG)

        round_idx = jnp.where(state.next_round_animation_counter != 0, state.round_number - 2, state.round_number - 1)
        raster = jr.render_at(raster, 68, 40, M['cube_shadow_right'][round_idx])

        raster = jr.render_at(raster, 56, 69, M['cube_shadow_right'][round_idx])
        raster = jr.render_at(raster, 80, 69, M['cube_shadow_left'][round_idx])

        raster = jr.render_at(raster, 44, 98, M['cube_shadow_right'][round_idx])
        raster = jr.render_at(raster, 68, 98, M['cube_shadow_right'][round_idx])
        raster = jr.render_at(raster, 92, 98, M['cube_shadow_left'][round_idx])

        raster = jr.render_at(raster, 32, 127, M['cube_shadow_right'][round_idx])
        raster = jr.render_at(raster, 56, 127, M['cube_shadow_right'][round_idx])
        raster = jr.render_at(raster, 80, 127, M['cube_shadow_left'][round_idx])
        raster = jr.render_at(raster, 104, 127, M['cube_shadow_left'][round_idx])

        raster = jr.render_at(raster, 20, 156, M['cube_shadow_right'][round_idx])
        raster = jr.render_at(raster, 44, 156, M['cube_shadow_right'][round_idx])
        raster = jr.render_at(raster, 68, 156, M['cube_shadow_right'][round_idx])
        raster = jr.render_at(raster, 92, 156, M['cube_shadow_left'][round_idx])
        raster = jr.render_at(raster, 116, 156, M['cube_shadow_left'][round_idx])

        raster = jr.render_at(raster, 8, 185, M['cube_shadow_right'][round_idx])
        raster = jr.render_at(raster, 32, 185, M['cube_shadow_right'][round_idx])
        raster = jr.render_at(raster, 56, 185, M['cube_shadow_right'][round_idx])
        raster = jr.render_at(raster, 80, 185, M['cube_shadow_left'][round_idx])
        raster = jr.render_at(raster, 104, 185, M['cube_shadow_left'][round_idx])
        raster = jr.render_at(raster, 128, 185, M['cube_shadow_left'][round_idx])

        raster = jnp.where(state.last_pyramid[2][0] == -2, jr.render_at(raster, 38, 84, M['disc']), raster)
        raster = jnp.where(state.last_pyramid[2][3] == -2, jr.render_at(raster, 110, 84, M['disc']), raster)
        raster = jnp.where(state.last_pyramid[4][0] == -2, jr.render_at(raster, 14, 142, M['disc']), raster)
        raster = jnp.where(state.last_pyramid[4][5] == -2, jr.render_at(raster, 134, 142, M['disc']), raster)

        pyra = state.pyramid.at[state.player_position[1], state.player_position[0]].set(state.last_pyramid[state.player_position[1]][state.player_position[0]])
        raster = jax.lax.fori_loop(
            lower=1,
            upper=7,
            body_fun=lambda i, val: jax.lax.fori_loop(
                lower=1,
                upper=i + 1,
                body_fun=lambda j, val2: jnp.select(
                    condlist=[
                        state.next_round_animation_counter != 0,
                        pyra[i, j] == 0,
                        pyra[i, j] == 1,
                        pyra[i, j] == 2,
                    ],
                    choicelist=[
                        jr.render_at(val2, self.COLOR_POSITIONS[jnp.array(i * (i - 1) / 2 + (j - 1)).astype(jnp.int32)][0], self.COLOR_POSITIONS[jnp.array(i * (i - 1) / 2 + (j - 1)).astype(jnp.int32)][1] + 2, M['win_animation'][jnp.floor(state.next_round_animation_counter / 32).astype(jnp.int32)]),
                        jr.render_at(val2, self.COLOR_POSITIONS[jnp.array(i * (i - 1) / 2 + (j - 1)).astype(jnp.int32)][0], self.COLOR_POSITIONS[jnp.array(i * (i - 1) / 2 + (j - 1)).astype(jnp.int32)][1] + 2, M['color_start']),
                        jr.render_at(val2, self.COLOR_POSITIONS[jnp.array(i * (i - 1) / 2 + (j - 1)).astype(jnp.int32)][0], self.COLOR_POSITIONS[jnp.array(i * (i - 1) / 2 + (j - 1)).astype(jnp.int32)][1] + 2, M['color_intermediate']),
                        jr.render_at(val2, self.COLOR_POSITIONS[jnp.array(i * (i - 1) / 2 + (j - 1)).astype(jnp.int32)][0], self.COLOR_POSITIONS[jnp.array(i * (i - 1) / 2 + (j - 1)).astype(jnp.int32)][1] + 2, M['color_destination']),
                    ],
                    default=val2,
                ),
                init_val=val,
            ),
            init_val=raster,
        )

        player_score_digits = jr.int_to_digits(state.player_score, max_digits=5)
        raster = jnp.where(state.player_position[1] >= 3, jr.render_label_selective(raster, 34, 6, player_score_digits, M['score_digits'], 0, 5, spacing=8, max_digits_to_render=5), raster)

        raster = jax.lax.fori_loop(
            lower=0,
            upper=state.lives,
            body_fun=lambda i, val: jnp.where(state.player_position[1] >= 3, jr.render_at(val, self.LIVE_POSITIONS[i][0], self.LIVE_POSITIONS[i][1], M['qbert_live']), val),
            init_val=raster,
        )

        qbert_j = state.player_last_position[0]
        qbert_i = state.player_last_position[1]
        move_positions = jnp.array([self.QBERT_MOVE_LEFT_UP, self.QBERT_MOVE_LEFT_DOWN, self.QBERT_MOVE_RIGHT_DOWN, self.QBERT_MOVE_RIGHT_UP]).astype(jnp.int32)
        qbert_sprites = M['qbert_sprites']
        raster = jax.lax.switch(
            index=state.player_position_category,
            branches=[
                lambda state: jr.render_at(raster,
                                           self.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][0] + move_positions[state.player_direction][state.player_moving_counter][0],
                                           self.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][1] + move_positions[state.player_direction][state.player_moving_counter][1],
                                           qbert_sprites[state.player_direction]),
                lambda state: jr.render_at(raster,
                                           self.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][0] + self.QBERT_MOVE_DISC_LEFT_BOTTOM[state.player_moving_counter][0] * (state.player_direction == 0) + self.QBERT_MOVE_DISC_RIGHT_BOTTOM[state.player_moving_counter][0] * (state.player_direction == 3),
                                           self.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][1] + self.QBERT_MOVE_DISC_LEFT_BOTTOM[state.player_moving_counter][1] * (state.player_direction == 0) + self.QBERT_MOVE_DISC_RIGHT_BOTTOM[state.player_moving_counter][1] * (state.player_direction == 3),
                                           qbert_sprites[state.player_direction]),
                lambda state: jr.render_at(raster,
                                           self.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][0] + self.QBERT_MOVE_DISC_LEFT_TOP[state.player_moving_counter][0] * (state.player_direction == 0) + self.QBERT_MOVE_DISC_RIGHT_TOP[state.player_moving_counter][0] * (state.player_direction == 3),
                                           self.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][1] + self.QBERT_MOVE_DISC_LEFT_TOP[state.player_moving_counter][1] * (state.player_direction == 0) + self.QBERT_MOVE_DISC_RIGHT_TOP[state.player_moving_counter][1] * (state.player_direction == 3),
                                           qbert_sprites[state.player_direction]),
                lambda state: jr.render_at(raster,
                                           self.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][0] + self.QBERT_MOVE_OUT_PYRAMID_1[state.player_moving_counter][0] * (- 1 * (state.player_direction > 1)) + self.QBERT_MOVE_OUT_PYRAMID_6[state.player_moving_counter][0] * (state.player_direction < 1),
                                           self.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][1] + self.QBERT_MOVE_OUT_PYRAMID_1[state.player_moving_counter][1],
                                           qbert_sprites[state.player_direction]),
                lambda state: jr.render_at(raster,
                                           self.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][0] + self.QBERT_MOVE_OUT_PYRAMID_2[state.player_moving_counter][0] * (- 1 * (state.player_direction > 1)) + self.QBERT_MOVE_OUT_PYRAMID_6[state.player_moving_counter][0] * (state.player_direction < 1),
                                           self.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][1] + self.QBERT_MOVE_OUT_PYRAMID_2[state.player_moving_counter][1],
                                           qbert_sprites[state.player_direction]),
                lambda state: jr.render_at(raster,
                                           self.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][0] + self.QBERT_MOVE_OUT_PYRAMID_3[state.player_moving_counter][0] * (- 1 * (state.player_direction > 1)) + self.QBERT_MOVE_OUT_PYRAMID_6[state.player_moving_counter][0] * (state.player_direction < 1),
                                           self.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][1] + self.QBERT_MOVE_OUT_PYRAMID_3[state.player_moving_counter][1],
                                           qbert_sprites[state.player_direction]),
                lambda state: jr.render_at(raster,
                                           self.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][0] + self.QBERT_MOVE_OUT_PYRAMID_4[state.player_moving_counter][0] * (- 1 * (state.player_direction > 1)) + self.QBERT_MOVE_OUT_PYRAMID_6[state.player_moving_counter][0] * (state.player_direction < 1),
                                           self.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][1] + self.QBERT_MOVE_OUT_PYRAMID_4[state.player_moving_counter][1],
                                           qbert_sprites[state.player_direction]),
                lambda state: jr.render_at(raster,
                                           self.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][0] + self.QBERT_MOVE_OUT_PYRAMID_5[state.player_moving_counter][0] * (- 1 * (state.player_direction > 1)) + self.QBERT_MOVE_OUT_PYRAMID_6[state.player_moving_counter][0] * (state.player_direction < 1),
                                           self.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][1] + self.QBERT_MOVE_OUT_PYRAMID_5[state.player_moving_counter][1],
                                           qbert_sprites[state.player_direction]),
                lambda state: jr.render_at(raster,
                                           self.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][0] + self.QBERT_MOVE_OUT_PYRAMID_6[state.player_moving_counter][0] * (- 1 * (state.player_direction > 1)) + self.QBERT_MOVE_OUT_PYRAMID_6[state.player_moving_counter][0] * (state.player_direction < 1),
                                           self.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][1] + self.QBERT_MOVE_OUT_PYRAMID_6[state.player_moving_counter][1],
                                           qbert_sprites[state.player_direction]),
            ],
            operand=state,
        )

        sam_j = state.sam_position[0]
        sam_i = state.sam_position[1]
        raster = jnp.where(jnp.logical_and(state.sam_position[0] == -1, state.sam_position[1] == -1), raster, jr.render_at(raster,
                                                                                                                           self.QBERT_POSITIONS[jnp.array(sam_i * (sam_i - 1) / 2 + (sam_j - 1)).astype(jnp.int32)][0],
                                                                                                                           self.QBERT_POSITIONS[jnp.array(sam_i * (sam_i - 1) / 2 + (sam_j - 1)).astype(jnp.int32)][1] + 1,
                                                                                                                           M['sam']))

        green_ball_j = state.green_ball_position[0]
        green_ball_i = state.green_ball_position[1]
        green_ball_index = jnp.floor(state.enemy_moving_counter / jnp.ceil(self._enemy_move_tick[state.level_number] / 2).astype(jnp.int32)).astype(jnp.int32)
        raster = jnp.where(jnp.logical_and(state.green_ball_position[0] == -1, state.green_ball_position[1] == -1), raster, jr.render_at(raster,
                                                                                                                                         self.QBERT_POSITIONS[jnp.array(green_ball_i * (green_ball_i - 1) / 2 + (green_ball_j - 1)).astype(jnp.int32)][0] + self.BALL_MOVE[green_ball_index][0],
                                                                                                                                         self.QBERT_POSITIONS[jnp.array(green_ball_i * (green_ball_i - 1) / 2 + (green_ball_j - 1)).astype(jnp.int32)][1] + self.BALL_MOVE[green_ball_index][1],
                                                                                                                                         M['green_ball'][green_ball_index]))

        purple_ball_j = state.purple_ball_position[0]
        purple_ball_i = state.purple_ball_position[1]
        purple_ball_index = jnp.floor(state.enemy_moving_counter / jnp.ceil(self._enemy_move_tick[state.level_number] / 2).astype(jnp.int32)).astype(jnp.int32)
        raster = jnp.where(jnp.logical_and(state.purple_ball_position[0] == -1, state.purple_ball_position[1] == -1), raster, jr.render_at(raster,
                                                                                                                                           self.QBERT_POSITIONS[jnp.array(purple_ball_i * (purple_ball_i - 1) / 2 + (purple_ball_j - 1)).astype(jnp.int32)][0] + self.BALL_MOVE[purple_ball_index][0],
                                                                                                                                           self.QBERT_POSITIONS[jnp.array(purple_ball_i * (purple_ball_i - 1) / 2 + (purple_ball_j - 1)).astype(jnp.int32)][1] + self.BALL_MOVE[purple_ball_index][1],
                                                                                                                                           M['purple_ball'][purple_ball_index]))

        snake_j = state.snake_position[0]
        snake_i = state.snake_position[1]
        snake_index = jnp.floor(state.enemy_moving_counter / jnp.ceil(self._enemy_move_tick[state.level_number] / 5).astype(jnp.int32)).astype(jnp.int32)
        raster = jnp.where(jnp.logical_and(state.snake_position[0] == -1, state.snake_position[1] == -1), raster, jr.render_at(raster,
                                                                                                                               self.QBERT_POSITIONS[jnp.array(snake_i * (snake_i - 1) / 2 + (snake_j - 1)).astype(jnp.int32)][0] + self.SNAKE_MOVE[snake_index][0],
                                                                                                                               self.QBERT_POSITIONS[jnp.array(snake_i * (snake_i - 1) / 2 + (snake_j - 1)).astype(jnp.int32)][1] + self.SNAKE_MOVE[snake_index][1],
                                                                                                                               M['snake'][snake_index]))

        def render_red_ball(i, r):
            red_ball_j = state.red_ball_positions[i][0]
            red_ball_i = state.red_ball_positions[i][1]
            red_ball_index = jnp.floor(state.enemy_moving_counter / jnp.ceil(self._enemy_move_tick[state.level_number] / 2).astype(jnp.int32)).astype(jnp.int32)
            r = jnp.where(jnp.logical_and(state.red_ball_positions[i][0] == -1, state.red_ball_positions[i][1] == -1), r, jr.render_at(r,
                                                                                                                                            self.QBERT_POSITIONS[jnp.array(red_ball_i * (red_ball_i - 1) / 2 + (red_ball_j - 1)).astype(jnp.int32)][0] + self.BALL_MOVE[red_ball_index][0],
                                                                                                                                            self.QBERT_POSITIONS[jnp.array(red_ball_i * (red_ball_i - 1) / 2 + (red_ball_j - 1)).astype(jnp.int32)][1] + self.BALL_MOVE[red_ball_index][1],
                                                                                                                                            M['red_ball'][red_ball_index]))
            return r

        raster = jax.lax.fori_loop(
            lower=0,
            upper=3,
            body_fun=render_red_ball,
            init_val=raster,
        )

        raster = jnp.where(state.dead_animation_counter != 0, jr.render_at(raster,
                                                                           self.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][0] + 15,
                                                                           self.QBERT_POSITIONS[jnp.array(qbert_i * (qbert_i - 1) / 2 + (qbert_j - 1)).astype(jnp.int32)][1] - 9,
                                                                           M['dead']), raster)

        return jr.render_from_palette(raster, self.PALETTE)
