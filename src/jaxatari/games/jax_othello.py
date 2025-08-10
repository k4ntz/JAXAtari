import os
from functools import partial
from typing import NamedTuple, Tuple
import jax.lax
import jax.numpy as jnp
import chex
import pygame
import enum
import time
from gymnax.environments import spaces

from jaxatari.renderers import AtraJaxisRenderer
from jaxatari.rendering import atraJaxis as aj
from jaxatari.environment import JaxEnvironment
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action

import json

# Game Environment
WIDTH = 160
HEIGHT = 210
FIELD_WIDTH = 8
FIELD_HEIGHT = 8

# Pygame window dimensions
WINDOW_WIDTH = 160 * 3
WINDOW_HEIGHT = 210 * 3

# Actions constants
NOOP = Action.NOOP
UP = Action.UP
DOWN = Action.DOWN
LEFT = Action.LEFT
RIGHT = Action.RIGHT
UPRIGHT = Action.UPRIGHT
UPLEFT = Action.UPLEFT
DOWNRIGHT = Action.DOWNRIGHT
DOWNLEFT = Action.DOWNLEFT
PLACE = Action.FIRE

# attribution of cases for the strategic tile score //starting at bottom right, then going left und and then up
STRATEGIC_TILE_SCORE_CASES =  jnp.array([
 17, 16, 16, 16, 16, 16, 16, 15,
 11, 14, 13, 13, 13, 13, 12, 6,
 11, 10, 8, 9, 9, 8, 7, 6,
 11, 10, 9, 9, 9, 9, 7, 6,
 11, 10, 9, 9, 9, 9, 7, 6,
 11, 10, 8, 9, 9, 8, 7, 6,
 11, 5, 4, 4, 4, 4, 3, 6,
 2, 1, 1, 1, 1, 1, 1, 0
])


__F3BA = jnp.array([
    0x60, 0x40, 0x42, 0x40, 0x00, 0x00, 0x00, 0x46, 0x46, 0x44, 0x04, 0x08, 0x0c, 0x0a, 0x08,
    0x04, 0x10, 0x14, 0xbe, 0x9e, 0x02, 0x02, 0x02, 0x12, 0x48, 0x28, 0x10, 0x08, 0x18, 0x38,
    0x40, 0x00, 0x02, 0x02
], dtype=jnp.uint8)

__F3DC = jnp.array([
    0x14, 0x28, 0x28, 0x2c, 0x46, 0x44, 0x40, 0x20, 0x08, 0x20, 0x60, 0x40, 0x40, 0x40, 0x42,
    0x40, 0x40, 0x40, 0x40, 0x60, 0x20, 0x28, 0x2c, 0x24, 0x32, 0x12, 0x4c, 0xf2, 0xe2, 0xc2,
    0x02, 0xbe, 0x18, 0x48
], dtype=jnp.uint8)

__F3FE = jnp.array([
    0x30, 0x30, 0x30, 0x30, 0xc0, 0xc0, 0xc0, 0x30, 0x30, 0x30, 0xbb, 0xbb, 0xbb, 0xbb, 0xbb,
    0xbb, 0xbb, 0xbb, 0x60, 0x60, 0x40, 0x30, 0x30, 0x50, 0xe0, 0xbb, 0xbb, 0xd0, 0xd0, 0xd0,
    0xd0, 0xd8, 0xf0, 0xf0
], dtype=jnp.uint8)

BIT_MASKS_LOW_TO_HIGH = jnp.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=jnp.uint8)

BIT_MASKS_HIGH_TO_LOW = jnp.array([0x80, 0x40, 0x20, 0x10, 8, 4, 2, 1], dtype=jnp.uint8)

__F7EC = jnp.array([
    0x20, 0x20, 0x20, 0x20, 0x20, 0x10, 0x40, 0xe0, 0x20, 0x40, 0x15, 0xe0, 0x20, 0xe0, 0xe0,
    0x50, 0x00, 0xf0, 0xb0
], dtype=jnp.uint8) #TODO difference to np.int8

# Constants to decide in which side the discs will be flipped
FLIP_UP_SIDE = 0
FLIP_DOWN_SIDE = 1
FLIP_RIGHT_SIDE = 2
FLIP_LEFT_SIDE = 3
FLIP_UP_RIGHT_SIDE = 4
FLIP_DOWN_RIGHT_SIDE = 5
FLIP_DOWN_LEFT_SIDE = 6
FLIP_UP_LEFT_SIDE = 7


def get_human_action() -> chex.Array:
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w] and keys[pygame.K_a]:
        return jnp.array(UPLEFT)
    elif keys[pygame.K_w] and keys[pygame.K_d]:
        return jnp.array(UPRIGHT)
    elif keys[pygame.K_s] and keys[pygame.K_a]:
        return jnp.array(DOWNLEFT)
    elif keys[pygame.K_s] and keys[pygame.K_d]:
        return jnp.array(DOWNRIGHT)
    elif keys[pygame.K_w]:
        return jnp.array(UP)
    elif keys[pygame.K_s]:
        return jnp.array(DOWN)
    elif keys[pygame.K_a]:
        return jnp.array(LEFT)
    elif keys[pygame.K_d]:
        return jnp.array(RIGHT)
    elif keys[pygame.K_LCTRL]:
        return jnp.array(PLACE)
    else:
        return jnp.array(NOOP)



# state container
class FieldColor(enum.IntEnum):
    EMPTY = 0
    WHITE = 1
    BLACK = 2

class Field(NamedTuple):
    field_id: chex.Array
    field_color: chex.Array

class OthelloState(NamedTuple):
    player_score: chex.Array
    enemy_score: chex.Array
    step_counter: chex.Array
    field: Field
    field_choice_player: chex.Array
    difficulty: chex.Array
    end_of_game_reached: chex.Array
    random_key: int

class OthelloObservation(NamedTuple):
    field: Field
    player_score: jnp.ndarray
    enemy_score: jnp.ndarray
    field_choice_player: jnp.ndarray

class OthelloInfo(NamedTuple):
    time: jnp.ndarray
    all_rewards: chex.Array


@jax.jit
def has_player_decided_field(field_choice_player, action: chex.Array):
    # field_choice_player: the current positioning of the disc -> it is not jet placed down
    # action may include the action FIRE to place the disc down
    
    is_place = jnp.equal(action, PLACE)
    is_up = jnp.equal(action, UP)
    is_right = jnp.equal(action, RIGHT)
    is_down = jnp.equal(action, DOWN)
    is_left = jnp.equal(action, LEFT)
    is_upleft = jnp.equal(action, UPLEFT)
    is_upright = jnp.equal(action, UPRIGHT)
    is_downleft = jnp.equal(action, DOWNLEFT)
    is_downright = jnp.equal(action, DOWNRIGHT)
    
    def place_disc(field_choice_player):
        return True, field_choice_player

    def move_disc_up(field_choice_player):
        cond = field_choice_player[0] > 0

        new_value = jax.lax.cond(
            cond, 
            lambda _: field_choice_player[0] - 1,
            lambda _: jnp.array(7).astype(jnp.int32),
            operand=None
        )
        field_choice_player = field_choice_player.at[0].set(new_value)
        return False, field_choice_player

    def move_disc_right(field_choice_player):
        cond = field_choice_player[1] < 7

        new_value = jax.lax.cond(
            cond,
            lambda _: field_choice_player[1] + 1,
            lambda _: jnp.array(0).astype(jnp.int32),
            operand=None
        )
        field_choice_player = field_choice_player.at[1].set(new_value)
        return False, field_choice_player
    
    def move_disc_down(field_choice_player):
        cond = field_choice_player[0] < 7

        new_value = jax.lax.cond(
            cond,
            lambda _: field_choice_player[0] + 1,
            lambda _: jnp.array(0).astype(jnp.int32),
            operand=None
        )
        field_choice_player = field_choice_player.at[0].set(new_value)
        return False, field_choice_player

    def move_disc_left(field_choice_player):
        cond = field_choice_player[1] > 0

        new_value = jax.lax.cond(
            cond,
            lambda _: field_choice_player[1] - 1,
            lambda _: jnp.array(7).astype(jnp.int32),
            operand=None
        )
        field_choice_player = field_choice_player.at[1].set(new_value)
        return False, field_choice_player
    
    def move_disc_upleft(field_choice_player):
        _, field_choice_player = move_disc_left(field_choice_player)
        return move_disc_up(field_choice_player)

    def move_disc_upright(field_choice_player):
        _, field_choice_player = move_disc_right(field_choice_player)
        return move_disc_up(field_choice_player)   

    def move_disc_downleft(field_choice_player):
        _, field_choice_player = move_disc_left(field_choice_player)
        return move_disc_down(field_choice_player)
    
    def move_disc_downright(field_choice_player):
        _, field_choice_player = move_disc_right(field_choice_player)
        return move_disc_down(field_choice_player)

    return jax.lax.cond(
        is_place,
        lambda x: place_disc(x),
        lambda x: jax.lax.cond(
            is_up,
            lambda x: move_disc_up(x),
            lambda x: jax.lax.cond(
                is_right,
                lambda x: move_disc_right(x),
                lambda x: jax.lax.cond(
                    is_down,
                    lambda x: move_disc_down(x),
                    lambda x: jax.lax.cond(
                        is_left,
                        lambda x: move_disc_left(x),
                        lambda x: jax.lax.cond(
                            is_upleft,
                            lambda x: move_disc_upleft(x),
                            lambda x: jax.lax.cond(
                                is_upright,
                                lambda x: move_disc_upright(x),
                                lambda x: jax.lax.cond(
                                    is_downleft,
                                    lambda x: move_disc_downleft(x),
                                    lambda x: jax.lax.cond(
                                        is_downright,
                                        lambda x: move_disc_downright(x),
                                        lambda x: (False, x),
                                        x
                                    ),
                                    x
                                ),
                                x
                            ),
                            x
                        ),
                        x
                    ),
                    x
                ),
                x
            ),
            x
        ),
        field_choice_player
    )


@jax.jit
def field_step(field_choice, curr_state, white_player):  # -> valid_choice, new_state
    x, y = field_choice
    enemy_color = jax.lax.cond(
        white_player,
        lambda _: FieldColor.BLACK,
        lambda _: FieldColor.WHITE,
        operand=None
    )
    friendly_color = jax.lax.cond(
        white_player,
        lambda _: FieldColor.WHITE,
        lambda _: FieldColor.BLACK,
        operand=None
    )

    def if_empty(curr_state):
        # new state after player's move
        # it needs to check, if the disc is a valid choice by the rules of othello
        valid_choice = False

        def loop_horizontal_and_vertical_line_to_flip_discs(i, value):
            dummy_state = value[0]
            discs_flippable = value[1]
            break_cond = value[2]
            side_in_which_discs_are_flipped = value[3]

            # FLIP_UP, FLIP_DOWN, FLIP_RIGHT, FLIP_LEFT, FLIP_UP_RIGHT, FLIP_DOWN_RIGHT, FLIP_DOWN_LEFT, FLIP_UP_LEFT
            idx, idy = jax.lax.switch(side_in_which_discs_are_flipped, [lambda:(x-i-1, y), lambda:(x+i+1, y), lambda:(x, y+i+1), lambda:(x, y-i-1),
                                                                        lambda:(x-i-1, y+i+1), lambda:(x+i+1, y+i+1), lambda:(x+i+1, y-i-1), lambda:(x-i-1,y-i-1)])

            def empty_field(dummy_state):
                break_cond = True
                return (dummy_state[0], discs_flippable, break_cond, side_in_which_discs_are_flipped)

            def friendly_field(dummy_state):
                discs_flippable = jax.lax.cond(i == 0, lambda _: False, lambda _: True, operand=None)
                break_cond = True
                return (dummy_state[0], discs_flippable, break_cond, side_in_which_discs_are_flipped)

            def enemy_field(dummy_state):
                dummy_state = dummy_state[0]._replace(
                    field=dummy_state[0].field._replace(
                        field_color=dummy_state[0].field.field_color.at[idx, idy].set(friendly_color)
                    )
                )
                return (dummy_state, discs_flippable, break_cond, side_in_which_discs_are_flipped)

            return jax.lax.cond(
                break_cond,
                lambda _: (dummy_state, discs_flippable, break_cond, side_in_which_discs_are_flipped),
                lambda _: jax.lax.cond(
                    jnp.equal(dummy_state.field.field_color[idx, idy], FieldColor.EMPTY),
                    lambda _: empty_field((dummy_state, _, _, _)),
                    lambda _: jax.lax.cond(
                        dummy_state.field.field_color[idx, idy] == friendly_color,
                        # jnp.equal(dummy_state.field.field_color[idx, idy], friendly_color)
                        lambda _: friendly_field((dummy_state, _, _, _)),
                        lambda _: enemy_field((dummy_state, _, _, _)),
                        operand=None
                    ),
                    operand=None
                ),
                operand=None
            )

        # flip disc on the upper side
        new_state, discs_flippable, _, _ = jax.lax.fori_loop(0, x, loop_horizontal_and_vertical_line_to_flip_discs, (curr_state, False, False, FLIP_UP_SIDE))
        # in case, that the the discs aren't flippable (== False) reverse the state back
        new_state = jax.lax.cond(discs_flippable, lambda _: new_state, lambda _: curr_state, operand=None)
        valid_choice = valid_choice | discs_flippable
        curr_state = new_state

        # flip disc on the down side
        new_state, discs_flippable, _, _ = jax.lax.fori_loop(0, FIELD_HEIGHT - x - 1, loop_horizontal_and_vertical_line_to_flip_discs, (curr_state, False, False, FLIP_DOWN_SIDE))
        new_state = jax.lax.cond(discs_flippable, lambda _: new_state, lambda _: curr_state, operand=None)
        valid_choice = valid_choice | discs_flippable
        curr_state = new_state

        # flip disc on the right side
        new_state, discs_flippable, _, _ = jax.lax.fori_loop(0, FIELD_WIDTH - y - 1, loop_horizontal_and_vertical_line_to_flip_discs, (curr_state, False, False, FLIP_RIGHT_SIDE))
        new_state = jax.lax.cond(discs_flippable, lambda _: new_state, lambda _: curr_state, operand=None)
        valid_choice = valid_choice | discs_flippable
        curr_state = new_state

        # flip disc on the left side
        new_state, discs_flippable, _, _ = jax.lax.fori_loop(0, y, loop_horizontal_and_vertical_line_to_flip_discs, (curr_state, False, False, FLIP_LEFT_SIDE))
        new_state = jax.lax.cond(discs_flippable, lambda _: new_state, lambda _: curr_state, operand=None)
        valid_choice = valid_choice | discs_flippable
        curr_state = new_state

        # flip disc on upper right side
        gap_upper_border, gap_right_border = x, FIELD_WIDTH - y - 1
        it_number = jax.lax.cond((gap_upper_border < gap_right_border), lambda _: gap_upper_border, lambda _: gap_right_border, operand=None)
        new_state, discs_flippable, _, _ = jax.lax.fori_loop(0, it_number, loop_horizontal_and_vertical_line_to_flip_discs, (curr_state, False, False, FLIP_UP_RIGHT_SIDE))
        new_state = jax.lax.cond(discs_flippable, lambda _: new_state, lambda _: curr_state, operand=None)
        valid_choice = valid_choice | discs_flippable
        curr_state = new_state

        # flip disc on down right side
        gap_down_border, gap_right_border = FIELD_WIDTH - x - 1, FIELD_WIDTH - y - 1
        it_number = jax.lax.cond((gap_down_border < gap_right_border), lambda _: gap_down_border, lambda _: gap_right_border, operand=None)
        new_state, discs_flippable, _, _ = jax.lax.fori_loop(0, it_number, loop_horizontal_and_vertical_line_to_flip_discs, (curr_state, False, False, FLIP_DOWN_RIGHT_SIDE))
        new_state = jax.lax.cond(discs_flippable, lambda _: new_state, lambda _: curr_state, operand=None)
        valid_choice = valid_choice | discs_flippable
        curr_state = new_state

        # flip disc on down left side
        gap_down_border, gap_left_border = FIELD_WIDTH - x - 1, y
        it_number = jax.lax.cond((gap_down_border < gap_left_border), lambda _: gap_down_border, lambda _: gap_left_border, operand=None)
        new_state, discs_flippable, _, _ = jax.lax.fori_loop(0, it_number, loop_horizontal_and_vertical_line_to_flip_discs, (curr_state, False, False, FLIP_DOWN_LEFT_SIDE))
        new_state = jax.lax.cond(discs_flippable, lambda _: new_state, lambda _: curr_state, operand=None)
        valid_choice = valid_choice | discs_flippable
        curr_state = new_state

        # flip disc on up left side
        gap_upper_border, gap_left_border = x, y
        it_number = jax.lax.cond((gap_upper_border < gap_left_border), lambda _: gap_upper_border, lambda _: gap_left_border, operand=None)
        new_state, discs_flippable, _, _ = jax.lax.fori_loop(0, it_number, loop_horizontal_and_vertical_line_to_flip_discs, (curr_state, False, False, FLIP_UP_LEFT_SIDE))
        new_state = jax.lax.cond(discs_flippable, lambda _: new_state, lambda _: curr_state, operand=None)
        valid_choice = valid_choice | discs_flippable
        curr_state = new_state

        # if the choice is valid, the empty field is now set in the friendly (human player = white color) color.
        new_state = jax.lax.cond(
            valid_choice,
            lambda _: new_state._replace(
                field=new_state.field._replace(
                    field_color=new_state.field.field_color.at[x, y].set(friendly_color)
                ),
                field_choice_player=field_choice
            ),
            lambda _: new_state,
            operand=None
        )

        # update scores
        player_score = jnp.sum(new_state.field.field_color == FieldColor.WHITE)
        enemy_score = jnp.sum(new_state.field.field_color == FieldColor.BLACK)
        new_state = new_state._replace(
            player_score=player_score
        )
        new_state = new_state._replace(
            enemy_score=enemy_score
        )
        return valid_choice, new_state
    return jax.lax.cond(
        curr_state.field.field_color[x, y] == FieldColor.EMPTY,
        lambda x: if_empty(x),
        lambda x: (False, x),
        curr_state
    )


@jax.jit
def check_if_there_is_a_valid_choice(curr_state, white_player):
    # white_player == True --> player
    # white_player == False --> enemy
    enemy_color = jax.lax.cond(
        white_player,
        lambda _: FieldColor.BLACK,
        lambda _: FieldColor.WHITE,
        operand=None
    )
    friendly_color = jax.lax.cond(
        white_player,
        lambda _: FieldColor.WHITE,
        lambda _: FieldColor.BLACK,
        operand=None
    )

    # If True is returned -> there will be atleast a valid choice which can be made
    # Use loop to check for every field places
    valid_field = False
    def outer_loop(i, carry): 
        valid_field = carry
        def inner_loop(j, carry):
            def if_empty(valid_field):
                def check_horizontal_vertical_and_diagonal_line(i, value):
                    valid_field = value[0]
                    break_cond = value[1]
                    side_in_which_discs_are_flipped = value[2]

                    # FLIP_UP, FLIP_DOWN, FLIP_RIGHT, FLIP_LEFT, FLIP_UP_RIGHT, FLIP_DOWN_RIGHT, FLIP_DOWN_LEFT, FLIP_UP_LEFT
                    idx, idy = jax.lax.switch(side_in_which_discs_are_flipped, [lambda:(x-i-1, y), lambda:(x+i+1, y), lambda:(x, y+i+1), lambda:(x, y-i-1),
                                                                                lambda:(x-i-1, y+i+1), lambda:(x+i+1, y+i+1), lambda:(x+i+1, y-i-1), lambda:(x-i-1,y-i-1)])

                    def empty_field(value):
                        break_cond = True
                        return (valid_field, break_cond, side_in_which_discs_are_flipped)

                    def friendly_field(value):
                        valid_field = jax.lax.cond(i == 0, lambda _: False, lambda _: True, operand=None)
                        break_cond = True
                        return (valid_field, break_cond, side_in_which_discs_are_flipped)

                    def enemy_field(value):
                        return (valid_field, break_cond, side_in_which_discs_are_flipped)

                    return jax.lax.cond(
                        break_cond | valid_field,
                        lambda _: (valid_field, break_cond, side_in_which_discs_are_flipped),
                        lambda _: jax.lax.cond(
                            jnp.equal(curr_state.field.field_color[idx, idy], FieldColor.EMPTY),
                            lambda _: empty_field((_, _, _)),
                            lambda _: jax.lax.cond(
                                curr_state.field.field_color[idx, idy] == friendly_color,
                                lambda _: friendly_field((_, _, _)),
                                lambda _: enemy_field((_, _, _)),
                                operand=None
                            ),
                            operand=None
                        ),
                        operand=None
                    )

                x = i
                y = j

                # check disc on the upper side
                valid_field, _, _ = jax.lax.fori_loop(0, x, check_horizontal_vertical_and_diagonal_line,  (valid_field, False, FLIP_UP_SIDE))

                # check disc on the down side
                valid_field, _, _ = jax.lax.fori_loop(0, FIELD_HEIGHT - x - 1, check_horizontal_vertical_and_diagonal_line, (valid_field, False, FLIP_DOWN_SIDE))

                # check disc on the right side
                valid_field, _, _ = jax.lax.fori_loop(0, FIELD_WIDTH - y - 1, check_horizontal_vertical_and_diagonal_line, (valid_field, False, FLIP_RIGHT_SIDE))
                
                # check disc on the left side
                valid_field, _, _ = jax.lax.fori_loop(0, y, check_horizontal_vertical_and_diagonal_line, (valid_field, False, FLIP_LEFT_SIDE))
                
                # check disc on upper right side
                gap_upper_border, gap_right_border = x, FIELD_WIDTH - y - 1
                it_number = jax.lax.cond((gap_upper_border < gap_right_border), lambda _: gap_upper_border, lambda _: gap_right_border, operand=None)
                valid_field, _, _ = jax.lax.fori_loop(0, it_number, check_horizontal_vertical_and_diagonal_line, (valid_field, False, FLIP_UP_RIGHT_SIDE))
                
                # check disc on down right side
                gap_down_border, gap_right_border = FIELD_WIDTH - x - 1, FIELD_WIDTH - y - 1
                it_number = jax.lax.cond((gap_down_border < gap_right_border), lambda _: gap_down_border, lambda _: gap_right_border, operand=None)
                valid_field, _, _ = jax.lax.fori_loop(0, it_number, check_horizontal_vertical_and_diagonal_line, (valid_field, False, FLIP_DOWN_RIGHT_SIDE))
                
                # check disc on down left side
                gap_down_border, gap_left_border = FIELD_WIDTH - x - 1, y
                it_number = jax.lax.cond((gap_down_border < gap_left_border), lambda _: gap_down_border, lambda _: gap_left_border, operand=None)
                valid_field, _, _ = jax.lax.fori_loop(0, it_number, check_horizontal_vertical_and_diagonal_line, (valid_field, False, FLIP_DOWN_LEFT_SIDE))
                
                # check disc on up left side
                gap_upper_border, gap_left_border = x, y
                it_number = jax.lax.cond((gap_upper_border < gap_left_border), lambda _: gap_upper_border, lambda _: gap_left_border, operand=None)
                valid_field, _, _ = jax.lax.fori_loop(0, it_number, check_horizontal_vertical_and_diagonal_line, (valid_field, False, FLIP_UP_LEFT_SIDE))
                
                return valid_field

            valid_field = carry
            return jax.lax.cond(
                jnp.logical_and(curr_state.field.field_color[i, j] == FieldColor.EMPTY, jnp.logical_not(valid_field)),
                lambda x: if_empty(x),
                lambda x: x,
                valid_field
            )
        return jax.lax.fori_loop(0, FIELD_WIDTH, inner_loop, (valid_field))
    valid_field = jax.lax.fori_loop(0, FIELD_HEIGHT, outer_loop, (valid_field))
    return (curr_state, valid_field)


@jax.jit
def get_bot_move(game_field: Field, difficulty: chex.Array, player_score: chex.Array, enemy_score: chex.Array, random_key: int):
    game_score = player_score+enemy_score
    list_of_all_moves = jnp.arange(64)

    #calculate flipped tiles for all possible moves and adjust score based on game stage
    vectorized_compute_score_of_tiles = jax.vmap(compute_score_of_tiles, in_axes=(0, None, None, None))
    list_of_all_move_values = vectorized_compute_score_of_tiles(list_of_all_moves, game_field, game_score, difficulty)


    #Randomly choose one of the best moves    
    random_chosen_max_index = random_max_index(list_of_all_move_values,random_key)

    
    return jnp.array([jnp.floor_divide(random_chosen_max_index, 8), jnp.mod(random_chosen_max_index, 8)])

def compute_score_of_tiles(i, game_field, game_score, difficulty):
    # Decode tile position
    tile_y = jnp.floor_divide(i, 8)
    tile_x = jnp.mod(i, 8)

    #jax.debug.print("tile_y: {}, tile_x: {}", tile_y, tile_x)

    def helper_return_default_pos():
        return (jnp.int32(-2147483648),(jnp.int32(-2147483648), jnp.int32(-2147483648)),(jnp.int32(-2147483648), jnp.int32(-2147483648)))


    # If tile is already taken, set invalid move (return very low score)
    args_compute_score_of_tile_1 = (tile_y, tile_x, game_field, game_score)
    tiles_flipped, secondary_tile, default_pos = jax.lax.cond(
        game_field.field_color[tile_y, tile_x] != FieldColor.EMPTY,
        lambda args_compute_score_of_tile_1: helper_return_default_pos(),
        lambda args_compute_score_of_tile_1: compute_score_of_tile_1(args_compute_score_of_tile_1),
        args_compute_score_of_tile_1
    )
    jax.debug.print("tiles_flipped: {}, secondary_tile: {}, default_pos: {}", tiles_flipped, secondary_tile, default_pos)
    

    def handle_calculation_of_strategic_score(args):
        i, game_field, default_pos, difficulty, secondary_tile = args

        
        #Calculate the strategic value (score) of the current_square itself
        ((score, _), skip_secondary_eval) = calculate_strategic_tile_score(i, game_field, jax.lax.cond(default_pos[0] == -2147483648, lambda _: (0,0),lambda _: default_pos, None), difficulty)

        #jax.debug.print("score: {}, skip_secondary_eval: {}", score, skip_secondary_eval)
        secondary_tile = jax.lax.cond(
            skip_secondary_eval,
            lambda _: (-2147483648, -2147483648),
            lambda _: secondary_tile,
            None
        )

        jax.debug.print("i: {},secondary_tile[0]: {},secondary_tile[1]: {}", i, secondary_tile[0], secondary_tile[1])
        #Re-evaluate or combine with a secondary related square's score
        args_handle_secondary_calculation_of_strategic_score = (secondary_tile[0] + secondary_tile[1]*8, game_field, default_pos, difficulty, score)
        score = jax.lax.cond(
            secondary_tile[0] != -2147483648,
            lambda args_handle_secondary_calculation_of_strategic_score: handle_secondary_calculation_of_strategic_score(args_handle_secondary_calculation_of_strategic_score),
            lambda args_handle_secondary_calculation_of_strategic_score: score,
            args_handle_secondary_calculation_of_strategic_score
        )
        return score
    
    #only execute the strategic score calculation if tiles were flipped = is a valid move
    jax.debug.print(str(tiles_flipped != jnp.array(-2147483648, dtype=jnp.int32)))
    jax.debug.print(str(tiles_flipped != -2147483648))

    args = (i, game_field, default_pos, difficulty, secondary_tile)
    #score = jax.lax.cond(tiles_flipped != jnp.array(-2147483648, dtype=jnp.int32),
    score = jax.lax.cond(False,
                 lambda args: tiles_flipped + handle_calculation_of_strategic_score(args),
                 lambda args: tiles_flipped,
                 args)


    #jax.debug.print("score: {}", score)
    return score

@jax.jit
def compute_score_of_tile_1(args):
    tile_y, tile_x, game_field, game_score = args
    inner_corner_in_any_direction = (-2147483648, -2147483648)
    inner_corner_in_final_direction = (-2147483648, -2147483648)

    #compute flipped tiles by each direction
    list_of_all_directions = jnp.arange(8)
    vectorised_flipped_tiles_by_direction = jax.vmap(compute_flipped_tiles_by_direction,in_axes=(0, None, None, None, None, None))
    flipped_tiles_by_direction, inner_corner_by_direction_any_direction, inner_corner_by_direction_final_direction = vectorised_flipped_tiles_by_direction(list_of_all_directions, tile_y, tile_x, game_field, inner_corner_in_any_direction, inner_corner_in_final_direction)

    def find_last_valid_corner(arr):
        idxs = jnp.arange(len(arr[0]))
        valid_positions = jnp.where(arr != -2147483648, idxs, -1)
        return jnp.max(valid_positions)

    inner_corner_in_any_direction = find_last_valid_corner(inner_corner_by_direction_any_direction)
    inner_corner_in_final_direction = find_last_valid_corner(inner_corner_by_direction_final_direction)

    flipped_tiles = jnp.nansum(flipped_tiles_by_direction)
    # If no tiles were flipped, set invalid move (return very low score)
    score_of_tile = jax.lax.cond(
        flipped_tiles == 0,
        lambda _: -2147483648,  
        lambda _: 0,  
        None
    )

    # Adjust for game stage
    score_of_tile = jax.lax.cond(
        score_of_tile == -2147483648,
        lambda _: score_of_tile,  
        lambda _: jax.lax.cond(  
            jnp.squeeze(jnp.logical_and(game_score > 18, game_score < 41)),  
            lambda _: -flipped_tiles - 1,  
            lambda _: flipped_tiles,  
            None
        ),
        None
    )
    return jax.lax.cond(
                jnp.logical_and(inner_corner_by_direction_any_direction == -2147483648, inner_corner_by_direction_final_direction == -2147483648),
                lambda _: (score_of_tile, (-2147483648, -2147483648), (-2147483648, -2147483648)),
                lambda _: jax.lax.cond(
                    jnp.logical_and(inner_corner_by_direction_any_direction == -2147483648, inner_corner_by_direction_final_direction != -2147483648),
                    lambda _: (score_of_tile, (-2147483648, -2147483648), (inner_corner_in_final_direction % 8, inner_corner_in_final_direction // 8)),
                    lambda _: jax.lax.cond(
                        jnp.logical_and(inner_corner_by_direction_any_direction != -2147483648, inner_corner_by_direction_final_direction == -2147483648),
                        lambda _: (score_of_tile, (inner_corner_in_any_direction % 8, inner_corner_in_any_direction // 8), (-2147483648, -2147483648)),
                        lambda _: (score_of_tile, (inner_corner_in_any_direction % 8, inner_corner_in_any_direction // 8), (inner_corner_in_final_direction % 8, inner_corner_in_final_direction // 8)),
                        None
                        ),
                    None
                    ),
                None
                )

# array size fixed at 64!!
@jax.jit
def random_max_index(array, key:int):
    max_value  = jnp.max(array)
    max_value_count = 0
    max_values = jnp.zeros_like(array)
    index = 0
    init_val = (index, array, max_value, max_value_count, max_values)

    # loop that iterates over all 64 possible moves and checks, how many top moves are there, returns a touple with:
    # (  index(to be disregarded) array(the input array), max_value(the maximum value within the array), max_value_count(how often the maxmium appeared), max_values(array where max values are marked with their index, everythin else is zero))
    def count_max_value(i, val):
        array, max_value = val[1],  val[2]
        tmp = jax.lax.cond(array[i] == max_value, lambda val: true_fun(val), lambda val: false_fun(val),val)
        return tmp

    def true_fun(val):
        index, array,  max_value, max_value_count, max_values = val
        max_values = max_values.at[index].set(index+1)
        index+=1
        max_value_count+=1
        return (index, array, max_value, max_value_count, max_values)

    def false_fun(val):
        index= val[0]
        index+=1
        return (index, val[1], val[2], val[3], val[4])
    
    _, _, _, max_value_count, max_indexes = jax.lax.fori_loop(0, array.size, count_max_value, init_val)

    #sorts all max Value indexes to the left, then generates a "random" number and takes the max value at the random index
    max_indexes_for_random = jnp.nonzero(max_indexes, size= 64, fill_value=-1)[0]
    rand_index = jax.random.randint(key, shape=(), minval=jnp.array(0), maxval=max_value_count)

    return jnp.take(max_indexes_for_random, rand_index, mode="clip")


@jax.jit
def compute_flipped_tiles_by_direction(i, tile_y: int, tile_x: int, game_field: Field, inner_corner_in_any_direction: tuple[int, int], inner_corner_in_final_direction: tuple[int, int]):
    args = (tile_y,tile_x,game_field,inner_corner_in_any_direction,inner_corner_in_final_direction)

    branches = [
        lambda args: compute_flipped_tiles_top(args),
        lambda args: compute_flipped_tiles_top_rigth(args),
        lambda args: compute_flipped_tiles_rigth(args),
        lambda args: compute_flipped_tiles_bottom_rigth(args),
        lambda args: compute_flipped_tiles_bottom(args),
        lambda args: compute_flipped_tiles_bottom_left(args),
        lambda args: compute_flipped_tiles_left(args),
        lambda args: compute_flipped_tiles_top_left(args),
    ]

    return jax.lax.switch(i, branches, args)

def check_if_inner_corner(tile_y, tile_x):
    return jax.lax.cond(jnp.logical_and(tile_y == 1, tile_x == 1),
            lambda _: (6,6), #account for flipped game field in calculate_strategic_tile_score
            lambda _: jax.lax.cond(jnp.logical_and(tile_y == 1, tile_x == 6),
                        lambda _: (6,1),
                        lambda _: jax.lax.cond(jnp.logical_and(tile_y == 6, tile_x == 1),
                                    lambda _: (1,6),
                                    lambda _: jax.lax.cond(jnp.logical_and(tile_y == 6, tile_x == 6),
                                                lambda _: (1,1),
                                                lambda _: (tile_y, tile_x),
                                                None),
                                    None),
                        None),
            None)

@jax.jit   
def compute_flipped_tiles_top(input) -> int:
    # Returns the number of tiles to be flipped (only flipped tiles, same color tiles are disregarded)

    #checks wether the first element in the direction of look-up is invalid, becasause its already taken by bot, sets it to nan, to prevent while loop from running
    args = jax.lax.cond(
        input[2].field_color[input[0]-1][input[1]] == FieldColor.BLACK,
        lambda input: (input[0] - 1, input[1], input[2], jnp.nan, 0, input[3], input[4]),
        lambda input: (input[0] - 1, input[1], input[2], 0.0, 0, input[3], input[4]),
        input
    ) 

    #args has the following structure: tile_y, tile_x, game_field, flipped, tmp_flipped, inner_corner_in_final_direction, inner_corner_in_any_direction
    def while_cond(args):
        #check if to be checked field is still part of game field
        check_for_outside_borders_of_game_field = jnp.all(
            jnp.array([args[0] >= 0, args[0] < 8, args[1] >= 0, args[1] < 8, args[3] != jnp.nan])
        )
        return check_for_outside_borders_of_game_field

    def while_body(args):
        #when we encounter a not black field increase tmp_flipped, when we encounter a black field add the tmp_flipped (tiles we encountered before black field:= tiles to be flipped) to flipped tiles      
        return jax.lax.cond(
            args[2].field_color[args[0], args[1]] == FieldColor.EMPTY,  
            lambda args: (-2, args[1], args[2], args[3],  args[4], args[5], args[6]),#if field is empthy, no further tiles can be flipped use set y to -2 to exit next loop iteration
            lambda args: (jax.lax.cond(
                args[2].field_color[args[0], args[1]] == FieldColor.BLACK,  
                lambda args: (args[0]-1, args[1], args[2], args[3] + args[4], 0, args[5], args[5]), #args[5] is no typo (set inner_corner_in_any_direction only when valid move is found)
                lambda args: (args[0]-1, args[1], args[2], args[3], args[4] + 1, check_if_inner_corner(args[0], args[1]), args[6]),
                args
                )),
            args
        )

    while_loop_return = jax.lax.while_loop(while_cond, while_body, args)
    return jax.lax.cond(
        input[0] < 0,
        lambda args: (jnp.int32(0.0), (-2147483648,-2147483648), (-2147483648,-2147483648)),
        lambda args: (jnp.int32(while_loop_return[3]), while_loop_return[6], while_loop_return[5]),
        args
    )
@jax.jit
def compute_flipped_tiles_rigth(input) -> int:
    # Returns the number of tiles to be flipped (only flipped tiles, same color tiles are disregarded)

    #checks wether the first element in the direction of look-up is invalid, becasause its already taken by bot, sets it to nan, to prevent while loop from running
    args = jax.lax.cond(
        input[2].field_color[input[0]][input[1]+1] == FieldColor.BLACK,
        lambda input: (input[0], input[1]+1, input[2], jnp.nan, 0, input[3], input[4]),
        lambda input: (input[0], input[1]+1, input[2], 0.0, 0, input[3], input[4]),
        input
    ) 

    #args has the following structure: tile_y, tile_x, game_field, flipped, tmp_flipped
    def while_cond(args):
        #check if to be checked field is still part of game field
        check_for_outside_borders_of_game_field = jnp.all(
            jnp.array([args[0] >= 0, args[0] < 8, args[1] >= 0, args[1] < 8, args[3] != jnp.nan])
        )
        return check_for_outside_borders_of_game_field

    def while_body(args):
        #when we encounter a not black field increase tmp_flipped, when we encounter a black field add the tmp_flipped (tiles we encountered before black field:= tiles to be flipped) to flipped tiles      
        return jax.lax.cond(
            args[2].field_color[args[0], args[1]] == FieldColor.EMPTY,  
            lambda args: (-2, args[1], args[2], args[3],  args[4], args[5], args[6]),#if field is empthy, no further tiles can be flipped use set y to -2 to exit next loop iteration
            lambda args: (jax.lax.cond(
                args[2].field_color[args[0], args[1]] == FieldColor.BLACK,  
                lambda args: (args[0], args[1]+1, args[2], args[3] + args[4], 0,  args[5], args[5]),
                lambda args: (args[0], args[1]+1, args[2], args[3], args[4] + 1, check_if_inner_corner(args[0], args[1]), args[6]),
                args
                )),
            args
        )

    while_loop_return = jax.lax.while_loop(while_cond, while_body, args)
    return jax.lax.cond(
        input[1] > 7,
        lambda args: (jnp.int32(0.0), (-2147483648,-2147483648), (-2147483648,-2147483648)),
        lambda args: (jnp.int32(while_loop_return[3]), while_loop_return[6], while_loop_return[5]),
        args
    )
@jax.jit
def compute_flipped_tiles_bottom(input) -> int:
    # Returns the number of tiles to be flipped (only flipped tiles, same color tiles are disregarded)

    #checks wether the first element in the direction of look-up is invalid, becasause its already taken by bot, sets it to nan, to prevent while loop from running
    args = jax.lax.cond(
        input[2].field_color[input[0]+1][input[1]] == FieldColor.BLACK,
        lambda input: (input[0] + 1, input[1], input[2], jnp.nan, 0, input[3], input[4]),
        lambda input: (input[0] + 1, input[1], input[2], 0.0, 0, input[3], input[4]),
        input
    ) 

    #args has the following structure: tile_y, tile_x, game_field, flipped, tmp_flipped
    def while_cond(args):
        #check if to be checked field is still part of game field
        check_for_outside_borders_of_game_field = jnp.all(
            jnp.array([args[0] >= 0, args[0] < 8, args[1] >= 0, args[1] < 8, args[3] != jnp.nan])
        )
        return check_for_outside_borders_of_game_field

    def while_body(args):
        #when we encounter a not black field increase tmp_flipped, when we encounter a black field add the tmp_flipped (tiles we encountered before black field:= tiles to be flipped) to flipped tiles      
        return jax.lax.cond(
            args[2].field_color[args[0], args[1]] == FieldColor.EMPTY,  
            lambda args: (-2, args[1], args[2], args[3],  args[4], args[5], args[6]),#if field is empthy, no further tiles can be flipped use set y to -2 to exit next loop iteration
            lambda args: (jax.lax.cond(
                args[2].field_color[args[0], args[1]] == FieldColor.BLACK,  
                lambda args: (args[0]+1, args[1], args[2], args[3] + args[4], 0, args[5], args[5]),
                lambda args: (args[0]+1, args[1], args[2], args[3], args[4] + 1, check_if_inner_corner(args[0], args[1]), args[6]),
                args
                )),
            args
        )

    while_loop_return = jax.lax.while_loop(while_cond, while_body, args)
    return jax.lax.cond(
        input[0] > 7,
        lambda args: (jnp.int32(0.0), (-2147483648,-2147483648), (-2147483648,-2147483648)),
        lambda args: (jnp.int32(while_loop_return[3]), while_loop_return[6], while_loop_return[5]),
        args
    )
@jax.jit
def compute_flipped_tiles_left(input) -> int:
    # Returns the number of tiles to be flipped (only flipped tiles, same color tiles are disregarded)

    #checks wether the first element in the direction of look-up is invalid, becasause its already taken by bot, sets it to nan, to prevent while loop from running
    args = jax.lax.cond(
        input[2].field_color[input[0]][input[1]-1] == FieldColor.BLACK,
        lambda input: (input[0], input[1]-1, input[2], jnp.nan, 0, input[3], input[4]),
        lambda input: (input[0], input[1]-1, input[2], 0.0, 0, input[3], input[4]),
        input
    ) 

    #args has the following structure: tile_y, tile_x, game_field, flipped, tmp_flipped
    def while_cond(args):
        #check if to be checked field is still part of game field
        check_for_outside_borders_of_game_field = jnp.all(
            jnp.array([args[0] >= 0, args[0] < 8, args[1] >= 0, args[1] < 8, args[3] != jnp.nan])
        )
        return check_for_outside_borders_of_game_field

    def while_body(args):
        #when we encounter a not black field increase tmp_flipped, when we encounter a black field add the tmp_flipped (tiles we encountered before black field:= tiles to be flipped) to flipped tiles      
        return jax.lax.cond(
            args[2].field_color[args[0], args[1]] == FieldColor.EMPTY,
            lambda args: (-2, args[1], args[2], args[3], args[4], args[5], args[6]),  # if field is empthy, no further tiles can be flipped use set y to -2 to exit next loop iteration
            lambda args: (jax.lax.cond(
                args[2].field_color[args[0], args[1]] == FieldColor.BLACK,
                lambda args: (args[0], args[1]-1, args[2], args[3] + args[4], 0, args[5], args[5]),
                lambda args: (args[0], args[1]-1, args[2], args[3], args[4] + 1, check_if_inner_corner(args[0], args[1]), args[6]),
                args
                )),
            args
        )

    while_loop_return = jax.lax.while_loop(while_cond, while_body, args)
    return jax.lax.cond(
        input[1] > 7,
        lambda _: (jnp.int32(0.0), (-2147483648,-2147483648), (-2147483648,-2147483648)),
        lambda _: (jnp.int32(while_loop_return[3]), while_loop_return[6], while_loop_return[5]),
        None
    )
@jax.jit
def compute_flipped_tiles_top_rigth(input) -> int:
    # Returns the number of tiles to be flipped (only flipped tiles, same color tiles are disregarded)

    #checks wether the first element in the direction of look-up is invalid, becasause its already taken by bot, sets it to nan, to prevent while loop from running
    args = jax.lax.cond(
        input[2].field_color[input[0]-1][input[1]+1] == FieldColor.BLACK,
        lambda input: (input[0]-1, input[1]+1, input[2], jnp.nan, 0, input[3], input[4]),
        lambda input: (input[0]-1, input[1]+1, input[2], 0.0, 0, input[3], input[4]),
        input
    ) 

    #args has the following structure: tile_y, tile_x, game_field, flipped, tmp_flipped
    def while_cond(args):
        #check if to be checked field is still part of game field
        check_for_outside_borders_of_game_field = jnp.all(
            jnp.array([args[0] >= 0, args[0] < 8, args[1] >= 0, args[1] < 8, args[3] != jnp.nan])
        )
        return check_for_outside_borders_of_game_field

    def while_body(args):
        #when we encounter a not black field increase tmp_flipped, when we encounter a black field add the tmp_flipped (tiles we encountered before black field:= tiles to be flipped) to flipped tiles      
        return jax.lax.cond(
            args[2].field_color[args[0], args[1]] == FieldColor.EMPTY,  
            lambda args: (-2, args[1], args[2], args[3],  args[4], args[5], args[6]),#if field is empthy, no further tiles can be flipped use set y to -2 to exit next loop iteration
            lambda args: (jax.lax.cond(
                args[2].field_color[args[0], args[1]] == FieldColor.BLACK,
                lambda args: (args[0]-1, args[1]+1, args[2], args[3] + args[4], 0, args[5], args[5]),
                lambda args: (args[0]-1, args[1]+1, args[2], args[3], args[4] + 1, check_if_inner_corner(args[0], args[1]), args[6]),
                args
                )),
            args
        )

    while_loop_return = jax.lax.while_loop(while_cond, while_body, args)
    return jax.lax.cond(
        jnp.logical_and(input[0] < 0, input[1] > 7),
        lambda args: (jnp.int32(0.0), (-2147483648, -2147483648), (-2147483648, -2147483648)),
        lambda args: (jnp.int32(while_loop_return[3]), while_loop_return[6], while_loop_return[5]),
        args
    )
@jax.jit
def compute_flipped_tiles_bottom_rigth(input) -> int:
    # Returns the number of tiles to be flipped (only flipped tiles, same color tiles are disregarded)

    #checks wether the first element in the direction of look-up is invalid, becasause its already taken by bot, sets it to nan, to prevent while loop from running
    args = jax.lax.cond(
        input[2].field_color[input[0]+1][input[1]+1] == FieldColor.BLACK,
        lambda input: (input[0]+1, input[1]+1, input[2], jnp.nan, 0, input[3], input[4]),
        lambda input: (input[0]+1, input[1]+1, input[2], 0.0, 0, input[3], input[4]),
        input
    ) 

    #args has the following structure: tile_y, tile_x, game_field, flipped, tmp_flipped
    def while_cond(args):
        #check if to be checked field is still part of game field
        check_for_outside_borders_of_game_field = jnp.all(
            jnp.array([args[0] >= 0, args[0] < 8, args[1] >= 0, args[1] < 8, args[3] != jnp.nan])
        )
        return check_for_outside_borders_of_game_field

    def while_body(args):
        #when we encounter a not black field increase tmp_flipped, when we encounter a black field add the tmp_flipped (tiles we encountered before black field:= tiles to be flipped) to flipped tiles      
        return jax.lax.cond(
            args[2].field_color[args[0], args[1]] == FieldColor.EMPTY,  
            lambda args: (-2, args[1], args[2], args[3],  args[4], args[5], args[6]),#if field is empthy, no further tiles can be flipped use set y to -2 to exit next loop iteration
            lambda args: (jax.lax.cond(
                args[2].field_color[args[0], args[1]] == FieldColor.BLACK,
                lambda args: (args[0]+1, args[1]+1, args[2], args[3] + args[4], 0, args[5], args[5]),
                lambda args: (args[0]+1, args[1]+1, args[2], args[3], args[4] + 1, check_if_inner_corner(args[0], args[1]), args[6]),
                args
                )),
            args
        )

    while_loop_return = jax.lax.while_loop(while_cond, while_body, args)
    return jax.lax.cond(
        jnp.logical_and(input[0] > 7, input[1] > 7),
        lambda args: (jnp.int32(0.0), (-2147483648, -2147483648), (-2147483648, -2147483648)),
        lambda args: (jnp.int32(while_loop_return[3]), while_loop_return[6], while_loop_return[5]),
        args
    )
@jax.jit
def compute_flipped_tiles_bottom_left(input) -> int:
    # Returns the number of tiles to be flipped (only flipped tiles, same color tiles are disregarded)

    #checks wether the first element in the direction of look-up is invalid, becasause its already taken by bot, sets it to nan, to prevent while loop from running
    args = jax.lax.cond(
        input[2].field_color[input[0]+1][input[1]-1] == FieldColor.BLACK,
        lambda input: (input[0]+1, input[1]-1, input[2], jnp.nan, 0, input[3], input[4]),
        lambda input: (input[0]+1, input[1]-1, input[2], 0.0, 0, input[3], input[4]),
        input
    ) 

    #args has the following structure: tile_y, tile_x, game_field, flipped, tmp_flipped
    def while_cond(args):
        #check if to be checked field is still part of game field
        check_for_outside_borders_of_game_field = jnp.all(
            jnp.array([args[0] >= 0, args[0] < 8, args[1] >= 0, args[1] < 8, args[3] != jnp.nan])
        )
        return check_for_outside_borders_of_game_field

    def while_body(args):
        #when we encounter a not black field increase tmp_flipped, when we encounter a black field add the tmp_flipped (tiles we encountered before black field:= tiles to be flipped) to flipped tiles      
        return jax.lax.cond(
            args[2].field_color[args[0], args[1]] == FieldColor.EMPTY,  
            lambda args: (-2, args[1], args[2], args[3],  args[4], args[5], args[6]),#if field is empthy, no further tiles can be flipped use set y to -2 to exit next loop iteration
            lambda args: (jax.lax.cond(
                args[2].field_color[args[0], args[1]] == FieldColor.BLACK,
                lambda args: (args[0]+1, args[1]-1, args[2], args[3] + args[4], 0, args[5], args[5]),
                lambda args: (args[0]+1, args[1]-1, args[2], args[3], args[4] + 1, check_if_inner_corner(args[0], args[1]), args[6]),
                args
                )),
            args
        )        


    while_loop_return = jax.lax.while_loop(while_cond, while_body, args)
    return jax.lax.cond(
        jnp.logical_and(input[0] < 0, input[1] > 7),
        lambda args: (jnp.int32(0.0), (-2147483648, -2147483648), (-2147483648, -2147483648)),
        lambda args: (jnp.int32(while_loop_return[3]), while_loop_return[6], while_loop_return[5]),
        args
    )
@jax.jit
def compute_flipped_tiles_top_left(input) -> int:
    # Returns the number of tiles to be flipped (only flipped tiles, same color tiles are disregarded)

    #checks wether the first element in the direction of look-up is invalid, becasause its already taken by bot, sets it to nan, to prevent while loop from running
    args = jax.lax.cond(
        input[2].field_color[input[0]-1][input[1]-1] == FieldColor.BLACK,
        lambda input: (input[0]-1, input[1]-1, input[2], jnp.nan, 0, input[3], input[4]),
        lambda input: (input[0]-1, input[1]-1, input[2], 0.0, 0, input[3], input[4]),
        input
    ) 

    #args has the following structure: tile_y, tile_x, game_field, flipped, tmp_flipped
    def while_cond(args):
        #check if to be checked field is still part of game field
        check_for_outside_borders_of_game_field = jnp.all(
            jnp.array([args[0] >= 0, args[0] < 8, args[1] >= 0, args[1] < 8, args[3] != jnp.nan])
        )
        return check_for_outside_borders_of_game_field

    def while_body(args):
        #when we encounter a not black field increase tmp_flipped, when we encounter a black field add the tmp_flipped (tiles we encountered before black field:= tiles to be flipped) to flipped tiles      
        return jax.lax.cond(
            args[2].field_color[args[0], args[1]] == FieldColor.EMPTY,
            lambda args: (-2, args[1], args[2], args[3],  args[4], args[5], args[6]),#if field is empthy, no further tiles can be flipped use set y to -2 to exit next loop iteration
            lambda args: (jax.lax.cond(
                args[2].field_color[args[0], args[1]] == FieldColor.BLACK,
                lambda args: (args[0]-1, args[1]-1, args[2], args[3] + args[4], 0, args[5], args[6]),
                lambda args: (args[0]-1, args[1]-1, args[2], args[3], args[4] + 1, check_if_inner_corner(args[0], args[1]), args[6]),
                args
                )),
            args
        )        


    while_loop_return = jax.lax.while_loop(while_cond, while_body, args)
    return jax.lax.cond(
        jnp.logical_and(input[0] < 0, input[1] < 0),
        lambda args: (jnp.int32(0.0), (-2147483648, -2147483648), (-2147483648, -2147483648)),
        lambda args: (jnp.int32(while_loop_return[3]), while_loop_return[6], while_loop_return[5]),
        args
    )

def handle_secondary_calculation_of_strategic_score(args) -> int:
    tile_index, game_field, default_pos, difficulty, new_score = args
    jax.debug.print("tile_index: {}, default_pos: {}, difficulty: {}", tile_index, default_pos, difficulty)
    new_score_copy = new_score
    ((new_score, new_alt_position), _) = calculate_strategic_tile_score(tile_index, game_field, default_pos, difficulty)



    new_score_copy = jax.lax.cond(
        new_alt_position != -2147483648,
        lambda _: new_alt_position,
        lambda _: new_score_copy,
        None
    )

    total_score = new_score + new_score_copy

    total_score = jax.lax.cond(jnp.logical_or(total_score > 127, total_score < -128),
        lambda _: -104,
        lambda _: total_score,
        None
    ) 

    return total_score

@jax.jit
def calculate_strategic_tile_score(tile_index: int, game_field: Field, default_pos: Tuple[int, int], difficulty: int):
    game_field_flipped = Field(field_id=game_field.field_id,
                               field_color=jnp.flip(game_field.field_color))
    default_pos_combined = default_pos[0] + default_pos[1] * 8

    #determine the position of the tile within the game field, account for flipped game field
    args = (7 - tile_index // 8, 7 - tile_index % 8, game_field_flipped, default_pos_combined, difficulty)
    branches = [
        lambda args: compute_strategic_score_top_left(args), 
        lambda args: compute_strategic_score_top(args),  
        lambda args: compute_strategic_score_top_right(args),  
        lambda args: compute_strategic_score_top_left_inner(args),
        lambda args: compute_strategic_score_top_inner(args), 
        lambda args: compute_strategic_score_top_right_inner(args),
        lambda args: compute_strategic_score_left(args),
        lambda args: compute_strategic_score_left_inner(args),
        lambda args: compute_strategic_score_center_corner(args),
        lambda args: compute_strategic_score_center(args),
        lambda args: compute_strategic_score_right_inner(args),
        lambda args: compute_strategic_score_right(args),
        lambda args: compute_strategic_score_bottom_left_inner(args),
        lambda args: compute_strategic_score_bottom_inner(args),
        lambda args: compute_strategic_score_bottom_right_inner(args),
        lambda args: compute_strategic_score_bottom_left(args),
        lambda args: compute_strategic_score_bottom(args),
        lambda args: compute_strategic_score_bottom_right(args),
    ]

    def compute_strategic_score_top_left(args):
        _, _, game_field_flipped, unknown_arg, difficulty = args
        return ((css_calculate_top_left_score(game_field_flipped, unknown_arg, difficulty)), True)

    def compute_strategic_score_top(args):
        tile_y, tile_x, game_field_flipped, unknown_arg, difficulty = args

        return ((css__f2d3_count_tiles_horizontally(game_field_flipped, tile_y, tile_x, unknown_arg, difficulty)), False)

    def compute_strategic_score_top_right(args):
        _, _, game_field_flipped, unknown_arg, difficulty = args

        return ((css_calculate_top_right_score(game_field_flipped, unknown_arg, difficulty)), True)

    def compute_strategic_score_top_left_inner(args):
        _, _, game_field_flipped, _, _ = args
        return ((css_check_three_tiles(game_field_flipped, 0xf, 0x39, 0x3f), -2147483648), False)

    def compute_strategic_score_top_inner(args):
        #Strategic score for top inner field is influenced by the field above it
        tile_y, tile_x, game_field_flipped, _, _ = args
        return ((css_check_tile_up(tile_y, tile_x, game_field_flipped), -2147483648), False)

    def compute_strategic_score_top_right_inner(args):
        _, _, game_field_flipped, _, _ = args
        return ((css_check_three_tiles(game_field_flipped, 0x8, 0x3e, 0x38), -2147483648), False)

    def compute_strategic_score_left(args):
        tile_y, tile_x, game_field_flipped, unknown_arg, difficulty = args

        return ((css__f2d3_count_tiles_vertically(game_field_flipped, tile_y, tile_x, unknown_arg, difficulty)), False)

    def compute_strategic_score_left_inner(args):
        #Strategic score for left inner field is influenced by the field to the left of it
        tile_y, tile_x, game_field_flipped, _, _ = args
        return ((css_check_tile_left(tile_y, tile_x, game_field_flipped), -2147483648), False)

    def compute_strategic_score_center_corner(args):
        # Center corner has a fixed score of 4, no further explanation needed
        return ((4, -2147483648), False)

    def compute_strategic_score_center(args):
        #center has a fixed score of 0, no further evaluation needed
        return ((0, -2147483648), False)

    def compute_strategic_score_right_inner(args):
        #Strategic score for right inner field is influenced by the field to the right of it
        tile_y, tile_x, game_field_flipped, unknown_arg, difficulty = args
        return ((css_check_tile_right(tile_y, tile_x, game_field_flipped), -2147483648), False)

    def compute_strategic_score_right(args):
        tile_y, tile_x, game_field_flipped, unknown_arg, difficulty = args

        return ((css__f2d3_count_tiles_vertically(game_field_flipped, tile_y, tile_x, unknown_arg, difficulty)), False)


    def compute_strategic_score_bottom_left_inner(args):
        _, _, game_field_flipped, _, _ = args
        return ((css_check_three_tiles(game_field_flipped, 0x1, 0x37, 0x7), -2147483648), False)

    def compute_strategic_score_bottom_inner(args):
        #Strategic score for bottom inner field is influenced by the field below it
        tile_y, tile_x, game_field_flipped, _, _ = args
        return ((css_check_tile_down(tile_y, tile_x, game_field_flipped), -2147483648), False)

    def compute_strategic_score_bottom_right_inner(args):
        _, _, game_field_flipped, _, _ = args
        return ((css_check_three_tiles(game_field_flipped, 0x6, 0x30, 0x0), -2147483648), False)

    def compute_strategic_score_bottom_left(args):
        _, _, game_field_flipped, unknown_arg, difficulty = args
        
        return ((css_calculate_bottom_left_score(game_field_flipped, unknown_arg, difficulty)), True)

    def compute_strategic_score_bottom(args):
        tile_y, tile_x, game_field_flipped, unknown_arg, difficulty = args

        return ((css__f2d3_count_tiles_horizontally(game_field_flipped, tile_y, tile_x, unknown_arg, difficulty)), False)

    def compute_strategic_score_bottom_right(args):
        _, _, game_field_flipped, unknown_arg, difficulty = args

        return ((css_calculate_bottom_right_score(game_field_flipped, unknown_arg, difficulty)), True)

    return jax.lax.switch(STRATEGIC_TILE_SCORE_CASES[tile_index], branches, args)

def css_check_tile_down(tile_y: int, tile_x: int, game_field: Field) -> int:
    # If field below has the same color as the current tile, return 4, else return -8
    return jax.lax.cond(
        game_field.field_color[tile_y - 1, tile_x] == game_field.field_color[tile_y , tile_x],
        lambda _: 4,
        lambda _: -8,
        None)

def css_check_tile_up(tile_y: int, tile_x: int, game_field: Field) -> int:
    # If field above has the same color as the current tile, return 4, else return -8
    return jax.lax.cond(
        game_field.field_color[tile_y + 1, tile_x] == game_field.field_color[tile_y , tile_x],
        lambda _: 4,
        lambda _: -8,
        None)

def css_check_tile_left(tile_y: int, tile_x: int, game_field: Field) -> int:
    # If field to the left has the same color as the current tile, return 4, else return -8
    return jax.lax.cond(
        game_field.field_color[tile_y, tile_x + 1] == game_field.field_color[tile_y , tile_x],
        lambda _: 4,
        lambda _: -8,
        None)

def css_check_tile_right(tile_y: int, tile_x: int, game_field: Field) -> int:
    # If field to the right has the same color as the current tile, return 4, else return -8
    return jax.lax.cond(
        game_field.field_color[tile_y, tile_x - 1] == game_field.field_color[tile_y , tile_x],
        lambda _: 4,
        lambda _: -8,
        None)

def css_check_three_tiles(game_field: Field, field_1: int, field_2: int, field_3: int) -> int:
    # checks for colors of the fields and returns a score based on the colors
    # field_1, field_2 are are on the opposing line end of the to be checked field, field_3 is in the opposing corner
    secondary_condition = jnp.logical_or(game_field.field_color[field_1 % 8, field_1 // 8 ] == FieldColor.WHITE,
    game_field.field_color[(field_2 % 8),field_2 // 8 ] == FieldColor.WHITE)

    return jax.lax.cond(game_field.field_color[field_3 % 8,field_3 // 8 ] == FieldColor.EMPTY,
        lambda secondary_condition: 0,
        lambda secondary_condition: jax.lax.cond(secondary_condition, 
            lambda _: -68, 
            lambda _: -80, 
            None),
        secondary_condition)

def css_calculate_bottom_right_score(game_field: Field, default_pos: int, difficulty: int):
    #alt singature: game_field: Field, ai_think_timer: int, default_pos: int, difficulty: int, y_pos: int, x_pos: int
    return_value = ((56,default_pos), False) # (Touple to be returned, Aborted)
    (horizontal_score, alternate_pos_horz) = css__f2d3_count_tiles_horizontally(game_field, 0, 7, default_pos, difficulty) #TODO check for correctness of 0,7
    
    jax.lax.cond(horizontal_score < 0,
        lambda _: ((horizontal_score, alternate_pos_horz), True),
        lambda _: return_value,
        None)

    (vertical_score, alternate_pos_vert) = css__f2d3_count_tiles_vertically(game_field, 7, 0, default_pos, difficulty) #TODO check for correctness of 7,0

    jax.lax.cond(jnp.logical_and(vertical_score < 0, return_value[1] == False),
        lambda _: ((vertical_score, alternate_pos_vert), True),
        lambda _: return_value,
        None)

    return return_value[0]

def css_calculate_top_left_score(game_field: Field, default_pos: int, difficulty: int):
    return_value = ((56,default_pos), False) # (Touple to be returned, Aborted)
    (horizontal_score, alternate_pos_horz) = css__f2d3_count_tiles_horizontally(game_field, 7, 0, default_pos, difficulty) #TODO check for correctness of 7,0

    jax.lax.cond(horizontal_score < 0,
        lambda _: ((horizontal_score, alternate_pos_horz), True),
        lambda _: return_value,
        None)

    (vertical_score, alternate_pos_vert) = css__f2d3_count_tiles_vertically(game_field, 0, 7, default_pos, difficulty) #TODO check for correctness of 0,7

    jax.lax.cond(jnp.logical_and(vertical_score < 0, return_value[1] == False),
        lambda _: ((vertical_score, alternate_pos_vert), True),
        lambda _: return_value,
        None)

    return return_value[0]

def css_calculate_top_right_score(game_field: Field, default_pos: int, difficulty: int):
    return_value = ((56,default_pos), False) # (Touple to be returned, Aborted)
    (horizontal_score, alternate_pos_horz) = css__f2d3_count_tiles_horizontally(game_field, 0, 0, default_pos, difficulty) #TODO check for correctness of 0,0
    
    jax.lax.cond(horizontal_score < 0,
        lambda _: ((horizontal_score, alternate_pos_horz), True),
        lambda _: return_value,
        None)

    (vertical_score, alternate_pos_vert) = css__f2d3_count_tiles_vertically(game_field, 7, 7, default_pos, difficulty) #TODO check for correctness of 7,7

    jax.lax.cond(jnp.logical_and(vertical_score < 0, return_value[1] == False),
        lambda _: ((vertical_score, alternate_pos_vert), True),
        lambda _: return_value,
        None)

    return return_value[0]

def css_calculate_bottom_left_score(game_field: Field, default_pos: int, difficulty: int):
    return_value = ((56,default_pos), False) # (Touple to be returned, Aborted)
    (horizontal_score, alternate_pos_horz) = css__f2d3_count_tiles_horizontally(game_field, 7, 7, default_pos, difficulty) #TODO check for correctness of 7,7
    
    jax.lax.cond(horizontal_score < 0,
        lambda _: ((horizontal_score, alternate_pos_horz), True),
        lambda _: return_value,
        None)

    (vertical_score, alternate_pos_vert) = css__f2d3_count_tiles_vertically(game_field, 0, 0, default_pos, difficulty) #TODO check for correctness of 0,0

    jax.lax.cond(jnp.logical_and(vertical_score < 0, return_value[1] == False),
        lambda _: ((vertical_score, alternate_pos_vert), True),
        lambda _: return_value,
        None)

    return return_value[0]

def css__f2d3_count_tiles_vertically(game_field: Field, y_pos:int, x_pos: int, default_pos: int, difficulty: int):
    empty_line = jnp.arange(8, dtype=jnp.int32)

    def get_field_color_tiles_vertically(i, y_pos, x_pos, game_field):
        return game_field.field_color[i, x_pos]

    vectorized_array_of_tiles = jax.vmap(get_field_color_tiles_vertically, in_axes=(0, None, None, None))
    array_of_tiles = Field(field_id=jnp.arange(8),
                           field_color=vectorized_array_of_tiles(empty_line, y_pos, x_pos, game_field)) #TODO check if flip is needed

    return css__f2d3_count_tiles_in_line(array_of_tiles, y_pos, default_pos, difficulty)

def css__f2d3_count_tiles_horizontally(game_field: Field, y_pos:int, x_pos: int, default_pos: int, difficulty: int):
    array_of_tiles = Field(
        field_id=jnp.arange(8),
        field_color=game_field.field_color[y_pos] #TODO check if flip is needed
    )


    return css__f2d3_count_tiles_in_line(array_of_tiles, x_pos, default_pos, difficulty)



def css__f2d3_count_tiles_in_line(array_of_tiles, pos: int, default_pos_combined: int, difficulty: int):
    return_value = ((-1, -1), False) # (Touple to be returned, Aborted) 

    return_value = jax.lax.cond(difficulty == 1,
        lambda _: ((0, -1), True),
        lambda _: return_value,
        None)

    reversed_array_of_tiles = Field(
        field_id=jnp.flip(array_of_tiles.field_id),
        field_color=jnp.flip(array_of_tiles.field_color)
    )

    left_state, left_pos_opt = css_sub_f5c1_count_tiles_in_line_descending(array_of_tiles, pos)
    right_state, right_pos_opt = css_sub_f5c1_count_tiles_in_line_descending(reversed_array_of_tiles,7 - pos)

    left_pos = jax.lax.cond(left_pos_opt != -1, 
        lambda _: left_pos_opt, 
        lambda _: default_pos_combined, 
        None)
    right_pos = jax.lax.cond(right_pos_opt != -1, 
        lambda _: 7-right_pos_opt, #TODO check for correctnes in assembly (reconverts right pos from flipped array to normal array)
        lambda _: left_pos,
        None) #TODO check in assembly if left_pos is the correct value

    combined_state = (left_state << 2) | right_state

    return_value = jax.lax.cond(jnp.logical_and(
        return_value[1] == False,
        jnp.logical_and(
            combined_state == 0b0101, 
            jnp.logical_or(
                reversed_array_of_tiles.field_color[0] != FieldColor.EMPTY, 
                reversed_array_of_tiles.field_color[7] != FieldColor.EMPTY))),
        lambda _: ((-40, right_pos), True), #TODO check why right_pos is used and if correct
        lambda _: return_value,
        None)

    return_value = jax.lax.cond(jnp.logical_and(
        return_value[1] == False,
        jnp.logical_and(
            jnp.logical_or(combined_state == 0b0111, combined_state == 0b1101),
            jnp.logical_or(right_pos == 7, right_pos == 0))),
        lambda _: ((-96, right_pos), True), #TODO check why right_pos is used and if correct (randomness?)
        lambda _: return_value,
        None)

    reverse_pos = 7 - pos
    flag = True
    condition_break = False


    #Simulate if-else structure without using nesting
    flag, condition_break= jax.lax.cond(jnp.logical_or(reverse_pos == 0, reverse_pos == 7),
        lambda _: (False, True),
        lambda _: (flag, False),
        None)

    flag, condition_break = jax.lax.cond(jnp.logical_and(condition_break == False, reverse_pos < 2), #TODO check if <> is correct
        lambda _: (True, True),
        lambda _: (flag, False),
        None)

    flag, condition_break = jax.lax.cond(jnp.logical_and(
        condition_break == False, 
        jnp.logical_or(
            reversed_array_of_tiles.field_color[reverse_pos - 1] != FieldColor.EMPTY, 
            reversed_array_of_tiles.field_color[reverse_pos - 2] == FieldColor.EMPTY)),
            lambda _: (True, True),
            lambda _: (flag, False),
            None)

    flag, condition_break, combined_state = jax.lax.cond(jnp.logical_and(condition_break == False, reversed_array_of_tiles.field_color[reverse_pos - 2] != FieldColor.WHITE),
        lambda _: (False, True, 18),
        lambda _: (flag, False, combined_state),
        None)


    conditions = jnp.logical_and.reduce(
        jnp.array(
            [flag,
            reverse_pos < 6,
            reversed_array_of_tiles.field_color[reverse_pos + 1] == FieldColor.EMPTY,
            reversed_array_of_tiles.field_color[reverse_pos + 2] != FieldColor.WHITE,
            reversed_array_of_tiles.field_color[reverse_pos + 2] != FieldColor.EMPTY,
        ]
    ))

    combined_state = jax.lax.cond(conditions,
        lambda _: 18,
        lambda _: combined_state,
        None
    )

    return_value = jax.lax.cond(jnp.logical_and(difficulty == 2, return_value[1] == False),
        lambda _: ((jnp.int32(__F7EC[combined_state]), right_pos), True),
        lambda _: return_value,
        None,
    )

    black_mask_low = 1 << reverse_pos
    black_mask_high = 1 << (7-reverse_pos)
    white_mask_low = 0
    white_mask_high = 0

    init_val = (reversed_array_of_tiles, black_mask_low, black_mask_high, white_mask_low,white_mask_high)


    def css_sub_f5c1_count_tiles_in_line_loop_mask_tiles(i, loop_vals):
        _, black_mask_low, black_mask_high, white_mask_low, white_mask_high = loop_vals

        black_mask_low, black_mask_high = jax.lax.cond( reversed_array_of_tiles.field_color[i] == FieldColor.BLACK,
            lambda _: (black_mask_low | 1 << i, black_mask_high | 1 << (7 - i)),
            lambda _: (black_mask_low, black_mask_high),
            None
        )

        white_mask_low, white_mask_high = jax.lax.cond( reversed_array_of_tiles.field_color[i] == FieldColor.WHITE,
            lambda _: (white_mask_low | 1 << i, white_mask_high | 1 << (7 - i)),
            lambda _: (white_mask_low, white_mask_high),
            None
        )

        return (loop_vals[0], black_mask_low, black_mask_high, white_mask_low, white_mask_high)

    _, black_mask_low, black_mask_high, white_mask_low, white_mask_high = jax.lax.fori_loop(0,8, css_sub_f5c1_count_tiles_in_line_loop_mask_tiles, init_val)

    init_val2 = (return_value, black_mask_low, black_mask_high,white_mask_low,white_mask_high)

    def css_sub_f5c1_count_tiles_in_line_loop_check_masks(i, loop_vals):
        #TODO can be vectorised
        return_value, black_mask_low, black_mask_high, white_mask_low, white_mask_high = loop_vals
        return_value = jax.lax.cond(
            jnp.logical_and(__F3BA[33-i] == white_mask_low, __F3DC[33-i] == black_mask_low),
            lambda _: ((jnp.int32(__F3FE[33-i]), white_mask_high),True),
            lambda _: return_value,
            None)
        
        return_value = jax.lax.cond(
            jnp.logical_and(return_value[1] == False, 
                jnp.logical_and(__F3BA[33-i] == white_mask_high, __F3DC[33-i] == black_mask_high)),
            lambda _: ((jnp.int32(__F3FE[33-i]), white_mask_high),True), #TODO check in Assembly for correctnes!
            lambda _: return_value,
            None)
        
        return (return_value, black_mask_low, black_mask_high, white_mask_low, white_mask_high)

    return_value, _, _, _, _ = jax.lax.cond(return_value[1] == False, 
                                            lambda init_val2: jax.lax.fori_loop(0,34, css_sub_f5c1_count_tiles_in_line_loop_check_masks, init_val2),
                                            lambda init_val2: (return_value, black_mask_low, black_mask_high, white_mask_low, white_mask_high),
                                            init_val2)

    return jax.lax.cond(return_value[1] == True,
        lambda _: return_value[0],
        lambda _: (jnp.int32(__F3FE[combined_state]), white_mask_high),
        None)
   
@jax.jit
#return a tuple of (state, position)
def css_sub_f5c1_count_tiles_in_line_descending(array_of_tiles, start_index: int) ->Tuple[int, int]:
    adjacent_index = start_index-1

    args = (array_of_tiles, adjacent_index)
    return jax.lax.cond(adjacent_index < 0,
    lambda args: (0b00, -1),
    lambda args: css_sub_f5c1_count_tiles_in_line_descending_not_first_tile(args),
    args
)

def css_sub_f5c1_count_tiles_in_line_descending_not_first_tile(args) ->Tuple[int, int]:
    array_of_tiles, adjacent_index = args

    return jax.lax.cond(
        array_of_tiles.field_color[adjacent_index] == FieldColor.EMPTY,
        lambda args: (0b01, adjacent_index),
        lambda args: jax.lax.cond(
            array_of_tiles.field_color[adjacent_index] == FieldColor.WHITE,
            lambda args: css_sub_f5c1_count_tiles_in_line_descending_handle_white(args),
            lambda args: css_sub_f5c1_count_tiles_in_line_descending_handle_black(args),
            args
        ),
        args
    )

def css_sub_f5c1_count_tiles_in_line_descending_handle_white(args) ->Tuple[int, int]:
    array_of_tiles, adjacent_index = args
    init_val = array_of_tiles, (-1, -1), adjacent_index-1

    output = jax.lax.fori_loop(
        0,
        adjacent_index,
        lambda i, init_val: css_sub_f5c1_count_tiles_in_line_descending_handle_white_loop(i, init_val),
        init_val
    )

    return jax.lax.cond(output[1][0] == -1,
        lambda _: (0b11, -1),
        lambda _: output[1],
        None
    )

def css_sub_f5c1_count_tiles_in_line_descending_handle_white_loop(i, loop_vals) ->Tuple[int, int]:
    array_of_tiles, touple, custom_iterator = loop_vals

    touple = jax.lax.cond(
        jnp.logical_or(array_of_tiles.field_color[custom_iterator] == FieldColor.WHITE, jnp.logical_or(touple[0] == 0b11, jnp.logical_or(touple[0] == 0b10, touple[0] == 0b00))),
        lambda loop_vals: touple , # interrrupt the loop if there is an empty space before a Black tile
        lambda loop_vals: jax.lax.cond(
            array_of_tiles.field_color[custom_iterator] == FieldColor.EMPTY,
            lambda loop_vals: (0b11, -1), # do nothing for white tiles
            lambda loop_vals: css_sub_f5c1_count_tiles_in_line_descending_handle_white_loop_black_tile(loop_vals), 
            loop_vals
        ),
        loop_vals
    )

    custom_iterator -= 1
    return (array_of_tiles, touple, custom_iterator)

def css_sub_f5c1_count_tiles_in_line_descending_handle_white_loop_black_tile(args) -> Tuple[int, int]:
    #guard against us being already in the last iteration
    _, _, custom_iterator = args
    return jax.lax.cond(custom_iterator != 0,
        lambda args: css_sub_f5c1_count_tiles_in_line_descending_handle_white_loop_black_tile_2(args),
        lambda args: (0b00, -1),   # Black found at end (possible valid reversal pattern)
        args
    )

def css_sub_f5c1_count_tiles_in_line_descending_handle_white_loop_black_tile_2(args) -> Tuple[int, int]:
    array_of_tiles, _, custom_iterator = args

    init_val = array_of_tiles, (-1, -1), custom_iterator

    output = jax.lax.fori_loop(
        0,
        custom_iterator+1,
        lambda i, init_val: css_sub_f5c1_count_tiles_in_line_descending_handle_white_loop_black_tile_loop(i, init_val),
        init_val
    )
    # if the next tile is black, increase the count of white tiles
    return jax.lax.cond(output[1][0] == -1,
        lambda _: (0b00, -1),
        lambda _: output[1],
        None
    )

def css_sub_f5c1_count_tiles_in_line_descending_handle_white_loop_black_tile_loop(i, loop_vals) :
    array_of_tiles, touple, custom_iterator_2 = loop_vals
    #jax.debug.print("custom_iterator_2: {}", custom_iterator_2)
    touple = jax.lax.cond(
        jnp.logical_or(array_of_tiles.field_color[custom_iterator_2] == FieldColor.BLACK, 
            jnp.logical_or(touple[0] == 0b10, 
                jnp.logical_or(touple[0] == 0b11, custom_iterator_2 < 0))), #TODO check why this is executed in base case
        # assure the interuption found in previuos iteration is not overwritten
        lambda _: touple,  # do nothing if we encounter a black tile, or if we already have an interruption
        lambda _: jax.lax.cond(
            array_of_tiles.field_color[custom_iterator_2] == FieldColor.WHITE,
            lambda _: (0b11, -1),  # interrupt the loop if we encounter a white tile
            lambda _: (0b10, -1),  # interrupt the loop if we encounter an empty tile
            None
        ),
        None
    )
    custom_iterator_2 -= 1
    return (array_of_tiles, touple, custom_iterator_2)

def css_sub_f5c1_count_tiles_in_line_descending_handle_black(args) -> Tuple[int, int]:
    array_of_tiles, adjacent_index = args
    init_val = array_of_tiles, (-1, -1), adjacent_index-1

    output = jax.lax.fori_loop(
        0,
        adjacent_index,
        lambda i, init_val: css_sub_f5c1_count_tiles_in_line_descending_handle_black_loop(i, init_val),
        init_val
    )
    return jax.lax.cond(output[1][0] == -1,
        lambda _: (0b00, -1),
        lambda _: output[1],
        None
    )

def css_sub_f5c1_count_tiles_in_line_descending_handle_black_loop(i, loop_vals):
    array_of_tiles, touple, custom_iterator = loop_vals
    touple = jax.lax.cond(
            jnp.logical_or(array_of_tiles.field_color[custom_iterator] == FieldColor.BLACK, 
                jnp.logical_or(touple[0] == 0b01, 
                    jnp.logical_or(touple[0] == 0b11, custom_iterator < 0))),
            # assure the interuption found in previuos iteration is not overwritten
        lambda _: touple, # do nothing if we encounter a black tile, or if we already have an interruption
        lambda _: jax.lax.cond(
            array_of_tiles.field_color[custom_iterator] == FieldColor.WHITE,
            lambda _: (0b11, -1), # interrupt the loop if we encounter a white tile
            lambda _: (0b01, custom_iterator), # interrupt the loop if we encounter an empty tile
            None
        ),
        None
    )

    custom_iterator -= 1
    return (array_of_tiles, touple, custom_iterator)


class JaxOthello(JaxEnvironment[OthelloState, OthelloObservation, OthelloInfo]):
    def __init__(self, frameskip: int = 0, reward_funcs: list[callable]=None):
        super().__init__()
        self.frameskip = frameskip + 1
        self.frame_stack_size = 4
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.action_set = self.action_set = [
            NOOP,
            PLACE,
            RIGHT,
            LEFT,
            UP,
            DOWN,
            UPLEFT,
            UPRIGHT,
            DOWNLEFT,
            DOWNRIGHT
        ]
        self.obs_size = 130


    def reset(self, key = [0,0]) -> OthelloState:
        """ Reset the game state to the initial state """
        field_color_init = jnp.full((8, 8), FieldColor.EMPTY.value, dtype=jnp.int32)
        field_color_init = field_color_init.at[3,3].set(FieldColor.BLACK.value)
        field_color_init = field_color_init.at[4,3].set(FieldColor.WHITE.value)
        field_color_init = field_color_init.at[3,4].set(FieldColor.WHITE.value)
        field_color_init = field_color_init.at[4,4].set(FieldColor.BLACK.value)
        
        #################### Testing
        
        # field_color_init = field_color_init.at[0,0].set(FieldColor.BLACK.value)
        # field_color_init = field_color_init.at[1,1].set(FieldColor.WHITE.value)
        # field_color_init = field_color_init.at[2,2].set(FieldColor.WHITE.value)
        # field_color_init = field_color_init.at[3,3].set(FieldColor.EMPTY.value)
        # field_color_init = field_color_init.at[4,4].set(FieldColor.WHITE.value)
        # field_color_init = field_color_init.at[5,5].set(FieldColor.WHITE.value)

        
        

        state = OthelloState(
            player_score = jnp.array(2).astype(jnp.int32),
            enemy_score = jnp.array(2).astype(jnp.int32),
            step_counter =jnp.array(0).astype(jnp.int32),
            field = Field(
                field_id = jnp.arange(64, dtype=jnp.int32).reshape((8,8)),
                field_color = field_color_init
            ),
            field_choice_player = jnp.array([7, 7], dtype=jnp.int32),
            difficulty = jnp.array(1).astype(jnp.int32),
            end_of_game_reached = False,
            # TODO Figure out why key has shape (2,)
            random_key = jax.random.PRNGKey(key[0])
        )
        initial_obs = self._get_observation(state)
        return initial_obs, state

    @partial(jax.jit, static_argnums=0)
    def step(self, state: OthelloState, action: chex.Array) -> Tuple[OthelloObservation, OthelloState, float, bool, OthelloInfo]:

        state = state._replace(step_counter=state.step_counter+1)

        decided, new_field_choice = has_player_decided_field(state.field_choice_player, action)  # 2D Array new_field_choice[i, j]

        state = state._replace(field_choice_player=new_field_choice)


        # first, it need to be checked if there is a valid place on the field the disc to be set
        _, valid_field = check_if_there_is_a_valid_choice(state, white_player=True)

        # check if the new_field_choice is a valid option
        valid_choice, new_state = jax.lax.cond(
            decided,
            lambda _: field_step(new_field_choice, state, True),
            lambda _: (False, state),
            operand=None
        )

        # now enemy step are required
        # for now - choose a random field if a given step by agent/human was valid
        def condition_fun(value):
            valid_choice, new_state, _ = value
            valid_choice = jnp.logical_not(valid_choice)
            return valid_choice

        def body_fun(value):
            valid_choice, state, key = value

            best_val = get_bot_move(state.field,state.difficulty, state.player_score,state.enemy_score,state.random_key)

            valid_choice, new_state = field_step(best_val, state, False)

            return jax.lax.cond(
                valid_choice,
                lambda _: (True, new_state, key),
                lambda _: (False, state, key),
                operand=None
            )

        _, valid_field_enemy = check_if_there_is_a_valid_choice(new_state, white_player=False)

        key = state.random_key
        initial_x_y = (False, new_state, key)
        _, final__step_state, _ = jax.lax.cond(
            jnp.logical_and(
                jnp.logical_or(valid_choice, jnp.logical_not(valid_field)),
                valid_field_enemy
            ), 
            lambda _: jax.lax.while_loop(condition_fun, body_fun, initial_x_y),
            lambda _: (valid_choice, new_state, key),
            operand=None
        )

        # check if game is ended
        def outer_loop(i, ended):
            def inner_loop(j, ended):
                ended = jax.lax.cond(
                    final__step_state.field.field_color[i,j] == FieldColor.EMPTY,
                    lambda _: False,
                    lambda x: x,
                    ended
                )
                return ended
            return jax.lax.fori_loop(0, FIELD_WIDTH, inner_loop, ended)
        has_game_ended = jax.lax.fori_loop(0, FIELD_HEIGHT, outer_loop, True)
        has_game_ended = jax.lax.cond(
            jnp.logical_or(final__step_state.player_score == 0, final__step_state.enemy_score == 0),
            lambda _: True,
            lambda x: x,
            has_game_ended
        )
        final__step_state = final__step_state._replace(end_of_game_reached=has_game_ended)

        done = self._get_done(final__step_state)
        env_reward = self._get_env_reward(state, final__step_state)
        all_rewards = self._get_all_reward(state, final__step_state)
        info = self._get_info(new_state, all_rewards)
        observation = self._get_observation(final__step_state)

        return observation, final__step_state, env_reward, done, info        

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: OthelloState, all_rewards: chex.Array) -> OthelloInfo:
        return OthelloInfo(time=state.step_counter, all_rewards=all_rewards)

    @partial(jax.jit, static_argnums=(0,))
    def _get_env_reward(self, previous_state: OthelloState, state: OthelloState):
        return (state.player_score - state.enemy_score) - (
            previous_state.player_score - previous_state.enemy_score
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: OthelloState):
        return OthelloObservation(
            field=state.field,
            enemy_score=state.enemy_score,
            player_score=state.player_score,
            field_choice_player=state.field_choice_player
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(self, previous_state: OthelloState, state: OthelloState):
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array(
            [reward_func(previous_state, state) for reward_func in self.reward_funcs]
        )
        return rewards

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: OthelloState) -> bool:
        return state.end_of_game_reached

    def get_action_space(self) -> jnp.ndarray:
        return jnp.array(self.action_set)

    def get_observation_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=2,
            shape=(FIELD_HEIGHT, FIELD_WIDTH),
            dtype=jnp.uint8,
        )



def load_sprites():
    """Load all sprites required for Pong rendering."""
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Load sprites
    player = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/othello/player_white_disc.npy"), transpose=True)
    enemy = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/othello/enemy_black_disc.npy"), transpose=True)

    bg = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/othello/othello_background.npy"), transpose=True)

    # TODO: get a correctly sized background image / resize the saved image..
    #bg = jax.image.resize(bg, (WIDTH, HEIGHT, 4), method='bicubic')

    # Convert all sprites to the expected format (add frame dimension)
    SPRITE_BG = jnp.expand_dims(bg, axis=0)
    SPRITE_PLAYER = jnp.expand_dims(player, axis=0)
    SPRITE_ENEMY = jnp.expand_dims(enemy, axis=0)

    # Load digits for scores
    PLAYER_DIGIT_SPRITES = aj.load_and_pad_digits(
        os.path.join(MODULE_DIR, "sprites/othello/number_{}_player.npy"),
        num_chars=10,
    )
    ENEMY_DIGIT_SPRITES = aj.load_and_pad_digits(
        os.path.join(MODULE_DIR, "sprites/othello/number_{}_enemy.npy"),
        num_chars=10,
    )

    return (
        SPRITE_BG,
        SPRITE_PLAYER,
        SPRITE_ENEMY,
        PLAYER_DIGIT_SPRITES,
        ENEMY_DIGIT_SPRITES
    )


@jax.jit
def render_point_of_disc(id):
    return jnp.array([18 + 16 * id[1], 22 + 22 * id[0]], dtype=jnp.int32)


class OthelloRenderer(AtraJaxisRenderer):
    def __init__(self):
        (
            self.SPRITE_BG,
            self.SPRITE_PLAYER,
            self.SPRITE_ENEMY,
            self.PLAYER_DIGIT_SPRITES,
            self.ENEMY_DIGIT_SPRITES,
        ) = load_sprites()

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        # Create empty raster with CORRECT orientation for atraJaxis framework
        # Note: For pygame, the raster is expected to be (width, height, channels)
        # where width corresponds to the horizontal dimension of the screen
        raster = jnp.zeros((WIDTH, HEIGHT, 3))

        # Render Background - (0, 0) is top-left corner
        frame_bg = aj.get_sprite_frame(self.SPRITE_BG, 0)
        raster = aj.render_at(raster, 0, 0, frame_bg)

        # disc sprites
        frame_player = aj.get_sprite_frame(self.SPRITE_PLAYER, 0)
        frame_enemy = aj.get_sprite_frame(self.SPRITE_ENEMY, 0)
        
        # Render all fixed discs
        def set_discs_to_the_raster(raster, field_color): 
            def outer_loop(i, carry):
                def inner_loop(j, carry):
                    raster = carry
                    color = field_color[i, j]
                    render_point = render_point_of_disc(jnp.array([i,j], dtype=jnp.int32))

                    return jax.lax.cond(
                        color == FieldColor.EMPTY, 
                        lambda x: raster,
                        lambda x: jax.lax.cond(
                            color == FieldColor.WHITE,
                            lambda x: aj.render_at(raster, render_point[0], render_point[1], frame_player),
                            lambda x: aj.render_at(raster, render_point[0], render_point[1], frame_enemy),
                            x
                        ),
                        color
                    )

                return jax.lax.fori_loop(0, FIELD_HEIGHT, inner_loop, carry)

            current_raster = raster
            return jax.lax.fori_loop(0, FIELD_WIDTH, outer_loop, current_raster)
        raster = set_discs_to_the_raster(raster, state.field.field_color)

        # rendering the disc in flipping modus to show where the current disc is 
        # for better orientation
        current_player_choice = render_point_of_disc(state.field_choice_player)
        raster = jax.lax.cond(
            jnp.logical_and(
                jnp.logical_not(state.end_of_game_reached),
                jnp.logical_and(
                    (state.step_counter % 2) == 0,
                    jnp.logical_and(
                        state.enemy_score != 0,
                        state.player_score != 0
                    )
                )
            ),
            lambda x: aj.render_at(x, current_player_choice[0], current_player_choice[1], frame_player),
            lambda x: raster,
            raster
        ) 


        # rendering scores
        first_digit_player_score = state.player_score % 10
        second_digit_player_score = state.player_score // 10
        first_digit_enemy_score = state.enemy_score % 10
        second_digit_enemy_score = state.enemy_score // 10

        digit_render_y = 2
        first_digit_player_x = 17 + 16 * 1
        second_digit_player_x = 17 + 16 * 0
        first_digit_enemy_x = 17 + 16 * 6
        second_digit_enemy_x = 17 + 16 * 5

        frame_player_digit = aj.get_sprite_frame(self.PLAYER_DIGIT_SPRITES, first_digit_player_score)
        raster = aj.render_at(raster, first_digit_player_x, digit_render_y, frame_player_digit)
        frame_player_digit = aj.get_sprite_frame(self.PLAYER_DIGIT_SPRITES, second_digit_player_score)
        raster = jax.lax.cond(
            second_digit_player_score == 0,
            lambda _: raster,
            lambda _: aj.render_at(raster, second_digit_player_x, digit_render_y, frame_player_digit),
            operand=None
        )
        frame_player_digit = aj.get_sprite_frame(self.ENEMY_DIGIT_SPRITES, first_digit_enemy_score)
        raster = aj.render_at(raster, first_digit_enemy_x, digit_render_y, frame_player_digit)
        frame_player_digit = aj.get_sprite_frame(self.ENEMY_DIGIT_SPRITES, second_digit_enemy_score)
        raster = jax.lax.cond(
            second_digit_enemy_score == 0,
            lambda _: raster,
            lambda _: aj.render_at(raster, second_digit_enemy_x, digit_render_y, frame_player_digit),
            operand=None
        )

        return raster


if __name__ == "__main__":
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Othello Game")

    game = JaxOthello(frameskip=1)

    # Create the JAX renderer
    renderer = OthelloRenderer()

    # get jitted functions
    jitted_step = jax.jit(game.step)
    jitted_reset = jax.jit(game.reset)

    obs, curr_state = jitted_reset()

    # Game Loop
    running = True
    move_delay = 0.15
    last_move_time = 0

    while running:
        now = time.time()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False     

        if (now - last_move_time) > move_delay:
            action = get_human_action()
            obs, curr_state, reward, done, info = jitted_step(curr_state, action)
            last_move_time = now
            
        # Render and display
        raster = renderer.render(curr_state)
        aj.update_pygame(screen, raster, 3, WIDTH, HEIGHT)

    pygame.quit()