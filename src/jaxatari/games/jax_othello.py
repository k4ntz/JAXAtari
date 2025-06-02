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

# index of cells first covers x-axis, then y-axis (a = 0 -> (0,0), a = 1 -> (0,1), a = 8 -> (1,0), ..., a = 63 -> (7,7))
STRATEGIC_TILE_SCORE_CASES = 
[0,1,1,1,1,1,1,2,
 6,3,4,4,4,4,5,11,
 6,7,8,9,9,8,10,11,
 6,7,9,9,9,9,10,11,
 6,7,8,9,9,8,10,11,
 6,7,8,9,9,8,10,11,
 6,12,13,13,13,13,14,11,
 15,16,16,16,16,16,16,18
]



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
def get_bot_move(game_field: Field, difficulty: chex.Array, player_score: chex.Array, enemy_score: chex.Array, random_key: int)-> jnp.array:
    game_score = player_score+enemy_score
    list_of_all_mov
    
    es = jnp.arange(64)

    #calculate flipped tiles for all possible moves and adjust score based on game stage
    vectorized_compute_score_of_tiles = jax.vmap(compute_score_of_tiles, in_axes=(0, None, None))
    list_of_all_move_values = vectorized_compute_score_of_tiles(list_of_all_moves, game_field, game_score)

    #Calculate the strategic value (score) of the current_square itself

    #Re-evaluate or combine with a secondary related square's score

    #Combine the score from flipped pieces and the strategic tile score

    #Randomly choose one of the best moves    
    random_chosen_max_index = random_max_index(list_of_all_move_values,random_key)

    
    return jnp.array([jnp.floor_divide(random_chosen_max_index, 8), jnp.mod(random_chosen_max_index, 8)])

def compute_score_of_tiles(i, game_field, game_score):
    # Decode tile position
    tile_y = jnp.floor_divide(i, 8)
    tile_x = jnp.mod(i, 8)


    # If tile is already taken, set invalid move (return very low score)
    score_of_tile = jax.lax.cond(
        game_field.field_color[tile_y, tile_x] != FieldColor.EMPTY,
        lambda _: -2147483648,  
        lambda _: compute_score_of_tile_1(tile_y, tile_x,game_field, game_score),  
        None
    )
    return score_of_tile

@jax.jit
def compute_score_of_tile_1(tile_y, tile_x, game_field, game_score):
    #compute flipped tiles by each direction
    list_of_all_directions = jnp.arange(8)
    vectorised_flipped_tiles_by_direction = jax.vmap(compute_flipped_tiles_by_direction,in_axes=(0, None, None, None))
    flipped_tiles_by_direction = vectorised_flipped_tiles_by_direction(list_of_all_directions, tile_y, tile_x, game_field)

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
    return score_of_tile
    
# array size fixed at 64!!
@jax.jit
def random_max_index(array, key:int):
    max_value:int  = jnp.max(array)
    max_value_count:int = 0
    max_values = jnp.zeros_like(array)
    index:int = 0
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
def compute_flipped_tiles_by_direction(i, tile_y: int, tile_x: int, game_field: Field):
    args = (tile_y,tile_x,game_field)

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

@jax.jit   
def compute_flipped_tiles_top(input) -> int:
    # Returns the number of tiles to be flipped (only flipped tiles, same color tiles are disregarded)

    #checks wether the first element in the direction of look-up is invalid, becasause its already taken by bot, sets it to nan, to prevent while loop from running
    args = jax.lax.cond(
        input[2].field_color[input[0]-1][input[1]] == FieldColor.BLACK,
        lambda input: (input[0] - 1, input[1], input[2], jnp.nan, 0),
        lambda input: (input[0] - 1, input[1], input[2], 0.0, 0),
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
            lambda args: (-2, args[1], args[2], args[3],  args[4]),#if field is empthy, no further tiles can be flipped use set y to -2 to exit next loop iteration
            lambda args: (jax.lax.cond(
                args[2].field_color[args[0], args[1]] == FieldColor.BLACK,  
                lambda args: (args[0]-1, args[1], args[2], args[3] + args[4], 0),
                lambda args: (args[0]-1, args[1], args[2], args[3], args[4] + 1),
                args
                )),
            args
        )

    return jax.lax.cond(
        input[0] < 0,
        lambda args: jnp.int32(0.0),
        lambda args: jnp.int32(jax.lax.while_loop(while_cond, while_body, args)[3]),  
        args
    )
@jax.jit
def compute_flipped_tiles_rigth(input) -> int:
    # Returns the number of tiles to be flipped (only flipped tiles, same color tiles are disregarded)

    #checks wether the first element in the direction of look-up is invalid, becasause its already taken by bot, sets it to nan, to prevent while loop from running
    args = jax.lax.cond(
        input[2].field_color[input[0]][input[1]+1] == FieldColor.BLACK,
        lambda input: (input[0], input[1]+1, input[2], jnp.nan, 0),
        lambda input: (input[0], input[1]+1, input[2], 0.0, 0),
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
            lambda args: (-2, args[1], args[2], args[3],  args[4]),#if field is empthy, no further tiles can be flipped use set y to -2 to exit next loop iteration
            lambda args: (jax.lax.cond(
                args[2].field_color[args[0], args[1]] == FieldColor.BLACK,  
                lambda args: (args[0], args[1]+1, args[2], args[3] + args[4], 0),
                lambda args: (args[0], args[1]+1, args[2], args[3], args[4] + 1),
                args
                )),
            args
        )

    return jax.lax.cond(
        input[1] > 7,
        lambda args: jnp.int32(0.0),
        lambda args: jnp.int32(jax.lax.while_loop(while_cond, while_body, args)[3]),  
        args
    )
@jax.jit
def compute_flipped_tiles_bottom(input) -> int:
    # Returns the number of tiles to be flipped (only flipped tiles, same color tiles are disregarded)

    #checks wether the first element in the direction of look-up is invalid, becasause its already taken by bot, sets it to nan, to prevent while loop from running
    args = jax.lax.cond(
        input[2].field_color[input[0]+1][input[1]] == FieldColor.BLACK,
        lambda input: (input[0] + 1, input[1], input[2], jnp.nan, 0),
        lambda input: (input[0] + 1, input[1], input[2], 0.0, 0),
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
            lambda args: (-2, args[1], args[2], args[3],  args[4]),#if field is empthy, no further tiles can be flipped use set y to -2 to exit next loop iteration
            lambda args: (jax.lax.cond(
                args[2].field_color[args[0], args[1]] == FieldColor.BLACK,  
                lambda args: (args[0]+1, args[1], args[2], args[3] + args[4], 0),
                lambda args: (args[0]+1, args[1], args[2], args[3], args[4] + 1),
                args
                )),
            args
        )

    return jax.lax.cond(
        input[0] > 7,
        lambda args: jnp.int32(0.0),
        lambda args: jnp.int32(jax.lax.while_loop(while_cond, while_body, args)[3]),  
        args
    )
@jax.jit
def compute_flipped_tiles_left(input) -> int:
    # Returns the number of tiles to be flipped (only flipped tiles, same color tiles are disregarded)

    #checks wether the first element in the direction of look-up is invalid, becasause its already taken by bot, sets it to nan, to prevent while loop from running
    args = jax.lax.cond(
        input[2].field_color[input[0]][input[1]-1] == FieldColor.BLACK,
        lambda input: (input[0], input[1]-1, input[2], jnp.nan, 0),
        lambda input: (input[0], input[1]-1, input[2], 0.0, 0),
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
            lambda args: (-2, args[1], args[2], args[3],  args[4]),#if field is empthy, no further tiles can be flipped use set y to -2 to exit next loop iteration
            lambda args: (jax.lax.cond(
                args[2].field_color[args[0], args[1]] == FieldColor.BLACK,  
                lambda args: (args[0], args[1]-1, args[2], args[3] + args[4], 0),
                lambda args: (args[0], args[1]-1, args[2], args[3], args[4] + 1),
                args
                )),
            args
        )

    return jax.lax.cond(
        input[0] > 7,        
        lambda args: jnp.int32(0.0),
        lambda args: jnp.int32(jax.lax.while_loop(while_cond, while_body, args)[3]),  
        args
    )
@jax.jit
def compute_flipped_tiles_top_rigth(input) -> int:
    # Returns the number of tiles to be flipped (only flipped tiles, same color tiles are disregarded)

    #checks wether the first element in the direction of look-up is invalid, becasause its already taken by bot, sets it to nan, to prevent while loop from running
    args = jax.lax.cond(
        input[2].field_color[input[0]-1][input[1]+1] == FieldColor.BLACK,
        lambda input: (input[0]-1, input[1]+1, input[2], jnp.nan, 0),
        lambda input: (input[0]-1, input[1]+1, input[2], 0.0, 0),
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
            lambda args: (-2, args[1], args[2], args[3],  args[4]),#if field is empthy, no further tiles can be flipped use set y to -2 to exit next loop iteration
            lambda args: (jax.lax.cond(
                args[2].field_color[args[0], args[1]] == FieldColor.BLACK,  
                lambda args: (args[0]-1, args[1]+1, args[2], args[3] + args[4], 0),
                lambda args: (args[0]-1, args[1]+1, args[2], args[3], args[4] + 1),
                args
                )),
            args
        )

    return jax.lax.cond(
        jnp.logical_and(input[0] < 0, input[1]>7),
        lambda args: jnp.int32(0.0),
        lambda args: jnp.int32(jax.lax.while_loop(while_cond, while_body, args)[3]),  
        args
    )
@jax.jit
def compute_flipped_tiles_bottom_rigth(input) -> int:
    # Returns the number of tiles to be flipped (only flipped tiles, same color tiles are disregarded)

    #checks wether the first element in the direction of look-up is invalid, becasause its already taken by bot, sets it to nan, to prevent while loop from running
    args = jax.lax.cond(
        input[2].field_color[input[0]+1][input[1]+1] == FieldColor.BLACK,
        lambda input: (input[0]+1, input[1]+1, input[2], jnp.nan, 0),
        lambda input: (input[0]+1, input[1]+1, input[2], 0.0, 0),
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
            lambda args: (-2, args[1], args[2], args[3],  args[4]),#if field is empthy, no further tiles can be flipped use set y to -2 to exit next loop iteration
            lambda args: (jax.lax.cond(
                args[2].field_color[args[0], args[1]] == FieldColor.BLACK,  
                lambda args: (args[0]+1, args[1]+1, args[2], args[3] + args[4], 0),
                lambda args: (args[0]+1, args[1]+1, args[2], args[3], args[4] + 1),
                args
                )),
            args
        )

    return jax.lax.cond(
        jnp.logical_and(input[0] > 7, input[1]>7),
        lambda args: jnp.int32(0.0),
        lambda args: jnp.int32(jax.lax.while_loop(while_cond, while_body, args)[3]),  
        args
    )
@jax.jit
def compute_flipped_tiles_bottom_left(input) -> int:
    # Returns the number of tiles to be flipped (only flipped tiles, same color tiles are disregarded)

    #checks wether the first element in the direction of look-up is invalid, becasause its already taken by bot, sets it to nan, to prevent while loop from running
    args = jax.lax.cond(
        input[2].field_color[input[0]+1][input[1]-1] == FieldColor.BLACK,
        lambda input: (input[0]+1, input[1]-1, input[2], jnp.nan, 0),
        lambda input: (input[0]+1, input[1]-1, input[2], 0.0, 0),
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
            lambda args: (-2, args[1], args[2], args[3],  args[4]),#if field is empthy, no further tiles can be flipped use set y to -2 to exit next loop iteration
            lambda args: (jax.lax.cond(
                args[2].field_color[args[0], args[1]] == FieldColor.BLACK,  
                lambda args: (args[0]+1, args[1]-1, args[2], args[3] + args[4], 0),
                lambda args: (args[0]+1, args[1]-1, args[2], args[3], args[4] + 1),
                args
                )),
            args
        )

    return jax.lax.cond(
        jnp.logical_and(input[0] <0, input[1]>7),
        lambda args: jnp.int32(0.0),
        lambda args: jnp.int32(jax.lax.while_loop(while_cond, while_body, args)[3]),  
        args
    )
@jax.jit
def compute_flipped_tiles_top_left(input) -> int:
    # Returns the number of tiles to be flipped (only flipped tiles, same color tiles are disregarded)

    #checks wether the first element in the direction of look-up is invalid, becasause its already taken by bot, sets it to nan, to prevent while loop from running
    args = jax.lax.cond(
        input[2].field_color[input[0]-1][input[1]-1] == FieldColor.BLACK,
        lambda input: (input[0]-1, input[1]-1, input[2], jnp.nan, 0),
        lambda input: (input[0]-1, input[1]-1, input[2], 0.0, 0),
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
            lambda args: (-2, args[1], args[2], args[3],  args[4]),#if field is empthy, no further tiles can be flipped use set y to -2 to exit next loop iteration
            lambda args: (jax.lax.cond(
                args[2].field_color[args[0], args[1]] == FieldColor.BLACK,  
                lambda args: (args[0]-1, args[1]-1, args[2], args[3] + args[4], 0),
                lambda args: (args[0]-1, args[1]-1, args[2], args[3], args[4] + 1),
                args
                )),
            args
        )

    return jax.lax.cond(
        jnp.logical_and(input[0]<0, input[1]<0),
        lambda args: jnp.int32(0.0),
        lambda args: jnp.int32(jax.lax.while_loop(while_cond, while_body, args)[3]),  
        args
    )

@jax.jit
def calculate_strategic_tile_score(tile_index: int, game_field: Field, unknown_arg, difficulty: int):
    #determine the position of the tile within the game field
    args = (tile_index // 8, tile_index % 8, game_field, unknown_arg, difficulty)
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

    return jax.lax.switch(tile_index, branches, args)

    def compute_strategic_score_top_left(args):
        tile_y, tile_x, game_field, unknown_arg, difficulty = args
        #TODO implement the strategic score for the top left corner

        return ((None, None), None)

    def compute_strategic_score_top(args):
        tile_y, tile_x, game_field, unknown_arg, difficulty = args
        #TODO implement the strategic score for the top side

        return ((None, None), None)

    def compute_strategic_score_top_right(args):
        tile_y, tile_x, game_field, unknown_arg, difficulty = args
        #TODO implement the strategic score for the top right corner

        return ((None, None), None)
    
    def compute_strategic_score_top_left_inner(args):
        tile_y, tile_x, game_field, unknown_arg, difficulty = args
        #TODO implement the strategic score for the top left inner corner

        return ((None, None), None)
    
    def compute_strategic_score_top_inner(args):
        #Strategic score for top inner field is influenced by the field above it
        tile_y, tile_x, game_field, _, _ = args 
        return ((css_check_tile_up(tile_y, tile_x, game_field), None), False)

    def compute_strategic_score_top_right_inner(args):
        tile_y, tile_x, game_field, unknown_arg, difficulty = args
        #TODO implement the strategic score for the top right inner corner

        return ((None, None), None)

    def compute_strategic_score_left(args):
        tile_y, tile_x, game_field, unknown_arg, difficulty = args
        #TODO implement the strategic score for the left side

        return ((None, None), None)

    def compute_strategic_score_left_inner(args):
        #Strategic score for left inner field is influenced by the field to the left of it
        tile_y, tile_x, game_field, _, _ = args
        return ((css_check_tile_left(tile_y, tile_x, game_field), None), False)

    def compute_strategic_score_center_corner(args):
        # Center corner has a fixed score of 4, no further explanation needed
        return ((4, None), False)

    def compute_strategic_score_center(args):
        #center has a fixed score of 0, no further evaluation needed
        return (0, None), False

    def compute_strategic_score_right_inner(args):
        #Strategic score for right inner field is influenced by the field to the right of it
        tile_y, tile_x, game_field, unknown_arg, difficulty = args
        return ((css_check_tile_right(tile_y, tile_x, game_field), None), False)

    def compute_strategic_score_right(args):
        tile_y, tile_x, game_field, unknown_arg, difficulty = args
        #TODO implement the strategic score for the right side

        return ((None, None), None)

    def compute_strategic_score_bottom_left_inner(args):
        tile_y, tile_x, game_field, unknown_arg, difficulty = args
        #TODO implement the strategic score for the bottom left inner corner

        return ((None, None), None)

    def compute_strategic_score_bottom_inner(args):
        #Strategic score for bottom inner field is influenced by the field below it
        tile_y, tile_x, game_field, _, _ = args
        return ((css_check_tile_down(tile_y, tile_x, game_field), None), False)

    def compute_strategic_score_bottom_right_inner(args):
        tile_y, tile_x, game_field, unknown_arg, difficulty = args
        #TODO implement the strategic score for the bottom right inner corner

        return ((None, None), None)

    def compute_strategic_score_bottom_left(args):
        tile_y, tile_x, game_field, unknown_arg, difficulty = args
        #TODO implement the strategic score for the bottom left corner

        return ((None, None), None)

    def compute_strategic_score_bottom(args):
        tile_y, tile_x, game_field, unknown_arg, difficulty = args
        #TODO implement the strategic score for the bottom side

        return ((None, None), None)

    def compute_strategic_score_bottom_right(args):
        tile_y, tile_x, game_field, unknown_arg, difficulty = args
        #TODO implement the strategic score for the bottom right corner

        return ((None, None), None)

def css_check_tile_down(tile_y: int, tile_x: int, game_field: Field) -> int:
    # If field below has the same color as the current tile, return 4, else return -8
    return jax.lax.cond(
        game_field.field_color[tile_y + 1, tile_x] == game_field.field_color[tile_y , tile_x],
        lambda _: 4,
        lambda _: -8,
        None)

def css_check_tile_up(tile_y: int, tile_x: int, game_field: Field) -> int:
    # If field above has the same color as the current tile, return 4, else return -8
    return jax.lax.cond(
        game_field.field_color[tile_y - 1, tile_x] == game_field.field_color[tile_y , tile_x],
        lambda _: 4,
        lambda _: -8,
        None)

def css_check_tile_left(tile_y: int, tile_x: int, game_field: Field) -> int:
    # If field to the left has the same color as the current tile, return 4, else return -8
    return jax.lax.cond(
        game_field.field_color[tile_y, tile_x - 1] == game_field.field_color[tile_y , tile_x],
        lambda _: 4,
        lambda _: -8,
        None)

def css_check_tile_right(tile_y: int, tile_x: int, game_field: Field) -> int:
    # If field to the right has the same color as the current tile, return 4, else return -8
    return jax.lax.cond(
        game_field.field_color[tile_y, tile_x + 1] == game_field.field_color[tile_y , tile_x],
        lambda _: 4,
        lambda _: -8,
        None)

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