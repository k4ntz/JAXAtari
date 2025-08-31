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

from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as jr
from jaxatari.environment import JaxEnvironment
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action


class OthelloConstants(NamedTuple):
    # attribution of cases for the strategic tile score //starting at bottom right, then going left und and then up
    STRATEGIC_TILE_SCORE_CASES:  chex.Array = jnp.array([
    17, 16, 16, 16, 16, 16, 16, 15,
    11, 14, 13, 13, 13, 13, 12, 6,
    11, 10, 8, 9, 9, 8, 7, 6,
    11, 10, 9, 9, 9, 9, 7, 6,
    11, 10, 9, 9, 9, 9, 7, 6,
    11, 10, 8, 9, 9, 8, 7, 6,
    11, 5, 4, 4, 4, 4, 3, 6,
    2, 1, 1, 1, 1, 1, 1, 0
    ])

    # stores patterns used in the strategic tile score for stability of white lines
    F3BA = jnp.array([
        0x60, 0x40, 0x42, 0x40, 0x00, 0x00, 0x00, 0x46, 0x46, 0x44, 0x04, 0x08, 0x0c, 0x0a, 0x08,
        0x04, 0x10, 0x14, 0xbe, 0x9e, 0x02, 0x02, 0x02, 0x12, 0x48, 0x28, 0x10, 0x08, 0x18, 0x38,
        0x40, 0x00, 0x02, 0x02
    ], dtype=jnp.uint8)

    # stores patterns used in the strategic tile score for stability of black lines
    F3DC = jnp.array([
        0x14, 0x28, 0x28, 0x2c, 0x46, 0x44, 0x40, 0x20, 0x08, 0x20, 0x60, 0x40, 0x40, 0x40, 0x42,
        0x40, 0x40, 0x40, 0x40, 0x60, 0x20, 0x28, 0x2c, 0x24, 0x32, 0x12, 0x4c, 0xf2, 0xe2, 0xc2,
        0x02, 0xbe, 0x18, 0x48
    ], dtype=jnp.uint8)

    # stores strategic tile scores for certain patterns of black and white configurations (used when only one side of the line is considered)
    F3FE = jnp.array([
        0x30, 0x30, 0x30, 0x30, 0xc0, 0xc0, 0xc0, 0x30, 0x30, 0x30, 0xbb, 0xbb, 0xbb, 0xbb, 0xbb,
        0xbb, 0xbb, 0xbb, 0x60, 0x60, 0x40, 0x30, 0x30, 0x50, 0xe0, 0xbb, 0xbb, 0xd0, 0xd0, 0xd0,
        0xd0, 0xd8, 0xf0, 0xf0
    ], dtype=jnp.uint8)

    # stores strategic tile scores for certain patterns of of whole lines 
    F7EC = jnp.array([
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

    # Game Environment
    HEIGHT = 210
    WIDTH = 160
    FIELD_WIDTH = 8
    FIELD_HEIGHT = 8

    # Pygame window dimensions
    WINDOW_HEIGHT = 210 * 3
    WINDOW_WIDTH = 160 * 3

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


# Describes the possible configurations of one individual field (Not Taken, Player and Enemy)
class FieldColor(enum.IntEnum):
    EMPTY = 0
    WHITE = 1
    BLACK = 2

# Describes the structure of the game field, each individual field has an ID and a color, the id enumerated from the left top the right bottom with 0-63
class Field(NamedTuple):
    field_id: chex.Array
    field_color: chex.Array

# Basis State of an Othello game
class OthelloState(NamedTuple):
    player_score: chex.Array #Stores the number of disks owned by player, used as 0d int
    enemy_score: chex.Array #Stores the number of disks owned by enemy, used as 0d int
    step_counter: chex.Array #Stores the number of steps passed in the game, used as 0d int
    field: Field #Stores the current state of the game board
    field_choice_player: chex.Array #Stores the currently selected disk for the player to place, used as 1d int array with shape (2,) and (y,x)
    difficulty: chex.Array #Stores the selected difficulty level, currently the game supports 1-3, but not 4, since this would be a multiagent game
    end_of_game_reached: chex.Array #Used to check if the game has ended to reset, only true for one state and afterwards resets with a new field to false
    random_key: chex.Array #Stores a random key for random decision used as 0d int

class OthelloObservation(NamedTuple):
    field: Field
    player_score: jnp.ndarray
    enemy_score: jnp.ndarray
    field_choice_player: jnp.ndarray

class OthelloInfo(NamedTuple):
    time: jnp.ndarray
    all_rewards: chex.Array





class JaxOthello(JaxEnvironment[OthelloState, OthelloObservation, OthelloInfo, OthelloConstants ]):
    def __init__(self, consts: OthelloConstants = None, frameskip: int = 0, reward_funcs: list[callable]=None):
        consts = consts or OthelloConstants()
        super().__init__(consts)
        self.frameskip = frameskip + 1
        self.frame_stack_size = 4
        
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.action_set = self.action_set = [
            self.consts.NOOP,
            self.consts.PLACE,
            self.consts.RIGHT,
            self.consts.LEFT,
            self.consts.UP,
            self.consts.DOWN,
            self.consts.UPLEFT,
            self.consts.UPRIGHT,
            self.consts.DOWNLEFT,
            self.consts.DOWNRIGHT
        ]
        self.obs_size = 130

    @partial(jax.jit, static_argnums=(0,))
    def has_player_decided_field(self, field_choice_player, action: chex.Array):
        # field_choice_player: the current positioning of the disc -> it is not jet placed down
        # action may include the action FIRE to place the disc down

        is_place = jnp.equal(action, self.consts.PLACE)
        is_up = jnp.equal(action, self.consts.UP)
        is_right = jnp.equal(action, self.consts.RIGHT)
        is_down = jnp.equal(action, self.consts.DOWN)
        is_left = jnp.equal(action, self.consts.LEFT)
        is_upleft = jnp.equal(action, self.consts.UPLEFT)
        is_upright = jnp.equal(action, self.consts.UPRIGHT)
        is_downleft = jnp.equal(action, self.consts.DOWNLEFT)
        is_downright = jnp.equal(action, self.consts.DOWNRIGHT)

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


    @partial(jax.jit, static_argnums=(0,))
    def field_step(self, field_choice, curr_state, white_player) -> Tuple[bool, OthelloState]:
        """
        Executes a single move in the Othello game.
        Given the chosen board position (field_choice), the current game state 
        (curr_state), and the active player's color (white_player), this function:
        - Determines if the chosen position is valid according to Othello rules.
        - Flips opponent discs along all valid directions (horizontal, vertical, diagonal).
        - Updates the board state, player scores, and last move.

        Args:
            curr_state (OthelloState): The current game state, including the board.
            white_player (bool): True if checking moves for the white player, 
                                 False if for the black player.

        Returns:
            tuple ([bool, OthelloState]):
                - bool: whether the move was valid and discs were flipped.
                - OthelloState: the updated game state after the move.
        """
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
            new_state, discs_flippable, _, _ = jax.lax.fori_loop(0, x, loop_horizontal_and_vertical_line_to_flip_discs, (curr_state, False, False, self.consts.FLIP_UP_SIDE))
            # in case, that the the discs aren't flippable (== False) reverse the state back
            new_state = jax.lax.cond(discs_flippable, lambda _: new_state, lambda _: curr_state, operand=None)
            valid_choice = valid_choice | discs_flippable
            curr_state = new_state

            # flip disc on the down side
            new_state, discs_flippable, _, _ = jax.lax.fori_loop(0, self.consts.FIELD_HEIGHT - x - 1, loop_horizontal_and_vertical_line_to_flip_discs, (curr_state, False, False, self.consts.FLIP_DOWN_SIDE))
            new_state = jax.lax.cond(discs_flippable, lambda _: new_state, lambda _: curr_state, operand=None)
            valid_choice = valid_choice | discs_flippable
            curr_state = new_state

            # flip disc on the right side
            new_state, discs_flippable, _, _ = jax.lax.fori_loop(0, self.consts.FIELD_WIDTH - y - 1, loop_horizontal_and_vertical_line_to_flip_discs, (curr_state, False, False, self.consts.FLIP_RIGHT_SIDE))
            new_state = jax.lax.cond(discs_flippable, lambda _: new_state, lambda _: curr_state, operand=None)
            valid_choice = valid_choice | discs_flippable
            curr_state = new_state

            # flip disc on the left side
            new_state, discs_flippable, _, _ = jax.lax.fori_loop(0, y, loop_horizontal_and_vertical_line_to_flip_discs, (curr_state, False, False, self.consts.FLIP_LEFT_SIDE))
            new_state = jax.lax.cond(discs_flippable, lambda _: new_state, lambda _: curr_state, operand=None)
            valid_choice = valid_choice | discs_flippable
            curr_state = new_state

            # flip disc on upper right side
            gap_upper_border, gap_right_border = x, self.consts.FIELD_WIDTH - y - 1
            it_number = jax.lax.cond((gap_upper_border < gap_right_border), lambda _: gap_upper_border, lambda _: gap_right_border, operand=None)
            new_state, discs_flippable, _, _ = jax.lax.fori_loop(0, it_number, loop_horizontal_and_vertical_line_to_flip_discs, (curr_state, False, False, self.consts.FLIP_UP_RIGHT_SIDE))
            new_state = jax.lax.cond(discs_flippable, lambda _: new_state, lambda _: curr_state, operand=None)
            valid_choice = valid_choice | discs_flippable
            curr_state = new_state

            # flip disc on down right side
            gap_down_border, gap_right_border = self.consts.FIELD_WIDTH - x - 1, self.consts.FIELD_WIDTH - y - 1
            it_number = jax.lax.cond((gap_down_border < gap_right_border), lambda _: gap_down_border, lambda _: gap_right_border, operand=None)
            new_state, discs_flippable, _, _ = jax.lax.fori_loop(0, it_number, loop_horizontal_and_vertical_line_to_flip_discs, (curr_state, False, False, self.consts.FLIP_DOWN_RIGHT_SIDE))
            new_state = jax.lax.cond(discs_flippable, lambda _: new_state, lambda _: curr_state, operand=None)
            valid_choice = valid_choice | discs_flippable
            curr_state = new_state

            # flip disc on down left side
            gap_down_border, gap_left_border = self.consts.FIELD_WIDTH - x - 1, y
            it_number = jax.lax.cond((gap_down_border < gap_left_border), lambda _: gap_down_border, lambda _: gap_left_border, operand=None)
            new_state, discs_flippable, _, _ = jax.lax.fori_loop(0, it_number, loop_horizontal_and_vertical_line_to_flip_discs, (curr_state, False, False, self.consts.FLIP_DOWN_LEFT_SIDE))
            new_state = jax.lax.cond(discs_flippable, lambda _: new_state, lambda _: curr_state, operand=None)
            valid_choice = valid_choice | discs_flippable
            curr_state = new_state

            # flip disc on up left side
            gap_upper_border, gap_left_border = x, y
            it_number = jax.lax.cond((gap_upper_border < gap_left_border), lambda _: gap_upper_border, lambda _: gap_left_border, operand=None)
            new_state, discs_flippable, _, _ = jax.lax.fori_loop(0, it_number, loop_horizontal_and_vertical_line_to_flip_discs, (curr_state, False, False, self.consts.FLIP_UP_LEFT_SIDE))
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


    @partial(jax.jit, static_argnums=(0,))
    def check_if_there_is_a_valid_choice(self, curr_state: OthelloState, white_player: bool) -> tuple[OthelloState, bool]:
        """
        Checks if the active player has at least one valid move available.

        Given the current game state (curr_state) and the active player's color 
        (white_player), this function systematically scans the board to determine 
        whether any empty position allows a legal Othello move:
          - Iterates over all empty board positions.
          - For each candidate position, evaluates in all eight directions 
            (horizontal, vertical, diagonal).
          - Verifies if placing a disc would flip at least one opponent disc.
          - Terminates early if a valid move is found.

        Args:
            curr_state (OthelloState): The current game state, including the board.
            white_player (bool): True if checking moves for the white player, 
                                 False if for the black player.

        Returns:
            tuple:
                - OthelloState: The unchanged current state.
                - bool: True if the player has at least one valid move, 
                        False otherwise.
        """
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
                    valid_field, _, _ = jax.lax.fori_loop(0, x, check_horizontal_vertical_and_diagonal_line,  (valid_field, False, self.consts.FLIP_UP_SIDE))

                    # check disc on the down side
                    valid_field, _, _ = jax.lax.fori_loop(0, self.consts.FIELD_HEIGHT - x - 1, check_horizontal_vertical_and_diagonal_line, (valid_field, False, self.consts.FLIP_DOWN_SIDE))

                    # check disc on the right side
                    valid_field, _, _ = jax.lax.fori_loop(0, self.consts.FIELD_WIDTH - y - 1, check_horizontal_vertical_and_diagonal_line, (valid_field, False, self.consts.FLIP_RIGHT_SIDE))

                    # check disc on the left side
                    valid_field, _, _ = jax.lax.fori_loop(0, y, check_horizontal_vertical_and_diagonal_line, (valid_field, False, self.consts.FLIP_LEFT_SIDE))
                    
                    # check disc on upper right side
                    gap_upper_border, gap_right_border = x, self.consts.FIELD_WIDTH - y - 1
                    it_number = jax.lax.cond((gap_upper_border < gap_right_border), lambda _: gap_upper_border, lambda _: gap_right_border, operand=None)
                    valid_field, _, _ = jax.lax.fori_loop(0, it_number, check_horizontal_vertical_and_diagonal_line, (valid_field, False, self.consts.FLIP_UP_RIGHT_SIDE))
                    
                    # check disc on down right side
                    gap_down_border, gap_right_border = self.consts.FIELD_WIDTH - x - 1, self.consts.FIELD_WIDTH - y - 1
                    it_number = jax.lax.cond((gap_down_border < gap_right_border), lambda _: gap_down_border, lambda _: gap_right_border, operand=None)
                    valid_field, _, _ = jax.lax.fori_loop(0, it_number, check_horizontal_vertical_and_diagonal_line, (valid_field, False, self.consts.FLIP_DOWN_RIGHT_SIDE))
                    
                    # check disc on down left side
                    gap_down_border, gap_left_border = self.consts.FIELD_WIDTH - x - 1, y
                    it_number = jax.lax.cond((gap_down_border < gap_left_border), lambda _: gap_down_border, lambda _: gap_left_border, operand=None)
                    valid_field, _, _ = jax.lax.fori_loop(0, it_number, check_horizontal_vertical_and_diagonal_line, (valid_field, False, self.consts.FLIP_DOWN_LEFT_SIDE))
                    
                    # check disc on up left side
                    gap_upper_border, gap_left_border = x, y
                    it_number = jax.lax.cond((gap_upper_border < gap_left_border), lambda _: gap_upper_border, lambda _: gap_left_border, operand=None)
                    valid_field, _, _ = jax.lax.fori_loop(0, it_number, check_horizontal_vertical_and_diagonal_line, (valid_field, False, self.consts.FLIP_UP_LEFT_SIDE))
                    
                    return valid_field

                valid_field = carry
                return jax.lax.cond(
                    jnp.logical_and(curr_state.field.field_color[i, j] == FieldColor.EMPTY, jnp.logical_not(valid_field)),
                    lambda x: if_empty(x),
                    lambda x: x,
                    valid_field
                )
            return jax.lax.fori_loop(0, self.consts.FIELD_WIDTH, inner_loop, (valid_field))
        valid_field = jax.lax.fori_loop(0, self.consts.FIELD_HEIGHT, outer_loop, (valid_field))
        return (curr_state, valid_field)


    @partial(jax.jit, static_argnums=(0,))
    def get_bot_move(self, game_field: Field, difficulty: chex.Array, player_score: chex.Array, enemy_score: chex.Array, random_key: chex.Array):
        """
        Determines the bot’s next move on the Othello board.

        Steps:
          1. Generate all possible board positions (0–63).
          2. For each position, evaluate its quality parallelly using jax.vmap
          3. Identify all moves that achieve the maximum score.
          4. Randomly choose one of these best moves using `random_max_index`
             and the provided PRNG key.
          5. Convert the chosen index into 2D board coordinates (row, column).

        Args:
            game_field (Field): Current board state, including disc colors.
            difficulty (chex.Array): Difficulty level that influences how strongly
                                     strategic heuristics weigh into scoring.
            player_score (chex.Array): Current score of the bot/player.
            enemy_score (chex.Array): Current score of the opponent.
            random_key (chex.Array): PRNG key used for random tie-breaking.

        Returns:
            chex.Array: A length-2 array `[y, x]` representing the chosen move’s
                        board coordinates.
        """
        game_score = player_score+enemy_score
        list_of_all_moves = jnp.arange(64)

        #calculate flipped tiles for all possible moves and adjust score based on game stage
        vectorized_compute_score_of_tiles = jax.vmap(self.compute_score_of_tiles, in_axes=(0, None, None, None))
        list_of_all_move_values = vectorized_compute_score_of_tiles(list_of_all_moves, game_field, game_score, difficulty)


        #Randomly choose one of the best moves    
        random_chosen_max_index = self.random_max_index(list_of_all_move_values,random_key)

        
        return jnp.array([jnp.floor_divide(random_chosen_max_index, 8), jnp.mod(random_chosen_max_index, 8)])

    @partial(jax.jit, static_argnums=(0,))
    def compute_score_of_tiles(self, i: int, game_field: Field, game_score: chex.Array, difficulty: chex.Array) -> int:
        """
        Evaluates the score of placing a tile at a given board position.
       

        Steps:
          1. Decode the board index `i` into (x, y) coordinates.
          2. If the position is already occupied, return a sentinel invalid score
             (`-2147483648`) and placeholder positions.
          3. Otherwise, call `compute_tiles_flipped_bot_move` to determine:
                - how many discs would be flipped (`tiles_flipped`),
                - a secondary tile relevant for strategic evaluation,
                - default fallback positions.
          4. If the move is valid (i.e., flips at least one disc):
                - Compute the *strategic score* of the move using 
                  `calculate_strategic_tile_score`, factoring in positional weights 
                  and difficulty settings.
                - Optionally consider a related secondary tile’s influence on the move.
                - Combine flipped-disc score with strategic score.
          5. If the move is invalid, return the sentinel score directly.

        Args:
            i (int): Linear index of the tile on an 8x8 board (0–63).
            game_field (Field): Current board state, including disc colors.
            game_score (chex.Array): Current score values used for evaluation.
            difficulty (int): Difficulty setting, influencing how strongly strategic
                              heuristics weigh into scoring.

        Returns:
            int: A scalar score (int32) representing the quality of the move.
                        Invalid moves return `-2147483648`.
        """
        # Decode tile position
        tile_y = jnp.floor_divide(i, 8)
        tile_x = jnp.mod(i, 8)

        def helper_return_default_pos():
            return (jnp.int32(-2147483648),(jnp.int32(-2147483648), jnp.int32(-2147483648)),(jnp.int32(-2147483648), jnp.int32(-2147483648)))


        # If tile is already taken, set invalid move (return very low score)
        args_compute_score_of_tile_1 = (tile_y, tile_x, game_field, game_score)
        tiles_flipped, secondary_tile, default_pos = jax.lax.cond(
            game_field.field_color[tile_y, tile_x] != FieldColor.EMPTY,
            lambda args_compute_score_of_tile_1: helper_return_default_pos(),
            lambda args_compute_score_of_tile_1: self.compute_tiles_flipped_bot_move(args_compute_score_of_tile_1),
            args_compute_score_of_tile_1
        )
        
        def handle_calculation_of_strategic_score(args: Tuple[int, Field, Tuple[int, int], chex.Array, Tuple[int, int]]) -> int:
            """
            Handles the calculation of the strategic score for a given tile.
            Seperated into calculation of strategic score for tile itself and a possible secondary tile.
            Args:
                tuple ([int, Field, Tuple[int, int], chex.Array, Tuple[int, int]])
                    - i (int): Linear index of the tile on an 8x8 board (0–63).
                    - game_field (Field): Current board state, including disc colors.
                    - default_pos (Tuple[int, int]): Default position to use if the tile is invalid.
                    - difficulty (chex.Array): Difficulty setting, influencing how strongly strategic
                                        heuristics weigh into scoring.
                    - secondary_tile (Tuple[int, int]): A secondary tile relevant for strategic evaluation.
            
            Returns:
                int: A scalar score (int32) representing the quality of the move.
            """
            i, game_field, default_pos, difficulty, secondary_tile = args

            #Calculate the strategic value (score) of the current_square itself
            ((score, _), skip_secondary_eval) = self.calculate_strategic_tile_score(i, game_field, jax.lax.cond(default_pos[0] == -2147483648, lambda _: (0,0),lambda _: default_pos, None), difficulty)

            secondary_tile = jax.lax.cond(
                skip_secondary_eval,
                lambda _: (-2147483648, -2147483648),
                lambda _: secondary_tile,
                None
            )

            #Re-evaluate or combine with a secondary related square's score
            args_handle_secondary_calculation_of_strategic_score = (secondary_tile[0] + secondary_tile[1]*8, game_field, default_pos, difficulty, score)
            score = jax.lax.cond(
                skip_secondary_eval,
                #lambda args_handle_secondary_calculation_of_strategic_score: handle_secondary_calculation_of_strategic_score(args_handle_secondary_calculation_of_strategic_score), #TODO fix outer corners- take a look at index changes by check neighbor conflict with loop borders
                lambda args_handle_secondary_calculation_of_strategic_score: score,
                lambda args_handle_secondary_calculation_of_strategic_score: score,
                args_handle_secondary_calculation_of_strategic_score
            )
            return score
        
        #only execute the strategic score calculation if tiles were flipped = is a valid move
        args = (i, game_field, default_pos, difficulty, secondary_tile)
        score = jax.lax.cond(tiles_flipped != -2147483648,
                    lambda args: tiles_flipped + handle_calculation_of_strategic_score(args),
                    lambda args: tiles_flipped,
                    args)


        return score

    @partial(jax.jit, static_argnums=(0,))
    def compute_tiles_flipped_bot_move(self, args: Tuple[int, int, Field, chex.Array]) -> Tuple[int, Tuple[int, int], Tuple[int, int]]:
        """
        Computes the number of discs flipped and determines positions for further evaluation
        for a given tile.

        Steps:
          1. For each of the 8 directions around the candidate tile:
                - Use `compute_flipped_tiles_by_direction` (via `jax.vmap`)
                  to determine:
                    • how many discs would be flipped in that direction,
                    • whether an "inner corner" (strategically relevant square)
                      is encountered along the line.
          2. Aggregate results across all directions:
                - Sum the number of flipped discs.
                - Identify the last valid "inner corner" encountered, both in
                  *any* direction and in the *final* direction.
          3. Compute the base score of the tile (includes adjusting for game stage):
                - If no discs are flipped → mark as invalid (`-2147483648`).
                - Otherwise → adjust the raw flipped-disc score depending on
                  the current game stage (`game_score`), penalizing flips in
                  the midgame.

        Args:
            args (Tuple[int, int, Field, chex.Array]):
                - tile_y (int): Row index of the candidate move.
                - tile_x (int): Column index of the candidate move.
                - game_field (Field): Current board state.
                - game_score (chex.Array): Total number of discs currently placed,used to infer game stage.

        Returns:
            Tuple[int, Tuple[int, int], Tuple[int, int]]:
                - score_of_tile (int): The flipped-disc-based score or sentinel `-2147483648` if invalid.
                - inner_corner_any (Tuple[int, int]): Coordinates of a relevant inner corner (if any).
                - inner_corner_final (Tuple[int, int]): Coordinates of a final direction inner corner (if any).
        """
        tile_y, tile_x, game_field, game_score = args
        inner_corner_in_any_direction = (-2147483648, -2147483648)
        inner_corner_in_final_direction = (-2147483648, -2147483648)

        #compute flipped tiles by each direction
        list_of_all_directions = jnp.arange(8)
        vectorised_flipped_tiles_by_direction = jax.vmap(self.compute_flipped_tiles_by_direction,in_axes=(0, None, None, None, None, None))
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

    def random_max_index(self, array: chex.Array, key: chex.Array) -> chex.Array:
        """
        Selects a random index among the maximum values in the input array.

        The function assumes the array has a fixed size of 64 (Important! Only works with arrays of size 64). It performs the following steps:

          1. Finds the maximum value in the array.
          2. Iterates over all elements to count how many times the maximum value appears 
             and records their indices.
          3. Constructs an array (`max_indexes_for_random`) where valid maximum-value 
             indices are compacted to the left (others filled with -1).
          4. Uses a PRNG key to select a random index among these maximum positions.

        Args:
            array (chex.Array): Input array of fixed size 64 containing move values.
            key (int): JAX random key used to generate a reproducible random choice.

        Returns:
            chex.Array: A single index (int32 scalar) corresponding to one of the 
                        maximum-value positions in the array, chosen at random.
        """
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


    @partial(jax.jit, static_argnums=(0,))
    def compute_flipped_tiles_by_direction(self, i, tile_y: int, tile_x: int, game_field: Field, inner_corner_in_any_direction: tuple[int, int], inner_corner_in_final_direction: tuple[int, int]) -> tuple[int, tuple[int, int], tuple[int, int]]:
        """
        Computes the number of tiles that would be flipped in a specific direction
        if a tile were placed at the given position (tile_y, tile_x).

        The direction is selected by the parameter `i` (0–7).

        Each branch calls the corresponding directional helper
        (e.g., `compute_flipped_tiles_top`, `compute_flipped_tiles_right`, etc.)
        to determine flips and update potential "inner corner" positions relevant 
        for strategic evaluation.

        Args:
            i (int): Direction index (0–7).
            tile_y (int): Row index of the tile (0–7).
            tile_x (int): Column index of the tile (0–7).
            game_field (Field): Current board state, including disc colors.
            inner_corner_in_any_direction (tuple[int, int]): Tracks the first inner-corner 
                encountered in any direction (if any).
            inner_corner_in_final_direction (tuple[int, int]): Tracks the first inner-corner 
                encountered specifically in the evaluated direction (if any).

        Returns:
            Tuple ([int, tuple[int, int], tuple[int, int]]):
                - int: Number of flipped tiles in the given direction.
                - tuple[int, int]: "inner corner in any direction".
                - tuple[int, int]: "inner corner in final direction".
        """
        args = (tile_y,tile_x,game_field,inner_corner_in_any_direction,inner_corner_in_final_direction)

        branches = [
            lambda args: self.compute_flipped_tiles_top(args),
            lambda args: self.compute_flipped_tiles_top_right(args),
            lambda args: self.compute_flipped_tiles_right(args),
            lambda args: self.compute_flipped_tiles_bottom_right(args),
            lambda args: self.compute_flipped_tiles_bottom(args),
            lambda args: self.compute_flipped_tiles_bottom_left(args),
            lambda args: self.compute_flipped_tiles_left(args),
            lambda args: self.compute_flipped_tiles_top_left(args),
        ]

        return jax.lax.switch(i, branches, args)

    def check_if_inner_corner(self, tile_y: int, tile_x: int) -> Tuple[bool, Tuple[int, int]]:
        """
        Checks whether a given board position corresponds to an "inner corner"
        (a square directly adjacent to one of the four outer corners, diagonally inward).

        If the tile is an inner corner ((1,1), (1,6), (6,1), or (6,6)), this method 
        returns a mirrored coordinate used for consistent strategic evaluation 
        in `calculate_strategic_tile_score`. Otherwise, it simply returns the 
        original (tile_y, tile_x).

        Args:
            tile_y (int): Row index of the tile (0–7).
            tile_x (int): Column index of the tile (0–7).

        Returns:
            Tuple ([bool, Tuple[int, int]]): 
                - Whether the tile is an inner corner.
                - Either the mirrored coordinates for an inner corner or the original (tile_y, tile_x) if not an inner corner.
        """
        return jax.lax.cond(jnp.logical_and(tile_y == 1, tile_x == 1),
                lambda _: (True, (6,6)), #account for flipped game field in calculate_strategic_tile_score
                lambda _: jax.lax.cond(jnp.logical_and(tile_y == 1, tile_x == 6),
                            lambda _: (True, (6,1)),
                            lambda _: jax.lax.cond(jnp.logical_and(tile_y == 6, tile_x == 1),
                                        lambda _: (True, (1,6)),
                                        lambda _: jax.lax.cond(jnp.logical_and(tile_y == 6, tile_x == 6),
                                                    lambda _: (True, (1,1)),
                                                    lambda _: (False, (tile_y, tile_x)),
                                                    None),
                                        None),
                            None),
                None)


    def compute_flipped_tiles_top(self, input:Tuple[int, int, Field, int, int, Tuple[int, int], Tuple[int, int]]) -> Tuple[int, tuple[int, int], tuple[int, int]]:
        """
        Handle case 1: Upward direction

        Computes how many tiles would be flipped when placing a piece
        and searching upwards (towards lower row indices) on the Othello board.

        The algorithm iteratively checks tiles in the upward direction starting
        from the current tile, applying Othello rules:

        - If the immediate neighbor is already the bot's tile (black),
        the move is invalid → no tiles flipped.
        - If empty space is encountered before finding a black tile,
        the move is invalid → no tiles flipped.
        - Otherwise, counts opponent tiles until a black tile is found,
        flipping all tiles in between.

        Additionally, the method checks whether an encountered tile is an
        "inner corner". If we encounter one the value for inner_corner_in_final_direction is set to the coordinates of the inner corner. 
        If we additionally encounter a flippable configuration after that, we update inner_corner_in_any_direction.

        Args:
            input (Tuple[int, int, Field, int, int, Tuple[int, int], Tuple[int, int]]): A tuple with the structure
                (tile_y, tile_x, game_field, flipped, tmp_flipped,
                inner_corner_in_final_direction, inner_corner_in_any_direction),
                where:
                    - tile_y (int): Starting row index.
                    - tile_x (int): Starting column index.
                    - game_field (Field): Current game state (disc colors).
                    - flipped (int): Accumulated flipped tile count.
                    - tmp_flipped (int): Tiles tentatively flipped until a closing black tile is found.
                    - inner_corner_in_final_direction (tuple[int, int]): Tracks inner corner found in this direction.
                    - inner_corner_in_any_direction (tuple[int, int]): Tracks first inner corner found in any direction.

        Returns:
            Tuple ([int, tuple[int, int], tuple[int, int]]):
                - int: Number of tiles flipped in the upward direction.
                - tuple[int, int]: Updated "inner corner in any direction".
                - tuple[int, int]: Updated "inner corner in final direction".
        """
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
            #check if we are a inner corner
            inner_corner_check = self.check_if_inner_corner(args[0], args[1])
            #when we encounter a not black field increase tmp_flipped, when we encounter a black field add the tmp_flipped (tiles we encountered before black field:= tiles to be flipped) to flipped tiles
            return jax.lax.cond(
                args[2].field_color[args[0], args[1]] == FieldColor.EMPTY,  
                lambda args: (-2, args[1], args[2], args[3],  args[4], args[5], args[6]),#if field is empthy, no further tiles can be flipped use set y to -2 to exit next loop iteration
                lambda args: (jax.lax.cond(
                    args[2].field_color[args[0], args[1]] == FieldColor.BLACK,  
                    lambda args: (args[0]-1, args[1], args[2], args[3] + args[4], 0, args[5], args[5]), #args[5] is no typo (set inner_corner_in_any_direction only when valid move is found)
                    lambda args: (args[0]-1, args[1], args[2], args[3], args[4] + 1, jax.lax.cond(inner_corner_check[0], lambda _: inner_corner_check[1], lambda _: args[5], None), args[6]),
                    args
                    )),
                args
            )

        while_loop_return = jax.lax.while_loop(while_cond, while_body, args)
        return jax.lax.cond(
            #check if we are end of line, then return no tiles flipped and no alt positions found
            input[0] < 0,
            lambda args: (jnp.int32(0.0), (-2147483648,-2147483648), (-2147483648,-2147483648)),
            lambda args: (jnp.int32(while_loop_return[3]), while_loop_return[6], while_loop_return[5]),
            args
        )
    
    def compute_flipped_tiles_right(self, input:Tuple[int, int, Field, int, int, Tuple[int, int], Tuple[int, int]]) -> Tuple[int, tuple[int, int], tuple[int, int]]:
        """
        Handle case 3: Right direction

        Computes how many tiles would be flipped when placing a piece
        and searching right (towards higher column indices) on the Othello board.

        The algorithm iteratively checks tiles in the right direction starting
        from the current tile, applying Othello rules:

        - If the immediate neighbor is already the bot's tile (black),
        the move is invalid → no tiles flipped.
        - If empty space is encountered before finding a black tile,
        the move is invalid → no tiles flipped.
        - Otherwise, counts opponent tiles until a black tile is found,
        flipping all tiles in between.

        Additionally, the method checks whether an encountered tile is an
        "inner corner". If we encounter one the value for inner_corner_in_final_direction is set to the coordinates of the inner corner. 
        If we additionally encounter a flippable configuration after that, we update inner_corner_in_any_direction.

        Args:
            input (Tuple[int, int, Field, int, int, Tuple[int, int], Tuple[int, int]]): A tuple with the structure
                (tile_y, tile_x, game_field, flipped, tmp_flipped,
                inner_corner_in_final_direction, inner_corner_in_any_direction),
                where:
                    - tile_y (int): Starting row index.
                    - tile_x (int): Starting column index.
                    - game_field (Field): Current game state (disc colors).
                    - flipped (int): Accumulated flipped tile count.
                    - tmp_flipped (int): Tiles tentatively flipped until a closing black tile is found.
                    - inner_corner_in_final_direction (tuple[int, int]): Tracks inner corner found in this direction.
                    - inner_corner_in_any_direction (tuple[int, int]): Tracks first inner corner found in any direction.

        Returns:
            Tuple ([int, tuple[int, int], tuple[int, int]]):
                - int: Number of tiles flipped in the upward direction.
                - tuple[int, int]: Updated "inner corner in any direction".
                - tuple[int, int]: Updated "inner corner in final direction".
        """
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
            #check if we are a inner corner
            inner_corner_check = self.check_if_inner_corner(args[0], args[1])
            #when we encounter a not black field increase tmp_flipped, when we encounter a black field add the tmp_flipped (tiles we encountered before black field:= tiles to be flipped) to flipped tiles      
            return jax.lax.cond(
                args[2].field_color[args[0], args[1]] == FieldColor.EMPTY,  
                lambda args: (-2, args[1], args[2], args[3],  args[4], args[5], args[6]),#if field is empthy, no further tiles can be flipped use set y to -2 to exit next loop iteration
                lambda args: (jax.lax.cond(
                    args[2].field_color[args[0], args[1]] == FieldColor.BLACK,  
                    lambda args: (args[0], args[1]+1, args[2], args[3] + args[4], 0,  args[5], args[5]),
                    lambda args: (args[0], args[1]+1, args[2], args[3], args[4] + 1, jax.lax.cond(inner_corner_check[0], lambda _: inner_corner_check[1], lambda _: args[5], None), args[6]),
                    args
                    )),
                args
            )

        while_loop_return = jax.lax.while_loop(while_cond, while_body, args)
        return jax.lax.cond(
            #check if we are end of line, then return no tiles flipped and no alt positions found
            input[1] > 7,
            lambda args: (jnp.int32(0.0), (-2147483648,-2147483648), (-2147483648,-2147483648)),
            lambda args: (jnp.int32(while_loop_return[3]), while_loop_return[6], while_loop_return[5]),
            args
        )
    
    def compute_flipped_tiles_bottom(self, input:Tuple[int, int, Field, int, int, Tuple[int, int], Tuple[int, int]]) -> Tuple[int, tuple[int, int], tuple[int, int]]:
        """
        Handle case 5: Bottom direction

        Computes how many tiles would be flipped when placing a piece
        and searching down (towards higher row indices) on the Othello board.

        The algorithm iteratively checks tiles in the down direction starting
        from the current tile, applying Othello rules:

        - If the immediate neighbor is already the bot's tile (black),
        the move is invalid → no tiles flipped.
        - If empty space is encountered before finding a black tile,
        the move is invalid → no tiles flipped.
        - Otherwise, counts opponent tiles until a black tile is found,
        flipping all tiles in between.

        Additionally, the method checks whether an encountered tile is an
        "inner corner". If we encounter one the value for inner_corner_in_final_direction is set to the coordinates of the inner corner. 
        If we additionally encounter a flippable configuration after that, we update inner_corner_in_any_direction.

        Args:
            input (Tuple[int, int, Field, int, int, Tuple[int, int], Tuple[int, int]]): A tuple with the structure
                (tile_y, tile_x, game_field, flipped, tmp_flipped,
                inner_corner_in_final_direction, inner_corner_in_any_direction),
                where:
                    - tile_y (int): Starting row index.
                    - tile_x (int): Starting column index.
                    - game_field (Field): Current game state (disc colors).
                    - flipped (int): Accumulated flipped tile count.
                    - tmp_flipped (int): Tiles tentatively flipped until a closing black tile is found.
                    - inner_corner_in_final_direction (tuple[int, int]): Tracks inner corner found in this direction.
                    - inner_corner_in_any_direction (tuple[int, int]): Tracks first inner corner found in any direction.

        Returns:
            Tuple ([int, tuple[int, int], tuple[int, int]]):
                - int: Number of tiles flipped in the upward direction.
                - tuple[int, int]: Updated "inner corner in any direction".
                - tuple[int, int]: Updated "inner corner in final direction".
        """
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
            #check if we are a inner corner
            inner_corner_check = self.check_if_inner_corner(args[0], args[1])
            #when we encounter a not black field increase tmp_flipped, when we encounter a black field add the tmp_flipped (tiles we encountered before black field:= tiles to be flipped) to flipped tiles      
            return jax.lax.cond(
                args[2].field_color[args[0], args[1]] == FieldColor.EMPTY,  
                lambda args: (-2, args[1], args[2], args[3],  args[4], args[5], args[6]),#if field is empthy, no further tiles can be flipped use set y to -2 to exit next loop iteration
                lambda args: (jax.lax.cond(
                    args[2].field_color[args[0], args[1]] == FieldColor.BLACK,  
                    lambda args: (args[0]+1, args[1], args[2], args[3] + args[4], 0, args[5], args[5]),
                    lambda args: (args[0]+1, args[1], args[2], args[3], args[4] + 1, jax.lax.cond(inner_corner_check[0], lambda _: inner_corner_check[1], lambda _: args[5], None), args[6]),
                    args
                    )),
                args
            )

        while_loop_return = jax.lax.while_loop(while_cond, while_body, args)
        return jax.lax.cond(
            #check if we are end of line, then return no tiles flipped and no alt positions found
            input[0] > 7,
            lambda args: (jnp.int32(0.0), (-2147483648,-2147483648), (-2147483648,-2147483648)),
            lambda args: (jnp.int32(while_loop_return[3]), while_loop_return[6], while_loop_return[5]),
            args
        )
    
    def compute_flipped_tiles_left(self, input:Tuple[int, int, Field, int, int, Tuple[int, int], Tuple[int, int]]) -> Tuple[int, tuple[int, int], tuple[int, int]]:
        """
        Handle case 7: Left direction

        Computes how many tiles would be flipped when placing a piece
        and searching left (towards lower column indices) on the Othello board.

        The algorithm iteratively checks tiles in the left direction starting
        from the current tile, applying Othello rules:

        - If the immediate neighbor is already the bot's tile (black),
        the move is invalid → no tiles flipped.
        - If empty space is encountered before finding a black tile,
        the move is invalid → no tiles flipped.
        - Otherwise, counts opponent tiles until a black tile is found,
        flipping all tiles in between.

        Additionally, the method checks whether an encountered tile is an
        "inner corner". If we encounter one the value for inner_corner_in_final_direction is set to the coordinates of the inner corner. 
        If we additionally encounter a flippable configuration after that, we update inner_corner_in_any_direction.

        Args:
            input (Tuple[int, int, Field, int, int, Tuple[int, int], Tuple[int, int]]): A tuple with the structure
                (tile_y, tile_x, game_field, flipped, tmp_flipped,
                inner_corner_in_final_direction, inner_corner_in_any_direction),
                where:
                    - tile_y (int): Starting row index.
                    - tile_x (int): Starting column index.
                    - game_field (Field): Current game state (disc colors).
                    - flipped (int): Accumulated flipped tile count.
                    - tmp_flipped (int): Tiles tentatively flipped until a closing black tile is found.
                    - inner_corner_in_final_direction (tuple[int, int]): Tracks inner corner found in this direction.
                    - inner_corner_in_any_direction (tuple[int, int]): Tracks first inner corner found in any direction.

        Returns:
            Tuple ([int, tuple[int, int], tuple[int, int]]):
                - int: Number of tiles flipped in the upward direction.
                - tuple[int, int]: Updated "inner corner in any direction".
                - tuple[int, int]: Updated "inner corner in final direction".
        """
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
            #check if we are a inner corner
            inner_corner_check = self.check_if_inner_corner(args[0], args[1])
            #when we encounter a not black field increase tmp_flipped, when we encounter a black field add the tmp_flipped (tiles we encountered before black field:= tiles to be flipped) to flipped tiles      
            return jax.lax.cond(
                args[2].field_color[args[0], args[1]] == FieldColor.EMPTY,
                lambda args: (-2, args[1], args[2], args[3], args[4], args[5], args[6]),  # if field is empthy, no further tiles can be flipped use set y to -2 to exit next loop iteration
                lambda args: (jax.lax.cond(
                    args[2].field_color[args[0], args[1]] == FieldColor.BLACK,
                    lambda args: (args[0], args[1]-1, args[2], args[3] + args[4], 0, args[5], args[5]),
                    lambda args: (args[0], args[1]-1, args[2], args[3], args[4] + 1, jax.lax.cond(inner_corner_check[0], lambda _: inner_corner_check[1], lambda _: args[5], None), args[6]),
                    args
                    )),
                args
            )

        while_loop_return = jax.lax.while_loop(while_cond, while_body, args)
        return jax.lax.cond(
            input[1] > 7,
            #check if we are end of line, then return no tiles flipped and no alt positions found
            lambda _: (jnp.int32(0.0), (-2147483648,-2147483648), (-2147483648,-2147483648)),
            lambda _: (jnp.int32(while_loop_return[3]), while_loop_return[6], while_loop_return[5]),
            None
        )
    
    def compute_flipped_tiles_top_right(self, input:Tuple[int, int, Field, int, int, Tuple[int, int], Tuple[int, int]]) -> Tuple[int, tuple[int, int], tuple[int, int]]:
        """
        Handle case 2: Top-right direction

        Computes how many tiles would be flipped when placing a piece
        and searching top-right (towards higher row indices and lower column indices) on the Othello board.

        The algorithm iteratively checks tiles in the top-right direction starting
        from the current tile, applying Othello rules:

        - If the immediate neighbor is already the bot's tile (black),
        the move is invalid → no tiles flipped.
        - If empty space is encountered before finding a black tile,
        the move is invalid → no tiles flipped.
        - Otherwise, counts opponent tiles until a black tile is found,
        flipping all tiles in between.

        Additionally, the method checks whether an encountered tile is an
        "inner corner". If we encounter one the value for inner_corner_in_final_direction is set to the coordinates of the inner corner. 
        If we additionally encounter a flippable configuration after that, we update inner_corner_in_any_direction.

        Args:
            input (Tuple[int, int, Field, int, int, Tuple[int, int], Tuple[int, int]]): A tuple with the structure
                (tile_y, tile_x, game_field, flipped, tmp_flipped,
                inner_corner_in_final_direction, inner_corner_in_any_direction),
                where:
                    - tile_y (int): Starting row index.
                    - tile_x (int): Starting column index.
                    - game_field (Field): Current game state (disc colors).
                    - flipped (int): Accumulated flipped tile count.
                    - tmp_flipped (int): Tiles tentatively flipped until a closing black tile is found.
                    - inner_corner_in_final_direction (tuple[int, int]): Tracks inner corner found in this direction.
                    - inner_corner_in_any_direction (tuple[int, int]): Tracks first inner corner found in any direction.

        Returns:
            Tuple ([int, tuple[int, int], tuple[int, int]]):
                - int: Number of tiles flipped in the upward direction.
                - tuple[int, int]: Updated "inner corner in any direction".
                - tuple[int, int]: Updated "inner corner in final direction".
        """
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
            #check if we are a inner corner
            inner_corner_check = self.check_if_inner_corner(args[0], args[1])
            #when we encounter a not black field increase tmp_flipped, when we encounter a black field add the tmp_flipped (tiles we encountered before black field:= tiles to be flipped) to flipped tiles      
            return jax.lax.cond(
                args[2].field_color[args[0], args[1]] == FieldColor.EMPTY,  
                lambda args: (-2, args[1], args[2], args[3],  args[4], args[5], args[6]),#if field is empthy, no further tiles can be flipped use set y to -2 to exit next loop iteration
                lambda args: (jax.lax.cond(
                    args[2].field_color[args[0], args[1]] == FieldColor.BLACK,
                    lambda args: (args[0]-1, args[1]+1, args[2], args[3] + args[4], 0, args[5], args[5]),
                    lambda args: (args[0]-1, args[1]+1, args[2], args[3], args[4] + 1, jax.lax.cond(inner_corner_check[0], lambda _: inner_corner_check[1], lambda _: args[5], None), args[6]),
                    args
                    )),
                args
            )

        while_loop_return = jax.lax.while_loop(while_cond, while_body, args)
        return jax.lax.cond(
            #check if we are end of line, then return no tiles flipped and no alt positions found
            jnp.logical_and(input[0] < 0, input[1] > 7),
            lambda args: (jnp.int32(0.0), (-2147483648, -2147483648), (-2147483648, -2147483648)),
            lambda args: (jnp.int32(while_loop_return[3]), while_loop_return[6], while_loop_return[5]),
            args
        )
    
    def compute_flipped_tiles_bottom_right(self, input:Tuple[int, int, Field, int, int, Tuple[int, int], Tuple[int, int]]) -> Tuple[int, tuple[int, int], tuple[int, int]]:
        """
        Handle case 4: Bottom-right direction

        Computes how many tiles would be flipped when placing a piece
        and searching bottom-right (towards lower row indices and lower column indices) on the Othello board.

        The algorithm iteratively checks tiles in the bottom-right direction starting
        from the current tile, applying Othello rules:

        - If the immediate neighbor is already the bot's tile (black),
        the move is invalid → no tiles flipped.
        - If empty space is encountered before finding a black tile,
        the move is invalid → no tiles flipped.
        - Otherwise, counts opponent tiles until a black tile is found,
        flipping all tiles in between.

        Additionally, the method checks whether an encountered tile is an
        "inner corner". If we encounter one the value for inner_corner_in_final_direction is set to the coordinates of the inner corner. 
        If we additionally encounter a flippable configuration after that, we update inner_corner_in_any_direction.

        Args:
            input (Tuple[int, int, Field, int, int, Tuple[int, int], Tuple[int, int]]): A tuple with the structure
                (tile_y, tile_x, game_field, flipped, tmp_flipped,
                inner_corner_in_final_direction, inner_corner_in_any_direction),
                where:
                    - tile_y (int): Starting row index.
                    - tile_x (int): Starting column index.
                    - game_field (Field): Current game state (disc colors).
                    - flipped (int): Accumulated flipped tile count.
                    - tmp_flipped (int): Tiles tentatively flipped until a closing black tile is found.
                    - inner_corner_in_final_direction (tuple[int, int]): Tracks inner corner found in this direction.
                    - inner_corner_in_any_direction (tuple[int, int]): Tracks first inner corner found in any direction.

        Returns:
            Tuple ([int, tuple[int, int], tuple[int, int]]):
                - int: Number of tiles flipped in the upward direction.
                - tuple[int, int]: Updated "inner corner in any direction".
                - tuple[int, int]: Updated "inner corner in final direction".
        """
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
            #check if we are a inner corner
            inner_corner_check = self.check_if_inner_corner(args[0], args[1])
            #when we encounter a not black field increase tmp_flipped, when we encounter a black field add the tmp_flipped (tiles we encountered before black field:= tiles to be flipped) to flipped tiles      
            return jax.lax.cond(
                args[2].field_color[args[0], args[1]] == FieldColor.EMPTY,  
                lambda args: (-2, args[1], args[2], args[3],  args[4], args[5], args[6]),#if field is empthy, no further tiles can be flipped use set y to -2 to exit next loop iteration
                lambda args: (jax.lax.cond(
                    args[2].field_color[args[0], args[1]] == FieldColor.BLACK,
                    lambda args: (args[0]+1, args[1]+1, args[2], args[3] + args[4], 0, args[5], args[5]),
                    lambda args: (args[0]+1, args[1]+1, args[2], args[3], args[4] + 1, jax.lax.cond(inner_corner_check[0], lambda _: inner_corner_check[1], lambda _: args[5], None), args[6]),
                    args
                    )),
                args
            )

        while_loop_return = jax.lax.while_loop(while_cond, while_body, args)
        return jax.lax.cond(
            #check if we are end of line, then return no tiles flipped and no alt positions found
            jnp.logical_and(input[0] > 7, input[1] > 7),
            lambda args: (jnp.int32(0.0), (-2147483648, -2147483648), (-2147483648, -2147483648)),
            lambda args: (jnp.int32(while_loop_return[3]), while_loop_return[6], while_loop_return[5]),
            args
        )
    
    def compute_flipped_tiles_bottom_left(self, input:Tuple[int, int, Field, int, int, Tuple[int, int], Tuple[int, int]]) -> Tuple[int, tuple[int, int], tuple[int, int]]:
        """
        Handle case 6: Bottom-left direction

        Computes how many tiles would be flipped when placing a piece
        and searching bottom-left (towards lower row indices and higher column indices) on the Othello board.

        The algorithm iteratively checks tiles in the bottom-left direction starting
        from the current tile, applying Othello rules:

        - If the immediate neighbor is already the bot's tile (black),
        the move is invalid → no tiles flipped.
        - If empty space is encountered before finding a black tile,
        the move is invalid → no tiles flipped.
        - Otherwise, counts opponent tiles until a black tile is found,
        flipping all tiles in between.

        Additionally, the method checks whether an encountered tile is an
        "inner corner". If we encounter one the value for inner_corner_in_final_direction is set to the coordinates of the inner corner. 
        If we additionally encounter a flippable configuration after that, we update inner_corner_in_any_direction.

        Args:
            input (Tuple[int, int, Field, int, int, Tuple[int, int], Tuple[int, int]]): A tuple with the structure
                (tile_y, tile_x, game_field, flipped, tmp_flipped,
                inner_corner_in_final_direction, inner_corner_in_any_direction),
                where:
                    - tile_y (int): Starting row index.
                    - tile_x (int): Starting column index.
                    - game_field (Field): Current game state (disc colors).
                    - flipped (int): Accumulated flipped tile count.
                    - tmp_flipped (int): Tiles tentatively flipped until a closing black tile is found.
                    - inner_corner_in_final_direction (tuple[int, int]): Tracks inner corner found in this direction.
                    - inner_corner_in_any_direction (tuple[int, int]): Tracks first inner corner found in any direction.

        Returns:
            Tuple ([int, tuple[int, int], tuple[int, int]]):
                - int: Number of tiles flipped in the upward direction.
                - tuple[int, int]: Updated "inner corner in any direction".
                - tuple[int, int]: Updated "inner corner in final direction".
        """
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
            #check if we are a inner corner
            inner_corner_check = self.check_if_inner_corner(args[0], args[1])
            #when we encounter a not black field increase tmp_flipped, when we encounter a black field add the tmp_flipped (tiles we encountered before black field:= tiles to be flipped) to flipped tiles      
            return jax.lax.cond(
                args[2].field_color[args[0], args[1]] == FieldColor.EMPTY,  
                lambda args: (-2, args[1], args[2], args[3],  args[4], args[5], args[6]),#if field is empthy, no further tiles can be flipped use set y to -2 to exit next loop iteration
                lambda args: (jax.lax.cond(
                    args[2].field_color[args[0], args[1]] == FieldColor.BLACK,
                    lambda args: (args[0]+1, args[1]-1, args[2], args[3] + args[4], 0, args[5], args[5]),
                    lambda args: (args[0]+1, args[1]-1, args[2], args[3], args[4] + 1, jax.lax.cond(inner_corner_check[0], lambda _: inner_corner_check[1], lambda _: args[5], None), args[6]),
                    args
                    )),
                args
            )        


        while_loop_return = jax.lax.while_loop(while_cond, while_body, args)
        return jax.lax.cond(
            #check if we are end of line, then return no tiles flipped and no alt positions found
            jnp.logical_and(input[0] < 0, input[1] > 7),
            lambda args: (jnp.int32(0.0), (-2147483648, -2147483648), (-2147483648, -2147483648)),
            lambda args: (jnp.int32(while_loop_return[3]), while_loop_return[6], while_loop_return[5]),
            args
        )
    
    def compute_flipped_tiles_top_left(self, input:Tuple[int, int, Field, int, int, Tuple[int, int], Tuple[int, int]]) -> Tuple[int, tuple[int, int], tuple[int, int]]:
        """
        Handle case 8: Top-left direction

        Computes how many tiles would be flipped when placing a piece
        and searching top-left (towards higher row indices and lower column indices) on the Othello board.

        The algorithm iteratively checks tiles in the top-left direction starting
        from the current tile, applying Othello rules:

        - If the immediate neighbor is already the bot's tile (black),
        the move is invalid → no tiles flipped.
        - If empty space is encountered before finding a black tile,
        the move is invalid → no tiles flipped.
        - Otherwise, counts opponent tiles until a black tile is found,
        flipping all tiles in between.

        Additionally, the method checks whether an encountered tile is an
        "inner corner". If we encounter one the value for inner_corner_in_final_direction is set to the coordinates of the inner corner. 
        If we additionally encounter a flippable configuration after that, we update inner_corner_in_any_direction.

        Args:
            input (Tuple[int, int, Field, int, int, Tuple[int, int], Tuple[int, int]]): A tuple with the structure
                (tile_y, tile_x, game_field, flipped, tmp_flipped,
                inner_corner_in_final_direction, inner_corner_in_any_direction),
                where:
                    - tile_y (int): Starting row index.
                    - tile_x (int): Starting column index.
                    - game_field (Field): Current game state (disc colors).
                    - flipped (int): Accumulated flipped tile count.
                    - tmp_flipped (int): Tiles tentatively flipped until a closing black tile is found.
                    - inner_corner_in_final_direction (tuple[int, int]): Tracks inner corner found in this direction.
                    - inner_corner_in_any_direction (tuple[int, int]): Tracks first inner corner found in any direction.

        Returns:
            Tuple ([int, tuple[int, int], tuple[int, int]]):
                - int: Number of tiles flipped in the upward direction.
                - tuple[int, int]: Updated "inner corner in any direction".
                - tuple[int, int]: Updated "inner corner in final direction".
        """
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
            #check if we are a inner corner
            inner_corner_check = self.check_if_inner_corner(args[0], args[1])
            #when we encounter a not black field increase tmp_flipped, when we encounter a black field add the tmp_flipped (tiles we encountered before black field:= tiles to be flipped) to flipped tiles      
            return jax.lax.cond(
                args[2].field_color[args[0], args[1]] == FieldColor.EMPTY,
                lambda args: (-2, args[1], args[2], args[3],  args[4], args[5], args[6]),#if field is empthy, no further tiles can be flipped use set y to -2 to exit next loop iteration
                lambda args: (jax.lax.cond(
                    args[2].field_color[args[0], args[1]] == FieldColor.BLACK,
                    lambda args: (args[0]-1, args[1]-1, args[2], args[3] + args[4], 0, args[5], args[6]),
                    lambda args: (args[0]-1, args[1]-1, args[2], args[3], args[4] + 1, jax.lax.cond(inner_corner_check[0], lambda _: inner_corner_check[1], lambda _: args[5], None), args[6]),
                    args
                    )),
                args
            )        


        while_loop_return = jax.lax.while_loop(while_cond, while_body, args)
        return jax.lax.cond(
            #check if we are end of line, then return no tiles flipped and no alt positions found
            jnp.logical_and(input[0] < 0, input[1] < 0),
            lambda args: (jnp.int32(0.0), (-2147483648, -2147483648), (-2147483648, -2147483648)),
            lambda args: (jnp.int32(while_loop_return[3]), while_loop_return[6], while_loop_return[5]),
            args
        )

    def handle_secondary_calculation_of_strategic_score(self, args: tuple[int, Field, tuple[int, int], chex.Array, int]) -> int:
        """
        Computes the strategic score for a secondary tile associated 
        with a potential move in Othello.

        Steps:
        2. Call `calculate_strategic_tile_score` for the secondary tile.
        3. If a valid alternate position is returned, its strategic score takes precedence over the score of the original tile, 
            if no valid position is found the tile gets the same score as the primary tile.
        4. Computes the total score by summing the primary and secondary contributions.
        5. Caps the total score to a valid range (-128 to 127); if out of range, 
            sets to a sentinel value (-104). This is due to potential overflow issues in the original code.
        6. If the tile index is outside valid bounds (<0 or ≥64), returns an invalid score (-2147483648).

        Args:
            args (tuple[int, Field, tuple[int, int], chex.Array, int]): A tuple containing:
                - tile_index (int): Linear index of the secondary tile.
                - game_field (Field): Current board state.
                - default_pos (tuple[int, int]): Default fallback position.
                - difficulty (chex.Array): Difficulty setting for strategic evaluation.
                - new_score (int): Primary tile's computed score.

        Returns:
            int: Adjusted score combining primary and secondary contributions, 
        """
        def handle_secondary_calculation_of_strategic_score_limit_to_valid(args: tuple[int, Field, tuple[int, int], chex.Array, int]) -> int:
            """
            Handles the logic of handle_secondary_calculation_of_strategic_score, guarded by a check for valid tile indexes.

            Args:
            args (tuple[int, Field, tuple[int, int], chex.Array, int]): A tuple containing:
                - tile_index (int): Linear index of the secondary tile.
                - game_field (Field): Current board state.
                - default_pos (tuple[int, int]): Default fallback position.
                - difficulty (chex.Array): Difficulty setting for strategic evaluation.
                - new_score (int): Primary tile's computed score.

            Returns:
                int: Adjusted score combining primary and secondary contributions,
            """
            tile_index, game_field, default_pos, difficulty, new_score = args
            new_score_copy = new_score
            ((new_score, new_alt_position), _) = self.calculate_strategic_tile_score(63-tile_index, game_field, default_pos, difficulty)


            # if no new alternative position found, keep original score
            new_score_copy = jax.lax.cond(
                new_alt_position != -2147483648,
                lambda _: new_alt_position,
                lambda _: new_score_copy,
                None
            )

            total_score = new_score + new_score_copy

            #account for possible overflow handling of original code
            total_score = jax.lax.cond(jnp.logical_or(total_score > 127, total_score < -128),
                lambda _: -104,
                lambda _: total_score,
                None
            ) 

            return total_score
        
        return jax.lax.cond(jnp.logical_and(args[0] >=64, args[0] < 0),
            lambda args: -2147483648,  
            lambda args: handle_secondary_calculation_of_strategic_score_limit_to_valid(args),  
            args
        )

    def calculate_strategic_tile_score(self, tile_index: int, game_field: Field, default_pos: Tuple[int, int], difficulty: chex.Array) -> Tuple[Tuple[int, int], bool]:
        """
        Calculates the strategic value of placing a tile at a specific board position.

        The strategic score is determined by a set of precalculated heuristics that
        evaluate the importance of each position on the board. 

        For an easier reimplementation of the original code, the board was flipped.

        Othello uses 18 different evaluation functions to assess the strategic value of each tile position.
        The attribution of these functions to the tile indices is stored in STRATEGIC_TILE_SCORE_CASES.

        Each case has its own helper function, which is called using a `jax.lax.switch`.
        Different functions have different complexities and call for different subroutines.

        Args : 
            tile_index (int): Linear index of the tile on an 8x8 board (0–63).
            game_field (Field): Current state of the game board including disc colors.
            default_pos (Tuple[int, int]): Default fallback position.
            difficulty (chex.Array): Difficulty setting affecting the weighting of strategic heuristics.

        Returns:
            Tuple ([[int, int], bool]): A tuple containing:
                - A Tuple with: 
                    - int: primary strategic score of the tile,
                    - int: secondary tile position for further strategic evaluation, or a sentinel (-2147483648) if not applicable.
                - bool: Whether a secondary evaluation needs to be performed.
        """
        game_field_flipped = Field(field_id=game_field.field_id, field_color=jnp.flip(game_field.field_color))
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
            _, _, game_field_flipped, default_pos_combined, difficulty = args
            return ((self.css_calculate_top_left_score(game_field_flipped, default_pos_combined, difficulty)), True)

        def compute_strategic_score_top(args):
            tile_y, tile_x, game_field_flipped, default_pos_combined, difficulty = args

            return ((self.css__f2d3_count_tiles_horizontally(game_field_flipped, tile_y, tile_x, default_pos_combined, difficulty)), False)

        def compute_strategic_score_top_right(args):
            _, _, game_field_flipped, default_pos_combined, difficulty = args

            return ((self.css_calculate_top_right_score(game_field_flipped, default_pos_combined, difficulty)), True)

        def compute_strategic_score_top_left_inner(args):
            _, _, game_field_flipped, _, _ = args
            return ((self.css_check_three_tiles(game_field_flipped, 0xf, 0x39, 0x3f), -2147483648), False)

        def compute_strategic_score_top_inner(args):
            #Strategic score for top inner field is influenced by the field above it
            tile_y, tile_x, game_field_flipped, _, _ = args
            return ((self.css_check_tile_up(tile_y, tile_x, game_field_flipped), -2147483648), False)

        def compute_strategic_score_top_right_inner(args):
            _, _, game_field_flipped, _, _ = args
            return ((self.css_check_three_tiles(game_field_flipped, 0x8, 0x3e, 0x38), -2147483648), False)

        def compute_strategic_score_left(args):
            tile_y, tile_x, game_field_flipped, default_pos_combined, difficulty = args

            return ((self.css__f2d3_count_tiles_vertically(game_field_flipped, tile_y, tile_x, default_pos_combined, difficulty)), False)

        def compute_strategic_score_left_inner(args):
            #Strategic score for left inner field is influenced by the field to the left of it
            tile_y, tile_x, game_field_flipped, _, _ = args
            return ((self.css_check_tile_left(tile_y, tile_x, game_field_flipped), -2147483648), False)

        def compute_strategic_score_center_corner(args):
            # Center corner has a fixed score of 4, no further explanation needed
            return ((4, -2147483648), False)

        def compute_strategic_score_center(args):
            #center has a fixed score of 0, no further evaluation needed
            return ((0, -2147483648), False)

        def compute_strategic_score_right_inner(args):
            #Strategic score for right inner field is influenced by the field to the right of it
            tile_y, tile_x, game_field_flipped, _, _ = args
            return ((self.css_check_tile_right(tile_y, tile_x, game_field_flipped), -2147483648), False)

        def compute_strategic_score_right(args):
            tile_y, tile_x, game_field_flipped, default_pos_combined, difficulty = args

            return ((self.css__f2d3_count_tiles_vertically(game_field_flipped, tile_y, tile_x, default_pos_combined, difficulty)), False)


        def compute_strategic_score_bottom_left_inner(args):
            _, _, game_field_flipped, _, _ = args
            return ((self.css_check_three_tiles(game_field_flipped, 0x1, 0x37, 0x7), -2147483648), False)

        def compute_strategic_score_bottom_inner(args):
            #Strategic score for bottom inner field is influenced by the field below it
            tile_y, tile_x, game_field_flipped, _, _ = args
            return ((self.css_check_tile_down(tile_y, tile_x, game_field_flipped), -2147483648), False)

        def compute_strategic_score_bottom_right_inner(args):
            _, _, game_field_flipped, _, _ = args
            return ((self.css_check_three_tiles(game_field_flipped, 0x6, 0x30, 0x0), -2147483648), False)

        def compute_strategic_score_bottom_left(args):
            _, _, game_field_flipped, default_pos_combined, difficulty = args

            return ((self.css_calculate_bottom_left_score(game_field_flipped, default_pos_combined, difficulty)), True)

        def compute_strategic_score_bottom(args):
            tile_y, tile_x, game_field_flipped, default_pos_combined, difficulty = args

            return ((self.css__f2d3_count_tiles_horizontally(game_field_flipped, tile_y, tile_x, default_pos_combined, difficulty)), False)

        def compute_strategic_score_bottom_right(args):
            _, _, game_field_flipped, default_pos_combined, difficulty = args

            return ((self.css_calculate_bottom_right_score(game_field_flipped, default_pos_combined, difficulty)), True)

        return jax.lax.switch(self.consts.STRATEGIC_TILE_SCORE_CASES[tile_index], branches, args)

    def css_check_tile_down(self,tile_y: int, tile_x: int, game_field: Field) -> int:
        """
        Handles the evaluation of bottom_inner tiles.
        If the field below has the same color as the current tile, the score is 4, else it is -8.

        Args:
            tile_y (int): The y-coordinate of the current tile.
            tile_x (int): The x-coordinate of the current tile.
            game_field (Field): The game field in its current state.

        Returns:
            int: The strategic score for the tile.
        """
        return jax.lax.cond(
            game_field.field_color[tile_y - 1, tile_x] == game_field.field_color[tile_y , tile_x],
            lambda _: 4,
            lambda _: -8,
            None)

    def css_check_tile_up(self,tile_y: int, tile_x: int, game_field: Field) -> int:
        """
        Handles the evaluation of top_inner tiles.
        If the field above has the same color as the current tile, the score is 4, else it is -8.

        Args:
            tile_y (int): The y-coordinate of the current tile.
            tile_x (int): The x-coordinate of the current tile.
            game_field (Field): The game field in its current state.

        Returns:
            int: The strategic score for the tile.
        """
        return jax.lax.cond(
            game_field.field_color[tile_y + 1, tile_x] == game_field.field_color[tile_y , tile_x],
            lambda _: 4,
            lambda _: -8,
            None)

    def css_check_tile_left(self, tile_y: int, tile_x: int, game_field: Field) -> int:
        """
        Handles the evaluation of left_inner tiles.
        If the field to the left has the same color as the current tile, the score is 4, else it is -8.

        Args:
            tile_y (int): The y-coordinate of the current tile.
            tile_x (int): The x-coordinate of the current tile.
            game_field (Field): The game field in its current state.

        Returns:
            int: The strategic score for the tile.
        """
        return jax.lax.cond(
            game_field.field_color[tile_y, tile_x + 1] == game_field.field_color[tile_y , tile_x],
            lambda _: 4,
            lambda _: -8,
            None)

    def css_check_tile_right(self, tile_y: int, tile_x: int, game_field: Field) -> int:
        """
        Handles the evaluation of right_inner tiles.
        If the field to the right has the same color as the current tile, the score is 4, else it is -8.

        Args:
            tile_y (int): The y-coordinate of the current tile.
            tile_x (int): The x-coordinate of the current tile.
            game_field (Field): The game field in its current state.

        Returns:
            int: The strategic score for the tile.
        """
        return jax.lax.cond(
            game_field.field_color[tile_y, tile_x - 1] == game_field.field_color[tile_y , tile_x],
            lambda _: 4,
            lambda _: -8,
            None)

    def css_check_three_tiles(self, game_field: Field, field_1: int, field_2: int, field_3: int) -> int:
        """
        Handles the evaluation of the inner_corner tiles.
        Checks for the color of the other three inner corners, with field_3 posing as the opposing corner.
        Inner corners are heavily punished, if one of the adjacent corners is already occupied by the player and even more if the opposing on is alredy owned by the player,

        Args:
            game_field (Field): The game field in its current state.
            field_1 (int): The position of the first adjacent corner.
            field_2 (int): The position of the second adjacent corner.
            field_3 (int): The position of the opposing corner.

        Returns:
            int: The strategic score for the tile.
        """
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

    def css_calculate_bottom_right_score(self, game_field: Field, default_pos_combined: int, difficulty: int) -> Tuple[int, int]:
        """
        Computes the strategic score for the bottom-right corner tile on the board.
        A score for the bottom row and right column is used to determine the strategic value of this position.
        By default the position receives a score of 56, and the default secondary position is taken from the initial calculation of the flipped tiles.
        -> Represents one of the inner corners

        if a valid configuration is found in one of the evaluated lines, the score and secondary position is adjusted accordingly.

        Args:
            game_field (Field): Current state of the board including disc colors.
            default_pos_combined (int): Default fallback position for secondary evaluation.
            difficulty (int): Difficulty level.

        Returns:
            Tuple([int, int]): A tuple containing:
                - Calculated strategic score for the bottom-right tile,
                - Position to consider as an alternate if necessary.
        """
        #alt singature: game_field: Field, ai_think_timer: int, default_pos: int, difficulty: int, y_pos: int, x_pos: int
        return_value = ((56,default_pos_combined), False) # (Touple to be returned, Aborted)
        (horizontal_score, alternate_pos_horz) = self.css__f2d3_count_tiles_horizontally(game_field, 0, 7, default_pos_combined, difficulty) #TODO check for correctness of 0,7

        jax.lax.cond(horizontal_score < 0,
            lambda _: ((horizontal_score, alternate_pos_horz), True),
            lambda _: return_value,
            None)

        (vertical_score, alternate_pos_vert) = self.css__f2d3_count_tiles_vertically(game_field, 7, 0, default_pos_combined, difficulty) #TODO check for correctness of 7,0

        jax.lax.cond(jnp.logical_and(vertical_score < 0, return_value[1] == False),
            lambda _: ((vertical_score, alternate_pos_vert), True),
            lambda _: return_value,
            None)

        return return_value[0]

    def css_calculate_top_left_score(self,game_field: Field, default_pos_combined: int, difficulty: int) -> Tuple([int, int]):
        """
        Computes the strategic score for the top-left corner tile on the board.
        A score for the top row and left column is used to determine the strategic value of this position.
        By default the position receives a score of 56, and the default secondary position is taken from the initial calculation of the flipped tiles.
        -> Represents one of the inner corners

        if a valid configuration is found in one of the evaluated lines, the score and secondary position is adjusted accordingly.

        Args:
            game_field (Field): Current state of the board including disc colors.
            default_pos_combined (int): Default fallback position for secondary evaluation.
            difficulty (int): Difficulty level.

        Returns:
            Tuple([int, int]): A tuple containing:
                - Calculated strategic score for the bottom-right tile,
                - Position to consider as an alternate if necessary.
        """
        return_value = ((56,default_pos_combined), False) # (Touple to be returned, Aborted)
        (horizontal_score, alternate_pos_horz) = self.css__f2d3_count_tiles_horizontally(game_field, 7, 0, default_pos_combined, difficulty) #TODO check for correctness of 7,0

        jax.lax.cond(horizontal_score < 0,
            lambda _: ((horizontal_score, alternate_pos_horz), True),
            lambda _: return_value,
            None)

        (vertical_score, alternate_pos_vert) = self.css__f2d3_count_tiles_vertically(game_field, 0, 7, default_pos_combined, difficulty) #TODO check for correctness of 0,7

        jax.lax.cond(jnp.logical_and(vertical_score < 0, return_value[1] == False),
            lambda _: ((vertical_score, alternate_pos_vert), True),
            lambda _: return_value,
            None)

        return return_value[0]

    def css_calculate_top_right_score(self, game_field: Field, default_pos_combined: int, difficulty: int) -> Tuple[int, int]:
        """
        Computes the strategic score for the top-right corner tile on the board.
        A score for the top row and right column is used to determine the strategic value of this position.
        By default the position receives a score of 56, and the default secondary position is taken from the initial calculation of the flipped tiles.
        -> Represents one of the inner corners

        if a valid configuration is found in one of the evaluated lines, the score and secondary position is adjusted accordingly.

        Args:
            game_field (Field): Current state of the board including disc colors.
            default_pos_combined (int): Default fallback position for secondary evaluation.
            difficulty (int): Difficulty level.

        Returns:
            Tuple([int, int]): A tuple containing:
                - Calculated strategic score for the bottom-right tile,
                - Position to consider as an alternate if necessary.
        """
        return_value = ((56,default_pos_combined), False) # (Touple to be returned, Aborted)
        (horizontal_score, alternate_pos_horz) = self.css__f2d3_count_tiles_horizontally(game_field, 0, 0, default_pos_combined, difficulty) #TODO check for correctness of 0,0

        jax.lax.cond(horizontal_score < 0,
            lambda _: ((horizontal_score, alternate_pos_horz), True),
            lambda _: return_value,
            None)

        (vertical_score, alternate_pos_vert) = self.css__f2d3_count_tiles_vertically(game_field, 7, 7, default_pos_combined, difficulty) #TODO check for correctness of 7,7

        jax.lax.cond(jnp.logical_and(vertical_score < 0, return_value[1] == False),
            lambda _: ((vertical_score, alternate_pos_vert), True),
            lambda _: return_value,
            None)

        return return_value[0]

    def css_calculate_bottom_left_score(self, game_field: Field, default_pos_combined: int, difficulty: int) -> Tuple[int, int]:
        """
        Computes the strategic score for the bottom-left corner tile on the board.
        A score for the bottom row and left column is used to determine the strategic value of this position.
        By default the position receives a score of 56, and the default secondary position is taken from the initial calculation of the flipped tiles.
        -> Represents one of the inner corners

        if a valid configuration is found in one of the evaluated lines, the score and secondary position is adjusted accordingly.

        Args:
            game_field (Field): Current state of the board including disc colors.
            default_pos_combined (int): Default fallback position for secondary evaluation.
            difficulty (int): Difficulty level.

        Returns:
            Tuple([int, int]): A tuple containing:
                - Calculated strategic score for the bottom-right tile,
                - Position to consider as an alternate if necessary.
        """
        return_value = ((56,default_pos_combined), False) # (Touple to be returned, Aborted)
        (horizontal_score, alternate_pos_horz) = self.css__f2d3_count_tiles_horizontally(game_field, 7, 7, default_pos_combined, difficulty) #TODO check for correctness of 7,7

        jax.lax.cond(horizontal_score < 0,
            lambda _: ((horizontal_score, alternate_pos_horz), True),
            lambda _: return_value,
            None)

        (vertical_score, alternate_pos_vert) = self.css__f2d3_count_tiles_vertically(game_field, 0, 0, default_pos_combined, difficulty) #TODO check for correctness of 0,0

        jax.lax.cond(jnp.logical_and(vertical_score < 0, return_value[1] == False),
            lambda _: ((vertical_score, alternate_pos_vert), True),
            lambda _: return_value,
            None)

        return return_value[0]

    def css__f2d3_count_tiles_vertically(self,game_field: Field, y_pos:int, x_pos: int, default_pos: int, difficulty: int):
        empty_line = jnp.arange(8, dtype=jnp.int32)

        def get_field_color_tiles_vertically(i, y_pos, x_pos, game_field):
            return game_field.field_color[i, x_pos]

        vectorized_array_of_tiles = jax.vmap(get_field_color_tiles_vertically, in_axes=(0, None, None, None))
        array_of_tiles = Field(field_id=jnp.arange(8),
                            field_color=vectorized_array_of_tiles(empty_line, y_pos, x_pos, game_field)) #TODO check if flip is needed

        return self.css__f2d3_count_tiles_in_line(array_of_tiles, y_pos, default_pos, difficulty)

    def css__f2d3_count_tiles_horizontally(self, game_field: Field, y_pos:int, x_pos: int, default_pos: int, difficulty: int):
        array_of_tiles = Field(
            field_id=jnp.arange(8),
            field_color=game_field.field_color[y_pos] #TODO check if flip is needed
        )


        return self.css__f2d3_count_tiles_in_line(array_of_tiles, x_pos, default_pos, difficulty)



    def css__f2d3_count_tiles_in_line(self, array_of_tiles, pos: int, default_pos_combined: int, difficulty: int):
        return_value = ((-1, -1), False) # (Touple to be returned, Aborted) 

        return_value = jax.lax.cond(difficulty == 1,
            lambda _: ((0, -1), True),
            lambda _: return_value,
            None)

        reversed_array_of_tiles = Field(
            field_id=jnp.flip(array_of_tiles.field_id),
            field_color=jnp.flip(array_of_tiles.field_color)
        )

        left_state, left_pos_opt = self.css_sub_f5c1_count_tiles_in_line_descending(array_of_tiles, pos)
        right_state, right_pos_opt = self.css_sub_f5c1_count_tiles_in_line_descending(reversed_array_of_tiles,7 - pos)

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
            lambda _: ((jnp.int32(self.consts.F7EC[combined_state]), right_pos), True),
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
                jnp.logical_and(self.consts.F3BA[33-i] == white_mask_low, self.consts.F3DC[33-i] == black_mask_low),
                lambda _: ((jnp.int32(self.consts.F3FE[33-i]), white_mask_high),True),
                lambda _: return_value,
                None)
            
            return_value = jax.lax.cond(
                jnp.logical_and(return_value[1] == False, 
                    jnp.logical_and(self.consts.F3BA[33-i] == white_mask_high, self.consts.F3DC[33-i] == black_mask_high)),
                lambda _: ((jnp.int32(self.consts.F3FE[33-i]), white_mask_high),True), #TODO check in Assembly for correctnes!
                lambda _: return_value,
                None)
            
            return (return_value, black_mask_low, black_mask_high, white_mask_low, white_mask_high)

        return_value, _, _, _, _ = jax.lax.cond(return_value[1] == False, 
                                                lambda init_val2: jax.lax.fori_loop(0,34, css_sub_f5c1_count_tiles_in_line_loop_check_masks, init_val2),
                                                lambda init_val2: (return_value, black_mask_low, black_mask_high, white_mask_low, white_mask_high),
                                                init_val2)

        return jax.lax.cond(return_value[1] == True,
            lambda _: return_value[0],
            lambda _: (jnp.int32(self.consts.F3FE[combined_state]), white_mask_high),
            None)
    
    #return a tuple of (state, position)
    def css_sub_f5c1_count_tiles_in_line_descending(self, array_of_tiles, start_index: int) ->Tuple[int, int]:
        adjacent_index = start_index-1

        args = (array_of_tiles, adjacent_index)
        return jax.lax.cond(adjacent_index < 0,
        lambda args: (0b00, -1),
        lambda args: self.css_sub_f5c1_count_tiles_in_line_descending_not_first_tile(args),
        args
    )

    def css_sub_f5c1_count_tiles_in_line_descending_not_first_tile(self, args) ->Tuple[int, int]:
        array_of_tiles, adjacent_index = args

        return jax.lax.cond(
            array_of_tiles.field_color[adjacent_index] == FieldColor.EMPTY,
            lambda args: (0b01, adjacent_index),
            lambda args: jax.lax.cond(
                array_of_tiles.field_color[adjacent_index] == FieldColor.WHITE,
                lambda args: self.css_sub_f5c1_count_tiles_in_line_descending_handle_white(args),
                lambda args: self.css_sub_f5c1_count_tiles_in_line_descending_handle_black(args),
                args
            ),
            args
        )

    def css_sub_f5c1_count_tiles_in_line_descending_handle_white(self, args) ->Tuple[int, int]:
        array_of_tiles, adjacent_index = args
        init_val = array_of_tiles, (-1, -1), adjacent_index-1

        output = jax.lax.fori_loop(
            0,
            adjacent_index,
            lambda i, init_val: self.css_sub_f5c1_count_tiles_in_line_descending_handle_white_loop(i, init_val),
            init_val
        )

        return jax.lax.cond(output[1][0] == -1,
            lambda _: (0b11, -1),
            lambda _: output[1],
            None
        )

    def css_sub_f5c1_count_tiles_in_line_descending_handle_white_loop(self, i, loop_vals):
        array_of_tiles, touple, custom_iterator = loop_vals

        touple = jax.lax.cond(
            jnp.logical_or(array_of_tiles.field_color[custom_iterator] == FieldColor.WHITE, jnp.logical_or(touple[0] == 0b11, jnp.logical_or(touple[0] == 0b10, touple[0] == 0b00))),
            lambda loop_vals: touple , # interrrupt the loop if there is an empty space before a Black tile
            lambda loop_vals: jax.lax.cond(
                array_of_tiles.field_color[custom_iterator] == FieldColor.EMPTY,
                lambda loop_vals: (0b11, -1), # do nothing for white tiles
                lambda loop_vals: self.css_sub_f5c1_count_tiles_in_line_descending_handle_white_loop_black_tile(loop_vals), 
                loop_vals
            ),
            loop_vals
        )

        custom_iterator -= 1
        return (array_of_tiles, touple, custom_iterator)

    def css_sub_f5c1_count_tiles_in_line_descending_handle_white_loop_black_tile(self, args) -> Tuple[int, int]:
        #guard against us being already in the last iteration
        _, _, custom_iterator = args
        return jax.lax.cond(custom_iterator != 0,
            lambda args: self.css_sub_f5c1_count_tiles_in_line_descending_handle_white_loop_black_tile_2(args),
            lambda args: (0b00, -1),   # Black found at end (possible valid reversal pattern)
            args
        )

    def css_sub_f5c1_count_tiles_in_line_descending_handle_white_loop_black_tile_2(self, args) -> Tuple[int, int]:
        array_of_tiles, _, custom_iterator = args

        init_val = array_of_tiles, (-1, -1), custom_iterator

        output = jax.lax.fori_loop(
            0,
            custom_iterator+1,
            lambda i, init_val: self.css_sub_f5c1_count_tiles_in_line_descending_handle_white_loop_black_tile_loop(i, init_val),
            init_val
        )
        # if the next tile is black, increase the count of white tiles
        return jax.lax.cond(output[1][0] == -1,
            lambda _: (0b00, -1),
            lambda _: output[1],
            None
        )

    def css_sub_f5c1_count_tiles_in_line_descending_handle_white_loop_black_tile_loop(self, i, loop_vals):
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

    def css_sub_f5c1_count_tiles_in_line_descending_handle_black(self, args) -> Tuple[int, int]:
        array_of_tiles, adjacent_index = args
        init_val = array_of_tiles, (-1, -1), adjacent_index-1

        output = jax.lax.fori_loop(
            0,
            adjacent_index,
            lambda i, init_val: self.css_sub_f5c1_count_tiles_in_line_descending_handle_black_loop(i, init_val),
            init_val
        )
        return jax.lax.cond(output[1][0] == -1,
            lambda _: (0b00, -1),
            lambda _: output[1],
            None
        )

    def css_sub_f5c1_count_tiles_in_line_descending_handle_black_loop(self, i, loop_vals):
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

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: OthelloState, action: chex.Array) -> Tuple[OthelloObservation, OthelloState, float, bool, OthelloInfo]:

        state = state._replace(step_counter=state.step_counter+1)

        decided, new_field_choice = self.has_player_decided_field(state.field_choice_player, action)  # 2D Array new_field_choice[i, j]

        state = state._replace(field_choice_player=new_field_choice)


        # first, it need to be checked if there is a valid place on the field the disc to be set
        _, valid_field = self.check_if_there_is_a_valid_choice(state, white_player=True)

        # check if the new_field_choice is a valid option
        valid_choice, new_state = jax.lax.cond(
            decided,
            lambda _: self.field_step(new_field_choice, state, True),
            lambda _: (False, state),
            operand=None
        )

        #check whether there is a valid move for the bot, otherwise mark the game as ended
        has_game_ended = jnp.logical_not(self.check_if_there_is_a_valid_choice(new_state, white_player=False)[1])
        # now enemy step are required

        def condition_fun(value):
            valid_choice, new_state, _ = value
            valid_choice = jnp.logical_not(valid_choice)
            return valid_choice

        def body_fun(value):
            valid_choice, state, key = value

            best_val = self.get_bot_move(state.field,state.difficulty, state.player_score,state.enemy_score,state.random_key)

            valid_choice, new_state = self.field_step(best_val, state, False)

            return jax.lax.cond(
                valid_choice,
                lambda _: (True, new_state, key),
                lambda _: (False, state, key),
                operand=None
            )

        _, valid_field_enemy = self.check_if_there_is_a_valid_choice(new_state, white_player=False)

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
        # def outer_loop(i, ended):
        #     def inner_loop(j, ended):
        #         ended = jax.lax.cond(
        #             final__step_state.field.field_color[i,j] == FieldColor.EMPTY,
        #             lambda _: False,
        #             lambda x: x,
        #             ended
        #         )
        #         return ended
        #     return jax.lax.fori_loop(0, self.consts.FIELD_WIDTH, inner_loop, ended)
        #has_game_ended = jax.lax.fori_loop(0, self.consts.FIELD_HEIGHT, outer_loop, True)
        has_game_ended = jax.lax.cond(has_game_ended,
            lambda _: True,
            lambda x: jnp.logical_not(self.check_if_there_is_a_valid_choice(final__step_state, white_player=True)[1]),
            has_game_ended
        )

        has_game_ended = jax.lax.cond(
            jnp.logical_or(final__step_state.player_score == 0, final__step_state.enemy_score == 0),
            lambda _: True,
            lambda x: x,
            has_game_ended
        )
        final__step_state = final__step_state._replace(end_of_game_reached=has_game_ended)

        jax.debug.print("has_game_ended: {}, Valid Choice: {}", final__step_state.end_of_game_reached, self.check_if_there_is_a_valid_choice(final__step_state, white_player=True)[1])

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

    def action_space(self) -> jnp.ndarray:
        return jnp.array(len(self.action_set))

    def observation_space(self) -> spaces.Box:
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
    player = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/othello/player_white_disc.npy"), transpose=False)
    enemy = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/othello/enemy_black_disc.npy"), transpose=False)

    bg = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/othello/othello_background.npy"), transpose=False)

    # TODO: get a correctly sized background image / resize the saved image..
    #bg = jax.image.resize(bg, (WIDTH, HEIGHT, 4), method='bicubic')

    # Convert all sprites to the expected format (add frame dimension)
    SPRITE_BG = jnp.expand_dims(bg, axis=0)
    SPRITE_PLAYER = jnp.expand_dims(player, axis=0)
    SPRITE_ENEMY = jnp.expand_dims(enemy, axis=0)

    # Load digits for scores
    PLAYER_DIGIT_SPRITES = jr.load_and_pad_digits(
        os.path.join(MODULE_DIR, "sprites/othello/number_{}_player.npy"),
        num_chars=10,
    )
    ENEMY_DIGIT_SPRITES = jr.load_and_pad_digits(
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


class OthelloRenderer(JAXGameRenderer):
    def __init__(self,consts: OthelloConstants = None):
        super().__init__()
        self.consts = consts or OthelloConstants()
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
        raster = jnp.zeros((self.consts.HEIGHT, self.consts.WIDTH, 3))

        # Render Background - (0, 0) is top-left corner
        frame_bg = jr.get_sprite_frame(self.SPRITE_BG, 0)
        raster = jr.render_at(raster, 0, 0, frame_bg)

        # disc sprites
        frame_player = jr.get_sprite_frame(self.SPRITE_PLAYER, 0)
        frame_enemy = jr.get_sprite_frame(self.SPRITE_ENEMY, 0)

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
                            lambda x: jr.render_at(raster, render_point[0], render_point[1], frame_player),
                            lambda x: jr.render_at(raster, render_point[0], render_point[1], frame_enemy),
                            x
                        ),
                        color
                    )

                return jax.lax.fori_loop(0, self.consts.FIELD_HEIGHT, inner_loop, carry)

            current_raster = raster
            return jax.lax.fori_loop(0, self.consts.FIELD_WIDTH, outer_loop, current_raster)
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
            lambda x: jr.render_at(x, current_player_choice[0], current_player_choice[1], frame_player),
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

        frame_player_digit = jr.get_sprite_frame(self.PLAYER_DIGIT_SPRITES, first_digit_player_score)
        raster = jr.render_at(raster, first_digit_player_x, digit_render_y, frame_player_digit)
        frame_player_digit = jr.get_sprite_frame(self.PLAYER_DIGIT_SPRITES, second_digit_player_score)
        raster = jax.lax.cond(
            second_digit_player_score == 0,
            lambda _: raster,
            lambda _: jr.render_at(raster, second_digit_player_x, digit_render_y, frame_player_digit),
            operand=None
        )
        frame_player_digit = jr.get_sprite_frame(self.ENEMY_DIGIT_SPRITES, first_digit_enemy_score)
        raster = jr.render_at(raster, first_digit_enemy_x, digit_render_y, frame_player_digit)
        frame_player_digit = jr.get_sprite_frame(self.ENEMY_DIGIT_SPRITES, second_digit_enemy_score)
        raster = jax.lax.cond(
            second_digit_enemy_score == 0,
            lambda _: raster,
            lambda _: jr.render_at(raster, second_digit_enemy_x, digit_render_y, frame_player_digit),
            operand=None
        )

        return raster
