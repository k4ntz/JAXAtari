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

from jaxatari.rendering import atraJaxis as aj
from jaxatari.environment import JaxEnvironment

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
NOOP = 0
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4
PLACE = 5
DIFFICULTY = 6
RESET = 7

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
    if keys[pygame.K_1]:
        return jnp.array(DIFFICULTY)
    elif keys[pygame.K_2]:
        return jnp.array(RESET)
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
    render_helper_field_before_player_step: chex.Array
    render_helper_field_after_player_step: chex.Array

class OthelloObservation(NamedTuple):
    field: Field
    player_score: jnp.ndarray
    enemy_score: jnp.ndarray

class OthelloInfo(NamedTuple):
    time: jnp.ndarray
    all_rewards: chex.Array


@jax.jit
def has_human_player_decided_field(field_choice_player, action: chex.Array):
    is_place = jnp.equal(action, PLACE)
    is_up = jnp.equal(action, UP)
    is_right = jnp.equal(action, RIGHT)
    is_down = jnp.equal(action, DOWN)
    is_left = jnp.equal(action, LEFT)
    
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
                        lambda x: (False, x),
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
def enemy_step(state):


    return 0, 0


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
                )
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
    
    #jax.debug.print(str(curr_state.player_score))
    #TODO make bot only run, when player move is valid && handle return
    get_bot_move(curr_state.field, curr_state.difficulty, curr_state.player_score, curr_state.enemy_score)

    return jax.lax.cond(
        curr_state.field.field_color[x, y] == FieldColor.EMPTY,
        lambda x: if_empty(x),
        lambda x: (False, x),
        curr_state
    )

@jax.jit
def get_bot_move(game_field: Field, difficulty: chex.Array, player_score: chex.Array, enemy_score: chex.Array):
    game_score = player_score+enemy_score

    best_move_score = -2147483648
    list_of_chosen_moves = None

    list_of_all_moves = jnp.arange(64)

    #Iterate over the game field using vmap
    vectorized_compute_score_of_tiles = jax.vmap(compute_score_of_tiles, in_axes=(0, None, None))
    list_of_all_move_values = vectorized_compute_score_of_tiles(list_of_all_moves, game_field, game_score)
    #TODO Determine best move amongst list_of_all_move_values
    return 0

def compute_score_of_tiles(i, game_field, game_score):
    # Decode tile position
    tile_x = jnp.floor_divide(i, 8)
    tile_y = jnp.mod(i, 8)

    # If tile is already taken, set invalid move (return very low score)
    score_of_tile = jax.lax.cond(
        game_field.field_color[tile_x, tile_y] != FieldColor.EMPTY,
        lambda _: jnp.int32(-2147483648),  
        lambda _: jnp.int32(compute_score_of_tile_1(tile_x, tile_y,game_field, game_score)),  
        None
    )
    return score_of_tile

def compute_score_of_tile_1(tile_x, tile_y, game_field, game_score):
    #compute flipped tiles by each direction
    list_of_all_directions = jnp.arange(8)
    vectorised_flipped_tiles_by_direction = jax.vmap(compute_flipped_tiles_by_direction,in_axes=(0, None, None, None))
    flipped_tiles_by_direction = vectorised_flipped_tiles_by_direction(list_of_all_directions, tile_x, tile_y, game_field)

    flipped_tiles = jnp.sum(flipped_tiles_by_direction)
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
            jnp.logical_or(game_score < 18, game_score > 41),  
            lambda _: -flipped_tiles - 1,  
            lambda _: flipped_tiles,  
            None
        ),
        None
    )
    return score_of_tile
    


def compute_flipped_tiles_by_direction(i, tile_x, tile_y, game_field):
    #TODO implement 

    return 0


class JaxOthello(JaxEnvironment[OthelloState, OthelloObservation, OthelloInfo]):
    def __init__(self, frameskip: int = 0, reward_funcs: list[callable]=None):
        super().__init__()
        self.frameskip = frameskip + 1
        self.frame_stack_size = 4
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.action_set = set(range(64)) | {NOOP}
        self.obs_size = 130


    def reset(self) -> OthelloState:
        """ Reset the game state to the initial state """
        field_color_init = jnp.full((8, 8), FieldColor.EMPTY.value, dtype=jnp.int32)
        field_color_init = field_color_init.at[3,3].set(FieldColor.BLACK.value)
        field_color_init = field_color_init.at[4,3].set(FieldColor.WHITE.value)
        field_color_init = field_color_init.at[3,4].set(FieldColor.WHITE.value)
        field_color_init = field_color_init.at[4,4].set(FieldColor.BLACK.value)
        
        #################### Testing
        
        # field_color_init = field_color_init.at[0,0].set(FieldColor.WHITE.value)
        # field_color_init = field_color_init.at[1,1].set(FieldColor.BLACK.value)
        # field_color_init = field_color_init.at[2,2].set(FieldColor.BLACK.value)
        # field_color_init = field_color_init.at[3,3].set(FieldColor.EMPTY.value)
        # field_color_init = field_color_init.at[4,4].set(FieldColor.BLACK.value)
        # field_color_init = field_color_init.at[5,5].set(FieldColor.BLACK.value)
        
        
        
        #######################

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
            render_helper_field_before_player_step = field_color_init,
            render_helper_field_after_player_step = field_color_init
        )
        initial_obs = self._get_observation(state)
        return state, initial_obs

    @partial(jax.jit, static_argnums=0, static_argnames=["is_human"])
    def step(self, state: OthelloState, action: chex.Array, is_human: bool) -> Tuple[OthelloState, OthelloObservation, float, bool, OthelloInfo]:
        # human player has actions like moving the "cursor" to decide which empty field they want the disc to be placed
        # an agent decides directly with field id 0 - 63.
    
        is_human_condition = jax.lax.convert_element_type(is_human, bool)
        new_field_choice = jax.lax.cond(
            is_human_condition,  
            lambda _: has_human_player_decided_field(state.field_choice_player, action), 
            lambda _: (True, state.field_choice_player),  
            operand=None
        )
        decided, new_field_choice = new_field_choice  # 2D Array new_field_choice[i, j]

        # if agent is activated, convert his action into 2D array new_field_choice
        def convert_1d_id_to_2d(action):
            return jnp.array([action // FIELD_WIDTH, action % FIELD_HEIGHT])

        new_field_choice = jax.lax.cond(
            is_human_condition,
            lambda _: new_field_choice,
            lambda x: convert_1d_id_to_2d(x),
            action
        ) 
        state = state._replace(field_choice_player=new_field_choice)

        # now human player and agent are on the same "page"
        # difference: decided must be True for human player to place his disc
        # check if the new_field_choice is a valid option
        valid_choice, new_state = jax.lax.cond(
            decided,
            lambda _: field_step(new_field_choice, state, True),
            lambda _: (False, state),
            operand=None
        )

        new_state = jax.lax.cond(
            valid_choice,
            lambda _: new_state._replace(render_helper_field_before_player_step=state.field.field_color),
            lambda _: new_state._replace(render_helper_field_before_player_step=new_state.field.field_color),
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

            key, subkey = jax.random.split(key)  # neuen Key erzeugen
            rand_vals = jax.random.randint(subkey, shape=(2,), minval=0, maxval=7)

            valid_choice, new_state = field_step(rand_vals, state, False)

            jax.debug.print("rand_vals: {} {}", rand_vals[0], rand_vals[1])

            return jax.lax.cond(
                valid_choice,
                lambda _: (True, new_state, key),
                lambda _: (False, state, key),
                operand=None
            )


        key = jax.random.PRNGKey(0)
        initial_x_y = (False, new_state, key)
        valid_choice, final__step_state, _ = jax.lax.cond(
            valid_choice, 
            lambda _: jax.lax.while_loop(condition_fun, body_fun, initial_x_y),
            lambda _: (valid_choice, new_state, key),
            operand=None
        )


        return final__step_state, None, 0.0, False, None        

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: OthelloState):
        return OthelloObservation(
            field=state.field,
            enemy_score=state.enemy_score,
            player_score=state.player_score
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
    # PLAYER_DIGIT_SPRITES = aj.load_and_pad_digits(
    #     os.path.join(MODULE_DIR, "sprites/othello/number_{}_player.npy"),
    #     num_chars=1,
    # )
    # ENEMY_DIGIT_SPRITES = aj.load_and_pad_digits(
    #     os.path.join(MODULE_DIR, "sprites/othello/number_{}_enemy.npy"),
    #     num_chars=1,
    # )

    number_player = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/othello/number_4_player.npy"), transpose=True)
    number_enemy = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/othello/number_4_enemy.npy"), transpose=True)
    PLAYER_DIGIT_SPRITES = jnp.expand_dims(number_player, axis=0)
    ENEMY_DIGIT_SPRITES = jnp.expand_dims(number_enemy, axis=0)
    return (
        SPRITE_BG,
        SPRITE_PLAYER,
        SPRITE_ENEMY,
        PLAYER_DIGIT_SPRITES,
        ENEMY_DIGIT_SPRITES
    )

@jax.jit
def render_point_of_disc(id):
    return jnp.array([22 + 22 * id[0], 18 + 16 * id[1]], dtype=jnp.int32)


class Renderer_AtraJaxisOthello:
    def __init__(self):
        (
            self.SPRITE_BG,
            self.SPRITE_PLAYER,
            self.SPRITE_ENEMY,
            self.PLAYER_DIGIT_SPRITES,
            self.ENEMY_DIGIT_SPRITES,
        ) = load_sprites()

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state, counter):
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
        


        # get the index of the white disc which was putted on the last game step
        # this function will be used to do better rendering
        # returns False rendering is made also if a game step is not finished jet
        # idx_before_game_step = -1
        # idy_before_game_step = -1
        # game_step_is_changed = False
        # def get_index_white_disc_was_putted_last_step():
        #     def outer_loop(i, value):
        #         def inner_loop(j, value):
        #             condition = jnp.logical_and(
        #                 state.render_helper_field_before_player_step[i, j] == FieldColor.EMPTY,
        #                 state.field.field_color[i, j] == FieldColor.WHITE
        #             )
        #             return jax.lax.cond(
        #                 condition,
        #                 lambda _: (i,j,True),
        #                 lambda _: (value[0], value[1], value[2]),
        #                 operand=None
        #             )
        #         return jax.lax.fori_loop(0, FIELD_WIDTH, inner_loop, (value[0], value[1], value[2]))
        #     return jax.lax.fori_loop(0, FIELD_HEIGHT, outer_loop, (idx_before_game_step, idy_before_game_step, game_step_is_changed))

        # idx_before_game_step, idy_before_game_step, game_step_is_changed = get_index_white_disc_was_putted_last_step()
        
        # jax.debug.print("{}, id: {} {}", game_step_is_changed, idx_before_game_step, idy_before_game_step)

        # # Render all fixed discs from the last step
        # def set_discs_from_last_step_to_the_raster(raster, field_color):
        #     def outer_loop(i, carry):
        #         def inner_loop(j, carry):
        #             raster = carry
        #             color = field_color[i, j]
        #             render_point = render_point_of_disc(jnp.array([i,j], dtype=jnp.int32))

        #             return jax.lax.cond(
        #                 color == FieldColor.EMPTY, 
        #                 lambda x: raster,
        #                 lambda x: jax.lax.cond(
        #                     color == FieldColor.WHITE,
        #                     lambda x: aj.render_at(raster, render_point[0], render_point[1], frame_player),
        #                     lambda x: aj.render_at(raster, render_point[0], render_point[1], frame_enemy),
        #                     x
        #                 ),
        #                 color
        #             )

        #         return jax.lax.fori_loop(0, FIELD_HEIGHT, inner_loop, carry)

        #     current_raster = raster
        #     return jax.lax.fori_loop(0, FIELD_WIDTH, outer_loop, current_raster)
        # raster = set_discs_from_last_step_to_the_raster(raster, state.render_helper_field_before_player_step)

        # # Now render new discs in kind an "animation" one by one
        # def set_new_disc_one_by_one_player(raster, field_color):
        #     x, y = idx_before_game_step, idy_before_game_step
        #     render_point = render_point_of_disc(jnp.array([x,y], dtype=jnp.int32))
        #     raster = aj.render_at(raster, render_point[0], render_point[1], frame_player)

        #     def animate_vertical_line(i, raster):
        #         render_point = render_point_of_disc(jnp.array([i,y], dtype=jnp.int32))

        #         return raster
            
            
            
        #     return raster
        # raster = jax.lax.cond(game_step_is_changed, lambda _: set_new_disc_one_by_one_player(raster, state.field.field_color), lambda _: raster, operand=None)

        
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

        return raster


    @partial(jax.jit, static_argnums=(0,))
    def render_disc_ping(self, state, counter):
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

        # Render the disc which shows the current player choice -> is not a fixed disc
        current_player_choice = render_point_of_disc(state.field_choice_player)
        raster = jax.lax.cond(
            counter % 2 == 0,
            lambda x: aj.render_at(x, current_player_choice[0], current_player_choice[1], frame_player),
            lambda x: raster,
            raster
        ) 

        # render scores
        raster = aj.render_at(raster, 0, 0, aj.get_sprite_frame(self.PLAYER_DIGIT_SPRITES, 0))

        return raster


if __name__ == "__main__":
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Othello Game")

    game = JaxOthello(frameskip=1)

    # Create the JAX renderer
    renderer = Renderer_AtraJaxisOthello()

    # get jitted functions
    jitted_step = jax.jit(game.step)
    jitted_reset = jax.jit(game.reset)

    curr_state, obs = jitted_reset()

    # Game Loop
    running = True
    frameskip = 120
    counter = 1

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False     

        if counter % frameskip == 0:
            action = get_human_action()
            curr_state, obs, reward, done, info = jitted_step(curr_state, action, is_human=True)
            # Render and display
            raster = renderer.render(curr_state, counter)
            aj.update_pygame(screen, raster, 3, WIDTH, HEIGHT)

        raster = renderer.render_disc_ping(curr_state, counter)
        aj.update_pygame(screen, raster, 3, WIDTH, HEIGHT)
        counter += 1

    pygame.quit()