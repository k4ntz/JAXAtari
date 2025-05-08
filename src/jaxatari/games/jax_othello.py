import os
from functools import partial
from typing import NamedTuple, Tuple
import jax.lax
import jax.numpy as jnp
import chex
import pygame
import enum
from gymnax.environments import spaces

from jaxatari.rendering import atraJaxis as aj
from jaxatari.environment import JaxEnvironment


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

        jax.debug.print("UP")

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

        jax.debug.print("x = {}, y = {}", x, y)
        # upper disc - are they flippable?
        discs_flippable = False
        break_cond = False
        def loop_upper_discs(i, dummy_state):
            idx = x - i - 1

            def empty_field(dummy_state):
                break_cond = True
                return dummy_state

            def friendly_field(dummy_state):
                discs_flippable = True
                valid_choice = True
                break_cond = True
                return dummy_state

            def enemy_field(dummy_state):
                dummy_state = dummy_state._replace(
                    field=dummy_state.field._replace(
                        field_color=dummy_state.field.field_color.at[idx, y].set(friendly_color)
                    )
                )
                return dummy_state

            return jax.lax.cond(
                break_cond,
                lambda _: dummy_state,
                lambda _: jax.lax.cond(
                    dummy_state.field.field_color[idx, y] == FieldColor.EMPTY,
                    lambda _: empty_field(dummy_state),
                    lambda _: jax.lax.cond(
                        dummy_state.field.field_color[idx, y] == friendly_color,
                        lambda _: friendly_field(dummy_state),
                        lambda _: enemy_field(dummy_state),
                        operand=None
                    ),
                    operand=None
                ),
                operand=None
            )
        new_state = jax.lax.fori_loop(0, x, loop_upper_discs, curr_state)
        

        field_color_init = jnp.full((8, 8), FieldColor.EMPTY.value, dtype=jnp.int32)
        field_color_init = field_color_init.at[3,3].set(FieldColor.BLACK.value)
        field_color_init = field_color_init.at[4,3].set(FieldColor.WHITE.value)
        field_color_init = field_color_init.at[3,4].set(FieldColor.WHITE.value)
        field_color_init = field_color_init.at[4,4].set(FieldColor.BLACK.value)
        new_state2 = OthelloState(
            player_score = jnp.array(2).astype(jnp.int32),
            enemy_score = jnp.array(2).astype(jnp.int32),
            step_counter =jnp.array(0).astype(jnp.int32),
            field = Field(
                field_id = jnp.arange(64, dtype=jnp.int32).reshape((8,8)),
                field_color = field_color_init
            ),
            field_choice_player = curr_state.field_choice_player,
            difficulty = jnp.array(1).astype(jnp.int32)
        )
        return valid_choice, new_state

    return jax.lax.cond(
        curr_state.field.field_color[x, y] == FieldColor.EMPTY,
        lambda x: if_empty(x),
        lambda x: (False, x),
        curr_state
    )


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

        state = OthelloState(
            player_score = jnp.array(2).astype(jnp.int32),
            enemy_score = jnp.array(2).astype(jnp.int32),
            step_counter =jnp.array(0).astype(jnp.int32),
            field = Field(
                field_id = jnp.arange(64, dtype=jnp.int32).reshape((8,8)),
                field_color = field_color_init
            ),
            field_choice_player = jnp.array([7, 7], dtype=jnp.int32),
            difficulty = jnp.array(1).astype(jnp.int32)
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

        

        
        # create new state here:
        field_color_init = jnp.full((8, 8), FieldColor.EMPTY.value, dtype=jnp.int32)
        field_color_init = field_color_init.at[3,3].set(FieldColor.BLACK.value)
        field_color_init = field_color_init.at[4,3].set(FieldColor.WHITE.value)
        field_color_init = field_color_init.at[3,4].set(FieldColor.WHITE.value)
        field_color_init = field_color_init.at[4,4].set(FieldColor.BLACK.value)
        new_state2 = OthelloState(
            player_score = jnp.array(2).astype(jnp.int32),
            enemy_score = jnp.array(2).astype(jnp.int32),
            step_counter =jnp.array(0).astype(jnp.int32),
            field = Field(
                field_id = jnp.arange(64, dtype=jnp.int32).reshape((8,8)),
                field_color = field_color_init
            ),
            field_choice_player = new_field_choice,
            difficulty = jnp.array(1).astype(jnp.int32)
        )

        return new_state, None, 0.0, False, None        


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
    #     os.path.join(MODULE_DIR, "sprites/pong/player_score_{}.npy"),
    #     num_chars=10,
    # )
    # ENEMY_DIGIT_SPRITES = aj.load_and_pad_digits(
    #     os.path.join(MODULE_DIR, "sprites/pong/enemy_score_{}.npy"),
    #     num_chars=10,
    # )

    return (
        SPRITE_BG,
        SPRITE_PLAYER,
        SPRITE_ENEMY,
        # PLAYER_DIGIT_SPRITES,
        # ENEMY_DIGIT_SPRITES
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
            # self.PLAYER_DIGIT_SPRITES,
            # self.ENEMY_DIGIT_SPRITES,
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

        # Render all fixed discs
        def set_discs_to_the_raster(raster, field_color):
            def outer_loop(i, carry):
                def inner_loop(j, carry):
                    raster = carry
                    color = field_color[i, j]
                    render_point = render_point_of_disc(jnp.array([i,j], dtype=jnp.int32))

                    def set_value(c):
                        return raster.at[i, j].set(c)

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
    frameskip = 180
    counter = 1

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False     

        if counter % frameskip == 0:
            action = get_human_action()
            curr_state, obs, reward, done, info = jitted_step(curr_state, action, is_human=True)
            # print(curr_state.field.field_color[4,3])


        # Render and display
        raster = renderer.render(curr_state, counter)
        aj.update_pygame(screen, raster, 3, WIDTH, HEIGHT)

        counter += 1

    pygame.quit()