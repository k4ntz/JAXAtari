import os
from functools import partial
from typing import NamedTuple, Tuple
import jax.lax
import jax.numpy as jnp
import chex
import pygame
from gymnax.environments import spaces

from jaxatari.renderers import AtraJaxisRenderer
from jaxatari.rendering import atraJaxis as aj
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action


WIDTH = 160
HEIGHT = 210

# Pygame window dimensions
WINDOW_WIDTH = 160 * 3
WINDOW_HEIGHT = 210 * 3

# Donkey Kong position
DONKEYKONG_X = 33
DONKEYKONG_Y = 14

# Girlfriend position
GIRLFRIEND_X = 62
GIRLFRIEND_Y = 17

# Life Bar positions
LEVEL_1_LIFE_BAR_1_X = 116
LEVEL_1_LIFE_BAR_2_X = 124
LEVEL_2_LIFE_BAR_1_X = 112
LEVEL_2_LIFE_BAR_2_X = 120
LIFE_BAR_Y = 23

# Hammer default position
LEVEL_1_HAMMER_X = 39
LEVEL_1_HAMMER_Y = 68
LEVEL_2_HAMMER_X = 78
LEVEL_2_HAMMER_Y = 68

# Drop Pit position
DP_LEFT_X = 52
DP_RIGHT_X = 104
DP_FLOOR_2_Y = 144
DP_FLOOR_3_Y = 116
DP_FLOOR_4_Y = 88
DP_FLOOR_5_Y = 60

# Digits position
DIGIT_Y = 7
FIRST_DIGIT_X = 96
DISTANCE_DIGIT_X = 8
NUMBER_OF_DIGITS_FOR_GAME_SCORE = 6
NUMBER_OF_DIGITS_FOR_TIMER_SCORE = 4

# Mario start position
MARIO_START_X = 49
MARIO_START_Y = 48


class DonkeyKongState(NamedTuple):
    mario_x: chex.Array
    mario_y: chex.Array


class DonkeyKongObservation(NamedTuple):
    total_score: jnp.ndarray


class DonkeyKongInfo(NamedTuple):
    time: jnp.ndarray
    all_rewards: chex.Array



class JaxDonkeyKong(JaxEnvironment[DonkeyKongState, DonkeyKongObservation, DonkeyKongInfo]):
    def __init__(self, reward_funcs: list[callable]=None):
        super().__init__()
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
            Action.DOWN
        ]
        self.obs_size = 0

    def reset(self, key=None) -> Tuple[DonkeyKongObservation, DonkeyKongState]:
        """
        Resets the game state to the initial state.
        Returns the initial state and the reward (i.e. 0)
        """

        state = DonkeyKongState(
            mario_x=jnp.array(MARIO_START_X).astype(jnp.int32),
            mario_y=jnp.array(MARIO_START_Y).astype(jnp.int32),
        )
        initial_obs = self._get_observation(state)

        return initial_obs, state

    
    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: DonkeyKongState):
        
        return DonkeyKongObservation(
            total_score = 0,
        )



def load_sprites():
    """Load all sprites required for Pong rendering."""
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Load sprites
    bg_level_1 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/donkeyKong_background_level_1.npy"), transpose=True)
    bg_level_2 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/donkeyKong_background_level_2.npy"), transpose=True)

    donkeyKong_pose_1 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/donkeyKong1.npy"), transpose=True)
    donkeyKong_pose_2 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/donkeyKong2.npy"), transpose=True)

    girlfriend = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/girlfriend.npy"), transpose=True)

    level_1_life_bar = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/level_1_life_bar.npy"), transpose=True)
    level_2_life_bar = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/level_2_life_bar.npy"), transpose=True)

    mario_standing_right = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/mario_standing_right.npy"), transpose=True)
    mario_standing_left = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/mario_standing_left.npy"), transpose=True)
    mario_jumping_right = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/mario_jumping_right.npy"), transpose=True)
    mario_jumping_left = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/mario_jumping_left.npy"), transpose=True)
    mario_walking_1_right = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/mario_walking_1_right.npy"), transpose=True)
    mario_walking_1_left = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/mario_walking_1_left.npy"), transpose=True)
    mario_walking_2_right = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/mario_walking_2_right.npy"), transpose=True)
    mario_walking_2_left = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/mario_walking_2_left.npy"), transpose=True)
    mario_climbing_right = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/mario_climbing_right.npy"), transpose=True)
    mario_climbing_left = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/mario_climbing_left.npy"), transpose=True)

    hammer_up_level_1 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/hammer_up_level_1.npy"), transpose=True)
    hammer_up_level_2 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/hammer_up_level_2.npy"), transpose=True)
    hammer_down_right_level_1 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/hammer_down_right_level_1.npy"), transpose=True)
    hammer_down_left_level_1 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/hammer_down_left_level_1.npy"), transpose=True)
    hammer_down_right_level_2 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/hammer_down_right_level_2.npy"), transpose=True)
    hammer_down_left_level_2 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/hammer_down_left_level_2.npy"), transpose=True)

    ghost = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/ghost.npy"), transpose=True)

    drop_pit = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/drop_pit.npy"), transpose=True)

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
    SPRITE_GHOST = jnp.expand_dims(ghost, axis=0)
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
        SPRITE_GHOST,
        SPRITE_DROP_PIT,
        SPRITES_BLUE_DIGITS,
        SPRITES_YELLOW_DIGITS,
    )


class DonkeyKongRenderer(AtraJaxisRenderer):
    """JAX-based Pong game renderer, optimized with JIT compilation."""

    def __init__(self):
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
            self.SPRITE_GHOST,
            self.SPRITE_DROP_PIT,
            self.SPRITES_BLUE_DIGITS,
            self.SPRITES_YELLOW_DIGITS,
        ) = load_sprites()

    
    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        """
        Renders the current game state using JAX operations.

        Args:
            state: A DonkeyKongState object containing the current game state.

        Returns:
            A JAX array representing the rendered frame.
        """

        def create_bg_raster_for_level_2_regarding_drop_pits(raster):
            frame_bg = aj.get_sprite_frame(self.SPRITES_BG, 1)
            raster = aj.render_at(raster, 0, 0, frame_bg)
            frame_drop_pit = aj.get_sprite_frame(self.SPRITE_DROP_PIT, 0)

            # some drop pits might be already triggered - in that case, drop pits at those position will not be rendered
            raster = aj.render_at(raster, DP_LEFT_X, DP_FLOOR_2_Y, frame_drop_pit)
            raster = aj.render_at(raster, DP_LEFT_X, DP_FLOOR_3_Y, frame_drop_pit)
            raster = aj.render_at(raster, DP_LEFT_X, DP_FLOOR_4_Y, frame_drop_pit)
            raster = aj.render_at(raster, DP_LEFT_X, DP_FLOOR_5_Y, frame_drop_pit)
            raster = aj.render_at(raster, DP_RIGHT_X, DP_FLOOR_2_Y, frame_drop_pit)
            raster = aj.render_at(raster, DP_RIGHT_X, DP_FLOOR_3_Y, frame_drop_pit)
            raster = aj.render_at(raster, DP_RIGHT_X, DP_FLOOR_4_Y, frame_drop_pit)
            raster = aj.render_at(raster, DP_RIGHT_X, DP_FLOOR_5_Y, frame_drop_pit)
            return raster            

        raster = jnp.zeros((WIDTH, HEIGHT, 3))

        # Background raster
        level = 1
        raster = jax.lax.cond(
            level == 1,
            lambda _: aj.render_at(raster, 0, 0, aj.get_sprite_frame(self.SPRITES_BG, 0)),
            lambda x: create_bg_raster_for_level_2_regarding_drop_pits(x),
            raster 
        )

        # DonkeyKong
        frame_donkeyKong = aj.get_sprite_frame(self.SPRITES_DONKEYKONG, 0)
        raster = aj.render_at(raster, DONKEYKONG_X, DONKEYKONG_Y, frame_donkeyKong)

        # Girlfriend
        frame_girlfriend = aj.get_sprite_frame(self.SPRITE_GIRLFRIEND, 0)
        raster = aj.render_at(raster, GIRLFRIEND_X, GIRLFRIEND_Y, frame_girlfriend)

        # Life Bars - depending if lifes are still given 
        frame_life_bar = aj.get_sprite_frame(self.SPRITES_LIFEBAR, 0)
        raster = aj.render_at(raster, LEVEL_1_LIFE_BAR_1_X, LIFE_BAR_Y, frame_life_bar)
        raster = aj.render_at(raster, LEVEL_1_LIFE_BAR_2_X, LIFE_BAR_Y, frame_life_bar)

        # Mario
        frame_mario = aj.get_sprite_frame(self.SPRITES_MARIO_STANDING, 0)
        raster = aj.render_at(raster, state.mario_x, state.mario_y, frame_mario)


        # Hammer
        frame_hammer = aj.get_sprite_frame(self.SPRITES_HAMMER_UP, 0)
        raster = aj.render_at(raster, LEVEL_1_HAMMER_X, LEVEL_1_HAMMER_Y, frame_hammer)


        # Scores
        score = 5000
        show_game_score = False
        def create_score_in_raster(i, raster):
            digit = score // (10 ** i)
            pos_x = FIRST_DIGIT_X - DISTANCE_DIGIT_X * i
            pos_y = DIGIT_Y
            return jax.lax.cond(
                show_game_score == True,
                lambda _: aj.render_at(raster, pos_x, pos_y, aj.get_sprite_frame(self.SPRITES_BLUE_DIGITS, digit)),
                lambda _: aj.render_at(raster, pos_x, pos_y, aj.get_sprite_frame(self.SPRITES_YELLOW_DIGITS, digit)),
                operand=None
            )
        raster = jax.lax.cond(
            show_game_score == True,
            lambda x: jax.lax.fori_loop(0, NUMBER_OF_DIGITS_FOR_GAME_SCORE, create_score_in_raster, x),
            lambda x: jax.lax.fori_loop(0, NUMBER_OF_DIGITS_FOR_TIMER_SCORE, create_score_in_raster, x),
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

        # Ghost
        frame_ghost = aj.get_sprite_frame(self.SPRITE_GHOST, 0)
        raster = aj.render_at(raster, 5, 165, frame_ghost)

        return raster


if __name__ == "__main__":
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Donkey Kong Game")
    clock = pygame.time.Clock()

    game = JaxDonkeyKong()

    # Create the JAX renderer
    renderer = DonkeyKongRenderer()

    # Get jitted functions
    jitted_reset = jax.jit(game.reset)

    obs, curr_state = jitted_reset()

    # Game Loop
    running = True
    while running:

        # Render and Display
        raster = renderer.render(state=curr_state)
        aj.update_pygame(screen, raster, 3, WIDTH, HEIGHT)

    pygame.quit()