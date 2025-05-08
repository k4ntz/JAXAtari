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
from jaxatari.environment import JaxEnvironment

#
# by Tim Morgner and Jan Larionow
#

# region Constants
# Constants for game environment
WIDTH = 210
HEIGHT = 160
# endregion

# region Pygame Constants
# Pygame window dimensions
WINDOW_WIDTH = 210 * 3
WINDOW_HEIGHT = 160 * 3
# endregion

# region Action Space
# Define action space
NOOP = 0
FIRE = 1
UP = 2
RIGHT = 3
LEFT = 4
DOWN = 5
UPRIGHT = 6
UPLEFT = 7
DOWNRIGHT = 8
DOWNLEFT = 9
UPFIRE = 10
RIGHTFIRE = 11
LEFTFIRE = 12
DOWNFIRE = 13
UPRIGHTFIRE = 14
UPLEFTFIRE = 15
DOWNRIGHTFIRE = 16
DOWNLEFTFIRE = 17
# endregion

# region Field Types
# Define field types
FIELD_TYPE_FLAG = 0
FIELD_TYPE_BOMB = 1
FIELD_TYPE_NUMBER_CLUE = 2
FIELD_TYPE_DIRECTION = 3
# endregion

# region Player Status
# Define player status
PLAYER_STATUS_ALIVE = 0
PLAYER_STATUS_BOMB = 1
PLAYER_STATUS_FLAG = 2
PLAYER_STATUS_NUMBER_1 = 3
PLAYER_STATUS_NUMBER_2 = 4
PLAYER_STATUS_NUMBER_3 = 5
PLAYER_STATUS_NUMBER_4 = 6
PLAYER_STATUS_NUMBER_5 = 7
PLAYER_STATUS_NUMBER_6 = 8
PLAYER_STATUS_NUMBER_7 = 9
PLAYER_STATUS_NUMBER_8 = 10
PLAYER_STATUS_DIRECTION_UP = 11
PLAYER_STATUS_DIRECTION_RIGHT = 12
PLAYER_STATUS_DIRECTION_DOWN = 13
PLAYER_STATUS_DIRECTION_LEFT = 14
PLAYER_STATUS_DIRECTION_UPRIGHT = 15
PLAYER_STATUS_DIRECTION_UPLEFT = 16
PLAYER_STATUS_DIRECTION_DOWNRIGHT = 17
PLAYER_STATUS_DIRECTION_DOWNLEFT = 18
_____________________________________________________________ = 0  # Separator for Structure view #TODO REMOVE

# endregion

# region Game Field Constants
# Define game field constants
NUM_FIELDS_X = 9
NUM_FIELDS_Y = 7
# endregion

# region Colors
# Background color and object colors
BACKGROUND_COLOR = 55, 84, 168
PLAYER_COLOR = SCORE_COLOR = 132, 140, 76
TIMER_COLOR = 192, 88, 88
# endregion

# region Field Padding and Gaps
# Define field padding and gaps
FIELD_PADDING_LEFT = 48
FIELD_PADDING_TOP = 44
FIELD_GAP_X = 32
FIELD_GAP_Y = 16
FIELD_WIDTH = FIELD_HEIGHT = 32
NUMBER_WIDTH = 16
NUMBER_HEIGHT = 28


class PlayerEntity(NamedTuple):
    x: chex.Array
    y: chex.Array
    width: chex.Array
    height: chex.Array
    status: chex.Array


class FlagCaptureState(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    time: chex.Array
    score: chex.Array
    field: chex.Array  # Shape (9, 7)


class FlagCaptureObservation(NamedTuple):
    player: PlayerEntity
    score: chex.Array


class FlagCaptureInfo(NamedTuple):
    time: chex.Array
    score: chex.Array
    all_rewards: chex.Array


class JaxFlagCapture(JaxEnvironment[FlagCaptureState, FlagCaptureObservation, FlagCaptureInfo]):
    def __init__(self, reward_funcs: list[callable] = None):
        super().__init__()
        self.frame_stack_size = 4
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.action_set = {
            NOOP,
            FIRE,
            UP,
            RIGHT,
            LEFT,
            DOWN,
            UPRIGHT,
            UPLEFT,
            DOWNRIGHT,
            DOWNLEFT,
            UPFIRE,
            RIGHTFIRE,
            LEFTFIRE,
            DOWNFIRE,
            UPRIGHTFIRE,
            UPLEFTFIRE,
            DOWNRIGHTFIRE,
            DOWNLEFTFIRE
        }

    def reset(self, key=None) -> Tuple[FlagCaptureObservation, FlagCaptureState]:
        """
        Resets the game state to the initial state.
        Returns the initial state and the reward (i.e. 0)
        """

        # generate new random field
        #TODO
        state = FlagCaptureState(
            player_x=jnp.array(0).astype(jnp.int32),
            player_y=jnp.array(0).astype(jnp.int32),
        time=jnp.array(75).astype(jnp.int32),
        score=jnp.array(0).astype(jnp.int32),
        field=jnp.zeros((NUM_FIELDS_X, NUM_FIELDS_Y), dtype=jnp.int32)
        )
        initial_obs = self._get_observation(state)

        def expand_and_copy(x):
            x_expanded = jnp.expand_dims(x, axis=0)
            return jnp.concatenate([x_expanded] * self.frame_stack_size, axis=0)

        # Apply transformation to each leaf in the pytree
        initial_obs = jax.tree.map(expand_and_copy, initial_obs)

        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: FlagCaptureState):
        # create player
        return FlagCaptureObservation(
            player=PlayerEntity(
                x=state.player_x, #TODO this is currently 0-9 not screenspace
                y=state.player_y, #TODO this is currently 0-7 not screenspace
                width=jnp.array(FIELD_WIDTH).astype(jnp.int32),
                height=jnp.array(FIELD_HEIGHT).astype(jnp.int32),
                status=jnp.array(PLAYER_STATUS_ALIVE).astype(jnp.int32),
            ),
            score=state.score
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state:FlagCaptureState, action: chex.Array) -> Tuple[FlagCaptureObservation, FlagCaptureState, float, bool, FlagCaptureInfo]:

        new_state = FlagCaptureState(
            player_x=state.player_x,
            player_y=state.player_y,
            time=state.time,
            score=state.score,
            field=state.field,
        )

        done = self._get_done(new_state)
        env_reward = self._get_env_reward(state, new_state)
        all_rewards = self._get_all_reward(state, new_state)
        info = self._get_info(new_state, all_rewards)

        observation = self._get_observation(new_state)

        return observation, new_state, env_reward, done, info

    def get_action_space(self) -> Tuple:
        return self.action_set

    def reset_player(self):
        pass

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: FlagCaptureState, all_rewards: chex.Array) -> FlagCaptureInfo:
        return FlagCaptureInfo(time=state.time, all_rewards=all_rewards, score=state.score)

    @partial(jax.jit, static_argnums=(0,))
    def _get_env_reward(self, previous_state: FlagCaptureState, state: FlagCaptureState):
        return state.score - previous_state.score

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(self, previous_state: FlagCaptureState, state: FlagCaptureState):
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array(
            [reward_func(previous_state, state) for reward_func in self.reward_funcs]
        )
        return rewards

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: FlagCaptureState) -> bool:
        return jnp.less_equal(state.time, 0)

def load_sprites():
    """Load all sprites required for Flag Capture rendering."""
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

    background = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/flagcapture/background.npy"), transpose=True)

    # Convert all sprites to the expected format (add frame dimension)
    SPRITE_BG = jnp.expand_dims(background, axis=0)

    # print the shape of SPRITE_BG
    print("SPRITE_BG shape:", SPRITE_BG.shape)

    return (
        SPRITE_BG,
    )

class FlagCaptureRenderer(AtraJaxisRenderer):
    def __init__(self):
        (
            self.SPRITE_BG
        ) = load_sprites()

    def render(self, state):
        """
        Renders the current game state using JAX operations.

        Args:
            state: A FlagCaptureState object containing the current game state.

        Returns:
            A JAX array representing the rendered frame.
        """
        # Create empty raster with CORRECT orientation for atraJaxis framework
        # Note: For pygame, the raster is expected to be (width, height, channels)
        # where width corresponds to the horizontal dimension of the screen
        raster: jnp.ndarray = jnp.zeros((WIDTH, HEIGHT, 3))

        background_color = jnp.array(BACKGROUND_COLOR, dtype=jnp.uint8)
        player_color = jnp.array(PLAYER_COLOR, dtype=jnp.uint8)
        score_color = jnp.array(SCORE_COLOR, dtype=jnp.uint8)
        timer_color = jnp.array(TIMER_COLOR, dtype=jnp.uint8)


        # Draw the background
        # raster = raster.at[:, :, :].set(background_color)

        print("sprite:", self.SPRITE_BG)
        print("sprite shape:", self.SPRITE_BG[0].shape)
        frame_bg = aj.get_sprite_frame(self.SPRITE_BG[0], 0)
        raster = aj.render_at(raster, 0, 0, frame_bg)

        # Draw the player
        def render_player_sprite(raster_base,sprite):
            """
            Returns a funtion that draws the player sprite based on the player information passed to the returned function.

            Args:
                sprite: The sprite to be drawn.

            Returns:
                A function that takes player information and draws the sprite.
            """
            def _render_player(player):
                x = player.x
                y = player.y

                # Draw the player sprite on the raster
                raster = raster_base.at[x:x + FIELD_WIDTH, y:y + FIELD_HEIGHT,:].set(player_color)
                return raster

            return _render_player

        """
        dummysprite = jnp.array(BACKGROUND_COLOR, dtype=jnp.uint8)
        player_sprite_cases = {
            PLAYER_STATUS_ALIVE: render_player_sprite(raster,dummysprite),
            PLAYER_STATUS_BOMB: render_player_sprite(raster,dummysprite),
            PLAYER_STATUS_FLAG: render_player_sprite(raster,dummysprite),
            PLAYER_STATUS_NUMBER_1: render_player_sprite(raster,dummysprite),
            PLAYER_STATUS_NUMBER_2: render_player_sprite(raster,dummysprite),
            PLAYER_STATUS_NUMBER_3: render_player_sprite(raster,dummysprite),
            PLAYER_STATUS_NUMBER_4: render_player_sprite(raster,dummysprite),
            PLAYER_STATUS_NUMBER_5: render_player_sprite(raster,dummysprite),
            PLAYER_STATUS_NUMBER_6: render_player_sprite(raster,dummysprite),
            PLAYER_STATUS_NUMBER_7: render_player_sprite(raster,dummysprite),
            PLAYER_STATUS_NUMBER_8: render_player_sprite(raster,dummysprite),
            PLAYER_STATUS_DIRECTION_UP: render_player_sprite(raster,dummysprite),
            PLAYER_STATUS_DIRECTION_RIGHT: render_player_sprite(raster,dummysprite),
            PLAYER_STATUS_DIRECTION_DOWN: render_player_sprite(raster,dummysprite),
            PLAYER_STATUS_DIRECTION_LEFT: render_player_sprite(raster,dummysprite),
            PLAYER_STATUS_DIRECTION_UPRIGHT: render_player_sprite(raster,dummysprite),
            PLAYER_STATUS_DIRECTION_UPLEFT: render_player_sprite(raster,dummysprite),
            PLAYER_STATUS_DIRECTION_DOWNRIGHT: render_player_sprite(raster,dummysprite),
            PLAYER_STATUS_DIRECTION_DOWNLEFT: render_player_sprite(raster,dummysprite),
        }

        jax.lax.switch(
            state.player.status,
            branches=player_sprite_cases,
            operand=state.player,
        )

        # Draw the score
        score = state.score
        #TODO

        # Draw the timer
        time = state.time
        #TODO
        """

        return raster


def get_human_action() -> chex.Array:
    """
    Records human input for the game.

    Returns:
        action: int, action taken by the player (FIRE, LEFT, RIGHT, UPRIGHT, FIREDOWNRIGHT, etc.)
    """
    keys = pygame.key.get_pressed()
    # pygame.K_a  Links
    # pygame.K_d  Rechts
    # pygame.K_w  Hoch
    # pygame.K_s  Runter
    # pygame.K_SPACE  Aufdecken (fire)

    pressed_buttons = 0
    if keys[pygame.K_a]:
        pressed_buttons += 1
    if keys[pygame.K_d]:
        pressed_buttons += 1
    if keys[pygame.K_w]:
        pressed_buttons += 1
    if keys[pygame.K_s]:
        pressed_buttons += 1
    if pressed_buttons > 3:
        print("You have pressed a physically impossible combination of buttons")
        return jnp.array(NOOP)

    if keys[pygame.K_SPACE]:
        # All actions with fire
        if keys[pygame.K_w] and keys[pygame.K_a]:
            return jnp.array(UPLEFTFIRE)
        elif keys[pygame.K_w] and keys[pygame.K_d]:
            return jnp.array(UPRIGHTFIRE)
        elif keys[pygame.K_s] and keys[pygame.K_a]:
            return jnp.array(DOWNLEFTFIRE)
        elif keys[pygame.K_s] and keys[pygame.K_d]:
            return jnp.array(DOWNRIGHTFIRE)
        elif keys[pygame.K_w]:
            return jnp.array(UPFIRE)
        elif keys[pygame.K_a]:
            return jnp.array(LEFTFIRE)
        elif keys[pygame.K_d]:
            return jnp.array(RIGHTFIRE)
        elif keys[pygame.K_s]:
            return jnp.array(DOWNFIRE)
        else:
            return jnp.array(FIRE)
    else:
        # All actions without fire
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
        elif keys[pygame.K_a]:
            return jnp.array(LEFT)
        elif keys[pygame.K_d]:
            return jnp.array(RIGHT)
        elif keys[pygame.K_s]:
            return jnp.array(DOWN)
        else:
            return jnp.array(NOOP)


if __name__ == "__main__":
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Flag Capture Game")
    clock = pygame.time.Clock()

    game = JaxFlagCapture()

    # Create the JAX renderer
    renderer = FlagCaptureRenderer()

    # Get jitted functions
    jitted_step = jax.jit(game.step)
    jitted_reset = jax.jit(game.reset)

    obs, curr_state = jitted_reset()

    # Game loop
    running = True
    frame_by_frame = False
    frameskip = 1
    counter = 1

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    frame_by_frame = not frame_by_frame
            elif event.type == pygame.KEYDOWN or (
                    event.type == pygame.KEYUP and event.key == pygame.K_n
            ):
                if event.key == pygame.K_n and frame_by_frame:
                    if counter % frameskip == 0:
                        action = get_human_action()
                        obs, curr_state, reward, done, info = jitted_step(
                            curr_state, action
                        )

        if not frame_by_frame:
            if counter % frameskip == 0:
                action = get_human_action()
                obs, curr_state, reward, done, info = jitted_step(curr_state, action)

        # Render and display
        raster = renderer.render(curr_state)

        aj.update_pygame(screen, raster, 3, WIDTH, HEIGHT)

        counter += 1
        clock.tick(60)

    pygame.quit()
