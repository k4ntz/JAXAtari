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
WIDTH = 160
HEIGHT = 210
# endregion

# P12 - F8 - S8 - F8 - S8 - F8 - S8 - F8 - S8 - F8 - S8 - F8 - S8 - F8 - S8 - F8 - S8 - F8 - P12 = 160
# y=(7*8)+(6*4)

# region Pygame Constants
# Pygame window dimensions
SCALING_FACTOR = 3
WINDOW_WIDTH = WIDTH * SCALING_FACTOR
WINDOW_HEIGHT = HEIGHT * SCALING_FACTOR
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

# region Player Status
# Define player status (these are also used as field types)
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

# region Field Padding and Gaps
# Define field padding and gaps
FIELD_PADDING_LEFT = 12
FIELD_PADDING_TOP = 11
FIELD_GAP_X = 8
FIELD_GAP_Y = 4
FIELD_WIDTH = FIELD_HEIGHT = 8
NUMBER_WIDTH = 12
NUMBER_HEIGHT = 5
# endregion

# region Game Constants
# Define game constants
NUM_BOMBS = 3
NUM_NUMBER_CLUES = 29
NUM_DIRECTION_CLUES = 29


# endregion


class FlagCaptureState(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    time: chex.Array
    is_checking: chex.Array
    score: chex.Array
    field: chex.Array  # Shape (9, 7)
    rng_key: chex.PRNGKey


class PlayerEntity(NamedTuple):
    x: chex.Array
    y: chex.Array
    width: chex.Array
    height: chex.Array
    status: chex.Array


class FlagCaptureObservation(NamedTuple):
    player: PlayerEntity
    score: chex.Array


class FlagCaptureInfo(NamedTuple):
    time: chex.Array
    score: chex.Array
    all_rewards: chex.Array


def generate_field(field, rng_key, n_bombs, n_number_clues, n_direction_clues):
    """
    Generates a game field with a flag, bombs, number clues, and direction clues.

    Args:
        field: A 2D array representing the game field.
        n_bombs: Number of bombs to place on the field.
        n_number_clues: Number of number clues to place on the field.
        n_direction_clues: Number of direction clues to place on the field.

    Returns:
        field: A 2D array representing the game field with the placed items.
    """

    placed_bombs = 0
    placed_number_clues = 0
    placed_direction_clues = 0

    # Place the flag at a random position
    jax.random.split(rng_key, 1)

    # Get random position for the flag
    flag_x = jax.random.randint(rng_key, (1,), 0, NUM_FIELDS_X)
    flag_y = jax.random.randint(rng_key, (1,), 0, NUM_FIELDS_Y)

    # Place the flag on the field
    field = field.at[flag_x, flag_y].set(PLAYER_STATUS_FLAG)

    return field


@jax.jit
def player_step(player_x, player_y, is_checking, action):
    """
    Updates the player's position and state based on the action taken.

    """
    # check if the player is firing(checking). This is any action like FIRE, UPFIRE, DOWNFIRE, etc.
    # do this by checking if the action is 1 (FIRE) or between 10 and 17 (UPFIRE, DOWNFIRE, etc.)
    new_is_checking = jax.lax.cond(jnp.logical_or(jnp.equal(action, FIRE),jnp.logical_and(jnp.greater_equal(action, UPRIGHTFIRE), jnp.less_equal(action, DOWNLEFTFIRE))),lambda : 1, lambda: 0)
    # check if the player is moving upwards. This is any action like UP, UPRIGHT, UPLEFT, UPFIRE, UPRIGHTFIRE, UPLEFTFIRE
    is_up = jnp.logical_or(jnp.equal(action, UP), jnp.logical_or(jnp.equal(action, UPFIRE), jnp.logical_or(jnp.equal(action, UPRIGHT), jnp.logical_or(jnp.equal(action, UPLEFT), jnp.logical_or(jnp.equal(action, UPRIGHTFIRE), jnp.equal(action, UPLEFTFIRE))))))
    is_down = jnp.logical_or(jnp.equal(action, DOWN), jnp.logical_or(jnp.equal(action, DOWNFIRE), jnp.logical_or(jnp.equal(action, DOWNRIGHT), jnp.logical_or(jnp.equal(action, DOWNLEFT), jnp.logical_or(jnp.equal(action, DOWNRIGHTFIRE), jnp.equal(action, DOWNLEFTFIRE))))))
    is_left = jnp.logical_or(jnp.equal(action, LEFT), jnp.logical_or(jnp.equal(action, UPLEFT), jnp.logical_or(jnp.equal(action, DOWNLEFT), jnp.logical_or(jnp.equal(action, LEFTFIRE), jnp.logical_or(jnp.equal(action, UPLEFTFIRE), jnp.equal(action, DOWNLEFTFIRE))))))
    is_right = jnp.logical_or(jnp.equal(action, RIGHT), jnp.logical_or(jnp.equal(action, UPRIGHT), jnp.logical_or(jnp.equal(action, DOWNRIGHT), jnp.logical_or(jnp.equal(action, RIGHTFIRE), jnp.logical_or(jnp.equal(action, UPRIGHTFIRE), jnp.equal(action, DOWNRIGHTFIRE))))))

    # if player is moving down add 1 to player_y
    new_player_y = jax.lax.cond(is_down, lambda: player_y + 1, lambda: player_y)
    # if player is moving up subtract 1 from player_y
    new_player_y = jax.lax.cond(is_up, lambda: player_y - 1, lambda: new_player_y)
    # if player is moving left subtract 1 from player_x
    new_player_x = jax.lax.cond(is_left, lambda: player_x - 1, lambda: player_x)
    # if player is moving right add 1 to player_x
    new_player_x = jax.lax.cond(is_right, lambda: player_x + 1, lambda: new_player_x)
    # modulo the player_x and player_y to be on the field
    new_player_x = jnp.mod(new_player_x, NUM_FIELDS_X)
    new_player_y = jnp.mod(new_player_y, NUM_FIELDS_Y)

    return new_player_x, new_player_y, new_is_checking


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

    def reset(self, key: jax.random.PRNGKey = jax.random.PRNGKey(187)) -> Tuple[
        FlagCaptureObservation, FlagCaptureState]:
        """
        Resets the game state to the initial state.
        Returns the initial state and the reward (i.e. 0)
        """
        generated_field = generate_field(jnp.zeros((NUM_FIELDS_X, NUM_FIELDS_Y), dtype=jnp.int32), key, NUM_BOMBS,
                                         NUM_NUMBER_CLUES, NUM_DIRECTION_CLUES)
        state = FlagCaptureState(
            player_x=jnp.array(2).astype(jnp.int32),
            player_y=jnp.array(4).astype(jnp.int32),
            time=jnp.array(75).astype(jnp.int32),
            score=jnp.array(28).astype(jnp.int32),
            is_checking=jnp.array(1).astype(jnp.int32),
            field=generated_field,
            rng_key=key,
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
                x=state.player_x,  # TODO this is currently 0-9 not screenspace
                y=state.player_y,  # TODO this is currently 0-7 not screenspace
                width=jnp.array(FIELD_WIDTH).astype(jnp.int32),
                height=jnp.array(FIELD_HEIGHT).astype(jnp.int32),
                status=jnp.array(PLAYER_STATUS_ALIVE).astype(jnp.int32),
            ),
            score=state.score
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: FlagCaptureState, action: chex.Array) -> Tuple[
        FlagCaptureObservation, FlagCaptureState, float, bool, FlagCaptureInfo]:

        new_player_x, player_speed_y, new_is_checking = player_step(
            state.player_x,
            state.player_y,
            state.is_checking,
            action,
        )

        new_state = FlagCaptureState(
            player_x=new_player_x,
            player_y=player_speed_y,
            time=state.time - 1,
            is_checking=new_is_checking,
            score=state.score,
            field=state.field,
            rng_key=state.rng_key,
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
        return state.time < 0


def load_sprites():
    """Load all sprites required for Flag Capture rendering."""
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

    background = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/flagcapture/background.npy"), transpose=True)

    # Convert all sprites to the expected format (add frame dimension)
    SPRITE_BG = jnp.expand_dims(background, axis=0)
    SPRITE_PLAYER = aj.load_and_pad_digits("./sprites/flagcapture/player_states/player_{}.npy", num_chars=19)
    SPRITE_SCORE = aj.load_and_pad_digits("./sprites/flagcapture/green_digits/{}.npy", num_chars=10)
    SPRITE_TIMER = aj.load_and_pad_digits("./sprites/flagcapture/red_digits/{}.npy", num_chars=10)

    return (
        SPRITE_BG,
        SPRITE_PLAYER,
        SPRITE_SCORE,
        SPRITE_TIMER,
    )


class FlagCaptureRenderer(AtraJaxisRenderer):
    def __init__(self):
        (
            self.SPRITE_BG,
            self.SPRITE_PLAYER,
            self.SPRITE_SCORE,
            self.SPRITE_TIMER,
        ) = load_sprites()

    @partial(jax.jit, static_argnums=(0,))
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

        frame_bg = aj.get_sprite_frame(self.SPRITE_BG, 0)
        raster = aj.render_at(raster, 0, 0, frame_bg)

        # Draw the player
        player_x = FIELD_PADDING_LEFT + (state.player_x * FIELD_WIDTH) + (state.player_x * FIELD_GAP_X)
        player_y = FIELD_PADDING_TOP + (state.player_y * FIELD_HEIGHT) + (state.player_y * FIELD_GAP_Y)

        # Wenn is_checking == 1, dann ist die Spieler Sprite >= 1 (Bombe, Flag, Zahl, ect.)
        # Die genaue Sprite wird durch den wert von field[x][y] bestimmt
        # Ansonsten ist die Spieler Sprite == 0 aka Männchen
        raster = jax.lax.cond(jax.lax.eq(state.is_checking, 0),
                              lambda: aj.render_at(raster, player_x, player_y, self.SPRITE_PLAYER[0]),
                              lambda: aj.render_at(raster, player_x, player_y,
                                                   self.SPRITE_PLAYER[state.field[player_x][player_y]]),
                              )

        raster = render_header(state.score, raster, self.SPRITE_SCORE, 32, 16, 3)
        raster = render_header(state.time, raster, self.SPRITE_TIMER, 112, 96, 3)

        return raster


def render_header(number, raster, sprites, single_digit_x, double_digit_x, y):
    # 1. Get digit arrays (always 2 digits)
    digits = aj.int_to_digits(number, max_digits=2)

    # 2. Determine parameters for timer rendering using jax.lax.select
    is_single_digit = number < 10
    start_index = jax.lax.select(is_single_digit, 1, 0)  # Start at index 1 if single, 0 if double
    num_to_render = jax.lax.select(is_single_digit, 1, 2)  # Render 1 digit if single, 2 if double
    # Adjust X position: If single digit, center it slightly by moving right by one spacing
    render_x = jax.lax.select(is_single_digit,
                              single_digit_x,
                              double_digit_x)

    # 3. Render player score using the selective renderer
    raster = aj.render_label_selective(raster, render_x, 3,
                                       digits, sprites,
                                       start_index, num_to_render,
                                       spacing=16)

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
