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



# TODO : remove unnecessary constants

# Constants for game environment
MAX_SPEED = 12
ENEMY_STEP_SIZE = 2
WIDTH = 160
HEIGHT = 210

# Background color and object colors
BACKGROUND_COLOR = 144, 72, 17
PLAYER_COLOR = 92, 186, 92
BALL_COLOR = 236, 236, 236  # White ball
WALL_COLOR = 236, 236, 236  # White walls
SCORE_COLOR = 236, 236, 236  # White score

# Player and enemy paddle positions
PLAYER_X = 140

# Object sizes (width, height)
PLAYER_SIZE = (4, 16)
WALL_TOP_Y = 24
WALL_TOP_HEIGHT = 10
WALL_BOTTOM_Y = 194
WALL_BOTTOM_HEIGHT = 16

# Pygame window dimensions
WINDOW_WIDTH = 210 * 3
WINDOW_HEIGHT = 160 * 3

# define the positions of the state information
STATE_TRANSLATOR: dict = {
    0: "player_x",
    1: "player_y",
    2: "player_speed",
    3: "cooldown_timer",
    4: "asteroid_x",
    5: "asteroid_y",
    6: "asteroid_speed",
    7: "asteroid_alive",
    8: "letters_x",
    9: "letters_y",
    10: "letters_char",
    11: "letters_alive",
    12: "letters_speed",
    13: "current_word",
    14: "current_letter_index",
    15: "player_score",
    16: "timer",
    17: "step_counter",
    18: "buffer",
}


def get_human_action() -> list: ## TODO make this return chex.Array !
    """
    Records if multiple keys are being pressed and returns the corresponding actions.

    Returns:
        actions: A list of actions taken by the player (e.g., LEFT, RIGHT, UP, DOWN, FIRE, etc.).
    """
    keys = pygame.key.get_pressed()
    actions = []

    # Movement keys
    if keys[pygame.K_a]:
        actions.append("LEFT")
    if keys[pygame.K_d]:
        actions.append("RIGHT")
    if keys[pygame.K_w]:
        actions.append("UP")
    if keys[pygame.K_s]:
        actions.append("DOWN")

    # Firing keys with diagonal combinations
    if keys[pygame.K_SPACE]:
        if keys[pygame.K_a] and keys[pygame.K_w]:
            actions.append("UPLEFTFIRE")
        elif keys[pygame.K_d] and keys[pygame.K_w]:
            actions.append("UPRIGHTFIRE")
        elif keys[pygame.K_a] and keys[pygame.K_s]:
            actions.append("DOWNLEFTFIRE")
        elif keys[pygame.K_d] and keys[pygame.K_s]:
            actions.append("DOWNRIGHTFIRE")
        elif keys[pygame.K_a]:
            actions.append("LEFTFIRE")
        elif keys[pygame.K_d]:
            actions.append("RIGHTFIRE")
        elif keys[pygame.K_w]:
            actions.append("UPFIRE")
        elif keys[pygame.K_s]:
            actions.append("DOWNFIRE")
        else:
            actions.append("FIRE")

    # If no keys are pressed, return NOOP
    if not actions:
        actions.append("NOOP")

    return actions
    


class WordZapperState(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    player_speed: chex.Array
    cooldown_timer: chex.Array

    asteroid_x: chex.Array
    asteroid_y: chex.Array
    asteroid_speed: chex.Array
    asteroid_alive: chex.Array

    letters_x: chex.Array # letters at the top
    letters_y: chex.Array
    letters_char: chex.Array
    letters_alive: chex.Array
    letters_speed: chex.Array

    current_word: chex.Array # the actual word
    current_letter_index: chex.Array

    player_score: chex.Array
    timer: chex.Array
    step_counter: chex.Array
    buffer: chex.Array # TODO: do we need this?

class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray

class EntityBatchPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray

class WordZapperObservation(NamedTuple):
    player: EntityPosition
    asteroids: EntityBatchPosition
    letters: EntityBatchPosition

    letters_char: jnp.ndarray 
    letters_alive: jnp.ndarray  # active letters

    current_word: jnp.ndarray  # word to form
    current_letter_index: jnp.ndarray  # current position in word

    cooldown_timer: jnp.ndarray
    timer: jnp.ndarray
    player_lives: jnp.ndarray

class WordZapperInfo(NamedTuple):
    timer: jnp.ndarray
    current_word: jnp.ndarray
    game_over: jnp.ndarray


@jax.jit
def player_step(
    player_x: chex.Array,
    player_y: chex.Array,
    player_speed: chex.Array,
    cooldown_timer: chex.Array,
    action: chex.Array
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
    MOVE_UP = jnp.array([Action.UP, Action.UPFIRE, Action.UPLEFTFIRE, Action.UPRIGHTFIRE])
    MOVE_DOWN = jnp.array([Action.DOWN, Action.DOWNFIRE, Action.DOWNLEFTFIRE, Action.DOWNRIGHTFIRE])
    FIRE_ACTIONS = jnp.array([Action.FIRE, Action.LEFTFIRE, Action.RIGHTFIRE])

    move_up = jnp.any(jnp.equal(action, MOVE_UP))
    move_down = jnp.any(jnp.equal(action, MOVE_DOWN))
    is_firing = jnp.any(jnp.equal(action, FIRE_ACTIONS))

    # Movement logic
    delta_y = jnp.where(move_up, -player_speed, jnp.where(move_down, player_speed, 0))
    new_player_y = jnp.clip(player_y + delta_y, 0, HEIGHT - PLAYER_SIZE[1])

    # Firing and cooldown
    can_fire = cooldown_timer == 0
    fired = jnp.logical_and(is_firing, can_fire)
    new_cooldown_timer = jnp.where(fired, 8, jnp.maximum(cooldown_timer - 1, 0))

    return player_x, new_player_y, player_speed, new_cooldown_timer, fired

def shooting_letter(letters_x):
    pass

def scrolling_letters(letters_x, letters_speed, letters_alive):
    letters_x = letters_x - letters_speed
    wrapped_x = jnp.where(letters_x < -8, 160, letters_x)
    updated_x = jnp.where(letters_alive == 1, wrapped_x, letters_x)
    return updated_x


def enemy_step():
    #TODO
    pass


@jax.jit
def _reset_ball_after_goal():
    #TODO : give a better name 
    #TODO
    pass



def load_sprites():
    """Load all sprites required for Word Zapper rendering."""
    #TODO
    pass

class JaxWordZapper(JaxEnvironment[WordZapperState, WordZapperObservation, WordZapperInfo]) :
    def __init__(self, reward_funcs: list[callable] =None):
        super().__init__()
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.action_set = [
            Action.NOOP,
            Action.FIRE,
            Action.UP,
            Action.RIGHT,
            Action.LEFT,
            Action.DOWN,
            Action.UPRIGHT,
            Action.UPLEFT,
            Action.DOWNRIGHT,
            Action.DOWNLEFT,
            Action.UPFIRE,
            Action.RIGHTFIRE,
            Action.LEFTFIRE,
            Action.DOWNFIRE,
            Action.UPRIGHTFIRE,
            Action.UPLEFTFIRE,
            Action.DOWNRIGHTFIRE,
            Action.DOWNLEFTFIRE
        ]
        self.frame_stack_size = 4
        self.obs_size = 3*4 + 1 + 1

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jax.random.PRNGKey = jax.random.PRNGKey(42)) -> Tuple["WordZapperObservation", "WordZapperState"]:
        """Reset the Word Zapper environment state with a new word and initial player/letter positions."""
        
        # Define dictionary of words (1 to 5 letters each)
        WORD_DICT = ["CAT", "MOON", "SUN", "ZAP", "ROBOT", "AXE", "JAX", "FIRE", "COLD", "FLUX", "BYTE", "RAY"]
        MAX_WORD_LEN = 5
        MAX_WORDS = len(WORD_DICT)

        # Convert dictionary to jax-friendly format
        def encode_words(word_list):
            padded_array = jnp.zeros((len(word_list), MAX_WORD_LEN), dtype=jnp.int32)
            word_lengths = []
            for i, word in enumerate(word_list):
                ascii_vals = [ord(c) for c in word]
                padded_array = padded_array.at[i, :len(ascii_vals)].set(jnp.array(ascii_vals))
                word_lengths.append(len(word))
            return padded_array, jnp.array(word_lengths)

        ENCODED_WORDS, WORD_LENGTHS = encode_words(WORD_DICT)

        # Sample a random word
        word_idx = jax.random.randint(key, (), 0, MAX_WORDS)
        word = ENCODED_WORDS[word_idx]
        word_len = WORD_LENGTHS[word_idx]

        # Prepare current word with padding
        current_word = jnp.zeros(MAX_WORD_LEN, dtype=jnp.int32).at[:word_len].set(word[:word_len])

        # Initialize letter scrolling positions (example: one letter per word character)
        letters_x = jnp.linspace(160, 160 + 20 * word_len, num=word_len, dtype=jnp.int32)
        letters_y = jnp.full((word_len,), 32, dtype=jnp.int32)
        letters_char = word[:word_len]
        letters_alive = jnp.ones((word_len,), dtype=jnp.int32)
        letters_speed = jnp.full((word_len,), 1, dtype=jnp.int32)

        # Initialize player state
        player_x = jnp.array(PLAYER_X)
        player_y = jnp.array(HEIGHT // 2)
        player_speed = jnp.array(0)
        cooldown_timer = jnp.array(0)

        # Asteroid placeholder (can be extended later)
        asteroid_x = jnp.array(0)
        asteroid_y = jnp.array(0)
        asteroid_speed = jnp.array(0)
        asteroid_alive = jnp.array(0)

        # Other state variables
        player_score = jnp.array(0)
        timer = jnp.array(0)
        step_counter = jnp.array(0)
        buffer = jnp.zeros((5,), dtype=jnp.int32)

        # Construct state object
        state = WordZapperState(
        player_x=player_x,
        player_y=player_y,
        player_speed=player_speed,
        cooldown_timer=cooldown_timer,
        asteroid_x=asteroid_x,
        asteroid_y=asteroid_y,
        asteroid_speed=asteroid_speed,
        asteroid_alive=asteroid_alive,
        letters_x=letters_x,
        letters_y=letters_y,
        letters_char=letters_char,
        letters_alive=letters_alive,
        letters_speed=letters_speed,
        current_word=current_word,
        current_letter_index=jnp.array(0),
        player_score=player_score,
        timer=timer,
        step_counter=step_counter,
        buffer=buffer,
        )

        # Observation builder (you would define this based on your state model)
        obs = self._get_observation(state)

        return obs, state   
    
    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, state: WordZapperState, action: chex.Array
    ) -> Tuple[WordZapperObservation, WordZapperState, float, bool, WordZapperInfo]:
        super().step(state, action)

        previous_state = state
        _, reset_state = self.reset()

        # First handle death animation 
        def handle_death_animation():
            pass
        
        def handle_score_freeze():
            pass
        
        # Normal game logic starts here
        def normal_game_step():
            pass


        return_state = jax.lax.cond(
            state.death_counter > 0,
            lambda _: handle_death_animation(),
            lambda _: jax.lax.cond(
                state.death_counter < 0,
                lambda _: handle_score_freeze(),
                lambda _: normal_game_step(),
                operand=None,
            ),
            operand=None,
        )

        # Get observation and info
        observation = self._get_observation(return_state)

        done = self._get_done(return_state)
        env_reward = self._get_env_reward(previous_state, return_state)
        all_rewards = self._get_all_rewards(previous_state, return_state)
        info = self._get_info(return_state, all_rewards)

        # Choose between death animation and normal game step
        return observation, return_state, env_reward, done, info


class WordZapperRenderer(AtraJaxisRenderer):
    def __init__(self, screen):
        """
        Initialize the renderer with all necessary rectangles and colors.
        """
        super().__init__()  # No arguments passed to AtraJaxisRenderer's __init__

        # Store the screen (Pygame surface)
        self.screen = screen

        # Define colors
        self.background_color = (0, 0, 0)  # Black
        self.spaceship_color = (255, 0, 0)  # Red
        self.title_color = (0, 0, 255)  # Blue
        self.text_color = (255, 255, 255)  # White

        # Define rectangles
        self.wordzapper_rect = jnp.array([0, 0, 800, 100])  # Title bar
        self.spaceship_rect = jnp.array([111, 365, 50, 30])  # Spaceship

    def render(self, spaceship_rect, current_time):
        """
        Render all game elements on the screen.
        """
        # Fill the screen with the background color
        self.screen.fill(self.background_color)  # Use Pygame's fill method

        # Render the title bar
        pygame.draw.rect(self.screen, self.title_color, self.wordzapper_rect)

        # Render the spaceship
        pygame.draw.rect(self.screen, self.spaceship_color, spaceship_rect)

        # Render the time
        font = pygame.font.SysFont(None, 36)
        time_surf = font.render(f"TIME: {current_time}", True, self.text_color)
        self.screen.blit(time_surf, (400, 50))

        # Update the display
        pygame.display.update()  # Use Pygame's display update method


if __name__ == "__main__":
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Word Zapper")
    clock = pygame.time.Clock()


    ##     game = JaxWordZapper() <-- eventual game


    # Initialize the renderer
    renderer = WordZapperRenderer(screen)

    # Define the spaceship rectangle
    spaceship_rect = pygame.Rect(111, 365, 50, 30)  # Use Pygame Rect for rendering

    # Game loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get player actions
        actions = get_human_action()
        for action in actions:
            if action == "LEFT":
                spaceship_rect.x -= 5
            elif action == "RIGHT":
                spaceship_rect.x += 5
            elif action == "UP":
                spaceship_rect.y -= 5
            elif action == "DOWN":
                spaceship_rect.y += 5
            elif action == "FIRE":
                print("FIRE")
            elif action == "UPFIRE":
                print("UPFIRE")
            elif action == "DOWNFIRE":
                print("DOWNFIRE")
            elif action == "LEFTFIRE":
                print("LEFTFIRE")
            elif action == "RIGHTFIRE":
                print("RIGHTFIRE")
            elif action == "UPLEFTFIRE":
                spaceship_rect.x -= 2.5
                spaceship_rect.y -= 2.5
                print("UPLEFTFIRE")
            elif action == "UPRIGHTFIRE":
                spaceship_rect.x += 2.5
                spaceship_rect.y -= 2.5
                print("UPRIGHTFIRE")
            elif action == "DOWNLEFTFIRE":
                spaceship_rect.x -= 2.5
                spaceship_rect.y += 2.5
                print("DOWNLEFTFIRE")
            elif action == "DOWNRIGHTFIRE":
                spaceship_rect.x += 2.5
                spaceship_rect.y += 2.5
                print("DOWNRIGHTFIRE")

        # Prevent the spaceship from going out of bounds
        if spaceship_rect.x < 0:
            spaceship_rect.x = 0
        if spaceship_rect.x > WINDOW_WIDTH - spaceship_rect.width:
            spaceship_rect.x = WINDOW_WIDTH - spaceship_rect.width
        if spaceship_rect.y < 0:
            spaceship_rect.y = 0
        if spaceship_rect.y > WINDOW_HEIGHT - spaceship_rect.height:
            spaceship_rect.y = WINDOW_HEIGHT - spaceship_rect.height

        # Calculate the current time
        current_time = 90 - pygame.time.get_ticks() // 1000

        # Render the game
        renderer.render(spaceship_rect, current_time)

        # Control the frame rate
        clock.tick(60)

    pygame.quit()