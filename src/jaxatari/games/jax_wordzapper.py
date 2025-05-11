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

# Background color and object colors
BACKGROUND_COLOR = 144, 72, 17
PLAYER_COLOR = 92, 186, 92
BALL_COLOR = 236, 236, 236  # White ball
WALL_COLOR = 236, 236, 236  # White walls
SCORE_COLOR = 236, 236, 236  # White score

# Player and letter positions
PLAYER_START_X = 140
PLAYER_START_Y = 46

LETTERS_Y = 20

# Object sizes (width, height)
PLAYER_SIZE = (4, 16)
ASTEROID_SIZE = (8, 7)
MISSILE_SIZE = (8, 1)
LETTER_SIZE = (8, 7)
TIMER_SIZE = (8, 8)
WALL_TOP_Y = 24
WALL_TOP_HEIGHT = 10
WALL_BOTTOM_Y = 194
WALL_BOTTOM_HEIGHT = 16

# Pygame window dimensions
WINDOW_WIDTH = 160 * 3
WINDOW_HEIGHT = 210 * 3

WIDTH = 160
HEIGHT = 210
SCALING_FACTOR = 3

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
    asteroid_positions: (
        chex.Array
    ) # (12, 3) array for asteroids - separated into 4 lanes, 3 slots per lane [left to right]

    letters_x: chex.Array # letters at the top
    letters_y: chex.Array
    letters_char: chex.Array
    letters_alive: chex.Array
    letters_speed: chex.Array
    letters_positions: (
        chex.Array
    ) # (26,1) y coorinate does not change and deined in LETTERS_Y

    player_missile_position: chex.Array  # shape: (3,) -> [x, y, direction]

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
    active: jnp.ndarray


class WordZapperObservation(NamedTuple):
    player: EntityPosition
    asteroids: jnp.ndarray  # Shape (12, 5) - 12 asteroids each with x,y,w,h,active
    letters: jnp.ndarray # Shape (26, 5) - 26 letters each with x,y,w,h,active

    letters_char: jnp.ndarray 
    letters_alive: jnp.ndarray  # active letters

    current_word: jnp.ndarray  # word to form
    current_letter_index: jnp.ndarray  # current position in word

    player_missile: EntityPosition

    cooldown_timer: jnp.ndarray
    timer: jnp.ndarray

class WordZapperInfo(NamedTuple):
    timer: jnp.ndarray
    current_word: jnp.ndarray
    game_over: jnp.ndarray


def load_sprites():
    """Load all sprites required for Word Zapper rendering."""
    def make_rect(h, w, color):
        return jnp.ones((h, w, 3), dtype=jnp.uint8) * jnp.array(color, dtype=jnp.uint8)

    ## TODO for now just rectangles, no sprites
    SPRITE_BG = make_rect(WINDOW_WIDTH, WINDOW_HEIGHT, [0, 0, 0]) 

    SPRITE_PLAYER = make_rect(PLAYER_SIZE[0], PLAYER_SIZE[1], [0, 0, 255]) 

    TIMER_INDICATOR = make_rect(TIMER_SIZE[0], TIMER_SIZE[1], [0, 255, 0]) 
    
    return (
        SPRITE_BG,
        SPRITE_PLAYER,
        TIMER_INDICATOR,
    )


# Load sprites once at module level
(
    SPRITE_BG,
    SPRITE_PLAYER,
    TIMER_INDICATOR,
) = load_sprites()


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
    FIRE_ACTIONS = jnp.array([
        Action.FIRE, Action.UPFIRE, Action.DOWNFIRE,
        Action.LEFTFIRE, Action.RIGHTFIRE,
        Action.UPLEFTFIRE, Action.UPRIGHTFIRE,
        Action.DOWNLEFTFIRE, Action.DOWNRIGHTFIRE
    ])

    move_up = jnp.any(jnp.equal(action, MOVE_UP))
    move_down = jnp.any(jnp.equal(action, MOVE_DOWN))
    is_firing = jnp.any(jnp.equal(action, FIRE_ACTIONS))

    delta_y = jnp.where(move_up, -player_speed, jnp.where(move_down, player_speed, 0))
    new_player_y = jnp.clip(player_y + delta_y, 0, HEIGHT - PLAYER_SIZE[1])

    can_fire = cooldown_timer == 0
    fired = jnp.logical_and(is_firing, can_fire)
    new_cooldown_timer = jnp.where(fired, 8, jnp.maximum(cooldown_timer - 1, 0))

    return player_x, new_player_y, player_speed, new_cooldown_timer, fired

@jax.jit
def scrolling_letters(letters_x, letters_speed, letters_alive):
    letters_x = letters_x - letters_speed
    wrapped_x = jnp.where(letters_x < -8, 160, letters_x)
    updated_x = jnp.where(letters_alive == 1, wrapped_x, letters_x)
    return updated_x

@jax.jit
def player_missile_step(
    missile_pos: chex.Array,
    player_x: chex.Array,
    player_y: chex.Array,
    action: chex.Array,
    cooldown_timer: chex.Array
) -> Tuple[chex.Array, chex.Array]:
    """
    Handle firing logic and missile movement for Word Zapper.

    missile_pos: [x, y, dx, dy] or [0, 0, 0, 0] if inactive
    Returns: new missile position, new cooldown timer
    """
    # Define direction for each firing action
    DIRECTION_MAP = {
        Action.FIRE: (0, -1),
        Action.UPFIRE: (0, -1),
        Action.LEFTFIRE: (-1, 0),
        Action.RIGHTFIRE: (1, 0),
        Action.UPLEFTFIRE: (-1, -1),
        Action.UPRIGHTFIRE: (1, -1),
        Action.DOWNLEFTFIRE: (-1, 1),
        Action.DOWNRIGHTFIRE: (1, 1),
        Action.DOWNFIRE: (0, 1),
    }

    fire = jnp.any(jnp.array([action in DIRECTION_MAP and cooldown_timer == 0]))

    def fire_missile():
        dx, dy = DIRECTION_MAP[action]
        new_pos = jnp.array([player_x + 2, player_y + 2, dx, dy], dtype=jnp.int32)
        return new_pos

    # If no active missile, allow firing
    is_active = missile_pos[2] != 0 or missile_pos[3] != 0
    can_fire = jnp.logical_and(~is_active, fire)

    # Either fire or update missile
    new_missile = jax.lax.cond(
        can_fire,
        fire_missile,
        lambda: jnp.where(
            is_active,
            jnp.array([
                missile_pos[0] + missile_pos[2] * 4,
                missile_pos[1] + missile_pos[3] * 4,
                missile_pos[2],
                missile_pos[3],
            ]),
            missile_pos
        )
    )

    # Check if missile out of bounds
    out_of_bounds = jnp.logical_or(
        jnp.logical_or(new_missile[0] < 0, new_missile[0] > WIDTH),
        jnp.logical_or(new_missile[1] < 0, new_missile[1] > HEIGHT)
    )

    final_missile = jnp.where(out_of_bounds, jnp.array([0, 0, 0, 0]), new_missile)

    # Update cooldown
    new_cooldown = jnp.where(fire, 6, jnp.maximum(cooldown_timer - 1, 0))

    return final_missile, new_cooldown


def shooting_letter():
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
        player_x = jnp.array(PLAYER_START_X)
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
    
    
    def get_action_space(self) -> jnp.ndarray:
        return jnp.array(self.action_set)

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: WordZapperState) -> WordZapperObservation:
        # Create player (already scalar, no need for vectorization)
        player = EntityPosition(
            x=state.player_x,
            y=state.player_y,
            width=jnp.array(PLAYER_SIZE[0]),
            height=jnp.array(PLAYER_SIZE[1]),
            active=jnp.array(1),  # Player is always active
        )

        # Define a function to convert entity position data into the correct format
        def convert_to_entity(pos, size):
            return jnp.array([
                pos[0],  # x position
                pos[1],  # y position
                size[0],  # width
                size[1],  # height
                pos[2] != 0,  # active flag
            ])

        # Apply conversion to asteroid positions
        asteroids = jax.vmap(lambda pos: convert_to_entity(pos, ASTEROID_SIZE))(state.asteroid_positions)

        # Convert letter positions into the correct entity format
        letters = jax.vmap(lambda pos: convert_to_entity(pos, LETTER_SIZE))(state.letters_positions)

        # Convert player missile position into the correct entity format
        missile_pos = state.player_missile_position
        player_missile = EntityPosition(
            x=missile_pos[0],
            y=missile_pos[1],
            width=jnp.array(MISSILE_SIZE[0]),
            height=jnp.array(MISSILE_SIZE[1]),
            active=jnp.array(missile_pos[2] != 0),
        )

        return WordZapperObservation(
            player=player,
            asteroids=asteroids,
            letters=letters,
            letters_char=state.letters_char,
            letters_alive=state.letters_alive,
            current_word=state.current_word,
            current_letter_index=state.current_letter_index,
            player_missile=player_missile,
            cooldown_timer=state.cooldown_timer,
            timer=state.timer,
        )

    
    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: WordZapperState) -> bool:
        """Check if the game should end due to timer expiring."""
        MAX_TIME = 60 * 90  # 90 seconds at 60 FPS
        return state.timer >= MAX_TIME

    @partial(jax.jit, static_argnums=(0,))
    def _get_env_reward(self, previous_state: WordZapperState, state: WordZapperState):
        pass
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_all_rewards(self, previous_state: WordZapperState, state: WordZapperState) -> jnp.ndarray:
        pass

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: WordZapperState, all_rewards: jnp.ndarray) -> WordZapperInfo:
        pass

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, state: WordZapperState, action: chex.Array
    ) -> Tuple[WordZapperObservation, WordZapperState, float, bool, WordZapperInfo]:
        super().step(state, action)

        previous_state = state

        def normal_game_step():
            # Step player
            new_player_x, new_player_y, new_player_speed, new_cooldown_timer, fired = player_step(
                state.player_x,
                state.player_y,
                state.player_speed,
                state.cooldown_timer,
                action,
            )

            # Step letters
            new_letters_x = scrolling_letters(state.letters_x, state.letters_speed, state.letters_alive)

            new_state = state._replace(
                player_x=new_player_x,
                player_y=new_player_y,
                player_speed=new_player_speed,
                cooldown_timer=new_cooldown_timer,
                letters_x=new_letters_x,
                step_counter=state.step_counter + 1,
            )
            return new_state

        return_state = normal_game_step()

        observation = self._get_observation(return_state)
        done = self._get_done(return_state)
        env_reward = self._get_env_reward(previous_state, return_state)
        all_rewards = self._get_all_rewards(previous_state, return_state)
        info = self._get_info(return_state, all_rewards)

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

    def render(self, state: WordZapperState):
        """
        Render all game elements based on current WordZapperState (JAX).
        """
        self.screen.fill(self.background_color)

        # 🎯 Draw the player spaceship
        player_rect = pygame.Rect(
            int(state.player_x.item()),
            int(state.player_y.item()),
            PLAYER_SIZE[0],
            PLAYER_SIZE[1],
        )
        pygame.draw.rect(self.screen, self.spaceship_color, player_rect)

        # 🚀 Draw player missile if active
        missile_x, missile_y, direction = state.player_missile_position
        if missile_x > 0 and missile_y > 0 and direction != 0:
            missile_rect = pygame.Rect(int(missile_x.item()), int(missile_y.item()), 2, 6)
            pygame.draw.rect(self.screen, (255, 255, 0), missile_rect)  # Yellow

        # 🔡 Draw scrolling letters
        for i in range(len(state.letters_x)):
            if state.letters_alive[i] == 1:
                char_code = int(state.letters_char[i].item())
                letter_x = int(state.letters_x[i].item())
                letter_y = int(state.letters_y[i].item())
                letter_char = chr(char_code)

                font = pygame.font.SysFont(None, 24)
                letter_surface = font.render(letter_char, True, self.text_color)
                self.screen.blit(letter_surface, (letter_x, letter_y))

        # 🧱 Draw asteroid if alive
        if state.asteroid_alive.item() == 1:
            asteroid_rect = pygame.Rect(
                int(state.asteroid_x.item()), int(state.asteroid_y.item()), 8, 8
            )
            pygame.draw.rect(self.screen, (128, 128, 128), asteroid_rect)

        # 🕒 Draw timer
        font = pygame.font.SysFont(None, 36)
        time_surface = font.render(f"TIME: {int(state.timer.item())}", True, self.text_color)
        self.screen.blit(time_surface, (10, 10))

        pygame.display.update()


    # @partial(jax.jit, static_argnums=(0,))
    # def render(self, state):
    #     raster = jnp.zeros((WIDTH, HEIGHT, 3))

    #     # Render the spaceship
    #     pygame.draw.rect(self.screen, self.spaceship_color, spaceship_rect)

    #     # render background
    #     frame_bg = aj.get_sprite_frame(SPRITE_BG, 0)
    #     raster = aj.render_at(raster, 0, 0, frame_bg)
    
    #     return raster




if __name__ == "__main__":
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Word Zapper")
    clock = pygame.time.Clock()


    #game = JaxWordZapper() # <-- eventual game

    # Initialize the renderer
    renderer = WordZapperRenderer(screen) # TODO screen must not be a parameter eventually, see seaquest

        # Get jitted functions
    # jitted_step = jax.jit(game.step)
    # jitted_reset = jax.jit(game.reset)

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

        current_time = 90 - pygame.time.get_ticks() // 1000
        renderer.render(spaceship_rect, current_time)



    # # Initialize game and renderer

    # curr_obs, curr_state = jitted_reset()

    # # Game loop with rendering
    # running = True
    # frame_by_frame = False
    # frameskip = 1
    # counter = 1

    # while running:
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             running = False
    #         elif event.type == pygame.KEYDOWN:
    #             if event.key == pygame.K_f:
    #                 frame_by_frame = not frame_by_frame
    #         elif event.type == pygame.KEYDOWN or (
    #             event.type == pygame.KEYUP and event.key == pygame.K_n
    #         ):
    #             if event.key == pygame.K_n and frame_by_frame:
    #                 if counter % frameskip == 0:
    #                     action = get_human_action()
    #                     curr_obs, curr_state, reward, done, info = jitted_step(
    #                         curr_state, action
    #                     )
    #                     print(f"Observations: {curr_obs}")
    #                     print(f"Reward: {reward}, Done: {done}, Info: {info}")

    #     if not frame_by_frame:
    #         if counter % frameskip == 0:
    #             action = get_human_action()
    #             curr_obs, curr_state, reward, done, info = jitted_step(
    #                 curr_state, action
    #             )

    #     # render and update pygame
    #     raster = renderer.render(curr_state)
    #     aj.update_pygame(screen, raster, SCALING_FACTOR, WIDTH, HEIGHT)
    #     counter += 1


    clock.tick(60)

        current_time = 90 - pygame.time.get_ticks() // 1000
        renderer.render(curr_state, current_time)

    pygame.quit()