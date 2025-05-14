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
MAX_ASTEROIDS_COUNT = 6 #TODO not sure about value
ENEMY_STEP_SIZE = 2

# Background color and object colors
BACKGROUND_COLOR = 144, 72, 17
PLAYER_COLOR = 92, 186, 92
BALL_COLOR = 236, 236, 236  # White ball
WALL_COLOR = 236, 236, 236  # White walls
SCORE_COLOR = 236, 236, 236  # White score


# define object orientations
FACE_LEFT = -1
FACE_RIGHT = 1

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

MIN_BOUND = (0,0)
MAX_BOUND = (WINDOW_WIDTH, WINDOW_HEIGHT)

X_BORDERS = (0, 160)

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


def get_human_action() -> chex.Array:
    """Get human action from keyboard with support for diagonal movement and combined fire"""
    keys = pygame.key.get_pressed()
    up = keys[pygame.K_UP] or keys[pygame.K_w]
    down = keys[pygame.K_DOWN] or keys[pygame.K_s]
    left = keys[pygame.K_LEFT] or keys[pygame.K_a]
    right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
    fire = keys[pygame.K_SPACE]

    # Diagonal movements with fire
    if up and right and fire:
        return jnp.array(Action.UPRIGHTFIRE)
    if up and left and fire:
        return jnp.array(Action.UPLEFTFIRE)
    if down and right and fire:
        return jnp.array(Action.DOWNRIGHTFIRE)
    if down and left and fire:
        return jnp.array(Action.DOWNLEFTFIRE)

    # Cardinal directions with fire
    if up and fire:
        return jnp.array(Action.UPFIRE)
    if down and fire:
        return jnp.array(Action.DOWNFIRE)
    if left and fire:
        return jnp.array(Action.LEFTFIRE)
    if right and fire:
        return jnp.array(Action.RIGHTFIRE)

    # Diagonal movements
    if up and right:
        return jnp.array(Action.UPRIGHT)
    if up and left:
        return jnp.array(Action.UPLEFT)
    if down and right:
        return jnp.array(Action.DOWNRIGHT)
    if down and left:
        return jnp.array(Action.DOWNLEFT)

    # Cardinal directions
    if up:
        return jnp.array(Action.UP)
    if down:
        return jnp.array(Action.DOWN)
    if left:
        return jnp.array(Action.LEFT)
    if right:
        return jnp.array(Action.RIGHT)
    if fire:
        return jnp.array(Action.FIRE)

    return jnp.array(Action.NOOP)
    


class WordZapperState(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    player_speed: chex.Array
    player_direction: chex.Array
    cooldown_timer: chex.Array

    asteroid_x: chex.Array
    asteroid_y: chex.Array
    asteroid_speed: chex.Array
    asteroid_alive: chex.Array
    asteroid_positions: (
        chex.Array
    ) # (12, 3) array for asteroids - separated into 4 lanes, 3 slots per lane [left to right]

    # letters_x: chex.Array # letters at the top
    # letters_y: chex.Array
    # letters_char: chex.Array
    # letters_alive: chex.Array
    # letters_speed: chex.Array
    # letters_positions: (
    #     chex.Array
    # ) # (26,1) y coorinate does not change and deined in LETTERS_Y

    player_missile_position: chex.Array  # shape: (1,3) -> [x, y, direction]

    # current_word: chex.Array # the actual word
    # current_letter_index: chex.Array

    timer: chex.Array
    step_counter: chex.Array
    rng_key: chex.PRNGKey

class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray
    active: jnp.ndarray


class WordZapperObservation(NamedTuple):
    player: EntityPosition
    asteroids: jnp.ndarray  # Shape (12, 5) - 12 asteroids each with x,y,w,h,active
    # letters: jnp.ndarray # Shape (26, 5) - 26 letters each with x,y,w,h,active

    # letters_char: jnp.ndarray 
    # letters_alive: jnp.ndarray  # active letters

    # current_word: jnp.ndarray  # word to form
    # current_letter_index: jnp.ndarray  # current position in word

    player_missile: EntityPosition

    cooldown_timer: jnp.ndarray
    timer: jnp.ndarray

class WordZapperInfo(NamedTuple):
    timer: jnp.ndarray
    current_word: jnp.ndarray
    game_over: jnp.ndarray


def load_sprites(bg_file="sprites/seaquest/bg/1.npy"):

    """Load all sprites required for Word Zapper rendering."""
    # def make_rect(h, w, color):
    #     return jnp.ones((h, w, 3), dtype=jnp.uint8) * jnp.array(color, dtype=jnp.uint8)

    # ## TODO for now just rectangles, no sprites
    # SPRITE_BG = make_rect(WINDOW_WIDTH, WINDOW_HEIGHT, [0, 0, 0]) 

    # SPRITE_PLAYER = make_rect(PLAYER_SIZE[0], PLAYER_SIZE[1], [0, 0, 255]) 

    # TIMER_INDICATOR = make_rect(TIMER_SIZE[0], TIMER_SIZE[1], [0, 255, 0])

    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Load sprites - no padding needed for background since it's already full size
    bg_path = os.path.join(MODULE_DIR, bg_file)
    bg1 = aj.loadFrame(bg_path)

    pl_sub1 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/seaquest/player_sub/1.npy"))
    pl_sub2 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/seaquest/player_sub/2.npy"))
    pl_sub3 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/seaquest/player_sub/3.npy"))

    pl_torp = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/seaquest/player_torp/1.npy"))

    # Pad player submarine sprites to match each other
    pl_sub_sprites = aj.pad_to_match([pl_sub1, pl_sub2, pl_sub3])
    # Pad player torpedo sprites to match each other
    pl_torp_sprites = [pl_torp]

    SPRITE_BG = jnp.expand_dims(bg1, axis=0)
    SPRITE_PL_TORP = jnp.repeat(pl_torp_sprites[0][None], 1, axis=0)

    # Player submarine sprites
    SPRITE_PL_SUB = jnp.concatenate(
        [
            jnp.repeat(pl_sub_sprites[0][None], 4, axis=0),
            jnp.repeat(pl_sub_sprites[1][None], 4, axis=0),
            jnp.repeat(pl_sub_sprites[2][None], 4, axis=0),
        ]
    )

    return (
        SPRITE_BG,
        SPRITE_PL_SUB,
        SPRITE_PL_TORP
    )


# Load sprites once at module level
(
    SPRITE_BG,
    SPRITE_PL_SUB,
    SPRITE_PL_TORP
) = load_sprites("sprites/wordzapper/WZbackground.npy")



@jax.jit
def player_step(
    state: WordZapperState, action: chex.Array
) -> tuple[chex.Array, chex.Array, chex.Array]:
    # implement all the possible movement directions for the player, the mapping is:
    # anything with left in it, add -1 to the x position
    # anything with right in it, add 1 to the x position
    # anything with up in it, add -1 to the y position
    # anything with down in it, add 1 to the y position
    up = jnp.any(
        jnp.array(
            [
                action == Action.UP,
                action == Action.UPRIGHT,
                action == Action.UPLEFT,
                action == Action.UPFIRE,
                action == Action.UPRIGHTFIRE,
                action == Action.UPLEFTFIRE,
            ]
        )
    )
    down = jnp.any(
        jnp.array(
            [
                action == Action.DOWN,
                action == Action.DOWNRIGHT,
                action == Action.DOWNLEFT,
                action == Action.DOWNFIRE,
                action == Action.DOWNRIGHTFIRE,
                action == Action.DOWNLEFTFIRE,
            ]
        )
    )
    left = jnp.any(
        jnp.array(
            [
                action == Action.LEFT,
                action == Action.UPLEFT,
                action == Action.DOWNLEFT,
                action == Action.LEFTFIRE,
                action == Action.UPLEFTFIRE,
                action == Action.DOWNLEFTFIRE,
            ]
        )
    )
    right = jnp.any(
        jnp.array(
            [
                action == Action.RIGHT,
                action == Action.UPRIGHT,
                action == Action.DOWNRIGHT,
                action == Action.RIGHTFIRE,
                action == Action.UPRIGHTFIRE,
                action == Action.DOWNRIGHTFIRE,
            ]
        )
    )

    player_x = jnp.where(
        right, state.player_x + 2, jnp.where(left, state.player_x - 2, state.player_x)
    )

    player_y = jnp.where(
        down, state.player_y + 2, jnp.where(up, state.player_y - 2, state.player_y)
    )

    # set the direction according to the movement
    player_direction = jnp.where(right, 1, jnp.where(left, -1, state.player_direction))

    # perform out of bounds checks
    player_x = jnp.where(
        player_x < MIN_BOUND[0],
        MIN_BOUND[0],  # Clamp to min player bound
        jnp.where(
            player_x > MAX_BOUND[0],
            MAX_BOUND[0],  # Clamp to max player bound
            player_x,
        ),
    )

    player_y = jnp.where(
        player_y < MIN_BOUND[1],
        0,
        jnp.where(player_y > MAX_BOUND[1], MAX_BOUND[1], player_y),
    )

    return player_x, player_y, player_direction

@jax.jit
def scrolling_letters(letters_x, letters_speed, letters_alive):
    letters_x = letters_x - letters_speed
    wrapped_x = jnp.where(letters_x < -8, 160, letters_x)
    updated_x = jnp.where(letters_alive == 1, wrapped_x, letters_x)
    return updated_x

@jax.jit
def player_missile_step(
    state: WordZapperState, curr_player_x, curr_player_y, action: chex.Array
) -> chex.Array:
    # check if the player shot this frame
    fire = jnp.any(
        jnp.array(
            [
                action == Action.FIRE,
                action == Action.UPRIGHTFIRE,
                action == Action.UPLEFTFIRE,
                action == Action.DOWNFIRE,
                action == Action.DOWNRIGHTFIRE,
                action == Action.DOWNLEFTFIRE,
                action == Action.RIGHTFIRE,
                action == Action.LEFTFIRE,
                action == Action.UPFIRE,
            ]
        )
    )

    # IMPORTANT: do not change the order of this check, since the missile does not move in its first frame!!
    # also check if there is currently a missile in frame by checking if the player_missile_position is empty
    missile_exists = state.player_missile_position[2] != 0

    # if the player shot and there is no missile in frame, then we can shoot a missile
    # the missile y is the current player y position + 7
    # the missile x is either player x + 3 if facing left or player x + 13 if facing right
    new_missile = jnp.where(
        jnp.logical_and(fire, jnp.logical_not(missile_exists)),
        jnp.where(
            state.player_direction == -1,
            jnp.array([curr_player_x + 3, curr_player_y + 7, -1]),
            jnp.array([curr_player_x + 13, curr_player_y + 7, 1]),
        ),
        state.player_missile_position,
    )

    # if a missile is in frame and exists, we move the missile further in the specified direction (5 per tick), also always put the missile at the current player y position
    new_missile = jnp.where(
        missile_exists,
        jnp.array(
            [new_missile[0] + new_missile[2] * 5, curr_player_y + 7, new_missile[2]]
        ),
        new_missile,
    )

    # check if the new positions are still in bounds
    new_missile = jnp.where(
        new_missile[0] < X_BORDERS[0],
        jnp.array([0, 0, 0]),
        jnp.where(new_missile[0] > X_BORDERS[1], jnp.array([0, 0, 0]), new_missile),
    )

    return new_missile

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
    def reset(self, key: jax.random.PRNGKey = jax.random.PRNGKey(42)) -> Tuple[WordZapperObservation, WordZapperState]:
        """Initialize game state"""
        reset_state = WordZapperState(
            player_x=jnp.array(PLAYER_START_X),
            player_y=jnp.array(PLAYER_START_Y),
            player_speed=jnp.array(0),
            player_direction=jnp.array(0),
            cooldown_timer=jnp.array(0),
            asteroid_x=jnp.array(0),
            asteroid_y=jnp.array(0),
            asteroid_speed=jnp.array(0), ## TODO these values are not clear, some need change
            asteroid_alive=jnp.array(0),
            asteroid_positions=jnp.zeros((MAX_ASTEROIDS_COUNT, 3)),
            #spawn_state=initialize_spawn_state(), 
            player_missile_position=jnp.zeros(3),  # x,y,direction
            timer=jnp.array(90),
            step_counter=jnp.array(0),
            rng_key=key,
        )

        initial_obs = self._get_observation(reset_state)
        return initial_obs, reset_state
    
    
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
        # letters = jax.vmap(lambda pos: convert_to_entity(pos, LETTER_SIZE))(state.letters_positions)

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
            # letters=letters,
            # letters_char=state.letters_char,
            # letters_alive=state.letters_alive,
            # current_word=state.current_word,
            # current_letter_index=state.current_letter_index,
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
        
        previous_state = state
        _, reset_state = self.reset()

        def normal_game_step():
            # Step player
            new_player_x, new_player_y, new_palyer_direction = player_step(
                state,
                action,
            )

            player_missile_position = player_missile_step(
                state, new_player_x, new_player_y, action
            )

            new_step_counter = jnp.where(
                state.step_counter == 1024,
                jnp.array(0),
                state.step_counter + 1,
            )

            new_state = state._replace(
                player_x=new_player_x,
                player_y=new_player_y,
                player_direction=new_palyer_direction,
                player_missile_position=player_missile_position,
                step_counter=new_step_counter,
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
    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        raster = jnp.zeros((WIDTH, HEIGHT, 3))

        ## render background
        frame_bg = aj.get_sprite_frame(SPRITE_BG, 0)
        raster = aj.render_at(raster, 0, 0, frame_bg)

        frame_pl_sub = aj.get_sprite_frame(SPRITE_PL_SUB, state.step_counter)
        raster = aj.render_at(
            raster,
            state.player_x,
            state.player_y,
            frame_pl_sub,
            flip_horizontal=state.player_direction == FACE_LEFT,
        )

        # render player torpedo
        frame_pl_torp = aj.get_sprite_frame(SPRITE_PL_TORP, state.step_counter)

        should_render = state.player_missile_position[0] > 0
        
        raster = jax.lax.cond(
            should_render,
            lambda r: aj.render_at(
                r,
                state.player_missile_position[0],
                state.player_missile_position[1],
                frame_pl_torp,
                flip_horizontal=state.player_missile_position[2] == FACE_LEFT,
            ),
            lambda r: r,
            raster,
        )

    
        return raster




if __name__ == "__main__":
    # Initialize Pygame
    game = JaxWordZapper()
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Word Zapper")
    clock = pygame.time.Clock()


    game = JaxWordZapper()

    # Initialize the renderer
    renderer = WordZapperRenderer()


    # Get jitted functions
    jitted_step = jax.jit(game.step)
    jitted_reset = jax.jit(game.reset)


    curr_obs, curr_state = jitted_reset()

    # Game loop with rendering
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
                        curr_obs, curr_state, reward, done, info = jitted_step(
                            curr_state, action
                        )
                        print(f"Observations: {curr_obs}")
                        print(f"Reward: {reward}, Done: {done}, Info: {info}")

        if not frame_by_frame:
            if counter % frameskip == 0:
                action = get_human_action()
                curr_obs, curr_state, reward, done, info = jitted_step(
                    curr_state, action
                )

        # render and update pygame
        raster = renderer.render(curr_state)
        aj.update_pygame(screen, raster, SCALING_FACTOR, WIDTH, HEIGHT)
        counter += 1

        clock.tick(60)

    pygame.quit()