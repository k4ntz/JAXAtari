import os
from functools import partial
from typing import NamedTuple, Tuple
from jax._src.core import stash_axis_env, reset_trace_state
import jax.lax
import jax.numpy as jnp
import chex
import pygame
from gymnax.environments import spaces

from jaxatari.renderers import AtraJaxisRenderer
from jaxatari.rendering import atraJaxis as aj
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action

# Constants for game environment
MAX_ASTEROIDS_COUNT = 6 #TODO not sure about value

# define object orientations
FACE_LEFT = -1
FACE_RIGHT = 1

# Player and letter positions
PLAYER_START_X = 80
PLAYER_START_Y = 110

# Object sizes (width, height)
PLAYER_SIZE = (4, 16)
ASTEROID_SIZE = (8, 7)
MISSILE_SIZE = (8, 1)
ZAPPER_SIZE = (8, 1)
LETTER_SIZE = (8, 8)

# Pygame window dimensions
WIDTH = 160
HEIGHT = 210
SCALING_FACTOR = 3

WINDOW_WIDTH = 160 * 3
WINDOW_HEIGHT = 210 * 3

MIN_BOUND = (0,0)
MAX_BOUND = (WIDTH, HEIGHT)

X_BORDERS = (0, 160)

# letters to apper and disapper
LETTER_VISIBLE_MIN_X = 36   # Letters become visible at
LETTER_VISIBLE_MAX_X = 124  # Letters disappear at
LETTER_RESET_X = 5 # at this coordinate letters reset back to right ! only coordinates change not real reset
LETTERS_DISTANCE = 14 # spacing between letters
LETTERS_END = LETTER_VISIBLE_MIN_X + 26 * LETTERS_DISTANCE # 27 symbols (letters + special) but 26 gaps

# Enemies
MAX_ENEMIES = 6
ENEMY_MIN_X = -16
ENEMY_MAX_X = WIDTH + 16  
ENEMY_Y_MIN = 50
ENEMY_Y_MAX = 150
ENEMY_ANIM_SWITCH_RATE = 15
ENEMY_Y_MIN_SEPARATION = 16  

# zapper
ZAPPER_COLOR = (252,252,84,255)
MAX_ZAPPER_POS = 49
ZAPPER_SPR_WIDTH = 4
ZAPPER_SPR_HEIGHT = 200 # this is approximate, can also be changed, but this works fine. TODO define this value based on max/min ship coordinate 


TIME = 99

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

    letters_x: chex.Array # letters at the top
    letters_y: chex.Array
    letters_char: chex.Array
    letters_alive: chex.Array
    letters_speed: chex.Array
    letters_positions: chex.Array
    # Add these for word logic:
    current_word: chex.Array
    current_letter_index: chex.Array
    # shape: (1,3) -> [x, y, direction]
    player_missile_position: chex.Array  
    # shape: (1,4) -> x, y, active, cooldown
    player_zapper_position: chex.Array 
    enemy_positions: chex.Array  # shape (MAX_ENEMIES, 4): x, y, type, vx
    enemy_active: chex.Array     # shape (MAX_ENEMIES,)
    enemy_global_spawn_timer: chex.Array



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
    letters: jnp.ndarray # Shape (27, 5) - 27 letters+special each with x,y,w,h,active

    letters_char: jnp.ndarray 
    letters_alive: jnp.ndarray  # active letters

    current_word: jnp.ndarray  # word to form
    current_letter_index: jnp.ndarray  # current position in word

    enemies: jnp.ndarray 
    player_missile: EntityPosition
    player_zapper: EntityPosition

    cooldown_timer: jnp.ndarray
    timer: jnp.ndarray

class WordZapperInfo(NamedTuple):
    timer: jnp.ndarray
    current_word: jnp.ndarray
    game_over: jnp.ndarray

def load_sprites():
    """Load all sprites required for Word Zapper rendering."""

    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Background
    bg1 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/wordzapper/bg/1.npy"))

    # Player
    pl_1 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/wordzapper/player/1.npy"))
    pl_2 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/wordzapper/player/2.npy"))

    pl_missile = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/wordzapper/bullet/1.npy"))

    # Enemies Bonker
    bonker_1 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/wordzapper/enemies/bonker/1.npy"))
    bonker_2 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/wordzapper/enemies/bonker/2.npy"))

    # Enemies Zonker
    zonker_1 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/wordzapper/enemies/zonker/1.npy"))
    zonker_2 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/wordzapper/enemies/zonker/2.npy"))

    # Loading Digit Sprites
    digits = aj.load_and_pad_digits(os.path.join(MODULE_DIR, "sprites/wordzapper/digits/{}.npy"))    
 
    # Letters above 
    letters = [aj.loadFrame(os.path.join(MODULE_DIR, f"sprites/wordzapper/letters/normal_letters/{chr(i)}.npy")) for i in range(ord('a'), ord('z') + 1)] # MAYBE SPECIAL CHAR MISSING
    
    special = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/wordzapper/letters/normal_letters/1special_symbol.npy"))
    letters.append(special)

    # Pad player sprites to match
    pl_sub_sprites = aj.pad_to_match([pl_1, pl_2])
    
    pl_missile_sprites = [pl_missile]

    # Pad bonker sprites to match 
    bonker_sprites = aj.pad_to_match([bonker_1, bonker_2])

    #Pad zonker sprites to match
    zonker_sprites = aj.pad_to_match([zonker_1, zonker_2])
    
    # pad letters with special special_symbol
    letters = aj.pad_to_match(letters)


    SPRITE_BG = jnp.expand_dims(bg1, axis=0)
    
    SPRITE_PL_MISSILE = jnp.repeat(pl_missile_sprites[0][None], 1, axis=0)

    SPRITE_PL = jnp.concatenate(
        [
            jnp.repeat(pl_sub_sprites[0][None], 4, axis=0),
            jnp.repeat(pl_sub_sprites[1][None], 4, axis=0),
        ]
    )
    
    SPRITE_BONKER = jnp.concatenate(
        [
            jnp.repeat(bonker_sprites[0][None], 4, axis=0),
            jnp.repeat(bonker_sprites[1][None], 4, axis=0),
        ]
    )

    SPRITE_ZONKER = jnp.concatenate(
        [
            jnp.repeat(zonker_sprites[0][None], 4, axis=0),
            jnp.repeat(zonker_sprites[1][None], 4, axis=0),
        ]
    )

    DIGITS = digits

    LETTERS = jnp.stack(letters, axis=0)  

    return (
        SPRITE_BG,
        SPRITE_PL,
        SPRITE_PL_MISSILE,
        SPRITE_BONKER,
        SPRITE_ZONKER,
        DIGITS,
        LETTERS,
    )
(
    SPRITE_BG,
    SPRITE_PL,
    SPRITE_PL_MISSILE,
    SPRITE_BONKER,
    SPRITE_ZONKER,
    DIGITS,
    LETTERS,  
) = load_sprites()




@jax.jit
def player_step(
    state: WordZapperState, action: chex.Array
) -> tuple[chex.Array, chex.Array, chex.Array]:
    '''
    implement all the possible movement directions for the player, the mapping is:
    anything with left in it, add -2 to the x position
    anything with right in it, add 2 to the x position
    anything with up in it, add -2 to the y position
    anything with down in it, add 2 to the y position
    '''
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
    new_letters_x = letters_x - letters_speed

    reset_x = jnp.max(new_letters_x) + LETTERS_DISTANCE

    new_letters_x = jnp.where(
        new_letters_x < LETTER_RESET_X,
        reset_x,
        new_letters_x
    )
    return new_letters_x


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
                action == Action.UPFIRE, ## TODO downfire upfire are not missile step? right? right? look at frate extractor, when running it in terminal you can see key mappings
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
            [new_missile[0] + new_missile[2] * 5, new_missile[1], new_missile[2]]
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


@jax.jit
def player_zapper_step(
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
    zapper_exists = state.player_zapper_position[2]

    # if the player shot and there is no missile in frame, then we can shoot a missile
    # the missile y is the current player y position + 7
    # the missile x is either player x + 3 if facing left or player x + 13 if facing right
    new_zapper = jnp.where(
        jnp.logical_and(fire, jnp.logical_not(zapper_exists)),
        # TODO remove hard-coded values below
        jnp.array([curr_player_x+6, curr_player_y-2, 1, state.step_counter]),
        state.player_zapper_position,
    )
    
    new_deactive_zapper = jnp.array([
        state.player_zapper_position[0],
        state.player_zapper_position[1],
        0,
        state.player_zapper_position[3]
    ])
    
    new_active_zapper = jnp.array([
        state.player_zapper_position[0],
        state.player_zapper_position[1],
        1,
        state.player_zapper_position[3]
    ])


    delta = jnp.abs(state.step_counter - new_zapper[3])

    out_zapper = jnp.where(
        delta < 10,
        new_zapper,
        jnp.where(
            delta < 25,
            new_deactive_zapper,
            jnp.where(
                delta < 40,
                new_active_zapper,
                jnp.array([0, 0, 0, 0])
            )
        )
    )

    return out_zapper

@jax.jit
def enemy_step(state: WordZapperState) -> Tuple[chex.Array, chex.PRNGKey]:
    rng_key, subkey = jax.random.split(state.rng_key)

    # Move enemies left
    enemy_positions = state.enemy_positions

    new_x = enemy_positions[:, 0] - enemy_positions[:, 2]
    new_active = jnp.where(new_x < -16, 0, enemy_positions[:, 4])

    new_frame_index = (state.step_counter // 30) % 2

    updated_enemy_positions = jnp.stack(
        [
            new_x,
            enemy_positions[:, 1],
            enemy_positions[:, 2],
            jnp.full_like(enemy_positions[:, 3], new_frame_index),
            new_active,
        ],
        axis=1,
    )

    # Spawn logic
    spawn_chance = jax.random.uniform(subkey) < 0.1

    empty_slot = jnp.argmax(enemy_positions[:, 4] == 0)

    spawn_x = jnp.array(WIDTH + 16)
    spawn_y = jax.random.randint(subkey, (), 40, HEIGHT - 40)
    spawn_speed = jnp.array(2)
    spawn_frame = jnp.array(0)
    spawn_active = jnp.array(1)

    def do_spawn(pos):
        return pos.at[empty_slot].set(
            jnp.array([spawn_x, spawn_y, spawn_speed, spawn_frame, spawn_active])
        )

    updated_enemy_positions = jax.lax.cond(
        spawn_chance,
        do_spawn,
        lambda pos: pos,
        updated_enemy_positions,
    )

    return updated_enemy_positions, rng_key


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
        # TODO get rid of these for more elegant solution, these are reused in reset below
        letters_x = jnp.linspace(WIDTH, WIDTH + 25 * 14, 27) # 12px apart, offscreen right
        letters_y = jnp.full((27,), 30)  # All at y=30

        reset_state = WordZapperState(
            player_x=jnp.array(PLAYER_START_X),
            player_y=jnp.array(PLAYER_START_Y),
            player_speed=jnp.array(0),
            player_direction=jnp.array(0),
            cooldown_timer=jnp.array(0),
            asteroid_x=jnp.array(0),
            asteroid_y=jnp.array(0),
            asteroid_speed=jnp.array(0),  
            asteroid_alive=jnp.array(0),
            asteroid_positions=jnp.zeros((MAX_ASTEROIDS_COUNT, 3)),
            
            enemy_positions=jnp.zeros((MAX_ENEMIES, 5)),          
            enemy_active=jnp.zeros((MAX_ENEMIES,), dtype=jnp.int32), 
            enemy_global_spawn_timer=jnp.array(60),  # start with 60 frames wait


            player_missile_position=jnp.zeros(3), 
            player_zapper_position=jnp.zeros(4),
            
            letters_x = jnp.linspace(LETTER_VISIBLE_MIN_X, LETTERS_END, 27), # 12px apart, offscreen right
            letters_y = jnp.full((27,), 30),  # All at y=30
            letters_char = jnp.arange(27),  # A-Z and special
            letters_alive = jnp.ones((27,), dtype=jnp.int32),
            letters_speed = jnp.ones((27,)) * 1,  # All move at 1px/frame
            letters_positions = jnp.stack([letters_x, letters_y], axis=1),  # shape (27,2)
            current_word = jnp.array([0, 1, 2, 3, 4]),  # Example: word "ABCDE"
            current_letter_index = jnp.array(0),


            timer=jnp.array(TIME),

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
        # Enemies
        def convert_enemy(pos, active):
            return jnp.array([
                pos[0],  # x
                pos[1],  # y
                16,      # width (you can adjust this to your enemy sprite size)
                16,      # height
                active,  # active flag
            ])

        enemies = jax.vmap(convert_enemy)(state.enemy_positions, state.enemy_active)


        # Convert letter positions into the correct entity format
        def convert_to_entity(pos, size):
            return jnp.array([
                pos[0],  # x position
                pos[1],  # y position
                size[0],  # width
                size[1],  # height
                1,        # active flag (use state.letters_alive if needed)
            ])
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

        zapper_pos = state.player_zapper_position
        player_zapper = EntityPosition(
            x=zapper_pos[0],
            y=zapper_pos[1],
            width=jnp.array(ZAPPER_SIZE[0]),
            height=jnp.array(ZAPPER_SIZE[1]),
            active=zapper_pos[2],
        )

        return WordZapperObservation(
            player=player,
            asteroids=asteroids,
            enemies=enemies,
            letters=letters,
            letters_char=state.letters_char,
            letters_alive=state.letters_alive,
            current_word=state.current_word,
            current_letter_index=state.current_letter_index,
            player_missile=player_missile,
            player_zapper=player_zapper,
            cooldown_timer=state.cooldown_timer,
            timer=state.timer,
        )

    
    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: WordZapperState) -> bool:
        """Check if the game should end due to timer expiring."""
        MAX_TIME = 60 * TIME  # 90 seconds at 60 FPS
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

            # ----------------- Player movement -----------------
            new_player_x, new_player_y, new_player_direction = player_step(
                state,
                action,
            )

            new_timer = jnp.where(
                jnp.bitwise_and(jnp.bitwise_and((state.step_counter % 60 == 0), (state.step_counter > 0)), (state.timer > 0)),
                state.timer - 1,
                state.timer,
            )

            player_missile_position = player_missile_step(
                state, new_player_x, new_player_y, action
            )

            player_zapper_position = player_zapper_step(
                state, new_player_x, new_player_y, action
            )

            new_step_counter = jnp.where(
                state.step_counter == 1024,
                jnp.array(0),
                state.step_counter + 1,
            )

            # ----------------- Enemies movement -----------------
            new_enemy_positions = state.enemy_positions.at[:,0].add(state.enemy_positions[:,3] * 1.2)  # reduced speed 20%

            # Out of screen → deactivate
            new_enemy_active = jnp.where(
                jnp.logical_or(
                    new_enemy_positions[:,0] < ENEMY_MIN_X - 16,
                    new_enemy_positions[:,0] > ENEMY_MAX_X + 16
                ),
                0,
                state.enemy_active
            )

            # ----------------- Enemy spawn logic -----------------
            # Decrease global spawn timer
            new_enemy_global_spawn_timer = state.enemy_global_spawn_timer - 1
            new_enemy_global_spawn_timer = jnp.clip(new_enemy_global_spawn_timer, 0, 9999)

            # Check if any inactive enemies exist
            has_free_slot = jnp.any(new_enemy_active == 0)

            # Cond: spawn if timer == 0 and there is a free slot
            spawn_cond = jnp.logical_and(new_enemy_global_spawn_timer == 0, has_free_slot)

            # Define spawn_one_enemy_fn with Y overlap protection
            def spawn_one_enemy_fn(rng_key_in, existing_enemy_positions, existing_enemy_active):
                rng_key_out, subkey_dir = jax.random.split(rng_key_in)
                rng_key_out, subkey_y = jax.random.split(rng_key_out)
                rng_key_out, subkey_type = jax.random.split(rng_key_out)

                direction = jax.random.choice(subkey_dir, jnp.array([-1.0, 1.0]))

                # Fix → x_pos based on direction:
                x_pos = jnp.where(direction == 1.0, ENEMY_MIN_X, ENEMY_MAX_X)

                # Y position sampling with collision avoidance (same as before):
                def sample_valid_y(subkey_y_inner):
                    def body_fn(val):
                        y_candidate, subkey_y_inner = val
                        subkey_y_inner, next_subkey = jax.random.split(subkey_y_inner)
                        y_candidate_new = jax.random.randint(next_subkey, shape=(), minval=ENEMY_Y_MIN, maxval=ENEMY_Y_MAX)

                        dists = jnp.abs(existing_enemy_positions[:,1] - y_candidate_new)
                        collision = jnp.any(jnp.logical_and(existing_enemy_active == 1, dists < 16))

                        y_candidate_new = jnp.where(collision, y_candidate, y_candidate_new)
                        return (y_candidate_new, next_subkey)

                    def cond_fn(val):
                        y_candidate, subkey = val
                        dists = jnp.abs(existing_enemy_positions[:,1] - y_candidate)
                        collision = jnp.any(jnp.logical_and(existing_enemy_active == 1, dists < 16))
                        return collision

                    init_y = jax.random.randint(subkey_y_inner, shape=(), minval=ENEMY_Y_MIN, maxval=ENEMY_Y_MAX)
                    final_y, _ = jax.lax.while_loop(cond_fn, body_fn, (init_y, subkey_y_inner))
                    return final_y

                y_pos = sample_valid_y(subkey_y)

                enemy_type = jax.random.randint(subkey_type, shape=(), minval=0, maxval=2)

                new_enemy = jnp.array([x_pos, y_pos, enemy_type, direction, 1.0])
                return new_enemy, rng_key_out


            def spawn_enemy_branch(carry):
                positions, active, global_timer, rng_key_inner = carry

                free_idx = jnp.argmax(active == 0)

                new_enemy, rng_key_out = spawn_one_enemy_fn(rng_key_inner, positions, active)

                positions = positions.at[free_idx].set(new_enemy)
                active = active.at[free_idx].set(1)
                global_timer = jax.random.randint(rng_key_out, shape=(), minval=30, maxval=120)

                return positions, active, global_timer, rng_key_out

            def no_spawn_branch(carry):
                return carry

            # Apply conditional spawn
            positions, active, global_timer, rng_key = jax.lax.cond(
                spawn_cond,
                spawn_enemy_branch,
                no_spawn_branch,
                (new_enemy_positions, new_enemy_active, new_enemy_global_spawn_timer, state.rng_key)
            )

            new_state = state._replace(
                player_x=new_player_x,
                player_y=new_player_y,
                player_direction=new_player_direction,
                player_missile_position=player_missile_position,
                player_zapper_position=player_zapper_position,
                enemy_positions=positions,
                enemy_active=active,
                enemy_global_spawn_timer=global_timer,
                step_counter=new_step_counter,
                rng_key=rng_key,
                timer=new_timer,
            )

            # Scroll letters
            new_letters_x = scrolling_letters(state.letters_x, state.letters_speed, state.letters_alive)

            new_state = new_state._replace(
                letters_x=new_letters_x,
            )

            return new_state

        # Apply game step
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

        # render background
        frame_bg = aj.get_sprite_frame(SPRITE_BG, 0)
        raster = aj.render_at(raster, 0, 0, frame_bg)


        # render player

        # show the countdown timer
        timer_array = aj.int_to_digits(state.timer, max_digits=2)
        raster = aj.render_label(raster, 70, 10, timer_array, DIGITS, spacing=10)

        ## render player
        frame_pl_sub = aj.get_sprite_frame(SPRITE_PL, state.step_counter)
        raster = aj.render_at(
            raster,
            state.player_x,
            state.player_y,
            frame_pl_sub,
            flip_horizontal=state.player_direction == FACE_LEFT,
        )

        # render player missile
        frame_pl_torp = aj.get_sprite_frame(SPRITE_PL_MISSILE, state.step_counter)

        should_render_torp = state.player_missile_position[0] > 0
        
        raster = jax.lax.cond(
            should_render_torp,
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

        # render player zapper
        zapper_spr = jnp.full((ZAPPER_SPR_WIDTH, ZAPPER_SPR_HEIGHT, 4), jnp.asarray(ZAPPER_COLOR, dtype=jnp.uint8), dtype=jnp.uint8)

        raster = jax.lax.cond(
            state.player_zapper_position[2],
            lambda r : aj.render_at(
                r,
                state.player_zapper_position[0],
                MAX_ZAPPER_POS,
                zapper_spr * (jnp.arange(ZAPPER_SPR_HEIGHT) < state.player_zapper_position[1] - MAX_ZAPPER_POS).astype(jnp.uint8)[None, :, None]
                # this is used to mask parts of zapper_spr, as it is longer to make impression of dynamic length
            ),
            lambda r : r,
            raster,
        )

        # render enemies
        frame_bonker = aj.get_sprite_frame(SPRITE_BONKER, state.step_counter)
        frame_zonker = aj.get_sprite_frame(SPRITE_ZONKER ,state.step_counter)

        def body_fn(i, raster):
            should_render_enemy = state.enemy_active[i]
            x = state.enemy_positions[i, 0]
            y = state.enemy_positions[i, 1]
            enemy_type = state.enemy_positions[i, 2].astype(jnp.int32)

            raster = jax.lax.cond(
                should_render_enemy,
                lambda r: jax.lax.cond(
                    enemy_type == 0,
                    lambda rr: aj.render_at(rr, x, y, frame_bonker),
                    lambda rr: aj.render_at(rr, x, y, frame_zonker),
                    r
                ),
                lambda r: r,
                raster
            )
            return raster

        raster = jax.lax.fori_loop(0, MAX_ENEMIES, body_fn, raster)

        # Render letters
        def render_letter(i, raster):
            is_alive = state.letters_alive[i]
            x = state.letters_x[i]
            y = state.letters_y[i]
            char_idx = state.letters_char[i]
            sprite = aj.get_sprite_frame(LETTERS, char_idx)  # (H, W, C)
        
            def render_visible(r):
                sprite_w, sprite_h = sprite.shape[:2]
        
                # x positions of each column of the sprite
                sprite_xs = x + jnp.arange(sprite_w)
        
                # create mask where x is within bounds
                x_mask = jnp.logical_and((sprite_xs >= LETTER_VISIBLE_MIN_X), (sprite_xs < LETTER_VISIBLE_MAX_X)).astype(sprite.dtype)
                
                # expand themask
                x_mask = x_mask[:, None, None]  # (W, 1, 1)

                # apply mask to zero out offscreen pixels
                masked_sprite = sprite * x_mask
        
                return aj.render_at(r, x, y, masked_sprite)
            
            
            return jax.lax.cond(is_alive == 1, render_visible, lambda r: r, raster)
        
    
        raster = jax.lax.fori_loop(0, state.letters_x.shape[0], render_letter, raster)

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
