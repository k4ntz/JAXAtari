import os
from functools import partial
from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
import chex

import jaxatari.spaces as spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import JAXGameRenderer
import jaxatari.rendering.jax_rendering_utils as jr


class WordZapperConstants(NamedTuple) :
    # define object orientations
    FACE_LEFT = -1
    FACE_RIGHT = 1

    # Player 
    PLAYER_START_X = 36
    PLAYER_START_Y = 135
    
    # Pygame window dimensions
    WIDTH = 160
    HEIGHT = 210

    X_BOUNDS = (10, 134) # (min X, max X)
    Y_BOUNDS = (56, 135)

    # Object sizes (width, height)
    PLAYER_SIZE = (16, 12)
    MISSILE_SIZE = (8, 2)
    ZAPPER_SIZE = (4, Y_BOUNDS[1])
    LETTER_SIZE = (8, 8)
    BONKER_SIZE = (5, 5)
    ZONKER_SIZE = (7, 10)

    # letters to apper and disapper
    LETTER_VISIBLE_MIN_X = 36   # Letters become visible at
    LETTER_VISIBLE_MAX_X = 124  # Letters disappear at
    LETTER_RESET_X = 5 # at this coordinate letters reset back to right ! only coordinates change not real reset
    LETTERS_DISTANCE = 17 # spacing between letters
    LETTERS_END = LETTER_VISIBLE_MIN_X + 26 * LETTERS_DISTANCE # 27 symbols (letters + special) but 26 gaps
    LETTER_COOLDOWN = 200 # cooldown after letters zapperd till they reappear
    LETTER_SCROLLING_SPEED = 1 # speed at which letters move left

    # Enemies
    MAX_ENEMIES = 6
    ENEMY_MIN_X = -16
    ENEMY_MAX_X = WIDTH + 16
    ENEMY_Y_MIN = 55
    ENEMY_Y_MAX = 133
    ENEMY_ANIM_SWITCH_RATE = 2
    ENEMY_Y_MIN_SEPARATION = 16

    ENEMY_GAME_SPEED = 0.7
    LEVEL_PAUSE_FRAMES = 4 * 60

    # zapper
    ZAPPER_COLOR = (252,252,84,255)
    MAX_ZAPPER_POS = 49
    ZAPPER_SPR_WIDTH = ZAPPER_SIZE[0]
    ZAPPER_SPR_HEIGHT = ZAPPER_SIZE[1] # we assume this max zapper height
    
    ZAPPING_BOUNDS = (LETTER_VISIBLE_MIN_X, LETTER_VISIBLE_MAX_X - ZAPPER_SPR_WIDTH) # min x, max x
    
    PLAYER_ZAPPER_COOLDOWN_TIME = 32 # amount letters stop moving and zapper is active
    ZAPPER_BLOCK_TIME = 50 # dont allow zapper action during this time

    LEVEL_WORD_LENGTHS = (4, 5, 6)
    SPECIAL_CHAR_INDEX = 26
    
    TIME = 99

    ENEMY_EXPLOSION_FRAME_DURATION = 8  # Number of ticks per explosion frame
    ENEMY_EXPLOSION_FRAMES = 4          # NEW: number of explosion frames/sprites
    # Letter explosion animation
    LETTER_EXPLOSION_FRAME_DURATION = 8
    LETTER_EXPLOSION_FRAMES = 4


WORD_LIST = [
    ["WAVE", "BYTE", "NODE", "BEAM", "SHIP", "CODE", "GRID"],
    ["PIXEL", "ROBOT", "LASER", "POWER", "SMART", "INPUT", "GHOST"],
    ["BONKER","ZONKER", "ROCKET", "PLAYER", "VECTOR", "BINARY", "MATRIX"]
]

def _encode_word(word, max_len=6):
    vals = [ord(c) - 65 for c in word] + [-1] * (max_len - len(word))
    return jnp.array(vals, dtype=jnp.int32)

# Encode each bank to a fixed (7, 6) array (padded to 6 with -1s)
ENCODED_WORD_LIST_4 = jnp.stack([_encode_word(w, 6) for w in WORD_LIST[0]])
ENCODED_WORD_LIST_5 = jnp.stack([_encode_word(w, 6) for w in WORD_LIST[1]])
ENCODED_WORD_LIST_6 = jnp.stack([_encode_word(w, 6) for w in WORD_LIST[2]])

@jax.jit
def choose_target_word(rng_key: jax.random.PRNGKey, lvl_word_len: chex.Array) -> tuple[chex.Array, jax.random.PRNGKey]:
    def pick_bank(lvl_word_len: chex.Array) -> chex.Array:
        """
        select encoded word list based on level
        """
        idx = jnp.where(lvl_word_len == 4, 0, jnp.where(lvl_word_len == 5, 1, 2))
        return jax.lax.switch(
            idx,
            (
                lambda: ENCODED_WORD_LIST_4,  # (7, 6)
                lambda: ENCODED_WORD_LIST_5,  # (7, 6)
                lambda: ENCODED_WORD_LIST_6,  # (7, 6)
            ),
        )

    bank = pick_bank(lvl_word_len)
    n = bank.shape[0]
    rng_key, sub = jax.random.split(rng_key)
    idx = jax.random.randint(sub, (), 0, n, dtype=jnp.int32)

    return bank[idx], rng_key


STATE_TRANSLATOR: dict = {
    0: "player_x",
    1: "player_y",
    2: "letters_x",
    3: "letters_y",
    4: "letters_char",
    5: "letters_alive",
    6: "letters_speed",
    7: "current_letter_index",
    8: "player_score",
    9: "timer",
    10: "step_counter",
    11: "buffer",
}

class WordZapperState(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    player_direction: chex.Array

    letters_x: chex.Array # letters at the top
    letters_y: chex.Array
    letters_char: chex.Array
    letters_alive: chex.Array # (27, 2) -> # letter_is_alive, letter_cooldown
    letters_speed: chex.Array
    letters_positions: chex.Array

    current_letter_index: chex.Array
    
    player_missile_position: chex.Array  # shape: (4,) -> x, y, active, direction
    player_zapper_position: chex.Array  # shape: (7,) -> x, y, active, cooldown, pulse, initial_x, block_zapper
                                        # (initial_x  keeps track of which letter was zapped)

    enemy_positions: chex.Array  # shape (MAX_ENEMIES, 4): x, y, type, vx
    enemy_active: chex.Array     # shape (MAX_ENEMIES,)
    enemy_global_spawn_timer: chex.Array

    enemy_explosion_frame: chex.Array  # shape (MAX_ENEMIES,) - explosion frame index (0=none, 1-3=anim)
    enemy_explosion_timer: chex.Array  # shape (MAX_ENEMIES,) - ticks left for explosion anim
    enemy_explosion_frame_timer: chex.Array  # shape (MAX_ENEMIES,)
    enemy_explosion_pos: chex.Array  # shape (MAX_ENEMIES, 2) - position where explosion anim is rendered

    target_word: chex.Array
    game_phase: chex.Array  # 0 -> game start, show word, player can't move
                            # 1 -> gameplay
                            # 2 -> player return back animation, player can't move
    phase_timer: chex.Array    

    timer: chex.Array
    step_counter: chex.Array
    rng_key: chex.PRNGKey
    score: chex.Array

    # Letter explosion animation state
    letter_explosion_frame: chex.Array  # shape (27,)
    letter_explosion_timer: chex.Array  # shape (27,)
    letter_explosion_frame_timer: chex.Array  # shape (27,)
    letter_explosion_pos: chex.Array  # shape (27, 2)
    level_word_len: chex.Array       # 4/5/6 letters
    waiting_for_special: chex.Array  # 1 once all letters collected; then require shooting special

    finised_level_count: chex.Array 

class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray
    active: jnp.ndarray

class PlayerEntity(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    direction: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray
    active: jnp.ndarray

class WordZapperObservation(NamedTuple):
    player: PlayerEntity

    player_missile: EntityPosition
    player_zapper: jnp.ndarray # shape (6,) -> x, y, width, height, active, cooldown

    letters: jnp.ndarray # shape (27, 6) -> x, y, width, height, active, char_id
    current_letter_index: jnp.ndarray
    target_word: jnp.ndarray     # shape (max_word_len,) max_word_len=6

    enemies: jnp.ndarray # shape (MAX_ENEMIES, 5) -> x, y, width, height, active

    score: jnp.ndarray
    game_phase: jnp.ndarray
    level_word_len: jnp.ndarray
    waiting_for_special: jnp.ndarray

# Info container for debugging / logging
class WordZapperInfo(NamedTuple):
    score: jnp.ndarray
    step_counter: jnp.ndarray
    timer: jnp.ndarray
    finished_level_count: jnp.ndarray
    all_rewards: jnp.ndarray


def load_sprites():
    """Load all sprites required for Word Zapper rendering."""
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Background
    bg1 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wordzapper/bg/1.npy"))

    # Player
    pl_1 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wordzapper/player/1.npy"))
    pl_2 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wordzapper/player/2.npy"))
    pl_missile = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wordzapper/bullet/1.npy"))

    # Enemies Bonker
    bonker_1 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wordzapper/enemies/bonker/1.npy"))
    bonker_2 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wordzapper/enemies/bonker/2.npy"))

    # Enemies Zonker
    zonker_1 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wordzapper/enemies/zonker/1.npy"))
    zonker_2 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wordzapper/enemies/zonker/2.npy"))

    # yellow letters
    yellow_letters = [jr.loadFrame(os.path.join(MODULE_DIR, f"sprites/wordzapper/letters/yellow_letters/{chr(i)}.npy")) for i in range(ord('a'), ord('z') + 1)]

    qmark = jr.loadFrame(
        os.path.join(MODULE_DIR, "sprites/wordzapper/special/qmark.npy")
    )
    
    # Loading Digit Sprites
    digits = jr.load_and_pad_digits(os.path.join(MODULE_DIR, "sprites/wordzapper/digits/{}.npy"))    
 
    # Letters above 
    letters = [jr.loadFrame(os.path.join(MODULE_DIR, f"sprites/wordzapper/letters/normal_letters/{chr(i)}.npy")) for i in range(ord('a'), ord('z') + 1)] # MAYBE SPECIAL CHAR MISSING
    
    special = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wordzapper/letters/normal_letters/1special_symbol.npy"))
    letters.append(special)

    # enemy explosion
    exp1 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wordzapper/explosions/enemy_explosions/exp1.npy"))
    exp2 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wordzapper/explosions/enemy_explosions/exp2.npy"))
    exp3 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wordzapper/explosions/enemy_explosions/exp3.npy"))
    exp4 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/wordzapper/explosions/enemy_explosions/exp4.npy"))

    # Letter explosion sprites
    lexp_frames = [
        jr.loadFrame(os.path.join(MODULE_DIR, f"sprites/wordzapper/explosions/normal_explosions/{i}.npy"))
        for i in range(1, 5)
    ]
    lexp_frames_padded, _ = jr.pad_to_match(lexp_frames)

    # Pad player sprites to match
    pl_sub_sprites, pl_sub_offsets = jr.pad_to_match([pl_1, pl_2])
    pl_sub_offsets = jnp.array(pl_sub_offsets)

    bonker_sprites, bonker_offsets = jr.pad_to_match([bonker_1, bonker_2])
    bonker_offsets = jnp.array(bonker_offsets)

    zonker_sprites, zonker_offsets = jr.pad_to_match([zonker_1, zonker_2])
    zonker_offsets = jnp.array(zonker_offsets)

    yellow_letters, yellow_letters_offsets = jr.pad_to_match(yellow_letters)
    yellow_letters_offsets = jnp.array(yellow_letters_offsets)
    
    # pad letters with special special_symbol
    letters, letters_offsets = jr.pad_to_match(letters)
    letters_offsets = jnp.array(letters_offsets)


    SPRITE_BG = jnp.expand_dims(bg1, axis=0)
    
    SPRITE_PL_MISSILE = pl_missile
    
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


    # Loading Digit Sprites
    DIGITS = jr.load_and_pad_digits(
        os.path.join(MODULE_DIR, "sprites/wordzapper/digits/{}.npy")
    )

    YELLOW_LETTERS = jnp.stack(yellow_letters, axis=0)

    QMARK_SPRITE = qmark

    DIGITS = digits

    LETTERS = jnp.stack(letters, axis=0)  
    
    ENEMY_EXPLOSION_SPRITES = jnp.stack([exp1, exp2, exp3, exp4], axis=0)
    LETTER_EXPLOSION_SPRITES = jnp.stack(lexp_frames_padded, axis=0)

    return (
        SPRITE_BG,
        SPRITE_PL,
        SPRITE_PL_MISSILE,
        SPRITE_BONKER,
        SPRITE_ZONKER,
        DIGITS,
        YELLOW_LETTERS,
        QMARK_SPRITE,
        LETTERS,
        ENEMY_EXPLOSION_SPRITES,
        LETTER_EXPLOSION_SPRITES,
        pl_sub_offsets,
        bonker_offsets,
        zonker_offsets,
        yellow_letters_offsets,
        letters_offsets,
    )

(
    SPRITE_BG,
    SPRITE_PL,
    SPRITE_PL_MISSILE,
    SPRITE_BONKER,
    SPRITE_ZONKER,
    DIGITS,
    YELLOW_LETTERS,
    QMARK_SPRITE,
    LETTERS,
    ENEMY_EXPLOSION_SPRITES,
    LETTER_EXPLOSION_SPRITES,
    PL_OFFSETS,
    BONKER_OFFSETS,
    ZONKER_OFFSETS,
    YELLOW_LETTERS_OFFSETS,
    LETTERS_OFFSETS,
    ) = load_sprites()


@jax.jit
def player_step(
    state: WordZapperState, action: chex.Array, consts: WordZapperConstants
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
            right,
            state.player_x + 2,
            jnp.where(
                left,
                state.player_x - 2,
                state.player_x
            )
        )

    player_y = jnp.where(
            down,
            state.player_y + 2,
            jnp.where(
                up,
                state.player_y - 2,
                state.player_y
            )
        )
    
    player_direction = jnp.where(right, 1, jnp.where(left, -1, state.player_direction))

    player_x = jnp.where(
        player_x < consts.X_BOUNDS[0],
        consts.X_BOUNDS[0],
        jnp.where(
            player_x > consts.X_BOUNDS[1],
            consts.X_BOUNDS[1],
            player_x,
        ),
    )

    player_y = jnp.where(
        player_y < consts.Y_BOUNDS[0],
        consts.Y_BOUNDS[0],
        jnp.where(
            player_y > consts.Y_BOUNDS[1],
            consts.Y_BOUNDS[1],
            player_y
        ),
    )


    return player_x, player_y, player_direction


@jax.jit
def scrolling_letters(state: WordZapperState, consts: WordZapperConstants) -> chex.Array:

    new_letters_x = state.letters_x - state.letters_speed

    reset_x = jnp.max(new_letters_x) + consts.LETTERS_DISTANCE

    # Track which letters are being reset (scrolled off screen)
    just_reset = (new_letters_x < consts.LETTER_RESET_X)

    # Letters reappear as soon as they are fully off screen and cooldown is over
    new_letters_alive = state.letters_alive
    # Decrement cooldown for all letters
    new_letters_alive = new_letters_alive.at[:, 1].set(
        jnp.where(
            new_letters_alive[:, 1] > 0,
            new_letters_alive[:, 1] - 1,
            new_letters_alive[:, 1]
        )
    )
    # Revive letters that are being reset to the rightmost position (i.e., when a new alphabet sequence starts)
    # Only revive if the letter is currently not alive (was shot)
    just_reset = (new_letters_x < consts.LETTER_RESET_X)
    new_letters_alive = new_letters_alive.at[:, 0].set(
        jnp.where(
            just_reset & (new_letters_alive[:, 0] == 0),
            1,
            new_letters_alive[:, 0]
        )
    )

    # Actually reset the x position for letters that scrolled off
    new_letters_x = jnp.where(
        state.player_zapper_position[2], # if zapper active
        state.letters_x,
        jnp.where(
            just_reset,
            reset_x,
            new_letters_x
        )
    )
    
    # zap if zapper is within the bounds of a letter
    zapper_x = state.player_zapper_position[5]
    letter_half_width = consts.LETTER_SIZE[0]
    letter_lefts = state.letters_x - letter_half_width
    letter_rights = state.letters_x + letter_half_width

    # Find all letters where zapper_x is within bounds
    in_bounds_mask = jnp.logical_and(zapper_x >= letter_lefts, zapper_x < letter_rights)

    # Only consider letters that are visible
    visible_mask = jnp.logical_and(state.letters_x >= consts.LETTER_VISIBLE_MIN_X, state.letters_x < consts.LETTER_VISIBLE_MAX_X)
    valid_mask = jnp.logical_and(in_bounds_mask, visible_mask)
    any_in_bounds = jnp.any(valid_mask)

    def get_first_in_bounds():
        return jnp.argmax(valid_mask)
    
    def get_invalid():
        return -1
    
    target_letter_id = jax.lax.cond(any_in_bounds, get_first_in_bounds, get_invalid)

    def zap_letter(args):
        l, frame, timer, frame_timer, pos = args
        l = l.at[target_letter_id].set(jnp.array([0, consts.LETTER_COOLDOWN], dtype=jnp.int32))
        frame = frame.at[target_letter_id].set(1)
        timer = timer.at[target_letter_id].set(consts.LETTER_EXPLOSION_FRAMES)
        frame_timer = frame_timer.at[target_letter_id].set(consts.LETTER_EXPLOSION_FRAME_DURATION)
        pos = pos.at[target_letter_id].set(jnp.array([state.letters_x[target_letter_id], state.letters_y[target_letter_id]]))
        return l, frame, timer, frame_timer, pos

    def no_zap(args):
        l, frame, timer, frame_timer, pos = args
        return l, frame, timer, frame_timer, pos


    # Only allow zapping if the letter is alive (not already shot/disappeared)
    is_letter_alive = jnp.where(target_letter_id != -1, state.letters_alive[target_letter_id, 0] == 1, False)
    zap_condition = jnp.logical_and(state.player_zapper_position[2], jnp.logical_and(target_letter_id != -1, is_letter_alive))

    def safe_zap_letter(args):
        l, frame, timer, frame_timer, pos = args
        already_exploding = frame[target_letter_id] > 0
        return jax.lax.cond(already_exploding, no_zap, zap_letter, (l, frame, timer, frame_timer, pos))
    
    (
        new_letters_alive,
        letter_explosion_frame,
        letter_explosion_timer,
        letter_explosion_frame_timer,
        letter_explosion_pos
    ) = jax.lax.cond(
        zap_condition,
        safe_zap_letter,
        lambda args: args,
        (
            new_letters_alive,
            state.letter_explosion_frame,
            state.letter_explosion_timer,
            state.letter_explosion_frame_timer,
            state.letter_explosion_pos
        )
    )

    # Custom explosion animation sequence (independent of PLAYER_ZAPPER_COOLDOWN_TIME)
    explosion_sequence = jnp.array([
        [0, 1],  # 1.npy - 1 frame
        [1, 2],  # 2.npy - 2 frames
        [0, 2],  # 1.npy - 2 frames
        [1, 1],  # 2.npy - 1 frame
        [2, 1],  # 3.npy - 1 frame
        [1, 2],  # 2.npy - 2 frames
        [2, 4],  # 3.npy - 4 frames
        [3, 2],  # 4.npy - 2 frames
        [2, 1],  # 3.npy - 1 frame
    ])
    max_seq_len = explosion_sequence.shape[0]

    # Advance explosion frames
    frame_should_advance = (letter_explosion_frame_timer == 0) & (letter_explosion_frame > 0)
    next_frame = letter_explosion_frame + jnp.where(frame_should_advance, 1, 0)
    next_frame = jnp.where(next_frame > max_seq_len, 0, next_frame)

    def get_frame_timer(frame, should_advance, prev_timer):
        idx = jnp.clip(frame - 1, 0, max_seq_len - 1)
        duration = explosion_sequence[idx, 1]
        return jnp.where(should_advance, duration, prev_timer - 1)

    new_timer = jnp.where(
        next_frame > 0,
        jax.vmap(get_frame_timer)(next_frame, frame_should_advance, letter_explosion_frame_timer),
        0
    )
    new_timer = jnp.where(next_frame == 0, 0, new_timer)

    letter_explosion_frame = next_frame
    letter_explosion_frame_timer = new_timer
    letter_explosion_timer = jnp.where(letter_explosion_frame == 0, 0, letter_explosion_timer)

    return new_letters_x, new_letters_alive, letter_explosion_frame, letter_explosion_timer, letter_explosion_frame_timer, letter_explosion_pos


@jax.jit
def player_missile_step(
    state: WordZapperState, action: chex.Array, consts: WordZapperConstants
) -> chex.Array:
    fire = jnp.any(
        jnp.array(
            [
                action == Action.UPRIGHTFIRE,
                action == Action.UPLEFTFIRE,
                action == Action.DOWNRIGHTFIRE,
                action == Action.DOWNLEFTFIRE,
                action == Action.RIGHTFIRE,
                action == Action.LEFTFIRE,
            ]
        )
    )

    # if player fired and there is no active missile, create on in player_direction
    new_missile = jnp.where(
        jnp.logical_and(fire, jnp.logical_not(state.player_missile_position[2])),
        jnp.where(
            state.player_direction == -1,
            jnp.array([
                state.player_x - consts.MISSILE_SIZE[0], # x, y, active, direction
                state.player_y + consts.PLAYER_SIZE[1] / 2,
                1,
                -1
            ]),
            jnp.array([
                state.player_x + consts.PLAYER_SIZE[0],
                state.player_y + consts.PLAYER_SIZE[1] / 2,
                1,
                1
            ]),
        ),
        state.player_missile_position,
    )
    
    # if a missile is in frame and exists, we move the missile further in the specified direction (5 per tick), also always put the missile at the current player y position
    new_missile = jnp.where(
        state.player_missile_position[2],
        jnp.array([
            new_missile[0] + new_missile[3] * 3, # missile speed
            new_missile[1],
            new_missile[2],
            new_missile[3]
        ]),
        new_missile,
    )

    # check if the new positions are still in bounds
    new_missile = jnp.where(
        new_missile[0] < consts.X_BOUNDS[0] - 2,
        jnp.array([0, 0, 0, 0]),
        jnp.where(
            new_missile[0] > consts.X_BOUNDS[1] + consts.MISSILE_SIZE[0] + 2,
            jnp.array([0, 0, 0, 0]),
            new_missile
        ),
    )

    return new_missile


@jax.jit
def player_zapper_step(
    state: WordZapperState, action: chex.Array, consts: WordZapperConstants
) -> chex.Array:
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

    new_zapper = jnp.where(
        state.player_zapper_position[2], # active zapper exists
        jnp.array([
            state.player_x + consts.PLAYER_SIZE[0] / 2 - 2,
            state.player_y,
            state.player_zapper_position[2],
            state.player_zapper_position[3] - 1, # cooldown/letter block speed
            state.player_zapper_position[4],
            state.player_zapper_position[5],
            state.player_zapper_position[6] - 1, # zapper block time speed
        ]),
        jnp.where(
            jnp.logical_and(fire, state.player_zapper_position[6] <= 0),
            jnp.array([
                state.player_x + consts.PLAYER_SIZE[0] / 2 - 2,
                state.player_y,
                1,
                consts.PLAYER_ZAPPER_COOLDOWN_TIME,
                state.step_counter,
                state.player_x + consts.PLAYER_SIZE[0] / 2 - 2,
                consts.ZAPPER_BLOCK_TIME
            ]),
            jnp.array([
                0, 0, 0, 0, 0, 0, state.player_zapper_position[6] - 1
            ])
        )
    )

    # if cooldown is 0, deactivate zapper
    new_zapper = jnp.where(
        new_zapper[3] <= 0,
        new_zapper.at[2].set(0),
        new_zapper
    )

    return new_zapper

def handle_missile_enemy_explosions(
    state: WordZapperState,
    positions,
    new_enemy_active,
    player_missile_position,
    consts: WordZapperConstants
):
    # Missile-Enemy Collision Logic 
    missile_pos = player_missile_position
    missile_active = missile_pos[2] != 0

    def missile_enemy_collision(missile, enemies, active):
        missile_x, missile_y = missile[0], missile[1]
        missile_w, missile_h = consts.MISSILE_SIZE
        enemy_x = enemies[:, 0]
        enemy_y = enemies[:, 1]
        enemy_w, enemy_h = 16, 16

        m_left = missile_x
        m_right = missile_x + missile_w
        m_top = missile_y
        m_bottom = missile_y + missile_h

        e_left = enemy_x
        e_right = enemy_x + enemy_w
        e_top = enemy_y
        e_bottom = enemy_y + enemy_h

        h_overlap = (m_left <= e_right) & (m_right >= e_left)
        v_overlap = (m_top <= e_bottom) & (m_bottom >= e_top)           
        collisions = h_overlap & v_overlap & (active == 1)
        return collisions

    missile_collisions = jax.lax.cond(
        missile_active & (missile_pos[2] != 0),
        lambda: missile_enemy_collision(missile_pos, positions[:, 0:2], new_enemy_active),
        lambda: jnp.zeros_like(new_enemy_active, dtype=bool)
    )

    # Explosion logic: start explosion for hit enemies
    new_enemy_explosion_frame = jnp.where(missile_collisions, 1, state.enemy_explosion_frame)
    new_enemy_explosion_timer = jnp.where(missile_collisions, consts.ENEMY_EXPLOSION_FRAMES, state.enemy_explosion_timer)
    new_enemy_explosion_frame_timer = jnp.where(missile_collisions, consts.ENEMY_EXPLOSION_FRAME_DURATION, state.enemy_explosion_frame_timer)

    # Prevent explosion animation from moving
    if not hasattr(state, 'enemy_explosion_pos'):
        enemy_explosion_pos = jnp.zeros((consts.MAX_ENEMIES, 2))
    else:
        enemy_explosion_pos = state.enemy_explosion_pos

    explosion_started = missile_collisions
    enemy_explosion_pos = jnp.where(
        explosion_started[:, None],
        positions[:, 0:2],
        enemy_explosion_pos
    )

    frame_should_advance = (new_enemy_explosion_frame_timer == 0) & (new_enemy_explosion_frame > 0)
    new_enemy_explosion_frame = jnp.where(
        frame_should_advance,
        new_enemy_explosion_frame + 1,
        new_enemy_explosion_frame
    )
    new_enemy_explosion_frame_timer = jnp.where(
        (new_enemy_explosion_frame > 0),
        jnp.where(frame_should_advance, consts.ENEMY_EXPLOSION_FRAME_DURATION, new_enemy_explosion_frame_timer - 1),
        0
    )
    new_enemy_explosion_frame = jnp.where(new_enemy_explosion_frame > consts.ENEMY_EXPLOSION_FRAMES, 0, new_enemy_explosion_frame)

    new_enemy_active = jnp.where(missile_collisions, 0, new_enemy_active)
    player_missile_position = jnp.where(
        jnp.any(missile_collisions),
        jnp.zeros_like(player_missile_position),
        player_missile_position
    )

    return (
        new_enemy_explosion_frame,
        new_enemy_explosion_timer,
        new_enemy_explosion_frame_timer,
        enemy_explosion_pos,
        new_enemy_active,
        player_missile_position,
    )

def handle_player_enemy_collisions(
    new_player_x,
    new_player_y,
    new_player_direction,
    positions,
    active,
    consts: WordZapperConstants
):
    # Player rectangle
    player_pos = jnp.array([new_player_x, new_player_y])
    player_size = jnp.array(consts.PLAYER_SIZE)

    # Enemy rectangles and actives
    enemy_pos = positions[:, 0:2]  # shape (MAX_ENEMIES, 2)
    enemy_size = jnp.array([16, 16])
    enemy_active = active

    # Calculate edges for player
    p_left = player_pos[0] + player_size[0]/2
    p_right = player_pos[0] + player_size[0]
    p_top = player_pos[1]
    p_bottom = player_pos[1] + player_size[1]

    # Calculate edges for all enemies
    e_left = enemy_pos[:, 0]
    e_right = enemy_pos[:, 0] + enemy_size[0]
    e_top = enemy_pos[:, 1]
    e_bottom = enemy_pos[:, 1] + enemy_size[1]

    # Check overlap for all enemies (trigger on edge contact)
    horizontal_overlaps = (p_left <= e_right) & (p_right >= e_left)
    vertical_overlaps = (p_top <= e_bottom) & (p_bottom >= e_top)           
    collisions = horizontal_overlaps & vertical_overlaps & (enemy_active == 1)

    # If any collision, move player by 13 in direction of enemy (positions[:,3])
    any_collision = jnp.any(collisions)

    # Find the first colliding enemy (lowest index)
    colliding_idx = jnp.argmax(collisions)

    # Only use the direction if there is a collision
    enemy_dir = jnp.where(any_collision, positions[colliding_idx, 3], 0.0)

    # Move player by 13 in direction of enemy_dir
    new_player_x = jnp.where(any_collision & (enemy_dir < 0), new_player_x - 13, new_player_x)
    new_player_x = jnp.where(any_collision & (enemy_dir > 0), new_player_x + 13, new_player_x)

    # Deactivate ("disappear") collided enemy
    new_enemy_active = jnp.where(collisions, 0, active)

    return new_player_x, new_enemy_active

class JaxWordZapper(JaxEnvironment[WordZapperState, WordZapperObservation, WordZapperInfo, WordZapperConstants]) :
    def __init__(self, consts: WordZapperConstants = None, frameskip: int = 1, reward_funcs: list[callable] = None):
        super().__init__(consts)
        self.consts = consts or WordZapperConstants()

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
        self.frameskip = frameskip
        self.frame_stack_size = 4  # Standard 
        self.obs_size = 6 + 5 + 6 + 27 * 6 + 1 + 6 +  self.consts.MAX_ENEMIES * 5 + 1 + 1 + 1 + 1 
        self.renderer = WordZapperRenderer(self.consts)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: WordZapperState) -> jnp.ndarray:
        """Render the game state to a raster image."""
        return self.renderer.render(state)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jax.random.PRNGKey = jax.random.PRNGKey(42)) -> Tuple[WordZapperObservation, WordZapperState]:
        key, next_key = jax.random.split(key, 2)
        level_len_init = jnp.array(self.consts.LEVEL_WORD_LENGTHS[0], dtype=jnp.int32)
        encoded, next_key = choose_target_word(next_key, level_len_init)

        letters_x = jnp.linspace(self.consts.LETTER_VISIBLE_MIN_X, self.consts.LETTERS_END, 27)
        letters_y = jnp.full((27,), 30)

        reset_state = WordZapperState(
            player_x=jnp.array(self.consts.PLAYER_START_X),
            player_y=jnp.array(self.consts.PLAYER_START_Y),
            player_direction=jnp.array(0),

            enemy_positions=jnp.zeros((self.consts.MAX_ENEMIES, 5), dtype=jnp.float32),
            enemy_active=jnp.zeros((self.consts.MAX_ENEMIES,), jnp.int32),
            enemy_global_spawn_timer=jnp.array(60),

            player_missile_position=jnp.zeros(4),
            player_zapper_position=jnp.zeros(7),

            letters_x=letters_x,
            letters_y=letters_y,
            letters_char=jnp.arange(27),
            letters_alive=jnp.stack([jnp.ones((27,), dtype=jnp.int32), jnp.zeros((27,), dtype=jnp.int32)], axis=1),
            letters_speed=jnp.ones((27,)) * self.consts.LETTER_SCROLLING_SPEED,
            letters_positions=jnp.stack([letters_x, letters_y], axis=1),
            current_letter_index=jnp.array(0),

            level_word_len=level_len_init,
            waiting_for_special=jnp.array(0, dtype=jnp.int32),

            timer=jnp.array(self.consts.TIME),
            target_word=encoded,
            step_counter=jnp.array(0),

            game_phase=jnp.array(0),
            phase_timer=jnp.array(0),

            enemy_explosion_frame=jnp.zeros((self.consts.MAX_ENEMIES,), dtype=jnp.int32),
            enemy_explosion_timer=jnp.zeros((self.consts.MAX_ENEMIES,), dtype=jnp.int32),
            enemy_explosion_frame_timer=jnp.zeros((self.consts.MAX_ENEMIES,), dtype=jnp.int32),
            enemy_explosion_pos=jnp.zeros((self.consts.MAX_ENEMIES, 2)),

            letter_explosion_frame=jnp.zeros((27,), dtype=jnp.int32),
            letter_explosion_timer=jnp.zeros((27,), dtype=jnp.int32),
            letter_explosion_frame_timer=jnp.zeros((27,), dtype=jnp.int32),
            letter_explosion_pos=jnp.zeros((27, 2)),

            rng_key=next_key,
            score=jnp.array(0),
            finised_level_count=jnp.array(0),
        )

        initial_obs = self._get_observation(reset_state)
        return initial_obs, reset_state

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))

    def observation_space(self) -> spaces.Dict:
        """
        Returns observation space for WordZapper, contains:
        - player: PlayerEntity (x, y, direction, width, height, active)
        - player_missile: EntityPosition (x, y, width, height, active)
        - player_zapper: array of shape (6,) -> x, y, width, height, active, cooldown
        - letters: array of shape (27, 6) -> x, y, width, height, active, char_id
        - current_letter_index: int (0-5)
        - target_word: array of shape (6,) -> for letters of target word max 27
        - enemies: array of shape (MAX_ENEMIES, 5) -> x, y, width, height, active
        - score: int (0-999999)
        - game_phase: int (0-2)
        - level_word_len: int (4-6)
        - waiting_for_special: int (0-1)
        """
        return spaces.Dict({
            "player": spaces.Dict({
                "x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                "direction": spaces.Box(low=-1, high=1, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                "active": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
            }),
            "player_missile": spaces.Dict({
                "x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                "active": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
            }),
            "player_zapper": spaces.Box(low=0, high=255, shape=(6,), dtype=jnp.int32),
            "letters": spaces.Box(low=0, high=self.consts.LETTER_RESET_X+27*(self.consts.LETTER_SIZE[0]+self.consts.LETTERS_DISTANCE)+10, shape=(27, 6), dtype=jnp.int32),
            "current_letter_index": spaces.Box(low=0, high=5, shape=(), dtype=jnp.int32),
            "target_word": spaces.Box(low=-1, high=27, shape=(6,), dtype=jnp.int32),
            "enemies": spaces.Box(low=0, high=255, shape=(self.consts.MAX_ENEMIES, 5), dtype=jnp.int32),
            "score": spaces.Box(low=0, high=999999, shape=(), dtype=jnp.int32),
            "game_phase": spaces.Box(low=0, high=2, shape=(), dtype=jnp.int32),
            "level_word_len": spaces.Box(low=4, high=6, shape=(), dtype=jnp.int32),
            "waiting_for_special": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
        })


    def image_space(self) -> spaces.Box:
        """Returns the image space for WordZapper.
        The image is a RGB image with shape (210, 160, 3).
        """
        return spaces.Box(
            low=0,
            high=255,
            shape=(210, 160, 3),
            dtype=jnp.uint8
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: WordZapperState) -> WordZapperObservation:
        # Player entity (always active)
        player = PlayerEntity(
            x=state.player_x,
            y=state.player_y,
            direction=state.player_direction,
            width=jnp.array(self.consts.PLAYER_SIZE[0]),
            height=jnp.array(self.consts.PLAYER_SIZE[1]),
            active=jnp.array(1),
        )

        # Player missile: shape (4,) -> x, y, active, direction
        missile_pos = state.player_missile_position
        player_missile = EntityPosition(
            x=missile_pos[0],
            y=missile_pos[1],
            width=jnp.array(self.consts.MISSILE_SIZE[0]),
            height=jnp.array(self.consts.MISSILE_SIZE[1]),
            active=missile_pos[2],
        )

        # Player zapper: shape (7,) -> x, y, active, cooldown, pulse, initial_x, block_zapper
        zapper_pos = state.player_zapper_position
        # Keep only x, y, width, height, active, cooldown (6 fields)
        player_zapper = jnp.array([
            zapper_pos[0],  # x
            zapper_pos[1],  # y
            self.consts.ZAPPER_SIZE[0],
            self.consts.ZAPPER_SIZE[1],
            zapper_pos[2],  # active
            zapper_pos[3],  # cooldown
        ])

        # Letters: shape (27, 6) -> x, y, width, height, active, char_id
        def convert_letter(x, y, alive, char_id):
            return jnp.array([
                x,
                y,
                self.consts.LETTER_SIZE[0],
                self.consts.LETTER_SIZE[1],
                alive,
                char_id,
            ])

        letters = jax.vmap(convert_letter)(
            state.letters_x,
            state.letters_y,
            state.letters_alive[:, 0],  # first channel: alive flag
            state.letters_char,
        )

        # Enemies: shape (MAX_ENEMIES, 5) -> x, y, width, height, active
        def convert_enemy(pos, active):
            return jnp.where(
                pos[2] == 1, # if zonker
                jnp.array([
                pos[0],  # x
                pos[1],  # y
                self.consts.ZONKER_SIZE[0],
                self.consts.ZONKER_SIZE[1],
                active,
                ]),
                jnp.array([
                    pos[0],  # x
                    pos[1],  # y
                    self.consts.BONKER_SIZE[0],
                    self.consts.BONKER_SIZE[1],
                    active,
                ])
            )

        enemies = jax.vmap(convert_enemy)(state.enemy_positions, state.enemy_active)

        # Return observation
        return WordZapperObservation(
            player=player,
            player_missile=player_missile,
            player_zapper=player_zapper,
            letters=letters,
            current_letter_index=state.current_letter_index,
            target_word=state.target_word,
            enemies=enemies,
            score=state.score,
            game_phase=state.game_phase,
            level_word_len=state.level_word_len,
            waiting_for_special=state.waiting_for_special,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: WordZapperState) -> bool:
        """Check if the game should end due to timer expiring."""
        return jnp.logical_or(state.timer == 0, state.finised_level_count == 3)
    
    def flatten_entity_position(self, entity: EntityPosition) -> jnp.ndarray:
        return jnp.concatenate([
            jnp.array([entity.x], dtype=jnp.int32),
            jnp.array([entity.y], dtype=jnp.int32),
            jnp.array([entity.width], dtype=jnp.int32),
            jnp.array([entity.height], dtype=jnp.int32),
            jnp.array([entity.active], dtype=jnp.int32)
        ])

    def flatten_player_entity(self, entity: PlayerEntity) -> jnp.ndarray:
        return jnp.concatenate([
            jnp.array([entity.x], dtype=jnp.int32),
            jnp.array([entity.y], dtype=jnp.int32),
            jnp.array([entity.direction], dtype=jnp.int32),
            jnp.array([entity.width], dtype=jnp.int32),
            jnp.array([entity.height], dtype=jnp.int32),
            jnp.array([entity.active], dtype=jnp.int32)
        ])

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: WordZapperObservation) -> jnp.ndarray:
        return jnp.concatenate([
            self.flatten_player_entity(obs.player),
            self.flatten_entity_position(obs.player_missile),
            obs.player_zapper.flatten().astype(jnp.int32),
            obs.letters.flatten().astype(jnp.int32),
            obs.current_letter_index.flatten().astype(jnp.int32),
            obs.target_word.flatten().astype(jnp.int32),
            obs.enemies.flatten().astype(jnp.int32),
            obs.score.flatten().astype(jnp.int32),
            obs.game_phase.flatten().astype(jnp.int32),
            obs.level_word_len.flatten().astype(jnp.int32),
            obs.waiting_for_special.flatten().astype(jnp.int32),
        ])

    @partial(jax.jit, static_argnums=(0,))
    def _get_env_reward(self, previous_state: WordZapperState, state: WordZapperState):
        # Reward is score difference (if score field exists)
        return state.score - previous_state.score

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: WordZapperState, state: WordZapperState):
        return state.score - previous_state.score
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_all_rewards(self, previous_state: WordZapperState, state: WordZapperState) -> jnp.ndarray:
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array([reward_func(previous_state, state) for reward_func in self.reward_funcs])
        return rewards

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: WordZapperState, all_rewards: jnp.ndarray = None) -> WordZapperInfo:
        return WordZapperInfo(
            score=state.score,
            step_counter=state.step_counter,
            timer=state.timer,
            finished_level_count=state.finised_level_count,
            all_rewards=all_rewards,
        )
    
    def _advance_phase(self, state: WordZapperState):
        phase, timer = state.game_phase, state.phase_timer + 1

        def stay():
            return phase, timer

        def to_game():
            return (
                jnp.array(1),
                jnp.array(0),
            )

        return jax.lax.switch(
            phase,
            [
                # 0: pause, show word -> gameplay after pause frames
                lambda: jax.lax.cond(timer >= self.consts.LEVEL_PAUSE_FRAMES, to_game, stay),
                # 1: gameplay stays
                stay,
                # 2: animation
            ],
        )


    @partial(jax.jit, static_argnums=(0,))
    def _game_start_step(self, state: WordZapperState, action: chex.Array):
        """
        time to show the word for the level, no player input, no enemies
        """
        new_step_counter = jnp.where(
            state.step_counter == 1023,
            jnp.array(0),
            state.step_counter + 1
        )

        # Advance/carry phase-related state (start -> play -> animation)
        phase, p_timer = self._advance_phase(state)


        state = state._replace(
            step_counter=new_step_counter,
            game_phase=phase,
            phase_timer=p_timer,
        )

        return state
    
    @partial(jax.jit, static_argnums=(0,))
    def next_level(self, state: WordZapperState):
        """
        modify state for next level
        """
        next_lvl_word_len = state.level_word_len + 1

        next_target_word, rng_rest = choose_target_word(state.rng_key, next_lvl_word_len)

        letters_x = jnp.linspace(self.consts.LETTER_VISIBLE_MIN_X, self.consts.LETTERS_END, 27)
        letters_y = jnp.full((27,), 30)

        # reset state for new level
        return state._replace(
            player_x=jnp.array(self.consts.PLAYER_START_X),
            player_y=jnp.array(self.consts.PLAYER_START_Y),
            player_direction=jnp.array(0),

            enemy_positions=jnp.zeros((self.consts.MAX_ENEMIES, 5), dtype=jnp.float32),
            enemy_active=jnp.zeros((self.consts.MAX_ENEMIES,), jnp.int32),

            player_missile_position=jnp.zeros(4),
            player_zapper_position=jnp.zeros(7),

            letters_x=letters_x,
            letters_y=letters_y,
            letters_char=jnp.arange(27),
            letters_alive=jnp.stack([jnp.ones((27,), dtype=jnp.int32), jnp.zeros((27,), dtype=jnp.int32)], axis=1),
            letters_speed=jnp.ones((27,)) * self.consts.LETTER_SCROLLING_SPEED,
            letters_positions=jnp.stack([letters_x, letters_y], axis=1),
            current_letter_index=jnp.array(0),

            level_word_len=next_lvl_word_len,
            waiting_for_special=jnp.array(0, dtype=jnp.int32),
            target_word=next_target_word,

            game_phase=jnp.array(0, dtype=jnp.int32),
            phase_timer=jnp.array(0, dtype=jnp.int32),

            enemy_explosion_frame=jnp.zeros((self.consts.MAX_ENEMIES,), dtype=jnp.int32),
            enemy_explosion_timer=jnp.zeros((self.consts.MAX_ENEMIES,), dtype=jnp.int32),
            enemy_explosion_frame_timer=jnp.zeros((self.consts.MAX_ENEMIES,), dtype=jnp.int32),
            enemy_explosion_pos=jnp.zeros((self.consts.MAX_ENEMIES, 2)),

            letter_explosion_frame=jnp.zeros((27,), dtype=jnp.int32),
            letter_explosion_timer=jnp.zeros((27,), dtype=jnp.int32),
            letter_explosion_frame_timer=jnp.zeros((27,), dtype=jnp.int32),
            letter_explosion_pos=jnp.zeros((27, 2)),

            rng_key=rng_rest,
        )


    @partial(jax.jit, static_argnums=(0,))
    def _normal_game_step(self, state: WordZapperState, action: chex.Array):
        # player missile and zapper
        player_missile_position = player_missile_step(
            state, action, self.consts
        )

        player_zapper_position = player_zapper_step(
            state, action, self.consts
        )

        # player movement
        new_player_x, new_player_y, new_player_direction = player_step(
            state, action, self.consts
        )

        new_step_counter = jnp.where(
            state.step_counter == 1023,
            jnp.array(0),
            state.step_counter + 1
        )

        new_timer = jnp.where(
            (new_step_counter % 60 == 0) & (state.timer > 0),
            state.timer - 1,
            state.timer,
        )

        new_enemy_positions = state.enemy_positions.at[:, 0].add(
            state.enemy_positions[:, 3]        
        )


        new_enemy_active = jnp.where(
            (new_enemy_positions[:, 0] < self.consts.ENEMY_MIN_X - 16) |
            (new_enemy_positions[:, 0] > self.consts.ENEMY_MAX_X + 16),
            0,
            state.enemy_active,
        )
        
        # Scroll letters and handle explosions
        (
            new_letters_x,
            new_letters_alive,
            new_letter_explosion_frame,
            new_letter_explosion_timer,
            new_letter_explosion_frame_timer,
            new_letter_explosion_pos,
        ) = scrolling_letters(
            state,
            self.consts
        )


        new_enemy_global_spawn_timer = jnp.maximum(state.enemy_global_spawn_timer - 1, 0)
        has_free_slot = jnp.any(new_enemy_active == 0)
        spawn_cond = (new_enemy_global_spawn_timer == 0) & has_free_slot

        def spawn_one_enemy_fn(rng_key_in, existing_pos, existing_act):
            rng_key_out, sk_dir, sk_lane, sk_type = jax.random.split(rng_key_in, 4)
            direction = jnp.where(jax.random.bernoulli(sk_dir),  1.0, -1.0)
            vx = direction * self.consts.ENEMY_GAME_SPEED
            x_pos = jnp.where(direction == 1.0, self.consts.ENEMY_MIN_X, self.consts.ENEMY_MAX_X)
            lanes = jnp.linspace(self.consts.ENEMY_Y_MIN, self.consts.ENEMY_Y_MAX, 4)
            def lane_is_free(lane_y):
                return jnp.all(jnp.logical_or((existing_act == 0), (jnp.abs(existing_pos[:, 1] - lane_y) > 1e-3)))
            lane_free_mask = jax.vmap(lane_is_free)(lanes)
            perm = jax.random.permutation(sk_lane, 4)
            def pick_lane(i, chosen):
                lane = perm[i]
                is_free = lane_free_mask[lane]
                return jnp.where((chosen == -1) & is_free, lane, chosen)
            lane_idx = jax.lax.fori_loop(0, 4, pick_lane, -1)
            final_y = jnp.where(lane_idx == -1, -9999, lanes[0])
            enemy_type = jax.random.randint(sk_type, (), 0, 2)
            new_enemy = jnp.where(lane_idx == -1,
                                  jnp.array([x_pos, final_y, enemy_type, vx, 0.0]),
                                  jnp.array([x_pos, lanes[lane_idx], enemy_type, vx, 1.0]))
            return new_enemy, rng_key_out

        def spawn_enemy_branch(carry):
            pos, act, g_timer, rng_key_inner = carry
            free_idx = jnp.argmax(act == 0)
            new_enemy, rng_key_out = spawn_one_enemy_fn(rng_key_inner, pos, act)
            pos = pos.at[free_idx].set(new_enemy)
            act = act.at[free_idx].set(1)
            g_timer = jax.random.randint(rng_key_out, (), 30, 70)
            return pos, act, g_timer, rng_key_out

        def no_spawn_branch(carry):
            return carry

        positions, active, global_timer, rng_key = jax.lax.cond(
            spawn_cond,
            spawn_enemy_branch,
            no_spawn_branch,
            (new_enemy_positions, new_enemy_active, new_enemy_global_spawn_timer, state.rng_key),
        )

        # Integrated Player-Enemy Collision Logic
        new_player_x, new_enemy_active = handle_player_enemy_collisions(
            new_player_x,
            new_player_y,
            new_player_direction,
            positions,
            active,
            self.consts
        )

        # Missile-Enemy collision and explosion logic
        (
            new_enemy_explosion_frame,
            new_enemy_explosion_timer,
            new_enemy_explosion_frame_timer,
            enemy_explosion_pos,
            new_enemy_active,
            player_missile_position,
        ) = handle_missile_enemy_explosions(
            state,
            positions,
            new_enemy_active,
            player_missile_position,
            self.consts
        )

        target_word = state.target_word
        zapped_letters = (new_letter_explosion_frame == 1)
        allow_progress = (new_timer > 0)

        def update_letter_index(idx, zapped, chars, current_idx, word):
            is_correct = jax.lax.cond(
                current_idx < 6,
                lambda: (chars[idx] == word[current_idx]),
                lambda: jnp.array(False, dtype=jnp.bool_),
            )
            return jnp.where(
                zapped & is_correct & (current_idx < 6) & allow_progress,
                current_idx + 1,
                current_idx
            )


        new_current_letter_index = state.current_letter_index
        for i in range(27):
            new_current_letter_index = update_letter_index(
                i, zapped_letters[i], state.letters_char, new_current_letter_index, target_word
            )

        # Word length & special-gate logic
        word_len = jnp.sum(target_word >= 0).astype(jnp.int32)
        now_waiting_for_special = (new_current_letter_index >= word_len).astype(jnp.int32)

        special_idx = jnp.array(self.consts.SPECIAL_CHAR_INDEX, dtype=jnp.int32)
        special_was_zapped = jnp.any(
            zapped_letters & (state.letters_char == special_idx)
        ).astype(jnp.int32)

        level_cleared = ((now_waiting_for_special & special_was_zapped) & allow_progress).astype(jnp.int32)

        
        # Advance/carry phase-related state (start -> play -> animation)
        phase, p_timer = self._advance_phase(state)

        new_finised_level_count = jnp.where(
            level_cleared,
            state.finised_level_count + 1,
            state.finised_level_count
        )

        updated_state = state._replace(
            player_x=new_player_x,
            player_y=new_player_y,
            player_direction=new_player_direction,
            player_missile_position=player_missile_position,
            player_zapper_position=player_zapper_position,
            enemy_positions=positions,
            enemy_active=new_enemy_active,
            enemy_global_spawn_timer=global_timer,
            letters_x=new_letters_x,
            letters_alive=new_letters_alive,
            step_counter=new_step_counter,
            timer=new_timer,
            rng_key=rng_key,
            enemy_explosion_frame=new_enemy_explosion_frame,
            enemy_explosion_timer=new_enemy_explosion_timer,
            enemy_explosion_frame_timer=new_enemy_explosion_frame_timer,
            enemy_explosion_pos=enemy_explosion_pos,
            letter_explosion_frame=new_letter_explosion_frame,
            letter_explosion_timer=new_letter_explosion_timer,
            letter_explosion_frame_timer=new_letter_explosion_frame_timer,
            letter_explosion_pos=new_letter_explosion_pos,
            current_letter_index=new_current_letter_index,
            waiting_for_special=now_waiting_for_special,
            game_phase=phase,
            phase_timer=p_timer,
            finised_level_count=new_finised_level_count
        )

        # if level cleared and it is not final level move to next level
        out_state = jax.lax.cond(
            jnp.logical_and(level_cleared == 1, state.finised_level_count != 3),
            self.next_level,
            lambda s: s,
            updated_state
        )

        return out_state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: WordZapperState,
        action: chex.Array,
    ) -> Tuple[
            WordZapperObservation,
            WordZapperState,
            float,
            bool,
            WordZapperInfo,
        ]:
        previous_state = state

        state = jax.lax.switch(
            state.game_phase,
            [
                lambda s: self._game_start_step(s, action),
                lambda s: self._normal_game_step(s, action),
            ],
            state,
        )

        # Outputs
        observation = self._get_observation(state)
        done = self._get_done(state)
        env_reward = self._get_env_reward(previous_state, state)
        all_rewards = self._get_all_rewards(previous_state, state)
        info = self._get_info(state, all_rewards)

        return observation, state, env_reward, done, info


class WordZapperRenderer(JAXGameRenderer):
    def __init__(self, consts: WordZapperConstants = None):
        super().__init__()
        self.pl_sub_offsets_length = len(PL_OFFSETS)
        self.bonker_offsets_length = len(BONKER_OFFSETS)
        self.zonker_offsets_length = len(ZONKER_OFFSETS)
        self.yellow_letters_offsets_length = len(YELLOW_LETTERS_OFFSETS)
        self.letters_offsets_length = len(LETTERS_OFFSETS)
        self.consts = consts or WordZapperConstants()

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        raster = jr.create_initial_frame(width=160, height=210)
        
        # render background
        frame_bg = jr.get_sprite_frame(SPRITE_BG, 0)
        raster = jr.render_at(raster, 0, 0, frame_bg)

        def _draw_timer(raster):
            digits = jr.int_to_digits(state.timer.astype(jnp.int32), max_digits=2)
            raster = jr.render_label(raster, 70, 10, digits, DIGITS, spacing=10)
            return raster

        raster = _draw_timer(raster)

        def _draw_player_bundle(raster):
            # player sprite
            frame_pl = jr.get_sprite_frame(SPRITE_PL, state.step_counter)
            raster = jr.render_at(
                raster,
                state.player_x,
                state.player_y,
                frame_pl,
                flip_horizontal=state.player_direction == self.consts.FACE_LEFT,
            )

            # missile (if any)
            raster = jax.lax.cond(
                state.player_missile_position[2],
                lambda r: jr.render_at(
                    r,
                    state.player_missile_position[0],
                    state.player_missile_position[1],
                    SPRITE_PL_MISSILE,
                ),
                lambda r: r,
                raster,
            )

            # render player zapper
            zapper_spr = jnp.full(
                (self.consts.ZAPPER_SPR_HEIGHT, self.consts.ZAPPER_SPR_WIDTH, 4),
                jnp.asarray(self.consts.ZAPPER_COLOR, dtype=jnp.uint8),
                dtype=jnp.uint8
            )

            pulse_timer = jnp.where(
                state.player_zapper_position[4] > state.step_counter,
                state.step_counter + 1023,
                state.step_counter
            )

            raster = jax.lax.cond(
                jnp.logical_and(
                    state.player_zapper_position[2],
                    jnp.any(jnp.array([
                        (pulse_timer - state.player_zapper_position[4] == 1),
                        (pulse_timer - state.player_zapper_position[4] == 2), 
                        (pulse_timer - state.player_zapper_position[4] == 6),
                        (pulse_timer - state.player_zapper_position[4] == 10)
                    ]))
                ),
                lambda r : jr.render_at(
                    r,
                    state.player_zapper_position[0],
                    self.consts.MAX_ZAPPER_POS,
                    zapper_spr * (jnp.arange(self.consts.ZAPPER_SPR_HEIGHT) < state.player_zapper_position[1] - self.consts.MAX_ZAPPER_POS).astype(jnp.uint8)[:, None, None]
                ),
                lambda r : r,
                raster,
            )
            
            return raster

        raster = _draw_player_bundle(raster)


        # render enemies
        def _render_enemies(raster):
            frame_bonker = jr.get_sprite_frame(SPRITE_BONKER, state.step_counter // self.consts.ENEMY_ANIM_SWITCH_RATE)
            frame_zonker  = jr.get_sprite_frame(SPRITE_ZONKER,  state.step_counter // self.consts.ENEMY_ANIM_SWITCH_RATE)

            def body_fn(i, raster_inner):
                should_render_enemy = state.enemy_active[i]
                explosion_frame = state.enemy_explosion_frame[i]
                is_exploding = explosion_frame > 0
                x = jnp.where(is_exploding, state.enemy_explosion_pos[i, 0], state.enemy_positions[i, 0])
                y = jnp.where(is_exploding, state.enemy_explosion_pos[i, 1], state.enemy_positions[i, 1])
                enemy_type = state.enemy_positions[i, 2].astype(jnp.int32)

                def render_explosion(rr):
                    idx = jnp.clip(explosion_frame - 1, 0, 3)
                    return jr.render_at(rr, x, y, ENEMY_EXPLOSION_SPRITES[idx])

                raster_inner = jax.lax.cond(
                    explosion_frame > 0,
                    render_explosion,
                    lambda r: jax.lax.cond(
                        should_render_enemy,
                        lambda r: jax.lax.cond(
                            enemy_type == 0,
                            lambda r: jr.render_at(r, x, y, frame_bonker),
                            lambda r: jr.render_at(r, x, y, frame_zonker),
                            r
                        ),
                        lambda r: r,
                        r
                    ),
                    raster_inner
                )
                return raster_inner

            return jax.lax.fori_loop(0, self.consts.MAX_ENEMIES, body_fn, raster)

        raster = _render_enemies(raster)


        # Render normal letters and their explosions
        def _render_letter(i, raster):
            is_alive = state.letters_alive[i, 0]
            x = state.letters_x[i]
            y = state.letters_y[i]
            char_idx = state.letters_char[i]
            sprite = jr.get_sprite_frame(LETTERS, char_idx)  # (H, W, C)
            explosion_frame = state.letter_explosion_frame[i]
            explosion_pos = state.letter_explosion_pos[i]

            def render_visible(r):
                sprite_h, sprite_w = sprite.shape[:2]
                sprite_xs = x + jnp.arange(sprite_w)
                x_mask = jnp.logical_and((sprite_xs >= self.consts.LETTER_VISIBLE_MIN_X), (sprite_xs < self.consts.LETTER_VISIBLE_MAX_X)).astype(sprite.dtype)
                x_mask = x_mask[None, :, None]  # (1, W, 1)
                masked_sprite = sprite * x_mask
                return jr.render_at(r, x, y, masked_sprite)

            def render_explosion(r):
                # Use custom sequence for sprite index
                explosion_sequence = jnp.array([
                    0, # 1.npy
                    1, # 2.npy
                    0, # 1.npy
                    1, # 2.npy
                    2, # 3.npy
                    1, # 2.npy
                    2, # 3.npy
                    3, # 4.npy
                    2, # 3.npy
                ])
                seq_len = explosion_sequence.shape[0]
                idx = jnp.where(
                    (explosion_frame > 0) & (explosion_frame <= seq_len),
                    explosion_sequence[explosion_frame - 1],
                    0
                )
                sprite = LETTER_EXPLOSION_SPRITES[idx]
                sprite_h, sprite_w = sprite.shape[:2]
                sprite_xs = explosion_pos[0] + jnp.arange(sprite_w)
                x_mask = jnp.logical_and((sprite_xs >= self.consts.LETTER_VISIBLE_MIN_X), (sprite_xs < self.consts.LETTER_VISIBLE_MAX_X)).astype(sprite.dtype)
                x_mask = x_mask[None, :, None]  # (1, W, 1)
                masked_sprite = sprite * x_mask
                return jr.render_at(r, explosion_pos[0], explosion_pos[1], masked_sprite)

            # If explosion is active, render explosion
            raster = jax.lax.cond(explosion_frame > 0, render_explosion, lambda r: r, raster)
            # If letter is alive, render letter
            raster = jax.lax.cond(is_alive == 1, render_visible, lambda r: r, raster)
            return raster

        raster = jax.lax.fori_loop(0, state.letters_x.shape[0], _render_letter, raster)


        def _draw_word(raster, word_arr):
            def measure_letter_width(idx):
                spr = YELLOW_LETTERS[idx]
                cols = jnp.any(spr[..., 3] > 0, axis=0)
                return jnp.sum(cols).astype(jnp.int32)
    
            def measure_qmark_width():
                cols = jnp.any(QMARK_SPRITE[..., 3] > 0, axis=0)
                return jnp.sum(cols).astype(jnp.int32)
            
            def layout_params(word_arr, gap_px=10, baseline_shift=22):
                # Count real letters (>=0)
                n_letters = jnp.sum(word_arr >= 0)
                letter_idxs = jnp.arange(26, dtype=jnp.int32)
                letter_ws = jax.vmap(measure_letter_width)(letter_idxs)
                max_letter_w = jnp.max(letter_ws)
                q_w = measure_qmark_width()
                cell_w = jnp.maximum(max_letter_w, q_w)

                # Total width = n * cell + gaps
                total_w = n_letters * cell_w + jnp.maximum(n_letters - 1, 0) * gap_px
                start_x = (self.consts.WIDTH - total_w) // 2

                # Baseline Y
                sprite_h = YELLOW_LETTERS.shape[1]
                y_pos = self.consts.HEIGHT - sprite_h - baseline_shift

                return cell_w, start_x, y_pos, gap_px, n_letters
            # Draw full word with fixed cell spacing, centered
            cell_w, start_x, y_pos, gap_px, n_letters = layout_params(word_arr)
            def body_fn(i, carry):
                ras, x_base = carry
                idx = word_arr[i]
                def draw_letter(c):
                    r, xb = c
                    w = measure_letter_width(idx)
                    offset = (cell_w - w) // 2
                    r = jr.render_at(r, xb + offset, y_pos, YELLOW_LETTERS[idx])
                    return (r, xb + cell_w + gap_px)
                def skip(c):
                    r, xb = c
                    advance = jnp.where(i < n_letters, cell_w + gap_px, 0)
                    return (r, xb + advance)
                return jax.lax.cond(idx >= 0, draw_letter, skip, (ras, x_base))

            carry0 = (raster, start_x)
            ras_final, _ = jax.lax.fori_loop(0, word_arr.shape[0], body_fn, carry0)
            return ras_final

        def _draw_progress_word_fixed6(raster, word_arr, current_letter_index):
            # Always render 6 slots centered; revealed letters fill left->right
            GAP_PX = 10
            BASELINE_SHIFT = 22

            sprite_h = YELLOW_LETTERS.shape[1]
            y_pos = self.consts.HEIGHT - sprite_h - BASELINE_SHIFT
            letter_idxs = jnp.arange(26, dtype=jnp.int32)
            def _letter_w(i):
                spr = YELLOW_LETTERS[i]
                cols = jnp.any(spr[..., 3] > 0, axis=0)
                return jnp.sum(cols).astype(jnp.int32)

            all_w = jax.vmap(_letter_w)(letter_idxs)
            max_letter_w = jnp.max(all_w)

            q_cols = jnp.any(QMARK_SPRITE[..., 3] > 0, axis=0)
            q_w = jnp.sum(q_cols).astype(jnp.int32)

            CELL_W = jnp.maximum(max_letter_w, q_w)
            NUM_SLOTS = jnp.int32(6)

            total = NUM_SLOTS * CELL_W + GAP_PX * (NUM_SLOTS - 1)
            start = (self.consts.WIDTH - total) // 2

            word_len = jnp.sum(word_arr >= 0).astype(jnp.int32)

            def _letter_w_idx(idx):
                spr = YELLOW_LETTERS[idx]
                cols = jnp.any(spr[..., 3] > 0, axis=0)
                return jnp.sum(cols).astype(jnp.int32)

            carry0 = (raster, start)

            def body_fn(i, carry):
                ras, x = carry
                show_letter = (i < current_letter_index) & (i < word_len)

                def draw_letter(c):
                    r, xb = c
                    idx = word_arr[i]                
                    w = _letter_w_idx(idx)
                    offset = (CELL_W - w) // 2
                    r = jr.render_at(r, xb + offset, y_pos, YELLOW_LETTERS[idx])
                    return (r, xb + CELL_W + GAP_PX)

                def draw_q(c):
                    r, xb = c
                    offset = (CELL_W - q_w) // 2
                    r = jr.render_at(r, xb + offset, y_pos, QMARK_SPRITE)
                    return (r, xb + CELL_W + GAP_PX)

                return jax.lax.cond(show_letter, draw_letter, draw_q, (ras, x))

            ras_final, _ = jax.lax.fori_loop(0, 6, body_fn, carry0)
            return ras_final

        raster = jax.lax.switch(
            state.game_phase,
            [ 
                lambda ras: _draw_word(ras, state.target_word), 
                lambda ras: _draw_progress_word_fixed6(ras, state.target_word, state.current_letter_index),
            ],
            raster,
        )

        return raster
