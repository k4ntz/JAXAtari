import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.random as jrandom
from functools import partial
from typing import NamedTuple, Tuple, Optional

import chex
import jaxatari.spaces as spaces

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import JAXGameRenderer


#constants
WIDTH, HEIGHT = 160, 210  
ALPHABET_SIZE = 26
PAD_TOKEN = 26
L_MAX = 8

MAX_MISSES = 11
STEP_PENALTY = 0.0

#colours of environment
BG_COLOR    = jnp.array([150, 40, 30],   dtype=jnp.uint8)
BLUE_COLOR  = jnp.array([110, 170, 240], dtype=jnp.uint8)
GOLD_COLOR  = jnp.array([213, 130, 74],  dtype=jnp.uint8)
WHITE_COLOR = jnp.array([236, 236, 236], dtype=jnp.uint8)

#the gallows layoutt
GALLOWS_X = 18
GALLOWS_Y = 58
GALLOWS_THICK = 6
GALLOWS_TOP_LEN = 88
GALLOWS_POST_H = 42
ROPE_X = GALLOWS_X + GALLOWS_TOP_LEN - GALLOWS_THICK - 4
ROPE_W = 4
ROPE_H = 30
ROPE_TOP_Y = GALLOWS_Y - GALLOWS_THICK

#the underscores layout
UND_Y   = 190
UND_W   = 12
UND_H   = 6
UND_GAP = 6

#glyphs(chatgpt term) layout/ can also be called characters
GLYPH_ROWS = 7
GLYPH_COLS = 5
GLYPH_SCALE_SMALL   = 2   
GLYPH_SCALE_PREVIEW = 3   

# A..Z. '1' = filled pixel, '0' = empty
# each entry is 7 strings of length 5
# not sure if this is the best way to store the glyphs, but it works, need feedback
_FONT_5x7 = {
"A": ["01110","10001","10001","11111","10001","10001","10001"],
"B": ["11110","10001","11110","10001","10001","11110","00000"],
"C": ["01111","10000","10000","10000","10000","01111","00000"],
"D": ["11110","10001","10001","10001","10001","11110","00000"],
"E": ["11111","10000","11110","10000","10000","11111","00000"],
"F": ["11111","10000","11110","10000","10000","10000","00000"],
"G": ["01111","10000","10000","10111","10001","01111","00000"],
"H": ["10001","10001","11111","10001","10001","10001","00000"],
"I": ["01110","00100","00100","00100","00100","01110","00000"],
"J": ["00001","00001","00001","00001","10001","01110","00000"],
"K": ["10001","10010","11100","10010","10001","10001","00000"],
"L": ["10000","10000","10000","10000","10000","11111","00000"],
"M": ["10001","11011","10101","10001","10001","10001","00000"],
"N": ["10001","11001","10101","10011","10001","10001","00000"],
"O": ["01110","10001","10001","10001","10001","01110","00000"],
"P": ["11110","10001","10001","11110","10000","10000","00000"],
"Q": ["01110","10001","10001","10001","10101","01110","00001"],
"R": ["11110","10001","10001","11110","10010","10001","00000"],
"S": ["01111","10000","01110","00001","00001","11110","00000"],
"T": ["11111","00100","00100","00100","00100","00100","00000"],
"U": ["10001","10001","10001","10001","10001","01110","00000"],
"V": ["10001","10001","10001","10001","01010","00100","00000"],
"W": ["10001","10001","10001","10101","11011","10001","00000"],
"X": ["10001","01010","00100","00100","01010","10001","00000"],
"Y": ["10001","01010","00100","00100","00100","00100","00000"],
"Z": ["11111","00010","00100","01000","10000","11111","00000"],
}

# again, not sure if this is the best way to store the digits, but it works, feedback needed
_DIGITS_5x7 = {
    "0": ["01110","10001","10011","10101","11001","10001","01110"],
    "1": ["00100","01100","00100","00100","00100","00100","01110"],
    "2": ["01110","10001","00001","00010","00100","01000","11111"],
    "3": ["11110","00001","00001","01110","00001","00001","11110"],
    "4": ["00010","00110","01010","10010","11111","00010","00010"],
    "5": ["11111","10000","11110","00001","00001","10001","01110"],
    "6": ["00110","01000","10000","11110","10001","10001","01110"],
    "7": ["11111","00001","00010","00100","01000","01000","01000"],
    "8": ["01110","10001","10001","01110","10001","10001","01110"],
    "9": ["01110","10001","10001","01111","00001","00010","01100"],
}

DIGIT_GLYPHS = jnp.array(
    [[[int(c) for c in row] for row in _DIGITS_5x7[ch]]
     for ch in "0123456789"],
    dtype=jnp.uint8
)

#build a (26, 7, 5) uint8 array in A to Z order
GLYPHS = jnp.array(
    [[[int(c) for c in row] for row in _FONT_5x7[chr(ord('A') + i)]]
     for i in range(26)],
    dtype=jnp.uint8
)


GOLD_SQ = 10
GOLD_XL = 2
GOLD_XR = WIDTH - GOLD_SQ - 2
GOLD_Y  = 2

#body parts
HEAD_SIZE = 12
HEAD_X = ROPE_X + (ROPE_W // 2) - (HEAD_SIZE // 2)
HEAD_Y = ROPE_TOP_Y + ROPE_H

TORSO_W = 6
TORSO_H = 26
TORSO_X = HEAD_X + (HEAD_SIZE // 2) - (TORSO_W // 2)
TORSO_Y = HEAD_Y + HEAD_SIZE

ARM_H = 6
ARM_W = 18
ARM_Y = TORSO_Y + 4
ARM_L_X = TORSO_X - ARM_W
ARM_R_X = TORSO_X + TORSO_W

LEG_W = 6
LEG_H = 18
LEG_Y = TORSO_Y + TORSO_H
LEG_L_X = TORSO_X - 4
LEG_R_X = TORSO_X + TORSO_W - 2

#lives bar 
PIP_N   = MAX_MISSES        
PIP_W   = 6
PIP_H   = 8
PIP_GAP = 4
PIP_TOTAL = PIP_N * PIP_H + (PIP_N - 1) * PIP_GAP     

PIP_X   = WIDTH - PIP_W - 4                           
PIP_Y0  = HEIGHT - PIP_TOTAL - 10                 

RIGHT_MARGIN = 10
PREVIEW_W = GLYPH_COLS * GLYPH_SCALE_PREVIEW + 8
PREVIEW_H = GLYPH_ROWS * GLYPH_SCALE_PREVIEW + 8
PREVIEW_X = PIP_X - PREVIEW_W - 8
PREVIEW_Y = 120
DRAW_PREVIEW_BORDER = False
    

#scoreboard positions
SCORE_X = GOLD_XL + GOLD_SQ + 4   
SCORE_Y = GOLD_Y
ROUND_RIGHT_X = GOLD_XR - 2       
ROUND_Y       = GOLD_Y 
SCORE_SCALE = 2

# timer
TIMER_W = 40
TIMER_H = 4
TIMER_X = WIDTH//2 - TIMER_W//2
TIMER_Y = GOLD_Y + GOLD_SQ + 4


# list of words used in the game
# not sure if this is the best way to store the words, feedback needed
_WORDS = ["CAT", "TREE", "MOUSE", "ROBOT", "LASER", "JAX"]

def _encode_word(w: str) -> jnp.ndarray:
    arr = [ord(c) - 65 for c in w.upper()]
    arr = arr[:L_MAX]
    arr += [PAD_TOKEN] * (L_MAX - len(arr))
    return jnp.array(arr, dtype=jnp.int32)

WORDS_ENC = jnp.stack([_encode_word(w) for w in _WORDS], axis=0)
WORDS_LEN = jnp.array([min(len(w), L_MAX) for w in _WORDS], dtype=jnp.int32)
N_WORDS = WORDS_ENC.shape[0]

class HangmanState(NamedTuple):
    key: chex.Array
    word: chex.Array          
    length: chex.Array        
    mask: chex.Array          
    guessed: chex.Array       
    misses: chex.Array        
    lives: chex.Array         
    cursor_idx: chex.Array    
    done: chex.Array          
    reward: chex.Array        
    step_counter: chex.Array
    score: chex.Array   
    round_no: chex.Array
    time_left_steps: chex.Array
    cpu_score: chex.Array
    timer_max_steps: chex.Array      
    last_commit: chex.Array
  

class HangmanObservation(NamedTuple):
    revealed: chex.Array      
    mask: chex.Array          
    guessed: chex.Array       
    misses: chex.Array        
    lives: chex.Array         
    cursor_idx: chex.Array    

class HangmanInfo(NamedTuple):
    time: chex.Array
    all_rewards: chex.Array

# helpers funcutions
@jax.jit
def _sample_word(key: chex.Array) -> Tuple[chex.Array, chex.Array, chex.Array]:
    key, sub = jrandom.split(key)
    idx = jrandom.randint(sub, shape=(), minval=0, maxval=N_WORDS, dtype=jnp.int32)
    return key, WORDS_ENC[idx], WORDS_LEN[idx]

@jax.jit
def _compute_revealed(word: chex.Array, mask: chex.Array) -> chex.Array:
    return jnp.where(mask.astype(bool), word, PAD_TOKEN)

def _action_delta_cursor(action: chex.Array) -> chex.Array:
    up_like = jnp.logical_or(action == Action.UP, action == Action.UPFIRE)
    down_like = jnp.logical_or(action == Action.DOWN, action == Action.DOWNFIRE)
    return jnp.where(up_like, -1, jnp.where(down_like, 1, 0)).astype(jnp.int32)

def _action_commit(action: chex.Array) -> chex.Array:
    return jnp.logical_or(
        jnp.logical_or(action == Action.FIRE, action == Action.UPFIRE),
        action == Action.DOWNFIRE
    )

@jax.jit
def _advance_cursor_skip_guessed(cursor: chex.Array,
                                 delta: chex.Array,
                                 guessed: chex.Array) -> chex.Array:
    """Move one step in the delta direction, then keep stepping while guessed[cur]==1.
       Bounded by 26 steps to avoid infinite loops if all letters are guessed."""
    # First step (if delta==0, stay)
    step = jnp.where(delta > 0, 1, jnp.where(delta < 0, -1, 0)).astype(jnp.int32)
    cur0 = jnp.where(step == 0, cursor, (cursor + step) % ALPHABET_SIZE)

    def cond_fun(carry):
        cur, n = carry
        need_move = jnp.logical_and(step != 0, guessed[cur] == 1)
        return jnp.logical_and(n < ALPHABET_SIZE, need_move)

    def body_fun(carry):
        cur, n = carry
        return ((cur + step) % ALPHABET_SIZE, n + 1)

    cur, _ = lax.while_loop(cond_fun, body_fun, (cur0, jnp.int32(0)))
    return cur
    

def _draw_rect(r, x, y, w, h, color):
    tile = jnp.broadcast_to(color, (int(h), int(w), 3))
    return lax.dynamic_update_slice(r, tile, (jnp.int32(y), jnp.int32(x), jnp.int32(0)))

def _draw_outline(r, x, y, w, h, color, t=1):
    #top
    r = _draw_rect(r, x, y, w, t, color)                 
    #bottom
    r = _draw_rect(r, x, y + h - t, w, t, color)         
    
    #left 
    r = _draw_rect(r, x, y, t, h, color)                 
    
    #rifht
    r = _draw_rect(r, x + w - t, y, t, h, color)         
    return r

def _draw_if(cond, fn, r):
    return lax.cond(cond, lambda rr: fn(rr), lambda rr: rr, r)

def _draw_glyph_bitmap(r, x, y, bitmap, scale, color):
    """bitmap: (7,5) uint8 0/1. Scales each pixel to (scale x scale)."""
    def row_loop(i, rr):
        def col_loop(j, r2):
            on = bitmap[i, j] == 1
            return lax.cond(
                on,
                lambda r3: _draw_rect(r3, x + j*scale, y + i*scale, scale, scale, color),
                lambda r3: r3,
                r2
            )
        return lax.fori_loop(0, GLYPH_COLS, col_loop, rr)
    return lax.fori_loop(0, GLYPH_ROWS, row_loop, r)

def _draw_glyph_idx(r, x, y, idx, scale, color):
    bitmap = GLYPHS[jnp.clip(idx, 0, 25)]
    return _draw_glyph_bitmap(r, x, y, bitmap, scale, color)


def _draw_digit_idx(r, x, y, idx, scale, color):
    bitmap = DIGIT_GLYPHS[jnp.clip(idx, 0, 9)]
    return _draw_glyph_bitmap(r, x, y, bitmap, scale, color)

def _draw_number_left(r, x, y, val, scale, color, max_digits=4):
    w = GLYPH_COLS * scale
    gap = 2
    a = jnp.maximum(val, 0)

    # least to most
    d0 = a % 10
    d1 = (a // 10)   % 10
    d2 = (a // 100)  % 10
    d3 = (a // 1000) % 10

    digits = jnp.array([d3, d2, d1, d0], dtype=jnp.int32)

    n = jnp.maximum(1, jnp.where(a >= 1000, 4, jnp.where(a >= 100, 3, jnp.where(a >= 10, 2, 1))))
    start = 4 - n  

    def body(i, rr):
        xi = x + i * (w + gap)
        idx = jnp.clip(start + i, 0, 3)
        return lax.cond(i < n,
                        lambda r2: _draw_digit_idx(r2, xi, y, digits[idx], scale, color),
                        lambda r2: r2,
                        rr)
    return lax.fori_loop(0, max_digits, body, r)

def _draw_number_right(r, right_x, y, val, scale, color, max_digits=4):
    w = GLYPH_COLS * scale
    gap = 2
    a = jnp.maximum(val, 0)

    d3 = (a // 1000) % 10
    d2 = (a // 100)  % 10
    d1 = (a // 10)   % 10
    d0 = a % 10

    n = jnp.maximum(1, jnp.where(a >= 1000, 4, jnp.where(a >= 100, 3, jnp.where(a >= 10, 2, 1))))
    
    # most to least
    digits = jnp.array([d3, d2, d1, d0], dtype=jnp.int32)  
    
    #right-aligned
    mask = jnp.arange(4) >= (4 - n)

    def body(i, rr):
        i_right = 3 - i
        xi = right_x - (i + 1) * w - i * gap
        return lax.cond(mask[i_right],
                        lambda r2: _draw_digit_idx(r2, xi, y, digits[i_right], scale, color),
                        lambda r2: r2,
                        rr)
    return lax.fori_loop(0, max_digits, body, r)

# render
class HangmanRenderer(JAXGameRenderer):
    def __init__(self):
        pass

    # @partial(jax.jit, static_argnums=(0,))
    def render(self, state) -> jnp.ndarray:
        
        raster = jnp.broadcast_to(BG_COLOR, (HEIGHT, WIDTH, 3))

        # gold accents
        raster = _draw_rect(raster, GOLD_XL, GOLD_Y, GOLD_SQ, GOLD_SQ, GOLD_COLOR)
        raster = _draw_rect(raster, GOLD_XR, GOLD_Y, GOLD_SQ, GOLD_SQ, GOLD_COLOR)

        # gallows + rope
        raster = _draw_rect(raster, GALLOWS_X, GALLOWS_Y, GALLOWS_THICK, GALLOWS_POST_H, BLUE_COLOR)
        raster = _draw_rect(raster, GALLOWS_X, GALLOWS_Y - GALLOWS_THICK, GALLOWS_TOP_LEN, GALLOWS_THICK, BLUE_COLOR)
        raster = _draw_rect(raster, ROPE_X, ROPE_TOP_Y, ROPE_W, ROPE_H, BLUE_COLOR)

        #underscores (bottom-center
        length  = state.length
        total_w = length * UND_W + jnp.maximum(length - 1, 0) * UND_GAP
        start_x = (WIDTH - total_w) // 2

        def draw_underscore(i, r):
            x = start_x + i * (UND_W + UND_GAP)
            return lax.cond(
                i < length,
                lambda rr: _draw_rect(rr, x, UND_Y, UND_W, UND_H, BLUE_COLOR),
                lambda rr: rr,
                r
            )
        raster = lax.fori_loop(0, L_MAX, draw_underscore, raster)

        #revealed letters above underscores chars
        #not working 
        glyph_w = GLYPH_COLS * GLYPH_SCALE_SMALL 
        x_inset = (UND_W - glyph_w) // 2         
        def draw_revealed(i, r):
            x = start_x + i * (UND_W + UND_GAP) + x_inset
            cond = jnp.logical_and(i < length, state.mask[i] == 1)
            idx  = state.word[i]  
            return lax.cond(
                cond,
                lambda rr: _draw_glyph_idx(rr, x, UND_Y - GLYPH_ROWS*GLYPH_SCALE_SMALL - 4, idx, GLYPH_SCALE_SMALL, BLUE_COLOR),
                lambda rr: rr,
                r
            )
        raster = lax.fori_loop(0, L_MAX, draw_revealed, raster)

        #preview letter on the right
        if DRAW_PREVIEW_BORDER:
            raster = _draw_outline(raster, PREVIEW_X, PREVIEW_Y, PREVIEW_W, PREVIEW_H, WHITE_COLOR, t=1)
        px = PREVIEW_X + (PREVIEW_W - GLYPH_COLS*GLYPH_SCALE_PREVIEW)//2
        py = PREVIEW_Y + (PREVIEW_H - GLYPH_ROWS*GLYPH_SCALE_PREVIEW)//2
        raster = _draw_glyph_idx(raster, px, py, state.cursor_idx, GLYPH_SCALE_PREVIEW, BLUE_COLOR)
        
        #lives bar
        lives_clamped = jnp.clip(state.lives, 0, PIP_N)
        def draw_pip(i, r):
            y = PIP_Y0 + i * (PIP_H + PIP_GAP)
            return lax.cond(i < lives_clamped,
                            lambda rr: _draw_rect(rr, PIP_X, y, PIP_W, PIP_H, BLUE_COLOR),
                            lambda rr: rr,
                            r)
        raster = lax.fori_loop(0, PIP_N, draw_pip, raster)

        def _draw_timer(rr):
            denom = jnp.maximum(state.timer_max_steps, 1)
            frac  = jnp.clip(state.time_left_steps.astype(jnp.float32) / denom.astype(jnp.float32), 0.0, 1.0)
            fill  = jnp.int32(jnp.round(frac * TIMER_W))

            r1 = _draw_outline(rr, TIMER_X, TIMER_Y, TIMER_W, TIMER_H, WHITE_COLOR, t=1)

            #take background slice to preserve red where dont fill
            base = lax.dynamic_slice(
                r1,
                (jnp.int32(TIMER_Y + 1), jnp.int32(TIMER_X), jnp.int32(0)),
                (TIMER_H - 2, TIMER_W, 3)
            )
            full_tile = jnp.broadcast_to(BLUE_COLOR, (TIMER_H - 2, TIMER_W, 3))
            col_mask  = (jnp.arange(TIMER_W, dtype=jnp.int32) < fill)[jnp.newaxis, :, jnp.newaxis]
            tile      = jnp.where(col_mask, full_tile, base)

            return lax.dynamic_update_slice(
                r1, tile,
                (jnp.int32(TIMER_Y + 1), jnp.int32(TIMER_X), jnp.int32(0))
            )

        # only draw bar when A-mode is active
        raster = lax.cond(state.timer_max_steps > 0, _draw_timer, lambda rr: rr, raster)


        # progressive hangman body (misses)
        # commenting out the old hangman body parts because they were not working well
        # m = jnp.clip(state.misses, 0, 6)
        # raster = _draw_if(m >= 1, lambda rr: _draw_rect(rr, HEAD_X, HEAD_Y, HEAD_SIZE, HEAD_SIZE, BLUE_COLOR), raster)
        # raster = _draw_if(m >= 2, lambda rr: _draw_rect(rr, TORSO_X, TORSO_Y, TORSO_W, TORSO_H, BLUE_COLOR), raster)
        # raster = _draw_if(m >= 3, lambda rr: _draw_rect(rr, ARM_L_X, ARM_Y, ARM_W, ARM_H, BLUE_COLOR), raster)
        # raster = _draw_if(m >= 4, lambda rr: _draw_rect(rr, ARM_R_X, ARM_Y, ARM_W, ARM_H, BLUE_COLOR), raster)
        # raster = _draw_if(m >= 5, lambda rr: _draw_rect(rr, LEG_L_X, LEG_Y, LEG_W, LEG_H, BLUE_COLOR), raster)
        # raster = _draw_if(m >= 6, lambda rr: _draw_rect(rr, LEG_R_X, LEG_Y, LEG_W, LEG_H, BLUE_COLOR), raster)
        
        #progressive hangman body (11 parts)
        m = jnp.clip(state.misses, 0, 11)

        #head
        raster = _draw_if(m >= 1,  lambda rr: _draw_rect(rr, HEAD_X, HEAD_Y, HEAD_SIZE, HEAD_SIZE, BLUE_COLOR), raster)

        #split torso into two segments
        TORSO1_H = TORSO_H // 2
        TORSO2_H = TORSO_H - TORSO1_H

        #torso top
        raster = _draw_if(m >= 2,  lambda rr: _draw_rect(rr, TORSO_X, TORSO_Y, TORSO_W, TORSO1_H, BLUE_COLOR), raster)
        # 3) torso bottom
        raster = _draw_if(m >= 3,  lambda rr: _draw_rect(rr, TORSO_X, TORSO_Y + TORSO1_H, TORSO_W, TORSO2_H, BLUE_COLOR), raster)

        #arms split into upper + forearm
        ARM_UP_W  = 12
        ARM_LOW_W = ARM_W - ARM_UP_W

        #left upper arm
        raster = _draw_if(m >= 4,  lambda rr: _draw_rect(rr, TORSO_X - ARM_UP_W, ARM_Y, ARM_UP_W, ARM_H, BLUE_COLOR), raster)
        #left forearm
        raster = _draw_if(m >= 5,  lambda rr: _draw_rect(rr, TORSO_X - ARM_W,     ARM_Y, ARM_LOW_W, ARM_H, BLUE_COLOR), raster)

        #right upper arm
        raster = _draw_if(m >= 6,  lambda rr: _draw_rect(rr, TORSO_X + TORSO_W,            ARM_Y, ARM_UP_W,  ARM_H, BLUE_COLOR), raster)
        #right forearm
        raster = _draw_if(m >= 7,  lambda rr: _draw_rect(rr, TORSO_X + TORSO_W + ARM_UP_W, ARM_Y, ARM_LOW_W, ARM_H, BLUE_COLOR), raster)

        #legs split into thigh + shin
        LEG1_H = (LEG_H // 2) + 1
        LEG2_H = LEG_H - LEG1_H

        #left thigh
        raster = _draw_if(m >= 8,  lambda rr: _draw_rect(rr, LEG_L_X, LEG_Y, LEG_W, LEG1_H, BLUE_COLOR), raster)
        #left shin
        raster = _draw_if(m >= 9,  lambda rr: _draw_rect(rr, LEG_L_X, LEG_Y + LEG1_H, LEG_W, LEG2_H, BLUE_COLOR), raster)

        #right thigh
        raster = _draw_if(m >= 10, lambda rr: _draw_rect(rr, LEG_R_X, LEG_Y, LEG_W, LEG1_H, BLUE_COLOR), raster)
        
        raster = _draw_if(m >= 11, lambda rr: _draw_rect(rr, LEG_R_X, LEG_Y + LEG1_H, LEG_W, LEG2_H, BLUE_COLOR), raster)


        raster = _draw_number_left(
            raster, SCORE_X, SCORE_Y, jnp.asarray(state.score, jnp.int32), SCORE_SCALE, GOLD_COLOR
        )

        raster = _draw_number_right(
            raster, ROUND_RIGHT_X, ROUND_Y, jnp.asarray(state.cpu_score, jnp.int32), SCORE_SCALE, GOLD_COLOR
        )

        
        return raster
    
    
#environment

class JaxHangman(JaxEnvironment[HangmanState, HangmanObservation, HangmanInfo, Action]):
    def __init__(self, reward_funcs: Optional[list] = None, *,
                 max_misses: int = MAX_MISSES,
                 step_penalty: float = STEP_PENALTY,
                 difficulty_mode: str = "B",          
                 timer_seconds: int = 20,
                 steps_per_second: int = 30):
        super().__init__()
        self.renderer = HangmanRenderer()
        self.max_misses = int(max_misses)
        self.step_penalty = float(step_penalty)

        # difficulty / timer config 
        self.timed = 1 if str(difficulty_mode).upper() == "A" else 0
        self.timer_steps = int(timer_seconds * steps_per_second)

        self.action_set = [Action.NOOP, Action.FIRE, Action.UP, Action.DOWN, Action.UPFIRE, Action.DOWNFIRE]
        self.obs_size = L_MAX + L_MAX + ALPHABET_SIZE + 3
        self.reward_funcs = tuple(reward_funcs) if reward_funcs is not None else None


    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key=None) -> Tuple[HangmanObservation, HangmanState]:
        if key is None:
            key = jrandom.PRNGKey(0)
        key, word, length = _sample_word(key)

        # init round timer 
        time0 = jnp.array(self.timer_steps if self.timed == 1 else 0, dtype=jnp.int32)
        tmax  = jnp.array(self.timer_steps if self.timed == 1 else 0, dtype=jnp.int32)


        state = HangmanState(
            key=key,
            word=word,
            length=length,
            mask=jnp.zeros((L_MAX,), dtype=jnp.int32),
            guessed=jnp.zeros((ALPHABET_SIZE,), dtype=jnp.int32),
            misses=jnp.array(0, dtype=jnp.int32),
            lives=jnp.array(self.max_misses, dtype=jnp.int32),
            cursor_idx=jnp.array(0, dtype=jnp.int32),
            done=jnp.array(False),
            reward=jnp.array(0.0, dtype=jnp.float32),
            step_counter=jnp.array(0, dtype=jnp.int32),
            score=jnp.array(0, dtype=jnp.int32),
            round_no=jnp.array(1, dtype=jnp.int32),
            time_left_steps=time0,
            timer_max_steps=tmax,
            cpu_score=jnp.array(0, dtype=jnp.int32),
            last_commit=jnp.array(0, dtype=jnp.int32),
        )
        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: HangmanState, action: chex.Array) -> Tuple[HangmanObservation, HangmanState, float, bool, HangmanInfo]:
        commit = _action_commit(action).astype(jnp.int32)
        delta  = _action_delta_cursor(action)
        start_new_round = jnp.logical_and(state.done,
                          jnp.logical_and(commit == 1, state.last_commit == 0))



        def _new_round(s: HangmanState) -> HangmanState:
            key, word, length = _sample_word(s.key)
            time0 = jnp.array(self.timer_steps if self.timed == 1 else 0, dtype=jnp.int32)
            tmax  = jnp.array(self.timer_steps if self.timed == 1 else 0, dtype=jnp.int32)

            return HangmanState(
                key=key, word=word, length=length,
                mask=jnp.zeros((L_MAX,), dtype=jnp.int32),
                guessed=jnp.zeros((ALPHABET_SIZE,), dtype=jnp.int32),
                misses=jnp.array(0, dtype=jnp.int32),
                lives=jnp.array(self.max_misses, dtype=jnp.int32),
                cursor_idx=jnp.array(0, dtype=jnp.int32),
                done=jnp.array(False),
                reward=jnp.array(0.0, dtype=jnp.float32),
                step_counter=jnp.array(0, dtype=jnp.int32),
                score=s.score,
                round_no=s.round_no,
                time_left_steps=time0,
                timer_max_steps=tmax,
                cpu_score=s.cpu_score,
                last_commit=jnp.array(0, dtype=jnp.int32),
            )


        def _continue_round(s: HangmanState) -> HangmanState:
            #(skip guessed)
            cursor = _advance_cursor_skip_guessed(s.cursor_idx, delta, s.guessed)

            #tier tick every step
            t0 = s.time_left_steps
            t1 = jnp.where(self.timed == 1, jnp.maximum(t0 - 1, 0), t0)
            timed_out = jnp.logical_and(self.timed == 1, t1 == 0)

            idx = jnp.arange(L_MAX, dtype=jnp.int32)
            within = idx < s.length
            
            def on_commit(s2: HangmanState) -> HangmanState:
                already  = s2.guessed[cursor] == 1
                guessed  = s2.guessed.at[cursor].set(1)

                # reveal any matches at valid positions
                pos_hits = (s2.word == cursor).astype(jnp.int32) * within.astype(jnp.int32)
                any_hit  = jnp.any(pos_hits == 1)
                mask     = jnp.where(pos_hits.astype(bool), 1, s2.mask)

                # wrong guess
                wrong   = jnp.logical_and(jnp.logical_not(any_hit), jnp.logical_not(already))
                misses  = s2.misses + wrong.astype(jnp.int32)
                lives   = s2.lives  - wrong.astype(jnp.int32)

                #win
                n_revealed   = jnp.sum(jnp.where(within, mask, 0))           
                all_revealed = (n_revealed == s2.length)

                #loss check
                lost = misses >= self.max_misses

                #reveal
                mask_final = jnp.where(lost, jnp.where(within, 1, mask), mask)

                # reward
                step_reward = jnp.where(all_revealed, 1.0,
                                jnp.where(lost, -1.0, self.step_penalty)).astype(jnp.float32)

                #bump counters
                won          = all_revealed.astype(jnp.int32)
                lost_i32     = lost.astype(jnp.int32)
                round_ended  = jnp.logical_or(all_revealed, lost).astype(jnp.int32)

                new_score    = (s2.score + won).astype(jnp.int32)        
                cpu_new      = (s2.cpu_score + lost_i32).astype(jnp.int32)  
                new_roundno  = (s2.round_no + round_ended).astype(jnp.int32)

                new_time = jnp.array(self.timer_steps if self.timed == 1 else 0, dtype=jnp.int32)

                return HangmanState(
                    key=s2.key, word=s2.word, length=s2.length,
                    mask=mask_final, guessed=guessed, misses=misses, lives=lives,
                    cursor_idx=cursor,
                    done=(round_ended == 1),                 
                    reward=step_reward,
                    step_counter=s2.step_counter + 1,
                    score=new_score,                         
                    round_no=new_roundno,                    
                    time_left_steps=new_time,                
                    cpu_score=cpu_new,                       
                    timer_max_steps=s2.timer_max_steps,
                    last_commit=commit,
                )




            def no_commit(s2: HangmanState) -> HangmanState:
                #gate the timer with "round is active"
                active    = (self.timed == 1)
                t0        = s2.time_left_steps
                t1        = jnp.where(active, jnp.maximum(t0 - 1, 0), t0)
                timed_out = jnp.logical_and(active, t1 == 0)
                add_miss  = jnp.where(timed_out, 1, 0).astype(jnp.int32)

                misses = s2.misses + add_miss
                lives  = s2.lives  - add_miss
                lost   = misses >= self.max_misses

                idx = jnp.arange(L_MAX, dtype=jnp.int32)
                within = idx < s2.length
                mask_final = jnp.where(lost, jnp.where(within, 1, s2.mask), s2.mask)

                t_next = jnp.where(
                    timed_out,
                    jnp.array(self.timer_steps if self.timed == 1 else 0, dtype=jnp.int32),
                    t1
                )

                cpu_new     = s2.cpu_score + jnp.where(lost, 1, 0)
                round_ended = lost.astype(jnp.int32)
                new_roundno = s2.round_no + round_ended

                return HangmanState(
                    key=s2.key, word=s2.word, length=s2.length,
                    mask=mask_final, guessed=s2.guessed, misses=misses, lives=lives,
                    cursor_idx=cursor,
                    done=jnp.logical_or(s2.done, lost),
                    reward=jnp.where(lost, jnp.array(-1.0, dtype=jnp.float32),
                                    jnp.array(self.step_penalty, dtype=jnp.float32)),
                    step_counter=s2.step_counter + 1,
                    score=s2.score,
                    round_no=new_roundno,          
                    time_left_steps=t_next,
                    cpu_score=cpu_new,             
                    timer_max_steps=s2.timer_max_steps,
                    last_commit=commit,
                )



            return lax.cond(commit, on_commit, no_commit, s)

        def _freeze(_s):
            return HangmanState(
                key=_s.key, word=_s.word, length=_s.length,
                mask=_s.mask, guessed=_s.guessed, misses=_s.misses, lives=_s.lives,
                cursor_idx=_s.cursor_idx,
                done=_s.done,
                reward=jnp.array(0.0, dtype=jnp.float32),   
                step_counter=_s.step_counter,
                score=_s.score, round_no=_s.round_no,
                time_left_steps=_s.time_left_steps,
                cpu_score=_s.cpu_score,
                timer_max_steps=_s.timer_max_steps,
                last_commit=commit,                         
            )

        next_state = lax.cond(
            start_new_round,
            _new_round,
            lambda s: lax.cond(s.done, _freeze, _continue_round, s),
            state,
        )

        done = self._get_done(next_state)
        env_reward = self._get_env_reward(state, next_state)
        all_rewards = self._get_all_reward(state, next_state)
        obs = self._get_observation(next_state)
        info = self._get_info(next_state, all_rewards)
        return obs, next_state, env_reward, done, info

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(6)

    def observation_space(self) -> spaces:
        return spaces.Dict({
            "revealed": spaces.Box(low=0, high=PAD_TOKEN, shape=(L_MAX,), dtype=jnp.int32),
            "mask":     spaces.Box(low=0, high=1,          shape=(L_MAX,), dtype=jnp.int32),
            "guessed":  spaces.Box(low=0, high=1,          shape=(ALPHABET_SIZE,), dtype=jnp.int32),
            "misses":   spaces.Box(low=0, high=self.max_misses, shape=(), dtype=jnp.int32),
            "lives":    spaces.Box(low=0, high=self.max_misses, shape=(), dtype=jnp.int32),
            "cursor_idx": spaces.Box(low=0, high=ALPHABET_SIZE-1, shape=(), dtype=jnp.int32),
        })

    def image_space(self) -> spaces.Box:
        return spaces.Box(low=0, high=255, shape=(HEIGHT, WIDTH, 3), dtype=jnp.uint8)


    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: HangmanState) -> HangmanObservation:
        revealed = _compute_revealed(state.word, state.mask)
        return HangmanObservation(
            revealed=revealed, mask=state.mask, guessed=state.guessed,
            misses=state.misses, lives=state.lives, cursor_idx=state.cursor_idx,
        )

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: HangmanObservation) -> jnp.ndarray:
        return jnp.concatenate([
            obs.revealed.flatten(),
            obs.mask.flatten(),
            obs.guessed.flatten(),
            obs.misses.reshape((1,)).astype(jnp.int32),
            obs.lives.reshape((1,)).astype(jnp.int32),
            obs.cursor_idx.reshape((1,)).astype(jnp.int32),
        ])

    @partial(jax.jit, static_argnums=(0,))
    def _get_env_reward(self, previous_state: HangmanState, state: HangmanState):
        #return the reward 
        return state.reward

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(self, previous_state: HangmanState, state: HangmanState):
        if self.reward_funcs is None:
            return jnp.zeros(1, dtype=jnp.float32)
        rewards = jnp.array([rf(previous_state, state) for rf in self.reward_funcs], dtype=jnp.float32)
        return rewards

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: HangmanState) -> bool:
        return jnp.array(False)


    # @partial(jax.jit, static_argnums=(0,))
    def render(self, state: HangmanState) -> jnp.ndarray:
        return self.renderer.render(state)

    def _get_info(
        self,
        state: HangmanState,
        all_rewards: Optional[chex.Array] = None,
    ) -> HangmanInfo:
        if all_rewards is None:
            all_rewards = jnp.zeros(1, dtype=jnp.float32)
        return HangmanInfo(time=state.step_counter, all_rewards=all_rewards)
