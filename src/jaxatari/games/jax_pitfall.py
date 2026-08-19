import os
from functools import partial
import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax import lax
from typing import Tuple, List, Dict, Optional, Any

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action, ObjectObservation
import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils


SEED = 0xC4


# --- ShowDigits, transcribed from pitfall.asm --------------------------------
# DIGIT_H is 8 and each glyph is one eight-bit GRP pattern, so a digit occupies
# eight screen pixels either way. Like every other pattern table in this port
# the ROM lists the rows bottom first, because the kernel reads `(digitPtr),y`
# with y counting down from DIGIT_H-1 as the raster descends.
DIGIT_H = 8
DIGIT_W = 8
DIGIT_PATTERNS = (
    (   # Zero
        0b00111100, 0b01100110, 0b01100110, 0b01100110,
        0b01100110, 0b01100110, 0b01100110, 0b00111100,
    ),
    (   # One
        0b00111100, 0b00011000, 0b00011000, 0b00011000,
        0b00011000, 0b00011000, 0b00111000, 0b00011000,
    ),
    (   # Two
        0b01111110, 0b01100000, 0b01100000, 0b00111100,
        0b00000110, 0b00000110, 0b01000110, 0b00111100,
    ),
    (   # Three
        0b00111100, 0b01000110, 0b00000110, 0b00001100,
        0b00001100, 0b00000110, 0b01000110, 0b00111100,
    ),
    (   # Four
        0b00001100, 0b00001100, 0b00001100, 0b01111110,
        0b01001100, 0b00101100, 0b00011100, 0b00001100,
    ),
    (   # Five
        0b01111100, 0b01000110, 0b00000110, 0b00000110,
        0b01111100, 0b01100000, 0b01100000, 0b01111110,
    ),
    (   # Six
        0b00111100, 0b01100110, 0b01100110, 0b01100110,
        0b01111100, 0b01100000, 0b01100010, 0b00111100,
    ),
    (   # Seven
        0b00011000, 0b00011000, 0b00011000, 0b00011000,
        0b00001100, 0b00000110, 0b01000010, 0b01111110,
    ),
    (   # Eight
        0b00111100, 0b01100110, 0b01100110, 0b00111100,
        0b00111100, 0b01100110, 0b01100110, 0b00111100,
    ),
    (   # Nine
        0b00111100, 0b01000110, 0b00000110, 0b00111110,
        0b01100110, 0b01100110, 0b01100110, 0b00111100,
    ),
    (   # DoublePoint, the timer colon
        0b00000000, 0b00011000, 0b00011000, 0b00000000,
        0b00000000, 0b00011000, 0b00011000, 0b00000000,
    ),
    (   # Space
        0b00000000, 0b00000000, 0b00000000, 0b00000000,
        0b00000000, 0b00000000, 0b00000000, 0b00000000,
    ),
)
DIGIT_COLON = 10
DIGIT_SPACE = 11


def digit_pattern_bitmap(glyph_index: int) -> np.ndarray:
    """One ShowDigits glyph as a top-down DIGIT_H x DIGIT_W boolean bitmap."""
    rows = DIGIT_PATTERNS[glyph_index][::-1]
    return np.array(
        [[bool((row >> (DIGIT_W - 1 - col)) & 1) for col in range(DIGIT_W)] for row in rows],
        dtype=bool,
    )


_DIGIT_GLYPHS = jnp.asarray(
    np.stack([digit_pattern_bitmap(i) for i in range(len(DIGIT_PATTERNS))])
)

# ShowDigits sets NUSIZ0 and NUSIZ1 to THREE_COPIES and nudges GRP1 eight pixels
# with HMP1, so the six slots interleave into one row eight pixels apart. The
# first slot's column and both row numbers are read off the reference frame.
HUD_SLOT_X = (21, 29, 37, 45, 53, 61)
HUD_SCORE_ROW = 9
HUD_TIMER_ROW = 22
# `lda colorLst / sta COLUP0 / sta COLUP1`: a light grey, not white.
HUD_COLOR = (214, 214, 214)

# livesPat is a pattern byte rather than a count - `$a0 = 3, $80 = 2, $00 = 1` -
# and `ora temp3` folds it into slot 0 of the timer line on all eight rows, so
# it shows up as one or two full-height bars beside the clock.
LIVES_PAT = (0x00, 0x00, 0x80, 0xA0)

# Two TIA display artefacts. HMOVE's comb blacks the leftmost eight pixels of
# every scanline, and the frame ALE hands back opens with six scanlines still in
# vertical blank. Both hide pixels; neither moves a coordinate.
HMOVE_BLANK_COLS = 8
VBLANK_ROWS = 6


# The ROM runs at 60 Hz and this port steps at 30, so anything the ROM counts in
# frames advances twice per JAX step.
NTSC_FRAMES_PER_STEP = 2


# --- KilledHarry: the death and restart state machine ------------------------
# The fatal frame writes two bytes and nothing else:
#
#   KilledHarry SUBROUTINE
#       lda    #SOUND_DEAD      ; 2
#       sta    soundIdx         ; 3
#       lda    #$84             ; 2                 start copyright..
#       sta    noGameScroll     ; 3                 ..animation
#       jmp    ProcessObjects   ; 3
#
# Harry keeps whatever position he had, because `jmp ProcessObjects` skips the
# rest of .processHarry and `lda noGameScroll / beq .processHarry` keeps skipping
# it on every later frame. soundIdx then walks up SoundTab one note every fourth
# frame, and only when it reaches SOUND_FALLING-1 does the vertical-blank block
# take the life and put Harry back.
SOUND_DEAD = 0x31
SOUND_FALLING = 0x53
# `lda #SOUND_TREASURE / sta soundIdx` in `.incScore`: the treasure tune index.
SOUND_TREASURE = 0x25
KILLED_HARRY_SCROLL = 0x84

# SoundTab, verbatim. Only the sign bit is load-bearing here: `lda SoundTab-1,x
# / bpl .contSound / sty soundIdx` ends a tune on the first negative byte, and
# for the dead tune that byte is entry SOUND_FALLING-1, which is exactly the
# entry the life-loss test watches for.
SOUND_TAB = (
    0x13, 0x13, 0x13, 0x13, 0x13, 0x13, 0x13, 0x09,
    0x0B, 0x0B, 0x0B, 0x0B, 0x0B, 0x0B, 0x0B, 0x0B,
    0x0B, 0x0B, 0x0B, 0x0B, 0x09, 0x0B, 0x09, 0x0B,
    0x0B, 0x0B, 0x0B, 0x0B, 0x0B, 0x0B, 0x8B, 0x06,
    0x04, 0x03, 0x02, 0x84,
    0x13, 0x13, 0x0E, 0x0B, 0x09, 0x09, 0x09, 0x0B,
    0x09, 0x09, 0x09, 0x89,
    0x1D, 0x1D, 0x1D, 0x1D, 0x1D, 0x1D, 0x1D, 0x1D,
    0x1D, 0x1A, 0x1A, 0x19, 0x19, 0x19, 0x19, 0x19,
    0x19, 0x1D, 0x1D, 0x1D, 0x1D, 0x1D, 0x14, 0x15,
    0x14, 0x15, 0x14, 0x15, 0x14, 0x15, 0x14, 0x15,
    0x14, 0x95,
    0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x9F,
)
_SOUND_TAB = jnp.asarray(SOUND_TAB, dtype=jnp.int32)

# The life-loss block, reached from the vertical blank once the dead tune ends:
#
#   ldy    #$d0|NO_MOVE     ; 2                 upper Harry restart y-position
#   sty    oldJoystick      ; 3                 clear joystick
#   lda    #20              ; 2
#   sta    xPosHarry        ; 3
#   ldx    #JUMP_LEN        ; 2
#   lda    yPosHarry        ; 3
#   cmp    #71              ; 2                 Harry at underground?
#   bcc    LF5D2            ; 2³                 no, skip
#   ldy    #64              ; 2                  yes, lower Harry restart y-pos.
#
# That one `ldy` is used twice: as the cleared joystick and, on the upper branch,
# as yPosHarry itself. jumpIndex is JUMP_LEN, whose JumpTab entry is -1, so
# yPosHarry counts *up* one per frame - from 223 through 255, round to 0, and
# only from there does Harry drop into view and stop at JUNGLE_GROUND.
RESTART_X = 20
RESTART_Y_UPPER = 0xD0 | 0x0F
RESTART_Y_UNDER = 64
RESTART_UNDERGROUND_TEST_Y = 71
JUNGLE_GROUND = 32
UNDER_GROUND = 86

# The animated copyright, drawn by a third ShowDigits pass after the kernel. Six
# sixteen-row blocks feed six eight-row slots, so the eight-row window can slide
# through them; the row is read off the reference frame.
COPYRIGHT_H = 16
COPYRIGHT_ROW = 189
COPYRIGHT_PATTERNS = (
    (   # CopyRight0
        0b00000000, 0b00000000, 0b11110111, 0b10010101,
        0b10000111, 0b10000000, 0b10010000, 0b11110000,
        0b10101101, 0b10101001, 0b11101001, 0b10101001,
        0b11101101, 0b01000001, 0b00001111, 0b00000000,
    ),
    (   # CopyRight1
        0b01000111, 0b01000001, 0b01110111, 0b01010101,
        0b01110101, 0b00000000, 0b00000000, 0b00000000,
        0b01010000, 0b01011000, 0b01011100, 0b01010110,
        0b01010011, 0b00010001, 0b11110000, 0b00000000,
    ),
    (   # CopyRight2
        0b00000011, 0b00000000, 0b01001011, 0b01001010,
        0b01101011, 0b00000000, 0b00001000, 0b00000000,
        0b10111010, 0b10001010, 0b10111010, 0b10100010,
        0b00111010, 0b10000000, 0b11111110, 0b00000000,
    ),
    (   # CopyRight3
        0b10000000, 0b10000000, 0b10101010, 0b10101010,
        0b10111010, 0b00100010, 0b00100111, 0b00000010,
        0b11101001, 0b10101011, 0b10101111, 0b10101101,
        0b11101001, 0b00000000, 0b00000000, 0b00000000,
    ),
    (   # CopyRight4
        0b00000000, 0b00000000, 0b00010001, 0b00010001,
        0b00010111, 0b00010101, 0b00010111, 0b00000000,
        0b00000000, 0b00000000, 0b00000000, 0b00000000,
        0b00000000, 0b00000000, 0b00000000, 0b00000000,
    ),
    (   # CopyRight5
        0b00000000, 0b00000000, 0b01110111, 0b01010100,
        0b01110111, 0b01010001, 0b01110111, 0b00000000,
        0b00000000, 0b00000000, 0b00000000, 0b00000000,
        0b00000000, 0b00000000, 0b00000000, 0b00000000,
    ),
)

# `Space` immediately precedes CopyRight0 in the ROM, and the pointer arithmetic
# below runs off the front of CopyRight0 into it. That is not an accident: it is
# what lets the message scroll in from nothing.
_COPYRIGHT_STREAM = (0,) * DIGIT_H + tuple(b for block in COPYRIGHT_PATTERNS for b in block)


def copyright_slot_bitmap(scroll: int, slot: int) -> np.ndarray:
    """One copyright slot as a top-down DIGIT_H x DIGIT_W boolean bitmap.

    `.loopCopyright` seeds digitPtr+10 with `CopyRight5 - COPYRIGHT_H/2 + scroll`
    and walks back COPYRIGHT_H a slot at a time, so slot n reads the eight bytes
    starting at `CopyRightN - COPYRIGHT_H/2 + scroll`. For n > 0 a low scroll
    runs that window back into CopyRight(n-1), which is what makes the message
    travel sideways as it slides; for slot 0 it runs back into Space.
    """
    start = COPYRIGHT_H * slot + scroll
    rows = _COPYRIGHT_STREAM[start:start + DIGIT_H][::-1]
    return np.array(
        [[bool((row >> (DIGIT_W - 1 - col)) & 1) for col in range(DIGIT_W)] for row in rows],
        dtype=bool,
    )


COPYRIGHT_SCROLL_STOPS = COPYRIGHT_H // 2 + 1
_COPYRIGHT_GLYPHS = jnp.asarray(
    np.stack([
        np.stack([copyright_slot_bitmap(s, n) for n in range(len(COPYRIGHT_PATTERNS))])
        for s in range(COPYRIGHT_SCROLL_STOPS)
    ])
)


def copyright_scroll(no_game_scroll, sound_idx):
    """The copyright window offset, 0..COPYRIGHT_H/2.

        ldy    #COPYRIGHT_H/2   ; 2
        lda    noGameScroll     ; 3
        ldx    soundIdx         ; 3
        beq    .noSound0        ; 2³
        lda    #0               ; 2
    .noSound0:
        lsr / lsr / lsr
        cmp    #20              ; 2                 scroll-animation
        bcs    .ok              ; 2³
        ldy    #0               ; 2
        cmp    #12              ; 2
        bcc    .ok              ; 2³
        sbc    #12              ; 2
        tay                     ; 2

    A tune in progress substitutes 0 for noGameScroll, so the marquee is pinned
    shut for the whole of the dead tune. It only moves once soundIdx is back to
    zero *and* noGameScroll is still counting - the intro, and the frames after
    the last life, where the game-over branch leaves the counter running.
    """
    a = jnp.where(sound_idx != jnp.int32(0), jnp.int32(0), no_game_scroll) >> jnp.int32(3)
    return jnp.where(
        a >= jnp.int32(20),
        jnp.int32(COPYRIGHT_H // 2),
        jnp.where(a < jnp.int32(12), jnp.int32(0), a - jnp.int32(12)),
    )


def advance_death_frame(no_game_scroll, sound_idx, sound_delay):
    """One NTSC frame of the copyright counter and the sound cadence.

    Ordered as the ROM frame is: the copyright counter sits just before the
    vertical blank, the soundIdx test just after it, and the sound routine last.
    `life_loss` therefore reads the soundIdx this frame *started* with, which is
    why the note that ends the tune and the frame that takes the life are the
    same frame.
    """
    # `dec noGameScroll / bne .endCopyright / dec noGameScroll` - the second dec
    # steps over zero, so a stopped game can never restart itself by counting.
    stepped = (no_game_scroll - jnp.int32(1)) & jnp.int32(0xFF)
    stepped = jnp.where(stepped == jnp.int32(0), jnp.int32(0xFF), stepped)
    scroll = jnp.where(no_game_scroll != jnp.int32(0), stepped, no_game_scroll)

    # `lda soundIdx / cmp #SOUND_FALLING-1 / bne .slipDecrease`
    life_loss = sound_idx == jnp.int32(SOUND_FALLING - 1)

    # `ldx soundIdx / beq .noSound / inc soundDelay / lda soundDelay / and #$03 /
    #  bne .skipNext / inc soundIdx`, then `lda SoundTab-1,x` - X is the index
    # from *before* the increment, so the tune runs one frame past its last note.
    playing = sound_idx != jnp.int32(0)
    delay = jnp.where(playing, (sound_delay + jnp.int32(1)) & jnp.int32(0xFF), sound_delay)
    idx = jnp.where(
        playing & ((delay & jnp.int32(3)) == jnp.int32(0)),
        sound_idx + jnp.int32(1),
        sound_idx,
    )
    note = _SOUND_TAB[jnp.clip(sound_idx - jnp.int32(1), 0, len(SOUND_TAB) - 1)]
    idx = jnp.where(playing & ((note & jnp.int32(0x80)) != jnp.int32(0)), jnp.int32(0), idx)
    return scroll, idx, delay, life_loss


# --- Static pits: sceneType 2 (tar) and 3 (swamp) ----------------------------
# GroundTypeTab points both at the same playfield block:
#
#   .byte <[Pit - PF2PatTab] ; tar pit
#   .byte <[Pit - PF2PatTab] ; swamp
#
# which is offset 16. Both entries are positive, so `lda GroundTypeTab,x /
# bpl .noQuickSand` is taken: these two scenes never reach the quicksand code,
# xPosQuickSand stays 0 and the shape is static. `.loopPF2Lst` then runs with
# `adc #6` -> Y = 22 and X = 6, filling PF2Lst[6..0] from PF2PatTab[22..16], so
# PF2Lst[i] is simply the i'th byte of `Pit`.
PIT_PF2_LST = (0x00, 0x01, 0x03, 0x0F, 0x7F, 0xFF, 0xFF)

# --- Composed-frame scanline budget -----------------------------------------
# Every kernel from 3 to 9 decrements Harry's Y counter exactly once per
# scanline - Kernel 6 included, by way of `tya / sec / sbc #8 / sta temp1`, which
# is why ContKernel resumes with the right value. yPosHarry is therefore a plain
# scanline counter and the mapping onto composed rows is one continuous line with
# no per-band term anywhere.
#
# The line counts, read off the kernel structure and its own comments:
KERNEL5_LINES = 7            # `ldx #6 / .loopGround` .. `bpl .loopGround`
KERNEL6_LINES = 8            # "Kernel 6 (8 lines)", y counting 7 -> 0
EXIT_HOLES_LINES = 1         # `.exitHoles` runs to its own `sta WSYNC`
CONT_KERNEL_LINES = 2        # ContKernel's two `sta WSYNC`
KERNEL7_LINES = 12           # "Kernel 7 (12 lines)", x counting 11 -> 0
EXIT_LADDER_TOP_LINES = 1    # `.exitLadderTop` likewise
KERNEL8_LINES = 15           # "Kernel 8 (15 lines)"
KERNEL9_LINES = 16           # "Kernel 9 (16 lines)"

# Harry standing on the jungle line has yPosHarry = JUNGLE_GROUND, and his feet -
# pattern byte 0, drawn when the counter reaches 0 - land on the last line Kernel
# 5 emits. That row is the anchor for the whole mapping; it is measured on the
# reference frame and it is the only tune in the port.
HARRY_FEET_OFFSET = -6

# The lines the kernels emit for the ground band, top-down. Kernel 5's first line
# carries the `ldx #$ff / stx PF0 / stx PF1 / stx PF2` all-solid writes before its
# own `lda PF2Lst,x / sta PF2` overwrites PF2 further along the same scanline, so
# those writes never occupy a line of their own: the band is fifteen lines, not
# sixteen. PF2Lst[6] is $ff in any case, so dropping the synthetic line leaves
# every rendered row byte-identical.
#   7 lines  Kernel 5's `lda PF2Lst,x` with x counting 6 -> 0
#   8 lines  Kernel 6's `lda PF2Lst,x` with x counting 0 -> 6; its last line
#            exits on `bmi .exitHoles` before writing, so PF2Lst[6] is held.
PIT_KERNEL5_ROWS = tuple(PIT_PF2_LST[i] for i in (6, 5, 4, 3, 2, 1, 0))
PIT_KERNEL6_ROWS = tuple(PIT_PF2_LST[i] for i in (0, 1, 2, 3, 4, 5, 6)) + (PIT_PF2_LST[6],)
PIT_PF2_ROWS = PIT_KERNEL5_ROWS + PIT_KERNEL6_ROWS
PIT_BAND_H = len(PIT_PF2_ROWS)

# `cpx #54 / tya / bne .endDoJump / jmp KilledHarry`: with ladderFlag zero the
# fall is fatal at yPosHarry 54, which is JUNGLE_GROUND + 22.
PIT_KILL_DEPTH = 54 - 32


def pf2_open_columns(pf2_byte: int) -> np.ndarray:
    """Screen columns a single reflected PF2 byte leaves uncovered.

    Kernel 2 sets `lda #%001 / sta CTRLPF`, so the playfield is reflected, and
    PF0/PF1 are held at $ff for the whole band. Only PF2 can open, and its eight
    four-pixel groups run bit 0 at columns 48..51 up to bit 7 at 76..79, mirrored
    back down over 80..111. A set bit is solid ground, a clear bit is pit.
    """
    open_columns = np.zeros(160, dtype=bool)
    for bit in range(8):
        if not (pf2_byte >> bit) & 1:
            left = 48 + bit * 4
            open_columns[left:left + 4] = True
            right = 108 - bit * 4
            open_columns[right:right + 4] = True
    return open_columns


PIT_OPEN_MASK = jnp.asarray(np.stack([pf2_open_columns(b) for b in PIT_PF2_ROWS]))


def harry_feet_row_for_rom_y(rom_y: int, ground_y: int) -> int:
    """The one mapping: composed row of Harry's feet for a given yPosHarry.

    `feet_row = K + yPosHarry`, with K pinned by the standing position on the
    jungle line - yPosHarry JUNGLE_GROUND puts his feet on Kernel 5's last row.
    K works out to 92 for a ground_y of 130. Nothing about this depends on which
    band Harry is in, because nothing in the kernels does either.
    """
    return int(ground_y) + HARRY_FEET_OFFSET - JUNGLE_GROUND + int(rom_y)


def pit_band_top_row(ground_y: int) -> int:
    """Screen row of the ground band's first line, i.e. Kernel 5's first.

    Kernel 5 ends on the row Harry's feet occupy while standing, so its seven
    lines are the seven ending there.
    """
    return harry_feet_row_for_rom_y(JUNGLE_GROUND, ground_y) - (KERNEL5_LINES - 1)


def underground_floor_row(underground_y: int) -> int:
    """Composed row Harry's feet rest on underground: K + UNDER_GROUND.

    Counting the kernel budget down from Kernel 5's last line puts this two rows
    above the end of Kernel 9, which is exactly where `K + 86` lands. The two
    derivations are independent and they agree, which is what fixes the row.
    """
    return int(underground_y) + HARRY_FEET_OFFSET


def underground_last_row(underground_y: int) -> int:
    """Kernel 9's final scanline. yPosHarry 86 sits one line above it."""
    return underground_floor_row(underground_y) + 1


def pit_harry_blank_top_row(ground_y: int) -> int:
    """First scanline on which the kernel stops putting Harry on the screen.

    Kernel 5 runs `jsr DrawHarry` on each of its seven lines, and the setup line
    above it writes no GRP0 at all, so it holds the latch from the DrawHarry that
    precedes it. Everything down to the end of Kernel 5 shows Harry.

    Kernel 6 opens every one of its eight iterations with

        .loopHoles:
            lda    #0               ; 2
            sta    GRP0             ; 3

    and `.exitHoles` repeats the pair, so GRP0 is zero for all eight lines. Harry
    is not drawn there whatever the playfield holds, and that is a property of the
    kernel, not of priority.

    Below the band ContKernel takes over with `lda #$ff / sta PF1` and
    `stx CTRLPF`, where X came from

            lda    ladderFlag       ; 3                 calculate playfield reflection
            and    #%000100         ; 2
            eor    #%100101         ; 2

    so bit 2 - playfield priority - is set for every scene whose ladderFlag is
    zero. The static pits are exactly those scenes, so from here on a solid PF1
    sits in front of Harry as well.

    The two mechanisms meet with no gap, which makes this single row the point
    where a sinking Harry stops being visible.
    """
    return pit_band_top_row(ground_y) + KERNEL5_LINES


def scene_is_static_pit(room_byte: jnp.ndarray) -> jnp.ndarray:
    """sceneType 2 (tar pit) and 3 (swamp) - the two static `Pit` scenes."""
    pt = pit_code_u8(room_byte.astype(jnp.uint8))
    return (pt == jnp.uint8(2)) | (pt == jnp.uint8(3))


# --- Dynamic ground: GroundTypeTab / PF2PatTab / the quicksand ---------------
# GroundTypeTab, indexed by sceneType, verbatim from pitfall.asm:
#
#     .byte <[OneHole    - PF2PatTab] ; one hole                sceneType 0
#     .byte <[ThreeHoles - PF2PatTab] ; three holes             sceneType 1
#     .byte <[Pit        - PF2PatTab] ; tar pit                 sceneType 2
#     .byte <[Pit        - PF2PatTab] ; swamp                   sceneType 3
#     .byte <[Pit        - PF2PatTab] ; swamp with crocodiles   sceneType 4
#     .byte $80                       ; black quicksand with treasure  5
#     .byte $80                       ; black quicksand         sceneType 6
#     .byte $80                       ; blue quicksand          sceneType 7
#
# The first five entries are positive, so `lda GroundTypeTab,x / bpl
# .noQuickSand` takes the static branch: xPosQuickSand is pinned at 0 and PF2Lst
# is refilled from the table every frame. The $80 entries are negative, which
# routes scenes 5..7 through `.doQuickSand`, where both the PF2Lst window and
# xPosQuickSand become functions of frameCnt. So scenes 2, 3 and 4 share the one
# static `Pit` shape and only scenes 5..7 move.
GROUND_TYPE_PIT = 16          # <[Pit - PF2PatTab]
GROUND_TYPE_QUICKSAND = 0x80  # the negative entry shared by scenes 5..7

# PF2PatTab, verbatim. OneHole at offset 0, ThreeHoles at 8, Pit at 16, and the
# ROM pads the rest with $ff bytes that the sliding quicksand window reads.
PF2_PAT_TAB = (
    0x7F, 0x7F, 0x7F, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,  # OneHole
    0x78, 0x78, 0x78, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,  # ThreeHoles
    0x00, 0x01, 0x03, 0x0F, 0x7F, 0xFF, 0xFF, 0xFF,  # Pit
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,  # $ff padding
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,  # $ff padding
)

# QuickSandTab, verbatim. The loop reads QuickSandTab+2,x and QuickSandTab,x
# with x = frameCnt>>6 in 0..3, so index 5 is reached; the ROM's own comment
# ("next byte (0) overlaps") marks that read, which lands on LadderTab[0] =
# BLACKPIT|DISABLE = $80. Its bit 7 is masked off by the AND, so the value
# behaves as 0 there.
QUICK_SAND_TAB = (0x00, 0x0F, 0x0F, 0x00, 0x0F, 0x80)

# QuickSandSize is only five bytes (0, 4, 8, 16, 28). The index y produced by
# `.doQuickSand` runs 0..15, and entries 5..15 read straight into ClimbColTab,
# whose first eleven bytes are all DARK_GREEN = GREEN-$04 = $d2. That large
# value is what closes the pit: 44+210 = 254 >= xPosHarry always, so the bounds
# loop's first compare reports "in bounds" for every column.
DARK_GREEN_NTSC = 0xD2
QUICK_SAND_SIZE = (0, 4, 8, 16, 28) + (DARK_GREEN_NTSC,) * 11

_QUICK_SAND_TAB = jnp.asarray(QUICK_SAND_TAB, dtype=jnp.int32)
_QUICK_SAND_SIZE = jnp.asarray(QUICK_SAND_SIZE, dtype=jnp.int32)


def quicksand_index(frame_cnt_ntsc) -> jnp.ndarray:
    """The quicksand window index y (0..15) for one NTSC frame.

        lda    frameCnt
        lsr / lsr / pha / lsr / lsr / lsr / lsr / tax   ; x = frameCnt>>6
        pla / and QuickSandTab+2,x / eor QuickSandTab,x / tay

    Only bits 2..5 of frameCnt survive the AND, so y changes every fourth NTSC
    frame - every second JAX step.
    """
    f = frame_cnt_ntsc.astype(jnp.int32) & jnp.int32(0xFF)
    x = f >> jnp.int32(6)
    a = f >> jnp.int32(2)
    return (a & _QUICK_SAND_TAB[x + 2]) ^ _QUICK_SAND_TAB[x]


def quicksand_border(frame_cnt_ntsc) -> jnp.ndarray:
    """xPosQuickSand for one NTSC frame: `lda QuickSandSize,y`."""
    return _QUICK_SAND_SIZE[quicksand_index(frame_cnt_ntsc)]


def pit_band_pf2_bytes(window: int) -> tuple:
    """The fifteen PF2 bytes of the ground band for a PF2PatTab window.

    `.loopPF2Lst` fills PF2Lst[i] = PF2PatTab[window+i] for i in 0..6. Kernel 5
    then writes PF2Lst[6..0] down its seven lines and Kernel 6 writes
    PF2Lst[0..6], holding PF2Lst[6] on its eighth (`dey / bmi .exitHoles` exits
    before the last write). Static scenes sit on window 16 (`Pit`); the
    quicksand slides the window to 16+y.
    """
    lst = [PF2_PAT_TAB[window + i] for i in range(7)]
    kernel5 = [lst[6], lst[5], lst[4], lst[3], lst[2], lst[1], lst[0]]
    kernel6 = [lst[0], lst[1], lst[2], lst[3], lst[4], lst[5], lst[6], lst[6]]
    return tuple(kernel5 + kernel6)


# The six reachable windows are 16..21 (y = 0..5); y >= 5 all read the $ff
# padding, so window 21 stands for every closed phase. The renderer reads this
# table (indexed by the quicksand state), and the collision bounds read
# HoleBoundsTab displaced by the same xPosQuickSand, so the drawn opening and
# the falling interval are always two views of the one ROM byte.
PIT_OPEN_MASKS = jnp.asarray(
    np.stack([
        np.stack([pf2_open_columns(b) for b in pit_band_pf2_bytes(GROUND_TYPE_PIT + y)])
        for y in range(6)
    ])
)


def quicksand_window_index(x_pos_quicksand) -> jnp.ndarray:
    """Recover the quicksand index y (clamped to 5) from the xPosQuickSand byte.

    QuickSandSize is a bijection over the reachable values 0/4/8/16/28, and the
    closed phases all share DARK_GREEN ($d2), which maps to the all-$ff window.
    """
    x = x_pos_quicksand.astype(jnp.int32) & jnp.int32(0xFF)
    return jnp.where(
        x == jnp.int32(0), jnp.int32(0),
        jnp.where(x == jnp.int32(4), jnp.int32(1),
                  jnp.where(x == jnp.int32(8), jnp.int32(2),
                            jnp.where(x == jnp.int32(16), jnp.int32(3),
                                      jnp.where(x == jnp.int32(28), jnp.int32(4),
                                                jnp.int32(5))))),
    )


def scene_is_quicksand(room_byte: jnp.ndarray) -> jnp.ndarray:
    """sceneType 5, 6, 7 - the negative GroundTypeTab entries ($80)."""
    pt = pit_code_u8(room_byte.astype(jnp.uint8))
    return pt >= jnp.uint8(5)


def scene_is_pit(room_byte: jnp.ndarray) -> jnp.ndarray:
    """sceneType 2..7 - every scene built on the `Pit` playfield block.

    These are exactly the no-ladder scenes, so they are also the scenes where a
    fall through the opening is fatal (`cpx #54 / tya / bne / jmp KilledHarry`).
    """
    pt = pit_code_u8(room_byte.astype(jnp.uint8))
    return pt >= jnp.uint8(2)


# LadderTab, indexed by sceneType, verbatim. Only bit 7 matters here: it is the
# BLACKPIT/BLUEPIT selector read by `lda LadderTab,x / bpl .noPit` - a negative
# entry (BLACKPIT) makes MainLoop overwrite colorLst+8 with colorLst+4 (BLACK),
# a positive one (BLUEPIT) leaves the swamp's BLUE in place.
LADDER_TAB = (
    0x80, 0x80, 0x82, 0x02,  # BLACKPIT|DIS, BLACKPIT|DIS, BLACKPIT|EN, BLUEPIT|EN
    0x02, 0x80, 0x82, 0x02,  # BLUEPIT|DIS, BLACKPIT|DIS, BLACKPIT|EN, BLUEPIT|EN
)
_LADDER_TAB = jnp.asarray(LADDER_TAB, dtype=jnp.uint8)


def pit_is_blue(room_byte: jnp.ndarray) -> jnp.ndarray:
    """LadderTab bit 7 clear -> the pit shows the BLUE swamp colour, not BLACK."""
    scene = pit_code_u8(room_byte).astype(jnp.int32)
    return (_LADDER_TAB[scene] & jnp.uint8(0x80)) == jnp.uint8(0)


# --- Crocodiles (sceneType 4) -------------------------------------------------
# CrocoTab is 1 for sceneType 4 alone (`.ds CROCO_SCENE,0 / .byte 1 / ...`), so
# the crocodile branch of ProcessObjects runs only there. ContRandom then puts
# xPosObject at 60 (`lda #60 / sta xPosObject`) and ProcessObjects forces
# NUSIZ1 = THREE_COPIES, so the hardware draws three copies of the one GRP1
# pattern, 16 pixels apart: boxes at 60, 76 and 92.
CROCO_X = 60
CROCO_COPY_SPACING = 16
CROCO_XS = (CROCO_X, CROCO_X + CROCO_COPY_SPACING, CROCO_X + 2 * CROCO_COPY_SPACING)
CROCO_W = 8

# Croco0 (jaws open) and Croco1 (jaws closed), verbatim. As with every pattern
# table in this port, index 0 is the BOTTOM row: Kernel 5 draws object rows
# 14..8 and Kernel 6 rows 7..0, so band line j shows pattern row 14-j. Row 15
# is never drawn and is empty in both frames. CrocoColor is DARK_GREEN-2 ($d0)
# for all nine occupied rows, so the crocodile is one flat dark green.
CROCO_PATTERNS = (
    (   # Croco0 - jaws open
        0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b11111111,
        0b10101011, 0b00000011, 0b00000011, 0b00001011, 0b00101110, 0b10111010,
        0b11100000, 0b10000000, 0b00000000, 0b00000000,
    ),
    (   # Croco1 - jaws closed
        0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b11111111,
        0b10101011, 0b01010101, 0b11111111, 0b00000110, 0b00000100, 0b00000000,
        0b00000000, 0b00000000, 0b00000000, 0b00000000,
    ),
)
CROCO_BAND_ROWS = 15  # the ground band's line count; pattern rows 14..0


def croc_band_bitmap(pattern_index: int) -> np.ndarray:
    """One crocodile as a top-down 15x8 bitmap positioned inside the band."""
    pattern = CROCO_PATTERNS[pattern_index]
    return np.array(
        [[bool((pattern[14 - j] >> (7 - i)) & 1) for i in range(CROCO_W)] for j in range(CROCO_BAND_ROWS)],
        dtype=bool,
    )


def croc_jaws_open(frame_cnt_ntsc) -> jnp.ndarray:
    """`bit frameCnt / bpl`: jaws hang on bit 7 of frameCnt alone.

    ProcessObjects picks Croco0 while bit 7 is clear (`bpl .skipClosed`,
    "open croco jaws? yes") and Croco1 while it is set, and the bounds loop's
    `.noCroco1` applies the same bit to pick HoleBoundsTab row 4 (open) or row 3
    (closed). So the mouth is open for frameCnt 0..127 and closed for 128..255 -
    128 NTSC frames (~2.1s) each way, and every crocodile switches together.
    """
    f = frame_cnt_ntsc.astype(jnp.int32) & jnp.int32(0xFF)
    return (f & jnp.int32(0x80)) == jnp.int32(0)


# --- Surface objects: fire and treasures (the Kernel 5/6 GRP1 object) --------
# Fire, the cobra and the four treasures all reach the screen through the same
# path: ProcessObjects points objPatPtr/objColPtr at a pattern and a color
# table, and Kernels 5/6 draw object rows 14..0 on the fifteen ground-band
# lines, so object pattern row r lands on band line 15 - r. The sixteenth
# pattern row is never emitted, and the box's top line sits one row above the
# band. xPosObject (= 124, set by ContRandom) is the box's left edge; NUSIZ1 is
# ONE_COPY for all of them, so each is one eight-pixel-wide player.
OBJECT_BOX_H = 16


def object_box_top_row(ground_y: int) -> int:
    """Screen row of the GRP1 object box's top line (pattern row 15)."""
    return pit_band_top_row(ground_y) - 1


# Atari color byte -> RGB, read off raw ALE frames of the objects concerned.
OBJ_COLOR_RGB = {
    0x04: (111, 111, 111),   # GREY-2      (money bag)
    0x06: (142, 142, 142),   # GREY        (silver bar)
    0x0E: (236, 236, 236),   # WHITE       (bar/ring sparkle)
    0x10: (72, 72, 0),       # BROWN-2     (fire logs)
    0x12: (105, 105, 15),    # BROWN       (money bag tie)
    0x1E: (252, 252, 84),    # YELLOW      (gold bar, ring)
    0x2E: (236, 200, 96),    # fire flame (a literal in the ASM's FireColor)
    0x3E: (252, 188, 116),   # ORANGE      (fire body)
    0x42: (167, 26, 26),     # DARK_RED    (RingColor's two real bytes; unseen)
}

# Fire0 and Fire1, verbatim; index 0 is the bottom row. The flame leans right in
# Fire1. FireColor is indexed by the same pattern row.
FIRE_PATTERNS = (
    (   # Fire0
        0b00000000, 0b11000011, 0b11100111, 0b01111110, 0b00111100, 0b00011000,
        0b00111100, 0b01111100, 0b01111100, 0b01111000, 0b00111000, 0b00111000,
        0b00110000, 0b00110000, 0b00010000, 0b00010000,
    ),
    (   # Fire1
        0b00000000, 0b11000011, 0b11100111, 0b01111110, 0b00111100, 0b00011000,
        0b00111100, 0b00111110, 0b00111110, 0b00011110, 0b00011100, 0b00011100,
        0b00001100, 0b00001100, 0b00001000, 0b00001000,
    ),
)
FIRE_COLOR_ROWS = (
    0x10, 0x10, 0x10, 0x10, 0x10, 0x3E, 0x3E, 0x3E,
    0x2E, 0x2E, 0x2E, 0x2E, 0x2E, 0x2E, 0x2E, 0x2E,
)

# Treasure patterns, verbatim; index 0 is the bottom row. Bar0/Bar1 are the two
# sparkle frames shared by the silver and gold bars.
MONEY_BAG_PATTERN = (
    0b00000000, 0b00111110, 0b01110111, 0b01110111, 0b01100011, 0b01111011,
    0b01100011, 0b01101111, 0b01100011, 0b00110110, 0b00110110, 0b00011100,
    0b00001000, 0b00011100, 0b00110110, 0b00000000,
)
BAR0_PATTERN = (
    0b00000000, 0b11111000, 0b11111100, 0b11111110, 0b11111110, 0b01111110,
    0b00111110, 0b00000000, 0b00010000, 0b00000000, 0b01010100, 0b00000000,
    0b10010010, 0b00000000, 0b00010000, 0b00000000,
)
BAR1_PATTERN = (
    0b00000000, 0b11111000, 0b11111100, 0b11111110, 0b11111110, 0b01111110,
    0b00111110, 0b00000000, 0b00000000, 0b00101000, 0b00000000, 0b01010100,
    0b00000000, 0b00010000, 0b00000000, 0b00000000,
)
RING_PATTERN = (
    0b00000000, 0b00000000, 0b00111000, 0b01101100, 0b01000100, 0b01000100,
    0b01000100, 0b01101100, 0b00111000, 0b00010000, 0b00111000, 0b01111100,
    0b00111000, 0b00000000, 0b00000000, 0b00000000,
)

# Treasure color rows, per object row, verbatim. MoneyBagColor is GREY-2 with a
# single BROWN tie row; SilverBarColor is a grey bar under white sparkles;
# Color1PtrTab points the gold bar at GoldBarColor+1, so its rows are shifted by
# one; RingColor has only two real bytes (DARK_RED, on the empty top rows) and
# the kernel's reads run straight on into GoldBarColor, giving a yellow band and
# a white gem.
MONEY_BAG_COLOR_ROWS = (0x04,) * 8 + (0x04, 0x04, 0x04, 0x04, 0x12, 0x04, 0x04, 0x04)
SILVER_BAR_COLOR_ROWS = (0x06,) * 7 + (0x0E,) * 9
_GOLD_BAR_COLOR_TABLE = (0x1E,) * 8 + (0x0E,) * 9
GOLD_BAR_COLOR_ROWS = _GOLD_BAR_COLOR_TABLE[1:17]  # GoldBarColor+1
RING_COLOR_ROWS = (0x42, 0x42) + _GOLD_BAR_COLOR_TABLE[0:14]

# The four treasure identities, indexed by objectType & 3 (ProcessObjects:
# `lda objectType / and #$03 / ora #$08 / tax`). AnimateTab is OBJECT_H for the
# two bars (they shimmer between Bar0 and Bar1 on random2 bit 4) and 0 for the
# money bag and the ring (static).
TREASURE_PATTERNS = (MONEY_BAG_PATTERN, (BAR0_PATTERN, BAR1_PATTERN), (BAR0_PATTERN, BAR1_PATTERN), RING_PATTERN)
TREASURE_COLOR_ROWS = (
    MONEY_BAG_COLOR_ROWS,
    SILVER_BAR_COLOR_ROWS,
    GOLD_BAR_COLOR_ROWS,
    RING_COLOR_ROWS,
)
TREASURE_ANIMATED = (False, True, True, False)
_TREASURE_ANIMATED = jnp.asarray(TREASURE_ANIMATED, dtype=jnp.bool_)

# TreasureMask, verbatim: the persistence bit for objectType y is 1 << (7 - y).
TREASURE_MASK = tuple(1 << (7 - i) for i in range(8))

# `.incScore`: `lda objectType / and #$03 / asl x4 / adc #$20 / sed / adc
# scoreMed`. (ti*16 + 32) read as a BCD byte is 20/30/40/50 for ti = 0..3, and
# scoreMed is the thousands:hundreds pair, so each treasure adds (ti+2) * 1000.
TREASURE_SCORES = (2000, 3000, 4000, 5000)
_TREASURE_SCORES = jnp.asarray(TREASURE_SCORES, dtype=jnp.int32)

# `lda #31 / sta treasureCnt` in InitGame: 32 treasures, counted down to -1.
TREASURE_COUNT_INIT = 31


def object_band_bitmap(pattern, color_rows, color_id_by_atari, transparent_id):
    """One GRP1 object frame as a top-down 16x8 palette-id bitmap.

    Box row j shows object row 15 - j (the kernels count the pointer down as the
    raster descends), and object row 15 is never drawn, so box row 0 is left
    transparent. Each lit pixel takes that object row's color-table entry.
    """
    rows = np.full((OBJECT_BOX_H, 8), transparent_id, dtype=np.int32)
    for j in range(OBJECT_BOX_H):
        r = (OBJECT_BOX_H - 1) - j
        if r == OBJECT_BOX_H - 1:
            continue  # row 15 is never drawn
        byte = pattern[r]
        color_id = color_id_by_atari[color_rows[r]]
        for c in range(8):
            if (byte >> (7 - c)) & 1:
                rows[j, c] = color_id
    return rows


def treasure_collected(room_byte, treasure_bits) -> jnp.ndarray:
    """CheckTreasures: is this room's treasure bit already set?

        lda random / rol / rol / rol / and #$03 / tax   ; x = bits 6..7 (treePat)
        ldy objectType / lda TreasureMask,y / tay
        and treasureBits,x                              ; Z set iff not collected
    """
    rb = room_byte.astype(jnp.uint8)
    tree = tree_variant_u8(rb).astype(jnp.int32)
    obj = obj_code_u8(rb).astype(jnp.int32)
    mask = jnp.asarray(TREASURE_MASK, dtype=jnp.uint8)[obj]
    return (treasure_bits[tree] & mask) != jnp.uint8(0)


# --- Ground objects: logs ----------------------------------------------------
# `ContRandom` puts every ground object here on entering a scene:
# `lda #124 / sta xPosObject`. It is the left edge of the GRP1 box.
ASM_OBJECT_START_X = 124
ASM_SCREENWIDTH = 160

# objectType ids: 0..3 are rolling logs, 4..5 stationary ones.
ID_STATIONARY = 4

# NuSize1Tab, resolved into the offsets the TIA's NUSIZ1 replicas land on. The
# ROM never positions logs individually: it sets one xPosObject and lets the
# hardware repeat the player, so the spacing is a property of the NUSIZ value.
NUSIZ1_COPY_OFFSETS = (
    (0,),          # ONE_COPY          - one rolling log
    (0, 16),       # TWO_COPIES        - two rolling logs, close
    (0, 32),       # TWO_WIDE_COPIES   - two rolling logs, medium
    (0, 32, 64),   # THREE_MED_COPIES  - three rolling logs
    (0,),          # ONE_COPY          - one stationary log
    (0, 32, 64),   # THREE_MED_COPIES  - three stationary logs
)
LOG_MAX_COPIES = 3

# `lda frameCnt / lsr / bcs .skipLogs` then `dex`: one pixel left every second
# NTSC frame. One JAX step is two NTSC frames, so that is one pixel per step.
LOG_MOVE_PX_PER_STEP = 1


def log_left_edges(log_xs, logs_are_rolling, steps_elapsed, screen_width):
    """Left edges of the log copies after `steps_elapsed` steps of rolling.

    `ldx xPosObject / bne .skipResetLogs / ldx #SCREENWIDTH / dex` decrements the
    shared x and turns 0 into SCREENWIDTH-1, which is a plain modular decrement.
    Stationary logs never reach that code. The renderer and the collision test
    both come through here, so the drawn logs are the collidable ones.
    """
    width = jnp.int32(screen_width)
    travelled = steps_elapsed.astype(jnp.int32) * jnp.int32(LOG_MOVE_PX_PER_STEP)
    base = jnp.mod(log_xs.astype(jnp.int32), width)
    return jnp.where(logs_are_rolling, jnp.mod(base - travelled, width), base)


# --- Ladder ------------------------------------------------------------------
# climbPos is the ROM variable itself: 0 means "not on the ladder", otherwise it
# runs from LADDER_TOP at the top rung down to LADDER_BOTTOM at the floor. It is
# stored as-is here, so no rung counting has to be translated.
LADDER_TOP = 11
LADDER_BOTTOM = 22

# `lda xPosHarry / sec / sbc #68 / cmp #15 / bcs .endStartClimb`: an unsigned
# byte compare, so the window is the fifteen columns 68..82 and nothing else.
LADDER_ENTRY_X_MIN = 68
LADDER_ENTRY_X_SPAN = 15
# `lda #SCREENWIDTH/2-4 / sta xPosHarry` snaps Harry to the ladder on entry.
LADDER_SNAP_X = 76
# `lda yPosHarry / cmp #84 / bcc .skipClimbUp`: only this close to the floor may
# UP put Harry on the ladder from below.
LADDER_ENTRY_FROM_BELOW_Y = 84
# `.skipLadderBottom` writes `ldx #SCREENWIDTH/2+6 / stx yPosHarry`, which is 86,
# the same UNDER_GROUND the climb arithmetic produces for climbPos 21.
LADDER_BOTTOM_EXIT_Y = 86
# `.notAtTop` leaves Harry one pixel above the jungle line and lets JumpTab take
# over from index 1 on the following frame.
LADDER_EXIT_Y = 31
# `lda frameCnt / and #$07 / bne .skipAnimClimb`: one rung every eighth frame.
LADDER_CLIMB_MASK = 0x07
# `lda frameCnt / and #$03 / tax / lsr / bcs .endHarryId`: the walking block runs
# when bit 0 of frameCnt is clear, and animates only when both low bits are.
HARRY_MOVE_MASK = 0x03


# --- Underground wall, from the ASM literals ---------------------------------
# `Wall`, verbatim. All eight hardware bits are kept: bit 0 is clear in every
# byte, and that always-clear bit is what leaves the one-pixel gap down the
# column's right side. Trimming it would narrow the sprite.
WALL_PATTERN = (
    0b11111110, 0b10111010, 0b10111010, 0b10111010,
    0b11111110, 0b11101110, 0b11101110, 0b11101110,
    0b11111110, 0b10111010, 0b10111010, 0b10111010,
    0b11111110, 0b11101110, 0b11101110, 0b11101110,
)
WALL_W = 8

# `WallColor` is only fourteen bytes:
#
#   WallColor:
#       .byte GREY, DARK_RED, DARK_RED, DARK_RED, GREY, DARK_RED, DARK_RED, DARK_RED
#       .byte GREY, DARK_RED, DARK_RED, DARK_RED, GREY, DARK_RED
#   RingColor:
#       .byte DARK_RED, DARK_RED
#
# and the kernel's y = 14 and 15 reads run straight on into RingColor, whose two
# bytes are DARK_RED as well. GREY therefore lands on indices 0, 4, 8 and 12 -
# exactly the $fe bytes - so every full-width mortar course is grey and every
# brick row is dark red. The row prepared before Kernel 8 is written DARK_RED
# outright (`lda #DARK_RED / sta COLUP1`), which agrees with index 15 anyway.
WALL_COLOR_GREY_INDICES = (0, 4, 8, 12)

# Kernel 8 walks `(wallPatPtr),y` from 15 down to 0 and Kernel 9 walks
# `(undrPatPtr),y` over the same range. In a ladder scene ContRandom points both
# at `Wall` (`stx wallPatPtr` and `stx undrPatPtr`), so the column is the same
# sixteen-row bitmap laid down twice, top-down, with no row repeated or dropped.
WALL_KERNEL_COPIES = 2


def wall_render_rows() -> tuple[tuple[int, bool], ...]:
    """(pattern byte, row is grey) per screen row, top-down, for the whole wall.

    The kernels count y down as the raster descends, so the first screen row is
    `Wall[15]` and the last of each copy is `Wall[0]`.
    """
    rows = []
    for _ in range(WALL_KERNEL_COPIES):
        for y in range(len(WALL_PATTERN) - 1, -1, -1):
            rows.append((WALL_PATTERN[y], y in WALL_COLOR_GREY_INDICES))
    return tuple(rows)


def climb_pos_to_rom_y(climb_pos):
    """`lda climbPos / asl / sec / rol / adc #1` - yPosHarry from climbPos.

    `asl` leaves 2c with the old bit 7 in carry, `sec` forces carry, `rol` gives
    4c+1 and moves bit 7 of 2c into carry, and `adc #1` adds that carry back. For
    every climbPos the ladder can hold, 2c stays under 128, so the carry into the
    `adc` is clear and the result is 4c+2 - 46 at LADDER_TOP, 86 at
    LADDER_BOTTOM-1, which is exactly UNDER_GROUND.
    """
    c = climb_pos.astype(jnp.int32) & jnp.int32(0xFF)
    shifted = (c << jnp.int32(1)) & jnp.int32(0xFF)          # asl  (carry discarded by sec)
    rolled = ((shifted << jnp.int32(1)) | jnp.int32(1)) & jnp.int32(0xFF)   # sec / rol
    carry = (shifted >> jnp.int32(7)) & jnp.int32(1)         # rol's carry out
    return (rolled + jnp.int32(1) + carry) & jnp.int32(0xFF)  # adc #1


def rom_y_to_player_y(rom_y, consts):
    """The Part 3 mapping: player_y = yPosHarry + (ground_y - JUNGLE_GROUND)."""
    return rom_y.astype(jnp.float32) + jnp.float32(
        float(consts.ground_y) - float(JUNGLE_GROUND)
    )


def _get_default_pitfall_asset_config() -> tuple:
    """Default declarative asset manifest for Pitfall."""
    return (
        {'name': 'background', 'type': 'background'},
        {
            'name': 'background_tree_variant_0',
            'type': 'single',
            'file': 'background_tree_variant_0.npy',
        },
        {
            'name': 'background_tree_variant_1',
            'type': 'single',
            'file': 'background_tree_variant_1.npy',
        },
        {
            'name': 'background_tree_variant_2',
            'type': 'single',
            'file': 'background_tree_variant_2.npy',
        },
        {
            'name': 'background_tree_variant_3',
            'type': 'single',
            'file': 'background_tree_variant_3.npy',
        },
        {
            'name': 'wall',
            'type': 'single',
            'file': 'wall.npy',
        },
        {
            'name': 'harry_idle',
            'type': 'group',
            'files': ['harryidle1.npy'],
        },
        {
            'name': 'harry_run',
            'type': 'group',
            # ROM Harry0..Harry4. Filenames do not match those ids.
            'files': [
                'harryrunning1.npy',  # Harry0
                'harryrunning5.npy',  # Harry1
                'harryrunning4.npy',  # Harry2
                'harryrunning3.npy',  # Harry3
                'harryrunning2.npy',  # Harry4
            ],
        },
        {
            'name': 'harry_swing',
            'type': 'group',
            'files': ['harryswinging.npy'],  # Harry6
        },
        {
            'name': 'harry_climb',
            'type': 'group',
            'files': ['harryclimb2.npy', 'harryclimb1.npy'],  # Harry7, Harry8
        },
        # harryjumping1.npy and harryjumping2.npy are byte-identical copies of
        # harryrunning3 (Harry3) and harryrunning1 (Harry0), so they are not
        # loaded: HarryPtrTab has no separate jump pattern.
        # The scorpion has no capture entry: ScorpionColor is WHITE throughout,
        # so a grab could only contribute shape, and Scorpion0/Scorpion1 give
        # that exactly. See SCORPION_PATTERNS.
        # ROM order, which the filenames invert: cobra0.npy captured Cobra1 and
        # cobra1.npy captured Cobra0. `random2 & AnimateTab[ID_COBRA]` indexes
        # this list, so listing them the other way round drew the wrong frame of
        # the pair. Each capture's alpha is the pattern and its RGB is
        # CobraColor; see COBRA_PATTERNS.
        {
            'name': 'cobra',
            'type': 'group',
            'files': ['cobra1.npy', 'cobra0.npy'],
        },
    )


def pit_code_u8(room_byte: jnp.ndarray) -> jnp.ndarray:
    """Bits 3..5 (uint8 0..7)."""
    return (room_byte.astype(jnp.uint8) >> jnp.uint8(3)) & jnp.uint8(0x7)


def obj_code_u8(room_byte: jnp.ndarray) -> jnp.ndarray:
    """Bits 0..2 (uint8 0..7)."""
    return room_byte.astype(jnp.uint8) & jnp.uint8(0x7)


def wall_side_u8(room_byte: jnp.ndarray) -> jnp.ndarray:
    """Bit 7 (uint8 0..1). 0=left, 1=right."""
    return (room_byte.astype(jnp.uint8) >> jnp.uint8(7)) & jnp.uint8(1)


def tree_variant_u8(room_byte: jnp.ndarray) -> jnp.ndarray:
    """Bits 6..7 (uint8 0..3)."""
    return (room_byte.astype(jnp.uint8) >> jnp.uint8(6)) & jnp.uint8(0x3)


def pit_type(room_byte: jnp.ndarray) -> jnp.ndarray:
    """Pit type is bits 3..5 of the room byte (uint8 0..7)."""
    return pit_code_u8(room_byte)


def has_scorpion_from_room_byte(room_byte: jnp.ndarray) -> jnp.ndarray:
    """Scorpion is present when pit type does not include a ladder (pit_type not in {0,1})."""
    pt = pit_code_u8(room_byte.astype(jnp.uint8))
    has_ladder = (pt == jnp.uint8(0)) | (pt == jnp.uint8(1))
    return ~has_ladder


def has_vine_from_room_byte(room_byte: jnp.ndarray) -> jnp.ndarray:
    """Vine/rope presence from room-byte decode rules."""
    rb = room_byte.astype(jnp.uint8)
    pt = pit_code_u8(rb)
    obj = obj_code_u8(rb)

    is_croc_pit = pt == jnp.uint8(0b100)
    is_croc_vine_obj = (
        (obj == jnp.uint8(0b010)) |
        (obj == jnp.uint8(0b011)) |
        (obj == jnp.uint8(0b110)) |
        (obj == jnp.uint8(0b111))
    )

    is_shifting_tar = pt == jnp.uint8(0b110)
    return (is_croc_pit & is_croc_vine_obj) | is_shifting_tar


def room_hazards_from_room_byte(room_byte: jnp.ndarray) -> tuple[
    chex.Array,
    chex.Array,
    chex.Array,
    chex.Array,
    chex.Array,
    chex.Array,
]:
    """Decode hazards/objects from the original Pitfall room byte.

    Mapping (bits 0..2) applies for non-croc (pit!=100b) and non-treasure (pit!=101b) rooms:
      000: 1 rolling log
      001: 2 rolling logs
      010: 2 rolling logs
      011: 3 rolling logs
      100: 1 stationary log
      101: 3 stationary logs
      110: fire
      111: snake

    Overrides:
      pit==100 (croc room): no objects/hazards (crocodiles instead)
      pit==101 (treasure+quicksand): no logs/fire/snake (a treasure instead,
        rendered and collected through the CheckTreasures path, not here)

    """
    rb = room_byte.astype(jnp.uint8)
    pit = pit_code_u8(rb)
    obj = obj_code_u8(rb)

    is_croc_room = pit == jnp.uint8(0b100)
    is_treasure_tar = pit == jnp.uint8(0b101)
    suppress = is_croc_room | is_treasure_tar

    # NuSize1Tab decides how many copies of the log the hardware draws, and the
    # NUSIZ value decides how far apart they sit. Both come from the object code,
    # and every copy hangs off the one xPosObject that ContRandom set to 124.
    counts = jnp.asarray(
        [len(NUSIZ1_COPY_OFFSETS[i]) for i in range(len(NUSIZ1_COPY_OFFSETS))]
        + [0] * (8 - len(NUSIZ1_COPY_OFFSETS)),
        dtype=jnp.uint8,
    )
    log_count_u8 = counts[obj.astype(jnp.int32)]

    has_logs = (log_count_u8 > jnp.uint8(0)) & (~suppress)
    # `cmp #ID_STATIONARY / bcs .skipLogs`: only object types 0..3 roll.
    logs_are_rolling = (obj < jnp.uint8(ID_STATIONARY)) & has_logs
    log_count = has_logs.astype(jnp.int32) * log_count_u8.astype(jnp.int32)

    offsets = jnp.asarray(
        [
            list(NUSIZ1_COPY_OFFSETS[i]) + [0] * (LOG_MAX_COPIES - len(NUSIZ1_COPY_OFFSETS[i]))
            if i < len(NUSIZ1_COPY_OFFSETS)
            else [0] * LOG_MAX_COPIES
            for i in range(8)
        ],
        dtype=jnp.int32,
    )
    log_xs_by_obj = jnp.mod(
        jnp.int32(ASM_OBJECT_START_X) + offsets[obj.astype(jnp.int32)],
        jnp.int32(ASM_SCREENWIDTH),
    )
    log_xs = jnp.where(has_logs, log_xs_by_obj, jnp.zeros((LOG_MAX_COPIES,), dtype=jnp.int32))

    has_fire = (obj == jnp.uint8(0b110)) & (~suppress)
    has_snake = (obj == jnp.uint8(0b111)) & (~suppress)

    return has_logs, logs_are_rolling, log_count, log_xs, has_fire, has_snake


def debug_room_byte(room_byte: int) -> str:
    """Small non-JAX helper for quick printing in scripts."""
    rb = room_byte & 0xFF
    pit = (rb >> 3) & 0x7
    obj = rb & 0x7
    wall = (rb >> 7) & 0x1
    return f"room_byte=0x{rb:02X} pit={pit} obj={obj} wall_side={wall}"


def lfsr_right_u8(b: jnp.ndarray) -> jnp.ndarray:
    """Moving right: shift left; bit0 = XOR(bit3, bit4, bit5, bit7)."""
    b = b.astype(jnp.uint8)
    bit = ((b >> 3) ^ (b >> 4) ^ (b >> 5) ^ (b >> 7)) & jnp.uint8(1)
    return jnp.uint8(((b << 1) & jnp.uint8(0xFF)) | bit)


def lfsr_left_u8(b: jnp.ndarray) -> jnp.ndarray:
    """Moving left: shift right; bit7 = XOR(bit4, bit5, bit6, bit0)."""
    b = b.astype(jnp.uint8)
    bit = ((b >> 4) ^ (b >> 5) ^ (b >> 6) ^ b) & jnp.uint8(1)  # b includes bit0
    return jnp.uint8((b >> 1) | (bit << 7))


def step_lfsr(b: jnp.ndarray, fn, n_steps: jnp.ndarray) -> jnp.ndarray:
    """Apply an LFSR step function n_steps times (n_steps is typically 1 or 3)."""

    def body(_, bb):
        return fn(bb)

    return lax.fori_loop(0, n_steps.astype(jnp.int32), body, b)

@struct.dataclass
class PitfallState:
    screen_id: chex.Array
    room_byte: chex.Array

    player_x: chex.Array
    player_y: chex.Array
    player_vx: chex.Array
    player_vy: chex.Array
    on_ground: chex.Array
    score: chex.Array
    timer_started: chex.Array

    time_left: chex.Array
    lives_left: chex.Array
    done: chex.Array
    hurt_cooldown: chex.Array

    down_pressed: chex.Array
    on_ladder: chex.Array
    current_ground_y: chex.Array 
    # ROM xPosScorpion: the left edge of the 8-pixel GRP1 box, held integral.
    scorpion_x: chex.Array
    scorpion_facing_right: chex.Array
    touching_wood: chex.Array
    touching_rolling_wood: chex.Array
    rolling_wood_contact_x: chex.Array
    climb_active: chex.Array
    # ROM climbPos: 0 off the ladder, else LADDER_TOP..LADDER_BOTTOM-1.
    climb_pos: chex.Array
    # ROM patIdHarry / frameCnt. One JAX step is two NTSC frames, so frame_cnt
    # advances once per step and the run cycle ticks when frame_cnt is odd.
    pat_id_harry: chex.Array
    frame_cnt: chex.Array
    facing_left: chex.Array
    # 0 while the game runs, 1 for the frozen KilledHarry sequence, 2 for the
    # restart drop. 1 and 2 are the two halves of `lda noGameScroll / beq
    # .processHarry`: in 1 Harry is skipped entirely, in 2 he is back under the
    # jump table with nothing but JumpTab's trailing -1 moving him.
    respawn_phase: chex.Array
    respawn_target_ground_y: chex.Array

    # ROM noGameScroll, soundIdx and soundDelay. The first freezes the game and
    # drives the copyright marquee, the other two time the death sequence.
    no_game_scroll: chex.Array
    sound_idx: chex.Array
    sound_delay: chex.Array
    # ROM yPosHarry, held only across the death sequence: frozen at the fatal
    # frame's value so the `cmp #71` branch can read it, then overwritten with
    # the restart value and counted up as the drop plays out.
    restart_rom_y: chex.Array

    # Jump-only state: used for edge-triggered input and fixed airborne carry.
    jump_pressed_prev: chex.Array
    jump_lock_active: chex.Array
    jump_lock_vx: chex.Array
    # ROM jumpIndex: 0 when no ground jump is running, else 1..JUMP_LEN.
    jump_index: chex.Array

    # The liana. lianaPosHi/lianaPosLo are the 16-bit swing oscillator; hmblAdd,
    # hmblDir and lianaBottom are the three bytes `.swingLiana` derives from it
    # every NTSC frame and the kernels then read. All five are plain ROM bytes,
    # written nowhere but `.swingLiana`, so they start from cleared RAM and are
    # never reset by a scene change or a restart.
    liana_pos_hi: chex.Array
    liana_pos_lo: chex.Array
    hmbl_add: chex.Array
    hmbl_dir: chex.Array
    liana_bottom: chex.Array
    # ROM atLiana (0 off the liana, 1 on it) and jumpMode. Reserved for the grab
    # and release slices; nothing reads either yet.
    at_liana: chex.Array
    jump_mode: chex.Array

    # ROM xPosQuickSand: the quicksand pit's border byte. Recomputed from
    # frameCnt every frame in a quicksand scene (5..7) while Harry is not
    # falling into the pit and the game is running, pinned to 0 in every other
    # scene, and frozen in place while Harry sinks (33 <= yPosHarry <= 54) or
    # the game is stopped. The rendered PF2 window and the collision bounds both
    # derive from this one byte, so they can never disagree.
    x_pos_quicksand: chex.Array

    # ROM treasureBits (4 bytes) and treasureCnt. Each of the 32 treasure rooms
    # is identified by its room byte's treePat (bits 6..7, picking the byte) and
    # objectType (bits 0..2, picking the bit via TreasureMask = 1 << (7 - obj)).
    # Cleared RAM at reset; a set bit means that treasure was already collected
    # and ProcessObjects draws Nothing in its place. treasureCnt starts at 31
    # (remaining - 1) and the 32nd collection ends the game (`dec noGameScroll`).
    treasure_bits: chex.Array  # (4,) uint8
    treasure_cnt: chex.Array

class PitfallConstants(struct.PyTreeNode):
    screen_width: int = 160     # Atari 2600 horizontal resolution
    screen_height: int = 210   # Atari vertical resolution used in ALE
    ground_y: int = 130         # approximate ground line in pixels
    # player_y of Harry's feet at UNDER_GROUND, on the one continuous scale:
    # ground_y + (86 - 32) = 184. It is a logical coordinate, not a raster row;
    # underground_floor_row turns it into one.
    underground_y: int = 184
    player_start_x: int = 20    # where Harry starts (left side)
    player_start_y: int = 130  # same as ground_y (standing on ground)

    # Horizontal run speed. NTSC Pitfall moves Harry 1px on even Atari frames
    # (30 px/s at 60 Hz). One JAX step is two Atari frames, so this is 1 px/step.
    player_speed: float = 1.0
    gravity: float = 0.55       # (global gravity for falls/ladder-exit; keep stable)
    fall_speed: float = 3.0    # terminal velocity cap on descent

    # The ground jump has no tuning constants: it replays JumpTab (see JUMP_TAB).

    fps: int = 30
    initial_time_seconds: int = 1200  # 20 minutes
    max_lives: int = 3          # Pitfall lives
    # Room-transition thresholds, from XMIN_HARRY / XMAX_HARRY in pitfall.asm.
    screen_exit_x_min: int = 8
    screen_exit_x_max: int = 148

    ladder_x: int = 80
    ladder_width: int = 16
    ladder_opening_inset: int = 4  # px from the ladder sprite's left edge to the opening
    initial_score: int = 2000
    tunnel_wall_width: int = 8

    # Side holes beside ladder (underground)
    hole_width: int = 12            # px (from ladder_with_pits sprite, cols 2-13 and 54-65)
    hole_gap_from_ladder: int = 12   # px floor bridge between ladder edge and hole

    # HoleBoundsTab is tested against xPosHarry (the box's left edge) while the
    # opening the player sees is centred on that box, so the drawn opening sits
    # this many columns right of the table. Confirmed by decoding PF2PatTab.
    hole_visible_offset_px: int = 4

    # Depth below the upper ground at which a fall switches from the standing
    # pose to Harry0's open-leg pose. ROM: 60 - JUNGLE_GROUND(32).
    hole_fall_open_leg_depth: float = 28.0

    # --- Fall-through bounds, ported verbatim from the original ROM ----------
    # HoleBoundsTab in pitfall.asm. Each row holds up to 4 (left, right) pairs
    # in xPosHarry coordinates; a left bound of 0 terminates the row.
    #
    #   .byte 72, 79,   0,  0,   0,  0,   0,  0    ; single hole
    #   .byte 44, 55,  72, 79,  96,107,   0,  0    ; triple hole
    #   .byte 44,107,   0,  0,   0,  0,   0,  0    ; pit
    #   .byte 44, 55,  64, 71,  80, 87,  96,107    ; closed croco jaws
    #   .byte 44, 61,  64, 77,  80, 93,  96,107    ; open croco jaws
    #
    # The ROM falls when  left < xPosHarry <= right  (see _hole_fall_test).
    # Rows 3 and 4 are kept for the crocodile work and are not selectable yet.
    hole_bounds_tab: tuple = (
        ((72, 79), (0, 0), (0, 0), (0, 0)),      # 0: single hole (plain ladder)
        ((44, 55), (72, 79), (96, 107), (0, 0)),  # 1: triple hole (ladder + pits)
        ((44, 107), (0, 0), (0, 0), (0, 0)),      # 2: pit
        ((44, 55), (64, 71), (80, 87), (96, 107)),  # 3: closed croco jaws
        ((44, 61), (64, 77), (80, 93), (96, 107)),  # 4: open croco jaws
    )

    # Stationary wood logs (upper ground hazard)
    # DecScoreLo subtracts one point on every NTSC frame whose CXPPMM latch is
    # set. A JAX step contains two such frames; the collision block counts them.
    wood_drain_per_frame: int = 1
    wood_w: int = 6               # log width in px (from log sprite)
    wood_h: int = 14              # log height in px (from log sprite)
    wood_y_offset: int = 0         # fine-tune vertical placement relative to ground
    wood_visual_contact_pad_x: int = 3  # start log interaction pose slightly before full overlap
    wood_visual_contact_shift_x: int = -3  # shift visual slide trigger slightly left

    # Rolling logs only: use a slightly larger early-contact zone so Harry is
    # forced into the stable freeze/contact pose and can't comfortably pace a
    # moving log in near lockstep. This does not affect stationary log scoring.
    rolling_wood_contact_pad_x: int = 6
    rolling_wood_contact_shift_x: int = -3

    # Fire hazard (upper ground). Geometry, cadence, position and the fatal
    # collision all come from FIRE_PATTERNS / FIRE_COLOR_ROWS and the ROM's
    # object pipeline; like the cobra, it kills through KilledHarry, so it needs
    # no fire-specific size, position, cooldown or respawn constants.

    # Ground object x (ASM: lda #124 / sta xPosObject). Left edge of GRP1.
    object_x: int = 124

    # Snake / cobra hazard (upper ground). Geometry comes from COBRA_PATTERNS;
    # the sprite files only supply the body colour.
    snake_w: int = 8
    snake_h: int = 16
    snake_hurt_cooldown_frames: int = 30

    # Scorpion hazard (underground). Geometry, cadence, animation and collision
    # all come from SCORPION_PATTERNS and the ROM's own rules; the only tunable
    # left is how long Harry is invulnerable after a hit.
    # ContRandom: `ldx #SCREENWIDTH/2-4 / stx xPosScorpion` on every scene entry.
    scorpion_spawn_x: int = 76
    scorpion_hurt_cooldown_frames: int = 30
    # `lda frameCnt / and #$07 / bne .endMoveScorpion` moves it one pixel every
    # eighth NTSC frame, which is one pixel every fourth JAX step.
    scorpion_move_period_steps: int = 4

    # `lda #20 / sta xPosHarry` is the restart column for both branches, so this
    # is the same 20 as player_start_x rather than a second tunable.
    underground_respawn_x: int = 20
    underground_respawn_reveal_from_ground: int = 17  # reveal boundary is this many px below ground
    underground_respawn_reveal_y_offset: int = 1  # clip reveal just below the upper ledge

    # Visual datum: player_y is Harry's feet, and harry_box_top_row is the only
    # place it reaches, so the drawing and the collision masks move together.
    # One offset for the whole screen, because yPosHarry is one scanline counter.
    harry_y_tune: int = HARRY_FEET_OFFSET
    ASSET_CONFIG: tuple = _get_default_pitfall_asset_config()


# ROM Harry pattern ids (HarryPtrTab / ID_* in pitfall.asm).
ID_KNEEING = 0
ID_RUNNING4 = 4
ID_STANDING = 5
ID_SWINGING = 6
ID_CLIMBING = 7


# Ground-jump vertical profile, verbatim from JUMP_LEN / JumpTab in pitfall.asm.
# The ROM starts a jump with `lda #1 / sta jumpIndex / dec yPosHarry`, then once
# per NTSC frame runs `yPosHarry -= JumpTab[jumpIndex-1] / inc jumpIndex`, with
# jumpIndex clamped at JUMP_LEN so the trailing -1 repeats until Harry is back
# on the floor and `.stopJump` clears the index.
JUMP_LEN = 32
JUMP_TAB = (
    1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1,
    -1, 0, 0, 0, -1, 0, 0, -1, 0, -1, -1, -1, -1, -1, -1, -1,
)
_JUMP_TAB = jnp.asarray(JUMP_TAB, dtype=jnp.float32)


def jump_table_subframe(y, jump_index, ground_y):
    """Advance one Atari frame of the ROM ground jump.

    `jump_index` is the ROM's jumpIndex: 0 when no jump is running, 1..32 while
    one is. Returns the new y, the new index, and whether this subframe is the
    one that put Harry back on `ground_y`.

    An inactive jump passes straight through, which is what makes the second
    subframe of a JAX step a no-op once the first one has already landed.
    """
    running = jump_index > jnp.int32(0)
    delta = _JUMP_TAB[jnp.clip(jump_index - jnp.int32(1), 0, JUMP_LEN - 1)]
    y_next = y - delta.astype(y.dtype)
    landed = y_next >= ground_y
    y_next = jnp.where(landed, ground_y, y_next)
    index_next = jnp.where(
        landed,
        jnp.int32(0),
        jnp.minimum(jump_index + jnp.int32(1), jnp.int32(JUMP_LEN)),
    )
    return (
        jnp.where(running, y_next, y),
        jnp.where(running, index_next, jump_index),
        running & landed,
    )


# --- The liana ---------------------------------------------------------------
# LianaTab holds the byte ProcessObjects writes straight into ENABL:
#     ldy sceneType / lda LianaTab,y / sta ENABL
# so the two entries are the register's own values, not a flag of this port's.
LIANA_DISABLE = 0b00
LIANA_ENABLE = 0b10

# LianaTab, indexed by sceneType, verbatim from pitfall.asm:
#     .byte DISABLE, DISABLE, ENABLE, ENABLE, ENABLE, DISABLE, ENABLE, DISABLE
LIANA_TAB = (
    LIANA_DISABLE,  # 0
    LIANA_DISABLE,  # 1
    LIANA_ENABLE,   # 2
    LIANA_ENABLE,   # 3
    LIANA_ENABLE,   # 4
    LIANA_DISABLE,  # 5
    LIANA_ENABLE,   # 6
    LIANA_DISABLE,  # 7
)
_LIANA_TAB = jnp.asarray(LIANA_TAB, dtype=jnp.uint8)

# `ldy #$f0` / `ldy #$10`: the HMBL nibble the kernels write on the scanlines
# where the liana accumulator carries. $f0 moves the ball one pixel right, $10
# one pixel left.
HMBL_DIR_RIGHT = 0xF0
HMBL_DIR_LEFT = 0x10


def liana_swing_frame(pos_hi, pos_lo):
    """One NTSC frame of `.swingLiana`, byte for byte.

        lda    lianaPosLo
        asl
        lda    lianaPosHi
        rol
        bpl    .skipNeg
        eor    #$ff
    .skipNeg:
        sta    hmblAdd
        ldy    #$f0
        lda    lianaPosHi
        bmi    .skipMoveLeft
        ldy    #$10
    .skipMoveLeft:
        sty    hmblDir
        sec
        lda    #143
        sbc    hmblAdd
        clc
        adc    lianaPosLo
        sta    lianaPosLo
        bcc    .skipAddHi
        lda    lianaPosHi
        adc    #3
        sta    lianaPosHi
    .skipAddHi:
        lda    hmblAdd
        lsr
        lsr
        lsr
        cmp    #6-1
        bcs    .limitBottom
        lda    #6
    .limitBottom:
        adc    #4
        sta    lianaBottom

    Both shifts are accumulator shifts, not read-modify-write on the two bytes,
    so the doubling exists only to make hmblAdd: lianaPosHi and lianaPosLo keep
    their own values until the `adc` pair below writes them.

    Returns (pos_hi, pos_lo, hmbl_add, hmbl_dir, liana_bottom), all uint8.
    """
    hi = pos_hi.astype(jnp.int32) & jnp.int32(0xFF)
    lo = pos_lo.astype(jnp.int32) & jnp.int32(0xFF)

    # `asl` leaves lianaPosLo's bit 7 in the carry and `rol` shifts it into A,
    # so this is the top byte of the 16-bit position doubled.
    doubled = ((hi << jnp.int32(1)) | (lo >> jnp.int32(7))) & jnp.int32(0xFF)
    # `eor #$ff` is a one's complement, so $80 folds to $7f, not to $80. That
    # caps hmblAdd at 127 and is why the subtract below can never borrow.
    hmbl_add = jnp.where(doubled >= jnp.int32(0x80), doubled ^ jnp.int32(0xFF), doubled)

    # `bmi` tests lianaPosHi itself, one bit to the left of the sign the fold
    # above used, so the direction flips half a swing away from the angle's fold.
    hmbl_dir = jnp.where(
        hi >= jnp.int32(0x80), jnp.int32(HMBL_DIR_RIGHT), jnp.int32(HMBL_DIR_LEFT)
    )

    # `clc` discards the carry the subtract produced, so the only carry that
    # survives is this addition's.
    total = (jnp.int32(143) - hmbl_add) + lo
    lo_next = total & jnp.int32(0xFF)
    carry = total > jnp.int32(0xFF)
    # `adc #3` is reached only when that carry is set, and it is still set when
    # the instruction runs, so the high byte gains four.
    hi_next = jnp.where(carry, (hi + jnp.int32(4)) & jnp.int32(0xFF), hi)

    # `cmp #6-1` leaves the carry set from 5 upwards, and `lda #6` on the other
    # path keeps it clear, so `adc #4` is either q+5 or a flat 10.
    q = hmbl_add >> jnp.int32(3)
    liana_bottom = jnp.where(q >= jnp.int32(5), q + jnp.int32(5), jnp.int32(10))

    return (
        hi_next.astype(jnp.uint8),
        lo_next.astype(jnp.uint8),
        hmbl_add.astype(jnp.uint8),
        hmbl_dir.astype(jnp.uint8),
        liana_bottom.astype(jnp.uint8),
    )


# --- The liana raster, generated across Kernels 1-4 --------------------------
# The liana is the TIA ball. Nothing gives it an absolute column per frame: the
# kernels only nudge it, one pixel at a time, through HMBL and HMOVE. Every
# nudge is driven by the same accumulator, `hmblSum`, which the vertical blank
# leaves at zero (`stx hmblSum` with x counting down to 0):
#
#     clc / lda hmblSum / adc hmblAdd / sta hmblSum
#     bcc .noMoveN / lda hmblDir
#     sta HMBL ... sta WSYNC / sta HMOVE
#
# so accumulator step i moves the ball on the scanline that step i's HMOVE
# starts, and the displacement after i steps is just the number of carries,
# floor(i * hmblAdd / 256). There are 83 accumulator updates spread over an
# 84-line band - the band's first line shows the ball still on its anchor,
# because update 1 runs in that line's own body and only lands with the HMOVE
# that opens the second line:
#
#   Kernel 1                      31   .loopBranches, y 31 -> 1
#   prepare Kernel 2               1   `sty HMBL` at .noMove1
#   Kernel 2                       4   .loopLianaPos, two updates per iteration
#   the two `jsr DrawLiana` lines  2
#   Kernel 3                      21   .loopLianaHarry, x 20 -> 0
#   Kernel 4                      24   .loopEndLiana, x 23 -> 0
#                                 --
#                                 83 updates, landing on band lines 2..84
#
# The band's last line is the one before Kernel 5's first. Kernel 5 opens on
# row 117, not 118: the ground artwork's first bright row is raster row 117
# (backdrop asset, row 116 is still trunk-and-sky), and the ROM run in ALE puts
# the ball's last enabled scanline at row 116 - lianaBottom, which is only
# consistent with the band ending on 116. So the top line is 117 - 84 = 33.
# ALE confirms the whole mapping directly: with hmblAdd 89 / hmblDir $f0 the
# real cart draws the rope at column 78 + floor((row-33)*89/256) on every
# visible row, an exact fit with zero error.
LIANA_TOP_ROW = 33
LIANA_ROWS = 84

# `sta RESBL` only ever runs in ContKernel, below the liana, so the ball enters
# every Kernel 1 at the same column and the anchor is a constant. ContKernel is
# entered at cycle 6 of its line (`sta HMOVE / jmp ContKernel` after the
# preceding WSYNC) and runs COLUBK, COLUPF, GRP1, PF1, PF2, CTRLPF, temp1 and
# the `lda #$90 / sta.w HMBL` pair before `sta RESBL` finishes on cycle 45 =
# colour clock 135, so the ball's counter latches at screen column 135-68+4 =
# 71, and the HMOVE that opens the next line applies the $90 nibble, -7,
# moving the ball seven pixels right: 71 + 7 = 78. Kernel 7's `sta HMCLR`
# clears the nibble before another HMOVE can repeat it, and nothing else
# touches the ball's position before the next frame's Kernel 1, so 78 is the
# anchor. Measured on the real cart in ALE the fit is exact: at hmblAdd 89
# the rope runs column 92 at row 74 to column 101 at row 100, which is
# 78 + floor((row-33)*89/256) with zero error on every row.
#
# `.skipSwingHarry` puts an attached Harry at `75 +/- (hmblAdd >> 2)` - three
# columns to the left of the anchor. That offset is the ROM's own: Harry6's
# raised hand occupies box columns 0-1, so the +3 puts the ball on the hand
# (the ball sits at box column 1..3 across the phases, touching the hand or
# one pixel from it, exactly as the original raster draws it).
LIANA_ANCHOR_X = 78

# `.skipSwingHarry`'s own literal: `adc #75` (both signs share it). Harry's
# attached column comes from here, not from the ball anchor above.
LIANA_HARRY_X = 75

# `lda #%101 / sta CTRLPF` and `lda #%001 / sta CTRLPF` both leave the ball-size
# field (bits 4-5) clear, so the rope is one colour clock wide the whole way down.
LIANA_W = 1

# The ball takes COLUPF. `.loopColors` leaves it at ColorTab+2, the leaves
# colour, so the rope is green for the whole of Kernel 1 and the prepare line
# (band lines 1..32 = rows 33..64). The line that opens Kernel 2 loads
# ColorTab+5, the branch and log colour (`lda colorLst+5 / sta COLUPF`), which
# then holds until Kernel 5: brown from row 65 down. ALE confirms both: the
# real cart's rope pixels are $D2 dark green on rows up to 64 and $10 brown
# from row 65. In Kernel 1 CTRLPF is %101 - playfield priority on - so the
# leaves cover the green rope, and the branches (players) sit above the ball
# as well; the green section only shows where the scanline is bare background.
LIANA_BROWN_TOP_ROW = 65

# CrocoTab is 1 for sceneType 4 alone, and that branch of ProcessObjects ends
# with `lda objectType / sta ENABL`, overwriting the LianaTab byte. So a croco
# scene carries a liana only when objectType has bit 1 set.
CROCO_SCENE = 4

_LIANA_ROW_STEPS = jnp.arange(LIANA_ROWS, dtype=jnp.int32)


def liana_column(step, hmbl_add, hmbl_dir):
    """Ball column `step` accumulator steps into the band.

    floor(step * hmblAdd / 256) is the carry count, and hmblDir decides which way
    each carry moves it: $f0 is one pixel right, $10 one pixel left.
    """
    displacement = (
        jnp.maximum(step, jnp.int32(0)) * hmbl_add.astype(jnp.int32)
    ) >> jnp.int32(8)
    moving_right = hmbl_dir.astype(jnp.int32) == jnp.int32(HMBL_DIR_RIGHT)
    return jnp.int32(LIANA_ANCHOR_X) + jnp.where(
        moving_right, displacement, -displacement
    )


def liana_last_row(liana_bottom):
    """Last row the ball is still enabled on.

    Kernel 4 counts x from 23 down to 0, its 24 HMOVEs opening rows 93..116, and
    runs `cpx lianaBottom / bcs .skipDisable / sta ENABL` in each row's blank, so
    the write lands before the visible part of the row it belongs to and the last
    row that still shows the ball is 116 - lianaBottom (ALE: lianaBottom 10, 12,
    13, 16 and 17 end the drawn rope on rows 106, 104, 103, 100 and 99).
    """
    return jnp.int32(LIANA_TOP_ROW + LIANA_ROWS - 1) - liana_bottom.astype(jnp.int32)


def liana_enabled(room_byte):
    """ENABL for this scene: LianaTab, then the croco scene's override."""
    scene = pit_code_u8(room_byte).astype(jnp.int32)
    from_tab = _LIANA_TAB[scene] != jnp.uint8(0)
    object_bit = (obj_code_u8(room_byte).astype(jnp.int32) & jnp.int32(0x02)) != 0
    return jnp.where(scene == jnp.int32(CROCO_SCENE), object_bit, from_tab)


# Harry occupies a fixed 8-column x 22-row hardware player box: GRP0 is eight
# bits wide, NUSIZ0 is ONE_COPY so one bit is one screen pixel, and HARRY_H is
# the kernel's row budget.
HARRY_W = 8
HARRY_H = 22

# Harry0..Harry8, transcribed from pitfall.asm. Each tuple keeps the ROM's own
# byte order, so index 0 is the BOTTOM row of the pose: the kernel reads the
# pattern with `dey / cpy #HARRY_H / lda (harryPatPtr),y`, counting the pointer
# down as the raster counts down the screen. NumPy images are top-down, which
# is why harry_pattern_bitmap reverses the tuple.
#
# Harry8 is only 21 bytes in the ROM; its 22nd row is BranchTab's first byte,
# which the assembler places immediately after it. That byte is included here.
HARRY_PATTERNS = (
    (   # Harry0 - airborne / kneeing
        0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00110011,
        0b01110010, 0b11011010, 0b00011110, 0b00011100, 0b00011000, 0b01011000,
        0b01011000, 0b01111100, 0b00111110, 0b00011010, 0b00011000, 0b00010000,
        0b00011000, 0b00011000, 0b00011000, 0b00000000,
    ),
    (   # Harry1 - running
        0b00000000, 0b10000000, 0b10000000, 0b11000011, 0b01100010, 0b01100010,
        0b00110110, 0b00111110, 0b00011100, 0b00011000, 0b00011000, 0b00111100,
        0b00111110, 0b00111010, 0b00111000, 0b00011000, 0b00011000, 0b00010000,
        0b00011000, 0b00011000, 0b00011000, 0b00000000,
    ),
    (   # Harry2 - running
        0b00010000, 0b00100000, 0b00100010, 0b00100100, 0b00110100, 0b00110010,
        0b00010110, 0b00011110, 0b00011100, 0b00011000, 0b00011000, 0b00011100,
        0b00011100, 0b00011000, 0b00011000, 0b00011000, 0b00011000, 0b00010000,
        0b00011000, 0b00011000, 0b00011000, 0b00000000,
    ),
    (   # Harry3 - running / one pixel above the jungle floor
        0b00001100, 0b00001000, 0b00101000, 0b00101000, 0b00111110, 0b00001010,
        0b00001110, 0b00011100, 0b00011000, 0b00011000, 0b00011100, 0b00011100,
        0b00011000, 0b00011000, 0b00011000, 0b00011000, 0b00011000, 0b00010000,
        0b00011000, 0b00011000, 0b00011000, 0b00000000,
    ),
    (   # Harry4 - running
        0b00000000, 0b00000010, 0b01000011, 0b01000100, 0b01110100, 0b00010100,
        0b00011100, 0b00011100, 0b00011000, 0b00011000, 0b00011000, 0b00111100,
        0b00111110, 0b00111010, 0b00111000, 0b00011000, 0b00011000, 0b00010000,
        0b00011000, 0b00011000, 0b00011000, 0b00000000,
    ),
    (   # Harry5 - standing
        0b00011000, 0b00010000, 0b00011100, 0b00011000, 0b00011000, 0b00011000,
        0b00011000, 0b00011000, 0b00011000, 0b00011000, 0b00011000, 0b00011000,
        0b00011100, 0b00011110, 0b00011010, 0b00011000, 0b00011000, 0b00010000,
        0b00011000, 0b00011000, 0b00011000, 0b00000000,
    ),
    (   # Harry6 - swinging on the liana
        0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00000000,
        0b01100011, 0b11110010, 0b11110110, 0b11011100, 0b11000000, 0b11000000,
        0b11000000, 0b11000000, 0b11000000, 0b11110000, 0b11010000, 0b10010000,
        0b11010000, 0b11010000, 0b11000000, 0b00000000,
    ),
    (   # Harry7 - climbing
        0b00110000, 0b00010000, 0b00010000, 0b00010000, 0b00010110, 0b00010100,
        0b00010100, 0b00010110, 0b00010010, 0b00010110, 0b00011110, 0b00011100,
        0b00011000, 0b00111000, 0b00111000, 0b00111100, 0b00011110, 0b00011010,
        0b00000010, 0b00011000, 0b00011000, 0b00011000,
    ),
    (   # Harry8 - climbing (last byte is BranchTab[0])
        0b00001100, 0b00001000, 0b00001000, 0b00001000, 0b01101000, 0b00101000,
        0b00101000, 0b01101000, 0b01001000, 0b01101000, 0b01111000, 0b00111000,
        0b00011000, 0b00011100, 0b00011100, 0b00111100, 0b01111000, 0b01011000,
        0b01000000, 0b00011000, 0b00011000, 0b00011000,
    ),
)


def harry_pattern_bitmap(pattern_index: int) -> np.ndarray:
    """One ROM pattern as a top-down HARRY_H x HARRY_W boolean bitmap."""
    rows = HARRY_PATTERNS[pattern_index][::-1]
    return np.array(
        [[bool((row >> (HARRY_W - 1 - col)) & 1) for col in range(HARRY_W)] for row in rows],
        dtype=bool,
    )


def harry_pattern_bounds(pattern_index: int) -> tuple[int, int, int, int]:
    """(first_row, last_row, first_col, last_col) of a pose inside its box.

    These are what give each pose its ROM padding: Harry0 for instance ends at
    row 16, five rows above the baseline, because its jumping legs are drawn
    tucked up.
    """
    bitmap = harry_pattern_bitmap(pattern_index)
    rows = np.flatnonzero(bitmap.any(axis=1))
    cols = np.flatnonzero(bitmap.any(axis=0))
    return int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])


def harry_pattern_collision_box(pattern_index: int, reflected: bool = False) -> tuple[int, int, int, int]:
    """Half-open [x0, x1) x [y0, y1) bounds of a pose's lit pixels in its box.

    Coordinates are local to the 8x22 player box: column 0 is the box's left
    edge and row 0 its top row. REFP0 maps column c to HARRY_W-1-c without
    moving the box, so reflecting a span [c0, c1] gives [W-1-c1, W-1-c0] and
    leaves the rows alone.
    """
    row0, row1, col0, col1 = harry_pattern_bounds(pattern_index)
    if reflected:
        col0, col1 = HARRY_W - 1 - col1, HARRY_W - 1 - col0
    return col0, col1 + 1, row0, row1 + 1


# [reflected][pattern] -> (x0, x1, y0, y1), all local to the player box.
HARRY_COLLISION_BOXES = tuple(
    tuple(harry_pattern_collision_box(i, reflected) for i in range(len(HARRY_PATTERNS)))
    for reflected in (False, True)
)
_HARRY_COLLISION_BOXES = jnp.asarray(HARRY_COLLISION_BOXES, dtype=jnp.int32)

# [reflected][pattern] -> the HARRY_H x HARRY_W lit-pixel mask itself.
_HARRY_BITMAPS = jnp.asarray(
    np.stack(
        [
            np.stack(
                [
                    harry_pattern_bitmap(i)[:, ::-1] if r else harry_pattern_bitmap(i)
                    for i in range(len(HARRY_PATTERNS))
                ]
            )
            for r in (False, True)
        ]
    )
)


def harry_box_top_row(player_y, current_ground_y, consts):
    """Screen row of Harry's box top row.

    The single place that maps player_y onto the raster. The renderer and the
    collision masks both go through it, so the pixels a player sees are exactly
    the pixels that can be hit.

    One offset, HARRY_FEET_OFFSET, for the whole screen: `feet_row = K + yPosHarry`
    is what the kernels do, so a band change must not move Harry by so much as a
    row. current_ground_y is accepted and discarded on purpose - branching on it
    is what used to teleport him twelve rows on the frame he reached the
    underground floor.
    """
    del current_ground_y
    return player_y.astype(jnp.int32) - jnp.int32(HARRY_H - 1) + jnp.int32(
        int(consts.harry_y_tune)
    )


def harry_collision_bounds(player_x, box_top, pattern_index, reflected):
    """Harry's lit pixels in world space, as half-open [x0, x1) x [y0, y1).

    player_x is the ROM's xPosHarry, the left edge of the player box, so a box
    column c sits at player_x + c. `box_top` comes from harry_box_top_row.
    """
    box = _HARRY_COLLISION_BOXES[reflected.astype(jnp.int32), pattern_index]
    left = player_x.astype(jnp.int32)
    top = box_top.astype(jnp.int32)
    return left + box[0], left + box[1], top + box[2], top + box[3]


def liana_hit_harry(
    hmbl_add, hmbl_dir, liana_bottom, enabled, harry_x, box_top, pattern_index, reflected
):
    """hitLiana - CXP0FB bit 6 - for one rendered raster.

        jsr    DrawHarry
        ldx    CXP0FB-$30
        stx    hitLiana

    The latch is the ball against GRP0, so this walks Harry's own box: for each of
    its 22 rows it generates that scanline's ball column from the same accumulator
    the renderer uses and asks whether the pose has a lit pixel there. No radius,
    no rectangle - one ball pixel against one sprite pixel.

    `sta CXCLR` runs on the line that opens Kernel 3, so the ROM only counts rows
    71 down. That bound never bites: Harry's box top is 71 + yPosHarry, so it is
    never above row 71, and the ball is already disabled below row 106.
    """
    bitmap = _HARRY_BITMAPS[reflected.astype(jnp.int32), pattern_index]
    rows = box_top.astype(jnp.int32) + jnp.arange(HARRY_H, dtype=jnp.int32)
    step = rows - jnp.int32(LIANA_TOP_ROW)
    column = liana_column(step, hmbl_add, hmbl_dir)
    lit_rope = (step >= jnp.int32(0)) & (rows <= liana_last_row(liana_bottom))
    box_column = column - harry_x.astype(jnp.int32)
    inside_box = (box_column >= jnp.int32(0)) & (box_column < jnp.int32(HARRY_W))
    lit_harry = bitmap[
        jnp.arange(HARRY_H, dtype=jnp.int32), jnp.clip(box_column, 0, HARRY_W - 1)
    ]
    return enabled & jnp.any(lit_rope & inside_box & lit_harry)


def player_object_pixels_collide(harry_bitmap, harry_x, harry_top, object_bitmap, object_x, object_top):
    """CXPPMM: true when a lit pixel of each object shares a screen coordinate.

    Walks Harry's fixed-size box and samples the object's box at the matching
    world position, so the shapes stay static for jit and vmap. Anything that
    falls outside the object's box is masked off rather than wrapped.
    """
    dy = harry_top.astype(jnp.int32) - object_top.astype(jnp.int32)
    dx = harry_x.astype(jnp.int32) - object_x.astype(jnp.int32)
    object_h, object_w = object_bitmap.shape[-2], object_bitmap.shape[-1]

    rows = jnp.arange(harry_bitmap.shape[-2], dtype=jnp.int32)[:, None] + dy
    cols = jnp.arange(harry_bitmap.shape[-1], dtype=jnp.int32)[None, :] + dx
    inside = (rows >= 0) & (rows < object_h) & (cols >= 0) & (cols < object_w)
    sampled = object_bitmap[
        jnp.clip(rows, 0, object_h - 1), jnp.clip(cols, 0, object_w - 1)
    ]
    return jnp.any(harry_bitmap & sampled & inside)


def harry_display_facing_left(state) -> chex.Array:
    """The facing the renderer will use, so collision can use the same one."""
    moving = jnp.abs(state.player_vx) > jnp.asarray(0.0, dtype=jnp.float32)
    return jnp.where(moving, state.player_vx < 0.0, state.facing_left)


# The scorpion is a GRP1 object drawn with NUSIZ1 = ONE_COPY in the underground
# kernel, so it is eight screen pixels wide, inside the 16-row underground
# object box. Like Harry's, each tuple is the ROM's byte order: index 0 is the
# bottom row. ScorpionColor is WHITE for all eleven of its rows, so the sprite
# carries no colour information beyond that.
SCORPION_W = 8
SCORPION_H = 16
SCORPION_PATTERNS = (
    (   # Scorpion0 - 11 occupied rows
        0b10000101, 0b00110010, 0b00111101, 0b01111000, 0b11111000, 0b11000110,
        0b10000010, 0b10010000, 0b10001000, 0b11011000, 0b01110000, 0b00000000,
        0b00000000, 0b00000000, 0b00000000, 0b00000000,
    ),
    (   # Scorpion1 - 10 occupied rows
        0b01001001, 0b00110011, 0b00111100, 0b01111000, 0b11111010, 0b11000100,
        0b10010010, 0b10001000, 0b11011000, 0b01110000, 0b00000000, 0b00000000,
        0b00000000, 0b00000000, 0b00000000, 0b00000000,
    ),
)


def scorpion_pattern_bitmap(pattern_index: int, reflected: bool = False) -> np.ndarray:
    """One ROM pattern as a top-down SCORPION_H x SCORPION_W boolean bitmap."""
    rows = SCORPION_PATTERNS[pattern_index][::-1]
    bitmap = np.array(
        [[bool((row >> (SCORPION_W - 1 - col)) & 1) for col in range(SCORPION_W)] for row in rows],
        dtype=bool,
    )
    return bitmap[:, ::-1] if reflected else bitmap


_SCORPION_BITMAPS = jnp.asarray(
    np.stack(
        [
            np.stack([scorpion_pattern_bitmap(i, r) for i in range(len(SCORPION_PATTERNS))])
            for r in (False, True)
        ]
    )
)


def scorpion_box_top_row(underground_y: int) -> int:
    """Screen row of the object box's top row.

    Kernel 9 walks `(undrPatPtr),y` from 15 down to 0 as the raster descends, so
    pattern index 0 - the bottom row - lands on the underground floor, the same
    row Harry's feet rest on. That is a raster row, so it comes from
    underground_floor_row rather than from the logical coordinate.
    """
    return underground_floor_row(underground_y) - (SCORPION_H - 1)


# The cobra is a GRP1 object drawn with NUSIZ1 = ONE_COPY, so it is eight
# screen pixels wide inside the 16-row ground-object box. Cobra0/Cobra1 are
# listed bottom row first, matching Harry and the scorpion. CobraColor notes
# that only 14 of the 16 rows are occupied.
COBRA_W = 8
COBRA_H = 16
COBRA_PATTERNS = (
    (   # Cobra0
        0b00000000, 0b11111110, 0b11111001, 0b11111001, 0b11111001, 0b11111001,
        0b01100000, 0b00010000, 0b00001000, 0b00001100, 0b00001100, 0b00001000,
        0b00111000, 0b00110000, 0b01000000, 0b00000000,
    ),
    (   # Cobra1
        0b00000000, 0b11111110, 0b11111001, 0b11111001, 0b11111010, 0b11111010,
        0b01100000, 0b00010000, 0b00001000, 0b00001100, 0b00001100, 0b00001000,
        0b00111000, 0b00110000, 0b10000000, 0b00000000,
    ),
)


def cobra_pattern_bitmap(pattern_index: int) -> np.ndarray:
    """One ROM pattern as a top-down COBRA_H x COBRA_W boolean bitmap."""
    rows = COBRA_PATTERNS[pattern_index][::-1]
    return np.array(
        [[bool((row >> (COBRA_W - 1 - col)) & 1) for col in range(COBRA_W)] for row in rows],
        dtype=bool,
    )


_COBRA_BITMAPS = jnp.asarray(
    np.stack([cobra_pattern_bitmap(i) for i in range(len(COBRA_PATTERNS))])
)


def cobra_box_top_row(ground_y: int) -> int:
    """Screen row of the 16-row cobra box's top.

    The existing 8x14 draw put the last occupied row on `ground_y` via
    `ground_y - 14 + 1`. The ROM box adds one empty pattern row above and
    below that band, so the 16-row origin is one row above the old crop.
    """
    return int(ground_y) - (COBRA_H - 2)


def cobra_animation_frame(time_left, timer_started, consts, anim_bit4):
    """Incoming `random2 & OBJECT_H` frame: 0 for Cobra0, 1 for Cobra1.

    The renderer and collision both go through this so CXPPMM sees the same
    bits that were on screen. `time_left` must be the incoming state's value;
    decrementing it first would sample the next LFSR step.
    """
    total = jnp.int32(int(consts.initial_time_seconds) * int(consts.fps))
    elapsed = jnp.maximum(total - time_left.astype(jnp.int32), jnp.int32(0))
    elapsed = elapsed * timer_started.astype(jnp.int32)
    return anim_bit4[jnp.mod(elapsed, jnp.int32(anim_bit4.shape[0]))]


def is_falling_through_hole(state, consts: PitfallConstants) -> chex.Array:
    """True while Harry is in the forced descent down through an opening.

    Mirrors the ROM, which picks the pattern from height alone: ID_STANDING when
    yPosHarry is between JUNGLE_GROUND and 60 and not climbing. Here:
    current_ground_y rules out an underground jump (the ROM's `cmp #60`),
    player_y > upper rules out a jump over a hole (its arc stays above the
    line), and vy >= 0 rules out the rising half of a ladder-exit hop.
    Deliberately not `over_hole`: being above an opening is not a descent.
    """
    upper_ground = jnp.asarray(consts.ground_y, dtype=jnp.float32)
    return (
        (~state.on_ground)
        & (~state.on_ladder)
        & (state.current_ground_y == upper_ground)
        & (state.player_y > upper_ground)
        & (state.player_vy >= jnp.asarray(0.0, dtype=jnp.float32))
    )


def harry_display_pat_id(state, consts: PitfallConstants, touching_wood=None) -> chex.Array:
    """ROM patIdHarry display selection from yPosHarry, not a blanket airborne rule.

    After the run cycle, the ROM does:

      y == 31              -> Harry3   (one pixel above jungle ground)
      y == 32 or y == 86   -> keep the ground run/stand id
      y < 32               -> Harry0
      y >= 60              -> Harry0
      32 < y < 60, no climb -> Harry5

    Our player_y is feet, ground_y is jungle floor. rel = player_y - ground_y
    maps onto (yPosHarry - JUNGLE_GROUND), so y==31 is rel in [-1, 0) and
    y>=60 is rel >= 28. The 32<y<60 band is gated by is_falling_through_hole
    so a rising ladder-exit hop is not treated as a hole fall.
    """
    hit_log = state.touching_wood | state.touching_rolling_wood
    if touching_wood is not None:
        hit_log = touching_wood

    moving_h = jnp.abs(state.player_vx) > jnp.asarray(0.0, dtype=jnp.float32)
    run_pat = state.pat_id_harry.astype(jnp.int32)
    ground_pat = jnp.where(
        moving_h & (run_pat >= jnp.int32(ID_KNEEING)) & (run_pat <= jnp.int32(ID_RUNNING4)),
        run_pat,
        jnp.int32(ID_STANDING),
    )

    upper = jnp.asarray(consts.ground_y, dtype=jnp.float32)
    rel = state.player_y - upper
    open_leg_depth = jnp.asarray(consts.hole_fall_open_leg_depth, dtype=jnp.float32)
    at_lip = (
        (~state.on_ground)
        & (~state.on_ladder)
        & (state.current_ground_y == upper)
        & (rel < jnp.asarray(0.0, dtype=jnp.float32))
        & (rel >= jnp.asarray(-1.0, dtype=jnp.float32))
    )
    falling = is_falling_through_hole(state, consts)

    airborne_pat = jnp.int32(ID_KNEEING)
    airborne_pat = jnp.where(at_lip, jnp.int32(3), airborne_pat)
    airborne_pat = jnp.where(
        falling & (rel < open_leg_depth),
        jnp.int32(ID_STANDING),
        airborne_pat,
    )

    pat = jnp.where(state.on_ground, ground_pat, airborne_pat)
    # `.contPatId`: `lda atLiana / beq .skipStanding2 / ldx #ID_SWINGING`. It comes
    # after the whole yPosHarry ladder above and before the climb and kneeing
    # overrides below, which is where it is put here. Neither of those can fire
    # while Harry hangs on the rope - climbPos is zero and the collision block
    # that writes patOfsHarry is the one atLiana skips.
    pat = jnp.where(state.at_liana != jnp.uint8(0), jnp.int32(ID_SWINGING), pat)
    # `lda climbPos / and #%1 / clc / adc #ID_CLIMBING`: the pose is the rung's
    # parity, not a timer, so Harry7 and Harry8 alternate with the climb itself.
    climb_bit = jnp.bitwise_and(state.climb_pos.astype(jnp.int32), jnp.int32(1))
    pat = jnp.where(state.on_ladder, jnp.int32(ID_CLIMBING) + climb_bit, pat)
    # `.hitLogs` only writes patOfsHarry on the non-ladder path. A ladder hit
    # increments climbPos and keeps the climbing pose.
    pat = jnp.where(hit_log & (~state.on_ladder), jnp.int32(ID_KNEEING), pat)
    return pat


@struct.dataclass
class PitfallObservation:
    player_x: chex.Array
    player_y: chex.Array
    screen_id: chex.Array
    room_byte: chex.Array
    current_ground_y: chex.Array
    on_ground: chex.Array
    on_ladder: chex.Array
    facing_left: chex.Array

    scorpion_x: chex.Array
    has_scorpion: chex.Array
    has_fire: chex.Array
    has_snake: chex.Array
    has_logs: chex.Array
    log_count: chex.Array
    log_xs: chex.Array
    logs_are_rolling: chex.Array

    has_ladder: chex.Array
    ladder_x: chex.Array
    has_wall: chex.Array
    wall_x: chex.Array
    wall_side: chex.Array

    time_left: chex.Array
    lives_left: chex.Array
    score: chex.Array

@struct.dataclass
class PitfallInfo:
    time_left: chex.Array
    lives_left: chex.Array

@struct.dataclass
class ScreenLayout:
    has_ladder: chex.Array
    ladder_x: chex.Array
    has_wall: chex.Array
    wall_x: chex.Array
    wall_side: chex.Array

class JaxPitfall(JaxEnvironment[PitfallState, PitfallObservation, PitfallInfo, PitfallConstants]):
    def __init__(self, consts: PitfallConstants | None = None):
        if consts is None:
            consts = PitfallConstants()
        super().__init__(consts)
        self.consts = consts
        self.num_screens = 255
        W = self.consts.screen_width
        WW = self.consts.tunnel_wall_width

        LEFT_WALL_X = 23
        RIGHT_WALL_X = 132

        def clamp_wall_x(x: int) -> int:
            return max(0, min(W - WW, x))

        self.left_wall_x_px = jnp.array(clamp_wall_x(LEFT_WALL_X), dtype=jnp.int32)
        self.right_wall_x_px = jnp.array(clamp_wall_x(RIGHT_WALL_X), dtype=jnp.int32)

        # Gameplay ladder reference, in xPosHarry coordinates. The ROM's climb
        # test is `lda xPosHarry / sec / sbc #68 / cmp #15 / bcs skip`, so 68 is
        # where the climbable band starts, and the snap is `lda #SCREENWIDTH/2-4`
        # = 76, the left edge of an 8px Harry centred on the screen.
        single_hole_left = int(consts.hole_bounds_tab[0][0][0])
        ladder_x_px = single_hole_left - int(consts.ladder_opening_inset)
        ladder_x_px = max(0, min(W - consts.ladder_width, ladder_x_px))
        self.ladder_x_px = jnp.array(ladder_x_px, dtype=jnp.int32)

        # Drawing reference. HoleBoundsTab is compared against xPosHarry, the
        # box's left edge, while the opening a player sees is centred on the
        # box, so decoding PF2PatTab puts the artwork four columns right of the
        # table: OneHole's gap is screen 76-83 against bounds (72, 79), and
        # ThreeHoles' are 48-59, 76-83 and 100-111 against (44,55), (72,79) and
        # (96,107). Both ladder captures were grabbed at those screen columns,
        # so this offset restores each sprite to its own capture origin.
        self.hole_render_offset_px = jnp.array(
            int(consts.hole_visible_offset_px), dtype=jnp.int32
        )
        self.ladder_render_x_px = self.ladder_x_px + self.hole_render_offset_px

        self.renderer = PitfallRenderer(
            consts=self.consts,
            ladder_x_px=self.ladder_render_x_px,
            left_wall_x_px=self.left_wall_x_px,
            right_wall_x_px=self.right_wall_x_px,
        )

        self.cobra_w_px = jnp.array(COBRA_W, dtype=jnp.int32)
        self.cobra_h_px = jnp.array(COBRA_H, dtype=jnp.int32)
        self.cobra_box_top_px = jnp.array(
            cobra_box_top_row(self.consts.ground_y), dtype=jnp.int32
        )

        # The scorpion's box bottom row is the underground floor, the same row
        # Harry's feet rest on, because kernel 9 walks the object pattern from
        # index 15 down to 0 as the raster descends.
        self.scorpion_box_top_px = jnp.array(
            scorpion_box_top_row(self.consts.underground_y), dtype=jnp.int32
        )
        self.wall_render_height_px = jnp.array(int(self.renderer.WALL_RENDER_MASK.shape[0]), dtype=jnp.int32)


    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key = None
    ) -> tuple[PitfallObservation, PitfallState]:
        state = self._init_state()
        obs = self._get_observation(state)
        return obs, state
    
    def _apply_ladder(
        self,
        state: PitfallState,
        room_byte: chex.Array,
        x: chex.Array,
        vx: chex.Array,
        y: chex.Array,
        vy: chex.Array,
        down_pressed: chex.Array,
        move_jump: chex.Array,
        move_left: chex.Array,
        move_right: chex.Array,
        on_ground: chex.Array,
        current_ground_y: chex.Array,
        log_hits=None,
        log_push_enabled=None,
    ) -> tuple[
        chex.Array,
        chex.Array,
        chex.Array,
        chex.Array,
        chex.Array,
        chex.Array,
        chex.Array,
        chex.Array,
        chex.Array,
        chex.Array,
        chex.Array,
    ]:
        """`.endStartClimb`, `.endClimbLadder` and `.notAtTop`, in ROM order.

        Two ordered NTSC frames per call. frameCnt is incremented once per JAX
        step, so this step covers frameCnt 2n-1 and 2n; `frameCnt & 7` therefore
        only comes up on the second of them, and `frameCnt & 3` likewise, which is
        what decides whether the walking block moves x before the exit fires.

        Returns:
            x, vx, y, vy, on_ground, on_ladder, current_ground_y, climb_active,
            exit_top_jump, climb_pos, ladder_log_frames
        """
        consts = self.consts
        layout = self._screen_layout(room_byte)
        has_ladder = layout.has_ladder

        upper_ground = jnp.asarray(consts.ground_y, dtype=jnp.float32)
        lower_ground = jnp.asarray(consts.underground_y, dtype=jnp.float32)
        speed = jnp.asarray(consts.player_speed, dtype=jnp.float32)

        # yPosHarry, from the one continuous mapping. This is the value every ROM
        # test below compares against, so the tests are the ROM's own.
        rom_y0 = (y - (upper_ground - jnp.float32(JUNGLE_GROUND))).astype(jnp.int32)

        climb_pos = state.climb_pos.astype(jnp.int32)
        frame_cnt = state.frame_cnt.astype(jnp.int32) + jnp.int32(1)

        climb_active = jnp.array(False, dtype=jnp.bool_)
        exit_top_jump = jnp.array(False, dtype=jnp.bool_)
        entered = jnp.array(False, dtype=jnp.bool_)
        rom_y = rom_y0
        pat_bottom_exit = jnp.array(False, dtype=jnp.bool_)
        # How many NTSC frames of this step ran `.hitLogs` with Harry on a rung.
        # Each one is one `inc climbPos` and one `jsr DecScoreLo`.
        ladder_log_frames = jnp.int32(0)
        push_ok = (
            jnp.array(True, dtype=jnp.bool_) if log_push_enabled is None else log_push_enabled
        )
        facing_left = harry_display_facing_left(state)

        for sub in range(NTSC_FRAMES_PER_STEP):
            # frameCnt for this NTSC frame: 2n-1 then 2n.
            ntsc_frame = frame_cnt * jnp.int32(NTSC_FRAMES_PER_STEP) - jnp.int32(
                NTSC_FRAMES_PER_STEP - 1 - sub
            )
            on_ladder_in = climb_pos != jnp.int32(0)

            # --- ladder entry, `.endStartClimb` -----------------------------
            #     lda climbPos / bne .endStartClimb
            #     lda ladderFlag / beq .endStartClimb
            #     lda xPosHarry / sec / sbc #68 / cmp #15 / bcs .endStartClimb
            # The subtract-and-compare is an unsigned byte test, so it passes for
            # exactly the fifteen columns 68..82.
            x_window = (x.astype(jnp.int32) - jnp.int32(LADDER_ENTRY_X_MIN)) & jnp.int32(0xFF)
            in_x_window = has_ladder & (x_window < jnp.int32(LADDER_ENTRY_X_SPAN))
            eligible = (~on_ladder_in) & in_x_window

            #     lda yPosHarry / cmp #84 / bcc .skipClimbUp
            #     lda joystick / lsr / bcs .skipClimbUp      (UP is bit 0)
            from_below = eligible & (rom_y >= jnp.int32(LADDER_ENTRY_FROM_BELOW_Y)) & move_jump
            #     lda yPosHarry / cmp #JUNGLE_GROUND / bne .endStartClimb
            #     lda joystick / and #%10 / bne .endStartClimb   (DOWN is bit 1)
            # Reached only when the branch above did not take, so UP near the
            # floor can never fall through into the jungle entry.
            from_above = (
                eligible
                & (~from_below)
                & (rom_y == jnp.int32(JUNGLE_GROUND))
                & down_pressed
            )
            entering = from_below | from_above
            climb_pos = jnp.where(
                entering,
                jnp.where(from_below, jnp.int32(LADDER_BOTTOM - 1), jnp.int32(LADDER_TOP + 1)),
                climb_pos,
            )
            x = jnp.where(entering, jnp.asarray(LADDER_SNAP_X, dtype=x.dtype), x)
            entered = entered | entering

            # --- `.hitLogs`, which the collision block reaches long before the
            # ladder block below:
            #
            #     .hitLogs:
            #         lda    climbPos         ; 3                 Harry at ladder?
            #         beq    .notAtLadder     ; 2³                 no, skip push
            #         inc    climbPos         ; 5                  yes, push down Harry
            #         bne    .decScore        ; 3
            #
            # Taking that branch skips `.notAtLadder` entirely, so patOfsHarry is
            # never written and the joystick is never forced to NO_MOVE: no
            # kneeling pose and no lost input while Harry is on a rung. This frame
            # gets its own CXPPMM read from its own rendered state, so a push that
            # already carried Harry's pixels clear cannot push him twice.
            if log_hits is not None:
                on_rung = climb_pos != jnp.int32(0)
                pose = jnp.int32(ID_CLIMBING) + (climb_pos & jnp.int32(1))
                box_top = harry_box_top_row(
                    rom_y_to_player_y(rom_y, consts), current_ground_y, consts
                )
                hit = log_hits(x, box_top, pose, facing_left)
                # `.contCollision` sends anything at yPosHarry 64 or below to
                # `.checkWallHit`, so only the rungs above that can be pushed.
                pushed = (
                    hit
                    & on_rung
                    & push_ok
                    & (rom_y < jnp.int32(64))
                )
                climb_pos = jnp.where(
                    pushed, (climb_pos + jnp.int32(1)) & jnp.int32(0xFF), climb_pos
                )
                rom_y = jnp.where(pushed, climb_pos_to_rom_y(climb_pos), rom_y)
                ladder_log_frames = ladder_log_frames + pushed.astype(jnp.int32)

            # --- climbing, `.endClimbLadder` --------------------------------
            on_ladder_now = climb_pos != jnp.int32(0)
            #     lda frameCnt / and #$07 / bne .skipAnimClimb
            climb_tick = on_ladder_now & (
                (ntsc_frame & jnp.int32(LADDER_CLIMB_MASK)) == jnp.int32(0)
            )
            #     lda joystick / lsr / bcs .notClimbUp / dec climbPos
            #     lsr / bcs .notClimbDown / inc climbPos
            stepped = climb_pos - move_jump.astype(jnp.int32) + down_pressed.astype(jnp.int32)
            stepped = jnp.where(climb_tick, stepped, climb_pos)

            #     cmp #LADDER_TOP / bcs .skipLadderTop
            #     lda #NO_MOVE / sta oldJoystick / lda #LADDER_TOP
            hit_top = climb_tick & (stepped < jnp.int32(LADDER_TOP))
            clamped = jnp.where(hit_top, jnp.int32(LADDER_TOP), stepped)
            #     cmp #LADDER_BOTTOM / bcc .skipLadderBottom
            #     lda #0 / ldx #ID_STANDING / stx patIdHarry
            #     ldx #SCREENWIDTH/2+6 / stx yPosHarry
            hit_bottom = climb_tick & (clamped >= jnp.int32(LADDER_BOTTOM))
            climb_pos = jnp.where(hit_bottom, jnp.int32(0), clamped)
            rom_y = jnp.where(hit_bottom, jnp.int32(LADDER_BOTTOM_EXIT_Y), rom_y)
            pat_bottom_exit = pat_bottom_exit | hit_bottom
            climb_active = climb_active | (climb_tick & (climb_pos != jnp.int32(0)))

            #     lda climbPos / beq .endClimbLadder
            #     asl / sec / rol / adc #1 / sta yPosHarry
            still_climbing = climb_pos != jnp.int32(0)
            rom_y = jnp.where(still_climbing, climb_pos_to_rom_y(climb_pos), rom_y)

            # --- the walking block, gated the way the ROM gates it ----------
            #     lda climbPos / cmp #LADDER_TOP+1 / bcs .endHarryId
            #     lda frameCnt / and #$03 / tax / lsr / bcs .endHarryId
            # At the top rung climbPos is 11, so this block is live, and on an
            # even frameCnt it moves xPosHarry one pixel *before* the exit below.
            walk_live = (
                (climb_pos < jnp.int32(LADDER_TOP + 1))
                & ((ntsc_frame & jnp.int32(1)) == jnp.int32(0))
                & still_climbing
            )
            walk_dx = jnp.where(move_left, -speed, jnp.where(move_right, speed, jnp.float32(0.0)))
            x = jnp.where(walk_live, x + walk_dx.astype(x.dtype), x)

            # --- the lateral exit, `.notAtTop` ------------------------------
            #     lda climbPos / cmp #LADDER_TOP / bne .notAtTop
            #     lda joystick / and #JOY_HORZ / cmp #JOY_HORZ / beq .notAtTop
            #     lda joystick / sta oldJoystick
            #     lda #1 / sta jumpIndex
            #     lsr / sta climbPos
            #     lda #31 / sta yPosHarry
            leaving = (climb_pos == jnp.int32(LADDER_TOP)) & (move_left | move_right)
            climb_pos = jnp.where(leaving, jnp.int32(0), climb_pos)
            rom_y = jnp.where(leaving, jnp.int32(LADDER_EXIT_Y), rom_y)
            exit_top_jump = exit_top_jump | leaving

        on_ladder = climb_pos != jnp.int32(0)
        touched_ladder = on_ladder | exit_top_jump | pat_bottom_exit | entered

        y = jnp.where(touched_ladder, rom_y_to_player_y(rom_y, consts), y)
        vy = jnp.where(touched_ladder, jnp.asarray(0.0, dtype=vy.dtype), vy)
        vx = jnp.where(touched_ladder, jnp.asarray(0.0, dtype=vx.dtype), vx)
        # `.skipLadderBottom` is the only exit that leaves Harry standing; the
        # lateral one hands him to JumpTab, so he is airborne.
        on_ground = jnp.where(
            on_ladder | exit_top_jump,
            jnp.array(False, dtype=jnp.bool_),
            jnp.where(pat_bottom_exit, jnp.array(True, dtype=jnp.bool_), on_ground),
        )
        current_ground_y = jnp.where(
            pat_bottom_exit,
            lower_ground,
            jnp.where(exit_top_jump, upper_ground, current_ground_y),
        )

        return (
            x, vx, y, vy, on_ground, on_ladder, current_ground_y, climb_active,
            exit_top_jump, climb_pos, ladder_log_frames,
        )

    def _screen_layout(self, room_byte: chex.Array) -> ScreenLayout:
        rb = room_byte.astype(jnp.uint8)
        pt = pit_type(rb)

        has_ladder = (pt == jnp.uint8(0)) | (pt == jnp.uint8(1))

        wall_side_bit = (rb >> jnp.uint8(7)) & jnp.uint8(1)
        wall_side = jnp.where(
            has_ladder,
            jnp.where(wall_side_bit == jnp.uint8(1), jnp.int32(1), jnp.int32(-1)),
            jnp.int32(0),
        )
        wall_x = jnp.where(wall_side_bit == jnp.uint8(1), self.right_wall_x_px, self.left_wall_x_px)

        has_wall = has_ladder
        ladder_x = self.ladder_x_px

        return ScreenLayout(
            has_ladder=has_ladder,
            ladder_x=ladder_x,
            has_wall=has_wall,
            wall_x=wall_x,
            wall_side=wall_side,
        )

    def _hole_bounds_for_room(
        self, room_byte: chex.Array, croc_open: chex.Array
    ) -> tuple[chex.Array, chex.Array]:
        """Select this room's row of HoleBoundsTab.

        The ROM's row select, from the bounds loop in pitfall.asm:

            ldx sceneType
            cpx #CROCO_SCENE      / bne .noCroco1
            bit frameCnt          / bpl .contCroco   ; jaws open -> keep row 4
            dex                   / bne .contCroco   ; jaws closed -> row 3
        .noCroco1:
            cpx #HOLE3_SCENE+2    / bcc .contCroco
            ldx #HOLE3_SCENE+1                         ; clamp scenes >= 3 to row 2
        .contCroco:

        So the rows are 0 (single hole), 1 (triple hole), 2 (pit - also every
        non-crocodile scene from 3 up, quicksand included), 3 (closed jaws) and
        4 (open jaws). Returns two int32 arrays of shape (4,): the left and
        right bound of each of the row's up-to-four openings. Unused slots are
        (0, 0) and are rejected by the `left > 0` guard in _over_hole, mirroring
        the ROM's `beq .exitBounds` terminator.
        """
        pt = pit_type(room_byte.astype(jnp.uint8)).astype(jnp.int32)
        row = jnp.where(
            pt == jnp.int32(CROCO_SCENE),
            jnp.where(croc_open, jnp.int32(4), jnp.int32(3)),
            jnp.where(pt < jnp.int32(3), pt, jnp.int32(2)),
        )

        table = jnp.asarray(self.consts.hole_bounds_tab, dtype=jnp.int32)  # (5, 4, 2)
        bounds = table[row]                                                # (4, 2)
        return bounds[:, 0], bounds[:, 1]

    def _over_hole(
        self,
        room_byte: chex.Array,
        x_anchor: chex.Array,
        x_pos_quicksand: chex.Array,
        croc_open: chex.Array,
    ) -> chex.Array:
        """True when Harry's anchor is inside one of this room's openings.

        Direct port of the bounds loop in pitfall.asm:

            lda HoleBoundsTab,x   ; left bound
            beq .exitBounds       ; 0 terminates the list
            clc / adc xPosQuickSand
            cmp xPosHarry
            bcs .inBounds         ; left+xqs >= x -> Harry is left of the hole, safe
            lda HoleBoundsTab+1,x ; right bound
            sec / sbc xPosQuickSand
            cmp xPosHarry
            bcs .outOfBounds      ; right-xqs >= x -> Harry falls in

        so the falling interval is  left+xqs < x <= right-xqs. The add, the
        subtract and both compares are 6502 byte operations, so the bounds are
        masked to 0..255 and the comparisons are unsigned. That is what closes a
        quicksand pit: xPosQuickSand = DARK_GREEN = $d2 gives 44+210 = 254,
        which is >= every reachable xPosHarry, so the loop reports "in bounds"
        for the whole row without ever reaching the right-hand compare.
        """
        lefts, rights = self._hole_bounds_for_room(room_byte, croc_open)
        xqs = x_pos_quicksand.astype(jnp.int32) & jnp.int32(0xFF)
        left_plus = (lefts + xqs) & jnp.int32(0xFF)
        right_minus = (rights - xqs) & jnp.int32(0xFF)
        slot_used = lefts > jnp.int32(0)
        inside = slot_used & (x_anchor > left_plus) & (x_anchor <= right_minus)
        return jnp.any(inside)

    def step(
        self,
        state: PitfallState,
        action: int,
    ) -> tuple[PitfallObservation, PitfallState, float, bool, PitfallInfo]:
        consts = self.consts

        action = jnp.asarray(action, dtype=jnp.int32)

        x = state.player_x
        y = state.player_y
        vx = state.player_vx
        vy = state.player_vy
        on_ground = state.on_ground
        transition_active = state.respawn_phase != jnp.int32(0)
        gameplay_active = ~transition_active
        time_left = state.time_left
        lives_left = state.lives_left
        hurt_cooldown = state.hurt_cooldown
        scorpion_x = state.scorpion_x
        scorpion_facing_right = state.scorpion_facing_right

        # Which Harry pattern to collide with. The ROM reads CXPPMM at the start
        # of a frame, so the bits it reports were produced by the pattern the
        # previous kernel drew; taking the pose from the incoming state
        # reproduces that one-frame latency and, more importantly, keeps the
        # pose out of the touching_wood cycle it would otherwise close.
        collision_pat = jnp.clip(
            harry_display_pat_id(state, consts), jnp.int32(0), jnp.int32(len(HARRY_PATTERNS) - 1)
        )
        collision_flip = harry_display_facing_left(state)

        # The ROM's frameCnt, incremented at the top of `.processHarry` and read
        # later in the same frame by both the run cadence and the scorpion.
        frame_cnt = jnp.where(
            gameplay_active,
            state.frame_cnt + jnp.int32(1),
            state.frame_cnt,
        )

        down_action = (
            (action == Action.DOWN) |
            (action == Action.DOWNLEFT) |
            (action == Action.DOWNRIGHT) |
            (action == Action.DOWNFIRE) |
            (action == Action.DOWNLEFTFIRE) |
            (action == Action.DOWNRIGHTFIRE)
        )       
        down_pressed = down_action
        move_left = (
            (action == Action.LEFT) |
            (action == Action.UPLEFT) |
            (action == Action.DOWNLEFT) |
            (action == Action.LEFTFIRE) |
            (action == Action.UPLEFTFIRE) |
            (action == Action.DOWNLEFTFIRE)
        )
        move_right = (
            (action == Action.RIGHT) |
            (action == Action.UPRIGHT) |
            (action == Action.DOWNRIGHT) |
            (action == Action.RIGHTFIRE) |
            (action == Action.UPRIGHTFIRE) |
            (action == Action.DOWNRIGHTFIRE)
        )
        move_jump = (
            (action == Action.UP) |
            (action == Action.UPLEFT) |
            (action == Action.UPRIGHT) |
            (action == Action.UPFIRE) |
            (action == Action.UPLEFTFIRE) |
            (action == Action.UPRIGHTFIRE)
        )

        down_pressed = down_pressed & gameplay_active
        move_left = move_left & gameplay_active
        move_right = move_right & gameplay_active
        move_jump = move_jump & gameplay_active

        has_input = action != Action.NOOP
        timer_started = state.timer_started | (has_input & gameplay_active)

        time_left = state.time_left - (timer_started & gameplay_active).astype(jnp.int32)
        time_left = jnp.maximum(time_left, 0)

        layout = self._screen_layout(state.room_byte)
        ladder_x = layout.ladder_x
        has_ladder = layout.has_ladder
        ladder_w = jnp.asarray(consts.ladder_width, dtype=jnp.int32)
        player_w = jnp.asarray(4, dtype=jnp.int32)

        x_int = x.astype(jnp.int32)
        player_right = x_int + player_w
        ladder_right = ladder_x + ladder_w

        overlap_left = player_right > ladder_x
        overlap_right = x_int < ladder_right
        near_ladder = has_ladder & overlap_left & overlap_right

        upper_ground = jnp.asarray(consts.ground_y, dtype=jnp.float32)
        lower_ground = jnp.asarray(consts.underground_y, dtype=jnp.float32)
        
        on_upper_level = state.current_ground_y == upper_ground
        on_lower_level = state.current_ground_y == lower_ground

        # --- The liana, part one: `.exitBounds` -------------------------------
        # The grab is not in .processHarry at all. It sits in the vertical blank
        # that follows the kernel which produced the latch, above `.waitTim`:
        #
        #     .exitBounds:
        #         lda    jumpMode         ; 3
        #         bne    .waitTim         ; 2³
        #         bit    hitLiana         ; 3                 collison with liana
        #         bvc    .waitTim         ; 2³                 no, skip
        #         lda    jumpIndex        ; 3                 currently jumping?
        #         beq    .waitTim         ; 2³                 no, skip
        #         ldx    atLiana          ; 3                 Harry already at liana?
        #         bne    .waitTim         ; 2³                 yes, skip
        #         stx    jumpIndex        ; 3                  no, stop jump
        #         inx                     ; 2
        #         stx    atLiana          ; 3                 enter "liana mode"
        #         stx    soundIdx         ; 3                 start tarzan sound (=0)
        #
        # so it runs before `.doJump` and clears jumpIndex in time to stop the
        # jump on that same frame. No button is read anywhere in it: touching the
        # rope with a jumping sprite is the whole condition.
        liana_present = liana_enabled(state.room_byte)
        # The latch this step is handed belongs to the raster drawn at the end of
        # the previous one, so both Harry and the rope come from the incoming
        # state - never from a position this step is about to produce.
        hit_liana_sub1 = liana_hit_harry(
            state.hmbl_add,
            state.hmbl_dir,
            state.liana_bottom,
            liana_present,
            state.player_x,
            harry_box_top_row(state.player_y, state.current_ground_y, consts),
            collision_pat,
            collision_flip,
        )

        # --- Ground jump ----------------------------------------------------
        # Edge-triggered jump: prevents repeated hops when holding UP.
        jump_pressed = move_jump
        jump_rise = jump_pressed & (~state.jump_pressed_prev)

        speed = jnp.asarray(consts.player_speed, dtype=jnp.float32)

        vx = jnp.where(move_left, -speed, jnp.where(move_right, speed, 0.0))
        # ROM jumping uses the same 1px xPosHarry inc/dec, directed by oldJoystick.
        airborne = (~on_ground) & (~state.on_ladder)
        air_vx = jnp.where(
            state.jump_lock_active,
            state.jump_lock_vx,
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        vx = jnp.where(airborne, air_vx, vx)
        vx = jnp.where(state.on_ladder, 0.0, vx)

        # `lda atLiana / bne .endHarryId` jumps over the whole movement block, so
        # a Harry on the rope has no joystick horizontal at all - and no room
        # transition either, since `.oneScene` is inside the part being skipped.
        on_liana_in = state.at_liana != jnp.uint8(0)
        vx = jnp.where(on_liana_in, jnp.asarray(0.0, dtype=jnp.float32), vx)

        trying_to_enter_ladder = near_ladder & on_lower_level & move_jump
        # `.notJumping: ora climbPos / ora patOfsHarry / ora atLiana / bne .noFire`
        jump_mask = (
            on_ground
            & jump_rise
            & (~state.on_ladder)
            & (~trying_to_enter_ladder)
            & (~on_liana_in)
        )
        jump_launch_vx = jnp.where(move_left, -speed, jnp.where(move_right, speed, 0.0))
        vx = jnp.where(jump_mask, jump_launch_vx, vx)

        jump_lock_active = jnp.where(jump_mask, jnp.array(True, dtype=jnp.bool_), state.jump_lock_active)
        jump_lock_vx = jnp.where(jump_mask, jump_launch_vx, state.jump_lock_vx)

        # --- Vertical --------------------------------------------------------
        # A ground jump follows JumpTab exactly, and so does a descent through an
        # opening: `.outOfBounds` hands Harry to the same table at its last entry.
        #
        # One JAX step is two NTSC frames, so two subframes run per step in
        # order. The ROM spends the launch frame on `dec yPosHarry`, so the
        # first step is that decrement plus one table entry, never two entries.
        #
        # `.stopJump` is reached from two `beq` tests against fixed values, not
        # from "the floor Harry left":
        #
        #     ldx    yPosHarry        ; 3
        #     cpx    #JUNGLE_GROUND   ; 2                 Harry at jungle ground?
        #     beq    .stopJump        ; 2³+1               yes, stop any jump
        #     ...
        #     cpx    #UNDER_GROUND    ; 2                 is Harry at underground bottom?
        #     beq    .stopJump        ; 2³                 yes, stop any jump
        #
        # Because they are equalities, a descent walks straight past the jungle
        # line - yPosHarry goes 33, 34, ... and never equals 32 again - and only
        # UNDER_GROUND ends it. So while one is running the table's target is the
        # underground floor, which is also what lets it descend at all.
        descending_through_opening = is_falling_through_hole(state, consts)
        jump_floor = jnp.where(
            descending_through_opening, lower_ground, state.current_ground_y
        )
        y_start = y
        jump_index_prev = state.jump_index.astype(jnp.int32)

        # `.exitBounds` is above `.doJump`, so the grab has to clear jumpIndex
        # before the table gets its turn - that is what stops the jump dead rather
        # than letting one more entry through.
        liana_grab_sub1 = (
            gameplay_active
            & (state.jump_mode == jnp.uint8(0))
            & hit_liana_sub1
            & (jump_index_prev != jnp.int32(0))
            & (~on_liana_in)
        )
        at_liana_sub1 = on_liana_in | liana_grab_sub1
        jump_index_in = jnp.where(liana_grab_sub1, jnp.int32(0), jump_index_prev)

        y_a, index_a, landed_a = jump_table_subframe(y, jump_index_in, jump_floor)
        y_a = jnp.where(jump_mask, y - jnp.asarray(1.0, dtype=y.dtype), y_a)
        index_a = jnp.where(jump_mask, jnp.int32(1), index_a)

        # The second NTSC frame reads the latch the first frame's raster left, so
        # it sees the rope one `.swingLiana` on and Harry at the y the table just
        # gave him. His column cannot have changed yet: `lda frameCnt / and #$03 /
        # lsr / bcs .endHarryId` only lets the movement block run on the even
        # frame, which is the second of the two this step covers.
        _, _, hmbl_add_sub1, hmbl_dir_sub1, liana_bottom_sub1 = liana_swing_frame(
            state.liana_pos_hi, state.liana_pos_lo
        )
        hit_liana_sub2 = liana_hit_harry(
            hmbl_add_sub1,
            hmbl_dir_sub1,
            liana_bottom_sub1,
            liana_present,
            state.player_x,
            harry_box_top_row(y_a, state.current_ground_y, consts),
            collision_pat,
            collision_flip,
        )
        liana_grab_sub2 = (
            gameplay_active
            & (state.jump_mode == jnp.uint8(0))
            & hit_liana_sub2
            & (index_a != jnp.int32(0))
            & (~at_liana_sub1)
        )
        at_liana_step = at_liana_sub1 | liana_grab_sub2
        index_a = jnp.where(liana_grab_sub2, jnp.int32(0), index_a)

        y_b, index_b, landed_b = jump_table_subframe(y_a, index_a, jump_floor)

        jumping = (jump_index_in > jnp.int32(0)) | jump_mask
        jump_index = jnp.where(jumping, index_b, jnp.int32(0))
        jump_landed = jumping & (landed_a | landed_b)

        gravity = jnp.asarray(consts.gravity, dtype=jnp.float32)
        fall_speed = jnp.asarray(consts.fall_speed, dtype=jnp.float32)
        apply_gravity = (~on_ground) & (~state.on_ladder) & (~jumping)
        # Symmetric gravity on both ascent and descent (capped at fall_speed)
        vy = jnp.where(
            apply_gravity,
            jnp.minimum(vy + gravity, fall_speed),
            vy,
        )

        y = jnp.where(jumping, y_b, y + vy)
        # Renderer predicates read the sign of player_vy, so during a table jump
        # report the motion the table produced rather than a gravity accumulator
        # that no longer drives anything.
        vy = jnp.where(jumping, y - y_start, vy)
        x = x + vx

        # The table owns the lock and releases it on the frame it reaches the
        # floor. The ROM does the same by hand: when the hole-bounds test takes
        # over from a finished jump it writes oldJoystick = $1f, "no direction".
        jump_lock_active = jnp.where(
            jump_landed, jnp.array(False, dtype=jnp.bool_), jump_lock_active
        )

        wall_w = jnp.int32(consts.tunnel_wall_width)

        # `.checkWallHit`. The ROM never clamps Harry to a wall edge - it reads the
        # previous raster's collision latch and nudges him a single pixel away,
        # then reverses the carried direction:
        #
        #     lda    xPosHarry        ; 3                 determine where Harry hit the wall
        #     cmp    #140             ; 2                 right wall from the right?
        #     bcs    .hitFromRight    ; 2³                 yes, continue
        #     cmp    #13              ; 2                 left wall from the left?
        #     bcc    .hitFromLeft     ; 2³                 yes, continue
        #     cmp    #80              ; 2                 left or right wall?
        #     bcs    .hitFromLeft     ; 2³
        #   .hitFromRight:
        #     inc    xPosHarry        ; 5                 bounce back one pixel and..
        #     ldx    #MOVE_RIGHT      ; 2                 ..change direction to right
        #   .hitFromLeft:
        #     dec    xPosHarry        ; 5                 bounce back one pixel and..
        #     ldx    #MOVE_LEFT       ; 2                 ..change direction to left
        #   .contWallHit:
        #     stx    oldJoystick      ; 3
        #
        # The three compares split the screen at its midpoint and push Harry away
        # from whichever wall he is beside, so entering at XMAX_HARRY on the right
        # of a right-hand wall can only ever move him further right.
        wall_left = layout.wall_x
        wall_right = layout.wall_x + wall_w
        pat_box = _HARRY_COLLISION_BOXES[collision_flip.astype(jnp.int32), collision_pat]

        # CXPPMM reports what the previous kernel drew, and that kernel belonged
        # to the incoming room, so both the position and the wall come from the
        # state this step was handed - never from the room just selected.
        drawn_x = state.player_x.astype(jnp.int32)
        wall_hit = (
            layout.has_wall
            & on_lower_level
            & ((drawn_x + pat_box[1]) > wall_left)
            & ((drawn_x + pat_box[0]) < wall_right)
        )
        from_right = (drawn_x >= jnp.int32(140)) | (
            (drawn_x >= jnp.int32(13)) & (drawn_x < jnp.int32(80))
        )
        bounce = jnp.where(from_right, jnp.float32(1.0), jnp.float32(-1.0))
        x = jnp.where(wall_hit, x + bounce.astype(x.dtype), x)
        # `stx oldJoystick` with MOVE_RIGHT / MOVE_LEFT.
        jump_lock_vx = jnp.where(wall_hit, bounce, jump_lock_vx)

        x_after_move = x

        # Screen-exit thresholds from the ROM (XMIN_HARRY / XMAX_HARRY). These
        # are in the same xPosHarry coordinate as the hole bounds, so now that
        # player_x is that anchor they apply directly. They also keep Harry's
        # widest pose fully on screen, which a 0..156 range would not.
        left_edge = jnp.asarray(consts.screen_exit_x_min, dtype=jnp.float32)
        right_edge_for_left_of_player = jnp.asarray(consts.screen_exit_x_max, dtype=jnp.float32)

        exited_left  = x_after_move < left_edge
        exited_right = x_after_move > right_edge_for_left_of_player

        on_lower_level_for_stride = state.current_ground_y == jnp.asarray(consts.underground_y, jnp.float32)
        stride = jnp.where(on_lower_level_for_stride, jnp.int32(3), jnp.int32(1))

        room_byte = state.room_byte
        room_byte = jnp.where(
            exited_right,
            step_lfsr(room_byte, lfsr_right_u8, stride),
            room_byte,
        )
        room_byte = jnp.where(
            exited_left,
            step_lfsr(room_byte, lfsr_left_u8, stride),
            room_byte,
        )

        screen_id = state.screen_id
        screen_id = jnp.where(exited_right, jnp.mod(screen_id + stride, jnp.int32(255)), screen_id)
        screen_id = jnp.where(exited_left, jnp.mod(screen_id - stride, jnp.int32(255)), screen_id)

        new_screen_id = screen_id
        new_room_byte = room_byte

        x_if_left_exit = right_edge_for_left_of_player
        x_if_right_exit = left_edge

        x = jnp.where(
            exited_left,
            x_if_left_exit,
            jnp.where(exited_right, x_if_right_exit, x_after_move)
        )

        entered_new_room = exited_left | exited_right
        # ContRandom runs on every scene change and writes xPosScorpion = 76.
        scorpion_spawn_x = jnp.asarray(consts.scorpion_spawn_x, dtype=jnp.int32)
        scorpion_x = jnp.where(entered_new_room, scorpion_spawn_x, scorpion_x)

        x = jnp.clip(x, left_edge, right_edge_for_left_of_player)

        # --- xPosQuickSand: MainLoop's "calculate pits, quicksand etc." -------
        # The ROM recomputes the border from frameCnt on every frame of a
        # quicksand scene, pins it to 0 in every other scene, and freezes it
        # (along with PF2Lst) while Harry is falling into the pit -
        # `cmp #55 / bcs .doQuickSand` then `cmp #JUNGLE_GROUND+1 / bcs
        # .stopQuickSand` - or the game is stopped. One JAX step is two NTSC
        # frames and the end-of-step frame is 2*frame_cnt; that is the frame the
        # renderer shows, so the bounds below and the raster read one value.
        rom_y_start = (state.player_y - (upper_ground - jnp.float32(JUNGLE_GROUND))).astype(jnp.int32)
        quicksand_falling = (rom_y_start >= jnp.int32(JUNGLE_GROUND + 1)) & (
            rom_y_start < jnp.int32(55)
        )
        quicksand_scene = scene_is_quicksand(new_room_byte)
        x_pos_quicksand = jnp.where(
            quicksand_scene,
            jnp.where(
                quicksand_falling,
                state.x_pos_quicksand,
                quicksand_border(jnp.int32(2) * frame_cnt).astype(jnp.uint8),
            ),
            jnp.array(0, dtype=jnp.uint8),
        )
        # `bit frameCnt / bpl`: the crocodile jaws hang on bit 7 of the same
        # end-of-step frame, so the drawn mouth and the bounds always agree.
        croc_open = croc_jaws_open(jnp.int32(2) * frame_cnt)

        # The hole test runs at Harry's post-movement position, in the room he
        # ended up in, and only feeds the ground decision below. The ROM does
        # the same: its bounds loop is gated on `yPosHarry == JUNGLE_GROUND`,
        # so being above an opening never affects an airborne Harry.
        over_any_hole = self._over_hole(new_room_byte, x.astype(jnp.int32), x_pos_quicksand, croc_open)

        previous_ground = state.current_ground_y
        clamp_mask = ~state.on_ladder

        raw_on_ground_upper = (y >= previous_ground) & (~over_any_hole)

        # No pit scene has a ladder and so none has an underground floor to
        # arrive on. `ContRandom` only writes WITHLADDER for sceneType 0 and 1
        # (`cmp #HOLE3_SCENE+1 / bcs .setFlag`), so the transfer does not apply
        # to the tar pit, the swamp, the crocodiles or any quicksand. The -100
        # is *not* here: it belongs to the first frame of the fall, not to the
        # arrival - see `.skipFalling` below.
        in_pit_scene = scene_is_pit(new_room_byte)
        falling_to_lower = (
            on_upper_level
            & over_any_hole
            & (~in_pit_scene)
            & (y >= lower_ground)
        )

        score = state.score

        raw_on_ground_lower = (y >= previous_ground)

        raw_on_ground = jnp.where(
            on_upper_level,
            raw_on_ground_upper | falling_to_lower,
            raw_on_ground_lower,
        )

        current_ground_y = jnp.where(
            falling_to_lower,
            lower_ground,
            state.current_ground_y
        )

        on_ground = jnp.where(clamp_mask, raw_on_ground, on_ground)

        vy = jnp.where(
            clamp_mask & (on_ground & (vy > 0)),
            0.0,
            vy,
        )

        landing_y = jnp.where(falling_to_lower, lower_ground, previous_ground)
        y = jnp.where(clamp_mask & on_ground, landing_y, y)

        # `.outOfBounds` - the one frame that starts a descent, ordinary hole and
        # fatal pit alike:
        #
        #     .outOfBounds:
        #         inc    yPosHarry        ; 5                 Harry is falling down
        #         ldx    #JUMP_LEN        ; 2
        #         stx    jumpIndex        ; 3
        #         dex                     ; 2
        #         stx    oldJoystick      ; 3                 x=$1f -> no direction
        #
        # It lives in the vertical blank, above `.processHarry`, so the very same
        # NTSC frame goes on to run `.doJump` and take its first table step. That
        # is why the entry frame moves two pixels - the `inc` and
        # JumpTab[JUMP_LEN-1], which is -1 - while every frame after it moves one.
        # The bounds loop cannot repeat, and the reason is that its gate is an
        # equality, not a threshold:
        #
        #     lda    climbPos         ; 3                 Harry at ladder?
        #     bne    .exitBounds      ; 2³+1               yes, skip bounds check
        #     lda    yPosHarry        ; 3
        #     cmp    #JUNGLE_GROUND   ; 2                 Harry at ground?
        #     bne    .exitBounds      ; 2³                 no, skip bounds check
        #
        # `bne` means the loop is reached on exactly one frame - the one where
        # Harry is still standing on the jungle line. From frame 2 on, yPosHarry is
        # 34, 35, ... and never equals JUNGLE_GROUND again, so the `inc` happens
        # once for the whole fall and the table carries the rest.
        entering_opening = (
            clamp_mask
            & on_upper_level
            & over_any_hole
            & (state.player_y == previous_ground)
        )

        # NTSC frame 1: the `inc`, then .doJump's first table step.
        y_inc = state.player_y + jnp.asarray(1.0, dtype=y.dtype)
        y_f1, index_f1, _ = jump_table_subframe(y_inc, jnp.int32(JUMP_LEN), lower_ground)
        # NTSC frame 2: bounds skipped, .doJump only, fed the whole of frame 1.
        y_f2, index_f2, _ = jump_table_subframe(y_f1, index_f1, lower_ground)

        # `.doJump` continues straight into the falling branch, still inside the
        # very frame `.outOfBounds` ran:
        #
        #     ldy    ladderFlag       ; 3                 ladder in scene?
        #     beq    .skipFalling     ; 2³+1               no, skip falling
        #     cpx    #JUNGLE_GROUND+2 ; 2
        #     bne    .skipFalling     ; 2³+1
        #     lda    #SOUND_FALLING   ; 2                 Harry is falling into a hole
        #     sta    soundIdx         ; 3                 start falling-sound
        #     lda    #$00             ; 2
        #     jsr    DecScoreHi       ; 6                 subtract 100 points from score
        #
        # X is the yPosHarry the table just produced, and frame 1 always lands on
        # JUNGLE_GROUND+2 - the `inc` and JumpTab[JUMP_LEN-1] together. So the
        # deduction is taken on the first frame of the fall and, because no later
        # frame equals 34 again, exactly once. `ldy ladderFlag / beq` keeps the tar
        # pit and the swamp out of it entirely.
        started_falling = (
            entering_opening
            & has_ladder
            & (y_f1 == previous_ground + jnp.asarray(2.0, dtype=y.dtype))
        )
        score = score + jnp.where(started_falling, jnp.int32(-100), jnp.int32(0))
        score = jnp.maximum(score, jnp.int32(0))
        sound_idx_falling = jnp.where(
            started_falling, jnp.int32(SOUND_FALLING), state.sound_idx
        )

        y = jnp.where(entering_opening, y_f2, y)
        jump_index = jnp.where(entering_opening, index_f2, jump_index)
        on_ground = jnp.where(entering_opening, jnp.array(False, dtype=jnp.bool_), on_ground)
        # `stx oldJoystick` with x = $1f is "no direction", so the descent carries
        # no sideways motion at all; air_vx reads the lock and finds nothing.
        jump_lock_active = jnp.where(
            entering_opening, jnp.array(True, dtype=jnp.bool_), jump_lock_active
        )
        jump_lock_vx = jnp.where(entering_opening, jnp.asarray(0.0, dtype=jnp.float32), jump_lock_vx)
        vx = jnp.where(entering_opening, jnp.asarray(0.0, dtype=vx.dtype), vx)
        # Descriptive only: the renderer reads the sign of player_vy to pick a
        # falling pose, so report what the table displaced, never the reverse.
        vy = jnp.where(entering_opening, y - state.player_y, vy)

        has_logs, logs_are_rolling, log_count, log_xs, has_fireplace, has_snake = room_hazards_from_room_byte(new_room_byte)

        # --- Log geometry, needed before the ladder runs ----------------------
        # `.hitLogs` sits above the ladder block in .processHarry, so each NTSC
        # frame of the ladder has to be able to read CXPPMM for itself. The log
        # side of that read depends only on the room and the timer, never on
        # Harry, so it is computed once here and sampled twice below.
        screen_w_i = jnp.int32(consts.screen_width)
        total_frames = jnp.int32(self.consts.initial_time_seconds * self.consts.fps)
        # The rolling-log move is later in ProcessObjects and only runs on the
        # even NTSC frame, so both reads in this step see the position the
        # incoming state rendered.
        frames_elapsed_at_draw = jnp.maximum(
            total_frames - state.time_left.astype(jnp.int32), jnp.int32(0)
        )
        frames_elapsed_at_draw = (
            frames_elapsed_at_draw * state.timer_started.astype(jnp.int32)
        )
        log_left_x = log_left_edges(
            log_xs, logs_are_rolling, frames_elapsed_at_draw, consts.screen_width
        )
        wood_w = jnp.int32(consts.wood_w)
        wood_h = jnp.int32(consts.wood_h)
        wood_top = jnp.int32(consts.ground_y - consts.wood_h + consts.wood_y_offset)
        wood_y0 = wood_top
        wood_y1 = wood_top + wood_h
        active = jnp.arange(LOG_MAX_COPIES, dtype=jnp.int32) < log_count
        # xPosObject is the left edge of the GRP1 box, the same convention the
        # renderer draws at, so no centring correction belongs here.
        seg1_x0 = log_left_x
        seg1_x1 = jnp.minimum(log_left_x + wood_w, screen_w_i)
        wraps = (log_left_x + wood_w) > screen_w_i
        seg2_x0 = jnp.zeros_like(seg1_x0)
        seg2_x1 = (log_left_x + wood_w) - screen_w_i

        def _log_hits(harry_x, harry_box_top, pattern, flip):
            """CXPPMM for one raster: this frame's Harry pixels against the logs.

            The bounds come from the pose's own lit pixels, which is the same
            extent the accepted off-ladder contact uses, so a rung push and a
            ground hit agree about what touching means.
            """
            hx0, hx1, hy0, hy1 = harry_collision_bounds(
                harry_x, harry_box_top, pattern, flip
            )
            hit_y = (hy1 > wood_y0) & (hy0 < wood_y1)
            hit_s1 = (hx1 > seg1_x0) & (hx0 < seg1_x1)
            hit_s2 = wraps & (hx1 > seg2_x0) & (hx0 < seg2_x1)
            return has_logs & jnp.any(active & (hit_s1 | hit_s2) & hit_y)

        x, vx, y, vy, on_ground, on_ladder, current_ground_y, climb_active, started_ladder_exit, climb_pos, ladder_log_frames = self._apply_ladder(
            state=state,
            room_byte=new_room_byte,
            x=x,
            vx=vx,
            y=y,
            vy=vy,
            down_pressed=down_pressed,
            move_jump=move_jump,
            move_left=move_left,
            move_right=move_right,
            on_ground=on_ground,
            current_ground_y=current_ground_y,
            log_hits=_log_hits,
            log_push_enabled=gameplay_active,
        )

        # `lda #1 / sta jumpIndex` and `sta oldJoystick`: the ladder exit hands
        # Harry to the ordinary jump, at table index 1, carrying the direction he
        # was pushing. No grace period is needed to stop him re-grabbing the
        # ladder, because both entry tests require him to be on the ground.
        exit_vx = jnp.where(move_left, -speed, jnp.where(move_right, speed, 0.0))
        vx = jnp.where(started_ladder_exit, exit_vx, vx)
        jump_index = jnp.where(started_ladder_exit, jnp.int32(1), jump_index)
        jump_lock_active = jnp.where(
            started_ladder_exit, jnp.array(True, dtype=jnp.bool_), jump_lock_active
        )
        jump_lock_vx = jnp.where(started_ladder_exit, exit_vx, jump_lock_vx)

        # Clear jump lock once grounded (or while on ladder).
        jump_lock_active = jump_lock_active & (~on_ground) & (~on_ladder)
        jump_index = jnp.where(on_ground | on_ladder, jnp.int32(0), jump_index)

        has_scorpion = has_scorpion_from_room_byte(new_room_byte)
        player_is_underground = current_ground_y == lower_ground

        # --- Scorpion movement, ported from "move the scorpion towards harry" --
        #     lda ladderFlag / bne .noMoveScorpion   (ladder rooms show a wall)
        #     lda xPosHarry / sec / sbc xPosScorpion
        #     beq .noMoveScorpion                    (equal: no move, keep facing)
        #     bcs .rightOfScorpion / ldx #REFLECT    (Harry left -> reflected)
        #     lda frameCnt / and #$07 / bne .end     (one pixel every 8th frame)
        #     inc xPosScorpion / bcs .end / dec / dec
        #
        # frameCnt is incremented before this block, and one JAX step is two
        # NTSC frames, so the step covers frameCnt 2*frame_cnt-1 and 2*frame_cnt.
        # `frameCnt & 7 == 0` needs an even multiple of eight, so it lands on the
        # step where frame_cnt itself is a multiple of four.
        # What the previous kernel put on screen, which is what CXPPMM reports:
        # the ROM reads it at `; check collisions between Harry and object:`,
        # long before `; move the scorpion towards harry:` runs. Collision uses
        # these, and this frame's move only shows up in the next state.
        scorpion_x_at_draw = scorpion_x.astype(jnp.int32)
        scorpion_facing_at_draw = scorpion_facing_right

        # Base is the post-entry value, so a fresh room's spawn survives.
        scorpion_x_int = scorpion_x.astype(jnp.int32)
        harry_delta = x.astype(jnp.int32) - scorpion_x_int
        harry_is_right = harry_delta > jnp.int32(0)
        harry_is_left = harry_delta < jnp.int32(0)

        scorpion_active = has_scorpion & gameplay_active
        move_tick = jnp.mod(frame_cnt, jnp.int32(consts.scorpion_move_period_steps)) == jnp.int32(0)
        scorpion_step = jnp.where(harry_is_right, jnp.int32(1), jnp.where(harry_is_left, jnp.int32(-1), jnp.int32(0)))
        scorpion_x = jnp.where(
            scorpion_active & move_tick,
            scorpion_x_int + scorpion_step,
            scorpion_x_int,
        )

        # Facing tracks Harry's side, and an exactly-equal x keeps the old one
        # because the ROM branches past `stx reflectScorpion` in that case.
        scorpion_facing_right = jnp.where(
            scorpion_active & (harry_is_right | harry_is_left),
            harry_is_right,
            scorpion_facing_right,
        )

        screen_w_i = jnp.int32(consts.screen_width)

        # Harry's lit pixels for this frame's pose, from HARRY_PATTERNS. Replaces
        # the old fixed 4x8 torso, which was neither the ROM's shape nor the
        # drawn one. harry_box_top is the same origin the renderer draws at, so
        # collidable pixels and visible pixels are the same pixels.
        harry_box_top = harry_box_top_row(y, current_ground_y, consts)
        x0, x1, y0, y1 = harry_collision_bounds(x, harry_box_top, collision_pat, collision_flip)

        overlap_y = (y1 > wood_y0) & (y0 < wood_y1)
        W = screen_w_i

        overlap_seg1 = (x1 > seg1_x0) & (x0 < seg1_x1)
        overlap_seg2 = wraps & (x1 > seg2_x0) & (x0 < seg2_x1)
        overlap_x = overlap_seg1 | overlap_seg2

        # Rolling-log contact uses an earlier padded/shifted overlap zone.
        # Compute it before we potentially freeze X so the grab point is stable.
        roll_pad_x = jnp.int32(consts.rolling_wood_contact_pad_x)
        roll_shift_x = jnp.int32(consts.rolling_wood_contact_shift_x)
        roll_seg1_x0 = seg1_x0 + roll_shift_x
        roll_seg1_x1 = seg1_x1 + roll_shift_x
        roll_seg2_x0 = seg2_x0 + roll_shift_x
        roll_seg2_x1 = seg2_x1 + roll_shift_x
        roll_overlap_seg1 = (x1 > (roll_seg1_x0 - roll_pad_x)) & (x0 < (roll_seg1_x1 + roll_pad_x))
        roll_overlap_seg2 = wraps & (x1 > (roll_seg2_x0 - roll_pad_x)) & (x0 < (roll_seg2_x1 + roll_pad_x))
        roll_overlap_x = roll_overlap_seg1 | roll_overlap_seg2

        # Rolling logs: block Harry without dragging him. If a rolling log
        # would overlap this frame, freeze X at the contact point.
        rolling_active = has_logs & logs_are_rolling & gameplay_active & on_upper_level & on_ground & (~on_ladder)
        rolling_would_overlap = rolling_active & jnp.any(active & roll_overlap_x & overlap_y)
        # Both the blocking position and the contact flag come from the previous
        # frame, which is a different room once Harry has just wrapped. Fall back
        # to the already-wrapped x and drop the stale flag in that case.
        contact_start_x = jnp.where(entered_new_room, x, state.player_x)
        previously_touching = state.touching_rolling_wood & (~entered_new_room)
        started_rolling_contact = rolling_would_overlap & (~previously_touching)
        rolling_contact_x = jnp.where(
            started_rolling_contact,
            contact_start_x,
            jnp.where(rolling_would_overlap, state.rolling_wood_contact_x, jnp.asarray(0.0, dtype=jnp.float32)),
        )
        x = jnp.where(rolling_would_overlap, rolling_contact_x, x)
        vx = jnp.where(rolling_would_overlap, jnp.asarray(0.0, dtype=jnp.float32), vx)

        # Update player bbox after rolling-log contact resolution.
        x0, x1, y0, y1 = harry_collision_bounds(x, harry_box_top, collision_pat, collision_flip)

        wood_visual_pad_x = jnp.int32(consts.wood_visual_contact_pad_x)
        wood_visual_shift_x = jnp.int32(consts.wood_visual_contact_shift_x)
        visual_seg1_x0 = seg1_x0 + wood_visual_shift_x
        visual_seg1_x1 = seg1_x1 + wood_visual_shift_x
        visual_seg2_x0 = seg2_x0 + wood_visual_shift_x
        visual_seg2_x1 = seg2_x1 + wood_visual_shift_x
        visual_overlap_seg1 = (x1 > (visual_seg1_x0 - wood_visual_pad_x)) & (x0 < (visual_seg1_x1 + wood_visual_pad_x))
        visual_overlap_seg2 = wraps & (x1 > (visual_seg2_x0 - wood_visual_pad_x)) & (x0 < (visual_seg2_x1 + wood_visual_pad_x))
        visual_overlap_x = visual_overlap_seg1 | visual_overlap_seg2

        # `.contCollision: lda yPosHarry / cmp #64 / bcs .checkWallHit / lda
        # atLiana / bne .endCollision` - on the rope the whole upper-ground half of
        # the collision block is jumped over, so no log drain, no kneeing pose and
        # none of the hazards below.
        off_liana = ~at_liana_step
        touching_any = jnp.any(active & overlap_x & overlap_y)
        touching_wood = has_logs & jnp.any(active & visual_overlap_x & overlap_y) & off_liana
        # Moving logs (rolling): visual slide/contact matches the blocked state.
        touching_rolling_wood = rolling_would_overlap
        log_contact_first = has_logs & touching_any & gameplay_active & off_liana

        upper_ground = jnp.asarray(consts.ground_y, dtype=jnp.float32)
        lower_ground = jnp.asarray(consts.underground_y, dtype=jnp.float32)

        # --- The score drain, `.decScore` -------------------------------------
        # Both `.hitLogs` paths end at `jsr DecScoreLo`, one call per NTSC frame
        # that saw a collision. On the ladder that count came out of the ladder
        # subframe loop above, where each frame read its own raster; off the
        # ladder this port keeps treating a standing contact as both frames.
        contact_frames = jnp.where(
            on_ladder | (ladder_log_frames > jnp.int32(0)),
            ladder_log_frames,
            log_contact_first.astype(jnp.int32) * jnp.int32(NTSC_FRAMES_PER_STEP),
        )
        drain = contact_frames * jnp.int32(consts.wood_drain_per_frame)
        score = jnp.maximum(score - drain, jnp.int32(0))
        climb_active = climb_active | (ladder_log_frames > jnp.int32(0))
        # ----------------------------------------------------------------------

        can_hurt = (hurt_cooldown == jnp.int32(0)) & gameplay_active & off_liana

        # Scorpion and cobra collision are both CXPPMM: a lit GRP0 pixel and a
        # lit GRP1 pixel on the same screen coordinate. The cobra previously
        # used Harry's lit-pixel AABB against the 8x14 sprite rectangle, which
        # killed jumps that never shared a pixel with Cobra0/Cobra1.
        harry_bitmap = _HARRY_BITMAPS[collision_flip.astype(jnp.int32), collision_pat]
        scorpion_bitmap = _SCORPION_BITMAPS[
            (~scorpion_facing_at_draw).astype(jnp.int32),
            jnp.bitwise_and(scorpion_x_at_draw, jnp.int32(1)),
        ]
        overlap_scorpion = player_object_pixels_collide(
            harry_bitmap,
            x.astype(jnp.int32),
            harry_box_top,
            scorpion_bitmap,
            scorpion_x_at_draw,
            self.scorpion_box_top_px,
        )

        cobra_frame_at_draw = cobra_animation_frame(
            state.time_left,
            state.timer_started,
            consts,
            self.renderer.COBRA_ANIM_BIT4,
        )
        overlap_snake = player_object_pixels_collide(
            harry_bitmap,
            x.astype(jnp.int32),
            harry_box_top,
            _COBRA_BITMAPS[cobra_frame_at_draw],
            jnp.int32(consts.object_x),
            self.cobra_box_top_px,
        )

        # Fire is the same CXPPMM read as the cobra, over the same GRP1 object
        # box, and shares the cobra's random2 bit-4 animation source (AnimateTab
        # gives both OBJECT_H). The collision reads the incoming raster's frame
        # and position, exactly like the cobra and the scorpion.
        overlap_fire = player_object_pixels_collide(
            harry_bitmap,
            x.astype(jnp.int32),
            harry_box_top,
            self.renderer.FIRE_BITMAPS[cobra_frame_at_draw],
            jnp.int32(consts.object_x),
            self.renderer.object_box_top_px,
        )

        hit_scorpion = has_scorpion & player_is_underground & overlap_scorpion & can_hurt
        hit_snake = has_snake & overlap_snake & can_hurt
        hit_fire = has_fireplace & overlap_fire & can_hurt

        # `cpx #54 / bne .endDoJump / tya / bne .endDoJump / jmp KilledHarry`:
        # once the sink reaches yPosHarry 54 and there is no ladder in the scene,
        # Harry is killed. `tya` is ladderFlag, which ContRandom leaves NOLADDER
        # for every sceneType from 2 up, so the tar pit, the swamp, the
        # crocodile jaws and every quicksand share this one fatal depth.
        # KilledHarry itself only starts the death tune, so the life and the
        # restart come from the same path every other hazard uses.
        pit_kill_y = jnp.asarray(
            float(consts.ground_y + PIT_KILL_DEPTH), dtype=y.dtype
        )
        hit_pit = (
            in_pit_scene
            & on_upper_level
            & (~on_ground)
            & (y >= pit_kill_y)
            & can_hurt
        )

        hit_other_hazard = hit_fire | hit_snake | hit_pit
        hit_hazard = hit_scorpion | hit_other_hazard

        # Every fatal path in the ROM ends at the same three instructions, and
        # none of them touches livesPat or Harry: the life and the restart come
        # 133 frames later, when the dead tune reaches SOUND_FALLING-1.
        no_game_scroll = jnp.where(
            hit_hazard, jnp.int32(KILLED_HARRY_SCROLL), state.no_game_scroll
        )
        # The collision test sits below `.doJump` in .processHarry, so a hazard on
        # the same frame overwrites the falling tune with the dead one.
        sound_idx = jnp.where(hit_hazard, jnp.int32(SOUND_DEAD), sound_idx_falling)

        # yPosHarry is frozen from here, and `cmp #71` reads it at restart time
        # to pick the branch, so record it against whichever ground band Harry
        # was standing on.
        died_underground = current_ground_y >= jnp.asarray(
            consts.underground_y, dtype=jnp.float32
        )
        rom_y_at_death = jnp.where(
            died_underground,
            y - jnp.asarray(consts.underground_y, dtype=jnp.float32) + jnp.float32(UNDER_GROUND),
            y - jnp.asarray(consts.ground_y, dtype=jnp.float32) + jnp.float32(JUNGLE_GROUND),
        )
        restart_rom_y = jnp.where(
            hit_hazard, rom_y_at_death.astype(jnp.int32), state.restart_rom_y
        )

        # --- Treasure collection: `.contCollision` -> CheckTreasures -> .incScore
        # The same CXPPMM bit-7 latch the fire and cobra read, with the incoming
        # raster's Harry pose and object frame. The ROM collects when Harry's
        # pixels meet the treasure's, on the surface (yPosHarry < 64), not on the
        # liana, in a treasure scene whose persistence bit is still clear.
        obj_type = obj_code_u8(new_room_byte).astype(jnp.int32)
        treasure_type = obj_type & jnp.int32(3)
        is_treasure_scene = pit_code_u8(new_room_byte) == jnp.uint8(5)
        already_collected = treasure_collected(new_room_byte, state.treasure_bits)
        bar_frame = cobra_animation_frame(
            state.time_left, state.timer_started, consts, self.renderer.COBRA_ANIM_BIT4
        )
        treasure_frame = jnp.where(
            _TREASURE_ANIMATED[treasure_type], bar_frame, jnp.int32(0)
        )
        overlap_treasure = player_object_pixels_collide(
            harry_bitmap,
            x.astype(jnp.int32),
            harry_box_top,
            self.renderer.TREASURE_BITMAPS[treasure_type, treasure_frame],
            jnp.int32(consts.object_x),
            self.renderer.object_box_top_px,
        )
        collect = (
            is_treasure_scene
            & (~already_collected)
            & overlap_treasure
            & gameplay_active
            & off_liana
            & (~player_is_underground)
        )

        # `sta treasureBits,x` - set the bit, so the treasure stays collected.
        tree = tree_variant_u8(new_room_byte).astype(jnp.int32)
        tmask = jnp.asarray(TREASURE_MASK, dtype=jnp.uint8)[obj_type]
        tb = state.treasure_bits
        treasure_bits = tb.at[tree].set(jnp.where(collect, tb[tree] | tmask, tb[tree]))
        # `.incScore`: the treasure's value lands on the collection frame only,
        # and the now-set bit keeps any later overlap from scoring again.
        score = score + jnp.where(
            collect, _TREASURE_SCORES[treasure_type], jnp.int32(0)
        )
        # `dec treasureCnt / bpl .incScore / dec noGameScroll`: the 32nd
        # collection ends the game.
        treasure_cnt = state.treasure_cnt - collect.astype(jnp.int32)
        game_won = collect & (treasure_cnt < jnp.int32(0))
        # `lda #SOUND_TREASURE / sta soundIdx` - record the tune. The death sound
        # takes priority where both could fire; in a treasure scene they cannot.
        sound_idx = jnp.where(collect & (~hit_hazard), jnp.int32(SOUND_TREASURE), sound_idx)

        # --- `.swingLiana` ---------------------------------------------------
        # `.endCollision / jmp .swingLiana` puts the swing immediately below the
        # collision block, so this sits where the ROM has it. Two ordered NTSC
        # frames per step, the second reading everything the first wrote.
        #
        # `jmp KilledHarry` leaves .processHarry above this label, so the frame
        # that kills Harry never swings the liana, and the frozen pause that
        # follows never reaches .processHarry at all.
        liana_swings = gameplay_active & (~hit_hazard)
        liana_pos_hi = state.liana_pos_hi
        liana_pos_lo = state.liana_pos_lo
        hmbl_add = state.hmbl_add
        hmbl_dir = state.hmbl_dir
        liana_bottom = state.liana_bottom
        for _ in range(NTSC_FRAMES_PER_STEP):
            swung = liana_swing_frame(liana_pos_hi, liana_pos_lo)
            liana_pos_hi = jnp.where(liana_swings, swung[0], liana_pos_hi)
            liana_pos_lo = jnp.where(liana_swings, swung[1], liana_pos_lo)
            hmbl_add = jnp.where(liana_swings, swung[2], hmbl_add)
            hmbl_dir = jnp.where(liana_swings, swung[3], hmbl_dir)
            liana_bottom = jnp.where(liana_swings, swung[4], liana_bottom)

        # --- `.stopJump` and `.skipJumpOff` -----------------------------------
        # jumpMode is written in exactly two places. `.stopJump` clears it when the
        # table puts Harry back on a floor, and the release below sets it, so it is
        # what keeps a Harry who has just let go from grabbing the rope again on
        # the way down.
        jump_mode = jnp.where(jump_landed, jnp.uint8(0), state.jump_mode)

        #     lda    atLiana          ; 3                 Harry at liana?
        #     beq    .skipJumpOff     ; 2³                 no, skip jump of liana
        #     lda    joystick         ; 3
        #     and    #~[$f0|MOVE_DOWN]; 2                 joystick down?
        #     bne    .skipJumpOff     ; 2³                 no, skip
        #     sta    atLiana          ; 3                  yes, leave "liana mode"
        #     lda    #JUMP_LEN/2      ; 2                 start jump down
        #     sta    jumpIndex        ; 3
        #     sta    jumpMode         ; 3
        #     ldy    #MOVE_RIGHT      ; 2
        #     lda    hmblDir          ; 3                 jump in liana direction
        #     bmi    .jumpRight       ; 2³
        #     ldy    #MOVE_LEFT       ; 2
        #   .jumpRight:
        #     sty    oldJoystick      ; 3
        #
        # The mask leaves bit 1 alone and nothing else, so any DOWN releases -
        # down, down-left, down-right, down-and-fire alike.
        releasing = at_liana_step & down_pressed
        at_liana_out = jnp.where(
            releasing, jnp.uint8(0), at_liana_step.astype(jnp.uint8)
        )

        # `.skipJumpOff` is below `.doJump`, so the release frame writes jumpIndex
        # after the table has already had its turn and consumes nothing. This step
        # holds two NTSC frames, so the second one is the first to take an entry -
        # JumpTab[15], one pixel up - and the arc carries on from index 17.
        moving_right = hmbl_dir.astype(jnp.int32) == jnp.int32(HMBL_DIR_RIGHT)
        release_vx = jnp.where(moving_right, speed, -speed)
        y_release, index_release, landed_release = jump_table_subframe(
            state.player_y, jnp.int32(JUMP_LEN // 2), jump_floor
        )
        jump_index = jnp.where(releasing, index_release, jump_index)
        jump_mode = jnp.where(releasing, jnp.uint8(JUMP_LEN // 2), jump_mode)
        jump_mode = jnp.where(releasing & landed_release, jnp.uint8(0), jump_mode)
        # oldJoystick is MOVE_RIGHT or MOVE_LEFT, which is what carries Harry
        # sideways for the rest of the arc; jump_lock_vx is this port's name for it.
        jump_lock_active = jnp.where(
            releasing, jnp.array(True, dtype=jnp.bool_), jump_lock_active
        )
        jump_lock_vx = jnp.where(releasing, release_vx, jump_lock_vx)

        # --- `.skipSwingHarry` -------------------------------------------------
        #     lda    hmblAdd / lsr / lsr        ; A = hmblAdd >> 2
        #     clc
        #     ldy    hmblDir / bmi .isNeg
        #     eor    #$ff / sec                 ; the left-moving branch negates
        #   .isNeg:
        #     adc    #75 / sta xPosHarry
        #     lda    #$29 / sec / sbc lianaBottom / sta yPosHarry
        #
        # $f0 is negative, so `bmi` takes the right-moving branch straight to the
        # add and Harry ends up at 75 + offset; the left branch one's-complements
        # the offset and sets carry, which is 75 - offset in a byte. Both are
        # written every frame Harry hangs there, so his x and y are the rope's,
        # never anything of his own. The 75 is `adc #75`'s own literal - three
        # columns left of the ball's anchor 78, which is what lays the rope onto
        # Harry6's raised hand at box columns 0-1.
        swing_offset = (hmbl_add.astype(jnp.int32) >> jnp.int32(2)) & jnp.int32(0xFF)
        swing_x = (
            jnp.int32(LIANA_HARRY_X)
            + jnp.where(moving_right, swing_offset, -swing_offset)
        ).astype(jnp.uint8).astype(jnp.float32)
        swing_rom_y = (jnp.int32(0x29) - liana_bottom.astype(jnp.int32)) & jnp.int32(0xFF)
        swing_y = (
            upper_ground - jnp.float32(JUNGLE_GROUND) + swing_rom_y.astype(jnp.float32)
        )

        attached = at_liana_out != jnp.uint8(0)
        # The release frame is not a `.skipSwingHarry` frame - atLiana is already
        # clear by the time it is reached - so Harry lets go from the column and row
        # the previous frame's placement left him on. `.skipJumpOff` is also above
        # the movement block, so that block does run on the release frame, and with
        # jumpIndex nonzero it reads the oldJoystick the release just wrote: one
        # pixel in the rope's direction, on the even NTSC frame of the two.
        x = jnp.where(releasing, state.player_x + release_vx, x)
        x = jnp.where(attached, swing_x, x)
        y = jnp.where(releasing, y_release, y)
        y = jnp.where(attached, swing_y, y)
        # That same movement block writes `ldy #REFLECT / sty reflectHarry` on the
        # branch it takes, so letting go also turns Harry to face the way he is
        # thrown. player_vx is what this port's facing and pose predicates read.
        vx = jnp.where(releasing, release_vx, vx)
        vx = jnp.where(attached, jnp.asarray(0.0, dtype=jnp.float32), vx)
        vy = jnp.where(releasing, y_release - state.player_y, vy)
        vy = jnp.where(attached, jnp.asarray(0.0, dtype=jnp.float32), vy)
        on_ground = jnp.where(attached, jnp.array(False, dtype=jnp.bool_), on_ground)
        on_ladder = jnp.where(attached, jnp.array(False, dtype=jnp.bool_), on_ladder)
        current_ground_y = jnp.where(attached, upper_ground, current_ground_y)
        jump_index = jnp.where(attached, jnp.int32(0), jump_index)
        jump_lock_active = jump_lock_active & (~attached)
        # `stx soundIdx` after `inx`, so the tarzan yell starts at index 1.
        sound_idx = jnp.where(
            (liana_grab_sub1 | liana_grab_sub2) & (~releasing), jnp.int32(1), sound_idx
        )

        # --- ROM running cadence (patIdHarry / frameCnt) --------------------
        # NTSC: move on frameCnt%4 in {0,2}; dec patId only when %4==0.
        # One JAX step is two Atari frames, so frame_cnt += 1 per step and the
        # pose advances when that counter is odd: once every 2 JAX steps, not 4.
        moving_h = move_left | move_right
        anim_tick = (frame_cnt & jnp.int32(1)) == jnp.int32(1)
        can_cycle = (
            gameplay_active
            & on_ground
            & (~on_ladder)
            & moving_h
            & anim_tick
        )
        pat_id_harry = state.pat_id_harry.astype(jnp.int32)
        pat_id_harry = jnp.where(can_cycle, pat_id_harry - jnp.int32(1), pat_id_harry)
        pat_id_harry = jnp.where(pat_id_harry < jnp.int32(0), jnp.int32(ID_RUNNING4), pat_id_harry)
        pat_id_harry = jnp.where(moving_h, pat_id_harry, jnp.int32(ID_STANDING))
        # --------------------------------------------------------------------

        # `jmp KilledHarry` lands on `jmp ProcessObjects`, and the collision test
        # it came from sits above both the horizontal-movement block and
        # `stx patIdHarry`, so the fatal frame leaves Harry's pose and heading
        # exactly as the frame before drew them. vx is left alone for that reason:
        # `x = x + vx` has already run this frame, and the frozen branch never
        # runs it again, so keeping it costs no motion and is what tells the
        # renderer which run frame and which way round to hold him.
        vy = jnp.where(hit_hazard, jnp.asarray(0.0, dtype=jnp.float32), vy)
        jump_index = jnp.where(hit_hazard, jnp.int32(0), jump_index)
        on_ladder = jnp.where(hit_hazard, jnp.array(False, dtype=jnp.bool_), on_ladder)

        next_hurt_cooldown = jnp.maximum(hurt_cooldown - jnp.int32(1), jnp.int32(0))
        next_hurt_cooldown = jnp.where(
            hit_scorpion,
            jnp.int32(consts.scorpion_hurt_cooldown_frames),
            next_hurt_cooldown,
        )
        next_hurt_cooldown = jnp.where(
            hit_other_hazard,
            jnp.int32(consts.snake_hurt_cooldown_frames),
            next_hurt_cooldown,
        )

        respawn_phase = jnp.where(hit_hazard, jnp.int32(1), state.respawn_phase)
        stored_respawn_target_ground_y = jnp.where(
            hit_hazard, current_ground_y, state.respawn_target_ground_y
        )

        done = (time_left <= 0)

        new_state = PitfallState(
            player_x=x,
            player_y=y,
            player_vx=vx,
            player_vy=vy,
            on_ground=on_ground,
            score=score,
            timer_started=timer_started,
            time_left=time_left,
            lives_left=lives_left,
            done=done,
            hurt_cooldown=next_hurt_cooldown,
            down_pressed=down_pressed,
            on_ladder=on_ladder,
            current_ground_y=current_ground_y,
            scorpion_x=scorpion_x,
            scorpion_facing_right=scorpion_facing_right,
            touching_wood=touching_wood,
            touching_rolling_wood=touching_rolling_wood,
            rolling_wood_contact_x=rolling_contact_x,
            climb_active=climb_active,
            climb_pos=climb_pos,
            pat_id_harry=pat_id_harry,
            frame_cnt=frame_cnt,
            # `ldy #REFLECT / sty reflectHarry` is inside the block atLiana skips,
            # so a swinging Harry keeps whichever way round he was when he caught
            # the rope.
            facing_left=jnp.where(
                attached,
                state.facing_left,
                jnp.where(
                    releasing,
                    release_vx < jnp.float32(0.0),
                    jnp.where(
                        move_left, jnp.array(True, dtype=jnp.bool_),
                        jnp.where(move_right, jnp.array(False, dtype=jnp.bool_), state.facing_left),
                    ),
                ),
            ),
            respawn_phase=respawn_phase,
            respawn_target_ground_y=stored_respawn_target_ground_y,
            no_game_scroll=no_game_scroll,
            sound_idx=sound_idx,
            sound_delay=state.sound_delay,
            restart_rom_y=restart_rom_y,

            jump_pressed_prev=jump_pressed,
            jump_lock_active=jump_lock_active,
            jump_lock_vx=jump_lock_vx,
            jump_index=jump_index,
            liana_pos_hi=liana_pos_hi,
            liana_pos_lo=liana_pos_lo,
            hmbl_add=hmbl_add,
            hmbl_dir=hmbl_dir,
            liana_bottom=liana_bottom,
            at_liana=at_liana_out,
            jump_mode=jump_mode,
            x_pos_quicksand=x_pos_quicksand,
            treasure_bits=treasure_bits,
            treasure_cnt=treasure_cnt,
            screen_id=new_screen_id,
            room_byte=new_room_byte,
        )

        # --- the frozen half of the frame ------------------------------------
        # `lda noGameScroll / beq .processHarry / jmp ProcessObjects` skips the
        # whole of .processHarry, so nothing below reads the joystick, moves
        # Harry, animates the quicksand or rolls the logs. One JAX step is two
        # NTSC frames, so the counters below tick twice.
        d_phase = state.respawn_phase
        d_scroll = state.no_game_scroll
        d_sound = state.sound_idx
        d_delay = state.sound_delay
        d_rom_y = state.restart_rom_y
        d_lives = state.lives_left
        d_x = state.player_x
        d_facing = state.facing_left
        d_scorpion = state.scorpion_x
        d_jump_index = jnp.int32(0)
        d_liana_hi = state.liana_pos_hi
        d_liana_lo = state.liana_pos_lo
        d_hmbl_add = state.hmbl_add
        d_hmbl_dir = state.hmbl_dir
        d_liana_bottom = state.liana_bottom
        d_jump_mode = state.jump_mode

        for _ in range(NTSC_FRAMES_PER_STEP):
            frozen = d_phase == jnp.int32(1)
            scroll_next, sound_next, delay_next, life_loss = advance_death_frame(
                d_scroll, d_sound, d_delay
            )
            d_scroll = jnp.where(frozen, scroll_next, d_scroll)
            d_sound = jnp.where(frozen, sound_next, d_sound)
            d_delay = jnp.where(frozen, delay_next, d_delay)

            # `lda livesPat / beq .slipDecrease`: livesPat is $a0 / $80 / $00 for
            # three, two and one life, so a death on the last life falls straight
            # through - no decrement, no restart, and noGameScroll left running.
            take_life = frozen & life_loss & (d_lives > jnp.int32(1))
            game_over = frozen & life_loss & (d_lives <= jnp.int32(1))

            # `cmp #71 / bcc LF5D2` picks the branch off the frozen yPosHarry.
            under = d_rom_y >= jnp.int32(RESTART_UNDERGROUND_TEST_Y)

            d_lives = jnp.where(take_life, d_lives - jnp.int32(1), d_lives)
            d_lives = jnp.where(game_over, jnp.int32(0), d_lives)
            d_scroll = jnp.where(take_life, jnp.int32(0), d_scroll)
            # `lda #NOREFLECT / sta reflectHarry`
            d_facing = jnp.where(take_life, jnp.array(False, dtype=jnp.bool_), d_facing)
            d_x = jnp.where(take_life, jnp.float32(RESTART_X), d_x)
            d_jump_index = jnp.where(take_life, jnp.int32(JUMP_LEN), d_jump_index)
            # `lda #SCREENWIDTH/2-4 / sta xPosScorpion`, underground branch only.
            d_scorpion = jnp.where(take_life & under, scorpion_spawn_x, d_scorpion)
            d_rom_y = jnp.where(
                take_life,
                jnp.where(under, jnp.int32(RESTART_Y_UNDER), jnp.int32(RESTART_Y_UPPER)),
                d_rom_y,
            )
            d_phase = jnp.where(take_life, jnp.int32(2), d_phase)

            # The restart frame is also the first frame Harry is processed again,
            # so JumpTab's trailing -1 already applies here. yPosHarry is a byte:
            # from 223 it counts through 255 and wraps to 0, which is the whole of
            # the pause before Harry drops back into view.
            dropping = d_phase == jnp.int32(2)
            d_rom_y = jnp.where(dropping, (d_rom_y + jnp.int32(1)) & jnp.int32(0xFF), d_rom_y)
            # `cpx #JUNGLE_GROUND / beq .stopJump` and its UNDER_GROUND twin.
            stopped = dropping & (
                (d_rom_y == jnp.int32(JUNGLE_GROUND)) | (d_rom_y == jnp.int32(UNDER_GROUND))
            )
            d_phase = jnp.where(stopped, jnp.int32(0), d_phase)
            d_jump_index = jnp.where(stopped, jnp.int32(0), d_jump_index)
            # `.stopJump` writes both bytes, so the restart drop's landing is also
            # what clears a jumpMode left over from a release Harry did not survive.
            d_jump_mode = jnp.where(stopped, jnp.uint8(0), d_jump_mode)

            # `.swingLiana` again: the restart drop clears noGameScroll, so from
            # the restart frame on .processHarry runs and the liana swings with
            # it. Only the frozen pause above it is skipped, and the oscillator
            # is never reset - it comes out of the death at the phase it went in
            # at, plus however many frames Harry was falling.
            liana_frame = d_phase != jnp.int32(1)
            swung = liana_swing_frame(d_liana_hi, d_liana_lo)
            d_liana_hi = jnp.where(liana_frame, swung[0], d_liana_hi)
            d_liana_lo = jnp.where(liana_frame, swung[1], d_liana_lo)
            d_hmbl_add = jnp.where(liana_frame, swung[2], d_hmbl_add)
            d_hmbl_dir = jnp.where(liana_frame, swung[3], d_hmbl_dir)
            d_liana_bottom = jnp.where(liana_frame, swung[4], d_liana_bottom)

        # yPosHarry mapped straight onto the raster, wrap included. The 223..255
        # run is not flagged as hidden anywhere: carried through the same
        # arithmetic it simply lands hundreds of rows below the screen, which is
        # the ROM's own situation - the kernel's Y counter never reaches Harry's
        # window there either. render_at_clipped drops it for the same reason the
        # kernel does, so nothing needs to know that a restart is in progress.
        restart_is_under = (d_rom_y > jnp.int32(JUNGLE_GROUND)) & (
            d_rom_y <= jnp.int32(UNDER_GROUND)
        )
        restart_ground = jnp.where(
            restart_is_under,
            jnp.asarray(consts.underground_y, dtype=jnp.float32),
            jnp.asarray(consts.ground_y, dtype=jnp.float32),
        )
        restart_datum = jnp.where(
            restart_is_under, jnp.float32(UNDER_GROUND), jnp.float32(JUNGLE_GROUND)
        )
        restart_y = restart_ground + (d_rom_y.astype(jnp.float32) - restart_datum)

        dropped = state.respawn_phase == jnp.int32(2)
        restarted = dropped | (d_phase == jnp.int32(2)) | (d_phase == jnp.int32(0))

        transition_state = PitfallState(
            player_x=jnp.where(restarted, d_x, state.player_x),
            player_y=jnp.where(restarted, restart_y, state.player_y),
            # Frozen means frozen, not reset: .processHarry is skipped whole, so
            # every byte describing Harry keeps the value the fatal frame left.
            # player_vx is what the renderer reads for facing and for the run
            # pose, so clearing it here would snap a dying Harry to standing.
            player_vx=jnp.where(restarted, jnp.float32(0.0), state.player_vx),
            player_vy=jnp.array(0.0, dtype=jnp.float32),
            on_ground=jnp.where(restarted, d_phase == jnp.int32(0), state.on_ground),
            score=state.score,
            timer_started=state.timer_started,
            time_left=state.time_left,
            lives_left=d_lives,
            done=state.done,
            hurt_cooldown=state.hurt_cooldown,
            down_pressed=jnp.array(False, dtype=jnp.bool_),
            on_ladder=jnp.array(False, dtype=jnp.bool_),
            current_ground_y=jnp.where(restarted, restart_ground, state.current_ground_y),
            scorpion_x=d_scorpion,
            scorpion_facing_right=state.scorpion_facing_right,
            touching_wood=jnp.array(False, dtype=jnp.bool_),
            touching_rolling_wood=jnp.array(False, dtype=jnp.bool_),
            rolling_wood_contact_x=jnp.array(0.0, dtype=jnp.float32),
            climb_active=jnp.array(False, dtype=jnp.bool_),
            climb_pos=jnp.int32(0),
            # `stx patIdHarry` lives past the collision check in .processHarry, so
            # the freeze never reaches it and the impact pose stands until the
            # restart puts Harry back on his feet.
            pat_id_harry=jnp.where(restarted, jnp.int32(ID_STANDING), state.pat_id_harry),
            frame_cnt=jnp.int32(0),
            facing_left=d_facing,
            respawn_phase=d_phase,
            respawn_target_ground_y=jnp.where(
                restarted, restart_ground, state.respawn_target_ground_y
            ),
            no_game_scroll=d_scroll,
            sound_idx=d_sound,
            sound_delay=d_delay,
            restart_rom_y=d_rom_y,

            jump_pressed_prev=state.jump_pressed_prev,
            jump_lock_active=jnp.array(False, dtype=jnp.bool_),
            jump_lock_vx=jnp.array(0.0, dtype=jnp.float32),
            jump_index=d_jump_index,
            liana_pos_hi=d_liana_hi,
            liana_pos_lo=d_liana_lo,
            hmbl_add=d_hmbl_add,
            hmbl_dir=d_hmbl_dir,
            liana_bottom=d_liana_bottom,
            at_liana=state.at_liana,
            jump_mode=d_jump_mode,
            # The frozen pause leaves xPosQuickSand exactly where the fatal
            # frame put it (`.doQuickSand` reads noGameScroll and skips the
            # update), and the restart drop cannot re-animate it because this
            # abstraction holds frameCnt for the whole sequence - so it is held.
            x_pos_quicksand=state.x_pos_quicksand,
            # treasureBits and treasureCnt are never touched by KilledHarry or
            # the restart; only Reset clears them. So they ride through death
            # unchanged and a collected treasure stays collected.
            treasure_bits=state.treasure_bits,
            treasure_cnt=state.treasure_cnt,
            screen_id=state.screen_id,
            room_byte=state.room_byte,
        )

        final_state = jax.tree.map(
            lambda normal_value, transition_value: jnp.where(transition_active, transition_value, normal_value),
            new_state,
            transition_state,
        )
        # livesPat only reaches zero on the game-over branch, and that branch
        # never restarts Harry, so this is the ROM's "game is stopped" state.
        # The 32nd treasure's `dec noGameScroll` is the other way the game stops.
        final_done = (final_state.time_left <= 0) | (final_state.lives_left <= 0) | game_won
        final_state = final_state.replace(done=final_done)

        obs = self._get_observation(final_state)
        reward = self._get_reward(state, final_state)
        info = self._get_info(final_state)

        return obs, final_state, reward, final_state.done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: PitfallState) -> PitfallObservation:
        has_logs, logs_are_rolling, log_count, log_xs, has_fire, has_snake = room_hazards_from_room_byte(state.room_byte)
        has_scorpion = has_scorpion_from_room_byte(state.room_byte)
        layout = self._screen_layout(state.room_byte)
        return PitfallObservation(
            player_x=state.player_x,
            player_y=state.player_y,
            screen_id=state.screen_id,
            room_byte=state.room_byte,
            current_ground_y=state.current_ground_y,
            on_ground=state.on_ground,
            on_ladder=state.on_ladder,
            facing_left=state.facing_left,
            scorpion_x=state.scorpion_x,
            has_scorpion=has_scorpion,
            has_fire=has_fire,
            has_snake=has_snake,
            has_logs=has_logs,
            log_count=log_count,
            log_xs=log_xs,
            logs_are_rolling=logs_are_rolling,
            has_ladder=layout.has_ladder,
            ladder_x=layout.ladder_x,
            has_wall=layout.has_wall,
            wall_x=layout.wall_x,
            wall_side=layout.wall_side,
            time_left=state.time_left,
            lives_left=state.lives_left,
            score=state.score,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: PitfallState) -> PitfallInfo:
        return PitfallInfo(
            time_left=state.time_left,
            lives_left=state.lives_left
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, prev: PitfallState, new: PitfallState) -> float:
        # Score-delta, the framework-wide convention: the treasure's ROM score
        # value (and the log drain and hole penalty) flow through automatically.
        return new.score - prev.score

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: PitfallState) -> bool:
        return (state.time_left <= 0) | (state.lives_left <= 0)

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(18)

    def observation_space(self) -> spaces.Dict:
        return spaces.Dict(
            {
                "player_x": spaces.Box(low=0.0, high=float(self.consts.screen_width - 1), shape=(), dtype=jnp.float32),
                "player_y": spaces.Box(low=0.0, high=float(self.consts.screen_height - 1), shape=(), dtype=jnp.float32),
                "screen_id": spaces.Box(low=0, high=254, shape=(), dtype=jnp.int32),
                "room_byte": spaces.Box(low=0, high=255, shape=(), dtype=jnp.uint8),
                "current_ground_y": spaces.Box(low=0.0, high=float(self.consts.screen_height - 1), shape=(), dtype=jnp.float32),
                "on_ground": spaces.Discrete(2),
                "on_ladder": spaces.Discrete(2),
                "facing_left": spaces.Discrete(2),
                "scorpion_x": spaces.Box(low=0.0, high=float(self.consts.screen_width - 1), shape=(), dtype=jnp.float32),
                "has_scorpion": spaces.Discrete(2),
                "has_fire": spaces.Discrete(2),
                "has_snake": spaces.Discrete(2),
                "has_logs": spaces.Discrete(2),
                "log_count": spaces.Box(low=0, high=3, shape=(), dtype=jnp.int32),
                "log_xs": spaces.Box(low=0, high=int(self.consts.screen_width - 1), shape=(3,), dtype=jnp.int32),
                "logs_are_rolling": spaces.Discrete(2),
                "has_ladder": spaces.Discrete(2),
                "ladder_x": spaces.Box(low=0, high=int(self.consts.screen_width - 1), shape=(), dtype=jnp.int32),
                "has_wall": spaces.Discrete(2),
                "wall_x": spaces.Box(low=0, high=int(self.consts.screen_width - 1), shape=(), dtype=jnp.int32),
                "wall_side": spaces.Box(low=-1, high=1, shape=(), dtype=jnp.int32),
                "time_left": spaces.Box(
                    low=0,
                    high=int(self.consts.initial_time_seconds * self.consts.fps),
                    shape=(),
                    dtype=jnp.int32,
                ),
                "lives_left": spaces.Box(low=0, high=int(self.consts.max_lives), shape=(), dtype=jnp.int32),
                "score": spaces.Box(low=0, high=jnp.iinfo(jnp.int32).max, shape=(), dtype=jnp.int32),
            }
        )

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(
                int(self.consts.screen_height),
                int(self.consts.screen_width),
                3,
            ),
            dtype=jnp.uint8,
        )

    def render(self, state: PitfallState) -> jnp.ndarray:
        return self.renderer.render(state)

    def _init_state(self) -> PitfallState:
        consts = self.consts
        state = PitfallState(
            player_x=jnp.array(consts.player_start_x, dtype=jnp.float32),
            player_y=jnp.array(consts.player_start_y, dtype=jnp.float32),
            player_vx=jnp.array(0.0, dtype=jnp.float32),
            player_vy=jnp.array(0.0, dtype=jnp.float32),
            on_ground=jnp.array(True, dtype=jnp.bool_),
            score=jnp.array(consts.initial_score, dtype=jnp.int32),
            timer_started=jnp.array(False, dtype=jnp.bool_),
            time_left=jnp.array(consts.initial_time_seconds * consts.fps, dtype=jnp.int32),
            lives_left=jnp.array(consts.max_lives, dtype=jnp.int32),
            done=jnp.array(False, dtype=jnp.bool_),
            hurt_cooldown=jnp.array(0, dtype=jnp.int32),
            down_pressed=jnp.array(False, dtype=jnp.bool_),
            on_ladder=jnp.array(False, dtype=jnp.bool_),
            current_ground_y=jnp.array(consts.ground_y, dtype=jnp.float32),
            scorpion_x=jnp.array(consts.scorpion_spawn_x, dtype=jnp.int32),
            scorpion_facing_right=jnp.array(True, dtype=jnp.bool_),
            touching_wood=jnp.array(False, dtype=jnp.bool_),
            touching_rolling_wood=jnp.array(False, dtype=jnp.bool_),
            rolling_wood_contact_x=jnp.array(0.0, dtype=jnp.float32),
            climb_active=jnp.array(False, dtype=jnp.bool_),
            climb_pos=jnp.int32(0),
            pat_id_harry=jnp.int32(ID_STANDING),
            frame_cnt=jnp.int32(0),
            facing_left=jnp.array(False, dtype=jnp.bool_),
            respawn_phase=jnp.array(0, dtype=jnp.int32),
            respawn_target_ground_y=jnp.array(consts.ground_y, dtype=jnp.float32),
            # InitGame leaves noGameScroll at $ff and waits for the first
            # joystick nudge to clear it; this port starts on the running game,
            # so it starts where `stx noGameScroll` would have left it.
            no_game_scroll=jnp.array(0, dtype=jnp.int32),
            sound_idx=jnp.array(0, dtype=jnp.int32),
            sound_delay=jnp.array(0, dtype=jnp.int32),
            restart_rom_y=jnp.array(JUNGLE_GROUND, dtype=jnp.int32),
            jump_pressed_prev=jnp.array(False, dtype=jnp.bool_),
            jump_lock_active=jnp.array(False, dtype=jnp.bool_),
            jump_lock_vx=jnp.array(0.0, dtype=jnp.float32),
            jump_index=jnp.int32(0),
            # Cleared RAM. Reset zeroes the whole page and nothing in the game
            # ever writes these seven outside `.swingLiana` and the grab and
            # release, so a new game always starts the swing from a dead centre.
            liana_pos_hi=jnp.array(0, dtype=jnp.uint8),
            liana_pos_lo=jnp.array(0, dtype=jnp.uint8),
            hmbl_add=jnp.array(0, dtype=jnp.uint8),
            hmbl_dir=jnp.array(0, dtype=jnp.uint8),
            liana_bottom=jnp.array(0, dtype=jnp.uint8),
            at_liana=jnp.array(0, dtype=jnp.uint8),
            jump_mode=jnp.array(0, dtype=jnp.uint8),
            # Cleared RAM: Reset zeroes the whole page, and the first scene
            # (SEED $c4 -> sceneType 0) is not a quicksand scene anyway, so the
            # first MainLoop would write 0 regardless.
            x_pos_quicksand=jnp.array(0, dtype=jnp.uint8),
            # Cleared RAM: Reset zeroes treasureBits, and InitGame writes
            # `lda #31 / sta treasureCnt` (32 treasures, counted down to -1).
            treasure_bits=jnp.zeros((4,), dtype=jnp.uint8),
            treasure_cnt=jnp.array(TREASURE_COUNT_INIT, dtype=jnp.int32),
            screen_id=jnp.array(0, dtype=jnp.int32),
            room_byte=jnp.array(SEED, dtype=jnp.uint8),
        )
        return state
    

class PitfallRenderer(JAXGameRenderer):
    """Pitfall renderer using the shared raster+palette pipeline."""

    def __init__(
        self,
        consts: PitfallConstants | None = None,
        config: render_utils.RendererConfig | None = None,
        ladder_x_px: chex.Array | None = None,
        left_wall_x_px: chex.Array | None = None,
        right_wall_x_px: chex.Array | None = None,
    ):
        super().__init__()
        self.consts = consts or PitfallConstants()

        screen_w = int(self.consts.screen_width)
        wall_w = int(self.consts.tunnel_wall_width)

        def _clamp_wall_x(x: int) -> int:
            return max(0, min(screen_w - wall_w, x))

        if left_wall_x_px is None:
            left_wall_x_px = jnp.array(_clamp_wall_x(23), dtype=jnp.int32)
        else:
            left_wall_x_px = jnp.asarray(left_wall_x_px, dtype=jnp.int32)

        if right_wall_x_px is None:
            right_wall_x_px = jnp.array(_clamp_wall_x(132), dtype=jnp.int32)
        else:
            right_wall_x_px = jnp.asarray(right_wall_x_px, dtype=jnp.int32)

        if ladder_x_px is None:
            # Same derivation as JaxPitfall.__init__: put the drawn opening on
            # the ROM's single-hole bounds.
            ladder_x_default = int(self.consts.hole_bounds_tab[0][0][0]) - int(self.consts.ladder_opening_inset)
            ladder_x_default = max(0, min(screen_w - int(self.consts.ladder_width), ladder_x_default))
            ladder_x_px = jnp.array(ladder_x_default, dtype=jnp.int32)
        else:
            ladder_x_px = jnp.asarray(ladder_x_px, dtype=jnp.int32)

        self.ladder_x_px = ladder_x_px
        self.left_wall_x_px = left_wall_x_px
        self.right_wall_x_px = right_wall_x_px

        self.config = config or render_utils.RendererConfig(
            game_dimensions=(self.consts.screen_height, self.consts.screen_width),
            channels=3,
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)

        sprite_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sprites', 'pitfall')
        asset_config = list(self.consts.ASSET_CONFIG)

        h = int(self.consts.screen_height)
        w = int(self.consts.screen_width)

        bg = jnp.zeros((h, w, 4), dtype=jnp.uint8)
        bg = bg.at[:, :, 3].set(255)
        ground = int(self.consts.ground_y)
        underground = underground_floor_row(self.consts.underground_y)
        bg = bg.at[ground:ground + 2, :, 1].set(200)
        bg = bg.at[underground:underground + 2, :, 1].set(120)

        asset_config = [
            {'name': 'background', 'type': 'background', 'data': bg},
            *[a for a in asset_config if a.get('type') != 'background'],
        ]

        def _normalize_to_rgba_u8(image: np.ndarray, asset_name: str) -> np.ndarray:
            if image.ndim == 2:
                alpha = np.where(image > 0, np.uint8(255), np.uint8(0))
                rgb = np.stack([image, image, image], axis=2).astype(np.uint8)
                return np.concatenate([rgb, alpha[:, :, None]], axis=2).astype(np.uint8)
            if image.ndim == 3 and image.shape[2] == 3:
                image_u8 = image.astype(np.uint8)
                alpha = np.where(np.any(image_u8 != 0, axis=2), np.uint8(255), np.uint8(0))
                return np.concatenate([image_u8, alpha[:, :, None]], axis=2).astype(np.uint8)
            if image.ndim == 3 and image.shape[2] == 4:
                return image.astype(np.uint8)
            raise ValueError(f"Unsupported backdrop format for {asset_name}: shape={image.shape}")

        def _cleanup_leading_black_edges(backdrop_rgba: np.ndarray, max_strip: int = 16) -> np.ndarray:
            rgb = backdrop_rgba[:, :, :3]
            h_img, w_img = rgb.shape[:2]

            top_limit = min(max_strip, h_img)
            left_limit = min(max_strip, w_img)

            top_strip = 0
            for y in range(top_limit):
                if np.all(rgb[y] == 0):
                    top_strip += 1
                else:
                    break

            left_strip = 0
            for x in range(left_limit):
                if np.all(rgb[:, x] == 0):
                    left_strip += 1
                else:
                    break

            cleaned = backdrop_rgba.copy()
            if 0 < top_strip < h_img:
                cleaned[:top_strip, :, :] = cleaned[top_strip:top_strip + 1, :, :]
            if 0 < left_strip < w_img:
                cleaned[:, :left_strip, :] = cleaned[:, left_strip:left_strip + 1, :]
            return cleaned

        def _load_fullscreen_backdrop(asset_name: str, file_name: str) -> dict | None:
            file_path = os.path.join(sprite_path, file_name)
            if not os.path.exists(file_path):
                return None
            backdrop_np = np.load(file_path)
            backdrop_rgba = _normalize_to_rgba_u8(backdrop_np, asset_name)

            bh, bw = int(backdrop_rgba.shape[0]), int(backdrop_rgba.shape[1])
            if (bh, bw) != (h, w):
                raise ValueError(f"{asset_name} has unexpected size {(bh, bw)}; expected {(h, w)}")

            backdrop_rgba = _cleanup_leading_black_edges(backdrop_rgba)
            return {
                'name': asset_name,
                'type': 'single',
                'data': jnp.asarray(backdrop_rgba, dtype=jnp.uint8),
            }

        def _load_trimmed_sprite(
            file_name: str,
            crop_box: tuple[int, int, int, int],
            black_transparent: bool = True,
        ) -> jnp.ndarray:
            """Load a full-screen sprite .npy, crop to *crop_box* (y0,x0,y1,x1),
            optionally mark black pixels transparent, and return RGBA."""
            file_path = os.path.join(sprite_path, file_name)
            img_np = np.load(file_path)
            img_rgba = _normalize_to_rgba_u8(img_np, file_name)
            y0, x0, y1, x1 = crop_box
            cropped = img_rgba[y0:y1, x0:x1].copy()
            if black_transparent:
                is_black = np.all(cropped[:, :, :3] == 0, axis=2)
                cropped[is_black, 3] = 0
            return jnp.asarray(cropped, dtype=jnp.uint8)

        # Load trimmed log / ladder sprites (bounding boxes measured from sprite files)
        log_left_rgba = _load_trimmed_sprite('log_left.npy', (118, 25, 132, 31))
        log_right_rgba = _load_trimmed_sprite('log_right.npy', (119, 24, 133, 30))
        ladder_rgba = _load_trimmed_sprite('ladder.npy', (117, 72, 178, 88), black_transparent=False)
        ladder_with_pits_rgba = _load_trimmed_sprite(
            'ladder_with_pits.npy', (119, 46, 178, 114), black_transparent=False,
        )

        converted_asset_config = []
        for asset in asset_config:
            if (
                asset.get('type') == 'single'
                and asset.get('name') in {
                    'background_tree_variant_0',
                    'background_tree_variant_1',
                    'background_tree_variant_2',
                    'background_tree_variant_3',
                }
                and 'file' in asset
            ):
                converted_backdrop = _load_fullscreen_backdrop(asset['name'], asset['file'])
                if converted_backdrop is not None:
                    converted_asset_config.append(converted_backdrop)
            elif asset.get('name') == 'cobra' and asset.get('type') == 'group' and 'files' in asset:
                # Full-screen captures: crop the 8x14 GRP1 box at xPosObject=124.
                # Keep black (the cobra body); do not treat it as transparent.
                cobra_crop = (118, 124, 132, 132)
                converted_asset_config.append(
                    {
                        'name': 'cobra',
                        'type': 'group',
                        'data': [
                            _load_trimmed_sprite(file_name, cobra_crop, black_transparent=False)
                            for file_name in asset['files']
                        ],
                    }
                )
            else:
                converted_asset_config.append(asset)
        asset_config = converted_asset_config

        def _color_swatch(rgb: tuple[int, int, int]) -> jnp.ndarray:
            return jnp.array([rgb[0], rgb[1], rgb[2], 255], dtype=jnp.uint8).reshape(1, 1, 4)

        asset_config.extend(
            [
                {'name': 'color_wood', 'type': 'procedural', 'data': _color_swatch((110, 70, 25))},
                {'name': 'color_snake', 'type': 'procedural', 'data': _color_swatch((20, 200, 0))},
                # ColorTab+8, the swamp colour, is ROM `BLUE`. This is its ALE
                # rendering, read out of croc1.npy at rows 125-126 where
                # Croco1's sparse open-jaw rows let COLUBK show through in a
                # BLUEPIT scene; it is the only blue in the sprite set.
                {'name': 'color_swamp', 'type': 'procedural', 'data': _color_swatch((45, 109, 152))},
                # ScorpionColor is WHITE for every row of the pattern.
                {'name': 'color_scorpion', 'type': 'procedural', 'data': _color_swatch((255, 255, 255))},
                # CrocoColor is DARK_GREEN-2 = $d0 for every one of the nine
                # occupied rows. (20, 60, 0) is its ALE rendering, verified
                # against the croc0/croc1 captures, whose crocodile pixels are
                # exactly this colour.
                {'name': 'color_croc', 'type': 'procedural', 'data': _color_swatch((20, 60, 0))},
                # The liana is the ball, and the ball draws in COLUPF. Kernel 2's
                # opening line loads that from `colorLst+5`, ColorTab+5, which is
                # ROM BROWN-2 = $10. The palette is not guessed: the log sprite's
                # own pixels are (105, 105, 15), which is $12 - ROM BROWN, one
                # luminance step up - so this set renders $10 as (72, 72, 0).
                {'name': 'color_liana', 'type': 'procedural', 'data': _color_swatch((72, 72, 0))},
                {'name': 'color_hole', 'type': 'procedural', 'data': _color_swatch((0, 0, 0))},
                # The fire and treasure per-row colors, one swatch per Atari
                # color byte used by FireColor / the treasure color tables.
                *[{'name': f'objcol_{k:02x}', 'type': 'procedural', 'data': _color_swatch(v)}
                  for k, v in OBJ_COLOR_RGB.items()],
                {'name': 'log_left_sprite', 'type': 'single', 'data': log_left_rgba},
                {'name': 'log_right_sprite', 'type': 'single', 'data': log_right_rgba},
                {'name': 'ladder_sprite', 'type': 'single', 'data': ladder_rgba},
                {'name': 'ladder_with_pits_sprite', 'type': 'single', 'data': ladder_with_pits_rgba},
            ]
        )

        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS,
        ) = self.jr.load_and_setup_assets(asset_config, sprite_path)

        self.WOOD_ID = self.SHAPE_MASKS['color_wood'][0, 0].astype(self.BACKGROUND.dtype)
        self.SNAKE_ID = self.SHAPE_MASKS['color_snake'][0, 0].astype(self.BACKGROUND.dtype)
        self.SCORPION_ID = self.SHAPE_MASKS['color_scorpion'][0, 0].astype(self.BACKGROUND.dtype)
        self.HOLE_ID = self.SHAPE_MASKS['color_hole'][0, 0].astype(self.BACKGROUND.dtype)
        self.SWAMP_ID = self.SHAPE_MASKS['color_swamp'][0, 0].astype(self.BACKGROUND.dtype)
        self.CROC_ID = self.SHAPE_MASKS['color_croc'][0, 0].astype(self.BACKGROUND.dtype)
        self.PIT_BAND_TOP = int(pit_band_top_row(self.consts.ground_y))
        self.object_box_top_px = jnp.int32(object_box_top_row(self.consts.ground_y))

        # Fire and the four treasures: masks generated from the ROM patterns and
        # per-row color tables (no captures exist for these, and none are needed
        # - the pattern bytes are the whole artwork). The palette-id masks drive
        # the raster; the boolean lit masks (FIRE_BITMAPS, TREASURE_BITMAPS) are
        # the same pixels for CXPPMM, so the drawn and collidable shapes agree.
        obj_color_id = {
            k: int(self.SHAPE_MASKS[f'objcol_{k:02x}'][0, 0]) for k in OBJ_COLOR_RGB
        }
        tid = int(self.jr.TRANSPARENT_ID)

        def _obj_mask(pattern, color_rows):
            return object_band_bitmap(pattern, color_rows, obj_color_id, tid)

        fire_masks = np.stack([_obj_mask(p, FIRE_COLOR_ROWS) for p in FIRE_PATTERNS])
        self.FIRE_MASKS = jnp.asarray(fire_masks, dtype=self.BACKGROUND.dtype)
        self.FIRE_BITMAPS = jnp.asarray(fire_masks != tid)

        treasure_masks = np.stack([
            np.stack([
                _obj_mask(p, TREASURE_COLOR_ROWS[t])
                for p in (TREASURE_PATTERNS[t] if TREASURE_ANIMATED[t] else (TREASURE_PATTERNS[t], TREASURE_PATTERNS[t]))
            ])
            for t in range(4)
        ])
        self.TREASURE_MASKS = jnp.asarray(treasure_masks, dtype=self.BACKGROUND.dtype)
        self.TREASURE_BITMAPS = jnp.asarray(treasure_masks != tid)

        transparent_pixel = jnp.full((1, 1), int(self.jr.TRANSPARENT_ID), dtype=self.BACKGROUND.dtype)
        self.BACKGROUND_TREE_VARIANT_0 = self.SHAPE_MASKS.get('background_tree_variant_0', transparent_pixel)
        self.BACKGROUND_TREE_VARIANT_1 = self.SHAPE_MASKS.get('background_tree_variant_1', self.BACKGROUND_TREE_VARIANT_0)
        self.BACKGROUND_TREE_VARIANT_2 = self.SHAPE_MASKS.get('background_tree_variant_2', self.BACKGROUND_TREE_VARIANT_1)
        self.BACKGROUND_TREE_VARIANT_3 = self.SHAPE_MASKS.get('background_tree_variant_3', self.BACKGROUND_TREE_VARIANT_2)
        # --- The wall, generated from `Wall` and `WallColor` ------------------
        # The capture was 29 rows and the old code reached 32 by prepending three
        # of its own rows, which put a duplicated brick course a third of the way
        # down. Building from the literals gives the two sixteen-row kernel copies
        # exactly, so the pattern carries across the Kernel 8/9 boundary on its
        # own. wall.npy is still read, but only to pick up the two palette ids the
        # rest of the port already uses for this sprite.
        wall_capture = self.SHAPE_MASKS.get('wall', transparent_pixel)
        wall_capture = wall_capture[0] if wall_capture.ndim == 3 else wall_capture
        wall_capture_np = np.asarray(wall_capture)
        tid_wall = int(self.jr.TRANSPARENT_ID)
        wall_red_id = int(self.COLOR_TO_ID.get((167, 26, 26), int(self.WOOD_ID)))
        # A `$fe` row is a full mortar course, so the capture's own top row holds
        # the grey; taking it from there keeps the wall on the accepted palette.
        capture_lit = wall_capture_np[wall_capture_np != tid_wall]
        wall_grey_id = int(capture_lit[0]) if capture_lit.size else wall_red_id

        wall_rows_np = np.full(
            (len(wall_render_rows()), WALL_W), tid_wall, dtype=wall_capture_np.dtype
        )
        for row_index, (pattern_byte, is_grey) in enumerate(wall_render_rows()):
            colour = wall_grey_id if is_grey else wall_red_id
            for col in range(WALL_W):
                if (pattern_byte >> (WALL_W - 1 - col)) & 1:
                    wall_rows_np[row_index, col] = colour
        self.WALL_RENDER_MASK = jnp.asarray(wall_rows_np)
        self.WALL_MASK = self.WALL_RENDER_MASK

        # Log / ladder sprite masks (trimmed from full-screen captures)
        _log_left = self.SHAPE_MASKS.get('log_left_sprite', transparent_pixel)
        self.LOG_LEFT_MASK = _log_left[0] if _log_left.ndim == 3 else _log_left
        _log_right = self.SHAPE_MASKS.get('log_right_sprite', transparent_pixel)
        self.LOG_RIGHT_MASK = _log_right[0] if _log_right.ndim == 3 else _log_right
        _ladder = self.SHAPE_MASKS.get('ladder_sprite', transparent_pixel)
        self.LADDER_SPRITE_MASK = _ladder[0] if _ladder.ndim == 3 else _ladder
        _lwp = self.SHAPE_MASKS.get('ladder_with_pits_sprite', transparent_pixel)
        self.LADDER_WITH_PITS_MASK = _lwp[0] if _lwp.ndim == 3 else _lwp

        self.LIANA_ID = self.SHAPE_MASKS['color_liana'][0, 0].astype(self.BACKGROUND.dtype)
        # Kernel 1's COLUPF is ColorTab+2, the same register value the leaves
        # playfield draws in, so the green section's colour is the canopy's own
        # palette entry: (53, 95, 24) in the captured backdrop, ROM DARK_GREEN.
        # The bare-background test the priority mask needs is the jungle's
        # (110, 156, 66), ROM GREEN, also taken from the backdrop's palette.
        self.LIANA_GREEN_ID = jnp.asarray(
            int(self.COLOR_TO_ID.get((53, 95, 24), int(self.LIANA_ID))),
            dtype=self.BACKGROUND.dtype,
        )
        self.JUNGLE_BG_ID = jnp.asarray(
            int(self.COLOR_TO_ID.get((110, 156, 66), -1)),
            dtype=self.BACKGROUND.dtype,
        )

        # --- Cobra: replay each capture inside the ROM's 16-row GRP1 box ------
        # The captures are pixel-exact: alpha carries the pattern and RGB
        # carries CobraColor. They hold only the fourteen occupied rows, so the
        # two empty box rows have to be put back before the raster and CXPPMM
        # can agree on a coordinate. The comparison against COBRA_PATTERNS is
        # the load-time guarantee that drawn pixels are collidable pixels; a
        # swapped, resized or repainted asset stops the environment here rather
        # than quietly making the cobra unhittable.
        cobra_masks = self.SHAPE_MASKS.get('cobra', transparent_pixel)
        cobra_masks = cobra_masks[None, :, :] if cobra_masks.ndim == 2 else cobra_masks
        cobra_tid = int(self.jr.TRANSPARENT_ID)
        cobra_rows = np.flatnonzero(cobra_pattern_bitmap(0).any(axis=1))
        cobra_captures = np.array(cobra_masks)
        if cobra_captures.shape[0] != len(COBRA_PATTERNS):
            raise ValueError(
                f"cobra: {cobra_captures.shape[0]} frames loaded, "
                f"{len(COBRA_PATTERNS)} ROM patterns"
            )

        cobra_boxes = []
        for index, captured in enumerate(cobra_captures):
            if captured.shape != (cobra_rows.size, COBRA_W):
                raise ValueError(
                    f"cobra frame {index}: capture is {captured.shape}, "
                    f"ROM Cobra{index} occupies {(cobra_rows.size, COBRA_W)} "
                    f"of its {COBRA_H}-row box"
                )
            box = np.full((COBRA_H, COBRA_W), cobra_tid, dtype=captured.dtype)
            box[cobra_rows[0]:cobra_rows[-1] + 1] = captured
            if not np.array_equal(box != cobra_tid, cobra_pattern_bitmap(index)):
                raise ValueError(
                    f"cobra frame {index} does not draw ROM Cobra{index}; the "
                    "raster and CXPPMM would use different pixels"
                )
            cobra_boxes.append(box)

        self.COBRA_MASKS = jnp.asarray(np.stack(cobra_boxes), dtype=self.BACKGROUND.dtype)
        self.COBRA_BOX_TOP = jnp.int32(cobra_box_top_row(self.consts.ground_y))
        # ASM: objPatPtr = Cobra0 + (random2 & OBJECT_H). OBJECT_H=16 is bit 4.
        # random2 is the 8-bit LFSR stepped once per running frame from seed 1.
        r2 = 1
        anim_bits = []
        for _ in range(63):
            anim_bits.append(1 if (r2 & 16) else 0)
            r2 = (((r2 << 1) | (((r2 >> 6) ^ (r2 >> 7)) & 1)) & 0xFF)
        self.COBRA_ANIM_BIT4 = jnp.array(anim_bits, dtype=jnp.int32)
        self.COBRA_ANIM_PERIOD = jnp.int32(len(anim_bits))

        def _ensure_3d(mask_stack: jnp.ndarray) -> jnp.ndarray:
            return mask_stack[None, :, :] if mask_stack.ndim == 2 else mask_stack

        # --- Harry: rebuild every pose inside the ROM's 8x22 player box -------
        # The captures are 2x horizontally and 1x vertically, and each .npy is
        # already cropped to its own artwork, so the ROM's blank rows and
        # columns are missing. Halving the width and replaying the pattern's
        # own bounds puts every pose back where the hardware drew it. A uniform
        # rescale cannot do this: it would resample the vertical axis as well
        # and would still leave every pose bottom-aligned on a shared canvas.
        tid_np = int(self.jr.TRANSPARENT_ID)

        def _trim_to_artwork(mask2d: np.ndarray) -> np.ndarray:
            occupied = mask2d != tid_np
            rows = np.flatnonzero(occupied.any(axis=1))
            cols = np.flatnonzero(occupied.any(axis=0))
            return mask2d[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

        def _halve_width(mask2d: np.ndarray, label: str) -> np.ndarray:
            """Undo the 2x capture without interpolating: palette IDs are ids.

            Every captured column must be one half of an identical pair, which
            is what makes dropping the odd columns lossless.
            """
            if mask2d.shape[1] % 2 != 0:
                raise ValueError(f"{label}: capture width {mask2d.shape[1]} is odd")
            even, odd = mask2d[:, 0::2], mask2d[:, 1::2]
            if not np.array_equal(even, odd):
                raise ValueError(f"{label}: capture columns are not doubled pairs")
            return even

        def _place_in_rom_box(mask2d: np.ndarray, pattern_index: int, label: str) -> np.ndarray:
            row0, row1, col0, col1 = harry_pattern_bounds(pattern_index)
            expected = (row1 - row0 + 1, col1 - col0 + 1)
            if mask2d.shape != expected:
                raise ValueError(
                    f"{label}: artwork is {mask2d.shape}, ROM pattern "
                    f"{pattern_index} occupies {expected}"
                )
            canvas = np.full((HARRY_H, HARRY_W), tid_np, dtype=np.uint8)
            canvas[row0:row1 + 1, col0:col1 + 1] = mask2d
            return canvas

        # HarryPtrTab order: the groups already concatenate to Harry0..Harry8.
        harry_groups = (('harry_run', 5), ('harry_idle', 1), ('harry_swing', 1), ('harry_climb', 2))
        harry_boxes = []
        for group_name, frame_count in harry_groups:
            frames = np.array(_ensure_3d(self.SHAPE_MASKS[group_name]))
            if int(frames.shape[0]) != frame_count:
                raise ValueError(f"{group_name}: expected {frame_count} frames, got {frames.shape[0]}")
            for frame_index, frame in enumerate(frames):
                pattern_index = len(harry_boxes)
                label = f"{group_name}[{frame_index}] (Harry{pattern_index})"
                artwork = _halve_width(_trim_to_artwork(frame), label)
                harry_boxes.append(_place_in_rom_box(artwork, pattern_index, label))

        self.HARRY_PAT_MASKS = jnp.asarray(np.stack(harry_boxes, axis=0), dtype=jnp.uint8)
        # Named views for the poses other code still asks for by role. There are
        # no per-pose draw anchors: player_x is the box's left edge, so every
        # pose is drawn at the same place and the artwork moves inside the box,
        # exactly as the hardware does it.
        self.HARRY_RUN_MASKS = self.HARRY_PAT_MASKS[0:5]
        self.HARRY_IDLE_MASKS = self.HARRY_PAT_MASKS[5:6]
        self.HARRY_SWING_MASKS = self.HARRY_PAT_MASKS[6:7]
        self.HARRY_CLIMB_MASKS = self.HARRY_PAT_MASKS[7:9]

        # --- Scorpion: built from the ROM patterns, not from a capture --------
        # ScorpionColor is WHITE for every one of its rows, so the sprite has no
        # colour information to recover; the only thing a capture could supply
        # is shape, and three of the four scorpion grabs have no duplicated row
        # or column pairs, so they were never clean 2x captures. Synthesising
        # the mask from SCORPION_PATTERNS is both exact and simpler, and the
        # facing comes from reflecting in place the way REFP1 does.
        self.SCORPION_MASKS = jnp.stack(
            [
                jnp.where(
                    jnp.asarray(scorpion_pattern_bitmap(i)),
                    jnp.asarray(self.SCORPION_ID, dtype=self.BACKGROUND.dtype),
                    jnp.asarray(int(self.jr.TRANSPARENT_ID), dtype=self.BACKGROUND.dtype),
                )
                for i in range(len(SCORPION_PATTERNS))
            ]
        )
        self.SCORPION_BOX_TOP = jnp.int32(scorpion_box_top_row(self.consts.underground_y))

        # --- Crocodiles: built from Croco0/Croco1, not from a capture --------
        # CrocoColor is DARK_GREEN-2 for every occupied row, so like the
        # scorpion the ROM pattern is the whole artwork. The croc0/croc1
        # captures verified the shapes, the three NUSIZ copies at 60/76/92 and
        # the colour; they are not loaded as assets. Each frame is the fifteen
        # rows the ground band draws (pattern rows 14..0), so the mask's top row
        # sits on the band's first line.
        self.CROCO_MASKS = jnp.stack(
            [
                jnp.where(
                    jnp.asarray(croc_band_bitmap(i)),
                    jnp.asarray(self.CROC_ID, dtype=self.BACKGROUND.dtype),
                    jnp.asarray(int(self.jr.TRANSPARENT_ID), dtype=self.BACKGROUND.dtype),
                )
                for i in range(len(CROCO_PATTERNS))
            ]
        )
        self.TREE_VARIANT_TO_ASSET_IDX = jnp.array([0, 1, 2, 3], dtype=jnp.int32)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: PitfallState) -> jnp.ndarray:
        raster = self.jr.create_object_raster(self.BACKGROUND)

        rb = state.room_byte.astype(jnp.uint8)
        pt = pit_code_u8(rb)
        tree_variant = tree_variant_u8(rb)

        tree_bg_asset_idx = self.TREE_VARIANT_TO_ASSET_IDX[tree_variant.astype(jnp.int32)]

        def _render_tree_variant_0(r: jnp.ndarray) -> jnp.ndarray:
            return self.jr.render_at_clipped(
                r,
                jnp.int32(0),
                jnp.int32(0),
                self.BACKGROUND_TREE_VARIANT_0,
                flip_horizontal=jnp.array(False, dtype=jnp.bool_),
                flip_offset=jnp.array([0, 0], dtype=jnp.int32),
            )

        def _render_tree_variant_1(r: jnp.ndarray) -> jnp.ndarray:
            return self.jr.render_at_clipped(
                r,
                jnp.int32(0),
                jnp.int32(0),
                self.BACKGROUND_TREE_VARIANT_1,
                flip_horizontal=jnp.array(False, dtype=jnp.bool_),
                flip_offset=jnp.array([0, 0], dtype=jnp.int32),
            )

        def _render_tree_variant_2(r: jnp.ndarray) -> jnp.ndarray:
            return self.jr.render_at_clipped(
                r,
                jnp.int32(0),
                jnp.int32(0),
                self.BACKGROUND_TREE_VARIANT_2,
                flip_horizontal=jnp.array(False, dtype=jnp.bool_),
                flip_offset=jnp.array([0, 0], dtype=jnp.int32),
            )

        def _render_tree_variant_3(r: jnp.ndarray) -> jnp.ndarray:
            return self.jr.render_at_clipped(
                r,
                jnp.int32(0),
                jnp.int32(0),
                self.BACKGROUND_TREE_VARIANT_3,
                flip_horizontal=jnp.array(False, dtype=jnp.bool_),
                flip_offset=jnp.array([0, 0], dtype=jnp.int32),
            )

        raster = lax.switch(
            tree_bg_asset_idx,
            (
                _render_tree_variant_0,
                _render_tree_variant_1,
                _render_tree_variant_2,
                _render_tree_variant_3,
            ),
            raster,
        )

        # --- The pit band: sceneType 2..7 (tar, swamp, crocodiles, quicksand) --
        # PF2Lst carves the pit out of the jungle floor, so only the columns PF2
        # leaves clear are repainted and the ground the playfield still covers is
        # left exactly as the floor artwork drew it. LadderTab's bit 7 picks what
        # shows through: `lda LadderTab,x / bpl .noPit` keeps ColorTab+8 (BLUE)
        # for the BLUEPIT scenes (swamp, crocodile swamp, blue quicksand), while
        # the BLACKPIT scenes (tar, both black quicksands) fall through to `lda
        # colorLst+4 / sta colorLst+8` and show BLACK, the holes' own colour.
        #
        # The opening's shape is the same PF2PatTab window the collision bounds
        # read: static scenes sit on `Pit` (window 16) and the quicksand slides
        # the window to 16+y as xPosQuickSand moves. PIT_OPEN_MASKS holds the
        # decoded band for every reachable window, so the drawn opening and the
        # falling bounds are always the same bytes.
        in_pit = pt >= jnp.uint8(2)
        pit_color = jnp.where(pit_is_blue(rb), self.SWAMP_ID, self.HOLE_ID).astype(raster.dtype)
        pit_open_mask = PIT_OPEN_MASKS[quicksand_window_index(state.x_pos_quicksand)]

        def _draw_pit(r: jnp.ndarray) -> jnp.ndarray:
            top = self.PIT_BAND_TOP
            band = r[top:top + PIT_BAND_H]
            return r.at[top:top + PIT_BAND_H].set(
                jnp.where(pit_open_mask, pit_color, band)
            )

        raster = lax.cond(in_pit, _draw_pit, lambda r: r, raster)

        # --- Crocodiles: sceneType 4 ------------------------------------------
        # GRP1 with NUSIZ1 = THREE_COPIES off xPosObject = 60, so the three
        # crocodiles sit at 60, 76 and 92 and are drawn in the ground band over
        # the water. The pattern is Croco0 while frameCnt bit 7 is clear (jaws
        # open) and Croco1 while it is set, and the bounds loop reads the same
        # bit - so the mouth Harry sees is the mouth that can swallow him. With
        # CTRLPF = %001 in this kernel the players sit above the playfield, so
        # the crocodiles draw over both the ground and the water.
        has_crocs = pt == jnp.uint8(CROCO_SCENE)
        croc_open = croc_jaws_open(jnp.int32(2) * state.frame_cnt)
        croc_mask = self.CROCO_MASKS[jnp.where(croc_open, jnp.int32(0), jnp.int32(1))]

        def _draw_crocs(r: jnp.ndarray) -> jnp.ndarray:
            for cx in CROCO_XS:
                r = self.jr.render_at_clipped(
                    r,
                    jnp.int32(cx),
                    jnp.int32(self.PIT_BAND_TOP),
                    croc_mask,
                    flip_horizontal=jnp.array(False, dtype=jnp.bool_),
                    flip_offset=jnp.array([0, 0], dtype=jnp.int32),
                )
            return r

        raster = lax.cond(has_crocs, _draw_crocs, lambda r: r, raster)

        has_ladder = (pt == jnp.uint8(0)) | (pt == jnp.uint8(1))
        has_scorpion = has_scorpion_from_room_byte(rb)

        # ---- Underground elements: holes + ladder sprites ----
        has_simple_ladder = pt == jnp.uint8(0)
        has_ladder_with_pits = pt == jnp.uint8(1)

        ladder_x = self.ladder_x_px.astype(jnp.int32)
        ladder_w = jnp.int32(self.consts.ladder_width)
        hole_w = jnp.int32(self.consts.hole_width)
        hole_top = jnp.int32(int(self.consts.ground_y))
        # The shaft fill is a raster rect, so it ends on Kernel 9's last line.
        hole_h = jnp.int32(
            max(0, underground_last_row(self.consts.underground_y) + 1 - int(self.consts.ground_y))
        )

        # Center hole for standalone ladder – fill shaft below ground,
        # then the ladder sprite overlays with rungs + hole opening.
        ladder_hole_pos = jnp.where(
            has_simple_ladder,
            jnp.array([ladder_x, hole_top], dtype=jnp.int32),
            jnp.array([-1, -1], dtype=jnp.int32),
        )
        ladder_hole_size = jnp.array([ladder_w, hole_h], dtype=jnp.int32)
        raster = self.jr.draw_rects(raster, ladder_hole_pos[None, :], ladder_hole_size[None, :], int(self.HOLE_ID))

        # Simple ladder sprite (center hole drawn above, sprite only adds the ladder)
        ladder_sprite_top = jnp.int32(int(self.consts.ground_y) - 13)
        raster = lax.cond(
            has_simple_ladder,
            lambda r: self.jr.render_at_clipped(
                r,
                ladder_x,
                ladder_sprite_top,
                self.LADDER_SPRITE_MASK,
                flip_horizontal=jnp.array(False, dtype=jnp.bool_),
                flip_offset=jnp.array([0, 0], dtype=jnp.int32),
            ),
            lambda r: r,
            raster,
        )

        # Ladder-with-pits sprite (includes left hole + ladder + right hole)
        # Fixed offset 26: ladder structure starts at sprite col 26
        lwp_x = ladder_x - jnp.int32(26)
        lwp_top = jnp.int32(int(self.consts.ground_y) - 11)
        raster = lax.cond(
            has_ladder_with_pits,
            lambda r: self.jr.render_at_clipped(
                r,
                lwp_x,
                lwp_top,
                self.LADDER_WITH_PITS_MASK,
                flip_horizontal=jnp.array(False, dtype=jnp.bool_),
                flip_offset=jnp.array([0, 0], dtype=jnp.int32),
            ),
            lambda r: r,
            raster,
        )

        wall_side_bit = (rb >> jnp.uint8(7)) & jnp.uint8(1)
        has_wall = has_simple_ladder | has_ladder_with_pits
        wall_x = jnp.where(wall_side_bit == jnp.uint8(1), self.right_wall_x_px, self.left_wall_x_px).astype(jnp.int32)
        wall_h = jnp.int32(int(self.WALL_RENDER_MASK.shape[0]))
        # Kernel 8 draws `(wallPatPtr),y` and Kernel 9 draws `(undrPatPtr),y` over
        # the same y 15..0, and in a ladder scene both pointers hold `Wall`, so the
        # column runs from Kernel 8's first line to Kernel 9's last. Anchoring it
        # on that last line is what keeps its foot on the floor instead of two rows
        # short of it.
        wall_top = (
            jnp.int32(underground_last_row(self.consts.underground_y)) - wall_h + jnp.int32(1)
        )
        draw_wall_sprite = has_wall

        has_logs, logs_are_rolling, log_count, log_xs, has_fireplace, has_snake = room_hazards_from_room_byte(rb)

        touching_wood_render = jnp.where(
            logs_are_rolling,
            state.touching_rolling_wood,
            state.touching_wood,
        )

        total_frames = jnp.int32(self.consts.initial_time_seconds * self.consts.fps)
        frames_elapsed = jnp.maximum(total_frames - state.time_left.astype(jnp.int32), jnp.int32(0))
        frames_elapsed = frames_elapsed * state.timer_started.astype(jnp.int32)

        W = jnp.int32(self.consts.screen_width)
        log_left_x = log_left_edges(
            log_xs, logs_are_rolling, frames_elapsed, self.consts.screen_width
        )

        wood_top_static = int(self.consts.ground_y - self.consts.wood_h + self.consts.wood_y_offset)
        wood_top = jnp.int32(wood_top_static)

        # `lda xPosObject / asl asl asl / and #$30 / cmp #$30 / and #$10 / adc
        # objPatPtr`: bit 1 of xPosObject swaps Log0 for Log1, and the carry from
        # bits 1 and 2 both being set nudges the pattern one row, which is the
        # bounce the ROM's comment describes. Both are functions of position, not
        # of a timer, so a stationary log neither rolls nor bounces.
        anim_x = log_left_x[0].astype(jnp.int32)
        use_right_frame = logs_are_rolling & (jnp.bitwise_and(anim_x, jnp.int32(2)) != 0)
        bobble_y = jnp.where(
            logs_are_rolling & (jnp.bitwise_and(anim_x, jnp.int32(6)) == jnp.int32(6)),
            jnp.int32(1),
            jnp.int32(0),
        )

        log_mask = jnp.where(use_right_frame, self.LOG_RIGHT_MASK, self.LOG_LEFT_MASK)

        def _draw_logs(r: jnp.ndarray) -> jnp.ndarray:
            def body(i, rr):
                active_i = jnp.int32(i) < log_count
                x = log_left_x[i].astype(jnp.int32)
                y = wood_top + bobble_y
                return lax.cond(
                    active_i,
                    lambda rr: self.jr.render_at_clipped(
                        rr, x, y, log_mask,
                        flip_horizontal=jnp.array(False, dtype=jnp.bool_),
                        flip_offset=jnp.array([0, 0], dtype=jnp.int32),
                    ),
                    lambda rr: rr,
                    rr,
                )
            return lax.fori_loop(0, 3, body, r)

        # Fire and the four treasures share the cobra's GRP1 object path: one
        # eight-pixel player at xPosObject = 124, drawn over the ground band.
        # All three animate off the same random2 bit-4 source (AnimateTab gives
        # fire, cobra and the two bars OBJECT_H; the money bag and the ring are
        # static). A collected treasure is Nothing - ProcessObjects draws it only
        # while its treasureBits bit is still clear.
        obj_anim_frame = cobra_animation_frame(
            state.time_left, state.timer_started, self.consts, self.COBRA_ANIM_BIT4
        )
        fire_mask = self.FIRE_MASKS[obj_anim_frame]
        raster = lax.cond(
            has_fireplace,
            lambda r: self.jr.render_at_clipped(
                r,
                jnp.int32(self.consts.object_x),
                self.object_box_top_px,
                fire_mask,
                flip_horizontal=jnp.array(False, dtype=jnp.bool_),
                flip_offset=jnp.array([0, 0], dtype=jnp.int32),
            ),
            lambda r: r,
            raster,
        )

        obj_type = obj_code_u8(rb).astype(jnp.int32)
        treasure_type = obj_type & jnp.int32(3)
        treasure_visible = (pt == jnp.uint8(5)) & (
            ~treasure_collected(rb, state.treasure_bits)
        )
        treasure_frame = jnp.where(
            _TREASURE_ANIMATED[treasure_type], obj_anim_frame, jnp.int32(0)
        )
        treasure_mask = self.TREASURE_MASKS[treasure_type, treasure_frame]
        raster = lax.cond(
            treasure_visible,
            lambda r: self.jr.render_at_clipped(
                r,
                jnp.int32(self.consts.object_x),
                self.object_box_top_px,
                treasure_mask,
                flip_horizontal=jnp.array(False, dtype=jnp.bool_),
                flip_offset=jnp.array([0, 0], dtype=jnp.int32),
            ),
            lambda r: r,
            raster,
        )

        cobra_frame = obj_anim_frame
        cobra_mask = self.COBRA_MASKS[cobra_frame]
        cobra_x = jnp.int32(self.consts.object_x)
        raster = lax.cond(
            has_snake,
            lambda r: self.jr.render_at_clipped(
                r,
                cobra_x,
                self.COBRA_BOX_TOP,
                cobra_mask,
                flip_horizontal=jnp.array(False, dtype=jnp.bool_),
                flip_offset=jnp.array([0, 0], dtype=jnp.int32),
            ),
            lambda r: r,
            raster,
        )

        # The animation frame is bit 0 of xPosScorpion: `lda xPosScorpion / lsr /
        # bcc .scorpion0`. It is not timed, so a stationary scorpion is still.
        scorpion_x_i = state.scorpion_x.astype(jnp.int32)
        scorpion_mask = self.SCORPION_MASKS[jnp.bitwise_and(scorpion_x_i, jnp.int32(1))]
        # REFP1 reflects inside the box: reflected when Harry is to the left.
        scorpion_flip = ~state.scorpion_facing_right.astype(jnp.bool_)

        raster = lax.cond(
            has_scorpion,
            lambda r: self.jr.render_at_clipped(
                r,
                scorpion_x_i,
                self.SCORPION_BOX_TOP,
                scorpion_mask,
                flip_horizontal=scorpion_flip,
                flip_offset=jnp.array([0, 0], dtype=jnp.int32),
            ),
            lambda r: r,
            raster,
        )

        # --- The liana ---------------------------------------------------------
        # One ball pixel per scanline, its column generated by the same
        # accumulator the collision test walks, so what is drawn is what can be
        # caught. The rope is enabled for the whole band: COLUPF just changes
        # under it. Rows 33..64 (Kernel 1 and the prepare line) draw in
        # ColorTab+2, the leaves' own dark green; the line that opens Kernel 2
        # switches COLUPF to ColorTab+5 and rows 65 down are brown. Kernel 1
        # runs with CTRLPF = %101 - playfield priority on - so the leaves cover
        # the green section, and the branches (players) sit above the ball too:
        # a green rope pixel only survives where the scanline is bare
        # background, which is what the backdrop test below reproduces. From
        # Kernel 2 down CTRLPF is %001, priority off, so the brown section
        # draws over the playfield (the trunks) but stays under the players -
        # drawing it here, above the backdrop and below Harry and the logs, is
        # that order.
        liana_rows = jnp.arange(LIANA_ROWS, dtype=jnp.int32)
        liana_cols = liana_column(liana_rows, state.hmbl_add, state.hmbl_dir)
        liana_screen_rows = liana_rows + jnp.int32(LIANA_TOP_ROW)
        liana_row_lit = (
            liana_enabled(rb)
            & (liana_screen_rows <= liana_last_row(state.liana_bottom))
        )
        liana_band = raster[LIANA_TOP_ROW:LIANA_TOP_ROW + LIANA_ROWS]
        liana_mask = liana_row_lit[:, None] & (
            jnp.arange(int(self.consts.screen_width), dtype=jnp.int32)[None, :]
            == liana_cols[:, None]
        )
        liana_is_green_row = (
            liana_screen_rows < jnp.int32(LIANA_BROWN_TOP_ROW)
        )[:, None]
        # Playfield priority: the green section is only visible on bare
        # background pixels; leaves and branches both cover the ball there.
        liana_mask = liana_mask & (
            (~liana_is_green_row) | (liana_band == self.JUNGLE_BG_ID)
        )
        liana_colour = jnp.where(liana_is_green_row, self.LIANA_GREEN_ID, self.LIANA_ID)
        raster = raster.at[LIANA_TOP_ROW:LIANA_TOP_ROW + LIANA_ROWS].set(
            jnp.where(liana_mask, liana_colour, liana_band)
        )

        moving = jnp.abs(state.player_vx) > jnp.asarray(0.0, dtype=jnp.float32)
        # Use facing_left for idle sprites so Harry remembers direction
        flip = jnp.where(moving, state.player_vx < 0.0, state.facing_left)

        harry_pat = harry_display_pat_id(state, self.consts, touching_wood_render)
        harry_pat = jnp.clip(harry_pat, jnp.int32(0), jnp.int32(8))
        harry_mask = self.HARRY_PAT_MASKS[harry_pat]

        harry_falling_through_hole = is_falling_through_hole(state, self.consts)

        # player_x is xPosHarry, the left edge of the player box, so the whole
        # 8x22 canvas is drawn there for every pose and both facings. REFP0
        # reverses the bits inside the box without moving it, so the flip needs
        # no correction, and any sideways motion between poses is the animation.
        harry_x_draw = state.player_x.astype(jnp.int32)
        harry_flip_offset = jnp.array([0, 0], dtype=jnp.int32)

        # The same origin the collision masks use, so a change to the visual
        # datum cannot desynchronise what is drawn from what can be hit.
        y_top = harry_box_top_row(state.player_y, state.current_ground_y, self.consts)

        # Kernel 6's `lda #0 / sta GRP0`, and the playfield priority ContKernel
        # turns on below it. Both start at the same row, so blanking Harry's
        # pattern from there down is the whole of it - the terrain is not redrawn
        # over him, he is simply never emitted, which is what the hardware does.
        # A standing Harry's lowest row is the last line Kernel 5 draws, so this
        # takes nothing off him until he starts to sink.
        harry_blank_top = jnp.int32(pit_harry_blank_top_row(self.consts.ground_y))
        harry_sprite_rows = jnp.arange(harry_mask.shape[0], dtype=jnp.int32)
        harry_row_blanked = (y_top + harry_sprite_rows) >= harry_blank_top
        harry_mask = jnp.where(
            (in_pit & harry_row_blanked)[:, None],
            jnp.asarray(self.jr.TRANSPARENT_ID, dtype=harry_mask.dtype),
            harry_mask,
        )

        # There is no "dead, so do not draw" rule to apply. This build sets
        # SCREENSAVER = 1, so the position loop's guard is
        #
        #   IF SCREENSAVER
        #     ldy    SS_Delay         ; 3                 game running?
        #     bmi    .skipHarryPos    ; 2³                 no, don't draw Harry
        #   ELSE
        #     ldy    noGameScroll     ;                   TODO: bugfix, wall isn't drawn
        #     bne    .skipHarryPos    ; 2³                 no, don't draw Harry
        #   ENDIF
        #
        # and it is SS_Delay - the idle timer that only goes negative after
        # minutes without input - that suppresses Harry, never noGameScroll. A
        # frozen Harry keeps his coordinate and the kernel keeps drawing him.
        # Whether he is actually seen is decided by the raster alone: where he
        # sits, and what the terrain draws over him.
        def _draw_harry(r: jnp.ndarray) -> jnp.ndarray:
            return self.jr.render_at_clipped(
                r,
                harry_x_draw,
                y_top,
                harry_mask,
                flip_horizontal=flip,
                flip_offset=harry_flip_offset,
            )

        # Save raster before Harry+logs for lip occlusion (lip only covers
        # ladder area so logs outside the lip bbox are unaffected).
        raster_base = raster

        raster = lax.cond(
            touching_wood_render,
            lambda r: _draw_harry(lax.cond(has_logs, _draw_logs, lambda rr: rr, r)),
            lambda r: lax.cond(has_logs, _draw_logs, lambda rr: rr, _draw_harry(r)),
            raster,
        )

        underground_respawn_reveal = (
            (state.respawn_phase == jnp.int32(2))
            & (state.respawn_target_ground_y == jnp.asarray(self.consts.underground_y, dtype=jnp.float32))
        )
        reveal_y = jnp.int32(
            int(
                self.consts.ground_y
                + self.consts.underground_respawn_reveal_from_ground
                + self.consts.underground_respawn_reveal_y_offset
            )
        )

        def _clip_underground_respawn(r: jnp.ndarray) -> jnp.ndarray:
            H, W = r.shape
            yy = jnp.arange(H, dtype=jnp.int32)[:, None]
            xx = jnp.arange(W, dtype=jnp.int32)[None, :]
            # The box never moves, so one span covers both facings.
            harry_x0 = harry_x_draw
            harry_x1 = harry_x0 + jnp.int32(harry_mask.shape[1])
            hidden_mask = (
                (xx >= harry_x0)
                & (xx < harry_x1)
                & (yy >= y_top)
                & (yy < jnp.minimum(y_top + jnp.int32(harry_mask.shape[0]), reveal_y))
            )
            return jnp.where(hidden_mask, raster_base, r)

        raster = lax.cond(
            underground_respawn_reveal,
            _clip_underground_respawn,
            lambda r: r,
            raster,
        )

        raster = lax.cond(
            draw_wall_sprite,
            lambda r: self.jr.render_at_clipped(
                r,
                wall_x,
                wall_top,
                self.WALL_RENDER_MASK,
                flip_horizontal=jnp.array(False, dtype=jnp.bool_),
                flip_offset=jnp.array([0, 0], dtype=jnp.int32),
            ),
            lambda r: r,
            raster,
        )

        # Re-stamp logs into lip region so lip occlusion doesn't erase them
        raster_base = lax.cond(has_logs, _draw_logs, lambda r: r, raster_base)

        # Lip occlusion: restore backdrop pixels in a thin strip around the
        # ladder/pit opening so ladder rungs appear in front of Harry.
        lip_y0 = jnp.int32(int(self.consts.ground_y) - 4)
        lip_y1 = jnp.int32(int(self.consts.ground_y) + 6)
        has_any_ladder = has_simple_ladder | has_ladder_with_pits

        # For simple ladder, lip covers just the ladder width.
        # For ladder_with_pits, lip covers the full sprite.
        lip_x0 = jnp.where(has_ladder_with_pits, lwp_x, ladder_x)
        lip_x1 = jnp.where(
            has_ladder_with_pits,
            lwp_x + jnp.int32(self.LADDER_WITH_PITS_MASK.shape[1]),
            ladder_x + jnp.int32(self.consts.ladder_width),
        )
        transparent_id = jnp.asarray(self.jr.TRANSPARENT_ID, dtype=raster.dtype)
        hole_id = jnp.asarray(self.HOLE_ID, dtype=raster.dtype)

        def _apply_lip(r: jnp.ndarray) -> jnp.ndarray:
            # For each pixel in the lip bbox, if the base raster (before Harry)
            # is non-transparent, restore it on top of Harry.
            H, W = r.shape
            yy = jnp.arange(H, dtype=jnp.int32)[:, None]
            xx = jnp.arange(W, dtype=jnp.int32)[None, :]
            in_lip = (yy >= lip_y0) & (yy < lip_y1) & (xx >= lip_x0) & (xx < lip_x1)
            base_not_transparent = raster_base != transparent_id

            # While dropping through an opening the occluder must be solid ground
            # only. HOLE_ID marks where the ground is cut away, so skipping those
            # pixels keeps Harry visible inside the opening. Left alone otherwise:
            # the solid black is what makes him rise out of the hole off a ladder.
            base_is_opening = raster_base == hole_id
            occludes = base_not_transparent & ~(base_is_opening & harry_falling_through_hole)

            mask = in_lip & occludes
            return jnp.where(mask, raster_base, r)

        raster = lax.cond(has_any_ladder, _apply_lip, lambda r: r, raster)

        frame = self.jr.render_from_palette(raster, self.PALETTE)

        # --- ShowDigits ------------------------------------------------------
        # Two lines of six eight-pixel slots, each line a single COLUP colour.
        # The glyphs come from DIGIT_PATTERNS, so the HUD is the ROM's font
        # rather than a hand-drawn approximation of it.
        channels = frame.shape[2]
        if channels == 1:
            hud_color = jnp.array([int(round(sum(HUD_COLOR) / 3))], dtype=jnp.uint8)
        else:
            hud_color = jnp.asarray(HUD_COLOR, dtype=jnp.uint8)

        def _show_digits(f, row: int, glyph_ids, lives_pattern):
            band = jnp.zeros((DIGIT_H, int(self.consts.screen_width)), dtype=jnp.bool_)
            glyphs = _DIGIT_GLYPHS[glyph_ids]
            for slot, left in enumerate(HUD_SLOT_X):
                band = band.at[:, left:left + DIGIT_W].set(
                    band[:, left:left + DIGIT_W] | glyphs[slot]
                )
            # `lda (digitPtr),y / ora temp3`: livesPat rides on slot 0's pattern
            # for every row of the line, which is why it reads as vertical bars.
            bits = jnp.arange(DIGIT_W - 1, -1, -1, dtype=jnp.int32)
            lives_cols = ((lives_pattern >> bits) & jnp.int32(1)).astype(jnp.bool_)
            left0 = HUD_SLOT_X[0]
            band = band.at[:, left0:left0 + DIGIT_W].set(
                band[:, left0:left0 + DIGIT_W] | lives_cols[None, :]
            )
            region = f[row:row + DIGIT_H]
            return f.at[row:row + DIGIT_H].set(
                jnp.where(band[:, :, None], hud_color[None, None, :], region)
            )

        # `.loopSpace` walks the first five slots replacing leading Zeros with
        # Space and stops before the last one, so a score of zero still shows a
        # digit. Zero is page-aligned in the ROM, which is what makes its
        # pointer test a test for the digit itself.
        score_digits = self.jr.int_to_digits(
            state.score.astype(jnp.int32), max_digits=len(HUD_SLOT_X)
        )
        leading = jnp.cumsum((score_digits != jnp.int32(0)).astype(jnp.int32)) > jnp.int32(0)
        is_last = jnp.arange(len(HUD_SLOT_X)) == (len(HUD_SLOT_X) - 1)
        score_glyphs = jnp.where(leading | is_last, score_digits, jnp.int32(DIGIT_SPACE))

        time_seconds = state.time_left.astype(jnp.int32) // jnp.int32(self.consts.fps)
        minutes = time_seconds // jnp.int32(60)
        seconds = time_seconds - minutes * jnp.int32(60)
        mm = self.jr.int_to_digits(minutes, max_digits=2)
        ss = self.jr.int_to_digits(seconds, max_digits=2)
        # `ldy digitPtr+2 / bne .noSpace`: only the minutes tens digit blanks.
        mm_tens = jnp.where(mm[0] == jnp.int32(0), jnp.int32(DIGIT_SPACE), mm[0])
        timer_glyphs = jnp.stack(
            [
                jnp.int32(DIGIT_SPACE),   # slot 0 carries livesPat instead
                mm_tens,
                mm[1].astype(jnp.int32),
                jnp.int32(DIGIT_COLON),
                ss[0].astype(jnp.int32),
                ss[1].astype(jnp.int32),
            ]
        )

        lives = jnp.clip(state.lives_left.astype(jnp.int32), jnp.int32(0), jnp.int32(3))
        lives_pattern = jnp.asarray(LIVES_PAT, dtype=jnp.int32)[lives]

        frame = _show_digits(frame, HUD_SCORE_ROW, score_glyphs, jnp.int32(0))
        frame = _show_digits(frame, HUD_TIMER_ROW, timer_glyphs, lives_pattern)

        # The third ShowDigits pass. It runs on every frame of a running game
        # too, where the offset is nailed to zero and the window sits in Space,
        # so the row is blank unless the game is stopped and the tune is over.
        scroll = copyright_scroll(state.no_game_scroll, state.sound_idx)
        copyright_slots = _COPYRIGHT_GLYPHS[scroll]
        copyright_band = jnp.zeros((DIGIT_H, int(self.consts.screen_width)), dtype=jnp.bool_)
        for slot, left in enumerate(HUD_SLOT_X):
            copyright_band = copyright_band.at[:, left:left + DIGIT_W].set(
                copyright_band[:, left:left + DIGIT_W] | copyright_slots[slot]
            )
        copyright_region = frame[COPYRIGHT_ROW:COPYRIGHT_ROW + DIGIT_H]
        frame = frame.at[COPYRIGHT_ROW:COPYRIGHT_ROW + DIGIT_H].set(
            jnp.where(copyright_band[:, :, None], hud_color[None, None, :], copyright_region)
        )

        # TIA blanking, applied last because it does not care what was drawn.
        black = jnp.zeros((), dtype=frame.dtype)
        frame = frame.at[:, :HMOVE_BLANK_COLS, :].set(black)
        frame = frame.at[:VBLANK_ROWS, :, :].set(black)

        return frame
