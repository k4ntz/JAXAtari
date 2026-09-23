"""
JAX implementation of H.E.R.O. (Helicopter Emergency Rescue Operation) —
Activision, 1984. Twenty levels, measured from the real ROM. Levels 1 and 2
have been rebuilt cell-for-cell against the reference pack in ../level_images
by tools/hero_gen/build_level.py; levels 3-16 still come from the first
capture pass and are wrong in the corridor band of nearly every room past
room 0 (level_images/CORRECTIONS.md), and 17-20 are placeholders. See
hero_levels.py.

Ground truth / provenance
-------------------------
Everything gameplay-visible here was measured directly from the Activision
H.E.R.O. ROM running under ALE (see scripts/hero_record_level.py and the
capture notes below): the per-room background pixels, wall collision
rectangles, palette, HUD layout, player/creature/miner sprites, physics
constants, weapon behaviour and room-transition model. The measured data
lives in hero_levels.py (palette-indexed RLE screens + object tables).

Screen-flip world model (measured)
----------------------------------
The original game does NOT scroll: each level is a stack of STATIC screens
("rooms", 142 rows of cave + fixed HUD below). The player's screen y spans
~8..144; crossing the bottom flips to the next room with the player placed
just above the top (and vice versa flying up). The room counts are
hero_levels.ROOMS_PER_LEVEL: 2, 4, 6, 8, 8, 10, 12, 14, then 16 each. Rescuing the miner in the last room advances to the
next level (score/lives carry over, power and dynamite refill); rescuing the
LAST level's miner completes the episode.

Measured mechanics
------------------
  * Fall: constant 1 px/frame (no acceleration). Thrust (UP): the fall stops
    immediately, then rise ramps ~0.125 px/frame^2 up to 2 px/frame.
  * Walk: exactly 1 px/frame.
  * Laser: NOT a beam that grows out of the helmet. Firing launches an 8x1
    bolt on the hero's eye row that flies 3 px/frame for 6 frames; while the
    button is held a fresh bolt is launched behind it, so what the player
    sees is his two eye pixels and a detached dash travelling away. Kills
    creatures at range (+50).
  * The beam also EATS ROCK, slowly: 256 frames of fire on one 4 px column
    clears it from the ceiling and middle bands together (never the floor),
    for no points. The burn is remembered when the button is released. A
    wall run that reaches a screen edge never melts, so the hero cannot burn
    his way out of the cave. HERO_SPEC.md says the laser "does not cut rock";
    the ROM disagrees and the ROM wins - this is the slow, free alternative
    to a stick of dynamite, which is instant but costs one of six and pays 75.
  * Touching a creature kills the HERO, and holding fire does not save him.
    Measured by walking into level 1 room 1's spider: not firing, he stops
    dead against it and loses a life 107 frames later (the ROM's death
    freeze); firing, the bolt kills the spider while he is still 20 px away.
  * Dynamite: laid with DOWN (a stick at the feet), fuse 60 frames (playable
    stand-in for the ROM's 34 - see dyn_fuse_rom), then a brief blast that
    kills creatures (+50) and takes a wall column outright (+75). Blasting
    the opening-room pillar is the way down through each level. Dynamite and
    the laser are the two ways through rock and both are real: the stick is
    instant but costs one of six and pays 75, the beam is free but wants 256
    frames a column and pays nothing. (An earlier note here called the
    laser-melt a raw-ROM artefact and dropped it; it is genuine - measured
    again 2026-09-20 - and is now implemented.)
  * Death (creature touch, blast, power depletion): lose a life and respawn
    at the top of the CURRENT room (measured); power refills.
  * Miner rescue: +1000 plus an end-of-level tally of 20 points per power-bar
    pixel still lit (measured: a clean level-1 clear pays ~1300-1600).
  * Lives: start 4 (measured), +1 every 20000 points, capped at 6 heroes in
    reserve (a counter of 7, since one of them is the one on screen).

Known approximations (non-measured details)
-------------------------------------------
Leg/rotor animation frames, the explosion flash sprite and the score digit
font for digits not observed in captures are hand-drawn approximations in the
measured palette. The publisher's ACTIVISION wordmark is deliberately not
reproduced: rows 189-209 stay black. The power gauge is an internal frame
counter (78 bar pixels x 68 frames of movement each, measured) so the HUD bar
and the end-of-level bonus both fall out of the same number.

Conventions follow the other games in this package (see jax_freeway.py):
constants subclass AutoDerivedConstants; state/observation/info are
flax.struct dataclass pytrees; the env subclasses JaxEnvironment and the
renderer subclasses JAXGameRenderer.
"""

import os
from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, List
from flax import struct

from jaxatari.environment import JaxEnvironment, ObjectObservation, JAXAtariAction as Action
import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.modification import AutoDerivedConstants
from jaxatari.games import hero_levels as HL


# ---------------------------------------------------------------------------
# Pack the measured level data into fixed-shape (padded) arrays indexed by
# (level, room). Unused slots are marked invalid so they never collide/draw.
# ---------------------------------------------------------------------------
_NUM_LEVELS = HL.NUM_LEVELS
_ROOMS = HL.ROOMS_PER_LEVEL
_MAX_ROOMS = max(_ROOMS)
_CELLS = 38          # 38 cells of 4 px per band (HERO_SPEC.md section 2)
_RECTS = [getattr(HL, f"WALL_RECTS_L{i}") for i in range(1, _NUM_LEVELS + 1)]
def _carve_room(rects, zones):
    """The static wall rects of one room with its blastable zones cut out.

    A column must not be both a fixed wall and a blastable one, and a zone
    need not line up with a rect: the band decomposition may have split a
    pillar into several columns, and a zone can equally sit strictly INSIDE a
    wider run. Level 2 room 1 is the case that matters - its blastable pillar
    is two cells of the corridor, but the ceiling above it is part of a
    16-cell slab, and a stick takes the ceiling column along with the corridor
    one (HERO_SPEC.md section 6).

    So each rect is cut on BOTH axes: the columns of it beside the zone
    survive at full height, and only the columns inside the zone lose the
    zone's rows.
    """
    out = [tuple(int(v) for v in r) for r in rects]
    for (x, y, w, h) in zones:
        nxt = []
        for (wx, wy, ww, wh) in out:
            if wx >= x + w or wx + ww <= x or wy >= y + h or wy + wh <= y:
                nxt.append((wx, wy, ww, wh))          # no overlap, untouched
                continue
            if wx < x:                                 # the part left of it
                nxt.append((wx, wy, x - wx, wh))
            if wx + ww > x + w:                        # the part right of it
                nxt.append((x + w, wy, wx + ww - (x + w), wh))
            mid_x = max(wx, x)                         # and the part over it
            mid_w = min(wx + ww, x + w) - mid_x
            if y > wy:
                nxt.append((mid_x, wy, mid_w, y - wy))
            if wy + wh > y + h:
                nxt.append((mid_x, y + h, mid_w, wy + wh - (y + h)))
        out = nxt
    return out


# The carve is done once here, in plain Python, so the padded array can be
# sized from what it actually produces. Guessing the spare slots instead got
# this badly wrong once: budgeting three per zone PER LEVEL rather than per
# room made the array 13x wider than it needed to be, and every collision
# check walks it.
_CARVED = {
    (li, ri): _carve_room(rects,
                          [(z[1], z[2], z[3], z[4])
                           for z in HL.DESTRUCTIBLE[li] if z[0] == ri])
    for li in range(_NUM_LEVELS)
    for ri, rects in enumerate(_RECTS[li])
}
_MAX_WALLS = max(len(v) for v in _CARVED.values()) + 2
_MAX_SPIDERS = max(len(s) for s in HL.SPIDERS)
_MAX_DWALLS = max(len(d) for d in HL.DESTRUCTIBLE)
_MAX_LANTERNS = max(1, max(len(l) for l in HL.LANTERNS))
_MAX_DEADLY = max(1, max(len(d) for d in HL.DEADLY))
_MAX_FLARES = max(1, max(len(f) for f in HL.FLARES))
_MAX_MAGMA = max(1, max(len(m) for m in HL.MAGMA))


def _bolt_pos(c, px, py, facing, laser_timer):
    """Top-left of the laser bolt, and the eye row it travels along.

    The bolt steps laser_bolt_speed px away from the hero each frame and a
    fresh one is launched every laser_bolt_frames frames, which is what the
    ROM does while the button is held (see HeroConstants). Shared by the env
    and the renderer so what kills is exactly what is drawn.
    """
    phase = jnp.maximum(laser_timer - 1, 0) % c.laser_bolt_frames
    reach = phase * c.laser_bolt_speed
    bx = jnp.where(facing < 0,
                   px - c.laser_bolt_length - reach,
                   px + c.player_width + reach).astype(jnp.int32)
    return bx, (py + c.laser_eye_offset).astype(jnp.int32)


# Archetype fallbacks for a creature nobody has measured yet: a spider on a
# thread bobs about +-5 px, a bat about +-2, magma and snakes not at all, and
# every one of them swaps pose about every 8 frames. Levels 3-16 still run on
# these; levels 1 and 2 are measured (HL.CREATURE_MOTION). These stay
# AMPLITUDES about the y in HL.SPIDERS, because that is what the y in those
# unmeasured rows was authored to mean; _build_level_arrays turns them into
# the one-sided (top row, travel) pair _creature_pos wants, which leaves an
# unmeasured creature bobbing over exactly the rows it always did, and leaves
# SPIDER_Y meaning the y in HL.SPIDERS for every level.
_DEFAULT_BOB = {0: 5, 1: 2}
_DEFAULT_BOB_HALF = 20
_DEFAULT_ANIM_HOLD = 8
_DEFAULT_PATROL_HALF = 40           # HeroConstants.spider_patrol_half_period


# --- The wall snake's stretch (measured on the ROM 2026-09-22, level 4 room
# 3: 150 consecutive frames with the hero pinned clear of it) ---------------
# A snake does not bob, patrol or cycle poses the way the other creatures do.
# It is a green tongue anchored in a wall's side that STRETCHES out of the
# rock and pulls back in, on a free-running 64-frame cycle that ignores the
# player entirely (verified: identical traces with the hero pinned at nine
# different spots, near and far).
#
#   frames  0-7   inside the rock, nothing drawn at all
#           8-31  1, 2, 3, 4, 5, 6 px out, a pixel every 4 frames
#          32-39  7 px out, at full stretch
#          40-63  6, 5, 4, 3, 2, 1 px, pulling back in
#
# Alongside that its head flutters between two shapes every 8 frames, on its
# own clock: the phase between the two (head up while (t + 5) % 16 < 8) is
# measured off the same film. The old model - four blobs held 3 frames each,
# none of them the shape of anything the ROM draws - was invented from a
# reference whose "four poses" were a frame-differencer splitting the snake's
# DISCONNECTED pixels into pieces (see [[hero-characters-md-motion-artefact]]).
_SNAKE_CYCLE = 64
_SNAKE_STEP = 4                     # frames per pixel of stretch
_SNAKE_REACH = 7                    # pixels at full stretch
_SNAKE_POSES = 1 + 2 * _SNAKE_REACH  # pose 0 is "inside the rock"


def _snake_length(step):
    """How many pixels of snake are out of the rock on a given frame (0..7).

    This is the creature's whole extent: the box the engine collides with
    and the sprite the renderer draws are both this many pixels wide, so a
    snake that is pulled in cannot be touched, shot or drawn.
    """
    k = (step % _SNAKE_CYCLE) // _SNAKE_STEP          # 0..15, a pixel each
    out = jnp.minimum(jnp.maximum(k - 1, 0), _SNAKE_REACH)
    return jnp.where(k < 10, out, 16 - k).astype(jnp.int32)


def _snake_head_down(step):
    """The head flutters between two shapes every 8 frames (measured)."""
    return ((step % _SNAKE_CYCLE) + 5) % 16 >= 8


def _snake_pose(length, head_down):
    """Index into the snake's sprite table: 0 is the empty canvas it draws
    while it is inside the rock, then 1-7 head up and 8-14 head down."""
    return jnp.where(length == 0, 0,
                     length + jnp.where(head_down, _SNAKE_REACH, 0)
                     ).astype(jnp.int32)


def _creature_pos(c, lvl, step):
    """(x, y) of every creature slot of a level on a given frame.

    Shared by the env and the renderer so what kills is exactly what is
    drawn. Both the bob and the patrol are triangle waves at the creature's
    OWN measured travel and period - nothing here is per kind, because the
    ROM's creatures do not share a rhythm: level 2's spiders bob 9 px over
    64 frames, a bat patrols 22 px over ~184, and level 1's spider does not
    move at all.

    The bob is a triangle SPIDER_BOB px tall, lifted by SPIDER_BOB_BASE, so
    it runs over rows SPIDER_Y - base .. SPIDER_Y - base + travel. A MEASURED
    creature has base 0, because the ROM reports it at the TOP row of the box
    it is drawn in and the bob runs downwards from there - level 2's spiders
    bob 9 px, and 9 is odd, so a symmetric +-amplitude could not place them
    on the rows the ROM does. An UNMEASURED one keeps base = its archetype
    amplitude, which is what the y in HL.SPIDERS was authored to mean, so it
    bobs over exactly the rows it always did.
    """
    travel = c.SPIDER_BOB[lvl]
    half = c.SPIDER_BOB_HALF[lvl]
    base = c.SPIDER_BOB_BASE[lvl]
    t = step % (2 * half)
    bob = jnp.abs(t - half) * travel // half - base
    # The sweep runs on its OWN clock, not the bob's: level 3's bats bob on
    # 64 frames and patrol on 184, so a creature that used one period for
    # both would trace a diagonal the ROM never draws. SPIDER_PATROL_HALF is
    # the measured half-cycle, defaulting to spider_patrol_half_period for
    # the levels nobody has remeasured. SPIDER_X is the CENTRE of the sweep.
    ph = c.SPIDER_PATROL_HALF[lvl]
    tp = step % (2 * ph)
    tri = jnp.abs(tp - ph) * 2 - ph                # -ph .. +ph triangle
    sweep = (c.SPIDER_PATROL[lvl] * tri) // ph
    return c.SPIDER_X[lvl] + sweep, c.SPIDER_Y[lvl] + bob


def _build_level_arrays():
    nL, nR, nW, nS = _NUM_LEVELS, _MAX_ROOMS, _MAX_WALLS, _MAX_SPIDERS
    walls = np.zeros((nL, nR, nW, 4), np.int32)
    wall_valid = np.zeros((nL, nR, nW), bool)
    rooms_n = np.zeros((nL,), np.int32)
    miner = np.zeros((nL, 3), np.int32)           # room, x, y
    sp_room = np.zeros((nL, nS), np.int32)
    sp_x = np.zeros((nL, nS), np.int32)
    sp_y = np.zeros((nL, nS), np.int32)
    sp_patrol = np.zeros((nL, nS), np.int32)
    sp_kind = np.zeros((nL, nS), np.int32)
    sp_valid = np.zeros((nL, nS), bool)
    sp_bob = np.zeros((nL, nS), np.int32)          # bob travel, px downwards
    sp_bob_base = np.zeros((nL, nS), np.int32)     # px the bob starts above y
    sp_bob_half = np.ones((nL, nS), np.int32)      # half a bob cycle, frames
    sp_hold = np.ones((nL, nS), np.int32)          # frames per drawn pose
    sp_poses = np.zeros((nL, nS), np.int32)        # distinct sprites, 0 = kind's
    sp_patrol_half = np.full((nL, nS), _DEFAULT_PATROL_HALF, np.int32)
    nD = _MAX_DWALLS
    dw = np.zeros((nL, nD, 6), np.int32)          # room, x, y, w, h, dyn_ok
    dw_valid = np.zeros((nL, nD), bool)
    # Shared walls: one physical wall drawn on two screens. dw_group is the
    # group id of each slot (0 = its own wall); dw_scores marks the one slot
    # of each group that pays the 75 points, so a shared wall is not paid for
    # twice. See HL.SHARED_WALLS.
    dw_group = np.zeros((nL, nD), np.int32)
    dw_scores = np.ones((nL, nD), bool)
    nLan = _MAX_LANTERNS
    lan = np.zeros((nL, nLan, 3), np.int32)       # room, x, y
    lan_valid = np.zeros((nL, nLan), bool)
    nDe = _MAX_DEADLY
    de = np.zeros((nL, nDe, 5), np.int32)         # room, x, y, w, h
    de_valid = np.zeros((nL, nDe), bool)
    nFl = _MAX_FLARES
    fl = np.zeros((nL, nFl, 7), np.int32)         # room,x,y,w,h,period,duty
    fl_valid = np.zeros((nL, nFl), bool)
    nMg = _MAX_MAGMA
    mg = np.zeros((nL, nMg, 5), np.int32)         # room, x, y, w, h
    mg_valid = np.zeros((nL, nMg), bool)
    # Side exits: per (level, room), whether that room's corridor is open at
    # the left / right edge of the screen, and which way along the chain that
    # edge goes. The direction is data because it is not the same on every
    # level: level 5 descends through a LEFT edge and level 7 through a RIGHT
    # one. See HL.SIDE_EXITS.
    side_l = np.zeros((nL, _MAX_ROOMS), bool)
    side_r = np.zeros((nL, _MAX_ROOMS), bool)
    side_l_d = np.zeros((nL, _MAX_ROOMS), np.int32)
    side_r_d = np.zeros((nL, _MAX_ROOMS), np.int32)
    for li in range(nL):
        rooms_n[li] = _ROOMS[li]
        for (rm, side, delta) in HL.SIDE_EXITS[li]:
            if side < 0:
                side_l[li, rm], side_l_d[li, rm] = True, delta
            else:
                side_r[li, rm], side_r_d[li, rm] = True, delta
        for ri in range(len(_RECTS[li])):
            for wi, (x, y, w, h) in enumerate(_CARVED[(li, ri)]):
                walls[li, ri, wi] = (x, y, w, h)
                wall_valid[li, ri, wi] = True
        miner[li] = HL.MINER_POS[li]
        for si, (rm, x, y, patrol, kind) in enumerate(HL.SPIDERS[li]):
            sp_room[li, si], sp_x[li, si], sp_y[li, si] = rm, x, y
            sp_patrol[li, si] = patrol
            sp_kind[li, si] = kind
            sp_valid[li, si] = True
            # Motion is per creature, not per kind. A slot that has been
            # measured says how far it bobs, how long a cycle takes and how
            # long it holds a pose; the rest fall back to their archetype
            # (see HL.CREATURE_MOTION and HeroConstants).
            amp = _DEFAULT_BOB.get(kind, 0)
            half, hold = _DEFAULT_BOB_HALF, _DEFAULT_ANIM_HOLD
            travel, base = 2 * amp, amp          # the unmeasured archetype
            if (li + 1, si) in HL.CREATURE_MOTION:
                travel, half, hold = HL.CREATURE_MOTION[(li + 1, si)]
                base = 0                         # measured: y IS the top row
            sp_bob[li, si] = travel
            sp_bob_base[li, si] = base
            sp_bob_half[li, si] = max(1, half)
            sp_hold[li, si] = max(1, hold)
            sp_patrol_half[li, si] = max(1, HL.CREATURE_PATROL.get(
                (li + 1, si), _DEFAULT_PATROL_HALF))
            # How many sprites the ROM actually draws this creature as. 0
            # means "nobody counted", and the renderer then runs the kind's
            # whole cycle, which is what every unmeasured level expects. A
            # measured STILL creature counts 1 and so is drawn as one bitmap
            # - the pose clock still ticks, it just has nowhere to go.
            sp_poses[li, si] = HL.CREATURE_SPRITES.get((li + 1, si), 0)
        for gi, (rm, x, y) in enumerate(HL.LANTERNS[li]):
            lan[li, gi] = (rm, x, y)
            lan_valid[li, gi] = True
        for gi, (rm, x, y, w, h) in enumerate(HL.DEADLY[li]):
            de[li, gi] = (rm, x, y, w, h)
            de_valid[li, gi] = True
        for gi, (rm, x, y, w, h, per, duty) in enumerate(HL.FLARES[li]):
            fl[li, gi] = (rm, x, y, w, h, per, duty)
            fl_valid[li, gi] = True
        for gi, (rm, x, y, w, h) in enumerate(HL.MAGMA[li]):
            mg[li, gi] = (rm, x, y, w, h)
            mg_valid[li, gi] = True
        for gi, group in enumerate(HL.SHARED_WALLS[li], start=1):
            for k, slot in enumerate(group):
                assert slot < len(HL.DESTRUCTIBLE[li]), "shared-wall slot out of range"
                assert dw_group[li, slot] == 0, "a slot belongs to one wall only"
                dw_group[li, slot] = gi
                dw_scores[li, slot] = (k == 0)
        for di, (rm, x, y, w, h, dyn_ok) in enumerate(HL.DESTRUCTIBLE[li]):
            dw[li, di] = (rm, x, y, w, h, dyn_ok)
            dw_valid[li, di] = True
            # The zone is already cut out of the static rects by
            # _carve_room, which runs once at import so the padded
            # array can be sized from what it produces.
    # --- cell grid -----------------------------------------------------
    # The cave is 38 cells of 4 px in each of three bands (HERO_SPEC.md
    # section 2). Rasterise every wall - static rects and destructible zones
    # alike - into (band, cell) occupancy. The laser melt works on this grid;
    # it can only ever REMOVE collision, so a level with nothing melted
    # behaves exactly as its rects always did.
    BANDS = ((16, 59), (60, 98), (99, 141))
    cell_solid = np.zeros((nL, nR, 3, _CELLS), bool)
    for li in range(nL):
        for ri in range(nR):
            boxes = [tuple(int(v) for v in walls[li, ri, wi])
                     for wi in range(walls.shape[2]) if wall_valid[li, ri, wi]]
            boxes += [(int(dw[li, di, 1]), int(dw[li, di, 2]),
                       int(dw[li, di, 3]), int(dw[li, di, 4]))
                      for di in range(nD)
                      if dw_valid[li, di] and int(dw[li, di, 0]) == ri]
            for (x, y, w, h) in boxes:
                for bi, (r0, r1) in enumerate(BANDS):
                    if y > r1 or y + h <= r0:
                        continue
                    for cell in range(_CELLS):
                        cx = 8 + 4 * cell
                        if cx < x + w and cx + 4 > x:
                            cell_solid[li, ri, bi, cell] = True
    # Which columns the laser is allowed to eat. Measured on the ROM: an
    # interior pillar melts (256 frames per 4 px cell) but a wall run that
    # touches the screen edge never does - 700 frames against level 1 room
    # 0's left wall and room 1's right wall changed nothing. That is also
    # what stops the hero burning his way out of the cave.
    # Per band, because the rule is about the run the BEAM meets: the beam
    # travels along the hero's eye row, so it is that band's run which either
    # melts or does not. All three clean ROM readings are explained by the
    # middle band alone (room 0 cell 1 immune, cell 13 melts, room 1 cell 32
    # immune); melting then takes the ceiling cell of the column with it.
    meltable = np.zeros((nL, nR, 3, _CELLS), bool)
    for li in range(nL):
        for ri in range(nR):
            for bi in range(3):
                row = cell_solid[li, ri, bi]
                cell = 0
                while cell < _CELLS:
                    if not row[cell]:
                        cell += 1; continue
                    run = cell
                    while run < _CELLS and row[run]:
                        run += 1
                    if cell > 0 and run < _CELLS:     # an interior run
                        meltable[li, ri, bi, cell:run] = True
                    cell = run
            # the floor band is never eaten, by the laser or by dynamite
            meltable[li, ri, 2, :] = False
    # Magma is immune to the laser (HERO_SPEC.md section 3): the beam does
    # nothing to a red block however long it is held. Dynamite still takes it,
    # which is a DESTRUCTIBLE zone and not this grid.
    BAND_OF = {16: 0, 60: 1, 99: 2}
    for li in range(nL):
        for (rm, x, y, w, h) in HL.MAGMA[li]:
            bi = BAND_OF[y]
            for cell in range(_CELLS):
                cx = 8 + 4 * cell
                if cx < x + w and cx + 4 > x:
                    meltable[li, rm, bi, cell] = False
            # A magma rect stops burning when a blasted destructible zone
            # COVERS it (see died_magma in step). That test is all-or-nothing,
            # so a rect must never be only partly blastable - otherwise half a
            # column could be blown away and the whole rect would go on
            # killing. Every interior magma run measured so far is exactly one
            # 8 px column, which satisfies this; if a wider one ever turns up,
            # split it here (and in the generator) into one rect per zone.
            for (zr, zx, zy, zw, zh, _ok) in HL.DESTRUCTIBLE[li]:
                if zr != rm or zy != y:
                    continue
                overlaps = zx < x + w and zx + zw > x
                contains = zx <= x and zx + zw >= x + w
                assert not overlaps or contains, (
                    f"level {li + 1} room {rm}: magma rect {(x, y, w, h)} is "
                    f"only partly covered by blastable zone {(zx, zy, zw, zh)}")
    return dict(walls=walls, wall_valid=wall_valid, rooms_n=rooms_n, miner=miner,
                cell_solid=cell_solid, meltable=meltable,
                sp_room=sp_room, sp_x=sp_x, sp_y=sp_y, sp_patrol=sp_patrol,
                sp_kind=sp_kind, sp_valid=sp_valid,
                sp_bob=sp_bob, sp_bob_half=sp_bob_half, sp_hold=sp_hold,
                sp_poses=sp_poses,
                sp_bob_base=sp_bob_base, sp_patrol_half=sp_patrol_half,
                dw=dw, dw_valid=dw_valid,
                dw_group=dw_group, dw_scores=dw_scores,
                lan=lan, lan_valid=lan_valid, de=de, de_valid=de_valid,
                fl=fl, fl_valid=fl_valid, mg=mg, mg_valid=mg_valid,
                side_l=side_l, side_r=side_r,
                side_l_d=side_l_d, side_r_d=side_r_d)


_LV = _build_level_arrays()


class HeroConstants(AutoDerivedConstants):
    # --- Screen (ALE native) ---
    screen_width: int = struct.field(pytree_node=False, default=160)
    screen_height: int = struct.field(pytree_node=False, default=210)
    cave_bottom: int = struct.field(pytree_node=False, default=142)

    num_levels: int = struct.field(pytree_node=False, default=_NUM_LEVELS)
    max_rooms: int = struct.field(pytree_node=False, default=_MAX_ROOMS)

    # --- Player: 6-px collision box; the drawn sprite is 9x24 and centred
    # over the box (measured box; the wider art matches the running pose). ---
    player_width: int = struct.field(pytree_node=False, default=6)
    player_height: int = struct.field(pytree_node=False, default=24)
    spawn_x: int = struct.field(pytree_node=False, default=HL.SPAWN[0])
    spawn_y: int = struct.field(pytree_node=False, default=HL.SPAWN[1])
    respawn_x: int = struct.field(pytree_node=False, default=HL.RESPAWN[0])
    respawn_y: int = struct.field(pytree_node=False, default=HL.RESPAWN[1])

    # Room flip thresholds (player TOP y; the flip happens when the measured
    # centre row crosses the screen edge: centre = top + 11).
    flip_bottom_y: int = struct.field(pytree_node=False, default=133)
    flip_enter_top_y: int = struct.field(pytree_node=False, default=-3)
    # Where a side exit lands him, measured on level 5 (HL.SIDE_ENTER_X). It
    # is a property of the EDGE he left by, not of which way along the chain
    # that edge goes: off the LEFT edge he arrives near the right of the other
    # room, off the RIGHT edge near its left. His y does not change. Level 7
    # room 10 leaves by the right edge and the ROM puts him at x 16 in room
    # 11, which is the same rule.
    side_enter_left_x: int = struct.field(pytree_node=False,
                                          default=HL.SIDE_ENTER_X[-1])
    side_enter_right_x: int = struct.field(pytree_node=False,
                                           default=HL.SIDE_ENTER_X[1])

    # --- Flight physics (measured) ---
    fall_speed: float = struct.field(pytree_node=False, default=1.0)
    thrust_accel: float = struct.field(pytree_node=False, default=0.125)
    thrust_spinup: int = struct.field(pytree_node=False, default=16)
    max_rise_speed: float = struct.field(pytree_node=False, default=-2.0)
    # DOWN held in mid-air sinks faster (measured behaviour; the rate itself
    # is a guess - the ROM capture only established that it is faster).
    sink_speed: float = struct.field(pytree_node=False, default=2.0)
    move_speed: int = struct.field(pytree_node=False, default=1)

    # --- Laser (measured frame by frame on the ROM) ---
    # Firing does NOT draw a beam that grows out of the helmet. It launches a
    # short bolt that flies away and is relaunched while the button is held,
    # so what the player sees is the hero's two eye pixels and a detached
    # dash travelling off into the dark.
    #
    # Measured at level 1 room 0, hero facing right from x 25 (eye row 79):
    #   f1 x 30-37   f2 33-40   f3 36-43   f4 39-46   f5 42-49   f6 45-52
    #   f7 restarts near the helmet again
    # so: 8 px long, 1 px tall, 3 px per frame, 6 frames of flight, then a
    # fresh bolt. On release the bolt in flight finishes its frame and goes.
    laser_bolt_length: int = struct.field(pytree_node=False, default=8)
    laser_bolt_speed: int = struct.field(pytree_node=False, default=3)
    laser_bolt_frames: int = struct.field(pytree_node=False, default=6)
    laser_height: int = struct.field(pytree_node=False, default=1)
    # the bolt leaves on the hero's eye row, 4 rows below his sprite top
    laser_eye_offset: int = struct.field(pytree_node=False, default=4)
    # The beam also EATS ROCK. Measured on the ROM by pinning the hero
    # against level 1 room 0's pillar and holding fire: cell 13 fell at
    # frame 257 and cell 14 at frame 513 - 256 frames of fire per 4 px
    # column, clearing it from the CEILING and MIDDLE bands together and
    # never the floor, for no points at all. The burn is remembered when the
    # button is released (200 frames + a 120-frame pause + 57 more = 257).
    # Walls whose run reaches a screen edge never melt, however long you
    # fire - 700 frames against room 0's left wall and room 1's right wall
    # did nothing - which is what keeps the hero inside the cave.
    laser_burn_frames: int = struct.field(pytree_node=False, default=256)

    # --- Dynamite (HERO_SPEC.md section 6) ---
    # A stick is placed with DOWN while standing on solid ground; DOWN in
    # mid-air only makes the hero sink faster. The blast takes the wall
    # column it touches and costs a life if the hero has not got clear.
    #
    # FUSE: the ROM's fuse is 34 frames (dyn_fuse_rom below). This build runs
    # a 60-frame fuse instead, because at the measured 1 px/frame walk speed
    # 34 frames leaves a human almost no escape margin. The divergence is
    # deliberate and is recorded here rather than hidden: dyn_fuse_playable
    # is what the game uses, dyn_fuse_rom is what the console does.
    dyn_width: int = struct.field(pytree_node=False, default=3)   # ROM bitmap
    dyn_height: int = struct.field(pytree_node=False, default=10)  # 3x10
    dyn_fuse_playable: int = struct.field(pytree_node=False, default=60)
    dyn_fuse_rom: int = struct.field(pytree_node=False, default=34)
    explosion_frames: int = struct.field(pytree_node=False, default=4)
    # Vertical reach of the blast. MEASURED TRUTH (not yet implemented): the
    # blast takes the CEILING cell and the MIDDLE cell of the column together
    # and never touches the floor band. The levels captured so far model the
    # descent through sealed floors as blasting floor-band rects, so this
    # build still uses a symmetric vertical radius; the band rule lands with
    # the per-level rebuild.
    explosion_radius: int = struct.field(pytree_node=False, default=16)
    # Horizontal reach, measured: a wall with up to 5 px of clear air between
    # it and the hero's own edge comes down; at 9 px nothing happens.
    blast_reach: int = struct.field(pytree_node=False, default=5)
    # Measured self-damage: retreating 8 px or less from the stick costs a
    # life, 10 px or more is safe.
    blast_safe_gap: int = struct.field(pytree_node=False, default=10)
    starting_dynamite: int = struct.field(pytree_node=False, default=6)

    # --- Power / lives / scoring (HERO_SPEC.md sections 6 and 7) ---
    # The gauge is 78 bar pixels wide and each pixel is 68 frames of MOVEMENT
    # (walking or hovering). Standing perfectly still costs nothing at all -
    # 3,000 idle frames were measured against an unmoved gauge.
    power_frames_per_pixel: int = struct.field(pytree_node=False, default=68)
    max_power: int = struct.field(pytree_node=False, default=78 * 68)
    power_drain_per_frame: int = struct.field(pytree_node=False, default=1)
    # brief creature-proof grace after a respawn (the respawn spot can sit
    # inside a creature's patrol zone, as on the real console)
    respawn_invuln: int = struct.field(pytree_node=False, default=60)
    # 4 lives at the start (measured) and an extra one every 20,000 points,
    # up to 6 heroes IN RESERVE - the HUD draws lives - 1 reserve icons, so
    # the cap on the counter is 7.
    starting_lives: int = struct.field(pytree_node=False, default=4)
    max_lives: int = struct.field(pytree_node=False, default=7)
    extra_life_score: int = struct.field(pytree_node=False, default=20000)
    creature_points: int = struct.field(pytree_node=False, default=50)
    wall_points: int = struct.field(pytree_node=False, default=75)
    miner_points: int = struct.field(pytree_node=False, default=1000)
    # End-of-level tally: measured as an animated payout "almost all of it in
    # ticks of 20", proportional to the power left. 20 points per bar pixel
    # still lit puts a full gauge at 1560, which is the measured range for a
    # clean level-1 clear (1300-1600).
    bonus_per_power_pixel: int = struct.field(pytree_node=False, default=20)

    # --- Creatures and magma (the "spiders" arrays hold every kind):
    # kind 0 spider: 6 thread rows then a 5 row body; the drawn canvas is
    #   7x12 because the thread stretches a pixel between poses. Bob
    #   amplitude is per creature, measured (HL.CREATURE_MOTION);
    # kind 1 bat: X-wing critter, small bob, patrols horizontally (measured
    #   patrol range per creature);
    # kind 3 snake: a green tongue anchored in a wall's side that stretches
    #   out of the rock and pulls back on a 64-frame cycle (_snake_length),
    #   kills on touch and dies to the laser like any other creature
    #   (re-measured on the ROM 2026-09-22; the earlier "laser-immune" note
    #   was an artefact of a probe that counted the rock as the snake);
    # kind 2 MAGMA: not a creature at all but a red block of the cave.
    #   Static, lethal on contact, immune to the laser - but DESTRUCTIBLE BY
    #   DYNAMITE for the same 75 points as ordinary rock (measured on the
    #   ROM; on level 9 room 0 blasting the red pillar is the only way
    #   down). Earlier builds of this file called it a "torch" and had it
    #   immune to the blast as well, which was wrong. ---
    num_spiders: int = struct.field(pytree_node=False, default=_MAX_SPIDERS)
    spider_width: int = struct.field(pytree_node=False, default=7)
    spider_height: int = struct.field(pytree_node=False, default=11)
    # the bat is 11 rows of wing throughout and has no thread above it, so
    # its whole sprite is body (CHARACTERS.md, measured on level 3)
    bat_height: int = struct.field(pytree_node=False, default=11)
    # the snake has no thread either: all 7 rows of it are body. Its WIDTH is
    # not a constant at all - it is however far out of the rock the stretch
    # cycle has it on this frame (_snake_length), from nothing to 7 px.
    # Measured on level 4 rooms 3 and 7, anchored at x 112/88, rows 72-78.
    snake_height: int = struct.field(pytree_node=False, default=7)
    spider_body_top: int = struct.field(pytree_node=False, default=6)
    # The untethered spider (kind 4) is seven rows of body, and its second
    # pose is drawn two rows lower, so the box it is ever inside is ten.
    free_spider_height: int = struct.field(pytree_node=False, default=10)
    # Archetype fallbacks, used only for a creature nobody has measured yet.
    # A measured creature carries its own amplitude, period and pose hold in
    # SPIDER_BOB / SPIDER_BOB_HALF / SPIDER_HOLD (HL.CREATURE_MOTION), which
    # is what the env and the renderer actually read - the ROM's creatures do
    # not share one rhythm.
    spider_bob_amp: int = struct.field(pytree_node=False,
                                       default=_DEFAULT_BOB[0])
    bat_bob_amp: int = struct.field(pytree_node=False, default=_DEFAULT_BOB[1])
    creature_anim_period: int = struct.field(pytree_node=False,
                                             default=_DEFAULT_ANIM_HOLD)
    spider_bob_half_period: int = struct.field(pytree_node=False,
                                               default=_DEFAULT_BOB_HALF)
    spider_patrol_half_period: int = struct.field(
        pytree_node=False, default=_DEFAULT_PATROL_HALF)

    # --- Lanterns (3x4 lamp; touch or blast darkens the room, measured) ---
    num_lanterns: int = struct.field(pytree_node=False, default=_MAX_LANTERNS)
    lantern_width: int = struct.field(pytree_node=False, default=5)   # ROM bitmap
    lantern_height: int = struct.field(pytree_node=False, default=8)  # 5x8

    # --- Miner (sprite 8x12) ---
    miner_width: int = struct.field(pytree_node=False, default=8)
    miner_height: int = struct.field(pytree_node=False, default=12)

    num_walls: int = struct.field(pytree_node=False, default=_MAX_WALLS)

    # --- Level-indexed measured data ---
    LEVEL_ROOMS: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["rooms_n"], dtype=jnp.int32))
    # (level, room) -> the corridor reaches that edge of the screen as open
    # air, so walking into it leaves the room. Measured; see HL.SIDE_EXITS.
    SIDE_EXIT_LEFT: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["side_l"], dtype=jnp.bool_))
    SIDE_EXIT_RIGHT: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["side_r"], dtype=jnp.bool_))
    # ... and which way along the chain that edge goes: +1 down, -1 back up.
    # Level 5 descends through a left edge, level 7 through a right one.
    SIDE_EXIT_LEFT_DELTA: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["side_l_d"], dtype=jnp.int32))
    SIDE_EXIT_RIGHT_DELTA: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["side_r_d"], dtype=jnp.int32))
    ROOM_WALLS: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["walls"], dtype=jnp.int32))
    ROOM_WALL_VALID: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["wall_valid"], dtype=jnp.bool_))
    LEVEL_MINER: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["miner"], dtype=jnp.int32))
    SPIDER_ROOM: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["sp_room"], dtype=jnp.int32))
    SPIDER_X: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["sp_x"], dtype=jnp.int32))
    SPIDER_Y: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["sp_y"], dtype=jnp.int32))
    SPIDER_PATROL: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["sp_patrol"], dtype=jnp.int32))
    SPIDER_KIND: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["sp_kind"], dtype=jnp.int32))
    # measured per creature, not per kind; see HL.CREATURE_MOTION
    SPIDER_BOB: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["sp_bob"], dtype=jnp.int32))
    SPIDER_BOB_HALF: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["sp_bob_half"], dtype=jnp.int32))
    SPIDER_BOB_BASE: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["sp_bob_base"], dtype=jnp.int32))
    SPIDER_HOLD: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["sp_hold"], dtype=jnp.int32))
    # distinct sprites the ROM draws this creature as; 0 = use the kind's
    # full cycle (HL.CREATURE_SPRITES)
    SPIDER_POSES: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["sp_poses"], dtype=jnp.int32))
    SPIDER_PATROL_HALF: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["sp_patrol_half"], dtype=jnp.int32))
    SPIDER_VALID: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["sp_valid"], dtype=jnp.bool_))
    LANTERN: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["lan"], dtype=jnp.int32))
    LANTERN_VALID: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["lan_valid"], dtype=jnp.bool_))
    # --- Deadly zones (L7-10, measured): water strips kill when stood in
    # (clipped to non-gap columns so falling through a floor gap is safe,
    # exactly like the ROM's slow-sink drowning). ---
    num_deadly: int = struct.field(pytree_node=False, default=_MAX_DEADLY)
    DEADLY_R: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["de"], dtype=jnp.int32))
    DEADLY_VALID: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["de_valid"], dtype=jnp.bool_))
    # --- Flare-ups (L7-10, measured at gap mouths): the ROM erupts them on
    # approach; recreated as readable periodic cycles — deadly while
    # (step_counter % period) < duty. ---
    num_flares: int = struct.field(pytree_node=False, default=_MAX_FLARES)
    FLARES_T: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["fl"], dtype=jnp.int32))
    FLARES_VALID: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["fl_valid"], dtype=jnp.bool_))
    # Magma: (room, x, y, w, h) rects of glowing rock, taken from the '%'
    # cells of the reference band strings (HL.MAGMA). Magma is cave, not a
    # creature: it is already solid because it is in the wall rects, and it is
    # already blastable where it forms an interior pillar because that is a
    # DESTRUCTIBLE zone. What this table adds is the burn - the hero dies the
    # moment his box TOUCHES one, so the lethal box is the rect grown by a
    # pixel on every side (he can never overlap it; collision stops him first).
    num_magma: int = struct.field(pytree_node=False, default=_MAX_MAGMA)
    MAGMA_R: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["mg"], dtype=jnp.int32))
    MAGMA_VALID: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["mg_valid"], dtype=jnp.bool_))
    # Destructible walls: (room, x, y, w, h, dynamite_ok) per slot. A dynamite
    # blast destroys a dyn_ok wall outright. The laser gets there too, but
    # slowly and column by column - see laser_burn_frames.
    num_dwalls: int = struct.field(pytree_node=False, default=_MAX_DWALLS)
    DESTRUCT: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["dw"], dtype=jnp.int32))
    DESTRUCT_VALID: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["dw_valid"], dtype=jnp.bool_))
    # Shared walls (see HL.SHARED_WALLS): slots with the same non-zero group
    # id are the same physical wall drawn on two screens - they break
    # together and pay 75 once.
    DESTRUCT_GROUP: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["dw_group"], dtype=jnp.int32))
    DESTRUCT_SCORES: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["dw_scores"], dtype=jnp.bool_))
    # (level, room, band, cell) occupancy of the 4 px cave grid, and which of
    # those columns the laser is allowed to eat. See _build_level_arrays.
    num_cells: int = struct.field(pytree_node=False, default=_CELLS)
    CELL_SOLID: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["cell_solid"], dtype=jnp.bool_))
    MELTABLE: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["meltable"], dtype=jnp.bool_))

    # --- HUD layout (measured from the ROM frame) ---
    hud_top: int = struct.field(pytree_node=False, default=142)
    hud_bottom: int = struct.field(pytree_node=False, default=189)
    power_bar_x: int = struct.field(pytree_node=False, default=49)
    power_bar_y: int = struct.field(pytree_node=False, default=145)
    power_bar_width: int = struct.field(pytree_node=False, default=78)
    power_bar_height: int = struct.field(pytree_node=False, default=5)
    lives_x: int = struct.field(pytree_node=False, default=81)
    lives_y: int = struct.field(pytree_node=False, default=152)
    lives_spacing: int = struct.field(pytree_node=False, default=8)
    dyn_icons_x: int = struct.field(pytree_node=False, default=59)
    dyn_icons_y: int = struct.field(pytree_node=False, default=166)
    dyn_icons_spacing: int = struct.field(pytree_node=False, default=8)
    # The score's digit cells sit at columns 65, 73, 81, 89 and 97, six
    # pixels wide on an eight pixel pitch, on rows 179-186. Measured twice:
    # off the ROM with the score at 0 (one digit, columns 97-102) and off
    # the recorded playthrough with a five digit score. `score_right_x` is
    # the number `x_start = score_right_x - n_digits * pitch` is taken from,
    # so it is 97 + 8. It was 103, which drew the whole score two pixels
    # left of where the ROM draws it.
    score_right_x: int = struct.field(pytree_node=False, default=105)
    score_y: int = struct.field(pytree_node=False, default=179)
    score_digit_pitch: int = struct.field(pytree_node=False, default=8)

    # --- The level banner (HERO_SPEC.md section 5) ---
    # At the start of every level the score line is replaced by "LEVEL:  n"
    # in the power gauge's yellow. Measured on the ROM: the word sits at
    # x 59-86 on rows 179-186 and the number is right aligned into the
    # score's own digit cells, so it shares score_right_x and the pitch.
    #
    # It stays up for exactly 111 frames, and the count does not depend on
    # what the player does - booting level 1 and holding NOOP, and booting it
    # and holding RIGHT, both show it on frames 0-110 and not on 111. That is
    # 1.85 s, the "about 2 seconds" HERO_SPEC.md records, and it matches the
    # recorded playthroughs, which are 5 fps and show it on frames 1-8.
    level_banner_frames: int = struct.field(pytree_node=False, default=111)
    level_banner_x: int = struct.field(pytree_node=False, default=59)

    # --- Colors (measured) ---
    bg_color: Tuple[int, int, int] = struct.field(pytree_node=False, default=(0, 0, 0))
    hud_color: Tuple[int, int, int] = struct.field(pytree_node=False, default=(111, 111, 111))
    laser_color: Tuple[int, int, int] = struct.field(pytree_node=False, default=(200, 72, 72))
    # POWER bar: full #e8e84a, eaten from the right in #a71a1a (measured)
    power_color: Tuple[int, int, int] = struct.field(pytree_node=False, default=(232, 232, 74))
    power_spent_color: Tuple[int, int, int] = struct.field(pytree_node=False, default=(167, 26, 26))
    text_color: Tuple[int, int, int] = struct.field(pytree_node=False, default=(214, 214, 214))
    explosion_color: Tuple[int, int, int] = struct.field(pytree_node=False, default=(252, 232, 120))

    def compute_derived(self):
        return {}


# ---------------------------------------------------------------------------
# State / Observation / Info
# ---------------------------------------------------------------------------
@struct.dataclass
class HeroState:
    player_x: chex.Array          # sprite-left, screen px
    player_y: chex.Array          # sprite-top, screen px (room-local)
    player_vy: chex.Array         # <0 rising, >0 falling
    thrust_timer: chex.Array      # frames UP has been held (rotor spin-up)
    room: chex.Array              # current room within the level
    facing: chex.Array            # -1 left, +1 right
    walk_timer: chex.Array
    has_moved: chex.Array         # power only drains after the first move
    laser_timer: chex.Array       # frames of continuous fire; 0 = not firing
    melted: chex.Array            # (max_rooms, 38) columns the beam has eaten
    burn_cell: chex.Array         # column the beam is currently eating, -1 = none
    burn_timer: chex.Array        # frames of fire accumulated on that column
    power: chex.Array
    lives: chex.Array
    score: chex.Array
    level: chex.Array
    dynamite_count: chex.Array
    dyn_active: chex.Array
    dyn_x: chex.Array
    dyn_y: chex.Array
    dyn_room: chex.Array
    dyn_fuse: chex.Array
    explosion_timer: chex.Array
    spider_alive: chex.Array      # (num_spiders,)
    wall_stage: chex.Array        # (num_dwalls,) 0 intact / 2 blasted away
    room_dark: chex.Array         # (max_rooms,) lantern destroyed -> dark
    invuln_timer: chex.Array      # creature-proof frames after a respawn
    miner_rescued: chex.Array
    level_complete: chex.Array
    banner_timer: chex.Array      # frames of "LEVEL: n" left over the score
    step_counter: chex.Array
    game_over: chex.Array
    rng_key: chex.PRNGKey


@struct.dataclass
class HeroObservation:
    player: ObjectObservation
    laser: ObjectObservation
    spiders: ObjectObservation
    miner: ObjectObservation
    dynamite: ObjectObservation
    walls: ObjectObservation
    lanterns: ObjectObservation
    power: chex.Array
    lives: chex.Array
    score: chex.Array
    dynamite_count: chex.Array
    level: chex.Array
    room: chex.Array


@struct.dataclass
class HeroInfo:
    time: chex.Array
    power: chex.Array
    lives: chex.Array
    dynamite_count: chex.Array
    all_rewards: chex.Array


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class JaxHero(JaxEnvironment[HeroState, HeroObservation, HeroInfo, HeroConstants]):
    # Compact agent action set -> ALE console actions.
    # DOWN (and DOWNFIRE) lays dynamite, as on the real console. DOWNLEFT /
    # DOWNRIGHT plant while already running, so "plant and flee" works as a
    # single held input (a human holding LEFT and tapping DOWN must not stall).
    ACTION_SET: jnp.ndarray = jnp.array([
        Action.NOOP,
        Action.FIRE,
        Action.UP,
        Action.RIGHT,
        Action.LEFT,
        Action.DOWN,
        Action.UPRIGHT,
        Action.UPLEFT,
        Action.RIGHTFIRE,
        Action.LEFTFIRE,
        Action.UPFIRE,
        Action.DOWNFIRE,
        Action.DOWNRIGHT,
        Action.DOWNLEFT,
    ], dtype=jnp.int32)

    def __init__(self, consts: HeroConstants = None):
        if consts is None:
            consts = HeroConstants()
        super().__init__(consts)
        self.renderer = HeroRenderer(self.consts)

    # --- reset ------------------------------------------------------------
    def reset(self, key: jax.random.PRNGKey = None) -> Tuple[HeroObservation, HeroState]:
        if key is None:
            key = jax.random.PRNGKey(0)
        c = self.consts
        state = HeroState(
            player_x=jnp.array(c.spawn_x, dtype=jnp.int32),
            player_y=jnp.array(c.spawn_y, dtype=jnp.int32),
            player_vy=jnp.array(0.0, dtype=jnp.float32),
            thrust_timer=jnp.array(0, dtype=jnp.int32),
            room=jnp.array(0, dtype=jnp.int32),
            facing=jnp.array(1, dtype=jnp.int32),
            walk_timer=jnp.array(0, dtype=jnp.int32),
            has_moved=jnp.array(False, dtype=jnp.bool_),
            laser_timer=jnp.array(0, dtype=jnp.int32),
            melted=jnp.zeros((c.max_rooms, c.num_cells), dtype=jnp.bool_),
            burn_cell=jnp.array(-1, dtype=jnp.int32),
            burn_timer=jnp.array(0, dtype=jnp.int32),
            power=jnp.array(c.max_power, dtype=jnp.int32),
            lives=jnp.array(c.starting_lives, dtype=jnp.int32),
            score=jnp.array(0, dtype=jnp.int32),
            level=jnp.array(0, dtype=jnp.int32),
            dynamite_count=jnp.array(c.starting_dynamite, dtype=jnp.int32),
            dyn_active=jnp.array(False, dtype=jnp.bool_),
            dyn_x=jnp.array(0, dtype=jnp.int32),
            dyn_y=jnp.array(0, dtype=jnp.int32),
            dyn_room=jnp.array(0, dtype=jnp.int32),
            dyn_fuse=jnp.array(0, dtype=jnp.int32),
            explosion_timer=jnp.array(0, dtype=jnp.int32),
            spider_alive=self.consts.SPIDER_VALID[0],
            wall_stage=jnp.zeros((c.num_dwalls,), dtype=jnp.int32),
            room_dark=jnp.zeros((c.max_rooms,), dtype=jnp.bool_),
            invuln_timer=jnp.array(0, dtype=jnp.int32),
            miner_rescued=jnp.array(False, dtype=jnp.bool_),
            level_complete=jnp.array(False, dtype=jnp.bool_),
            banner_timer=jnp.array(c.level_banner_frames, dtype=jnp.int32),
            step_counter=jnp.array(0, dtype=jnp.int32),
            game_over=jnp.array(False, dtype=jnp.bool_),
            rng_key=key,
        )
        return self._get_observation(state), state

    # --- helpers ----------------------------------------------------------
    @staticmethod
    def _aabb(ax, ay, aw, ah, bx, by, bw, bh):
        return ((ax < bx + bw) & (ax + aw > bx) &
                (ay < by + bh) & (ay + ah > by))

    def _spider_pos(self, state):
        """Current (x, y) of every creature slot of the level.

        One creature, one rhythm: the bob uses that creature's own measured
        travel and period, so a spider measured at 0 px hangs dead still
        however its kind usually behaves. Plus the measured horizontal
        patrol (halfwidth 0 = no patrol).
        """
        return _creature_pos(self.consts, state.level, state.step_counter)

    def _dwall_rects(self, state):
        """Solid rect of every destructible wall of the level: intact = the
        full rect, blasted (stage 2) = returned with w=0 so it never collides
        and the passage behind it opens."""
        c = self.consts
        dw = c.DESTRUCT[state.level]                    # (nD, 6)
        broken = state.wall_stage >= 2
        w = jnp.where(broken, 0, dw[:, 3])
        solid = c.DESTRUCT_VALID[state.level] & (~broken)
        return dw[:, 0], dw[:, 1], dw[:, 2], w, dw[:, 4], solid

    def _hits_wall(self, state, px, py):
        """Player box vs the current room's measured wall rects (+ the
        remaining parts of the destructible walls). The top 2 rows of the
        sprite are the 1-px rotor shaft and do not collide (the blast hole in
        level 2 is 23 px tall and the real player fits through it)."""
        c = self.consts
        box_y, box_h = py + 2, c.player_height - 2
        lvl, room = state.level, state.room
        walls = c.ROOM_WALLS[lvl, room]
        valid = c.ROOM_WALL_VALID[lvl, room]
        hit = jnp.any(self._aabb(px, box_y, c.player_width, box_h,
                                 walls[:, 0], walls[:, 1], walls[:, 2], walls[:, 3]) & valid)
        d_room, dx, dy_, dwid, dh, solid = self._dwall_rects(state)
        d_hit = jnp.any(solid & (d_room == room) &
                        self._aabb(px, box_y, c.player_width, box_h,
                                   dx, dy_, dwid, dh))
        # A column the beam has eaten is a hole through the ceiling and
        # middle bands. The cell grid is rasterised from these same rects, so
        # with nothing melted it agrees with them and this changes nothing;
        # it can only ever REMOVE a collision, never add one.
        return (hit | d_hit) & self._grid_hit(state, px, box_y, box_h)

    def _grid_hit(self, state, px, box_y, box_h):
        """Does the player box still meet solid rock on the 4 px cell grid?"""
        c = self.consts
        cell_x = 8 + 4 * jnp.arange(c.num_cells)
        over_x = (cell_x < px + c.player_width) & (cell_x + 4 > px)
        band_lo = jnp.array([16, 60, 99], dtype=jnp.int32)
        band_hi = jnp.array([59, 98, 141], dtype=jnp.int32)
        over_y = (band_lo <= box_y + box_h - 1) & (band_hi >= box_y)
        gone = state.melted[state.room]                       # (cells,)
        # the floor band is never eaten
        eaten = gone[None, :] & jnp.array([True, True, False])[:, None]
        solid = c.CELL_SOLID[state.level, state.room] & (~eaten)
        return jnp.any(solid & over_y[:, None] & over_x[None, :])

    # --- step -------------------------------------------------------------
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: HeroState, action: int) -> Tuple[HeroObservation, HeroState, float, bool, HeroInfo]:
        c = self.consts
        atari_action = jnp.take(self.ACTION_SET, action)
        lvl = state.level

        up = jnp.isin(atari_action, jnp.array(
            [Action.UP, Action.UPRIGHT, Action.UPLEFT, Action.UPFIRE], dtype=jnp.int32))
        left = jnp.isin(atari_action, jnp.array(
            [Action.LEFT, Action.UPLEFT, Action.LEFTFIRE, Action.DOWNLEFT], dtype=jnp.int32))
        right = jnp.isin(atari_action, jnp.array(
            [Action.RIGHT, Action.UPRIGHT, Action.RIGHTFIRE, Action.DOWNRIGHT], dtype=jnp.int32))
        down = jnp.isin(atari_action, jnp.array(
            [Action.DOWN, Action.DOWNFIRE, Action.DOWNLEFT, Action.DOWNRIGHT], dtype=jnp.int32))
        laser_fire = jnp.isin(atari_action, jnp.array(
            [Action.FIRE, Action.UPFIRE, Action.LEFTFIRE, Action.RIGHTFIRE], dtype=jnp.int32))

        # --- lay dynamite (DOWN while standing on solid ground, measured;
        # DOWN in mid-air only makes him sink faster, see below) ---
        on_ground = self._hits_wall(state, state.player_x, state.player_y + 2)
        can_place = (down & on_ground & (state.dynamite_count > 0) &
                     (~state.dyn_active) & (state.explosion_timer <= 0))
        dyn_active = state.dyn_active | can_place
        dyn_x = jnp.where(can_place, state.player_x - 2, state.dyn_x).astype(jnp.int32)
        dyn_y = jnp.where(can_place, state.player_y + c.player_height - c.dyn_height, state.dyn_y).astype(jnp.int32)
        dyn_room = jnp.where(can_place, state.room, state.dyn_room).astype(jnp.int32)
        dyn_fuse = jnp.where(can_place, c.dyn_fuse_playable, state.dyn_fuse).astype(jnp.int32)
        dynamite_count = jnp.where(can_place, state.dynamite_count - 1, state.dynamite_count).astype(jnp.int32)

        # --- horizontal movement (measured 1 px/frame) ---
        dx = (right.astype(jnp.int32) - left.astype(jnp.int32)) * c.move_speed
        cand_x = jnp.clip(state.player_x + dx, 8, c.screen_width - 8 - c.player_width).astype(jnp.int32)
        x_blocked = self._hits_wall(state, cand_x, state.player_y)
        new_x = jnp.where(x_blocked, state.player_x, cand_x).astype(jnp.int32)
        new_facing = jnp.where(right, 1, jnp.where(left, -1, state.facing)).astype(jnp.int32)

        # --- vertical: constant fall; thrust cancels the fall instantly,
        # hovers through the measured 16-frame rotor spin-up, then ramps ---
        thrust_timer = jnp.where(up, state.thrust_timer + 1, 0).astype(jnp.int32)
        vy_up = jnp.where(thrust_timer <= c.thrust_spinup, 0.0,
                          jnp.maximum(jnp.minimum(state.player_vy, 0.0) - c.thrust_accel,
                                      c.max_rise_speed))
        # DOWN in mid-air is not a dynamite press: it makes him sink faster
        # (measured). On the ground it plants a stick and does nothing here.
        sink = down & (~up) & (~on_ground)
        new_vy = jnp.where(up, vy_up,
                           jnp.where(sink, c.sink_speed, c.fall_speed)).astype(jnp.float32)
        dy = jnp.round(new_vy).astype(jnp.int32)
        cand_y = (state.player_y + dy).astype(jnp.int32)
        # ceiling of the topmost room: don't fly above the level start screen
        cand_y = jnp.where((state.room == 0) & (cand_y < 8), jnp.int32(8), cand_y)
        y_blocked = self._hits_wall(state, new_x, cand_y)
        new_y = jnp.where(y_blocked, state.player_y, cand_y).astype(jnp.int32)
        new_vy = jnp.where(y_blocked, 0.0, new_vy).astype(jnp.float32)

        # --- room flips (measured screen-flip model) ---
        n_rooms = c.LEVEL_ROOMS[lvl]
        flip_down = (new_y > c.flip_bottom_y) & (state.room < n_rooms - 1)
        flip_up = (new_y < c.flip_enter_top_y) & (state.room > 0)
        new_room = (state.room + flip_down.astype(jnp.int32) - flip_up.astype(jnp.int32)).astype(jnp.int32)
        new_y = jnp.where(flip_down, c.flip_enter_top_y,
                          jnp.where(flip_up, c.flip_bottom_y, new_y)).astype(jnp.int32)
        # keep the player inside the last room's floor
        new_y = jnp.clip(new_y, -24, c.cave_bottom - 2)

        # The cave also turns sideways. Where a room's corridor reaches the
        # edge of the screen as open air, walking into that edge leaves the
        # room exactly as falling through the floor does, and the hero keeps
        # his height. Level 5 room 1 is sealed floor to ceiling and this is the
        # only way on from it; so is level 7 room 10, whose floor is all
        # liquid.
        #
        # WHICH WAY an edge goes is measured per room, not fixed. Level 5
        # descends through a LEFT edge and level 7 through a RIGHT one, so the
        # delta comes out of HL.SIDE_EXITS. Measured on the ROM, 2026-09-22.
        x_min, x_max = 8, c.screen_width - 8 - c.player_width
        side_l = c.SIDE_EXIT_LEFT[lvl, state.room] & left & (new_x <= x_min)
        side_r = c.SIDE_EXIT_RIGHT[lvl, state.room] & right & (new_x >= x_max)
        # a room flip has already happened this frame -> that one wins
        flipped = flip_down | flip_up
        side_l = side_l & ~flipped
        side_r = side_r & ~flipped
        delta = jnp.where(side_l, c.SIDE_EXIT_LEFT_DELTA[lvl, state.room],
                          jnp.where(side_r,
                                    c.SIDE_EXIT_RIGHT_DELTA[lvl, state.room],
                                    0)).astype(jnp.int32)
        # never off either end of the chain
        delta = jnp.where((state.room + delta < 0)
                          | (state.room + delta > n_rooms - 1), 0, delta)
        took_side = delta != 0
        side_l = side_l & took_side
        side_r = side_r & took_side
        new_room = (new_room + delta).astype(jnp.int32)
        new_x = jnp.where(side_l, c.side_enter_left_x,
                          jnp.where(side_r, c.side_enter_right_x,
                                    new_x)).astype(jnp.int32)

        moved_input = up | left | right | down
        has_moved = state.has_moved | moved_input
        moved_h = new_x != state.player_x
        walk_timer = jnp.where(moved_h, state.walk_timer + 1, 0).astype(jnp.int32)

        # --- laser: a bolt that flies, relaunched while fire is held ---
        laser_timer = jnp.where(laser_fire, state.laser_timer + 1, 0).astype(jnp.int32)
        laser_on = laser_timer > 0
        lx, ly = _bolt_pos(self.consts, new_x, new_y, new_facing, laser_timer)

        # --- the beam eats rock (measured: 256 frames per 4 px column) ---
        # The column it is eating is the first solid, meltable one the bolt
        # reaches, in the band the bolt is flying through. The burn is kept
        # when the button is released and only restarts on a new column.
        # The bolt sweeps its whole range every laser_bolt_frames frames, so
        # what it is eating is the first solid column anywhere in that sweep,
        # not just the one the bolt happens to cover this frame.
        cell_x = 8 + 4 * jnp.arange(c.num_cells)
        beam_band = jnp.where(ly < 60, 0, jnp.where(ly < 99, 1, 2))
        reach = (c.laser_bolt_frames - 1) * c.laser_bolt_speed + c.laser_bolt_length
        lo = jnp.where(new_facing < 0, new_x - reach, new_x + c.player_width)
        hi = lo + reach - 1
        gone_here = state.melted[new_room]
        reachable = ((cell_x <= hi) & (cell_x + 3 >= lo) &
                     c.CELL_SOLID[lvl, new_room, beam_band] &
                     c.MELTABLE[lvl, new_room, beam_band] & (~gone_here))
        order = jnp.where(new_facing < 0, -jnp.arange(c.num_cells),
                          jnp.arange(c.num_cells))          # nearest first
        nearest = jnp.argmin(jnp.where(reachable, order, jnp.int32(1 << 20)))
        burning = laser_on & jnp.any(reachable)
        target = jnp.where(burning, nearest, -1).astype(jnp.int32)
        same = burning & (target == state.burn_cell)
        burn_timer = jnp.where(same, state.burn_timer + 1,
                               jnp.where(burning, 1, state.burn_timer)).astype(jnp.int32)
        burn_cell = jnp.where(burning, target, state.burn_cell).astype(jnp.int32)
        melt_now = burning & (burn_timer >= c.laser_burn_frames)
        melted = state.melted.at[new_room, burn_cell].max(melt_now & (burn_cell >= 0))
        burn_timer = jnp.where(melt_now, 0, burn_timer).astype(jnp.int32)
        burn_cell = jnp.where(melt_now, -1, burn_cell).astype(jnp.int32)

        # --- dynamite fuse / explosion ---
        new_fuse = jnp.where(dyn_active, dyn_fuse - 1, dyn_fuse).astype(jnp.int32)
        explode_now = dyn_active & (new_fuse <= 0)
        explosion_timer = jnp.where(explode_now, c.explosion_frames,
                                    jnp.maximum(0, state.explosion_timer - 1)).astype(jnp.int32)
        dyn_active = dyn_active & (~explode_now)

        # Blast boxes (measured, HERO_SPEC.md section 6).
        #   destruction: reaches a wall with up to blast_reach px of clear
        #     air between it and the hero's own edge at the moment he planted
        #     the stick (dyn_x is his left edge minus 2), nothing at 9 px;
        #   self-damage: a tighter box - he dies at a gap of 8 px or less and
        #     lives at 10 px or more.
        r = c.explosion_radius
        ey, eh = dyn_y - r, 2 * r + c.dyn_height
        reach = c.blast_reach + 1                  # +1: 5 px of air still goes
        ex = dyn_x + 2 - reach
        ew = c.player_width + 2 * reach
        kill_pad = c.blast_safe_gap - 1            # 10 px clear is safe
        kx = dyn_x - kill_pad
        kw = c.dyn_width + 2 * kill_pad
        blast_here = explode_now & (dyn_room == new_room)

        # --- destructible walls: a dynamite blast destroys a dyn_ok wall
        # outright (the laser does not affect walls) ---
        dw = c.DESTRUCT[lvl]                              # (nD, 6)
        d_room, d_x, d_y, d_w, d_h, d_solid = self._dwall_rects(state)
        dyn_break = (d_solid & (dw[:, 5] > 0) & explode_now & (d_room == dyn_room) &
                     self._aabb(ex, ey, ew, eh, d_x, d_y, d_w, d_h))
        wall_stage = jnp.where(dyn_break, 2, state.wall_stage).astype(jnp.int32)
        # Shared walls: the ROM keeps one copy of each distinct band pattern,
        # so a wall drawn on two screens is one wall. Slots with the same
        # non-zero group id go together. (Level 1 used to be cited for this
        # and no longer is - its room 1 has no pillar at all; see the
        # SHARED_WALLS comment in hero_levels.py.)
        grp = c.DESTRUCT_GROUP[lvl]
        broke_direct = (state.wall_stage < 2) & (wall_stage >= 2)
        shared = (grp[:, None] == grp[None, :]) & (grp[:, None] > 0)
        wall_stage = jnp.where((broke_direct[:, None] & shared).any(axis=0),
                               2, wall_stage).astype(jnp.int32)
        # a wall removed this frame scores once (+75) — the other slots of a
        # shared wall, and the twins below, are the same physical wall and do
        # not score again
        walls_broken = jnp.sum((((state.wall_stage < 2) & (wall_stage >= 2)) &
                                c.DESTRUCT_SCORES[lvl]).astype(jnp.int32))
        # The screen flip also splits one wall across two rooms (a room's
        # floor band IS the next room's top band on the real console).
        # Breaking a floor-band slot therefore also opens the x-overlapping
        # top-band slot of the room below.
        broke_now = (state.wall_stage < 2) & (wall_stage >= 2)
        d_x0, d_x1 = dw[:, 1], dw[:, 1] + dw[:, 3]
        twin = (broke_now[:, None] & (dw[:, 2][:, None] >= 99) &
                (dw[:, 2][None, :] < 60) &
                (dw[:, 0][None, :] == dw[:, 0][:, None] + 1) &
                (d_x0[None, :] < d_x1[:, None]) & (d_x1[None, :] > d_x0[:, None]))
        wall_stage = jnp.where(twin.any(axis=0), 2, wall_stage).astype(jnp.int32)

        # --- creatures and magma ---
        # Laser: kills every CREATURE - spiders, bats and snakes alike - for
        # 50 points. It does nothing to magma (kind 2) and nothing to rock.
        # A snake used to be laser-proof here, on the strength of one probe
        # whose liveness test counted every non-black pixel in the snake's
        # box; a snake's box overlaps the rock it is planted in, so that test
        # could only ever report "alive". Re-measured 2026-09-22 with a green
        # -only test and a spider control that passes: 18 of 18 aligned
        # bursts killed level 4 room 3's snake, each paying 50.
        # Dynamite: kills creatures AND removes magma. Magma is destroyed by
        # a stick exactly like ordinary rock, for the same 75 points
        # (measured on the ROM - on level 9 room 0 blowing up the red pillar
        # is the only way down). It stays lethal on contact and stays immune
        # to the laser.
        sp_x, sp_y = self._spider_pos(state)
        sp_room = c.SPIDER_ROOM[lvl]
        is_magma = c.SPIDER_KIND[lvl] == 2
        laser_killable = ~is_magma
        blast_killable = c.SPIDER_KIND[lvl] != 3          # snakes sit in rock
        sp_here = state.spider_alive & (sp_room == new_room)
        # A spider's body hangs under a thread, so only the rows from
        # spider_body_top down are it. A BAT is body all the way: 11 rows of
        # wing, top-aligned on the canvas (CHARACTERS.md), so its box is the
        # whole sprite and not the spider's lower five rows.
        # A SNAKE has no thread either, and its box is only 7 rows, so it
        # is body from sp_y down like the bat - not sp_y + 6. Taking the
        # spider's offset put level 4's two snakes' kill box six rows BELOW
        # the sprite, in the rock under it.
        # A snake's WIDTH is however much of it is out of the rock on this
        # frame (_snake_length), so the box is the sprite: pulled in, it is
        # 0 px wide and there is nothing there to shoot, blast or walk into.
        is_bat = c.SPIDER_KIND[lvl] == 1
        is_snake = c.SPIDER_KIND[lvl] == 3
        # kind 4, the untethered spider, has no thread over it either, so it
        # is body from its own top row down like the bat and the snake.
        is_free = c.SPIDER_KIND[lvl] == 4
        top_aligned = is_bat | is_snake | is_free
        body_y = jnp.where(top_aligned, sp_y, sp_y + c.spider_body_top)
        body_h = jnp.where(is_snake, c.snake_height,
                           jnp.where(is_bat, c.bat_height,
                                     jnp.where(is_free, c.free_spider_height,
                                               c.spider_height
                                               - c.spider_body_top)))
        body_w = jnp.where(is_snake, _snake_length(state.step_counter),
                           c.spider_width)
        # a 0-wide box overlaps nothing, which _aabb's strict inequalities do
        # not say on their own
        drawn = body_w > 0
        spider_laser = (laser_on & sp_here & laser_killable & drawn &
                        self._aabb(lx, ly, c.laser_bolt_length, c.laser_height,
                                   sp_x, body_y, body_w, body_h))
        spider_blast = (state.spider_alive & blast_killable & drawn &
                        (sp_room == dyn_room) & explode_now &
                        self._aabb(ex, ey, ew, eh, sp_x, body_y,
                                   body_w, body_h))
        spider_kill = spider_laser | spider_blast
        spider_alive = state.spider_alive & (~spider_kill)
        creatures_killed = jnp.sum((spider_kill & (~is_magma)).astype(jnp.int32))
        magma_blasted = jnp.sum((spider_kill & is_magma).astype(jnp.int32))

        # --- lanterns: touching one (or catching it in a blast) plunges the
        # room into darkness for the rest of the level (measured; the laser
        # does not affect lanterns) ---
        lan = c.LANTERN[lvl]                              # (nLan, 3)
        lan_alive = c.LANTERN_VALID[lvl] & (~state.room_dark[lan[:, 0]])
        lan_touch = (lan_alive & (lan[:, 0] == new_room) &
                     self._aabb(new_x, new_y, c.player_width, c.player_height,
                                lan[:, 1], lan[:, 2],
                                c.lantern_width, c.lantern_height))
        lan_blast = (lan_alive & (lan[:, 0] == dyn_room) & explode_now &
                     self._aabb(ex, ey, ew, eh, lan[:, 1], lan[:, 2],
                                c.lantern_width, c.lantern_height))
        lan_hit = lan_touch | lan_blast
        room_dark = state.room_dark | jnp.zeros_like(state.room_dark).at[
            lan[:, 0]].max(lan_hit)

        # --- player death conditions (creatures can't kill during the brief
        # post-respawn grace). Touch boxes per kind: spiders/bats use the
        # 7x5 body; magma blocks only their 3x4 core (playability tuning —
        # the ROM's magma-guarded gaps are threadable by hugging their edge,
        # measured on L5R5: deaths at ram-x 70-82, survival beside them). ---
        kind_l = c.SPIDER_KIND[lvl]
        tb_x = jnp.where(kind_l == 2, sp_x + 2, sp_x)
        tb_y = jnp.where(kind_l == 2, sp_y + c.spider_body_top + 1, body_y)
        tb_w = jnp.where(kind_l == 2, 3, body_w)
        tb_h = jnp.where(kind_l == 2, 4, body_h)
        touching = (sp_here & drawn &
                    self._aabb(new_x, new_y, c.player_width, c.player_height,
                               tb_x, tb_y, tb_w, tb_h))
        # Touching a creature kills the HERO, and holding fire does not save
        # him: measured on the ROM by walking into level 1 room 1's spider.
        # Not firing, he stops dead against it and loses a life 107 frames
        # later (the death freeze). Firing, the spider dies at f7 while he is
        # still 20 px away - killed by the bolt in flight, never by the
        # touch. An earlier build here let a touch-while-firing kill the
        # creature instead; that was wrong.
        died_spider = ((state.invuln_timer <= 0) &
                       jnp.any(touching & (~spider_kill)))
        died_blast = blast_here & self._aabb(new_x, new_y, c.player_width, c.player_height,
                                             kx, ey, kw, eh)

        # --- deadly zones (water strips) + periodic flare-ups (L7-10) ---
        # The liquid surface is SOLID - it is in the wall rects, on the rows
        # the ROM draws it - so the hero can never be inside it and the only
        # way to meet it is to come to rest on it. Its lethal box is therefore
        # the measured rect grown by one pixel, exactly as magma's is below
        # and for the same reason: standing on a rect leaves his feet on the
        # row above its top, which does not overlap the rect itself.
        de = c.DEADLY_R[lvl]
        died_deadly = ((state.invuln_timer <= 0) &
                       jnp.any(c.DEADLY_VALID[lvl] & (de[:, 0] == new_room) &
                               self._aabb(new_x, new_y, c.player_width,
                                          c.player_height, de[:, 1] - 1,
                                          de[:, 2] - 1, de[:, 3] + 2,
                                          de[:, 4] + 2)))
        # --- magma: cave rock that burns (HERO_SPEC.md section 3) ---
        # It is solid because it is in the wall rects, so the hero can never
        # be INSIDE it - the only way to meet it is to end up against it.
        # The lethal box is therefore the rect grown by one pixel all round.
        # A magma column blasted out with a stick stops burning with it.
        mg = c.MAGMA_R[lvl]
        dwr = c.DESTRUCT[lvl]
        gone = (wall_stage >= 2) & c.DESTRUCT_VALID[lvl]          # (nD,)
        # (nD, nMagma): does a blasted zone cover this magma rect?
        covers = (gone[:, None]
                  & (dwr[:, 0:1] == mg[None, :, 0])
                  & (dwr[:, 1:2] <= mg[None, :, 1])
                  & (dwr[:, 1:2] + dwr[:, 3:4] >= mg[None, :, 1] + mg[None, :, 3])
                  & (dwr[:, 2:3] <= mg[None, :, 2])
                  & (dwr[:, 2:3] + dwr[:, 4:5] >= mg[None, :, 2] + mg[None, :, 4]))
        magma_alive = c.MAGMA_VALID[lvl] & (~jnp.any(covers, axis=0))
        died_magma = ((state.invuln_timer <= 0) &
                      jnp.any(magma_alive & (mg[:, 0] == new_room) &
                              self._aabb(new_x, new_y, c.player_width,
                                         c.player_height,
                                         mg[:, 1] - 1, mg[:, 2] - 1,
                                         mg[:, 3] + 2, mg[:, 4] + 2)))

        flr = c.FLARES_T[lvl]
        flare_period = jnp.maximum(1, flr[:, 5])
        flare_active = (state.step_counter % flare_period) < flr[:, 6]
        died_flare = ((state.invuln_timer <= 0) &
                      jnp.any(c.FLARES_VALID[lvl] & flare_active &
                              (flr[:, 0] == new_room) &
                              self._aabb(new_x, new_y, c.player_width,
                                         c.player_height, flr[:, 1], flr[:, 2],
                                         flr[:, 3], flr[:, 4])))

        # --- power drain: measured, the gauge moves ONLY while the hero
        # walks or hovers. Standing perfectly still costs nothing (3,000
        # idle frames were measured against an unmoved gauge). ---
        burning_power = has_moved & (left | right | up)
        drain = jnp.where(burning_power, c.power_drain_per_frame, 0)
        new_power = jnp.maximum(0, state.power - drain).astype(jnp.int32)
        died_power = (new_power <= 0) & (state.power > 0)

        # --- miner rescue (last room of the level) ---
        m = c.LEVEL_MINER[lvl]
        touch_miner = ((~state.miner_rescued) & (new_room == m[0]) &
                       self._aabb(new_x, new_y, c.player_width, c.player_height,
                                  m[1], m[2], c.miner_width, c.miner_height))
        # End-of-level tally: 20 points per power-bar pixel still lit
        # (measured as ticks of 20 proportional to the power left).
        power_pixels = (new_power // c.power_frames_per_pixel).astype(jnp.int32)
        power_bonus = jnp.where(touch_miner,
                                power_pixels * c.bonus_per_power_pixel,
                                0).astype(jnp.int32)

        is_last = lvl >= (c.num_levels - 1)
        finish = touch_miner & is_last
        advance = touch_miner & (~is_last)

        # --- scoring (+ extra life every 20000 points, manual) ---
        # measured: wall 75, magma column 75 (the same as rock), creature 50,
        # miner 1000, then the end-of-level tally
        new_score = (state.score
                     + creatures_killed * c.creature_points
                     + (walls_broken + magma_blasted) * c.wall_points
                     + touch_miner.astype(jnp.int32) * c.miner_points
                     + power_bonus).astype(jnp.int32)
        extra_lives = (new_score // c.extra_life_score - state.score // c.extra_life_score).astype(jnp.int32)

        # --- death / lives / respawn (top of the CURRENT room, measured) ---
        died = (died_blast | died_spider | died_power | died_deadly |
                died_flare | died_magma) & (~touch_miner)
        new_lives = jnp.clip(state.lives - died.astype(jnp.int32) + extra_lives,
                             0, c.max_lives).astype(jnp.int32)
        respawned = died & (new_lives > 0)

        next_lvl = jnp.clip(lvl + advance.astype(jnp.int32), 0, c.num_levels - 1)
        # The measured respawn spot can sit inside a wall in some rooms (e.g.
        # level 2 room 1's pillar): fall back to the spawn column, then to
        # the descent-shaft column, whichever is free in this room.
        rs_state = state.replace(room=new_room)
        rs_bad_1 = self._hits_wall(rs_state, jnp.int32(c.respawn_x), jnp.int32(c.respawn_y))
        rs_bad_2 = self._hits_wall(rs_state, jnp.int32(c.spawn_x), jnp.int32(c.spawn_y))
        rs_x = jnp.where(rs_bad_1, jnp.where(rs_bad_2, 76, c.spawn_x), c.respawn_x)
        rs_y = jnp.where(rs_bad_1, jnp.where(rs_bad_2, 8, c.spawn_y), c.respawn_y)
        final_x = jnp.where(advance, c.spawn_x,
                            jnp.where(respawned, rs_x, new_x)).astype(jnp.int32)
        final_y = jnp.where(advance, c.spawn_y,
                            jnp.where(respawned, rs_y, new_y)).astype(jnp.int32)
        final_room = jnp.where(advance, 0, new_room).astype(jnp.int32)

        reset_pose = respawned | advance
        final_vy = jnp.where(reset_pose, 0.0, new_vy).astype(jnp.float32)
        final_thrust = jnp.where(reset_pose, 0, thrust_timer).astype(jnp.int32)
        final_facing = jnp.where(reset_pose, 1, new_facing).astype(jnp.int32)
        final_walk_timer = jnp.where(reset_pose, 0, walk_timer).astype(jnp.int32)
        final_has_moved = has_moved & (~reset_pose)
        final_laser = jnp.where(reset_pose, 0, laser_timer).astype(jnp.int32)
        # melted rock stays melted for the rest of the level, like a blasted
        # wall and like the darkness; a fresh level starts whole again
        final_melted = jnp.where(advance | finish, False, melted)
        final_burn_cell = jnp.where(advance | finish, -1, burn_cell).astype(jnp.int32)
        final_burn_timer = jnp.where(advance | finish, 0, burn_timer).astype(jnp.int32)
        final_dyn_active = dyn_active & (~reset_pose)
        final_explosion = jnp.where(reset_pose, 0, explosion_timer).astype(jnp.int32)
        final_dyn_fuse = jnp.where(reset_pose, 0, new_fuse).astype(jnp.int32)
        # Power and dynamite refill on respawn or advance (as on the console:
        # every new life carries six fresh sticks — the deep levels' sealed
        # rooms need more than one life's worth of dynamite, and trading a
        # life for sticks is the intended economy there).
        final_power = jnp.where(reset_pose, c.max_power, new_power).astype(jnp.int32)
        final_dyn_count = jnp.where(reset_pose, c.starting_dynamite, dynamite_count).astype(jnp.int32)

        final_spider_alive = jnp.where(advance, c.SPIDER_VALID[next_lvl], spider_alive)
        # walls reset intact on a new level; a respawn keeps a blasted wall gone
        final_wall_stage = jnp.where(advance, 0, wall_stage).astype(jnp.int32)
        # darkness lasts until the end of the level (a respawn keeps it)
        final_room_dark = jnp.where(advance, False, room_dark)
        final_invuln = jnp.where(respawned, c.respawn_invuln,
                                 jnp.maximum(0, state.invuln_timer - 1)).astype(jnp.int32)

        final_level = next_lvl.astype(jnp.int32)
        # The banner runs again on every new level, and only on a new level:
        # a respawn does not bring it back (the ROM shows it on level init,
        # not on a fresh life), and it runs on its own clock whatever the
        # player does.
        final_banner = jnp.where(
            advance, c.level_banner_frames,
            jnp.maximum(0, state.banner_timer - 1)).astype(jnp.int32)
        final_miner_rescued = jnp.where(advance, False, state.miner_rescued | touch_miner)
        level_complete = state.level_complete | finish

        game_over = state.game_over | (died & (new_lives <= 0))
        new_step = (state.step_counter + 1).astype(jnp.int32)

        new_state = HeroState(
            player_x=final_x,
            player_y=final_y,
            player_vy=final_vy,
            thrust_timer=final_thrust,
            room=final_room,
            facing=final_facing,
            walk_timer=final_walk_timer,
            has_moved=final_has_moved,
            laser_timer=final_laser,
            melted=final_melted,
            burn_cell=final_burn_cell,
            burn_timer=final_burn_timer,
            power=final_power,
            lives=new_lives,
            score=new_score,
            level=final_level,
            dynamite_count=final_dyn_count,
            dyn_active=final_dyn_active,
            dyn_x=dyn_x,
            dyn_y=dyn_y,
            dyn_room=dyn_room,
            dyn_fuse=final_dyn_fuse,
            explosion_timer=final_explosion,
            spider_alive=final_spider_alive,
            wall_stage=final_wall_stage,
            room_dark=final_room_dark,
            invuln_timer=final_invuln,
            miner_rescued=final_miner_rescued,
            level_complete=level_complete,
            banner_timer=final_banner,
            step_counter=new_step,
            game_over=game_over,
            rng_key=state.rng_key,
        )

        done = self._get_done(new_state)
        reward = self._get_reward(state, new_state)
        all_rewards = self._get_all_rewards(state, new_state)
        obs = self._get_observation(new_state)
        info = self._get_info(new_state, all_rewards)
        return obs, new_state, reward, done, info

    # --- observation / info / reward / done -------------------------------
    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: HeroState) -> HeroObservation:
        c = self.consts
        lvl = state.level
        player = ObjectObservation.create(
            x=state.player_x, y=state.player_y,
            width=jnp.array(c.player_width, jnp.int32),
            height=jnp.array(c.player_height, jnp.int32),
            active=jnp.array(True, jnp.bool_),
            orientation=jnp.where(state.facing < 0, 270, 90).astype(jnp.int32),
        )

        laser_on = state.laser_timer > 0
        lx, ly_obs = _bolt_pos(self.consts, state.player_x, state.player_y,
                                    state.facing, state.laser_timer)
        laser = ObjectObservation.create(
            x=lx, y=ly_obs,
            width=jnp.array(c.laser_bolt_length, jnp.int32),
            height=jnp.array(c.laser_height, jnp.int32),
            active=laser_on,
            orientation=jnp.where(state.facing < 0, 270, 90).astype(jnp.int32),
        )

        sp_x, sp_y = self._spider_pos(state)
        # same boxes the collision uses: a spider is the body under its
        # thread, a bat is all 11 rows of it, a snake all 7 of its box - and
        # a snake is only as WIDE as the part of it that is out of the rock
        obs_is_bat = c.SPIDER_KIND[lvl] == 1
        obs_is_snake = c.SPIDER_KIND[lvl] == 3
        obs_is_free = c.SPIDER_KIND[lvl] == 4
        obs_y = jnp.where(obs_is_bat | obs_is_snake | obs_is_free, sp_y,
                          sp_y + c.spider_body_top)
        obs_h = jnp.where(obs_is_snake, c.snake_height,
                          jnp.where(obs_is_bat, c.bat_height,
                                    jnp.where(obs_is_free,
                                              c.free_spider_height,
                                              c.spider_height
                                              - c.spider_body_top)))
        obs_w = jnp.where(obs_is_snake, _snake_length(state.step_counter),
                          c.spider_width)
        spiders = ObjectObservation.create(
            x=sp_x.astype(jnp.int32),
            y=obs_y.astype(jnp.int32),
            width=obs_w.astype(jnp.int32),
            height=obs_h.astype(jnp.int32),
            active=state.spider_alive & (c.SPIDER_ROOM[lvl] == state.room),
            visual_id=jnp.arange(c.num_spiders, dtype=jnp.int32),
        )

        m = c.LEVEL_MINER[lvl]
        miner = ObjectObservation.create(
            x=m[1].astype(jnp.int32), y=m[2].astype(jnp.int32),
            width=jnp.array(c.miner_width, jnp.int32),
            height=jnp.array(c.miner_height, jnp.int32),
            active=(~state.miner_rescued) & (state.room == m[0]),
        )

        dynamite = ObjectObservation.create(
            x=state.dyn_x, y=state.dyn_y,
            width=jnp.array(c.dyn_width, jnp.int32),
            height=jnp.array(c.dyn_height, jnp.int32),
            active=((state.dyn_active | (state.explosion_timer > 0)) &
                    (state.dyn_room == state.room)),
            state=state.explosion_timer,
        )

        walls_r = c.ROOM_WALLS[lvl, state.room]
        walls = ObjectObservation.create(
            x=walls_r[:, 0], y=walls_r[:, 1], width=walls_r[:, 2], height=walls_r[:, 3],
            active=c.ROOM_WALL_VALID[lvl, state.room],
        )

        lan = c.LANTERN[lvl]
        lanterns = ObjectObservation.create(
            x=lan[:, 1], y=lan[:, 2],
            width=jnp.full((c.num_lanterns,), c.lantern_width, jnp.int32),
            height=jnp.full((c.num_lanterns,), c.lantern_height, jnp.int32),
            active=(c.LANTERN_VALID[lvl] & (lan[:, 0] == state.room) &
                    (~state.room_dark[lan[:, 0]])),
        )

        return HeroObservation(
            player=player, laser=laser, spiders=spiders,
            miner=miner, dynamite=dynamite, walls=walls, lanterns=lanterns,
            power=state.power, lives=state.lives, score=state.score,
            dynamite_count=state.dynamite_count, level=state.level, room=state.room,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: HeroState, all_rewards: jnp.ndarray = None) -> HeroInfo:
        if all_rewards is None:
            all_rewards = jnp.zeros(1, dtype=jnp.float32)
        return HeroInfo(
            time=state.step_counter, power=state.power, lives=state.lives,
            dynamite_count=state.dynamite_count, all_rewards=all_rewards,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: HeroState, state: HeroState) -> float:
        return (state.score - previous_state.score).astype(jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_rewards(self, previous_state: HeroState, state: HeroState) -> jnp.ndarray:
        return jnp.array([self._get_reward(previous_state, state)], dtype=jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: HeroState) -> bool:
        return state.game_over | state.level_complete

    # --- spaces -----------------------------------------------------------
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTION_SET))

    def observation_space(self) -> spaces.Dict:
        c = self.consts
        screen = (c.screen_height, c.screen_width)
        return spaces.Dict({
            "player": spaces.get_object_space(n=None, screen_size=screen),
            "laser": spaces.get_object_space(n=None, screen_size=screen),
            "spiders": spaces.get_object_space(n=c.num_spiders, screen_size=screen),
            "miner": spaces.get_object_space(n=None, screen_size=screen),
            "dynamite": spaces.get_object_space(n=None, screen_size=screen),
            "walls": spaces.get_object_space(n=c.num_walls, screen_size=screen),
            "lanterns": spaces.get_object_space(n=c.num_lanterns, screen_size=screen),
            "power": spaces.Box(low=0, high=c.max_power, shape=(), dtype=jnp.int32),
            "lives": spaces.Box(low=0, high=c.max_lives, shape=(), dtype=jnp.int32),
            "score": spaces.Box(low=0, high=jnp.iinfo(jnp.int32).max, shape=(), dtype=jnp.int32),
            "dynamite_count": spaces.Box(low=0, high=c.starting_dynamite, shape=(), dtype=jnp.int32),
            "level": spaces.Box(low=0, high=c.num_levels, shape=(), dtype=jnp.int32),
            "room": spaces.Box(low=0, high=c.max_rooms, shape=(), dtype=jnp.int32),
        })

    def image_space(self) -> spaces.Box:
        c = self.consts
        return spaces.Box(low=0, high=255,
                          shape=(c.screen_height, c.screen_width, 3), dtype=jnp.uint8)

    def render(self, state: HeroState) -> jnp.ndarray:
        return self.renderer.render(state)


# ---------------------------------------------------------------------------
# Renderer — measured sprites and HUD over the measured room backgrounds.
# ---------------------------------------------------------------------------
# Pixel-art tables transcribed from ROM frame captures. Characters map to RGB
# via _ART_PALETTE ('.' = transparent).
_ART_PALETTE = {
    '.': None,
    'Y': (232, 232, 74),      # rotor / dynamite-icon fuse (yellow)
    'R': (184, 50, 50),       # helmet red
    'r': (200, 72, 72),       # helmet highlight
    'B': (84, 138, 210),      # suit blue
    'W': (214, 214, 214),     # legs / dynamite stick / digits (white)
    'S': (170, 170, 170),     # spider thread / miner girder / bat wing
    'T': (192, 192, 192),     # the bat's brightest wing row (#c0c0c0)
    '1': (223, 183, 85),      # spider body gradient (measured warm ramp)
    '2': (210, 164, 74),
    '3': (195, 144, 61),
    '4': (180, 122, 48),
    '5': (162, 98, 33),
    'P': (228, 111, 111),     # miner head pink
    'G': (92, 186, 92),       # miner body green / wall snake
    'g': (72, 160, 72),       # wall snake shading (measured L4)
    'h': (111, 210, 111),     # wall snake highlight
    'N': (111, 111, 111),     # the flat grey the ROM draws creatures in
    'M': (142, 142, 142),     # lamp bracket (#8e8e8e)
    'y': (252, 252, 84),      # miner lamp spark / lamp glow (#fcfc54)
    'b': (45, 87, 176),       # HUD mini-hero suit blue
    'K': (50, 132, 50),       # level-2 breakable pillar / wall snake fringe
}

# Roderick (facing right), 9 wide x 24 tall. Built from stacked sections:
# rotor (3) + helmet/prop-pack (5) + suit (6) + legs (10). The visual sprite
# is wider than the 6-px collision box (player_width) and is drawn centred
# over it (see the render call). While walking the legs animate through a
# running stride (wide<->narrow stance); while airborne the rotor spins and
# the legs dangle together.
# The rotor has THREE widths and they cycle one per frame, standing or
# hovering alike (CHARACTERS.md, "Roderick Hero / Animation"):
#
#     .#....        ###...        #####..
#     narrow        medium        wide
#
# The wide one is the resting T of the reference art. They all hang off the
# same shaft at column 3.
_HERO_TOP_NARROW = [
    "...Y.....",
    "...Y.....",
    "...Y.....",
]
_HERO_TOP_MEDIUM = [
    "..YYY....",
    "...Y.....",
    "...Y.....",
]
_HERO_TOP_WIDE = [
    ".YYYYY...",
    "...Y.....",
    "...Y.....",
]
_HERO_ROTOR = [_HERO_TOP_NARROW, _HERO_TOP_MEDIUM, _HERO_TOP_WIDE]
_HERO_HEAD = [               # red helmet (left) + prop-pack "c" (right)
    ".RR..RRR.",
    ".RR..R...",
    ".RR..RRR.",
    ".RR......",
    ".RR......",
]
_HERO_SUIT = [               # blue body with the arms notched out
    "BBBB.BBBB",
    "BBBBBBBBB",
    "BBBBBBBBB",
    ".BBBBBBB.",
    ".BB...BB.",
    "..B...B..",
]
_HERO_LEGS_CONTACT = [       # run, contact pose: front leg planted forward,
    "...WWW...",             # back leg trailing bent up behind (the pose of
    "..WWWWW..",             # the reference art)
    ".WWW.WW..",
    "WWW..WW..",
    "WW....WW.",
    "......WW.",
    "......WW.",
    ".....WW..",
    ".....WW..",
    ".....WWWW",
]
_HERO_LEGS_PASS = [          # run, passing pose: planted leg under the hip,
    "...WWW...",             # other knee lifted (alternates with contact)
    "..WWWWW..",
    "..WW.WW..",
    "..WW.WWW.",
    "..WW..WW.",
    "..WW.....",
    "..WW.....",
    "..WW.....",
    "..WW.....",
    "..WWWW...",
]
_HERO_LEGS_TOGETHER = [      # dangling (flight / idle)
    "...WWW...",
    "...WWW...",
    "..WWWWW..",
    "..WW.WW..",
    "..WW.WW..",
    "..WW.WW..",
    "..WW.WW..",
    "..WW.WW..",
    "..WW.WW..",
    "..WWWWW..",
]
# Standing and hovering both cycle the three rotor poses over dangling legs,
# a new pose every frame. Walking holds the wide rotor still and steps the
# legs instead, four frames to a pose - the two states genuinely do not
# animate at the same rate (CHARACTERS.md).
_PLAYER_STAND = [top + _HERO_HEAD + _HERO_SUIT + _HERO_LEGS_TOGETHER
                 for top in _HERO_ROTOR]
_PLAYER_WALK0 = _HERO_TOP_WIDE + _HERO_HEAD + _HERO_SUIT + _HERO_LEGS_CONTACT
_PLAYER_WALK1 = _HERO_TOP_WIDE + _HERO_HEAD + _HERO_SUIT + _HERO_LEGS_PASS

# Hanging spider, read pixel for pixel off the ROM in level 1 room 1
# (x 49-55, y 70-81): a 1 px silver thread and a 5-row body. The two poses
# are the SAME body one row apart - the thread stretches and the spider
# swings a pixel, it does not move its legs - and it swaps about every 8
# frames. The body is coloured by SCREEN ROW out of the level's own ramp,
# brightest at the top, which is why pose 2 is not pose 1 shifted: each row
# keeps its own shade.
_SPIDER_ART = [
    "...S...",
    "...S...",
    "...S...",
    "...S...",
    "...S...",
    "...S...",
    ".1.1.1.",
    "2.222.2",
    "3.333.3",
    "4..4..4",
    "5.....5",
    ".......",
]
_SPIDER_ART2 = [
    "...S...",
    "...S...",
    "...S...",
    "...S...",
    "...S...",
    "...S...",
    "...1...",
    ".2.2.2.",
    "3.333.3",
    "4.444.4",
    "5..5..5",
    "5.....5",
]

# Bat: the ROM's own bitmap, read pixel for pixel off level 7 room 5 and
# level 6 rooms 1 and 3 (tools/creature_atlas.py, 2026-09-23). 11 rows
# throughout, 5 px wide with the wings closed and 7 with them spread, and
# FOUR poses in the cycle - closed, straight, mid, spread - each held about
# 4 frames. Row 0 is the top of the bat, so the y in HL.SPIDERS, which is the
# top row the ROM draws it on, is where it lands.
#
# IT IS NOT ONE FLAT COLOUR. The ROM paints it in six shades laid out by row
# and MIRRORED about the body: #8e8e8e #aaaaaa #c0c0c0 for the wings, then
# #c3903d #b47a30 #a26221 for the body, then back out again. Drawing the
# whole bat in one grey - which is what this art did - loses the warm body
# CHARACTERS.md describes ("an orange body with grey wings above and below
# it") and makes every bat in the game a flat slab.
#
# The first three poses are 5 px wide and the spread one is 7; they are all
# centred on the same body column, so the narrow ones are padded out to the
# 7 px canvas the other creatures share and the x in HL.SPIDERS is the
# canvas's left edge, which is the leftmost column the ROM ever paints.
_BAT_ART = [                 # closed
    "...M...",
    "..SSS..",
    "..T.T..",
    "..3.3..",
    "..444..",
    ".55555.",
    "..444..",
    "..3.3..",
    "..T.T..",
    "..SSS..",
    "...M...",
    ".......",
]
_BAT_ART2 = [                # straight
    "..M.M..",
    "..S.S..",
    "..T.T..",
    "..3.3..",
    "..444..",
    ".55555.",
    "..444..",
    "..3.3..",
    "..T.T..",
    "..S.S..",
    "..M.M..",
    ".......",
]
_BAT_ART3 = [                # mid
    ".M...M.",
    ".SS.SS.",
    "..T.T..",
    "..3.3..",
    "..444..",
    ".55555.",
    "..444..",
    "..3.3..",
    "..T.T..",
    ".SS.SS.",
    ".M...M.",
    ".......",
]
_BAT_ART4 = [                # spread
    "M.....M",
    "SS...SS",
    ".TT.TT.",
    "..3.3..",
    "..444..",
    ".55555.",
    "..444..",
    "..3.3..",
    ".TT.TT.",
    "SS...SS",
    "M.....M",
    ".......",
]

# The UNTETHERED spider (kind 4). The same warm body as the hanging one and
# no thread at all: it floats in mid-corridor and flips between two poses,
# legs spread below the body and legs gathered above it. Read off the ROM at
# level 6 rooms 1, 5 and 7 and level 7 room 11 (tools/creature_atlas.py,
# 2026-09-23); levels 2, 3 and 4 carry it too.
#
# It is the creature CHARACTERS.md's per-room tables cannot name: the census
# tells a spider from a bat by whether a 1 px thread is drawn above the body,
# so with no thread it called this one a "Bat" where it sits still (level 7
# room 6) and a "Spider" where it bobs (level 6 rooms 1, 5, 7) - the same
# seven pixels wide of the same sprite, twice. It is neither: the bat is
# eleven rows of grey-and-orange wing and the hanging spider is a five row
# body under six rows of silver, and this is seven rows of body on its own.
#
# The two poses are NOT drawn on the same row. Pose 1 sits exactly TWO rows
# below pose 0 for the same position of the creature, which is why its
# measured box is 17 rows tall for a sprite that is never more than 8 - so
# the offset is baked into the canvas here and HL.CREATURE_MOTION carries the
# travel WITHOUT it (box height minus canvas height).
_SPIDER_FREE_ART = [         # legs up and out, body below
    "2.....2",
    "2.2.2.2",
    "33.3.33",
    "444.444",
    "4444444",
    ".44444.",
    "...4...",
    ".......",
    ".......",
    ".......",
    ".......",
    ".......",
]
_SPIDER_FREE_ART2 = [        # body up, legs gathered below - two rows lower
    ".......",
    ".......",
    "..3.3..",
    "...4...",
    "..444..",
    ".44444.",
    "44.4.44",
    "33...33",
    "2.....2",
    ".3...3.",
    ".......",
    ".......",
]

# Magma block: a red block of the cave, drawn here as a glowing column.
# Static, lethal on contact, immune to the laser, but a stick of dynamite
# takes it out like any other wall for 75 points (measured on the ROM).
_MAGMA_ART = [
    "...S...",
    "...S...",
    "...S...",
    "...S...",
    "...S...",
    "...S...",
    ".2.Y.2.",
    "..3Y3..",
    ".33433.",
    "..444..",
    "...4...",
    ".......",
]

# Snake: the ROM's stretching tongue, transcribed from 150 consecutive frames
# of level 4 room 3 (2026-09-22). It is anchored in the LEFT-hand rock and
# grows rightwards out of it, so column 0 of the canvas is the face of the
# wall and the sprite is `length` px wide - see _snake_length for the cycle.
# Its 7 rows sit in the top 7 of the 7x12 creature canvas, which is what the
# "7x7 box" in the older notes was describing.
#
# It is green whatever the level's hue, which is what tells it apart from
# everything else in the cave. All four of its shades are the ROM's:
# #5cba5c body, #48a048 outline, #6fd26f highlight and #328432 at the fringe.
#
# Every pose is this one shape, written against the tip t = length - 1 and
# clipped at the wall face, with the head in one of two flutters:
#
#     row 0  d    t-2, t-1                 row 3  h    0..t-2 (up) / 0..t-3
#     row 1  g    t-3, t-2, t                          (down), and t
#     row 2  G    0..t                      row 4  G    0..t (up) / 0..t-2
#                                           row 5  g    t-1 (up) / t-2..t
#                                           row 6  d    -  (up) / t-1 (down)
#
# The superseded bitmaps were four blobs that neither stretched nor matched
# any frame the ROM draws: they came from a reference whose "four poses" were
# a frame-differencer splitting the snake's disconnected pixels into pieces.
def _snake_art(length, head_down):
    """The ROM's snake with `length` px of it out of the rock (0 draws
    nothing), as rows of the shared 7x12 creature canvas."""
    rows = [["."] * _SNAKE_REACH for _ in range(12)]
    t = length - 1

    def put(row, cols, ch):
        for col in cols:
            if 0 <= col <= t:
                rows[row][col] = ch

    if length > 0:
        put(0, (t - 2, t - 1), 'K')                   # #328432, the fringe
        put(1, (t - 3, t - 2, t), 'g')
        put(2, range(t + 1), 'G')
        if head_down:
            put(3, list(range(t - 2)) + [t], 'h')
            put(4, range(t - 1), 'G')
            put(5, (t - 2, t - 1, t), 'g')
            put(6, (t - 1,), 'K')
        else:
            put(3, list(range(t - 1)) + [t], 'h')
            put(4, range(t + 1), 'G')
            put(5, (t - 1,), 'g')
    return ["".join(row) for row in rows]


# pose 0 is the empty canvas it draws while it is inside the rock, then
# 1-7 head up and 8-14 head down, exactly as _snake_pose indexes them
_SNAKE_ARTS = ([_snake_art(0, False)] +
               [_snake_art(n, False) for n in range(1, _SNAKE_REACH + 1)] +
               [_snake_art(n, True) for n in range(1, _SNAKE_REACH + 1)])

# Lantern lamp, exactly the ROM's bitmap (bestiary.json `lamp`, 5x8): a
# silver bracket over two bright yellow bands. Touching it (or blasting it)
# darkens the room for the rest of the level.
_LANTERN_ART = [
    "..MMM",
    "..M..",
    ".MMM.",
    "MMMMM",
    ".yyy.",
    "MMMMM",
    ".yyy.",
    "..y..",
]

# Trapped miner, 8 wide x 12 tall. Lifted pixel for pixel out of the ROM's
# own level 1 room 1: forcing the room register parks the hero out of shot but
# leaves the miner in the capture, and what is left matches the bitmap in
# HERO_SPEC.md section 4. The sprite's own origin is its top-left pixel, so
# it sits flush with the measured MINER_POS (level 1: x 25, y 86) instead of
# the 1 px inset the earlier art carried.
_MINER_ART = [
    "..y.....",
    ".SSS....",
    ".PP.....",
    ".PP.....",
    ".P......",
    "GGG.....",
    "G.G.....",
    "G.G.....",
    "G.G.G...",
    "SS.SSS..",
    "SSSS.S..",
    ".SS..SS.",
]

# A placed stick, exactly the ROM's bitmap (bestiary.json `dynamite`, 3x10):
# a red body under a yellow fuse burning down.
_DYN_ART = [
    "..Y",
    ".YY",
    ".Y.",
    ".Y.",
    "RRR",
    "RRR",
    "RRR",
    "RRR",
    "RRR",
    "RRR",
]

# (Destructible walls stay baked in the room backgrounds; the renderer
# stamps black over the melted/blasted slices — see _build_dwall_stamps.)

# HUD mini-hero life icon, 5 wide x 13 tall (measured).
_LIFE_ICON_ART = [
    "YYY..",
    ".Y...",
    ".R.RR",
    ".R.RR",
    ".b.b.",
    ".bb.b",
    "..b.b",
    "..b.b",
    "...W.",
    "...W.",
    "..WW.",
    "..W..",
    "..W..",
]

# HUD dynamite stick icon, 3 wide x 11 tall (measured).
_DYN_ICON_ART = [".YY"] * 5 + ["RRR"] * 6

# The ROM's 8-row score digit font, every glyph read off the real ROM.
#
# The digits are drawn in a 6 px cell on an 8 px pitch. The level banner
# and the score share this font and share the cell grid - the ROM draws a
# score of 0 and a "LEVEL: 10" with their last digit in exactly the same
# columns, 97-102 on rows 179-186 - so one table serves both.
#
# Captured by booting each of the 20 levels and reading the banner: levels
# 1-9 give the ones digit, level 10 gives 0 (and its tens digit re-reads as
# the same 1), level 20's tens digit re-reads as the same 2. Only '0' of the
# superseded table was ever checked against the ROM, and it was the only one
# that was right; the other nine were Activision-styled guesses. The
# differences are small but visible - the ROM's 4 has a one-pixel diagonal
# where the old table had a two-pixel one, its 1 has a short foot rather
# than a full-width one, and its 2, 3, 5, 7 and 9 are shaped differently
# again.
_DIGIT_FONT_8 = {
    0: ['011110', '110011', '110011', '110011', '110011', '110011', '110011', '011110'],
    1: ['001100', '011100', '001100', '001100', '001100', '001100', '001100', '011110'],
    2: ['011110', '100011', '000011', '000011', '011110', '110000', '110000', '111111'],
    3: ['011110', '100011', '000011', '000110', '000110', '000011', '100011', '011110'],
    4: ['000110', '001110', '010110', '100110', '111111', '000110', '000110', '000110'],
    5: ['111111', '110000', '110000', '111110', '000011', '000011', '100011', '111110'],
    6: ['011110', '110001', '110000', '111110', '110011', '110011', '110011', '011110'],
    7: ['111111', '100001', '000011', '000110', '001100', '001100', '001100', '001100'],
    8: ['011110', '110011', '110011', '011110', '011110', '110011', '110011', '011110'],
    9: ['011110', '110011', '110011', '110011', '011111', '000011', '100011', '011110'],
}

# The word "LEVEL:" of the level banner, read off the same capture. It is a
# different font from the little 5x3 one the POWER label uses: 8 rows tall,
# proportionally spaced (L and E are 4 px wide, V is 5, the colon 2), and
# identical in all 20 levels.
_BANNER_FONT = {
    'L': ['1100', '1100', '1100', '1100', '1100', '1100', '1111', '1111'],
    'E': ['1111', '1111', '1100', '1110', '1110', '1100', '1111', '1111'],
    'V': ['11011', '11011', '11011', '11011', '11011', '01110', '01110', '00100'],
    ':': ['11', '11', '11', '00', '00', '11', '11', '11'],
}
# (glyph, x offset from the word's left edge), as the ROM lays them out:
# L at 59, E at 64, V at 69, E at 75, L at 80, ':' at 85, ending at 86.
_BANNER_WORD = [('L', 0), ('E', 5), ('V', 10), ('E', 16), ('L', 21), (':', 26)]
_BANNER_WORD_W = 28

# 5x3 letter font for the HUD "POWER" label and the logo lettering.
_LETTER_FONT = {
    'P': ["111", "101", "111", "100", "100"],
    'O': ["111", "101", "101", "101", "111"],
    'W': ["101", "101", "101", "111", "101"],
    'E': ["111", "100", "111", "100", "111"],
    'R': ["111", "101", "111", "110", "101"],
    'A': ["111", "101", "111", "101", "101"],
    'C': ["111", "100", "100", "100", "111"],
    'T': ["111", "010", "010", "010", "010"],
    'I': ["111", "010", "010", "010", "111"],
    'V': ["101", "101", "101", "101", "010"],
    'S': ["111", "100", "111", "001", "111"],
    'N': ["101", "111", "111", "111", "101"],
    ' ': ["000", "000", "000", "000", "000"],
}


# ---------------------------------------------------------------------------
# Darkness (HERO_SPEC.md section 3, measured frame by frame)
#
# A room whose lamp has been knocked out is NOT dimmed and NOT washed grey.
# The three bands of rock are simply not drawn at all. What is left on the
# screen is:
#   * the wavy trim on rows 16-19 and 138-141, in the greys below;
#   * magma, still at its full #a71a1a red;
#   * the hero, the creatures, the lamp and a lit fuse, in their own colours
#     (those are sprites, so they come for free).
# A colour census of one measured dark room: 21,578 black pixels against 296
# of magma red and about 800 of trim grey, and nothing else.
# ---------------------------------------------------------------------------
_DARK_TRIM_ROWS = ((16, 20), (138, 142))               # [start, stop)
_DARK_TRIM_GREYS = ((142, 142, 142), (170, 170, 170),  # #8e8e8e #aaaaaa
                    (192, 192, 192), (214, 214, 214))  # #c0c0c0 #d6d6d6
# the four shades of a level's own hue that the trim is drawn in, in the same
# order as _DARK_TRIM_GREYS
_TRIM_SHADE_KEYS = ("light", "hi1", "hi2", "hi3")
_MAGMA_DARK = (167, 26, 26)                            # #a71a1a
# magma alternates between these two frame to frame, which is the glow
_MAGMA_COLORS = ((167, 26, 26), (184, 50, 50))


def _hex_rgb(h: str) -> Tuple[int, int, int]:
    return (int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16))


class HeroRenderer(JAXGameRenderer):
    """Draws the measured per-room backgrounds (decoded from hero_levels.py)
    with dynamic sprites and the measured HUD on top. Rooms are static
    screens; state.room selects which background is shown (screen-flip)."""

    def __init__(self, consts: HeroConstants = None, config: render_utils.RendererConfig = None):
        self.consts = consts or HeroConstants()
        super().__init__(self.consts)
        c = self.consts

        if config is None:
            self.config = render_utils.RendererConfig(
                game_dimensions=(c.screen_height, c.screen_width), channels=3, downscale=None)
        else:
            self.config = config
        self.jr = render_utils.JaxRenderingUtils(self.config)

        nlev = HL.NUM_LEVELS
        palettes = [getattr(HL, f"PALETTE_L{i}") for i in range(1, nlev + 1)]
        blobs = [getattr(HL, f"BG_RLE_L{i}") for i in range(1, nlev + 1)]

        asset_config = [
            {'name': 'background', 'type': 'background', 'data': self._build_background()},
            {'name': 'player_rotor0', 'type': 'procedural', 'data': self._sprite(_PLAYER_STAND[0])},
            {'name': 'player_rotor1', 'type': 'procedural', 'data': self._sprite(_PLAYER_STAND[1])},
            {'name': 'player_rotor2', 'type': 'procedural', 'data': self._sprite(_PLAYER_STAND[2])},
            {'name': 'player_walk0', 'type': 'procedural', 'data': self._sprite(_PLAYER_WALK0)},
            {'name': 'player_walk1', 'type': 'procedural', 'data': self._sprite(_PLAYER_WALK1)},
            {'name': 'spider', 'type': 'procedural', 'data': self._sprite(_SPIDER_ART)},
            {'name': 'spider2', 'type': 'procedural', 'data': self._sprite(_SPIDER_ART2)},
            {'name': 'spiderfree', 'type': 'procedural', 'data': self._sprite(_SPIDER_FREE_ART)},
            {'name': 'spiderfree2', 'type': 'procedural', 'data': self._sprite(_SPIDER_FREE_ART2)},
            {'name': 'bat', 'type': 'procedural', 'data': self._sprite(_BAT_ART)},
            {'name': 'bat2', 'type': 'procedural', 'data': self._sprite(_BAT_ART2)},
            {'name': 'bat3', 'type': 'procedural', 'data': self._sprite(_BAT_ART3)},
            {'name': 'bat4', 'type': 'procedural', 'data': self._sprite(_BAT_ART4)},
            {'name': 'magma', 'type': 'procedural', 'data': self._sprite(_MAGMA_ART)},
            *({'name': f'snake{i}', 'type': 'procedural',
               'data': self._sprite(art)}
              for i, art in enumerate(_SNAKE_ARTS)),
            {'name': 'lantern', 'type': 'procedural', 'data': self._sprite(_LANTERN_ART)},
            {'name': 'miner', 'type': 'procedural', 'data': self._sprite(_MINER_ART)},
            {'name': 'dynamite', 'type': 'procedural', 'data': self._sprite(_DYN_ART)},
            {'name': 'explosion', 'type': 'procedural', 'data': self._build_explosion(c.explosion_radius)},
            {'name': 'laser_bolt', 'type': 'procedural',
             'data': self._solid(c.laser_height, c.laser_bolt_length, c.laser_color)},
            {'name': 'power_unit', 'type': 'procedural', 'data': self._solid(c.power_bar_height, 1, c.power_color)},
            {'name': 'power_spent_unit', 'type': 'procedural', 'data': self._solid(c.power_bar_height, 1, c.power_spent_color)},
            {'name': 'life_icon', 'type': 'procedural', 'data': self._sprite(_LIFE_ICON_ART)},
            {'name': 'dyn_icon', 'type': 'procedural', 'data': self._sprite(_DYN_ICON_ART)},
            {'name': 'digits', 'type': 'digits', 'data': self._build_digits(c.text_color)},
            # the level banner re-uses the same digits in the power gauge's
            # yellow, which is the colour the ROM draws "LEVEL: n" in
            {'name': 'banner_digits', 'type': 'digits',
             'data': self._build_digits(c.power_color)},
            {'name': 'banner_word', 'type': 'procedural',
             'data': self._build_banner_word(c.power_color)},
        ]
        # measured per-room cave backgrounds (black-padded for short levels).
        # 16 levels x 16 rooms of full 142x160 screens is ~6M pixels, far too
        # many for the asset loader's per-pixel Python palette scan (minutes
        # per env). Only their COLOURS go through the loader, as one tiny
        # "palette seed" asset; the room images themselves are mapped to
        # colour ids with numpy below, once the palette exists.
        # The same goes for the destructible-wall stamps (16 x 129 x 3 padded
        # 120x44 stamps, ~33M mostly transparent pixels) and the flare stamps:
        # they are plain rectangles, built as id masks below.
        baked_colors = [(0, 0, 0)]                       # padding rooms / wall blanks
        for pal in palettes:
            for col in pal:
                col = tuple(int(v) for v in col)
                if col not in baked_colors:
                    baked_colors.append(col)
        for col in ((252, 232, 120), (184, 50, 50),      # flare flame tones
                    _MAGMA_DARK,                        # magma in a dark room
                    *_DARK_TRIM_GREYS):                 # trim in a dark room
            if col not in baked_colors:
                baked_colors.append(col)
        seed = np.zeros((1, len(baked_colors), 4), np.uint8)
        seed[0, :, 0:3] = np.array(baked_colors, np.uint8)
        seed[0, :, 3] = 255
        asset_config.append({'name': 'baked_palette', 'type': 'procedural',
                             'data': jnp.asarray(seed)})

        sprite_path = os.path.join(render_utils.get_base_sprite_dir(), "hero")
        (
            self.PALETTE, self.SHAPE_MASKS, self.BACKGROUND,
            self.COLOR_TO_ID, self.FLIP_OFFSETS,
        ) = self.jr.load_and_setup_assets(asset_config, sprite_path)

        # 0-2: the three rotor poses, used standing and hovering alike;
        # 3-4: the two walking strides. See _PLAYER_STAND.
        self.PLAYER_FRAMES = jnp.stack([
            self.SHAPE_MASKS["player_rotor0"],
            self.SHAPE_MASKS["player_rotor1"],
            self.SHAPE_MASKS["player_rotor2"],
            self.SHAPE_MASKS["player_walk0"],
            self.SHAPE_MASKS["player_walk1"],
        ])
        self.PLAYER_ROTOR_POSES = 3
        self.PLAYER_WALK_FRAME0 = 3
        # sprite per (kind, animation frame): 0 spider, 1 bat, 2 magma,
        # 3 snake - all on the same 7x12 canvas, the height the ROM's spider
        # needs when its thread is at full stretch.
        #
        # The kinds do not have the same NUMBER of poses either, not just the
        # same rate: the ROM's spider swaps between two, its bat runs a
        # four-pose wing cycle (CHARACTERS.md), so a bat animated over two
        # poses skips half its flap, and a snake has one bitmap per pixel of
        # stretch per head flutter (_snake_art). CREATURE_POSES says how many
        # of the slots below are real; the rest repeat so the gather is still
        # a fixed-size array.
        def _cycle(*names):
            frames = [self.SHAPE_MASKS[n] for n in names]
            return jnp.stack([frames[i % len(frames)]
                              for i in range(_SNAKE_POSES)])

        self.CREATURE_FRAMES = jnp.stack([
            _cycle("spider", "spider2"),
            _cycle("bat", "bat2", "bat3", "bat4"),
            _cycle("magma"),
            _cycle(*(f"snake{i}" for i in range(_SNAKE_POSES))),
            _cycle("spiderfree", "spiderfree2"),
        ])
        self.CREATURE_POSES = jnp.array([2, 4, 1, _SNAKE_POSES, 2],
                                        dtype=jnp.int32)
        self.BLACK_ID = jnp.asarray(self.COLOR_TO_ID[(0, 0, 0)])
        # room backgrounds as colour-id masks, same dtype as the raster they
        # are slotted into: decode each RLE screen to palette INDICES (via an
        # identity palette), then look the indices up in the loader's ids.
        mask_dtype = np.asarray(self.BACKGROUND).dtype
        bgs = np.full((c.num_levels, c.max_rooms, c.cave_bottom, c.screen_width),
                      self.COLOR_TO_ID[(0, 0, 0)], dtype=mask_dtype)
        for li in range(c.num_levels):
            n_col = len(palettes[li])
            assert n_col <= 256, "decode_bg yields uint8 palette indices"
            index_palette = [(k, k, k) for k in range(n_col)]
            lut = np.array([self.COLOR_TO_ID[tuple(int(v) for v in col)]
                            for col in palettes[li]], dtype=mask_dtype)
            for ri, blob in enumerate(blobs[li]):
                idx = HL.decode_bg(blob, index_palette)[..., 0]
                bgs[li, ri] = lut[idx]
        self.BGS = jnp.asarray(bgs)  # (nL, nR, 142, 160)
        self.BGS_DARK = jnp.asarray(self._darken(bgs))
        # a melted column is a 4 px hole through the ceiling and middle
        # bands (rows 16-98); the floor band is never eaten
        self.MELT_TOP, self.MELT_H = 16, 99 - 16
        self.MELT_STAMP = jnp.full((self.MELT_H, 4),
                                   self.COLOR_TO_ID[(0, 0, 0)], dtype=mask_dtype)
        transparent = self.jr.TRANSPARENT_ID
        black_id = self.COLOR_TO_ID[(0, 0, 0)]
        # black stamps for the destructible walls, padded to the largest wall.
        # wall_stage is 0 (intact, draws nothing) or 2 (blasted, blanks the
        # whole wall); an intact wall simply is not stamped, so only the
        # blasted stamp is stored - the three-stage table it replaced was the
        # renderer's biggest array by far, and two thirds of it transparent.
        max_dw_h = max(int(v) for v in _LV["dw"][:, :, 4].flatten()) or 1
        max_dw_w = max(int(v) for v in _LV["dw"][:, :, 3].flatten()) or 1
        dwall = np.full((c.num_levels, c.num_dwalls, max_dw_h, max_dw_w),
                        transparent, dtype=mask_dtype)
        for li in range(c.num_levels):
            for di in range(c.num_dwalls):
                if _LV["dw_valid"][li, di]:
                    _, _, _, w, h, _ = (int(v) for v in _LV["dw"][li, di])
                    dwall[li, di, :h, :w] = black_id  # opaque black over the wall
        self.DWALL_STAMPS = jnp.asarray(dwall)  # (nL, nD, H, W)
        # flare flames (L7-10): a solid warm stamp per (level, slot), padded
        max_fl_h = max(1, max(int(v) for v in _LV["fl"][:, :, 4].flatten()))
        max_fl_w = max(1, max(int(v) for v in _LV["fl"][:, :, 3].flatten()))
        flare = np.full((c.num_levels, c.num_flares, max_fl_h, max_fl_w),
                        transparent, dtype=mask_dtype)
        for li in range(c.num_levels):
            for fi in range(c.num_flares):
                if _LV["fl_valid"][li, fi]:
                    _, _, _, w, h, _, _ = (int(v) for v in _LV["fl"][li, fi])
                    flare[li, fi, :h, :w] = self.COLOR_TO_ID[(252, 232, 120)]
                    flare[li, fi, :h, 1:min(max(2, w - 1), w)] = self.COLOR_TO_ID[(184, 50, 50)]
        self.FLARE_STAMPS = jnp.asarray(flare)  # (nL, nF, H, W)

    # --- procedural asset builders ----------------------------------------
    @staticmethod
    def _solid(h: int, w: int, color: Tuple[int, int, int]) -> jnp.ndarray:
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, 0:3] = np.array(color, dtype=np.uint8)
        rgba[:, :, 3] = 255
        return jnp.asarray(rgba)

    @staticmethod
    def _sprite(art: List[str]) -> jnp.ndarray:
        h, w = len(art), len(art[0])
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        for r, row in enumerate(art):
            for col, ch in enumerate(row):
                color = _ART_PALETTE[ch]
                if color is not None:
                    rgba[r, col, 0:3] = np.array(color, dtype=np.uint8)
                    rgba[r, col, 3] = 255
        return jnp.asarray(rgba)

    def _build_explosion(self, radius: int) -> jnp.ndarray:
        """Brief radial flash (the real blast lasts 1-2 frames; approximate)."""
        c = self.consts
        size = 2 * radius
        yy, xx = np.mgrid[0:size, 0:size]
        dist = np.sqrt((xx - radius + 0.5) ** 2 + (yy - radius + 0.5) ** 2)
        rgba = np.zeros((size, size, 4), dtype=np.uint8)
        outer = dist <= radius
        core = dist <= radius * 0.5
        rgba[outer, 0:3] = np.array((232, 232, 74), dtype=np.uint8)
        rgba[outer, 3] = 255
        rgba[core, 0:3] = np.array(c.explosion_color, dtype=np.uint8)
        return jnp.asarray(rgba)

    def _build_digits(self, color: Tuple[int, int, int]) -> jnp.ndarray:
        digits = np.zeros((10, 8, 6, 4), dtype=np.uint8)
        for d, rows in _DIGIT_FONT_8.items():
            for r, row in enumerate(rows):
                for col, ch in enumerate(row):
                    if ch == "1":
                        digits[d, r, col, 0:3] = np.array(color, dtype=np.uint8)
                        digits[d, r, col, 3] = 255
        return jnp.asarray(digits)

    def _build_banner_word(self, color: Tuple[int, int, int]) -> jnp.ndarray:
        """The word "LEVEL:" as one 8 x 28 stamp, laid out as the ROM does.

        Proportional, not on a grid: L and E are 4 px wide, V is 5 and the
        colon 2, and each glyph starts where the ROM starts it.
        """
        art = np.zeros((8, _BANNER_WORD_W, 4), dtype=np.uint8)
        for name, dx in _BANNER_WORD:
            for r, row in enumerate(_BANNER_FONT[name]):
                for col, ch in enumerate(row):
                    if ch == "1":
                        art[r, dx + col, 0:3] = np.array(color, dtype=np.uint8)
                        art[r, dx + col, 3] = 255
        return jnp.asarray(art)

    def _darken(self, bgs: np.ndarray) -> np.ndarray:
        """The dark version of every baked room, as colour ids.

        Measured (HERO_SPEC.md section 3): rock is not drawn at all, the trim
        on rows 16-19 and 138-141 keeps its shape but is recoloured to the
        grey family, and magma stays at full red. Everything else goes black.
        """
        black = self.COLOR_TO_ID[(0, 0, 0)]
        out = np.full_like(bgs, black)
        trim = np.zeros(bgs.shape[2], bool)
        for r0, r1 in _DARK_TRIM_ROWS:
            trim[r0:r1] = True
        grey_ids = [self.COLOR_TO_ID[g] for g in _DARK_TRIM_GREYS]
        magma_ids = [self.COLOR_TO_ID[m] for m in _MAGMA_COLORS]
        magma_dark = self.COLOR_TO_ID[_MAGMA_DARK]
        for li in range(bgs.shape[0]):
            shades = HL.SHADES[li + 1]
            lit, dark = bgs[li], out[li]
            for key, grey in zip(_TRIM_SHADE_KEYS, grey_ids):
                src = self.COLOR_TO_ID.get(_hex_rgb(shades[key]))
                if src is None:            # shade never used by this level
                    continue
                hit = lit == src
                hit[:, ~trim, :] = False
                dark[hit] = grey
            for mid in magma_ids:
                dark[lit == mid] = magma_dark
        return out

    def _build_background(self) -> jnp.ndarray:
        """Static screen layer: black cave area (rooms composited per frame)
        + the measured HUD: gray panel rows 142-188 (x8+) and the POWER
        label. Rows 189-209 (the console's ACTIVISION logo strip) stay
        black - the wordmark is the publisher's mark, not part of the
        game, and is deliberately not reproduced."""
        c = self.consts
        bg = np.zeros((c.screen_height, c.screen_width, 4), dtype=np.uint8)
        bg[:, :, 3] = 255
        bg[c.hud_top:c.hud_bottom, 8:c.screen_width, 0:3] = np.array(c.hud_color, dtype=np.uint8)

        def text(s, x, y, color):
            cx = x
            for ch in s:
                glyph = _LETTER_FONT.get(ch, _LETTER_FONT[' '])
                for r, row in enumerate(glyph):
                    for cc, v in enumerate(row):
                        if v == '1':
                            bg[y + r, cx + cc, 0:3] = np.array(color, dtype=np.uint8)
                cx += 4
            return cx

        text("POWER", 26, c.power_bar_y, c.text_color)
        return jnp.asarray(bg)

    # --- render -----------------------------------------------------------
    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: HeroState) -> jnp.ndarray:
        c = self.consts
        lvl, room = state.level, state.room

        raster = self.jr.create_object_raster(self.BACKGROUND)
        # a room whose lantern was knocked out renders from its pre-baked dark
        # version: no rock at all, only the trim and the magma (see _darken)
        cave = jnp.where(state.room_dark[room],
                         self.BGS_DARK[lvl, room], self.BGS[lvl, room])
        raster = jax.lax.dynamic_update_slice(raster, cave, (0, 0))

        def maybe(cond, x, y, mask, ras, flip=False):
            return jax.lax.cond(
                cond,
                lambda r: self.jr.render_at_clipped(
                    r, jnp.asarray(x, jnp.int32), jnp.asarray(y, jnp.int32),
                    mask, flip_horizontal=flip),
                lambda r: r,
                ras,
            )

        # columns the beam has eaten: stamp them black, exactly the rows the
        # melt clears in the collision grid
        for ci in range(c.num_cells):
            raster = maybe(state.melted[room, ci], 8 + 4 * ci, self.MELT_TOP,
                           self.MELT_STAMP, raster)

        # destructible walls: baked in the background; stamp black over the
        # melted/blasted slices per the wall's current stage
        for di in range(c.num_dwalls):
            dwr = c.DESTRUCT[lvl, di]
            raster = maybe(c.DESTRUCT_VALID[lvl, di] & (room == dwr[0]) &
                           (state.wall_stage[di] >= 2),
                           dwr[1], dwr[2], self.DWALL_STAMPS[lvl, di], raster)

        # miner (last room)
        m = c.LEVEL_MINER[lvl]
        raster = maybe((~state.miner_rescued) & (room == m[0]),
                       m[1], m[2], self.SHAPE_MASKS["miner"], raster)

        # creatures: bob/patrol per creature; sprite selected by kind and by
        # its own pose cycle. Neither the rate nor the number of poses is
        # shared - a spider swaps between two poses every 8 frames and a bat
        # flaps through four every 4 (CHARACTERS.md). A SNAKE does not run a
        # pose cycle at all: its bitmap is how far out of the rock it is on
        # this frame, so it reads its stretch clock instead of SPIDER_HOLD
        # and draws nothing at all while it is pulled in.
        kind = c.SPIDER_KIND[lvl]
        sx, sy = _creature_pos(c, lvl, state.step_counter)
        snake_frame = _snake_pose(_snake_length(state.step_counter),
                                  _snake_head_down(state.step_counter))
        # How many poses this creature runs through is its own property, not
        # its kind's: the census counted the distinct sprites the ROM draws
        # each one as, and a creature it saw draw ONE all scan is still. A
        # slot nobody counted carries 0 and keeps the kind's whole cycle.
        poses = jnp.where(c.SPIDER_POSES[lvl] > 0,
                          c.SPIDER_POSES[lvl], self.CREATURE_POSES[kind])
        anim_frame = jnp.where(
            kind == 3, snake_frame,
            (state.step_counter // c.SPIDER_HOLD[lvl]) % poses)
        sp_room = c.SPIDER_ROOM[lvl]
        for i in range(c.num_spiders):
            raster = maybe(state.spider_alive[i] & (sp_room[i] == room),
                           sx[i], sy[i],
                           self.CREATURE_FRAMES[kind[i], anim_frame[i]], raster)

        # lanterns (alive while their room is still lit)
        lan = c.LANTERN[lvl]
        for i in range(c.num_lanterns):
            raster = maybe(c.LANTERN_VALID[lvl, i] & (lan[i, 0] == room) &
                           (~state.room_dark[lan[i, 0]]),
                           lan[i, 1], lan[i, 2], self.SHAPE_MASKS["lantern"], raster)

        # flare-ups (L7-10): drawn while their cycle is on
        flr = c.FLARES_T[lvl]
        fl_period = jnp.maximum(1, flr[:, 5])
        fl_on = (state.step_counter % fl_period) < flr[:, 6]
        for i in range(c.num_flares):
            raster = maybe(c.FLARES_VALID[lvl, i] & fl_on[i] &
                           (flr[i, 0] == room),
                           flr[i, 1], flr[i, 2], self.FLARE_STAMPS[lvl, i], raster)

        # dynamite + explosion flash
        dyn_here = state.dyn_room == room
        raster = maybe(state.dyn_active & dyn_here, state.dyn_x, state.dyn_y,
                       self.SHAPE_MASKS["dynamite"], raster)
        raster = maybe((state.explosion_timer > 0) & dyn_here,
                       state.dyn_x - c.explosion_radius, state.dyn_y - c.explosion_radius,
                       self.SHAPE_MASKS["explosion"], raster)

        # player: standing and hovering cycle the three rotor poses one frame
        # each; walking holds each of its two strides for four frames
        # (CHARACTERS.md - the states do not share a rate). Hovering through
        # the thrust spin-up counts as airborne.
        airborne = (jnp.abs(state.player_vy) > 0.5) | (state.thrust_timer > 0)
        rotor_frame = state.step_counter % self.PLAYER_ROTOR_POSES
        walk_frame = self.PLAYER_WALK_FRAME0 + (state.walk_timer // 4) % 2
        frame = jnp.where(airborne | (state.walk_timer <= 0),
                          rotor_frame, walk_frame)
        # the 9-px sprite is drawn centred over the 6-px collision box
        sprite_dx = (self.PLAYER_FRAMES.shape[2] - c.player_width) // 2
        raster = self.jr.render_at_clipped(
            raster, state.player_x - sprite_dx, state.player_y,
            self.PLAYER_FRAMES[frame],
            flip_horizontal=(state.facing < 0))

        # laser: one 8x1 bolt in flight on the hero's eye row (measured)
        lx0, ly0 = _bolt_pos(self.consts, state.player_x, state.player_y,
                                  state.facing, state.laser_timer)
        raster = maybe(state.laser_timer > 0, lx0, ly0,
                       self.SHAPE_MASKS["laser_bolt"], raster)

        # keep sprites out of the HUD band
        raster = raster.at[c.cave_bottom:, :].set(self.BACKGROUND[c.cave_bottom:, :])

        # --- HUD (measured layout) ---
        # POWER bar: 78 px on rows 145-149 from x 49, yellow and eaten from
        # the RIGHT in red - so every one of the 78 columns is painted, the
        # lit ones yellow and the spent ones red.
        #
        # While the level banner is up the whole bar is red, and it turns
        # yellow on the same frame the banner goes. Measured on the ROM at
        # level 1: on frame 0 rows 145-149 are #a71a1a from x 49 to 127 with
        # no yellow in them at all, on frame 150 they are #e8e84a from 49 to
        # 126, and after a long run right they are yellow to 120 and red from
        # 121. So the gauge reads "not started yet" for exactly the 111
        # frames of the banner - which is also what the recorded playthroughs
        # show on their first frames.
        power_units = jnp.where(state.banner_timer > 0, 0,
                                state.power // c.power_frames_per_pixel)

        def draw_power(i, ras):
            return jax.lax.cond(
                i < power_units,
                lambda r: self.jr.render_at_clipped(
                    r, jnp.array(c.power_bar_x) + i, jnp.array(c.power_bar_y),
                    self.SHAPE_MASKS["power_unit"]),
                lambda r: self.jr.render_at_clipped(
                    r, jnp.array(c.power_bar_x) + i, jnp.array(c.power_bar_y),
                    self.SHAPE_MASKS["power_spent_unit"]),
                ras,
            )
        raster = jax.lax.fori_loop(0, c.power_bar_width, draw_power, raster)

        # reserve lives as mini-heroes (lives - 1, measured 3 icons at start)
        raster = self.jr.render_indicator(raster, c.lives_x, c.lives_y,
                                          jnp.maximum(0, state.lives - 1),
                                          self.SHAPE_MASKS["life_icon"],
                                          spacing=c.lives_spacing,
                                          max_value=c.max_lives - 1)

        # dynamite sticks remaining
        raster = self.jr.render_indicator(raster, c.dyn_icons_x, c.dyn_icons_y,
                                          state.dynamite_count, self.SHAPE_MASKS["dyn_icon"],
                                          spacing=c.dyn_icons_spacing,
                                          max_value=c.starting_dynamite)

        # For the first 111 frames of a level the ROM replaces the score
        # line with "LEVEL: n" in the power gauge's yellow. It is the score's
        # own row and the score's own digit cells, so the two are drawn the
        # same way and simply take turns: whichever is showing renders its
        # digits, the other renders none.
        banner_on = state.banner_timer > 0
        raster = maybe(banner_on, c.level_banner_x, c.score_y,
                       self.SHAPE_MASKS["banner_word"], raster)

        def label(ras, value, masks, showing):
            digits = self.jr.int_to_digits(value, max_digits=6)
            n = jnp.maximum(1, (jnp.floor(jnp.log10(
                jnp.maximum(value, 1).astype(jnp.float32))) + 1).astype(jnp.int32))
            x0 = c.score_right_x - n * c.score_digit_pitch
            return self.jr.render_label_selective(
                ras, x0, c.score_y, digits, masks, 6 - n,
                n * showing.astype(jnp.int32),
                spacing=c.score_digit_pitch, max_digits_to_render=6)

        # the level is 0-based in the state and 1-based on the panel
        raster = label(raster, state.level + 1,
                       self.SHAPE_MASKS["banner_digits"], banner_on)
        # score: right-aligned, no leading zeros (measured position)
        raster = label(raster, state.score, self.SHAPE_MASKS["digits"],
                       ~banner_on)

        return self.jr.render_from_palette(raster, self.PALETTE)
