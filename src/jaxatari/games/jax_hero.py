"""
JAX implementation of H.E.R.O. (Helicopter Emergency Rescue Operation) — Activision, 1983.

Roderick Hero descends a vertical mine shaft with a helicopter backpack to
rescue a trapped miner. The world is taller than the screen: a camera follows
the player down and the renderer shows a viewport (the upper ~142 rows) into it,
with the HUD fixed below.

Fidelity to the original
------------------------
The opening chamber's geometry, the cave/HUD palette, and the HUD band layout
were measured pixel-for-pixel from the real Activision H.E.R.O. ROM (via ALE);
see scripts/hero_record_level.py for the tool that captures the full scrolling
shaft from a human descent of the ROM. The level world is authored from those
captures (_WALLS, in world coordinates).

Multiple levels
---------------
The env is multi-level: per-level geometry (walls, breakable panel, spiders,
bats, moths, lamps, miner, world height, rock/floor colour, player start) is
packed into padded, level-indexed arrays selected by ``state.level``. Rescuing a
level's miner ADVANCES to the next level — score and lives carry over,
power/dynamite refill, and the next chamber loads — while the episode ends only
when the LAST level's miner is rescued (or lives run out). Level 1 is the brown
opening descent; Level 2 is a taller green zig-zag shaft; Level 3 is a deeper
blue 5-frame shaft, each transcribed from its template image.

Enemies / weapons
-----------------
Three creature types share a parallel implementation: springing SPIDERS (patrol
a vertical line), shaft BATS (purple horizontal flyers), and mine MOTHS (pink
horizontal flyers, introduced in Level 3). All three can be killed by the laser
OR by a dynamite blast (dynamite is a valid weapon, not just a wall-breaker);
each kill scores points. Decorative LAMPS hang in the Level-3 shaft as scenery
(no collision).

Build status
------------
  * Level 1 is a TWO-CHAMBER vertical shaft the camera scrolls down (the real
    H.E.R.O. descent):
      - TOP CHAMBER (frame 1): Roderick on the ledge, the two-part blast wall
        (fixed dark rock upper part + lighter blastable panel at his height),
        and the drop pit in the ledge.
      - LOWER CHAMBER (frame 2): the springing spider and the trapped miner on
        the left shelf, reached by dropping the shaft.
    Blasting the panel scores +75 (dynamite 6->5), lasering the spider +50, and
    reaching the miner converts remaining power to points and ends the level.
  * The top chamber's palette is ROM-exact (pixel-measured); Roderick's sprite
    (blue suit + white legs + yellow rotor) and the HUD palette/colors were
    measured from the ROM. The lower chamber is transcribed from a ROM frame-2
    screenshot — its layout (ceiling band + drop-gap, hanging spider, green
    miner on the left) matches the capture, but to make it pixel-EXACT, record a
    descent with scripts/hero_record_level.py and refine the _WALLS coordinates.
  * Still approximate: the lower chamber's exact pixels and the real enemy roster
    — the single spider keeps the laser / death mechanics exercised.

Active objects / rules
----------------------
  * Roderick Hero (player): moves left/right, flies up (gravity/drift momentum),
    shoots a horizontal laser, and places dynamite.
  * Spider: a springing spider that patrols up/down a vertical line in the lower
    chamber, in Roderick's path to the miner.
  * Walls: solid blocks that block movement; one is breakable with dynamite.
  * Power drains 1 unit/frame after the first move (0 = death). Touching a
    spider or being caught in a blast costs a life; laser+50, wall+75, miner
    rescue converts remaining power to points and ends the level.

Conventions follow the other games in this package (see jax_freeway.py):
constants subclass AutoDerivedConstants; state/observation/info are
flax.struct.dataclass pytrees; the env subclasses JaxEnvironment and the
renderer subclasses JAXGameRenderer. Graphics use procedural colored sprites;
real Atari sprites can be dropped into the hero sprite dir later by swapping the
asset entries.
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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Level geometry — WORLD coordinates (y grows downward into the mine).
# ---------------------------------------------------------------------------
# Each level is a tall vertically-scrolling shaft: a camera follows Roderick
# down and the renderer shows a 142px-high window into the world. Geometry is
# authored from the per-level template images (placement of the terrain
# "equipment" and the hero/spider/miner "cast"). After rescuing a level's
# trapped miner the player advances to the next level (keeping score + lives);
# the game ends only when the LAST level's miner is rescued (or lives run out).
#
# A wall row is (x, y, w, h). Within each level the breakable wall (blastable
# with dynamite, as in the original) is at that level's break_idx; non-breakable
# walls are baked into the cave background, the breakable one is drawn
# dynamically so it can vanish. Walls are baked by colour with a width>=height
# heuristic — wide horizontal bands read as the lighter "floor" tone, tall
# vertical pieces as the darker "rock" tone.
#
# ===== LEVEL 1 (brown rock) — the real H.E.R.O. opening descent =====
#   TOP CHAMBER (world y 16..118, shown at camera_y = 0): two ceiling corner
#     blocks, a hanging central pillar whose lower part is the BLASTABLE panel,
#     and Roderick in the open pocket on the left. He blasts the panel, walks to
#     the central descent CHANNEL (x 74..92) and drops.
#   LOWER CHAMBER: two stacked middle bands (split by the channel) form its
#     ceiling; below them he lasers the springing spider (centre) and reaches the
#     trapped miner on the low left shelf, with a rock block on the lower right.
_L1_WALLS = [
    (8,    16, 152,  8),   # 0 ceiling
    (8,    16,   8, 246),  # 1 left wall  (y16..262)
    (152,  16,   8, 246),  # 2 right wall (y16..262)
    (16,   16,  14,  28),  # 3 top-LEFT corner block
    (124,  16,  28,  44),  # 4 top-RIGHT corner block
    (60,   16,  14,  74),  # 5 hanging pillar: FIXED dark rock (x60..74, y16..90)
    (60,   90,  14,  28),  # 6 pillar foot: BLASTABLE panel (y90..118) [breakable]
    (8,   118,  66,  14),  # 7  upper band LEFT  (x8..74), channel x74..92
    (92,  118,  60,  14),  # 8  upper band RIGHT (x92..152)
    (8,   146,  66,  22),  # 9  lower band LEFT
    (92,  146,  60,  22),  # 10 lower band RIGHT
    (8,   220,  44,   6),  # 11 miner shelf (low, left)
    (124, 180,  28,  58),  # 12 lower-RIGHT rock block
    (8,   238, 152,  24),  # 13 bottom floor (full width, y238..262)
]
_L1_BREAKABLE = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
_L1_BREAKABLE_IDX = 6
_L1_WORLD_HEIGHT = 262
_L1_PLAYER_START = (22, 52)
_L1_MINER = (24, 208)
_L1_ROCK_COLOR = (144, 72, 17)    # dark rock brown
_L1_FLOOR_COLOR = (180, 122, 48)  # lighter floor brown
# (x, y_min, y_max, start_y, start_dir) per springing spider (vertical patrol).
_L1_SPIDERS = [
    (80, 176, 214, 195, 1),
]
# (x_min, x_max, y, start_x, start_dir) per shaft bat (horizontal flyer).
_L1_BATS = []
# (x_min, x_max, y, start_x, start_dir) per mine moth (horizontal flyer).
_L1_MOTHS = []
# (x, y) per decorative lamp hanging in the shaft (scenery, no collision).
_L1_LAMPS = []

# ===== LEVEL 2 (green rock) — taller zig-zag shaft, transcribed from the
# Level-2 template image =====
#   The descent fans down a longer green shaft, screen ("frame") by screen:
#     FRAME 1 (top chamber): the pillar + BLASTABLE foot the hero blows open.
#     FRAME 2: a single springing spider in the first gap.
#     FRAME 3: a shaft bat (higher) then a springing spider (lower) — met one by
#              one as Roderick drops the centre gap.
#     LAST FRAME (lower chamber): a shaft bat, and the trapped miner tucked in
#              the BOTTOM-RIGHT corner under a green rock block.
_L2_WALLS = [
    (8,    16, 152,  8),   # 0 ceiling
    (8,    16,   8, 384),  # 1 left wall  (y16..400)
    (152,  16,   8, 384),  # 2 right wall (y16..400)
    (16,   16,  14,  28),  # 3 top-LEFT corner block
    (124,  16,  28,  44),  # 4 top-RIGHT corner block
    (60,   16,  14,  74),  # 5 hanging pillar: FIXED dark rock (x60..74, y16..90)
    (60,   90,  14,  28),  # 6 pillar foot: BLASTABLE panel (y90..118) [breakable]
    (8,   118,  66,  14),  # 7  upper band LEFT  (x8..74), channel x74..92
    (92,  118,  60,  14),  # 8  upper band RIGHT (x92..152)
    # --- green zig-zag descent (gaps alternate right / left / centre) ---
    (8,   168,  96,  14),  # 9  band A LEFT  (gap RIGHT x104..152)
    (60,  216,  92,  14),  # 10 band B RIGHT (gap LEFT  x8..60)
    (8,   264,  52,  14),  # 11 band C LEFT
    (100, 264,  52,  14),  # 12 band C RIGHT (gap CENTRE x60..100)
    (8,   312, 100,  14),  # 13 band D LEFT  (gap RIGHT x108..152)
    # --- lower chamber ---
    (124, 330,  28,  20),  # 14 lower-RIGHT rock block (miner sits below it)
    (8,   360,  44,   6),  # 15 left ledge
    (8,   378, 152,  22),  # 16 bottom floor (full width, y378..400)
]
_L2_BREAKABLE = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
_L2_BREAKABLE_IDX = 6
_L2_WORLD_HEIGHT = 400
_L2_PLAYER_START = (22, 52)
_L2_MINER = (138, 366)            # bottom-RIGHT corner, on the floor below the block
_L2_ROCK_COLOR = (46, 110, 38)    # dark cave green
_L2_FLOOR_COLOR = (86, 150, 58)   # lighter ledge green
# Two springing spiders (vertical patrol).
_L2_SPIDERS = [
    (120, 186, 220, 203, 1),   # FRAME 2: band A's right gap
    (80,  286, 320, 303, 1),   # FRAME 3: band C's centre gap (lower of the pair)
]
# Two shaft bats (horizontal flyers): (x_min, x_max, y, start_x, start_dir).
_L2_BATS = [
    (20, 140, 240, 40,  1),    # FRAME 3: flies above the centre-gap spider
    (16, 104, 348, 30,  1),    # LAST FRAME: patrols the lower chamber
]
_L2_MOTHS = []
_L2_LAMPS = []

# ===== LEVEL 3 (blue rock) — a deeper 5-frame shaft, transcribed from the
# Level-3 template image. The descent reads frame by frame:
#   FRAME 1 (top chamber): Roderick blasts the pillar foot open; a lamp hangs in
#            the shaft midway down to frame 2.
#   FRAME 2: TWO springing spiders (in the two gaps) and ONE mine moth fluttering
#            across; a second lamp hangs midway down to frame 3.
#   FRAME 3: ONE shaft bat; a springing spider waits midway down to frame 4.
#   FRAME 4: ONE mine moth.
#   FRAME 5 (lower chamber): another mine moth, and the trapped miner seated on
#            the LEFT-MOST bottom shelf.
_L3_WALLS = [
    (8,    16, 152,  8),   # 0 ceiling
    (8,    16,   8, 524),  # 1 left wall  (y16..540)
    (152,  16,   8, 524),  # 2 right wall (y16..540)
    (16,   16,  14,  28),  # 3 top-LEFT corner block
    (124,  16,  28,  44),  # 4 top-RIGHT corner block
    (60,   16,  14,  74),  # 5 hanging pillar: FIXED dark rock (x60..74, y16..90)
    (60,   90,  14,  28),  # 6 pillar foot: BLASTABLE panel (y90..118) [breakable]
    (8,   118,  66,  14),  # 7  upper band LEFT  (x8..74), channel x74..92
    (92,  118,  60,  14),  # 8  upper band RIGHT (x92..152)
    # --- FRAME 2 zig-zag ---
    (8,   168,  96,  14),  # 9  band, gap RIGHT  x104..152
    (60,  216,  92,  14),  # 10 band, gap LEFT   x8..60
    # --- FRAME 3 zig-zag ---
    (8,   264,  52,  14),  # 11 band C LEFT
    (100, 264,  52,  14),  # 12 band C RIGHT (gap CENTRE x60..100)
    (8,   312, 100,  14),  # 13 band, gap RIGHT  x108..152
    # --- FRAME 4 zig-zag ---
    (60,  360,  92,  14),  # 14 band, gap LEFT   x8..60
    (8,   408,  96,  14),  # 15 band, gap RIGHT  x104..152
    # --- FRAME 5 lower chamber ---
    (124, 456,  28,  44),  # 16 lower-RIGHT rock block
    (8,   500,  44,   6),  # 17 miner shelf (low, LEFT-most)
    (8,   518, 152,  22),  # 18 bottom floor (full width, y518..540)
]
_L3_BREAKABLE = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
_L3_BREAKABLE_IDX = 6
_L3_WORLD_HEIGHT = 540
_L3_PLAYER_START = (22, 52)
_L3_MINER = (16, 488)             # LEFT-most bottom shelf (feet rest on y500)
_L3_ROCK_COLOR = (36, 44, 120)    # dark cave navy
_L3_FLOOR_COLOR = (72, 92, 190)   # lighter ledge blue
# Three springing spiders (vertical patrol): two in FRAME 2's gaps, one waiting
# midway between FRAME 3 and FRAME 4.
_L3_SPIDERS = [
    (120, 176, 210, 193, 1),   # FRAME 2: right gap
    (30,  224, 258, 240, 1),   # FRAME 2: left gap
    (120, 322, 352, 337, 1),   # midway FRAME 3 -> 4
]
# One shaft bat (FRAME 3), horizontal flyer: (x_min, x_max, y, start_x, start_dir).
_L3_BATS = [
    (20, 140, 288, 40, 1),
]
# Three mine moths (horizontal flyers): FRAME 2, FRAME 4, FRAME 5.
_L3_MOTHS = [
    (20, 140, 196, 40, 1),     # FRAME 2: flutters across, with the two spiders
    (16, 140, 384, 30, 1),     # FRAME 4
    (16, 104, 478, 24, 1),     # FRAME 5: lower chamber, near the miner
]
# Two decorative lamps (scenery, no collision): midway 1->2 and midway 2->3.
_L3_LAMPS = [
    (112, 150),                # midway FRAME 1 -> 2
    (40,  248),                # midway FRAME 2 -> 3
]

# ---------------------------------------------------------------------------
# Pack the levels into fixed-shape (padded) arrays so the jitted env/renderer
# can index everything by state.level. Unused wall / spider slots are marked
# invalid (and start inactive) so they never collide or draw.
# ---------------------------------------------------------------------------
_LEVELS = [
    dict(walls=_L1_WALLS, breakable=_L1_BREAKABLE, breakable_idx=_L1_BREAKABLE_IDX,
         world_height=_L1_WORLD_HEIGHT, player_start=_L1_PLAYER_START, miner=_L1_MINER,
         rock_color=_L1_ROCK_COLOR, floor_color=_L1_FLOOR_COLOR,
         spiders=_L1_SPIDERS, bats=_L1_BATS, moths=_L1_MOTHS, lamps=_L1_LAMPS),
    dict(walls=_L2_WALLS, breakable=_L2_BREAKABLE, breakable_idx=_L2_BREAKABLE_IDX,
         world_height=_L2_WORLD_HEIGHT, player_start=_L2_PLAYER_START, miner=_L2_MINER,
         rock_color=_L2_ROCK_COLOR, floor_color=_L2_FLOOR_COLOR,
         spiders=_L2_SPIDERS, bats=_L2_BATS, moths=_L2_MOTHS, lamps=_L2_LAMPS),
    dict(walls=_L3_WALLS, breakable=_L3_BREAKABLE, breakable_idx=_L3_BREAKABLE_IDX,
         world_height=_L3_WORLD_HEIGHT, player_start=_L3_PLAYER_START, miner=_L3_MINER,
         rock_color=_L3_ROCK_COLOR, floor_color=_L3_FLOOR_COLOR,
         spiders=_L3_SPIDERS, bats=_L3_BATS, moths=_L3_MOTHS, lamps=_L3_LAMPS),
]
_NUM_LEVELS = len(_LEVELS)
_MAX_WALLS = max(len(l["walls"]) for l in _LEVELS)
_MAX_SPIDERS = max(len(l["spiders"]) for l in _LEVELS)
_MAX_BATS = max(max(len(l["bats"]) for l in _LEVELS), 1)  # >=1 to keep arrays non-empty
_MAX_MOTHS = max(max(len(l["moths"]) for l in _LEVELS), 1)
_MAX_LAMPS = max(max(len(l["lamps"]) for l in _LEVELS), 1)
_MAX_WORLD_HEIGHT = max(l["world_height"] for l in _LEVELS)


def _build_level_arrays():
    """Build the padded, level-indexed numpy arrays from _LEVELS."""
    nL, nW, nS = _NUM_LEVELS, _MAX_WALLS, _MAX_SPIDERS
    walls = np.zeros((nL, nW, 4), np.int32)
    wall_valid = np.zeros((nL, nW), bool)
    wall_break = np.zeros((nL, nW), bool)
    break_idx = np.zeros((nL,), np.int32)
    world_h = np.zeros((nL,), np.int32)
    p_start = np.zeros((nL, 2), np.int32)
    miner = np.zeros((nL, 2), np.int32)
    rock_col = np.zeros((nL, 3), np.int32)
    floor_col = np.zeros((nL, 3), np.int32)
    sx = np.zeros((nL, nS), np.int32)
    symin = np.zeros((nL, nS), np.int32)
    symax = np.zeros((nL, nS), np.int32)
    sstart = np.zeros((nL, nS), np.int32)
    sdir = np.ones((nL, nS), np.int32)
    svalid = np.zeros((nL, nS), bool)
    nB = _MAX_BATS
    bxmin = np.zeros((nL, nB), np.int32)
    bxmax = np.zeros((nL, nB), np.int32)
    by = np.zeros((nL, nB), np.int32)
    bstart = np.zeros((nL, nB), np.int32)
    bdir = np.ones((nL, nB), np.int32)
    bvalid = np.zeros((nL, nB), bool)
    nM = _MAX_MOTHS
    mxmin = np.zeros((nL, nM), np.int32)
    mxmax = np.zeros((nL, nM), np.int32)
    my = np.zeros((nL, nM), np.int32)
    mstart = np.zeros((nL, nM), np.int32)
    mdir = np.ones((nL, nM), np.int32)
    mvalid = np.zeros((nL, nM), bool)
    nLa = _MAX_LAMPS
    lampx = np.zeros((nL, nLa), np.int32)
    lampy = np.zeros((nL, nLa), np.int32)
    lampvalid = np.zeros((nL, nLa), bool)
    for li, lv in enumerate(_LEVELS):
        for wi, w in enumerate(lv["walls"]):
            walls[li, wi] = w
            wall_valid[li, wi] = True
            wall_break[li, wi] = bool(lv["breakable"][wi])
        break_idx[li] = lv["breakable_idx"]
        world_h[li] = lv["world_height"]
        p_start[li] = lv["player_start"]
        miner[li] = lv["miner"]
        rock_col[li] = lv["rock_color"]
        floor_col[li] = lv["floor_color"]
        for si, sp in enumerate(lv["spiders"]):
            sx[li, si], symin[li, si], symax[li, si], sstart[li, si], sdir[li, si] = sp
            svalid[li, si] = True
        for bi, bt in enumerate(lv["bats"]):
            bxmin[li, bi], bxmax[li, bi], by[li, bi], bstart[li, bi], bdir[li, bi] = bt
            bvalid[li, bi] = True
        for mi, mt in enumerate(lv["moths"]):
            mxmin[li, mi], mxmax[li, mi], my[li, mi], mstart[li, mi], mdir[li, mi] = mt
            mvalid[li, mi] = True
        for lai, la in enumerate(lv["lamps"]):
            lampx[li, lai], lampy[li, lai] = la
            lampvalid[li, lai] = True
    return dict(walls=walls, wall_valid=wall_valid, wall_break=wall_break,
                break_idx=break_idx, world_h=world_h, p_start=p_start, miner=miner,
                rock_col=rock_col, floor_col=floor_col, sx=sx, symin=symin,
                symax=symax, sstart=sstart, sdir=sdir, svalid=svalid,
                bxmin=bxmin, bxmax=bxmax, by=by, bstart=bstart, bdir=bdir, bvalid=bvalid,
                mxmin=mxmin, mxmax=mxmax, my=my, mstart=mstart, mdir=mdir, mvalid=mvalid,
                lampx=lampx, lampy=lampy, lampvalid=lampvalid)


_LV = _build_level_arrays()


class HeroConstants(AutoDerivedConstants):
    # --- Screen ---
    screen_width: int = struct.field(pytree_node=False, default=160)
    screen_height: int = struct.field(pytree_node=False, default=210)

    # Cave viewport: rows [0, cave_bottom) show a window into the world; the
    # HUD occupies the rows below. The cave fills the screen top in the real
    # game (no banner), so play_top = 0.
    play_top: int = struct.field(pytree_node=False, default=16)
    cave_bottom: int = struct.field(pytree_node=False, default=142)
    play_left: int = struct.field(pytree_node=False, default=8)
    play_right: int = struct.field(pytree_node=False, default=160)
    # Tallest level world (the cave raster is baked this tall for every level;
    # shorter levels are black-padded below their own floor). Per-level heights
    # live in LEVEL_WORLD_HEIGHT and drive the camera clamp / vertical bounds.
    num_levels: int = struct.field(pytree_node=False, default=_NUM_LEVELS)
    world_height: int = struct.field(pytree_node=False, default=_MAX_WORLD_HEIGHT)
    # The player is kept roughly this many px below the top of the viewport;
    # the camera scrolls the world to maintain it (clamped to world bounds).
    # Set to the top-chamber resting depth so the camera stays at 0 while
    # Roderick is up top and only scrolls once he drops down the shaft.
    camera_anchor: int = struct.field(pytree_node=False, default=105)

    # --- Player ---
    player_width: int = struct.field(pytree_node=False, default=6)
    player_height: int = struct.field(pytree_node=False, default=13)
    player_start_x: int = struct.field(pytree_node=False, default=22)
    # Roderick starts in the open pocket on the upper LEFT (as in the template),
    # below the corner block; gravity settles him onto the upper band (y118),
    # left of the blastable pillar foot.
    player_start_y: int = struct.field(pytree_node=False, default=52)

    # Title banner ("- LEVEL 1 -") band height at the very top of the screen.
    banner_height: int = struct.field(pytree_node=False, default=16)

    # --- Flight physics (prop thrust vs. gravity) ---
    gravity: float = struct.field(pytree_node=False, default=0.20)
    thrust: float = struct.field(pytree_node=False, default=-0.45)
    max_fall_speed: float = struct.field(pytree_node=False, default=2.5)
    max_rise_speed: float = struct.field(pytree_node=False, default=-2.5)
    move_speed: int = struct.field(pytree_node=False, default=1)

    # --- Laser ---
    laser_length: int = struct.field(pytree_node=False, default=24)
    laser_height: int = struct.field(pytree_node=False, default=2)
    laser_duration: int = struct.field(pytree_node=False, default=6)

    # --- Dynamite ---
    dyn_width: int = struct.field(pytree_node=False, default=4)
    dyn_height: int = struct.field(pytree_node=False, default=8)
    dyn_fuse: int = struct.field(pytree_node=False, default=60)
    explosion_frames: int = struct.field(pytree_node=False, default=8)
    explosion_radius: int = struct.field(pytree_node=False, default=14)
    starting_dynamite: int = struct.field(pytree_node=False, default=6)

    # --- Power / lives / scoring ---
    max_power: int = struct.field(pytree_node=False, default=4000)
    power_drain_per_frame: int = struct.field(pytree_node=False, default=1)
    starting_lives: int = struct.field(pytree_node=False, default=3)
    spider_points: int = struct.field(pytree_node=False, default=50)
    wall_points: int = struct.field(pytree_node=False, default=75)

    # --- Spiders (WORLD coords) ---
    # Springing spiders hang in the descent gaps and patrol up/down a vertical
    # line, so Roderick must laser them (+50) on his way to the miner. The state
    # carries num_spiders (= the most any level uses) slots; per level, unused
    # slots are marked invalid (start dead) so they never collide or draw.
    # SPIDER_* below are the LEVEL-1 reference arrays (full padded row).
    num_spiders: int = struct.field(pytree_node=False, default=_MAX_SPIDERS)
    spider_width: int = struct.field(pytree_node=False, default=7)
    spider_height: int = struct.field(pytree_node=False, default=7)
    spider_move_period: int = struct.field(pytree_node=False, default=3)
    SPIDER_X: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["sx"][0], dtype=jnp.int32))
    SPIDER_Y_MIN: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["symin"][0], dtype=jnp.int32))
    SPIDER_Y_MAX: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["symax"][0], dtype=jnp.int32))
    SPIDER_START_Y: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["sstart"][0], dtype=jnp.int32))
    SPIDER_START_DIR: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["sdir"][0], dtype=jnp.int32))

    # --- Miner (WORLD coords) — LEVEL-1 reference position ---
    miner_x: int = struct.field(pytree_node=False, default=_L1_MINER[0])
    miner_y: int = struct.field(pytree_node=False, default=_L1_MINER[1])
    miner_width: int = struct.field(pytree_node=False, default=8)
    miner_height: int = struct.field(pytree_node=False, default=12)

    # --- Walls --- (num_walls = padded max; breakable_idx/WALLS are LEVEL-1) ---
    num_walls: int = struct.field(pytree_node=False, default=_MAX_WALLS)
    breakable_idx: int = struct.field(pytree_node=False, default=_L1_BREAKABLE_IDX)
    WALLS: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["walls"][0], dtype=jnp.int32))
    WALL_BREAKABLE: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["wall_break"][0], dtype=jnp.bool_))

    # --- Level-indexed (padded) geometry: index by state.level ---
    LEVEL_WALLS: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["walls"], dtype=jnp.int32))
    LEVEL_WALL_VALID: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["wall_valid"], dtype=jnp.bool_))
    LEVEL_WALL_BREAK: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["wall_break"], dtype=jnp.bool_))
    LEVEL_BREAK_IDX: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["break_idx"], dtype=jnp.int32))
    LEVEL_WORLD_HEIGHT: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["world_h"], dtype=jnp.int32))
    LEVEL_PLAYER_START: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["p_start"], dtype=jnp.int32))
    LEVEL_MINER: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["miner"], dtype=jnp.int32))
    LEVEL_SPIDER_X: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["sx"], dtype=jnp.int32))
    LEVEL_SPIDER_Y_MIN: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["symin"], dtype=jnp.int32))
    LEVEL_SPIDER_Y_MAX: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["symax"], dtype=jnp.int32))
    LEVEL_SPIDER_START_Y: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["sstart"], dtype=jnp.int32))
    LEVEL_SPIDER_START_DIR: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["sdir"], dtype=jnp.int32))
    LEVEL_SPIDER_VALID: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["svalid"], dtype=jnp.bool_))

    # --- Shaft bats (WORLD coords) --- horizontal flyers that patrol a band ---
    # Bats fly side to side on a fixed y-line; laser them (+points) or be killed
    # on contact. num_bats slots are carried by the state; per level unused slots
    # are invalid (start dead). Level 1 has none; Level 2 has two.
    num_bats: int = struct.field(pytree_node=False, default=_MAX_BATS)
    bat_width: int = struct.field(pytree_node=False, default=9)
    bat_height: int = struct.field(pytree_node=False, default=5)
    bat_move_period: int = struct.field(pytree_node=False, default=2)
    bat_points: int = struct.field(pytree_node=False, default=50)
    LEVEL_BAT_X_MIN: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["bxmin"], dtype=jnp.int32))
    LEVEL_BAT_X_MAX: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["bxmax"], dtype=jnp.int32))
    LEVEL_BAT_Y: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["by"], dtype=jnp.int32))
    LEVEL_BAT_START_X: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["bstart"], dtype=jnp.int32))
    LEVEL_BAT_START_DIR: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["bdir"], dtype=jnp.int32))
    LEVEL_BAT_VALID: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["bvalid"], dtype=jnp.bool_))

    # --- Mine moths (WORLD coords) --- horizontal flyers, parallel to bats ---
    # A distinct enemy type introduced in Level 3: orange moths that flutter side
    # to side on a fixed y-line. Laser or DYNAMITE BLAST kills them (+points); they
    # kill Roderick on contact. Levels 1-2 have none (valid mask all False).
    num_moths: int = struct.field(pytree_node=False, default=_MAX_MOTHS)
    moth_width: int = struct.field(pytree_node=False, default=9)
    moth_height: int = struct.field(pytree_node=False, default=5)
    moth_move_period: int = struct.field(pytree_node=False, default=2)
    moth_points: int = struct.field(pytree_node=False, default=50)
    LEVEL_MOTH_X_MIN: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["mxmin"], dtype=jnp.int32))
    LEVEL_MOTH_X_MAX: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["mxmax"], dtype=jnp.int32))
    LEVEL_MOTH_Y: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["my"], dtype=jnp.int32))
    LEVEL_MOTH_START_X: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["mstart"], dtype=jnp.int32))
    LEVEL_MOTH_START_DIR: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["mdir"], dtype=jnp.int32))
    LEVEL_MOTH_VALID: jnp.ndarray = struct.field(pytree_node=False,
        default_factory=lambda: jnp.array(_LV["mvalid"], dtype=jnp.bool_))

    # --- Lamps (WORLD coords) --- decorative lanterns hanging in the shaft ---
    # Scenery only (no collision, no state): a visual landmark partway down the
    # shaft, as on the Level-3 map. Per the design guide, lamps are completely
    # static, so they are BAKED into each level's cave background by
    # _build_cave (from the _LV lamp arrays) rather than carried as render-time
    # constants or drawn each frame.

    # --- HUD layout (measured from the ROM; HUD band is rows 142-188) ---
    hud_top: int = struct.field(pytree_node=False, default=142)
    hud_bottom: int = struct.field(pytree_node=False, default=189)
    power_bar_x: int = struct.field(pytree_node=False, default=49)
    power_bar_y: int = struct.field(pytree_node=False, default=146)
    power_bar_width: int = struct.field(pytree_node=False, default=74)
    power_bar_height: int = struct.field(pytree_node=False, default=4)
    # Row of 6 little-Hero life figures (yellow over red) at rows 166-176.
    lives_x: int = struct.field(pytree_node=False, default=58)
    lives_y: int = struct.field(pytree_node=False, default=166)

    # --- Colors (RGB) — exact values sampled from the H.E.R.O. ROM ---
    bg_color: Tuple[int, int, int] = struct.field(pytree_node=False, default=(0, 0, 0))
    wall_color: Tuple[int, int, int] = struct.field(pytree_node=False, default=(144, 72, 17))
    floor_color: Tuple[int, int, int] = struct.field(pytree_node=False, default=(180, 122, 48))
    breakable_color: Tuple[int, int, int] = struct.field(pytree_node=False, default=(180, 122, 48))
    hud_color: Tuple[int, int, int] = struct.field(pytree_node=False, default=(111, 111, 111))
    player_color: Tuple[int, int, int] = struct.field(pytree_node=False, default=(45, 50, 184))
    laser_color: Tuple[int, int, int] = struct.field(pytree_node=False, default=(232, 232, 74))
    spider_color: Tuple[int, int, int] = struct.field(pytree_node=False, default=(214, 214, 214))
    bat_color: Tuple[int, int, int] = struct.field(pytree_node=False, default=(170, 120, 200))
    moth_color: Tuple[int, int, int] = struct.field(pytree_node=False, default=(236, 150, 200))
    miner_color: Tuple[int, int, int] = struct.field(pytree_node=False, default=(110, 156, 66))
    dyn_color: Tuple[int, int, int] = struct.field(pytree_node=False, default=(184, 50, 50))
    explosion_color: Tuple[int, int, int] = struct.field(pytree_node=False, default=(252, 160, 40))
    power_color: Tuple[int, int, int] = struct.field(pytree_node=False, default=(167, 26, 26))
    lamp_color: Tuple[int, int, int] = struct.field(pytree_node=False, default=(232, 232, 74))
    text_color: Tuple[int, int, int] = struct.field(pytree_node=False, default=(214, 214, 214))

    def compute_derived(self):
        return {}


# ---------------------------------------------------------------------------
# State / Observation / Info
# ---------------------------------------------------------------------------
@struct.dataclass
class HeroState:
    player_x: chex.Array
    player_y: chex.Array          # WORLD y (grows downward into the mine)
    player_vy: chex.Array
    camera_y: chex.Array          # top world-row currently shown in the viewport
    facing: chex.Array            # -1 left, +1 right
    walk_timer: chex.Array        # frames spent walking horizontally (0 = idle)
    has_moved: chex.Array         # power only drains after first move
    laser_timer: chex.Array
    power: chex.Array
    lives: chex.Array
    score: chex.Array
    level: chex.Array
    dynamite_count: chex.Array
    dyn_active: chex.Array
    dyn_x: chex.Array
    dyn_y: chex.Array
    dyn_fuse: chex.Array
    explosion_timer: chex.Array
    spider_y: chex.Array          # (num_spiders,)
    spider_dir: chex.Array        # (num_spiders,)
    spider_alive: chex.Array      # (num_spiders,)
    bat_x: chex.Array             # (num_bats,)  horizontal flyer position
    bat_dir: chex.Array           # (num_bats,)
    bat_alive: chex.Array         # (num_bats,)
    moth_x: chex.Array            # (num_moths,) horizontal flyer position
    moth_dir: chex.Array          # (num_moths,)
    moth_alive: chex.Array        # (num_moths,)
    wall_active: chex.Array       # (num_walls,)
    miner_rescued: chex.Array
    level_complete: chex.Array
    step_counter: chex.Array
    game_over: chex.Array
    rng_key: chex.PRNGKey


@struct.dataclass
class HeroObservation:
    player: ObjectObservation
    laser: ObjectObservation
    spiders: ObjectObservation
    bats: ObjectObservation
    moths: ObjectObservation
    miner: ObjectObservation
    dynamite: ObjectObservation
    walls: ObjectObservation
    power: chex.Array
    lives: chex.Array
    score: chex.Array
    dynamite_count: chex.Array
    level: chex.Array


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
    # DOWNFIRE places dynamite; the other *FIRE variants shoot the laser.
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
            player_x=c.LEVEL_PLAYER_START[0, 0].astype(jnp.int32),
            player_y=c.LEVEL_PLAYER_START[0, 1].astype(jnp.int32),
            player_vy=jnp.array(0.0, dtype=jnp.float32),
            camera_y=jnp.array(0, dtype=jnp.int32),
            facing=jnp.array(1, dtype=jnp.int32),
            walk_timer=jnp.array(0, dtype=jnp.int32),
            has_moved=jnp.array(False, dtype=jnp.bool_),
            laser_timer=jnp.array(0, dtype=jnp.int32),
            power=jnp.array(c.max_power, dtype=jnp.int32),
            lives=jnp.array(c.starting_lives, dtype=jnp.int32),
            score=jnp.array(0, dtype=jnp.int32),
            level=jnp.array(0, dtype=jnp.int32),
            dynamite_count=jnp.array(c.starting_dynamite, dtype=jnp.int32),
            dyn_active=jnp.array(False, dtype=jnp.bool_),
            dyn_x=jnp.array(0, dtype=jnp.int32),
            dyn_y=jnp.array(0, dtype=jnp.int32),
            dyn_fuse=jnp.array(0, dtype=jnp.int32),
            explosion_timer=jnp.array(0, dtype=jnp.int32),
            spider_y=c.LEVEL_SPIDER_START_Y[0].astype(jnp.int32),
            spider_dir=c.LEVEL_SPIDER_START_DIR[0].astype(jnp.int32),
            spider_alive=c.LEVEL_SPIDER_VALID[0],
            bat_x=c.LEVEL_BAT_START_X[0].astype(jnp.int32),
            bat_dir=c.LEVEL_BAT_START_DIR[0].astype(jnp.int32),
            bat_alive=c.LEVEL_BAT_VALID[0],
            moth_x=c.LEVEL_MOTH_START_X[0].astype(jnp.int32),
            moth_dir=c.LEVEL_MOTH_START_DIR[0].astype(jnp.int32),
            moth_alive=c.LEVEL_MOTH_VALID[0],
            wall_active=c.LEVEL_WALL_VALID[0],
            miner_rescued=jnp.array(False, dtype=jnp.bool_),
            level_complete=jnp.array(False, dtype=jnp.bool_),
            step_counter=jnp.array(0, dtype=jnp.int32),
            game_over=jnp.array(False, dtype=jnp.bool_),
            rng_key=key,
        )
        return self._get_observation(state), state

    # --- collision helpers ------------------------------------------------
    @staticmethod
    def _aabb(ax, ay, aw, ah, bx, by, bw, bh):
        """Axis-aligned bounding-box overlap (broadcasts over arrays)."""
        return ((ax < bx + bw) & (ax + aw > bx) &
                (ay < by + bh) & (ay + ah > by))

    def _hits_wall(self, px, py, walls, wall_active):
        """True if a player-sized box at (px, py) overlaps any active wall of the
        current level. `walls` is that level's (num_walls, 4) array; inactive /
        unused slots are excluded via `wall_active`."""
        c = self.consts
        overlap = self._aabb(px, py, c.player_width, c.player_height,
                             walls[:, 0], walls[:, 1], walls[:, 2], walls[:, 3]) & wall_active
        return jnp.any(overlap)

    # --- step -------------------------------------------------------------
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: HeroState, action: int) -> Tuple[HeroObservation, HeroState, float, bool, HeroInfo]:
        c = self.consts
        atari_action = jnp.take(self.ACTION_SET, action)

        # --- gather the CURRENT level's geometry (indexed by state.level) ---
        lvl = state.level
        L_walls = c.LEVEL_WALLS[lvl]                 # (num_walls, 4)
        L_break = c.LEVEL_WALL_BREAK[lvl]            # (num_walls,)
        L_break_idx = c.LEVEL_BREAK_IDX[lvl]
        L_world_h = c.LEVEL_WORLD_HEIGHT[lvl]
        L_spider_x = c.LEVEL_SPIDER_X[lvl]           # (num_spiders,)
        L_spider_ymin = c.LEVEL_SPIDER_Y_MIN[lvl]
        L_spider_ymax = c.LEVEL_SPIDER_Y_MAX[lvl]
        L_bat_xmin = c.LEVEL_BAT_X_MIN[lvl]          # (num_bats,)
        L_bat_xmax = c.LEVEL_BAT_X_MAX[lvl]
        L_bat_y = c.LEVEL_BAT_Y[lvl]
        L_moth_xmin = c.LEVEL_MOTH_X_MIN[lvl]        # (num_moths,)
        L_moth_xmax = c.LEVEL_MOTH_X_MAX[lvl]
        L_moth_y = c.LEVEL_MOTH_Y[lvl]
        L_miner_x = c.LEVEL_MINER[lvl, 0]
        L_miner_y = c.LEVEL_MINER[lvl, 1]

        up = jnp.isin(atari_action, jnp.array(
            [Action.UP, Action.UPRIGHT, Action.UPLEFT, Action.UPFIRE], dtype=jnp.int32))
        left = jnp.isin(atari_action, jnp.array(
            [Action.LEFT, Action.UPLEFT, Action.LEFTFIRE], dtype=jnp.int32))
        right = jnp.isin(atari_action, jnp.array(
            [Action.RIGHT, Action.UPRIGHT, Action.RIGHTFIRE], dtype=jnp.int32))
        down = jnp.isin(atari_action, jnp.array(
            [Action.DOWN, Action.DOWNFIRE], dtype=jnp.int32))
        laser_fire = jnp.isin(atari_action, jnp.array(
            [Action.FIRE, Action.UPFIRE, Action.LEFTFIRE, Action.RIGHTFIRE], dtype=jnp.int32))
        # Dynamite is laid ONLY with DOWN + FIRE (down arrow + space together);
        # pressing the down arrow on its own does not drop a charge.
        dyn_fire = atari_action == Action.DOWNFIRE

        # --- place dynamite (uses current player position) ---
        can_place = dyn_fire & (state.dynamite_count > 0) & (~state.dyn_active) & (state.explosion_timer <= 0)
        dyn_active = jnp.where(can_place, True, state.dyn_active)
        dyn_x = jnp.where(can_place, state.player_x + c.player_width // 2, state.dyn_x).astype(jnp.int32)
        dyn_y = jnp.where(can_place, state.player_y + c.player_height - c.dyn_height, state.dyn_y).astype(jnp.int32)
        dyn_fuse = jnp.where(can_place, c.dyn_fuse, state.dyn_fuse).astype(jnp.int32)
        dynamite_count = jnp.where(can_place, state.dynamite_count - 1, state.dynamite_count).astype(jnp.int32)

        # --- horizontal movement with wall collision ---
        dx = (right.astype(jnp.int32) - left.astype(jnp.int32)) * c.move_speed
        cand_x = jnp.clip(state.player_x + dx, c.play_left, c.play_right - c.player_width).astype(jnp.int32)
        x_blocked = self._hits_wall(cand_x, state.player_y, L_walls, state.wall_active)
        new_x = jnp.where(x_blocked, state.player_x, cand_x).astype(jnp.int32)
        new_facing = jnp.where(right, 1, jnp.where(left, -1, state.facing)).astype(jnp.int32)

        # --- vertical flight with wall collision ---
        # Up thrusts the prop; otherwise gravity pulls Roderick down. (DOWN is the
        # dynamite key, not a fast-descend, so it does not add downward accel.)
        accel = jnp.where(up, c.thrust, c.gravity)
        new_vy = jnp.clip(state.player_vy + accel, c.max_rise_speed, c.max_fall_speed).astype(jnp.float32)
        dy = jnp.round(new_vy).astype(jnp.int32)
        cand_y = jnp.clip(state.player_y + dy, c.play_top, L_world_h - c.player_height).astype(jnp.int32)
        y_blocked = self._hits_wall(new_x, cand_y, L_walls, state.wall_active) | (cand_y != state.player_y + dy)
        new_y = jnp.where(y_blocked, state.player_y, cand_y).astype(jnp.int32)
        new_vy = jnp.where(y_blocked, 0.0, new_vy).astype(jnp.float32)

        moved_input = up | left | right | down
        has_moved = state.has_moved | moved_input

        # Walk animation timer: advances while actually moving horizontally.
        moved_h = new_x != state.player_x
        walk_timer = jnp.where(moved_h, state.walk_timer + 1, 0).astype(jnp.int32)

        # --- dynamite fuse / explosion ---
        new_fuse = jnp.where(dyn_active, dyn_fuse - 1, dyn_fuse).astype(jnp.int32)
        explode_now = dyn_active & (new_fuse <= 0)
        explosion_timer = jnp.where(explode_now, c.explosion_frames,
                                    jnp.maximum(0, state.explosion_timer - 1)).astype(jnp.int32)
        dyn_active = jnp.where(explode_now, False, dyn_active)

        # Explosion box (centered on the dynamite).
        r = c.explosion_radius
        ex, ey, ew, eh = dyn_x - r, dyn_y - r, 2 * r, 2 * r

        # Breakable wall destroyed by blast.
        wall_hit = (explode_now &
                    self._aabb(ex, ey, ew, eh,
                               L_walls[:, 0], L_walls[:, 1], L_walls[:, 2], L_walls[:, 3]) &
                    L_break & state.wall_active)
        wall_active = state.wall_active & (~wall_hit)
        walls_broken = jnp.sum(wall_hit.astype(jnp.int32))

        # Player caught in the blast.
        died_blast = explode_now & self._aabb(
            new_x, new_y, c.player_width, c.player_height, ex, ey, ew, eh)

        # --- spiders move up/down on their vertical line ---
        move = (state.step_counter % c.spider_move_period) == 0
        sy = state.spider_y + jnp.where(move, state.spider_dir, 0)
        hit_top = sy <= L_spider_ymin
        hit_bot = sy >= L_spider_ymax
        new_spider_dir = jnp.where(hit_top, 1, jnp.where(hit_bot, -1, state.spider_dir)).astype(jnp.int32)
        new_spider_y = jnp.clip(sy, L_spider_ymin, L_spider_ymax).astype(jnp.int32)

        # --- bats fly left/right along their horizontal line ---
        bmove = (state.step_counter % c.bat_move_period) == 0
        bx = state.bat_x + jnp.where(bmove, state.bat_dir, 0)
        bhit_left = bx <= L_bat_xmin
        bhit_right = bx >= L_bat_xmax
        new_bat_dir = jnp.where(bhit_left, 1, jnp.where(bhit_right, -1, state.bat_dir)).astype(jnp.int32)
        new_bat_x = jnp.clip(bx, L_bat_xmin, L_bat_xmax).astype(jnp.int32)

        # --- moths fly left/right along their horizontal line (parallel to bats) ---
        mmove = (state.step_counter % c.moth_move_period) == 0
        mx = state.moth_x + jnp.where(mmove, state.moth_dir, 0)
        mhit_left = mx <= L_moth_xmin
        mhit_right = mx >= L_moth_xmax
        new_moth_dir = jnp.where(mhit_left, 1, jnp.where(mhit_right, -1, state.moth_dir)).astype(jnp.int32)
        new_moth_x = jnp.clip(mx, L_moth_xmin, L_moth_xmax).astype(jnp.int32)

        # --- laser ---
        new_laser_timer = jnp.where(
            laser_fire & (state.laser_timer <= 0),
            c.laser_duration,
            jnp.maximum(0, state.laser_timer - 1),
        ).astype(jnp.int32)
        laser_on = new_laser_timer > 0
        lx = jnp.where(new_facing < 0, new_x - c.laser_length, new_x + c.player_width).astype(jnp.int32)
        ly = (new_y + c.player_height // 3).astype(jnp.int32)

        # Creatures (spiders / bats / moths) are killed by the LASER it overlaps
        # OR by being caught in a DYNAMITE BLAST — dynamite is a valid weapon
        # against them, not only the breakable wall. Each grants the same points.
        def _blast_hit(alive, ox, oy, ow, oh):
            return explode_now & alive & self._aabb(ex, ey, ew, eh, ox, oy, ow, oh)

        # Spiders.
        spider_laser = (laser_on & state.spider_alive &
                        self._aabb(lx, ly, c.laser_length, c.laser_height,
                                   L_spider_x, new_spider_y, c.spider_width, c.spider_height))
        spider_kill = spider_laser | _blast_hit(state.spider_alive, L_spider_x, new_spider_y,
                                                c.spider_width, c.spider_height)
        spider_alive = state.spider_alive & (~spider_kill)
        spiders_killed = jnp.sum(spider_kill.astype(jnp.int32))

        # Bats.
        bat_laser = (laser_on & state.bat_alive &
                     self._aabb(lx, ly, c.laser_length, c.laser_height,
                                new_bat_x, L_bat_y, c.bat_width, c.bat_height))
        bat_kill = bat_laser | _blast_hit(state.bat_alive, new_bat_x, L_bat_y,
                                          c.bat_width, c.bat_height)
        bat_alive = state.bat_alive & (~bat_kill)
        bats_killed = jnp.sum(bat_kill.astype(jnp.int32))

        # Moths.
        moth_laser = (laser_on & state.moth_alive &
                      self._aabb(lx, ly, c.laser_length, c.laser_height,
                                 new_moth_x, L_moth_y, c.moth_width, c.moth_height))
        moth_kill = moth_laser | _blast_hit(state.moth_alive, new_moth_x, L_moth_y,
                                            c.moth_width, c.moth_height)
        moth_alive = state.moth_alive & (~moth_kill)
        moths_killed = jnp.sum(moth_kill.astype(jnp.int32))

        # --- player touches spider / bat / moth ---
        player_spider = (spider_alive &
                         self._aabb(new_x, new_y, c.player_width, c.player_height,
                                    L_spider_x, new_spider_y, c.spider_width, c.spider_height))
        player_bat = (bat_alive &
                      self._aabb(new_x, new_y, c.player_width, c.player_height,
                                 new_bat_x, L_bat_y, c.bat_width, c.bat_height))
        player_moth = (moth_alive &
                       self._aabb(new_x, new_y, c.player_width, c.player_height,
                                  new_moth_x, L_moth_y, c.moth_width, c.moth_height))
        died_spider = jnp.any(player_spider) | jnp.any(player_bat) | jnp.any(player_moth)

        # --- power drain ---
        drain = jnp.where(has_moved, c.power_drain_per_frame, 0)
        new_power = jnp.maximum(0, state.power - drain).astype(jnp.int32)
        died_power = (new_power <= 0) & (state.power > 0)

        # --- miner rescue ---
        touch_miner = (~state.miner_rescued) & self._aabb(
            new_x, new_y, c.player_width, c.player_height,
            L_miner_x, L_miner_y, c.miner_width, c.miner_height)
        power_bonus = jnp.where(touch_miner, (new_power * 100) // c.max_power, 0).astype(jnp.int32)

        # Rescuing the miner on the LAST level finishes the game; on any earlier
        # level it ADVANCES to the next level (score + lives carry over, the rest
        # of the run state is freshly loaded from the next level below).
        is_last = lvl >= (c.num_levels - 1)
        finish = touch_miner & is_last
        advance = touch_miner & (~is_last)

        # --- scoring ---
        new_score = (state.score
                     + spiders_killed * c.spider_points
                     + bats_killed * c.bat_points
                     + moths_killed * c.moth_points
                     + walls_broken * c.wall_points
                     + power_bonus).astype(jnp.int32)

        # --- death / lives / respawn ---
        died = (died_blast | died_spider | died_power) & (~touch_miner)
        new_lives = (state.lives - died.astype(jnp.int32)).astype(jnp.int32)
        respawned = died & (new_lives > 0)

        # --- next-level load (when advancing) ---
        next_lvl = jnp.clip(lvl + advance.astype(jnp.int32), 0, c.num_levels - 1)
        nl_start = c.LEVEL_PLAYER_START[next_lvl]
        cur_start = c.LEVEL_PLAYER_START[lvl]
        # A respawn returns to the CURRENT level's start; an advance loads the
        # NEXT level's start. Otherwise keep the computed position.
        final_x = jnp.where(advance, nl_start[0],
                            jnp.where(respawned, cur_start[0], new_x)).astype(jnp.int32)
        final_y = jnp.where(advance, nl_start[1],
                            jnp.where(respawned, cur_start[1], new_y)).astype(jnp.int32)

        # Things that reset on EITHER a respawn or a level advance.
        reset_pose = respawned | advance
        final_vy = jnp.where(reset_pose, 0.0, new_vy).astype(jnp.float32)
        final_facing = jnp.where(reset_pose, 1, new_facing).astype(jnp.int32)
        final_walk_timer = jnp.where(reset_pose, 0, walk_timer).astype(jnp.int32)
        final_has_moved = has_moved & (~reset_pose)
        final_laser = jnp.where(reset_pose, 0, new_laser_timer).astype(jnp.int32)
        final_dyn_active = dyn_active & (~reset_pose)
        final_explosion = jnp.where(reset_pose, 0, explosion_timer).astype(jnp.int32)
        final_dyn_fuse = jnp.where(reset_pose, 0, new_fuse).astype(jnp.int32)
        # Power refills on a respawn or a level advance; dynamite refills only on
        # a new level (a respawn keeps whatever sticks remain).
        final_power = jnp.where(reset_pose, c.max_power, new_power).astype(jnp.int32)
        final_dyn_count = jnp.where(advance, c.starting_dynamite, dynamite_count).astype(jnp.int32)

        # Spiders / walls: reset to the next level's roster on an advance; a plain
        # respawn leaves the spiders patrolling and broken walls broken.
        final_spider_y = jnp.where(advance, c.LEVEL_SPIDER_START_Y[next_lvl], new_spider_y).astype(jnp.int32)
        final_spider_dir = jnp.where(advance, c.LEVEL_SPIDER_START_DIR[next_lvl], new_spider_dir).astype(jnp.int32)
        final_spider_alive = jnp.where(advance, c.LEVEL_SPIDER_VALID[next_lvl], spider_alive)
        final_bat_x = jnp.where(advance, c.LEVEL_BAT_START_X[next_lvl], new_bat_x).astype(jnp.int32)
        final_bat_dir = jnp.where(advance, c.LEVEL_BAT_START_DIR[next_lvl], new_bat_dir).astype(jnp.int32)
        final_bat_alive = jnp.where(advance, c.LEVEL_BAT_VALID[next_lvl], bat_alive)
        final_moth_x = jnp.where(advance, c.LEVEL_MOTH_START_X[next_lvl], new_moth_x).astype(jnp.int32)
        final_moth_dir = jnp.where(advance, c.LEVEL_MOTH_START_DIR[next_lvl], new_moth_dir).astype(jnp.int32)
        final_moth_alive = jnp.where(advance, c.LEVEL_MOTH_VALID[next_lvl], moth_alive)
        final_wall_active = jnp.where(advance, c.LEVEL_WALL_VALID[next_lvl], wall_active)

        final_level = next_lvl.astype(jnp.int32)
        # miner_rescued is per-level: it latches true within a level but clears
        # when we drop into the next one (whose miner is still trapped).
        final_miner_rescued = jnp.where(advance, False, state.miner_rescued | touch_miner)
        level_complete = state.level_complete | finish

        game_over = state.game_over | (died & (new_lives <= 0))
        new_step = (state.step_counter + 1).astype(jnp.int32)

        # --- camera follows the player down, clamped to the (new) level's world ---
        cam_max = jnp.maximum(0, c.LEVEL_WORLD_HEIGHT[final_level] - c.cave_bottom)
        final_camera_y = jnp.clip(final_y - c.camera_anchor, 0, cam_max).astype(jnp.int32)

        new_state = HeroState(
            player_x=final_x,
            player_y=final_y,
            player_vy=final_vy,
            camera_y=final_camera_y,
            facing=final_facing,
            walk_timer=final_walk_timer,
            has_moved=final_has_moved,
            laser_timer=final_laser,
            power=final_power,
            lives=new_lives,
            score=new_score,
            level=final_level,
            dynamite_count=final_dyn_count,
            dyn_active=final_dyn_active,
            dyn_x=dyn_x,
            dyn_y=dyn_y,
            dyn_fuse=final_dyn_fuse,
            explosion_timer=final_explosion,
            spider_y=final_spider_y,
            spider_dir=final_spider_dir,
            spider_alive=final_spider_alive,
            bat_x=final_bat_x,
            bat_dir=final_bat_dir,
            bat_alive=final_bat_alive,
            moth_x=final_moth_x,
            moth_dir=final_moth_dir,
            moth_alive=final_moth_alive,
            wall_active=final_wall_active,
            miner_rescued=final_miner_rescued,
            level_complete=level_complete,
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
        L_walls = c.LEVEL_WALLS[lvl]
        L_break = c.LEVEL_WALL_BREAK[lvl]
        L_spider_x = c.LEVEL_SPIDER_X[lvl]
        L_bat_y = c.LEVEL_BAT_Y[lvl]
        L_moth_y = c.LEVEL_MOTH_Y[lvl]
        L_miner = c.LEVEL_MINER[lvl]
        player = ObjectObservation.create(
            x=state.player_x, y=state.player_y,
            width=jnp.array(c.player_width, jnp.int32),
            height=jnp.array(c.player_height, jnp.int32),
            active=jnp.array(True, jnp.bool_),
            orientation=jnp.where(state.facing < 0, 270, 90).astype(jnp.int32),
        )

        laser_active = state.laser_timer > 0
        lx = jnp.where(state.facing < 0, state.player_x - c.laser_length,
                       state.player_x + c.player_width).astype(jnp.int32)
        laser = ObjectObservation.create(
            x=lx, y=(state.player_y + c.player_height // 3).astype(jnp.int32),
            width=jnp.array(c.laser_length, jnp.int32),
            height=jnp.array(c.laser_height, jnp.int32),
            active=laser_active,
            orientation=jnp.where(state.facing < 0, 270, 90).astype(jnp.int32),
        )

        spiders = ObjectObservation.create(
            x=L_spider_x.astype(jnp.int32),
            y=state.spider_y.astype(jnp.int32),
            width=jnp.full((c.num_spiders,), c.spider_width, jnp.int32),
            height=jnp.full((c.num_spiders,), c.spider_height, jnp.int32),
            active=state.spider_alive,
            visual_id=jnp.arange(c.num_spiders, dtype=jnp.int32),
        )

        bats = ObjectObservation.create(
            x=state.bat_x.astype(jnp.int32),
            y=L_bat_y.astype(jnp.int32),
            width=jnp.full((c.num_bats,), c.bat_width, jnp.int32),
            height=jnp.full((c.num_bats,), c.bat_height, jnp.int32),
            active=state.bat_alive,
            visual_id=jnp.arange(c.num_bats, dtype=jnp.int32),
        )

        moths = ObjectObservation.create(
            x=state.moth_x.astype(jnp.int32),
            y=L_moth_y.astype(jnp.int32),
            width=jnp.full((c.num_moths,), c.moth_width, jnp.int32),
            height=jnp.full((c.num_moths,), c.moth_height, jnp.int32),
            active=state.moth_alive,
            visual_id=jnp.arange(c.num_moths, dtype=jnp.int32),
        )

        miner = ObjectObservation.create(
            x=L_miner[0].astype(jnp.int32), y=L_miner[1].astype(jnp.int32),
            width=jnp.array(c.miner_width, jnp.int32),
            height=jnp.array(c.miner_height, jnp.int32),
            active=(~state.miner_rescued),
        )

        dynamite = ObjectObservation.create(
            x=state.dyn_x, y=state.dyn_y,
            width=jnp.array(c.dyn_width, jnp.int32),
            height=jnp.array(c.dyn_height, jnp.int32),
            active=(state.dyn_active | (state.explosion_timer > 0)),
            state=state.explosion_timer,
        )

        walls = ObjectObservation.create(
            x=L_walls[:, 0], y=L_walls[:, 1], width=L_walls[:, 2], height=L_walls[:, 3],
            active=state.wall_active,
            visual_id=L_break.astype(jnp.int32),
        )

        return HeroObservation(
            player=player, laser=laser, spiders=spiders, bats=bats, moths=moths,
            miner=miner, dynamite=dynamite, walls=walls,
            power=state.power, lives=state.lives, score=state.score,
            dynamite_count=state.dynamite_count, level=state.level,
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
        # World objects live in WORLD coordinates whose y/height span the tall
        # multi-screen shaft (up to world_height), not just the visible screen,
        # so bound y/height by world_height. (get_object_space ties x/width to
        # the width arg and y/height to the height arg.)
        world = (c.world_height, c.screen_width)
        return spaces.Dict({
            "player": spaces.get_object_space(n=None, screen_size=world),
            "laser": spaces.get_object_space(n=None, screen_size=world),
            "spiders": spaces.get_object_space(n=c.num_spiders, screen_size=world),
            "bats": spaces.get_object_space(n=c.num_bats, screen_size=world),
            "moths": spaces.get_object_space(n=c.num_moths, screen_size=world),
            "miner": spaces.get_object_space(n=None, screen_size=world),
            "dynamite": spaces.get_object_space(n=None, screen_size=world),
            "walls": spaces.get_object_space(n=c.num_walls, screen_size=world),
            "power": spaces.Box(low=0, high=c.max_power, shape=(), dtype=jnp.int32),
            "lives": spaces.Box(low=0, high=c.starting_lives, shape=(), dtype=jnp.int32),
            "score": spaces.Box(low=0, high=jnp.iinfo(jnp.int32).max, shape=(), dtype=jnp.int32),
            "dynamite_count": spaces.Box(low=0, high=c.starting_dynamite, shape=(), dtype=jnp.int32),
            "level": spaces.Box(low=0, high=jnp.iinfo(jnp.int32).max, shape=(), dtype=jnp.int32),
        })

    def image_space(self) -> spaces.Box:
        c = self.consts
        return spaces.Box(low=0, high=255,
                          shape=(c.screen_height, c.screen_width, 3), dtype=jnp.uint8)

    def render(self, state: HeroState) -> jnp.ndarray:
        return self.renderer.render(state)


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------
# 5x3 bitmap font for digits 0-9 (used for the score and the level banner).
_DIGIT_FONT = {
    0: ["111", "101", "101", "101", "111"],
    1: ["010", "110", "010", "010", "111"],
    2: ["111", "001", "111", "100", "111"],
    3: ["111", "001", "111", "001", "111"],
    4: ["101", "101", "111", "001", "001"],
    5: ["111", "100", "111", "001", "111"],
    6: ["111", "100", "111", "101", "111"],
    7: ["111", "001", "010", "010", "010"],
    8: ["111", "101", "111", "101", "111"],
    9: ["111", "101", "111", "001", "111"],
}

# 5x3 bitmap font for the few HUD letters we need (POWER / LEVEL).
_LETTER_FONT = {
    'P': ["111", "101", "111", "100", "100"],
    'O': ["111", "101", "101", "101", "111"],
    'W': ["101", "101", "101", "111", "101"],
    'E': ["111", "100", "111", "100", "111"],
    'R': ["111", "101", "111", "110", "101"],
    'L': ["100", "100", "100", "100", "111"],
    'V': ["101", "101", "101", "101", "010"],
    ' ': ["000", "000", "000", "000", "000"],
}

# --- Pixel-art sprites -----------------------------------------------------
# Each sprite is a list of equal-length rows; characters map to RGBA colors via
# _ART_PALETTE ('.' = transparent). Built into (H, W, 4) arrays and fed through
# the same asset pipeline the other games use for their .npy sprites.
_ART_PALETTE = {
    '.': None,                       # transparent
    'Y': (232, 232, 74),             # rotor / fuse spark (yellow, ROM)
    'R': (167, 26, 26),              # helmet / dynamite (red, ROM)
    'B': (84, 138, 210),             # Roderick's suit (blue, ROM-measured)
    'b': (140, 168, 236),            # suit highlight (light blue)
    'W': (214, 214, 214),            # legs / boots (white, ROM-measured)
    'G': (110, 156, 66),             # miner body (green, ROM)
    'g': (60, 110, 40),              # miner shade (dark green)
    'P': (224, 156, 168),            # miner head (pink)
    'O': (196, 112, 60),             # spider body (orange-brown)
    'o': (230, 150, 92),             # spider highlight
    'S': (170, 170, 170),            # spider string (gray)
    'r': (150, 40, 40),              # dynamite shade (dark red)
    'M': (170, 120, 200),            # bat wing (purple)
    'm': (120, 84, 150),             # bat body (dark purple)
    'T': (236, 150, 200),            # moth wing (pink — distinct from orange spider)
    't': (150, 70, 110),             # moth body (magenta)
    'k': (50, 50, 50),               # lamp frame / hook (dark gray)
}

# Roderick Hero — pixel pattern measured from the ROM (blue suit/arms over white
# dangling legs), with the iconic yellow prop rotor added on top for the flight
# pose. Facing RIGHT; flipped horizontally for facing left. The shared upper body
# is followed by one of three interchangeable leg poses (idle + two walk frames).
_PLAYER_TOP = [
    ".YYYYYY.",  # rotor blade (yellow prop)
    "...YY...",  # prop shaft
    "BBBBBB..",  # shoulders / arms (ROM)
    "BBB.BB..",  # suit + gap (ROM)
    "BBB.BB..",  # suit (ROM)
    ".BB.B...",  # waist (ROM)
    ".BB..B..",  # hip (ROM)
]
_PLAYER_LEGS_IDLE = [
    "..WWW...",  # white legs (ROM: dangle below the blue suit)
    "..WW....",
    "..WW....",
    ".WW.W...",
]
_PLAYER_LEGS_WALK0 = [
    "..WWW...",
    "..WW....",
    ".WW.....",
    "WW......",
]
_PLAYER_LEGS_WALK1 = [
    "..WWW...",
    "...WW...",
    "...WW...",
    "..W.WW..",
]
_PLAYER_IDLE = _PLAYER_TOP + _PLAYER_LEGS_IDLE
_PLAYER_WALK0 = _PLAYER_TOP + _PLAYER_LEGS_WALK0
_PLAYER_WALK1 = _PLAYER_TOP + _PLAYER_LEGS_WALK1

# Springing spider hanging from a string.
_SPIDER_ART = [
    "...S...",
    "O.OOO.O",
    ".OoooO.",
    "OOoooOO",
    "O.O.O.O",
    "O.....O",
    ".......",
]

# Shaft bat — wings spread, seen head-on (9 wide x 5 tall). Flies side to side.
_BAT_ART = [
    "M.......M",
    "MM.m.m.MM",
    "MMMmmmMMM",
    ".MMmmmMM.",
    "..M.m.M..",
]

# Mine moth — broad orange wings around a brown body (9 wide x 5 tall). A new
# Level-3 enemy; flies side to side like the bat but visually distinct.
_MOTH_ART = [
    "TT.....TT",
    "TTT.t.TTT",
    ".TTtttTT.",
    ".T.ttt.T.",
    "....t....",
]

# Decorative lamp/lantern hanging in the shaft (5 wide x 7 tall): a yellow glow
# in a dark frame on a short hook. Scenery only — drawn, never collided with.
_LAMP_ART = [
    "..k..",
    ".kYk.",
    "kYYYk",
    "kYkYk",
    "kYYYk",
    ".kYk.",
    "..k..",
]

# Trapped miner, seated with a pink head.
_MINER_ART = [
    "..PP....",
    ".PPPP...",
    ".PPPP...",
    "..GG....",
    ".gGGg...",
    "GGGGGG..",
    "GGGGGGG.",
    "gG.gGGg.",
    "G...GG.G",
    "....GG..",
    "...g..g.",
    "..GG..GG",
]

# Dynamite stick with a lit fuse.
_DYN_ART = [
    ".Y..",
    ".S..",
    "rRRr",
    "RRRR",
    "RRRR",
    "RRRR",
    "RRRR",
    "rRRr",
]


class HeroRenderer(JAXGameRenderer):
    """
    Camera-based renderer. The tall cave (non-breakable walls baked in) is held
    as a 'cave' asset; each frame a viewport is sliced at `camera_y` and
    composited onto a static layer carrying the fixed HUD panel. Dynamic objects
    are drawn at screen-y = world-y - camera_y with procedural colored sprites.
    Replace the 'procedural' asset entries with real .npy sprites (see
    jax_freeway.py) when art is available.
    """

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

        # Per-level breakable-wall sprite size: pad every level's blastable panel
        # to the largest so the sprites can be stacked and indexed by level.
        max_bw = max(int(_LV["walls"][li, int(_LV["break_idx"][li]), 2]) for li in range(c.num_levels))
        max_bh = max(int(_LV["walls"][li, int(_LV["break_idx"][li]), 3]) for li in range(c.num_levels))

        asset_config = [
            {'name': 'background', 'type': 'background',
             'data': self._build_background()},
            {'name': 'player_idle', 'type': 'procedural',
             'data': self._sprite_from_art(_PLAYER_IDLE)},
            {'name': 'player_walk0', 'type': 'procedural',
             'data': self._sprite_from_art(_PLAYER_WALK0)},
            {'name': 'player_walk1', 'type': 'procedural',
             'data': self._sprite_from_art(_PLAYER_WALK1)},
            {'name': 'laser', 'type': 'procedural',
             'data': self._solid(c.laser_height, c.laser_length, c.laser_color)},
            {'name': 'spider', 'type': 'procedural',
             'data': self._sprite_from_art(_SPIDER_ART)},
            {'name': 'bat', 'type': 'procedural',
             'data': self._sprite_from_art(_BAT_ART)},
            {'name': 'moth', 'type': 'procedural',
             'data': self._sprite_from_art(_MOTH_ART)},
            {'name': 'miner', 'type': 'procedural',
             'data': self._sprite_from_art(_MINER_ART)},
            {'name': 'dynamite', 'type': 'procedural',
             'data': self._sprite_from_art(_DYN_ART)},
            {'name': 'explosion', 'type': 'procedural',
             'data': self._build_explosion(c.explosion_radius)},
            {'name': 'power_unit', 'type': 'procedural',
             'data': self._solid(c.power_bar_height, 1, c.power_color)},
            {'name': 'dyn_icon', 'type': 'procedural',
             'data': self._dyn_icon()},
            {'name': 'life_icon', 'type': 'procedural',
             'data': self._life_icon()},
            {'name': 'digits', 'type': 'digits', 'data': self._build_digits(c.text_color)},
            {'name': 'digits_dark', 'type': 'digits', 'data': self._build_digits((20, 20, 20))},
        ]
        # One cave raster + one breakable-panel sprite per level (selected by
        # state.level at render time). The cave is baked to the tallest world; a
        # shorter level is black-padded below its own floor.
        for li in range(c.num_levels):
            walls_np = _LV["walls"][li]
            break_np = _LV["wall_break"][li]
            valid_np = _LV["wall_valid"][li]
            rock = tuple(int(v) for v in _LV["rock_col"][li])
            floor = tuple(int(v) for v in _LV["floor_col"][li])
            asset_config.append({'name': f'cave_{li}', 'type': 'procedural',
                                 'data': self._build_cave(walls_np, break_np, valid_np, rock, floor,
                                                          _LV["lampx"][li], _LV["lampy"][li],
                                                          _LV["lampvalid"][li])})
            bidx = int(_LV["break_idx"][li])
            bw = int(walls_np[bidx, 2])
            bh = int(walls_np[bidx, 3])
            asset_config.append({'name': f'breakable_{li}', 'type': 'procedural',
                                 'data': self._build_breakable(max_bh, max_bw, floor, rock, bh, bw)})

        sprite_path = os.path.join(render_utils.get_base_sprite_dir(), "hero")

        (
            self.PALETTE, self.SHAPE_MASKS, self.BACKGROUND,
            self.COLOR_TO_ID, self.FLIP_OFFSETS,
        ) = self.jr.load_and_setup_assets(asset_config, sprite_path)

        # Stack player frames [idle, walk0, walk1] for dynamic frame selection.
        self.PLAYER_FRAMES = jnp.stack([
            self.SHAPE_MASKS["player_idle"],
            self.SHAPE_MASKS["player_walk0"],
            self.SHAPE_MASKS["player_walk1"],
        ])

        # Per-level cave rasters and breakable sprites, stacked for indexing by
        # state.level inside the jitted render().
        self.CAVES = jnp.stack([self.SHAPE_MASKS[f"cave_{li}"] for li in range(c.num_levels)])
        self.BREAKABLES = jnp.stack([self.SHAPE_MASKS[f"breakable_{li}"] for li in range(c.num_levels)])

        # Per-level "- LEVEL N -" title banners (black band + white text), drawn
        # on top of the final image; plus the white frame colour.
        self.BANNERS = jnp.stack([self._build_banner(li + 1) for li in range(c.num_levels)])
        self.WHITE = jnp.array((236, 236, 236), dtype=jnp.uint8)

    # --- procedural asset builders ----------------------------------------
    @staticmethod
    def _solid(h: int, w: int, color: Tuple[int, int, int]) -> jnp.ndarray:
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3] = color[0], color[1], color[2], 255
        return jnp.asarray(rgba)

    def _life_icon(self) -> jnp.ndarray:
        """A little-Hero life figure for the HUD: a BLUE Roderick (blue suit over
        white legs). Deliberately blue/white — not red/yellow — so lives can't be
        mistaken for the red dynamite sticks next to them."""
        c = self.consts
        rgba = np.zeros((9, 5, 4), dtype=np.uint8)   # transparent by default
        blue = np.array(c.player_color, dtype=np.uint8)
        white = np.array(c.text_color, dtype=np.uint8)
        rgba[0:6, 1:4, 0:3] = blue   # suit/body
        rgba[0:6, 1:4, 3] = 255
        rgba[6:9, 1:4, 0:3] = white  # dangling legs
        rgba[6:9, 1:4, 3] = 255
        return jnp.asarray(rgba)

    def _dyn_icon(self) -> jnp.ndarray:
        """A HUD dynamite stick: a red body with a yellow fuse spark on top, so
        the dynamite count reads unmistakably as sticks of dynamite."""
        c = self.consts
        rgba = np.zeros((7, 3, 4), dtype=np.uint8)
        rgba[0, 1, 0:3] = np.array(c.lamp_color, dtype=np.uint8)   # yellow fuse
        rgba[0, 1, 3] = 255
        rgba[1:7, :, 0:3] = np.array(c.dyn_color, dtype=np.uint8)  # red stick
        rgba[1:7, :, 3] = 255
        return jnp.asarray(rgba)

    @staticmethod
    def _sprite_from_art(art: List[str]) -> jnp.ndarray:
        """Build an (H, W, 4) RGBA sprite from a pixel-map using _ART_PALETTE."""
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
        """Radial blast: yellow core fading to orange, transparent outside."""
        c = self.consts
        size = 2 * radius
        yy, xx = np.mgrid[0:size, 0:size]
        dist = np.sqrt((xx - radius + 0.5) ** 2 + (yy - radius + 0.5) ** 2)
        rgba = np.zeros((size, size, 4), dtype=np.uint8)
        outer = dist <= radius
        core = dist <= radius * 0.55
        rgba[outer, 0:3] = np.array(c.explosion_color, dtype=np.uint8)
        rgba[outer, 3] = 255
        rgba[core, 0:3] = np.array((252, 232, 120), dtype=np.uint8)  # yellow core
        return jnp.asarray(rgba)

    def _build_breakable(self, canvas_h: int, canvas_w: int,
                         panel_color, frame_color, h: int = None, w: int = None) -> jnp.ndarray:
        """Breakable wall, drawn as a planked DOOR (the blastable panel) so it
        reads as a passage to blow open rather than a closed wall: a lighter
        panel inside a darker frame with horizontal plank lines. Dynamite makes
        the whole panel vanish, opening the way. The door is drawn at its true
        (h, w) in the top-left of a (canvas_h, canvas_w) transparent sprite so
        per-level doors of different sizes can be stacked and indexed."""
        if h is None:
            h = canvas_h
        if w is None:
            w = canvas_w
        panel = np.array(panel_color, dtype=np.uint8)   # lighter door panel
        frame = np.array(frame_color, dtype=np.uint8)   # darker rock frame
        rgba = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)  # transparent pad
        rgba[:h, :w, 3] = 255
        rgba[:h, :w, 0:3] = panel

        t = 2                          # frame thickness
        rgba[0:t, :w, 0:3] = frame     # top
        rgba[h - t:h, :w, 0:3] = frame # bottom
        rgba[:h, 0:t, 0:3] = frame     # left
        rgba[:h, w - t:w, 0:3] = frame # right

        # Horizontal plank seams so the panel reads as a (vertical) door/hatch.
        for yy in range(h // 4, h, max(1, h // 4)):
            rgba[yy:yy + 1, t:w - t, 0:3] = frame
        return jnp.asarray(rgba)

    def _build_background(self) -> jnp.ndarray:
        """Screen-sized static layer: the cave area is left black (the scrolling
        cave is composited on top each frame); the HUD gray panel is baked into
        the bottom band, matching the ROM (gray rows 142-188, inset 8px)."""
        c = self.consts
        bg = np.zeros((c.screen_height, c.screen_width, 4), dtype=np.uint8)
        bg[:, :, 3] = 255
        bg[:, :, 0:3] = np.array(c.bg_color, dtype=np.uint8)
        bg[c.hud_top:c.hud_bottom, 8:c.screen_width, 0:3] = np.array(c.hud_color, dtype=np.uint8)

        def text(s, x, y, color, scale=1):
            cx = x
            for ch in s:
                glyph = _LETTER_FONT.get(ch, _LETTER_FONT[' '])
                for r, row in enumerate(glyph):
                    for cc, v in enumerate(row):
                        if v == '1':
                            bg[y + r*scale:y + (r+1)*scale,
                               cx + cc*scale:cx + (cc+1)*scale, 0:3] = np.array(color, dtype=np.uint8)
                cx += 4 * scale
            return cx

        # "POWER" label (white) left of the power bar. (The level number lives in
        # the top title banner, as on the map, so there is no HUD "LEVEL" label.)
        text("POWER", 24, c.power_bar_y, c.text_color)

        # ACTIVISION rainbow stripe (the iconic logo bar) at the very bottom.
        rainbow = [(180, 60, 60), (200, 120, 60), (210, 200, 70),
                   (90, 180, 90), (70, 120, 200), (120, 90, 190)]
        for i, col in enumerate(rainbow):
            bg[194 + i, 56:104, 0:3] = np.array(col, dtype=np.uint8)
        return jnp.asarray(bg)

    def _build_cave(self, walls_np, breakable_np, valid_np, rock_color, floor_color,
                    lamp_x=None, lamp_y=None, lamp_valid=None) -> jnp.ndarray:
        """One level's tall vertical world (max_world_height x screen_width).
        Non-breakable walls AND the static decorative lamps are baked here; the
        camera slices a viewport out of it each frame. Wide horizontal bands use
        the lighter floor tone, tall vertical walls the darker rock tone. Unused
        (padded) wall slots are skipped. Shorter levels are left black below
        their floor."""
        c = self.consts
        cave = np.zeros((c.world_height, c.screen_width, 4), dtype=np.uint8)
        cave[:, :, 3] = 255
        cave[:, :, 0:3] = np.array(c.bg_color, dtype=np.uint8)
        for i in range(walls_np.shape[0]):
            if breakable_np[i] or not valid_np[i]:
                continue  # breakable drawn dynamically; padded slots are empty
            x, y, w, h = (int(v) for v in walls_np[i])
            color = floor_color if w >= h else rock_color
            cave[y:y + h, x:x + w, 0:3] = np.array(color, dtype=np.uint8)
        # Bake the lamps in. Per the design guide, completely static scenery
        # (the lamps never move and carry no state) belongs in the baked
        # background, not in the per-frame render loop. Only opaque pixels of the
        # lantern art are stamped, leaving the rest of the cave untouched.
        if lamp_x is not None:
            lamp = np.asarray(self._sprite_from_art(_LAMP_ART))  # (lh, lw, 4)
            lh, lw = lamp.shape[0], lamp.shape[1]
            opaque = lamp[:, :, 3] > 0
            for i in range(len(lamp_x)):
                if not bool(lamp_valid[i]):
                    continue
                ly, lx = int(lamp_y[i]), int(lamp_x[i])
                if 0 <= ly and ly + lh <= cave.shape[0] and 0 <= lx and lx + lw <= cave.shape[1]:
                    region = cave[ly:ly + lh, lx:lx + lw, 0:3]
                    region[opaque] = lamp[:, :, 0:3][opaque]
        return jnp.asarray(cave)

    def _build_banner(self, level_number: int) -> jnp.ndarray:
        """Top title band: solid black with the white '- LEVEL N -' caption
        centred, matching the banner drawn across the top of the level map."""
        c = self.consts
        bh, w = c.banner_height, c.screen_width
        img = np.zeros((bh, w, 3), dtype=np.uint8)  # black band

        s = f"- LEVEL {level_number} -"
        scale = 2
        glyph_w = 3 * scale
        spacing = 1 * scale
        text_w = len(s) * glyph_w + (len(s) - 1) * spacing
        x = (w - text_w) // 2
        y = (bh - 5 * scale) // 2
        color = np.array(c.text_color, dtype=np.uint8)

        def glyph(ch):
            if ch.isdigit():
                return _DIGIT_FONT[int(ch)]
            return _LETTER_FONT.get(ch, _LETTER_FONT[' '])

        cx = x
        for ch in s:
            for r, row in enumerate(glyph(ch)):
                for cc, v in enumerate(row):
                    if v == '1':
                        img[y + r * scale:y + (r + 1) * scale,
                            cx + cc * scale:cx + (cc + 1) * scale] = color
            cx += glyph_w + spacing
        return jnp.asarray(img)

    def _build_digits(self, color: Tuple[int, int, int]) -> jnp.ndarray:
        digits = np.zeros((10, 5, 3, 4), dtype=np.uint8)
        for d, rows in _DIGIT_FONT.items():
            for r, row in enumerate(rows):
                for col, ch in enumerate(row):
                    if ch == "1":
                        digits[d, r, col, 0:3] = np.array(color, dtype=np.uint8)
                        digits[d, r, col, 3] = 255
        return jnp.asarray(digits)

    # --- render -----------------------------------------------------------
    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: HeroState) -> jnp.ndarray:
        c = self.consts
        cam = state.camera_y
        lvl = state.level

        # Composite the scrolling cave viewport (rows [cam, cam+cave_bottom)) of
        # the CURRENT level onto the static layer (carrying the fixed HUD below).
        raster = self.jr.create_object_raster(self.BACKGROUND)
        cave_mask = self.CAVES[lvl]
        cave_slice = jax.lax.dynamic_slice(
            cave_mask, (cam, 0), (c.cave_bottom, c.screen_width))
        raster = jax.lax.dynamic_update_slice(raster, cave_slice, (0, 0))

        # World objects are drawn at screen-y = world-y - camera_y (clipped).
        def maybe(cond, x, y, mask, ras):
            return jax.lax.cond(
                cond,
                lambda r: self.jr.render_at_clipped(
                    r, x.astype(jnp.int32), (y - cam).astype(jnp.int32), mask),
                lambda r: r,
                ras,
            )

        # Breakable wall (this level's blastable panel + sprite).
        bidx = c.LEVEL_BREAK_IDX[lvl]
        bwall = c.LEVEL_WALLS[lvl, bidx]
        raster = maybe(state.wall_active[bidx], bwall[0], bwall[1], self.BREAKABLES[lvl], raster)

        # Miner.
        L_miner = c.LEVEL_MINER[lvl]
        raster = maybe(~state.miner_rescued, L_miner[0], L_miner[1],
                       self.SHAPE_MASKS["miner"], raster)

        # Spiders.
        L_spider_x = c.LEVEL_SPIDER_X[lvl]
        for i in range(c.num_spiders):
            raster = maybe(state.spider_alive[i], L_spider_x[i], state.spider_y[i],
                           self.SHAPE_MASKS["spider"], raster)

        # Bats (horizontal flyers).
        L_bat_y = c.LEVEL_BAT_Y[lvl]
        for i in range(c.num_bats):
            raster = maybe(state.bat_alive[i], state.bat_x[i], L_bat_y[i],
                           self.SHAPE_MASKS["bat"], raster)

        # Mine moths (horizontal flyers).
        L_moth_y = c.LEVEL_MOTH_Y[lvl]
        for i in range(c.num_moths):
            raster = maybe(state.moth_alive[i], state.moth_x[i], L_moth_y[i],
                           self.SHAPE_MASKS["moth"], raster)

        # (Decorative lamps are static scenery baked into the cave background by
        # _build_cave, so there is nothing to draw for them here.)

        # Dynamite.
        raster = maybe(state.dyn_active, state.dyn_x, state.dyn_y, self.SHAPE_MASKS["dynamite"], raster)

        # Explosion.
        raster = maybe(state.explosion_timer > 0,
                       state.dyn_x - c.explosion_radius, state.dyn_y - c.explosion_radius,
                       self.SHAPE_MASKS["explosion"], raster)

        # Player: pick idle/walk frame, flip to face the travel direction, and
        # bottom/centre-align the (taller/wider) sprite over the collision box.
        frame = jnp.where(state.walk_timer <= 0, 0, 1 + (state.walk_timer // 4) % 2)
        player_mask = self.PLAYER_FRAMES[frame]
        sh, sw = self.PLAYER_FRAMES.shape[1], self.PLAYER_FRAMES.shape[2]
        px = state.player_x + (c.player_width - sw) // 2
        py = state.player_y + (c.player_height - sh) - cam
        raster = self.jr.render_at_clipped(
            raster, px, py, player_mask, flip_horizontal=(state.facing < 0))

        # Laser.
        laser_on = state.laser_timer > 0
        lx = jnp.where(state.facing < 0, state.player_x - c.laser_length, state.player_x + c.player_width)
        ly = state.player_y + c.player_height // 3
        raster = maybe(laser_on, lx, ly, self.SHAPE_MASKS["laser"], raster)

        # Clip world objects to the cave viewport: restore the baked HUD panel
        # over any sprite that spilled past the bottom of the cave.
        raster = raster.at[c.cave_bottom:, :].set(self.BACKGROUND[c.cave_bottom:, :])

        # --- HUD ---
        # Power bar (filled width proportional to remaining power).
        power_units = (state.power * c.power_bar_width) // c.max_power

        def draw_unit(i, ras):
            return jax.lax.cond(
                i < power_units,
                lambda r: self.jr.render_at_clipped(
                    r, jnp.array(c.power_bar_x) + i, jnp.array(c.power_bar_y), self.SHAPE_MASKS["power_unit"]),
                lambda r: r,
                ras,
            )

        raster = jax.lax.fori_loop(0, c.power_bar_width, draw_unit, raster)

        # Dynamite remaining: a row of red sticks (yellow fuse) on the LEFT. This
        # only ever decreases when a stick is USED — losing a life never touches
        # it — and starts at 6 (starting_dynamite).
        raster = self.jr.render_indicator(raster, 12, c.lives_y,
                                          state.dynamite_count, self.SHAPE_MASKS["dyn_icon"],
                                          spacing=5, max_value=c.starting_dynamite)

        # Lives: row of little BLUE Roderick figures (distinct from the dynamite
        # sticks). These decrease when a life is lost.
        raster = self.jr.render_indicator(raster, c.lives_x, c.lives_y,
                                          state.lives, self.SHAPE_MASKS["life_icon"],
                                          spacing=8, max_value=c.starting_lives)

        # Score (5 digits, leading zeros) in the lower HUD band.
        score_digits = self.jr.int_to_digits(state.score, max_digits=5)
        raster = self.jr.render_label_selective(
            raster, 58, 180, score_digits,
            self.SHAPE_MASKS["digits"], 0, 5, spacing=4, max_digits_to_render=5)

        out = self.jr.render_from_palette(raster, self.PALETTE)

        # --- Map chrome (drawn on top of the finished frame) ---
        # "- LEVEL N -" title banner across the top (per current level). It
        # belongs to the top chamber (frame 1); as the camera scrolls down the
        # shaft it fades out so the lower chamber shows its ceiling band, banner-free.
        bh = c.banner_height
        banner = self.BANNERS[lvl]
        alpha = jnp.clip(1.0 - cam.astype(jnp.float32) / 40.0, 0.0, 1.0)
        under = out[0:bh, :, :].astype(jnp.float32)
        blended = (banner.astype(jnp.float32) * alpha
                   + under * (1.0 - alpha)).astype(jnp.uint8)
        out = out.at[0:bh, :, :].set(blended)
        # White frame border (2px) around the whole screen, as on the map.
        out = out.at[0:2, :, :].set(self.WHITE)
        out = out.at[-2:, :, :].set(self.WHITE)
        out = out.at[:, 0:2, :].set(self.WHITE)
        out = out.at[:, -2:, :].set(self.WHITE)
        return out
