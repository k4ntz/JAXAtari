"""Checks a ROM-measured H.E.R.O. level has to pass.

These prove a level's committed data from the committed data alone - no
ale-py, no ROM, no reference database in the repository - so the repo can
check itself. Each check takes the band strings the level is supposed to draw
from its caller, because the expectation belongs in the test that states it.

The ROM itself stays the source of truth, and lives outside this repository:
``level_images/tools/hero_extract.py`` reads it and
``level_images/tools/check_level.py N`` diffs this environment's rendering
against it, room by room. Run that after changing a level; these checks
cannot see the ROM and will happily agree with wrong data that is internally
consistent.

Not a test module itself: ``test_levelN.py`` calls into it.
"""
from jaxatari.games import hero_levels as HL

# HERO_SPEC.md section 2: three bands of 38 cells of 4 px, first cell at x 8.
BANDS = {"A": (16, 59), "B": (60, 98), "C": (99, 141)}
BAND_HEIGHT = {16: 44, 60: 39, 99: 43}
CELLS, CELL_W, X0 = 38, 4, 8
# the rows a '~' cell IS drawn on; everything above them in that cell is air
LIQUID_ROWS = (136, 141)
ROLES = ("edge", "dark", "mid", "light", "hi1", "hi2", "hi3")
MAGMA_RGB = {(167, 26, 26), (184, 50, 50)}


def rgb(hexstr):
    return (int(hexstr[1:3], 16), int(hexstr[3:5], 16), int(hexstr[5:7], 16))


def level_data(level):
    """(palette, [background per room]) for a 1-based level number."""
    return (getattr(HL, f"PALETTE_L{level}"),
            [HL.decode_bg(blob, getattr(HL, f"PALETTE_L{level}"))
             for blob in getattr(HL, f"BG_RLE_L{level}")])


def bands_of(image):
    """The three 38-character band strings a rendered room actually draws.

    Sampled down the middle of each band, the way the reference pack's own
    checker does it, so the wavy trim on the outer rows cannot confuse it.

    There are FOUR cell values, not three. A floor cell that is black down the
    middle may still be `~`, the lethal LIQUID surface: it is empty from row 99
    to about 135 and drawn only on rows 136-141, so the middle sample sees
    black and a reader that stops there calls a water room a row of holes. It
    is a floor the hero dies on and it is not a way down (CORRECTIONS.md, the
    second bug). Level 7 room 10 is the first room in this repository that has
    any, and it is twenty-four cells of it.
    """
    out = {}
    for band, (r0, r1) in BANDS.items():
        row = (r0 + r1) // 2
        s = ""
        for i in range(CELLS):
            x = X0 + CELL_W * i
            px = tuple(int(v) for v in image[row, x + 1])
            if px != (0, 0, 0):
                s += "%" if px in MAGMA_RGB else "#"
                continue
            strip = image[LIQUID_ROWS[0]:LIQUID_ROWS[1] + 1, x:x + CELL_W]
            drawn = (strip.reshape(-1, strip.shape[-1]).sum(axis=-1) > 0).mean()
            s += "~" if (band == "C" and drawn > 0.5) else "."
        out[band] = s
    return out


# --- the checks -------------------------------------------------------------
def check_shape(level, hue, rooms):
    """The level is the size and the colour the ROM says it is."""
    idx = level - 1
    assert HL.ROOMS_PER_LEVEL[idx] == rooms
    assert ("brown", "green", "blue", "grey")[idx % 4] == hue, \
        "the cave hue cycles brown / green / blue / grey"
    palette, bgs = level_data(level)
    assert len(bgs) == rooms
    for role in ROLES:
        assert rgb(HL.SHADES[level][role]) in palette, \
            f"level {level}: {role} {HL.SHADES[level][role]} missing from the palette"


def check_bands(level, want):
    """Every room draws exactly the bands the test says it should.

    ``want`` is [{'A':..., 'B':..., 'C':...}] in descent order.
    """
    _, bgs = level_data(level)
    assert len(bgs) == len(want), "one expected band set per room"
    for room, bg in enumerate(bgs):
        got = bands_of(bg)
        for band in "ABC":
            assert got[band] == want[room][band], (
                f"level {level} room {room} band {band}\n"
                f"  ours     {got[band]}\n  expected {want[room][band]}")


def check_walls_on_the_grid(level):
    """Every rect starts on a 4 px column and fills exactly one band.

    HERO_SPEC.md: a wall edge at an x that is not a multiple of 4, or a band
    boundary on a row other than 16 / 60 / 99 / 142, is a bug.

    The LIQUID surface is the one exception, and it is not a band: a `~` cell
    is drawn only on rows 136-141 and is black above them, so its rect is
    those six rows. Filling the whole floor band instead puts an invisible
    ledge across the room thirty-seven pixels above the water.
    """
    liquid_h = LIQUID_ROWS[1] - LIQUID_ROWS[0] + 1
    for room, rects in enumerate(getattr(HL, f"WALL_RECTS_L{level}")):
        for x, y, w, h in rects:
            assert (x - X0) % CELL_W == 0 and w % CELL_W == 0, \
                f"level {level} room {room}: {x},{w} off the 4 px grid"
            if (y, h) == (LIQUID_ROWS[0], liquid_h):
                continue
            assert y in BAND_HEIGHT, \
                f"level {level} room {room}: band top {y} is not 16/60/99"
            assert h == BAND_HEIGHT[y], \
                f"level {level} room {room}: {y}+{h} does not fill its band"


def check_rects_cover_the_rock(level):
    """No invisible walls and no phantom gaps.

    Compared cell by cell rather than pixel by pixel: the wavy trim on rows
    16-19 and 138-141 leaves some rows of a solid cell black, which is what
    the ROM draws, so a pixel-exact comparison would be wrong.
    """
    _, bgs = level_data(level)
    rects_per_room = getattr(HL, f"WALL_RECTS_L{level}")
    zones = HL.DESTRUCTIBLE[level - 1]
    for room, bg in enumerate(bgs):
        drawn = bands_of(bg)
        covered = {b: [False] * CELLS for b in "ABC"}
        boxes = list(rects_per_room[room])
        boxes += [(x, y, w, h) for r, x, y, w, h, _ok in zones if r == room]
        for (x, y, w, h) in boxes:
            for band, (r0, r1) in BANDS.items():
                if y > r1 or y + h <= r0:
                    continue
                for cell in range(CELLS):
                    cx = X0 + CELL_W * cell
                    if cx < x + w and cx + CELL_W > x:
                        covered[band][cell] = True
        for band in "ABC":
            for cell in range(CELLS):
                solid = drawn[band][cell] != "."
                assert solid == covered[band][cell], (
                    f"level {level} room {room} band {band} cell {cell} "
                    f"(x {X0 + CELL_W * cell}): drawn {'rock' if solid else 'air'} "
                    f"but {'covered' if covered[band][cell] else 'uncovered'}")


def check_miner(level, want):
    """He is on the last screen of the level, standing in open corridor."""
    room, x, y = HL.MINER_POS[level - 1]
    assert room == HL.ROOMS_PER_LEVEL[level - 1] - 1, \
        "the miner is on the last screen of the level"
    assert want[room]["B"][(x - X0) // CELL_W] == ".", \
        "the miner stands in open corridor"


def check_destructibles(level, want):
    """Blastable walls sit on the grid, inside a band, and never on the edge.

    A stick takes a column of the ceiling and middle bands together and never
    the floor band (HERO_SPEC.md section 6) - except in a room whose floor has
    no hole at all, where blasting the floor is the only way down. And a run
    of rock that reaches the side of the screen never breaks, which is what
    keeps the hero inside the cave.
    """
    zones = HL.DESTRUCTIBLE[level - 1]
    sealed = {r for r in range(HL.ROOMS_PER_LEVEL[level - 1] - 1)
              if "." not in want[r]["C"]}
    for (room, x, y, w, h, dyn_ok) in zones:
        assert (x - X0) % CELL_W == 0 and w % CELL_W == 0, f"{x},{w} off the grid"
        assert dyn_ok in (0, 1)
        if y == 99:                       # a floor zone only exists to open a
            assert room in sealed, \
                f"level {level} room {room}: floor zone in a room that has a hole"
            continue
        if y == 16 and h == 44 and room - 1 in sealed:
            continue                      # the ceiling twin of the room above
        cells = want[room]["B" if y >= 60 else "A"]
        first = (x - X0) // CELL_W
        last = first + w // CELL_W - 1
        assert first > 0 and last < CELLS - 1, \
            f"level {level} room {room}: zone at x {x} touches the screen edge"
        assert all(ch != "." for ch in cells[first:last + 1]), \
            f"level {level} room {room}: zone at x {x} is not over rock"


def check_shared_walls(level):
    """Shared-wall groups name real slots, and every slot is in one group."""
    zones = HL.DESTRUCTIBLE[level - 1]
    groups = HL.SHARED_WALLS[level - 1]
    seen = set()
    for group in groups:
        assert len(group) > 1, "a group of one is not a shared wall"
        for slot in group:
            assert 0 <= slot < len(zones), f"slot {slot} out of range"
            assert slot not in seen, f"slot {slot} is in two groups"
            seen.add(slot)
        # every member is the same 8 px column of the same level
        xs = {zones[s][1] for s in group}
        ws = {zones[s][3] for s in group}
        assert len(xs) == len(ws) == 1, \
            f"level {level}: group {group} spans more than one column"


def check_creatures_are_sane(level):
    """Kinds the engine knows, patrols that make sense, rooms that exist.

    Every creature must also carry a measured motion entry - the archetype
    fallback exists for levels nobody has remeasured, and a regenerated level
    is not allowed to fall back to it.
    """
    for slot, (room, x, y, patrol, kind) in enumerate(HL.SPIDERS[level - 1]):
        # 0 hanging spider, 1 bat, 3 wall snake, 4 untethered spider
        assert kind in (0, 1, 3, 4), \
            f"level {level}: unexpected creature kind {kind}"
        assert patrol >= 0
        assert 0 <= room < HL.ROOMS_PER_LEVEL[level - 1]
        assert (level, slot) in HL.CREATURE_MOTION, (
            f"level {level} creature {slot} has no measured motion; it would "
            f"fall back to the kind-{kind} archetype")
        assert (patrol == 0) == ((level, slot) not in HL.CREATURE_PATROL), (
            f"level {level} creature {slot}: a creature that sweeps sideways "
            f"needs its own measured patrol period, and one that does not "
            f"must not carry one")
    for room, x, y in HL.LANTERNS[level - 1]:
        assert BANDS["A"][0] <= y <= BANDS["A"][1], \
            "a lantern hangs in the ceiling band, not at the miner"
        assert 0 <= room < HL.ROOMS_PER_LEVEL[level - 1]


def check_magma_is_cave(level, want):
    """Magma rects are exactly the '%' cells, and they are solid rock too."""
    expect = []
    for room, bands in enumerate(want):
        for band in "ABC":
            y = BANDS[band][0]
            s = bands[band]
            run = None
            for i, ch in enumerate(s + "."):
                if i < CELLS and ch == "%" and run is None:
                    run = i
                elif (i == CELLS or ch != "%") and run is not None:
                    expect.append((room, X0 + CELL_W * run, y,
                                   CELL_W * (i - run), BAND_HEIGHT[y]))
                    run = None
    assert sorted(HL.MAGMA[level - 1]) == sorted(expect)
