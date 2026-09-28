"""The global rules of H.E.R.O., the ones that hold on every level.

Level geometry lives in test_hero.py and the per-level files; this module
pins the rules that a level rebuild must never quietly break: the level
count, the missing publisher logo, magma being destructible, the measured
dynamite distances, the power gauge, and what a dark room actually looks
like.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxatari.games import hero_levels as HL
from jaxatari.games.jax_hero import (
    JaxHero, _DARK_TRIM_GREYS, _DARK_TRIM_ROWS, _MAGMA_DARK,
)

NOOP, FIRE, UP, RIGHT, LEFT, DOWN = 0, 1, 2, 3, 4, 5


@pytest.fixture(scope="module")
def env():
    return JaxHero()


# --- the level ladder -------------------------------------------------------
def test_twenty_levels_matching_the_rom_reference(env):
    assert HL.NUM_LEVELS == 20
    assert HL.ROOMS_PER_LEVEL == [2, 4, 6, 8, 8, 10, 12, 14] + [16] * 12, (
        "the ROM's room counts")
    assert int(env.consts.num_levels) == 20
    assert len(env.consts.LEVEL_ROOMS) == 20


# --- the publisher wordmark is gone ----------------------------------------
def test_the_logo_strip_is_black_and_the_hud_panel_survives(env):
    c = env.consts
    _, state = env.reset(jax.random.PRNGKey(0))
    img = np.asarray(env.render(state))
    assert (img[189:210] == 0).all(), "rows 189-209 must stay black"
    panel = img[c.hud_top:c.hud_bottom, 8:]
    assert (panel == np.array(c.hud_color, np.uint8)).all(axis=-1).any()


# --- dynamite ---------------------------------------------------------------
def test_the_rom_fuse_is_recorded_next_to_the_playable_one(env):
    c = env.consts
    assert c.dyn_fuse_rom == 34                      # measured on the ROM
    assert c.dyn_fuse_playable == 60                 # deliberate divergence
    assert not hasattr(c, "dyn_fuse")                # no ambiguous name left


def _plant_and_blow(env, x, flee):
    """Plant against level 1 room 0's pillar from x, hold `flee` frames of
    running left, then let the fuse burn down."""
    c = env.consts
    _, s = env.reset()
    s = s.replace(player_x=jnp.int32(x), player_y=jnp.int32(75))
    _, s, _, _, _ = env.step(s, DOWN)
    assert bool(s.dyn_active), "a stick must go down on solid ground"
    for _ in range(flee):
        _, s, _, _, _ = env.step(s, LEFT)
    for _ in range(c.dyn_fuse_playable + 6):
        _, s, _, _, _ = env.step(s, NOOP)
    return s


def test_a_stick_needs_solid_ground(env):
    """DOWN in mid-air lays nothing and only makes him sink faster."""
    c = env.consts
    _, s = env.reset()
    s = s.replace(player_x=jnp.int32(33), player_y=jnp.int32(30))   # mid-air
    _, air, _, _, _ = env.step(s, DOWN)
    assert not bool(air.dyn_active)
    assert int(air.dynamite_count) == c.starting_dynamite
    _, fall, _, _, _ = env.step(s, NOOP)
    assert int(air.player_y) > int(fall.player_y), "DOWN must sink faster"


@pytest.mark.parametrize("facing, walk_away", [(1, LEFT), (-1, RIGHT), (1, RIGHT), (-1, LEFT)])
def test_the_stick_lands_under_the_middle_of_the_hero_whichever_way_he_faces(
        env, facing, walk_away):
    """Measured on the ROM 2026-09-24 and in the recorded playthrough: the
    stick goes down under the MIDDLE of the hero, at his feet - not in front
    of him and not behind him - and it is the same facing right or left:

        facing right at RAM x 52   suit columns 53-58   stick 55-57
        facing left  at RAM x 16   suit columns 17-22   stick 19-21
        level_02 frame 11          suit 53-58, fuse under 56; frames 12-15 he
                                   walks away and the stick stays put

    It used to be drawn 2 px LEFT of the hero whichever way he faced. That
    was a drawing bug only, and the fix must stay one: state.dyn_x is the
    anchor the measured blast boxes are built on, so it must not move with
    the picture (see HeroConstants.dyn_draw_dx).
    """
    c = env.consts
    _, s = env.reset()
    px = 33
    s = s.replace(player_x=jnp.int32(px), player_y=jnp.int32(75),
                  facing=jnp.int32(facing))
    _, s, _, _, _ = env.step(s, DOWN)
    assert bool(s.dyn_active), "a stick must go down on solid ground"

    # the physics anchor did not move: the blast is still measured from here
    assert int(s.dyn_x) == px - 2
    # the observation reports the stick where it is drawn
    obs = env._get_observation(s)
    assert int(obs.dynamite.x) == px + 2

    # walk clear (either way) and read the stick off the screen
    for _ in range(20):
        _, s, _, _, _ = env.step(s, walk_away)
    assert bool(s.dyn_active), "the fuse must still be burning"
    img = np.asarray(env.render(s)).astype(int)
    rows = img[int(s.dyn_y):int(s.dyn_y) + c.dyn_height]
    stick = (np.all(rows == (184, 50, 50), axis=-1) |
             np.all(rows == (232, 232, 74), axis=-1))
    cols = sorted(set(np.nonzero(stick)[1].tolist()))
    assert cols == [px + 2, px + 3, px + 4], (
        f"the stick is drawn at columns {cols}; the ROM puts it under the "
        f"middle of a hero standing at x {px}, columns {px + 2}-{px + 4}")


@pytest.mark.parametrize("gap, breaks", [(0, True), (5, True), (6, False), (9, False)])
def test_blast_reaches_five_pixels_of_clear_air(env, gap, breaks):
    """Measured: a wall up to 5 px from the hero's own edge comes down; at
    9 px the stick does nothing at all. The pillar's left edge is x=60."""
    x = 60 - gap - env.consts.player_width
    s = _plant_and_blow(env, x, flee=30)
    assert (int(s.wall_stage[0]) == 2) is breaks


@pytest.mark.parametrize("gap, survives", [(0, False), (8, False), (10, True), (14, True)])
def test_ten_pixels_of_clearance_is_safe_and_eight_is_not(env, gap, survives):
    """Measured: retreating 8 px or less from the stick costs a life, 10 px
    or more is safe - and the wall comes down either way. Planting at x=54
    puts the stick's left edge at 52 and the hero's right edge at 60, so he
    opens a gap of one pixel per frame after the first eight."""
    c = env.consts
    s = _plant_and_blow(env, 54, flee=gap + 8)
    assert int(s.wall_stage[0]) == 2
    assert (int(s.lives) == c.starting_lives) is survives


# --- magma ------------------------------------------------------------------
# Magma used to be carried as a creature of kind 2, and these tests ran on
# whichever level still had that form - levels 8 to 12, then 13. Level 13 was
# the last: since its regeneration on 2026-09-24 every level carries magma as
# MAGMA rects from the '%' cells, and a two-cell magma pillar is a
# DESTRUCTIBLE zone like any rock one. So they run on level 13 room 1's pillar
# at cells 6-7 (x 32-39), which has open corridor on both sides and no other
# breakable wall within a blast of it.
MAGMA_LVL, MAGMA_ROOM, MAGMA_X = 12, 1, 32


def _magma_pillar(env):
    """(slot in DESTRUCTIBLE, base state) for level 13 room 1's magma pillar."""
    c = env.consts
    lvl, room = MAGMA_LVL, MAGMA_ROOM
    assert not any(row[4] == 2 for lv in range(HL.NUM_LEVELS)
                   for row in HL.SPIDERS[lv]), "no kind-2 stand-in is left"
    assert (room, MAGMA_X, 60, 8, 39) in HL.MAGMA[lvl]
    slot = HL.DESTRUCTIBLE[lvl].index((room, MAGMA_X, 60, 8, 39, 1))
    others = [z for i, z in enumerate(HL.DESTRUCTIBLE[lvl])
              if i != slot and z[0] == room]
    assert all(abs(z[1] - MAGMA_X) > 40 for z in others), "a blast alone"
    _, s = env.reset()
    s = s.replace(level=jnp.int32(lvl), room=jnp.int32(room),
                  spider_alive=c.SPIDER_VALID[lvl], invuln_timer=jnp.int32(0))
    return slot, s


def test_dynamite_destroys_magma_for_the_same_points_as_rock(env):
    """Measured on the ROM: a stick removes a magma column exactly like
    ordinary rock, for 75 points, and what it removes stops burning. (On
    level 9 room 0 blasting the red pillar is the only way down, so this
    cannot be a no-op.)"""
    c = env.consts
    slot, s = _magma_pillar(env)
    # planted by a hero standing at x 42, 2 px clear of the pillar's face
    # (dyn_x is his left edge minus 2), then out of the blast
    s = s.replace(player_x=jnp.int32(76), player_y=jnp.int32(75),
                  dyn_active=jnp.bool_(True), dyn_fuse=jnp.int32(1),
                  dyn_x=jnp.int32(40), dyn_y=jnp.int32(99 - c.dyn_height),
                  dyn_room=jnp.int32(MAGMA_ROOM))
    score0 = int(s.score)
    assert int(s.wall_stage[slot]) == 0
    _, s, _, _, _ = env.step(s, NOOP)
    stage = np.asarray(s.wall_stage)
    assert stage[slot] == 2, "the blast must take the magma"
    assert (np.flatnonzero(stage) == [slot]).all(), "and nothing else"
    assert c.wall_points == 75 and c.creature_points == 50
    assert int(s.score) - score0 == c.wall_points, "the WALL rate, once"
    # where it stood no longer burns: against the same face, alive
    s = s.replace(player_x=jnp.int32(34), player_y=jnp.int32(75),
                  invuln_timer=jnp.int32(0))
    _, s, _, _, _ = env.step(s, NOOP)
    assert int(s.lives) == c.starting_lives


def test_the_laser_does_not_remove_magma(env):
    """The laser kills creatures and melts rock; it does not touch magma
    (measured) - not in the 256 frames a rock column takes, nor after."""
    c = env.consts
    slot, s = _magma_pillar(env)
    stand = dict(player_x=jnp.int32(42), player_y=jnp.int32(75),
                 player_vy=jnp.float32(0.0))
    # the room's creatures go: the engine's bolt reaches room 1's hanging
    # spider at x 20, BEYOND the pillar, and its 50 points are not the
    # question here
    s = s.replace(facing=jnp.int32(-1),
                  spider_alive=jnp.zeros_like(s.spider_alive), **stand)
    score0 = int(s.score)
    for _ in range(c.laser_burn_frames + 40):
        _, s, _, _, _ = env.step(s, FIRE)
        s = s.replace(**stand)
    first = (MAGMA_X - 8) // 4
    assert not np.asarray(s.melted[MAGMA_ROOM])[first:first + 2].any()
    assert int(s.wall_stage[slot]) == 0, "magma is immune to the laser"
    assert int(s.score) == score0
    assert int(s.lives) == c.starting_lives


# --- the power gauge --------------------------------------------------------
def test_the_gauge_is_seventy_eight_pixels_of_sixty_eight_frames(env):
    c = env.consts
    assert c.power_bar_width == 78
    assert c.power_bar_x == 49 and c.power_bar_x + c.power_bar_width - 1 == 126
    assert c.power_bar_y == 145 and c.power_bar_height == 5       # rows 145-149
    assert c.power_frames_per_pixel == 68
    assert c.max_power == 78 * 68


def test_standing_still_costs_no_power_but_walking_does(env):
    c = env.consts
    _, s = env.reset()
    for _ in range(20):                       # get him moving and grounded
        _, s, _, _, _ = env.step(s, RIGHT)
    p0 = int(s.power)
    for _ in range(600):
        _, s, _, _, _ = env.step(s, NOOP)
    assert int(s.power) == p0, "an idle hero must not burn power"
    for _ in range(c.power_frames_per_pixel):
        _, s, _, _, _ = env.step(s, RIGHT)
    assert p0 - int(s.power) == c.power_frames_per_pixel          # one bar pixel


def test_the_bar_is_yellow_and_eaten_from_the_right_in_red(env):
    """How the gauge reads once the level is under way. For the 111 frames
    of the level banner the ROM draws the whole bar red whatever the power
    is, so the banner is run out first - see test_renderer.py."""
    c = env.consts
    _, s = env.reset()
    s = s.replace(power=jnp.int32(c.max_power // 3),
                  banner_timer=jnp.int32(0))
    row = np.asarray(env.render(s))[147, c.power_bar_x:c.power_bar_x + c.power_bar_width]
    lit = int(s.power) // c.power_frames_per_pixel
    assert (row[:lit] == np.array(c.power_color, np.uint8)).all()
    assert (row[lit:] == np.array(c.power_spent_color, np.uint8)).all()


def test_the_end_of_level_tally_pays_twenty_a_pixel(env):
    c = env.consts
    m = c.LEVEL_MINER[0]
    _, s = env.reset()
    s = s.replace(room=m[0], player_x=m[1], player_y=m[2],
                  spider_alive=jnp.zeros_like(s.spider_alive))
    power0 = int(s.power)
    _, _, reward, _, _ = env.step(s, NOOP)
    bonus = (power0 // c.power_frames_per_pixel) * c.bonus_per_power_pixel
    assert bonus == 1560                                  # a full gauge
    assert float(reward) == float(c.miner_points + bonus)


# --- darkness ---------------------------------------------------------------
def test_a_dark_room_draws_no_rock_only_trim_and_magma(env):
    """Measured: the three bands of rock are not drawn at all, the wavy trim
    on rows 16-19 and 138-141 stays in greys, and magma keeps its full red.
    Checked on the pre-baked dark backgrounds, so no sprite gets in the way.
    """
    c = env.consts
    r = env.renderer
    palette = np.asarray(r.PALETTE)
    dark = palette[np.asarray(r.BGS_DARK)]            # (nL, nR, 142, 160, 3)
    black, greys = (0, 0, 0), set(_DARK_TRIM_GREYS)

    trim = np.zeros(c.cave_bottom, bool)
    for r0, r1 in _DARK_TRIM_ROWS:
        trim[r0:r1] = True

    seen_trim, seen_magma = False, False
    for li in range(c.num_levels):
        for ri in range(int(c.LEVEL_ROOMS[li])):
            room = dark[li, ri]
            cols = {tuple(int(v) for v in px)
                    for px in np.unique(room.reshape(-1, 3), axis=0)}
            assert cols <= {black, _MAGMA_DARK} | greys, \
                f"level {li + 1} room {ri} shows {cols - ({black, _MAGMA_DARK} | greys)}"
            rock = {tuple(int(v) for v in px)
                    for px in np.unique(room[~trim].reshape(-1, 3), axis=0)}
            assert rock <= {black, _MAGMA_DARK}, \
                f"level {li + 1} room {ri} still draws rock in the dark"
            seen_trim |= bool({tuple(int(v) for v in px)
                               for px in np.unique(room[trim].reshape(-1, 3), axis=0)} & greys)
            seen_magma |= _MAGMA_DARK in cols
    assert seen_trim, "no room kept its trim"
    assert seen_magma, "no room kept its magma"


# --- the beam eats rock -----------------------------------------------------
RIGHTFIRE, LEFTFIRE = 8, 9


def _pin_against_the_pillar(env):
    """Walk right until level 1 room 0's pillar stops him."""
    _, s = env.reset()
    for _ in range(40):
        _, s, _, _, _ = env.step(s, RIGHT)
    assert int(s.player_x) == 54 and int(s.room) == 0
    return s


def test_the_beam_eats_a_column_every_256_frames(env):
    """Measured by pinning the hero against level 1 room 0's pillar and
    holding fire: cell 13 fell at ROM frame 257 and cell 14 at 513 - 256
    frames per 4 px column, and no points for either."""
    c = env.consts
    assert c.laser_burn_frames == 256
    s = _pin_against_the_pillar(env)
    score0 = int(s.score)
    falls = []
    seen = 0
    for f in range(1, 600):
        _, s, _, _, _ = env.step(s, RIGHTFIRE)
        n = int(np.asarray(s.melted[0]).sum())
        if n > seen:
            falls.append(f)
            seen = n
        if seen >= 2:
            break
    assert falls == [c.laser_burn_frames, 2 * c.laser_burn_frames], falls
    assert list(np.flatnonzero(np.asarray(s.melted[0]))) == [13, 14]
    assert int(s.score) == score0, "melting rock scores nothing"


def test_melting_clears_the_ceiling_and_middle_but_never_the_floor(env):
    """The column goes from both bands at once, exactly like a stick."""
    c = env.consts
    s = _pin_against_the_pillar(env)
    for _ in range(c.laser_burn_frames + 2):
        _, s, _, _, _ = env.step(s, RIGHTFIRE)
    assert bool(s.melted[0, 13])
    img = np.asarray(env.render(s))[:142]
    for x in (60, 63):                      # the melted column
        assert (img[16:60, x] == 0).all(), "ceiling band still drawn"
        assert (img[99:137, x] != 0).any(), "the floor band must survive"


def test_a_wall_that_reaches_a_screen_edge_never_melts(env):
    """700 frames against level 1 room 0's left wall changed nothing on the
    ROM - which is what keeps the hero from burning out of the cave."""
    _, s = env.reset()
    for _ in range(60):
        _, s, _, _, _ = env.step(s, LEFT)
    x0 = int(s.player_x)
    for _ in range(700):
        _, s, _, _, _ = env.step(s, LEFTFIRE)
    assert not bool(np.asarray(s.melted).any())
    assert int(s.player_x) == x0, "still walled in"


def test_the_burn_is_remembered_when_the_button_is_released(env):
    """Measured: 200 frames of burn, a 120-frame pause, then 57 more finished
    the column - the ROM keeps the progress."""
    c = env.consts
    s = _pin_against_the_pillar(env)
    part = 200
    for _ in range(part):
        _, s, _, _, _ = env.step(s, RIGHTFIRE)
    assert int(s.burn_timer) == part and int(s.burn_cell) == 13
    for _ in range(120):
        _, s, _, _, _ = env.step(s, RIGHT)
    assert int(s.burn_timer) == part, "the pause must not reset the burn"
    for f in range(1, 120):
        _, s, _, _, _ = env.step(s, RIGHTFIRE)
        if bool(s.melted[0, 13]):
            break
    assert f == c.laser_burn_frames - part


def test_melted_rock_resets_on_the_next_level(env):
    """Like a blasted wall and like the darkness, the melt lasts the level."""
    c = env.consts
    s = _pin_against_the_pillar(env)
    for _ in range(c.laser_burn_frames + 2):
        _, s, _, _, _ = env.step(s, RIGHTFIRE)
    assert bool(np.asarray(s.melted).any())
    m = c.LEVEL_MINER[0]
    s = s.replace(room=m[0], player_x=m[1], player_y=m[2],
                  spider_alive=jnp.zeros_like(s.spider_alive),
                  miner_rescued=jnp.bool_(False))
    _, s, _, _, _ = env.step(s, NOOP)
    assert int(s.level) == 1
    assert not bool(np.asarray(s.melted).any())
    assert int(s.burn_cell) == -1 and int(s.burn_timer) == 0


def test_every_level_has_meltable_walls_and_keeps_its_edges(env):
    """The beam works on all twenty levels, and no level's screen-edge run is
    ever meltable - otherwise the hero could burn his way out of the cave."""
    c = env.consts
    solid = np.asarray(c.CELL_SOLID)
    melt = np.asarray(c.MELTABLE)
    assert not melt[:, :, 2, :].any(), "the floor band is never eaten"
    assert (melt <= solid).all(), "only solid cells can melt"
    for lvl in range(c.num_levels):
        rooms = int(c.LEVEL_ROOMS[lvl])
        assert melt[lvl, :rooms].any(), f"level {lvl + 1} has nothing to melt"
        for room in range(rooms):
            for band in range(2):
                row = solid[lvl, room, band]
                if row[0]:
                    run = int(np.argmin(row)) if not row.all() else c.num_cells
                    assert not melt[lvl, room, band, :run].any(), \
                        f"L{lvl+1} r{room} b{band}: left edge run is meltable"
                if row[-1]:
                    run = c.num_cells - int(np.argmin(row[::-1]))
                    assert not melt[lvl, room, band, run:].any(), \
                        f"L{lvl+1} r{room} b{band}: right edge run is meltable"


# --- the raft ---------------------------------------------------------------
# Measured on the ROM 2026-09-24: a yellow 8x2 platform on the liquid of
# level 10 room 13, level 11 room 12 and level 12 room 14, which the recorded
# playthroughs ride under each room's magma wall. Keyed (level, room): level 16 has two, in rooms 11 and 13.
RAFT_ROOMS = {(10, 13): 124, (11, 12): 28, (12, 14): 124, (13, 12): 124,
              (14, 12): 28, (15, 13): 124, (16, 11): 28, (16, 13): 28,
              (17, 14): 124, (18, 12): 28, (19, 12): 124,
              (20, 11): 28}


def test_the_rafts_are_the_eight_measured_ones():
    """Level 13 room 12's raft was found by its ROM capture on 2026-09-24,
    under a six-cell magma wall; the playthrough rides it (frames 198-208).
    Level 14 room 12's, under the same wall, waits at the LEFT end: the level
    descends right, so the hero comes in from the left. Level 15 room 13's
    (2026-09-25), under a sixteen-cell wall, waits at the RIGHT end: that
    level descends left. Level 16 floats one in room 11 AND room 13, both at
    the LEFT end (it descends right); the ROM keeps one raft x for the two
    and puts it back at the entry end on entering either (2026-09-25).
    Level 17 room 14's (2026-09-26), under a sixteen-cell magma wall, waits
    at the RIGHT end: that level descends left. Level 18 room 12's
    (2026-09-26), under a sixteen-cell magma wall, waits at the LEFT end:
    that level descends right. Level 19 room 12's (2026-09-27), under a
    sixteen-cell magma wall, waits at the RIGHT end: that level descends left.
    Level 20 room 11's (2026-09-27), under a twenty-cell magma wall, waits at
    the LEFT end: that level descends right."""
    assert HL.RAFT_ENDS == (28, 124)
    assert (HL.RAFT_Y, HL.RAFT_W, HL.RAFT_H) == (136, 8, 2)
    got = {(lv + 1, rm): x for lv, r in enumerate(HL.RAFTS) for rm, x in r}
    assert got == RAFT_ROOMS


@pytest.mark.parametrize("level, room", sorted(RAFT_ROOMS))
def test_the_raft_is_not_painted_into_the_background(level, room):
    img = np.asarray(HL.decode_bg(getattr(HL, f"BG_RLE_L{level}")[room],
                                  getattr(HL, f"PALETTE_L{level}")))
    yellow = (img[136:142] == np.array(HL.RAFT_COLOUR)).all(axis=-1)
    assert not yellow.any(), "the engine draws the raft; the capture's copy must go"


def _on_the_raft(env, level, x_off=0, room=None):
    """The hero just above the waiting raft, every creature gone (level 12
    room 14's hanging spider sits in its path and kills him - on the ROM too)."""
    c = env.consts
    if room is None:
        (room,) = [rm for lv, rm in RAFT_ROOMS if lv == level]
    start = RAFT_ROOMS[(level, room)]
    _, s = env.reset()
    s = s.replace(level=jnp.int32(level - 1), room=jnp.int32(room),
                  raft_x=jnp.int32(start),
                  raft_dir=jnp.int32(-1 if start == c.raft_max_x else 1),
                  player_x=jnp.int32(start + c.raft_ride_dx + x_off),
                  player_y=jnp.int32(c.raft_y - c.player_height - 6),
                  spider_alive=jnp.zeros_like(s.spider_alive), has_moved=jnp.bool_(True))
    for _ in range(10):                      # fall the last few pixels
        _, s, _, _, _ = env.step(s, NOOP)
    return s


@pytest.mark.parametrize("level, room", sorted(RAFT_ROOMS))
def test_standing_on_the_raft_it_carries_him_one_px_a_frame(env, level, room):
    c = env.consts
    start = RAFT_ROOMS[(level, room)]
    s = _on_the_raft(env, level, room=room)
    assert int(s.player_y) == c.raft_y - c.player_height
    assert int(s.lives) == c.starting_lives, "the raft keeps him off the liquid"
    x0 = int(s.raft_x)
    step = -1 if start == c.raft_max_x else 1
    for k in range(1, 21):
        _, s, _, _, _ = env.step(s, NOOP)
        assert int(s.raft_x) == x0 + step * k
        assert int(s.player_x) == int(s.raft_x) + c.raft_ride_dx
    assert int(s.lives) == c.starting_lives


def test_the_raft_turns_at_each_end_holding_two_frames(env):
    c = env.consts
    s = _on_the_raft(env, 12)
    xs = []
    for _ in range(2 * (c.raft_max_x - c.raft_min_x) + 10):
        _, s, _, _, _ = env.step(s, NOOP)
        xs.append(int(s.raft_x))
    assert min(xs) == c.raft_min_x and max(xs) == c.raft_max_x
    i = xs.index(c.raft_min_x)
    assert xs[i - 1:i + 3] == [29, 28, 28, 29], "measured: ...29, 28, 28, 29..."
    assert int(s.lives) == c.starting_lives


def test_his_own_left_and_right_do_not_move_him_along_it(env):
    c = env.consts
    s = _on_the_raft(env, 12)
    x0 = int(s.raft_x)
    for a in (RIGHT, LEFT, RIGHT):
        for _ in range(10):
            _, s, _, _, _ = env.step(s, a)
            assert int(s.player_x) == int(s.raft_x) + c.raft_ride_dx
    assert int(s.raft_x) == x0 - 30, "it kept its own course, left"


def test_off_the_raft_the_liquid_kills(env):
    c = env.consts
    s = _on_the_raft(env, 12, x_off=-8)            # measured: 8 px off misses it
    for _ in range(5):
        _, s, _, _, _ = env.step(s, NOOP)
    assert int(s.lives) == c.starting_lives - 1


def test_the_raft_stops_the_frame_he_lifts_off(env):
    c = env.consts
    s = _on_the_raft(env, 12)
    for _ in range(10):
        _, s, _, _, _ = env.step(s, NOOP)
    for _ in range(60):
        _, s, _, _, _ = env.step(s, UP)
        if int(s.player_y) < c.raft_y - c.player_height:
            break
    parked = int(s.raft_x)
    for _ in range(30):
        _, s, _, _, _ = env.step(s, UP)
    assert int(s.raft_x) == parked


@pytest.mark.parametrize("from_room, key, end", [(13, LEFT, 124), (15, RIGHT, 28)])
def test_entering_the_room_puts_it_at_the_end_he_comes_in_from(env, from_room, key, end):
    """Level 12: room 13's left edge leads down into room 14's right side,
    room 15's right edge back up into its left side."""
    c = env.consts
    _, s = env.reset()
    x = 8 if key == LEFT else c.screen_width - 8 - c.player_width
    s = s.replace(level=jnp.int32(11), room=jnp.int32(from_room),
                  raft_x=jnp.int32(76), raft_dir=jnp.int32(1),
                  player_x=jnp.int32(x), player_y=jnp.int32(70),
                  spider_alive=jnp.zeros_like(s.spider_alive), has_moved=jnp.bool_(True))
    _, s, _, _, _ = env.step(s, key)
    assert int(s.room) == 14
    assert int(s.raft_x) == end
    assert int(s.raft_dir) == (-1 if end == 124 else 1)


def test_the_renderer_draws_the_raft_where_it_is(env):
    c = env.consts
    s = _on_the_raft(env, 12)
    for _ in range(40):
        _, s, _, _, _ = env.step(s, NOOP)
    img = np.asarray(env.render(s))
    x = int(s.raft_x)
    band = (img[136:138] == np.array(c.raft_color, np.uint8)).all(axis=-1)
    cols = np.nonzero(band.all(axis=0))[0]
    assert cols.min() == x and cols.max() == x + 7


# --- the respawn ------------------------------------------------------------
# Measured on the ROM 2026-09-25: after a death the hero drops in from the top
# of the screen AT THE X HE DIED AT and stops at corridor height, PY 73 (the
# engine's respawn_y, 62). Level 14 room 5, pinned deaths at x 30, 40 and 120
# at three heights; room 1 (bat, x 72) and room 2 (bat, x 101) free deaths.
def _die_here(env, level, room, x):
    """Kill the hero at (x, corridor) in this room: the power runs out."""
    _, s = env.reset(jax.random.PRNGKey(0))
    s = s.replace(level=jnp.int32(level - 1), room=jnp.int32(room),
                  player_x=jnp.int32(x), player_y=jnp.int32(62),
                  player_vy=jnp.float32(0), power=jnp.int32(1),
                  has_moved=jnp.bool_(True), banner_timer=jnp.int32(0),
                  spider_alive=jnp.zeros_like(s.spider_alive))
    lives = int(s.lives)
    _, s, *_ = env.step(s, 2)                         # UP: a moving frame
    assert int(s.lives) == lives - 1
    return s


@pytest.mark.parametrize("level, room, x", [(14, 1, 72), (14, 2, 101),
                                            (14, 5, 77), (15, 12, 140)])
def test_the_hero_respawns_in_the_column_he_died_in(env, level, room, x):
    s = _die_here(env, level, room, x)
    assert int(s.room) == room
    assert abs(int(s.player_x) - x) <= 1, "back where he died, not on the left"
    assert int(s.player_y) == env.consts.respawn_y


def test_a_death_right_of_a_wall_respawns_right_of_it(env):
    """Level 15 room 12's corridor has rock at cells 8-11, 15-16 and 28-31. A
    hero who dies at the right end comes back there, not at the left end."""
    band = "........####...##...........####......"
    s = _die_here(env, 15, 12, 140)
    cell = (int(s.player_x) - 8) // 4
    assert cell > 31 and band[cell] == "."


def test_a_column_solid_at_corridor_height_takes_the_nearest_free_one(env):
    """Level 15 room 10: rock at cells 10-17 (x 48-79). A death at x 64 comes
    back in the nearest free column on the same row, never inside the rock."""
    s = _die_here(env, 15, 10, 64)
    x = int(s.player_x)
    assert not bool(env._hits_wall(s, jnp.int32(x), jnp.int32(env.consts.respawn_y)))
    assert abs(x - 64) <= 24
