"""The global rules of HERO_SPEC.md, the ones that hold on every level.

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
        "the ROM's room counts (level_images/hero_rooms.py)")
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
def _magma_slot(env, level, room):
    """A kind-2 magma slot in that room whose blast reaches no breakable rock.

    Magma used to be carried as a creature of kind 2, and the levels that
    still are - 12 and 13 - are the ones no ROM regeneration has reached yet. A
    regenerated level carries its magma as MAGMA rects instead, and the cells
    a stick can take are in DESTRUCTIBLE like any other pillar, so these two
    tests have to run against a level that still has the old form. Levels 8,
    9, 10 and 11 were that level until they were regenerated on 2026-09-23/24.

    The slot is chosen rather than taken first: the test measures the score a
    blast pays for MAGMA ALONE, so a slot with a breakable wall inside the
    same blast box would add 75 that did not come from the magma.
    """
    c = env.consts
    hits = np.flatnonzero((np.asarray(c.SPIDER_KIND[level]) == 2) &
                          (np.asarray(c.SPIDER_ROOM[level]) == room) &
                          np.asarray(c.SPIDER_VALID[level]))
    assert hits.size, (
        f"no magma of kind 2 in level {level + 1} room {room}. If this level "
        f"has just been regenerated from the ROM, its magma is now in MAGMA "
        f"and DESTRUCTIBLE - point these tests at a level that still carries "
        f"the old form, or rewrite them against the rect tables.")
    _, s = env.reset()
    s = s.replace(level=jnp.int32(level), room=jnp.int32(room))
    px, py = (np.asarray(v) for v in env._spider_pos(s))
    d_room, d_x, d_y, d_w, d_h, d_solid = (np.asarray(v) for v in env._dwall_rects(s))
    dw = np.asarray(c.DESTRUCT[level])
    for slot in hits:
        mx, my = int(px[slot]), int(py[slot])
        r = c.explosion_radius
        ey, eh = my - r, 2 * r + c.dyn_height
        reach = c.blast_reach + 1
        ex, ew = mx + 2 - reach, c.player_width + 2 * reach
        clash = (d_solid & (dw[:, 5] > 0) & (d_room == room) &
                 (ex < d_x + d_w) & (d_x < ex + ew) &
                 (ey < d_y + d_h) & (d_y < ey + eh))
        if not clash.any():
            return int(slot)
    raise AssertionError(
        f"every magma slot in level {level + 1} room {room} has breakable "
        f"rock inside its blast box, so no score delta there is magma alone")


def test_dynamite_destroys_magma_for_the_same_points_as_rock(env):
    """Measured on the ROM: a stick removes a magma column exactly like
    ordinary rock, for 75 points. (On level 9 room 0 blasting the red pillar
    is the only way down, so this cannot be a no-op.)"""
    c = env.consts
    # level 12 room 1: a magma block with no destructible rock inside its
    # blast box, so every point of the delta has to come from the magma
    # itself. (Levels 8, 9, 10 and 11 were used here until they were
    # regenerated from the ROM and stopped carrying magma as a creature.)
    lvl, room = 11, 1
    slot = _magma_slot(env, lvl, room)
    _, s = env.reset()
    # reset() leaves LEVEL 1's roster in spider_alive, so every slot past its
    # first is already dead and a magma block here would read as "blasted"
    # before a stick was anywhere near it. Put this level's roster in.
    s = s.replace(level=jnp.int32(lvl), room=jnp.int32(room),
                  spider_alive=c.SPIDER_VALID[lvl])
    mx, my = (int(np.asarray(v)[slot]) for v in env._spider_pos(s))
    s = s.replace(player_x=jnp.int32(mx + 40), player_y=jnp.int32(75),
                  dyn_active=jnp.bool_(True), dyn_fuse=jnp.int32(1),
                  dyn_x=jnp.int32(mx), dyn_y=jnp.int32(my),
                  dyn_room=jnp.int32(room))
    score0 = int(s.score)
    alive0, stage0 = np.asarray(s.spider_alive), np.asarray(s.wall_stage)
    assert bool(alive0[slot])
    _, s, _, _, _ = env.step(s, NOOP)
    assert not bool(s.spider_alive[slot]), "the blast must take the magma"
    assert int((alive0 & ~np.asarray(s.spider_alive)).sum()) == 1
    assert (np.asarray(s.wall_stage) == stage0).all(), "no rock in reach"
    # magma pays the WALL rate, not the creature rate
    assert c.wall_points == 75 and c.creature_points == 50
    assert int(s.score) - score0 == c.wall_points


def test_the_laser_does_not_remove_magma(env):
    """The laser kills creatures; it does not cut rock and it does not touch
    magma (measured)."""
    lvl, room = 11, 1
    slot = _magma_slot(env, lvl, room)
    _, s = env.reset()
    # see the note in the dynamite test: reset() hands back level 1's roster
    s = s.replace(level=jnp.int32(lvl), room=jnp.int32(room),
                  spider_alive=env.consts.SPIDER_VALID[lvl])
    mx, my = (int(np.asarray(v)[slot]) for v in env._spider_pos(s))
    s = s.replace(player_x=jnp.int32(max(8, mx - 20)),
                  player_y=jnp.int32(my - 6), facing=jnp.int32(1))
    score0 = int(s.score)
    for _ in range(30):
        _, s, _, _, _ = env.step(s, FIRE)
    assert bool(s.spider_alive[slot]), "magma is immune to the laser"
    assert int(s.score) == score0


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
