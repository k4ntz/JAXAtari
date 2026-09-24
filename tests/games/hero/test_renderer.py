"""The H.E.R.O. renderer pre-bakes rooms, wall blanks and flares as colour-id masks.

Whatever the baking strategy, looking those masks up in the palette must give
back exactly the measured RLE screens from hero_levels.py (black for the
padding rooms of short levels) and the authored stamp rectangles, all in the
dtype of the raster they are drawn into.
"""
import numpy as np
import pytest

from jaxatari.games import hero_levels as HL
from jaxatari.games.jax_hero import JaxHero, _LV


@pytest.fixture(scope="module")
def env():
    return JaxHero()


def test_room_backgrounds_reproduce_the_measured_screens(env):
    r, c = env.renderer, env.consts
    palette = np.asarray(r.PALETTE)
    bgs = np.asarray(r.BGS)
    assert bgs.shape == (c.num_levels, c.max_rooms, c.cave_bottom, c.screen_width)
    assert bgs.dtype == np.asarray(r.BACKGROUND).dtype

    palettes = [getattr(HL, f"PALETTE_L{n}") for n in range(1, c.num_levels + 1)]
    blobs = [getattr(HL, f"BG_RLE_L{n}") for n in range(1, c.num_levels + 1)]
    black = np.zeros((c.cave_bottom, c.screen_width, 3), np.uint8)
    for li in range(c.num_levels):
        for ri in range(c.max_rooms):
            rgb = palette[bgs[li, ri]]
            expected = HL.decode_bg(blobs[li][ri], palettes[li]) if ri < len(blobs[li]) else black
            assert np.array_equal(rgb, expected), f"level {li + 1} room {ri} differs"


def test_wall_stamps_blank_exactly_the_wall_when_blasted(env):
    r, c = env.renderer, env.consts
    palette = np.asarray(r.PALETTE)
    stamps = np.asarray(r.DWALL_STAMPS)
    transparent = r.jr.TRANSPARENT_ID
    assert stamps.dtype == np.asarray(r.BACKGROUND).dtype
    # one stamp per wall: an intact wall is simply not drawn over
    assert stamps.shape[:2] == (c.num_levels, c.num_dwalls)
    for li in range(c.num_levels):
        for di in range(c.num_dwalls):
            opaque = stamps[li, di] != transparent
            if not _LV["dw_valid"][li, di]:
                assert not opaque.any()
                continue
            _, _, _, w, h, _ = (int(v) for v in _LV["dw"][li, di])
            expected = np.zeros(opaque.shape, bool)
            expected[:h, :w] = True
            assert np.array_equal(opaque, expected), f"level {li + 1} wall {di} footprint"
            assert (palette[stamps[li, di][opaque]] == 0).all()      # black


def test_flare_stamps_are_the_two_tone_flame_rectangles(env):
    r, c = env.renderer, env.consts
    palette = np.asarray(r.PALETTE)
    stamps = np.asarray(r.FLARE_STAMPS)
    transparent = r.jr.TRANSPARENT_ID
    assert stamps.dtype == np.asarray(r.BACKGROUND).dtype
    assert stamps.shape[:2] == (c.num_levels, c.num_flares)
    seen_valid = False
    for li in range(c.num_levels):
        for fi in range(c.num_flares):
            opaque = stamps[li, fi] != transparent
            if not _LV["fl_valid"][li, fi]:
                assert not opaque.any()
                continue
            seen_valid = True
            _, _, _, w, h, _, _ = (int(v) for v in _LV["fl"][li, fi])
            expected_rgb = np.zeros(opaque.shape + (3,), np.uint8)
            expected_rgb[:h, :w] = (252, 232, 120)
            expected_rgb[:h, 1:max(2, w - 1)] = (184, 50, 50)
            footprint = np.zeros(opaque.shape, bool)
            footprint[:h, :w] = True
            assert np.array_equal(opaque, footprint), f"level {li + 1} flare {fi} footprint"
            assert np.array_equal(palette[stamps[li, fi]][footprint], expected_rgb[footprint])
    assert seen_valid                                             # levels 7-10 have flares


def test_the_hero_animates_at_the_three_rates_the_rom_uses(env):
    """CHARACTERS.md, "Roderick Hero / Animation": standing on rock is ONE
    still picture - the rotor does not turn on the ground, not even while UP
    spins the thrust up. Off the ground the rotor cycles three poses at ONE
    frame each, and walking runs its stride at FOUR frames a pose.
    """
    import jax
    import jax.numpy as jnp
    r = env.renderer
    assert r.PLAYER_FRAMES.shape[0] == 5           # 3 rotor + 2 stride
    assert r.PLAYER_ROTOR_POSES == 3

    def frame_of(state):
        airborne = (jnp.abs(state.player_vy) > 0.5) | (state.thrust_timer > 0)
        rotor = state.step_counter % r.PLAYER_ROTOR_POSES
        walk = r.PLAYER_WALK_FRAME0 + (state.walk_timer // 4) % 2
        frame = jnp.where(airborne | (state.walk_timer <= 0), rotor, walk)
        return int(jnp.where(r._player_on_ground(state) & (state.walk_timer <= 0),
                             r.PLAYER_STAND_FRAME, frame))

    # stand him on the floor: a fresh reset, left to settle
    _, s = env.reset(jax.random.PRNGKey(0))
    for _ in range(150):
        _, s, _, _, _ = env.step(s, 0)
    assert bool(r._player_on_ground(s)), "the test hero must be on rock"
    still = s.replace(player_vy=jnp.float32(0.0), thrust_timer=jnp.int32(0),
                      walk_timer=jnp.int32(0))
    standing = [frame_of(still.replace(step_counter=jnp.int32(t)))
                for t in range(9)]
    assert standing == [r.PLAYER_STAND_FRAME] * 9, "one still picture"
    spinning = [frame_of(still.replace(step_counter=jnp.int32(t),
                                       thrust_timer=jnp.int32(t + 1)))
                for t in range(9)]
    assert spinning == [r.PLAYER_STAND_FRAME] * 9,         "the rotor does not turn on the ground while UP spins it up"

    # off the rock the rotor spins, hovering or flying
    air = still.replace(player_y=still.player_y - 20)
    assert not bool(r._player_on_ground(air))
    assert [frame_of(air.replace(step_counter=jnp.int32(t)))
            for t in range(6)] == [0, 1, 2, 0, 1, 2], "hovering spins the rotor"
    flying = air.replace(player_vy=jnp.float32(-1.0))
    assert [frame_of(flying.replace(step_counter=jnp.int32(t)))
            for t in range(6)] == [0, 1, 2, 0, 1, 2],         "the rotor spins at the same one-frame rate in the air"

    # and the renderer really draws it still: his rotor rows are the same
    # pixels frame after frame while he stands
    render = jax.jit(env.render)
    y, x = int(still.player_y), int(still.player_x)

    def rotor_rows(t):
        img = np.asarray(render(still.replace(step_counter=jnp.int32(t))))
        return img[y:y + 3, max(0, x - 2):x + 8]

    assert all(np.array_equal(rotor_rows(0), rotor_rows(t)) for t in (1, 2, 5))

    walking = [frame_of(still.replace(walk_timer=jnp.int32(t)))
               for t in range(1, 17)]
    assert walking == [3] * 3 + [4] * 4 + [3] * 4 + [4] * 4 + [3], \
        "two strides, each held four frames"
    # and the rotor is NOT spinning while he walks: both strides carry the
    # same wide rotor, so a walking hero redraws only when his legs move
    top = np.asarray(r.PLAYER_FRAMES)[:, :3]
    assert np.array_equal(top[3], top[4])
    assert np.array_equal(top[3], top[2]), "the stride keeps the wide rotor"
    assert not np.array_equal(top[0], top[1])
    assert not np.array_equal(top[1], top[2])


# --- the level banner -------------------------------------------------------
# Measured off the real ROM by booting each of the twenty levels and reading
# rows 176-189 of the status panel. In every one of them the banner is drawn
# on rows 179-186 in the power gauge's yellow #e8e84a, the word "LEVEL:"
# occupies exactly these six column runs, and the level number is right
# aligned into the score's own digit cells (the rightmost is 97-102, on an
# 8 px pitch). See HeroConstants.level_banner_frames for the duration.
BANNER_ROWS = (179, 186)
BANNER_YELLOW = (232, 232, 74)
WORD_RUNS = [(59, 62), (64, 67), (69, 73), (75, 78), (80, 83), (85, 86)]
# the ones digit of level 1 is a "1", which does not fill its 6 px cell
ROM_BANNER_RUNS = {
    1: WORD_RUNS + [(98, 101)],
    4: WORD_RUNS + [(97, 102)],
    10: WORD_RUNS + [(90, 93), (97, 102)],
    20: WORD_RUNS + [(89, 94), (97, 102)],
}
# level 4's banner, pixel for pixel, columns 50-111 of rows 179-186
ROM_LEVEL_4_BANNER = [
    ".........##...####.##.##.####.##...##.............##..........",
    ".........##...####.##.##.####.##...##............###..........",
    ".........##...##...##.##.##...##...##...........#.##..........",
    ".........##...###..##.##.###..##...............#..##..........",
    ".........##...###..##.##.###..##...............######.........",
    ".........##...##....###..##...##...##.............##..........",
    ".........####.####..###..####.####.##.............##..........",
    ".........####.####...#...####.####.##.............##..........",
]
# where the ROM puts a five-digit score, read off the recorded playthrough
ROM_SCORE_RUNS = [(66, 69), (73, 78), (81, 86), (89, 94), (97, 102)]


def _runs(mask):
    """[(first_col, last_col)] of each run of non-empty columns."""
    on = mask.any(axis=0)
    out, start = [], None
    for col in range(len(on) + 1):
        lit = col < len(on) and on[col]
        if lit and start is None:
            start = col
        elif not lit and start is not None:
            out.append((start, col - 1))
            start = None
    return out


def _panel(env, state, colour):
    img = np.asarray(env.renderer.render(state))[..., :3].astype(int)
    strip = img[BANNER_ROWS[0]:BANNER_ROWS[1] + 1]
    return (np.abs(strip - np.array(colour)).sum(axis=2) == 0)


def test_the_banner_draws_the_rom_bitmap_for_every_width_of_level_number(env):
    """One digit, one digit that does not fill its cell, and two digits."""
    import jax.numpy as jnp
    _, s = env.reset()
    for level, want in ROM_BANNER_RUNS.items():
        m = _panel(env, s.replace(level=jnp.int32(level - 1)), BANNER_YELLOW)
        assert _runs(m) == want, f"level {level}: {_runs(m)}"
        rows = np.nonzero(m.any(axis=1))[0]
        assert (BANNER_ROWS[0] + int(rows.min()),
                BANNER_ROWS[0] + int(rows.max())) == BANNER_ROWS


def test_the_banner_is_pixel_for_pixel_the_one_the_rom_draws(env):
    import jax.numpy as jnp
    _, s = env.reset()
    m = _panel(env, s.replace(level=jnp.int32(3)), BANNER_YELLOW)
    got = ["".join("#" if m[r, c] else "." for c in range(50, 112))
           for r in range(m.shape[0])]
    assert got == ROM_LEVEL_4_BANNER, "\n".join(
        f"  ours {a}\n  rom  {b}" for a, b in zip(got, ROM_LEVEL_4_BANNER)
        if a != b)


def test_the_banner_replaces_the_score_and_gives_it_back(env):
    """The ROM draws one or the other on that row, never both."""
    import jax.numpy as jnp
    _, s = env.reset()
    up = s.replace(level=jnp.int32(3), score=jnp.int32(12345))
    assert _panel(env, up, BANNER_YELLOW).any(), "the banner is up"
    assert not _panel(env, up, env.consts.text_color).any(), \
        "and the score is not drawn under it"

    down = up.replace(banner_timer=jnp.int32(0))
    assert not _panel(env, down, BANNER_YELLOW).any()
    assert _runs(_panel(env, down, env.consts.text_color)) == ROM_SCORE_RUNS, \
        "the score goes back into the ROM's own digit cells"


def test_the_banner_lasts_the_measured_111_frames(env):
    """Measured on the ROM: visible on frames 0-110 of a level and gone on
    111, and the count does not depend on what the player does - booting
    level 1 and holding NOOP and holding RIGHT give the same 111."""
    c = env.consts
    assert c.level_banner_frames == 111
    for action in (0, 3):                       # NOOP, RIGHT
        _, s = env.reset()
        assert int(s.banner_timer) == c.level_banner_frames
        seen = []
        for _ in range(c.level_banner_frames + 5):
            seen.append(int(s.banner_timer) > 0)
            _, s, _, _, _ = env.step(s, action)
        assert seen == [True] * c.level_banner_frames + [False] * 5, \
            f"action {action}: the banner ran for {sum(seen)} frames"


def test_the_banner_runs_again_on_a_new_level_but_not_on_a_new_life(env):
    import jax.numpy as jnp
    c = env.consts
    _, s = env.reset()
    for _ in range(c.level_banner_frames):      # let it run out
        _, s, _, _, _ = env.step(s, 0)
    assert int(s.banner_timer) == 0

    # rescue level 1's miner -> the next level's banner
    m = c.LEVEL_MINER[0]
    rescue = s.replace(room=m[0], player_x=m[1], player_y=m[2],
                       spider_alive=jnp.zeros_like(s.spider_alive))
    _, after, _, _, _ = env.step(rescue, 0)
    assert int(after.level) == 1
    assert int(after.banner_timer) == c.level_banner_frames

    # dying does not bring it back: the ROM shows it on level init only
    died = s.replace(power=jnp.int32(1), has_moved=jnp.bool_(True))
    _, after_death, _, _, _ = env.step(died, 3)
    assert int(after_death.lives) == c.starting_lives - 1, "he died"
    assert int(after_death.banner_timer) == 0, "and got no banner for it"


def test_the_power_gauge_reads_empty_while_the_banner_is_up(env):
    """Measured on the ROM at level 1: rows 145-149 are solid #a71a1a from
    x 49 to 127 on frame 0, and #e8e84a from 49 to 126 on frame 150. The
    gauge turns yellow on the same frame the banner goes."""
    import jax.numpy as jnp
    c = env.consts

    def bar(state):
        img = np.asarray(env.renderer.render(state))[..., :3].astype(int)
        rows = img[c.power_bar_y:c.power_bar_y + c.power_bar_height,
                   c.power_bar_x:c.power_bar_x + c.power_bar_width]
        lit = int((np.abs(rows - np.array(c.power_color)).sum(axis=2) == 0).sum())
        spent = int((np.abs(rows - np.array(c.power_spent_color)).sum(axis=2) == 0).sum())
        return lit, spent

    _, s = env.reset()
    whole = c.power_bar_width * c.power_bar_height
    assert bar(s) == (0, whole), "all red while the banner is up"
    assert bar(s.replace(banner_timer=jnp.int32(0))) == (whole, 0), \
        "and full yellow the moment it goes, with the power untouched"
    assert int(s.power) == c.max_power, "the gauge only READS empty"
