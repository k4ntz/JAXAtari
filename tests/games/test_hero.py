"""Tests for the measured H.E.R.O. environment (jax_hero, levels 1-3).

The environment models the ROM's screen-flip world: each level is a stack of
static rooms; geometry, physics and object positions were measured from ALE
captures (see jax_hero.py's provenance notes).
"""
import jax
import jax.numpy as jnp
import numpy as np

from jaxatari.games import hero_levels as HL
from jaxatari.games.jax_hero import JaxHero

# Compact action indices (see JaxHero.ACTION_SET).
NOOP, FIRE, UP, RIGHT, LEFT, DOWN = 0, 1, 2, 3, 4, 5
UPRIGHT, UPLEFT, RIGHTFIRE, LEFTFIRE, UPFIRE, DOWNFIRE = 6, 7, 8, 9, 10, 11
DOWNRIGHT, DOWNLEFT = 12, 13


def _env():
    return JaxHero()


def test_reset_starting_inventory():
    env = _env()
    c = env.consts
    obs, state = env.reset(jax.random.PRNGKey(0))
    assert int(state.lives) == c.starting_lives == 4     # measured from the ROM
    assert int(state.dynamite_count) == c.starting_dynamite == 6
    assert int(state.power) == c.max_power
    assert int(state.level) == 0
    assert int(state.room) == 0
    assert bool((state.spider_alive == c.SPIDER_VALID[0]).all())
    assert not bool(state.game_over)
    assert not bool(state.level_complete)
    # measured spawn pose
    assert int(state.player_x) == c.spawn_x
    assert int(state.player_y) == c.spawn_y


def test_power_only_drains_after_first_move():
    env = _env()
    _, state = env.reset(jax.random.PRNGKey(0))
    _, s, _, _, _ = env.step(state, NOOP)
    assert int(s.power) == int(state.power)
    _, s, _, _, _ = env.step(s, RIGHT)
    _, s, _, _, _ = env.step(s, NOOP)
    assert int(s.power) < int(state.power)


def test_fall_is_constant_one_pixel_per_frame():
    """Measured: free fall moves exactly 1 px/frame with no acceleration."""
    env = _env()
    _, state = env.reset()
    # mid-air in the central shaft (x~74..88 is open in room 0)
    state = state.replace(player_x=jnp.int32(76), player_y=jnp.int32(40),
                          player_vy=jnp.float32(1.0))
    ys = []
    for _ in range(5):
        _, state, _, _, _ = env.step(state, NOOP)
        ys.append(int(state.player_y))
    assert ys == [41, 42, 43, 44, 45]


def test_thrust_stops_fall_then_rises():
    """Measured: UP cancels the fall immediately, then ramps into a rise."""
    env = _env()
    _, state = env.reset()
    state = state.replace(player_x=jnp.int32(76), player_y=jnp.int32(60),
                          player_vy=jnp.float32(1.0))
    _, state, _, _, _ = env.step(state, UP)
    assert int(state.player_y) == 60          # fall cancelled instantly
    for _ in range(24):
        _, state, _, _, _ = env.step(state, UP)
    assert int(state.player_y) < 60           # eventually rising


def test_walk_speed_one_pixel_per_frame():
    env = _env()
    _, state = env.reset()
    x0 = int(state.player_x)
    _, s, _, _, _ = env.step(state, RIGHT)
    assert int(s.player_x) == x0 + 1
    assert int(s.facing) == 1
    _, s, _, _, _ = env.step(s, LEFT)
    assert int(s.facing) == -1


def test_walls_block_movement():
    """Dropped over the room-0 floor band (top at y=99), the player must
    come to rest on it, never inside it."""
    env = _env()
    _, state = env.reset()
    state = state.replace(player_x=jnp.int32(40), player_y=jnp.int32(60),
                          player_vy=jnp.float32(1.0))
    for _ in range(60):
        _, state, _, _, _ = env.step(state, NOOP)
    assert int(state.player_y) + env.consts.player_height <= 99
    assert int(state.room) == 0


def test_room_flip_down_and_up():
    """Crossing the screen bottom flips to the next room (measured model)."""
    env = _env()
    c = env.consts
    _, state = env.reset()
    state = state.replace(player_x=jnp.int32(76), player_y=jnp.int32(132),
                          player_vy=jnp.float32(1.0))
    for _ in range(4):
        _, state, _, _, _ = env.step(state, NOOP)
        if int(state.room) == 1:
            break
    assert int(state.room) == 1
    assert int(state.player_y) == c.flip_enter_top_y
    # and back up through the same gap (holding UP through the rotor
    # spin-up until the rise carries him across the top edge)
    state = state.replace(player_y=jnp.int32(c.flip_enter_top_y))
    for _ in range(c.thrust_spinup + 20):
        _, state, _, _, _ = env.step(state, UP)
        if int(state.room) == 0:
            break
    assert int(state.room) == 0
    assert int(state.player_y) == c.flip_bottom_y


def test_laser_extends_and_kills_spider():
    """The beam grows while fire is held and kills the level-1 spider (+50).
    The hero is lined up on the spider's own body row so the test does not
    depend on where in its bob it happens to be."""
    env = _env()
    c = env.consts
    _, state = env.reset()
    # stand on the floor to the spider's left and fire right along the middle
    # of its bob, so the test does not depend on where in the bob it is
    state = state.replace(room=jnp.int32(1), player_x=jnp.int32(38),
                          player_y=jnp.int32(75), facing=jnp.int32(1))
    score0 = int(state.score)
    killed = False
    for _ in range(30):
        _, state, _, _, _ = env.step(state, RIGHTFIRE)
        if not bool(state.spider_alive[0]):
            killed = True
            break
    assert killed
    assert int(state.score) - score0 == c.creature_points
    assert int(state.lives) == c.starting_lives


def test_laser_bolt_flies_and_relaunches():
    """Measured on the ROM: firing launches an 8x1 bolt on the eye row that
    steps 3 px per frame for 6 frames, then a fresh one starts near the
    helmet again. Releasing the button clears it."""
    from jaxatari.games.jax_hero import _bolt_pos
    env = _env()
    c = env.consts
    _, state = env.reset()
    s = state.replace(player_x=jnp.int32(25), player_y=jnp.int32(75),
                      facing=jnp.int32(1))
    xs = []
    for _ in range(c.laser_bolt_frames + 2):
        _, s, _, _, _ = env.step(s, FIRE)
        bx, by = _bolt_pos(c, s.player_x, s.player_y, s.facing, s.laser_timer)
        assert int(by) == int(s.player_y) + c.laser_eye_offset   # the eye row
        xs.append(int(bx))
    step = c.laser_bolt_speed
    assert xs[:c.laser_bolt_frames] == [xs[0] + i * step
                                        for i in range(c.laser_bolt_frames)]
    assert xs[c.laser_bolt_frames] == xs[0], "a fresh bolt every 6 frames"
    _, s, _, _, _ = env.step(s, NOOP)
    assert int(s.laser_timer) == 0

    # facing left it flies the other way
    s = state.replace(player_x=jnp.int32(100), player_y=jnp.int32(75),
                      facing=jnp.int32(-1))
    _, s, _, _, _ = env.step(s, FIRE)
    bx0, _ = _bolt_pos(c, s.player_x, s.player_y, s.facing, s.laser_timer)
    _, s, _, _, _ = env.step(s, FIRE)
    bx1, _ = _bolt_pos(c, s.player_x, s.player_y, s.facing, s.laser_timer)
    assert int(bx1) == int(bx0) - step


def test_spider_touch_kills_and_respawns_in_room():
    """Death respawns at the top of the CURRENT room (measured)."""
    env = _env()
    c = env.consts
    _, state = env.reset()
    sx = int(c.SPIDER_X[0, 0])
    sy = int(c.SPIDER_Y[0, 0]) + c.spider_body_top
    state = state.replace(room=jnp.int32(1),
                          player_x=jnp.int32(sx), player_y=jnp.int32(sy))
    _, s, _, _, _ = env.step(state, NOOP)
    assert int(s.lives) == c.starting_lives - 1
    assert int(s.room) == 1                    # same room, not the level top
    assert int(s.player_x) == c.respawn_x
    assert int(s.player_y) == c.respawn_y
    assert int(s.power) == c.max_power         # gauge refills


def test_dynamite_lays_with_down_and_fuse_matches():
    """DOWN lays a stick while he is standing on solid ground (measured);
    the fuse then runs dyn_fuse_playable frames."""
    env = _env()
    c = env.consts
    _, state = env.reset()
    state = state.replace(player_y=jnp.int32(75))    # feet on the floor band
    _, s, _, _, _ = env.step(state, DOWN)
    assert bool(s.dyn_active)
    assert int(s.dynamite_count) == c.starting_dynamite - 1
    assert int(s.dyn_fuse) == c.dyn_fuse_playable - 1
    for _ in range(c.dyn_fuse_playable):
        _, s, _, _, _ = env.step(s, NOOP)
        if not bool(s.dyn_active):
            break
    assert not bool(s.dyn_active)              # exploded at ~26 frames
    assert int(s.step_counter) <= c.dyn_fuse_playable + 1


def test_dynamite_breaks_level2_wall_and_scores():
    """Level 2 room 1's pillar is dynamite-destructible (ROM-measured).

    It stands at x 52-59, NOT at x 60-67. This test used to say 60, and to
    say that room 0's, room 1's and room 3's pillar were one wall drawn three
    times, because the first capture read a room by writing the room index
    into RAM and so kept room 0's corridor for every room
    (level_images/CORRECTIONS.md). The ROM gives all four rooms a different
    corridor, so nothing in this level is shared and each wall pays its own
    75.
    """
    env = _env()
    c = env.consts
    _, state = env.reset()
    state = state.replace(
        level=jnp.int32(1), room=jnp.int32(1),
        spider_alive=jnp.zeros_like(state.spider_alive),
        player_x=jnp.int32(46), player_y=jnp.int32(75),   # flush against it
    )
    _, state, _, _, _ = env.step(state, DOWN)
    assert bool(state.dyn_active)
    score0 = int(state.score)
    for _ in range(c.dyn_fuse_playable + 6):
        _, state, _, _, _ = env.step(state, LEFT)   # flee the blast
    zones = HL.DESTRUCTIBLE[1]
    slot = next(i for i, z in enumerate(zones) if z[0] == 1 and z[1] == 52)
    assert int(state.wall_stage[slot]) == 2         # wall gone
    assert int(state.score) - score0 == c.wall_points
    assert int(state.lives) == c.starting_lives     # fled in time
    assert HL.SHARED_WALLS[1] == [], "level 2 draws four different corridors"


def test_dynamite_breaks_opening_pillar_and_opens_passage():
    """The opening room's central pillar is destroyed by a dynamite blast
    (project decision: dynamite breaks walls) — the way down through each
    level. Scores +75 and opens the passage the descent shaft sits behind."""
    env = _env()
    c = env.consts
    _, state = env.reset()
    # stand against the pillar (blocked at x=54) and plant a stick: the
    # measured blast only reaches ~5 px past the hero's own edge
    state = state.replace(player_x=jnp.int32(54), player_y=jnp.int32(75))
    score0 = int(state.score)
    _, state, _, _, _ = env.step(state, DOWN)
    assert bool(state.dyn_active)
    for _ in range(c.dyn_fuse_playable + 6):
        _, state, _, _, _ = env.step(state, LEFT)   # flee the blast
    assert int(state.wall_stage[0]) == 2            # pillar gone
    assert int(state.score) - score0 == c.wall_points
    assert int(state.lives) == c.starting_lives     # fled in time
    # the passage is now open: walking right crosses the old pillar span
    state = state.replace(player_x=jnp.int32(48), player_y=jnp.int32(75))
    for _ in range(40):
        _, state, _, _, _ = env.step(state, RIGHT)
    assert int(state.player_x) > 68


def test_fleeing_the_blast_survives_with_human_reaction_delay():
    """Regression: plant against the pillar, hesitate a few frames (human
    reaction time), then run — the hero must clear the blast and survive."""
    env = _env()
    c = env.consts
    _, state = env.reset()
    state = state.replace(player_x=jnp.int32(54), player_y=jnp.int32(75))
    _, state, _, _, _ = env.step(state, DOWN)
    for _ in range(8):                              # ~1/4 s of hesitation
        _, state, _, _, _ = env.step(state, NOOP)
    for _ in range(c.dyn_fuse_playable + 6):
        _, state, _, _, _ = env.step(state, LEFT)
    assert int(state.wall_stage[0]) == 2            # pillar destroyed
    assert int(state.lives) == c.starting_lives     # and the hero lived


def test_downleft_plants_and_moves():
    """DOWN+LEFT both lays the stick and keeps the hero running (plant-and-
    flee as one held input; previously this mapped to nothing)."""
    env = _env()
    c = env.consts
    _, state = env.reset()
    state = state.replace(player_x=jnp.int32(48), player_y=jnp.int32(75))
    x0 = int(state.player_x)
    _, s, _, _, _ = env.step(state, DOWNLEFT)
    assert bool(s.dyn_active)                       # stick planted
    assert int(s.dynamite_count) == c.starting_dynamite - 1
    assert int(s.player_x) == x0 - 1                # and still moving


def test_touching_a_creature_kills_the_player_even_while_firing():
    """Measured on the ROM by walking into level 1 room 1's spider: holding
    fire does NOT save him. Not firing, he stops dead against it and loses a
    life 107 frames later (the death freeze); firing, the bolt kills the
    spider while he is still ~20 px away, never on the touch. An earlier
    build let a touch-while-firing kill the creature instead.
    Uses L4 room 6's guard spider under the right entry shaft. The ROM puts
    it at (132, 70) and the shaft at x 132-139; the superseded capture had
    the spider a pixel higher."""
    env = _env()
    c = env.consts
    slot = next(i for i in range(c.num_spiders)
                if bool(c.SPIDER_VALID[3, i])
                and int(c.SPIDER_X[3, i]) == 132 and int(c.SPIDER_Y[3, i]) == 70)
    _, state = env.reset()
    base = state.replace(level=jnp.int32(3), room=jnp.int32(6),
                         player_x=jnp.int32(133), player_y=jnp.int32(40),
                         spider_alive=c.SPIDER_VALID[3])
    for action in (FIRE, NOOP):
        s = base
        for _ in range(50):
            _, s, _, _, _ = env.step(s, action)
            if int(s.lives) < c.starting_lives:
                break
        assert int(s.lives) == c.starting_lives - 1, (
            "dropping onto a creature costs a life whether or not fire is held")
        assert bool(s.spider_alive[slot]), "the touch must not kill the creature"


def test_laser_does_not_break_walls():
    """The laser is for enemies only; sustained fire never harms a wall."""
    env = _env()
    _, state = env.reset()
    # stand against the opening pillar (blocked at x=54) and hold fire
    state = state.replace(player_x=jnp.int32(48), player_y=jnp.int32(75))
    for _ in range(80):
        _, state, _, _, _ = env.step(state, FIRE)
    assert int(state.wall_stage[0]) == 0            # pillar untouched
    # still blocked: cannot walk through the intact pillar
    for _ in range(30):
        _, state, _, _, _ = env.step(state, RIGHT)
    assert int(state.player_x) < 60


def test_dynamite_blast_kills_spider():
    env = _env()
    c = env.consts
    _, state = env.reset()
    sx = int(c.SPIDER_X[0, 0])
    sy = int(c.SPIDER_Y[0, 0])
    state = state.replace(
        room=jnp.int32(1),
        player_x=jnp.int32(110), player_y=jnp.int32(60),
        dyn_active=jnp.bool_(True), dyn_fuse=jnp.int32(1),
        dyn_x=jnp.int32(sx), dyn_y=jnp.int32(sy + 6), dyn_room=jnp.int32(1),
        # room 1's pillar is within the blast's reach of the spider; take it
        # out of the picture so the reward is the creature and nothing else
        wall_stage=jnp.full_like(state.wall_stage, 2),
    )
    _, s, reward, _, _ = env.step(state, NOOP)
    assert not bool(s.spider_alive[0])
    assert float(reward) == float(c.creature_points)
    assert int(s.lives) == c.starting_lives    # player clear of the blast


def test_player_in_blast_radius_dies():
    env = _env()
    c = env.consts
    _, state = env.reset()
    state = state.replace(player_x=jnp.int32(33), player_y=jnp.int32(75))
    _, state, _, _, _ = env.step(state, DOWN)
    for _ in range(c.dyn_fuse_playable + 4):
        _, state, _, _, _ = env.step(state, NOOP)
    assert int(state.lives) < c.starting_lives


def test_power_depletion_kills_player():
    env = _env()
    c = env.consts
    _, state = env.reset()
    state = state.replace(power=jnp.int32(2), has_moved=jnp.bool_(True))
    _, state, _, _, _ = env.step(state, RIGHT)
    _, state, _, _, _ = env.step(state, RIGHT)
    assert int(state.lives) == c.starting_lives - 1
    assert int(state.power) == c.max_power


def test_rescuing_miner_advances_level_with_power_bonus():
    """Rescue pays 1000 + remaining power, then loads the next level."""
    env = _env()
    c = env.consts
    _, state = env.reset()
    m = c.LEVEL_MINER[0]
    state = state.replace(room=m[0], player_x=m[1], player_y=m[2],
                          spider_alive=jnp.zeros_like(state.spider_alive))
    power0 = int(state.power)
    _, s, reward, done, _ = env.step(state, NOOP)
    assert int(s.level) == 1
    assert int(s.room) == 0
    assert not bool(done)
    assert not bool(s.miner_rescued)           # next level's miner still trapped
    # the tally pays 20 points per power-bar pixel still lit (measured)
    bonus = (power0 // c.power_frames_per_pixel) * c.bonus_per_power_pixel
    assert float(reward) == float(c.miner_points + bonus)
    assert int(s.power) == c.max_power
    assert int(s.dynamite_count) == c.starting_dynamite
    assert bool((s.spider_alive == c.SPIDER_VALID[1]).all())


def test_completing_last_level_ends_game():
    env = _env()
    c = env.consts
    _, state = env.reset()
    last = c.num_levels - 1
    m = c.LEVEL_MINER[last]
    state = state.replace(level=jnp.int32(last), room=m[0],
                          player_x=m[1], player_y=m[2],
                          spider_alive=jnp.zeros_like(state.spider_alive))
    _, s, _, done, _ = env.step(state, NOOP)
    assert bool(s.level_complete)
    assert bool(s.miner_rescued)
    assert bool(done)


def test_game_over_when_out_of_lives():
    env = _env()
    c = env.consts
    _, state = env.reset()
    sx = int(c.SPIDER_X[0, 0])
    sy = int(c.SPIDER_Y[0, 0]) + c.spider_body_top
    state = state.replace(lives=jnp.int32(1), room=jnp.int32(1),
                          player_x=jnp.int32(sx), player_y=jnp.int32(sy))
    _, s, _, done, _ = env.step(state, NOOP)
    assert int(s.lives) == 0
    assert bool(s.game_over)
    assert bool(done)


def test_extra_life_every_20000_points():
    env = _env()
    c = env.consts
    _, state = env.reset()
    m = c.LEVEL_MINER[0]
    state = state.replace(room=m[0], player_x=m[1], player_y=m[2],
                          score=jnp.int32(19000),
                          spider_alive=jnp.zeros_like(state.spider_alive))
    _, s, _, _, _ = env.step(state, NOOP)      # +1000 + power crosses 20000
    assert int(s.lives) == c.starting_lives + 1


def test_spiders_only_active_in_their_room():
    env = _env()
    c = env.consts
    obs, state = env.reset()
    # room 0 has no creatures (measured); the level-1 spider lives in room 1
    assert not bool(obs.spiders.active.any())
    state = state.replace(room=jnp.int32(1))
    obs = env._get_observation(state)
    assert bool(obs.spiders.active.any())


def test_levels_4_to_16_data_present():
    """Twenty levels (16 measured, 17-20 still placeholders): room counts,
    creature 5-tuples, lantern tables, deadly/flare tables, and every
    background blob decodes."""
    from jaxatari.games import hero_levels as HL
    assert HL.NUM_LEVELS == 20
    assert HL.ROOMS_PER_LEVEL == [2, 4, 6, 8, 8, 10, 12, 14, 16, 16,
                                  16, 16, 16, 16, 16, 16, 16, 16, 16, 16]
    for lv in range(HL.NUM_LEVELS):
        assert all(len(t) == 5 for t in HL.SPIDERS[lv])
        assert all(len(t) == 3 for t in HL.LANTERNS[lv])
        assert all(len(t) == 5 for t in HL.DEADLY[lv])
        assert all(len(t) == 7 for t in HL.FLARES[lv])
    # levels 1 and 2 have no lamp at all; the "lamp" their level files list
    # beside the miner is the top pixel of his own bitmap. Levels 3, 4 and 5
    # have real ones, and every one of them measured at y 35 off the ROM -
    # not the y 33 the superseded reference gave every lamp in the game.
    # Level 5's four were the last to be remeasured (2026-09-22); its room 4
    # has none at all, though the entity census listed one at (79, 35): the
    # capture found no lamp yellow anywhere in that room.
    assert HL.LANTERNS[0] == HL.LANTERNS[1] == []
    assert HL.LANTERNS[2] == [(1, 83, 35), (3, 107, 35)]
    assert HL.LANTERNS[3] == [(1, 83, 35), (2, 23, 35), (3, 39, 35),
                              (4, 27, 35), (5, 83, 35), (6, 135, 35)]
    assert HL.LANTERNS[4] == [(1, 83, 35), (3, 131, 35),
                              (5, 131, 35), (6, 83, 35)]
    assert all(16 <= y <= 59 for lv in range(5) for _r, _x, y in HL.LANTERNS[lv])
    # deadly water strips + flare-ups only exist in the deep levels
    # the flooded rooms: level 1 room 1 (water), 2 room 3 (lava), 3 room 5
    # (brown), 4 room 7 (slime), 5 room 7 (water).
    # Level 1's strip is the regenerated measurement - the ROM's surface
    # animates over the wavy floor trim and reaches rows 139-141, one row
    # higher than the first pass recorded.
    assert HL.DEADLY[0] == [(1, 8, 139, 152, 3)]
    # level 2's is the regenerated measurement too: the magenta surface in
    # its miner's room covers the same rows and columns level 1's water does.
    assert HL.DEADLY[1] == [(3, 8, 139, 152, 3)]
    # level 3's came out of the regeneration as well, and it is a CHANGE:
    # the superseded data said the level had no liquid anywhere. The ROM
    # floods the miner's room the same way, in brown (#a26221), and the
    # recorded playthrough shows it - level_03 frame 0078 has #a05e2a on
    # rows 139-141 and pure blue on 130-138.
    assert HL.DEADLY[2] == [(5, 8, 139, 152, 3)]
    # level 4's is the regenerated measurement: rows 139-141 like every other
    # rebuilt level, where the first pass recorded only 140-141.
    assert HL.DEADLY[3] == [(7, 8, 139, 152, 3)]
    assert HL.DEADLY[4] == [(7, 8, 140, 152, 2)]
    # level 6's miner's room is flooded too - the same trim on rows 139-141
    # that levels 1 to 4 have. It came out of the level-6 regeneration; this
    # assertion still said [] from before it.
    assert HL.DEADLY[5] == [(9, 8, 139, 152, 3)]
    # level 7's water room, and its own flooded miner's room. Room 10's rect
    # is all six rows a '~' cell is drawn on, because the top one is the row
    # the hero comes to rest on.
    assert HL.DEADLY[6] == [(10, 8, 136, 152, 6), (11, 8, 139, 152, 3)]
    # No flare-up survives a regeneration: the rows levels 7-13 used to carry
    # were authored on top of the superseded reference, and no capture finds
    # a periodic eruption anywhere. Level 7's three went with its rebuild.
    assert all(HL.FLARES[lv] == [] for lv in range(7))
    for blobs, pal, n in [(HL.BG_RLE_L4, HL.PALETTE_L4, 8),
                          (HL.BG_RLE_L5, HL.PALETTE_L5, 8),
                          (HL.BG_RLE_L6, HL.PALETTE_L6, 10),
                          (HL.BG_RLE_L7, HL.PALETTE_L7, 12),
                          (HL.BG_RLE_L8, HL.PALETTE_L8, 14),
                          (HL.BG_RLE_L9, HL.PALETTE_L9, 16),
                          (HL.BG_RLE_L10, HL.PALETTE_L10, 16),
                          (HL.BG_RLE_L11, HL.PALETTE_L11, 16),
                          (HL.BG_RLE_L12, HL.PALETTE_L12, 16),
                          (HL.BG_RLE_L13, HL.PALETTE_L13, 16),
                          (HL.BG_RLE_L14, HL.PALETTE_L14, 16),
                          (HL.BG_RLE_L15, HL.PALETTE_L15, 16),
                          (HL.BG_RLE_L16, HL.PALETTE_L16, 16)]:
        assert len(blobs) == n
        for b in blobs:
            assert HL.decode_bg(b, pal).shape == (142, 160, 3)
    # Every level's way down starts with the room-0 central pillar. Levels
    # 1-7 have been rebuilt from the ROM and state it on the band grid (rows
    # 16-98, the ceiling and middle cells a stick takes together); levels
    # 8-16 still carry the older y=19 authoring until their turn comes.
    for lv in range(7):
        assert (0, 60, 16, 8, 83, 1) in HL.DESTRUCTIBLE[lv], f"level {lv + 1}"
    for lv in range(7, 16):
        assert (0, 60, 19, 8, 80, 1) in HL.DESTRUCTIBLE[lv]
    for lv in range(16):
        assert HL.MINER_POS[lv][0] < HL.ROOMS_PER_LEVEL[lv]


def test_bat_killed_by_laser():
    """A bat dies to the bolt like a spider (+50).

    Level 4 has exactly one, in room 4, sweeping 22 px about x 75 on row 64
    (CHARACTERS.md). The superseded census put a bat at (120, 67) in room 2
    and no bat in room 4 at all; the creature the ROM draws at x 120 in room
    2 is a bobbing spider.
    """
    env = _env()
    c = env.consts
    _, state = env.reset()
    # Stand in the open left half of the corridor and let the bolt fly. The
    # sweep never brings the bat left of x 64, so x 50 is close enough for
    # the bolt's range and still clear of it.
    state = state.replace(level=jnp.int32(3), room=jnp.int32(4),
                          player_x=jnp.int32(50), player_y=jnp.int32(70),
                          facing=jnp.int32(1), spider_alive=c.SPIDER_VALID[3])
    slot = next(i for i in range(c.num_spiders)
                if bool(c.SPIDER_VALID[3, i])
                and int(c.SPIDER_KIND[3, i]) == 1)
    score0 = int(state.score)
    killed = False
    for _ in range(60):
        _, state, _, _, _ = env.step(state, FIRE)
        if not bool(state.spider_alive[slot]):
            killed = True
            break
    assert killed
    assert int(state.score) - score0 == c.creature_points


def test_magma_sprite_is_laser_proof_and_deadly():
    """Levels 6-16 still carry their magma as kind-2 rows in SPIDERS, from
    the earlier capture pass: a static red block that the laser cannot touch
    and that kills the hero on contact.

    Levels 1-5 were rebuilt from the ROM and model magma as what it is -
    cave, from the '%' cells of the band strings - so they have no kind-2
    rows left. See tests/games/test_hero_level5.py for that model.
    """
    env = _env()
    c = env.consts
    _, state = env.reset()
    lvl, slot = next((lv, i) for lv in range(HL.NUM_LEVELS)
                     for i, row in enumerate(HL.SPIDERS[lv]) if row[4] == 2)
    assert lvl >= 5, "levels 1-5 no longer use the kind-2 stand-in"
    room, mx, my, _patrol, _kind = HL.SPIDERS[lvl][slot]
    base = state.replace(level=jnp.int32(lvl), room=jnp.int32(room),
                         spider_alive=c.SPIDER_VALID[lvl],
                         invuln_timer=jnp.int32(0))

    # the beam does nothing to it, however long it is held
    s = base.replace(player_x=jnp.int32(mx - 24), player_y=jnp.int32(my - 4),
                     facing=jnp.int32(1))
    for _ in range(60):
        _, s, _, _, _ = env.step(s, FIRE)
        s = s.replace(player_x=jnp.int32(mx - 24), player_y=jnp.int32(my - 4),
                      player_vy=jnp.float32(0.0))
    assert bool(s.spider_alive[slot]), "magma survives the beam"

    # and touching it costs a life
    s = base.replace(player_x=jnp.int32(mx), player_y=jnp.int32(my))
    live_x, live_y = env._spider_pos(s)
    s = s.replace(player_x=jnp.int32(int(live_x[slot])),
                  player_y=jnp.int32(int(live_y[slot]) + c.spider_body_top - 2))
    _, s, _, _, _ = env.step(s, NOOP)
    assert int(s.lives) == c.starting_lives - 1


def test_lantern_touch_darkens_room_until_next_level():
    """Touching a lantern darkens its room for the rest of the level
    (measured: brightness collapses; the laser does not affect lanterns).
    Advancing to the next level restores the light."""
    env = _env()
    c = env.consts
    _, state = env.reset()
    # the L4 lantern in room 4 at (28,39)
    li = next(i for i in range(c.num_lanterns)
              if bool(c.LANTERN_VALID[3, i]) and int(c.LANTERN[3, i, 0]) == 4)
    state = state.replace(level=jnp.int32(3), room=jnp.int32(4),
                          player_x=jnp.int32(26), player_y=jnp.int32(24),
                          spider_alive=jnp.zeros_like(state.spider_alive))
    obs = env._get_observation(state)
    assert bool(obs.lanterns.active[li])
    _, s, _, _, _ = env.step(state, NOOP)        # falls onto the lantern
    for _ in range(20):
        if bool(s.room_dark[4]):
            break
        _, s, _, _, _ = env.step(s, NOOP)
    assert bool(s.room_dark[4])                  # room went dark
    assert int(s.lives) == c.starting_lives      # touching does not kill
    obs = env._get_observation(s.replace(room=jnp.int32(4)))
    assert not bool(obs.lanterns.active[0])      # lamp gone
    env.render(s.replace(room=jnp.int32(4)))     # dark room renders
    # rescue the miner -> next level restores light
    m = c.LEVEL_MINER[3]
    s = s.replace(room=m[0], player_x=m[1], player_y=m[2])
    _, s, _, _, _ = env.step(s, NOOP)
    assert int(s.level) == 4
    assert not bool(s.room_dark.any())


def test_l5r1_is_left_through_the_open_left_edge():
    """Level 5 room 1 has no hole in its floor and no blastable wall either.

    It used to be modelled as a floor the hero blasts through, which no
    capture ever supported. Measured on the ROM 2026-09-22: its corridor
    reaches the LEFT edge of the screen as open air, and walking into that
    edge runs the game's own room-transition code - the hero leaves room 1 at
    x 13 and arrives in room 2 at x 148, keeping his height, with RAM 28
    stepping 1 -> 2 on its own.
    """
    from jaxatari.games import hero_levels as HL
    env = _env()
    c = env.consts
    assert (1, -1, 1) in HL.SIDE_EXITS[4], \
        "room 1's left edge is the way on, and it goes DOWN the chain"
    assert [z for z in HL.DESTRUCTIBLE[4] if z[0] == 1] == [], \
        "nothing in room 1 is blastable"
    _, state = env.reset()
    state = state.replace(level=jnp.int32(4), room=jnp.int32(1),
                          player_x=jnp.int32(60), player_y=jnp.int32(75),
                          spider_alive=jnp.zeros_like(state.spider_alive))
    y_before = int(state.player_y)
    for _ in range(90):
        _, state, _, _, _ = env.step(state, LEFT)
        if int(state.room) != 1:
            break
    assert int(state.room) == 2
    assert int(state.player_x) == c.side_enter_left_x
    assert int(state.player_y) == y_before
    assert int(state.lives) == c.starting_lives, \
        "the magma of room 1 is at the RIGHT edge, so walking left is safe"


def test_l5r1_magma_is_at_the_right_edge_not_the_left():
    """The superseded capture put room 1's magma at x 8-15. The ROM draws it
    in the last two cells, x 152-159, and the corridor is open all the way to
    the left edge - which is what makes the side exit reachable."""
    from jaxatari.games import hero_levels as HL
    rects = [m for m in HL.MAGMA[4] if m[0] == 1]
    assert rects == [(1, 152, 60, 8, 39)]


def test_opening_pillar_breakable_on_deep_levels():
    """Every level's opening pillar is dynamite-breakable (convention).

    He plants beside it and RUNS, then stands. He cannot simply hold LEFT for
    the whole fuse any more: from level 6 on, room 0's corridor is magma at
    both ends as well as in the pillar, so walking left for ever walks into
    the magma at x 8-19 - and planting from x 54 puts him inside the lethal
    box of the magma pillar itself before the fuse even starts. Eighteen
    frames of retreat clears the blast on all four levels and stops short of
    the magma on the two that have it.
    """
    env = _env()
    c = env.consts
    for lvl in (3, 4, 5, 6):
        _, state = env.reset()
        state = state.replace(level=jnp.int32(lvl),
                              player_x=jnp.int32(52), player_y=jnp.int32(75),
                              spider_alive=jnp.zeros_like(state.spider_alive))
        _, state, _, _, _ = env.step(state, DOWN)
        for i in range(c.dyn_fuse_playable + 6):
            _, state, _, _, _ = env.step(state, LEFT if i < 18 else NOOP)
        assert int(state.wall_stage[0]) == 2, f"pillar not broken on level {lvl}"
        assert int(state.lives) == c.starting_lives,             f"and he got clear of it on level {lvl}"


def test_advance_chain_walks_every_level():
    """Rescuing each miner walks the level counter through the whole measured
    set; the last rescue completes the episode. Driven off the level data so
    adding measured levels does not need this test rewritten."""
    env = _env()
    c = env.consts
    n = int(c.num_levels)
    _, state = env.reset()
    for lvl in range(n):
        m = c.LEVEL_MINER[lvl]
        state = state.replace(level=jnp.int32(lvl), room=m[0],
                              player_x=m[1], player_y=m[2],
                              spider_alive=jnp.zeros_like(state.spider_alive),
                              miner_rescued=jnp.bool_(False))
        _, state, _, done, _ = env.step(state, NOOP)
        if lvl < n - 1:
            assert int(state.level) == lvl + 1, f"level {lvl} did not advance"
            assert not bool(done)
        else:
            assert bool(state.level_complete)
            assert bool(done)


def test_water_strip_kills_when_stood_in():
    """Level 9's flooded floors (measured): the water strip is deadly under
    standable columns; the strips are clipped so a fall through a floor gap
    never touches them."""
    env = _env()
    c = env.consts
    from jaxatari.games import hero_levels as HL
    rm, wx, wy, ww, wh = HL.DEADLY[8][0]
    _, state = env.reset()
    s = state.replace(level=jnp.int32(8), room=jnp.int32(rm),
                      player_x=jnp.int32(wx + 1),
                      player_y=jnp.int32(wy + wh - c.player_height + 2),
                      spider_alive=c.SPIDER_VALID[8])
    _, s, _, _, _ = env.step(s, NOOP)
    assert int(s.lives) == c.starting_lives - 1
    assert int(s.power) == c.max_power          # respawn refills power


def test_flare_kills_only_while_its_cycle_is_on():
    """The flare mechanism: deadly during the on-window, harmless while off.

    Checked on level 9, which still carries the authored rows. Level 7's three
    went when it was regenerated - they were invented on top of the superseded
    reference, and the ROM capture finds no periodic eruption in room 6 or
    along room 10's water line. The mechanism is kept and tested because
    levels 9-13 still use it; the next level to be rebuilt will most likely
    empty its rows too.
    """
    env = _env()
    c = env.consts
    from jaxatari.games import hero_levels as HL
    lvl = 8
    rm, fx, fy, fw, fh, period, duty = HL.FLARES[lvl][0]
    _, state = env.reset()
    base = state.replace(level=jnp.int32(lvl), room=jnp.int32(rm),
                         player_x=jnp.int32(fx + 1),
                         player_y=jnp.int32(fy - c.player_height + fh - 1),
                         spider_alive=c.SPIDER_VALID[lvl])
    on = base.replace(step_counter=jnp.int32(0))          # cycle on
    _, s, _, _, _ = env.step(on, NOOP)
    assert int(s.lives) == c.starting_lives - 1
    off = base.replace(step_counter=jnp.int32(duty + 1))  # cycle off
    _, s, _, _, _ = env.step(off, NOOP)
    assert int(s.lives) == c.starting_lives


def test_levels_7_to_10_miners_on_the_measured_ledges():
    """The deep-level miners sit at the classic side positions (measured by
    RAM-teleport room scans): L7 R11, L8 R13, L9 R15, L10 R15."""
    from jaxatari.games import hero_levels as HL
    # level 7's is the ROM capture's, from a real entry into room 11 rather
    # than a RAM teleport: x 128, one pixel left of what the teleport scan
    # reported, and where the recorded playthrough draws him.
    assert HL.MINER_POS[6] == (11, 128, 86)
    assert HL.MINER_POS[7] == (13, 23, 86)
    assert HL.MINER_POS[8] == (15, 129, 86)
    assert HL.MINER_POS[9] == (15, 23, 86)


def test_render_shape_and_jit():
    env = _env()
    _, state = env.reset()
    img = env.render(state)
    assert img.shape == (env.consts.screen_height, env.consts.screen_width, 3)
    assert img.dtype == jnp.uint8


def test_render_all_rooms_all_levels():
    """Every measured room background renders without error."""
    env = _env()
    c = env.consts
    _, state = env.reset()
    for lv in range(c.num_levels):
        for rm in range(int(c.LEVEL_ROOMS[lv])):
            s = state.replace(level=jnp.int32(lv), room=jnp.int32(rm))
            img = env.render(s)
            assert img.shape == (c.screen_height, c.screen_width, 3)
