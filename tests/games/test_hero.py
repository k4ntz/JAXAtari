"""Tests for the measured H.E.R.O. environment (jax_hero, levels 1-3).

The environment models the ROM's screen-flip world: each level is a stack of
static rooms; geometry, physics and object positions were measured from ALE
captures (see jax_hero.py's provenance notes).
"""
import jax
import jax.numpy as jnp

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
    """The beam grows while fire is held and kills the level-1 spider (+50)."""
    env = _env()
    c = env.consts
    _, state = env.reset()
    state = state.replace(room=jnp.int32(1),
                          player_x=jnp.int32(33), player_y=jnp.int32(82))
    score0 = int(state.score)
    killed = False
    for _ in range(6):
        _, state, _, _, _ = env.step(state, UPFIRE)
        if not bool(state.spider_alive[0]):
            killed = True
            break
    assert killed
    assert int(state.score) - score0 == c.creature_points
    assert int(state.lives) == c.starting_lives


def test_laser_length_grows_and_resets():
    env = _env()
    c = env.consts
    _, state = env.reset()
    _, s, _, _, _ = env.step(state, FIRE)
    assert int(s.laser_len) == c.laser_growth
    _, s, _, _, _ = env.step(s, FIRE)
    assert int(s.laser_len) == 2 * c.laser_growth
    for _ in range(10):
        _, s, _, _, _ = env.step(s, FIRE)
    assert int(s.laser_len) == c.laser_max_length
    _, s, _, _, _ = env.step(s, NOOP)
    assert int(s.laser_len) == 0


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
    """DOWN lays a stick (measured console behaviour), fuse = 26 frames."""
    env = _env()
    c = env.consts
    _, state = env.reset()
    _, s, _, _, _ = env.step(state, DOWN)
    assert bool(s.dyn_active)
    assert int(s.dynamite_count) == c.starting_dynamite - 1
    assert int(s.dyn_fuse) == c.dyn_fuse - 1
    for _ in range(c.dyn_fuse):
        _, s, _, _, _ = env.step(s, NOOP)
        if not bool(s.dyn_active):
            break
    assert not bool(s.dyn_active)              # exploded at ~26 frames
    assert int(s.step_counter) <= c.dyn_fuse + 1


def test_dynamite_breaks_level2_wall_and_scores():
    """Level 2 room 1's pillar foot is dynamite-destructible (ROM-verified)."""
    env = _env()
    c = env.consts
    _, state = env.reset()
    state = state.replace(
        level=jnp.int32(1), room=jnp.int32(1),
        spider_alive=jnp.zeros_like(state.spider_alive),
        player_x=jnp.int32(40), player_y=jnp.int32(74),
    )
    _, state, _, _, _ = env.step(state, DOWN)
    assert bool(state.dyn_active)
    score0 = int(state.score)
    for _ in range(c.dyn_fuse + 6):
        _, state, _, _, _ = env.step(state, LEFT)   # flee the blast
    assert int(state.wall_stage[1]) == 2            # wall gone
    assert int(state.score) - score0 == c.wall_points
    assert int(state.lives) == c.starting_lives     # fled in time


def test_dynamite_breaks_opening_pillar_and_opens_passage():
    """The opening room's central pillar is destroyed by a dynamite blast
    (project decision: dynamite breaks walls) — the way down through each
    level. Scores +75 and opens the passage the descent shaft sits behind."""
    env = _env()
    c = env.consts
    _, state = env.reset()
    # stand against the pillar (blocked at x=54) and plant a stick
    state = state.replace(player_x=jnp.int32(48), player_y=jnp.int32(75))
    score0 = int(state.score)
    _, state, _, _, _ = env.step(state, DOWN)
    assert bool(state.dyn_active)
    for _ in range(c.dyn_fuse + 6):
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
    state = state.replace(player_x=jnp.int32(48), player_y=jnp.int32(75))
    _, state, _, _, _ = env.step(state, DOWN)
    for _ in range(8):                              # ~1/4 s of hesitation
        _, state, _, _, _ = env.step(state, NOOP)
    for _ in range(c.dyn_fuse + 6):
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


def test_firing_through_a_shaft_guard_kills_it_not_the_player():
    """ROM-measured rule: contact with a killable creature while the laser
    is firing kills the CREATURE ("the kill is processed before the touch").
    This is how the shaft guards below descent gaps are cleared — falling
    onto one while holding fire. Without firing, the touch kills the player.
    Uses L4 room 6's guard spider under the right entry shaft (132,69)."""
    env = _env()
    c = env.consts
    slot = next(i for i in range(c.num_spiders)
                if bool(c.SPIDER_VALID[3, i])
                and int(c.SPIDER_X[3, i]) == 132 and int(c.SPIDER_Y[3, i]) == 69)
    _, state = env.reset()
    base = state.replace(level=jnp.int32(3), room=jnp.int32(6),
                         player_x=jnp.int32(133), player_y=jnp.int32(40),
                         spider_alive=c.SPIDER_VALID[3])
    # falling while firing: guard dies, player lives (+50)
    s = base
    score0 = int(s.score)
    for _ in range(50):
        _, s, _, _, _ = env.step(s, FIRE)
        if not bool(s.spider_alive[slot]):
            break
    assert not bool(s.spider_alive[slot])
    assert int(s.lives) == c.starting_lives
    assert int(s.score) - score0 == c.creature_points
    # falling without firing: the guard kills the player
    s = base
    for _ in range(50):
        _, s, _, _, _ = env.step(s, NOOP)
        if int(s.lives) < c.starting_lives:
            break
    assert int(s.lives) == c.starting_lives - 1


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
    for _ in range(c.dyn_fuse + 4):
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
    assert float(reward) == float(c.miner_points + power0)
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


def test_levels_4_to_13_data_present():
    """Thirteen measured levels: room counts, creature 5-tuples, lantern
    tables, deadly/flare tables, and every background blob decodes."""
    from jaxatari.games import hero_levels as HL
    assert HL.NUM_LEVELS == 13
    assert HL.ROOMS_PER_LEVEL == [2, 4, 6, 8, 8, 10, 12, 14, 16, 16,
                                  16, 16, 16]
    for lv in range(13):
        assert all(len(t) == 5 for t in HL.SPIDERS[lv])
        assert all(len(t) == 3 for t in HL.LANTERNS[lv])
        assert all(len(t) == 5 for t in HL.DEADLY[lv])
        assert all(len(t) == 7 for t in HL.FLARES[lv])
    assert HL.LANTERNS[0] == HL.LANTERNS[1] == HL.LANTERNS[2] == []
    # deadly water strips + flare-ups only exist in the deep levels
    assert all(HL.DEADLY[lv] == [] for lv in range(6))
    assert all(HL.FLARES[lv] == [] for lv in range(6))
    for blobs, pal, n in [(HL.BG_RLE_L4, HL.PALETTE_L4, 8),
                          (HL.BG_RLE_L5, HL.PALETTE_L5, 8),
                          (HL.BG_RLE_L6, HL.PALETTE_L6, 10),
                          (HL.BG_RLE_L7, HL.PALETTE_L7, 12),
                          (HL.BG_RLE_L8, HL.PALETTE_L8, 14),
                          (HL.BG_RLE_L9, HL.PALETTE_L9, 16),
                          (HL.BG_RLE_L10, HL.PALETTE_L10, 16),
                          (HL.BG_RLE_L11, HL.PALETTE_L11, 16),
                          (HL.BG_RLE_L12, HL.PALETTE_L12, 16),
                          (HL.BG_RLE_L13, HL.PALETTE_L13, 16)]:
        assert len(blobs) == n
        for b in blobs:
            assert HL.decode_bg(b, pal).shape == (142, 160, 3)
    # every level's way down starts with the room-0 central pillar
    for lv in range(13):
        assert (0, 60, 19, 8, 80, 1) in HL.DESTRUCTIBLE[lv]
        assert HL.MINER_POS[lv][0] < HL.ROOMS_PER_LEVEL[lv]


def test_bat_killed_by_laser():
    """The level-4 bats die to the beam like spiders (+50). Bat 0 lives in
    L4 room 2 at (120,68) with no patrol."""
    env = _env()
    c = env.consts
    _, state = env.reset()
    state = state.replace(level=jnp.int32(3), room=jnp.int32(2),
                          player_x=jnp.int32(100), player_y=jnp.int32(76),
                          spider_alive=c.SPIDER_VALID[3])
    score0 = int(state.score)
    killed = False
    for _ in range(8):
        _, state, _, _, _ = env.step(state, FIRE)
        if not bool(state.spider_alive[0]):
            killed = True
            break
    assert killed
    assert int(state.score) - score0 == c.creature_points


def test_torch_is_laser_proof_and_deadly():
    """The dark level-5 rooms hold torches: the laser never kills one, and
    touching it costs a life (both measured on the ROM)."""
    env = _env()
    c = env.consts
    _, state = env.reset()
    # torch slot 9 of level 5 (index 4): room 5 at (76,71)
    assert int(c.SPIDER_KIND[4, 9]) == 2
    base = state.replace(level=jnp.int32(4), room=jnp.int32(5),
                         spider_alive=c.SPIDER_VALID[4])
    s = base.replace(player_x=jnp.int32(56), player_y=jnp.int32(78))
    for _ in range(10):
        _, s, _, _, _ = env.step(s, FIRE)
    assert bool(s.spider_alive[9])              # torch survives the beam
    s = base.replace(player_x=jnp.int32(74), player_y=jnp.int32(72))
    _, s, _, _, _ = env.step(s, NOOP)
    assert int(s.lives) == c.starting_lives - 1  # touch kills


def test_lantern_touch_darkens_room_until_next_level():
    """Touching a lantern darkens its room for the rest of the level
    (measured: brightness collapses; the laser does not affect lanterns).
    Advancing to the next level restores the light."""
    env = _env()
    c = env.consts
    _, state = env.reset()
    # L4 lantern 0: room 4 at (28,39)
    state = state.replace(level=jnp.int32(3), room=jnp.int32(4),
                          player_x=jnp.int32(26), player_y=jnp.int32(24),
                          spider_alive=jnp.zeros_like(state.spider_alive))
    obs = env._get_observation(state)
    assert bool(obs.lanterns.active[0])
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


def test_l5r1_floor_blast_opens_the_way_down():
    """Level 5 room 1 has no floor gap (measured): dynamite blasts a hole
    through the floor band and the player falls through to room 2."""
    env = _env()
    c = env.consts
    _, state = env.reset()
    state = state.replace(level=jnp.int32(4), room=jnp.int32(1),
                          player_x=jnp.int32(48), player_y=jnp.int32(75),
                          spider_alive=jnp.zeros_like(state.spider_alive))
    _, state, _, _, _ = env.step(state, DOWN)
    assert bool(state.dyn_active)
    for _ in range(c.dyn_fuse + 6):
        _, state, _, _, _ = env.step(state, LEFT)   # flee along the floor
    assert int(state.lives) == c.starting_lives
    assert bool((state.wall_stage >= 2).any())      # floor segment(s) gone
    # walk back over the hole and fall through
    state = state.replace(player_x=jnp.int32(48), player_y=jnp.int32(75))
    for _ in range(80):
        _, state, _, _, _ = env.step(state, NOOP)
        if int(state.room) == 2:
            break
    assert int(state.room) == 2
    # the blast opened the SAME wall as seen from room 2 (the screen flip
    # splits one wall across both rooms): the fall continues through room
    # 2's ceiling band instead of bouncing off it
    for _ in range(70):
        _, state, _, _, _ = env.step(state, NOOP)
    assert int(state.room) == 2
    assert int(state.player_y) > 60


def test_opening_pillar_breakable_on_deep_levels():
    """Every level's opening pillar is dynamite-breakable (convention)."""
    env = _env()
    c = env.consts
    for lvl in (3, 4, 5):
        _, state = env.reset()
        state = state.replace(level=jnp.int32(lvl),
                              player_x=jnp.int32(48), player_y=jnp.int32(75),
                              spider_alive=jnp.zeros_like(state.spider_alive))
        _, state, _, _, _ = env.step(state, DOWN)
        for _ in range(c.dyn_fuse + 6):
            _, state, _, _, _ = env.step(state, LEFT)
        assert int(state.wall_stage[0]) == 2, f"pillar not broken on level {lvl}"


def test_advance_chain_levels_1_to_10():
    """Rescuing each miner walks the level counter 1->10; the last rescue
    completes the episode."""
    env = _env()
    c = env.consts
    _, state = env.reset()
    for lvl in range(10):
        m = c.LEVEL_MINER[lvl]
        state = state.replace(level=jnp.int32(lvl), room=m[0],
                              player_x=m[1], player_y=m[2],
                              spider_alive=jnp.zeros_like(state.spider_alive),
                              miner_rescued=jnp.bool_(False))
        _, state, _, done, _ = env.step(state, NOOP)
        if lvl < 9:
            assert int(state.level) == lvl + 1
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
    """Levels 7-10 gap-mouth flare-ups (measured eruptions, recreated as
    periodic cycles): deadly during the on-window, harmless while off."""
    env = _env()
    c = env.consts
    from jaxatari.games import hero_levels as HL
    rm, fx, fy, fw, fh, period, duty = HL.FLARES[6][0]
    _, state = env.reset()
    base = state.replace(level=jnp.int32(6), room=jnp.int32(rm),
                         player_x=jnp.int32(fx + 1),
                         player_y=jnp.int32(fy - c.player_height + fh - 1),
                         spider_alive=c.SPIDER_VALID[6])
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
    assert HL.MINER_POS[6] == (11, 129, 86)
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
