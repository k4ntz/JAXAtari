"""Tests for the H.E.R.O. Level 1 environment (jax_hero)."""
import jax
import jax.numpy as jnp

from jaxatari.games.jax_hero import JaxHero

# Compact action indices (see JaxHero.ACTION_SET).
NOOP, FIRE, UP, RIGHT, LEFT, DOWN = 0, 1, 2, 3, 4, 5
UPRIGHT, UPLEFT, RIGHTFIRE, LEFTFIRE, UPFIRE, DOWNFIRE = 6, 7, 8, 9, 10, 11


def _env():
    return JaxHero()


def test_reset_starting_inventory():
    env = _env()
    c = env.consts
    obs, state = env.reset(jax.random.PRNGKey(0))
    assert int(state.lives) == c.starting_lives
    assert int(state.dynamite_count) == c.starting_dynamite
    assert int(state.power) == c.max_power
    # Level 1's valid spiders / walls start alive / intact (padded slots for the
    # taller levels stay inactive).
    assert bool((state.spider_alive == c.LEVEL_SPIDER_VALID[0]).all())
    assert bool((state.wall_active == c.LEVEL_WALL_VALID[0]).all())
    assert int(state.level) == 0
    assert not bool(state.game_over)
    assert not bool(state.level_complete)


def test_power_only_drains_after_first_move():
    env = _env()
    _, state = env.reset(jax.random.PRNGKey(0))
    # NOOP should not drain power before the first movement.
    _, s, _, _, _ = env.step(state, NOOP)
    assert int(s.power) == int(state.power)
    # A movement action starts the drain.
    _, s, _, _, _ = env.step(s, RIGHT)
    _, s, _, _, _ = env.step(s, NOOP)
    assert int(s.power) < int(state.power)


def test_walls_block_movement():
    env = _env()
    _, state = env.reset()
    # Drop the player over the solid LEFT part of the upper middle band (band
    # top y=118) and let gravity settle him; he must rest on it, never through.
    state = state.replace(player_x=jnp.int32(40), player_y=jnp.int32(60))
    for _ in range(60):
        _, state, _, _, _ = env.step(state, NOOP)
    assert int(state.player_y) + env.consts.player_height <= 118


def test_laser_kills_spider_and_scores():
    env = _env()
    c = env.consts
    _, state = env.reset()
    # Stand left of spider 0, aligned to its row, and fire to the right.
    state = state.replace(
        player_x=jnp.int32(int(c.SPIDER_X[0]) - 16),
        player_y=jnp.int32(int(c.SPIDER_START_Y[0]) - 3),
        facing=jnp.int32(1),
    )
    _, s, reward, _, _ = env.step(state, RIGHTFIRE)
    assert not bool(s.spider_alive[0])
    assert float(reward) == float(c.spider_points)


def test_player_touching_spider_dies_and_respawns():
    env = _env()
    c = env.consts
    _, state = env.reset()
    state = state.replace(
        player_x=jnp.int32(int(c.SPIDER_X[0])),
        player_y=jnp.int32(int(c.SPIDER_START_Y[0])),
    )
    _, s, _, _, _ = env.step(state, NOOP)
    assert int(s.lives) == c.starting_lives - 1
    # Respawned at the top.
    assert int(s.player_x) == c.player_start_x
    assert int(s.player_y) == c.player_start_y


def test_dynamite_breaks_breakable_wall_and_scores():
    env = _env()
    c = env.consts
    _, state = env.reset()
    # Stand on the band just left of the blastable pillar foot (x60..74), lay
    # dynamite, then flee left along the band, clear of the blast.
    state = state.replace(player_x=jnp.int32(52), player_y=jnp.int32(105))
    _, state, _, _, _ = env.step(state, DOWNFIRE)
    assert bool(state.dyn_active)
    assert int(state.dynamite_count) == c.starting_dynamite - 1

    score_before = int(state.score)
    # Flee left (clear of the blast and the lower-chamber hazards) and outlast
    # the fuse + explosion.
    for _ in range(67):
        _, state, _, _, _ = env.step(state, LEFT)
    assert not bool(state.wall_active[c.breakable_idx])
    assert int(state.score) - score_before == c.wall_points
    # Having fled the blast radius, the player should have survived.
    assert int(state.lives) == c.starting_lives


def test_player_in_blast_radius_dies():
    env = _env()
    c = env.consts
    _, state = env.reset()
    # Lay dynamite and sit still right on top of it (on the top-chamber ledge).
    state = state.replace(player_x=jnp.int32(20), player_y=jnp.int32(107))
    _, state, _, _, _ = env.step(state, DOWNFIRE)
    for _ in range(80):
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
    assert int(state.power) == c.max_power  # refilled on respawn


def test_rescuing_miner_advances_to_next_level():
    """Rescuing the miner on a non-final level advances to the next level
    (score + lives carry over), and does NOT end the episode."""
    env = _env()
    c = env.consts
    _, state = env.reset()
    state = state.replace(
        player_x=jnp.int32(int(c.miner_x)),
        player_y=jnp.int32(int(c.miner_y)),
        power=jnp.int32(c.max_power),
    )
    _, s, reward, done, _ = env.step(state, NOOP)
    assert int(s.level) == 1                 # advanced to level 2
    assert not bool(s.level_complete)        # game continues
    assert not bool(done)
    assert not bool(s.miner_rescued)         # the next level's miner is still trapped
    assert float(reward) == 100.0            # full power converts to 100 points
    # Fresh run state loaded from the new level.
    assert int(s.power) == c.max_power
    assert int(s.dynamite_count) == c.starting_dynamite


def _level2_state(env):
    """A fresh state placed on Level 2 with that level's spiders/bats/walls
    loaded (spiders disabled so bat tests are isolated)."""
    c = env.consts
    _, state = env.reset()
    return state.replace(
        level=jnp.int32(1),
        spider_alive=jnp.zeros_like(state.spider_alive),
        bat_x=c.LEVEL_BAT_START_X[1].astype(jnp.int32),
        bat_dir=c.LEVEL_BAT_START_DIR[1].astype(jnp.int32),
        bat_alive=c.LEVEL_BAT_VALID[1],
        wall_active=c.LEVEL_WALL_VALID[1],
    )


def test_laser_kills_bat_and_scores():
    env = _env()
    c = env.consts
    state = _level2_state(env)
    bx = int(c.LEVEL_BAT_START_X[1, 0])
    by = int(c.LEVEL_BAT_Y[1, 0])
    # Stand left of bat 0, aligned to its row, facing right, and fire.
    state = state.replace(
        player_x=jnp.int32(bx - 16),
        player_y=jnp.int32(by - 4),
        facing=jnp.int32(1),
    )
    _, s, reward, _, _ = env.step(state, RIGHTFIRE)
    assert not bool(s.bat_alive[0])
    assert float(reward) == float(c.bat_points)


def test_player_touching_bat_dies():
    env = _env()
    c = env.consts
    state = _level2_state(env)
    state = state.replace(
        player_x=jnp.int32(int(c.LEVEL_BAT_START_X[1, 0])),
        player_y=jnp.int32(int(c.LEVEL_BAT_Y[1, 0])),
    )
    _, s, _, _, _ = env.step(state, NOOP)
    assert int(s.lives) == c.starting_lives - 1


def test_completing_last_level_ends_game():
    """Rescuing the miner on the LAST level finishes the game."""
    env = _env()
    c = env.consts
    _, state = env.reset()
    last = c.num_levels - 1
    state = state.replace(
        level=jnp.int32(last),
        player_x=jnp.int32(int(c.LEVEL_MINER[last, 0])),
        player_y=jnp.int32(int(c.LEVEL_MINER[last, 1])),
        power=jnp.int32(c.max_power),
    )
    _, s, reward, done, _ = env.step(state, NOOP)
    assert bool(s.level_complete)
    assert bool(s.miner_rescued)
    assert bool(done)
    assert float(reward) == 100.0


def test_game_over_when_out_of_lives():
    env = _env()
    c = env.consts
    _, state = env.reset()
    state = state.replace(lives=jnp.int32(1))
    # Walk into a spider to lose the last life.
    state = state.replace(
        player_x=jnp.int32(int(c.SPIDER_X[0])),
        player_y=jnp.int32(int(c.SPIDER_START_Y[0])),
    )
    _, s, _, done, _ = env.step(state, NOOP)
    assert int(s.lives) == 0
    assert bool(s.game_over)
    assert bool(done)


def test_facing_follows_horizontal_direction():
    env = _env()
    _, state = env.reset()
    # Start mid-air in a clear column so movement isn't blocked by walls.
    state = state.replace(player_x=jnp.int32(30), player_y=jnp.int32(60))
    _, s, _, _, _ = env.step(state, RIGHT)
    assert int(s.facing) == 1
    _, s, _, _, _ = env.step(s, LEFT)
    assert int(s.facing) == -1


def test_walk_timer_animates_while_moving():
    env = _env()
    _, state = env.reset()
    state = state.replace(player_x=jnp.int32(30), player_y=jnp.int32(60))
    # Walking horizontally advances the animation timer.
    _, s, _, _, _ = env.step(state, RIGHT)
    _, s, _, _, _ = env.step(s, RIGHT)
    assert int(s.walk_timer) >= 2
    # Standing still (no horizontal move) resets it to idle.
    _, s, _, _, _ = env.step(s, NOOP)
    assert int(s.walk_timer) == 0


def _level3_state(env):
    """A fresh state placed on Level 3 with that level's spiders / bats / moths /
    walls loaded (the level that introduces mine moths)."""
    c = env.consts
    _, state = env.reset()
    L = 2
    return state.replace(
        level=jnp.int32(L),
        spider_y=c.LEVEL_SPIDER_START_Y[L].astype(jnp.int32),
        spider_dir=c.LEVEL_SPIDER_START_DIR[L].astype(jnp.int32),
        spider_alive=c.LEVEL_SPIDER_VALID[L],
        bat_x=c.LEVEL_BAT_START_X[L].astype(jnp.int32),
        bat_dir=c.LEVEL_BAT_START_DIR[L].astype(jnp.int32),
        bat_alive=c.LEVEL_BAT_VALID[L],
        moth_x=c.LEVEL_MOTH_START_X[L].astype(jnp.int32),
        moth_dir=c.LEVEL_MOTH_START_DIR[L].astype(jnp.int32),
        moth_alive=c.LEVEL_MOTH_VALID[L],
        wall_active=c.LEVEL_WALL_VALID[L],
    )


def test_laser_kills_moth_and_scores():
    """Level 3's mine moths can be shot with the laser (+points), like bats."""
    env = _env()
    c = env.consts
    state = _level3_state(env)
    L = 2
    mx = int(c.LEVEL_MOTH_START_X[L, 0])
    my = int(c.LEVEL_MOTH_Y[L, 0])
    # Stand left of moth 0, aligned to its row, facing right, and fire.
    state = state.replace(
        player_x=jnp.int32(mx - 16),
        player_y=jnp.int32(my - 4),
        facing=jnp.int32(1),
    )
    _, s, reward, _, _ = env.step(state, RIGHTFIRE)
    assert not bool(s.moth_alive[0])
    assert float(reward) == float(c.moth_points)


def test_player_touching_moth_dies():
    env = _env()
    c = env.consts
    state = _level3_state(env)
    L = 2
    state = state.replace(
        player_x=jnp.int32(int(c.LEVEL_MOTH_START_X[L, 0])),
        player_y=jnp.int32(int(c.LEVEL_MOTH_Y[L, 0])),
    )
    _, s, _, _, _ = env.step(state, NOOP)
    assert int(s.lives) == c.starting_lives - 1


def test_dynamite_blast_kills_spider_and_scores():
    """Dynamite is a valid weapon against creatures, not just the breakable wall:
    a spider caught in the explosion dies and scores (+spider_points)."""
    env = _env()
    c = env.consts
    state = _level3_state(env)
    L = 2
    sx = int(c.LEVEL_SPIDER_X[L, 0])
    sy = int(c.LEVEL_SPIDER_START_Y[L, 0])
    # Park the player safely in the top pocket, with the other creatures cleared,
    # and arm a stick of dynamite one frame from detonation right at spider 0.
    state = state.replace(
        bat_alive=jnp.zeros_like(state.bat_alive),
        moth_alive=jnp.zeros_like(state.moth_alive),
        player_x=jnp.int32(22), player_y=jnp.int32(52),
        dyn_active=jnp.bool_(True), dyn_fuse=jnp.int32(1),
        dyn_x=jnp.int32(sx + 3), dyn_y=jnp.int32(sy),
    )
    score_before = int(state.score)
    _, s, reward, _, _ = env.step(state, NOOP)
    assert not bool(s.spider_alive[0])
    assert int(s.score) - score_before == c.spider_points
    assert float(reward) == float(c.spider_points)
    assert int(s.lives) == c.starting_lives   # the player was clear of the blast


def test_dynamite_blast_kills_moth_and_scores():
    env = _env()
    c = env.consts
    state = _level3_state(env)
    L = 2
    mx = int(c.LEVEL_MOTH_START_X[L, 0])
    my = int(c.LEVEL_MOTH_Y[L, 0])
    state = state.replace(
        spider_alive=jnp.zeros_like(state.spider_alive),
        bat_alive=jnp.zeros_like(state.bat_alive),
        moth_alive=c.LEVEL_MOTH_VALID[L],
        player_x=jnp.int32(22), player_y=jnp.int32(52),
        dyn_active=jnp.bool_(True), dyn_fuse=jnp.int32(1),
        dyn_x=jnp.int32(mx + 3), dyn_y=jnp.int32(my),
    )
    _, s, reward, _, _ = env.step(state, NOOP)
    assert not bool(s.moth_alive[0])
    assert float(reward) == float(c.moth_points)
    assert int(s.lives) == c.starting_lives


def test_render_shape_and_jit():
    env = _env()
    _, state = env.reset()
    img = env.render(state)
    assert img.shape == (env.consts.screen_height, env.consts.screen_width, 3)
    assert img.dtype == jnp.uint8
