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
    assert bool(state.spider_alive.all())
    assert bool(state.wall_active.all())
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
    # Place the player just above the floor (wall at y=148) and push DOWN.
    state = state.replace(player_x=jnp.int32(60), player_y=jnp.int32(134))
    for _ in range(40):
        _, state, _, _, _ = env.step(state, DOWN)
    # Player must rest on the floor, never inside/through it.
    assert int(state.player_y) + env.consts.player_height <= 148


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
    # Stand on top of the breakable cap, lay dynamite, then fly clear of the blast.
    state = state.replace(player_x=jnp.int32(20), player_y=jnp.int32(107))
    _, state, _, _, _ = env.step(state, DOWNFIRE)
    assert bool(state.dyn_active)
    assert int(state.dynamite_count) == c.starting_dynamite - 1

    score_before = int(state.score)
    # Flee horizontally clear of the blast's x-range and outlast the fuse +
    # explosion (~67 frames); stop short of the spider further right.
    for _ in range(67):
        _, state, _, _, _ = env.step(state, RIGHT)
    assert not bool(state.wall_active[c.breakable_idx])
    assert int(state.score) - score_before == c.wall_points
    # Having fled the blast radius, the player should have survived.
    assert int(state.lives) == c.starting_lives


def test_player_in_blast_radius_dies():
    env = _env()
    c = env.consts
    _, state = env.reset()
    # Lay dynamite and sit still right on top of it.
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


def test_touching_miner_completes_level():
    env = _env()
    c = env.consts
    _, state = env.reset()
    state = state.replace(
        player_x=jnp.int32(int(c.miner_x)),
        player_y=jnp.int32(int(c.miner_y)),
        power=jnp.int32(c.max_power),
    )
    _, s, reward, done, _ = env.step(state, NOOP)
    assert bool(s.level_complete)
    assert bool(s.miner_rescued)
    assert bool(done)
    assert float(reward) == 100.0  # full power converts to 100 points


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
    # Start mid-air so movement isn't blocked by walls.
    state = state.replace(player_x=jnp.int32(70), player_y=jnp.int32(58))
    _, s, _, _, _ = env.step(state, RIGHT)
    assert int(s.facing) == 1
    _, s, _, _, _ = env.step(s, LEFT)
    assert int(s.facing) == -1


def test_walk_timer_animates_while_moving():
    env = _env()
    _, state = env.reset()
    state = state.replace(player_x=jnp.int32(70), player_y=jnp.int32(58))
    # Walking horizontally advances the animation timer.
    _, s, _, _, _ = env.step(state, RIGHT)
    _, s, _, _, _ = env.step(s, RIGHT)
    assert int(s.walk_timer) >= 2
    # Standing still (no horizontal move) resets it to idle.
    _, s, _, _, _ = env.step(s, NOOP)
    assert int(s.walk_timer) == 0


def test_render_shape_and_jit():
    env = _env()
    _, state = env.reset()
    img = env.render(state)
    assert img.shape == (env.consts.screen_height, env.consts.screen_width, 3)
    assert img.dtype == jnp.uint8
