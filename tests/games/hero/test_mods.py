"""Tests for the H.E.R.O. debugging mods.

``start_level_N`` drops the player straight into level N (1-based, as printed
on the console), ``unlimited_lives`` makes death free and ``unlimited_dynamite``
makes planting free. Together they let a developer check a single level
without replaying everything before it:

    python scripts/play.py -g hero -l 5 -lifes -granades

Constructing a JaxHero (its renderer pre-bakes every room) takes minutes, so
the tests share ONE base environment and wrap it per mod combination. The mods
are post-step plugins and never touch the base env, so sharing is safe. A
single test goes through ``jaxatari.make`` to prove the registration itself.
"""
from functools import lru_cache

import jax
import jax.numpy as jnp
import pytest

from jaxatari.core import MOD_MODULES, make
from jaxatari.games.jax_hero import JaxHero
from jaxatari.games.mods.hero_mods import HeroEnvMod
from jaxatari.modification import JaxAtariModWrapper, _load_from_string

# Compact action indices (see JaxHero.ACTION_SET).
NOOP, FIRE, UP, RIGHT, LEFT, DOWN = 0, 1, 2, 3, 4, 5


@lru_cache(maxsize=None)
def _base_env():
    return JaxHero()


@lru_cache(maxsize=None)
def _modded(*mods):
    """The same Controller -> Wrapper pipeline core.make builds, on the shared base env."""
    mods = list(mods)
    controller = HeroEnvMod(env=_base_env(), mods_config=mods)
    return JaxAtariModWrapper(env=controller, mods_config=mods)


def _registry():
    return _load_from_string(MOD_MODULES["hero"]).REGISTRY


def _arm_power_death(state):
    """One more MOVING frame kills the player (power runs out). The gauge only
    drains while he walks or hovers, so these tests step with RIGHT."""
    return state.replace(power=jnp.int32(1), has_moved=jnp.bool_(True))


def _grounded(state):
    """Put his feet on the floor band: a stick is only laid from solid
    ground (measured)."""
    return state.replace(player_y=jnp.int32(75))


# ---------------------------------------------------------------------------
# start_level_N
# ---------------------------------------------------------------------------
def test_start_level_mod_jumps_to_that_level():
    env = _modded("start_level_5")
    c = env.consts
    obs, state = env.reset(jax.random.PRNGKey(0))
    assert int(state.level) == 4                      # 1-based on the CLI, 0-based inside
    assert int(obs.level) == 4                        # observation rebuilt for the new level
    assert int(state.room) == 0
    assert int(state.player_x) == c.spawn_x
    assert int(state.player_y) == c.spawn_y
    assert bool((state.spider_alive == c.SPIDER_VALID[4]).all())
    assert int(state.lives) == c.starting_lives
    assert int(state.dynamite_count) == c.starting_dynamite


def test_start_level_registry_covers_every_level():
    n = _base_env().consts.num_levels
    keys = _registry().keys()
    for lvl in range(1, n + 1):
        assert f"start_level_{lvl}" in keys
    assert "start_level_0" not in keys
    assert f"start_level_{n + 1}" not in keys


def test_start_level_1_is_the_normal_start():
    _, plain = _base_env().reset(jax.random.PRNGKey(3))
    _, modded = _modded("start_level_1").reset(jax.random.PRNGKey(3))
    assert int(modded.level) == int(plain.level) == 0
    assert bool((modded.spider_alive == plain.spider_alive).all())


def test_two_start_level_mods_conflict():
    with pytest.raises(ValueError):
        _modded("start_level_2", "start_level_3")


def test_play_continues_normally_after_the_jump():
    """Rescuing the miner on the chosen level advances to the next one."""
    env = _modded("start_level_5")
    c = env.consts
    _, state = env.reset(jax.random.PRNGKey(0))
    m = c.LEVEL_MINER[4]
    state = state.replace(room=m[0], player_x=m[1], player_y=m[2],
                          spider_alive=jnp.zeros_like(state.spider_alive))
    _, s, _, done, _ = env.step(state, NOOP)
    assert int(s.level) == 5
    assert not bool(done)


def test_start_level_applies_on_every_reset():
    """Every reset lands on the chosen level, not back on level 1."""
    env = _modded("start_level_5")
    _, s1 = env.reset(jax.random.PRNGKey(1))
    _, s2 = env.reset(jax.random.PRNGKey(2))
    assert int(s1.level) == int(s2.level) == 4


# ---------------------------------------------------------------------------
# unlimited_lives
# ---------------------------------------------------------------------------
def test_unlimited_lives_death_costs_nothing_but_still_respawns():
    env = _modded("unlimited_lives")
    c = env.consts
    _, state = env.reset(jax.random.PRNGKey(0))
    state = _arm_power_death(state)
    _, s, _, done, _ = env.step(state, RIGHT)
    assert int(s.lives) == c.starting_lives          # no life lost
    assert int(s.power) == c.max_power               # ...but the respawn happened
    assert int(s.player_x) in (c.respawn_x, c.spawn_x, 76)
    assert not bool(s.game_over)
    assert not bool(done)


def test_unlimited_lives_never_reaches_game_over():
    env = _modded("unlimited_lives")
    c = env.consts
    _, state = env.reset(jax.random.PRNGKey(0))
    for _ in range(c.starting_lives + 4):
        _, state, _, done, _ = env.step(_arm_power_death(state), NOOP)
        assert not bool(done)
    assert int(state.lives) == c.starting_lives
    assert not bool(state.game_over)


def test_unlimited_lives_still_awards_bonus_lives():
    """The cap is only on losing lives; the 20000-point bonus still counts."""
    env = _modded("unlimited_lives")
    c = env.consts
    _, state = env.reset(jax.random.PRNGKey(0))
    m = c.LEVEL_MINER[0]
    state = state.replace(room=m[0], player_x=m[1], player_y=m[2],
                          score=jnp.int32(19000),
                          spider_alive=jnp.zeros_like(state.spider_alive))
    _, s, _, _, _ = env.step(state, NOOP)
    assert int(s.lives) == c.starting_lives + 1


# ---------------------------------------------------------------------------
# unlimited_dynamite
# ---------------------------------------------------------------------------
def test_unlimited_dynamite_planting_does_not_consume_a_stick():
    env = _modded("unlimited_dynamite")
    c = env.consts
    _, state = env.reset(jax.random.PRNGKey(0))
    _, s, _, _, _ = env.step(_grounded(state), DOWN)
    assert bool(s.dyn_active)                         # the stick was laid
    assert int(s.dynamite_count) == c.starting_dynamite


def test_unlimited_dynamite_outlasts_the_normal_supply():
    env = _modded("unlimited_dynamite")
    c = env.consts
    _, state = env.reset(jax.random.PRNGKey(0))
    state = _grounded(state)
    planted = 0
    for _ in range(c.starting_dynamite + 3):
        _, state, _, _, _ = env.step(_grounded(state), DOWN)
        planted += int(state.dyn_active)
        # wait for the stick to blow and the blast to clear
        for _ in range(200):
            if not bool(state.dyn_active) and int(state.explosion_timer) <= 0:
                break
            _, state, _, _, _ = env.step(state, NOOP)
    assert planted == c.starting_dynamite + 3
    assert int(state.dynamite_count) == c.starting_dynamite


# ---------------------------------------------------------------------------
# all together, through jaxatari.make (what the CLI flags produce)
# ---------------------------------------------------------------------------
def test_all_debug_mods_combine_via_make():
    env = make("hero", mods=["start_level_5", "unlimited_lives", "unlimited_dynamite"])
    c = env.consts
    _, state = env.reset(jax.random.PRNGKey(0))
    assert int(state.level) == 4
    _, state, _, _, _ = env.step(_grounded(state), DOWN)
    assert bool(state.dyn_active)
    assert int(state.dynamite_count) == c.starting_dynamite
    _, state, _, done, _ = env.step(_arm_power_death(state), RIGHT)
    assert int(state.lives) == c.starting_lives
    assert int(state.level) == 4
    assert not bool(done)
