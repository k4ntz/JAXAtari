"""Debugging mods for H.E.R.O.

They exist so one level can be checked without replaying everything before it:

    python scripts/play.py -g hero -l 5 -lifes -granades

* ``start_level_N``      -- every reset() lands on level N (1-based, as the
                            console counts). Play then continues normally:
                            rescuing that level's miner advances to N+1.
* ``unlimited_lives``    -- dying still respawns the player (power and
                            dynamite refill, the usual grace period) but never
                            costs a life, so the game cannot end by death.
* ``unlimited_dynamite`` -- planting a stick never uses one up.

All three are post-step plugins: the base step/reset run untouched and the
plugin patches the resulting state, so the measured physics stay as they are.
"""
from functools import partial

import jax
import jax.numpy as jnp

from jaxatari.games.hero_levels import NUM_LEVELS
from jaxatari.games.jax_hero import HeroState
from jaxatari.modification import JaxAtariPostStepModPlugin


class UnlimitedLivesMod(JaxAtariPostStepModPlugin):
    """Lives never decrease.

    Death is still processed by the base step (respawn at the top of the
    room, power/dynamite refill, invulnerability grace); only the life the
    step subtracted is handed back. Bonus lives from scoring still count.
    """

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: HeroState, new_state: HeroState) -> HeroState:
        lives = jnp.maximum(new_state.lives, prev_state.lives).astype(jnp.int32)
        return new_state.replace(lives=lives)


class UnlimitedDynamiteMod(JaxAtariPostStepModPlugin):
    """Planting a stick never consumes one: the count can only go up."""

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: HeroState, new_state: HeroState) -> HeroState:
        count = jnp.maximum(new_state.dynamite_count, prev_state.dynamite_count).astype(jnp.int32)
        return new_state.replace(dynamite_count=count)


def make_start_level_mod(level_number: int) -> type:
    """Build the ``start_level_<level_number>`` plugin class (1-based level)."""
    if not 1 <= level_number <= NUM_LEVELS:
        raise ValueError(
            f"H.E.R.O. has levels 1..{NUM_LEVELS}; cannot build a start_level mod for {level_number}."
        )
    level_index = level_number - 1

    class StartLevelMod(JaxAtariPostStepModPlugin):
        # Only one starting level can be in force at a time.
        conflicts_with = [f"start_level_{n}" for n in range(1, NUM_LEVELS + 1) if n != level_number]

        @partial(jax.jit, static_argnums=(0,))
        def after_reset(self, obs, state: HeroState):
            c = self._env.consts
            lvl = min(level_index, c.num_levels - 1)
            state = state.replace(
                level=jnp.array(lvl, dtype=jnp.int32),
                spider_alive=c.SPIDER_VALID[lvl],
            )
            return self._env._get_observation(state), state

    StartLevelMod.__name__ = StartLevelMod.__qualname__ = f"StartLevel{level_number}Mod"
    StartLevelMod.__doc__ = (
        f"Every reset() starts on level {level_number} (spawn pose, fresh lives, "
        f"power and dynamite; that level's creatures alive)."
    )
    return StartLevelMod


# start_level_1 .. start_level_<NUM_LEVELS>
START_LEVEL_MODS = {
    f"start_level_{n}": make_start_level_mod(n) for n in range(1, NUM_LEVELS + 1)
}
