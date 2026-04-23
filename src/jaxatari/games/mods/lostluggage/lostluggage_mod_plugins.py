import jax
import jax.numpy as jnp
from functools import partial
import jax.numpy as jnp
from functools import partial
from jax import random as jrandom
from jaxatari.modification import (
    JaxAtariInternalModPlugin,
    JaxAtariPostStepModPlugin,
)
from jaxatari.games.jax_lostluggage import LostLuggageState


class LinearMovementMod(JaxAtariInternalModPlugin):
    constants_overrides = {
        "PLAYER_SPEED":  jnp.int32(1),
        "PLAYER_VERT_SPEED": jnp.array(1.0, dtype=jnp.float32),
    }


class AlwaysZeroScoreMod(JaxAtariPostStepModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: LostLuggageState, new_state: LostLuggageState):
        return new_state.replace(score=jnp.int32(0))


class SoftEscapePenaltyMod(JaxAtariPostStepModPlugin):
    PENALTY_PERCENT = jnp.int32(30)

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: LostLuggageState, new_state: LostLuggageState):
        prev_lives = prev_state.lives
        new_lives = new_state.lives
        lives_lost = prev_lives - new_lives

        def no_change(ns):
            return ns

        def apply_soft_penalty(ns):
            # 30% of score, rounded down
            penalty = (ns.score * self.PENALTY_PERCENT) // jnp.int32(100)
            new_score = jnp.maximum(ns.score - penalty, 0)
            return ns.replace(
                lives=prev_lives,
                score=new_score
            )

        return jax.lax.cond(lives_lost > 0, apply_soft_penalty, no_change, new_state)


class NoExtraLifeMod(JaxAtariInternalModPlugin):
    constants_overrides = {
        "EXTRA_LIFE_THRESHOLDS": jnp.array([-1, -1], dtype=jnp.int32),
    }


class MoreSuitcasesMod(JaxAtariInternalModPlugin):
    constants_overrides = {
        "SUITS_PER_ROUND": jnp.int32(50),
        "MAX_ACTIVE_SUITS": jnp.int32(50),
    }


class TimedRoundFastSpawnMod(JaxAtariPostStepModPlugin):
    END_TICKS = jnp.int32(2000)

    constants_overrides = {
        "SPAWN_INTERVAL_START": jnp.array(5.0),
        "SPAWN_DECREASE_EVERY": jnp.array(1.0),
    }

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: LostLuggageState, new_state: LostLuggageState):
        def end_round(ns):
            return ns.replace(round_failed=jnp.bool_(True))

        return jax.lax.cond(new_state.tick >= self.END_TICKS, end_round, lambda ns: ns, new_state)




class LowerPlayerUpperBoundMod(JaxAtariInternalModPlugin):
    constants_overrides = {
        "PLAYER_Y_TOP": jnp.int32(100),
    }



    
class DisappearingSuitcasesMod(JaxAtariInternalModPlugin):
    MAX_Y = jnp.int32(140)  # area of disappearance

    @partial(jax.jit, static_argnums=(0,))
    def _apply_gravity(self, state: LostLuggageState) -> LostLuggageState:
        drop = jnp.int32(16 + state.round_num * 2)

        new_y_fp = jnp.where(
            state.suit_active,
            state.suit_y_fp + drop,
            state.suit_y_fp
        )

        new_x_fp = jnp.where(
            state.suit_active,
            state.suit_x_fp + state.suit_dx_fp,
            state.suit_x_fp
        )

        # Convert fixed-point to pixel
        sy = new_y_fp >> 4
        still_visible = sy < self.MAX_Y
        new_active = state.suit_active & still_visible

        return state.replace(
            suit_x_fp=new_x_fp,
            suit_y_fp=new_y_fp,
            suit_active=new_active
        )

class RandomDisappearSuitcasesMod(JaxAtariInternalModPlugin):
    DISAPPEAR_PROB = 0.005

    @partial(jax.jit, static_argnums=(0,))
    def _apply_gravity(self, state: LostLuggageState) -> LostLuggageState:
        key, subkey = jrandom.split(state.key)

        drop = jnp.int32(16)
        new_y_fp = jnp.where(
            state.suit_active,
            state.suit_y_fp + drop,
            state.suit_y_fp
        )

        rand = jrandom.uniform(subkey, state.suit_active.shape)
        disappear = rand < self.DISAPPEAR_PROB

        new_active = state.suit_active & (~disappear)

        return state.replace(
            suit_y_fp=new_y_fp,
            suit_active=new_active,
            key=key
        )