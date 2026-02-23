import jax
import jax.numpy as jnp
from functools import partial
from jaxatari.games.jax_gopher import GopherState
from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin


# 1. Lazy Gopher (Simple Type 1 - Constant Override)
class LazyGopherMod(JaxAtariInternalModPlugin):
    constants_overrides = {"GOPHER_SPEED_X": 0.65}

# 2. Fast Farmer (Simple Type 1 - Constant Override)
class FastFarmerMod(JaxAtariInternalModPlugin):
    constants_overrides = {"PLAYER_ACCELERATION": jnp.array([12])}

# 3. Greedy Gopher (Simple Type 1 - Constant Override)
class ShyGopherMod(JaxAtariInternalModPlugin):
    constants_overrides = {"PROB_STEAL_NORMAL": 0.2}

# 4. Pink Gopher (Simple Type 1 - Asset/Visual Override)
class PinkGopherMod(JaxAtariInternalModPlugin):
    constants_overrides = {"ENEMY_COLOR": jnp.array([255, 105, 180], dtype=jnp.uint8)}

# 5. Generous Bird (Simple Type 2 - Post-Step Reward Logic)
class GenerousBirdMod(JaxAtariPostStepModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        # Double the reward whenever the agent gets points
        return new_state._replace(reward=new_state.reward * 2.0)