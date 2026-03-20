import jax
import jax.numpy as jnp
from functools import partial
from jaxatari.games.jax_roadrunner import RoadRunnerState
from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin


# --- Internal Mod Plugins (patch methods / override constants) ---

# Example: Override a constant
# class SlowPlayerMod(JaxAtariInternalModPlugin):
#     constants_overrides = {
#         "PLAYER_MOVE_SPEED": 1,
#     }

# Example: Patch a method on the environment
# class CustomEnemyMod(JaxAtariInternalModPlugin):
#     @partial(jax.jit, static_argnums=(0,))
#     def _move_enemy(self, state: RoadRunnerState) -> RoadRunnerState:
#         # Access env constants via self._env.consts
#         # Must return a new state via state._replace(...)
#         return state


# --- Post-Step Mod Plugins (modify state after each step) ---

# Example: Modify state after each step
# class AlwaysZeroScoreMod(JaxAtariPostStepModPlugin):
#     @partial(jax.jit, static_argnums=(0,))
#     def run(self, prev_state: RoadRunnerState, new_state: RoadRunnerState) -> RoadRunnerState:
#         return new_state._replace(score=jnp.array(0, dtype=jnp.int32))

# Example: Modify state after reset
# class CustomStartMod(JaxAtariPostStepModPlugin):
#     @partial(jax.jit, static_argnums=(0,))
#     def after_reset(self, obs, state: RoadRunnerState):
#         state = state._replace(lives=jnp.array(5, dtype=jnp.int32))
#         return obs, state
