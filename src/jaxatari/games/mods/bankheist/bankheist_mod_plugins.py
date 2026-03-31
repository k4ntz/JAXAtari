import jax
import jax.numpy as jnp
from functools import partial

from jaxatari.modification import JaxAtariInternalModPlugin
from jaxatari.games.jax_bankheist import BankHeistState


class RandomBankSpawnsMod(JaxAtariInternalModPlugin):
    """
    Restores the procedural, fully random bank spawns over all valid map tiles
    instead of using the 16-step deterministic ALE loop.
    """

    @partial(jax.jit, static_argnums=(0,))
    def spawn_banks_fn(
        self, state: BankHeistState, step_random_key: jax.Array
    ) -> BankHeistState:
        # We override the base function and inject the procedural logic using state.spawn_points
        new_bank_spawns = jax.random.randint(
            step_random_key,
            shape=(state.bank_positions.position.shape[0],),
            minval=0,
            maxval=state.spawn_points.shape[0],
        )
        chosen_points = state.spawn_points[new_bank_spawns]

        spawning_mask = state.bank_spawn_timers == 0
        new_bank_positions = jnp.where(
            spawning_mask[:, None], chosen_points, state.bank_positions.position
        )
        new_visibility = jnp.where(
            spawning_mask,
            jnp.array([1, 1, 1]),
            state.bank_positions.visibility,
        )

        new_banks = state.bank_positions.replace(
            position=new_bank_positions, visibility=new_visibility
        )
        return state.replace(bank_positions=new_banks)

