import jax
import jax.numpy as jnp
import jaxatari

key = jax.random.PRNGKey(0)

env = jaxatari.make(
    "tictactoe3d",
    mods_config=["random_static_blockers"]
)

obs, state = env.reset(key)

print(state.board)
print("blockers:", jnp.sum(state.board == env.consts.BLOCKER))