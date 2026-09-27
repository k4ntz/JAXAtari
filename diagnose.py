import jax, jax.numpy as jnp
from jaxatari.games.jax_stargunner import JaxStarGunner

env = JaxStarGunner(start_in_play=True)

# random policy über 200 Episoden
def run_episode(rng):
    def step_fn(carry, _):
        state, rng, done = carry
        rng, k = jax.random.split(rng)
        a = jax.random.randint(k, (), 0, 18)
        obs, state, r, done, info = env.step(state, a)
        return (state, rng, done), (r, done, state.score)
    _, (rews, dones, scores) = jax.lax.scan(
        step_fn,
        (env.reset(jax.random.PRNGKey(0))[1], rng, jnp.bool_(False)),
        None, length=5000
    )
    return rews.sum(), scores[-1]

keys = jax.random.split(jax.random.PRNGKey(0), 32)
total_rew, final_score = jax.vmap(run_episode)(keys)
print("random policy — mean total reward:", float(total_rew.mean()))
print("random policy — mean final score: ", float(final_score.mean()))
print("random policy — max final score:  ", int(final_score.max()))