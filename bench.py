import time
import jax, jax.numpy as jnp
from jaxatari.games.jax_stargunner import JaxStarGunner

def bench(B, steps=1000):
    env = JaxStarGunner(start_in_play=True)
    step = jax.jit(jax.vmap(env.step))
    obs, s = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(0), B))
    a = jnp.zeros(B, jnp.int32)
    for _ in range(5):
        obs, s, r, d, i = step(s, a)
    jax.block_until_ready(s)
    t0 = time.perf_counter()
    for _ in range(steps):
        obs, s, r, d, i = step(s, a)
    jax.block_until_ready(s)
    return B * steps / (time.perf_counter() - t0)

print("devices:", jax.devices())
base = bench(1)
for B in [1, 8, 32, 64, 128, 256]:
    sps = bench(B)
    print(f"B={B:4d}  {sps:>10.1f} steps/sec  ({sps/base:.1f}x vs B=1)")
