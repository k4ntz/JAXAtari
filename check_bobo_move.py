import jax, jax.numpy as jnp
from jaxatari.games.jax_stargunner import JaxStarGunner

env = JaxStarGunner(start_in_play=True)
obs, s = env.reset(jax.random.PRNGKey(0))

print("frame | bobo_x | bobo_vx | bobo_y (BOBO_Y-Konstante)")
for i in range(100):
    obs, s, r, d, info = env.step(s, jnp.array(0, jnp.int32))  # NOOP
    if i % 5 == 0:
        print(f"  {i:3d} | {float(s.bobo_x):6.1f} | {float(s.bobo_vx):+6.2f} | "
              f"{env.consts.BOBO_Y}")
