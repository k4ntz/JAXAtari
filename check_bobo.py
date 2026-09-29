import jax, jax.numpy as jnp
from jaxatari.games.jax_stargunner import JaxStarGunner

env = JaxStarGunner(start_in_play=True)
obs, s = env.reset(jax.random.PRNGKey(0))

print("step | bobo_x | bobo_vx | bombs_active")
last_drop = -1
for i in range(300):
    obs, s, r, d, info = env.step(s, jnp.array(0, jnp.int32))
    n_bombs = int(s.bomb_active.sum())
    if n_bombs > 0 and i - last_drop > 5:
        print(f"{i:4d} | {float(s.bobo_x):6.1f} | {float(s.bobo_vx):+5.2f} | BOMB DROPPED ({n_bombs} active)")
        last_drop = i
    elif i % 20 == 0:
        print(f"{i:4d} | {float(s.bobo_x):6.1f} | {float(s.bobo_vx):+5.2f} | {n_bombs}")

print(f"\nBomb period: {env.consts.BOBO_BOMB_PERIOD} steps")
print(f"Bobo speed:  {env.consts.BOBO_SPEED}")
