import jax, jax.numpy as jnp
import numpy as np
from jaxatari.games.jax_stargunner import JaxStarGunner, ENEMY_SPRITE, ENEMY_W, ENEMY_H

print("=== ENEMY_SPRITE shape ===")
print(f"Size: {ENEMY_SPRITE.shape[0]} x {ENEMY_SPRITE.shape[1]}")
for row in np.array(ENEMY_SPRITE):
    print("  " + "".join("X" if v else "." for v in row))

print(f"\nENEMY_W = {ENEMY_W}")
print(f"ENEMY_H = {ENEMY_H}")

env = JaxStarGunner(start_in_play=True)
obs, s = env.reset(jax.random.PRNGKey(0))

print("\n=== Enemy movement over 100 frames ===")
print("frame | enemy_x | enemy_y | enemy_vx | enemy_vy")
for i in range(100):
    obs, s, r, d, info = env.step(s, jnp.array(0, jnp.int32))
    if i % 10 == 0:
        print(f"  {i:3d} | {float(s.enemy_x[0]):6.1f} | {float(s.enemy_y[0]):6.1f} | "
              f"{float(s.enemy_vx[0]):+6.2f} | {float(s.enemy_vy[0]):+6.2f}")
