import warnings
warnings.filterwarnings("ignore")
import ale_py
ale_py.register_v5_envs()
import gymnasium as gym
import numpy as np
import jax, jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from jaxatari.games.jax_stargunner import JaxStarGunner

def is_play(f):
    return (f[84:95, 60:92].sum(axis=2) > 60).sum() < 20

# ALE: capture a frame with the ring
env = gym.make("ALE/StarGunner-v5", render_mode="rgb_array", frameskip=1)
env.reset(seed=0)
for _ in range(60): env.step(1)
ale_frame = None
for _ in range(3000):
    env.step(1)
    f = env.render()
    if is_play(f):
        # look for small oval in mid area
        r = f[40:170, :, 0].astype(int)
        g = f[40:170, :, 1].astype(int)
        b = f[40:170, :, 2].astype(int)
        # grey/white ring OR colored ring
        non_black = (r+g+b) > 60
        if non_black.sum() > 20 and non_black.sum() < 200:
            ale_frame = f.copy()
            break
env.close()

# JAX: capture a frame
jenv = JaxStarGunner(start_in_play=True)
obs, s = jenv.reset(jax.random.PRNGKey(0))
for _ in range(50):
    obs, s, r, d, i = jenv.step(s, jnp.array(0, jnp.int32))
jax_frame = np.asarray(jenv.render(s)).astype(np.uint8)

# Save both
fig, axes = plt.subplots(1, 2, figsize=(12, 8))
axes[0].imshow(ale_frame); axes[0].set_title("ALE"); axes[0].axis("off")
axes[1].imshow(jax_frame); axes[1].set_title("JAXAtari"); axes[1].axis("off")
plt.tight_layout()
plt.savefig("enemy_shape_compare.png", dpi=100)
print("saved enemy_shape_compare.png")
