import warnings
warnings.filterwarnings("ignore")

import ale_py
ale_py.register_v5_envs()

import jax, jax.numpy as jnp
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from jaxatari.games.jax_stargunner import JaxStarGunner

# ---------- ALE ----------
ale = gym.make("ALE/StarGunner-v5", render_mode="rgb_array")
ale.reset(seed=0)

# Long sequence: FIRE, then play, then FIRE again.
for _ in range(80):
    ale.step(1)   # FIRE
for _ in range(200):
    ale.step(0)   # NOOP
ale_frame = np.array(ale.render())

# ---------- JAXAtari ----------
env = JaxStarGunner(start_in_play=True)
obs, s = env.reset(jax.random.PRNGKey(0))
for _ in range(80):
    obs, s, r, d, i = env.step(s, jnp.array(1, jnp.int32))
for _ in range(200):
    obs, s, r, d, i = env.step(s, jnp.array(0, jnp.int32))
jax_frame = np.asarray(env.render(s)).astype(np.uint8)

# ---------- plot ----------
fig, ax = plt.subplots(1, 2, figsize=(8, 5))
ax[0].imshow(ale_frame);  ax[0].set_title("ALE StarGunner");     ax[0].axis("off")
ax[1].imshow(jax_frame);  ax[1].set_title("JAXAtari StarGunner"); ax[1].axis("off")
plt.tight_layout()
plt.savefig("parity_play.png", dpi=120)
print("saved parity_play.png")