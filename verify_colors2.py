import warnings
warnings.filterwarnings("ignore")
import ale_py
ale_py.register_v5_envs()
import gymnasium as gym
import numpy as np

# frameskip=4 — Standard ALE
env = gym.make("ALE/StarGunner-v5", render_mode="rgb_array", frameskip=4)
env.reset(seed=0)

for _ in range(30):
    env.step(1)  # FIRE

frame = env.render()
np.save("ale_play_frame.npy", frame)

# Prüfen ob wir in Play sind
region = frame[84:95, 60:92]
text_pixels = (region.sum(axis=2) > 60).sum()
print(f"STAR region non-black pixels: {text_pixels}")
print("Play mode" if text_pixels < 20 else "Still attract")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.imshow(frame)
plt.title("ALE with frameskip=4")
plt.savefig("ale_play_check.png")
print("saved ale_play_check.png")
env.close()
