import warnings
warnings.filterwarnings("ignore")
import ale_py
ale_py.register_v5_envs()
import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

env = gym.make("ALE/StarGunner-v5", render_mode="rgb_array", frameskip=1)
env.reset(seed=0)
for _ in range(60):
    env.step(1)
frames = []
for _ in range(80):
    env.step(0)
    frames.append(env.render().copy())
env.close()

fig, axes = plt.subplots(2, 4, figsize=(16, 10))
for ax, i in zip(axes.flat, [0, 10, 20, 30, 40, 50, 60, 70]):
    ax.imshow(frames[i])
    ax.set_title(f"Frame {i}")
    ax.axis("off")
plt.tight_layout()
plt.savefig("bobo_movement.png")
print("saved bobo_movement.png")
