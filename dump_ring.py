import warnings
warnings.filterwarnings("ignore")
import ale_py
ale_py.register_v5_envs()
import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def is_play_mode(frame):
    region = frame[84:95, 60:92]
    return (region.sum(axis=2) > 60).sum() < 20

env = gym.make("ALE/StarGunner-v5", render_mode="rgb_array", frameskip=1)
env.reset(seed=0)
for _ in range(60):
    env.step(1)
for _ in range(150):
    env.step(1)

frame = env.render()
env.close()

print("Play mode:", is_play_mode(frame))
print("Frame shape:", frame.shape)

# Crop the region around (24, 69)
y0, y1 = 60, 85
x0, x1 = 10, 45
crop = frame[y0:y1, x0:x1]

# Print non-black pixel positions and colors
print(f"\nNon-black pixels in y={y0}-{y1}, x={x0}-{x1}:")
for yy in range(crop.shape[0]):
    for xx in range(crop.shape[1]):
        c = crop[yy, xx]
        if c.sum() > 30:
            print(f"  ({y0+yy}, {x0+xx}) = {tuple(int(v) for v in c)}")

# Show the crop as image
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(frame)
axes[0].set_title("Full ALE frame")
axes[1].imshow(crop, interpolation='nearest')
axes[1].set_title(f"Crop y={y0}-{y1}, x={x0}-{x1}")
plt.tight_layout()
plt.savefig("ring_crop.png", dpi=120)
print("\nsaved ring_crop.png")
