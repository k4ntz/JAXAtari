import warnings
warnings.filterwarnings("ignore")
import ale_py
ale_py.register_v5_envs()
import gymnasium as gym
import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def is_play_mode(frame):
    region = frame[84:95, 60:92]
    return (region.sum(axis=2) > 60).sum() < 20

env = gym.make("ALE/StarGunner-v5", render_mode="rgb_array", frameskip=1)
env.reset(seed=0)

# Find a frame with the purple humanoid enemy at y=35-55
frame = None
for step in range(5000):
    env.step(1)
    f = env.render()
    if not is_play_mode(f):
        continue
    # Look in y=35-55 for purple pixels
    band = f[35:55, :, :]
    r = band[:,:,0].astype(int)
    g = band[:,:,1].astype(int)
    b = band[:,:,2].astype(int)
    # Purple: high blue, moderate red, low green
    purple = (b > 150) & (r > 90) & (r < 200) & (g < 120)
    if purple.sum() > 10:
        frame = f.copy()
        print(f"Enemy visible at step {step}, purple px = {purple.sum()}")
        break

if frame is None:
    print("Never saw purple enemy")
    env.close()
    raise SystemExit
env.close()

plt.imshow(frame)
plt.title("ALE with purple enemy")
plt.savefig("ale_purple_enemy.png", dpi=120)
print("saved ale_purple_enemy.png")

# Extract the purple cluster
r = frame[:,:,0].astype(int)
g = frame[:,:,1].astype(int)
b = frame[:,:,2].astype(int)
purple = (b > 150) & (r > 90) & (r < 200) & (g < 120)
purple[:30, :] = False  # HUD
purple[80:, :] = False  # only upper area

labeled, n = ndimage.label(purple)
print(f"\nPurple clusters: {n}")
best = None
for i in range(1, n + 1):
    ys, xs = np.where(labeled == i)
    if len(ys) < 5:
        continue
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    h, w = y1 - y0, x1 - x0
    if h > 25 or w > 25 or h < 5 or w < 5:
        continue
    print(f"\nCluster {i}: y={y0}-{y1-1} x={x0}-{x1-1}  size={h}x{w}")
    sub = purple[y0:y1, x0:x1]
    for row in sub:
        print("  " + "".join("X" if v else "." for v in row))
    # Sample actual color
    sample_colors = frame[ys, xs]
    print(f"  Sample colors: {set(tuple(int(v) for v in c) for c in sample_colors[:5])}")

    print("\nAs jnp.array:")
    print("ALE_ENEMY_SPRITE = jnp.array([")
    for row in sub:
        print("    [" + ", ".join("1" if v else "0" for v in row) + "],")
    print("], dtype=jnp.bool_)")
    print()
    best = sub

