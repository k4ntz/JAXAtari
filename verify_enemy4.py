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

# Look for a frame where enemies are definitely visible
# Loop longer, and check for multiple distinct red clusters
frame = None
for step in range(5000):
    env.step(1)
    f = env.render()
    if not is_play_mode(f):
        continue
    # Count red pixels in mid-screen area (y=40-170)
    r = f[40:170, :, 0].astype(int)
    g = f[40:170, :, 1].astype(int)
    b = f[40:170, :, 2].astype(int)
    red_mask = (np.abs(r-200)<30) & (np.abs(g-72)<30) & (np.abs(b-72)<30)
    if red_mask.sum() > 20:  # enough red pixels for at least one enemy
        frame = f.copy()
        print(f"Play frame with enemies at step {step}, red px = {red_mask.sum()}")
        break

if frame is None:
    print("Never found frame with enemies")
    env.close()
    raise SystemExit

env.close()

# Save for inspection
plt.imshow(frame)
plt.title("ALE play with enemies")
plt.savefig("ale_play_enemy.png", dpi=120)
print("saved ale_play_enemy.png")

# Find ALL clusters (no color filter)
r = frame[:,:,0].astype(int)
g = frame[:,:,1].astype(int)
b = frame[:,:,2].astype(int)
mask = (r + g + b) > 60
mask[:30, :] = False
mask[180:, :] = False
green = (g > r) & (g > b) & (g > 80)
mask = mask & (~green)

labeled, n = ndimage.label(mask)
print(f"\nFound {n} clusters:")
for i in range(1, n + 1):
    ys, xs = np.where(labeled == i)
    if len(ys) < 4:
        continue
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    h, w = y1 - y0, x1 - x0
    if h > 30 or w > 30:
        continue
    colors = frame[ys, xs]
    dominant = tuple(int(v) for v in np.median(colors, axis=0))
    print(f"\nCluster {i}: y={y0}-{y1-1} x={x0}-{x1-1}  size={h}x{w}  color={dominant}")
    sub = mask[y0:y1, x0:x1]
    for row in sub:
        print("  " + "".join("X" if v else "." for v in row))
