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
for _ in range(60):
    env.step(1)

# Find one play frame with several objects
frame = None
for step in range(3000):
    env.step(1)
    f = env.render()
    if not is_play_mode(f):
        continue
    # Count non-black pixels in mid-region
    r = f[:,:,0].astype(int)
    g = f[:,:,1].astype(int)
    b = f[:,:,2].astype(int)
    mask = (r+g+b) > 60
    mask[:30,:] = False
    mask[170:,:] = False
    if mask.sum() > 50:
        frame = f.copy()
        print(f"Good frame at step {step}")
        break

env.close()

if frame is None:
    print("No good frame found")
    raise SystemExit

plt.imshow(frame)
plt.title("ALE frame")
plt.savefig("ale_frame_full.png", dpi=120)
print("saved ale_frame_full.png")

# List ALL clusters
r = frame[:,:,0].astype(int)
g = frame[:,:,1].astype(int)
b = frame[:,:,2].astype(int)
mask = (r+g+b) > 60
mask[:30,:] = False
mask[170:,:] = False
# Remove green (hills)
green = (g > r) & (g > b) & (g > 80)
mask = mask & (~green)

labeled, n = ndimage.label(mask)
print(f"\nAll clusters: {n}\n")
for i in range(1, n+1):
    ys, xs = np.where(labeled == i)
    if len(ys) < 4:
        continue
    y0, y1 = ys.min(), ys.max()+1
    x0, x1 = xs.min(), xs.max()+1
    h, w = y1-y0, x1-x0
    if h > 20 or w > 20:
        continue
    sub = mask[y0:y1, x0:x1]
    # Dominant color
    sample = frame[ys, xs]
    color = tuple(int(v) for v in np.median(sample, axis=0))
    print(f"Cluster {i}: pos=({(x0+x1)//2},{(y0+y1)//2}) size={h}x{w} color={color}")
    for row in sub:
        print("  " + "".join("X" if v else "." for v in row))
    print()
