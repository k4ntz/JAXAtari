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

found = None
for step in range(5000):
    env.step(1)
    f = env.render()
    if not is_play_mode(f):
        continue

    r = f[:,:,0].astype(int)
    g = f[:,:,1].astype(int)
    b = f[:,:,2].astype(int)

    # Ring: white/grey (all channels similar, medium-high)
    grey = (np.abs(r - g) < 25) & (np.abs(g - b) < 25) & (r > 100) & (r < 230)
    grey[:35, :] = False
    grey[170:, :] = False

    # At least 10 grey pixels
    if grey.sum() < 10:
        continue

    labeled, n = ndimage.label(grey)
    for i in range(1, n + 1):
        ys, xs = np.where(labeled == i)
        if len(ys) < 8 or len(ys) > 80:
            continue
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        h, w = y1 - y0, x1 - x0
        if w < 3 or h < 4 or w > 15 or h > 15:
            continue
        # Found a plausible ring
        sub = grey[y0:y1, x0:x1]
        found = (f.copy(), sub, (x0, y0, x1, y1), step)
        print(f"Ring at step {step}, bbox x={x0}-{x1-1}, y={y0}-{y1-1}, size {w}x{h}")
        print("Shape:")
        for row in sub:
            print("  " + "".join("X" if v else "." for v in row))
        break
    if found:
        break

env.close()

if found is None:
    print("No ring found in 5000 frames")
    raise SystemExit

frame, sub, (x0, y0, x1, y1), step = found

# Save crop around the ring
crop = frame[max(0,y0-5):y1+5, max(0,x0-5):x1+5]
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(frame); axes[0].set_title(f"ALE frame (step {step})")
axes[1].imshow(crop, interpolation='nearest'); axes[1].set_title("Ring crop")
plt.tight_layout()
plt.savefig("ring_found.png", dpi=120)
print("\nsaved ring_found.png")

# Print as jnp
print("\nRING_SPRITE = jnp.array([")
for row in sub:
    print("    [" + ", ".join("1" if v else "0" for v in row) + "],")
print("], dtype=jnp.bool_)")

# Sample color
ys, xs = np.where(sub)
full_ys = ys + y0
full_xs = xs + x0
sample = frame[full_ys, full_xs]
median = np.median(sample, axis=0)
print(f"\nMedian ring color: {tuple(int(v) for v in median)}")
