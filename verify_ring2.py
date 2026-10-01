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

def is_humanoid(mask):
    """Humanoid has 2 eye pixels in a row 2-3, not hollow."""
    h, w = mask.shape
    if h < 6 or w < 6:
        return False
    # Row with eyes: X.X.X pattern
    eye_row = mask[2] if h > 2 else None
    if eye_row is not None:
        # count alternating pattern
        transitions = ((eye_row[1:] != eye_row[:-1]).sum())
        if transitions >= 3:
            return True
    return False

env = gym.make("ALE/StarGunner-v5", render_mode="rgb_array", frameskip=1)
env.reset(seed=0)
for _ in range(60):
    env.step(1)

ring_positions = []
frame_with_ring = None
ring_shape = None

for step in range(5000):
    env.step(1)
    f = env.render()
    if not is_play_mode(f):
        continue

    r = f[:,:,0].astype(int)
    g = f[:,:,1].astype(int)
    b = f[:,:,2].astype(int)

    mask = (r + g + b) > 60
    mask[:30, :] = False
    mask[170:, :] = False
    green = (g > r) & (g > b) & (g > 80)
    mask = mask & (~green)

    labeled, n = ndimage.label(mask)
    for i in range(1, n + 1):
        ys, xs = np.where(labeled == i)
        if len(ys) < 25:
            continue
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        h, w = y1 - y0, x1 - x0
        # Ring is roughly 7x10 or 9x10
        if not (5 <= w <= 11 and 8 <= h <= 12):
            continue
        sub = mask[y0:y1, x0:x1]
        if is_humanoid(sub):
            continue  # skip Bobo

        # Check hollow
        inner = sub[2:-2, 2:-2]
        if inner.size == 0:
            continue
        hollow_ratio = inner.sum() / sub.sum()
        if hollow_ratio < 0.35:  # hollow
            cx = (x0 + x1) / 2
            cy = (y0 + y1) / 2
            ring_positions.append((step, cx, cy, w, h))
            if frame_with_ring is None:
                frame_with_ring = f.copy()
                ring_shape = sub
                print(f"Ring found at step {step}, pos ({cx:.1f}, {cy:.1f}), size {w}x{h}")
                print("Shape:")
                for row in sub:
                    print("  " + "".join("X" if v else "." for v in row))

env.close()

print(f"\nTotal ring detections: {len(ring_positions)}")
if len(ring_positions) > 10:
    xs = np.array([p[1] for p in ring_positions])
    ys = np.array([p[2] for p in ring_positions])
    print(f"x range: {xs.min():.1f} to {xs.max():.1f}  (span {xs.max()-xs.min():.1f})")
    print(f"y range: {ys.min():.1f} to {ys.max():.1f}  (span {ys.max()-ys.min():.1f})")
    dx = np.diff(xs)
    dy = np.diff(ys)
    print(f"mean dx: {dx.mean():.3f}")
    print(f"mean dy: {dy.mean():.3f}")
    print(f"std dx:  {dx.std():.3f}")
    print(f"std dy:  {dy.std():.3f}")
    pos_dx = (dx > 0.5).sum()
    neg_dx = (dx < -0.5).sum()
    still = ((dx >= -0.5) & (dx <= 0.5)).sum()
    print(f"direction: right={pos_dx}, left={neg_dx}, still={still}")

if frame_with_ring is not None:
    plt.imshow(frame_with_ring)
    plt.title("ALE with ring")
    plt.savefig("ale_ring_real.png", dpi=120)
    print("\nsaved ale_ring_real.png")

    # Print as jnp array
    if ring_shape is not None:
        print("\nALE_RING = jnp.array([")
        for row in ring_shape:
            print("    [" + ", ".join("1" if v else "0" for v in row) + "],")
        print("], dtype=jnp.bool_)")
