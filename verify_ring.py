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

# Track the ring (Enemy) over time
# Ring is a distinct circular/oval shape, gold colored
ring_positions = []
frame_with_ring = None

for step in range(3000):
    env.step(1)
    f = env.render()
    if not is_play_mode(f):
        continue

    r = f[:,:,0].astype(int)
    g = f[:,:,1].astype(int)
    b = f[:,:,2].astype(int)

    # Ring is gold (162,134,56) in some frames, or blue-ish in others
    # But its SHAPE is distinctive: 7x10 oval
    # Search all mid-screen objects and find the ring-shaped one
    mask = (r + g + b) > 60
    mask[:30, :] = False
    mask[170:, :] = False
    green = (g > r) & (g > b) & (g > 80)
    mask = mask & (~green)

    labeled, n = ndimage.label(mask)
    for i in range(1, n + 1):
        ys, xs = np.where(labeled == i)
        if len(ys) < 20:
            continue
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        h, w = y1 - y0, x1 - x0
        # Ring is roughly 7-10 wide and 10-12 tall
        if 6 <= w <= 12 and 8 <= h <= 14:
            # Extract mask
            sub = mask[y0:y1, x0:x1]
            # Check if it looks like a ring (hollow)
            inner_hole = sub[2:-2, 2:-2].sum()
            total = sub.sum()
            if total > 0 and inner_hole < total * 0.5:
                cx = (x0 + x1) / 2
                cy = (y0 + y1) / 2
                ring_positions.append((step, cx, cy, w, h, total))
                if frame_with_ring is None:
                    frame_with_ring = f.copy()
                    print(f"Ring found at step {step}, pos ({cx:.1f}, {cy:.1f}), size {w}x{h}")
                    print("Shape:")
                    for row in sub:
                        print("  " + "".join("X" if v else "." for v in row))

env.close()

# Movement analysis
print(f"\n=== Ring Movement Analysis ===")
print(f"Total detections: {len(ring_positions)}")
if len(ring_positions) > 5:
    xs = np.array([p[1] for p in ring_positions])
    ys = np.array([p[2] for p in ring_positions])
    print(f"x range: {xs.min():.1f} to {xs.max():.1f}  (span {xs.max()-xs.min():.1f} px)")
    print(f"y range: {ys.min():.1f} to {ys.max():.1f}  (span {ys.max()-ys.min():.1f} px)")
    dx = np.diff(xs)
    dy = np.diff(ys)
    print(f"mean dx: {dx.mean():.3f} px/step")
    print(f"mean dy: {dy.mean():.3f} px/step")
    print(f"std dx:  {dx.std():.3f}")
    print(f"std dy:  {dy.std():.3f}")

    # If most dx is same sign, movement is directional
    pos_dx = (dx > 0.5).sum()
    neg_dx = (dx < -0.5).sum()
    still = ((dx >= -0.5) & (dx <= 0.5)).sum()
    print(f"\nDirection counts: positive={pos_dx}, negative={neg_dx}, still={still}")
    if pos_dx > neg_dx * 2:
        print("=> predominantly moves RIGHT")
    elif neg_dx > pos_dx * 2:
        print("=> predominantly moves LEFT")
    else:
        print("=> mixed / random")

# Save screenshot
if frame_with_ring is not None:
    plt.imshow(frame_with_ring)
    plt.title("ALE with ring (Enemy)")
    plt.savefig("ale_ring.png", dpi=120)
    print("\nsaved ale_ring.png")
