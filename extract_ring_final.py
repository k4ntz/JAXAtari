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

# Track the ring over 2000 frames
positions = []
first_ring_shape = None
first_ring_frame = None

for step in range(2000):
    env.step(1)
    f = env.render()
    if not is_play_mode(f):
        continue
    r = f[:,:,0].astype(int)
    g = f[:,:,1].astype(int)
    b = f[:,:,2].astype(int)

    # Ring is white/grey: R≈G≈B, high value
    is_white = (np.abs(r - g) < 30) & (np.abs(g - b) < 30) & (r > 100) & (r < 220)
    # Exclude HUD (top) and hills (bottom)
    is_white[:35, :] = False
    is_white[170:, :] = False

    labeled, n = ndimage.label(is_white)
    for i in range(1, n+1):
        ys, xs = np.where(labeled == i)
        if len(ys) < 5:
            continue
        y0, y1 = ys.min(), ys.max()+1
        x0, x1 = xs.min(), xs.max()+1
        h, w = y1-y0, x1-x0
        if 3 <= w <= 12 and 4 <= h <= 12:
            cx = (x0+x1)/2
            cy = (y0+y1)/2
            positions.append((step, cx, cy, w, h))
            if first_ring_shape is None:
                first_ring_shape = is_white[y0:y1, x0:x1]
                first_ring_frame = f.copy()
                # crop wider region for context
                print(f"Ring found at step {step}, pos ({cx:.1f}, {cy:.1f}), size {w}x{h}")
                print("Shape:")
                for row in first_ring_shape:
                    print("  " + "".join("X" if v else "." for v in row))
                print()

env.close()

print(f"\n=== Ring movement analysis ({len(positions)} detections) ===")
if len(positions) > 10:
    xs = np.array([p[1] for p in positions])
    ys = np.array([p[2] for p in positions])
    print(f"x range: {xs.min():.1f} -> {xs.max():.1f}  (span {xs.max()-xs.min():.1f})")
    print(f"y range: {ys.min():.1f} -> {ys.max():.1f}  (span {ys.max()-ys.min():.1f})")
    dx = np.diff(xs)
    dy = np.diff(ys)
    print(f"mean dx: {dx.mean():.3f}  std dx: {dx.std():.3f}")
    print(f"mean dy: {dy.mean():.3f}  std dy: {dy.std():.3f}")
    print(f"right: {(dx>0.5).sum()}  left: {(dx<-0.5).sum()}  still: {((dx>=-0.5)&(dx<=0.5)).sum()}")

if first_ring_frame is not None:
    plt.imshow(first_ring_frame)
    plt.title("ALE with ring (Enemy)")
    plt.savefig("ale_ring_final.png", dpi=120)
    print("\nsaved ale_ring_final.png")

    if first_ring_shape is not None:
        print("\n=== Ring sprite as jnp.bool_ ===")
        print("RING_SPRITE = jnp.array([")
        for row in first_ring_shape:
            print("    [" + ", ".join("1" if v else "0" for v in row) + "],")
        print("], dtype=jnp.bool_)")

        # Dominant color
        ys, xs = np.where(first_ring_shape)
        # use original frame region

