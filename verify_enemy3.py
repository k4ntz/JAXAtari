import warnings
warnings.filterwarnings("ignore")
import ale_py
ale_py.register_v5_envs()
import gymnasium as gym
import numpy as np
from scipy import ndimage

def is_play_mode(frame):
    """No STAR text at y=84-95 = play mode."""
    region = frame[84:95, 60:92]
    return (region.sum(axis=2) > 60).sum() < 20

env = gym.make("ALE/StarGunner-v5", render_mode="rgb_array", frameskip=1)
env.reset(seed=0)

# FIRE until we're actually in play mode
frame = None
for step in range(3000):
    env.step(1)
    f = env.render()
    if is_play_mode(f):
        # additionally check Bobo/enemy pixels exist
        r = f[:80, :, 0].astype(int)
        g = f[:80, :, 1].astype(int)
        b = f[:80, :, 2].astype(int)
        bobo_px = ((np.abs(r-180)<40) & (np.abs(g-122)<40) & (np.abs(b-48)<40)).sum()
        if bobo_px > 3:
            frame = f.copy()
            print(f"Play frame at step {step}")
            break

if frame is None:
    print("Never reached play mode")
    env.close()
    raise SystemExit

env.close()

# Find all clusters
r = frame[:,:,0].astype(int)
g = frame[:,:,1].astype(int)
b = frame[:,:,2].astype(int)
mask = (r + g + b) > 60
mask[:30, :] = False
mask[180:, :] = False
green = (g > r) & (g > b) & (g > 80)
mask = mask & (~green)

labeled, n = ndimage.label(mask)
print(f"\nFound {n} clusters:\n")
for i in range(1, n + 1):
    ys, xs = np.where(labeled == i)
    if len(ys) < 8:
        continue
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    h, w = y1 - y0, x1 - x0
    if h > 25 or w > 25 or h < 4 or w < 4:
        continue

    colors = frame[ys, xs]
    dominant = tuple(int(v) for v in np.median(colors, axis=0))

    # Only show clusters whose color matches enemy/player/Bobo palette
    is_player = abs(dominant[0]-214)<30 and abs(dominant[1]-92)<30 and abs(dominant[2]-92)<30
    is_enemy  = (abs(dominant[0]-200)<30 and abs(dominant[1]-72)<30 and abs(dominant[2]-72)<30) or \
                (abs(dominant[0]-125)<40 and abs(dominant[1]-48)<40 and abs(dominant[2]-173)<40) or \
                (abs(dominant[0]-184)<40 and abs(dominant[1]-70)<40 and abs(dominant[2]-162)<40)
    is_bobo   = abs(dominant[0]-180)<40 and abs(dominant[1]-122)<40 and abs(dominant[2]-48)<40

    label = "?"
    if is_player: label = "PLAYER"
    elif is_bobo: label = "BOBO"
    elif is_enemy: label = "ENEMY"

    if label != "?":
        print(f"Cluster {i}: {label}  bbox y={y0}-{y1-1} x={x0}-{x1-1}  size={h}x{w}  color={dominant}")
        sub = mask[y0:y1, x0:x1]
        for row in sub:
            print("  " + "".join("X" if v else "." for v in row))
        print()
