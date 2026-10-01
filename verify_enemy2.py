import warnings
warnings.filterwarnings("ignore")
import ale_py
ale_py.register_v5_envs()
import gymnasium as gym
import numpy as np
from scipy import ndimage

env = gym.make("ALE/StarGunner-v5", render_mode="rgb_array", frameskip=1)
env.reset(seed=0)
for _ in range(60):
    env.step(1)
for _ in range(5):
    env.step(1)
frame = env.render()
env.close()

# Playbereich: y=30 bis 180, alles außer Hills unten
r = frame[:,:,0].astype(int)
g = frame[:,:,1].astype(int)
b = frame[:,:,2].astype(int)

# Maske: nicht schwarz, nicht Hügel (grün), nicht HUD (oben)
mask = (r + g + b) > 60
mask[:30, :] = False   # HUD weg
mask[180:, :] = False  # Hills weg
# Grün weg (Hills)
green = (g > r) & (g > b) & (g > 80)
mask = mask & (~green)

# Verbundene Komponenten
labeled, n = ndimage.label(mask)
print(f"Found {n} distinct clusters in play area:\n")

for i in range(1, n + 1):
    ys, xs = np.where(labeled == i)
    if len(ys) < 3:
        continue
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    h, w = y1 - y0, x1 - x0
    if h > 25 or w > 25:
        continue  # zu groß, kein Gegner

    # Farbe dieses Clusters
    colors = frame[ys, xs]
    dominant = tuple(int(v) for v in np.median(colors, axis=0))

    print(f"Cluster {i}: bbox y={y0}-{y1-1} x={x0}-{x1-1} size={h}x{w} color={dominant}")
    sub = mask[y0:y1, x0:x1]
    for row in sub:
        print("  " + "".join("X" if v else "." for v in row))
    print()
