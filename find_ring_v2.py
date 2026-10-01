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

# Keep looking until we find a frame with many different colors
# (i.e. more than just background+hills+player+hud)
best_frame = None
best_score = 0

for step in range(3000):
    env.step(1)
    f = env.render()
    if not is_play_mode(f):
        continue

    # Count unique non-background, non-hill colors
    colors, counts = np.unique(f.reshape(-1,3), axis=0, return_counts=True)
    interesting = 0
    for c, n in zip(colors, counts):
        r, g, b = int(c[0]), int(c[1]), int(c[2])
        # Skip black, greens (hills), player red, HUD blue
        if (r+g+b) < 30: continue
        if g > r and g > b and g > 80: continue  # hills
        if abs(r-214)<10 and abs(g-92)<10 and abs(b-92)<10: continue  # player
        if abs(r-101)<10 and abs(g-160)<10 and abs(b-225)<10: continue  # HUD
        if n >= 3:
            interesting += n

    if interesting > best_score:
        best_score = interesting
        best_frame = f.copy()
        print(f"Step {step}: interesting px = {interesting}")

    if interesting > 30:
        break

env.close()

if best_frame is None:
    print("Never saw enemies/ring in 3000 frames")
    raise SystemExit

print(f"\nBest frame: {best_score} interesting pixels")

# List unique colors in that frame
print("\nColors in best frame:")
colors, counts = np.unique(best_frame.reshape(-1,3), axis=0, return_counts=True)
order = np.argsort(-counts)
for i in order[:15]:
    c = tuple(int(v) for v in colors[i])
    if c == (0,0,0): continue
    print(f"  {c}  n={int(counts[i])}")

# Save
plt.figure(figsize=(6, 8))
plt.imshow(best_frame)
plt.title(f"ALE frame step ~{step}")
plt.savefig("ale_best_frame.png", dpi=120)
print("\nsaved ale_best_frame.png")
