import warnings
warnings.filterwarnings("ignore")
import ale_py
ale_py.register_v5_envs()
import gymnasium as gym
import numpy as np
from scipy import ndimage

def is_play_mode(frame):
    """No STAR text in the middle = play mode."""
    region = frame[84:95, 60:92]
    return (region.sum(axis=2) > 60).sum() < 20

def find_enemies(frame):
    """Find enemy clusters in the play area (excluding HUD, hills, player)."""
    r = frame[:,:,0].astype(int)
    g = frame[:,:,1].astype(int)
    b = frame[:,:,2].astype(int)

    # Enemy colors: red (200,72,72), purple, pink, blue (101,111,228), etc.
    # Look for non-black pixels in play area (y=30-170)
    mask = (r + g + b) > 60
    mask[:30, :] = False   # HUD
    mask[170:, :] = False  # Hills
    # Exclude green (hills)
    green = (g > r) & (g > b) & (g > 80)
    mask = mask & (~green)

    labeled, n = ndimage.label(mask)
    enemies = []
    for i in range(1, n + 1):
        ys, xs = np.where(labeled == i)
        if len(ys) < 8:
            continue
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        h, w = y1 - y0, x1 - x0
        if h > 25 or w > 25 or h < 4 or w < 4:
            continue
        # Get dominant color
        colors = frame[ys, xs]
        dominant = tuple(int(v) for v in np.median(colors, axis=0))
        enemies.append({
            'x': (x0 + x1) / 2,
            'y': (y0 + y1) / 2,
            'w': w,
            'h': h,
            'color': dominant,
        })
    return enemies

# Run ALE
env = gym.make("ALE/StarGunner-v5", render_mode="rgb_array", frameskip=1)
env.reset(seed=0)

# Start the game
for _ in range(60):
    env.step(1)

# Collect data over 500 frames
print("frame | enemy_x | enemy_y | color")
records = []
for step in range(500):
    env.step(1)  # FIRE keeps player alive
    frame = env.render()
    if not is_play_mode(frame):
        continue
    enemies = find_enemies(frame)
    if enemies:
        # Take the first (most prominent) enemy
        e = enemies[0]
        records.append((step, e['x'], e['y'], e['color']))
        if step % 20 == 0:
            print(f"  {step:3d} | {e['x']:6.1f} | {e['y']:6.1f} | {e['color']}")

env.close()

# Analyze movement
print("\n=== Movement Analysis ===")
if len(records) > 10:
    xs = np.array([r[1] for r in records])
    ys = np.array([r[2] for r in records])
    print(f"x range: {xs.min():.1f} to {xs.max():.1f} (span: {xs.max()-xs.min():.1f} px)")
    print(f"y range: {ys.min():.1f} to {ys.max():.1f} (span: {ys.max()-ys.min():.1f} px)")

    # Speed
    dx = np.diff(xs)
    dy = np.diff(ys)
    dx = dx[np.abs(dx) < 20]
    dy = dy[np.abs(dy) < 20]
    print(f"mean dx: {dx.mean():.2f} px/frame")
    print(f"mean dy: {dy.mean():.2f} px/frame")
    print(f"std dx: {dx.std():.2f}")
    print(f"std dy: {dy.std():.2f}")

    # Color changes
    colors = [r[3] for r in records]
    color_changes = sum(1 for i in range(1, len(colors)) if colors[i] != colors[i-1])
    print(f"\n=== Color Analysis ===")
    print(f"total frames: {len(records)}")
    print(f"color changes: {color_changes}")
    if color_changes > 0:
        print(f"average frames between changes: {len(records)/color_changes:.1f}")
    unique_colors = set(colors)
    print(f"unique colors seen: {len(unique_colors)}")
    for c in sorted(unique_colors):
        count = colors.count(c)
        print(f"  {c}: {count} frames")
else:
    print("Not enough data collected")
