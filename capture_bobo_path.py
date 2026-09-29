import warnings
warnings.filterwarnings("ignore")
import ale_py
ale_py.register_v5_envs()
import gymnasium as gym
import numpy as np

env = gym.make("ALE/StarGunner-v5", render_mode="rgb_array", frameskip=1)
env.reset(seed=0)

# Start the game
for _ in range(60):
    env.step(1)

def is_play(frame):
    # No STAR text in the middle = play mode
    region = frame[84:95, 60:92]
    return (region.sum(axis=2) > 60).sum() < 20

def find_bobo(frame):
    """
    Bobo is at y ≈ 30-55. Search that band for gold/blue/purple colors.
    Exclude the score display (y < 28) and ship icons (y = 24-30).
    """
    band = frame[30:55, :, :]
    r = band[:,:,0].astype(int)
    g = band[:,:,1].astype(int)
    b = band[:,:,2].astype(int)

    # Bobo colors: gold (180,122,48), blue (84,92,214), purple (146,70,192)
    is_gold   = (np.abs(r-180)<40) & (np.abs(g-122)<40) & (np.abs(b-48)<40)
    is_blue   = (np.abs(r-84)<40)  & (np.abs(g-92)<40)  & (np.abs(b-214)<40)
    is_purple = (np.abs(r-146)<40) & (np.abs(g-70)<40)  & (np.abs(b-192)<40)

    mask = is_gold | is_blue | is_purple
    ys, xs = np.where(mask)
    if len(xs) < 3:
        return None
    return float(np.median(xs))

# Keep player alive by firing every frame
print("frame | bobo_x")
positions = []
for i in range(300):
    env.step(1)  # FIRE keeps player alive
    frame = env.render()
    if not is_play(frame):
        continue
    x = find_bobo(frame)
    if x is not None:
        positions.append((i, x))
        if i % 5 == 0:
            print(f"  {i:3d} | {x:.1f}")

env.close()

if len(positions) > 5:
    xs = np.array([x for _, x in positions])
    print(f"\nBobo x-range: {xs.min():.1f} to {xs.max():.1f}")
    print(f"Total movement: {xs.max() - xs.min():.1f} pixels")
    print(f"Frames captured: {len(xs)}")
