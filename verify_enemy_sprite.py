import warnings
warnings.filterwarnings("ignore")
import ale_py
ale_py.register_v5_envs()
import gymnasium as gym
import numpy as np

# === 1. ALE Play-Frame holen ===
env = gym.make("ALE/StarGunner-v5", render_mode="rgb_array", frameskip=1)
env.reset(seed=0)
for _ in range(60):
    env.step(1)
for _ in range(5):
    env.step(1)

frame = env.render()
env.close()

# === 2. Gegner-Sprite isolieren ===
region = frame[30:60, 20:120]
region_mask = (region.sum(axis=2) > 60)
ys, xs = np.where(region_mask)

if len(ys) > 0:
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    ale_enemy = region_mask[y0:y1, x0:x1]
    print("=== ALE enemy sprite (largest cluster) ===")
    print(f"size: {ale_enemy.shape[0]}x{ale_enemy.shape[1]}")
    for row in ale_enemy:
        print("  " + "".join("X" if v else "." for v in row))
    print()
    print("As jnp.bool_ array:")
    print("ALE_SPRITE = jnp.array([")
    for row in ale_enemy:
        print("    [" + ", ".join("1" if v else "0" for v in row) + "],")
    print("], dtype=jnp.bool_)")
else:
    print("No enemy found in region 30:60, 20:120")

# === 3. Dein aktueller Sprite ===
try:
    from jaxatari.games.jax_stargunner import ENEMY_SPRITE
    print("\n=== Your ENEMY_SPRITE ===")
    print(f"size: {ENEMY_SPRITE.shape[0]}x{ENEMY_SPRITE.shape[1]}")
    for row in np.array(ENEMY_SPRITE):
        print("  " + "".join("X" if v else "." for v in row))
except Exception as e:
    print(f"\nCould not import ENEMY_SPRITE: {e}")

# === 4. Farben im Gegner-Bereich ===
print("\n=== Colors in enemy region ===")
crop = frame[30:60, 20:120]
colors, counts = np.unique(crop.reshape(-1,3), axis=0, return_counts=True)
order = np.argsort(-counts)
for i in order[:10]:
    c = tuple(int(v) for v in colors[i])
    if c != (0,0,0):
        print(f"  {c}  n={int(counts[i])}")
