import warnings
warnings.filterwarnings("ignore")
import ale_py
ale_py.register_v5_envs()
import gymnasium as gym
import numpy as np

def is_play_mode(frame):
    """Kein STAR-Text in der Mitte = Play-Modus."""
    region = frame[84:95, 60:92]
    non_black = (region.sum(axis=2) > 60).sum()
    return non_black < 20

env = gym.make("ALE/StarGunner-v5", render_mode="rgb_array", frameskip=1)
env.reset(seed=0)

# FIRE halten, bis wir einen echten Play-Frame sehen
ale_play_frame = None
for step in range(2000):
    env.step(1)  # FIRE
    frame = env.render()
    if is_play_mode(frame):
        # Prüfe, dass Bobo und/oder Gegner sichtbar sind
        # Bobo ist gold (162,134,56) im oberen Bereich
        r = frame[:80, :, 0].astype(int)
        g = frame[:80, :, 1].astype(int)
        b = frame[:80, :, 2].astype(int)
        bobo_pixels = ((np.abs(r-162)<30) & (np.abs(g-134)<30) & (np.abs(b-56)<30)).sum()
        if bobo_pixels > 3:
            print(f"Play-Frame mit Bobo gefunden bei Step {step}")
            ale_play_frame = frame.copy()
            break

env.close()

if ale_play_frame is None:
    print("Kein Play-Frame gefunden — ALE startet nicht mit FIRE allein")
    raise SystemExit

np.save("ale_play_frame.npy", ale_play_frame)

# Farben analysieren
ale_colors = np.unique(ale_play_frame.reshape(-1,3), axis=0)
print(f"\n=== ALE Play-Frame Farben ({len(ale_colors)}) ===")
for c in ale_colors:
    if not np.array_equal(c, [0,0,0]):
        count = ((ale_play_frame == c).all(axis=2)).sum()
        print(f"  {tuple(int(v) for v in c)}  n={count}")

# Bild speichern
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.imshow(ale_play_frame)
plt.title("ALE play mode")
plt.savefig("ale_play_only.png")
print("\nsaved ale_play_only.png")
