 warnings.filterwarnings("ignore")
import ale_py
ale_py.register_v5_envs()
import gymnasium as gym
import numpy as np

env = gym.make("ALE/StarGunner-v5", render_mode="rgb_array", frameskip=1)
env.reset(seed=0)

# Viele FIREs zum Starten (funktioniert zuverlässig)
for _ in range(60):
    env.step(1)

def is_play_mode(frame):
    region = frame[84:95, 60:92]
    non_black = (region.sum(axis=2) > 60).sum()
    return non_black < 20

def find_bobo_x(frame):
    r = frame[:,:,0].astype(int)
    g = frame[:,:,1].astype(int)
    b = frame[:,:,2].astype(int)
    mask = (np.abs(r-162)<30) & (np.abs(g-134)<30) & (np.abs(b-56)<30)
    mask[100:,:] = False
    ys, xs = np.where(mask)
    if len(xs) < 3:
        return None
    return float(np.median(xs))

# Prüfe Startbedingung
frame = env.render()
print(f"play mode erkannt: {is_play_mode(frame)}")

# Miss Bobo während FIRE
positions = []
for i in range(200):
    env.step(1)  # FIRE — hält den Spieler am Leben
    frame = env.render()
    x = find_bobo_x(frame)
    if x is not None:
        positions.append(x)

env.close()
print(f"erkannte Bobo-Frames: {len(positions)}")
if len(positions) > 5:
    print("erste 20 x-Werte:", [round(x,1) for x in positions[:20]])
    diffs = np.diff(positions)
    small = diffs[np.abs(diffs) < 30]
    if len(small) > 0:
        print(f"mean dx: {small.mean():.3f} px/frame")
        print(f"median dx: {np.median(small):.3f} px/frame")
    else:
        print("keine Bewegung erkannt — Bobo bewegt sich langsam")
