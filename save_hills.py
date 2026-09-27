import warnings
warnings.filterwarnings("ignore")
import ale_py
ale_py.register_v5_envs()
import gymnasium as gym
import numpy as np

env = gym.make("ALE/StarGunner-v5", render_mode="rgb_array", frameskip=1)
env.reset(seed=0)
for _ in range(30):
    env.step(1)  # start

# Sammle 200 Frames bei NOOP — Hügel scrollen gleichmäßig
frames = []
for _ in range(200):
    env.step(0)
    frames.append(env.render().copy())
env.close()
frames = np.array(frames)

# Finde die Periode: vergleiche Frame 0 mit späteren
for lag in range(20, 150):
    if np.array_equal(frames[0, 185:], frames[lag, 185:]):
        print(f"Periode gefunden: {lag}")
        period = lag
        break
else:
    period = 80
    print(f"Keine exakte Periode, benutze {period}")

# Speichere nur den Hügel-Streifen (y 185-210, 160 breit)
hills = frames[:period, 185:, :, :]
np.savez_compressed("hills_cycle.npz", hills=hills)
print("saved hills_cycle.npz:", hills.shape)
