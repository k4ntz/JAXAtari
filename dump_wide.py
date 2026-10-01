import warnings
warnings.filterwarnings("ignore")
import ale_py
ale_py.register_v5_envs()
import gymnasium as gym
import numpy as np

def is_play_mode(frame):
    region = frame[84:95, 60:92]
    return (region.sum(axis=2) > 60).sum() < 20

env = gym.make("ALE/StarGunner-v5", render_mode="rgb_array", frameskip=1)
env.reset(seed=0)
for _ in range(60):
    env.step(1)
for _ in range(150):
    env.step(1)

frame = env.render()
env.close()

# List ALL unique colors in the whole frame
print("Play mode:", is_play_mode(frame))
print("\nAll unique colors in frame:")
colors, counts = np.unique(frame.reshape(-1,3), axis=0, return_counts=True)
order = np.argsort(-counts)
for i in order[:30]:
    c = tuple(int(v) for v in colors[i])
    print(f"  {c}  n={int(counts[i])}")
