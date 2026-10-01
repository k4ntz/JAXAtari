import warnings
warnings.filterwarnings("ignore")
import ale_py
ale_py.register_v5_envs()
import gymnasium as gym
import numpy as np

env = gym.make("ALE/StarGunner-v5", render_mode="rgb_array", frameskip=1)
env.reset(seed=0)

frames = []
for _ in range(117):
    env.step(0)  # NOOP
    frames.append(env.render().copy())
env.close()

frames = np.array(frames, np.uint8)
np.savez_compressed("attract_frames.npz", frames=frames)
print("saved", frames.shape)
print("f0 == f116?", np.array_equal(frames[0], frames[116]))
