
import ale_py
import gymnasium as gym
import numpy as np

def find_moving_bytes_relaxed():
    env = gym.make("ALE/BeamRider-v5", render_mode="rgb_array", frameskip=1)
    env.reset(seed=42)
    
    rams = []
    for _ in range(10):
        obs, reward, term, trunc, info = env.step(0)
        rams.append(env.unwrapped.ale.getRAM().copy())
    
    rams = np.array(rams)
    for i in range(128):
        col = rams[:, i]
        deltas = np.diff(col.astype(np.int16))
        # Ignore wrap around
        if np.all(deltas >= 0) and np.sum(deltas) > 0:
            print(f"Address {i} increases: {col}")
        elif np.all(deltas <= 0) and np.sum(deltas) < 0:
            print(f"Address {i} decreases: {col}")

if __name__ == "__main__":
    find_moving_bytes_relaxed()
