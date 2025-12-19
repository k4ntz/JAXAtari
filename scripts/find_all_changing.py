
import ale_py
import gymnasium as gym
import numpy as np

def find_all_changing():
    env = gym.make("ALE/BeamRider-v5", render_mode="rgb_array", frameskip=1)
    env.reset(seed=42)
    
    rams = []
    for _ in range(50):
        obs, reward, term, trunc, info = env.step(0)
        rams.append(env.unwrapped.ale.getRAM().copy())
    
    rams = np.array(rams)
    for i in range(128):
        col = rams[:, i]
        if not np.all(col == col[0]):
            print(f"Address {i} changed: {col[:20]} ...")

if __name__ == "__main__":
    find_all_changing()
