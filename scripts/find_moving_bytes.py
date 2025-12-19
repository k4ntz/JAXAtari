
import ale_py
import gymnasium as gym
import numpy as np

def find_moving_bytes():
    env = gym.make("ALE/BeamRider-v5", render_mode="rgb_array", frameskip=1)
    env.reset(seed=42)
    
    # Skip a few frames to let things settle
    for _ in range(10): env.step(0)
    
    ram1 = env.unwrapped.ale.getRAM().copy()
    env.step(0)
    ram2 = env.unwrapped.ale.getRAM().copy()
    env.step(0)
    ram3 = env.unwrapped.ale.getRAM().copy()
    
    diff1 = (ram2.astype(np.int16) - ram1.astype(np.int16))
    diff2 = (ram3.astype(np.int16) - ram2.astype(np.int16))
    
    for i in range(128):
        if diff1[i] > 0 and diff2[i] > 0:
            print(f"Address {i} is strictly increasing: {ram1[i]} -> {ram2[i]} -> {ram3[i]} (deltas {diff1[i]}, {diff2[i]})")
        elif diff1[i] < 0 and diff2[i] < 0:
             print(f"Address {i} is strictly decreasing: {ram1[i]} -> {ram2[i]} -> {ram3[i]} (deltas {diff1[i]}, {diff2[i]})")

if __name__ == "__main__":
    find_moving_bytes()
