
import ale_py
import gymnasium as gym
import numpy as np

def search_ram_for_values():
    env = gym.make("ALE/BeamRider-v5", render_mode="rgb_array", frameskip=1)
    env.reset(seed=42)
    
    targets = [118, 90, 156, 49, 58, 71]
    
    for i in range(10):
        obs, reward, terminated, truncated, info = env.step(0)
        ram = env.unwrapped.ale.getRAM()
        found = []
        for t in targets:
            indices = np.where(np.abs(ram.astype(np.int16) - t) <= 2)[0]
            if len(indices) > 0:
                found.append((t, indices.tolist()))
        
        if found:
            print(f"Frame {i}: Found matches {found}")
    
    env.close()

if __name__ == "__main__":
    search_ram_for_values()
