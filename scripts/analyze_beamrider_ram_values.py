
import ale_py
import gymnasium as gym
import numpy as np

def analyze_ram_values():
    env = gym.make("ALE/BeamRider-v5", render_mode="rgb_array", frameskip=1)
    env.reset(seed=42)
    
    print("Frame | 67 | 68 | 69 | 70 | 71 | 72 |  4")
    print("-" * 40)
    for i in range(20):
        obs, reward, terminated, truncated, info = env.step(0) # NOOP
        ram = env.unwrapped.ale.getRAM()
        vals = [ram[67], ram[68], ram[69], ram[70], ram[71], ram[72], ram[4]]
        print(f"{i:5} | " + " | ".join(f"{v:2}" for v in vals))
    
    env.close()

if __name__ == "__main__":
    analyze_ram_values()
