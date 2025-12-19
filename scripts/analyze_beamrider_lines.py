
import ale_py
import gymnasium as gym
import numpy as np

def analyze_line_ram():
    env = gym.make("ALE/BeamRider-v5", render_mode="rgb_array", frameskip=1)
    env.reset(seed=42)
    
    print("Frame | 108 | 109 | 110 | 111 | 112 | 113 | 114 | Master(4)")
    print("-" * 60)
    for i in range(30):
        obs, reward, terminated, truncated, info = env.step(0)
        ram = env.unwrapped.ale.getRAM()
        vals = [ram[108], ram[109], ram[110], ram[111], ram[112], ram[113], ram[114], ram[4]]
        print(f"{i:5} | " + " | ".join(f"{v:3}" for v in vals))
    
    env.close()

if __name__ == "__main__":
    analyze_line_ram()
