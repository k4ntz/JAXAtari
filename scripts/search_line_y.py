
import ale_py
import gymnasium as gym
import numpy as np

def search_for_line_y():
    env = gym.make("ALE/BeamRider-v5", render_mode="rgb_array", frameskip=1)
    env.reset(seed=42)
    
    target_y = 45 # From frame 0
    
    ram = env.unwrapped.ale.getRAM()
    indices = np.where(ram == target_y)[0]
    print(f"Found {target_y} at indices {indices}")
    
    # Step once
    env.step(0)
    ram = env.unwrapped.ale.getRAM()
    indices = np.where(ram == target_y)[0]
    print(f"After 1 step, found {target_y} at indices {indices}")

    env.close()

if __name__ == "__main__":
    search_for_line_y()
