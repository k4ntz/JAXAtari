
import ale_py
import gymnasium as gym
import numpy as np

def find_counters():
    env = gym.make("ALE/BeamRider-v5", render_mode="rgb_array", frameskip=1)
    env.reset(seed=42)
    
    ram_history = []
    for _ in range(50):
        obs, reward, terminated, truncated, info = env.step(0)
        ram = env.unwrapped.ale.getRAM().copy()
        ram_history.append(ram)
    
    env.close()
    ram_history = np.array(ram_history)
    
    for i in range(128):
        col = ram_history[:, i]
        deltas = np.diff(col.astype(np.int16))
        # Handle wrap around
        deltas = np.where(deltas < -200, deltas + 256, deltas)
        deltas = np.where(deltas > 200, deltas - 256, deltas)
        
        if np.all(deltas == 1) or np.all(deltas == -1):
            print(f"Address {i} is a perfect counter (delta={deltas[0]})")
        elif np.all(np.isin(deltas, [0, 1])) and np.sum(deltas) > 10:
             print(f"Address {i} is a semi-regular counter (sum delta={np.sum(deltas)})")

if __name__ == "__main__":
    find_counters()
