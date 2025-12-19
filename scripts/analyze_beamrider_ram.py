
import ale_py
import gymnasium as gym
import numpy as np
def analyze_ram():
    env = gym.make("ALE/BeamRider-v5", render_mode="rgb_array", frameskip=1)
    env.reset(seed=42)
    
    ram_history = []
    for _ in range(200):
        obs, reward, terminated, truncated, info = env.step(0) # NOOP
        ram = env.unwrapped.ale.getRAM().copy()
        ram_history.append(ram)
    
    env.close()
    ram_history = np.array(ram_history)
    
    # Look for bytes that change linearly or periodically
    print("Address | Min | Max | Mean Delta | Variance of Delta")
    print("-" * 50)
    for i in range(128):
        col = ram_history[:, i]
        deltas = np.diff(col.astype(np.int16))
        # Handle wrap-around at 255
        deltas = np.where(deltas < -200, deltas + 256, deltas)
        deltas = np.where(deltas > 200, deltas - 256, deltas)
        
        if np.any(deltas != 0):
            mean_delta = np.mean(deltas)
            var_delta = np.var(deltas)
            # Lines usually move at a constant-ish speed or constant acceleration
            if 0.1 < abs(mean_delta) < 10:
                print(f"{i:7} | {col.min():3} | {col.max():3} | {mean_delta:10.4f} | {var_delta:10.4f}")

if __name__ == "__main__":
    analyze_ram()
