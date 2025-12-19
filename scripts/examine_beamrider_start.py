
import ale_py
import gymnasium as gym
import numpy as np

def examine_start():
    env = gym.make("ALE/BeamRider-v5", render_mode="rgb_array", frameskip=1)
    env.reset(seed=42)
    
    def get_lines(frame):
        blue_mask = (frame[:, :, 2] > 150) & (frame[:, :, 0] < 100)
        row_counts = np.sum(blue_mask, axis=1)
        line_rows = np.where(row_counts > 100)[0]
        line_rows = line_rows[line_rows < 170]
        return line_rows.tolist()

    for f in range(300):
        obs, reward, term, trunc, info = env.step(0)
        frame = env.render()
        lines = get_lines(frame)
        print(f"Frame {f}: Count={len(lines)} {lines}")
    
    env.close()

if __name__ == "__main__":
    examine_start()
