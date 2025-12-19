
import ale_py
import gymnasium as gym
import numpy as np

def find_line_causative_ram():
    env = gym.make("ALE/BeamRider-v5", render_mode="rgb_array", frameskip=1)
    env.reset(seed=42)
    
    # Heuristic for blue lines
    def get_lines(frame):
        blue_mask = (frame[:, :, 2] > 150) & (frame[:, :, 0] < 100)
        row_counts = np.sum(blue_mask, axis=1)
        line_rows = np.where(row_counts > 100)[0]
        line_rows = line_rows[line_rows < 170]
        return line_rows.tolist()

    original_state = env.unwrapped.ale.cloneState()
    # Step once to get a reference frame
    env.step(0)
    original_frame = env.render()
    original_lines = get_lines(original_frame)
    
    print(f"Original lines: {len(original_lines)} pixels")
    
    causative = []
    for i in range(128):
        env.unwrapped.ale.restoreState(original_state)
        # Try a few different values
        for val in [0, 64, 128, 255]:
            env.unwrapped.ale.setRAM(i, val)
            env.step(0)
            frame = env.render()
            lines = get_lines(frame)
            if lines != original_lines:
                print(f"RAM[{i}] = {val} CHANGED lines!")
                causative.append(i)
                break
    
    print(f"Found causative RAM indices: {causative}")
    env.close()

if __name__ == "__main__":
    find_line_causative_ram()
