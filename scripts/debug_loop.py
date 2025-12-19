
import ale_py
import gymnasium as gym
import numpy as np

env = gym.make("ALE/BeamRider-v5", render_mode="rgb_array", frameskip=1)
env.reset(seed=42)

def get_lines(frame):
    blue_mask = (frame[:, :, 2] > 150) & (frame[:, :, 0] < 100)
    row_counts = np.sum(blue_mask, axis=1)
    line_rows = np.where(row_counts > 100)[0]
    line_rows = line_rows[line_rows < 170]
    return tuple(line_rows.tolist())

all_lines = []
for f in range(500):
    env.step(0)
    all_lines.append(get_lines(env.render()))
env.close()

# Print first 20 frames
for i in range(20):
    print(f"Frame {i}: {all_lines[i]}")

# Look for duplicates
for i in range(500):
    for j in range(i+1, 500):
        if all_lines[i] == all_lines[j]:
            print(f"Frame {i} is same as {j}")
            # Check if i+1 is same as j+1
            if i+1 < 500 and j+1 < 500 and all_lines[i+1] == all_lines[j+1]:
                print(f"LOOP DETECTED! Length {j-i} starting at {i}")
                import sys; sys.exit(0)
