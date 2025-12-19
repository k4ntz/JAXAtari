
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
    if len(line_rows) == 0: return []
    lines = []
    current_line = [line_rows[0]]
    for i in range(1, len(line_rows)):
        if line_rows[i] == line_rows[i-1] + 1:
            current_line.append(line_rows[i])
        else:
            lines.append(int(np.mean(current_line)))
            current_line = [line_rows[i]]
    lines.append(int(np.mean(current_line)))
    return lines

all_lines = []
for f in range(3000):
    env.step(0)
    all_lines.append(get_lines(env.render()))
env.close()

line_strs = [str(l) for l in all_lines]

# Check if frame 128 repeats later
start_frame = 128
for j in range(start_frame + 1, 3000):
    if line_strs[j] == line_strs[start_frame]:
        length = j - start_frame
        # Check if next few frames also match
        match = True
        for k in range(1, 20):
            if line_strs[start_frame + k] != line_strs[j + k]:
                match = False
                break
        if match:
            print(f"Found loop starting at {start_frame} with length {length}")
            # See if it repeats again
            if line_strs[j:j+length] == line_strs[j+length:j+2*length]:
                print(f"Confirmed triple repeat of length {length}")
                import sys; sys.exit(0)
