
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
for f in range(1000):
    env.step(0)
    all_lines.append(get_lines(env.render()))
env.close()

line_strs = [str(l) for l in all_lines]
for length in range(2, 500):
    if line_strs[-length:] == line_strs[-2*length:-length]:
        print(f"Found loop length {length}")
        # Find start
        for start in range(1000 - 2*length):
            if line_strs[start:start+length] == line_strs[start+length:start+2*length]:
                print(f"Starts at {start}")
                init = all_lines[:start]
                loop = all_lines[start:start+length]
                def pad(l): return l + [-1] * (7 - len(l))
                np.save("beamrider_init_v3.npy", np.array([pad(l) for l in init]))
                np.save("beamrider_loop_v3.npy", np.array([pad(l) for l in loop]))
                import sys; sys.exit(0)
