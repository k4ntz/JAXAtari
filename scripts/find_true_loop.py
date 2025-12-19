
import ale_py
import gymnasium as gym
import numpy as np

def find_true_loop():
    env = gym.make("ALE/BeamRider-v5", render_mode="rgb_array", frameskip=1)
    env.reset(seed=42)
    
    def get_lines(frame):
        blue_mask = (frame[:, :, 2] > 150) & (frame[:, :, 0] < 100)
        row_counts = np.sum(blue_mask, axis=1)
        line_rows = np.where(row_counts > 100)[0]
        line_rows = line_rows[line_rows < 170]
        if len(line_rows) == 0: return tuple()
        lines = []
        current_line = [line_rows[0]]
        for i in range(1, len(line_rows)):
            if line_rows[i] == line_rows[i-1] + 1:
                current_line.append(line_rows[i])
            else:
                lines.append(int(np.mean(current_line)))
                current_line = [line_rows[i]]
        lines.append(int(np.mean(current_line)))
        return tuple(lines)

    all_lines = []
    for f in range(2000):
        obs, reward, term, trunc, info = env.step(0)
        all_lines.append(get_lines(env.render()))
    env.close()
    
    # Search for a loop at the end of 2000 frames
    for length in range(2, 500):
        if all_lines[-length:] == all_lines[-2*length:-length]:
            print(f"Stable loop of length {length} found at end.")
            # Trace back to find where it FIRST starts repeating this loop
            # A loop is defined by the sequence of frames.
            loop_seq = all_lines[-length:]
            for start in range(2000 - length):
                if all_lines[start:start+length] == loop_seq:
                    # Check if it continues
                    if all_lines[start+length:start+2*length] == loop_seq:
                        print(f"Loop sequence first appears at {start}")
                        return all_lines[:start], loop_seq
    return None, None

if __name__ == "__main__":
    init, loop = find_true_loop()
    if init is not None:
        print(f"Init: {len(init)}, Loop: {len(loop)}")
        def pad(l): return list(l) + [-1] * (7 - len(l))
        np.save("beamrider_init_true.npy", np.array([pad(l) for l in init]))
        np.save("beamrider_loop_true.npy", np.array([pad(l) for l in loop]))
