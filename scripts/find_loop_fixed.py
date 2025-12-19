
import ale_py
import gymnasium as gym
import numpy as np

def find_loop():
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
    # Capture a lot of frames
    for f in range(5000):
        obs, reward, term, trunc, info = env.step(0)
        frame = env.render()
        all_lines.append(get_lines(frame))
    env.close()
    
    line_strs = [str(l) for l in all_lines]
    
    # Search for any loop length from 2 to 1000
    for length in range(2, 1000):
        # Check last 3 repetitions
        if line_strs[-length:] == line_strs[-2*length:-length] == line_strs[-3*length:-2*length]:
            print(f"Found stable loop of length {length}")
            # Find the FIRST time this loop starts
            for start in range(len(all_lines) - 3*length):
                if line_strs[start:start+length] == line_strs[start+length:start+2*length] == line_strs[start+2*length:start+3*length]:
                    print(f"Loop starts at frame {start}")
                    return all_lines[:start], all_lines[start:start+length]
    return None, None

if __name__ == "__main__":
    init, loop = find_loop()
    if init is not None:
        print(f"Init: {len(init)}, Loop: {len(loop)}")
        def pad(l): return l + [-1] * (7 - len(l))
        np.save("beamrider_init_fixed.npy", np.array([pad(l) for l in init]))
        np.save("beamrider_loop_fixed.npy", np.array([pad(l) for l in loop]))
