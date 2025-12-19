
import ale_py
import gymnasium as gym
import numpy as np

def find_steady_state():
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
    for f in range(6000):
        obs, reward, term, trunc, info = env.step(0)
        frame = env.render()
        lines = get_lines(frame)
        all_lines.append(lines)
    
    env.close()
    
    line_strs = [str(l) for l in all_lines]
    
    # Many Atari games use powers of 2 for counters
    for length in [256, 512, 1024, 128, 64, 100, 200, 400]:
        rep1 = line_strs[-length:]
        rep2 = line_strs[-2*length:-length]
        if rep1 == rep2:
            print(f"Confirmed loop of length {length} at the end.")
            for start in range(len(all_lines) - 2*length):
                if line_strs[start:start+length] == line_strs[start+length:start+2*length]:
                    print(f"Steady state starts at frame {start}")
                    return all_lines[:start], all_lines[start:start+length]
    return None, None

if __name__ == "__main__":
    init, loop = find_steady_state()
    if init is not None:
        print(f"Init frames: {len(init)}, Loop frames: {len(loop)}")
        def pad(l): return l + [-1] * (7 - len(l))
        np.save("beamrider_init.npy", np.array([pad(l) for l in init]))
        np.save("beamrider_loop.npy", np.array([pad(l) for l in loop]))
    else:
        print("No loop found")
