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
    for f in range(2000):
        obs, reward, term, trunc, info = env.step(0)
        frame = env.render()
        lines = get_lines(frame)
        all_lines.append(lines)
    
    env.close()
    
    line_strs = [str(l) for l in all_lines]
    
    for length in range(10, 1000):
        # Check the last few repetitions
        match = True
        for rep in range(1, 3):
            for i in range(length):
                if line_strs[1999 - i] != line_strs[1999 - i - rep*length]:
                    match = False
                    break
            if not match: break
        
        if match:
            print(f"Found loop of length {length}!")
            for start in range(2000 - 3*length):
                is_loop = True
                for rep in range(3):
                    for i in range(length):
                        if line_strs[start + i] != line_strs[start + i + rep*length]:
                            is_loop = False
                            break
                    if not is_loop: break
                if is_loop:
                    print(f"Loop starts at frame {start}")
                    init_part = all_lines[:start]
                    loop_part = all_lines[start:start+length]
                    return init_part, loop_part
    print("No loop found")
    return None, None

if __name__ == "__main__":
    init, loop = find_loop()
    if init is not None:
        print(f"Init length: {len(init)}, Loop length: {len(loop)}")
        def pad(l): return l + [-1] * (7 - len(l))
        np.save("beamrider_init.npy", np.array([pad(l) for l in init]))
        np.save("beamrider_loop.npy", np.array([pad(l) for l in loop]))