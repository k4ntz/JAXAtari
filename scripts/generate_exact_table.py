
import ale_py
import gymnasium as gym
import numpy as np

def generate_exact_table():
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

    table = []
    for f in range(256):
        obs, reward, term, trunc, info = env.step(0)
        frame = env.render()
        lines = get_lines(frame)
        # Pad to 7 lines with -1
        padded = lines + [-1] * (7 - len(lines))
        table.append(padded)
    
    env.close()
    np.save("beamrider_blue_lines_table.npy", np.array(table))
    print("Table saved to beamrider_blue_lines_table.npy")
    print(f"Sample (frame 0): {table[0]}")
    print(f"Sample (frame 128): {table[128]}")

if __name__ == "__main__":
    generate_exact_table()
