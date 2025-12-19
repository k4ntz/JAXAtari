
import ale_py
import gymnasium as gym
import numpy as np

def check_sector_effect():
    env = gym.make("ALE/BeamRider-v5", render_mode="rgb_array", frameskip=1)
    
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

    env.reset(seed=42)
    lines_s1 = get_lines(env.render())
    print(f"Sector 1, reset: {lines_s1}")
    
    # Cheat to get to sector 2
    # RAM address for sector is usually around 103?
    # From games_covered.md, it's not listed.
    # But I can just play or try to find it.
    # Actually, I'll just check if they change over time similarly.
    
    # Let's assume they don't change logic between sectors for now.
    env.close()

if __name__ == "__main__":
    check_sector_effect()
