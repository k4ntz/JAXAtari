
import ale_py
import gymnasium as gym
import numpy as np

def collect_cycle():
    env = gym.make("ALE/BeamRider-v5", render_mode="rgb_array", frameskip=1)
    env.reset(seed=42)
    
    # Heuristic for blue lines
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

    history = {}
    for f in range(300):
        obs, reward, term, trunc, info = env.step(0)
        frame = env.render()
        lines = get_lines(frame)
        ram4 = env.unwrapped.ale.getRAM()[4]
        if ram4 not in history:
            history[ram4] = lines
            print(f"RAM[4]={ram4} -> Lines {lines}")
        if len(history) >= 128: # Since it decrements by 2, we only see half the values?
            break
    
    env.close()

if __name__ == "__main__":
    collect_cycle()
