
import ale_py
import gymnasium as gym
import numpy as np

def test_ram4_control():
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

    # Manually set RAM[4] and see if lines move immediately
    print("Setting RAM[4] to 100...")
    env.unwrapped.ale.setRAM(4, 100)
    # Step once to apply (Atari games often need a frame)
    env.step(0)
    frame = env.render()
    lines = get_lines(frame)
    print(f"Lines at RAM[4]=100: {lines}")
    
    print("Setting RAM[4] to 200...")
    env.unwrapped.ale.setRAM(4, 200)
    env.step(0)
    frame = env.render()
    lines = get_lines(frame)
    print(f"Lines at RAM[4]=200: {lines}")

    env.close()

if __name__ == "__main__":
    test_ram4_control()
