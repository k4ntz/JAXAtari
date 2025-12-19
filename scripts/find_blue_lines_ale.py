
import ale_py
import gymnasium as gym
import numpy as np

def find_blue_lines_in_frame():
    env = gym.make("ALE/BeamRider-v5", render_mode="rgb_array", frameskip=1)
    env.reset(seed=42)
    
    results = []
    for f in range(50):
        obs, reward, term, trunc, info = env.step(0)
        frame = env.render()
        
        # Find horizontal blue lines
        # Blue color in BeamRider is roughly (104, 140, 210) or similar
        # Let's look for pixels with high blue component and low red/green
        # Based on the palette, blue lines are likely color ID 0x9A or similar
        
        # Simple heuristic: blue > 150 and red < 100
        blue_mask = (frame[:, :, 2] > 150) & (frame[:, :, 0] < 100)
        
        # Find rows where there are many blue pixels
        row_counts = np.sum(blue_mask, axis=1)
        # Horizontal lines span most of the width (160)
        line_rows = np.where(row_counts > 100)[0]
        
        # Only keep rows in the play area (above 170)
        line_rows = line_rows[line_rows < 170]
        
        # Group adjacent rows
        if len(line_rows) > 0:
            lines = []
            if len(line_rows) > 0:
                current_line = [line_rows[0]]
                for i in range(1, len(line_rows)):
                    if line_rows[i] == line_rows[i-1] + 1:
                        current_line.append(line_rows[i])
                    else:
                        lines.append(int(np.mean(current_line)))
                        current_line = [line_rows[i]]
                lines.append(int(np.mean(current_line)))
            
            # Get RAM state
            ram = env.unwrapped.ale.getRAM().copy()
            results.append((f, lines, ram[4]))
            print(f"Frame {f}: Lines at Y={lines}, Master RAM[4]={ram[4]}")

    env.close()

if __name__ == "__main__":
    find_blue_lines_in_frame()
