
import ale_py
import gymnasium as gym
import numpy as np

def find_game_start_frame():
    env = gym.make("ALE/BeamRider-v5", render_mode="rgb_array", frameskip=1)
    env.reset(seed=42)
    
    # Heuristic for UFO: They are white and usually around Y=43 initially
    def count_ufo_pixels(frame):
        # White UFO color is roughly (236, 236, 236)
        white_mask = (frame[40:160, :, 0] > 200) & (frame[40:160, :, 1] > 200) & (frame[40:160, :, 2] > 200)
        return np.sum(white_mask)

    for f in range(1000):
        obs, reward, term, trunc, info = env.step(0)
        ufo_pixels = count_ufo_pixels(env.render())
        if ufo_pixels > 20: # Threshold for a small UFO sprite
            print(f"First UFO detected at frame {f}")
            break
    
    env.close()

if __name__ == "__main__":
    find_game_start_frame()
