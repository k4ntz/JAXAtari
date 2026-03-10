
import numpy as np
import os

SPRITE_DIR = "src/jaxatari/games/sprites/pacman"

# Constants matching User Request (Atari 2600 Specs)
BG_COLOR = [45, 45, 160]      # Blue
WALL_COLOR_FINAL = [200, 160, 50] # Yellow/Orange

def save_npy(rel_path, arr):
    path = os.path.join(SPRITE_DIR, rel_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr.astype(np.uint8))
    print(f"Saved {rel_path} ({arr.shape})")

def update_background():
    # 250x160 (Height x Width) per spec user said "Image size of the original is 250, 160"
    # ALE observation shape is (250, 160, 3)
    bg = np.full((250, 160, 3), BG_COLOR, dtype=np.uint8)
    save_npy("background.npy", bg)

def update_walls():
    # 8x8 Solid Block
    # Or Hollow? Atari is usually solid or double line.
    # Let's make a simple solid block with a border to look nice?
    # Or just solid as per "generate_atari_assets".
    # User said sprites still have weird pixels.
    # Walls are generated.
    # Let's make them clean solid blocks of WALL_COLOR.
    
    wall_block = np.full((8, 8, 3), WALL_COLOR_FINAL, dtype=np.uint8)
    # Maybe add a small detail? No, solid is safest for "clean".
    
    for i in range(16):
        save_npy(f"wall/wall_{i}.npy", wall_block)

def check_gamestate():
    print("Regenerating Background and Walls to match Code Constants (Black BG, Blue Wall)...")
    update_background()
    update_walls()

if __name__ == "__main__":
    check_gamestate()
