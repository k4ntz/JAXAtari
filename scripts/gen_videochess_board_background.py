# /Users/nico/Repos/JAXAtari/scripts/gen_videochess_board_background.py
import numpy as np
from pathlib import Path

BOARD_PATH = Path("src/jaxatari/games/sprites/videochess/background.npy")

DARK_BLUE  = np.array([0, 0, 170, 255], dtype=np.uint8)
LIGHT_BLUE = np.array([68, 68, 255, 255], dtype=np.uint8)

def main():
    arr = np.load(BOARD_PATH)
    print("Loaded:", BOARD_PATH, arr.shape)

    H, W = arr.shape[:2]
    out = np.zeros_like(arr)

    cell_h = H // 8
    cell_w = W // 8

    for r in range(8):
        for c in range(8):
            y0, y1 = r * cell_h, (r + 1) * cell_h if r < 7 else H
            x0, x1 = c * cell_w, (c + 1) * cell_w if c < 7 else W

            color = DARK_BLUE if (r + c) % 2 == 1 else LIGHT_BLUE

            out[y0:y1, x0:x1, :3] = color[:3]
            out[y0:y1, x0:x1, 3] = 255

    np.save(BOARD_PATH, out)
    print("Saved new blue board background")
    
if __name__ == "__main__":
    main()