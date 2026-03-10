
import numpy as np
import os

DIR = "extracted_sprites"

def show_sprite(name):
    path = os.path.join(DIR, name)
    if not os.path.exists(path):
        print(f"Not found: {path}")
        return
    arr = np.load(path)
    print(f"\n--- {name} ({arr.shape}) ---")
    
    # Simple ASCII
    # Take Max channel or just check for non-zero
    mask = np.any(arr != 0, axis=-1)
    for row in mask:
        line = "".join(["#" if x else "." for x in row])
        print(line)

files_to_check = [
    "src/jaxatari/games/sprites/pacman/pellet_dot.npy",
    "src/jaxatari/games/sprites/pacman/pellet_power/pellet_power_on.npy"
]

for f in files_to_check:
    path = f # Relative to root directly
    if not os.path.exists(path):
        print(f"Not found: {path}")
        continue
    arr = np.load(path)
    print(f"\n--- {f} ({arr.shape}) ---")
    
    # Simple ASCII
    # Take Max channel or just check for non-zero
    mask = np.any(arr != 0, axis=-1)
    for row in mask:
        line = "".join(["#" if x else "." for x in row])
        print(line)
