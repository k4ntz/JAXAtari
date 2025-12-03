import numpy as np
from pathlib import Path

# 🔧 adjust this if needed
CURSOR_PATH = Path("src/jaxatari/games/sprites/videochess/pieces/13.npy")

sprite = np.load(CURSOR_PATH)
print("Original shape:", sprite.shape)

# If it's RGB, add alpha channel
if sprite.shape[-1] == 3:
    h, w, _ = sprite.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = sprite
else:
    rgba = sprite.copy()

# Treat pure black as background → transparent
rgb = rgba[:, :, :3]
alpha = rgba[:, :, 3] if rgba.shape[-1] == 4 else np.full(rgb.shape[:2], 255, dtype=np.uint8)

# mask: pixels that are exactly black (0,0,0)
mask_black = np.all(rgb == [0, 0, 0], axis=-1)

# Set alpha: 0 for black, 255 for everything else
alpha = np.where(mask_black, 0, 255).astype(np.uint8)
rgba[:, :, 3] = alpha

np.save(CURSOR_PATH, rgba)
print("Saved updated cursor with transparency to:", CURSOR_PATH)