import os
import numpy as np

# Absolute target directory from your error path
target_dir = r"C:\Users\manas\PycharmProjects\JAXAtari3\src\jaxatari\games\sprites\demonattack"

# Your exact game SCORE_COLOR (194, 169, 53) + 255 for full opacity
color_rgba = [194, 169, 53, 255]
transparent_rgba = [0, 0, 0, 0]

# Pure binary mask templates for the digits 0-9
patterns = [
    [[1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1]],  # 0
    [[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]],  # 1
    [[1, 1, 1], [0, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1]],  # 2
    [[1, 1, 1], [0, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1]],  # 3
    [[1, 0, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1]],  # 4
    [[1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 1], [1, 1, 1]],  # 5
    [[1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 0, 1], [1, 1, 1]],  # 6
    [[1, 1, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]],  # 7
    [[1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 1, 1]],  # 8
    [[1, 1, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1]],  # 9
]

for i, pattern in enumerate(patterns):
    # Create an empty RGBA array canvas of shape (height=5, width=3, channels=4)
    compiled_rgba = np.zeros((5, 3, 4), dtype=np.uint8)

    # Fill in the pixel values matching the layout pattern coordinates
    for r, row in enumerate(pattern):
        for c, val in enumerate(row):
            if val:
                compiled_rgba[r, c] = color_rgba
            else:
                compiled_rgba[r, c] = transparent_rgba

    filename = f"demonattack_score_{i}.npy"
    np.save(os.path.join(target_dir, filename), compiled_rgba)

print(f"🎉 Successfully re-written RGBA 4-channel demonattack_score_0.npy through _9.npy to:\n{target_dir}")