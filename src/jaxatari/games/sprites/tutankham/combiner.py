import numpy as np
import os

# Load the two files
path = f"{os.path.dirname(os.path.abspath(__file__))}"

a = np.load(f"{path}/floor_one.npy")
b = np.load(f"{path}/floor_two.npy")
#c = np.load(f"file_b.npy")

# Sanity check
if a.shape[1:] != b.shape[1:]:
    raise ValueError(
        f"Shape mismatch: file_a {a.shape} and file_b {b.shape} "
        "cannot be stacked vertically"
    )

# Stack A on top of B
combined = np.vstack((a, b))

# Save result
np.save("combined_floor.npy", combined)

print("Saved combined.npy with shape:", combined.shape)
