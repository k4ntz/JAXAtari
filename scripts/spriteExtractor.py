import numpy as np
import matplotlib.pyplot as plt

filename = 'frame_00001.npy'  # Replace with your actual file name
# Load the image data
file_path = 'new_screenshots/' + filename # Replace with your actual file path
image = np.load(file_path)

# Ensure image is RGBA
if image.shape[-1] != 4:
    raise ValueError("Image must be RGBA (have 4 channels)")

# Create a mask of non-black pixels (not (0,0,0,255))
non_black_mask = ~np.all(image == [0, 0, 0, 255], axis=-1)

# Get bounding box of non-black pixels
coords = np.argwhere(non_black_mask)
if coords.size == 0:
    raise ValueError("Image contains no non-black pixels!")

y_min, x_min = coords.min(axis=0)
y_max, x_max = coords.max(axis=0) + 1  # add 1 for slicing

# Crop the image to the bounding box
cropped = image[y_min:y_max, x_min:x_max].copy()

# Set all black pixels (0,0,0,255) inside crop to transparent (0,0,0,0)
black_inside = np.all(cropped == [0, 0, 0, 255], axis=-1)
cropped[black_inside] = [0, 0, 0, 0]

# Show original and cropped image
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[0].axis('off')

axes[1].imshow(cropped)
axes[1].set_title("Cropped & Transparent")
axes[1].axis('off')

plt.tight_layout()
plt.show()

np.save('new_screenshots/'+filename,cropped)
    