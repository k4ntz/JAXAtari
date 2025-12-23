import numpy as np
from PIL import Image
import os

def convert_npy_to_png():
    """Converts alien death animation sprites from .npy to .png format."""

    # Get current directory path
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Loop through death animation sprites 1-4
    for i in range(1, 5):
        # Load .npy file
        npy_path = os.path.join(current_dir, f'frame_000{i}.npy')
        sprite_array = np.load(npy_path)

        # Convert to RGBA format
        if sprite_array.shape[-1] == 4:  # If already RGBA
            rgba_array = sprite_array
        else:  # If RGB
            rgba_array = np.dstack((sprite_array, np.full(sprite_array.shape[:-1], 255)))

        # Convert to uint8 if not already
        rgba_array = rgba_array.astype(np.uint8)

        # Create PIL Image
        image = Image.fromarray(rgba_array)

        # Save as PNG
        png_path = os.path.join(current_dir, f'frame_000{i}.png')
        image.save(png_path)

        print(f'Converted {npy_path} to {png_path}')

if __name__ == "__main__":
    convert_npy_to_png()