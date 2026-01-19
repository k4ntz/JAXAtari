import sys
import numpy as np
from PIL import Image
import os

scale_factor = 40


files = os.listdir(sys.argv[1])
output_folder = os.path.join(sys.argv[1], "png_sprites")
os.makedirs(output_folder, exist_ok=True)
for filename in files:
    filepath = os.path.join(sys.argv[1], filename)
    output_filepath = os.path.join(output_folder, filename.replace('.npy', '.png'))
    if not filepath.endswith('.npy') or os.path.exists(output_filepath):
        continue
    print(f"Processing {filepath}...")
    # Load the sprite data from the binary file
    sprite_data = np.load(filepath) # RGBA values stored as uint8

    new_size = (sprite_data.shape[1] * scale_factor, sprite_data.shape[0] * scale_factor)
    bigger_sprite = np.repeat(np.repeat(sprite_data, scale_factor, axis=0), scale_factor, axis=1)

    # Convert the sprite data to an image
    try:
        image = Image.fromarray(bigger_sprite, 'RGBA')
    except Exception as e:
        print(f"Error converting sprite data to image for {filepath}: {e}")
        continue

    # Save the image as a PNG file
    image.save(output_filepath)
    print(f"Saved PNG image to {output_filepath}")