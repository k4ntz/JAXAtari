import numpy as np
import os


def create_digit_array(directory_path=".", color='black'):
    """
    Loads numbered sprite files (0-9) and stacks them vertically in reverse order.
    Adds blank white lines between each sprite.

    Args:
        directory_path (str): Path to directory containing the sprite files.
                             Defaults to current directory.
        color (str): Color name for the sprite files. Defaults to 'black'.

    Returns:
        numpy.ndarray: The stacked array with separators
    """
    sprites = []
    numbers = [0] + list(range(9, -1, -1))
    print(numbers)

    # Load sprites from 9 down to 0 (so they appear in order 9,8,7...0 when stacked)
    for i in numbers:
        filename = f"{i}_{color}.npy"
        filepath = os.path.join(directory_path, filename)

        try:
            sprite = np.load(filepath)
            sprites.append(sprite)
            print(f"Loaded {filename} with shape {sprite.shape}")
        except FileNotFoundError:
            print(f"Warning: {filename} not found in {directory_path}")
        except Exception as e:
            print(f"Error loading {filename}: {e}")

    if not sprites:
        print("No sprite files found!")
        return

    # Get dimensions and standardize channel count
    sprite_height, sprite_width = sprites[0].shape[:2]

    # Print shapes for debugging
    for i, sprite in enumerate(sprites[:3]):  # Print first few shapes
        print(f"Sprite {i} shape: {sprite.shape}")

    # Find the maximum number of channels across all sprites (3 for RGB, 4 for RGBA)
    max_channels = max(sprite.shape[2] for sprite in sprites)

    # Standardize all sprites to have the same number of channels
    standardized_sprites = []
    for sprite in sprites:
        if sprite.shape[2] < max_channels:  # Need to add alpha channel
            # Add alpha channel (fully opaque = 255)
            alpha_channel = np.full((sprite.shape[0], sprite.shape[1], 1), 255, dtype=sprite.dtype)
            sprite = np.concatenate([sprite, alpha_channel], axis=2)
        elif sprite.shape[2] > max_channels:  # Need to remove channels (shouldn't happen)
            sprite = sprite[:, :, :max_channels]

        standardized_sprites.append(sprite)

    # Create blank line with correct dimensions and channels
    if max_channels == 4:  # RGBA
        # Create fully transparent separator line (alpha = 0)
        blank_line = np.full((1, sprite_width, 4), [0, 0, 0, 0], dtype=sprites[0].dtype)
    else:  # RGB - this shouldn't happen with your RGBA sprites, but just in case
        blank_line = np.full((1, sprite_width, 3), [0, 0, 0], dtype=sprites[0].dtype)

    # Stack sprites with blank lines between them and after the last sprite
    stacked_parts = []
    for i, sprite in enumerate(standardized_sprites):
        stacked_parts.append(sprite)
        # Add blank line after each sprite (including the last one)
        stacked_parts.append(blank_line)

    stacked_array = np.vstack(stacked_parts)

    # Save the result
    output_path = os.path.join(directory_path, f"{color}_digit_array.npy")
    np.save(output_path, stacked_array)

    print(f"Successfully created {output_path}")
    print(f"Final array shape: {stacked_array.shape}")

    return stacked_array


# Example usage:
if __name__ == "__main__":
    # Use current directory with brown color
    create_digit_array(color='black')

    # Or specify a different directory and color
    # create_digit_array("/path/to/your/sprites", color='black')