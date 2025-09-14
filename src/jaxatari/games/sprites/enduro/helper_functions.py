import numpy as np


def crop_first_line_from_sprite(input_path: str, output_path: str) -> None:
    """
    Loads a sprite .npy file, crops the first line (row), and saves it.

    Args:
        input_path: Path to input .npy file (e.g., "/misc/flags.npy")
        output_path: Path to save the cropped .npy file
    """
    # Load the array
    array = np.load(input_path)

    # Crop first line (remove first row)
    cropped_array = array[:, 1:, :, :]

    # Save the modified sprite
    np.save(output_path, cropped_array)

    print(f"Original shape: {array.shape}")
    print(f"Cropped shape: {cropped_array.shape}")
    print(f"Saved cropped sprite to: {output_path}")

# Usage:
crop_first_line_from_sprite("misc/flags.npy", "misc/flags.npy")