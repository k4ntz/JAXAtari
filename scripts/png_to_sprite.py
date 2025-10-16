import os
import numpy as np
from PIL import Image

def png_to_npy(png_path, npy_path=None):
    """
    Convert a PNG image to a .npy file with shape (W, H, RGBA).
    """
    # PNG laden und in RGBA umwandeln
    img = Image.open(png_path).convert("RGBA")
    arr = np.array(img)

    # Zielpfad
    if npy_path is None:
        npy_path = os.path.splitext(png_path)[0] + ".npy"

    # Speichern
    np.save(npy_path, arr)
    print(f"Saved: {npy_path} (Shape: {arr.shape})")

def convert_directory(directory):
    """Convert all PNG files in a directory to .npy"""
    for filename in os.listdir(directory):
        if filename.lower().endswith(".png"):
            png_path = os.path.join(directory, filename)
            png_to_npy(png_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert PNG images to .npy sprite files.")
    parser.add_argument("path", help="Path to a PNG file or a directory containing PNG files.")
    args = parser.parse_args()

    if os.path.isdir(args.path):
        convert_directory(args.path)
    elif os.path.isfile(args.path) and args.path.lower().endswith(".png"):
        png_to_npy(args.path)
    else:
        print("Error: Please provide a PNG file or a directory containing PNG files.")
