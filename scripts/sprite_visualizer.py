#!/usr/bin/env python3
"""
sprite_visualizer.py

A simple script to load and visualize a NumPy-based sprite (.npy) file.
For example from the project folder run:
    python scripts/sprite_visualizer.py src/jaxatari/games/sprites/kangaroo/kangaroo_jump_high.npy
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Load and visualize a .npy sprite file (RGBA).")
    parser.add_argument(
        'npy_path',
        type=Path,
        help="Path to the .npy file containing the sprite array"
    )
    args = parser.parse_args()

    # Ensure the file exists
    if not args.npy_path.exists():
        print(f"Error: file not found: {args.npy_path}")
        return

    # Load the sprite array
    sprite = np.load(str(args.npy_path))
    print(f"Loaded sprite shape: {sprite.shape}, dtype: {sprite.dtype}")

    # Visualize the sprite
    plt.figure(figsize=(4, 4))
    plt.imshow(sprite)
    plt.axis('off')
    plt.title(f"Sprite: {args.npy_path.name}")
    plt.show()

if __name__ == "__main__":
    main()
