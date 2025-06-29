import os
import pathlib
import sys
from pathlib import Path

import cv2
import numpy as np

#
# by Tim Morgner and Jan Larionow
#

def traverse_dir(root:str):
    for root, dirs, files in os.walk(root):
        for file in files:
            file_path = Path(root, file)
            if file_path.suffix == ".png":
                img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
                print(f"Converting {file_path} to npy (shape: {img.shape})")
                np.save(file_path.with_suffix(".npy"), img, allow_pickle=False)
            else:
                print(f"Skipping {file_path} as it is not a png file")

if __name__ == '__main__':
    print("Converting all png files to npy in the directory")
    if len(sys.argv) < 2:
        print("Please provide a directory to convert png files to npy")
        sys.exit(1)

    traverse_dir(sys.argv[1])
    print("Finished converting png files to npy")