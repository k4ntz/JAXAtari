import numpy as np
import os
import sys

BASECOLOR = (0, 0, 0, 255)
NEWCOLOR = (0, 0, 0, 0)
BASEDIR = os.path.expanduser("~") +  "/Library/Application Support/jaxatari/sprites"
if not os.path.exists(BASEDIR):
    print("Please fix BASEDIR in the script")
    exit(1)
sprites_dir = os.path.join(BASEDIR, sys.argv[1])

if not os.path.exists(sprites_dir):
    print(sys.argv[1] + f" not available in {BASEDIR}")
    print("Available:\n" + os.listdir(BASEDIR))
    exit(1)
    
for path, subdirs, files in os.walk(sprites_dir):
    for name in files:
        filesprite = os.path.join(path, name)
        sprite = np.load(filesprite)
        if np.any(np.all(sprite == BASECOLOR, axis=-1)):
            mask = np.all(sprite == BASECOLOR, axis=-1)
            sprite[mask] = NEWCOLOR
            np.save(filesprite, sprite)
            print(filesprite, ": Done !")
        else:
            print(f"Color {BASECOLOR} not found in ", filesprite)
