import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

# ------- Alles entfernen und hier einfügen ---------

sprite = np.zeros((2, 8, 4), dtype=np.uint8)

sprite[0, 0] = [255, 64, 255, 255]


# --------------------------------


img = Image.fromarray(sprite, mode='RGBA')

plt.imshow(img)
plt.show()