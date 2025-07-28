import numpy as np
from PIL import Image

# ------- Alles entfernen und hier einfügen ---------


sprite = np.zeros((9, 16, 4), dtype=np.uint8)

sprite[10, 0] = [255, 228, 104, 255]


# --------------------------------

np.save('my_sprite.npy', sprite)
print("Sprite gespeichert als 'my_sprite.npy'")