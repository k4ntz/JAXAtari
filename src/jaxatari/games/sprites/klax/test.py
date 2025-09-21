import numpy as np
import matplotlib.pyplot as plt

# Sprite laden
arr = np.load("tile_blue.npy")

# Neuen schwarzen Hintergrund erstellen (selbe Größe, RGB)
h, w = arr.shape[:2]
background = np.zeros((h, w, 3), dtype=np.uint8)

# Prüfen, ob Bild einen Alpha-Kanal hat
if arr.shape[-1] == 4:
    rgb, alpha = arr[..., :3], arr[..., 3] / 255.0
    # Alphablending mit schwarzem Hintergrund
    blended = (rgb * alpha[..., None]).astype(np.uint8)
else:
    print("Fehler: Das Bild hat keinen Alpha-Kanal (4 Kanäle erwartet).")
    blended = arr[..., :3] if arr.shape[-1] >= 3 else arr  # Fallback, damit es trotzdem darstellbar ist

# Anzeigen
plt.imshow(blended)
plt.axis("off")
plt.show()




'''

Nächste ToDos: 
✓ If, for los werden 
✓ spawn speed realistischer/langsamer machen (möglichkeit zu Variabilität mit waves)
✓ Up button schießt hoch 62
✓ tiles fallen nicht von ganz oben, sondern vom Spielfeldrand
✓ waves einrichten
✓ Boni am ende der Wave
✓ Wave Ziele dem Agenten mitgeben
✓ Sprites für remaining lives
✓ Sprites für remaining bei Wave task
✓ Sprites für Anzeige Wave task
✓ Sprites für Gesamtpunktzahl
✓ Sprites einbinden
- git pull
- Waves speed
- Tiles erscheinen erst in späteren waves
        level 1: orange, yellow, red, green tiles (✓)
        level 2: add blue tiles (✓)
        level 3: add purple, white tiles (✓)
        level 6: add light blue tiles, "wild" tiles
        level 60: dark green tiles appear
        level 90: grey tiles appear
- Warp bonus in wave 6 brings u to wave 51 & gives u 50.000 points (big X of two 5-tile diagonals)

'''

