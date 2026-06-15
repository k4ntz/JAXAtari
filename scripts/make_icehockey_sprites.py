#!/usr/bin/env python3
"""
make_icehockey_sprites.py
─────────────────────────
Run this script ONCE from the project root to generate placeholder .npy
sprites for IceHockey.  Replace them later with real Atari sprites extracted
via  scripts/frame_extractor.py  +  scripts/spriteEditor.py.

Usage (from project root):
    python scripts/make_icehockey_sprites.py

Writes to:  src/jaxatari/games/sprites/icehockey/
"""

import os
import sys
import numpy as np

# ── Output directory ──────────────────────────────────────────────────────────
# Resolves to  <repo_root>/src/jaxatari/games/sprites/icehockey/
OUTPUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "src", "jaxatari", "games", "sprites", "icehockey",
)


def _save(name: str, arr: np.ndarray) -> None:
    path = os.path.join(OUTPUT, name)
    np.save(path, arr)
    print(f"  {name:<32s}  shape={str(arr.shape):<20s}  dtype={arr.dtype}")


# ─────────────────────────────────────────────────────────────────────────────
# Background  (210 × 160 RGBA)
# Reproduces the key visual zones of ALE IceHockey:
#   • Black score bars at top/bottom rows
#   • Light-green ice surface
#   • White boards on all four sides of the rink
#   • Red centre line (horizontal, ~row 103)
#   • Two blue lines (one in each zone)
#   • Light-grey goal creases at top (enemy) and bottom (player)
# ─────────────────────────────────────────────────────────────────────────────
def make_background() -> np.ndarray:
    H, W = 210, 160
    bg = np.zeros((H, W, 4), dtype=np.uint8)

    # Ice surface
    bg[:] = (167, 222, 186, 255)

    # Score bars (black strips)
    bg[:16]  = (0, 0, 0, 255)
    bg[194:] = (0, 0, 0, 255)

    # Boards (white)
    wh = (236, 236, 236, 255)
    bg[16:194, :4]    = wh   # left board
    bg[16:194, 156:]  = wh   # right board
    bg[16:20,  4:156] = wh   # top board
    bg[190:194, 4:156] = wh  # bottom board

    # Centre red line
    bg[101:106, 4:156] = (213, 72, 72, 255)

    # Blue lines
    bg[68:72,   4:156] = (66, 72, 200, 255)   # upper (in player's attacking zone)
    bg[133:137, 4:156] = (66, 72, 200, 255)   # lower (in enemy's attacking zone)

    # Goal creases (light grey, centred horizontally)
    crease = (220, 220, 220, 255)
    bg[20:27,   60:100] = crease   # enemy goal (top)
    bg[187:194, 60:100] = crease   # player goal (bottom)

    return bg


# ─────────────────────────────────────────────────────────────────────────────
# Skater sprites  (12 H × 8 W RGBA)
# Solid-colour rectangle with a 1-px transparent border.
# ─────────────────────────────────────────────────────────────────────────────
def make_skater(rgb) -> np.ndarray:
    s = np.zeros((12, 8, 4), dtype=np.uint8)
    s[1:-1, 1:-1] = (*rgb, 255)
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Puck  (3 H × 4 W RGBA)
# ─────────────────────────────────────────────────────────────────────────────
def make_puck() -> np.ndarray:
    s = np.zeros((3, 4, 4), dtype=np.uint8)
    s[:] = (20, 20, 20, 255)
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Digit sprites  (8 H × 5 W RGBA, white, 7-segment style)
# ─────────────────────────────────────────────────────────────────────────────
def make_digit(d: int) -> np.ndarray:
    H, W = 8, 5
    s = np.zeros((H, W, 4), dtype=np.uint8)
    c = np.array([236, 236, 236, 255], dtype=np.uint8)

    # Segment bounding boxes: (row_start, row_end, col_start, col_end)
    T  = (0, 2, 1, 4)   # top horizontal
    M  = (3, 5, 1, 4)   # middle horizontal
    B  = (6, 8, 1, 4)   # bottom horizontal
    TL = (0, 4, 0, 2)   # top-left vertical
    TR = (0, 4, 3, 5)   # top-right vertical
    BL = (4, 8, 0, 2)   # bottom-left vertical
    BR = (4, 8, 3, 5)   # bottom-right vertical

    segs = {
        0: [T, TL, TR, BL, BR, B],
        1: [TR, BR],
        2: [T, TR, M, BL, B],
        3: [T, TR, M, BR, B],
        4: [TL, TR, M, BR],
        5: [T, TL, M, BR, B],
        6: [T, TL, M, BL, BR, B],
        7: [T, TR, BR],
        8: [T, TL, TR, M, BL, BR, B],
        9: [T, TL, TR, M, BR, B],
    }
    for r0, r1, c0, c1 in segs.get(d, []):
        s[r0:r1, c0:c1] = c
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(OUTPUT, exist_ok=True)
    abs_out = os.path.abspath(OUTPUT)
    print(f"Writing placeholder sprites to:\n  {abs_out}\n")

    _save("background.npy",  make_background())
    _save("player.npy",      make_skater((92, 186, 92)))    # green = player team
    _save("enemy.npy",       make_skater((213, 130, 74)))   # orange = enemy team
    _save("puck.npy",        make_puck())
    for i in range(10):
        _save(f"digit_{i}.npy", make_digit(i))

    total = 4 + 10
    print(f"\nDone — {total} files written to:\n  {abs_out}")
    print(
        "\nNOTE: These are placeholder sprites.  Once the game is running,\n"
        "replace them with real Atari sprites extracted via:\n"
        "  python scripts/frame_extractor.py  (to capture ALE frames)\n"
        "  python scripts/spriteEditor.py     (to cut out individual sprites)"
    )
