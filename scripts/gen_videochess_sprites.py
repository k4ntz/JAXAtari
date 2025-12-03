import numpy as np
from pathlib import Path

# 🔧 Adjust this if needed:
# This should be the folder where your VideoChess piece .npy files live.
SPRITE_DIR = Path("src/jaxatari/games/sprites/videochess/pieces")

SIZE = 9  # 9x9 sprites, like the existing cursor

# Colors (RGBA)
TRANSPARENT = np.array([0, 0, 0, 0], dtype=np.uint8)
WHITE       = np.array([255, 255, 255, 255], dtype=np.uint8)
ORANGE      = np.array([255, 140, 0, 255], dtype=np.uint8)


def blank():
    """Return a blank transparent RGBA sprite."""
    return np.zeros((SIZE, SIZE, 4), dtype=np.uint8)


def put(px, x, y, color):
    """Set one pixel (x,y) to color, if inside bounds."""
    if 0 <= y < SIZE and 0 <= x < SIZE:
        px[y, x] = color


def save_sprite(index: int, img: np.ndarray):
    SPRITE_DIR.mkdir(parents=True, exist_ok=True)
    path = SPRITE_DIR / f"{index}.npy"
    np.save(path, img)
    print(f"Saved {path}")


# ---------- Simple shapes for each piece type ----------

def make_pawn(color):
    img = blank()
    # Simple 3x3 blob in centre
    for y in range(3, 6):
        for x in range(3, 6):
            put(img, x, y, color)
    # Little top pixel
    put(img, 4, 2, color)
    return img


def make_rook(color):
    img = blank()
    # Base rectangle
    for y in range(4, 8):
        for x in range(2, 7):
            put(img, x, y, color)
    # Battlements
    for x in [2, 4, 6]:
        put(img, x, 3, color)
    return img


def make_bishop(color):
    img = blank()
    # Diagonal body
    for i in range(2, 7):
        put(img, 4, i, color)
    put(img, 3, 3, color)
    put(img, 5, 3, color)
    # Small head
    put(img, 4, 1, color)
    return img


def make_knight(color):
    img = blank()
    # Rough horse head / L-shape
    for y in range(3, 7):
        put(img, 3, y, color)
    for x in range(3, 7):
        put(img, x, 6, color)
    put(img, 4, 3, color)
    put(img, 5, 2, color)
    # Eye
    put(img, 5, 3, color)
    return img


def make_queen(color):
    img = blank()
    # Crown base
    for x in range(2, 7):
        put(img, x, 6, color)
    # Middle body
    for y in range(3, 6):
        put(img, 3, y, color)
        put(img, 5, y, color)
    # Crown points
    put(img, 3, 2, color)
    put(img, 4, 1, color)
    put(img, 5, 2, color)
    return img


def make_king(color):
    img = blank()
    # Base
    for x in range(2, 7):
        put(img, x, 6, color)
    # Body
    for y in range(3, 6):
        put(img, 4, y, color)
        put(img, 3, y, color)
        put(img, 5, y, color)
    # Cross on top
    put(img, 4, 1, color)
    put(img, 4, 2, color)
    put(img, 3, 2, color)
    put(img, 5, 2, color)
    return img


def make_cursor():
    img = blank()
    # Orange X cursor
    for i in range(SIZE):
        put(img, i, i, ORANGE)
        put(img, SIZE - 1 - i, i, ORANGE)
    # Optionally thin it a bit by clearing corners
    put(img, 0, 0, TRANSPARENT)
    put(img, 0, SIZE - 1, TRANSPARENT)
    put(img, SIZE - 1, 0, TRANSPARENT)
    put(img, SIZE - 1, SIZE - 1, TRANSPARENT)
    return img


def main():
    # 0: empty (fully transparent)
    save_sprite(0, blank())

    # Top side (opponent) → ORANGE → indices 1–6
    save_sprite(1, make_pawn(ORANGE))
    save_sprite(2, make_knight(ORANGE))
    save_sprite(3, make_bishop(ORANGE))
    save_sprite(4, make_rook(ORANGE))
    save_sprite(5, make_queen(ORANGE))
    save_sprite(6, make_king(ORANGE))

    # Bottom side (you) → WHITE → indices 7–12
    save_sprite(7, make_pawn(WHITE))
    save_sprite(8, make_knight(WHITE))
    save_sprite(9, make_bishop(WHITE))
    save_sprite(10, make_rook(WHITE))
    save_sprite(11, make_queen(WHITE))
    save_sprite(12, make_king(WHITE))

    # Cursor
    save_sprite(13, make_cursor())


if __name__ == "__main__":
    main()