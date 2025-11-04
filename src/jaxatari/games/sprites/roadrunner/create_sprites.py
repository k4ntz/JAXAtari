import numpy as np


def create_rgba_sprite(mask, color):
    """Converts a 2D mask into a 3D RGBA sprite."""
    height, width = mask.shape
    rgba_sprite = np.zeros((height, width, 4), dtype=np.uint8)

    sprite_color_rgba = (*color, 255)  # Add full alpha channel
    transparent_rgba = (0, 0, 0, 0)  # Transparent

    for h in range(height):
        for w in range(width):
            if mask[h, w] == 1:
                rgba_sprite[h, w] = sprite_color_rgba
            else:
                rgba_sprite[h, w] = transparent_rgba
    return rgba_sprite


# --- Corrected Sprite Masks ---

# Corrected Sprite 1: Road Runner Standing
roadrunner_stand_mask = np.array(
    [
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 1, 1],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 1, 1],
        [1, 1, 1, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 1, 0, 0, 1, 1, 1],
        [0, 0, 1, 0, 1, 1, 1, 0],
        [0, 0, 1, 0, 1, 1, 1, 1],
        [0, 0, 1, 0, 0, 1, 1, 0],
        [0, 0, 1, 0, 1, 1, 1, 1],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 1, 0, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0],
    ],
    dtype=np.uint8,
)

# Corrected Sprite 3: Road Runner Running (Frame 1)
roadrunner_run1_mask = np.array(
    [
        [0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 1, 1],
        [0, 0, 1, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 1, 0, 1, 1, 1, 1],
        [0, 1, 0, 0, 1, 1, 1, 1],
        [0, 0, 1, 1, 0, 1, 1, 0],
        [1, 1, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 1],
        [0, 0, 1, 0, 1, 1, 1, 1],
        [0, 1, 1, 0, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 1, 1, 1],
        [0, 1, 0, 0, 1, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 0, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 0],
        [0, 1, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 1],
        [0, 0, 1, 1, 0, 1, 1, 0],
        [0, 0, 1, 0, 1, 0, 1, 1],
        [0, 0, 1, 0, 0, 0, 1, 1],
        [0, 0, 1, 0, 0, 1, 0, 1],
        [0, 0, 1, 1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 0, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 1, 1, 0],
    ],
    dtype=np.uint8,
)

# Corrected Sprite 4: Road Runner Running (Frame 2)
roadrunner_run2_mask = np.array(
    [
        [0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 1, 1],
        [0, 0, 1, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 1, 0, 1, 1, 1, 1],
        [0, 1, 0, 0, 1, 1, 1, 1],
        [0, 0, 1, 1, 0, 1, 1, 0],
        [1, 1, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 1],
        [0, 0, 1, 0, 1, 1, 1, 1],
        [0, 1, 1, 0, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 1, 1, 1],
        [0, 1, 0, 0, 1, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 0, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 1],
        [1, 0, 1, 1, 0, 1, 1, 0],
        [0, 0, 1, 0, 0, 0, 1, 1],
        [0, 0, 1, 0, 1, 0, 1, 1],
        [1, 0, 1, 1, 0, 0, 0, 1],
        [0, 0, 1, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 1, 1, 0, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 1, 0],
    ],
    dtype=np.uint8,
)

if __name__ == "__main__":
    # Define the colors from the game's constants
    PLAYER_COLOR = (92, 186, 92)
    ENEMY_COLOR = (213, 130, 74)

    # Create the full RGBA sprites
    roadrunner_stand_rgba = create_rgba_sprite(roadrunner_stand_mask, PLAYER_COLOR)
    roadrunner_run1_rgba = create_rgba_sprite(roadrunner_run1_mask, PLAYER_COLOR)
    roadrunner_run2_rgba = create_rgba_sprite(roadrunner_run2_mask, PLAYER_COLOR)

    # Save the RGBA sprites
    np.save("roadrunner_stand.npy", roadrunner_stand_rgba)
    np.save("roadrunner_run1.npy", roadrunner_run1_rgba)
    np.save("roadrunner_run2.npy", roadrunner_run2_rgba)

    print(
        "Saved corrected RGBA sprites to roadrunner_stand.npy, roadrunner_run1.npy, and roadrunner_run2.npy"
    )
    print(
        f"All sprites have shape (height, width, channels): {roadrunner_stand_rgba.shape}"
    )
