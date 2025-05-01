import os
import numpy as np
# from jaxatari.rendering import atraJAxis as aj
# import jaxatari.rendering.atraJaxis as aj

# Create output directory - use absolute path to be safe
sprites_dir = os.path.abspath("sprites/airraid")
os.makedirs(sprites_dir, exist_ok=True)
print(f"Saving sprites to: {sprites_dir}")

# Define a function to create simple sprite rectangles
def create_simple_sprite(height, width, color):
    """Create a simple rectangular sprite with given dimensions and color"""
    sprite = np.zeros((height, width, 4), dtype=np.uint8)
    sprite[:, :, :3] = color  # Set RGB values
    sprite[:, :, 3] = 255     # Set alpha (fully opaque)
    return sprite

# Create player sprite (gray rectangle)
print("Creating player sprite...")
player = create_simple_sprite(12, 14, (169, 169, 169))
np.save(os.path.join(sprites_dir, "player.npy"), player)
print("Saved player sprite")

# Create building sprite (darker gray rectangle)
print("Creating building sprite...")
building = create_simple_sprite(32, 32, (114, 114, 114))
np.save(os.path.join(sprites_dir, "building.npy"), building)
print("Saved building sprite")

# Create enemy sprites
enemy_types = {
    "enemy25": (18, 16, (135, 135, 135)),
    "enemy50": (16, 14, (129, 129, 129)),
    "enemy75": (16, 14, (86, 86, 186)),
    "enemy100": (14, 14, (137, 137, 137))
}

for name, (h, w, color) in enemy_types.items():
    print(f"Creating {name} sprite...")
    enemy = create_simple_sprite(h, w, color)
    np.save(os.path.join(sprites_dir, f"{name}.npy"), enemy)
    print(f"Saved {name} sprite")

# Create missile sprite (white small rectangle)
print("Creating missile sprite...")
missile = create_simple_sprite(2, 2, (236, 236, 236))
np.save(os.path.join(sprites_dir, "missile.npy"), missile)
print("Saved missile sprite")

# Create background (black)
print("Creating background sprite...")
background = create_simple_sprite(210, 160, (0, 0, 0))
np.save(os.path.join(sprites_dir, "background.npy"), background)
print("Saved background sprite")

# Create score digits (0-9)
print("Creating score digits...")
for i in range(10):
    # Create a simple digit sprite (9x8)
    digit = np.zeros((9, 8, 4), dtype=np.uint8)
    digit[:, :, 3] = 0  # Transparent by default
    
    # Different pattern for each digit
    if i == 0:  # "0"
        digit[1:-1, 1:-1, :3] = (236, 236, 236)
        digit[1:-1, 1:-1, 3] = 255
        digit[3:-3, 3:-3, 3] = 0
    elif i == 1:  # "1"
        digit[1:-1, 3:5, :3] = (236, 236, 236)
        digit[1:-1, 3:5, 3] = 255
    elif i == 2:  # "2"
        digit[1:3, 1:-1, :3] = (236, 236, 236)  # Top
        digit[1:3, 1:-1, 3] = 255
        digit[4:6, 1:-1, :3] = (236, 236, 236)  # Middle
        digit[4:6, 1:-1, 3] = 255
        digit[7:-1, 1:-1, :3] = (236, 236, 236)  # Bottom
        digit[7:-1, 1:-1, 3] = 255
        digit[3:5, 5:-1, :3] = (236, 236, 236)  # Top-right
        digit[3:5, 5:-1, 3] = 255
        digit[5:8, 1:3, :3] = (236, 236, 236)  # Bottom-left
        digit[5:8, 1:3, 3] = 255
    else:
        # Simple pattern for other digits
        digit[1:-1, 1:-1, :3] = (236, 236, 236)
        digit[1:-1, 1:-1, 3] = 255
    
    np.save(os.path.join(sprites_dir, f"score_{i}.npy"), digit)
    print(f"Saved score_{i} sprite")

print(f"\nAll sprites created successfully in {sprites_dir}")
print("You can now edit them with the sprite editor if needed")