import os
import numpy as np
from PIL import Image, ImageDraw

# Create output directory - use absolute path to be safe
sprites_dir = os.path.abspath("sprites/airraid")
os.makedirs(sprites_dir, exist_ok=True)
print(f"Saving sprites to: {sprites_dir}")

# Define a function to create simple sprite rectangles
def create_simple_sprite(height, width, color, details=None):
    """
    Create a sprite with the given dimensions and color
    
    Args:
        height: Sprite height in pixels
        width: Sprite width in pixels
        color: Base RGB color tuple
        details: Optional function to add details to the sprite
    
    Returns:
        NumPy array with shape (height, width, 4) including alpha channel
    """
    # Create PIL Image for better drawing capabilities
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw base shape
    draw.rectangle([(0, 0), (width-1, height-1)], fill=(*color, 255))
    
    # Apply custom details if provided
    if details:
        details(draw, width, height)
    
    # Convert to numpy array
    sprite = np.array(img)
    return sprite

# Helper functions to add details to sprites
def player_details(draw, width, height):
    # Add cockpit
    draw.rectangle([(width//3, 0), (2*width//3, height//4)], fill=(200, 200, 200, 255))
    # Add wings
    draw.polygon([(0, height*3//4), (width//4, height//2), (3*width//4, height//2), (width, height*3//4)], 
                 fill=(190, 190, 190, 255))

def enemy_details(draw, width, height, enemy_type):
    if enemy_type == 25:
        # Circular enemy
        draw.ellipse([(width//4, height//4), (3*width//4, 3*height//4)], fill=(150, 150, 150, 255))
    elif enemy_type == 50:
        # X-shaped enemy
        draw.polygon([(0, 0), (width//3, height//3), (width//2, 0), 
                      (2*width//3, height//3), (width, 0), 
                      (2*width//3, height//2), (width, height), 
                      (2*width//3, 2*height//3), (width//2, height),
                      (width//3, 2*height//3), (0, height),
                      (width//3, height//2)], fill=(140, 140, 140, 255))
    elif enemy_type == 75:
        # UFO-like enemy
        draw.ellipse([(width//4, height//3), (3*width//4, 2*height//3)], fill=(86, 86, 186, 255))
        draw.rectangle([(width//3, height//6), (2*width//3, height//3)], fill=(100, 100, 200, 255))
    else:  # enemy_type == 100
        # Diamond enemy
        draw.polygon([(width//2, 0), (width, height//2), (width//2, height), (0, height//2)], 
                     fill=(150, 150, 150, 255))

def building_details(draw, width, height):
    # Windows
    window_size = min(width, height) // 8
    spacing = window_size * 2
    
    for y in range(window_size, height-window_size, spacing):
        for x in range(window_size, width-window_size, spacing):
            draw.rectangle([(x, y), (x+window_size, y+window_size)], fill=(180, 180, 180, 255))
    
    # Roof details
    draw.rectangle([(0, 0), (width, height//10)], fill=(100, 100, 100, 255))

# Create player sprite with details
print("Creating player sprite...")
player = create_simple_sprite(12, 14, (169, 169, 169), player_details)
np.save(os.path.join(sprites_dir, "player.npy"), player)
print("Saved player sprite")

# Save as PNG for inspection
Image.fromarray(player).save(os.path.join(sprites_dir, "player.png"))

# Create building sprite with details
print("Creating building sprite...")
building = create_simple_sprite(32, 32, (114, 114, 114), building_details)
np.save(os.path.join(sprites_dir, "building.npy"), building)
print("Saved building sprite")
Image.fromarray(building).save(os.path.join(sprites_dir, "building.png"))

# Create enemy sprites with consistent dimensions but different details
enemy_types = {
    "enemy25": (18, 18, (135, 135, 135), lambda d, w, h: enemy_details(d, w, h, 25)),
    "enemy50": (18, 18, (129, 129, 129), lambda d, w, h: enemy_details(d, w, h, 50)),
    "enemy75": (18, 18, (86, 86, 186), lambda d, w, h: enemy_details(d, w, h, 75)),
    "enemy100": (18, 18, (137, 137, 137), lambda d, w, h: enemy_details(d, w, h, 100))
}

for name, (h, w, color, detail_func) in enemy_types.items():
    print(f"Creating {name} sprite...")
    enemy = create_simple_sprite(h, w, color, detail_func)
    np.save(os.path.join(sprites_dir, f"{name}.npy"), enemy)
    print(f"Saved {name} sprite")
    Image.fromarray(enemy).save(os.path.join(sprites_dir, f"{name}.png"))

# Create missile sprite (white small rectangle)
print("Creating missile sprite...")
missile = create_simple_sprite(2, 2, (236, 236, 236))
np.save(os.path.join(sprites_dir, "missile.npy"), missile)
print("Saved missile sprite")
Image.fromarray(missile).save(os.path.join(sprites_dir, "missile.png"))

# Create background (black)
print("Creating background sprite...")
background = create_simple_sprite(210, 160, (0, 0, 0))
np.save(os.path.join(sprites_dir, "background.npy"), background)
print("Saved background sprite")
Image.fromarray(background).save(os.path.join(sprites_dir, "background.png"))

# Create score digits (0-9)
print("Creating score digits...")
for i in range(10):
    # Create a digit sprite (9x8)
    digit = np.zeros((9, 8, 4), dtype=np.uint8)
    
    # Different pattern for each digit using PIL
    img = Image.new('RGBA', (8, 9), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    if i == 0:  # "0"
        draw.rectangle([(1, 1), (6, 7)], outline=(236, 236, 236, 255), width=1)
    elif i == 1:  # "1"
        draw.line([(4, 1), (4, 7)], fill=(236, 236, 236, 255), width=1)
    elif i == 2:  # "2"
        draw.line([(1, 1), (6, 1)], fill=(236, 236, 236, 255), width=1)  # Top
        draw.line([(6, 1), (6, 4)], fill=(236, 236, 236, 255), width=1)  # Right top
        draw.line([(1, 4), (6, 4)], fill=(236, 236, 236, 255), width=1)  # Middle
        draw.line([(1, 4), (1, 7)], fill=(236, 236, 236, 255), width=1)  # Left bottom
        draw.line([(1, 7), (6, 7)], fill=(236, 236, 236, 255), width=1)  # Bottom
    elif i == 3:  # "3"
        draw.line([(1, 1), (6, 1)], fill=(236, 236, 236, 255), width=1)  # Top
        draw.line([(1, 4), (6, 4)], fill=(236, 236, 236, 255), width=1)  # Middle
        draw.line([(1, 7), (6, 7)], fill=(236, 236, 236, 255), width=1)  # Bottom
        draw.line([(6, 1), (6, 7)], fill=(236, 236, 236, 255), width=1)  # Right
    elif i == 4:  # "4"
        draw.line([(1, 1), (1, 4)], fill=(236, 236, 236, 255), width=1)  # Left top
        draw.line([(1, 4), (6, 4)], fill=(236, 236, 236, 255), width=1)  # Middle
        draw.line([(6, 1), (6, 7)], fill=(236, 236, 236, 255), width=1)  # Right
    elif i == 5:  # "5"
        draw.line([(1, 1), (6, 1)], fill=(236, 236, 236, 255), width=1)  # Top
        draw.line([(1, 1), (1, 4)], fill=(236, 236, 236, 255), width=1)  # Left top
        draw.line([(1, 4), (6, 4)], fill=(236, 236, 236, 255), width=1)  # Middle
        draw.line([(6, 4), (6, 7)], fill=(236, 236, 236, 255), width=1)  # Right bottom
        draw.line([(1, 7), (6, 7)], fill=(236, 236, 236, 255), width=1)  # Bottom
    elif i == 6:  # "6"
        draw.line([(1, 1), (6, 1)], fill=(236, 236, 236, 255), width=1)  # Top
        draw.line([(1, 1), (1, 7)], fill=(236, 236, 236, 255), width=1)  # Left
        draw.line([(1, 4), (6, 4)], fill=(236, 236, 236, 255), width=1)  # Middle
        draw.line([(6, 4), (6, 7)], fill=(236, 236, 236, 255), width=1)  # Right bottom
        draw.line([(1, 7), (6, 7)], fill=(236, 236, 236, 255), width=1)  # Bottom
    elif i == 7:  # "7"
        draw.line([(1, 1), (6, 1)], fill=(236, 236, 236, 255), width=1)  # Top
        draw.line([(6, 1), (6, 7)], fill=(236, 236, 236, 255), width=1)  # Right
    elif i == 8:  # "8"
        draw.rectangle([(1, 1), (6, 7)], outline=(236, 236, 236, 255), width=1)
        draw.line([(1, 4), (6, 4)], fill=(236, 236, 236, 255), width=1)  # Middle
    else:  # i == 9, "9"
        draw.line([(1, 1), (6, 1)], fill=(236, 236, 236, 255), width=1)  # Top
        draw.line([(1, 1), (1, 4)], fill=(236, 236, 236, 255), width=1)  # Left top
        draw.line([(6, 1), (6, 7)], fill=(236, 236, 236, 255), width=1)  # Right
        draw.line([(1, 4), (6, 4)], fill=(236, 236, 236, 255), width=1)  # Middle
        draw.line([(1, 7), (6, 7)], fill=(236, 236, 236, 255), width=1)  # Bottom
    
    # Convert to numpy array
    digit = np.array(img)
    np.save(os.path.join(sprites_dir, f"score_{i}.npy"), digit)
    print(f"Saved score_{i} sprite")
    Image.fromarray(digit).save(os.path.join(sprites_dir, f"score_{i}.png"))

# Create damaged building variants
print("Creating damaged building variants...")
for damage in range(1, 15):  # Damage levels 1-14
    # Calculate height based on damage level
    if damage < 4:
        height = 29  # Slightly damaged
    elif damage < 5:
        height = 27  # More damaged
    elif damage < 6:
        height = 23  # Even more damaged
    elif damage < 7:
        height = 21  # Heavily damaged
    elif damage < 8:
        height = 19  # Very heavily damaged
    elif damage < 9:
        height = 15  # Extremely damaged
    elif damage < 10:
        height = 19  # Partial rebuild
    elif damage < 11:
        height = 23  # More rebuilt
    elif damage < 13:
        height = 25  # Almost rebuilt
    elif damage < 14:
        height = 8   # Nearly destroyed
    else:
        height = 32  # Fully rebuilt
    
    # Create a damaged version of the building
    damaged_building = create_simple_sprite(height, 32, (114, 114, 114), 
                                           lambda d, w, h: building_details(d, w, h))
    np.save(os.path.join(sprites_dir, f"building_d{damage}.npy"), damaged_building)
    Image.fromarray(damaged_building).save(os.path.join(sprites_dir, f"building_d{damage}.png"))

print("\nAll sprites created successfully!")
print(f"Sprites saved to: {sprites_dir}")
print("You can now use these sprites in your Air Raid game.")