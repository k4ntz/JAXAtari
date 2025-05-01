import os
import numpy as np
import jax.numpy as jnp

# Define game dimensions
WIDTH = 160
HEIGHT = 210

# Define colors
BACKGROUND_COLOR = (144, 72, 17)  # Brown background
PLAYER_COLOR = (169, 169, 169)    # Gray player
BUILDING_COLOR = (114, 114, 114)  # Darker gray buildings
MISSILE_COLOR = (236, 236, 236)   # White missiles
SCORE_COLOR = (236, 236, 236)     # White score digits

# Create sprites directory if it doesn't exist
sprites_dir = os.path.join("sprites", "airraid")
os.makedirs(sprites_dir, exist_ok=True)

# Define sprite dimensions from constants
PLAYER_WIDTH, PLAYER_HEIGHT = 14, 12
BUILDING_WIDTH, BUILDING_HEIGHT = 32, 32
ENEMY_TYPES = {
    0: (16, 18, (135, 135, 135)),   # Enemy25
    1: (14, 16, (129, 129, 129)),   # Enemy50
    2: (14, 16, (86, 86, 186)),     # Enemy75 - blue tint
    3: (14, 14, (137, 137, 137))    # Enemy100
}
MISSILE_WIDTH, MISSILE_HEIGHT = 2, 2

# Create background (brown color)
background = np.ones((HEIGHT, WIDTH, 4), dtype=np.uint8)
background[:, :, :3] = np.array(BACKGROUND_COLOR, dtype=np.uint8)
background[:, :, 3] = 255  # Full opacity
np.save(os.path.join(sprites_dir, "background.npy"), background)

# Create player sprite (gray rectangle with some detail)
player = np.zeros((PLAYER_HEIGHT, PLAYER_WIDTH, 4), dtype=np.uint8)
player[:, :, :3] = PLAYER_COLOR
player[:, :, 3] = 255  # Full opacity
# Add a little detail to make it more ship-like
player[0:2, PLAYER_WIDTH//3:PLAYER_WIDTH*2//3, :3] = [200, 200, 200]  # cockpit
np.save(os.path.join(sprites_dir, "player.npy"), player)

# Create building sprite (gray rectangle with some detail)
building = np.zeros((BUILDING_HEIGHT, BUILDING_WIDTH, 4), dtype=np.uint8)
building[:, :, :3] = BUILDING_COLOR
building[:, :, 3] = 255  # Full opacity
# Add some windows/detail
for i in range(3):
    for j in range(5):
        if (i + j) % 2 == 0:  # Checker pattern for windows
            building[5+i*8:9+i*8, 4+j*6:8+j*6, :3] = [180, 180, 180]
np.save(os.path.join(sprites_dir, "building.npy"), building)

# Create enemy sprites
for enemy_type, (width, height, color) in ENEMY_TYPES.items():
    enemy = np.zeros((height, width, 4), dtype=np.uint8)
    enemy[:, :, :3] = color
    enemy[:, :, 3] = 255  # Full opacity
    
    # Add some detail to distinguish enemy types
    if enemy_type == 0:  # Enemy25
        enemy[height//3:height*2//3, 3:width-3, :3] = [color[0]+30, color[1]+30, color[2]+30]
    elif enemy_type == 1:  # Enemy50
        enemy[2:height-2, width//3:width*2//3, :3] = [color[0]+30, color[1]+30, color[2]+30]
    elif enemy_type == 2:  # Enemy75
        enemy[height//4:height*3//4, width//4:width*3//4, :3] = [color[0]+30, color[1]+30, color[2]+30]
    else:  # Enemy100
        enemy[2:height-2, 2:width-2, :3] = [color[0]+30, color[1]+30, color[2]+30]
        enemy[4:height-4, 4:width-4, :3] = color
        
    np.save(os.path.join(sprites_dir, f"enemy{25 * (enemy_type + 1)}.npy"), enemy)

# Create missile sprite (white rectangle)
missile = np.zeros((MISSILE_HEIGHT, MISSILE_WIDTH, 4), dtype=np.uint8)
missile[:, :, :3] = MISSILE_COLOR
missile[:, :, 3] = 255  # Full opacity
np.save(os.path.join(sprites_dir, "missile.npy"), missile)

# Create digit sprites (0-9)
for i in range(10):
    # Create a simple 8x9 digit representation
    digit = np.zeros((9, 8, 4), dtype=np.uint8)
    digit[:, :, 3] = 0  # Start with full transparency
    
    # Different pattern for each digit
    if i == 0:
        digit[1:-1, 1:-1, :3] = SCORE_COLOR
        digit[1:-1, 1:-1, 3] = 255  # Make visible pixels opaque
        digit[2:-2, 2:-2, 3] = 0  # Make center transparent
    elif i == 1:
        digit[1:-1, 4:6, :3] = SCORE_COLOR
        digit[1:-1, 4:6, 3] = 255
    elif i == 2:
        # Set all digit pixels first as transparent
        # Then set only the visible parts as opaque
        digit[1:3, 1:-1, :3] = SCORE_COLOR  # Top horizontal
        digit[1:3, 1:-1, 3] = 255
        digit[4:6, 1:-1, :3] = SCORE_COLOR  # Middle horizontal
        digit[4:6, 1:-1, 3] = 255
        digit[7:-1, 1:-1, :3] = SCORE_COLOR  # Bottom horizontal
        digit[7:-1, 1:-1, 3] = 255
        digit[3:5, 5:-1, :3] = SCORE_COLOR  # Top-right vertical
        digit[3:5, 5:-1, 3] = 255
        digit[5:8, 1:3, :3] = SCORE_COLOR   # Bottom-left vertical
        digit[5:8, 1:3, 3] = 255
    elif i == 3:
        digit[1:3, 1:-1, :3] = SCORE_COLOR  # Top horizontal
        digit[1:3, 1:-1, 3] = 255
        digit[4:6, 1:-1, :3] = SCORE_COLOR  # Middle horizontal
        digit[4:6, 1:-1, 3] = 255
        digit[7:-1, 1:-1, :3] = SCORE_COLOR  # Bottom horizontal
        digit[7:-1, 1:-1, 3] = 255
        digit[1:8, 5:-1, :3] = SCORE_COLOR  # Right vertical
        digit[1:8, 5:-1, 3] = 255
    elif i == 4:
        digit[1:5, 1:3, :3] = SCORE_COLOR   # Top-left vertical
        digit[1:5, 1:3, 3] = 255
        digit[4:6, 1:-1, :3] = SCORE_COLOR  # Middle horizontal
        digit[4:6, 1:-1, 3] = 255
        digit[1:8, 5:-1, :3] = SCORE_COLOR  # Right vertical
        digit[1:8, 5:-1, 3] = 255
    elif i == 5:
        digit[1:3, 1:-1, :3] = SCORE_COLOR  # Top horizontal
        digit[1:3, 1:-1, 3] = 255
        digit[4:6, 1:-1, :3] = SCORE_COLOR  # Middle horizontal
        digit[4:6, 1:-1, 3] = 255
        digit[7:-1, 1:-1, :3] = SCORE_COLOR  # Bottom horizontal
        digit[7:-1, 1:-1, 3] = 255
        digit[1:5, 1:3, :3] = SCORE_COLOR   # Top-left vertical
        digit[1:5, 1:3, 3] = 255
        digit[5:8, 5:-1, :3] = SCORE_COLOR  # Bottom-right vertical
        digit[5:8, 5:-1, 3] = 255
    elif i == 6:
        digit[1:3, 1:-1, :3] = SCORE_COLOR  # Top horizontal
        digit[1:3, 1:-1, 3] = 255
        digit[4:6, 1:-1, :3] = SCORE_COLOR  # Middle horizontal
        digit[4:6, 1:-1, 3] = 255
        digit[7:-1, 1:-1, :3] = SCORE_COLOR  # Bottom horizontal
        digit[7:-1, 1:-1, 3] = 255
        digit[1:8, 1:3, :3] = SCORE_COLOR   # Left vertical
        digit[1:8, 1:3, 3] = 255
        digit[5:8, 5:-1, :3] = SCORE_COLOR  # Bottom-right vertical
        digit[5:8, 5:-1, 3] = 255
    elif i == 7:
        digit[1:3, 1:-1, :3] = SCORE_COLOR  # Top horizontal
        digit[1:3, 1:-1, 3] = 255
        digit[1:8, 5:-1, :3] = SCORE_COLOR  # Right vertical
        digit[1:8, 5:-1, 3] = 255
    elif i == 8:
        digit[1:3, 1:-1, :3] = SCORE_COLOR  # Top horizontal
        digit[1:3, 1:-1, 3] = 255
        digit[4:6, 1:-1, :3] = SCORE_COLOR  # Middle horizontal
        digit[4:6, 1:-1, 3] = 255
        digit[7:-1, 1:-1, :3] = SCORE_COLOR  # Bottom horizontal
        digit[7:-1, 1:-1, 3] = 255
        digit[1:8, 1:3, :3] = SCORE_COLOR   # Left vertical
        digit[1:8, 1:3, 3] = 255
        digit[1:8, 5:-1, :3] = SCORE_COLOR  # Right vertical
        digit[1:8, 5:-1, 3] = 255
    elif i == 9:
        digit[1:3, 1:-1, :3] = SCORE_COLOR  # Top horizontal
        digit[1:3, 1:-1, 3] = 255
        digit[4:6, 1:-1, :3] = SCORE_COLOR  # Middle horizontal
        digit[4:6, 1:-1, 3] = 255
        digit[7:-1, 1:-1, :3] = SCORE_COLOR  # Bottom horizontal
        digit[7:-1, 1:-1, 3] = 255
        digit[1:5, 1:3, :3] = SCORE_COLOR   # Top-left vertical
        digit[1:5, 1:3, 3] = 255
        digit[1:8, 5:-1, :3] = SCORE_COLOR  # Right vertical
        digit[1:8, 5:-1, 3] = 255
        
    np.save(os.path.join(sprites_dir, f"score_{i}.npy"), digit)

print(f"Sprite generation complete. Files saved to {os.path.abspath(sprites_dir)}")