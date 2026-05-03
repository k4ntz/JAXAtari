from typing import Dict, Any, Tuple
from jaxatari.modification import JaxAtariInternalModPlugin

class NightMod(JaxAtariInternalModPlugin):
    """Dims the entire screen by 50% for a night mode experience."""
    name = "night_mode"
    
    # 50% of the original values based on user constraints
    constants_overrides = {
        'RGB_BACKGROUND': (0, 0, 0),
        'RGB_PLAYER_DETAILED': (83, 13, 13),
        'RGB_W1_WALLS': (84, 24, 71),
        'RGB_W2_WALLS': (22, 43, 88),
        'RGB_MONSTER_W1_MAP': (41, 63, 22),
        'RGB_MONSTER_W1_R2': (41, 63, 22),
        'RGB_MONSTER_W1_R3': (39, 25, 90),
        'RGB_MONSTER_W1_R4': (55, 55, 55),
        'RGB_MONSTER_W2_MAP': (90, 41, 20),
        'RGB_MONSTER_W2_R1': (92, 25, 25),
        'RGB_MONSTER_W2_R2': (55, 55, 55),
        'RGB_MONSTER_W2_R3': (67, 67, 14),
        'RGB_MONSTER_W2_R4': (90, 41, 20),
    }

class GrayscaleMod(JaxAtariInternalModPlugin):
    """Turns the entire game into grayscale."""
    name = "grayscale"
    
    # Using luminosity method (0.3R + 0.59G + 0.11B)
    constants_overrides = {
        'RGB_BACKGROUND': (0, 0, 0),
        'RGB_PLAYER_DETAILED': (65, 65, 65),
        'RGB_W1_WALLS': (94, 94, 94),
        'RGB_W2_WALLS': (84, 84, 84),
        'RGB_MONSTER_W1_MAP': (103, 103, 103),
        'RGB_MONSTER_W1_R2': (103, 103, 103),
        'RGB_MONSTER_W1_R3': (72, 72, 72),
        'RGB_MONSTER_W1_R4': (111, 111, 111),
        'RGB_MONSTER_W2_MAP': (107, 107, 107),
        'RGB_MONSTER_W2_R1': (90, 90, 90),
        'RGB_MONSTER_W2_R2': (111, 111, 111),
        'RGB_MONSTER_W2_R3': (119, 119, 119),
        'RGB_MONSTER_W2_R4': (107, 107, 107),
    }

class InvertedColorsMod(JaxAtariInternalModPlugin):
    """Inverts all colors in the game."""
    name = "inverted_colors"
    
    constants_overrides = {
        'RGB_BACKGROUND': (255, 255, 255),
        'RGB_PLAYER_DETAILED': (88, 229, 229),
        'RGB_W1_WALLS': (87, 207, 112),
        'RGB_W2_WALLS': (210, 168, 79),
        'RGB_MONSTER_W1_MAP': (173, 129, 210),
        'RGB_MONSTER_W1_R2': (173, 129, 210),
        'RGB_MONSTER_W1_R3': (177, 205, 74),
        'RGB_MONSTER_W1_R4': (144, 144, 144),
        'RGB_MONSTER_W2_MAP': (74, 172, 215),
        'RGB_MONSTER_W2_R1': (71, 205, 205),
        'RGB_MONSTER_W2_R2': (144, 144, 144),
        'RGB_MONSTER_W2_R3': (121, 121, 226),
        'RGB_MONSTER_W2_R4': (74, 172, 215),
    }

class MatrixMod(JaxAtariInternalModPlugin):
    """A Matrix-themed mod: black background, green walls, green monsters, white player."""
    name = "matrix_theme"
    
    constants_overrides = {
        'RGB_BACKGROUND': (0, 0, 0),
        'RGB_PLAYER_DETAILED': (255, 255, 255),
        'RGB_W1_WALLS': (0, 200, 0),
        'RGB_W2_WALLS': (0, 150, 0),
        'RGB_MONSTER_W1_MAP': (50, 255, 50),
        'RGB_MONSTER_W1_R2': (50, 255, 50),
        'RGB_MONSTER_W1_R3': (0, 255, 100),
        'RGB_MONSTER_W1_R4': (0, 180, 0),
        'RGB_MONSTER_W2_MAP': (50, 255, 50),
        'RGB_MONSTER_W2_R1': (0, 255, 100),
        'RGB_MONSTER_W2_R2': (0, 180, 0),
        'RGB_MONSTER_W2_R3': (100, 255, 100),
        'RGB_MONSTER_W2_R4': (50, 255, 50),
    }

class BloodMoonMod(JaxAtariInternalModPlugin):
    """A dark red themed mod."""
    name = "blood_moon"
    
    constants_overrides = {
        'RGB_BACKGROUND': (40, 0, 0),
        'RGB_PLAYER_DETAILED': (255, 255, 255),
        'RGB_W1_WALLS': (180, 0, 0),
        'RGB_W2_WALLS': (150, 0, 0),
        'RGB_MONSTER_W1_MAP': (255, 100, 100),
        'RGB_MONSTER_W1_R2': (255, 100, 100),
        'RGB_MONSTER_W1_R3': (255, 50, 50),
        'RGB_MONSTER_W1_R4': (200, 50, 50),
        'RGB_MONSTER_W2_MAP': (255, 100, 100),
        'RGB_MONSTER_W2_R1': (255, 50, 50),
        'RGB_MONSTER_W2_R2': (200, 50, 50),
        'RGB_MONSTER_W2_R3': (255, 150, 150),
        'RGB_MONSTER_W2_R4': (255, 100, 100),
    }