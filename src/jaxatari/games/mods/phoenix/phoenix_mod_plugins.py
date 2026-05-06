from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin
import jax.numpy as jnp
from jaxatari.games.jax_phoenix import PhoenixState


class BossLateMissilesMod(JaxAtariInternalModPlugin):
    """
    Make boss missiles appear a few pixels after spawn so they are visible
    later (e.g., after leaving dense boss-block area).
    """

    constants_overrides = {
        "BOSS_PROJECTILE_RENDER_DELAY_PX": 8,
    }


class InfiniteLivesMod(JaxAtariInternalModPlugin):
    """
    Set player lives to 99.
    """
    constants_overrides = {
        "PLAYER_LIVES": 99,
    }


class FastPlayerMod(JaxAtariInternalModPlugin):
    """
    Increases player movement speed.
    """
    constants_overrides = {
        "PLAYER_STEP_SIZE": 2,
    }


class InvinciblePlayerMod(JaxAtariPostStepModPlugin):
    """
    Player is always invincible.
    """
    def run(self, prev_state: PhoenixState, new_state: PhoenixState) -> PhoenixState:
        return new_state.replace(invincibility=jnp.array(True))


class FastEnemyBulletsMod(JaxAtariInternalModPlugin):
    """
    Increases speed of enemy projectiles.
    """
    constants_overrides = {
        "ENEMY_PROJECTILE_SPEED": 4,
    }


class NoAbilityCooldownMod(JaxAtariInternalModPlugin):
    """
    Removes cooldown for the special ability (shield).
    """
    constants_overrides = {
        "ABILITY_COOLDOWN": 0,
    }

class NightMod(JaxAtariInternalModPlugin):
    """Dims the entire screen by 50% for a night mode experience."""
    name = "night_mode"
    constants_overrides = {
        'SCORE_COLOR': (105, 105, 32),
        'PLAYER_COLOR': (106, 65, 37),
        'BOSS_BLUE_COLOR': (42, 46, 107),
        'BOSS_RED_COLOR': (100, 36, 36),
        'BOSS_GREEN_COLOR': (42, 80, 30),
        'RGB_BACKGROUND': (0, 0, 0),
        'RGB_FLOOR': (73, 35, 96),
        'RGB_PHOENIX_MAIN': (62, 24, 86),
        'RGB_BATS_BLUE': (66, 72, 126),
        'RGB_BATS_RED': (83, 13, 13),
    }

class GrayscaleMod(JaxAtariInternalModPlugin):
    """Turns the entire game into grayscale."""
    name = "grayscale"
    constants_overrides = {
        'SCORE_COLOR': (170, 170, 170),
        'PLAYER_COLOR': (120, 120, 120),
        'BOSS_BLUE_COLOR': (90, 90, 90),
        'BOSS_RED_COLOR': (100, 100, 100),
        'BOSS_GREEN_COLOR': (110, 110, 110),
        'RGB_BACKGROUND': (0, 0, 0),
        'RGB_FLOOR': (100, 100, 100),
        'RGB_PHOENIX_MAIN': (80, 80, 80),
        'RGB_BATS_BLUE': (120, 120, 120),
        'RGB_BATS_RED': (70, 70, 70),
    }

class InvertedColorsMod(JaxAtariInternalModPlugin):
    """Inverts all colors in the game."""
    name = "inverted_colors"
    constants_overrides = {
        'SCORE_COLOR': (45, 45, 191),
        'PLAYER_COLOR': (42, 125, 181),
        'BOSS_BLUE_COLOR': (171, 163, 41),
        'BOSS_RED_COLOR': (55, 183, 183),
        'BOSS_GREEN_COLOR': (171, 95, 195),
        'RGB_BACKGROUND': (255, 255, 255),
        'RGB_FLOOR': (109, 185, 63),
        'RGB_PHOENIX_MAIN': (130, 207, 82),
        'RGB_BATS_BLUE': (123, 111, 3),
        'RGB_BATS_RED': (88, 229, 229),
    }

class MatrixMod(JaxAtariInternalModPlugin):
    """A Matrix-themed mod: black background, green elements."""
    name = "matrix_theme"
    constants_overrides = {
        'SCORE_COLOR': (0, 255, 0),
        'PLAYER_COLOR': (255, 255, 255),
        'BOSS_BLUE_COLOR': (0, 150, 0),
        'BOSS_RED_COLOR': (0, 200, 0),
        'BOSS_GREEN_COLOR': (50, 255, 50),
        'RGB_BACKGROUND': (0, 0, 0),
        'RGB_FLOOR': (0, 180, 0),
        'RGB_PHOENIX_MAIN': (0, 255, 100),
        'RGB_BATS_BLUE': (0, 255, 0),
        'RGB_BATS_RED': (50, 255, 50),
    }

class BloodMoonMod(JaxAtariInternalModPlugin):
    """A dark red themed mod."""
    name = "blood_moon"
    constants_overrides = {
        'SCORE_COLOR': (255, 100, 100),
        'PLAYER_COLOR': (255, 255, 255),
        'BOSS_BLUE_COLOR': (150, 0, 0),
        'BOSS_RED_COLOR': (200, 50, 50),
        'BOSS_GREEN_COLOR': (180, 0, 0),
        'RGB_BACKGROUND': (40, 0, 0),
        'RGB_FLOOR': (200, 0, 0),
        'RGB_PHOENIX_MAIN': (255, 50, 50),
        'RGB_BATS_BLUE': (150, 0, 0),
        'RGB_BATS_RED': (255, 100, 100),
    }
