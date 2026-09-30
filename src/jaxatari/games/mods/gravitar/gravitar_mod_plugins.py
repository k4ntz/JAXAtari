from jaxatari.modification import JaxAtariInternalModPlugin


class RapidFireMod(JaxAtariInternalModPlugin):
    """Increase active bullet caps for ship, saucer, and enemies to 4."""

    constants_overrides = {
        "MAX_ACTIVE_PLAYER_BULLETS_MAP": 4,
        "MAX_ACTIVE_PLAYER_BULLETS_LEVEL": 4,
        "MAX_ACTIVE_PLAYER_BULLETS_ARENA": 4,
        "MAX_ACTIVE_SAUCER_BULLETS": 4,
        "MAX_ACTIVE_ENEMY_BULLETS": 8,  # 2 bullets per enemy * 4 enemies
    }


class ZeroGravityMod(JaxAtariInternalModPlugin):
    """Disable all gravity from sun, planets, and reactors."""

    constants_overrides = {
        "SOLAR_GRAVITY": 0.0,
        "PLANETARY_GRAVITY": 0.0,
        "REACTOR_GRAVITY": 0.0,
    }


class HyperGravityMod(JaxAtariInternalModPlugin):
    """Increase all gravity from sun, planets, and reactors substantially."""

    constants_overrides = {
        "SOLAR_GRAVITY": 0.132,  # 0.044 * 3
        "PLANETARY_GRAVITY": 0.0096,  # 0.0032 * 3
        "REACTOR_GRAVITY": 0.001,  # 0.0001 * 10
    }


class FuelCrisisMod(JaxAtariInternalModPlugin):
    """Increase fuel consumption rate by 5x."""

    constants_overrides = {
        "FUEL_CONSUME_THRUST": 20.0,
        "FUEL_CONSUME_SHIELD_TRACTOR": 50.0,
    }

class HarmlessEnemiesMod(JaxAtariInternalModPlugin):
    """Make all enemies harmless by disabling their bullets."""

    constants_overrides = {
        "MAX_ACTIVE_ENEMY_BULLETS": 0,
        "MAX_ACTIVE_SAUCER_BULLETS": 0,
    }


class ValuableReactorMod(JaxAtariInternalModPlugin):
    """Populate reactor level with 3 enemies and 2 fuel tanks."""

    constants_overrides = {
        "ALLOW_TRACTOR_IN_REACTOR": True,
        "REACTOR_LEVEL_LAYOUT": (
            {"type": 5, "coords": (104, 104)},   # ENEMY_ORANGE
            {"type": 6, "coords": (56, 144)},   # ENEMY_GREEN
            {"type": 39, "coords": (80, 18)},  # ENEMY_ORANGE_FLIPPED
            {"type": 12, "coords": (66, 88)}, # FUEL_TANK
        ),
    }


class AntiGravityMod(JaxAtariInternalModPlugin):
    """Reverse gravity from sun, planets, and reactors."""

    constants_overrides = {
        "SOLAR_GRAVITY": -0.005,
        "PLANETARY_GRAVITY": -0.0032,
        "REACTOR_GRAVITY": -0.0001,
    }


class HighSpeedMod(JaxAtariInternalModPlugin):
    """Make the ship faster by increasing thrust power and max speed."""

    constants_overrides = {
        "THRUST_POWER": 0.075,
        "MAX_SPEED": 6.0,
    }


class InfiniteFuelMod(JaxAtariInternalModPlugin):
    """Disable fuel consumption."""

    constants_overrides = {
        "FUEL_CONSUME_THRUST": 0.0,
        "FUEL_CONSUME_SHIELD_TRACTOR": 0.0,
    }


class SlowEnemiesMod(JaxAtariInternalModPlugin):
    """Decrease the movement speed of saucers and bullets."""

    constants_overrides = {
        "SAUCER_SPEED_MAP": 0.09,
        "SAUCER_SPEED_ARENA": 0.18,
        "SAUCER_BULLET_SPEED": 1.0,
        "ENEMY_BULLET_SPEED": 0.65,
    }


class LongRangeTractorMod(JaxAtariInternalModPlugin):
    """Increase the range of the tractor beam."""

    constants_overrides = {
        "TRACTOR_BEAM_RANGE": 50.0,
    }

class NeonMod(JaxAtariInternalModPlugin):
    """
    Changes colors to bright neon variants.
    """
    constants_overrides = {
        "RECOLOR_RULES": (
            {"source": (101, 183, 217), "target": (255, 20, 147)},
            {"source": (198, 108, 58), "target": (0, 255, 0)},
            {"source": (72, 160, 72), "target": (255, 255, 0)},
            {"source": (223, 183, 85), "target": (0, 255, 255)},
        )
    }

class RedAlertMod(JaxAtariInternalModPlugin):
    """
    Makes all terrain red/orange for a high-alert aesthetic.
    """
    constants_overrides = {
        "RECOLOR_RULES": (
            {"source": (223, 183, 85), "target": (255, 50, 50)},
            {"source": (84, 160, 197), "target": (220, 40, 40)},
            {"source": (66, 72, 200), "target": (200, 30, 30)},
            {"source": (213, 130, 74), "target": (255, 0, 0)},
        )
    }

class GrayscaleMod(JaxAtariInternalModPlugin):
    """
    Converts the visual palette to grayscale.
    """
    constants_overrides = {
        "RECOLOR_RULES": (
            {"source": (223, 183, 85), "target": (150, 150, 150)},
            {"source": (84, 160, 197), "target": (120, 120, 120)},
            {"source": (66, 72, 200), "target": (80, 80, 80)},
            {"source": (228, 111, 111), "target": (140, 140, 140)},
            {"source": (213, 130, 74), "target": (160, 160, 160)},
            {"source": (101, 183, 217), "target": (220, 220, 220)},
            {"source": (198, 108, 58), "target": (110, 110, 110)},
            {"source": (72, 160, 72), "target": (90, 90, 90)},
        )
    }

class InvertedColorsMod(JaxAtariInternalModPlugin):
    """
    Inverts the primary colors.
    """
    constants_overrides = {
        "RECOLOR_RULES": (
            {"source": (101, 183, 217), "target": (154, 72, 38)}, 
            {"source": (223, 183, 85), "target": (32, 72, 170)},  
            {"source": (84, 160, 197), "target": (171, 95, 58)},
        )
    }
