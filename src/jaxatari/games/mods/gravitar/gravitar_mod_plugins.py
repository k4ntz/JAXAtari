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
        "SOLAR_GRAVITY": -0.044,
        "PLANETARY_GRAVITY": -0.0032,
        "REACTOR_GRAVITY": -0.0001,
    }


class HighSpeedMod(JaxAtariInternalModPlugin):
    """Make the ship faster by increasing thrust power and max speed."""

    constants_overrides = {
        "THRUST_POWER": 0.07,
        "MAX_SPEED": 6.0,
    }
