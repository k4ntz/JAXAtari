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
