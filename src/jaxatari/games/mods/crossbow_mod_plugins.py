from jaxatari.modification import JaxAtariInternalModPlugin


class CursorVisibleOnFireOnlyMod(JaxAtariInternalModPlugin):
    """Makes the cursor invisible by default, only showing it when the player fires."""

    constants_overrides = {
        "CURSOR_VISIBLE_ON_FIRE_ONLY": True,
    }


class DoubleEnemiesMod(JaxAtariInternalModPlugin):
    """Doubles the maximum number of enemies across all maps."""

    constants_overrides = {
        "MAX_ENEMIES": 12,
        "CAVERN_MAX_BATS": 2,
        "CAVERN_MAX_STALACTITES": 4,
        "VOLCANO_MAX_ENEMIES": 4,
        "JUNGLE_MAX_ENEMIES": 6,
        "DRAWBRIDGE_MAX_ARCHERS": 8,
        "CASTLE_HALL_MAX_ENEMIES": 4,
    }
