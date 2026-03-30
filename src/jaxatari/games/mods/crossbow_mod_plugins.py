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
        "DESERT_MAX_ENEMIES": 8,
        "CAVERN_MAX_BATS": 2,
        "CAVERN_MAX_STALACTITES": 4,
        "VOLCANO_MAX_ENEMIES": 4,
        "JUNGLE_MAX_ENEMIES": 6,
        "DRAWBRIDGE_MAX_ARCHERS": 8,
        "CASTLE_HALL_MAX_ENEMIES": 4,
    }

class FastCursorMod(JaxAtariInternalModPlugin):
    """Increases the cursor speed for faster aiming."""
    constants_overrides = {'CURSOR_SPEED': 4}


class LargeCursorMod(JaxAtariInternalModPlugin):
    """Increases the cursor size, making it easier to see."""
    constants_overrides = {'CURSOR_SIZE': (8, 8)}


class FastFriendMod(JaxAtariInternalModPlugin):
    """Increases the speed of the friend character."""
    constants_overrides = {'FRIEND_SPEED': 1.0}
    
    
class FastEnemyAttackMod(JaxAtariInternalModPlugin):
    """Enemies kill the friend faster by reducing the dying duration."""
    constants_overrides = {
        'DYING_DURATION': 20,  # default is 45, lower = friend dies faster
    }
    
class SlowEnemyAttackMod(JaxAtariInternalModPlugin):
    """Enemies kill the friend slower by increasing the dying duration."""
    constants_overrides = {
        'DYING_DURATION': 90,  # default is 45, higher = friend survives longer when attacked
        
    }    
        
    
    