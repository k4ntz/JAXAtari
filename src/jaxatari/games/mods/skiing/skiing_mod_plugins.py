from jaxatari.modification import JaxAtariInternalModPlugin

class MoreTreesMod(JaxAtariInternalModPlugin):
    """
    Spawns more trees during the race.
    """
    constants_overrides = {
        "max_num_trees": 12,
    }

class TreesEverywhereMod(JaxAtariInternalModPlugin):
    """
    Allows trees to spawn anywhere across the entire horizontal axis,
    instead of forcing a central gap.
    """
    constants_overrides = {
        "trees_everywhere": True,
    }

class MoreMogulsMod(JaxAtariInternalModPlugin):
    """
    Spawns more moguls (rocks) during the race.
    """
    constants_overrides = {
        "max_num_moguls": 6,
    }

class DangerousMogulsMod(JaxAtariInternalModPlugin):
    """
    Makes colliding with moguls cause the skier to fall.
    """
    constants_overrides = {
        "moguls_collidable": True,
    }

class JumpToBreakMod(JaxAtariInternalModPlugin):
    """
    Allows the skier to jump over moguls using the FIRE action.
    This mod specifically causes the skier to stop moving while jumping.
    """
    constants_overrides = {
        "jump_speed_multiplier": 0.0,
    }

class SpeedBurstMod(JaxAtariInternalModPlugin):
    """
    Allows the skier to accelerate beyond the default maximum speed using the DOWN action.
    """
    constants_overrides = {
        "down_max_speed": 1.8,
        "down_accel": 0.15,
    }

class HallOfFameMod(JaxAtariInternalModPlugin):
    """
    Places the gates dead center and creates a corridor of trees.
    """
    constants_overrides = {
        "hall_of_fame": True,
    }
