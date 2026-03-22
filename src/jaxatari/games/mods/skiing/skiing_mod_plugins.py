from jaxatari.modification import JaxAtariInternalModPlugin

class MoreTreesMod(JaxAtariInternalModPlugin):
    """
    Spawns more trees during the race.
    """
    constants_overrides = {
        "max_num_trees": 12,
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
        "allow_jump": True,
        "jump_stops_skier": True,
    }

class SpeedBurstMod(JaxAtariInternalModPlugin):
    """
    Allows the skier to accelerate beyond the default maximum speed using the DOWN action.
    """
    constants_overrides = {
        "allow_down_acceleration": True,
    }
