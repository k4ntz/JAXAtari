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

class JumpMod(JaxAtariInternalModPlugin):
    """
    Allows the skier to jump over moguls using the FIRE action.
    """
    constants_overrides = {
        "allow_jump": True,
    }
