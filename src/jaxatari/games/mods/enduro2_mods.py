from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.enduro2.enduro2_mod_plugins import SpeedAndXPosHudMod, StartInCurveMod

class Enduro2EnvMod(JaxAtariModController):    
    """
    Game-specific Mod Controller for Enduro2.
    """
    REGISTRY = {
        "hud": SpeedAndXPosHudMod,
        "start_in_curve": StartInCurveMod,
    }

    def __init__(self,
                 env,
                 mods_config: list = [],
                 allow_conflicts: bool = False
                 ):

        super().__init__(
            env=env,
            mods_config=mods_config,
            allow_conflicts=allow_conflicts,
            registry=self.REGISTRY
        )
