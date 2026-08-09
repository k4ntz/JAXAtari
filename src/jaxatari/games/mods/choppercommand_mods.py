from jaxatari.games.mods.choppercommand.choppercommand_mod_plugins import AlwaysCenteredMod, MoreTrucksMod
from jaxatari.modification import JaxAtariModController


class ChopperCommandEnvMod(JaxAtariModController):
    """
        Game-specific Mod Controller for ChopperCommand.
        It simply inherits all logic from JaxAtariModController and defines the CHOPPERCOMMAND_MOD_REGISTRY.
    """

    REGISTRY = {
        "always_centered": AlwaysCenteredMod,
        "more_trucks": MoreTrucksMod,
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