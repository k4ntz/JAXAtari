from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.seaquest_mod_plugins import DisableEnemiesMod

class SeaquestEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for Seaquest.
    It simply inherits all logic from JaxAtariModController and defines the SEAQUEST_MOD_REGISTRY.
    """

    REGISTRY = {
        "disable_enemies": DisableEnemiesMod,
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
