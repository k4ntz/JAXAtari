from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.pong_mod_plugins import LazyEnemyMod, RandomEnemyMod, AlwaysZeroScoreMod

class PongEnvMod(JaxAtariModController):    
    """
    Game-specific Mod Controller for Pong.
    It simply inherits all logic from JaxAtariModController and defines the PONG_MOD_REGISTRY.
    """

    REGISTRY = {
        "lazy_enemy": LazyEnemyMod,
        "random_enemy": RandomEnemyMod,
        "zero_score": AlwaysZeroScoreMod,
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
            registry=self.REGISTRY  # for pong this is the only specific part, but other games might need to do execute some other logic in the constructor.
        )
