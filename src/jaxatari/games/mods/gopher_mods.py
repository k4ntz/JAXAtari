from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.gopher_mod_plugins import (
    LazyGopherMod, 
    FastFarmerMod, 
    GreedyGopherMod, 
    PinkGopherMod, 
    GenerousBirdMod
)

class GopherEnvMod(JaxAtariModController):    
    """
    Game-specific Mod Controller for Gopher.
    """
    REGISTRY = {
        "lazy_gopher": LazyGopherMod,
        "fast_farmer": FastFarmerMod,
        "greedy_gopher": GreedyGopherMod,
        "pink_gopher": PinkGopherMod,
        "double_reward": GenerousBirdMod,
    }

    def __init__(self, env, mods_config: list = [], allow_conflicts: bool = False):
        super().__init__(
            env=env,
            mods_config=mods_config,
            allow_conflicts=allow_conflicts,
            registry=self.REGISTRY
        )