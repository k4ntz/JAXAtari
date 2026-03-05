import os
from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.gopher.gopher_mod_plugins import (
    GreedyGopherMod,
    MirrorButtonMod,
    PinkGopherMod,
    HeavyShovelMod,
    FastSeedMod

)

class GopherEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for Gopher.
    """

    REGISTRY = {
        
        "greedy_gopher": GreedyGopherMod,
        "mirror_button": MirrorButtonMod,
        "pink_gopher": PinkGopherMod,
        "heavy_shovel": HeavyShovelMod,
        "fast_seed": FastSeedMod,
    }

    
    _mod_sprite_dir = os.path.join(os.path.dirname(__file__), "gopher", "sprites")

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
