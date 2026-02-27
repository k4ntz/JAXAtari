import os
from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.gopher.gopher_mod_plugins import (
    RewardShapingMod,
    GreedyGopherMod,
    MirrorButtonMod,
    RewardInversionMod
)

class GopherEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for Gopher.
    """

    REGISTRY = {
        "reward_shaping": RewardShapingMod,
        "greedy_gopher": GreedyGopherMod,
        "mirror_button": MirrorButtonMod,
        "reward_inversion": RewardInversionMod,
    }

    # Points to where gopher sprites would be if you had modded ones
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
