from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.miniature_golf_mod_plugins import (LargeHoleMod, MovingHoleMod, PermeableObstacleMod,
                                                            PermeableWallMod, SecondHoleMod, SoftShotRequiredMod,
                                                            StationaryObstacleMod, AlwaysZeroShotsMod)

class MiniatureGolfEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for Miniature Golf.
    It simply inherits all logic from JaxAtariModController and defines the MINIATURE_GOLF_MOD_REGISTRY.
    """

    REGISTRY = {
        "large_hole": LargeHoleMod,
        "moving_hole": MovingHoleMod,
        "permeable_obstacle": PermeableObstacleMod,
        "permeable_wall": PermeableWallMod,
        "second_hole": SecondHoleMod,
        "soft_shot_required": SoftShotRequiredMod,
        "stationary_obstacle": StationaryObstacleMod,
        "zero_shots": AlwaysZeroShotsMod,
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
