import os
from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.fishingderby.fishingderby_mod_plugins import (
    SharkNoMovementEasyMod,
    SharkNoMovementMiddleMod,
    SharkTeleportMod,
    FishOnPlayerSideMod,
    FishOnDifferentSidesMod,
    FishInMiddleMod,
)

class FishingDerbyEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for FishingDerby.
    """

    REGISTRY = {
        "shark_no_movement_easy": SharkNoMovementEasyMod,
        "shark_no_movement_middle": SharkNoMovementMiddleMod,
        "shark_teleport": SharkTeleportMod,
        "fish_on_player_side": FishOnPlayerSideMod,
        "fish_on_different_sides": FishOnDifferentSidesMod,
        "fish_in_middle": FishInMiddleMod,
    }

    _mod_sprite_dir = os.path.join(os.path.dirname(__file__), "fishingderby", "sprites")

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
