import os
from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.miniature_golf.miniature_golf_mod_plugins import (
    LargeHoleMod, MovingHoleMod, PermeableObstacleMod, PermeableWallMod, SecondHoleMod,
    SoftShotRequiredMod, StationaryObstacleMod, AlwaysZeroShotsMod, ManhattanRewardMod,
    DiagonalMovementMod, StartLevel1Mod, StartLevel2Mod, StartLevel3Mod, StartLevel4Mod,
    StartLevel5Mod, StartLevel6Mod, StartLevel7Mod, StartLevel8Mod, StartLevel9Mod,
)

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
        "manhattan_reward": ManhattanRewardMod,
        "diagonal_movement": DiagonalMovementMod,
        "start_level_1": StartLevel1Mod,
        "start_level_2": StartLevel2Mod,
        "start_level_3": StartLevel3Mod,
        "start_level_4": StartLevel4Mod,
        "start_level_5": StartLevel5Mod,
        "start_level_6": StartLevel6Mod,
        "start_level_7": StartLevel7Mod,
        "start_level_8": StartLevel8Mod,
        "start_level_9": StartLevel9Mod,
    }

    _mod_sprite_dir = os.path.join(os.path.dirname(__file__), "miniature_golf", "sprites")

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
