from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.upndown.upndown_mod_plugins import (
    AllowJumpBackwardsMod,
    RemoveStepRoadsMod,
    HigherPlayerSpeedMod,
    MoreCollectiblesMod,
    MinCarSpawnGapMod,
    SingleLaneCarSpawnMod,
    ProgressiveCarSpawnRateMod,
    TimeDecayCollectibleValueMod,
)


UPNDOWN_MOD_REGISTRY = {
    "allow_jump_backwards": AllowJumpBackwardsMod,
    "remove_step_roads": RemoveStepRoadsMod,
    "higher_player_speed": HigherPlayerSpeedMod,
    "spawn_more_collectibles": MoreCollectiblesMod,
    "minimum_car_spawn_gap": MinCarSpawnGapMod,
    "single_lane_car_spawn": SingleLaneCarSpawnMod,
    "progressive_car_spawn_rate": ProgressiveCarSpawnRateMod,
    "collectible_value_time_decay": TimeDecayCollectibleValueMod,
}


class UpNDownEnvMod(JaxAtariModController):
    REGISTRY = UPNDOWN_MOD_REGISTRY

    def __init__(
        self,
        env,
        mods_config: list = [],
        allow_conflicts: bool = False,
    ):
        super().__init__(
            env=env,
            mods_config=mods_config,
            allow_conflicts=allow_conflicts,
            registry=self.REGISTRY,
        )
