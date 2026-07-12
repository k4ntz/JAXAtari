import os
from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.boxing2.boxing2_mod_plugins import (
    CenterEnemyMod, AlwaysPunchEnemyMod, DifficultyEasyMod, DifficultyMediumMod, DifficultyHardMod, DifficultyImpossibleMod,
    PeacefulEnemyMod, ShowCollisionZoneMod
)

class Boxing2EnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for Boxing2.
    """

    REGISTRY = {
        "center_enemy": CenterEnemyMod,
        "always_punch": AlwaysPunchEnemyMod,
        "easy": DifficultyEasyMod,
        "medium": DifficultyMediumMod,
        "normal": DifficultyMediumMod,
        "hard": DifficultyHardMod,
        "impossible": DifficultyImpossibleMod,
        "difficulty_easy": DifficultyEasyMod,
        "difficulty_medium": DifficultyMediumMod,
        "difficulty_normal": DifficultyMediumMod,
        "difficulty_hard": DifficultyHardMod,
        "difficulty_impossible": DifficultyImpossibleMod,
        "peaceful_enemy": PeacefulEnemyMod,
        "show_collision_zone": ShowCollisionZoneMod,
    }

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
