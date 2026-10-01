import os
from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.icehockey.icehockey_mod_plugins import (
    RoundedCornersMod,
    EnemySpeedUpMod,
    MovingGoalsMod,
    PlayerSlidingMod,
    NoAttackingZonesMod,
    DecreasedGoalSizeMod,
    IncreasedGoalSizeMod,
    DisableTacklingMod,
    TackleSlowdownMod,
)


class IceHockeyEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for IceHockey.
    It simply inherits all logic from JaxAtariModController and defines the ICEHOCKEY_MOD_REGISTRY.
    """

    REGISTRY = {
        "rounded_corners": RoundedCornersMod,
        "moving_goals": MovingGoalsMod,
        "player_sliding": PlayerSlidingMod,
        "enemy_speedup_on_goal": EnemySpeedUpMod,
        "no_attacking_zones": NoAttackingZonesMod,
        "decreased_goal_size": DecreasedGoalSizeMod,
        "increased_goal_size": IncreasedGoalSizeMod,
        "disable_tackling": DisableTacklingMod,
        "tackle_slowdown": TackleSlowdownMod,
    }

    _mod_sprite_dir = os.path.join(os.path.dirname(__file__), "icehockey", "sprites")

    def __init__(self, env, mods_config: list = [], allow_conflicts: bool = False):

        super().__init__(
            env=env,
            mods_config=mods_config,
            allow_conflicts=allow_conflicts,
            registry=self.REGISTRY,
        )
