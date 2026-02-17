from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.defender_mod_plugins import (
    SmartBombsUnlimitedMod,
    BulletTimewarpMod,
    ParachutesEquippedMod,
    EnemyEmpMod,
    NoBackupMod,
    MissingFundingMod,
    EnemiesOnSpeedMod,
    NoBreaksInSpaceMod,
)


class DefenderEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for Defender.
    """

    REGISTRY = {
        "smart_bombs_unlimited": SmartBombsUnlimitedMod,
        "bullet_timewarp": BulletTimewarpMod,
        "parachutes_equipped": ParachutesEquippedMod,
        "enemy_emp": EnemyEmpMod,
        "no_backup": NoBackupMod,
        "missing_funding": MissingFundingMod,
        "enemies_on_speed": EnemiesOnSpeedMod,
        "no_brakes_in_space": NoBreaksInSpaceMod,
    }

    def __init__(self, env, mods_config: list = [], allow_conflicts: bool = False):
        super().__init__(
            env=env,
            mods_config=mods_config,
            allow_conflicts=allow_conflicts,
            registry=self.REGISTRY,
        )
