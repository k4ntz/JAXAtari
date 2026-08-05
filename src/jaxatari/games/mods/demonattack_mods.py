from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.demonattack.demonattack_mod_plugins import (
    FastPlayerMod,
    FastLaserMod,
    SlowEnemyShotsMod,
    ShortWavesMod,
    NoEnemyShotsMod,
    InfiniteLivesMod,
    RelentlessWavesMod,
    LateWaveStartMod,
    PlayerGuidedLaserMod,
    HomingLaserMod,
    TeleportingDemonsMod,
    SideStepLowestDemonsMod,
    ZigZagMovementDemonsMod,
)


class DemonAttackEnvMod(JaxAtariModController):
    """
    Game-specific mod controller for Demon Attack.
    """

    REGISTRY = {
        "fast_player": FastPlayerMod,
        "fast_laser": FastLaserMod,
        "slow_enemy_shots": SlowEnemyShotsMod,
        "short_waves": ShortWavesMod,
        "no_enemy_shots": NoEnemyShotsMod,
        "infinite_lives": InfiniteLivesMod,
        "relentless_waves": RelentlessWavesMod,
        "late_wave_start": LateWaveStartMod,
        "player_guided_laser": PlayerGuidedLaserMod,
        "homing_laser": HomingLaserMod,
        "teleporting_demons": TeleportingDemonsMod,
        "sidestep_lowest_demon": SideStepLowestDemonsMod,
        "zigzag_movement_demons": ZigZagMovementDemonsMod,
        "advanced_survival": [
            "late_wave_start",
            "relentless_waves",
            "player_guided_laser",
            "teleporting_demons",
            "sidestep_lowest_demon",
        ],
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
