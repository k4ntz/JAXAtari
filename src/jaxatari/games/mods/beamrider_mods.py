from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.beamrider.beamrider_mod_plugins import (
    DontKillMod,
    DoubleEnemySpeedMod,
    FogOfWarMod,
    HardcoreMod,
    MothershipLaserMod,
    SameEnemiesMod,
    TeleportUFOsMod,
    ThreeLanesMod,
    ToasterMod,
)


class BeamRiderEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for Beamrider.
    """

    REGISTRY = {
        "dont_kill": DontKillMod,
        "double_enemy_speed": DoubleEnemySpeedMod,
        "fog_of_war": FogOfWarMod,
        "hardcore": HardcoreMod,
        "mothership_laser": MothershipLaserMod,
        "same_enemies": SameEnemiesMod,
        "teleport_ufos": TeleportUFOsMod,
        "three_lanes": ThreeLanesMod,
        "toaster": ToasterMod,
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
