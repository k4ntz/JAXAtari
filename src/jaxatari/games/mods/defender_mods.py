from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.defender_mod_plugins import (
    InfiniteSmartBombsMod,
    SlowerBulletsMod,
    NoLanderPickupMod,
    StaticInvadersMod,
    FasterLevelClearMod,
    HardcoreStartMod,
    FasterInvadersMod,
    NoBrakesMod,
)

class DefenderEnvMod(JaxAtariModController):    
    """
    Game-specific Mod Controller for Defender.
    """

    REGISTRY = {
        "infinite_smart_bombs": InfiniteSmartBombsMod,
        "slower_bullets": SlowerBulletsMod,
        "no_lander_pickup": NoLanderPickupMod,
        "static_invaders": StaticInvadersMod,
        "faster_level_clear": FasterLevelClearMod,
        "hardcore_start": HardcoreStartMod,
        "faster_invaders": FasterInvadersMod,
        "no_brakes": NoBrakesMod,
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
