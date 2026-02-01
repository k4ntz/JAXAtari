from jaxatari.games.mods.yarsrevenge_mod_plugins import MoreSwirlsMod, NoAnimationsMod, OneShieldShapeMod, SpeedUpMod, StaticEnergyShieldMod
from jaxatari.modification import JaxAtariModController

class YarsRevengeEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for Yar's Revenge.
    It simply inherits all logic from JaxAtariModController and defines the REGISTRY.
    """

    REGISTRY = {
        # Basic Mods
        "no_animations": NoAnimationsMod,
        "speed_up": SpeedUpMod,
        "more_swirls": MoreSwirlsMod,
        "static_energy_shield": StaticEnergyShieldMod,
        "one_shield_shape": OneShieldShapeMod
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
