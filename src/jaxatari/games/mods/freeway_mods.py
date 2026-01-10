from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.freeway_mod_plugins import StopAllCarsMod, StaticCarsMod, SlowCarsMod, BlackCarsMod, CenterCarsOnResetMod, InvertSpeed, HallOfFameMod

class FreewayEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for Freeway.
    It simply inherits all logic from JaxAtariModController and defines the FREEWAY_MOD_REGISTRY.
    """

    REGISTRY = {
        "stop_all_cars": StopAllCarsMod,
        "static_cars": StaticCarsMod,
        "slow_cars": SlowCarsMod,
        "invert_speed": InvertSpeed,
        "black_cars": BlackCarsMod,
        "center_cars_on_reset": CenterCarsOnResetMod,
        "hall_of_fame": ["_hall_of_fame_start", "static_cars"],
        "_hall_of_fame_start": HallOfFameMod,
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
