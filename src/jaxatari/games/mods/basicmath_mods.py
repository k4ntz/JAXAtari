from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.basicmath_mod_plugins import (
    BackgroundBlackColorMod,
    BackgroundRandomColorMod,
    NumberRandomColorMod,
    BiggerNumbersMod
)

class BasicMathEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for Breakout.
    It simply inherits all logic from JaxAtariModController and defines the BREAKOUT_MOD_REGISTRY.
    """

    REGISTRY = {
        "background_black": BackgroundBlackColorMod,
        "background_random": BackgroundRandomColorMod,
        "number_random": NumberRandomColorMod,
        "bigger_numbers": BiggerNumbersMod
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