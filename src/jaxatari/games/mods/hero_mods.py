from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.hero.hero_mod_plugins import (
    START_LEVEL_MODS,
    UnlimitedDynamiteMod,
    UnlimitedLivesMod,
)


class HeroEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for H.E.R.O.
    It inherits all logic from JaxAtariModController and defines the REGISTRY.

    The mods here are debugging aids (see hero/hero_mod_plugins.py):
    ``start_level_N`` (N = 1..20), ``unlimited_lives`` and ``unlimited_dynamite``.
    ``scripts/play.py -g hero -l N -lifes -granades`` maps onto them.
    """

    REGISTRY = {
        "unlimited_lives": UnlimitedLivesMod,
        "unlimited_dynamite": UnlimitedDynamiteMod,
        **START_LEVEL_MODS,
    }

    def __init__(self,
                 env,
                 mods_config: list = (),
                 allow_conflicts: bool = False
                 ):
        super().__init__(
            env=env,
            mods_config=mods_config,
            allow_conflicts=allow_conflicts,
            registry=self.REGISTRY,
        )
