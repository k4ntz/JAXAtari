from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.hero.hero_mod_plugins import (
    START_LEVEL_MODS,
    SlowHeroMod,
    UnlimitedDynamiteMod,
    UnlimitedLivesMod,
)


class HeroEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for H.E.R.O.
    It inherits all logic from JaxAtariModController and defines the REGISTRY.

    The mods here are debugging aids (see hero/hero_mod_plugins.py):
    ``start_level_N`` (N = 1..20), ``unlimited_lives`` and ``unlimited_dynamite``.
    Select them like any game's mods: ``scripts/play.py -g hero -m start_level_5
    unlimited_lives unlimited_dynamite``.
    ``slow`` swaps the ROM hero (the default) for the earlier one.
    """

    REGISTRY = {
        "unlimited_lives": UnlimitedLivesMod,
        "unlimited_dynamite": UnlimitedDynamiteMod,
        "slow": SlowHeroMod,
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
