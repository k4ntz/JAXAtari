from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.venture.venture_mod_plugins import (
    FastWinkyMod,
    SlowMonstersMod,
    WealthyVentureMod,
    PatientChaserMod,
    FastArrowsMod,
    LongRangeArrowsMod,
    GodModeMod,
)


class VentureEnvMod(JaxAtariModController):
    """Game-specific Mod Controller for Venture."""

    REGISTRY = {
        "fast_winky": FastWinkyMod,
        "slow_monsters": SlowMonstersMod,
        "wealthy_venture": WealthyVentureMod,
        "patient_chaser": PatientChaserMod,
        "fast_arrows": FastArrowsMod,
        "long_range_arrows": LongRangeArrowsMod,
        "god_mode": GodModeMod,
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
