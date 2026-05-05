from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.gravitar.gravitar_mod_plugins import (
    RapidFireMod,
    ZeroGravityMod,
    HyperGravityMod,
    FuelCrisisMod,
    HarmlessEnemiesMod,
    ValuableReactorMod,
    AntiGravityMod,
    HighSpeedMod,
    InfiniteFuelMod,
    SlowEnemiesMod,
    LongRangeTractorMod,
    NeonMod,
    RedAlertMod,
    GrayscaleMod,
    InvertedColorsMod,
    SharpshooterMod,
)


class GravitarEnvMod(JaxAtariModController):
    """Game-specific Mod Controller for Gravitar."""

    REGISTRY = {
        "rapid_fire": RapidFireMod,
        "zero_gravity": ZeroGravityMod,
        "hyper_gravity": HyperGravityMod,
        "fuel_crisis": FuelCrisisMod,
        "harmless_enemies": HarmlessEnemiesMod,
        "valuable_reactor": ValuableReactorMod,
        "anti_gravity": AntiGravityMod,
        "high_speed": HighSpeedMod,
        "infinite_fuel": InfiniteFuelMod,
        "slow_enemies": SlowEnemiesMod,
        "long_range_tractor": LongRangeTractorMod,
        "neon_mode": NeonMod,
        "red_alert": RedAlertMod,
        "grayscale": GrayscaleMod,
        "inverted_colors": InvertedColorsMod,
        "sharpshooter": SharpshooterMod,
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
