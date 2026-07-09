from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.demonattack.demonattack_mod_plugins import (
    FastPlayerMod,
    FastLaserMod,
    SlowEnemyShotsMod,
    ShortWavesMod,
    NoEnemyShotsMod,
    InfiniteLivesMod,
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
