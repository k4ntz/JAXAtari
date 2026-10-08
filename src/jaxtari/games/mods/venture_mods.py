from jaxtari.modification import JaxtariModController, JaxtariModWrapper
from jaxtari.games.jax_venture import JaxVenture, VentureConstants
from jaxtari.games.mods.venture.venture_mod_plugins import (
    NightMod,
    GrayscaleMod,
    InvertedColorsMod,
    MatrixMod,
    BloodMoonMod,
    SlowEnemiesMod,
    FastEnemiesMod,
    RemoveOverworldMobsMod,
    RewardForKillMod,
)

class VentureEnvMod(JaxtariModController):
    """
    Controller for Venture mods.
    """
    
    REGISTRY = {
        'night_mode': NightMod,
        'grayscale': GrayscaleMod,
        'inverted_colors': InvertedColorsMod,
        'matrix_theme': MatrixMod,
        'blood_moon': BloodMoonMod,
        'slow_enemies': SlowEnemiesMod,
        'fast_enemies': FastEnemiesMod,
        'remove_overworld_mobs': RemoveOverworldMobsMod,
        'reward_for_kill': RewardForKillMod,
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
