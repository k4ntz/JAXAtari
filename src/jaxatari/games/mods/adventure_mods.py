import os
from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.adventure.adventure_mod_plugins import FasterDragonsMod, FasterBiteMod, FleaingDragonMod, DragonReviveMod, RandomPlayerSpawnMod, LevelTwoMod, LevelThreeMod, EasterEggMod
class AdventureEnvMod(JaxAtariModController):    
    """
    Game-specific Mod Controller for Adventure.
    It simply inherits all logic from JaxAtariModController and defines the ADVENTURE_MOD_REGISTRY.
    """

    REGISTRY = {
        "faster_dragon": FasterDragonsMod,
        "faster_bite": FasterBiteMod,
        "fleaing_dragon": FleaingDragonMod,
        "dragon_revive": DragonReviveMod,
        "random_player_spawn": RandomPlayerSpawnMod,
        "level_two": LevelTwoMod,
        "level_three": LevelThreeMod,
        "easter_egg": EasterEggMod
    }

    _mod_sprite_dir = os.path.join(os.path.dirname(__file__), "adventure", "sprites")

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
