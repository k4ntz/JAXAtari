import os
from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.skiing.skiing_mod_plugins import (
    MoreTreesMod, MoreMogulsMod, DangerousMogulsMod, JumpToBreakMod, 
    SpeedBurstMod, TreesEverywhereMod, HallOfFameMod,
    InvertFlagsMod, InvertFlagColorsMod, MovingFlagsMod, RandomFlagsMod, FlagFlurryMod, MogulsToTreesMod,
    ClassicTreesMod, ThinMogulsMod, BlueSkiierMod, GreenFlagsMod
)

class SkiingEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for Skiing.
    It inherits all logic from JaxAtariModController and defines the REGISTRY.
    """

    # Define the path relative to this file (mod sprites fallback)
    _mod_sprite_dir = os.path.join(os.path.dirname(__file__), "skiing", "sprites")

    REGISTRY = {
        "_more_trees": MoreTreesMod,
        "_trees_everywhere": TreesEverywhereMod,
        "_more_moguls": MoreMogulsMod,
        "_dangerous_moguls": DangerousMogulsMod,
        "jump_to_break": JumpToBreakMod,
        "speed_burst": SpeedBurstMod,
        "hall_of_fame": HallOfFameMod,
        "invert_flags": InvertFlagsMod,
        "invert_flag_colors": InvertFlagColorsMod,
        "moving_flags": MovingFlagsMod,
        "random_flags": RandomFlagsMod,
        "flag_flurry": FlagFlurryMod,
        "moguls_to_trees": MogulsToTreesMod,
        "classic_trees": ClassicTreesMod,
        "thin_moguls": ThinMogulsMod,
        "blue_skiier": BlueSkiierMod,
        "green_flags": GreenFlagsMod,
        "off_piste": ["_more_trees", "_trees_everywhere", "_more_moguls", "_dangerous_moguls"],
        "change_sprites": ["classic_trees", "thin_moguls", "blue_skiier", "green_flags"],
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
