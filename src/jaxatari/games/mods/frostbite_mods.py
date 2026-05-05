import os
from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.frostbite.frostbite_mod_plugins import (
    NoEnemiesMod, LightBlueIceMod, _StaticIceMod, _MisalignedIceMod, _AlignedIceMod, RecoloredObstaclesMod, TigerMod,
    WhiteIglooMod, LeftIglooMod, EarlyBearMod, DarkNightMod, BadFishesRewardMod
)

# --- The Registry ---
FROSTBITE_MOD_REGISTRY = {
    "no_enemies": NoEnemiesMod,
    "lightblue_ice": LightBlueIceMod,
    "recolored_obstacles": RecoloredObstaclesMod,
    "tiger": TigerMod,
    "white_igloo": WhiteIglooMod,
    "left_igloo": LeftIglooMod,
    "early_bear": EarlyBearMod,
    "dark_night": DarkNightMod,
    "bad_fishes_reward": BadFishesRewardMod,
    "_static_ice": _StaticIceMod,
    "_misaligned_ice": _MisalignedIceMod,
    "_aligned_ice": _AlignedIceMod,
    "static_aligned_ice": ["_static_ice", "_aligned_ice"],
    "static_misaligned_ice": ["_static_ice", "_misaligned_ice"],
    "change_sprites": ["tiger", "white_igloo", "recolored_obstacles", "lightblue_ice"]
}

class FrostbiteEnvMod(JaxAtariModController):
    """
    Game-specific (Group 1) Mod Controller for Frostbite.
    It inherits all logic from JaxAtariModController and defines
    the REGISTRY.
    """

    REGISTRY = FROSTBITE_MOD_REGISTRY

    # Define the path relative to this file (mod sprites fallback)
    _mod_sprite_dir = os.path.join(os.path.dirname(__file__), "frostbite", "sprites")

    def __init__(self,
                 env,
                 mods_config: list = [],
                 allow_conflicts: bool = True
                 ):
        super().__init__(
            env=env,
            mods_config=mods_config,
            allow_conflicts=allow_conflicts,
            registry=self.REGISTRY
        )
