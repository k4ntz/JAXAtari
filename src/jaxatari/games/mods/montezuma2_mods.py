import os
from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.montezuma2.montezuma2_mod_plugins import (
    InfiniteAmuletMod, SuperJumpMod, NoFallDamageMod,
    RevealMapMod, DebugHudMod, NoEnemiesMod, CenterBouncingSkullMod
)

# --- The Registry ---
MONTEZUMA2_MOD_REGISTRY = {
    "infinite_amulet": InfiniteAmuletMod,
    "super_jump": SuperJumpMod,
    "no_fall_damage": NoFallDamageMod,
    "reveal_map": RevealMapMod,
    "debug_hud": DebugHudMod,
    "no_enemies": NoEnemiesMod,
    "center_bouncing_skull": CenterBouncingSkullMod,
    "god_mode": ["infinite_amulet", "no_fall_damage", "no_enemies", "super_jump"] # bundle into a modpack
}

class Montezuma2EnvMod(JaxAtariModController):
    """
    Game-specific (Group 1) Mod Controller for Montezuma2.
    It inherits all logic from JaxAtariModController and defines
    the REGISTRY.
    """

    REGISTRY = MONTEZUMA2_MOD_REGISTRY

    # Define the path relative to this file (mod sprites fallback)
    _mod_sprite_dir = os.path.join(os.path.dirname(__file__), "montezuma2", "sprites")

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
