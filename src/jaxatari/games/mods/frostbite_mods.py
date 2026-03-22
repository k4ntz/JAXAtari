import os
from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.frostbite.frostbite_mod_plugins import (
    NoEnemiesMod
)

# --- The Registry ---
FROSTBITE_MOD_REGISTRY = {
    "no_enemies": NoEnemiesMod,
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