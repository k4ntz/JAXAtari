from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.pitfall.pitfall_mod_plugins import (
    GodModeMod,
    StationaryScorpionMod,
    FastPlayerMod,
)

# --- The Registry ---
PITFALL_MOD_REGISTRY = {
    "god_mode": GodModeMod,
    "stationary_scorpion": StationaryScorpionMod,
    "fast_player": FastPlayerMod,
}


class PitfallEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for Pitfall.
    It inherits all logic from JaxAtariModController and defines
    the REGISTRY.
    """

    REGISTRY = PITFALL_MOD_REGISTRY

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
