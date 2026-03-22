from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.pacman_mod_plugins import (
    FasterPacmanMod, SlowerGhostsMod, NoFrightMod,
    HalfDotsMod, RandomStartMod, LimitedVisionMod, CoopMultiplayerMod
)

class PacmanEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for Pacman.
    It inherits all logic from JaxAtariModController and defines the PACMAN_MOD_REGISTRY.
    """

    REGISTRY = {
        "faster_pacman": FasterPacmanMod,
        "slower_ghosts": SlowerGhostsMod,
        "no_fright": NoFrightMod,
        "half_dots": HalfDotsMod,
        "random_start": RandomStartMod,
        "limited_vision": LimitedVisionMod,
        "coop_multiplayer": CoopMultiplayerMod,
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
