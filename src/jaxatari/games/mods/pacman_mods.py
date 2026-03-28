from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.pacman_mod_plugins import (
    FasterPacmanMod, SlowerGhostsMod, NoFrightMod,
    HalfDotsMod, RandomStartMod, LimitedVisionMod, CoopMultiplayerMod,
    MultiMazeCampaignMod,
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
        "multi_maze_campaign": MultiMazeCampaignMod,
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
        # Pac-Man-only: plugins may define attach_to_env(base_env) after internal patches (e.g. multi-maze preload).
        for mod_key in mods_config:
            plugin_class = self.REGISTRY.get(mod_key)
            if plugin_class is None or not isinstance(plugin_class, type):
                continue
            attacher = getattr(plugin_class, "attach_to_env", None)
            if callable(attacher):
                attacher(self._env)
