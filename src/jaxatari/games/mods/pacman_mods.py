import os
from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.pacman.pacman_mod_plugins import (
    CagedGhostsMod,
    OftenVitaminsMod,
    SetMaze2Mod,
    Only1GhostMod,
    Only2GhostMod,
    Only3GhostMod,
    RandomGhostNavigationMod
)

class PacmanEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for Pacman.
    """
    REGISTRY = {
        "caged_ghosts": CagedGhostsMod,
        "often_fruit": OftenVitaminsMod,
        "set_maze_2": SetMaze2Mod,
        "only_1_ghost": Only1GhostMod,
        "only_2_ghost": Only2GhostMod,
        "only_3_ghost": Only3GhostMod,
        "random_ghost_navigation": RandomGhostNavigationMod,
    }

    _mod_sprite_dir = os.path.join(os.path.dirname(__file__), "pacman", "sprites")

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
