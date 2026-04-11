import os
from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.mspacman.mspacman_mod_plugins import (
    CagedGhostsMod,
    ConstantFruitsMod,
    SetMaze1Mod,
    SetMaze2Mod,
    SetMaze3Mod,
    SetMaze4Mod,
    Only1GhostMod,
    Only2GhostMod,
    Only3GhostMod,
    RandomGhostNavigationMod
)

class MsPacmanEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for MsPacman.
    """
    REGISTRY = {
        "caged_ghosts": CagedGhostsMod,
        "constant_fruits": ConstantFruitsMod,
        "set_maze_1": SetMaze1Mod,
        "set_maze_2": SetMaze2Mod,
        "set_maze_3": SetMaze3Mod,
        "set_maze_4": SetMaze4Mod,
        "only_1_ghost": Only1GhostMod,
        "only_2_ghost": Only2GhostMod,
        "only_3_ghost": Only3GhostMod,
        "random_ghost_navigation": RandomGhostNavigationMod,
    }

    _mod_sprite_dir = os.path.join(os.path.dirname(__file__), "mspacman", "sprites")

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
