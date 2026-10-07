import os

from jaxtari.games.mods.kaboom.kaboom_mod_plugins import BombsMoveHorizontally, BombsFallFaster, BombsRed, MadBomberGreen
from jaxtari.modification import JaxtariModController

KABOOM_MOD_REGISTRY = {
    "bombs_move_horizontally": BombsMoveHorizontally,
    "bombs_fall_randomlyfast": BombsFallFaster,
    "bombs_red": BombsRed,
    "madbomber_green": MadBomberGreen,
}

class KaboomEnvMod(JaxtariModController):
    """
    Game-specific Mod Controller for Kaboom.
    It inherits all logic from JaxtariModController and defines the REGISTRY.
    """

    REGISTRY = KABOOM_MOD_REGISTRY

    _mod_sprite_dir = os.path.join(os.path.dirname(__file__), "kaboom", "sprites")

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
