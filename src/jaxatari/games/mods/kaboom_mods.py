from jaxatari.games.mods.kaboom_mod_plugins import BombsMoveHorizontally, BombsFallFaster
from jaxatari.modification import JaxAtariModController

class KaboomEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for Kaboom.
    It inherits all logic from JaxAtariModController and defines the REGISTRY.
    """

    REGISTRY = {
        "bombs_move_horizontally": BombsMoveHorizontally,
        "bombs_fall_randomlyfast": BombsFallFaster,
        "bombs_red": None,
        "madbomber_green": None,
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
