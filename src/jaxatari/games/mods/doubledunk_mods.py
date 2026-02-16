import os
from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.doubledunk.doubledunk_mod_plugins import TimerMod, SuperDunkMod

class DoubleDunkEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for Double Dunk.
    """

    REGISTRY = {
        "timer_mod": TimerMod,
        "super_dunk": SuperDunkMod,
    }

    _mod_sprite_dir = os.path.join(os.path.dirname(__file__), "doubledunk", "sprites")

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