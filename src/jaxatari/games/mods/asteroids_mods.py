import os
from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.asteroids.asteroids_mod_plugins import DontShootMod, MatrixMod, SlowAsteroidsMod, InstantTurnMod

class AsteroidsEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for Asteroids.
    """

    REGISTRY = {
        "dont_shoot": DontShootMod,
        "matrix_theme": MatrixMod,
        "slow_asteroids": SlowAsteroidsMod,
        "instant_turn": InstantTurnMod,
    }

    _mod_sprite_dir = os.path.join(os.path.dirname(__file__), "asteroids", "sprites")

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
