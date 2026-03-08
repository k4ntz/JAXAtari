import os
from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.tennis.tennis_mod_plugins import RandomWalkSpeedWrapper, RandomBallSpeedWrapper

class TennisEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for Tennis.
    It simply inherits all logic from JaxAtariModController and defines the REGISTRY.
    """

    REGISTRY = {
        "random_ball_speed": RandomBallSpeedWrapper,
        "random_walk_speed": RandomWalkSpeedWrapper,
    }

    _mod_sprite_dir = os.path.join(os.path.dirname(__file__), "tennis", "sprites")

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
