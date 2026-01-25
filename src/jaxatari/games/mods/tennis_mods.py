from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.tennis_mod_plugins import RandomWalkSpeedWrapper, RandomBallSpeedWrapper

class TennisEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for Seaquest.
    It simply inherits all logic from JaxAtariModController and defines the SEAQUEST_MOD_REGISTRY.
    """

    REGISTRY = {
        "random_ball_speed": RandomBallSpeedWrapper,
        "random_walk_speed": RandomWalkSpeedWrapper,
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
