import os
from jaxatari.modification import JaxAtariModController


class RoadRunnerEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for RoadRunner.
    Inherits all logic from JaxAtariModController and defines the mod registry.
    """

    REGISTRY = {
        # Add mods here, e.g.:
        # "slow_player": SlowPlayerMod,
        # "no_enemies": NoEnemiesMod,
        # Modpacks (bundled mods) use lists:
        # "easy_mode": ["slow_enemies", "extra_lives"],
    }

    _mod_sprite_dir = os.path.join(os.path.dirname(__file__), "roadrunner", "sprites")

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
