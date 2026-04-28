from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.qbert.qbert_mod_plugins import (
    NoRedBallsMod,
    NoPurpleBallMod,
    NoCoilyMod,
    NoGreenBallMod,
    NoSamMod,
    NoEnemiesMod
)

class QbertEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for Qbert.
    """
    REGISTRY = {
        "no_red_balls": NoRedBallsMod,
        "no_purple_ball": NoPurpleBallMod,
        "no_coily": NoCoilyMod,
        "no_green_ball": NoGreenBallMod,
        "no_sam": NoSamMod,
        "no_enemies": NoEnemiesMod,
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
