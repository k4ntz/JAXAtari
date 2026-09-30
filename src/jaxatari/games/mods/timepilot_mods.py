import os
from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.timepilot.timepilot_mod_plugins import (
    DontKillMod,
    MatrixMod,
    ExtraLivesMod,
    InstantTurnMod,
    ReverseChronologyMod
)

class TimePilotEnvMod(JaxAtariModController):    
    """
    Game-specific Mod Controller for TimePilot.
    It simply inherits all logic from JaxAtariModController and defines the REGISTRY.
    """

    REGISTRY = {
        "dont_kill": DontKillMod,
        "matrix_theme": MatrixMod,
        "extra_lives": ExtraLivesMod,
        "instant_turn": InstantTurnMod,
        "reverse_chronology": ReverseChronologyMod,
    }

    _mod_sprite_dir = os.path.join(os.path.dirname(__file__), "timepilot", "sprites")

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
