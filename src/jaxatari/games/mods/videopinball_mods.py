import os
from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.videopinball.videopinball_mod_plugins import NeverActivateTiltMode, NoScoringCooldown, LowTiltEffect, ConstantBallDynamics

class VideoPinballEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for Video Pinball.
    It simply inherits all logic from JaxAtariModController and defines the REGISTRY.
    """

    REGISTRY = {
        "never_activate_tilt_mode": NeverActivateTiltMode,
        "no_scoring_cooldown": NoScoringCooldown,
        "low_tilt_effect": LowTiltEffect,
        "constant_ball_dynamics": ConstantBallDynamics,
    }

    _mod_sprite_dir = os.path.join(os.path.dirname(__file__), "videopinball", "sprites")

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
