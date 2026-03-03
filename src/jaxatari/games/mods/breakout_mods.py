import os
from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.breakout.breakout_mod_plugins import (
    SpeedModeMod,
    SmallPaddleMod,
    BigPaddleMod,
    BallDriftMod,
    BallGravityMod,
    BallColorMod,
    BlockColorMod,
    PlayerColorMod,
)

class BreakoutEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for Breakout.
    It simply inherits all logic from JaxAtariModController and defines the REGISTRY.
    """

    REGISTRY = {
        "speed_mode": SpeedModeMod,
        "small_paddle": SmallPaddleMod,
        "big_paddle": BigPaddleMod,
        "ball_drift": BallDriftMod,
        "ball_gravity": BallGravityMod,
        "ball_color": BallColorMod,
        "block_color": BlockColorMod,
        "player_color": PlayerColorMod,
    }

    _mod_sprite_dir = os.path.join(os.path.dirname(__file__), "breakout", "sprites")

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
