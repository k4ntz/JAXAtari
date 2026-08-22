import os
from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.othello.othello_mod_plugins import InstantMovementMod


class OthelloEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for Othello.
    """

    REGISTRY = {
        "instant_movement": InstantMovementMod,
    }

    _mod_sprite_dir = os.path.join(os.path.dirname(__file__), "othello", "sprites")

    def __init__(
        self,
        env,
        mods_config: list = [],
        allow_conflicts: bool = False,
    ):
        super().__init__(
            env=env,
            mods_config=mods_config,
            allow_conflicts=allow_conflicts,
            registry=self.REGISTRY,
        )
