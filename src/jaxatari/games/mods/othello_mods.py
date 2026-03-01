import jax
import jax.numpy as jnp
from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.othello_mod_plugins import (
    RandomAIMod,
    LargeBoardMod,
    SmallBoardMod,
    BombMod
)

class OthelloEnvMod(JaxAtariModController):    
    """
    Game-specific Mod Controller for Othello.
    Defines the OTHELLO_MOD_REGISTRY.
    """

    REGISTRY = {
        "random_ai": RandomAIMod,
        "large_board": LargeBoardMod,
        "small_board": SmallBoardMod,
        "bomb": BombMod,
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
