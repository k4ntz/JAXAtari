from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.tictactoe3d_mod_plugins import (RandomStaticBlockersMod, RandomTurnOrderMod, StrictIllegalMoveMod)


class Tictactoe3dEnvMod(JaxAtariModController):
    """
    Mod controller for TicTacToe3D.
    """

    REGISTRY = {
        "random_static_blockers": RandomStaticBlockersMod,
        "random_turn_order": RandomTurnOrderMod,
        "strict_illegal_move": StrictIllegalMoveMod, 
    }

    def __init__(
        self,
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