from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.tictactoe3d_mod_plugins import RandomStaticBlockersMod


class TicTacToe3DMod(JaxAtariModController):
    """
    Mod controller for TicTacToe3D.
    """

    REGISTRY = {
        "random_static_blockers": RandomStaticBlockersMod,
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