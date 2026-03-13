from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.videochess_mod_plugins import (
    RandomBotBlackMod,
    GreedyBotBlackMod,
    MinimaxBotBlackMod,
    PawnsOnlyMod,
    QueensOnlyMod,
    KingsOnlyMod,
)


class VideochessEnvMod(JaxAtariModController):
    """Game-specific Mod Controller for VideoChess."""

    REGISTRY = {
        "random_bot_black": RandomBotBlackMod,
        "greedy_bot_black": GreedyBotBlackMod,
        "minimax_bot_black": MinimaxBotBlackMod,
        "pawns_only": PawnsOnlyMod,
        "queens_only": QueensOnlyMod,
        "kings_only": KingsOnlyMod,
    }

    def __init__(self, env, mods_config=None, allow_conflicts: bool = False):
        super().__init__(
            env=env,
            mods_config=mods_config or [],
            allow_conflicts=allow_conflicts,
            registry=self.REGISTRY,
        )