from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.videochess_mod_plugins import (
    RandomBotBlackMod,
    GreedyBotBlackMod,
    MinimaxBotBlackMod,
    PawnsOnlyMod,
    QueensOnlyMod,
    RooksOnlyMod,
    KnightsOnlyMod,
    BishopsOnlyMod,
    LegalMovesDisplayMod,
    CheckmateTestMod,
)


class VideochessEnvMod(JaxAtariModController):
    """Game-specific Mod Controller for VideoChess."""

    REGISTRY = {
        "random_bot_black": RandomBotBlackMod,
        "greedy_bot_black": GreedyBotBlackMod,
        "minimax_bot_black": MinimaxBotBlackMod,
        "pawns_only": PawnsOnlyMod,
        "queens_only": QueensOnlyMod,
        "rooks_only": RooksOnlyMod,
        "knights_only": KnightsOnlyMod,
        "bishops_only": BishopsOnlyMod,
        "legal_moves_display": LegalMovesDisplayMod,
        "checkmate_test": CheckmateTestMod,
    }

    def __init__(self, env, mods_config=None, allow_conflicts: bool = False):
        super().__init__(
            env=env,
            mods_config=mods_config or [],
            allow_conflicts=allow_conflicts,
            registry=self.REGISTRY,
        )