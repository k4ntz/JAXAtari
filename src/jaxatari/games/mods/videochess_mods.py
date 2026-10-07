from jaxatari.modification import (
    JaxAtariModController,
    JaxAtariInternalModPlugin,
    JaxAtariPostStepModPlugin,
)
from jaxatari.games.mods.videochess_mod_plugins import (
    RandomBotBlackMod,
    GreedyBotBlackMod,
    MinimaxBotBlackMod,
    PlayBothSidesMod,
    InstantMovementMod,
    LegalMovesDisplayMod,
    PawnsOnlyMod,
    QueensOnlyMod,
    RooksOnlyMod,
    KnightsOnlyMod,
    BishopsOnlyMod,
    CheckmateTestMod,
)


class VideochessEnvMod(JaxAtariModController):
    """Game-specific Mod Controller for VideoChess."""

    REGISTRY = {
        # Opponent / control
        "play_both_sides": PlayBothSidesMod,
        "random_bot_black": RandomBotBlackMod,
        "greedy_bot_black": GreedyBotBlackMod,
        "minimax_bot_black": MinimaxBotBlackMod,  # default behavior; explicit re-enable
        # Input / display
        "instant_movement": InstantMovementMod,
        "legal_moves_display": LegalMovesDisplayMod,
        # Board setups
        "pawns_only": PawnsOnlyMod,
        "queens_only": QueensOnlyMod,
        "rooks_only": RooksOnlyMod,
        "knights_only": KnightsOnlyMod,
        "bishops_only": BishopsOnlyMod,
        "checkmate_test": CheckmateTestMod,
    }

    def __init__(self, env, mods_config=None, allow_conflicts: bool = False):
        super().__init__(
            env=env,
            mods_config=mods_config or [],
            allow_conflicts=allow_conflicts,
            registry=self.REGISTRY,
        )
