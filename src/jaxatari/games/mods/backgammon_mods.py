import os

from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.backgammon_mod_plugins import (
    # Simple mods
    BrownThemeMod,
    BlueThemeMod,
    ClassicThemeMod, #Default theme without any mod applied
    SimplifyBackgammonMod,
    ShortGameMod,
    SetupModeMod,
    HighlightLegalMovesMod,
    # Complex mods
    NoHitsMod,
    RewardShapingMod,
    ALEControlsMod,
)


class BackgammonEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for Backgammon.
    Inherits all logic from JaxAtariModController and defines the mod registry.
    """

    REGISTRY = {
        # Simple mods
        "brown_theme": BrownThemeMod,
        "blue_theme": BlueThemeMod,
        "classic_theme": ClassicThemeMod, #Default theme without any mod applied
        "short_game": ShortGameMod,
        "simplify": SimplifyBackgammonMod,
        "highlight_legal_moves": HighlightLegalMovesMod,
        "setup_mode": SetupModeMod,
        # Complex mods
        "no_hits": NoHitsMod,
        "reward_shaping": RewardShapingMod,
        "ale_controls": ALEControlsMod,
    }

    _mod_sprite_dir = os.path.join(os.path.dirname(__file__), "backgammon", "sprites")

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


