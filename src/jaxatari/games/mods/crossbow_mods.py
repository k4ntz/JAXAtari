from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.crossbow_mod_plugins import (
    CursorVisibleOnFireOnlyMod,
    DoubleEnemiesMod, FastCursorMod, LargeCursorMod, FastFriendMod,
)

class CrossbowEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for Crossbow.
    It simply inherits all logic from JaxAtariModController and defines the REGISTRY.
    """

    REGISTRY = {
        "fast_cursor": FastCursorMod,
        "large_cursor": LargeCursorMod,
        "fast_friend": FastFriendMod,
        "cursor_visible_on_fire_only": CursorVisibleOnFireOnlyMod,
        "double_enemies": DoubleEnemiesMod,
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
