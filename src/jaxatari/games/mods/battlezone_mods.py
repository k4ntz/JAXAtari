import os
from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.battlezone.battlezone_mod_plugins import GlobalVisionMod, SmallFOVMod, AfkEnemiesMod

class BattlezoneEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for Pong.
    It simply inherits all logic from JaxAtariModController and defines the PONG_MOD_REGISTRY.
    """

    REGISTRY = {
        "global_vision": GlobalVisionMod,
        "small_fov": SmallFOVMod,
        "afk_enemies": AfkEnemiesMod

    }

    _mod_sprite_dir = os.path.join(os.path.dirname(__file__), "battlezone", "sprites")

    def __init__(self,
                 env,
                 mods_config: list = [],
                 allow_conflicts: bool = False
                 ):

        super().__init__(
            env=env,
            mods_config=mods_config,
            allow_conflicts=allow_conflicts,
            registry=self.REGISTRY  # for pong this is the only specific part, but other games might need to do execute some other logic in the constructor.
        )