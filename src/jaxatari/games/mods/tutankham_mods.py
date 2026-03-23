from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.tutankham.tutankham_mod_plugins import (
    NightModeMod,
    MimicModeMod,
    UpsideDownMod,
    WhipMod,
)

class TutankhamEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for Tutankham.
    It simply inherits all logic from JaxAtariModController and defines the TUTANKHAM_MOD_REGISTRY.
    """

    REGISTRY = {
        "night_mode": NightModeMod,
        "mimics": MimicModeMod,
        "upsidedown": UpsideDownMod,
        "whip": WhipMod,
    }

    _mod_sprite_dir = os.path.join(os.path.dirname(__file__), "tutankham", "sprites")

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
