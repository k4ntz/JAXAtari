import os
from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.donkeykong.donkeykong_mod_plugins import (
    SpeedrunnerMod,
    AggressiveBarrelsMod,
    PacifistMod,
    ShiftedLaddersMod,
    NoBarrelsMod
)

class DonkeyKongEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for Donkey Kong.
    """

    REGISTRY = {
        "speedrunner": SpeedrunnerMod,
        "aggressive_barrels": AggressiveBarrelsMod,
        "pacifist": PacifistMod,
        "shifted_ladders": ShiftedLaddersMod,
        "no_barrels": NoBarrelsMod,
    }

    _mod_sprite_dir = os.path.join(os.path.dirname(__file__), "donkeykong", "sprites")

    def __init__(self, env, mods_config: list = [], allow_conflicts: bool = False):
        super().__init__(
            env=env,
            mods_config=mods_config,
            allow_conflicts=allow_conflicts,
            registry=self.REGISTRY
        )
