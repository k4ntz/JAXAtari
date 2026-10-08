import os
from jaxtari.modification import JaxtariModController
from jaxtari.games.mods.spaceinvaders.spaceinvaders_mod_plugins import (
    DisableShieldLeftMod,
    DisableShieldMiddleMod,
    DisableShieldRightMod,
    ShiftShieldsMod,
    ControllableMissileMod,
    NoDangerMod
)

class SpaceInvadersEnvMod(JaxtariModController):    
    """
    Game-specific Mod Controller for SpaceInvaders.
    """

    REGISTRY = {
        "disable_shield_left": DisableShieldLeftMod,
        "disable_shield_middle": DisableShieldMiddleMod,
        "disable_shield_right": DisableShieldRightMod,
        "shift_shields": ShiftShieldsMod,
        "controllable_missile": ControllableMissileMod,
        "no_danger": NoDangerMod,
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
