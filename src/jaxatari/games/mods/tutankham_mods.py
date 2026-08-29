import os
from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.tutankham.tutankham_mod_plugins import (
    NightModeMod,
    NightModeStepMod,
    MimicMod,
    MimicStepMod,
    UpsideDownMod,
    MovingItemsMod,
    GhostMod,
    ShrinkPlayerMod,
    KnockbackMod,
    WhipMod
)

class TutankhamEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for Tutankham.
    It simply inherits all logic from JaxAtariModController and defines the TUTANKHAM_MOD_REGISTRY.
    """

    REGISTRY = {
        # SIMPLE MODS
        "knockback": KnockbackMod,
        "moving_items": MovingItemsMod,
        "night_mode": ["night_mode_render", "night_mode_step"],  # night modpack
        "night_mode_render": NightModeMod,
        "night_mode_step":   NightModeStepMod,
        "shrink": ShrinkPlayerMod,
        "upsidedown": UpsideDownMod,
        
        # HARD MODS
        "ghost": GhostMod,
        "mimic": ["mimic_render", "mimic_step"],  # mimic modpack
        "mimic_render": MimicMod,
        "mimic_step": MimicStepMod,
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
