import os
from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.wizardofwor.wizardofwor_mod_plugins import (
    SkipStartupWaitMod,
    AutoSpawnMod,
)


class WizardOfWorEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for Wizard of Wor.
    """

    REGISTRY = {
        "skip_startup_wait": SkipStartupWaitMod,
        "auto_spawn": AutoSpawnMod,
        # Skip intro freeze and spawn into the maze immediately.
        "skip_intro": ["skip_startup_wait", "auto_spawn"],
    }

    _mod_sprite_dir = os.path.join(os.path.dirname(__file__), "wizardofwor", "sprites")

    def __init__(
        self,
        env,
        mods_config: list = [],
        allow_conflicts: bool = False,
    ):
        super().__init__(
            env=env,
            mods_config=mods_config,
            allow_conflicts=allow_conflicts,
            registry=self.REGISTRY,
        )
