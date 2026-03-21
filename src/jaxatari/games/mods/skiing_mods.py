import os
from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.skiing.skiing_mod_plugins import MoreTreesMod, MoreMogulsMod, DangerousMogulsMod, JumpMod

class SkiingEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for Skiing.
    It inherits all logic from JaxAtariModController and defines the REGISTRY.
    """

    REGISTRY = {
        "_more_trees": MoreTreesMod,
        "_more_moguls": MoreMogulsMod,
        "_dangerous_moguls": DangerousMogulsMod,
        "jump": JumpMod,
        "off_piste": ["_more_trees", "_more_moguls", "_dangerous_moguls"],
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
