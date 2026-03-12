import os
from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.doubledunk.doubledunk_mod_plugins import (
    TimerMod,
    SuperDunkMod,
    TenSecondViolationInternalMod,
    TenSecondViolationPostStepMod,
    SingleMode,
    OneVsOneInternalMod,
    OneVsOnePostMod,
    HalfCourtMod
)

class DoubleDunkEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for Double Dunk.
    """

    REGISTRY = {
        "timer_mod": TimerMod,
        "super_dunk": SuperDunkMod,
        "ten_second_violation_internal": TenSecondViolationInternalMod,
        "ten_second_violation_post": TenSecondViolationPostStepMod,
        "ten_second_violation": ["ten_second_violation_internal", "ten_second_violation_post"],
        "single_mode": SingleMode,
        "1v1_internal": OneVsOneInternalMod,
        "1v1_post": OneVsOnePostMod,
        "1v1_mode": ["1v1_internal", "1v1_post"],
        "half_court": HalfCourtMod,
    }

    _mod_sprite_dir = os.path.join(os.path.dirname(__file__), "doubledunk", "sprites")

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