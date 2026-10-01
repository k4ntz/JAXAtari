import os
from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.lostluggage.lostluggage_mod_plugins import (
    LinearMovementMod,
    AlwaysZeroScoreMod,
    SoftEscapePenaltyMod,
    NoExtraLifeMod,
    MoreSuitcasesMod,
    TimedRoundFastSpawnMod,
    LowerPlayerUpperBoundMod,
    DisappearingSuitcasesMod,
    RandomDisappearSuitcasesMod,
)

class LostluggageEnvMod(JaxAtariModController):
    REGISTRY = {
        "linear_movement": LinearMovementMod,
        "zero_score": AlwaysZeroScoreMod,
        "soft_penalty": SoftEscapePenaltyMod,
        "no_extra_life": NoExtraLifeMod,
        "more_suitcases": MoreSuitcasesMod,
        "fast_spawn": TimedRoundFastSpawnMod,
        "upper_bound": LowerPlayerUpperBoundMod,
        "disappearing_suits": DisappearingSuitcasesMod,
        "random_disappearing": RandomDisappearSuitcasesMod
    }

    _mod_sprite_dir = os.path.join(
        os.path.dirname(__file__),
        "lostluggage",
        "sprites"
    )

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
