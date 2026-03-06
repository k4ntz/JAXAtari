import os
from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.boxing.boxing_mod_plugins import (
    StaticEnemyMod,
    RandomWalkEnemyMod,
    MirrorEnemyMod,
    FastStunMod,
    SlowStunMod,
    BodyHitsScoreMod,
    InfiniteTimeMod,
    OneHitKOMod,
    DoubleSpeedMod,
)


class BoxingEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for Boxing.
    """

    REGISTRY = {
        "static_enemy": StaticEnemyMod,
        "random_walk_enemy": RandomWalkEnemyMod,
        "mirror_enemy": MirrorEnemyMod,
        "fast_stun": FastStunMod,
        "slow_stun": SlowStunMod,
        "body_hits_score": BodyHitsScoreMod,
        "infinite_time": InfiniteTimeMod,
        "one_hit_ko": OneHitKOMod,
        "double_speed": DoubleSpeedMod,
    }

    _mod_sprite_dir = os.path.join(os.path.dirname(__file__), "boxing", "sprites")

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
