from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.phoenix.phoenix_mod_plugins import (
    BossLateMissilesMod,
    InfiniteLivesMod,
    FastPlayerMod,
    InvinciblePlayerMod,
    FastEnemyBulletsMod,
    NoAbilityCooldownMod,
)


class PhoenixEnvMod(JaxAtariModController):
    """
    Game-specific mod controller for Phoenix.
    """

    REGISTRY = {
        "boss_late_missiles": BossLateMissilesMod,
        "infinite_lives": InfiniteLivesMod,
        "fast_player": FastPlayerMod,
        "invincible_player": InvinciblePlayerMod,
        "fast_enemy_bullets": FastEnemyBulletsMod,
        "no_ability_cooldown": NoAbilityCooldownMod,
    }

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

