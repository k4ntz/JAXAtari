from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.phoenix.phoenix_mod_plugins import BossLateMissilesMod


class PhoenixEnvMod(JaxAtariModController):
    """
    Game-specific mod controller for Phoenix.
    """

    REGISTRY = {
        "boss_late_missiles": BossLateMissilesMod,
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

