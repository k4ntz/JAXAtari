from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.gravitar.gravitar_mod_plugins import RapidFireMod


class GravitarEnvMod(JaxAtariModController):
    """Game-specific Mod Controller for Gravitar."""

    REGISTRY = {
        "rapid_fire": RapidFireMod,
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
