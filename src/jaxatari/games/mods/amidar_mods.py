"""Amidar env mod controller (content preserved from PR #161, modern registry)."""

from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.amidar.amidar_mod_plugins import NoEnemiesMod


class AmidarEnvMod(JaxAtariModController):
    """Game-specific mod controller for Amidar."""

    REGISTRY = {
        "no_enemies": NoEnemiesMod,
        # Alias matching the old wrapper name from PR #161 / play.py docs.
        "NoEnemiesMaze": NoEnemiesMod,
    }

    def __init__(self, env, mods_config: list = [], allow_conflicts: bool = False):
        super().__init__(
            env=env,
            mods_config=mods_config,
            allow_conflicts=allow_conflicts,
            registry=self.REGISTRY,
        )
