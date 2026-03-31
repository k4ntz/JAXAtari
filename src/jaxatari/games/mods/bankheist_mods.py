import os

from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.bankheist.bankheist_mod_plugins import RandomBankSpawnsMod


class BankHeistEnvMod(JaxAtariModController):
    """Game-specific Mod Controller for BankHeist."""

    REGISTRY = {
        "random_spawns": RandomBankSpawnsMod,
    }

    _mod_sprite_dir = os.path.join(os.path.dirname(__file__), "bankheist", "sprites")

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

