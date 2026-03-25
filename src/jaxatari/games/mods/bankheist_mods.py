import os
from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.bankheist.bankheist_mod_plugins import (
    UnlimitedGasMod, NoPoliceMod, TwoPoliceCarsMod, RandomCityMod, RevisitCityMod
)

# --- The Registry ---
BANKHEIST_MOD_REGISTRY = {
    "unlimited_gas": UnlimitedGasMod,
    "no_police": NoPoliceMod,
    "2_police_cars": TwoPoliceCarsMod,
    "random_city": RandomCityMod,
    "revisit_city": RevisitCityMod,
}

class BankHeistEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for BankHeist.
    It inherits all logic from JaxAtariModController and defines
    the REGISTRY.
    """

    REGISTRY = BANKHEIST_MOD_REGISTRY

    # Define the path relative to this file (mod sprites fallback)
    _mod_sprite_dir = os.path.join(os.path.dirname(__file__), "bankheist", "sprites")

    def __init__(self,
                 env,
                 mods_config: list = [],
                 allow_conflicts: bool = True
                 ):
        super().__init__(
            env=env,
            mods_config=mods_config,
            allow_conflicts=allow_conflicts,
            registry=self.REGISTRY
        )