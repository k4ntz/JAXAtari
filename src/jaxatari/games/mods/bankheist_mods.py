import os

from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.bankheist.bankheist_mod_plugins import (
    RandomBankSpawnsMod,
    UnlimitedGasMod,
    NoPoliceMod,
    TwoPoliceCarsMod,
    RandomCityMod,
    RevisitCityMod,
    MovingBanksMod,
    DoubleSpeedMod,
    GreyRoadMod,
    RedPoliceCarsMod,
    GoldenBanksMod,
    BluePlayerMod,
    DynamitePenaltyMod,
    FuelForBanksMod,
)

# --- The Registry ---
BANKHEIST_MOD_REGISTRY = {
    "random_spawns": RandomBankSpawnsMod,
    "unlimited_gas": UnlimitedGasMod,
    "no_police": NoPoliceMod,
    "2_police_cars": TwoPoliceCarsMod,
    "random_city": RandomCityMod,
    "revisit_city": RevisitCityMod,
    "moving_banks": MovingBanksMod,
    "double_speed": DoubleSpeedMod,
    "grey_road": GreyRoadMod,
    "red_police_cars": RedPoliceCarsMod,
    "golden_banks": GoldenBanksMod,
    "blue_player": BluePlayerMod,
    "dynamite_penalty": DynamitePenaltyMod,
    "fuel_for_banks": FuelForBanksMod,
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
            registry=self.REGISTRY,
        )
