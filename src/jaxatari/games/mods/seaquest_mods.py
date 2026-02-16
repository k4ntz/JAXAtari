import os
from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.seaquest.seaquest_mod_plugins import DisableEnemiesMod, NoDiversMod, EnemyMinesMod, FireBallsMod

class SeaquestEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for Seaquest.
    It simply inherits all logic from JaxAtariModController and defines the REGISTRY.
    """

    REGISTRY = {
        "disable_enemies": DisableEnemiesMod,
        "no_divers": NoDiversMod,
        "fireballs": FireBallsMod,
        # "peaceful_enemies": PeacefulEnemiesMod,
        # "lethal_divers": LethalDiversMod,
        # "infinite_oxygen": NoOxygenDepletionMod,
        # "polluted_water": PollutedWaterMod,
        "mines": EnemyMinesMod,
        # "fireball": ReplaceTorpedoWithFireBallMod
    }

    _mod_sprite_dir = os.path.join(os.path.dirname(__file__), "seaquest", "sprites")

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
