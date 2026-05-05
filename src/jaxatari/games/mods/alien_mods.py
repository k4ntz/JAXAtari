import os
from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.alien.alien_mod_plugins import (
    LastEggMod, EndGameMod, MatrixMod, 
    PacifistMod, AggressiveSwarmMod, DontKillMod, 
    ShortCircuitMod, ExtraLivesMod
)

class AlienEnvMod(JaxAtariModController):    
    """
    Game-specific Mod Controller for Alien.
    It simply inherits all logic from JaxAtariModController and defines the Alien_MOD_REGISTRY.
    """

    REGISTRY = {
        "last_egg": LastEggMod,
        "end_game": EndGameMod,
        "matrix_theme": MatrixMod,
        "pacifist_mode": PacifistMod,
        "aggressive_swarm": AggressiveSwarmMod,
        "dont_kill": DontKillMod,
        "short_circuit": ShortCircuitMod,
        "extra_lives": ExtraLivesMod,
    }

    _mod_sprite_dir = os.path.join(os.path.dirname(__file__), "alien", "sprites")

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
