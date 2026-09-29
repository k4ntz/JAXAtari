from jaxatari.games.mods.centipede.centipede_mod_plugins import SlowSpellMod, RandomMushroomsMod, \
    RandomPlayerMovementMod, DeadlyMushroomsMod, FastSpellMod, MaxLivesResetMod, InvincibleMobsMod, FriendlyMobsMod, \
    RandomCentipedeMod, FastCentipedeMod
from jaxatari.modification import JaxAtariModController


class CentipedeEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for Centipede.
    It simply inherits all logic from JaxAtariModController and defines the CENTIPEDE_MOD_REGISTRY.
    """

    REGISTRY = {
        "slow_spell": SlowSpellMod, # Spells have 1/3 the speed
        "fast_spell": FastSpellMod, # Spells have 2x the speed
        "random_mushrooms": RandomMushroomsMod, # Mushrooms are randomly placed on the screen in initialization, new wave and after death
        "random_player_movement": RandomPlayerMovementMod, # Player movement gets randomly altered
        "deadly_mushrooms": DeadlyMushroomsMod, # Mushrooms are deadly and cause the player to lose a life on contact
        "max_lives_reset": MaxLivesResetMod, # Player starts with the maximum number of lives
        "invincible_mobs": InvincibleMobsMod, # Mobs cannot be killed by the player
        "friendly_mobs": FriendlyMobsMod, # Mobs are friendly to the player and do not harm them
        "random_centipede": RandomCentipedeMod, # Centipede spawns are randomized
        "fast_centipede": FastCentipedeMod, # Centipede moves faster each time being hit
    }

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