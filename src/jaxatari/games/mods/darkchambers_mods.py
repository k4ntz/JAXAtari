from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.darkchambers_mod_plugins import (
    SpeedPotionMod,
    HealPotionMod,
    PoisonPotionMod,
)


class DarkchambersEnvMod(JaxAtariModController):
    """
    Mod Controller for DarkChambers.
    
    Defines the registry of all available mods for the DarkChambers game.
    Each mod can be applied independently or in combination with others.
    
    IMPORTANT: These mods implement OPTION B - Proper Integration:
    - Items spawn with real probability distributions (set in jax_darkchambers.py)
    - Item pickup is detected via bounding box collision
    - Effects are tracked via state variables (timers, cloud position)
    - Effects apply each step based on active timers
    
    Available Mods:
    ---------------
    1. speed_potion
       - New item type that spawns randomly throughout levels
       - On pickup: player gets 2x movement speed for 120 steps
       - Auto-expires after duration
       - Useful for dodging enemies or escaping danger zones
    
    2. heal_potion
       - New item type that spawns randomly throughout levels
       - On pickup: instantly restores player health to MAX_HEALTH
       - Item consumed on use (marked inactive)
       - Strategic timing - use before dangerous encounters
    
    3. poison_potion
       - New item type that spawns randomly throughout levels
       - On pickup: creates poison cloud at player location
       - Cloud lasts 180 steps, damages enemies within 80px radius
       - Tracks position and countdown timer
       - Enables area control and crowd management
    
    Integration Details:
    --------------------
    - Item constants added to jax_darkchambers.py (ITEM_SPEED_POTION=15, etc.)
    - Item colors added to DarkChambersConstants
    - Item effect timers tracked in DarkChambersState
    - Collision detection via AABB (axis-aligned bounding box)
    - All effects use JAX pure functional updates (immutable state)
    
    Usage Examples:
    ---------------
    python3 scripts/play.py -g Darkchambers -m speed_potion
    python3 scripts/play.py -g Darkchambers -m heal_potion poison_potion
    python3 scripts/play.py -g Darkchambers -m speed_potion heal_potion poison_potion
    python3 scripts/play.py -g Darkchambers -m speed_potion heal_potion --allow_conflicts
    """

    REGISTRY = {
        "speed_potion": SpeedPotionMod,
        "heal_potion": HealPotionMod,
        "poison_potion": PoisonPotionMod,
    }

    def __init__(self,
                 env,
                 mods_config: list = [],
                 allow_conflicts: bool = False
                 ):
        """
        Initialize the DarkChambers Mod Controller.
        
        Args:
            env: The base DarkChambers environment to apply mods to
            mods_config: List of mod names to apply (e.g., ["speed_potion", "heal_potion"])
            allow_conflicts: If True, allow conflicting mods (last one wins)
        """

        super().__init__(
            env=env,
            mods_config=mods_config,
            allow_conflicts=allow_conflicts,
            registry=self.REGISTRY
        )
