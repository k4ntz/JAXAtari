import os
from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.montezuma_revenge.montezuma_revenge_mod_plugins import (
    InfiniteAmuletMod, SuperJumpMod, FastPlayerMod, NoFallDamageMod,
    RevealMapMod, DebugHudMod, NoEnemiesMod, CenterBouncingSkullMod,
    RollingSkullsMod, MovingSnakesMod, JumpingSpidersMod, SwordKillBonusMod, ThreeSwordsMod
)

# --- The Registry ---
MONTEZUMA_REVENGE_MOD_REGISTRY = {
    "infinite_amulet": InfiniteAmuletMod,
    "super_jump": SuperJumpMod,
    "no_fall_damage": NoFallDamageMod,
    "reveal_map": RevealMapMod,
    "debug_hud": DebugHudMod,
    "no_enemies": NoEnemiesMod,
    "center_bouncing_skull": CenterBouncingSkullMod,
    "fast_player": FastPlayerMod,
    "rolling_skulls": RollingSkullsMod,
    "moving_snakes": MovingSnakesMod,
    "jumping_spiders": JumpingSpidersMod,
    "sword_kill_bonus": SwordKillBonusMod,
    "three_swords": ThreeSwordsMod,
    "alternative_enemies": ["center_bouncing_skull", "rolling_skulls", "moving_snakes", "jumping_spiders"],
    "god_mode": ["infinite_amulet", "no_fall_damage", "no_enemies", "super_jump", "fast_player"]
}

class MontezumaRevengeEnvMod(JaxAtariModController):
    """
    Game-specific (Group 1) Mod Controller for MontezumaRevenge.
    It inherits all logic from JaxAtariModController and defines
    the REGISTRY.
    """

    REGISTRY = MONTEZUMA_REVENGE_MOD_REGISTRY

    # Define the path relative to this file (mod sprites fallback)
    _mod_sprite_dir = os.path.join(os.path.dirname(__file__), "montezuma_revenge", "sprites")

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
