import jax
import jax.numpy as jnp
from functools import partial
from jaxatari.games.jax_roadrunner import RoadRunnerState
from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin


# --- Color Palette Mods ---

class InvertColorsMod(JaxAtariInternalModPlugin):
    """Inverts all colors in the game palette (RGB -> 255 - RGB)."""
    pass


class HueShiftMod(JaxAtariInternalModPlugin):
    """Shifts all colors in the game palette by rotating hue 120 degrees (R->G->B->R)."""
    pass


# --- Visibility Mods ---

class InvisibleEnemyMod(JaxAtariInternalModPlugin):
    """Makes the enemy invisible by replacing all enemy sprites with transparent masks.
    The enemy still exists and can catch the player — it's just not rendered."""
    pass
