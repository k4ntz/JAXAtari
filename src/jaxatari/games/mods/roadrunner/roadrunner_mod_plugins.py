import jax
import jax.numpy as jnp
from functools import partial
from jaxatari.games.jax_roadrunner import RoadRunnerState
from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin
from jaxatari.games.jax_roadrunner import DEFAULT_LEVELS


# --- Color Palette Mods ---

class InvertColorsMod(JaxAtariInternalModPlugin):
    """Inverts all colors in the game palette (RGB -> 255 - RGB)."""
    pass


class HueShiftMod(JaxAtariInternalModPlugin):
    """Shifts all colors in the game palette by rotating hue 120 degrees (R->G->B->R)."""
    pass


# --- Visibility and Sprite Mods ---

class InvisibleEnemyMod(JaxAtariInternalModPlugin):
    """Makes the enemy invisible by replacing all enemy sprites with transparent masks.
    The enemy still exists and can catch the player — it's just not rendered."""
    pass

NO_STRIPES_LEVELS = tuple(
    level._replace(render_road_stripes=False) 
    for level in DEFAULT_LEVELS
)

class NoRoadStripesMod(JaxAtariInternalModPlugin):
    """Always renders the road without stripes by overriding the 
    level configurations directly in the constants at initialization."""
    conflicts_with = []
    
    constants_overrides = {
        "levels": NO_STRIPES_LEVELS
    }

class InvisibleTrucksMod(JaxAtariInternalModPlugin):
    """Makes trucks invisible by replacing all trucj sprites with transparent masks.
    The trucks still exist and can hit the player and enemy — they're just not rendered."""
    pass

# --- Ravine Mod ---

class HarmlessRavinesMod(JaxAtariInternalModPlugin):
    """Deactivates ravines by disabling collision detection.
    Ravines will still spawn and scroll on the screen, but the player will 
    simply walk over them without triggering the falling animation.
    """
    @partial(jax.jit, static_argnums=(0,))
    def _check_ravine_collisions(self, state: RoadRunnerState) -> RoadRunnerState:
        return state