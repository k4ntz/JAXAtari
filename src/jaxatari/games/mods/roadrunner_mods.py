import os
import jax.numpy as jnp
from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.roadrunner.roadrunner_mod_plugins import (
    InvertColorsMod,
    HueShiftMod,
    InvisibleEnemyMod,
    HarmlessRavinesMod,
    NoRoadStripesMod,
    InvisibleTrucksMod,
)


def _invert_palette(palette: jnp.ndarray) -> jnp.ndarray:
    """Invert all colors: RGB -> (255 - RGB)."""
    return (255 - palette).astype(palette.dtype)


def _hue_shift_palette(palette: jnp.ndarray) -> jnp.ndarray:
    """Rotate RGB channels: R->G, G->B, B->R (120 degree hue shift)."""
    # Roll channels: axis -1 shifts (R,G,B) -> (B,R,G) which is a 120° hue rotation
    return jnp.roll(palette, shift=1, axis=-1)


# Map mod classes to their palette transform functions
_PALETTE_TRANSFORMS = {
    InvertColorsMod: _invert_palette,
    HueShiftMod: _hue_shift_palette,
}


class RoadRunnerEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for RoadRunner.
    Inherits all logic from JaxAtariModController and defines the mod registry.
    After standard patching, applies palette transforms for color mods.
    """

    REGISTRY = {
        "invert_colors": InvertColorsMod,
        "hue_shift": HueShiftMod,
        "invisible_enemy": InvisibleEnemyMod,
        "harmless_ravines": HarmlessRavinesMod,
        "no_road_stripes": NoRoadStripesMod,
        "invisible_trucks": InvisibleTrucksMod,
    }

    _mod_sprite_dir = os.path.join(os.path.dirname(__file__), "roadrunner", "sprites")

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

        # Apply palette transforms for any active color mods
        for mod_key in mods_config:
            plugin_class = self.REGISTRY.get(mod_key)
            if plugin_class in _PALETTE_TRANSFORMS:
                transform = _PALETTE_TRANSFORMS[plugin_class]
                self._env.renderer.PALETTE = transform(self._env.renderer.PALETTE)

        # Replace enemy sprites with transparent masks
        if InvisibleEnemyMod in [self.REGISTRY.get(k) for k in mods_config]:
            renderer = self._env.renderer
            transparent_id = renderer.jr.TRANSPARENT_ID
            enemy_sprite_keys = [
                "enemy", "enemy_run1", "enemy_run2",
                "enemy_burnt", "enemy_run_over",
                "enemy_rocket", "enemy_hoverboard1",
            ]
            for key in enemy_sprite_keys:
                if key in renderer.SHAPE_MASKS:
                    shape = renderer.SHAPE_MASKS[key].shape
                    renderer.SHAPE_MASKS[key] = jnp.full(shape, transparent_id, dtype=renderer.SHAPE_MASKS[key].dtype)

        # Replace truck sprites with transparent masks
        if InvisibleTrucksMod in [self.REGISTRY.get(k) for k in mods_config]:
            renderer = self._env.renderer
            transparent_id = renderer.jr.TRANSPARENT_ID
            enemy_sprite_keys = ["truck"]
            for key in enemy_sprite_keys:
                if key in renderer.SHAPE_MASKS:
                    shape = renderer.SHAPE_MASKS[key].shape
                    renderer.SHAPE_MASKS[key] = jnp.full(shape, transparent_id, dtype=renderer.SHAPE_MASKS[key].dtype)
 