"""
Example Road Runner mod plugin templates.

This module currently only contains commented-out examples for internal and
post-step JAX Atari mod plugins. The imports required for these examples
have been intentionally omitted to avoid unnecessary import overhead and
unused-import issues. Uncomment and add the appropriate imports when you
are ready to implement real plugins.
"""

# --- Color Palette Mods ---

class InvertColorsMod(JaxAtariInternalModPlugin):
    """Inverts all colors in the game palette (RGB -> 255 - RGB)."""
    pass


class HueShiftMod(JaxAtariInternalModPlugin):
    """Shifts all colors in the game palette by rotating hue 120 degrees (R->G->B->R)."""
    pass
