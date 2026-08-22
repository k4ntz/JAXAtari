from jaxatari.modification import JaxAtariInternalModPlugin


class InstantMovementMod(JaxAtariInternalModPlugin):
    """Move the cursor every held frame instead of waiting for ALE's 8-frame delay."""

    constants_overrides = {
        "CURSOR_MOVE_DELAY": 1,
    }
