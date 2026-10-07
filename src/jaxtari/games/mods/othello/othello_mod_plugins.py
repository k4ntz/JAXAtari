from jaxtari.modification import JaxtariInternalModPlugin


class InstantMovementMod(JaxtariInternalModPlugin):
    """Move the cursor every held frame instead of waiting for ALE's 8-frame delay."""

    constants_overrides = {
        "CURSOR_MOVE_DELAY": 1,
    }
