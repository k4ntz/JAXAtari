from jaxatari.modification import JaxAtariInternalModPlugin


class SkipStartupWaitMod(JaxAtariInternalModPlugin):
    """Skip the ~210-frame ALE intro freeze so gameplay (and spawn) can start immediately."""

    constants_overrides = {
        "STARTUP_FREEZE_FRAMES": 0,
    }


class AutoSpawnMod(JaxAtariInternalModPlugin):
    """Spawn the player into the maze on reset without requiring a button press."""

    constants_overrides = {
        "REQUIRE_SPAWN_INPUT": False,
    }
