"""Amidar mod plugins preserved from PR #161 (adapted to the current mod system)."""

from __future__ import annotations

import jax.numpy as jnp

from jaxatari.modification import JaxAtariInternalModPlugin


class NoEnemiesMod(JaxAtariInternalModPlugin):
    """Disable enemies while keeping the original maze geometry.

    Equivalent to the NoEnemiesMaze wrapper from PR #161: MAX_ENEMIES=0 and an
    empty INITIAL_ENEMY_POSITIONS array so reset/step keep shape (0, 2).
    """

    name = "no_enemies"
    constants_overrides = {
        "MAX_ENEMIES": 0,
        "START_ENEMIES": 0,
        "INITIAL_ENEMY_POSITIONS": jnp.empty((0, 2), dtype=jnp.int32),
    }
