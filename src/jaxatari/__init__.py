"""Deprecated compatibility import for the renamed :mod:`jaxtari` package."""

from __future__ import annotations

import warnings

warnings.warn(
    "'jaxatari' was renamed to 'jaxtari' and will be removed in a future "
    "release. Use 'pip install jaxtari' and 'import jaxtari'.",
    DeprecationWarning,
    stacklevel=2,
)

from jaxtari import *  # noqa: F403
from jaxtari import (  # noqa: F401
    ALT_SPRITES_MARKER_FILE,
    DATA_DIR,
    MARKER_FILE,
    check_ownership,
    list_available_games,
    make,
)

__all__ = [
    "ALT_SPRITES_MARKER_FILE",
    "DATA_DIR",
    "MARKER_FILE",
    "check_ownership",
    "list_available_games",
    "make",
]
