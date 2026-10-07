"""Temporary import shim for the renamed JAXtari package.

Prefer::

    pip install jaxtari
    import jaxtari

This ``jaxatari`` distribution and import path are a short-lived compatibility
layer and will be removed soon.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "The 'jaxatari' package/import is a temporary compatibility shim and will "
    "be removed soon. Use 'pip install jaxtari' and 'import jaxtari' instead.",
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
