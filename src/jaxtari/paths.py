"""Filesystem locations for sprite/state assets.

Canonical install root is ``~/.local/share/jaxtari``. Legacy installs under
``~/.local/share/jaxatari`` are still discovered and used until the user
reinstalls into the new location.
"""

from __future__ import annotations

import os
from pathlib import Path

from platformdirs import user_data_dir

# Canonical appdir name after the repo rename.
APP_NAME = "jaxtari"
# Legacy appdir — keep this literal so older installs keep working.
LEGACY_APP_NAME = "jaxatari"

# Bump when the remote sprite zip contents change in a way that needs a refresh.
REQUIRED_SPRITE_VERSION = 1
VERSION_FILENAME = ".version"
DECLINED_UPDATE_FILENAME = ".sprite_update_declined"
OWNERSHIP_MARKER_NAME = ".ownership_confirmed"
ALT_SPRITES_MARKER_NAME = ".alternative_sprites_installed"


def canonical_storage_dir() -> Path:
    """Preferred install location (``~/.local/share/jaxtari``)."""
    return Path(user_data_dir(APP_NAME))


def legacy_storage_dir() -> Path:
    """Pre-rename install location (``~/.local/share/jaxatari``)."""
    return Path(user_data_dir(LEGACY_APP_NAME))


def _marker_present(storage: Path) -> bool:
    return (storage / OWNERSHIP_MARKER_NAME).exists() or (
        storage / ALT_SPRITES_MARKER_NAME
    ).exists()


def _sprites_present(storage: Path) -> bool:
    sprites = storage / "sprites"
    if not sprites.is_dir():
        return False
    try:
        next(sprites.iterdir())
    except StopIteration:
        return False
    return True


def get_storage_dir() -> Path:
    """Resolve the active asset root.

    Prefer the canonical ``jaxtari`` directory when it already has assets or
    install markers; otherwise fall back to a legacy ``jaxtari`` install.
    New installs always target the canonical directory.
    """
    preferred = canonical_storage_dir()
    legacy = legacy_storage_dir()
    if _sprites_present(preferred) or _marker_present(preferred):
        return preferred
    if _sprites_present(legacy) or _marker_present(legacy):
        return legacy
    return preferred


def get_base_sprite_dir() -> str:
    """Directory containing per-game sprite packs."""
    return str(get_storage_dir() / "sprites")


def get_base_state_dir() -> str:
    """Directory containing downloadable game states."""
    return str(get_storage_dir() / "states")


def get_game_state_dir(game_name: str) -> str:
    return str(Path(get_base_state_dir()) / game_name)


def ownership_or_alt_markers_present() -> bool:
    """True if any known storage root has an install-mode marker."""
    for base in {canonical_storage_dir(), legacy_storage_dir(), get_storage_dir()}:
        if _marker_present(base):
            return True
    return False


def has_ownership_marker(storage: Path | None = None) -> bool:
    storage = storage or get_storage_dir()
    if (storage / OWNERSHIP_MARKER_NAME).exists():
        return True
    # Also honour a marker left only under the other root.
    other = legacy_storage_dir() if storage == canonical_storage_dir() else canonical_storage_dir()
    return (other / OWNERSHIP_MARKER_NAME).exists()


def has_alt_sprites_marker(storage: Path | None = None) -> bool:
    storage = storage or get_storage_dir()
    if (storage / ALT_SPRITES_MARKER_NAME).exists():
        return True
    other = legacy_storage_dir() if storage == canonical_storage_dir() else canonical_storage_dir()
    return (other / ALT_SPRITES_MARKER_NAME).exists()


def _version_file(storage: Path) -> Path:
    sprites_version = storage / "sprites" / VERSION_FILENAME
    if sprites_version.exists():
        return sprites_version
    return storage / VERSION_FILENAME


def read_installed_sprite_version(storage: Path | None = None) -> int | None:
    """Return the installed sprite pack version, or ``None`` if missing/unreadable."""
    storage = storage or get_storage_dir()
    version_path = _version_file(storage)
    if not version_path.exists():
        return None
    try:
        return int(version_path.read_text(encoding="utf-8").strip().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def write_sprite_version(
    storage: Path | None = None, version: int = REQUIRED_SPRITE_VERSION
) -> None:
    """Stamp ``sprites/.version`` after a successful install."""
    storage = storage or canonical_storage_dir()
    sprites = storage / "sprites"
    sprites.mkdir(parents=True, exist_ok=True)
    (sprites / VERSION_FILENAME).write_text(f"{version}\n", encoding="utf-8")


def sprite_update_needed(storage: Path | None = None) -> bool:
    installed = read_installed_sprite_version(storage)
    if installed is None:
        # Treat missing version as older than any stamped pack.
        return _sprites_present(storage or get_storage_dir())
    return installed < REQUIRED_SPRITE_VERSION


def declined_update_for_required_version(storage: Path | None = None) -> bool:
    storage = storage or get_storage_dir()
    declined = storage / DECLINED_UPDATE_FILENAME
    if not declined.exists():
        return False
    try:
        return int(declined.read_text(encoding="utf-8").strip().split()[0]) >= REQUIRED_SPRITE_VERSION
    except (OSError, ValueError, IndexError):
        return False


def record_declined_update(storage: Path | None = None) -> None:
    storage = storage or get_storage_dir()
    storage.mkdir(parents=True, exist_ok=True)
    (storage / DECLINED_UPDATE_FILENAME).write_text(
        f"{REQUIRED_SPRITE_VERSION}\n", encoding="utf-8"
    )


def clear_declined_update(storage: Path | None = None) -> None:
    storage = storage or get_storage_dir()
    declined = storage / DECLINED_UPDATE_FILENAME
    if declined.exists():
        declined.unlink()


def env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "y"}
