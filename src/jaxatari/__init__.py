from pathlib import Path

from jaxatari.paths import (
    ALT_SPRITES_MARKER_NAME,
    OWNERSHIP_MARKER_NAME,
    REQUIRED_SPRITE_VERSION,
    canonical_storage_dir,
    clear_declined_update,
    declined_update_for_required_version,
    env_flag,
    get_storage_dir,
    has_alt_sprites_marker,
    has_ownership_marker,
    ownership_or_alt_markers_present,
    read_installed_sprite_version,
    record_declined_update,
    sprite_update_needed,
)

# Back-compat aliases: canonical install target (new installs go here).
DATA_DIR = canonical_storage_dir()
MARKER_FILE = DATA_DIR / OWNERSHIP_MARKER_NAME
ALT_SPRITES_MARKER_FILE = DATA_DIR / ALT_SPRITES_MARKER_NAME


def _install_hint() -> str:
    return (
        "    .venv/bin/install-sprites\n"
        "    or\n"
        "    python3 -m jaxatari.install_sprites\n"
    )


def _maybe_refresh_sprites() -> None:
    """Opt-in prompt when the installed sprite pack is behind REQUIRED_SPRITE_VERSION."""
    if env_flag("JAXATARI_SKIP_SPRITE_UPDATE") or env_flag("JAXTARI_SKIP_SPRITE_UPDATE"):
        return

    storage = get_storage_dir()
    if not sprite_update_needed(storage):
        return
    if declined_update_for_required_version(storage):
        return

    installed = read_installed_sprite_version(storage)
    installed_label = "unknown" if installed is None else str(installed)

    auto_update = (
        env_flag("JAXATARI_AUTO_UPDATE_SPRITES")
        or env_flag("JAXTARI_AUTO_UPDATE_SPRITES")
        or env_flag("JAXATARI_CONFIRM_OWNERSHIP")
        or env_flag("JAXTARI_CONFIRM_OWNERSHIP")
    )

    print(
        "\n"
        "⚠️  SPRITE PACK UPDATE AVAILABLE\n"
        "----------------------------------------------------\n"
        f"Installed sprite version: {installed_label}\n"
        f"Required sprite version:  {REQUIRED_SPRITE_VERSION}\n"
        "A newer sprite pack is available for this JaxAtari release.\n"
    )

    should_update = auto_update
    if not auto_update:
        response = input(
            "Download and install the updated sprites now? [y/N]: "
        ).strip().lower()
        should_update = response in ("y", "yes")

    if should_update:
        from jaxatari.install_sprites import download_and_extract

        # Preserve the user's previous pack choice (ownership vs alternative).
        accepted_ownership = has_ownership_marker(storage) or (
            not has_alt_sprites_marker(storage)
        )
        download_and_extract(accepted_ownership=accepted_ownership)
        clear_declined_update(canonical_storage_dir())
        return

    record_declined_update(storage)
    print(
        "Skipping sprite update for now.\n"
        "You can install the new pack later with:\n\n"
        f"{_install_hint()}"
        "----------------------------------------------------\n"
    )


def check_ownership():
    """
    Verifies that sprite assets are installed, and optionally offers a pack update
    when the on-disk ``.version`` is behind the version required by this release.
    """
    if not ownership_or_alt_markers_present():
        raise RuntimeError(
            "\n"
            "❌  SPRITES NOT INSTALLED\n"
            "----------------------------------------------------\n"
            "JaxAtari needs sprite assets before environments can start.\n"
            "You can either confirm your ownership of the original Atari 2600 ROMs and install sprites,\n"
            "or continue with replacement/custom sprites.\n\n"
            "Please run the following command in your terminal:\n\n"
            f"{_install_hint()}"
            "----------------------------------------------------\n"
        )

    _maybe_refresh_sprites()


# ... rest of your package imports ...
from jaxatari.core import make, list_available_games
