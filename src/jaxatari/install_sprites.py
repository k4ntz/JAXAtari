import sys
import requests
import zipfile
import io
import os
import tempfile
import shutil
from pathlib import Path
from platformdirs import user_data_dir

# 1. Configuration
SPRITES_URL = os.environ.get(
    "JAXATARI_SPRITES_URL",
    "https://drive.google.com/uc?export=download&id=1HX2TS8ulXGSnjrzUCAV83cINj0usBTvM",
)
ALT_SPRITES_URL = os.environ.get("JAXATARI_ALT_SPRITES_URL", "https://drive.google.com/uc?export=download&id=1qZ7mber7tcCrOxFsALk7V8_PYoq9HHSr")
STATES_URL = os.environ.get(
    "JAXATARI_STATES_URL",
    "https://drive.google.com/uc?export=download&id=1GRFPXwVJcUhSRTvwWsIUoOMSygmCSrdM",
)
STORAGE_DIR = Path(user_data_dir("jaxatari"))
OWNERSHIP_MARKER_FILE = STORAGE_DIR / ".ownership_confirmed"
ALT_SPRITES_MARKER_FILE = STORAGE_DIR / ".alternative_sprites_installed"
LICENSE_TEXT = """
OWNERSHIP CONFIRMATION
------------------------------------------
I declare to legally own a license to the original Atari 2600 ROMs.
I agree to not distribute these extracted game assets (sprites/states) and wish to proceed.
"""
FALLBACK_NOTICE_TEXT = """
NOTE
------------------------------------------
If you do not have ownership of the original Atari 2600 ROMs, JaxAtari can still be used with replacement/custom sprites.
In that case, the installer will download the alternative sprites package.
You can also use your own sprites by placing them in the ~/.local/share/jaxatari/sprites directory.
"""

def _download_archive(url: str) -> bytes:
    response = requests.get(url, stream=True)
    response.raise_for_status()
    return response.content

def _extract_named_dir(archive_bytes: bytes, folder_name: str, dest_dir: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        archive = zipfile.ZipFile(io.BytesIO(archive_bytes))
        archive.extractall(tmp_path)

        sources = [p for p in tmp_path.rglob(folder_name) if p.is_dir()]
        if not sources:
            raise RuntimeError(f"Invalid archive: missing '{folder_name}/' directory.")

        sources.sort(key=lambda p: len(p.parts))
        target_dir = dest_dir / folder_name
        for src in sources:
            shutil.copytree(src, target_dir, dirs_exist_ok=True)

def _extract_first_existing_dir(
    archive_bytes: bytes,
    folder_names: tuple[str, ...],
    dest_dir: Path,
    target_folder_name: str,
) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        archive = zipfile.ZipFile(io.BytesIO(archive_bytes))
        archive.extractall(tmp_path)

        for folder_name in folder_names:
            sources = [p for p in tmp_path.rglob(folder_name) if p.is_dir()]
            if not sources:
                continue

            sources.sort(key=lambda p: len(p.parts))
            target_dir = dest_dir / target_folder_name
            for src in sources:
                shutil.copytree(src, target_dir, dirs_exist_ok=True)
            return

        expected = ", ".join(f"'{name}/'" for name in folder_names)
        raise RuntimeError(f"Invalid archive: missing one of {expected} directories.")

def download_and_extract():
    auto_accept = os.environ.get("JAXATARI_CONFIRM_OWNERSHIP", "0") == "1"
    accepted_ownership = auto_accept

    if not auto_accept:
        print(FALLBACK_NOTICE_TEXT)
        # A. Display the Gate
        print(LICENSE_TEXT)
        response = input("Do you confirm ownership ? [y/N]: ").strip().lower()
        
        if response not in ('y', 'yes'):
            accepted_ownership = False
            if not ALT_SPRITES_URL:
                print("Declined. Installation aborted.")
                print("Set JAXATARI_ALT_SPRITES_URL to install alternate sprites instead.")
                sys.exit(1)
            print("Ownership declined. Installing alternate sprites instead.")
        else:
            accepted_ownership = True
    else:
        print("Auto-confirming ownership confirmation via environment variable.")
        accepted_ownership = True

    # B. The Download (Only happens if accepted)
    sprites_url = SPRITES_URL if accepted_ownership else ALT_SPRITES_URL
    print(f"Downloading sprites from {sprites_url}...")
    try:
        # Create destination directory
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)

        sprites_archive = _download_archive(sprites_url)
        print("Extracting sprites...")
        if accepted_ownership:
            _extract_named_dir(sprites_archive, "sprites", STORAGE_DIR)
        else:
            # Alternate archives may use either "sprites/" or "custom_sprites/".
            # In both cases, install to the canonical destination: STORAGE_DIR/sprites.
            _extract_first_existing_dir(
                sprites_archive,
                ("sprites", "custom_sprites"),
                STORAGE_DIR,
                "sprites",
            )

        # Optional for backward compatibility: install states if available.
        if accepted_ownership:
            try:
                print(f"Downloading states from {STATES_URL}...")
                states_archive = _download_archive(STATES_URL)
                print("Extracting states...")
                _extract_named_dir(states_archive, "states", STORAGE_DIR)
            except Exception as states_exc:
                print(f"⚠️ States were not installed: {states_exc}")
                print("Continuing with sprites only for backward compatibility.")
        
        # D. Persist install mode so runtime checks can skip prompting.
        if accepted_ownership:
            OWNERSHIP_MARKER_FILE.touch()
            if ALT_SPRITES_MARKER_FILE.exists():
                ALT_SPRITES_MARKER_FILE.unlink()
        else:
            ALT_SPRITES_MARKER_FILE.touch()
            if OWNERSHIP_MARKER_FILE.exists():
                OWNERSHIP_MARKER_FILE.unlink()
        
        print(f"✅ Success! Assets installed to: {STORAGE_DIR}")
        
    except Exception as e:
        print(f"❌ Error downloading/installing assets: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_and_extract()
