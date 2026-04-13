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
    "https://github.com/k4ntz/JAXAtari/releases/download/v0.1/sprites.zip",
)
STATES_URL = os.environ.get(
    "JAXATARI_STATES_URL",
    "https://github.com/k4ntz/JAXAtari/releases/download/v0.1/states.zip",
)
STORAGE_DIR = Path(user_data_dir("jaxatari"))
LICENSE_TEXT = """
OWNERSHIP CONFIRMATION
------------------------------------------
I declare to legally own a license to the original Atari 2600 ROMs.
I agree to not distribute these extracted game assets (sprites/states) and wish to proceed.
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

def download_and_extract():
    auto_accept = os.environ.get("JAXATARI_CONFIRM_OWNERSHIP", "0") == "1"

    if not auto_accept:
        # A. Display the Gate
        print(LICENSE_TEXT)
        response = input("Do you confirm ownership ? [y/N]: ").strip().lower()
        
        if response not in ('y', 'yes'):
            print("Declined. Installation aborted.")
            sys.exit(1)
    else:
        print("Auto-confirming ownership confirmation via environment variable.")

    # B. The Download (Only happens if accepted)
    print(f"Downloading sprites from {SPRITES_URL}...")
    try:
        # Create destination directory
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)

        sprites_archive = _download_archive(SPRITES_URL)
        print("Extracting sprites...")
        _extract_named_dir(sprites_archive, "sprites", STORAGE_DIR)

        # Optional for backward compatibility: install states if available.
        try:
            print(f"Downloading states from {STATES_URL}...")
            states_archive = _download_archive(STATES_URL)
            print("Extracting states...")
            _extract_named_dir(states_archive, "states", STORAGE_DIR)
        except Exception as states_exc:
            print(f"⚠️ States were not installed: {states_exc}")
            print("Continuing with sprites only for backward compatibility.")
        
        # D. Mark as accepted (Optional, for your internal logic)
        (STORAGE_DIR / ".ownership_confirmed").touch()
        
        print(f"✅ Success! Assets installed to: {STORAGE_DIR}")
        
    except Exception as e:
        print(f"❌ Error downloading/installing assets: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_and_extract()