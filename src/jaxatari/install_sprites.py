import sys
import shutil
import requests
import zipfile
import io
import os
from pathlib import Path
from platformdirs import user_data_dir

# 1. Configuration
ASSET_URL = "https://github.com/k4ntz/JAXAtari/releases/download/v0.1/sprites.zip"
STORAGE_DIR = Path(user_data_dir("jaxatari"))
LICENSE_TEXT = """
OWNERSHIP CONFIRMATION
------------------------------------------
I declare to legally own a license to the original Atari 2600 ROMs.
I agree to not distribute these extracted game sprites and wish to proceed.
"""

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
    print(f"Downloading assets from {ASSET_URL}...")
    try:
        r = requests.get(ASSET_URL, stream=True)
        r.raise_for_status()
        
        # Create destination directory
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        
        # C. Extract (Assuming it's a zip)
        print("Extracting files...")
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(STORAGE_DIR)
        
        # D. Mark as accepted (Optional, for your internal logic)
        (STORAGE_DIR / ".ownership_confirmed").touch()
        
        print(f"✅ Success! Assets installed to: {STORAGE_DIR}")
        
    except Exception as e:
        print(f"❌ Error downloading/installing assets: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_and_extract()