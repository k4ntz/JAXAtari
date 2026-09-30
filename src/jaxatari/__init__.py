from pathlib import Path
from platformdirs import user_data_dir

# 1. Define the path (Must match the installer script exactly)
# appname="jaxatari", appauthor="mycompany" (or whatever you used)
DATA_DIR = Path(user_data_dir("jaxatari"))
MARKER_FILE = DATA_DIR / ".ownership_confirmed"
ALT_SPRITES_MARKER_FILE = DATA_DIR / ".alternative_sprites_installed"

def check_ownership():
    """
    Verifies that the user has accepted the license and confirmed ownership
    of the original hardware/software by looking for the marker file.
    """
    if not (MARKER_FILE.exists() or ALT_SPRITES_MARKER_FILE.exists()):
        # Raise a clear, blocking error
        raise RuntimeError(
            "\n"
            "❌  SPRITES NOT INSTALLED\n"
            "----------------------------------------------------\n"
            "JaxAtari needs sprite assets before environments can start.\n"
            "You can either confirm your ownership of the original Atari 2600 ROMs and install sprites,\n"
            "or continue with replacement/custom sprites.\n\n"
            "Please run the following command in your terminal:\n\n"
            "    .venv/bin/install-sprites\n"
            "    or\n"
            "    python3 scripts/install_sprites.py\n"
            "----------------------------------------------------\n"
        )

# ... rest of your package imports ...
from jaxatari.core import make, list_available_games
