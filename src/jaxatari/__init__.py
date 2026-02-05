from pathlib import Path
from platformdirs import user_data_dir

# 1. Define the path (Must match the installer script exactly)
# appname="jaxatari", appauthor="mycompany" (or whatever you used)
DATA_DIR = Path(user_data_dir("jaxatari"))
MARKER_FILE = DATA_DIR / ".ownership_confirmed"

def check_ownership():
    """
    Verifies that the user has accepted the license and confirmed ownership
    of the original hardware/software by looking for the marker file.
    """
    if not MARKER_FILE.exists():
        # Raise a clear, blocking error
        raise RuntimeError(
            "\n"
            "❌  OWNERSHIP NOT CONFIRMED\n"
            "----------------------------------------------------\n"
            "To use JaxAtari, you must confirm ownership of the original \n"
            " Atari 2600 ROMs and download the required assets.\n\n"
            "Please run the following command in your terminal:\n\n"
            "    .venv/bin/install-sprites\n"
            "----------------------------------------------------\n"
        )

# ... rest of your package imports ...
from jaxatari.core import make, make_renderer, list_available_games