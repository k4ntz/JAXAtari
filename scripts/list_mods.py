import importlib
import sys
import os
import warnings
import textwrap

# Suppress noisy warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

# Add src to path so we can import jaxatari if it's not installed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from jaxatari.core import GAME_MODULES, MOD_MODULES

def _load_from_string(path: str):
    """Dynamically import an attribute from a module path string."""
    module_path, attr_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)

def list_mods():
    all_games = sorted(GAME_MODULES.keys())
    
    print(f"{'GAME':<20} | MODS")
    print("-" * 80)
    
    for game in all_games:
        mods = []
        if game in MOD_MODULES:
            try:
                controller_class = _load_from_string(MOD_MODULES[game])
                if hasattr(controller_class, 'REGISTRY'):
                    # Get all keys from the registry
                    mods = sorted(controller_class.REGISTRY.keys())
            except Exception as e:
                mods = [f"Error loading mods: {e}"]
        
        if mods:
            mod_str = ", ".join(mods)
        else:
            mod_str = "(no mods)"
            
        # Wrap the mod string for better readability
        wrapped_mods = textwrap.wrap(mod_str, width=57)
        
        if not wrapped_mods:
            print(f"{game:<20} | (no mods)")
        else:
            print(f"{game:<20} | {wrapped_mods[0]}")
            for line in wrapped_mods[1:]:
                print(f"{' ':<20} | {line}")

if __name__ == "__main__":
    list_mods()
