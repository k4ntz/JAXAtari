import importlib
import inspect

from jaxatari.environment import JaxEnvironment
from jaxatari.renderers import JAXGameRenderer
from jaxatari.wrappers import JaxatariWrapper


# Map of game names to their module paths
GAME_MODULES = {
    "amidar": "jaxatari.games.jax_amidar",
    "airraid": "jaxatari.games.jax_airraid",
    "alien": "jaxatari.games.jax_alien",
    "asterix": "jaxatari.games.jax_asterix",
    "asteroids": "jaxatari.games.jax_asteroids",
    "atlantis": "jaxatari.games.jax_atlantis",
    "bankheist": "jaxatari.games.jax_bankheist",
    "berzerk": "jaxatari.games.jax_berzerk",
    "blackjack": "jaxatari.games.jax_blackjack",
    "breakout": "jaxatari.games.jax_breakout",
    "centipede": "jaxatari.games.jax_centipede",
    "choppercommand": "jaxatari.games.jax_choppercommand",
    "enduro": "jaxatari.games.jax_enduro",
    "fishingderby": "jaxatari.games.jax_fishingderby",
    "flagcapture": "jaxatari.games.jax_flagcapture",
    "freeway": "jaxatari.games.jax_freeway",
    "frostbite": "jaxatari.games.jax_frostbite",
    "galaxian": "jaxatari.games.jax_galaxian",
    "hangman": "jaxatari.games.jax_hangman",
    "hauntedhouse": "jaxatari.games.jax_hauntedhouse",
    "humancannonball": "jaxatari.games.jax_humancannonball",
    "kangaroo": "jaxatari.games.jax_kangaroo",
    "kingkong": "jaxatari.games.jax_kingkong",
    "klax": "jaxatari.games.jax_klax",
    "lasergates": "jaxatari.games.jax_lasergates",
    "namethisgame": "jaxatari.games.jax_namethisgame",
    "phoenix": "jaxatari.games.jax_phoenix",
    "pong": "jaxatari.games.jax_pong",
    "qbert": "jaxatari.games.jax_qbert",
    "riverraid": "jaxatari.games.jax_riverraid",
    "seaquest": "jaxatari.games.jax_seaquest",
    "sirlancelot": "jaxatari.games.jax_sirlancelot",
    "skiing": "jaxatari.games.jax_skiing",
    "slotmachine": "jaxatari.games.jax_slotmachine",
    "spaceinvaders": "jaxatari.games.jax_spaceinvaders",
    "spacewar": "jaxatari.games.jax_spacewar",
    # "surround": "jaxatari.games.jax_surround", currently not in a state that can be used
    "tennis": "jaxatari.games.jax_tennis",
    "tetris": "jaxatari.games.jax_tetris",
    "timepilot": "jaxatari.games.jax_timepilot",
    "tron": "jaxatari.games.jax_tron",
    "turmoil": "jaxatari.games.jax_turmoil",
    "venture": "jaxatari.games.jax_venture",
    "videocheckers": "jaxatari.games.jax_videocheckers",
    "videocube": "jaxatari.games.jax_videocube",
    "videopinball": "jaxatari.games.jax_videopinball",
    "wordzapper": "jaxatari.games.jax_wordzapper",
}

MOD_MODULES = {
    "pong": "jaxatari.games.mods.pong_mods",
    "seaquest": "jaxatari.games.mods.seaquest_mods",
    "kangaroo": "jaxatari.games.mods.kangaroo_mods",
    "freeway": "jaxatari.games.mods.freeway_mods",
    "breakout": "jaxatari.games.mods.breakout_mods",
}

def list_available_games() -> list[str]:
    """Lists all available, registered games."""
    return list(GAME_MODULES.keys())

def make(game_name: str, mode: int = 0, difficulty: int = 0) -> JaxEnvironment:
    """
    Creates and returns a JaxAtari game environment instance.
    This is the main entry point for creating environments.

    Args:
        game_name: Name of the game to load (e.g., "pong").
        mode: Game mode.
        difficulty: Game difficulty.

    Returns:
        An instance of the specified game environment.
    """
    if game_name not in GAME_MODULES:
        raise NotImplementedError(
            f"The game '{game_name}' does not exist. Available games: {list_available_games()}"
        )
    
    try:
        # 1. Dynamically load the module
        module = importlib.import_module(GAME_MODULES[game_name])
        
        # 2. Find the correct environment class within the module
        env_class = None
        for _, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, JaxEnvironment) and obj is not JaxEnvironment:
                env_class = obj
                break # Found it
        
        if env_class is None:
            raise ImportError(f"No JaxEnvironment subclass found in {GAME_MODULES[game_name]}")
        
        # 3. Instantiate the class, passing along the arguments, and return it
        # TODO: none of our environments use mode / difficulty yet, but we might want to add it here and in the single envs
        return env_class()

    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to load game '{game_name}': {e}") from e

def make_renderer(game_name: str) -> JAXGameRenderer:
    """
    Creates and returns a JaxAtari game environment renderer.

    Args:
        game_name: Name of the game to load (e.g., "pong").

    Returns:
        An instance of the specified game environment renderer.
    """
    if game_name not in GAME_MODULES:
        raise NotImplementedError(
            f"The game '{game_name}' does not exist. Available games: {list_available_games()}"
        )
    
    try:
        # 1. Dynamically load the module
        module = importlib.import_module(GAME_MODULES[game_name])
        
        # 2. Find the correct environment class within the module
        renderer_class = None
        for _, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, JAXGameRenderer) and obj is not JAXGameRenderer:
                renderer_class = obj
                break # Found it

        if renderer_class is None:
            raise ImportError(f"No AXGameRenderer subclass found in {GAME_MODULES[game_name]}")

        # 3. Instantiate the class, passing along the arguments, and return it
        return renderer_class()
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to load renderer for '{game_name}': {e}") from e
    
def modify(env: JaxEnvironment, game_name: str, mod_name: str) -> JaxatariWrapper:
    """
    Modifies a JaxAtari game environment with a specified modification using wrappers.

    Args:
        env: The JaxAtari game environment to modify.
        mod_name: Name of the modification to apply (e.g., "lazy_enemy").

    Returns:
        An wrapped instance of the specified game environment with the modification applied. 
    """
    try:
        # 1. Dynamically load the module
        module = importlib.import_module(MOD_MODULES[game_name])
        
        # 2. Find the correct environment class within the module
        wrapper_class = None
        for _, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, JaxatariWrapper) and obj.__name__.lower() == mod_name.lower():
                wrapper_class = obj
                break # Found it

        if wrapper_class is None:
            raise ImportError(f"No mod {mod_name} subclass found in {MOD_MODULES[game_name]}")
        
        # 3. Instantiate the class, passing along the arguments, and return it
        return wrapper_class(env)

    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to load mod '{mod_name}': {e}") from e