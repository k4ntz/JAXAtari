import importlib
import inspect

from jaxatari.environment import JaxEnvironment
from jaxatari.renderers import JAXGameRenderer
from jaxatari.wrappers import JaxatariWrapper



# Map of game names to their module paths
GAME_MODULES = {
    "pong": "jaxatari.games.jax_pong",
    "seaquest": "jaxatari.games.jax_seaquest",
    "kangaroo": "jaxatari.games.jax_kangaroo",
    "freeway": "jaxatari.games.jax_freeway",
    "breakout": "jaxatari.games.jax_breakout",
    # Add new games here
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

def list_available_mods() -> list[str]:
    """Lists all available, registered mods."""
    return list(MOD_MODULES.keys())

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
        env_class = None
        for _, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, JAXGameRenderer) and obj is not JAXGameRenderer:
                env_class = obj
                break # Found it
        
        if env_class is None:
            raise ImportError(f"No JaxRenderer subclass found in {GAME_MODULES[game_name]}")
        
        # 3. Instantiate the class, passing along the arguments, and return it
        # TODO: none of our environments use mode / difficulty yet, but we might want to add it here and in the single envs
        return env_class()
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
        # TODO: none of our environments use mode / difficulty yet, but we might want to add it here and in the single envs
        return wrapper_class(env)

    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to load mod '{mod_name}': {e}") from e