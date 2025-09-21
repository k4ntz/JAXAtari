import importlib
import inspect

from jaxatari.environment import JaxEnvironment


# Map of game names to their module paths
GAME_MODULES = {
    "pong": "jaxatari.games.jax_pong",
    "seaquest": "jaxatari.games.jax_seaquest",
    "kangaroo": "jaxatari.games.jax_kangaroo",
    "freeway": "jaxatari.games.jax_freeway",
    "breakout": "jaxatari.games.jax_breakout",
    # Add new games here
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