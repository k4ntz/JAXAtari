import importlib
import inspect
import jax

from jaxatari.environment import JaxEnvironment
from jaxatari.renderers import JAXGameRenderer
from jaxatari.modification import JaxAtariModWrapper


# Map of game names to their module paths
GAME_MODULES = {
    "pong": "jaxatari.games.jax_pong",
    "seaquest": "jaxatari.games.jax_seaquest",
    "kangaroo": "jaxatari.games.jax_kangaroo",
    "freeway": "jaxatari.games.jax_freeway",
    "breakout": "jaxatari.games.jax_breakout",
    # Add new games here
}

# Mod modules registry: for each game, provide the Controller class path
MOD_MODULES = {
    "pong": "jaxatari.games.mods.pong_mods.PongEnvMod",
    "kangaroo": "jaxatari.games.mods.kangaroo_mods.KangarooEnvMod",
    #"freeway": "jaxatari.games.mods.freeway_mods.FreewayEnvMod",
    #"breakout": "jaxatari.games.mods.breakout_mods.BreakoutEnvMod",
}

def _load_from_string(path: str):
    """Dynamically import an attribute from a module path string."""
    module_path, attr_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)


def list_available_games() -> list[str]:
    """Lists all available, registered games."""
    return list(GAME_MODULES.keys())

def make(game_name: str, 
         mode: int = 0, 
         difficulty: int = 0,
         mods_config: list = None,
         allow_conflicts: bool = False
         ) -> JaxEnvironment:
    """
    Creates and returns a JaxAtari game environment instance.
    This is the main entry point for creating environments.

    If 'mods_config' is provided, this function applies the
    full two-stage modding pipeline:
    1. Pre-scans for constant overrides.
    2. Instantiates the base env with modded constants.
    3. Applies the internal 'JaxAtariModController'.
    4. Wraps the env with the 'JaxAtariModWrapper'.

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
        # 1. Load the base environment *class* (don't instantiate yet)
        module = importlib.import_module(GAME_MODULES[game_name])
        env_class = None
        for _, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, JaxEnvironment) and obj is not JaxEnvironment:
                env_class = obj
                break
        if env_class is None:
            raise ImportError(f"No JaxEnvironment subclass found in {GAME_MODULES[game_name]}")

        # 2. Handle mods if they are requested
        if mods_config:
            if game_name not in MOD_MODULES:
                raise NotImplementedError(f"No mod module defined for '{game_name}'.")

            # 3. Load the Controller and Registry
            ControllerClass = _load_from_string(MOD_MODULES[game_name])
            registry = ControllerClass.REGISTRY

            # 4. --- PRE-SCAN FOR CONSTANT OVERRIDES ---
            base_consts = env_class().consts # Get default constants
            const_overrides = {}
            for mod_key in mods_config:
                if mod_key not in registry:
                    raise ValueError(f"Mod '{mod_key}' not recognized.")
                plugin_class = registry[mod_key]
                if hasattr(plugin_class, "constants_overrides"):
                    const_overrides.update(plugin_class.constants_overrides)
            
            # 5. Create the base env WITH modded constants
            modded_consts = base_consts._replace(**const_overrides)
            base_env = env_class(consts=modded_consts)
            
            # 6. --- BUILD STAGE 1 (Internal Controller) ---
            modded_env = ControllerClass(
                env=base_env,
                mods_config=mods_config,
                allow_conflicts=allow_conflicts
            )
            
            # 7. --- BUILD STAGE 2 (Post-Step Wrapper) ---
            final_env = JaxAtariModWrapper(
                env=modded_env,
                mods_config=mods_config,
                allow_conflicts=allow_conflicts
            )
            
            return final_env

        # 3. If no mods, just return the base env
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
    