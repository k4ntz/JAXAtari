import importlib
import inspect

from jaxatari.environment import JaxEnvironment
from jaxatari.renderers import JAXGameRenderer
from jaxatari.modification import apply_modifications
from jaxatari.wrappers import JaxatariWrapper



# Map of game names to their module paths
GAME_MODULES = {
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
    "freeway": "jaxatari.games.jax_freeway",
    "frostbite": "jaxatari.games.jax_frostbite",
    "galaxian": "jaxatari.games.jax_galaxian",
    "haunted_house": "jaxatari.games.jax_haunted_house",
    "human_cannonball": "jaxatari.games.jax_human_cannonball",
    "kangaroo": "jaxatari.games.jax_kangaroo",
    "kingkong": "jaxatari.games.jax_kingkong",
    "klax": "jaxatari.games.jax_klax",
    "lasergates": "jaxatari.games.jax_lasergates",
    "namethisgame": "jaxatari.games.jax_namethisgame",
    "phoenix": "jaxatari.games.jax_phoenix",
    "pong": "jaxatari.games.jax_pong",
    "riverraid": "jaxatari.games.jax_riverraid",
    "seaquest": "jaxatari.games.jax_seaquest",
    "sir_lancelot": "jaxatari.games.jax_sir_lancelot",
    "skiing": "jaxatari.games.jax_skiing",
    "slotmachine": "jaxatari.games.jax_slotmachine",
    "spaceinvaders": "jaxatari.games.jax_spaceinvaders",
    "spacewar": "jaxatari.games.jax_spacewar",
    "surround": "jaxatari.games.jax_surround",
    "tennis": "jaxatari.games.jax_tennis",
    "tetris": "jaxatari.games.jax_tetris",
    "timepilot": "jaxatari.games.jax_timepilot",
    "tron": "jaxatari.games.jax_tron",
    "turmoil": "jaxatari.games.jax_turmoil",
    "videocheckers": "jaxatari.games.jax_videocheckers",
    "videocube": "jaxatari.games.jax_videocube",
    "videopinball": "jaxatari.games.jax_videopinball",
    "wordzapper": "jaxatari.games.jax_wordzapper",
    # Add new games here

    "darkchambers": "jaxatari.games.jax_darkchambers_demo_julius",
    "darkchambers2": "jaxatari.games.jax_darkchambers_paolo"

}

# Mod modules registry: for each game, provide the Controller class path
MOD_MODULES = {
    "pong": "jaxatari.games.mods.pong_mods.PongEnvMod",
    "kangaroo": "jaxatari.games.mods.kangaroo_mods.KangarooEnvMod",
    "freeway": "jaxatari.games.mods.freeway_mods.FreewayEnvMod",
    "breakout": "jaxatari.games.mods.breakout_mods.BreakoutEnvMod",
    "seaquest": "jaxatari.games.mods.seaquest_mods.SeaquestEnvMod",
}


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
        # 1. Load the base environment class
        module = importlib.import_module(GAME_MODULES[game_name])
        env_class = None
        for _, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, JaxEnvironment) and obj is not JaxEnvironment:
                env_class = obj
                break
        if env_class is None:
            raise ImportError(f"No JaxEnvironment subclass found in {GAME_MODULES[game_name]}")

        # 2. Get default constants
        base_consts = env_class().consts

        # 3. Handle mods if requested
        if mods_config:
            return apply_modifications(
                game_name=game_name,
                mods_config=mods_config,
                allow_conflicts=allow_conflicts,
                base_consts=base_consts,
                env_class=env_class,
                MOD_MODULES=MOD_MODULES
            )

        # No mods: return default base env with default constants
        return env_class(consts=base_consts)

    except (ImportError, AttributeError, ValueError, NotImplementedError) as e:
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
