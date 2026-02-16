import importlib
import inspect
import warnings

from jaxatari.environment import JaxEnvironment
from jaxatari.renderers import JAXGameRenderer
from jaxatari.modification import apply_modifications
from jaxatari.wrappers import JaxatariWrapper

from . import check_ownership


def _warn_deprecated_obs_to_flat_array(env: JaxEnvironment) -> None:
    """Warn if legacy obs_to_flat_array is present on the environment."""
    if hasattr(env, "obs_to_flat_array") and callable(getattr(env, "obs_to_flat_array")):
        warnings.warn(
            "Environment exposes deprecated obs_to_flat_array(). "
            "Observations should now be flax.struct.dataclasses using ObjectObservation "
            "for objects or plain arrays for observations like lives, score, etc. "
            "Depending on legacy obs_to_flat_array might lead to unforseen issues with wrappers.",
            DeprecationWarning,
            stacklevel=2,
        )



# Map of game names to their module paths
GAME_MODULES = {
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
    "doubledunk": "jaxatari.games.jax_doubledunk",
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
    "videocheckers": "jaxatari.games.jax_videocheckers",
    "videocube": "jaxatari.games.jax_videocube",
    "videopinball": "jaxatari.games.jax_videopinball",
    "wordzapper": "jaxatari.games.jax_wordzapper",
    # Add new games here
}

# Mod modules registry: for each game, provide the Controller class path
MOD_MODULES = {
    "pong": "jaxatari.games.mods.pong_mods.PongEnvMod",
    "kangaroo": "jaxatari.games.mods.kangaroo_mods.KangarooEnvMod",
    "freeway": "jaxatari.games.mods.freeway_mods.FreewayEnvMod",
    "breakout": "jaxatari.games.mods.breakout_mods.BreakoutEnvMod",
    "seaquest": "jaxatari.games.mods.seaquest_mods.SeaquestEnvMod",
    "doubledunk": "jaxatari.games.mods.doubledunk_mods.DoubleDunkEnvMod",
    "videopinball": "jaxatari.games.mods.videopinball_mods.VideoPinballEnvMod",
    'tennis': "jaxatari.games.mods.tennis_mods.TennisEnvMod",
}


def list_available_games() -> list[str]:
    """Lists all available, registered games."""
    return list(GAME_MODULES.keys())


def make(game_name: str, 
         mode: int = 0, 
         difficulty: int = 0,
         mods_config: list = None, # deprecated, output warning if its used
         mods: list = None,
         allow_conflicts: bool = False
         ) -> JaxEnvironment:
    """
    Creates and returns a JaxAtari game environment instance.
    This is the main entry point for creating environments.

    If 'mods' is provided, this function applies the
    full two-stage modding pipeline:
    1. Pre-scans for constant overrides.
    2. Instantiates the base env with modded constants.
    3. Applies the internal 'JaxAtariModController'.
    4. Wraps the env with the 'JaxAtariModWrapper'.

    Args:
        game_name: Name of the game to load (e.g., "pong").
        mode: Game mode.
        difficulty: Game difficulty.
        mods: List of modifications to apply (default: None).
        allow_conflicts: Whether to allow conflicting mods (default: False).
    Returns:
        An instance of the specified game environment.
    """

    check_ownership()  # Ensure ownership confirmed

    if isinstance(game_name, str):
        game_name = game_name.lower()

    if mods_config is not None:
        warnings.warn(
            "'mods_config' is deprecated and will be removed in future versions. "
            "Please use 'mods' instead.",
            DeprecationWarning
        )
        mods = mods_config

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
        if mods:
            try:
                env = apply_modifications(
                    game_name=game_name,
                    mods_config=mods,
                    allow_conflicts=allow_conflicts,
                    base_consts=base_consts,
                    env_class=env_class,
                    MOD_MODULES=MOD_MODULES
                )
                _warn_deprecated_obs_to_flat_array(env)
                return env
            except NotImplementedError as e:
                # Mod module not defined for this game - fall back to base environment
                warnings.warn(
                    f"Mods requested for '{game_name}' but no mod module is available. "
                    f"Creating base environment without mods. Error: {e}",
                    UserWarning
                )

        # No mods: return default base env with default constants
        env = env_class(consts=base_consts)
        _warn_deprecated_obs_to_flat_array(env)
        return env

    except (ImportError, NotImplementedError) as e:
        # Only wrap registration/import errors - let intentional errors (ValueError, etc.) propagate
        raise ImportError(f"Failed to load game '{game_name}': {e}") from e
