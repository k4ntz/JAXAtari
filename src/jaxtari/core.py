import importlib
import inspect
import warnings

from jaxtari.environment import JaxEnvironment
from jaxtari.renderers import JAXGameRenderer
from jaxtari.modification import apply_modifications
from jaxtari.wrappers import JaxtariWrapper

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



# Map of game names to their module paths (commented out games are WIP and will be supported in the near future)
GAME_MODULES = {
    "amidar": "jaxtari.games.jax_amidar",
    "airraid": "jaxtari.games.jax_airraid",
    "alien": "jaxtari.games.jax_alien",
    "asterix": "jaxtari.games.jax_asterix",
    "asteroids": "jaxtari.games.jax_asteroids",
    "atlantis": "jaxtari.games.jax_atlantis",
    "bankheist": "jaxtari.games.jax_bankheist",
    "beamrider": "jaxtari.games.jax_beamrider",
    "berzerk": "jaxtari.games.jax_berzerk",
    "blackjack": "jaxtari.games.jax_blackjack",
    "boxing": "jaxtari.games.jax_boxing",
    "breakout": "jaxtari.games.jax_breakout",
    "casinoblackjack": "jaxtari.games.jax_casino_blackjack",
    "casinofivestudpoker": "jaxtari.games.jax_casino_five_stud_poker",
    "casinopokersolitaire": "jaxtari.games.jax_casino_poker_solitaire",
    "centipede": "jaxtari.games.jax_centipede",
    "choppercommand": "jaxtari.games.jax_choppercommand",
    "donkeykong": "jaxtari.games.jax_donkeykong",
    "enduro": "jaxtari.games.jax_enduro",
    "fishingderby": "jaxtari.games.jax_fishingderby",
    "flagcapture": "jaxtari.games.jax_flagcapture",
    "freeway": "jaxtari.games.jax_freeway",
    "frostbite": "jaxtari.games.jax_frostbite",
    "galaxian": "jaxtari.games.jax_galaxian",
    "gopher": "jaxtari.games.jax_gopher",
    "gravitar": "jaxtari.games.jax_gravitar",
    "hangman": "jaxtari.games.jax_hangman",
    "hauntedhouse": "jaxtari.games.jax_hauntedhouse",
    "humancannonball": "jaxtari.games.jax_humancannonball",
    "journeyescape": "jaxtari.games.jax_journeyescape",
    "kangaroo": "jaxtari.games.jax_kangaroo",
    "kingkong": "jaxtari.games.jax_kingkong",
    "klax": "jaxtari.games.jax_klax",
    "lasergates": "jaxtari.games.jax_lasergates",
    "namethisgame": "jaxtari.games.jax_namethisgame",
    "phoenix": "jaxtari.games.jax_phoenix",
    "pong": "jaxtari.games.jax_pong",
    "qbert": "jaxtari.games.jax_qbert",
    "riverraid": "jaxtari.games.jax_riverraid",
    "roadrunner": "jaxtari.games.jax_roadrunner",
    "seaquest": "jaxtari.games.jax_seaquest",
    "sirlancelot": "jaxtari.games.jax_sirlancelot",
    "skiing": "jaxtari.games.jax_skiing",
    "slotmachine": "jaxtari.games.jax_slotmachine",
    "spaceinvaders": "jaxtari.games.jax_spaceinvaders",
    "spacewar": "jaxtari.games.jax_spacewar",
    "surround": "jaxtari.games.jax_surround",
    "tennis": "jaxtari.games.jax_tennis",
    "tetris": "jaxtari.games.jax_tetris",
    "timepilot": "jaxtari.games.jax_timepilot",
    "tron": "jaxtari.games.jax_tron",
    "turmoil": "jaxtari.games.jax_turmoil",
    "upndown": "jaxtari.games.jax_upndown",
    "venture": "jaxtari.games.jax_venture",
    "videocheckers": "jaxtari.games.jax_videocheckers",
    "videochess": "jaxtari.games.jax_videochess",
    "videocube": "jaxtari.games.jax_videocube",
    "videopinball": "jaxtari.games.jax_videopinball",
    "wordzapper": "jaxtari.games.jax_wordzapper",
    "wizardofwor": "jaxtari.games.jax_wizardofwor",
    "assault": "jaxtari.games.jax_assault",
    "backgammon": "jaxtari.games.jax_backgammon",
    "othello": "jaxtari.games.jax_othello",
    "mspacman": "jaxtari.games.jax_mspacman",
    "montezumarevenge": "jaxtari.games.jax_montezumarevenge",
    "pacman": "jaxtari.games.jax_pacman",
    "kaboom": "jaxtari.games.jax_kaboom",
    "basicmath": "jaxtari.games.jax_basicmath",
    "miniaturegolf": "jaxtari.games.jax_miniature_golf",
    "yarsrevenge": "jaxtari.games.jax_yarsrevenge",
    # Add new games here
}

# ALE / Gymnasium names that differ from the Jaxtari registry key.
GAME_ALIASES = {
    "journey_escape": "journeyescape",
    "road_runner": "roadrunner",
    "trondead": "tron",
    "up_n_down": "upndown",
    "video_chess": "videochess",
}

# Mod modules registry: for each game, provide the Controller class path
MOD_MODULES = {
    "donkeykong": "jaxtari.games.mods.donkeykong_mods.DonkeyKongEnvMod",
    "pong": "jaxtari.games.mods.pong_mods.PongEnvMod",
    "kangaroo": "jaxtari.games.mods.kangaroo_mods.KangarooEnvMod",
    "freeway": "jaxtari.games.mods.freeway_mods.FreewayEnvMod",
    "breakout": "jaxtari.games.mods.breakout_mods.BreakoutEnvMod",
    "seaquest": "jaxtari.games.mods.seaquest_mods.SeaquestEnvMod",
    "videochess": "jaxtari.games.mods.videochess_mods.VideochessEnvMod",
    "videopinball": "jaxtari.games.mods.videopinball_mods.VideoPinballEnvMod",
    "tennis": "jaxtari.games.mods.tennis_mods.TennisEnvMod",
    "upndown": "jaxtari.games.mods.upndown_mods.UpNDownEnvMod",
    "fishingderby": "jaxtari.games.mods.fishingderby_mods.FishingDerbyEnvMod",
    "atlantis": "jaxtari.games.mods.atlantis_mods.AtlantisEnvMod",
    "bankheist": "jaxtari.games.mods.bankheist_mods.BankHeistEnvMod",
    "montezumarevenge": "jaxtari.games.mods.montezuma_revenge_mods.MontezumaRevengeEnvMod",
    "frostbite": "jaxtari.games.mods.frostbite_mods.FrostbiteEnvMod",
    "gopher": "jaxtari.games.mods.gopher_mods.GopherEnvMod",
    "gravitar": "jaxtari.games.mods.gravitar_mods.GravitarEnvMod",
    "journeyescape": "jaxtari.games.mods.journey_escape_mods.JourneyEscapeEnvMod",
    "phoenix": "jaxtari.games.mods.phoenix_mods.PhoenixEnvMod",
    "enduro": "jaxtari.games.mods.enduro_mods.EnduroEnvMod",
    "qbert": "jaxtari.games.mods.qbert_mods.QbertEnvMod",
    "roadrunner": "jaxtari.games.mods.roadrunner_mods.RoadRunnerEnvMod",
    "mspacman": "jaxtari.games.mods.mspacman_mods.MsPacmanEnvMod",
    "beamrider": "jaxtari.games.mods.beamrider_mods.BeamRiderEnvMod",
    "venture": "jaxtari.games.mods.venture_mods.VentureEnvMod",
    "spaceinvaders": "jaxtari.games.mods.spaceinvaders_mods.SpaceInvadersEnvMod",
    "skiing": "jaxtari.games.mods.skiing_mods.SkiingEnvMod",
    "alien": "jaxtari.games.mods.alien_mods.AlienEnvMod",
    "asteroids": "jaxtari.games.mods.asteroids_mods.AsteroidsEnvMod",
    "choppercommand": "jaxtari.games.mods.choppercommand_mods.ChopperCommandEnvMod",
    "centipede": "jaxtari.games.mods.centipede_mods.CentipedeEnvMod",
    "pacman": "jaxtari.games.mods.pacman_mods.PacmanEnvMod",
    "othello": "jaxtari.games.mods.othello_mods.OthelloEnvMod",
    "wizardofwor": "jaxtari.games.mods.wizardofwor_mods.WizardOfWorEnvMod",
    "backgammon": "jaxtari.games.mods.backgammon_mods.BackgammonEnvMod",
    "kaboom": "jaxtari.games.mods.kaboom_mods.KaboomEnvMod",
    "basicmath": "jaxtari.games.mods.basicmath_mods.BasicmathEnvMod",
    "miniaturegolf": "jaxtari.games.mods.miniature_golf_mods.MiniatureGolfEnvMod",
    "yarsrevenge": "jaxtari.games.mods.yarsrevenge_mods.YarsRevengeEnvMod",
    "boxing": "jaxtari.games.mods.boxing_mods.BoxingEnvMod",
    "amidar": "jaxtari.games.mods.amidar_mods.AmidarEnvMod",
    "timepilot": "jaxtari.games.mods.timepilot_mods.TimePilotEnvMod",
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
    Creates and returns a Jaxtari game environment instance.
    This is the main entry point for creating environments.

    If 'mods' is provided, this function applies the
    full two-stage modding pipeline:
    1. Pre-scans for constant overrides.
    2. Instantiates the base env with modded constants.
    3. Applies the internal 'JaxtariModController'.
    4. Wraps the env with the 'JaxtariModWrapper'.

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
        game_name_clean = game_name.lower().replace("_", "").replace("-", "")
        game_name_clean = GAME_ALIASES.get(game_name_clean, game_name_clean)
        for key in GAME_MODULES:
            if key.lower().replace("_", "").replace("-", "") == game_name_clean:
                game_name = key
                break

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

        # 2. Mods need default consts for pre-scan; otherwise a single env_class() is enough.
        if mods:
            try:
                base_consts = env_class().consts
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

        env = env_class()
        _warn_deprecated_obs_to_flat_array(env)
        return env

    except (ImportError, NotImplementedError) as e:
        # Only wrap registration/import errors - let intentional errors (ValueError, etc.) propagate
        raise ImportError(f"Failed to load game '{game_name}': {e}") from e
