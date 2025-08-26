import pytest
import sys
import importlib.util
import inspect
from pathlib import Path
from jaxatari.environment import JaxEnvironment
from jaxatari.wrappers import (
    AtariWrapper,
    PixelObsWrapper,
    ObjectCentricWrapper,
    PixelAndObjectCentricWrapper,
    FlattenObservationWrapper,
    NormalizeObservationWrapper,
    LogWrapper,
    MultiRewardLogWrapper,
)

def pytest_addoption(parser):
    """Adds the --game command-line option to pytest."""
    parser.addoption(
        "--game", action="store", default=None, help="Run tests for a specific game by name"
    )

# ==============================================================================
# 1. DYNAMIC ENVIRONMENT LOADER AND DISCOVERY (SHARED UTILITIES)
# ==============================================================================

def discover_games() -> list[str]:
    """Scans the games directory and returns a list of all available game names."""
    try:
        games_dir = Path(__file__).parent.parent / "src" / "jaxatari" / "games"
        game_files = games_dir.glob("jax_*.py")
        # Extracts 'breakout' from a path like '.../src/jaxatari/games/jax_breakout.py'
        game_names = [p.stem.replace('jax_', '') for p in game_files if p.name!= 'jax___init__.py']
        if not game_names:
            raise FileNotFoundError("No game files found in the games directory.")
        return game_names
    except FileNotFoundError as e:
        print(f"Could not discover games: {e}", file=sys.stderr)
        return

def pytest_generate_tests(metafunc):
    """
    Dynamically parametrizes any test that uses the 'game_name' fixture.
    If --game is passed on the command line, it runs only that game.
    Otherwise, it discovers and runs all games.
    """
    if 'game_name' in metafunc.fixturenames:
        specified_game = metafunc.config.getoption("--game")
        if specified_game:
            # If a specific game is provided via the command line, use only that one.
            metafunc.parametrize("game_name", [specified_game])
        else:
            # Otherwise, discover all games and run tests for each one.
            all_games = discover_games()
            metafunc.parametrize("game_name", all_games)

def load_game_environment(game_name: str) -> JaxEnvironment:
    """Dynamically loads a game environment from a.py file."""
    test_file_dir = Path(__file__).parent.resolve()
    project_root = test_file_dir.parent
    game_file_path = project_root / "src" / "jaxatari" / "games" / f"jax_{game_name.lower()}.py"

    if not game_file_path.is_file():
        raise FileNotFoundError(f"Game file not found: {game_file_path}")

    module_name = game_file_path.stem
    spec = importlib.util.spec_from_file_location(module_name, game_file_path)
    if spec is None:
        raise ImportError(f"Could not load spec for module {module_name} from {game_file_path}")

    game_module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(game_file_path.parent))
    try:
        spec.loader.exec_module(game_module)
    finally:
        sys.path.pop(0)

    for name, obj in inspect.getmembers(game_module):
        if inspect.isclass(obj) and issubclass(obj, JaxEnvironment) and obj is not JaxEnvironment:
            return obj()

    raise ImportError(f"No class inheriting from JaxEnvironment found in {game_file_path}")


# ==============================================================================
# 3. PYTEST SHARED FIXTURES
# ==============================================================================

# This fixture is now just a placeholder; its values are injected by pytest_generate_tests.
@pytest.fixture
def game_name(request):
    """A fixture that receives the game name from the dynamic parametrization."""
    return request.param

@pytest.fixture
def raw_env(game_name):
    """Provides a single, raw, unwrapped instance of the specified game environment."""
    return load_game_environment(game_name)

# Define the wrapper combinations to test against.
WRAPPER_RECIPES = {
    "Atari": lambda env: AtariWrapper(env),
    "Pixel": lambda env: PixelObsWrapper(AtariWrapper(env)),
    "ObjectCentric": lambda env: ObjectCentricWrapper(AtariWrapper(env)),
    "PixelAndObjectCentric": lambda env: PixelAndObjectCentricWrapper(AtariWrapper(env)),
    "FlattenedObjectCentric": lambda env: FlattenObservationWrapper(ObjectCentricWrapper(AtariWrapper(env))),
    "NormalizedPixel": lambda env: NormalizeObservationWrapper(PixelObsWrapper(AtariWrapper(env))),
    "LoggedFlattenedPixelAndObject": lambda env: LogWrapper(
        FlattenObservationWrapper(PixelAndObjectCentricWrapper(AtariWrapper(env)))
    ),
    "MultiRewardLogged": lambda env: MultiRewardLogWrapper(
        PixelAndObjectCentricWrapper(AtariWrapper(env))
    ),
}

@pytest.fixture(params=WRAPPER_RECIPES.values(), ids=WRAPPER_RECIPES.keys())
def wrapped_env(game_name, request):
    """
    Parameterized fixture. For each wrapper recipe, it creates a fresh instance
    of the raw environment and then wraps it.
    """
    wrapper_recipe = request.param
    fresh_raw_env = load_game_environment(game_name)
    return wrapper_recipe(fresh_raw_env)