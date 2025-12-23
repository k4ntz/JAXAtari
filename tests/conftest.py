import pytest
import sys
import importlib.util
import inspect
import gc
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

def pytest_configure(config):
    """Registers the 'requires_snapshot' marker."""
    config.addinivalue_line("markers", "requires_snapshot: mark test as requiring a snapshot")

def discover_games_with_snapshots() -> list[str]:
    """
    Scans the snapshot file for `test_regression.py` and returns a list of game names
    that have at least one existing snapshot.
    """
    snapshot_file = Path(__file__).parent / "__snapshots__" / "test_regression.ambr"
    if not snapshot_file.is_file():
        return []

    with open(snapshot_file, "rb") as f:
        content = f.read().decode("utf-8", errors="ignore")
        
    all_games = discover_games()
    games_with_snapshots = []
    for game in all_games:
        # Heuristic: snapshot names are like test_name[wrapper-game], e.g. test_trajectory_snapshot[Atari-breakout]
        if f"-{game}]" in content:
            games_with_snapshots.append(game)
            
    return list(set(games_with_snapshots))

def pytest_collection_modifyitems(config, items):
    """
    Skips tests marked with `requires_snapshot` if no snapshot is found for the game.
    """

    games_with_snapshots = discover_games_with_snapshots()
    
    for item in items:
        if item.get_closest_marker("requires_snapshot"):
            if hasattr(item, "callspec") and 'game_name' in item.callspec.params:
                game_name = item.callspec.params['game_name']
                if game_name not in games_with_snapshots:
                    item.add_marker(pytest.mark.skip(reason=f"No snapshot found for game '{game_name}'"))

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
    For regression tests, it only runs games that have snapshots.
    """
    
    is_regression_test = metafunc.cls is not None and "TestRegression" in metafunc.cls.__name__

    if 'game_name' in metafunc.fixturenames:
        specified_game = metafunc.config.getoption("--game")
        if specified_game:
            metafunc.parametrize("game_name", [specified_game])
        else:
            if is_regression_test:
                # For regression tests, only use games that have snapshots.
                game_list = discover_games_with_snapshots()
                if not game_list:
                    # If no games with snapshots are found, we should probably skip
                    # all regression tests. We can do this by parametrizing with an
                    # empty list, but it's better to let pytest handle it.
                    # A single dummy value and a skip inside the test might be better.
                    pass
            else:
                game_list = discover_games()
            
            if is_regression_test:
                metafunc.parametrize("game_name", discover_games_with_snapshots())
            else:
                metafunc.parametrize("game_name", discover_games())

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

@pytest.fixture(autouse=True)
def force_gc_between_tests():
    yield
    gc.collect()

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