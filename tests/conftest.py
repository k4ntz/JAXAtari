import pytest
import sys
import os
import importlib.util
import inspect
import gc
from pathlib import Path

# Force CPU before any JAX import (jaxatari pulls JAX in transitively).
# Overrides a user/shell JAX_PLATFORMS so pytest never tries to init a missing GPU.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

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

CORE_INFRA_GAMES = ("seaquest", "kangaroo", "montezumarevenge", "pong", "phoenix")
SMOKE_STEPS = 5
INTEGRATION_STEPS = 25
STRESS_STEPS = 100

def pytest_addoption(parser):
    """Adds the --game command-line option to pytest."""
    parser.addoption(
        "--game",
        action="store",
        default=None,
        help="Run tests for specific games by name (comma-separated, e.g. 'skiing,spacewar')",
    )
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Include integration-marked tests.",
    )
    parser.addoption(
        "--slow",
        action="store_true",
        default=False,
        help="Include slow tests (implies integration).",
    )
    parser.addoption(
        "--wrapper-full",
        action="store_true",
        default=False,
        help="Include full wrapper-matrix tests.",
    )

def pytest_configure(config):
    """Registers the 'requires_snapshot' marker."""
    config.addinivalue_line("markers", "requires_snapshot: mark test as requiring a snapshot")
    config.addinivalue_line("markers", "smoke: fast checks for PR validation")
    config.addinivalue_line("markers", "integration: broader integration checks")
    config.addinivalue_line("markers", "slow: expensive checks for exhaustive validation")
    config.addinivalue_line("markers", "wrapper_full: runs only with full wrapper matrix")
    config.addinivalue_line("markers", "serial: must run without xdist parallel workers")

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
    Applies folder-specific game scoping and skips snapshot-dependent tests
    when no matching snapshot exists for the parametrized game.
    """

    games_with_snapshots = discover_games_with_snapshots()
    specified_games = parse_game_list(config.getoption("--game"))
    include_integration = bool(config.getoption("integration"))
    include_slow = bool(config.getoption("slow"))
    include_wrapper_full = bool(config.getoption("wrapper_full"))
    if include_slow:
        include_integration = True
        include_wrapper_full = True
    normalized_specified_games = set()
    if specified_games:
        normalized_specified_games = {normalize_game_name(game) for game in specified_games}

    folder_game_scopes = discover_folder_game_scopes(Path(__file__).parent)

    deselected = []
    kept = []

    for item in items:
        item_path = Path(str(item.fspath))
        required_games = get_required_games_for_test(item_path, folder_game_scopes)
        if required_games and normalized_specified_games and required_games.isdisjoint(normalized_specified_games):
            deselected.append(item)
            continue

        if item.get_closest_marker("requires_snapshot"):
            if hasattr(item, "callspec") and 'game_name' in item.callspec.params:
                game_name = item.callspec.params['game_name']
                if game_name not in games_with_snapshots:
                    item.add_marker(pytest.mark.skip(reason=f"No snapshot found for game '{game_name}'"))

        if item.get_closest_marker("slow") and not include_slow:
            deselected.append(item)
            continue
        if item.get_closest_marker("integration") and not include_integration:
            deselected.append(item)
            continue
        if item.get_closest_marker("wrapper_full") and not include_wrapper_full:
            deselected.append(item)
            continue
        kept.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = kept

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

def parse_game_list(option_value: str | None) -> list[str] | None:
    """
    Parse a comma-separated --game option into a list of normalized game names.
    Returns None if option_value is None or empty.
    """
    if not option_value:
        return None
    items = [item.strip().lower() for item in option_value.split(",")]
    items = [item for item in items if item]
    return items or None

def normalize_game_name(game_name: str) -> str:
    """Normalizes game names so aliases like montezuma_revenge still match."""
    return game_name.lower().replace("_", "").replace("-", "").strip()

def parse_folder_game_scope(config_file: Path) -> set[str]:
    """
    Parse folder game scope config (.pytest-game).
    Supports comma-separated values and line-separated values.
    """
    content = config_file.read_text(encoding="utf-8")
    normalized_games = set()
    for line in content.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        for token in line.split(","):
            normalized = normalize_game_name(token)
            if normalized:
                normalized_games.add(normalized)
    return normalized_games

def discover_folder_game_scopes(tests_dir: Path) -> dict[Path, set[str]]:
    """
    Discover all folder-level game scopes in tests via `.pytest-game` files.
    The config file applies to its containing folder and all descendants.
    """
    scope_files = tests_dir.rglob(".pytest-game")
    scopes = {}
    for scope_file in scope_files:
        folder = scope_file.parent.resolve()
        games = parse_folder_game_scope(scope_file)
        if games:
            scopes[folder] = games
    return scopes

def get_required_games_for_test(
    test_path: Path,
    folder_game_scopes: dict[Path, set[str]],
) -> set[str] | None:
    """Return nearest folder scope for a test file, if one exists."""
    resolved = test_path.resolve()
    best_match = None
    best_depth = -1
    for folder, games in folder_game_scopes.items():
        if folder == resolved or folder in resolved.parents:
            depth = len(folder.parts)
            if depth > best_depth:
                best_match = games
                best_depth = depth
    return best_match

def pytest_generate_tests(metafunc):
    """
    Dynamically parametrizes any test that uses the 'game_name' fixture.
    If --game is passed on the command line, it runs only that game.
    Otherwise, it discovers and runs all games.
    For regression tests, it only runs games that have snapshots.
    """
    
    is_regression_test = metafunc.cls is not None and "TestRegression" in metafunc.cls.__name__

    if 'game_name' in metafunc.fixturenames:
        specified_games = parse_game_list(metafunc.config.getoption("--game"))

        test_path = Path(str(getattr(metafunc.definition, "path", metafunc.definition.fspath)))
        folder_game_scopes = discover_folder_game_scopes(Path(__file__).parent)
        required_games = get_required_games_for_test(test_path, folder_game_scopes)

        if is_regression_test:
            game_list = discover_games_with_snapshots()
        else:
            game_list = discover_games()

        if specified_games:
            game_list = [game for game in game_list if normalize_game_name(game) in {normalize_game_name(g) for g in specified_games}]

        if required_games:
            game_list = [game for game in game_list if normalize_game_name(game) in required_games]

        metafunc.parametrize("game_name", game_list)
    
    if "representative_game_name" in metafunc.fixturenames:
        specified_games = parse_game_list(metafunc.config.getoption("--game"))
        selected_core_games = list(CORE_INFRA_GAMES)
        if specified_games:
            normalized_requested = {normalize_game_name(g) for g in specified_games}
            selected_core_games = [
                game for game in selected_core_games
                if normalize_game_name(game) in normalized_requested
            ]
        if selected_core_games:
            metafunc.parametrize("representative_game_name", selected_core_games)

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

@pytest.fixture
def representative_game_name(request):
    """Parameterized representative game fixture for heavy tests."""
    return request.param

@pytest.fixture
def raw_env_representative(representative_game_name):
    """Raw env fixture restricted to representative games only."""
    return load_game_environment(representative_game_name)

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

SMOKE_WRAPPER_RECIPE_NAMES = (
    "Pixel",
    "LoggedFlattenedPixelAndObject",
)

def _build_wrapped_env(game_name, wrapper_recipe):
    fresh_raw_env = load_game_environment(game_name)
    return wrapper_recipe(fresh_raw_env)

@pytest.fixture(params=[WRAPPER_RECIPES[name] for name in SMOKE_WRAPPER_RECIPE_NAMES], ids=SMOKE_WRAPPER_RECIPE_NAMES)
def wrapped_env_smoke(game_name, request):
    """Wrapper fixture for fast PR lanes."""
    return _build_wrapped_env(game_name, request.param)

@pytest.fixture(params=WRAPPER_RECIPES.values(), ids=WRAPPER_RECIPES.keys())
def wrapped_env_full(game_name, request):
    """Wrapper fixture for exhaustive lanes."""
    return _build_wrapped_env(game_name, request.param)

@pytest.fixture(params=WRAPPER_RECIPES.values(), ids=WRAPPER_RECIPES.keys())
def wrapped_env_full_representative(representative_game_name, request):
    """Full wrapper fixture restricted to representative games."""
    return _build_wrapped_env(representative_game_name, request.param)

@pytest.fixture(params=[WRAPPER_RECIPES[name] for name in SMOKE_WRAPPER_RECIPE_NAMES], ids=SMOKE_WRAPPER_RECIPE_NAMES)
def wrapped_env(game_name, request):
    """Default wrapper fixture used in PR lanes."""
    return _build_wrapped_env(game_name, request.param)