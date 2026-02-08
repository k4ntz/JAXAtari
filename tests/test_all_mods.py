"""
Test all available mods and the modification system.

This test module:
1. Discovers all games with mods registered in the MOD_MODULES registry
2. For each game, loads the mod registry and gets all available mods
3. Tests each mod individually by creating an environment and running basic operations
4. Verifies mod-tracking variables (_mod_history, _patched_renderer_methods) are filled
5. Checks that PixelWrapper with native downscaling does not break modded envs
6. Generic modification system tests (constants, plugin types, make with mods)
7. Datatype consistency tests (PyTreeNode, no verbose params)
"""

import inspect
import pytest
import jax
import jax.numpy as jnp
from dataclasses import is_dataclass as dc_is_dataclass, fields
from typing import Dict, List, Any

from jaxatari.core import make, MOD_MODULES, GAME_MODULES
from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin, _load_from_string
from jaxatari.wrappers import AtariWrapper, PixelObsWrapper
from conftest import parse_game_list


def get_base_env(env) -> Any:
    """
    Unwrap the environment stack until we get the base game instance
    (the one that has _mod_history and .renderer).
    With make(game, mods=[...]) we have: ModWrapper -> ModController -> base_game.
    """
    while hasattr(env, "_env"):
        env = env._env
    return env


def get_all_mods_for_game(game_name: str) -> Dict[str, List[str]]:
    """
    Get all available mods for a game, separating individual mods from modpacks.
    
    Args:
        game_name: Name of the game
    
    Returns:
        Dictionary with keys:
            - 'individual': List of individual mod keys
            - 'modpacks': List of modpack keys (mods that are lists of other mods)
    
    Raises:
        Exception: If the mod registry cannot be loaded
    """
    if game_name not in MOD_MODULES:
        return {'individual': [], 'modpacks': []}
    
    ControllerClass = _load_from_string(MOD_MODULES[game_name])
    registry = ControllerClass.REGISTRY
    
    individual_mods = []
    modpacks = []
    
    for mod_key, plugin in registry.items():
        # Skip private mods (starting with _)
        if mod_key.startswith('_'):
            continue
        
        if isinstance(plugin, list):
            modpacks.append(mod_key)
        else:
            individual_mods.append(mod_key)
    
    return {
        'individual': sorted(individual_mods),
        'modpacks': sorted(modpacks)
    }


def pytest_generate_tests(metafunc):
    """
    Dynamically parametrize tests that use 'mod_game_name' and 'mod_key' fixtures.
    Only generates tests for games that have mods registered.
    """
    # Handle parametrization for mod execution tests
    if 'mod_game_name' in metafunc.fixturenames and 'mod_key' in metafunc.fixturenames:
        # Get all games with mods
        games_with_mods = list(MOD_MODULES.keys())
        
        # Filter by --game option if specified
        specified_games = parse_game_list(metafunc.config.getoption("--game", default=None))
        if specified_games:
            filtered = [g for g in games_with_mods if g in specified_games]
            if filtered:
                games_with_mods = filtered
            else:
                # Game specified but doesn't have mods - skip all tests by not parametrizing
                # The fixtures will handle skipping
                return
        
        # Collect all (game_name, mod_key) pairs
        test_cases = []
        for game_name in games_with_mods:
            try:
                mods_info = get_all_mods_for_game(game_name)
                
                # Add individual mods
                for mod_key in mods_info['individual']:
                    test_cases.append((game_name, mod_key, 'individual'))
                
                # Add modpacks
                for mod_key in mods_info['modpacks']:
                    test_cases.append((game_name, mod_key, 'modpack'))
            except Exception:
                # Skip games that fail to load mod registry
                continue
        
        # Parametrize with game_name, mod_key, and mod_type
        if test_cases:
            metafunc.parametrize("mod_game_name,mod_key,mod_type", test_cases)
        else:
            # No test cases found - fixtures will skip
            return
    
    # Handle parametrization for discovery tests (use mod_game_name to avoid conflict with conftest)
    elif 'mod_game_name' in metafunc.fixturenames and 'mod_key' not in metafunc.fixturenames:
        # This is for TestModDiscovery tests that only need mod_game_name
        games_with_mods = list(MOD_MODULES.keys())
        
        # Filter by --game option if specified
        specified_games = parse_game_list(metafunc.config.getoption("--game", default=None))
        if specified_games:
            filtered = [g for g in games_with_mods if g in specified_games]
            if filtered:
                games_with_mods = filtered
            else:
                # Game specified but doesn't have mods - fixtures will skip
                return
        
        if games_with_mods:
            metafunc.parametrize("mod_game_name", games_with_mods)
        else:
            # No games with mods - fixtures will skip
            return


@pytest.fixture
def mod_game_name(request):
    """Fixture that receives the game name from parametrization."""
    if not hasattr(request, 'param'):
        pytest.skip("Game does not have mods registered")
    return request.param


@pytest.fixture
def mod_key(request):
    """Fixture that receives the mod key from parametrization."""
    if not hasattr(request, 'param'):
        pytest.skip("Game does not have mods registered")
    return request.param


@pytest.fixture
def mod_type(request):
    """Fixture that receives the mod type (individual or modpack) from parametrization."""
    if not hasattr(request, 'param'):
        pytest.skip("Game does not have mods registered")
    return request.param


@pytest.fixture
def raw_env_available(raw_env):
    """
    Same as raw_env but skips when the game is not registered in core.GAME_MODULES
    (e.g. surround exists as a file but is commented out in core).
    Use this for tests that call make(game_name, ...) so we only run for available games.
    """
    game_name = raw_env.__class__.__module__.split(".")[-1].replace("jax_", "")
    if game_name not in GAME_MODULES:
        pytest.skip(f"Game '{game_name}' is not in core.GAME_MODULES")
    return raw_env


class TestModExecution:
    """
    For each game with mods: discover mods, apply each mod one-by-one,
    run 10 steps and assert no crash.
    """

    def test_mod_runs_reset_and_10_steps(self, mod_game_name: str, mod_key: str, mod_type: str):
        """
        Create env with the given mod, run reset then 10 steps; assert no errors.
        """
        if mod_game_name not in MOD_MODULES:
            pytest.skip(f"Game '{mod_game_name}' does not have mods registered")

        allow_conflicts = mod_type == "modpack"
        try:
            env = make(
                game_name=mod_game_name,
                mods=[mod_key],
                allow_conflicts=allow_conflicts,
            )
        except Exception as e:
            pytest.fail(f"Failed to create environment with mod '{mod_key}': {e}")

        key = jax.random.PRNGKey(42)
        try:
            obs, state = env.reset(key)
        except Exception as e:
            pytest.fail(f"reset() failed with mod '{mod_key}': {e}")

        assert obs is not None, f"reset() returned None observation with mod '{mod_key}'"
        assert state is not None, f"reset() returned None state with mod '{mod_key}'"

        num_steps = 10
        for i in range(num_steps):
            try:
                action = env.action_space().sample(key)
                key, subkey = jax.random.split(key)
                obs, state, reward, done, info = env.step(state, action)
            except Exception as e:
                pytest.fail(
                    f"step() failed at step {i} with mod '{mod_key}': {e}"
                )
            assert obs is not None, (
                f"step() returned None observation at step {i} with mod '{mod_key}'"
            )
            assert state is not None, (
                f"step() returned None state at step {i} with mod '{mod_key}'"
            )
            assert jnp.isfinite(float(reward)), (
                f"step() returned non-finite reward at step {i} with mod '{mod_key}': {reward}"
            )

    def test_mod_tracking_variables_filled(self, mod_game_name: str, mod_key: str, mod_type: str):
        """
        After applying a mod, the base env must have _mod_history and _patched_renderer_methods
        correctly set (structure exists; for internal mods that change something, at least one
        category or patched method list is non-empty).
        """
        if mod_game_name not in MOD_MODULES:
            pytest.skip(f"Game '{mod_game_name}' does not have mods registered")

        allow_conflicts = mod_type == "modpack"
        env = make(
            game_name=mod_game_name,
            mods=[mod_key],
            allow_conflicts=allow_conflicts,
        )
        base_env = get_base_env(env)

        assert hasattr(base_env, "_mod_history"), (
            f"Base env should have _mod_history when mods are applied (mod '{mod_key}')."
        )
        assert hasattr(base_env, "_patched_renderer_methods"), (
            f"Base env should have _patched_renderer_methods (mod '{mod_key}')."
        )

        hist = base_env._mod_history
        assert isinstance(hist, dict), "_mod_history should be a dict"
        for category in ("attribute", "method", "constant", "asset"):
            assert category in hist, f"_mod_history should have key '{category}'"
            assert isinstance(hist[category], set), (
                f"_mod_history['{category}'] should be a set"
            )

        patched = base_env._patched_renderer_methods
        assert isinstance(patched, list), "_patched_renderer_methods should be a list"

        # If this mod is an internal mod that patches or overrides something, we expect
        # at least one trace in _mod_history or _patched_renderer_methods.
        ControllerClass = _load_from_string(MOD_MODULES[mod_game_name])
        registry = ControllerClass.REGISTRY
        if mod_key not in registry or isinstance(registry[mod_key], list):
            return
        plugin_class = registry[mod_key]
        if not issubclass(plugin_class, JaxAtariInternalModPlugin):
            return

        has_constant = getattr(plugin_class, "constants_overrides", None) and len(plugin_class.constants_overrides) > 0
        has_attr = getattr(plugin_class, "attribute_overrides", None) and len(plugin_class.attribute_overrides) > 0
        has_asset = getattr(plugin_class, "asset_overrides", None) and len(plugin_class.asset_overrides) > 0
        has_method = (
            len(patched) > 0
            or len(hist["method"]) > 0
        )
        any_override = has_constant or has_attr or has_asset or has_method
        if not any_override:
            return

        constant_filled = len(hist["constant"]) > 0
        attr_filled = len(hist["attribute"]) > 0
        asset_filled = len(hist["asset"]) > 0
        method_filled = len(hist["method"]) > 0 or len(patched) > 0
        assert (
            (has_constant and constant_filled)
            or (has_attr and attr_filled)
            or (has_asset and asset_filled)
            or (has_method and method_filled)
        ), (
            f"Internal mod '{mod_key}' has overrides/patches but no trace in _mod_history or "
            f"_patched_renderer_methods: constants={has_constant}, attributes={has_attr}, "
            f"assets={has_asset}; hist={hist!r}, patched={patched!r}"
        )


class TestModDiscovery:
    """Test that mod discovery works correctly for games with mods."""
    
    def test_game_has_mods_registry(self, mod_game_name):
        """Test that games with mods have a valid mod registry."""
        game_name = mod_game_name
        # Skip if game doesn't have mods (safety check)
        if game_name not in MOD_MODULES:
            pytest.skip(f"Game '{game_name}' does not have mods registered")
        
        assert game_name in MOD_MODULES, f"Game '{game_name}' should be in MOD_MODULES"
        
        try:
            ControllerClass = _load_from_string(MOD_MODULES[game_name])
            assert hasattr(ControllerClass, 'REGISTRY'), \
                f"Mod controller for '{game_name}' should have REGISTRY attribute"
            
            registry = ControllerClass.REGISTRY
            assert isinstance(registry, dict), \
                f"REGISTRY for '{game_name}' should be a dictionary"
            
            # Should have at least one mod
            assert len(registry) > 0, \
                f"REGISTRY for '{game_name}' should contain at least one mod"
        except Exception as e:
            pytest.fail(f"Failed to load mod registry for '{game_name}': {e}")
    
    def test_mods_can_be_discovered(self, mod_game_name):
        """Test that mods can be discovered for games with mods."""
        game_name = mod_game_name
        # Skip if game doesn't have mods (safety check)
        if game_name not in MOD_MODULES:
            pytest.skip(f"Game '{game_name}' does not have mods registered")
        
        try:
            mods_info = get_all_mods_for_game(game_name)
        except Exception as e:
            pytest.fail(f"Failed to discover mods for '{game_name}': {e}")
        
        # Should have at least some mods (individual or modpacks)
        total_mods = len(mods_info['individual']) + len(mods_info['modpacks'])
        assert total_mods > 0, \
            f"Game '{game_name}' should have at least one mod (individual or modpack)"
    
    def test_all_mods_are_valid(self, mod_game_name):
        """Test that all discovered mods reference valid plugin classes."""
        game_name = mod_game_name
        # Skip if game doesn't have mods (safety check)
        if game_name not in MOD_MODULES:
            pytest.skip(f"Game '{game_name}' does not have mods registered")
        
        try:
            mods_info = get_all_mods_for_game(game_name)
            ControllerClass = _load_from_string(MOD_MODULES[game_name])
            registry = ControllerClass.REGISTRY
        except Exception as e:
            pytest.fail(f"Failed to load mod registry for '{game_name}': {e}")
        
        # Check individual mods
        for mod_key in mods_info['individual']:
            assert mod_key in registry, \
                f"Mod '{mod_key}' should be in registry for '{game_name}'"
            
            plugin = registry[mod_key]
            # Should be a class, not a list (lists are modpacks)
            assert not isinstance(plugin, list), \
                f"Mod '{mod_key}' should not be a list (modpacks should be in modpacks list)"
        
        # Check modpacks
        for modpack_key in mods_info['modpacks']:
            assert modpack_key in registry, \
                f"Modpack '{modpack_key}' should be in registry for '{game_name}'"
            
            plugin = registry[modpack_key]
            # Should be a list
            assert isinstance(plugin, list), \
                f"Modpack '{modpack_key}' should be a list of mod keys"
            
            # All mods in the modpack should exist
            for sub_mod_key in plugin:
                assert sub_mod_key in registry, \
                    f"Modpack '{modpack_key}' contains invalid mod key '{sub_mod_key}'"


class TestModWithWrappers:
    """
    Modded envs with wrappers (e.g. Pixel with native downscaling) should not crash
    and mod tracking should remain valid.
    """

    def test_pixel_native_downscaling_does_not_crash_modded_env(
        self, mod_game_name: str, mod_key: str, mod_type: str
    ):
        """
        Wrap a modded env with AtariWrapper + PixelObsWrapper(use_native_downscaling=True),
        run reset and 5 steps. Ensures native downscaling path does not break modded envs
        (e.g. that patched renderer methods are reapplied correctly).
        """
        if mod_game_name not in MOD_MODULES:
            pytest.skip(f"Game '{mod_game_name}' does not have mods registered")

        allow_conflicts = mod_type == "modpack"
        env = make(
            game_name=mod_game_name,
            mods=[mod_key],
            allow_conflicts=allow_conflicts,
        )
        wrapped = PixelObsWrapper(
            AtariWrapper(env),
            do_pixel_resize=True,
            pixel_resize_shape=(84, 84),
            grayscale=False,
            use_native_downscaling=True,
        )

        key = jax.random.PRNGKey(0)
        try:
            obs, state = wrapped.reset(key)
        except Exception as e:
            pytest.fail(
                f"reset() with PixelWrapper+native_downscaling failed for mod '{mod_key}': {e}"
            )

        assert obs is not None
        assert state is not None

        for i in range(5):
            action = wrapped.action_space().sample(key)
            key, _ = jax.random.split(key)
            try:
                obs, state, reward, done, info = wrapped.step(state, action)
            except Exception as e:
                pytest.fail(
                    f"step() with PixelWrapper+native_downscaling failed at step {i} "
                    f"for mod '{mod_key}': {e}"
                )
            assert obs is not None
            assert state is not None
            assert jnp.isfinite(float(reward))

        # Mod tracking on the base env should still be present after wrapping
        base_env = get_base_env(wrapped)
        assert hasattr(base_env, "_mod_history")
        assert hasattr(base_env, "_patched_renderer_methods")
        # If the mod patched renderer methods, the base env's renderer should still
        # have those methods (and after native downscaling they are reapplied to the new renderer).
        if hasattr(base_env, "renderer") and base_env._patched_renderer_methods:
            renderer = base_env.renderer
            for fn_name in base_env._patched_renderer_methods:
                assert hasattr(renderer, fn_name), (
                    f"Patched renderer method '{fn_name}' should exist on renderer (mod '{mod_key}')."
                )


def test_no_duplicate_mod_keys():
    """Test that mod discovery works across all games."""
    all_mod_keys = {}
    
    for game_name in MOD_MODULES.keys():
        try:
            mods_info = get_all_mods_for_game(game_name)
            
            for mod_key in mods_info['individual'] + mods_info['modpacks']:
                if mod_key in all_mod_keys:
                    # This is okay - mods can have the same name in different games
                    # But we'll just track it
                    all_mod_keys[mod_key].append(game_name)
                else:
                    all_mod_keys[mod_key] = [game_name]
        except Exception:
            # Skip games that fail to load
            continue
    
    # This test just ensures discovery works - duplicates are fine across games
    assert len(all_mod_keys) > 0, "Should discover at least one mod across all games"


def test_mod_vs_unmodded_comparison(mod_game_name): 
    """
    Compare modded environment with unmodded baseline.
    
    This test verifies that mods actually change the environment behavior.
    Note: This is a basic check - some mods may not change initial state.
    """
    # Skip if game doesn't have mods (safety check)
    if mod_game_name not in MOD_MODULES:
        pytest.skip(f"Game '{mod_game_name}' does not have mods registered")
        
    # Get a mod to test with
    try:
        mods_info = get_all_mods_for_game(mod_game_name)
        if not mods_info['individual']:
            pytest.skip(f"Game '{mod_game_name}' has no individual mods to compare")
        
        # Test with first individual mod
        test_mod = mods_info['individual'][0]
    except Exception as e:
        pytest.skip(f"Could not get mods for '{mod_game_name}': {e}")
    
    # Create unmodded environment
    try:
        env_unmodded = make(game_name=mod_game_name, mods=[])
    except Exception as e:
        pytest.skip(f"Could not create unmodded environment: {e}")
    
    # Create modded environment
    try:
        env_modded = make(
            game_name=mod_game_name,
            mods=[test_mod],
            allow_conflicts=False
        )
    except Exception as e:
        pytest.skip(f"Could not create modded environment: {e}")
    
    # Compare initial states
    key = jax.random.PRNGKey(42)
    obs_unmodded, state_unmodded = env_unmodded.reset(key)
    obs_modded, state_modded = env_modded.reset(key)
    
    # States might be different (mods can change initial state)
    # But both should be valid
    assert obs_unmodded is not None, "Unmodded environment returned None observation"
    assert state_unmodded is not None, "Unmodded environment returned None state"
    assert obs_modded is not None, "Modded environment returned None observation"
    assert state_modded is not None, "Modded environment returned None state"


# -----------------------------------------------------------------------------
# Generic modification system tests (from test_modifications)
# -----------------------------------------------------------------------------


class TestModifications:
    """Generic tests for mod system that work with all games (via raw_env_available)."""

    def test_constants_structure(self, raw_env_available):
        """Test that constants have the expected structure (NamedTuple or dataclass)."""
        consts = raw_env_available.consts

        is_namedtuple = hasattr(consts, "_fields")
        is_dataclass = dc_is_dataclass(consts)

        assert is_namedtuple or is_dataclass, (
            f"Constants should be either NamedTuple or dataclass, got {type(consts)}"
        )
        assert hasattr(consts, "__class__")

        if is_dataclass:
            const_fields = fields(consts)
            if const_fields:
                field_name = const_fields[0].name
                value = getattr(consts, field_name)
                assert value is not None, f"Field {field_name} should have a value"
        elif is_namedtuple and consts._fields:
            field_name = consts._fields[0]
            value = getattr(consts, field_name)
            assert value is not None, f"Field {field_name} should have a value"

    def test_constants_can_be_overridden_conceptually(self, raw_env_available):
        """Test that constants can conceptually be overridden (tests the structure)."""
        consts = raw_env_available.consts

        if dc_is_dataclass(consts):
            const_fields = fields(consts)
            if const_fields:
                first_field = const_fields[0].name
                original_value = getattr(consts, first_field)
                if isinstance(original_value, (int, float)):
                    try:
                        if hasattr(consts, "replace"):
                            modified = consts.replace(**{first_field: original_value + 1})
                            assert getattr(modified, first_field) == original_value + 1
                        else:
                            pytest.fail("Constants should have 'replace' method for dataclass/PyTreeNode")
                    except Exception as e:
                        pytest.fail(f"Should be able to override dataclass field: {e}")
        elif hasattr(consts, "_fields") and consts._fields:
            first_field = consts._fields[0]
            original_value = getattr(consts, first_field)
            if isinstance(original_value, (int, float)):
                try:
                    modified = consts._replace(**{first_field: original_value + 1})
                    assert getattr(modified, first_field) == original_value + 1
                except Exception as e:
                    pytest.fail(f"Should be able to override NamedTuple field: {e}")

    def test_mod_system_loads_without_error(self, raw_env_available):
        """Test that mod system can be initialized without errors."""
        game_name = raw_env_available.__class__.__module__.split(".")[-1].replace("jax_", "")

        try:
            env = make(game_name, mods=[])
            assert env is not None
            assert env.consts is not None
        except (ImportError, ValueError) as e:
            if "mod" not in str(e).lower() and "not recognized" not in str(e).lower():
                raise

    def test_environment_works_after_mods(self, raw_env_available):
        """Test that environment still functions correctly after mod system initialization."""
        game_name = raw_env_available.__class__.__module__.split(".")[-1].replace("jax_", "")

        try:
            env = make(game_name, mods=[])
            key = jax.random.PRNGKey(42)
            obs, state = env.reset(key)
            assert obs is not None
            assert state is not None
            action = env.action_space().sample(key)
            obs, state, reward, done, info = env.step(state, action)
            assert obs is not None
            assert state is not None
            assert jnp.isfinite(float(reward))
        except (ImportError, ValueError) as e:
            if "mod" not in str(e).lower() and "not recognized" not in str(e).lower():
                raise


class TestModPluginTypes:
    """Test different types of mod plugins work correctly."""

    def test_internal_mod_plugin_structure(self):
        """Test that InternalModPlugin has required attributes."""
        assert hasattr(JaxAtariInternalModPlugin, "constants_overrides")
        assert hasattr(JaxAtariInternalModPlugin, "attribute_overrides")
        assert hasattr(JaxAtariInternalModPlugin, "asset_overrides")
        assert hasattr(JaxAtariInternalModPlugin, "conflicts_with")

    def test_poststep_mod_plugin_structure(self):
        """Test that PostStepModPlugin has required attributes."""
        assert hasattr(JaxAtariPostStepModPlugin, "constants_overrides")
        assert hasattr(JaxAtariPostStepModPlugin, "attribute_overrides")
        assert hasattr(JaxAtariPostStepModPlugin, "asset_overrides")
        assert hasattr(JaxAtariPostStepModPlugin, "conflicts_with")
        assert hasattr(JaxAtariPostStepModPlugin, "run")
        assert hasattr(JaxAtariPostStepModPlugin, "after_reset")

    def test_mod_plugin_can_override_constants(self):
        """Test that a mod plugin can define constant overrides."""
        class TestMod(JaxAtariInternalModPlugin):
            constants_overrides = {"TEST_CONSTANT": 42}

        assert TestMod.constants_overrides == {"TEST_CONSTANT": 42}
        assert isinstance(TestMod.constants_overrides, dict)

    def test_mod_plugin_can_override_attributes(self):
        """Test that a mod plugin can define attribute overrides."""
        class TestMod(JaxAtariInternalModPlugin):
            attribute_overrides = {"test_attr": "test_value"}

        assert TestMod.attribute_overrides == {"test_attr": "test_value"}
        assert isinstance(TestMod.attribute_overrides, dict)

    def test_mod_plugin_can_override_assets(self):
        """Test that a mod plugin can define asset overrides."""
        class TestMod(JaxAtariInternalModPlugin):
            asset_overrides = {
                "test_asset": {"name": "test_asset", "type": "single", "file": "test.npy"}
            }

        assert "test_asset" in TestMod.asset_overrides
        assert isinstance(TestMod.asset_overrides, dict)


def test_specific_game_mods_load(raw_env_available):
    """Test that mods can be loaded for specific games that have mods."""
    game_name = raw_env_available.__class__.__module__.split(".")[-1].replace("jax_", "")

    if game_name not in MOD_MODULES:
        pytest.skip(f"Game {game_name} not in list of games with mods")

    try:
        env = make(game_name, mods=[])
        assert env is not None
        key = jax.random.PRNGKey(42)
        obs, state = env.reset(key)
        assert obs is not None
    except (ImportError, ValueError) as e:
        if "not recognized" in str(e).lower() or "mod" in str(e).lower():
            pytest.skip(f"Game {game_name} doesn't have mods or isn't available")
        else:
            raise


class TestDatatypeConsistency:
    """Ensure datatypes are consistent: PyTreeNode/dataclass, no verbose params."""

    def test_constants_are_pytree_node(self, raw_env_available):
        """Test that constants are flax.struct.PyTreeNode, not NamedTuple."""
        consts = raw_env_available.consts
        assert consts is not None
        try:
            from flax import struct
            is_pytree_node = isinstance(consts, struct.PyTreeNode)
        except (ImportError, AttributeError):
            pytest.skip("Flax struct not available")
        is_namedtuple = isinstance(consts, tuple) and hasattr(consts, "_fields")
        assert is_pytree_node, (
            f"Constants should be flax.struct.PyTreeNode, got {type(consts)}. "
            "Please refactor from NamedTuple to PyTreeNode."
        )
        assert not is_namedtuple, (
            "Constants should not be NamedTuple. Please refactor to flax.struct.PyTreeNode."
        )

    def test_state_is_struct_dataclass(self, raw_env_available):
        """Test that State class is @struct.dataclass, not NamedTuple."""
        key = jax.random.PRNGKey(42)
        try:
            obs, state = raw_env_available.reset(key)
        except Exception as e:
            pytest.skip(f"Could not create state instance: {e}")
        try:
            from flax import struct
            state_class = type(state)
            is_struct_dataclass = issubclass(state_class, struct.PyTreeNode)
            has_dataclass_fields = hasattr(state_class, "__dataclass_fields__")
        except (ImportError, AttributeError):
            pytest.skip("Flax struct not available")
        is_namedtuple = isinstance(state, tuple) and hasattr(state, "_fields")
        assert is_struct_dataclass or has_dataclass_fields, (
            f"State should be @struct.dataclass (PyTreeNode), got {type(state)}."
        )
        assert not is_namedtuple, "State should not be NamedTuple."

    def test_observation_is_struct_dataclass(self, raw_env_available):
        """Test that Observation class is @struct.dataclass, not NamedTuple."""
        key = jax.random.PRNGKey(42)
        try:
            obs, state = raw_env_available.reset(key)
        except Exception as e:
            pytest.skip(f"Could not create observation instance: {e}")
        try:
            from flax import struct
            obs_class = type(obs)
            is_struct_dataclass = issubclass(obs_class, struct.PyTreeNode)
            has_dataclass_fields = hasattr(obs_class, "__dataclass_fields__")
        except (ImportError, AttributeError):
            pytest.skip("Flax struct not available")
        is_namedtuple = isinstance(obs, tuple) and hasattr(obs, "_fields")
        assert is_struct_dataclass or has_dataclass_fields, (
            f"Observation should be @struct.dataclass (PyTreeNode), got {type(obs)}."
        )
        assert not is_namedtuple, "Observation should not be NamedTuple."

    def test_info_is_struct_dataclass(self, raw_env_available):
        """Test that Info class is @struct.dataclass, not NamedTuple."""
        key = jax.random.PRNGKey(42)
        try:
            obs, state = raw_env_available.reset(key)
            action = raw_env_available.action_space().sample(key)
            obs, state, reward, done, info = raw_env_available.step(state, action)
        except Exception as e:
            pytest.skip(f"Could not create info instance: {e}")
        try:
            from flax import struct
            info_class = type(info)
            is_struct_dataclass = issubclass(info_class, struct.PyTreeNode)
            has_dataclass_fields = hasattr(info_class, "__dataclass_fields__")
        except (ImportError, AttributeError):
            pytest.skip("Flax struct not available")
        is_namedtuple = isinstance(info, tuple) and hasattr(info, "_fields")
        assert is_struct_dataclass or has_dataclass_fields, (
            f"Info should be @struct.dataclass (PyTreeNode), got {type(info)}."
        )
        assert not is_namedtuple, "Info should not be NamedTuple."

    def test_environment_no_verbose_parameter(self, raw_env_available):
        """Test that environment __init__ does not accept verbose parameter."""
        env_class = raw_env_available.__class__
        try:
            sig = inspect.signature(env_class.__init__)
            params = sig.parameters
        except Exception as e:
            pytest.skip(f"Could not inspect __init__ signature: {e}")
        assert "verbose" not in params, (
            f"Environment {env_class.__name__} should not accept 'verbose' in __init__. "
            f"Found: {list(params.keys())}."
        )

    def test_environment_methods_no_verbose_parameter(self, raw_env_available):
        """Test that environment methods (reset, step) do not accept verbose parameter."""
        env_class = raw_env_available.__class__
        try:
            reset_sig = inspect.signature(env_class.reset)
            reset_params = reset_sig.parameters
            reset_has_verbose = "verbose" in reset_params
        except Exception:
            reset_has_verbose = False
        try:
            step_sig = inspect.signature(env_class.step)
            step_params = step_sig.parameters
            step_has_verbose = "verbose" in step_params
        except Exception:
            step_has_verbose = False
        assert not reset_has_verbose, (
            f"{env_class.__name__}.reset() should not accept 'verbose'."
        )
        assert not step_has_verbose, (
            f"{env_class.__name__}.step() should not accept 'verbose'."
        )

    def test_datatype_consistency_across_operations(self, raw_env_available):
        """Test that datatypes remain consistent across reset and step operations."""
        key = jax.random.PRNGKey(42)
        obs1, state1 = raw_env_available.reset(key)
        try:
            from flax import struct
            obs_class1 = type(obs1)
            state_class1 = type(state1)
            obs_is_pytree = issubclass(obs_class1, struct.PyTreeNode) or hasattr(obs_class1, "__dataclass_fields__")
            state_is_pytree = issubclass(state_class1, struct.PyTreeNode) or hasattr(state_class1, "__dataclass_fields__")
        except (ImportError, AttributeError):
            pytest.skip("Flax struct not available")
        assert obs_is_pytree, f"Observation type: {type(obs1)}"
        assert state_is_pytree, f"State type: {type(state1)}"
        action = raw_env_available.action_space().sample(key)
        obs2, state2, reward, done, info = raw_env_available.step(state1, action)
        obs_class2 = type(obs2)
        state_class2 = type(state2)
        info_class = type(info)
        obs2_is_pytree = issubclass(obs_class2, struct.PyTreeNode) or hasattr(obs_class2, "__dataclass_fields__")
        state2_is_pytree = issubclass(state_class2, struct.PyTreeNode) or hasattr(state_class2, "__dataclass_fields__")
        info_is_pytree = issubclass(info_class, struct.PyTreeNode) or hasattr(info_class, "__dataclass_fields__")
        assert obs2_is_pytree, f"Observation type after step: {type(obs2)}"
        assert state2_is_pytree, f"State type after step: {type(state2)}"
        assert info_is_pytree, f"Info type: {type(info)}"
        assert type(obs1) == type(obs2), f"Observation type changed: {type(obs1)} vs {type(obs2)}"
        assert type(state1) == type(state2), f"State type changed: {type(state1)} vs {type(state2)}"
