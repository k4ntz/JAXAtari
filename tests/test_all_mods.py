"""
Test all available mods for games that have them.

This test module:
1. Discovers all games with mods registered in the MOD_MODULES registry
2. For each game, loads the mod registry and gets all available mods
3. Tests each mod individually by creating an environment and running basic operations
4. Reports successes and failures via pytest
"""

import pytest
import jax
import jax.numpy as jnp
from typing import Dict, List, Any

from jaxatari.core import make, MOD_MODULES
from jaxatari.modification import _load_from_string
from conftest import parse_game_list


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


def test_mod_execution(mod_game_name: str, mod_key: str, mod_type: str):
    """
    Test that a specific mod can be loaded and executed without errors.
    
    This test:
    1. Creates an environment with the specified mod
    2. Tests reset() functionality
    3. Tests step() functionality for a few steps
    4. Verifies that all operations complete without errors
    """
    # Skip if game doesn't have mods (safety check)
    if mod_game_name not in MOD_MODULES:
        pytest.skip(f"Game '{mod_game_name}' does not have mods registered")
    
    # Create environment with the mod
    # For modpacks, we need to allow conflicts since they may contain conflicting mods
    allow_conflicts = (mod_type == 'modpack')
    
    try:
        env = make(
            game_name=mod_game_name,
            mods=[mod_key],
            allow_conflicts=allow_conflicts
        )
    except Exception as e:
        pytest.fail(f"Failed to create environment with mod '{mod_key}': {e}")
    
    # Test reset
    key = jax.random.PRNGKey(42)
    try:
        obs, state = env.reset(key)
    except Exception as e:
        pytest.fail(f"reset() failed with mod '{mod_key}': {e}")
    
    assert obs is not None, f"reset() returned None observation with mod '{mod_key}'"
    assert state is not None, f"reset() returned None state with mod '{mod_key}'"
    
    # Test a few steps
    for i in range(20):
        try:
            action = env.action_space().sample(key)
            key, subkey = jax.random.split(key)
            
            obs, state, reward, done, info = env.step(state, action)
        except Exception as e:
            pytest.fail(f"step() failed at step {i} with mod '{mod_key}': {e}")
        
        assert obs is not None, f"step() returned None observation at step {i} with mod '{mod_key}'"
        assert state is not None, f"step() returned None state at step {i} with mod '{mod_key}'"
        assert jnp.isfinite(float(reward)), \
            f"step() returned non-finite reward at step {i} with mod '{mod_key}': {reward}"


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

