"""
Generic tests for the modification system that work with all environments.
Tests constant overrides, method overrides, and other mod features.
"""
import pytest
import jax
import jax.numpy as jnp
import inspect
from functools import partial
from dataclasses import is_dataclass as dc_is_dataclass, fields, asdict
from typing import get_type_hints
from jaxatari.core import make
from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin
from jaxatari.environment import JaxEnvironment


class TestModifications:
    """Generic tests for mod system that work with all games."""

    def test_constants_structure(self, raw_env):
        """Test that constants have the expected structure (NamedTuple or dataclass)."""
        consts = raw_env.consts
        
        # Verify constants structure
        is_namedtuple = hasattr(consts, '_fields')
        is_dataclass = dc_is_dataclass(consts)
        
        assert is_namedtuple or is_dataclass, \
            f"Constants should be either NamedTuple or dataclass, got {type(consts)}"
        
        # Verify constants are accessible
        assert hasattr(consts, '__class__')
        
        # Verify we can access at least one constant
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

    def test_constants_can_be_overridden_conceptually(self, raw_env):
        """Test that constants can conceptually be overridden (tests the structure)."""
        consts = raw_env.consts
        
        # Test dataclass/PyTreeNode field override capability (preferred)
        if dc_is_dataclass(consts):
            const_fields = fields(consts)
            if const_fields:
                first_field = const_fields[0].name
                original_value = getattr(consts, first_field)
                
                if isinstance(original_value, (int, float)):
                    try:
                        # Test that replace works for PyTreeNode/dataclass
                        if hasattr(consts, 'replace'):
                            modified = consts.replace(**{first_field: original_value + 1})
                            assert getattr(modified, first_field) == original_value + 1
                        else:
                            pytest.fail(f"Constants should have 'replace' method for dataclass/PyTreeNode")
                    except Exception as e:
                        pytest.fail(f"Should be able to override dataclass field: {e}")
        
        # Test NamedTuple field override capability (legacy fallback)
        elif hasattr(consts, '_fields') and consts._fields:
            # Verify _replace works for fields
            first_field = consts._fields[0]
            original_value = getattr(consts, first_field)
            
            # Try to create a modified version (if it's a simple type)
            if isinstance(original_value, (int, float)):
                try:
                    # Test that _replace would work
                    modified = consts._replace(**{first_field: original_value + 1})
                    assert getattr(modified, first_field) == original_value + 1
                except Exception as e:
                    pytest.fail(f"Should be able to override NamedTuple field: {e}")

    def test_mod_system_loads_without_error(self, raw_env):
        """Test that mod system can be initialized without errors."""
        game_name = raw_env.__class__.__module__.split('.')[-1].replace('jax_', '')
        
        # Try to create environment with empty mod list
        try:
            env = make(game_name, mods=[])
            assert env is not None
            assert env.consts is not None
        except (ImportError, ValueError) as e:
            # Some games might not have mods registered, that's okay
            if "mod" not in str(e).lower() and "not recognized" not in str(e).lower():
                raise

    def test_environment_works_after_mods(self, raw_env):
        """Test that environment still functions correctly after mod system initialization."""
        game_name = raw_env.__class__.__module__.split('.')[-1].replace('jax_', '')
        
        try:
            # Create with empty mod list (tests mod system initialization)
            env = make(game_name, mods=[])
            
            # Test basic functionality
            key = jax.random.PRNGKey(42)
            obs, state = env.reset(key)
            assert obs is not None
            assert state is not None
            
            # Test step
            action = env.action_space().sample(key)
            obs, state, reward, done, info = env.step(state, action)
            assert obs is not None
            assert state is not None
            assert jnp.isfinite(float(reward))
            
        except (ImportError, ValueError) as e:
            # Some games might not have mods registered
            if "mod" not in str(e).lower() and "not recognized" not in str(e).lower():
                raise


class TestModPluginTypes:
    """Test different types of mod plugins work correctly."""

    def test_internal_mod_plugin_structure(self):
        """Test that InternalModPlugin has required attributes."""
        # Verify the base class structure
        assert hasattr(JaxAtariInternalModPlugin, 'constants_overrides')
        assert hasattr(JaxAtariInternalModPlugin, 'attribute_overrides')
        assert hasattr(JaxAtariInternalModPlugin, 'asset_overrides')
        assert hasattr(JaxAtariInternalModPlugin, 'conflicts_with')

    def test_poststep_mod_plugin_structure(self):
        """Test that PostStepModPlugin has required attributes."""
        # Verify the base class structure
        assert hasattr(JaxAtariPostStepModPlugin, 'constants_overrides')
        assert hasattr(JaxAtariPostStepModPlugin, 'attribute_overrides')
        assert hasattr(JaxAtariPostStepModPlugin, 'asset_overrides')
        assert hasattr(JaxAtariPostStepModPlugin, 'conflicts_with')
        assert hasattr(JaxAtariPostStepModPlugin, 'run')
        assert hasattr(JaxAtariPostStepModPlugin, 'after_reset')

    def test_mod_plugin_can_override_constants(self):
        """Test that a mod plugin can define constant overrides."""
        class TestMod(JaxAtariInternalModPlugin):
            constants_overrides = {
                "TEST_CONSTANT": 42
            }
        
        assert TestMod.constants_overrides == {"TEST_CONSTANT": 42}
        assert isinstance(TestMod.constants_overrides, dict)

    def test_mod_plugin_can_override_attributes(self):
        """Test that a mod plugin can define attribute overrides."""
        class TestMod(JaxAtariInternalModPlugin):
            attribute_overrides = {
                "test_attr": "test_value"
            }
        
        assert TestMod.attribute_overrides == {"test_attr": "test_value"}
        assert isinstance(TestMod.attribute_overrides, dict)

    def test_mod_plugin_can_override_assets(self):
        """Test that a mod plugin can define asset overrides."""
        class TestMod(JaxAtariInternalModPlugin):
            asset_overrides = {
                "test_asset": {
                    'name': 'test_asset',
                    'type': 'single',
                    'file': 'test.npy'
                }
            }
        
        assert "test_asset" in TestMod.asset_overrides
        assert isinstance(TestMod.asset_overrides, dict)


def test_specific_game_mods_load(raw_env):
    """Test that mods can be loaded for specific games that have mods."""
    game_name = raw_env.__class__.__module__.split('.')[-1].replace('jax_', '')
    
    # Only test games that have mods
    games_with_mods = ["seaquest", "pong", "freeway", "kangaroo"]
    if game_name not in games_with_mods:
        pytest.skip(f"Game {game_name} not in list of games with mods")
    
    try:
        # Try to create environment with empty mod list
        env = make(game_name, mods=[])
        assert env is not None
        
        # Verify it works
        key = jax.random.PRNGKey(42)
        obs, state = env.reset(key)
        assert obs is not None
        
    except (ImportError, ValueError) as e:
        # If game doesn't exist or mods aren't registered, skip
        if "not recognized" in str(e).lower() or "mod" in str(e).lower():
            pytest.skip(f"Game {game_name} doesn't have mods or isn't available")
        else:
            raise


class TestDatatypeConsistency:
    """Tests to ensure datatypes are consistent during refactor.
    
    These tests verify that:
    1. Constants use flax.struct.PyTreeNode (not NamedTuple)
    2. State, Observation, and Info use @struct.dataclass (not NamedTuple)
    3. Environments don't accept verbose parameters
    """
    
    def test_constants_are_pytree_node(self, raw_env):
        """Test that constants are flax.struct.PyTreeNode, not NamedTuple."""
        consts = raw_env.consts
        assert consts is not None, "Constants should exist"
        
        # Check if it's a PyTreeNode
        try:
            from flax import struct
            is_pytree_node = isinstance(consts, struct.PyTreeNode)
        except (ImportError, AttributeError):
            pytest.skip("Flax struct not available")
        
        # Check if it's a NamedTuple (legacy)
        is_namedtuple = isinstance(consts, tuple) and hasattr(consts, '_fields')
        
        assert is_pytree_node, \
            f"Constants should be flax.struct.PyTreeNode, got {type(consts)}. " \
            f"Please refactor from NamedTuple to PyTreeNode (see jax_flagcapture.py as example)."
        
        assert not is_namedtuple, \
            f"Constants should not be NamedTuple. Found NamedTuple with fields: {getattr(consts, '_fields', None)}. " \
            f"Please refactor to flax.struct.PyTreeNode (see jax_flagcapture.py as example)."
    
    def test_state_is_struct_dataclass(self, raw_env):
        """Test that State class is @struct.dataclass, not NamedTuple."""
        # Create a state instance to check its type
        key = jax.random.PRNGKey(42)
        try:
            obs, state = raw_env.reset(key)
        except Exception as e:
            pytest.skip(f"Could not create state instance: {e}")
        
        # Check if state is a struct.dataclass
        try:
            from flax import struct
            state_class = type(state)
            # Check if class is a subclass of PyTreeNode
            is_struct_dataclass = issubclass(state_class, struct.PyTreeNode)
            # Also check if it has dataclass fields (double check)
            has_dataclass_fields = hasattr(state_class, '__dataclass_fields__')
        except (ImportError, AttributeError):
            pytest.skip("Flax struct not available")
        
        # Check if state is a NamedTuple (legacy)
        is_namedtuple = isinstance(state, tuple) and hasattr(state, '_fields')
        
        assert is_struct_dataclass or has_dataclass_fields, \
            f"State should be @struct.dataclass (PyTreeNode), got {type(state)}. " \
            f"Please refactor from NamedTuple to @struct.dataclass (see jax_flagcapture.py as example)."
        
        assert not is_namedtuple, \
            f"State should not be NamedTuple. Found NamedTuple with fields: {getattr(state, '_fields', None)}. " \
            f"Please refactor to @struct.dataclass (see jax_flagcapture.py as example)."
    
    def test_observation_is_struct_dataclass(self, raw_env):
        """Test that Observation class is @struct.dataclass, not NamedTuple."""
        key = jax.random.PRNGKey(42)
        try:
            obs, state = raw_env.reset(key)
        except Exception as e:
            pytest.skip(f"Could not create observation instance: {e}")
        
        # Check if observation is a struct.dataclass
        try:
            from flax import struct
            obs_class = type(obs)
            # Check if class is a subclass of PyTreeNode
            is_struct_dataclass = issubclass(obs_class, struct.PyTreeNode)
            # Also check if it has dataclass fields (double check)
            has_dataclass_fields = hasattr(obs_class, '__dataclass_fields__')
        except (ImportError, AttributeError):
            pytest.skip("Flax struct not available")
        
        # Check if observation is a NamedTuple (legacy)
        is_namedtuple = isinstance(obs, tuple) and hasattr(obs, '_fields')
        
        assert is_struct_dataclass or has_dataclass_fields, \
            f"Observation should be @struct.dataclass (PyTreeNode), got {type(obs)}. " \
            f"Please refactor from NamedTuple to @struct.dataclass (see jax_flagcapture.py as example)."
        
        assert not is_namedtuple, \
            f"Observation should not be NamedTuple. Found NamedTuple with fields: {getattr(obs, '_fields', None)}. " \
            f"Please refactor to @struct.dataclass (see jax_flagcapture.py as example)."
    
    def test_info_is_struct_dataclass(self, raw_env):
        """Test that Info class is @struct.dataclass, not NamedTuple."""
        key = jax.random.PRNGKey(42)
        try:
            obs, state = raw_env.reset(key)
            action = raw_env.action_space().sample(key)
            obs, state, reward, done, info = raw_env.step(state, action)
        except Exception as e:
            pytest.skip(f"Could not create info instance: {e}")
        
        # Check if info is a struct.dataclass
        try:
            from flax import struct
            info_class = type(info)
            # Check if class is a subclass of PyTreeNode
            is_struct_dataclass = issubclass(info_class, struct.PyTreeNode)
            # Also check if it has dataclass fields (double check)
            has_dataclass_fields = hasattr(info_class, '__dataclass_fields__')
        except (ImportError, AttributeError):
            pytest.skip("Flax struct not available")
        
        # Check if info is a NamedTuple (legacy)
        is_namedtuple = isinstance(info, tuple) and hasattr(info, '_fields')
        
        assert is_struct_dataclass or has_dataclass_fields, \
            f"Info should be @struct.dataclass (PyTreeNode), got {type(info)}. " \
            f"Please refactor from NamedTuple to @struct.dataclass (see jax_flagcapture.py as example)."
        
        assert not is_namedtuple, \
            f"Info should not be NamedTuple. Found NamedTuple with fields: {getattr(info, '_fields', None)}. " \
            f"Please refactor to @struct.dataclass (see jax_flagcapture.py as example)."
    
    def test_environment_no_verbose_parameter(self, raw_env):
        """Test that environment __init__ does not accept verbose parameter."""
        env_class = raw_env.__class__
        
        # Get the __init__ signature
        try:
            sig = inspect.signature(env_class.__init__)
            params = sig.parameters
        except Exception as e:
            pytest.skip(f"Could not inspect __init__ signature: {e}")
        
        # Check if verbose parameter exists
        has_verbose = 'verbose' in params
        
        assert not has_verbose, \
            f"Environment {env_class.__name__} should not accept 'verbose' parameter in __init__. " \
            f"Found parameters: {list(params.keys())}. " \
            f"Please remove verbose handling (it will be removed in future versions)."
    
    def test_environment_methods_no_verbose_parameter(self, raw_env):
        """Test that environment methods (reset, step) do not accept verbose parameter."""
        env_class = raw_env.__class__
        
        # Check reset method
        try:
            reset_sig = inspect.signature(env_class.reset)
            reset_params = reset_sig.parameters
            reset_has_verbose = 'verbose' in reset_params
        except Exception as e:
            reset_has_verbose = False  # If we can't inspect, assume it's fine
        
        # Check step method
        try:
            step_sig = inspect.signature(env_class.step)
            step_params = step_sig.parameters
            step_has_verbose = 'verbose' in step_params
        except Exception as e:
            step_has_verbose = False  # If we can't inspect, assume it's fine
        
        assert not reset_has_verbose, \
            f"Environment {env_class.__name__}.reset() should not accept 'verbose' parameter. " \
            f"Found parameters: {list(reset_params.keys())}. " \
            f"Please remove verbose handling (it will be removed in future versions)."
        
        assert not step_has_verbose, \
            f"Environment {env_class.__name__}.step() should not accept 'verbose' parameter. " \
            f"Found parameters: {list(step_params.keys())}. " \
            f"Please remove verbose handling (it will be removed in future versions)."
    
    def test_datatype_consistency_across_operations(self, raw_env):
        """Test that datatypes remain consistent across reset and step operations."""
        key = jax.random.PRNGKey(42)
        
        # Reset and check types
        obs1, state1 = raw_env.reset(key)
        
        try:
            from flax import struct
            obs_class1 = type(obs1)
            state_class1 = type(state1)
            obs_is_pytree = issubclass(obs_class1, struct.PyTreeNode) or hasattr(obs_class1, '__dataclass_fields__')
            state_is_pytree = issubclass(state_class1, struct.PyTreeNode) or hasattr(state_class1, '__dataclass_fields__')
        except (ImportError, AttributeError):
            pytest.skip("Flax struct not available")
        
        assert obs_is_pytree, f"Observation type changed: {type(obs1)}"
        assert state_is_pytree, f"State type changed: {type(state1)}"
        
        # Step and check types
        action = raw_env.action_space().sample(key)
        obs2, state2, reward, done, info = raw_env.step(state1, action)
        
        obs_class2 = type(obs2)
        state_class2 = type(state2)
        info_class = type(info)
        
        obs2_is_pytree = issubclass(obs_class2, struct.PyTreeNode) or hasattr(obs_class2, '__dataclass_fields__')
        state2_is_pytree = issubclass(state_class2, struct.PyTreeNode) or hasattr(state_class2, '__dataclass_fields__')
        info_is_pytree = issubclass(info_class, struct.PyTreeNode) or hasattr(info_class, '__dataclass_fields__')
        
        assert obs2_is_pytree, f"Observation type changed after step: {type(obs2)}"
        assert state2_is_pytree, f"State type changed after step: {type(state2)}"
        assert info_is_pytree, f"Info type is incorrect: {type(info)}"
        
        # Verify types are consistent
        assert type(obs1) == type(obs2), \
            f"Observation type changed between reset and step: {type(obs1)} vs {type(obs2)}"
        assert type(state1) == type(state2), \
            f"State type changed between reset and step: {type(state1)} vs {type(state2)}"