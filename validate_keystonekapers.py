#!/usr/bin/env python3
"""
Simple test to validate KeystoneKapers basic functionality without JAX dependencies.
This does basic import and structure validation.
"""

import sys
import os

# Add the source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:
        from jaxatari.games.jax_keystonekapers import (
            KeystoneKapersConstants,
            PlayerState,
            ThiefState,
            ObstacleState,
            ElevatorState,
            GameState,
            KeystoneKapersObservation,
            KeystoneKapersInfo,
            JaxKeystoneKapers,
            KeystoneKapersRenderer
        )
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_class_structure():
    """Test class structure and basic instantiation."""
    print("Testing class structure...")
    
    try:
        from jaxatari.games.jax_keystonekapers import KeystoneKapersConstants, JaxKeystoneKapers
        
        # Test constants
        consts = KeystoneKapersConstants()
        print(f"✓ Constants created: screen size {consts.SCREEN_WIDTH}x{consts.SCREEN_HEIGHT}")
        
        # Test custom constants
        custom_consts = KeystoneKapersConstants(PLAYER_SPEED=5)
        print(f"✓ Custom constants: player speed {custom_consts.PLAYER_SPEED}")
        
        return True
    except Exception as e:
        print(f"✗ Class structure test failed: {e}")
        return False

def test_namedtuple_structure():
    """Test that all NamedTuple structures are properly defined."""
    print("Testing NamedTuple structures...")
    
    try:
        from jaxatari.games.jax_keystonekapers import (
            PlayerState, ThiefState, ObstacleState, ElevatorState, GameState,
            KeystoneKapersObservation, KeystoneKapersInfo
        )
        
        # Test NamedTuple field definitions
        player_fields = PlayerState._fields
        print(f"✓ PlayerState fields: {player_fields}")
        
        thief_fields = ThiefState._fields
        print(f"✓ ThiefState fields: {thief_fields}")
        
        obs_fields = KeystoneKapersObservation._fields
        print(f"✓ Observation fields: {obs_fields}")
        
        # Verify expected fields exist
        required_player_fields = {'x', 'y', 'floor', 'is_jumping'}
        if not required_player_fields.issubset(set(player_fields)):
            missing = required_player_fields - set(player_fields)
            print(f"✗ Missing player fields: {missing}")
            return False
        
        required_obs_fields = {'player_x', 'player_y', 'thief_x', 'thief_y', 'score', 'timer'}
        if not required_obs_fields.issubset(set(obs_fields)):
            missing = required_obs_fields - set(obs_fields)
            print(f"✗ Missing observation fields: {missing}")
            return False
        
        print("✓ All required fields present")
        return True
        
    except Exception as e:
        print(f"✗ NamedTuple structure test failed: {e}")
        return False

def test_action_constants():
    """Test that action constants are properly imported."""
    print("Testing action constants...")
    
    try:
        from jaxatari.environment import JAXAtariAction as Action
        
        # Test that key actions exist
        required_actions = ['NOOP', 'LEFT', 'RIGHT', 'UP', 'DOWN', 'FIRE']
        for action in required_actions:
            if not hasattr(Action, action):
                print(f"✗ Missing action: {action}")
                return False
            value = getattr(Action, action)
            print(f"  {action}: {value}")
        
        print("✓ All required actions available")
        return True
        
    except Exception as e:
        print(f"✗ Action constants test failed: {e}")
        return False

def test_method_signatures():
    """Test that required methods exist with correct signatures."""
    print("Testing method signatures...")
    
    try:
        from jaxatari.games.jax_keystonekapers import JaxKeystoneKapers
        import inspect
        
        # Get class without instantiating (to avoid JAX dependency)
        methods_to_check = [
            'reset', 'step', 'render', 'action_space', 'observation_space',
            'image_space', '_get_observation', '_get_reward', '_get_done'
        ]
        
        for method_name in methods_to_check:
            if not hasattr(JaxKeystoneKapers, method_name):
                print(f"✗ Missing method: {method_name}")
                return False
            
            method = getattr(JaxKeystoneKapers, method_name)
            sig = inspect.signature(method)
            print(f"  {method_name}: {sig}")
        
        print("✓ All required methods present")
        return True
        
    except Exception as e:
        print(f"✗ Method signature test failed: {e}")
        return False

def test_game_constants_values():
    """Test that game constants have reasonable values."""
    print("Testing game constants values...")
    
    try:
        from jaxatari.games.jax_keystonekapers import KeystoneKapersConstants
        
        consts = KeystoneKapersConstants()
        
        # Check screen dimensions
        assert 0 < consts.SCREEN_WIDTH <= 1000, f"Invalid screen width: {consts.SCREEN_WIDTH}"
        assert 0 < consts.SCREEN_HEIGHT <= 1000, f"Invalid screen height: {consts.SCREEN_HEIGHT}"
        
        # Check floor positions are in order
        floors = [consts.FLOOR_1_Y, consts.FLOOR_2_Y, consts.FLOOR_3_Y, consts.ROOF_Y]
        assert floors == sorted(floors, reverse=True), f"Floors not in descending order: {floors}"
        
        # Check speeds are positive
        assert consts.PLAYER_SPEED > 0, f"Invalid player speed: {consts.PLAYER_SPEED}"
        assert consts.THIEF_BASE_SPEED > 0, f"Invalid thief speed: {consts.THIEF_BASE_SPEED}"
        
        # Check timer values
        assert consts.BASE_TIMER > 0, f"Invalid base timer: {consts.BASE_TIMER}"
        assert consts.COLLISION_PENALTY > 0, f"Invalid collision penalty: {consts.COLLISION_PENALTY}"
        
        # Check array sizes
        assert consts.MAX_OBSTACLES > 0, f"Invalid max obstacles: {consts.MAX_OBSTACLES}"
        assert consts.MAX_ITEMS > 0, f"Invalid max items: {consts.MAX_ITEMS}"
        
        print("✓ All constant values are reasonable")
        print(f"  Screen: {consts.SCREEN_WIDTH}x{consts.SCREEN_HEIGHT}")
        print(f"  Floors: {floors}")
        print(f"  Speeds: Player={consts.PLAYER_SPEED}, Thief={consts.THIEF_BASE_SPEED}")
        print(f"  Arrays: Obstacles={consts.MAX_OBSTACLES}, Items={consts.MAX_ITEMS}")
        
        return True
        
    except Exception as e:
        print(f"✗ Constants values test failed: {e}")
        return False

def test_renderer_structure():
    """Test renderer class structure."""
    print("Testing renderer structure...")
    
    try:
        from jaxatari.games.jax_keystonekapers import KeystoneKapersRenderer
        from jaxatari.renderers import JAXGameRenderer
        import inspect
        
        # Check inheritance
        assert issubclass(KeystoneKapersRenderer, JAXGameRenderer), "Renderer doesn't inherit from JAXGameRenderer"
        
        # Check required methods
        required_methods = ['render']
        for method_name in required_methods:
            if not hasattr(KeystoneKapersRenderer, method_name):
                print(f"✗ Missing renderer method: {method_name}")
                return False
            
            method = getattr(KeystoneKapersRenderer, method_name)
            sig = inspect.signature(method)
            print(f"  {method_name}: {sig}")
        
        print("✓ Renderer structure valid")
        return True
        
    except Exception as e:
        print(f"✗ Renderer structure test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("=" * 60)
    print("KeystoneKapers Implementation Validation")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Class Structure", test_class_structure),
        ("NamedTuple Structure", test_namedtuple_structure),
        ("Action Constants", test_action_constants),
        ("Method Signatures", test_method_signatures),
        ("Constants Values", test_game_constants_values),
        ("Renderer Structure", test_renderer_structure),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "PASSED" if success else "FAILED"
        icon = "✓" if success else "✗"
        print(f"{icon} {test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All validation tests passed!")
        print("\nKeystoneKapers implementation structure is correct.")
        print("The game follows JAXAtari patterns and should be compatible with the framework.")
        print("\nTo use the game:")
        print("  from jaxatari.games.jax_keystonekapers import JaxKeystoneKapers")
        print("  game = JaxKeystoneKapers()")
        print("\nTo play the game:")
        print("  python scripts/play.py -g keystonekapers")
    else:
        failed = total - passed
        print(f"\n❌ {failed} test(s) failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)