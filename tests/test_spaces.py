import copy
import pickle
import pytest
import jax
import jax.numpy as jnp
import numpy as np
import collections

import jaxatari
from jaxatari.core import list_available_games
from jaxatari.wrappers import AtariWrapper, ObjectCentricWrapper, PixelAndObjectCentricWrapper
import jaxatari.spaces as spaces
from jaxatari.environment import JAXAtariAction
from jaxatari.spaces import Space, Discrete, Box, Dict, Tuple, stack_space


def test_discrete_space():
    """Tests the functionality of the Discrete space."""
    key = jax.random.PRNGKey(42)
    # Get a discrete space from the environment's action space
    env = jaxatari.make("pong")
    space = env.action_space()

    assert isinstance(space, spaces.Discrete)
    assert space.n == 6  # Pong has 6 actions
    assert space.shape == ()
    assert space.dtype == jnp.int32

    # Test sample()
    sample = space.sample(key)
    assert isinstance(sample, jnp.ndarray)
    assert sample.shape == space.shape
    assert sample.dtype == space.dtype

    # Test contains()
    assert space.contains(sample)
    assert space.contains(jnp.array(0))
    assert space.contains(jnp.array(5))
    assert not space.contains(jnp.array(6))
    assert not space.contains(jnp.array(-1))
    # It should correctly handle floats by casting
    assert space.contains(jnp.array(2.0))

    # Test range()
    low, high = space.range()
    assert low == 0
    assert high == 5

def test_box_space():
    """Tests the functionality of the Box space."""
    key = jax.random.PRNGKey(43)
    # Get a Box space from a wrapper's observation space
    env = ObjectCentricWrapper(AtariWrapper(jaxatari.make("pong")))
    space = env.observation_space()

    assert isinstance(space, spaces.Box)
    assert len(space.shape) == 2  # (stack_size, features)

    # Test sample()
    sample = space.sample(key)
    assert isinstance(sample, jnp.ndarray)
    assert sample.shape == space.shape
    assert sample.dtype == space.dtype
    # Check that sample values are within the defined bounds
    assert jnp.all(sample >= space.low)
    assert jnp.all(sample <= space.high)

    # Test contains()
    assert space.contains(sample)
    # Create an out-of-bounds sample
    out_of_bounds_high = sample.at[0, 0].set(jnp.max(space.high) + 1)
    out_of_bounds_low = sample.at[0, 0].set(jnp.min(space.low) - 1)
    assert not space.contains(out_of_bounds_high)
    assert not space.contains(out_of_bounds_low)
    # Test wrong shape
    assert not space.contains(jnp.zeros((1, 1)))

    # Test range()
    low, high = space.range()
    assert jnp.array_equal(low, space.low)
    assert jnp.array_equal(high, space.high)

def test_dict_space():
    """Tests the functionality of the Dict space."""
    key = jax.random.PRNGKey(44)
    # Get a Dict space from the base environment's observation space
    env = jaxatari.make("pong")
    space = env.observation_space()

    assert isinstance(space, spaces.Dict)
    assert isinstance(space.spaces, collections.OrderedDict)

    # Test sample()
    sample = space.sample(key)
    assert isinstance(sample, collections.OrderedDict)
    assert list(sample.keys()) == list(space.spaces.keys())

    # Test that each value in the sampled dict is contained in its subspace
    for k, v in sample.items():
        assert space.spaces[k].contains(v)

    # Test contains()
    assert space.contains(sample)
    
    # Create an invalid sample with a value out of bounds
    invalid_sample = sample.copy()
    player_pos_y_space = space.spaces["player"].spaces["y"]
    invalid_sample["player"]["y"] = jnp.array(player_pos_y_space.high + 10)
    assert not space.contains(invalid_sample)

    # Create a dict with a missing key
    invalid_sample_missing_key = sample.copy()
    del invalid_sample_missing_key["player"]
    assert not space.contains(invalid_sample_missing_key)
    
def test_tuple_space():
    """Tests the functionality of the Tuple space."""
    key = jax.random.PRNGKey(45)
    # Get a Tuple space from the combined wrapper
    env = PixelAndObjectCentricWrapper(AtariWrapper(jaxatari.make("pong")))
    space = env.observation_space()

    assert isinstance(space, spaces.Tuple)
    assert isinstance(space.spaces, tuple)
    assert len(space.spaces) == 2

    # Test sample()
    sample = space.sample(key)
    assert isinstance(sample, tuple)
    assert len(sample) == 2

    # Test that each item in the sampled tuple is contained in its subspace
    assert space.spaces[0].contains(sample[0])
    assert space.spaces[1].contains(sample[1])

    # Test contains()
    assert space.contains(sample)

    # Create an invalid sample with an out-of-bounds second element
    pixel_obs, object_obs = sample
    object_space = space.spaces[1]
    invalid_object_obs = object_obs.at[0, 0].set(jnp.max(object_space.high) + 1)
    invalid_sample = (pixel_obs, invalid_object_obs)
    assert not space.contains(invalid_sample)

    # Create a tuple with the wrong number of elements
    assert not space.contains((pixel_obs,))

def test_jaxatari_action_constants():
    """Test that JAXAtariAction constants are correctly defined."""
    # Test all action constants
    assert JAXAtariAction.NOOP == 0
    assert JAXAtariAction.FIRE == 1
    assert JAXAtariAction.UP == 2
    assert JAXAtariAction.RIGHT == 3
    assert JAXAtariAction.LEFT == 4
    assert JAXAtariAction.DOWN == 5
    assert JAXAtariAction.UPRIGHT == 6
    assert JAXAtariAction.UPLEFT == 7
    assert JAXAtariAction.DOWNRIGHT == 8
    assert JAXAtariAction.DOWNLEFT == 9
    assert JAXAtariAction.UPFIRE == 10
    assert JAXAtariAction.RIGHTFIRE == 11
    assert JAXAtariAction.LEFTFIRE == 12
    assert JAXAtariAction.DOWNFIRE == 13
    assert JAXAtariAction.UPRIGHTFIRE == 14
    assert JAXAtariAction.UPLEFTFIRE == 15
    assert JAXAtariAction.DOWNRIGHTFIRE == 16
    assert JAXAtariAction.DOWNLEFTFIRE == 17
    
    # Test get_all_values method
    all_values = JAXAtariAction.get_all_values()
    assert isinstance(all_values, jnp.ndarray), "get_all_values should return jnp.ndarray"
    assert all_values.shape == (18,), "get_all_values should have shape (18,)"
    assert all_values.dtype == jnp.int32, "get_all_values should have dtype int32"


def test_spaces_functionality():
    """Test that the spaces module functions work correctly."""
    key = jax.random.PRNGKey(0)
    
    # Test Discrete space
    discrete_space = Discrete(6)
    assert discrete_space.n == 6
    assert discrete_space.shape == ()
    assert discrete_space.dtype == jnp.int32
    
    sample = discrete_space.sample(key)
    assert isinstance(sample, jnp.ndarray)
    assert sample.shape == ()
    assert sample.dtype == jnp.int32
    assert discrete_space.contains(sample)
    
    low, high = discrete_space.range()
    assert low == 0
    assert high == 5
    
    # Test Box space
    box_space = Box(low=0, high=1, shape=(3, 4))
    assert box_space.shape == (3, 4)
    assert box_space.dtype == jnp.float32
    
    sample = box_space.sample(key)
    assert isinstance(sample, jnp.ndarray)
    assert sample.shape == (3, 4)
    assert box_space.contains(sample)
    
    low, high = box_space.range()
    # The low and high values should be arrays with the same shape as the space
    assert low.shape == (3, 4)
    assert high.shape == (3, 4)
    assert jnp.all(low == 0)
    assert jnp.all(high == 1)
    
    # Test Dict space
    dict_space = Dict({
        "x": Box(low=0, high=10, shape=()),
        "y": Box(low=0, high=10, shape=())
    })
    
    sample = dict_space.sample(key)
    assert isinstance(sample, dict)
    assert "x" in sample
    assert "y" in sample
    assert dict_space.contains(sample)
    
    # Test Tuple space
    tuple_space = Tuple([
        Box(low=0, high=1, shape=(2,)),
        Discrete(5)
    ])
    
    sample = tuple_space.sample(key)
    assert isinstance(sample, tuple)
    assert len(sample) == 2
    assert tuple_space.contains(sample)
    
    # Test stack_space function
    stacked_space = stack_space(box_space, 4)
    assert isinstance(stacked_space, Box)
    assert stacked_space.shape == (4, 3, 4)


def test_space_enhancements():
    """Tests the new Pythonic features: equality, iteration, and slicing."""
    
    # --- 1. Test __eq__ (Equality) ---
    # Discrete
    assert spaces.Discrete(5) == spaces.Discrete(5)
    assert spaces.Discrete(5) != spaces.Discrete(6)
    assert spaces.Discrete(5) != spaces.Box(low=0, high=1, shape=())

    # Box
    base_box = spaces.Box(low=0, high=1, shape=(2, 2), dtype=jnp.float32)
    assert base_box == spaces.Box(low=0, high=1, shape=(2, 2), dtype=jnp.float32)
    assert base_box != spaces.Box(low=0, high=2, shape=(2, 2))  # Different high
    assert base_box != spaces.Box(low=0, high=1, shape=(3, 3))  # Different shape
    assert base_box != spaces.Box(low=0, high=1, shape=(2, 2), dtype=jnp.int32)  # Different dtype
    
    # Tuple
    base_tuple = spaces.Tuple([spaces.Discrete(5), base_box])
    assert base_tuple == spaces.Tuple([spaces.Discrete(5), base_box])
    assert base_tuple != spaces.Tuple([spaces.Discrete(6), base_box]) # Different content
    assert base_tuple != spaces.Tuple([base_box, spaces.Discrete(5)]) # Different order

    # Dict
    base_dict = spaces.Dict(collections.OrderedDict([
        ("d", spaces.Discrete(5)),
        ("b", base_box)
    ]))
    assert base_dict == spaces.Dict(collections.OrderedDict([
        ("d", spaces.Discrete(5)),
        ("b", base_box)
    ]))
    assert base_dict != spaces.Dict(collections.OrderedDict([
        ("d", spaces.Discrete(6)), # Different content
        ("b", base_box)
    ]))
    assert base_dict != spaces.Dict(collections.OrderedDict([
        ("b", base_box), # Different order
        ("d", spaces.Discrete(5))
    ]))

    # --- 2. Test Container Methods (__len__, __iter__, __getitem__) ---
    s1 = spaces.Box(0, 1)
    s2 = spaces.Discrete(10)
    tuple_space = spaces.Tuple([s1, s2])

    # Test __len__
    assert len(tuple_space) == 2

    # Test __iter__ for Tuple
    subspaces = [s for s in tuple_space]
    assert subspaces[0] == s1
    assert subspaces[1] == s2
    assert len(subspaces) == 2

    # Test slicing for Tuple
    sliced_space = tuple_space[0:1]
    assert isinstance(sliced_space, spaces.Tuple)
    assert len(sliced_space) == 1
    assert sliced_space[0] == s1
    assert sliced_space == spaces.Tuple([s1])

    # Test __iter__ for Dict
    dict_space = spaces.Dict(collections.OrderedDict([("box", s1), ("discrete", s2)]))
    keys = [k for k in dict_space]
    assert keys == ["box", "discrete"]


def test_space_serialization():
    """Tests that all space objects can be pickled and unpickled correctly."""
    # 1. Create a list of diverse space objects to test
    original_spaces = [
        spaces.Discrete(10),
        spaces.Box(low=-1.0, high=1.0, shape=(3, 4), dtype=jnp.float32),
        spaces.Box(low=0, high=255, shape=(2,), dtype=jnp.uint8),
        spaces.Tuple([
            spaces.Discrete(5),
            spaces.Box(low=0, high=1, shape=(2,))
        ]),
        spaces.Dict(collections.OrderedDict([
            ('position', spaces.Box(low=-1, high=1, shape=(2,))),
            ('type', spaces.Discrete(3))
        ]))
    ]

    # 2. For each space, pickle it, then unpickle it
    for original_space in original_spaces:
        # Serialize the object to a byte string
        pickled_space = pickle.dumps(original_space)
        # Deserialize the byte string back into an object
        unpickled_space = pickle.loads(pickled_space)

        # 3. Assert that the restored object is equal to the original
        assert original_space == unpickled_space
        # As a bonus, check that their hashes are also identical
        assert hash(original_space) == hash(unpickled_space)


def test_space_copying():
    """Tests that all space objects can be shallow and deep copied correctly."""
    # Create a complex, nested space to test both copy types
    original_space = spaces.Tuple([
        spaces.Discrete(5),
        spaces.Box(low=0, high=1, shape=(2,))
    ])

    # 1. Test shallow copy
    shallow_copy = copy.copy(original_space)

    # The new object should be equal in value to the original
    assert shallow_copy == original_space
    # But it should be a different object in memory
    assert shallow_copy is not original_space
    # In a shallow copy, the inner list/tuple of subspaces is the *same* object
    assert shallow_copy.spaces is original_space.spaces


    # 2. Test deep copy
    deep_copy = copy.deepcopy(original_space)

    # The new object should be equal in value to the original
    assert deep_copy == original_space
    # And it should be a different object in memory
    assert deep_copy is not original_space
    # In a deep copy, the inner list/tuple of subspaces is a *different* object
    assert deep_copy.spaces is not original_space.spaces


if __name__ == "__main__":
    pytest.main([__file__]) 