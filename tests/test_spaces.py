import pytest
import jax
import jax.numpy as jnp
import numpy as np
import collections

import jaxatari
from jaxatari.wrappers import AtariWrapper, ObjectCentricWrapper, PixelAndObjectCentricWrapper
import jaxatari.spaces as spaces


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


if __name__ == "__main__":
    pytest.main([__file__]) 