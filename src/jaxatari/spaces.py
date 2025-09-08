# Copyright 2020 Rémi Louf
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file is in part derived from the Gymnax project (https://github.com/RobertTLange/gymnax)
# and has been modified for the purposes of this project.

import collections
from collections.abc import Sequence
from typing import Optional, Tuple, Union, Any

import jax
from jax.tree_util import register_pytree_node
import jax.numpy as jnp
import numpy as np

class Space:
    """Minimal jittable class for abstract spaces."""

    def sample(self, key: jax.Array) -> jax.Array:
        raise NotImplementedError

    def contains(self, x: jax.Array) -> Any:
        raise NotImplementedError
    
    '''
    Returns the range of the space with the first value being the minimum and the second value being the maximum.
    Only implemented for numerically bounded spaces.
    '''
    def range(self):
        raise NotImplementedError


class Discrete(Space):
    """Minimal jittable class for discrete spaces."""

    def __init__(self, num_categories: int):
        assert num_categories >= 0
        self.n = num_categories
        self.shape = ()
        self.dtype = jnp.int32

    def sample(self, key: jax.Array) -> jax.Array:
        """Sample random action uniformly from set of categorical choices."""
        return jax.random.randint(
            key, shape=self.shape, minval=0, maxval=self.n
        ).astype(self.dtype)

    def contains(self, x: jax.Array) -> jax.Array:
        """Check whether specific object is within space."""
        x = x.astype(jnp.int32)
        range_cond = jnp.logical_and(x >= 0, x < self.n)
        return range_cond
    
    def range(self) -> tuple[float, float]:
        return 0, self.n - 1


class Box(Space):
    """
    A jittable n-dimensional box space.

    This space represents the Cartesian product of n closed intervals. 
    Each interval has its own lower and upper bound.

    It can be initialized in two ways:
    1. With scalar bounds and an explicit shape, creating a box with uniform bounds.
       Example: Box(low=0.0, high=1.0, shape=(3, 4))
    2. With array-like bounds, where the shape is inferred from the bounds arrays.
       Example: Box(low=jnp.array([0., -1.]), high=jnp.array([1., 1.]))
    """

    def __init__(
        self,
        low: Union[float, np.ndarray, jnp.ndarray],
        high: Union[float, np.ndarray, jnp.ndarray],
        shape: Optional[Tuple[int, ...]] = None,
        dtype: jnp.dtype = jnp.float32,
    ):
        """
        Initializes the Box space.

        Args:
            low: The lower bound of the box. Can be a scalar or an array-like object.
            high: The upper bound of the box. Can be a scalar or an array-like object.
            shape: The shape of the space. If None, it's inferred from `low` and `high`.
            dtype: The data type of the space.
        """
        # Determine the shape of the space
        if shape is not None:
            self.shape = shape
        else:
            # If shape is not provided, it must be inferred from the bounds.
            # We require the bounds to have the same shape.
            low_arr = np.asarray(low)
            high_arr = np.asarray(high)
            if low_arr.shape != high_arr.shape:
                raise ValueError(
                    f"low and high must have the same shape, got {low_arr.shape} and {high_arr.shape}"
                )
            self.shape = low_arr.shape

        # Broadcast low and high to the correct shape
        self.low = jnp.broadcast_to(jnp.asarray(low, dtype=dtype), self.shape)
        self.high = jnp.broadcast_to(jnp.asarray(high, dtype=dtype), self.shape)
        self.dtype = dtype
        
        # Broadcasting checks to ensure compatibility
        try:
            np.broadcast_to(self.low, self.shape)
            np.broadcast_to(self.high, self.shape)
        except ValueError:
            raise ValueError(
                f"low and high bounds must be broadcastable to shape {self.shape}. "
                f"Got low.shape={self.low.shape}, high.shape={self.high.shape}"
            )


    def sample(self, key: jax.Array) -> jax.Array:
        """
        Generates a random sample from the space.
        
        The sample is uniformly distributed over the box.
        """
        # Check if the dtype is floating-point or integer and use the appropriate JAX random function.
        if jnp.issubdtype(self.dtype, jnp.floating):
            # Use jax.random.uniform for float dtypes.
            return jax.random.uniform(
                key, shape=self.shape, minval=self.low, maxval=self.high, dtype=self.dtype
            )
        elif jnp.issubdtype(self.dtype, jnp.integer):
            # Use jax.random.randint for integer dtypes.
            # The `Box` space is inclusive of `high`, but `jax.random.randint`'s
            # `maxval` is exclusive. Therefore, we add 1 to `self.high`.
            return jax.random.randint(
                key, shape=self.shape, minval=self.low, maxval=self.high + 1, dtype=self.dtype
            )
        else:
            # Raise an error for unsupported dtypes.
            raise ValueError(f"Unsupported dtype for sampling in Box space: {self.dtype}")

    def contains(self, x: jax.Array) -> jax.Array:
        """Check if a point `x` is contained within the box."""
        # Ensure the input has the correct dtype for comparison
        x = x.astype(self.dtype)
        
        # Check shape compatibility
        if x.shape != self.shape:
            return jnp.asarray(False)
            
        return jnp.logical_and(
            jnp.all(x >= self.low),
            jnp.all(x <= self.high)
        )
    
    def range(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Returns the lower and upper bounds of the space."""
        return self.low, self.high


class Dict(Space):
    """A jittable dictionary of simpler jittable spaces (Pytree container)."""

    def __init__(self, spaces: dict):
        self.spaces = collections.OrderedDict(spaces)
        self.num_spaces = len(self.spaces)

    def sample(self, key: jax.Array) -> collections.OrderedDict:
        key_split = jax.random.split(key, self.num_spaces)
        return collections.OrderedDict(
            [(k, self.spaces[k].sample(key_split[i])) for i, k in enumerate(self.spaces)]
        )

    def contains(self, x: dict) -> jax.Array:
        """Check whether the given Pytree is contained in the space."""
        # Handle named tuples by converting to dict
        if hasattr(x, '_asdict'):
            x = x._asdict()
        
        if not isinstance(x, dict) or self.spaces.keys() != x.keys():
            return jnp.asarray(False)

        # Use explicit iteration
        bools = [
            self.spaces[k].contains(x[k]) for k in self.spaces.keys()
        ]
        return jnp.all(jnp.asarray(bools))
    
    def __repr__(self) -> str:
        return "Dict(" + ", ".join([f"{k}: {s}" for k, s in self.spaces.items()]) + ")"


# Register Dict as a Pytree node for JAX utilities
register_pytree_node(
    Dict,
    lambda s: (list(s.spaces.values()), list(s.spaces.keys())),
    lambda keys, values: Dict(collections.OrderedDict(zip(keys, values)))
)


class Tuple(Space):
    """A jittable tuple of simpler jittable spaces (Pytree container)."""

    def __init__(self, spaces: Sequence[Space]):
        self.spaces = tuple(spaces)
        self.num_spaces = len(self.spaces)

    def sample(self, key: jax.Array) -> tuple:
        """Sample a random tuple from all subspaces."""
        key_split = jax.random.split(key, self.num_spaces)
        return tuple([s.sample(key_split[i]) for i, s in enumerate(self.spaces)])

    def contains(self, x: tuple) -> jax.Array:
        """
        Check whether the given Pytree is contained in the space.
        """
        # Handle named tuples by converting to tuple
        if hasattr(x, '_asdict'):
            # Convert named tuple to regular tuple
            x = tuple(x._asdict().values())
        
        # 1. Initial validation: check if x is a tuple of the correct length.
        if not isinstance(x, (tuple, list)) or len(x) != len(self.spaces):
            return jnp.asarray(False)

        # 2. Correctly iterate and check containment for each subspace.
        #    This zip-based approach is robust, explicit, and avoids the error.
        bools = [
            space.contains(val) for space, val in zip(self.spaces, x)
        ]

        # 3. Combine all boolean results into a single JAX boolean scalar.
        return jnp.all(jnp.asarray(bools))

    def __repr__(self) -> str:
        return "Tuple(" + ", ".join([str(s) for s in self.spaces]) + ")"

# Register Tuple as a Pytree node for JAX utilities
register_pytree_node(
    Tuple,
    lambda s: (s.spaces, None),
    lambda _, children: Tuple(children)
)


def stack_space(space: Space, stack_size: int) -> Space:
    """
    Recursively wraps a space or a Pytree of spaces to add a stacking dimension
    to each leaf space. Handles Box and Discrete spaces as leaves.
    """

    def _stack_leaf(leaf_space: Space) -> Box:
        """Applies stacking logic to a single leaf space."""
        if isinstance(leaf_space, Box):
            # Prepend the stack size to the shape of the Box.
            new_shape = (stack_size,) + leaf_space.shape
            return Box(
                low=leaf_space.low,
                high=leaf_space.high,
                shape=new_shape,
                dtype=leaf_space.dtype,
            )
        if isinstance(leaf_space, Discrete):
            # A stack of Discrete values becomes a Box of integers.
            return Box(
                low=0,
                high=leaf_space.n - 1,
                shape=(stack_size,),
                dtype=leaf_space.dtype,
            )
        # This part should not be reached if `is_leaf` is correctly defined.
        raise TypeError(f"Unsupported leaf space type for stacking: {type(leaf_space)}")

    # Use jax.tree.map to apply the stacking function to every leaf in the space Pytree.
    # The `is_leaf` predicate tells tree_map to stop recursing when it hits a Box or Discrete space,
    # treating them as the leaves to which `_stack_leaf` should be applied.
    return jax.tree.map(
        _stack_leaf, space, is_leaf=lambda n: isinstance(n, (Box, Discrete))
    )