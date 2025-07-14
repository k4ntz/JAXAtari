"""Jaxatari Wrappers"""

import functools
from typing import Any, Dict, Tuple, Union, Optional

import chex
from flax import struct
import jax
import jax.numpy as jnp
from jaxatari.environment import EnvState, JAXAtariAction as Action
import jaxatari.spaces as spaces
import numpy as np

class JaxatariWrapper(object):
    """Base class for JAXAtark wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)

@struct.dataclass
class AtariState:
    env_state: EnvState
    key: chex.PRNGKey
    step: int
    prev_action: int
    obs_stack: chex.Array
    
class AtariWrapper(JaxatariWrapper):
    """
    Wrapper for Atari environments that returns the rendered image and object-centric observations unflattened.
    Both are stacked by frame_stack_size.
    Args:
        env: The environment to wrap.
        sticky_actions: Whether to use sticky actions.
        frame_stack_size: The number of frames to stack.
        frame_skip: The number of frames to skip.
    """
    def __init__(self, env, sticky_actions: bool = True, frame_stack_size: int = 4, frame_skip: int = 4, max_episode_length: int = 10_000, episodic_life: bool = True, first_fire: bool = True):
        super().__init__(env)
        self._env = env
        self.sticky_actions = sticky_actions
        self.frame_stack_size = frame_stack_size
        self.frame_skip = frame_skip
        self.max_episode_length = max_episode_length
        self.episodic_life = episodic_life
        self.first_fire = first_fire

        self._observation_space = spaces.stack_space(self._env.observation_space(), self.frame_stack_size)

    def observation_space(self) -> spaces.Space:
        """Returns the stacked observation space."""
        return self._observation_space

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        obs, env_state = self._env.reset(key)
        step = jnp.array(0)
        prev_action = jnp.array(0)
        if self.first_fire:
            prev_action = Action.FIRE
            obs, env_state, _, _, _ = self._env.step(env_state, prev_action)

        # Create multiple observations directly
        obs = jax.tree.map(lambda x: jnp.stack([x] * self.frame_stack_size), obs)

        return obs, AtariState(env_state, key, step, prev_action, obs)

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state: AtariState, action: Union[int, float]) -> Tuple[Tuple[chex.Array, chex.Array], AtariState, float, bool, Dict[Any, Any]]:
        step_key, next_state_key = jax.random.split(state.key)

        new_action = action
        if self.sticky_actions:
            # Use the local step_key for this step's random operations
            repeat_prev_action_mask = jax.random.uniform(step_key, shape=action.shape) < 0.25
            new_action = jnp.where(repeat_prev_action_mask, state.prev_action, action)

        # use scan to step the env for frame_skip times
        def body_fn(carry, _):
            env_state, action = carry
            obs, new_env_state, reward, done, info = self._env.step(env_state, action) 
            return (new_env_state, action), (obs, reward, done, info)

        (new_env_state, new_action), (obs, rewards, dones, infos) = jax.lax.scan(
            body_fn,
            (state.env_state, new_action),
            None,
            length=self.frame_skip,
        )

        # all results are now shaped: (frame_skip, ...)
        latest_obs = jax.tree.map(lambda x: x[-1], obs)
        # push latest obs into the stack
        new_obs_stack = jax.tree.map(lambda stack, obs_leaf: jnp.concatenate([stack[1:], jnp.expand_dims(obs_leaf, axis=0)], axis=0), state.obs_stack, latest_obs)

        reward = jnp.sum(rewards)
        done = jnp.logical_or(dones.any(), state.step >= self.max_episode_length)
        if self.episodic_life:
            # If the player has lost a life, we consider the episode done
            if hasattr(state.env_state, "lives"):
                done = jnp.logical_or(done, new_env_state.lives < state.env_state.lives)
            elif hasattr(state.env_state, "lives_lost"):
                done = jnp.logical_or(done, new_env_state.lives_lost > state.env_state.lives_lost)

        def reduce_info(k, v):
            if k == "all_rewards":
                return v.sum(axis=0)
            else:
                return v[-1]

        # Convert info to dict and reduce values
        info_dict = {
            k: reduce_info(k, v) for k, v in infos._asdict().items()
        }

        # Use jax.lax.cond to correctly handle state and key propagation on reset
        def _reset_fn(_):
            # When done, reset. The new state will contain the properly advanced next_state_key.
            return self.reset(next_state_key)

        def _step_fn(_):
            # When not done, create the next state, passing next_state_key for the *next* step.
            next_state = AtariState(new_env_state, next_state_key, state.step + 1, new_action, new_obs_stack)
            return new_obs_stack, next_state

        new_obs, new_state = jax.lax.cond(done, _reset_fn, _step_fn, operand=None)

        return new_obs, new_state, reward, done, info_dict

class ObjectCentricWrapper(JaxatariWrapper):
    """
    Wrapper for Atari environments that returns stacked object-centric observations.
    The output observation is a 2D array of shape (frame_stack_size, num_features).
    Apply this wrapper after the AtariWrapper!
    """

    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env, AtariWrapper), "ObjectCentricWrapper must be applied after AtariWrapper"

        # First, get the space for a SINGLE, UNSTACKED frame from the base env.
        single_frame_space = self._env._env.observation_space()

        # Calculate the bounds and size for a single flattened frame.
        lows, highs = [], []
        single_frame_flat_size = 0
        for leaf_space in jax.tree.leaves(single_frame_space):
            if isinstance(leaf_space, spaces.Box):
                low_arr = np.broadcast_to(leaf_space.low, leaf_space.shape)
                high_arr = np.broadcast_to(leaf_space.high, leaf_space.shape)
                lows.append(low_arr.flatten())
                highs.append(high_arr.flatten())
                single_frame_flat_size += np.prod(leaf_space.shape)

        single_frame_lows = np.concatenate(lows)
        single_frame_highs = np.concatenate(highs)

        # create the 2D Box space
        self._observation_space = spaces.Box(
            low=single_frame_lows,
            high=single_frame_highs,
            shape=(self._env.frame_stack_size, int(single_frame_flat_size)),
            dtype=single_frame_lows.dtype
        )
    
    def observation_space(self) -> spaces.Box:
        """Returns a Box space for the flattened observation."""
        return self._observation_space

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey
    ) -> Tuple[chex.Array, EnvState]:
        obs, state = self._env.reset(key)
        # Flatten each frame in the stack
        flat_obs = jax.vmap(self._env.obs_to_flat_array)(obs)
        return flat_obs, state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: AtariState,
        action: Union[int, float],
    ) -> Tuple[chex.Array, EnvState, float, bool, Any]:  # dict]:
        obs, state, reward, done, info = self._env.step(state, action)
        # Flatten each frame in the stack
        flat_obs = jax.vmap(self._env.obs_to_flat_array)(obs)
        return flat_obs, state, reward, done, info
    

@struct.dataclass 
class PixelState:
    atari_state: AtariState 
    key: chex.PRNGKey
    step: int
    prev_action: int
    image_stack: chex.Array

class PixelObsWrapper(JaxatariWrapper):
    """
    Wrapper for Atari environments that returns the flattened pixel observations.
    Apply this wrapper after the AtariWrapper!
    """

    def __init__(self, env):
        super().__init__(env)
        # make sure that env is an AtariWrapper
        assert isinstance(env, AtariWrapper), "PixelObsWrapper has to be applied after AtariWrapper"

        # Calculate observation space once
        image_space = self._env.image_space()
        self._observation_space = spaces.stack_space(image_space, self._env.frame_stack_size)

    def observation_space(self) -> spaces.Box:
        """Returns the stacked image space."""
        return self._observation_space
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey
    ) -> Tuple[chex.Array, EnvState]:
        obs, atari_state = self._env.reset(key)
        image = self._env.render(atari_state.env_state)
        # Create a stack of identical images for the initial state
        image_stack = jnp.stack([image] * self._env.frame_stack_size)
        return image_stack, PixelState(atari_state, key, 0, 0, image_stack)
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: PixelState,
        action: Union[int, float],
    ) -> Tuple[chex.Array, EnvState, float, bool, Any]:
        # Pass the AtariState to the AtariWrapper
        obs, atari_state, reward, done, info = self._env.step(state.atari_state, action)
        image = self._env.render(atari_state.env_state)
        # Update the image stack by shifting and adding the new image
        image_stack = jnp.concatenate([state.image_stack[1:], jnp.expand_dims(image, axis=0)], axis=0)
        new_state = PixelState(atari_state, state.key, state.step + 1, action, image_stack)
        return image_stack, new_state, reward, done, info
    

@struct.dataclass 
class PixelAndObjectCentricState:
    atari_state: AtariState 
    key: chex.PRNGKey
    step: int
    prev_action: int
    image_stack: chex.Array
    obs_stack: chex.Array

class PixelAndObjectCentricWrapper(JaxatariWrapper):
    """
    Wrapper for Atari environments that returns the flattened pixel observations and object-centric observations.
    Apply this wrapper after the AtariWrapper!
    """
    
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env, AtariWrapper), "PixelAndObjectCentricWrapper must be applied after AtariWrapper"
        
        # Part 1: Define the stacked image space. (Correct)
        stacked_image_space = spaces.stack_space(self._env.image_space(), self._env.frame_stack_size)
        
        # Part 2: Define the FLATTENED object space. (This is the FIX)
        # We borrow the exact same logic from ObjectCentricWrapper to ensure consistency.
        single_frame_space = self._env._env.observation_space()
        lows, highs = [], []
        single_frame_flat_size = 0
        for leaf_space in jax.tree.leaves(single_frame_space):
            if isinstance(leaf_space, spaces.Box):
                low_arr = np.broadcast_to(leaf_space.low, leaf_space.shape)
                high_arr = np.broadcast_to(leaf_space.high, leaf_space.shape)
                lows.append(low_arr.flatten())
                highs.append(high_arr.flatten())
                single_frame_flat_size += np.prod(leaf_space.shape)

        single_frame_lows = np.concatenate(lows)
        single_frame_highs = np.concatenate(highs)

        # Create the 2D Box space for the flattened object data.
        stacked_object_space_flat = spaces.Box(
            low=single_frame_lows,
            high=single_frame_highs,
            shape=(self._env.frame_stack_size, int(single_frame_flat_size)),
            dtype=single_frame_lows.dtype
        )

        # Part 3: Combine them into the final Tuple space.
        self._observation_space = spaces.Tuple((
            stacked_image_space,
            stacked_object_space_flat
        ))
    
    
    def observation_space(self) -> spaces.Tuple:
        """Returns a Tuple space containing stacked image and object spaces."""
        return self._observation_space
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey
    ) -> Tuple[chex.Array, EnvState]:
        obs, atari_state = self._env.reset(key)

        # Flatten each frame in the stack
        flat_obs = jax.vmap(self._env.obs_to_flat_array)(obs)

        image = self._env.render(atari_state.env_state)
        # Create a stack of identical images for the initial state
        image_stack = jnp.stack([image] * self._env.frame_stack_size)
        return (image_stack, flat_obs), PixelAndObjectCentricState(atari_state, key, 0, 0, image_stack, flat_obs)
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: PixelAndObjectCentricState,
        action: Union[int, float],
    ) -> Tuple[chex.Array, EnvState, float, bool, Any]:
        # Pass the AtariState to the AtariWrapper
        obs, atari_state, reward, done, info = self._env.step(state.atari_state, action)

        # Flatten each observation in the stack
        flat_obs = jax.vmap(self._env.obs_to_flat_array)(obs)

        image = self._env.render(atari_state.env_state)
        # Update the image stack by shifting and adding the new image
        image_stack = jnp.concatenate([state.image_stack[1:], jnp.expand_dims(image, axis=0)], axis=0)
        new_state = PixelAndObjectCentricState(atari_state, state.key, state.step + 1, action, image_stack, flat_obs)
        return (image_stack, flat_obs), new_state, reward, done, info


class FlattenObservationWrapper(JaxatariWrapper):
    """
    A wrapper that flattens each leaf array in an observation Pytree.

    Compatible with all the other wrappers, flattens the observations whilst preserving the overarching structure (i.e. if the observation is a tuple of multiple observations, the flattened observation will be a tuple of flattened observations).
    """

    def __init__(self, env):
        super().__init__(env)

        # build the new (flattened) observation space
        original_space = self._env.observation_space()

        def flatten_space(space: spaces.Box) -> spaces.Box:
            # Create flattened low/high arrays by broadcasting the original bounds
            # and then reshaping. This preserves the bounds for each element.
            flat_low = np.broadcast_to(space.low, space.shape).flatten()
            flat_high = np.broadcast_to(space.high, space.shape).flatten()
            
            return spaces.Box(
                low=jnp.array(flat_low),
                high=jnp.array(flat_high),
                dtype=space.dtype
            )
        
        self._observation_space = jax.tree.map(
            flatten_space, 
            original_space,
            is_leaf=lambda x: isinstance(x, spaces.Box)
        )


    def observation_space(self) -> spaces.Space:
        """Returns a space where each leaf array is flattened."""
        return self._observation_space

    def _process_obs(self, obs_tree: chex.ArrayTree) -> chex.ArrayTree:
        """Applies .flatten() to each leaf array in the pytree."""
        return jax.tree.map(lambda leaf: leaf.flatten(), obs_tree)

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.ArrayTree, Any]:
        obs, state = self._env.reset(key)
        processed_obs = self._process_obs(obs)
        return processed_obs, state # State can be passed through directly

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: Any,
        action: Union[int, float],
    ) -> Tuple[chex.ArrayTree, Any, float, bool, Dict[str, Any]]:
        obs, next_state, reward, done, info = self._env.step(state, action)
        processed_obs = self._process_obs(obs)
        return processed_obs, next_state, reward, done, info

@struct.dataclass
class PixelState:
    # Only store atari_state and the image_stack. Key, step, etc. are in atari_state.
    atari_state: AtariState
    image_stack: chex.Array

class PixelObsWrapper(JaxatariWrapper):
    """
    Wrapper for Atari environments that returns the flattened pixel observations.
    Apply this wrapper after the AtariWrapper!
    """

    def __init__(self, env):
        super().__init__(env)
        # make sure that env is an AtariWrapper
        assert isinstance(env, AtariWrapper), "PixelObsWrapper has to be applied after AtariWrapper"

        # Calculate observation space once
        image_space = self._env.image_space()
        self._observation_space = spaces.stack_space(image_space, self._env.frame_stack_size)

    def observation_space(self) -> spaces.Box:
        """Returns the stacked image space."""
        return self._observation_space
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey
    ) -> Tuple[chex.Array, PixelState]:
        # Get the full initial AtariState
        _, atari_state = self._env.reset(key)
        image = self._env.render(atari_state.env_state)
        # Create a stack of identical images for the initial state
        image_stack = jnp.stack([image] * self._env.frame_stack_size)
        # Pass the whole atari_state through in the new PixelState
        return image_stack, PixelState(atari_state, image_stack)

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: PixelState,
        action: Union[int, float],
    ) -> Tuple[chex.Array, PixelState, float, bool, Any]:
        # Pass the atari_state from the current PixelState to the underlying env
        _, atari_state, reward, done, info = self._env.step(state.atari_state, action)
        image = self._env.render(atari_state.env_state)
        # Update the image stack by shifting and adding the new image
        image_stack = jnp.concatenate([state.image_stack[1:], jnp.expand_dims(image, axis=0)], axis=0)
        # Create the new state with the NEW atari_state. No other args needed.
        new_state = PixelState(atari_state, image_stack)
        return image_stack, new_state, reward, done, info
    

@struct.dataclass
class PixelAndObjectCentricState:
    atari_state: AtariState
    image_stack: chex.Array
    obs_stack: chex.Array # contains the object-centric stack

class PixelAndObjectCentricWrapper(JaxatariWrapper):
    """
    Wrapper for Atari environments that returns the flattened pixel observations and object-centric observations.
    Apply this wrapper after the AtariWrapper!
    """
    
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env, AtariWrapper), "PixelAndObjectCentricWrapper must be applied after AtariWrapper"
        # Create the 2D Box space for the image data.
        stacked_image_space = spaces.stack_space(self._env.image_space(), self._env.frame_stack_size)

        # Calculate the bounds and size for a single flattened frame.
        single_frame_space = self._env._env.observation_space()
        lows, highs = [], []
        single_frame_flat_size = 0
        for leaf_space in jax.tree.leaves(single_frame_space):
            if isinstance(leaf_space, spaces.Box):
                low_arr = np.broadcast_to(leaf_space.low, leaf_space.shape)
                high_arr = np.broadcast_to(leaf_space.high, leaf_space.shape)
                lows.append(low_arr.flatten())
                highs.append(high_arr.flatten())
                single_frame_flat_size += np.prod(leaf_space.shape)

        single_frame_lows = np.concatenate(lows)
        single_frame_highs = np.concatenate(highs)

        # Create the 2D Box space for the flattened object data.
        stacked_object_space_flat = spaces.Box(
            low=single_frame_lows,
            high=single_frame_highs,
            shape=(self._env.frame_stack_size, int(single_frame_flat_size)),
            dtype=single_frame_lows.dtype
        )

        # Create tuple of both spaces
        self._observation_space = spaces.Tuple((
            stacked_image_space,
            stacked_object_space_flat
        ))

    def observation_space(self) -> spaces.Tuple:
        """Returns a Tuple space containing stacked image and object spaces."""
        return self._observation_space
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey
    ) -> Tuple[Tuple[chex.Array, chex.Array], PixelAndObjectCentricState]:
        # Get the full initial atari_state
        obs_stack, atari_state = self._env.reset(key)

        # Flatten each frame in the stack for the object-centric part
        flat_obs = jax.vmap(self._env.obs_to_flat_array)(obs_stack)

        image = self._env.render(atari_state.env_state)
        # Create a stack of identical images for the initial state
        image_stack = jnp.stack([image] * self._env.frame_stack_size)

        # Create the new state
        new_state = PixelAndObjectCentricState(atari_state, image_stack, flat_obs)
        return (image_stack, flat_obs), new_state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: PixelAndObjectCentricState,
        action: Union[int, float],
    ) -> Tuple[Tuple[chex.Array, chex.Array], PixelAndObjectCentricState, float, bool, Any]:
        # Pass the atari_state from the current state object
        obs_stack, atari_state, reward, done, info = self._env.step(state.atari_state, action)

        # Flatten each observation in the stack
        flat_obs = jax.vmap(self._env.obs_to_flat_array)(obs_stack)

        image = self._env.render(atari_state.env_state)
        # Update the image stack by shifting and adding the new image
        image_stack = jnp.concatenate([state.image_stack[1:], jnp.expand_dims(image, axis=0)], axis=0)
        
        # Create new state using the new atari_state from the step
        new_state = PixelAndObjectCentricState(atari_state, image_stack, flat_obs)
        return (image_stack, flat_obs), new_state, reward, done, info


class FlattenObservationWrapper(JaxatariWrapper):
    """
    A wrapper that flattens each leaf array in an observation Pytree.

    Compatible with all the other wrappers, flattens the observations whilst preserving the overarching structure (i.e. if the observation is a tuple of multiple observations, the flattened observation will be a tuple of flattened observations).
    """

    def __init__(self, env):
        super().__init__(env)

        # build the new (flattened) observation space
        original_space = self._env.observation_space()

        def flatten_space(space: spaces.Box) -> spaces.Box:
            # Create flattened low/high arrays by broadcasting the original bounds
            # and then reshaping. This preserves the bounds for each element.
            flat_low = np.broadcast_to(space.low, space.shape).flatten()
            flat_high = np.broadcast_to(space.high, space.shape).flatten()
            
            return spaces.Box(
                low=jnp.array(flat_low),
                high=jnp.array(flat_high),
                dtype=space.dtype
            )
        
        self._observation_space = jax.tree.map(
            flatten_space,
            original_space,
            is_leaf=lambda x: isinstance(x, spaces.Box)
        )

    def observation_space(self) -> spaces.Space:
        """Returns a space where each leaf array is flattened."""
        return self._observation_space

    def _process_obs(self, obs_tree: chex.ArrayTree) -> chex.ArrayTree:
        """Applies .flatten() to each leaf array in the pytree."""
        return jax.tree.map(lambda leaf: leaf.flatten(), obs_tree)

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.ArrayTree, Any]:
        obs, state = self._env.reset(key)
        processed_obs = self._process_obs(obs)
        return processed_obs, state # State can be passed through directly

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: Any,
        action: Union[int, float],
    ) -> Tuple[chex.ArrayTree, Any, float, bool, Dict[str, Any]]:
        obs, next_state, reward, done, info = self._env.step(state, action)
        processed_obs = self._process_obs(obs)
        return processed_obs, next_state, reward, done, info

@struct.dataclass
class LogState:
    atari_state: Any # Can be any of the states from wrappers above
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int

class LogWrapper(JaxatariWrapper):
    """Log the episode returns and lengths."""

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey
    ) -> Tuple[chex.Array, LogState]:
        obs, atari_state = self._env.reset(key)
        state = LogState(atari_state, 0.0, 0, 0.0, 0)
        return obs, state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: LogState,
        action: Union[int, float],
    ) -> Tuple[chex.Array, LogState, float, bool, Dict[Any, Any]]:
        obs, atari_state, reward, done, info = self._env.step(state.atari_state, action)
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogState(
            atari_state=atari_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = done
        return obs, state, reward, done, info

@struct.dataclass
class MultiRewardLogState:
    atari_state: Any # Can be any of the states from wrappers above
    episode_returns_env: float
    episode_returns: chex.Array
    episode_lengths: int
    returned_episode_returns_env: float
    returned_episode_returns: chex.Array
    returned_episode_lengths: int

class MultiRewardLogWrapper(JaxatariWrapper):
    """Log the episode returns and lengths for multiple rewards."""

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey,
    ) -> Tuple[chex.Array, MultiRewardLogState]:
        obs, atari_state = self._env.reset(key)
        # Dummy step to get info structure
        _, _, _, _, dummy_info = self._env.step(atari_state, 0)
        episode_returns_init = jnp.zeros_like(dummy_info["all_rewards"])
        state = MultiRewardLogState(atari_state, 0.0, episode_returns_init, 0, 0.0, episode_returns_init, 0)
        return obs, state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: MultiRewardLogState,
        action: Union[int, float],
    ) -> Tuple[chex.Array, MultiRewardLogState, float, bool, Dict[Any, Any]]:
        obs, atari_state, reward, done, info = self._env.step(state.atari_state, action)
        new_episode_return_env = state.episode_returns_env + reward
        new_episode_return = state.episode_returns + info["all_rewards"]
        new_episode_length = state.episode_lengths + 1
        state = MultiRewardLogState(
            atari_state=atari_state,
            episode_returns_env=new_episode_return_env * (1 - done),
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns_env=state.returned_episode_returns_env * (1 - done)
            + new_episode_return_env * done,
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
        )
        info["returned_episode_env_returns"] = state.returned_episode_returns_env
        for i, r in enumerate(new_episode_return):
            info[f"returned_episode_returns_{i}"] = state.returned_episode_returns[i]
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = done
        return obs, state, reward, done, info