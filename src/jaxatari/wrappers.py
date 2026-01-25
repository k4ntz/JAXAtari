"""Jaxatari Wrappers"""

import functools
from typing import Any, Dict, Tuple, Union, Optional, Callable
from dataclasses import is_dataclass, asdict

import chex
from flax import struct
import jax
import jax.image as jim
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
    
class MultiRewardWrapper(JaxatariWrapper):
    """
    Allows providing multiple reward functions to be computed at every step.
    Apply this wrapper directly after the base environment, before any other wrappers.
    """

    def __init__(self, env, reward_funcs: list[Callable]):
        super().__init__(env)
        assert isinstance(reward_funcs, list) and len(reward_funcs) > 0, "reward_funcs must be a non-empty list of callables" 
        self._reward_funcs = reward_funcs

    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_all_rewards(self, previous_state: EnvState, state: EnvState) -> chex.Array:
        """Compute multiple rewards based on the provided reward functions."""
        if self._reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array(
            [reward_func(previous_state, state) for reward_func in self._reward_funcs]
        )
        return rewards

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state: EnvState, action: int) -> Tuple[chex.Array, EnvState, float, bool, Dict]: 
        obs, new_state, reward, done, info = self._env.step(state, action)
        all_rewards = self._get_all_rewards(state, new_state)
        # Convert info to dict: handle NamedTuple (has _asdict) or dataclass (use asdict)
        if hasattr(info, '_asdict'):
            info = info._asdict()
        elif is_dataclass(info):
            info = asdict(info)
        info["all_rewards"] = all_rewards
        return obs, new_state, reward, done, info 

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
    # TODO: change sticky_actions to float
    def __init__(self, env, sticky_actions: bool = True, frame_stack_size: int = 4, frame_skip: int = 4, max_episode_length: int = 10_000, episodic_life: bool = True, first_fire: bool = True, noop_reset: int = 0, clip_reward: bool = False, max_pooling: bool = False, full_action_space: bool = False,):
        super().__init__(env)
        self._env = env
        self.sticky_actions = sticky_actions
        self.frame_stack_size = frame_stack_size
        self.frame_skip = frame_skip
        self.max_episode_length = max_episode_length
        self.episodic_life = episodic_life
        self.first_fire = first_fire
        self.noop_reset = False if noop_reset == 0 else True
        self.noop_max = noop_reset
        self.clip_reward = clip_reward
        self.max_pooling = max_pooling
        self.full_action_space = full_action_space

        # --- 1) HANDLE FULL ACTION SPACE LOGIC ---
        # If requested, swap the environment's (minimal) action set for the full identity set.
        # This keeps each game env "clean" while enabling a central switch for experimentation.
        if self.full_action_space and hasattr(self._env, "ACTION_SET"):
            # Overwrite the instance attribute with [0, 1, ... 17]
            self._env.ACTION_SET = jnp.arange(18, dtype=jnp.int32)

        # --- 2) RESOLVE CORRECT 'FIRE' ACTION INDEX ---
        # The wrapped env expects an *index* into ACTION_SET (agent action), not the ALE action constant.
        self.fire_action_index: int = int(Action.FIRE)  # fallback if env doesn't expose ACTION_SET
        self.first_fire = first_fire

        if hasattr(self._env, "ACTION_SET"):
            # Convert to numpy for search (safe in __init__)
            action_set_np = np.array(self._env.ACTION_SET)
            fire_indices = np.where(action_set_np == int(Action.FIRE))[0]
            if len(fire_indices) > 0:
                self.fire_action_index = int(fire_indices[0])
            else:
                # Game has no FIRE action (e.g. Freeway).
                # Disable first_fire to prevent sending a random command by mistake.
                self.first_fire = False

        self._observation_space = spaces.stack_space(self._env.observation_space(), self.frame_stack_size)

    def observation_space(self) -> spaces.Space:
        """Returns the stacked observation space."""
        return self._observation_space
    
    def image_space(self) -> spaces.Box:
        """Returns the image space."""
        return self._env.image_space()

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        # Split keys for all potential random operations
        env_key, wrapper_key, noop_key = jax.random.split(key, 3)
        obs, env_state = self._env.reset(env_key)
        step = jnp.array(0, dtype=jnp.int32)
        prev_action = jnp.array(0, dtype=jnp.int32)

        # TODO: in which order should the noop and first_fire be done?
        # ========== NOOP RESET ==========
        def perform_noop_reset(carry):
            # This function will be executed if self.noop_reset is True
            env_state, obs, step = carry
            # Generate the random number of no-op steps to take.
            num_noops = jax.random.randint(noop_key, shape=(), minval=0, maxval=self.noop_max + 1)

            def noop_body_fn(i, loop_carry):
                current_env_state, current_obs = loop_carry
                # We always compute the next step for static graph tracing...
                next_obs, next_env_state, _, _, _ = self._env.step(current_env_state, Action.NOOP)
                # ...but only apply the update if the loop index is less than our dynamic random number.
                env_state_out = jax.lax.cond(i < num_noops, lambda: next_env_state, lambda: current_env_state)
                obs_out = jax.lax.cond(i < num_noops, lambda: next_obs, lambda: current_obs)
                return env_state_out, obs_out

            # Loop for the static maximum number of no-ops.
            final_env_state, final_obs = jax.lax.fori_loop(0, self.noop_max, noop_body_fn, (env_state, obs))
            
            # Update the step counter by the dynamic number of no-ops performed.
            final_step = step + num_noops
            return final_env_state, final_obs, final_step

        # Use lax.cond to conditionally apply the whole no-op block based on the static self.noop_reset flag.
        env_state, obs, step = jax.lax.cond(
            self.noop_reset,
            lambda carry: perform_noop_reset(carry),
            lambda carry: carry,
            (env_state, obs, step)
        )

        # ========== FIRST FIRE ==========
        def perform_first_fire(carry):
            env_state, obs, step, _ = carry
            fire_obs, fire_env_state, _, _, _ = self._env.step(env_state, self.fire_action_index)
            return fire_env_state, fire_obs, step + 1, self.fire_action_index

        def identity_fire(carry):
            return carry
        
        # Conditionally apply the fire action based on the static self.first_fire flag.
        env_state, obs, step, prev_action = jax.lax.cond(
            self.first_fire,
            perform_first_fire,
            identity_fire,
            (env_state, obs, step, prev_action)
        )

        # Create the initial frame stack from the final observation.
        obs = jax.tree.map(lambda x: jnp.stack([x] * self.frame_stack_size), obs)

        return obs, AtariState(env_state, wrapper_key, step, prev_action, obs)

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state: AtariState, action: Union[int, float]) -> Tuple[Tuple[chex.Array, chex.Array], AtariState, float, bool, Dict[Any, Any]]:
        step_key, next_state_key = jax.random.split(state.key)

        new_action = action
        # Use lax.cond and fix shape for scalar actions
        use_sticky_action = jax.random.uniform(step_key, shape=()) < 0.25
        new_action = jax.lax.cond(self.sticky_actions & use_sticky_action, lambda: state.prev_action, lambda: action)

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

        # ========== MAX POOLING LOGIC ==========
        def do_max_pool(obs_pytree):
            # Take the element-wise maximum over the last two frames.
            last_obs = jax.tree.map(lambda x: x[-1], obs_pytree)
            second_last_obs = jax.tree.map(lambda x: x[-2], obs_pytree)
            return jax.tree.map(jnp.maximum, last_obs, second_last_obs)

        def take_last_frame(obs_pytree):
            # Default behavior: just take the final frame.
            return jax.tree.map(lambda x: x[-1], obs_pytree)
        
        # Conditionally apply max-pooling based on the static flag.
        latest_obs = jax.lax.cond(self.max_pooling, do_max_pool, take_last_frame, obs)

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

        if hasattr(infos, '_asdict'):
            # It's a namedtuple or similar, convert to dict
            info_items = infos._asdict().items()
        elif is_dataclass(infos):
            # It's a dataclass, convert to dict
            info_items = asdict(infos).items()
        else:
            # It's already a dict
            info_items = infos.items()

        info_dict = {k: reduce_info(k, v) for k, v in info_items}

        # Use jax.lax.cond to correctly handle state and key propagation on reset
        def _reset_fn(_):
            # When done, reset. The new state will contain the properly advanced next_state_key.
            return self.reset(next_state_key)

        def _step_fn(_):
            # When not done, create the next state, passing next_state_key for the *next* step.
            next_state = AtariState(new_env_state, next_state_key, state.step + 1, new_action, new_obs_stack)
            return new_obs_stack, next_state

        new_obs, new_state = jax.lax.cond(done, _reset_fn, _step_fn, operand=None)

        reward = jax.lax.cond(
            self.clip_reward,
            lambda reward: jnp.sign(reward),
            lambda reward: reward,
            reward
        )

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

        # Calculate the bounds and size for a single flattened frame for all leaf spaces.
        lows, highs = [], []
        single_frame_flat_size = 0
        for leaf_space in jax.tree.leaves(single_frame_space):
            if isinstance(leaf_space, spaces.Box):
                # Flatten the bounds arrays for Box spaces
                low_arr = np.broadcast_to(leaf_space.low, leaf_space.shape).flatten()
                high_arr = np.broadcast_to(leaf_space.high, leaf_space.shape).flatten()
                lows.append(low_arr)
                highs.append(high_arr)
                single_frame_flat_size += low_arr.size
            elif isinstance(leaf_space, spaces.Discrete):
                # A Discrete space flattens to a single value
                lows.append(np.array([0], dtype=leaf_space.dtype))
                highs.append(np.array([leaf_space.n - 1], dtype=leaf_space.dtype))
                single_frame_flat_size += 1
            else:
                raise TypeError(f"Unsupported space type for flattening: {type(leaf_space)}")
        
        if not lows:
            raise ValueError("The observation space appears to be empty or contain unsupported types.")

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
    image_stack: chex.Array


class PixelObsWrapper(JaxatariWrapper):
    """
    Wrapper for Atari environments that returns the flattened pixel observations.
    Apply this wrapper after the AtariWrapper!
    """

    def __init__(self, env, do_pixel_resize: bool = False, pixel_resize_shape: tuple[int, int] = (84, 84), grayscale: bool = False):
        super().__init__(env)
        assert isinstance(env, AtariWrapper), "PixelObsWrapper has to be applied after AtariWrapper"

        self.do_pixel_resize = do_pixel_resize
        self.pixel_resize_shape = pixel_resize_shape
        self.grayscale = grayscale

        # Dynamically calculate the final observation space shape
        base_shape = self._env.image_space().shape
        height, width, channels = base_shape

        if self.do_pixel_resize:
            height, width = self.pixel_resize_shape
        if self.grayscale:
            channels = 1
        
        final_shape = (height, width, channels)
        # Create the space for a single preprocessed frame
        image_space = spaces.Box(low=0, high=255, shape=final_shape, dtype=jnp.uint8)
        # Stack the single-frame space
        self._observation_space = spaces.stack_space(image_space, self._env.frame_stack_size)

    def observation_space(self) -> spaces.Box:
        """Returns the stacked image space."""
        return self._observation_space
    
    def _preprocess_image(self, image: chex.Array) -> chex.Array:
        """Applies resizing and grayscaling to a single image frame."""
        image = image.astype(jnp.float32)

        # Has to use a standard Python `if` since jax.lax.cond would fail due to different shapes. This is possible since do_pixel_resize is a static parameter.
        if self.do_pixel_resize:
            image = jim.resize(image, (self.pixel_resize_shape[0], self.pixel_resize_shape[1], image.shape[-1]), method='bilinear')
        
        # applies grayscale if enabled with the same method as for resize
        if self.grayscale:
            image = jnp.dot(image, jnp.array([0.2989, 0.5870, 0.1140]))[..., jnp.newaxis] # numbers for grayscale transformation as in https://en.wikipedia.org/wiki/Luma_(video)
        
        return image.astype(jnp.uint8)

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, PixelState]:
        # The underlying AtariWrapper returns its own state, which we store.
        _, atari_state = self._env.reset(key)
        image = self._env.render(atari_state.env_state)
        
        processed_image = self._preprocess_image(image)

        # Create a stack of identical processed images for the initial state
        image_stack = jnp.stack([processed_image] * self._env.frame_stack_size)
        
        return image_stack, PixelState(atari_state, image_stack)
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: PixelState,
        action: Union[int, float],
    ) -> Tuple[chex.Array, EnvState, float, bool, Any]:
        # Pass the nested atari_state to the underlying wrapper's step function
        _, atari_state, reward, done, info = self._env.step(state.atari_state, action)
        
        image = self._env.render(atari_state.env_state)
        processed_image = self._preprocess_image(image)

        # Update the image stack by shifting and adding the new processed image
        image_stack = jnp.concatenate([state.image_stack[1:], jnp.expand_dims(processed_image, axis=0)], axis=0)

        # Create the new state with the *new* atari_state from the step
        new_state = PixelState(atari_state, image_stack)
        return image_stack, new_state, reward, done, info

@struct.dataclass 
class PixelAndObjectCentricState:
    atari_state: AtariState
    image_stack: chex.Array
    obs_stack: chex.Array

class PixelAndObjectCentricWrapper(JaxatariWrapper):
    """
    Wrapper for Atari environments that returns the flattened pixel observations and object-centric observations.
    Apply this wrapper after the AtariWrapper!
    """
    
    def __init__(self, env, do_pixel_resize: bool = False, pixel_resize_shape: tuple[int, int] = (84, 84), grayscale: bool = False):
        super().__init__(env)
        assert isinstance(env, AtariWrapper), "PixelAndObjectCentricWrapper must be applied after AtariWrapper"
        
        # Part 1: Define the stacked image space.
        self.do_pixel_resize = do_pixel_resize
        self.pixel_resize_shape = pixel_resize_shape
        self.grayscale = grayscale

        base_shape = self._env.image_space().shape
        height, width, channels = base_shape
        if self.do_pixel_resize:
            height, width = self.pixel_resize_shape
        if self.grayscale:
            channels = 1
        final_shape = (height, width, channels)
        image_space = spaces.Box(low=0, high=255, shape=final_shape, dtype=jnp.uint8)
        stacked_image_space = spaces.stack_space(image_space, self._env.frame_stack_size)

        # Part 2: Define the FLATTENED object space (with the bug fix).
        single_frame_space = self._env._env.observation_space()
        lows, highs = [], []
        single_frame_flat_size = 0
        for leaf_space in jax.tree.leaves(single_frame_space):
            if isinstance(leaf_space, spaces.Box):
                low_arr = np.broadcast_to(leaf_space.low, leaf_space.shape).flatten()
                high_arr = np.broadcast_to(leaf_space.high, leaf_space.shape).flatten()
                lows.append(low_arr)
                highs.append(high_arr)
                single_frame_flat_size += low_arr.size
            elif isinstance(leaf_space, spaces.Discrete):
                lows.append(np.array([0], dtype=leaf_space.dtype))
                highs.append(np.array([leaf_space.n - 1], dtype=leaf_space.dtype))
                single_frame_flat_size += 1
            else:
                raise TypeError(f"Unsupported space type for flattening: {type(leaf_space)}")
        
        if not lows:
            raise ValueError("The observation space appears to be empty or contain unsupported types.")

        single_frame_lows = np.concatenate(lows)
        single_frame_highs = np.concatenate(highs)

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
    
    def _preprocess_image(self, image: chex.Array) -> chex.Array:
        """Applies resizing and grayscaling to a single image frame."""
        image = image.astype(jnp.float32)

        # Has to use a standard Python `if` since jax.lax.cond would fail due to different shapes. This is possible since do_pixel_resize is a static parameter.
        if self.do_pixel_resize:
            image = jim.resize(image, (self.pixel_resize_shape[0], self.pixel_resize_shape[1], image.shape[-1]), method='bilinear')
        
        # applies grayscale if enabled with the same method as for resize
        if self.grayscale:
            image = jnp.dot(image, jnp.array([0.2989, 0.5870, 0.1140]))[..., jnp.newaxis] # numbers for grayscale transformation as in https://en.wikipedia.org/wiki/Luma_(video)
        
        return image.astype(jnp.uint8)
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey
    ) -> Tuple[chex.Array, EnvState]:
        # 1. Get the initial object observation stack and state from the AtariWrapper
        obs_stack, atari_state = self._env.reset(key)

        # 2. Flatten the object-centric part
        flat_obs = jax.vmap(self._env.obs_to_flat_array)(obs_stack)

        # 3. Render and preprocess the image
        image = self._env.render(atari_state.env_state)
        processed_image = self._preprocess_image(image)
        image_stack = jnp.stack([processed_image] * self._env.frame_stack_size)

        # 4. Create the state and observation tuple
        new_state = PixelAndObjectCentricState(atari_state, image_stack, flat_obs)
        return (image_stack, flat_obs), new_state
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: PixelAndObjectCentricState,
        action: Union[int, float],
    ) -> Tuple[chex.Array, EnvState, float, bool, Any]:
        # 1. Step the underlying environment using its state
        obs_stack, atari_state, reward, done, info = self._env.step(state.atari_state, action)

        # 2. Flatten the new object-centric observation stack
        flat_obs = jax.vmap(self._env.obs_to_flat_array)(obs_stack)

        # 3. Render and preprocess the new image
        image = self._env.render(atari_state.env_state)
        processed_image = self._preprocess_image(image)
        
        # 4. Update the image stack with the new processed image
        image_stack = jnp.concatenate([state.image_stack[1:], jnp.expand_dims(processed_image, axis=0)], axis=0)
        
        # 5. Create the new state with the new atari_state
        new_state = PixelAndObjectCentricState(atari_state, image_stack, flat_obs)
        return (image_stack, flat_obs), new_state, reward, done, info
    
class PixelAndObjectObsWrapper(PixelAndObjectCentricWrapper):
    """
    Exactly the same as PixelAndObjectCentricWrapper, but return structured OC-obs instead of flattened array.
    """

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey
    ) -> Tuple[chex.Array, EnvState]:
        # 1. Get the initial object observation stack and state from the AtariWrapper
        obs_stack, atari_state = self._env.reset(key)

        # 3. Render and preprocess the image
        image = self._env.render(atari_state.env_state)
        processed_image = self._preprocess_image(image)
        image_stack = jnp.stack([processed_image] * self._env.frame_stack_size)

        # 4. Create the state and observation tuple
        new_state = PixelAndObjectCentricState(atari_state, image_stack, obs_stack)
        return (image_stack, obs_stack), new_state
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: PixelAndObjectCentricState,
        action: Union[int, float],
    ) -> Tuple[chex.Array, EnvState, float, bool, Any]:
        # 1. Step the underlying environment using its state
        obs_stack, atari_state, reward, done, info = self._env.step(state.atari_state, action)

        # 3. Render and preprocess the new image
        image = self._env.render(atari_state.env_state)
        processed_image = self._preprocess_image(image)
        
        # 4. Update the image stack with the new processed image
        image_stack = jnp.concatenate([state.image_stack[1:], jnp.expand_dims(processed_image, axis=0)], axis=0)
        
        # 5. Create the new state with the new atari_state
        new_state = PixelAndObjectCentricState(atari_state, image_stack, obs_stack)
        return (image_stack, obs_stack), new_state, reward, done, info


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
    

class NormalizeObservationWrapper(JaxatariWrapper):
    """
    A wrapper that normalizes each leaf in an observation Pytree.
    This wrapper is compatible with any observation structure (Pytrees).
    """

    def __init__(self, env, to_neg_one: bool = False, dtype=jnp.float16):
        super().__init__(env)
        self._to_neg_one = to_neg_one
        self._dtype = dtype

        original_space = self._env.observation_space()

        # Create Pytrees of the same structure as observations, but holding the low/high bounds.
        self._low = jax.tree.map(
            lambda s: jnp.array(s.low, dtype=s.dtype),
            original_space,
            is_leaf=lambda x: isinstance(x, spaces.Box)
        )
        self._high = jax.tree.map(
            lambda s: jnp.array(s.high, dtype=s.dtype),
            original_space,
            is_leaf=lambda x: isinstance(x, spaces.Box)
        )

        # The new observation space will have the same structure, but all leaves
        def _normalize_space(space: spaces.Box) -> spaces.Box:
            low_val = -1.0 if self._to_neg_one else 0.0
            return spaces.Box(
                low=low_val,
                high=1.0,
                shape=space.shape,
                dtype=self._dtype
            )

        self._observation_space = jax.tree.map(
            _normalize_space,
            original_space,
            is_leaf=lambda x: isinstance(x, spaces.Box)
        )

    def observation_space(self) -> spaces.Space:
        """Returns the normalized observation space where leaves are in [0, 1]."""
        return self._observation_space

    def _normalize_leaf(self, obs_leaf, low_leaf, high_leaf):
        """Helper function to normalize a single leaf array."""
        obs_leaf = obs_leaf.astype(self._dtype)
        
        # Calculate the range and scale for normalization
        range_leaf = high_leaf.astype(self._dtype) - low_leaf.astype(self._dtype)
        scale = 1.0 / jnp.where(range_leaf > 1e-8, range_leaf, 1.0)
        
        # Normalize to [0, 1]
        normalized_0_1 = (obs_leaf - low_leaf.astype(self._dtype)) * scale

        # Conditionally shift to [-1, 1]
        final_normalized = jax.lax.cond(
            self._to_neg_one,
            lambda x: 2.0 * x - 1.0,
            lambda x: x,
            normalized_0_1
        )
        
        # Clip to ensure values are within the target range
        clip_low = -1.0 if self._to_neg_one else 0.0
        return jnp.clip(final_normalized, clip_low, 1.0)

    def _normalize_obs(self, obs: chex.ArrayTree) -> chex.ArrayTree:
        """
        Applies normalization to each leaf array in the observation pytree,
        robustly handling structural mismatches between observation and space Pytrees.
        """
        # Get the leaves of all pytrees. Since the number of leaves and their
        # order is guaranteed to be the same, we can work with the flat lists.
        obs_leaves = jax.tree.leaves(obs)
        low_leaves = jax.tree.leaves(self._low)
        high_leaves = jax.tree.leaves(self._high)

        # Apply the normalization to each corresponding leaf triplet.
        normalized_leaves = [
            self._normalize_leaf(o, l, h)
            for o, l, h in zip(obs_leaves, low_leaves, high_leaves)
        ]

        # Reconstruct the output pytree with the same structure as the input 'obs'.
        return jax.tree.unflatten(jax.tree.structure(obs), normalized_leaves)

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.ArrayTree, Any]:
        obs, state = self._env.reset(key)
        normalized_obs = self._normalize_obs(obs)
        return normalized_obs, state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: Any,
        action: Union[int, float],
    ) -> Tuple[chex.ArrayTree, Any, float, bool, Dict[str, Any]]:
        obs, next_state, reward, done, info = self._env.step(state, action)
        normalized_obs = self._normalize_obs(obs)
        return normalized_obs, next_state, reward, done, info



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
    """Log the episode returns and lengths for multiple rewards.
    Make sure to apply MultiRewardWrapper to the core env when using this wrapper.
    The final logs will be 'returned_episode_returns_0', ... for each reward function provided.
    """

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey,
    ) -> Tuple[chex.Array, MultiRewardLogState]:
        obs, atari_state = self._env.reset(key)
        # Dummy step to get info structure 
        _, _, _, _, dummy_info = self._env.step(atari_state, 0)
        rewards_shape_provider = dummy_info.get("all_rewards", jnp.zeros(1))
        episode_returns_init = jnp.zeros_like(rewards_shape_provider)
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
        # Safely get all_rewards, defaulting to a zero array that matches the shape of our tracker.
        all_rewards_step = info.get("all_rewards", jnp.zeros_like(state.episode_returns))
        new_episode_return = state.episode_returns + all_rewards_step
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
