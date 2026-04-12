"""Jaxatari Wrappers"""
from absl.logging import info

import functools
import types
import warnings
from typing import Any, Dict, Tuple, Union, Optional, Callable
from dataclasses import is_dataclass, asdict

import chex
from flax import struct
import jax
import jax.image as jim
import jax.numpy as jnp
from jax import flatten_util
from jaxatari.environment import EnvState, JAXAtariAction as Action
import jaxatari.spaces as spaces
import numpy as np
from jaxatari.rendering.jax_rendering_utils import RendererConfig

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
    env_state: Any
    key: chex.PRNGKey
    step: int
    prev_action: int
    
class AtariWrapper(JaxatariWrapper):
    """
    Wrapper for Atari environments that applies Atari-specific control logic.
    Returns single-step, single-frame observations from the wrapped base env.
    Args:
        env: The environment to wrap.
        sticky_actions: Sticky action probability in [0, 1]. Defaults to 0.25.
        episodic_life: Loss of life -> terminated. Does not reset the environment. Defaults to True.
        first_fire: Take FIRE action on reset. Defaults to True.
        noop_max: Max number of no-op actions to take on reset. Defaults to 30.
        full_action_space: Use full action space of 18 actions. Defaults to False (minimal action set).
        max_frames_per_episode: Maximum number of frames per episode before truncation. Defaults to 108,000 (30 minutes at 60fps).

    Note: Typically, this wrapper is followed by PixelObsWrapper, ObjectCentricWrapper or PixelAndObjectCentricWrapper.
    Frame-skipping, max-pooling, frame-stacking and reward clipping are handled in those.
    """
    def __init__(self, env, sticky_actions: float = 0.25, episodic_life: bool = True, first_fire: bool = True, noop_max: int = 30, full_action_space: bool = False, max_frames_per_episode: int = 108_000):
        super().__init__(env)
        self._env = env
        self.sticky_actions = float(np.clip(sticky_actions, 0.0, 1.0))
        self.episodic_life = episodic_life
        self.first_fire = first_fire
        self.noop_reset = False if noop_max == 0 else True
        self.noop_max = noop_max
        self.full_action_space = full_action_space
        self.max_frames_per_episode = max_frames_per_episode

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

        self._observation_space = self._env.observation_space()

    def observation_space(self) -> spaces.Space:
        """Returns the single-frame base observation space."""
        return self._observation_space
    
    def image_space(self) -> spaces.Box:
        """Returns the image space."""
        return self._env.image_space()

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, AtariState]:
        # Split keys for all potential random operations
        env_key, wrapper_key, noop_key = jax.random.split(key, 3)
        obs, env_state = self._env.reset(env_key)
        step = jnp.array(0, dtype=jnp.int32)
        prev_action = jnp.array(0, dtype=jnp.int32)

        # ========== NOOP RESET ==========
        def perform_noop_reset(carry):
            # This function will be executed if self.noop_reset is True
            env_state, obs, step = carry
            # Generate the random number of no-op steps to take.
            num_noops = jax.random.randint(noop_key, shape=(), minval=0, maxval=self.noop_max + 1)

            def noop_body_fn(i, loop_carry):
                current_obs, current_env_state = loop_carry
                # We always compute the next step for static graph tracing...
                next_obs, next_env_state, reward, done, info = self._env.step(current_env_state, Action.NOOP)
                # Might be done from the no-op step, in which case we reset. 
                next_obs, next_env_state = jax.lax.cond(
                    done,
                    lambda: self._env.reset(env_key),
                    lambda: (next_obs, next_env_state) 
                )
                return next_obs, next_env_state

            # Loop for the static maximum number of no-ops.
            final_obs, final_env_state = jax.lax.fori_loop(0, num_noops, noop_body_fn, (obs, env_state)) 
            
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

        return obs, AtariState(env_state, wrapper_key, step, prev_action)

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state: AtariState, action: int) -> Tuple[chex.Array, AtariState, float, bool, bool, Dict[Any, Any]]:
        step_key, next_state_key = jax.random.split(state.key)

        use_sticky_action = jax.random.uniform(step_key, shape=()) < self.sticky_actions
        new_action = jnp.where(use_sticky_action, state.prev_action, action)
        obs, new_env_state, reward, env_done, infos = self._env.step(state.env_state, new_action)

        terminated = env_done
        if self.episodic_life:
            # If the player has lost a life, we consider the episode done
            if hasattr(state.env_state, "lives"):
                condition = jnp.logical_and(state.env_state.lives > 0, new_env_state.lives < state.env_state.lives)
                terminated = jnp.logical_or(terminated, condition)
            elif hasattr(state.env_state, "lives_lost"):
                terminated = jnp.logical_or(terminated, new_env_state.lives_lost > state.env_state.lives_lost)

        if hasattr(infos, '_asdict'):
            # It's a namedtuple or similar, convert to dict
            info_items = infos._asdict().items()
        elif is_dataclass(infos):
            # It's a dataclass, convert to dict
            info_items = asdict(infos).items()
        else:
            # It's already a dict
            info_items = infos.items()

        info_dict = {k: v for k, v in info_items}

        next_state = AtariState(new_env_state, next_state_key, state.step + 1, new_action)

        # store actual done - not affected by episodic life
        info_dict["env_done"] = env_done

        # store actual reward in info dict before clipping
        info_dict["env_reward"] = reward

        truncated = jnp.where(state.step + 1 >= self.max_frames_per_episode, True, False)

        return obs, next_state, reward, terminated, truncated, info_dict


@struct.dataclass
class ObjectCentricState:
    atari_state: AtariState
    obs_stack: jax.Array

class ObjectCentricWrapper(JaxatariWrapper):
    """
    Wrapper for Atari environments that returns stacked object-centric observations.
    The output observation is a 2D array of shape (frame_stack_size, num_features).
    Apply this wrapper after the AtariWrapper!
    """

    def __init__(self, env, frame_stack_size: int = 4, frame_skip: int = 4, clip_reward: bool = True):
        super().__init__(env)
        assert isinstance(env, AtariWrapper), "ObjectCentricWrapper must be applied after AtariWrapper"
        self.frame_stack_size = frame_stack_size
        self.frame_skip = frame_skip
        self.clip_reward = clip_reward

        # Calculate exact bounds for the flattened single-frame observation.
        single_frame_space = self._env.observation_space()
        lows, highs = [], []

        for leaf_space in jax.tree.leaves(single_frame_space):
            if isinstance(leaf_space, spaces.Box):
                low_arr = np.broadcast_to(leaf_space.low, leaf_space.shape).flatten()
                high_arr = np.broadcast_to(leaf_space.high, leaf_space.shape).flatten()
                lows.append(low_arr)
                highs.append(high_arr)
            else:
                raise TypeError(f"Unsupported space type for flattening: {type(leaf_space)}")

        if not lows:
            raise ValueError("The observation space appears to be empty or contain unsupported types.")

        single_frame_lows = np.concatenate(lows)
        single_frame_highs = np.concatenate(highs)

        self._observation_space = spaces.Box(
            low=single_frame_lows,
            high=single_frame_highs,
            shape=(self.frame_stack_size, int(single_frame_lows.shape[0])),
            dtype=jnp.float32
        )

    def observation_space(self) -> spaces.Box:
        """Returns a Box space for the flattened observation."""
        return self._observation_space

    @functools.partial(jax.jit, static_argnums=(0,))
    def _flatten_single_obs(self, obs):
        """Flatten a single object-centric observation using ravel_pytree."""
        return flatten_util.ravel_pytree(obs)[0].astype(jnp.float32)

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey
    ) -> Tuple[chex.Array, ObjectCentricState]:
        obs, atari_state = self._env.reset(key)
        flat_obs = self._flatten_single_obs(obs)
        obs_stack = jnp.stack([flat_obs] * self.frame_stack_size)
        return obs_stack, ObjectCentricState(atari_state, obs_stack)

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: ObjectCentricState,
        action: int,
    ) -> Tuple[chex.Array, ObjectCentricState, float, bool, bool, Dict[Any, Any]]:

        def body_fn(carry, _):
            atari_state, action = carry
            obs, new_atari_state, reward, terminated, truncated, info = self._env.step(atari_state, action)
            return (new_atari_state, action), (obs, reward, terminated, truncated, info)

        (atari_state, _), (obs, rewards, terminations, truncations, infos) = jax.lax.scan(
            body_fn,
            (state.atari_state, action),
            None,
            length=self.frame_skip,
        )

        latest_obs = jax.tree.map(lambda x: x[-1], obs)
        flat_latest_obs = self._flatten_single_obs(latest_obs)
        obs_stack = jnp.concatenate([state.obs_stack[1:], jnp.expand_dims(flat_latest_obs, axis=0)], axis=0)

        reward = jnp.sum(rewards)
        if self.clip_reward:
            reward = jnp.sign(reward)

        terminated = terminations.any()
        truncated = truncations.any()
        # Autoreset (gym's SAME_STEP mode) -> reset whole stack
        obs_stack, oc_state = jax.lax.cond(
            infos["env_done"].any(),  # use actual env_done for reset condition, not affected by episodic life
            lambda: self.reset(atari_state.key),  # reset if done, using the current key for proper random state advancement
            lambda: (obs_stack, ObjectCentricState(atari_state, obs_stack)),  # step if not done
        )

        def reduce_info(k, v):
            if k in ["env_reward", "all_rewards"]:
                return jnp.sum(v, axis=0)
            if k == "env_done":
                return jnp.any(v)
            return v[-1]

        info_dict = {k: reduce_info(k, v) for k, v in infos.items()}
        return obs_stack, oc_state, reward, terminated, truncated, info_dict


@functools.partial(jax.jit, static_argnums=(0,))
def preprocess_image(class_instance: JaxatariWrapper, image: chex.Array) -> chex.Array:
    """Applies resizing and grayscaling to a single image frame."""
    image = image.astype(jnp.float32)

    # Has to use a standard Python `if` since jax.lax.cond would fail due to different shapes. This is possible since do_pixel_resize is a static parameter.
    if class_instance.do_pixel_resize:
        image = jim.resize(image, (class_instance.pixel_resize_shape[0], class_instance.pixel_resize_shape[1], image.shape[-1]), method='bilinear')
    
    # applies grayscale if enabled with the same method as for resize
    if class_instance.grayscale:
        image = jnp.dot(image, jnp.array([0.2989, 0.5870, 0.1140]))[..., jnp.newaxis] # numbers for grayscale transformation as in https://en.wikipedia.org/wiki/Luma_(video)

    # Apply gaussian smoothing to natively downscaled images to get similar effect to actual downscaling
    if class_instance.native_downscaling and class_instance.smooth_image:

        @functools.partial(jax.jit, static_argnames=['sigma'])
        def gaussian_blur_2d(image, sigma=3.0):
            # image input: [N, C, H, W]
            c = image.shape[1]
            radius = int(sigma * 3)
            size = radius * 2 + 1
            x = jnp.linspace(-radius, radius, size)
            phi_x = jnp.exp(-0.5 * (x / sigma)**2)
            phi_x = (phi_x / phi_x.sum()).astype(image.dtype)

            # 1D Kernels for depthwise conv
            # Horizontal: (C, 1, 1, K)
            h_kernel = phi_x[None, None, None, :]
            h_kernel = jnp.tile(h_kernel, (c, 1, 1, 1))
            
            # Vertical: (C, 1, K, 1)
            v_kernel = phi_x[None, None, :, None]
            v_kernel = jnp.tile(v_kernel, (c, 1, 1, 1))

            # Apply Horizontal pass
            out = jax.lax.conv_general_dilated(
                image, h_kernel, (1, 1), padding='SAME',
                feature_group_count=c,
                dimension_numbers=('NCHW', 'OIHW', 'NCHW')
            )
            # Apply Vertical pass
            out = jax.lax.conv_general_dilated(
                out, v_kernel, (1, 1), padding='SAME',
                feature_group_count=c,
                dimension_numbers=('NCHW', 'OIHW', 'NCHW')
            )
            return out
        image_gauss = gaussian_blur_2d(image[None].transpose(0, 3, 1, 2))
        image = image_gauss.squeeze().reshape(image.shape)
    
    return image.astype(jnp.uint8)

@struct.dataclass 
class PixelState:
    atari_state: AtariState
    image_stack: jax.Array

class PixelObsWrapper(JaxatariWrapper):
    """
    Wrapper for Atari environments that returns the flattened pixel observations.
    Apply this wrapper after the AtariWrapper!
    """
    # TODO: remove do_pixel_resize and resize whenever a different shape / grayscale is given?
    def __init__(self, env, do_pixel_resize: bool = False, pixel_resize_shape: tuple[int, int] = (84, 84), grayscale: bool = False, use_native_downscaling: bool = False, smooth_image: bool = True, frame_stack_size: int = 4, frame_skip: int = 4, max_pooling: bool = True, clip_reward: bool = True):
        super().__init__(env)
        assert isinstance(env, AtariWrapper), "PixelObsWrapper has to be applied after AtariWrapper"
        self.frame_stack_size = frame_stack_size
        self.frame_skip = frame_skip
        self.max_pooling = max_pooling
        self.clip_reward = clip_reward
        self.smooth_image = smooth_image
        self.native_downscaling = False

        # Access the Base Environment
        base_env = self._env._env if isinstance(self._env, AtariWrapper) else self._env

        if do_pixel_resize and use_native_downscaling:
            # call helper from modifications to make sure that applied mods remain applied after native downscaling (lazy import to avoid circular dependency)
            from jaxatari.modification import apply_native_downscaling
            self.do_pixel_resize, self.grayscale = apply_native_downscaling(
                base_env, pixel_resize_shape, grayscale
            )
            self.pixel_resize_shape = pixel_resize_shape
            self.native_downscaling = True
        else:
            self.do_pixel_resize = do_pixel_resize
            self.pixel_resize_shape = pixel_resize_shape
            self.grayscale = grayscale

        # Dynamically calculate the final observation space shape
        # If we hot-swapped, image_space() will now return the correct small size automatically
        final_shape = self._env.image_space().shape
        
        # If we are doing wrapper-side resizing (legacy), we still calculate manually
        if self.do_pixel_resize:
            height, width = self.pixel_resize_shape
            channels = 1 if self.grayscale else final_shape[2]
            final_shape = (height, width, channels)

        # Create the space for a single preprocessed frame
        image_space = spaces.Box(low=0, high=255, shape=final_shape, dtype=jnp.uint8)
        # Stack the single-frame space
        self._observation_space = spaces.stack_space(image_space, self.frame_stack_size)

    def observation_space(self) -> spaces.Space:
        """Returns the stacked image space."""
        return self._observation_space
    

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, PixelState]:
        _, atari_state = self._env.reset(key)
        image = self._env.render(atari_state.env_state)
        processed_image = preprocess_image(self, image)

        image_stack = jnp.stack([processed_image] * self.frame_stack_size)
        return image_stack, PixelState(atari_state, image_stack)
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: PixelState,
        action: int,
    ) -> Tuple[chex.Array, PixelState, float, bool, bool, Dict[Any, Any]]:

        def body_fn(carry, _):
            atari_state, action = carry
            _, new_atari_state, reward, terminated, truncated, info = self._env.step(atari_state, action)
            return (new_atari_state, action), (new_atari_state.env_state, reward, terminated, truncated, info)

        (atari_state, _), (env_states, rewards, terminations, truncations, infos) = jax.lax.scan(
            body_fn,
            (state.atari_state, action),
            None,
            length=self.frame_skip,
        )

        last_env_state = jax.tree.map(lambda x: x[-1], env_states)
        if self.max_pooling and self.frame_skip > 1:
            image = self._env.render(last_env_state)
            prev_env_state = jax.tree.map(lambda x: x[-2], env_states)
            prev_image = self._env.render(prev_env_state)
            latest_image = jnp.maximum(image, prev_image)
        else:
            latest_image = self._env.render(last_env_state)
        processed_image = preprocess_image(self, latest_image)
        image_stack = jnp.concatenate([state.image_stack[1:], jnp.expand_dims(processed_image, axis=0)], axis=0)

        reward = jnp.sum(rewards)
        if self.clip_reward:
            reward = jnp.sign(reward)
        terminated = terminations.any()
        truncated = truncations.any()
        # Autoreset (gym's SAME_STEP mode) -> reset whole stack
        image_stack, pixel_state = jax.lax.cond(
            infos["env_done"].any(),  # use actual env_done for reset condition, not affected by episodic life
            lambda: self.reset(atari_state.key),  
            lambda: (image_stack, PixelState(atari_state, image_stack))
        )

        def reduce_info(k, v):
            if k in ["env_reward", "all_rewards"]:
                return jnp.sum(v, axis=0)
            if k == "env_done":
                return jnp.any(v)
            return v[-1]

        info_dict = {k: reduce_info(k, v) for k, v in infos.items()}
        return image_stack, pixel_state, reward, terminated, truncated, info_dict

@struct.dataclass 
class PixelAndObjectCentricState:
    atari_state: AtariState
    image_stack: jax.Array
    obs_stack: Any

class PixelAndObjectCentricWrapper(JaxatariWrapper):
    """
    Wrapper for Atari environments that returns the flattened pixel observations and object-centric observations.
    Apply this wrapper after the AtariWrapper!
    """
    
    def __init__(self, env, do_pixel_resize: bool = False, pixel_resize_shape: tuple[int, int] = (84, 84), grayscale: bool = False, use_native_downscaling: bool = False, smooth_image: bool = False, frame_stack_size: int = 4, frame_skip: int = 4, max_pooling: bool = True, clip_reward: bool = True):
        super().__init__(env)
        assert isinstance(env, AtariWrapper), "PixelAndObjectCentricWrapper must be applied after AtariWrapper"
        self.frame_stack_size = frame_stack_size
        self.frame_skip = frame_skip
        self.max_pooling = max_pooling
        self.clip_reward = clip_reward
        self.smooth_image = smooth_image
        self.native_downscaling = False
        
        # Access the Base Environment
        base_env = self._env._env if isinstance(self._env, AtariWrapper) else self._env

        if do_pixel_resize and use_native_downscaling:
            # call helper from modifications to make sure that applied mods remain applied after native downscaling (lazy import to avoid circular dependency)
            from jaxatari.modification import apply_native_downscaling
            self.do_pixel_resize, self.grayscale = apply_native_downscaling(
                base_env, pixel_resize_shape, grayscale
            )
            self.pixel_resize_shape = pixel_resize_shape
            self.native_downscaling = True
        else:
            self.do_pixel_resize = do_pixel_resize
            self.pixel_resize_shape = pixel_resize_shape
            self.grayscale = grayscale

        # Part 1: Define the stacked image space.
        # If we hot-swapped, image_space() will now return the correct small size automatically
        final_shape = self._env.image_space().shape
        
        # If we are doing wrapper-side resizing (legacy), we still calculate manually
        if self.do_pixel_resize:
            height, width = self.pixel_resize_shape
            channels = 1 if self.grayscale else final_shape[2]
            final_shape = (height, width, channels)
        
        image_space = spaces.Box(low=0, high=255, shape=final_shape, dtype=jnp.uint8)
        stacked_image_space = spaces.stack_space(image_space, self.frame_stack_size)

        # Part 2: Define the FLATTENED object space with exact bounds.
        single_frame_space = self._env.observation_space()
        lows, highs = [], []

        for leaf_space in jax.tree.leaves(single_frame_space):
            if isinstance(leaf_space, spaces.Box):
                low_arr = np.broadcast_to(leaf_space.low, leaf_space.shape).flatten()
                high_arr = np.broadcast_to(leaf_space.high, leaf_space.shape).flatten()
                lows.append(low_arr)
                highs.append(high_arr)
            else:
                raise TypeError(f"Unsupported space type for flattening: {type(leaf_space)}")
        
        if not lows:
            raise ValueError("The observation space appears to be empty or contain unsupported types.")

        single_frame_lows = np.concatenate(lows)
        single_frame_highs = np.concatenate(highs)

        stacked_object_space_flat = spaces.Box(
            low=single_frame_lows,
            high=single_frame_highs,
            shape=(self.frame_stack_size, int(single_frame_lows.shape[0])),
            dtype=jnp.float32
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
    def _flatten_single_obs(self, obs):
        """Flatten a single object-centric observation using ravel_pytree."""
        return flatten_util.ravel_pytree(obs)[0].astype(jnp.float32)
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey
    ) -> Tuple[Tuple[chex.Array, chex.Array], PixelAndObjectCentricState]:
        obs, atari_state = self._env.reset(key)
        flat_obs = self._flatten_single_obs(obs)
        obs_stack = jnp.stack([flat_obs] * self.frame_stack_size)

        image = self._env.render(atari_state.env_state)
        processed_image = preprocess_image(self, image)
        image_stack = jnp.stack([processed_image] * self.frame_stack_size)

        new_state = PixelAndObjectCentricState(atari_state, image_stack, obs_stack)
        return (image_stack, obs_stack), new_state
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: PixelAndObjectCentricState,
        action: int,
    ) -> Tuple[Tuple[chex.Array, chex.Array], PixelAndObjectCentricState, float, bool, bool, Dict[Any, Any]]:
        def body_fn(carry, _):
            atari_state, action = carry
            obs, new_atari_state, reward, terminated, truncated, info = self._env.step(atari_state, action)
            return (new_atari_state, action), (obs, new_atari_state.env_state, reward, terminated, truncated, info)

        (atari_state, _), (obs, env_states, rewards, terminations, truncations, infos) = jax.lax.scan(
            body_fn,
            (state.atari_state, action),
            None,
            length=self.frame_skip,
        )

        latest_obs = jax.tree.map(lambda x: x[-1], obs)
        flat_latest_obs = self._flatten_single_obs(latest_obs)
        obs_stack = jnp.concatenate([state.obs_stack[1:], jnp.expand_dims(flat_latest_obs, axis=0)], axis=0)

        last_env_state = jax.tree.map(lambda x: x[-1], env_states)
        if self.max_pooling and self.frame_skip > 1:
            image = self._env.render(last_env_state)
            prev_env_state = jax.tree.map(lambda x: x[-2], env_states)
            prev_image = self._env.render(prev_env_state)
            latest_image = jnp.maximum(image, prev_image)
        else:
            latest_image = self._env.render(last_env_state)

        processed_image = preprocess_image(self, latest_image)
        image_stack = jnp.concatenate([state.image_stack[1:], jnp.expand_dims(processed_image, axis=0)], axis=0)

        reward = jnp.sum(rewards)
        if self.clip_reward:
            reward = jnp.sign(reward)
        terminated = terminations.any()
        truncated = truncations.any()
        # Autoreset (gym's SAME_STEP mode) -> reset whole stack
        (image_stack, obs_stack), pixel_oc_state = jax.lax.cond(
            infos["env_done"].any(),  # use actual env_done for reset condition, not affected by episodic life
            lambda: self.reset(atari_state.key),
            lambda: ((image_stack, obs_stack), PixelAndObjectCentricState(atari_state, image_stack, obs_stack))
        )

        def reduce_info(k, v):
            if k in ["env_reward", "all_rewards"]:
                return jnp.sum(v, axis=0)
            if k == "env_done":
                return jnp.any(v)
            return v[-1]

        info_dict = {k: reduce_info(k, v) for k, v in infos.items()}
        return (image_stack, obs_stack), pixel_oc_state, reward, terminated, truncated, info_dict

class PixelAndObjectObsWrapper(PixelAndObjectCentricWrapper):
    """
    Exactly the same as PixelAndObjectCentricWrapper, but return structured OC-obs instead of flattened array.
    """

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey
    ) -> Tuple[Tuple[chex.Array, Any], PixelAndObjectCentricState]:
        obs, atari_state = self._env.reset(key)
        image = self._env.render(atari_state.env_state)
        processed_image = preprocess_image(self, image)
        image_stack = jnp.stack([processed_image] * self.frame_stack_size)

        obs_stack = jax.tree.map(
            lambda leaf: jnp.stack([leaf] * self.frame_stack_size),
            obs,
        )

        new_state = PixelAndObjectCentricState(atari_state, image_stack, obs_stack)
        return (image_stack, obs_stack), new_state
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: PixelAndObjectCentricState,
        action: int,
    ) -> Tuple[Tuple[chex.Array, chex.Array], PixelAndObjectCentricState, float, bool, bool, Dict[Any, Any]]:
        def body_fn(carry, _):
            atari_state, action = carry
            obs, new_atari_state, reward, terminated, truncated, info = self._env.step(atari_state, action)
            return (new_atari_state, action), (obs, new_atari_state.env_state, reward, terminated, truncated, info)

        (atari_state, _), (obs, env_states, rewards, terminations, truncations, infos) = jax.lax.scan(
            body_fn,
            (state.atari_state, action),
            None,
            length=self.frame_skip,
        )

        latest_obs = jax.tree.map(lambda x: x[-1], obs)
        obs_stack = jax.tree.map(
            lambda stack_leaf, obs_leaf: jnp.concatenate([stack_leaf[1:], jnp.expand_dims(obs_leaf, axis=0)], axis=0),
            state.obs_stack,
            latest_obs,
        )

        last_env_state = jax.tree.map(lambda x: x[-1], env_states)
        if self.max_pooling and self.frame_skip > 1:
            image = self._env.render(last_env_state)
            prev_env_state = jax.tree.map(lambda x: x[-2], env_states)
            prev_image = self._env.render(prev_env_state)
            latest_image = jnp.maximum(image, prev_image)
        else:
            latest_image = self._env.render(last_env_state)

        processed_image = preprocess_image(self, latest_image)
        image_stack = jnp.concatenate([state.image_stack[1:], jnp.expand_dims(processed_image, axis=0)], axis=0)

        reward = jnp.sum(rewards)
        terminated = terminations.any()
        truncated = truncations.any()
        # Autoreset (gym's SAME_STEP mode) -> reset whole stack
        (image_stack, obs_stack), pixel_oc_state = jax.lax.cond(
            infos["env_done"].any(),
            lambda: self.reset(atari_state.key),
            lambda: ((image_stack, obs_stack), PixelAndObjectCentricState(atari_state, image_stack, obs_stack)),
        )

        def reduce_info(k, v):
            if k in ["env_reward", "all_rewards"]:
                return jnp.sum(v, axis=0)
            if k == "env_done":
                return jnp.any(v)
            return v[-1]

        info_dict = {k: reduce_info(k, v) for k, v in infos.items()}
        new_state = PixelAndObjectCentricState(atari_state, image_stack, obs_stack)
        return (image_stack, obs_stack), pixel_oc_state, reward, terminated, truncated, info_dict


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
        def flatten_and_cast(leaf):
            flattened = leaf.flatten()
            # Cast to float32 to match space dtype
            return flattened.astype(jnp.float32) if isinstance(leaf, jnp.ndarray) else flattened
        return jax.tree.map(flatten_and_cast, obs_tree)

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.ArrayTree, Any]:
        obs, state = self._env.reset(key)
        processed_obs = self._process_obs(obs)
        return processed_obs, state # State can be passed through directly

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: Any,
        action: int,
    ) -> Tuple[chex.ArrayTree, Any, float, bool, bool, Dict[str, Any]]:
        obs, next_state, reward, terminated, truncated, info = self._env.step(state, action)
        processed_obs = self._process_obs(obs)
        return processed_obs, next_state, reward, terminated, truncated, info


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
        action: int,
    ) -> Tuple[chex.ArrayTree, Any, float, bool, bool, Dict[str, Any]]:
        obs, next_state, reward, terminated, truncated, info = self._env.step(state, action)
        normalized_obs = self._normalize_obs(obs)
        return normalized_obs, next_state, reward, terminated, truncated, info


@struct.dataclass
class LogState:
    atari_state: Any # Can be any of the states from wrappers above
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int

class LogWrapper(JaxatariWrapper):
    """Log episode returns and lengths. An episode ends when the wrapped env returns done=True.
    Uses env_reward from info when present (unclipped); otherwise uses the step reward.
    """

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
        action: int,
    ) -> Tuple[chex.Array, LogState, float, bool, bool, Dict[Any, Any]]:
        obs, atari_state, reward, terminated, truncated, info = self._env.step(state.atari_state, action)
        # use env_reward (unclipped/unchanged) for logging when available
        new_episode_return = state.episode_returns + info.get("env_reward", reward)
        new_episode_length = state.episode_lengths + 1
        # use env_done for logging when available (e.g. to ignore episodic_life)
        done = info.get("env_done", jnp.bool_(terminated))
        state = LogState(
            atari_state=atari_state,
            episode_returns=jnp.where(done, jnp.float32(0), jnp.float32(new_episode_return)),
            episode_lengths=jnp.where(done, jnp.int32(0), jnp.int32(new_episode_length)),
            returned_episode_returns=jnp.where(
                done, jnp.float32(new_episode_return), jnp.float32(state.returned_episode_returns)
            ),
            returned_episode_lengths=jnp.where(
                done, jnp.int32(new_episode_length), jnp.int32(state.returned_episode_lengths)
            ),
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = done
        # Still need to return the actual/wrapped termination signal (affected by episodic life)
        return obs, state, reward, terminated, truncated, info

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
    """Log episode returns and lengths for multiple rewards. An episode ends when the wrapped env returns done=True.
    Apply MultiRewardWrapper to the core env when using this wrapper.
    Final logs: 'returned_episode_returns_0', ... for each reward function; env reward in 'returned_episode_env_returns'.
    """

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey,
    ) -> Tuple[chex.Array, MultiRewardLogState]:
        obs, atari_state = self._env.reset(key)
        # Dummy step to get info structure 
        _, _, _, _, _, dummy_info = self._env.step(atari_state, 0)
        rewards_shape_provider = dummy_info.get("all_rewards", jnp.zeros(1))
        episode_returns_init = jnp.zeros_like(rewards_shape_provider)
        state = MultiRewardLogState(atari_state, 0.0, episode_returns_init, 0, 0.0, episode_returns_init, 0)
        return obs, state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: MultiRewardLogState,
        action: int,
    ) -> Tuple[chex.Array, MultiRewardLogState, float, bool, bool, Dict[Any, Any]]:
        obs, atari_state, reward, terminated, truncated, info = self._env.step(state.atari_state, action)
        new_episode_return_env = state.episode_returns_env + info.get("env_reward", reward)
        all_rewards_step = info.get("all_rewards", jnp.zeros_like(state.episode_returns))
        new_episode_return = state.episode_returns + all_rewards_step
        new_episode_length = state.episode_lengths + 1
        done_ = info.get("env_done", jnp.bool_(terminated))
        state = MultiRewardLogState(
            atari_state=atari_state,
            episode_returns_env=jnp.where(done_, jnp.float32(0), jnp.float32(new_episode_return_env)),
            episode_returns=jnp.where(done_, jnp.zeros_like(state.episode_returns), new_episode_return),
            episode_lengths=jnp.where(done_, jnp.int32(0), jnp.int32(new_episode_length)),
            returned_episode_returns_env=jnp.where(
                done_, jnp.float32(new_episode_return_env), jnp.float32(state.returned_episode_returns_env)
            ),
            returned_episode_returns=jnp.where(
                done_, new_episode_return, state.returned_episode_returns
            ),
            returned_episode_lengths=jnp.where(
                done_, jnp.int32(new_episode_length), jnp.int32(state.returned_episode_lengths)
            ),
        )
        info["returned_episode_env_returns"] = state.returned_episode_returns_env
        for i, r in enumerate(new_episode_return):
            info[f"returned_episode_returns_{i}"] = state.returned_episode_returns[i]
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = done_
        return obs, state, reward, terminated, truncated, info
