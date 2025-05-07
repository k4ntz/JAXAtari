"""Wrappers for pure RL."""

import functools
from typing import Any, Dict, Tuple, Union


import chex
from flax import struct
import jax
import jax.numpy as jnp
from jaxatari.environment import EnvState
from gymnax.environments import spaces


class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


class FlattenObservationWrapper(GymnaxWrapper):
    """Transform the observations of the environment into jnp arrays and flatten.
    Apply this wrapper after the AtariWrapper.
    """

    def observation_space(self) -> spaces.Box:
        assert isinstance(
            self._env.observation_space(), spaces.Box
        ), "Only Box spaces are supported for now."
        new_shape = (self._env.frame_stack_size * self._env.obs_size,)
        return spaces.Box(
            low=self._env.observation_space().low,
            high=self._env.observation_space().high,
            shape=new_shape,
            dtype=self._env.observation_space().dtype,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey
    ) -> Tuple[chex.Array, EnvState]:
        obs, state = self._env.reset(key)
        obs = self._env.obs_to_flat_array(obs)
        chex.assert_shape(obs, (self._env.obs_size * self._env.frame_stack_size,))
        return obs, state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float],
    ) -> Tuple[chex.Array, EnvState, float, bool, Any]:  # dict]:
        obs, state, reward, done, info = self._env.step(key, state, action)
        obs = self._env.obs_to_flat_array(obs)
        return obs, state, reward, done, info

@struct.dataclass 
class AtariState:
    env_state: EnvState 
    step: int
    prev_action: int
    obs_stack: chex.Array
    
class AtariWrapper(GymnaxWrapper):
    def __init__(self, env, sticky_actions: bool = True, frame_stack_size: int = 4, frame_skip: int = 4, max_episode_length: int = 10_000):
        super().__init__(env)
        self.sticky_actions = sticky_actions
        self.frame_stack_size = frame_stack_size
        self.frame_skip = frame_skip
        self.max_episode_length = max_episode_length

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        obs, env_state = self._env.reset(key)
        step = jnp.array(0)
        prev_action = jnp.array(0)

        def expand_and_copy(x):
            x_expanded = jnp.expand_dims(x, axis=0)
            return jnp.concatenate([x_expanded] * self.frame_stack_size, axis=0)

        # Apply transformation to each leaf in the pytree
        obs = jax.tree.map(expand_and_copy, obs)

        return obs, AtariState(env_state, step, prev_action, obs)

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, key: chex.PRNGKey, state: AtariState, action: Union[int, float]) -> Tuple[chex.Array, EnvState, float, bool, Dict[Any, Any]]:
        new_action = action
        if self.sticky_actions:
            # With probability 0.25, we repeat the previous action
            key, repeat_key = jax.random.split(key)
            repeat_prev_action_mask = jax.random.uniform(repeat_key, shape=action.shape) < 0.25
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
        # all results are now shaped: (env_num, frame_skip, obs_size)
        latest_obs = jax.tree.map(lambda x: x[-1], obs)
        # push latest obs into the stack 
        new_obs = jax.tree.map(lambda stack, obs: jnp.concatenate([stack[1:], jnp.expand_dims(obs, axis=0)], axis=0), state.obs_stack, latest_obs)

        reward = jnp.sum(rewards)

        done = jnp.logical_or(dones.any(), state.step >= self.max_episode_length)

        def reduce_info(k, v):
            if k == "all_rewards":
                return v.sum(axis=0)
            else:
                return v[-1]

        info = {
            k: reduce_info(k, v) for k, v in infos._asdict().items()
        }

        new_state = AtariState(new_env_state, state.step + 1, new_action, new_obs)

        # Reset the environment if done
        new_obs, new_state = jax.lax.cond(
            done,
            lambda _: self.reset(key),
            lambda _: (new_obs, new_state),
            operand=None
        )

        return new_obs, new_state, reward, done, info
        

@struct.dataclass
class LogEnvState:
    env_state: EnvState
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int

class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey
    ) -> Tuple[chex.Array, LogEnvState]:
        obs, env_state = self._env.reset(key)
        state = LogEnvState(env_state, 0, 0, 0, 0)
        return obs, state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: LogEnvState,
        action: Union[int, float],
        
    ) -> Tuple[chex.Array, LogEnvState, jnp.ndarray, bool, Dict[Any, Any]]:
        """Step the env.


        Args:
          key: PRNG key.
          state: The current state of the env.
          action: The action to take.


        Returns:
          A tuple of (observation, state, reward, done, info).
        """
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
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
class MultiRewardLogEnvState:
    env_state: EnvState
    episode_returns_env: float
    episode_returns: chex.Array#[float]
    episode_lengths: int
    returned_episode_returns_env: float
    returned_episode_returns: chex.Array#[float]
    returned_episode_lengths: int

class MultiRewardLogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, 
    ) -> Tuple[chex.Array, MultiRewardLogEnvState]:
        obs, env_state = self._env.reset(key)
        dummy_info = self._env.step(key, env_state, 0)[4]
        episode_returns_init = jnp.zeros_like(dummy_info["all_rewards"])
        state = MultiRewardLogEnvState(env_state, 0, episode_returns_init, 0, 0, episode_returns_init, 0)
        return obs, state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: MultiRewardLogEnvState,
        action: Union[int, float],
        
    ) -> Tuple[chex.Array, MultiRewardLogEnvState, jnp.ndarray, bool, Dict[Any, Any]]:
        """Step the env.
        Args:
          key: PRNG key.
          state: The current state of the env.
          action: The action to take.

        Returns:
          A tuple of (observation, state, reward, done, info).
        """
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action
        )
        new_episode_return_env = state.episode_returns_env + reward 
        new_episode_return = state.episode_returns + info["all_rewards"]
        new_episode_length = state.episode_lengths + 1
        state = MultiRewardLogEnvState(
            env_state=env_state,
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