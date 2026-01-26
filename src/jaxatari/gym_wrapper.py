import gymnasium
import gymnasium.envs.functional_jax_env
import jax
import chex
import warnings
from typing import Any, Dict, Union, Optional
import functools
import numpy as np
from dataclasses import is_dataclass, asdict
from gymnasium.envs.functional_jax_env import FunctionalJaxEnv

# Import necessary components from the user's framework
from jaxatari.environment import JaxEnvironment
from jaxatari.spaces import Space, Discrete, Box, Tuple as TupleSpace, Dict as DictSpace


def to_gymnasium_space(space: Space) -> gymnasium.Space:
    """Recursively converts a jaxatari space to a Gymnasium space."""
    if isinstance(space, Discrete):
        return gymnasium.spaces.Discrete(space.n)
    elif isinstance(space, Box):
        # Pass shape explicitly to allow gymnasium to handle scalar bounds correctly.
        return gymnasium.spaces.Box(
            low=np.array(space.low),
            high=np.array(space.high),
            shape=space.shape,
            dtype=space.dtype
        )
    elif isinstance(space, TupleSpace):
        return gymnasium.spaces.Tuple(tuple(to_gymnasium_space(s) for s in space.spaces))
    elif isinstance(space, DictSpace):
        return gymnasium.spaces.Dict(
            {k: to_gymnasium_space(v) for k, v in space.spaces.items()}
        )
    else:
        raise TypeError(f"Unsupported jaxatari space type for conversion: {type(space)}")


class JaxAtariFuncEnv:
    """
    A 'thin' FuncEnv adapter for a raw jaxatari base environment.

    This class's only responsibility is to map the methods of a raw
    JaxEnvironment instance to the pure function API required by FuncEnv.
    It does NOT perform any environment logic like frame skipping itself.
    """

    def __init__(self, jaxatari_env: JaxEnvironment):
        """
        Initializes the functional adapter.

        Args:
            jaxatari_env: A raw, unwrapped jaxatari environment instance.
        """
        self._base_env = jaxatari_env
        self.action_space = self._base_env.action_space()
        # TODO: change this in case we use the OC observations
        self.observation_space = self._base_env.image_space()

    @functools.partial(jax.jit, static_argnums=(0,))
    def initial(self, rng: chex.PRNGKey) -> Any:
        """Generates the initial state of the environment."""
        _obs, state = self._base_env.reset(rng)
        return state

    @functools.partial(jax.jit, static_argnums=(0,))
    def transition(self, state: Any, action: Union[int, float], rng: chex.PRNGKey) -> Any:
        """
        Transitions the environment by a single step.
        Note: The `rng` argument from FuncEnv is unused here because the jaxatari
        environment's `step` function is deterministic given a state. The state itself
        contains any PRNG keys needed for the *next* transition.
        """
        _obs, new_state, _reward, _done, _info = self._base_env.step(state, action)
        return new_state

    @functools.partial(jax.jit, static_argnums=(0,))
    def observation(self, state: Any, rng: chex.PRNGKey) -> chex.ArrayTree:
        """Extracts the observation from the state."""
        obs = self._base_env.render(state)
        return obs

    @functools.partial(jax.jit, static_argnums=(0,))
    def reward(self, state: Any, action: int, next_state: Any, rng: chex.PRNGKey) -> float:
        """Calculates the reward for a single-step transition."""
        return self._base_env._get_reward(state, next_state)

    @functools.partial(jax.jit, static_argnums=(0,))
    def terminal(self, state: Any, rng: chex.PRNGKey) -> bool:
        """Determines if a state is terminal."""
        return self._base_env._get_done(state)

    @functools.partial(jax.jit, static_argnums=(0,))
    def transition_info(self, state: Any, action: int, next_state: Any) -> Dict:
        """
        Extracts info from the environment state resulting from a transition.
        It also calculates the standard `terminated` and `truncated` flags.
        """
        base_info = self._base_env._get_info(next_state)
        
        # 1. Check Modern (Dataclass / Flax)
        if is_dataclass(base_info):
            info = asdict(base_info)
        elif isinstance(base_info, dict):
            info = base_info
        
        # 2. Check Legacy (NamedTuple)
        elif isinstance(base_info, tuple) and hasattr(base_info, '_fields'):
             warnings.warn(
                 "Environment returned a NamedTuple for 'info'. This is deprecated. "
                 "Please return a Dict or a Flax PyTreeNode.",
                 UserWarning
             )
             info = base_info._asdict()
        else:
            info = {}
            
        return info

    def state_info(self, state: Any) -> Dict:
        """Returns info about a state, primarily for the reset info dict."""
        base_info = self._base_env._get_info(state)
        
        if is_dataclass(base_info):
            return asdict(base_info)
        elif isinstance(base_info, dict):
            return base_info
        elif isinstance(base_info, tuple) and hasattr(base_info, '_fields'):
             warnings.warn(
                 "Environment returned a NamedTuple for 'info'. This is deprecated.",
                 UserWarning
             )
             return base_info._asdict()
        else:
            return {}

    def render_init(self, **kwargs):
        """Initializes the renderer. Required by FunctionalJaxEnv if render_mode is used."""
        pass # No state needed for simple rendering

    def render(self, state: Any, **kwargs):
        """Renders the environment."""
        return self._base_env.render(state)

    def render_close(self):
        """Closes the renderer."""
        pass

#TODO: This is a dirty hack, required for gymnasium EpisodicLifeEnv wrapper
class FakeALE:
    def __init__(self, wrapped_env):
        self.wrapped_env = wrapped_env
    def lives(self):
        if hasattr(self.wrapped_env.state, 'lives'):
            return self.wrapped_env.state.lives
        return 0

class GymnasiumJaxAtariWrapper(FunctionalJaxEnv):
    """
    A 'thin' Gymnasium-compatible wrapper for raw jaxatari environments.

    This class bridges a raw jaxatari environment to the standard Gymnasium env
    API. Its sole purpose is to handle the API translation.

    For any preprocessing like frame stacking, frame skipping, or time limits,
    use the standard wrappers from `gymnasium.wrappers` on the instance of this class.

    Example:
        import gymnasium
        from jaxatari.environments import JaxPong
        from jaxatari.gym_wrapper import GymnasiumJaxAtariWrapper

        raw_env = JaxPong()
        gym_env = GymnasiumJaxAtariWrapper(raw_env)
        # Now apply standard gymnasium wrappers
        gym_env = gymnasium.wrappers.FrameStack(gym_env, 4)
        gym_env = gymnasium.wrappers.TimeLimit(gym_env, 1000)
    """
    
    @staticmethod
    def _convert_to_numpy(data):
        """Recursively converts JAX arrays in a nested structure to NumPy arrays."""
        if isinstance(data, (jax.Array, jax.numpy.ndarray)):
            return np.asarray(data)
        elif isinstance(data, tuple):
            return tuple(GymnasiumJaxAtariWrapper._convert_to_numpy(v) for v in data)
        elif isinstance(data, list):
            return [GymnasiumJaxAtariWrapper._convert_to_numpy(v) for v in data]
        elif isinstance(data, dict):
            return {k: GymnasiumJaxAtariWrapper._convert_to_numpy(v) for k, v in data.items()}
        else:
            # Return primitives (int, float, bool) and other types as-is
            return data
    
    def __init__(self,
                 jaxatari_env: JaxEnvironment,
                 metadata: Optional[Dict[str, Any]] = None,
                 render_mode: Optional[str] = None):
        """
        Initializes the Gymnasium wrapper.

        Args:
            jaxatari_env: The raw base jaxatari environment (e.g., JaxPong).
                          Do not pass any jaxatari wrappers.
            metadata: Optional metadata dictionary for Gymnasium.
            render_mode: Optional render mode for Gymnasium.
        """
        # Validate that we're getting a raw base environment
        if not isinstance(jaxatari_env, JaxEnvironment) or hasattr(jaxatari_env, '_env'):
            raise ValueError(
                "GymnasiumJaxAtariWrapper only accepts raw, unwrapped `JaxEnvironment` instances. "
                "Apply wrappers from `gymnasium.wrappers` after initialization."
            )

        # 1. Create the thin functional adapter.
        func_env = JaxAtariFuncEnv(jaxatari_env)

        # 2. Add the 'jax' flag to metadata for downstream tools.
        if metadata is None:
            metadata = {}
        # TODO: this is commented out due to a bug in the gymnasium numpy_to_jax conversion wrapper that I could not resolve
        #metadata["jax"] = True

        # 3. Initialize the parent FunctionalJaxEnv with our adapter.
        # We handle the space conversion outside of the super().__init__
        super().__init__(
            func_env=func_env,
            metadata=metadata,
            render_mode=render_mode
        )

        # 4. Set the correctly converted action and observation spaces.
        self.action_space = to_gymnasium_space(func_env.action_space)
        self.observation_space = to_gymnasium_space(func_env.observation_space)
        
        # Keep a reference to the raw env for rendering
        self._jaxatari_env = jaxatari_env

        # Required for compatibility with EpisodicLifeEnv wrapper (ale.lives())
        self.ale = FakeALE(self)

    @property
    def unwrapped(self):
        """Returns the base environment, satisfying the Gymnasium wrapper API."""
        return self

    def get_action_meanings(self) -> list[str]:
        """
        Returns the standard action meanings for Atari games.
        This is required for compatibility with wrappers like AtariPreprocessing.
        The meanings are hardcoded as they are consistent across all jaxatari games.
        """
        return [
            "NOOP",
            "FIRE",
            "UP",
            "RIGHT",
            "LEFT",
            "DOWN",
            "UPRIGHT",
            "UPLEFT",
            "DOWNRIGHT",
            "DOWNLEFT",
            "UPFIRE",
            "RIGHTFIRE",
            "LEFTFIRE",
            "DOWNFIRE",
            "UPRIGHTFIRE",
            "UPLEFTFIRE",
            "DOWNRIGHTFIRE",
            "DOWNLEFTFIRE",
        ]

    def step(self, action):
        """Overrides the parent step method to manually convert JAX output to NumPy."""
        # The parent method returns JAX arrays
        jax_obs, jax_reward, jax_terminated, jax_truncated, jax_info = super().step(action)
        
        # Manually convert all outputs to NumPy before returning (this is a hacky fix for a bug in the internal gymnasium conversion function)
        np_obs = self._convert_to_numpy(jax_obs)
        np_reward = self._convert_to_numpy(jax_reward)
        np_terminated = self._convert_to_numpy(jax_terminated)
        np_truncated = self._convert_to_numpy(jax_truncated)
        np_info = self._convert_to_numpy(jax_info)
        
        # Ensure correct dtype (e.g. uint8 instead of float32)
        np_obs = self._cast_obs_dtype(np_obs)
        
        return np_obs, np_reward, np_terminated, np_truncated, np_info

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Overrides the parent reset method to manually convert JAX output to NumPy."""
        # The parent method returns JAX arrays
        jax_obs, jax_info = super().reset(seed=seed, options=options)
        # Manually convert all outputs to NumPy before returning
        np_obs = self._convert_to_numpy(jax_obs)
        np_info = self._convert_to_numpy(jax_info)
        
        # Ensure correct dtype (e.g. uint8 instead of float32)
        np_obs = self._cast_obs_dtype(np_obs)
        
        return np_obs, np_info

    def render(self):
        """
        Renders the environment by calling the raw environment's render method.
        """
        if self.render_mode == "rgb_array":
            # self.state is the raw EnvState managed by FunctionalJaxEnv
            frame = self._jaxatari_env.render(self.state)
            # Convert to numpy array (force copy to host if needed)
            frame = np.array(frame)
            # Cast to correct dtype if necessary
            frame = self._cast_obs_dtype(frame)
            # If the frame has a batch dimension, squeeze it out
            if frame.ndim == 4 and frame.shape[0] == 1:
                frame = frame[0]
            return frame
        else:
            raise ValueError(f"Render mode {self.render_mode} not supported")

    # ---------------------------------------------------------------------
    # Utility helpers for dtype handling
    # ---------------------------------------------------------------------

    def _cast_obs_dtype(self, obs):
        """
        New, stricter implementation that walks the observation_space tree and only
        casts *floating-point* arrays to the dtype declared by the corresponding
        leaf space (typically uint8 for Atari image observations).  This prevents
        accidental coercion of unrelated arrays (e.g. float features) when the
        top-level space is a Dict/Tuple without a global `dtype` attribute.
        """

        def _cast_leaf(arr: np.ndarray, target_dtype: np.dtype):
            """Cast *floating-type* arrays to the given `target_dtype` when needed."""
            if arr.dtype != target_dtype and np.issubdtype(arr.dtype, np.floating):
                return arr.astype(target_dtype, copy=False)
            return arr

        def _walk(obs_part, space_part):
            """Recursively walk `obs_part` alongside `space_part` and cast leaves."""
            # Dict space ------------------------------------------------------
            if isinstance(space_part, gymnasium.spaces.Dict):
                if not isinstance(obs_part, dict):
                    return obs_part  # Mismatch – return as-is.
                return {
                    key: _walk(obs_part[key], space_part.spaces[key])
                    for key in space_part.spaces
                }

            # Tuple space -----------------------------------------------------
            if isinstance(space_part, gymnasium.spaces.Tuple):
                if not isinstance(obs_part, (tuple, list)):
                    return obs_part  # Mismatch – return as-is.
                casted = [
                    _walk(o, s) for o, s in zip(obs_part, space_part.spaces)
                ]
                return tuple(casted) if isinstance(obs_part, tuple) else casted

            # Box (leaf) ------------------------------------------------------
            if isinstance(space_part, gymnasium.spaces.Box) and isinstance(obs_part, np.ndarray):
                return _cast_leaf(obs_part, space_part.dtype)

            return obs_part

        return _walk(obs, self.observation_space)
