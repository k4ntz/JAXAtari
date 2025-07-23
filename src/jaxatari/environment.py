from enum import Enum
from typing import Tuple, Generic, TypeVar
import jax.numpy as jnp
import jax.random as jrandom
from src.jaxatari.spaces import Space


EnvObs = TypeVar("EnvObs")
EnvState = TypeVar("EnvState")
EnvInfo = TypeVar("EnvInfo")
EnvConstants = TypeVar("EnvConstants")

class JAXAtariAction:
    """
    "Namespace" for Atari action integer constants.
    These are directly usable in JAX arrays.
    """
    NOOP: int = 0
    FIRE: int = 1
    UP: int = 2
    RIGHT: int = 3
    LEFT: int = 4
    DOWN: int = 5
    UPRIGHT: int = 6
    UPLEFT: int = 7
    DOWNRIGHT: int = 8
    DOWNLEFT: int = 9
    UPFIRE: int = 10
    RIGHTFIRE: int = 11
    LEFTFIRE: int = 12
    DOWNFIRE: int = 13
    UPRIGHTFIRE: int = 14
    UPLEFTFIRE: int = 15
    DOWNRIGHTFIRE: int = 16
    DOWNLEFTFIRE: int = 17

    @classmethod
    def get_all_values(cls) -> jnp.ndarray:
        # For fixed action sets, explicit listing is safest and clearest.
        return jnp.array([
            cls.NOOP, cls.FIRE, cls.UP, cls.RIGHT, cls.LEFT, cls.DOWN,
            cls.UPRIGHT, cls.UPLEFT, cls.DOWNRIGHT, cls.DOWNLEFT,
            cls.UPFIRE, cls.RIGHTFIRE, cls.LEFTFIRE, cls.DOWNFIRE,
            cls.UPRIGHTFIRE, cls.UPLEFTFIRE, cls.DOWNRIGHTFIRE, cls.DOWNLEFTFIRE
        ], dtype=jnp.int32)

class JaxEnvironment(Generic[EnvState, EnvObs, EnvInfo, EnvConstants]):
    """
    Abstract class for a JAX environment.
    Generics:
    EnvState: The type of the environment state.
    EnvObs: The type of the observation.
    EnvInfo: The type of the additional information.
    EnvConstants: The type of the environment constants.
    """

    def __init__(self, consts: EnvConstants = None):
        self.consts = consts

    def reset(self, key: jrandom.PRNGKey=None) -> Tuple[EnvObs, EnvState]:
        """
        Resets the environment to the initial state.
        Returns: The initial observation and the initial environment state.

        """
        raise NotImplementedError("Abstract method")

    def step(
        self, state: EnvState, action
    ) -> Tuple[EnvObs, EnvState, float, bool, EnvInfo]:
        """
        Takes a step in the environment.
        Args:
            state: The current environment state.
            action: The action to take.

        Returns: The observation, the new environment state, the reward, whether the state is terminal, and additional info.

        """
        raise NotImplementedError("Abstract method")

    def render(self, state: EnvState) -> Tuple[jnp.ndarray]:
        """
        Renders the environment state to a single image.
        Args:
            state: The environment state.

        Returns: A single image of the environment state.

        """
        raise NotImplementedError("Abstract method")

    def action_space(self) -> Space:
        """
        Returns the action space of the environment as an array containing the actions that can be taken.
        Returns: The action space of the environment as an array.
        """
        raise NotImplementedError("Abstract method")

    def observation_space(self) -> Space:
        """
        Returns the observation space of the environment.
        Returns: The observation space of the environment.
        """
        raise NotImplementedError("Abstract method")
    
    def image_space(self) -> Space:
        """
        Returns the image space of the environment.
        Returns: The image space of the environment.
        """
        raise NotImplementedError("Abstract method")

    def _get_observation(self, state: EnvState) -> EnvObs:
        """
        Converts the environment state to the observation by filtering out non-relevant information.
        Args:
            state: The environment state.

        Returns: observation

        """
        raise NotImplementedError("Abstract method")

    def obs_to_flat_array(self, obs: EnvObs) -> jnp.ndarray:
        """
        Converts the observation to a flat array.
        Args:
            obs: The observation.
        """
        raise NotImplementedError("Abstract method")

    def _get_info(self, state: EnvState) -> EnvInfo:
        """
        Extracts information from the environment state that is not relevant for the agent.
        Args:
            state: The environment state.

        Returns: info

        """
        raise NotImplementedError("Abstract method")

    def _get_reward(self, previous_state: EnvState, state: EnvState) -> float:
        """
        Calculates the reward from the environment state.
        Args:
            previous_state: The previous environment state.
            state: The environment state.

        Returns: reward

        """
        raise NotImplementedError("Abstract method")

    def _get_done(self, state: EnvState) -> bool:
        """
        Determines if the environment state is a terminal state
        Args:
            state: The environment state.

        Returns: True if the state is terminal, False otherwise.

        """
        raise NotImplementedError("Abstract method")