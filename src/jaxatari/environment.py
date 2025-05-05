from typing import Tuple, Generic, TypeVar
import jax.numpy as jnp
import jax.random as jrandom


EnvObs = TypeVar("EnvObs")
EnvState = TypeVar("EnvState")
EnvInfo = TypeVar("EnvInfo")


class JaxEnvironment(Generic[EnvState, EnvObs, EnvInfo]):
    """
    Abstract class for a JAX environment.
    Generics:
    EnvState: The type of the environment state.
    EnvObs: The type of the observation.
    EnvInfo: The type of the additional information.
    """

    def __init__(self):
        pass

    def reset(self, key: jrandom.PRNGKey=None) -> Tuple[EnvState, EnvObs]:
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

    def get_action_space(self) -> Tuple:
        """
        Returns the action space of the environment.
        Returns: The action space of the environment as a tuple.
        """
        raise NotImplementedError("Abstract method")

    def get_observation_space(self) -> Tuple:
        """
        Returns the observation space of the environment.
        Returns: The observation space of the environment as a tuple.
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
