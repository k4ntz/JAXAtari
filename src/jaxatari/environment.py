from enum import Enum
from typing import Tuple, Generic, TypeVar
import jax.numpy as jnp
import jax.random as jrandom
import warnings
from jaxatari.spaces import Space
from flax import struct

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

@struct.dataclass
class ObjectObservation:
    """
    Dataclass for object centric observations of objects in jaxatari environments. 
    Can hold 1 to N objects of the same type (for example 12 sharks in seaquest or 1 player ship in asteroids).
    Should always be instantiated via the create() classmethod to ensure proper default handling.
    Attributes:
        x: x position of the object.
        y: y position of the object.
        width: width of the object.
        height: height of the object.
        active: whether the object is currently active.
    """
    x: jnp.ndarray          # obligatory (int8)
    y: jnp.ndarray          # obligatory (int8)
    width: jnp.ndarray      # obligatory (int8)
    height: jnp.ndarray     # obligatory (int8)

    # --- Additional attributes (will be set to 0 if not used) ---
    active: jnp.ndarray = struct.field(default_factory=lambda: jnp.array(1))          # whether the object is currently active (0 or 1)
    visual_id: jnp.ndarray = struct.field(default_factory=lambda: jnp.array(0))   # visual identifier of the object (different color sprites for example)
    state: jnp.ndarray = struct.field(default_factory=lambda: jnp.array(0))       # state of the object, for example is the ghost in pacman vulnerable [blinking] or not [static] (format depends on game, see the game docs)
    orientation: jnp.ndarray = struct.field(default_factory=lambda: jnp.array(0))       # angle of the object (format depends on game, see the game docs)

    @classmethod
    def create(cls, x, y, width, height, active=None, visual_id=None, state=None, orientation=None):
        # Helper to handle defaults
        if active is None: active = jnp.ones_like(x, dtype=jnp.int32)
        if visual_id is None: visual_id = jnp.zeros_like(x, dtype=jnp.int32)
        if state is None: state = jnp.zeros_like(x, dtype=jnp.int32)
        if orientation is None: orientation = jnp.zeros_like(x, dtype=jnp.int32)
        return cls(x=x, y=y, width=width, height=height, active=active, visual_id=visual_id, state=state, orientation=orientation)

    def __repr__(self):
        try:
            # Handle scalar case (0-d arrays)
            if self.x.ndim == 0:
                try:
                    # Try to get concrete values for cleaner output
                    x, y = int(self.x), int(self.y)
                    w, h = int(self.width), int(self.height)
                    act = int(self.active)
                    ori = float(self.orientation)
                    st = int(self.state)
                    vid = int(self.visual_id)
                    status = "ACTIVE" if act else "INACTIVE"
                    return (f"Object(Single, {status}): Pos=({x}, {y}) | Size=({w}, {h}) | "
                            f"Ori={ori:.1f} | State={st} | VisID={vid}")
                except:
                    # Fallback for Tracers
                    return f"Object(Single): Pos=({self.x}, {self.y}) | Active={self.active}"
            
            # Handle vector case (1-d arrays)
            n = self.x.shape[0]
            lines = [f"ObjectGroup(count={n}):"]
            
            # Limit print length if too huge
            limit = min(n, 20) 
            
            for i in range(limit):
                try:
                    # Try to extract concrete values
                    act = int(self.active[i])
                    status = "ACTIVE" if act else "   -  " # Dim inactive ones
                    
                    x, y = int(self.x[i]), int(self.y[i])
                    w, h = int(self.width[i]), int(self.height[i])
                    ori = float(self.orientation[i])
                    st = int(self.state[i])
                    vid = int(self.visual_id[i])
                    
                    # Formatted table row
                    line = (f"  [{i:2d}] {status} | Pos: ({x:3d}, {y:3d}) | Size: ({w:2d}, {h:2d}) | "
                            f"Ori: {ori:5.1f} | State: {st:2d} | VisID: {vid:2d}")
                except:
                    # Fallback for Tracers
                    line = f"  [{i}] Active={self.active[i]} | Pos=({self.x[i]}, {self.y[i]})"
                
                lines.append(line)
            
            if n > limit:
                lines.append(f"  ... ({n - limit} more objects) ...")
                
            return "\n".join(lines)
        except Exception as e:
            return f"ObjectObservation(Error in __repr__: {e})"


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
        if consts is not None:
            # Check for legacy NamedTuple usage (has _fields but is not a PyTreeNode)
            is_named_tuple = isinstance(consts, tuple) and hasattr(consts, '_fields')
            # Check if it's a Flax PyTreeNode (flax.struct.dataclass instances)
            try:
                from flax import struct
                is_flax_node = isinstance(consts, struct.PyTreeNode)
            except (ImportError, AttributeError):
                is_flax_node = False

            if is_named_tuple and not is_flax_node:
                warnings.warn(
                    f"Performance Warning: {self.__class__.__name__}.consts is a 'NamedTuple'. "
                    "This prevents JAX from treating constants as static metadata, potentially causing excessive recompilation. "
                    "Future versions will require 'flax.struct.PyTreeNode' (and the states/observations/info to flax.struct.dataclass/PyTreeNode). "
                    "Please refactor your constants class.",
                    UserWarning,
                    stacklevel=2
                )

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

    def _get_info(self, state: EnvState, all_rewards: jnp.array = None) -> EnvInfo:
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