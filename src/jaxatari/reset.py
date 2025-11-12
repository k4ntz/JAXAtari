import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple, Dict, Any


class EnvState(NamedTuple):
    """Container for the current environment state."""
    obs: jnp.ndarray
    rng: jax.Array
    step_count: jnp.int32
    done: jnp.bool_


def reset_env(rng: jax.Array, config: Dict[str, Any]) -> Tuple[jnp.ndarray, EnvState]:
    """Resets an Atari environment to its initial state.

    Args:
        rng: JAX random key.
        config: Environment configuration from init_env_config().

    Returns:
        obs: Initial observation (array).
        state: EnvState with initial values.
    """
    rng, obs_key = jax.random.split(rng)

    # Start with a blank (zero) observation or small noise
    obs = jax.random.uniform(obs_key, config["screen_shape"], dtype=jnp.float32)

    state = EnvState(
        obs=obs,
        rng=rng,
        step_count=jnp.int32(0),
        done=jnp.bool_(False),
    )

    return obs, state
