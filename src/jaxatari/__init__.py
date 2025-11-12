from jaxatari.core import make, list_available_games
import jax
import jax.numpy as jnp
from typing import Dict, Any


def init_env_config(game_name: str) -> Dict[str, Any]:
    """Initializes static environment configuration for a given Atari game.

    Args:
        game_name: Name of the Atari game (e.g., "pong", "seaquest").

    Returns:
        A dictionary of environment metadata and configuration parameters.
    """
    # Basic shape & size defaults — can be modified per game later
    base_config = {
        "screen_shape": (84, 84, 4),   # frame stack shape
        "frame_skip": 4,
        "max_steps": 108000,
        "num_actions": 6,              # default; game-specific override below
        "reward_scale": 1.0,
    }

    # Optional per-game overrides
    game_specifics = {
        "pong": {"num_actions": 6},
        "seaquest": {"num_actions": 18},
        "kangaroo": {"num_actions": 18},
        "freeway": {"num_actions": 3},
    }

    if game_name.lower() not in game_specifics:
        raise ValueError(f"Unsupported game: {game_name}")

    base_config.update(game_specifics[game_name.lower()])
    base_config["game_name"] = game_name.lower()

    return base_config
