import jax
import jax.numpy as jnp
from functools import partial

from jaxatari.games.jax_kaboom import KaboomState
from jaxatari.modification import JaxAtariPostStepModPlugin, JaxAtariInternalModPlugin
from jaxatari.games.jax_freeway import FreewayState


class BombsMoveHorizontally(JaxAtariInternalModPlugin):
    """Bombs can move also on the x-axis while falling"""

    @partial(jax.jit, static_argnums=(0,))
    def bomb_move_horizontally(self, bomb_pos_x,  min_allowed_pos_x, max_allowed_pos_x, subkey):
        shift_x = jnp.round(
            jax.random.normal(subkey, ()) * 1.5  # std deviation controls spread
        ).astype(jnp.int32)
        shift_x = jnp.clip(shift_x, -10, 10)

        new_x = jnp.clip(
            bomb_pos_x + shift_x,
            min_allowed_pos_x,
            max_allowed_pos_x)
        return new_x

class BombsFallFaster(JaxAtariInternalModPlugin):
    """Bombs can fall faster"""

    @partial(jax.jit, static_argnums=(0,))
    def bomb_move_vertically(self, bomb_pos_y, subkey):
        extra_speed = jax.random.randint(subkey, (), 1, 5)
        new_y = bomb_pos_y + extra_speed
        return new_y

class BombsRed(JaxAtariInternalModPlugin):
    """Bombs are red"""

    asset_overrides = {
        "bombs" : {
            'name': 'bombs',
            'type': 'group',
            'files': ['bomb1_red.npy', 'bomb2_red.npy']
        }
    }

class MadBomberGreen(JaxAtariInternalModPlugin):
    """Mad bomber is green"""

    asset_overrides = {
        "mad_bomber" : {
            'name': 'mad_bomber',
            'type': 'single',
            'file': 'mad_bomber_green.npy'
        }
    }
