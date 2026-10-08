import jax
import jax.numpy as jnp
from functools import partial

from jaxtari.games.jax_kaboom import KaboomState
from jaxtari.modification import JaxtariPostStepModPlugin, JaxtariInternalModPlugin
from jaxtari.games.jax_freeway import FreewayState


class BombsMoveHorizontally(JaxtariInternalModPlugin):
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

class BombsFallFaster(JaxtariInternalModPlugin):
    """Bombs can fall faster"""

    @partial(jax.jit, static_argnums=(0,))
    def bomb_move_vertically(self, bomb_pos_y, subkey):
        extra_speed = jax.random.randint(subkey, (), 1, 5)
        new_y = bomb_pos_y + extra_speed
        return new_y

class BombsRed(JaxtariInternalModPlugin):
    """Bombs are red"""

    asset_overrides = {
        "bombs" : {
            'name': 'bombs',
            'type': 'group',
            'files': ['bomb1_red.npy', 'bomb2_red.npy']
        }
    }

class MadBomberGreen(JaxtariInternalModPlugin):
    """Mad bomber is green"""

    asset_overrides = {
        "mad_bomber" : {
            'name': 'mad_bomber',
            'type': 'single',
            'file': 'mad_bomber_green.npy'
        }
    }
