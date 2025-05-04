from functools import partial
import jax
import jax.numpy as jnp
import jaxatari.rendering.atraJaxis as aj
import os
from tennis_main import WIDTH, HEIGHT, TennisState

def load_sprites():
    MODULE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    BG = jnp.expand_dims(aj.loadFrame(os.path.join(MODULE_DIR, "tennis_from_scratch/sprites/background.npy")), axis=0)

    return BG

class TennisRenderer:

    def __init__(self):
        self.BG = load_sprites()

    #@partial(jax.jit, static_argnums=(0,))
    def render(self, state: TennisState):
        raster = jnp.zeros((WIDTH, HEIGHT, 3))

        frame_bg = aj.get_sprite_frame(self.BG, 0)
        raster = aj.render_at(raster, 0, 0, frame_bg)

        #rectangle = jnp.zeros((152, 206, 4))
        #rectangle = rectangle.at[:, :, 0].set(255)  # Red
        #rectangle = rectangle.at[:, :, 3].set(255)  # Alpha

        #raster = aj.render_at(raster, 0, 0, rectangle)

        return raster
