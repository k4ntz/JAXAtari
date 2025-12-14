import os
from functools import partial
from typing import NamedTuple, Tuple
import jax 
import jax.lax
import jax.numpy as jnp
import chex
import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from typing import Optional

def _create_tictactoe3d_static_grid():
    
    
    
    
    
    
def _get_default_asset_config():
    
    
    return (
        {'name': 'background', 'type': 'background', 'file': 'background.npy'},
        {'name': 'grid', 'type': 'single', 'file': 'grid.npy'},
        {'name': 'x', 'type': 'single', 'file': 'x.npy'},
        {'name': 'o', 'type': 'single', 'file': 'o.npy'},
    )
    
class TicTacToe3DConstants(NamedTuple):
        
        
        
class TicTacToe3DState(NamedTuple):
    
    


class JaxTicTacToe3D(JaxEnvironment):
    
    

class TicTacToe3DRenderer(JAXGameRenderer):
    def __init__(self, consts : TicTacToe3DConstants= None):
        super().__init__(consts)
        self.consts = consts or TicTacToe3DConstants()
        self.config = render_utils.RendererConfig(
            game_dimensions=(210, 160),
            channels=3,
        )
        self.jr = render_utils.JaxRenderingUtils(self.config) 
    
        
