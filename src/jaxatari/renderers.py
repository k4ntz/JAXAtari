from typing import TypeVar
from jaxatari.rendering.jax_rendering_utils import RenderingManager

class PyGameRenderer:
    def __init__(self):
        pass

EnvConstants = TypeVar("EnvConstants")
class JAXGameRenderer():
    def __init__(self, consts: EnvConstants = None, mode: str = 'performance'):
        self.manager = RenderingManager(mode=mode)

    def render(self, state):
        pass
