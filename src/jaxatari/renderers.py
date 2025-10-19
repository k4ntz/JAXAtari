from typing import TypeVar
from jaxatari.rendering.jax_rendering_utils import RendererConfig

class PyGameRenderer:
    def __init__(self):
        pass

EnvConstants = TypeVar("EnvConstants")
class JAXGameRenderer():
    # TODO: include this properly in the workflow so the config can be set from the core!
    def __init__(self, consts: EnvConstants = None, config: RendererConfig = None):
        self.config = config or RendererConfig()

    def render(self, state):
        pass
