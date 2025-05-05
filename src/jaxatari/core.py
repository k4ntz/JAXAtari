import json
import jax

from jaxatari.environment import JaxEnvironment
from jaxatari.games.jax_pong import JaxPong, PongRenderer
from jaxatari.games.jax_seaquest import JaxSeaquest, SeaquestRenderer
from jaxatari.games.jax_kangaroo import JaxKangaroo, KangarooRenderer
from jaxatari.games.jax_freeway import JaxFreeway, FreewayRenderer

class JAXAtari:
    def __init__(self, game_name):
        renderer = None
        match game_name:
            case "pong":
                env = JaxPong()
                renderer = PongRenderer()
            case "seaquest":
                env = JaxSeaquest()
                renderer = SeaquestRenderer()
            case "kangaroo":
                env = JaxKangaroo()
                renderer = KangarooRenderer()
            case "freeway":
                env = JaxFreeway()
                renderer = FreewayRenderer()
            case _:
                raise NotImplementedError(f"The game {game_name} does not exist")
        self.env: JaxEnvironment = env
        self.renderer = renderer

    def reset(self, key=None):
        fn = jax.jit(self.env.reset)
        obs, state = fn(key)
        return obs, state

    def get_init_state(self):
        fn = jax.jit(self.env.reset)
        obs, state = fn()
        return state

    def step_state_only(self, state, action):
        fn = jax.jit(self.env.step)
        obs, state, reward, done, info = fn(state, action)
        return state

    def step_with_render(self, state, action):
        fn = jax.jit(self.env.step)
        obs, state, reward, done, info = fn(state, action)

        fn_2 = jax.jit(self.env.render)
        fn_2(state)
        return state

    def step(self, state, action):
        fn = jax.jit(self.env.step)
        return fn(state, action)

    def render(self, state):
        fn = jax.jit(self.env.render)
        fn(state)

    def save_state_as_json(self, state, path):
        state_dict = state._asdict()
        for item in state_dict:
            state_dict[item] = state_dict[item].tolist()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state_dict, f, ensure_ascii=False, indent=4)

    def load_state_from_json(self, curr_state, path):
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        new_state = curr_state.__class__(**state)
        return new_state