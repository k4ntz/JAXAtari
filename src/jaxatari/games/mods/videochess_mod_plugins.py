from functools import partial
import jax
from jaxatari.modification import JaxAtariPostStepModPlugin

class RandomBotBlackMod(JaxAtariPostStepModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        # Unwrap controller/wrapper chain to reach JaxVideoChess
        env = self._env
        for _ in range(6):
            if hasattr(env, "_env"):
                env = env._env
            else:
                break

        return env.random_black_reply(prev_state, new_state)


class GreedyBotBlackMod(JaxAtariPostStepModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        # Unwrap controller/wrapper chain to reach JaxVideoChess
        env = self._env
        for _ in range(6):
            if hasattr(env, "_env"):
                env = env._env
            else:
                break

        return env.greedy_black_reply(prev_state, new_state)