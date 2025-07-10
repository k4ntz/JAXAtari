import functools
import jax

from jaxatari.wrappers import JaxatariWrapper

class SpeedMode(JaxatariWrapper):
    """Increase speed to maximum at all time steps."""
    def __init__(self, env):
        super().__init__(env)
        self._env = env
        # Overrides get_ball_velocity from env
        self._env.get_ball_velocity = self.get_ball_velocity.__get__(self._env, self._env.__class__) 

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_ball_velocity(self, speed_idx, direction_idx, step_counter):
        """Returns the ball's velocity based on the speed and direction indices."""
        # Overrides the default function from the env
        direction = self._env.consts.BALL_DIRECTIONS[direction_idx]
        speed = 3
        return speed * direction[0], speed * direction[1]

class SmallPaddle(JaxatariWrapper):
    """Always use a small paddle."""
    def __init__(self, env):
        super().__init__(env)
        self._env = env
        self._env.consts.PLAYER_SIZE = (4, 4)
        self._env.consts.PLAYER_SIZE_SMALL = (4, 4)

class BigPaddle(JaxatariWrapper):
    """Always use a bigger paddle."""
    def __init__(self, env):
        super().__init__(env)
        self._env = env
        self._env.consts.PLAYER_SIZE = (40, 4)
        self._env.consts.PLAYER_SIZE_SMALL = (40, 4)