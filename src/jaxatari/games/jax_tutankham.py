from typing import NamedTuple, Tuple
import numpy as np
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.spaces import Discrete, Box
from jaxatari.renderers import JAXGameRenderer
import chex


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
class TutankhamConstants(NamedTuple):
    WIDTH: int = 160
    HEIGHT: int = 210
    SPEED: int = 4
    PIXEL_COLOR: Tuple[int, int, int] = (255, 255, 255)  # white

    #PLAYER_SIZE: Tuple[int, int] = (5, 10)
    #MISSILE_SIZE: Tuple[int, int] = (1, 2)

# ---------------------------------------------------------------------
# Game State
# ---------------------------------------------------------------------
class TutankhamState(NamedTuple):
    player_x: int
    player_y: int

    #missile_states: chex.Array # (2, 6) array with (x, y, speed_x, speed_y, rotation, lifespan) for each missile
    #missile_rdy: chex.Array # tracks whether the player can fire a missile


# ---------------------------------------------------------------------
# Renderer (No JAX)
# ---------------------------------------------------------------------
class TutankhamRenderer(JAXGameRenderer):
    def __init__(self):
        super().__init__()
        self.consts = TutankhamConstants()

    def render(self, state: TutankhamState) -> np.ndarray:
        frame = np.zeros(
            (self.consts.HEIGHT, self.consts.WIDTH, 3), dtype=np.uint8
        )

        x = min(max(state.player_x, 0), self.consts.WIDTH - 1)
        y = min(max(state.player_y, 0), self.consts.HEIGHT - 1)

        frame[y, x] = np.array(self.consts.PIXEL_COLOR, dtype=np.uint8)
        return frame


# ---------------------------------------------------------------------
# Environment (No JAX)
# ---------------------------------------------------------------------
class JaxTutankham(JaxEnvironment):
    def __init__(self):
        consts = TutankhamConstants()
        super().__init__(consts)
        self.renderer = TutankhamRenderer()
        self.consts = consts

        self.action_set = [
            Action.NOOP,
            Action.UP,
            Action.DOWN,
            Action.LEFT,
            Action.RIGHT,
            Action.RIGHTFIRE,
            Action.LEFTFIRE,
            Action.UPLEFTFIRE,
            Action.UPRIGHTFIRE
        ]

    # -----------------------------
    # Reset
    # -----------------------------
    def reset(self, key=None):
        start_x = self.consts.WIDTH // 2
        start_y = self.consts.HEIGHT // 2

        state = TutankhamState(player_x=start_x, player_y=start_y)
        return state, state

    # -----------------------------
    # Step logic (pure Python)
    # -----------------------------
    def step(self, state: TutankhamState, action: int):

        x, y = state.player_x, state.player_y

        if action == Action.LEFT:
            x -= self.consts.SPEED
        elif action == Action.RIGHT:
            x += self.consts.SPEED
        elif action == Action.UP:
            y -= self.consts.SPEED
        elif action == Action.DOWN:
            y += self.consts.SPEED

        # Clip bounds
        x = max(0, min(x, self.consts.WIDTH - 1))
        y = max(0, min(y, self.consts.HEIGHT - 1))

        state = TutankhamState(player_x=x, player_y=y)

        reward = 0.0
        done = False
        info = None

        return state, state, reward, done, info

    # -----------------------------
    # Rendering
    # -----------------------------
    def render(self, state: TutankhamState) -> np.ndarray:
        return self.renderer.render(state)

    # -----------------------------
    # Action & Observation Space
    # -----------------------------
    def action_space(self):
        return Discrete(len(self.action_set))

    def observation_space(self):
        return Box(
            low=0,
            high=max(self.consts.WIDTH, self.consts.HEIGHT),
            shape=(2,),
            dtype=np.int32,
        )
