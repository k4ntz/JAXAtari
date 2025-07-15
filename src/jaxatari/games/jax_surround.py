# src/jaxatari/games/jax_surround.py
import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple, Tuple

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import JAXGameRenderer
import jaxatari.spaces as spaces

class SurroundConstants(NamedTuple):
    GRID_WIDTH: int = 20
    GRID_HEIGHT: int = 20
    SCREEN_SIZE: Tuple[int, int] = (160, 210)
    PLAYER_COLOR: Tuple[int, int, int] = (255, 255, 255)
    BACKGROUND_COLOR: Tuple[int, int, int] = (0, 0, 0)

class SurroundState(NamedTuple):
    p1_pos: jnp.ndarray          # shape (2,)
    p2_pos: jnp.ndarray          # shape (2,)
    p1_trail: jnp.ndarray        # shape (GRID_WIDTH, GRID_HEIGHT)
    p2_trail: jnp.ndarray
    terminated: jnp.ndarray  # shape (1,)

class JaxSurroundEnvironment(JaxEnvironment):
    def __init__(self, constants: 'SurroundConstants'):
        super().__init__(constants)
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(constants.GRID_HEIGHT, constants.GRID_WIDTH, 3),
            dtype=jnp.uint8
        )
        self.renderer = JAXGameRenderer(constants.SCREEN_SIZE)

    @partial(jax.jit, static_argnames='self')
    def step(self, action: Action):
        # Implement the logic for taking a step in the game
        pass

    @partial(jax.jit, static_argnames='self')
    def reset(self):
        # Reset the game state
        pass

    @partial(jax.jit, static_argnames='self')
    def render(self):
        # Render the current game state
        pass


class JaxSurround(JaxEnvironment[SurroundState, ...]):
    action_set = (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT)

    @partial(jax.jit, static_argnums=0)
    def step_env(self, state: SurroundState, actions):
        # actions -> (a1, a2)
        offsets = jnp.array([[0, -1], [0, 1], [-1, 0], [1, 0]])
        new_p1 = state.p1_pos + offsets[actions[0]]
        new_p2 = state.p2_pos + offsets[actions[1]]

        p1_trail = state.p1_trail.at[tuple(state.p1_pos)].set(1)
        p2_trail = state.p2_trail.at[tuple(state.p2_pos)].set(1)

        # collision detection
        hit = (
            (new_p1 == new_p2).all() |
            p1_trail[tuple(new_p1)] | p2_trail[tuple(new_p1)] |
            p1_trail[tuple(new_p2)] | p2_trail[tuple(new_p2)]
        )
        terminated = jnp.where(hit, 1, 0)

        next_state = SurroundState(new_p1, new_p2, p1_trail, p2_trail, terminated)
        return next_state
    
class SurroundObservation(NamedTuple):
    grid: jnp.ndarray  # e.g., 0 empty, 1 player1 trail, 2 player2 trail

def _compute_observation(self, state: SurroundState) -> SurroundObservation:
    grid = jnp.zeros((self.consts.GRID_WIDTH, self.consts.GRID_HEIGHT), dtype=jnp.int32)
    grid = grid.at[state.p1_trail == 1].set(1)
    grid = grid.at[state.p2_trail == 1].set(2)
    grid = grid.at[tuple(state.p1_pos)].set(1)
    grid = grid.at[tuple(state.p2_pos)].set(2)
    return SurroundObservation(grid)

def _reward(self, state: SurroundState) -> jnp.ndarray:
    # Simple reward: +1 if opponent crashes, 0 otherwise
    return jnp.where(state.terminated, jnp.array([1.0, -1.0]), 0.0)

class SurroundRenderer(JAXGameRenderer):
    def __init__(self, consts: SurroundConstants):
        super().__init__()
        import pygame
        self.consts = consts
        self.scale = 8
        self.screen = pygame.display.set_mode(
            (consts.GRID_WIDTH * self.scale, consts.GRID_HEIGHT * self.scale))

    def render(self, state: SurroundState):
        import pygame
        self.screen.fill(self.consts.BACKGROUND_COLOR)
        for y in range(self.consts.GRID_HEIGHT):
            for x in range(self.consts.GRID_WIDTH):
                if state.p1_trail[x, y]:
                    pygame.draw.rect(
                        self.screen, self.consts.PLAYER_COLOR,
                        (x * self.scale, y * self.scale, self.scale, self.scale))
                if state.p2_trail[x, y]:
                    pygame.draw.rect(
                        self.screen, (200, 100, 100),
                        (x * self.scale, y * self.scale, self.scale, self.scale))
        pygame.display.flip()


