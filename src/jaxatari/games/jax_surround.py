import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple, Tuple, Optional

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import JAXGameRenderer
import jaxatari.spaces as spaces


class SurroundConstants(NamedTuple):
    """Parameters defining the Surround grid."""

    GRID_WIDTH: int = 20
    GRID_HEIGHT: int = 20
    SCREEN_SIZE: Tuple[int, int] = (160, 210)
    PLAYER_COLOR: Tuple[int, int, int] = (255, 255, 255)
    BACKGROUND_COLOR: Tuple[int, int, int] = (0, 0, 0)


class SurroundState(NamedTuple):
    """Immutable game state container."""

    p1_pos: jnp.ndarray  # (x, y)
    p2_pos: jnp.ndarray  # (x, y)
    p1_trail: jnp.ndarray  # (GRID_WIDTH, GRID_HEIGHT)
    p2_trail: jnp.ndarray  # (GRID_WIDTH, GRID_HEIGHT)
    terminated: jnp.ndarray  # () bool
    time: jnp.ndarray  # step counter


class SurroundObservation(NamedTuple):
    """Observation returned to the agent."""

    grid: jnp.ndarray  # (GRID_WIDTH, GRID_HEIGHT) int32


class SurroundInfo(NamedTuple):
    """Additional environment information."""

    time: jnp.ndarray


class SurroundRenderer(JAXGameRenderer):
    """Very small dummy renderer used for tests."""

    def __init__(self, consts: Optional[SurroundConstants] = None):
        consts = consts or SurroundConstants()
        super().__init__(consts)
        self.consts = consts

    def render(self, state: SurroundState) -> jnp.ndarray:  # pragma: no cover - visual
        # Produce a simple RGB array where player trails are drawn.
        img = jnp.zeros((self.consts.GRID_HEIGHT, self.consts.GRID_WIDTH, 3), dtype=jnp.uint8)
        img = img.at[state.p1_trail == 1].set(jnp.array(self.consts.PLAYER_COLOR, dtype=jnp.uint8))
        img = img.at[state.p2_trail == 1].set(jnp.array([200, 100, 100], dtype=jnp.uint8))
        img = img.at[tuple(state.p1_pos[::-1])].set(jnp.array(self.consts.PLAYER_COLOR, dtype=jnp.uint8))
        img = img.at[tuple(state.p2_pos[::-1])].set(jnp.array([200, 100, 100], dtype=jnp.uint8))
        return img


class JaxSurround(
    JaxEnvironment[SurroundState, SurroundObservation, SurroundInfo, SurroundConstants]
):
    """A very small two player Surround implementation."""

    def __init__(self, consts: Optional[SurroundConstants] = None):
        consts = consts or SurroundConstants()
        super().__init__(consts)
        self.renderer = SurroundRenderer(self.consts)
        self.action_set = [
            Action.NOOP,
            Action.FIRE,
            Action.UP,
            Action.RIGHT,
            Action.LEFT,
            Action.DOWN,
        ]

    def reset(self, key: Optional[jax.random.PRNGKey] = None) -> Tuple[SurroundObservation, SurroundState]:
        del key
        p1_start = jnp.array([self.consts.GRID_WIDTH // 4, self.consts.GRID_HEIGHT // 2], dtype=jnp.int32)
        p2_start = jnp.array([3 * self.consts.GRID_WIDTH // 4, self.consts.GRID_HEIGHT // 2], dtype=jnp.int32)
        p1_trail = jnp.zeros((self.consts.GRID_WIDTH, self.consts.GRID_HEIGHT), dtype=jnp.int32)
        p2_trail = jnp.zeros_like(p1_trail)
        state = SurroundState(p1_start, p2_start, p1_trail, p2_trail, jnp.array(0, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32))
        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, state: SurroundState, actions: jnp.ndarray
    ) -> Tuple[SurroundObservation, SurroundState, jnp.ndarray, bool, SurroundInfo]:
        """Takes a step for both agents.

        Parameters
        ----------
        state : SurroundState
            Current environment state.
        actions : jnp.ndarray
            Array of shape ``(2,)`` containing the actions for ``first_0`` and
            ``second_0`` respectively.
        """

        offsets = jnp.array(
            [
                [0, 0],  # NOOP
                [0, 0],  # FIRE -> no-op
                [0, -1],  # UP
                [1, 0],  # RIGHT
                [-1, 0],  # LEFT
                [0, 1],  # DOWN
            ],
            dtype=jnp.int32,
        )

        offset_p1 = offsets[actions[0]]
        offset_p2 = offsets[actions[1]]
        new_p1 = state.p1_pos + offset_p1
        new_p2 = state.p2_pos + offset_p2

        def clip_pos(pos):
            return jnp.clip(pos, jnp.array([0, 0]), jnp.array([self.consts.GRID_WIDTH - 1, self.consts.GRID_HEIGHT - 1]))

        hit_p1_wall = jnp.logical_or(
            jnp.any(new_p1 < 0),
            jnp.any(new_p1 >= jnp.array([self.consts.GRID_WIDTH, self.consts.GRID_HEIGHT])),
        )
        hit_p2_wall = jnp.logical_or(
            jnp.any(new_p2 < 0),
            jnp.any(new_p2 >= jnp.array([self.consts.GRID_WIDTH, self.consts.GRID_HEIGHT])),
        )

        new_p1 = clip_pos(new_p1)
        new_p2 = clip_pos(new_p2)

        p1_trail = state.p1_trail.at[tuple(state.p1_pos)].set(1)
        p2_trail = state.p2_trail.at[tuple(state.p2_pos)].set(1)

        hit_p1_trail = jnp.logical_or(p1_trail[tuple(new_p1)], p2_trail[tuple(new_p1)])
        hit_p2_trail = jnp.logical_or(p1_trail[tuple(new_p2)], p2_trail[tuple(new_p2)])

        head_on = jnp.all(new_p1 == new_p2)

        p1_hit = jnp.logical_or(hit_p1_wall, jnp.logical_or(hit_p1_trail, head_on))
        p2_hit = jnp.logical_or(hit_p2_wall, jnp.logical_or(hit_p2_trail, head_on))

        terminated = jnp.logical_or(p1_hit, p2_hit)

        next_state = SurroundState(
            new_p1,
            new_p2,
            p1_trail,
            p2_trail,
            terminated.astype(jnp.int32),
            state.time + 1,
        )

        reward_p1 = jnp.where(p1_hit, -1.0, jnp.where(p2_hit, 1.0, 0.0))
        reward_p2 = jnp.where(p2_hit, -1.0, jnp.where(p1_hit, 1.0, 0.0))
        reward = jnp.array([reward_p1, reward_p2])
        obs = self._get_observation(next_state)
        done = self._get_done(next_state)
        info = self._get_info(next_state)
        return obs, next_state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: SurroundState) -> SurroundObservation:
        grid = jnp.zeros((self.consts.GRID_WIDTH, self.consts.GRID_HEIGHT), dtype=jnp.int32)
        grid = jnp.where(state.p1_trail == 1, 1, grid)
        grid = jnp.where(state.p2_trail == 1, 2, grid)
        grid = grid.at[tuple(state.p1_pos)].set(1)
        grid = grid.at[tuple(state.p2_pos)].set(2)
        return SurroundObservation(grid)

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: SurroundState) -> SurroundInfo:
        return SurroundInfo(time=state.time)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: SurroundState, state: SurroundState) -> jnp.ndarray:
        del previous_state
        hit_p1 = state.terminated & (state.p1_pos == state.p2_pos).all()
        reward_p1 = jnp.where(hit_p1, -1.0, jnp.where(state.terminated, 1.0, 0.0))
        reward_p2 = jnp.where(state.terminated & ~hit_p1, -1.0, jnp.where(hit_p1, 1.0, 0.0))
        return jnp.array([reward_p1, reward_p2])

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: SurroundState) -> jnp.ndarray:
        return state.terminated.astype(jnp.bool_)

    def action_space(self) -> spaces.Tuple:
        """Returns the joint action space for both agents."""
        return spaces.Tuple(
            [
                spaces.Discrete(len(self.action_set)),
                spaces.Discrete(len(self.action_set)),
            ]
        )

    def observation_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=2,
            shape=(self.consts.GRID_WIDTH, self.consts.GRID_HEIGHT),
            dtype=jnp.int32,
        )

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.consts.GRID_HEIGHT, self.consts.GRID_WIDTH, 3),
            dtype=jnp.uint8,
        )

    def render(self, state: SurroundState) -> jnp.ndarray:
        return self.renderer.render(state)

    def obs_to_flat_array(self, obs: SurroundObservation) -> jnp.ndarray:
        return obs.grid.reshape(-1).astype(jnp.int32)


def _pygame_action() -> int:
    """Map pressed keys to a Surround action."""
    import pygame

    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        return Action.UP
    if keys[pygame.K_RIGHT]:
        return Action.RIGHT
    if keys[pygame.K_LEFT]:
        return Action.LEFT
    if keys[pygame.K_DOWN]:
        return Action.DOWN
    if keys[pygame.K_SPACE]:
        return Action.FIRE
    return Action.NOOP


def main() -> None:  # pragma: no cover - visual helper
    """Simple interactive loop to play Surround using pygame."""
    import pygame
    import numpy as np

    env = JaxSurround()
    _obs, state = env.reset()

    scale = 20
    width = env.consts.GRID_WIDTH * scale
    height = env.consts.GRID_HEIGHT * scale

    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("JAX Surround")
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = _pygame_action()

        _obs, state, reward, done, _info = env.step(state, jnp.int32(action))

        frame = np.array(env.render(state))
        surface = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
        surface = pygame.transform.scale(surface, (width, height))
        screen.blit(surface, (0, 0))
        pygame.display.flip()
        clock.tick(10)

        if bool(done):
            _obs, state = env.reset()

    pygame.quit()


if __name__ == "__main__":  # pragma: no cover - manual play
    main()
