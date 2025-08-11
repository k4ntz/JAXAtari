import os
import jax
import os
import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple, Tuple, Optional

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as jr
import jaxatari.spaces as spaces


class SurroundConstants(NamedTuple):
    """Parameters defining the Surround grid and visuals."""

    # Playfield layout
    GRID_WIDTH: int = 40
    GRID_HEIGHT: int = 24

    # Mapping from grid cells to screen pixels
    CELL_SIZE: Tuple[int, int] = (4, 8)  # (width, height)
    SCREEN_SIZE: Tuple[int, int] = (160, 210)

    # Colors
    P1_TRAIL_COLOR: Tuple[int, int, int] = (255, 221, 51) # Gelb
    P2_TRAIL_COLOR: Tuple[int, int, int] = (221, 51, 136)   # Pink-Magenta
    BACKGROUND_COLOR: Tuple[int, int, int] = (153, 153, 255) # Blau-Lila Hintergrund

    # Starting positions (x, y)
    P1_START_POS: Tuple[int, int] = (5, 12)
    P2_START_POS: Tuple[int, int] = (34, 12)

    # Starting directions
    P1_START_DIR: int = Action.RIGHT
    P2_START_DIR: int = Action.LEFT

    # Border
    BORDER_COLOR = (255, 102, 204)  # Pink wie im Bild
    BORDER_THICKNESS = 10           # Dicke des Randes in Pixeln

    # Rules
    ALLOW_REVERSE: bool = True
    # Maximum number of environment steps before truncation
    MAX_STEPS: int = 1000


class SurroundState(NamedTuple):
    """Immutable game state container."""

    pos0: jnp.ndarray  # (x, y)
    pos1: jnp.ndarray  # (x, y)
    dir0: jnp.ndarray  # () int32
    dir1: jnp.ndarray  # () int32
    trail: jnp.ndarray  # (GRID_WIDTH, GRID_HEIGHT)
    terminated: jnp.ndarray  # () bool
    time: jnp.ndarray  # step counter
    score0: jnp.ndarray  # () int32
    score1: jnp.ndarray  # () int32


class SurroundObservation(NamedTuple):
    """Observation returned to the agent."""

    grid: jnp.ndarray  # (GRID_WIDTH, GRID_HEIGHT) int32
    pos0: jnp.ndarray  # (2,) int32
    pos1: jnp.ndarray  # (2,) int32
    agent_id: jnp.ndarray  # () int32


class SurroundInfo(NamedTuple):
    """Additional environment information."""

    time: jnp.ndarray


@partial(jax.jit, static_argnames=("border_x", "border_y", "grid_w", "grid_h"))
def check_border_collision(
    player_pos: jnp.ndarray,
    border_x: int,
    border_y: int,
    grid_w: int,
    grid_h: int,
) -> jnp.ndarray:
    x, y = player_pos
    return jnp.logical_or(
        jnp.logical_or(x < border_x, x > grid_w - border_x - 1),
        jnp.logical_or(y < border_y, y > grid_h - border_y - 1),
    )


class SurroundRenderer(JAXGameRenderer):
    """Very small dummy renderer used for tests."""

    def __init__(self, consts: Optional[SurroundConstants] = None):
        consts = consts or SurroundConstants()
        super().__init__(consts)
        self.consts = consts

        module_dir = os.path.dirname(os.path.abspath(__file__))
        digit_path = os.path.join(module_dir, "sprites/seaquest/digits/{}" + ".npy")
        digits = jr.load_and_pad_digits(digit_path)
        p1_color = jnp.array(self.consts.P1_TRAIL_COLOR, dtype=jnp.uint8)
        p2_color = jnp.array(self.consts.P2_TRAIL_COLOR, dtype=jnp.uint8)
        self.p1_digits = digits.at[..., :3].set(jnp.where(digits[..., 3:] > 0, p1_color, 0))
        self.p2_digits = digits.at[..., :3].set(jnp.where(digits[..., 3:] > 0, p2_color, 0))

    def render(self, state: SurroundState) -> jnp.ndarray:  # pragma: no cover - visual
        """Render the current game state as a simple RGB image."""
        bg = jnp.array(self.consts.BACKGROUND_COLOR, dtype=jnp.uint8)
        width, height = self.consts.SCREEN_SIZE
        img = jnp.ones((height, width, 3), dtype=jnp.uint8) * bg

        field_h = self.consts.GRID_HEIGHT * self.consts.CELL_SIZE[1]
        y_off = height - field_h
        playfield = jnp.ones((field_h, width, 3), dtype=jnp.uint8) * bg

        def upscale(mask):
            mask = jnp.repeat(mask, self.consts.CELL_SIZE[1], axis=0)
            return jnp.repeat(mask, self.consts.CELL_SIZE[0], axis=1)

        p1_mask = upscale((state.trail == 1).T)[..., None]
        p1_color = jnp.array(self.consts.P1_TRAIL_COLOR, dtype=jnp.uint8)
        playfield = jnp.where(p1_mask, p1_color, playfield)
        p2_mask = upscale((state.trail == 2).T)[..., None]
        p2_color = jnp.array(self.consts.P2_TRAIL_COLOR, dtype=jnp.uint8)
        playfield = jnp.where(p2_mask, p2_color, playfield)

        p1x = state.pos0[0] * self.consts.CELL_SIZE[0]
        p1y = state.pos0[1] * self.consts.CELL_SIZE[1]
        p1_patch = jnp.ones((self.consts.CELL_SIZE[1], self.consts.CELL_SIZE[0], 3), dtype=jnp.uint8) * p1_color
        playfield = jax.lax.dynamic_update_slice(playfield, p1_patch, (p1y, p1x, 0))

        p2x = state.pos1[0] * self.consts.CELL_SIZE[0]
        p2y = state.pos1[1] * self.consts.CELL_SIZE[1]
        p2_patch = jnp.ones((self.consts.CELL_SIZE[1], self.consts.CELL_SIZE[0], 3), dtype=jnp.uint8) * p2_color
        playfield = jax.lax.dynamic_update_slice(playfield, p2_patch, (p2y, p2x, 0))

        img = img.at[y_off:y_off+field_h, :width, :].set(playfield)

        border = self.consts.BORDER_THICKNESS
        border_color = jnp.array(self.consts.BORDER_COLOR, dtype=jnp.uint8)
        img = img.at[:border, :, :].set(border_color)
        img = img.at[-border:, :, :].set(border_color)
        img = img.at[:, :border, :].set(border_color)
        img = img.at[:, -border:, :].set(border_color)

        digit_p1 = jr.get_sprite_frame(self.p1_digits, state.score0)
        digit_p2 = jr.get_sprite_frame(self.p2_digits, state.score1)
        img = jr.render_at(img, 10, 2, digit_p1)
        img = jr.render_at(img, width - 10 - digit_p2.shape[1], 2, digit_p2)
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
        p0_start = jnp.array(self.consts.P1_START_POS, dtype=jnp.int32)
        p1_start = jnp.array(self.consts.P2_START_POS, dtype=jnp.int32)
        grid = jnp.zeros((self.consts.GRID_WIDTH, self.consts.GRID_HEIGHT), dtype=jnp.int32)
        state = SurroundState(
            p0_start,
            p1_start,
            jnp.array(self.consts.P1_START_DIR, dtype=jnp.int32),
            jnp.array(self.consts.P2_START_DIR, dtype=jnp.int32),
            grid,
            jnp.array(0, dtype=jnp.int32),
            jnp.array(0, dtype=jnp.int32),
            jnp.array(0, dtype=jnp.int32),
            jnp.array(0, dtype=jnp.int32),
        )
        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, state: SurroundState, actions: jnp.ndarray | tuple | list
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

        actions = jnp.asarray(actions, dtype=jnp.int32)
        actions = jnp.broadcast_to(actions, (2,))

        is_move0 = jnp.logical_and(actions[0] >= Action.UP, actions[0] <= Action.DOWN)
        is_move1 = jnp.logical_and(actions[1] >= Action.UP, actions[1] <= Action.DOWN)

        def update_dir(curr_dir, action):
            is_move = jnp.logical_and(action >= Action.UP, action <= Action.DOWN)
            candidate = jax.lax.select(is_move, action, curr_dir)
            if not self.consts.ALLOW_REVERSE:
                opp = jnp.array([
                    Action.NOOP,
                    Action.NOOP,
                    Action.DOWN,
                    Action.LEFT,
                    Action.RIGHT,
                    Action.UP,
                ], dtype=jnp.int32)
                candidate = jax.lax.cond(candidate == opp[curr_dir], lambda: curr_dir, lambda: candidate)
            return candidate

        new_dir0 = update_dir(state.dir0, actions[0])
        new_dir1 = update_dir(state.dir1, actions[1])

        offset_p0 = jax.lax.select(is_move0, offsets[new_dir0], jnp.array([0, 0], dtype=jnp.int32))
        offset_p1 = jax.lax.select(is_move1, offsets[new_dir1], jnp.array([0, 0], dtype=jnp.int32))

        new_p0 = state.pos0 + offset_p0
        new_p1 = state.pos1 + offset_p1

        border_x = self.consts.BORDER_THICKNESS // self.consts.CELL_SIZE[0]
        border_y = self.consts.BORDER_THICKNESS // self.consts.CELL_SIZE[1]
        grid_w = self.consts.GRID_WIDTH
        grid_h = self.consts.GRID_HEIGHT

        bounds_min = jnp.array([border_x, border_y])
        bounds_max = jnp.array([grid_w - border_x, grid_h - border_y])
        clip = lambda pos: jnp.clip(pos, bounds_min, bounds_max - 1)

        hit_p0_wall = check_border_collision(new_p0, border_x, border_y, grid_w, grid_h)
        hit_p1_wall = check_border_collision(new_p1, border_x, border_y, grid_w, grid_h)

        safe_p0 = clip(new_p0)
        safe_p1 = clip(new_p1)

        hit_p0_trail = jax.lax.cond(
            hit_p0_wall,
            lambda: False,
            lambda: state.trail[tuple(safe_p0)] != 0,
        )
        hit_p1_trail = jax.lax.cond(
            hit_p1_wall,
            lambda: False,
            lambda: state.trail[tuple(safe_p1)] != 0,
        )

        p0_hit = jnp.logical_or(hit_p0_wall, hit_p0_trail)
        p1_hit = jnp.logical_or(hit_p1_wall, hit_p1_trail)

        grid0 = jax.lax.select(
            is_move0, state.trail.at[tuple(state.pos0)].set(1), state.trail
        )
        grid = jax.lax.select(is_move1, grid0.at[tuple(state.pos1)].set(2), grid0)

        new_p0 = safe_p0
        new_p1 = safe_p1

        terminated = jnp.logical_or(p0_hit, p1_hit)
        time_limit_reached = (state.time + 1) >= self.consts.MAX_STEPS
        terminated = jnp.logical_or(terminated, time_limit_reached)

        new_score0 = state.score0 + jnp.where(p1_hit & ~p0_hit, 1, 0)
        new_score1 = state.score1 + jnp.where(p0_hit & ~p1_hit, 1, 0)
        terminated = jnp.logical_or(
            terminated, jnp.logical_or(new_score0 >= 10, new_score1 >= 10)
        )

        next_state = SurroundState(
            new_p0,
            new_p1,
            new_dir0,
            new_dir1,
            grid,
            terminated.astype(jnp.int32),
            state.time + 1,
            new_score0,
            new_score1,
        )

        reward = self._get_reward(state, next_state)
        obs = self._get_observation(next_state)
        done = self._get_done(next_state)
        info = self._get_info(next_state)
        return obs, next_state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: SurroundState) -> SurroundObservation:
        grid = state.trail
        grid = grid.at[tuple(state.pos0)].set(1)
        grid = grid.at[tuple(state.pos1)].set(2)
        return SurroundObservation(
            grid=grid,
            pos0=state.pos0.astype(jnp.int32),
            pos1=state.pos1.astype(jnp.int32),
            agent_id=jnp.array(0, dtype=jnp.int32),
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: SurroundState) -> SurroundInfo:
        return SurroundInfo(time=state.time)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: SurroundState, state: SurroundState) -> jnp.ndarray:
        previous_diff = previous_state.score0 - previous_state.score1
        diff = state.score0 - state.score1
        return diff - previous_diff

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: SurroundState) -> jnp.ndarray:
        reached_score = jnp.logical_or(state.score0 >= 10, state.score1 >= 10)
        time_exceeded = state.time >= self.consts.MAX_STEPS
        done = jnp.logical_or(state.terminated, reached_score)
        done = jnp.logical_or(done, time_exceeded)
        return done.astype(jnp.bool_)

    def action_space(self) -> spaces.Discrete:
        """Returns the action space for the controllable player."""
        return spaces.Discrete(len(self.action_set))

    def observation_space(self) -> spaces.Dict:
        return spaces.Dict({
            "grid": spaces.Box(
                low=0,
                high=2,
                shape=(self.consts.GRID_WIDTH, self.consts.GRID_HEIGHT),
                dtype=jnp.int32,
            ),
            "pos0": spaces.Box(0, self.consts.GRID_WIDTH, shape=(2,), dtype=jnp.int32),
            "pos1": spaces.Box(0, self.consts.GRID_WIDTH, shape=(2,), dtype=jnp.int32),
            "agent_id": spaces.Discrete(2),
        })

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.consts.SCREEN_SIZE[1], self.consts.SCREEN_SIZE[0], 3),
            dtype=jnp.uint8,
        )

    def render(self, state: SurroundState) -> jnp.ndarray:
        return self.renderer.render(state)

    def obs_to_flat_array(self, obs: SurroundObservation) -> jnp.ndarray:
        flat = [obs.grid.reshape(-1), obs.pos0.reshape(-1), obs.pos1.reshape(-1), jnp.array([obs.agent_id], dtype=jnp.int32)]
        return jnp.concatenate(flat).astype(jnp.int32)


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

        # env.step expects an action for each player. When playing manually we
        # control only the first player, so keep the second player idle.
        joint_action = jnp.array([action, Action.NOOP], dtype=jnp.int32)

        _obs, state, reward, done, _info = env.step(state, joint_action)

        frame = np.array(env.render(state))
        surface = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
        surface = pygame.transform.scale(surface, (width, height))
        screen.blit(surface, (0, 0))

        border = env.consts.BORDER_THICKNESS
        color = env.consts.BORDER_COLOR
        pygame.draw.rect(screen, color, (0, 0, width, border))
        pygame.draw.rect(screen, color, (0, height - border, width, border))
        pygame.draw.rect(screen, color, (0, 0, border, height))
        pygame.draw.rect(screen, color, (width - border, 0, border, height))

        pygame.display.flip()
        clock.tick(5)

        if bool(done):
            _obs, state = env.reset()

    pygame.quit()


if __name__ == "__main__":  # pragma: no cover - manual play
    main()
