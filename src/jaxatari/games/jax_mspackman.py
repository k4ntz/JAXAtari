from functools import partial
from typing import NamedTuple, Tuple

import chex
import jax
import jax.numpy as jnp

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils


class EntityPosition(NamedTuple):
    x: chex.Array
    y: chex.Array
    width: chex.Array
    height: chex.Array


class MsPackmanObservation(NamedTuple):
    pacman: EntityPosition
    ghosts: chex.Array
    pellets: chex.Array
    power_timer: chex.Array
    lives: chex.Array


class MsPackmanInfo(NamedTuple):
    time: chex.Array
    pellets_remaining: chex.Array
    lives: chex.Array


class MsPackmanConstants(NamedTuple):
    screen_width: int = 190
    screen_height: int = 210
    cell_size: int = 10
    grid_width: int = 19
    grid_height: int = 21
    num_ghosts: int = 2
    pellet_reward: int = 1
    power_pellet_reward: int = 5
    ghost_reward: int = 10
    collision_penalty: int = -5
    frightened_duration: int = 60
    initial_lives: int = 3
    max_steps: int = 2000
    background_color: Tuple[int, int, int] = (0, 0, 0)
    wall_color: Tuple[int, int, int] = (33, 33, 255)
    pellet_color: Tuple[int, int, int] = (255, 184, 151)
    power_pellet_color: Tuple[int, int, int] = (255, 255, 255)
    pacman_color: Tuple[int, int, int] = (255, 255, 0)
    ghost_colors: Tuple[Tuple[int, int, int], ...] = (
        (255, 0, 0),
        (255, 184, 222),
        (0, 255, 255),
        (255, 128, 0),
    )
    maze_layout: Tuple[str, ...] = (
        "###################",
        "#P..#.....#.....o.#",
        "#.#.#.###.#.###.#.#",
        "#.#.#.###.#.###.#.#",
        "#.#.#.....#.....#.#",
        "#.#.#####.#.#####.#",
        "#...#.#####.#...#.#",
        "#.#.....###.....#.#",
        "#.###.#.....#.###.#",
        "#.....#.###.#.....#",
        "###.###.#.#.###..##",
        "....#...GG..#.##...",
        "###.###.#.#.###..##",
        "#.....#.###.#.....#",
        "#.###.#.....#.###.#",
        "#.#.....###.....#.#",
        "#.#.#####.#.#####.#",
        "#.#.#.....#.....#.#",
        "#.#.#.###.#.###.#.#",
        "#o..#.....#.....o.#",
        "###################",
    )


class MsPackmanState(NamedTuple):
    pacman_x: chex.Array
    pacman_y: chex.Array
    ghost_positions: chex.Array
    pellets: chex.Array
    power_timer: chex.Array
    score: chex.Array
    time: chex.Array
    lives: chex.Array
    pellets_remaining: chex.Array
    game_over: chex.Array


def _parse_layout(layout: Tuple[str, ...], expected_ghosts: int):
    wall_rows = []
    pellet_rows = []
    pacman_spawn = None
    ghost_positions: list[Tuple[int, int]] = []

    for y, row in enumerate(layout):
        wall_row = []
        pellet_row = []
        for x, char in enumerate(row):
            if char == "#":
                wall_row.append(1)
                pellet_row.append(0)
            elif char == ".":
                wall_row.append(0)
                pellet_row.append(1)
            elif char == "o":
                wall_row.append(0)
                pellet_row.append(2)
            elif char == "P":
                pacman_spawn = (x, y)
                wall_row.append(0)
                pellet_row.append(0)
            elif char == "G":
                ghost_positions.append((x, y))
                wall_row.append(0)
                pellet_row.append(0)
            else:
                wall_row.append(0)
                pellet_row.append(0)
        wall_rows.append(wall_row)
        pellet_rows.append(pellet_row)

    if pacman_spawn is None:
        pacman_spawn = (1, 1)

    if not ghost_positions:
        ghost_positions.append(pacman_spawn)

    while len(ghost_positions) < expected_ghosts:
        ghost_positions.append(ghost_positions[-1])

    ghost_positions = ghost_positions[:expected_ghosts]
    pellet_count = sum(1 for row in pellet_rows for cell in row if cell > 0)

    return (
        jnp.array(wall_rows, dtype=jnp.int32),
        jnp.array(pellet_rows, dtype=jnp.int32),
        jnp.array(pacman_spawn, dtype=jnp.int32),
        jnp.array(ghost_positions, dtype=jnp.int32),
        jnp.array(pellet_count, dtype=jnp.int32),
    )


class JaxMsPackman(JaxEnvironment[MsPackmanState, MsPackmanObservation, MsPackmanInfo, MsPackmanConstants]):
    def __init__(self, consts: MsPackmanConstants = None, reward_funcs: list[callable] = None):
        consts = consts or MsPackmanConstants()
        super().__init__(consts)
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs

        (
            self.wall_grid,
            self.initial_pellets,
            self.pacman_spawn,
            self.ghost_spawn_positions,
            self.initial_pellet_count,
        ) = _parse_layout(self.consts.maze_layout, self.consts.num_ghosts)

        self._action_deltas = jnp.array(
            [
                [0, 0],   # NOOP
                [0, 0],   # FIRE treated as NOOP
                [0, -1],  # UP
                [1, 0],   # RIGHT
                [-1, 0],  # LEFT
                [0, 1],   # DOWN
                [1, -1],  # UPRIGHT
                [-1, -1], # UPLEFT
                [1, 1],   # DOWNRIGHT
                [-1, 1],  # DOWNLEFT
                [0, -1],  # UPFIRE
                [1, 0],   # RIGHTFIRE
                [-1, 0],  # LEFTFIRE
                [0, 1],   # DOWNFIRE
                [1, -1],  # UPRIGHTFIRE
                [-1, -1], # UPLEFTFIRE
                [1, 1],   # DOWNRIGHTFIRE
                [-1, 1],  # DOWNLEFTFIRE
            ],
            dtype=jnp.int32,
        )

        self.state = self.reset()[1]
        self.renderer = MsPackmanRenderer(self.consts, self.wall_grid, self.initial_pellets)

    def reset(self, key: jax.random.PRNGKey = None) -> Tuple[MsPackmanObservation, MsPackmanState]:
        state = MsPackmanState(
            pacman_x=self.pacman_spawn[0],
            pacman_y=self.pacman_spawn[1],
            ghost_positions=self.ghost_spawn_positions,
            pellets=self.initial_pellets,
            power_timer=jnp.array(0, dtype=jnp.int32),
            score=jnp.array(0, dtype=jnp.int32),
            time=jnp.array(0, dtype=jnp.int32),
            lives=jnp.array(self.consts.initial_lives, dtype=jnp.int32),
            pellets_remaining=self.initial_pellet_count,
            game_over=jnp.array(False, dtype=jnp.bool_),
        )

        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: MsPackmanState, action: int) -> Tuple[MsPackmanObservation, MsPackmanState, float, bool, MsPackmanInfo]:
        action = jnp.asarray(action, dtype=jnp.int32)
        action_idx = jnp.clip(action, 0, self._action_deltas.shape[0] - 1)
        delta = self._action_deltas[action_idx]

        tentative_x = jnp.mod(state.pacman_x + delta[0], self.consts.grid_width)
        tentative_y = jnp.clip(state.pacman_y + delta[1], 0, self.consts.grid_height - 1)

        hit_wall = self.wall_grid[tentative_y, tentative_x] == 1
        pacman_x = jnp.where(hit_wall, state.pacman_x, tentative_x)
        pacman_y = jnp.where(hit_wall, state.pacman_y, tentative_y)

        pellet_value = state.pellets[pacman_y, pacman_x]
        ate_pellet = pellet_value > 0
        ate_power = pellet_value == 2

        pellets = state.pellets.at[pacman_y, pacman_x].set(jnp.where(ate_pellet, 0, pellet_value))

        pellet_reward = jnp.where(
            ate_power,
            self.consts.power_pellet_reward,
            jnp.where(ate_pellet, self.consts.pellet_reward, 0),
        )
        score = state.score + pellet_reward
        pellets_remaining = jnp.maximum(
            state.pellets_remaining - ate_pellet.astype(jnp.int32),
            0,
        )

        power_timer = jnp.where(
            ate_power,
            jnp.array(self.consts.frightened_duration, dtype=jnp.int32),
            jnp.maximum(state.power_timer - 1, 0),
        )
        frightened = power_timer > 0

        ghost_positions = self._move_ghosts(state.ghost_positions, pacman_x, pacman_y, frightened)

        ghost_overlap = jnp.logical_and(
            ghost_positions[:, 0] == pacman_x,
            ghost_positions[:, 1] == pacman_y,
        )
        ghosts_eaten = jnp.logical_and(ghost_overlap, frightened)
        ghost_bonus = jnp.sum(ghosts_eaten.astype(jnp.int32)) * self.consts.ghost_reward
        score = score + ghost_bonus

        ghost_positions = jnp.where(
            jnp.broadcast_to(ghosts_eaten[:, None], ghost_positions.shape),
            self.ghost_spawn_positions,
            ghost_positions,
        )

        any_collision = jnp.any(ghost_overlap)
        hit_without_power = jnp.logical_and(any_collision, jnp.logical_not(frightened))

        score = score + jnp.where(hit_without_power, self.consts.collision_penalty, 0)
        lives = state.lives - hit_without_power.astype(jnp.int32)

        pacman_x = jax.lax.select(hit_without_power, self.pacman_spawn[0], pacman_x)
        pacman_y = jax.lax.select(hit_without_power, self.pacman_spawn[1], pacman_y)
        ghost_positions = jax.lax.select(hit_without_power, self.ghost_spawn_positions, ghost_positions)
        power_timer = jax.lax.select(hit_without_power, jnp.array(0, dtype=jnp.int32), power_timer)

        time = state.time + 1

        game_over = jnp.logical_or(
            state.game_over,
            jnp.logical_or(
                lives <= 0,
                jnp.logical_or(pellets_remaining <= 0, time >= self.consts.max_steps),
            ),
        )

        new_state = MsPackmanState(
            pacman_x=pacman_x,
            pacman_y=pacman_y,
            ghost_positions=ghost_positions,
            pellets=pellets,
            power_timer=power_timer,
            score=score,
            time=time,
            lives=lives,
            pellets_remaining=pellets_remaining,
            game_over=game_over,
        )

        done = self._get_done(new_state)
        env_reward = self._get_reward(state, new_state)
        obs = self._get_observation(new_state)
        info = self._get_info(new_state)

        return obs, new_state, env_reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _move_ghosts(
        self,
        ghost_positions: chex.Array,
        pacman_x: chex.Array,
        pacman_y: chex.Array,
        frightened: chex.Array,
    ) -> chex.Array:
        gx = ghost_positions[:, 0]
        gy = ghost_positions[:, 1]
        dx = pacman_x - gx
        dy = pacman_y - gy

        dx_sign = jnp.sign(dx)
        dy_sign = jnp.sign(dy)

        dx_sign = jnp.where(frightened, -dx_sign, dx_sign)
        dy_sign = jnp.where(frightened, -dy_sign, dy_sign)

        prefer_x = jnp.abs(dx) >= jnp.abs(dy)
        primary_dx = jnp.where(prefer_x, dx_sign, 0)
        primary_dy = jnp.where(prefer_x, 0, dy_sign)
        secondary_dx = jnp.where(prefer_x, 0, dx_sign)
        secondary_dy = jnp.where(prefer_x, dy_sign, 0)

        cand1_x = jnp.mod(gx + primary_dx, self.consts.grid_width)
        cand1_y = jnp.clip(gy + primary_dy, 0, self.consts.grid_height - 1)
        blocked1 = self.wall_grid[cand1_y, cand1_x] == 1

        cand2_x = jnp.mod(gx + secondary_dx, self.consts.grid_width)
        cand2_y = jnp.clip(gy + secondary_dy, 0, self.consts.grid_height - 1)
        blocked2 = self.wall_grid[cand2_y, cand2_x] == 1

        use_second = jnp.logical_and(blocked1, jnp.logical_not(blocked2))
        stay = jnp.logical_and(blocked1, blocked2)

        new_x = jnp.where(stay, gx, jnp.where(use_second, cand2_x, cand1_x))
        new_y = jnp.where(stay, gy, jnp.where(use_second, cand2_y, cand1_y))

        return jnp.stack([new_x, new_y], axis=1).astype(jnp.int32)

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: MsPackmanState) -> MsPackmanObservation:
        pacman = EntityPosition(
            x=state.pacman_x,
            y=state.pacman_y,
            width=jnp.array(1, dtype=jnp.int32),
            height=jnp.array(1, dtype=jnp.int32),
        )

        ghost_dims = jnp.ones((self.consts.num_ghosts, 2), dtype=jnp.int32)
        ghosts = jnp.concatenate([state.ghost_positions, ghost_dims], axis=1)

        return MsPackmanObservation(
            pacman=pacman,
            ghosts=ghosts,
            pellets=state.pellets,
            power_timer=state.power_timer,
            lives=state.lives,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: MsPackmanState) -> MsPackmanInfo:
        return MsPackmanInfo(
            time=state.time,
            pellets_remaining=state.pellets_remaining,
            lives=state.lives,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: MsPackmanState, state: MsPackmanState):
        return state.score - previous_state.score

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: MsPackmanState) -> bool:
        return state.game_over

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(18)

    def observation_space(self) -> spaces.Dict:
        return spaces.Dict({
            "pacman": spaces.Dict({
                "x": spaces.Box(low=0, high=self.consts.grid_width - 1, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=self.consts.grid_height - 1, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
            }),
            "ghosts": spaces.Box(
                low=0,
                high=max(self.consts.grid_width, self.consts.grid_height),
                shape=(self.consts.num_ghosts, 4),
                dtype=jnp.int32,
            ),
            "pellets": spaces.Box(
                low=0,
                high=2,
                shape=(self.consts.grid_height, self.consts.grid_width),
                dtype=jnp.int32,
            ),
            "power_timer": spaces.Box(low=0, high=self.consts.frightened_duration, shape=(), dtype=jnp.int32),
            "lives": spaces.Box(low=0, high=self.consts.initial_lives, shape=(), dtype=jnp.int32),
        })

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.consts.screen_height, self.consts.screen_width, 3),
            dtype=jnp.uint8,
        )

    def render(self, state: MsPackmanState) -> jnp.ndarray:
        return self.renderer.render(state)

    def obs_to_flat_array(self, obs: MsPackmanObservation) -> jnp.ndarray:
        pacman_flat = jnp.array([obs.pacman.x, obs.pacman.y, obs.pacman.width, obs.pacman.height], dtype=jnp.int32)
        ghosts_flat = obs.ghosts.reshape(-1)
        pellets_flat = obs.pellets.reshape(-1)
        extras = jnp.array([obs.power_timer, obs.lives], dtype=jnp.int32)
        return jnp.concatenate([pacman_flat, ghosts_flat, pellets_flat, extras]).astype(jnp.int32)


class MsPackmanRenderer(JAXGameRenderer):
    def __init__(
        self,
        consts: MsPackmanConstants = None,
        wall_grid: jnp.ndarray = None,
        pellet_template: jnp.ndarray = None,
    ):
        super().__init__()
        self.consts = consts or MsPackmanConstants()
        self.config = render_utils.RendererConfig(
            game_dimensions=(self.consts.screen_height, self.consts.screen_width),
            channels=3,
        )
        if wall_grid is None:
            wall_grid = jnp.zeros((self.consts.grid_height, self.consts.grid_width), dtype=jnp.int32)
        self.wall_grid = jnp.asarray(wall_grid, dtype=jnp.int32)
        self.offset_x = (self.consts.screen_width - self.consts.grid_width * self.consts.cell_size) // 2
        self.offset_y = (self.consts.screen_height - self.consts.grid_height * self.consts.cell_size) // 2

        self.background_color = jnp.asarray(self.consts.background_color, dtype=jnp.uint8)
        self.wall_color = jnp.asarray(self.consts.wall_color, dtype=jnp.uint8)
        self.pellet_color = jnp.asarray(self.consts.pellet_color, dtype=jnp.uint8)
        self.power_pellet_color = jnp.asarray(self.consts.power_pellet_color, dtype=jnp.uint8)
        self.pacman_color = jnp.asarray(self.consts.pacman_color, dtype=jnp.uint8)
        self.ghost_colors = jnp.asarray(self.consts.ghost_colors, dtype=jnp.uint8)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: MsPackmanState) -> jnp.ndarray:
        cell = self.consts.cell_size

        base_shape = (self.consts.grid_height, self.consts.grid_width, 3)
        wall_mask = (self.wall_grid == 1)[..., None]
        background_layer = jnp.ones(base_shape, dtype=jnp.uint8) * self.background_color
        wall_layer = jnp.ones(base_shape, dtype=jnp.uint8) * self.wall_color
        grid = jnp.where(wall_mask, wall_layer, background_layer)

        pellet_tiles = (state.pellets == 1)[..., None]
        power_tiles = (state.pellets == 2)[..., None]

        grid_pixels = jnp.repeat(grid, cell, axis=0)
        grid_pixels = jnp.repeat(grid_pixels, cell, axis=1)

        pellet_pixels = jnp.repeat(pellet_tiles, cell, axis=0)
        pellet_pixels = jnp.repeat(pellet_pixels, cell, axis=1)
        power_pixels = jnp.repeat(power_tiles, cell, axis=0)
        power_pixels = jnp.repeat(power_pixels, cell, axis=1)

        row_idx = jnp.arange(grid_pixels.shape[0]) % cell
        col_idx = jnp.arange(grid_pixels.shape[1]) % cell
        row_idx = row_idx[:, None]
        col_idx = col_idx[None, :]

        pellet_half = jnp.maximum(cell // 6, 1)
        power_half = jnp.maximum(cell // 4, 2)

        pellet_center = jnp.logical_and(
            jnp.abs(row_idx - cell // 2) < pellet_half,
            jnp.abs(col_idx - cell // 2) < pellet_half,
        )[..., None]
        power_center = jnp.logical_and(
            jnp.abs(row_idx - cell // 2) < power_half,
            jnp.abs(col_idx - cell // 2) < power_half,
        )[..., None]

        pellet_pixels = jnp.logical_and(pellet_pixels, pellet_center)
        power_pixels = jnp.logical_and(power_pixels, power_center)

        pellet_layer_px = jnp.ones_like(grid_pixels) * self.pellet_color
        power_layer_px = jnp.ones_like(grid_pixels) * self.power_pellet_color
        grid_pixels = jnp.where(pellet_pixels, pellet_layer_px, grid_pixels)
        grid_pixels = jnp.where(power_pixels, power_layer_px, grid_pixels)

        canvas = jnp.ones((self.consts.screen_height, self.consts.screen_width, 3), dtype=jnp.uint8) * self.background_color

        canvas = canvas.at[
            self.offset_y:self.offset_y + grid_pixels.shape[0],
            self.offset_x:self.offset_x + grid_pixels.shape[1],
        ].set(grid_pixels)

        pac_block = jnp.ones((cell, cell, 3), dtype=jnp.uint8) * self.pacman_color
        pac_px = self.offset_x + state.pacman_x * cell
        pac_py = self.offset_y + state.pacman_y * cell
        canvas = jax.lax.dynamic_update_slice(canvas, pac_block, (pac_py, pac_px, 0))

        color_indices = jnp.mod(jnp.arange(self.consts.num_ghosts, dtype=jnp.int32), self.ghost_colors.shape[0])
        ghost_palette = self.ghost_colors[color_indices]
        ghost_blocks = jnp.ones((self.consts.num_ghosts, cell, cell, 3), dtype=jnp.uint8) * ghost_palette[:, None, None, :]
        ghost_positions = state.ghost_positions

        def draw_ghost(img, inputs):
            idx, pos = inputs
            block = ghost_blocks[idx]
            gx = self.offset_x + pos[0] * cell
            gy = self.offset_y + pos[1] * cell
            img = jax.lax.dynamic_update_slice(img, block, (gy, gx, 0))
            return img, None

        canvas, _ = jax.lax.scan(
            draw_ghost,
            canvas,
            (jnp.arange(self.consts.num_ghosts, dtype=jnp.int32), ghost_positions),
        )

        return canvas






















import time
import pygame
import jax
import jax.random as jr
from jaxatari.games.jax_mspackman import JaxMsPackman
from jaxatari.environment import JAXAtariAction as Action

UPSCALE = 4
FPS = 8

def main():
    pygame.init()
    env = JaxMsPackman()
    render = jax.jit(env.render)
    step = jax.jit(env.step)

    obs, state = env.reset()
    frame = render(state)
    h, w = frame.shape[:2]
    window = pygame.display.set_mode((w * UPSCALE, h * UPSCALE))
    clock = pygame.time.Clock()

    rng = jr.PRNGKey(0)
    running = True
    action = Action.NOOP

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action = Action.UP
        elif keys[pygame.K_DOWN]:
            action = Action.DOWN
        elif keys[pygame.K_LEFT]:
            action = Action.LEFT
        elif keys[pygame.K_RIGHT]:
            action = Action.RIGHT
        else:
            action = Action.NOOP

        obs, state, reward, done, info = step(state, action)
        if done:
            obs, state = env.reset()

        frame = render(state)
        surf = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
        surf = pygame.transform.scale(surf, (w * UPSCALE, h * UPSCALE))
        window.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()
