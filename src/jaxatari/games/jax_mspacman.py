from functools import partial
from typing import NamedTuple, Tuple
import os

import chex
import jax
import jax.numpy as jnp
import numpy as np

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils


class EntityPosition(NamedTuple):
    x: chex.Array
    y: chex.Array
    width: chex.Array
    height: chex.Array


class MsPacmanObservation(NamedTuple):
    pacman: EntityPosition
    ghosts: chex.Array
    pellets: chex.Array
    power_timer: chex.Array
    lives: chex.Array


class MsPacmanInfo(NamedTuple):
    time: chex.Array
    pellets_remaining: chex.Array
    lives: chex.Array


class MsPacmanConstants(NamedTuple):
    screen_width: int = 200
    screen_height: int = 210
    cell_size: int = 10
    grid_width: int = 20
    grid_height: int = 21
    num_ghosts: int = 2
    pellet_reward: int = 1
    power_pellet_reward: int = 5
    ghost_reward: int = 10
    collision_penalty: int = -5
    frightened_duration: int = 60
    initial_lives: int = 3
    max_steps: int = 2000
    ghost_spawn_delay: int = 60
    ghost_move_period: int = 6  # ghosts move every N frames
    player_move_period: int = 3  # pacman moves every N frames
    ghost_move_period: int = 2  # ghosts move every N frames
    background_color: Tuple[int, int, int] = (0, 0, 0)
    wall_color: Tuple[int, int, int] = (33, 33, 255)
    pellet_color: Tuple[int, int, int] = (255, 184, 151)
    power_pellet_color: Tuple[int, int, int] = (255, 255, 255)
    button_color: Tuple[int, int, int] = (255, 105, 180)
    pacman_color: Tuple[int, int, int] = (255, 255, 0)
    button_power_duration: int = 40
    ghost_colors: Tuple[Tuple[int, int, int], ...] = (
        (255, 0, 0),
        (255, 184, 222),
        (0, 255, 255),
        (255, 128, 0),
    )
    maze_layout: Tuple[str, ...] = (
        "####################",
        "#....#........#....#",
        "#P##.#.######.#.##P#",
        "#..................#",
        "#.#.##.####.#.##.#.#",
        "#.#..............#.#",
        "##...#.#####.#.B.#.#",
        "##......###..B..#.##",
        "#####.#.....#.###.##",
        "#.....#.###.#.....##",
        "###.###.#.#.###..###",
        "#.B.#..BGG..#.##.B##",
        "###.###.#.#.###..###",
        "#.....#.###.#.....##",
        "#.###.#.....#.###.##",
        "#.#.....###.....#.##",
        "#.#.#####.#.#####.##",
        "#.#.#.....#.....#.##",
        "#.#.#.###.#.###.#.##",
        "#o..#..B..#..B..o.##",
        "####################",
    )


class MsPacmanState(NamedTuple):
    pacman_x: chex.Array
    pacman_y: chex.Array
    direction: chex.Array
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
    button_rows = []
    pacman_spawn = None
    ghost_positions: list[Tuple[int, int]] = []

    for y, row in enumerate(layout):
        wall_row = []
        pellet_row = []
        button_row = []
        for x, char in enumerate(row):
            if char == "#":
                wall_row.append(1)
                pellet_row.append(0)
                button_row.append(0)
            elif char == ".":
                wall_row.append(0)
                pellet_row.append(1)
                button_row.append(0)
            elif char == "o":
                wall_row.append(0)
                pellet_row.append(2)
                button_row.append(0)
            elif char == "B":
                wall_row.append(0)
                pellet_row.append(0)
                button_row.append(1)
            elif char == "P":
                pacman_spawn = (x, y)
                wall_row.append(0)
                pellet_row.append(0)
                button_row.append(0)
            elif char == "G":
                ghost_positions.append((x, y))
                wall_row.append(0)
                pellet_row.append(0)
                button_row.append(0)
            else:
                wall_row.append(0)
                pellet_row.append(0)
                button_row.append(0)
        wall_rows.append(wall_row)
        pellet_rows.append(pellet_row)
        button_rows.append(button_row)

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
        jnp.array(button_rows, dtype=jnp.int32),
    )


class JaxMsPacman(JaxEnvironment[MsPacmanState, MsPacmanObservation, MsPacmanInfo, MsPacmanConstants]):
    def __init__(self, consts: MsPacmanConstants = None, reward_funcs: list[callable] = None):
        consts = consts or MsPacmanConstants()
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
            self.button_grid,
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
        self.renderer = MsPacmanRenderer(
            self.consts,
            self.wall_grid,
            self.initial_pellets,
            self.button_grid,
        )

    def reset(self, key: jax.random.PRNGKey = None) -> Tuple[MsPacmanObservation, MsPacmanState]:
        state = MsPacmanState(
            pacman_x=self.pacman_spawn[0],
            pacman_y=self.pacman_spawn[1],
            direction=jnp.array([0, 0], dtype=jnp.int32),
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
    def step(self, state: MsPacmanState, action: int) -> Tuple[MsPacmanObservation, MsPacmanState, float, bool, MsPacmanInfo]:
        action = jnp.asarray(action, dtype=jnp.int32)
        action_idx = jnp.clip(action, 0, self._action_deltas.shape[0] - 1)
        delta = self._action_deltas[action_idx]

        has_new_direction = jnp.any(delta != 0)
        direction = jnp.where(has_new_direction, delta, state.direction)
        can_move = (state.time % jnp.maximum(self.consts.player_move_period, 1)) == 0

        tentative_x = jnp.mod(state.pacman_x + direction[0], self.consts.grid_width)
        tentative_y = jnp.clip(state.pacman_y + direction[1], 0, self.consts.grid_height - 1)

        hit_wall = self.wall_grid[tentative_y, tentative_x] == 1
        pacman_x = jnp.where(hit_wall, state.pacman_x, tentative_x)
        pacman_y = jnp.where(hit_wall, state.pacman_y, tentative_y)
        zero_dir = jnp.array([0, 0], dtype=jnp.int32)
        direction = jnp.where(hit_wall, zero_dir, direction)
        pacman_x = jnp.where(can_move, pacman_x, state.pacman_x)
        pacman_y = jnp.where(can_move, pacman_y, state.pacman_y)

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

        decayed_timer = jnp.maximum(state.power_timer - 1, 0)
        power_timer = jnp.where(
            ate_power,
            jnp.array(self.consts.frightened_duration, dtype=jnp.int32),
            decayed_timer,
        )
        on_button = self.button_grid[pacman_y, pacman_x] == 1
        power_timer = jnp.where(
            on_button,
            jnp.array(self.consts.button_power_duration, dtype=jnp.int32),
            power_timer,
        )
        frightened = power_timer > 0

        ghost_positions = self._move_ghosts(state.ghost_positions, pacman_x, pacman_y, frightened, state.time)

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
        direction = jax.lax.select(hit_without_power, zero_dir, direction)

        time = state.time + 1

        game_over = jnp.logical_or(
            state.game_over,
            jnp.logical_or(
                lives <= 0,
                jnp.logical_or(pellets_remaining <= 0, time >= self.consts.max_steps),
            ),
        )

        new_state = MsPacmanState(
            pacman_x=pacman_x,
            pacman_y=pacman_y,
            direction=direction,
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
        time: chex.Array,
    ) -> chex.Array:
        idx = jnp.arange(self.consts.num_ghosts, dtype=jnp.int32)
        active = time >= (idx * self.consts.ghost_spawn_delay)
        can_move = (time % jnp.maximum(self.consts.ghost_move_period, 1)) == 0
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
        new_x = jnp.where(can_move, new_x, gx)
        new_y = jnp.where(can_move, new_y, gy)

        # Add more randomness so ghosts are less perfect: 70% chance to take a random step (including staying put).
        key = jax.random.PRNGKey(time)
        keys = jax.random.split(key, self.consts.num_ghosts + 1)
        noise_mask = jax.random.bernoulli(keys[0], p=0.7, shape=(self.consts.num_ghosts,))
        rand_keys = keys[1:]
        rand_idx = jax.vmap(lambda k: jax.random.randint(k, (), 0, 5))(rand_keys)
        dir_table = jnp.array([[1, 0], [-1, 0], [0, 1], [0, -1], [0, 0]], dtype=jnp.int32)
        rand_delta = dir_table[rand_idx]
        rand_x = jnp.mod(gx + rand_delta[:, 0], self.consts.grid_width)
        rand_y = jnp.clip(gy + rand_delta[:, 1], 0, self.consts.grid_height - 1)
        rand_blocked = self.wall_grid[rand_y, rand_x] == 1
        rand_x = jnp.where(rand_blocked, gx, rand_x)
        rand_y = jnp.where(rand_blocked, gy, rand_y)

        final_x = jnp.where(noise_mask, rand_x, new_x)
        final_y = jnp.where(noise_mask, rand_y, new_y)

        # Respect spawn delays: inactive ghosts stay at their spawn positions.
        spawn_x = self.ghost_spawn_positions[:, 0]
        spawn_y = self.ghost_spawn_positions[:, 1]
        final_x = jnp.where(active, final_x, spawn_x)
        final_y = jnp.where(active, final_y, spawn_y)

        return jnp.stack([final_x, final_y], axis=1).astype(jnp.int32)

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: MsPacmanState) -> MsPacmanObservation:
        pacman = EntityPosition(
            x=state.pacman_x,
            y=state.pacman_y,
            width=jnp.array(1, dtype=jnp.int32),
            height=jnp.array(1, dtype=jnp.int32),
        )

        ghost_dims = jnp.ones((self.consts.num_ghosts, 2), dtype=jnp.int32)
        ghosts = jnp.concatenate([state.ghost_positions, ghost_dims], axis=1)

        return MsPacmanObservation(
            pacman=pacman,
            ghosts=ghosts,
            pellets=state.pellets,
            power_timer=state.power_timer,
            lives=state.lives,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: MsPacmanState) -> MsPacmanInfo:
        return MsPacmanInfo(
            time=state.time,
            pellets_remaining=state.pellets_remaining,
            lives=state.lives,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: MsPacmanState, state: MsPacmanState):
        return state.score - previous_state.score

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: MsPacmanState) -> bool:
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

    def render(self, state: MsPacmanState) -> jnp.ndarray:
        return self.renderer.render(state)

    def obs_to_flat_array(self, obs: MsPacmanObservation) -> jnp.ndarray:
        pacman_flat = jnp.array([obs.pacman.x, obs.pacman.y, obs.pacman.width, obs.pacman.height], dtype=jnp.int32)
        ghosts_flat = obs.ghosts.reshape(-1)
        pellets_flat = obs.pellets.reshape(-1)
        extras = jnp.array([obs.power_timer, obs.lives], dtype=jnp.int32)
        return jnp.concatenate([pacman_flat, ghosts_flat, pellets_flat, extras]).astype(jnp.int32)


class MsPacmanRenderer(JAXGameRenderer):
    def __init__(self, consts: MsPacmanConstants = None, wall_grid: jnp.ndarray = None, pellet_template: jnp.ndarray = None, button_grid: jnp.ndarray = None,):
        super().__init__()
        self.consts = consts or MsPacmanConstants()
        self.config = render_utils.RendererConfig(
            game_dimensions=(self.consts.screen_height, self.consts.screen_width),
            channels=3,
        )
        if wall_grid is None:
            wall_grid = jnp.zeros((self.consts.grid_height, self.consts.grid_width), dtype=jnp.int32)
        self.wall_grid = jnp.asarray(wall_grid, dtype=jnp.int32)
        if button_grid is None:
            button_grid = jnp.zeros_like(self.wall_grid)
        self.button_grid = jnp.asarray(button_grid, dtype=jnp.int32)
        self.offset_x = (self.consts.screen_width - self.consts.grid_width * self.consts.cell_size) // 2
        self.offset_y = (self.consts.screen_height - self.consts.grid_height * self.consts.cell_size) // 2

        self.background_color = jnp.asarray(self.consts.background_color, dtype=jnp.uint8)
        self.wall_color = jnp.asarray(self.consts.wall_color, dtype=jnp.uint8)
        self.pellet_color = jnp.asarray(self.consts.pellet_color, dtype=jnp.uint8)
        self.power_pellet_color = jnp.asarray(self.consts.power_pellet_color, dtype=jnp.uint8)
        self.button_color = jnp.asarray(self.consts.button_color, dtype=jnp.uint8)
        self.pacman_color = jnp.asarray(self.consts.pacman_color, dtype=jnp.uint8)
        self.ghost_colors = jnp.asarray(self.consts.ghost_colors, dtype=jnp.uint8)
        self.background_image = self._load_freeway_background()
        self.pac_sprite = self._load_car_sprite()
        self._base_grid_pixels = self._create_base_grid()
        self._base_canvas = self._create_base_canvas(self._base_grid_pixels)

    def _load_freeway_background(self) -> jnp.ndarray | None:
        """Load the Freeway background sprite as a quick visual test, if available."""
        try:
            path = os.path.join(os.path.dirname(__file__), "sprites", "freeway", "background.npy")
            bg_rgba = np.load(path)
            return jnp.asarray(bg_rgba, dtype=jnp.uint8)
        except Exception:
            return None

    def _load_car_sprite(self) -> jnp.ndarray | None:
        """Load the Freeway red car sprite to use as MsPacman."""
        try:
            path = os.path.join(os.path.dirname(__file__), "sprites", "freeway", "car_red.npy")
            car_rgba = np.load(path)
            return jnp.asarray(car_rgba, dtype=jnp.uint8)
        except Exception:
            return None

    def _create_base_grid(self) -> jnp.ndarray:
        """Pre-draw the static maze (background + walls) once."""
        cell = self.consts.cell_size
        base_shape = (self.consts.grid_height, self.consts.grid_width, 3)
        wall_mask = (self.wall_grid == 1)[..., None]
        background_layer = jnp.ones(base_shape, dtype=jnp.uint8) * self.background_color
        wall_layer = jnp.ones(base_shape, dtype=jnp.uint8) * self.wall_color
        grid = jnp.where(wall_mask, wall_layer, background_layer)
        grid_pixels = jnp.repeat(grid, cell, axis=0)
        grid_pixels = jnp.repeat(grid_pixels, cell, axis=1)
        return grid_pixels

    def _create_base_canvas(self, grid_pixels: jnp.ndarray) -> jnp.ndarray:
        """Place the pre-drawn maze onto a background-colored canvas."""
        canvas = jnp.ones((self.consts.screen_height, self.consts.screen_width, 3), dtype=jnp.uint8) * self.background_color
        if self.background_image is not None:
            bg = self.background_image
            h = min(bg.shape[0], canvas.shape[0])
            w = min(bg.shape[1], canvas.shape[1])
            alpha = bg[:h, :w, 3:4].astype(jnp.uint16)
            fg_rgb = bg[:h, :w, :3].astype(jnp.uint16)
            bg_rgb = canvas[:h, :w, :].astype(jnp.uint16)
            blended = ((fg_rgb * alpha) + (bg_rgb * (255 - alpha))) // 255
            canvas = canvas.at[:h, :w, :].set(blended.astype(jnp.uint8))
        return canvas.at[
            self.offset_y:self.offset_y + grid_pixels.shape[0],
            self.offset_x:self.offset_x + grid_pixels.shape[1],
        ].set(grid_pixels)

    def _alpha_blend(self, canvas: jnp.ndarray, patch: jnp.ndarray, top: int, left: int) -> jnp.ndarray:
        """Alpha blend an RGBA patch onto canvas at (top, left)."""
        h, w = patch.shape[0], patch.shape[1]
        region = jax.lax.dynamic_slice(canvas, (top, left, 0), (h, w, 3))
        alpha = patch[:, :, 3:4].astype(jnp.uint16)
        fg_rgb = patch[:, :, :3].astype(jnp.uint16)
        bg_rgb = region.astype(jnp.uint16)
        blended = ((fg_rgb * alpha) + (bg_rgb * (255 - alpha))) // 255
        return jax.lax.dynamic_update_slice(canvas, blended.astype(jnp.uint8), (top, left, 0))

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: MsPacmanState) -> jnp.ndarray:
        cell = self.consts.cell_size

        pellet_tiles = (state.pellets == 1)[..., None]
        power_tiles = (state.pellets == 2)[..., None]
        button_tiles = (self.button_grid == 1)[..., None]

        grid_pixels = self._base_grid_pixels
        pellet_pixels = jnp.repeat(pellet_tiles, cell, axis=0)
        pellet_pixels = jnp.repeat(pellet_pixels, cell, axis=1)
        power_pixels = jnp.repeat(power_tiles, cell, axis=0)
        power_pixels = jnp.repeat(power_pixels, cell, axis=1)
        button_pixels = jnp.repeat(button_tiles, cell, axis=0)
        button_pixels = jnp.repeat(button_pixels, cell, axis=1)

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
        button_center = jnp.logical_and(
            jnp.abs(row_idx - cell // 2) < power_half,
            jnp.logical_or(
                jnp.abs(row_idx - cell // 4) < power_half // 2,
                jnp.abs(col_idx - cell // 4) < power_half // 2,
            ),
        )[..., None]
        button_pixels = jnp.logical_and(button_pixels, button_center)

        pellet_layer_px = jnp.ones_like(grid_pixels) * self.pellet_color
        power_layer_px = jnp.ones_like(grid_pixels) * self.power_pellet_color
        button_layer_px = jnp.ones_like(grid_pixels) * self.button_color
        grid_pixels = jnp.where(pellet_pixels, pellet_layer_px, grid_pixels)
        grid_pixels = jnp.where(power_pixels, power_layer_px, grid_pixels)
        grid_pixels = jnp.where(button_pixels, button_layer_px, grid_pixels)

        canvas = self._base_canvas
        canvas = canvas.at[
            self.offset_y:self.offset_y + grid_pixels.shape[0],
            self.offset_x:self.offset_x + grid_pixels.shape[1],
        ].set(grid_pixels)

        pac_px = self.offset_x + state.pacman_x * cell
        pac_py = self.offset_y + state.pacman_y * cell
        if self.pac_sprite is not None:
            sx = max((cell - self.pac_sprite.shape[1]) // 2, 0)
            sy = max((cell - self.pac_sprite.shape[0]) // 2, 0)
            canvas = self._alpha_blend(canvas, self.pac_sprite, pac_py + sy, pac_px + sx)
        else:
            pac_block = jnp.ones((cell, cell, 3), dtype=jnp.uint8) * self.pacman_color
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
from jaxatari.games.jax_mspacman import JaxMsPacman
from jaxatari.environment import JAXAtariAction as Action

UPSCALE = 4
FPS = 8

def main():
    pygame.init()
    env = JaxMsPacman()
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
