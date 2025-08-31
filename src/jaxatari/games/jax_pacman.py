# JAX Pacman — refactor (jit + spaces)
# Group: Sooraj Rathore, Kadir Özen
#
# Key changes vs your previous version:


from typing import NamedTuple, Tuple, Dict
import jax
import jax.numpy as jnp
import chex
import pygame
from jax import random,lax
import random as pyrandom
from jaxatari.environment import JaxEnvironment
from jaxatari import spaces

# ------------------------------
# Constants & layout
# ------------------------------
CHASE = 0
FRIGHTENED = 1

PACMAN_START_POS = jnp.array([9, 9], dtype=jnp.int32)
GHOST_HOME_POS = jnp.array([9, 5], dtype=jnp.int32)

GRID_WIDTH = 19
GRID_HEIGHT = 11

CELL_SIZE = 20
TOP_OFFSET = 40

EMPTY = 0
WALL = 1
PACMAN = 2
GHOST = 3
POWER_PELLET = 4
PELLET = 5

LEVELS = [
    {"ghost_move_interval": 3, "power_time": 60, "ghost_count": 2},  # L1
    {"ghost_move_interval": 3, "power_time": 55, "ghost_count": 2},  # L2
    {"ghost_move_interval": 2, "power_time": 50, "ghost_count": 3},  # L3
    {"ghost_move_interval": 2, "power_time": 45, "ghost_count": 3},  # L4
    {"ghost_move_interval": 2, "power_time": 40, "ghost_count": 4},  # L5
    {"ghost_move_interval": 2, "power_time": 35, "ghost_count": 4},  # L6
    {"ghost_move_interval": 1, "power_time": 30, "ghost_count": 4},  # L7
    {"ghost_move_interval": 1, "power_time": 25, "ghost_count": 4},  # L8
    {"ghost_move_interval": 1, "power_time": 20, "ghost_count": 4},  # L9
    {"ghost_move_interval": 1, "power_time": 15, "ghost_count": 4},  # L10
]

MAX_LEVEL = len(LEVELS)
MAX_GHOSTS = 4

# Base static maze (19x11). 1 = wall, 0 = empty
maze_layout = jnp.array([
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,1],
    [1,0,1,1,0,1,0,1,1,1,1,1,0,1,0,1,1,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,1,1,0,1,0,1,1,1,1,1,0,1,0,1,1,0,1],
    [1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1],
    [1,1,0,1,1,1,1,1,0,1,0,1,1,1,1,1,0,1,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
], dtype=jnp.int32)


# Directions in [dx, dy]
DIRECTIONS = jnp.array([
    [0, -1],  # UP
    [0, 1],   # DOWN
    [-1, 0],  # LEFT
    [1, 0],   # RIGHT
], dtype=jnp.int32)

# ------------------------------
# Maze generation (host-side only)
# ------------------------------

def generate_maze(width: int, height: int):
    """Recursive backtracker. Host-side only; uses Python RNG."""
    maze = [[WALL for _ in range(width)] for _ in range(height)]

    def carve(x, y):
        dirs = [(2, 0), (-2, 0), (0, 2), (0, -2)]
        pyrandom.shuffle(dirs)
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 1 <= nx < width - 1 and 1 <= ny < height - 1 and maze[ny][nx] == WALL:
                maze[ny - dy // 2][nx - dx // 2] = EMPTY
                maze[ny][nx] = EMPTY
                carve(nx, ny)

    maze[1][1] = EMPTY
    carve(1, 1)
    return maze


def create_random_map(width: int, height: int, wall_density: float = 0.2):
    maze = generate_maze(width, height)
    for y in range(height):
        for x in range(width):
            if maze[y][x] == EMPTY:
                maze[y][x] = PELLET
    corners = [(1, 1), (1, height - 2), (width - 2, 1), (width - 2, height - 2)]
    for cx, cy in corners:
        if maze[cy][cx] != WALL:
            maze[cy][cx] = POWER_PELLET
    maze[1][1] = PACMAN
    maze[height - 2][width - 2] = GHOST
    maze[height - 2][1] = GHOST
    return jnp.array(maze, dtype=jnp.int32)

# ------------------------------
# Helpers
# ------------------------------

@jax.jit
def move_entity(pos, direction, grid):
    next_pos = pos + direction
    x, y = next_pos[0], next_pos[1]
    h, w = grid.shape[0], grid.shape[1]
    can_move = jax.lax.cond(
        (0 <= x) & (x < w) & (0 <= y) & (y < h),
        lambda _: (grid[y, x] != WALL),
        lambda _: False,
        operand=None,
    )
    return jax.lax.cond(can_move, lambda _: next_pos, lambda _: pos, operand=None)

@jax.jit
def grid_to_rgb(grid: chex.Array) -> chex.Array:
    palette = jnp.array([
        [0, 0, 0],      # 0 EMPTY
        [0, 0, 255],    # 1 WALL
        [255, 255, 0],  # 2 PACMAN
        [255, 0, 0],    # 3 GHOST
        [0, 255, 255],  # 4 POWER_PELLET
        [255, 255, 255] # 5 PELLET
    ], dtype=jnp.uint8)
    return palette[grid]

# ------------------------------
# Ghost movement
# ------------------------------

@jax.jit
def ghost_chase_step(ghost_pos, target_pos, maze, key):
    directions = DIRECTIONS
    new_positions = ghost_pos + directions
    xs, ys = new_positions[:, 0], new_positions[:, 1]
    h, w = maze.shape[0], maze.shape[1]
    in_bounds = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    cell_vals = jnp.where(in_bounds, maze[ys, xs], jnp.array(WALL, dtype=maze.dtype))
    valid_moves = cell_vals != WALL
    distances = jnp.where(
        valid_moves,
        jnp.sum((new_positions - target_pos) ** 2, axis=1),
        jnp.inf,
    )
    sorted_indices = jnp.argsort(distances)
    pick_second = random.uniform(key) < 0.6
    chosen_idx = jax.lax.select(pick_second, sorted_indices[1], sorted_indices[0])
    chosen_valid = valid_moves[chosen_idx]
    chosen_pos = jax.lax.cond(chosen_valid, lambda _: new_positions[chosen_idx], lambda _: ghost_pos, operand=None)
    return chosen_pos

@jax.jit
def ghost_frightened_step(ghost_pos, maze, key):
    directions = DIRECTIONS
    new_positions = ghost_pos + directions
    xs, ys = new_positions[:, 0], new_positions[:, 1]
    h, w = maze.shape[0], maze.shape[1]
    in_bounds = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    cell_vals = jnp.where(in_bounds, maze[ys, xs], jnp.array(WALL, dtype=maze.dtype))
    valid_moves = cell_vals != WALL
    rand_vals = random.uniform(key, (4,))
    scores = jnp.where(valid_moves, rand_vals, -jnp.inf)
    best_idx = jnp.argmax(scores)
    best_valid = valid_moves[best_idx]
    chosen_pos = jax.lax.cond(best_valid, lambda _: new_positions[best_idx], lambda _: ghost_pos, operand=None)
    return chosen_pos

@jax.jit
def blinky_move(ghost_pos, pacman_pos, maze, key):
    return ghost_chase_step(ghost_pos, pacman_pos, maze, key)

@jax.jit
def pinky_move(ghost_pos, pacman_pos, pacman_dir, maze, key):
    target = pacman_pos + 4 * pacman_dir
    return ghost_chase_step(ghost_pos, target, maze, key)

@jax.jit
def clyde_move(ghost_pos, pacman_pos, maze, key):
    distance = jnp.linalg.norm((ghost_pos - pacman_pos).astype(jnp.float32))
    return jax.lax.cond(
        distance < 4.0,
        lambda _: ghost_frightened_step(ghost_pos, maze, key),
        lambda _: ghost_chase_step(ghost_pos, pacman_pos, maze, key),
        operand=None,
    )

# ------------------------------
# State / Observation / Info / Consts
# ------------------------------

class PacmanState(NamedTuple):
    pacman_pos: chex.Array
    pacman_dir: chex.Array
    ghost_positions: chex.Array
    ghost_dirs: chex.Array
    ghost_states: chex.Array
    pellets: chex.Array
    power_pellets: chex.Array
    score: chex.Array
    step_count: chex.Array
    game_over: chex.Array
    power_mode_timer: chex.Array
    lives: chex.Array
    rng_key: chex.Array
    ghost_count: chex.Array       # number of active ghosts

class PacmanObservation(NamedTuple):
    grid: chex.Array            # HxW int map
    image: chex.Array           # HxWx3 uint8 (RGB)
    pacman_pos: chex.Array
    ghost_positions: chex.Array
    pellets: chex.Array
    power_pellets: chex.Array

class PacmanInfo(NamedTuple):
    score: chex.Array
    done: chex.Array

class PacmanConstants(NamedTuple):
    level: int
    ghost_move_interval: int
    power_time_ticks: int
    ghost_count: int
    maze: chex.Array
    pellets: chex.Array
    power_pellets: chex.Array

# ------------------------------
# Observation builder (pure)
# ------------------------------

@jax.jit
def get_observation_fn(state: PacmanState, maze: chex.Array) -> PacmanObservation:
    grid = maze
    grid = grid.at[state.pacman_pos[1], state.pacman_pos[0]].set(PACMAN)
    grid = grid.at[state.ghost_positions[:, 1], state.ghost_positions[:, 0]].set(GHOST)
    pellet_mask = (state.pellets > 0) & (grid == EMPTY)
    grid = jnp.where(pellet_mask, jnp.array(PELLET, grid.dtype), grid)
    power_mask = (state.power_pellets > 0) & (grid == EMPTY)
    grid = jnp.where(power_mask, jnp.array(POWER_PELLET, grid.dtype), grid)
    return PacmanObservation(
        grid=grid,
        image=grid_to_rgb(grid),
        pacman_pos=state.pacman_pos,
        ghost_positions=state.ghost_positions,
        pellets=state.pellets,
        power_pellets=state.power_pellets,
    )

# ------------------------------
# Step function (pure jitted)
# ------------------------------

@jax.jit
def step_fn(state: PacmanState, action: chex.Array, maze: chex.Array,
            ghost_move_interval: int, power_time_ticks: int):
    new_dir = DIRECTIONS[action]
    new_pos = move_entity(state.pacman_pos, new_dir, maze)

    has_pellet = state.pellets[new_pos[1], new_pos[0]] > 0
    has_power = state.power_pellets[new_pos[1], new_pos[0]] > 0

    pellets = state.pellets.at[new_pos[1], new_pos[0]].set(0)
    power_pellets = state.power_pellets.at[new_pos[1], new_pos[0]].set(0)

    score = state.score + jax.lax.select(
        has_pellet, jnp.array(10.0, jnp.float32), jnp.array(0.0, jnp.float32)
    )

    power_mode_timer = jax.lax.select(
        has_power,
        jnp.array(power_time_ticks, dtype=jnp.int32),
        jnp.maximum(jnp.array(0, dtype=jnp.int32), state.power_mode_timer - 1),
    )

    n_ghosts = state.ghost_positions.shape[0]
    ghost_types = jnp.arange(n_ghosts, dtype=jnp.int32) % 3

    keys = random.split(state.rng_key, n_ghosts + 2)
    ghost_keys = keys[1:1 + n_ghosts]
    next_key = keys[-1]

    def move_one(pos, gtype, key):
        return jax.lax.switch(
            gtype,
            (
                lambda _: blinky_move(pos, new_pos, maze, key),
                lambda _: pinky_move(pos, new_pos, new_dir, maze, key),
                lambda _: clyde_move(pos, new_pos, maze, key),
            ),
            operand=None,
        )

    ghost_positions = jax.lax.cond(
        (state.step_count % jnp.array(ghost_move_interval, dtype=jnp.int32) == 0),
        lambda _: jax.vmap(move_one)(state.ghost_positions, ghost_types, ghost_keys),
        lambda _: state.ghost_positions,
        operand=None,
    )

    def collide_body(i, carry):
        pac_pos, g_positions, score_val, lives_val = carry
        gpos = g_positions[i]
        same = jnp.all(pac_pos == gpos)

        def on_collide(_):
            def eat_case(_):
                new_g_positions = g_positions.at[i].set(GHOST_HOME_POS)
                return (pac_pos, new_g_positions,
                        score_val + jnp.array(200.0, dtype=jnp.float32),
                        lives_val)

            def death_case(_):
                return (PACMAN_START_POS, g_positions, score_val,
                        lives_val - jnp.array(1, dtype=jnp.int32))

            return jax.lax.cond(power_mode_timer > 0, eat_case, death_case, operand=None)

        return jax.lax.cond(same, on_collide,
                            lambda _: (pac_pos, g_positions, score_val, lives_val),
                            operand=None)

    pacman_pos, ghost_positions, score, lives = jax.lax.fori_loop(
        0, ghost_positions.shape[0], collide_body,
        (new_pos, ghost_positions, score, state.lives)
    )

    game_over = jnp.logical_or(state.game_over, lives <= 0)

    new_state = PacmanState(
        pacman_pos=pacman_pos,
        pacman_dir=new_dir,
        ghost_positions=ghost_positions,
        ghost_dirs=state.ghost_dirs,       # keep old until ghost dirs are updated
        ghost_states=state.ghost_states,   # same for ghost states
        pellets=pellets,
        power_pellets=power_pellets,
        score=score,
        step_count=state.step_count + 1,
        game_over=game_over,
        power_mode_timer=power_mode_timer,
        lives=lives,
        rng_key=next_key,
        ghost_count=state.ghost_count,
    )

    obs = get_observation_fn(new_state, maze)
    reward = jax.lax.select(has_pellet, jnp.array(10.0, jnp.float32), jnp.array(0.0, jnp.float32))
    done = game_over
    info = PacmanInfo(score=score, done=done)
    return obs, new_state, reward, done, info




# ------------------------------
# Level bank (host-side once), jittable reset kernel
# ------------------------------

def _build_level_bank() -> Dict[int, PacmanConstants]:
    bank: Dict[int, PacmanConstants] = {}
    for lvl, cfg in enumerate(LEVELS, start=1):
        if lvl <= 3:
            maze = maze_layout
        else:
            maze = create_random_map(GRID_WIDTH, GRID_HEIGHT, 0.25 + (lvl - 3) * 0.05)
        pellets = (maze == EMPTY).astype(jnp.int32)
        power = jnp.zeros_like(pellets)
        corners = jnp.array([[1, 1], [1, GRID_WIDTH - 2], [GRID_HEIGHT - 2, 1], [GRID_HEIGHT - 2, GRID_WIDTH - 2]], dtype=jnp.int32)
        y, x = corners[:, 0], corners[:, 1]
        is_open = maze[y, x] == EMPTY
        power = power.at[y, x].set(is_open.astype(jnp.int32))
        pellets = pellets.at[y, x].set(0)
        bank[lvl] = PacmanConstants(
            level=lvl,
            ghost_move_interval=cfg["ghost_move_interval"],
            power_time_ticks=cfg["power_time"],
            ghost_count=cfg["ghost_count"],
            maze=maze,
            pellets=pellets,
            power_pellets=power,
        )
    return bank



@jax.jit
def _reset_kernel(consts: PacmanConstants, key: chex.PRNGKey,
                  keep_score_lives: bool, carry_score: float, carry_lives: int):

    H = consts.maze.shape[0]
    W = consts.maze.shape[1]
    HW = H * W

    # mask for EMPTY (0)
    mask = (consts.maze == EMPTY).reshape(-1)              # (H*W,)
    key, k_cells, sk = random.split(key, 3)

    # static-size random scores and masked -inf
    rand_scores = random.uniform(k_cells, (HW,))
    scores = jnp.where(mask, rand_scores, -jnp.inf)

    K_STATIC = 1 + MAX_GHOSTS
    _, flat_idx = jax.lax.top_k(scores, K_STATIC)         # shape (K_STATIC,)
    ys = flat_idx // W
    xs = flat_idx % W

    #pacman_pos = jnp.array([xs[0], ys[0]], dtype=jnp.int32)
    pacman_pos = jnp.array([9, 9], dtype=jnp.int32)

    gxs_full = xs[1:1 + MAX_GHOSTS]
    gys_full = ys[1:1 + MAX_GHOSTS]
    ghost_positions_full = jnp.stack([gxs_full, gys_full], axis=1).astype(jnp.int32)  # (MAX_GHOSTS, 2)

    # deactivating any ghosts beyond consts.ghost_count by placing them at home
    gc = jnp.array(consts.ghost_count, dtype=jnp.int32)  # active count (may be < MAX_GHOSTS)
    idxs = jnp.arange(MAX_GHOSTS, dtype=jnp.int32)
    active_mask = idxs < gc                                   # (MAX_GHOSTS,)

    # Where inactive, setting to GHOST_HOME_POS
    home = jnp.broadcast_to(GHOST_HOME_POS, (MAX_GHOSTS, 2)).astype(jnp.int32)
    ghost_positions = jnp.where(active_mask[:, None], ghost_positions_full, home)
    ghost_dirs = jnp.zeros((MAX_GHOSTS, 2), dtype=jnp.int32)
    ghost_states = jnp.full((MAX_GHOSTS,), CHASE, jnp.int32)
    score = jnp.array(jax.lax.select(keep_score_lives, carry_score, 0.0), jnp.float32)
    lives = jnp.array(jax.lax.select(keep_score_lives, carry_lives, 3), jnp.int32)

    state = PacmanState(
        pacman_pos=pacman_pos,
        pacman_dir=jnp.array([0, 0], dtype=jnp.int32),
        ghost_positions=ghost_positions,
        ghost_dirs=ghost_dirs,
        ghost_states=ghost_states,
        pellets=consts.pellets,
        power_pellets=consts.power_pellets,
        score=score,
        step_count=jnp.array(0, dtype=jnp.int32),
        game_over=jnp.array(False),
        power_mode_timer=jnp.array(0, dtype=jnp.int32),
        lives=lives,
        rng_key=sk,
        ghost_count=gc,
    )
    obs = get_observation_fn(state, consts.maze)
    return obs, state



# ------------------------------
# Environment
# ------------------------------

class JaxPacman(JaxEnvironment[PacmanState, PacmanObservation, PacmanInfo, PacmanConstants]):
    def __init__(self):
        super().__init__(consts=None)
        self._level_bank = _build_level_bank()  # host-side precompute
        self.frame_stack_size = 1
        self._action_set = jnp.arange(4, dtype=jnp.int32)

    # Gym-like spaces for jaxatari
    @property
    def action_space(self):
        return spaces.Discrete(4)

    @property
    def observation_space(self):
        img_box = spaces.Box(low=0, high=255, shape=(GRID_HEIGHT, GRID_WIDTH, 3), dtype=jnp.uint8)
        obj = spaces.Dict({
            "pacman_pos": spaces.Box(low=0, high=max(GRID_HEIGHT, GRID_WIDTH), shape=(2,), dtype=jnp.int32),
            "ghost_positions": spaces.Box(low=0, high=max(GRID_HEIGHT, GRID_WIDTH), shape=(MAX_GHOSTS, 2), dtype=jnp.int32),
            "pellets": spaces.Box(low=0, high=1, shape=(GRID_HEIGHT, GRID_WIDTH), dtype=jnp.int32),
            "power_pellets": spaces.Box(low=0, high=1, shape=(GRID_HEIGHT, GRID_WIDTH), dtype=jnp.int32),
        })
        return spaces.Dict({"image": img_box, "objects": obj})

    @property
    def action_set(self):  # keep old API compatibility
        return self._action_set

    def render(self, obs: PacmanObservation, mode: str = "rgb_array"):
        if mode == "rgb_array":
            return obs.image
        raise NotImplementedError("Only rgb_array supported by env; use PacmanRenderer for pygame display.")

    def reset(self, key: chex.PRNGKey, level: int = 1, keep_score_lives: bool = False,
              carry_score: float = 0.0, carry_lives: int = 3):
        level = int(max(1, min(MAX_LEVEL, int(level))))
        consts = self._level_bank[level]
        self.consts = consts
        return _reset_kernel(consts, key, keep_score_lives, carry_score, carry_lives)

    def step(self, state, action):
        return step_fn(state, jnp.array(action), self.consts.maze,
                       self.consts.ghost_move_interval, self.consts.power_time_ticks)

# ------------------------------
# Renderer & Game loop (unchanged)
# ------------------------------

class PacmanRenderer:
    def __init__(self, screen, font):
        self.screen = screen
        self.font = font

    def render(self, obs: PacmanObservation, state: PacmanState, total_reward, level):
        self.screen.fill((0, 0, 0))
        score_surf = self.font.render(f"Score: {int(total_reward)}", True, (255, 255, 255))
        level_surf = self.font.render(f"Level: {level}", True, (255, 255, 255))
        self.screen.blit(score_surf, (10, 10))
        self.screen.blit(level_surf, (10 + score_surf.get_width() + 20, 10))
        lives_start_x = 10 + score_surf.get_width() + 20 + level_surf.get_width() + 20
        lives_y = 20
        for i in range(int(state.lives)):
            center = (lives_start_x + i * (CELL_SIZE + 5), lives_y)
            pygame.draw.circle(self.screen, (255, 255, 0), center, CELL_SIZE // 2)
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                cell = int(obs.grid[y, x])
                rect = pygame.Rect(x * CELL_SIZE, TOP_OFFSET + y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if cell == WALL:
                    pygame.draw.rect(self.screen, (0, 0, 255), rect)
                elif cell == PACMAN:
                    pygame.draw.circle(self.screen, (255, 255, 0), rect.center, CELL_SIZE // 2)
                elif cell == GHOST:
                    color = (0, 191, 255) if int(state.power_mode_timer) > 0 else (255, 0, 0)
                    pygame.draw.circle(self.screen, color, rect.center, CELL_SIZE // 2)
                elif cell == POWER_PELLET:
                    pygame.draw.circle(self.screen, (0, 255, 255), rect.center, CELL_SIZE // 4)
                elif cell == PELLET:
                    pygame.draw.circle(self.screen, (255, 255, 255), rect.center, CELL_SIZE // 6)
        pygame.display.flip()

class GameManager:
    def __init__(self, env, key):
        self.env = env
        self.key = key
        self.level = 1
        self.total_reward = 0.0
        self.obs, self.state = env.reset(key, level=self.level)

    def step(self, action):
        self.key, subkey = random.split(self.key)
        self.obs, self.state, reward, done, info = self.env.step(self.state, jnp.array(action))
        self.total_reward += float(reward)
        pellets_left = int(jnp.sum(self.state.pellets)) + int(jnp.sum(self.state.power_pellets))
        level_up = False
        game_won = False
        if pellets_left == 0:
            self.level += 1
            if self.level > MAX_LEVEL:
                game_won = True
            else:
                carry_score = float(self.state.score)
                carry_lives = int(self.state.lives)
                self.obs, self.state = self.env.reset(self.key, level=self.level,
                                                      keep_score_lives=True,
                                                      carry_score=carry_score,
                                                      carry_lives=carry_lives)
                level_up = True
        return self.obs, self.state, reward, done, info, level_up, game_won

# ------------------------------
# Main (unchanged)
# ------------------------------

def main():
    pygame.init()
    screen = pygame.display.set_mode((GRID_WIDTH * CELL_SIZE, TOP_OFFSET + GRID_HEIGHT * CELL_SIZE))
    pygame.display.set_caption("Pacman - JAX Edition")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    env = JaxPacman()
    key = random.PRNGKey(0)

    manager = GameManager(env, key)
    renderer = PacmanRenderer(screen, font)

    running = True
    action = 1  # Default DOWN

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: action = 0
                elif event.key == pygame.K_DOWN: action = 1
                elif event.key == pygame.K_LEFT: action = 2
                elif event.key == pygame.K_RIGHT: action = 3

        obs, state, reward, done, info, level_up, game_won = manager.step(action)

        if game_won:
            print("🎉 You beat all levels! Final score:", int(manager.total_reward))
            pygame.time.wait(1500)
            break

        if level_up:
            continue

        renderer.render(obs, state, manager.total_reward, manager.level)
        clock.tick(12)

        if bool(done):
            print("Game Over! Final score:", int(manager.total_reward))
            pygame.time.wait(1500)
            running = False

    pygame.quit()

if __name__ == "__main__":
    main()

