# Jax Pacman implementation
# Group: Sooraj Rathore, Kadir Özen

from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
import chex
import pygame
from jax import random
import random as pyrandom
from jaxatari.environment import JaxEnvironment

# Ghost state constants
CHASE = 0
FRIGHTENED = 1  # The Pacman will not be caught by the Ghosts. It can hit ghosts and increase score

# Start positions
PACMAN_START_POS = jnp.array([9, 9], dtype=jnp.int32)
GHOST_HOME_POS = jnp.array([9, 5], dtype=jnp.int32)

# GRID_WIDTH and GRID_HEIGHT are defined in the environment
GRID_WIDTH = 19
GRID_HEIGHT = 11

CELL_SIZE = 20
TOP_OFFSET = 40  # pixels reserved for score & lives

WALL = 1
PACMAN = 2
GHOST = 3
POWER_PELLET = 4
PELLET = 5
EMPTY = 0

# Level configuration (1 to 10)
LEVELS = [
    {"ghost_move_interval": 3, "power_time": 60, "ghost_count": 2},  # L1 easy
    {"ghost_move_interval": 3, "power_time": 55, "ghost_count": 2},  # L2
    {"ghost_move_interval": 2, "power_time": 50, "ghost_count": 3},  # L3
    {"ghost_move_interval": 2, "power_time": 45, "ghost_count": 3},  # L4
    {"ghost_move_interval": 2, "power_time": 40, "ghost_count": 4},  # L5
    {"ghost_move_interval": 2, "power_time": 35, "ghost_count": 4},  # L6
    {"ghost_move_interval": 1, "power_time": 30, "ghost_count": 4},  # L7
    {"ghost_move_interval": 1, "power_time": 25, "ghost_count": 4},  # L8
    {"ghost_move_interval": 1, "power_time": 20, "ghost_count": 4},  # L9
    {"ghost_move_interval": 1, "power_time": 15, "ghost_count": 4},  # L10 hard
]

MAX_LEVEL = len(LEVELS)
MAX_GHOSTS = 4  # design bound for splitting PRNG safely


def generate_maze(width, height):
    """Generating a maze (list of lists) using recursive backtracker.
    It uses Python's `random` (pyrandom) shuffle so this runs outside JAX safely.
    It returns the maze as a list of lists (WALL/EMPTY ints)."""
    maze = [[WALL for _ in range(width)] for _ in range(height)]

    def carve(x, y):
        dirs = [(2, 0), (-2, 0), (0, 2), (0, -2)]
        # Using the plain Python RNG shuffle (pyrandom)
        pyrandom.shuffle(dirs)
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 1 <= nx < width - 1 and 1 <= ny < height - 1 and maze[ny][nx] == WALL:
                maze[ny - dy // 2][nx - dx // 2] = EMPTY
                maze[ny][nx] = EMPTY
                carve(nx, ny)

    # Carving from (1,1)
    maze[1][1] = EMPTY
    carve(1, 1)
    return maze  # note: returns Python list of lists; callers convert to jnp.array if needed


def create_random_map(width, height, wall_density=0.2):
    """
    Creating a random map (converted to jnp.array) using generate_maze.
    Currently, wall_density is not used by the carve generator, left as an argument
    for future extension (e.g., adding random additional walls).
    """
    maze = generate_maze(width, height)

    # Adding pellets in empty spaces
    for y in range(height):
        for x in range(width):
            if maze[y][x] == EMPTY:
                maze[y][x] = PELLET

    # Adding power pellets in corners of empty cells
    corners = [(1, 1), (1, height - 2), (width - 2, 1), (width - 2, height - 2)]
    for cx, cy in corners:
        if maze[cy][cx] != WALL:
            maze[cy][cx] = POWER_PELLET
    # Pac-Man start position
    maze[1][1] = PACMAN

    # Placing 2 ghosts far from Pac-Man
    maze[height - 2][width - 2] = GHOST
    maze[height - 2][1] = GHOST

    return jnp.array(maze, dtype=jnp.int32)


# Simplified 19x11 Pacman map
# 1 = wall, 0 = empty/path
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


@jax.jit
def get_valid_moves(pos: chex.Array, maze: chex.Array) -> Tuple[chex.Array, chex.Array]:
    """Returns candidate directions and a boolean mask of valid moves.
    Positions are [x, y]; maze indexed as [y, x]."""
    directions = DIRECTIONS  # (4,2)

    def is_valid(direction):
        next_pos = pos + direction  # [x,y]
        x, y = next_pos[0], next_pos[1]
        h, w = maze.shape[0], maze.shape[1]
        in_bounds = (0 <= x) & (x < w) & (0 <= y) & (y < h)
        # Only open cells are valid (0 = open)
        return jax.lax.cond(
            in_bounds,
            lambda _: (maze[y, x] == 0),
            lambda _: False,
            operand=None,
        )

    valids = jax.vmap(is_valid)(directions)
    return directions, valids


# Ghost move helpers with bounds checks and safe fallback to staying still
@jax.jit
def ghost_chase_step(ghost_pos, target_pos, maze, key):
    directions = jnp.array([[0, -1], [0, 1], [-1, 0], [1, 0]], dtype=jnp.int32)
    new_positions = ghost_pos + directions  # (4,2)

    xs = new_positions[:, 0]
    ys = new_positions[:, 1]
    h = maze.shape[0]
    w = maze.shape[1]

    in_bounds = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    # Treating out-of-bounds as walls
    cell_vals = jnp.where(in_bounds, maze[ys, xs], jnp.array(WALL, dtype=maze.dtype))
    valid_moves = cell_vals != WALL

    distances = jnp.where(
        valid_moves,
        jnp.sum((new_positions - target_pos) ** 2, axis=1),
        jnp.inf,
    )

    sorted_indices = jnp.argsort(distances)

    # 30% chance to pick second-best instead of best. Influences the difficulty of the game.
    pick_second = random.uniform(key) < 0.3
    chosen_idx = jax.lax.select(pick_second, sorted_indices[1], sorted_indices[0])

    chosen_valid = valid_moves[chosen_idx]
    chosen_pos = jax.lax.cond(chosen_valid, lambda _: new_positions[chosen_idx], lambda _: ghost_pos, operand=None)
    return chosen_pos


@jax.jit
def ghost_frightened_step(ghost_pos, maze, key):
    directions = jnp.array([[0, -1], [0, 1], [-1, 0], [1, 0]], dtype=jnp.int32)
    new_positions = ghost_pos + directions

    xs = new_positions[:, 0]
    ys = new_positions[:, 1]
    h = maze.shape[0]
    w = maze.shape[1]

    in_bounds = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    cell_vals = jnp.where(in_bounds, maze[ys, xs], jnp.array(WALL, dtype=maze.dtype))
    valid_moves = cell_vals != WALL

    # Assigns equal random values to valid moves, -inf to invalid
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


class PacmanState(NamedTuple):
    pacman_pos: chex.Array       # [x, y]
    pacman_dir: chex.Array       # [dx, dy]
    ghost_positions: chex.Array  # (N_ghosts, 2)
    ghost_dirs: chex.Array       # (N_ghosts, 2)
    ghost_states: chex.Array     # (N_ghosts,) 0=CHASE, 1=FRIGHTENED
    pellets: chex.Array          # 2D pellet grid
    power_pellets: chex.Array    # 2D power pellet grid
    score: chex.Array            # float32
    step_count: chex.Array       # int32
    game_over: chex.Array        # bool
    power_mode_timer: chex.Array # int32
    lives: chex.Array            # int32
    rng_key: chex.Array          # RNG key threaded in state


class PacmanObservation(NamedTuple):
    grid: chex.Array  # 2D array showing layout of walls, pellets, pacman, ghosts

class PacmanInfo(NamedTuple):
    score: chex.Array
    done: chex.Array


@jax.jit
def move_entity(pos, direction, grid):
    """Move entity if next cell is not a wall. Positions [x,y]."""
    next_pos = pos + direction
    x, y = next_pos[0], next_pos[1]
    h, w = grid.shape[0], grid.shape[1]

    can_move = jax.lax.cond(
        (0 <= x) & (x < w) & (0 <= y) & (y < h),
        lambda _: (grid[y, x] != 1),
        lambda _: False,
        operand=None,
    )
    return jax.lax.cond(can_move, lambda _: next_pos, lambda _: pos, operand=None)


@jax.jit
def get_observation_fn(state: PacmanState, maze: chex.Array) -> PacmanObservation:
    """
        Constructs the current observation for the Pac-Man environment.
        This function gathers the relevant game state into a structured
        observation dictionary suitable for agents or rendering code.
        It is intended to be pure (no side effects) so it can be safely
        used inside JAX transformations such as `jit` or `vmap`.

    """

    # Copying to avoid mutating input.
    grid = maze

    # Place actors first; pellets will only draw on empty cells (grid==0), so actors stay visible.
    grid = grid.at[state.pacman_pos[1], state.pacman_pos[0]].set(PACMAN)
    grid = grid.at[state.ghost_positions[:, 1], state.ghost_positions[:, 0]].set(GHOST)

    pellet_mask = (state.pellets > 0) & (grid == 0)
    grid = jnp.where(pellet_mask, jnp.array(PELLET, grid.dtype), grid)

    power_mask = (state.power_pellets > 0) & (grid == 0)
    grid = jnp.where(power_mask, jnp.array(POWER_PELLET, grid.dtype), grid)

    return PacmanObservation(grid=grid)


@jax.jit
def step_fn(state: PacmanState,
            action: chex.Array,
            maze: chex.Array,
            ghost_move_interval: int,
            power_time_ticks: int):
    """Pure jitted step function. All environment-config is passed explicitly.
    Returns: obs, new_state, reward, done, info
    """
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

    # Ghost movement (cycle blinky/pinky/clyde)
    n_ghosts = state.ghost_positions.shape[0]
    ghost_types = jnp.arange(n_ghosts, dtype=jnp.int32) % 3

    # Thread RNG: splitting a small array of keys and use only the first n_ghosts keys.
    # We split into (n_ghosts + 2) keys to leave a key for the next step too.
    keys = random.split(state.rng_key, n_ghosts + 2)
    step_key = keys[0]
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

    # Collisions: if on the same tile. During power mode, Pac-Man eats the ghost; otherwise he loses a life.
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

            return jax.lax.cond(power_mode_timer > jnp.array(0, dtype=jnp.int32), eat_case, death_case, operand=None)

        return jax.lax.cond(same, on_collide, lambda _: (pac_pos, g_positions, score_val, lives_val), operand=None)

    pacman_pos, ghost_positions, score, lives = jax.lax.fori_loop(
        0, ghost_positions.shape[0], collide_body,
        (new_pos, ghost_positions, score, state.lives)
    )

    game_over = jnp.logical_or(state.game_over, lives <= 0)

    new_state = PacmanState(
        pacman_pos=pacman_pos,
        pacman_dir=new_dir,
        ghost_positions=ghost_positions,
        ghost_dirs=state.ghost_dirs,
        ghost_states=state.ghost_states,
        pellets=pellets,
        power_pellets=power_pellets,
        score=score,
        step_count=state.step_count + 1,
        game_over=game_over,
        power_mode_timer=power_mode_timer,
        lives=lives,
        rng_key=next_key,
    )

    obs = get_observation_fn(new_state, maze)
    reward = jax.lax.select(has_pellet, jnp.array(10.0, jnp.float32), jnp.array(0.0, jnp.float32))
    done = game_over
    info = PacmanInfo(score=score, done=done)
    return obs, new_state, reward, done, info


class PacmanConstants(NamedTuple):
    level: int
    ghost_move_interval: int
    power_time_ticks: int
    ghost_count: int
    maze: chex.Array
    pellets: chex.Array
    power_pellets: chex.Array


class JaxPacman(JaxEnvironment[PacmanState, PacmanObservation, PacmanInfo, PacmanConstants]):

    def __init__(self):
        super().__init__(consts=None)  # constants will be set in reset()
        self.frame_stack_size = 1
        self.action_set = jnp.arange(4)  # UP, DOWN, LEFT, RIGHT

    def _generate_level_maze(self, level: int):
        """
        Generates a maze layout for the given level.
        Uses a static maze for easy levels, and procedural maps for higher ones.
        """
        if level <= 3:
            maze = jnp.array(maze_layout, dtype=jnp.int32)
        else:
            wall_density = 0.25 + (level - 3) * 0.05
            maze = create_random_map(GRID_WIDTH, GRID_HEIGHT, wall_density=wall_density)

        # Pellets and power pellets
        pellets = (maze == 0).astype(jnp.int32)
        power = jnp.zeros_like(pellets)

        # Placing power pellets at 4 corners (if not a wall)
        for (py, px) in [(1, 1), (1, GRID_WIDTH - 2),
                         (GRID_HEIGHT - 2, 1), (GRID_HEIGHT - 2, GRID_WIDTH - 2)]:
            if maze[py, px] == 0:
                power = power.at[py, px].set(1)
                pellets = pellets.at[py, px].set(0)

        return maze, pellets, power

    def reset(
            self,
            key: chex.PRNGKey,
            level: int = 1,
            keep_score_lives: bool = False,
            carry_score: float = 0.0,
            carry_lives: int = 3,
    ):
        # Clamping level and fetching config
        level = max(1, min(MAX_LEVEL, int(level)))
        cfg = LEVELS[level - 1]

        # Building maze and pellet layout
        maze, pellets, power = self._generate_level_maze(level)

        # Building constants object
        self.consts = PacmanConstants(
            level=level,
            ghost_move_interval=cfg["ghost_move_interval"],
            power_time_ticks=cfg["power_time"],
            ghost_count=cfg["ghost_count"],
            maze=maze,
            pellets=pellets,
            power_pellets=power,
        )

        # Choosing spawn positions from free cells
        free_yx = jnp.argwhere(maze == 0)
        pac_yx = free_yx[0]
        pacman_pos = jnp.array([pac_yx[1], pac_yx[0]], dtype=jnp.int32)

        g_yx = free_yx[1:1 + self.consts.ghost_count]
        ghost_positions = jnp.stack([g_yx[:, 1], g_yx[:, 0]], axis=1).astype(jnp.int32)

        ghost_dirs = jnp.zeros_like(ghost_positions)
        ghost_states = jnp.full((ghost_positions.shape[0],), CHASE, dtype=jnp.int32)

        score = jnp.array(carry_score if keep_score_lives else 0.0, jnp.float32)
        lives = jnp.array(carry_lives if keep_score_lives else 3, dtype=jnp.int32)

        # RNG
        key, sk = random.split(key)

        state = PacmanState(
            pacman_pos=pacman_pos,
            pacman_dir=jnp.array([0, 0], dtype=jnp.int32),
            ghost_positions=ghost_positions,
            ghost_dirs=ghost_dirs,
            ghost_states=ghost_states,
            pellets=self.consts.pellets,
            power_pellets=self.consts.power_pellets,
            score=score,
            step_count=jnp.array(0, dtype=jnp.int32),
            game_over=jnp.array(False),
            power_mode_timer=jnp.array(0, dtype=jnp.int32),
            lives=lives,
            rng_key=sk,
        )
        obs = get_observation_fn(state, maze)
        return obs, state

    def step(self, state, action):
        return step_fn(
            state,
            jnp.array(action),
            self.consts.maze,
            self.consts.ghost_move_interval,
            self.consts.power_time_ticks,
        )



class PacmanRenderer:
    def __init__(self, screen, font):
        self.screen = screen
        self.font = font

    def render(self, obs, state, total_reward, level):
        self.screen.fill((0, 0, 0))

        # Score & Level
        score_surf = self.font.render(f"Score: {int(total_reward)}", True, (255, 255, 255))
        level_surf = self.font.render(f"Level: {level}", True, (255, 255, 255))
        self.screen.blit(score_surf, (10, 10))
        self.screen.blit(level_surf, (10 + score_surf.get_width() + 20, 10))

        # Lives
        lives_start_x = 10 + score_surf.get_width() + 20 + level_surf.get_width() + 20
        lives_y = 20
        for i in range(int(state.lives)):
            center = (lives_start_x + i * (CELL_SIZE + 5), lives_y)
            pygame.draw.circle(self.screen, (255, 255, 0), center, CELL_SIZE // 2)

        # Grid
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                cell = int(obs.grid[y, x])
                rect = pygame.Rect(x * CELL_SIZE, TOP_OFFSET + y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if cell == WALL:
                    pygame.draw.rect(self.screen, (0, 0, 255), rect)  # Wall
                elif cell == PACMAN:
                    pygame.draw.circle(self.screen, (255, 255, 0), rect.center, CELL_SIZE // 2)  # Pacman
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
        # Step environment
        self.key, subkey = random.split(self.key)
        self.obs, self.state, reward, done, info = self.env.step(self.state, jnp.array(action))
        self.total_reward += float(reward)

        # Checking pellets left
        pellets_left = int(jnp.sum(self.state.pellets)) + int(jnp.sum(self.state.power_pellets))

        level_up = False
        game_won = False

        if pellets_left == 0:
            self.level += 1
            if self.level > MAX_LEVEL:
                game_won = True
            else:
                # score & lives
                carry_score = float(self.state.score)
                carry_lives = int(self.state.lives)
                self.obs, self.state = self.env.reset(self.key,
                                                      level=self.level,
                                                      keep_score_lives=True,
                                                      carry_score=carry_score,
                                                      carry_lives=carry_lives)
                level_up = True

        return self.obs, self.state, reward, done, info, level_up, game_won



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
        # Input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: action = 0
                elif event.key == pygame.K_DOWN: action = 1
                elif event.key == pygame.K_LEFT: action = 2
                elif event.key == pygame.K_RIGHT: action = 3

        # Game Step
        obs, state, reward, done, info, level_up, game_won = manager.step(action)

        if game_won:
            print("🎉 You beat all levels! Final score:", int(manager.total_reward))
            pygame.time.wait(1500)
            break

        if level_up:
            continue  # Skipping stale frame, go to next loop

        # Render
        renderer.render(obs, state, manager.total_reward, manager.level)
        clock.tick(10)

        # Game Over
        if bool(done):
            print("Game Over! Final score:", int(manager.total_reward))
            pygame.time.wait(1500)
            running = False

    pygame.quit()




if __name__ == "__main__":
    main()
