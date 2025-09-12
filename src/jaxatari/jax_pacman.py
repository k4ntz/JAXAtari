# pacman_arcade_sprites.py
from typing import NamedTuple, Dict, Tuple
import jax
import jax.numpy as jnp
import chex
from jax import random, lax

# ---- robuste Imports für jaxatari (mit Fallback-Stubs für Standalone-Run) ----
try:
    from jaxatari.environment import JaxEnvironment
    from jaxatari import spaces
except Exception:
    try:
        from src.jaxatari.environment import JaxEnvironment  # type: ignore
        from src.jaxatari import spaces  # type: ignore
    except Exception:
        class _Box:
            def __init__(self, low, high, shape, dtype): ...
        class _Discrete:
            def __init__(self, n): self.n = n
        class _Dict(dict): ...
        class _Spaces:
            Box = _Box; Discrete = _Discrete; Dict = _Dict
        spaces = _Spaces()  # type: ignore
        class JaxEnvironment:  # type: ignore
            def __init__(self, consts=None): self.consts = consts

# ==============================
# Konstanten & Arcade-Layout
# ==============================
G_CHASE  = 0
G_FRIGHT = 1
G_EYES   = 2

TICK_RATE = 10
MODE_DURATIONS = jnp.array([7*TICK_RATE, 20*TICK_RATE, 7*TICK_RATE, 20*TICK_RATE, 5*TICK_RATE, 999_999_999],
                           jnp.int32)

GRID_WIDTH = 28
GRID_HEIGHT = 31

EMPTY = 0
WALL  = 1
PACMAN = 2
GHOST  = 3
POWER_PELLET = 4
PELLET = 5
DOOR   = 6

MAX_GHOSTS = 4
LEVELS = [
    {"ghost_move_base": 2, "power_time": 60, "ghost_count": 4},
    {"ghost_move_base": 2, "power_time": 50, "ghost_count": 4},
    {"ghost_move_base": 2, "power_time": 40, "ghost_count": 4},
    {"ghost_move_base": 1, "power_time": 35, "ghost_count": 4},
    {"ghost_move_base": 1, "power_time": 30, "ghost_count": 4},
]
MAX_LEVEL = len(LEVELS)

PACMAN_START_POS = jnp.array([13, 23], jnp.int32)
GHOST_HOME_POS   = jnp.array([13, 14], jnp.int32)

DIRECTIONS = jnp.array([[0,-1],[0,1],[-1,0],[1,0]], jnp.int32)

BLACK          = jnp.array([0,0,0], jnp.uint8)
PACMAN_YELLOW  = jnp.array([255,255,  0], jnp.uint8)
WALL_BLUE      = jnp.array([  0,  0,255], jnp.uint8)
PELLET_WHITE   = jnp.array([255,255,255], jnp.uint8)
PWR_WHITE      = jnp.array([255,255,255], jnp.uint8)
BLINKY_RED     = jnp.array([255,  0,  0], jnp.uint8)
PINKY_PINK     = jnp.array([255,182,193], jnp.uint8)
INKY_CYAN      = jnp.array([  0,255,255], jnp.uint8)
CLYDE_ORANGE   = jnp.array([255,165,  0], jnp.uint8)
FRIGHT_BLUE    = jnp.array([  0, 28,216], jnp.uint8)
FLASH_WHITE    = jnp.array([255,255,255], jnp.uint8)
DOOR_WHITE     = jnp.array([255,255,255], jnp.uint8)
EYES_WHITE     = jnp.array([255,255,255], jnp.uint8)
PUPIL_BLUE     = jnp.array([ 64, 64,255], jnp.uint8)

FRUIT_POS = jnp.array([13, 17], jnp.int32)
FRUIT_POINTS = jnp.array([100, 300, 500, 700, 1000, 2000, 3000, 5000], jnp.int32)
FRUIT_COLORS = jnp.array([
    [255,  0,  0],
    [255, 64, 64],
    [255,165,  0],
    [  0,255,  0],
    [  0,200,  0],
    [  0,255,255],
    [255,255,  0],
    [255,255,255],
], jnp.uint8)
LEVEL_FRUITS = jnp.array([
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [5, 6],
], jnp.int32)
FRUIT_THRESHOLDS = jnp.array([70, 170], jnp.int32)

_maze_rows = [
    "############################",
    "#............##............#",
    "#.####.#####.##.#####.####.#",
    "#o####.#####.##.#####.####o#",
    "#.####.#####.##.#####.####.#",
    "#............##............#",
    "#.####.##.########.##.####.#",
    "#.####.##.########.##.####.#",
    "#......##....##....##......#",
    "######.##### ## #####.######",
    "######.##### ## #####.######",
    "######.##          ##.######",
    "######.## ###--### ##.######",
    "      .   # GGGG #   .      ",
    "######.## #      # ##.######",
    "######.## # #### # ##.######",
    "######.## # #### # ##.######",
    "#............##............#",
    "#.####.#####.##.#####.####.#",
    "#o..##................##..o#",
    "###.##.##.########.##.##.###",
    "#......##....##....##......#",
    "#.##########.##.##########.#",
    "#............##............#",
    "############################",
    "############################",
    "############################",
    "############################",
    "############################",
    "############################",
    "############################",
]

def _encode_layers(rows):
    H, W = len(rows), len(rows[0])
    maze = jnp.zeros((H, W), jnp.int32)
    pellets = jnp.zeros((H, W), jnp.int32)
    power = jnp.zeros((H, W), jnp.int32)
    door = jnp.zeros((H, W), jnp.int32)
    house = jnp.zeros((H, W), jnp.int32)
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch == '#':
                maze = maze.at[y, x].set(WALL)
            elif ch == '.':
                pellets = pellets.at[y, x].set(1)
            elif ch == 'o':
                power = power.at[y, x].set(1)
            elif ch == '-':
                door = door.at[y, x].set(1)
            elif ch == 'G':
                house = house.at[y, x].set(1)
            else:
                pass
    maze = jnp.where(door == 1, jnp.array(DOOR, jnp.int32), maze)
    return maze, pellets, power, door, house

maze_layout, PELLET_MASK_RAW, POWER_MASK, DOOR_MASK, HOUSE_MASK = _encode_layers(_maze_rows)

if int(PELLET_MASK_RAW.sum()) == 0:
    walkable = (maze_layout != WALL) & (maze_layout != DOOR) & (HOUSE_MASK == 0)
    PELLET_MASK = (walkable & (POWER_MASK == 0)).astype(jnp.int32)
else:
    PELLET_MASK = PELLET_MASK_RAW

TUNNEL_ROWS = jnp.array([14], jnp.int32)

# ==============================
# Helpers & Movement Primitives
# ==============================
@jax.jit
def in_tunnel_row(y):
    return (y == TUNNEL_ROWS[0])

@jax.jit
def move_pacman(pos, direction, grid):
    next_pos = pos + direction
    x, y = next_pos[0], next_pos[1]
    h, w = grid.shape

    def wrap_left(_):  return jnp.array([w - 1, y], jnp.int32)
    def wrap_right(_): return jnp.array([0, y], jnp.int32)
    next_pos = lax.cond((x < 0) & in_tunnel_row(y), wrap_left,  lambda _: next_pos, operand=None)
    x, y = next_pos[0], next_pos[1]
    next_pos = lax.cond((x >= w) & in_tunnel_row(y), wrap_right, lambda _: next_pos, operand=None)
    x, y = next_pos[0], next_pos[1]

    in_bounds = (0 <= x) & (x < w) & (0 <= y) & (y < h)
    cell = jnp.where(in_bounds, grid[y, x], jnp.array(WALL, grid.dtype))
    can_move = (cell != WALL) & (cell != DOOR)  # Tür blockiert Pac-Man
    return lax.cond(can_move, lambda _: next_pos, lambda _: pos, operand=None)

@jax.jit
def door_center_x(door_mask_row: chex.Array) -> jnp.int32:
    idxs = jnp.arange(GRID_WIDTH, dtype=jnp.int32)
    mask = (door_mask_row == 1).astype(jnp.int32)
    cnt  = jnp.maximum(mask.sum(), 1)
    sx   = (idxs * mask).sum()
    return (sx // cnt).astype(jnp.int32)

@jax.jit
def ghost_step_towards(ghost_pos, target_pos, maze, key, allow_door: bool):
    allow_door = jnp.array(allow_door)
    new_positions = ghost_pos + DIRECTIONS
    xs, ys = new_positions[:,0], new_positions[:,1]
    h, w = maze.shape
    in_bounds = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    cell_vals = jnp.where(in_bounds, maze[ys, xs], jnp.array(WALL, maze.dtype))
    valid = (cell_vals != WALL) & (allow_door | (cell_vals != DOOR))
    d2 = jnp.where(valid, jnp.sum((new_positions - target_pos) ** 2, axis=1), jnp.inf)
    idx = jnp.argmin(d2)
    return jnp.where(valid[idx], new_positions[idx], ghost_pos)

@jax.jit
def ghost_random_step(ghost_pos, maze, key, allow_door: bool):
    allow_door = jnp.array(allow_door)
    new_positions = ghost_pos + DIRECTIONS
    xs, ys = new_positions[:,0], new_positions[:,1]
    h, w = maze.shape
    in_bounds = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    cell_vals = jnp.where(in_bounds, maze[ys, xs], jnp.array(WALL, maze.dtype))
    valid = (cell_vals != WALL) & (allow_door | (cell_vals != DOOR))
    rnd = random.uniform(key, (4,))
    scores = jnp.where(valid, rnd, -jnp.inf)
    idx = jnp.argmax(scores)
    return jnp.where(valid[idx], new_positions[idx], ghost_pos)

@jax.jit
def blinky_move(ghost_pos, pacman_pos, maze, key, allow_door: bool):
    return ghost_step_towards(ghost_pos, pacman_pos, maze, key, allow_door)

@jax.jit
def pinky_move(ghost_pos, pacman_pos, pacman_dir, maze, key, allow_door: bool):
    target = pacman_pos + 4 * pacman_dir
    return ghost_step_towards(ghost_pos, target, maze, key, allow_door)

@jax.jit
def inky_move(ghost_pos, pacman_pos, pacman_dir, blinky_pos, maze, key, allow_door: bool):
    ahead = pacman_pos + 2 * pacman_dir
    vec = ahead - blinky_pos
    target = blinky_pos + 2 * vec
    return ghost_step_towards(ghost_pos, target, maze, key, allow_door)

@jax.jit
def clyde_move(ghost_pos, pacman_pos, maze, key, allow_door: bool):
    dist = jnp.linalg.norm((ghost_pos - pacman_pos).astype(jnp.float32))
    return lax.cond(
        dist < 8.0,
        lambda _: ghost_random_step(ghost_pos, maze, key, allow_door),
        lambda _: ghost_step_towards(ghost_pos, pacman_pos, maze, key, allow_door),
        operand=None,
    )

# ==============================
# State / Obs / Info
# ==============================
class PacmanState(NamedTuple):
    pacman_pos: chex.Array
    pacman_dir: chex.Array
    ghost_positions: chex.Array
    ghost_dirs: chex.Array
    ghost_mode: chex.Array
    pellets: chex.Array
    power_pellets: chex.Array
    score: chex.Array
    step_count: chex.Array
    game_over: chex.Array
    power_mode_timer: chex.Array
    lives: chex.Array
    rng_key: chex.Array
    ghost_count: chex.Array
    maze: chex.Array
    base_move_interval: chex.Array
    power_time_ticks: chex.Array
    respawn_cooldown: chex.Array
    mode_timer: chex.Array
    mode_index: chex.Array
    door_mask: chex.Array
    door_row: chex.Array
    ghost_in_house: chex.Array
    ghost_release_timer: chex.Array
    fruit_active: chex.Array
    fruit_timer: chex.Array
    fruit_type_idx: chex.Array
    fruit_pos: chex.Array
    pellets_eaten: chex.Array
    next_fruit_idx: chex.Array
    level_id: chex.Array

class PacmanObservation(NamedTuple):
    pacman_pos: chex.Array
    ghost_positions: chex.Array
    pellets: chex.Array
    power_pellets: chex.Array

class PacmanInfo(NamedTuple):
    score: chex.Array
    done: chex.Array

# ==============================
# Observation & Sprite-Render
# ==============================
@jax.jit
def build_grid(state: PacmanState) -> chex.Array:
    grid = state.maze
    grid = grid.at[state.pacman_pos[1], state.pacman_pos[0]].set(PACMAN)
    grid = grid.at[state.ghost_positions[:,1], state.ghost_positions[:,0]].set(GHOST)
    grid = jnp.where((state.pellets > 0) & (grid == EMPTY), jnp.array(PELLET, grid.dtype), grid)
    grid = jnp.where((state.power_pellets > 0) & (grid == EMPTY), jnp.array(POWER_PELLET, grid.dtype), grid)
    return grid

SPR_H = 6  # 31*6=186
SPR_W = 5  # 28*5=140

def _tile_view(sub: jnp.ndarray, ty: int, tx: int) -> jnp.ndarray:
    y0, x0 = ty*SPR_H, tx*SPR_W
    return sub[y0:y0+SPR_H, x0:x0+SPR_W, :]

def _stamp(sub: jnp.ndarray, ty: int, tx: int, tile: jnp.ndarray) -> jnp.ndarray:
    y0, x0 = ty*SPR_H, tx*SPR_W
    return sub.at[y0:y0+SPR_H, x0:x0+SPR_W, :].set(tile)

def _mk_coords() -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    Y = jnp.arange(SPR_H, dtype=jnp.float32)[:, None]
    X = jnp.arange(SPR_W, dtype=jnp.float32)[None, :]
    cy = (SPR_H - 1) / 2.0
    cx = (SPR_W - 1) / 2.0
    xn = (X - cx) / (SPR_W / 2.0)
    yn = (Y - cy) / (SPR_H / 2.0)
    return X, Y, xn, yn

_X, _Y, _XN, _YN = _mk_coords()

def _angle_wrap(a: jnp.ndarray) -> jnp.ndarray:
    return (a + jnp.pi) % (2*jnp.pi) - jnp.pi

def _make_pacman_tile(dir_vec: jnp.ndarray, step_count: int) -> jnp.ndarray:
    dx, dy = float(dir_vec[0]), float(dir_vec[1])
    phi = 0.0 if (dx == 0.0 and dy == 0.0) else float(jnp.arctan2(dy, dx))
    open_phase = (step_count // 2) % 2
    mouth_theta = 0.90 if open_phase == 1 else 0.25
    theta = jnp.arctan2(_YN, _XN)
    r2 = _XN**2 + _YN**2
    circle = r2 <= 1.0
    mouth = jnp.abs(_angle_wrap(theta - phi)) < mouth_theta
    mask = circle & (~mouth)

    tile = jnp.zeros((SPR_H, SPR_W, 3), jnp.uint8)
    color = PACMAN_YELLOW[None, None, :]
    tile = jnp.where(mask[..., None], color, tile)
    return tile

def _make_ghost_tile(color: jnp.ndarray, mode: int, dir_vec: jnp.ndarray, frightened: bool, blink: bool) -> jnp.ndarray:
    r2 = (_XN**2 + (_YN + 0.1)**2)
    body = r2 <= 1.0
    tile = jnp.zeros((SPR_H, SPR_W, 3), jnp.uint8)

    if mode != G_EYES:
        base_col = FRIGHT_BLUE if frightened else color
        body_col = FLASH_WHITE if (frightened and blink) else base_col
        tile = jnp.where(body[..., None], body_col[None, None, :], tile)

    ex_l, ey = -0.35, -0.05
    ex_r = 0.35
    eye_rx, eye_ry = 0.20, 0.28
    eye_l = (((_XN-ex_l)/eye_rx)**2 + ((_YN-ey)/eye_ry)**2) <= 1.0
    eye_r = (((_XN-ex_r)/eye_rx)**2 + ((_YN-ey)/eye_ry)**2) <= 1.0
    eye = eye_l | eye_r
    tile = jnp.where(eye[..., None], EYES_WHITE[None,None,:], tile)

    dx, dy = float(dir_vec[0]), float(dir_vec[1])
    if dx == 0.0 and dy == 0.0:
        pdx, pdy = 0.0, 0.0
    else:
        vlen = max(abs(dx)+abs(dy), 1.0)
        pdx, pdy = dx / vlen * 0.1, dy / vlen * 0.1
    pup_rx, pup_ry = 0.10, 0.13
    pup_l = (((_XN-(ex_l+pdx))/pup_rx)**2 + ((_YN-(ey+pdy))/pup_ry)**2) <= 1.0
    pup_r = (((_XN-(ex_r+pdx))/pup_rx)**2 + ((_YN-(ey+pdy))/pup_ry)**2) <= 1.0
    pup = pup_l | pup_r
    tile = jnp.where(pup[..., None], PUPIL_BLUE[None,None,:], tile)
    return tile

def _make_fruit_tile(color: jnp.ndarray) -> jnp.ndarray:
    tile = jnp.zeros((SPR_H, SPR_W, 3), jnp.uint8)
    y0, y1 = SPR_H//2 - 1, SPR_H//2 + 2
    x0, x1 = SPR_W//2 - 1, SPR_W//2 + 2
    tile = tile.at[y0:y1, x0:x1, :].set(color)
    return tile

def render_arcade(state: PacmanState) -> jnp.ndarray:
    H_OUT, W_OUT = 210, 160
    sub = jnp.zeros((GRID_HEIGHT*SPR_H, GRID_WIDTH*SPR_W, 3), jnp.uint8)

    frightened = bool(state.power_mode_timer > 0)
    blink = frightened and (int(state.power_mode_timer) % 8 < 4)

    # Wände/Tür
    for ty in range(GRID_HEIGHT):
        for tx in range(GRID_WIDTH):
            cell = int(state.maze[ty, tx])
            tile = _tile_view(sub, ty, tx)
            if cell == WALL:
                tile = jnp.where(jnp.ones((SPR_H, SPR_W, 1), jnp.bool_), WALL_BLUE[None,None,:], tile)
                sub = _stamp(sub, ty, tx, tile)
            elif cell == DOOR:
                ymid = SPR_H // 2
                tile = tile.at[ymid:ymid+1, :, :].set(DOOR_WHITE)
                sub = _stamp(sub, ty, tx, tile)

    # Pellets / Power
    for ty in range(GRID_HEIGHT):
        for tx in range(GRID_WIDTH):
            if int(state.pellets[ty, tx]) == 1:
                y = ty*SPR_H + SPR_H//2
                x = tx*SPR_W + SPR_W//2
                sub = sub.at[y, x, :].set(PELLET_WHITE)
            if int(state.power_pellets[ty, tx]) == 1:
                y0 = ty*SPR_H + SPR_H//2 - 1
                x0 = tx*SPR_W + SPR_W//2 - 1
                sub = sub.at[y0:y0+2, x0:x0+2, :].set(PWR_WHITE)

    # Fruit
    if bool(state.fruit_active):
        fcol = FRUIT_COLORS[int(state.fruit_type_idx)]
        f_tile = _make_fruit_tile(fcol)
        sub = _stamp(sub, int(state.fruit_pos[1]), int(state.fruit_pos[0]), f_tile)

    # Pac-Man
    p_tile = _make_pacman_tile(state.pacman_dir, int(state.step_count))
    sub = _stamp(sub, int(state.pacman_pos[1]), int(state.pacman_pos[0]), p_tile)

    # Geister
    cols = [BLINKY_RED, PINKY_PINK, INKY_CYAN, CLYDE_ORANGE]
    for gi in range(int(state.ghost_positions.shape[0])):
        gx, gy = int(state.ghost_positions[gi, 0]), int(state.ghost_positions[gi, 1])
        mode = int(state.ghost_mode[gi])
        dv = jnp.array([state.pacman_pos[0] - gx, state.pacman_pos[1] - gy], jnp.float32)
        g_tile = _make_ghost_tile(cols[gi % 4], mode, dv, frightened, blink)
        sub = _stamp(sub, gy, gx, g_tile)

    img = jnp.zeros((H_OUT, W_OUT, 3), jnp.uint8)
    img = img.at[:sub.shape[0], :sub.shape[1], :].set(sub)
    return img

# ==============================
# Level-Bank & Reset
# ==============================
def _build_level_bank_arcade() -> Dict[int, Dict[str, chex.Array]]:
    bank: Dict[int, Dict[str, chex.Array]] = {}
    door_row = jnp.argmax(jnp.any(DOOR_MASK == 1, axis=1)).astype(jnp.int32)
    for lvl, cfg in enumerate(LEVELS, start=1):
        bank[lvl] = {
            "maze": maze_layout,
            "pellets": PELLET_MASK,
            "power": POWER_MASK,
            "door": DOOR_MASK,
            "door_row": door_row,
            "ghost_move_base": jnp.array(cfg["ghost_move_base"], jnp.int32),
            "power_time_ticks": jnp.array(cfg["power_time"], jnp.int32),
            "ghost_count": jnp.array(cfg["ghost_count"], jnp.int32),
            "mode_durations": MODE_DURATIONS,
            "level_id": jnp.array(lvl-1, jnp.int32),
        }
    return bank

@jax.jit
def _reset_kernel_arcade(level_payload: Dict[str, chex.Array], key: chex.PRNGKey):
    pacman_pos = PACMAN_START_POS
    home = jnp.broadcast_to(GHOST_HOME_POS, (MAX_GHOSTS, 2)).astype(jnp.int32)
    ghost_in_house = jnp.ones((MAX_GHOSTS,), jnp.int32)
    ghost_release_timer = jnp.array([0, 20, 40, 60], jnp.int32)

    state = PacmanState(
        pacman_pos=pacman_pos,
        pacman_dir=jnp.array([0,0], jnp.int32),
        ghost_positions=home,
        ghost_dirs=jnp.zeros((MAX_GHOSTS,2), jnp.int32),
        ghost_mode=jnp.full((MAX_GHOSTS,), G_CHASE, jnp.int32),
        pellets=level_payload["pellets"],
        power_pellets=level_payload["power"],
        score=jnp.array(0.0, jnp.float32),
        step_count=jnp.array(0, jnp.int32),
        game_over=jnp.array(False),
        power_mode_timer=jnp.array(0, jnp.int32),
        lives=jnp.array(3, jnp.int32),
        rng_key=key,
        ghost_count=level_payload["ghost_count"],
        maze=level_payload["maze"],
        base_move_interval=level_payload["ghost_move_base"],
        power_time_ticks=level_payload["power_time_ticks"],
        respawn_cooldown=jnp.array(20, jnp.int32),
        mode_timer=level_payload["mode_durations"][0],
        mode_index=jnp.array(0, jnp.int32),
        door_mask=level_payload["door"],
        door_row=level_payload["door_row"],
        ghost_in_house=ghost_in_house,
        ghost_release_timer=ghost_release_timer,
        fruit_active=jnp.array(False),
        fruit_timer=jnp.array(0, jnp.int32),
        fruit_type_idx=jnp.array(0, jnp.int32),
        fruit_pos=FRUIT_POS,
        pellets_eaten=jnp.array(0, jnp.int32),
        next_fruit_idx=jnp.array(0, jnp.int32),
        level_id=level_payload["level_id"],
    )
    obs = PacmanObservation(
        pacman_pos=state.pacman_pos,
        ghost_positions=state.ghost_positions,
        pellets=state.pellets,
        power_pellets=state.power_pellets,
    )
    return obs, state

# ==============================
# Step
# ==============================
@jax.jit
def step_fn_arcade(state: PacmanState, action: chex.Array):
    maze = state.maze
    new_dir = DIRECTIONS[action]
    new_pos = move_pacman(state.pacman_pos, new_dir, maze)

    respawn_cooldown = jnp.maximum(jnp.array(0, jnp.int32), state.respawn_cooldown - 1)

    had_pellet = state.pellets[new_pos[1], new_pos[0]] > 0
    had_power  = state.power_pellets[new_pos[1], new_pos[0]] > 0

    pellets = state.pellets.at[new_pos[1], new_pos[0]].set(0)
    power_pellets = state.power_pellets.at[new_pos[1], new_pos[0]].set(0)

    pellets_eaten = state.pellets_eaten + lax.select(had_pellet, jnp.int32(1), jnp.int32(0))
    score = state.score + lax.select(had_pellet, jnp.array(10.0, jnp.float32), jnp.array(0.0, jnp.float32))

    power_mode_timer = lax.select(
        had_power,
        state.power_time_ticks,
        jnp.maximum(jnp.array(0, jnp.int32), state.power_mode_timer - 1),
    )

    mode_timer = state.mode_timer - 1
    mode_index = state.mode_index
    def advance_mode(_):
        next_idx = jnp.minimum(mode_index + 1, jnp.array(5, jnp.int32))
        return MODE_DURATIONS[next_idx], next_idx
    mode_timer, mode_index = lax.cond((mode_timer <= 0) & (power_mode_timer == 0),
                                      advance_mode, lambda _: (mode_timer, mode_index), operand=None)

    # Früchte
    fruit_active = state.fruit_active
    fruit_timer  = state.fruit_timer
    fruit_type_idx = state.fruit_type_idx
    next_fruit_idx = state.next_fruit_idx

    def spawn_logic(_):
        can_spawn = next_fruit_idx < 2
        threshold = FRUIT_THRESHOLDS[jnp.minimum(next_fruit_idx, jnp.int32(1))]
        should_spawn = can_spawn & (pellets_eaten >= threshold) & (~fruit_active)
        lvl = jnp.minimum(state.level_id, jnp.int32(LEVEL_FRUITS.shape[0]-1))
        pair = LEVEL_FRUITS[lvl]
        spawn_type = pair[jnp.minimum(next_fruit_idx, jnp.int32(1))]
        new_active = jnp.where(should_spawn, True, fruit_active)
        new_timer  = jnp.where(should_spawn, jnp.int32(200), fruit_timer)
        new_type   = jnp.where(should_spawn, spawn_type, fruit_type_idx)
        new_next   = jnp.where(should_spawn, next_fruit_idx + 1, next_fruit_idx)
        return new_active, new_timer, new_type, new_next

    fruit_active, fruit_timer, fruit_type_idx, next_fruit_idx = spawn_logic(None)
    fruit_timer = jnp.maximum(fruit_timer - jnp.int32(1), jnp.int32(0))
    fruit_active = jnp.where(fruit_timer == 0, False, fruit_active)

    got_fruit = fruit_active & jnp.all(new_pos == state.fruit_pos)
    fruit_points = FRUIT_POINTS[jnp.minimum(fruit_type_idx, jnp.int32(FRUIT_POINTS.shape[0]-1))]
    score = score + lax.select(got_fruit, fruit_points.astype(jnp.float32), jnp.float32(0.0))
    fruit_active = jnp.where(got_fruit, False, fruit_active)
    fruit_timer  = jnp.where(got_fruit, jnp.int32(0), fruit_timer)

    # Geister
    n = state.ghost_positions.shape[0]
    ghost_types = jnp.arange(n, dtype=jnp.int32) % 4
    keys = random.split(state.rng_key, n + 2)
    ghost_keys = keys[1:1+n]
    next_key = keys[-1]
    blinky_pos = state.ghost_positions[0]

    base = state.base_move_interval
    is_fright = (power_mode_timer > 0)
    def ghost_interval(g_mode, in_house):
        return jnp.where(g_mode == G_EYES, jnp.int32(1),
               jnp.where(in_house == 1, base * 2,
               jnp.where(is_fright, base * 2, base)))
    intervals = jax.vmap(ghost_interval)(state.ghost_mode, state.ghost_in_house)

    def move_one(i, pos, key, gmode, in_house, rel_t):
        step_now = (state.step_count % intervals[i]) == 0

        def no_move():
            return pos, gmode, in_house, rel_t

        def do_move():
            def eyes_path(_):
                cx = door_center_x(state.door_mask[state.door_row])
                above = pos[1] < state.door_row
                on_cx = pos[0] == cx
                def go_horiz(__):
                    dx = jnp.sign(cx - pos[0]).astype(jnp.int32)
                    return pos + jnp.array([dx, 0], jnp.int32)
                def go_down(__):
                    return pos + jnp.array([0, 1], jnp.int32)
                def to_home(__):
                    return ghost_step_towards(pos, GHOST_HOME_POS, maze, key, True)
                newp = lax.cond(above,
                                lambda __: lax.cond(on_cx, go_down, go_horiz, operand=None),
                                to_home, operand=None)
                at_home = jnp.all(newp == GHOST_HOME_POS)
                new_mode = jnp.where(at_home, jnp.int32(G_CHASE), gmode)
                new_in_house = jnp.where(at_home, jnp.int32(1), in_house)
                new_rel = jnp.where(at_home, jnp.int32(30), rel_t)
                return newp, new_mode, new_in_house, new_rel

            def in_house_path(_):
                def stay(__): return pos, gmode, in_house, rel_t
                def route(__):
                    cx = door_center_x(state.door_mask[state.door_row])
                    on_cx = (pos[0] == cx)
                    def go_up(__):   return pos + jnp.array([0, -1], jnp.int32)
                    def go_h(__):
                        dx = jnp.sign(cx - pos[0]).astype(jnp.int32)
                        return pos + jnp.array([dx, 0], jnp.int32)
                    newp = lax.cond(on_cx, go_up, go_h, operand=None)
                    out = (newp[1] < state.door_row)
                    new_in = jnp.where(out, jnp.int32(0), jnp.int32(1))
                    return newp, gmode, new_in, rel_t
                return lax.cond(rel_t > 0, stay, route, operand=None)

            def outside_path(_):
                allow_door = False
                def frightened(_):
                    return ghost_random_step(pos, maze, key, allow_door), gmode, in_house, rel_t
                def normal(_):
                    is_scatter = (mode_index % 2 == 0)
                    corners = jnp.array([[1,1],[GRID_WIDTH-2,1],[GRID_WIDTH-2,GRID_HEIGHT-2],[1,GRID_HEIGHT-2]], jnp.int32)
                    corner = corners[i % 4]
                    def do_scatter(__):
                        return ghost_step_towards(pos, corner, maze, key, allow_door), gmode, in_house, rel_t
                    def do_chase(__):
                        new_pos_pac = new_pos
                        newp = lax.switch(
                            ghost_types[i],
                            (
                                lambda _: blinky_move(pos, new_pos_pac, maze, key, allow_door),
                                lambda _: pinky_move(pos, new_pos_pac, new_dir, maze, key, allow_door),
                                lambda _: inky_move(pos, new_pos_pac, new_dir, blinky_pos, maze, key, allow_door),
                                lambda _: clyde_move(pos, new_pos_pac, maze, key, allow_door),
                            ),
                            operand=None,
                        )
                        return newp, gmode, in_house, rel_t
                    return lax.cond(is_scatter, do_scatter, do_chase, operand=None)
                return lax.cond(power_mode_timer > 0, frightened, normal, operand=None)

            return lax.switch(
                jnp.where(gmode == G_EYES, 0, jnp.where(in_house == 1, 1, 2)),
                (eyes_path, in_house_path, outside_path),
                operand=None,
            )

        return lax.cond(step_now, do_move, no_move)

    idxs = jnp.arange(n, dtype=jnp.int32)
    ghost_positions, ghost_mode, ghost_in_house, ghost_release_timer = jax.vmap(
        lambda i, pos, key, gm, ih, rt: move_one(i, pos, key, gm, ih, jnp.maximum(rt - 1, 0))
    )(idxs, state.ghost_positions, ghost_keys, state.ghost_mode, state.ghost_in_house, state.ghost_release_timer)

    def collide_body(i, carry):
        pac_pos, g_positions, score_val, lives_val, cooldown_val, g_mode, g_in, g_rel = carry
        gpos = g_positions[i]
        same = jnp.all(pac_pos == gpos)

        def on_collide(_):
            def eat_case(_):
                g_mode2 = g_mode.at[i].set(G_EYES)
                return (pac_pos, g_positions, score_val + jnp.array(200.0, jnp.float32),
                        lives_val, cooldown_val, g_mode2, g_in, g_rel)

            def death_case(_):
                def do_nothing(__):
                    return (pac_pos, g_positions, score_val, lives_val, cooldown_val, g_mode, g_in, g_rel)
                def lose_life(__):
                    return (PACMAN_START_POS, g_positions, score_val,
                            lives_val - jnp.array(1, jnp.int32), jnp.array(20, jnp.int32),
                            g_mode, g_in, g_rel)
                return lax.cond(cooldown_val > 0, do_nothing, lose_life, operand=None)

            return lax.cond((power_mode_timer > 0), eat_case, death_case, operand=None)

        return lax.cond(same, on_collide,
                        lambda _: (pac_pos, g_positions, score_val, lives_val, cooldown_val, g_mode, g_in, g_rel),
                        operand=None)

    pacman_pos, ghost_positions, score, lives, respawn_cooldown, ghost_mode, ghost_in_house, ghost_release_timer = lax.fori_loop(
        0, ghost_positions.shape[0], collide_body,
        (new_pos, ghost_positions, score, state.lives, respawn_cooldown, ghost_mode, ghost_in_house, ghost_release_timer)
    )

    game_over = jnp.logical_or(state.game_over, lives <= 0)

    new_state = PacmanState(
        pacman_pos=pacman_pos,
        pacman_dir=new_dir,
        ghost_positions=ghost_positions,
        ghost_dirs=state.ghost_dirs,
        ghost_mode=ghost_mode,
        pellets=pellets,
        power_pellets=power_pellets,
        score=score,
        step_count=state.step_count + 1,
        game_over=game_over,
        power_mode_timer=power_mode_timer,
        lives=lives,
        rng_key=next_key,
        ghost_count=state.ghost_count,
        maze=state.maze,
        base_move_interval=state.base_move_interval,
        power_time_ticks=state.power_time_ticks,
        respawn_cooldown=respawn_cooldown,
        mode_timer=mode_timer,
        mode_index=mode_index,
        door_mask=state.door_mask,
        door_row=state.door_row,
        ghost_in_house=ghost_in_house,
        ghost_release_timer=ghost_release_timer,
        fruit_active=fruit_active,
        fruit_timer=fruit_timer,
        fruit_type_idx=fruit_type_idx,
        fruit_pos=state.fruit_pos,
        pellets_eaten=pellets_eaten,
        next_fruit_idx=next_fruit_idx,
        level_id=state.level_id,
    )

    obs = PacmanObservation(
        pacman_pos=new_state.pacman_pos,
        ghost_positions=new_state.ghost_positions,
        pellets=new_state.pellets,
        power_pellets=new_state.power_pellets,
    )
    reward = lax.select(had_pellet, jnp.array(10.0, jnp.float32), jnp.array(0.0, jnp.float32))
    reward = reward + lax.select(got_fruit, fruit_points.astype(jnp.float32), jnp.float32(0.0))
    done = game_over
    info = PacmanInfo(score=score, done=done)
    return obs, new_state, reward, done, info

# ==============================
# Environment
# ==============================
class JaxPacman(JaxEnvironment[PacmanState, PacmanObservation, PacmanInfo, Dict[str, chex.Array]]):
    def __init__(self, *, level: int = 1):
        super().__init__(consts=None)
        self._level_bank = _build_level_bank_arcade()
        self._initial_level = int(max(1, min(MAX_LEVEL, int(level))))
        self._action_set = jnp.arange(4, dtype=jnp.int32)

    def action_space(self):
        return spaces.Discrete(4)

    def observation_space(self):
        return spaces.Dict({
            "pacman_pos": spaces.Box(low=0, high=max(GRID_WIDTH, GRID_HEIGHT), shape=(2,), dtype=jnp.int32),
            "ghost_positions": spaces.Box(low=0, high=max(GRID_WIDTH, GRID_HEIGHT), shape=(MAX_GHOSTS, 2), dtype=jnp.int32),
            "pellets": spaces.Box(low=0, high=1, shape=(GRID_HEIGHT, GRID_WIDTH), dtype=jnp.int32),
            "power_pellets": spaces.Box(low=0, high=1, shape=(GRID_HEIGHT, GRID_WIDTH), dtype=jnp.int32),
        })

    def image_space(self):
        return spaces.Box(low=0, high=255, shape=(210, 160, 3), dtype=jnp.uint8)

    def action_set(self):
        return self._action_set

    def reset(self, key: chex.PRNGKey):
        payload = self._level_bank[self._initial_level]
        # Wichtig: keine self.consts-Zuweisung (Framework-Feedback)
        return _reset_kernel_arcade(payload, key)

    def step(self, state: PacmanState, action):
        return step_fn_arcade(state, jnp.array(action, dtype=jnp.int32))

    def render(self, state: PacmanState):
        return render_arcade(state)

    def obs_to_flat_array(self, obs: PacmanObservation) -> jnp.ndarray:
        parts = [
            obs.pacman_pos.reshape(-1),
            obs.ghost_positions.reshape(-1),
            obs.pellets.reshape(-1),
            obs.power_pellets.reshape(-1),
        ]
        return jnp.concatenate([p.astype(jnp.float32) for p in parts], axis=0)

# ==============================
# Minimaler Runner + HUD
# ==============================
if __name__ == "__main__":
    import numpy as np
    import pygame
    pygame.init()
    env = JaxPacman(level=1)
    key = random.PRNGKey(0)
    obs, state = env.reset(key)

    frame = env.render(state)
    H, W = int(frame.shape[0]), int(frame.shape[1])
    SCALE = 2  # Fenster-Scaling
    screen = pygame.display.set_mode((W*SCALE, H*SCALE))
    pygame.display.set_caption("Pac-Man — JAX Arcade (Sprites)")
    clock = pygame.time.Clock()
    action = 1  # DOWN
    font = pygame.font.SysFont(None, 18)

    def surf(img):
        arr_np = np.array(img)  # JAX -> NumPy
        s = pygame.surfarray.make_surface(arr_np.transpose(1, 0, 2))
        return pygame.transform.scale(s, (W*SCALE, H*SCALE))

    def draw_hud(surface, state):
        # Score links oben
        score_txt = font.render(f"Score: {int(state.score)}", True, (255,255,255))
        surface.blit(score_txt, (6, 6))
        # Level daneben
        level_txt = font.render(f"Level: {int(state.level_id) + 1}", True, (255,255,255))
        surface.blit(level_txt, (6 + score_txt.get_width() + 16, 6))
        # Leben unten links
        base_y = H*SCALE - 14
        base_x = 8
        for i in range(int(state.lives)):
            cx = base_x + i * 18
            pygame.draw.circle(surface, (255,255,0), (cx, base_y), 6)
        # Power-Timer-Balken rechts oben
        if int(state.power_mode_timer) > 0:
            t = int(state.power_mode_timer)
            w = max(1, min(80, t))
            pygame.draw.rect(surface, (0,200,255), pygame.Rect(W*SCALE - w - 8, 8, w, 10), 0)

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_UP:
                    action = 0
                elif e.key == pygame.K_DOWN:
                    action = 1
                elif e.key == pygame.K_LEFT:
                    action = 2
                elif e.key == pygame.K_RIGHT:
                    action = 3

        obs, state, r, done, info = env.step(state, jnp.int32(action))
        frame = env.render(state)
        screen.blit(surf(frame), (0, 0))
        draw_hud(screen, state)
        pygame.display.flip()

        if bool(done):
            running = False

        clock.tick(30)

    pygame.quit()
