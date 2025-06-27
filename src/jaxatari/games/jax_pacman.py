# Group: Sooraj Rathore, Kadir Özen

from functools import partial
from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
import chex
import pygame
from jaxatari.environment import JaxEnvironment
from jax import random, Array

# Make sure GRID_WIDTH and GRID_HEIGHT are defined in your environment
GRID_WIDTH = 19
GRID_HEIGHT = 11
CELL_SIZE = 20


# Simplified 19x21 Pacman map (standard size)
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

def get_valid_moves(pos: chex.Array, maze: chex.Array) -> Tuple[chex.Array, chex.Array]:
    directions = jnp.array([
        (0, -1),  # up
        (0, 1),   # down
        (-1, 0),  # left
        (1, 0),   # right
    ])

    def is_valid(direction):
        new_pos = pos + direction
        y, x = new_pos[0], new_pos[1]  # ACHTUNG: (y, x)

        in_bounds = (0 <= y) & (y < maze.shape[0]) & (0 <= x) & (x < maze.shape[1])
        is_open = maze[y, x] == 0
        return in_bounds & is_open

    valids = jax.vmap(is_valid)(directions)
    return directions, valids




# Choose direction to chase Pacman (Euclidean distance)
def ghost_chase_step(ghost_pos, pacman_pos, maze, key):
    valid_moves = get_valid_moves(ghost_pos, maze)
    if not valid_moves:
        return ghost_pos  # Stuck

    distances = [jnp.linalg.norm(jnp.array([ghost_pos[0] + dx, ghost_pos[1] + dy]) - jnp.array(pacman_pos))
                 for dx, dy in valid_moves]
    best_move = valid_moves[jnp.argmin(jnp.array(distances))]
    return (ghost_pos[0] + best_move[0], ghost_pos[1] + best_move[1])


def ghost_frightened_step(ghost_pos: chex.Array, maze: chex.Array, key: chex.PRNGKey) -> chex.Array:
    directions, valids = get_valid_moves(ghost_pos, maze)
    num_valid = jnp.sum(valids)

    def choose_move():
        valid_indices = jnp.nonzero(valids, size=4)[0]
        idx = random.randint(key, (), 0, num_valid)
        move = directions[valid_indices[idx]]
        new_pos = ghost_pos + move
        return new_pos
        #return ghost_pos + move

    def no_move():
        return ghost_pos

    return jax.lax.cond(num_valid > 0, choose_move, no_move)




class PacmanState(NamedTuple):


    pacman_pos: chex.Array  # (x, y)
    pacman_dir: chex.Array  # (dx, dy)
    ghost_positions: chex.Array  # (N_ghosts, 2)
    ghost_dirs: chex.Array  # (N_ghosts, 2)
    pellets: chex.Array  # 2D grid of 0 (empty) or 1 (pellet)
    power_pellets: chex.Array
    score: chex.Array
    step_count: chex.Array
    game_over: chex.Array
    power_mode_timer: chex.Array

class PacmanObservation(NamedTuple):
    grid: chex.Array  # 2D array showing layout of walls, pellets, pacman, ghosts

class PacmanInfo(NamedTuple):
    score: chex.Array
    done: chex.Array

# Example directions
DIRECTIONS = jnp.array([
    [0, -1],  # UP
    [0, 1],   # DOWN
    [-1, 0],  # LEFT
    [1, 0],   # RIGHT
])

@jax.jit
def move_entity(pos, direction, grid):
    """Move entity if next cell is not a wall."""
    next_pos = pos + direction
    can_move = grid[next_pos[1], next_pos[0]] != 1  # 1 = wall
    return jax.lax.cond(can_move, lambda _: next_pos, lambda _: pos, operand=None)

class JaxPacman(JaxEnvironment[PacmanState, PacmanObservation, PacmanInfo]):
    def __init__(self):
        super().__init__()
        self.frame_stack_size = 1
        self.action_set = jnp.arange(4)  # UP, DOWN, LEFT, RIGHT

    def reset(self, key=None) -> Tuple[PacmanObservation, PacmanState]:
        pacman_pos = jnp.array([9, 9])
        pacman_dir = jnp.array([0, 0])
        ghost_positions = jnp.array([[9, 3], [9, 5], [9, 7]])
        ghost_dirs = jnp.zeros_like(ghost_positions)
        pellets = (maze_layout == 0).astype(jnp.int32)
        power_pellets = jnp.zeros_like(pellets)
        power_pellets = power_pellets.at[1, 1].set(1)
        power_pellets = power_pellets.at[1, 17].set(1)
        power_pellets = power_pellets.at[9, 1].set(1)
        power_pellets = power_pellets.at[9, 17].set(1)
        pellets = pellets - power_pellets

        power_mode_timer = jnp.array(0)

        state = PacmanState(
            pacman_pos=pacman_pos,
            pacman_dir=pacman_dir,
            ghost_positions=ghost_positions,
            ghost_dirs=ghost_dirs,
            pellets=pellets,
            power_pellets=power_pellets,
            score=jnp.array(0),
            step_count=jnp.array(0),
            game_over=jnp.array(False),
            power_mode_timer=power_mode_timer
        )
        obs = self._get_observation(state)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: PacmanState, action: chex.Array) -> tuple[
        PacmanObservation, PacmanState, Array, Array, PacmanInfo]:
        new_dir = DIRECTIONS[action]
        new_pos = move_entity(state.pacman_pos, new_dir, self._get_wall_grid())
        has_pellet = state.pellets[new_pos[1], new_pos[0]] > 0
        has_power = state.power_pellets[new_pos[1], new_pos[0]] > 0

        # Consume pellet
        pellets = state.pellets.at[new_pos[1], new_pos[0]].set(0)
        power_pellets = state.power_pellets.at[new_pos[1], new_pos[0]].set(0)
        score = state.score + jax.lax.select(has_pellet, 10, 0)

        #Update power mode timer
        power_mode_timer = jax.lax.select(
            has_power,
            jnp.array(50), #Reset timer when power pellet is consumed
            jnp.maximum(0, state.power_mode_timer - 1)
        )

        # Ghost random movement
        def move_one_ghost(ghost_pos, key):
            return ghost_frightened_step(ghost_pos, self._get_wall_grid(), key)

        keys = random.split(random.PRNGKey(state.step_count), state.ghost_positions.shape[0])
        ghost_positions = jax.vmap(move_one_ghost)(state.ghost_positions, keys)

        # Check collision
        def ghost_hits_pacman(ghost_pos):
            collision = jnp.all(jnp.array([ghost_pos[1], ghost_pos[0]]) == new_pos)
            # Wenn Power-Modus aktiv ist, führt Kollision nicht zum Tod
            return jax.lax.cond(
                state.power_mode_timer > 0,
                lambda _: False,  # Im Power-Modus: keine tödliche Kollision
                lambda _: collision,  # Normaler Modus: Kollision ist tödlich
                operand=None
            )

        hits = jax.vmap(ghost_hits_pacman)(ghost_positions)
        pacman_dead = jnp.any(hits)
        game_over = jnp.logical_or(state.game_over, pacman_dead)

        new_state = PacmanState(
            pacman_pos=new_pos,
            pacman_dir=new_dir,
            ghost_positions=ghost_positions,
            ghost_dirs=state.ghost_dirs,
            pellets=pellets,
            power_pellets=power_pellets,
            score=score,
            step_count=state.step_count + 1,
            game_over=game_over,
            power_mode_timer=power_mode_timer
        )
        obs = self._get_observation(new_state)
        reward = jax.lax.select(has_pellet, 10.0, 0.0)
        done = game_over
        info = PacmanInfo(score=score, done=done)
        return obs, new_state, reward, done, info

    def _get_wall_grid(self):
        return maze_layout

    def _get_observation(self, state: PacmanState) -> PacmanObservation:
        grid = maze_layout.copy()
        # Place Pacman
        grid = grid.at[state.pacman_pos[1], state.pacman_pos[0]].set(2)
        # Place Ghosts
        for g in state.ghost_positions:
            grid = grid.at[g[0], g[1]].set(3)
        # Add pellets (only to empty spaces)
        pellet_mask = (state.pellets > 0) & (grid == 0)
        grid = jnp.where(pellet_mask, 5, grid)
        # Add power pellets
        power_pellet_mask = (state.power_pellets > 0) & (grid == 0)
        grid = jnp.where(power_pellet_mask, 4, grid)
        return PacmanObservation(grid=grid)




def main():
    pygame.init()
    screen = pygame.display.set_mode((GRID_WIDTH * CELL_SIZE, GRID_HEIGHT * CELL_SIZE))
    pygame.display.set_caption("Pacman - JAX Edition")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    env = JaxPacman()  # You must have implemented this
    key = random.PRNGKey(0)
    obs, state = env.reset(key)

    running = True
    action = 1  # Default to DOWN
    total_reward = 0

    while running:
        # --- Handle Input ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action = 0
                elif event.key == pygame.K_DOWN:
                    action = 1
                elif event.key == pygame.K_LEFT:
                    action = 2
                elif event.key == pygame.K_RIGHT:
                    action = 3

        # --- Step Environment ---
        key, subkey = random.split(key)
        obs, state, reward, done, info = env.step(state, jnp.array(action))
        total_reward += float(reward)

        # --- Render ---
        screen.fill((0, 0, 0))  # Clear screen
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                cell = obs.grid[y, x]
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if cell == 1:
                    pygame.draw.rect(screen, (0, 0, 255), rect)  # Wall
                elif cell == 2:
                    pygame.draw.circle(screen, (255, 255, 0), rect.center, CELL_SIZE // 2)  # Pacman
                elif cell == 3:
                    color = (0, 191, 255) if state.power_mode_timer > 0 else (255, 0, 0)  # Ghost when frightened
                    pygame.draw.circle(screen, color, rect.center, CELL_SIZE // 2)
                elif cell == 4:
                    pygame.draw.circle(screen, (0, 255, 255), rect.center, CELL_SIZE // 4)  # Power Pellet
                elif cell == 5:
                    pygame.draw.circle(screen, (255, 255, 255), rect.center, CELL_SIZE // 6)  # Pellet

        # --- Draw Score ---
        score_surf = font.render(f"Score: {int(total_reward)}", True, (255, 255, 255))
        screen.blit(score_surf, (10, 10))

        pygame.display.flip()
        clock.tick(10)

        if done:
            print("Game Over!")
            pygame.time.wait(1500)
            running = False

    pygame.quit()

# 👇 This ensures the game runs only when executed directly
if __name__ == "__main__":
    main()