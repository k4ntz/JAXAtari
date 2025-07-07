from typing import NamedTuple, Tuple
from functools import partial
import jax
import jax.numpy as jnp
import chex
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from gymnax.environments import spaces
import pygame

# Group: Kaan Yilmaz, Jonathan Frey
# Game: Berzerk
# We implemented a basic game loop with one enemy. You can shoot it and it will disappear but nothing will change afterwards and you have to reset manually.
# Also, if you touch the enemy you will die and respawn after a short and simple death animation inspired by the real game.
# Tested on Ubuntu Virtual Machine
# Also we currently don't use JAXAtariAction as Action. We will change this in the future

WIDTH = 160
HEIGHT = 210
SCALING_FACTOR = 3
PLAYER_SIZE = (10, 10)
BULLET_SPEED = 5
MAX_BULLETS = 1
WALL_THICKNESS = 20  # pixel thickness of deadly walls
WALL_OFFSET = 30

SCORE_OFFSET_X = 50
SCORE_OFFSET_Y = 30

EXIT_WIDTH = 30
EXIT_HEIGHT = 30

MOVEMENT_PROB = 0.05

UI_OFFSET = 30  # pixels reserved for score at bottom
PLAYER_BOUNDS = (
    (WALL_THICKNESS + WALL_OFFSET, WIDTH * SCALING_FACTOR - WALL_THICKNESS - WALL_OFFSET),
    (WALL_THICKNESS + WALL_OFFSET, HEIGHT * SCALING_FACTOR - WALL_THICKNESS - WALL_OFFSET - SCORE_OFFSET_Y)
)


class BerzerkState(NamedTuple):
    player_pos: chex.Array  # (x, y)
    lives: chex.Array
    bullets: chex.Array     # (N, 2)
    bullet_dirs: chex.Array # (N, 2)
    bullet_active: chex.Array  # (N,)
    enemy_pos: chex.Array  # (M, 2) - M enemies
    enemy_move_axis: chex.Array  # (M,) -1: nicht bewegen, 0: x-Achse, 1: y-Achse
    enemy_move_dir: chex.Array   # (M,) -1 oder 1 für Bewegungsrichtung
    enemy_move_prob: chex.Array 
    enemy_alive: chex.Array  # (M,) - ob Gegner lebt
    last_dir: chex.Array # (2,)
    rng: chex.PRNGKey
    score: chex.Array

class BerzerkObservation(NamedTuple):
    player: chex.Array
    bullets: chex.Array
    bullet_dirs: chex.Array
    bullet_active: chex.Array

class BerzerkInfo(NamedTuple):
    dummy: chex.Array  # placeholder (will be added later on)

@jax.jit
def player_step(
    state: BerzerkState, action: chex.Array
) -> tuple[chex.Array, chex.Array, chex.Array]:
    # implement all the possible movement directions for the player, the mapping is:
    # anything with left in it, add -1 to the x position
    # anything with right in it, add 1 to the x position
    # anything with up in it, add -1 to the y position
    # anything with down in it, add 1 to the y position
    up = jnp.any(
        jnp.array(
            [
                action == Action.UP,
                action == Action.UPRIGHT,
                action == Action.UPLEFT,
                action == Action.UPFIRE,
                action == Action.UPRIGHTFIRE,
                action == Action.UPLEFTFIRE,
            ]
        )
    )
    down = jnp.any(
        jnp.array(
            [
                action == Action.DOWN,
                action == Action.DOWNRIGHT,
                action == Action.DOWNLEFT,
                action == Action.DOWNFIRE,
                action == Action.DOWNRIGHTFIRE,
                action == Action.DOWNLEFTFIRE,
            ]
        )
    )
    left = jnp.any(
        jnp.array(
            [
                action == Action.LEFT,
                action == Action.UPLEFT,
                action == Action.DOWNLEFT,
                action == Action.LEFTFIRE,
                action == Action.UPLEFTFIRE,
                action == Action.DOWNLEFTFIRE,
            ]
        )
    )
    right = jnp.any(
        jnp.array(
            [
                action == Action.RIGHT,
                action == Action.UPRIGHT,
                action == Action.DOWNRIGHT,
                action == Action.RIGHTFIRE,
                action == Action.UPRIGHTFIRE,
                action == Action.DOWNRIGHTFIRE,
            ]
        )
    )

    dx = jnp.where(right, 1, jnp.where(left, -1, 0))
    dy = jnp.where(down, 1, jnp.where(up, -1, 0))

    # movement scaled
    player_x = state.player_pos[0] + dx * 2
    player_y = state.player_pos[1] + dy * 2

    # set the direction according to the movement
    player_direction = jnp.where(
            (dx != 0) | (dy != 0),
            jnp.array([dx, dy]),
            state.last_dir
        )
    
    # perform out of bounds checks
    #player_x = jnp.where(
    #    player_x < PLAYER_BOUNDS[0][0],
    #    PLAYER_BOUNDS[0][0],  # Clamp to min player bound
    #    jnp.where(
    #        player_x > PLAYER_BOUNDS[0][1],
    #        PLAYER_BOUNDS[0][1],  # Clamp to max player bound
    #        player_x,
    #    ),
    #)

    #player_y = jnp.where(
    #    player_y < PLAYER_BOUNDS[1][0],
    #    PLAYER_BOUNDS[1][0],
    #    jnp.where(player_y > PLAYER_BOUNDS[1][1], PLAYER_BOUNDS[1][1], player_y),
    #)

    return player_x, player_y, player_direction


@jax.jit
def check_exit_crossing(player_pos: chex.Array) -> chex.Array:
    """Return True if player touches an exit region (centered on wall)."""
    x, y = player_pos[0], player_pos[1]

    # Top exit
    top = (PLAYER_BOUNDS[0][0] + (PLAYER_BOUNDS[0][1] - PLAYER_BOUNDS[0][0]) / 2 - EXIT_WIDTH / 2,
           PLAYER_BOUNDS[0][0] + (PLAYER_BOUNDS[0][1] - PLAYER_BOUNDS[0][0]) / 2 + EXIT_WIDTH / 2)
    top_exit = (x > top[0]) & (x < top[1]) & (y < PLAYER_BOUNDS[1][0] + PLAYER_SIZE[1])
    # Bottom exit
    bottom_exit = (x > top[0]) & (x < top[1]) & (y > PLAYER_BOUNDS[1][1] - PLAYER_SIZE[1])

    # Left exit
    left = (PLAYER_BOUNDS[1][0] + (PLAYER_BOUNDS[1][1] - PLAYER_BOUNDS[1][0]) / 2 - EXIT_HEIGHT / 2,
            PLAYER_BOUNDS[1][0] + (PLAYER_BOUNDS[1][1] - PLAYER_BOUNDS[1][0]) / 2 + EXIT_HEIGHT / 2)
    left_exit = (y > left[0]) & (y < left[1]) & (x < PLAYER_BOUNDS[0][0] + PLAYER_SIZE[0])

    # Right exit
    right_exit = (y > left[0]) & (y < left[1]) & (x > PLAYER_BOUNDS[0][1] - PLAYER_SIZE[0])

    return top_exit | bottom_exit | left_exit | right_exit


@jax.jit
def update_enemy_position(player_pos: chex.Array, enemy_pos: chex.Array, 
                         enemy_move_axis: chex.Array, enemy_move_dir: chex.Array,
                         rng: chex.PRNGKey, move_prob: float, speed: float = 1.0):
    """
    Update enemy position with movement probability.
    Once started moving, continues until aligned with player.
    
    Args:
        enemy_move_axis: 0 for x-axis, 1 for y-axis movement
        enemy_move_dir: 1 for positive, -1 for negative direction
    """
    rng, move_rng = jax.random.split(rng)
    
    # Check if already moving (axis != -1)
    is_moving = enemy_move_axis != -1
    
    # If not moving, decide whether to start moving
    start_moving = jax.random.bernoulli(move_rng, move_prob)
    should_move = is_moving | (~is_moving & start_moving)
    
    def start_new_movement(_):
        # Choose random axis (0=x, 1=y) and direction (1 or -1)
        rng1, rng2 = jax.random.split(move_rng)
        axis = jax.random.bernoulli(rng1, 0.5).astype(jnp.int32)  # 0 or 1
        dir_ = jnp.where(player_pos[axis] > enemy_pos[axis], 1, -1)
        return axis, dir_
    
    # If not moving but should start, initialize movement
    new_axis, new_dir = jax.lax.cond(
        ~is_moving & should_move,
        start_new_movement,
        lambda _: (enemy_move_axis, enemy_move_dir),
        operand=None
    )
    
    # Update position if moving
    new_pos = jnp.where(
        new_axis == 0,  # moving in x-axis
        enemy_pos.at[0].add(new_dir * speed),
        enemy_pos
    )
    new_pos = jnp.where(
        new_axis == 1,  # moving in y-axis
        new_pos.at[1].add(new_dir * speed),
        new_pos
    )
    
    # Check if aligned with player in movement axis
    aligned_x = jnp.abs(player_pos[0] - new_pos[0]) < 10
    aligned_y = jnp.abs(player_pos[1] - new_pos[1]) < 10
    aligned = jax.lax.select(new_axis == 0, aligned_x, aligned_y)

    # Stop moving if aligned
    final_axis = jnp.where(aligned, -1, new_axis)
    final_dir = jnp.where(aligned, 0, new_dir)
    
    return new_pos, final_axis, final_dir, rng


class JaxBerzerk(JaxEnvironment[BerzerkState, BerzerkObservation, BerzerkInfo]):
    def __init__(self):
        super().__init__()
        self._action_space = spaces.Discrete(9)
        self._obs_space = spaces.Dict({
            "player": spaces.Box(0, 255, (2,), jnp.float32),
            "bullets": spaces.Box(0, 255, (MAX_BULLETS, 2), jnp.float32),
            "bullet_active": spaces.Box(0, 1, (MAX_BULLETS,), jnp.bool_),
        })
        
    @partial(jax.jit, static_argnums=(0, ))
    def _get_observation(self, state: BerzerkState) -> BerzerkObservation:
        return BerzerkObservation(
            player=state.player_pos,
            bullets=state.bullets,
            bullet_dirs=state.bullet_dirs, 
            bullet_active=state.bullet_active
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: BerzerkState) -> bool:
        return state.lives < 0
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: BerzerkState) -> BerzerkInfo: # later as params: self, state: BerzerkState, all_rewards: chex.Array
        return BerzerkInfo(
            jnp.array(0)
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: chex.PRNGKey = jax.random.PRNGKey(42)) -> Tuple[BerzerkObservation, BerzerkState]:
        pos = jnp.array([WIDTH * SCALING_FACTOR // 2, HEIGHT * SCALING_FACTOR // 2], dtype=jnp.float32)
        lives = jnp.array(2)
        bullets = jnp.zeros((MAX_BULLETS, 2), dtype=jnp.float32)
        bullet_dirs = jnp.zeros((MAX_BULLETS, 2), dtype=jnp.float32)
        active = jnp.zeros((MAX_BULLETS,), dtype=bool)
        enemy_pos = jnp.array([WIDTH * SCALING_FACTOR // 4, HEIGHT * SCALING_FACTOR // 4], dtype=jnp.float32)
        enemy_move_axis = jnp.array(-1, dtype=jnp.int32)  # Startet nicht bewegt
        enemy_move_dir = jnp.array(0, dtype=jnp.int32)    # Keine Richtung
        enemy_move_prob = jnp.array(MOVEMENT_PROB, dtype=jnp.float32) 
        last_dir = jnp.array([0.0, -1.0])  # default = up
        score = jnp.array(0, dtype=jnp.int32)
        state = BerzerkState(pos, lives, bullets, bullet_dirs, active, enemy_pos, enemy_move_axis, enemy_move_dir, enemy_move_prob, last_dir, rng, score)
        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=0)
    def step(self, state: BerzerkState, action: chex.Array) -> Tuple[BerzerkObservation, BerzerkState, float, bool, BerzerkInfo]:
        # 1. Spielerbewegung
        player_x, player_y, move_dir = player_step(state, action)
        new_pos = jnp.array([player_x, player_y])

        _, reset_state = self.reset()

        # 1a. Wandkollision prüfen (außerhalb erlaubtem Bereich = Tod)
        hit_exit = check_exit_crossing(new_pos)
        hit_wall = (
            (player_x <= PLAYER_BOUNDS[0][0] + PLAYER_SIZE[0] // 2) |
            (player_x >= PLAYER_BOUNDS[0][1] - PLAYER_SIZE[0] // 2) |
            (player_y <= PLAYER_BOUNDS[1][0] + PLAYER_SIZE[0] // 2) |
            (player_y >= PLAYER_BOUNDS[1][1] - PLAYER_SIZE[1] // 2)
        ) & ~hit_exit

        # 2. Gegnerposition + Bewegungsrichtung aktualisieren
        updated_enemy_pos, updated_enemy_axis, updated_enemy_dir, new_rng = update_enemy_position(
            new_pos, state.enemy_pos, state.enemy_move_axis, state.enemy_move_dir, 
            state.rng, state.enemy_move_prob
        )

        # 3. Kollisionsabfrage
        col_dx = jnp.abs(player_x - updated_enemy_pos[0])
        col_dy = jnp.abs(player_y - updated_enemy_pos[1])
        player_hits_enemy = (col_dx < 10.0) & (col_dy < 10.0)
        
        hit_something = player_hits_enemy | hit_wall
        lives_after = jnp.where(hit_something, state.lives - 1, state.lives)


        # 4. Schießen
        def shoot_bullet(state):
            def try_spawn(i, carry):
                bullets, directions, active = carry
                return jax.lax.cond(
                    ~active[i],
                    lambda _: (
                        bullets.at[i].set(new_pos),
                        directions.at[i].set(move_dir),
                        active.at[i].set(True),
                    ),
                    lambda _: (bullets, directions, active),
                    operand=None
                )
            return jax.lax.fori_loop(0, MAX_BULLETS, try_spawn, (state.bullets, state.bullet_dirs, state.bullet_active))

        is_shooting = jnp.any(jnp.array([
            action == Action.FIRE,
            action == Action.UPRIGHTFIRE,
            action == Action.UPLEFTFIRE,
            action == Action.DOWNFIRE,
            action == Action.DOWNRIGHTFIRE,
            action == Action.DOWNLEFTFIRE,
            action == Action.RIGHTFIRE,
            action == Action.LEFTFIRE,
            action == Action.UPFIRE,
        ]))

        bullets, bullet_dirs, bullet_active = jax.lax.cond(
            is_shooting,
            lambda _: shoot_bullet(state),
            lambda _: (state.bullets, state.bullet_dirs, state.bullet_active),
            operand=None
        )

        # 5. Kugeln bewegen
        bullets += bullet_dirs * BULLET_SPEED * bullet_active[:, None]
        bullet_active = bullet_active & (
            (bullets[:, 0] >= 0) &
            (bullets[:, 0] <= WIDTH * SCALING_FACTOR) &
            (bullets[:, 1] >= 0) &
            (bullets[:, 1] <= HEIGHT * SCALING_FACTOR)
        )

        # 6. Kollisionsabfrage (Kugeln vs Gegner)
        def bullet_hits_enemy(bullet_pos):
            dx = jnp.abs(bullet_pos[0] - updated_enemy_pos[0])
            dy = jnp.abs(bullet_pos[1] - updated_enemy_pos[1])
            return (dx < 10.0) & (dy < 10.0)

        hits = jax.vmap(bullet_hits_enemy)(bullets) & bullet_active
        enemy_alive = ~jnp.any(hits)
        score_after = jnp.where(~enemy_alive, state.score + 50, state.score)

        # 7. Neue Gegnerposition/Bewegungsrichtung nur setzen, wenn Gegner lebt
        new_enemy_pos = jnp.where(enemy_alive, updated_enemy_pos, jnp.array([-100.0, -100.0]))
        new_enemy_axis = jnp.where(enemy_alive, updated_enemy_axis, 0)

        # 8. Inaktive Kugeln durch Treffer
        bullet_active = bullet_active & ~hits

        # 9. Neuer State
        new_state = BerzerkState(
            player_pos=new_pos,
            lives=lives_after,
            bullets=bullets,
            bullet_dirs=bullet_dirs,
            bullet_active=bullet_active,
            enemy_pos=new_enemy_pos,
            enemy_move_axis=new_enemy_axis,
            enemy_move_dir=updated_enemy_dir,
            enemy_move_prob=MOVEMENT_PROB,
            last_dir=move_dir,
            rng=new_rng,
            score=score_after
        )

        new_state = jax.lax.cond(hit_something, lambda _: reset_state._replace(lives=state.lives - 1, score=state.score), lambda _: new_state, operand=None)

        # 10. Beobachtung + Info + Reward/Done
        observation = self._get_observation(new_state)
        info = self._get_info(new_state)
        reward = 0.0
        done = jnp.equal(lives_after, -1) 
        
        jax.debug.print("Leben: {}", lives_after)

        return observation, new_state, reward, done, info
    

    def action_space(self, seed: int = 0) -> spaces.Discrete:
        return self._action_space

    def observation_space(self, seed: int = 0) -> spaces.Dict:
        return self._obs_space


# not jittable render (will be replaced in the future)
def render(screen, state: BerzerkState):
    screen.fill((0, 0, 0))

    # draw player (green box)
    x = int(state.player_pos[0].item())
    y = int(state.player_pos[1].item())
    pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(x - 5, y - 5, 10, 10))

    # draw bullets (yellow)
    for i in range(MAX_BULLETS):
        if state.bullet_active[i]:
            bx = int(state.bullets[i][0].item())
            by = int(state.bullets[i][1].item())
            pygame.draw.circle(screen, (255, 255, 0), (bx, by), 3)

    # draw enemy (red)
    pygame.draw.rect(
        screen, (255, 0, 0),
        pygame.Rect(
            int(state.enemy_pos[0].item()) - 5,
            int(state.enemy_pos[1].item()) - 5,
            10, 10
        )
    )

    # draw deadly walls (gray)
    wall_color = (100, 100, 100)
    pygame.draw.rect(screen, wall_color, pygame.Rect(WALL_OFFSET, WALL_OFFSET, PLAYER_BOUNDS[0][1] - PLAYER_BOUNDS[0][0] + 2 * WALL_THICKNESS, WALL_THICKNESS))  # top
    pygame.draw.rect(screen, wall_color, pygame.Rect(WALL_OFFSET, PLAYER_BOUNDS[1][1], PLAYER_BOUNDS[0][1] - PLAYER_BOUNDS[0][0] + 2 * WALL_THICKNESS, WALL_THICKNESS))  # bottom
    pygame.draw.rect(screen, wall_color, pygame.Rect(WALL_OFFSET, PLAYER_BOUNDS[1][0], WALL_THICKNESS, PLAYER_BOUNDS[1][1] - PLAYER_BOUNDS[1][0]))  # left
    pygame.draw.rect(screen, wall_color, pygame.Rect(PLAYER_BOUNDS[0][1], PLAYER_BOUNDS[1][0], WALL_THICKNESS, PLAYER_BOUNDS[1][1] - PLAYER_BOUNDS[1][0] + WALL_THICKNESS))  # right

    # draw exits
    exit_color = (0, 100, 255)

    top_x = PLAYER_BOUNDS[0][0] + (PLAYER_BOUNDS[0][1] - PLAYER_BOUNDS[0][0]) // 2 - EXIT_WIDTH // 2
    pygame.draw.rect(screen, exit_color, pygame.Rect(top_x, PLAYER_BOUNDS[1][0] - WALL_THICKNESS, EXIT_WIDTH, WALL_THICKNESS))  # top
    pygame.draw.rect(screen, exit_color, pygame.Rect(top_x, PLAYER_BOUNDS[1][1], EXIT_WIDTH, WALL_THICKNESS))  # bottom
    left_y = PLAYER_BOUNDS[1][0] + (PLAYER_BOUNDS[1][1] - PLAYER_BOUNDS[1][0]) // 2 - EXIT_HEIGHT // 2
    pygame.draw.rect(screen, exit_color, pygame.Rect(PLAYER_BOUNDS[0][0] - WALL_THICKNESS, left_y, WALL_THICKNESS, EXIT_HEIGHT))  # left
    pygame.draw.rect(screen, exit_color, pygame.Rect(PLAYER_BOUNDS[0][1], left_y, WALL_THICKNESS, EXIT_HEIGHT))  # right


    # draw score at bottom
    font = pygame.font.SysFont(None, 24)
    score_text = font.render(f"Score: {int(state.score.item())}", True, (255, 255, 255))
    screen.blit(score_text, (SCORE_OFFSET_X, HEIGHT * SCALING_FACTOR - SCORE_OFFSET_Y))

    pygame.display.flip()



def get_human_action() -> chex.Array:
    """Get human action from keyboard with support for diagonal movement and combined fire"""
    keys = pygame.key.get_pressed()
    up = keys[pygame.K_UP] or keys[pygame.K_w]
    down = keys[pygame.K_DOWN] or keys[pygame.K_s]
    left = keys[pygame.K_LEFT] or keys[pygame.K_a]
    right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
    fire = keys[pygame.K_SPACE]

    # Diagonal movements with fire
    if up and right and fire:
        return jnp.array(Action.UPRIGHTFIRE)
    if up and left and fire:
        return jnp.array(Action.UPLEFTFIRE)
    if down and right and fire:
        return jnp.array(Action.DOWNRIGHTFIRE)
    if down and left and fire:
        return jnp.array(Action.DOWNLEFTFIRE)

    # Cardinal directions with fire
    if up and fire:
        return jnp.array(Action.UPFIRE)
    if down and fire:
        return jnp.array(Action.DOWNFIRE)
    if left and fire:
        return jnp.array(Action.LEFTFIRE)
    if right and fire:
        return jnp.array(Action.RIGHTFIRE)

    # Diagonal movements
    if up and right:
        return jnp.array(Action.UPRIGHT)
    if up and left:
        return jnp.array(Action.UPLEFT)
    if down and right:
        return jnp.array(Action.DOWNRIGHT)
    if down and left:
        return jnp.array(Action.DOWNLEFT)

    # Cardinal directions
    if up:
        return jnp.array(Action.UP)
    if down:
        return jnp.array(Action.DOWN)
    if left:
        return jnp.array(Action.LEFT)
    if right:
        return jnp.array(Action.RIGHT)
    if fire:
        return jnp.array(Action.FIRE)

    return jnp.array(Action.NOOP)


def main():
    game = JaxBerzerk()
    pygame.init()
    screen = pygame.display.set_mode((WIDTH * SCALING_FACTOR, HEIGHT * SCALING_FACTOR))
    clock = pygame.time.Clock()

    # Get jitted functions
    jitted_step = jax.jit(game.step)
    jitted_reset = jax.jit(game.reset)

    curr_obs, curr_state = jitted_reset()

    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        action = get_human_action()
        curr_obs, curr_state, reward, done, info = jitted_step(
                            curr_state, action
                        )

        if done:
            for i in range(60):
                screen.fill((0, 0, 0))
                # Show player and enemy every three frames to simulate blinking
                if i % 3 == 0:
                    x = int(curr_state.player_pos[0].item())
                    y = int(curr_state.player_pos[1].item())
                    pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(x - 5, y - 5, 10, 10))

                    ex = int(curr_state.enemy_pos[0].item())
                    ey = int(curr_state.enemy_pos[1].item())
                    pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(ex - 5, ey - 5, 10, 10))
                pygame.display.flip()
                clock.tick(60)

            # Reset after death
            _, curr_state = jitted_reset()


        render(screen, curr_state)
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()