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

WIDTH = 210
HEIGHT = 160
SCALING_FACTOR = 3
PLAYER_SIZE = (10, 10)
BULLET_SPEED = 5
MAX_BULLETS = 1

PLAYER_BOUNDS = (0, WIDTH*SCALING_FACTOR), (0, HEIGHT*SCALING_FACTOR)

class BerzerkState(NamedTuple):
    player_pos: chex.Array  # (x, y)
    lives: chex.Array
    bullets: chex.Array     # (N, 2)
    bullet_dirs: chex.Array # (N, 2)
    bullet_active: chex.Array  # (N,)
    enemy_pos: chex.Array  # (2,)
    enemy_move_axis: chex.Array
    last_dir: chex.Array # (2,)
    rng: chex.PRNGKey

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
    player_x = jnp.where(
        player_x < PLAYER_BOUNDS[0][0],
        PLAYER_BOUNDS[0][0],  # Clamp to min player bound
        jnp.where(
            player_x > PLAYER_BOUNDS[0][1],
            PLAYER_BOUNDS[0][1],  # Clamp to max player bound
            player_x,
        ),
    )

    player_y = jnp.where(
        player_y < PLAYER_BOUNDS[1][0],
        PLAYER_BOUNDS[1][0],
        jnp.where(player_y > PLAYER_BOUNDS[1][1], PLAYER_BOUNDS[1][1], player_y),
    )

    return player_x, player_y, player_direction

@jax.jit
def update_enemy_position(player_pos: chex.Array, enemy_pos: chex.Array, enemy_move_axis: int, speed: float = 1.0):
    """Move enemy strictly in one axis until aligned, then switch axis."""

    dx = player_pos[0] - enemy_pos[0]
    dy = player_pos[1] - enemy_pos[1]

    # Check if axis alignment is reached -> switch axis
    switch_axis = jnp.logical_or(
        jnp.logical_and(enemy_move_axis == 0, jnp.abs(dx) < 1e-2),  # aligned in x
        jnp.logical_and(enemy_move_axis == 1, jnp.abs(dy) < 1e-2)   # aligned in y
    )

    # Compute new axis: 0 = x, 1 = y
    abs_dx_greater = jnp.abs(dx) > jnp.abs(dy)
    new_axis = jnp.where(abs_dx_greater, 0, 1)

    # Only switch axis if alignment reached
    enemy_move_axis = jnp.where(switch_axis, new_axis, enemy_move_axis)

    # Update position along current axis
    move_x = jnp.where(
        enemy_move_axis == 0,
        enemy_pos[0] + jnp.sign(dx) * speed,
        enemy_pos[0]
    )

    move_y = jnp.where(
        enemy_move_axis == 1,
        enemy_pos[1] + jnp.sign(dy) * speed,
        enemy_pos[1]
    )

    new_pos = jnp.array([move_x, move_y])
    return new_pos, enemy_move_axis



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
        last_dir = jnp.array([0.0, -1.0])  # default = up
        state = BerzerkState(pos, lives, bullets, bullet_dirs, active, enemy_pos, last_dir, rng)
        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=0)
    def step(self, state: BerzerkState, action: chex.Array) -> Tuple[BerzerkObservation, BerzerkState, float, bool, BerzerkInfo]:
        
        #dx, dy, shoot = action
        #move = jnp.array([dx, dy], dtype=jnp.float32)
        #new_pos = jnp.clip(state.player_pos + move * 2.0, jnp.array([0.0, 0.0]), jnp.array([WIDTH * SCALING_FACTOR, HEIGHT * SCALING_FACTOR]))
        
        # Bewegung und Richtung
        player_x, player_y, move_dir = player_step(state, action)
        new_pos = jnp.array([player_x, player_y])

        new_enemy_pos, new_enemy_axis = update_enemy_position(new_pos, state.enemy_pos, state.enemy_move_axis)

        col_dx = jnp.abs(player_x - new_enemy_pos[0])
        col_dy = jnp.abs(player_y - new_enemy_pos[1])
        player_hits_enemy = (col_dx < 10.0) & (col_dy < 10.0)

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

        bullets, bullet_dirs, bullet_active = jax.lax.cond(
            jnp.any(
                jnp.array(
                    [
                        action == Action.FIRE,
                        action == Action.UPRIGHTFIRE,
                        action == Action.UPLEFTFIRE,
                        action == Action.DOWNFIRE,
                        action == Action.DOWNRIGHTFIRE,
                        action == Action.DOWNLEFTFIRE,
                        action == Action.RIGHTFIRE,
                        action == Action.LEFTFIRE,
                        action == Action.UPFIRE,
                    ]
                )
            ) == 1,
            lambda _: shoot_bullet(state),
            lambda _: (state.bullets, state.bullet_dirs, state.bullet_active),
            operand=None
        )

        bullets += bullet_dirs * BULLET_SPEED * bullet_active[:, None]
        bullet_active = bullet_active & (bullets[:, 0] >= 0) & (bullets[:, 0] <= WIDTH * SCALING_FACTOR) & (bullets[:, 1] >= 0) & (bullets[:, 1] <= HEIGHT * SCALING_FACTOR)

        def bullet_hits_enemy(bullet_pos):
            dx = jnp.abs(bullet_pos[0] - state.enemy_pos[0])
            dy = jnp.abs(bullet_pos[1] - state.enemy_pos[1])
            return (dx < 10.0) & (dy < 10.0)

        hits = jax.vmap(bullet_hits_enemy)(bullets) & bullet_active

        enemy_alive = ~jnp.any(hits)
        
        # Position und Bewegungsrichtung des Gegners aktualisieren
        new_enemy_pos, new_enemy_axis = update_enemy_position(new_pos, state.enemy_pos, state.enemy_move_axis)

        # Falls Gegner tot, setze Position weit weg und Bewegungsachse auf 0
        new_enemy_pos = jnp.where(enemy_alive, new_enemy_pos, jnp.array([-100.0, -100.0]))
        new_enemy_axis = jnp.where(enemy_alive, new_enemy_axis, 0)


        bullet_active = bullet_active & ~hits

        new_state = BerzerkState(
            new_pos, 2, bullets, bullet_dirs, bullet_active, new_enemy_pos, move_dir, state.rng
        )

        observation = self._get_observation(new_state)

        info = self._get_info(new_state)


        return observation, new_state, 0.0, player_hits_enemy, info

    

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
    # draw bullet (yellow box)
    for i in range(MAX_BULLETS):
        if state.bullet_active[i]:
            bx = int(state.bullets[i][0].item())
            by = int(state.bullets[i][1].item())
            pygame.draw.circle(screen, (255, 255, 0), (bx, by), 3)
    # draw enemy (red box)
    pygame.draw.rect(
        screen, (255, 0, 0),
        pygame.Rect(
            int(state.enemy_pos[0].item()) - 5,
            int(state.enemy_pos[1].item()) - 5,
            10, 10
        )
    )
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