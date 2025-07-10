from turtle import width
from typing import NamedTuple, Tuple, Optional, Dict, Any
from functools import partial
import jax
import jax.numpy as jnp
import chex
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from gymnax.environments import spaces
import pygame
import os

from jaxatari.games.jax_pong import ENEMY_SIZE
from jaxatari.renderers import AtraJaxisRenderer
import jaxatari.rendering.atraJaxis as aj

# Group: Kaan Yilmaz, Jonathan Frey
# Game: Berzerk
# We implemented a basic game loop with one enemy. You can shoot it and it will disappear but nothing will change afterwards and you have to reset manually.
# Also, if you touch the enemy you will die and respawn after a short and simple death animation inspired by the real game.
# Tested on Ubuntu Virtual Machine
# Also we currently don't use JAXAtariAction as Action. We will change this in the future

WIDTH = 160
HEIGHT = 210
SCALING_FACTOR = 3

PLAYER_SIZE = (6, 20)
PLAYER_SPEED = 1

ENEMY_SIZE = (8, 16)
NUM_ENEMIES = 5
MOVEMENT_PROB = 0.0025
ENEMY_SPEED = 0.1

BULLET_SIZE_HORIZONTAL = (4, 2)
BULLET_SIZE_VERTICAL = (1, 6)
BULLET_SPEED = 1
MAX_BULLETS = 1

WALL_THICKNESS = 4
WALL_OFFSET = (4, 4, 4, 30) # left, top, right, bottom
EXIT_WIDTH = 40
EXIT_HEIGHT = 64

SCORE_OFFSET_X = 50
SCORE_OFFSET_Y = 30

UI_OFFSET = 30  # pixels reserved for score at bottom
PLAYER_BOUNDS = (
    (WALL_THICKNESS + WALL_OFFSET[0], WIDTH - WALL_THICKNESS - WALL_OFFSET[2]),
    (WALL_THICKNESS + WALL_OFFSET[1], HEIGHT - WALL_THICKNESS - WALL_OFFSET[3])
)


class BerzerkState(NamedTuple):
    player_pos: chex.Array             # (2,)
    lives: chex.Array
    bullets: chex.Array                # (MAX_BULLETS, 2)
    bullet_dirs: chex.Array            # (MAX_BULLETS, 2)
    bullet_active: chex.Array          # (MAX_BULLETS,)
    enemy_pos: chex.Array              # (NUM_ENEMIES, 2)
    enemy_move_axis: chex.Array        # (NUM_ENEMIES,)
    enemy_move_dir: chex.Array         # (NUM_ENEMIES,)
    enemy_move_prob: chex.Array        # (1,)
    last_dir: chex.Array               # (2,)
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
    player_x = state.player_pos[0] + dx * PLAYER_SPEED
    player_y = state.player_pos[1] + dy * PLAYER_SPEED

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
           PLAYER_BOUNDS[0][0] + (PLAYER_BOUNDS[0][1] - PLAYER_BOUNDS[0][0]) / 2 + EXIT_WIDTH / 2 - PLAYER_SIZE[0])
    top_exit = (x > top[0]) & (x < top[1]) & (y < PLAYER_BOUNDS[1][0])
    # Bottom exit
    bottom_exit = (x > top[0]) & (x < top[1]) & (y > PLAYER_BOUNDS[1][1] - PLAYER_SIZE[1])

    # Left exit
    left = (PLAYER_BOUNDS[1][0] + (PLAYER_BOUNDS[1][1] - PLAYER_BOUNDS[1][0]) / 2 - EXIT_HEIGHT / 2,
            PLAYER_BOUNDS[1][0] + (PLAYER_BOUNDS[1][1] - PLAYER_BOUNDS[1][0]) / 2 + EXIT_HEIGHT / 2 - PLAYER_SIZE[1])
    left_exit = (y > left[0]) & (y < left[1]) & (x < PLAYER_BOUNDS[0][0])

    # Right exit
    right_exit = (y > left[0]) & (y < left[1]) & (x > PLAYER_BOUNDS[0][1] - PLAYER_SIZE[0])

    return top_exit | bottom_exit | left_exit | right_exit


@jax.jit
def update_enemies(player_pos, enemy_pos, enemy_axis, enemy_dir, rng, move_prob):
    enemy_rngs = jax.random.split(rng, NUM_ENEMIES)

    def update_one_enemy(_, inputs):
        rng, pos, axis, dir_, prob = inputs
        new_pos, new_axis, new_dir, _ = update_enemy_position(
            player_pos, pos, axis, dir_, rng, prob
        )
        return None, (new_pos, new_axis, new_dir)

    _, (positions, axes, dirs) = jax.lax.scan(
        update_one_enemy,
        None,
        (enemy_rngs, enemy_pos, enemy_axis, enemy_dir, move_prob)
    )

    return positions, axes, dirs


@jax.jit
def update_enemy_position(player_pos: chex.Array, enemy_pos: chex.Array, 
                         enemy_move_axis: chex.Array, enemy_move_dir: chex.Array,
                         rng: chex.PRNGKey, move_prob: float):
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
        enemy_pos.at[0].add(new_dir * ENEMY_SPEED),
        enemy_pos
    )
    new_pos = jnp.where(
        new_axis == 1,  # moving in y-axis
        new_pos.at[1].add(new_dir * ENEMY_SPEED),
        new_pos
    )
    
    # Check if aligned with player in movement axis
    aligned_x = jnp.abs(player_pos[0] - new_pos[0]) < 5
    aligned_y = jnp.abs(player_pos[1] - new_pos[1]) < 5
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
        pos = jnp.array([WIDTH // 2, HEIGHT // 2], dtype=jnp.float32)
        lives = jnp.array(2)
        bullets = jnp.zeros((MAX_BULLETS, 2), dtype=jnp.float32)
        bullet_dirs = jnp.zeros((MAX_BULLETS, 2), dtype=jnp.float32)
        active = jnp.zeros((MAX_BULLETS,), dtype=bool)
        enemy_pos = jax.random.uniform(
            rng, shape=(NUM_ENEMIES, 2),
            minval=jnp.array([PLAYER_BOUNDS[0][0], PLAYER_BOUNDS[1][0]]),
            maxval=jnp.array([PLAYER_BOUNDS[0][1] - ENEMY_SIZE[0], PLAYER_BOUNDS[1][1] - ENEMY_SIZE[1]])
        )
        enemy_move_axis = -jnp.ones((NUM_ENEMIES,), dtype=jnp.int32)
        enemy_move_dir = jnp.zeros((NUM_ENEMIES,), dtype=jnp.int32)
        enemy_move_prob = jnp.full((NUM_ENEMIES,), MOVEMENT_PROB, dtype=jnp.float32)
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
        player_left   = player_x
        player_right  = player_x + PLAYER_SIZE[0]
        player_top    = player_y
        player_bottom = player_y + PLAYER_SIZE[1]

        hit_wall = (
            (player_left   < PLAYER_BOUNDS[0][0]) |
            (player_right  > PLAYER_BOUNDS[0][1]) |
            (player_top    < PLAYER_BOUNDS[1][0]) |
            (player_bottom > PLAYER_BOUNDS[1][1])
        ) & ~hit_exit

        # 2. Gegnerposition + Bewegungsrichtung aktualisieren
        rng, enemy_rng = jax.random.split(state.rng)

        updated_enemy_pos, updated_enemy_axis, updated_enemy_dir = update_enemies(
            new_pos, state.enemy_pos, state.enemy_move_axis, state.enemy_move_dir,
            enemy_rng, state.enemy_move_prob
        )

        # 3. Kollisionsabfrage
        def object_hits_enemy(object_pos, object_size, enemy_pos):
            object_left   = object_pos[0]
            object_right  = object_pos[0] + object_size[0]
            object_top    = object_pos[1]
            object_bottom = object_pos[1] + object_size[1]

            enemy_left   = enemy_pos[0]
            enemy_right  = enemy_pos[0] + ENEMY_SIZE[0]
            enemy_top    = enemy_pos[1]
            enemy_bottom = enemy_pos[1] + ENEMY_SIZE[1]

            overlap_x = (object_left < enemy_right) & (object_right > enemy_left)
            overlap_y = (object_top < enemy_bottom) & (object_bottom > enemy_top)

            return overlap_x & overlap_y

        player_pos = jnp.array([player_x, player_y])
        player_hits = jax.vmap(
            lambda enemy_pos: object_hits_enemy(player_pos, PLAYER_SIZE, enemy_pos)
        )(updated_enemy_pos)
        
        hit_by_enemy = jnp.any(player_hits)
        hit_something = hit_by_enemy | hit_wall
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
            (bullets[:, 0] <= WIDTH) &
            (bullets[:, 1] >= 0) &
            (bullets[:, 1] <= HEIGHT)
        )

        # 6. Kollisionsabfrage (Kugeln vs Gegner)
        # Größe pro Bullet je nach Richtung auswählen
        bullet_sizes = jax.vmap(
            lambda d: jax.lax.select(
                d[0] == 0,  # vertikal wenn dx == 0
                jnp.array(BULLET_SIZE_VERTICAL, dtype=jnp.float32),
                jnp.array(BULLET_SIZE_HORIZONTAL, dtype=jnp.float32)
            )
        )(bullet_dirs)

        # Ergebnis: (NUM_ENEMIES, MAX_BULLETS)
        def bullet_hits_enemy(bullet_pos, bullet_size, enemy_pos):
            return object_hits_enemy(bullet_pos, bullet_size, enemy_pos)

        all_hits = jax.vmap(  # über Gegner
            lambda enemy_pos: jax.vmap(  # über Kugeln
                lambda bullet, size: bullet_hits_enemy(bullet, size, enemy_pos)
            )(bullets, bullet_sizes)
        )(updated_enemy_pos)

        enemy_hit = jnp.any(all_hits, axis=1)  # (NUM_ENEMIES,)
        enemy_alive = ~enemy_hit

        score_after = jnp.where(
            jnp.any(~enemy_alive),  # Prüft, ob mindestens ein Gegner getroffen wurde
            state.score + 50,
            state.score
        )

        # 7. Neue Gegnerposition/Bewegungsrichtung nur setzen, wenn Gegner lebt
        invisible = jnp.array([-100.0, -100.0])
        updated_enemy_pos = jnp.where(enemy_alive[:, None], updated_enemy_pos, invisible)

        updated_enemy_axis = jnp.where(enemy_alive, updated_enemy_axis, 0)

        # 8. Inaktive Kugeln durch Treffer
        bullet_hit = jnp.any(all_hits, axis=0)  # (MAX_BULLETS,)
        bullet_active = bullet_active & ~bullet_hit

        # 9. Neuer State
        new_state = BerzerkState(
            player_pos=new_pos,
            lives=lives_after,
            bullets=bullets,
            bullet_dirs=bullet_dirs,
            bullet_active=bullet_active,
            enemy_pos=updated_enemy_pos,
            enemy_move_axis=updated_enemy_axis,
            enemy_move_dir=updated_enemy_dir,
            enemy_move_prob = state.enemy_move_prob,
            last_dir=move_dir,
            rng=rng,
            score=score_after
        )

        # === Exit erreicht? Neues "Level" vorbereiten ===
        reset_for_new_level = BerzerkState(
            player_pos=jnp.array([WIDTH // 2, HEIGHT // 2], dtype=jnp.float32),
            lives=state.lives,
            bullets=jnp.zeros_like(state.bullets),
            bullet_dirs=jnp.zeros_like(state.bullet_dirs),
            bullet_active=jnp.zeros_like(state.bullet_active),
            enemy_pos=jax.random.uniform(
                state.rng, shape=(NUM_ENEMIES, 2),
                minval=jnp.array([PLAYER_BOUNDS[0][0], PLAYER_BOUNDS[1][0]]),
                maxval=jnp.array([
                    PLAYER_BOUNDS[0][1] - ENEMY_SIZE[0],
                    PLAYER_BOUNDS[1][1] - ENEMY_SIZE[1]
                ])
            ),
            enemy_move_axis=-jnp.ones_like(state.enemy_move_axis),
            enemy_move_dir=jnp.zeros_like(state.enemy_move_dir),
            enemy_move_prob=state.enemy_move_prob,
            last_dir=jnp.array([0.0, -1.0], dtype=jnp.float32),
            rng=rng,
            score=score_after
        )

        _, reset_state = self.reset(rng)

        jax.debug.print("EXIT: {}", hit_exit)
        new_state = jax.lax.cond(
            hit_exit,
            lambda: reset_for_new_level,
            lambda: jax.lax.cond(
                hit_something,
                lambda: reset_state._replace(lives=state.lives - 1, score=score_after),  # Tod
                lambda: new_state
            )
        )

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


class BerzerkRenderer(AtraJaxisRenderer):

    def __init__(self):
        super().__init__()
        self.sprites = self._load_sprites()

    def _load_sprites(self):
        """Load sprites for player, enemy and walls."""
        MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
        sprite_path = os.path.join(MODULE_DIR, "sprites/berzerk/")

        sprites: Dict[str, Any] = {}

        def _load_sprite(name: str) -> Optional[chex.Array]:
            path = os.path.join(sprite_path, f'{name}.npy')
            frame = aj.loadFrame(path)
            return frame.astype(jnp.uint8)

        sprite_names = [
            'player_idle', 'enemy_idle_1', 'level_outer_walls',
            'bullet_horizontal', 'bullet_vertical'
        ]

        for name in sprite_names:
            sprite = _load_sprite(name)
            if sprite is not None:
                sprites[name] = jnp.expand_dims(sprite, axis=0)

        return sprites

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        raster = jnp.zeros((HEIGHT * SCALING_FACTOR, WIDTH * SCALING_FACTOR, 3), dtype=jnp.uint8)

        # Draw walls (assuming fixed positions based on bounds)
        wall_sprite = aj.get_sprite_frame(self.sprites['level_outer_walls'], 0)

        # Outer wall
        raster = aj.render_at(raster, WALL_OFFSET[0], WALL_OFFSET[1], wall_sprite)

        # Draw player
        player_sprite = aj.get_sprite_frame(self.sprites['player_idle'], 0)
        raster = aj.render_at(raster, state.player_pos[0], state.player_pos[1], player_sprite)

        # Draw enemy
        enemy_sprite = aj.get_sprite_frame(self.sprites['enemy_idle_1'], 0)

        for i in range(state.enemy_pos.shape[0]):
            raster = aj.render_at(raster, state.enemy_pos[i][0], state.enemy_pos[i][1], enemy_sprite)

        # Draw bullets
        for i in range(state.bullets.shape[0]):
            is_active = state.bullet_active[i]
            bullet_pos = state.bullets[i]
            bullet_dir = state.bullet_dirs[i]

            def draw_bullet(raster):
                dx, dy = bullet_dir[0], bullet_dir[1]

                type_idx = jax.lax.select(dx != 0, 0, 1)  # 0=horizontal, 1=vertical

                def render_horizontal(r):
                    sprite = aj.get_sprite_frame(self.sprites['bullet_horizontal'], 0)
                    return aj.render_at(r, bullet_pos[0], bullet_pos[1], sprite)

                def render_vertical(r):
                    sprite = aj.get_sprite_frame(self.sprites['bullet_vertical'], 0)
                    return aj.render_at(r, bullet_pos[0], bullet_pos[1], sprite)

                return jax.lax.switch(
                    type_idx,
                    [render_horizontal, render_vertical],
                    raster
                )

            raster = jax.lax.cond(is_active, draw_bullet, lambda r: r, raster)

        return raster

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


    renderer_AtraJaxis = BerzerkRenderer()

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
                #if i % 3 == 0:
                #    x = int(curr_state.player_pos[0].item())
                #    y = int(curr_state.player_pos[1].item())
                #    pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(x - 5, y - 5, 10, 10))

                #    ex = int(curr_state.enemy_pos[0].item())
                #    ey = int(curr_state.enemy_pos[1].item())
                #    pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(ex - 5, ey - 5, 10, 10))
                pygame.display.flip()
                clock.tick(60)

            # Reset after death
            _, curr_state = jitted_reset()


        #render(screen, curr_state)
        raster = renderer_AtraJaxis.render(curr_state)
        aj.update_pygame(screen, raster, SCALING_FACTOR, WIDTH, HEIGHT)
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()