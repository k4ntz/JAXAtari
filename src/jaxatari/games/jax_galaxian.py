import os
from typing import NamedTuple, Tuple
import jax.numpy as jnp
import chex
import pygame
from functools import partial
from jax import lax
import jax.lax

from gymnax.environments import spaces

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import AtraJaxisRenderer
from jaxatari.rendering import atraJaxis as aj


"""
README README README README README README README 
Aaron Reinhardt
Aaron Weis
Leon Denis Kristof

Starting our game with direct call provides more features than the play.py call.
Game over, death and score display do not work in play.py call.

Float casting is a temporary solution and will be fixed.

direct call: python -m jaxatari.games.jax_galaxian
play.py call: python scripts/play.py --game src/jaxatari/games/jax_galaxian.py --record my_record_file.npz
"""



# -------- Game constants --------
SHOOTING_COOLDOWN = 80
ENEMY_MOVE_SPEED = 0.5
BULLET_MOVE_SPEED = 5
GRID_ROWS = 5
GRID_COLS = 5
NATIVE_GAME_WIDTH = 160
NATIVE_GAME_HEIGHT = 210
PYGAME_SCALE_FACTOR = 3
PYGAME_WINDOW_WIDTH = NATIVE_GAME_WIDTH * PYGAME_SCALE_FACTOR
PYGAME_WINDOW_HEIGHT = NATIVE_GAME_HEIGHT * PYGAME_SCALE_FACTOR
ENEMY_SPACING_X = 20
ENEMY_SPACING_Y = 12
ENEMY_GRID_Y = 70
START_X = NATIVE_GAME_WIDTH // 4
START_Y = NATIVE_GAME_HEIGHT
ENEMY_ATTACK_SPEED = 2
ENEMY_ATTACK_BULLET_SPEED = 5
ENEMY_ATTACK_BULLET_DELAY = 75
ENEMY_ATTACK_MAX_BULLETS = 2
LIVES = 3
PLAYER_DEATH_DELAY = 50
ENEMY_LEFT_BOUND = 17
ENEMY_RIGHT_BOUND = NATIVE_GAME_WIDTH - 25
DIVE_KILL_Y = 175
DIVE_SPEED = 0.5

class GalaxianState(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    player_shooting_cooldown: chex.Array
    bullet_x: chex.Array
    bullet_y: chex.Array
    enemy_grid_x: chex.Array
    enemy_grid_y: chex.Array
    enemy_grid_alive: chex.Array        # 0: dead, 1: alive, 2: attacking
    enemy_grid_direction: chex.Array
    enemy_attack_state: chex.Array      # 0: init, 1: attack, 2: respawn
    enemy_attack_pos: chex.Array
    enemy_attack_x: chex.Array
    enemy_attack_y: chex.Array
    enemy_attack_direction: chex.Array
    enemy_attack_target_x: chex.Array# -1: left, 1: right
    enemy_attack_target_y: chex.Array
    enemy_attack_respawn_timer: chex.Array
    enemy_attack_bullet_x: chex.Array
    enemy_attack_bullet_y: chex.Array
    enemy_attack_bullet_timer: chex.Array
    lives: chex.Array
    player_alive: chex.Array
    player_respawn_timer: chex.Array
    score: chex.Array
    step_counter: chex.Array


@jax.jit
def update_player_position(state: GalaxianState, action) -> GalaxianState:

    press_right = jnp.any(
        jnp.array([action == Action.RIGHT, action == Action.RIGHTFIRE])
    )

    press_left = jnp.any(
        jnp.array([action == Action.LEFT, action == Action.LEFTFIRE])
    )

    # rohe neue X-Position
    new_x = state.player_x + (press_right * 5) - (press_left * 5)

    # clamp zwischen 0 und (SCREEN_WIDTH − PLAYER_WIDTH)
    new_x = jnp.clip(new_x, 17, NATIVE_GAME_WIDTH - 25)
    return state._replace(player_x=new_x)

@jax.jit
def update_enemy_positions(state: GalaxianState) -> GalaxianState:

    new_x = state.enemy_grid_x + ENEMY_MOVE_SPEED * state.enemy_grid_direction

    # clamp auf [LEFT, RIGHT]
    new_x = jnp.clip(new_x, ENEMY_LEFT_BOUND, ENEMY_RIGHT_BOUND)

    # Rand-Bounce wie gehabt
    hit_left = jnp.any(new_x <= ENEMY_LEFT_BOUND)
    hit_right = jnp.any(new_x >= ENEMY_RIGHT_BOUND)
    new_dir = jnp.where(hit_left, 1,
                        jnp.where(hit_right, -1, state.enemy_grid_direction))

    return state._replace(
        enemy_grid_x=new_x,
        enemy_grid_direction=new_dir
    )



@jax.jit
def update_enemy_attack(state: GalaxianState) -> GalaxianState:

    def pick_enemy(state):
        def body(idx, carry):
            found, pos = carry
            i, j = idx // GRID_COLS, idx % GRID_COLS
            alive = state.enemy_grid_alive[i, j] == 1
            new_found = found | alive
            new_pos   = jnp.where(alive & ~found,
                                  jnp.array((i, j), jnp.int32),
                                  pos)
            return new_found, new_pos

        _, pos = lax.fori_loop(
            0, GRID_ROWS * GRID_COLS,
            body,
            (False, jnp.array((0, 0), jnp.int32))
        )
        return pos
    # Prüfe ob noch mindestens ein Gegner lebt
    any_alive = jnp.any(state.enemy_grid_alive == 1)

    # Start‐Attack
    def start_attack(state):
        pos = pick_enemy(state)
        x0, y0 = state.enemy_grid_x[tuple(pos)], state.enemy_grid_y[tuple(pos)]
        return state._replace(
            enemy_attack_state        = jnp.array(1),
            enemy_attack_pos          = pos,
            enemy_attack_x            = x0,
            enemy_attack_y            = y0,
            enemy_attack_target_x     = state.player_x,               # Ziel‐X bleibt der Spieler
            enemy_attack_target_y     = jnp.array(NATIVE_GAME_HEIGHT, dtype=jnp.float32), # Ziel‐Y jetzt Boden
            enemy_grid_alive          = state.enemy_grid_alive.at[tuple(pos)].set(2),
            enemy_attack_respawn_timer= jnp.array(ENEMY_ATTACK_BULLET_DELAY),
            enemy_attack_bullet_x     = jnp.array(-1.0, dtype=jnp.float32),
            enemy_attack_bullet_y     = jnp.array(-1.0, dtype=jnp.float32),
            enemy_attack_bullet_timer = jnp.array(ENEMY_ATTACK_BULLET_DELAY),
        )

    state1 = lax.cond(
        (state.enemy_attack_state == 0) & any_alive,
        start_attack,
        lambda state: state,
        state
    )

    # Do Dive
    def do_dive(state):
        dx   = state.enemy_attack_target_x - state.enemy_attack_x
        dist = jnp.sqrt(dx**2 + 1.0) + 1e-6
        velocity_x   = dx / dist * DIVE_SPEED
        velocity_y   = DIVE_SPEED  # konstant nach unten

        x1 = jnp.clip(state.enemy_attack_x + velocity_x, ENEMY_LEFT_BOUND, ENEMY_RIGHT_BOUND)
        y1 = state.enemy_attack_y + velocity_y
        return state._replace(enemy_attack_x=x1, enemy_attack_y=y1)
    state2 = lax.cond(state1.enemy_attack_state == 1,
                  do_dive,
                  lambda state: state,
                  state1)

    # Auto‐Kill sobald unter DIVE_KILL_Y
    def kill(state):
        pos      = state.enemy_attack_pos
        new_grid = state.enemy_grid_alive.at[tuple(pos)].set(0)
        return state._replace(
            enemy_attack_state = jnp.array(0),
            enemy_grid_alive   = new_grid
        )
    state3 = lax.cond(state2.enemy_attack_y > DIVE_KILL_Y,
                  kill,
                  lambda state: state,
                  state2)

    # Respawn‐Timer (State 2)
    timer4 = jnp.where(state3.enemy_attack_state == 2,
                       state3.enemy_attack_respawn_timer - 1,
                       state3.enemy_attack_respawn_timer)
    state4 = state3._replace(enemy_attack_respawn_timer=timer4)

    # Nach Ablauf → zurück auf State 0
    state5 = lax.cond(
        (state4.enemy_attack_state == 2) & (state4.enemy_attack_respawn_timer <= 0),
        lambda state: state._replace(enemy_attack_state=jnp.array(0)),
        lambda state: state,
        state4
    )

    # Bullet‐Timer & Schuss-Logik im Dive-State
    bt        = jnp.where(state5.enemy_attack_state == 1,
                          state5.enemy_attack_bullet_timer - 1,
                          state5.enemy_attack_bullet_timer)
    can_shoot = (state5.enemy_attack_state == 1) & (bt <= 0) & (state5.enemy_attack_bullet_y < 0)
    new_bx    = jnp.where(can_shoot, state5.enemy_attack_x, state5.enemy_attack_bullet_x)
    new_by    = jnp.where(can_shoot, state5.enemy_attack_y, state5.enemy_attack_bullet_y)
    bt        = jnp.where(can_shoot, ENEMY_ATTACK_BULLET_DELAY, bt)

    return state5._replace(
        enemy_attack_bullet_x     = new_bx,
        enemy_attack_bullet_y     = new_by,
        enemy_attack_bullet_timer = bt,
    )

def update_enemy_bullets(state: GalaxianState) -> GalaxianState:

    bullet_available = (state.enemy_attack_bullet_timer == 0) & (state.enemy_attack_bullet_y == -1)
    new_enemy_attack_bullet_x = jnp.where(
        bullet_available,
        state.enemy_attack_x,
        state.enemy_attack_bullet_x
    )
    new_enemy_attack_bullet_y = jnp.where(
        bullet_available,
        state.enemy_attack_y,
        state.enemy_attack_bullet_y
    )

    bullet_exists = (state.enemy_attack_bullet_y != -1)
    new_enemy_attack_bullet_y = jnp.where(
        bullet_exists,
        state.enemy_attack_bullet_y + ENEMY_ATTACK_BULLET_SPEED,
        new_enemy_attack_bullet_y
    )

    bullet_out_of_bounds = (state.enemy_attack_bullet_y > NATIVE_GAME_HEIGHT)
    new_enemy_attack_bullet_x = jnp.where(
        bullet_out_of_bounds,
        -1,
        new_enemy_attack_bullet_x
    )
    new_enemy_attack_bullet_y = jnp.where(
        bullet_out_of_bounds,
        -1,
        new_enemy_attack_bullet_y
    )

    return state._replace(
        enemy_attack_bullet_x=new_enemy_attack_bullet_x,
        enemy_attack_bullet_y=new_enemy_attack_bullet_y,
    )



def get_action_from_keyboard():
    keys = pygame.key.get_pressed()
    left = keys[pygame.K_a] or keys[pygame.K_LEFT]
    right = keys[pygame.K_d] or keys[pygame.K_RIGHT]
    shooting = keys[pygame.K_SPACE]

    left_only = left and not right
    right_only = right and not left

    if shooting:
        if left_only:
            return Action.LEFTFIRE
        elif right_only:
            return Action.RIGHTFIRE
        else:
            return Action.FIRE
    else:
        if left_only:
            return Action.LEFT
        elif right_only:
            return Action.RIGHT
        else:
            return Action.NOOP



@jax.jit
def handle_shooting(state: GalaxianState, action) -> GalaxianState:
    def shoot(_):
        return state._replace(
            bullet_x=jnp.array(state.player_x, dtype=state.bullet_x.dtype),
            bullet_y=jnp.array(state.player_y, dtype=state.bullet_y.dtype),
            player_shooting_cooldown=jnp.array(SHOOTING_COOLDOWN)
        )

    def decrease_cooldown(_):
        return state._replace(
            player_shooting_cooldown=state.player_shooting_cooldown - 1
        )

    def idle(_):
        return state


    shooting = jnp.any(
        jnp.array(
            [
                action == Action.FIRE,
                action == Action.RIGHTFIRE,
                action == Action.LEFTFIRE,
            ]
        )
    )

    can_shoot = jnp.logical_and(shooting, state.player_shooting_cooldown == 0)
    on_cooldown = jnp.logical_not(can_shoot) & (state.player_shooting_cooldown > 0)

    return lax.cond(
        can_shoot,
        shoot,
        lambda _: lax.cond(on_cooldown, decrease_cooldown, idle, operand=None),
        operand=None
    )


@jax.jit
def update_bullets(state: GalaxianState) -> GalaxianState:
    new_bullets_y = jnp.array(state.bullet_y - BULLET_MOVE_SPEED)
    return state._replace(bullet_y=new_bullets_y)


@jax.jit
def remove_bullets(state: GalaxianState) -> GalaxianState:
    # Wenn die Spieler-Kugel den Bildschirm (y<0) verlässt, reset Kugel + Cooldown
    def bullet_reset(state: GalaxianState) -> GalaxianState:
        return state._replace(
            bullet_x                = jnp.array(-1.0, dtype=state.bullet_x.dtype),
            bullet_y                = jnp.array(-1.0, dtype=state.bullet_y.dtype),
        )
    def keep(state: GalaxianState) -> GalaxianState:
        return state

    return lax.cond(state.bullet_y < 0,
                    bullet_reset,
                    keep,
                    state)


@jax.jit
def bullet_collision(state: GalaxianState) -> GalaxianState:
    x_diff = jnp.abs(state.bullet_x - state.enemy_grid_x)
    y_diff = jnp.abs(state.bullet_y - state.enemy_grid_y)

    mask = (x_diff <= 10) & (y_diff <= 10) & (state.enemy_grid_alive == 1)
    hit  = jnp.any(mask)

    def process_hit(s_and_idx):
        s, (rows, cols) = s_and_idx
        # setze alle getroffenen auf dead
        new_alive = state.enemy_grid_alive.at[rows, cols].set(0)
        return state._replace(
            enemy_grid_alive         = new_alive,
            bullet_x                 = jnp.array(-1.0, dtype=s.bullet_x.dtype),
            bullet_y                 = jnp.array(-1.0, dtype=s.bullet_y.dtype),
            score                    = s.score + 30,
        )

    def no_hit(state_and_idx):
        state, _ = state_and_idx
        return state

    max_size = state.enemy_grid_alive.size
    collision_indices = jnp.where(mask, size=max_size, fill_value=-1)
    hit_rows = collision_indices[0]
    hit_cols = collision_indices[1]

    return lax.cond(hit,
                    process_hit,
                    no_hit,
                    (state, (hit_rows, hit_cols)))

@jax.jit
def bullet_collision_attack(state: GalaxianState) -> GalaxianState:
    # Abstände
    x_diff = jnp.abs(state.bullet_x - state.enemy_attack_x)
    y_diff = jnp.abs(state.bullet_y - state.enemy_attack_y)
    # Kollisionsmaske: innerhalb 10px und aktuell angreifend (state 1)
    hit = jnp.any((x_diff <= 10) & (y_diff <= 10) & (state.enemy_attack_state == 1))

    def process_hit(state: GalaxianState) -> GalaxianState:
        # Als getötet markieren
        pos      = state.enemy_attack_pos
        new_grid = state.enemy_grid_alive.at[tuple(pos)].set(0)
        # Score erhöhen und Angreifer zurücksetzen
        return state._replace(
            enemy_grid_alive       = new_grid,
            bullet_x               = jnp.array(-1.0, dtype=state.bullet_x.dtype),
            bullet_y               = jnp.array(-1.0, dtype=state.bullet_y.dtype),
            enemy_attack_state     = jnp.array(0),                  # zurück auf State 0
            score                  =state.score + jnp.array(50, dtype=state.score.dtype)  # z.B. 50 Punkte
        )

    # Kein Hit → unverändert zurückgeben
    def no_hit(state: GalaxianState) -> GalaxianState:
        return state

    return lax.cond(hit, process_hit, no_hit, state)

@jax.jit
def check_player_death_by_enemy(state: GalaxianState) -> GalaxianState:

    x_diff = jnp.abs(state.player_x - state.enemy_attack_x)
    y_diff = jnp.abs(state.player_y - state.enemy_attack_y)

    collision = (x_diff <= 10) & (y_diff <= 10) & (state.enemy_grid_alive[tuple(state.enemy_attack_pos)] != 0)
    hit = jnp.any(collision)

    def process_hit(operands):
        current_state = operands
        new_lives = current_state.lives - 1
        new_enemy_grid_alive = current_state.enemy_grid_alive.at[tuple(current_state.enemy_attack_pos)].set(0)

        return state._replace(lives=new_lives, enemy_grid_alive=new_enemy_grid_alive)

    return lax.cond(hit, process_hit, lambda _: state, operand=state)

def check_player_death_by_bullet(state: GalaxianState) -> GalaxianState:

    x_diff = jnp.abs(state.player_x - state.enemy_attack_bullet_x)
    y_diff = jnp.abs(state.player_y - state.enemy_attack_bullet_y)

    collision = (x_diff <= 10) & (y_diff <= 10)
    hit = jnp.any(collision)

    def process_hit(operands):
        current_state = operands
        new_lives = current_state.lives - 1
        new_enemy_attack_bullet_x = jnp.array(-1,dtype=jnp.float32)
        new_enemy_attack_bullet_y = jnp.array(-1,dtype=jnp.float32)

        return state._replace(lives=new_lives, enemy_attack_bullet_x=new_enemy_attack_bullet_x,enemy_attack_bullet_y=new_enemy_attack_bullet_y)

    return lax.cond(hit, process_hit, lambda _: state, operand=state)


class GalaxianObservation(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    bullet_x: chex.Array
    bullet_y: chex.Array
    enemy_grid_x: chex.Array
    enemy_grid_y: chex.Array
    enemy_grid_alive: chex.Array  # 0: dead, 1: alive, 2: attacking
    enemy_attack_pos: chex.Array
    enemy_attack_x: chex.Array
    enemy_attack_y: chex.Array

class GalaxianInfo(NamedTuple):
    time: jnp.ndarray
    all_rewards: chex.Array

class JaxGalaxian(JaxEnvironment[GalaxianState, GalaxianObservation, GalaxianInfo]):
    def __init__(self, frameskip: int = 0, reward_funcs: list[callable]=None):
        super().__init__()
        self.frameskip = frameskip + 1  #den stuff kp hab copy paste aus pong
        self.frame_stack_size = 4
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.action_set = {
            Action.NOOP,
            Action.FIRE,
            Action.RIGHT,
            Action.LEFT,
            Action.RIGHTFIRE,
            Action.LEFTFIRE
        }
        self.obs_size = 3*2+1
        #Größe der Beobachtungsdaten
        #3*4 = 3 entitys mit je 2 werten: enemy, player, bullet -> x,y
        # +1 ein extra wert für den score
        # sind erstmal nur temporary values

    def reset(self, key = None) -> Tuple[GalaxianObservation, GalaxianState]:
        grid_rows = GRID_ROWS
        grid_cols = GRID_COLS
        enemy_spacing_x = ENEMY_SPACING_X
        start_x = START_X

        x_positions = jnp.arange(grid_cols) * enemy_spacing_x + start_x  # arange schreibt so 0 1 2 3....
        enemy_grid = jnp.tile(x_positions, (grid_rows, 1))  # kopiert die zeile untereinander

        row_indices = jnp.arange(grid_rows).reshape(-1, 1)  # erzeugt 1. enemy jeder zeile
        enemy_y_rows = ENEMY_GRID_Y - row_indices * ENEMY_SPACING_Y  # jeweils y pos
        enemy_grid_y = jnp.broadcast_to(enemy_y_rows, (
            grid_rows, grid_cols))  # kopiert die werte für 1. enemy auf die rechts in der zeile

        enemy_alive = jnp.ones((grid_rows, grid_cols))  # alles auf 1



        state = GalaxianState(player_x=jnp.array(NATIVE_GAME_WIDTH / 2.0),
                              player_y=jnp.array(NATIVE_GAME_HEIGHT - 40.0),
                              player_shooting_cooldown=jnp.array(0),
                              bullet_x=jnp.array(-1.0,dtype=jnp.float32),
                              bullet_y=jnp.array(-1.0,dtype=jnp.float32),
                              enemy_grid_x=enemy_grid.astype(jnp.float32),
                              enemy_grid_y=enemy_grid_y.astype(jnp.float32),
                              enemy_grid_alive=enemy_alive,
                              enemy_grid_direction=jnp.array(1),
                              enemy_attack_state=jnp.array(0),
                              enemy_attack_pos=jnp.array((-1, -1)),
                              enemy_attack_direction=jnp.array(1),
                              enemy_attack_x=jnp.array(-1.0, dtype=jnp.float32),
                              enemy_attack_y=jnp.array(-1.0, dtype=jnp.float32),
                              enemy_attack_respawn_timer=jnp.array(20),
                              enemy_attack_bullet_x=jnp.array(-1, dtype=jnp.float32),
                              enemy_attack_bullet_y=jnp.array(-1, dtype=jnp.float32),
                              enemy_attack_bullet_timer=jnp.array(ENEMY_ATTACK_BULLET_DELAY),
                              lives=jnp.array(3),
                              player_alive=jnp.array(True),
                              player_respawn_timer=jnp.array(PLAYER_DEATH_DELAY),
                              score=jnp.array(0, dtype=jnp.int32),
                              enemy_attack_target_x=jnp.array(-1.0, dtype=jnp.float32),
                              enemy_attack_target_y=jnp.array(-1.0, dtype=jnp.float32),
                              step_counter=jnp.array(0),
                             )

        initial_obs = self._get_observation(state)
        return initial_obs, state

        #alles aus galaxianState was man aus Spielersicht wahrnimmt
    def _get_observation(self, state: GalaxianState) -> GalaxianObservation:
        return GalaxianObservation(
            player_x=state.player_x,
            player_y=state.player_y,
            bullet_x=state.bullet_x,
            bullet_y=state.bullet_y,
            enemy_grid_x=state.enemy_grid_x,
            enemy_grid_y=state.enemy_grid_y,
            enemy_grid_alive=state.enemy_grid_alive,
            enemy_attack_pos=state.enemy_attack_pos,
            enemy_attack_x=state.enemy_attack_x,
            enemy_attack_y=state.enemy_attack_y,
        )

    @partial(jax.jit, static_argnums=(0,))
    def get_action_space(self):
        return jnp.array([Action.NOOP, Action.LEFT, Action.RIGHT, Action.FIRE, Action.LEFTFIRE, Action.RIGHTFIRE])


    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))

    def observation_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=None,
            dtype=jnp.uint8,
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(
            self, state: GalaxianState, action: chex.Array
    ) -> Tuple[GalaxianObservation, GalaxianState, float, bool, GalaxianInfo]:
        #TODO: refactor like the other games
        # create new state instead of replacing old state step by step
        new_state = update_player_position(state, action)
        new_state = handle_shooting(new_state, action)
        new_state = update_enemy_positions(new_state)
        new_state = update_bullets(new_state)
        new_state = remove_bullets(new_state)
        new_state = bullet_collision(new_state)
        new_state = bullet_collision_attack(new_state)
        new_state = update_enemy_attack(new_state)
        new_state = update_enemy_bullets(new_state)
        new_state = check_player_death_by_enemy(new_state)
        new_state = check_player_death_by_bullet(new_state)
        new_state = update_enemy_attack(new_state)
        new_state = new_state._replace(step_counter=new_state.step_counter + 1)

        done = self._get_done(new_state)
        env_reward = self._get_env_reward(state, new_state)
        all_rewards = self._get_all_reward(state, new_state)
        info = self._get_info(new_state, all_rewards)

        observation = self._get_observation(new_state)

        return observation, new_state, env_reward, done, info


    @partial(jax.jit, static_argnums=(0,))
    def _get_env_reward(self, previous_state: GalaxianState, state: GalaxianState):
        return state.score - previous_state.score


    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(self, previous_state: GalaxianState, state: GalaxianState) -> chex.Array:
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array(
            [reward_func(previous_state, state) for reward_func in self.reward_funcs]
        )
        return rewards

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: GalaxianState) -> bool:
        # Game Over, wenn alle Leben weg sind
        no_lives = state.lives <= 0
        # You Win, wenn **keine** lebenden Enemies (==1) mehr da sind
        no_enemies = jnp.all(state.enemy_grid_alive == 0)
        return jnp.logical_or(no_lives, no_enemies)

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: GalaxianState, all_rewards: chex.Array) -> GalaxianInfo:
        return GalaxianInfo(time=state.step_counter, all_rewards=all_rewards)


def load_sprites():
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

    # load sprites
    bg = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/galaxian/background.npy"),transpose=True)
    player = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/galaxian/player.npy"),transpose=True)
    bullet = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/galaxian/bullet.npy"),transpose=True)
    enemy_gray = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/galaxian/gray_enemy_1.npy"),transpose=True)
    life = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/galaxian/life.npy"),transpose=True)
    enemy_bullet = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/galaxian/enemy_bullet.npy"),transpose=True)
    SPRITE_BG = jnp.expand_dims(bg, axis= 0)
    SPRITE_PLAYER = jnp.expand_dims(player, axis = 0)
    SPRITE_BULLET = jnp.expand_dims(bullet, axis = 0)
    SPRITE_ENEMY_GRAY = jnp.expand_dims(enemy_gray, axis=0)
    SPRITE_LIFE = jnp.expand_dims(life, axis=0)
    SPRITE_ENEMY_BULLET = jnp.expand_dims(enemy_bullet, axis=0)
    return(
        SPRITE_BG,
        SPRITE_PLAYER,
        SPRITE_BULLET,
        SPRITE_ENEMY_GRAY,
        SPRITE_LIFE,
        SPRITE_ENEMY_BULLET
    )

class GalaxianRenderer(AtraJaxisRenderer):
    def __init__(self):
        (
            self.SPRITE_BG,
            self.SPRITE_PLAYER,
            self.SPRITE_BULLET,
            self.SPRITE_ENEMY_GRAY,
            self.SPRITE_LIFE,
            self.SPRITE_ENEMY_BULLET,
        ) = load_sprites()

        # Sprite-Dimensionen für Life-Icons
        life_frame = jnp.squeeze(self.SPRITE_LIFE, axis=0)  # (h, w, 4)
        self.life_h, self.life_w, _ = life_frame.shape
        self.life_spacing = 5

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: GalaxianState):
        # Hintergrund
        raster = jnp.zeros((NATIVE_GAME_WIDTH, NATIVE_GAME_HEIGHT, 3), dtype=jnp.uint8)
        bg_frame = aj.get_sprite_frame(self.SPRITE_BG, 0)
        raster = aj.render_at(raster, 0, 0, bg_frame)

        # Spieler
        player_frame = aj.get_sprite_frame(self.SPRITE_PLAYER, 0)
        px = jnp.round(state.player_x).astype(jnp.int32)
        py = jnp.round(state.player_y).astype(jnp.int32)
        raster = aj.render_at(raster, px, py, player_frame)

        # Spieler-Kugel
        def draw_bullet(r):
            b = aj.get_sprite_frame(self.SPRITE_BULLET, 0)
            bx = jnp.round(state.bullet_x).astype(jnp.int32)
            by = jnp.round(state.bullet_y).astype(jnp.int32)
            return aj.render_at(r, bx, by, b)
        raster = lax.cond(state.bullet_y > 0, draw_bullet, lambda r: r, raster)

        # Feind-Kugel
        def draw_enemy_bullet(r):
            eb = aj.get_sprite_frame(self.SPRITE_ENEMY_BULLET, 0)
            ex = jnp.round(state.enemy_attack_bullet_x).astype(jnp.int32)
            ey = jnp.round(state.enemy_attack_bullet_y).astype(jnp.int32)
            return aj.render_at(r, ex, ey, eb)
        raster = lax.cond(state.enemy_attack_bullet_y >= 0, draw_enemy_bullet, lambda r: r, raster)

        # Angreifender Feind
        def draw_attacker(r):
            e = aj.get_sprite_frame(self.SPRITE_ENEMY_GRAY, 0)
            ex = jnp.round(state.enemy_attack_x).astype(jnp.int32)
            ey = jnp.round(state.enemy_attack_y).astype(jnp.int32)
            return aj.render_at(r, ex, ey, e)
        raster = lax.cond(jnp.all(state.enemy_attack_pos >= 0), draw_attacker, lambda r: r, raster)

        # Feindgitter
        def row_body(i, r_acc):
            def col_body(j, r_inner):
                e = aj.get_sprite_frame(self.SPRITE_ENEMY_GRAY, 0)
                cond = state.enemy_grid_alive[i, j] == 1
                def draw(r0):
                    x = jnp.round(state.enemy_grid_x[i, j]).astype(jnp.int32)
                    y = jnp.round(state.enemy_grid_y[i, j]).astype(jnp.int32)
                    return aj.render_at(r0, x, y, e)
                return lax.cond(cond, draw, lambda r0: r0, r_inner)
            return lax.fori_loop(0, GRID_COLS, col_body, r_acc)
        raster = lax.fori_loop(0, GRID_ROWS, row_body, raster)

        # Lebens-Icons unten rechts
        life_sprite = aj.get_sprite_frame(self.SPRITE_LIFE, 0)
        def life_loop_body(i, r_acc):
            # nur zeichnen, wenn Leben vorhanden
            def draw(r0):
                x0 = jnp.int32(
                    NATIVE_GAME_WIDTH - (i + 1) * (self.life_w + self.life_spacing)
                )
                y0 = jnp.int32(
                    NATIVE_GAME_HEIGHT - self.life_h - self.life_spacing
                )
                return aj.render_at(r0, x0, y0, life_sprite)
            return lax.cond(i < state.lives, draw, lambda r0: r0, r_acc)
        raster = lax.fori_loop(0, LIVES, life_loop_body, raster)

        return raster

# run with: python -m jaxatari.games.jax_galaxian
# run with: python scripts/play.py --game src/jaxatari/games/jax_galaxian.py --record my_record_file.npz
if __name__ == "__main__":
    pygame.init()
    font = pygame.font.Font(None, 24)

    game = JaxGalaxian(frameskip=1)
    jitted_step = jax.jit(game.step)
    jitted_reset = jax.jit(game.reset)
    initial_observation, state  = jitted_reset()

    screen = pygame.display.set_mode((PYGAME_WINDOW_WIDTH, PYGAME_WINDOW_HEIGHT))
    pygame.display.set_caption("Galaxian")
    clock = pygame.time.Clock()

    renderer = GalaxianRenderer()

    initial_observation, state = jax.jit(game.reset)()

    jitted_step = game.step

    running = True
    while running:
        action_int = Action.NOOP
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.fill((0, 0, 0))
        pygame.display.flip()
        action = get_action_from_keyboard()
        observation, state, reward, done, info = jitted_step(state, action)

        render_output = renderer.render(state)
        aj.update_pygame(screen, render_output, PYGAME_SCALE_FACTOR, NATIVE_GAME_WIDTH,
                         NATIVE_GAME_HEIGHT)
        score_surf = font.render(f"Score: {int(state.score)}", True, (255, 255, 255))
        screen.blit(score_surf, (10, 10))

        font = pygame.font.Font(None, 24)

        # Score
        score_surf = font.render(f"Score: {int(state.score)}", True, (255, 255, 255))
        screen.blit(score_surf, (10, 10))

        pygame.display.flip()
        if done:
            # Win vs. Game Over
            if int(state.lives) > 0:
                text = "You Win!"
            else:
                text = "Game Over!"
            msg = font.render(text, True, (255, 255, 255))
            x = (PYGAME_WINDOW_WIDTH - msg.get_width()) // 2
            y = (PYGAME_WINDOW_HEIGHT - msg.get_height()) // 2
            screen.blit(msg, (x, y))
            pygame.display.flip()
            pygame.time.wait(2000)
            break
        clock.tick(30)

    pygame.quit()




