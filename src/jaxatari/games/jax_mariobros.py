import os
from functools import partial
from typing import NamedTuple, Tuple, Dict, Any, Optional
import jax
import jax.numpy as jnp
import chex
import pygame
from jax import Array
from gymnax.environments import spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action

# --- Screen params---
SCREEN_WIDTH = 160
SCREEN_HEIGHT = 210
WINDOW_SCALE = 3

# --- Physik params ---
MOVE_SPEED = 1
ASCEND_VY     = -2.0         # ↑ 2 px / frame
DESCEND_VY    =  2.0         # ↓ 2 px / frame
ASCEND_FRAMES = 21         # 42 px tall jump (21 × 2)

# -------- Movement params ----------------------------------
movement_pattern        = jnp.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0],dtype=jnp.float32)
pat_len        = movement_pattern.shape[0]


# -------- Break params ----------------------------------
BRAKE_DURATION = 10           # in Frames
BRAKE_TOTAL_DISTANCE = 7.0    # in Pixels
BRAKE_SPEED = jnp.array(BRAKE_TOTAL_DISTANCE / BRAKE_DURATION, dtype=jnp.float32)  # ≈ 0.7 px/frame

# --- Player/Enemy/Plattform params ---
PLAYER_SIZE = (9, 21)  # w, h
PLAYER_COLOR = (181, 83, 40)
PLAYER_START_X, PLAYER_START_Y = 37.0, 74.0
ENEMY_SIZE = (8, 8)  # w, h

PLATFORMS = jnp.array([
    [0, 168, 160, 24],  # Ground x, y, w, h
    [0, 57, 64, 3],     # 3.FLoor Left
    [96, 57, 68, 3],    # 3.Floor Right
    [31, 95, 97, 3],    # 2.Floor Middle
    [0, 95, 16, 3],     # 2.Floor Left
    [144, 95, 18, 3],   # 2.Floor Right
    [0, 135, 48, 3],    # 1.Floor Left
    [112, 135, 48, 3]   # 1.Floor Right
])

# --- Pow_Block ---
POW_BLOCK = jnp.array([[72, 135, 16, 7]])  # x, y, w, h

class GameState(NamedTuple):    #Enemy movement
    enemy_pos: jnp.ndarray
    enemy_vel: jnp.ndarray

class PlayerState(NamedTuple):  # Player movement
    pos: jnp.ndarray
    vel: jnp.ndarray
    on_ground: bool
    jump_phase: chex.Array
    ascend_frames: chex.Array
    idx_right: chex.Array
    idx_left: chex.Array
    jump: chex.Array
    move: chex.Array
    jumpL: chex.Array
    jumpR: chex.Array
    last_dir: chex.Array
    brake_frames_left: chex.Array

class MarioBrosState(NamedTuple):
    player: PlayerState
    game: GameState
    lives: chex.Array

class MarioBrosObservation(NamedTuple): # Copied from jax_kangaroo.py ln.166-168
    player_x: chex.Array
    player_y: chex.Array

class MarioBrosInfo(NamedTuple):    # Copied from jax_kangaroo.py ln.186-187
    score: chex.Array

def check_collision(pos: jnp.ndarray, vel: jnp.ndarray, platforms: jnp.ndarray, pow_block: jnp.ndarray):
    # checks Player collision with all Platforms or Pow_Block

    x, y = pos
    vx, vy = vel
    w, h = PLAYER_SIZE

    left, right = x, x + w
    top, bottom = y, y + h

    # Platforms
    px, py, pw, ph = platforms[:, 0], platforms[:, 1], platforms[:, 2], platforms[:, 3]
    p_left, p_right = px, px + pw
    p_top, p_bottom = py, py + ph

    overlap_x = (right > p_left) & (left < p_right)
    overlap_y = (bottom > p_top) & (top < p_bottom)
    collided = overlap_x & overlap_y

    landed = collided & (vy > 0) & (bottom - vy <= p_top)
    bumped = collided & (vy < 0) & (top - vy >= p_bottom)

    # Höhenkorrektur
    landing_y = jnp.where(landed, p_top - h, jnp.inf)
    bumping_y = jnp.where(bumped, p_bottom, -jnp.inf)
    new_y_land = jnp.min(landing_y)
    new_y_bump = jnp.max(bumping_y)

    # POW block (only bump from below)
    pow_x, pow_y, pow_w, pow_h = pow_block[0]
    pow_left, pow_right = pow_x, pow_x + pow_w
    pow_top, pow_bottom = pow_y, pow_y + pow_h

    pow_overlap_x = (right > pow_left) & (left < pow_right)
    pow_hit_from_below = pow_overlap_x & (vy < 0) & (top - vy >= pow_bottom) & (top <= pow_bottom)
    pow_bump_y = jnp.where(pow_hit_from_below, pow_bottom, -jnp.inf)

    pow_bumped = pow_hit_from_below
    pow_y_new = jnp.max(pow_bump_y)

    return jnp.any(landed), jnp.any(bumped | pow_bumped), new_y_land, jnp.maximum(new_y_bump, pow_y_new), pow_bumped

def check_enemy_collision(player_pos, enemy_pos):
    # checks Player collision with enemys

    px, py = player_pos
    pw, ph = PLAYER_SIZE
    ex, ey = enemy_pos[:, 0], enemy_pos[:, 1]
    ew, eh = ENEMY_SIZE

    overlap_x = (px < ex + ew) & (px + pw > ex)
    overlap_y = (py < ey + eh) & (py + ph > ey)
    return jnp.any(overlap_x & overlap_y)

@jax.jit
def movement(state: PlayerState, action: jnp.ndarray) -> PlayerState:
    # Calculates movement of Player based on given state and action taken

    move, jump_btn = action[0], action[1].astype(jnp.int32)
    vx = MOVE_SPEED * move
    # -------- phase / frame bookkeeping --------------------------
    start_jump = (jump_btn == 1) & state.on_ground & (state.jump_phase == 0)

    jump_phase = jnp.where(start_jump, 1, state.jump_phase)
    asc_left   = jnp.where(start_jump, ASCEND_FRAMES, state.ascend_frames)

    # vertical speed for this frame
    vy = jnp.where(
            jump_phase == 1, ASCEND_VY,
            jnp.where(jump_phase == 2, DESCEND_VY,
                      jnp.where(state.on_ground, 0.0, DESCEND_VY))
         )

    # integrate position
    new_pos = state.pos + jnp.array([vx, vy])

    landed, bumped, y_land, y_bump, pow_hit = check_collision(new_pos, jnp.array([vx, vy]), PLATFORMS, POW_BLOCK)

    new_y = jnp.where(landed, y_land,
              jnp.where(bumped, y_bump, new_pos[1]))

    # ---------- update phases after collision & time -------------
    # decrement ascend frames while ascending
    asc_left = jnp.where(jump_phase == 1, jnp.maximum(asc_left - 1, 0), asc_left)
    # switch to descend when ascent finished
    jump_phase = jnp.where((jump_phase == 1) & (asc_left == 0), 2, jump_phase)
    # head bump → descend immediately
    jump_phase = jnp.where(bumped & (vy < 0), 2, jump_phase)
    asc_left   = jnp.where(bumped & (vy < 0), 0, asc_left)
    # landing → reset
    jump_phase = jnp.where(landed, 0, jump_phase)
    asc_left   = jnp.where(landed, 0, asc_left)
    # walked off ledge → fall
    jump_phase = jnp.where((jump_phase == 0) & (~landed), 2, jump_phase)

    vy_final = jnp.where(
        jump_phase == 1, ASCEND_VY,
        jnp.where(jump_phase == 2, DESCEND_VY, 0.0)
    )

    new_x = jnp.clip(new_pos[0], 0, SCREEN_WIDTH - PLAYER_SIZE[0])

    return PlayerState(
        pos=jnp.array([new_x, new_y]),
        vel=jnp.array([vx, vy_final]),
        on_ground=landed,
        jump_phase=jump_phase.astype(jnp.int32),
        ascend_frames=asc_left.astype(jnp.int32),
        idx_right=state.idx_right,
        idx_left=state.idx_left,
        jump=state.jump,
        move=state.move,
        jumpL=state.jumpL,
        jumpR=state.jumpR,
        last_dir=state.last_dir,
        brake_frames_left=state.brake_frames_left
    )

def player_step(state: PlayerState, action: chex.Array):
    # Calculates next state based on given PlayerState and action taken

    # Determine presses
    press_fire = jnp.logical_or(
        action == Action.FIRE,
        jnp.logical_or(action == Action.LEFTFIRE, action == Action.RIGHTFIRE)
    )

    # resets movement after jump for new movement callculations
    def reset_ground(s):
        return s._replace(move=0.0, jump=0, jumpL=False, jumpR=False)

    state = jax.lax.cond(state.on_ground, reset_ground, lambda s: s, state)

    press_right = jnp.logical_or(action == Action.RIGHT, action == Action.RIGHTFIRE)
    press_left = jnp.logical_or(action == Action.LEFT, action == Action.LEFTFIRE)

    # Jump logic
    state = state._replace(jump=jnp.where(press_fire, 1, state.jump))

    def walk_or_brake(s):
        # on ground and not jumping
        def move_right(ss):
            return ss._replace(
                move=movement_pattern[ss.idx_right],
                idx_right=(ss.idx_right + 1) % pat_len,
                idx_left=0,
                last_dir=1,
                brake_frames_left=0
            )
        def move_left(ss):
            return ss._replace(
                move=-movement_pattern[ss.idx_left],
                idx_left=(ss.idx_left + 1) % pat_len,
                idx_right=0,
                last_dir=-1,
                brake_frames_left=0
            )
        def brake(ss):
            def apply_brake(ss_inner):
                new_move = ss_inner.last_dir * BRAKE_SPEED
                new_brake = ss_inner.brake_frames_left - 1
                new_last = jnp.where(new_brake == 0, 0, ss_inner.last_dir)
                return ss_inner._replace(move=new_move, brake_frames_left=new_brake, last_dir=new_last)
            ss = ss._replace(brake_frames_left=jnp.where(
                jnp.logical_and(ss.last_dir != 0, ss.brake_frames_left == 0), BRAKE_DURATION, ss.brake_frames_left
            ))
            return jax.lax.cond(ss.brake_frames_left > 0, apply_brake, lambda x: x._replace(move=0.0), ss)

        return jax.lax.cond(press_right, move_right,
               lambda ss: jax.lax.cond(press_left, move_left, brake, ss), s)

    def jump_move(s):
        # mid-jump or jump start
        def jump_right(ss):
            return ss._replace(
                move=movement_pattern[ss.idx_right],
                idx_right=(ss.idx_right + 1) % pat_len,
                idx_left=0,
                jumpR=True,
                brake_frames_left=0,
                last_dir=0
            )
        def jump_left(ss):
            return ss._replace(
                move=-movement_pattern[ss.idx_left],
                idx_left=(ss.idx_left + 1) % pat_len,
                idx_right=0,
                jumpL=True,
                brake_frames_left=0,
                last_dir=0
            )
        condR = jnp.logical_or(press_right, s.jumpR)
        condL = jnp.logical_or(press_left, s.jumpL)
        return jax.lax.cond(condR, jump_right,
               lambda ss: jax.lax.cond(condL, jump_left, lambda x: x, ss), s)

    # Choose between walk/brake and jump move
    state = jax.lax.cond(state.jump == 0, walk_or_brake, jump_move, state)

    # Final movement application
    return movement(state, jnp.array([state.move, state.jump], dtype=jnp.int32))


import jaxatari.rendering.atraJaxis as aj
from jaxatari.renderers import AtraJaxisRenderer

class MarioBrosRenderer(AtraJaxisRenderer): 
    # holds functions to render given Gamestates

    def __init__(self):
        pass

    def draw_rect(self, surface, color, rect):
        r = pygame.Rect(rect)
        pygame.draw.rect(surface, color, r)

    def render(self, state: MarioBrosState) -> jnp.ndarray:
       
        surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        # Background
        surf.fill((0, 0, 0))

        # Player
        px, py = state.player.pos.tolist()
        color = (255, 0, 0) if state.player.brake_frames_left > 0 else PLAYER_COLOR
        self.draw_rect(surf, color, (px, py, *PLAYER_SIZE))

        # Platforms + Pow-Block
        for plat in PLATFORMS:
            self.draw_rect(surf, (228, 111, 111), plat.tolist())
        self.draw_rect(surf, (201, 164, 74), POW_BLOCK[0].tolist())

        # Enemy
        for ep in state.game.enemy_pos:
            ex, ey = ep.tolist()
            self.draw_rect(surf, (255, 0, 0), (ex, ey, *ENEMY_SIZE))

        

        
        arr = pygame.surfarray.array3d(surf)
        return arr


class JaxMarioBros(JaxEnvironment[MarioBrosState, MarioBrosObservation, MarioBrosInfo]):    # copied and adapted from jax_kangaroo.py ln.1671
    # holds reset and main step function

    def __init__(self):
        self.action_set = [
            Action.NOOP,
            Action.FIRE,
            Action.UP,
            Action.RIGHT,
            Action.LEFT,
            Action.DOWN,
            Action.UPRIGHT,
            Action.UPLEFT,
            Action.DOWNRIGHT,
            Action.DOWNLEFT,
            Action.UPFIRE,
            Action.RIGHTFIRE,
            Action.LEFTFIRE,
            Action.DOWNFIRE,
            Action.UPRIGHTFIRE,
            Action.UPLEFTFIRE,
            Action.DOWNRIGHTFIRE,
            Action.DOWNLEFTFIRE
        ]

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))

    def get_action_space(self) -> jnp.ndarray:
        return jnp.array(self.action_set)
    
    def _get_observation(self, state: MarioBrosState) -> MarioBrosObservation:
        return MarioBrosObservation(
            player_x=state.player.pos[0],
            player_y=state.player.pos[1]
        )

    def _get_info(self, state: MarioBrosState) -> MarioBrosInfo:
        return MarioBrosInfo(
            score=0
        )
    
    def reset(self, key = None) -> Tuple[MarioBrosObservation, MarioBrosState]:
        game = self.reset_game()
        obs = self._get_observation(game)
        return obs, game
    
    def reset_game(self) -> MarioBrosState:
        
        new_state = MarioBrosState(
            player= PlayerState(
                pos=jnp.array([PLAYER_START_X, PLAYER_START_Y]),
                vel=jnp.array([0.0, 0.0]),
                on_ground=False,
                jump_phase=jnp.int32(0),
                ascend_frames=jnp.int32(0),
                idx_right=jnp.int32(0),
                idx_left=jnp.int32(0),
                jump=jnp.int32(0),
                move=jnp.array(0.0, dtype=jnp.float32),
                jumpL=False,
                jumpR=False,
                last_dir=jnp.int32(0),
                brake_frames_left=jnp.int32(0)
            ),
            game = GameState(
                enemy_pos = jnp.array([[50.0, 160.0], [120.0, 160.0]]),
                enemy_vel = jnp.array([[0.5, 0.0], [-0.5, 0.0]])
            ),
            lives = jnp.int32(4)
        )

        return new_state
    
    def step(self, state: MarioBrosState, action: chex.Array) -> Tuple[MarioBrosObservation, MarioBrosState, float, bool, MarioBrosInfo]:
        # calls player_step function and check for collision with enemy
        def enemy_collision(s):
            obs, rS = self.reset()
            return obs, rS, 0.0, True, self._get_info(rS)

        def no_enemy_collision(s):
            return self._get_observation(state), MarioBrosState(
            player=s,
            game=state.game,
            lives=state.lives
        ), 0.0, True, self._get_info(state)
        
        return jax.lax.cond(check_enemy_collision(state.player.pos, state.game.enemy_pos), enemy_collision, no_enemy_collision, player_step(state.player, action))


# run game with: python scripts\play.py --game src\jaxatari\games\jax_mariobros.py --play
if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode(
        (SCREEN_WIDTH * WINDOW_SCALE, SCREEN_HEIGHT * WINDOW_SCALE)
    )
    pygame.display.set_caption("JAX Mario Bros Prototype")
    clock = pygame.time.Clock()
    game = JaxMarioBros()
    renderer = MarioBrosRenderer()

    _, state = game.reset()
    running = True
    
    while running:
    #    _,state,_,_,_ = game.step(state, ) 
        renderer.render(state)
    
        clock.tick(60)

    pygame.quit()