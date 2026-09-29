from functools import partial
import os
import jax
import jax.lax as lax
import jax.numpy as jnp
import chex
import numpy as np
from flax import struct

import jaxatari.spaces as spaces
from jaxatari.environment import (
    JaxEnvironment,
    JAXAtariAction as Action,
    ObjectObservation,
)

ATTRACT = 0
PLAY = 1
GAME_OVER = 2

SAUCER = 0
BUZZIE = 1
SQUEEZER = 2

SUBWAVE_KILL_TARGET = jnp.array([10, 20, 30], jnp.int32)
SUBWAVE_CONCURRENT = jnp.array([1, 1, 1], jnp.int32)

ENEMY_EMPTY = 0
ENEMY_ALIVE = 1
ENEMY_EXPLODING = 2
ENEMY_REFORMING = 3


def wave_bonus_for_level(level):
    base = jnp.array([400, 600, 1000, 1100], jnp.int32)
    cycle = (level - 1) % 4
    group = (level - 1) // 4
    return base[cycle] + 1000 * group


# ============================================================
# SPRITES
# ============================================================

PLAYER_SPRITE_RIGHT = jnp.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
], dtype=jnp.bool_)
PLAYER_SPRITE_LEFT = jnp.flip(PLAYER_SPRITE_RIGHT, axis=1)

# BOBO = humanoid (person-like). Moves horizontally, drops bombs.
# 7x10 walking sprite: top line, head with eyes, narrow body, two spread legs.
BOBO_SPRITE = jnp.array([
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 1, 1, 0, 1, 1, 0],
    [1, 1, 0, 0, 0, 1, 1],
    [1, 0, 0, 0, 0, 0, 1],
], dtype=jnp.bool_)

# 7x10 shooting sprite: same head/body, but one central leg.
BOBO_SPRITE_SHOOT = jnp.array([
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
], dtype=jnp.bool_)

# ENEMY = ring/UFO. Moves freely in all directions.
ENEMY_SPRITE = jnp.array([
    [0, 0, 1, 1, 0, 0, 0],
    [0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0],
    [1, 0, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 0, 0, 1, 1],
    [0, 1, 1, 0, 0, 1, 1],
    [0, 1, 1, 0, 0, 1, 1],
    [1, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 0, 0],
], dtype=jnp.bool_)
SAUCER_SPRITE = ENEMY_SPRITE
BUZZIE_SPRITE = ENEMY_SPRITE
SQUEEZER_SPRITE = ENEMY_SPRITE

BULLET_SPRITE_LOCAL = jnp.array([[1, 1, 1, 1]], dtype=jnp.bool_)

ENEMY_W = jnp.array([7.0, 7.0, 7.0], jnp.float32)
ENEMY_H = jnp.array([10.0, 10.0, 10.0], jnp.float32)

ENEMY_COLOR = jnp.array(
    [
        [200, 72, 72],     # red
        [125, 48, 173],    # purple
        [184, 70, 162],    # pink
    ],
    jnp.uint8,
)

C_PLAYER_RED = (214, 92, 92)
C_PLAYER_GREEN = (144, 252, 144)
C_BOBO_GOLD = (180, 122, 48)
C_BULLET_GREEN = (144, 252, 144)

C_ATTRACT_STAR = (210, 182, 86)
C_ATTRACT_GUNNER = (84, 184, 153)
C_ATTRACT_RED = (200, 72, 72)
C_ATTRACT_BLUE = (101, 160, 225)
C_ATTRACT_YELLOW = (210, 210, 64)


def _find_asset(name):
    module_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(module_dir, name),
        os.path.abspath(os.path.join(module_dir, "..", "..", "..", name)),
        os.path.abspath(os.path.join(module_dir, "..", "..", "..", "..", name)),
        os.path.join(os.getcwd(), name),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


class StarGunnerConstants(struct.PyTreeNode):
    WIDTH: int = struct.field(pytree_node=False, default=160)
    HEIGHT: int = struct.field(pytree_node=False, default=210)

    FRAMESKIP: int = struct.field(pytree_node=False, default=4)
    STICKY_ACTION_PROB: float = struct.field(pytree_node=False, default=0.25)

    MOD_ID: int = struct.field(pytree_node=False, default=0)

    MAX_EPISODE_FRAMES: int = struct.field(pytree_node=False, default=108_000)
    PLAY_TOP: int = struct.field(pytree_node=False, default=50)
    PLAY_BOTTOM: int = struct.field(pytree_node=False, default=175)
    HILL_Y: int = struct.field(pytree_node=False, default=190)

    PLAYER_WIDTH: int = struct.field(pytree_node=False, default=9)
    PLAYER_HEIGHT: int = struct.field(pytree_node=False, default=4)
    PLAYER_SPEED: float = struct.field(pytree_node=False, default=0.7)
    PLAYER_START_X: int = struct.field(pytree_node=False, default=80)
    PLAYER_START_Y: int = struct.field(pytree_node=False, default=165)
    PLAYER_LIVES_START: int = struct.field(pytree_node=False, default=5)
    EXTRA_LIFE_THRESHOLD: int = struct.field(pytree_node=False, default=10_000)
    MAX_LIVES: int = struct.field(pytree_node=False, default=255)
    INVULN_FRAMES: int = struct.field(pytree_node=False, default=60)

    BULLET_WIDTH: int = struct.field(pytree_node=False, default=4)
    BULLET_HEIGHT: int = struct.field(pytree_node=False, default=1)
    BULLET_SPEED: float = struct.field(pytree_node=False, default=4.0)
    MAX_BULLETS: int = struct.field(pytree_node=False, default=2)
    FIRE_COOLDOWN: int = struct.field(pytree_node=False, default=8)

    REFORM_FRAMES: int = struct.field(pytree_node=False, default=20)
    NUM_ENEMIES: int = struct.field(pytree_node=False, default=1)
    ENEMY_SPEED: float = struct.field(pytree_node=False, default=0.15)
    ENEMY_SPEED_INCREMENT: float = struct.field(pytree_node=False, default=0.10)
    MAX_ENEMY_SPEED_MULTIPLIER: float = struct.field(pytree_node=False, default=2.0)
    ENEMY_AMP: float = struct.field(pytree_node=False, default=0.0)

    BOBO_WIDTH: int = struct.field(pytree_node=False, default=7)
    BOBO_HEIGHT: int = struct.field(pytree_node=False, default=10)
    BOBO_Y: int = struct.field(pytree_node=False, default=40)

    ENEMY_Y_MAX: int = struct.field(pytree_node=False, default=165)
    ENEMY_Y_MIN: int = struct.field(pytree_node=False, default=45)
    ENEMY_CHANGE_PROB: float = struct.field(pytree_node=False, default=0.04)
    ENEMY_SPAWN_X: int = struct.field(pytree_node=False, default=8)
    COLLISION_POINTS: int = struct.field(pytree_node=False, default=200)
    BOBO_SPEED: float = struct.field(pytree_node=False, default=0.5)
    BOBO_BOMB_PERIOD: int = struct.field(pytree_node=False, default=30)

    MAX_BOMBS: int = struct.field(pytree_node=False, default=4)
    BOMB_WIDTH: int = struct.field(pytree_node=False, default=4)
    BOMB_HEIGHT: int = struct.field(pytree_node=False, default=1)
    BOMB_SPEED: float = struct.field(pytree_node=False, default=1.5)

    HILL_SCROLL_SPEED: float = struct.field(pytree_node=False, default=0.85)

    EXPLOSION_SIZE: int = struct.field(pytree_node=False, default=10)
    EXPLOSION_DURATION: int = struct.field(pytree_node=False, default=10)

    DEATH_PENALTY: float = struct.field(pytree_node=False, default=0.0)
    PLAYER_EXPLOSION_SIZE: int = struct.field(pytree_node=False, default=14)
    PLAYER_EXPLOSION_DURATION: int = struct.field(pytree_node=False, default=140)


class StarGunnerState(struct.PyTreeNode):
    mode: chex.Array
    player_x: chex.Array
    player_y: chex.Array
    player_facing: chex.Array
    prev_player_x: chex.Array
    prev_player_y: chex.Array
    step_counter: chex.Array

    previous_action: chex.Array
    key: chex.PRNGKey

    bullet_x: chex.Array
    bullet_y: chex.Array
    bullet_vx: chex.Array
    bullet_vy: chex.Array
    bullet_active: chex.Array
    fire_cooldown: chex.Array

    enemy_x: chex.Array
    enemy_y: chex.Array
    enemy_type: chex.Array
    enemy_vy: chex.Array
    enemy_vx: chex.Array
    enemy_phase: chex.Array
    enemy_mode: chex.Array
    enemy_mode_timer: chex.Array
    enemy_sweep_dir: chex.Array

    bobo_x: chex.Array
    bobo_base_x: chex.Array
    bobo_vx: chex.Array
    bomb_x: chex.Array
    bomb_y: chex.Array
    bomb_active: chex.Array

    explosion_x: chex.Array
    explosion_y: chex.Array
    explosion_timer: chex.Array
    explosion_active: chex.Array
    explosion_frag_vx: chex.Array
    explosion_frag_vy: chex.Array

    player_frag_vx: chex.Array
    player_frag_vy: chex.Array
    player_explosion_x: chex.Array
    player_explosion_y: chex.Array
    player_explosion_timer: chex.Array
    player_explosion_active: chex.Array

    enemy_state: chex.Array
    enemy_timer: chex.Array
    level: chex.Array
    subwave: chex.Array
    kills_in_subwave: chex.Array
    score: chex.Array
    lives: chex.Array
    invuln_timer: chex.Array
    respawn_timer: chex.Array
    extra_lives_earned: chex.Array


class StarGunnerObservation(struct.PyTreeNode):
    player: ObjectObservation
    enemies: ObjectObservation
    bullets: ObjectObservation
    bombs: ObjectObservation
    bobo: ObjectObservation


class StarGunnerInfo(struct.PyTreeNode):
    time: jnp.ndarray
    lives: jnp.ndarray
    wave: jnp.ndarray


def draw_rect(img, x, y, w, h, color):
    H, W, _ = img.shape
    yy = jnp.arange(H)[:, None]
    xx = jnp.arange(W)[None, :]
    mask = (xx >= x) & (xx < x + w) & (yy >= y) & (yy < y + h)
    return jnp.where(mask[:, :, None], color, img)


def draw_sprite(img, x, y, sprite, color):
    H, W = sprite.shape
    x = jnp.asarray(x).astype(jnp.int32)
    y = jnp.asarray(y).astype(jnp.int32)
    yy = jnp.arange(H)[:, None]
    xx = jnp.arange(W)[None, :]
    img_h, img_w, _ = img.shape

    y_pos = y + yy
    x_pos = x + xx

    valid_y = (y_pos >= 0) & (y_pos < img_h)
    valid_x = (x_pos >= 0) & (x_pos < img_w)
    valid = valid_y & valid_x & sprite

    y_safe = jnp.clip(y_pos, 0, img_h - 1)
    x_safe = jnp.clip(x_pos, 0, img_w - 1)

    img_region = img[y_safe, x_safe]
    masked = jnp.where(valid[..., None], color[None, None, :], img_region)
    img = img.at[y_safe, x_safe].set(masked)
    return img


def _aabb_overlap(ax, ay, aw, ah, bx, by, bw, bh):
    return (ax < bx + bw) & (ax + aw > bx) & (ay < by + bh) & (ay + ah > by)


def enemy_display_y(enemy_y, enemy_type, enemy_phase, step_counter, amp):
    bob = amp * jnp.sin(enemy_phase + step_counter.astype(jnp.float32) * 0.1)
    return enemy_y + jnp.where(enemy_type == SAUCER, bob, 0.0)


_FONT = {
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01111"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    "(": ["01110", "10001", "10110", "10100", "10110", "10001", "01110"],
}


def _glyph_np(ch):
    rows = _FONT.get(ch, _FONT[" "])
    return np.array([[1 if b == "1" else 0 for b in r] for r in rows], np.uint8)


def _bake_text(canvas, text, x, y, scale):
    cx = x
    for ch in text:
        g = _glyph_np(ch)
        ys, xs = np.where(g == 1)
        for gy, gx in zip(ys, xs):
            canvas[
                y + gy * scale : y + gy * scale + scale,
                cx + gx * scale : cx + gx * scale + scale,
            ] = True
        cx += 6 * scale
    return canvas


def _centered_x(text, scale, width):
    return (width - (len(text) * 6 * scale - scale)) // 2


class JaxStarGunner(
    JaxEnvironment[
        StarGunnerState, StarGunnerObservation, StarGunnerInfo, StarGunnerConstants
    ]
):
    ACTION_SET = jnp.array(
        [
            Action.NOOP, Action.FIRE, Action.UP, Action.RIGHT, Action.LEFT,
            Action.DOWN, Action.UPRIGHT, Action.UPLEFT, Action.DOWNRIGHT,
            Action.DOWNLEFT, Action.UPFIRE, Action.RIGHTFIRE, Action.LEFTFIRE,
            Action.DOWNFIRE, Action.UPRIGHTFIRE, Action.UPLEFTFIRE,
            Action.DOWNRIGHTFIRE, Action.DOWNLEFTFIRE,
        ],
        dtype=jnp.int32,
    )

    def __init__(self, consts: StarGunnerConstants = None, start_in_play: bool = True):
        consts = consts or StarGunnerConstants()
        super().__init__(consts)
        self.start_in_play = start_in_play
        self.renderer = StarGunnerRenderer(consts)

    def _is_fire(self, a):
        return (
            (a == Action.FIRE)
            | (a == Action.UPFIRE)
            | (a == Action.RIGHTFIRE)
            | (a == Action.LEFTFIRE)
            | (a == Action.DOWNFIRE)
            | (a == Action.UPRIGHTFIRE)
            | (a == Action.UPLEFTFIRE)
            | (a == Action.DOWNRIGHTFIRE)
            | (a == Action.DOWNLEFTFIRE)
        )

    def _spawn_positions(self):
        n = self.consts.NUM_ENEMIES
        x = jnp.full((n,), 0.0, jnp.float32)
        y = jnp.full((n,), float(self.consts.ENEMY_Y_MIN), jnp.float32)
        return x, y

    def _init_subwave(self, level, subwave):
        n = self.consts.NUM_ENEMIES
        idx = jnp.arange(n)
        concurrent = SUBWAVE_CONCURRENT[subwave]
        active = idx < concurrent

        x, y = self._spawn_positions()

        type_index = jnp.mod(idx + level - 1, 3).astype(jnp.int32)
        direction = jnp.where(idx % 2 == 0, 1.0, -1.0)
        vy = jnp.where(type_index == BUZZIE, direction * 0.3, 0.0)
        phase = idx.astype(jnp.float32) * (2.0 * jnp.pi / n)

        enemy_state = jnp.where(active, ENEMY_ALIVE, ENEMY_EMPTY)
        return x, y, type_index, vy, phase, enemy_state

    def _fresh_game_fields(self, key):
        n = self.consts.NUM_ENEMIES
        ex, ey, et, evy, eph, est = self._init_subwave(
            jnp.array(1, jnp.int32), jnp.array(0, jnp.int32)
        )

        start_x = jnp.array(self.consts.PLAYER_START_X, jnp.float32)
        start_y = jnp.array(self.consts.PLAYER_START_Y, jnp.float32)
        return dict(
            key=key,
            player_x=start_x,
            player_y=start_y,
            player_facing=jnp.array(1, jnp.int32),
            prev_player_x=start_x,
            prev_player_y=start_y,
            previous_action=jnp.array(0, jnp.int32),
            player_frag_vx=jnp.zeros((6,), jnp.float32),
            player_frag_vy=jnp.zeros((6,), jnp.float32),
            bullet_x=jnp.zeros((self.consts.MAX_BULLETS,), jnp.float32),
            bullet_y=jnp.zeros((self.consts.MAX_BULLETS,), jnp.float32),
            bullet_vx=jnp.zeros((self.consts.MAX_BULLETS,), jnp.float32),
            bullet_vy=jnp.zeros((self.consts.MAX_BULLETS,), jnp.float32),
            bullet_active=jnp.zeros((self.consts.MAX_BULLETS,), jnp.bool_),
            fire_cooldown=jnp.array(0, jnp.int32),
            enemy_x=ex,
            enemy_y=ey,
            enemy_type=et,
            enemy_vy=evy,
            enemy_vx=jnp.zeros((n,), jnp.float32),
            enemy_phase=eph,
            enemy_mode=jnp.zeros((n,), jnp.int32),
            enemy_mode_timer=jnp.zeros((n,), jnp.int32),
            enemy_sweep_dir=jnp.ones((n,), jnp.float32),
            enemy_state=est,
            enemy_timer=jnp.zeros((n,), jnp.int32),
            level=jnp.array(1, jnp.int32),
            subwave=jnp.array(0, jnp.int32),
            kills_in_subwave=jnp.array(0, jnp.int32),
            extra_lives_earned=jnp.array(0, jnp.int32),
            bobo_x=jnp.array(105.0, jnp.float32),
            bobo_base_x=jnp.array(105.0, jnp.float32),
            bobo_vx=jnp.array(self.consts.BOBO_SPEED, jnp.float32),
            bomb_x=jnp.zeros((self.consts.MAX_BOMBS,), jnp.float32),
            bomb_y=jnp.zeros((self.consts.MAX_BOMBS,), jnp.float32),
            bomb_active=jnp.zeros((self.consts.MAX_BOMBS,), jnp.bool_),
            explosion_x=jnp.zeros((n,), jnp.float32),
            explosion_y=jnp.zeros((n,), jnp.float32),
            explosion_timer=jnp.zeros((n,), jnp.int32),
            explosion_active=jnp.zeros((n,), jnp.bool_),
            explosion_frag_vx=jnp.zeros((n, 6), jnp.float32),
            explosion_frag_vy=jnp.zeros((n, 6), jnp.float32),
            player_explosion_x=jnp.array(0.0, jnp.float32),
            player_explosion_y=jnp.array(0.0, jnp.float32),
            player_explosion_timer=jnp.array(0, jnp.int32),
            player_explosion_active=jnp.array(False, jnp.bool_),
            score=jnp.array(0, jnp.int32),
            lives=jnp.array(self.consts.PLAYER_LIVES_START, jnp.int32),
            invuln_timer=jnp.array(0, jnp.int32),
            respawn_timer=jnp.array(0, jnp.int32),
        )

    def _apply_sticky_action(self, state, action):
        key, sticky_key = jax.random.split(state.key)
        repeat_previous = jax.random.uniform(sticky_key) < self.consts.STICKY_ACTION_PROB
        effective_action = jnp.where(repeat_previous, state.previous_action, action)
        state = state.replace(key=key, previous_action=effective_action)
        return state, effective_action

    def reset(self, key: chex.PRNGKey = jax.random.PRNGKey(0)):
        fields = self._fresh_game_fields(key)
        mode = jnp.array(PLAY if self.start_in_play else ATTRACT, jnp.int32)
        state = StarGunnerState(mode=mode, step_counter=jnp.array(0, jnp.int32), **fields)
        return self._get_observation(state), state

    def _player_step(self, state, a):
        left = (
            (a == Action.LEFT) | (a == Action.LEFTFIRE) | (a == Action.UPLEFT)
            | (a == Action.DOWNLEFT) | (a == Action.UPLEFTFIRE) | (a == Action.DOWNLEFTFIRE)
        )
        right = (
            (a == Action.RIGHT) | (a == Action.RIGHTFIRE) | (a == Action.UPRIGHT)
            | (a == Action.DOWNRIGHT) | (a == Action.UPRIGHTFIRE) | (a == Action.DOWNRIGHTFIRE)
        )
        up = (
            (a == Action.UP) | (a == Action.UPFIRE) | (a == Action.UPRIGHT)
            | (a == Action.UPLEFT) | (a == Action.UPRIGHTFIRE) | (a == Action.UPLEFTFIRE)
        )
        down = (
            (a == Action.DOWN) | (a == Action.DOWNFIRE) | (a == Action.DOWNRIGHT)
            | (a == Action.DOWNLEFT) | (a == Action.DOWNRIGHTFIRE) | (a == Action.DOWNLEFTFIRE)
        )

        mx = right.astype(jnp.float32) - left.astype(jnp.float32)
        my = down.astype(jnp.float32) - up.astype(jnp.float32)

        prev_x = state.player_x
        prev_y = state.player_y

        nx = state.player_x + mx * self.consts.PLAYER_SPEED
        ny = state.player_y + my * self.consts.PLAYER_SPEED

        nx = jnp.mod(nx, self.consts.WIDTH)
        ny = jnp.clip(ny, float(self.consts.PLAY_TOP), float(self.consts.PLAY_BOTTOM - self.consts.PLAYER_HEIGHT))

        facing = jnp.where(
            right, jnp.array(1, dtype=jnp.int32),
            jnp.where(left, jnp.array(-1, dtype=jnp.int32), state.player_facing),
        )

        return state.replace(
            player_x=nx, player_y=ny, player_facing=facing,
            prev_player_x=prev_x, prev_player_y=prev_y,
        )

    def _bullet_step(self, state, a):
        fire = self._is_fire(a)

        facing_dir = jnp.where(state.player_facing > 0, 1.0, -1.0)

        goes_left = (
                (a == Action.LEFTFIRE) | (a == Action.UPLEFTFIRE) | (a == Action.DOWNLEFTFIRE)
        )
        goes_right = (
                (a == Action.RIGHTFIRE) | (a == Action.UPRIGHTFIRE) | (a == Action.DOWNRIGHTFIRE)
        )

        vx_dir = jnp.where(goes_left, -1.0, jnp.where(goes_right, 1.0, facing_dir))
        vx = vx_dir * self.consts.BULLET_SPEED
        vy = jnp.zeros_like(vx)

        free = jnp.argmax(~state.bullet_active)
        ready = state.fire_cooldown <= 0
        can_fire = fire & ready & (~jnp.all(state.bullet_active))

        spawn_x = jnp.where(
            vx > 0,
            state.player_x + self.consts.PLAYER_WIDTH,
            state.player_x - self.consts.BULLET_WIDTH,
        )
        spawn_y = state.player_y + self.consts.PLAYER_HEIGHT - self.consts.BULLET_HEIGHT

        bx = state.bullet_x.at[free].set(spawn_x)
        by = state.bullet_y.at[free].set(spawn_y)
        bvx = state.bullet_vx.at[free].set(vx)
        bvy = state.bullet_vy.at[free].set(vy)
        ba = state.bullet_active.at[free].set(True)

        bullet_x = jnp.where(can_fire, bx, state.bullet_x)
        bullet_y = jnp.where(can_fire, by, state.bullet_y)
        bullet_vx = jnp.where(can_fire, bvx, state.bullet_vx)
        bullet_vy = jnp.where(can_fire, bvy, state.bullet_vy)
        bullet_active = jnp.where(can_fire, ba, state.bullet_active)

        bullet_x = bullet_x + bullet_vx
        bullet_y = bullet_y + bullet_vy

        bullet_active = (
            bullet_active
            & (bullet_x >= 0) & (bullet_x < self.consts.WIDTH)
            & (bullet_y >= 0) & (bullet_y < self.consts.HEIGHT)
        )

        cooldown = jnp.where(can_fire, self.consts.FIRE_COOLDOWN, jnp.maximum(state.fire_cooldown - 1, 0))

        return state.replace(
            bullet_x=bullet_x, bullet_y=bullet_y,
            bullet_vx=bullet_vx, bullet_vy=bullet_vy,
            bullet_active=bullet_active, fire_cooldown=cooldown,
        )

    def _enemy_display_y(self, state):
        return enemy_display_y(
            state.enemy_y, state.enemy_type, state.enemy_phase,
            state.step_counter, self.consts.ENEMY_AMP,
        )

    def _enemy_step(self, state):
        """
        Enemy (ring) behaviour:

        Above bottom:
            mode 0 = diagonal (descend + right)
            mode 1 = vertical (straight down)
            Random switches between them.

        At bottom (y >= ENEMY_Y_MAX):
            Pause for PAUSE_FRAMES, then
            mode 2 = horizontal sweep rightward, wraps at edges.

        Depth never exceeds ENEMY_Y_MAX.
        """
        n = self.consts.NUM_ENEMIES
        is_alive = state.enemy_state == ENEMY_ALIVE

        key, k_mode = jax.random.split(state.key)
        r = jax.random.uniform(k_mode, (n,))

        speed = self.consts.ENEMY_SPEED
        mode = state.enemy_mode

        y_top = float(self.consts.ENEMY_Y_MIN)
        y_bottom = float(self.consts.ENEMY_Y_MAX)
        at_bottom = state.enemy_y >= (y_bottom - 1.0)

        # Bottom timer counts only while at bottom
        new_timer = jnp.where(at_bottom, state.enemy_mode_timer + 1, 0)

        PAUSE_FRAMES = 20
        pause = at_bottom & (new_timer < PAUSE_FRAMES)

        # Transition to horizontal when at bottom after pause
        mode = jnp.where(at_bottom & (new_timer >= PAUSE_FRAMES), 2, mode)

        # Above bottom: never horizontal
        mode = jnp.where((~at_bottom) & (mode == 2), 0, mode)

        # Random switching between diagonal (0) and vertical (1) above bottom
        switch_to_vert = (mode == 0) & (r < 0.010) & (~at_bottom)
        switch_to_diag = (mode == 1) & (r < 0.010) & (~at_bottom)
        mode = jnp.where(switch_to_vert, 1, jnp.where(switch_to_diag, 0, mode))

        # --- Velocities ---
        vx_diag = speed * 1.2
        vy_diag = speed * 0.4

        vx_horiz = speed * 1.5   # always right; wraps at edge
        vy_horiz = 0.0

        vx_vert = 0.0
        vy_vert = speed * 0.8

        vx = jnp.where(mode == 0, vx_diag,
              jnp.where(mode == 2, vx_horiz,
                                       vx_vert))
        vy = jnp.where(mode == 0, vy_diag,
              jnp.where(mode == 2, vy_horiz,
                                       vy_vert))

        # Pause overrides movement
        vx = jnp.where(pause, 0.0, vx)
        vy = jnp.where(pause, 0.0, vy)

        # Never descend past bottom
        vy = jnp.where(at_bottom & (vy > 0), 0.0, vy)

        nx = state.enemy_x + jnp.where(is_alive, vx, 0.0)
        ny = state.enemy_y + jnp.where(is_alive, vy, 0.0)

        # Horizontal wrap across the whole screen
        nx = jnp.mod(nx, float(self.consts.WIDTH))
        ny = jnp.clip(ny, y_top, y_bottom)

        return state.replace(
            key=key,
            enemy_x=nx,
            enemy_y=ny,
            enemy_vx=vx,
            enemy_vy=vy,
            enemy_mode=mode,
            enemy_mode_timer=new_timer,
        )

    def _bobo_step(self, state):
        """
        Bobo moves within a sliding window.
        Window spans 2 units (80 px). Within a window he moves back and
        forth 3 times, dropping a bomb after each full traversal.
        After 3 traversals the window slides one unit (40 px) right.
        At the screen edge the window direction reverses.
        """
        c = self.consts

        POSITION_SIZE = 40.0
        WINDOW_UNITS = 2
        WINDOW_WIDTH = POSITION_SIZE * WINDOW_UNITS      # 80
        SPEED = 1.0
        TRAVEL_FRAMES = int(WINDOW_WIDTH / SPEED)        # 40 frames one way
        CYCLE_FRAMES = 2 * TRAVEL_FRAMES                 # 80 frames full traverse
        CYCLES_PER_WINDOW = 3
        WINDOW_FRAMES = CYCLES_PER_WINDOW * CYCLE_FRAMES # 240

        # How many window start positions fit on screen
        num_windows = int((c.WIDTH - WINDOW_WIDTH) / POSITION_SIZE) + 1  # 3

        # Triangle wave over window positions (e.g. 0, 1, 2, 1, 0, 1, ...)
        tri_period = 2 * (num_windows - 1) if num_windows > 1 else 1
        tri_m = (state.step_counter // WINDOW_FRAMES) % tri_period
        window_idx = jnp.where(
            tri_m <= (num_windows - 1),
            tri_m,
            tri_period - tri_m,
        ).astype(jnp.float32)

        window_left = window_idx * POSITION_SIZE

        # Within-window phase: triangle 0 -> WINDOW_WIDTH -> 0
        cycle_t = state.step_counter % CYCLE_FRAMES
        tri = jnp.where(cycle_t <= TRAVEL_FRAMES, cycle_t, CYCLE_FRAMES - cycle_t)
        bobo_x = window_left + tri.astype(jnp.float32) * SPEED

        bobo_vx = jnp.where(cycle_t <= TRAVEL_FRAMES, SPEED, -SPEED)

        return state.replace(bobo_x=bobo_x, bobo_vx=bobo_vx)

    def _bomb_step(self, state):
        """Bobo drops horizontal bombs that fall straight down.
        No bomb while the player is exploding or invulnerable."""
        CYCLE_FRAMES = 80
        # Player is safe during explosion and invulnerability window
        no_bomb = state.player_explosion_active | (state.invuln_timer > 0)
        drop = (
            ((state.step_counter % CYCLE_FRAMES) == (CYCLE_FRAMES - 1))
            & (~no_bomb)
        )

        free = jnp.argmax(~state.bomb_active)
        can_drop = drop & (~jnp.all(state.bomb_active))

        # Spawn directly below Bobo
        spawn_x = state.bobo_x + self.consts.BOBO_WIDTH / 2 - self.consts.BOMB_WIDTH / 2
        spawn_y = self.consts.BOBO_Y + self.consts.BOBO_HEIGHT

        bx = state.bomb_x.at[free].set(spawn_x)
        by = state.bomb_y.at[free].set(spawn_y)
        ba = state.bomb_active.at[free].set(True)

        bomb_x = jnp.where(can_drop, bx, state.bomb_x)
        bomb_y = jnp.where(can_drop, by, state.bomb_y)
        bomb_active = jnp.where(can_drop, ba, state.bomb_active)

        # Fall down
        bomb_y = bomb_y + self.consts.BOMB_SPEED

        # Deactivate when hitting hills
        bomb_active = bomb_active & (bomb_y < self.consts.HILL_Y)

        return state.replace(bomb_x=bomb_x, bomb_y=bomb_y, bomb_active=bomb_active)

    def _resolve_collisions(self, state):
        edy = self._enemy_display_y(state)
        ew, eh = ENEMY_W[state.enemy_type], ENEMY_H[state.enemy_type]
        is_alive = state.enemy_state == ENEMY_ALIVE

        def row(bx, by, active):
            return (
                _aabb_overlap(bx, by, self.consts.BULLET_WIDTH, self.consts.BULLET_HEIGHT,
                              state.enemy_x, edy, ew, eh)
                & active & is_alive
            )

        hits = jax.vmap(row)(state.bullet_x, state.bullet_y, state.bullet_active)
        enemy_hit = jnp.any(hits, axis=0)
        bullet_used = jnp.any(hits, axis=1)

        vulnerable = state.invuln_timer <= 0
        reforming_danger = (state.enemy_state == ENEMY_REFORMING) & (state.enemy_timer == 1)
        dangerous = is_alive | reforming_danger

        enemy_touch_mask = (
            _aabb_overlap(state.player_x, state.player_y, self.consts.PLAYER_WIDTH,
                          self.consts.PLAYER_HEIGHT, state.enemy_x, edy, ew, eh)
            & dangerous
        )
        enemy_touch = jnp.any(enemy_touch_mask)

        bullet_kill = enemy_hit
        collision_kill = enemy_touch_mask & vulnerable & is_alive & (~bullet_kill)
        enemy_hit = bullet_kill | collision_kill

        points_per_kill = ((state.subwave + 1) * 100).astype(jnp.int32)
        gained = (
                jnp.sum(jnp.where(bullet_kill, points_per_kill, 0))
                + jnp.sum(jnp.where(collision_kill, self.consts.COLLISION_POINTS, 0))
        )
        new_kills = state.kills_in_subwave + jnp.sum(enemy_hit.astype(jnp.int32))

        new_enemy_state = jnp.where(enemy_hit, ENEMY_EXPLODING, state.enemy_state)
        new_enemy_timer = jnp.where(enemy_hit, self.consts.EXPLOSION_DURATION, state.enemy_timer)

        exp_active = state.explosion_active | enemy_hit
        exp_timer = jnp.where(enemy_hit, self.consts.EXPLOSION_DURATION, state.explosion_timer)
        exp_x = jnp.where(enemy_hit, state.enemy_x, state.explosion_x)
        exp_y = jnp.where(enemy_hit, edy, state.explosion_y)

        bomb_each = (
            _aabb_overlap(state.player_x, state.player_y, self.consts.PLAYER_WIDTH,
                          self.consts.PLAYER_HEIGHT, state.bomb_x, state.bomb_y,
                          self.consts.BOMB_WIDTH, self.consts.BOMB_HEIGHT)
            & state.bomb_active
        )
        damaged = vulnerable & (enemy_touch | jnp.any(bomb_each))

        spawn_x, spawn_y = self._spawn_positions()
        new_enemy_x = jnp.where(damaged, spawn_x, state.enemy_x)
        new_enemy_y = jnp.where(damaged, spawn_y, state.enemy_y)
        new_enemy_vx = jnp.where(damaged, 0.0, state.enemy_vx)
        new_enemy_vy = jnp.where(damaged, 0.0, state.enemy_vy)

        # 6 fragments radiating outward (like ALE)
        angles = jnp.array([0.0, jnp.pi/3, 2*jnp.pi/3, jnp.pi, 4*jnp.pi/3, 5*jnp.pi/3])
        p_frag_speed = 0.2
        new_p_frag_vx = jnp.where(damaged, jnp.cos(angles) * p_frag_speed, state.player_frag_vx)
        new_p_frag_vy = jnp.where(damaged, jnp.sin(angles) * p_frag_speed, state.player_frag_vy)

        p_exp_active = state.player_explosion_active | damaged
        p_exp_timer = jnp.where(damaged, self.consts.PLAYER_EXPLOSION_DURATION,
                                jnp.maximum(state.player_explosion_timer - 1, 0))
        p_exp_active = p_exp_active & (p_exp_timer > 0)
        p_exp_x = jnp.where(damaged, state.player_x, state.player_explosion_x)
        p_exp_y = jnp.where(damaged, state.player_y, state.player_explosion_y)

        respawn_timer = jnp.where(damaged, self.consts.PLAYER_EXPLOSION_DURATION,
                                  jnp.maximum(state.respawn_timer - 1, 0))

        new_score = state.score + gained
        should_have = new_score // self.consts.EXTRA_LIFE_THRESHOLD
        new_extra = jnp.minimum(should_have, self.consts.MAX_LIVES - self.consts.PLAYER_LIVES_START)
        gained_lives = jnp.maximum(new_extra - state.extra_lives_earned, 0)

        frag_speed = 1.5
        new_frag_vx = jnp.where(enemy_hit[:, None], jnp.cos(angles)[None, :] * frag_speed, state.explosion_frag_vx)
        new_frag_vy = jnp.where(enemy_hit[:, None], jnp.sin(angles)[None, :] * frag_speed, state.explosion_frag_vy)

        return (
            state.replace(
                score=new_score,
                lives=jnp.minimum(
                    self.consts.MAX_LIVES,
                    jnp.maximum(0, state.lives - damaged.astype(jnp.int32)) + gained_lives,
                ),
                extra_lives_earned=state.extra_lives_earned + gained_lives,
                invuln_timer=jnp.where(damaged, self.consts.INVULN_FRAMES, state.invuln_timer),
                respawn_timer=respawn_timer,
                enemy_state=new_enemy_state,
                enemy_timer=new_enemy_timer,
                enemy_x=new_enemy_x,
                enemy_y=new_enemy_y,
                enemy_vx=new_enemy_vx,
                enemy_vy=new_enemy_vy,
                kills_in_subwave=new_kills,
                bullet_active=state.bullet_active & (~bullet_used),
                bomb_active=state.bomb_active & (~(bomb_each & damaged)),
                explosion_active=exp_active,
                explosion_timer=exp_timer,
                explosion_x=exp_x,
                explosion_y=exp_y,
                explosion_frag_vx=new_frag_vx,
                explosion_frag_vy=new_frag_vy,
                player_frag_vx=new_p_frag_vx,
                player_frag_vy=new_p_frag_vy,
                player_explosion_active=p_exp_active,
                player_explosion_timer=p_exp_timer,
                player_explosion_x=p_exp_x,
                player_explosion_y=p_exp_y,
            ),
            damaged,
        )

    def _enemy_lifecycle_step(self, state):
        n = self.consts.NUM_ENEMIES
        timer = jnp.maximum(state.enemy_timer - 1, 0)

        finished_explosion = (state.enemy_state == ENEMY_EXPLODING) & (timer == 0)
        finished_reform = (state.enemy_state == ENEMY_REFORMING) & (timer == 0)

        target = SUBWAVE_KILL_TARGET[state.subwave]
        quota_reached = state.kills_in_subwave >= target

        goes_to_reform = finished_explosion & (~quota_reached)
        goes_empty = finished_explosion & quota_reached

        new_enemy_state = state.enemy_state
        new_enemy_state = jnp.where(goes_to_reform, ENEMY_REFORMING, new_enemy_state)
        new_enemy_state = jnp.where(goes_empty, ENEMY_EMPTY, new_enemy_state)
        # When an enemy finishes reforming and becomes ALIVE again,
        # respawn it at the top of the screen with a new random X and mode.
        key_r, k_x, k_mode = jax.random.split(state.key, 3)
        respawn_x = jax.random.uniform(k_x, (n,), minval=10.0,
                                       maxval=float(self.consts.WIDTH - 10))
        respawn_y = jnp.full((n,), float(self.consts.ENEMY_Y_MIN), jnp.float32)
        respawn_mode = jax.random.randint(k_mode, (n,), 0, 2).astype(jnp.int32)

        # Only reset position/mode for enemies that JUST finished reforming
        reset_pos = finished_reform

        new_enemy_state = jnp.where(finished_reform, ENEMY_ALIVE, new_enemy_state)

        new_timer = jnp.where(goes_to_reform, self.consts.REFORM_FRAMES, timer)

        slot_idx = jnp.arange(n)
        concurrent = SUBWAVE_CONCURRENT[state.subwave]
        all_slots_done = jnp.all((new_enemy_state == ENEMY_EMPTY) | (slot_idx >= concurrent))
        subwave_complete = quota_reached & all_slots_done

        bonus = jnp.where(subwave_complete, wave_bonus_for_level(state.level), 0)

        next_subwave = jnp.where(subwave_complete, (state.subwave + 1) % 3, state.subwave)
        level_up = subwave_complete & (state.subwave == 2)
        next_level = state.level + level_up.astype(jnp.int32)

        ex, ey, et, evy, eph, est = self._init_subwave(next_level, next_subwave)

        final_enemy_state = jnp.where(subwave_complete, est, new_enemy_state)
        final_x = jnp.where(subwave_complete, ex, state.enemy_x)
        final_y = jnp.where(subwave_complete, ey, state.enemy_y)
        final_type = jnp.where(subwave_complete, et, state.enemy_type)
        final_vy = jnp.where(subwave_complete, evy, state.enemy_vy)
        final_phase = jnp.where(subwave_complete, eph, state.enemy_phase)
        final_timer = jnp.where(subwave_complete, jnp.zeros((n,), jnp.int32), new_timer)
        final_kills = jnp.where(subwave_complete, jnp.array(0, jnp.int32), state.kills_in_subwave)

        # Apply respawn position/mode only to enemies that just reformed
        final_x = jnp.where(reset_pos, respawn_x, final_x)
        final_y = jnp.where(reset_pos, respawn_y, final_y)
        final_mode = jnp.where(reset_pos, respawn_mode, state.enemy_mode)
        final_timer2 = jnp.where(reset_pos, 0, state.enemy_mode_timer)

        return state.replace(
            key=key_r,
            enemy_state=final_enemy_state, enemy_timer=final_timer,
            enemy_x=final_x, enemy_y=final_y, enemy_type=final_type,
            enemy_vy=final_vy, enemy_phase=final_phase,
            enemy_mode=final_mode,
            enemy_mode_timer=final_timer2,
            subwave=next_subwave, level=next_level,
            kills_in_subwave=final_kills, score=state.score + bonus,
        )

    def _explosion_step(self, state):
        t = jnp.maximum(state.explosion_timer - 1, 0)
        return state.replace(explosion_timer=t, explosion_active=t > 0)

    def _play_frame(self, state, action):
        old_score = state.score

        respawning = state.respawn_timer > 0
        action = jnp.where(respawning, jnp.int32(Action.NOOP), action)

        state = self._player_step(state, action)
        state = self._bullet_step(state, action)
        state = self._enemy_step(state)
        state = self._bobo_step(state)
        state = self._bomb_step(state)
        state, damaged = self._resolve_collisions(state)

        just_respawned = (state.respawn_timer == 1)
        state = state.replace(
            player_x=jnp.where(just_respawned, jnp.float32(self.consts.PLAYER_START_X), state.player_x),
            player_y=jnp.where(just_respawned, jnp.float32(self.consts.PLAYER_START_Y), state.player_y),
        )

        state = self._enemy_lifecycle_step(state)
        state = self._explosion_step(state)

        state = state.replace(
            step_counter=state.step_counter + 1,
            invuln_timer=jnp.maximum(state.invuln_timer - 1, 0),
        )

        done = state.lives <= 0
        reward = (state.score - old_score).astype(jnp.float32) - damaged.astype(jnp.float32) * self.consts.DEATH_PENALTY

        state = state.replace(mode=jnp.where(done, GAME_OVER, PLAY))
        return state, reward, done

    def _play_branch(self, state, action):
        def body_fn(_, carry):
            state, total_reward, done = carry

            def run_frame(_):
                new_state, reward, frame_done = self._play_frame(state, action)
                return new_state, total_reward + reward, done | frame_done

            def skip_frame(_):
                return state, total_reward, done

            return jax.lax.cond(done, skip_frame, run_frame, operand=None)

        state, reward, done = jax.lax.fori_loop(
            0, self.consts.FRAMESKIP, body_fn,
            (state, jnp.float32(0.0), jnp.bool_(False)),
        )

        truncated = state.step_counter >= self.consts.MAX_EPISODE_FRAMES
        done = done | truncated
        return state, reward, done

    def _attract_branch(self, state, a):
        fire = self._is_fire(a)

        def start(_):
            fields = self._fresh_game_fields(state.key)
            return StarGunnerState(mode=jnp.array(PLAY, jnp.int32), step_counter=jnp.array(0, jnp.int32), **fields)

        def wait(_):
            return state.replace(step_counter=state.step_counter + 1)

        new = jax.lax.cond(fire, start, wait, operand=None)
        return new, jnp.float32(0.0), jnp.bool_(False)

    def _gameover_branch(self, state, a):
        fire = self._is_fire(a)
        new_mode = jnp.where(fire, ATTRACT, GAME_OVER)
        new = state.replace(mode=new_mode, step_counter=state.step_counter + 1)
        return new, jnp.float32(0.0), jnp.bool_(False)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: StarGunnerState, action: chex.Array):
        a = jnp.take(self.ACTION_SET, action.astype(jnp.int32))

        in_play = state.mode == PLAY
        sticky_state, sticky_action = self._apply_sticky_action(state, a)
        state = jax.tree.map(lambda new, old: jnp.where(in_play, new, old), sticky_state, state)
        effective_action = jnp.where(in_play, sticky_action, a)

        state, reward, done = jax.lax.switch(
            state.mode,
            [self._attract_branch, self._play_branch, self._gameover_branch],
            state, effective_action,
        )
        return self._get_observation(state), state, reward, done, self._get_info(state)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: StarGunnerState):
        return self.renderer.render(state)

    def _get_observation(self, state):
        edy = self._enemy_display_y(state)
        player = ObjectObservation.create(
            x=state.player_x, y=state.player_y,
            width=jnp.array(self.consts.PLAYER_WIDTH),
            height=jnp.array(self.consts.PLAYER_HEIGHT),
        )
        enemies = ObjectObservation.create(
            x=jnp.where(state.enemy_state == ENEMY_ALIVE, state.enemy_x, -1.0),
            y=jnp.where(state.enemy_state == ENEMY_ALIVE, edy, -1.0),
            width=ENEMY_W[state.enemy_type], height=ENEMY_H[state.enemy_type],
        )
        bullets = ObjectObservation.create(
            x=jnp.where(state.bullet_active, state.bullet_x, -1.0),
            y=jnp.where(state.bullet_active, state.bullet_y, -1.0),
            width=jnp.full((self.consts.MAX_BULLETS,), self.consts.BULLET_WIDTH, jnp.float32),
            height=jnp.full((self.consts.MAX_BULLETS,), self.consts.BULLET_HEIGHT, jnp.float32),
        )
        bombs = ObjectObservation.create(
            x=jnp.where(state.bomb_active, state.bomb_x, -1.0),
            y=jnp.where(state.bomb_active, state.bomb_y, -1.0),
            width=jnp.full((self.consts.MAX_BOMBS,), self.consts.BOMB_WIDTH, jnp.float32),
            height=jnp.full((self.consts.MAX_BOMBS,), self.consts.BOMB_HEIGHT, jnp.float32),
        )
        bobo = ObjectObservation.create(
            x=state.bobo_x, y=jnp.array(float(self.consts.BOBO_Y)),
            width=jnp.array(self.consts.BOBO_WIDTH),
            height=jnp.array(self.consts.BOBO_HEIGHT),
        )
        return StarGunnerObservation(player, enemies, bullets, bombs, bobo)

    def action_space(self):
        return spaces.Discrete(len(self.ACTION_SET))

    def observation_space(self):
        s = (self.consts.HEIGHT, self.consts.WIDTH)
        return spaces.Dict({
            "player": spaces.get_object_space(n=None, screen_size=s),
            "enemies": spaces.get_object_space(n=self.consts.NUM_ENEMIES, screen_size=s),
            "bullets": spaces.get_object_space(n=self.consts.MAX_BULLETS, screen_size=s),
            "bombs": spaces.get_object_space(n=self.consts.MAX_BOMBS, screen_size=s),
            "bobo": spaces.get_object_space(n=None, screen_size=s),
        })

    def image_space(self):
        return spaces.Box(low=0, high=255, shape=(self.consts.HEIGHT, self.consts.WIDTH, 3), dtype=jnp.uint8)

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state):
        return StarGunnerInfo(time=state.step_counter, lives=state.lives, wave=state.level)


class StarGunnerRenderer:
    def __init__(self, consts: StarGunnerConstants):
        self.consts = consts
        W, H = consts.WIDTH, consts.HEIGHT

        C_STAR = C_ATTRACT_STAR
        C_GUNNER = C_ATTRACT_GUNNER
        C_RED = C_ATTRACT_RED
        C_BLUE = C_ATTRACT_BLUE
        C_YELLOW = C_ATTRACT_YELLOW

        _ATTRACT = [
            (["111100001111", "100100001001", "100100001001", "000000000000",
              "100100001001", "100100001001", "111100001111"], 81, 16, C_BLUE),
            (["1000000000000000100000000000000010000000",
              "1100000000000000110000000000000011000000",
              "1111000000000000111100000000000011110000",
              "1111100000000000111110000000000011111000",
              "1111111100000000111111110000000011111111"], 55, 24, C_RED),
            (["00011111101111100111100011110000",
              "00011000100011001100010011001000",
              "00110001100010001000010010001100",
              "00100000000010001000010010000010",
              "00011100000010001111110010111100",
              "00000010000110001100001001111000",
              "00000010000100001000001001001000",
              "01000100000100001000001001000100",
              "10000100000100001000001001000010",
              "11111000000100001000001001000001"], 60, 84, C_STAR),
            (["0110000000000000011000", "1100000000000000000000",
              "1011010101110111011011", "1001010101110111000011",
              "0110011101010101011010"], 65, 97, C_GUNNER),
            (["011110000000000000000", "100001001011101110111",
              "101101001011101110001", "101101001011101110111",
              "101101001000101110110", "110001001000101110110",
              "011110001000101110111"], 65, 106, C_RED),
            (["11111100010000000000000000", "00000000010000000000000000",
              "11111100010000000000000000", "00110000010000000000000000",
              "00000111010111011101010111", "00110111010111011001110110",
              "00110111010111011100100111", "00110110010110000100100001",
              "00110111010111011101000111"], 63, 115, C_RED),
            (["1", "1", "0", "0", "1", "1"], 71, 142, C_YELLOW),
        ]

        attract_rgb = np.zeros((H, W, 3), np.uint8)
        attract_mask = np.zeros((H, W), bool)

        SHIP_ROW_Y = 24
        SHIP_COLS = (0, 16, 32)
        SHIP_W = 8

        temp_masks = []
        temp_rgbs = []

        for rows, ox, oy, col in _ATTRACT:
            for gy, r in enumerate(rows):
                for gx, ch in enumerate(r):
                    if ch == "1":
                        attract_rgb[oy + gy, ox + gx] = col
                        attract_mask[oy + gy, ox + gx] = True

            if oy == SHIP_ROW_Y and col == C_RED:
                for c0 in SHIP_COLS:
                    m = np.zeros((H, W), bool)
                    g = np.zeros((H, W, 3), np.uint8)
                    for gy, r in enumerate(rows):
                        for gx in range(c0, min(c0 + SHIP_W, len(r))):
                            if r[gx] == "1":
                                m[oy + gy, ox + gx] = True
                                g[oy + gy, ox + gx] = col
                    temp_masks.append(jnp.array(m))
                    temp_rgbs.append(jnp.array(g))

        self.attract_rgb = jnp.array(attract_rgb)
        self.attract_mask = jnp.array(attract_mask)
        self.red_icon_masks = jnp.stack(temp_masks)
        self.red_icon_rgbs = jnp.stack(temp_rgbs)

        go = np.zeros((H, W), bool)
        _bake_text(go, "GAME OVER", _centered_x("GAME OVER", 2, W), 95, 2)
        self.gameover_mask = jnp.array(go)

        self.digit_font = jnp.array(np.stack([_glyph_np(str(d)) for d in range(10)]))

        attract_path = _find_asset("attract_frames.npz")
        if attract_path is not None:
            self.attract_frames = jnp.array(np.load(attract_path)["frames"])
            self.attract_cycle = int(self.attract_frames.shape[0])
            print(f"Attract animation loaded from {attract_path}: {self.attract_cycle} frames")
        else:
            self.attract_frames = None
            self.attract_cycle = 117
            print("WARNING: attract_frames.npz not found — using fallback")

        hills_path = _find_asset("hills_cycle.npz")
        if hills_path is not None:
            h = np.load(hills_path)["hills"]
            self.hills_cycle = jnp.array(h)
            self.hills_period = int(h.shape[0])
            self.hills_top = 185
            print(f"Hills cycle loaded from {hills_path}: {self.hills_period} frames")
        else:
            self.hills_cycle = None
            self.hills_period = 20
            self.hills_top = 185
            print("WARNING: hills_cycle.npz not found — using procedural hills")

    def _blit_number_spaced(self, img, value, x0, y0, color, ndigits=4, spacing=10):
        clear_width = ndigits * spacing
        img = img.at[y0: y0 + 7, x0: x0 + clear_width].set(jnp.array([0, 0, 0], jnp.uint8))
        for i in range(ndigits):
            place = 10 ** (ndigits - 1 - i)
            d = (value // place) % 10
            glyph = self.digit_font[d]
            xi = x0 + i * spacing
            should_show = (value >= place) | (i == ndigits - 1)
            masked = jnp.where(
                should_show & (glyph[..., None] > 0),
                color,
                img[y0: y0 + 7, xi: xi + 5],
            )
            img = img.at[y0: y0 + 7, xi: xi + 5].set(masked)
        return img

    def _background(self, state):
        c = self.consts
        img = jnp.zeros((c.HEIGHT, c.WIDTH, 3), jnp.uint8)

        if self.hills_cycle is not None:
            facing = state.player_facing
            idx_fwd = (state.step_counter // 4) % self.hills_period
            idx_back = (-(state.step_counter // 4)) % self.hills_period
            idx = jnp.where(facing > 0, idx_fwd, idx_back)
            strip = self.hills_cycle[idx]
            img = img.at[self.hills_top:, :, :].set(strip)
            return img

        yy = jnp.arange(c.HEIGHT)[:, None]
        xx = jnp.arange(c.WIDTH)[None, :]
        scroll = state.step_counter.astype(jnp.float32) * c.HILL_SCROLL_SPEED
        u = jnp.mod(xx + scroll, 80.0) / 80.0 * 2 * jnp.pi
        wave = jnp.sin(u)
        horizon = c.HILL_Y + (wave * 4.0).astype(jnp.int32)
        grass = yy >= horizon
        depth = jnp.clip((yy - horizon).astype(jnp.float32) / 6.0, 0.0, 1.0)
        top = jnp.array([111, 210, 111], jnp.float32)
        base = jnp.array([26, 102, 26], jnp.float32)
        shade = (top[None, None, :] * (1.0 - depth[..., None])
                 + base[None, None, :] * depth[..., None]).astype(jnp.uint8)
        return jnp.where(grass[..., None], shade, img)

    def _draw_entities(self, img, state):
        c = self.consts

                # Bobo colors: yellow, purple, pink, blue — same speed as enemy
        BOBO_PALETTE = jnp.array([
            # Yellows
            [148, 116, 0], [181, 143, 0], [210, 164, 0], [232, 204, 99],
            [252, 224, 112], [252, 240, 148],
            # Purples
            [80, 0, 132], [104, 25, 154], [125, 48, 173], [146, 70, 192],
            [164, 89, 208], [181, 108, 224], [197, 124, 238],
            # Pinks
            [200, 72, 140], [212, 108, 195], [224, 124, 210],
            [236, 140, 224], [240, 128, 200], [252, 144, 224],
            # Blues
            [0, 0, 148], [24, 26, 167], [45, 50, 184], [66, 72, 200],
            [84, 92, 214], [101, 111, 228], [117, 128, 240],
        ], jnp.uint8)
        bobo_cycle_color = BOBO_PALETTE[(state.step_counter // 18) % 26]

        # Bobo itself uses the same colour
        # Bobo shows 1 leg during the shooting window (after dropping a bomb)
        bobo_shooting = (state.step_counter % self.consts.BOBO_BOMB_PERIOD) < 15
        bobo_sprite = jax.lax.select(
            bobo_shooting,
            BOBO_SPRITE_SHOOT,
            BOBO_SPRITE,
        )
        # Bobo hidden while player is exploding
        img = jax.lax.cond(
            state.player_explosion_active,
            lambda _: img,
            lambda _: draw_sprite(img, state.bobo_x, c.BOBO_Y, bobo_sprite, bobo_cycle_color),
            operand=None,
        )

        for i in range(c.MAX_BOMBS):
            def draw_bomb(active):
                def draw(_):
                    bomb_sprite = jnp.array([[1, 1, 1, 1]], dtype=jnp.bool_)
                    return draw_sprite(img, state.bomb_x[i], state.bomb_y[i], bomb_sprite,
                                       bobo_cycle_color)
                def skip(_):
                    return img
                # Hide bombs while player is exploding
                visible = active & (~state.player_explosion_active)
                return jax.lax.cond(visible, draw, skip, operand=None)
            img = draw_bomb(state.bomb_active[i])

        # Enemies = ring/UFO, move freely
        edy = enemy_display_y(state.enemy_y, state.enemy_type, state.enemy_phase,
                              state.step_counter, c.ENEMY_AMP)

        # Enemy palette — blue, purple, red, yellow, green
        ENEMY_PALETTE = jnp.array([
            # Blues
            [0, 0, 148], [24, 26, 167], [45, 50, 184], [66, 72, 200],
            [84, 92, 214], [101, 111, 228], [117, 128, 240],
            # Purples
            [80, 0, 132], [104, 25, 154], [125, 48, 173], [146, 70, 192],
            [164, 89, 208], [181, 108, 224], [197, 124, 238],
            # Reds
            [148, 0, 0], [167, 26, 26], [184, 50, 50], [200, 72, 72],
            [214, 92, 92], [228, 111, 111], [240, 128, 128],
            # Yellows
            [148, 116, 0], [181, 143, 0], [210, 164, 0], [232, 204, 99],
            [252, 224, 112], [252, 240, 148],
            # Greens
            [0, 68, 0], [20, 60, 0], [72, 160, 72], [92, 186, 92],
            [111, 210, 111], [144, 252, 144],
        ], jnp.uint8)
        enemy_cycle_color = ENEMY_PALETTE[(state.step_counter // 12) % 34]

        for i in range(c.NUM_ENEMIES):
            def draw_enemy(est_val, timer_val):
                def alive_fn(_):
                    # Cycle phase
                    WAVE_FRAMES = 90
                    DESCEND_FRAMES = 40
                    CYCLE = WAVE_FRAMES + DESCEND_FRAMES
                    phase = state.step_counter % CYCLE
                    in_wave = phase < WAVE_FRAMES

                    # Green/yellow palette during WAVE
                    color = enemy_cycle_color


                    sprite = jax.lax.switch(
                        state.enemy_type[i],
                        [lambda: SAUCER_SPRITE, lambda: BUZZIE_SPRITE, lambda: SQUEEZER_SPRITE],
                    )
                    return jax.lax.cond(
                        state.player_explosion_active,
                        lambda _: img,
                        lambda _: draw_sprite(img, state.enemy_x[i], edy[i], sprite, color),
                        operand=None,
                    )

                def reforming_fn(_):
                    progress = 1.0 - (timer_val / c.REFORM_FRAMES)
                    etype = state.enemy_type[i]
                    color = enemy_cycle_color
                    angles = jnp.array([0.0, jnp.pi/3, 2*jnp.pi/3, jnp.pi, 4*jnp.pi/3, 5*jnp.pi/3])
                    spread = 12.0 * (1.0 - progress)
                    img_local = img
                    for f in range(6):
                        fx = state.enemy_x[i] + jnp.cos(angles[f]) * spread
                        fy = edy[i] + jnp.sin(angles[f]) * spread
                        img_local = draw_rect(img_local, fx, fy, 2, 2, color)
                    return img_local

                def empty_fn(_):
                    return img

                return jax.lax.switch(est_val, [empty_fn, alive_fn, empty_fn, reforming_fn], operand=None)

            img = draw_enemy(state.enemy_state[i], state.enemy_timer[i])

        # Player bullets = green
        for i in range(c.MAX_BULLETS):
            def draw_bullet(active, x, y):
                def draw_fn(_):
                    return draw_sprite(img, x, y, BULLET_SPRITE_LOCAL,
                                       jnp.array(C_BULLET_GREEN, jnp.uint8))
                def skip_fn(_):
                    return img
                return jax.lax.cond(active, draw_fn, skip_fn, operand=None)
            img = draw_bullet(state.bullet_active[i], state.bullet_x[i], state.bullet_y[i])

        # Enemy explosions
        for i in range(c.NUM_ENEMIES):
            elapsed = c.EXPLOSION_DURATION - state.explosion_timer[i]
            for f in range(6):
                fx = state.explosion_x[i] + state.explosion_frag_vx[i, f] * elapsed
                fy = state.explosion_y[i] + state.explosion_frag_vy[i, f] * elapsed
                active = state.explosion_active[i]
                col = jnp.where(state.explosion_timer[i] > 6,
                                jnp.array([255, 220, 0], jnp.uint8),
                                jnp.array([255, 80, 0], jnp.uint8))
                w = jnp.where(active, 2, 0)
                img = draw_rect(img, fx, fy, w, w, col)

        # Player explosion — three clear phases:
        #   Phase 1 (0-35%):    explode outward (visible)
        #   Phase 2 (35-65%):   invisible (fragments gone)
        #   Phase 3 (65-100%):  reappear and merge to center (visible)
        p_t = state.player_explosion_timer
        elapsed = c.PLAYER_EXPLOSION_DURATION - p_t
        total = float(c.PLAYER_EXPLOSION_DURATION)

        phase1_end = total * 0.25
        phase2_end = total * 0.60

        # Spread factor and visibility
        spread = jnp.where(
            elapsed < phase1_end,
            elapsed / phase1_end,                            # 0 -> 1
            jnp.where(
                elapsed < phase2_end,
                1.0,                                          # still at max, but hidden
                (total - elapsed) / (total - phase2_end),     # 1 -> 0 (merge)
            ),
        )
        visible = ~((elapsed >= phase1_end) & (elapsed < phase2_end))

        offsets_x = jnp.array([-70.0, 70.0, -70.0, 70.0, 90.0])
        offsets_y = jnp.array([-80.0, -80.0, 80.0, 80.0, 0.0])

        for f in range(5):
            fx = state.player_explosion_x + offsets_x[f] * spread
            fy = state.player_explosion_y + offsets_y[f] * spread
            draw_now = state.player_explosion_active & visible
            w = jnp.where(draw_now, 10, 0)
            h = jnp.where(draw_now, 1, 0)
            img = draw_rect(img, fx, fy, w, h, jnp.array([214, 92, 92], jnp.uint8))

        show = ((state.invuln_timer <= 0) | ((state.step_counter // 4) % 2 == 0)) & (~state.player_explosion_active)

        def draw_player(should_show):
            def show_fn(_):
                sprite = jnp.where(state.player_facing > 0, PLAYER_SPRITE_RIGHT, PLAYER_SPRITE_LEFT)
                return draw_sprite(img, state.player_x, state.player_y, sprite,
                                   jnp.array(C_PLAYER_RED, jnp.uint8))
            def hide_fn(_):
                return img
            return jax.lax.cond(should_show, show_fn, hide_fn, operand=None)

        img = draw_player(show)
        return img

    def _render_attract(self, state):
        if self.attract_frames is not None:
            idx = state.step_counter % self.attract_cycle
            return self.attract_frames[idx]
        img = self._background(state)
        img = jnp.where(self.attract_mask[..., None], self.attract_rgb, img)
        return img

    def _render_play(self, state):
        img = self._background(state)

        color_blue = jnp.array(C_ATTRACT_BLUE, jnp.uint8)
        img = self._blit_number_spaced(img, state.score, 60, 16, color_blue, ndigits=4, spacing=11)

        lives = state.lives
        img = jnp.where((lives > 0) & self.red_icon_masks[0][..., None], self.red_icon_rgbs[0], img)
        img = jnp.where((lives > 1) & self.red_icon_masks[1][..., None], self.red_icon_rgbs[1], img)
        img = jnp.where((lives > 2) & self.red_icon_masks[2][..., None], self.red_icon_rgbs[2], img)


        img = self._draw_entities(img, state)
        return img

    def _render_gameover(self, state):
        if self.attract_frames is not None:
            idx = state.step_counter % self.attract_cycle
            img = self.attract_frames[idx]
        else:
            img = self._background(state)
            img = jnp.where(self.attract_mask[..., None], self.attract_rgb, img)

        black = jnp.array([0, 0, 0], jnp.uint8)
        img = img.at[0:35, :].set(black)

        color_blue = jnp.array(C_ATTRACT_BLUE, jnp.uint8)
        img = self._blit_number_spaced(img, state.score, 60, 16, color_blue, ndigits=4, spacing=11)
        return img

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: StarGunnerState):
        return jax.lax.switch(
            state.mode,
            [self._render_attract, self._render_play, self._render_gameover],
            state,
        )


def capture_sprites_from_ale(num_episodes=2, every_n=30, max_frames=400,
                             save_path="star_gunner_frames.npz"):
    import gymnasium as gym
    env = gym.make("ALE/StarGunner-v5", render_mode="rgb_array", frameskip=1)
    frames = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done, t = False, 0
        while not done and len(frames) < max_frames:
            obs, _, term, trunc, _ = env.step(env.action_space.sample())
            done = term or trunc
            if t % every_n == 0:
                frames.append(obs.copy())
            t += 1
    env.close()
    arr = np.asarray(frames, np.uint8)
    np.savez_compressed(save_path, frames=arr)
    print(f"{len(arr)} Frames gespeichert nach {save_path}")
    return arr


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    env = JaxStarGunner(start_in_play=False)
    _, g_state = env.reset(jax.random.PRNGKey(0))
    held = {"up": False, "down": False, "left": False, "right": False, "fire": False}

    def action_idx():
        u, d, l, r, f = (held["up"], held["down"], held["left"], held["right"], held["fire"])
        if f and u and r: return 14
        if f and u and l: return 15
        if f and d and r: return 16
        if f and d and l: return 17
        if f and u: return 10
        if f and r: return 11
        if f and l: return 12
        if f and d: return 13
        if f: return 1
        if u and r: return 6
        if u and l: return 7
        if d and r: return 8
        if d and l: return 9
        if u: return 2
        if r: return 3
        if l: return 4
        if d: return 5
        return 0

    fig, ax = plt.subplots(figsize=(4, 5.25))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.set_position([0, 0, 1, 1])
    ax.axis("off")
    im = ax.imshow(np.asarray(env.render(g_state)).astype(np.uint8), aspect="auto")

    def on_press(e):
        if e.key in ("up", "down", "left", "right"): held[e.key] = True
        elif e.key == " ": held["fire"] = True

    def on_release(e):
        if e.key in ("up", "down", "left", "right"): held[e.key] = False
        elif e.key == " ": held["fire"] = False

    fig.canvas.mpl_connect("key_press_event", on_press)
    fig.canvas.mpl_connect("key_release_event", on_release)

    def update(_):
        global g_state
        _, g_state, _, _, _ = env.step(g_state, jnp.array(action_idx(), jnp.int32))
        im.set_data(np.asarray(env.render(g_state)).astype(np.uint8))
        return [im]

    print("Arrow keys move, SPACE fires. Press SPACE on the title to start.")
    _ani = animation.FuncAnimation(fig, update, interval=33, blit=False, cache_frame_data=False)
    plt.show()