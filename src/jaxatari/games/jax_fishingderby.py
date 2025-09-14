import os
from functools import partial
from typing import NamedTuple, Tuple, List, Dict

import jax
import jax.numpy as jnp
import chex
import pygame
from dataclasses import dataclass

from gymnax.environments import spaces

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
import jaxatari.rendering.atraJaxis as aj
from jaxatari.renderers import AtraJaxisRenderer


def load_sprite_frame(path: str) -> chex.Array:
    import numpy as np
    if os.path.exists(path):
        return jnp.array(np.load(path))
    return None


@dataclass
class GameConfig:
    """All static configuration parameters for the game."""
    SCREEN_WIDTH: int = 160
    SCREEN_HEIGHT: int = 210
    SKY_COLOR: Tuple[int, int, int] = (100, 149, 237)
    WATER_COLOR: Tuple[int, int, int] = (60, 60, 160)
    WATER_Y_START: int = 64
    RESET: int = 18

    # Player and Rod/Hook
    P1_START_X: int = 20
    P2_START_X: int = 124
    PLAYER_Y: int = 34
    ROD_Y: int = 44  # Y position where rod extends horizontally

    # Rod mechanics
    MIN_ROD_LENGTH: int = 10  # Minimum rod extension
    MAX_ROD_LENGTH: int = 50  # Maximum rod extension
    ROD_SPEED: float = 1.5  # Speed of rod extension/retraction

    HOOK_WIDTH: int = 3
    HOOK_HEIGHT: int = 5
    HOOK_SPEED_V: float = 1.0
    REEL_SLOW_SPEED: float = 0.5
    REEL_FAST_SPEED: float = 1.5
    LINE_Y_START: int = 48
    LINE_Y_END: int = 160

    # Physics
    Acceleration: float = 0.2
    Damping: float = 0.85

    # Fish
    FISH_WIDTH: int = 8
    FISH_HEIGHT: int = 7
    FISH_SPEED: float = 0.8
    NUM_FISH: int = 6
    FISH_ROW_YS: Tuple[int] = (85, 101, 117, 133, 149, 165)
    FISH_ROW_SCORES: Tuple[int] = (2, 2, 4, 4, 6, 6)

    # Shark
    SHARK_WIDTH: int = 16
    SHARK_HEIGHT: int = 7
    SHARK_SPEED: float = 1.0
    SHARK_Y: int = 68


class PlayerState(NamedTuple):
    rod_length: chex.Array  # Length of horizontal rod extension
    hook_y: chex.Array  # Vertical position of hook (relative to rod end)
    score: chex.Array
    hook_state: chex.Array  # 0=free, 1=hooked/reeling slow, 2=reeling fast
    hooked_fish_idx: chex.Array
    hook_velocity_y: chex.Array  # Only vertical velocity now


class GameState(NamedTuple):
    p1: PlayerState
    p2: PlayerState
    fish_positions: chex.Array
    fish_directions: chex.Array
    fish_active: chex.Array
    shark_x: chex.Array
    shark_dir: chex.Array
    reeling_priority: chex.Array
    time: chex.Array
    game_over: chex.Array
    key: jax.random.PRNGKey


class FishingDerbyObservation(NamedTuple):
    player1_hook_xy: chex.Array
    fish_xy: chex.Array
    shark_x: chex.Array
    score: chex.Array


class FishingDerbyInfo(NamedTuple):
    p1_score: int
    p2_score: int
    time: int


# Game Logic
class FishingDerby(JaxEnvironment):
    def __init__(self):
        super().__init__()
        self.config = GameConfig()
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

    def _get_hook_position(self, player_x: float, player_state: PlayerState) -> Tuple[float, float]:
        """Calculate the actual hook position based on rod length and hook depth."""
        cfg = self.config
        rod_end_x = player_x + player_state.rod_length
        hook_x = rod_end_x
        hook_y = cfg.ROD_Y + player_state.hook_y
        return hook_x, hook_y

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jax.random.PRNGKey = jax.random.PRNGKey(10)) -> Tuple[FishingDerbyObservation, GameState]:
        key, fish_key = jax.random.split(key)

        p1_state = PlayerState(
            rod_length=jnp.array(float(self.config.MIN_ROD_LENGTH)),
            hook_y=jnp.array(0.0),  # Hook starts at rod level
            score=jnp.array(0),
            hook_state=jnp.array(0),
            hooked_fish_idx=jnp.array(-1, dtype=jnp.int32),
            hook_velocity_y=jnp.array(0.0)
        )

        p2_state = PlayerState(
            rod_length=jnp.array(float(self.config.MIN_ROD_LENGTH)),
            hook_y=jnp.array(0.0),
            score=jnp.array(0),
            hook_state=jnp.array(0),
            hooked_fish_idx=jnp.array(-1, dtype=jnp.int32),
            hook_velocity_y=jnp.array(0.0)
        )

        fish_x = jax.random.uniform(fish_key, (self.config.NUM_FISH,), minval=10.0,
                                    maxval=self.config.SCREEN_WIDTH - 20.0)
        fish_y = jnp.array(self.config.FISH_ROW_YS, dtype=jnp.float32)

        state = GameState(
            p1=p1_state, p2=p2_state,
            fish_positions=jnp.stack([fish_x, fish_y], axis=1),
            fish_directions=jax.random.choice(key, jnp.array([-1.0, 1.0]), (self.config.NUM_FISH,)),
            fish_active=jnp.ones(self.config.NUM_FISH, dtype=jnp.bool_),
            shark_x=jnp.array(self.config.SCREEN_WIDTH / 2.0),
            shark_dir=jnp.array(1.0),
            reeling_priority=jnp.array(-1),
            time=jnp.array(0),
            game_over=jnp.array(False),
            key=key
        )
        return self._get_observation(state), state

    def _get_observation(self, state: GameState) -> FishingDerbyObservation:
        hook_x, hook_y = self._get_hook_position(self.config.P1_START_X, state.p1)
        return FishingDerbyObservation(
            player1_hook_xy=jnp.array([hook_x, hook_y]),
            fish_xy=state.fish_positions,
            shark_x=state.shark_x,
            score=state.p1.score
        )

    def _get_reward(self, old_state: GameState, new_state: GameState) -> float:
        return new_state.p1.score - old_state.p1.score

    def _get_done(self, state: GameState) -> bool:
        return state.game_over

    def _get_info(self, state: GameState) -> FishingDerbyInfo:
        return FishingDerbyInfo(p1_score=state.p1.score, p2_score=state.p2.score, time=state.time)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: GameState, action: int) -> Tuple[
        FishingDerbyObservation, GameState, float, bool, FishingDerbyInfo]:
        """Processes one frame of the game and returns the full tuple."""
        new_state = self._step_logic(state, action)
        observation = self._get_observation(new_state)
        reward = self._get_reward(state, new_state)
        done = self._get_done(new_state)
        info = self._get_info(new_state)
        return observation, new_state, reward, done, info

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))

    def get_action_space(self) -> jnp.ndarray:
        return jnp.array(self.action_set)

    def _step_logic(self, state: GameState, p1_action: int) -> GameState:
        """The core logic for a single game step, returning only the new state."""
        cfg = self.config

        def reset_branch(_):
            _, new_state = self.reset(state.key)
            return new_state

        def safe_set_at(arr, idx, value, pred):
            def do_set(a):
                return a.at[idx].set(value)

            return jax.lax.cond(pred, do_set, lambda a: a, arr)

        def game_branch(_):
            # Fish movement
            new_fish_x = state.fish_positions[:, 0] + state.fish_directions * cfg.FISH_SPEED
            new_fish_dirs = jnp.where(
                (new_fish_x < 0) | (new_fish_x > cfg.SCREEN_WIDTH - cfg.FISH_WIDTH),
                -state.fish_directions,
                state.fish_directions
            )
            new_fish_x = jnp.clip(new_fish_x, 0, cfg.SCREEN_WIDTH - cfg.FISH_WIDTH)
            new_fish_pos = state.fish_positions.at[:, 0].set(new_fish_x)

            # Shark movement
            new_shark_x = state.shark_x + state.shark_dir * cfg.SHARK_SPEED
            new_shark_dir = jnp.where(
                (new_shark_x < 0) | (new_shark_x > cfg.SCREEN_WIDTH - cfg.SHARK_WIDTH),
                -state.shark_dir,
                state.shark_dir
            )
            new_shark_x = jnp.clip(new_shark_x, 0, cfg.SCREEN_WIDTH - cfg.SHARK_WIDTH)

            # Player 1 Rod and Hook Logic
            p1 = state.p1

            # Rod length control (horizontal extension)
            rod_change = 0.0
            rod_change = jnp.where(p1_action == Action.RIGHT, +cfg.ROD_SPEED, rod_change)
            rod_change = jnp.where(p1_action == Action.LEFT, -cfg.ROD_SPEED, rod_change)

            new_rod_length = jnp.clip(
                p1.rod_length + rod_change,
                cfg.MIN_ROD_LENGTH,
                cfg.MAX_ROD_LENGTH
            )

            # Hook vertical movement (only when not hooked to fish)
            hook_change = 0.0
            # Only allow vertical movement when free (hook_state == 0)
            can_move_vertically = (p1.hook_state == 0)
            hook_change = jnp.where(can_move_vertically & (p1_action == Action.DOWN), +cfg.Acceleration, hook_change)
            hook_change = jnp.where(can_move_vertically & (p1_action == Action.UP), -cfg.Acceleration, hook_change)

            # Update hook velocity with damping
            new_hook_velocity_y = p1.hook_velocity_y * cfg.Damping + hook_change

            # Calculate hook position limits
            min_hook_y = 0.0  # At rod level
            max_hook_y = cfg.LINE_Y_END - cfg.ROD_Y  # Maximum depth

            # Update hook position
            new_hook_y = jnp.clip(
                p1.hook_y + new_hook_velocity_y,
                min_hook_y,
                max_hook_y
            )

            # Kill velocity if hitting bounds
            new_hook_velocity_y = jnp.where(
                (new_hook_y == min_hook_y) | (new_hook_y == max_hook_y),
                0.0,
                new_hook_velocity_y
            )

            # Get actual hook position in world coordinates
            hook_x, hook_y = self._get_hook_position(cfg.P1_START_X, PlayerState(
                rod_length=new_rod_length,
                hook_y=new_hook_y,
                score=p1.score,
                hook_state=p1.hook_state,
                hooked_fish_idx=p1.hooked_fish_idx,
                hook_velocity_y=new_hook_velocity_y
            ))

            # Collision and Game Logic
            fish_active, reeling_priority = state.fish_active, state.reeling_priority
            can_hook = (p1.hook_state == 0)
            hook_collides_fish = (jnp.abs(new_fish_pos[:, 0] - hook_x) < cfg.FISH_WIDTH) & (
                    jnp.abs(new_fish_pos[:, 1] - hook_y) < cfg.FISH_HEIGHT)
            valid_hook_targets = can_hook & fish_active & hook_collides_fish

            hooked_fish_idx, did_hook_fish = jnp.argmax(valid_hook_targets), jnp.any(valid_hook_targets)

            p1_hook_state = jnp.where(did_hook_fish, 1, p1.hook_state)
            p1_hooked_fish_idx = jnp.where(did_hook_fish, hooked_fish_idx, p1.hooked_fish_idx)
            fish_active = fish_active.at[hooked_fish_idx].set(
                jnp.where(did_hook_fish, False, fish_active[hooked_fish_idx])
            )
            reeling_priority = jnp.where(did_hook_fish & (reeling_priority == -1), 0, reeling_priority)

            # Fast reel with FIRE button
            can_reel_fast = (p1_action == Action.FIRE) & (p1_hook_state == 1) & (
                    (reeling_priority == -1) | (reeling_priority == 0))
            p1_hook_state = jnp.where(can_reel_fast, 2, p1_hook_state)
            reeling_priority = jnp.where(can_reel_fast, 0, reeling_priority)

            # Reeling mechanics (moves hook upward toward rod)
            reel_speed = jnp.where(p1_hook_state == 2, cfg.REEL_FAST_SPEED, cfg.REEL_SLOW_SPEED)
            new_hook_y = jnp.where(p1_hook_state > 0,
                                   jnp.clip(new_hook_y - reel_speed, min_hook_y, max_hook_y),
                                   new_hook_y)
            new_hook_velocity_y = jnp.where(p1_hook_state > 0, 0.0, new_hook_velocity_y)

            # Update hook position after reeling
            hook_x, hook_y = self._get_hook_position(cfg.P1_START_X, PlayerState(
                rod_length=new_rod_length,
                hook_y=new_hook_y,
                score=p1.score,
                hook_state=p1_hook_state,
                hooked_fish_idx=p1_hooked_fish_idx,
                hook_velocity_y=new_hook_velocity_y
            ))

            # Hooked fish follows the hook
            hooked_fish_pos = jnp.array([hook_x, hook_y])
            has_hook = (p1_hook_state > 0) & (p1_hooked_fish_idx >= 0)
            new_fish_pos = safe_set_at(new_fish_pos, p1_hooked_fish_idx, hooked_fish_pos, has_hook)

            # Scoring and collision detection
            p1_score, key = p1.score, state.key
            shark_collides = (p1_hook_state > 0) & (jnp.abs(hook_x - new_shark_x) < cfg.SHARK_WIDTH) & (
                    jnp.abs(hook_y - cfg.SHARK_Y) < cfg.SHARK_HEIGHT)
            scored_fish = (p1_hook_state > 0) & (hook_y <= cfg.ROD_Y + 5)  # Fish reaches near the rod
            reset_hook = shark_collides | scored_fish

            prev_idx = p1_hooked_fish_idx

            fish_scores = jnp.array(cfg.FISH_ROW_SCORES)
            p1_score += jnp.where(scored_fish, fish_scores[p1_hooked_fish_idx], 0)

            # Fish respawn
            def respawn_fish(all_pos, all_dirs, all_active, idx, key):
                kx, kdir = jax.random.split(key)
                new_x = jax.random.uniform(kx, minval=10.0, maxval=cfg.SCREEN_WIDTH - cfg.FISH_WIDTH)
                new_y = jnp.array(cfg.FISH_ROW_YS, dtype=jnp.float32)[idx]
                new_pos = all_pos.at[idx].set(jnp.array([new_x, new_y]))
                new_dir = all_dirs.at[idx].set(jax.random.choice(kdir, jnp.array([-1.0, 1.0])))
                new_act = all_active.at[idx].set(True)
                return new_pos, new_dir, new_act

            key, respawn_key = jax.random.split(key)
            do_respawn = reset_hook & (prev_idx >= 0)

            new_fish_pos, new_fish_dirs, fish_active = jax.lax.cond(
                do_respawn,
                lambda _: respawn_fish(new_fish_pos, new_fish_dirs, fish_active, prev_idx, respawn_key),
                lambda _: (new_fish_pos, new_fish_dirs, fish_active),
                operand=None
            )

            # Clear reeling priority and reset hook state
            reeling_priority = jnp.where(do_respawn & (reeling_priority == 0), -1, reeling_priority)
            fish_active = jnp.where(
                reset_hook,
                fish_active.at[p1_hooked_fish_idx].set(True),
                fish_active
            )

            # Reset hook after scoring/shark collision
            p1_hook_state = jnp.where(reset_hook, 0, p1_hook_state)
            p1_hooked_fish_idx = jnp.where(reset_hook, -1, p1_hooked_fish_idx)
            new_rod_length = jnp.where(reset_hook, float(cfg.MIN_ROD_LENGTH), new_rod_length)
            new_hook_y = jnp.where(reset_hook, 0.0, new_hook_y)
            new_hook_velocity_y = jnp.where(reset_hook, 0.0, new_hook_velocity_y)

            game_over = (p1_score >= 99) | (state.p2.score >= 99)

            return GameState(
                p1=PlayerState(
                    rod_length=new_rod_length,
                    hook_y=new_hook_y,
                    score=p1_score,
                    hook_state=p1_hook_state,
                    hooked_fish_idx=p1_hooked_fish_idx,
                    hook_velocity_y=new_hook_velocity_y
                ),
                p2=state.p2,
                fish_positions=new_fish_pos,
                fish_directions=new_fish_dirs,
                fish_active=fish_active,
                shark_x=new_shark_x,
                shark_dir=new_shark_dir,
                reeling_priority=reeling_priority,
                time=state.time + 1,
                game_over=game_over,
                key=key
            )

        return jax.lax.cond(p1_action == cfg.RESET, reset_branch, game_branch, state)


def normalize_frame(frame: chex.Array, target_shape: Tuple[int, int, int]) -> chex.Array:
    """Crop or pad a sprite to the target shape with transparent (255) background."""
    h, w, c = frame.shape
    th, tw, tc = target_shape
    assert c == tc

    # Crop if larger
    frame = frame[:min(h, th), :min(w, tw), :]

    # Pad if smaller (with 255 = white = transparent)
    pad_h = th - frame.shape[0]
    pad_w = tw - frame.shape[1]

    frame = jnp.pad(
        frame,
        ((0, pad_h), (0, pad_w), (0, 0)),
        mode="constant",
        constant_values=255
    )
    return frame


def load_sprites():
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

    sprite_paths = {
        'background': os.path.join(MODULE_DIR, "sprites/fishingderby/background.npy"),
        'player1': os.path.join(MODULE_DIR, "sprites/fishingderby/player1.npy"),
        'player2': os.path.join(MODULE_DIR, "sprites/fishingderby/player2.npy"),
        'shark1': os.path.join(MODULE_DIR, "sprites/fishingderby/shark_new_1.npy"),
        'shark2': os.path.join(MODULE_DIR, "sprites/fishingderby/shark_new_2.npy"),
        'fish1': os.path.join(MODULE_DIR, "sprites/fishingderby/fish1.npy"),
        'fish2': os.path.join(MODULE_DIR, "sprites/fishingderby/fish2.npy"),
        'sky': os.path.join(MODULE_DIR, "sprites/fishingderby/sky.npy"),
        **{f"score_{i}": os.path.join(MODULE_DIR, f"sprites/fishingderby/score_{i}.npy") for i in range(10)}
    }

    sprites = {}
    for name, path in sprite_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Sprite file not found: {path}")
        sprite = aj.loadFrame(path, transpose=False)
        if sprite is None:
            raise ValueError(f"Failed to load sprite: {path}")
        sprites[name] = sprite

    # Normalize shark frames
    shark_sprites = [sprites['shark1'], sprites['shark2']]
    max_shape = (
        max(s.shape[0] for s in shark_sprites),
        max(s.shape[1] for s in shark_sprites),
        shark_sprites[0].shape[2]
    )
    sprites['shark1'] = normalize_frame(sprites['shark1'], max_shape)
    sprites['shark2'] = normalize_frame(sprites['shark2'], max_shape)

    # Normalize fish frames
    fish_sprites = [sprites['fish1'], sprites['fish2']]
    max_shape = (
        max(s.shape[0] for s in fish_sprites),
        max(s.shape[1] for s in fish_sprites),
        fish_sprites[0].shape[2]
    )
    sprites['fish1'] = normalize_frame(sprites['fish1'], max_shape)
    sprites['fish2'] = normalize_frame(sprites['fish2'], max_shape)

    # Score digits
    score_digits = jnp.stack([sprites[f'score_{i}'] for i in range(10)])

    return (
        sprites['background'], sprites['player1'], sprites['player2'],
        sprites['shark1'], sprites['shark2'], sprites['fish1'], sprites['fish2'],
        sprites['sky'], score_digits
    )


class FishingDerbyRenderer(AtraJaxisRenderer):
    def __init__(self):
        self.config = GameConfig()
        (
            self.SPRITE_BG, self.SPRITE_PLAYER1, self.SPRITE_PLAYER2,
            self.SPRITE_SHARK1, self.SPRITE_SHARK2, self.SPRITE_FISH1,
            self.SPRITE_FISH2, self.SPRITE_SKY, self.SPRITE_SCORE_DIGITS
        ) = load_sprites()

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: GameState) -> chex.Array:
        cfg = self.config
        raster = jnp.zeros((cfg.SCREEN_HEIGHT, cfg.SCREEN_WIDTH, 3), dtype=jnp.uint8)

        # Draw sky
        raster = raster.at[:cfg.WATER_Y_START, :, :].set(jnp.array(cfg.SKY_COLOR, dtype=jnp.uint8))
        # Draw water
        raster = raster.at[cfg.WATER_Y_START:, :, :].set(jnp.array(cfg.WATER_COLOR, dtype=jnp.uint8))

        # Draw players
        raster = self._render_at(raster, cfg.P1_START_X, cfg.PLAYER_Y, self.SPRITE_PLAYER1)
        raster = self._render_at(raster, cfg.P2_START_X, cfg.PLAYER_Y, self.SPRITE_PLAYER2)

        # Draw fishing rods and lines (proper Atari-style)
        # Player 1 rod and line
        p1_rod_end_x = cfg.P1_START_X + state.p1.rod_length
        p1_hook_x, p1_hook_y = cfg.P1_START_X + state.p1.rod_length, cfg.ROD_Y + state.p1.hook_y

        # Draw horizontal rod
        raster = self._render_line(raster, cfg.P1_START_X + 10, cfg.ROD_Y, p1_rod_end_x, cfg.ROD_Y,
                                   color=(139, 69, 19))  # Brown rod
        # Draw vertical line from rod end to hook
        raster = self._render_line(raster, p1_rod_end_x, cfg.ROD_Y, p1_hook_x, p1_hook_y,
                                   color=(200, 200, 200))  # Gray line

        # Player 2 rod and line (simplified for now, same as before)
        raster = self._render_line(raster, cfg.P2_START_X + 2, cfg.PLAYER_Y + 10, state.p2.rod_length + cfg.P2_START_X,
                                   cfg.ROD_Y + state.p2.hook_y)

        # Draw shark
        shark_frame = jax.lax.cond((state.time // 8) % 2 == 0, lambda: self.SPRITE_SHARK1, lambda: self.SPRITE_SHARK2)
        raster = self._render_at(raster, state.shark_x, cfg.SHARK_Y, shark_frame, flip_h=state.shark_dir < 0)

        # Draw fish
        fish_frame = jax.lax.cond((state.time // 10) % 2 == 0, lambda: self.SPRITE_FISH1, lambda: self.SPRITE_FISH2)

        def draw_one_fish(i, r):
            pos, direction, active = state.fish_positions[i], state.fish_directions[i], state.fish_active[i]
            return jax.lax.cond(active,
                                lambda r_in: self._render_at(r_in, pos[0], pos[1], fish_frame, flip_h=direction > 0),
                                lambda r_in: r_in, r)

        raster = jax.lax.fori_loop(0, cfg.NUM_FISH, draw_one_fish, raster)

        # Draw hooked fish
        def draw_hooked_p1(r):
            hook_x, hook_y = cfg.P1_START_X + state.p1.rod_length, cfg.ROD_Y + state.p1.hook_y
            return self._render_at(r, hook_x, hook_y, fish_frame)

        def draw_hooked_p2(r):
            hook_x, hook_y = cfg.P2_START_X + state.p2.rod_length, cfg.ROD_Y + state.p2.hook_y
            return self._render_at(r, hook_x, hook_y, fish_frame)

        raster = jax.lax.cond(state.p1.hook_state > 0, draw_hooked_p1, lambda r: r, raster)
        raster = jax.lax.cond(state.p2.hook_state > 0, draw_hooked_p2, lambda r: r, raster)

        # Draw scores
        raster = self._render_score(raster, state.p1.score, 50, 20)
        raster = self._render_score(raster, state.p2.score, 100, 20)

        return jnp.transpose(raster, (1, 0, 2))

    def _render_score(self, raster, score, x, y):
        s1, s0 = score // 10, score % 10
        digit1_sprite, digit0_sprite = self.SPRITE_SCORE_DIGITS[s1], self.SPRITE_SCORE_DIGITS[s0]
        raster = self._render_at(raster, x, y, digit1_sprite)
        raster = self._render_at(raster, x + 7, y, digit0_sprite)
        return raster

    @staticmethod
    @jax.jit
    def _render_at(raster, x, y, sprite, flip_h=False):
        sprite_rgb = sprite[:, :, :3]
        h, w = sprite.shape[0], sprite.shape[1]
        x, y = jnp.round(x).astype(jnp.int32), jnp.round(y).astype(jnp.int32)
        sprite_to_draw = jnp.where(flip_h, jnp.fliplr(sprite_rgb), sprite_rgb)

        # Check if sprite has an alpha channel (4th channel)
        has_alpha = sprite.shape[2] > 3

        if has_alpha:
            # Use the alpha channel for transparency
            alpha = sprite[:, :, 3:4]
            alpha = jnp.where(flip_h, jnp.fliplr(alpha), alpha)
            # Create mask where alpha > 0 (non-transparent pixels)
            mask = alpha > 0
        else:
            # Fallback: treat both pure black (0,0,0) and pure white (255,255,255) as transparent
            is_black = jnp.all(sprite_to_draw == 0, axis=-1, keepdims=True)
            is_white = jnp.all(sprite_to_draw == 255, axis=-1, keepdims=True)
            # Mask is True where pixel is NOT black and NOT white
            mask = ~(is_black | is_white)

        region = jax.lax.dynamic_slice(raster, (y, x, 0), (h, w, 3))
        patch = jnp.where(mask, sprite_to_draw, region)

        return jax.lax.dynamic_update_slice(raster, patch, (y, x, 0))

    @staticmethod
    @jax.jit
    def _render_line(raster, x0, y0, x1, y1, color=(200, 200, 200)):
        x0, y0, x1, y1 = jnp.round(jnp.array([x0, y0, x1, y1])).astype(jnp.int32)
        dx, sx, dy, sy = jnp.abs(x1 - x0), jnp.sign(x1 - x0), -jnp.abs(y1 - y0), jnp.sign(y1 - y0)
        err = dx + dy
        color_uint8 = jnp.array(color, dtype=jnp.uint8)

        def loop_body(carry):
            x, y, r, e = carry
            safe_y, safe_x = jnp.clip(y, 0, r.shape[0] - 1), jnp.clip(x, 0, r.shape[1] - 1)
            r = r.at[safe_y, safe_x, :].set(color_uint8)
            e2 = 2 * e
            e_new = jnp.where(e2 >= dy, e + dy, e)
            x_new = jnp.where(e2 >= dy, x + sx, x)
            e_final = jnp.where(e2 <= dx, e_new + dx, e_new)
            y_new = jnp.where(e2 <= dx, y + sy, y)
            return x_new, y_new, r, e_final

        def loop_cond(carry):
            return ~((carry[0] == x1) & (carry[1] == y1))

        _, _, raster, _ = jax.lax.while_loop(loop_cond, loop_body, (x0, y0, raster, err))
        return raster


def get_human_action() -> chex.Array:
    keys = pygame.key.get_pressed()
    up = keys[pygame.K_w] or keys[pygame.K_UP]
    down = keys[pygame.K_s] or keys[pygame.K_DOWN]
    left = keys[pygame.K_a] or keys[pygame.K_LEFT]
    right = keys[pygame.K_d] or keys[pygame.K_RIGHT]
    fire = keys[pygame.K_SPACE]
    reset = keys[pygame.K_r]

    if reset:
        return jnp.array(GameConfig.RESET)

    x, y = 0, 0
    if up and not down:
        y = 1
    elif not up and down:
        y = -1

    if left and not right:
        x = -1
    elif not left and right:
        x = 1

    if fire:
        if x == -1 and y == -1:
            return jnp.array(Action.DOWNLEFTFIRE)
        elif x == -1 and y == 1:
            return jnp.array(Action.UPLEFTFIRE)
        elif x == 1 and y == -1:
            return jnp.array(Action.DOWNRIGHTFIRE)
        elif x == 1 and y == 1:
            return jnp.array(Action.UPRIGHTFIRE)
        elif x == 0 and y == -1:
            return jnp.array(Action.DOWNFIRE)
        else:
            return jnp.array(Action.FIRE)
    else:
        if x == -1 and y == -1:
            return jnp.array(Action.DOWNLEFT)
        elif x == -1 and y == 1:
            return jnp.array(Action.UPLEFT)
        elif x == 1 and y == -1:
            return jnp.array(Action.DOWNRIGHT)
        elif x == 1 and y == 1:
            return jnp.array(Action.UPRIGHT)
        elif x == -1:
            return jnp.array(Action.LEFT)
        elif x == 1:
            return jnp.array(Action.RIGHT)
        elif y == -1:
            return jnp.array(Action.DOWN)
        elif y == 1:
            return jnp.array(Action.UP)

    return jnp.array(Action.NOOP)


if __name__ == "__main__":
    pygame.init()
    game = FishingDerby()
    renderer = FishingDerbyRenderer()
    jitted_step = jax.jit(game.step)
    jitted_reset = jax.jit(game.reset)
    scaling = 4
    screen = pygame.display.set_mode((GameConfig.SCREEN_WIDTH * scaling, GameConfig.SCREEN_HEIGHT * scaling))
    (_, curr_state) = jitted_reset()
    running = True
    frame_by_frame = False
    frameskip = 1
    counter = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    frame_by_frame = not frame_by_frame
                elif event.key == pygame.K_n and frame_by_frame:
                    action = get_human_action()
                    (_, curr_state, _, _, _) = jitted_step(curr_state, action)

        if not frame_by_frame:
            action = get_human_action()
            if counter % frameskip == 0:
                (_, curr_state, _, _, _) = jitted_step(curr_state, action)

        # Render and display
        raster = renderer.render(curr_state)
        raster_np = jnp.array(raster).transpose((1, 0, 2))
        aj.update_pygame(screen, raster, scaling, GameConfig.SCREEN_WIDTH, GameConfig.SCREEN_HEIGHT)

        counter += 1
        pygame.time.Clock().tick(60)
    pygame.quit()

# run with: python scripts/play.py --game src/jaxatari/games/jax_fishingderby.py --record my_record_file.npz