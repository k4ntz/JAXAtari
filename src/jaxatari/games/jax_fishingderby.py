import os
from functools import partial
from typing import NamedTuple, Tuple, List, Dict

import aj
import jax
import jax.numpy as jnp
import chex
import pygame
from dataclasses import dataclass
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
    RESET:int  = 18
    # Player and Hook
    P1_START_X: int = 20
    P2_START_X: int = 124
    PLAYER_Y: int = 34
    HOOK_WIDTH: int = 3
    HOOK_HEIGHT: int = 5
    HOOK_SPEED_H: float = 1.2
    HOOK_SPEED_V: float = 1.0
    REEL_SLOW_SPEED: float = 0.5
    REEL_FAST_SPEED: float = 1.5
    LINE_Y_START: int = 48
    LINE_Y_END: int = 160

    # Fish
    FISH_WIDTH: int = 8
    FISH_HEIGHT: int = 7
    FISH_SPEED: float = 0.6
    NUM_FISH: int = 6
    FISH_ROW_YS: Tuple[int] = (85, 101, 117, 133, 149, 165)
    FISH_ROW_SCORES: Tuple[int] = (2, 2, 4, 4, 6, 6)

    # Shark
    SHARK_WIDTH: int = 16
    SHARK_HEIGHT: int = 7
    SHARK_SPEED: float = 1.0
    SHARK_Y: int = 68



class PlayerState(NamedTuple):
    hook_x: chex.Array
    hook_y: chex.Array
    score: chex.Array
    hook_state: chex.Array
    hooked_fish_idx: chex.Array


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

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jax.random.PRNGKey = jax.random.PRNGKey(10)) -> Tuple[FishingDerbyObservation, GameState, ]:
        key, fish_key = jax.random.split(key)

        p1_state = PlayerState(
            hook_x=jnp.array(self.config.P1_START_X + 8.0), hook_y=jnp.array(float(self.config.LINE_Y_START)),
            score=jnp.array(0), hook_state=jnp.array(0), hooked_fish_idx=jnp.array(-1, dtype=jnp.int32)
        )
        p2_state = PlayerState(
            hook_x=jnp.array(self.config.P2_START_X + 8.0), hook_y=jnp.array(float(self.config.LINE_Y_START)),
            score=jnp.array(0), hook_state=jnp.array(0), hooked_fish_idx=jnp.array(-1, dtype=jnp.int32)
        )

        fish_x = jax.random.uniform(fish_key, (self.config.NUM_FISH,), minval=10.0,
                                    maxval=self.config.SCREEN_WIDTH - 20.0)
        fish_y = jnp.array(self.config.FISH_ROW_YS, dtype=jnp.float32)

        state = GameState(
            p1=p1_state, p2=p2_state, fish_positions=jnp.stack([fish_x, fish_y], axis=1),
            fish_directions=jax.random.choice(key, jnp.array([-1.0, 1.0]), (self.config.NUM_FISH,)),
            fish_active=jnp.ones(self.config.NUM_FISH, dtype=jnp.bool_),
            shark_x=jnp.array(self.config.SCREEN_WIDTH / 2.0), shark_dir=jnp.array(1.0),
            reeling_priority=jnp.array(-1), time=jnp.array(0), game_over=jnp.array(False), key=key
        )
        return self._get_observation(state), state


    def _get_observation(self, state: GameState) -> FishingDerbyObservation:
        return FishingDerbyObservation(
            player1_hook_xy=jnp.array([state.p1.hook_x, state.p1.hook_y]),
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

    def _step_logic(self, state: GameState, p1_action: int) -> GameState:
        """The core logic for a single game step, returning only the new state."""
        cfg = self.config

        def reset_branch(_):
            _, new_state = self.reset(state.key)
            return new_state

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

            # Player 1 Hook Logic
            p1 = state.p1
            dx = jnp.where(p1_action == Action.RIGHT, cfg.HOOK_SPEED_H,
                           jnp.where(p1_action == Action.LEFT, -cfg.HOOK_SPEED_H, 0.0))
            p1_hook_x = jnp.clip(p1.hook_x + dx, 0, cfg.SCREEN_WIDTH / 2 - cfg.HOOK_WIDTH)
            dy = jnp.where(p1_action == Action.DOWN, cfg.HOOK_SPEED_V,
                           jnp.where(p1_action == Action.UP, -cfg.HOOK_SPEED_V, 0.0))
            p1_hook_y = jnp.where(p1.hook_state == 0,
                                  jnp.clip(p1.hook_y + dy, cfg.LINE_Y_START, cfg.LINE_Y_END),
                                  p1.hook_y)

            # Collision and Game Logic
            fish_active, reeling_priority = state.fish_active, state.reeling_priority
            can_hook = (p1.hook_state == 0)
            hook_collides_fish = (jnp.abs(new_fish_pos[:, 0] - p1_hook_x) < cfg.FISH_WIDTH) & (
                jnp.abs(new_fish_pos[:, 1] - p1_hook_y) < cfg.FISH_HEIGHT)
            valid_hook_targets = can_hook & fish_active & hook_collides_fish

            hooked_fish_idx, did_hook_fish = jnp.argmax(valid_hook_targets), jnp.any(valid_hook_targets)

            p1_hook_state = jnp.where(did_hook_fish, 1, p1.hook_state)
            p1_hooked_fish_idx = jnp.where(did_hook_fish, hooked_fish_idx, p1.hooked_fish_idx)
            fish_active = fish_active.at[hooked_fish_idx].set(
                jnp.where(did_hook_fish, False, fish_active[hooked_fish_idx])
            )
            reeling_priority = jnp.where(did_hook_fish & (reeling_priority == -1), 0, reeling_priority)

            can_reel_fast = (p1_action == Action.FIRE) & (p1_hook_state == 1) & (
                (reeling_priority == -1) | (reeling_priority == 0))
            p1_hook_state = jnp.where(can_reel_fast, 2, p1_hook_state)
            reeling_priority = jnp.where(can_reel_fast, 0, reeling_priority)

            reel_speed = jnp.where(p1_hook_state == 2, cfg.REEL_FAST_SPEED, cfg.REEL_SLOW_SPEED)
            p1_hook_y = jnp.where(p1_hook_state > 0, p1_hook_y - reel_speed, p1_hook_y)

            hooked_fish_pos = jnp.array([p1_hook_x, p1_hook_y])
            new_fish_pos = new_fish_pos.at[p1_hooked_fish_idx].set(
                jnp.where(p1_hook_state > 0, hooked_fish_pos, new_fish_pos[p1_hooked_fish_idx])
            )

            p1_score, key = p1.score, state.key
            shark_collides = (p1_hook_state > 0) & (jnp.abs(p1_hook_x - new_shark_x) < cfg.SHARK_WIDTH) & (
                jnp.abs(p1_hook_y - cfg.SHARK_Y) < cfg.SHARK_HEIGHT)
            scored_fish = (p1_hook_state > 0) & (p1_hook_y <= cfg.LINE_Y_START)
            reset_hook = shark_collides | scored_fish

            fish_scores = jnp.array(cfg.FISH_ROW_SCORES)
            p1_score += jnp.where(scored_fish, fish_scores[p1_hooked_fish_idx], 0)

            def respawn_fish_fn(all_fish_pos, idx_to_respawn, respawn_key):
                new_x = jax.random.uniform(respawn_key, minval=10.0, maxval=cfg.SCREEN_WIDTH - 20.0)
                return all_fish_pos.at[idx_to_respawn, 0].set(new_x)

            key, respawn_key = jax.random.split(key)
            new_fish_pos = jax.lax.cond(
                reset_hook,
                lambda: respawn_fish_fn(new_fish_pos, p1_hooked_fish_idx, respawn_key),
                lambda: new_fish_pos
            )

            fish_active = jnp.where(
                reset_hook,
                fish_active.at[p1_hooked_fish_idx].set(True),
                fish_active
            )
            reeling_priority = jnp.where(
                reset_hook & (reeling_priority == 0),
                -1,
                reeling_priority
            )
            p1_hook_state = jnp.where(reset_hook, 0, p1_hook_state)
            p1_hooked_fish_idx = jnp.where(reset_hook, -1, p1_hooked_fish_idx)

            game_over = (p1_score >= 99) | (state.p2.score >= 99)

            return GameState(
                p1=PlayerState(p1_hook_x, p1_hook_y, p1_score, p1_hook_state, p1_hooked_fish_idx),
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

        return jax.lax.cond(p1_action == cfg.RESET, reset_branch, game_branch, None)

def load_sprites():
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

    sprite_paths = {
        'background': os.path.join(MODULE_DIR, "sprites/fishingderby/background.npy"),
        'player1': os.path.join(MODULE_DIR, "sprites/fishingderby/player1.npy"),
        'player2': os.path.join(MODULE_DIR, "sprites/fishingderby/player2.npy"),
        'shark1': os.path.join(MODULE_DIR, "sprites/fishingderby/shark1.npy"),
        'shark2': os.path.join(MODULE_DIR, "sprites/fishingderby/shark2.npy"),
        'fish1': os.path.join(MODULE_DIR, "sprites/fishingderby/fish1.npy"),
        'fish2': os.path.join(MODULE_DIR, "sprites/fishingderby/fish2.npy"),
        'sky': os.path.join(MODULE_DIR, "sprites/fishingderby/sky.npy"),
        'score_0': os.path.join(MODULE_DIR, "sprites/fishingderby/score_0.npy"),
        'score_1': os.path.join(MODULE_DIR, "sprites/fishingderby/score_1.npy"),
        'score_2': os.path.join(MODULE_DIR, "sprites/fishingderby/score_2.npy"),
        'score_3': os.path.join(MODULE_DIR, "sprites/fishingderby/score_3.npy"),
        'score_4': os.path.join(MODULE_DIR, "sprites/fishingderby/score_4.npy"),
        'score_5': os.path.join(MODULE_DIR, "sprites/fishingderby/score_5.npy"),
        'score_6': os.path.join(MODULE_DIR, "sprites/fishingderby/score_6.npy"),
        'score_7': os.path.join(MODULE_DIR, "sprites/fishingderby/score_7.npy"),
        'score_8': os.path.join(MODULE_DIR, "sprites/fishingderby/score_8.npy"),
        'score_9': os.path.join(MODULE_DIR, "sprites/fishingderby/score_9.npy"),
    }

    sprites = {}
    for name, path in sprite_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Sprite file not found: {path}")
        sprite = aj.loadFrame(path, transpose=True)
        if sprite is None:
            raise ValueError(f"Failed to load sprite: {path}")
        sprites[name] =	sprite

    # Pad shark sprites to the same size
    shark_sprites = [sprites['shark1'], sprites['shark2']]
    max_shark_height = max(s.shape[0] for s in shark_sprites)
    max_shark_width = max(s.shape[1] for s in shark_sprites)
    padded_shark_sprites = [
        jnp.pad(
            s,
            ((0, max_shark_height - s.shape[0]), (0, max_shark_width - s.shape[1]), (0, 0)),
            mode='constant',
            constant_values=0
        )
        for s in shark_sprites
    ]
    sprites['shark1'], sprites['shark2'] = padded_shark_sprites

    # Pad fish sprites to the same size
    fish_sprites = [sprites['fish1'], sprites['fish2']]
    max_fish_height = max(s.shape[0] for s in fish_sprites)
    max_fish_width = max(s.shape[1] for s in fish_sprites)
    padded_fish_sprites = [
        jnp.pad(
            s,
            ((0, max_fish_height - s.shape[0]), (0, max_fish_width - s.shape[1]), (0, 0)),
            mode='constant',
            constant_values=0
        )
        for s in fish_sprites
    ]
    sprites['fish1'], sprites['fish2'] = padded_fish_sprites

    # Create score digits array
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

        # Draw fishing lines
        raster = self._render_line(raster, cfg.P1_START_X + 10, cfg.PLAYER_Y + 10, state.p1.hook_x, state.p1.hook_y)
        raster = self._render_line(raster, cfg.P2_START_X + 2, cfg.PLAYER_Y + 10, state.p2.hook_x, state.p2.hook_y)

        # Draw shark
        shark_frame = jax.lax.cond((state.time // 8) % 2 == 0, lambda: self.SPRITE_SHARK1, lambda: self.SPRITE_SHARK2)
        raster = self._render_at(raster, state.shark_x, cfg.SHARK_Y, shark_frame, flip_h=state.shark_dir < 0)

        # Draw fish
        fish_frame = jax.lax.cond((state.time // 10) % 2 == 0, lambda: self.SPRITE_FISH1, lambda: self.SPRITE_FISH2)

        def draw_one_fish(i, r):
            pos, direction, active = state.fish_positions[i], state.fish_directions[i], state.fish_active[i]
            return jax.lax.cond(active,
                                lambda r_in: self._render_at(r_in, pos[0], pos[1], fish_frame, flip_h=direction < 0),
                                lambda r_in: r_in, r)

        raster = jax.lax.fori_loop(0, cfg.NUM_FISH, draw_one_fish, raster)

        # Draw hooked fish
        def draw_hooked(p_state, r):
            return jax.lax.cond(p_state.hook_state > 0,
                                lambda r_in: self._render_at(r_in, p_state.hook_x, p_state.hook_y, fish_frame),
                                lambda r_in: r_in, r)

        raster = draw_hooked(state.p1, raster)
        raster = draw_hooked(state.p2, raster)

        # Draw scores
        raster = self._render_score(raster, state.p1.score, 50, 20)
        raster = self._render_score(raster, state.p2.score, 100, 20)

        return raster

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
        x, y = jnp.round(x).astype(jnp.int32), jnp.round(y).astype(jnp.int32)
        sprite_to_draw = jnp.where(flip_h, jnp.fliplr(sprite_rgb), sprite_rgb)
        return jax.lax.dynamic_update_slice(raster, sprite_to_draw, (y, x, 0))

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
    try:
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
            raster = jnp.transpose(raster, (1, 0, 2))
            aj.update_pygame(screen, raster, scaling, GameConfig.SCREEN_WIDTH, GameConfig.SCREEN_HEIGHT)
            pygame.display.flip()
            counter += 1
            pygame.time.Clock().tick(60)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        pygame.quit()

# run with: python scripts/play.py --game src/jaxatari/games/jax_fishingderby.py --record my_record_file.npz