import os
from dataclasses import dataclass, field

from jax import config
from jax._src.dtypes import dtype
import jax.lax
import jax.numpy as jnp
import chex
from numpy import array
import pygame
from typing import Dict, Any, Optional, NamedTuple, Tuple
from functools import partial

from jaxatari.rendering import atraJaxis as aj
from jaxatari.renderers import AtraJaxisRenderer
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action

@dataclass(frozen=True)
class GameConfig:
    """ Game configuration parameters"""
    screen_width: int = 160
    screen_height: int = 210
    scaling_factor: int = 3
    bullet_height: int = 5
    bullet_width: int = 5
    bullet_speed: int = 3
    cannon_height: int = 10
    cannon_width: int = 10
    cannon_y: int = 200
    cannon_x: jnp.ndarray  = field(
        default_factory=lambda: jnp.array([20, 80, 140], dtype=jnp.int32)
    )
    max_bullets: int = 20
    max_enemies: int = 20
    fire_cooldown_frames: int = 10 # delay between shots


# Each value of this class is a list.
# e.g. if i have 3 entities, then each of these lists would have a length of 3
class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray


class AtlantisState(NamedTuple):
    score: chex.Array

    # columns = [ x,  y,  dx,   type_id,  active_flag ]
    #   x, y        → position
    #   dx          → horizontal speed (positive or negative)
    #   type_id     → integer index into your enemy_specs dict
    #   active_flag → 1 if on-screen, 0 otherwise
    enemies: chex.Array # shape: (max_enemies, 5)

    # columns = [ x, y, dx, dy]. dx and dy is the velocity
    bullets: chex.Array # shape: (max_bullets, 4)
    bullets_alive: chex.Array # stores all the active bullets as bools
    fire_cooldown: chex.Array # frames left until next shot
    fire_button_prev: chex.Array # was fire button down last frame


class AtlantisObservation(NamedTuple):
    score: jnp.ndarray
    enemy: EntityPosition
    bullet: EntityPosition

class AtlantisInfo(NamedTuple):
    time: jnp.ndarray

class Renderer_AtraJaxis(AtraJaxisRenderer):
    sprites: Dict[str, Any]

    def __init__(self, config: GameConfig | None = None):
        super().__init__()
        self.config = config or GameConfig()
        self.sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/atlantis"
        self.sprites = self._load_sprites()

    def _load_sprites(self) -> dict[str, Any]:
        """Loads all necessary sprites from .npy files."""
        sprites: Dict[str, Any] = {}

        # Helper function to load a single sprite frame
        def _load_sprite_frame(name: str) -> Optional[chex.Array]:
            path = os.path.join(self.sprite_path, f'{name}.npy')
            frame = aj.loadFrame(path)
            if isinstance(frame, jnp.ndarray) and frame.ndim >= 2:
                return frame.astype(jnp.uint8)


        # --- Load Sprites ---
        # Backgrounds + Dynamic elements + UI elements
        sprite_names = [
            # 'background_0', 'background_1', 'background_2',
        ]
        for name in sprite_names:
            loaded_sprite = _load_sprite_frame(name)
            if loaded_sprite is not None:
                 sprites[name] = loaded_sprite

        return sprites

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: AtlantisState):
        cfg   = self.config
        W, H  = cfg.screen_width, cfg.screen_height

        # background
        light_blue = jnp.array([173, 216, 230], dtype=jnp.uint8)
        raster     = jnp.broadcast_to(light_blue, (W, H, 3))

        # cannons (black rectangles)
        cannon_sprite = jnp.zeros((cfg.cannon_width,
                                   cfg.cannon_height, 3),
                                  dtype=jnp.uint8)   # RGB=(0,0,0)

        def draw_cannon(i, img):
            x = cfg.cannon_x[i]
            return jax.lax.dynamic_update_slice(img, cannon_sprite,
                                                (x, cfg.cannon_y, 0))
        raster = jax.lax.fori_loop(0, cfg.cannon_x.shape[0],
                                   draw_cannon, raster)

        # bullets (white squares)
        bullet_sprite = jnp.full((cfg.bullet_width,
                                  cfg.bullet_height, 3),
                                 255, dtype=jnp.uint8)   # RGB=(255,255,255)

        def draw_bullet(i, img):
            active = state.bullets_alive[i]
            bx, by = state.bullets[i, 0], state.bullets[i, 1]
            return jax.lax.cond(
                active,
                lambda im: jax.lax.dynamic_update_slice(im, bullet_sprite,
                                                         (bx, by, 0)),
                lambda im: im,
                img
            )
        raster = jax.lax.fori_loop(0, cfg.max_bullets, draw_bullet, raster)

        return raster

class JaxAtlantis(JaxEnvironment[AtlantisState, AtlantisObservation, AtlantisInfo]):
    def __init__(self, frameskip: int = 1, reward_funcs: list[callable] = None, config: GameConfig | None = None):
        super().__init__()
        # if no config was provided, instantiate the default one
        self.config = config or GameConfig()
        self.frameskip = frameskip
        self.frame_stack_size = 4
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs

    def reset(self, key: jax.random.PRNGKey = jax.random.PRNGKey(42)) -> Tuple[AtlantisObservation, AtlantisState]:
        # --- empty tables ---
        empty_enemies = jnp.zeros(
            (self.config.max_enemies, 5),
            dtype=jnp.int32
        )
        empty_bullets = jnp.zeros(
            (self.config.max_bullets, 4),
            dtype=jnp.int32
        )
        empty_bullets_alive = jnp.zeros(
            (self.config.max_bullets,),
            dtype=jnp.bool_
        )

        # --- initial state ---
        new_state = AtlantisState(
            score         = jnp.array(0, dtype=jnp.int32),
            enemies       = empty_enemies,
            bullets       = empty_bullets,
            bullets_alive = empty_bullets_alive,
            fire_cooldown = jnp.array(0, dtype=jnp.int32),
            fire_button_prev= jnp.array(False, dtype=jnp.bool_)
        )

        obs = self._get_observation(new_state)
        return obs, new_state

    def _interpret_action(self, state, action) -> Tuple[bool, bool, int]:
        """
        Translate action into control signals
        Returns three vars:

        fire_pressed: If any button is currently pressed
        can_shoot: cooldown expired and just pressed a button
        cannon_idx: (0) left, (1) centre, (2) right or -1.
        """
        fire_pressed = (
            (action == Action.LEFTFIRE) |
            (action == Action.FIRE)     |
            (action == Action.RIGHTFIRE)
        )
        # It is important to keep track if the button just got pressed
        # to prevent holding the button down and spamming bullets
        just_pressed  = fire_pressed & (~state.fire_button_prev)
        can_shoot    = (state.fire_cooldown == 0) & just_pressed 

        cannon_idx = jnp.where(
            can_shoot,
            jnp.where(
                action == Action.LEFTFIRE,
                0,
                jnp.where(
                    action == Action.FIRE,
                    1,
                    jnp.where(
                        action == Action.RIGHTFIRE, 
                        2, 
                        -1
                    )
                )
            ),
            -1
        )
        return fire_pressed, cannon_idx

    # ..................................................................

    def _spawn_bullet(self, state, cannon_idx):
        """Insert newly spawned bullet in first free slot"""
        cfg = self.config

        def _do_spawn(s):            
            # To identify which slots are free
            # bullets_alive is a boolean array. If an entry is true, then it holds an active bullet
            # ~ inverts the boolean array, such that a slot is free, when bullets_alive[i] == False
            free_slots     = ~s.bullets_alive
            slot_available = jnp.any(free_slots) # at least one free?
            slot_idx       = jnp.argmax(free_slots) # first free slot

            # horizontal component dx:
            # - if cannon_idx == 0 (left), shoot rightwards -> +bullet_speed
            # - if cannond_idx == 2 (right), shoot leftwards -> -bullet_speed
            # else go straigt -> 0
            dx = jnp.where(
                cannon_idx == 0, # true for left cannon
                cfg.bullet_speed, # e.g. +3 pixels/frame
                jnp.where(
                    cannon_idx == 2, # true for right cannon
                    -cfg.bullet_speed, # e.g. -3 px
                    0 #zero horizontal velocity
                )
            )

            # vertcal componend dy:
            # - all bullets move up at the same speed. Because origin is in top left, its negative
            dy = -cfg.bullet_speed

            new_bullet = jnp.array([
                cfg.cannon_x[cannon_idx], 
                cfg.cannon_y,
                dx, dy # velocity
            ], dtype=jnp.int32)

            # write into state
            def _write(s2):
                b2 = s2.bullets.at[slot_idx].set(new_bullet)
                a2 = s2.bullets_alive.at[slot_idx].set(True)
                return s2._replace(bullets=b2, bullets_alive=a2)

            # Conditionally write if a free slot exists
            return jax.lax.cond(slot_available, _write, lambda x: x, s)

        # Only attempt the spawn when a cannon actually fired this frame
        return jax.lax.cond(cannon_idx >= 0, _do_spawn, lambda x: x, state)


    def _update_cooldown(self, state, cannon_idx):
        """Reset after a shot or decrement the fire cooldown timer."""
        cfg = self.config
        new_cd = jnp.where(
            cannon_idx >= 0, # -1 means no cannon fired
            jnp.array(cfg.fire_cooldown_frames, dtype=jnp.int32),
            jnp.maximum(state.fire_cooldown - 1, 0)
        )
        return state._replace(fire_cooldown=new_cd)


    def _move_bullets(self, state):
        """Move bullets by their velocity and deactivate offscreen bullets"""
        cfg = self.config
        
        # compute new x and y positions by adding the velocity dx and dy
        # state.bullets has shape (max_bullets, 4): (x,y,dx,dy)
        # [:, :2] takes all rows, but only columns 0 and 1 which are x and y
        # 2:4 then is dx and dy
        positions = state.bullets[:, :2] + state.bullets[:, 2:4]
        # Write updated position back into bullets array
        moved = state.bullets.at[:, :2].set(positions)

        # check if bullets are still onscreen
        in_bounds = (
            (positions[:, 0] >= 0) & (positions[:, 0] < cfg.screen_width)  &
            (positions[:, 1] >= 0) & (positions[:, 1] < cfg.screen_height)
        )

        # abullet only remains alive if it was already alive and still on-screen
        alive = state.bullets_alive & in_bounds
        return state._replace(bullets=moved, bullets_alive=alive)


    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: AtlantisState, action: chex.Array) -> Tuple[AtlantisObservation, AtlantisState, float, bool, AtlantisInfo]:
        # input handling
        fire_pressed,  cannon_idx = self._interpret_action(state, action)

        state = self._spawn_bullet(state, cannon_idx)

        # update cooldown and remember current button state
        state = self._update_cooldown(state, cannon_idx)._replace(
            fire_button_prev=fire_pressed
        )

        state = self._move_bullets(state)

        observation = self._get_observation(state)
        info = AtlantisInfo(time=jnp.array(0, dtype=jnp.int32))
        reward = 0.0      # Placeholder: no scoring yet
        done = False    # Never terminates for now

        return observation, state, reward, done, info


    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: "AtlantisState") -> "AtlantisObservation":
        # just placeholders
        enemies_pos = EntityPosition(
            0,
            0,
            0,
            0,
        )

        bullets_pos = EntityPosition(
            0,
            0,
            0,
            0,
        )

        return AtlantisObservation(state.score, enemies_pos, bullets_pos)

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: AtlantisState) -> AtlantisInfo:
        """
        Placeholder info: returns zero time and empty reward array.
        """
        return AtlantisInfo(
            time=jnp.array(0, dtype=jnp.int32),
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(
        self,
        previous_state: AtlantisState,
        state: AtlantisState
    ) -> float:
        """
        Placeholder reward: always zero.
        """
        return 0.0

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: AtlantisState) -> bool:
        """
        Placeholder done: never terminates.
        """
        return False

    @partial(jax.jit, static_argnums=(0,))
    def get_action_space(self) -> jnp.ndarray:
        """
        Placeholder done: never terminates.
        """
        return jnp.array([], dtype=jnp.int32)



# Keyboard inputs
def get_human_action() -> chex.Array:
    keys = pygame.key.get_pressed()
    # up = keys[pygame.K_w] or keys[pygame.K_UP]
    # down = keys[pygame.K_s] or keys[pygame.K_DOWN]
    left = keys[pygame.K_a] or keys[pygame.K_LEFT]
    right = keys[pygame.K_d] or keys[pygame.K_RIGHT]
    fire = keys[pygame.K_SPACE]

    if right and fire:
        return jnp.array(Action.RIGHTFIRE)
    if left and fire:
        return jnp.array(Action.LEFTFIRE)
    if fire:
        return jnp.array(Action.FIRE)

    return jnp.array(Action.NOOP)


def main():
    config = GameConfig()
    pygame.init()
    screen = pygame.display.set_mode((
        config.screen_width * config.scaling_factor,
        config.screen_height * config.scaling_factor
    ))
    pygame.display.set_caption("Atlantis")
    clock = pygame.time.Clock()

    game = JaxAtlantis(config=config)

    renderer = Renderer_AtraJaxis(config=config)
    jitted_step = jax.jit(game.step)
    jitted_reset = jax.jit(game.reset)

    #(curr_state, _) = jitted_reset()
    (_, curr_state) = jitted_reset()

    # Game loop
    running = True
    frame_by_frame = False
    frameskip = game.frameskip
    counter = 1


    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    frame_by_frame = not frame_by_frame
            elif event.type == pygame.KEYDOWN or (
                    event.type == pygame.KEYUP and event.key == pygame.K_n
            ):
                if event.key == pygame.K_n and frame_by_frame:
                    if counter % frameskip == 0:
                        action = get_human_action()
                        (obs, curr_state, _, _, _) = jitted_step(curr_state, action)

        if not frame_by_frame:
            if counter % frameskip == 0:
                action = get_human_action()
                (obs, curr_state, _, _, _) = jitted_step(curr_state, action)

        # Render and display
        raster = renderer.render(curr_state)

        aj.update_pygame(
            screen,
            raster,
            config.scaling_factor,
            config.screen_width,
            config.screen_height
        )

        counter += 1
        # FPS
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
