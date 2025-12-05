from typing import NamedTuple, Tuple
import numpy as np
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.spaces import Discrete, Box
from jaxatari.renderers import JAXGameRenderer
import jax.numpy as jnp
import chex
import jax.lax


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
class TutankhamConstants(NamedTuple):
    WIDTH: int = 160
    HEIGHT: int = 210
    SPEED: int = 4
    PIXEL_COLOR: Tuple[int, int, int] = (255, 255, 255)  # white

    PLAYER_SIZE: Tuple[int, int] = (5, 10)

    # Missile constants
    BULLET_SIZE: Tuple[int, int] = (1, 2)
    BULLET_SPEED: int = 8

# ---------------------------------------------------------------------
# Game State
# ---------------------------------------------------------------------
class TutankhamState(NamedTuple):
    player_x: int
    player_y: int

    bullet_state: chex.Array #(, 4) array with (x, y, bullet_rotation, bullet_active)
    amonition_timer: int # if timer runs out, player can not fire again


# ---------------------------------------------------------------------
# Renderer (No JAX)
# ---------------------------------------------------------------------
class TutankhamRenderer(JAXGameRenderer):
    def __init__(self):
        super().__init__()
        self.consts = TutankhamConstants()

    def render(self, state: TutankhamState) -> np.ndarray:
        frame = np.zeros(
            (self.consts.HEIGHT, self.consts.WIDTH, 3), dtype=np.uint8
        )

        # -------------------------
        # Draw player
        # -------------------------
        x = min(max(state.player_x, 0), self.consts.WIDTH - 1)
        y = min(max(state.player_y, 0), self.consts.HEIGHT - 1)
        frame[y, x] = self.consts.PIXEL_COLOR

        # -------------------------
        # Draw bullets (1×1 pixels)
        # -------------------------
        bx, by, rot, active = state.bullet_state
        if active:
            # Clip
            #if 0 <= bx < self.consts.WIDTH and 0 <= by < self.consts.HEIGHT:
            frame[int(by), int(bx)] = self.consts.PIXEL_COLOR

        return frame


# ---------------------------------------------------------------------
# Environment (No JAX)
# ---------------------------------------------------------------------
class JaxTutankham(JaxEnvironment):
    def __init__(self):
        consts = TutankhamConstants()
        super().__init__(consts)
        self.renderer = TutankhamRenderer()
        self.consts = consts

        self.action_set = [
            Action.NOOP,
            Action.UP,
            Action.DOWN,
            Action.LEFT,
            Action.RIGHT,
            Action.RIGHTFIRE,
            Action.LEFTFIRE,
            Action.UPLEFTFIRE,
            Action.UPRIGHTFIRE
        ]

    # -----------------------------
    # Reset
    # -----------------------------
    def reset(self, key=None):
        start_x = self.consts.WIDTH // 2
        start_y = self.consts.HEIGHT // 2
        bullet_state = np.array([0, 0, 0, False])
        amonition_timer = 300

        state = TutankhamState(player_x=start_x, player_y=start_y, bullet_state=bullet_state, amonition_timer=amonition_timer)
        return state, state

    # Player Step
    def player_step(
            self,
            player_x,
            player_y,
            action
    ):

        if action == Action.LEFT:
            player_x -= self.consts.SPEED
        elif action == Action.RIGHT:
            player_x += self.consts.SPEED
        elif action == Action.UP:
            player_y -= self.consts.SPEED
        elif action == Action.DOWN:
            player_y += self.consts.SPEED

        # Clip bounds
        player_x = max(0, min(player_x, self.consts.WIDTH - 1))
        player_y = max(0, min(player_y, self.consts.HEIGHT - 1))

        return player_x, player_y
    
    
    #Bullet Step
    def bullet_step(self, tutankham_state, player_x, player_y, bullet_speed, action):

        def get_rotation(action):
            if action == Action.RIGHTFIRE: return 1
            if action == Action.LEFTFIRE: return -1
            return 0  # default if firing up/down/etc

        space = (
                (action == Action.LEFTFIRE)
                or (action == Action.RIGHTFIRE)
            )

        bullet = tutankham_state.bullet_state #array with (x, y, bullet_rotation, bullet_active)
        new_bullet = bullet.copy()

        amonition_timer = tutankham_state.amonition_timer

        
        # --- update existing bullets ---
        if bullet[3]:
            bullet_x = bullet[0] + bullet_speed * bullet[2]
            new_bullet[0] = bullet_x

            # Deactivate if out of bounds
            if not (0 <= bullet_x < self.consts.WIDTH):
                new_bullet = [0, 0, 0, False]


        # --- firing logic ---
        bullet_rdy = not bullet[3]


        if space and bullet_rdy and amonition_timer > 0:
            new_bullet = np.array([player_x, player_y, get_rotation(action), True])


        
        return new_bullet, amonition_timer



    # -----------------------------
    # Step logic (pure Python)
    # -----------------------------
    def step(self, state: TutankhamState, action: int):

        player_x, player_y = state.player_x, state.player_y

        player_x, player_y = self.player_step(player_x, player_y, action)

        bullet_states, bullet_rdy =self.bullet_step(state, player_x, player_y, self.consts.BULLET_SPEED, action)

        state = TutankhamState(player_x=player_x, player_y=player_y, bullet_state=bullet_states, amonition_timer=bullet_rdy)

        reward = 0.0
        done = False
        info = None

        return state, state, reward, done, info

    # -----------------------------
    # Rendering
    # -----------------------------
    def render(self, state: TutankhamState) -> np.ndarray:
        return self.renderer.render(state)

    # -----------------------------
    # Action & Observation Space
    # -----------------------------
    def action_space(self):
        return Discrete(len(self.action_set))

    def observation_space(self):
        return Box(
            low=0,
            high=max(self.consts.WIDTH, self.consts.HEIGHT),
            shape=(2,),
            dtype=np.int32,
        )
