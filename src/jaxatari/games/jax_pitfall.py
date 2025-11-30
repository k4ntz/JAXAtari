import os
from functools import partial
import chex
import jax
import jax.numpy as jnp
from jax import lax
from dataclasses import dataclass
from typing import Tuple, NamedTuple, List, Dict, Optional, Any

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils

class PitfallState(NamedTuple):
    # Player position & motion
    player_x: chex.Array   # scalar, e.g. jnp.array(10.0)
    player_y: chex.Array   # scalar
    player_vx: chex.Array  # scalar
    player_vy: chex.Array  # scalar
    on_ground: chex.Array  # bool scalar

    # Resources / game status
    time_left: chex.Array  # int scalar (e.g. 2000)
    lives_left: chex.Array # int scalar (e.g. 3, even if not used yet)
    done: chex.Array       # bool scalar

    down_pressed: chex.Array

class PitfallConstants(NamedTuple):
    screen_width: int = 160     # Atari 2600 horizontal resolution
    screen_height: int = 210   # Atari vertical resolution used in ALE
    ground_y: int = 180         # approximate ground line in pixels
    player_start_x: int = 20    # where Harry starts (left side)
    player_start_y: int = 180   # same as ground_y (standing on ground)

    player_speed: float = 1.5  # pixels per frame horizontally
    jump_velocity: float = -4.0  # initial upward velocity
    gravity: float = 0.3       # downward accel each frame

    initial_time: int = 2000    # Pitfall timer
    max_lives: int = 3          # Pitfall lives

class PitfallObservation(NamedTuple):
    # for now, can just mirror some fields from state
    player_x: chex.Array
    player_y: chex.Array
    time_left: chex.Array
    lives_left: chex.Array
    down_pressed: chex.Array

class PitfallInfo(NamedTuple):
    # extra logging. minimal for now
    time_left: chex.Array
    lives_left: chex.Array

class JaxPitfall(JaxEnvironment[PitfallState, PitfallObservation, PitfallInfo, PitfallConstants]):
    def __init__(self, consts: PitfallConstants | None = None):
        # If no constants are passed, use defaults
        if consts is None:
            consts = PitfallConstants()
        # Call base class constructor
        super().__init__(consts)
        # Store constants on self (the env instance)
        self.consts = consts

        self.renderer = PitfallRenderer(self.consts)

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key = None
    ) -> tuple[PitfallObservation, PitfallState]:
        state = self._init_state()
        obs = self._get_observation(state)
        return obs, state

    def step(
        self,
        state: PitfallState,
        action: int,
    ) -> tuple[PitfallObservation, PitfallState, float, bool, PitfallInfo]:
        consts = self.consts

        # unpack
        x = state.player_x
        y = state.player_y
        vx = state.player_vx
        vy = state.player_vy
        on_ground = state.on_ground
        time_left = state.time_left
        lives_left = state.lives_left

        # --- horizontal movement (0=NOOP, 1=LEFT, 2=RIGHT, 3=JUMP) ---
        speed = jnp.asarray(consts.player_speed, dtype=jnp.float32)

        #for later when I wish to climb down ladders:
        down_action = (
            (action == Action.DOWN) |
            (action == Action.DOWNLEFT) |
            (action == Action.DOWNRIGHT) |
            (action == Action.DOWNFIRE) |
            (action == Action.DOWNLEFTFIRE) |
            (action == Action.DOWNRIGHTFIRE)
        )       
        down_pressed = down_action

        move_left = (
            (action == Action.LEFT) |
            (action == Action.UPLEFT) |
            (action == Action.DOWNLEFT) |
            (action == Action.LEFTFIRE) |
            (action == Action.UPLEFTFIRE) |
            (action == Action.DOWNLEFTFIRE)
        )
        move_right = (
            (action == Action.RIGHT) |
            (action == Action.UPRIGHT) |
            (action == Action.DOWNRIGHT) |
            (action == Action.RIGHTFIRE) |
            (action == Action.UPRIGHTFIRE) |
            (action == Action.DOWNRIGHTFIRE)
        )
        move_jump = (
            (action == Action.UP) |
            (action == Action.UPLEFT) |
            (action == Action.UPRIGHT) |
            (action == Action.UPFIRE) |
            (action == Action.UPLEFTFIRE) |
            (action == Action.UPRIGHTFIRE)
        )

        #if key right and key left then do some op (left is negative x) 
        vx = jnp.where(move_left, -speed, jnp.where(move_right, speed, 0.0))


        # --- jump (only if on_ground and action==3) ---
        jump_mask = jnp.logical_and(on_ground, move_jump)
        vy = jnp.where(jump_mask, jnp.asarray(consts.jump_velocity, dtype=jnp.float32), vy) #else keep prev/current vy
        vy = vy + jnp.asarray(consts.gravity, dtype=jnp.float32) #gravity always applies

        y = y + vy
        x = x + vx

        # --- clamp to ground & reset velocity on landing ---
        ground_y = jnp.asarray(consts.ground_y, dtype=jnp.float32) #convert to float32 for comparison
        on_ground = y >= ground_y #if y >= ground_y, then on_ground is True
        y = jnp.where(on_ground, ground_y, y)

        # if we've hit the ground while moving downward, kill vertical velocity
        vy = jnp.where(on_ground & (vy > 0), 0.0, vy)

        # --- time & done ---
        time_left = time_left - 1
        done = time_left <= 0

        # build new state
        new_state = PitfallState(
            player_x=x,
            player_y=y,
            player_vx=vx,
            player_vy=vy,
            on_ground=on_ground,
            time_left=time_left,
            lives_left=lives_left,
            done=done,
            down_pressed=down_pressed,
        )

        obs = self._get_observation(new_state)
        reward = self._get_reward(state, new_state)
        info = self._get_info(new_state)

        return obs, new_state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: PitfallState) -> PitfallObservation:
        return PitfallObservation(
            player_x=state.player_x,
            player_y=state.player_y,
            time_left=state.time_left,
            lives_left=state.lives_left
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: PitfallState) -> PitfallInfo:
        return PitfallInfo(
            time_left=state.time_left,
            lives_left=state.lives_left
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, prev: PitfallState, new: PitfallState) -> float:
        # for now, no reward
        return 0.0

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: PitfallState) -> bool:
        return state.time_left <= 0

    def action_space(self) -> spaces.Discrete:
        # 0=NOOP, 1=LEFT, 2=RIGHT, 3=JUMP for example
        return spaces.Discrete(4)

    def observation_space(self) -> spaces.Dict:
        # later we can make this more precise
        raise NotImplementedError

    def image_space(self) -> spaces.Box:
        # later, when we hook up the renderer
        raise NotImplementedError

    def render(self, state: PitfallState) -> jnp.ndarray:
        # play.py will JIT this 
        return self.renderer.render(state)

    def obs_to_flat_array(self, obs: PitfallObservation) -> jnp.ndarray:
        # pack obs into 1D array later
        raise NotImplementedError

    def _init_state(self) -> PitfallState:
        consts = self.consts
        state = PitfallState(
            player_x=jnp.array(consts.player_start_x, dtype=jnp.float32),
            player_y=jnp.array(consts.player_start_y, dtype=jnp.float32),
            player_vx=jnp.array(0.0, dtype=jnp.float32),
            player_vy=jnp.array(0.0, dtype=jnp.float32),
            on_ground=jnp.array(True, dtype=jnp.bool_),
            time_left=jnp.array(consts.initial_time, dtype=jnp.int32),
            lives_left=jnp.array(consts.max_lives, dtype=jnp.int32),
            done=jnp.array(False, dtype=jnp.bool_),
            down_pressed=jnp.array(False, dtype=jnp.bool_),
        )
        return state
    


##debug renderer
class PitfallRenderer(JAXGameRenderer):
    """Very simple renderer: black background, green ground, white player block."""

    def __init__(self, consts: PitfallConstants | None = None):
        super().__init__()
        self.consts = consts or PitfallConstants()

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: PitfallState) -> jnp.ndarray:
        h = self.consts.screen_height
        w = self.consts.screen_width

        frame = jnp.zeros((h, w, 3), dtype=jnp.uint8)

        # ----- static ground line -----
        ground = self.consts.ground_y
        frame = frame.at[ground:ground + 2, :, 1].set(200)  # green band

        # ----- dynamic player rect -----
        player_w, player_h = 8, 16

        # x position (clamped)
        x = jnp.clip(state.player_x.astype(jnp.int32), 0, w - player_w)

        # treat state.player_y as feet (bottom)
        bottom = jnp.clip(state.player_y.astype(jnp.int32), 0, h - 1)

        # we want the rect fully inside the screen: choose a valid top
        top = jnp.clip(bottom - player_h + 1, 0, h - player_h)

        # build a (player_h, player_w, 3) white rectangle
        color = jnp.array([255, 255, 255], dtype=jnp.uint8)
        rect = jnp.ones((player_h, player_w, 3), dtype=jnp.uint8) * color

        # dynamic insert: indices (top, x, 0) can be JAX values
        frame = lax.dynamic_update_slice(frame, rect, (top, x, 0))
        def add_down_banner(f):
            return f.at[0:5, :, 0].set(255)  # red strip at top

        frame = lax.cond(
            state.down_pressed,
            add_down_banner,   # then branch
            lambda f: f,       # else branch (no change)
            frame,
        )

        return frame
