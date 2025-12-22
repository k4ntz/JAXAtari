from jax._src.pjit import JitWrapped
import os
from functools import partial
from typing import NamedTuple, Tuple
import jax
import jax.lax
import jax.numpy as jnp
import chex

import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action


class LostLuggageConstants(NamedTuple):
    WIDTH: int = 160
    HEIGHT: int = 210

    PLAYER_SPEED: int = 2
    PLAYER_SIZE: tuple = (16, 16)

    SUITCASE_SIZE: tuple = (8, 8)
    SUITCASE_FALL_SPEED: int = 1
    SUITCASE_FAST_SPEED: int = 3

    MAX_SUITCASES_PER_ROUND: int = 26
    INITIAL_LIVES: int = 3

    CEILING_Y: int = int(0.4 * 210)   

    ROUND_DELAY_FRAMES: int = 180
    HIT_PAUSE_FRAMES: int = 90

    ASSET_CONFIG: tuple = (
        {'name': 'background', 'type': 'background', 'file': 'background.npy'},
        {'name': 'player', 'type': 'single', 'file': 'player_idle.npy'},
        {'name': 'suitcase', 'type': 'single', 'file': 'suitcase.npy'},
        {'name': 'digits', 'type': 'digits', 'pattern': 'digits_{}.npy'},
    )


class LostLuggageState(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array

    suitcase_x: chex.Array
    suitcase_y: chex.Array
    suitcase_vx: chex.Array
    suitcase_vy: chex.Array
    suitcase_active: chex.Array

    suitcases_caught: chex.Array
    lives: chex.Array
    score: chex.Array

    pause_timer: chex.Array
    round_timer: chex.Array

    key: chex.PRNGKey
    previous_score: chex.Array


class JaxLostLuggage(JaxEnvironment[
    LostLuggageState, None, None, LostLuggageConstants
]):
    def __init__(self, consts=None):
        consts = consts or LostLuggageConstants()
        super().__init__(consts)
        self.renderer = LostLuggageRenderer(consts)

        self.action_set = [
            Action.NOOP,
            Action.UP,
            Action.DOWN,
            Action.LEFT,
            Action.RIGHT,
        ]

    def _player_step(self, state, action):
        dx = jnp.where(action == Action.LEFT, -self.consts.PLAYER_SPEED,
            jnp.where(action == Action.RIGHT, self.consts.PLAYER_SPEED, 0))
        dy = jnp.where(action == Action.UP, -self.consts.PLAYER_SPEED,
            jnp.where(action == Action.DOWN, self.consts.PLAYER_SPEED, 0))

        new_x = jnp.clip(
            state.player_x + dx, 0, self.consts.WIDTH - self.consts.PLAYER_SIZE[0]
        )
        new_y = jnp.clip(
            state.player_y + dy,
            self.consts.CEILING_Y,
            self.consts.HEIGHT - self.consts.PLAYER_SIZE[1],
        )

        return state._replace(player_x=new_x, player_y=new_y)

    def _suitcase_step(self, state):
        active = state.suitcase_active

        x = state.suitcase_x + state.suitcase_vx * active
        y = state.suitcase_y + state.suitcase_vy * active

        hit_floor = y > self.consts.HEIGHT - self.consts.SUITCASE_SIZE[1]

        hit_player = jnp.logical_and(
            jnp.abs(x - state.player_x) < self.consts.PLAYER_SIZE[0],
            jnp.abs(y - state.player_y) < self.consts.PLAYER_SIZE[1],
        )

        score = jax.lax.cond(
            hit_player,
            lambda s: s + 3,
            lambda s: s,
            state.score,
        )

        suitcases_caught = jax.lax.cond(
            hit_player,
            lambda c: c + 1,
            lambda c: c,
            state.suitcases_caught,
        )

        lives = jax.lax.cond(
            hit_floor,
            lambda l: l - 1,
            lambda l: l,
            state.lives,
        )

        pause = jnp.where(hit_floor, self.consts.HIT_PAUSE_FRAMES, state.pause_timer)
        active = jnp.where(jnp.logical_or(hit_player, hit_floor), 0, active)

        return state._replace(
            suitcase_x=x,
            suitcase_y=y,
            suitcase_active=active,
            score=score,
            suitcases_caught=suitcases_caught,
            lives=lives,
            pause_timer=pause,
        )

    def _maybe_spawn_suitcase(self, state):
        can_spawn = jnp.logical_and(
            state.suitcase_active == 0,
            state.suitcases_caught < self.consts.MAX_SUITCASES_PER_ROUND,
        )

        key, sub = jax.random.split(state.key)
        sideways = jax.random.bernoulli(sub, 0.2)

        vx = jnp.where(sideways, jax.random.choice(sub, jnp.array([-1, 1])) * 2, 0)
        vy = jnp.where(sideways, self.consts.SUITCASE_FAST_SPEED,
                    self.consts.SUITCASE_FALL_SPEED)

        return jax.lax.cond(
            can_spawn,
            lambda _: state._replace(
                suitcase_x=jnp.array(80),
                suitcase_y=jnp.array(0),
                suitcase_vx=vx,
                suitcase_vy=vy,
                suitcase_active=1,
                key=key,
            ),
            lambda _: state._replace(key=key),
            operand=None,
        )

    def reset(self, key: chex.PRNGKey = jax.random.PRNGKey(0)):
        state_key, _ = jax.random.split(key)

        state = LostLuggageState(
            player_x=jnp.array(72, dtype=jnp.int32),
            player_y=jnp.array(150, dtype=jnp.int32),

            suitcase_x=jnp.array(0, dtype=jnp.int32),
            suitcase_y=jnp.array(0, dtype=jnp.int32),
            suitcase_vx=jnp.array(0, dtype=jnp.int32),
            suitcase_vy=jnp.array(0, dtype=jnp.int32),
            suitcase_active=jnp.array(0, dtype=jnp.int32),

            suitcases_caught=jnp.array(0, dtype=jnp.int32),
            lives=jnp.array(self.consts.INITIAL_LIVES, dtype=jnp.int32),
            score=jnp.array(0, dtype=jnp.int32),

            pause_timer=jnp.array(0, dtype=jnp.int32),
            round_timer=jnp.array(0, dtype=jnp.int32),

            key=state_key,
            previous_score=jnp.array(0, dtype=jnp.int32),
        )

        return None, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        key, step_key = jax.random.split(state.key)
        state = state._replace(key=step_key)

        state = jax.lax.cond(
            state.pause_timer > 0,
            lambda s: s._replace(pause_timer=s.pause_timer - 1),
            lambda s: self._player_step(s, action),
            state,
        )

        state = self._suitcase_step(state)
        state = self._maybe_spawn_suitcase(state)

        reward = state.score - state.previous_score
        state = state._replace(previous_score=state.score)

        done = state.lives <= 0

        return None, state._replace(key=key), reward, done, {}

    def action_space(self):
        return spaces.Discrete(len(self.action_set))

    def observation_space(self):
        return spaces.Box(low=0, high=1, shape=())

    def image_space(self):
        return spaces.Box(
            low=0,
            high=255,
            shape=(210, 160, 3),
            dtype=jnp.uint8,
        )

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        return self.renderer.render(state)


class LostLuggageRenderer(JAXGameRenderer):
    def __init__(self, consts):
        super().__init__(consts)
        self.consts = consts
        self.config = render_utils.RendererConfig(
            game_dimensions=(210, 160),
            channels=3,
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)

        sprite_path = f"{os.path.dirname(__file__)}/sprites/lostluggage"
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            _
        ) = self.jr.load_and_setup_assets(consts.ASSET_CONFIG, sprite_path)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        raster = self.jr.create_object_raster(self.BACKGROUND)

        raster = self.jr.render_at(
            raster, state.player_x, state.player_y, self.SHAPE_MASKS["player"]
        )

        raster = jax.lax.cond(
            state.suitcase_active == 1,
            lambda r: self.jr.render_at(
                r, state.suitcase_x, state.suitcase_y, self.SHAPE_MASKS["suitcase"]
            ),
            lambda r: r,
            raster,
        )

        digits = self.jr.int_to_digits(state.score, max_digits=3)
        raster = self.jr.render_label(
            raster, 130, 190, digits, self.SHAPE_MASKS["digits"], spacing=1
        )

        return self.jr.render_from_palette(raster, self.PALETTE)