from functools import partial
from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
import chex
import pygame

import jaxatari.spaces as spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import JAXGameRenderer
import jaxatari.rendering.jax_rendering_utils as jr


class KlaxConstants(NamedTuple):
    SCREEN_WIDTH: int = 160
    SCREEN_HEIGHT: int = 210
    BACKGROUND_COLOR: Tuple[int, int, int] = (0, 0, 0)
    PLAYER_COLOR: Tuple[int, int, int] = (255, 255, 255)
    TILE_COLORS: Tuple[Tuple[int, int, int], ...] = (
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
    )
    PLAYER_SIZE: Tuple[int, int] = (16, 8)
    TILE_SIZE: Tuple[int, int] = (8, 8)
    BOARD_ROWS: int = 5
    BOARD_COLS: int = 8


class EntityPosition(NamedTuple):
    x: chex.Array
    y: chex.Array
    width: chex.Array
    height: chex.Array


class KlaxObservation(NamedTuple):
    player: EntityPosition
    falling_tile: EntityPosition
    board: chex.Array
    score: chex.Array
    lives: chex.Array


class KlaxInfo(NamedTuple):
    time: chex.Array
    all_rewards: chex.Array


class KlaxState(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    tile_x: chex.Array
    tile_y: chex.Array
    board: chex.Array
    score: chex.Array
    lives: chex.Array
    step_counter: chex.Array
    rng_key: chex.PRNGKey


class JaxKlax(JaxEnvironment[KlaxState, KlaxObservation, KlaxInfo, KlaxConstants]):
    def __init__(self, consts: KlaxConstants | None = None, reward_funcs: list[callable] | None = None):
        consts = consts or KlaxConstants()
        super().__init__(consts)
        self.renderer = KlaxRenderer(self.consts)
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.action_set = [
            Action.NOOP,
            Action.LEFT,
            Action.RIGHT,
            Action.DOWN,
            Action.FIRE,
        ]

    def get_human_action(self) -> chex.Array:
        keys = pygame.key.get_pressed()
        if keys[pygame.K_a]:
            return jnp.array(Action.LEFT)
        if keys[pygame.K_d]:
            return jnp.array(Action.RIGHT)
        if keys[pygame.K_s]:
            return jnp.array(Action.DOWN)
        if keys[pygame.K_SPACE]:
            return jnp.array(Action.FIRE)
        return jnp.array(Action.NOOP)

    def reset(self, key: jax.random.PRNGKey | None = None) -> tuple[KlaxObservation, KlaxState]:
        if key is None:
            key = jax.random.PRNGKey(0)
        board = jnp.zeros((self.consts.BOARD_ROWS, self.consts.BOARD_COLS), dtype=jnp.int32)
        state = KlaxState(
            player_x=jnp.array(self.consts.SCREEN_WIDTH // 2, dtype=jnp.int32),
            player_y=jnp.array(self.consts.SCREEN_HEIGHT - self.consts.PLAYER_SIZE[1], dtype=jnp.int32),
            tile_x=jnp.array(self.consts.SCREEN_WIDTH // 2, dtype=jnp.int32),
            tile_y=jnp.array(0, dtype=jnp.int32),
            board=board,
            score=jnp.array(0, dtype=jnp.int32),
            lives=jnp.array(3, dtype=jnp.int32),
            step_counter=jnp.array(0, dtype=jnp.int32),
            rng_key=key,
        )
        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: KlaxState, action: chex.Array) -> tuple[KlaxObservation, KlaxState, float, bool, KlaxInfo]:
        new_state = state._replace(step_counter=state.step_counter + 1)
        reward = self._get_reward(state, new_state)
        if self.reward_funcs is not None:
            all_rewards = jnp.array([rf(state, new_state) for rf in self.reward_funcs])
        else:
            all_rewards = jnp.array([reward])
        done = self._get_done(new_state)
        obs = self._get_observation(new_state)
        info = self._get_info(new_state, all_rewards)
        return obs, new_state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: KlaxState) -> KlaxObservation:
        player = EntityPosition(
            x=state.player_x,
            y=state.player_y,
            width=jnp.array(self.consts.PLAYER_SIZE[0], dtype=jnp.int32),
            height=jnp.array(self.consts.PLAYER_SIZE[1], dtype=jnp.int32),
        )
        tile = EntityPosition(
            x=state.tile_x,
            y=state.tile_y,
            width=jnp.array(self.consts.TILE_SIZE[0], dtype=jnp.int32),
            height=jnp.array(self.consts.TILE_SIZE[1], dtype=jnp.int32),
        )
        return KlaxObservation(
            player=player,
            falling_tile=tile,
            board=state.board,
            score=state.score,
            lives=state.lives,
        )

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: KlaxObservation) -> jnp.ndarray:
        return jnp.concatenate(
            [
                obs.player.x.flatten(),
                obs.player.y.flatten(),
                obs.player.width.flatten(),
                obs.player.height.flatten(),
                obs.falling_tile.x.flatten(),
                obs.falling_tile.y.flatten(),
                obs.falling_tile.width.flatten(),
                obs.falling_tile.height.flatten(),
                obs.board.flatten(),
                obs.score.flatten(),
                obs.lives.flatten(),
            ]
        )

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))

    def observation_space(self) -> spaces.Dict:
        max_color = max(len(self.consts.TILE_COLORS), 1)
        return spaces.Dict(
            {
                "player": spaces.Dict(
                    {
                        "x": spaces.Box(low=0, high=self.consts.SCREEN_WIDTH, shape=(), dtype=jnp.int32),
                        "y": spaces.Box(low=0, high=self.consts.SCREEN_HEIGHT, shape=(), dtype=jnp.int32),
                        "width": spaces.Box(low=0, high=self.consts.SCREEN_WIDTH, shape=(), dtype=jnp.int32),
                        "height": spaces.Box(low=0, high=self.consts.SCREEN_HEIGHT, shape=(), dtype=jnp.int32),
                    }
                ),
                "falling_tile": spaces.Dict(
                    {
                        "x": spaces.Box(low=0, high=self.consts.SCREEN_WIDTH, shape=(), dtype=jnp.int32),
                        "y": spaces.Box(low=0, high=self.consts.SCREEN_HEIGHT, shape=(), dtype=jnp.int32),
                        "width": spaces.Box(low=0, high=self.consts.SCREEN_WIDTH, shape=(), dtype=jnp.int32),
                        "height": spaces.Box(low=0, high=self.consts.SCREEN_HEIGHT, shape=(), dtype=jnp.int32),
                    }
                ),
                "board": spaces.Box(
                    low=0,
                    high=max_color,
                    shape=(self.consts.BOARD_ROWS, self.consts.BOARD_COLS),
                    dtype=jnp.int32,
                ),
                "score": spaces.Box(low=0, high=1_000_000, shape=(), dtype=jnp.int32),
                "lives": spaces.Box(low=0, high=99, shape=(), dtype=jnp.int32),
            }
        )

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.consts.SCREEN_HEIGHT, self.consts.SCREEN_WIDTH, 3),
            dtype=jnp.uint8,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: KlaxState, all_rewards: chex.Array = None) -> KlaxInfo:
        if all_rewards is None:
            all_rewards = jnp.zeros(1)
        return KlaxInfo(time=state.step_counter, all_rewards=all_rewards)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: KlaxState, state: KlaxState) -> float:
        return float(state.score - previous_state.score)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: KlaxState) -> bool:
        return False

    def render(self, state: KlaxState) -> jnp.ndarray:
        return self.renderer.render(state)


class KlaxRenderer(JAXGameRenderer):
    def __init__(self, consts: KlaxConstants | None = None):
        super().__init__()
        self.consts = consts or KlaxConstants()
        self.SPRITE_BG, self.SPRITE_PLAYER, self.SPRITE_TILE = self.load_sprites()

    def load_sprites(self):
        bg = jr.create_initial_frame(width=self.consts.SCREEN_WIDTH, height=self.consts.SCREEN_HEIGHT)
        player = jr.create_initial_frame(
            width=self.consts.PLAYER_SIZE[0], height=self.consts.PLAYER_SIZE[1]
        )
        tile = jr.create_initial_frame(
            width=self.consts.TILE_SIZE[0], height=self.consts.TILE_SIZE[1]
        )
        return (
            jnp.expand_dims(bg, axis=0),
            jnp.expand_dims(player, axis=0),
            jnp.expand_dims(tile, axis=0),
        )

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: KlaxState) -> jnp.ndarray:
        raster = jr.create_initial_frame(width=self.consts.SCREEN_WIDTH, height=self.consts.SCREEN_HEIGHT)
        # Placeholder: add rendering logic for player, tiles, and board here
        return raster
