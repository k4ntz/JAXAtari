from __future__ import annotations

from functools import partial
from typing import NamedTuple, Tuple

import chex
import jax
import jax.numpy as jnp

import jaxatari.spaces as spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils


class JamesBondConstants(NamedTuple):
    WIDTH: int = 160
    HEIGHT: int = 210

    ROAD_LEFT: int = 40
    ROAD_RIGHT: int = 120
    ROAD_EDGE_LINE_W: int = 2

    PLAYER_Y: int = 170
    PLAYER_SIZE: Tuple[int, int] = (8, 12)
    PLAYER_STEP_X: int = 4

    MAX_ENEMIES: int = 10
    ENEMY_SIZE: Tuple[int, int] = (8, 12)
    ENEMY_SPEED_Y: int = 2

    NUM_LANES: int = 5
    SPAWN_PROB: float = 0.12
    SPAWN_Y: int = 0

    SCROLL_SPEED: int = 2
    DASH_W: int = 2
    DASH_H: int = 8
    DASH_SPACING: int = 22

    SURVIVE_REWARD: float = 0.01
    PASS_REWARD: float = 0.10

    GRASS_COLOR: Tuple[int, int, int] = (34, 139, 34)
    ROAD_COLOR: Tuple[int, int, int] = (80, 80, 80)
    LINE_COLOR: Tuple[int, int, int] = (236, 236, 236)
    PLAYER_COLOR: Tuple[int, int, int] = (236, 236, 0)
    ENEMY_COLOR: Tuple[int, int, int] = (200, 0, 0)


class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray


class EntityPositionBatch(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray


class JamesBondObservation(NamedTuple):
    player: EntityPosition
    enemies: EntityPositionBatch


class JamesBondInfo(NamedTuple):
    time: jnp.ndarray
    passed: jnp.ndarray
    collisions: jnp.ndarray


class JamesBondState(NamedTuple):
    player_x: chex.Array
    enemy_x: chex.Array
    enemy_y: chex.Array
    enemy_active: chex.Array
    scroll_offset: chex.Array
    step_counter: chex.Array
    passed_counter: chex.Array
    collided: chex.Array
    key: chex.PRNGKey


def _rects_overlap(px: jnp.ndarray, py: jnp.ndarray, pw: int, ph: int,
                   ex: jnp.ndarray, ey: jnp.ndarray, ew: int, eh: int) -> jnp.ndarray:
    overlap_x = (px < (ex + ew)) & ((px + pw) > ex)
    overlap_y = (py < (ey + eh)) & ((py + ph) > ey)
    return overlap_x & overlap_y


class JaxJamesbond(JaxEnvironment[JamesBondState, JamesBondObservation, JamesBondInfo, JamesBondConstants]):
    def __init__(self, consts: JamesBondConstants | None = None):
        consts = consts or JamesBondConstants()
        super().__init__(consts)
        self.renderer = JamesBondRenderer(self.consts)
        self.action_set = [
            Action.NOOP,
            Action.FIRE,
            Action.RIGHT,
            Action.LEFT,
            Action.RIGHTFIRE,
            Action.LEFTFIRE,
        ]

    def _player_step(self, state: JamesBondState, action: chex.Array) -> JamesBondState:
        left = jnp.logical_or(action == Action.LEFT, action == Action.LEFTFIRE)
        right = jnp.logical_or(action == Action.RIGHT, action == Action.RIGHTFIRE)

        dx = jnp.where(left, -self.consts.PLAYER_STEP_X, 0)
        dx = jnp.where(right, self.consts.PLAYER_STEP_X, dx)

        min_x = jnp.array(self.consts.ROAD_LEFT, dtype=jnp.int32)
        max_x = jnp.array(self.consts.ROAD_RIGHT - self.consts.PLAYER_SIZE[0], dtype=jnp.int32)
        new_player_x = jnp.clip(state.player_x + dx.astype(jnp.int32), min_x, max_x).astype(jnp.int32)

        return state._replace(player_x=new_player_x)

    def _enemy_step_and_spawn(self, state: JamesBondState) -> Tuple[JamesBondState, jnp.ndarray]:
        active_i32 = state.enemy_active.astype(jnp.int32)
        new_enemy_y = (state.enemy_y + active_i32 * jnp.array(self.consts.ENEMY_SPEED_Y, dtype=jnp.int32)).astype(jnp.int32)

        passed = state.enemy_active & (new_enemy_y > jnp.array(self.consts.HEIGHT, dtype=jnp.int32))
        passed_count = passed.astype(jnp.int32).sum()

        enemy_active = state.enemy_active & (~passed)
        enemy_y = jnp.where(enemy_active, new_enemy_y, jnp.array(0, dtype=jnp.int32)).astype(jnp.int32)
        enemy_x = jnp.where(enemy_active, state.enemy_x, jnp.array(0, dtype=jnp.int32)).astype(jnp.int32)

        spawn_key, lane_key = jax.random.split(state.key)
        roll = jax.random.uniform(spawn_key, ())
        should_spawn = roll < jnp.array(self.consts.SPAWN_PROB)

        free_mask = ~enemy_active
        any_free = jnp.any(free_mask)
        spawn_idx = jnp.argmax(free_mask.astype(jnp.int32))

        road_w = self.consts.ROAD_RIGHT - self.consts.ROAD_LEFT
        lane_w = jnp.array(road_w // self.consts.NUM_LANES, dtype=jnp.int32)
        lane_margin = jnp.array((road_w - lane_w * self.consts.NUM_LANES) // 2, dtype=jnp.int32)
        lane_centers = (
            jnp.arange(self.consts.NUM_LANES, dtype=jnp.int32) * lane_w
            + jnp.array(self.consts.ROAD_LEFT, dtype=jnp.int32)
            + lane_margin
            + (lane_w // 2)
        )
        lane_idx = jax.random.randint(lane_key, (), 0, self.consts.NUM_LANES, dtype=jnp.int32)
        spawn_x = (lane_centers[lane_idx] - (self.consts.ENEMY_SIZE[0] // 2)).astype(jnp.int32)
        spawn_y = jnp.array(self.consts.SPAWN_Y, dtype=jnp.int32)

        def do_spawn(carry):
            ex, ey, ea = carry
            ex = ex.at[spawn_idx].set(spawn_x)
            ey = ey.at[spawn_idx].set(spawn_y)
            ea = ea.at[spawn_idx].set(True)
            return ex, ey, ea

        enemy_x, enemy_y, enemy_active = jax.lax.cond(
            should_spawn & any_free,
            do_spawn,
            lambda carry: carry,
            operand=(enemy_x, enemy_y, enemy_active),
        )

        new_scroll = (state.scroll_offset + jnp.array(self.consts.SCROLL_SPEED, dtype=jnp.int32)) % jnp.array(
            self.consts.DASH_SPACING, dtype=jnp.int32
        )

        new_state = state._replace(
            enemy_x=enemy_x.astype(jnp.int32),
            enemy_y=enemy_y.astype(jnp.int32),
            enemy_active=enemy_active,
            scroll_offset=new_scroll.astype(jnp.int32),
        )
        return new_state, passed_count

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, state: JamesBondState, action: chex.Array
    ) -> Tuple[JamesBondObservation, JamesBondState, float, bool, JamesBondInfo]:
        new_state_key, step_key = jax.random.split(state.key)

        state = state._replace(key=step_key)

        prev_state = state

        state = self._player_step(state, action)
        state, passed_count = self._enemy_step_and_spawn(state)

        px = state.player_x
        py = jnp.array(self.consts.PLAYER_Y, dtype=jnp.int32)
        collisions = _rects_overlap(
            px, py,
            self.consts.PLAYER_SIZE[0], self.consts.PLAYER_SIZE[1],
            state.enemy_x, state.enemy_y,
            self.consts.ENEMY_SIZE[0], self.consts.ENEMY_SIZE[1],
        )
        collided = jnp.any(state.enemy_active & collisions)

        state = state._replace(
            step_counter=(state.step_counter + jnp.array(1, dtype=jnp.int32)).astype(jnp.int32),
            passed_counter=(state.passed_counter + passed_count).astype(jnp.int32),
            collided=collided,
            key=new_state_key,
        )

        done = self._get_done(state)
        env_reward = self._get_reward(prev_state, state)
        info = self._get_info(state)
        obs = self._get_observation(state)
        return obs, state, env_reward, done, info

    def reset(self, key: chex.PRNGKey = jax.random.PRNGKey(42)) -> Tuple[JamesBondObservation, JamesBondState]:
        state_key, _ = jax.random.split(key)

        start_x = jnp.array((self.consts.ROAD_LEFT + self.consts.ROAD_RIGHT) // 2, dtype=jnp.int32)
        start_x = (start_x - (self.consts.PLAYER_SIZE[0] // 2)).astype(jnp.int32)

        n = self.consts.MAX_ENEMIES
        state = JamesBondState(
            player_x=start_x,
            enemy_x=jnp.zeros((n,), dtype=jnp.int32),
            enemy_y=jnp.zeros((n,), dtype=jnp.int32),
            enemy_active=jnp.zeros((n,), dtype=bool),
            scroll_offset=jnp.array(0, dtype=jnp.int32),
            step_counter=jnp.array(0, dtype=jnp.int32),
            passed_counter=jnp.array(0, dtype=jnp.int32),
            collided=jnp.array(False),
            key=state_key,
        )
        return self._get_observation(state), state

    def render(self, state: JamesBondState) -> jnp.ndarray:
        return self.renderer.render(state)

    def _get_observation(self, state: JamesBondState) -> JamesBondObservation:
        player = EntityPosition(
            x=state.player_x.astype(jnp.int32),
            y=jnp.array(self.consts.PLAYER_Y, dtype=jnp.int32),
            width=jnp.array(self.consts.PLAYER_SIZE[0], dtype=jnp.int32),
            height=jnp.array(self.consts.PLAYER_SIZE[1], dtype=jnp.int32),
        )
        n = self.consts.MAX_ENEMIES
        enemies = EntityPositionBatch(
            x=state.enemy_x.astype(jnp.int32).reshape((n,)),
            y=state.enemy_y.astype(jnp.int32).reshape((n,)),
            width=jnp.full((n,), self.consts.ENEMY_SIZE[0], dtype=jnp.int32),
            height=jnp.full((n,), self.consts.ENEMY_SIZE[1], dtype=jnp.int32),
        )
        return JamesBondObservation(player=player, enemies=enemies)

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: JamesBondObservation) -> jnp.ndarray:
        return jnp.concatenate(
            [
                obs.player.x.flatten(),
                obs.player.y.flatten(),
                obs.player.width.flatten(),
                obs.player.height.flatten(),
                obs.enemies.x.flatten(),
                obs.enemies.y.flatten(),
                obs.enemies.width.flatten(),
                obs.enemies.height.flatten(),
            ],
            axis=0,
        )

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(6)

    def observation_space(self) -> spaces.Space:
        n = self.consts.MAX_ENEMIES
        return spaces.Dict(
            {
                "player": spaces.Dict(
                    {
                        "x": spaces.Box(low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32),
                        "y": spaces.Box(low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.int32),
                        "width": spaces.Box(low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32),
                        "height": spaces.Box(low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.int32),
                    }
                ),
                "enemies": spaces.Dict(
                    {
                        "x": spaces.Box(low=0, high=self.consts.WIDTH, shape=(n,), dtype=jnp.int32),
                        "y": spaces.Box(low=0, high=self.consts.HEIGHT, shape=(n,), dtype=jnp.int32),
                        "width": spaces.Box(low=0, high=self.consts.WIDTH, shape=(n,), dtype=jnp.int32),
                        "height": spaces.Box(low=0, high=self.consts.HEIGHT, shape=(n,), dtype=jnp.int32),
                    }
                ),
            }
        )

    def image_space(self) -> spaces.Box:
        return spaces.Box(low=0, high=255, shape=(self.consts.HEIGHT, self.consts.WIDTH, 3), dtype=jnp.uint8)

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: JamesBondState) -> JamesBondInfo:
        return JamesBondInfo(
            time=state.step_counter,
            passed=state.passed_counter,
            collisions=state.collided.astype(jnp.int32),
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: JamesBondState, state: JamesBondState) -> float:
        passed_delta = (state.passed_counter - previous_state.passed_counter).astype(jnp.int32)
        pass_reward = passed_delta.astype(jnp.float32) * jnp.array(self.consts.PASS_REWARD, dtype=jnp.float32)
        survive_reward = jnp.array(self.consts.SURVIVE_REWARD, dtype=jnp.float32) * (~state.collided)
        return (pass_reward + survive_reward).astype(jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: JamesBondState) -> bool:
        return state.collided


class JamesBondRenderer(JAXGameRenderer):
    _ID_GRASS = 0
    _ID_ROAD = 1
    _ID_LINE = 2
    _ID_PLAYER = 3
    _ID_ENEMY = 4

    def __init__(self, consts: JamesBondConstants | None = None):
        super().__init__(consts)
        self.consts = consts or JamesBondConstants()

        self.config = render_utils.RendererConfig(game_dimensions=(self.consts.HEIGHT, self.consts.WIDTH), channels=3)
        self.jr = render_utils.JaxRenderingUtils(self.config)

        self.PALETTE = jnp.array(
            [
                self.consts.GRASS_COLOR,
                self.consts.ROAD_COLOR,
                self.consts.LINE_COLOR,
                self.consts.PLAYER_COLOR,
                self.consts.ENEMY_COLOR,
            ],
            dtype=jnp.uint8,
        )

        self.BACKGROUND = jnp.full(
            (self.consts.HEIGHT, self.consts.WIDTH),
            jnp.array(self._ID_GRASS, dtype=jnp.uint8),
            dtype=jnp.uint8,
        )

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: JamesBondState) -> jnp.ndarray:
        raster = self.jr.create_object_raster(self.BACKGROUND)

        road_pos = jnp.array([[self.consts.ROAD_LEFT, 0]], dtype=jnp.int32)
        road_size = jnp.array([[self.consts.ROAD_RIGHT - self.consts.ROAD_LEFT, self.consts.HEIGHT]], dtype=jnp.int32)
        raster = self.jr.draw_rects(raster, road_pos, road_size, self._ID_ROAD)

        edge_pos = jnp.array(
            [
                [self.consts.ROAD_LEFT - self.consts.ROAD_EDGE_LINE_W, 0],
                [self.consts.ROAD_RIGHT, 0],
            ],
            dtype=jnp.int32,
        )
        edge_size = jnp.array(
            [
                [self.consts.ROAD_EDGE_LINE_W, self.consts.HEIGHT],
                [self.consts.ROAD_EDGE_LINE_W, self.consts.HEIGHT],
            ],
            dtype=jnp.int32,
        )
        raster = self.jr.draw_rects(raster, edge_pos, edge_size, self._ID_LINE)

        num_dashes = (self.consts.HEIGHT // self.consts.DASH_SPACING) + 3
        dash_x = jnp.array((self.consts.ROAD_LEFT + self.consts.ROAD_RIGHT) // 2 - (self.consts.DASH_W // 2), dtype=jnp.int32)
        ys = (
            (jnp.arange(num_dashes, dtype=jnp.int32) * jnp.array(self.consts.DASH_SPACING, dtype=jnp.int32) + state.scroll_offset)
            % jnp.array(self.consts.HEIGHT + self.consts.DASH_SPACING, dtype=jnp.int32)
        ) - jnp.array(self.consts.DASH_H, dtype=jnp.int32)
        visible = (ys >= 0) & (ys < jnp.array(self.consts.HEIGHT, dtype=jnp.int32))
        dash_xs = jnp.where(visible, dash_x, jnp.array(-1, dtype=jnp.int32))
        dash_pos = jnp.stack([dash_xs, ys], axis=1).astype(jnp.int32)
        dash_size = jnp.tile(
            jnp.array([[self.consts.DASH_W, self.consts.DASH_H]], dtype=jnp.int32),
            (num_dashes, 1),
        )
        raster = self.jr.draw_rects(raster, dash_pos, dash_size, self._ID_LINE)

        player_pos = jnp.array([[state.player_x, self.consts.PLAYER_Y]], dtype=jnp.int32)
        player_size = jnp.array([[self.consts.PLAYER_SIZE[0], self.consts.PLAYER_SIZE[1]]], dtype=jnp.int32)
        raster = self.jr.draw_rects(raster, player_pos, player_size, self._ID_PLAYER)

        enemy_x = jnp.where(state.enemy_active, state.enemy_x, jnp.array(-1, dtype=jnp.int32))
        enemy_pos = jnp.stack([enemy_x, state.enemy_y], axis=1).astype(jnp.int32)
        enemy_size = jnp.tile(
            jnp.array([[self.consts.ENEMY_SIZE[0], self.consts.ENEMY_SIZE[1]]], dtype=jnp.int32),
            (self.consts.MAX_ENEMIES, 1),
        )
        raster = self.jr.draw_rects(raster, enemy_pos, enemy_size, self._ID_ENEMY)

        return self.jr.render_from_palette(raster, self.PALETTE)


