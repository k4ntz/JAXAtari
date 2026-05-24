import os
from functools import partial
from typing import Tuple

import numpy as np
import chex
import jax.lax
import jax.numpy as jnp
from flax import struct

import jaxatari.spaces as spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action, ObjectObservation
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils

def _create_demon_sprite(consts: "DemonAttackConstants") -> jnp.ndarray:
    mask = jnp.array([
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
    ], dtype=jnp.uint8)

    sprite = jnp.zeros((*consts.DEMON_SIZE, 4), dtype=jnp.uint8)
    color = jnp.array((*consts.DEMON_COLOR, 255), dtype=jnp.uint8)

    # Center the 8x10 mask in the 8x12 sprite
    start_col = (consts.DEMON_SIZE[1] - 10) // 2

    mask_rgba = jnp.where(mask[:, :, None] == 1, color, jnp.zeros(4, dtype=jnp.uint8))
    sprite = sprite.at[:, start_col:start_col + 10, :].set(mask_rgba)

    return sprite

def _create_player_sprite(consts: "DemonAttackConstants") -> jnp.ndarray:
    mask = jnp.array([
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 1, 1, 1, 1, 0, 1, 1],
    ], dtype=jnp.uint8)

    sprite = jnp.zeros((*consts.PLAYER_SIZE, 4), dtype=jnp.uint8)
    color = jnp.array((*consts.PLAYER_COLOR, 255), dtype=jnp.uint8)

    # Center the 8x10 mask in the 8x12 sprite
    start_col = (consts.PLAYER_SIZE[1] - 10) // 2

    mask_rgba = jnp.where(mask[:, :, None] == 1, color, jnp.zeros(4, dtype=jnp.uint8))
    sprite = sprite.at[:, start_col:start_col + 10, :].set(mask_rgba)

    return sprite

def _create_projectile_sprite(size: Tuple[int, int], color_rgb: Tuple[int, int, int]) -> jnp.ndarray:
    sprite = np.zeros((*size, 4), dtype=np.uint8)
    sprite[:, :] = (*color_rgb, 255)
    return jnp.array(sprite)

def _create_digit_sprites(consts: "DemonAttackConstants") -> jnp.ndarray:
    digits = np.zeros((10, 8, 8, 4), dtype=np.uint8)
    color = (*consts.SCORE_COLOR, 255)

    patterns = [
        [[1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1]],
        [[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]],
        [[1, 1, 1], [0, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1]],
        [[1, 1, 1], [0, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1]],
        [[1, 0, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1]],
        [[1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 1], [1, 1, 1]],
        [[1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 0, 1], [1, 1, 1]],
        [[1, 1, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]],
        [[1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 1, 1]],
        [[1, 1, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1]],
    ]

    for i, pattern in enumerate(patterns):
        for r, row in enumerate(pattern):
            for c, val in enumerate(row):
                if val:
                    digits[i, r + 1, c + 2] = color

    return jnp.array(digits)

def _create_explosion_sprite(consts: "DemonAttackConstants") -> jnp.ndarray:
    mask = jnp.array([
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [1, 0, 0, 1, 1, 1, 1, 0, 0, 1],
        [0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
        [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        [0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
        [1, 0, 0, 1, 1, 1, 1, 0, 0, 1],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    ], dtype=jnp.uint8)

    sprite = jnp.zeros((*consts.PLAYER_SIZE, 4), dtype=jnp.uint8)
    color = jnp.array((*consts.PLAYER_COLOR, 255), dtype=jnp.uint8)

    start_col = (consts.PLAYER_SIZE[1] - 10) // 2
    mask_rgba = jnp.where(mask[:, :, None] == 1, color, jnp.zeros(4, dtype=jnp.uint8))
    sprite = sprite.at[:, start_col:start_col + 10, :].set(mask_rgba)

    return sprite

def _get_default_asset_config() -> tuple:
    return (
        {'name': 'background', 'type': 'procedural'},
        {'name': 'player', 'type': 'procedural'},
        {'name': 'demon', 'type': 'procedural'},
        {'name': 'projectile_player', 'type': 'procedural'},
        {'name': 'projectile_demon', 'type': 'procedural'},
        {'name': 'explosion', 'type': 'procedural'},
        {'name': 'score_digits', 'type': 'procedural'},
    )

class DemonAttackConstants(struct.PyTreeNode):
    # Static Configuration
    WIDTH: int = struct.field(pytree_node=False, default=160)
    HEIGHT: int = struct.field(pytree_node=False, default=210)
    PLAYER_SPEED: int = struct.field(pytree_node=False, default=2)
    MAX_DEMONS: int = struct.field(pytree_node=False, default=3)
    DEMON_SPEED: int = struct.field(pytree_node=False, default=1)
    LASER_SPEED: int = struct.field(pytree_node=False, default=4)
    BOMB_SPEED: int = struct.field(pytree_node=False, default=2)

    # Coordinates & Sizes
    PLAYER_Y: int = struct.field(pytree_node=False, default=184)
    PLAYER_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default=(8, 12))
    DEMON_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default=(8, 12))
    LASER_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default=(1, 6))
    BOMB_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default=(2, 4))

    # Boundaries
    PLAYER_MIN_X: int = struct.field(pytree_node=False, default=16)
    PLAYER_MAX_X: int = struct.field(pytree_node=False, default=136)
    DEMON_MIN_X: int = struct.field(pytree_node=False, default=16)
    DEMON_MAX_X: int = struct.field(pytree_node=False, default=136)

    # Colors
    BACKGROUND_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(0, 0, 0))
    PLAYER_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(184, 70, 162))
    DEMON_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(184, 70, 162))
    LASER_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(252, 252, 252))
    BOMB_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(252, 252, 252))
    SCORE_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(252, 252, 252))

    ASSET_CONFIG: tuple = struct.field(pytree_node=False, default_factory=_get_default_asset_config)

class DemonAttackState(struct.PyTreeNode):
    player_x: chex.Array
    laser_x: chex.Array
    laser_y: chex.Array
    laser_active: chex.Array

    demons_x: chex.Array
    demons_y: chex.Array  # Shape: (MAX_DEMONS,)
    demons_dir: chex.Array  # Shape: (MAX_DEMONS,) 1 for right, -1 for left
    demons_alive: chex.Array  # Shape: (MAX_DEMONS,) bool

    bomb_x: chex.Array
    bomb_y: chex.Array
    bomb_active: chex.Array

    score: chex.Array
    lives: chex.Array
    player_exploding: chex.Array
    explosion_timer: chex.Array
    step_counter: chex.Array
    key: chex.PRNGKey

class DemonAttackObservation(struct.PyTreeNode):
    player: ObjectObservation
    demons: ObjectObservation
    laser: ObjectObservation
    bomb: ObjectObservation
    score: jnp.ndarray
    lives: jnp.ndarray

class DemonAttackInfo(struct.PyTreeNode):
    time: jnp.ndarray

class JaxDemonAttack(JaxEnvironment[DemonAttackState, DemonAttackObservation, DemonAttackInfo, DemonAttackConstants]):
    ACTION_SET: jnp.ndarray = jnp.array(
        [Action.NOOP, Action.FIRE, Action.RIGHT, Action.LEFT, Action.RIGHTFIRE, Action.LEFTFIRE],
        dtype=jnp.int32,
    )

    def __init__(self, consts: DemonAttackConstants = None):
        consts = consts or DemonAttackConstants()
        super().__init__(consts)
        self.renderer = DemonAttackRenderer(self.consts)

    def reset(self, key: chex.PRNGKey = jax.random.PRNGKey(42)) -> Tuple[DemonAttackObservation, DemonAttackState]:
        key, player_key, demon_key = jax.random.split(key, 3)

        state = DemonAttackState(
            player_x=jnp.array(76, dtype=jnp.int32),
            laser_x=jnp.array(0, dtype=jnp.int32),
            laser_y=jnp.array(0, dtype=jnp.int32),
            laser_active=jnp.array(False, dtype=jnp.bool_),
            demons_x=jnp.linspace(20, 120, self.consts.MAX_DEMONS, dtype=jnp.int32),
            demons_y=jnp.full((self.consts.MAX_DEMONS,), 40, dtype=jnp.int32),
            demons_dir=jnp.ones((self.consts.MAX_DEMONS,), dtype=jnp.int32),
            demons_alive=jnp.ones((self.consts.MAX_DEMONS,), dtype=jnp.bool_),
            bomb_x=jnp.array(0, dtype=jnp.int32),
            bomb_y=jnp.array(0, dtype=jnp.int32),
            bomb_active=jnp.array(False, dtype=jnp.bool_),
            score=jnp.array(0, dtype=jnp.int32),
            lives=jnp.array(3, dtype=jnp.int32),
            player_exploding=jnp.array(False, dtype=jnp.bool_),
            explosion_timer=jnp.array(0, dtype=jnp.int32),
            step_counter=jnp.array(0, dtype=jnp.int32),
            key=key
        )
        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: DemonAttackState, action: chex.Array) -> Tuple[
        DemonAttackObservation, DemonAttackState, float, bool, DemonAttackInfo]:
        atari_action = jnp.take(self.ACTION_SET, action.astype(jnp.int32))

        prev_state = state

        # Handle Explosion Timer
        def update_explosion(s):
            new_timer = s.explosion_timer - 1
            exploding = new_timer > 0
            # If timer reaches 0, player hit logic should have already reduced lives.
            # We just need to stop exploding.
            return s.replace(explosion_timer=new_timer, player_exploding=exploding)

        def normal_step(s, act):
            # 1. Player Step
            s = self._player_step(s, act)
            # 2. Laser Step
            s = self._laser_step(s, act)
            # 3. Demon Step
            s = self._demons_step(s)
            # 4. Bomb Step
            s = self._bomb_step(s)
            # 5. Collision Detection
            s = self._handle_collisions(s)
            return s

        state = jax.lax.cond(
            state.player_exploding,
            update_explosion,
            lambda s: normal_step(s, atari_action),
            operand=state
        )

        # Update key and step counter
        key, next_key = jax.random.split(state.key)
        state = state.replace(key=next_key, step_counter=state.step_counter + 1)

        observation = self._get_observation(state)
        reward = self._get_reward(prev_state, state)
        done = self._get_done(state)
        info = self._get_info(state)

        return observation, state, reward, done, info

    def _player_step(self, state: DemonAttackState, action: chex.Array) -> DemonAttackState:
        move_right = jnp.logical_or(action == Action.RIGHT, action == Action.RIGHTFIRE)
        move_left = jnp.logical_or(action == Action.LEFT, action == Action.LEFTFIRE)

        dx = jax.lax.select(move_right, self.consts.PLAYER_SPEED,
                            jax.lax.select(move_left, -self.consts.PLAYER_SPEED, 0))

        new_x = jnp.clip(state.player_x + dx, self.consts.PLAYER_MIN_X, self.consts.PLAYER_MAX_X)
        return state.replace(player_x=new_x)

    def _laser_step(self, state: DemonAttackState, action: chex.Array) -> DemonAttackState:
        # Fire laser if not active and FIRE action
        fire = jnp.logical_or(jnp.logical_or(action == Action.FIRE, action == Action.RIGHTFIRE),
                              action == Action.LEFTFIRE)

        should_fire = jnp.logical_and(fire, jnp.logical_not(state.laser_active))

        laser_x = jax.lax.select(should_fire, state.player_x + self.consts.PLAYER_SIZE[0] // 2, state.laser_x)
        laser_y = jax.lax.select(should_fire,
                                 jnp.array(self.consts.PLAYER_Y - self.consts.LASER_SIZE[1], dtype=jnp.int32),
                                 state.laser_y)
        laser_active = jnp.logical_or(should_fire, state.laser_active)

        # Move laser
        laser_y = jax.lax.select(laser_active, laser_y - self.consts.LASER_SPEED, laser_y)

        # Deactivate if out of bounds
        laser_active = jnp.logical_and(laser_active, laser_y > 0)

        return state.replace(laser_x=laser_x, laser_y=laser_y, laser_active=laser_active)

    def _demons_step(self, state: DemonAttackState) -> DemonAttackState:
        # Simple demon movement: move horizontally, bounce at edges
        new_x = state.demons_x + state.demons_dir * self.consts.DEMON_SPEED

        at_right_edge = new_x >= self.consts.DEMON_MAX_X
        at_left_edge = new_x <= self.consts.DEMON_MIN_X

        new_dir = jnp.where(at_right_edge, -1, jnp.where(at_left_edge, 1, state.demons_dir))
        new_x = jnp.clip(new_x, self.consts.DEMON_MIN_X, self.consts.DEMON_MAX_X)

        # Respawn demons if all are dead
        all_dead = jnp.logical_not(jnp.any(state.demons_alive))
        demons_alive = jnp.where(all_dead, True, state.demons_alive)

        return state.replace(demons_x=new_x, demons_dir=new_dir, demons_alive=demons_alive)

    def _bomb_step(self, state: DemonAttackState) -> DemonAttackState:
        # Drop bomb from a random living demon if no bomb is active
        key, drop_key, demon_idx_key = jax.random.split(state.key, 3)

        should_drop = jnp.logical_and(jnp.logical_not(state.bomb_active), jax.random.uniform(drop_key) < 0.05)

        # Pick a random demon index
        demon_idx = jax.random.randint(demon_idx_key, (), 0, self.consts.MAX_DEMONS)
        demon_idx = jnp.where(state.demons_alive[demon_idx], demon_idx, jnp.argmax(state.demons_alive))

        bomb_x = jax.lax.select(should_drop, state.demons_x[demon_idx] + self.consts.DEMON_SIZE[0] // 2, state.bomb_x)
        bomb_y = jax.lax.select(should_drop, state.demons_y[demon_idx] + self.consts.DEMON_SIZE[1], state.bomb_y)
        bomb_active = jnp.logical_or(should_drop, state.bomb_active)

        # Move bomb
        bomb_y = jax.lax.select(bomb_active, bomb_y + self.consts.BOMB_SPEED, bomb_y)

        # Deactivate if out of bounds
        bomb_active = jnp.logical_and(bomb_active, bomb_y < self.consts.HEIGHT)

        return state.replace(bomb_x=bomb_x, bomb_y=bomb_y, bomb_active=bomb_active, key=key)

    def _handle_collisions(self, state: DemonAttackState) -> DemonAttackState:
        # Laser vs Demons
        def check_demon_collision(i, carry):
            s_alive, s_score, l_active = carry

            demon_hit = jnp.logical_and(
                s_alive[i],
                jnp.logical_and(
                    l_active,
                    jnp.logical_and(
                        jnp.abs(state.laser_x - state.demons_x[i]) < self.consts.DEMON_SIZE[0],
                        jnp.logical_and(
                            state.laser_y < state.demons_y[i] + self.consts.DEMON_SIZE[1],
                            state.laser_y + self.consts.LASER_SIZE[1] > state.demons_y[i]
                        )
                    )
                )
            )

            new_alive = s_alive.at[i].set(jnp.logical_and(s_alive[i], jnp.logical_not(demon_hit)))
            new_score = jnp.where(demon_hit, s_score + 10, s_score)
            new_laser_active = jnp.logical_and(l_active, jnp.logical_not(demon_hit))

            return (new_alive, new_score, new_laser_active)

        init_carry = (state.demons_alive, state.score, state.laser_active)
        demons_alive, score, laser_active = jax.lax.fori_loop(0, self.consts.MAX_DEMONS, check_demon_collision,
                                                              init_carry)

        # Bomb vs Player
        player_hit = jnp.logical_and(
            state.bomb_active,
            jnp.logical_and(
                jnp.abs(state.bomb_x - state.player_x) < self.consts.PLAYER_SIZE[0],
                jnp.logical_and(
                    state.bomb_y < self.consts.PLAYER_Y + self.consts.PLAYER_SIZE[1],
                    state.bomb_y + self.consts.BOMB_SIZE[1] > self.consts.PLAYER_Y
                )
            )
        )

        lives = jnp.where(player_hit, state.lives - 1, state.lives)
        score = jnp.where(player_hit, 0, score)
        bomb_active = jnp.logical_and(state.bomb_active, jnp.logical_not(player_hit))

        # If player hit, start explosion
        player_exploding = jnp.logical_or(state.player_exploding, player_hit)
        explosion_timer = jnp.where(player_hit, 20, state.explosion_timer)

        return state.replace(demons_alive=demons_alive, score=score, laser_active=laser_active,
                             lives=lives, bomb_active=bomb_active,
                             player_exploding=player_exploding, explosion_timer=explosion_timer)

    def render(self, state: DemonAttackState) -> jnp.ndarray:
        return self.renderer.render(state)

    def _get_observation(self, state: DemonAttackState):
        player = ObjectObservation.create(
            x=state.player_x,
            y=jnp.array(self.consts.PLAYER_Y),
            width=jnp.array(self.consts.PLAYER_SIZE[0]),
            height=jnp.array(self.consts.PLAYER_SIZE[1]),
        )

        demons = ObjectObservation.create(
            x=state.demons_x,
            y=state.demons_y,
            width=jnp.array(self.consts.DEMON_SIZE[0]),
            height=jnp.array(self.consts.DEMON_SIZE[1]),
            active=state.demons_alive
        )

        laser = ObjectObservation.create(
            x=state.laser_x,
            y=state.laser_y,
            width=jnp.array(self.consts.LASER_SIZE[0]),
            height=jnp.array(self.consts.LASER_SIZE[1]),
            active=state.laser_active
        )

        bomb = ObjectObservation.create(
            x=state.bomb_x,
            y=state.bomb_y,
            width=jnp.array(self.consts.BOMB_SIZE[0]),
            height=jnp.array(self.consts.BOMB_SIZE[1]),
            active=state.bomb_active
        )

        return DemonAttackObservation(
            player=player,
            demons=demons,
            laser=laser,
            bomb=bomb,
            score=state.score,
            lives=state.lives
        )

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTION_SET))

    def observation_space(self) -> spaces.Dict:
        object_space = spaces.get_object_space(n=None, screen_size=(self.consts.HEIGHT, self.consts.WIDTH))
        demons_space = spaces.get_object_space(n=self.consts.MAX_DEMONS,
                                               screen_size=(self.consts.HEIGHT, self.consts.WIDTH))

        return spaces.Dict({
            "player": object_space,
            "demons": demons_space,
            "laser": object_space,
            "bomb": object_space,
            "score": spaces.Box(low=0, high=99999, shape=(), dtype=jnp.int32),
            "lives": spaces.Box(low=0, high=6, shape=(), dtype=jnp.int32),
        })

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(210, 160, 3),
            dtype=jnp.uint8
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: DemonAttackState, ) -> DemonAttackInfo:
        return DemonAttackInfo(time=state.step_counter)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: DemonAttackState, state: DemonAttackState):
        return (state.score - previous_state.score).astype(jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: DemonAttackState) -> bool:
        return state.lives <= 0

class DemonAttackRenderer(JAXGameRenderer):
    def __init__(self, consts: DemonAttackConstants = None, config: render_utils.RendererConfig = None):
        super().__init__(consts)
        self.consts = consts or DemonAttackConstants()

        if config is None:
            self.config = render_utils.RendererConfig(
                game_dimensions=(210, 160),
                channels=3,
                downscale=None
            )
        else:
            self.config = config

        self.jr = render_utils.JaxRenderingUtils(self.config)

        # Create procedural assets
        bg_rgba = jnp.zeros((*self.config.game_dimensions, 4), dtype=jnp.uint8)
        bg_rgba = bg_rgba.at[:, :, :3].set(jnp.array(self.consts.BACKGROUND_COLOR))
        bg_rgba = bg_rgba.at[:, :, 3].set(255)

        player_sprite = _create_player_sprite(self.consts)
        demon_sprite = _create_demon_sprite(self.consts)
        laser_sprite = _create_projectile_sprite(self.consts.LASER_SIZE, self.consts.LASER_COLOR)
        bomb_sprite = _create_projectile_sprite(self.consts.BOMB_SIZE, self.consts.BOMB_COLOR)
        explosion_sprite = _create_explosion_sprite(self.consts)
        digit_sprites = _create_digit_sprites(self.consts)

        # Update asset config with procedural data
        asset_config = [
            {'name': 'background', 'type': 'background', 'data': bg_rgba},
            {'name': 'player', 'type': 'procedural', 'data': player_sprite},
            {'name': 'demon', 'type': 'procedural', 'data': demon_sprite},
            {'name': 'projectile_player', 'type': 'procedural', 'data': laser_sprite},
            {'name': 'projectile_demon', 'type': 'procedural', 'data': bomb_sprite},
            {'name': 'explosion', 'type': 'procedural', 'data': explosion_sprite},
            {'name': 'score_digits', 'type': 'procedural', 'data': digit_sprites},
        ]

        # Bake assets
        sprite_path = os.path.join(render_utils.get_base_sprite_dir(), "demonattack")
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(asset_config, sprite_path)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: DemonAttackState):
        raster = self.jr.create_object_raster(self.BACKGROUND)

        # Render player or explosion
        player_mask = jax.lax.select(state.player_exploding, self.SHAPE_MASKS["explosion"], self.SHAPE_MASKS["player"])
        raster = self.jr.render_at(raster, state.player_x, self.consts.PLAYER_Y, player_mask)

        # Render demons
        demon_mask = self.SHAPE_MASKS["demon"]

        def render_demon(i, r):
            return jax.lax.cond(
                state.demons_alive[i],
                lambda: self.jr.render_at(r, state.demons_x[i], state.demons_y[i], demon_mask),
                lambda: r
            )

        raster = jax.lax.fori_loop(0, self.consts.MAX_DEMONS, render_demon, raster)

        # Render laser
        laser_mask = self.SHAPE_MASKS["projectile_player"]
        raster = jax.lax.cond(
            state.laser_active,
            lambda: self.jr.render_at(raster, state.laser_x, state.laser_y, laser_mask),
            lambda: raster
        )

        # Render bomb
        bomb_mask = self.SHAPE_MASKS["projectile_demon"]
        raster = jax.lax.cond(
            state.bomb_active,
            lambda: self.jr.render_at(raster, state.bomb_x, state.bomb_y, bomb_mask),
            lambda: raster
        )

        # Render Score
        score_digits = self.jr.int_to_digits(state.score, max_digits=4)
        digit_masks = self.SHAPE_MASKS["score_digits"]

        is_single_digit = state.score < 10
        is_double_digit = jnp.logical_and(state.score >= 10, state.score < 100)
        is_triple_digit = jnp.logical_and(state.score >= 100, state.score < 1000)

        start_index = jax.lax.select(is_single_digit, 3,
                                     jax.lax.select(is_double_digit, 2,
                                                    jax.lax.select(is_triple_digit, 1, 0)))
        num_to_render = jax.lax.select(is_single_digit, 1,
                                       jax.lax.select(is_double_digit, 2,
                                                      jax.lax.select(is_triple_digit, 3, 4)))

        raster = self.jr.render_label_selective(raster, 70, 10, score_digits, digit_masks,
                                                start_index, num_to_render, spacing=8)

        return self.jr.render_from_palette(raster, self.PALETTE)