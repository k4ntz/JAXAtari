import os
from functools import partial
from typing import Tuple, Optional

import jax
import jax.numpy as jnp
import jax.random as jrandom
import chex
from flax import struct

import jaxatari.rendering.jax_rendering_utils as render_utils
import jaxatari.spaces as spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action, ObjectObservation
from jaxatari.renderers import JAXGameRenderer


def _get_default_asset_config() -> tuple:
    """Manifest of the .npy sprites the renderer loads from sprites/icehockey/.

    Run scripts/make_icehockey_sprites.py once to create the placeholder files.
    """
    return (
        {"name": "background", "type": "background", "file": "background.npy"},
        {"name": "player", "type": "single", "file": "player.npy"},
        {"name": "enemy", "type": "single", "file": "enemy.npy"},
        {"name": "puck", "type": "single", "file": "puck.npy"},
        {"name": "digits", "type": "digits", "pattern": "digit_{}.npy"},
    )


class IceHockeyConstants(struct.PyTreeNode):
    # Static parameters. Marked pytree_node=False so JAX keeps them as static
    # metadata instead of tracing them.
    WIDTH: int = struct.field(pytree_node=False, default=160)
    HEIGHT: int = struct.field(pytree_node=False, default=210)

    # Rink interior in pixels (inside the boards).
    RINK_LEFT: int = struct.field(pytree_node=False, default=4)
    RINK_RIGHT: int = struct.field(pytree_node=False, default=155)
    RINK_TOP: int = struct.field(pytree_node=False, default=20)
    RINK_BOTTOM: int = struct.field(pytree_node=False, default=190)

    # Goals. Player defends the bottom, enemy the top.
    GOAL_X0: int = struct.field(pytree_node=False, default=60)
    GOAL_X1: int = struct.field(pytree_node=False, default=100)
    ENEMY_GOAL_Y: int = struct.field(pytree_node=False, default=20)
    PLAYER_GOAL_Y: int = struct.field(pytree_node=False, default=187)
    GOAL_HEIGHT: int = struct.field(pytree_node=False, default=7)

    # Sprite sizes, used for observation bounding boxes.
    PLAYER_W: int = struct.field(pytree_node=False, default=8)
    PLAYER_H: int = struct.field(pytree_node=False, default=12)
    PUCK_W: int = struct.field(pytree_node=False, default=4)
    PUCK_H: int = struct.field(pytree_node=False, default=3)

    PLAYER_SPEED: float = struct.field(pytree_node=False, default=1.5)

    # 3 min * 60 s * 60 fps = 10800 raw frames.
    TIME_LIMIT: int = struct.field(pytree_node=False, default=10800)
    FACE_OFF_FRAMES: int = struct.field(pytree_node=False, default=40)

    # Face-off layout. [x, y] = [col, row]. Estimated from the ALE screen;
    # refine against captured frames once real sprites are in.
    FACEOFF_X: float = struct.field(pytree_node=False, default=78.0)
    FACEOFF_Y: float = struct.field(pytree_node=False, default=103.0)
    P1_X: float = struct.field(pytree_node=False, default=60.0)
    P1_Y: float = struct.field(pytree_node=False, default=115.0)
    P2_X: float = struct.field(pytree_node=False, default=85.0)
    P2_Y: float = struct.field(pytree_node=False, default=150.0)
    E1_X: float = struct.field(pytree_node=False, default=85.0)
    E1_Y: float = struct.field(pytree_node=False, default=89.0)
    E2_X: float = struct.field(pytree_node=False, default=60.0)
    E2_Y: float = struct.field(pytree_node=False, default=54.0)

    # Asset manifest lives in the constants so the modding framework can apply
    # asset_overrides before the renderer is constructed.
    ASSET_CONFIG: tuple = struct.field(
        pytree_node=False, default_factory=_get_default_asset_config
    )


@struct.dataclass
class GameState:
    pause_counter: chex.Array
    player_score: chex.Array
    enemy_score: chex.Array
    remaining_time: chex.Array
    is_faceoff: chex.Array
    goal_scored: chex.Array
    is_finished: chex.Array


@struct.dataclass
class CharacterState:
    is_tackled: chex.Array
    position: chex.Array        # float32 [x, y]
    orientation: chex.Array     # 0 = left, 1 = right
    has_puck: chex.Array
    shooting_cooldown: chex.Array


@struct.dataclass
class PuckState:
    position: chex.Array        # float32 [x, y]
    velocity: chex.Array        # float32 [vx, vy]
    direction: chex.Array       # shot angle slot, 0-31
    position_stick: chex.Array  # slot on the stick arc while carried, 0-31


@struct.dataclass
class PlayerState:
    player1: CharacterState
    player2: CharacterState
    active_character: chex.Array   # 0 = player1 controlled, 1 = player2


@struct.dataclass
class EnemyState:
    enemy1: CharacterState
    enemy2: CharacterState
    active_character: chex.Array


@struct.dataclass
class IceHockeyState:
    player_state: PlayerState
    enemy_state: EnemyState
    puck_state: PuckState
    counter: chex.Array
    game_state: GameState


@struct.dataclass
class IceHockeyInfo:
    player_score: chex.Array
    enemy_score: chex.Array
    remaining_time: chex.Array


@struct.dataclass
class IceHockeyObservation:
    player1: ObjectObservation
    player2: ObjectObservation
    enemy1: ObjectObservation
    enemy2: ObjectObservation
    puck: ObjectObservation
    player_score: chex.Array
    enemy_score: chex.Array
    remaining_time: chex.Array
    active_player: chex.Array


class JaxIceHockey(JaxEnvironment):

    # IceHockey uses the full ALE action set, so the agent index maps straight
    # onto the ALE action integer.
    ACTION_SET = jnp.array([
        Action.NOOP, Action.FIRE, Action.UP, Action.RIGHT, Action.LEFT, Action.DOWN,
        Action.UPRIGHT, Action.UPLEFT, Action.DOWNRIGHT, Action.DOWNLEFT,
        Action.UPFIRE, Action.RIGHTFIRE, Action.LEFTFIRE, Action.DOWNFIRE,
        Action.UPRIGHTFIRE, Action.UPLEFTFIRE, Action.DOWNRIGHTFIRE, Action.DOWNLEFTFIRE,
    ], dtype=jnp.int32)

    def __init__(self, consts: Optional[IceHockeyConstants] = None):
        consts = consts or IceHockeyConstants()
        super().__init__(consts)
        self.renderer = IceHockeyRenderer(self.consts)

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTION_SET))

    def observation_space(self) -> spaces.Dict:
        obj = spaces.get_object_space(n=None, screen_size=(self.consts.HEIGHT, self.consts.WIDTH))
        return spaces.Dict({
            "player1": obj,
            "player2": obj,
            "enemy1": obj,
            "enemy2": obj,
            "puck": obj,
            "player_score": spaces.Box(0, 99, shape=(), dtype=jnp.int32),
            "enemy_score": spaces.Box(0, 99, shape=(), dtype=jnp.int32),
            "remaining_time": spaces.Box(0, self.consts.TIME_LIMIT, shape=(), dtype=jnp.int32),
            "active_player": spaces.Box(0, 1, shape=(), dtype=jnp.int32),
        })

    def image_space(self) -> spaces.Box:
        return spaces.Box(low=0, high=255, shape=(210, 160, 3), dtype=jnp.uint8)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey = None) -> Tuple:
        # Face-off: puck at centre, each team's active skater near the dot and
        # the second skater back in its zone, everyone frozen for a short while.
        c = self.consts

        def char(x, y):
            return CharacterState(
                is_tackled=jnp.array(False),
                position=jnp.array([x, y], dtype=jnp.float32),
                orientation=jnp.array(0, dtype=jnp.int32),
                has_puck=jnp.array(False),
                shooting_cooldown=jnp.array(0, dtype=jnp.int32),
            )

        state = IceHockeyState(
            player_state=PlayerState(
                player1=char(c.P1_X, c.P1_Y),
                player2=char(c.P2_X, c.P2_Y),
                active_character=jnp.array(0, dtype=jnp.int32),
            ),
            enemy_state=EnemyState(
                enemy1=char(c.E1_X, c.E1_Y),
                enemy2=char(c.E2_X, c.E2_Y),
                active_character=jnp.array(0, dtype=jnp.int32),
            ),
            puck_state=PuckState(
                position=jnp.array([c.FACEOFF_X, c.FACEOFF_Y], dtype=jnp.float32),
                velocity=jnp.array([0.0, 0.0], dtype=jnp.float32),
                direction=jnp.array(0, dtype=jnp.int32),
                position_stick=jnp.array(0, dtype=jnp.int32),
            ),
            counter=jnp.array(0, dtype=jnp.int32),
            game_state=GameState(
                pause_counter=jnp.array(c.FACE_OFF_FRAMES, dtype=jnp.int32),
                player_score=jnp.array(0, dtype=jnp.int32),
                enemy_score=jnp.array(0, dtype=jnp.int32),
                remaining_time=jnp.array(c.TIME_LIMIT, dtype=jnp.int32),
                is_faceoff=jnp.array(True),
                goal_scored=jnp.array(False),
                is_finished=jnp.array(False),
            ),
        )
        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: IceHockeyState, action):
        # Stub: returns the state unchanged with the correct return signature.
        # Game logic (player/enemy/puck/collision/goal/timer steps) comes next.
        previous_state = state
        obs = self._get_observation(state)
        reward = self._get_reward(previous_state, state)
        done = self._get_done(state)
        info = self._get_info(state)
        return obs, state, reward, done, info

    def render(self, state: IceHockeyState) -> jnp.ndarray:
        return self.renderer.render(state)

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: IceHockeyState) -> IceHockeyObservation:
        c = self.consts

        def obj(pos, w, h):
            return ObjectObservation.create(
                x=pos[0].astype(jnp.int32),
                y=pos[1].astype(jnp.int32),
                width=jnp.array(w, dtype=jnp.int32),
                height=jnp.array(h, dtype=jnp.int32),
            )

        return IceHockeyObservation(
            player1=obj(state.player_state.player1.position, c.PLAYER_W, c.PLAYER_H),
            player2=obj(state.player_state.player2.position, c.PLAYER_W, c.PLAYER_H),
            enemy1=obj(state.enemy_state.enemy1.position, c.PLAYER_W, c.PLAYER_H),
            enemy2=obj(state.enemy_state.enemy2.position, c.PLAYER_W, c.PLAYER_H),
            puck=obj(state.puck_state.position, c.PUCK_W, c.PUCK_H),
            player_score=state.game_state.player_score,
            enemy_score=state.game_state.enemy_score,
            remaining_time=state.game_state.remaining_time,
            active_player=state.player_state.active_character,
        )

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: IceHockeyObservation) -> jnp.ndarray:
        def flat(o):
            return jnp.array([o.x, o.y, o.width, o.height, o.active], dtype=jnp.float32)

        return jnp.concatenate([
            flat(obs.player1), flat(obs.player2),
            flat(obs.enemy1), flat(obs.enemy2),
            flat(obs.puck),
            jnp.array([obs.player_score, obs.enemy_score,
                       obs.remaining_time, obs.active_player], dtype=jnp.float32),
        ])

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: IceHockeyState) -> IceHockeyInfo:
        return IceHockeyInfo(
            player_score=state.game_state.player_score,
            enemy_score=state.game_state.enemy_score,
            remaining_time=state.game_state.remaining_time,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: IceHockeyState, state: IceHockeyState) -> chex.Array:
        # Reward is the change in goal difference: +1 scored, -1 conceded.
        prev_diff = previous_state.game_state.player_score - previous_state.game_state.enemy_score
        diff = state.game_state.player_score - state.game_state.enemy_score
        return (diff - prev_diff).astype(jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: IceHockeyState) -> chex.Array:
        return state.game_state.is_finished


class IceHockeyRenderer(JAXGameRenderer):
    # Palette-based renderer. The rink (boards, lines, goals, score bars) is
    # baked into the background, so render() only stamps the moving objects.

    def __init__(self, consts: Optional[IceHockeyConstants] = None):
        self.consts = consts or IceHockeyConstants()
        super().__init__(self.consts)

        self.config = render_utils.RendererConfig(
            game_dimensions=(210, 160), channels=3, downscale=None,
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)

        # Branch-local sprite folder for now; move to the shared sprite dir later.
        self.sprite_path = os.path.join(os.path.dirname(__file__), "sprites", "icehockey")

        final_asset_config = list(self.consts.ASSET_CONFIG)
        (self.PALETTE, self.SHAPE_MASKS, self.BACKGROUND,
         self.COLOR_TO_ID, self.FLIP_OFFSETS) = self.jr.load_and_setup_assets(
            final_asset_config, self.sprite_path
        )

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: IceHockeyState) -> jnp.ndarray:
        raster = self.jr.create_object_raster(self.BACKGROUND)

        pm = self.SHAPE_MASKS["player"]
        em = self.SHAPE_MASKS["enemy"]
        puck_m = self.SHAPE_MASKS["puck"]

        def col(pos):
            return jnp.round(pos[0]).astype(jnp.int32)

        def row(pos):
            return jnp.round(pos[1]).astype(jnp.int32)

        p1 = state.player_state.player1.position
        p2 = state.player_state.player2.position
        e1 = state.enemy_state.enemy1.position
        e2 = state.enemy_state.enemy2.position
        pp = state.puck_state.position

        # render_at_clipped because skaters can reach the board pixels at the
        # edge; render_at would slice out of bounds there.
        raster = self.jr.render_at_clipped(raster, col(p2), row(p2), pm)
        raster = self.jr.render_at_clipped(raster, col(e2), row(e2), em)
        raster = self.jr.render_at_clipped(raster, col(p1), row(p1), pm)
        raster = self.jr.render_at_clipped(raster, col(e1), row(e1), em)
        raster = self.jr.render_at_clipped(raster, col(pp), row(pp), puck_m)

        dm = self.SHAPE_MASKS["digits"]

        def draw_score(r, value, x_single, x_double):
            digits = self.jr.int_to_digits(value, max_digits=2)
            is_single = value < 10
            start = jax.lax.select(is_single, jnp.int32(1), jnp.int32(0))
            count = jax.lax.select(is_single, jnp.int32(1), jnp.int32(2))
            x = jax.lax.select(is_single, jnp.int32(x_single), jnp.int32(x_double))
            return self.jr.render_label_selective(
                r, x, 3, digits, dm, start, count, spacing=7, max_digits_to_render=2
            )

        raster = draw_score(raster, state.game_state.enemy_score, 43, 33)
        raster = draw_score(raster, state.game_state.player_score, 113, 103)

        return self.jr.render_from_palette(raster, self.PALETTE)