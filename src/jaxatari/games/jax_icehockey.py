"""
jax_icehockey.py
────────────────
JAX reimplementation of Atari 2600 Ice Hockey (ALE/IceHockey-v5).

Game overview (from ALE documentation):
  • Two-on-two top-down ice hockey.
  • Single-player: you control the orange team (bottom half of rink),
    AI controls the blue/white team (top half).
  • Whichever of your two skaters is nearest the puck is auto-selected.
  • Shoot with FIRE (+ direction for angle).  32 shot angles possible.
  • You can pass off the side-boards.
  • 3-minute match (≈10 800 frames at 60 fps).
  • Action space: Discrete(18) — full ALE action set.
  • Screen: 210 (H) × 160 (W).

Tasks covered: 1.2 (assets), 1.3 (spaces / reset), 1.4 (rendering).
Game logic (step) is stubbed and will be added in Task 2.x.
"""

import os
from functools import partial
from typing import Tuple, Optional

import jax
import jax.lax
import jax.numpy as jnp
import jax.random as jrandom
import chex
import numpy as np
from flax import struct

import jaxatari.rendering.jax_rendering_utils as render_utils
import jaxatari.spaces as spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action, ObjectObservation
from jaxatari.renderers import JAXGameRenderer


# ══════════════════════════════════════════════════════════════════════════════
#  Asset configuration
#  Must be a module-level function so struct.PyTreeNode can reference it as a
#  default_factory without capturing anything that could change after class
#  definition.
# ══════════════════════════════════════════════════════════════════════════════

def _get_default_asset_config() -> tuple:
    """
    Declarative manifest of every .npy sprite file the renderer needs.

    All paths are relative to the local sprite folder
    src/jaxatari/games/sprites/icehockey/.

    Run  scripts/make_icehockey_sprites.py  once to populate that folder
    with placeholder images before starting the renderer.
    """
    return (
        # Static background – baked into BACKGROUND raster at init time
        {"name": "background", "type": "background", "file": "background.npy"},
        # Per-frame moving entities
        {"name": "player",  "type": "single", "file": "player.npy"},
        {"name": "enemy",   "type": "single", "file": "enemy.npy"},
        {"name": "puck",    "type": "single", "file": "puck.npy"},
        # Score UI  (digit_0.npy … digit_9.npy)
        {"name": "digits",  "type": "digits", "pattern": "digit_{}.npy"},
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Constants
# ══════════════════════════════════════════════════════════════════════════════

class IceHockeyConstants(struct.PyTreeNode):
    """
    All static, non-learnable parameters of the environment.

    IMPORTANT: every field must have  pytree_node=False  so that JAX treats
    it as a compile-time constant and does *not* try to trace through it.
    Omitting this causes massive recompilation or silent wrong behaviour.
    """

    # ── Screen ────────────────────────────────────────────────────────────
    WIDTH:  int = struct.field(pytree_node=False, default=160)
    HEIGHT: int = struct.field(pytree_node=False, default=210)

    # ── Rink pixel bounds (inclusive) ─────────────────────────────────────
    # Rows 16-193 and cols 4-155 are inside the boards.
    RINK_LEFT:   int = struct.field(pytree_node=False, default=4)
    RINK_RIGHT:  int = struct.field(pytree_node=False, default=155)
    RINK_TOP:    int = struct.field(pytree_node=False, default=20)
    RINK_BOTTOM: int = struct.field(pytree_node=False, default=190)

    # ── Goals  (player = bottom; enemy = top) ────────────────────────────
    GOAL_X0:       int = struct.field(pytree_node=False, default=60)   # left  edge of both goals
    GOAL_X1:       int = struct.field(pytree_node=False, default=100)  # right edge of both goals
    ENEMY_GOAL_Y:  int = struct.field(pytree_node=False, default=20)   # top row of enemy goal
    PLAYER_GOAL_Y: int = struct.field(pytree_node=False, default=187)  # top row of player goal
    GOAL_HEIGHT:   int = struct.field(pytree_node=False, default=7)

    # ── Sprite sizes  (used for observation bounding-boxes) ──────────────
    PLAYER_W: int = struct.field(pytree_node=False, default=8)
    PLAYER_H: int = struct.field(pytree_node=False, default=12)
    PUCK_W:   int = struct.field(pytree_node=False, default=4)
    PUCK_H:   int = struct.field(pytree_node=False, default=3)

    # ── Movement ──────────────────────────────────────────────────────────
    PLAYER_SPEED:      float = struct.field(pytree_node=False, default=1.5)
    PUCK_SPEED:        float = struct.field(pytree_node=False, default=3.0)
    MAX_PUCK_SPEED:    float = struct.field(pytree_node=False, default=5.0)
    PUCK_SPEED_DECAY:  float = struct.field(pytree_node=False, default=0.98)
    POSSESSION_RADIUS: float = struct.field(pytree_node=False, default=9.0)

    # ── Shooting ──────────────────────────────────────────────────────────
    MAX_SHOOTING_ANGLE:    int   = struct.field(pytree_node=False, default=32)
    MIN_SHOOTING_INTERVAL: int   = struct.field(pytree_node=False, default=20)
    MIN_VERTICAL_DISTANCE: float = struct.field(pytree_node=False, default=5.0)

    # ── Tackling ──────────────────────────────────────────────────────────
    MAX_PUSH_DISTANCE: float = struct.field(pytree_node=False, default=20.0)
    FRAMES_TACKLED:    int   = struct.field(pytree_node=False, default=60)

    # ── Timing ────────────────────────────────────────────────────────────
    # ALE runs at 60 fps; 3 min × 60 s × 60 fps = 10 800 raw frames.
    # With default frameskip=4 that is 2 700 agent steps – we track raw frames.
    TIME_LIMIT:        int = struct.field(pytree_node=False, default=10800)
    FACE_OFF_FRAMES:   int = struct.field(pytree_node=False, default=40)
    GOAL_PAUSE_FRAMES: int = struct.field(pytree_node=False, default=90)

    # ── Face-off starting positions  [x=col, y=row] ───────────────────────
    # Verified against ALE game layout (approximate pixel positions):
    #   enemy half = rows 20-103, player half = rows 103-190
    FACEOFF_X: float = struct.field(pytree_node=False, default=78.0)
    FACEOFF_Y: float = struct.field(pytree_node=False, default=103.0)
    # Player team (bottom, orange)
    P1_X: float = struct.field(pytree_node=False, default=60.0)
    P1_Y: float = struct.field(pytree_node=False, default=115.0)
    P2_X: float = struct.field(pytree_node=False, default=85.0)
    P2_Y: float = struct.field(pytree_node=False, default=150.0)
    # Enemy team (top, blue/white)
    E1_X: float = struct.field(pytree_node=False, default=85.0)
    E1_Y: float = struct.field(pytree_node=False, default=89.0)
    E2_X: float = struct.field(pytree_node=False, default=60.0)
    E2_Y: float = struct.field(pytree_node=False, default=54.0)

    # ── Colour hints (used by sprite generator; can be overridden by mods) ─
    ICE_COLOR:    Tuple[int, int, int] = struct.field(pytree_node=False, default=(167, 222, 186))
    PLAYER_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(92,  186,  92))
    ENEMY_COLOR:  Tuple[int, int, int] = struct.field(pytree_node=False, default=(213, 130,  74))
    PUCK_COLOR:   Tuple[int, int, int] = struct.field(pytree_node=False, default=(20,   20,  20))
    SCORE_COLOR:  Tuple[int, int, int] = struct.field(pytree_node=False, default=(236, 236, 236))

    # ── Asset manifest ─────────────────────────────────────────────────────
    # Baked into constants so the JAXAtari modding framework (JaxAtariModController)
    # can intercept asset_overrides *before* the renderer's __init__ is called.
    ASSET_CONFIG: tuple = struct.field(
        pytree_node=False,
        default_factory=_get_default_asset_config,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  State dataclasses
#  All use @struct.dataclass which is Flax's decorator-based PyTreeNode.
#  JAX can trace through all fields automatically.
# ══════════════════════════════════════════════════════════════════════════════

@struct.dataclass
class GameState:
    pause_counter:  chex.Array   # frames left in freeze (face-off / goal celebration)
    player_score:   chex.Array
    enemy_score:    chex.Array
    remaining_time: chex.Array   # counts DOWN from TIME_LIMIT → 0
    is_faceoff:     chex.Array   # True → players frozen, puck at centre
    goal_scored:    chex.Array   # True on the single frame a goal was just scored
    is_finished:    chex.Array   # True when remaining_time == 0


@struct.dataclass
class CharacterState:
    is_tackled:        chex.Array   # True while stun is active
    position:          chex.Array   # float32 [x, y]  (col, row)
    orientation:       chex.Array   # int32: 0=left, 1=right
    has_puck:          chex.Array   # True if carrying the puck
    shooting_cooldown: chex.Array   # frames until next shot allowed


@struct.dataclass
class PuckState:
    position:       chex.Array   # float32 [x, y]
    velocity:       chex.Array   # float32 [vx, vy]
    direction:      chex.Array   # int32 slot 0-31  (last shot angle)
    position_stick: chex.Array   # int32 slot 0-31  (position on stick arc when carried)


@struct.dataclass
class AnimatorState:
    player_frame:           chex.Array
    enemy_frame:            chex.Array
    player_stick_frame:     chex.Array
    player_stick_animation: chex.Array
    enemy_stick_frame:      chex.Array
    enemy_stick_animation:  chex.Array


@struct.dataclass
class PlayerState:
    player1:          CharacterState
    player2:          CharacterState
    active_character: chex.Array   # 0 = player1 has control, 1 = player2


@struct.dataclass
class EnemyState:
    enemy1:           CharacterState
    enemy2:           CharacterState
    active_character: chex.Array   # 0 = enemy1 is AI-active, 1 = enemy2


@struct.dataclass
class IceHockeyState:
    player_state:   PlayerState
    enemy_state:    EnemyState
    puck_state:     PuckState
    counter:        chex.Array    # global step counter
    animator_state: AnimatorState
    game_state:     GameState


# ══════════════════════════════════════════════════════════════════════════════
#  Observation and Info
# ══════════════════════════════════════════════════════════════════════════════

@struct.dataclass
class IceHockeyInfo:
    """Auxiliary data returned by step() – not used for training."""
    player_score:   chex.Array
    enemy_score:    chex.Array
    remaining_time: chex.Array


@struct.dataclass
class IceHockeyObservation:
    """
    Object-centric observation for the RL agent.
    Contains everything an agent needs to play the game.
    """
    player1:        ObjectObservation   # active player skater
    player2:        ObjectObservation   # second player skater
    enemy1:         ObjectObservation   # active enemy skater
    enemy2:         ObjectObservation   # second enemy skater
    puck:           ObjectObservation
    player_score:   chex.Array
    enemy_score:    chex.Array
    remaining_time: chex.Array
    active_player:  chex.Array   # 0 or 1 – which player the human controls


# ══════════════════════════════════════════════════════════════════════════════
#  Main environment class
# ══════════════════════════════════════════════════════════════════════════════

class JaxIceHockey(JaxEnvironment):
    """
    JAX reimplementation of ALE/IceHockey-v5.

    ALE game: single-player (you vs AI), 2v2 top-down ice hockey,
    3-minute match, full 18-action set, 210×160 screen.
    """

    # ── Action set ────────────────────────────────────────────────────────
    # IceHockey uses every ALE action (ALE doc: "full_action_space=True will
    # not modify the action space").  ACTION_SET[i] is the ALE action integer
    # the agent produces when it outputs index i.
    ACTION_SET: jnp.ndarray = jnp.array(
        [
            Action.NOOP,       Action.FIRE,          Action.UP,
            Action.RIGHT,      Action.LEFT,           Action.DOWN,
            Action.UPRIGHT,    Action.UPLEFT,         Action.DOWNRIGHT,
            Action.DOWNLEFT,   Action.UPFIRE,         Action.RIGHTFIRE,
            Action.LEFTFIRE,   Action.DOWNFIRE,       Action.UPRIGHTFIRE,
            Action.UPLEFTFIRE, Action.DOWNRIGHTFIRE,  Action.DOWNLEFTFIRE,
        ],
        dtype=jnp.int32,
    )

    # ── Constructor ───────────────────────────────────────────────────────

    def __init__(self, consts: Optional[IceHockeyConstants] = None):
        consts = consts or IceHockeyConstants()
        super().__init__(consts)
        self.renderer = IceHockeyRenderer(self.consts)

    # ── Spaces ────────────────────────────────────────────────────────────

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTION_SET))

    def observation_space(self) -> spaces.Dict:
        obj = spaces.get_object_space(
            n=None, screen_size=(self.consts.HEIGHT, self.consts.WIDTH)
        )
        return spaces.Dict(
            {
                "player1":        obj,
                "player2":        obj,
                "enemy1":         obj,
                "enemy2":         obj,
                "puck":           obj,
                "player_score":   spaces.Box(0, 99,                       shape=(), dtype=jnp.int32),
                "enemy_score":    spaces.Box(0, 99,                       shape=(), dtype=jnp.int32),
                "remaining_time": spaces.Box(0, self.consts.TIME_LIMIT,   shape=(), dtype=jnp.int32),
                "active_player":  spaces.Box(0, 1, shape=(), dtype=jnp.int32),
            }
        )

    def image_space(self) -> spaces.Box:
        return spaces.Box(low=0, high=255, shape=(210, 160, 3), dtype=jnp.uint8)

    # ── Reset ─────────────────────────────────────────────────────────────

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey = None) -> Tuple:
        """
        Standard ALE face-off layout:
          • Puck at centre ice.
          • Each team's "active" skater lines up near the face-off dot;
            the second skater is in their defensive zone.
          • All characters frozen for FACE_OFF_FRAMES frames.
        """
        key = key if key is not None else jrandom.PRNGKey(0)
        c = self.consts

        def _char(x: float, y: float) -> CharacterState:
            return CharacterState(
                is_tackled=jnp.array(False),
                position=jnp.array([x, y], dtype=jnp.float32),
                orientation=jnp.array(0, dtype=jnp.int32),
                has_puck=jnp.array(False),
                shooting_cooldown=jnp.array(0, dtype=jnp.int32),
            )

        state = IceHockeyState(
            player_state=PlayerState(
                player1=_char(c.P1_X, c.P1_Y),
                player2=_char(c.P2_X, c.P2_Y),
                active_character=jnp.array(0, dtype=jnp.int32),
            ),
            enemy_state=EnemyState(
                enemy1=_char(c.E1_X, c.E1_Y),
                enemy2=_char(c.E2_X, c.E2_Y),
                active_character=jnp.array(0, dtype=jnp.int32),
            ),
            puck_state=PuckState(
                position=jnp.array([c.FACEOFF_X, c.FACEOFF_Y], dtype=jnp.float32),
                velocity=jnp.array([0.0, 0.0],                  dtype=jnp.float32),
                direction=jnp.array(0,                           dtype=jnp.int32),
                position_stick=jnp.array(0,                      dtype=jnp.int32),
            ),
            counter=jnp.array(0, dtype=jnp.int32),
            animator_state=AnimatorState(
                player_frame=jnp.array(0,           dtype=jnp.int32),
                enemy_frame=jnp.array(0,            dtype=jnp.int32),
                player_stick_frame=jnp.array(0,     dtype=jnp.int32),
                player_stick_animation=jnp.array(0, dtype=jnp.int32),
                enemy_stick_frame=jnp.array(0,      dtype=jnp.int32),
                enemy_stick_animation=jnp.array(0,  dtype=jnp.int32),
            ),
            game_state=GameState(
                pause_counter=jnp.array(c.FACE_OFF_FRAMES, dtype=jnp.int32),
                player_score=jnp.array(0,                  dtype=jnp.int32),
                enemy_score=jnp.array(0,                   dtype=jnp.int32),
                remaining_time=jnp.array(c.TIME_LIMIT,     dtype=jnp.int32),
                is_faceoff=jnp.array(True),
                goal_scored=jnp.array(False),
                is_finished=jnp.array(False),
            ),
        )

        return self._get_observation(state), state

    # ── Step  (game logic stub – implemented in Task 2.x) ─────────────────

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: IceHockeyState, action):
        """
        Placeholder that returns unchanged state with correct return signature.
        Implement  _player_step / _enemy_step / _puck_step / _collision_step /
        _goal_check / _timer_step  in Task 2.x, then wire them here.
        """
        # TODO Task 2.x: translate action index → ALE action integer
        # atari_action = jnp.take(self.ACTION_SET, action.astype(jnp.int32))
        # state = self._player_step(state, atari_action)
        # state = self._enemy_step(state)
        # state = self._puck_step(state, atari_action)
        # state = self._collision_step(state)
        # state = self._goal_check(state)
        # state = self._timer_step(state)

        previous_state = state
        done   = self._get_done(state)
        reward = self._get_reward(previous_state, state)
        info   = self._get_info(state)
        obs    = self._get_observation(state)
        return obs, state, reward, done, info

    # ── Render ────────────────────────────────────────────────────────────

    def render(self, state: IceHockeyState) -> jnp.ndarray:
        return self.renderer.render(state)

    # ── Internal JIT helpers ──────────────────────────────────────────────

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: IceHockeyState) -> IceHockeyObservation:
        c = self.consts

        def _obj(pos, w: int, h: int) -> ObjectObservation:
            return ObjectObservation.create(
                x=pos[0].astype(jnp.int32),
                y=pos[1].astype(jnp.int32),
                width=jnp.array(w,  dtype=jnp.int32),
                height=jnp.array(h, dtype=jnp.int32),
            )

        return IceHockeyObservation(
            player1=_obj(state.player_state.player1.position, c.PLAYER_W, c.PLAYER_H),
            player2=_obj(state.player_state.player2.position, c.PLAYER_W, c.PLAYER_H),
            enemy1= _obj(state.enemy_state.enemy1.position,   c.PLAYER_W, c.PLAYER_H),
            enemy2= _obj(state.enemy_state.enemy2.position,   c.PLAYER_W, c.PLAYER_H),
            puck=   _obj(state.puck_state.position,           c.PUCK_W,   c.PUCK_H),
            player_score=state.game_state.player_score,
            enemy_score=state.game_state.enemy_score,
            remaining_time=state.game_state.remaining_time,
            active_player=state.player_state.active_character,
        )

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: IceHockeyObservation) -> jnp.ndarray:
        """Flattens the structured observation to a 1-D float32 array (29 elements)."""
        def _flat(o: ObjectObservation) -> jnp.ndarray:
            return jnp.array(
                [o.x, o.y, o.width, o.height, o.active], dtype=jnp.float32
            )

        return jnp.concatenate(
            [
                _flat(obs.player1), _flat(obs.player2),
                _flat(obs.enemy1),  _flat(obs.enemy2),
                _flat(obs.puck),
                jnp.array(
                    [obs.player_score, obs.enemy_score,
                     obs.remaining_time, obs.active_player],
                    dtype=jnp.float32,
                ),
            ]
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: IceHockeyState) -> IceHockeyInfo:
        return IceHockeyInfo(
            player_score=state.game_state.player_score,
            enemy_score=state.game_state.enemy_score,
            remaining_time=state.game_state.remaining_time,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(
        self,
        previous_state: IceHockeyState,
        state: IceHockeyState,
    ) -> chex.Array:
        """
        Reward = change in (player_score - enemy_score).
        +1 per goal scored by player, -1 per goal conceded.
        """
        delta = (
            state.game_state.player_score - state.game_state.enemy_score
        ) - (
            previous_state.game_state.player_score
            - previous_state.game_state.enemy_score
        )
        return delta.astype(jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: IceHockeyState) -> chex.Array:
        """Episode ends when the 3-minute timer reaches zero."""
        return state.game_state.is_finished


# ══════════════════════════════════════════════════════════════════════════════
#  Renderer
# ══════════════════════════════════════════════════════════════════════════════

class IceHockeyRenderer(JAXGameRenderer):
    """
    Palette-based renderer for IceHockey.

    Two-stage pipeline (per JAXAtari design guide):
      1. Object raster planning – stamp integer colour-IDs onto a 2-D raster.
      2. PALETTE[raster] lookup  – convert ID raster → RGB image in one shot.

    The static background (rink geometry, boards, lines, goal creases) is
    pre-baked into  self.BACKGROUND  during __init__.  The per-frame render()
    method only needs to stamp the four skaters, the puck, and the score UI.
    """

    def __init__(self, consts: Optional[IceHockeyConstants] = None):
        self.consts = consts or IceHockeyConstants()
        super().__init__(self.consts)

        self.config = render_utils.RendererConfig(
            game_dimensions=(210, 160),
            channels=3,
            downscale=None,
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)

        # ── Sprite path: branch-local folder (per TA advice) ─────────────
        # Replace later with  render_utils.get_base_sprite_dir() / "icehockey"
        # when sprites are merged into the central sprites folder.
        self.sprite_path = os.path.join(
            os.path.dirname(__file__), "sprites", "icehockey"
        )

        # ── Build final asset list from constants ─────────────────────────
        # Start from the constant manifest so the modding framework can
        # override assets before we load anything.
        final_asset_config = list(self.consts.ASSET_CONFIG)

        # Load everything in one call: palette, shape masks, background raster,
        # colour→ID dict, and flip offsets for all registered sprites.
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS,
        ) = self.jr.load_and_setup_assets(final_asset_config, self.sprite_path)

    # ── Per-frame render ──────────────────────────────────────────────────

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: IceHockeyState) -> jnp.ndarray:
        """
        Renders one frame of IceHockey.

        Draw order (painter's algorithm, last drawn is on top):
          1. Background raster (rink, boards, lines, goals – static).
          2. Player skater 2  (inactive / defender).
          3. Enemy skater 2   (inactive / defender).
          4. Player skater 1  (active / near puck).
          5. Enemy skater 1   (active / near puck).
          6. Puck.
          7. Score UI (digits in score bars).
        """
        # ── 1. Start from the pre-baked rink background ───────────────────
        raster = self.jr.create_object_raster(self.BACKGROUND)

        # ── Pre-load masks (Python lookups happen at trace time only) ──────
        pm     = self.SHAPE_MASKS["player"]
        em     = self.SHAPE_MASKS["enemy"]
        puck_m = self.SHAPE_MASKS["puck"]

        # ── 2-5. Skaters ─────────────────────────────────────────────────
        # Helpers: convert float32 position array → int32 col / row.
        # These lambdas are inlined at JAX trace time – no Python overhead.
        def _x(pos):
            return jnp.round(pos[0]).astype(jnp.int32)

        def _y(pos):
            return jnp.round(pos[1]).astype(jnp.int32)

        p1 = state.player_state.player1.position
        p2 = state.player_state.player2.position
        e1 = state.enemy_state.enemy1.position
        e2 = state.enemy_state.enemy2.position

        # render_at_clipped is preferred here because skaters can approach
        # the board pixels at the raster boundary; render_at uses
        # dynamic_slice which will raise if given an out-of-bounds index.
        raster = self.jr.render_at_clipped(raster, _x(p2), _y(p2), pm)
        raster = self.jr.render_at_clipped(raster, _x(e2), _y(e2), em)
        raster = self.jr.render_at_clipped(raster, _x(p1), _y(p1), pm)
        raster = self.jr.render_at_clipped(raster, _x(e1), _y(e1), em)

        # ── 6. Puck ───────────────────────────────────────────────────────
        pp = state.puck_state.position
        raster = self.jr.render_at_clipped(raster, _x(pp), _y(pp), puck_m)

        # ── 7. Score UI ───────────────────────────────────────────────────
        # Digits live in the black score bars (rows 0-15).
        # Layout: enemy score on the left, player score on the right
        # (matches ALE IceHockey display).
        dm = self.SHAPE_MASKS["digits"]   # shape: (10, digit_H, digit_W)

        def _score(r, score_val, x_single: int, x_double: int):
            """Renders a 1- or 2-digit score at the correct x position."""
            digits    = self.jr.int_to_digits(score_val, max_digits=2)
            is_single = score_val < 10
            # start_index / num_to_render can be JAX-traced values (not static)
            start = jax.lax.select(is_single, jnp.int32(1), jnp.int32(0))
            count = jax.lax.select(is_single, jnp.int32(1), jnp.int32(2))
            x     = jax.lax.select(
                is_single,
                jnp.int32(x_single),
                jnp.int32(x_double),
            )
            # spacing=7 and max_digits_to_render=2 must be Python ints
            # (they are static_argnames in render_label_selective).
            return self.jr.render_label_selective(
                r, x, 3, digits, dm, start, count,
                spacing=7,
                max_digits_to_render=2,
            )

        raster = _score(raster, state.game_state.enemy_score,  43, 33)
        raster = _score(raster, state.game_state.player_score, 113, 103)

        # ── Final palette lookup → RGB image ──────────────────────────────
        return self.jr.render_from_palette(raster, self.PALETTE)