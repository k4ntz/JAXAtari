import os
from functools import partial
from typing import Tuple, Optional
import jax
import jax.numpy as jnp
import chex
from flax import struct
import jaxatari.rendering.jax_rendering_utils as render_utils
import jaxatari.spaces as spaces
from jaxatari.environment import (
    JaxEnvironment,
    JAXAtariAction as Action,
    ObjectObservation,
)
from jaxatari.renderers import JAXGameRenderer


def _get_default_asset_config() -> tuple:
    """Manifest of the .npy sprites the renderer loads from sprites/icehockey/.

    Run scripts/make_icehockey_sprites.py once to create the placeholder files.
    """
    return (
        {"name": "background", "type": "background", "file": "background.npy"},
        {
            "name": "player_walking_left",
            "type": "group",
            "files": [
                "player_walking_left_0.npy",
                "player_walking_left_1.npy",
                "player_walking_left_2.npy",
                "player_walking_left_3.npy",
            ],
        },
        {
            "name": "player_walking_right",
            "type": "group",
            "files": [
                "player_walking_right_0.npy",
                "player_walking_right_1.npy",
                "player_walking_right_2.npy",
                "player_walking_right_3.npy",
            ],
        },
        {
            "name": "enemy_walking_left",
            "type": "group",
            "files": [
                "enemy_walking_left_0.npy",
                "enemy_walking_left_1.npy",
                "enemy_walking_left_2.npy",
                "enemy_walking_left_3.npy",
            ],
        },
        {
            "name": "enemy_walking_right",
            "type": "group",
            "files": [
                "enemy_walking_right_0.npy",
                "enemy_walking_right_1.npy",
                "enemy_walking_right_2.npy",
                "enemy_walking_right_3.npy",
            ],
        },
        {"name": "player_idle_left", "type": "single", "file": "player_idle_left.npy"},
        {
            "name": "player_idle_right",
            "type": "single",
            "file": "player_idle_right.npy",
        },
        {"name": "enemy_idle_left", "type": "single", "file": "enemy_idle_left.npy"},
        {"name": "enemy_idle_right", "type": "single", "file": "enemy_idle_right.npy"},
        {
            "name": "player_active_standing_left",
            "type": "single",
            "file": "player_active_standing_left.npy",
        },
        {
            "name": "player_active_standing_right",
            "type": "single",
            "file": "player_active_standing_right.npy",
        },
        {
            "name": "enemy_active_standing_left",
            "type": "single",
            "file": "enemy_active_standing_left.npy",
        },
        {
            "name": "enemy_active_standing_right",
            "type": "single",
            "file": "enemy_active_standing_right.npy",
        },
        {"name": "player_faceoff", "type": "single", "file": "player_faceoff.npy"},
        {"name": "enemy_faceoff", "type": "single", "file": "enemy_faceoff.npy"},
        {"name": "player_tackled", "type": "single", "file": "player_tackled.npy"},
        {"name": "enemy_tackled", "type": "single", "file": "enemy_tackled.npy"},
        {
            "name": "player_shooting_right",
            "type": "group",
            "files": [
                "player_shooting_right_0.npy",
                "player_shooting_right_1.npy",
                "player_shooting_right_2.npy",
                "player_shooting_right_3.npy",
                "player_shooting_right_4.npy",
            ],
        },
        {
            "name": "player_shooting_left",
            "type": "group",
            "files": [
                "player_shooting_left_0.npy",
                "player_shooting_left_1.npy",
                "player_shooting_left_2.npy",
                "player_shooting_left_3.npy",
                "player_shooting_left_4.npy",
            ],
        },
        {
            "name": "enemy_shooting_right",
            "type": "group",
            "files": [
                "enemy_shooting_right_0.npy",
                "enemy_shooting_right_1.npy",
                "enemy_shooting_right_2.npy",
                "enemy_shooting_right_3.npy",
                "enemy_shooting_right_4.npy",
            ],
        },
        {
            "name": "enemy_shooting_left",
            "type": "group",
            "files": [
                "enemy_shooting_left_0.npy",
                "enemy_shooting_left_1.npy",
                "enemy_shooting_left_2.npy",
                "enemy_shooting_left_3.npy",
                "enemy_shooting_left_4.npy",
            ],
        },
        {"name": "puck", "type": "single", "file": "puck.npy"},
        {
            "name": "digits",
            "type": "digits",
            "pattern": "digit_{}.npy",
            "recolorings": {"gold": (236, 200, 96)},
        },
    )


class IceHockeyConstants(struct.PyTreeNode):
    DEBUG_RENDER: bool = struct.field(pytree_node=False, default=False)

    # Static parameters. Marked pytree_node=False so JAX keeps them as static
    # metadata instead of tracing them.
    WIDTH: int = struct.field(pytree_node=False, default=160)
    HEIGHT: int = struct.field(pytree_node=False, default=210)

    # Rink interior in pixels (inside the boards)
    RINK_LEFT: int = struct.field(pytree_node=False, default=32)
    RINK_RIGHT: int = struct.field(pytree_node=False, default=128)
    RINK_TOP: int = struct.field(pytree_node=False, default=42)
    RINK_BOTTOM: int = struct.field(pytree_node=False, default=187)

    # Goals. Player defends the top, enemy the bottom.
    GOAL_X0: int = struct.field(pytree_node=False, default=64)
    GOAL_X1: int = struct.field(pytree_node=False, default=96)
    ENEMY_GOAL_Y: int = struct.field(pytree_node=False, default=186)  # Bottom goal line
    PLAYER_GOAL_Y: int = struct.field(pytree_node=False, default=42)  # top goal line
    GOAL_HEIGHT_TOP: int = struct.field(pytree_node=False, default=4)
    GOAL_HEIGHT_BOTTOM: int = struct.field(pytree_node=False, default=3)

    # Sprite sizes, used for observation bounding boxes
    PLAYER_W: int = struct.field(pytree_node=False, default=26)
    PLAYER_H: int = struct.field(pytree_node=False, default=26)
    PUCK_W: int = struct.field(pytree_node=False, default=2)
    PUCK_H: int = struct.field(pytree_node=False, default=2)

    # Character positions advance on an integer gameplay grid: one pixel
    # horizontally or two pixels vertically per character update.
    CHARACTER_SPEED_X: float = struct.field(pytree_node=False, default=1.0)
    CHARACTER_SPEED_Y: float = struct.field(pytree_node=False, default=2.0)
    # Characters and a free puck advance on separate fixed cadences.
    # The game clock still advances every frame.
    CHARACTER_UPDATE_CADENCE: int = struct.field(pytree_node=False, default=4)
    FREE_PUCK_UPDATE_CADENCE: int = struct.field(pytree_node=False, default=2)

    # Skater leg walk-cycle: number of frames in the loop and how many game
    # frames each phase is shown for. The cycle advances only while a skater has
    # directional input.
    ANIM_CADENCE: int = struct.field(pytree_node=False, default=4)

    # Swing state advances every character cadence: 0 = idle, 1..7 = active swing.
    SWING_PHASE_COUNT: int = struct.field(pytree_node=False, default=8)

    # Exact vertical movement ranges in the game's integer character grid.
    # Player goalie / enemy skater occupy the upper range; player skater / enemy
    # goalie occupy the lower range.
    UPPER_CHARACTER_GRID_Y_MIN: int = struct.field(pytree_node=False, default=55)
    UPPER_CHARACTER_GRID_Y_MAX: int = struct.field(pytree_node=False, default=139)
    LOWER_CHARACTER_GRID_Y_MIN: int = struct.field(pytree_node=False, default=9)
    LOWER_CHARACTER_GRID_Y_MAX: int = struct.field(pytree_node=False, default=91)
    # Character contact box and fixed separation impulse.
    CONTACT_X_MIN: float = struct.field(pytree_node=False, default=-8.0)
    CONTACT_X_MAX: float = struct.field(pytree_node=False, default=7.0)
    CONTACT_Y_MIN: float = struct.field(pytree_node=False, default=-3.0)
    CONTACT_Y_MAX: float = struct.field(pytree_node=False, default=5.0)
    CONTACT_PUSH_X: float = struct.field(pytree_node=False, default=1.0)
    CONTACT_PUSH_Y: float = struct.field(pytree_node=False, default=2.0)
    # Minimum vertical gap between two same-team players (and between the two goalies
    # of opposing teams), ~30% of the rink, measured top-left corner to top-left
    # corner. The passive player is pushed vertically to keep this gap.
    MIN_VERTICAL_DISTANCE: float = struct.field(pytree_node=False, default=44.0)

    STICK_VISIBLE_CADENCE: int = struct.field(pytree_node=False, default=4)

    # Knockdown timers advance at a slower cadence than the main game loop.
    TACKLE_TIMER_CADENCE: int = struct.field(pytree_node=False, default=8)
    PLAYER_GOALIE_PROTECTED_GRID_Y: int = struct.field(pytree_node=False, default=129)
    ENEMY_GOALIE_PROTECTED_GRID_Y: int = struct.field(pytree_node=False, default=19)

    # 3 min * 60 s * 60 fps = 10800 frames, in frames of active play
    TIME_LIMIT: int = struct.field(pytree_node=False, default=10800)
    # freeze duration of the face-off countdown (after reset and after goals)
    FACE_OFF_FRAMES: int = struct.field(pytree_node=False, default=63)
    # freeze after a goal before everyone snaps back to the face-off spots
    GOAL_PAUSE_FRAMES: int = struct.field(pytree_node=False, default=64)

    # Integer coordinate grid used by the controller and puck-contact rules.
    # These values are fixed game mechanics rather than tunable AI parameters.
    PUCK_GRID_X_OFFSET: int = struct.field(pytree_node=False, default=10)
    CHARACTER_GRID_X_OFFSET: int = struct.field(pytree_node=False, default=17)
    PUCK_GRID_Y_ORIGIN: int = struct.field(pytree_node=False, default=188)
    CHARACTER_GRID_Y_ORIGIN: int = struct.field(pytree_node=False, default=166)
    ACTIVE_TOP_THRESHOLD: int = struct.field(pytree_node=False, default=95)
    ACTIVE_BOTTOM_THRESHOLD: int = struct.field(pytree_node=False, default=55)
    GOAL_CENTER_GRID_X: int = struct.field(pytree_node=False, default=89)
    PLAYER_GOAL_GRID_Y: int = struct.field(pytree_node=False, default=146)
    REGRAB_BLOCK_FRAMES: int = struct.field(pytree_node=False, default=48)

    # Face-off layout. [x, y] = [col, row].
    FACEOFF_X: float = struct.field(pytree_node=False, default=79.0)
    FACEOFF_Y: float = struct.field(pytree_node=False, default=114.0)
    PLAYER_SKATER_X: float = struct.field(pytree_node=False, default=52.0)
    PLAYER_SKATER_Y: float = struct.field(pytree_node=False, default=81.0)
    PLAYER_GOALIE_X: float = struct.field(pytree_node=False, default=61.0)
    PLAYER_GOALIE_Y: float = struct.field(pytree_node=False, default=33.0)
    ENEMY_SKATER_X: float = struct.field(pytree_node=False, default=83.0)
    ENEMY_SKATER_Y: float = struct.field(pytree_node=False, default=103.0)
    ENEMY_GOALIE_X: float = struct.field(pytree_node=False, default=63.0)
    ENEMY_GOALIE_Y: float = struct.field(pytree_node=False, default=152.0)

    #  Puck wandert über 32 Slots am Stock hin und her.
    STICK_SLOTS: int = struct.field(pytree_node=False, default=32)
    STICK_CADENCE: int = struct.field(pytree_node=False, default=1)  # Frames pro Slot
    # Stock-Endpunkte relativ zum Box-MITTELPUNKT, autoriert für Blick nach rechts.
    # # Slot 0 = an den Händen (Basis), Slot 31 = Schlägerspitze.
    STICK_MIN_DX: float = struct.field(pytree_node=False, default=5.0)  # Slot 0
    STICK_MAX_DX: float = struct.field(pytree_node=False, default=12.0)  # Slot 31
    # Vertical offset of an attached puck from the character's top-left position.
    PLAYER_CARRIED_PUCK_OFFSET_Y: float = struct.field(pytree_node=False, default=22.0)
    ENEMY_CARRIED_PUCK_OFFSET_Y: float = struct.field(pytree_node=False, default=20.0)

    PUCK_MAX_SPEED: float = struct.field(pytree_node=False, default=2.0)
    # Pick up region around the end of the stick in which the puck can be picked up
    PICKUP_BOX_W: float = struct.field(pytree_node=False, default=16.0)
    PICKUP_BOX_H: float = struct.field(pytree_node=False, default=4.0)
    PLAYER_PICKUP_BOX_OFFSET_Y: float = struct.field(pytree_node=False, default=19.0)
    ENEMY_PICKUP_BOX_OFFSET_Y: float = struct.field(pytree_node=False, default=19.0)
    PICKUP_BOX_OFFSET_X_LEFT: float = struct.field(pytree_node=False, default=0.0)
    PICKUP_BOX_OFFSET_X_RIGHT: float = struct.field(pytree_node=False, default=9.0)

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
    position: chex.Array  # float32 [x, y]
    velocity: chex.Array  # float32 [vx, vy]; unused by the base game, available to mods
    orientation: chex.Array  # 0 = left, 1 = right
    has_puck: chex.Array
    swing_phase: chex.Array
    walk_counter: chex.Array  # leg walk-cycle phase counter (advances while moving)
    tackle_timer: chex.Array  # slow-cadence ticks left; is_tackled == (tackle_timer > 0)
    times_tackled: (
        chex.Array
    )  # knockdowns suffered this match; unused by the base game, read by mods


@struct.dataclass
class PuckState:
    position: chex.Array  # float32 [x, y]
    velocity: chex.Array  # float32 [vx, vy]
    position_stick: chex.Array  # slot on the stick arc while carried, 0-31
    # -1 = no blocked shooter; otherwise 0..3 identifies the last shooter.
    pickup_blocker: chex.Array
    pickup_blocker_timer: chex.Array  # active-play ticks left in the re-grab lockout
    carry_timer: chex.Array
    holder: chex.Array


@struct.dataclass
class PlayerState:
    skater: CharacterState
    goalie: CharacterState
    active_character: chex.Array  # 0 = skater controlled, 1 = goalie controlled


@struct.dataclass
class EnemyState:
    skater: CharacterState
    goalie: CharacterState
    active_character: chex.Array


@struct.dataclass
class EnemyControllerState:
    """Persistent state for the computer-controlled team."""

    target_position: chex.Array  # float32 [x, y]
    move_dx: chex.Array  # int32 in {-1, 0, 1}
    move_dy: chex.Array  # int32 in {-1, 0, 1}
    fire: chex.Array  # bool
    random_byte: chex.Array  # int32, low 8 bits used


@struct.dataclass
class IceHockeyState:
    player_state: PlayerState
    enemy_state: EnemyState
    enemy_controller: EnemyControllerState
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
    player_skater: ObjectObservation
    player_goalie: ObjectObservation
    enemy_skater: ObjectObservation
    enemy_goalie: ObjectObservation
    puck: ObjectObservation
    player_score: chex.Array
    enemy_score: chex.Array
    remaining_time: chex.Array
    active_player: chex.Array


class JaxIceHockey(JaxEnvironment):

    # IceHockey uses the full ALE action set, so the agent index maps straight
    # onto the ALE action integer.
    ACTION_SET = jnp.array(
        [
            Action.NOOP,
            Action.FIRE,
            Action.UP,
            Action.RIGHT,
            Action.LEFT,
            Action.DOWN,
            Action.UPRIGHT,
            Action.UPLEFT,
            Action.DOWNRIGHT,
            Action.DOWNLEFT,
            Action.UPFIRE,
            Action.RIGHTFIRE,
            Action.LEFTFIRE,
            Action.DOWNFIRE,
            Action.UPRIGHTFIRE,
            Action.UPLEFTFIRE,
            Action.DOWNRIGHTFIRE,
            Action.DOWNLEFTFIRE,
        ],
        dtype=jnp.int32,
    )

    def __init__(self, consts: Optional[IceHockeyConstants] = None):
        consts = consts or IceHockeyConstants()
        super().__init__(consts)
        self.renderer = IceHockeyRenderer(self.consts)

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTION_SET))

    def observation_space(self) -> spaces.Dict:
        obj = spaces.get_object_space(
            n=None, screen_size=(self.consts.HEIGHT, self.consts.WIDTH)
        )
        return spaces.Dict(
            {
                "player_skater": obj,
                "player_goalie": obj,
                "enemy_skater": obj,
                "enemy_goalie": obj,
                "puck": obj,
                "player_score": spaces.Box(0, 99, shape=(), dtype=jnp.int32),
                "enemy_score": spaces.Box(0, 99, shape=(), dtype=jnp.int32),
                "remaining_time": spaces.Box(
                    0, self.consts.TIME_LIMIT, shape=(), dtype=jnp.int32
                ),
                "active_player": spaces.Box(0, 1, shape=(), dtype=jnp.int32),
            }
        )

    def image_space(self) -> spaces.Box:
        return spaces.Box(low=0, high=255, shape=(210, 160, 3), dtype=jnp.uint8)

    def _faceoff_launch_velocity(self, random_byte: chex.Array) -> chex.Array:
        """Generate the deterministic face-off puck velocity."""
        r = random_byte.astype(jnp.int32) & jnp.int32(255)

        def component(byte):
            fixed = byte - jnp.where(byte < 128, jnp.int32(256), jnp.int32(0))
            return fixed.astype(jnp.float32) / 256.0

        vx = component((r << 1) & jnp.int32(255))
        vy = -component(r)
        return jnp.array([vx, vy], dtype=jnp.float32)

    def _faceoff_positions(
        self, random_byte: chex.Array
    ) -> Tuple[PlayerState, EnemyState, PuckState]:
        """Fresh character/puck states on the face-off spots (reset + after goals)."""
        c = self.consts

        def char(x, y, orientation):
            return CharacterState(
                is_tackled=jnp.array(False),
                position=jnp.array([x, y], dtype=jnp.float32),
                velocity=jnp.zeros(2, dtype=jnp.float32),
                orientation=jnp.array(orientation, dtype=jnp.int32),
                has_puck=jnp.array(False),
                swing_phase=jnp.array(0, dtype=jnp.int32),
                walk_counter=jnp.array(0, dtype=jnp.int32),
                tackle_timer=jnp.array(0, dtype=jnp.int32),
                times_tackled=jnp.array(0, dtype=jnp.int32),
            )

        player_state = PlayerState(
            skater=char(c.PLAYER_SKATER_X, c.PLAYER_SKATER_Y, orientation=1),
            goalie=char(c.PLAYER_GOALIE_X, c.PLAYER_GOALIE_Y, orientation=1),
            active_character=jnp.array(0, dtype=jnp.int32),
        )
        enemy_state = EnemyState(
            skater=char(c.ENEMY_SKATER_X, c.ENEMY_SKATER_Y, orientation=0),
            goalie=char(c.ENEMY_GOALIE_X, c.ENEMY_GOALIE_Y, orientation=0),
            active_character=jnp.array(0, dtype=jnp.int32),
        )
        puck_state = PuckState(
            position=jnp.array([c.FACEOFF_X, c.FACEOFF_Y], dtype=jnp.float32),
            velocity=self._faceoff_launch_velocity(random_byte),
            position_stick=jnp.array(0, dtype=jnp.int32),
            pickup_blocker=jnp.array(-1, dtype=jnp.int32),
            pickup_blocker_timer=jnp.array(0, dtype=jnp.int32),
            carry_timer=jnp.array(0, dtype=jnp.int32),
            holder=jnp.array(-1, dtype=jnp.int32),
        )
        return player_state, enemy_state, puck_state

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey = None) -> Tuple:
        del key
        # Face-off: puck at centre, characters on start positions
        c = self.consts
        initial_random_byte = jnp.array(0, dtype=jnp.int32)
        player_state, enemy_state, puck_state = self._faceoff_positions(initial_random_byte)

        enemy_controller = EnemyControllerState(
            target_position=enemy_state.skater.position,
            move_dx=jnp.array(0, dtype=jnp.int32),
            move_dy=jnp.array(0, dtype=jnp.int32),
            fire=jnp.array(False),
            random_byte=initial_random_byte,
        )

        state = IceHockeyState(
            player_state=player_state,
            enemy_state=enemy_state,
            enemy_controller=enemy_controller,
            puck_state=puck_state,
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
        previous_state = state
        gs = state.game_state
        frozen = gs.is_finished | gs.goal_scored | gs.is_faceoff

        advanced_controller = state.enemy_controller.replace(
            random_byte=self._next_random_byte(state.enemy_controller.random_byte)
        )

        computer_controlled = not isinstance(action, (tuple, list))
        if computer_controlled:
            player_action = action
            controller_state = state.replace(enemy_controller=advanced_controller)
            policy_action, policy_controller = self._enemy_policy(controller_state)
            new_enemy_controller = jax.lax.cond(
                frozen,
                lambda _: advanced_controller,
                lambda _: policy_controller,
                operand=None,
            )
            enemy_action = jnp.where(frozen, Action.NOOP, policy_action)
        else:
            player_action, enemy_action = action
            new_enemy_controller = advanced_controller

        character_update = (
            state.counter % self.consts.CHARACTER_UPDATE_CADENCE == 0
        )
        free_puck_update = (
            state.counter % self.consts.FREE_PUCK_UPDATE_CADENCE == 0
        )

        player_fire = self._action_has_fire(player_action)
        enemy_fire = self._action_has_fire(enemy_action)

        def play_step(_):
            player_state, enemy_state = jax.lax.cond(
                character_update,
                lambda _: self._move_characters(
                    state.player_state,
                    state.enemy_state,
                    player_action,
                    enemy_action,
                ),
                lambda _: (state.player_state, state.enemy_state),
                operand=None,
            )

            player_state, enemy_state = self._tick_tackle_timers(
                player_state, enemy_state, state.counter
            )
            player_state, enemy_state = self._update_swing_states(
                player_state,
                enemy_state,
                player_action,
                enemy_action,
                state.counter,
            )

            player_state, enemy_state, puck_state, force_enemy_fire = jax.lax.cond(
                character_update,
                lambda _: self._contact_step(
                    player_state,
                    enemy_state,
                    state.puck_state,
                    random_byte=new_enemy_controller.random_byte,
                    counter=state.counter,
                    player_score=gs.player_score,
                    enemy_score=gs.enemy_score,
                ),
                lambda _: (
                    player_state,
                    enemy_state,
                    state.puck_state,
                    jnp.array(False),
                ),
                operand=None,
            )

            controller = new_enemy_controller.replace(
                fire=new_enemy_controller.fire
                | (jnp.array(computer_controlled) & force_enemy_fire)
            )

            active_player, active_enemy = self._resolve_active_characters(
                player_state, enemy_state, puck_state.position
            )
            player_state = player_state.replace(active_character=active_player)
            enemy_state = enemy_state.replace(active_character=active_enemy)
            player_state, enemy_state = self._finalize_character_positions(
                player_state, enemy_state
            )

            player_state, enemy_state, puck_state = self._puck_pickup(
                player_state,
                enemy_state,
                puck_state,
                random_byte=controller.random_byte,
            )
            puck_state = self._puck_carry(
                player_state,
                enemy_state,
                puck_state,
                advance_free_puck=free_puck_update,
            )
            player_state, enemy_state, puck_state = self._puck_shoot(
                player_state,
                enemy_state,
                puck_state,
                player_fire=player_fire,
                enemy_fire=enemy_fire,
                advance_free_puck=free_puck_update,
            )
            return player_state, enemy_state, puck_state, controller

        def hold_step(_):
            return (
                state.player_state,
                state.enemy_state,
                state.puck_state,
                new_enemy_controller,
            )

        (
            new_player_state,
            new_enemy_state,
            new_puck_state,
            new_enemy_controller,
        ) = jax.lax.cond(frozen, hold_step, play_step, None)

        (
            new_player_state,
            new_enemy_state,
            new_puck_state,
            new_game_state,
        ) = self._goal_and_reset_step(
            gs,
            new_player_state,
            new_enemy_state,
            new_puck_state,
            frozen,
            random_byte=new_enemy_controller.random_byte,
        )

        state = state.replace(
            player_state=new_player_state,
            enemy_state=new_enemy_state,
            enemy_controller=new_enemy_controller,
            puck_state=new_puck_state,
            counter=state.counter + 1,
            game_state=new_game_state,
        )

        obs = self._get_observation(state)
        reward = self._get_reward(previous_state, state)
        done = self._get_done(state)
        info = self._get_info(state)
        return obs, state, reward, done, info

    def _next_random_byte(self, value: chex.Array) -> chex.Array:
        """Advance the controller's deterministic 8-bit random sequence."""
        old = value.astype(jnp.int32) & jnp.int32(255)
        feedback = jnp.where(
            old == 0,
            jnp.int32(1),
            ((old >> 7) ^ (old >> 4)) & jnp.int32(1),
        )
        return ((old << 1) & jnp.int32(255)) | feedback

    def _puck_grid_position(self, position: chex.Array) -> chex.Array:
        c = self.consts
        return jnp.array(
            [
                jnp.round(position[0] + c.PUCK_GRID_X_OFFSET),
                jnp.round(c.PUCK_GRID_Y_ORIGIN - position[1]),
            ],
            dtype=jnp.int32,
        ) & jnp.int32(255)

    def _character_grid_position(self, position: chex.Array) -> chex.Array:
        c = self.consts
        return jnp.array(
            [
                jnp.round(position[0] + c.CHARACTER_GRID_X_OFFSET),
                jnp.round(c.CHARACTER_GRID_Y_ORIGIN - position[1]),
            ],
            dtype=jnp.int32,
        ) & jnp.int32(255)

    def _enemy_policy(
        self, state: IceHockeyState
    ) -> Tuple[chex.Array, EnemyControllerState]:
        """Stateful controller for the enemy's active character.

        The controller follows a persistent target and only refreshes that target on
        selected frames. It chases a jittered point around the puck when defending,
        moves toward randomized goalward targets while carrying, and times shots from
        puck distance and stick phase. Direction and fire inputs persist between
        controller updates.
        """
        c = self.consts
        controller = state.enemy_controller
        enemy = state.enemy_state
        player = state.player_state
        puck = state.puck_state

        counter = state.counter.astype(jnp.int32) & jnp.int32(255)
        random_byte = controller.random_byte.astype(jnp.int32) & jnp.int32(255)
        puck_grid = self._puck_grid_position(puck.position)

        control_tick = (counter & jnp.int32(3)) == jnp.int32(2)

        def update_control(current: EnemyControllerState) -> EnemyControllerState:
            active_is_goalie = enemy.active_character == 1
            active = jax.lax.cond(
                active_is_goalie, lambda: enemy.goalie, lambda: enemy.skater
            )

            refresh_mask = jnp.where(
                state.game_state.player_score >= state.game_state.enemy_score,
                jnp.int32(7),
                jnp.int32(31),
            )
            refresh_target = (counter & refresh_mask) == jnp.int32(2)

            player_has_puck = player.skater.has_puck | player.goalie.has_puck
            enemy_has_puck = enemy.skater.has_puck | enemy.goalie.has_puck
            random_3 = random_byte & jnp.int32(7)
            random_5 = random_byte & jnp.int32(31)
            random_6 = random_byte & jnp.int32(63)
            counter_high = (counter & jnp.int32(128)) != 0

            def refresh(_: None):
                previous_target = current.target_position

                def chase_puck(__: None):
                    extra_y = jnp.where(player_has_puck & ~counter_high, 8, 0)
                    target_grid_y_raw = (
                        puck_grid[1] - 4 + random_3 - extra_y
                    ) & jnp.int32(255)
                    y_clamped = target_grid_y_raw >= jnp.int32(c.PLAYER_GOAL_GRID_Y)
                    target_grid_y = jnp.where(
                        y_clamped, jnp.int32(0), target_grid_y_raw
                    )
                    target_grid_x = (
                        puck_grid[0]
                        + random_3
                        + jnp.where(active.orientation == 1, -11, -4)
                        + y_clamped.astype(jnp.int32)
                    ) & jnp.int32(255)
                    target = jnp.array(
                        [
                            target_grid_x - c.CHARACTER_GRID_X_OFFSET,
                            c.CHARACTER_GRID_Y_ORIGIN - target_grid_y,
                        ],
                        dtype=jnp.float32,
                    )
                    return target, jnp.array(False)

                def carry_puck(__: None):
                    goalie_pass = active_is_goalie & counter_high
                    target_grid_x = (
                        jnp.int32(48)
                        + random_6
                        + active_is_goalie.astype(jnp.int32)
                    )

                    player_goalie_down = player.goalie.is_tackled
                    forward_past_goalie = (
                        enemy.skater.position[1] <= player.goalie.position[1]
                    )
                    force_goalward = player_goalie_down | forward_past_goalie
                    shot_distance_carry = (~player_goalie_down) & forward_past_goalie
                    target_grid_y = jnp.where(
                        force_goalward,
                        jnp.int32(c.PLAYER_GOAL_GRID_Y),
                        random_byte | jnp.int32(120),
                    )
                    target = jnp.array(
                        [
                            target_grid_x - c.CHARACTER_GRID_X_OFFSET,
                            c.CHARACTER_GRID_Y_ORIGIN - target_grid_y,
                        ],
                        dtype=jnp.float32,
                    )

                    near_goal = (
                        puck_grid[1]
                        + random_5
                        + shot_distance_carry.astype(jnp.int32)
                    ) >= jnp.int32(c.PLAYER_GOAL_GRID_Y)
                    slot_low_half = puck.position_stick < 16
                    facing_left = enemy.skater.orientation == 0
                    puck_left = puck_grid[0] < jnp.int32(c.GOAL_CENTER_GRID_X)
                    angle_ok = jnp.logical_xor(
                        slot_low_half, jnp.logical_xor(facing_left, puck_left)
                    )
                    normal_shot = near_goal & angle_ok

                    target = jnp.where(goalie_pass, previous_target, target)
                    fire = goalie_pass | normal_shot
                    return target, fire

                return jax.lax.cond(
                    enemy_has_puck, carry_puck, chase_puck, operand=None
                )

            target, fire = jax.lax.cond(
                refresh_target,
                refresh,
                lambda _: (current.target_position, current.fire),
                operand=None,
            )

            px, py = active.position
            tx, ty = target
            dx = jnp.where(
                px < tx,
                jnp.int32(1),
                jnp.where(px > tx, jnp.int32(-1), current.move_dx),
            )
            dy = jnp.where(
                py > ty,
                jnp.int32(-1),
                jnp.where(py < ty, jnp.int32(1), current.move_dy),
            )
            return current.replace(
                target_position=target,
                move_dx=dx,
                move_dy=dy,
                fire=fire,
            )

        controller = jax.lax.cond(
            control_tick, update_control, lambda current: current, controller
        )
        action = self._compose_action(
            controller.move_dx, controller.move_dy, controller.fire
        )
        return action, controller

    def _compose_action(
        self, dx: chex.Array, dy: chex.Array, fire: chex.Array
    ) -> chex.Array:
        """Map (dx, dy in {-1, 0, 1}, fire) onto the matching ALE action integer."""
        table = jnp.array(
            [
                [
                    [Action.UPLEFT, Action.UPLEFTFIRE],
                    [Action.UP, Action.UPFIRE],
                    [Action.UPRIGHT, Action.UPRIGHTFIRE],
                ],
                [
                    [Action.LEFT, Action.LEFTFIRE],
                    [Action.NOOP, Action.FIRE],
                    [Action.RIGHT, Action.RIGHTFIRE],
                ],
                [
                    [Action.DOWNLEFT, Action.DOWNLEFTFIRE],
                    [Action.DOWN, Action.DOWNFIRE],
                    [Action.DOWNRIGHT, Action.DOWNRIGHTFIRE],
                ],
            ],
            dtype=jnp.int32,
        )
        return table[dy + 1, dx + 1, fire.astype(jnp.int32)]

    def _action_has_fire(self, action: chex.Array) -> chex.Array:
        return (action == Action.FIRE) | (action >= Action.UPFIRE)

    def _goal_and_reset_step(
        self,
        game_state: GameState,
        player_state: PlayerState,
        enemy_state: EnemyState,
        puck_state: PuckState,
        frozen: chex.Array,
        random_byte: chex.Array,
    ) -> Tuple[PlayerState, EnemyState, PuckState, GameState]:
        """Detect goals, advance pause phases, and reset to face-off positions."""
        c = self.consts

        # Goal: a free (not carried) puck touching a goal line inside the mouth.
        puck_pos = puck_state.position
        carried = (
            player_state.skater.has_puck
            | player_state.goalie.has_puck
            | enemy_state.skater.has_puck
            | enemy_state.goalie.has_puck
        )
        # GOAL_X0 and GOAL_X1 are rigid post pixels.  Only the open interval
        # between them is a scoring mouth.
        in_mouth = (puck_pos[0] > c.GOAL_X0) & (puck_pos[0] < c.GOAL_X1)
        player_scored = ~frozen & ~carried & in_mouth & (puck_pos[1] >= c.ENEMY_GOAL_Y)
        enemy_scored = ~frozen & ~carried & in_mouth & (puck_pos[1] <= c.PLAYER_GOAL_Y)
        goal = player_scored | enemy_scored

        # Clock only runs during play; a goal rounds it up to the next full
        # second so the tick restarts fresh after the face-off.
        remaining_time = game_state.remaining_time
        remaining_time = jnp.where(
            goal, ((remaining_time + 59) // 60) * 60, remaining_time
        )
        clock_runs = ~frozen & ~goal
        remaining_time = jnp.where(
            clock_runs, jnp.maximum(remaining_time - 1, 0), remaining_time
        )
        time_up = clock_runs & (remaining_time == 0)

        # Pause phases: goal pause -> face-off countdown -> play.
        pause_counter = jnp.where(
            frozen,
            jnp.maximum(game_state.pause_counter - 1, 0),
            game_state.pause_counter,
        )
        goal_phase_over = game_state.goal_scored & (pause_counter == 0)
        faceoff_over = game_state.is_faceoff & (pause_counter == 0)

        new_goal_scored = (game_state.goal_scored & ~goal_phase_over) | goal
        new_is_faceoff = (game_state.is_faceoff & ~faceoff_over) | goal_phase_over
        pause_counter = jnp.where(
            goal,
            jnp.int32(c.GOAL_PAUSE_FRAMES),
            jnp.where(
                goal_phase_over,
                jnp.int32(c.FACE_OFF_FRAMES),
                pause_counter,
            ),
        )

        # After the goal pause everyone snaps back to the face-off spots.
        # times_tackled is a per-match statistic, so the fresh face-off
        # characters inherit it instead of restarting at 0.
        faceoff_random_byte = self._next_random_byte(random_byte)
        fo_player, fo_enemy, fo_puck = self._faceoff_positions(faceoff_random_byte)
        fo_player = fo_player.replace(
            skater=fo_player.skater.replace(
                times_tackled=player_state.skater.times_tackled
            ),
            goalie=fo_player.goalie.replace(
                times_tackled=player_state.goalie.times_tackled
            ),
        )
        fo_enemy = fo_enemy.replace(
            skater=fo_enemy.skater.replace(
                times_tackled=enemy_state.skater.times_tackled
            ),
            goalie=fo_enemy.goalie.replace(
                times_tackled=enemy_state.goalie.times_tackled
            ),
        )
        player_state, enemy_state, puck_state = jax.lax.cond(
            goal_phase_over,
            lambda: (fo_player, fo_enemy, fo_puck),
            lambda: (player_state, enemy_state, puck_state),
        )

        game_state = game_state.replace(
            pause_counter=pause_counter,
            player_score=game_state.player_score + player_scored.astype(jnp.int32),
            enemy_score=game_state.enemy_score + enemy_scored.astype(jnp.int32),
            remaining_time=remaining_time,
            is_faceoff=new_is_faceoff,
            goal_scored=new_goal_scored,
            is_finished=game_state.is_finished | time_up,
        )
        return player_state, enemy_state, puck_state, game_state

    def _tick_tackle_timers(
        self,
        player_state: PlayerState,
        enemy_state: EnemyState,
        counter: chex.Array,
    ) -> Tuple[PlayerState, EnemyState]:
        tick = (counter % self.consts.TACKLE_TIMER_CADENCE) == 0

        def update(char: CharacterState) -> CharacterState:
            timer = jnp.where(
                tick & (char.tackle_timer > 0),
                char.tackle_timer - 1,
                char.tackle_timer,
            )
            return char.replace(tackle_timer=timer, is_tackled=timer > 0)

        return (
            player_state.replace(
                skater=update(player_state.skater),
                goalie=update(player_state.goalie),
            ),
            enemy_state.replace(
                skater=update(enemy_state.skater),
                goalie=update(enemy_state.goalie),
            ),
        )

    def _update_swing_states(
        self,
        player_state: PlayerState,
        enemy_state: EnemyState,
        player_action: chex.Array,
        enemy_action: chex.Array,
        counter: chex.Array,
    ) -> Tuple[PlayerState, EnemyState]:
        def apply_fire(
            skater: CharacterState,
            goalie: CharacterState,
            active: chex.Array,
            fire: chex.Array,
        ):
            def update(char: CharacterState) -> CharacterState:
                phase = jnp.where(
                    fire & char.is_tackled,
                    jnp.int32(0),
                    jnp.where(
                        fire & (char.swing_phase == 0),
                        jnp.int32(1),
                        char.swing_phase,
                    ),
                )
                return char.replace(swing_phase=phase)

            return (
                jax.lax.cond(active == 0, update, lambda ch: ch, skater),
                jax.lax.cond(active == 1, update, lambda ch: ch, goalie),
            )

        p_sk, p_go = apply_fire(
            player_state.skater,
            player_state.goalie,
            player_state.active_character,
            self._action_has_fire(player_action),
        )
        e_sk, e_go = apply_fire(
            enemy_state.skater,
            enemy_state.goalie,
            enemy_state.active_character,
            self._action_has_fire(enemy_action),
        )

        phase_tick = (counter % self.consts.CHARACTER_UPDATE_CADENCE) == 0

        def advance(char: CharacterState) -> CharacterState:
            phase = jnp.where(
                phase_tick & (char.swing_phase > 0),
                char.swing_phase + 1,
                char.swing_phase,
            )
            phase = jnp.where(
                phase >= self.consts.SWING_PHASE_COUNT, jnp.int32(0), phase
            )
            return char.replace(swing_phase=phase)

        return (
            player_state.replace(skater=advance(p_sk), goalie=advance(p_go)),
            enemy_state.replace(skater=advance(e_sk), goalie=advance(e_go)),
        )

    def _characters_touch(
        self, pos_a: chex.Array, pos_b: chex.Array
    ) -> chex.Array:
        c = self.consts
        delta = pos_a - pos_b
        return (
            (delta[0] >= c.CONTACT_X_MIN)
            & (delta[0] <= c.CONTACT_X_MAX)
            & (delta[1] >= c.CONTACT_Y_MIN)
            & (delta[1] <= c.CONTACT_Y_MAX)
        )

    def _knock_down(
        self, char: CharacterState, random_byte: chex.Array
    ) -> CharacterState:
        timer = (random_byte.astype(jnp.int32) & jnp.int32(31)) | jnp.int32(8)
        return char.replace(
            tackle_timer=timer,
            is_tackled=jnp.array(True),
            has_puck=jnp.array(False),
            times_tackled=char.times_tackled + 1,
        )

    def _tackle_drop_velocity(
        self, random_byte: chex.Array, victim_is_second: chex.Array
    ) -> chex.Array:
        r = random_byte.astype(jnp.int32) & jnp.int32(255)

        def signed_byte(value):
            return jnp.where(value < 128, value, value - 256).astype(jnp.float32)

        vx = signed_byte((r << 1) & jnp.int32(255)) / 256.0
        vy = jnp.where(
            victim_is_second,
            -signed_byte(r) / 256.0,
            1.0 - r.astype(jnp.float32) / 256.0,
        )
        return jnp.array([vx, vy], dtype=jnp.float32)

    def _contact_pair(
        self,
        first: CharacterState,
        second: CharacterState,
        puck_state: PuckState,
        random_byte: chex.Array,
        protect_first: chex.Array = False,
        protect_second: chex.Array = False,
    ) -> Tuple[CharacterState, CharacterState, PuckState, chex.Array]:
        protect_first = jnp.asarray(protect_first, dtype=jnp.bool_)
        protect_second = jnp.asarray(protect_second, dtype=jnp.bool_)
        touching = self._characters_touch(first.position, second.position)
        random_phase = random_byte.astype(jnp.int32) & jnp.int32(7)
        first_available = ~first.is_tackled

        first_hits = (
            touching
            & first_available
            & (first.swing_phase > 0)
            & (random_phase == 4)
            & ~protect_second
        )
        second_was_holder = second.has_puck
        second = jax.lax.cond(
            first_hits,
            lambda ch: self._knock_down(ch, random_byte),
            lambda ch: ch,
            second,
        )
        drop_second = first_hits & second_was_holder
        puck_state = puck_state.replace(
            velocity=jnp.where(
                drop_second,
                self._tackle_drop_velocity(random_byte, jnp.array(True)),
                puck_state.velocity,
            ),
            holder=jnp.where(drop_second, jnp.int32(-1), puck_state.holder),
        )

        second_hits = (
            touching
            & first_available
            & (second.swing_phase > 0)
            & (random_phase == 0)
            & ~protect_first
        )
        first_was_holder = first.has_puck
        first = jax.lax.cond(
            second_hits,
            lambda ch: self._knock_down(ch, random_byte),
            lambda ch: ch,
            first,
        )
        drop_first = second_hits & first_was_holder
        puck_state = puck_state.replace(
            velocity=jnp.where(
                drop_first,
                self._tackle_drop_velocity(random_byte, jnp.array(False)),
                puck_state.velocity,
            ),
            holder=jnp.where(drop_first, jnp.int32(-1), puck_state.holder),
        )

        delta = first.position - second.position
        shift_x = jnp.where(
            delta[0] >= 0.0, self.consts.CONTACT_PUSH_X, -self.consts.CONTACT_PUSH_X
        )
        shift_y = jnp.where(
            delta[1] <= 0.0, -self.consts.CONTACT_PUSH_Y, self.consts.CONTACT_PUSH_Y
        )
        shift = jnp.where(
            touching,
            jnp.array([shift_x, shift_y], dtype=jnp.float32),
            jnp.zeros(2, dtype=jnp.float32),
        )
        first = first.replace(position=first.position + shift)
        second = second.replace(position=second.position - shift)
        return first, second, puck_state, touching

    def _contact_step(
        self,
        player_state: PlayerState,
        enemy_state: EnemyState,
        puck_state: PuckState,
        random_byte: chex.Array,
        counter: chex.Array,
        player_score: chex.Array,
        enemy_score: chex.Array,
    ) -> Tuple[PlayerState, EnemyState, PuckState, chex.Array]:
        p_sk, p_go = player_state.skater, player_state.goalie
        e_sk, e_go = enemy_state.skater, enemy_state.goalie
        aggressive = (player_score >= enemy_score) | (
            (counter.astype(jnp.int32) & jnp.int32(255)) < 64
        )

        enemy_holds = e_sk.has_puck | e_go.has_puck
        contact_1 = self._characters_touch(p_sk.position, e_go.position)
        force_fire_1 = contact_1 & ~enemy_holds & aggressive
        enemy_goalie_grid_y = self._character_grid_position(e_go.position)[1]
        p_sk, e_go, puck_state, _ = self._contact_pair(
            p_sk,
            e_go,
            puck_state,
            random_byte,
            protect_second=enemy_goalie_grid_y < self.consts.ENEMY_GOALIE_PROTECTED_GRID_Y,
        )

        enemy_holds = e_sk.has_puck | e_go.has_puck
        contact_2 = self._characters_touch(e_sk.position, p_sk.position)
        force_fire_2 = contact_2 & ~enemy_holds & aggressive
        e_sk, p_sk, puck_state, _ = self._contact_pair(
            e_sk, p_sk, puck_state, random_byte
        )

        enemy_holds = e_sk.has_puck | e_go.has_puck
        contact_3 = self._characters_touch(p_go.position, e_sk.position)
        force_fire_3 = contact_3 & ~enemy_holds & aggressive
        player_goalie_grid_y = self._character_grid_position(p_go.position)[1]
        p_go, e_sk, puck_state, _ = self._contact_pair(
            p_go,
            e_sk,
            puck_state,
            random_byte,
            protect_first=player_goalie_grid_y >= self.consts.PLAYER_GOALIE_PROTECTED_GRID_Y,
        )

        return (
            player_state.replace(skater=p_sk, goalie=p_go),
            enemy_state.replace(skater=e_sk, goalie=e_go),
            puck_state,
            force_fire_1 | force_fire_2 | force_fire_3,
        )

    def _resolve_active_characters(
        self,
        player_state: PlayerState,
        enemy_state: EnemyState,
        puck_position: chex.Array,
    ) -> Tuple[chex.Array, chex.Array]:
        """Choose the controlled character for each team."""
        c = self.consts
        puck_grid = self._puck_grid_position(puck_position)
        puck_y_even = puck_grid[1] & jnp.int32(254)

        def wrapped_abs_delta(a, b):
            d = (a - b) & jnp.int32(255)
            return jnp.where(d < 128, d, d ^ jnp.int32(255))

        def distance(char: CharacterState):
            char_grid = self._character_grid_position(char.position)
            return wrapped_abs_delta(puck_y_even, char_grid[1]) + wrapped_abs_delta(
                puck_grid[0], char_grid[0]
            )

        player_goalie_distance = distance(player_state.goalie)
        player_skater_distance = distance(player_state.skater)
        enemy_skater_distance = distance(enemy_state.skater)
        enemy_goalie_distance = distance(enemy_state.goalie)

        # Ties favor the forward for the player and the goalie for the enemy.
        player_active = jnp.where(
            player_goalie_distance < player_skater_distance, 1, 0
        ).astype(jnp.int32)
        enemy_active = jnp.where(
            enemy_skater_distance < enemy_goalie_distance, 0, 1
        ).astype(jnp.int32)

        bottom_region = puck_grid[1] < jnp.int32(c.ACTIVE_BOTTOM_THRESHOLD)
        top_region = puck_grid[1] >= jnp.int32(c.ACTIVE_TOP_THRESHOLD)
        player_active = jnp.where(
            bottom_region,
            jnp.int32(0),
            jnp.where(top_region, jnp.int32(1), player_active),
        )
        enemy_active = jnp.where(
            bottom_region,
            jnp.int32(1),
            jnp.where(top_region, jnp.int32(0), enemy_active),
        )

        # A downed character always hands control to its teammate.
        player_active = jnp.where(
            player_state.goalie.is_tackled,
            jnp.int32(0),
            jnp.where(player_state.skater.is_tackled, jnp.int32(1), player_active),
        )
        enemy_active = jnp.where(
            enemy_state.skater.is_tackled,
            jnp.int32(1),
            jnp.where(enemy_state.goalie.is_tackled, jnp.int32(0), enemy_active),
        )
        return player_active, enemy_active

    def _apply_action(
        self,
        character: CharacterState,
        action: chex.Array,
    ) -> CharacterState:
        """Apply one frame of joystick *input* movement to a single character.

        This is the per-character movement primitive shared by the human player and the
        computer opponent: each chooses an action through its own policy, but the action
        is applied identically here. Directions are absolute screen directions.

        Only the active skater of a team should receive a real action; the inactive
        teammate never moves from input (the caller passes NOOP or simply skips it).
        A tackled character ignores input entirely (it is frozen for the tackle period).

        Args:
            character: The character to move.
            action: The chosen Atari action integer.
            Movement is intentionally left unclamped here. Position constraints are
            applied only after opponent contacts, matching the gameplay update order.

        Returns:
            The updated ``CharacterState`` (position + orientation; other fields kept).
        """
        up = jnp.any(
            jnp.array(
                [
                    action == Action.UP,
                    action == Action.UPRIGHT,
                    action == Action.UPLEFT,
                    action == Action.UPFIRE,
                    action == Action.UPRIGHTFIRE,
                    action == Action.UPLEFTFIRE,
                ]
            )
        )
        down = jnp.any(
            jnp.array(
                [
                    action == Action.DOWN,
                    action == Action.DOWNRIGHT,
                    action == Action.DOWNLEFT,
                    action == Action.DOWNFIRE,
                    action == Action.DOWNRIGHTFIRE,
                    action == Action.DOWNLEFTFIRE,
                ]
            )
        )
        left = jnp.any(
            jnp.array(
                [
                    action == Action.LEFT,
                    action == Action.UPLEFT,
                    action == Action.DOWNLEFT,
                    action == Action.LEFTFIRE,
                    action == Action.UPLEFTFIRE,
                    action == Action.DOWNLEFTFIRE,
                ]
            )
        )
        right = jnp.any(
            jnp.array(
                [
                    action == Action.RIGHT,
                    action == Action.UPRIGHT,
                    action == Action.DOWNRIGHT,
                    action == Action.RIGHTFIRE,
                    action == Action.UPRIGHTFIRE,
                    action == Action.DOWNRIGHTFIRE,
                ]
            )
        )

        # A tackled character is frozen: ignore all input movement this frame.
        movable = jnp.logical_not(character.is_tackled)
        dx = jnp.where(
            movable & right,
            self.consts.CHARACTER_SPEED_X,
            jnp.where(movable & left, -self.consts.CHARACTER_SPEED_X, 0.0),
        )
        # Screen y grows downward. Vertical movement is two pixels per update.
        dy = jnp.where(
            movable & down,
            self.consts.CHARACTER_SPEED_Y,
            jnp.where(movable & up, -self.consts.CHARACTER_SPEED_Y, 0.0),
        )
        new_position = character.position + jnp.array([dx, dy], dtype=jnp.float32)

        # Orientation: 0 = facing left, 1 = facing right.
        # input keeps the current facing; a tackled character keeps it too (frozen).
        new_orientation = jnp.where(
            movable & right, 1, jnp.where(movable & left, 0, character.orientation)
        )

        # Leg walk-cycle advances whenever the skater has any directional input
        # and freezes on frame 0 when idle (NOOP) or tackled.
        has_input = movable & (up | down | left | right)
        new_walk_counter = jnp.where(has_input, character.walk_counter + 1, 0)

        return character.replace(
            position=new_position,
            orientation=new_orientation,
            walk_counter=new_walk_counter,
        )

    # ------------------------------------------------------------------ #
    # Phase 1 — intended input movement (uniform over a team's two skaters)
    # ------------------------------------------------------------------ #
    def _apply_team_inputs(
        self,
        char1: CharacterState,
        char2: CharacterState,
        active: chex.Array,
        action: chex.Array,
    ) -> Tuple[CharacterState, CharacterState]:
        """Apply one team's chosen action as phase-1 intended movement.

        The reframed phase 1: instead of "only the active skater moves", every character
        is handled uniformly by the same ``_apply_action`` — the active skater receives
        the real action and the teammate receives ``NOOP`` (a zero intended delta). The
        active/passive split therefore collapses to "what action does this character get
        this frame", and the inactive teammate simply gets a no-op move.

        This is shared by both teams: the player's action comes from the agent, the
        computer's from ``_enemy_policy``, but routing + application are identical.

        Returns the two characters with their unclamped provisional movement positions.
        """
        action1 = jnp.where(active == 0, action, Action.NOOP)
        action2 = jnp.where(active == 1, action, Action.NOOP)
        return (
            self._apply_action(char1, action1),
            self._apply_action(char2, action2),
        )

    # ------------------------------------------------------------------ #
    # Position constraints
    # ------------------------------------------------------------------ #
    def _enforce_min_vertical(
        self,
        active_pos: chex.Array,
        passive_pos: chex.Array,
        min_vertical_distance: chex.Array,
    ) -> chex.Array:
        """Resolve a same-team overlap: vertical-only push, mover holds / passive yields.

        Distinct from the opponent mechanic. The active (moving) skater keeps its
        position; the passive teammate is displaced along y only so the pair are at
        least ``min_vertical_distance`` apart (e.g. the goalie skating forward pushes
        his teammate straight up). The passive's x is untouched.

        Returns the passive teammate's new position. No-op when the gap is already large
        enough.
        """
        dy = passive_pos[1] - active_pos[1]
        too_close = jnp.abs(dy) < min_vertical_distance
        # Preserve which side the passive is on; if exactly level, default to below.
        side = jnp.where(dy != 0.0, jnp.sign(dy), 1.0)
        new_y = jnp.where(
            too_close, active_pos[1] + side * min_vertical_distance, passive_pos[1]
        )
        return jnp.array([passive_pos[0], new_y])

    def _clamp_to_bounds(self, pos: chex.Array, bounds: chex.Array) -> chex.Array:
        """Authoritative wall/zone clamp: ``(x_min, x_max, y_min, y_max)``."""
        return jnp.array(
            [
                jnp.clip(pos[0], bounds[0], bounds[1]),
                jnp.clip(pos[1], bounds[2], bounds[3]),
            ]
        )

    def _character_bounds(self):
        c = self.consts
        x_min = c.RINK_LEFT
        x_max = c.RINK_RIGHT - c.PLAYER_W

        upper_y_min = c.CHARACTER_GRID_Y_ORIGIN - c.UPPER_CHARACTER_GRID_Y_MAX
        upper_y_max = c.CHARACTER_GRID_Y_ORIGIN - c.UPPER_CHARACTER_GRID_Y_MIN
        lower_y_min = c.CHARACTER_GRID_Y_ORIGIN - c.LOWER_CHARACTER_GRID_Y_MAX
        lower_y_max = c.CHARACTER_GRID_Y_ORIGIN - c.LOWER_CHARACTER_GRID_Y_MIN

        return (
            # player skater
            jnp.array([x_min, x_max, lower_y_min, lower_y_max], dtype=jnp.float32),
            # player goalie
            jnp.array([x_min, x_max, upper_y_min, upper_y_max], dtype=jnp.float32),
            # enemy skater
            jnp.array([x_min, x_max, upper_y_min, upper_y_max], dtype=jnp.float32),
            # enemy goalie
            jnp.array([x_min, x_max, lower_y_min, lower_y_max], dtype=jnp.float32),
        )

    def _finalize_character_positions(
        self, player_state: PlayerState, enemy_state: EnemyState
    ) -> Tuple[PlayerState, EnemyState]:
        p1, p2 = player_state.skater.position, player_state.goalie.position
        e1, e2 = enemy_state.skater.position, enemy_state.goalie.position
        min_distance = jnp.float32(self.consts.MIN_VERTICAL_DISTANCE)

        p2 = jnp.where(
            player_state.active_character == 0,
            self._enforce_min_vertical(p1, p2, min_distance),
            p2,
        )
        p1 = jnp.where(
            player_state.active_character == 1,
            self._enforce_min_vertical(p2, p1, min_distance),
            p1,
        )
        e2 = jnp.where(
            enemy_state.active_character == 0,
            self._enforce_min_vertical(e1, e2, min_distance),
            e2,
        )
        e1 = jnp.where(
            enemy_state.active_character == 1,
            self._enforce_min_vertical(e2, e1, min_distance),
            e1,
        )

        bp1, bp2, be1, be2 = self._character_bounds()
        return (
            player_state.replace(
                skater=player_state.skater.replace(
                    position=self._clamp_to_bounds(p1, bp1)
                ),
                goalie=player_state.goalie.replace(
                    position=self._clamp_to_bounds(p2, bp2)
                ),
            ),
            enemy_state.replace(
                skater=enemy_state.skater.replace(
                    position=self._clamp_to_bounds(e1, be1)
                ),
                goalie=enemy_state.goalie.replace(
                    position=self._clamp_to_bounds(e2, be2)
                ),
            ),
        )

    def _advance_puck_with_walls(
        self, position: chex.Array, velocity: chex.Array
    ) -> Tuple[chex.Array, chex.Array]:
        """Advance a puck and resolve the rink boards and rigid goal posts.

        The goal-post pixels use the same clamp-and-reflect rule as the outer
        rink walls.  Their vertical span is the rink-facing goal-height
        pixels at the top and bottom board, while ``GOAL_X0`` and ``GOAL_X1``
        are the left and right post pixels respectively.
        """
        c = self.consts
        tentative = position + velocity

        hit_left = tentative[0] < c.RINK_LEFT
        hit_right = tentative[0] > c.RINK_RIGHT
        hit_top = tentative[1] < c.RINK_TOP
        hit_bottom = tentative[1] > c.RINK_BOTTOM
        resolved_y = jnp.clip(tentative[1], c.RINK_TOP, c.RINK_BOTTOM)

        # A goal-height pixel segment includes its board pixel, so its inward
        # extent is height - 1.  A post only blocks a puck entering the
        # goal mouth from its corresponding outside side; this mirrors the
        # outer-board convention, which permits occupying a wall pixel and
        # reflects only on the attempted step beyond it.
        top_post_depth = c.GOAL_HEIGHT_TOP - 1
        bottom_post_depth = c.GOAL_HEIGHT_BOTTOM - 1
        in_top_post_band = (resolved_y >= c.RINK_TOP) & (
            resolved_y <= c.RINK_TOP + top_post_depth
        )
        in_bottom_post_band = (resolved_y >= c.RINK_BOTTOM - bottom_post_depth) & (
            resolved_y <= c.RINK_BOTTOM
        )
        in_post_band = in_top_post_band | in_bottom_post_band
        hit_goal_left = (
            in_post_band
            & (position[0] <= c.GOAL_X0)
            & (tentative[0] > c.GOAL_X0)
        )
        hit_goal_right = (
            in_post_band
            & (position[0] >= c.GOAL_X1)
            & (tentative[0] < c.GOAL_X1)
        )

        vx = jnp.where(
            hit_left | hit_right | hit_goal_left | hit_goal_right,
            -velocity[0],
            velocity[0],
        )
        vy = jnp.where(hit_top | hit_bottom, -velocity[1], velocity[1])

        board_x = jnp.clip(tentative[0], c.RINK_LEFT, c.RINK_RIGHT)
        resolved_x = jnp.where(
            hit_goal_left,
            jnp.float32(c.GOAL_X0),
            jnp.where(hit_goal_right, jnp.float32(c.GOAL_X1), board_x),
        )
        resolved_position = jnp.array([resolved_x, resolved_y], dtype=jnp.float32)
        resolved_velocity = jnp.array([vx, vy], dtype=jnp.float32)
        return resolved_position, resolved_velocity

    def _decay_puck_velocity(self, velocity: chex.Array) -> chex.Array:
        # Puck velocity uses 1/256-pixel increments. Each component loses roughly
        # 1/64 of its value per puck update, with the smallest quantized speeds
        # retained instead of decaying to zero.
        fixed = jnp.rint(velocity * 256.0).astype(jnp.int32)
        decay = fixed >> jnp.int32(6)
        decay = jnp.where((decay == 0) | (decay == -1), 0, decay)
        fixed = fixed - decay
        return fixed.astype(jnp.float32) / 256.0

    def _puck_step(self, puck: PuckState) -> PuckState:
        vel = self._decay_puck_velocity(puck.velocity)
        pos, vel = self._advance_puck_with_walls(puck.position, vel)
        return puck.replace(position=pos, velocity=vel)

    def _stick_slot(self, carry_timer: chex.Array) -> chex.Array:
        c = self.consts
        n = c.STICK_SLOTS  # 32
        phase = (carry_timer // c.STICK_CADENCE) % (2 * n)  # 0..63
        slot = jnp.where(phase < n, phase, 2 * n - 1 - phase)  # 0..31, 31..0
        return slot.astype(jnp.int32)

    def _visible_stick_pos(self, carry_timer: chex.Array) -> chex.Array:
        # 8 visible carry positions as their own (coarser, faster) triangle
        # wave, independent of the fine 32-slot shot-angle wave in
        # _stick_slot. Same reflecting-triangle shape, so each outer position
        # (0 and 7) is naturally held for 2 ticks at the turnaround while the
        # inner ones hold for 1 tick each: 0,1,2,3,4,5,6,7,7,6,5,4,3,2,1,0,...
        c = self.consts
        n = 8
        phase = (carry_timer // c.STICK_VISIBLE_CADENCE) % (2 * n)  # 0..15
        visible = jnp.where(phase < n, phase, 2 * n - 1 - phase)  # 0..7, 7..0
        return visible.astype(jnp.int32)

    def _carried_puck_pos(self, char, visible, offset_y):
        c = self.consts
        base = jnp.round(char.position)
        t = visible.astype(jnp.float32) / 7.0
        dx = c.STICK_MIN_DX + t * (c.STICK_MAX_DX - c.STICK_MIN_DX)
        x = jnp.where(
            char.orientation == 1,
            base[0] + c.PLAYER_W / 2.0 + dx,
            base[0] + c.PLAYER_W / 2.0 - dx - c.PUCK_W,
        )
        y = base[1] + offset_y
        return jnp.array([x, y], dtype=jnp.float32)

    def _puck_pickup(
        self,
        player_state: PlayerState,
        enemy_state: EnemyState,
        puck_state: PuckState,
        random_byte: chex.Array,
    ) -> tuple[PlayerState, EnemyState, PuckState]:
        """Resolve free-puck pickups and steals."""
        chars = (
            player_state.skater,
            player_state.goalie,
            enemy_state.skater,
            enemy_state.goalie,
        )
        indices = jnp.arange(4, dtype=jnp.int32)

        # Character-specific pickup phase and simultaneous-contact priority.
        pickup_rank = jnp.array([2, 0, 1, 3], dtype=jnp.int32)
        puck_grid = self._puck_grid_position(puck_state.position)

        def in_pickup_region(char: CharacterState) -> chex.Array:
            char_grid = self._character_grid_position(char.position)
            dx = (puck_grid[0] - char_grid[0]) & jnp.int32(255)
            dy = (puck_grid[1] - char_grid[1]) & jnp.int32(255)

            left_x_ok = (dx < 8) | (dx >= 249)
            right_x_ok = dx < 18
            x_ok = jnp.where(char.orientation == 0, left_x_ok, right_x_ok)
            y_ok = (dy < 4) | (dy == 255)
            return x_ok & y_ok

        in_range = jnp.array([in_pickup_region(char) for char in chars])
        holds = jnp.array([char.has_puck for char in chars])
        tackled = jnp.array([char.is_tackled for char in chars])
        puck_is_held = jnp.any(holds)

        blocker_timer = jnp.maximum(puck_state.pickup_blocker_timer - 1, 0)
        blocked = (
            (blocker_timer > 0)
            & (indices == puck_state.pickup_blocker)
        )
        eligible = in_range & ~holds & ~tackled & ~blocked

        # A free puck is collected immediately. Stealing an attached puck succeeds
        # on one of four deterministic random phases for each character.
        random_phase = random_byte.astype(jnp.int32) & jnp.int32(255)
        steal_allowed = ((pickup_rank ^ random_phase) & jnp.int32(3)) == 0
        successful = eligible & (~puck_is_held | steal_allowed)

        priority = jnp.where(successful, pickup_rank, jnp.int32(-1))
        acquired = jnp.max(priority) >= 0
        winner = jnp.argmax(priority)
        new_has_puck = jnp.where(acquired, indices == winner, holds)

        blocker_timer = jnp.where(acquired, 0, blocker_timer).astype(jnp.int32)
        pickup_blocker = jnp.where(
            acquired | (blocker_timer == 0),
            jnp.int32(-1),
            puck_state.pickup_blocker,
        )

        player_state = player_state.replace(
            skater=player_state.skater.replace(has_puck=new_has_puck[0]),
            goalie=player_state.goalie.replace(has_puck=new_has_puck[1]),
        )
        enemy_state = enemy_state.replace(
            skater=enemy_state.skater.replace(has_puck=new_has_puck[2]),
            goalie=enemy_state.goalie.replace(has_puck=new_has_puck[3]),
        )
        puck_state = puck_state.replace(
            pickup_blocker=pickup_blocker,
            pickup_blocker_timer=blocker_timer,
        )
        return player_state, enemy_state, puck_state

    def _puck_carry(
        self,
        player_state,
        enemy_state,
        puck_state,
        advance_free_puck: chex.Array,
    ):
        c = self.consts
        p_sk, p_go = player_state.skater, player_state.goalie
        e_sk, e_go = enemy_state.skater, enemy_state.goalie
        anyone_has = p_sk.has_puck | p_go.has_puck | e_sk.has_puck | e_go.has_puck

        holder = jnp.where(
            p_sk.has_puck,
            0,
            jnp.where(
                p_go.has_puck,
                1,
                jnp.where(e_sk.has_puck, 2, jnp.where(e_go.has_puck, 3, -1)),
            ),
        ).astype(jnp.int32)

        same_holder = holder == puck_state.holder
        # A new holder starts at phase 1 of the stick cycle.
        carry_timer = jnp.where(
            anyone_has,
            jnp.where(same_holder, puck_state.carry_timer + 1, jnp.int32(1)),
            jnp.int32(0),
        ).astype(jnp.int32)

        slot = self._stick_slot(carry_timer)
        visible = self._visible_stick_pos(carry_timer)

        carry_pos = jnp.where(
            p_sk.has_puck,
            self._carried_puck_pos(
                p_sk, visible, offset_y=c.PLAYER_CARRIED_PUCK_OFFSET_Y
            ),
            jnp.where(
                p_go.has_puck,
                self._carried_puck_pos(
                    p_go, visible, offset_y=c.PLAYER_CARRIED_PUCK_OFFSET_Y
                ),
                jnp.where(
                    e_sk.has_puck,
                    self._carried_puck_pos(
                        e_sk, visible, offset_y=c.ENEMY_CARRIED_PUCK_OFFSET_Y
                    ),
                    self._carried_puck_pos(
                        e_go, visible, offset_y=c.ENEMY_CARRIED_PUCK_OFFSET_Y
                    ),
                ),
            ),
        )

        free_puck = jax.lax.cond(
            advance_free_puck,
            self._puck_step,
            lambda puck: puck,
            puck_state,
        )
        new_puck_pos = jnp.where(anyone_has, carry_pos, free_puck.position)
        new_puck_vel = jnp.where(
            anyone_has, jnp.zeros(2, dtype=jnp.float32), free_puck.velocity
        )
        return puck_state.replace(
            position=new_puck_pos,
            velocity=new_puck_vel,
            position_stick=slot,
            carry_timer=carry_timer,
            holder=holder,
        )

    def _puck_shoot(
        self,
        player_state,
        enemy_state,
        puck_state,
        player_fire: chex.Array,
        enemy_fire: chex.Array,
        advance_free_puck: chex.Array,
    ):
        c = self.consts
        p_sk, p_go = player_state.skater, player_state.goalie
        e_sk, e_go = enemy_state.skater, enemy_state.goalie
        sk_shoots = p_sk.has_puck & player_fire
        go_shoots = p_go.has_puck & player_fire
        e_sk_shoots = e_sk.has_puck & enemy_fire
        e_go_shoots = e_go.has_puck & enemy_fire
        should_shoot = sk_shoots | go_shoots | e_sk_shoots | e_go_shoots

        slot = puck_state.position_stick  # 0..31

        def shot_for(char, team_sign):
            s_eff = jnp.where(char.orientation == 1, slot, 31 - slot)
            vx = (s_eff.astype(jnp.float32) - 16.0) / 8.0
            vy = team_sign * jnp.float32(c.PUCK_MAX_SPEED)
            return jnp.array([vx, vy], dtype=jnp.float32)

        shot_vel = jnp.where(
            sk_shoots,
            shot_for(p_sk, 1.0),
            jnp.where(
                go_shoots,
                shot_for(p_go, 1.0),
                jnp.where(e_sk_shoots, shot_for(e_sk, -1.0), shot_for(e_go, -1.0)),
            ),
        )
        released_puck = puck_state.replace(velocity=shot_vel)
        released_puck = jax.lax.cond(
            should_shoot & advance_free_puck,
            self._puck_step,
            lambda puck: puck,
            released_puck,
        )
        new_velocity = jnp.where(
            should_shoot, released_puck.velocity, puck_state.velocity
        )
        new_position = jnp.where(
            should_shoot, released_puck.position, puck_state.position
        )
        shooter_index = jnp.where(
            sk_shoots,
            jnp.array(0, dtype=jnp.int32),
            jnp.where(
                go_shoots,
                jnp.array(1, dtype=jnp.int32),
                jnp.where(
                    e_sk_shoots,
                    jnp.array(2, dtype=jnp.int32),
                    jnp.array(3, dtype=jnp.int32),
                ),
            ),
        )
        new_pickup_blocker = jnp.where(
            should_shoot, shooter_index, puck_state.pickup_blocker
        )
        new_pickup_blocker_timer = jnp.where(
            should_shoot,
            jnp.int32(c.REGRAB_BLOCK_FRAMES),
            puck_state.pickup_blocker_timer,
        )
        new_player_state = player_state.replace(
            skater=p_sk.replace(has_puck=p_sk.has_puck & ~sk_shoots),
            goalie=p_go.replace(has_puck=p_go.has_puck & ~go_shoots),
        )
        new_enemy_state = enemy_state.replace(
            skater=e_sk.replace(has_puck=e_sk.has_puck & ~e_sk_shoots),
            goalie=e_go.replace(has_puck=e_go.has_puck & ~e_go_shoots),
        )
        return (
            new_player_state,
            new_enemy_state,
            puck_state.replace(
                position=new_position,
                velocity=new_velocity,
                pickup_blocker=new_pickup_blocker,
                pickup_blocker_timer=new_pickup_blocker_timer,
            ),
        )

    def _move_characters(
        self,
        player_state: PlayerState,
        enemy_state: EnemyState,
        player_action: chex.Array,
        enemy_action: chex.Array,
    ) -> Tuple[PlayerState, EnemyState]:
        """Apply one scheduled movement update to each team's active character."""
        p1, p2 = self._apply_team_inputs(
            player_state.skater,
            player_state.goalie,
            player_state.active_character,
            player_action,
        )
        e1, e2 = self._apply_team_inputs(
            enemy_state.skater,
            enemy_state.goalie,
            enemy_state.active_character,
            enemy_action,
        )
        return (
            player_state.replace(skater=p1, goalie=p2),
            enemy_state.replace(skater=e1, goalie=e2),
        )

    def render(self, state: IceHockeyState) -> jnp.ndarray:
        return self.renderer.render(state)

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
            player_skater=obj(
                state.player_state.skater.position, c.PLAYER_W, c.PLAYER_H
            ),
            player_goalie=obj(
                state.player_state.goalie.position, c.PLAYER_W, c.PLAYER_H
            ),
            enemy_skater=obj(state.enemy_state.skater.position, c.PLAYER_W, c.PLAYER_H),
            enemy_goalie=obj(state.enemy_state.goalie.position, c.PLAYER_W, c.PLAYER_H),
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

        return jnp.concatenate(
            [
                flat(obs.player_skater),
                flat(obs.player_goalie),
                flat(obs.enemy_skater),
                flat(obs.enemy_goalie),
                flat(obs.puck),
                jnp.array(
                    [
                        obs.player_score,
                        obs.enemy_score,
                        obs.remaining_time,
                        obs.active_player,
                    ],
                    dtype=jnp.float32,
                ),
            ]
        )

    def _get_info(self, state: IceHockeyState) -> IceHockeyInfo:
        return IceHockeyInfo(
            player_score=state.game_state.player_score,
            enemy_score=state.game_state.enemy_score,
            remaining_time=state.game_state.remaining_time,
        )

    def _get_reward(
        self, previous_state: IceHockeyState, state: IceHockeyState
    ) -> chex.Array:
        # Reward is the change in goal difference: +1 scored, -1 conceded.
        prev_diff = (
            previous_state.game_state.player_score
            - previous_state.game_state.enemy_score
        )
        diff = state.game_state.player_score - state.game_state.enemy_score
        return (diff - prev_diff).astype(jnp.float32)

    def _get_done(self, state: IceHockeyState) -> chex.Array:
        return state.game_state.is_finished


class IceHockeyRenderer(JAXGameRenderer):
    # Palette-based renderer. The rink (boards, lines, goals, score bars) is
    # baked into the background, so render() only stamps the moving objects.

    NUM_CHARACTER_SPRITES = 22  # idle×2 + stand×2 + walk×8 + swing×10

    @staticmethod
    def _build_character_sprite_table(
        idle_l: jnp.ndarray,
        idle_r: jnp.ndarray,
        stand_l: jnp.ndarray,
        stand_r: jnp.ndarray,
        walk_l: jnp.ndarray,
        walk_r: jnp.ndarray,
        swing_l: jnp.ndarray,
        swing_r: jnp.ndarray,
    ) -> jnp.ndarray:
        """Pack every ordinary character pose into one directly indexable table."""
        return jnp.stack(
            [
                idle_l,
                idle_r,
                stand_l,
                stand_r,
                walk_l[0],
                walk_l[1],
                walk_l[2],
                walk_l[3],
                walk_r[0],
                walk_r[1],
                walk_r[2],
                walk_r[3],
                swing_l[0],
                swing_l[1],
                swing_l[2],
                swing_l[3],
                swing_l[4],
                swing_r[0],
                swing_r[1],
                swing_r[2],
                swing_r[3],
                swing_r[4],
            ],
            axis=0,
        )

    @staticmethod
    def _character_sprite_index(
        char: CharacterState,
        is_active: chex.Array,
        cadence: int,
        swing_phase_count: int,
    ) -> chex.Array:
        """Return the ordinary-pose table index without rendering branches."""
        facing_left = char.orientation == 0
        orient = jnp.where(facing_left, jnp.int32(0), jnp.int32(1))

        idle_idx = orient
        stand_idx = jnp.int32(2) + orient

        walk_frame = (char.walk_counter // cadence) % 4
        walk_idx = (
            jnp.int32(4)
            + walk_frame
            + jnp.where(facing_left, jnp.int32(0), jnp.int32(4))
        )

        # This is exactly the same phase-to-frame mapping as the previous renderer:
        # phases 1..7 -> swing frames [0, 0, 1, 2, 3, 4, 4].
        swing_frame = jnp.clip(
            ((char.swing_phase - 1) * 5) // (swing_phase_count - 1),
            0,
            4,
        )
        swing_idx = (
            jnp.int32(12)
            + swing_frame
            + jnp.where(facing_left, jnp.int32(0), jnp.int32(5))
        )

        normal_idx = jnp.where(
            is_active,
            jnp.where(char.walk_counter > 0, walk_idx, stand_idx),
            idle_idx,
        )
        # Swing rendering is independent of which teammate is currently active.
        return jnp.where(char.swing_phase > 0, swing_idx, normal_idx)

    @staticmethod
    def _goalie_sprite_index(
        char: CharacterState,
        is_active: chex.Array,
        cadence: int,
        swing_phase_count: int,
    ) -> chex.Array:
        base = IceHockeyRenderer._character_sprite_index(
            char, is_active, cadence, swing_phase_count
        )
        # Goalie table: tackled + ordinary poses.
        return jnp.where(char.is_tackled, jnp.int32(0), base + 1)

    @staticmethod
    def _skater_sprite_index(
        char: CharacterState,
        is_active: chex.Array,
        is_faceoff: chex.Array,
        cadence: int,
        swing_phase_count: int,
    ) -> chex.Array:
        base = IceHockeyRenderer._character_sprite_index(
            char, is_active, cadence, swing_phase_count
        )
        # Skater table: tackled + face-off + ordinary poses. This preserves the
        # previous precedence: tackled > face-off > swing > normal pose.
        return jnp.where(
            char.is_tackled,
            jnp.int32(0),
            jnp.where(is_faceoff, jnp.int32(1), base + 2),
        )

    def _stamp_character_sprite(
        self,
        raster: jnp.ndarray,
        char: CharacterState,
        sprite_table: jnp.ndarray,
        sprite_idx: chex.Array,
    ) -> jnp.ndarray:
        return self.jr.render_at(
            raster,
            jnp.round(char.position[0]).astype(jnp.int32),
            jnp.round(char.position[1]).astype(jnp.int32),
            sprite_table[sprite_idx],
        )

    def __init__(
        self,
        consts: Optional[IceHockeyConstants] = None,
        config: Optional[render_utils.RendererConfig] = None,
    ):
        self.consts = consts or IceHockeyConstants()
        super().__init__(self.consts)

        self.config = config or render_utils.RendererConfig(
            game_dimensions=(210, 160), channels=3, downscale=None
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)

        # Branch-local sprite folder for now; move to the shared sprite dir later.
        self.sprite_path = os.path.join(
            os.path.dirname(__file__), "sprites", "icehockey"
        )

        final_asset_config = list(self.consts.ASSET_CONFIG)
        debug_red = jnp.array([255, 0, 0, 255], dtype=jnp.uint8)
        debug_dot = debug_red.reshape(1, 1, 4)
        debug_box_h = int(round(self.consts.PICKUP_BOX_H)) + 1
        debug_box_w = int(round(self.consts.PICKUP_BOX_W)) + 1
        debug_pickup_box = jnp.zeros((debug_box_h, debug_box_w, 4), dtype=jnp.uint8)
        debug_pickup_box = debug_pickup_box.at[0, :, :].set(debug_red)
        debug_pickup_box = debug_pickup_box.at[-1, :, :].set(debug_red)
        debug_pickup_box = debug_pickup_box.at[:, 0, :].set(debug_red)
        debug_pickup_box = debug_pickup_box.at[:, -1, :].set(debug_red)

        # Colon for the clock, two dots in the blue scoreboard colour.
        clock_blue = jnp.array([84, 92, 214, 255], dtype=jnp.uint8)
        clock_colon = jnp.zeros((7, 2, 4), dtype=jnp.uint8)
        clock_colon = clock_colon.at[1, :, :].set(clock_blue)
        clock_colon = clock_colon.at[5, :, :].set(clock_blue)
        final_asset_config.extend(
            [
                {"name": "debug_position_dot", "type": "procedural", "data": debug_dot},
                {
                    "name": "debug_pickup_box",
                    "type": "procedural",
                    "data": debug_pickup_box,
                },
                {"name": "clock_colon", "type": "procedural", "data": clock_colon},
            ]
        )
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS,
        ) = self.jr.load_and_setup_assets(final_asset_config, self.sprite_path)

        # These tables move sprite-state control flow out of the jitted render graph.
        self.PLAYER_SPRITE_TABLE = self._build_character_sprite_table(
            self.SHAPE_MASKS["player_idle_left"],
            self.SHAPE_MASKS["player_idle_right"],
            self.SHAPE_MASKS["player_active_standing_left"],
            self.SHAPE_MASKS["player_active_standing_right"],
            self.SHAPE_MASKS["player_walking_left"],
            self.SHAPE_MASKS["player_walking_right"],
            self.SHAPE_MASKS["player_shooting_left"],
            self.SHAPE_MASKS["player_shooting_right"],
        )
        self.ENEMY_SPRITE_TABLE = self._build_character_sprite_table(
            self.SHAPE_MASKS["enemy_idle_left"],
            self.SHAPE_MASKS["enemy_idle_right"],
            self.SHAPE_MASKS["enemy_active_standing_left"],
            self.SHAPE_MASKS["enemy_active_standing_right"],
            self.SHAPE_MASKS["enemy_walking_left"],
            self.SHAPE_MASKS["enemy_walking_right"],
            self.SHAPE_MASKS["enemy_shooting_left"],
            self.SHAPE_MASKS["enemy_shooting_right"],
        )

        assert self.PLAYER_SPRITE_TABLE.shape[0] == self.NUM_CHARACTER_SPRITES
        assert self.ENEMY_SPRITE_TABLE.shape[0] == self.NUM_CHARACTER_SPRITES

        self.PLAYER_GOALIE_SPRITE_TABLE = jnp.concatenate(
            [
                jnp.expand_dims(self.SHAPE_MASKS["player_tackled"], 0),
                self.PLAYER_SPRITE_TABLE,
            ],
            axis=0,
        )
        self.ENEMY_GOALIE_SPRITE_TABLE = jnp.concatenate(
            [
                jnp.expand_dims(self.SHAPE_MASKS["enemy_tackled"], 0),
                self.ENEMY_SPRITE_TABLE,
            ],
            axis=0,
        )
        self.PLAYER_SKATER_SPRITE_TABLE = jnp.concatenate(
            [
                jnp.expand_dims(self.SHAPE_MASKS["player_tackled"], 0),
                jnp.expand_dims(self.SHAPE_MASKS["player_faceoff"], 0),
                self.PLAYER_SPRITE_TABLE,
            ],
            axis=0,
        )
        self.ENEMY_SKATER_SPRITE_TABLE = jnp.concatenate(
            [
                jnp.expand_dims(self.SHAPE_MASKS["enemy_tackled"], 0),
                jnp.expand_dims(self.SHAPE_MASKS["enemy_faceoff"], 0),
                self.ENEMY_SPRITE_TABLE,
            ],
            axis=0,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _render_hook_post_background(
        self, raster: jnp.ndarray, state: IceHockeyState
    ) -> jnp.ndarray:
        """No-op hook for mods to redraw the ice/boards before objects are stamped."""
        return raster

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: IceHockeyState) -> jnp.ndarray:
        raster = self.jr.create_object_raster(self.BACKGROUND)
        raster = self._render_hook_post_background(raster, state)

        puck_m = self.SHAPE_MASKS["puck"]
        debug_dot = self.SHAPE_MASKS["debug_position_dot"]
        debug_pickup_box = self.SHAPE_MASKS["debug_pickup_box"]
        cadence = self.consts.ANIM_CADENCE
        swing_phase_count = self.consts.SWING_PHASE_COUNT

        def col(pos):
            return jnp.round(pos[0]).astype(jnp.int32)

        def row(pos):
            return jnp.round(pos[1]).astype(jnp.int32)

        def draw_position_dot(r, char):
            return self.jr.render_at(
                r,
                col(char.position),
                row(char.position),
                debug_dot,
            )

        def pickup_box_pos(char, offset_y):
            offset_x = jnp.where(
                char.orientation == 0,
                self.consts.PICKUP_BOX_OFFSET_X_LEFT,
                self.consts.PICKUP_BOX_OFFSET_X_RIGHT,
            )
            return char.position + jnp.array([offset_x, offset_y])

        def draw_pickup_box(r, char, offset_y):
            box_pos = pickup_box_pos(char, offset_y)
            return self.jr.render_at(
                r,
                col(box_pos),
                row(box_pos),
                debug_pickup_box,
            )

        p_act = state.player_state.active_character
        e_act = state.enemy_state.active_character

        raster = self._stamp_character_sprite(
            raster,
            state.player_state.goalie,
            self.PLAYER_GOALIE_SPRITE_TABLE,
            self._goalie_sprite_index(
                state.player_state.goalie,
                p_act == 1,
                cadence,
                swing_phase_count,
            ),
        )
        raster = self._stamp_character_sprite(
            raster,
            state.enemy_state.goalie,
            self.ENEMY_GOALIE_SPRITE_TABLE,
            self._goalie_sprite_index(
                state.enemy_state.goalie,
                e_act == 1,
                cadence,
                swing_phase_count,
            ),
        )
        raster = self._stamp_character_sprite(
            raster,
            state.player_state.skater,
            self.PLAYER_SKATER_SPRITE_TABLE,
            self._skater_sprite_index(
                state.player_state.skater,
                p_act == 0,
                state.game_state.is_faceoff,
                cadence,
                swing_phase_count,
            ),
        )
        raster = self._stamp_character_sprite(
            raster,
            state.enemy_state.skater,
            self.ENEMY_SKATER_SPRITE_TABLE,
            self._skater_sprite_index(
                state.enemy_state.skater,
                e_act == 0,
                state.game_state.is_faceoff,
                cadence,
                swing_phase_count,
            ),
        )

        raster = self.jr.render_at(
            raster,
            col(state.puck_state.position),
            row(state.puck_state.position),
            puck_m,
        )

        if self.consts.DEBUG_RENDER:
            player_offset_y = self.consts.PLAYER_PICKUP_BOX_OFFSET_Y
            enemy_offset_y = self.consts.ENEMY_PICKUP_BOX_OFFSET_Y
            raster = draw_pickup_box(raster, state.player_state.goalie, player_offset_y)
            raster = draw_pickup_box(raster, state.enemy_state.goalie, enemy_offset_y)
            raster = draw_pickup_box(raster, state.player_state.skater, player_offset_y)
            raster = draw_pickup_box(raster, state.enemy_state.skater, enemy_offset_y)
            raster = draw_position_dot(raster, state.player_state.goalie)
            raster = draw_position_dot(raster, state.enemy_state.goalie)
            raster = draw_position_dot(raster, state.player_state.skater)
            raster = draw_position_dot(raster, state.enemy_state.skater)

        dm_blue = self.SHAPE_MASKS["digits"]  # player score (blue team)
        dm_gold = self.SHAPE_MASKS["digits_gold"]  # enemy score (gold team)

        def draw_score(r, value, x_single, x_double, dm):
            digits = self.jr.int_to_digits(value, max_digits=2)
            is_single = value < 10
            start = jax.lax.select(is_single, jnp.int32(1), jnp.int32(0))
            count = jax.lax.select(is_single, jnp.int32(1), jnp.int32(2))
            x = jax.lax.select(is_single, jnp.int32(x_single), jnp.int32(x_double))
            return self.jr.render_label_selective(
                r, x, 14, digits, dm, start, count, spacing=7, max_digits_to_render=2
            )

        # Blue (player) score on the left, gold (enemy) on the right.
        raster = draw_score(raster, state.game_state.player_score, 46, 33, dm_blue)
        raster = draw_score(raster, state.game_state.enemy_score, 110, 103, dm_gold)

        # Clock "M:SS" at the top. remaining_time is in frames; round up so
        # 3:00 stays visible until the first tick.
        clock_secs = (state.game_state.remaining_time + 59) // 60
        clock_min = clock_secs // 60
        clock_sec = clock_secs % 60
        min_digits = self.jr.int_to_digits(clock_min, max_digits=2)
        sec_digits = self.jr.int_to_digits(clock_sec, max_digits=2)
        raster = self.jr.render_label_selective(
            raster, 65, 5, min_digits, dm_blue, 1, 1, spacing=8, max_digits_to_render=2
        )
        raster = self.jr.render_at(
            raster, 75, 5, self.SHAPE_MASKS["clock_colon"]
        )
        raster = self.jr.render_label_selective(
            raster, 81, 5, sec_digits, dm_blue, 0, 2, spacing=8, max_digits_to_render=2
        )

        frame = self.jr.render_from_palette(raster, self.PALETTE)

        if self.consts.DEBUG_RENDER:
            c = self.consts
            blue = jnp.array([0, 0, 255], dtype=frame.dtype)
            x0 = c.RINK_LEFT
            x1 = min(c.RINK_RIGHT + 1, c.WIDTH)
            player_attack_line_y = (
                c.CHARACTER_GRID_Y_ORIGIN - c.LOWER_CHARACTER_GRID_Y_MAX
            )
            enemy_attack_line_y = (
                c.CHARACTER_GRID_Y_ORIGIN - c.UPPER_CHARACTER_GRID_Y_MIN
            )

            frame = frame.at[player_attack_line_y, x0:x1, :].set(blue)
            frame = frame.at[enemy_attack_line_y, x0:x1, :].set(blue)

        return frame
