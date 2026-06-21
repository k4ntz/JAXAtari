import os
from functools import partial
from typing import Tuple, Optional

import jax
import jax.numpy as jnp
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
        # Skater sprites per team, authored facing right (the renderer mirrors them
        # for left-facing skaters). Per character the renderer picks:
        #   - "<team>" walk-cycle frames (with stick) when the ACTIVE skater moves,
        #   - "<team>_idle" (with stick) when the active skater stands still,
        #   - "<team>_nostick" for the INACTIVE teammate (drawn without a stick).
        {"name": "player", "type": "group",
         "files": ["player_0.npy", "player_1.npy"]},
        {"name": "enemy", "type": "group",
         "files": ["enemy_0.npy", "enemy_1.npy"]},
        {"name": "player_idle", "type": "single", "file": "player_idle.npy"},
        {"name": "enemy_idle", "type": "single", "file": "enemy_idle.npy"},
        {"name": "player_nostick", "type": "single", "file": "player_nostick.npy"},
        {"name": "enemy_nostick", "type": "single", "file": "enemy_nostick.npy"},
        # "<team>_shoot" = the active skater's 2-frame swing animation (wind-up ->
        # follow-through), played while shooting_cooldown > 0.
        {"name": "player_shoot", "type": "group",
         "files": ["player_shoot_0.npy", "player_shoot_1.npy"]},
        {"name": "enemy_shoot", "type": "group",
         "files": ["enemy_shoot_0.npy", "enemy_shoot_1.npy"]},
        {"name": "puck", "type": "single", "file": "puck.npy"},
        {"name": "digits", "type": "digits", "pattern": "digit_{}.npy",
         "recolorings": {"gold": (236, 200, 96)}},
    )


class IceHockeyConstants(struct.PyTreeNode):
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
    ENEMY_GOAL_Y: int = struct.field(pytree_node=False, default=186)
    PLAYER_GOAL_Y: int = struct.field(pytree_node=False, default=42)
    GOAL_HEIGHT: int = struct.field(pytree_node=False, default=4)

    # Sprite sizes, used for observation bounding boxes
    PLAYER_W: int = struct.field(pytree_node=False, default=26)
    PLAYER_H: int = struct.field(pytree_node=False, default=20)
    PUCK_W: int = struct.field(pytree_node=False, default=2)
    PUCK_H: int = struct.field(pytree_node=False, default=2)

    PLAYER_SPEED: float = struct.field(pytree_node=False, default=1.5)

    # Skater leg walk-cycle: number of frames in the loop and how many game
    # frames each phase is shown for. The cycle advances only while a skater has
    # directional input.
    ANIM_FRAMES: int = struct.field(pytree_node=False, default=2)
    ANIM_CADENCE: int = struct.field(pytree_node=False, default=4)

    # Shooting/swing animation: how many game frames the active skater holds the
    # swing (wind-up) pose after a FIRE press. Drives shooting_cooldown.
    SHOOT_ANIM_FRAMES: int = struct.field(pytree_node=False, default=8)

    # Offset from the goal lines defining zone where goalie/skater can't move.
    # A skater is kept out of its own defensive zone (this deep); a goalie is kept
    # out of the opponent's far zone (this deep).
    ATTACKING_ZONE_OFFSET_Y: int = struct.field(pytree_node=False, default=30)
    # How far a goalie may poke into its own goal crease (beyond the rink edge).
    GOALIE_CREASE_DEPTH: int = struct.field(pytree_node=False, default=6)

    # Phase-2 collision tunables for _characters_step
    MIN_SEPARATION: float = struct.field(pytree_node=False, default=8.0)
    MIN_VERTICAL_DISTANCE: float = struct.field(pytree_node=False, default=40.0)

    # 3 min * 60 s * 60 fps = 10800 raw frames.
    TIME_LIMIT: int = struct.field(pytree_node=False, default=10800)
    FACE_OFF_FRAMES: int = struct.field(pytree_node=False, default=40)

    # Face-off layout. [x, y] = [col, row].
    FACEOFF_X: float = struct.field(pytree_node=False, default=79.0)
    FACEOFF_Y: float = struct.field(pytree_node=False, default=114.0)
    PLAYER_SKATER_X: float = struct.field(pytree_node=False, default=54.0)
    PLAYER_SKATER_Y: float = struct.field(pytree_node=False, default=84.0)
    PLAYER_GOALIE_X: float = struct.field(pytree_node=False, default=62.0)
    PLAYER_GOALIE_Y: float = struct.field(pytree_node=False, default=36.0)
    ENEMY_SKATER_X: float = struct.field(pytree_node=False, default=80.0)
    ENEMY_SKATER_Y: float = struct.field(pytree_node=False, default=105.0)
    ENEMY_GOALIE_X: float = struct.field(pytree_node=False, default=64.0)
    ENEMY_GOALIE_Y: float = struct.field(pytree_node=False, default=155.0)

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
    walk_counter: chex.Array    # leg walk-cycle phase counter (advances while moving)


@struct.dataclass
class PuckState:
    position: chex.Array        # float32 [x, y]
    velocity: chex.Array        # float32 [vx, vy]
    direction: chex.Array       # shot angle slot, 0-31
    position_stick: chex.Array  # slot on the stick arc while carried, 0-31


@struct.dataclass
class PlayerState:
    skater: CharacterState
    goalie: CharacterState
    active_character: chex.Array   # 0 = skater controlled, 1 = goalie controlled


@struct.dataclass
class EnemyState:
    skater: CharacterState
    goalie: CharacterState
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
            "player_skater": obj,
            "player_goalie": obj,
            "enemy_skater": obj,
            "enemy_goalie": obj,
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
        # Face-off: puck at centre, characters on start positions
        c = self.consts

        def char(x, y, orientation=0):
            return CharacterState(
                is_tackled=jnp.array(False),
                position=jnp.array([x, y], dtype=jnp.float32),
                orientation=jnp.array(orientation, dtype=jnp.int32),
                has_puck=jnp.array(False),
                shooting_cooldown=jnp.array(0, dtype=jnp.int32),
                walk_counter=jnp.array(0, dtype=jnp.int32),
            )

        state = IceHockeyState(
            player_state=PlayerState(
                skater=char(c.PLAYER_SKATER_X, c.PLAYER_SKATER_Y, orientation=1), # oriented right at start
                goalie=char(c.PLAYER_GOALIE_X, c.PLAYER_GOALIE_Y),
                active_character=jnp.array(0, dtype=jnp.int32),
            ),
            enemy_state=EnemyState(
                skater=char(c.ENEMY_SKATER_X, c.ENEMY_SKATER_Y),
                goalie=char(c.ENEMY_GOALIE_X, c.ENEMY_GOALIE_Y),
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
        previous_state = state

        new_player_state, new_enemy_state = self._characters_step(
            state.player_state,
            state.enemy_state,
            state.puck_state.position,
            player_action=action,
            enemy_action=jnp.array(Action.NOOP, dtype=jnp.int32),
        )
        state = state.replace(
            player_state=new_player_state,
            enemy_state=new_enemy_state,
            counter=state.counter + 1,
        )

        obs = self._get_observation(state)
        reward = self._get_reward(previous_state, state)
        done = self._get_done(state)
        info = self._get_info(state)
        return obs, state, reward, done, info

    def _resolve_active_character(
        self,
        char1: CharacterState,
        char2: CharacterState,
        puck_position: chex.Array,
        current_active: chex.Array,
    ) -> chex.Array:
        """Resolve which of two characters is closest to the puck.

        Control in Ice Hockey goes to whichever of a team's two skaters is closest to
        the puck, so ``_player_step`` and ``_enemy_step`` both call this on their own
        pair.

        Args:
            char1: First character (corresponds to index 0).
            char2: Second character (corresponds to index 1).
            puck_position: ``(x, y)`` position of the puck.
            current_active: Index (0 or 1) returned on an exact distance tie, which
                avoids the result flickering between equidistant characters.

        Returns:
            An ``int32`` scalar: 0 if ``char1`` is closer, 1 if ``char2`` is.
        """
        # Squared distance from each character to the puck; sqrt is unnecessary for ordering.
        dist1_sq = jnp.sum((char1.position - puck_position) ** 2)
        dist2_sq = jnp.sum((char2.position - puck_position) ** 2)

        # Closest character wins; on an exact tie keep whoever is currently active.
        closest = jnp.where(
            dist1_sq < dist2_sq,
            0,
            jnp.where(dist2_sq < dist1_sq, 1, current_active),
        )
        return closest.astype(jnp.int32)

    def _apply_action(
        self,
        character: CharacterState,
        action: chex.Array,
        bounds: chex.Array,
        velocity: chex.Array,
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
            bounds: ``(x_min, x_max, y_min, y_max)`` provisional wall/zone clamp (see above).
            velocity: Per-axis movement distance for this frame (e.g. ``PLAYER_SPEED``).

        Returns:
            The updated ``CharacterState`` (position + orientation; other fields kept).
        """
        up = jnp.any(jnp.array([
            action == Action.UP, action == Action.UPRIGHT, action == Action.UPLEFT,
            action == Action.UPFIRE, action == Action.UPRIGHTFIRE, action == Action.UPLEFTFIRE,
        ]))
        down = jnp.any(jnp.array([
            action == Action.DOWN, action == Action.DOWNRIGHT, action == Action.DOWNLEFT,
            action == Action.DOWNFIRE, action == Action.DOWNRIGHTFIRE, action == Action.DOWNLEFTFIRE,
        ]))
        left = jnp.any(jnp.array([
            action == Action.LEFT, action == Action.UPLEFT, action == Action.DOWNLEFT,
            action == Action.LEFTFIRE, action == Action.UPLEFTFIRE, action == Action.DOWNLEFTFIRE,
        ]))
        right = jnp.any(jnp.array([
            action == Action.RIGHT, action == Action.UPRIGHT, action == Action.DOWNRIGHT,
            action == Action.RIGHTFIRE, action == Action.UPRIGHTFIRE, action == Action.DOWNRIGHTFIRE,
        ]))

        # A tackled character is frozen: ignore all input movement this frame.
        movable = jnp.logical_not(character.is_tackled)
        dx = jnp.where(movable & right, velocity, jnp.where(movable & left, -velocity, 0.0))
        # Screen y grows downward, so DOWN increases y and UP decreases it.
        # NOTE: diagonals move by `velocity` on each axis, i.e. ~1.41x faster than a
        # straight move.    
        dy = jnp.where(movable & down, velocity, jnp.where(movable & up, -velocity, 0.0))

        new_x = jnp.clip(character.position[0] + dx, bounds[0], bounds[1])
        new_y = jnp.clip(character.position[1] + dy, bounds[2], bounds[3])
        new_position = jnp.array([new_x, new_y])

        # Orientation: 0 = facing left, 1 = facing right.
        # input keeps the current facing; a tackled character keeps it too (frozen).
        new_orientation = jnp.where(
            movable & right, 1, jnp.where(movable & left, 0, character.orientation)
        )

        # Leg walk-cycle advances whenever the skater has any directional input 
        # and freezes on frame 0 when idle (NOOP) or tackled.
        has_input = movable & (up | down | left | right)
        new_walk_counter = jnp.where(has_input, character.walk_counter + 1, 0)

        # Shooting/swing animation: a FIRE press starts the swing. 
        # shooting_cooldown counts the swing pose down to 0; 
        # a fresh press only (re)starts it when not already
        # swinging, so holding FIRE replays the full swing.
        # A tackled character cannot swing.
        fire = movable & jnp.any(jnp.array([
            action == Action.FIRE,
            action == Action.UPFIRE, action == Action.DOWNFIRE,
            action == Action.LEFTFIRE, action == Action.RIGHTFIRE,
            action == Action.UPRIGHTFIRE, action == Action.UPLEFTFIRE,
            action == Action.DOWNRIGHTFIRE, action == Action.DOWNLEFTFIRE,
        ]))
        decremented = jnp.maximum(character.shooting_cooldown - 1, 0)
        new_cooldown = jnp.where(
            fire & (character.shooting_cooldown == 0),
            self.consts.SHOOT_ANIM_FRAMES,
            decremented,
        )

        return character.replace(
            position=new_position,
            orientation=new_orientation,
            walk_counter=new_walk_counter,
            shooting_cooldown=new_cooldown,
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
        bounds1: chex.Array,
        bounds2: chex.Array,
        velocity: chex.Array,
    ) -> Tuple[CharacterState, CharacterState]:
        """Apply one team's chosen action as phase-1 intended movement.

        The reframed phase 1: instead of "only the active skater moves", every character
        is handled uniformly by the same ``_apply_action`` — the active skater receives
        the real action and the teammate receives ``NOOP`` (a zero intended delta). The
        active/passive split therefore collapses to "what action does this character get
        this frame", and the inactive teammate simply gets a no-op move.

        This is shared by both teams: the player's action comes from the agent, the
        computer's from its (future) policy, but routing + application are identical.

        Returns the two characters with their *provisional* (wall/zone-clamped) intended
        positions; the authoritative position is decided later by ``_resolve_interactions``.
        """
        action1 = jnp.where(active == 0, action, Action.NOOP)
        action2 = jnp.where(active == 1, action, Action.NOOP)
        return (
            self._apply_action(char1, action1, bounds1, velocity),
            self._apply_action(char2, action2, bounds2, velocity),
        )

    # ------------------------------------------------------------------ #
    # Phase 2 — interaction resolution (pure geometry, single fixed-order pass)
    # ------------------------------------------------------------------ #
    def _separate_opponents(
        self,
        pos_a: chex.Array,
        pos_b: chex.Array,
        min_separation: chex.Array,
    ) -> Tuple[chex.Array, chex.Array]:
        """Resolve a cross-team (opponent) overlap: the confirmed "both shift".

        If the two characters are closer than ``min_separation``, push BOTH apart along
        their centre-to-centre direction, each by half the penetration, so they end up
        exactly ``min_separation`` apart. The centre-to-centre normal is what makes the
        displacement diagonal. No-op when already separated.

        Pure geometry of the *post-move* positions only — it needs no pre-move state.

        Efficiency: the overlap *test* is done on squared distance (no sqrt). The push
        itself needs the true distance for the unit normal and the linear penetration,
        so a root is unavoidable here — but we fold sqrt + divide into a single
        reciprocal-sqrt: 1/dist == rsqrt(dist**2), giving
        ``offset = 0.5 * delta * (min_separation / dist - 1)``.

        NOTE: ``min_separation`` (derived from body sizes) is not yet finalised; passed
        in. Whether a tackled/downed character is still pushable is also unverified.
        """
        delta = pos_a - pos_b
        dist_sq = jnp.sum(delta ** 2)
        overlapping = dist_sq < min_separation ** 2

        # True distance is only needed via its reciprocal: 1/dist == rsqrt(dist**2).
        coincident = dist_sq <= 0.0
        inv_dist = jnp.where(coincident, 0.0, jax.lax.rsqrt(dist_sq))
        # offset = 0.5 * (min_sep - dist) * (delta / dist) = 0.5 * delta * (min_sep/dist - 1)
        offset = 0.5 * delta * (min_separation * inv_dist - 1.0)
        # Coincident centres give no direction: default to separating along +x / -x.
        offset = jnp.where(coincident, jnp.array([min_separation * 0.5, 0.0]), offset)
        offset = jnp.where(overlapping, offset, jnp.array([0.0, 0.0]))
        return pos_a + offset, pos_b - offset

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
        return jnp.array([
            jnp.clip(pos[0], bounds[0], bounds[1]),
            jnp.clip(pos[1], bounds[2], bounds[3]),
        ])

    def _resolve_interactions(
        self,
        player_skater: CharacterState,
        player_goalie: CharacterState,
        enemy_skater: CharacterState,
        enemy_goalie: CharacterState,
        player_active: chex.Array,
        enemy_active: chex.Array,
        min_separation: chex.Array,
        min_vertical_distance: chex.Array,
        bounds_p1: chex.Array,
        bounds_p2: chex.Array,
        bounds_e1: chex.Array,
        bounds_e2: chex.Array,
    ) -> Tuple[CharacterState, CharacterState, CharacterState, CharacterState]:
        """Phase 2: resolve all interactions on the four post-phase-1 characters.

        A single, fixed-order, fully branchless pass (deliberately NOT an iterative
        constraint solver — see the efficiency rationale: with 4 characters and shallow
        per-frame overlaps this is faithful and cheap):

          1. Opponent (cross-team) separations for the 4 player×enemy pairs.
          2. Same-team vertical pushes (active holds, passive yields).
          3. Authoritative wall/zone clamp on all four (covers pushes into walls and the
             passive teammate, which never passes through phase 1).

        The sub-step ORDER is a behavioural choice to verify against the game; in rare
        triple-contact frames a later step can nudge an earlier constraint sub-pixel,
        which the design guide tolerates.
        """
        p1, p2 = player_skater.position, player_goalie.position
        e1, e2 = enemy_skater.position, enemy_goalie.position

        # 1) Opponent collisions — both shift along centre-to-centre.
        p1, e1 = self._separate_opponents(p1, e1, min_separation)
        p1, e2 = self._separate_opponents(p1, e2, min_separation)
        p2, e1 = self._separate_opponents(p2, e1, min_separation)
        p2, e2 = self._separate_opponents(p2, e2, min_separation)

        # 2) Same-team vertical push — the active skater holds, the teammate yields.
        p2 = jnp.where(player_active == 0, self._enforce_min_vertical(p1, p2, min_vertical_distance), p2)
        p1 = jnp.where(player_active == 1, self._enforce_min_vertical(p2, p1, min_vertical_distance), p1)
        e2 = jnp.where(enemy_active == 0, self._enforce_min_vertical(e1, e2, min_vertical_distance), e2)
        e1 = jnp.where(enemy_active == 1, self._enforce_min_vertical(e2, e1, min_vertical_distance), e1)

        # 3) Authoritative clamp.
        p1 = self._clamp_to_bounds(p1, bounds_p1)
        p2 = self._clamp_to_bounds(p2, bounds_p2)
        e1 = self._clamp_to_bounds(e1, bounds_e1)
        e2 = self._clamp_to_bounds(e2, bounds_e2)

        return (
            player_skater.replace(position=p1),
            player_goalie.replace(position=p2),
            enemy_skater.replace(position=e1),
            enemy_goalie.replace(position=e2),
        )

    # ------------------------------------------------------------------ #
    # Orchestrator — runs phase 1 then phase 2 for all four characters
    # ------------------------------------------------------------------ #
    def _characters_step(
        self,
        player_state: PlayerState,
        enemy_state: EnemyState,
        puck_position: chex.Array,
        player_action: chex.Array,
        enemy_action: chex.Array,
    ) -> Tuple[PlayerState, EnemyState]:
        """Advance all four characters one frame: active resolution -> phase 1 -> phase 2.

        This is the character-movement orchestrator shared by both teams. The movement
        speed, collision tunables, and zone bounds are read from ``self.consts``;
        ``enemy_action`` will come from the (future) ``_enemy_policy`` and is ``NOOP``
        until then. The lower-level geometry primitives still take these as parameters so
        they stay generic/unit-testable — only this orchestrator binds them to consts.

        Steps:
          1. Resolve each team's active (controlled) skater = closest to the puck.
          2. Phase 1: apply each team's action as intended input movement (uniformly via
             ``_apply_team_inputs`` — active skater gets the action, teammate gets NOOP).
          3. Phase 2: resolve interactions across all four post-move characters
             (opponent separation, teammate vertical push, authoritative clamp).

        Returns the updated ``PlayerState`` and ``EnemyState`` (positions/orientation
        from movement, plus the resolved ``active_character`` for each team).
        """
        c = self.consts
        velocity = jnp.float32(c.PLAYER_SPEED)
        min_separation = jnp.float32(c.MIN_SEPARATION)
        min_vertical_distance = jnp.float32(c.MIN_VERTICAL_DISTANCE)
        x_min = c.RINK_LEFT
        x_max = c.RINK_RIGHT - c.PLAYER_W
        y_top = c.RINK_TOP
        y_bot = c.RINK_BOTTOM - c.PLAYER_H
        off = c.ATTACKING_ZONE_OFFSET_Y     # depth of the restricted zone
        crease = c.GOALIE_CREASE_DEPTH       # how far a goalie pokes into its goal
        # Player defends the TOP goal, enemy the BOTTOM. A skater is kept out of its
        # own defensive zone (so it plays toward the goal it attacks); a goalie stays
        # in its defensive half but may enter its own goal crease.
        bounds_player_skater = jnp.array(
            [x_min, x_max, y_top + off, y_bot], dtype=jnp.float32
        )
        bounds_player_goalie = jnp.array(
            [x_min, x_max, y_top - crease, y_bot - off], dtype=jnp.float32
        )
        bounds_enemy_skater = jnp.array(
            [x_min, x_max, y_top, y_bot - off], dtype=jnp.float32
        )
        bounds_enemy_goalie = jnp.array(
            [x_min, x_max, y_top + off, y_bot + crease], dtype=jnp.float32
        )

        # 1) Active-skater resolution (per team, against the shared puck).
        player_active = self._resolve_active_character(
            player_state.skater, player_state.goalie,
            puck_position, player_state.active_character,
        )
        enemy_active = self._resolve_active_character(
            enemy_state.skater, enemy_state.goalie,
            puck_position, enemy_state.active_character,
        )
        
        # 2) Phase 1 — intended input movement, uniform over each team's two skaters.
        p1, p2 = self._apply_team_inputs(
            player_state.skater, player_state.goalie,
            player_active, player_action, bounds_player_skater, bounds_player_goalie, velocity,
        )
        e1, e2 = self._apply_team_inputs(
            enemy_state.skater, enemy_state.goalie,
            enemy_active, enemy_action, bounds_enemy_skater, bounds_enemy_goalie, velocity,
        )

        # 3) Phase 2 — resolve interactions across all four post-move characters.
        p1, p2, e1, e2 = self._resolve_interactions(
            p1, p2, e1, e2,
            player_active, enemy_active,
            min_separation, min_vertical_distance,
            bounds_player_skater, bounds_player_goalie, bounds_enemy_skater, bounds_enemy_goalie,
        )

        new_player_state = player_state.replace(
            skater=p1, goalie=p2, active_character=player_active,
        )
        new_enemy_state = enemy_state.replace(
            skater=e1, goalie=e2, active_character=enemy_active,
        )
        return new_player_state, new_enemy_state

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
            player_skater=obj(state.player_state.skater.position, c.PLAYER_W, c.PLAYER_H),
            player_goalie=obj(state.player_state.goalie.position, c.PLAYER_W, c.PLAYER_H),
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

        return jnp.concatenate([
            flat(obs.player_skater), flat(obs.player_goalie),
            flat(obs.enemy_skater), flat(obs.enemy_goalie),
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

        puck_m = self.SHAPE_MASKS["puck"]

        # Skater sprites (all authored facing right; left-facing chars, orientation
        # 0, are mirrored). Per character the renderer picks one of three poses:
        #   - inactive teammate              -> "<team>_nostick" (no stick),
        #   - active skater standing still   -> "<team>_idle"    (with stick),
        #   - active skater moving           -> "<team>" walk-cycle frame.
        pm = self.SHAPE_MASKS["player"]
        em = self.SHAPE_MASKS["enemy"]
        p_idle = self.SHAPE_MASKS["player_idle"]
        e_idle = self.SHAPE_MASKS["enemy_idle"]
        p_nostick = self.SHAPE_MASKS["player_nostick"]
        e_nostick = self.SHAPE_MASKS["enemy_nostick"]
        p_shoot = self.SHAPE_MASKS["player_shoot"]
        e_shoot = self.SHAPE_MASKS["enemy_shoot"]
        p_off = self.FLIP_OFFSETS["player"]
        e_off = self.FLIP_OFFSETS["enemy"]
        cadence = self.consts.ANIM_CADENCE
        nframes = self.consts.ANIM_FRAMES

        def col(pos):
            return jnp.round(pos[0]).astype(jnp.int32)

        def row(pos):
            return jnp.round(pos[1]).astype(jnp.int32)

        def draw(r, char, move_masks, idle_mask, nostick_mask, shoot_masks, off, is_active):
            flip = char.orientation == 0   # 0 = left; sprites face right
            frame = (char.walk_counter // cadence) % nframes
            moving = char.walk_counter > 0
            shooting = char.shooting_cooldown > 0
            # Swing animation advances over the cooldown: wind-up (frame 0) then
            # follow-through (frame 1). Count elapsed frames up from the press.
            elapsed = self.consts.SHOOT_ANIM_FRAMES - char.shooting_cooldown
            shoot_frame = (elapsed // cadence) % shoot_masks.shape[0]
            # active skater: swing animation while shooting, else walk frame while
            # moving, else idle pose when still.
            active_mask = jnp.where(
                shooting, shoot_masks[shoot_frame],
                jnp.where(moving, move_masks[frame], idle_mask),
            )
            # inactive teammate: the stickless pose.
            mask = jnp.where(is_active, active_mask, nostick_mask)
            return self.jr.render_at_clipped(
                r, col(char.position), row(char.position), mask,
                flip_horizontal=flip, flip_offset=off,
            )

        # Active character of each team (0 = skater controlled, 1 = goalie).
        p_act = state.player_state.active_character
        e_act = state.enemy_state.active_character

        # render_at_clipped because skaters can reach the board pixels at the
        # edge; render_at would slice out of bounds there.
        raster = draw(raster, state.player_state.goalie, pm, p_idle, p_nostick, p_shoot, p_off, p_act == 1)
        raster = draw(raster, state.enemy_state.goalie, em, e_idle, e_nostick, e_shoot, e_off, e_act == 1)
        raster = draw(raster, state.player_state.skater, pm, p_idle, p_nostick, p_shoot, p_off, p_act == 0)
        raster = draw(raster, state.enemy_state.skater, em, e_idle, e_nostick, e_shoot, e_off, e_act == 0)
        raster = self.jr.render_at_clipped(raster, col(state.puck_state.position),
                                           row(state.puck_state.position), puck_m)

        dm_blue = self.SHAPE_MASKS["digits"]        # player score (blue team)
        dm_gold = self.SHAPE_MASKS["digits_gold"]   # enemy score (gold team)

        def draw_score(r, value, x_single, x_double, dm):
            digits = self.jr.int_to_digits(value, max_digits=2)
            is_single = value < 10
            start = jax.lax.select(is_single, jnp.int32(1), jnp.int32(0))
            count = jax.lax.select(is_single, jnp.int32(1), jnp.int32(2))
            x = jax.lax.select(is_single, jnp.int32(x_single), jnp.int32(x_double))
            return self.jr.render_label_selective(
                r, x, 3, digits, dm, start, count, spacing=7, max_digits_to_render=2
            )

        # Blue (player) score on the left, gold (enemy) on the right, as in the ROM.
        raster = draw_score(raster, state.game_state.player_score, 43, 33, dm_blue)
        raster = draw_score(raster, state.game_state.enemy_score, 113, 103, dm_gold)

        return self.jr.render_from_palette(raster, self.PALETTE)