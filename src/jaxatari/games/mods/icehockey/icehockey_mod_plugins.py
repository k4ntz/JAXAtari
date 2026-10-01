import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import jaxatari.games
from jaxatari.environment import JAXAtariAction as Action
from jaxatari.games.jax_icehockey import IceHockeyConstants
from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin

# Palette of the base background sprite: grey ice, black boards/goal mouths.
_ICE = np.array([192, 192, 192, 255], dtype=np.uint8)
_BOARDS = np.array([0, 0, 0, 255], dtype=np.uint8)


def _load_base_background() -> np.ndarray:
    sprite_path = os.path.join(
        os.path.dirname(jaxatari.games.__file__),
        "sprites",
        "icehockey",
        "background.npy",
    )
    return np.load(sprite_path).copy()


def _rink_corners(c):
    """The four rink corners as (sx, sy, ox, oy).

    edge_sum = sx*(x-ox) + sy*(y-oy) is the summed distance from the two
    straight boards meeting at that corner; (sx, sy)/sqrt2 points from the
    corner back into the rink. Shared by the octagon physics and its background
    so the two can never disagree.
    """
    return (
        (1.0, 1.0, c.RINK_LEFT, c.RINK_TOP),  # top-left
        (-1.0, 1.0, c.RINK_RIGHT, c.RINK_TOP),  # top-right
        (1.0, -1.0, c.RINK_LEFT, c.RINK_BOTTOM),  # bottom-left
        (-1.0, -1.0, c.RINK_RIGHT, c.RINK_BOTTOM),  # bottom-right
    )


def _make_resized_goal_background(new_x0: int, new_x1: int) -> np.ndarray:
    """Load the icehockey background and redraw both goal mouths as [new_x0, new_x1).

    The rink is baked into the background sprite: each goal is a black notch in
    the ice (rows PLAYER_GOAL_Y..+GOAL_HEIGHT_TOP at the top,
    ENEMY_GOAL_Y-GOAL_HEIGHT_BOTTOM+1..ENEMY_GOAL_Y at the bottom, columns
    GOAL_X0..GOAL_X1). The two goals are not equally deep, so each side uses its
    own base-game height constant. The old notch is filled with ice and the new
    one cut as boards, so this works for both narrower and wider goals.
    """
    c = IceHockeyConstants()
    bg = _load_base_background()
    top_rows = slice(c.PLAYER_GOAL_Y, c.PLAYER_GOAL_Y + c.GOAL_HEIGHT_TOP)
    bottom_rows = slice(c.ENEMY_GOAL_Y - c.GOAL_HEIGHT_BOTTOM + 1, c.ENEMY_GOAL_Y + 1)
    for rows in (top_rows, bottom_rows):
        bg[rows, c.GOAL_X0 : c.GOAL_X1] = _ICE
        bg[rows, new_x0:new_x1] = _BOARDS
    return bg


def _make_octagon_background(cut: float) -> np.ndarray:
    """Load the icehockey background and paint the four corner wedges as boards.

    A rink pixel belongs to a wedge when its edge_sum for that corner is below
    ``cut`` -- exactly where RoundedCornersMod._puck_step pushes the puck out.
    Generated from the base sprite (instead of shipping a copy) so goal or board
    changes in the base background carry over automatically.
    """
    c = IceHockeyConstants()
    bg = _load_base_background()
    ys, xs = np.mgrid[0 : bg.shape[0], 0 : bg.shape[1]]
    in_rink = (
        (xs >= c.RINK_LEFT)
        & (xs <= c.RINK_RIGHT)
        & (ys >= c.RINK_TOP)
        & (ys <= c.RINK_BOTTOM)
    )
    in_wedge = np.zeros(bg.shape[:2], dtype=bool)
    for sx, sy, ox, oy in _rink_corners(c):
        in_wedge |= sx * (xs - ox) + sy * (ys - oy) < cut
    is_ice = (bg == _ICE).all(axis=-1)
    bg[in_rink & in_wedge & is_ice] = _BOARDS
    return bg


# --- 1. Individual Mod Plugins ---
class NoAttackingZonesMod(JaxAtariInternalModPlugin):
    """Removes the attacking zones: every character may skate the whole rink.

    The base game ties each character to its zone in four places, all lifted here:

    1. Movement bands. _character_bounds derives an upper and a lower band from
       CHARACTER_GRID_Y_ORIGIN and the UPPER/LOWER_CHARACTER_GRID_Y_MIN/_MAX
       pairs. Setting both bands to their union lets all four characters reach
       every spot any character could reach in the base game.
    2. Teammate spacing. _finalize_character_positions pushes the passive
       teammate away until the pair is MIN_VERTICAL_DISTANCE apart. With a
       distance of 0 that check (|dy| < 0) never fires, so teammates may stand
       side by side or on top of each other.
    3. Control zones. _resolve_active_characters hands control to the skater or
       goalie by the puck's zone (ACTIVE_BOTTOM/TOP_THRESHOLD) instead of by
       distance. Puck grid rows are 0..255, so thresholds of 0 and 256 never
       match and control always goes to the teammate nearest the puck. On top of
       that the puck carrier always keeps control, so it can never be left
       frozen while its teammate is the one being steered.
    4. Teammate steals. With no spacing, a teammate's pickup box can overlap the
       carrier's puck and _puck_pickup would let it steal. The patch below keeps
       the carrier's team out of that check, so only opponents can take a held
       puck. A free puck (passes, rebounds) can still be collected by anyone.
    """

    _BASE = IceHockeyConstants()
    _GRID_Y_MIN = min(
        _BASE.UPPER_CHARACTER_GRID_Y_MIN, _BASE.LOWER_CHARACTER_GRID_Y_MIN
    )
    _GRID_Y_MAX = max(
        _BASE.UPPER_CHARACTER_GRID_Y_MAX, _BASE.LOWER_CHARACTER_GRID_Y_MAX
    )

    constants_overrides = {
        "UPPER_CHARACTER_GRID_Y_MIN": _GRID_Y_MIN,
        "UPPER_CHARACTER_GRID_Y_MAX": _GRID_Y_MAX,
        "LOWER_CHARACTER_GRID_Y_MIN": _GRID_Y_MIN,
        "LOWER_CHARACTER_GRID_Y_MAX": _GRID_Y_MAX,
        "MIN_VERTICAL_DISTANCE": 0.0,
        "ACTIVE_BOTTOM_THRESHOLD": 0,
        "ACTIVE_TOP_THRESHOLD": 256,
    }

    @partial(jax.jit, static_argnums=(0,))
    def _resolve_active_characters(self, player_state, enemy_state, puck_position):
        # Call the class implementation: the instance attribute is this patch.
        player_active, enemy_active = type(self._env)._resolve_active_characters(
            self._env, player_state, enemy_state, puck_position
        )

        def keep_carrier(team, active):
            # A carrier is never knocked down (_knock_down drops the puck), but
            # the guard keeps the base rule "a downed character hands over
            # control" authoritative even so.
            active = jnp.where(
                team.skater.has_puck & ~team.skater.is_tackled, jnp.int32(0), active
            )
            active = jnp.where(
                team.goalie.has_puck & ~team.goalie.is_tackled, jnp.int32(1), active
            )
            return active.astype(jnp.int32)

        return (
            keep_carrier(player_state, player_active),
            keep_carrier(enemy_state, enemy_active),
        )

    @partial(jax.jit, static_argnums=(0,))
    def _puck_pickup(self, player_state, enemy_state, puck_state, random_byte):
        # The base pickup never hands the puck to a knocked-down character, and
        # is_tackled is read there for nothing else. Marking the team that holds
        # the puck as knocked down for this one call therefore bars exactly that
        # team from stealing it, without touching pickups of a free puck or
        # steals by the opponents. The real is_tackled flags are restored after.
        def shield(team):
            team_holds = team.skater.has_puck | team.goalie.has_puck
            return team.replace(
                skater=team.skater.replace(
                    is_tackled=team.skater.is_tackled | team_holds
                ),
                goalie=team.goalie.replace(
                    is_tackled=team.goalie.is_tackled | team_holds
                ),
            )

        def restore(result, original):
            return result.replace(
                skater=result.skater.replace(is_tackled=original.skater.is_tackled),
                goalie=result.goalie.replace(is_tackled=original.goalie.is_tackled),
            )

        new_player, new_enemy, new_puck = type(self._env)._puck_pickup(
            self._env,
            shield(player_state),
            shield(enemy_state),
            puck_state,
            random_byte=random_byte,
        )
        return (
            restore(new_player, player_state),
            restore(new_enemy, enemy_state),
            new_puck,
        )


class DisableTacklingMod(JaxAtariInternalModPlugin):
    """Disables body-checks: no character can ever be knocked down.

    Every check in the base game is resolved in _contact_pair, where a swing only
    lands on a partner that is not protected (the goalie-crease rule). This patch
    runs the unmodified base _contact_pair with both characters marked as
    protected, so a swing never knocks anyone down or strips the puck. The
    ordinary contact push between characters, the FIRE swing and shooting stay
    exactly as in the base game.
    """

    @partial(jax.jit, static_argnums=(0,))
    def _contact_pair(
        self,
        first,
        second,
        puck_state,
        random_byte,
        protect_first=False,
        protect_second=False,
    ):
        # Call the class implementation: the instance attribute is this patch.
        return type(self._env)._contact_pair(
            self._env,
            first,
            second,
            puck_state,
            random_byte,
            protect_first=True,
            protect_second=True,
        )


class DecreasedGoalSizeMod(JaxAtariInternalModPlugin):
    """Halves the width of both goals (mouth 64..96 -> 72..88, centred).

    GOAL_X0/GOAL_X1 drive goal detection in _goal_and_reset_step and the rigid
    goal posts in _advance_puck_with_walls, so shots outside the narrowed mouth
    now bounce off the posts/boards instead of scoring. The background asset is
    rebuilt with the covered goal columns filled in as ice so the visuals match
    the new geometry.
    """

    _NEW_GOAL_X0 = 72
    _NEW_GOAL_X1 = 88

    constants_overrides = {
        "GOAL_X0": _NEW_GOAL_X0,
        "GOAL_X1": _NEW_GOAL_X1,
    }
    asset_overrides = {
        "background": {
            "name": "background",
            "type": "background",
            "data": _make_resized_goal_background(_NEW_GOAL_X0, _NEW_GOAL_X1),
        }
    }


class IncreasedGoalSizeMod(JaxAtariInternalModPlugin):
    """Doubles the width of both goals (mouth 64..96 -> 48..112, centred).

    Counterpart to DecreasedGoalSizeMod: GOAL_X0/GOAL_X1 drive goal detection in
    _goal_and_reset_step and the rigid goal posts in _advance_puck_with_walls, so
    the wider mouth scores and its posts sit at the new edges. The background
    asset is rebuilt with the extra goal columns cut in as boards so the visuals
    match the new geometry. The new mouth keeps 16px of ice to each side board
    (RINK_LEFT=32, RINK_RIGHT=128) so the posts stay reachable from outside.
    """

    _NEW_GOAL_X0 = 48
    _NEW_GOAL_X1 = 112

    constants_overrides = {
        "GOAL_X0": _NEW_GOAL_X0,
        "GOAL_X1": _NEW_GOAL_X1,
    }
    asset_overrides = {
        "background": {
            "name": "background",
            "type": "background",
            "data": _make_resized_goal_background(_NEW_GOAL_X0, _NEW_GOAL_X1),
        }
    }


class TackleSlowdownMod(JaxAtariInternalModPlugin):
    """Characters get permanently slower each time they are tackled.

    times_tackled is maintained by the base game (and carried across face-offs),
    so it is a per-match knockdown count. Each knockdown multiplies that
    character's movement by SLOWDOWN_PER_TACKLE, down to MIN_SPEED_FACTOR.

    The base _apply_action moves a character at the fixed CHARACTER_SPEED_X/Y and
    accepts no velocity, so rather than reimplementing movement this lets the
    base primitive move the character normally and then shrinks the displacement
    it produced. Freezing while tackled, orientation and the walk cycle therefore
    stay exactly as the base game computes them.
    """

    SLOWDOWN_PER_TACKLE = 0.8  # speed multiplier applied per suffered knockdown
    MIN_SPEED_FACTOR = 0.25  # lower bound on the accumulated slowdown

    @partial(jax.jit, static_argnums=(0,))
    def _apply_team_inputs(self, char1, char2, active, action):
        # Same routing as the base implementation: the active character gets the
        # real action, the teammate a NOOP.
        action1 = jnp.where(active == 0, action, Action.NOOP)
        action2 = jnp.where(active == 1, action, Action.NOOP)

        def slowed(char, char_action):
            moved = self._env._apply_action(char, char_action)
            factor = jnp.float32(self.SLOWDOWN_PER_TACKLE) ** char.times_tackled.astype(
                jnp.float32
            )
            factor = jnp.maximum(factor, jnp.float32(self.MIN_SPEED_FACTOR))
            delta = moved.position - char.position
            return moved.replace(position=char.position + delta * factor)

        return slowed(char1, action1), slowed(char2, action2)


def _goal_offset_x(consts, remaining_time, amplitude, speed=0.15):
    """Signed triangle-wave offset from the rink's horizontal center, starting at 0.

    Shape over one period (4*amplitude/speed frames): 0 -> +amplitude -> 0 ->
    -amplitude -> 0. Driven by the game clock (remaining_time counts down only
    while play is active) so it freezes in sync with everything else during
    face-offs/goal pauses, and both the physics and render-side callers agree
    on the exact same value every frame.
    """
    t = (consts.TIME_LIMIT - remaining_time).astype(jnp.float32)
    x = jnp.mod(t * speed, 4.0 * amplitude)
    return jnp.where(
        x < amplitude,
        x,
        jnp.where(x < 3.0 * amplitude, 2.0 * amplitude - x, x - 4.0 * amplitude),
    )


def _moving_post_collision(c, position, velocity, top_x0, bottom_x0, goal_width):
    """Bounce an already-moved free puck off the posts of the moving goals.

    Same rule as the fixed posts in the base _advance_puck_with_walls: inside the
    goal-line rows a post (at x0 and x0 + goal_width) only stops a puck entering
    the mouth from outside, which is then put on the post with its x velocity
    reversed. The puck physics never sees the game clock that moves the goals, so
    this runs after the puck has moved and takes position - velocity as the spot
    it came from.
    """
    in_top_band = position[1] <= c.RINK_TOP + c.GOAL_HEIGHT_TOP - 1
    in_bottom_band = position[1] >= c.RINK_BOTTOM - (c.GOAL_HEIGHT_BOTTOM - 1)
    x0 = jnp.where(in_top_band, top_x0, bottom_x0)
    x1 = x0 + goal_width

    x, vx = position[0], velocity[0]
    came_from = x - vx
    in_band = in_top_band | in_bottom_band
    hit_left = in_band & (came_from <= x0) & (x > x0)
    hit_right = in_band & (came_from >= x1) & (x < x1)

    new_x = jnp.where(hit_left, x0, jnp.where(hit_right, x1, x))
    new_vx = jnp.where(hit_left | hit_right, -vx, vx)
    return (
        jnp.array([new_x, position[1]], dtype=jnp.float32),
        jnp.array([new_vx, velocity[1]], dtype=jnp.float32),
    )


class RoundedCornersMod(JaxAtariInternalModPlugin):
    """Cuts the four rink corners off diagonally, turning the rink into an octagon.

    _puck_step keeps the base physics (friction, straight boards, goal posts) and
    additionally reflects the puck off the diagonal corner walls. The background
    is generated from the base sprite with the same corner geometry.
    """

    CORNER_CUT = 16.0
    asset_overrides = {
        "background": {
            "name": "background",
            "type": "background",
            "data": _make_octagon_background(CORNER_CUT),
        }
    }

    @partial(jax.jit, static_argnums=(0,))
    def _puck_step(self, puck):
        env = self._env
        cut = self.CORNER_CUT

        # Base puck update first, so friction and the straight boards and goal
        # posts behave exactly as in the unmodded game.
        vel = env._decay_puck_velocity(puck.velocity)
        pos, vel = env._advance_puck_with_walls(puck.position, vel)

        # Diagonal walls at the four corners (see _rink_corners).
        inv_sqrt2 = 0.70710678
        for sx, sy, ox, oy in _rink_corners(env.consts):
            n_in = jnp.array([sx, sy], dtype=jnp.float32) * inv_sqrt2
            edge_sum = sx * (pos[0] - ox) + sy * (pos[1] - oy)
            penetration = cut - edge_sum
            penetrating = penetration > 0.0
            # Only reflect velocity if the puck is actually heading further into the
            # wedge; if it's already moving back out (e.g. repositioned there by a
            # face-off/pickup), leave velocity alone so it isn't sent back in.
            approaching = jnp.dot(vel, n_in) < 0.0
            hit = penetrating & approaching

            pos = jnp.where(penetrating, pos + penetration * n_in, pos)
            vel = jnp.where(hit, vel - 2.0 * jnp.dot(vel, n_in) * n_in, vel)

        return puck.replace(position=pos, velocity=vel)


class MovingGoalsMod(JaxAtariInternalModPlugin):
    """Both goal mouths are smaller, start centered, and slide back and forth
    along the boards in opposite directions (when one moves left, the other
    moves right).

    Patches _goal_and_reset_step (env) so scoring is checked against each
    goal's own current dynamic x0/x1 instead of the fixed GOAL_X0/GOAL_X1, and
    _render_hook_post_background (renderer) so the drawn notches follow along.
    Both read the same _goal_offset_x(consts, remaining_time, ...) so they can
    never desync.

    The goal posts move along too: _advance_puck_with_walls keeps only the
    straight boards, and _goal_and_reset_step bounces the puck off the posts at
    each goal's current position (see _moving_post_collision).
    """

    GOAL_WIDTH = 22.0  # smaller than the original GOAL_X1 - GOAL_X0 (32px)
    AMPLITUDE = 32.0  # max distance (px) each goal travels from rink center

    @partial(jax.jit, static_argnums=(0,))
    def _advance_puck_with_walls(self, position, velocity):
        # Straight boards only, as in the base game. Its rigid posts sit at the
        # fixed GOAL_X0/GOAL_X1 and would be invisible walls next to the moving
        # goals; the moving posts are handled in _goal_and_reset_step.
        c = self._env.consts
        tentative = position + velocity
        low = jnp.array([c.RINK_LEFT, c.RINK_TOP], dtype=jnp.float32)
        high = jnp.array([c.RINK_RIGHT, c.RINK_BOTTOM], dtype=jnp.float32)
        hit = (tentative < low) | (tentative > high)
        return jnp.clip(tentative, low, high), jnp.where(hit, -velocity, velocity)

    @partial(jax.jit, static_argnums=(0,))
    def _goal_and_reset_step(
        self, game_state, player_state, enemy_state, puck_state, frozen, random_byte
    ):
        env = self._env
        c = env.consts

        mid_x = (c.RINK_LEFT + c.RINK_RIGHT) / 2.0
        half_width = self.GOAL_WIDTH / 2.0
        offset = _goal_offset_x(c, game_state.remaining_time, self.AMPLITUDE)

        # Top goal (defended by player, scored into by enemy) and bottom goal
        # (defended by enemy, scored into by player) move in opposite directions.
        top_center = mid_x + offset
        bottom_center = mid_x - offset

        # Moving posts first, like the base game resolves its posts before scoring.
        carried = (
            player_state.skater.has_puck
            | player_state.goalie.has_puck
            | enemy_state.skater.has_puck
            | enemy_state.goalie.has_puck
        )
        free = ~frozen & ~carried
        post_pos, post_vel = _moving_post_collision(
            c,
            puck_state.position,
            puck_state.velocity,
            top_center - half_width,
            bottom_center - half_width,
            self.GOAL_WIDTH,
        )
        puck_pos = jnp.where(free, post_pos, puck_state.position)
        puck_vel = jnp.where(free, post_vel, puck_state.velocity)

        in_top_goal = (jnp.abs(puck_pos[0] - top_center) <= half_width) & (
            puck_pos[1] <= c.PLAYER_GOAL_Y
        )
        in_bottom_goal = (jnp.abs(puck_pos[0] - bottom_center) <= half_width) & (
            puck_pos[1] >= c.ENEMY_GOAL_Y
        )

        # Scoring, clock, pause phases and the face-off reset all stay in the base
        # implementation, which only reads the puck position for its fixed-mouth
        # goal test. So hand it a proxy puck that sits in the fixed mouth on the
        # matching goal line exactly when the puck is inside a moving goal, and at
        # centre ice otherwise. The base game can only reset positions on a frozen
        # frame (goal pause over), where no goal can be scored and the real puck is
        # passed through untouched.
        base_mouth_x = (c.GOAL_X0 + c.GOAL_X1) / 2.0
        proxy_y = jnp.where(
            in_bottom_goal,
            jnp.float32(c.ENEMY_GOAL_Y),
            jnp.where(
                in_top_goal,
                jnp.float32(c.PLAYER_GOAL_Y),
                jnp.float32((c.RINK_TOP + c.RINK_BOTTOM) / 2.0),
            ),
        )
        proxy_pos = jnp.array([base_mouth_x, proxy_y], dtype=jnp.float32)
        proxy_puck = puck_state.replace(
            position=jnp.where(frozen, puck_pos, proxy_pos), velocity=puck_vel
        )

        player_state, enemy_state, new_puck, game_state = type(
            env
        )._goal_and_reset_step(
            env,
            game_state,
            player_state,
            enemy_state,
            proxy_puck,
            frozen,
            random_byte=random_byte,
        )
        new_puck = new_puck.replace(
            position=jnp.where(frozen, new_puck.position, puck_pos)
        )
        return player_state, enemy_state, new_puck, game_state

    @partial(jax.jit, static_argnums=(0,))
    def _render_hook_post_background(self, raster, state):
        env = self._env
        c = env.consts
        jr = env.renderer.jr
        ice_id = env.renderer.COLOR_TO_ID[(192, 192, 192)]
        board_id = env.renderer.COLOR_TO_ID[(0, 0, 0)]
        goal_width = c.GOAL_X1 - c.GOAL_X0  # width of the static art being erased

        mid_x = (c.RINK_LEFT + c.RINK_RIGHT) / 2.0
        offset = _goal_offset_x(c, state.game_state.remaining_time, self.AMPLITUDE)
        top_x0 = mid_x + offset - self.GOAL_WIDTH / 2.0
        bottom_x0 = mid_x - offset - self.GOAL_WIDTH / 2.0

        # Close the two static goal notches baked into the background, refilling
        # with ice. Row RINK_TOP-1 and row RINK_BOTTOM are boards across the whole
        # rink width, so the ice band is [RINK_TOP, RINK_BOTTOM - 1]. The two
        # notches are not equally deep: the top one spans
        # [RINK_TOP, RINK_TOP+GOAL_HEIGHT_TOP) and the bottom one
        # [RINK_BOTTOM-GOAL_HEIGHT_BOTTOM, RINK_BOTTOM).
        close_positions = jnp.array(
            [
                [c.GOAL_X0 - 1.0, c.RINK_TOP],
                [c.GOAL_X0 - 1.0, c.RINK_BOTTOM - c.GOAL_HEIGHT_BOTTOM],
            ],
            dtype=jnp.float32,
        )
        close_sizes = jnp.array(
            [
                [goal_width + 2.0, c.GOAL_HEIGHT_TOP],
                [goal_width + 2.0, c.GOAL_HEIGHT_BOTTOM],
            ],
            dtype=jnp.float32,
        )
        raster = jr.draw_rects(raster, close_positions, close_sizes, ice_id)

        # Cut new, smaller notches at each goal's own current dynamic position.
        xs = jnp.stack([top_x0, bottom_x0])
        ys = jnp.array(
            [float(c.RINK_TOP), float(c.RINK_BOTTOM - c.GOAL_HEIGHT_BOTTOM)],
            dtype=jnp.float32,
        )
        open_positions = jnp.stack([xs, ys], axis=1)
        open_sizes = jnp.array(
            [
                [self.GOAL_WIDTH, c.GOAL_HEIGHT_TOP],
                [self.GOAL_WIDTH, c.GOAL_HEIGHT_BOTTOM],
            ],
            dtype=jnp.float32,
        )
        raster = jr.draw_rects(raster, open_positions, open_sizes, board_id)

        return raster


class PlayerSlidingMod(JaxAtariInternalModPlugin):

    # Closer to 1.0 = slides longer and resists direction changes more.
    FRICTION_COEFF = 0.92
    MIN_SLIDE_SPEED = 0.15

    @partial(jax.jit, static_argnums=(0,))
    def _apply_action(self, character, action):
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

        # A tackled character is frozen: ignore input and kill any glide outright.
        movable = jnp.logical_not(character.is_tackled)

        # Target velocity from raw input: full speed on a pressed axis, else 0.
        # Note this is the same target regardless of the character's *current*
        # velocity, so pressing the opposite direction targets -speed even while
        # still sliding the old way — the blend below is what makes that a slide
        # instead of an instant reversal. The base game moves at the fixed
        # CHARACTER_SPEED_X/Y per update, so those are the glide target speeds.
        speed_x = jnp.float32(self._env.consts.CHARACTER_SPEED_X)
        speed_y = jnp.float32(self._env.consts.CHARACTER_SPEED_Y)
        target_vx = jnp.where(right, speed_x, jnp.where(left, -speed_x, 0.0))
        target_vy = jnp.where(down, speed_y, jnp.where(up, -speed_y, 0.0))
        target = jnp.array([target_vx, target_vy], dtype=jnp.float32)

        # Blend current velocity toward the target (exponential decay of the
        # difference). Once within MIN_SLIDE_SPEED of the target, snap to it
        # exactly so the character doesn't asymptotically creep forever.
        blended = target + (character.velocity - target) * self.FRICTION_COEFF
        residual = jnp.linalg.norm(blended - target)
        settled_velocity = jnp.where(residual > self.MIN_SLIDE_SPEED, blended, target)

        new_velocity = jnp.where(
            movable, settled_velocity, jnp.zeros(2, dtype=jnp.float32)
        )

        # No clamping here: like the base _apply_action this only produces the
        # intended movement; the base game clamps to the character's bounds
        # afterwards in _finalize_character_positions.
        new_position = character.position + new_velocity

        # Orientation: 0 = facing left, 1 = facing right. Input keeps the current
        # facing; a tackled character keeps it too (frozen).
        new_orientation = jnp.where(
            movable & right, 1, jnp.where(movable & left, 0, character.orientation)
        )

        # Leg walk-cycle advances whenever the character is actually moving
        # (either from input or still gliding) and freezes once it comes to rest.
        has_motion = movable & (jnp.linalg.norm(new_velocity) > self.MIN_SLIDE_SPEED)
        new_walk_counter = jnp.where(has_motion, character.walk_counter + 1, 0)

        return character.replace(
            position=new_position,
            velocity=new_velocity,
            orientation=new_orientation,
            walk_counter=new_walk_counter,
        )


class EnemySpeedUpMod(JaxAtariPostStepModPlugin):
    """Enemy skaters get progressively faster every time the player scores.

    Runs after the base step, which has already moved every character at the
    fixed base speed and resolved this frame's goal/reset logic. Rather than
    reaching into how speed is computed internally, this takes the enemy
    characters' already-computed per-frame displacement (new position minus
    previous position -- naturally zero while tackled/stationary) and stretches
    it by an extra factor derived from the player's cumulative goal count, then
    re-clips to the base game's _character_bounds. Frames that start frozen
    (face-off, goal pause) are left alone, so the snap back to the face-off spots
    is not stretched. This deliberately avoids ``CharacterState.velocity``, which
    ``PlayerSlidingMod`` repurposes for actual sliding physics, so the two mods
    don't fight over the same storage.
    """

    SPEED_INCREASE_PER_GOAL = 0.15  # +15% enemy speed per player goal
    MAX_SPEED_MULTIPLIER = 2.5  # cap so it stays playable at high scores

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        prev_gs = prev_state.game_state
        was_frozen = prev_gs.is_finished | prev_gs.goal_scored | prev_gs.is_faceoff

        multiplier = jnp.minimum(
            1.0
            + self.SPEED_INCREASE_PER_GOAL
            * new_state.game_state.player_score.astype(jnp.float32),
            jnp.float32(self.MAX_SPEED_MULTIPLIER),
        )
        extra_scale = jnp.where(was_frozen, 0.0, multiplier - 1.0)

        # Authoritative per-character (x_min, x_max, y_min, y_max) bounds.
        _, _, skater_bounds, goalie_bounds = self._env._character_bounds()

        def boosted(prev_char, new_char, bounds):
            delta = new_char.position - prev_char.position
            pos = new_char.position + delta * extra_scale
            return new_char.replace(position=self._env._clamp_to_bounds(pos, bounds))

        new_enemy_state = new_state.enemy_state.replace(
            skater=boosted(
                prev_state.enemy_state.skater,
                new_state.enemy_state.skater,
                skater_bounds,
            ),
            goalie=boosted(
                prev_state.enemy_state.goalie,
                new_state.enemy_state.goalie,
                goalie_bounds,
            ),
        )
        return new_state.replace(enemy_state=new_enemy_state)
