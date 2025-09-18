from dataclasses import dataclass

from keyboard import press
from numpy.ma.core import floor_divide

from jaxatari.renderers import JAXGameRenderer
from typing import NamedTuple, Tuple, TypeVar
from jax import Array, jit, random, numpy as jnp, debug
from functools import partial
from jaxatari.rendering import jax_rendering_utils as jr
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
import jaxatari.spaces as spaces
import jax.lax
from treescope import active_renderer

SIDE_TOP, SIDE_BOTTOM, SIDE_LEFT, SIDE_RIGHT = 0, 1, 2, 3


class TronConstants(NamedTuple):
    screen_width: int = 160
    screen_height: int = 210
    scaling_factor: int = 3

    # Player
    player_height: int = 10
    player_width: int = 10
    player_speed: int = 1  # Player speed in pixels per frame
    player_lives: int = 3  # starting number of lives

    # discs
    max_discs: int = 4  # Number of disc slots
    disc_size: int = 4  # Disc sprite is square, this is both width/height in pixels
    disc_speed: int = 2  # outbound (thrown) speed
    inbound_disc_speed: int = 4

    """
    Origin (0,0) is the top-left of the full screen.
    -------
    gamefield_rect = (game_x, game_y, game_w, game_h)
        The gray background area that contains everything else (scorebar + border + play area
        
    scorebar_rect = (game_x, game_y, game_w, score_h)
        A green strip at the top of the gamefield used for displaying the score
        
    Puple border (thickness per side)
        top: y = game_y + score_h + score_gap, height = bord_top,   width = game_w - bord_right
        bottom: y = game_y + game_h - bord_bot,   height = bord_bot,   width = game_w - bord_right
        left: x = game_x, width  = bord_left,  spans vertically between top/bottom bars
        right: x = game_x + game_w - bord_right, width  = bord_right, same vertical span as left

    Inner play area (actors must stay inside this)
        inner_play_rect = (
            game_x + bord_left,
            game_y + score_h + score_gap + bord_top,
            game_w - (bord_left + bord_right),
            game_h - (score_h + score_gap + bord_top + bord_bot),
    """
    game_x: int = 8  # Gamefield top-left X (creates a black left margin)
    game_y: int = 18  # Gamefield top-left Y (creates a black top margin)
    game_w: int = (
        160 - 8
    )  # Gamefield width; aligns right edge with the screens right edge
    game_h: int = 164  # Gamefield height in pixels

    # Scorebar
    score_h: int = 10  # Scorebar height in pixels
    score_gap: int = 1  # Vertical gap between scorebar bottom and the top purple border

    # purple border (thickness per side in pixels)
    bord_top: int = 16  # Height of the top purple band
    bord_bot: int = 16  # height of the bottom purple band
    bord_left: int = 8  # Width of the left purple band
    bord_right: int = 8  # Width of the right purple band

    # doors
    max_doors: int = 12  # 4 top, 4 bottom, 2 left, 2 right
    door_w: int = 8  # door width (matche sidebar width)
    door_h: int = 16  # door height (matches top/bottom bar height)
    door_respawn_cooldown: int = 120  # frames until a closed door may spawn again

    # enemies
    max_enemies: int = 3  # simultaneously there can only be three enemies in the arena
    enemy_spawn_offset: int = 3  # distance in pixels from when spawning
    enemy_respawn_timer: int = 100  # time in frames until the next enemy spawns
    enemy_recalc_target: Tuple[int, int] = (
        24,
        50,
    )  # After how many frames should the target be recalculated? min,max
    enemy_speed: int = 8  # inversed: move envery third frame
    enemy_target_radius: int = (
        80  # radius (px) around player to sample target (where to walk)
    )
    enemy_min_dist: int = 12  # min distance between the enemies
    enemy_firing_cooldown: int = 100  # frames until the enemy can fire again

    # gameplay
    wave_timeout: int = 100  # frames between enemy waves

    # Colors (RGBA, 0–255 each)
    rgba_purple: Tuple[int, int, int, int] = (93, 61, 191, 255)  # Border color
    rgba_green: Tuple[int, int, int, int] = (82, 121, 42, 255)  # Scorebar color
    rgba_gray: Tuple[int, int, int, int] = (
        132,
        132,
        131,
        255,
    )  # Gamefield background color
    rgba_door_spawn: Tuple[int, int, int, int] = (210, 81, 80, 255)  # visible door
    rgba_door_locked: Tuple[int, int, int, int] = (
        151,
        163,
        67,
        255,
    )  # locked open (teleportable)


class Rect(NamedTuple):
    x: int
    y: int
    w: int
    h: int


class Player(NamedTuple):
    x: Array  # (N,) int32: X-Position
    y: Array  # (N,) int32: Y-Position
    vx: Array  # (N,) int32: velocity in x-direction
    vy: Array  # (N,) int32: velocity in y-direction
    # width and height could also be be stored in TronConstants
    # it however is easier to just keep them here for collision-checking/rendering
    w: Array  # (N,) int32: Width
    h: Array  # (N,) int32: Height
    lives: Array  # (N, ) number of lives


class Enemies(NamedTuple):
    x: Array  # (N,) int32: X-Position
    y: Array  # (N,) int32: Y-Position
    vx: Array  # (N,) int32: velocity in x-direction
    vy: Array  # (N,) int32: velocity in y-direction
    w: Array  # (N,) int32: Width
    h: Array  # (N,) int32: Height
    alive: Array  # (N,) int32: Boolean mask
    goal_dx: Array  # (N,) int32: target to walk to
    goal_dy: Array  # (N,) int32: target to walk to
    goal_ttl: Array  # (N,) int32: time until the target gets recalculated


class Discs(NamedTuple):
    x: Array  # (N,) int32: X-Position
    y: Array  # (N,) int32: Y-Position
    vx: Array  # (N,) int32: velocity in x-direction
    vy: Array  # (N,) int32: velocity in y-direction
    # width and height could also be be stored in TronConstants
    # it however is easier to just keep them here for collision-checking/rendering
    w: Array  # (N,) int32: Width
    h: Array  # (N,) int32: Height
    owner: Array  # (D,) int32, 0 = player, 1 = enemy
    phase: Array  # (D,) int32, 0=idle/unused, 1=outbound, 2=returning (player only)


class Doors(NamedTuple):
    x: Array  # int 32
    y: Array  # int32
    w: Array  # int32
    h: Array  # int32

    is_spawned: Array  # bool: visible entrance
    is_locked_open: Array  # bool: color change / teleportable
    spawn_lockdown: Array  # int32 frames until this slot can spawn again

    side: Array  # int32: 0=top, 1=bottom, 2=left, 3=right
    pair: Array  # int32: index of the opposite door for teleporting


class TronState(NamedTuple):
    score: Array
    player: Player  # N = 1
    # enemies: Actors # N = MAX_ENEMIES
    # short cooldown after each wave/color
    wave_end_cooldown_remaining: Array
    aim_dx: Array  # remember last movement direction in X-dir
    aim_dy: Array  # remember last movement direction in Y-dir
    discs: Discs
    fire_down_prev: Array  # shape (), bool
    doors: Doors
    enemies: Enemies
    game_started: Array  # bool: becomes True after first movement
    inwave_spawn_cd: Array  # int32: frames until next single respawn in current wave
    enemies_alive_last: (
        Array  # int32: alive count from previous step (for wave-clear edge)
    )
    rng_key: random.PRNGKey
    frame_idx: Array  # int32: frame counter


class EntityPosition(NamedTuple):
    x: Array
    y: Array
    width: Array
    height: Array


class TronObservation(NamedTuple):
    pass


class TronInfo(NamedTuple):
    pass


# Organize all helper functions concerning the disc-movement in this class
class _DiscOps:

    @staticmethod
    @jit
    def check_wall_hit(
        discs: Discs, min_x: Array, min_y: Array, max_x: Array, max_y: Array
    ) -> Array:
        """
        Boolean mask that checks, if the next (x,y) step would leave the visible area
        """
        nx, ny = discs.x + discs.vx, discs.y + discs.vy
        return (
            (nx < min_x)
            | (ny < min_y)
            | ((nx + discs.w) > max_x)
            | ((ny + discs.h) > max_y)
        )

    @staticmethod
    @jit
    def compute_next_phase(
        discs: Discs, fire_pressed: Array, next_step_wall: Array
    ) -> Array:
        """
        Decide the next phase from current phase/owner and inputs.
        0=inactive, 1=outbound, 2=returning (player only)
        """
        current_phase = discs.phase
        is_outbound = current_phase == jnp.int32(1)
        is_owner_player = discs.owner == jnp.int32(0)
        is_owner_enemy = discs.owner == jnp.int32(1)

        # return disc back to player, if the player pressed fire again
        # or would hit a wall next
        # IMPORANT: Only the playerr can recall the disc. Enemies can only shoot them
        return_disc = is_outbound & is_owner_player & (fire_pressed | next_step_wall)
        next_phase = jnp.where(
            return_disc,
            jnp.int32(2),  # returning
            current_phase,  # keep the same phase if the disc shouldn't return
        )
        # Check if the discs of the enemies will hit a wall next step
        enemy_despawn_wall = is_outbound & is_owner_enemy & next_step_wall
        next_phase = jnp.where(
            enemy_despawn_wall,
            jnp.int32(0),  # Set to inactive
            next_phase,  # keep the same phase if the disc shouldn't return
        )

        # TODO: Handle returning disc. SHould then also be 0. Will i do later

        return next_phase

    @staticmethod
    @jit
    def compute_velocity(
        discs: Discs,
        next_phase: Array,
        player_center_x: Array,
        player_center_y: Array,
        inbound_speed: Array,
    ) -> Tuple[Array, Array]:
        """
        Recomputes a homing velocity every step for player-returning discs (phase==2)
        - Uses the normalized vector from the discs center to the players current center.
        - If the disc is within one step of the player, step exactly onto the player
          to avoid overshoot/orbit.
        - Inactive discs (phase==0) have zero velocity.
        """
        is_returning_player = (next_phase == jnp.int32(2)) & (
            discs.owner == jnp.int32(0)
        )
        is_inactive = next_phase == jnp.int32(0)

        # Use centers to compute a direction towards the players body
        disc_cx, disc_cy = rect_center(discs.x, discs.y, discs.w, discs.h)

        # Vector from disc -> player center
        # convert to float for normalization
        dx_f = (player_center_x - disc_cx).astype(jnp.float32)
        dy_f = (player_center_y - disc_cy).astype(jnp.float32)
        dist = jnp.sqrt(dx_f * dx_f + dy_f * dy_f)  # euclidean distance

        # Unit direction (float), scaled by inbound speed
        denom = jnp.maximum(dist, jnp.float32(1.0))  # avoid div-by-zero
        ux = dx_f / denom
        uy = dy_f / denom

        speed_f = jnp.asarray(inbound_speed, dtype=jnp.float32)

        # round to the nearest integer pixel velocity. THis preserves average grid while
        # keeping movement constraint to the integer grid
        vx_homing = jnp.round(ux * speed_f).astype(jnp.int32)
        vy_homing = jnp.round(uy * speed_f).astype(jnp.int32)

        # If the disc is within one step (<= speed) to the player, move exactly to the remaining
        # integer delte so the disc lands on the players center this frame
        dx_i = (player_center_x - disc_cx).astype(jnp.int32)
        dy_i = (player_center_y - disc_cy).astype(jnp.int32)
        close = dist <= speed_f
        vx_close = dx_i
        vy_close = dy_i

        vx_new = jnp.where(close, vx_close, vx_homing)
        vy_new = jnp.where(close, vy_close, vy_homing)

        # ensure progress when rounding yields (0,0)
        # this can happen when |ux|and |uy| are both < 0.5 with speed==1, producing
        # rounded zeros. If the distance is still nonzero, nudge one pixel in the
        # correct signed direction to guarantee forward progress.
        zero_pair = (
            (vx_new == jnp.int32(0))
            & (vy_new == jnp.int32(0))
            & (dist > jnp.float32(0))
        )
        vx_new = jnp.where(zero_pair, jnp.sign(dx_f).astype(jnp.int32), vx_new)
        vy_new = jnp.where(zero_pair, jnp.sign(dy_f).astype(jnp.int32), vy_new)

        # Apply homing velocity only for returning player discs; keep stored velocity otherwise
        velocity_x = jnp.where(is_returning_player, vx_new, discs.vx)
        velocity_y = jnp.where(is_returning_player, vy_new, discs.vy)

        # Inactive discs don't move
        velocity_x = jnp.where(is_inactive, jnp.int32(0), velocity_x)
        velocity_y = jnp.where(is_inactive, jnp.int32(0), velocity_y)

        return velocity_x, velocity_y

    @staticmethod
    @jit
    def add_and_clamp(
        discs: Discs,
        next_phase: Array,
        velocity_x: Array,
        velocity_y: Array,
        min_x: Array,
        min_y: Array,
        max_x: Array,
        max_y: Array,
    ) -> Tuple[Array, Array]:
        """
        Apply velocity to update position for active discs and clamp to screen size
        """
        is_active = next_phase > jnp.int32(0)

        # only update position if the disc is active
        x_next = jnp.where(is_active, discs.x + velocity_x, discs.x)
        y_next = jnp.where(is_active, discs.y + velocity_y, discs.y)

        # clamp position to stay within the screen boundaries
        x_next = jnp.clip(x_next, min_x, max_x - discs.w)
        y_next = jnp.clip(y_next, min_y, max_y - discs.h)
        return x_next, y_next

    @staticmethod
    @jit
    def player_pickup_returning_discs(
        discs: Discs,
        next_phase: Array,
        next_x: Array,
        next_y: Array,
        player_x0: Array,
        player_y0: Array,
        player_w: Array,
        player_h: Array,
        vx: Array,
        vy: Array,
    ) -> Tuple[Array, Array, Array]:
        """
        If a returning player disc overlaps the player after integration:
        - set phase to 0 (inactive)
        - zero its velocity
        """
        is_returning_player = (discs.owner == jnp.int32(0)) & (
            next_phase == jnp.int32(2)
        )
        overlaps_player = (
            (next_x < player_x0 + player_w)
            & ((next_x + discs.w) > player_x0)
            & (next_y < player_y0 + player_h)
            & ((next_y + discs.h) > player_y0)
        )
        picked_up = is_returning_player & overlaps_player

        final_phase = jnp.where(picked_up, jnp.int32(0), next_phase)
        final_vx = jnp.where(picked_up, jnp.int32(0), vx)
        final_vy = jnp.where(picked_up, jnp.int32(0), vy)
        return final_phase, final_vx, final_vy


class _ArenaOps:
    @staticmethod
    def compute_arena(
        c: TronConstants,
    ) -> Tuple[Rect, Rect, Tuple[Rect, Rect, Rect, Rect], Rect]:
        """
        Returns: (gamefield_rect, scorebar_rect, (top,bottom,left,right) border rects, inner_play_rect)
        All rects are in screen coordinates.
        """
        game = Rect(c.game_x, c.game_y, c.game_w, c.game_h)
        score = Rect(game.x, game.y, game.w, c.score_h)

        # y positions for purple border bands
        top_y = game.y + c.score_h + c.score_gap
        bottom_y = game.y + game.h - c.bord_bot

        # horizontal bars (top/bottom)
        # note: width ends before the right 8px margin to match the original layout
        horizontal_w = game.w - c.bord_right
        top = Rect(game.x, top_y, horizontal_w, c.bord_top)
        bottom = Rect(game.x, bottom_y, horizontal_w, c.bord_bot)

        # vertical bars (left/right)
        inner_h = bottom.y - (top.y + c.bord_top)  # space between top/bottom bars
        left = Rect(game.x, top.y + c.bord_top, c.bord_left, inner_h)
        right = Rect(game.x + game.w - 2 * c.bord_right, left.y, c.bord_right, inner_h)

        # inner play rectangle = area inside the purple bars
        inner = Rect(left.x + left.w, left.y, right.x - (left.x + left.w), inner_h)
        return game, score, (top, bottom, left, right), inner

    @staticmethod
    def _place_doors_evenly(start: int, length: int, n: int, size: int) -> list[int]:
        """
        Place n doors of `size` evenly within [start, start+length)
        Returns the list of left/top coordinates for door
        """
        # split free space into (n+1) gaps. one on the left, (n-1) between the slots and
        # one on the right
        gap = (length - n * size) // (n + 1)
        return [start + gap + i * (size + gap) for i in range(n)]

    @staticmethod
    def make_initial_doors(c: TronConstants) -> Doors:
        # Use arena rects (ints)
        game, score, (top, bottom, left, right), inner = _ArenaOps.compute_arena(c)

        door_w, door_h = c.door_w, c.door_h

        # Top/bottom: 4 each, doors sit ON the purple border
        top_xs = _ArenaOps._place_doors_evenly(top.x, top.w, 4, door_w)
        bottom_xs = _ArenaOps._place_doors_evenly(bottom.x, bottom.w, 4, door_w)
        top_ys, bottom_ys = [top.y] * 4, [bottom.y] * 4

        # Left/right: 2 each, vary along vertical span; x fixed at bars x
        left_ys = _ArenaOps._place_doors_evenly(left.y, left.h, 2, door_h)
        right_ys = _ArenaOps._place_doors_evenly(right.y, right.h, 2, door_h)
        left_xs, right_xs = [left.x] * 2, [right.x] * 2

        # Concatenate in order: top(4), bottom(4), left(2), right(2)
        xs = top_xs + bottom_xs + left_xs + right_xs
        ys = top_ys + bottom_ys + left_ys + right_ys
        ws = [door_w] * c.max_doors
        hs = [door_h] * c.max_doors

        # Sides
        # ids 0 - 3 for the sides
        sides = [SIDE_TOP] * 4 + [SIDE_BOTTOM] * 4 + [SIDE_LEFT] * 2 + [SIDE_RIGHT] * 2

        # Pair mapping (teleport targets): Top i <-> Bottom i, Left i <-> Right i
        # Indices: 0..3 top, 4..7 bottom, 8..9 left, 10..11 right
        pairs = [4 + i for i in range(4)] + [i for i in range(4)] + [10, 11] + [8, 9]

        # Initial state: show doors; not locked; no cooldown
        is_spawned = [False] * c.max_doors
        is_locked_open = [False] * c.max_doors
        lockdown = [0] * c.max_doors

        # Convert to JAX arrays
        to_i32 = lambda L: jnp.asarray(L, dtype=jnp.int32)
        to_b = lambda L: jnp.asarray(L, dtype=jnp.bool_)

        return Doors(
            x=to_i32(xs),
            y=to_i32(ys),
            w=to_i32(ws),
            h=to_i32(hs),
            is_spawned=to_b(is_spawned),
            is_locked_open=to_b(is_locked_open),
            spawn_lockdown=to_i32(lockdown),
            side=to_i32(sides),
            pair=to_i32(pairs),
        )


Actor = TypeVar("Actor", Player, Discs)


class UserAction(NamedTuple):
    """Boolean flags for the players action"""

    up: Array
    down: Array
    left: Array
    right: Array
    fire: Array
    moved: Array  # flag for any movement


@jit
def parse_action(action: Array) -> UserAction:
    """Translate the raw action integer into a UserAction"""
    is_up = (
        (action == Action.UP)
        | (action == Action.UPRIGHT)
        | (action == Action.UPLEFT)
        | (action == Action.UPFIRE)
        | (action == Action.UPRIGHTFIRE)
        | (action == Action.UPLEFTFIRE)
    )

    is_down = (
        (action == Action.DOWN)
        | (action == Action.DOWNRIGHT)
        | (action == Action.DOWNLEFT)
        | (action == Action.DOWNFIRE)
        | (action == Action.DOWNRIGHTFIRE)
        | (action == Action.DOWNLEFTFIRE)
    )

    is_right = (
        (action == Action.RIGHT)
        | (action == Action.UPRIGHT)
        | (action == Action.DOWNRIGHT)
        | (action == Action.RIGHTFIRE)
        | (action == Action.UPRIGHTFIRE)
        | (action == Action.DOWNRIGHTFIRE)
    )

    is_left = (
        (action == Action.LEFT)
        | (action == Action.UPLEFT)
        | (action == Action.DOWNLEFT)
        | (action == Action.LEFTFIRE)
        | (action == Action.UPLEFTFIRE)
        | (action == Action.DOWNLEFTFIRE)
    )

    is_fire = (
        (action == Action.FIRE)
        | (action == Action.UPFIRE)
        | (action == Action.RIGHTFIRE)
        | (action == Action.LEFTFIRE)
        | (action == Action.DOWNFIRE)
        | (action == Action.UPRIGHTFIRE)
        | (action == Action.UPLEFTFIRE)
        | (action == Action.DOWNRIGHTFIRE)
        | (action == Action.DOWNLEFTFIRE)
    )

    # The moved flag is just an OR of the directions
    has_moved = is_up | is_down | is_left | is_right

    return UserAction(
        up=is_up,
        down=is_down,
        left=is_left,
        right=is_right,
        fire=is_fire,
        moved=has_moved,
    )


@jit
def _color_rgba(rgba: Tuple[int, int, int, int]) -> Array:
    return jnp.asarray(rgba, dtype=jnp.uint8)


def _solid_sprite(h: int, w: int, rgba: Tuple[int, int, int, int]) -> Array:
    return jnp.broadcast_to(_color_rgba(rgba), (h, w, 4))


class TronRenderer(JAXGameRenderer):
    def __init__(self, consts: TronConstants = None) -> None:
        super().__init__()
        self.consts = consts or TronConstants()
        (self.game_rect, self.score_rect, self.border_rects, self.inner_rect) = (
            _ArenaOps.compute_arena(self.consts)
        )

    @partial(jit, static_argnums=(0,))
    def render(self, state) -> Array:
        c = self.consts

        raster = jr.create_initial_frame(width=c.screen_width, height=c.screen_height)

        game, score = self.game_rect, self.score_rect
        top, bottom, left, right = self.border_rects

        # gray gamefield
        raster = jr.render_at(
            raster, game.x, game.y, _solid_sprite(game.h, game.w, c.rgba_gray)
        )

        # green scorebar
        raster = jr.render_at(
            raster, score.x, score.y, _solid_sprite(score.h, score.w, c.rgba_green)
        )

        # purple border (2 sprites: one horizontal band, one vertical band)
        raster = jr.render_at(
            raster, top.x, top.y, _solid_sprite(top.h, top.w, c.rgba_purple)
        )
        raster = jr.render_at(
            raster, bottom.x, bottom.y, _solid_sprite(bottom.h, bottom.w, c.rgba_purple)
        )
        raster = jr.render_at(
            raster, left.x, left.y, _solid_sprite(left.h, left.w, c.rgba_purple)
        )
        raster = jr.render_at(
            raster, right.x, right.y, _solid_sprite(right.h, right.w, c.rgba_purple)
        )

        # door sprites (same shape for all sides: 16x8)
        door_spawn_sprite = _solid_sprite(c.door_h, c.door_w, c.rgba_door_spawn)
        door_locked_sprite = _solid_sprite(c.door_h, c.door_w, c.rgba_door_locked)

        def render_door(i, ras):
            doors = state.doors
            active = doors.is_spawned[i]
            locked = doors.is_locked_open[i]

            def draw(r):
                spr = jax.lax.select(locked, door_locked_sprite, door_spawn_sprite)
                return jr.render_at(r, doors.x[i], doors.y[i], spr)

            return jax.lax.cond(active, draw, lambda r: r, ras)

        raster = jax.lax.fori_loop(0, c.max_doors, render_door, raster)

        # render player
        player_color = jnp.array([0, 0, 255, 255], dtype=jnp.uint8)
        player_box_sprite = jnp.broadcast_to(
            player_color,
            (self.consts.player_width, self.consts.player_height, 4),
        )
        player_x, player_y = state.player.x[0], state.player.y[0]
        raster = jr.render_at(
            raster,
            player_x,
            player_y,
            player_box_sprite,
        )

        # render discs
        disc_color = jnp.array([0, 255, 0, 255], dtype=jnp.uint8)
        disc_size = self.consts.disc_size
        disc_box_sprite = jnp.broadcast_to(disc_color, (disc_size, disc_size, 4))

        def render_disc(i, ras):
            active = state.discs.phase[i] > jnp.int32(0)
            x_i = state.discs.x[i]
            y_i = state.discs.y[i]
            return jax.lax.cond(
                active,
                lambda r: jr.render_at(r, x_i, y_i, disc_box_sprite),
                lambda r: r,
                ras,
            )

        raster = jax.lax.fori_loop(0, self.consts.max_discs, render_disc, raster)

        enemy_color = jnp.array([255, 0, 0, 255], dtype=jnp.uint8)
        enemy_sprite = jnp.broadcast_to(
            enemy_color, (self.consts.player_height, self.consts.player_width, 4)
        )

        def render_enemy(i, ras):
            alive = state.enemies.alive[i]
            ex = state.enemies.x[i]
            ey = state.enemies.y[i]
            return jax.lax.cond(
                alive,
                lambda r: jr.render_at(r, ex, ey, enemy_sprite),
                lambda r: r,
                ras,
            )

        raster = jax.lax.fori_loop(0, self.consts.max_enemies, render_enemy, raster)
        return raster


####
# Helper functions
####
@jit
def set_velocity(actors: Actor, vx: Array, vy: Array) -> Actor:
    """Returns a new Actors instanc ce with updated velocity"""
    return actors._replace(vx=vx, vy=vy)


@jit
def move(actors: Actor) -> Actor:
    """Returns a new Actors instance with positions updated by velocity"""
    return actors._replace(x=actors.x + actors.vx, y=actors.y + actors.vy)


@jit
def clamp_actor_to_bounds(
    actors: Actor, min_x: int, min_y: int, max_x: int, max_y: int
) -> Actor:
    """Clamps positions according to the given max_x and max_y"""
    new_x = jnp.clip(actors.x, min_x, max_x - actors.w)
    new_y = jnp.clip(actors.y, min_y, max_y - actors.h)
    return actors._replace(x=new_x, y=new_y)


@jit
def rect_center(x, y, w, h) -> Tuple[Array, Array]:
    """Calculates the center of a rectangle"""
    return x + jnp.floor_divide(w, 2), y + jnp.floor_divide(h, 2)


@jit
def _find_first_true(mask: Array) -> Tuple[Array, Array]:
    """Return (has_any, first_index) for a 1D bool mask"""
    idx = jnp.argmax(mask.astype(jnp.int32))
    has = jnp.any(mask)
    return has, idx


@jit
def _get_quadrant_index(
    object_center_x: Array,
    object_center_y: Array,
    play_area_center_x: Array,
    play_area_center_y: Array,
) -> Array:
    """
    Map a point to a quadrant of the inner play area
    Encoding:
        0 = top-left
        1 = top-right
        2 = bottom-left
        3 = bottom-right
    """
    is_right = (object_center_x >= play_area_center_x).astype(jnp.int32)
    is_bottom = (object_center_y >= play_area_center_y).astype(jnp.int32)
    return (is_bottom << 1) | is_right


@jit
def _door_quadrants(
    doors: Doors,
    play_area_center_x: Array,
    play_area_center_y: Array,
) -> Array:
    """
    Quadrant index (0..3) for each door, computed from the door's rectangle center
    """
    door_center_x, door_center_y = rect_center(doors.x, doors.y, doors.w, doors.h)
    return _get_quadrant_index(
        door_center_x, door_center_y, play_area_center_x, play_area_center_y
    )


@jit
def _enemy_quadrants(
    enemies: Enemies,
    play_area_center_x: Array,
    play_area_center_y: Array,
) -> Array:
    """
    Quadrant index (0..3) for each enemy, computed from the enemy's rectangle center
    """
    enemy_center_x, enemy_center_y = rect_center(
        enemies.x, enemies.y, enemies.w, enemies.h
    )
    return _get_quadrant_index(
        enemy_center_x, enemy_center_y, play_area_center_x, play_area_center_y
    )


@jit
def get_user_disc_center(state: Actor) -> Tuple[Array, Array]:
    """
    Per-disc top-left (px, py) that would place each disc centered inside the player
    Returns:
        px, py: Arrays of shape (D,), D == number of disc slots.
        For slot i, setting discs.x[i]=px[i] and discs.y[i]=py[i]
        places that disc visually centered within the player's rectangle
    """
    p: Player = state.player
    d: Discs = state.discs
    px = p.x[0] + jnp.floor_divide(p.w[0] - d.w, 2)
    py = p.y[0] + jnp.floor_divide(p.h[0] - d.h, 2)
    return px.astype(jnp.int32), py.astype(jnp.int32)


@jit
def _select_door_for_spawn(
    doors: Doors,
    enemies: Enemies,
    player_center_x: Array,
    player_center_y: Array,
    inner_min_x: Array,
    inner_min_y: Array,
    inner_max_x: Array,
    inner_max_y: Array,
    rng_key: random.PRNGKey,
    prefer_new_prob: float = 0.4,
) -> Tuple[Array, Array, Doors, random.PRNGKey]:
    """
    Choose a door index to spawn an enemy from:
      - quadrant must be free of the player and any alive enemy
      - prefer reusing an existing spawned door
      - otherwise, try to spawn (make visible) a new door in a free quadrant
      - fallback: any spawned & unlocked door if all quadrants are busy
      - when BOTH reuse and new are available, pick NEW with probability `prefer_new_prob`

    Returns (has_choice, door_index, updated_doors, next_rng_key).
    """
    # Get center of the play-area inside the purple borders
    play_area_center_x = (inner_min_x + inner_max_x) // 2
    play_area_center_y = (inner_min_y + inner_max_y) // 2

    # The entire play_area can be split into 4 quadrants. Each of them has 3 doors
    door_quadrant = _door_quadrants(doors, play_area_center_x, play_area_center_y)

    # Get current quadrant of player
    player_quadrant = _get_quadrant_index(
        player_center_x, player_center_y, play_area_center_x, play_area_center_y
    )

    # Get quadrant of all the enemies
    enemy_quadrant = _enemy_quadrants(enemies, play_area_center_x, play_area_center_y)

    # For each door, is its quadrant occupied by any alive enemy?
    enemy_same_quad = (door_quadrant[:, None] == enemy_quadrant[None, :]) & (
        enemies.alive[None, :]
    )
    occupied_by_enemies = jnp.any(enemy_same_quad, axis=1)
    occupied_by_player = door_quadrant == player_quadrant
    # Check if a quadrant is free
    quadrant_is_free = ~occupied_by_enemies & ~occupied_by_player

    # All doors have a lockdown, between spawning enemies
    # Select only those, with a lockdown of 0
    door_unlocked = doors.spawn_lockdown == jnp.int32(0)

    # Select doors, that are already spawned, have a cooldown of 0 and a free quadrant
    prefer_reuse = doors.is_spawned & door_unlocked & quadrant_is_free
    # Select doors, that are not yet spawned, have a cooldown of 0 and a free quadrant
    try_new = (~doors.is_spawned) & door_unlocked & quadrant_is_free

    has_reuse, idx_reuse = _find_first_true(prefer_reuse)
    has_new, idx_new = _find_first_true(try_new)

    # RNG for tie-break when both reuse and new are possible
    rng_key, sample_key = random.split(rng_key)
    p = jnp.float32(prefer_new_prob)
    pick_new_sample = random.bernoulli(sample_key, p=p)  # True means "choose new"

    choose_new = has_new & (~has_reuse | pick_new_sample)
    choose_reuse = has_reuse & (~has_new | (~pick_new_sample))

    def _pick_reuse(_):
        return True, idx_reuse, doors, rng_key

    def _pick_new(_):
        updated = doors._replace(is_spawned=doors.is_spawned.at[idx_new].set(True))
        return True, idx_new, updated, rng_key

    def _fallback(_):
        mask = doors.is_spawned & door_unlocked
        has_fb, idx_fb = _find_first_true(mask)
        return has_fb, idx_fb, doors, rng_key

    return jax.lax.cond(
        choose_new,
        _pick_new,
        lambda _: jax.lax.cond(choose_reuse, _pick_reuse, _fallback, operand=None),
        operand=None,
    )


@jit
def _tick_door_lockdown(doors: Doors) -> Doors:
    """Decrement per-door spawn cooldown timers (floored at 0)."""
    return doors._replace(spawn_lockdown=jnp.maximum(doors.spawn_lockdown - 1, 0))


@jit
def _mark_door_used_for_spawn(
    doors: Doors, door_index: Array, cooldown_frames: int
) -> Doors:
    """When a door is used to spawn, start its spawn cooldown."""
    return doors._replace(
        spawn_lockdown=doors.spawn_lockdown.at[door_index].set(
            jnp.int32(cooldown_frames)
        )
    )


class JaxTron(JaxEnvironment[TronState, TronObservation, TronInfo, TronConstants]):
    def __init__(
        self, consts: TronConstants = None, reward_funcs: list[callable] = None
    ) -> None:
        consts = consts or TronConstants()
        super().__init__(consts)
        self.renderer = TronRenderer
        self.action_set = [
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
        ]

        # Precompute static rects
        (self.game_rect, self.score_rect, self.border_rects, self.inner_rect) = (
            _ArenaOps.compute_arena(self.consts)
        )

        # Precompute JAX scalars
        self.inner_min_x = jnp.int32(self.inner_rect.x)
        self.inner_min_y = jnp.int32(self.inner_rect.y)
        self.inner_max_x = jnp.int32(self.inner_rect.x + self.inner_rect.w)
        self.inner_max_y = jnp.int32(self.inner_rect.y + self.inner_rect.h)

        # Prebuild initial doors (geometry + default state)
        self.initial_doors = _ArenaOps.make_initial_doors(self.consts)

    def reset(self, key: random.PRNGKey = None) -> Tuple[TronObservation, TronState]:
        def _get_centered_player(consts: TronConstants) -> Player:
            screen_w, screen_h = consts.screen_width, consts.screen_height
            player_w, player_h = consts.player_width, consts.player_height
            x0 = (screen_w - player_w) // 2
            y0 = (screen_h - player_h) // 2
            return Player(
                x=jnp.array([x0], dtype=jnp.int32),
                y=jnp.array([y0], dtype=jnp.int32),
                vx=jnp.zeros(1, dtype=jnp.int32),
                vy=jnp.zeros(1, dtype=jnp.int32),
                w=jnp.full((1,), player_w, dtype=jnp.int32),
                h=jnp.full((1,), player_h, dtype=jnp.int32),
                lives=jnp.array([consts.player_lives], dtype=jnp.int32),
            )

        def _get_empty_discs(consts: TronConstants) -> Discs:
            D = consts.max_discs
            w = jnp.full((D,), consts.disc_size, dtype=jnp.int32)
            h = jnp.full((D,), consts.disc_size, dtype=jnp.int32)

            zeros = jnp.zeros((D,), dtype=jnp.int32)

            return Discs(
                x=zeros,
                y=zeros,
                vx=zeros,
                vy=zeros,
                w=w,
                h=h,
                owner=zeros,
                phase=zeros,
            )

        def _get_empty_enemies(consts: TronConstants) -> Enemies:
            N = consts.max_enemies
            # use player size as enemy size TODO: change lkater
            ew = jnp.full((N,), consts.player_width, dtype=jnp.int32)
            eh = jnp.full((N,), consts.player_height, dtype=jnp.int32)
            z = jnp.zeros((N,), dtype=jnp.int32)
            alive = jnp.zeros((N,), dtype=jnp.bool_)
            return Enemies(
                x=z,
                y=z,
                vx=z,
                vy=z,
                w=ew,
                h=eh,
                alive=alive,
                goal_dx=z,
                goal_dy=z,
                goal_ttl=z,
            )

        new_state: TronState = TronState(
            score=jnp.zeros((), dtype=jnp.int32),
            player=_get_centered_player(self.consts),
            wave_end_cooldown_remaining=jnp.zeros((), dtype=jnp.int32),
            aim_dx=jnp.zeros((), dtype=jnp.int32),
            aim_dy=jnp.zeros((), dtype=jnp.int32),
            discs=_get_empty_discs(self.consts),
            fire_down_prev=jnp.array(False),
            doors=self.initial_doors,
            enemies=_get_empty_enemies(self.consts),
            game_started=jnp.array(False),
            inwave_spawn_cd=jnp.zeros((), dtype=jnp.int32),
            enemies_alive_last=jnp.zeros((), dtype=jnp.int32),
            rng_key=key,
            frame_idx=jnp.zeros((), dtype=jnp.int32),
        )
        obs = self._get_observation(new_state)
        return obs, new_state

    @partial(jax.jit, static_argnums=(0,))
    def _cooldown_finished(self, state: TronState) -> Array:
        # TODO: Change later to ==
        return state.wave_end_cooldown_remaining != 0

    @partial(jit, static_argnums=(0,))
    def _player_step(self, state: TronState, action: UserAction) -> TronState:
        player = state.player
        speed = self.consts.player_speed

        # Calculate horizontal velocity
        # the boolean subtraction (right - left) results in +1, -1 or 0
        dx = speed * (action.right.astype(jnp.int32) - action.left.astype(jnp.int32))

        # Calculate vertical velocity
        # the boolean subtraction (down - up) results in +1, -1 or 0
        dy = speed * (action.down.astype(jnp.int32) - action.up.astype(jnp.int32))

        # Set the new velocity on the player actor
        player = set_velocity(player, dx, dy)
        # apply the velocity to the players position
        player = move(player)
        # Ensure the new position is within boundaries
        player = clamp_actor_to_bounds(
            player,
            self.inner_min_x,
            self.inner_min_y,
            self.inner_max_x,
            self.inner_max_y,
        )

        # only update the aiming direction, if movement key was pressed
        aim_dx = jnp.where(action.moved, dx, state.aim_dx)
        aim_dy = jnp.where(action.moved, dy, state.aim_dy)

        return state._replace(player=player, aim_dx=aim_dx, aim_dy=aim_dy)

    @partial(jit, static_argnums=(0,))
    def _spawn_disc(self, state: TronState, fire: Array) -> TronState:
        discs: Discs = state.discs

        # Check if the player already has any active discs
        # he has to be the owner and active means phase != 0
        has_player_disc: Array = jnp.any(
            (discs.owner == jnp.int32(0)) & (discs.phase > jnp.int32(0))
        )

        # any free slots to place a new disc?
        free_mask: Array = discs.phase == jnp.int32(0)
        any_free: Array = jnp.any(free_mask)

        # can spawn, if in the previous frame fire wasn't pressed, the player hasn't any active discs
        # and the discs array still has empty slots
        can_spawn: Array = fire & jnp.logical_not(has_player_disc) & any_free

        def do_spawn(s: TronState) -> TronState:
            # get the index of a free slot
            free_indices: Array = jnp.nonzero(s.discs.phase == 0, size=1)[0][0]
            # TODO: Should never happen, but what if there is no free slot? Then this will fail

            # Center the discs in the player-box
            px_all, py_all = get_user_disc_center(s)
            px, py = px_all[free_indices], py_all[free_indices]

            disc_speed: jnp.int32 = jnp.int32(self.consts.disc_speed)
            # Get the last direction (aim_d) in which the player walked
            # jnp.sign returns the values -1, 0, 1. Makes it independent of the player_speed
            disc_vel_x: jnp.int32 = jnp.sign(s.aim_dx).astype(jnp.int32) * disc_speed
            disc_vel_y: jnp.int32 = jnp.sign(s.aim_dy).astype(jnp.int32) * disc_speed

            # writes in the free slot
            new_discs: Discs = s.discs._replace(
                x=s.discs.x.at[free_indices].set(px),
                y=s.discs.y.at[free_indices].set(py),
                vx=s.discs.vx.at[free_indices].set(disc_vel_x),
                vy=s.discs.vy.at[free_indices].set(disc_vel_y),
                owner=s.discs.owner.at[free_indices].set(
                    jnp.int32(0)
                ),  # 0 = player # TODO: Change later when also enemies can spawn discs
                phase=s.discs.phase.at[free_indices].set(jnp.int32(1)),  # 1 = outbound
            )
            return s._replace(discs=new_discs)

        def no_spawn(s: TronState) -> TronState:
            return s

        return jax.lax.cond(can_spawn, do_spawn, no_spawn, state)

    @partial(jit, static_argnums=(0,))
    def _move_discs(self, state: TronState, fire_pressed: Array) -> TronState:
        discs = state.discs

        # Calculate the max value for both x and y
        # Subtract the discs width/height from the width/height
        # the discs should remain fully in the visible area
        max_x = jnp.int32(self.consts.screen_width) - discs.w
        max_y = jnp.int32(self.consts.screen_height) - discs.h

        will_hit_wall_next = _DiscOps.check_wall_hit(
            discs,
            self.inner_min_x,
            self.inner_min_y,
            self.inner_max_x,
            self.inner_max_y,
        )
        next_phase = _DiscOps.compute_next_phase(
            discs, fire_pressed, will_hit_wall_next
        )

        # # Compute the player center for x and y coordinates
        player_center_x, player_center_y = rect_center(
            state.player.x[0],
            state.player.y[0],
            state.player.w[0],
            state.player.h[0],
        )

        # compute the velocity depending on the phase
        velocity_x, velocity_y = _DiscOps.compute_velocity(
            discs,
            next_phase,
            player_center_x,
            player_center_y,
            self.consts.inbound_disc_speed,
        )

        # add the velocity to the position and clamp to the screen size
        x_next, y_next = _DiscOps.add_and_clamp(
            discs,
            next_phase,
            velocity_x,
            velocity_y,
            self.inner_min_x,
            self.inner_min_y,
            self.inner_max_x,
            self.inner_max_y,
        )

        # despawn returning player discs on pickup (after integration)
        final_phase, final_velocity_x, final_velocity_y = (
            _DiscOps.player_pickup_returning_discs(
                discs,
                next_phase,
                x_next,
                y_next,
                state.player.x[0],
                state.player.y[0],
                state.player.w[0],
                state.player.h[0],
                velocity_x,
                velocity_y,
            )
        )

        # Ensure any disc that is now inactive has zero velocity
        final_velocity_x = jnp.where(
            final_phase == jnp.int32(0), jnp.int32(0), final_velocity_x
        )
        final_velocity_y = jnp.where(
            final_phase == jnp.int32(0), jnp.int32(0), final_velocity_y
        )
        new_discs = discs._replace(
            x=x_next,
            y=y_next,
            vx=final_velocity_x,
            vy=final_velocity_y,
            phase=final_phase,
        )

        return state._replace(discs=new_discs)

    @partial(jit, static_argnums=(0,))
    def _spawn_pos_from_door(
        self,
        doors: Doors,
        door_index: Array,
        enemy_width_px: Array,
        enemy_height_px: Array,
    ) -> Tuple[Array, Array]:
        """
        Compute a valid (x, y) spawn position inside the inner play area based on door side.

        - Centers the enemy against the chosen door slot along the orthogonal axis.
        - Nudges the enemy `enemy_spawn_offset` pixels into the arena along the inward axis.
        """
        door_side = doors.side[door_index]

        # Center enemy against the door rectangle
        centered_x = doors.x[door_index] + jnp.floor_divide(
            doors.w[door_index] - enemy_width_px, 2
        )
        centered_y = doors.y[door_index] + jnp.floor_divide(
            doors.h[door_index] - enemy_height_px, 2
        )

        offset_inward = jnp.int32(self.consts.enemy_spawn_offset)

        def from_top(_):
            x = jnp.clip(
                centered_x, self.inner_min_x, self.inner_max_x - enemy_width_px
            )
            y = self.inner_min_y + offset_inward
            return x, y

        def from_bottom(_):
            x = jnp.clip(
                centered_x, self.inner_min_x, self.inner_max_x - enemy_width_px
            )
            y = self.inner_max_y - enemy_height_px - offset_inward
            return x, y

        def from_left(_):
            x = self.inner_min_x + offset_inward
            y = jnp.clip(
                centered_y, self.inner_min_y, self.inner_max_y - enemy_height_px
            )
            return x, y

        def from_right(_):
            x = self.inner_max_x - enemy_width_px - offset_inward
            y = jnp.clip(
                centered_y, self.inner_min_y, self.inner_max_y - enemy_height_px
            )
            return x, y

        return jax.lax.switch(
            door_side, (from_top, from_bottom, from_left, from_right), operand=None
        )

    @partial(jit, static_argnums=(0,))
    def _spawn_enemies_up_to(
        self,
        state: TronState,
        max_new: Array,  # jnp.int32 — try to spawn at most this many now
        prefer_new_prob: Array,  # jnp.float32 — P(pick NEW door) when reuse+new both valid
    ) -> TronState:
        """
        Spawn up to `max_new` enemies (bounded by free enemy slots).
        For each spawn:
          1) Choose a quadrant-safe door with `_select_door_for_spawn`
             (prefers reuse, but can probabilistically pick NEW with `prefer_new_prob`).
          2) Place one enemy slightly inside the arena based on the door side.
          3) Start that door's spawn cooldown.
          4) Thread the RNG key.

        Notes:
          - Uses a fixed-iteration `fori_loop` with a simple guard to keep JIT happy & nesting small.
          - No enemy movement is performed here — this function *only* spawns enemies.
        """
        # How many enemy slots are free right now?
        free_mask = ~state.enemies.alive
        free_count = jnp.sum(free_mask.astype(jnp.int32))

        # We'll never spawn more than there are free slots.
        to_spawn = jnp.minimum(max_new, free_count)

        # Player center (used by the door selector to keep quadrants safe)
        player_cx, player_cy = rect_center(
            state.player.x[0],
            state.player.y[0],
            state.player.w[0],
            state.player.h[0],
        )

        def place_one(carry_state):
            """
            Spawn exactly one enemy if a valid door is available.
            Returns the updated state (enemies, doors, rng_key possibly changed).
            """
            s = carry_state

            # Choose a door (handles reuse/new preference and returns next RNG key)
            has, door_idx, doors2, key2 = _select_door_for_spawn(
                s.doors,
                s.enemies,
                player_cx,
                player_cy,
                self.inner_min_x,
                self.inner_min_y,
                self.inner_max_x,
                self.inner_max_y,
                s.rng_key,
                prefer_new_prob=prefer_new_prob,  # ensure proper dtype
            )

            def do_spawn(s_in):
                # Find the first dead slot to reuse
                dead_mask = ~s_in.enemies.alive
                _, slot = _find_first_true(dead_mask)

                # Enemy size is already in the arrays (set in reset)
                ew = s_in.enemies.w[slot]
                eh = s_in.enemies.h[slot]

                # Compute a spawn position just inside the play area, aligned to the door
                ex, ey = self._spawn_pos_from_door(doors2, door_idx, ew, eh)

                # Activate enemy in that slot
                enemies2 = s_in.enemies._replace(
                    x=s_in.enemies.x.at[slot].set(ex),
                    y=s_in.enemies.y.at[slot].set(ey),
                    vx=s_in.enemies.vx.at[slot].set(jnp.int32(0)),
                    vy=s_in.enemies.vy.at[slot].set(jnp.int32(0)),
                    alive=s_in.enemies.alive.at[slot].set(True),
                    goal_dx=s_in.enemies.goal_dx.at[slot].set(jnp.int32(0)),
                    goal_dy=s_in.enemies.goal_dy.at[slot].set(jnp.int32(0)),
                    goal_ttl=s_in.enemies.goal_ttl.at[slot].set(jnp.int32(0)),
                )

                # Start cooldown on the used door
                doors3 = _mark_door_used_for_spawn(
                    doors2, door_idx, self.consts.door_respawn_cooldown
                )

                return s_in._replace(enemies=enemies2, doors=doors3, rng_key=key2)

            # If no valid door, still update RNG/doors returned by the selector
            def no_spawn(s_in):
                return s_in._replace(doors=doors2, rng_key=key2)

            return jax.lax.cond(has, do_spawn, no_spawn, s)

        def loop_body(i, carry_state: TronState):
            # Guard keeps the loop static-sized but only does work while i < to_spawn
            return jax.lax.cond(i < to_spawn, place_one, lambda s: s, carry_state)

        # Iterate a fixed number of times for JIT-friendliness; each iteration is tiny.
        # Upper bound = max_enemies is safe and static; the guard above prevents overwork.
        return jax.lax.fori_loop(0, self.consts.max_enemies, loop_body, state)

    @partial(jit, static_argnums=(0,))
    def _move_enemies(self, state: TronState) -> TronState:
        """
        Move all alive enemies with a chunky, throttled pursuit of per-enemy goals
        (sampled around the player) plus a simple separation rule and  wall slide
        """

        # Quick exit if nothing to move.
        any_alive = jnp.any(state.enemies.alive)

        def _no_op(s):
            return s

        def _do_move(s: TronState) -> TronState:
            en = s.enemies
            N = self.consts.max_enemies

            # -------------------------------
            # Small helpers (pure, JAX-traceable)
            # -------------------------------

            def step_gate(frame_idx: Array, frames_per_step: int) -> Array:
                """Return 0/1 mask: 1 only on frames that are allowed to move."""
                period = jnp.maximum(jnp.int32(frames_per_step), jnp.int32(1))
                do_step = jnp.equal(jnp.mod(frame_idx, period), jnp.int32(0))
                return do_step.astype(jnp.int32)

            def sample_noise(key: random.PRNGKey, n: int, ttl_min: int, ttl_max: int):
                """All random scalars needed this tick (vectorized per enemy)"""
                key, k_axis, k_twitch, k_strafe, k_ang, k_rad, k_ttl = random.split(
                    key, 7
                )

                r_axis = random.uniform(k_axis, (n,), dtype=jnp.float32)  # [0,1)
                r_twitch = random.uniform(k_twitch, (n,), dtype=jnp.float32)  # [0,1)
                r_strafe = random.uniform(k_strafe, (n,), dtype=jnp.float32)  # [0,1)

                # use named minval/maxval and set dtype
                angles = random.uniform(
                    k_ang,
                    (n,),
                    minval=jnp.float32(0.0),
                    maxval=jnp.float32(2.0) * jnp.float32(jnp.pi),
                    dtype=jnp.float32,
                )

                # small integer jitter for radius (round float32 then cast)
                rad_jit = jnp.round(
                    random.uniform(
                        k_rad,
                        (n,),
                        minval=jnp.float32(-2.0),
                        maxval=jnp.float32(2.0),
                        dtype=jnp.float32,
                    )
                ).astype(jnp.int32)

                # new TTL window
                ttl_new = random.randint(k_ttl, (n,), ttl_min, ttl_max, dtype=jnp.int32)

                return (r_axis, r_twitch, r_strafe, angles, rad_jit, ttl_new), key

            def player_center(s: TronState):
                """Center of the (single) player box"""
                return rect_center(
                    s.player.x[0], s.player.y[0], s.player.w[0], s.player.h[0]
                )

            def enemy_centers(enemies: Enemies):
                """Per-enemy centers"""
                return rect_center(enemies.x, enemies.y, enemies.w, enemies.h)

            def update_goals(
                enemies: Enemies,
                pcx: Array,
                pcy: Array,
                angles: Array,
                radius_jitter: Array,
                ttl_new: Array,
            ):
                """
                Keep/refresh each enemy's goal offset (goal_dx, goal_dy) and TTL,
                and return the current absolute goal positions (gx, gy).
                """
                # Current absolute goal position from stored offsets.
                gx = pcx + enemies.goal_dx
                gy = pcy + enemies.goal_dy

                # Distance to current goal (float) to decide if we refresh early.
                ecx, ecy = enemy_centers(enemies)
                dxf, dyf = (
                    (gx - ecx).astype(jnp.float32),
                    (gy - ecy).astype(jnp.float32),
                )
                dist_goal = jnp.sqrt(dxf * dxf + dyf * dyf)

                # Age TTL; refresh when TTL hits 0 or we are close to the goal.
                ttl_next = jnp.maximum(enemies.goal_ttl - 1, 0)
                close = dist_goal <= jnp.float32(2.0)
                need_new = enemies.alive & ((ttl_next == 0) | close)

                # Sample new offsets on a circle around player with tiny radius jitter.
                base_r = jnp.int32(self.consts.enemy_target_radius)
                r = (base_r + radius_jitter).astype(jnp.float32)
                new_dx = jnp.round(r * jnp.cos(angles)).astype(jnp.int32)
                new_dy = jnp.round(r * jnp.sin(angles)).astype(jnp.int32)

                # Selectively update dx/dy/ttl only where needed; zero out for dead slots.
                goal_dx = jnp.where(need_new, new_dx, enemies.goal_dx)
                goal_dy = jnp.where(need_new, new_dy, enemies.goal_dy)
                goal_ttl = jnp.where(need_new, ttl_new, ttl_next)

                goal_dx = jnp.where(enemies.alive, goal_dx, 0)
                goal_dy = jnp.where(enemies.alive, goal_dy, 0)
                goal_ttl = jnp.where(enemies.alive, goal_ttl, 0)

                # Recompute absolute goal position using possibly-updated offsets.
                gx = pcx + goal_dx
                gy = pcy + goal_dy
                return (goal_dx, goal_dy, goal_ttl, gx, gy)

            def chunky_heading_toward(
                ecx: Array,
                ecy: Array,
                gx: Array,
                gy: Array,
                r_axis: Array,
                r_twitch: Array,
            ):
                """
                Compute a 1-pixel intended heading (±1/0 per axis) toward (gx, gy),
                with a staircase/diagonal mix and tiny twitch.
                """
                # Integer deltas to goal + their signs.
                dx, dy = (gx - ecx).astype(jnp.int32), (gy - ecy).astype(jnp.int32)
                sgnx, sgny = (
                    jnp.sign(dx).astype(jnp.int32),
                    jnp.sign(dy).astype(jnp.int32),
                )

                # Primary axis = the larger magnitude delta.
                x_is_primary = jnp.abs(dx) >= jnp.abs(dy)

                # With some prob, drop the non-primary axis to form stair steps.
                axis_only = r_axis < jnp.float32(0.45)
                base_x = jnp.where(axis_only & (~x_is_primary), 0, sgnx)
                base_y = jnp.where(axis_only & (x_is_primary), 0, sgny)

                # Tiny twitch: sometimes re-enable the dropped axis.
                do_twitch = r_twitch < jnp.float32(0.25)
                base_x = jnp.where(
                    axis_only & (~x_is_primary) & do_twitch, sgnx, base_x
                )
                base_y = jnp.where(axis_only & (x_is_primary) & do_twitch, sgny, base_y)

                return base_x, base_y

            def add_strafe(
                base_x: Array, base_y: Array, enemy_y: Array, r_strafe: Array
            ):
                """
                Perpendicular nudge (orbit feel). Probability depends on vertical position.
                """
                # Perp of (x,y) = (-y, x); produces a sideways pixel.
                perp_x, perp_y = -base_y, base_x

                # 0..1 vertical position inside inner play area.
                inner_h_f = jnp.maximum(
                    (self.inner_max_y - self.inner_min_y).astype(jnp.float32),
                    jnp.float32(1.0),
                )
                rel_y = (enemy_y.astype(jnp.float32) - self.inner_min_y) / inner_h_f

                # More likely to strafe near top/bottom than center.
                strafe_prob = jnp.clip(
                    0.15 + 0.6 * jnp.abs(rel_y - 0.5) * 2.0, 0.15, 0.75
                )
                do_strafe = (r_strafe < strafe_prob).astype(jnp.int32)

                vx = base_x + do_strafe * perp_x
                vy = base_y + do_strafe * perp_y
                # Bound to one pixel per tick on each axis.
                return jnp.clip(vx, -1, 1), jnp.clip(vy, -1, 1)

            def separation_push(enemies: Enemies, min_dist_px: int):
                """
                One-pixel push away from nearby alive neighbors.
                Returns (sep_x, sep_y) in {-1,0,1}^N.
                """
                ecx, ecy = enemy_centers(enemies)
                ecx_f, ecy_f = ecx.astype(jnp.float32), ecy.astype(jnp.float32)

                # Pairwise deltas and squared distances.
                dxm = ecx_f[:, None] - ecx_f[None, :]
                dym = ecy_f[:, None] - ecy_f[None, :]
                dist2 = dxm * dxm + dym * dym

                # Consider only distinct, alive pairs that are closer than min_dist.
                alive_pair = enemies.alive[:, None] & enemies.alive[None, :]
                not_self = ~jnp.eye(self.consts.max_enemies, dtype=jnp.bool_)
                near = dist2 < (jnp.float32(min_dist_px) * jnp.float32(min_dist_px))
                mask = (alive_pair & not_self & near).astype(jnp.float32)

                # Sum (rough) unit vectors away from neighbors, then take sign → one-pixel push.
                invd = 1.0 / jnp.sqrt(jnp.maximum(dist2, 1e-6))
                sep_xf = jnp.sum((dxm * invd) * mask, axis=1)
                sep_yf = jnp.sum((dym * invd) * mask, axis=1)
                return jnp.sign(sep_xf).astype(jnp.int32), jnp.sign(sep_yf).astype(
                    jnp.int32
                )

            def wall_slide(enemies: Enemies, vx: Array, vy: Array):
                """
                Try the move, clamp to inner play area, shave the blocked component
                (slide along walls), then recompute final clamped position.
                """
                x_try = enemies.x + vx
                y_try = enemies.y + vy

                x_cl = jnp.clip(x_try, self.inner_min_x, self.inner_max_x - enemies.w)
                y_cl = jnp.clip(y_try, self.inner_min_y, self.inner_max_y - enemies.h)

                hit_x = x_cl != x_try
                hit_y = y_cl != y_try

                vx_ok = jnp.where(hit_x, 0, vx)
                vy_ok = jnp.where(hit_y, 0, vy)

                x_next = jnp.clip(
                    enemies.x + vx_ok, self.inner_min_x, self.inner_max_x - enemies.w
                )
                y_next = jnp.clip(
                    enemies.y + vy_ok, self.inner_min_y, self.inner_max_y - enemies.h
                )
                return x_next, y_next, vx_ok, vy_ok

            # Move only on allowed frames (global throttle).
            step_mask = step_gate(s.frame_idx, self.consts.enemy_speed)  # int32 0/1

            # All randomness for this tick.
            ttl_min, ttl_max = self.consts.enemy_recalc_target
            (r_axis, r_twitch, r_strafe, angles, rad_jit, ttl_new), key = sample_noise(
                s.rng_key, N, ttl_min, ttl_max
            )

            # Player + enemy centers (targets are based on player center).
            pcx, pcy = player_center(s)
            ecx, ecy = enemy_centers(en)

            # Keep or refresh goals (per-enemy offsets and TTL), compute absolute goal positions.
            gdx, gdy, gttl, gx, gy = update_goals(
                en, pcx, pcy, angles, rad_jit, ttl_new
            )

            # Chunky heading toward the current goal (1 px on an axis/diagonal).
            base_x, base_y = chunky_heading_toward(ecx, ecy, gx, gy, r_axis, r_twitch)

            # Optional perpendicular nudge (orbit/arc feel), still ±1 per axis.
            vx, vy = add_strafe(base_x, base_y, en.y, r_strafe)

            # One-pixel separation push away from nearby alive enemies.
            sep_x, sep_y = separation_push(en, self.consts.enemy_min_dist)
            vx = jnp.clip(vx + sep_x, -1, 1)
            vy = jnp.clip(vy + sep_y, -1, 1)

            # Apply throttle + alive mask (dead don't move; gated frames stand still).
            vx = jnp.where(en.alive, vx * step_mask, 0)
            vy = jnp.where(en.alive, vy * step_mask, 0)

            # Gentle wall slide and final clamped positions.
            x_next, y_next, vx_final, vy_final = wall_slide(en, vx, vy)

            # Write back: positions, velocities, and goal bookkeeping; advance RNG key.
            en2 = en._replace(
                x=x_next,
                y=y_next,
                vx=vx_final,
                vy=vy_final,
                goal_dx=gdx,
                goal_dy=gdy,
                goal_ttl=gttl,
            )
            return s._replace(enemies=en2, rng_key=key)

        return jax.lax.cond(any_alive, _do_move, _no_op, operand=state)

    @partial(jit, static_argnums=(0,))
    def step(
        self, state: TronState, action: Array
    ) -> Tuple[TronObservation, TronState, float, bool, TronInfo]:
        previous_state = state
        user_action: UserAction = parse_action(action)
        # pressed_fire should only be true, if in the previous frame it wasn't pressed

        # track whether fire was already pressed in the frame before
        # pressed_fire is checked 60 times per second (60 fps)
        # if not tracking the previous action, pressing space (fire)
        # for one second would spawn 60 discs
        # pressed_fire should only be true, if in the previous frame it wasn't pressed
        # and we have a change in state
        pressed_fire_changed: Array = user_action.fire & jnp.logical_not(
            state.fire_down_prev
        )

        # Advance global frame counter (used to throttle enemy movement)
        state = state._replace(frame_idx=state.frame_idx + jnp.int32(1))

        def _pause_step(s: TronState) -> TronState:
            # TODO: Activate later
            # s: TronState = s._replace(
            #    wave_end_cooldown_remaining=jnp.maximum(
            #        s.wave_end_cooldown_remaining - 1, 0
            #    )
            # )
            # s = self.move_discs(s)

            # Was there a player-owned active disc BEFORE changing s?
            # we cannot pass the same pressed_fire_change to both spawn and move disc
            # because it would immediately recall the just spawned disc
            had_player_disc_before = jnp.any(
                (s.discs.owner == jnp.int32(0)) & (s.discs.phase > jnp.int32(0))
            )
            # move the player
            s: TronState = self._player_step(s, user_action)
            s: TronState = self._spawn_disc(s, pressed_fire_changed)

            # Only recall if a player-owned disc existed before this step
            recall_edge = pressed_fire_changed & had_player_disc_before
            s: TronState = self._move_discs(s, recall_edge)
            return s

        def _wave_step(s: TronState) -> TronState:
            # Was there a player-owned active disc BEFORE changing s?
            # we cannot pass the same pressed_fire_change to both spawn and move disc
            # because it would immediately recall the just spawned disc
            had_player_disc_before = jnp.any(
                (s.discs.owner == jnp.int32(0)) & (s.discs.phase > jnp.int32(0))
            )
            s: TronState = self._player_step(s, user_action)
            s: TronState = self._spawn_disc(s, pressed_fire_changed)
            # Only recall if a player-owned disc existed before this step
            recall_edge = pressed_fire_changed & had_player_disc_before
            s: TronState = self._move_discs(s, recall_edge)
            return s

        state = jax.lax.cond(
            self._cooldown_finished(state), _wave_step, _pause_step, state
        )
        state = state._replace(fire_down_prev=user_action.fire)

        # tick door cooldowns so used doors eventually become available again
        state = state._replace(doors=_tick_door_lockdown(state.doors))

        # on the first input movement, spawn up to max_enemies once
        def _spawn_initial_wave(s: TronState) -> TronState:
            # how many to reach the cap this frame
            alive_now = jnp.sum(s.enemies.alive.astype(jnp.int32))
            need = jnp.maximum(jnp.int32(self.consts.max_enemies) - alive_now, 0)
            s2 = self._spawn_enemies_up_to(s, need, jnp.float32(0.4))
            return s2._replace(game_started=jnp.array(True))

        state = jax.lax.cond(
            ~state.game_started & user_action.moved,
            _spawn_initial_wave,
            lambda s: s,
            state,
        )

        state = self._move_enemies(state)

        obs: TronObservation = self._get_observation(state)
        env_reward: float = self._get_reward(state, state)
        done: bool = self._get_done(state)
        info: TronInfo = self._get_info(state)

        return obs, state, env_reward, done, info

    @partial(jit, static_argnums=(0,))
    def _get_observation(self, state: TronState) -> TronObservation:
        return TronObservation()

    @partial(jit, static_argnums=(0,))
    def _get_reward(self, previous_state: TronState, state: TronState) -> float:
        return 0.0

    @partial(jit, static_argnums=(0,))
    def _get_done(self, state: TronState) -> bool:
        return False

    @partial(jit, static_argnums=(0,))
    def _get_info(self, state: TronState) -> TronInfo:
        return TronInfo()

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))
