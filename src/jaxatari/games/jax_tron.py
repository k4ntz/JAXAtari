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


# class Enemies(Actors):
#    alive: Array  # (N,) int32: Boolean mask


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
        sides = [0] * 4 + [1] * 4 + [2] * 2 + [3] * 2

        # Pair mapping (teleport targets): Top i <-> Bottom i, Left i <-> Right i
        # Indices: 0..3 top, 4..7 bottom, 8..9 left, 10..11 right
        pairs = [4 + i for i in range(4)] + [i for i in range(4)] + [10, 11] + [8, 9]

        # Initial state: show doors; not locked; no cooldown
        is_spawned = [True] * c.max_doors
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

        new_state: TronState = TronState(
            score=jnp.zeros((), dtype=jnp.int32),
            player=_get_centered_player(self.consts),
            wave_end_cooldown_remaining=jnp.zeros((), dtype=jnp.int32),
            aim_dx=jnp.zeros((), dtype=jnp.int32),
            aim_dy=jnp.zeros((), dtype=jnp.int32),
            discs=_get_empty_discs(self.consts),
            fire_down_prev=jnp.array(False),
            doors=self.initial_doors,
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
