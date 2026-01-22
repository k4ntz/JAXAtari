from jaxatari.renderers import JAXGameRenderer
from typing import NamedTuple, Tuple, TypeVar, Dict, Any
from jax import Array, jit, random, numpy as jnp, vmap
import jax
from functools import partial
import jaxatari.rendering.jax_rendering_utils as render_utils
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action, EnvObs
import jaxatari.spaces as spaces
import jax.lax
import os

SIDE_TOP, SIDE_BOTTOM, SIDE_LEFT, SIDE_RIGHT = 0, 1, 2, 3

def _create_static_procedural_sprites(c: 'TronConstants') -> dict:
    """Creates procedural sprites that don't depend on dynamic values or file loading."""
    # Door sprites (solid color sprites)
    door_spawn_sprite = _solid_sprite(c.door_h, c.door_w, c.rgba_door_spawn)
    door_locked_sprite = _solid_sprite(c.door_h, c.door_w, c.rgba_door_locked)
    
    return {
        'door_spawn': door_spawn_sprite,
        'door_locked': door_locked_sprite,
    }

def _get_default_asset_config() -> tuple:
    """
    Returns the default declarative asset manifest for Tron.
    Kept immutable (tuple of dicts) to fit NamedTuple defaults.
    Note: Most assets are procedurally generated from loaded sprites in the renderer.
    This returns an empty config that will be populated in the renderer.
    """
    # Tron's asset config is mostly built dynamically from loaded sprites
    # We return an empty tuple here, and the renderer will build the full config
    # This allows the structure to be in constants while keeping file-dependent logic in renderer
    return ()

class TronConstants(NamedTuple):
    screen_width: int = 160
    screen_height: int = 210
    scaling_factor: int = 3

    # Player
    player_speed: int = 1  # Player speed in pixels per frame
    player_lives: int = 5  # starting number of lives
    # Player color progression by number of hits taken (index 0..5)
    # 0 = freshly spawned (5/5 lives), 1 = after first hit, ... 4 = fourth hit (1/5),
    # 5 = fifth hit color (0/5) used for death blink alternating with index 4.
    player_life_colors_rgba: Tuple[Tuple[int, int, int, int], ...] = (
        (18, 19, 157, 255),  # fresh
        (114, 39, 164, 255),  # 1st hit
        (160, 39, 132, 255),  # 2nd hit
        (210, 81, 80, 255),  # 3rd hit
        (189, 134, 50, 255),  # 4th hit
        (205, 155, 62, 255),  # 5th hit (blink color, alternates vs index 4)
    )
    # Death blink: 5 blinks, each “half” lasts this many frames
    player_blink_cycles: int = 5
    player_blink_period_frames: int = 6
    player_animation_steps: int = (
        6  # interval for changing the animation-sprite of the walking player
    )

    # discs
    max_discs: int = 2  # Number of disc slots (one for player, one enemy)
    disc_size_out: Tuple[int, int] = (
        4,
        2,
    )  # size for the outbound discs: width 4, height 2
    disc_size_ret: Tuple[int, int] = (2, 4)  # size for the returning discs
    disc_speed: int = 2  # outbound (thrown) speed
    enemy_disc_speed: int = 2  # 1px/step
    inbound_disc_speed: int = 3

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
    score_digits: int = 6
    score_spacing: int = 2  # horizontal spacing between digits
    score_y_offset: int = 5

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
    create_new_door_prob: float = (
        0.5  # probability that a new door gets created on spawn
    )

    # enemies
    max_enemies: int = 3  # simultaneously there can only be three enemies in the arena
    enemy_spawn_offset: int = 3  # distance in pixels from when spawning
    enemy_respawn_timer: int = 200  # time in frames until the next enemy spawns
    enemy_recalc_target: Tuple[int, int] = (
        200,
        350,
    )  # After how many frames should the target be recalculated? min,max
    enemy_speed: int = 3  # inversed: move envery third frame
    enemy_target_radius: int = (
        50  # radius (px) around player to sample target (where to walk)
    )

    # Asset config baked into constants (immutable default) for asset overrides
    # Note: Tron's assets are mostly procedurally generated from loaded sprites,
    # so the renderer builds the full config from loaded files
    ASSET_CONFIG: tuple = _get_default_asset_config()
    enemy_min_dist: int = 32  # min distance between the enemies
    enemy_firing_cooldown_range: Tuple[int, int] = (
        30,
        60,
    )  # frames until the enemy can fire again
    enemy_animation_steps: int = (
        6  # interval for changing the animation-sprite of walking enemies
    )

    # gameplay
    wave_timeout: int = 200  # frames between enemy waves

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
    rgba_score_color: Tuple[int, int, int, int] = (151, 163, 67, 255)

    # waves
    num_waves: int = 8  # How many distinct waves/colors exist
    wave_points: Tuple[int, ...] = (10, 20, 40, 75, 150, 300, 500, 800)
    # Enemy tint per wave/color (RGBA)
    #  LIGHT GREEN,  DARK GREEN,     BLUE,           WHITE,
    #  YELLOW,       BLACK,          RED,            GOLD
    wave_enemy_colors_rgba: Tuple[Tuple[int, int, int, int], ...] = (
        (151, 163, 67, 255),  # light green
        (0, 100, 0, 255),  # dark  green
        (0, 128, 255, 255),  # blue
        (255, 255, 255, 255),  # white
        (255, 255, 0, 255),  # yellow
        (16, 16, 16, 255),  # black
        (220, 20, 60, 255),  # red (crimson)
        (255, 215, 0, 255),  # gold
    )


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
    fx: Array  # float32 sub-pixel x (keeps diagonal speed equal to axial speed)
    fy: Array  # float32 sub-pixel y


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
    #  sub-pixel position & velocity for enemy bullets
    fx: Array  # (D,) float32 precise x
    fy: Array  # (D,) float32 precise y
    fvx: Array  # (D,) float32 velocity x (set once at spawn)
    fvy: Array  # (D,) float32 velocity y (set once at spawn)
    shooter_id: Array  # int32: -1 for player discs, else enemy index 0..max-enemies-1


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
    facing_dx: Array  # () int32, -1 = left, +1 = right
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
    wave_index: Array  # int32: which wave/color is active (0..num_waves -1)
    enemy_global_fire_cd: Array  # int32: frames until the next enemy shot
    enemy_next_shooter: Array  # int32: which enemy id will fire next
    enemy_disc_active_prev: Array  # bool : was an enemy bullet active last frame?
    player_blink_ticks_remaining: Array  # int32: frames left in death-blink
    player_gone: Array  # bool: player has disappeared after blink


class EntityPosition(NamedTuple):
    x: Array
    y: Array
    width: Array
    height: Array


class TronObservation(NamedTuple):
    score: Array  # int32 scalar
    wave_index: Array  # () int32

    # Player (single box as 1-length arrays) + status
    player: EntityPosition  # x,y,w,h (shape (1,))
    player_lives: Array  # (1,) int32
    player_gone: Array  # () bool

    # Enemies
    enemies: EntityPosition  # x,y,w,h (shape (max_enemies,))
    enemies_alive: Array  # (max_enemies,) bool

    # Discs
    discs: EntityPosition  # x,y,w,h (shape (max_discs,))
    disc_owner: Array  # (max_discs,) int32  (0=player,1=enemy)
    disc_phase: Array  # (max_discs,) int32  (0=inactive,1=outbound,2=returning)

    # Doors
    doors: EntityPosition  # x,y,w,h (shape (max_doors,))
    door_spawned: Array  # (max_doors,) bool (visible)
    door_locked: Array  # (max_doors,) bool (locked-open / teleportable)


class TronInfo(NamedTuple):
    score: Array
    wave_index: Array
    # Counts
    enemies_alive_count: Array  # () int32
    discs_active_count: Array  # () int32
    enemy_disc_active: Array  # () bool

    # Player status
    player_lives: Array  # (1,) int32
    player_gone: Array  # () bool
    player_blink_ticks_remaining: Array  # () int32

    # Cooldowns / timers / flags
    wave_end_cooldown_remaining: Array  # () int32
    inwave_spawn_cd: Array  # () int32
    enemy_global_fire_cd: Array  # () int32
    game_started: Array  # () bool
    frame_idx: Array  # () int32


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

    @staticmethod
    @jit
    def tick_door_lockdown(doors: Doors) -> Doors:
        """Decrement per-door spawn cooldown timers (floored at 0)."""
        return doors._replace(spawn_lockdown=jnp.maximum(doors.spawn_lockdown - 1, 0))


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


# --- Helper functions for procedural asset creation ---
def _solid_sprite(h: int, w: int, rgba: Tuple[int, int, int, int]) -> Array:
    """Creates a JAX array for a solid color sprite."""
    return jnp.broadcast_to(jnp.asarray(rgba, dtype=jnp.uint8), (h, w, 4))

def _normalize_rgba(arr: jnp.ndarray) -> jnp.ndarray:
    """Ensure sprite is RGBA uint8."""
    if arr.ndim == 2:  # Grayscale
        mask = (arr > 0).astype(jnp.uint8)
        arr_3c = jnp.stack([arr, arr, arr], axis=-1)
        alpha = mask * 255
        return jnp.concatenate([arr_3c, alpha[..., None]], axis=-1).astype(jnp.uint8)
    if arr.shape[-1] == 3:  # RGB
        a = (jnp.max(arr, axis=-1, keepdims=True) > 0).astype(jnp.uint8) * 255
        return jnp.concatenate([arr, a], axis=-1).astype(jnp.uint8)
    return arr.astype(jnp.uint8)

def _load_and_normalize_seq(sprite_path: str, prefix: str) -> jnp.ndarray:
    """Loads a sequence of 4 sprites (e.g., 'player1.npy'...) and normalizes to RGBA."""
    frames = []
    for i in range(1, 5):
        path = os.path.join(sprite_path, f"{prefix}{i}.npy")
        frame = jnp.load(path)
        if not isinstance(frame, jnp.ndarray):
            raise FileNotFoundError(path)
        frames.append(_normalize_rgba(frame))
    return jnp.stack(frames, axis=0)

def _load_base_digit_sprites(sprite_path: str) -> Tuple[Array, int, int]:
    """Loads 0..9.npy files, normalizes to RGBA, and stacks them."""
    frames = []
    for d in range(10):
        path = os.path.join(sprite_path, f"{d}.npy")
        frame = jnp.load(path)
        if not isinstance(frame, jnp.ndarray):
            raise FileNotFoundError(path)
        frames.append(_normalize_rgba(frame))
    
    arr = jnp.stack(frames, axis=0)  # (10, H, W, 4)
    H = int(arr.shape[1])
    W = int(arr.shape[2])
    return arr, W, H

@jit
def _tint_rgba(sprite_rgba: Array, rgba_any: Array) -> Array:
    """Colorizes a base sprite by multiplying channels (alpha preserved)."""
    base_rgb = sprite_rgba[..., :3].astype(jnp.float32)
    alpha = sprite_rgba[..., 3:4].astype(jnp.uint8)
    color_rgb = jnp.asarray(rgba_any, dtype=jnp.float32)[:3]  # (3,)
    
    rgb_tinted = jnp.clip(jnp.round((base_rgb / 255.0) * color_rgb), 0, 255).astype(
        jnp.uint8
    )
    return jnp.concatenate([rgb_tinted, alpha], axis=-1)

@partial(jit, static_argnames=("h", "w"))
def _make_solid_sprites(colors: Array, h: int, w: int) -> Array:
    """Build one solid color sprite (h, w, 4) for each color in colors (N, 4)."""
    # colors shape (N, 4) -> reshape to (N, 1, 1, 4)
    # broadcast to (N, h, w, 4)
    return jnp.broadcast_to(colors[:, None, None, :], (colors.shape[0], h, w, 4))

@jit
def _precompute_tints(seq: Array, colors: Array) -> Array:
    """
    Tints a sprite sequence (F, H, W, 4) by a color array (C, 4).
    Returns (C, F, H, W, 4).
    """
    def tint_one(color, frame):
        return _tint_rgba(frame, color)
    # vmap over frames (F), then vmap over colors (C)
    # Result shape (F, C, H, W, 4)
    tinted = vmap(
        lambda frame: vmap(lambda col: tint_one(col, frame))(colors)
    )(seq)
    # Reorder to (C, F, H, W, 4)
    return jnp.transpose(tinted, (1, 0, 2, 3, 4))


class TronRenderer(JAXGameRenderer):

    def __init__(self, consts: TronConstants = None) -> None:
        super().__init__()
        self.consts = consts or TronConstants()
        c = self.consts
        
        # 1. Configure the new renderer
        self.config = render_utils.RendererConfig(
            game_dimensions=(c.screen_height, c.screen_width),
            channels=3,
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)
        self.sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/tron"
        
        # 1.5. Compute arena geometry (needed for background generation)
        (self.game_rect, self.score_rect, self.border_rects, self.inner_rect) = (
            _ArenaOps.compute_arena(self.consts)
        )
        
        # 2. Create procedural background (RGBA array)
        procedural_background = self._build_static_background(c)
        
        # 3. Load base (grayscale/white) sprites
        player_seq = _load_and_normalize_seq(self.sprite_path, "player")
        enemy_seq = _load_and_normalize_seq(self.sprite_path, "enemy")
        base_digits, digit_w, digit_h = _load_base_digit_sprites(self.sprite_path)
        
        # 4. Create color arrays
        player_colors = jnp.asarray(c.player_life_colors_rgba, dtype=jnp.uint8)
        wave_colors = jnp.asarray(c.wave_enemy_colors_rgba, dtype=jnp.uint8)
        score_color = jnp.asarray(c.rgba_score_color, dtype=jnp.uint8)
        
        # 5. Pre-tint all procedural assets
        player_frames_by_color = _precompute_tints(player_seq, player_colors)  # (C, F, H, W, 4)
        enemy_frames_by_wave = _precompute_tints(enemy_seq, wave_colors)  # (W, F, H, W, 4)
        tinted_digits = vmap(lambda fr: _tint_rgba(fr, score_color))(base_digits)  # (10, H, W, 4)
        player_disc_out = _make_solid_sprites(
            player_colors, c.disc_size_out[1], c.disc_size_out[0]
        )  # (C, H, W, 4)
        player_disc_ret = _make_solid_sprites(
            player_colors, c.disc_size_ret[1], c.disc_size_ret[0]
        )  # (C, H, W, 4)
        enemy_disc_out = _make_solid_sprites(
            wave_colors, c.disc_size_out[1], c.disc_size_out[0]
        )  # (W, H, W, 4)
        door_spawn_sprite = _solid_sprite(c.door_h, c.door_w, c.rgba_door_spawn)
        door_locked_sprite = _solid_sprite(c.door_h, c.door_w, c.rgba_door_locked)
        
        # Reshape 5D arrays to 4D for asset loading (flatten color and frame dimensions)
        # player_frames_by_color: (C, F, H, W, 4) -> (C*F, H, W, 4)
        n_player_colors = player_frames_by_color.shape[0]
        n_player_frames = player_frames_by_color.shape[1]
        player_frames_flat = player_frames_by_color.reshape(-1, *player_frames_by_color.shape[2:])
        
        # enemy_frames_by_wave: (W, F, H, W, 4) -> (W*F, H, W, 4)
        n_waves = enemy_frames_by_wave.shape[0]
        n_enemy_frames = enemy_frames_by_wave.shape[1]
        enemy_frames_flat = enemy_frames_by_wave.reshape(-1, *enemy_frames_by_wave.shape[2:])
        
        # 6. Start from (possibly modded) asset config provided via constants
        final_asset_config = list(self.consts.ASSET_CONFIG)
        
        # 6.1. Build the full asset manifest from loaded sprites
        # Note: Most assets are procedurally generated from loaded files
        static_procedural = _create_static_procedural_sprites(c)
        
        final_asset_config.extend([
            {'name': 'background', 'type': 'background', 'data': procedural_background},
            
            # Pre-tinted animation stacks (reshaped to 4D for asset loading)
            {'name': 'player_all_tints', 'type': 'procedural', 'data': player_frames_flat},
            {'name': 'enemy_all_tints', 'type': 'procedural', 'data': enemy_frames_flat},
            
            # Pre-tinted digits
            {'name': 'digits_tinted', 'type': 'procedural', 'data': tinted_digits},
            
            # Doors (from static procedural)
            {'name': 'door_spawn', 'type': 'procedural', 'data': static_procedural['door_spawn']},
            {'name': 'door_locked', 'type': 'procedural', 'data': static_procedural['door_locked']},
            
            # Pre-tinted disc stacks
            {'name': 'player_disc_out_all_tints', 'type': 'procedural', 'data': player_disc_out},
            {'name': 'player_disc_ret_all_tints', 'type': 'procedural', 'data': player_disc_ret},
            {'name': 'enemy_disc_out_all_tints', 'type': 'procedural', 'data': enemy_disc_out},
        ])
        
        # 7. Load all assets and build palette/masks
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(final_asset_config, self.sprite_path)
        
        # 8. Store sprite dimensions
        # player_all_tints mask shape is (C*F, H, W) after flattening
        self.PLAYER_H, self.PLAYER_W = self.SHAPE_MASKS["player_all_tints"].shape[1:3]
        self.N_PLAYER_FRAMES = n_player_frames
        self.N_PLAYER_COLORS = n_player_colors
        
        # enemy_all_tints mask shape is (W*F, H, W) after flattening
        self.ENEMY_H, self.ENEMY_W = self.SHAPE_MASKS["enemy_all_tints"].shape[1:3]
        self.N_ENEMY_FRAMES = n_enemy_frames
        self.N_WAVES = n_waves
        
        # digits_tinted mask shape is (10, H, W)
        self.DIGIT_H, self.DIGIT_W = self.SHAPE_MASKS["digits_tinted"].shape[1:3]
    
    def _build_static_background(self, c: TronConstants) -> Array:
        """
        Compose the static background layer for the game as an RGBA array.
        """
        # Initialize with black background (alpha = 255)
        background = jnp.zeros((c.screen_height, c.screen_width, 4), dtype=jnp.uint8)
        background = background.at[:, :, 3].set(255)
        
        game, score = self.game_rect, self.score_rect
        top, bottom, left, right = self.border_rects

        # Fill gray gamefield
        background = background.at[
            game.y : game.y + game.h,
            game.x : game.x + game.w,
            :
        ].set(jnp.asarray(c.rgba_gray, dtype=jnp.uint8))

        # Fill green scorebar
        background = background.at[
            score.y : score.y + score.h,
            score.x : score.x + score.w,
            :
        ].set(jnp.asarray(c.rgba_green, dtype=jnp.uint8))

        # Fill purple borders
        background = background.at[
            top.y : top.y + top.h,
            top.x : top.x + top.w,
            :
        ].set(jnp.asarray(c.rgba_purple, dtype=jnp.uint8))
        
        background = background.at[
            bottom.y : bottom.y + bottom.h,
            bottom.x : bottom.x + bottom.w,
            :
        ].set(jnp.asarray(c.rgba_purple, dtype=jnp.uint8))
        
        background = background.at[
            left.y : left.y + left.h,
            left.x : left.x + left.w,
            :
        ].set(jnp.asarray(c.rgba_purple, dtype=jnp.uint8))
        
        background = background.at[
            right.y : right.y + right.h,
            right.x : right.x + right.w,
            :
        ].set(jnp.asarray(c.rgba_purple, dtype=jnp.uint8))
        
        return background

    @partial(jit, static_argnums=(0,))
    def render(self, state: TronState) -> Array:
        c = self.consts
        
        # 1. Start from cached background ID raster
        raster = self.jr.create_object_raster(self.BACKGROUND)
        
        # 2. Render Score Digits
        s = jnp.mod(state.score, jnp.int32(1_000_000))
        d0 = s % 10
        d1 = (s // 10) % 10
        d2 = (s // 100) % 10
        d3 = (s // 1_000) % 10
        d4 = (s // 10_000) % 10
        d5 = (s // 100_000) % 10
        digits_idx = jnp.stack([d5, d4, d3, d2, d1, d0], axis=0)
        digit_w = jnp.int32(self.DIGIT_W)
        digit_h = jnp.int32(self.DIGIT_H)
        spacing_val = jnp.int32(c.score_spacing)
        count = jnp.int32(c.score_digits)
        total_w = count * digit_w + (count - 1) * spacing_val
        x0 = jnp.int32(self.score_rect.x) + (
            jnp.int32(self.score_rect.w) - total_w
        ) // jnp.int32(2)
        
        y0_centered = jnp.int32(self.score_rect.y) + (
            jnp.int32(self.score_rect.h) - digit_h
        ) // jnp.int32(2)
        y0 = y0_centered + c.score_y_offset
        
        # Use the new render_label function
        # spacing and max_digits must be Python ints (static args)
        raster = self.jr.render_label(
            raster, 
            x0, 
            y0, 
            digits_idx, 
            self.SHAPE_MASKS["digits_tinted"], 
            spacing=self.DIGIT_W + c.score_spacing,  # Python int computation
            max_digits=c.score_digits  # Python int from constants
        )
        
        # 3. Render Doors
        def render_door(i, ras):
            doors = state.doors
            active = doors.is_spawned[i]
            def draw(r):
                # Select the correct ID mask
                mask = jax.lax.select(
                    doors.is_locked_open[i],
                    self.SHAPE_MASKS["door_locked"],
                    self.SHAPE_MASKS["door_spawn"],
                )
                # Use the new render_at
                return self.jr.render_at(r, doors.x[i], doors.y[i], mask)
            return jax.lax.cond(active, draw, lambda r: r, ras)
        raster = jax.lax.fori_loop(0, c.max_doors, render_door, raster)
        
        # 4. Render Player
        def draw_player(r):
            # Get correct color index
            color_idx = player_color_index(
                state.player.lives[0],
                c.player_lives,
                state.player_blink_ticks_remaining,
                c.player_blink_period_frames,
            )
            
            # Get correct animation frame index
            moving = jnp.any((state.player.vx != 0) | (state.player.vy != 0))
            step = jnp.int32(c.player_animation_steps)
            fidx = jax.lax.select(
                moving,
                jnp.mod(state.frame_idx // step, jnp.int32(self.N_PLAYER_FRAMES)),
                jnp.int32(0),
            )
            
            # Select the pre-tinted mask (flattened index: color_idx * n_frames + frame_idx)
            flat_idx = color_idx * jnp.int32(self.N_PLAYER_FRAMES) + fidx
            player_mask = self.SHAPE_MASKS["player_all_tints"][flat_idx]
            
            # Flipping logic
            face_left = state.facing_dx < jnp.int32(0)
            
            return self.jr.render_at(
                r, 
                state.player.x[0], 
                state.player.y[0], 
                player_mask, 
                flip_horizontal=face_left,
                flip_offset=self.FLIP_OFFSETS["player_all_tints"]
            )
        raster = jax.lax.cond(state.player_gone, lambda r: r, draw_player, raster)
        
        # 5. Render Discs
        # Get target color indices for this frame
        player_color_idx = player_color_index(
            state.player.lives[0],
            c.player_lives,
            state.player_blink_ticks_remaining,
            c.player_blink_period_frames,
        )
        wave_idx = jnp.clip(state.wave_index, 0, jnp.int32(c.num_waves - 1))
        
        def render_disc(i, ras):
            active = state.discs.phase[i] > jnp.int32(0)
            xi = state.discs.x[i]
            yi = state.discs.y[i]
            def draw(r):
                owner = state.discs.owner[i]
                phase = state.discs.phase[i]
                is_player = owner == jnp.int32(0)
                
                # Player discs have different shapes for outbound vs returning
                def draw_player(rr):
                    return jax.lax.cond(
                        phase == jnp.int32(1),  # outbound
                        lambda r2: self.jr.render_at(r2, xi, yi, self.SHAPE_MASKS["player_disc_out_all_tints"][player_color_idx]),
                        lambda r2: self.jr.render_at(r2, xi, yi, self.SHAPE_MASKS["player_disc_ret_all_tints"][player_color_idx]),
                        rr
                    )
                
                # Enemy discs always use the same shape
                def draw_enemy(rr):
                    mask = self.SHAPE_MASKS["enemy_disc_out_all_tints"][wave_idx]
                    return self.jr.render_at(rr, xi, yi, mask)
                
                return jax.lax.cond(is_player, draw_player, draw_enemy, r)
            return jax.lax.cond(active, draw, lambda r: r, ras)
        raster = jax.lax.fori_loop(0, c.max_discs, render_disc, raster)
        
        # 6. Render Enemies
        e_step = jnp.int32(c.enemy_animation_steps)
        e_idx = jnp.mod(state.frame_idx // e_step, jnp.int32(self.N_ENEMY_FRAMES))
        
        # Get the pre-tinted mask for this wave color and anim frame (flattened index)
        enemy_flat_idx = wave_idx * jnp.int32(self.N_ENEMY_FRAMES) + e_idx
        enemy_mask = self.SHAPE_MASKS["enemy_all_tints"][enemy_flat_idx]
        def render_enemy(i, ras):
            alive = state.enemies.alive[i]
            ex = state.enemies.x[i]
            ey = state.enemies.y[i]
            
            return jax.lax.cond(
                alive, 
                lambda r: self.jr.render_at(r, ex, ey, enemy_mask, flip_offset=self.FLIP_OFFSETS["enemy_all_tints"]), 
                lambda r: r, 
                ras
            )
        raster = jax.lax.fori_loop(0, c.max_enemies, render_enemy, raster)
        
        # 7. Final Palette Lookup (NO color swapping needed)
        return self.jr.render_from_palette(raster, self.PALETTE)


####
# Helper functions
####
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
def player_color_index(
    lives,  # () int32
    max_lives,  # python int or () int32
    blink_ticks_remaining,  # () int32
    blink_period_frames,  # python int or () int32
):
    # Map lives to color index
    # - while alive (lives > 0): index increases with number of hits (0..4)
    # - when dead (lives == 0): blink by alternating between indices 5 and 4
    max_lives = jnp.int32(max_lives)
    hits = jnp.clip(max_lives - lives, 0, jnp.int32(5))
    base_idx = jnp.clip(hits, 0, jnp.int32(4))

    # Compute blink toggle: every player_blink_period_frames frames flip 0/1
    period = jnp.int32(blink_period_frames)
    toggle = jnp.mod(blink_ticks_remaining // period, 2)

    return jax.lax.select(
        lives == jnp.int32(0),  # dead → blink 5/4
        jax.lax.select(toggle == 0, jnp.int32(5), jnp.int32(4)),
        base_idx,  # alive → 0..4
    )


@jit
def tick_cd(x: Array) -> Array:
    # Decrement by 1 but never below 0, preserving dtype
    one = jnp.asarray(1, dtype=x.dtype)
    zero = jnp.asarray(0, dtype=x.dtype)
    return jnp.maximum(x - one, zero)


@jit
def _select_door_for_spawn(
    doors: Doors,
    rng_key: random.PRNGKey,
    prefer_new_prob: float,
) -> Tuple[Array, Array, Doors, random.PRNGKey]:
    """
    Choose a door index to spawn an enemy from:
      - prefer reusing an existing spawned door
      - otherwise, try to spawn (make visible) a new door
      - fallback: any spawned & unlocked door if all quadrants are busy
      - when BOTH reuse and new are available, pick NEW with probability `prefer_new_prob`

    Returns (has_choice, door_index, updated_doors, next_rng_key).
    """
    # total number of doors
    n_doors = doors.x.shape[0]

    # All doors have a lockdown, between spawning enemies
    # Select only those, with a lockdown of 0
    door_unlocked = doors.spawn_lockdown == jnp.int32(0)

    # Select doors, that are already spawned, have a cooldown of 0
    reuse_mask = doors.is_spawned & door_unlocked
    # Select doors, that are not yet spawned, have a cooldown of 0
    new_mask = (~doors.is_spawned) & door_unlocked

    reuse_cnt = jnp.sum(reuse_mask.astype(jnp.int32))
    new_cnt = jnp.sum(new_mask.astype(jnp.int32))

    has_reuse = reuse_cnt > 0
    has_new = new_cnt > 0

    rng_key, k_pick_set, k_pick_new, k_pick_reuse = random.split(rng_key, 4)
    pick_new_sample = random.bernoulli(k_pick_set, p=jnp.float32(prefer_new_prob))

    choose_new = has_new & (~has_reuse | pick_new_sample)
    choose_reuse = has_reuse & (~has_new | (~pick_new_sample))

    def _sample(mask, count, key):
        idx_pad = jnp.nonzero(mask, size=n_doors)[0]
        pos = random.randint(key, (), 0, jnp.maximum(count, 1), dtype=jnp.int32)
        return idx_pad[pos]

    def _pick_reuse(_):
        idx = _sample(reuse_mask, reuse_cnt, k_pick_reuse)
        return True, idx, doors, rng_key

    def _pick_new(_):
        idx = _sample(new_mask, new_cnt, k_pick_new)
        upd = doors._replace(is_spawned=doors.is_spawned.at[idx].set(True))
        return True, idx, upd, rng_key

    def _fallback(_):
        # reuse_mask==spawned&unlocked, and we already know has_reuse==False here,
        # so return (False,0,…) to signal no choice.
        return False, jnp.int32(0), doors, rng_key

    return jax.lax.cond(
        choose_new,
        _pick_new,
        lambda _: jax.lax.cond(choose_reuse, _pick_reuse, _fallback, operand=None),
        operand=None,
    )


class JaxTron(JaxEnvironment[TronState, TronObservation, TronInfo, TronConstants]):
    # Minimal ALE action set for Tron (from scripts/action_space_helper.py)
    ACTION_SET: jnp.ndarray = jnp.array(
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

    def __init__(
        self, consts: TronConstants = None
    ) -> None:
        consts = consts or TronConstants()
        super().__init__(consts)
        self.renderer = TronRenderer(consts)

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

        self.player_w = jnp.int32(self.renderer.PLAYER_W)
        self.player_h = jnp.int32(self.renderer.PLAYER_H)
        self.enemy_w = jnp.int32(self.renderer.ENEMY_W)
        self.enemy_h = jnp.int32(self.renderer.ENEMY_H)

        # Disc sizes (match renderer atlases)
        self.disc_w = jnp.int32(self.consts.disc_size_out[0])  # 4
        self.disc_h = jnp.int32(self.consts.disc_size_out[1])  # 2

    def reset(self, key: random.PRNGKey = None) -> Tuple[TronObservation, TronState]:
        def _get_centered_player(consts: TronConstants) -> Player:
            screen_w, screen_h = consts.screen_width, consts.screen_height
            player_w, player_h = self.player_w, self.player_h
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
                fx=jnp.array([jnp.float32(x0)], jnp.float32),
                fy=jnp.array([jnp.float32(y0)], jnp.float32),
            )

        def _get_empty_discs(consts: TronConstants) -> Discs:
            D = consts.max_discs
            w = jnp.full((D,), self.disc_w, dtype=jnp.int32)
            h = jnp.full((D,), self.disc_h, dtype=jnp.int32)

            zeros = jnp.zeros((D,), dtype=jnp.int32)
            zf = jnp.zeros((D,), dtype=jnp.float32)
            neg1 = jnp.full((D,), -1, dtype=jnp.int32)

            return Discs(
                x=zeros,
                y=zeros,
                vx=zeros,
                vy=zeros,
                w=w,
                h=h,
                owner=zeros,
                phase=zeros,
                fx=zf,
                fy=zf,
                fvx=zf,
                fvy=zf,
                shooter_id=neg1,
            )

        def _get_empty_enemies(consts: TronConstants) -> Enemies:
            N = consts.max_enemies
            ew = jnp.full((N,), self.enemy_w, dtype=jnp.int32)
            eh = jnp.full((N,), self.enemy_h, dtype=jnp.int32)
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
            facing_dx=jnp.int32(1),
            discs=_get_empty_discs(self.consts),
            fire_down_prev=jnp.array(False),
            doors=self.initial_doors,
            enemies=_get_empty_enemies(self.consts),
            game_started=jnp.array(False),
            inwave_spawn_cd=jnp.zeros((), dtype=jnp.int32),
            enemies_alive_last=jnp.zeros((), dtype=jnp.int32),
            rng_key=key,
            frame_idx=jnp.zeros((), dtype=jnp.int32),
            wave_index=jnp.zeros((), dtype=jnp.int32),
            enemy_global_fire_cd=jnp.int32(0),
            enemy_next_shooter=jnp.int32(0),
            enemy_disc_active_prev=jnp.array(False),
            player_blink_ticks_remaining=jnp.int32(0),
            player_gone=jnp.array(False),
        )
        obs = self._get_observation(new_state)
        return obs, new_state

    @partial(jit, static_argnums=(0,))
    def _player_step(self, state: TronState, action: UserAction) -> TronState:
        player = state.player
        active = (~state.player_gone) & (state.player.lives[0] > jnp.int32(0))

        def do_move(s: TronState) -> TronState:
            speed = jnp.float32(self.consts.player_speed)

            # direct input per axis in {-1, 0, 1}
            raw_dx_i = action.right.astype(jnp.int32) - action.left.astype(jnp.int32)
            raw_dy_i = action.down.astype(jnp.int32) - action.up.astype(jnp.int32)

            # convert to float direction and compute L2 norm
            dx_f = raw_dx_i.astype(jnp.float32)
            dy_f = raw_dy_i.astype(jnp.float32)
            norm = jnp.maximum(jnp.sqrt(dx_f * dx_f + dy_f * dy_f), jnp.float32(1.0))

            # Normalize velocity * speed -> constant Eucld speed in any direction
            fvx = (dx_f / norm) * speed
            fvy = (dy_f / norm) * speed

            # precise (float) integration
            fx_try = s.player.fx[0] + fvx
            fy_try = s.player.fy[0] + fvy

            # integer intention (for raster/collision), then clamp to inner play area
            xi_try = jnp.round(fx_try).astype(jnp.int32)
            yi_try = jnp.round(fy_try).astype(jnp.int32)
            xi_clamp = jnp.clip(
                xi_try, self.inner_min_x, self.inner_max_x - s.player.w[0]
            )
            yi_clamp = jnp.clip(
                yi_try, self.inner_min_y, self.inner_max_y - s.player.h[0]
            )

            # final integer position
            x_next = xi_clamp
            y_next = yi_clamp

            # keep sub-pixel accumulation; only snap floats when we actually clamped
            fx_next = jnp.where(
                xi_clamp != xi_try, xi_clamp.astype(jnp.float32), fx_try
            )
            fy_next = jnp.where(
                yi_clamp != yi_try, yi_clamp.astype(jnp.float32), fy_try
            )

            # keep raw inputs for animations/door logic
            vx_int = jnp.full_like(s.player.vx, raw_dx_i)
            vy_int = jnp.full_like(s.player.vy, raw_dy_i)

            # remember last aim direction from raw input
            aim_dx = jnp.where(action.moved, raw_dx_i, s.aim_dx)
            aim_dy = jnp.where(action.moved, raw_dy_i, s.aim_dy)

            # remember last non-zero horizontal input for sprite facing
            facing_dx = jnp.where(raw_dx_i != 0, raw_dx_i, s.facing_dx)

            player2 = s.player._replace(
                x=s.player.x.at[0].set(x_next),
                y=s.player.y.at[0].set(y_next),
                vx=vx_int,
                vy=vy_int,
                fx=s.player.fx.at[0].set(fx_next),
                fy=s.player.fy.at[0].set(fy_next),
            )

            return s._replace(
                player=player2, aim_dx=aim_dx, aim_dy=aim_dy, facing_dx=facing_dx
            )

        return jax.lax.cond(active, do_move, lambda s: s, state)

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
        alive_ok = (~state.player_gone) & (state.player.lives[0] > jnp.int32(0))
        can_spawn: Array = fire & (~has_player_disc) & any_free & alive_ok

        def do_spawn(s: TronState) -> TronState:
            # get the index of a free slot
            free_indices: Array = jnp.nonzero(s.discs.phase == 0, size=1)[0][0]

            # Center the discs in the player-box
            px_all = (
                s.player.x[0] + jnp.floor_divide(s.player.w[0] - s.discs.w, 2)
            ).astype(jnp.int32)
            py_all = (
                s.player.y[0] + jnp.floor_divide(s.player.h[0] - s.discs.h, 2)
            ).astype(jnp.int32)

            px, py = px_all[free_indices], py_all[free_indices]

            # float, normalized outbound velocity for player discs
            speed_f = jnp.float32(self.consts.disc_speed)  # 2.0
            dx_f = jnp.sign(s.aim_dx).astype(jnp.float32)  # -1/0/1
            dy_f = jnp.sign(s.aim_dy).astype(jnp.float32)
            norm = jnp.sqrt(dx_f * dx_f + dy_f * dy_f)
            inv = jnp.float32(1.0) / jnp.maximum(norm, jnp.float32(1.0))  # avoid /0
            fvx = dx_f * inv * speed_f
            fvy = dy_f * inv * speed_f

            # writes in the free slot
            new_discs: Discs = s.discs._replace(
                x=s.discs.x.at[free_indices].set(px),
                y=s.discs.y.at[free_indices].set(py),
                vx=s.discs.vx.at[free_indices].set(jnp.int32(0)),
                vy=s.discs.vy.at[free_indices].set(jnp.int32(0)),
                owner=s.discs.owner.at[free_indices].set(jnp.int32(0)),
                phase=s.discs.phase.at[free_indices].set(jnp.int32(1)),  # 1 = outbound
                fx=s.discs.fx.at[free_indices].set(px.astype(jnp.float32)),
                fy=s.discs.fy.at[free_indices].set(py.astype(jnp.float32)),
                fvx=s.discs.fvx.at[free_indices].set(fvx),
                fvy=s.discs.fvy.at[free_indices].set(fvy),
                shooter_id=s.discs.shooter_id.at[free_indices].set(jnp.int32(-1)),
            )
            return s._replace(discs=new_discs)

        def no_spawn(s: TronState) -> TronState:
            return s

        return jax.lax.cond(can_spawn, do_spawn, no_spawn, state)

    @partial(jit, static_argnums=(0,))
    def _move_discs(self, state: TronState, fire_pressed: Array) -> TronState:
        discs = state.discs

        # is_enemy_out = (discs.owner == jnp.int32(1)) & (discs.phase == jnp.int32(1))
        is_out = discs.phase == jnp.int32(1)

        # next precise positions
        nx_f = discs.fx + discs.fvx
        ny_f = discs.fy + discs.fvy

        nx_i = jnp.round(nx_f).astype(jnp.int32)
        ny_i = jnp.round(ny_f).astype(jnp.int32)

        hit_next = is_out & (
            (nx_i < self.inner_min_x)
            | (ny_i < self.inner_min_y)
            | ((nx_i + discs.w) > self.inner_max_x)
            | ((ny_i + discs.h) > self.inner_max_y)
        )

        will_hit_wall_std = _DiscOps.check_wall_hit(
            discs,
            self.inner_min_x,
            self.inner_min_y,
            self.inner_max_x,
            self.inner_max_y,
        )
        # Use float-based prediction for enemy outbound discs
        will_hit_wall_next = jnp.where(is_out, hit_next, will_hit_wall_std)

        next_phase = _DiscOps.compute_next_phase(
            discs, fire_pressed, will_hit_wall_next
        )

        # Player center for homing
        pcx, pcy = rect_center(
            state.player.x[0], state.player.y[0], state.player.w[0], state.player.h[0]
        )

        velocity_x, velocity_y = _DiscOps.compute_velocity(
            discs, next_phase, pcx, pcy, self.consts.inbound_disc_speed
        )

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

        out_move_mask = is_out & (next_phase == jnp.int32(1)) & (~hit_next)
        x_next = jnp.where(
            out_move_mask,
            jnp.clip(nx_i, self.inner_min_x, self.inner_max_x - discs.w),
            x_next,
        )
        y_next = jnp.where(
            out_move_mask,
            jnp.clip(ny_i, self.inner_min_y, self.inner_max_y - discs.h),
            y_next,
        )

        # Update stored float positions for those enemy discs; zero on despawn.
        fx_next = jnp.where(out_move_mask, nx_f, discs.fx)
        fy_next = jnp.where(out_move_mask, ny_f, discs.fy)
        fvx_next = jnp.where(next_phase == jnp.int32(0), jnp.float32(0), discs.fvx)
        fvy_next = jnp.where(next_phase == jnp.int32(0), jnp.float32(0), discs.fvy)

        # Returning player discs pickup logic
        final_phase, final_vx, final_vy = _DiscOps.player_pickup_returning_discs(
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

        final_vx = jnp.where(final_phase == jnp.int32(0), jnp.int32(0), final_vx)
        final_vy = jnp.where(final_phase == jnp.int32(0), jnp.int32(0), final_vy)

        new_discs = discs._replace(
            x=x_next,
            y=y_next,
            vx=final_vx,
            vy=final_vy,
            phase=final_phase,
            fx=fx_next,
            fy=fy_next,
            fvx=fvx_next,
            fvy=fvy_next,
        )
        return state._replace(discs=new_discs)

    @partial(jit, static_argnums=(0,))
    def _spawn_pos_from_door(self, doors: Doors, idx: Array, ew: Array, eh: Array):
        ds = doors.side[idx]
        cx = doors.x[idx] + jnp.floor_divide(doors.w[idx] - ew, 2)
        cy = doors.y[idx] + jnp.floor_divide(doors.h[idx] - eh, 2)
        off = jnp.int32(self.consts.enemy_spawn_offset)

        # base centered (clamped) along the orthogonal axis
        x = jnp.clip(cx, self.inner_min_x, self.inner_max_x - ew)
        y = jnp.clip(cy, self.inner_min_y, self.inner_max_y - eh)

        is_top = ds == jnp.int32(SIDE_TOP)
        is_bottom = ds == jnp.int32(SIDE_BOTTOM)
        is_left = ds == jnp.int32(SIDE_LEFT)
        is_right = ds == jnp.int32(SIDE_RIGHT)

        x = jnp.where(is_left, self.inner_min_x + off, x)
        x = jnp.where(is_right, self.inner_max_x - ew - off, x)
        y = jnp.where(is_top, self.inner_min_y + off, y)
        y = jnp.where(is_bottom, self.inner_max_y - eh - off, y)
        return x, y

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
        """
        # How many enemy slots are free right now?
        free_mask = ~state.enemies.alive
        free_count = jnp.sum(free_mask.astype(jnp.int32))

        # We'll never spawn more than there are free slots.
        to_spawn = jnp.minimum(max_new, free_count)

        def place_one(carry_state):
            """
            Spawn exactly one enemy if a valid door is available.
            Returns the updated state (enemies, doors, rng_key possibly changed).
            """
            s = carry_state

            # Choose a door (handles reuse/new preference and returns next RNG key)
            has, door_idx, doors2, key2 = _select_door_for_spawn(
                s.doors,
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
                doors3 = doors2._replace(
                    spawn_lockdown=doors2.spawn_lockdown.at[door_idx].set(
                        jnp.int32(self.consts.door_respawn_cooldown)
                    )
                )

                return s_in._replace(enemies=enemies2, doors=doors3)

            # If no valid door, still update doors returned by the selector
            def no_spawn(s_in):
                return s_in._replace(doors=doors2)

            return jax.lax.cond(has, do_spawn, no_spawn, s)

        def scan_body(carry_state: TronState, i: Array):
            # only attempt a placement while i < to_spawn (keeps semantics)
            carry_state = jax.lax.cond(
                i < to_spawn, place_one, lambda s: s, carry_state
            )
            return carry_state, None

        state, _ = jax.lax.scan(
            scan_body,
            state,
            jnp.arange(self.consts.max_enemies, dtype=jnp.int32),
        )
        return state

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

                # THree independent uniform(0,1) vectors (flaot32). Each enemy gets one sample for
                # axis selection (only axis or diagonal), twitch chance and perpendicular "strafe" nudge
                r_axis = random.uniform(k_axis, (n,), dtype=jnp.float32)  # [0,1)
                r_twitch = random.uniform(k_twitch, (n,), dtype=jnp.float32)  # [0,1)
                r_strafe = random.uniform(k_strafe, (n,), dtype=jnp.float32)  # [0,1)

                # Uniform angles over [0, 2pi]. These pick a direction on a circle around the player
                # where an enemys "target point" is placed.
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
    def _disc_enemy_collisions(self, state: TronState) -> TronState:
        """
        Kill enemies hit by player-owned, outbound discs.
        - Discs do NOT despawn on hit (they keep flying).
        - Returning player discs (phase==2) do NOT kill.
        - Enemy-owned discs are ignored here.
        """
        e, d = state.enemies, state.discs

        # Consider only player discs that are actively flying outward.
        #   owner == 0 → player
        #   phase == 1 → outbound (NOT returning)
        disc_is_player_outbound = (d.owner == jnp.int32(0)) & (d.phase == jnp.int32(1))

        # Quick exit if nobody is alive or no relevant discs.
        any_alive = jnp.any(e.alive)
        any_disc = jnp.any(disc_is_player_outbound)
        do_check = any_alive & any_disc

        def _no_hit(s):
            return s

        def _check_and_kill(s: TronState) -> TronState:
            e2, d2 = s.enemies, s.discs

            # Broadcast rectangles to (E,D) for overlap checks.
            # Enemy bounds: [ex0, ex1) × [ey0, ey1)
            ex0 = e2.x[:, None]
            ey0 = e2.y[:, None]
            ex1 = ex0 + e2.w[:, None]
            ey1 = ey0 + e2.h[:, None]

            # Disc bounds: [dx0, dx1) × [dy0, dy1)
            dx0 = d2.x[None, :]
            dy0 = d2.y[None, :]
            dx1 = dx0 + d2.w[None, :]
            dy1 = dy0 + d2.h[None, :]

            x_overlap = (ex0 < dx1) & (ex1 > dx0)
            y_overlap = (ey0 < dy1) & (ey1 > dy0)

            pair_overlap = x_overlap & y_overlap

            # Only count hits where the enemy is alive AND the disc is player+outbound.
            mask_alive = e2.alive[:, None]
            mask_disc = disc_is_player_outbound[None, :]
            pair_hits = pair_overlap & mask_alive & mask_disc

            # Enemy i is hit if any disc j overlaps it.
            enemy_hit = jnp.any(pair_hits, axis=1)  # (E,)
            kills = jnp.sum(enemy_hit.astype(jnp.int32))

            # Kill enemies on hit; discs remain unchanged.
            new_alive = e2.alive & (~enemy_hit)
            e3 = e2._replace(alive=new_alive)

            # wave points
            wave_pts_tbl = jnp.asarray(self.consts.wave_points, dtype=jnp.int32)
            widx = jnp.clip(s.wave_index, 0, jnp.int32(self.consts.num_waves - 1))
            pts_per = wave_pts_tbl[widx]
            new_score = s.score + pts_per * kills
            # debug.print("score {s}", s=new_score)
            return s._replace(enemies=e3, score=new_score)

        return jax.lax.cond(do_check, _check_and_kill, _no_hit, operand=state)

    @partial(jit, static_argnums=(0,))
    def _update_respawn_cooldown_on_kills(self, state: TronState) -> TronState:
        alive_now = jnp.sum(state.enemies.alive.astype(jnp.int32))
        alive_prev = state.enemies_alive_last
        died_now = alive_now < alive_prev  # someone just died this frame?

        # Arm the cooldown only if we're not already counting down.
        start_cd = died_now & (state.inwave_spawn_cd == jnp.int32(0))

        new_cd = jnp.where(
            start_cd, jnp.int32(self.consts.enemy_respawn_timer), state.inwave_spawn_cd
        )

        return state._replace(
            inwave_spawn_cd=new_cd,
            enemies_alive_last=alive_now,  # keep this updated every frame
        )

    @partial(jit, static_argnums=(0,))
    def _maybe_respawn_enemy(self, state: TronState) -> TronState:
        alive_now = jnp.sum(state.enemies.alive.astype(jnp.int32))
        any_dead = alive_now < jnp.int32(self.consts.max_enemies)

        # Only tick the in-wave respawn cooldown if there is at least one enemy alive
        should_tick = any_dead & (alive_now > jnp.int32(0))
        cd_next = jnp.where(
            should_tick,
            tick_cd(state.inwave_spawn_cd),
            state.inwave_spawn_cd,
        )

        can_spawn = state.game_started & should_tick & (cd_next == jnp.int32(0))

        def _spawn_one(s: TronState) -> TronState:
            s2 = self._spawn_enemies_up_to(s, jnp.int32(1), jnp.float32(0.4))
            alive_after = jnp.sum(s2.enemies.alive.astype(jnp.int32))
            still_deficit = alive_after < jnp.int32(self.consts.max_enemies)
            # If there are still dead slots, start the next timer; else leave at 0.
            next_cd = jax.lax.select(
                still_deficit,
                jnp.int32(self.consts.enemy_respawn_timer),
                jnp.int32(0),
            )
            return s2._replace(inwave_spawn_cd=next_cd)

        def _no_spawn(s: TronState) -> TronState:
            return s._replace(inwave_spawn_cd=cd_next)

        return jax.lax.cond(can_spawn, _spawn_one, _no_spawn, state)

    @partial(jit, static_argnums=(0,))
    def _lock_doors_from_disc_hits(self, state: TronState) -> TronState:
        """
        If a player-owned, outbound disc tries to cross the inner wall this frame AND
        its span aligns with a visible door on that side, lock that door open.

        Note: discs stay inside the inner play area, so we detect hits by
        'next-step wall contact' + alignment with the door slot along that wall.
        """
        d, doors = state.discs, state.doors

        # Only player, outbound discs can lock doors
        disc_relevant = (d.owner == jnp.int32(0)) & (d.phase == jnp.int32(1))

        # Next-step bounds
        is_out = d.phase == jnp.int32(1)
        nx_out = jnp.round(d.fx + d.fvx).astype(jnp.int32)
        ny_out = jnp.round(d.fy + d.fvy).astype(jnp.int32)
        nx_std = d.x + d.vx
        ny_std = d.y + d.vy
        nx = jnp.where(is_out, nx_out, nx_std)
        ny = jnp.where(is_out, ny_out, ny_std)

        hit_left = nx < self.inner_min_x
        hit_right = (nx + d.w) > self.inner_max_x
        hit_top = ny < self.inner_min_y
        hit_bottom = (ny + d.h) > self.inner_max_y

        disc_top = disc_relevant & hit_top
        disc_bottom = disc_relevant & hit_bottom
        disc_left = disc_relevant & hit_left
        disc_right = disc_relevant & hit_right

        any_disc_might_hit = jnp.any(disc_top | disc_bottom | disc_left | disc_right)

        def _no_lock(s: TronState):
            return s

        def _do_lock(s: TronState) -> TronState:
            d2, doors2 = s.discs, s.doors

            # Disc spans
            dx0 = d2.x[None, :]
            dx1 = dx0 + d2.w[None, :]
            dy0 = d2.y[None, :]
            dy1 = dy0 + d2.h[None, :]

            # Door spans (broadcast along discs)
            tx0 = doors2.x[:, None]
            tx1 = tx0 + doors2.w[:, None]
            ty0 = doors2.y[:, None]
            ty1 = ty0 + doors2.h[:, None]

            # Alignment (AABB overlap on the axis parallel to door slot)
            x_overlap = (tx0 < dx1) & (tx1 > dx0)  # for top/bottom
            y_overlap = (ty0 < dy1) & (ty1 > dy0)  # for left/right

            # Door eligibility: must be visible (spawned) and not already locked
            visible_and_unlock = doors2.is_spawned & (~doors2.is_locked_open)

            # Side masks
            side_top = doors2.side == jnp.int32(SIDE_TOP)
            side_bottom = doors2.side == jnp.int32(SIDE_BOTTOM)
            side_left = doors2.side == jnp.int32(SIDE_LEFT)
            side_right = doors2.side == jnp.int32(SIDE_RIGHT)

            # Combine: per-door if ANY disc aligns & hits that wall this frame
            hit_top_any = jnp.any(
                x_overlap
                & side_top[:, None]
                & visible_and_unlock[:, None]
                & disc_top[None, :],
                axis=1,
            )
            hit_bot_any = jnp.any(
                x_overlap
                & side_bottom[:, None]
                & visible_and_unlock[:, None]
                & disc_bottom[None, :],
                axis=1,
            )
            hit_lft_any = jnp.any(
                y_overlap
                & side_left[:, None]
                & visible_and_unlock[:, None]
                & disc_left[None, :],
                axis=1,
            )
            hit_rgt_any = jnp.any(
                y_overlap
                & side_right[:, None]
                & visible_and_unlock[:, None]
                & disc_right[None, :],
                axis=1,
            )

            lock_hit = hit_top_any | hit_bot_any | hit_lft_any | hit_rgt_any

            new_doors = doors2._replace(
                is_locked_open=(doors2.is_locked_open | lock_hit)
            )
            return s._replace(doors=new_doors)

        return jax.lax.cond(any_disc_might_hit, _do_lock, _no_lock, state)

    @partial(jit, static_argnums=(0,))
    def _teleport_player_if_door(self, state: TronState) -> TronState:
        """
        If the player tries to move through a wall segment that has a visible, locked door
        and the paired door is also visible+locked, teleport to the paired door.
        The entry door becomes deactivated+unlocked; the exit door remains unchanged.
        """
        p, d = state.player, state.doors

        # Player box (positions are length-1 arrays; velocities may be 0-D scalars)
        px0, py0 = p.x[0], p.y[0]
        pw, ph = p.w[0], p.h[0]
        px1, py1 = px0 + pw, py0 + ph

        vx = jnp.squeeze(p.vx)
        vy = jnp.squeeze(p.vy)

        # Intent to push into a wall from the current clamped position
        up_intent = (vy < 0) & (py0 == self.inner_min_y)
        down_intent = (vy > 0) & (py0 == self.inner_max_y - ph)
        left_intent = (vx < 0) & (px0 == self.inner_min_x)
        right_intent = (vx > 0) & (px0 == self.inner_max_x - pw)

        # Local door must be visible+locked AND its pair must also be visible+locked
        local_ok = d.is_spawned & d.is_locked_open
        pair_ok = local_ok[d.pair]
        both_locked = local_ok & pair_ok

        # Overlap between player's span and a door slot on the relevant axis
        x_overlap = (d.x < px1) & ((d.x + d.w) > px0)  # for top/bottom doors
        y_overlap = (d.y < py1) & ((d.y + d.h) > py0)  # for left/right doors

        m_top = both_locked & (d.side == jnp.int32(SIDE_TOP)) & x_overlap & up_intent
        m_bottom = (
            both_locked & (d.side == jnp.int32(SIDE_BOTTOM)) & x_overlap & down_intent
        )
        m_left = (
            both_locked & (d.side == jnp.int32(SIDE_LEFT)) & y_overlap & left_intent
        )
        m_right = (
            both_locked & (d.side == jnp.int32(SIDE_RIGHT)) & y_overlap & right_intent
        )

        m_any = m_top | m_bottom | m_left | m_right
        has, idx = _find_first_true(m_any)

        def do_tp(s: TronState) -> TronState:
            doors2, player2 = s.doors, s.player

            # Exit is the paired door; place player just inside arena at that door
            exit_idx = doors2.pair[idx]
            ex, ey = self._spawn_pos_from_door(
                doors2, exit_idx, player2.w[0], player2.h[0]
            )

            # Move player
            player3 = player2._replace(
                x=player2.x.at[0].set(ex),
                y=player2.y.at[0].set(ey),
                fx=player2.fx.at[0].set(ex.astype(jnp.float32)),
                fy=player2.fy.at[0].set(ey.astype(jnp.float32)),
                vx=player2.vx.at[0].set(jnp.int32(0)),
                vy=player2.vy.at[0].set(jnp.int32(0)),
            )

            # Consume the entry door: deactivate and unlock it
            doors3 = doors2._replace(
                is_spawned=doors2.is_spawned.at[idx].set(False),
                is_locked_open=doors2.is_locked_open.at[idx].set(False),
            )

            return s._replace(player=player3, doors=doors3)

        return jax.lax.cond(has, do_tp, lambda s: s, state)

    @partial(jit, static_argnums=(0,))
    def _maybe_start_wave_pause(self, state: TronState) -> TronState:
        alive_now = jnp.sum(state.enemies.alive.astype(jnp.int32))

        # block the pause (and wave advance) if any enemy disc is still flying
        enemy_disc_active = jnp.any(
            (state.discs.owner == jnp.int32(1)) & (state.discs.phase == jnp.int32(1))
        )

        # clear the wave before respawn happens -> start pause + advance wave
        start_pause = (
            (alive_now == jnp.int32(0))
            & state.game_started
            & (state.wave_end_cooldown_remaining == jnp.int32(0))
            & (state.inwave_spawn_cd > jnp.int32(0))
            & (~enemy_disc_active)
        )

        def on(s: TronState) -> TronState:
            next_idx = jnp.minimum(
                s.wave_index + jnp.int32(1), jnp.int32(self.consts.num_waves - 1)
            )
            # +1 life, capped at max, but do not revive if already dead
            cur = s.player.lives[0]
            healed = jnp.where(
                cur > 0,
                jnp.minimum(cur + jnp.int32(1), jnp.int32(self.consts.player_lives)),
                cur,
            )
            player2 = s.player._replace(lives=s.player.lives.at[0].set(healed))
            return s._replace(
                wave_end_cooldown_remaining=jnp.int32(self.consts.wave_timeout),
                inwave_spawn_cd=jnp.int32(0),  # freeze in-wave respawns during pause
                wave_index=next_idx,  # advance color/points (saturates at GOLD)
                player=player2,
            )

        return jax.lax.cond(start_pause, on, lambda s: s, state)

    @partial(jit, static_argnums=(0,))
    def _on_player_hit(self, state: TronState, hits: Array) -> TronState:
        # No further processing if already gone
        def noop(s):
            return s

        def apply(s: TronState) -> TronState:
            cur = s.player.lives[0]
            # Cap damage to remaining lives; never below 0
            new_lives = jnp.maximum(cur - hits, jnp.int32(0))

            # Start death blink exactly on the transition (>0 -> 0)
            became_zero = (cur > 0) & (new_lives == 0)
            total_ticks = jnp.int32(
                self.consts.player_blink_cycles
                * 2
                * self.consts.player_blink_period_frames
            )
            new_ticks = jnp.where(
                became_zero, total_ticks, s.player_blink_ticks_remaining
            )

            player2 = s.player._replace(lives=s.player.lives.at[0].set(new_lives))
            return s._replace(player=player2, player_blink_ticks_remaining=new_ticks)

        return jax.lax.cond(state.player_gone, noop, apply, state)

    @partial(jit, static_argnums=(0,))
    def _enemy_disc_player_collisions(self, state: TronState) -> TronState:
        d = state.discs
        # If the player is gone, ignore further collisions

        # Enemy outbound discs only
        disc_mask = (d.owner == jnp.int32(1)) & (d.phase == jnp.int32(1))
        any_enemy_disc = jnp.any(disc_mask)

        def _no(s):
            return s

        def _check(s: TronState) -> TronState:
            d2 = s.discs

            # players recangle edges
            px0 = s.player.x[0]  # left
            py0 = s.player.y[0]  # top
            px1 = px0 + s.player.w[0]  # right
            py1 = py0 + s.player.h[0]  # bottom

            # Each discs rectangle edges
            dx0 = d2.x  # left
            dy0 = d2.y  # top
            dx1 = dx0 + d2.w  # right
            dy1 = dy0 + d2.h  # bottom

            # A disc hits the player if their horizontal spans overlap and their vertical
            # spans overlap
            hit = (
                disc_mask
                & (dx0 < px1)  # disc left is left of player right
                & (dx1 > px0)  # disc right is right of palyer left
                & (dy0 < py1)  # disc top is above player bottom
                & (dy1 > py0)  # disc bottom is below player rop
            )

            # total number of enemy discs that touched the player this frame
            n_hits = jnp.sum(hit.astype(jnp.int32))

            # For discs that hit: turn them off and clear their velocities/precise motion
            new_phase = jnp.where(hit, jnp.int32(0), d2.phase)
            new_vx = jnp.where(hit, jnp.int32(0), d2.vx)
            new_vy = jnp.where(hit, jnp.int32(0), d2.vy)
            new_fx = jnp.where(hit, jnp.float32(0), d2.fx)
            new_fy = jnp.where(hit, jnp.float32(0), d2.fy)
            new_fvx = jnp.where(hit, jnp.float32(0), d2.fvx)
            new_fvy = jnp.where(hit, jnp.float32(0), d2.fvy)
            s2 = s._replace(
                discs=d2._replace(
                    phase=new_phase,
                    vx=new_vx,
                    vy=new_vy,
                    fx=new_fx,
                    fy=new_fy,
                    fvx=new_fvx,
                    fvy=new_fvy,
                )
            )
            # Apply damage equal to how many enemy discs touched the player this frame
            return self._on_player_hit(s2, n_hits)

        # skip the check entirely if there are not active enemy discs
        return jax.lax.cond(any_enemy_disc, _check, _no, state)

    @partial(jit, static_argnums=(0,))
    def _spawn_enemy_discs(self, state: TronState) -> TronState:
        """
        One and only one enemy disc can exist at once.

        Behavior:
          - When an enemy disc disappears (wall/player hit), start a global cooldown in
            [enemy_firing_cooldown_range], and pick a random alive enemy as the next shooter.
          - While no disc is active: tick cooldown; when it hits zero, spawn exactly one disc
            from the chosen shooter (re-picking if they died).
          - While a disc is active: do nothing; no cooldown counts down.
        """
        c = self.consts
        e, d = state.enemies, state.discs

        # Is any enemy bullet currently flying?
        enemy_active_now = jnp.any(
            (d.owner == jnp.int32(1)) & (d.phase == jnp.int32(1))
        )

        # Edge-detect "disc just ended": last frame active -> now inactive
        just_ended = state.enemy_disc_active_prev & (~enemy_active_now)

        # Helper: start cooldown and pick a next shooter uniformly among alive enemies.
        def start_cd_and_pick_shooter(s: TronState) -> TronState:
            key = s.rng_key
            key, k_cd, k_pick = random.split(key, 3)

            # sample a new global cooldown (in frames)
            cd_min, cd_max = self.consts.enemy_firing_cooldown_range
            delay = random.randint(
                k_cd,
                (),
                minval=jnp.int32(cd_min),
                maxval=jnp.int32(cd_max + 1),
                dtype=jnp.int32,
            )

            # Get all enemies that are alive
            alive = s.enemies.alive

            # Build an array of indices for alive enemies. Length is fixed (max_enemies)
            alive_idx = jnp.nonzero(alive, size=self.consts.max_enemies)[0]

            # Count how many are actually alive right now
            alive_cnt = jnp.sum(alive.astype(jnp.int32))

            # Choose a random position in [0, alive_cnt]
            # If none are alive, just use 0 as a placeholder (no spawning)
            pos = jnp.where(
                alive_cnt > 0,
                random.randint(k_pick, (), 0, alive_cnt, dtype=jnp.int32),
                jnp.int32(0),
            )
            # Map the chosen position back to the actual enemy index
            next_shooter = alive_idx[pos]

            return s._replace(
                enemy_global_fire_cd=delay,
                enemy_next_shooter=next_shooter,
                rng_key=key,
            )

        # If a bullet just ended, start cooldown and choose next shooter.
        state = jax.lax.cond(just_ended, start_cd_and_pick_shooter, lambda s: s, state)

        state = jax.lax.cond(
            (~enemy_active_now) & (state.enemy_global_fire_cd > jnp.int32(0)),
            lambda s: s._replace(enemy_global_fire_cd=tick_cd(s.enemy_global_fire_cd)),
            lambda s: s,
            state,
        )

        # If no bullet active and cooldown == 0, spawn exactly one enemy disc
        def do_spawn(s: TronState) -> TronState:
            d, e = s.discs, s.enemies
            # Need a free disc slot.
            free_mask = d.phase == jnp.int32(0)
            has_free, slot = _find_first_true(free_mask)

            def spawn_into_free(ss: TronState) -> TronState:
                d, e = ss.discs, ss.enemies
                key = ss.rng_key

                # Use stored shooter if alive; otherwise pick a new alive shooter now.
                stored = ss.enemy_next_shooter
                alive = e.alive
                alive_any = jnp.any(alive)

                # fixed size list of indices for alive enemies
                alive_idx = jnp.nonzero(alive, size=self.consts.max_enemies)[0]

                # draw a random alive position; 0 if none are alive (wont be used)
                key, k_pick = random.split(key)
                alive_cnt = jnp.sum(alive.astype(jnp.int32))
                pos = jnp.where(
                    alive_cnt > 0,
                    random.randint(k_pick, (), 0, alive_cnt, dtype=jnp.int32),
                    jnp.int32(0),
                )
                picked = alive_idx[pos]

                # Use the stored shooter if still alive, otherwise the newly picked one
                shooter = jnp.where(alive[stored], stored, picked)

                def return_without_spawning(sin: TronState) -> TronState:
                    return sin._replace(rng_key=key)

                def really_spawn(sin: TronState) -> TronState:
                    d, e = sin.discs, sin.enemies

                    # Compute firing direction
                    # -----------------
                    # The enemy bullet should travel from the enemy's center towards the players center
                    # Both centers get computed, casted to float32 and then a unit direction vector is computed

                    # Player center (scalar each). Player arrays are shape (1,), so index [0]
                    # Keep integer centers for rendering/collision, but cast to float for direction match
                    pcx, pcy = rect_center(
                        sin.player.x[0],
                        sin.player.y[0],
                        sin.player.w[0],
                        sin.player.h[0],
                    )
                    pcx_f = pcx.astype(jnp.float32)
                    pcy_f = pcy.astype(jnp.float32)

                    # Enemy centers for all enemies (vectors). Select the chosen shooters center
                    ecx, ecy = rect_center(e.x, e.y, e.w, e.h)
                    ecx_f = ecx[shooter].astype(jnp.float32)
                    ecy_f = ecy[shooter].astype(jnp.float32)

                    # Vector from shooter to player (float). This is the direction prior normalization
                    dx = pcx_f - ecx_f
                    dy = pcy_f - ecy_f

                    # Length of the vector. If shooter sits exactly on the player, dist can be 0
                    dist = jnp.sqrt(dx * dx + dy * dy)

                    # Avoid div-by-zero by lower-bounding the denominator. this keeps ux/uy finite
                    invd = jnp.float32(1.0) / jnp.maximum(dist, jnp.float32(1e-6))

                    # Unit direction from shooter to player
                    ux = dx * invd
                    uy = dy * invd

                    # Scale the unit direction by the enemy bullet speed (float), to get the subpixel velocity
                    # Using floats enables precise motion accumulation. THen round for the raster
                    speed = jnp.float32(self.consts.enemy_disc_speed)
                    fvx = ux * speed
                    fvy = uy * speed

                    # Compute the discs initial position
                    # ---------------
                    # The discs should appear visually centered within the shooter box at spawn time
                    # That means: enemy top-left + half of the difference between enemy size and disc size

                    # Shooter top left
                    ex0 = e.x[shooter].astype(jnp.float32)
                    ey0 = e.y[shooter].astype(jnp.float32)
                    ew = e.w[shooter].astype(jnp.float32)
                    eh = e.h[shooter].astype(jnp.float32)

                    # Disc size for the target slot
                    dw = d.w[slot].astype(jnp.float32)
                    dh = d.h[slot].astype(jnp.float32)

                    # Center disc inside the shooter's rectangle
                    fx0 = ex0 + jnp.float32(0.5) * (ew - dw)
                    fy0 = ey0 + jnp.float32(0.5) * (eh - dh)

                    # Write the disc into the chosen free slot
                    # We keep both float and int positions:
                    #   - fx/fy: precise float state that advances by fvx, fvy each frame (subpixel motion)
                    #   - x/y: integer state for collision/rendering (rounded from fx/fy each frame)
                    #
                    # Enemy bullets use (fx, fy, fvx, fvy) for movement, the integer (vx, vy) is unised
                    # but kept at 0 to keep the structure uniform with player discs
                    d2 = d._replace(
                        # Integer top-left for raster/collision: round the float spawn position
                        x=d.x.at[slot].set(jnp.round(fx0).astype(jnp.int32)),
                        y=d.y.at[slot].set(jnp.round(fy0).astype(jnp.int32)),
                        # Integer velocities are not used for enemy bullets,
                        # so pin them to 0 and avoid mixing movement systems
                        vx=d.vx.at[slot].set(jnp.int32(0)),
                        vy=d.vy.at[slot].set(jnp.int32(0)),
                        owner=d.owner.at[slot].set(jnp.int32(1)),
                        phase=d.phase.at[slot].set(jnp.int32(1)),
                        fx=d.fx.at[slot].set(fx0),
                        fy=d.fy.at[slot].set(fy0),
                        fvx=d.fvx.at[slot].set(fvx),
                        fvy=d.fvy.at[slot].set(fvy),
                        shooter_id=d.shooter_id.at[slot].set(shooter),
                    )

                    return sin._replace(discs=d2, rng_key=key)

                # We only get here if there's a free disc slot (checked outside).
                # If at least one enemy is alive, spawn the disc; otherwise return the state unchanged
                return jax.lax.cond(
                    alive_any, really_spawn, return_without_spawning, ss
                )

            # Upstream gate: if there was no free disc slot, do nothing.
            # This preserves shapes and avoids partial writes when arrays are full.
            return jax.lax.cond(has_free, spawn_into_free, lambda ss: ss, s)

        state = jax.lax.cond(
            (~enemy_active_now) & (state.enemy_global_fire_cd == jnp.int32(0)),
            do_spawn,
            lambda s: s,
            state,
        )

        enemy_active_final = jnp.any(
            (state.discs.owner == jnp.int32(1)) & (state.discs.phase == jnp.int32(1))
        )
        return state._replace(enemy_disc_active_prev=enemy_active_final)

    @partial(jit, static_argnums=(0,))
    def step(
        self, state: TronState, action: Array
    ) -> Tuple[TronObservation, TronState, float, bool, TronInfo]:
        # Translate compact agent action index to ALE console action
        atari_action = jnp.take(self.ACTION_SET, action.astype(jnp.int32))
        
        previous_state = state
        user_action: UserAction = parse_action(atari_action)
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

        # Was there a player-owned active disc BEFORE changing s?
        # we cannot pass the same pressed_fire_change to both spawn and move disc
        # because it would immediately recall the just spawned disc
        had_player_disc_before = jnp.any(
            (state.discs.owner == jnp.int32(0)) & (state.discs.phase > jnp.int32(0))
        )

        # move the player
        state: TronState = self._player_step(state, user_action)
        # teleport player through doors
        state: TronState = self._teleport_player_if_door(state)
        # Spawn discs
        state: TronState = self._spawn_disc(state, pressed_fire_changed)

        # Only recall if a player-owned disc existed before this step
        recall_edge = pressed_fire_changed & had_player_disc_before
        state: TronState = self._lock_doors_from_disc_hits(state)
        state: TronState = self._move_discs(state, recall_edge)

        state = state._replace(fire_down_prev=user_action.fire)

        # tick door cooldowns so used doors eventually become available again
        state = state._replace(doors=_ArenaOps.tick_door_lockdown(state.doors))

        # on the first input movement, spawn up to max_enemies once
        def _spawn_initial_wave(s: TronState) -> TronState:
            # how many to reach the cap this frame
            alive_now = jnp.sum(s.enemies.alive.astype(jnp.int32))
            need = jnp.maximum(jnp.int32(self.consts.max_enemies) - alive_now, 0)
            s2 = self._spawn_enemies_up_to(s, need, jnp.float32(0.4))
            return s2._replace(
                game_started=jnp.array(True),
                inwave_spawn_cd=jnp.int32(self.consts.enemy_respawn_timer),
            )

        state = jax.lax.cond(
            ~state.game_started & user_action.moved,
            _spawn_initial_wave,
            lambda s: s,
            state,
        )

        def _pause_step(s: TronState) -> TronState:
            return s._replace(
                wave_end_cooldown_remaining=tick_cd(s.wave_end_cooldown_remaining)
            )

        def _wave_step(s: TronState) -> TronState:
            # If we just finished the pause and there are no enemies, spawn a fresh wave
            def spawn_new_enemies_if_needed(s0: TronState) -> TronState:
                alive_now = jnp.sum(s0.enemies.alive.astype(jnp.int32))

                # block respawns while any enemy disc is still active
                enemy_disc_active = jnp.any(
                    (s0.discs.owner == jnp.int32(1)) & (s0.discs.phase == jnp.int32(1))
                )

                # only spawn a new wave when:
                #   - the arena is empty
                #   - not in the between-wave pause (cd = 0)
                #   - the game has started
                #   - no enemy disc active anymore
                need_spawn = (
                    (alive_now == jnp.int32(0))
                    & (s0.wave_end_cooldown_remaining == jnp.int32(0))
                    & s0.game_started
                    & (~enemy_disc_active)
                )

                def do_spawn(ss: TronState) -> TronState:
                    s2 = self._spawn_enemies_up_to(
                        ss,
                        jnp.int32(self.consts.max_enemies),
                        jnp.float32(self.consts.create_new_door_prob),
                    )
                    return s2._replace(
                        inwave_spawn_cd=jnp.int32(self.consts.enemy_respawn_timer)
                    )

                return jax.lax.cond(need_spawn, do_spawn, lambda ss: ss, s0)

            # if the arena was cleared before the respawn, pause and skip respawns this frame
            s = self._maybe_start_wave_pause(s)
            s = spawn_new_enemies_if_needed(s)

            # Enemy discs that moved this frame can hit the player now
            s = self._enemy_disc_player_collisions(s)

            s = self._move_enemies(s)
            s = self._disc_enemy_collisions(s)
            s = self._update_respawn_cooldown_on_kills(s)

            # Tick enemy fire CDs and fire if ready (no-op during pause)
            s = jax.lax.cond(
                state.wave_end_cooldown_remaining > jnp.int32(0),
                lambda s: s,
                self._spawn_enemy_discs,
                s,
            )

            # Only allow in-wave respawns if we're not in a between-wave pause
            def do_respawn(ss: TronState) -> TronState:
                return self._maybe_respawn_enemy(ss)

            s = jax.lax.cond(
                s.wave_end_cooldown_remaining > jnp.int32(0),
                lambda ss: ss,  # in pause: no respawns
                do_respawn,
                s,
            )
            return s

        state = jax.lax.cond(
            state.wave_end_cooldown_remaining == 0, _wave_step, _pause_step, state
        )

        # Death blink countdown → disappear when finished
        def tick_blink(s: TronState) -> TronState:
            ticks = jnp.maximum(
                s.player_blink_ticks_remaining - jnp.int32(1), jnp.int32(0)
            )
            # When ticks hit zero and lives == 0, the player disappears permanently
            gone_now = s.player_gone | (
                (ticks == 0) & (s.player.lives[0] == jnp.int32(0))
            )
            return s._replace(player_blink_ticks_remaining=ticks, player_gone=gone_now)

        state = jax.lax.cond(
            state.player_blink_ticks_remaining > jnp.int32(0),
            tick_blink,
            lambda s: s,
            state,
        )

        obs: TronObservation = self._get_observation(state)
        env_reward = self._get_reward(previous_state, state)
        info: TronInfo = self._get_info(state)
        done: bool = self._get_done(state)

        return obs, state, env_reward, done, info

    def render(self, state: TronState) -> jnp.ndarray:
        return self.renderer.render(state)

    @partial(jit, static_argnums=(0,))
    def _get_observation(self, state: TronState) -> TronObservation:
        c = self.consts

        # Player
        px = jnp.atleast_1d(state.player.x[0]).astype(jnp.int32)
        py = jnp.atleast_1d(state.player.y[0]).astype(jnp.int32)
        pw = jnp.atleast_1d(state.player.w[0]).astype(jnp.int32)
        ph = jnp.atleast_1d(state.player.h[0]).astype(jnp.int32)
        player_entity = EntityPosition(x=px, y=py, width=pw, height=ph)

        # Enemies: mask inactive to x/y=-1 and w/h=0
        e = state.enemies
        alive_mask = e.alive
        ex = jnp.where(alive_mask, e.x, -jnp.ones_like(e.x)).astype(jnp.int32)
        ey = jnp.where(alive_mask, e.y, -jnp.ones_like(e.y)).astype(jnp.int32)
        ew = jnp.where(alive_mask, e.w, jnp.zeros_like(e.w)).astype(jnp.int32)
        eh = jnp.where(alive_mask, e.h, jnp.zeros_like(e.h)).astype(jnp.int32)
        enemies_entity = EntityPosition(x=ex, y=ey, width=ew, height=eh)

        # Discs: "active" = phase > 0
        d = state.discs
        disc_active = d.phase > jnp.int32(0)
        dx = jnp.where(disc_active, d.x, -jnp.ones_like(d.x)).astype(jnp.int32)
        dy = jnp.where(disc_active, d.y, -jnp.ones_like(d.y)).astype(jnp.int32)
        dw = jnp.where(disc_active, d.w, jnp.zeros_like(d.w)).astype(jnp.int32)
        dh = jnp.where(disc_active, d.h, jnp.zeros_like(d.h)).astype(jnp.int32)
        discs_entity = EntityPosition(x=dx, y=dy, width=dw, height=dh)

        # Doors
        doors = state.doors
        door_entity = EntityPosition(
            x=doors.x.astype(jnp.int32),
            y=doors.y.astype(jnp.int32),
            width=doors.w.astype(jnp.int32),
            height=doors.h.astype(jnp.int32),
        )

        return TronObservation(
            score=state.score.astype(jnp.int32),
            player=player_entity,
            player_lives=jnp.atleast_1d(state.player.lives[0]).astype(jnp.int32),
            player_gone=state.player_gone,
            enemies=enemies_entity,
            enemies_alive=alive_mask,
            discs=discs_entity,
            disc_owner=d.owner.astype(jnp.int32),
            disc_phase=d.phase.astype(jnp.int32),
            doors=door_entity,
            door_spawned=doors.is_spawned,
            door_locked=doors.is_locked_open,
            wave_index=state.wave_index.astype(jnp.int32),
        )

    def observation_space(self) -> spaces.Dict:
        c = self.consts
        # Shortcuts
        E = int(c.max_enemies)
        D = int(c.max_discs)
        DO = int(c.max_doors)

        # Generic helpers
        def entity_space(n: int, w_max: int, h_max: int) -> spaces.Dict:
            return spaces.Dict(
                {
                    "x": spaces.Box(
                        low=-w_max, high=c.screen_width, shape=(n,), dtype=jnp.int32
                    ),
                    "y": spaces.Box(
                        low=-h_max, high=c.screen_height, shape=(n,), dtype=jnp.int32
                    ),
                    "width": spaces.Box(low=0, high=w_max, shape=(n,), dtype=jnp.int32),
                    "height": spaces.Box(
                        low=0, high=h_max, shape=(n,), dtype=jnp.int32
                    ),
                }
            )

        # Maximum sizes for caps
        player_w_max = int(self.player_w)
        player_h_max = int(self.player_h)
        enemy_w_max = int(self.enemy_w)  # enemies use player-sized boxes for now
        enemy_h_max = int(self.enemy_h)
        out_w, out_h = c.disc_size_out
        ret_w, ret_h = c.disc_size_ret
        disc_w_max = int(max(out_w, ret_w))
        disc_h_max = int(max(out_h, ret_h))

        door_w_max = int(c.door_w)
        door_h_max = int(c.door_h)

        return spaces.Dict(
            {
                # Scalar
                "score": spaces.Box(
                    low=0, high=(10**6) - 1, shape=(), dtype=jnp.int32
                ),  # highest displayable number in game
                "wave_index": spaces.Box(
                    low=0, high=max(0, c.num_waves - 1), shape=(), dtype=jnp.int32
                ),
                # Player
                "player": entity_space(1, player_w_max, player_h_max),
                "player_lives": spaces.Box(
                    low=0, high=c.player_lives, shape=(1,), dtype=jnp.int32
                ),
                "player_gone": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
                # Enemies
                "enemies": entity_space(E, enemy_w_max, enemy_h_max),
                "enemies_alive": spaces.Box(low=0, high=1, shape=(E,), dtype=jnp.int32),
                # Discs
                "discs": entity_space(D, disc_w_max, disc_h_max),
                "disc_owner": spaces.Box(low=0, high=1, shape=(D,), dtype=jnp.int32),
                "disc_phase": spaces.Box(low=0, high=2, shape=(D,), dtype=jnp.int32),
                # Doors
                "doors": entity_space(DO, door_w_max, door_h_max),
                "door_spawned": spaces.Box(low=0, high=1, shape=(DO,), dtype=jnp.int32),
                "door_locked": spaces.Box(low=0, high=1, shape=(DO,), dtype=jnp.int32),
            }
        )

    @partial(jit, static_argnums=(0,))
    def _get_reward(self, previous_state: TronState, state: TronState) -> Array:
        return state.score - previous_state.score  # .astype(jnp.float32)

    @partial(jit, static_argnums=(0,))
    def _get_done(self, state: TronState) -> bool:
        return state.player_gone

    @partial(jit, static_argnums=(0,))
    def _get_info(self, state: TronState) -> TronInfo:
        # Counts / booleans
        enemies_alive_count = jnp.sum(state.enemies.alive.astype(jnp.int32))
        discs_active_count = jnp.sum(
            (state.discs.phase > jnp.int32(0)).astype(jnp.int32)
        )
        enemy_disc_active = jnp.any(
            (state.discs.owner == jnp.int32(1)) & (state.discs.phase == jnp.int32(1))
        )

        return TronInfo(
            score=state.score.astype(jnp.int32),
            wave_index=state.wave_index.astype(jnp.int32),
            enemies_alive_count=enemies_alive_count.astype(jnp.int32),
            discs_active_count=discs_active_count.astype(jnp.int32),
            enemy_disc_active=enemy_disc_active,
            player_lives=jnp.atleast_1d(state.player.lives[0]).astype(jnp.int32),
            player_gone=state.player_gone,
            player_blink_ticks_remaining=state.player_blink_ticks_remaining.astype(
                jnp.int32
            ),
            wave_end_cooldown_remaining=state.wave_end_cooldown_remaining.astype(
                jnp.int32
            ),
            inwave_spawn_cd=state.inwave_spawn_cd.astype(jnp.int32),
            enemy_global_fire_cd=state.enemy_global_fire_cd.astype(jnp.int32),
            game_started=state.game_started,
            frame_idx=state.frame_idx.astype(jnp.int32),
        )

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTION_SET))

    def image_space(self) -> spaces.Box:
        c = self.consts
        return spaces.Box(
            low=0,
            high=255,
            shape=(c.screen_height, c.screen_width, 3),
            dtype=jnp.uint8,
        )

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: EnvObs) -> Array:
        def _flat_entity(ep) -> jnp.ndarray:
            return jnp.concatenate(
                [
                    jnp.ravel(ep.x).astype(jnp.int32),
                    jnp.ravel(ep.y).astype(jnp.int32),
                    jnp.ravel(ep.width).astype(jnp.int32),
                    jnp.ravel(ep.height).astype(jnp.int32),
                ],
                axis=0,
            )

        return jnp.concatenate(
            [
                jnp.atleast_1d(obs.score).astype(jnp.int32),
                jnp.atleast_1d(obs.wave_index).astype(jnp.int32),
                _flat_entity(obs.player),
                jnp.ravel(obs.player_lives).astype(jnp.int32),
                jnp.atleast_1d(obs.player_gone.astype(jnp.int32)),
                _flat_entity(obs.enemies),
                jnp.ravel(obs.enemies_alive).astype(jnp.int32),
                _flat_entity(obs.discs),
                jnp.ravel(obs.disc_owner).astype(jnp.int32),
                jnp.ravel(obs.disc_phase).astype(jnp.int32),
                _flat_entity(obs.doors),
                jnp.ravel(obs.door_spawned).astype(jnp.int32),
                jnp.ravel(obs.door_locked).astype(jnp.int32),
            ],
            axis=0,
        )
