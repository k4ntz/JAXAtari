from functools import partial
from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
import chex
import os
from flax import struct
import jaxatari.spaces as spaces
from jaxatari.environment import JaxEnvironment, ObjectObservation, JAXAtariAction as Action
from jaxatari.renderers import JAXGameRenderer
import jaxatari.rendering.jax_rendering_utils as render_utils
from jaxatari.modification import AutoDerivedConstants

KLAX_MOD_SPRITE_DIR = os.path.join(os.path.dirname(__file__), "mods", "klax", "sprites")

class KlaxConstants(AutoDerivedConstants):
    SCREEN_WIDTH: int = struct.field(pytree_node=False, default=160)
    SCREEN_HEIGHT: int = struct.field(pytree_node=False, default=250)
    # Original background.npy is 210px. Slightly compress it so the well sits
    # closer to ALE (well ~y=40–176), then pad the rest of the 250 canvas black.
    CONTENT_HEIGHT: int = struct.field(pytree_node=False, default=204)

    # Tiles
    N_TILE_TYPES: int = struct.field(pytree_node=False, default=7)
    TILE_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default=(7, 4))
    
    # Optimized MAX_TILES
    MAX_TILES: int = struct.field(pytree_node=False, default=15)
    
    # Timings converted to frames (assuming 60 FPS as this was the default value before. If its too slow, here is the culprit)
    # 6.0s * 60 = 360 frames between tiles; first tile of a wave is sooner.
    SPAWN_INTERVAL_STEPS: int = struct.field(pytree_node=False, default=360)
    SPAWN_FIRST_DELAY_STEPS: int = struct.field(pytree_node=False, default=60)
    # 9.0s * 60 = 540 frames
    FALL_DURATION_STEPS: int = struct.field(pytree_node=False, default=540)
    
    SPEED_FACTOR: int = struct.field(pytree_node=False, default=10)

    SPAWN_START_Y: int = struct.field(pytree_node=False, default=43)
    SHOOT_UP_Y: int = struct.field(pytree_node=False, default=60)
    DESPAWN_Y: int = struct.field(pytree_node=False, default=112)

    BOARD_ROWS: int = struct.field(pytree_node=False, default=5)
    BOARD_COLS: int = struct.field(pytree_node=False, default=5)
    BOARD_BOTTOM_Y: int = struct.field(pytree_node=False, default=174)
    BOARD_GAP: int = struct.field(pytree_node=False, default=1)
    COLUMN_START_X: int = struct.field(pytree_node=False, default=60)
    COLUMN_STEP_X: int = struct.field(pytree_node=False, default=8)

    # Player
    PLAYER_WIDTH: int = struct.field(pytree_node=False, default=7)
    PLAYER_HEIGHT: int = struct.field(pytree_node=False, default=4)
    PLAYER_Y: int = struct.field(pytree_node=False, default=None)
    # ALE lane marker is 5×3 inside the slot (JAX slot interiors are 5px at y=176–178).
    PADDLE_Y: int = struct.field(pytree_node=False, default=176)
    PADDLE_WIDTH: int = struct.field(pytree_node=False, default=5)
    PADDLE_HEIGHT: int = struct.field(pytree_node=False, default=3)
    PADDLE_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default_factory=lambda: (210, 210, 210))
    # Falling-tile perspective: (height, width), height grows first, then width.
    # Last entry is the catcher/board size (TILE_SIZE is (width, height) = (7, 4)).
    TILE_SCALE_SIZES: tuple = struct.field(
        pytree_node=False,
        default_factory=lambda: (
            (1, 4),
            (2, 4),
            (3, 4),
            (3, 5),
            (4, 5),
            (4, 6),
            (4, 7),
        ),
    )
    N_TILE_SCALES: int = struct.field(pytree_node=False, default=7)
    HUD_SCORE_Y: int = struct.field(pytree_node=False, default=18)
    HUD_LIVES_Y: int = struct.field(pytree_node=False, default=31)
    HUD_TASK_Y: int = struct.field(pytree_node=False, default=181)
    HUD_TOGO_Y: int = struct.field(pytree_node=False, default=197)
    # Wave-intro overlay (score_digits on top of wave_intro.npy).
    WAVE_GREEN: Tuple[int, int, int] = struct.field(pytree_node=False, default_factory=lambda: (160, 194, 92))
    WAVE_NUM_X: int = struct.field(pytree_node=False, default=70)
    WAVE_NUM_Y: int = struct.field(pytree_node=False, default=114)
    WAVE_NUM_SPACING: int = struct.field(pytree_node=False, default=9)
    WAVE_TARGET_Y: int = struct.field(pytree_node=False, default=152)
    WAVE_TARGET_X_RIGHT: int = struct.field(pytree_node=False, default=93)
    RESPONSIVENESS: int = struct.field(pytree_node=False, default=1)
    PLAYER_BACKPACK_MAX: int = struct.field(pytree_node=False, default=5)

    # Waves
    # ~4.5s * 60 = 270 frames; FIRE skips the splash immediately.
    WAVES_COOLDOWN_STEPS: int = struct.field(pytree_node=False, default=270)

    ASSET_CONFIG: tuple = struct.field(pytree_node=False, default_factory=lambda: (
        {'name': 'background', 'type': 'background', 'file': 'background.npy'},
        {'name': 'player', 'type': 'single', 'file': 'player.npy'},
        {'name': 'tile', 'type': 'group', 'files': [
            'tile_blue.npy', 'tile_purple.npy', 'tile_yellow.npy', 'tile_white.npy',
            'tile_green.npy', 'tile_pink.npy', 'tile_red.npy',
        ]},
        {'name': 'score_digits', 'type': 'digits', 'pattern': '{}_score.npy'},
        {'name': 'togo_digits', 'type': 'digits', 'pattern': '{}_to_go.npy'},
        {'name': 'lives_remaining', 'type': 'group', 'files': [
            '0_lives_remaining.npy', '1_lives_remaining.npy', '2_lives_remaining.npy', '3_lives_remaining.npy',
        ]},
        {'name': 'task_labels', 'type': 'group', 'files': [
            'klaxs_to_go.npy', 'diagonals_to_go.npy', 'tiles_to_go.npy', 'points_to_go.npy', 'horizontal_to_go.npy',
        ]},
        # Wave-intro pieces live in games/mods/klax/sprites (repo-shipped, not the sprite pack).
        {'name': 'wave_intro_ring', 'type': 'single', 'file': 'wave_intro_ring.npy'},
        {'name': 'wave_intro_spiral', 'type': 'single', 'file': 'wave_intro_spiral.npy'},
        {'name': 'wave_intro_title', 'type': 'single', 'file': 'wave_intro_title.npy'},
        {'name': 'wave_intro_you_must', 'type': 'single', 'file': 'wave_intro_you_must.npy'},
        {'name': 'wave_intro_get', 'type': 'single', 'file': 'wave_intro_get.npy'},
        {'name': 'wave_intro_klaxs', 'type': 'single', 'file': 'wave_intro_klaxs.npy'},
        {'name': 'wave_intro_diagonals', 'type': 'single', 'file': 'wave_intro_diagonals.npy'},
        {'name': 'wave_intro_tiles', 'type': 'single', 'file': 'wave_intro_tiles.npy'},
        {'name': 'wave_intro_points', 'type': 'single', 'file': 'wave_intro_points.npy'},
        {'name': 'wave_intro_horizontal', 'type': 'single', 'file': 'wave_intro_horizontal.npy'},
    ))
    # (name, x, y, flip_horizontal) — right spiral is the left sprite mirrored.
    WAVE_INTRO_BLITS: tuple = struct.field(pytree_node=False, default_factory=lambda: (
        ("wave_intro_ring", 94, 17, False),
        ("wave_intro_spiral", 38, 48, False),
        ("wave_intro_spiral", 102, 48, True),
        ("wave_intro_title", 62, 86, False),
        ("wave_intro_you_must", 62, 128, False),
        ("wave_intro_get", 73, 139, False),
    ))
    # Bottom objective word; index = wave task_id. Centered at y=166.
    WAVE_INTRO_TASK_BLITS: tuple = struct.field(pytree_node=False, default_factory=lambda: (
        ("wave_intro_klaxs", 70, 166),
        ("wave_intro_diagonals", 64, 166),
        ("wave_intro_tiles", 70, 166),
        ("wave_intro_points", 68, 166),
        ("wave_intro_horizontal", 64, 166),
    ))

    # Task for each wave: [task_id, amount]
    klax_waves: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([
        [0, 3],  # 1
        [0, 5],  # 2
        [1, 3],  # 3
        [3, 10000],  # 4
        [2, 40],  # 5

        [0, 10],  # 6
        [1, 6],  # 7
        [2, 55],  # 8
        [3, 25000],  # 9
        [4, 5],  # 10

        [0, 15],  # 11
        [1, 13],  # 12
        [2, 75],  # 13
        [3, 25000],  # 14
        [4, 10],  # 15

        [0, 15],  # 16
        [1, 13],  # 17
        [2, 75],  # 18
        [3, 30000],  # 19
        [4, 13],  # 20

        [0, 20],  # 21
        [1, 13],  # 22
        [2, 65],  # 23
        [3, 35000],  # 24
        [4, 13],  # 25

        [0, 30],  # 26
        [1, 13],  # 27
        [2, 80],  # 28
        [3, 35000],  # 29
        [4, 15],  # 30

        [0, 30],  # 31
        [1, 13],  # 32
        [2, 80],  # 33
        [3, 40000],  # 34
        [4, 5],  # 35

        [0, 30],  # 36
        [1, 13],  # 37
        [2, 90],  # 38
        [3, 40000],  # 39
        [4, 15],  # 40

        [0, 35],  # 41
        [1, 13],  # 42
        [2, 90],  # 43
        [3, 50000],  # 44
        [4, 15],  # 45

        [0, 35],  # 46
        [1, 13],  # 47
        [2, 90],  # 48
        [3, 55000],  # 49
        [4, 18],  # 50

        [0, 30],  # 51
        [1, 13],  # 52
        [2, 90],  # 53
        [3, 60000],  # 54
        [4, 18],  # 55

        [0, 30],  # 56
        [1, 5],  # 57
        [2, 90],  # 58
        [3, 60000],  # 59
        [4, 18],  # 60

        [0, 30],  # 61
        [1, 15],  # 62
        [2, 90],  # 63
        [3, 70000],  # 64
        [4, 13],  # 65

        [0, 30],  # 66
        [1, 15],  # 67
        [2, 100],  # 68
        [3, 70000],  # 69
        [4, 15],  # 70

        [0, 35],  # 71
        [1, 13],  # 72
        [2, 100],  # 73
        [3, 75000],  # 74
        [4, 15],  # 75

        [0, 22],  # 76
        [1, 6],  # 77
        [2, 100],  # 78
        [3, 80000],  # 79
        [4, 20],  # 80

        [0, 40],  # 81
        [1, 20],  # 82
        [2, 100],  # 83
        [3, 100000],  # 84
        [4, 5],  # 85

        [0, 30],  # 86
        [1, 20],  # 87
        [2, 100],  # 88
        [3, 100000],  # 89
        [4, 20],  # 90

        [0, 25],  # 91
        [1, 13],  # 92
        [2, 100],  # 93
        [3, 70000],  # 94
        [4, 15],  # 95

        [0, 30],  # 96
        [1, 6],  # 97
        [2, 50],  # 98
        [4, 6],  # 99  (was a duplicate tile wave; this slot is horizontal in the 5-wave cycle)
        [3, 250000],  # 100
    ]))
    # ID-Task-Mapping
    # 0 = KLAX → X Klaxes (any)
    # 1 = DIAGONAL → X diagonal Klaxes
    # 2 = TILE → survive N tiles that land on the paddle or drop
    # 3 = POINTS → Score X points
    # 4 = HORIZONTAL → X horizontal Klaxes

    def compute_derived(self):
        return {
            "PLAYER_Y": self.DESPAWN_Y + 6
        }

@struct.dataclass
class TilesObservation:
    x: chex.Array
    y: chex.Array
    color: chex.Array
    active: chex.Array


@struct.dataclass
class KlaxObservation:
    player: ObjectObservation
    tiles: ObjectObservation
    backpack_items: ObjectObservation
    board: chex.Array
    score: chex.Array
    lives: chex.Array
    wave_task: chex.Array


@struct.dataclass
class KlaxInfo:
    time: chex.Array


@struct.dataclass
class KlaxState:
    player_x: chex.Array
    player_col: chex.Array
    player_target_col: chex.Array

    player_backpack_colors: chex.Array
    player_backpack_count: chex.Array

    tiles_x: chex.Array  # (MAX_TILES,) x-coordinates
    tiles_y: chex.Array  # (MAX_TILES,) y-coordinates
    tiles_color: chex.Array  # (MAX_TILES,) color indices [0..len(TILE_COLORS)-1]
    tiles_active: chex.Array  # (MAX_TILES,) 0/1 active flags
    tiles_col: chex.Array
    tiles_spawn_step: chex.Array

    board: chex.Array
    fire_lock: chex.Array # int32 (0/1)
    up_lock: chex.Array # int32 (0/1)

    score: chex.Array
    lives: chex.Array
    step_counter: chex.Array
    rng_key: chex.PRNGKey

    # --- Waves ---
    wave_idx: chex.Array
    wave_task_id: chex.Array
    wave_target: chex.Array
    wave_progress: chex.Array
    wave_score_base: chex.Array
    wave_active: chex.Array
    wave_cooldown_until: chex.Array

    # progresses; needed for action down
    tiles_progress_accum: chex.Array
    spawn_progress_accum: chex.Array


class JaxKlax(JaxEnvironment[KlaxState, KlaxObservation, KlaxInfo, KlaxConstants]):
    # --- Orientation codes ---
    ORIENT_H: int = 0     # horizontal
    ORIENT_V: int = 1     # vertical
    ORIENT_D1: int = 2    # diagonal down-right
    ORIENT_D2: int = 3    # diagonal up-right

    # Minimal ALE action set for Klax
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

    def __init__(self, consts: KlaxConstants = None):
        consts = consts or KlaxConstants()
        super().__init__(consts)
        self.renderer = KlaxRenderer(self.consts)

        # --- WAVES ---
        self._waves = jnp.array(self.consts.klax_waves, dtype=jnp.int32)
        self._n_waves = int(self._waves.shape[0])

        # Default: no reward functions (handled by wrapper)
        rf = tuple()
        self._rf_count: int = 0
        self._rf_eff_len: int = 1

        rf_branches = tuple()
        fallback_branch = (lambda op: op[2].astype(jnp.float32),)

        self._rf_branches = rf_branches + fallback_branch

        self._kernels = {
            3: (jnp.ones((1, 3), dtype=jnp.int32), jnp.ones((3, 1), dtype=jnp.int32),
                jnp.eye(3, dtype=jnp.int32), jnp.fliplr(jnp.eye(3, dtype=jnp.int32))),
            4: (jnp.ones((1, 4), dtype=jnp.int32), jnp.ones((4, 1), dtype=jnp.int32),
                jnp.eye(4, dtype=jnp.int32), jnp.fliplr(jnp.eye(4, dtype=jnp.int32))),
            5: (jnp.ones((1, 5), dtype=jnp.int32), jnp.ones((5, 1), dtype=jnp.int32),
                jnp.eye(5, dtype=jnp.int32), jnp.fliplr(jnp.eye(5, dtype=jnp.int32))),
        }

    @partial(jax.jit, static_argnums=(0,))
    def _conv2d_valid(self, x_2d, k_2d):
        x4 = x_2d[None, None, ...].astype(jnp.float32)
        k4 = k_2d[None, None, ...].astype(jnp.float32)
        y4 = jax.lax.conv_general_dilated(
            x4, k4, (1, 1), 'VALID',
            dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
        )
        return y4[0, 0].astype(jnp.int32)

    @partial(jax.jit, static_argnums=(0,))
    def _conv2d_full(self, x_valid, k_2d):
        Kh, Kw = int(k_2d.shape[0]), int(k_2d.shape[1])
        x4 = x_valid[None, None, ...].astype(jnp.float32)
        k4 = k_2d[None, None, ...].astype(jnp.float32)
        pad = ((Kh - 1, Kh - 1), (Kw - 1, Kw - 1))
        y4 = jax.lax.conv_general_dilated(
            x4, k4, (1, 1), pad,
            dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
        )
        return y4[0, 0].astype(jnp.int32)

    @partial(jax.jit, static_argnums=(0, 2))
    def _match_one_orientation_conv(
            self,
            board: chex.Array,
            orientation_code: int,
            points_tuple: tuple[int, int, int],  # (p3, p4, p5)
    ) -> tuple[chex.Array, chex.Array, chex.Array]:
        R, C = self.consts.BOARD_ROWS, self.consts.BOARD_COLS
        p3, p4, p5 = (jnp.int32(points_tuple[0]),
                      jnp.int32(points_tuple[1]),
                      jnp.int32(points_tuple[2]))

        remove_mask = jnp.zeros((R, C), dtype=jnp.bool_)
        score_add = jnp.int32(0)
        klax_add = jnp.int32(0)

        def run_length(L, pts, remove_mask, score_add, klax_add):
            Kh, Kv, Kd1, Kd2 = self._kernels[L]
            if orientation_code == self.ORIENT_H:
                K = Kh
            elif orientation_code == self.ORIENT_V:
                K = Kv
            elif orientation_code == self.ORIENT_D1:
                K = Kd1
            else:
                K = Kd2

            def per_color(c_idx):
                color_val = jnp.int32(c_idx + 1)
                color_mask = (board == color_val) & (~remove_mask)
                wins = (self._conv2d_valid(color_mask, K) == L)
                covered = self._conv2d_full(wins.astype(jnp.int32), K) > 0
                return wins, covered

            wins_c, covered_c = jax.vmap(per_color)(jnp.arange(self.consts.N_TILE_TYPES))
            wins_any = jnp.any(wins_c, axis=0)
            covered_any = jnp.any(covered_c, axis=0)

            k_len = wins_any.sum(dtype=jnp.int32)
            score_add = score_add + k_len * pts
            # 3-tile = 1 Klax, 4-tile = 2, 5-tile = 3 (2600 manual).
            klax_add = klax_add + k_len * jnp.int32(L - 2)
            remove_mask = remove_mask | covered_any
            return remove_mask, score_add, klax_add

        remove_mask, score_add, klax_add = run_length(5, p5, remove_mask, score_add, klax_add)
        remove_mask, score_add, klax_add = run_length(4, p4, remove_mask, score_add, klax_add)
        remove_mask, score_add, klax_add = run_length(3, p3, remove_mask, score_add, klax_add)

        return remove_mask, score_add, klax_add

    @partial(jax.jit, static_argnums=(0,))
    def _apply_matches_and_gravity(self, board: chex.Array):
        """
        Computes all orientations, removes matches, applies gravity, and
        repeats so gravity chains can score (bounded; 5x5 well).

        Returns:
          board_after (int32 [R,C])
          score_add (int32)
          total_klax_add (int32)
          horiz_klax_add (int32)
          diag_klax_add (int32)
        """
        def cascade_step(carry):
            board_in, score_add, total_klax_add, horiz_klax_add, diag_klax_add = carry
            rm_v, sc_v, k_v    = self._match_one_orientation_conv(board_in, self.ORIENT_V,  (50, 1000, 1500))
            rm_h, sc_h, k_h    = self._match_one_orientation_conv(board_in, self.ORIENT_H,  (100, 500, 1000))
            rm_d1, sc_d1, k_d1 = self._match_one_orientation_conv(board_in, self.ORIENT_D1, (500, 1000, 1500))
            rm_d2, sc_d2, k_d2 = self._match_one_orientation_conv(board_in, self.ORIENT_D2, (500, 1000, 1500))
            remove_mask = rm_v | rm_h | rm_d1 | rm_d2
            board_cleared = jnp.where(remove_mask, jnp.int32(0), board_in)
            board_after = self._compact_board_columns(board_cleared)
            return (
                board_after,
                score_add + sc_v + sc_h + sc_d1 + sc_d2,
                total_klax_add + k_v + k_h + k_d1 + k_d2,
                horiz_klax_add + k_h,
                diag_klax_add + k_d1 + k_d2,
            )

        zeros = (board, jnp.int32(0), jnp.int32(0), jnp.int32(0), jnp.int32(0))
        return jax.lax.fori_loop(0, 8, lambda _i, carry: cascade_step(carry), zeros)

    @partial(jax.jit, static_argnums=(0,))
    def _compact_board_columns(self, board: chex.Array) -> chex.Array:
        """
        Stable 'gravity' packing towards row 0 (bottom),
        preserving the original bottom-to-top order.
        Vectorized across columns with vmap; tiny fori over rows inside.
        """
        R = self.consts.BOARD_ROWS

        def pack_col(col: chex.Array) -> chex.Array:
            def body(r, carry):
                out, w = carry
                val = col[r]
                is_tile = (val > 0)
                out = jax.lax.cond(is_tile,
                                   lambda a: a.at[w].set(val),
                                   lambda a: a,
                                   out)
                w = jax.lax.select(is_tile, w + 1, w)
                return (out, w)

            out0 = jnp.zeros_like(col), jnp.int32(0)
            newcol, _ = jax.lax.fori_loop(0, R, body, out0)
            return newcol

        return jax.vmap(pack_col, in_axes=1, out_axes=1)(board)

    def reset(self, key: jax.random.PRNGKey = jax.random.PRNGKey(42)) -> tuple[KlaxObservation, KlaxState]:
        _, key = jax.random.split(key)
        board = jnp.zeros((self.consts.BOARD_ROWS, self.consts.BOARD_COLS), dtype=jnp.int32)

        start_col = jnp.int32(0)
        px0 = jnp.int32(self.consts.COLUMN_START_X + start_col * self.consts.COLUMN_STEP_X)

        state = KlaxState(
            player_x=px0,
            player_col=start_col,
            player_target_col=start_col,

            tiles_x=jnp.zeros((self.consts.MAX_TILES,), dtype=jnp.int32),
            tiles_y=jnp.zeros((self.consts.MAX_TILES,), dtype=jnp.int32),
            tiles_color=jnp.zeros((self.consts.MAX_TILES,), dtype=jnp.int32),
            tiles_active=jnp.zeros((self.consts.MAX_TILES,), dtype=jnp.int32),
            tiles_col=jnp.zeros((self.consts.MAX_TILES,), dtype=jnp.int32),
            tiles_spawn_step=jnp.zeros((self.consts.MAX_TILES,), dtype=jnp.int32),

            player_backpack_colors=jnp.zeros((self.consts.PLAYER_BACKPACK_MAX,), dtype=jnp.int32),
            player_backpack_count=jnp.array(0, dtype=jnp.int32),

            fire_lock=jnp.int32(0),
            up_lock=jnp.int32(0),
            board=board,

            score=jnp.array(0, dtype=jnp.int32),
            lives=jnp.array(3, dtype=jnp.int32),
            step_counter=jnp.array(0, dtype=jnp.int32),

            rng_key=key,

            wave_idx=jnp.int32(0),
            wave_task_id=self._waves[0, 0] if self._n_waves > 0 else jnp.int32(-1),
            wave_target=self._waves[0, 1] if self._n_waves > 0 else jnp.int32(0),
            wave_progress=jnp.int32(0),
            wave_score_base=jnp.int32(0),
            wave_active=jnp.int32(0),
            wave_cooldown_until=jnp.int32(self.consts.WAVES_COOLDOWN_STEPS if self._n_waves > 0 else 0),

            tiles_progress_accum=jnp.zeros((self.consts.MAX_TILES,), dtype=jnp.int32),
            spawn_progress_accum=jnp.int32(0),
        )
        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: KlaxState, action: chex.Array):
        # Translate agent action index to ALE console action
        atari_action = jnp.take(self.ACTION_SET, jnp.asarray(action, dtype=jnp.int32))
        new_state = self._advance_state(state, atari_action)
        base_reward = self._get_reward(state, new_state).astype(jnp.float32)
        operand = (state, new_state, base_reward)

        def body(i, arr):
            val = jax.lax.switch(i, self._rf_branches, operand)
            return arr.at[i].set(val)

        done = self._get_done(new_state)
        obs = self._get_observation(new_state)
        info = self._get_info(new_state)
        return obs, new_state, base_reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _advance_state(self, state: KlaxState, action: chex.Array) -> KlaxState:
        def press_once(pressed_i32, lock_i32):
            can_do = (pressed_i32 == 1) & (lock_i32 == 0)
            next_lock = pressed_i32
            return can_do, next_lock

        step_counter = state.step_counter + 1

        # --- per-wave speed multiplier ---
        s_num = jnp.int32(10) + state.wave_idx
        s_den = jnp.int32(10)

        # --- speed-up when down-key is pressed ---
        speed_pressed = (
                (action == Action.DOWN) |
                (action == Action.DOWNFIRE) |
                (action == Action.DOWNRIGHT) |
                (action == Action.DOWNRIGHTFIRE) |
                (action == Action.DOWNLEFT) |
                (action == Action.DOWNLEFTFIRE)
        )
        speed_mul = jnp.where(speed_pressed,
                              jnp.int32(self.consts.SPEED_FACTOR),
                              jnp.int32(1))

        k_spawn_col, k_spawn_color, k_after = jax.random.split(state.rng_key, 3)

        # ---- tiles fall ----
        tiles_x = state.tiles_x
        tiles_color = state.tiles_color
        tiles_active = state.tiles_active
        tiles_col = state.tiles_col
        tiles_spawn_step = state.tiles_spawn_step
        tiles_progress_accum = state.tiles_progress_accum

        dist = jnp.int32(self.consts.DESPAWN_Y - self.consts.SPAWN_START_Y)
        # CHANGED: Use pre-calculated steps
        den = jnp.int32(self.consts.FALL_DURATION_STEPS)

        # add "progress units" proportional to speed; convert to whole pixels
        add_units = speed_mul * dist * s_num
        den_scaled = den * s_den
        accum_new = tiles_progress_accum + add_units
        delta_px = accum_new // den_scaled
        accum_new = accum_new % den_scaled

        y_after = jnp.minimum(jnp.int32(self.consts.DESPAWN_Y), state.tiles_y + delta_px)

        # only update active tiles
        tiles_y = jnp.where(tiles_active == 1, y_after, state.tiles_y)
        tiles_progress_accum = jnp.where(tiles_active == 1, accum_new, tiles_progress_accum)

        # ---- capture / vanish at DESPAWN_Y ----
        reached = (tiles_y >= jnp.int32(self.consts.DESPAWN_Y)) & (tiles_active == 1)

        def capture_body(i, carry):
            (bp_colors, bp_count, t_active, t_y, t_spawn, t_accum, miss_count, cap_count, cap_score) = carry
            match = reached[i] & (tiles_col[i] == state.player_col) & (
                    bp_count < jnp.int32(self.consts.PLAYER_BACKPACK_MAX)
            )
            missed = reached[i] & (~match)
            miss_count = miss_count + missed.astype(jnp.int32)
            cap_count = cap_count + match.astype(jnp.int32)

            cap_score = cap_score + match.astype(jnp.int32) * jnp.int32(5)

            bp_colors = jax.lax.cond(
                match,
                lambda arr: arr.at[bp_count].set(tiles_color[i]),
                lambda arr: arr,
                bp_colors,
            )
            bp_count = jax.lax.select(match, bp_count + 1, bp_count)

            t_active = jax.lax.cond(reached[i], lambda a: a.at[i].set(jnp.int32(0)), lambda a: a, t_active)
            t_y = jax.lax.cond(reached[i], lambda y: y.at[i].set(jnp.int32(0)), lambda y: y, t_y)
            t_spawn = jax.lax.cond(reached[i], lambda s: s.at[i].set(jnp.int32(0)), lambda s: s, t_spawn)
            t_accum = jax.lax.cond(reached[i], lambda a: a.at[i].set(jnp.int32(0)), lambda a: a, t_accum)

            return (bp_colors, bp_count, t_active, t_y, t_spawn, t_accum, miss_count, cap_count, cap_score)

        bp_init = (
            state.player_backpack_colors,
            state.player_backpack_count,
            tiles_active,
            tiles_y,
            tiles_spawn_step,
            tiles_progress_accum,
            jnp.int32(0),
            jnp.int32(0),
            jnp.int32(0),
        )

        (player_backpack_colors, player_backpack_count, tiles_active, tiles_y, tiles_spawn_step, tiles_progress_accum,
         miss_count, cap_count, cap_score_add) = jax.lax.fori_loop(0, self.consts.MAX_TILES, capture_body, bp_init)

        new_lives = jnp.maximum(jnp.int32(0), state.lives - miss_count)  # lives update

        # ---- spawn ----
        # CHANGED: Use pre-calculated steps
        spawn_den = jnp.int32(self.consts.SPAWN_INTERVAL_STEPS)
        spawn_accum_prev = state.spawn_progress_accum

        spawn_accum_next = spawn_accum_prev + speed_mul * s_num

        # propose a spawn whenever the accumulator crosses the threshold
        spawn_den_scaled = spawn_den * s_den
        spawn_proposed = spawn_accum_next >= spawn_den_scaled
        free_mask = (tiles_active == 0)
        has_free = jnp.any(free_mask)
        spawn_ok = spawn_proposed & has_free & (state.wave_active == 1)
        # only consume accumulator when a spawn actually happens; otherwise keep the credit
        spawn_accum_after = jax.lax.select(spawn_ok, spawn_accum_next - spawn_den_scaled, spawn_accum_next)

        col = jax.random.randint(k_spawn_col, (), 0, self.consts.BOARD_COLS)
        color_idx = jax.random.randint(k_spawn_color, (), 0, self.consts.N_TILE_TYPES)

        spawn_x = jnp.int32(self.consts.COLUMN_START_X + col * self.consts.COLUMN_STEP_X)
        first_free = jnp.argmax(free_mask.astype(jnp.int32))

        def do_spawn(vals):
            tiles_x, tiles_y, tiles_color, tiles_active, tiles_col, tiles_spawn_step, tiles_progress_accum = vals
            tiles_x = tiles_x.at[first_free].set(spawn_x)
            tiles_y = tiles_y.at[first_free].set(jnp.int32(self.consts.SPAWN_START_Y))
            tiles_color = tiles_color.at[first_free].set(color_idx)
            tiles_active = tiles_active.at[first_free].set(jnp.int32(1))
            tiles_col = tiles_col.at[first_free].set(col)
            tiles_spawn_step = tiles_spawn_step.at[first_free].set(step_counter)
            tiles_progress_accum = tiles_progress_accum.at[first_free].set(jnp.int32(0))
            return tiles_x, tiles_y, tiles_color, tiles_active, tiles_col, tiles_spawn_step, tiles_progress_accum

        (tiles_x, tiles_y, tiles_color, tiles_active, tiles_col, tiles_spawn_step, tiles_progress_accum) = jax.lax.cond(
            spawn_ok, do_spawn, lambda v: v,
            (tiles_x, tiles_y, tiles_color, tiles_active, tiles_col, tiles_spawn_step, tiles_progress_accum))

        # ---- player movement ----
        def col_left_x(c):
            return jnp.int32(self.consts.COLUMN_START_X + c * self.consts.COLUMN_STEP_X)

        px = state.player_x
        player_col = state.player_col
        target_col = state.player_target_col

        at_center = (px == col_left_x(player_col)) & (player_col == target_col)

        move_right = (
                (action == Action.RIGHT) |
                (action == Action.RIGHTFIRE) |
                (action == Action.UPRIGHT) |
                (action == Action.UPRIGHTFIRE) |
                (action == Action.DOWNRIGHT) |
                (action == Action.DOWNRIGHTFIRE)
        )
        move_left = (
                (action == Action.LEFT) |
                (action == Action.LEFTFIRE) |
                (action == Action.UPLEFT) |
                (action == Action.UPLEFTFIRE) |
                (action == Action.DOWNLEFT) |
                (action == Action.DOWNLEFTFIRE)
        )
        dcol_input = move_right.astype(jnp.int32) - move_left.astype(jnp.int32)

        target_col_new = jax.lax.cond(
            at_center & (dcol_input != 0),
            lambda _: jnp.clip(player_col + dcol_input, 0, jnp.int32(self.consts.BOARD_COLS - 1)),
            lambda _: target_col,
            operand=None,
        )

        tx = col_left_x(target_col_new)
        dx = tx - px
        sign = jnp.where(dx > 0, jnp.int32(1), jnp.where(dx < 0, jnp.int32(-1), jnp.int32(0)))
        step_pix = jnp.minimum(jnp.abs(dx), jnp.int32(self.consts.RESPONSIVENESS))
        px_next = px + sign * step_pix

        arrived = (px_next == tx)
        player_col_next = jax.lax.select(arrived, target_col_new, player_col)

        # ---- FIRE ----
        board = state.board
        is_fire = (
                (action == Action.FIRE) |
                (action == Action.RIGHTFIRE) | (action == Action.LEFTFIRE) |
                (action == Action.UPFIRE) | (action == Action.DOWNFIRE) |
                (action == Action.UPRIGHTFIRE) | (action == Action.UPLEFTFIRE) |
                (action == Action.DOWNRIGHTFIRE) | (action == Action.DOWNLEFTFIRE)
        ).astype(jnp.int32)
        col_occ = jnp.sum((board[:, player_col_next] > 0).astype(jnp.int32))
        can_fire, fire_lock_next = press_once(is_fire, state.fire_lock)

        can_place = can_fire & \
                    (player_backpack_count > 0) & \
                    (col_occ < jnp.int32(self.consts.BOARD_ROWS))

        def do_place(vals):
            board, bp_colors, bp_count = vals
            color = bp_colors[bp_count - 1]  # LIFO
            row_idx = col_occ
            board = board.at[row_idx, player_col_next].set(color + 1)
            return board, bp_colors, bp_count - 1

        board, player_backpack_colors, player_backpack_count = jax.lax.cond(
            can_place, do_place, lambda v: v, (board, player_backpack_colors, player_backpack_count)
        )

        # ---- UP key: shoot tile up ----
        is_up = (
                (action == Action.UP) | (action == Action.UPFIRE) |
                (action == Action.UPRIGHT) | (action == Action.UPRIGHTFIRE) |
                (action == Action.UPLEFT) | (action == Action.UPLEFTFIRE)
        )
        can_up, up_lock_next = press_once(is_up.astype(jnp.int32), state.up_lock)

        up_free_mask = (tiles_active == 0)
        up_has_free = jnp.any(up_free_mask)
        up_first_free = jnp.argmax(up_free_mask.astype(jnp.int32))

        up_can_spawn = can_up & up_has_free & (player_backpack_count > 0)

        def do_up(vals):
            (tiles_x, tiles_y, tiles_color, tiles_active, tiles_col, tiles_spawn_step,
             tiles_progress_accum, bp_colors, bp_count) = vals

            color_top = bp_colors[bp_count - 1]  # LIFO

            tiles_x = tiles_x.at[up_first_free].set(
                jnp.int32(self.consts.COLUMN_START_X + player_col_next * self.consts.COLUMN_STEP_X)
            )
            tiles_y = tiles_y.at[up_first_free].set(jnp.int32(self.consts.SHOOT_UP_Y))
            tiles_color = tiles_color.at[up_first_free].set(color_top)
            tiles_active = tiles_active.at[up_first_free].set(jnp.int32(1))
            tiles_col = tiles_col.at[up_first_free].set(player_col_next)
            tiles_spawn_step = tiles_spawn_step.at[up_first_free].set(step_counter)
            tiles_progress_accum = tiles_progress_accum.at[up_first_free].set(jnp.int32(0))

            bp_count = bp_count - 1
            return (tiles_x, tiles_y, tiles_color, tiles_active, tiles_col, tiles_spawn_step,
                    tiles_progress_accum, bp_colors, bp_count)

        (tiles_x, tiles_y, tiles_color, tiles_active, tiles_col, tiles_spawn_step,
         tiles_progress_accum, player_backpack_colors, player_backpack_count) = jax.lax.cond(
            up_can_spawn,
            do_up,
            lambda v: v,
            (tiles_x, tiles_y, tiles_color, tiles_active, tiles_col, tiles_spawn_step,
             tiles_progress_accum, player_backpack_colors, player_backpack_count),
        )

        # ---- match detection + gravity ----
        board_after, score_add, total_klax_add, horiz_klax_add, diag_klax_add = \
            self._apply_matches_and_gravity(board)

        new_score = state.score + score_add + cap_score_add

        # --- bonuses ---
        falling_count = jnp.sum(tiles_active.astype(jnp.int32))
        backpack_count_now = player_backpack_count
        tile_bonus_raw = (falling_count + backpack_count_now) * jnp.int32(10)

        n_on_board = jnp.sum((board_after > 0).astype(jnp.int32))
        total_cells = jnp.int32(self.consts.BOARD_ROWS * self.consts.BOARD_COLS)
        empty_bonus_raw = (total_cells - n_on_board) * jnp.int32(100)

        # --- progress per task ---
        # Tile waves: every tile that lands on the paddle or drops off the ramp.
        tile_land_add = cap_count + miss_count
        wave_points_progress_now = new_score - state.wave_score_base

        def progress_next_fn(_):
            tid = state.wave_task_id
            add_by_task = jax.lax.switch(
                jnp.clip(tid, 0, 5),
                (
                    lambda: total_klax_add,  # 0 KLAX
                    lambda: diag_klax_add,  # 1 DIAGONAL
                    lambda: tile_land_add,  # 2 TILE landings / drops
                    lambda: jnp.int32(0),  # 3 POINTS
                    lambda: horiz_klax_add,  # 4 HORIZONTAL
                    lambda: jnp.int32(0),  # fallback
                )
            )
            prog = jax.lax.select(
                tid == jnp.int32(3),
                jnp.maximum(jnp.int32(0), wave_points_progress_now),
                state.wave_progress + add_by_task
            )
            return prog

        wave_progress_next = jax.lax.cond(state.wave_active == 1, progress_next_fn, lambda _: state.wave_progress, 0)

        # --- check if wave is done ---
        wave_done_now = (state.wave_active == 1) & (wave_progress_next >= state.wave_target)
        bonus_add = jax.lax.select(wave_done_now, tile_bonus_raw + empty_bonus_raw, jnp.int32(0))
        new_score = new_score + bonus_add

        # remove all tiles, set cooldown, increase wave index
        def on_wave_done(vals):
            (tiles_x, tiles_y, tiles_color, tiles_active, tiles_col, tiles_spawn_step, tiles_progress_accum_in,
             board, player_backpack_colors, player_backpack_count, step_counter, wave_idx) = vals

            # remove all tiles
            tiles_x = jnp.zeros_like(tiles_x)
            tiles_y = jnp.zeros_like(tiles_y)
            tiles_color = jnp.zeros_like(tiles_color)
            tiles_active = jnp.zeros_like(tiles_active)
            tiles_col = jnp.zeros_like(tiles_col)
            tiles_spawn_step = jnp.zeros_like(tiles_spawn_step)

            board = jnp.zeros_like(board)
            player_backpack_colors = jnp.zeros_like(player_backpack_colors)
            player_backpack_count = jnp.int32(0)

            # CHANGED: Use pre-calculated steps
            cooldown_steps = jnp.int32(self.consts.WAVES_COOLDOWN_STEPS)

            tiles_progress_accum_out = jnp.zeros_like(tiles_progress_accum_in)
            spawn_progress_accum_out = jnp.int32(0)

            return (tiles_x, tiles_y, tiles_color, tiles_active, tiles_col,
                    tiles_spawn_step, tiles_progress_accum_out,
                    board, player_backpack_colors, player_backpack_count,
                    step_counter + cooldown_steps, wave_idx + 1)

        (tiles_x, tiles_y, tiles_color, tiles_active, tiles_col, tiles_spawn_step, tiles_progress_accum,
         board, player_backpack_colors, player_backpack_count,
         cooldown_until_new, wave_idx_after) = jax.lax.cond(
            wave_done_now,
            on_wave_done,
            lambda v: v,
            (tiles_x, tiles_y, tiles_color, tiles_active, tiles_col, tiles_spawn_step, tiles_progress_accum,
             board, player_backpack_colors, player_backpack_count, state.wave_cooldown_until, state.wave_idx)
        )
        board_after = jax.lax.select(wave_done_now, jnp.zeros_like(board_after), board_after)

        wave_active_next = jax.lax.select(wave_done_now, jnp.int32(0), state.wave_active)
        wave_progress_final = jax.lax.select(wave_done_now, jnp.int32(0), wave_progress_next)

        # After cooldown start the next wave if there is one
        def try_start_next(_):
            in_range = (wave_idx_after < jnp.int32(self._n_waves))
            def _start(_):
                tid = self._waves[wave_idx_after, 0]
                tgt = self._waves[wave_idx_after, 1]
                return (jnp.int32(1), tid, tgt, jnp.int32(0), new_score)
            def _noop(_):
                return (jnp.int32(0), state.wave_task_id, state.wave_target, state.wave_progress, state.wave_score_base)
            return jax.lax.cond(in_range, _start, _noop, 0)

        skip_intro = (wave_active_next == 0) & (can_fire == 1)
        do_start = (wave_active_next == 0) & ((step_counter >= cooldown_until_new) | skip_intro)
        (started, task_id_new, target_new, progress_new, score_base_new) = jax.lax.cond(do_start, try_start_next,
            lambda _: (jnp.int32(0), state.wave_task_id, state.wave_target, wave_progress_final, state.wave_score_base),0
        )

        wave_active_next = jax.lax.select(started == 1, jnp.int32(1), wave_active_next)
        wave_task_id_next = jax.lax.select(started == 1, task_id_new, state.wave_task_id)
        wave_target_next = jax.lax.select(started == 1, target_new, state.wave_target)
        wave_progress_final = jax.lax.select(started == 1, progress_new, wave_progress_final)
        wave_score_base_next = jax.lax.select(started == 1, score_base_new, state.wave_score_base)
        wave_idx_final = wave_idx_after
        spawn_accum_final = jax.lax.select(wave_done_now, jnp.int32(0), spawn_accum_after)
        first_credit = spawn_den_scaled - jnp.int32(self.consts.SPAWN_FIRST_DELAY_STEPS) * s_num
        spawn_accum_final = jax.lax.select(
            started == 1, jnp.maximum(first_credit, jnp.int32(0)), spawn_accum_final
        )

        return state.replace(
            step_counter=step_counter, rng_key=k_after, tiles_x=tiles_x, tiles_y=tiles_y, tiles_color=tiles_color,
            tiles_active=tiles_active, tiles_col=tiles_col, player_x=px_next, player_col=player_col_next,
            player_target_col=target_col_new, player_backpack_colors=player_backpack_colors,
            player_backpack_count=player_backpack_count, board=board_after, score=new_score, lives=new_lives,
            fire_lock=fire_lock_next, up_lock=up_lock_next, tiles_spawn_step=tiles_spawn_step,
            wave_idx=wave_idx_final,
            wave_task_id=wave_task_id_next,
            wave_target=wave_target_next,
            wave_progress=wave_progress_final,
            wave_score_base=wave_score_base_next,
            wave_active=wave_active_next,
            wave_cooldown_until=cooldown_until_new,
            tiles_progress_accum=tiles_progress_accum,
            spawn_progress_accum=spawn_accum_final,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: KlaxState) -> KlaxObservation:
        # --- Player ---
        # Calculate dynamic base Y based on backpack count (matching renderer logic)
        tile_h = self.consts.TILE_SIZE[1]
        base_y = self.consts.PLAYER_Y + state.player_backpack_count * (tile_h + 1)
        
        player = ObjectObservation.create(
            x=jnp.array(state.player_x, dtype=jnp.int32),
            y=jnp.array(base_y, dtype=jnp.int32),
            width=jnp.array(self.consts.PLAYER_WIDTH, dtype=jnp.int32),
            height=jnp.array(self.consts.PLAYER_HEIGHT, dtype=jnp.int32),
            active=jnp.array(1, dtype=jnp.int32)
        )

        # --- Falling Tiles ---
        tiles = ObjectObservation.create(
            x=jnp.clip(state.tiles_x, 0, self.consts.SCREEN_WIDTH),
            y=jnp.clip(state.tiles_y, 0, self.consts.SCREEN_HEIGHT),
            width=jnp.full((self.consts.MAX_TILES,), self.consts.TILE_SIZE[0], dtype=jnp.int32),
            height=jnp.full((self.consts.MAX_TILES,), self.consts.TILE_SIZE[1], dtype=jnp.int32),
            visual_id=state.tiles_color,
            active=state.tiles_active
        )

        # --- Backpack Items ---
        # Calculate positions for all potential backpack slots
        k_indices = jnp.arange(self.consts.PLAYER_BACKPACK_MAX, dtype=jnp.int32)
        
        # Renderer logic: y = base_y - (k + 1) * (tile_h + gap)
        gap = self.consts.BOARD_GAP
        bp_y = base_y - (k_indices + 1) * (tile_h + gap)
        
        bp_active = (k_indices < state.player_backpack_count).astype(jnp.int32)
        
        backpack_items = ObjectObservation.create(
            x=jnp.full((self.consts.PLAYER_BACKPACK_MAX,), state.player_x, dtype=jnp.int32),
            y=jnp.clip(bp_y, 0, self.consts.SCREEN_HEIGHT),
            width=jnp.full((self.consts.PLAYER_BACKPACK_MAX,), self.consts.TILE_SIZE[0], dtype=jnp.int32),
            height=jnp.full((self.consts.PLAYER_BACKPACK_MAX,), self.consts.TILE_SIZE[1], dtype=jnp.int32),
            visual_id=state.player_backpack_colors,
            active=bp_active
        )

        return KlaxObservation(
            player=player,
            tiles=tiles,
            backpack_items=backpack_items,
            board=state.board,
            score=state.score,
            lives=state.lives,
            wave_task=jnp.array([state.wave_task_id, state.wave_target], dtype=jnp.int32),
        )

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTION_SET))

    def observation_space(self) -> spaces.Dict:
        max_color = max(int(self.consts.N_TILE_TYPES), 1)
        screen_size = (self.consts.SCREEN_HEIGHT, self.consts.SCREEN_WIDTH)
        
        return spaces.Dict({
            "player": spaces.get_object_space(n=None, screen_size=screen_size),
            "tiles": spaces.get_object_space(n=self.consts.MAX_TILES, screen_size=screen_size),
            "backpack_items": spaces.get_object_space(n=self.consts.PLAYER_BACKPACK_MAX, screen_size=screen_size),
            "board": spaces.Box(
                low=0,
                high=max_color,
                shape=(self.consts.BOARD_ROWS, self.consts.BOARD_COLS),
                dtype=jnp.int32,
            ),
            "score": spaces.Box(low=0, high=9_999_999, shape=(), dtype=jnp.int32),
            "lives": spaces.Box(low=0, high=99, shape=(), dtype=jnp.int32),
            "wave_task": spaces.Box(
                low=-1,
                high=300_000,
                shape=(2,),
                dtype=jnp.int32,
            ),
        })

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.consts.SCREEN_HEIGHT, self.consts.SCREEN_WIDTH, 3),
            dtype=jnp.uint8,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: KlaxState) -> KlaxInfo:
        return KlaxInfo(time=state.step_counter)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: KlaxState, state: KlaxState) -> float:
        return (state.score - previous_state.score).astype(jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: KlaxState) -> bool:
        waves_finished = state.wave_idx >= jnp.int32(self._n_waves)
        well_full = jnp.all(state.board > 0)
        return (state.lives <= 0) | waves_finished | well_full

    def render(self, state: KlaxState) -> jnp.ndarray:
        return self.renderer.render(state)


class KlaxRenderer(JAXGameRenderer):
    def __init__(self, consts: KlaxConstants = None, config: render_utils.RendererConfig = None):
        self.consts = consts or KlaxConstants()
        super().__init__(self.consts)

        # Use injected config if provided, else default
        if config is None:
            self.config = render_utils.RendererConfig(
                game_dimensions=(self.consts.SCREEN_HEIGHT, self.consts.SCREEN_WIDTH),
                channels=3,
                downscale=None
            )
        else:
            self.config = config

        self.jr = render_utils.JaxRenderingUtils(self.config)

        # Load and process all sprites
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self._load_sprites()
        self._build_tile_scales()
        self._build_wave_intro_digits()

    def _load_sprites(self):
        """Loads assets using ASSET_CONFIG from constants (for modding)."""
        sprite_path = os.path.join(render_utils.get_base_sprite_dir(), "klax")
        native_h = int(self.consts.SCREEN_HEIGHT)
        native_w = int(self.consts.SCREEN_WIDTH)
        bg_path = os.path.join(sprite_path, "background.npy")
        bg = self.jr.loadFrame(bg_path)
        content_h = int(self.consts.CONTENT_HEIGHT)
        if int(bg.shape[0]) != content_h or int(bg.shape[1]) != native_w:
            bg = jax.image.resize(
                bg.astype(jnp.float32),
                (content_h, native_w, int(bg.shape[2])),
                method="nearest",
            ).astype(jnp.uint8)
        canvas = jnp.zeros((native_h, native_w, 4), dtype=jnp.uint8).at[:, :, 3].set(255)
        ph = min(int(bg.shape[0]), native_h)
        pw = min(int(bg.shape[1]), native_w)
        canvas = canvas.at[:ph, :pw].set(bg[:ph, :pw])

        asset_config = list(self.consts.ASSET_CONFIG)
        asset_config[0] = {"name": "background", "type": "background", "data": canvas}
        intro_files = [
            asset["file"]
            for asset in asset_config
            if isinstance(asset, dict) and str(asset.get("file", "")).startswith("wave_intro_")
        ]
        if intro_files and os.path.isdir(KLAX_MOD_SPRITE_DIR):
            asset_config.append({"type": "mod_path", "path": KLAX_MOD_SPRITE_DIR})
            asset_config.append({"type": "mod_path_filenames", "filenames": intro_files})
        palette, masks, bg_ids, color_to_id, flips = self.jr.load_and_setup_assets(asset_config, sprite_path)
        palette, black_id = self.jr.add_palette_color(palette, (0, 0, 0))
        color_to_id[(0, 0, 0)] = int(black_id)
        self.BLACK_ID = int(black_id)
        palette, white_id = self.jr.add_palette_color(palette, self.consts.PADDLE_COLOR)
        color_to_id[tuple(self.consts.PADDLE_COLOR)] = int(white_id)
        self.WHITE_ID = int(white_id)
        palette, green_id = self.jr.add_palette_color(palette, self.consts.WAVE_GREEN)
        color_to_id[tuple(self.consts.WAVE_GREEN)] = int(green_id)
        self.WAVE_GREEN_ID = int(green_id)
        return palette, masks, bg_ids, color_to_id, flips

    def _build_tile_scales(self):
        """Nearest-neighbor tile sizes, padded to one canvas so render_at stays rank-static."""
        tiles = self.SHAPE_MASKS["tile"]
        sizes = [(int(h), int(w)) for h, w in self.consts.TILE_SCALE_SIZES]
        max_h = max(h for h, _ in sizes)
        max_w = max(w for _, w in sizes)
        trans = jnp.int32(self.jr.TRANSPARENT_ID)

        stacked = []
        for h, w in sizes:
            scaled_batch = []
            for i in range(int(tiles.shape[0])):
                scaled = jax.image.resize(
                    tiles[i].astype(jnp.float32), (h, w), method="nearest"
                ).astype(tiles.dtype)
                out = jnp.full((max_h, max_w), trans, dtype=tiles.dtype)
                scaled_batch.append(out.at[:h, :w].set(scaled))
            stacked.append(jnp.stack(scaled_batch))
        self.TILE_SCALE_MASKS = jnp.stack(stacked)
        self.TILE_SCALE_HW = jnp.array(sizes, dtype=jnp.int32)

    def _build_wave_intro_digits(self):
        """Recolor score-digit ink only; the sprites' opaque black padding is not the glyph."""
        trans = jnp.int32(self.jr.TRANSPARENT_ID)
        score = self.SHAPE_MASKS["score_digits"]
        ink = self.COLOR_TO_ID.get((193, 192, 193))
        if ink is None:
            vals, counts = jnp.unique(score, return_counts=True)
            skip = {int(trans), int(self.BLACK_ID), 0}
            ink = 0
            best = -1
            for v, c in zip(vals.tolist(), counts.tolist()):
                if int(v) in skip:
                    continue
                if int(c) > best:
                    best = int(c)
                    ink = int(v)
        ink = jnp.int32(ink)
        self.WAVE_GREEN_DIGITS = jnp.where(score == ink, jnp.int32(self.WAVE_GREEN_ID), trans).astype(score.dtype)
        self.WAVE_WHITE_DIGITS = jnp.where(score == ink, jnp.int32(self.WHITE_ID), trans).astype(score.dtype)

    def _scale_index_for_y(self, y):
        # Full size by the catcher, not the well floor (board tiles use the last scale).
        y0 = jnp.int32(self.consts.SPAWN_START_Y)
        y1 = jnp.int32(self.consts.DESPAWN_Y)
        n = jnp.int32(self.consts.N_TILE_SCALES)
        span = jnp.maximum(y1 - y0, jnp.int32(1))
        return jnp.clip(((y - y0) * n) // span, 0, n - 1)

    def _blit_scaled_tile(self, raster, x_left, y, color_idx, scale_idx, flip_offset):
        mask = self.TILE_SCALE_MASKS[scale_idx, color_idx]
        hw = self.TILE_SCALE_HW[scale_idx]
        base_w = jnp.int32(self.consts.TILE_SIZE[0])
        x = x_left + (base_w - hw[1]) // 2
        return self.jr.render_at(raster, x, y, mask)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: KlaxState) -> jnp.ndarray:
        return jax.lax.cond(
            state.wave_active == 0,
            lambda s: self._render_wave_intro(s),
            lambda s: self._render_gameplay(s),
            state,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _render_wave_intro(self, state: KlaxState) -> jnp.ndarray:
        raster = jnp.full(self.BACKGROUND.shape, self.BLACK_ID, dtype=self.BACKGROUND.dtype)
        for name, x, y, flip_h in self.consts.WAVE_INTRO_BLITS:
            raster = self.jr.render_at(
                raster,
                int(x),
                int(y),
                self.SHAPE_MASKS[name],
                flip_horizontal=bool(flip_h),
                flip_offset=self.FLIP_OFFSETS[name],
            )
        n_waves = jnp.int32(self.consts.klax_waves.shape[0])
        widx = jnp.clip(state.wave_idx, 0, jnp.maximum(n_waves - 1, 0))
        tid = jnp.clip(self.consts.klax_waves[widx, 0], 0, 4).astype(jnp.int32)

        def blit_task_word(i):
            name, x, y = self.consts.WAVE_INTRO_TASK_BLITS[i]
            def _inner(r):
                return self.jr.render_at(
                    r, int(x), int(y), self.SHAPE_MASKS[name],
                    flip_offset=self.FLIP_OFFSETS[name],
                )
            return _inner

        raster = jax.lax.switch(
            tid,
            tuple(blit_task_word(i) for i in range(5)),
            raster,
        )
        wave_num = widx + jnp.int32(1)
        target = self.consts.klax_waves[widx, 1]

        wave_digits = self.jr.int_to_digits(wave_num, max_digits=2)
        raster = self.jr.render_label(
            raster,
            int(self.consts.WAVE_NUM_X),
            int(self.consts.WAVE_NUM_Y),
            wave_digits,
            self.WAVE_GREEN_DIGITS,
            int(self.consts.WAVE_NUM_SPACING),
            max_digits=2,
        )

        target_clamped = jnp.minimum(jnp.maximum(target, jnp.int32(0)), jnp.int32(999999))
        target_digits = self.jr.int_to_digits(target_clamped, max_digits=6)
        abs_t = jnp.abs(target_clamped)
        digit_count = jax.lax.cond(
            abs_t == 0,
            lambda: jnp.int32(1),
            lambda: jnp.floor(jnp.log10(abs_t.astype(jnp.float32)) + 1.0).astype(jnp.int32),
        )
        start_index = jnp.int32(6) - digit_count
        digit_w = int(self.WAVE_WHITE_DIGITS.shape[2])
        raster = self.jr.render_label_selective(
            raster,
            jnp.int32(self.consts.WAVE_TARGET_X_RIGHT) - jnp.int32(digit_w),
            jnp.int32(self.consts.WAVE_TARGET_Y),
            target_digits,
            self.WAVE_WHITE_DIGITS,
            start_index,
            digit_count,
            spacing=digit_w + 1,
            max_digits_to_render=6,
            right_align=True,
        )
        return self.jr.render_from_palette(raster, self.PALETTE)

    @partial(jax.jit, static_argnums=(0,))
    def _render_gameplay(self, state: KlaxState) -> jnp.ndarray:
        # --- 1. Initialize Raster ---
        raster = self.jr.create_object_raster(self.BACKGROUND)

        # dims
        tile_h: int = int(self.consts.TILE_SIZE[1])
        col_start: int = int(self.consts.COLUMN_START_X)
        col_step: int = int(self.consts.COLUMN_STEP_X)
        gap: int = int(self.consts.BOARD_GAP)

        # ---- falling tiles ----
        def body_tile(i, rast):
            active = (state.tiles_active[i] == 1)

            def draw(r):
                color_idx = state.tiles_color[i]
                scale_idx = self._scale_index_for_y(state.tiles_y[i])
                tile_offset = self.FLIP_OFFSETS["tile"]
                return self._blit_scaled_tile(
                    r, state.tiles_x[i], state.tiles_y[i], color_idx, scale_idx, tile_offset
                )

            return jax.lax.cond(active, draw, lambda r: r, rast)

        raster = jax.lax.fori_loop(0, self.consts.MAX_TILES, body_tile, raster)

        # ---- tile sprites on the board (5x5) ----
        def draw_cell(rc, rast):
            r = rc // self.consts.BOARD_COLS
            c = rc % self.consts.BOARD_COLS
            val = state.board[r, c]

            def draw(raster_):
                color_idx = val - 1
                tile_offset = self.FLIP_OFFSETS["tile"]
                x = jnp.int32(col_start + c * col_step)
                bottom_edge = jnp.int32(self.consts.BOARD_BOTTOM_Y) - r * jnp.int32(tile_h + gap)
                y = bottom_edge - jnp.int32(tile_h)
                # Bottom bin row is largest; higher rows step down one scale.
                n = jnp.int32(self.consts.N_TILE_SCALES)
                scale_idx = n - 1
                return self._blit_scaled_tile(raster_, x, y, color_idx, scale_idx, tile_offset)

            return jax.lax.cond(val > 0, draw, lambda rr: rr, rast)

        raster = jax.lax.fori_loop(0, self.consts.BOARD_ROWS * self.consts.BOARD_COLS, draw_cell, raster)

        # ---- player catcher (original well-grey sprite) + bottom lane marker ----
        player_offset = self.FLIP_OFFSETS["player"]
        base_y = jnp.int32(self.consts.PLAYER_Y + state.player_backpack_count * (tile_h+1))
        raster = self.jr.render_at(
            raster, state.player_x, base_y, self.SHAPE_MASKS["player"], flip_offset=player_offset
        )
        paddle_w = jnp.int32(self.consts.PADDLE_WIDTH)
        paddle_h = jnp.int32(self.consts.PADDLE_HEIGHT)
        paddle_x = state.player_x
        raster = self.jr.draw_rects(
            raster,
            jnp.array([[paddle_x, jnp.int32(self.consts.PADDLE_Y)]], dtype=jnp.int32),
            jnp.array([[paddle_w, paddle_h]], dtype=jnp.int32),
            self.WHITE_ID,
        )

        # ---- backpack (tiles the player is carrying) ----
        tile_offset = self.FLIP_OFFSETS["tile"]
        max_idx = jnp.int32(self.consts.N_TILE_SCALES - 1)
        def body_stack(k, rast):
            cond = k < state.player_backpack_count

            def draw(r):
                idx = state.player_backpack_colors[k]
                y = jnp.int32(base_y - (k + 1) * (tile_h + gap))
                return self._blit_scaled_tile(r, state.player_x, y, idx, max_idx, tile_offset)

            return jax.lax.cond(cond, draw, lambda r: r, rast)

        raster = jax.lax.fori_loop(0, self.consts.PLAYER_BACKPACK_MAX, body_stack, raster)

        # --- helper: right-aligned number, significant digits only (ALE skips leading zeros) ---
        def draw_number_right_aligned(rast, value_i32, digit_masks, x_right, y_top, max_digits: int):
            max_digits_int = int(max_digits)
            digits = self.jr.int_to_digits(value_i32, max_digits=max_digits_int)
            digit_width = int(digit_masks.shape[2])
            spacing = digit_width + 1
            abs_v = jnp.abs(value_i32)
            digit_count = jax.lax.cond(
                abs_v == 0,
                lambda: jnp.int32(1),
                lambda: jnp.floor(jnp.log10(abs_v.astype(jnp.float32)) + 1.0).astype(jnp.int32),
            )
            start_index = jnp.int32(max_digits_int) - digit_count
            x_lsd = jnp.int32(x_right) - jnp.int32(digit_width)
            return self.jr.render_label_selective(
                rast,
                x_lsd,
                jnp.int32(y_top),
                digits,
                digit_masks,
                start_index,
                digit_count,
                spacing=spacing,
                max_digits_to_render=max_digits_int,
                right_align=True,
            )

        # --- compute value displayed under "to go" ---
        tid = state.wave_task_id
        tgt = state.wave_target

        # Remaining for POINTS uses score delta since wave start; others use wave_progress
        points_done = state.score - state.wave_score_base
        remaining_generic = jnp.maximum(jnp.int32(0), tgt - state.wave_progress)
        remaining_points  = jnp.maximum(jnp.int32(0), tgt - jnp.maximum(points_done, 0))
        n_to_go = jax.lax.select(tid == jnp.int32(3), remaining_points, remaining_generic)

        # --- draw task label ---
        def draw_task_label(r):
            def draw_idx(idx):
                def _inner(rr):
                    label_mask = self.SHAPE_MASKS["task_labels"][idx]
                    label_offset = self.FLIP_OFFSETS["task_labels"]
                    return self.jr.render_at(rr, jnp.int32(54), jnp.int32(self.consts.HUD_TASK_Y), label_mask, flip_offset=label_offset)
                return _inner
            noop = lambda rr: rr
            return jax.lax.switch(jnp.clip(tid, 0, 5).astype(jnp.int32),
                              (draw_idx(0), draw_idx(1), draw_idx(2), draw_idx(3), draw_idx(4), noop), r)

        raster = draw_task_label(raster)

        # --- draw {n}_to_go, right-aligned, max value=250.000 ---
        n_to_go_clamped = jnp.minimum(n_to_go, jnp.int32(250000))
        raster = draw_number_right_aligned(raster, n_to_go_clamped, self.SHAPE_MASKS["togo_digits"],
                                           x_right=jnp.int32(91), y_top=jnp.int32(self.consts.HUD_TOGO_Y), max_digits=6)

        # --- draw {n}_lives_remaining ---
        lives_idx = jnp.clip(state.lives, 0, jnp.int32(self.SHAPE_MASKS["lives_remaining"].shape[0] - 1))
        lives_mask = self.SHAPE_MASKS["lives_remaining"][lives_idx]
        lives_offset = self.FLIP_OFFSETS["lives_remaining"]
        raster = self.jr.render_at(raster, jnp.int32(78), jnp.int32(self.consts.HUD_LIVES_Y), lives_mask, flip_offset=lives_offset)

        # --- draw {n}_score, right-aligned, max value=9.999.999 ---
        score_clamped = jnp.minimum(state.score, jnp.int32(9_999_999))
        raster = draw_number_right_aligned(raster, score_clamped, self.SHAPE_MASKS["score_digits"],
                                           x_right=jnp.int32(101), y_top=jnp.int32(self.consts.HUD_SCORE_Y), max_digits=7)

        # --- Final Palette Lookup ---
        return self.jr.render_from_palette(raster, self.PALETTE)
