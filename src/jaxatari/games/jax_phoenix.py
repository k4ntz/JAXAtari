import math
import os
from functools import partial
from typing import Tuple, NamedTuple
import jax
import jax.numpy as jnp
import chex
import jaxatari.spaces as spaces
import jaxatari.rendering.jax_rendering_utils as render_utils
import numpy as np
from flax import struct
from jaxatari.environment import JaxEnvironment, ObjectObservation, JAXAtariAction as Action
from jaxatari.spaces import Space
from jaxatari.modification import AutoDerivedConstants


def _create_static_procedural_sprites() -> dict:
    """Creates procedural sprites that don't depend on dynamic values."""
    # Create a black background (210x160)
    # We must add at least one pixel with alpha > 0
    # so the color (0,0,0) is added to the palette.
    bg_data = jnp.zeros((210, 160, 4), dtype=jnp.uint8)
    bg_data = bg_data.at[0, 0, 3].set(255) # Add one black, opaque pixel
    
    return {
        'background': bg_data
    }

def _get_default_asset_config() -> tuple:
    """
    Returns the default declarative asset manifest for Phoenix.
    Kept immutable (tuple of dicts) to fit NamedTuple defaults.
    """
    static_procedural = _create_static_procedural_sprites()
    return (
        # --- Background & Field ---
        {'name': 'background', 'type': 'background', 'data': static_procedural['background']},
        {'name': 'floor', 'type': 'single', 'file': 'floor.npy'},
        
        # --- UI ---
        {'name': 'digits', 'type': 'digits', 'pattern': 'digits/{}.npy'},
        {'name': 'life_indicator', 'type': 'single', 'file': 'life_indicator.npy'},
        
        # --- Player ---
        # This group's order must match the logic in render()
        # 0: idle, 1: death_1, 2: death_2, 3: death_3, 4: move
        {'name': 'player', 'type': 'group', 'files': [
            'player/player.npy', 
            'player/player_death_1.npy', 
            'player/player_death_2.npy', 
            'player/player_death_3.npy', 
            'player/player_move.npy'
        ]},
        {'name': 'player_ability', 'type': 'single', 'file': 'ability.npy'},
        
        # --- Projectiles ---
        {'name': 'player_projectile', 'type': 'single', 'file': 'projectiles/player_projectile.npy'},
        {'name': 'enemy_projectile', 'type': 'single', 'file': 'projectiles/enemy_projectile.npy'},
        
        # --- Phoenix ---
        # This group's order must match the logic in render()
        # 0: phoenix_1, 1: phoenix_2, 2: attack, 3: death_1, 4: death_2
        {'name': 'phoenix', 'type': 'group', 'files': [
            'enemy_phoenix/enemy_phoenix.npy',
            'enemy_phoenix/enemy_phoenix_2.npy',
            'enemy_phoenix/enemy_phoenix_attack.npy',
            'enemy_phoenix/enemy_phoenix_death_1.npy',
            'enemy_phoenix/enemy_phoenix_death_2.npy'
        ]},
        {'name': 'green_enemy', 'type': 'single', 'file': 'enemy_phoenix/green_enemy.npy'},
        
        # --- Bat Blue ---
        # 0: main, 1: death_1, 2: death_2, 3: death_3
        {'name': 'bat_blue_body', 'type': 'group', 'files': [
            'enemy_bats/bats_blue/bat_blue_main.npy',
            'enemy_bats/bats_blue/bat_blue_death_1.npy',
            'enemy_bats/bats_blue/bat_blue_death_2.npy',
            'enemy_bats/bats_blue/bat_blue_death_3.npy'
        ]},
        # 0: left_mid, 1: right_mid, 2: left_up, 3: right_up, ...
        {'name': 'bat_blue_wings', 'type': 'group', 'files': [
            'enemy_bats/bats_blue/bat_blue_left_wing_middle.npy',
            'enemy_bats/bats_blue/bat_blue_right_wing_middle.npy',
            'enemy_bats/bats_blue/bat_blue_left_wing_up.npy',
            'enemy_bats/bats_blue/bat_blue_right_wing_up.npy',
            'enemy_bats/bats_blue/bat_blue_left_wing_down.npy',
            'enemy_bats/bats_blue/bat_blue_right_wing_down.npy',
            'enemy_bats/bats_blue/bat_blue_left_wing_down_2.npy',
            'enemy_bats/bats_blue/bat_blue_right_wing_down_2.npy'
        ]},
        
        # --- Bat Red ---
        {'name': 'bat_red_body', 'type': 'group', 'files': [
            'enemy_bats/bats_red/bat_red_main.npy',
            'enemy_bats/bats_red/bat_red_death_1.npy',
            'enemy_bats/bats_red/bat_red_death_2.npy',
            'enemy_bats/bats_red/bat_red_death_3.npy'
        ]},
        {'name': 'bat_red_wings', 'type': 'group', 'files': [
            'enemy_bats/bats_red/bat_red_left_wing_middle.npy',
            'enemy_bats/bats_red/bat_red_right_wing_middle.npy',
            'enemy_bats/bats_red/bat_red_left_wing_up.npy',
            'enemy_bats/bats_red/bat_red_right_wing_up.npy',
            'enemy_bats/bats_red/bat_red_left_wing_down.npy',
            'enemy_bats/bats_red/bat_red_right_wing_down.npy',
            'enemy_bats/bats_red/bat_red_left_wing_down_2.npy',
            'enemy_bats/bats_red/bat_red_right_wing_down_2.npy'
        ]},

        # --- Boss ---
        {'name': 'boss', 'type': 'single', 'file': 'boss/boss.npy'},
        {'name': 'boss_block_red', 'type': 'single', 'file': 'boss/red_block.npy'},
        {'name': 'boss_block_blue', 'type': 'single', 'file': 'boss/blue_block.npy'},
        {'name': 'boss_block_green', 'type': 'single', 'file': 'boss/green_block.npy'},
    )

class PhoenixConstants(AutoDerivedConstants):
    """Game constants for Phoenix."""

    # --- Viewport and HUD colors ---
    WIDTH: int = struct.field(pytree_node=False, default=160)
    HEIGHT: int = struct.field(pytree_node=False, default=210)

    FLOOR_Y: int = struct.field(pytree_node=False, default=185)

    WINDOW_WIDTH: int = struct.field(pytree_node=False, default_factory=lambda: 160 * 3)
    WINDOW_HEIGHT: int = struct.field(pytree_node=False, default_factory=lambda: 210 * 3)
    SCORE_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default_factory=lambda: (210, 210, 64))

    # --- Entity pool sizes (max instances per type) ---
    MAX_PLAYER: int = struct.field(pytree_node=False, default=1)
    MAX_PLAYER_PROJECTILE: int = struct.field(pytree_node=False, default=1)
    MAX_PHOENIX: int = struct.field(pytree_node=False, default=8)
    MAX_BATS: int = struct.field(pytree_node=False, default=7)
    MAX_BOSS: int = struct.field(pytree_node=False, default=1)
    MAX_BOSS_BLOCK_GREEN: int = struct.field(pytree_node=False, default=30)
    MAX_BOSS_BLOCK_BLUE: int = struct.field(pytree_node=False, default=48)
    MAX_BOSS_BLOCK_RED: int = struct.field(pytree_node=False, default=126)

    # --- Hitbox / sprite dimensions (pixels) ---
    PROJECTILE_WIDTH: int = struct.field(pytree_node=False, default=2)
    PROJECTILE_HEIGHT: int = struct.field(pytree_node=False, default=4)
    ENEMY_WIDTH: int = struct.field(pytree_node=False, default=6)
    ENEMY_HEIGHT: int = struct.field(pytree_node=False, default=5)
    WING_WIDTH: int = struct.field(pytree_node=False, default=5)
    BLOCK_WIDTH: int = struct.field(pytree_node=False, default=4)
    BLOCK_HEIGHT: int = struct.field(pytree_node=False, default=4)
    BOSS_HALF_WIDTH: int = struct.field(pytree_node=False, default=16)
    BOSS_PROJECTILE_Y_OFFSET: int = struct.field(pytree_node=False, default=21)

    # --- Player: spawn, bounds, movement, palette ---
    PLAYER_POSITION: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: (79, 173))
    PLAYER_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default_factory=lambda: (213, 130, 74))
    PLAYER_BOUNDS: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: (0, 155))  # (left, right)
    PLAYER_EDGE_MARGIN: int = struct.field(pytree_node=False, default=16)  # pixels from each horizontal edge where movement stops
    PLAYER_STEP_SIZE: int = struct.field(pytree_node=False, default=1)
    PLAYER_LIVES: int = struct.field(pytree_node=False, default=4)  # Anzahl der Leben

    # --- Player timing (frames at game tick rate unless noted) ---
    PLAYER_DEATH_DURATION: int = struct.field(pytree_node=False, default=45)  # ca. 0,375 Sekunden bei 30 FPS
    PLAYER_ANIMATION_SPEED: int = struct.field(pytree_node=False, default=6)  # ca. 0,05 Sekunden bei 30 FPS
    PLAYER_RESPAWN_DURATION: int = struct.field(pytree_node=False, default=120)
    INVINCIBILITY_DURATION: int = struct.field(pytree_node=False, default=120)
    ABILITY_COOLDOWN: int = struct.field(pytree_node=False, default=300)

    # --- Projectiles ---
    PLAYER_PROJECTILE_SPEED: int = struct.field(pytree_node=False, default=6)
    PLAYER_PROJECTILE_INITIAL_OFFSET: int = struct.field(pytree_node=False, default=-5)
    RESET_START_LEVEL: int = struct.field(pytree_node=False, default=1)
    ENEMY_PROJECTILE_SPEED: int = struct.field(pytree_node=False, default=2)

    # --- Global / shared enemy timing and odds ---
    ENEMY_DEATH_DURATION: int = struct.field(pytree_node=False, default=30)  # ca. 0,25 Sekunden bei 30 FPS
    ENEMY_ANIMATION_SPEED: int = struct.field(pytree_node=False, default=30)  # ca. 0,25 Sekunden bei 30 FPS
    FIRE_CHANCE: float = struct.field(pytree_node=False, default=0.005)
    LEVEL_TRANSITION_DURATION: int = struct.field(pytree_node=False, default=120)

    # --- Phoenix (ship enemies): movement and dive attack AI ---
    PHOENIX_ENEMY_STEP_SIZE: float = struct.field(pytree_node=False, default=0.25)
    PHOENIX_ATTACK_SPEED: float = struct.field(pytree_node=False, default=0.65)
    PHOENIX_ATTACK_TOLERANCE: float = struct.field(pytree_node=False, default=0.5)
    # Per-frame roll while lowest-row phoenixes are eligible; higher = more frequent dives (ALE-like).
    PHOENIX_ATTACK_CHANCE: float = struct.field(pytree_node=False, default=0.009)
    PHOENIX_DRIFT_PROB: float = struct.field(pytree_node=False, default=0.78)
    PHOENIX_DRIFT_MAX: float = struct.field(pytree_node=False, default=0.42)
    # Shorter hold at dive depth before climbing back (was ~0.5–2s; now closer to twitchy ALE pacing).
    PHOENIX_ATTACK_DELAY_MIN: int = struct.field(pytree_node=False, default=6)
    PHOENIX_ATTACK_DELAY_MAX: int = struct.field(pytree_node=False, default=28)
    PHOENIX_RETURN_COOLDOWN: int = struct.field(pytree_node=False, default=6)

    # --- Phoenix coordinated salvos (multi-ship firing pattern) ---
    PHOENIX_SALVO_SHOT_GAP: int = struct.field(pytree_node=False, default=12)
    PHOENIX_SALVO_PRIMARY_MIN: int = struct.field(pytree_node=False, default=3)
    PHOENIX_SALVO_PRIMARY_MAX: int = struct.field(pytree_node=False, default=5)
    PHOENIX_SALVO_SECONDARY_MIN: int = struct.field(pytree_node=False, default=1)
    PHOENIX_SALVO_SECONDARY_MAX: int = struct.field(pytree_node=False, default=2)
    PHOENIX_SALVO_CYCLE_SHOT_CAP: int = struct.field(pytree_node=False, default=5)
    PHOENIX_SALVO_PAUSE_MIN: int = struct.field(pytree_node=False, default=90)
    PHOENIX_SALVO_PAUSE_MAX: int = struct.field(pytree_node=False, default=150)
    PHOENIX_SALVO_LONG_PAUSE_EVERY: int = struct.field(pytree_node=False, default=5)
    PHOENIX_SALVO_LONG_PAUSE_MULTIPLIER: int = struct.field(pytree_node=False, default=4)

    # --- Bats ---
    BAT_STEP_SIZE: float = struct.field(pytree_node=False, default=0.25)
    BAT_Y_STEP: int = struct.field(pytree_node=False, default=1)
    BAT_Y_CHANCE: float = struct.field(pytree_node=False, default=0.05)
    BAT_Y_COOLDOWN_RESET: int = struct.field(pytree_node=False, default=25)
    BAT_REGEN: int = struct.field(pytree_node=False, default=125)
    # Body x clip for bats must leave room for wings; `render_at` uses dynamic_slice and wraps
    # negative / overflow indices, so wings drawn at x - WING_WIDTH must have x >= WING_WIDTH
    # when the left wing is visible, and similarly for the right wing (see bat_step).
    BAT_X_MIN_WITH_LEFT_WING: int = struct.field(pytree_node=False, default=5)  # == WING_WIDTH
    BAT_X_MAX_WITH_RIGHT_WING: int = struct.field(pytree_node=False, default=150)  # WIDTH - ENEMY_WIDTH - WING_WIDTH + 1
    BAT_SALVO_SHOT_GAP: int = struct.field(pytree_node=False, default=12)
    BAT_SALVO_LONG_PAUSE_EVERY: int = struct.field(pytree_node=False, default=5)
    BAT_SALVO_LONG_PAUSE_MULTIPLIER: int = struct.field(pytree_node=False, default=4)
    BAT_DIVE_INTERVAL: int = struct.field(pytree_node=False, default=120)
    BAT_DIVE_FAST_THRESHOLD: int = struct.field(pytree_node=False, default=4)
    BAT_DIVE_HOLD_PIXELS: float = struct.field(pytree_node=False, default=4.0)
    BAT_DIVE_SLOW_INTERVAL: int = struct.field(pytree_node=False, default=4)
    BAT_DIVE_SLOW_STEP: float = struct.field(pytree_node=False, default=1.0)
    BAT_DIVE_FAST_INTERVAL: int = struct.field(pytree_node=False, default=2)
    BAT_DIVE_FAST_STEP: float = struct.field(pytree_node=False, default=2.0)
    BAT_DIVE_EARLY_PULLUP_FRAMES: int = struct.field(pytree_node=False, default=14)
    BAT_DIVE_PLAYER_OVERLAP_PX: float = struct.field(pytree_node=False, default=1.0)
    BAT_DIVE_BOTTOM_MISSING_Y_GAP: float = struct.field(pytree_node=False, default=8.0)
    BAT_DIVE_EXTRA_DEPTH: float = struct.field(pytree_node=False, default=0)
    BAT_DIVE_EXTRA_HOLD_FRAMES: int = struct.field(pytree_node=False, default=4)
    # Upper y-boundaries for bat body-kill scoring zones (100..450); above last bound => 500.
    BAT_SCORE_Y_BOUNDS: jnp.ndarray = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array([41.0, 57.0, 75.0, 88.0, 102.0, 124.0, 136.0, 148.0], dtype=jnp.float32),
    )
    BAT_SCORE_VALUES: jnp.ndarray = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array([100, 150, 200, 250, 300, 350, 400, 450, 500], dtype=jnp.int32),
    )

    # --- Boss ship and destructible boss blocks ---
    BOSS_BLOCK_STEP_SIZE: float = struct.field(pytree_node=False, default=4.0)
    BOSS_BLOCK_STEP_INTERVAL: int = struct.field(pytree_node=False, default=480)
    BOSS_BLOCK_MAX_Y: int = struct.field(pytree_node=False, default=140)
    BOSS_FIRE_CHANCE_MULTIPLIER: float = struct.field(pytree_node=False, default=3.0)
    BOSS_DROP_INTERVAL: int = struct.field(pytree_node=False, default=60)
    BOSS_DROP_SPEED: float = struct.field(pytree_node=False, default=2.0)
    BOSS_LOWEST_Y: float = struct.field(pytree_node=False, default=126.0)
    BOSS_BLUE_SHIFT_INTERVAL: int = struct.field(pytree_node=False, default=16)
    # Boss-only render delay (mod hook): projectile must travel this many pixels
    # below its spawn line before being visible. Default 0 keeps vanilla visibility.
    BOSS_PROJECTILE_RENDER_DELAY_PX: int = struct.field(pytree_node=False, default=0)
    # Y-boundaries for boss proximity scoring bands (boss.y spans ~76–126).
    BOSS_SCORE_Y_BOUNDS: jnp.ndarray = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array([89.0, 101.0, 114.0], dtype=jnp.float32),
    )
    BOSS_SCORE_VALUES: jnp.ndarray = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array([1000, 2000, 3000, 4000], dtype=jnp.int32),
    )
    BOSS_BLUE_BLOCK_WIDTH: int = struct.field(pytree_node=False, default=4)
    BOSS_BLUE_BLOCK_HEIGHT: int = struct.field(pytree_node=False, default=2)
    BOSS_RED_BLOCK_WIDTH: int = struct.field(pytree_node=False, default=4)
    BOSS_RED_BLOCK_HEIGHT: int = struct.field(pytree_node=False, default=3)
    BOSS_GREEN_BLOCK_WIDTH: int = struct.field(pytree_node=False, default=4)
    BOSS_GREEN_BLOCK_HEIGHT: int = struct.field(pytree_node=False, default=3)
    BOSS_CORE_WIDTH: int = struct.field(pytree_node=False, default=8)
    BOSS_CORE_HEIGHT: int = struct.field(pytree_node=False, default=10)
    BOSS_CORE_Y_OFFSET: float = struct.field(pytree_node=False, default=-18.0)
    BOSS_SPAWN_POSITION: Tuple[float, float] = struct.field(
        pytree_node=False, default_factory=lambda: (80.0, 76.0)
    )
    BOSS_BLUE_COLOR: Tuple[int, int, int] = struct.field(
        pytree_node=False, default_factory=lambda: (45, 87, 176)
    )
    BOSS_RED_COLOR: Tuple[int, int, int] = struct.field(
        pytree_node=False, default_factory=lambda: (167, 26, 26)
    )
    BOSS_GREEN_COLOR: Tuple[int, int, int] = struct.field(
        pytree_node=False, default_factory=lambda: (135, 183, 84)
    )

    # --- Enemy spawn grids (5 formations x 8 slots). X: playfield x (unused slot = -1). Y: row height; 230 = below playfield (inactive). ---
    ENEMY_POSITIONS_X: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array([
        [66, 90, 53, 104, 53, 104, 66, 90],
        [61, 75, 54, 82, 47, 89, 40, 96],
        [122, 129, 143, 127, 54, 49, 45, -1],
        [71, 97, 49, 105, 55, 105, 59, -1],
        [72, -1, -1, -1, -1, -1, -1, -1],
    ], dtype=jnp.float32))
    ENEMY_POSITIONS_Y: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array([
        [33, 33, 51, 51, 69, 69, 87, 87],
        [32, 32, 50, 50, 68, 68, 86, 86],
        [32, 52, 63, 89, 106, 125, 143, 230],
        [29, 47, 64, 82, 100, 119, 136, 230],
        [76, 230, 230, 230, 230, 230, 230, 230],
    ], dtype=jnp.float32))

    # --- Boss relative block layout (derived once for fast step/render) ---
    BOSS_BLUE_DX: jnp.ndarray = struct.field(pytree_node=False, default=None)
    BOSS_BLUE_DY: jnp.ndarray = struct.field(pytree_node=False, default=None)
    BOSS_RED_DX: jnp.ndarray = struct.field(pytree_node=False, default=None)
    BOSS_RED_DY: jnp.ndarray = struct.field(pytree_node=False, default=None)
    BOSS_GREEN_DX: jnp.ndarray = struct.field(pytree_node=False, default=None)
    BOSS_GREEN_DY: jnp.ndarray = struct.field(pytree_node=False, default=None)
    # Flattened blue offsets for vectorized rendering (2 rows × 20 cols → 40 elements)
    BOSS_FLAT_BLUE_DX: jnp.ndarray = struct.field(pytree_node=False, default=None)
    BOSS_FLAT_BLUE_DY: jnp.ndarray = struct.field(pytree_node=False, default=None)
    # Python ints for per-pixel block rendering (needed at jit trace time)
    BOSS_BLUE_X0: int = struct.field(pytree_node=False, default=-40)
    BOSS_RED_DY0: int = struct.field(pytree_node=False, default=4)
    BOSS_GREEN_DY0: int = struct.field(pytree_node=False, default=-3)

    # --- Asset manifest (immutable default; override via constants.replace) ---
    ASSET_CONFIG: tuple = _get_default_asset_config()


    def compute_derived(self):
        blue_dx = jnp.arange(-40, 40, 4, dtype=jnp.float32)
        blue_dy = jnp.array([0.0, 2.0], dtype=jnp.float32)

        red_rows = [20, 18, 16, 14, 12, 8, 4]
        r_dx = []
        r_dy = []
        current_dy = 4.0
        for n in red_rows:
            start_x = -(n * self.BOSS_RED_BLOCK_WIDTH) / 2.0
            for i in range(n):
                r_dx.append(start_x + i * self.BOSS_RED_BLOCK_WIDTH)
                r_dy.append(current_dy)
            current_dy += float(self.BOSS_RED_BLOCK_HEIGHT)
        red_dx = jnp.array(r_dx, dtype=jnp.float32)
        red_dy = jnp.array(r_dy, dtype=jnp.float32)

        green_rows = [12, 10, 8, 6, 4]
        g_dx = []
        g_dy = []
        current_dy = -3.0
        for n in green_rows:
            start_x = -(n * self.BOSS_GREEN_BLOCK_WIDTH) / 2.0
            for i in range(n):
                if i == n // 2 or i == (n // 2) - 1:
                    continue
                g_dx.append(start_x + i * self.BOSS_GREEN_BLOCK_WIDTH)
                g_dy.append(current_dy)
            current_dy -= float(self.BOSS_GREEN_BLOCK_HEIGHT)
        green_dx = jnp.array(g_dx, dtype=jnp.float32)
        green_dy = jnp.array(g_dy, dtype=jnp.float32)

        return {
            "BOSS_BLUE_DX": blue_dx,
            "BOSS_BLUE_DY": blue_dy,
            "BOSS_RED_DX": red_dx,
            "BOSS_RED_DY": red_dy,
            "BOSS_GREEN_DX": green_dx,
            "BOSS_GREEN_DY": green_dy,
            "BOSS_FLAT_BLUE_DX": jnp.tile(blue_dx, 2),
            "BOSS_FLAT_BLUE_DY": jnp.repeat(blue_dy, blue_dx.shape[0]),
            "BOSS_BLUE_X0": int(blue_dx[0].item()),
            "BOSS_RED_DY0": int(red_dy[0].item()),
            "BOSS_GREEN_DY0": int(green_dy[0].item()),
        }

# === GAME STATE ===
@struct.dataclass
class BossState:
    active: chex.Array
    x: chex.Array
    y: chex.Array
    blue_alive: chex.Array
    red_alive: chex.Array
    green_alive: chex.Array


@struct.dataclass
class PhoenixState:
    player_x: chex.Array
    player_y: chex.Array
    step_counter: chex.Array
    enemies_x: chex.Array # Gegner X-Positionen
    enemies_y: chex.Array
    horizontal_direction_enemies: chex.Array
    vertical_direction_enemies: chex.Array
    boss: BossState
    invincibility: chex.Array
    invincibility_timer: chex.Array
    ability_cooldown: chex.Array

    bat_wings: chex.Array
    bat_dying: chex.Array # Bat dying status, (8,), bool
    bat_death_timer: chex.Array # Timer for Bat death animation, (8,), int
    bat_wing_regen_timer: chex.Array
    bat_y_cooldown: chex.Array
    bat_midzone_roll_done: chex.Array
    bat_edge_profile_timer: chex.Array
    bat_motion_tick: chex.Array
    bat_dive_phase: chex.Array
    bat_dive_timer: chex.Array
    bat_dive_hold_timer: chex.Array
    bat_dive_travelled: chex.Array
    bat_dive_goal: chex.Array
    bat_anim_phase_offset: chex.Array

    phoenix_do_attack: chex.Array  # Phoenix attack state
    phoenix_attack_target_y: chex.Array  # Target Y position for Phoenix attack
    phoenix_original_y: chex.Array  # Original Y position of the Phoenix
    phoenix_cooldown: chex.Array
    phoenix_drift: chex.Array
    phoenix_returning: chex.Array # Returning status of the Phoenix
    phoenix_dying: chex.Array # Dying status of the Phoenix, (8,), bool
    phoenix_death_timer: chex.Array # Timer for Phoenix death animation, (8,), int

    player_dying: chex.Array = struct.field(default_factory=lambda: jnp.array(False))  # Player dying status, bool
    player_death_timer: chex.Array = struct.field(default_factory=lambda: jnp.array(0))  # Timer for player death animation, int
    player_moving: chex.Array = struct.field(default_factory=lambda: jnp.array(False)) # Player moving status, bool

    projectile_x: chex.Array = struct.field(default_factory=lambda: jnp.array(-1))  # Standardwert: kein Projektil
    projectile_y: chex.Array = struct.field(default_factory=lambda: jnp.array(-1))  # Standardwert: kein Projektil # Gegner Y-Positionen
    enemy_projectile_x: chex.Array = struct.field(default_factory=lambda: jnp.full((8,), -1)) # Enemy projectile X-Positionen
    enemy_projectile_y: chex.Array = struct.field(default_factory=lambda: jnp.full((8,), -1)) # Enemy projectile Y-Positionen

    score: chex.Array = struct.field(default_factory=lambda: jnp.array(0))  # Score
    lives: chex.Array = struct.field(default_factory=lambda: jnp.array(5)) # Lives
    player_respawn_timer: chex.Array = struct.field(default_factory=lambda: jnp.array(0)) # Invincibility timer
    level: chex.Array = struct.field(default_factory=lambda: jnp.array(1))  # Level, starts at 1
    level_transition_timer: chex.Array = struct.field(default_factory=lambda: jnp.array(0)) # Timer for level transition
    phoenix_salvo_gap_timer: chex.Array = struct.field(default_factory=lambda: jnp.array(0, dtype=jnp.int32))
    phoenix_salvo_cycle_shots: chex.Array = struct.field(default_factory=lambda: jnp.array(0, dtype=jnp.int32))
    phoenix_salvo_enemy_cooldowns: chex.Array = struct.field(default_factory=lambda: jnp.full((8,), 0, dtype=jnp.int32))
    phoenix_salvo_enemy_shot_counts: chex.Array = struct.field(default_factory=lambda: jnp.full((8,), 0, dtype=jnp.int32))
    bat_salvo_gap_timer: chex.Array = struct.field(default_factory=lambda: jnp.array(0, dtype=jnp.int32))
    bat_salvo_enemy_cooldowns: chex.Array = struct.field(default_factory=lambda: jnp.full((8,), 0, dtype=jnp.int32))
    bat_salvo_enemy_shot_counts: chex.Array = struct.field(default_factory=lambda: jnp.full((8,), 0, dtype=jnp.int32))

@struct.dataclass
class PhoenixObservation:
    player: ObjectObservation
    player_projectile: ObjectObservation
    enemy_projectiles: ObjectObservation  # n=8
    enemies: ObjectObservation  # n=8, state encodes wings
    boss: ObjectObservation
    boss_blocks: ObjectObservation
    player_score: chex.Array
    lives: chex.Array

@struct.dataclass
class PhoenixInfo:
    step_counter: jnp.ndarray

@struct.dataclass
class CarryState:
    score: chex.Array

@struct.dataclass
class EntityPosition:## not sure
    x: chex.Array
    y: chex.Array


class NonPhoenixEnemyFire(NamedTuple):
    """Bat-wave (large birds) or boss: random shots + spawn offsets and effective enemy origins."""

    enemy_fire_mask: chex.Array
    proj_offsets: chex.Array
    eff_enemy_x: chex.Array
    eff_enemy_y: chex.Array
    bat_gap_timer: chex.Array
    bat_enemy_cooldowns: chex.Array
    bat_enemy_shot_counts: chex.Array


class PhoenixSalvoEnemyFire(NamedTuple):
    """Small phoenix formation waves: salvo machine + projectile spawn positions."""

    salvo_fire_mask: chex.Array
    salvo_x: chex.Array
    salvo_y: chex.Array
    gap_timer: chex.Array
    cycle_shots: chex.Array
    enemy_cooldowns: chex.Array
    enemy_shot_counts: chex.Array


class BossBlueRotationArrays(NamedTuple):
    """Blue barrier row roll: which slots stay active after the lateral pattern, and their X coords."""

    alive_pattern: chex.Array
    x_positions: chex.Array


class BossDestructibleBlocksStep(NamedTuple):
    """Boss block step: concatenated alive/positions plus per-color arrays for state."""

    blocks_alive: chex.Array
    blocks_xy: chex.Array
    blue_rotation_alive: chex.Array
    blue_rotation_x: chex.Array
    green_blocks: chex.Array
    red_blocks: chex.Array
    blue_blocks: chex.Array
    new_enemies_y: chex.Array
    projectile_hit_detected: chex.Array


class JaxPhoenix(JaxEnvironment[PhoenixState, PhoenixObservation, PhoenixInfo, None]):
    # Minimal ALE action set for Phoenix
    ACTION_SET: jnp.ndarray = jnp.array(
        [
            Action.NOOP,
            Action.FIRE,
            Action.RIGHT,
            Action.LEFT,
            Action.DOWN,
            Action.RIGHTFIRE,
            Action.LEFTFIRE,
            Action.DOWNFIRE,
        ],
        dtype=jnp.int32,
    )
    
    def __init__(self, consts: PhoenixConstants = None):
        consts = consts or PhoenixConstants()
        super().__init__(consts)
        self.renderer = PhoenixRenderer(self.consts)
        self.step_counter = 0

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: PhoenixState) -> PhoenixObservation:
        c = self.consts
        w, h = int(c.WIDTH), int(c.HEIGHT)

        # --- Player ---
        p_alive = (~state.player_dying & (state.player_respawn_timer == 0)).astype(jnp.int32)
        player = ObjectObservation.create(
            x=jnp.clip(jnp.array(state.player_x, dtype=jnp.int32), 0, w),
            y=jnp.clip(jnp.array(state.player_y, dtype=jnp.int32), 0, h),
            width=jnp.array(13, dtype=jnp.int32),
            height=jnp.array(8, dtype=jnp.int32),
            active=p_alive
        )

        # --- Player Projectile ---
        p_active = (state.projectile_y > -1).astype(jnp.int32)
        player_projectile = ObjectObservation.create(
            x=jnp.clip(jnp.array(state.projectile_x, dtype=jnp.int32), 0, w),
            y=jnp.clip(jnp.array(state.projectile_y, dtype=jnp.int32), 0, h),
            width=jnp.array(c.PROJECTILE_WIDTH, dtype=jnp.int32),
            height=jnp.array(c.PROJECTILE_HEIGHT, dtype=jnp.int32),
            active=p_active
        )

        # --- Enemy Projectiles ---
        ep_active = (state.enemy_projectile_y > -1).astype(jnp.int32)
        enemy_projectiles = ObjectObservation.create(
            x=jnp.clip(state.enemy_projectile_x.astype(jnp.int32), 0, w),
            y=jnp.clip(state.enemy_projectile_y.astype(jnp.int32), 0, h),
            width=jnp.full((8,), c.PROJECTILE_WIDTH, dtype=jnp.int32),
            height=jnp.full((8,), c.PROJECTILE_HEIGHT, dtype=jnp.int32),
            active=ep_active
        )

        # --- Enemies ---
        # Map bat_wings (-1..2) to positive state (0..3)
        # -1 (Left) -> 1
        #  0 (None) -> 0
        #  1 (Right) -> 2
        #  2 (Both) -> 3
        # In non-bat levels, reset to 3 (Full)
        is_bat_level = jnp.logical_or((state.level % 5) == 3, (state.level % 5) == 4)
        raw_wings = state.bat_wings
        
        # Remap: -1->1, 0->0, 1->2, 2->3
        wing_state = jnp.select(
            [raw_wings == 0, raw_wings == -1, raw_wings == 1, raw_wings == 2],
            [0, 1, 2, 3],
            3 # Default to full
        ).astype(jnp.int32)
        
        # If not bat level, visual state is generic (3/Full)
        final_state = jnp.where(is_bat_level, wing_state, 3)
        
        # Encode death animation in the observation state while keeping enemies active on-screen.
        # Convention for this env:
        # - base state (0..3): wing status (bat levels) or generic (non-bat)
        # - +4: enemy is in death animation (still present, but different)
        dying_mask = jnp.where(is_bat_level, state.bat_dying, state.phoenix_dying).astype(jnp.int32)
        obs_enemy_state = (final_state + (dying_mask * 4)).astype(jnp.int32)

        enemies = ObjectObservation.create(
            x=jnp.clip(state.enemies_x.astype(jnp.int32), 0, w),
            y=jnp.clip(state.enemies_y.astype(jnp.int32), 0, h),
            width=jnp.full((8,), c.ENEMY_WIDTH, dtype=jnp.int32),
            height=jnp.full((8,), c.ENEMY_HEIGHT, dtype=jnp.int32),
            active=((state.enemies_x > -1) & (state.enemies_y < h + 10)).astype(jnp.int32),
            state=obs_enemy_state
        )

        # --- Boss ---
        boss_active = (((state.level % 5) == 0) & state.boss.active).astype(jnp.int32)
        boss = ObjectObservation.create(
            x=jnp.clip(state.boss.x.astype(jnp.int32), 0, w),
            y=jnp.clip(state.boss.y.astype(jnp.int32), 0, h),
            width=jnp.array(32, dtype=jnp.int32),
            height=jnp.array(16, dtype=jnp.int32),
            active=boss_active
        )

        # --- Boss Blocks ---
        # Flatten blocks
        is_boss_level = ((state.level % 5) == 0)

        blue_abs_x = state.boss.x + c.BOSS_BLUE_DX
        blue_abs_y = state.boss.y + c.BOSS_BLUE_DY
        blue_x = jnp.broadcast_to(blue_abs_x, state.boss.blue_alive.shape)
        blue_y = jnp.broadcast_to(blue_abs_y[:, None], state.boss.blue_alive.shape)

        bx_b = blue_x.reshape((-1,)).astype(jnp.int32)
        by_b = blue_y.reshape((-1,)).astype(jnp.int32)
        ba_b = (state.boss.blue_alive.reshape((-1,)) & is_boss_level & state.boss.active).astype(jnp.int32)
        vid_b = jnp.full(bx_b.shape, 0, dtype=jnp.int32)

        bx_r = (state.boss.x + c.BOSS_RED_DX).astype(jnp.int32)
        by_r = (state.boss.y + c.BOSS_RED_DY).astype(jnp.int32)
        ba_r = (state.boss.red_alive & is_boss_level & state.boss.active).astype(jnp.int32)
        vid_r = jnp.full(bx_r.shape, 1, dtype=jnp.int32)

        bx_g = (state.boss.x + c.BOSS_GREEN_DX).astype(jnp.int32)
        by_g = (state.boss.y + c.BOSS_GREEN_DY).astype(jnp.int32)
        ba_g = (state.boss.green_alive & is_boss_level & state.boss.active).astype(jnp.int32)
        vid_g = jnp.full(bx_g.shape, 2, dtype=jnp.int32)
        
        blocks_x = jnp.concatenate([bx_b, bx_r, bx_g])
        blocks_y = jnp.concatenate([by_b, by_r, by_g])
        blocks_active = jnp.concatenate([ba_b, ba_r, ba_g])
        blocks_vid = jnp.concatenate([vid_b, vid_r, vid_g])
        
        total_blocks = (
            int(c.BOSS_BLUE_DX.shape[0] * c.BOSS_BLUE_DY.shape[0])
            + int(c.BOSS_RED_DX.shape[0])
            + int(c.BOSS_GREEN_DX.shape[0])
        )
        
        boss_blocks = ObjectObservation.create(
            x=jnp.clip(blocks_x, 0, w),
            y=jnp.clip(blocks_y, 0, h),
            width=jnp.full((total_blocks,), c.BLOCK_WIDTH, dtype=jnp.int32),
            height=jnp.full((total_blocks,), c.BLOCK_HEIGHT, dtype=jnp.int32),
            active=blocks_active,
            visual_id=blocks_vid
        )

        return PhoenixObservation(
            player=player,
            player_projectile=player_projectile,
            enemy_projectiles=enemy_projectiles,
            enemies=enemies,
            boss=boss,
            boss_blocks=boss_blocks,
            player_score=state.score,
            lives=state.lives
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: PhoenixState) -> PhoenixInfo:
        return PhoenixInfo(
            step_counter=state.step_counter,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: PhoenixState) -> Tuple[bool, PhoenixState]:
        return jnp.less_equal(state.lives, 0)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: PhoenixState, state: PhoenixState):
        return state.score - previous_state.score


    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTION_SET))

    def observation_space(self) -> spaces.Dict:
        screen_size = (int(self.consts.HEIGHT), int(self.consts.WIDTH))
        single_obj = spaces.get_object_space(n=None, screen_size=screen_size)
        total_blocks = (
            int(self.consts.BOSS_BLUE_DX.shape[0] * self.consts.BOSS_BLUE_DY.shape[0])
            + int(self.consts.BOSS_RED_DX.shape[0])
            + int(self.consts.BOSS_GREEN_DX.shape[0])
        )
        
        return spaces.Dict({
            "player": single_obj,
            "player_projectile": single_obj,
            "enemy_projectiles": spaces.get_object_space(n=8, screen_size=screen_size),
            "enemies": spaces.get_object_space(n=8, screen_size=screen_size),
            "boss": single_obj,
            "boss_blocks": spaces.get_object_space(n=total_blocks, screen_size=screen_size),
            "player_score": spaces.Box(low=0, high=99999, shape=(), dtype=jnp.int32),
            "lives": spaces.Box(low=0, high=9, shape=(), dtype=jnp.int32),
        })

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(210, 160, 3),
            dtype=jnp.uint8
        )

    @partial(jax.jit, static_argnums=(0,))
    def player_step(self, state: PhoenixState, action: chex.Array) -> tuple[chex.Array]:
        step_size = self.consts.PLAYER_STEP_SIZE  # Größerer Wert = schnellerer Schritt
        # left action
        left = jnp.any(
            jnp.array(
                [
                    action == Action.LEFT,
                    action == Action.LEFTFIRE,
                ]
            )
        )
        # right action
        right = jnp.any(
            jnp.array(
                [
                    action == Action.RIGHT,
                    action == Action.RIGHTFIRE,
                ]
            )
        )
         #Ability : it holds on for ... amount
        invinsibility = jnp.any(jnp.array([action == Action.DOWN])) & (state.ability_cooldown == 0) & (state.invincibility_timer == 0)

        new_invinsibility = jnp.where(invinsibility, True, state.invincibility)
        new_timer = jnp.where(
            invinsibility & (state.invincibility_timer == 0),
            self.consts.INVINCIBILITY_DURATION,
            state.invincibility_timer
        )

        new_timer = jnp.where(new_timer > 0, new_timer - 1, 0)
        new_invinsibility = jnp.where(new_timer == 0, False, new_invinsibility)

        new_cooldown = jnp.where(new_timer == 2, self.consts.ABILITY_COOLDOWN, state.ability_cooldown)
        new_cooldown = jnp.where(new_cooldown > 0, new_cooldown - 1, 0)
        # movement right/left
        player_x = jnp.where(
            right & jnp.logical_not(new_invinsibility),
            state.player_x + step_size,
            jnp.where(left & jnp.logical_not(new_invinsibility), state.player_x - step_size, state.player_x),
        )

        # apply horizontal edge margin
        left_limit = self.consts.PLAYER_BOUNDS[0] + self.consts.PLAYER_EDGE_MARGIN
        right_limit = self.consts.PLAYER_BOUNDS[1] - self.consts.PLAYER_EDGE_MARGIN
        player_x = jnp.where(
            player_x < left_limit,
            left_limit,
            jnp.where(player_x > right_limit, right_limit, player_x),
        )

        # Did the player move?
        player_moved = jnp.not_equal(player_x.astype(jnp.int32), state.player_x.astype(jnp.int32))
        new_player_moving = player_moved.astype(jnp.bool_)

        state = state.replace(player_x= player_x.astype(jnp.int32),
                               invincibility=new_invinsibility,
                               invincibility_timer=new_timer,
                                player_moving=new_player_moving,
                                 ability_cooldown=new_cooldown
        )

        return state

    def phoenix_step(self, state):
        enemy_step_size = self.consts.PHOENIX_ENEMY_STEP_SIZE
        attack_speed = self.consts.PHOENIX_ATTACK_SPEED
        tolerance = self.consts.PHOENIX_ATTACK_TOLERANCE

        # Nur Gegner mit gültiger Position im Spielfeld bewegen
        active_enemies = (state.enemies_x > -1) & (state.enemies_y < self.consts.HEIGHT + 10) & (~state.phoenix_dying)

        # Unterste aktive Phoenixe (zum Starten eines Angriffs)
        masked_enemies_y = jnp.where(active_enemies, state.enemies_y, -jnp.inf)
        max_y = jnp.max(masked_enemies_y)
        lowest_mask = (state.enemies_y == max_y) & active_enemies

        # Angriff starten nur wenn nicht bereits am Angreifen/Zurückkehren und kein Cooldown
        can_attack = (
                lowest_mask
                & (~state.phoenix_do_attack)
                & (~state.phoenix_returning)
                & (state.phoenix_cooldown == 0)
                & ((state.phoenix_original_y == -1) | (state.enemies_y == state.phoenix_original_y))
        )
        key = jax.random.PRNGKey(state.step_counter)
        attack_chance = jax.random.uniform(key, shape=()) < self.consts.PHOENIX_ATTACK_CHANCE
        attack_trigger = lowest_mask & jnp.any(can_attack & attack_chance)

        # Zielbereich für den Angriff
        min_attack_y = jnp.max(jnp.where(active_enemies & (~state.phoenix_do_attack), state.enemies_y, -jnp.inf)) + 20
        max_attack_y = jnp.minimum(state.player_y - 10, self.consts.HEIGHT - 50)
        common_target_y = jax.random.randint(key, (), minval=min_attack_y, maxval=max_attack_y)

        new_phoenix_do_attack = jnp.where(attack_trigger, True, state.phoenix_do_attack)
        new_phoenix_attack_target_y = jnp.where(
            attack_trigger,
            jnp.full_like(state.phoenix_attack_target_y, common_target_y),
            state.phoenix_attack_target_y
        ).astype(jnp.float32)
        new_phoenix_original_y = jnp.where(attack_trigger, state.enemies_y, state.phoenix_original_y).astype(
            jnp.float32)

        # Drift nur beim Abtauchen/Anflug
        drift_prob = self.consts.PHOENIX_DRIFT_PROB
        drift_max = self.consts.PHOENIX_DRIFT_MAX
        num = state.enemies_x.shape[0]
        drift_key = jax.random.PRNGKey(state.step_counter + 999)
        dir_key, mag_key, on_key = jax.random.split(drift_key, 3)
        dir_sign = jnp.where(jax.random.uniform(dir_key, (num,)) < 0.5, -1.0, 1.0)
        magnitude = jax.random.uniform(mag_key, (num,)) * drift_max
        apply = jax.random.uniform(on_key, (num,)) < drift_prob
        drift_sample = jnp.where(apply, dir_sign * magnitude, 0.0).astype(jnp.float32)
        new_phoenix_drift = jnp.where(attack_trigger, drift_sample, state.phoenix_drift)

        # Angriffsbewegung (runter/hoch zum Ziel)
        going_down = new_phoenix_do_attack & (state.enemies_y < new_phoenix_attack_target_y - tolerance)
        going_up = new_phoenix_do_attack & (state.enemies_y > new_phoenix_attack_target_y + tolerance)

        # WICHTIG: Y-Bewegung nur für aktive Gegner
        new_enemies_y = jnp.where(active_enemies & going_down, state.enemies_y + attack_speed, state.enemies_y)
        new_enemies_y = jnp.where(active_enemies & going_up, new_enemies_y - attack_speed, new_enemies_y)

        # Seiten-Drift nur während des Abtauchens/an Zielanflug
        lateral_drift = jnp.where(going_down | going_up, new_phoenix_drift, 0.0).astype(jnp.float32)

        # Ziel erreicht? -> gemeinsamen "unten bleiben"-Cooldown starten
        target_reached = (~going_down) & (~going_up) & new_phoenix_do_attack
        key_delay = jax.random.PRNGKey(state.step_counter + 123)
        common_delay = jax.random.randint(
            key_delay,
            (),
            self.consts.PHOENIX_ATTACK_DELAY_MIN,
            self.consts.PHOENIX_ATTACK_DELAY_MAX
        )
        any_reached_target = jnp.any(target_reached & (state.phoenix_cooldown == 0))

        new_phoenix_cooldown = jnp.where(
            any_reached_target,
            jnp.full_like(state.phoenix_cooldown, common_delay),
            state.phoenix_cooldown
        )

        # Rückflug-Start wenn Cooldown abgelaufen
        start_return = target_reached & (new_phoenix_cooldown == 1)
        new_phoenix_returning = jnp.where(start_return, True, state.phoenix_returning)
        new_phoenix_do_attack = jnp.where(start_return, False, new_phoenix_do_attack)
        new_phoenix_attack_target_y = jnp.where(start_return, -1, new_phoenix_attack_target_y)

        # Rückflug: gleiches Tempo wie Angriff (nur aktive Gegner)
        returning_active = new_phoenix_returning
        dy = new_phoenix_original_y - new_enemies_y
        step = jnp.clip(dy, -attack_speed, attack_speed)
        new_enemies_y = jnp.where(active_enemies & returning_active, new_enemies_y + step, new_enemies_y)

        arrived = active_enemies & returning_active & (jnp.abs(new_enemies_y - new_phoenix_original_y) <= tolerance)
        new_enemies_y = jnp.where(arrived, new_phoenix_original_y, new_enemies_y)
        new_phoenix_returning = jnp.where(arrived, False, new_phoenix_returning)
        new_phoenix_original_y = jnp.where(arrived, -1, new_phoenix_original_y)
        new_phoenix_cooldown = jnp.where(arrived, self.consts.PHOENIX_RETURN_COOLDOWN, new_phoenix_cooldown)

        # Gruppenbewegung: nur während des Abtauchens ausnehmen
        group_mask = active_enemies & (~going_down)

        # Richtungswechsel nur anhand der oberen Formation
        direction_mask = active_enemies & (new_phoenix_original_y == -1)

        at_left_boundary = jnp.any(jnp.logical_and(state.enemies_x <= self.consts.PLAYER_BOUNDS[0], direction_mask))
        at_right_boundary = jnp.any(
            jnp.logical_and(
                state.enemies_x >= self.consts.PLAYER_BOUNDS[1] - self.consts.ENEMY_WIDTH / 2,
                direction_mask
            )
        )
        new_direction = jnp.where(
            at_left_boundary,
            jnp.full_like(state.horizontal_direction_enemies, 1.0, dtype=jnp.float32),
            jnp.where(
                at_right_boundary,
                jnp.full_like(state.horizontal_direction_enemies, -1.0, dtype=jnp.float32),
                state.horizontal_direction_enemies.astype(jnp.float32),
            ),
        )

        # Horizontale Bewegung anwenden
        group_step = jnp.where(group_mask, new_direction * enemy_step_size, 0.0).astype(jnp.float32)
        new_enemies_x = jnp.where(
            active_enemies,
            state.enemies_x + group_step + lateral_drift,
            state.enemies_x
        )
        # WICHTIG: Clipping nur für aktive Gegner, damit Tote (-1) nicht auf 0 geclippt werden
        clipped_x = jnp.clip(new_enemies_x, self.consts.PLAYER_BOUNDS[0], self.consts.PLAYER_BOUNDS[1])
        new_enemies_x = jnp.where(active_enemies, clipped_x, state.enemies_x)

        # Cooldown am Ende einmal dekrementieren
        new_phoenix_cooldown = jnp.where(new_phoenix_cooldown > 0, new_phoenix_cooldown - 1, 0)


        state = state.replace(
            enemies_x=new_enemies_x.astype(jnp.float32),
            horizontal_direction_enemies=new_direction.astype(jnp.float32),
            enemies_y=new_enemies_y.astype(jnp.float32),
            vertical_direction_enemies=state.vertical_direction_enemies.astype(jnp.float32),
            boss=state.boss,
            phoenix_do_attack=new_phoenix_do_attack,
            phoenix_attack_target_y=new_phoenix_attack_target_y.astype(jnp.float32),
            phoenix_original_y=new_phoenix_original_y.astype(jnp.float32),
            phoenix_cooldown=new_phoenix_cooldown.astype(jnp.int32),
            phoenix_drift=new_phoenix_drift.astype(jnp.float32),
            phoenix_returning=new_phoenix_returning.astype(jnp.bool_),
            phoenix_dying=state.phoenix_dying.astype(jnp.bool_),
            phoenix_death_timer=state.phoenix_death_timer.astype(jnp.int32),
            player_dying=state.player_dying.astype(jnp.bool_),
            player_death_timer=state.player_death_timer.astype(jnp.int32),
        )
        return state, 0.0, False

    def bat_step(self, state):
        bat_step_size = 1.0
        active_bats = (state.enemies_x > -1) & (state.enemies_y < self.consts.HEIGHT + 10) & (~state.bat_dying)
        proj_pos = jnp.array([state.projectile_x, state.projectile_y])
        vertical_tick = (state.step_counter % 13) == 0
        vertical_phase = (state.step_counter // 13) % 10
        shared_vertical_dir = jnp.where(vertical_phase < 5, -1.0, 1.0)
        base_y_move = jnp.where(
            active_bats & vertical_tick,
            jnp.array(self.consts.BAT_Y_STEP, dtype=jnp.float32) * shared_vertical_dir,
            0.0,
        ).astype(jnp.float32)
        alive_count = jnp.sum(active_bats.astype(jnp.int32))
        is_fast_dive = alive_count <= self.consts.BAT_DIVE_FAST_THRESHOLD
        dive_move_interval = jnp.where(
            is_fast_dive, self.consts.BAT_DIVE_FAST_INTERVAL, self.consts.BAT_DIVE_SLOW_INTERVAL
        )
        dive_step_pixels = jnp.where(
            is_fast_dive, self.consts.BAT_DIVE_FAST_STEP, self.consts.BAT_DIVE_SLOW_STEP
        ).astype(jnp.float32)
        dive_move_tick = (state.step_counter % dive_move_interval) == 0

        formation_idx = (state.level - 1) % 5
        slot_y = self.consts.ENEMY_POSITIONS_Y[formation_idx]
        valid_slots = slot_y < self.consts.HEIGHT + 10
        valid_slot_count = jnp.sum(valid_slots.astype(jnp.int32))
        lowest_slot_y = jnp.max(jnp.where(valid_slots, slot_y, -jnp.inf))
        lowest_alive_y = jnp.max(jnp.where(active_bats, state.enemies_y, -jnp.inf))
        # Trigger dives only when the bottom lane is truly gone (not from normal 1px formation wobble).
        missing_bottom_slots = (
            (alive_count > 0)
            & (alive_count < valid_slot_count)
            & (lowest_alive_y < (lowest_slot_y - self.consts.BAT_DIVE_BOTTOM_MISSING_Y_GAP))
        )
        ready_to_start_dive = (
            (state.bat_dive_phase == 0) & (state.bat_dive_timer <= 0) & missing_bottom_slots
        )
        dive_goal = jnp.where(
            ready_to_start_dive,
            (lowest_slot_y - lowest_alive_y) + self.consts.BAT_DIVE_EXTRA_DEPTH,
            state.bat_dive_goal,
        )
        # Convert the requested early pull-up timing (in frames) into pixels at the current dive speed.
        dive_pixels_per_frame = dive_step_pixels / jnp.maximum(dive_move_interval.astype(jnp.float32), 1.0)
        early_pullup_px = self.consts.BAT_DIVE_EARLY_PULLUP_FRAMES * dive_pixels_per_frame
        # Keep a slight overlap budget, but pull up earlier so dives don't heavily enter the player's lane.
        graze_player_goal = (
            (state.player_y - self.consts.ENEMY_HEIGHT + self.consts.BAT_DIVE_PLAYER_OVERLAP_PX)
            - lowest_alive_y
            - early_pullup_px
        )
        dive_goal = jnp.where(
            ready_to_start_dive,
            jnp.maximum(dive_goal, graze_player_goal),
            dive_goal,
        )
        dive_goal = jnp.maximum(dive_goal, 0.0).astype(jnp.float32)

        # hold_frames takes 2 static values depending on is_fast_dive — pre-computed to avoid ceil/div at runtime.
        _fast_hold = int(math.ceil(self.consts.BAT_DIVE_HOLD_PIXELS / max(self.consts.BAT_DIVE_FAST_STEP, 1e-5)))
        _slow_hold = int(math.ceil(self.consts.BAT_DIVE_HOLD_PIXELS / max(self.consts.BAT_DIVE_SLOW_STEP, 1e-5)))
        _fast_hold_frames = max(1, _fast_hold) * self.consts.BAT_DIVE_FAST_INTERVAL + self.consts.BAT_DIVE_EXTRA_HOLD_FRAMES
        _slow_hold_frames = max(1, _slow_hold) * self.consts.BAT_DIVE_SLOW_INTERVAL + self.consts.BAT_DIVE_EXTRA_HOLD_FRAMES
        hold_frames = jnp.where(is_fast_dive, _fast_hold_frames, _slow_hold_frames)

        bat_dive_phase = jnp.where(ready_to_start_dive, 1, state.bat_dive_phase).astype(jnp.int32)
        bat_dive_timer = jnp.where(
            state.bat_dive_phase == 0, jnp.maximum(state.bat_dive_timer - 1, 0), state.bat_dive_timer
        ).astype(jnp.int32)
        bat_dive_hold_timer = state.bat_dive_hold_timer.astype(jnp.int32)
        bat_dive_travelled = jnp.where(ready_to_start_dive, 0.0, state.bat_dive_travelled).astype(jnp.float32)
        y_move = jnp.where(bat_dive_phase == 0, base_y_move, 0.0).astype(jnp.float32)

        descend_move = jnp.where((bat_dive_phase == 1) & dive_move_tick, dive_step_pixels, 0.0).astype(jnp.float32)
        descend_left = jnp.maximum(dive_goal - bat_dive_travelled, 0.0)
        descend_applied = jnp.minimum(descend_move, descend_left).astype(jnp.float32)
        y_move = jnp.where(
            (bat_dive_phase == 1) & active_bats,
            descend_applied,
            y_move,
        ).astype(jnp.float32)
        bat_dive_travelled = jnp.where(bat_dive_phase == 1, bat_dive_travelled + descend_applied, bat_dive_travelled)
        reached_bottom = (bat_dive_phase == 1) & (bat_dive_travelled >= (dive_goal - 1e-3))
        bat_dive_phase = jnp.where(reached_bottom, 2, bat_dive_phase)
        bat_dive_hold_timer = jnp.where(reached_bottom, hold_frames, bat_dive_hold_timer)

        hold_countdown = jnp.maximum(bat_dive_hold_timer - 1, 0)
        done_holding = (bat_dive_phase == 2) & (hold_countdown == 0)
        bat_dive_hold_timer = jnp.where(bat_dive_phase == 2, hold_countdown, bat_dive_hold_timer)
        bat_dive_phase = jnp.where(done_holding, 3, bat_dive_phase)

        ascend_move = jnp.where((bat_dive_phase == 3) & dive_move_tick, dive_step_pixels, 0.0).astype(jnp.float32)
        ascend_applied = jnp.minimum(ascend_move, bat_dive_travelled).astype(jnp.float32)
        y_move = jnp.where((bat_dive_phase == 3) & active_bats, -ascend_applied, y_move).astype(jnp.float32)
        bat_dive_travelled = jnp.where(bat_dive_phase == 3, bat_dive_travelled - ascend_applied, bat_dive_travelled)
        finished_dive = (bat_dive_phase == 3) & (bat_dive_travelled <= 1e-3)
        bat_dive_phase = jnp.where(finished_dive, 0, bat_dive_phase).astype(jnp.int32)
        bat_dive_travelled = jnp.where(finished_dive, 0.0, bat_dive_travelled).astype(jnp.float32)
        dive_goal = jnp.where(finished_dive, 0.0, dive_goal).astype(jnp.float32)
        bat_dive_hold_timer = jnp.where(finished_dive, 0, bat_dive_hold_timer).astype(jnp.int32)
        bat_dive_timer = jnp.where(finished_dive, self.consts.BAT_DIVE_INTERVAL, bat_dive_timer).astype(jnp.int32)

        # The composite sprite is always placed at comp_x = x - WING_WIDTH, so x must stay in
        # [WING_WIDTH, WIDTH - ENEMY_WIDTH - WING_WIDTH] = [5, 149] for all bats regardless of
        # current wing state. Going outside this range causes the composite to overflow the raster
        # and render_at wraps the overflow pixels to the opposite edge of the screen.
        bat_x_min = jnp.full((8,), self.consts.BAT_X_MIN_WITH_LEFT_WING, dtype=jnp.float32)
        bat_x_max = jnp.full(
            (8,), self.consts.WIDTH - self.consts.ENEMY_WIDTH - self.consts.WING_WIDTH, dtype=jnp.float32
        )

        # Initialisiere neue Richtungen für jede Fledermaus
        bounced_directions = jnp.where(
            jnp.logical_and(state.enemies_x <= bat_x_min + 3, active_bats),
            jnp.ones(state.horizontal_direction_enemies.shape, dtype=jnp.float32),  # Force array shape
            jnp.where(
                jnp.logical_and(state.enemies_x >= bat_x_max - self.consts.ENEMY_WIDTH / 2,
                                active_bats),
                jnp.ones(state.horizontal_direction_enemies.shape, dtype=jnp.float32) * -1,  # Force array shape
                state.horizontal_direction_enemies.astype(jnp.float32)  # Ensure consistency
            )
        )

        middle_zone_min = self.consts.WIDTH / 3.0
        middle_zone_max = (2.0 * self.consts.WIDTH) / 3.0
        in_middle_zone = (
            (state.enemies_x >= middle_zone_min)
            & (state.enemies_x <= middle_zone_max)
            & active_bats
        )
        needs_roll = in_middle_zone & (~state.bat_midzone_roll_done)
        roll_key = jax.random.PRNGKey(state.step_counter + 911)
        flip_roll = jax.random.uniform(roll_key, shape=state.enemies_x.shape) < 0.5
        flip_now = needs_roll & flip_roll
        new_directions = jnp.where(flip_now, -bounced_directions, bounced_directions)
        new_midzone_roll_done = state.bat_midzone_roll_done | needs_roll

        edge_turn = active_bats & (bounced_directions != state.horizontal_direction_enemies.astype(jnp.float32))
        post_turn_timer = jnp.where(
            edge_turn,
            jnp.full((8,), 20, dtype=jnp.int32),
            jnp.maximum(state.bat_edge_profile_timer - 1, 0),
        )

        # 22-frame pre-edge profile at 1px / 2 frames in the last ~11 px before the edge.
        near_left_edge = new_directions < 0
        near_right_edge = new_directions > 0
        dist_left = state.enemies_x - bat_x_min.astype(jnp.float32)
        dist_right = bat_x_max.astype(jnp.float32) - state.enemies_x
        in_pre_edge_zone = active_bats & (
            (near_left_edge & (dist_left <= 11.0))
            | (near_right_edge & (dist_right <= 11.0))
        )

        motion_tick = jnp.where(active_bats, state.bat_motion_tick + 1, state.bat_motion_tick)
        profile_interval = jnp.where(
            post_turn_timer > 12,
            4,  # 8 frames -> 1 px every 4 frames
            jnp.where((post_turn_timer > 0) | in_pre_edge_zone, 2, 1),
        ).astype(jnp.int32)
        move_this_frame = active_bats & ((motion_tick % profile_interval) == 0)
        x_step = jnp.where(move_this_frame, new_directions * bat_step_size, 0.0)

        # Horizontal: nur aktive Bats bewegen und clippen
        proposed_x = jnp.where(active_bats, state.enemies_x + x_step, state.enemies_x)
        clipped_x = jnp.clip(proposed_x, bat_x_min.astype(jnp.float32), bat_x_max.astype(jnp.float32))
        new_enemies_x = jnp.where(active_bats, clipped_x, state.enemies_x)

        # Vertikal: nur aktive Bats bewegen und clippen
        proposed_y = jnp.where(active_bats, state.enemies_y + y_move, state.enemies_y)
        clipped_y = jnp.clip(proposed_y, 0, self.consts.HEIGHT - self.consts.ENEMY_HEIGHT)
        new_enemies_y = jnp.where(active_bats, clipped_y, state.enemies_y)

        # Für Kollisionen die neuen Y-Werte verwenden
        enemy_pos = jnp.stack([new_enemies_x, new_enemies_y], axis=1)

        new_y_cooldown = state.bat_y_cooldown

        def check_collision(entity_pos, projectile_pos):
            enemy_x, enemy_y = entity_pos
            proj_x, proj_y = projectile_pos
            wing_left_x = enemy_x - 5
            wing_y = enemy_y + 2
            wing_right_x = enemy_x + 5
            wing_hitbox_extra = 1
            collision_x_left = (proj_x + self.consts.PROJECTILE_WIDTH > wing_left_x) & (
                    proj_x < wing_left_x + self.consts.WING_WIDTH + wing_hitbox_extra)
            collision_y = (proj_y + self.consts.PROJECTILE_HEIGHT > wing_y) & (
                    proj_y < wing_y + 1 + wing_hitbox_extra)
            collision_x_right = (proj_x + self.consts.PROJECTILE_WIDTH > wing_right_x) & (
                    proj_x < wing_right_x + self.consts.WING_WIDTH + wing_hitbox_extra)

            return collision_x_left & collision_y, collision_x_right & collision_y

        left_wing_collision, right_wing_collision = jax.vmap(lambda entity_pos: check_collision(entity_pos, proj_pos))(enemy_pos)
        raw_left_valid = left_wing_collision & ((state.bat_wings == 2) | (state.bat_wings == -1))
        raw_right_valid = right_wing_collision & ((state.bat_wings == 2) | (state.bat_wings == 1))

        # Find the lowest left wing hit
        y_left = jnp.where(raw_left_valid, new_enemies_y, -jnp.inf)
        best_left_idx = jnp.argmax(y_left)
        any_left = jnp.any(raw_left_valid)
        
        # Find the lowest right wing hit
        y_right = jnp.where(raw_right_valid, new_enemies_y, -jnp.inf)
        best_right_idx = jnp.argmax(y_right)
        any_right = jnp.any(raw_right_valid)

        # Prioritize whichever wing hit is lowest overall
        best_y_left = jnp.max(y_left)
        best_y_right = jnp.max(y_right)
        
        apply_left = any_left & (best_y_left >= best_y_right)
        apply_right = any_right & (~apply_left)
        
        left_hit_valid = (jnp.arange(8) == best_left_idx) & apply_left
        right_hit_valid = (jnp.arange(8) == best_right_idx) & apply_right

        # Only remove the projectile if any valid hit occurred
        any_valid_hit = jnp.any(left_hit_valid | right_hit_valid)
        def update_wing_state(current_state, left_hit, right_hit):
            # current_state: int (-1,0,1,2), left_hit & right_hit: bool

            # First handle left wing hit
            updated = jnp.where(
                left_hit,
                jnp.where(current_state == 2, 1,  # both wings → right wing only
                          jnp.where(current_state == -1, 0, current_state)),  # right only → none, else unchanged
                current_state
            )

            # Then handle right wing hit
            updated = jnp.where(
                right_hit,
                jnp.where(updated == 2, -1,  # both wings → left wing only
                          jnp.where(updated == 1, 0, updated)),  # left only → none, else unchanged
                updated
            )

            return updated

        new_bat_wings = jax.vmap(update_wing_state)(state.bat_wings, left_wing_collision, right_wing_collision)

        no_wings = (new_bat_wings == 0) & active_bats
        new_regen_timer = jnp.where(no_wings, state.bat_wing_regen_timer + 1, 0)
        regenerated = (new_regen_timer >= self.consts.BAT_REGEN)
        new_bat_wings = jnp.where(regenerated, 2, new_bat_wings)
        new_regen_timer = jnp.where(regenerated, 0, new_regen_timer)

        state = state.replace(
            enemies_x=new_enemies_x.astype(jnp.float32),
            enemies_y=new_enemies_y.astype(jnp.float32),
            horizontal_direction_enemies=new_directions.astype(jnp.float32),
            boss=state.boss,
            bat_wings= new_bat_wings,
            bat_wing_regen_timer=new_regen_timer,
            bat_y_cooldown=new_y_cooldown.astype(jnp.int32),
            bat_midzone_roll_done=new_midzone_roll_done.astype(jnp.bool_),
            bat_edge_profile_timer=post_turn_timer.astype(jnp.int32),
            bat_motion_tick=motion_tick.astype(jnp.int32),
            bat_dive_phase=bat_dive_phase,
            bat_dive_timer=bat_dive_timer,
            bat_dive_hold_timer=bat_dive_hold_timer,
            bat_dive_travelled=bat_dive_travelled.astype(jnp.float32),
            bat_dive_goal=dive_goal.astype(jnp.float32),
        )

        return state, jnp.where(any_valid_hit, 20.0, 0.0), any_valid_hit

    def _boss_player_missile_hits_barrier(
        self, state: PhoenixState, projectile_x: chex.Array, projectile_y: chex.Array
    ) -> chex.Array:
        """True if the player missile overlaps any visible boss barrier block."""
        c = self.consts
        boss = state.boss
        proj_x = projectile_x.astype(jnp.float32)
        proj_y = projectile_y.astype(jnp.float32)

        blue_abs_x = boss.x + c.BOSS_BLUE_DX
        blue_abs_y = boss.y + c.BOSS_BLUE_DY
        blue_x = jnp.broadcast_to(blue_abs_x, boss.blue_alive.shape)
        blue_y = jnp.broadcast_to(blue_abs_y[:, None], boss.blue_alive.shape)
        blue_hit_x = (proj_x + c.PROJECTILE_WIDTH >= blue_x) & (proj_x <= blue_x + c.BOSS_BLUE_BLOCK_WIDTH)
        blue_hit_y = (proj_y + c.PROJECTILE_HEIGHT >= blue_y) & (proj_y <= blue_y + c.BOSS_BLUE_BLOCK_HEIGHT)
        blue_hit = jnp.any(blue_hit_x & blue_hit_y & boss.blue_alive)

        red_x = boss.x + c.BOSS_RED_DX
        red_y = boss.y + c.BOSS_RED_DY
        red_hit_x = (proj_x + c.PROJECTILE_WIDTH >= red_x) & (proj_x <= red_x + c.BOSS_RED_BLOCK_WIDTH)
        red_hit_y = (proj_y + c.PROJECTILE_HEIGHT >= red_y) & (proj_y <= red_y + c.BOSS_RED_BLOCK_HEIGHT)
        red_hit = jnp.any(red_hit_x & red_hit_y & boss.red_alive)

        green_x = boss.x + c.BOSS_GREEN_DX
        green_y = boss.y + c.BOSS_GREEN_DY
        green_hit_x = (proj_x + c.PROJECTILE_WIDTH >= green_x) & (proj_x <= green_x + c.BOSS_GREEN_BLOCK_WIDTH)
        green_hit_y = (proj_y + c.PROJECTILE_HEIGHT >= green_y) & (proj_y <= green_y + c.BOSS_GREEN_BLOCK_HEIGHT)
        green_hit = jnp.any(green_hit_x & green_hit_y & boss.green_alive)
        return blue_hit | red_hit | green_hit

    def boss_step(self, state):
        c = self.consts
        boss = state.boss
        step = state.step_counter
        player_respawning = state.player_respawn_timer > 0

        drop_tick = (step % c.BOSS_DROP_INTERVAL == 0)
        dropped_y = jnp.minimum(boss.y + c.BOSS_DROP_SPEED, c.BOSS_LOWEST_Y)
        new_boss_y = jnp.where(drop_tick & boss.active & (~player_respawning), dropped_y, boss.y)

        shift_tick = (step % c.BOSS_BLUE_SHIFT_INTERVAL == 0) & (~player_respawning)
        new_blue_alive = jnp.where(shift_tick, jnp.roll(boss.blue_alive, shift=1, axis=1), boss.blue_alive)

        proj_x = state.projectile_x.astype(jnp.float32)
        proj_y = state.projectile_y.astype(jnp.float32)
        proj_active = (state.projectile_y >= 0)

        def check_block_hits(dx_array, dy_array, alive_mask, block_w, block_h, is_2d_blue=False):
            if is_2d_blue:
                abs_x = boss.x + dx_array
                abs_y = new_boss_y + dy_array
                abs_x = jnp.broadcast_to(abs_x, (2, 20))
                abs_y = jnp.broadcast_to(abs_y[:, None], (2, 20))
            else:
                abs_x = boss.x + dx_array
                abs_y = new_boss_y + dy_array

            hit_x = (proj_x + c.PROJECTILE_WIDTH >= abs_x) & (proj_x <= abs_x + block_w)
            hit_y = (proj_y + c.PROJECTILE_HEIGHT >= abs_y) & (proj_y <= abs_y + block_h)
            hits = hit_x & hit_y & alive_mask & proj_active & boss.active
            hit_detected = jnp.any(hits)

            flat_hits = hits.flatten()
            flat_abs_y = abs_y.flatten()
            hit_y_vals = jnp.where(flat_hits, flat_abs_y, -jnp.inf)
            best_idx = jnp.argmax(hit_y_vals)
            single_hit_flat = (jnp.arange(flat_hits.shape[0]) == best_idx) & hit_detected
            single_hit_mask = single_hit_flat.reshape(hits.shape)
            new_alive_mask = jnp.where(single_hit_mask, False, alive_mask)
            return new_alive_mask, hit_detected

        new_red_alive, red_hit = check_block_hits(
            c.BOSS_RED_DX,
            c.BOSS_RED_DY,
            boss.red_alive,
            c.BOSS_RED_BLOCK_WIDTH,
            c.BOSS_RED_BLOCK_HEIGHT,
        )
        _blue_alive_hit, _blue_hit = check_block_hits(
            c.BOSS_BLUE_DX,
            c.BOSS_BLUE_DY,
            new_blue_alive,
            c.BOSS_BLUE_BLOCK_WIDTH,
            c.BOSS_BLUE_BLOCK_HEIGHT,
            is_2d_blue=True,
        )
        new_blue_alive = jnp.where(red_hit, new_blue_alive, _blue_alive_hit)
        blue_hit = (~red_hit) & _blue_hit

        _green_alive_hit, _green_hit = check_block_hits(
            c.BOSS_GREEN_DX,
            c.BOSS_GREEN_DY,
            boss.green_alive,
            c.BOSS_GREEN_BLOCK_WIDTH,
            c.BOSS_GREEN_BLOCK_HEIGHT,
        )
        new_green_alive = jnp.where(red_hit | blue_hit, boss.green_alive, _green_alive_hit)
        green_hit = (~(red_hit | blue_hit)) & _green_hit

        core_abs_y = new_boss_y + c.BOSS_CORE_Y_OFFSET
        core_hit_x = (proj_x + c.PROJECTILE_WIDTH >= boss.x - c.BOSS_CORE_WIDTH / 2.0) & (
            proj_x <= boss.x + c.BOSS_CORE_WIDTH / 2.0
        )
        core_hit_y = (proj_y + c.PROJECTILE_HEIGHT >= core_abs_y) & (proj_y <= core_abs_y + c.BOSS_CORE_HEIGHT)
        core_hit = core_hit_x & core_hit_y & proj_active & boss.active & ~(red_hit | blue_hit | green_hit)

        any_hit = red_hit | blue_hit | green_hit | core_hit
        new_boss_active = jnp.where(core_hit, False, boss.active)
        score_idx = jnp.clip(
            jnp.searchsorted(c.BOSS_SCORE_Y_BOUNDS, boss.y, side="left"),
            0, c.BOSS_SCORE_VALUES.shape[0] - 1,
        )
        boss_round = (state.level - 1) // 5
        kill_score = jnp.clip(c.BOSS_SCORE_VALUES[score_idx] + boss_round * 1000, 1000, 9000)
        score_reward = jnp.where(core_hit, kill_score, 0.0)
        new_boss = boss.replace(
            active=new_boss_active,
            y=new_boss_y,
            blue_alive=new_blue_alive,
            red_alive=new_red_alive,
            green_alive=new_green_alive,
        )

        new_enemies_x = state.enemies_x.at[0].set(jnp.where(new_boss_active, boss.x, -1.0))
        new_enemies_y = state.enemies_y.at[0].set(
            jnp.where(new_boss_active, new_boss_y, c.HEIGHT + 20.0)
        )

        state = state.replace(
            boss=new_boss,
            enemies_x=new_enemies_x.astype(jnp.float32),
            enemies_y=new_enemies_y.astype(jnp.float32),
        )
        return state, score_reward, any_hit

    def _initial_enemy_horizontal_directions(
        self, level: chex.Array, enemies_x: chex.Array, enemies_y: chex.Array
    ) -> chex.Array:
        """Initial bat-wave directions: top 4 right, bottom 3 left."""
        default_dirs = jnp.full((8,), -1.0, dtype=jnp.float32)
        is_bat_level = jnp.logical_or((level % 5) == 3, (level % 5) == 4)

        active = (enemies_x > -1) & (enemies_y < self.consts.HEIGHT + 10)
        active_count = jnp.sum(active.astype(jnp.int32))
        sort_key = jnp.where(active, enemies_y, jnp.inf)
        order = jnp.argsort(sort_key)
        rank = jnp.zeros((8,), dtype=jnp.int32).at[order].set(jnp.arange(8, dtype=jnp.int32))

        top_mask = active & (rank < 4)
        bottom_start = jnp.maximum(active_count - 3, 0)
        bottom_mask = active & (rank >= bottom_start) & (~top_mask)

        bat_dirs = default_dirs
        bat_dirs = jnp.where(top_mask, 1.0, bat_dirs)
        bat_dirs = jnp.where(bottom_mask, -1.0, bat_dirs)
        return jnp.where(is_bat_level, bat_dirs, default_dirs).astype(jnp.float32)

    def _pack_alive_bats_to_top_slots(
        self, level: chex.Array, enemies_x: chex.Array, enemies_y: chex.Array
    ) -> tuple[chex.Array, chex.Array]:
        """After player death/respawn in bat waves, move alive bats into highest formation slots."""
        is_bat_level = jnp.logical_or((level % 5) == 3, (level % 5) == 4)
        formation_idx = (level - 1) % 5
        slot_x = self.consts.ENEMY_POSITIONS_X[formation_idx]
        slot_y = self.consts.ENEMY_POSITIONS_Y[formation_idx]
        valid_slots = (slot_x > -1) & (slot_y < self.consts.HEIGHT + 10)
        sorted_slot_ids = jnp.argsort(jnp.where(valid_slots, slot_y, jnp.inf))

        alive = (enemies_x > -1) & (enemies_y < self.consts.HEIGHT + 10)
        alive_count = jnp.sum(alive.astype(jnp.int32))
        sorted_enemy_ids = jnp.argsort(jnp.where(alive, enemies_y, jnp.inf))
        enemy_rank = jnp.zeros((8,), dtype=jnp.int32).at[sorted_enemy_ids].set(jnp.arange(8, dtype=jnp.int32))
        assign_mask = alive & (enemy_rank < alive_count)
        assigned_slot_id = jnp.take(sorted_slot_ids, jnp.clip(enemy_rank, 0, 7))

        packed_x = jnp.where(assign_mask, slot_x[assigned_slot_id], enemies_x)
        packed_y = jnp.where(assign_mask, slot_y[assigned_slot_id], enemies_y)
        out_x = jnp.where(is_bat_level, packed_x, enemies_x)
        out_y = jnp.where(is_bat_level, packed_y, enemies_y)
        return out_x.astype(jnp.float32), out_y.astype(jnp.float32)

    def reset(self, key: jax.random.PRNGKey = jax.random.PRNGKey(42)) -> Tuple[PhoenixObservation, PhoenixState]:
        initial_level = jnp.array(self.consts.RESET_START_LEVEL, dtype=jnp.int32)
        key, bat_anim_key = jax.random.split(key)
        initial_formation_idx = (initial_level - 1) % 5
        initial_enemies_x = self.consts.ENEMY_POSITIONS_X[initial_formation_idx]
        initial_enemies_y = self.consts.ENEMY_POSITIONS_Y[initial_formation_idx]
        initial_horizontal_dirs = self._initial_enemy_horizontal_directions(
            initial_level, initial_enemies_x, initial_enemies_y
        )

        return_state = PhoenixState(
            player_x=jnp.array(self.consts.PLAYER_POSITION[0], dtype=jnp.int32),
            player_y=jnp.array(self.consts.PLAYER_POSITION[1], dtype=jnp.int32),
            step_counter=jnp.array(0),
            enemies_x = initial_enemies_x,
            enemies_y = initial_enemies_y,
            horizontal_direction_enemies = initial_horizontal_dirs,
            vertical_direction_enemies = jnp.full((8,), 1.0),
            enemy_projectile_x=jnp.full((8,), -1),
            enemy_projectile_y=jnp.full((8,), -1),
            projectile_x=jnp.array(-1),  # Standardwert: kein Projektil
            score = jnp.array(0), # Standardwert: Score=0
            lives=jnp.array(self.consts.PLAYER_LIVES), # Standardwert: 4 Leben
            player_respawn_timer=jnp.array(0),
            level=initial_level,
            level_transition_timer=jnp.array(0),  # Timer for level transition, starts at 0
            phoenix_salvo_gap_timer=jnp.array(0, dtype=jnp.int32),
            phoenix_salvo_cycle_shots=jnp.array(0, dtype=jnp.int32),
            phoenix_salvo_enemy_cooldowns=jnp.full((8,), 0, dtype=jnp.int32),
            phoenix_salvo_enemy_shot_counts=jnp.full((8,), 0, dtype=jnp.int32),
            bat_salvo_gap_timer=jnp.array(0, dtype=jnp.int32),
            bat_salvo_enemy_cooldowns=jnp.full((8,), 0, dtype=jnp.int32),
            bat_salvo_enemy_shot_counts=jnp.full((8,), 0, dtype=jnp.int32),

            invincibility=jnp.array(False),
            invincibility_timer=jnp.array(0),
            ability_cooldown=jnp.array(0),

            bat_wings=jnp.full((8,), 2),
            bat_dying=jnp.full((8,), False, dtype=jnp.bool), # Bat dying status, (8,), bool
            bat_death_timer=jnp.full((8,), 0, dtype=jnp.int32), # Timer for Bat death animation, (8,), int
            bat_wing_regen_timer=jnp.full((8,), 0, dtype=jnp.int32),
            bat_y_cooldown=jnp.full((8,), 0, dtype=jnp.int32),
            bat_midzone_roll_done=jnp.full((8,), False, dtype=jnp.bool_),
            bat_edge_profile_timer=jnp.full((8,), 0, dtype=jnp.int32),
            bat_motion_tick=jnp.full((8,), 0, dtype=jnp.int32),
            bat_dive_phase=jnp.array(0, dtype=jnp.int32),
            bat_dive_timer=jnp.array(self.consts.BAT_DIVE_INTERVAL, dtype=jnp.int32),
            bat_dive_hold_timer=jnp.array(0, dtype=jnp.int32),
            bat_dive_travelled=jnp.array(0.0, dtype=jnp.float32),
            bat_dive_goal=jnp.array(0.0, dtype=jnp.float32),
            bat_anim_phase_offset=jax.random.randint(bat_anim_key, (8,), 0, 7, dtype=jnp.int32),
            phoenix_do_attack = jnp.full((8,), 0, dtype=jnp.bool),  # Phoenix attack state
            phoenix_attack_target_y = jnp.full((8,), -1, dtype=jnp.float32),  # Target Y position for Phoenix attack
            phoenix_original_y = jnp.full((8,), -1, dtype=jnp.float32),  # Original Y position of the Phoenix
            phoenix_cooldown=jnp.full((8,), 0),  # Cooldown für Phoenix-Angriff
            phoenix_drift=jnp.full((8,), 0.0, dtype=jnp.float32),  # Drift-Werte für Phoenix
            phoenix_returning=jnp.full((8,), False, dtype=jnp.bool),  # Returning status of the Phoenix
            phoenix_dying=jnp.full((8,), False, dtype=jnp.bool),  # Dying status of the Phoenix
            phoenix_death_timer=jnp.full((8,), 0, dtype=jnp.int32),  # Timer for Phoenix death animation

            player_dying=jnp.array(False, dtype = jnp.bool),  # Player dying status, bool
            player_death_timer=jnp.array(0, dtype = jnp.int32),  # Timer for player death animation, int
            player_moving=jnp.array(False, dtype = jnp.bool), # Player moving status, bool

            boss=BossState(
                active=jnp.array(True, dtype=jnp.bool_),
                x=jnp.array(self.consts.BOSS_SPAWN_POSITION[0], dtype=jnp.float32),
                y=jnp.array(self.consts.BOSS_SPAWN_POSITION[1], dtype=jnp.float32),
                blue_alive=jnp.ones((2, 20), dtype=jnp.bool_),
                red_alive=jnp.ones((92,), dtype=jnp.bool_),
                green_alive=jnp.ones((30,), dtype=jnp.bool_),
            ),
        )

        initial_obs = self._get_observation(return_state)
        return initial_obs, return_state

    def _bat_wave_enemy_fire(
        self, key: jax.random.PRNGKey, state: PhoenixState, not_attacking: chex.Array
    ) -> NonPhoenixEnemyFire:
        """Levels 3-4 (bats): center-line trigger + top-layer priority + bat-local cooldowns."""
        alive_bats = (state.enemies_x > -1) & (state.enemies_y < self.consts.HEIGHT + 10) & (~state.bat_dying)
        bat_enemy_cooldowns = jnp.maximum(state.bat_salvo_enemy_cooldowns - 1, 0)
        bat_enemy_shot_counts = state.bat_salvo_enemy_shot_counts
        bat_gap_timer = jnp.maximum(state.bat_salvo_gap_timer - 1, 0)

        player_center_x = state.player_x + 2  # Player sprite is 5 px wide in this implementation.
        enemy_left = state.enemies_x
        enemy_right = state.enemies_x + self.consts.ENEMY_WIDTH - 1
        intersects_enemy = (player_center_x >= enemy_left) & (player_center_x <= enemy_right)

        active_bats = alive_bats & not_attacking & (bat_enemy_cooldowns == 0)
        candidate_mask = active_bats & intersects_enemy
        has_candidate = jnp.any(candidate_mask)

        candidate_y = jnp.where(candidate_mask, state.enemies_y, jnp.inf)
        top_layer_y = jnp.min(candidate_y)
        top_layer_mask = candidate_mask & (state.enemies_y == top_layer_y)
        top_layer_count = jnp.sum(top_layer_mask.astype(jnp.float32))
        safe_probs = jnp.where(
            has_candidate,
            top_layer_mask.astype(jnp.float32) / jnp.maximum(top_layer_count, 1.0),
            jnp.full((8,), 1.0 / 8.0, dtype=jnp.float32),
        )
        key_pick, _ = jax.random.split(key)
        picked_idx = jax.random.choice(key_pick, 8, shape=(), p=safe_probs).astype(jnp.int32)
        picked_idx = jnp.where(has_candidate, picked_idx, jnp.array(-1, dtype=jnp.int32))

        available_slots = state.enemy_projectile_y < 0
        has_free_slot = jnp.any(available_slots)
        slot_idx = jnp.argmax(available_slots.astype(jnp.int32)).astype(jnp.int32)
        shooter_idx = jnp.clip(picked_idx, 0, 7)
        can_fire_now = has_candidate & has_free_slot & (bat_gap_timer == 0)

        # Spawn into the first free projectile slot (not enemy index slot) to avoid mid-air teleports.
        fire_slot_mask = (jnp.arange(8) == slot_idx) & can_fire_now
        bat_gap_timer = jnp.where(can_fire_now, self.consts.BAT_SALVO_SHOT_GAP, bat_gap_timer)

        fired_enemy_mask = (jnp.arange(8) == shooter_idx) & can_fire_now
        updated_shot_counts = bat_enemy_shot_counts + fired_enemy_mask.astype(jnp.int32)
        should_long_pause = fired_enemy_mask & (
            (updated_shot_counts % self.consts.BAT_SALVO_LONG_PAUSE_EVERY) == 0
        )
        long_pause_frames = self.consts.BAT_SALVO_SHOT_GAP * self.consts.BAT_SALVO_LONG_PAUSE_MULTIPLIER
        updated_cooldowns = jnp.where(should_long_pause, long_pause_frames, bat_enemy_cooldowns)
        updated_shot_counts = jnp.where(alive_bats, updated_shot_counts, 0)
        updated_cooldowns = jnp.where(alive_bats, updated_cooldowns, 0)

        half = jnp.full((8,), self.consts.ENEMY_WIDTH // 2, dtype=jnp.int32)
        proj_offsets = half
        shooter_x = state.enemies_x[shooter_idx]
        shooter_y = state.enemies_y[shooter_idx]
        eff_enemy_x = jnp.where(fire_slot_mask, shooter_x, state.enemies_x)
        eff_enemy_y = jnp.where(fire_slot_mask, shooter_y, state.enemies_y)
        enemy_fire_mask = fire_slot_mask
        return NonPhoenixEnemyFire(
            enemy_fire_mask,
            proj_offsets,
            eff_enemy_x,
            eff_enemy_y,
            bat_gap_timer.astype(jnp.int32),
            updated_cooldowns.astype(jnp.int32),
            updated_shot_counts.astype(jnp.int32),
        )

    def _boss_enemy_fire(
        self, key: jax.random.PRNGKey, state: PhoenixState, not_attacking: chex.Array
    ) -> NonPhoenixEnemyFire:
        """Boss enemy shooting: sampled interval cadence + empirically fitted spawn-X mixture."""
        cooldown = jnp.maximum(state.bat_salvo_gap_timer - 1, 0)
        available_slots = state.enemy_projectile_y < 0
        has_free_slot = jnp.any(available_slots)
        slot_idx = jnp.argmax(available_slots.astype(jnp.int32)).astype(jnp.int32)

        key_interval_mix, key_interval_value, key_x_mix, key_x_value = jax.random.split(key, 4)
        interval_mix = jax.random.uniform(key_interval_mix, shape=())
        # Empirical cadence from Phoenix boss logs:
        # 80% in [24,30], 20% in [31,36], clamped by game logic elsewhere.
        sampled_interval = jnp.where(
            interval_mix < 0.8,
            jax.random.randint(key_interval_value, (), 24, 31),
            jax.random.randint(key_interval_value, (), 31, 37),
        ).astype(jnp.int32)

        x_mix = jax.random.uniform(key_x_mix, shape=())
        # Empirical spawn-X mixture from observed boss missile starts:
        # 35%: [62,79], 45%: [80,99], 20%: [100,108].
        sampled_spawn_x = jnp.where(
            x_mix < 0.35,
            jax.random.randint(key_x_value, (), 62, 80),
            jnp.where(
                x_mix < 0.80,
                jax.random.randint(key_x_value, (), 80, 100),
                jax.random.randint(key_x_value, (), 100, 109),
            ),
        ).astype(jnp.int32)

        boss_half_width = self.consts.BOSS_HALF_WIDTH
        boss_is_alive = state.enemies_x[0] > -1
        boss_center_x = state.enemies_x[0].astype(jnp.int32)
        sampled_offset = jnp.clip(sampled_spawn_x - boss_center_x, -boss_half_width, boss_half_width)
        proj_offsets = jnp.full((8,), sampled_offset, dtype=jnp.int32)
        eff_enemy_x = jnp.where(boss_is_alive, boss_center_x, state.enemies_x)
        eff_enemy_y = jnp.where(boss_is_alive, state.enemies_y[0], state.enemies_y)

        # Keep cooldown scalar-compatible with bat branch: boss fire gate must be scalar.
        boss_not_attacking = jnp.all(not_attacking)
        can_fire_now = boss_is_alive & has_free_slot & boss_not_attacking & (cooldown == 0)
        enemy_fire_mask = (jnp.arange(8) == slot_idx) & can_fire_now
        cooldown = jnp.where(can_fire_now, sampled_interval, cooldown)
        return NonPhoenixEnemyFire(
            enemy_fire_mask,
            proj_offsets,
            eff_enemy_x,
            eff_enemy_y,
            cooldown.astype(jnp.int32),
            state.bat_salvo_enemy_cooldowns,
            state.bat_salvo_enemy_shot_counts,
        )



    def _phoenix_small_bird_enemy_fire(
        self,
        state: PhoenixState,
        pre_step_phoenix_do_attack: chex.Array,
        step_counter: chex.Array,
    ) -> PhoenixSalvoEnemyFire:
        """Levels 1-2 (small phoenix): edge-intersection trigger + top-layer priority fire."""
        _ = pre_step_phoenix_do_attack  # Kept for interface parity with previous behavior.

        def _active_small_birds(enemy_cooldowns: chex.Array) -> chex.Array:
            return (
                (state.enemies_x > -1)
                & (state.enemies_y < self.consts.HEIGHT + 10)
                & (~state.phoenix_dying)
                & (~state.phoenix_do_attack)
                & (~state.phoenix_returning)
                & (enemy_cooldowns == 0)
            )

        def _pick_trigger_enemy(active_mask: chex.Array) -> Tuple[chex.Array, chex.Array]:
            # Player trigger is the vertical line through the player's center pixel.
            player_center_x = state.player_x + 2  # Sprite width is 5 in this implementation.
            enemy_left = state.enemies_x
            enemy_right = state.enemies_x + self.consts.ENEMY_WIDTH - 1
            intersects_enemy = (player_center_x >= enemy_left) & (player_center_x <= enemy_right)
            candidate_mask = active_mask & intersects_enemy
            has_candidate = jnp.any(candidate_mask)

            candidate_y = jnp.where(candidate_mask, state.enemies_y, jnp.inf)
            top_layer_y = jnp.min(candidate_y)
            top_layer_mask = candidate_mask & (state.enemies_y == top_layer_y)

            top_layer_count = jnp.sum(top_layer_mask.astype(jnp.float32))
            safe_probs = jnp.where(
                has_candidate,
                top_layer_mask.astype(jnp.float32) / jnp.maximum(top_layer_count, 1.0),
                jnp.full((8,), 1.0 / 8.0, dtype=jnp.float32),
            )

            key_pick = jax.random.PRNGKey(step_counter + 1201)
            picked_idx = jax.random.choice(key_pick, 8, shape=(), p=safe_probs).astype(jnp.int32)
            picked_idx = jnp.where(has_candidate, picked_idx, jnp.array(-1, dtype=jnp.int32))
            return picked_idx, has_candidate

        alive_small_birds = (state.enemies_x > -1) & (~state.phoenix_dying)
        enemy_cooldowns = jnp.maximum(state.phoenix_salvo_enemy_cooldowns - 1, 0)
        enemy_shot_counts = state.phoenix_salvo_enemy_shot_counts
        active_small_birds = _active_small_birds(enemy_cooldowns)
        trigger_enemy_idx, has_trigger_enemy = _pick_trigger_enemy(active_small_birds)

        available_slots = state.enemy_projectile_y < 0
        has_free_slot = jnp.any(available_slots)
        slot_idx = jnp.argmax(available_slots.astype(jnp.int32)).astype(jnp.int32)

        gap_timer = jnp.maximum(state.phoenix_salvo_gap_timer - 1, 0)
        can_fire_now = has_trigger_enemy & (gap_timer == 0) & has_free_slot
        new_gap_timer = jnp.where(can_fire_now, self.consts.PHOENIX_SALVO_SHOT_GAP, gap_timer)

        salvo_fire_mask = jnp.zeros((8,), dtype=jnp.bool_)
        salvo_fire_mask = salvo_fire_mask.at[slot_idx].set(can_fire_now)

        shooter_idx = jnp.clip(trigger_enemy_idx, 0, 7)
        spawn_x = state.enemies_x[shooter_idx] + (self.consts.ENEMY_WIDTH // 2)
        spawn_y = state.enemies_y[shooter_idx] + self.consts.ENEMY_HEIGHT
        salvo_x = jnp.where(salvo_fire_mask, spawn_x, state.enemy_projectile_x)
        salvo_y = jnp.where(salvo_fire_mask, spawn_y, state.enemy_projectile_y)

        cycle_shots = jnp.where(
            has_trigger_enemy,
            state.phoenix_salvo_cycle_shots + can_fire_now.astype(jnp.int32),
            jnp.array(0, dtype=jnp.int32),
        )

        fired_enemy_mask = (jnp.arange(8) == shooter_idx) & can_fire_now
        updated_enemy_shot_counts = enemy_shot_counts + fired_enemy_mask.astype(jnp.int32)
        should_long_pause = fired_enemy_mask & (
            (updated_enemy_shot_counts % self.consts.PHOENIX_SALVO_LONG_PAUSE_EVERY) == 0
        )
        long_pause_frames = self.consts.PHOENIX_SALVO_SHOT_GAP * self.consts.PHOENIX_SALVO_LONG_PAUSE_MULTIPLIER
        updated_enemy_cooldowns = jnp.where(should_long_pause, long_pause_frames, enemy_cooldowns)

        # Destroyed/dying enemies do not keep cooldown/counter state.
        updated_enemy_shot_counts = jnp.where(alive_small_birds, updated_enemy_shot_counts, 0)
        updated_enemy_cooldowns = jnp.where(alive_small_birds, updated_enemy_cooldowns, 0)

        return PhoenixSalvoEnemyFire(
            salvo_fire_mask=salvo_fire_mask,
            salvo_x=salvo_x,
            salvo_y=salvo_y,
            gap_timer=new_gap_timer,
            cycle_shots=cycle_shots,
            enemy_cooldowns=updated_enemy_cooldowns.astype(jnp.int32),
            enemy_shot_counts=updated_enemy_shot_counts.astype(jnp.int32),
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action: int) -> Tuple[PhoenixObservation, PhoenixState, float, bool, PhoenixInfo]:
        # Translate agent action index to ALE console action
        atari_action = jnp.take(self.ACTION_SET, jnp.asarray(action, dtype=jnp.int32))
        
        new_respawn_timer = jnp.where(state.player_respawn_timer > 0, state.player_respawn_timer - 1, 0)
        respawn_ended = (state.player_respawn_timer > 0) & (new_respawn_timer == 0)

        state = state.replace(player_respawn_timer=new_respawn_timer.astype(jnp.int32))

        state = jax.lax.cond(
            jnp.logical_or(state.player_dying, state.player_respawn_timer > 0),
            lambda s: s,
            lambda s: self.player_step(s, atari_action),
            state
        ) # Player_step only if not dying

        projectile_active = state.projectile_y >= 0

        # Can fire only if inactive
        can_fire = (~projectile_active) & (~state.player_dying) & (state.player_respawn_timer <= 0)
        fire_actions = jnp.array([
            atari_action == Action.FIRE,
            atari_action == Action.LEFTFIRE,
            atari_action == Action.RIGHTFIRE,
            atari_action == Action.DOWNFIRE,
        ])
        firing = jnp.any(fire_actions) & can_fire

        pre_step_phoenix_do_attack = state.phoenix_do_attack
        state, sub_step_score, sub_step_hit = jax.lax.cond(
            jnp.logical_or((state.level % 5) == 1, (state.level % 5) == 2),
            lambda: self.phoenix_step(state),
            lambda: jax.lax.cond(
                jnp.logical_or((state.level % 5) == 3, (state.level % 5) == 4),
                lambda: self.bat_step(state),
                lambda: self.boss_step(state),
            )
        )

        # Clear projectile when sub-step detected a hit; otherwise spawn on FIRE or move active projectile
        projectile_active = state.projectile_y >= 0
        projectile_x = jnp.where(
            sub_step_hit,
            -1,
            jnp.where(firing, state.player_x + 2, state.projectile_x),
        )
        projectile_y = jnp.where(
            sub_step_hit,
            -1,
            jnp.where(
                firing,
                state.player_y - self.consts.PLAYER_PROJECTILE_INITIAL_OFFSET,
                jnp.where(
                    projectile_active,
                    state.projectile_y - self.consts.PLAYER_PROJECTILE_SPEED,
                    state.projectile_y,
                ),
            ),
        )
        projectile_y = jnp.where(projectile_y < 0, -1, projectile_y)
        projectile_x = projectile_x.astype(jnp.int32)
        projectile_y = projectile_y.astype(jnp.int32)

        projectile_active = projectile_y >= 0
        
        projectile_pos = jnp.array([projectile_x, projectile_y])

        level_mod = state.level % 5
        is_phoenix_level = jnp.logical_or(level_mod == 1, level_mod == 2)
        is_bat_level = jnp.logical_or(level_mod == 3, level_mod == 4)
        is_boss_level = (state.level % 5) == 0

        key = jax.random.PRNGKey(state.step_counter)
        not_attacking = jnp.logical_not(jnp.logical_or(state.phoenix_do_attack, state.phoenix_returning))

        # Bat vs boss: cond so only one RNG path runs (phoenix levels take the bat branch for the unused draw).
        non_phoenix_fire = jax.lax.cond(
            is_boss_level,
            lambda _: self._boss_enemy_fire(key, state, not_attacking),
            lambda _: self._bat_wave_enemy_fire(key, state, not_attacking),
            None,
        )

        phoenix_fire = jax.lax.cond(
            is_phoenix_level,
            lambda _: self._phoenix_small_bird_enemy_fire(
                state, pre_step_phoenix_do_attack, state.step_counter
            ),
            lambda _: PhoenixSalvoEnemyFire(
                salvo_fire_mask=jnp.zeros((8,), dtype=jnp.bool_),
                salvo_x=state.enemy_projectile_x.astype(jnp.float32),
                salvo_y=state.enemy_projectile_y.astype(jnp.float32),
                gap_timer=state.phoenix_salvo_gap_timer,
                cycle_shots=state.phoenix_salvo_cycle_shots,
                enemy_cooldowns=state.phoenix_salvo_enemy_cooldowns,
                enemy_shot_counts=state.phoenix_salvo_enemy_shot_counts,
            ),
            operand=None,
        )

        eff_enemy_x = non_phoenix_fire.eff_enemy_x
        eff_enemy_y = non_phoenix_fire.eff_enemy_y
        proj_offsets = non_phoenix_fire.proj_offsets
        bat_gap_timer = jnp.where(
            jnp.logical_or(is_bat_level, is_boss_level),
            non_phoenix_fire.bat_gap_timer,
            state.bat_salvo_gap_timer,
        )
        bat_enemy_cooldowns = jnp.where(
            is_bat_level, non_phoenix_fire.bat_enemy_cooldowns, state.bat_salvo_enemy_cooldowns
        )
        bat_enemy_shot_counts = jnp.where(
            is_bat_level, non_phoenix_fire.bat_enemy_shot_counts, state.bat_salvo_enemy_shot_counts
        )

        enemy_fire_mask = jnp.where(
            is_phoenix_level, phoenix_fire.salvo_fire_mask, non_phoenix_fire.enemy_fire_mask
        )
        # Suppress all enemy fire during the respawn blink window so the player
        # never faces bullets the moment they become vulnerable.
        enemy_fire_mask = jnp.where(
            state.player_respawn_timer > 0, jnp.zeros((8,), dtype=jnp.bool_), enemy_fire_mask
        )

        gap_timer = phoenix_fire.gap_timer
        cycle_shots = phoenix_fire.cycle_shots
        enemy_cooldowns = phoenix_fire.enemy_cooldowns
        enemy_shot_counts = phoenix_fire.enemy_shot_counts

        # Calculate exact X/Y origins
        enemy_projectile_x = jnp.where(enemy_fire_mask, eff_enemy_x + proj_offsets, state.enemy_projectile_x)

        # Standard enemies fire from bottom (height 5), Boss fires from below the blue layer (offset 21)
        spawn_y_offset = jnp.where(is_boss_level, self.consts.BOSS_PROJECTILE_Y_OFFSET, self.consts.ENEMY_HEIGHT)
        enemy_projectile_y = jnp.where(enemy_fire_mask, eff_enemy_y + spawn_y_offset, state.enemy_projectile_y)
        enemy_projectile_x = jnp.where(is_phoenix_level, phoenix_fire.salvo_x, enemy_projectile_x)
        enemy_projectile_y = jnp.where(is_phoenix_level, phoenix_fire.salvo_y, enemy_projectile_y)

        # Move enemy projectiles downwards
        enemy_projectile_y = jnp.where(state.enemy_projectile_y >= 0, state.enemy_projectile_y + self.consts.ENEMY_PROJECTILE_SPEED,
                                           enemy_projectile_y)

        # Remove enemy projectile if off-screen
        enemy_projectile_y = jnp.where(enemy_projectile_y > self.consts.FLOOR_Y - self.consts.PROJECTILE_HEIGHT, -1, enemy_projectile_y) # TODO 185 durch Konstante ersetzen, die global geändert werden kann.



        projectile_pos = jnp.array([projectile_x, projectile_y])
        enemy_positions = jnp.stack((state.enemies_x, state.enemies_y), axis=1)

        def check_collision(entity_pos, projectile_pos):
            enemy_x, enemy_y = entity_pos
            projectile_x, projectile_y = projectile_pos

            collision_x = (projectile_x + self.consts.PROJECTILE_WIDTH > enemy_x) & (projectile_x < enemy_x + self.consts.ENEMY_WIDTH)
            collision_y = (projectile_y + self.consts.PROJECTILE_HEIGHT > enemy_y) & (projectile_y < enemy_y + self.consts.ENEMY_HEIGHT)
            return collision_x & collision_y


        # Kollisionsprüfung Gegner
        enemy_collisions_raw = jax.vmap(lambda enemy_pos: check_collision(enemy_pos, projectile_pos))(enemy_positions)

        # On boss levels all scoring/kill logic runs in boss_step. Suppress the
        # generic collision on slot 0 entirely to prevent double-awarding.
        enemy_collisions_raw = enemy_collisions_raw.at[0].set(
            jnp.where(is_boss_level, False, enemy_collisions_raw[0])
        )
        is_bat_level = jnp.logical_or((state.level % 5) == 3, (state.level % 5) == 4)
        dying_mask = jnp.where(is_bat_level, state.bat_dying, state.phoenix_dying)
        is_vulnerable = (new_respawn_timer <= 0) & (~state.player_dying) & (~state.invincibility)
        player_body_width = 5
        player_body_height = 8

        active_enemies_for_body = (
            (state.enemies_x > -1)
            & (state.enemies_y < self.consts.HEIGHT + 10)
            & (~dying_mask)
        )
        body_hit_x = (state.enemies_x + self.consts.ENEMY_WIDTH > state.player_x) & (
            state.enemies_x < state.player_x + player_body_width
        )
        body_hit_y = (state.enemies_y + self.consts.ENEMY_HEIGHT > state.player_y) & (
            state.enemies_y < state.player_y + player_body_height
        )
        enemy_body_collisions_raw = (
            is_vulnerable
            & (state.invincibility == jnp.array(False))
            & active_enemies_for_body
            & body_hit_x
            & body_hit_y
        )
        body_hit_y_coords = jnp.where(enemy_body_collisions_raw, state.enemies_y, -jnp.inf)
        body_hit_enemy_idx = jnp.argmax(body_hit_y_coords)
        enemy_body_hit_detected = jnp.any(enemy_body_collisions_raw)
        enemy_body_collision_mask = (jnp.arange(8) == body_hit_enemy_idx) & enemy_body_hit_detected
        
        # Filter to only valid living enemies
        valid_enemy_collisions = enemy_collisions_raw & (~dying_mask)
        
        # Select ONLY the lowest enemy (highest Y) if multiple overlap the fast missile
        hit_y_coords_enemies = jnp.where(valid_enemy_collisions, state.enemies_y, -jnp.inf)
        lowest_enemy_idx = jnp.argmax(hit_y_coords_enemies)
        
        enemy_hit_detected = jnp.any(valid_enemy_collisions)
        
        # Create a new mask where ONLY the lowest hit enemy is True
        enemy_collisions = (jnp.arange(8) == lowest_enemy_idx) & enemy_hit_detected

        # Projectile hit and body-contact hit both kill exactly one enemy.
        enemy_kill_mask = enemy_collisions | enemy_body_collision_mask

        # Phoenix-Death-Animation starten (nur Phoenix-Levels)
        p_hit_mask = enemy_kill_mask & (~is_bat_level)
        new_phoenix_dying = jnp.where(p_hit_mask, True, state.phoenix_dying)
        new_phoenix_death_timer = jnp.where(
            p_hit_mask, self.consts.ENEMY_DEATH_DURATION, state.phoenix_death_timer
        )
        p_dec_timer = jnp.where(
            new_phoenix_dying & (new_phoenix_death_timer > 0),
            new_phoenix_death_timer - 1,
            new_phoenix_death_timer,
        )
        p_death_done = new_phoenix_dying & (p_dec_timer == 0)
        new_phoenix_dying = jnp.where(p_death_done, False, new_phoenix_dying)
        p_dec_timer = jnp.where(p_death_done, 0, p_dec_timer)

        # Bat-Death-Animation starten (nur Bat-Levels)
        b_hit_mask = enemy_kill_mask & is_bat_level
        new_bat_dying = jnp.where(b_hit_mask, True, state.bat_dying)
        new_bat_death_timer = jnp.where(
            b_hit_mask, self.consts.ENEMY_DEATH_DURATION, state.bat_death_timer
        )
        b_dec_timer = jnp.where(
            new_bat_dying & (new_bat_death_timer > 0),
            new_bat_death_timer - 1,
            new_bat_death_timer,
        )
        b_death_done = new_bat_dying & (b_dec_timer == 0)
        new_bat_dying = jnp.where(b_death_done, False, new_bat_dying)
        b_dec_timer = jnp.where(b_death_done, 0, b_dec_timer)

        # Phoenix-Angriffsstatus nur in Phoenix-Levels zurücksetzen
        phoenix_do_attack = jnp.where(p_hit_mask, False, state.phoenix_do_attack)
        phoenix_attack_target_y = jnp.where(p_hit_mask, -1, state.phoenix_attack_target_y)
        phoenix_original_y = jnp.where(p_hit_mask, -1, state.phoenix_original_y)

        # Projektil zurücksetzen bei Treffer
        projectile_x = jnp.where(enemy_hit_detected, -1, projectile_x)
        projectile_y = jnp.where(enemy_hit_detected, -1, projectile_y)

        # --- DYNAMIC SCORING LOGIC ---
        # 1. Small Birds (Levels 1 & 2)
        # 20 points horizontal, 80 points if swooping[cite: 96, 97].
        is_swooping = state.phoenix_do_attack | state.phoenix_returning
        small_bird_scores = jnp.where(is_swooping, 80, 20)

        # 2. Large Birds / Bats (Levels 3 & 4): y-position lookup table (100..500 in 50-point steps).
        bat_score_idx = jnp.searchsorted(self.consts.BAT_SCORE_Y_BOUNDS, state.enemies_y, side="left")
        bat_score_idx = jnp.clip(
            bat_score_idx,
            0,
            self.consts.BAT_SCORE_VALUES.shape[0] - 1,
        )
        large_bird_scores = self.consts.BAT_SCORE_VALUES[bat_score_idx]

        # Identify level type
        level_type = (state.level % 5)
        is_small_bird_level = (level_type == 1) | (level_type == 2)

        # Map the correct score logic to the enemies array based on current level.
        # Boss levels (level_type == 0) are handled entirely in boss_step; slot 0
        # is suppressed in enemy_collisions_raw so this fallback never fires for them.
        enemy_hit_scores = jnp.where(is_small_bird_level, small_bird_scores, large_bird_scores)

        # Mask scores so only hit enemies award points, then sum 
        actual_hit_scores = jnp.where(enemy_collisions, enemy_hit_scores, 0)
        total_hit_score = jnp.sum(actual_hit_scores)

        # Update overall score with sub_step (wings) + main kills
        score = (state.score + sub_step_score + total_hit_score).astype(jnp.int32)

        # Gegner entfernen nach Ablauf der jeweiligen Death-Animation
        death_done_any = jnp.where(is_bat_level, b_death_done, p_death_done)
        enemies_x = jnp.where(death_done_any, -1, state.enemies_x)
        enemies_y = jnp.where(death_done_any, self.consts.HEIGHT + 20, state.enemies_y)


        # Checken ob alle Gegner getroffen wurden
        #all_enemies_hit = jnp.all(enemies_y >= self.consts.HEIGHT + 10)
        #new_level = jnp.where(all_enemies_hit, (state.level % 5) + 1, state.level)
        #new_enemies_x = jax.lax.cond(
        #    all_enemies_hit,
        #    lambda: jax.lax.switch((new_level -1 )% 5, self.consts.ENEMY_POSITIONS_X_LIST).astype(jnp.float32),
        #    lambda: state.enemies_x.astype(jnp.float32)
        #)
        #new_enemies_y = jax.lax.cond(
        #    all_enemies_hit,
        #    lambda: jax.lax.switch((new_level -1 )% 5, self.consts.ENEMY_POSITIONS_Y_LIST).astype(jnp.float32),
        #    lambda: enemies_y.astype(jnp.float32)
        #)
        #enemies_x = new_enemies_x
        #enemies_y = new_enemies_y
        #level = new_level

        # 1) Level-Übergangstimer starten/fortschreiben
        all_enemies_cleared = jnp.all(enemies_y >= self.consts.HEIGHT + 10)
        start_transition = all_enemies_cleared & (state.level_transition_timer == 0)

        new_level_transition_timer = jnp.where(
            start_transition,
            self.consts.LEVEL_TRANSITION_DURATION,
            state.level_transition_timer
        )
        new_level_transition_timer = jnp.where(new_level_transition_timer > 0, new_level_transition_timer - 1, 0)

        transition_ended = (state.level_transition_timer > 0) & (new_level_transition_timer == 0)

        # 2) Nächstes Level vormerken und erst bei Timerende aktivieren
        pending_next_level = (state.level % 5) + 1
        level = jnp.where(transition_ended, pending_next_level, state.level)

        # 3) Gegner-Formationen nur bei Timerende spawnen
        formation_idx = (pending_next_level - 1) % 5
        next_enemies_x = self.consts.ENEMY_POSITIONS_X[formation_idx]
        next_enemies_y = self.consts.ENEMY_POSITIONS_Y[formation_idx]


        reset_mask = transition_ended
        enemies_x = jnp.where(reset_mask, next_enemies_x.astype(jnp.float32), enemies_x)
        enemies_y = jnp.where(reset_mask, next_enemies_y.astype(jnp.float32), enemies_y)

        # Richtungen der Formation zurücksetzen
        reset_horizontal_dirs = self._initial_enemy_horizontal_directions(level, enemies_x, enemies_y)
        new_horizontal_direction_enemies = jnp.where(
            reset_mask, reset_horizontal_dirs, state.horizontal_direction_enemies
        )
        new_vertical_direction_enemies = jnp.where(
            reset_mask, jnp.full((8,), 1.0, dtype=jnp.float32), state.vertical_direction_enemies
        )

        # Death-/Timer-/Flügel-Status zurücksetzen
        new_phoenix_dying = jnp.where(reset_mask, jnp.full((8,), False), new_phoenix_dying)
        p_dec_timer = jnp.where(reset_mask, jnp.full((8,), 0), p_dec_timer)

        new_bat_dying = jnp.where(reset_mask, jnp.full((8,), False), new_bat_dying)
        b_dec_timer = jnp.where(reset_mask, jnp.full((8,), 0), b_dec_timer)
        new_bat_wings = jnp.where(reset_mask, jnp.full((8,), 2, dtype=jnp.int32), state.bat_wings)

        # Boss status only resets when entering a boss level.
        enter_boss_next = ((pending_next_level % 5) == 0)
        reset_blocks = reset_mask & enter_boss_next
        boss = BossState(
            active=jnp.where(reset_blocks, jnp.array(True, dtype=jnp.bool_), state.boss.active),
            x=jnp.where(
                reset_blocks,
                jnp.array(self.consts.BOSS_SPAWN_POSITION[0], dtype=jnp.float32),
                state.boss.x,
            ),
            y=jnp.where(
                reset_blocks,
                jnp.array(self.consts.BOSS_SPAWN_POSITION[1], dtype=jnp.float32),
                state.boss.y,
            ),
            blue_alive=jnp.where(
                reset_blocks,
                jnp.ones((2, 20), dtype=jnp.bool_),
                state.boss.blue_alive,
            ),
            red_alive=jnp.where(
                reset_blocks,
                jnp.ones((92,), dtype=jnp.bool_),
                state.boss.red_alive,
            ),
            green_alive=jnp.where(
                reset_blocks,
                jnp.ones((30,), dtype=jnp.bool_),
                state.boss.green_alive,
            ),
        )

        # Gegner-Respawn nach Spieler-Respawn nur, wenn kein Level-Übergang läuft
        respawn_formation_idx = (level - 1) % 5
        enemy_respawn_x = self.consts.ENEMY_POSITIONS_X[respawn_formation_idx]
        enemy_respawn_y = self.consts.ENEMY_POSITIONS_Y[respawn_formation_idx]

        enemy_respawn_mask = respawn_ended & (new_level_transition_timer == 0)
        enemy_alive_mask = (enemies_x > -1) & (enemies_y < self.consts.HEIGHT + 10)
        enemies_x = jnp.where(enemy_respawn_mask & enemy_alive_mask, enemy_respawn_x, enemies_x)
        enemies_y = jnp.where(enemy_respawn_mask & enemy_alive_mask, enemy_respawn_y, enemies_y)
        packed_respawn_x, packed_respawn_y = self._pack_alive_bats_to_top_slots(level, enemies_x, enemies_y)
        enemies_x = jnp.where(enemy_respawn_mask, packed_respawn_x, enemies_x)
        enemies_y = jnp.where(enemy_respawn_mask, packed_respawn_y, enemies_y)



        def check_player_hit(projectile_xs, projectile_ys, player_x, player_y):
            def is_hit(px, py):
                hit_x = (px + self.consts.PROJECTILE_WIDTH > player_x) & (px < player_x + player_body_width)
                hit_y = (py + self.consts.PROJECTILE_HEIGHT > player_y) & (py < player_y + self.consts.PROJECTILE_HEIGHT)
                return hit_x & hit_y

            hits = jax.vmap(is_hit)(projectile_xs, projectile_ys)
            return jnp.any(hits)



        # Projectile-vs-player collision.
        projectile_player_hit_detected = jnp.where(
            is_vulnerable & (state.invincibility == jnp.array(False)),
            check_player_hit(enemy_projectile_x, enemy_projectile_y, state.player_x, state.player_y),
            False
        )
        # Player dies from projectile OR direct enemy body contact.
        player_hit_detected = projectile_player_hit_detected | enemy_body_hit_detected

        # Bei Treffer: Spieler-Dying-Status setzen und Timer starten
        player_death_duration = self.consts.PLAYER_DEATH_DURATION
        new_player_dying = jnp.where(player_hit_detected, True, state.player_dying)
        player_death_timer_start = jnp.where(player_hit_detected, player_death_duration, state.player_death_timer)

        lives = jnp.where(player_hit_detected, state.lives - 1, state.lives)


        # Remove enemy projectiles only when the projectile itself caused the hit.
        enemy_projectile_x = jnp.where(projectile_player_hit_detected, -1, enemy_projectile_x)
        enemy_projectile_y = jnp.where(projectile_player_hit_detected, -1, enemy_projectile_y)

        # Player-Death-Teimer herunterzählen
        dec_player_timer = jnp.where(
            new_player_dying & (player_death_timer_start > 0),
            player_death_timer_start - 1,
            player_death_timer_start
        )
        player_death_done = new_player_dying & (dec_player_timer == 0) & (player_death_timer_start > 0)

        # Clear all in-flight enemy bullets so they can't kill the player the instant they respawn.
        enemy_projectile_x = jnp.where(player_death_done, jnp.full((8,), -1, dtype=jnp.int32), enemy_projectile_x)
        enemy_projectile_y = jnp.where(player_death_done, jnp.full((8,), -1, dtype=jnp.int32), enemy_projectile_y)

        player_x = jnp.where(player_death_done, self.consts.PLAYER_POSITION[0], state.player_x)
        player_respawn_timer = jnp.where(
            player_death_done,
            self.consts.PLAYER_RESPAWN_DURATION,
            new_respawn_timer
        )

        new_player_moving = jnp.where(
            jnp.logical_or(new_player_dying, player_respawn_timer > 0),
            jnp.array(False, dtype=jnp.bool_),
            state.player_moving
        )

        #enemy_respawn_x = jax.lax.switch((level - 1) % 5, self.consts.ENEMY_POSITIONS_X_LIST).astype(jnp.float32)
        #enemy_respawn_y = jax.lax.switch((level - 1) % 5, self.consts.ENEMY_POSITIONS_Y_LIST).astype(jnp.float32)

        #enemies_x = jnp.where(respawn_ended, enemy_respawn_x, enemies_x)
        #enemies_y = jnp.where(respawn_ended, enemy_respawn_y, enemies_y)

        new_player_dying = jnp.where(player_death_done, False, new_player_dying).astype(jnp.bool_)
        new_player_death_timer = jnp.where(player_death_done, 0, dec_player_timer).astype(jnp.int32)

        formation_reset = transition_ended | (respawn_ended & (new_level_transition_timer == 0))
        bat_anim_key = jax.random.PRNGKey(
            state.step_counter.astype(jnp.uint32) + state.level.astype(jnp.uint32) * jnp.uint32(131)
        )
        reset_bat_anim_phase_offset = jax.random.randint(
            bat_anim_key, (8,), 0, 7, dtype=jnp.int32
        )
        new_bat_midzone_roll_done = jnp.where(
            formation_reset, jnp.full((8,), False, dtype=jnp.bool_), state.bat_midzone_roll_done
        )
        new_bat_edge_profile_timer = jnp.where(
            formation_reset, jnp.full((8,), 0, dtype=jnp.int32), state.bat_edge_profile_timer
        )
        new_bat_motion_tick = jnp.where(
            formation_reset, jnp.full((8,), 0, dtype=jnp.int32), state.bat_motion_tick
        )
        new_bat_dive_phase = jnp.where(formation_reset, 0, state.bat_dive_phase).astype(jnp.int32)
        new_bat_dive_timer = jnp.where(
            formation_reset, self.consts.BAT_DIVE_INTERVAL, state.bat_dive_timer
        ).astype(jnp.int32)
        new_bat_dive_hold_timer = jnp.where(formation_reset, 0, state.bat_dive_hold_timer).astype(jnp.int32)
        new_bat_dive_travelled = jnp.where(formation_reset, 0.0, state.bat_dive_travelled).astype(jnp.float32)
        new_bat_dive_goal = jnp.where(formation_reset, 0.0, state.bat_dive_goal).astype(jnp.float32)
        new_bat_anim_phase_offset = jnp.where(
            formation_reset, reset_bat_anim_phase_offset, state.bat_anim_phase_offset
        ).astype(jnp.int32)
        new_phoenix_do_attack = jnp.where(formation_reset, jnp.full((8,), False), state.phoenix_do_attack)
        new_phoenix_returning = jnp.where(formation_reset, jnp.full((8,), False), state.phoenix_returning)
        new_phoenix_attack_target = jnp.where(formation_reset, jnp.full((8,), -1.0), state.phoenix_attack_target_y)
        new_phoenix_cooldown = jnp.where(formation_reset, jnp.full((8,), 0), state.phoenix_cooldown)
        new_phoenix_drift = jnp.where(formation_reset, jnp.full((8,), 0.0), state.phoenix_drift)
        new_phoenix_original_y = jnp.where(formation_reset, jnp.full((8,), -1.0), state.phoenix_original_y)
        new_salvo_gap_timer = jnp.where(formation_reset, 0, gap_timer)
        new_salvo_cycle_shots = jnp.where(formation_reset, 0, cycle_shots)
        new_salvo_enemy_cooldowns = jnp.where(formation_reset, jnp.full((8,), 0), enemy_cooldowns)
        new_salvo_enemy_shot_counts = jnp.where(formation_reset, jnp.full((8,), 0), enemy_shot_counts)
        new_bat_salvo_gap_timer = jnp.where(formation_reset, 0, bat_gap_timer)
        new_bat_salvo_enemy_cooldowns = jnp.where(formation_reset, jnp.full((8,), 0), bat_enemy_cooldowns)
        new_bat_salvo_enemy_shot_counts = jnp.where(formation_reset, jnp.full((8,), 0), bat_enemy_shot_counts)

        return_state = PhoenixState(
            player_x = player_x,
            player_y = state.player_y,
            step_counter = state.step_counter + 1,
            projectile_x = projectile_x,
            projectile_y = projectile_y,
            enemies_x = enemies_x,
            enemies_y = enemies_y,
            horizontal_direction_enemies = new_horizontal_direction_enemies,
            score=score,
            enemy_projectile_x=enemy_projectile_x.astype(jnp.int32),
            enemy_projectile_y=enemy_projectile_y.astype(jnp.int32),
            lives=lives,
            player_respawn_timer = player_respawn_timer,
            level = level,
            vertical_direction_enemies=new_vertical_direction_enemies,
            boss=boss,
            invincibility=state.invincibility,
            invincibility_timer=state.invincibility_timer,
            bat_wings=new_bat_wings,
            bat_dying=new_bat_dying,
            bat_death_timer=b_dec_timer,
            phoenix_do_attack=new_phoenix_do_attack,
            phoenix_attack_target_y=new_phoenix_attack_target,
            phoenix_original_y=new_phoenix_original_y,
            phoenix_cooldown=new_phoenix_cooldown,
            phoenix_drift=new_phoenix_drift,
            phoenix_returning=new_phoenix_returning,
            phoenix_dying=new_phoenix_dying,
            phoenix_death_timer=p_dec_timer,
            player_dying=new_player_dying,
            player_death_timer=new_player_death_timer,
            player_moving=new_player_moving,
            level_transition_timer=new_level_transition_timer,
            ability_cooldown=state.ability_cooldown,
            bat_wing_regen_timer=state.bat_wing_regen_timer,
            bat_y_cooldown=state.bat_y_cooldown,
            bat_midzone_roll_done=new_bat_midzone_roll_done,
            bat_edge_profile_timer=new_bat_edge_profile_timer,
            bat_motion_tick=new_bat_motion_tick,
            bat_dive_phase=new_bat_dive_phase,
            bat_dive_timer=new_bat_dive_timer,
            bat_dive_hold_timer=new_bat_dive_hold_timer,
            bat_dive_travelled=new_bat_dive_travelled,
            bat_dive_goal=new_bat_dive_goal,
            bat_anim_phase_offset=new_bat_anim_phase_offset,
            phoenix_salvo_gap_timer=new_salvo_gap_timer.astype(jnp.int32),
            phoenix_salvo_cycle_shots=new_salvo_cycle_shots.astype(jnp.int32),
            phoenix_salvo_enemy_cooldowns=new_salvo_enemy_cooldowns.astype(jnp.int32),
            phoenix_salvo_enemy_shot_counts=new_salvo_enemy_shot_counts.astype(jnp.int32),
            bat_salvo_gap_timer=new_bat_salvo_gap_timer.astype(jnp.int32),
            bat_salvo_enemy_cooldowns=new_bat_salvo_enemy_cooldowns.astype(jnp.int32),
            bat_salvo_enemy_shot_counts=new_bat_salvo_enemy_shot_counts.astype(jnp.int32),

        )
        observation = self._get_observation(return_state)
        env_reward = self._get_reward(state, return_state)
        done = self._get_done(return_state)
        info = self._get_info(return_state)
        return observation, return_state, env_reward, done, info

    def render(self, state:PhoenixState) -> jnp.ndarray:
        return self.renderer.render(state)

from jaxatari.renderers import JAXGameRenderer

class PhoenixRenderer(JAXGameRenderer):
    @staticmethod
    def _overlay_rgba(base: np.ndarray, top: np.ndarray) -> np.ndarray:
        out = base.copy()
        top_alpha = top[:, :, 3] > 0
        out[top_alpha] = top[top_alpha]
        return out

    @staticmethod
    def _stamp_rgba(canvas: np.ndarray, sprite: np.ndarray, x: int, y: int) -> np.ndarray:
        out = canvas.copy()
        ch, cw, _ = out.shape
        sh, sw, _ = sprite.shape
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(cw, x + sw)
        y1 = min(ch, y + sh)
        if x0 >= x1 or y0 >= y1:
            return out
        sx0 = x0 - x
        sy0 = y0 - y
        sx1 = sx0 + (x1 - x0)
        sy1 = sy0 + (y1 - y0)
        src = sprite[sy0:sy1, sx0:sx1]
        dst = out[y0:y1, x0:x1]
        alpha = src[:, :, 3] > 0
        dst[alpha] = src[alpha]
        out[y0:y1, x0:x1] = dst
        return out

    def _build_legacy_normal_both_wings(self, sprite_path: str, color: str):
        body_path = os.path.join(sprite_path, f"enemy_bats/bats_{color}/bat_{color}_main.npy")
        left_path = os.path.join(
            sprite_path, f"enemy_bats/bats_{color}/bat_{color}_left_wing_middle.npy"
        )
        right_path = os.path.join(
            sprite_path, f"enemy_bats/bats_{color}/bat_{color}_right_wing_middle.npy"
        )
        if not (os.path.exists(body_path) and os.path.exists(left_path) and os.path.exists(right_path)):
            return None

        body = np.load(body_path)
        left = np.load(left_path)
        right = np.load(right_path)

        canvas_h = int(max(body.shape[0], 2 + left.shape[0], 2 + right.shape[0]))
        canvas_w = int(left.shape[1] + body.shape[1] + right.shape[1] - 2)
        canvas = np.zeros((canvas_h, canvas_w, 4), dtype=body.dtype)
        body_x = left.shape[1] - 1
        body_y = 0
        left_x = body_x - left.shape[1] + 1
        right_x = body_x + body.shape[1] - 1
        wing_y = body_y + 2

        canvas = self._stamp_rgba(canvas, body, body_x, body_y)
        canvas = self._stamp_rgba(canvas, left, left_x, wing_y)
        canvas = self._stamp_rgba(canvas, right, right_x, wing_y)
        return canvas

    def _build_wings_from_enemy_animation(self, sprite_path: str, color: str):
        anim_dir = os.path.join(sprite_path, "enemy_animation")
        phase_names = ("normal", "up", "down", "almost_down")
        left_files = [
            os.path.join(anim_dir, f"{color}_left_wing_{phase}.npy") for phase in phase_names
        ]
        if not all(os.path.exists(p) for p in left_files):
            return None

        data = []
        for lf in left_files:
            left = np.load(lf)
            right = np.flip(left, axis=1).copy()
            data.extend([left, right])
        return data

    def _build_composite_bat_animation(self, sprite_path: str, color: str):
        anim_dir = os.path.join(sprite_path, "enemy_animation")
        phase_names = ("normal", "up", "down", "almost_down")

        left_only = {}
        for phase in phase_names:
            p = os.path.join(anim_dir, f"{color}_left_wing_{phase}.npy")
            if not os.path.exists(p):
                return None
            left_only[phase] = np.load(p)
        right_only = {phase: np.flip(left_only[phase], axis=1).copy() for phase in phase_names}

        both = {}
        for phase in ("up", "down", "almost_down"):
            p = os.path.join(anim_dir, f"{color}_wing_{phase}.npy")
            if not os.path.exists(p):
                return None
            both[phase] = np.load(p)
        legacy_normal = self._build_legacy_normal_both_wings(sprite_path, color)
        both["normal"] = (
            legacy_normal
            if legacy_normal is not None
            else self._overlay_rgba(left_only["normal"], right_only["normal"])
        )

        body_main_path = os.path.join(
            sprite_path, f"enemy_bats/bats_{color}/bat_{color}_main.npy"
        )
        if not os.path.exists(body_main_path):
            return None
        body_main = np.load(body_main_path)
        no_wings = {phase: body_main for phase in phase_names}

        # frame index = wing_state_idx * 4 + phase_idx
        # wing_state_idx: 0=both, 1=left_only, 2=right_only, 3=no_wings
        frames = []
        for wing_state_map in (both, left_only, right_only, no_wings):
            for phase in phase_names:
                frames.append(wing_state_map[phase])
        return frames

    def __init__(self, consts: PhoenixConstants = None, config: render_utils.RendererConfig = None):
        self.consts = consts or PhoenixConstants()
        super().__init__(self.consts)
        
        # Use injected config if provided, else default
        if config is None:
            self.config = render_utils.RendererConfig(
                game_dimensions=(210, 160),
                channels=3,
                downscale=None
            )
        else:
            self.config = config
        self.jr = render_utils.JaxRenderingUtils(self.config)
        
        # 2. Define sprite path
        sprite_path = os.path.join(render_utils.get_base_sprite_dir(), "phoenix")
        
        # 3. Use asset config from constants
        final_asset_config = list(self.consts.ASSET_CONFIG)
        blue_composite_bat = self._build_composite_bat_animation(sprite_path, "blue")
        red_composite_bat = self._build_composite_bat_animation(sprite_path, "red")

        def make_block(w, h, color):
            block = np.zeros((h, w, 4), dtype=np.uint8)
            block[:, :, 0] = color[0]
            block[:, :, 1] = color[1]
            block[:, :, 2] = color[2]
            block[:, :, 3] = 255
            return block

        blue_block = make_block(
            self.consts.BOSS_BLUE_BLOCK_WIDTH,
            self.consts.BOSS_BLUE_BLOCK_HEIGHT,
            self.consts.BOSS_BLUE_COLOR,
        )
        red_block = make_block(
            self.consts.BOSS_RED_BLOCK_WIDTH,
            self.consts.BOSS_RED_BLOCK_HEIGHT,
            self.consts.BOSS_RED_COLOR,
        )
        green_block = make_block(
            self.consts.BOSS_GREEN_BLOCK_WIDTH,
            self.consts.BOSS_GREEN_BLOCK_HEIGHT,
            self.consts.BOSS_GREEN_COLOR,
        )

        patched_config = []
        for asset in final_asset_config:
            if asset["name"] in ["boss_block_red", "boss_block_blue", "boss_block_green"]:
                continue
            patched_config.append(asset)
        if blue_composite_bat is not None:
            patched_config.append(
                {"name": "bat_blue_composite_anim", "type": "group", "data": blue_composite_bat}
            )
        if red_composite_bat is not None:
            patched_config.append(
                {"name": "bat_red_composite_anim", "type": "group", "data": red_composite_bat}
            )
        patched_config.extend(
            [
                {"name": "boss_block_blue", "type": "group", "data": [blue_block]},
                {"name": "boss_block_red", "type": "group", "data": [red_block]},
                {"name": "boss_block_green", "type": "group", "data": [green_block]},
            ]
        )
        final_asset_config = patched_config
        
        # 4. Load all assets, create palette, and generate ID masks in one call
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(final_asset_config, sprite_path)
        tid = int(self.jr.TRANSPARENT_ID)

    @partial(jax.jit, static_argnums=(0,))
    def _draw_rect_outline(self, raster, x, y, w, h, color_id):
        x0 = x.astype(jnp.int32)
        y0 = y.astype(jnp.int32)
        x1 = (x + w - 1).astype(jnp.int32)
        y1 = (y + h - 1).astype(jnp.int32)
        rw = jnp.int32(raster.shape[1] - 1)
        rh = jnp.int32(raster.shape[0] - 1)
        x0 = jnp.clip(x0, 0, rw)
        y0 = jnp.clip(y0, 0, rh)
        x1 = jnp.clip(x1, 0, rw)
        y1 = jnp.clip(y1, 0, rh)

        xx = self.jr._xx
        yy = self.jr._yy
        horiz = (xx >= x0) & (xx <= x1) & ((yy == y0) | (yy == y1))
        vert = (yy >= y0) & (yy <= y1) & ((xx == x0) | (xx == x1))
        outline = horiz | vert
        cid = jnp.asarray(color_id, dtype=raster.dtype)
        return jnp.where(outline, cid, raster)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        # Start with the background raster
        raster = self.jr.create_object_raster(self.BACKGROUND)

        # Render common elements
        raster = self._render_common(state, raster)

        # Single switch for level-specific renderers (avoids 4 sequential conds)
        level_idx = (state.level - 1) % 5
        raster = jax.lax.switch(
            level_idx,
            [
                lambda r: self._render_phoenix_level(state, r, False),
                lambda r: self._render_phoenix_level(state, r, True),
                lambda r: self._render_bat_level(state, r, True),
                lambda r: self._render_bat_level(state, r, False),
                lambda r: self._render_boss_level(state, r),
            ],
            raster,
        )

        # UI on top
        raster = self._render_ui(state, raster)

        # Final palette lookup
        return self.jr.render_from_palette(raster, self.PALETTE)

    @partial(jax.jit, static_argnums=(0,))
    def _render_common(self, state, raster):
        raster = self.jr.render_at(raster, 0, self.consts.FLOOR_Y, self.SHAPE_MASKS['floor'])

        player_death_sprite_duration = self.consts.PLAYER_DEATH_DURATION // 3
        death_idx = jax.lax.select(
            state.player_death_timer >= 2 * player_death_sprite_duration,
            1,
            jax.lax.select(state.player_death_timer >= player_death_sprite_duration, 2, 3)
        )
        anim_toggle = (((state.step_counter // self.consts.PLAYER_ANIMATION_SPEED) % 2) == 0)
        alive_idx = jax.lax.select(
            state.invincibility,
            4,
            jax.lax.select(state.player_moving & anim_toggle, 4, 0)
        )
        player_frame_index = jax.lax.select(state.player_dying, death_idx, alive_idx)
        player_mask = self.SHAPE_MASKS["player"][player_frame_index]
        player_flip_offset = self.FLIP_OFFSETS["player"]

        def draw_player(r):
            return self.jr.render_at(r, state.player_x, state.player_y, player_mask, flip_offset=player_flip_offset)

        raster = jax.lax.cond(
            jnp.logical_or(state.player_dying, state.player_respawn_timer <= 0),
            draw_player, lambda r: r, raster
        )

        # Player projectile: don't render if it's inside the player sprite
        projectile_mask = self.SHAPE_MASKS["player_projectile"]
        proj_h, proj_w = projectile_mask.shape
        player_mask_local = self.SHAPE_MASKS["player"][player_frame_index]
        ph, pw = player_mask_local.shape

        overlap_x = (state.projectile_x + proj_w > state.player_x) & (state.projectile_x < state.player_x + pw)
        overlap_y = (state.projectile_y + proj_h > state.player_y) & (state.projectile_y < state.player_y + ph)
        projectile_inside_player = overlap_x & overlap_y

        def render_player_projectile(r):
            return self.jr.render_at(r, state.projectile_x, state.projectile_y, projectile_mask)

        raster = jax.lax.cond(
            (state.projectile_x > -1) & (~projectile_inside_player),
            render_player_projectile,
            lambda r: r,
            raster,
        )

        def render_ability(r):
            ability_mask = self.SHAPE_MASKS['player_ability']
            player_mask_local = self.SHAPE_MASKS["player"][player_frame_index]
            ah, aw = ability_mask.shape
            ph, pw = player_mask_local.shape
            ax = state.player_x + (pw - aw) // 2
            ay = state.player_y + (ph - ah) // 2
            return self.jr.render_at(r, ax, ay, ability_mask)

        ability_visible = state.invincibility & ((state.step_counter % 4) == 0)
        raster = jax.lax.cond(ability_visible, render_ability, lambda r: r, raster)

        def render_enemy_projectile(i, current_raster):
            x, y = state.enemy_projectile_x[i], state.enemy_projectile_y[i]
            return jax.lax.cond(
                y > -1,
                lambda r: self.jr.render_at(r, x, y, self.SHAPE_MASKS['enemy_projectile']),
                lambda r: r,
                current_raster
            )

        # Boss level handles projectile layering separately (above blocks), so skip here.
        is_boss_level = (state.level % 5) == 0
        raster = jax.lax.cond(
            is_boss_level,
            lambda r: r,
            lambda r: jax.lax.fori_loop(0, state.enemy_projectile_x.shape[0], render_enemy_projectile, r),
            raster,
        )
        return raster

    @partial(jax.jit, static_argnums=(0, 3))
    def _render_phoenix_level(self, state, raster, is_level_two: bool):
        tol = 0.5
        going_down = state.phoenix_do_attack & (state.enemies_y < state.phoenix_attack_target_y - tol)
        going_up = state.phoenix_do_attack & (state.enemies_y > state.phoenix_attack_target_y + tol)
        returning_moving = state.phoenix_returning & (jnp.abs(state.enemies_y - state.phoenix_original_y) > tol)
        is_moving_vert = going_down | going_up | returning_moving

        phoenix_death_flags = state.phoenix_dying
        phoenix_death_phase = (state.phoenix_death_timer <= self.consts.ENEMY_DEATH_DURATION // 2).astype(jnp.int32)
        anim_toggle = ((state.step_counter // self.consts.ENEMY_ANIMATION_SPEED) % 2) == 0
        phoenix_flip_offset = self.FLIP_OFFSETS['phoenix']
        green_enemy_mask = self.SHAPE_MASKS['green_enemy']

        def render_single_phoenix(i, current_raster):
            x, y = state.enemies_x[i], state.enemies_y[i]
            is_active = (x > -1) & (y < self.consts.HEIGHT + 10)

            def draw_enemy(r):
                death_idx = jax.lax.select(phoenix_death_phase[i] == 0, 3, 4)
                alive_idx = jax.lax.select(is_moving_vert[i], 2, jax.lax.select(anim_toggle, 0, 1))
                frame_idx = jax.lax.select(phoenix_death_flags[i], death_idx, alive_idx)
                phoenix_mask = self.SHAPE_MASKS['phoenix'][frame_idx]
                use_green_enemy = is_level_two & (~phoenix_death_flags[i]) & (~is_moving_vert[i])
                return jax.lax.cond(
                    use_green_enemy,
                    lambda r_in: self.jr.render_at(r_in, x, y, green_enemy_mask),
                    lambda r_in: self.jr.render_at(r_in, x, y, phoenix_mask, flip_offset=phoenix_flip_offset),
                    r,
                )

            return jax.lax.cond(is_active, draw_enemy, lambda r: r, current_raster)

        return jax.lax.fori_loop(0, state.enemies_x.shape[0], render_single_phoenix, raster)

    @partial(jax.jit, static_argnums=(0, 3))
    def _render_bat_level(self, state, raster, is_blue_level: bool):
        bat_death_seg = jnp.maximum(1, self.consts.ENEMY_DEATH_DURATION // 3)
        body_masks = self.SHAPE_MASKS['bat_blue_body'] if is_blue_level else self.SHAPE_MASKS['bat_red_body']
        body_offsets = self.FLIP_OFFSETS['bat_blue_body'] if is_blue_level else self.FLIP_OFFSETS['bat_red_body']
        composite_name = "bat_blue_composite_anim" if is_blue_level else "bat_red_composite_anim"
        has_composite = composite_name in self.SHAPE_MASKS
        composite_masks = self.SHAPE_MASKS.get(composite_name, None)
        composite_offsets = self.FLIP_OFFSETS.get(composite_name, jnp.array([0, 0], dtype=jnp.int32))
        wing_masks = self.SHAPE_MASKS['bat_blue_wings'] if is_blue_level else self.SHAPE_MASKS['bat_red_wings']
        wing_offsets = self.FLIP_OFFSETS['bat_blue_wings'] if is_blue_level else self.FLIP_OFFSETS['bat_red_wings']
        # 7-phase cycle, each phase lasts 8 frames:
        # middle -> down_2 -> down -> down_2 -> middle -> up -> middle
        phase_to_wing_variant = jnp.array([0, 3, 2, 3, 0, 1, 0], dtype=jnp.int32)
        global_anim_phase = ((state.step_counter // 8) % 7).astype(jnp.int32)
        # Per-variant placement tuning to align wing anchors with body:
        # variant index: 0=middle, 1=up, 2=down, 3=down_2
        left_x_shift_by_variant = jnp.array([0, 0, 0, 0], dtype=jnp.int32)
        right_x_shift_by_variant = jnp.array([0, 0, 0, 0], dtype=jnp.int32)
        y_shift_by_variant = jnp.array([0, 0, 0, 0], dtype=jnp.int32)

        def render_single_bat(i, current_raster):
            x = state.enemies_x[i].astype(jnp.int32)
            y = state.enemies_y[i].astype(jnp.int32)
            is_active = (x > -1) & (y < self.consts.HEIGHT + 10)
            is_dying = state.bat_dying[i]
            bat_phase = (global_anim_phase + state.bat_anim_phase_offset[i]) % 7
            wing_variant = phase_to_wing_variant[bat_phase]
            left_wing_mask = wing_masks[2 * wing_variant]
            right_wing_mask = wing_masks[2 * wing_variant + 1]

            def draw_one(rr):
                def draw_death(r):
                    death_timer = state.bat_death_timer[i].astype(jnp.int32)
                    death_idx = jax.lax.select(
                        death_timer > 2 * bat_death_seg, 1,
                        jax.lax.select(death_timer > bat_death_seg, 2, 3)
                    )
                    death_mask = body_masks[death_idx]
                    bh, bw = body_masks[0].shape
                    dh, dw = death_mask.shape
                    ox = x + (bw - dw) // 2 - 5
                    oy = y + (bh - dh) // 2
                    return self.jr.render_at(r, ox, oy, death_mask, flip_offset=body_offsets)

                def draw_alive(r):
                    def draw_composite(r2):
                        wing_state = state.bat_wings[i].astype(jnp.int32)
                        wing_state_idx = jnp.select(
                            [wing_state == 2, wing_state == -1, wing_state == 1],
                            [0, 1, 2],
                            default=3,
                        ).astype(jnp.int32)
                        frame_idx = (wing_state_idx * 4 + wing_variant).astype(jnp.int32)
                        comp_mask = composite_masks[frame_idx]
                        comp_x = x - self.consts.WING_WIDTH
                        comp_y = y
                        return self.jr.render_at(
                            r2, comp_x, comp_y, comp_mask, flip_offset=composite_offsets
                        )

                    def draw_legacy(r2):
                        r_new = self.jr.render_at(r2, x, y, body_masks[0], flip_offset=body_offsets)
                        wing_state = state.bat_wings[i].astype(jnp.int32)
                        draw_left = (wing_state == 2) | (wing_state == -1)
                        draw_right = (wing_state == 2) | (wing_state == 1)
                        x_left = x - self.consts.WING_WIDTH + left_x_shift_by_variant[wing_variant]
                        x_right = x + self.consts.ENEMY_WIDTH - 1 + right_x_shift_by_variant[wing_variant]
                        y_wings = y + 2 + y_shift_by_variant[wing_variant]
                        r_new = jax.lax.cond(
                            draw_left,
                            lambda r3: self.jr.render_at(
                                r3, x_left, y_wings, left_wing_mask, flip_offset=wing_offsets
                            ),
                            lambda r3: r3,
                            r_new,
                        )
                        r_new = jax.lax.cond(
                            draw_right,
                            lambda r3: self.jr.render_at(
                                r3, x_right, y_wings, right_wing_mask, flip_offset=wing_offsets
                            ),
                            lambda r3: r3,
                            r_new,
                        )
                        return r_new

                    # One-wing composite frames can drift relative to the body anchor.
                    # Keep composite only for full-wing bats; use legacy anchored body+wing
                    # rendering for damaged states so sprite and hitbox stay aligned.
                    if has_composite:
                        return jax.lax.cond(state.bat_wings[i] == 2, draw_composite, draw_legacy, r)
                    else:
                        return draw_legacy(r)

                return jax.lax.cond(is_dying, draw_death, draw_alive, rr)

            return jax.lax.cond(is_active, draw_one, lambda rr: rr, current_raster)

        return jax.lax.fori_loop(0, state.enemies_x.shape[0], render_single_bat, raster)

    @partial(jax.jit, static_argnums=(0,))
    def _render_boss_level(self, state, raster):
        c = self.consts
        boss = state.boss

        boss_mask = self.SHAPE_MASKS["boss"]
        boss_flip_offset = self.FLIP_OFFSETS["boss"]
        core_x = boss.x - (c.BOSS_CORE_WIDTH / 2.0)
        core_y = boss.y + c.BOSS_CORE_Y_OFFSET
        raster = jax.lax.cond(
            boss.active,
            lambda r: self.jr.render_at(
                r, core_x.astype(jnp.int32), core_y.astype(jnp.int32), boss_mask, flip_offset=boss_flip_offset
            ),
            lambda r: r,
            raster,
        )

        # Per-pixel direct lookup: 3 fully-vectorized O(H×W) passes, no loop over blocks.
        # For each raster pixel we compute which block it belongs to via integer arithmetic
        # and look up alive status — avoiding the 162-iteration sequential fori_loop.
        bx = jnp.round(boss.x).astype(jnp.int32)
        by = jnp.round(boss.y).astype(jnp.int32)
        rel_x = self.jr._xx - bx  # (H, W) pixel coords relative to boss
        rel_y = self.jr._yy - by

        # --- Blue blocks: 2 rows × 20 cols, 4×2 px each ---
        blue_mask = self.SHAPE_MASKS["boss_block_blue"][0]
        bh_b, bw_b = int(blue_mask.shape[0]), int(blue_mask.shape[1])  # (2, 4)
        n_cols_b, n_rows_b = 20, 2
        blue_x0 = c.BOSS_BLUE_X0                          # Python int = -40
        blue_col = ((rel_x - blue_x0) // bw_b).clip(0, n_cols_b - 1).astype(jnp.int32)
        blue_row = (rel_y // bh_b).clip(0, n_rows_b - 1).astype(jnp.int32)
        blue_alive_px = boss.blue_alive[blue_row, blue_col]
        in_blue = (
            (rel_x >= blue_x0) & (rel_x < -blue_x0) &
            (rel_y >= 0) & (rel_y < n_rows_b * bh_b) &
            blue_alive_px & boss.active
        )
        raster = jnp.where(in_blue, blue_mask[0, 0], raster)

        # --- Red blocks: pyramid (rows [20,18,16,14,12,8,4]), 4×3 px each ---
        red_mask = self.SHAPE_MASKS["boss_block_red"][0]
        bh_r, bw_r = int(red_mask.shape[0]), int(red_mask.shape[1])  # (3, 4)
        dy_r = c.BOSS_RED_DY0                              # Python int = 4
        n_per_row_r = jnp.array([20, 18, 16, 14, 12, 8, 4], dtype=jnp.int32)
        row_offsets_r = jnp.array([0, 20, 38, 54, 68, 80, 88], dtype=jnp.int32)
        red_row = ((rel_y - dy_r) // bh_r).clip(0, 6).astype(jnp.int32)
        nk_r = n_per_row_r[red_row]                       # (H, W)
        half_r = nk_r * (bw_r // 2)                       # = nk_r * 2, half-span in px
        red_col = ((rel_x + half_r) // bw_r).astype(jnp.int32)
        red_col = jnp.minimum(jnp.maximum(red_col, 0), nk_r - 1)
        flat_r = (row_offsets_r[red_row] + red_col).clip(0, 91)
        in_red = (
            (rel_y >= dy_r) & (rel_y < dy_r + 7 * bh_r) &
            (rel_x >= -half_r) & (rel_x < half_r) &
            boss.red_alive[flat_r] & boss.active
        )
        raster = jnp.where(in_red, red_mask[0, 0], raster)

        # --- Green blocks: 5 rows above boss, 4×3 px, center gap at rel_x in [-4,4) ---
        green_mask = self.SHAPE_MASKS["boss_block_green"][0]
        bh_g, bw_g = int(green_mask.shape[0]), int(green_mask.shape[1])  # (3, 4)
        dy_g = c.BOSS_GREEN_DY0                            # Python int = -3
        n_per_row_g = jnp.array([12, 10, 8, 6, 4], dtype=jnp.int32)
        row_offsets_g = jnp.array([0, 10, 18, 24, 28], dtype=jnp.int32)
        # Row 0: dy=-3, row 1: dy=-6, … pixel y in [dy_g - 4*bh_g, dy_g + bh_g) = [-15, 0)
        green_row = ((-rel_y - 1) // bh_g).clip(0, 4).astype(jnp.int32)
        nk_g = n_per_row_g[green_row]                     # (H, W)
        half_g = nk_g * (bw_g // 2)                       # = nk_g * 2
        col_full = ((rel_x + half_g) // bw_g).astype(jnp.int32)
        col_full = jnp.minimum(jnp.maximum(col_full, 0), nk_g - 1)
        center_col1 = nk_g // 2 - 1
        # Skip center 2 columns (always at rel_x in [-bw_g, bw_g))
        alive_col_g = jnp.where(col_full <= center_col1, col_full, col_full - 2).clip(0, 29)
        flat_g = (row_offsets_g[green_row] + alive_col_g).clip(0, 29)
        in_green = (
            (rel_y >= dy_g - 4 * bh_g) & (rel_y < 0) &
            (rel_x >= -half_g) & (rel_x < half_g) &
            ~((rel_x >= -bw_g) & (rel_x < bw_g)) &        # exclude center gap
            boss.green_alive[flat_g] & boss.active
        )
        raster = jnp.where(in_green, green_mask[0, 0], raster)

        # Boss-level layering: enemy missiles should stay visible above block rectangles.
        def render_enemy_projectile(i, current_raster):
            x, y = state.enemy_projectile_x[i], state.enemy_projectile_y[i]
            min_visible_y = (
                state.enemies_y[0].astype(jnp.int32)
                + jnp.int32(self.consts.BOSS_PROJECTILE_Y_OFFSET)
                + jnp.int32(self.consts.BOSS_PROJECTILE_RENDER_DELAY_PX)
            )
            visible = (y > -1) & (y >= min_visible_y)
            return jax.lax.cond(
                visible,
                lambda r: self.jr.render_at_clipped(r, x, y, self.SHAPE_MASKS["enemy_projectile"]),
                lambda r: r,
                current_raster,
            )

        raster = jax.lax.fori_loop(
            0, state.enemy_projectile_x.shape[0], render_enemy_projectile, raster
        )

        return raster

    @partial(jax.jit, static_argnums=(0,))
    def _render_ui(self, state, raster):
        max_digits = 5
        spacing = 8
        # HUD placement: score centered in WIDTH with optional pixel nudge.
        score_dx, score_dy = 3, -5
        score_y = 10 + score_dy
        digit_masks = self.SHAPE_MASKS['digits']
        digit_w = digit_masks[0].shape[1]
        score_digits = self.jr.int_to_digits(state.score, max_digits=max_digits)
        has_nonzero = jnp.any(score_digits != 0)
        first_idx = jnp.where(has_nonzero, jnp.argmax(score_digits != 0), max_digits - 1)
        num_to_render = jnp.where(has_nonzero, max_digits - first_idx, 1)
        start_index = first_idx
        field_total_w = max_digits * spacing
        base_left = (self.consts.WIDTH - field_total_w) // 2
        score_x = base_left + first_idx * spacing + score_dx
        raster = self.jr.render_label_selective(
            raster, score_x, score_y,
            score_digits, digit_masks,
            start_index, num_to_render,
            spacing=spacing, max_digits_to_render=max_digits
        )
        life_mask = self.SHAPE_MASKS['life_indicator']
        life_w = life_mask.shape[1]
        life_spacing = 4
        lives_dx, lives_dy = 3, -7
        lives_y = 20 + lives_dy
        lives_count = jnp.clip(state.lives.astype(jnp.int32), 0, 9)
        score_right_edge = base_left + (max_digits - 1) * spacing + digit_w
        total_lives_width = jnp.where(lives_count > 0, (lives_count - 1) * life_spacing + life_w, 0)
        lives_x = score_right_edge - total_lives_width + lives_dx
        raster = self.jr.render_indicator(
            raster, lives_x, lives_y,
            lives_count, life_mask,
            spacing=life_spacing, max_value=9
        )
        return raster