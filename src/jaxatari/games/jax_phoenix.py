import os
from functools import partial
from typing import Tuple, NamedTuple
import jax
import jax.numpy as jnp
import chex
import jaxatari.spaces as spaces
import jaxatari.rendering.jax_rendering_utils as render_utils
import numpy as np
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.spaces import Space


# Phoenix Game by: Florian Schmidt, Finn Keller

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

# new Constant class
class PhoenixConstants(NamedTuple):
    """Game constants for Phoenix."""
    PLAYER_POSITION: Tuple[int, int] = (76, 175)
    PLAYER_COLOR: Tuple[int, int, int] = (213, 130, 74)
    WIDTH: int = 160
    HEIGHT: int = 210
    WINDOW_WIDTH: int = 160 * 3
    WINDOW_HEIGHT: int = 210 * 3
    MAX_PLAYER: int = 1
    MAX_PLAYER_PROJECTILE: int = 1
    MAX_PHOENIX: int = 8
    MAX_BATS: int = 7
    MAX_BOSS: int = 1
    MAX_BOSS_BLOCK_GREEN: int = 2
    MAX_BOSS_BLOCK_BLUE: int = 24
    MAX_BOSS_BLOCK_RED: int = 104
    PROJECTILE_WIDTH: int = 2
    PROJECTILE_HEIGHT: int = 4
    ENEMY_WIDTH: int = 6
    ENEMY_HEIGHT:int = 5
    WING_WIDTH: int = 5
    BAT_REGEN: int = 250
    BLOCK_WIDTH:int = 4
    BLOCK_HEIGHT:int = 4
    SCORE_COLOR: Tuple[int, int, int] = (210, 210, 64)
    PLAYER_BOUNDS: Tuple[int, int] = (0, 155)  # (left, right)
    ENEMY_DEATH_DURATION: int = 30 # ca. 0,5 Sekunden bei 60 FPS
    PLAYER_DEATH_DURATION: int = 90 # ca. 1,5 Sekunden bei 60 FPS
    ENEMY_PROJECTILE_SPEED: int = 2
    PLAYER_RESPAWN_DURATION: int = 360 # ca. 6 Sekunden bei 60 FPS
    ABILITY_COOLDOWN: int = 600 # ca. 10 sekunden bei 60FPS
    FIRE_CHANCE: float = 0.005
    LEVEL_TRANSITION_DURATION: int = 240 # ca. 4 Sekunden bei 60 FPS
    ENEMY_ANIMATION_SPEED: int = 30  # ca. 0,5 Sekunden bei 60 FPS
    PLAYER_ANIMATION_SPEED: int = 6  # ca. 0,1 Sekunden bei 60 FPS
    PLAYER_LIVES: int = 4 # Anzahl der Leben
    ENEMY_POSITIONS_X_LIST = [
        lambda: jnp.array(
            [123 - 160 // 2, 123 - 160 // 2, 136 - 160 // 2, 136 - 160 // 2, 160 - 160 // 2, 160 - 160 // 2,
             174 - 160 // 2, 174 - 160 // 2]).astype(jnp.float32),
        lambda: jnp.array(
            [141 - 160 // 2, 155 - 160 // 2, 127 - 160 // 2, 169 - 160 // 2, 134 - 160 // 2, 162 - 160 // 2,
             120 - 160 // 2, 176 - 160 // 2]).astype(jnp.float32),
        lambda: jnp.array(
            [123 - 160 // 2, 170 - 160 // 2, 123 - 160 // 2, 180 - 160 // 2, 123 - 160 // 2, 170 - 160 // 2,
             123 - 160 // 2, -1]).astype(jnp.float32),
        lambda: jnp.array(
            [123 - 160 // 2, 180 - 160 // 2, 123 - 160 // 2, 170 - 160 // 2, 123 - 160 // 2, 180 - 160 // 2,
             123 - 160 // 2, -1]).astype(jnp.float32),
        lambda: jnp.array([78, -1, -1, -1, -1, -1, -1, -1]).astype(jnp.float32),
    ]
    ENEMY_POSITIONS_Y_LIST = [
        lambda: jnp.array(
            [210 - 135, 210 - 153, 210 - 117, 210 - 171, 210 - 117, 210 - 171, 210 - 135,
             210 - 153]).astype(jnp.float32),
        lambda: jnp.array(
            [210 - 171, 210 - 171, 210 - 135, 210 - 135, 210 - 153, 210 - 153, 210 - 117,
             210 - 117]).astype(jnp.float32),
        lambda: jnp.array(
            [210 - 99, 210 - 117, 210 - 135, 210 - 153, 210 - 171, 210 - 63, 210 - 81,
             210 + 20]).astype(jnp.float32),
        lambda: jnp.array(
            [210 - 63, 210 - 81, 210 - 99, 210 - 117, 210 - 135, 210 - 153, 210 - 171,
             210 + 20]).astype(jnp.float32),
        lambda: jnp.array([210 - 132, 210 + 20, 210 + 20, 210 + 20, 210 + 20, 210 + 20, 210 + 20,
                           210 + 20]).astype(jnp.float32),
    ]
    BLUE_BLOCK_X = jnp.linspace(PLAYER_BOUNDS[0] + 32, PLAYER_BOUNDS[1] - 32,
                                24).astype(jnp.int32)

    BLUE_BLOCK_Y_1 = jnp.full((24,), HEIGHT - 115, dtype=jnp.int32)
    BLUE_BLOCK_Y_2 = jnp.full((24,), HEIGHT - 117, dtype=jnp.int32)

    BLUE_BLOCK_POSITIONS = jnp.concatenate([
        jnp.stack((BLUE_BLOCK_X, BLUE_BLOCK_Y_1), axis=1),
        jnp.stack((BLUE_BLOCK_X, BLUE_BLOCK_Y_2), axis=1),
    ])

    # 1 Line with Blocks the same amount as Blue Blocks
    RED_BLOCK_X_1 = jnp.linspace(PLAYER_BOUNDS[0] + 32, PLAYER_BOUNDS[1] - 32, MAX_BOSS_BLOCK_BLUE).astype(jnp.int32)
    RED_BLOCK_X_2 = jnp.linspace(PLAYER_BOUNDS[0] + 36, PLAYER_BOUNDS[1] - 36, MAX_BOSS_BLOCK_BLUE - 2).astype(
        jnp.int32)
    RED_BLOCK_X_3 = jnp.linspace(PLAYER_BOUNDS[0] + 40, PLAYER_BOUNDS[1] - 40, MAX_BOSS_BLOCK_BLUE - 4).astype(
        jnp.int32)
    RED_BLOCK_X_4 = jnp.linspace(PLAYER_BOUNDS[0] + 44, PLAYER_BOUNDS[1] - 44, MAX_BOSS_BLOCK_BLUE - 6).astype(
        jnp.int32)
    RED_BLOCK_X_5 = jnp.linspace(PLAYER_BOUNDS[0] + 48, PLAYER_BOUNDS[1] - 48, MAX_BOSS_BLOCK_BLUE - 8).astype(
        jnp.int32)
    RED_BLOCK_X_6 = jnp.linspace(PLAYER_BOUNDS[0] + 52, PLAYER_BOUNDS[1] - 52, MAX_BOSS_BLOCK_BLUE - 10).astype(
        jnp.int32)
    RED_BLOCK_X_7 = jnp.linspace(PLAYER_BOUNDS[0] + 56, PLAYER_BOUNDS[1] - 56, MAX_BOSS_BLOCK_BLUE - 12).astype(
        jnp.int32)
    RED_BLOCK_POSITIONS = jnp.concatenate(
        [
            jnp.stack((RED_BLOCK_X_1, jnp.full((MAX_BOSS_BLOCK_BLUE,), HEIGHT - 111, dtype=jnp.int32)), axis=1),
            jnp.stack((RED_BLOCK_X_2, jnp.full((MAX_BOSS_BLOCK_BLUE - 2,), HEIGHT - 108, dtype=jnp.int32)), axis=1),
            jnp.stack((RED_BLOCK_X_3, jnp.full((MAX_BOSS_BLOCK_BLUE - 4,), HEIGHT - 105, dtype=jnp.int32)), axis=1),
            jnp.stack((RED_BLOCK_X_4, jnp.full((MAX_BOSS_BLOCK_BLUE - 6,), HEIGHT - 102, dtype=jnp.int32)), axis=1),
            jnp.stack((RED_BLOCK_X_5, jnp.full((MAX_BOSS_BLOCK_BLUE - 8,), HEIGHT - 99, dtype=jnp.int32)), axis=1),
            jnp.stack((RED_BLOCK_X_6, jnp.full((MAX_BOSS_BLOCK_BLUE - 10,), HEIGHT - 96, dtype=jnp.int32)), axis=1),
            jnp.stack((RED_BLOCK_X_7, jnp.full((MAX_BOSS_BLOCK_BLUE - 12,), HEIGHT - 93, dtype=jnp.int32)), axis=1)
        ],
        axis=0
    )

    GREEN_BLOCK_Y_1 = jnp.linspace(HEIGHT - 120, HEIGHT - 128, 5).astype(jnp.int32)
    GREEN_BLOCK_X_1 = jnp.full((5,), WIDTH // 2 + 8, dtype=jnp.int32)

    GREEN_BLOCK_X_2 = jnp.full((4,), WIDTH // 2 + 12, dtype=jnp.int32)
    GREEN_BLOCK_Y_2 = jnp.linspace(HEIGHT - 120, HEIGHT - 126, 4).astype(jnp.int32)

    GREEN_BLOCK_X_3 = jnp.full((3,), WIDTH // 2 + 16, dtype=jnp.int32)
    GREEN_BLOCK_Y_3 = jnp.linspace(HEIGHT - 120, HEIGHT - 124, 3).astype(jnp.int32)

    GREEN_BLOCK_X_4 = jnp.full((2,), WIDTH // 2 + 20, dtype=jnp.int32)
    GREEN_BLOCK_Y_4 = jnp.linspace(HEIGHT - 120, HEIGHT - 122, 2).astype(jnp.int32)

    GREEN_BLOCK_X_5 = jnp.full((1,), WIDTH // 2 + 24, dtype=jnp.int32)
    GREEN_BLOCK_Y_5 = jnp.linspace(HEIGHT - 120, HEIGHT - 120, 1).astype(jnp.int32)

    # mirror the blocks to the left side
    GREEN_BLOCK_Y_6 = jnp.linspace(HEIGHT - 120, HEIGHT - 128, 5).astype(jnp.int32)
    GREEN_BLOCK_X_6 = jnp.full((5,), WIDTH // 2 - 8, dtype=jnp.int32)

    GREEN_BLOCK_X_7 = jnp.full((4,), WIDTH // 2 - 12, dtype=jnp.int32)
    GREEN_BLOCK_Y_7 = jnp.linspace(HEIGHT - 120, HEIGHT - 126, 4).astype(jnp.int32)

    GREEN_BLOCK_X_8 = jnp.full((3,), WIDTH // 2 - 16, dtype=jnp.int32)
    GREEN_BLOCK_Y_8 = jnp.linspace(HEIGHT - 120, HEIGHT - 124, 3).astype(jnp.int32)

    GREEN_BLOCK_X_9 = jnp.full((2,), WIDTH // 2 - 20, dtype=jnp.int32)
    GREEN_BLOCK_Y_9 = jnp.linspace(HEIGHT - 120, HEIGHT - 122, 2).astype(jnp.int32)

    GREEN_BLOCK_X_10 = jnp.full((1,), WIDTH // 2 - 24, dtype=jnp.int32)
    GREEN_BLOCK_Y_10 = jnp.linspace(HEIGHT - 120, HEIGHT - 120, 1).astype(jnp.int32)

    GREEN_BLOCK_POSITIONS = jnp.concatenate(
        [
            jnp.stack((GREEN_BLOCK_X_1, GREEN_BLOCK_Y_1), axis=1),
            jnp.stack((GREEN_BLOCK_X_2, GREEN_BLOCK_Y_2), axis=1),
            jnp.stack((GREEN_BLOCK_X_3, GREEN_BLOCK_Y_3), axis=1),
            jnp.stack((GREEN_BLOCK_X_4, GREEN_BLOCK_Y_4), axis=1),
            jnp.stack((GREEN_BLOCK_X_5, GREEN_BLOCK_Y_5), axis=1),
            jnp.stack((GREEN_BLOCK_X_6, GREEN_BLOCK_Y_1), axis=1),
            jnp.stack((GREEN_BLOCK_X_7, GREEN_BLOCK_Y_2), axis=1),
            jnp.stack((GREEN_BLOCK_X_8, GREEN_BLOCK_Y_3), axis=1),
            jnp.stack((GREEN_BLOCK_X_9, GREEN_BLOCK_Y_4), axis=1),
            jnp.stack((GREEN_BLOCK_X_10, GREEN_BLOCK_Y_5), axis=1)

        ]
    )
    # Asset config baked into constants (immutable default) for asset overrides
    ASSET_CONFIG: tuple = _get_default_asset_config()

# === GAME STATE ===
class PhoenixState(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    step_counter: chex.Array
    enemies_x: chex.Array # Gegner X-Positionen
    enemies_y: chex.Array
    horizontal_direction_enemies: chex.Array
    vertical_direction_enemies: chex.Array
    blue_blocks: chex.Array
    red_blocks: chex.Array
    green_blocks: chex.Array
    invincibility: chex.Array
    invincibility_timer: chex.Array
    ability_cooldown: chex.Array

    bat_wings: chex.Array
    bat_dying: chex.Array # Bat dying status, (8,), bool
    bat_death_timer: chex.Array # Timer for Bat death animation, (8,), int
    bat_wing_regen_timer: chex.Array
    bat_y_cooldown: chex.Array

    phoenix_do_attack: chex.Array  # Phoenix attack state
    phoenix_attack_target_y: chex.Array  # Target Y position for Phoenix attack
    phoenix_original_y: chex.Array  # Original Y position of the Phoenix
    phoenix_cooldown: chex.Array
    phoenix_drift: chex.Array
    phoenix_returning: chex.Array # Returning status of the Phoenix
    phoenix_dying: chex.Array # Dying status of the Phoenix, (8,), bool
    phoenix_death_timer: chex.Array # Timer for Phoenix death animation, (8,), int

    player_dying: chex.Array = jnp.array(False)  # Player dying status, bool
    player_death_timer: chex.Array = jnp.array(0)  # Timer for player death animation, int
    player_moving: chex.Array = jnp.array(False) # Player moving status, bool

    projectile_x: chex.Array = jnp.array(-1)  # Standardwert: kein Projektil
    projectile_y: chex.Array = jnp.array(-1)  # Standardwert: kein Projektil # Gegner Y-Positionen
    enemy_projectile_x: chex.Array = jnp.full((8,), -1) # Enemy projectile X-Positionen
    enemy_projectile_y: chex.Array = jnp.full((8,), -1) # Enemy projectile Y-Positionen

    score: chex.Array = jnp.array(0)  # Score
    lives: chex.Array = jnp.array(5) # Lives
    player_respawn_timer: chex.Array = 0 # Invincibility timer
    level: chex.Array = jnp.array(1)  # Level, starts at 1
    level_transition_timer: chex.Array = jnp.array(0) # Timer for level transition

class PhoenixObservation(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    player_score: chex.Array
    lives: chex.Array

class PhoenixInfo(NamedTuple):
    step_counter: jnp.ndarray

class CarryState(NamedTuple):
    score: chex.Array

class EntityPosition(NamedTuple):## not sure
    x: chex.Array
    y: chex.Array

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
        player = EntityPosition(x=state.player_x, y=state.player_y)
        return PhoenixObservation(
            player_x=player[0],
            player_y=player[1],
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

    def observation_space(self) -> Space:
        return spaces.Dict({
            "player_x": spaces.Box(low=0, high=self.consts.WIDTH - 1, shape=(), dtype=jnp.int32),
            "player_y": spaces.Box(low=0, high=self.consts.HEIGHT - 1, shape=(), dtype=jnp.int32),
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
    def obs_to_flat_array(self, obs: PhoenixObservation) -> jnp.ndarray:
        return jnp.concatenate([
            obs.player_x.flatten(),
            obs.player_y.flatten(),
            obs.player_score.flatten(),
            obs.lives.flatten()
        ]
        )
    @partial(jax.jit, static_argnums=(0,))
    def player_step(self, state: PhoenixState, action: chex.Array) -> tuple[chex.Array]:
        step_size = 2  # Größerer Wert = schnellerer Schritt
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
        new_timer = jnp.where(invinsibility & (state.invincibility_timer == 0), 200, state.invincibility_timer)

        new_timer = jnp.where(new_timer > 0, new_timer - 1, 0)
        new_invinsibility = jnp.where(new_timer == 0, False, new_invinsibility)

        new_cooldown = jnp.where(new_timer == 2, self.consts.ABILITY_COOLDOWN, state.ability_cooldown)
        new_cooldown = jnp.where(new_cooldown > 0, new_cooldown - 1, 0)
        # movement right
        player_x = jnp.where(
            right & jnp.logical_not(new_invinsibility), state.player_x + step_size, jnp.where(left & jnp.logical_not(new_invinsibility), state.player_x - step_size, state.player_x)
        )
        # movement left
        player_x = jnp.where(
            player_x < self.consts.PLAYER_BOUNDS[0], self.consts.PLAYER_BOUNDS[0],
            jnp.where(player_x > self.consts.PLAYER_BOUNDS[1], self.consts.PLAYER_BOUNDS[1], player_x)
        )

        # Did the player move?
        player_moved = jnp.not_equal(player_x.astype(jnp.int32), state.player_x.astype(jnp.int32))
        new_player_moving = player_moved.astype(jnp.bool_)

        state = state._replace(player_x= player_x.astype(jnp.int32),
                               invincibility=new_invinsibility,
                               invincibility_timer=new_timer,
                                player_moving=new_player_moving,
                                 ability_cooldown=new_cooldown
        )

        return state

    def phoenix_step(self, state):
        enemy_step_size = 0.4
        attack_speed = 1#0.4
        tolerance = 0.5 #TODO Kann evtl entfernt werden

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
        attack_chance = jax.random.uniform(key, shape=()) < 0.005
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
        drift_prob = 0.6
        drift_max = 0.6#0.35
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
        common_delay = jax.random.randint(key_delay, (), 30, 120)
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
        new_phoenix_cooldown = jnp.where(arrived, 30, new_phoenix_cooldown)

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
        new_direction = jax.lax.cond(
            at_left_boundary,
            lambda: jnp.full_like(state.horizontal_direction_enemies, 1.0, dtype=jnp.float32),
            lambda: jax.lax.cond(
                at_right_boundary,
                lambda: jnp.full_like(state.horizontal_direction_enemies, -1.0, dtype=jnp.float32),
                lambda: state.horizontal_direction_enemies.astype(jnp.float32),
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


        state = state._replace(
            enemies_x=new_enemies_x.astype(jnp.float32),
            horizontal_direction_enemies=new_direction.astype(jnp.float32),
            enemies_y=new_enemies_y.astype(jnp.float32),
            vertical_direction_enemies=state.vertical_direction_enemies.astype(jnp.float32),
            blue_blocks=state.blue_blocks.astype(jnp.float32),
            red_blocks=state.red_blocks.astype(jnp.float32),
            green_blocks=state.green_blocks.astype(jnp.float32),
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
        return state

    def bat_step(self, state):
        bat_step_size = 0.5
        bat_y_step = 2
        bat_y_chance = 0.1
        active_bats = (state.enemies_x > -1) & (state.enemies_y < self.consts.HEIGHT + 10) & (~state.bat_dying)
        proj_pos = jnp.array([state.projectile_x, state.projectile_y])
        cooldown_ready = (state.bat_y_cooldown == 0) & active_bats

        key = jax.random.PRNGKey(state.step_counter)
        y_move_chance = jax.random.uniform(key, shape=state.enemies_y.shape) < bat_y_chance
        dir_key = jax.random.PRNGKey(state.step_counter + 123)
        y_direction = jnp.where(jax.random.uniform(dir_key, shape=state.enemies_y.shape) < 0.5,1.0,-1.0)

        y_move = jnp.where(cooldown_ready & y_move_chance, bat_y_step * y_direction, 0.0)

        # Initialisiere neue Richtungen für jede Fledermaus
        new_directions = jnp.where(
            jnp.logical_and(state.enemies_x <= self.consts.PLAYER_BOUNDS[0] + 3, active_bats),
            jnp.ones(state.horizontal_direction_enemies.shape, dtype=jnp.float32),  # Force array shape
            jnp.where(
                jnp.logical_and(state.enemies_x >= self.consts.PLAYER_BOUNDS[1] - self.consts.ENEMY_WIDTH / 2,
                                active_bats),
                jnp.ones(state.horizontal_direction_enemies.shape, dtype=jnp.float32) * -1,  # Force array shape
                state.horizontal_direction_enemies.astype(jnp.float32)  # Ensure consistency
            )
        )

        # Bewege Fledermäuse basierend auf ihrer individuellen Richtung
        #new_enemies_x = jnp.where(active_bats, state.enemies_x + (new_directions * bat_step_size), state.enemies_x)
        #enemy_pos = jnp.stack([new_enemies_x, state.enemies_y], axis=1)
        #new_enemies_x = jnp.clip(new_enemies_x, self.consts.PLAYER_BOUNDS[0], self.consts.PLAYER_BOUNDS[1])

        #new_enemies_y = jnp.where(active_bats, state.enemies_y + y_move, state.enemies_y)
        #new_enemies_y = jnp.clip(new_enemies_y, 0, self.consts.HEIGHT - self.consts.ENEMY_HEIGHT)
        #new_y_cooldown = jnp.where(cooldown_ready & y_move_chance, 50, jnp.maximum(state.bat_y_cooldown-1,0))

        # Horizontal: nur aktive Bats bewegen und clippen
        proposed_x = jnp.where(active_bats, state.enemies_x + (new_directions * bat_step_size), state.enemies_x)
        clipped_x = jnp.clip(proposed_x, self.consts.PLAYER_BOUNDS[0], self.consts.PLAYER_BOUNDS[1])
        new_enemies_x = jnp.where(active_bats, clipped_x, state.enemies_x)

        # Vertikal: nur aktive Bats bewegen und clippen
        proposed_y = jnp.where(active_bats, state.enemies_y + y_move, state.enemies_y)
        clipped_y = jnp.clip(proposed_y, 0, self.consts.HEIGHT - self.consts.ENEMY_HEIGHT)
        new_enemies_y = jnp.where(active_bats, clipped_y, state.enemies_y)

        # Für Kollisionen die neuen Y-Werte verwenden
        enemy_pos = jnp.stack([new_enemies_x, new_enemies_y], axis=1)

        new_y_cooldown = jnp.where(cooldown_ready & y_move_chance, 50, jnp.maximum(state.bat_y_cooldown - 1, 0))

        def check_collision(entity_pos, projectile_pos):
            enemy_x, enemy_y = entity_pos
            proj_x, proj_y = projectile_pos
            wing_left_x = enemy_x - 5
            wing_y = enemy_y + 2
            wing_right_x = enemy_x + 5
            collision_x_left = (proj_x + self.consts.PROJECTILE_WIDTH > wing_left_x) & (
                    proj_x < wing_left_x + self.consts.WING_WIDTH)
            collision_y = (proj_y + self.consts.PROJECTILE_HEIGHT > wing_y) & (
                    proj_y < enemy_y + 2)
            collision_x_right = (proj_x + self.consts.PROJECTILE_WIDTH > wing_right_x) & (
                    proj_x < wing_right_x + self.consts.WING_WIDTH)

            return collision_x_left & collision_y, collision_x_right & collision_y

        left_wing_collision, right_wing_collision = jax.vmap(lambda entity_pos: check_collision(entity_pos, proj_pos))(enemy_pos)
        left_hit_valid = left_wing_collision & ((state.bat_wings == 2) | (state.bat_wings == -1))
        right_hit_valid = right_wing_collision & ((state.bat_wings == 2) | (state.bat_wings == 1))

        # Only remove the projectile if any valid hit occurred
        any_valid_hit = jnp.any(left_hit_valid | right_hit_valid)
        new_proj_y = jnp.where(any_valid_hit, -1, state.projectile_y)
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

        state = state._replace(
            enemies_x=new_enemies_x.astype(jnp.float32),
            enemies_y=new_enemies_y.astype(jnp.float32),
            horizontal_direction_enemies=new_directions.astype(jnp.float32),
            projectile_y=new_proj_y,
            blue_blocks=state.blue_blocks.astype(jnp.float32),
            red_blocks=state.red_blocks.astype(jnp.float32),
            green_blocks=state.green_blocks.astype(jnp.float32),
            bat_wings= new_bat_wings,
            bat_wing_regen_timer=new_regen_timer,
            bat_y_cooldown=new_y_cooldown.astype(jnp.int32)
        )

        return state

    def boss_step(self, state):
        step_size = 0.05
        step_count = state.step_counter

        condition = (state.enemies_y[0] <= 100) & ((step_count % 30) == 0)

        def move_blocks(blocks):
            not_removed = blocks[:, 0] > -99  # Filter out removed blocks
            move_mask = condition & not_removed
            return blocks.at[:, 1].set(
                jnp.where(move_mask, blocks[:, 1] + step_size, blocks[:, 1])
            )

        new_green_blocks = move_blocks(state.green_blocks)
        new_red_blocks = move_blocks(state.red_blocks)
        new_blue_blocks = move_blocks(state.blue_blocks)

        new_enemy_y = jnp.where(condition, state.enemies_y + step_size, state.enemies_y.astype(jnp.float32))

        projectile_active = (state.projectile_x >= 0) & (state.projectile_y >= 0)
        projectile_pos = jnp.array([state.projectile_x, state.projectile_y])

        def check_collision(entity_pos, projectile_pos):
            enemy_x, enemy_y = entity_pos
            projectile_x, projectile_y = projectile_pos

            collision_x = (projectile_x + self.consts.PROJECTILE_WIDTH > enemy_x) & (
                        projectile_x < enemy_x + self.consts.BLOCK_WIDTH)
            collision_y = (projectile_y + self.consts.PROJECTILE_HEIGHT > enemy_y) & (
                        projectile_y < enemy_y + self.consts.BLOCK_HEIGHT)
            return collision_x & collision_y

        def process_collisions(_):
            # Check collisions for each block group
            green_block_collisions = jax.vmap(lambda entity_pos: check_collision(entity_pos, projectile_pos))(
                new_green_blocks)
            red_block_collisions = jax.vmap(lambda entity_pos: check_collision(entity_pos, projectile_pos))(
                new_red_blocks)
            blue_block_collisions = jax.vmap(lambda entity_pos: check_collision(entity_pos, projectile_pos))(
                new_blue_blocks)

            def remove_first_hit(blocks, collisions):
                hit_indices = jnp.where(collisions, size=1, fill_value=-1)[0]
                first_hit_index = hit_indices[0]  # scalar

                def remove_block(i, arr):
                    # Create a mask with True only at index i
                    mask = jnp.arange(arr.shape[0]) == i  # shape (50,)
                    # Broadcast mask to shape (50, 1) to align with arr shape (50, 2)
                    mask = mask[:, None]  # shape (50, 1)

                    # Create an array of -100 with the same shape as one block (2,)
                    replacement = jnp.full(arr.shape[1:], -100)  # shape (2,)

                    # Use jnp.where to replace the entire row at index i with -100 vector
                    return jnp.where(mask, replacement, arr)

                return jax.lax.cond(
                    first_hit_index >= 0,
                    lambda: remove_block(first_hit_index, blocks),
                    lambda: blocks
                ), first_hit_index

            new_green, _ = remove_first_hit(new_green_blocks, green_block_collisions)
            new_red, _ = remove_first_hit(new_red_blocks, red_block_collisions)
            new_blue, first_hit_idx = remove_first_hit(new_blue_blocks, blue_block_collisions)
            hit_any = jnp.any(green_block_collisions) | jnp.any(red_block_collisions) | jnp.any(blue_block_collisions)
            return new_green, new_red, new_blue, hit_any

        def skip_collisions(_):
            return (new_green_blocks, new_red_blocks, new_blue_blocks, False)

        # Use lax.cond to select between processing or skipping collision
        new_green_blocks, new_red_blocks, new_blue_blocks, projectile_hit_detected = jax.lax.cond(
            projectile_active,
            process_collisions,
            skip_collisions,
            operand=None
        )

        def rotate(arr):
            return jnp.stack([jnp.roll(arr[:, 0], 1), arr[:, 1]], axis=1)

        new_blue_blocks = jax.lax.cond(
            jnp.logical_and(jnp.any(new_blue_blocks <= -100), step_count % 20 == 0),
            lambda: rotate(new_blue_blocks),
            lambda: new_blue_blocks,
        )
        projectile_x = jnp.where(projectile_hit_detected, -1, state.projectile_x)
        projectile_y = jnp.where(projectile_hit_detected, -1, state.projectile_y)

        state = state._replace(
            enemies_y=new_enemy_y.astype(jnp.float32),
            blue_blocks=new_blue_blocks.astype(jnp.float32),
            red_blocks=new_red_blocks.astype(jnp.float32),
            green_blocks=new_green_blocks.astype(jnp.float32),
            projectile_x=projectile_x.astype(jnp.int32),
            projectile_y=projectile_y.astype(jnp.int32),
            enemies_x = state.enemies_x.astype(jnp.float32),
        )
        return state

    def reset(self, key: jax.random.PRNGKey = jax.random.PRNGKey(42)) -> Tuple[PhoenixObservation, PhoenixState]:

        return_state = PhoenixState(
            player_x=jnp.array(self.consts.PLAYER_POSITION[0], dtype=jnp.int32),
            player_y=jnp.array(self.consts.PLAYER_POSITION[1], dtype=jnp.int32),
            step_counter=jnp.array(0),
            enemies_x = self.consts.ENEMY_POSITIONS_X_LIST[0](),
            enemies_y = self.consts.ENEMY_POSITIONS_Y_LIST[0](),
            horizontal_direction_enemies = jnp.full((8,), -1.0),
            vertical_direction_enemies = jnp.full((8,), 1.0),
            enemy_projectile_x=jnp.full((8,), -1),
            enemy_projectile_y=jnp.full((8,), -1),
            projectile_x=jnp.array(-1),  # Standardwert: kein Projektil
            score = jnp.array(0), # Standardwert: Score=0
            lives=jnp.array(self.consts.PLAYER_LIVES), # Standardwert: 4 Leben
            player_respawn_timer=jnp.array(5),
            level=jnp.array(1),
            level_transition_timer=jnp.array(0),  # Timer for level transition, starts at 0

            invincibility=jnp.array(False),
            invincibility_timer=jnp.array(0),
            ability_cooldown=jnp.array(0),

            bat_wings=jnp.full((8,), 2),
            bat_dying=jnp.full((8,), False, dtype=jnp.bool), # Bat dying status, (8,), bool
            bat_death_timer=jnp.full((8,), 0, dtype=jnp.int32), # Timer for Bat death animation, (8,), int
            bat_wing_regen_timer=jnp.full((8,), 0, dtype=jnp.int32),
            bat_y_cooldown=jnp.full((8,), 0, dtype=jnp.int32),
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

            # Initialierung der Blockpositionen
            blue_blocks=self.consts.BLUE_BLOCK_POSITIONS.astype(jnp.float32),
            red_blocks=self.consts.RED_BLOCK_POSITIONS.astype(jnp.float32),
            green_blocks = self.consts.GREEN_BLOCK_POSITIONS.astype(jnp.float32),
        )

        initial_obs = self._get_observation(return_state)
        return initial_obs, return_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action: int) -> Tuple[PhoenixObservation, PhoenixState, float, bool, PhoenixInfo]:
        # Translate agent action index to ALE console action
        atari_action = jnp.take(self.ACTION_SET, jnp.asarray(action, dtype=jnp.int32))
        
        new_respawn_timer = jnp.where(state.player_respawn_timer > 0, state.player_respawn_timer - 1, 0)
        respawn_ended = (state.player_respawn_timer > 0) & (new_respawn_timer == 0)

        state = state._replace(player_respawn_timer=new_respawn_timer.astype(jnp.int32))

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

        state = jax.lax.cond(
            jnp.logical_or((state.level % 5) == 1, (state.level % 5) == 2),
            lambda: self.phoenix_step(state),
            lambda: jax.lax.cond(
                jnp.logical_or((state.level % 5) == 3, (state.level % 5) == 4),
                lambda: self.bat_step(state),
                lambda: self.boss_step(state),

            )
        )
        projectile_x = jnp.where(firing,
                                 state.player_x + 2,
                                 state.projectile_x).astype(jnp.int32)

        projectile_y = jnp.where(firing,
                                 state.player_y - 1,
                                 jnp.where(projectile_active,
                                           state.projectile_y - 3,  # move up if active
                                           state.projectile_y))  # stay
        projectile_y = jnp.where(projectile_y < 0, -6, projectile_y)
        # use step_counter for randomness
        def generate_fire_key_and_chance(step_counter: int, fire_chance: float) -> Tuple[jax.random.PRNGKey, float]:
            key = jax.random.PRNGKey(step_counter)
            return key, fire_chance

        key, fire_chance = generate_fire_key_and_chance(state.step_counter, self.consts.FIRE_CHANCE)

        # Random decision: should each enemy fire?
        enemy_should_fire = jax.random.uniform(key, (8,)) < fire_chance

        # Fire only from active enemies
        can_fire = (state.enemy_projectile_y < 0) & (state.enemies_x > -1)
        not_attacking = jnp.logical_not(jnp.logical_or(state.phoenix_do_attack, state.phoenix_returning))
        not_attacking = not_attacking & (~state.phoenix_returning)
        enemy_fire_mask = enemy_should_fire & can_fire & not_attacking


        # Fire from current enemy positions
        enemy_projectile_x = jnp.where(enemy_fire_mask, state.enemies_x + self.consts.ENEMY_WIDTH // 2,
                                           state.enemy_projectile_x)
        enemy_projectile_y = jnp.where(enemy_fire_mask, state.enemies_y + self.consts.ENEMY_HEIGHT, state.enemy_projectile_y)

        # Move enemy projectiles downwards
        enemy_projectile_y = jnp.where(state.enemy_projectile_y >= 0, state.enemy_projectile_y + self.consts.ENEMY_PROJECTILE_SPEED,
                                           enemy_projectile_y)

        # Remove enemy projectile if off-screen
        enemy_projectile_y = jnp.where(enemy_projectile_y > 185 - self.consts.PROJECTILE_HEIGHT, -1, enemy_projectile_y) # TODO 185 durch Konstante ersetzen, die global geändert werden kann.



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
        is_bat_level = jnp.logical_or((state.level % 5) == 3, (state.level % 5) == 4)
        dying_mask = jnp.where(is_bat_level, state.bat_dying, state.phoenix_dying)
        enemy_collisions = enemy_collisions_raw & (~dying_mask)
        enemy_hit_detected = jnp.any(enemy_collisions)

        # Phoenix-Death-Animation starten (nur Phoenix-Levels)
        p_hit_mask = enemy_collisions & (~is_bat_level)
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
        b_hit_mask = enemy_collisions & is_bat_level
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

        # Projektil/Score bei jedem Treffer (egal welches Level)
        projectile_x = jnp.where(enemy_hit_detected, -1, projectile_x)
        projectile_y = jnp.where(enemy_hit_detected, -1, projectile_y)
        score = jnp.where(enemy_hit_detected, state.score + 20, state.score)

        # Gegner entfernen nach Ablauf der jeweiligen Death-Animation
        death_done_any = jnp.where(is_bat_level, b_death_done, p_death_done)
        enemies_x = jnp.where(death_done_any, -1, state.enemies_x)
        enemies_y = jnp.where(death_done_any, self.consts.HEIGHT + 20, state.enemies_y)

        score = jnp.where(enemy_hit_detected, state.score + 20, state.score)


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
        next_enemies_x = jax.lax.switch((pending_next_level - 1) % 5, self.consts.ENEMY_POSITIONS_X_LIST).astype(
            jnp.float32)
        next_enemies_y = jax.lax.switch((pending_next_level - 1) % 5, self.consts.ENEMY_POSITIONS_Y_LIST).astype(
            jnp.float32)


        reset_mask = transition_ended
        enemies_x = jnp.where(reset_mask, next_enemies_x.astype(jnp.float32), enemies_x)
        enemies_y = jnp.where(reset_mask, next_enemies_y.astype(jnp.float32), enemies_y)

        # Richtungen der Formation zurücksetzen
        new_horizontal_direction_enemies = jnp.where(
            reset_mask, jnp.full((8,), -1.0, dtype=jnp.float32), state.horizontal_direction_enemies
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

        # Boss-Blöcke nur beim Eintritt in das Boss-Level neu initialisieren
        enter_boss_next = ((pending_next_level % 5) == 0)
        reset_blocks = reset_mask & enter_boss_next
        blue_blocks = jnp.where(reset_blocks, self.consts.BLUE_BLOCK_POSITIONS.astype(jnp.float32), state.blue_blocks)
        red_blocks = jnp.where(reset_blocks, self.consts.RED_BLOCK_POSITIONS.astype(jnp.float32), state.red_blocks)
        green_blocks = jnp.where(reset_blocks, self.consts.GREEN_BLOCK_POSITIONS.astype(jnp.float32), state.green_blocks)

        # Gegner-Respawn nach Spieler-Respawn nur, wenn kein Level-Übergang läuft
        enemy_respawn_x = jax.lax.switch((level - 1) % 5, self.consts.ENEMY_POSITIONS_X_LIST).astype(jnp.float32)
        enemy_respawn_y = jax.lax.switch((level - 1) % 5, self.consts.ENEMY_POSITIONS_Y_LIST).astype(jnp.float32)

        enemy_respawn_mask = respawn_ended & (new_level_transition_timer == 0)
        enemy_alive_mask = (enemies_x > -1) & (enemies_y < self.consts.HEIGHT + 10)
        enemies_x = jnp.where(enemy_respawn_mask & enemy_alive_mask, enemy_respawn_x, enemies_x)
        enemies_y = jnp.where(enemy_respawn_mask & enemy_alive_mask, enemy_respawn_y, enemies_y)



        is_vulnerable = (new_respawn_timer <= 0) & (~state.player_dying) & (~state.invincibility)

        def check_player_hit(projectile_xs, projectile_ys, player_x, player_y):
            def is_hit(px, py):
                hit_x = (px + self.consts.PROJECTILE_WIDTH > player_x) & (px < player_x + 5) # TODO 5 durch Konstante ersetzen, die global geändert werden kann.
                hit_y = (py + self.consts.PROJECTILE_HEIGHT > player_y) & (py < player_y + self.consts.PROJECTILE_HEIGHT)
                return hit_x & hit_y

            hits = jax.vmap(is_hit)(projectile_xs, projectile_ys)
            return jnp.any(hits)



        # Kollisionsüberprüfung Spieler
        player_hit_detected = jnp.where(
            is_vulnerable & (state.invincibility == jnp.array(False)),
            check_player_hit(enemy_projectile_x, enemy_projectile_y, state.player_x, state.player_y),
            False
        )

        # Bei Treffer: Spieler-Dying-Status setzen und Timer starten
        player_death_duration = self.consts.PLAYER_DEATH_DURATION
        new_player_dying = jnp.where(player_hit_detected, True, state.player_dying)
        player_death_timer_start = jnp.where(player_hit_detected, player_death_duration, state.player_death_timer)

        lives = jnp.where(player_hit_detected, state.lives - 1, state.lives)


        # Enemy Projectile entfernen wenn eine Kollision mit dem Spieler erkannt wurde
        enemy_projectile_x = jnp.where(player_hit_detected, -1, enemy_projectile_x)
        enemy_projectile_y = jnp.where(player_hit_detected, -1, enemy_projectile_y)

        # Player-Death-Teimer herunterzählen
        dec_player_timer = jnp.where(
            new_player_dying & (player_death_timer_start > 0),
            player_death_timer_start - 1,
            player_death_timer_start
        )
        player_death_done = new_player_dying & (dec_player_timer == 0) & (player_death_timer_start > 0)

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
        new_phoenix_do_attack = jnp.where(formation_reset, jnp.full((8,), False), state.phoenix_do_attack)
        new_phoenix_returning = jnp.where(formation_reset, jnp.full((8,), False), state.phoenix_returning)
        new_phoenix_attack_target = jnp.where(formation_reset, jnp.full((8,), -1.0), state.phoenix_attack_target_y)
        new_phoenix_cooldown = jnp.where(formation_reset, jnp.full((8,), 0), state.phoenix_cooldown)
        new_phoenix_drift = jnp.where(formation_reset, jnp.full((8,), 0.0), state.phoenix_drift)
        new_phoenix_original_y = jnp.where(formation_reset, jnp.full((8,), -1.0), state.phoenix_original_y)

        return_state = PhoenixState(
            player_x = player_x,
            player_y = state.player_y,
            step_counter = state.step_counter + 1,
            projectile_x = projectile_x,
            projectile_y = projectile_y,
            enemies_x = enemies_x,
            enemies_y = enemies_y,
            horizontal_direction_enemies = new_horizontal_direction_enemies,
            score= score,
            enemy_projectile_x=enemy_projectile_x.astype(jnp.int32),
            enemy_projectile_y=enemy_projectile_y.astype(jnp.int32),
            lives=lives,
            player_respawn_timer = player_respawn_timer,
            level = level,
            vertical_direction_enemies=new_vertical_direction_enemies,
            blue_blocks=blue_blocks.astype(jnp.float32),
            red_blocks=red_blocks.astype(jnp.float32),
            green_blocks=green_blocks.astype(jnp.float32),
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

        )
        observation = self._get_observation(return_state)
        env_reward = jnp.where(enemy_hit_detected, 1.0, 0.0)
        done = self._get_done(return_state)
        info = self._get_info(return_state)
        return observation, return_state, env_reward, done, info

    def render(self, state:PhoenixState) -> jnp.ndarray:
        return self.renderer.render(state)

from jaxatari.renderers import JAXGameRenderer

class PhoenixRenderer(JAXGameRenderer):
    def __init__(self, consts: PhoenixConstants = None):
        super().__init__()
        self.consts = consts or PhoenixConstants()
        
        # 1. Configure the renderer
        self.config = render_utils.RendererConfig(
            game_dimensions=(210, 160),
            channels=3,
            #downscale=(84, 84),
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)
        
        # 2. Define sprite path
        sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/phoenix"
        
        # 3. Use asset config from constants
        final_asset_config = list(self.consts.ASSET_CONFIG)
        
        # 4. Load all assets, create palette, and generate ID masks in one call
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(final_asset_config, sprite_path)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        # Start with the background raster
        raster = self.jr.create_object_raster(self.BACKGROUND)

        # Render common elements
        raster = self._render_common(state, raster)

        # Dispatch to level-specific renderers
        level_idx = (state.level - 1) % 5
        raster = jax.lax.cond(
            (level_idx == 0) | (level_idx == 1),
            lambda r: self._render_phoenix_level(state, r),
            lambda r: r,
            raster
        )
        raster = jax.lax.cond(
            level_idx == 2,
            lambda r: self._render_bat_level(state, r, True),
            lambda r: r,
            raster
        )
        raster = jax.lax.cond(
            level_idx == 3,
            lambda r: self._render_bat_level(state, r, False),
            lambda r: r,
            raster
        )
        raster = jax.lax.cond(
            level_idx == 4,
            lambda r: self._render_boss_level(state, r),
            lambda r: r,
            raster
        )

        # UI on top
        raster = self._render_ui(state, raster)

        # Final palette lookup
        return self.jr.render_from_palette(raster, self.PALETTE)

    @partial(jax.jit, static_argnums=(0,))
    def _render_common(self, state, raster):
        raster = self.jr.render_at(raster, 0, 185, self.SHAPE_MASKS['floor'])

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

        def render_player_projectile(r):
            return self.jr.render_at_clipped(r, state.projectile_x, state.projectile_y, self.SHAPE_MASKS['player_projectile'])

        raster = jax.lax.cond(
            state.projectile_x > -1, render_player_projectile, lambda r: r, raster
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
                lambda r: self.jr.render_at_clipped(r, x, y, self.SHAPE_MASKS['enemy_projectile']),
                lambda r: r,
                current_raster
            )

        raster = jax.lax.fori_loop(0, state.enemy_projectile_x.shape[0], render_enemy_projectile, raster)
        return raster

    @partial(jax.jit, static_argnums=(0,))
    def _render_phoenix_level(self, state, raster):
        tol = 0.5
        going_down = state.phoenix_do_attack & (state.enemies_y < state.phoenix_attack_target_y - tol)
        going_up = state.phoenix_do_attack & (state.enemies_y > state.phoenix_attack_target_y + tol)
        returning_moving = state.phoenix_returning & (jnp.abs(state.enemies_y - state.phoenix_original_y) > tol)
        is_moving_vert = going_down | going_up | returning_moving

        phoenix_death_flags = state.phoenix_dying
        phoenix_death_phase = (state.phoenix_death_timer <= self.consts.ENEMY_DEATH_DURATION // 2).astype(jnp.int32)
        anim_toggle = ((state.step_counter // self.consts.ENEMY_ANIMATION_SPEED) % 2) == 0
        phoenix_flip_offset = self.FLIP_OFFSETS['phoenix']

        def render_single_phoenix(i, current_raster):
            x, y = state.enemies_x[i], state.enemies_y[i]
            is_active = (x > -1) & (y < self.consts.HEIGHT + 10)

            def draw_enemy(r):
                death_idx = jax.lax.select(phoenix_death_phase[i] == 0, 3, 4)
                alive_idx = jax.lax.select(is_moving_vert[i], 2, jax.lax.select(anim_toggle, 0, 1))
                frame_idx = jax.lax.select(phoenix_death_flags[i], death_idx, alive_idx)
                mask = self.SHAPE_MASKS['phoenix'][frame_idx]
                return self.jr.render_at(r, x, y, mask, flip_offset=phoenix_flip_offset)

            return jax.lax.cond(is_active, draw_enemy, lambda r: r, current_raster)

        return jax.lax.fori_loop(0, state.enemies_x.shape[0], render_single_phoenix, raster)

    @partial(jax.jit, static_argnums=(0, 3))
    def _render_bat_level(self, state, raster, is_blue_level: bool):
        bat_death_seg = jnp.maximum(1, self.consts.ENEMY_DEATH_DURATION // 3)
        body_masks = self.SHAPE_MASKS['bat_blue_body'] if is_blue_level else self.SHAPE_MASKS['bat_red_body']
        body_offsets = self.FLIP_OFFSETS['bat_blue_body'] if is_blue_level else self.FLIP_OFFSETS['bat_red_body']
        wing_masks = self.SHAPE_MASKS['bat_blue_wings'] if is_blue_level else self.SHAPE_MASKS['bat_red_wings']
        wing_offsets = self.FLIP_OFFSETS['bat_blue_wings'] if is_blue_level else self.FLIP_OFFSETS['bat_red_wings']
        left_wing_mask = wing_masks[0]
        right_wing_mask = wing_masks[1]

        def render_single_bat(i, current_raster):
            x = state.enemies_x[i].astype(jnp.int32)
            y = state.enemies_y[i].astype(jnp.int32)
            is_active = (x > -1) & (y < self.consts.HEIGHT + 10)
            is_dying = state.bat_dying[i]

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
                    r_new = self.jr.render_at(r, x, y, body_masks[0], flip_offset=body_offsets)
                    wing_state = state.bat_wings[i].astype(jnp.int32)
                    draw_left = (wing_state == 2) | (wing_state == -1)
                    draw_right = (wing_state == 2) | (wing_state == 1)
                    x_left = x - self.consts.WING_WIDTH
                    x_right = x + self.consts.ENEMY_WIDTH - 1
                    y_wings = y + 2
                    r_new = jax.lax.cond(
                        draw_left,
                        lambda r2: self.jr.render_at(r2, x_left, y_wings, left_wing_mask, flip_offset=wing_offsets),
                        lambda r2: r2,
                        r_new
                    )
                    r_new = jax.lax.cond(
                        draw_right,
                        lambda r2: self.jr.render_at(r2, x_right, y_wings, right_wing_mask, flip_offset=wing_offsets),
                        lambda r2: r2,
                        r_new
                    )
                    return r_new

                return jax.lax.cond(is_dying, draw_death, draw_alive, rr)

            return jax.lax.cond(is_active, draw_one, lambda rr: rr, current_raster)

        return jax.lax.fori_loop(0, state.enemies_x.shape[0], render_single_bat, raster)

    @partial(jax.jit, static_argnums=(0,))
    def _render_boss_level(self, state, raster):
        boss_mask = self.SHAPE_MASKS['boss']
        boss_flip_offset = self.FLIP_OFFSETS['boss']

        def render_single_boss(i, current_raster):
            x, y = state.enemies_x[i], state.enemies_y[i]
            is_active = (x > -1) & (y < self.consts.HEIGHT + 10)
            return jax.lax.cond(
                is_active,
                lambda r: self.jr.render_at(r, x, y, boss_mask, flip_offset=boss_flip_offset),
                lambda r: r,
                current_raster
            )

        raster = jax.lax.fori_loop(0, state.enemies_x.shape[0], render_single_boss, raster)

        # Efficient grid-based block rendering using inverse mapping (like Breakout)
        grid_rows = (self.consts.HEIGHT + self.consts.BLOCK_HEIGHT - 1) // self.consts.BLOCK_HEIGHT
        grid_cols = (self.consts.WIDTH + self.consts.BLOCK_WIDTH - 1) // self.consts.BLOCK_WIDTH
        object_id_grid = jnp.zeros((grid_rows, grid_cols), dtype=jnp.int32)

        def positions_to_grid_ids(obj_grid, positions, obj_id):
            pos = positions[:, 0:2].astype(jnp.int32)
            valid = (pos[:, 0] >= 0) & (pos[:, 1] >= 0)
            pos = jnp.where(valid[:, None], pos, -1)
            cols = jnp.clip(pos[:, 0] // self.consts.BLOCK_WIDTH, 0, grid_cols - 1)
            rows = jnp.clip(pos[:, 1] // self.consts.BLOCK_HEIGHT, 0, grid_rows - 1)
            rows = jnp.where(valid, rows, 0)
            cols = jnp.where(valid, cols, 0)
            return obj_grid.at[rows, cols].set(
                jnp.where(valid, jnp.int32(obj_id), obj_grid[rows, cols])
            )

        object_id_grid = positions_to_grid_ids(object_id_grid, state.blue_blocks, 1)
        object_id_grid = positions_to_grid_ids(object_id_grid, state.red_blocks, 2)
        object_id_grid = positions_to_grid_ids(object_id_grid, state.green_blocks, 3)

        blue_color_id = jnp.asarray(self.SHAPE_MASKS['boss_block_blue'][0, 0], dtype=jnp.uint8)
        red_color_id = jnp.asarray(self.SHAPE_MASKS['boss_block_red'][0, 0], dtype=jnp.uint8)
        green_color_id = jnp.asarray(self.SHAPE_MASKS['boss_block_green'][0, 0], dtype=jnp.uint8)
        # ID 0 is background/no block; we won't paint over raster in that case
        color_map = jnp.array([self.BACKGROUND[0, 0], blue_color_id, red_color_id, green_color_id], dtype=jnp.uint8)

        raster = self.jr.render_grid_inverse(
            raster,
            grid_state=object_id_grid,
            grid_origin=(0, 0),
            cell_size=(self.consts.BLOCK_WIDTH, self.consts.BLOCK_HEIGHT),
            color_map=color_map,
        )

        return raster

    @partial(jax.jit, static_argnums=(0,))
    def _render_ui(self, state, raster):
        max_digits = 5
        spacing = 8
        score_y = 10
        digit_masks = self.SHAPE_MASKS['digits']
        digit_w = digit_masks[0].shape[1]
        score_digits = self.jr.int_to_digits(state.score, max_digits=max_digits)
        has_nonzero = jnp.any(score_digits != 0)
        first_idx = jnp.where(has_nonzero, jnp.argmax(score_digits != 0), max_digits - 1)
        num_to_render = jnp.where(has_nonzero, max_digits - first_idx, 1)
        start_index = first_idx
        field_total_w = max_digits * spacing
        base_left = (self.consts.WIDTH - field_total_w) // 2
        score_x = base_left + first_idx * spacing
        raster = self.jr.render_label_selective(
            raster, score_x, score_y,
            score_digits, digit_masks,
            start_index, num_to_render,
            spacing=spacing, max_digits_to_render=max_digits
        )
        life_mask = self.SHAPE_MASKS['life_indicator']
        life_w = life_mask.shape[1]
        life_spacing = 4
        lives_y = 20
        lives_count = jnp.clip(state.lives.astype(jnp.int32), 0, 99)
        score_right_edge = base_left + (max_digits - 1) * spacing + digit_w
        total_lives_width = jnp.where(lives_count > 0, (lives_count - 1) * life_spacing + life_w, 0)
        lives_x = score_right_edge - total_lives_width
        raster = self.jr.render_indicator(
            raster, lives_x, lives_y,
            lives_count, life_mask,
            spacing=life_spacing, max_value=99
        )
        return raster