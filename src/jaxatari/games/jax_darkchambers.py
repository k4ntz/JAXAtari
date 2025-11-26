from turtle import width
from typing import NamedTuple, Tuple
from functools import partial
import chex
import jax, jax.numpy as jnp
import time
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer

class DarkChambersConstants(NamedTuple):
    screen_width: int = 160
    screen_height: int = 210 
    tile_size: int = 8  
    map_cols: int = 20
    map_rows: int = 22 

    player_width: int = 6
    player_height: int = 8
    player_speed: float = 1.0
    player_proj_speed: float = 2.0
    player_fire_cooldown: int = 12
    player_invuln_frames: int = 20
    player_hit_inset_x: int = 1
    player_hit_inset_y: int = 1

    enemy_width: int = 6
    enemy_height: int = 8
    enemy_speed: float = 0.75
    max_enemies_on_screen: int = 8
    enemy_spawn_cooldown: int = 60
    enemy_hp_grunt: int = 1
    enemy_hp_spawner: int = 3

# -------- Entity Classes --------

class Entity(NamedTuple):
    x: chex.Array
    y: chex.Array
    w: chex.Array
    h: chex.Array
class PlayerState(NamedTuple):
    x: chex.Array
    y: chex.Array
    hp: chex.Array
    facing: chex.Array               # -1 = left, +1 = right
    fire_cooldown: chex.Array        # frames until next shot allowed

    # player projectile
    proj_x: chex.Array
    proj_y: chex.Array
    proj_active: chex.Array

class EnemyState(NamedTuple):
    positions: chex.Array          # shape (MAX_ENEMIES, 2)
    hp: chex.Array                 # shape (MAX_ENEMIES,)
    active: chex.Array             # shape (MAX_ENEMIES,)
    enemy_type: chex.Array         # shape (MAX_ENEMIES,) 

class SpawnerState(NamedTuple):
    positions: chex.Array          # shape (MAX_SPAWNERS, 2)
    hp: chex.Array                 # shape (MAX_SPAWNERS,)
    active: chex.Array             # shape (MAX_SPAWNERS,)
    cooldown: chex.Array           # shape (MAX_SPAWNERS,)

class LevelState(NamedTuple):
    tile_map: chex.Array           # shape (map_rows, map_cols)
    room_index: chex.Array         # which dungeon room
    step_counter: chex.Array
    score: chex.Array

class DarkChambersState(NamedTuple):
    player: PlayerState
    enemies: EnemyState
    spawners: SpawnerState
    level: LevelState
    rng_key: chex.PRNGKey

class DarkChambersObservation(NamedTuple):
    player: Entity
    enemies: jnp.ndarray            # shape (MAX_ENEMIES, 7)

    # Spawner rows are [x, y, w, h, hp, active, cooldown]
    spawners: jnp.ndarray           # shape (MAX_SPAWNERS, 7)

    # Projectile (simple: one projectile)
    projectile: Entity

    score: chex.Array
    room_index: chex.Array

class DarkChambersInfo(NamedTuple):
    room_index: chex.Array          # current dungeon room
    score: chex.Array               # total score
    step_counter: chex.Array        # global frame counter
    enemies_alive: chex.Array       # how many enemies are active
    spawners_alive: chex.Array      # how many spawners are active


class JaxDarkChamber(JaxEnvironment[DarkChambersConstants, DarkChambersState, DarkChambersObservation, DarkChambersInfo]):
    def __init__(self, consts: DarkChambersConstants = None):
        consts = consts or DarkChambersConstants()
        super().__init__(consts)
        self.consts = consts
        self.action_set = [
            Action.NOOP,
            Action.FIRE,

            Action.UP,
            Action.DOWN,
            Action.LEFT,
            Action.RIGHT,

            Action.UPLEFT,
            Action.UPRIGHT,
            Action.DOWNLEFT,
            Action.DOWNRIGHT,

            Action.UPFIRE,
            Action.DOWNFIRE,
            Action.LEFTFIRE,
            Action.RIGHTFIRE,

            Action.UPLEFTFIRE,
            Action.UPRIGHTFIRE,
            Action.DOWNLEFTFIRE,
            Action.DOWNRIGHTFIRE,
        ]
        self.obs_size = None
        self.renderer = None


    #Game Logic Function

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self,
        key: jax.random.PRNGKey = jax.random.PRNGKey(time.time_ns() % (2**32)),
    ) -> Tuple[DarkChambersObservation, DarkChambersState]:
        key_player, key_enemy, key_spawner, key_map = jax.random.split(key, 4)
        # --- 1. PLAYER INITIALIZATION ---
        player_state = PlayerState(
            x=jnp.array(80, dtype=jnp.int32),      # center horizontally
            y=jnp.array(100, dtype=jnp.int32),     # middle of screen
            hp=jnp.array(3, dtype=jnp.int32),
            facing=jnp.array(1, dtype=jnp.int32),  # face right
            fire_cooldown=jnp.array(0, dtype=jnp.int32),

            proj_x=jnp.array(-1, dtype=jnp.int32),
            proj_y=jnp.array(-1, dtype=jnp.int32),
            proj_active=jnp.array(0, dtype=jnp.int32),
        )
        # --- 2. ENEMY INITIALIZATION ---
        max_e = self.consts.max_enemies_on_screen

        enemy_positions = jnp.zeros((max_e, 2), dtype=jnp.int32)
        enemy_hp = jnp.zeros((max_e,), dtype=jnp.int32)
        enemy_active = jnp.zeros((max_e,), dtype=jnp.int32)
        enemy_type = jnp.zeros((max_e,), dtype=jnp.int32)

        enemies = EnemyState(
            positions=enemy_positions,
            hp=enemy_hp,
            active=enemy_active,
            enemy_type=enemy_type,
        )
        # --- 3. SPAWNER INITIALIZATION ---
        max_s = 4  
        spawner_positions = jnp.zeros((max_s, 2), dtype=jnp.int32)
        spawner_hp = jnp.zeros((max_s,), dtype=jnp.int32)
        spawner_active = jnp.zeros((max_s,), dtype=jnp.int32)
        spawner_cooldown = jnp.zeros((max_s,), dtype=jnp.int32)

        spawners = SpawnerState(
            positions=spawner_positions,
            hp=spawner_hp,
            active=spawner_active,
            cooldown=spawner_cooldown,
        )
        # --- 4. LEVEL INITIALIZATION ---

        tile_map = jnp.zeros(
            (self.consts.map_rows, self.consts.map_cols), dtype=jnp.int32
        )  

        level_state = LevelState(
            tile_map=tile_map,
            room_index=jnp.array(0, dtype=jnp.int32),
            step_counter=jnp.array(0, dtype=jnp.int32),
            score=jnp.array(0, dtype=jnp.int32),
        )
        # --- 5. FULL GAME STATE ---
        #

        state = DarkChambersState(
            player=player_state,
            enemies=enemies,
            spawners=spawners,
            level=level_state,
            rng_key=key,
        )
        # --- 6. INITIAL OBSERVATION ---
        #

        obs = self._get_observation(state)

        return obs, state



